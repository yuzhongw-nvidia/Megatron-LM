# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""End-to-end parity for layer-sharded Muon through the real training-loop path.

Runs ``setup_model_and_optimizer`` on a TP(2) x GTP_remat(2) grid and checks that the
loss trajectory of ``--muon-tp-mode layer_sharded`` (the two-stage exchange over a
genuine GTP_remat x TP domain) is bitwise identical to ``duplicated`` mode.

Skips at module level when GTP is unavailable (TransformerEngine < 2.19.0.dev0), like
``test_gtp_fp8_param_gather.py``.

Launch with 4 ranks:
  torchrun --nproc-per-node=4 -m pytest tests/unit_tests/optimizer/test_layer_sharded_e2e_parity.py
"""

import pytest
import torch

pytest.importorskip("emerging_optimizers", reason="LayerShardedMuon requires emerging-optimizers")

from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.19", allow_module_level=True)

from megatron.core.optimizer.layer_sharded_muon import LayerShardedMuon
from tests.unit_tests.test_fp8_param import TestFP8Param as _Harness
from tests.unit_tests.test_utilities import Utils

# Plain bf16 run: the harness defaults to FP8, which is orthogonal here. ``fp8=None`` turns
# it off; the harness's positional ``recipe`` must then not be "mxfp8", because validate_args
# ties GTP + mxfp8 to --fp8-param-gather.
_RECIPE = "tensorwise"
_COMMON = dict(
    fp8=None,
    num_steps=8,
    num_layers=2,
    padded_vocab_size=512,
    hidden_size=256,
    num_attention_heads=8,
    ffn_hidden_size=512,
    global_batch_size=2,
    optimizer="muon",
    muon_scalar_optimizer="adam",
    muon_momentum=0.9,
    muon_scale_mode="spectral",
    muon_num_ns_steps=5,
    muon_coefficient_type="quintic",
    muon_split_qkv=False,
    lr=1e-3,
    clip_grad=0.0,
    hidden_dropout=0.0,
    attention_dropout=0.0,
    # 4 shards per weight over TP 2 => GTP_remat 2: a genuinely 2-D (GTP_remat x TP) domain.
    tensor_parallel_num_weight_shards=4,
    untie_embeddings_and_output_weights=True,
    overlap_param_gather=False,
    overlap_grad_reduce=False,
)

_MOE = dict(
    num_experts=2,
    moe_grouped_gemm=True,
    moe_single_grouped_weight=False,
    moe_ffn_hidden_size=512,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=2,
    expert_tensor_parallel_num_weight_shards=4,
    moe_token_dispatcher_type="alltoall",
    moe_router_topk=1,
    moe_router_pre_softmax=True,
    moe_router_load_balancing_type="none",
    moe_aux_loss_coeff=0.0,
    add_bias_linear=False,
)


class _LayerShardedHarness(_Harness):
    """Records, right after construction, which optimizer instances and exchange paths exist."""

    def _on_model_built(self, model_chunks, optimizer, args):
        found = []

        def visit(opt):
            if isinstance(opt, LayerShardedMuon):
                found.append(opt)
                return
            children = getattr(opt, 'chained_optimizers', None)
            if children:
                # ChainedOptimizer (incl. LayerWiseDistributedOptimizer): its ``optimizer``
                # property asserts a single child, so recurse into the children instead.
                for child in children:
                    visit(child)
                return
            # Float16OptimizerWithFloat16Params / FP32Optimizer store the base optimizer as
            # a plain attribute; vars() avoids triggering any property.
            inner = vars(opt).get('optimizer')
            if inner is not None and inner is not opt:
                visit(inner)

        visit(optimizer)
        wired = [pgs for o in found for pgs in o._group_process_groups.values()]
        self.layer_sharded_instances = len(found)
        self.wired_groups = len(wired)
        self.rebalance_per_call = [o._rebalance_homes_per_call for o in found]
        # Chunked optimizer-state offload: the wrapper around each LayerShardedMuon owns the
        # offloader; record how many state chunks it will step per iteration.
        self.offload_chunks = []
        self.all_muon_offload_chunks = {}

        def visit_wrappers(opt):
            children = getattr(opt, 'chained_optimizers', None)
            if children:
                for child in children:
                    visit_wrappers(child)
                return
            inner = vars(opt).get('optimizer')
            offloader = vars(opt).get('_optimizer_state_offloader')
            from megatron.core.optimizer.emerging_optimizers import TensorParallelMuon

            if isinstance(inner, TensorParallelMuon) and offloader is not None:
                self.all_muon_offload_chunks[type(inner).__name__] = len(offloader.chunks)
            if isinstance(inner, LayerShardedMuon) and offloader is not None:
                self.offload_chunks.append(len(offloader.chunks))

        visit_wrappers(optimizer)
        # A domain is 2-D only when both axis groups are non-trivial.
        self.two_d_groups = sum(
            1
            for pgs in wired
            if pgs[0] is not None and pgs[0].size() > 1 and pgs[1] is not None and pgs[1].size() > 1
        )


def _assert_bitwise_across_ranks(loss_ref, loss_test, label):
    local = torch.stack((loss_ref, loss_test)).cuda()
    gathered = [torch.empty_like(local) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gathered, local)
    for rank, pair in enumerate(gathered):
        finite = bool(torch.isfinite(pair).all())
        assert finite, f"{label}: non-finite loss on rank {rank}: {pair.tolist()}"
        max_diff = (pair[0] - pair[1]).abs().max().item()
        assert torch.equal(pair[0], pair[1]), (
            f"{label}: rank {rank} diverges, max |diff|={max_diff:.3e}\n"
            f"  duplicated   : {pair[0].tolist()}\n  layer_sharded: {pair[1].tolist()}"
        )


def _run_pair(extra, label):
    from megatron.core.tensor_parallel import generalized_tensor_parallelism as gtp

    if Utils.world_size != 4:
        pytest.skip("Requires exactly 4 torchrun ranks for TP2 x GTP2")
    harness = _LayerShardedHarness()
    harness.setup_method(None)
    harness.seq_length = 64
    harness.micro_batch_size = 1
    try:
        common = {**_COMMON, **extra}
        loss_dup = harness._run_test_helper(
            tp_size=2, recipe=_RECIPE, fp8_param_gather=False, muon_tp_mode="duplicated", **common
        )
        assert harness.layer_sharded_instances == 0
        assert not gtp._GTP_PENDING_WGRAD_ACCUM, "duplicated run left GTP gradients unflushed"
        loss_lsh = harness._run_test_helper(
            tp_size=2,
            recipe=_RECIPE,
            fp8_param_gather=False,
            muon_tp_mode="layer_sharded",
            **common,
        )
        assert loss_dup.shape == loss_lsh.shape == (common["num_steps"],)
        assert not gtp._GTP_PENDING_WGRAD_ACCUM, "layer-sharded run left GTP gradients unflushed"
        assert harness.layer_sharded_instances >= 1, "no LayerShardedMuon was constructed"
        assert harness.wired_groups > 0, "the layer-wise wiring assigned no NS homes"
        assert harness.two_d_groups == harness.wired_groups, (
            f"only {harness.two_d_groups} of {harness.wired_groups} wired groups span a 2-D "
            "(GTP_remat x TP) domain; the test grid is not what it claims"
        )
        _assert_bitwise_across_ranks(loss_dup, loss_lsh, label)
        if torch.distributed.get_rank() == 0:
            print(
                f"[{label}] {len(loss_dup)} steps bitwise identical; LayerShardedMuon "
                f"instances={harness.layer_sharded_instances}, two-stage groups="
                f"{harness.wired_groups}, losses={loss_lsh.tolist()}"
            )
    finally:
        harness.teardown_method(None)
        from megatron.core.tensor_parallel.generalized_tensor_parallelism import update_gtp_config

        update_gtp_config(pad_for_alignment=16, calculate_per_token_loss=False)


@pytest.mark.launch_on_gb200
def test_dense_layer_sharded_matches_duplicated_end_to_end():
    """Dense GPT, TP2 x GTP2: layer_sharded (two-stage) through the training loop ==
    duplicated, bitwise."""
    _run_pair({}, "dense")


@pytest.mark.launch_on_gb200
def test_moe_layer_sharded_matches_duplicated_end_to_end():
    """GPT-MoE (EP1, ETP2 x EGTP2): layer_sharded (two-stage) == duplicated, bitwise."""
    _run_pair(_MOE, "moe")


# Chunked optimizer-state offload: fp32 masters and the Muon momentum live on the host
# between steps and the optimizer is stepped once per state chunk. A 1 MiB chunk target
# makes even this small model step through several chunks, so the per-chunk subsets of
# the param groups (re-packed positions, per-call home balancing) are exercised.
_OFFLOAD = dict(
    chunked_optimizer_state_offload=True,
    optimizer_state_offload_fraction=1.0,
    optimizer_state_offload_chunk_size_mb=1,
)


def _run_offload_pair(extra, label, tp_size=2):
    """layer_sharded + chunked offload == layer_sharded without offload == duplicated,
    all bitwise: offload only moves tensors and re-packs the groups per chunk, and the
    home placement only decides where Newton-Schulz runs."""
    if Utils.world_size != 4:
        pytest.skip("Requires exactly 4 torchrun ranks")
    harness = _LayerShardedHarness()
    harness.setup_method(None)
    harness.seq_length = 64
    harness.micro_batch_size = 1
    try:
        common = {**_COMMON, **extra}
        loss_dup = harness._run_test_helper(
            tp_size=tp_size,
            recipe=_RECIPE,
            fp8_param_gather=False,
            muon_tp_mode="duplicated",
            **common,
        )
        loss_lsh = harness._run_test_helper(
            tp_size=tp_size,
            recipe=_RECIPE,
            fp8_param_gather=False,
            muon_tp_mode="layer_sharded",
            **common,
        )
        assert harness.offload_chunks == [], "offload was not requested yet"
        assert not any(harness.rebalance_per_call), "per-call balancing without offload"
        loss_lsh_offload = harness._run_test_helper(
            tp_size=tp_size,
            recipe=_RECIPE,
            fp8_param_gather=False,
            muon_tp_mode="layer_sharded",
            **common,
            **_OFFLOAD,
        )
        assert harness.layer_sharded_instances >= 1, "no LayerShardedMuon was constructed"
        assert harness.offload_chunks, "no LayerShardedMuon child got a state offloader"
        assert max(harness.offload_chunks) > 1, (
            f"the offloader planned a single chunk ({harness.offload_chunks}); the per-chunk "
            "subset path was not exercised"
        )
        assert all(harness.rebalance_per_call), "offload must enable per-call home balancing"
        _assert_bitwise_across_ranks(loss_dup, loss_lsh, f"{label}: duplicated vs lsh")
        _assert_bitwise_across_ranks(loss_lsh, loss_lsh_offload, f"{label}: lsh vs lsh+offload")
        if torch.distributed.get_rank() == 0:
            print(
                f"[{label}] {len(loss_dup)} steps bitwise identical across duplicated, "
                f"layer_sharded and layer_sharded+offload ({harness.offload_chunks} state "
                f"chunks per step); losses={loss_lsh_offload.tolist()}"
            )
    finally:
        harness.teardown_method(None)
        from megatron.core.tensor_parallel.generalized_tensor_parallelism import update_gtp_config

        update_gtp_config(pad_for_alignment=16, calculate_per_token_loss=False)


@pytest.mark.launch_on_gb200
def test_dense_layer_sharded_with_chunked_offload_end_to_end():
    """Dense GPT, TP2 x GTP2: layer_sharded stepped per offload chunk == layer_sharded ==
    duplicated, bitwise."""
    _run_offload_pair({}, "dense+offload")


@pytest.mark.launch_on_gb200
def test_moe_layer_sharded_with_chunked_offload_end_to_end():
    """GPT-MoE (EP1, ETP2 x EGTP2): expert and dense chunks resolve their own domains
    through the group marker even when a chunk holds only one of them."""
    _run_offload_pair(_MOE, "moe+offload")


@pytest.mark.launch_on_gb200
def test_moe_kimi_grid_layer_sharded_with_chunked_offload_end_to_end():
    """GPT-MoE on TP1 x GTP4 / ETP1 x EGTP4 (the Kimi-K3 expert layout: GTP is the only
    sharding axis): layer_sharded stepped per offload chunk == layer_sharded ==
    duplicated, bitwise, on a grid where dense and expert groups live in different 1-D
    domains. Per-matrix Newton-Schulz on purpose: batched (3-D) NS groups matrices
    differently per chunk and per home, and baddbmm rounding would break bitwise parity."""
    _run_offload_pair(
        {
            **_MOE,
            "tensor_parallel_num_weight_shards": 4,
            "expert_tensor_parallel_size": 1,
            "expert_tensor_parallel_num_weight_shards": 4,
            # TP1 on 4 ranks => DP4 (GTP_remat shards over the DP group): one micro-batch
            # per DP rank.
            "global_batch_size": 4,
        },
        "moe-kimi-grid+offload",
        tp_size=1,
    )


class _DuplicatedHarness(_Harness):
    """Records the TensorParallelMuon instances so the batched-step counters can be read."""

    def _on_model_built(self, model_chunks, optimizer, args):
        from megatron.core.optimizer.emerging_optimizers import TensorParallelMuon

        found = []

        def visit(opt):
            if isinstance(opt, TensorParallelMuon):
                found.append(opt)
                return
            children = getattr(opt, 'chained_optimizers', None)
            if children:
                for child in children:
                    visit(child)
                return
            inner = vars(opt).get('optimizer')
            if inner is not None and inner is not opt:
                visit(inner)

        visit(optimizer)
        self.muon_optimizers = found


@pytest.mark.launch_on_gb200
def test_moe_duplicated_batched_ns_matches_per_param_end_to_end():
    """GPT-MoE on a TP1 x GTP4 / ETP1 x EGTP4 grid (the Kimi-K3 expert layout: GTP is the
    only sharding axis, so the gathered matrix needs no TP collective), duplicated mode:
    --muon-ns-batch-size 4 (same-shape GTP-sharded weights stacked into one batched
    Newton-Schulz, one all-gather per batch) tracks the per-parameter step
    (--muon-ns-batch-size 1) to baddbmm-vs-addmm rounding, and the batched path actually
    ran."""
    if Utils.world_size != 4:
        pytest.skip("Requires exactly 4 torchrun ranks for TP1 x GTP4")
    harness = _DuplicatedHarness()
    harness.setup_method(None)
    harness.seq_length = 64
    harness.micro_batch_size = 1
    try:
        common = {
            **_COMMON,
            **_MOE,
            "tensor_parallel_num_weight_shards": 4,
            "expert_tensor_parallel_size": 1,
            "expert_tensor_parallel_num_weight_shards": 4,
            # TP1 on 4 ranks => DP4 (GTP_remat shards over the DP group), so the global batch
            # must hold one micro-batch per DP rank.
            "global_batch_size": 4,
        }
        loss_per_param = harness._run_test_helper(
            tp_size=1,
            recipe=_RECIPE,
            fp8_param_gather=False,
            muon_tp_mode="duplicated",
            muon_ns_batch_size=1,
            **common,
        )
        assert all(opt.gtp_batch_counts == (0, 0) for opt in harness.muon_optimizers)
        loss_batched = harness._run_test_helper(
            tp_size=1,
            recipe=_RECIPE,
            fp8_param_gather=False,
            muon_tp_mode="duplicated",
            muon_ns_batch_size=4,
            **common,
        )
        batches, matrices = map(
            sum, zip(*(opt.gtp_batch_counts for opt in harness.muon_optimizers))
        )
        assert batches > 0 and matrices > batches, (
            "the batched duplicated-GTP step never stacked more than one matrix: "
            f"batches={batches}, matrices={matrices}"
        )
        assert loss_per_param.shape == loss_batched.shape == (common["num_steps"],)
        assert bool(torch.isfinite(loss_batched).all())
        torch.testing.assert_close(loss_batched, loss_per_param, rtol=1e-2, atol=1e-2)
        # Gathers one batch ahead on a side stream: same collectives, same math, so this
        # must track the in-order batched run far tighter than the batched-vs-per-param bound.
        loss_ahead = harness._run_test_helper(
            tp_size=1,
            recipe=_RECIPE,
            fp8_param_gather=False,
            muon_tp_mode="duplicated",
            muon_ns_batch_size=4,
            muon_ns_gather_ahead=1,
            **common,
        )
        assert all(opt.ns_gather_ahead == 1 for opt in harness.muon_optimizers)
        assert all(
            opt._gather_stream is not None for opt in harness.muon_optimizers
        ), "the side-stream gather path never ran"
        assert bool(torch.isfinite(loss_ahead).all())
        torch.testing.assert_close(loss_ahead, loss_batched, rtol=1e-4, atol=1e-4)
        if torch.distributed.get_rank() == 0:
            print(
                f"[moe_batched_ns] {batches} batches / {matrices} matrices; "
                f"max |loss diff| = {(loss_batched - loss_per_param).abs().max().item():.3e}; "
                f"gather-ahead vs in-order max |loss diff| = "
                f"{(loss_ahead - loss_batched).abs().max().item():.3e}"
            )
    finally:
        harness.teardown_method(None)
        from megatron.core.tensor_parallel.generalized_tensor_parallelism import update_gtp_config

        update_gtp_config(pad_for_alignment=16, calculate_per_token_loss=False)


@pytest.mark.launch_on_gb200
def test_hybrid_muon_kimi_grid_with_chunked_offload_end_to_end():
    """Dense auto/per-head + expert LSH: offloading both children preserves updates."""
    if Utils.world_size != 4:
        pytest.skip("Requires exactly 4 torchrun ranks")
    harness = _LayerShardedHarness()
    harness.setup_method(None)
    harness.seq_length = 64
    harness.micro_batch_size = 1
    try:
        common = {
            **_COMMON,
            **_MOE,
            'expert_tensor_parallel_size': 1,
            'expert_tensor_parallel_num_weight_shards': 4,
            'global_batch_size': 4,
            'muon_tp_mode': 'auto',
            'muon_expert_tp_mode': 'layer_sharded',
            'muon_split_qkv': True,
            'muon_split_qkv_per_head': True,
        }
        loss_resident = harness._run_test_helper(
            tp_size=1, recipe=_RECIPE, fp8_param_gather=False, **common
        )
        assert harness.layer_sharded_instances == 1
        assert harness.wired_groups > 0
        assert harness.all_muon_offload_chunks == {}
        loss_offload = harness._run_test_helper(
            tp_size=1, recipe=_RECIPE, fp8_param_gather=False, **common, **_OFFLOAD
        )
        assert harness.layer_sharded_instances == 1
        assert set(harness.all_muon_offload_chunks) == {'TensorParallelMuon', 'LayerShardedMuon'}
        assert all(count > 1 for count in harness.all_muon_offload_chunks.values())
        assert all(harness.rebalance_per_call)
        _assert_bitwise_across_ranks(loss_resident, loss_offload, 'hybrid-kimi-grid+offload')
        if torch.distributed.get_rank() == 0:
            print(
                f'[HYBRID OFFLOAD PARITY] {len(loss_resident)} steps bitwise identical; '
                f'chunks={harness.all_muon_offload_chunks}'
            )
    finally:
        harness.teardown_method(None)
        from megatron.core.tensor_parallel.generalized_tensor_parallelism import update_gtp_config

        update_gtp_config(pad_for_alignment=16, calculate_per_token_loss=False)
