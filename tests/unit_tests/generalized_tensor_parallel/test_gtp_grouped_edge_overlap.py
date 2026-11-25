# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the grouped-expert chain edge overlap (GTP_CONFIG.grouped_edge_overlap).

A grouped chain's head has no predecessor to prefetch its forward gather and its tail no
successor to prefetch its backward gather; the head's wgrad reduce-scatter also used to run
synchronously on the compute stream. With MoE recompute the recompute-forward chain's head (the
last MoE block) gathered on demand too. Edge overlap prefetches these gathers from the dense
consumes around the MoE blocks and moves the head's reduce-scatter onto rs_stream.

The model mirrors the production layout: a dense GTP linear, then MoE blocks of a grouped fc1
and a grouped fc2 (one chain per role), then a dense "output" GTP linear. The recompute variant
checkpoints each MoE block like --recompute-modules moe.

Test groups
-----------
TestGroupedEdgeOverlapSchedule    - steady state: no grouped on-demand gather, no grouped sync RS,
                                    with and without MoE recompute
TestGroupedEdgeOverlapNumerics    - input grad and every main_grad match the overlap-off run
"""

import pytest
import torch
import torch.distributed as dist

from megatron.core.tensor_parallel.gtp_api import HAVE_GTP
from megatron.core.tensor_parallel.random import CheckpointWithoutOutput

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.19", allow_module_level=True)

import megatron.core.tensor_parallel.generalized_tensor_parallelism as gtp_module
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (
    _make_gtp_linear,
    _make_gtp_remat_grouped_linear,
    _requires_multi_gpu,
    _run_distributed,
    _torchrun_dist_init,
    reset_fp8_state,
    reset_gtp_globals,
)

HIDDEN = 512
FFN = 256
NUM_BLOCKS = 3
NUM_GEMMS = 2
DTYPE = torch.bfloat16
FC1_CHAIN = "GTP_remat_grouped_fc1_ungraphed"
FC2_CHAIN = "GTP_remat_grouped_fc2_ungraphed"


def _build_model(world_size):
    """(dense_in, [(fc1, fc2)] * NUM_BLOCKS, dense_out), identically initialized on every call."""
    torch.manual_seed(0)
    group = dist.new_group(list(range(world_size)))
    dense_in = _make_gtp_linear(HIDDEN, HIDDEN, group, DTYPE, fuse_wgrad_accumulation=True)
    blocks = []
    for _ in range(NUM_BLOCKS):
        fc1 = _make_gtp_remat_grouped_linear(
            NUM_GEMMS, HIDDEN, FFN, group, DTYPE, fuse_wgrad_accumulation=True
        )
        fc2 = _make_gtp_remat_grouped_linear(
            NUM_GEMMS, FFN, HIDDEN, group, DTYPE, fuse_wgrad_accumulation=True
        )
        # Production assigns these in _classify_param_chain from the param name.
        for module, chain in ((fc1, FC1_CHAIN), (fc2, FC2_CHAIN)):
            for w in module.weight0.weight_list:
                w.chain_id = chain
        blocks.append((fc1, fc2))
    dense_out = _make_gtp_linear(HIDDEN, HIDDEN, group, DTYPE, fuse_wgrad_accumulation=True)
    for p in _gtp_params(dense_in, blocks, dense_out):
        p.main_grad = torch.zeros(p.shape, dtype=DTYPE, device="cuda")
    return dense_in, blocks, dense_out


def _gtp_params(dense_in, blocks, dense_out):
    params = [dense_in.weight, dense_out.weight]
    for fc1, fc2 in blocks:
        params += list(fc1.weight0.weight_list) + list(fc2.weight0.weight_list)
    return params


def _forward_backward(model, x, recompute=False):
    dense_in, blocks, dense_out = model
    m_splits = [x.shape[0] // NUM_GEMMS] * NUM_GEMMS

    def moe(inp, fc1, fc2):
        y = torch.nn.functional.gelu(fc1(inp, m_splits=m_splits, is_first_microbatch=True))
        return fc2(y, m_splits=m_splits, is_first_microbatch=True)

    h = torch.nn.functional.gelu(dense_in(x))
    for fc1, fc2 in blocks:
        if recompute:
            # gelu saves the block output, so discarding it forces the recompute in backward.
            checkpoint = CheckpointWithoutOutput()
            out = checkpoint.checkpoint(lambda inp, a=fc1, b=fc2: moe(inp, a, b), h)
            out = torch.nn.functional.gelu(out)
            checkpoint.discard_output_and_register_recompute(out)
        else:
            out = moe(h, fc1, fc2)
        h = h + out
    dense_out(h).float().sum().backward()
    # The end-of-backward fence production runs in finalize_model_grads.
    gtp_module.wait_for_gtp_grad_reduction_on_current_stream()


class _Recorder:
    """Record grouped-chain on-demand gathers and reduce-scatter modes during one step."""

    def __init__(self, monkeypatch):
        self.on_demand = []
        self.rs = []
        cls = gtp_module.GTPShardedParam
        orig_on_demand = cls._all_gather_weight_on_demand
        orig_rs = cls._reduce_scatter
        rec = self

        def on_demand(param, fwd, *args, **kwargs):
            if gtp_module._chain_is_grouped(param.chain_id):
                kind = "recompute" if kwargs.get("recompute") else ("fwd" if fwd else "bwd")
                rec.on_demand.append((param.chain_id, kind))
            return orig_on_demand(param, fwd, *args, **kwargs)

        def reduce_scatter(param, wgrads, async_op, *args, **kwargs):
            if gtp_module._chain_is_grouped(param.chain_id):
                rec.rs.append((param.chain_id, async_op))
            return orig_rs(param, wgrads, async_op, *args, **kwargs)

        monkeypatch.setattr(cls, "_all_gather_weight_on_demand", on_demand)
        monkeypatch.setattr(cls, "_reduce_scatter", reduce_scatter)


def _worker_schedule(rank, world_size, port, monkeypatch, edge_overlap, recompute):
    gtp_module.update_gtp_config(grouped_edge_overlap=edge_overlap)
    try:
        gtp_module.reset_gtp_state()
        model = _build_model(world_size)
        x = torch.randn(8 * NUM_GEMMS, HIDDEN, dtype=DTYPE, device="cuda", requires_grad=True)
        dist.broadcast(x, src=0)
        # Step 1 builds the chains (every gather on demand); step 2 arms the heads; step 3 is
        # the first that starts with armed heads, i.e. steady state.
        for _ in range(2):
            _forward_backward(model, x, recompute)
        rec = _Recorder(monkeypatch)
        _forward_backward(model, x, recompute)
    finally:
        gtp_module.update_gtp_config(grouped_edge_overlap=False)  # the core default

    sync_rs = [chain for chain, async_op in rec.rs if not async_op]
    if edge_overlap:
        assert not rec.on_demand, f"grouped chain edges still gather on demand: {rec.on_demand}"
        assert not sync_rs, f"grouped chain head still reduce-scatters synchronously: {sync_rs}"
    else:
        # Guards the test itself: without the feature both edges of both chains are exposed,
        # and with recompute so is the head of each recompute chain.
        kinds = ["fwd", "bwd"] + (["recompute"] if recompute else [])
        expected = [(chain, kind) for chain in (FC1_CHAIN, FC2_CHAIN) for kind in kinds]
        assert sorted(rec.on_demand) == sorted(expected), rec.on_demand
        assert sorted(sync_rs) == [FC1_CHAIN, FC2_CHAIN], sync_rs


def _worker_numerics(rank, world_size, port, recompute):
    """Edge overlap only moves work between streams; the gradients must not change."""
    results = {}
    for edge_overlap in (False, True):
        gtp_module.update_gtp_config(grouped_edge_overlap=edge_overlap)
        try:
            gtp_module.reset_gtp_state()
            model = _build_model(world_size)
            torch.manual_seed(1)
            x = torch.randn(8 * NUM_GEMMS, HIDDEN, dtype=DTYPE, device="cuda", requires_grad=True)
            dist.broadcast(x, src=0)
            for _ in range(3):
                for p in _gtp_params(*model):
                    p.main_grad.zero_()
                x.grad = None
                _forward_backward(model, x, recompute)
            torch.cuda.synchronize()
            results[edge_overlap] = (
                x.grad.clone(),
                [p.main_grad.clone() for p in _gtp_params(*model)],
            )
        finally:
            gtp_module.update_gtp_config(grouped_edge_overlap=False)  # the core default

    (dgrad_off, grads_off), (dgrad_on, grads_on) = results[False], results[True]
    # Same kernels on the same inputs, only on different streams: expect bitwise equality.
    torch.testing.assert_close(dgrad_on, dgrad_off, rtol=0, atol=0)
    for i, (g_on, g_off) in enumerate(zip(grads_on, grads_off)):
        torch.testing.assert_close(g_on, g_off, rtol=0, atol=0, msg=f"main_grad {i} differs")


class TestGroupedEdgeOverlapSchedule:
    @pytest.mark.parametrize("recompute", [False, True])
    @pytest.mark.parametrize("edge_overlap", [True, False])
    def test_edges_are_prefetched(self, monkeypatch, edge_overlap, recompute):
        _requires_multi_gpu(4)
        _run_distributed(_worker_schedule, 4, monkeypatch, edge_overlap, recompute)


class TestGroupedEdgeOverlapNumerics:
    @pytest.mark.parametrize("recompute", [False, True])
    def test_grads_match_overlap_off(self, recompute):
        _requires_multi_gpu(4)
        _run_distributed(_worker_numerics, 4, recompute)
