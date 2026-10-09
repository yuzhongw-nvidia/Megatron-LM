# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tests for the experimental latent-KV context-parallel MLA (``MLAWithLatentCP``)."""

import os
from unittest import mock

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.context_parallel_layout import prebuild_thd_cp_partition_routes
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TENorm,
)
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_layer_specs,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import ColumnParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import multi_latent_attention as mla_module
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.experimental_attention_variant import mla_latent_cp
from megatron.core.transformer.experimental_attention_variant.mla_latent_cp import (
    MLAWithLatentCP,
    apply_mla_latent_cp_spec,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.multi_latent_attention import MLASelfAttention
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.test_multi_latent_attention import (
    _gather_full_thd,
    _thd_sp_shard_token_indices,
)

HIDDEN = 64
HEADS = 8
QK_HEAD_DIM, POS_DIM, V_HEAD_DIM = 128, 64, 128
CU_SEQLENS_PADDED = [0, 64, 192, 256, 512]
CU_SEQLENS_ACTUAL = [0, 48, 160, 250, 480]
TOTAL_TOKENS = CU_SEQLENS_PADDED[-1]
_IS_BLACKWELL = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10
try:
    mla_latent_cp._fa4()
    HAVE_FA4_B11 = True
except Exception:  # pylint: disable=broad-except
    HAVE_FA4_B11 = False

requires_te = pytest.mark.skipif(
    not is_te_min_version("2.5.0", check_equality=True),
    reason="Requires TransformerEngine >= 2.5.0",
)
# The reference runs the production configuration: TE's cuDNN backend (NVTE_* are read at call
# time and the conftest disables both backends) on the native (192, 128) MLA shape. The old
# path pads V to the Q head dim for THD unless experimental_attention_variant is set (Kimi uses
# 'kda'); cuDNN has no (192, 192) training kernel on Blackwell, so skip that pad here. TE's
# ring is asked to batch its point-to-point sends/receives (as MLAWithLatentCP does): the
# per-peer NCCL communicators of unbatched isend/irecv can stall the test session's
# destroy_process_group on some clusters.
force_cudnn = mock.patch.dict(
    os.environ, {"NVTE_FUSED_ATTN": "1", "NVTE_FLASH_ATTN": "0", "NVTE_BATCH_MHA_P2P_COMM": "1"}
)
no_v_pad = mock.patch.object(
    mla_module,
    "_prepare_mla_core_attention_value",
    lambda attn, query, value, psp: (value, False, value.shape[-1], value.shape[-1]),
)


def _make_config(tp, cp, **overrides):
    kwargs = dict(
        num_layers=2,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_head_dim=QK_HEAD_DIM,
        qk_pos_emb_head_dim=POS_DIM,
        v_head_dim=V_HEAD_DIM,
        qk_layernorm=True,
        attention_output_gate=True,
        no_rope_freq=1,
        tensor_model_parallel_size=tp,
        sequence_parallel=tp > 1,
        context_parallel_size=cp,
        bf16=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attention_backend=AttnBackend.fused,
        mla_latent_cp=True,
    )
    kwargs.update(overrides)
    return MLATransformerConfig(**kwargs)


def _attention_spec(kv_norm=False):
    spec = get_gpt_layer_with_transformer_engine_spec(
        multi_latent_attention=True, qk_layernorm=True
    ).submodules.self_attention
    if kv_norm:  # a separate KV norm module instead of the one fused into the up-projection
        spec.submodules.kv_layernorm = TENorm
        spec.submodules.linear_kv_up_proj = TEColumnParallelLinear
    return spec


def _build(spec, config):
    model_parallel_cuda_manual_seed(123)
    return build_module(spec, config=config, layer_number=1).bfloat16().cuda()


def _packed(padded, cp_partition_mode="zigzag", **extra):
    cu = torch.tensor(CU_SEQLENS_ACTUAL if padded else CU_SEQLENS_PADDED, dtype=torch.int32).cuda()
    cup = torch.tensor(CU_SEQLENS_PADDED, dtype=torch.int32).cuda() if padded else None
    max_len = max(b - a for a, b in zip(CU_SEQLENS_PADDED[:-1], CU_SEQLENS_PADDED[1:]))
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cup,
        cu_seqlens_kv_padded=cup,
        max_seqlen_q=max_len,
        max_seqlen_kv=max_len,
        cp_partition_mode=cp_partition_mode,
        **extra,
    )


def _assert_close(actual, ref, rtol, mre, what):
    """Whole-tensor max bound plus a mean-relative bound (bf16 noise on near-zero elements)."""
    actual, ref = actual.float(), ref.float()
    diff = (actual - ref).abs()
    max_ref, mean_ref = ref.abs().max().item(), ref.abs().mean().item()
    print(
        f"[parity] {what}: max|d|/max|ref|={diff.max().item() / max(max_ref, 1e-12):.2e} "
        f"mean|d|/mean|ref|={diff.mean().item() / max(mean_ref, 1e-12):.2e}"
    )
    assert (
        diff.max().item() <= rtol * max_ref + 1e-5
    ), f"{what}: max|d|={diff.max().item():.3e} > {rtol}*max|ref|={rtol * max_ref:.3e}"
    assert (
        diff.mean().item() <= mre * mean_ref + 1e-6
    ), f"{what}: mean|d|={diff.mean().item():.3e} > {mre}*mean|ref|={mre * mean_ref:.3e}"


def _run_parity(tp, cp, padded, dynamic=False, contiguous=False, kv_norm=False, fp8=False):
    """Old TE-CP MLA vs MLAWithLatentCP on identical weights, inputs and upstream gradient."""
    world = tp * cp
    if Utils.world_size < world or Utils.world_size % world != 0:
        pytest.skip(f"Needs a multiple of {world} CUDA ranks.")
    Utils.initialize_model_parallel(
        tp, 1, context_parallel_size=cp, dynamic_context_parallel=dynamic
    )
    try:
        cp_group = parallel_state.get_context_parallel_group()
        tp_group = parallel_state.get_tensor_model_parallel_group()
        tp_cp_group = parallel_state.get_tensor_and_context_parallel_group()
        eff_group, eff_cp, extra = cp_group, cp, {}
        if dynamic:
            # A microbatch scheduled on a dynamic CP group of 2 inside the static CP group.
            eff_group = parallel_state.get_dynamic_data_context_parallel_groups(group_size=2)
            eff_cp = 2
            extra = dict(local_cp_size=2, cp_group=eff_group)
        device = torch.device("cuda", torch.cuda.current_device())
        cp_rank, tp_rank = eff_group.rank(), tp_group.rank()
        tokens = TOTAL_TOKENS // eff_cp

        spec = _attention_spec(kv_norm)
        precision = dict(fp8="e4m3", fp8_recipe="mxfp8") if fp8 else {}
        ref = _build(spec, _make_config(tp, cp, mla_latent_cp=False, **precision))
        new_config = _make_config(tp, cp, **precision)
        if contiguous:
            new_config.cp_partition_mode = "contiguous"
        new = _build(apply_mla_latent_cp_spec(spec), new_config)
        assert isinstance(new, MLAWithLatentCP) and isinstance(new.core_attention, IdentityOp)
        new.load_state_dict(ref.state_dict(), strict=False)

        torch.manual_seed(7)
        full_hidden = torch.randn(TOTAL_TOKENS, 1, HIDDEN, dtype=torch.bfloat16, device=device)
        full_grad = torch.randn_like(full_hidden)
        zigzag = _thd_sp_shard_token_indices(
            CU_SEQLENS_PADDED, eff_cp, cp_rank, tp, tp_rank, "zigzag"
        ).to(device)
        ref_in = full_hidden.index_select(0, zigzag).requires_grad_(True)
        with force_cudnn, no_v_pad, get_fp8_context(new_config):
            try:
                ref_out, _ = ref(ref_in, None, packed_seq_params=_packed(padded, **extra))
            except ValueError as error:
                if "No dot product attention backend" not in str(error):
                    raise
                pytest.skip("TE has no backend for padded THD MLA on this build (reference path)")
            ref_out.mul(full_grad.index_select(0, zigzag)).sum().backward()

        if contiguous:
            new_idx = _thd_sp_shard_token_indices(
                CU_SEQLENS_PADDED, eff_cp, cp_rank, tp, tp_rank, "contiguous"
            ).to(device)
            new_psp = _packed(padded, "contiguous", **extra)
            prebuild_thd_cp_partition_routes(
                new_psp, eff_group, tp_group=tp_group, tp_cp_group=tp_cp_group
            )
        else:
            new_idx, new_psp = zigzag, _packed(padded, **extra)
        new_in = full_hidden.index_select(0, new_idx).requires_grad_(True)
        saved_shapes, hook_fires = [], []
        new.linear_kv_up_proj.weight.register_post_accumulate_grad_hook(
            lambda p: hook_fires.append(1)
        )

        def pack(t):
            saved_shapes.append(tuple(t.shape))
            return t

        save_hooks = torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t)
        with save_hooks, get_fp8_context(new_config):
            new_out, _ = new(new_in, None, packed_seq_params=new_psp)
        new_out.mul(full_grad.index_select(0, new_idx)).sum().backward()
        if kv_norm:  # the separate norm is differentiated inside the ring (not fused into TE)
            assert new.kv_layernorm.weight.grad is not None

        # The memory property: only the query (and the output) are saved, never K/V.
        heads = HEADS // tp
        assert saved_shapes.count((tokens, heads, QK_HEAD_DIM + POS_DIM)) <= 1, saved_shapes
        assert saved_shapes.count((tokens, heads, V_HEAD_DIM)) <= 1, saved_shapes
        # The up-projection weight accumulates its gradient exactly once per microbatch.
        assert len(hook_fires) == 1

        # Both paths quantize the forward identically, so the output bound is the bf16 one. In the
        # backward, MXFP8 (E4M3, 3 mantissa bits) quantizes every rank's partial dK/dV before the
        # up-projection in the ring but their CP sum once in TE's ring, so the two paths carry
        # independent rounding noise of the same size: measured 8e-2 max / 5e-2 mean relative on
        # GB300 (tp=2, cp=2) against <= 1e-2 measured in bf16.
        out_tol = (1e-2, 2e-2)
        grad_tol = (1.5e-1, 1e-1) if fp8 else (2e-2, 3e-2)
        full_ref = _gather_full_thd(ref_out, zigzag, TOTAL_TOKENS, tp_cp_group)
        full_new = _gather_full_thd(new_out, new_idx, TOTAL_TOKENS, tp_cp_group)
        _assert_close(full_new, full_ref, *out_tol, "output")
        full_ref_grad = _gather_full_thd(ref_in.grad, zigzag, TOTAL_TOKENS, tp_cp_group)
        full_new_grad = _gather_full_thd(new_in.grad, new_idx, TOTAL_TOKENS, tp_cp_group)
        _assert_close(full_new_grad, full_ref_grad, *grad_tol, "input grad")
        ref_params = dict(ref.named_parameters())
        for name, param in new.named_parameters():
            if param.grad is None:
                assert ref_params[name].grad is None, name
                continue
            # Per-rank partial weight gradients differ (the up-projection runs on every owner's
            # latent, and its norm sees the whole local sequence with this TP rank's partial
            # gradient instead of the sequence shard with the full one); they agree after the
            # reductions DDP performs: CP for every parameter, TP for the sequence-parallel
            # (replicated) norm weights.
            new_grad, ref_grad = param.grad.float().clone(), ref_params[name].grad.float().clone()
            groups = [cp_group] + ([tp_group] if getattr(param, "sequence_parallel", False) else [])
            for group in groups:
                dist.all_reduce(new_grad, group=group)
                dist.all_reduce(ref_grad, group=group)
            _assert_close(new_grad, ref_grad, *grad_tol, name)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.experimental
@requires_te
class TestMLALatentCPParity:
    """Old vs new path with the cuDNN backend (the CI backend)."""

    @pytest.mark.parametrize(("tp", "cp"), ((1, 2), (2, 2), (1, 4), (2, 4), (4, 2)))
    @pytest.mark.parametrize("padded", (False, True), ids=("packed", "padded"))
    def test_parity(self, tp, cp, padded):
        _run_parity(tp, cp, padded)

    def test_parity_cp1(self):
        _run_parity(2, 1, False)

    def test_parity_dynamic_cp(self):
        _run_parity(1, 4, False, dynamic=True)

    def test_parity_contiguous_input(self):
        _run_parity(2, 2, False, contiguous=True)

    def test_parity_separate_kv_norm(self):
        _run_parity(1, 2, False, kv_norm=True)

    @pytest.mark.skipif(not _IS_BLACKWELL, reason="MXFP8 needs Blackwell")
    def test_parity_mxfp8(self):
        _run_parity(2, 2, False, fp8=True)


@pytest.mark.experimental
@pytest.mark.skipif(not (_IS_BLACKWELL and HAVE_FA4_B11), reason="FlashAttention-4 needs Blackwell")
class TestMLALatentCPFlashAttn4:
    """FA4 backend against the cuDNN backend of the same module."""

    @pytest.mark.parametrize(("tp", "cp"), ((1, 2), (1, 4)))
    def test_flash_matches_fused(self, tp, cp):
        world = tp * cp
        if Utils.world_size < world or Utils.world_size % world != 0:
            pytest.skip(f"Needs a multiple of {world} CUDA ranks.")
        Utils.initialize_model_parallel(tp, 1, context_parallel_size=cp)
        try:
            cp_rank = parallel_state.get_context_parallel_group().rank()
            tp_rank = parallel_state.get_tensor_model_parallel_group().rank()
            spec = apply_mla_latent_cp_spec(_attention_spec())
            fused = _build(spec, _make_config(tp, cp))
            flash = _build(spec, _make_config(tp, cp, attention_backend=AttnBackend.flash))
            flash.load_state_dict(fused.state_dict())
            torch.manual_seed(7)
            full_hidden = torch.randn(TOTAL_TOKENS, 1, HIDDEN, dtype=torch.bfloat16).cuda()
            idx = _thd_sp_shard_token_indices(
                CU_SEQLENS_PADDED, cp, cp_rank, tp, tp_rank, "zigzag"
            ).cuda()
            outs, grads = [], []
            for module in (fused, flash):
                x = full_hidden.index_select(0, idx).requires_grad_(True)
                out, _ = module(x, None, packed_seq_params=_packed(True))
                out.sum().backward()
                outs.append(out)
                grads.append(x.grad)
            _assert_close(outs[1], outs[0], 1e-2, 2e-2, "output")
            _assert_close(grads[1], grads[0], 1e-2, 2e-2, "input grad")
            for (name, p_flash), p_fused in zip(flash.named_parameters(), fused.parameters()):
                if p_fused.grad is not None:
                    _assert_close(p_flash.grad, p_fused.grad, 2e-2, 3e-2, name)
        finally:
            Utils.destroy_model_parallel()


def _backends():
    yield pytest.param((mla_latent_cp._cudnn_fwd, mla_latent_cp._cudnn_bwd), id="cudnn")
    if _IS_BLACKWELL and HAVE_FA4_B11:
        yield pytest.param((mla_latent_cp._fa4_fwd, mla_latent_cp._fa4_bwd), id="fa4")


@pytest.mark.experimental
@requires_te
@pytest.mark.parametrize("backend", list(_backends()))
def test_backend_half_kv_zero_copy(backend):
    """A ring step that attends only the first chunk of a received latent passes the full padded
    offsets with half lengths; the kernel must read the first half and leave the rest zero."""
    fwd, bwd = backend
    torch.manual_seed(0)
    n, h = 128, 2
    q = torch.randn(n, h, QK_HEAD_DIM + POS_DIM, dtype=torch.bfloat16, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn(n, h, V_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    off = torch.tensor([0, n], dtype=torch.int32, device="cuda")
    cu_half = torch.tensor([0, n // 2], dtype=torch.int32, device="cuda")
    scale = 0.1
    step = mla_latent_cp._Step(False, off, cu_half, off, off, n, n // 2, False)
    out, lse, rng = fwd(q, k, v, step, scale)
    dout = torch.randn_like(out)
    dq, dk, dv = bwd(q, k, v, out, dout, lse, rng, step, scale, False)

    qf, kf, vf = (
        t.float().transpose(0, 1).contiguous().requires_grad_(True) for t in (q, k, v)
    )  # [h, n, d]
    scores = scale * qf @ kf[:, : n // 2].transpose(1, 2)
    probs = scores.softmax(-1)
    ref_out = probs @ vf[:, : n // 2]
    ref_lse = torch.logsumexp(scores, -1)
    ref_out.backward(dout.float().transpose(0, 1))
    _assert_close(out.transpose(0, 1), ref_out, 2e-2, 2e-2, "out")
    _assert_close(lse.transpose(0, 1), ref_lse, 1e-2, 1e-2, "lse")
    _assert_close(dq.transpose(0, 1), qf.grad, 2e-2, 3e-2, "dq")
    _assert_close(dk.transpose(0, 1), kf.grad, 2e-2, 3e-2, "dk")
    _assert_close(dv.transpose(0, 1), vf.grad, 2e-2, 3e-2, "dv")
    assert torch.equal(dk[n // 2 :], torch.zeros_like(dk[n // 2 :]))
    assert torch.equal(dv[n // 2 :], torch.zeros_like(dv[n // 2 :]))


@pytest.mark.parametrize(
    "overrides",
    (
        dict(multi_latent_attention=False),
        dict(cp_comm_type="a2a+p2p"),
        dict(attention_backend=AttnBackend.unfused),
        dict(tensor_model_parallel_size=2, sequence_parallel=False),
        dict(attention_dropout=0.1),
        dict(recompute_granularity="selective", recompute_modules=["core_attn"]),
        dict(delay_wgrad_compute=True),
        dict(tp_comm_overlap=True),
        dict(fp8="e4m3", fp8_recipe="delayed"),
        dict(mla_down_proj_fusion=True),
    ),
)
def test_config_rejections(overrides):
    with pytest.raises(ValueError, match="mla_latent_cp does not support"):
        _make_config(1, 1, **overrides)


@requires_te
def test_module_rejections():
    Utils.initialize_model_parallel(1, 1)
    try:
        spec = apply_mla_latent_cp_spec(_attention_spec())
        with pytest.raises(NotImplementedError, match="without RoPE"):
            _build(spec, _make_config(1, 1, no_rope_freq=None))
        with pytest.raises(ValueError, match="mla_latent_cp=True"):
            _build(spec, _make_config(1, 1, mla_latent_cp=False))
        local = _attention_spec(kv_norm=True)
        local.submodules.linear_kv_up_proj = ColumnParallelLinear
        with pytest.raises(NotImplementedError, match="Transformer Engine linear"):
            _build(apply_mla_latent_cp_spec(local), _make_config(1, 1))
        module = _build(spec, _make_config(1, 1))
        x = torch.randn(16, 1, HIDDEN, dtype=torch.bfloat16, device="cuda")
        with pytest.raises(ValueError, match="THD"):
            module(x, None)
    finally:
        Utils.destroy_model_parallel()


def test_spec_helper_is_pure_and_default_unchanged():
    spec = _attention_spec()
    new_spec = apply_mla_latent_cp_spec(spec)
    assert new_spec.module is MLAWithLatentCP
    assert new_spec.submodules.core_attention is IdentityOp
    # The old spec (what every model builds with mla_latent_cp=False) is untouched.
    assert spec.module is MLASelfAttention
    assert spec.submodules.core_attention is TEDotProductAttention


@requires_te
@pytest.mark.parametrize(
    ("flag", "module", "core"),
    ((False, MLASelfAttention, TEDotProductAttention), (True, MLAWithLatentCP, IdentityOp)),
)
def test_gated_spec_swap_gpt_and_hybrid(flag, module, core):
    """The two call sites swap the MLA attention only when mla_latent_cp is set."""
    Utils.initialize_model_parallel(1, 1)
    try:
        # One dense and one MoE layer, so both GPT layer specs are checked.
        config = _make_config(1, 1, mla_latent_cp=flag, num_moe_experts=4, moe_layer_freq=[0, 1])
        layer_specs = get_gpt_decoder_layer_specs(config, use_transformer_engine=True)
        assert len(layer_specs) == 2
        for layer_spec in layer_specs:
            attention = layer_spec.submodules.self_attention
            assert attention.module is module and attention.submodules.core_attention is core
        stack = HybridStack(  # one MLA layer ('+'), as the Kimi hybrid pattern builds it
            config,
            hybrid_stack_spec.submodules,
            layer_type_list=["+"],
            post_layer_norm=False,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        )
        assert type(stack.layers[0].self_attention) is module
        assert type(stack.layers[0].self_attention.core_attention) is core
    finally:
        Utils.destroy_model_parallel()
