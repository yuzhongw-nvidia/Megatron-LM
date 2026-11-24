# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the cuDNN Kimi Delta Attention recurrence backend (``kda_backend="cudnn"``)."""

import os
from dataclasses import replace

import pytest
import torch

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.ssm.gated_delta_net import HAVE_FLA_KDA
from megatron.core.ssm.gated_delta_net import kda as kda_module
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.ssm.gated_delta_net.test_kda import (
    _KDA_INPUT_GRAD_ATOL,
    _build_kda,
    _make_config,
)
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.test_attention import _test_parallel_attention_correctness
from tests.unit_tests.transformer.test_multi_latent_attention import make_test_packed_seq_params

try:
    from megatron.core.ops.ssm.kda.cudnn_backend import cudnn_chunk_kda

    HAVE_CUDNN_KDA = True
except ImportError:
    HAVE_CUDNN_KDA = False

os.environ.update({"NCCL_NVLS_ENABLE": "0"})

# cuDNN's KDA kernels exist for Blackwell GPUs only.
pytestmark = pytest.mark.launch_on_gb200

requires_cudnn_kda = pytest.mark.skipif(
    not (
        HAVE_FLA_KDA
        and HAVE_CUDNN_KDA
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] >= 10
    ),
    reason="Needs FLA, cudnn-frontend with the KDA operators and a Blackwell GPU.",
)

# Bounds on the relative L2 distance to the FLA backend. The gradients that flow through the
# decay gate are sums of mixed-sign bf16 terms and agree less tightly than the others.
_OUTPUT_TOL = 2e-2
_GRAD_TOL = 3e-2
_GATE_GRAD_TOL = 5e-2
_GATE_PARAM_GRAD_TOL = 2e-1


def _make_backend_config(backend: str, **kwargs):
    """``test_kda``'s config with the 128-dim heads that cuDNN's kernels support."""
    overrides = {}
    if kwargs.get("linear_cp_mode") == "chunkwise":
        # replace() would carry the resolved zigzag layout over as an explicit setting.
        overrides["linear_cp_layout"] = "contiguous"
    return replace(
        _make_config(**kwargs),
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        kda_backend=backend,
        **overrides,
    )


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    diff = torch.linalg.vector_norm(actual.float() - expected.float())
    return (diff / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)).item()


def _grad_tol(name: str) -> float:
    if name in ("A_log", "dt_bias"):
        return _GATE_PARAM_GRAD_TOL
    if name.startswith(("f_proj.", "f_a_proj.", "f_b_proj.")):
        return _GATE_GRAD_TOL
    return _GRAD_TOL


def test_legacy_kda_kernel_backend_selects_recurrence():
    config = _make_backend_config("fla")
    assert replace(config, kda_kernel_backend="cudnn").kda_backend == "cudnn"
    with pytest.raises(ValueError, match="Conflicting"):
        replace(config, kda_backend="cudnn", kda_kernel_backend="fla")


@requires_cudnn_kda
@pytest.mark.parametrize(
    ("sequence_packing", "pre_gdr_fusion"),
    [(False, False), (True, False), (True, True)],
    ids=["sbhd", "thd", "thd-pre-gdr-fusion"],
)
def test_kda_cudnn_matches_fla(sequence_packing, pre_gdr_fusion):
    """Output, input gradient and every parameter gradient against the FLA backend."""
    if pre_gdr_fusion and kda_module.fused_streamed_pre_kda is None:
        pytest.skip("The streamed pre-KDA fusion is unavailable.")
    Utils.initialize_model_parallel(1, 1)
    try:
        # Seed before building the modules: their weights come from the CPU generator, and the
        # gate-parameter gradients are only comparable on heads whose gate is not saturated.
        torch.manual_seed(42)
        batch = 1 if sequence_packing else 2
        hidden_states = torch.randn((512, batch, 128), device="cuda", dtype=torch.bfloat16)
        model_parallel_cuda_manual_seed(123)
        config_kwargs = {"f_lora_rank": 8, "gate_lora_rank": 12}
        modules = {
            backend: _build_kda(
                replace(
                    _make_backend_config(backend, **config_kwargs),
                    gdn_pre_gated_delta_rule_fusion=pre_gdr_fusion,
                )
            )
            for backend in ("fla", "cudnn")
        }
        modules["cudnn"].load_state_dict(modules["fla"].state_dict())
        from fla.ops.kda import chunk_kda

        assert modules["fla"].gated_delta_rule is chunk_kda
        assert modules["cudnn"].gated_delta_rule is cudnn_chunk_kda

        results = {}
        for backend, module in modules.items():
            module_input = hidden_states.clone().requires_grad_(True)
            packed_seq_params = (
                make_test_packed_seq_params(cu_seqlens=[0, 96, 245, 512])
                if sequence_packing
                else None
            )
            output, _ = module(module_input, None, packed_seq_params=packed_seq_params)
            grad_output = torch.linspace(-0.1, 0.1, output.numel(), device="cuda")
            (output.float() * grad_output.view_as(output)).sum().backward()
            results[backend] = (output.detach(), module_input.grad)

        assert _relative_l2(results["cudnn"][0], results["fla"][0]) < _OUTPUT_TOL
        assert _relative_l2(results["cudnn"][1], results["fla"][1]) < _GRAD_TOL
        cudnn_params = dict(modules["cudnn"].named_parameters())
        for name, param in modules["fla"].named_parameters():
            if param.grad is not None:
                distance = _relative_l2(cudnn_params[name].grad, param.grad)
                assert distance < _grad_tol(name), f"{name}: relative L2 {distance:.3e}"
    finally:
        Utils.destroy_model_parallel()


@requires_cudnn_kda
@pytest.mark.parametrize("cp", [2, 4])
@pytest.mark.parametrize(
    ("sequence_packing", "micro_batch_size"), [(True, 5), (False, 1)], ids=["thd", "sbhd"]
)
def test_kda_cudnn_chunkwise_cp_matches_single_rank(
    tmp_path_dist_ckpt, sequence_packing, micro_batch_size, cp, run_config
):
    """Chunkwise CP against the single-rank cuDNN model.

    The packed sequences cross rank boundaries, with three of them on each CP2 rank (including
    segments that continue on the next rank). The single SBHD sequence (chunkwise CP requires
    ``micro_batch_size == 1`` there) spans every rank, so that with CP4 the last rank folds
    three predecessors and the first rank three successors.
    """
    if Utils.world_size < cp:
        pytest.skip(f"Needs at least {cp} distributed ranks.")
    _test_parallel_attention_correctness(
        transformer_config=_make_backend_config(
            "cudnn", linear_cp_mode="chunkwise", f_lora_rank=8, gate_lora_rank=12
        ),
        transformer_layer_spec=hybrid_stack_spec.submodules.kda_layer,
        tmp_path_dist_ckpt=tmp_path_dist_ckpt,
        atol=1e-2,
        rtol=1e-2,
        input_grad_atol=_KDA_INPUT_GRAD_ATOL,
        input_grad_rtol=1e-2,
        cosine_similarity_threshold=0.9999,
        compare_param_grads=True,
        tp=1,
        cp=cp,
        seed=42,
        sequence_length=128,
        micro_batch_size=micro_batch_size,
        sequence_packing=sequence_packing,
    )
