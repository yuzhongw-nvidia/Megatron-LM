# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Independent references for KDA's replicated complete-sequence projection."""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.ssm.gated_delta_net import HAVE_FLA_KDA
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.ssm.gated_delta_net.test_kda import _build_kda, _make_config
from tests.unit_tests.test_utilities import Utils

pytestmark = [
    pytest.mark.internal,
    pytest.mark.skipif(not HAVE_FLA_KDA, reason="FLA with KDA support is required."),
]


@pytest.fixture(autouse=True)
def full_precision_reference_matmul():
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(previous)


def _input(dtype, width, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    # TE permits TF32 for FP32 GEMMs independently of PyTorch's precision
    # setting. BF16-representable operands isolate the TP/autograd contract
    # from that backend choice while retaining FP32 accumulation/gradients.
    return torch.randn(128, 2, width, device="cuda", generator=generator).bfloat16().to(dtype)


def _assert_close(actual, expected, dtype):
    if dtype == torch.bfloat16:
        torch.testing.assert_close(actual.double(), expected, atol=8e-3, rtol=8e-3)
    else:
        torch.testing.assert_close(actual.double(), expected, atol=5e-5, rtol=3e-5)


@pytest.mark.parametrize("tp", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_f_decay_projection_matches_complete_matrix_reference(tp, dtype):
    """A replicated down-projection gradient is complete, not scaled by TP."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=tp)
    try:
        model_parallel_cuda_manual_seed(123)
        config = replace(
            _make_config(tp_size=tp, sequence_parallel=tp > 1, params_dtype=dtype, f_lora_rank=16),
            gradient_accumulation_fusion=False,
        )
        model = _build_kda(config)
        projection = model.f_a_proj
        if dtype == torch.float32:
            with torch.no_grad():
                projection.weight.copy_(projection.weight.bfloat16().float())
        assert not projection.sequence_parallel
        assert all(not parameter.sequence_parallel for parameter in projection.parameters())
        rank = parallel_state.get_tensor_model_parallel_rank()
        full_input = _input(dtype, config.hidden_size, 101)
        full_probe = _input(dtype, config.kda_f_lora_rank, 102)
        local_input = full_input.chunk(tp, dim=0)[rank].clone().requires_grad_(True)
        output = model._project_f_latent(local_input)
        reference = F.linear(full_input.double(), projection.weight.detach().double())
        assert output.shape == reference.chunk(tp, dim=0)[rank].shape
        _assert_close(output, reference.chunk(tp, dim=0)[rank], dtype)
        output.backward(full_probe.chunk(tp, dim=0)[rank])
        reference_dgrad = full_probe.double().matmul(projection.weight.detach().double())
        reference_wgrad = (
            full_probe.flatten(0, 1).double().t().matmul(full_input.flatten(0, 1).double())
        )
        _assert_close(local_input.grad, reference_dgrad.chunk(tp, dim=0)[rank], dtype)
        _assert_close(projection.weight.grad, reference_wgrad, dtype)
    finally:
        Utils.destroy_model_parallel()


def test_kda_forward_uses_complete_sequence_for_f_decay_projection():
    """Exercise the real KDA path and independently check its captured F gradient."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=2)
    try:
        model_parallel_cuda_manual_seed(123)
        config = replace(
            _make_config(tp_size=2, sequence_parallel=True, f_lora_rank=16),
            gradient_accumulation_fusion=False,
        )
        model = _build_kda(config)
        rank = parallel_state.get_tensor_model_parallel_rank()
        full_input = _input(torch.bfloat16, config.hidden_size, 103)
        captured = {}

        def capture(module, inputs, output):
            captured["input"] = inputs[0].detach().clone()
            output[0].register_hook(
                lambda gradient: captured.update(gradient=gradient.detach().clone())
            )

        handle = model.f_a_proj.register_forward_hook(capture)
        local_input = full_input.chunk(2, dim=0)[rank].clone().requires_grad_(True)
        output, _ = model(local_input, attention_mask=None)
        output.backward(_input(torch.bfloat16, config.hidden_size, 104).chunk(2, dim=0)[rank])
        handle.remove()
        torch.testing.assert_close(captured["input"], full_input, rtol=0, atol=0)
        assert captured["gradient"].shape[:2] == full_input.shape[:2]
        reference_wgrad = (
            captured["gradient"]
            .flatten(0, 1)
            .double()
            .t()
            .matmul(full_input.flatten(0, 1).double())
        )
        _assert_close(model.f_a_proj.weight.grad, reference_wgrad, torch.bfloat16)
        assert not model.f_a_proj.weight.sequence_parallel
    finally:
        Utils.destroy_model_parallel()
