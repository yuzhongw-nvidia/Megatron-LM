# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Independent router-gradient checks for changes in token partitioning."""

import pytest
import torch

import megatron.core.transformer.moe.moe_utils as moe_utils

pytestmark = [
    pytest.mark.internal,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def _run(x, dy, weight, bias, splits, fusion, main_grad=True, zero_out=False):
    w = weight.detach().clone().requires_grad_()
    b = bias.detach().clone().requires_grad_() if bias is not None else None
    for param in (w, b):
        if param is not None and main_grad:
            param.main_grad = torch.zeros_like(param, dtype=torch.float32)
            param.grad_added_to_main_grad = False
            param.zero_out_wgrad = zero_out
    outputs, input_grads = [], []
    ordinary_wgrad = torch.zeros_like(w, dtype=torch.float32)
    ordinary_bgrad = torch.zeros_like(b, dtype=torch.float32) if b is not None else None
    for input_chunk, dy_chunk in zip(x.tensor_split(splits), dy.tensor_split(splits)):
        inp = input_chunk.detach().clone().requires_grad_()
        out = moe_utils.router_gating_linear(
            inp, w, b, torch.float32, gradient_accumulation_fusion=fusion
        )
        out.backward(dy_chunk)
        outputs.append(out.detach())
        input_grads.append(inp.grad)
        for param, accumulator in ((w, ordinary_wgrad), (b, ordinary_bgrad)):
            if param is None:
                continue
            if fusion and main_grad:
                assert param.grad_added_to_main_grad
                assert param.grad is not None and torch.count_nonzero(param.grad) == 0
            else:
                assert param.grad is not None
                accumulator.add_(param.grad.float())
            param.grad = None
    return {
        'output': torch.cat(outputs),
        'dinput': torch.cat(input_grads),
        'dweight': w.main_grad if fusion and main_grad else ordinary_wgrad,
        'dbias': b.main_grad if fusion and main_grad and b is not None else ordinary_bgrad,
    }


def _relative_error(expected, actual):
    return ((actual.double() - expected.double()).norm() / expected.double().norm()).item()


@pytest.mark.parametrize('backend', ['te', 'torch'])
@pytest.mark.parametrize('has_bias,zero_out', [(False, False), (True, False), (True, True)])
def test_router_fp32_accumulation_token_partitions(monkeypatch, backend, has_bias, zero_out):
    if backend == 'torch':
        monkeypatch.setattr(moe_utils, 'te_general_gemm', None)
    elif moe_utils.te_general_gemm is None:
        pytest.skip('Transformer Engine unavailable')
    torch.manual_seed(17473)
    # BF16-representable operands isolate local-gradient rounding from TF32 input conversion.
    x = torch.randn(2048, 256, device='cuda', dtype=torch.bfloat16)
    dy = torch.randn(2048, 32, device='cuda', dtype=torch.bfloat16).float()
    weight = torch.randn(32, 256, device='cuda', dtype=torch.bfloat16)
    bias = torch.randn(32, device='cuda', dtype=torch.bfloat16) if has_bias else None
    oracle = dy.double().t() @ x.double()
    expected_bias = dy.double().sum(dim=0) if has_bias else None
    results = {}
    for splits in (1, 2, 4):
        legacy = _run(x, dy, weight, bias, splits, False, zero_out=zero_out)
        fused = _run(x, dy, weight, bias, splits, True, zero_out=zero_out)
        assert torch.equal(legacy['output'], fused['output'])
        assert torch.equal(legacy['dinput'], fused['dinput'])
        assert _relative_error(oracle, fused['dweight']) < 3e-6
        assert _relative_error(oracle, legacy['dweight']) > 1e-3
        if has_bias:
            assert _relative_error(expected_bias, fused['dbias']) < 1e-6
        results[splits] = fused
    for splits in (2, 4):
        assert _relative_error(results[1]['dweight'], results[splits]['dweight']) < 3e-6


@pytest.mark.parametrize('backend', ['te', 'torch'])
def test_router_accumulation_without_main_grad_preserves_autograd(monkeypatch, backend):
    if backend == 'torch':
        monkeypatch.setattr(moe_utils, 'te_general_gemm', None)
    elif moe_utils.te_general_gemm is None:
        pytest.skip('Transformer Engine unavailable')
    torch.manual_seed(3817)
    x = torch.randn(64, 128, device='cuda', dtype=torch.bfloat16)
    dy = torch.randn(64, 8, device='cuda', dtype=torch.bfloat16).float()
    w = torch.randn(8, 128, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(8, device='cuda', dtype=torch.bfloat16)
    legacy = _run(x, dy, w, b, 2, False, main_grad=False)
    enabled = _run(x, dy, w, b, 2, True, main_grad=False)
    for name in legacy:
        assert torch.equal(legacy[name], enabled[name]), name


def test_router_accumulation_rejects_wrong_main_grad_shape():
    inp = torch.randn(16, 32, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(8, 32, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    weight.main_grad = torch.zeros(4, 32, device='cuda', dtype=torch.float32)
    output = moe_utils.router_gating_linear(inp, weight, None, torch.float32, True)
    with pytest.raises(RuntimeError, match='full parameter-shaped buffer'):
        output.backward(torch.ones_like(output))
