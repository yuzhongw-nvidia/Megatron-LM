# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""FP64 norm references, parameter dtype preservation, and TP gradient reduction."""

import json

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import (
    TELayerNormColumnParallelLinear,
    TENorm,
    TERMSNormDuplicatedLinear,
    _get_norm_precision_kwargs,
)
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import Float16Module
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.internal


def _config(**kwargs):
    values = dict(
        num_layers=1, hidden_size=128, num_attention_heads=4, bf16=True, params_dtype=torch.bfloat16
    )
    values.update(kwargs)
    return TransformerConfig(**values)


def _relative_error(actual, expected):
    return float(
        (actual.double() - expected.double()).norm() / expected.double().norm().clamp_min(1e-30)
    )


@pytest.mark.parametrize("hidden", [128, 512, 3584, 7168])
@pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
def test_fp32_norm_parameter_gradient_matches_full_fp64(hidden, normalization):
    Utils.initialize_model_parallel(tensor_model_parallel_size=2)
    try:
        group = parallel_state.get_tensor_model_parallel_group()
        rank = parallel_state.get_tensor_model_parallel_rank()
        config = _config(
            tensor_model_parallel_size=2,
            sequence_parallel=True,
            normalization=normalization,
            normalization_in_fp32=True,
        )
        generator = torch.Generator(device="cuda").manual_seed(181)
        full_x = torch.randn(2048, 1, hidden, device="cuda", generator=generator).bfloat16()
        probe = (torch.randn(full_x.shape, device="cuda", generator=generator) * 1e-4).bfloat16()
        gamma = (torch.randn(hidden, device="cuda", generator=generator) * 0.1 + 1).bfloat16()
        beta = (torch.randn(hidden, device="cuda", generator=generator) * 0.1).bfloat16()
        norm = TENorm(config, hidden, eps=1e-6)
        baseline = TENorm(_config(normalization=normalization), hidden, eps=1e-6)
        original_keys = set(baseline.state_dict())
        with torch.no_grad():
            baseline.weight.copy_(gamma)
            if normalization == "LayerNorm":
                baseline.bias.copy_(beta)
        # Existing BF16 checkpoint values load into FP32 parameters without key remapping.
        norm.load_state_dict(baseline.state_dict(), strict=True)
        assert set(norm.state_dict()) == original_keys
        container = torch.nn.ModuleDict({"norm": norm, "linear": torch.nn.Linear(hidden, 8)})
        wrapped = Float16Module(config=config, module=container)
        assert wrapped.module["linear"].weight.dtype == torch.bfloat16
        for param in norm.parameters():
            assert param.dtype == torch.float32 and param.keep_in_fp32
            assert param.sequence_parallel is True
        x = full_x.chunk(2, dim=0)[rank].clone().requires_grad_()
        # Nested AMP must not silently recast gamma or the normalization arithmetic.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = norm(x)
        y.backward(probe.chunk(2, dim=0)[rank].contiguous())
        assert y.dtype == x.grad.dtype == torch.bfloat16
        grads = {}
        for name, param in norm.named_parameters():
            assert param.grad.dtype == torch.float32
            grads[name] = param.grad.detach().clone()
            torch.distributed.all_reduce(grads[name], group=group)
        ref_x = full_x.double().requires_grad_()
        ref_gamma = gamma.double().requires_grad_()
        ref_beta = beta.double().requires_grad_()
        centered = ref_x - ref_x.mean(-1, keepdim=True) if normalization == "LayerNorm" else ref_x
        ref_y = centered * torch.rsqrt(centered.square().mean(-1, keepdim=True) + 1e-6) * ref_gamma
        if normalization == "LayerNorm":
            ref_y = ref_y + ref_beta
        ref_y.backward(probe.double())
        errors = dict(
            output=_relative_error(y, ref_y.detach().bfloat16().chunk(2, dim=0)[rank]),
            dgrad=_relative_error(x.grad, ref_x.grad.bfloat16().chunk(2, dim=0)[rank]),
            dgamma=_relative_error(grads['weight'], ref_gamma.grad),
        )
        assert errors['output'] < 2e-4, errors
        assert errors['dgrad'] < 3e-4, errors
        assert errors['dgamma'] < 1e-5, errors
        if normalization == "LayerNorm":
            errors['dbeta'] = _relative_error(grads['bias'], ref_beta.grad)
            assert errors['dbeta'] < 1e-5, errors
        print(
            'MCORE_FP32_NORM_PASS '
            + json.dumps(dict(hidden=hidden, norm=normalization, rank=rank, errors=errors)),
            flush=True,
        )
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize('duplicated', [False, True])
def test_fused_norm_parameters_survive_float16_module(duplicated):
    Utils.initialize_model_parallel(tensor_model_parallel_size=2)
    try:
        group = parallel_state.get_tensor_model_parallel_group()
        config = _config(
            tensor_model_parallel_size=2,
            sequence_parallel=True,
            normalization='RMSNorm',
            normalization_in_fp32=True,
            gradient_accumulation_fusion=False,
        )
        kwargs = dict(
            config=config,
            init_method=torch.nn.init.normal_,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            is_expert=False,
            tp_group=group,
        )
        if duplicated:
            model = TERMSNormDuplicatedLinear(128, 128, parallel_mode='duplicated', **kwargs)
        else:
            model = TELayerNormColumnParallelLinear(128, 128, gather_output=False, **kwargs)
        wrapped = Float16Module(config=config, module=model)
        assert wrapped.module is model
        assert model.layer_norm_weight.dtype == torch.float32
        assert model.layer_norm_weight.keep_in_fp32
        assert model.layer_norm_weight.sequence_parallel
        assert model.weight.dtype == torch.bfloat16
        generator = torch.Generator(device='cuda').manual_seed(661)
        x = torch.randn(32, 1, 128, generator=generator, device='cuda').bfloat16().requires_grad_()
        y, _ = model(x)
        y.backward(torch.ones_like(y))
        assert model.layer_norm_weight.grad.dtype == torch.float32
        assert x.grad.dtype == y.dtype == model.weight.grad.dtype == torch.bfloat16
        assert torch.isfinite(model.layer_norm_weight.grad).all()
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize(
    'kwargs,message',
    [
        ({'bf16': False}, 'ordinary BF16'),
        ({'fp8': 'hybrid'}, 'ordinary BF16'),
        ({'transformer_impl': 'local'}, 'transformer_impl'),
        ({'symmetric_ar_type': 'one_shot'}, 'overlap'),
        ({'fused_residual_rmsnorm': True}, 'fused_residual_rmsnorm'),
    ],
)
def test_fp32_norm_rejects_unsupported_execution(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _config(normalization_in_fp32=True, **kwargs)


def test_fp32_norm_requires_explicit_te_interface():
    class LegacyModule:
        def __init__(self, **kwargs):
            pass

    class SupportedModule:
        def __init__(self, normalization_in_fp32=False):
            pass

    assert _get_norm_precision_kwargs(_config(), LegacyModule) == {}
    config = _config(normalization_in_fp32=True)
    with pytest.raises(RuntimeError, match='requires a Transformer Engine build'):
        _get_norm_precision_kwargs(config, LegacyModule)
    assert _get_norm_precision_kwargs(config, SupportedModule) == {'normalization_in_fp32': True}
