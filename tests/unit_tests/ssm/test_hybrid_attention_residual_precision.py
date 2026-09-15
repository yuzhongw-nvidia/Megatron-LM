# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Hybrid residual ownership against the Block AttnRes equations.

Reference: https://arxiv.org/abs/2603.15031, Block Attention Residuals.
The partial block accumulates raw sublayer outputs, independently of the
depth-aggregated sublayer input. Kimi-K3 uses a hidden size of 7168.
"""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.hybrid.hybrid_block import AttnResHybridLayer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from tests.unit_tests.test_utilities import Utils


class _SmallBranch(torch.nn.Module):
    """Keep the branch small enough to expose residual-add cancellation."""

    def __init__(self, config, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.full((config.hidden_size,), 1e-3, dtype=config.params_dtype)
        )

    def forward(self, hidden_states, **kwargs):
        return hidden_states * self.weight, None


def _native_aggregation(values, query, norm_weight, eps):
    stacked = torch.stack([value.float() for value in values])
    keys = stacked * torch.rsqrt(stacked.square().mean(dim=-1, keepdim=True) + eps)
    logits = (keys * norm_weight * query).sum(dim=-1)
    weights = torch.softmax(logits, dim=0)
    return (weights.unsqueeze(-1) * stacked).sum(dim=0).to(values[0].dtype)


def _assert_similar(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-7)
    a, b = actual.flatten().double(), expected.flatten().double()
    energy = a.square().sum() + b.square().sum()
    if energy == 0:
        return
    assert F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), eps=1e-30).item() > 1 - 1e-4
    assert (2 * (a * b).sum() / energy).item() > 1 - 1e-4


class TestHybridAttentionResidualPrecision:
    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("kind", ["attention", "mlp", "mamba"])
    @pytest.mark.parametrize("position", ["block_start", "inside_block", "mtp"])
    @pytest.mark.parametrize("fused", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_small_branch_matches_native(self, kind, position, fused, dtype):
        config = TransformerConfig(
            num_layers=18,
            hidden_size=7168,
            num_attention_heads=96,
            kv_channels=64,
            params_dtype=dtype,
            bf16=dtype == torch.bfloat16,
            use_cpu_initialization=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            bias_dropout_fusion=fused,
            enable_attention_residuals=True,
            attn_res_block_layers=8,
            attn_res_impl="eager",
            is_hybrid_model=True,
            mtp_num_layers=1,
        )
        groups = ProcessGroupCollection.use_mpu_process_groups()
        layer_number = 1 if position == "block_start" else 2
        branch_spec = ModuleSpec(module=_SmallBranch)
        if kind == "mamba":
            inner = MambaLayer(
                config,
                MambaLayerSubmodules(
                    norm=IdentityOp, mixer=branch_spec, mamba_bda=get_bias_dropout_add
                ),
                layer_number=layer_number,
                pg_collection=groups,
            )
            branch = inner.mixer
        else:
            submodules = (
                TransformerLayerSubmodules(
                    self_attention=branch_spec, self_attn_bda=get_bias_dropout_add
                )
                if kind == "attention"
                else TransformerLayerSubmodules(mlp=branch_spec, mlp_bda=get_bias_dropout_add)
            )
            inner = TransformerLayer(
                config, submodules, layer_number=layer_number, pg_collection=groups
            )
            branch = inner.self_attention if kind == "attention" else inner.mlp
        layer = AttnResHybridLayer(config, inner, is_mtp_layer=position == "mtp").cuda()
        source = torch.ones(8, 1, config.hidden_size, dtype=dtype, device="cuda")
        sources = [
            (source * (index + 1)).requires_grad_(True)
            for index in range(layer.attn_res_num_sources)
        ]
        partial = torch.full_like(source, 3e-5, requires_grad=True)
        values = sources if position == "block_start" else [*sources, partial]
        reference_values = [value.detach().clone().requires_grad_(True) for value in values]
        reference_query = layer.attn_res.pseudo_query.detach().clone().requires_grad_(True)
        reference_norm = layer.attn_res.key_norm_weight.detach().clone().requires_grad_(True)
        reference_weight = branch.weight.detach().clone().requires_grad_(True)
        aggregated = _native_aggregation(
            reference_values, reference_query, reference_norm, layer.attn_res.eps
        )
        raw_branch = aggregated * reference_weight
        expected = raw_branch if position == "block_start" else reference_values[-1] + raw_branch
        actual = layer(partial, attn_res_sources=tuple(sources))
        _assert_similar(actual, expected)

        torch.manual_seed(17)
        grad = torch.randn_like(actual)
        actual_inputs = [
            *values,
            layer.attn_res.pseudo_query,
            layer.attn_res.key_norm_weight,
            branch.weight,
        ]
        reference_inputs = [*reference_values, reference_query, reference_norm, reference_weight]
        actual_grads = torch.autograd.grad(actual, actual_inputs, grad)
        expected_grads = torch.autograd.grad(expected, reference_inputs, grad)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            _assert_similar(actual_grad, expected_grad)
