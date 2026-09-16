# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Configuration and dependency checks for FP32 tensor-parallel partial sums."""

import pytest

from megatron.core.extensions.transformer_engine import _get_tp_reduce_precision_kwargs
from megatron.core.transformer import TransformerConfig


def _config(**kwargs):
    values = dict(num_layers=1, hidden_size=128, num_attention_heads=4, bf16=True)
    values.update(kwargs)
    return TransformerConfig(**values)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"bf16": False}, "ordinary BF16"),
        ({"fp8": "hybrid"}, "ordinary BF16"),
        ({"transformer_impl": "local"}, "transformer_impl"),
        (
            {"tp_comm_overlap": True, "sequence_parallel": True, "tensor_model_parallel_size": 2},
            "overlap",
        ),
        ({"symmetric_ar_type": "one_shot"}, "overlap"),
        ({"num_moe_experts": 4, "expert_tensor_parallel_size": 2}, "expert_tensor_parallel_size"),
    ],
)
def test_fp32_tp_rejects_unsupported_execution(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _config(tp_reduce_in_fp32=True, **kwargs)


def test_fp32_tp_requires_explicit_te_interface():
    class LegacyModule:
        def __init__(self, **kwargs):
            pass

    class SupportedModule:
        def __init__(self, tp_reduce_in_fp32=False):
            pass

    assert _get_tp_reduce_precision_kwargs(_config(), LegacyModule) == {}
    enabled = _config(tp_reduce_in_fp32=True)
    with pytest.raises(RuntimeError, match="requires a Transformer Engine build"):
        _get_tp_reduce_precision_kwargs(enabled, LegacyModule)
    assert _get_tp_reduce_precision_kwargs(enabled, SupportedModule) == {"tp_reduce_in_fp32": True}
