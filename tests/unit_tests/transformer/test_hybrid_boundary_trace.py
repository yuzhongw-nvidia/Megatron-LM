# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU tests for opt-in hybrid-stack boundary sidecars."""

import torch

from megatron.core.models.hybrid.hybrid_block import AttnResHybridLayer
from megatron.core.models.hybrid.hybrid_boundary_trace import (
    load_hybrid_boundary_trace,
    record_hybrid_boundary,
    reset_hybrid_boundary_trace_state,
)
from megatron.core.transformer.transformer_config import TransformerConfig


class _ResidualEntry(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_number = 1

    def forward(self, hidden_states, **_kwargs):
        return hidden_states + 1


def _payloads(output_dir):
    return {
        payload["stage"]: payload
        for payload in (
            load_hybrid_boundary_trace(path)
            for path in sorted(output_dir.glob("hybrid_boundary_*.pt"))
        )
    }


def test_hybrid_boundary_trace_is_disabled_without_path(tmp_path, monkeypatch):
    monkeypatch.delenv("MCORE_HYBRID_BOUNDARY_TRACE_PATH", raising=False)
    reset_hybrid_boundary_trace_state()
    assert record_hybrid_boundary(0, "incoming", torch.ones(2, 3)) is None
    assert not list(tmp_path.iterdir())


def test_hybrid_boundary_trace_filters_and_bounds_occurrences(tmp_path, monkeypatch):
    monkeypatch.setenv("MCORE_HYBRID_BOUNDARY_TRACE_PATH", str(tmp_path))
    monkeypatch.setenv("MCORE_HYBRID_BOUNDARY_TRACE_GLOBAL_RANKS", "0")
    monkeypatch.setenv("MCORE_HYBRID_BOUNDARY_TRACE_DECODER_LAYERS", "0")
    monkeypatch.setenv("MCORE_HYBRID_BOUNDARY_TRACE_STAGES", "incoming,aggregated")
    monkeypatch.setenv("MCORE_HYBRID_BOUNDARY_TRACE_MAX_OCCURRENCES", "1")
    reset_hybrid_boundary_trace_state()

    incoming = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    assert record_hybrid_boundary(0, "incoming", incoming) is not None
    assert record_hybrid_boundary(0, "incoming", incoming + 1) is None
    assert record_hybrid_boundary(0, "aggregated", incoming + 2) is not None
    assert record_hybrid_boundary(1, "incoming", incoming) is None
    assert record_hybrid_boundary(0, "new_partial", incoming) is None

    payloads = _payloads(tmp_path)
    assert set(payloads) == {"incoming", "aggregated"}
    assert torch.equal(payloads["incoming"]["tensor"], incoming.to(torch.bfloat16))
    assert payloads["incoming"]["original_dtype"] == "float32"
    assert payloads["incoming"]["occurrence"] == 0


def test_attn_res_hybrid_wrapper_records_three_boundaries(tmp_path, monkeypatch):
    monkeypatch.setenv("MCORE_HYBRID_BOUNDARY_TRACE_PATH", str(tmp_path))
    monkeypatch.setenv("MCORE_HYBRID_BOUNDARY_TRACE_GLOBAL_RANKS", "0")
    monkeypatch.setenv("MCORE_HYBRID_BOUNDARY_TRACE_DECODER_LAYERS", "0")
    monkeypatch.setenv("MCORE_HYBRID_BOUNDARY_TRACE_STAGES", "incoming,aggregated,new_partial")
    reset_hybrid_boundary_trace_state()

    config = TransformerConfig(
        num_layers=1,
        hidden_size=4,
        num_attention_heads=1,
        enable_attention_residuals=True,
        attn_res_block_layers=1,
        attn_res_impl="eager",
    )
    layer = AttnResHybridLayer(config, _ResidualEntry())
    hidden = torch.arange(24, dtype=torch.bfloat16).reshape(3, 2, 4)
    output = layer(hidden_states=hidden, attn_res_sources=(hidden,))

    payloads = _payloads(tmp_path)
    assert set(payloads) == {"incoming", "aggregated", "new_partial"}
    assert torch.equal(payloads["incoming"]["tensor"], hidden)
    assert torch.equal(payloads["aggregated"]["tensor"], hidden)
    assert torch.equal(payloads["new_partial"]["tensor"], output)
    assert torch.equal(output, torch.ones_like(hidden))
