# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import multi_latent_attention
from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
    AbsorbedMLASelfAttention,
)
from megatron.core.transformer.multi_latent_attention import MultiLatentAttention


class _ReachedProjection(Exception):
    """Stop at the Q/K/V projection, before expensive MLA computation."""


def _model(cp_size, projection=None):
    return SimpleNamespace(
        config=SimpleNamespace(
            cp_partition_mode="contiguous", cache_mla_latents=False, sequence_parallel=False
        ),
        cache_mla_latents=False,
        training=False,
        offload_qkv_linear=False,
        pg_collection=SimpleNamespace(cp=SimpleNamespace(size=lambda: cp_size), tp=None),
        get_query_key_value_tensors=projection,
    )


@pytest.fixture
def conversions(monkeypatch):
    """Record MLA's layout-adapter calls; the adapter returns a tensor of its own so that a
    test can tell whether the projection read the converted hidden states."""
    calls = []

    def convert(**kwargs):
        calls.append(kwargs)
        return torch.ones_like(kwargs["hidden_states"]), None

    monkeypatch.setattr(
        multi_latent_attention, "convert_module_input_tensors_cp_partition_mode", convert
    )
    return calls


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("input_layout", ["zigzag", "contiguous"])
def test_mla_converts_the_input_layout_to_zigzag_before_rope(
    conversions, packed, cp_size, input_layout
):
    """MLA RoPE and core attention consume the zigzag layout. ``MultiLatentAttention`` hands
    every input to the layout adapter with the zigzag target, as ``Attention.forward`` does,
    instead of rejecting a contiguous one, and projects the hidden states the adapter returns."""
    projected = []

    def projection(hidden_states, *args, **kwargs):
        projected.append(hidden_states)
        raise _ReachedProjection

    model = _model(cp_size, projection)
    # The manager annotation takes precedence over config for SBHD. Packed input's
    # metadata takes precedence over the module annotation for THD.
    model._cp_input_partition_mode = "zigzag" if packed else input_layout
    params = PackedSeqParams(qkv_format="thd", cp_partition_mode=input_layout) if packed else None
    hidden_states = torch.zeros(4, 1, 8)
    with pytest.raises(_ReachedProjection):
        MultiLatentAttention.forward(model, hidden_states, None, packed_seq_params=params)

    (call,) = conversions
    assert call["hidden_states"] is hidden_states
    assert call["target_partition_mode"] == "zigzag"
    assert call["packed_seq_params"] is params
    assert call["cp_group"] is model.pg_collection.cp
    assert call["config"] is model.config
    assert call["sequence_parallel"] is False
    # The projection (and with it the output gate) reads the converted hidden states.
    assert len(projected) == 1 and bool(projected[0].eq(1).all())


def test_mla_sbhd_leaves_the_source_layout_to_the_adapter_without_manager_annotation(conversions):
    """Without a manager annotation the adapter resolves the source layout from the config."""
    model = _model(2, projection=lambda *args, **kwargs: (_ for _ in ()).throw(_ReachedProjection))
    with pytest.raises(_ReachedProjection):
        MultiLatentAttention.forward(model, torch.zeros(4, 1, 8), None)
    (call,) = conversions
    assert call["config"] is model.config


@pytest.fixture
def distributed_initialized(monkeypatch):
    """``get_pg_size`` reads a group's size only once torch.distributed is initialized, so the
    stand-in CP group of these tests counts as size one in a process that has not done so."""
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("input_layout", ["zigzag", "contiguous"])
def test_absorbed_mla_checks_actual_input_layout_before_rope(
    distributed_initialized, packed, cp_size, input_layout
):
    def projection(*args, **kwargs):
        raise _ReachedProjection

    model = _model(cp_size, projection)
    if not packed:
        model._cp_input_partition_mode = input_layout
    else:
        model._cp_input_partition_mode = "zigzag"
    params = PackedSeqParams(qkv_format="thd", cp_partition_mode=input_layout) if packed else None
    # The dev DSA path converts projected latents and supports contiguous input.
    with pytest.raises(_ReachedProjection):
        AbsorbedMLASelfAttention.forward(
            model, torch.zeros(4, 1, 8), None, packed_seq_params=params
        )
