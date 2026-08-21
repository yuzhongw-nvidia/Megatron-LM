# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CP orchestration tests for the internal fused GDR backward."""

from types import SimpleNamespace

import pytest
import torch


def _implementation():
    pytest.importorskip("fla")
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend import implementation

    return implementation


@pytest.mark.parametrize("use_saved_h", [False, True])
def test_cp_backward_preprocessing_produces_fused_dht_and_state(monkeypatch, use_saved_h):
    implementation = _implementation()
    shape = (1, 64, 2, 4)
    scalar_shape = shape[:-1]
    q = torch.empty(shape, dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    g = torch.empty(scalar_shape, dtype=torch.float32)
    beta = torch.empty(scalar_shape, dtype=torch.float32)
    A = torch.empty((*scalar_shape, 64), dtype=torch.bfloat16)
    do = torch.empty_like(q)
    compressed_initial_state = torch.empty((1, 2, 4, 4), dtype=torch.float32)
    expanded_initial_state = torch.empty_like(compressed_initial_state)
    recomputed_h = torch.empty((1, 1, 2, 4, 4), dtype=torch.float32)
    saved_h = torch.empty((1, 2, 4, 4), dtype=torch.bfloat16)
    expected_h = saved_h if use_saved_h else recomputed_h
    expected_dht = torch.empty((1, 2, 4, 4), dtype=torch.float32)
    w = torch.empty_like(k)
    u = torch.empty_like(v)
    dv = torch.empty_like(v)
    cp_context = SimpleNamespace(group=object())
    seen = {}

    monkeypatch.setattr(implementation, "recompute_w_u_fwd", lambda **_kwargs: (w, u))
    monkeypatch.setattr(
        implementation,
        "expand_h0",
        lambda initial_state, *, context: expanded_initial_state,
    )

    def fwd_h(**kwargs):
        seen["fwd_h_initial_state"] = kwargs["initial_state"]
        return recomputed_h, torch.empty_like(v), None

    def cp_preprocess(**kwargs):
        seen["cp_preprocess"] = kwargs
        return expected_dht, None

    monkeypatch.setattr(implementation, "chunk_gated_delta_rule_fwd_h", fwd_h)
    monkeypatch.setattr(implementation, "chunk_bwd_dv_local", lambda **_kwargs: dv)
    monkeypatch.setattr(implementation, "chunk_gated_delta_rule_bwd_dhu_pre_process", cp_preprocess)

    actual_dht, actual_h = implementation._fla_cp_backward_preprocess(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A=A,
        scale=0.5,
        do=do,
        dht=None,
        cu_seqlens=None,
        chunk_indices=None,
        initial_state=compressed_initial_state,
        cp_context=cp_context,
        h=saved_h if use_saved_h else None,
    )

    assert actual_dht is expected_dht
    assert actual_h is expected_h
    assert ("fwd_h_initial_state" in seen) is not use_saved_h
    if not use_saved_h:
        assert seen["fwd_h_initial_state"] is expanded_initial_state
    assert seen["cp_preprocess"]["w"] is w
    assert seen["cp_preprocess"]["dv"] is dv
    assert seen["cp_preprocess"]["initial_state"] is expanded_initial_state
    assert seen["cp_preprocess"]["context"] is cp_context


def test_cutedsl_cp_backward_feeds_preprocessed_dht_to_fused_kernel(monkeypatch):
    implementation = _implementation()
    shape = (1, 64, 64, 128)
    scalar_shape = shape[:-1]
    q = torch.empty(shape, dtype=torch.bfloat16)
    inputs = {
        "q": q,
        "k": torch.empty_like(q),
        "v": torch.empty_like(q),
        "g": torch.empty(scalar_shape, dtype=torch.float32),
        "beta": torch.empty(scalar_shape, dtype=torch.float32),
        "A": torch.empty((*scalar_shape, 64), dtype=torch.bfloat16),
        "do": torch.empty_like(q),
    }
    cp_context = SimpleNamespace(group=object())
    initial_state = torch.empty((1, 64, 128, 128), dtype=torch.float32)
    cp_dht = torch.empty_like(initial_state)
    cp_h = torch.empty((1, 64, 128, 128), dtype=torch.bfloat16)
    expected = tuple(torch.empty(1) for _ in range(5))
    seen = {}

    monkeypatch.setattr(implementation, "_fused_bwd_support_reason", lambda **_kwargs: None)
    monkeypatch.setattr(
        implementation,
        "_fla_cp_backward_preprocess",
        lambda **_kwargs: (cp_dht, cp_h),
    )

    def fused(**kwargs):
        seen.update(kwargs)
        return expected

    monkeypatch.setattr(implementation, "_call_fused_gdr_bwd_cute", fused)

    result = implementation._cutedsl_backward(
        **inputs,
        scale=128**-0.5,
        dht=None,
        cu_seqlens=None,
        chunk_indices=None,
        initial_state=initial_state,
        cp_context=cp_context,
    )

    assert result is expected
    assert seen["dht"] is cp_dht
    assert seen["h"] is cp_h


def test_cutedsl_cp_auto_fallback_preserves_cp_boundary_inputs(monkeypatch):
    implementation = _implementation()
    shape = (1, 64, 64, 128)
    scalar_shape = shape[:-1]
    q = torch.empty(shape, dtype=torch.float32)
    inputs = {
        "q": q,
        "k": torch.empty_like(q),
        "v": torch.empty_like(q),
        "g": torch.empty(scalar_shape),
        "beta": torch.empty(scalar_shape),
        "A": torch.empty((*scalar_shape, 64)),
        "do": torch.empty_like(q),
    }
    cp_context = SimpleNamespace(group=object())
    initial_state = torch.empty((1, 64, 128, 128), dtype=torch.float32)
    dht = torch.empty_like(initial_state)
    expected = tuple(torch.empty(1) for _ in range(5))
    seen = {}

    monkeypatch.setattr(implementation, "_fused_bwd_support_reason", lambda **_kwargs: "unsupported")
    monkeypatch.setattr(implementation, "_backend_mode", lambda: "auto")

    def fla_backward(**kwargs):
        seen.update(kwargs)
        return expected

    monkeypatch.setattr(implementation, "_fla_backward", fla_backward)

    result = implementation._cutedsl_backward(
        **inputs,
        scale=128**-0.5,
        dht=dht,
        cu_seqlens=None,
        chunk_indices=None,
        initial_state=initial_state,
        cp_context=cp_context,
    )

    assert result is expected
    assert seen["dht"] is dht
    assert seen["initial_state"] is initial_state
    assert seen["cp_context"] is cp_context


def test_cp_fla_fallback_uses_exp2_gate_semantics(monkeypatch):
    implementation = _implementation()
    tensor = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
    dht = torch.zeros_like(tensor)
    initial_state = torch.zeros_like(tensor)
    calls = {}

    def call_fla_compat(function, **kwargs):
        calls[function] = kwargs
        if function is implementation.recompute_w_u_fwd:
            return tensor, tensor
        if function is implementation.chunk_gated_delta_rule_fwd_h:
            return tensor, tensor, None
        if function is implementation.chunk_bwd_dv_local:
            return tensor
        if function is implementation.chunk_gated_delta_rule_bwd_dhu_pre_process:
            return dht, initial_state
        if function is implementation.chunk_gated_delta_rule_bwd_dhu:
            return tensor, tensor, tensor
        if function is implementation.chunk_bwd_dqkwg:
            return tensor, tensor.clone(), tensor, tensor.clone()
        if function is implementation.prepare_wy_repr_bwd:
            return tensor, tensor, tensor, tensor
        raise AssertionError(f"unexpected FLA primitive: {function}")

    monkeypatch.setattr(implementation, "_call_fla_compat", call_fla_compat)
    monkeypatch.setattr(implementation, "expand_h0", lambda state, *, context: state)
    monkeypatch.setattr(implementation, "chunk_local_cumsum", lambda value, **_kwargs: value)

    implementation._fla_backward(
        q=tensor,
        k=tensor,
        v=tensor,
        g=tensor,
        beta=tensor,
        A=tensor,
        scale=1.0,
        do=tensor,
        cu_seqlens=None,
        chunk_indices=None,
        dht=dht,
        initial_state=initial_state,
        cp_context=SimpleNamespace(group=object()),
    )

    gate_primitives = (
        implementation.recompute_w_u_fwd,
        implementation.chunk_gated_delta_rule_fwd_h,
        implementation.chunk_bwd_dv_local,
        implementation.chunk_gated_delta_rule_bwd_dhu_pre_process,
        implementation.chunk_gated_delta_rule_bwd_dhu,
        implementation.chunk_bwd_dqkwg,
        implementation.prepare_wy_repr_bwd,
    )
    assert all(calls[function]["use_exp2"] is True for function in gate_primitives)
