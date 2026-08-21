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


def test_dense_batch_cp_forward_batches_local_compute_and_slices_boundaries(monkeypatch):
    implementation = _implementation()
    shape = (2, 64, 2, 4)
    q = torch.empty(shape, dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    g = torch.empty(shape[:-1], dtype=torch.float32)
    beta = torch.empty(shape[:-1], dtype=torch.bfloat16)
    w = torch.empty_like(k)
    u = torch.empty_like(v)
    A = torch.empty((*shape[:-1], 64), dtype=torch.bfloat16)
    h = torch.empty((2, 1, 2, 4, 4), dtype=torch.bfloat16)
    output = torch.empty_like(q)
    cp_context = SimpleNamespace(cu_seqlens=torch.tensor([0, 64], dtype=torch.int32))
    calls = {"intra": [], "cp_boundary": [], "fwd_h": [], "output": []}

    monkeypatch.setattr(implementation, "chunk_local_cumsum", lambda value, **_kwargs: value)

    def intra(**kwargs):
        calls["intra"].append(kwargs["k"].shape)
        return w, u, A

    def cp_boundary(**kwargs):
        calls["cp_boundary"].append(kwargs["k"].shape)
        return torch.empty((1, 2, 4, 4), dtype=torch.float32)

    def fwd_h(**kwargs):
        calls["fwd_h"].append((kwargs["k"].shape, kwargs["initial_state"].shape))
        return h, u, None

    def fwd_o(**kwargs):
        calls["output"].append(kwargs["q"].shape)
        return output

    monkeypatch.setattr(implementation, "chunk_gated_delta_rule_fwd_intra", intra)
    monkeypatch.setattr(implementation, "chunk_gated_delta_rule_fwd_h_pre_process", cp_boundary)
    monkeypatch.setattr(implementation, "chunk_gated_delta_rule_fwd_h", fwd_h)
    monkeypatch.setattr(implementation, "chunk_fwd_o", fwd_o)
    monkeypatch.setattr(implementation, "compress_h0", lambda state, *, context: state)

    actual_g, actual_output, actual_A, saved_h, chunk_indices, initial_state = (
        implementation._fla_forward_for_fused_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=0.5,
            cu_seqlens=None,
            cu_seqlens_cpu=None,
            cp_context=cp_context,
            save_fused_bwd_state=False,
        )
    )

    assert actual_g is g
    assert actual_output is output
    assert actual_A is A
    assert saved_h is None
    assert chunk_indices is None
    assert initial_state.shape == (2, 2, 4, 4)
    assert calls == {
        "intra": [torch.Size([2, 64, 2, 4])],
        "cp_boundary": [torch.Size([1, 64, 2, 4])] * 2,
        "fwd_h": [(torch.Size([2, 64, 2, 4]), torch.Size([2, 2, 4, 4]))],
        "output": [torch.Size([2, 64, 2, 4])],
    }


def test_dense_batch_cp_backward_batches_local_compute_and_slices_boundaries(monkeypatch):
    implementation = _implementation()
    shape = (2, 64, 2, 4)
    q = torch.zeros(shape, dtype=torch.bfloat16)
    q[1].fill_(1)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    g = torch.empty(shape[:-1], dtype=torch.float32)
    beta = torch.empty(shape[:-1], dtype=torch.bfloat16)
    A = torch.empty((*shape[:-1], 64), dtype=torch.bfloat16)
    do = torch.empty_like(q)
    w = torch.empty_like(k)
    u = torch.empty_like(v)
    dv = torch.empty_like(v)
    initial_state = torch.empty((2, 2, 4, 4), dtype=torch.float32)
    h = torch.empty((2, 1, 2, 4, 4), dtype=torch.bfloat16)
    cp_context = SimpleNamespace(cu_seqlens=torch.tensor([0, 64], dtype=torch.int32))
    calls = {"recompute": [], "dv": [], "cp_boundary": []}

    def recompute(**kwargs):
        calls["recompute"].append(kwargs["k"].shape)
        return w, u

    def dv_local(**kwargs):
        calls["dv"].append(kwargs["q"].shape)
        return dv

    def cp_boundary(**kwargs):
        calls["cp_boundary"].append(kwargs["q"].shape)
        batch_value = int(kwargs["q"][0, 0, 0, 0].item())
        dht = torch.full((1, 2, 4, 4), batch_value, dtype=torch.float32)
        return dht, None

    monkeypatch.setattr(implementation, "recompute_w_u_fwd", recompute)
    monkeypatch.setattr(implementation, "expand_h0", lambda state, *, context: state)
    monkeypatch.setattr(implementation, "chunk_bwd_dv_local", dv_local)
    monkeypatch.setattr(implementation, "chunk_gated_delta_rule_bwd_dhu_pre_process", cp_boundary)

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
        initial_state=initial_state,
        cp_context=cp_context,
        h=h,
    )

    assert actual_h is h
    assert actual_dht.shape == (2, 2, 4, 4)
    assert torch.equal(actual_dht[:, 0, 0, 0], torch.tensor([0.0, 1.0]))
    assert calls == {
        "recompute": [torch.Size([2, 64, 2, 4])],
        "dv": [torch.Size([2, 64, 2, 4])],
        "cp_boundary": [torch.Size([1, 64, 2, 4])] * 2,
    }


def test_dense_cu_seqlens_reuses_cached_tensor():
    implementation = _implementation()
    first = implementation._dense_cu_seqlens(2, 64, torch.device("cpu"))
    second = implementation._dense_cu_seqlens(2, 64, torch.device("cpu"))

    assert first is second
    assert first.tolist() == [0, 64, 128]
