# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CPU checks of the fused KDA backward dataflow specification."""

import pytest
import torch

from tests.unit_tests.ssm.kda_fused_bwd_reference import (
    kda_chunk_fwd_reference,
    kda_fused_bwd_reference,
    kda_recurrent_reference,
)

H, K, V = 2, 32, 16


def _inputs(lengths, gate_low, gate_high, seed=0):
    generator = torch.Generator().manual_seed(seed)
    total = sum(lengths)

    def randn(*shape, std=1.0):
        return torch.randn(*shape, generator=generator, dtype=torch.float64) * std

    q = randn(total, H, K, std=0.3)
    k = torch.nn.functional.normalize(randn(total, H, K), dim=-1)
    v = randn(total, H, V)
    g = gate_low + (gate_high - gate_low) * torch.rand(
        total, H, K, generator=generator, dtype=torch.float64
    )
    beta = torch.rand(total, H, generator=generator, dtype=torch.float64)
    h0 = randn(len(lengths), H, K, V, std=0.5)
    do = randn(total, H, V)
    dht = randn(len(lengths), H, K, V)
    cu_seqlens = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0)), dtype=torch.int32)
    return q, k, v, g, beta, h0, do, dht, cu_seqlens


def _oracle(q, k, v, g, beta, h0, do, dht, cu_seqlens, scale):
    leaves = [x.clone().requires_grad_() for x in (q, k, v, g, beta, h0)]
    o, final = kda_recurrent_reference(*leaves[:5], scale, leaves[5], cu_seqlens)
    ((o * do).sum() + (final * dht).sum()).backward()
    return [x.grad for x in leaves]


@pytest.mark.parametrize(
    "lengths,gate_low,gate_high",
    [
        ([128], -0.2, 0.0),
        ([200, 67], -1.0, 0.0),
        ([17, 64, 65], -0.5, -0.01),
        # Per-token natural gates down to -20 make 64-token spans of ~1800 in
        # log2; any shared chunk anchor would overflow FP32 here.
        ([160], -20.0, -5.0),
    ],
)
def test_reference_matches_recurrent_autograd(lengths, gate_low, gate_high):
    q, k, v, g, beta, h0, do, dht, cu_seqlens = _inputs(lengths, gate_low, gate_high)
    scale = K**-0.5
    gc, aqk, akk, h, final = kda_chunk_fwd_reference(q, k, v, g, beta, scale, h0, cu_seqlens)
    _, final_recurrent = kda_recurrent_reference(q, k, v, g, beta, scale, h0, cu_seqlens)
    torch.testing.assert_close(final, final_recurrent, rtol=1e-10, atol=1e-10)

    expected = _oracle(q, k, v, g, beta, h0, do, dht, cu_seqlens, scale)
    actual = kda_fused_bwd_reference(q, k, v, gc, beta, aqk, akk, h, do, dht, scale, cu_seqlens)
    for name, got, ref in zip(("dq", "dk", "dv", "dg", "dbeta", "dh0"), actual, expected):
        assert torch.isfinite(got).all(), name
        torch.testing.assert_close(got, ref, rtol=1e-9, atol=1e-9, msg=name)


def test_uniform_channel_gate_reduces_to_scalar_gate():
    """Broadcast gates use the general path; channel dg sums equal the GDN dg."""
    q, k, v, g, beta, h0, do, dht, cu_seqlens = _inputs([150], -0.6, -0.05, seed=3)
    g = g[..., :1].expand_as(g).contiguous()
    scale = K**-0.5
    gc, aqk, akk, h, _ = kda_chunk_fwd_reference(q, k, v, g, beta, scale, h0, cu_seqlens)
    dq, dk, dv, dg, dbeta, dh0 = kda_fused_bwd_reference(
        q, k, v, gc, beta, aqk, akk, h, do, dht, scale, cu_seqlens
    )
    scalar = g[..., :1].clone().requires_grad_()
    leaves = [x.clone().requires_grad_() for x in (q, k, v, beta, h0)]
    o, final = kda_recurrent_reference(
        leaves[0],
        leaves[1],
        leaves[2],
        scalar.expand_as(g),
        leaves[3],
        scale,
        leaves[4],
        cu_seqlens,
    )
    ((o * do).sum() + (final * dht).sum()).backward()
    torch.testing.assert_close(dg.sum(-1, keepdim=True), scalar.grad, rtol=1e-9, atol=1e-9)
    for got, leaf in zip((dq, dk, dv, dbeta, dh0), leaves):
        torch.testing.assert_close(got, leaf.grad, rtol=1e-9, atol=1e-9)


def _mixed_gate_inputs(seed):
    q, k, v, g, beta, h0, do, dht, cu_seqlens = _inputs([130, 67], -0.3, -0.01, seed=seed)
    # A quarter of the channels decay by 2^-14..2^-29 per token, so their
    # 16-token diagonal spans exceed DIAG_SPAN_LIMIT; the rest stay weak.
    strong = -20.0 + 10.0 * torch.rand(
        g.shape[:-1] + (K // 4,), generator=torch.Generator().manual_seed(seed + 1), dtype=g.dtype
    )
    g[..., : K // 4] = strong
    return q, k, v, g, beta, h0, do, dht, cu_seqlens


def test_anchored_diagonal_with_channel_fallback():
    q, k, v, g, beta, h0, do, dht, cu_seqlens = _mixed_gate_inputs(seed=5)
    scale = K**-0.5
    gc, aqk, akk, h, _ = kda_chunk_fwd_reference(q, k, v, g, beta, scale, h0, cu_seqlens)
    expected = _oracle(q, k, v, g, beta, h0, do, dht, cu_seqlens, scale)
    actual = kda_fused_bwd_reference(
        q,
        k,
        v,
        gc,
        beta,
        aqk,
        akk,
        h,
        do,
        dht,
        scale,
        cu_seqlens,
        intra_operand_dtype=torch.float32,
        diag_mode="anchored",
    )
    for name, got, ref in zip(("dq", "dk", "dv", "dg", "dbeta", "dh0"), actual, expected):
        assert torch.isfinite(got).all(), name
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5, msg=name)


def test_anchored_diagonal_needs_fallback(monkeypatch):
    """Without the per-channel fallback the strong channels overflow."""
    from tests.unit_tests.ssm import kda_fused_bwd_reference as reference

    monkeypatch.setattr(reference, "DIAG_SPAN_LIMIT", float("inf"))
    q, k, v, g, beta, h0, do, dht, cu_seqlens = _mixed_gate_inputs(seed=5)
    scale = K**-0.5
    gc, aqk, akk, h, _ = kda_chunk_fwd_reference(q, k, v, g, beta, scale, h0, cu_seqlens)
    dq, dk = reference.kda_fused_bwd_reference(
        q,
        k,
        v,
        gc,
        beta,
        aqk,
        akk,
        h,
        do,
        dht,
        scale,
        cu_seqlens,
        intra_operand_dtype=torch.float32,
        diag_mode="anchored",
    )[:2]
    assert not (torch.isfinite(dq).all() and torch.isfinite(dk).all())
