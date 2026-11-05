# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU spec tests for the "eat-raw" CuTe KDA CP pre-process reference.

``cute_preprocess.eat_raw_ref`` is the device-independent numerical oracle for a
fused CuTe DSL producer that would consume the RAW KDA operands
(``k`` / ``v`` / ``g`` / ``beta``) and fold FLA's gate cumsum + WY transform
(``w`` / ``u`` / ``kg``) into the kernel, in place of the current ``cute``
producer's FLA Triton glue.  These tests pin that front-half's math against an
independent, self-contained FLA ``naive_chunk_kda`` state recurrence (no FLA,
cutlass or GPU needed), so the eventual tcgen05 kernel has a verified oracle:

* the forward boundary summary ``H`` equals the span's final state under a zero
  initial state (to fp64 machine precision);
* the transition ``M`` obeys the affine law ``S_out = M @ S_in + H`` the CP fold
  relies on, and spans compose as ``M = M_B @ M_A`` / ``H = M_B @ H_A + H_B``;
* the hierarchical (16-block) ``solve_tril`` the kernel uses to bound the
  decay-exponent range equals the direct forward substitution.
"""

import pytest
import torch

from megatron.core.ssm.gated_delta_net.cudnn_kda.cute_preprocess import eat_raw_ref

_F64 = torch.float64
_BT = eat_raw_ref.CHUNK_SIZE  # 64


def _naive_chunk_state(k, v, g, beta, initial_state=None):
    """Independent FLA ``naive_chunk_kda`` forward state recurrence (fp64, no q/o).

    Ground truth: ``k`` L2-normalised keys, ``v`` values, ``g`` per-key log-decay,
    ``beta`` post-sigmoid.  Returns the final state ``[HV, K, V]`` after ``T``
    tokens in ``BT``-chunks, seeded with ``initial_state`` if given.
    """
    t, hv, kdim = k.shape
    vdim = v.shape[-1]
    nt = t // _BT
    k, v, g, beta = (x.to(_F64) for x in (k, v, g, beta))
    kc = k.permute(1, 0, 2).reshape(hv, nt, _BT, kdim)
    vc = v.permute(1, 0, 2).reshape(hv, nt, _BT, vdim)
    gc = g.permute(1, 0, 2).reshape(hv, nt, _BT, kdim).cumsum(-2)
    bc = beta.permute(1, 0).reshape(hv, nt, _BT)
    # WY transform via the strictly-lower A build + exact forward-substitution.
    kd, ke = kc * gc.exp(), kc * (-gc).exp()
    a = (kd @ ke.transpose(-1, -2)) * bc[..., None]
    a = -a.masked_fill(torch.triu(torch.ones(_BT, _BT, dtype=torch.bool), 0), 0)
    for i in range(1, _BT):
        a[..., i, :i] = a[..., i, :i].clone() + (
            a[..., i, :, None].clone() * a[..., :, :i].clone()
        ).sum(-2)
    akk = (a + torch.eye(_BT, dtype=_F64)) * bc[..., None, :]
    w, u = akk @ (gc.exp() * kc), akk @ vc
    state = torch.zeros(hv, kdim, vdim, dtype=_F64)
    if initial_state is not None:
        state = state + initial_state.to(_F64)
    for i in range(nt):
        g_i, k_i, u_i, w_i = gc[:, i], kc[:, i], u[:, i], w[:, i]
        v_i = u_i - w_i @ state
        state = state * g_i[:, -1].exp()[..., None]
        state = state + ((g_i[:, -1:] - g_i).exp() * k_i).transpose(-1, -2) @ v_i
    return state


def _rel_l2(a, b):
    a, b = a.to(_F64), b.to(_F64)
    num = torch.linalg.vector_norm((a - b).flatten())
    den = torch.linalg.vector_norm(b.flatten()).clamp_min(1e-30)
    return (num / den).item()


def _rand_inputs(nt, hv=2, kdim=64, vdim=64, seed=0):
    g_ = torch.Generator().manual_seed(seed)
    t = nt * _BT
    k = torch.randn(t, hv, kdim, dtype=_F64, generator=g_)
    k = k / k.norm(dim=-1, keepdim=True)  # L2-normalised keys
    v = torch.randn(t, hv, vdim, dtype=_F64, generator=g_)
    beta = torch.rand(t, hv, dtype=_F64, generator=g_)  # post-sigmoid in (0, 1)
    g = -torch.rand(t, hv, kdim, dtype=_F64, generator=g_) * 0.5  # log-decay in (-0.5, 0)
    return k, v, g, beta


_TOL = 1e-9  # fp64; the reference is bit-equivalent to naive up to reassociation


@pytest.mark.parametrize("nt", [1, 2, 3])
def test_forward_summary_matches_naive(nt):
    """``H`` == final state under a zero initial state."""
    k, v, g, beta = _rand_inputs(nt)
    big_h, _ = eat_raw_ref.eat_raw_summary_fwd(k, v, g, beta)
    assert _rel_l2(big_h, _naive_chunk_state(k, v, g, beta)) < _TOL


@pytest.mark.parametrize("nt", [1, 2, 3])
def test_transition_affine_law(nt):
    """``S_out = M @ S_in + H`` for an arbitrary initial state."""
    k, v, g, beta = _rand_inputs(nt, seed=1)
    big_h, big_m = eat_raw_ref.eat_raw_summary_fwd(k, v, g, beta)
    s0 = torch.randn(
        k.shape[1], k.shape[2], v.shape[2], dtype=_F64, generator=torch.Generator().manual_seed(7)
    )
    predicted = torch.einsum("h k j, h j v -> h k v", big_m, s0) + big_h
    assert _rel_l2(predicted, _naive_chunk_state(k, v, g, beta, initial_state=s0)) < _TOL


@pytest.mark.parametrize("nt", [2, 4])
def test_cp_fold_composition(nt):
    """Spans A (earlier) then B (later) compose as ``M = M_B @ M_A``, ``H = M_B @ H_A + H_B``."""
    k, v, g, beta = _rand_inputs(nt, seed=2)
    big_h, big_m = eat_raw_ref.eat_raw_summary_fwd(k, v, g, beta)
    ta = (nt // 2) * _BT
    ha, ma = eat_raw_ref.eat_raw_summary_fwd(k[:ta], v[:ta], g[:ta], beta[:ta])
    hb, mb = eat_raw_ref.eat_raw_summary_fwd(k[ta:], v[ta:], g[ta:], beta[ta:])
    h_comp = torch.einsum("h k j, h j v -> h k v", mb, ha) + hb
    m_comp = torch.einsum("h a b, h b c -> h a c", mb, ma)
    assert _rel_l2(h_comp, big_h) < _TOL
    assert _rel_l2(m_comp, big_m) < _TOL


@pytest.mark.parametrize("bc", [16, 32])
def test_subchunk_solve_matches_direct(bc):
    """The hierarchical ``bc``-block triangular inverse == direct forward substitution."""
    gen = torch.Generator().manual_seed(3)
    # a dense strictly-lower matrix with a realistic magnitude
    raw = torch.randn(2, 3, _BT, _BT, dtype=_F64, generator=gen) * 0.1
    a = raw.masked_fill(torch.triu(torch.ones(_BT, _BT, dtype=torch.bool), 0), 0)
    assert _rel_l2(eat_raw_ref.solve_tril_subchunked(a, bc), eat_raw_ref.solve_tril(a)) < _TOL


def test_wy_representation_shapes_and_kg():
    """``wy_representation`` returns right-shaped w/u/kg; kg is unit at the chunk end."""
    k, v, g, beta = _rand_inputs(1)
    hv, kdim, vdim = k.shape[1], k.shape[2], v.shape[2]
    kc = k.permute(1, 0, 2).reshape(hv, 1, _BT, kdim)[:, 0]
    vc = v.permute(1, 0, 2).reshape(hv, 1, _BT, vdim)[:, 0]
    gc = g.permute(1, 0, 2).reshape(hv, 1, _BT, kdim).cumsum(-2)[:, 0]
    bc = beta.permute(1, 0).reshape(hv, 1, _BT)[:, 0]
    w, u, kg = eat_raw_ref.wy_representation(kc, vc, gc, bc)
    assert w.shape == (hv, _BT, kdim) and u.shape == (hv, _BT, vdim) and kg.shape == (hv, _BT, kdim)
    # kg = k * 2^(g_last - g); the last valid row has g == g_last, so kg == k there.
    assert _rel_l2(kg[:, -1], kc[:, -1]) < _TOL


# --------------------------------------------------------------------------- #
# front_half_fwd: the producer-facing front-half (raw window -> w/u/kg/gk).
# Verified by feeding its output through the kernel's inter-chunk summary MATH
# and comparing the boundary state to naive_chunk_kda.  front_half_fwd works in
# fp32 internally (the kernel's dtype), so the bound is fp32-level, not fp64.
# --------------------------------------------------------------------------- #
_RCP_LN2 = 1.4426950408889634
_FRONT_TOL = 1e-5


def _scan_from_front_half(w, u, kg, gk, t):
    """Kernel MATH inter-chunk scan from ``(w, u, kg, gk)`` with a ragged tail mask."""
    _, big_t, hv, kdim = w.shape
    vdim = u.shape[-1]
    nt = (big_t + _BT - 1) // _BT
    tpad = nt * _BT

    def _pad(x):
        out = x.new_zeros((1, tpad, hv, x.shape[-1]), dtype=_F64)
        out[:, :big_t] = x.to(_F64)
        return out

    w, u, kg = _pad(w), _pad(u), _pad(kg)
    gkp = torch.zeros(1, tpad, hv, kdim, dtype=_F64)
    gkp[:, :big_t] = gk.to(_F64)
    state = torch.zeros(hv, kdim, vdim, dtype=_F64)
    for n in range(nt):
        lo, hi = n * _BT, min(n * _BT + _BT, big_t)
        nvalid = hi - lo
        w_i = w[0, lo : lo + _BT].permute(1, 0, 2)
        u_i = u[0, lo : lo + _BT].permute(1, 0, 2)
        kg_i = kg[0, lo : lo + _BT].permute(1, 0, 2)
        gamma = (gkp[0, lo + nvalid - 1] / _RCP_LN2).exp()[..., None]  # 2^gk_last = exp(gc_last)
        lam = torch.zeros(_BT, dtype=_F64)
        lam[:nvalid] = 1.0
        state = state * gamma + kg_i.transpose(-1, -2) @ ((u_i - w_i @ state) * lam[None, :, None])
    return state


@pytest.mark.parametrize("t,safe", [(128, False), (192, False), (160, True), (100, True)])
def test_front_half_fwd_matches_naive(t, safe):
    """``front_half_fwd``'s (w,u,kg,gk), scanned, reproduce naive -- full / ragged, both gates."""
    gen = torch.Generator().manual_seed(11)
    hv, kdim, vdim = 2, 64, 64
    k = torch.randn(t, hv, kdim, dtype=_F64, generator=gen)
    k = k / k.norm(dim=-1, keepdim=True)
    v = torch.randn(t, hv, vdim, dtype=_F64, generator=gen)
    beta = torch.rand(t, hv, dtype=_F64, generator=gen)
    if safe:  # the bounded ("safe") gate: raw logits + a_log / dt_bias / lower_bound
        g = torch.randn(t, hv, kdim, dtype=_F64, generator=gen) * 2.0
        a_log = torch.randn(hv, dtype=_F64, generator=gen)
        dt_bias = torch.randn(hv, kdim, dtype=_F64, generator=gen)
        gate_kwargs = {
            "safe_gate": True,
            "a_log": a_log,
            "dt_bias": dt_bias,
            "gate_lower_bound": -5.0,
        }
        log_decay = -5.0 * torch.sigmoid(a_log.reshape(hv, 1).exp() * (g + dt_bias))
    else:  # g already the log decay
        g = -torch.rand(t, hv, kdim, dtype=_F64, generator=gen) * 0.5
        gate_kwargs = {}
        log_decay = g
    w, u, kg, gk = eat_raw_ref.front_half_fwd(k, v, g, beta, gate_kwargs, io_dtype=_F64)
    assert torch.isfinite(gk).all() and torch.isfinite(w).all()
    predicted = _scan_from_front_half(w, u, kg, gk, t)
    # naive needs whole chunks: pad the real tokens with neutral ones (k = v = beta = 0,
    # log-decay 0) up to a multiple of 64; they add nothing to the boundary state.
    tpad = ((t + _BT - 1) // _BT) * _BT

    def _pad_naive(x):
        out = x.new_zeros((tpad,) + tuple(x.shape[1:]), dtype=_F64)
        out[:t] = x
        return out

    expected = _naive_chunk_state(
        _pad_naive(k), _pad_naive(v), _pad_naive(log_decay), _pad_naive(beta)
    )
    assert _rel_l2(predicted, expected) < _FRONT_TOL
