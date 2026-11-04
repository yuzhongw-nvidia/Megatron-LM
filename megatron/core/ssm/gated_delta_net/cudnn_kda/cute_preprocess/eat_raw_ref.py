# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Numerical spec for an "eat-raw" CuTe KDA CP pre-process (forward + backward).

The shipped ``cute`` producer (``producer.py``) feeds the fused CuTe DSL kernels
FLA's chunk-level intermediates: the fp32 log2 gate cumsum ``gk``, the WY
representation ``w`` / ``u`` and the pre-gated keys ``kg``, all materialised to
HBM by a stack of FLA Triton kernels before the CuTe launch (the "glue").  The
goal of the ``yuzhongw/kda-cute-eat-raw`` line is to fold that glue -- gate
cumsum + intra-chunk ``A`` build + ``solve_tril`` + ``w`` / ``u`` / ``kg``
derivation -- into the fused kernel so it consumes the RAW operands
(``k`` / ``v`` / ``g`` / ``beta``) directly, as cudnn-frontend's single
``kimi_delta_attention_summary`` frost kernel does.

This module is the authoritative, device-independent fp32/fp64 reference for
that front-half composed with the kernel's own inter-chunk summary scan.  It is
*not* a fast path -- it is the oracle the tcgen05 kernel must match and the
executable spec of the math (verified against FLA ``naive_chunk_kda`` to machine
precision by ``tests/.../test_kda_cute_eat_raw.py``).  It imports only ``torch``;
no cutlass, no flash-linear-attention, no CUDA.

Conventions (cuDNN KDA / FLA, see ``cp.py`` and ``naive.py``):

* ``k`` ``[T, HV, K]`` L2-normalised keys, ``v`` ``[T, HV, V]`` values,
  ``beta`` ``[T, HV]`` post-sigmoid, ``g`` ``[T, HV, K]`` per-key log-decay
  (natural log; the kernel works in the ``2^`` / log2 domain, which is
  algebraically identical -- ``2^(gk) == exp(g_cumsum)`` with
  ``gk = RCP_LN2 * g_cumsum``).  ``B == 1`` THD, chunk size 64, ``K == V``.
* Boundary summary ``(H, M)``: with a zero initial state the span maps
  ``S_out = M @ S_in + H`` in the ``[K, V]`` state domain (``M`` acts on the
  left of ``S``).  The CP fold composes spans A (earlier) then B (later) as
  ``M = M_B @ M_A`` and ``H = M_B @ H_A + H_B`` -- exactly the fold
  ``CudnnSummaryProducer`` / the fused kernel perform across ranks.
"""

from __future__ import annotations

import torch

CHUNK_SIZE = 64


def _chunk(x: torch.Tensor, nt: int, last: int) -> torch.Tensor:
    """``[T, HV, D]`` -> ``[HV, NT, BT, D]`` (BT == ``CHUNK_SIZE``)."""
    hv = x.shape[1]
    return x.permute(1, 0, 2).reshape(hv, nt, CHUNK_SIZE, last)


def _intra_A(kc: torch.Tensor, gc: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
    """Strictly-lower intra-chunk matrix.

    ``A[..., c, i] = -beta_c * <k_c * 2^(g_c - g_i), k_i>`` for ``c > i`` (else 0),
    with ``gc`` the chunk-local gate cumsum.  Written as a single decayed gemm
    ``A = -tril(beta_c * (k e^{g}) @ (k e^{-g})^T, -1)``; the kernel bounds the
    ``e^{-g}`` range by sub-chunking (``BC = 16``), which does not change the
    result (``test_subchunk_solve_matches_direct``).
    """
    bt = kc.shape[-2]
    kd = kc * gc.exp()  # k * e^{g}
    ke = kc * (-gc).exp()  # k * e^{-g}
    a = (kd @ ke.transpose(-1, -2)) * bc[..., None]  # [..., BT, BT], row-scaled by beta
    mask_incl = torch.triu(torch.ones(bt, bt, dtype=torch.bool, device=a.device), diagonal=0)
    return -a.masked_fill(mask_incl, 0)


def solve_tril(a: torch.Tensor) -> torch.Tensor:
    """``(I - stril(A))^{-1} - I`` for a strictly-lower ``A`` (FLA's exact recurrence).

    Forward substitution: row ``i`` gains ``sum_j A[i, j] * inv[j, :i]`` from the
    already-solved rows above it.  This is the sequential UT-transform step the
    kernel runs per chunk (hierarchically, over 16x16 diagonal blocks).
    """
    bt = a.shape[-1]
    a = a.clone()
    for i in range(1, bt):
        a[..., i, :i] = a[..., i, :i].clone() + (
            a[..., i, :, None].clone() * a[..., :, :i].clone()
        ).sum(-2)
    return a


def solve_tril_subchunked(a: torch.Tensor, bc: int = 16) -> torch.Tensor:
    """``(I - stril(A))^{-1} - I`` via hierarchical ``bc``-block forward substitution.

    The strategy the fused kernel uses so the ``e^{-g}`` factors in ``A`` stay in
    range: invert the ``bc x bc`` strictly-lower DIAGONAL blocks exactly, then
    block-substitute ``X_rc = D_r @ sum_{c<=m<r} L_rm X_mc`` (``D_r`` the inverted
    diagonal block incl. identity).  Equivalent to :func:`solve_tril`
    (``test_subchunk_solve_matches_direct``).  ``A`` strictly-lower ``[..., BT, BT]``.
    """
    bt = a.shape[-1]
    assert bt % bc == 0
    nb = bt // bc
    eye = torch.eye(bc, dtype=a.dtype, device=a.device)
    diag_inv = []  # D_r = (I - L_rr)^{-1}, one per diagonal block
    for d in range(nb):
        sl = slice(d * bc, (d + 1) * bc)
        diag_inv.append(solve_tril(a[..., sl, sl]) + eye)
    x = torch.zeros_like(a)  # the full inverse (incl. identity)
    for r in range(nb):
        rs = slice(r * bc, (r + 1) * bc)
        x[..., rs, rs] = diag_inv[r]
        for c in range(r):
            cs = slice(c * bc, (c + 1) * bc)
            acc = torch.zeros_like(diag_inv[r])
            for m in range(c, r):
                ms = slice(m * bc, (m + 1) * bc)
                acc = acc + a[..., rs, ms] @ x[..., ms, cs]
            x[..., rs, cs] = diag_inv[r] @ acc
    return x - torch.eye(bt, dtype=a.dtype, device=a.device)


def wy_representation(
    kc: torch.Tensor, vc: torch.Tensor, gc: torch.Tensor, bc: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-chunk ``w``, ``u``, ``kg`` from raw chunk operands (the fused front-half).

    ``Akk = (I - stril(A))^{-1}`` with column ``i`` scaled by ``beta_i``;
    ``w = Akk @ (k * 2^g)``, ``u = Akk @ v``, ``kg = k * 2^(g_last - g)``.
    Shapes: ``kc/gc`` ``[..., BT, K]``, ``vc`` ``[..., BT, V]``, ``bc`` ``[..., BT]``.
    """
    bt = kc.shape[-2]
    eye = torch.eye(bt, dtype=kc.dtype, device=kc.device)
    akk = (solve_tril(_intra_A(kc, gc, bc)) + eye) * bc[..., None, :]
    w = akk @ (gc.exp() * kc)
    u = akk @ vc
    g_last = gc[..., -1:, :]
    kg = (g_last - gc).exp() * kc
    return w, u, kg


def eat_raw_summary_fwd(
    k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward boundary summary ``(H, M)`` from raw operands.

    Mirrors the fused kernel data flow: per chunk derive ``w`` / ``u`` / ``kg``
    (front-half), then update the ``[K, V] | [K, K]`` state
    ``x = u - w @ X`` ; ``X <- diag(gamma) @ X + kg^T @ x`` with
    ``gamma = 2^(g_last)`` per key row.  ``H`` is the ``[HV, K, V]`` state under a
    zero init, ``M`` the ``[HV, K, K]`` span transition.
    """
    t, hv, kdim = k.shape
    vdim = v.shape[-1]
    nt = t // CHUNK_SIZE
    kc, vc = _chunk(k, nt, kdim), _chunk(v, nt, vdim)
    bc = beta.permute(1, 0).reshape(hv, nt, CHUNK_SIZE)
    gc_all = _chunk(g, nt, kdim).cumsum(-2)
    big_h = k.new_zeros(hv, kdim, vdim)
    big_m = torch.eye(kdim, dtype=k.dtype, device=k.device).expand(hv, kdim, kdim).clone()
    for n in range(nt):
        gc = gc_all[:, n]
        w, u, kg = wy_representation(kc[:, n], vc[:, n], gc, bc[:, n])
        gamma = gc[:, -1].exp()[..., None]  # [HV, K, 1]
        xh = u - w @ big_h
        xm = -(w @ big_m)
        big_h = big_h * gamma + kg.transpose(-1, -2) @ xh
        big_m = big_m * gamma + kg.transpose(-1, -2) @ xm
    return big_h, big_m


# The backward boundary-gradient summary (G, Tr) is the reverse-scan mirror of
# the forward: it reuses the same WY front-half (w / kg / qg = q * 2^g) and adds
# the intra-chunk term dv = Aqk @ do (Aqk the lower-triangular decayed q-k
# attention, NOT a triangular inverse -- a plain masked gemm).  Its exact (G, Tr)
# convention and V-major layout are derived in worklogs/.../lebron_cute_eat_raw.md
# and must be validated against FLA's backward producer on a Blackwell GPU before
# it is wired in, so it is deliberately NOT shipped here as a verified oracle.
