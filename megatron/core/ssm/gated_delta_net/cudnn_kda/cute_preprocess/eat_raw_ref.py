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


def _intra_A(
    kc: torch.Tensor, gc: torch.Tensor, bc: torch.Tensor, bc_block: int = 16
) -> torch.Tensor:
    """Strictly-lower intra-chunk matrix (sub-chunk referenced, overflow-safe).

    ``A[..., c, i] = -beta_c * <k_c * 2^(g_c - g_i), k_i>`` for ``c > i`` (else 0),
    with ``gc`` the chunk-local gate cumsum.  The naive single-gemm factorisation
    ``(k e^{g}) @ (k e^{-g})^T`` overflows in fp32 once the cumulative chunk decay
    is large (the bounded gate reaches ``-5`` per token, so ``e^{-g}`` can exceed
    fp32's ``e^88`` over a 64-chunk).  The kernel -- and FLA -- bound the range by
    building ``A`` over ``bc_block`` (16) blocks, each referenced to its row
    block's leading cumsum ``gc_ref``: ``e^{g_c - g_ref}`` and ``e^{g_ref - g_i}``
    then both stay within one block's decay span.  Algebraically exact
    (``e^{g_c - g_ref} e^{g_ref - g_i} = e^{g_c - g_i}``), so the result equals the
    direct build up to fp reassociation.
    """
    bt, kdim = kc.shape[-2], kc.shape[-1]
    nb = bt // bc_block
    a = kc.new_zeros(tuple(kc.shape[:-1]) + (bt,))
    for r in range(nb):
        rs = slice(r * bc_block, (r + 1) * bc_block)
        ref = gc[..., r * bc_block, :].unsqueeze(-2)  # [..., 1, K] row-block leader
        kd_r = kc[..., rs, :] * (gc[..., rs, :] - ref).exp()  # e^{g_c - g_ref} <= 1
        for c in range(r + 1):  # only row block >= col block can be non-zero
            cs = slice(c * bc_block, (c + 1) * bc_block)
            ke_c = kc[..., cs, :] * (ref - gc[..., cs, :]).exp()
            a[..., rs, cs] = kd_r @ ke_c.transpose(-1, -2)
    a = a * bc[..., None]  # row scale by beta_c
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


_RCP_LN2 = 1.4426950408889634  # 1 / ln(2); FLA's log-decay -> log2 scale


def _apply_gate(g: torch.Tensor, gate_kwargs: dict) -> torch.Tensor:
    """Raw cuDNN gate operand ``g`` ``[T, HV, K]`` -> per-token log-decay (fp32).

    Mirrors ``producer._gate_cumsum`` / FLA's ``kda_gate_chunk_cumsum`` pre-cumsum
    step: the bounded ("safe") gate ``lower_bound * sigmoid(exp(a_log)*(g+dt_bias))``,
    or ``g`` already the log decay otherwise.  The softplus (unbounded) form has no
    fused kernel in this backend and is applied upstream, so it arrives here as the
    log decay too (``safe_gate`` false).
    """
    g = g.float()
    if not gate_kwargs.get("safe_gate"):
        return g
    a_log = gate_kwargs.get("a_log")
    lower_bound = gate_kwargs.get("gate_lower_bound")
    if a_log is None or lower_bound is None:
        raise ValueError("front_half_fwd safe_gate needs a_log and gate_lower_bound")
    hv, kdim = g.shape[-2], g.shape[-1]
    inner = g
    dt_bias = gate_kwargs.get("dt_bias")
    if dt_bias is not None:
        inner = inner + dt_bias.float().reshape(hv, kdim)
    return float(lower_bound) * torch.sigmoid(a_log.float().reshape(hv, 1).exp() * inner)


def _chunk_cumsum(ld: torch.Tensor) -> torch.Tensor:
    """Chunk-local (BT=64) cumsum of the log-decay ``ld`` ``[Tpad, HV, K]`` (fp32).

    ``Tpad`` is a multiple of 64; the cumsum restarts at every chunk boundary.
    """
    tpad, hv, kdim = ld.shape
    nt = tpad // CHUNK_SIZE
    return ld.reshape(nt, CHUNK_SIZE, hv, kdim).cumsum(1).reshape(tpad, hv, kdim)


def front_half_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    gate_kwargs: dict,
    io_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused forward front-half: raw window operands -> ``(w, u, kg, gk)``.

    Torch reference of the work the eat-raw kernel does, replacing FLA's
    ``kda_gate_chunk_cumsum`` + ``chunk_kda_fwd_intra`` glue.  Inputs are the
    boundary window (THD, ``B == 1`` dropped): ``k`` ``[T, HV, K]``, ``v``
    ``[T, HV, V]``, ``g`` ``[T, HV, K]`` the raw gate operand, ``beta`` ``[T, HV]``
    post-sigmoid; ``HV == H`` (the backend refuses grouped value attention).
    Returns the operands the fused CP scan kernel consumes, with the chunk/batch
    layout it expects: ``w`` ``[1, T, HV, K]`` io-dtype, ``u`` ``[1, T, HV, V]``
    io-dtype, ``kg`` ``[1, T, HV, K]`` io-dtype, ``gk`` ``[1, T, HV, K]`` fp32.

    A ragged last chunk is padded to 64 with neutral tokens (``k = v = beta = 0``,
    log-decay 0): the pad rows add nothing to the real rows' WY transform and are
    trimmed from the outputs, matching FLA's own chunk kernels.
    """
    t, hv, kdim = k.shape
    vdim = v.shape[-1]
    nt = (t + CHUNK_SIZE - 1) // CHUNK_SIZE
    tpad = nt * CHUNK_SIZE

    def _pad(x):
        if x.shape[0] == tpad:
            return x.float()
        out = x.new_zeros((tpad,) + tuple(x.shape[1:]), dtype=torch.float32)
        out[:t] = x.float()
        return out

    ld = _apply_gate(g, gate_kwargs)  # [T, HV, K] log-decay
    ld_p = _pad(ld)  # pad rows: log-decay 0 (decay == 1, neutral)
    gc = _chunk_cumsum(ld_p)  # [Tpad, HV, K] natural-log chunk cumsum
    k_p, v_p, b_p = _pad(k), _pad(v), _pad(beta)

    # [HV, NT, BT, .]
    kc = k_p.permute(1, 0, 2).reshape(hv, nt, CHUNK_SIZE, kdim)
    vc = v_p.permute(1, 0, 2).reshape(hv, nt, CHUNK_SIZE, vdim)
    gcc = gc.permute(1, 0, 2).reshape(hv, nt, CHUNK_SIZE, kdim)
    bcc = b_p.permute(1, 0).reshape(hv, nt, CHUNK_SIZE)

    w_c, u_c, kg_c = wy_representation(
        kc.reshape(hv * nt, CHUNK_SIZE, kdim),
        vc.reshape(hv * nt, CHUNK_SIZE, vdim),
        gcc.reshape(hv * nt, CHUNK_SIZE, kdim),
        bcc.reshape(hv * nt, CHUNK_SIZE),
    )

    def _unchunk(x, last):
        x = x.reshape(hv, nt, CHUNK_SIZE, last).permute(1, 2, 0, 3).reshape(tpad, hv, last)
        return x[:t]

    w = _unchunk(w_c, kdim).to(io_dtype).unsqueeze(0).contiguous()
    u = _unchunk(u_c, vdim).to(io_dtype).unsqueeze(0).contiguous()
    kg = _unchunk(kg_c, kdim).to(io_dtype).unsqueeze(0).contiguous()
    gk = (gc[:t] * _RCP_LN2).unsqueeze(0).contiguous()  # [1, T, HV, K] fp32
    return w, u, kg, gk


def _intra_Aqk(
    qc: torch.Tensor, kc: torch.Tensor, gc: torch.Tensor, scale: float, bc_block: int = 16
) -> torch.Tensor:
    """Intra-chunk query-key attention ``Aqk[..., c, j] = scale * <q_c 2^(g_c-g_j), k_j>``.

    Lower-triangular incl. the diagonal (``j <= c``, strictly-upper masked to 0);
    NOT a triangular inverse -- a plain masked decayed gemm.  Sub-chunk referenced
    the same way as :func:`_intra_A` so the ``2^(-g)`` factor stays overflow-safe in
    fp32 under the bounded gate.
    """
    bt = qc.shape[-2]
    nb = bt // bc_block
    a = qc.new_zeros(tuple(qc.shape[:-1]) + (bt,))
    for r in range(nb):
        rs = slice(r * bc_block, (r + 1) * bc_block)
        ref = gc[..., r * bc_block, :].unsqueeze(-2)
        qd_r = (qc[..., rs, :] * scale) * (gc[..., rs, :] - ref).exp()
        for c in range(r + 1):  # j-block <= c-block
            cs = slice(c * bc_block, (c + 1) * bc_block)
            ke_c = kc[..., cs, :] * (ref - gc[..., cs, :]).exp()
            a[..., rs, cs] = qd_r @ ke_c.transpose(-1, -2)
    mask_strict_upper = torch.triu(
        torch.ones(bt, bt, dtype=torch.bool, device=a.device), diagonal=1
    )
    return a.masked_fill(mask_strict_upper, 0)


def front_half_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    gate_kwargs: dict,
    io_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused backward front-half: raw window operands -> ``(qg, kg, w, dv, gk)``.

    Torch reference of the work the eat-raw backward kernel does, replacing FLA's
    backward glue (``kda_gate_chunk_cumsum`` + ``chunk_kda_fwd_intra`` +
    ``chunk_kda_bwd_dAv``).  Inputs are the backward boundary window (the FIRST
    local sequence; THD, ``B == 1`` dropped): ``q`` / ``k`` ``[T, HV, K]``, ``g``
    the raw gate operand, ``beta`` ``[T, HV]``, ``do`` ``[T, HV, V]``, ``scale``;
    ``HV == H``.  Returns the operands the fused CP backward scan kernel consumes,
    ``[1, T, HV, .]``: ``qg`` = ``q * 2^gk``, ``kg`` and ``w`` (the same WY
    front-half as the forward), ``dv`` = ``Aqk^T @ do`` (``Aqk`` the lower-incl.
    diagonal decayed q-k attention), all io-dtype, and ``gk`` fp32.

    ``dv = Aqk^T @ do`` (not ``Aqk @ do``): the ``^T`` is the transpose FLA's
    ``chunk_kda_bwd_dAv`` folds into its A load; this convention makes the backward
    boundary gradient agree with autodiff of ``naive_chunk_kda`` to machine
    precision (``test_kda_cute_eat_raw``).  A ragged last chunk is neutrally padded
    like the forward.
    """
    t, hv, kdim = q.shape
    vdim = do.shape[-1]
    nt = (t + CHUNK_SIZE - 1) // CHUNK_SIZE
    tpad = nt * CHUNK_SIZE

    def _pad(x):
        if x.shape[0] == tpad:
            return x.float()
        out = x.new_zeros((tpad,) + tuple(x.shape[1:]), dtype=torch.float32)
        out[:t] = x.float()
        return out

    ld = _apply_gate(g, gate_kwargs)
    gc = _chunk_cumsum(_pad(ld))  # [Tpad, HV, K] natural-log cumsum
    q_p, k_p, do_p, b_p = _pad(q), _pad(k), _pad(do), _pad(beta)

    qc = q_p.permute(1, 0, 2).reshape(hv * nt, CHUNK_SIZE, kdim)
    kc = k_p.permute(1, 0, 2).reshape(hv * nt, CHUNK_SIZE, kdim)
    doc = do_p.permute(1, 0, 2).reshape(hv * nt, CHUNK_SIZE, vdim)
    gcc = gc.permute(1, 0, 2).reshape(hv * nt, CHUNK_SIZE, kdim)
    bcc = b_p.permute(1, 0).reshape(hv * nt, CHUNK_SIZE)

    # w / kg from the forward WY (u, computed from `do` as a stand-in v, is unused).
    w_c, _u_c, kg_c = wy_representation(kc, doc, gcc, bcc)
    qg_c = qc * gcc.exp()  # q * 2^gk
    aqk = _intra_Aqk(qc, kc, gcc, scale)
    dv_c = aqk.transpose(-1, -2) @ doc  # dv = Aqk^T @ do

    def _unchunk(x, last):
        x = x.reshape(hv, nt, CHUNK_SIZE, last).permute(1, 2, 0, 3).reshape(tpad, hv, last)
        return x[:t]

    qg = _unchunk(qg_c, kdim).to(io_dtype).unsqueeze(0).contiguous()
    kg = _unchunk(kg_c, kdim).to(io_dtype).unsqueeze(0).contiguous()
    w = _unchunk(w_c, kdim).to(io_dtype).unsqueeze(0).contiguous()
    dv = _unchunk(dv_c, vdim).to(io_dtype).unsqueeze(0).contiguous()
    gk = (gc[:t] * _RCP_LN2).unsqueeze(0).contiguous()
    return qg, kg, w, dv, gk
