# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Executable dataflow specification for the fused KDA backward kernel.

The functions here mirror, chunk by chunk, the operation decomposition that
``fused_kda_bwd_cute`` implements.
They are written in plain PyTorch so the numerical scheme can be checked against
an independent autograd oracle without the device kernel:

* ``kda_recurrent_reference`` is the token-recurrent definition of KDA; its
  autograd gradients are the oracle.
* ``kda_chunk_fwd_reference`` produces the forward residuals the backward
  consumes (chunk-local log2 gate cumsum, ``Aqk``, ``Akk`` and chunk states).
* ``kda_fused_bwd_reference`` is the backward specification.  ``operand_dtype``
  rounds every tile the kernel materializes as an MMA operand, and
  ``intra_operand_dtype``/``intra_square_dtype`` round the anchored token tile
  and the 64x64 score-gradient tile of each intra-chunk product, so the
  same code models both exact math and the kernel's BF16 rounding points.
  ``diag_mode="anchored"`` models the tensor-core diagonal blocks with their
  per-channel exact fallback.

Units follow FLA KDA: ``gc`` is the chunk-local inclusive cumsum of the natural
log gate multiplied by log2(e); ``dg`` is the gradient with respect to the
per-token natural log gate (FLA's ``dg`` after its reverse chunk cumsum).
"""

import torch

BT = 64
BC = 16
LOG2E = 1.4426950408889634


def kda_recurrent_reference(q, k, v, g, beta, scale, h0, cu_seqlens):
    """Token-recurrent KDA forward over packed sequences.

    Shapes: q/k/g ``[N, H, K]``, v ``[N, H, V]``, beta ``[N, H]``,
    h0 ``[B, H, K, V]``.  Returns ``(o, final_state)``.
    """
    outputs, finals = [], []
    for n in range(len(cu_seqlens) - 1):
        bos, eos = int(cu_seqlens[n]), int(cu_seqlens[n + 1])
        state = h0[n]
        for t in range(bos, eos):
            state = state * g[t].exp()[..., None]
            residual = v[t] - torch.einsum("hk,hkv->hv", k[t], state)
            state = state + torch.einsum("hk,hv->hkv", beta[t, :, None] * k[t], residual)
            outputs.append(scale * torch.einsum("hk,hkv->hv", q[t], state))
        finals.append(state)
    return torch.stack(outputs), torch.stack(finals)


def chunk_ranges(cu_seqlens):
    """Yield ``(sequence, first_token, valid_rows, global_chunk_index)``."""
    chunk = 0
    for n in range(len(cu_seqlens) - 1):
        bos, eos = int(cu_seqlens[n]), int(cu_seqlens[n + 1])
        for start in range(bos, eos, BT):
            yield n, start, min(BT, eos - start), chunk
            chunk += 1


def _chunk_tile(x, start, rows):
    """Copy ``rows`` tokens into a zero-padded ``[H, BT, ...]`` tile."""
    tile = x.new_zeros((x.shape[1], BT) + tuple(x.shape[2:]))
    tile[:, :rows] = x[start : start + rows].transpose(0, 1)
    return tile


def _chunk_gate(gc, start, rows):
    """Gate tile whose invalid rows repeat the last valid cumulative gate."""
    tile = _chunk_tile(gc, start, rows)
    tile[:, rows:] = tile[:, rows - 1 : rows]
    return tile


def kda_chunk_fwd_reference(q, k, v, g, beta, scale, h0, cu_seqlens):
    """Chunked KDA forward producing the residuals the backward consumes.

    Returns ``(gc, aqk, akk, h, final_state)`` with ``gc`` ``[N, H, K]``,
    ``aqk``/``akk`` ``[N, H, BT]`` (row ``t`` holds that token's chunk-local
    row), and ``h`` ``[C, H, K, V]`` chunk-start states in global chunk order.
    """
    num_tokens, num_heads, _ = q.shape
    gc = torch.empty_like(g)
    aqk = q.new_zeros(num_tokens, num_heads, BT)
    akk = q.new_zeros(num_tokens, num_heads, BT)
    states, finals = [], []
    eye = torch.eye(BT, dtype=q.dtype, device=q.device)
    for n, start, rows, _ in chunk_ranges(cu_seqlens):
        if start == int(cu_seqlens[n]):
            state = h0[n]
        local = g[start : start + rows].cumsum(0) * LOG2E
        gc[start : start + rows] = local
        gq = _chunk_gate(gc, start, rows)
        qt, kt, vt = (_chunk_tile(x, start, rows) for x in (q, k, v))
        bt = _chunk_tile(beta, start, rows)
        decay = torch.exp2(gq[:, :, None, :] - gq[:, None, :, :])
        causal = torch.ones(BT, BT, dtype=torch.bool, device=q.device).tril()
        valid = torch.zeros(BT, dtype=torch.bool, device=q.device)
        valid[:rows] = True
        pair_valid = valid[:, None] & valid[None, :]
        qk = torch.einsum("hik,hjk,hijk->hij", qt, kt, decay.where(causal[..., None], 0))
        kk = torch.einsum("hik,hjk,hijk->hij", kt, kt, decay.where(causal[..., None], 0))
        a_qk = (scale * qk).where(causal & pair_valid, 0)
        lower = (bt[:, :, None] * kk).where(causal.tril(-1) & pair_valid, 0)
        a_kk = torch.linalg.solve_triangular(eye + lower, eye.expand_as(lower), upper=False)
        a_kk = a_kk.where(pair_valid, 0)
        aqk[start : start + rows] = a_qk[:, :rows].transpose(0, 1)
        akk[start : start + rows] = a_kk[:, :rows].transpose(0, 1)
        states.append(state)
        gamma = torch.exp2(gq)
        w = a_kk @ (kt * bt[..., None] * gamma)
        u = a_kk @ (vt * bt[..., None])
        v_new = u - w @ state
        last = gq[:, BT - 1]
        k_last = kt * torch.exp2(last[:, None, :] - gq)
        state = torch.exp2(last)[..., None] * state + k_last.transpose(1, 2) @ v_new
        if start + rows == int(cu_seqlens[n + 1]):
            finals.append(state)
    return gc, aqk, akk, torch.stack(states), torch.stack(finals)


# A diagonal block takes the tensor-core path for a channel only while the
# block's cumulative log2 gate span stays below this bound.  The epilogue factor
# is then below 2^96 and the anchored operand above 2^-96 times its input, both
# inside the FP32/BF16 normal range for typical operand magnitudes.
DIAG_SPAN_LIMIT = 112.0


def _round(x, dtype):
    """Round to ``dtype``; ``"bf16x2"`` keeps a BF16 high/low pair (two MMAs)."""
    if dtype is None:
        return x
    if dtype == "bf16x2":
        high = x.to(torch.bfloat16).to(x.dtype)
        return high + (x - high).to(torch.bfloat16).to(x.dtype)
    return x.to(dtype).to(x.dtype)


def _exact_diag(m, x, g_out, g_in, lower):
    decay = torch.exp2(g_out[:, :, None, :] - g_in[:, None, :, :])
    diag = m.tril()
    if lower:
        return torch.einsum("hij,hjk,hijk->hik", diag, x, decay.where(diag[..., None] != 0, 0))
    return torch.einsum("hij,hik,hijk->hjk", diag, x, decay.where(diag[..., None] != 0, 0))


def _lower_anchored(m, x, gq, operand_dtype, diag_mode="exact", square_dtype=None):
    """``out_i = sum_{j<=i} m_ij x_j 2^(G_i-G_j)`` with the kernel's anchors.

    Each 16-token column block ``J`` uses the anchor ``c_J`` at its last row.
    The operand ``x_j 2^(c_J-G_j)`` is at most one for any gate, and so is the
    epilogue factor ``2^(G_i-c_J)`` for rows after ``J``.  For rows inside
    ``J`` (the diagonal block) the factor can exceed one: ``diag_mode="exact"``
    evaluates that block directly, while ``"anchored"`` models the kernel, which
    keeps the anchored product for channels whose span is below
    ``DIAG_SPAN_LIMIT`` and recomputes the remaining channels exactly.
    """
    out = torch.zeros_like(x)
    for col in range(BT // BC):
        cols = slice(col * BC, (col + 1) * BC)
        anchor = gq[:, (col + 1) * BC - 1]
        operand = _round(x[:, cols] * torch.exp2(anchor[:, None, :] - gq[:, cols]), operand_dtype)
        rows = slice((col + 1) * BC, BT)
        partial = _round(m[:, rows, cols], square_dtype) @ operand
        out[:, rows] += torch.exp2(gq[:, rows] - anchor[:, None, :]) * partial
        exact = _exact_diag(m[:, cols, cols], x[:, cols], gq[:, cols], gq[:, cols], True)
        if diag_mode == "anchored":
            fast = _round(m[:, cols, cols].tril(), square_dtype) @ operand
            factor = torch.exp2((gq[:, cols] - anchor[:, None, :]).float()).to(x.dtype)
            safe = (gq[:, col * BC] - anchor < DIAG_SPAN_LIMIT)[:, None, :]
            exact = torch.where(safe, factor * fast, exact)
        out[:, cols] += exact
    return out


def _upper_anchored(m, x, gq, operand_dtype, diag_mode="exact", square_dtype=None):
    """``out_j = sum_{i>=j} m_ij x_i 2^(G_i-G_j)`` with the kernel's anchors.

    Each row block ``I`` uses the anchor ``c_I`` at its first row; both
    ``x_i 2^(G_i-c_I)`` and ``2^(c_I-G_j)`` (``j`` before ``I``) are at most
    one.  The diagonal block follows ``diag_mode`` as in ``_lower_anchored``.
    """
    out = torch.zeros_like(x)
    for row in range(BT // BC):
        rows = slice(row * BC, (row + 1) * BC)
        anchor = gq[:, row * BC]
        operand = _round(x[:, rows] * torch.exp2(gq[:, rows] - anchor[:, None, :]), operand_dtype)
        cols = slice(0, row * BC)
        partial = _round(m[:, rows, cols], square_dtype).transpose(1, 2) @ operand
        out[:, cols] += torch.exp2(anchor[:, None, :] - gq[:, cols]) * partial
        exact = _exact_diag(m[:, rows, rows], x[:, rows], gq[:, rows], gq[:, rows], False)
        if diag_mode == "anchored":
            fast = _round(m[:, rows, rows].tril(), square_dtype).transpose(1, 2) @ operand
            factor = torch.exp2((anchor[:, None, :] - gq[:, rows]).float()).to(x.dtype)
            safe = (anchor - gq[:, (row + 1) * BC - 1] < DIAG_SPAN_LIMIT)[:, None, :]
            exact = torch.where(safe, factor * fast, exact)
        out[:, rows] += exact
    return out


def kda_fused_bwd_reference(
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
    *,
    operand_dtype=None,
    intra_operand_dtype=None,
    intra_square_dtype=None,
    diag_mode="exact",
    trace=None,
):
    """Per-chunk specification of the fused KDA backward.

    Returns ``(dq, dk, dv, dg, dbeta, dh0)``; see the module docstring for
    units.  Every step below names the kernel operation it models.  When
    ``trace`` is a dict, it receives ``{global_chunk: {name: [H, ...] tile}}``
    intermediates for stage-by-stage comparison with kernel debug dumps.
    """
    dq, dk, dv, dg = (torch.zeros_like(x) for x in (q, k, v, q))
    dbeta = torch.zeros_like(beta)
    dh0 = torch.zeros_like(dht)
    chunks = list(chunk_ranges(cu_seqlens))
    causal = torch.ones(BT, BT, dtype=torch.bool, device=q.device).tril()
    for n, start, rows, chunk in reversed(chunks):
        if start + rows == int(cu_seqlens[n + 1]):
            d_state = dht[n].clone()
        valid = torch.zeros(BT, dtype=torch.bool, device=q.device)
        valid[:rows] = True
        pair_valid = valid[:, None] & valid[None, :]
        gq = _chunk_gate(gc, start, rows)
        qt, kt, vt, dot = (_chunk_tile(x, start, rows) for x in (q, k, v, do))
        bt = _chunk_tile(beta, start, rows)
        a_qk = _chunk_tile(aqk, start, rows).where(causal & pair_valid, 0)
        a_kk = _chunk_tile(akk, start, rows).where(causal & pair_valid, 0)
        state = h[chunk]
        gamma = torch.exp2(gq)
        last = gq[:, BT - 1]
        gamma_last = torch.exp2(last)

        # Gate-scaled K-side operand tiles; every exponent is at most zero.
        k_gamma = _round(kt * gamma, operand_dtype)
        k_last = _round(kt * torch.exp2(last[:, None, :] - gq) * valid[:, None], operand_dtype)
        q_gamma = _round(scale * qt * gamma, operand_dtype)

        # u_state / W / vprime: v_new = Akk @ (beta * (v - (k*Gamma) @ S)).
        x_state = k_gamma @ state
        y = _round(bt[..., None] * (vt - x_state), operand_dtype)
        v_new = _round(a_kk @ y, operand_dtype)

        # dvprime_state + dvprime_aqk: dv_new = (k*2^(GL-G)) @ dS + Aqk^T @ dO.
        dv_new = k_last @ d_state + a_qk.transpose(1, 2) @ dot
        # dv: D = Akk^T @ dv_new; the input value gradient is beta * D.
        d_akk_t = a_kk.transpose(1, 2) @ _round(dv_new, operand_dtype)
        dv_tile = bt[..., None] * d_akk_t
        beta_d = _round(dv_tile, operand_dtype)

        # dAqk: scale * dO @ v_new^T on the causal, valid triangle.
        d_aqk = (scale * dot @ v_new.transpose(1, 2)).where(causal & pair_valid, 0)
        # dag + inverse backward: dL = -Akk^T (dv_new Y^T) Akk^T, strictly lower.
        d_akk = (_round(dv_new, operand_dtype) @ y.transpose(1, 2)).where(
            causal.tril(-1) & pair_valid, 0
        )
        d_lower = -(a_kk.transpose(1, 2) @ _round(d_akk, operand_dtype) @ a_kk.transpose(1, 2))
        d_lower = d_lower.where(causal.tril(-1) & pair_valid, 0)

        # State-path token gradients.
        dq_state = gamma * (scale * dot @ state.transpose(1, 2))
        dk_state = (
            torch.exp2(last[:, None, :] - gq) * (v_new @ d_state.transpose(1, 2)) * valid[:, None]
        )
        dk_w = -gamma * (beta_d @ state.transpose(1, 2))
        dbeta_tile = (d_akk_t * (vt - x_state)).sum(-1)

        # Intra-chunk gated score gradients.
        intra = (intra_operand_dtype, diag_mode, intra_square_dtype)
        dq_intra = _lower_anchored(d_aqk, kt, gq, *intra)
        k_row = _lower_anchored(d_lower, kt, gq, *intra)
        dk_col = _upper_anchored(d_aqk, qt, gq, *intra)
        dk_col += _upper_anchored(d_lower, bt[..., None] * kt, gq, *intra)
        dk_row = bt[..., None] * k_row
        dbeta_tile += (kt * k_row).sum(-1)

        dq_tile = dq_state + dq_intra
        dk_tile = dk_state + dk_w + dk_row + dk_col
        dg_tile = qt * dq_tile + kt * (dk_w + dk_row - dk_col - dk_state)
        dg_last = (kt * dk_state).sum(1) + gamma_last * (state * d_state).sum(-1)
        dg_tile[:, rows - 1] += dg_last
        dg_tile = dg_tile * valid[:, None]
        dg_tile = dg_tile.flip(1).cumsum(1).flip(1)

        if trace is not None:
            trace[chunk] = {
                "d_state_next": d_state,
                "x_state": x_state,
                "dv_new": dv_new,
                "v_new": v_new,
                "d": d_akk_t,
                "d_aqk": d_aqk,
                "d_akk": d_akk,
                "d_lower": d_lower,
                "dq_state": dq_state,
                "dk_state": dk_state,
                "dk_w": dk_w,
                "dq_intra": dq_intra,
                "k_row": k_row,
                "dk_col": dk_col,
            }
        span = slice(start, start + rows)
        dq[span] = dq_tile[:, :rows].transpose(0, 1)
        dk[span] = dk_tile[:, :rows].transpose(0, 1)
        dv[span] = dv_tile[:, :rows].transpose(0, 1)
        dg[span] = dg_tile[:, :rows].transpose(0, 1)
        dbeta[span] = dbeta_tile[:, :rows].transpose(0, 1)

        # dh_decay + dh_q + dh_k: dS_c = Gamma_L * dS + (q*Gamma*scale)^T dO - (k*Gamma)^T (beta*D).
        d_state = (
            gamma_last[..., None] * d_state
            + q_gamma.transpose(1, 2) @ dot
            - k_gamma.transpose(1, 2) @ beta_d
        )
        if start == int(cu_seqlens[n]):
            dh0[n] = d_state
    return dq, dk, dv, dg, dbeta, dh0
