# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Numerical spec for the cuDNN-KDA chunkwise-CP boundary EXCHANGE + FOLD.

The default ``cp.CudnnSummaryProducer`` keeps cuDNN's fast per-span frost summary
(``kimi_delta_attention_summary`` -> ``(H, M)``) but assembles the cross-rank
boundary state with a stack of scattered PyTorch kernels: ``_all_gather_one``
twice (``contiguous`` + ``empty`` + ``all_gather`` staging), a ``torch.zeros_like``
seed, a Python ``for``-loop of ``matmul(state, M_r) + H_r``, and a
``torch.zeros`` + ``initial_state[0] = state`` d2d.  The
``yuzhongw/kda-prekda-cudnn-layout`` line replaces that tail with ONE fused
"exchange + fold" kernel that eats frost's ``(H, M)`` directly, does the
symmetric-memory exchange and the affine fold in-kernel, and writes the boundary
state out -- no glue, no staging, no per-rank launches.

This module is the authoritative, device-independent fp32/fp64 reference for the
MATH that kernel must reproduce.  It is *not* a fast path and imports only
``torch`` (no cutlass, no flash-linear-attention, no CUDA).  Two layers:

1. the single-span frost summary ``(H, M)`` from raw operands, derived from the
   naive token-by-token gated-delta-rule recurrence (the DEFINITION of the
   summary), cross-checked against the chunk-parallel WY form; and
2. the cross-rank fold -- the affine composition of per-rank spans -- which is
   exactly what ``cp.CudnnSummaryProducer`` does with its all-gather + matmul
   loop and what the fused kernel must do in one launch.

Conventions (match ``cp.py`` / ``ops.py``; THD, ``B == 1``, chunk size 64):

* raw operands per head: ``k`` ``[T, HV, K]`` L2-normalised keys, ``v``
  ``[T, HV, V]`` values, ``g`` ``[T, HV, K]`` per-key per-token log-decay
  (natural log; the decay applied at token ``t`` is ``exp(g_t)``), ``beta``
  ``[T, HV]`` post-sigmoid write strength.  ``K == V``.
* state domains.  The naive recurrence is written in the FLA ``[K, V]`` domain
  (``S``; the key index is the row).  cuDNN / ``cp.py`` store the state V-major
  as ``X = S^T`` in ``[V, K]`` (batched ``[HV, V, K]``), and the span is the
  affine map ``X_out = X_in @ M + H`` with ``M`` ``[HV, K, K]`` acting on the
  RIGHT and ``H`` ``[HV, V, K]`` -- the exact operands ``kimi_delta_attention_
  summary`` returns and ``CudnnSummaryProducer`` folds (``state @ M_r + H_r``).
  This module produces ``(H, M)`` in that cuDNN domain so the fold needs no
  transpose: the kernel consumes frost's ``(H, M)`` as-is.
* fold composition of span A (earlier) then span B (later):
  ``M_AB = M_A @ M_B`` and ``H_AB = H_A @ M_B + H_B`` in the cuDNN domain.  A
  consumer rank's forward initial state is the composition of all its predecessor
  ranks' spans, seeded from zero and walked document-start first -- identical to
  ``CudnnSummaryProducer.initial_state``'s ascending loop.  The backward is the
  mirror: successors, document-end first.
"""

from __future__ import annotations

import bisect
from typing import Optional

import torch

CHUNK_SIZE = 64


# --------------------------------------------------------------------------------------------
# Single-span frost summary (the DEFINITION, from the naive recurrence).
# --------------------------------------------------------------------------------------------


def naive_span_state_kv(
    k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, state_in: torch.Tensor
) -> torch.Tensor:
    """Naive token-by-token gated-delta-rule span scan in the ``[K, V]`` domain.

    Per token ``t`` (per head), with per-key decay ``a_t = exp(g_t)``:

        S_dec = diag(a_t) @ S            # apply the channel-wise decay
        delta = v_t - S_dec^T @ k_t      # prediction error (value minus read-out)
        S     = S_dec + beta_t * (k_t (x) delta)

    ``state_in`` is ``[HV, K, V]``; the returned state is ``[HV, K, V]``.  This is
    the executable definition the chunk-parallel WY form and cuDNN's frost kernel
    both implement; :func:`span_summary_cudnn` reads ``(H, M)`` out of it.
    """
    t_len, hv, kdim = k.shape
    vdim = v.shape[-1]
    state = state_in.clone()
    for t in range(t_len):
        decay = g[t].exp()[..., :, None]  # [HV, K, 1]
        s_dec = decay * state  # [HV, K, V]
        kt = k[t]  # [HV, K]
        read_out = (s_dec * kt[..., :, None]).sum(dim=-2)  # S_dec^T @ k_t -> [HV, V]
        delta = v[t] - read_out  # [HV, V]
        state = s_dec + beta[t][..., None, None] * kt[..., :, None] * delta[..., None, :]
    return state


def span_summary_cudnn(
    k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward boundary summary ``(H, M)`` in cuDNN's V-major domain.

    The span is affine in its input state, ``S_out = M_kv @ S_in + H_kv`` in the
    ``[K, V]`` domain, so ``H_kv`` is the span state under a zero initial state and
    ``M_kv`` is the linear part, recovered by scanning the value-free recurrence
    (``v = 0``) from ``S_in = I``.  Returned transposed to cuDNN's domain:
    ``H`` ``[HV, V, K]`` and ``M`` ``[HV, K, K]`` with ``X_out = X_in @ M + H`` --
    exactly ``kimi_delta_attention_summary``'s output contract.
    """
    t_len, hv, kdim = k.shape
    vdim = v.shape[-1]
    dtype, device = k.dtype, k.device
    zero_state = torch.zeros(hv, kdim, vdim, dtype=dtype, device=device)
    h_kv = naive_span_state_kv(k, v, g, beta, zero_state)  # [HV, K, V]
    eye = torch.eye(kdim, dtype=dtype, device=device).expand(hv, kdim, kdim).contiguous()
    zero_v = torch.zeros(t_len, hv, vdim, dtype=dtype, device=device)
    m_kv = naive_span_state_kv(k, zero_v[..., :kdim], g, beta, eye)  # [HV, K, K], v=0, S_in=I
    return h_kv.transpose(-1, -2).contiguous(), m_kv.transpose(-1, -2).contiguous()


# --------------------------------------------------------------------------------------------
# Chunk-parallel WY summary (the cross-check; mirrors the fused kernel's data flow).
# --------------------------------------------------------------------------------------------


def _chunk(x: torch.Tensor, nt: int, last: int) -> torch.Tensor:
    """``[T, HV, D]`` -> ``[HV, NT, BT, D]`` (BT == ``CHUNK_SIZE``)."""
    hv = x.shape[1]
    return x.permute(1, 0, 2).reshape(hv, nt, CHUNK_SIZE, last)


def _solve_tril(a: torch.Tensor) -> torch.Tensor:
    """``(I - stril(A))^{-1} - I`` for a strictly-lower ``A`` (forward substitution)."""
    bt = a.shape[-1]
    a = a.clone()
    for i in range(1, bt):
        a[..., i, :i] = a[..., i, :i].clone() + (
            a[..., i, :, None].clone() * a[..., :, :i].clone()
        ).sum(-2)
    return a


def span_summary_cudnn_wy(
    k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(H, M)`` via the chunk-parallel WY recurrence (cuDNN domain).

    Independent derivation of :func:`span_summary_cudnn` through the same WY
    intermediates the fused CuTe / frost kernels use (``w``, ``u``, ``kg`` and the
    per-chunk ``[K, V] | [K, K]`` state scan), so the two agreeing in fp64
    validates the recurrence itself, not just one transcription of it.  Requires
    ``T % CHUNK_SIZE == 0`` (the cuDNN CP path pads boundary windows to buckets).
    """
    t_len, hv, kdim = k.shape
    vdim = v.shape[-1]
    assert t_len % CHUNK_SIZE == 0, "WY cross-check expects a chunk-multiple span"
    nt = t_len // CHUNK_SIZE
    dtype, device = k.dtype, k.device
    kc, vc = _chunk(k, nt, kdim), _chunk(v, nt, vdim)
    bc = beta.permute(1, 0).reshape(hv, nt, CHUNK_SIZE)
    gc_all = _chunk(g, nt, kdim).cumsum(-2)  # chunk-local gate cumsum
    big_h = torch.zeros(hv, kdim, vdim, dtype=dtype, device=device)  # [K, V] state
    big_m = torch.eye(kdim, dtype=dtype, device=device).expand(hv, kdim, kdim).contiguous()
    eye_bt = torch.eye(CHUNK_SIZE, dtype=dtype, device=device)
    tri = torch.triu(torch.ones(CHUNK_SIZE, CHUNK_SIZE, dtype=torch.bool, device=device), 0)
    for n in range(nt):
        gc = gc_all[:, n]  # [HV, BT, K]
        kcn, vcn, bcn = kc[:, n], vc[:, n], bc[:, n]
        decayed = (kcn * gc.exp()) @ (kcn * (-gc).exp()).transpose(-1, -2)  # [HV, BT, BT]
        intra = -(decayed * bcn[..., None]).masked_fill(tri, 0)
        akk = (_solve_tril(intra) + eye_bt) * bcn[..., None, :]
        w = akk @ (gc.exp() * kcn)  # [HV, BT, K]
        u = akk @ vcn  # [HV, BT, V]
        kg = (gc[..., -1:, :] - gc).exp() * kcn  # [HV, BT, K]
        gamma = gc[:, -1].exp()[..., None]  # [HV, K, 1]
        xh = u - w @ big_h
        xm = -(w @ big_m)
        big_h = big_h * gamma + kg.transpose(-1, -2) @ xh
        big_m = big_m * gamma + kg.transpose(-1, -2) @ xm
    return big_h.transpose(-1, -2).contiguous(), big_m.transpose(-1, -2).contiguous()


# --------------------------------------------------------------------------------------------
# Cross-rank fold (the part the fused kernel replaces).
# --------------------------------------------------------------------------------------------


def compose_spans_fwd(
    h_list: list[torch.Tensor], m_list: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose spans ordered earliest-first into one ``(H, M)`` (cuDNN domain).

    ``state`` is seeded at zero and walked document-start first, exactly as
    ``CudnnSummaryProducer.initial_state``'s ascending ``state = state @ M_r + H_r``
    loop; the running ``M`` (``M_r0 @ M_r1 @ ...``) is tracked too so a partial
    composition can be chained further.
    """
    assert len(h_list) == len(m_list) and h_list, "need at least one span"
    h_acc = torch.zeros_like(h_list[0])
    m_acc = torch.eye(m_list[0].shape[-1], dtype=m_list[0].dtype, device=m_list[0].device)
    m_acc = m_acc.expand_as(m_list[0]).contiguous()
    for h_r, m_r in zip(h_list, m_list):
        h_acc = h_acc @ m_r + h_r
        m_acc = m_acc @ m_r
    return h_acc, m_acc


def fold_initial_state_fwd(
    h_list: list[torch.Tensor], m_list: list[torch.Tensor], rank: int, pre_num_ranks: int
) -> Optional[torch.Tensor]:
    """A rank's forward initial boundary state, or ``None`` on ``pre_num_ranks == 0``.

    ``h_list`` / ``m_list`` are every rank's span summary (as ``all_gather`` makes
    them available).  The live predecessors are ranks ``[rank - pre, rank)``; the
    fold over them is the boundary state entering this rank's first local sequence
    (``[HV, V, K]``).  Matches ``CudnnSummaryProducer.initial_state`` term for term.
    """
    if pre_num_ranks <= 0:
        return None
    sel_h = [h_list[r] for r in range(rank - pre_num_ranks, rank)]
    sel_m = [m_list[r] for r in range(rank - pre_num_ranks, rank)]
    return compose_spans_fwd(sel_h, sel_m)[0]


def fold_final_state_grad_bwd(
    g_list: list[torch.Tensor], t_list: list[torch.Tensor], rank: int, post_num_ranks: int
) -> Optional[torch.Tensor]:
    """A rank's backward final-state gradient, or ``None`` on ``post_num_ranks == 0``.

    Mirror of :func:`fold_initial_state_fwd`: the backward summary is ``(G, Tr)``
    with ``dX_in = dX_out @ Tr + G``, the live successors are ranks
    ``(rank, rank + post]`` and the fold walks them document-end first (descending),
    exactly as ``CudnnSummaryProducer.final_state_grad``:

        grad = 0 ; for r in [rank+post .. rank+1]: grad = grad @ Tr_r + G_r

    NOTE: the fold COMPOSITION here is the proven mirror of the forward (same
    affine-semigroup law, validated by :func:`_selftest`).  The raw ->``(G, Tr)``
    backward summary itself is defined by cuDNN's ``kimi_delta_attention_summary_bwd``;
    its exact V-major convention is derived in the design note and MUST be validated
    against FLA's backward producer on a Blackwell GPU before it is wired in (the
    forward summary above is fully derived and CPU-verified; the backward summary is
    deliberately left to the GPU loop, as Lebron's eat-raw oracle leaves its bwd).
    """
    if post_num_ranks <= 0:
        return None
    grad = torch.zeros_like(g_list[rank])
    for r in range(rank + post_num_ranks, rank, -1):
        grad = grad @ t_list[r] + g_list[r]
    return grad


def rank_chains(global_cu_cpu: list[int], world_size: int) -> Optional[list[tuple[int, int]]]:
    """``(pre_num_ranks, post_num_ranks)`` for every CP rank from the GLOBAL offsets.

    Mirror of ``cute_preprocess.backend._rank_chains``: with a contiguous equal
    split, rank ``r`` owns tokens ``[part*r, part*(r+1))``; ``pre`` is how many
    earlier ranks share ``r``'s first document, ``post`` how many later ranks share
    its last.  Returns ``None`` when the split is not well formed.
    """
    total = global_cu_cpu[-1]
    if world_size <= 0 or total % world_size:
        return None
    part = total // world_size
    if part <= 0:
        return None
    ends, starts = global_cu_cpu[1:], global_cu_cpu[:-1]
    chains = []
    for r in range(world_size):
        lo, hi = part * r, part * r + part
        first = bisect.bisect_right(ends, lo)
        last = bisect.bisect_left(starts, hi)
        pre = r - global_cu_cpu[first] // part
        post = (global_cu_cpu[last] - 1) // part - r
        chains.append((pre, post))
    return chains


# --------------------------------------------------------------------------------------------
# Self-test (CPU, fp64): recurrence == WY, fold == full recurrence.
# --------------------------------------------------------------------------------------------


def _random_operands(t_len, hv, kdim, *, seed, dtype=torch.float64, device="cpu"):
    """L2-normalised keys, values, a small negative log-decay and post-sigmoid beta."""
    gen = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, dtype=dtype, device=device)

    k = torch.nn.functional.normalize(randn(t_len, hv, kdim), dim=-1)
    v = randn(t_len, hv, kdim)
    g = -torch.nn.functional.softplus(randn(t_len, hv, kdim)) * 0.05  # small negative log-decay
    beta = torch.sigmoid(randn(t_len, hv))
    return k, v, g, beta


def _selftest() -> None:
    """Validate the recurrence, the summary and the fold composition in fp64."""
    torch.manual_seed(0)
    hv, kdim = 3, 128

    # 1. naive summary == chunk-parallel WY summary (both cuDNN domain).
    k, v, g, beta = _random_operands(128, hv, kdim, seed=1)
    h_naive, m_naive = span_summary_cudnn(k, v, g, beta)
    h_wy, m_wy = span_summary_cudnn_wy(k, v, g, beta)
    assert torch.allclose(h_naive, h_wy, atol=1e-9, rtol=1e-6), "H: naive vs WY disagree"
    assert torch.allclose(m_naive, m_wy, atol=1e-9, rtol=1e-6), "M: naive vs WY disagree"

    # 2. semigroup: summary(A ++ B) == compose(summary(A), summary(B)).
    ka, va, ga, ba = _random_operands(64, hv, kdim, seed=2)
    kb, vb, gb, bb = _random_operands(64, hv, kdim, seed=3)
    cat = lambda a, b: torch.cat([a, b], dim=0)
    h_cat, m_cat = span_summary_cudnn(cat(ka, kb), cat(va, vb), cat(ga, gb), cat(ba, bb))
    h_comp, m_comp = compose_spans_fwd(
        [span_summary_cudnn(ka, va, ga, ba)[0], span_summary_cudnn(kb, vb, gb, bb)[0]],
        [span_summary_cudnn(ka, va, ga, ba)[1], span_summary_cudnn(kb, vb, gb, bb)[1]],
    )
    assert torch.allclose(h_cat, h_comp, atol=1e-9, rtol=1e-6), "fold H breaks the semigroup"
    assert torch.allclose(m_cat, m_comp, atol=1e-9, rtol=1e-6), "fold M breaks the semigroup"

    # 3. end-to-end CP emulation: a single global sequence split into R equal shards;
    #    each rank's fold over its predecessors must equal the naive recurrence over the
    #    true prefix (the single-sequence chain `chains = [(r, R-r-1)]`).
    world_size, part = 4, 96
    kg_, vg_, gg_, bg_ = _random_operands(world_size * part, hv, kdim, seed=4)
    shards = [
        (
            kg_[r * part : (r + 1) * part],
            vg_[r * part : (r + 1) * part],
            gg_[r * part : (r + 1) * part],
            bg_[r * part : (r + 1) * part],
        )
        for r in range(world_size)
    ]
    summaries = [span_summary_cudnn(*s) for s in shards]
    h_all = [h for h, _ in summaries]
    m_all = [m for _, m in summaries]
    zero_state = torch.zeros(hv, kdim, kdim, dtype=torch.float64)
    for rank in range(world_size):
        folded = fold_initial_state_fwd(h_all, m_all, rank, pre_num_ranks=rank)
        prefix_state = zero_state  # [HV, K, V]
        for r in range(rank):
            prefix_state = naive_span_state_kv(*shards[r], prefix_state)
        truth = prefix_state.transpose(-1, -2).contiguous() if rank > 0 else None  # cuDNN domain
        if rank == 0:
            assert folded is None, "rank 0 must have no predecessor state"
        else:
            assert torch.allclose(folded, truth, atol=1e-9, rtol=1e-6), f"rank {rank} fold != truth"

    # 4. chain topology derivation matches the single-sequence expectation.
    chains = rank_chains([0, world_size * part], world_size)
    assert chains == [(r, world_size - r - 1) for r in range(world_size)], "chain topology wrong"
    print("boundary_fold_ref self-test PASSED (fp64: recurrence==WY, fold==truth)")


if __name__ == "__main__":
    _selftest()
