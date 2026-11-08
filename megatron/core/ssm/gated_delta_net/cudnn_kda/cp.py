# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Chunkwise context parallelism for the cuDNN KDA backend.

Megatron's ``linear_cp_mode="chunkwise"`` splits one packed batch of documents into equal
contiguous token shares, one per CP rank, and lets FLA's ``cp_context`` (``FLACPContext``) carry
the rank-local ``cu_seqlens`` plus the bookkeeping of the two sequences that may cross a rank
boundary: ``is_first_rank`` / ``is_last_rank`` (does the first / last local sequence start / end
on this rank) and ``pre_num_ranks`` / ``post_num_ranks`` (how many preceding / following ranks
hold earlier / later parts of those two documents).  FLA's own exchange then works per rank with
one block of (state, transition) for the crossing sequence: summarise, ``all_gather``, fold the
predecessors' blocks into this rank's initial state (forward) and the successors' blocks into the
gradient of this rank's final state (backward).

cudnn-frontend 1.32.0 exposes exactly those two primitives for its KDA kernels:

* ``kimi_delta_attention_summary(k, v, g, beta, cu_seqlens)`` -> ``(H, M)`` per sequence: the
  final state under a zero initial state and the span transition, ``X_out = X_in @ M + H`` in the
  stored ``[HO, V, K]`` domain;
* ``kimi_delta_attention_summary_bwd(q, k, g, beta, dO, cu_seqlens, output_transition=True)``
  -> ``(G, T)``: the state-gradient recurrence ``dX_in = dX_out @ T + G``.

This module composes them with ``torch.distributed`` over ``cp_context.group`` and hands the
folded states to the native ``kda_fwd`` / ``kda_bwd`` (``initial_state`` / ``d_final_state``).
Only the boundary-crossing sequence is summarised (the token tensors are sliced to it and padded
to a power-of-two token bucket with neutral tokens, so cuDNN's shape-specialised engines compile
a handful of times instead of once per packed micro-batch) and one entry per rank is exchanged,
like FLA's preprocess (``cu_seqlens[-2:]`` forward, ``[:2]`` backward).  The boundary-state production is behind :class:`BoundaryStateProducer` so an
alternative producer can replace the summary +
all_gather + fold stage without touching the kernel calls.

Gradient conventions match FLA's CP path: the gate parameters receive this rank's contribution
only (the model's gradient reduction over the CP group sums them), ``initial_state`` is not
supported under CP, and no final state is returned.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol

import torch
import torch.distributed as dist

from . import ops as _ops

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - environment dependent
    triton = None
    tl = None
    HAVE_TRITON = False


try:
    import torch.distributed._symmetric_memory as _symm_mem  # noqa: F401

    _SYMM_MEM_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    _symm_mem = None
    _SYMM_MEM_AVAILABLE = False


def _all_gather_stacked(t: torch.Tensor, group) -> torch.Tensor:
    """``all_gather`` of one ``[HO, ...]`` entry per rank as a stacked ``[R, HO, ...]`` tensor.

    The fused fold kernel indexes the rank axis directly, so the gathered buffer is kept whole (one
    Python list and no per-rank view fewer).  Used by :class:`AllGatherBoundaryProducer`; the
    exchange is the NCCL ``all_gather``.
    """
    t = t.contiguous()
    out = torch.empty(
        (dist.get_world_size(group),) + tuple(t.shape), dtype=t.dtype, device=t.device
    )
    dist.all_gather_into_tensor(out, t, group=group)
    return out


def _host_cu_seqlens(cp_context) -> list[int]:
    """Rank-local ``cu_seqlens`` as host ints (``cu_seqlens_cpu`` when the context carries it)."""
    cpu = getattr(cp_context, "cu_seqlens_cpu", None)
    if cpu is not None:
        return [int(x) for x in cpu.tolist()]
    return [int(x) for x in cp_context.cu_seqlens.tolist()]


# cuDNN's frost engines specialise on the token count of their operands, so a summary over the
# boundary sequence's exact length would compile anew for every packed micro-batch (the 9L proxy
# never finished its first step).  The boundary slice is padded to a power-of-two bucket of at least
# ``_MIN_BUCKET`` tokens with tokens the recurrence ignores: k, v, q and dO zero, beta zero and a
# gate whose decay is exactly 1 -- a raw logit of -1e4 under the in-kernel bounded gate
# (``sigmoid(exp(a_log) * (-1e4 + dt_bias)) == 0`` in fp32) or a log-decay of 0 when the caller
# already applied the gate.  ``(H, M)`` and ``(G, T)`` of the padded sequence equal those of the
# unpadded one, and the number of distinct shapes the engines see drops to a handful.
_MIN_BUCKET = 1024
_NEUTRAL_GATE_LOGIT = -1.0e4


def _bucket_len(tokens: int) -> int:
    """Smallest power of two >= ``tokens`` and >= ``_MIN_BUCKET``."""
    return max(_MIN_BUCKET, 1 << max(tokens - 1, 0).bit_length())


if HAVE_TRITON:

    @triton.jit
    def _padded_copy_kernel(win_ptr, out_ptr, t_rows, inner, total, fill, BLOCK: tl.constexpr):
        """One masked write of a ``[T, inner]`` window into a ``[length, inner]`` output.

        Flat over the output: a lane whose row ``offs // inner`` is ``< t_rows`` copies the window
        element (same flat index -- both are ``inner``-strided and share the leading rows); the rest
        read ``fill`` through the masked load's ``other``.  One launch replaces ``new_full`` (the
        elementwise fill of every row) + the copy of the first ``T`` rows.
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < total
        in_window = (offs // inner) < t_rows
        val = tl.load(win_ptr + offs, mask=mask & in_window, other=fill)
        tl.store(out_ptr + offs, val, mask=mask)


def _padded_copy(window: torch.Tensor, length: int, fill: float = 0.0) -> torch.Tensor:
    """``window`` ``[T, ...]`` -> contiguous ``[length, ...]`` in ONE masked-write kernel."""
    t_rows = window.shape[0]
    rest = tuple(window.shape[1:])
    if t_rows >= length:
        return window.contiguous()
    inner = 1
    for dim in rest:
        inner *= dim
    out = torch.empty((length,) + rest, dtype=window.dtype, device=window.device)
    win = window.contiguous()
    total = length * inner
    block = 1024
    grid = (triton.cdiv(total, block),)
    _padded_copy_kernel[grid](win, out, t_rows, inner, total, float(fill), BLOCK=block)
    return out


def _pad_window(window: torch.Tensor, length: int, fill: float = 0.0) -> torch.Tensor:
    """``window`` ``[T, ...]`` -> a contiguous ``[length, ...]`` copy, trailing rows ``fill``.

    The bucket padding before each frost summary: one masked-write kernel on CUDA when Triton is
    available (``_padded_copy``), instead of a ``new_full`` fill plus a copy of the live rows -- the
    4 (forward) / 5 (backward) per-operand ``new_full`` + ``copy`` pairs collapse to one kernel each.
    Falls back to the ``new_full`` + copy on CPU or without Triton (numerically identical).
    """
    if HAVE_TRITON and window.is_cuda:
        return _padded_copy(window, length, fill)
    out = window.new_full((length,) + tuple(window.shape[1:]), fill)
    out[: window.shape[0]] = window
    return out


def _neutral_gate_fill(gate_kwargs: dict) -> float:
    """Gate value of a padding token: raw logit or log-decay, depending on the gate form."""
    return _NEUTRAL_GATE_LOGIT if gate_kwargs.get("safe_gate") else 0.0


def _pad_cu_nseq(cu: torch.Tensor, target_n: int) -> torch.Tensor:
    """Pad rank-local ``cu_seqlens`` ``[N+1]`` to ``[target_n+1]`` with zero-length tail segments.

    The cuDNN main-kernel plan cache keys on ``(total_tokens, num_sequences, ...)``.  Under chunkwise CP
    the per-rank ``total_tokens`` is static when ``pad_packed_seq_alignment=max``, but the number of local
    sequences varies every step as packed-document boundaries fall in different places inside this rank's
    slice, so the kernel recompiles per new local count -- the residual varlen-CP recompile storm.  Padding
    the local ``cu_seqlens`` (and the per-sequence state) to a fixed ``target_n`` with trailing zero-length
    segments makes the plan key static; zero-length segments process no tokens and add no output, so the
    result is unchanged.  ``target_n`` is the model's ``thd_max_packed_sequences`` (the global document-count
    cap), which upper-bounds the per-rank count: a contiguous token slice overlaps at most that many packed
    documents, so the pad never truncates.
    """
    n_local = cu.shape[0] - 1
    if n_local >= target_n:
        return cu
    tail = cu.new_full((target_n - n_local,), int(cu[-1].item()))
    return torch.cat([cu, tail])


def _zeros_boundary_state(n_local: int, ho: int, v: int, k: int, device) -> torch.Tensor:
    """An all-zero per-sequence boundary state ``[n_local, HO, V, K]`` (cuDNN fp32 V-major layout).

    The chunkwise-CP producer hands this to the main kernel on a rank that starts its own first document
    (forward ``initial_state``) or ends its own last document (backward ``d_final_state``): such a boundary
    is zero by definition -- the recurrence starts from zero / receives no successor gradient -- so returning
    it materialised instead of ``None`` is numerically identical.  It also keeps the cuDNN main-kernel plan
    key constant: ``make_fprop_cache_key`` / ``make_bprop_cache_key`` include the state-operand *dtype*, and
    ``None`` vs fp32 are different keys.  Under varlen THD packing the per-step document boundaries make
    ``is_first_rank`` / ``is_last_rank`` -- and so the ``None`` vs tensor choice -- toggle step to step on the
    same rank, which would otherwise re-plan the main kernel every step (the varlen-CP recompile storm).
    """
    return torch.zeros((n_local, ho, v, k), dtype=torch.float32, device=device)


_PREWARMED_FWD = False
_PREWARMED_BWD = False


def _buckets_up_to(total: int) -> list:
    """Power-of-two buckets ``[_MIN_BUCKET, ..., _bucket_len(total)]`` the summary can see."""
    top = _bucket_len(total)
    out, b = [], _MIN_BUCKET
    while b <= top:
        out.append(b)
        b *= 2
    return out


def _prewarm_summary_fwd(ops, k, v, g, beta, total, gate_kwargs, flags) -> None:
    """Compile the forward boundary-summary engine for every bucket once (neutral dummy inputs)."""
    global _PREWARMED_FWD
    if _PREWARMED_FWD:
        return
    _PREWARMED_FWD = True
    HK, K = k.shape[1], k.shape[2]
    HV, V = v.shape[1], v.shape[2]
    HO = g.shape[1]
    fill_g = _neutral_gate_fill(gate_kwargs)
    for length in _buckets_up_to(total):
        cu = torch.tensor([0, length], dtype=torch.int32, device=k.device)
        ops.kimi_delta_attention_summary(
            torch.zeros((length, HK, K), dtype=k.dtype, device=k.device),
            torch.zeros((length, HV, V), dtype=v.dtype, device=v.device),
            torch.full((length, HO, K), fill_g, dtype=g.dtype, device=g.device),
            torch.zeros((length, HO), dtype=beta.dtype, device=beta.device),
            cu,
            initial_state=None,
            use_qk_l2norm_in_kernel=flags["use_qk_l2norm_in_kernel"],
            use_beta_sigmoid_in_kernel=flags["use_beta_sigmoid_in_kernel"],
            allow_neg_eigval=flags["allow_neg_eigval"],
            **gate_kwargs,
        )


def _prewarm_summary_bwd(ops, q, k, g, beta, do, total, scale, gate_kwargs, flags) -> None:
    """Compile the backward boundary-summary engine for every bucket once (neutral dummy inputs)."""
    global _PREWARMED_BWD
    if _PREWARMED_BWD:
        return
    _PREWARMED_BWD = True
    H, K = q.shape[1], q.shape[2]
    HK = k.shape[1]
    HO = g.shape[1]
    V = do.shape[2]
    fill_g = _neutral_gate_fill(gate_kwargs)
    for length in _buckets_up_to(total):
        cu = torch.tensor([0, length], dtype=torch.int32, device=k.device)
        ops.kimi_delta_attention_summary_bwd(
            torch.zeros((length, H, K), dtype=q.dtype, device=q.device),
            torch.zeros((length, HK, K), dtype=k.dtype, device=k.device),
            torch.full((length, HO, K), fill_g, dtype=g.dtype, device=g.device),
            torch.zeros((length, HO), dtype=beta.dtype, device=beta.device),
            torch.zeros((length, HO, V), dtype=do.dtype, device=do.device),
            cu,
            scale=scale,
            d_final_state=None,
            output_transition=True,
            use_qk_l2norm_in_kernel=flags["use_qk_l2norm_in_kernel"],
            use_beta_sigmoid_in_kernel=flags["use_beta_sigmoid_in_kernel"],
            allow_neg_eigval=flags["allow_neg_eigval"],
            **gate_kwargs,
        )


def _pad_state_nseq(state: Optional[torch.Tensor], target_n: int) -> Optional[torch.Tensor]:
    """Pad a per-sequence state ``[N_local, ...]`` to ``[target_n, ...]`` with zero rows.

    The main-kernel plan-key state padding (shared by every cuDNN producer): one masked-write kernel
    on CUDA when Triton is available (``_padded_copy`` with a zero fill), instead of a ``new_zeros``
    fill of the whole buffer plus a copy of the live rows -- the same 2->1 reduction as the
    frost-summary ``_pad_window``, and the last per-step staging pair on the cuDNN CP path.  Falls
    back to ``new_zeros`` + copy on CPU or without Triton (numerically identical).
    """
    if state is None or state.shape[0] >= target_n:
        return state
    if HAVE_TRITON and state.is_cuda:
        return _padded_copy(state, target_n, 0.0)
    out = state.new_zeros((target_n,) + tuple(state.shape[1:]))
    out[: state.shape[0]] = state
    return out


if HAVE_TRITON:

    @triton.jit
    def _affine_fold_kernel(
        h_ptr,  # [R, HO, V, K] fp32 (gathered H / Htilde)
        m_ptr,  # [R, HO, K, K] fp32 (gathered M / transition)
        out_ptr,  # [HO, V, K] fp32 (the live boundary-state slot)
        start,  # first rank index to fold
        count,  # number of ranks to fold (this rank's actual chain length; runtime)
        step,  # +1 ascending (forward), -1 descending (backward)
        num_ranks,  # R = CP group size, to clamp the padded loads in bounds
        h_r_s,
        h_ho_s,
        h_v_s,
        m_r_s,
        m_ho_s,
        m_k_s,
        o_ho_s,
        o_v_s,
        V,
        HEAD_DIM: tl.constexpr,
        BLOCK_V: tl.constexpr,
        MAX_COUNT: tl.constexpr,  # uniform loop bound (R-1) -- SAME on every rank
    ):
        """One head's (and V-tile's) affine fold ``S = S @ M_r + H_r`` over the rank chain.

        Collapses the per-rank affine composition into a single launch: ``S`` starts at zero and the
        ``count`` ranks ``start, start + step, ...`` are folded in fp32 (``tl.dot`` with
        ``allow_tf32=False`` to match the fp64 oracle).  Forward folds predecessors ascending; the
        backward passes ``step = -1`` to fold successors descending -- the same affine composition,
        mirrored.  See ``boundary_fold_ref`` for the device-independent reference.

        RANK-SYMMETRIC: the loop runs a fixed ``MAX_COUNT`` (= R-1, the group-wide max chain length)
        on EVERY rank, not this rank's ``count``.  Iterations ``i >= count`` fold the identity affine
        (H=0, M=I), a numerical no-op, so the result is bit-identical to folding exactly ``count``
        ranks -- but every CP rank now does the same work and finishes the fold at the same time.
        Without this, ``count = pre/post_num_ranks`` grows with CP-rank index, so high-index ranks
        lagged and desynced the whole CP group, inflating the *downstream* collective (e.g. the MoE
        expert all-gather) which waits on the slowest rank.  Padded loads are clamped to [0, R-1].
        """
        pid_ho = tl.program_id(0)
        pid_v = tl.program_id(1)
        v_off = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        k_off = tl.arange(0, HEAD_DIM)
        v_mask = v_off < V
        ident = (k_off[:, None] == k_off[None, :]).to(tl.float32)  # [K, K] identity (no-op fold)
        state = tl.zeros([BLOCK_V, HEAD_DIM], dtype=tl.float32)
        for i in range(MAX_COUNT):
            do_fold = i < count
            r = start + i * step
            r_safe = tl.minimum(tl.maximum(r, 0), num_ranks - 1)  # keep padded loads in bounds
            h_base = h_ptr + r_safe * h_r_s + pid_ho * h_ho_s
            h_r = tl.load(
                h_base + v_off[:, None] * h_v_s + k_off[None, :], mask=v_mask[:, None], other=0.0
            )
            h_r = tl.where(do_fold, h_r, 0.0)  # padding iteration contributes zero H
            m_base = m_ptr + r_safe * m_r_s + pid_ho * m_ho_s
            m_r = tl.load(m_base + k_off[:, None] * m_k_s + k_off[None, :])
            m_r = tl.where(do_fold, m_r, ident)  # padding iteration multiplies by identity
            state = tl.dot(state, m_r, allow_tf32=False) + h_r
        tl.store(
            out_ptr + pid_ho * o_ho_s + v_off[:, None] * o_v_s + k_off[None, :],
            state,
            mask=v_mask[:, None],
        )


def _launch_affine_fold(
    h_stacked: torch.Tensor,
    m_stacked: torch.Tensor,
    out_slot: torch.Tensor,
    start: int,
    count: int,
    step: int,
) -> None:
    """Fold ``count`` ranks' summaries into ``out_slot`` ``[HO, V, K]`` with the Triton kernel."""
    if not HAVE_TRITON:
        raise ImportError(
            "cuDNN KDA boundary exchange needs Triton for the fused boundary fold."
        )
    num_ranks, ho, v, k = h_stacked.shape
    assert k == m_stacked.shape[-1] == m_stacked.shape[-2], "fold expects square [K, K] transitions"
    block_v = 64 if v % 64 == 0 else v
    grid = (ho, triton.cdiv(v, block_v))
    # Uniform loop bound across every CP rank (= R-1, the max chain length): rank-symmetric fold so
    # all ranks finish together and don't desync the group / inflate the downstream collective. The
    # extra iterations beyond this rank's ``count`` fold the identity (no-op), so the result is
    # unchanged. MAX_COUNT is a constexpr that is constant for a given CP group -> compiles once.
    max_count = max(num_ranks - 1, 1)
    _affine_fold_kernel[grid](
        h_stacked,
        m_stacked,
        out_slot,
        int(start),
        int(count),
        int(step),
        int(num_ranks),
        h_stacked.stride(0),
        h_stacked.stride(1),
        h_stacked.stride(2),
        m_stacked.stride(0),
        m_stacked.stride(1),
        m_stacked.stride(2),
        out_slot.stride(0),
        out_slot.stride(1),
        v,
        HEAD_DIM=k,
        BLOCK_V=block_v,
        MAX_COUNT=int(max_count),
    )


def _fused_affine_fold(
    h_stacked: torch.Tensor,
    m_stacked: torch.Tensor,
    *,
    n_local: int,
    live_slot: int,
    start: int,
    count: int,
    step: int,
) -> torch.Tensor:
    """Assemble the ``[N_local, HO, V, K]`` boundary state with the fused fold written in place.

    ``h_stacked`` / ``m_stacked`` are the gathered ``[R, HO, V, K]`` / ``[R, HO, K, K]`` fp32
    summaries.  The live slot (0 forward, ``N_local - 1`` backward) receives the fold of ranks
    ``start, start + step, ...`` directly; the others are zeros (only the crossing sequence's slot
    carries a non-zero boundary state).  No ``zeros_like`` seed and no device-to-device slot copy.
    """
    _, ho, v, k = h_stacked.shape
    device = h_stacked.device
    out = torch.empty((n_local, ho, v, k), dtype=torch.float32, device=device)
    if n_local > 1:
        # Only the live slot carries state; the rest are zero.
        if live_slot == 0:
            out[1:].zero_()
        else:
            out[:-1].zero_()
    _launch_affine_fold(h_stacked, m_stacked, out[live_slot], start, count, step)
    return out


class BoundaryStateProducer(Protocol):
    """Produces the two cross-rank boundary tensors of a chunkwise-CP step on this rank.

    Both methods are collective over ``cp_context.group``: every rank calls them in the same
    order (the autograd function guarantees that) and they return ``None`` on ranks that start /
    end their own document.  States are fp32 ``[N_local, HO, V, K]`` in cuDNN's V-major layout.
    """

    def initial_state(
        self, k, v, g, beta, cu, cu_list, cp_context, gate_kwargs, flags
    ) -> Optional[torch.Tensor]:
        """Initial state of this rank's first local sequence (slot 0 live), or ``None``."""
        ...

    def final_state_grad(
        self, q, k, g, beta, do, cu, cu_list, cp_context, scale, gate_kwargs, flags
    ) -> Optional[torch.Tensor]:
        """Gradient w.r.t. the final state of this rank's last local sequence, or ``None``."""
        ...


class _CudnnBoundaryBase:
    """Summary-only base: cuDNN's per-span frost ``(H, M)`` / ``(G, Tr)`` boundary summaries.

    Not user-selectable (absent from :data:`_PRODUCERS`); the two concrete producers below subclass
    it for the single-sourced frost summary (``_summarize_fwd`` / ``_summarize_bwd``).  They differ
    only in how the summaries are exchanged and folded; the fp32 affine fold itself is the shared
    :func:`_fused_affine_fold` / :func:`_symm_fold` on the module-level Triton kernel, with
    ``boundary_fold_ref`` the device-independent fp32/fp64 reference.
    """

    def _summarize_fwd(self, k, v, g, beta, cu_list, cp_context, gate_kwargs, flags):
        """This rank's forward crossing-span summary ``(H, M)`` plus ``(HO, V, K)``.

        The LAST local sequence is the only one that can continue onto a later rank: summarise it
        alone, padded to a shape bucket with neutral tokens (see ``_bucket_len``); a rank whose last
        sequence ends locally (``is_last_rank``) contributes zeros.  Shared by both producers so the
        summary stays single-sourced -- they differ only in how the gathered summaries are folded.
        """
        ops = _ops.ops()
        # Fold every boundary-summary shape bucket's compile into the first forward (self-guards to
        # once, see _prewarm_summary_fwd) so the steady steps are flat -- always on, no numerical effect.
        _prewarm_summary_fwd(ops, k, v, g, beta, cu_list[-1], gate_kwargs, flags)
        HO, V, K = v.shape[1], v.shape[2], k.shape[2]
        if not cp_context.is_last_rank:
            start = cu_list[-2]
            length = _bucket_len(cu_list[-1] - start)
            cu_tail = torch.tensor([0, length], dtype=torch.int32, device=k.device)
            h, m = ops.kimi_delta_attention_summary(
                _pad_window(k[start:], length),
                _pad_window(v[start:], length),
                _pad_window(g[start:], length, _neutral_gate_fill(gate_kwargs)),
                _pad_window(beta[start:], length),
                cu_tail,
                initial_state=None,
                use_qk_l2norm_in_kernel=flags["use_qk_l2norm_in_kernel"],
                use_beta_sigmoid_in_kernel=flags["use_beta_sigmoid_in_kernel"],
                allow_neg_eigval=flags["allow_neg_eigval"],
                **gate_kwargs,
            )
            h, m = h[0], m[0]
        else:
            h = torch.zeros((HO, V, K), dtype=torch.float32, device=k.device)
            m = torch.zeros((HO, K, K), dtype=torch.float32, device=k.device)
        return h, m, HO, V, K

    def _summarize_bwd(self, q, k, g, beta, do, cu_list, cp_context, scale, gate_kwargs, flags):
        """This rank's backward crossing-span summary ``(G, Tr)`` plus ``(HO, V, K)``.

        Mirror of :meth:`_summarize_fwd`: the FIRST local sequence is the only continuation from an
        earlier rank; ``is_first_rank`` contributes zeros.
        """
        ops = _ops.ops()
        # Fold every backward boundary-summary bucket's compile into the first backward (self-guards to
        # once, see _prewarm_summary_bwd) so the steady steps are flat -- always on, no numerical effect.
        _prewarm_summary_bwd(ops, q, k, g, beta, do, cu_list[-1], scale, gate_kwargs, flags)
        HO, V, K = do.shape[1], do.shape[2], k.shape[2]
        if not cp_context.is_first_rank:
            end = cu_list[1]
            length = _bucket_len(end)
            cu_head = torch.tensor([0, length], dtype=torch.int32, device=k.device)
            gsum, tr = ops.kimi_delta_attention_summary_bwd(
                _pad_window(q[:end], length),
                _pad_window(k[:end], length),
                _pad_window(g[:end], length, _neutral_gate_fill(gate_kwargs)),
                _pad_window(beta[:end], length),
                _pad_window(do[:end], length),
                cu_head,
                scale=scale,
                d_final_state=None,
                output_transition=True,
                use_qk_l2norm_in_kernel=flags["use_qk_l2norm_in_kernel"],
                use_beta_sigmoid_in_kernel=flags["use_beta_sigmoid_in_kernel"],
                allow_neg_eigval=flags["allow_neg_eigval"],
                **gate_kwargs,
            )
            gsum, tr = gsum[0], tr[0]
        else:
            gsum = torch.zeros((HO, V, K), dtype=torch.float32, device=k.device)
            tr = torch.zeros((HO, K, K), dtype=torch.float32, device=k.device)
        return gsum, tr, HO, V, K


class AllGatherBoundaryProducer(_CudnnBoundaryBase):
    """``all_gather`` producer: cuDNN frost summaries + NCCL ``all_gather`` + ONE fused fold.

    The portable boundary exchange: runs on any topology and is CUDA-graph-safe.  Keeps the base's
    fast per-span frost summary, gathers the per-rank summaries with an NCCL ``all_gather``, and
    assembles the boundary state with :func:`_fused_affine_fold` -- the fp32 affine fold and the
    slot write in one Triton launch.  See ``boundary_fold_ref`` for the device-independent
    fp32/fp64 reference of that fold.
    """

    def initial_state(self, k, v, g, beta, cu, cu_list, cp_context, gate_kwargs, flags):
        """Summarise + exchange (all_gather) + fused fp32 fold of the predecessors' blocks."""
        h, m, HO, V, K = self._summarize_fwd(k, v, g, beta, cu_list, cp_context, gate_kwargs, flags)
        group = cp_context.group
        rank = dist.get_rank(group)
        n_local = len(cu_list) - 1
        hs = _all_gather_stacked(h, group)  # [R, HO, V, K]
        ms = _all_gather_stacked(m, group)  # [R, HO, K, K]
        if cp_context.is_first_rank:
            return _zeros_boundary_state(n_local, HO, V, K, k.device)
        return _fused_affine_fold(
            hs,
            ms,
            n_local=n_local,
            live_slot=0,
            start=rank - cp_context.pre_num_ranks,  # ascending, document start first
            count=cp_context.pre_num_ranks,
            step=1,
        )

    def final_state_grad(
        self, q, k, g, beta, do, cu, cu_list, cp_context, scale, gate_kwargs, flags
    ):
        """Reverse-summarise + exchange (all_gather) + fused fp32 fold of the successors' blocks."""
        gsum, tr, HO, V, K = self._summarize_bwd(
            q, k, g, beta, do, cu_list, cp_context, scale, gate_kwargs, flags
        )
        group = cp_context.group
        rank = dist.get_rank(group)
        n_local = len(cu_list) - 1
        gs = _all_gather_stacked(gsum, group)  # [R, HO, V, K]
        ts = _all_gather_stacked(tr, group)  # [R, HO, K, K]
        if cp_context.is_last_rank:
            return _zeros_boundary_state(n_local, HO, V, K, k.device)
        return _fused_affine_fold(
            gs,
            ts,
            n_local=n_local,
            live_slot=n_local - 1,
            start=rank + cp_context.post_num_ranks,  # descending, document end first
            count=cp_context.post_num_ranks,
            step=-1,
        )


class _SymmHMExchange:
    """One rendezvoused fp32 symmetric buffer that exchanges the per-rank (H, M) over NVLink.

    Instead of an NCCL all-gather, each rank PUSHES its packed ``[HO, V+K, K]`` summary straight
    into its consumers' buffer slots and signals them; a consumer waits for its producers' signals
    and then folds its own (now-populated) buffer.  The payload is fp32 -- the symmetric tensor is
    fp32 natively (cuDNN's summary ``hm_sym`` is fp32).  Correctness of the write-before-read across ranks rides torch's
    own ``put_signal`` (release) / ``wait_signal`` (acquire), NOT ``dist.barrier`` (a CPU/NCCL
    collective that does not order a peer-visible NVLink write).  The buffer is parity-buffered on
    ``step % 2`` so a step's writes do not clobber the previous step's unread slots.  One instance
    per ``(group, HO, V, K)`` serves every KDA layer (its ``step`` advances in the collective order
    the autograd function guarantees); constructing it is a collective ``rendezvous``.
    """

    _FWD, _BWD = 0, 1

    def __init__(self, group, ho: int, v: int, k: int, device):
        import torch.distributed._symmetric_memory as symm_mem

        self.group = group
        self.R = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.ho, self.v, self.k, self.width = ho, v, k, v + k
        self._shape = (2, self.R, ho, self.width, k)
        self.buf = symm_mem.empty(self._shape, dtype=torch.float32, device=device)
        self.buf.zero_()
        self.handle = symm_mem.rendezvous(self.buf, group)
        torch.cuda.synchronize()
        dist.barrier(
            group=group
        )  # buffer zeroed everywhere before any peer write (construction only)
        # Separate forward/backward step counters so each direction's parity ALTERNATES across its own
        # consecutive calls (a shared counter would make forward always land on one parity -- the two
        # directions interleave -- and lose the cross-step protection).  Forward writes a consumer's
        # predecessor slots and backward its successor slots, which are disjoint, so the two directions
        # safely share one buffer even when their parities coincide.
        self.fwd_step = 0
        self.bwd_step = 0

    def _write_peer(self, consumer: int, par: int, h: torch.Tensor, m: torch.Tensor) -> None:
        """Write this rank's (H, M) into ``consumer``'s buffer slot ``[par, self.rank]`` (NVLink)."""
        slot = self.handle.get_buffer(consumer, self._shape, torch.float32)[par, self.rank]
        slot[:, : self.v, :] = h
        slot[:, self.v :, :] = m

    def exchange_forward(self, h: torch.Tensor, m: torch.Tensor) -> int:
        """Push (H, M) to every later rank + wait every earlier rank's push; return the parity.

        A producer signals, and a consumer waits on, EVERY earlier/later rank (one matched
        signal per pair), so no signal is left dangling; the fold then reads only its
        ``[rank-pre, rank)`` sub-range of the populated buffer.
        """
        self.fwd_step += 1
        par = self.fwd_step % 2
        for consumer in range(self.rank + 1, self.R):
            self._write_peer(consumer, par, h, m)
            self.handle.put_signal(consumer, channel=self._FWD)
        for producer in range(0, self.rank):
            self.handle.wait_signal(producer, channel=self._FWD)
        return par

    def exchange_backward(self, g: torch.Tensor, t: torch.Tensor) -> int:
        """Mirror of :meth:`exchange_forward`: push to every earlier rank, wait every later rank."""
        self.bwd_step += 1
        par = self.bwd_step % 2
        for consumer in range(0, self.rank):
            self._write_peer(consumer, par, g, t)
            self.handle.put_signal(consumer, channel=self._BWD)
        for producer in range(self.rank + 1, self.R):
            self.handle.wait_signal(producer, channel=self._BWD)
        return par


_SYMM_EXCHANGES: dict[tuple, _SymmHMExchange] = {}


def _get_symm_hm_exchange(group, ho: int, v: int, k: int, device) -> _SymmHMExchange:
    """The cached per-``(group, HO, V, K)`` symmetric exchange (``rendezvous`` runs once).

    Raises a clear, actionable RuntimeError when the symmetric-memory rendezvous cannot establish
    P2P across the CP group (there is no fallback -- see :class:`SymmMemBoundaryProducer`).  A failed
    rendezvous is not cached, so it surfaces rather than being silently swallowed.
    """
    key = (id(group), ho, v, k, getattr(device, "index", device))
    exch = _SYMM_EXCHANGES.get(key)
    if exch is None:
        try:
            exch = _SymmHMExchange(group, ho, v, k, device)
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "kda_cudnn_cp_boundary_exchange='symm_mem' could not establish a symmetric-memory "
                f"exchange across the CP group (P2P required): {exc}. "
                "Set kda_cudnn_cp_boundary_exchange='all_gather'."
            ) from exc
        _SYMM_EXCHANGES[key] = exch
    return exch


def _symm_fold(
    hm_par: torch.Tensor, v: int, *, n_local: int, live_slot: int, start: int, count: int, step: int
) -> torch.Tensor:
    """Fold the live boundary slot straight from the symm buffer's populated parity slab.

    ``hm_par`` is ``[R, HO, V+K, K]`` (this rank's buffer after the exchange).  Its ``H`` / ``M``
    halves are strided views (rows ``[0,V)`` and ``[V,V+K)``), handed to the SAME verified stage-2
    ``_affine_fold_kernel`` -- no gather, no staging copy, the fold reads the exchanged state in
    place.
    """
    _, ho, _width, k = hm_par.shape
    h_view = hm_par[:, :, :v, :]  # [R, HO, V, K] (strided)
    m_view = hm_par[:, :, v:, :]  # [R, HO, K, K] (strided)
    out = torch.empty((n_local, ho, v, k), dtype=torch.float32, device=hm_par.device)
    if n_local > 1:
        if live_slot == 0:
            out[1:].zero_()
        else:
            out[:-1].zero_()
    _launch_affine_fold(h_view, m_view, out[live_slot], start, count, step)
    return out


class SymmMemBoundaryProducer(_CudnnBoundaryBase):
    """``symm_mem`` producer (default): cuDNN frost summaries, a symmetric-memory exchange, fused fold.

    The boundary exchange over CUDA symmetric memory (point-to-point writes across NVLink): each rank
    pushes its (H, M) into its consumers' buffers over NVLink (:class:`_SymmHMExchange`), waits on its
    producers' ``put_signal`` releases, and folds the populated buffer in place with
    :func:`_symm_fold`.  Numerically identical to :class:`AllGatherBoundaryProducer` (same frost
    summary, same fp32 fold); the exchange's cross-rank memory ordering is pinned by the
    ``test_symm_mem_exchange_stress`` round-trip.

    The symm-mem path is not CUDA-graphable (the signals spin) and needs torch symmetric memory plus
    P2P across the CP group.  There is NO fallback -- every incompatibility is an explicit error:
    selecting ``symm_mem`` with CUDA graphs is rejected in ``TransformerConfig`` validation; a
    missing torch symmetric-memory module raises at construction; and a rendezvous that cannot
    establish P2P raises a clear RuntimeError (see :func:`_get_symm_hm_exchange`).  Use
    ``kda_cudnn_cp_boundary_exchange='all_gather'`` for the portable, CUDA-graph-safe path.
    """

    def __init__(self):
        if not _SYMM_MEM_AVAILABLE:
            raise RuntimeError(
                "kda_cudnn_cp_boundary_exchange='symm_mem' needs torch.distributed._symmetric_memory, "
                "which is not importable here; set kda_cudnn_cp_boundary_exchange='all_gather'."
            )

    def initial_state(self, k, v, g, beta, cu, cu_list, cp_context, gate_kwargs, flags):
        """Summarise + symmetric-memory exchange + fused fp32 fold of the predecessors' blocks."""
        # Defensive guard for the hybrid-bypass path that skips TransformerConfig's CUDA-graph
        # rejection: the symm-mem signals are not capturable, so raise cleanly instead of hanging.
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "cuDNN KDA symm_mem boundary exchange is not CUDA-graphable; set "
                "kda_cudnn_cp_boundary_exchange='all_gather' or disable CUDA graphs."
            )
        h, m, HO, V, K = self._summarize_fwd(k, v, g, beta, cu_list, cp_context, gate_kwargs, flags)
        n_local = len(cu_list) - 1
        exch = _get_symm_hm_exchange(cp_context.group, HO, V, K, k.device)
        par = exch.exchange_forward(h, m)
        if cp_context.is_first_rank:
            return _zeros_boundary_state(n_local, HO, V, K, k.device)
        rank = exch.rank
        return _symm_fold(
            exch.buf[par],
            V,
            n_local=n_local,
            live_slot=0,
            start=rank - cp_context.pre_num_ranks,  # ascending, document start first
            count=cp_context.pre_num_ranks,
            step=1,
        )

    def final_state_grad(
        self, q, k, g, beta, do, cu, cu_list, cp_context, scale, gate_kwargs, flags
    ):
        """Reverse-summarise + symmetric-memory exchange + fused fp32 fold of the successors' blocks."""
        # Defensive guard (mirror of initial_state) for the hybrid-bypass path: symm-mem is not
        # CUDA-graph capturable, so raise cleanly instead of hanging under capture.
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "cuDNN KDA symm_mem boundary exchange is not CUDA-graphable; set "
                "kda_cudnn_cp_boundary_exchange='all_gather' or disable CUDA graphs."
            )
        gsum, tr, HO, V, K = self._summarize_bwd(
            q, k, g, beta, do, cu_list, cp_context, scale, gate_kwargs, flags
        )
        n_local = len(cu_list) - 1
        exch = _get_symm_hm_exchange(cp_context.group, HO, V, K, k.device)
        par = exch.exchange_backward(gsum, tr)
        if cp_context.is_last_rank:
            return _zeros_boundary_state(n_local, HO, V, K, k.device)
        rank = exch.rank
        return _symm_fold(
            exch.buf[par],
            V,
            n_local=n_local,
            live_slot=n_local - 1,
            start=rank + cp_context.post_num_ranks,  # descending, document end first
            count=cp_context.post_num_ranks,
            step=-1,
        )


# The two user-facing boundary-exchange modes (mirrors the config Literal in transformer_config.py
# and the dispatch in kda.py): "symm_mem" (symmetric-memory / NVLink exchange, default) and
# "all_gather" (NCCL all-gather, portable and CUDA-graph safe).  "symm_mem" has no fallback -- an
# incompatibility (CUDA graphs, missing symm-mem, or a rendezvous that cannot establish P2P) is an
# explicit error directing the user to "all_gather".
_PRODUCERS: dict[str, Any] = {
    "symm_mem": SymmMemBoundaryProducer,
    "all_gather": AllGatherBoundaryProducer,
}


def register_boundary_state_producer(name: str, factory) -> None:
    """Register an alternative :class:`BoundaryStateProducer` under ``name``."""
    _PRODUCERS[name] = factory


def get_boundary_state_producer(name: str) -> BoundaryStateProducer:
    """Instantiate the registered producer ``name``."""
    try:
        return _PRODUCERS[name]()
    except KeyError:
        raise ValueError(
            f"unknown cuDNN KDA boundary-state producer {name!r}; known: {sorted(_PRODUCERS)}"
        ) from None


class ChunkwiseCPKDAFunction(torch.autograd.Function):
    """cuDNN KDA forward/backward on this rank's THD shard with the CP boundary states folded in."""

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        g,
        beta,
        a_log,
        dt_bias,
        cu,
        cu_list,
        cp_context,
        scale,
        flags,
        gate_kwargs_static,
        producer,
        pad_nseq,
    ):
        """``kda_fwd`` of this rank's shard seeded with the folded boundary state."""
        ops = _ops.ops()
        gate_kwargs = dict(gate_kwargs_static)
        if a_log is not None:
            gate_kwargs["a_log"] = a_log
        if dt_bias is not None:
            gate_kwargs["dt_bias"] = dt_bias
        initial_state = producer.initial_state(
            k, v, g, beta, cu, cu_list, cp_context, gate_kwargs, flags
        )
        # Pin the main-kernel plan key's sequence count: pad the rank-local cu_seqlens and the
        # per-sequence initial state with zero-length tail segments (see _pad_cu_nseq).  ``pad_nseq``
        # is the model's ``thd_max_packed_sequences`` (None when unset -> no padding).
        target_n = pad_nseq
        cu_kernel = _pad_cu_nseq(cu, target_n) if target_n is not None else cu
        initial_state_kernel = (
            _pad_state_nseq(initial_state, target_n) if target_n is not None else initial_state
        )
        out, _, _ = ops.kda_fwd(
            q,
            k,
            v,
            g,
            beta,
            cu_kernel,
            float(scale),
            initial_state=initial_state_kernel,
            output_final_state=False,
            checkpoint_every_n_tokens=0,
            use_qk_l2norm_in_kernel=flags["use_qk_l2norm_in_kernel"],
            use_beta_sigmoid_in_kernel=flags["use_beta_sigmoid_in_kernel"],
            allow_neg_eigval=flags["allow_neg_eigval"],
            **gate_kwargs,
        )
        saved = [q, k, v, g, beta, cu]
        ctx.has_a_log, ctx.has_dt_bias, ctx.has_initial_state = (
            a_log is not None,
            dt_bias is not None,
            initial_state is not None,
        )
        # Only the first local sequence carries a non-zero initial state; keep that slot alone
        # (cloned, so the padded [N_local, ...] buffer is released) and re-expand in backward,
        # as FLA's CP path does (compress_h0 / expand_h0).
        saved_state = initial_state[:1].clone() if initial_state is not None else None
        saved += [t for t in (a_log, dt_bias, saved_state) if t is not None]
        ctx.save_for_backward(*saved)
        ctx.cu_list, ctx.cp_context, ctx.scale = cu_list, cp_context, float(scale)
        ctx.flags, ctx.gate_kwargs_static, ctx.producer = flags, gate_kwargs_static, producer
        ctx.pad_nseq = pad_nseq
        return out

    @staticmethod
    def backward(ctx, do):
        """``kda_bwd`` of this rank's shard with the folded boundary-state gradient."""
        ops = _ops.ops()
        saved = list(ctx.saved_tensors)
        q, k, v, g, beta, cu = saved[:6]
        rest = saved[6:]
        a_log = rest.pop(0) if ctx.has_a_log else None
        dt_bias = rest.pop(0) if ctx.has_dt_bias else None
        initial_state = None
        if ctx.has_initial_state:
            slot = rest.pop(0)
            initial_state = slot.new_zeros((len(ctx.cu_list) - 1,) + tuple(slot.shape[1:]))
            initial_state[:1] = slot
        gate_kwargs = dict(ctx.gate_kwargs_static)
        if a_log is not None:
            gate_kwargs["a_log"] = a_log
        if dt_bias is not None:
            gate_kwargs["dt_bias"] = dt_bias
        do = do.contiguous()
        d_final_state = ctx.producer.final_state_grad(
            q, k, g, beta, do, cu, ctx.cu_list, ctx.cp_context, ctx.scale, gate_kwargs, ctx.flags
        )
        # Pin the backward plan key's sequence count the same way as the forward.
        target_n = ctx.pad_nseq
        if target_n is not None:
            cu = _pad_cu_nseq(cu, target_n)
            initial_state = _pad_state_nseq(initial_state, target_n)
            d_final_state = _pad_state_nseq(d_final_state, target_n)
        dq, dk, dv, dg, dbeta, _d_initial_state, d_a_log, d_dt_bias = ops.kda_bwd(
            do,
            q,
            k,
            v,
            g,
            beta,
            cu,
            ctx.scale,
            initial_state=initial_state,
            d_final_state=d_final_state,
            state_checkpoints=None,
            checkpoint_every_n_tokens=0,
            use_qk_l2norm_in_kernel=ctx.flags["use_qk_l2norm_in_kernel"],
            use_beta_sigmoid_in_kernel=ctx.flags["use_beta_sigmoid_in_kernel"],
            allow_neg_eigval=ctx.flags["allow_neg_eigval"],
            **gate_kwargs,
        )
        return (
            dq,
            dk,
            dv,
            dg,
            dbeta,
            d_a_log if ctx.has_a_log else None,
            d_dt_bias if ctx.has_dt_bias else None,
            None,  # cu
            None,  # cu_list
            None,  # cp_context
            None,  # scale
            None,  # flags
            None,  # gate_kwargs_static
            None,  # producer
            None,  # pad_nseq
        )


def chunk_kda_cudnn_cp(
    q,
    k,
    v,
    g,
    beta,
    a_log,
    dt_bias,
    cu,
    cp_context,
    *,
    scale: float,
    flags: dict,
    gate_kwargs: dict,
    cp_boundary_exchange: str = "symm_mem",
    pad_nseq: Optional[int] = None,
) -> torch.Tensor:
    """THD chunkwise-CP KDA on cuDNN kernels.

    Operands are already in cuDNN's conventions (see ``ops``).  ``cp_boundary_exchange`` selects the
    boundary-state producer (``"symm_mem"`` default / ``"all_gather"``; see :data:`_PRODUCERS`).
    ``pad_nseq`` (the model's ``thd_max_packed_sequences``) pins the rank-local sequence count so the
    cuDNN plan key is static.
    """
    cu_list = _host_cu_seqlens(cp_context)
    static = {key: value for key, value in gate_kwargs.items() if key not in ("a_log", "dt_bias")}
    return ChunkwiseCPKDAFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        a_log,
        dt_bias,
        cu,
        cu_list,
        cp_context,
        scale,
        flags,
        static,
        get_boundary_state_producer(cp_boundary_exchange),
        pad_nseq,
    )
