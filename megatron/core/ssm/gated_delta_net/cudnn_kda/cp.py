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
alternative producer (for example a fused CuTe DSL preprocess) can replace the summary +
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


def _all_gather_one(t: torch.Tensor, group) -> list[torch.Tensor]:
    """``all_gather`` of one ``[HO, ...]`` entry per rank.

    The entry is the crossing sequence's summary, or zeros on ranks without one.
    """
    t = t.contiguous()
    out = torch.empty(
        (dist.get_world_size(group),) + tuple(t.shape), dtype=t.dtype, device=t.device
    )
    dist.all_gather_into_tensor(out, t, group=group)
    return list(out.unbind(0))


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


def _pad_window(window: torch.Tensor, length: int, fill: float = 0.0) -> torch.Tensor:
    """``window`` ``[T, ...]`` -> a contiguous ``[length, ...]`` copy, trailing rows ``fill``."""
    out = window.new_full((length,) + tuple(window.shape[1:]), fill)
    out[: window.shape[0]] = window
    return out


def _neutral_gate_fill(gate_kwargs: dict) -> float:
    """Gate value of a padding token: raw logit or log-decay, depending on the gate form."""
    return _NEUTRAL_GATE_LOGIT if gate_kwargs.get("safe_gate") else 0.0


def _cp_pad_nseq_target() -> Optional[int]:
    """Fixed rank-local sequence count to pad to, from ``MCORE_KDA_CP_PAD_NSEQ`` (else ``None``).

    The cuDNN main-kernel plan cache keys on ``(total_tokens, num_sequences, ...)``.  Under chunkwise
    CP the per-rank ``total_tokens`` is static when ``pad_packed_seq_alignment=max``, but the number of
    local sequences varies every step as packed-document boundaries fall in different places inside this
    rank's slice (``thd_max_packed_sequences`` pins only the *global* count), so the forward/backward
    kernels recompile per new local count -- the residual varlen-CP recompile storm.  Setting this env to
    a fixed count >= the per-rank maximum pads the local ``cu_seqlens`` (and the per-sequence state) with
    trailing zero-length segments so the plan key is static.  Zero-length segments process no tokens and
    add no output, so the result is unchanged.
    """
    import os

    raw = os.environ.get("MCORE_KDA_CP_PAD_NSEQ")
    if not raw:
        return None
    value = int(raw)
    return value if value > 0 else None


def _pad_cu_nseq(cu: torch.Tensor, target_n: int) -> torch.Tensor:
    """Pad rank-local ``cu_seqlens`` ``[N+1]`` to ``[target_n+1]`` with zero-length tail segments."""
    n_local = cu.shape[0] - 1
    if n_local >= target_n:
        return cu
    tail = cu.new_full((target_n - n_local,), int(cu[-1].item()))
    return torch.cat([cu, tail])


def _cp_prewarm_enabled() -> bool:
    """Whether to pre-compile all boundary-summary shape buckets up front (``MCORE_KDA_CP_PREWARM``).

    The dominant chunkwise-CP recompile cost is the boundary summary (forward and backward): it is
    bucketed to a power-of-two token count (see ``_bucket_len``), but a document of a new bucketed
    length can first reach this rank's boundary at any step, and across the CP/DP ranks that keeps a
    fresh bucket compiling on most steps, so the storm spreads over the whole run even though each
    rank compiles only a handful of engines.  Pre-warming compiles every bucket (``_MIN_BUCKET`` up to
    the per-rank shard) with neutral dummy inputs on the first forward/backward, folding all summary
    compiles into step 1 so the steady steps are flat.  Enabled by ``MCORE_KDA_CP_PREWARM=1``.
    """
    import os

    return os.environ.get("MCORE_KDA_CP_PREWARM") == "1"


def _cp_static_state_enabled() -> bool:
    """Whether to always hand the main kernel a materialised boundary state (``MCORE_KDA_CP_STATIC_STATE``).

    The cuDNN main-kernel plan cache keys (``make_fprop_cache_key`` / ``make_bprop_cache_key``) include the
    *dtype* of the per-sequence recurrent state operands: ``state_dtype`` (the forward's ``initial_state`` and
    the backward's reconstructed ``initial_state``) and ``dstate_in_dtype`` (the backward's ``d_final_state``).
    The :class:`CudnnSummaryProducer` returns ``None`` for those operands on a rank that starts its own first
    document (``is_first_rank``, no folded initial state) or ends its own last document (``is_last_rank``, no
    folded final-state gradient), and a tensor otherwise.  Under varlen THD packing the packed-document
    boundaries fall in different places inside this rank's fixed-size slice every step, so ``is_first_rank`` /
    ``is_last_rank`` -- and therefore whether the operand is ``None`` or an fp32 tensor -- *toggle step to
    step on the same rank*.  Each ``None<->fp32`` toggle is a new plan key even when ``total`` and ``N`` are
    pinned (by ``pad_packed_seq_alignment=max`` and ``MCORE_KDA_CP_PAD_NSEQ``), so the forward re-plans across
    two keys and the backward across up to four -- the residual varlen-CP recompile storm that pinning the
    token count and the sequence count does not remove.  A rank that starts/ends its own document has a *zero*
    boundary state by definition, so materialising an all-zero ``[N_local, HO, V, K]`` fp32 tensor instead of
    ``None`` is numerically identical (the recurrence already starts from zero / receives no successor
    gradient) and makes ``state_dtype`` / ``dstate_in_dtype`` constant fp32.  Combined with the token-count and
    sequence-count pins the main-kernel plan key is then fully static.  Enabled by ``MCORE_KDA_CP_STATIC_STATE=1``.
    """
    import os

    return os.environ.get("MCORE_KDA_CP_STATIC_STATE") == "1"


def _zeros_boundary_state(n_local: int, ho: int, v: int, k: int, device) -> torch.Tensor:
    """An all-zero per-sequence boundary state ``[n_local, HO, V, K]`` (cuDNN fp32 V-major layout)."""
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
    """Pad a per-sequence state ``[N_local, ...]`` to ``[target_n, ...]`` with zero rows."""
    if state is None or state.shape[0] >= target_n:
        return state
    out = state.new_zeros((target_n,) + tuple(state.shape[1:]))
    out[: state.shape[0]] = state
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


class CudnnSummaryProducer:
    """Default producer: cuDNN's per-span summaries + ``all_gather`` + transition folds."""

    def initial_state(self, k, v, g, beta, cu, cu_list, cp_context, gate_kwargs, flags):
        """Summarise the crossing-out sequence, exchange, fold the predecessors' blocks."""
        ops = _ops.ops()
        if _cp_prewarm_enabled():
            _prewarm_summary_fwd(ops, k, v, g, beta, cu_list[-1], gate_kwargs, flags)
        group = cp_context.group
        rank = dist.get_rank(group)
        HO, V, K = v.shape[1], v.shape[2], k.shape[2]
        if not cp_context.is_last_rank:
            # The last local sequence continues on the next rank: summarise it alone, padded to a
            # shape bucket with neutral tokens (see _bucket_len).
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
        hs, ms = _all_gather_one(h, group), _all_gather_one(m, group)
        if cp_context.is_first_rank:
            return None
        state = torch.zeros_like(h)
        for r in range(rank - cp_context.pre_num_ranks, rank):  # ascending, document start first
            state = torch.matmul(state, ms[r]) + hs[r]
        initial_state = torch.zeros(
            (len(cu_list) - 1, HO, V, K), dtype=torch.float32, device=k.device
        )
        initial_state[0] = state
        return initial_state

    def final_state_grad(
        self, q, k, g, beta, do, cu, cu_list, cp_context, scale, gate_kwargs, flags
    ):
        """Reverse-summarise the crossing-in sequence, exchange, fold the successors' blocks."""
        ops = _ops.ops()
        if _cp_prewarm_enabled():
            _prewarm_summary_bwd(ops, q, k, g, beta, do, cu_list[-1], scale, gate_kwargs, flags)
        group = cp_context.group
        rank = dist.get_rank(group)
        HO, V, K = do.shape[1], do.shape[2], k.shape[2]
        if not cp_context.is_first_rank:
            # The first local sequence continues from the previous rank: summarise it alone,
            # padded to a shape bucket with neutral tokens (see _bucket_len).
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
        gs, ts = _all_gather_one(gsum, group), _all_gather_one(tr, group)
        if cp_context.is_last_rank:
            return None
        grad = torch.zeros_like(gsum)
        for r in range(
            rank + cp_context.post_num_ranks, rank, -1
        ):  # descending, document end first
            grad = torch.matmul(grad, ts[r]) + gs[r]
        d_final_state = torch.zeros(
            (len(cu_list) - 1, HO, V, K), dtype=torch.float32, device=k.device
        )
        d_final_state[-1] = grad
        return d_final_state


_PRODUCERS: dict[str, Any] = {"cudnn": CudnnSummaryProducer}


def register_boundary_state_producer(name: str, factory) -> None:
    """Register an alternative :class:`BoundaryStateProducer` under ``name``."""
    _PRODUCERS[name] = factory


def get_boundary_state_producer(name: str) -> BoundaryStateProducer:
    """Instantiate the registered producer ``name`` (the optional ``cute`` one loads lazily)."""
    if name == "cute" and name not in _PRODUCERS:
        # Optional producer (PR #6511's fused CuTe DSL preprocess); registers itself when its
        # kernels can be imported, else raises with the reason.
        from .cute_preprocess import register

        register()
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
        # Pin the main-kernel plan key's state dtype: a first-rank's own document starts from a zero
        # state, so materialise it (instead of leaving it None) to stop state_dtype toggling None<->fp32
        # as boundaries move between steps (see _cp_static_state_enabled).  Same [N_local, HO, V, K] shape
        # as the folded state the producer builds on other ranks, so the downstream padding is unchanged.
        if initial_state is None and _cp_static_state_enabled():
            initial_state = _zeros_boundary_state(
                len(cu_list) - 1, v.shape[1], v.shape[2], k.shape[2], k.device
            )
        # Pin the main-kernel plan key's sequence count: pad the rank-local cu_seqlens and the
        # per-sequence initial state with zero-length tail segments (see _cp_pad_nseq_target).
        target_n = _cp_pad_nseq_target()
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
        # Pin the backward plan key's dstate_in dtype: a last-rank's own document has no successor
        # gradient, so its final-state gradient is zero -- materialise it (instead of None) to stop
        # dstate_in_dtype toggling None<->fp32 between steps (see _cp_static_state_enabled).  initial_state
        # is already materialised whenever the forward was (ctx.has_initial_state), pinning state_dtype too.
        if d_final_state is None and _cp_static_state_enabled():
            d_final_state = _zeros_boundary_state(
                len(ctx.cu_list) - 1, do.shape[1], do.shape[2], k.shape[2], k.device
            )
        # Pin the backward plan key's sequence count the same way as the forward.
        target_n = _cp_pad_nseq_target()
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
            None,
            None,
            None,
            None,
            None,
            None,
            None,
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
    producer: str = "cudnn",
) -> torch.Tensor:
    """THD chunkwise-CP KDA on cuDNN kernels.

    Operands are already in cuDNN's conventions (see ``ops``).
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
        get_boundary_state_producer(producer),
    )
