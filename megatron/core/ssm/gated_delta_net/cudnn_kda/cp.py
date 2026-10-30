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
Only the boundary-crossing sequence is summarised (the token tensors are sliced to it) and one
entry per rank is exchanged, like FLA's preprocess (``cu_seqlens[-2:]`` forward, ``[:2]``
backward).  The boundary-state production is behind :class:`BoundaryStateProducer` so an
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
        group = cp_context.group
        rank = dist.get_rank(group)
        HO, V, K = v.shape[1], v.shape[2], k.shape[2]
        if not cp_context.is_last_rank:
            # The last local sequence continues on the next rank: summarise it alone.
            start = cu_list[-2]
            cu_tail = torch.tensor([0, cu_list[-1] - start], dtype=torch.int32, device=k.device)
            h, m = ops.kimi_delta_attention_summary(
                k[start:],
                v[start:],
                g[start:],
                beta[start:],
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
        group = cp_context.group
        rank = dist.get_rank(group)
        HO, V, K = do.shape[1], do.shape[2], k.shape[2]
        if not cp_context.is_first_rank:
            # The first local sequence continues from the previous rank: summarise it alone.
            end = cu_list[1]
            cu_head = torch.tensor([0, end], dtype=torch.int32, device=k.device)
            gsum, tr = ops.kimi_delta_attention_summary_bwd(
                q[:end],
                k[:end],
                g[:end],
                beta[:end],
                do[:end],
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
        out, _, _ = ops.kda_fwd(
            q,
            k,
            v,
            g,
            beta,
            cu,
            float(scale),
            initial_state=initial_state,
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
