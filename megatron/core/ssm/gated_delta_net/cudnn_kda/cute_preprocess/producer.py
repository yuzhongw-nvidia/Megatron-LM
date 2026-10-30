# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""``CuTePreprocessProducer``: PR #6511's fused CuTe DSL CP pre-process as a cuDNN producer.

The cuDNN KDA chunkwise-CP path (``cudnn_kda/cp.py``) obtains its two cross-rank boundary
tensors through the ``BoundaryStateProducer`` protocol: the initial state of this rank's first
local sequence (forward) and the gradient of its last local sequence's final state (backward),
both fp32 ``[N_local, HO, V, K]`` in cuDNN's V-major layout, or ``None`` on ranks whose boundary
sequence starts / ends locally.  This producer answers them with the fused CuTe DSL kernels
ported from PR #6511 (``backend.py`` dispatcher, ``fused_ws.py`` / ``bwd_ws.py`` engines).

Those kernels are a drop-in for FLA's Triton CP pre-process and consume FLA's chunk-level
intermediates rather than raw operands: the fp32 chunk-local log2 gate cumsum ``gk``
(``kda_gate_chunk_cumsum`` / ``chunk_local_cumsum``), the WY representation ``w`` / ``u`` and the
pre-gated keys ``kg = k * 2^(gk_last - gk)`` (``chunk_kda_fwd_intra`` -> ``recompute_w_u_fwd``),
and for the backward the pre-gated queries ``qg = q * 2^gk`` and the intra-chunk term
``dv = Aqk @ do`` (``chunk_kda_bwd_dAv``).  The producer therefore runs exactly those FLA kernels
on the one boundary-crossing sequence -- the LAST local sequence in the forward and the FIRST in
the backward, the span FLA's own producer selects with ``cu_seqlens[-2:]`` / ``cu_seqlens[:2]`` --
and hands the results to the fused kernels, which push every rank's summary strips to their
consumers through symmetric memory, fold them in-kernel and return the merged state directly
(FLA layout ``[N, HV, K, V]``; forward slot 0, backward slot ``N - 1``).  The producer converts
to cuDNN's layout with ``ops.fla_state_to_cudnn``.

Both methods are collective over ``cp_context.group`` and every rank must call them in the same
order (the autograd function in ``cp.py`` guarantees that): the kernels run the whole-group chain
and a rank with nothing to contribute pushes zeros (``backend._emit_flags``).  The protocol's
``None`` is returned after the collective on ranks with ``is_first_rank`` (forward) /
``is_last_rank`` (backward), exactly where FLA's and cuDNN's producers leave the slot zero.

Dispatch metadata: the fused kernels decide ``fused`` / ``noop`` identically on every rank from
the GLOBAL packed offsets.  FLA's context only carries rank-local offsets, so the first call on a
context object reconstructs the global ones with one ``all_gather_object`` and attaches them
(``context_metadata.ensure_cp_context_metadata``), like PR #6511's attention module did with
``prepare_cp_context_metadata``.

CUDA graphs are refused at capture time: the fused kernels spin on symmetric-memory flags and the
producer runs host-side collectives, so both methods raise ``NotImplementedError`` when
``torch.cuda.is_current_stream_capturing()``.  The dispatcher's config-driven check
(``backend._cuda_graphs_enabled``) is inert here -- the context metadata is attached without a
model config -- so this capture-time guard is the effective one.
"""

from __future__ import annotations

import functools
from types import SimpleNamespace
from typing import Optional

import torch

from .. import ops as _ops
from . import backend as _backend
from .context_metadata import ensure_cp_context_metadata

CHUNK_SIZE = 64
# Kernel-side transformations the cuDNN operators can fuse but FLA's chunk kernels (and PR
# #6511's KDA adapter) expect done beforehand: Megatron's KDA module hands over L2-normalised
# q / k and post-sigmoid fp32 beta with all three unset.
_UNSUPPORTED_FLAGS = ("use_qk_l2norm_in_kernel", "use_beta_sigmoid_in_kernel", "allow_neg_eigval")


@functools.lru_cache(maxsize=1)
def _fla() -> SimpleNamespace:
    """The FLA Triton kernels the producer composes, imported on first use."""
    try:
        from fla.ops.kda.chunk_bwd import chunk_kda_bwd_dAv
        from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra
        from fla.ops.kda.gate import kda_gate_chunk_cumsum
        from fla.ops.utils import chunk_local_cumsum
        from fla.ops.utils.constant import RCP_LN2
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "CuTePreprocessProducer needs flash-linear-attention >= 0.5.1 (chunk_kda_fwd_intra, "
            f"chunk_kda_bwd_dAv, kda_gate_chunk_cumsum, chunk_local_cumsum): {e}"
        ) from e
    return SimpleNamespace(
        chunk_kda_bwd_dAv=chunk_kda_bwd_dAv,
        chunk_kda_fwd_intra=chunk_kda_fwd_intra,
        kda_gate_chunk_cumsum=kda_gate_chunk_cumsum,
        chunk_local_cumsum=chunk_local_cumsum,
        RCP_LN2=RCP_LN2,
    )


def _window(t: torch.Tensor, bos: int, eos: int) -> torch.Tensor:
    """Tokens ``[bos, eos)`` of a THD operand as a contiguous ``[1, eos - bos, ...]`` tensor.

    FLA's kernels index a sequence's chunks from its own first token, so running them on the
    sliced sequence (dense ``B == 1``, no ``cu_seqlens``) gives the same per-chunk tensors the
    whole-shard call would give for that sequence; the fused kernels then see exactly the window
    PR #6511's dispatcher would have narrowed to.
    """
    return t[bos:eos].unsqueeze(0).contiguous()


def _check_envelope(qk: torch.Tensor, v_like: torch.Tensor, flags: dict) -> None:
    """Adapter-level refusals mirroring PR #6511's KDA adapter; the kernel envelope is checked
    by ``backend._is_supported`` / ``_is_supported_bwd`` (K == V == 128, HV % 4, bf16, ...)."""
    for name in _UNSUPPORTED_FLAGS:
        if flags.get(name):
            raise NotImplementedError(
                f"CuTePreprocessProducer does not support {name}=True: it composes FLA's chunk "
                "kernels, which take L2-normalised q/k and post-sigmoid beta (PR #6511's adapter "
                "refuses the same options); use cp_state_producer='cudnn'"
            )
    if qk.dim() != 3 or v_like.dim() != 3:
        raise ValueError(
            "CuTePreprocessProducer expects THD operands [total_tokens, heads, dim], got "
            f"{tuple(qk.shape)} and {tuple(v_like.shape)}"
        )
    if qk.shape[1] != v_like.shape[1]:
        raise NotImplementedError(
            f"CuTePreprocessProducer does not support grouped value attention (H={qk.shape[1]}, "
            f"HV={v_like.shape[1]}); PR #6511's KDA adapter refuses it as well"
        )


def _gate_cumsum(fla: SimpleNamespace, g_w: torch.Tensor, gate_kwargs: dict) -> torch.Tensor:
    """FLA's fp32 chunk-local log2 gate cumsum ``gk`` ``[1, T, H, K]`` from the cuDNN gate operand.

    With ``safe_gate`` the cuDNN operand is the raw logits and the bounded gate
    ``lower_bound * sigmoid(exp(a_log) * (g + dt_bias))`` is fused with the cumsum in FLA's
    ``kda_gate_chunk_cumsum`` -- the call FLA's ``chunk_kda_fwd`` makes under
    ``use_gate_in_kernel``.  Otherwise the operand already is the natural-log decay
    (``ops.prepare_gate`` applied the softplus form, or the caller passed the log decay) and
    ``chunk_local_cumsum`` converts it to log2 (``scale=RCP_LN2``), as FLA does.
    """
    if g_w.dtype != torch.float32:
        g_w = g_w.float()
    if gate_kwargs.get("safe_gate"):
        a_log = gate_kwargs.get("a_log")
        lower_bound = gate_kwargs.get("gate_lower_bound")
        if a_log is None:
            raise NotImplementedError(
                "CuTePreprocessProducer: the bounded gate without a_log has no FLA equivalent "
                "(kda_gate_chunk_cumsum needs A_log)"
            )
        if lower_bound is None or not (-5 <= float(lower_bound) < 0):
            raise ValueError(
                f"CuTePreprocessProducer: gate_lower_bound must lie in [-5, 0) (FLA's safe-gate "
                f"range), got {lower_bound}"
            )
        dt_bias = gate_kwargs.get("dt_bias")
        return fla.kda_gate_chunk_cumsum(
            g=g_w,
            A_log=a_log.float().contiguous(),
            chunk_size=CHUNK_SIZE,
            scale=fla.RCP_LN2,
            dt_bias=None if dt_bias is None else dt_bias.float().contiguous(),
            cu_seqlens=None,
            output_dtype=torch.float32,
            chunk_indices=None,
            lower_bound=float(lower_bound),
        )
    return fla.chunk_local_cumsum(
        g=g_w,
        chunk_size=CHUNK_SIZE,
        scale=fla.RCP_LN2,
        cu_seqlens=None,
        output_dtype=torch.float32,
        chunk_indices=None,
    )


class CuTePreprocessProducer:
    """Boundary-state producer backed by the fused CuTe DSL CP pre-process (see the module doc).

    Registered as ``"cute"`` by :func:`cute_preprocess.register`; construction fails with the
    reason when the toolchain, the device or flash-linear-attention is missing.  Both methods
    refuse to run under CUDA-graph capture.
    """

    name = "cute"

    def __init__(self) -> None:
        from . import unavailable_reason

        reason = unavailable_reason()
        if reason is not None:
            raise RuntimeError(f"cuDNN KDA boundary-state producer 'cute' is unavailable: {reason}")

    def initial_state(
        self, k, v, g, beta, cu, cu_list, cp_context, gate_kwargs, flags
    ) -> Optional[torch.Tensor]:
        """Initial state of this rank's first local sequence (``BoundaryStateProducer`` contract).

        Collective over ``cp_context.group``: every rank calls it.  Returns the fp32
        ``[N_local, HO, V, K]`` cuDNN-layout state (slot 0 live) or ``None`` on ``is_first_rank``;
        refuses CUDA-graph capture.
        """
        if torch.cuda.is_current_stream_capturing():
            raise NotImplementedError(
                "the CuTe CP preprocess producer cannot run under CUDA-graph capture"
            )
        fla = _fla()
        _check_envelope(k, v, flags)
        ensure_cp_context_metadata(cp_context)

        # The LAST local sequence is the only one that can continue onto a later rank: FLA's
        # producer runs on `cu_seqlens[-2:]`, PR #6511's dispatcher narrows to the same window.
        bos, eos = int(cu_list[-2]), int(cu_list[-1])
        k_w, v_w, beta_w = _window(k, bos, eos), _window(v, bos, eos), _window(beta, bos, eos)
        gk = _gate_cumsum(fla, _window(g, bos, eos), gate_kwargs)
        # FLA's intra-chunk forward on the window: the inverse intra-chunk matrix Akk and from it
        # w = Akk @ (k * beta * 2^gk), u = Akk @ (v * beta), kg = k * 2^(gk_last - gk).  `q` and
        # `scale` only feed Aqk, which the forward pre-process does not consume, so `k` stands in
        # for `q` and Aqk is discarded.
        w, u, _qg, kg, _Aqk, _Akk = fla.chunk_kda_fwd_intra(
            q=k_w,
            k=k_w,
            v=v_w,
            gk=gk,
            beta=beta_w,
            scale=float(k.shape[-1]) ** -0.5,
            cu_seqlens=None,
            chunk_size=CHUNK_SIZE,
            chunk_indices=None,
            safe_gate=bool(gate_kwargs.get("safe_gate", False)),
            disable_recompute=False,
        )
        # Collective: every rank launches; the result is FLA-layout [N, HV, K, V], slot 0 live.
        state = _backend.chunk_gated_delta_rule_fwd_h_pre_process_cutedsl(
            k=kg,
            w=w,
            u=u,
            g=None,
            gk=gk,
            bg=None,
            v=None,
            chunk_size=CHUNK_SIZE,
            state_v_first=False,
            cu_seqlens=cu,
            context=cp_context,
            windowed=True,
        )
        if cp_context.is_first_rank:
            return None
        return _ops.fla_state_to_cudnn(state, state_v_first=False)

    def final_state_grad(
        self, q, k, g, beta, do, cu, cu_list, cp_context, scale, gate_kwargs, flags
    ) -> Optional[torch.Tensor]:
        """Gradient of this rank's last local sequence's final state (``BoundaryStateProducer``).

        Collective over ``cp_context.group``: every rank calls it.  Returns the fp32
        ``[N_local, HO, V, K]`` cuDNN-layout gradient (slot ``N_local - 1`` live) or ``None`` on
        ``is_last_rank``; refuses CUDA-graph capture.
        """
        if torch.cuda.is_current_stream_capturing():
            raise NotImplementedError(
                "the CuTe CP preprocess producer cannot run under CUDA-graph capture"
            )
        fla = _fla()
        _check_envelope(q, do, flags)
        ensure_cp_context_metadata(cp_context)

        # The FIRST local sequence is the only one that can continue from an earlier rank: FLA's
        # backward producer runs on `cu_seqlens[:2]`.
        bos, eos = int(cu_list[0]), int(cu_list[1])
        q_w, k_w, beta_w = _window(q, bos, eos), _window(k, bos, eos), _window(beta, bos, eos)
        do_w = _window(do, bos, eos)
        if do_w.dtype != q_w.dtype:
            do_w = do_w.to(q_w.dtype)
        gk = _gate_cumsum(fla, _window(g, bos, eos), gate_kwargs)
        # Recompute the intra-chunk tensors on the window with `disable_recompute=True`, which
        # makes `recompute_w_u_fwd` materialise qg = q * 2^gk: the gk-mode kernels never gate q
        # themselves (PR #6511's adapter passes `q` to `recompute_w_u_fwd` for the same reason).
        # `v` only feeds `u`, which the backward pre-process does not consume, so `do` (same
        # shape and dtype) stands in for it and `u` is discarded.
        w, _u, qg, kg, Aqk, _Akk = fla.chunk_kda_fwd_intra(
            q=q_w,
            k=k_w,
            v=do_w,
            gk=gk,
            beta=beta_w,
            scale=float(scale),
            cu_seqlens=None,
            chunk_size=CHUNK_SIZE,
            chunk_indices=None,
            safe_gate=bool(gate_kwargs.get("safe_gate", False)),
            disable_recompute=True,
        )
        # The intra-chunk term dv = Aqk @ do the CP pre-process adds to k @ dh.  FLA passes
        # `v=v_new`, but `v` only feeds the dAqk output (do @ v^T * scale), which is discarded
        # here, so `do` stands in once more.
        _dAqk, dv = fla.chunk_kda_bwd_dAv(
            q=q_w,
            k=k_w,
            v=do_w,
            do=do_w,
            A=Aqk,
            scale=float(scale),
            cu_seqlens=None,
            chunk_size=CHUNK_SIZE,
            chunk_indices=None,
        )
        # Collective: every rank launches; FLA-layout [N, HV, K, V], slot N - 1 live.
        dht = _backend.chunk_gated_delta_rule_bwd_dhu_pre_process_cutedsl(
            q=qg,
            k=kg,
            w=w,
            do=do_w,
            dv=dv,
            g=None,
            gk=gk,
            bg=None,
            scale=float(scale),
            state_v_first=False,
            cu_seqlens=cu,
            dht=None,
            chunk_size=CHUNK_SIZE,
            context=cp_context,
            windowed=True,
        )
        if cp_context.is_last_rank:
            return None
        return _ops.fla_state_to_cudnn(dht, state_v_first=False)
