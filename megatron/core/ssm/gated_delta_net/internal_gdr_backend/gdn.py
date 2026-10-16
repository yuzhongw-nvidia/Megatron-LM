# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Portions of this file follow the MIT-licensed FLA v0.5.1 GDR autograd
# contract. The SM100 forward implementation is local to Megatron Core.

"""Autograd orchestration for the in-tree SM100 CuTe DSL GDR kernel."""

from __future__ import annotations

import inspect
import re
from collections import OrderedDict
from typing import Any

import torch
from fla.ops.common.chunk_delta_h import (
    chunk_gated_delta_rule_bwd_dhu,
    chunk_gated_delta_rule_fwd_h,
)
from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
from fla.ops.cp.chunk_delta_h import compress_h0, expand_h0
from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule as fla_chunk_gated_delta_rule
from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

from .chunk import _validate_public_options
from .cp_preprocess import _cp_backward_preprocess, _cp_forward_preprocess
from .dispatch import (
    _FUSED_BWD_HEAD_DIM,
    _FUSED_BWD_HEADS,
    _backend_mode,
    _cutedsl_support_reason,
    _fused_bwd_support_reason,
)
from .metadata import (
    _CHUNK_SIZE,
    _current_stream_cache_key,
    _dense_chunk_metadata,
    _device_cache_key,
    _packed_chunk_metadata,
    _validate_validated_chunk_indices,
)

_fused_bwd_zero_dht_cache: OrderedDict[
    tuple[str, int | None, int, tuple[int, int] | None], torch.Tensor
] = OrderedDict()
_FUSED_BWD_ZERO_DHT_CACHE_MAX_BYTES = 256 * 1024 * 1024


def _check_fla_gate_contract() -> None:
    """Refuse FLA releases whose chunk primitives still take ``use_exp2``.

    This module calls FLA's chunk primitives without ``use_exp2`` or
    ``transpose_state_layout``. That is behaviour-identical only from FLA 0.5.1, where the
    gate is a log2 cumsum unconditionally and the state layout flag is ``state_v_first``;
    FLA 0.5.0 defaults ``use_exp2`` to False, so the same calls would silently change the
    numerics there.
    """
    try:
        parameters = inspect.signature(chunk_gated_delta_rule_fwd_h).parameters
    except (TypeError, ValueError):
        # The primitive is wrapped without an introspectable signature: fall back to
        # the release number rather than refusing a supported FLA build.
        import fla

        digits = [int(p) for p in re.findall(r"\d+", getattr(fla, "__version__", "0"))[:3]]
        if tuple(digits + [0] * (3 - len(digits))) >= (0, 5, 1):
            return
        parameters = {"use_exp2": None}
    if "use_exp2" in parameters or "state_v_first" not in parameters:
        raise ImportError(
            "The internal GDR backend requires flash-linear-attention >= 0.5.1 (log2 gate "
            "primitives with state_v_first); the installed release still exposes the "
            "pre-0.5.1 chunk_gated_delta_rule_fwd_h signature."
        )


_check_fla_gate_contract()


def _fla_backward(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    do: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None,
    dht: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    cp_context: object | None = None,
):
    """Recompose the FLA v0.5.1 backward from its public kernel primitives."""
    w, u = recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=A, g=g, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
    )
    if cp_context is not None:
        initial_state = expand_h0(initial_state, context=cp_context)
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    dv = chunk_bwd_dv_local(
        q=q, k=k, g=g, do=do, scale=scale, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
    )
    if cp_context is not None:
        dht, initial_state = _cp_backward_preprocess(
            q=q,
            k=k,
            w=w,
            do=do,
            dv=dv,
            g=g,
            scale=scale,
            cu_seqlens=cu_seqlens,
            dht=dht,
            initial_state=initial_state,
            context=cp_context,
        )
    dh, _dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=q,
        k=k,
        w=w,
        g=g,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    dq, dk, dw, dg = chunk_bwd_dqkwg(
        q=q,
        k=k,
        v=v_new,
        w=w,
        g=g,
        h=h,
        dv=dv,
        do=do,
        dh=dh,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    dk2, dv, db, dg2 = prepare_wy_repr_bwd(
        k=k,
        v=v,
        beta=beta,
        g=g,
        A=A,
        dw=dw,
        du=dv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    dk.add_(dk2)
    dg.add_(dg2)
    dg = chunk_local_cumsum(
        dg, chunk_size=_CHUNK_SIZE, reverse=True, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
    )
    return dq, dk, dv, db, dg


def _fla_cp_backward_preprocess(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    cp_context: object,
    h: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use FLA's CP prefix to produce the fused kernel's boundary state."""
    dense_local_cp = q.shape[0] == 1 and cu_seqlens is not None and cu_seqlens.numel() == 2
    compute_cu_seqlens = None if dense_local_cp else cu_seqlens
    compute_chunk_indices = None if dense_local_cp else chunk_indices
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g,
        cu_seqlens=compute_cu_seqlens,
        chunk_indices=compute_chunk_indices,
    )
    expanded_initial_state = expand_h0(initial_state, context=cp_context)
    if h is None:
        h, _v_new, _ = chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=u,
            g=g,
            initial_state=expanded_initial_state,
            output_final_state=False,
            save_new_value=True,
            cu_seqlens=compute_cu_seqlens,
            chunk_indices=compute_chunk_indices,
        )
    dv = chunk_bwd_dv_local(
        q=q,
        k=k,
        g=g,
        do=do,
        scale=scale,
        cu_seqlens=compute_cu_seqlens,
        chunk_indices=compute_chunk_indices,
    )
    dht, _ = _cp_backward_preprocess(
        q=q,
        k=k,
        w=w,
        do=do,
        dv=dv,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        dht=dht,
        initial_state=expanded_initial_state,
        context=cp_context,
    )
    return dht, h


def _recompute_fused_bwd_h(
    *,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None,
) -> torch.Tensor:
    """Recompute the recurrent state exactly as in latest-main's verified path."""
    w, u = recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=A, g=g, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
    )
    h, _v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=None,
        output_final_state=False,
        save_new_value=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return h


def _fused_bwd_zero_dht(device: torch.device, num_sequences: int) -> torch.Tensor:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    key = (*_device_cache_key(device), num_sequences, _current_stream_cache_key(device))
    cached = _fused_bwd_zero_dht_cache.get(key)
    if cached is not None and cached.device == device:
        _fused_bwd_zero_dht_cache.move_to_end(key)
        return cached
    if cached is not None:
        _fused_bwd_zero_dht_cache.pop(key, None)
    cached = torch.zeros(
        (num_sequences, _FUSED_BWD_HEADS, _FUSED_BWD_HEAD_DIM, _FUSED_BWD_HEAD_DIM),
        dtype=torch.float32,
        device=device,
    )
    _fused_bwd_zero_dht_cache[key] = cached
    cached_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in _fused_bwd_zero_dht_cache.values()
    )
    while cached_bytes > _FUSED_BWD_ZERO_DHT_CACHE_MAX_BYTES and _fused_bwd_zero_dht_cache:
        _, evicted = _fused_bwd_zero_dht_cache.popitem(last=False)
        cached_bytes -= evicted.numel() * evicted.element_size()
    return cached


def _prepare_fused_bwd_h(h: torch.Tensor, *, num_heads: int, head_size: int) -> torch.Tensor:
    """Normalize saved or recomputed chunk states to the fused-backward layout."""
    return (
        h.detach()
        .reshape(-1, num_heads, head_size, head_size)
        .to(torch.bfloat16)
        .contiguous()
    )


def _call_fused_gdr_bwd_cute(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    scale: float,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None,
    chunk_offsets: torch.Tensor | None = None,
    h: torch.Tensor | None = None,
):
    from .kernels.fused_gdr_bwd_cute import fused_gdr_bwd

    batch_size, seqlen, num_heads, head_size = q.shape
    total_tokens = batch_size * seqlen
    dense_metadata = None
    if cu_seqlens is None:
        dense_metadata = _dense_chunk_metadata(batch_size, seqlen, q.device)
        launch_cu_seqlens = dense_metadata.cu_seqlens
        if chunk_offsets is None:
            chunk_offsets = dense_metadata.chunk_offsets
    else:
        launch_cu_seqlens = cu_seqlens
    num_sequences = launch_cu_seqlens.numel() - 1
    if h is None:
        h = _recompute_fused_bwd_h(
            k=k, v=v, g=g, beta=beta, A=A, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
        )
    launch_h = _prepare_fused_bwd_h(h, num_heads=num_heads, head_size=head_size).unsqueeze(0)
    # FLA's saved gate is the log2-space cumulative value consumed by its
    # use_exp2 primitives. The fused backward kernel materializes exp2(g * log2e),
    # so it expects the same cumulative decay in natural-log units.
    launch_g = (
        g.detach().reshape(1, total_tokens, num_heads).to(torch.float32) / RCP_LN2
    ).contiguous()
    launch_beta = beta.detach().reshape(1, total_tokens, num_heads).to(torch.float32).contiguous()
    launch_dht = (
        _fused_bwd_zero_dht(q.device, num_sequences) if dht is None else dht.detach().contiguous()
    )

    dq, dk, dv, dg, db, _dh0 = fused_gdr_bwd(
        q=q.detach().reshape(1, total_tokens, num_heads, head_size).contiguous(),
        k=k.detach().reshape(1, total_tokens, num_heads, head_size).contiguous(),
        v=v.detach().reshape(1, total_tokens, num_heads, head_size).contiguous(),
        a=A.detach().reshape(1, total_tokens, num_heads, _CHUNK_SIZE).contiguous(),
        g=launch_g,
        beta=launch_beta,
        do=do.detach().reshape(1, total_tokens, num_heads, head_size).contiguous(),
        dht=launch_dht,
        h=launch_h,
        scale=scale,
        cu_seqlens=launch_cu_seqlens,
        chunk_offsets=chunk_offsets,
        chunk_size=_CHUNK_SIZE,
        state_v_first=False,
        trusted_chunk_offsets=chunk_offsets is not None,
    )
    dg = chunk_local_cumsum(
        dg.reshape_as(g),
        chunk_size=_CHUNK_SIZE,
        reverse=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return (
        dq.reshape_as(q),
        dk.reshape_as(k),
        dv.reshape_as(v),
        db.reshape_as(beta).to(beta.dtype),
        dg.to(g.dtype),
    )


def _reshape_bthd_to_thd(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0] * tensor.shape[1], *tensor.shape[2:]).contiguous()


def _reshape_bth_to_th(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0] * tensor.shape[1], tensor.shape[2]).contiguous()


def _fla_local_wy(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return FLA's log2 chunk-local gate cumsum and the WY representation ``(w, u, A)``."""
    g = chunk_local_cumsum(
        g,
        chunk_size=_CHUNK_SIZE,
        scale=RCP_LN2,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    w, u, A = chunk_gated_delta_rule_fwd_intra(
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=_CHUNK_SIZE,
    )
    return g, w, u, A


def _fla_forward_for_fused_bwd(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None = None,
    save_fused_bwd_state: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    None,
]:
    """Prepare the exact FLA state consumed by latest-main's fused backward (non-CP)."""
    g, w, u, A = _fla_local_wy(k, v, g, beta, cu_seqlens, chunk_indices)
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=None,
        output_final_state=False,
        save_new_value=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    output = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    saved_h = None
    if save_fused_bwd_state:
        saved_h = _prepare_fused_bwd_h(
            h,
            num_heads=q.shape[2],
            head_size=q.shape[3],
        )
    # The last slot is the saved CP boundary state; this path never has one.
    return g, output, A, saved_h, chunk_indices, None


def _cp_initial_state_for_cutedsl_forward(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None = None,
    cp_context: object,
    dense_local_cp: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Prepare CP h0 for the local CuTe forward while preserving backward state."""
    if dense_local_cp and (cp_context is None or q.shape[0] != 1):
        raise ValueError("dense_local_cp is only valid for single-batch CP inputs")
    compute_cu_seqlens = None if dense_local_cp else cu_seqlens
    compute_chunk_indices = None if dense_local_cp else chunk_indices
    local_g, w, u, _A = _fla_local_wy(k, v, g, beta, compute_cu_seqlens, compute_chunk_indices)
    initial_state = _cp_forward_preprocess(
        k=k,
        w=w,
        u=u,
        g=local_g,
        cu_seqlens=cu_seqlens,
        initial_state=None,
        context=cp_context,
    )
    return (
        initial_state,
        compress_h0(initial_state, context=cp_context),
        compute_chunk_indices,
    )


def _cutedsl_forward(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    save_fused_bwd_state: bool = False,
    initial_state: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    from .kernels.fused_gdr_fwd_cute import chunk_gated_delta_rule_prefill_cute

    batch_size, seqlen, num_heads, head_size = q.shape
    if cu_seqlens is None:
        dense_metadata = _dense_chunk_metadata(batch_size, seqlen, q.device)
        launch_cu_seqlens = dense_metadata.cu_seqlens
        chunk_offsets = dense_metadata.chunk_offsets
    else:
        if save_fused_bwd_state and (chunk_indices is None or chunk_offsets is None):
            raise ValueError(
                "packed inputs need caller-prepared chunk_indices and chunk_offsets to save "
                "the fused-backward state"
            )
        launch_cu_seqlens = cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous()
    flat_q = _reshape_bthd_to_thd(q.detach())
    flat_k = _reshape_bthd_to_thd(k.detach())
    flat_v = _reshape_bthd_to_thd(v.detach())
    flat_raw_g = _reshape_bth_to_th(g.detach())
    flat_g = torch.empty((batch_size * seqlen, num_heads), dtype=torch.float32, device=q.device)
    flat_beta = _reshape_bth_to_th(beta.detach())
    flat_A = torch.empty(
        (batch_size * seqlen, num_heads, _CHUNK_SIZE), dtype=q.dtype, device=q.device
    )
    output = torch.empty(
        (batch_size * seqlen, num_heads, head_size), dtype=q.dtype, device=q.device
    )
    flat_h = None
    checkpoint_cu_starts = None
    if save_fused_bwd_state:
        if cu_seqlens is not None:
            # One saved state per (sequence, chunk) pair, i.e. sum(ceil(len_i / 64)); the host
            # already knows this count because prepare_chunk_indices materializes the rows.
            num_chunks = int(chunk_indices.shape[0])
        else:
            num_chunks = batch_size * ((seqlen + _CHUNK_SIZE - 1) // _CHUNK_SIZE)
        flat_h = torch.empty(
            (num_chunks, num_heads, head_size, head_size), dtype=q.dtype, device=q.device
        )
        checkpoint_cu_starts = chunk_offsets
    chunk_gated_delta_rule_prefill_cute(
        q=flat_q,
        k=flat_k,
        v=flat_v,
        g=flat_raw_g,
        beta=flat_beta,
        scale=scale,
        initial_state=initial_state,
        cu_seqlens=launch_cu_seqlens,
        output=output,
        output_A=flat_A,
        output_g=flat_g,
        assume_valid_cu_seqlens=True,
        output_h=flat_h,
        checkpoint_every_n_tokens=_CHUNK_SIZE if save_fused_bwd_state else 0,
        checkpoint_cu_starts=checkpoint_cu_starts,
        gate_is_log_decay=True,
    )
    return (
        flat_g.reshape(batch_size, seqlen, num_heads),
        output.reshape(batch_size, seqlen, num_heads, head_size),
        flat_A.reshape(batch_size, seqlen, num_heads, _CHUNK_SIZE),
        flat_h,
        chunk_indices,
        chunk_offsets,
    )


def _cutedsl_backward(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None,
    chunk_offsets: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    cp_context: object | None = None,
    h: torch.Tensor | None = None,
):
    reason = _fused_bwd_support_reason(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A=A,
        do=do,
        dht=dht,
        cu_seqlens=cu_seqlens,
    )
    if reason is None:
        if cp_context is not None:
            dht, h = _fla_cp_backward_preprocess(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A=A,
                scale=scale,
                do=do,
                dht=dht,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                initial_state=initial_state,
                cp_context=cp_context,
                h=h,
            )
        return _call_fused_gdr_bwd_cute(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A=A,
            do=do,
            dht=dht,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            h=h,
        )
    if _backend_mode() == "cute":
        raise RuntimeError(f"Internal fused CuTe DSL GDR backward is unavailable: {reason}")
    return _fla_backward(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A=A,
        scale=scale,
        do=do,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        dht=dht,
        initial_state=initial_state,
        cp_context=cp_context,
    )


class InternalChunkGatedDeltaRuleFunction(torch.autograd.Function):
    """Use the local CuTe DSL forward or the verified fused-backward pipeline."""

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        cu_seqlens: torch.LongTensor | None,
        validated_chunk_indices: torch.LongTensor | None,
        validated_chunk_offsets: torch.LongTensor | None,
        recompute_h: bool,
        cp_context: object | None,
    ):
        mode = _backend_mode()
        _validate_validated_chunk_indices(validated_chunk_indices, cu_seqlens)
        if validated_chunk_offsets is not None and (
            cu_seqlens is None
            or validated_chunk_offsets.shape != cu_seqlens.shape
            or validated_chunk_offsets.dtype != cu_seqlens.dtype
            or validated_chunk_offsets.device != cu_seqlens.device
            or not validated_chunk_offsets.is_contiguous()
        ):
            raise ValueError(
                "validated_chunk_offsets must match cu_seqlens shape, dtype, and device"
            )
        use_fused_bwd = (
            _fused_bwd_support_reason(q=q, k=k, v=v, g=g, beta=beta, cu_seqlens=cu_seqlens)
            is None
        )
        save_fused_bwd_state = use_fused_bwd and not recompute_h
        dense_local_cp = (
            cp_context is not None
            and q.shape[0] == 1
            and cu_seqlens is not None
            and cu_seqlens.numel() == 2
        )
        saved_h = None
        saved_initial_state = None
        chunk_indices = validated_chunk_indices
        chunk_offsets = validated_chunk_offsets
        if (
            cu_seqlens is not None
            and not dense_local_cp
            and (chunk_indices is None or chunk_offsets is None)
        ):
            metadata = _packed_chunk_metadata(
                cu_seqlens, include_chunk_indices=chunk_indices is None
            )
            if chunk_indices is None:
                chunk_indices = metadata.chunk_indices
            if chunk_offsets is None:
                chunk_offsets = metadata.chunk_offsets

        if cp_context is not None:
            forward_initial_state, saved_initial_state, chunk_indices = (
                _cp_initial_state_for_cutedsl_forward(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    cu_seqlens=cu_seqlens,
                    chunk_indices=chunk_indices,
                    cp_context=cp_context,
                    dense_local_cp=dense_local_cp,
                )
            )
            forward_cu_seqlens = None if dense_local_cp else cu_seqlens
            g, output, A, saved_h, chunk_indices, generated_chunk_offsets = _cutedsl_forward(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                cu_seqlens=forward_cu_seqlens,
                chunk_indices=chunk_indices,
                chunk_offsets=None if dense_local_cp else chunk_offsets,
                save_fused_bwd_state=save_fused_bwd_state,
                initial_state=forward_initial_state,
            )
            if chunk_offsets is None:
                chunk_offsets = generated_chunk_offsets
        elif mode != "cute" and use_fused_bwd:
            g, output, A, saved_h, chunk_indices, saved_initial_state = _fla_forward_for_fused_bwd(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                save_fused_bwd_state=save_fused_bwd_state,
            )
        else:
            g, output, A, saved_h, chunk_indices, generated_chunk_offsets = _cutedsl_forward(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                chunk_offsets=chunk_offsets,
                save_fused_bwd_state=save_fused_bwd_state,
            )
            if chunk_offsets is None:
                chunk_offsets = generated_chunk_offsets
        ctx.save_for_backward(
            q,
            k,
            v,
            g,
            beta,
            A,
            saved_h,
            saved_initial_state,
            cu_seqlens,
            chunk_indices,
            chunk_offsets,
        )
        ctx.scale = scale
        ctx.cp_context = cp_context
        return output.to(q.dtype), None

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor | None):
        q, k, v, g, beta, A, saved_h, initial_state, cu_seqlens, chunk_indices, chunk_offsets = (
            ctx.saved_tensors
        )
        dq, dk, dv, db, dg = _cutedsl_backward(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A=A,
            scale=ctx.scale,
            do=do,
            dht=dht,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            initial_state=initial_state,
            cp_context=ctx.cp_context,
            h=saved_h,
        )
        return (
            dq.to(q),
            dk.to(k),
            dv.to(v),
            dg.to(g),
            db.to(beta),
            None,
            None,
            None,
            None,
            None,
            None,
        )


@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    validated_chunk_indices: torch.LongTensor | None = None,
    validated_chunk_offsets: torch.LongTensor | None = None,
    cp_context: object | None = None,
    recompute_h: bool = False,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Dispatch supported SM100 inputs to CuTe DSL and otherwise use FLA."""
    state_v_first = _validate_public_options(
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        state_v_first=state_v_first,
        kwargs=kwargs,
    )

    mode = _backend_mode()
    if mode == "fla":
        return fla_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            state_v_first=state_v_first,
            cu_seqlens=cu_seqlens,
            cp_context=cp_context,
            **kwargs,
        )
    if cp_context is not None:
        # CP context offsets replace the caller's packed offsets.
        cp_cu_seqlens = getattr(cp_context, "cu_seqlens", None)
        if cp_cu_seqlens is None:
            raise ValueError("cp_context.cu_seqlens is required for context parallel GDR")
        if q.shape[0] != 1:
            raise ValueError(
                "context parallel GDR requires batch size 1; pack the batch into cu_seqlens"
            )
        cu_seqlens = cp_cu_seqlens

    reason = _cutedsl_support_reason(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
        cp_context=cp_context,
        kwargs=kwargs,
    )
    if reason is not None:
        if mode == "cute":
            raise RuntimeError(f"Internal CuTe DSL GDR path is unavailable: {reason}")
        return fla_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            state_v_first=state_v_first,
            cu_seqlens=cu_seqlens,
            cp_context=cp_context,
            **kwargs,
        )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    return InternalChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        cu_seqlens,
        validated_chunk_indices,
        validated_chunk_offsets,
        recompute_h,
        cp_context,
    )
