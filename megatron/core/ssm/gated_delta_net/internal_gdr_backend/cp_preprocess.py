# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CP boundary pre-processing shims: in-tree CuTe DSL kernels first, FLA otherwise."""

from __future__ import annotations

import torch
from fla.ops.cp.chunk_delta_h import (
    chunk_gated_delta_rule_bwd_dhu_pre_process,
    chunk_gated_delta_rule_fwd_h_pre_process,
)

from .kernels.fused_gdr_cp_cute import (
    try_chunk_gated_delta_rule_bwd_dhu_pre_process_cutedsl,
    try_chunk_gated_delta_rule_fwd_h_pre_process_cutedsl,
)
from .metadata import _CHUNK_SIZE


def _cp_forward_preprocess(
    *,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    context: object,
) -> torch.Tensor:
    """Prefer the in-tree CuTeDSL CP boundary kernel, then fall back to FLA."""
    cutedsl_initial_state = try_chunk_gated_delta_rule_fwd_h_pre_process_cutedsl(
        k=k,
        w=w,
        u=u,
        g=g,
        gk=None,
        bg=None,
        v=None,
        chunk_size=_CHUNK_SIZE,
        state_v_first=False,
        cu_seqlens=cu_seqlens,
        context=context,
    )
    if cutedsl_initial_state is not None:
        return cutedsl_initial_state
    return chunk_gated_delta_rule_fwd_h_pre_process(
        k=k, w=w, u=u, g=g, cu_seqlens=cu_seqlens, initial_state=initial_state, context=context
    )


def _cp_backward_preprocess(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None,
    dht: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    context: object,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Prefer the in-tree CuTeDSL CP terminal-gradient kernel, then fall back to FLA."""
    cutedsl_dht = try_chunk_gated_delta_rule_bwd_dhu_pre_process_cutedsl(
        q=q,
        k=k,
        w=w,
        do=do,
        dv=dv,
        g=g,
        gk=None,
        bg=None,
        scale=scale,
        state_v_first=False,
        cu_seqlens=cu_seqlens,
        dht=dht,
        chunk_size=_CHUNK_SIZE,
        context=context,
    )
    if cutedsl_dht is not None:
        return cutedsl_dht, None
    return chunk_gated_delta_rule_bwd_dhu_pre_process(
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
        context=context,
    )
