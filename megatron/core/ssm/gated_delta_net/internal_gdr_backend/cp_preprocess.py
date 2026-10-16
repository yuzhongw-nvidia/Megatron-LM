# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CP boundary pre-processing shims: in-tree CuTe DSL kernels first, FLA otherwise.

Both shims take exactly one gate: GDN passes its scalar ``g`` (``[1,T,HV]``),
KDA its per-key ``gk`` (``[1,T,HV,K]`` fp32 chunk-local log2 cumsum).  The gate
is forwarded under its own name to the in-tree hook and to the FLA fallback
alike, so the two paths see the same operands.
"""

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
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    context: object,
) -> torch.Tensor:
    """Prefer the in-tree CuTeDSL CP boundary kernel, then fall back to FLA."""
    if (g is None) == (gk is None):
        raise ValueError("exactly one of g / gk must be given")
    cutedsl_initial_state = try_chunk_gated_delta_rule_fwd_h_pre_process_cutedsl(
        k=k,
        w=w,
        u=u,
        g=g,
        gk=gk,
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
        k=k,
        w=w,
        u=u,
        g=g,
        gk=gk,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
        context=context,
    )


def _cp_backward_preprocess(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    scale: float,
    cu_seqlens: torch.Tensor | None,
    dht: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    context: object,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Prefer the in-tree CuTeDSL CP terminal-gradient kernel, then fall back to FLA."""
    if (g is None) == (gk is None):
        raise ValueError("exactly one of g / gk must be given")
    cutedsl_dht = try_chunk_gated_delta_rule_bwd_dhu_pre_process_cutedsl(
        q=q,
        k=k,
        w=w,
        do=do,
        dv=dv,
        g=g,
        gk=gk,
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
        gk=gk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        dht=dht,
        initial_state=initial_state,
        context=context,
    )
