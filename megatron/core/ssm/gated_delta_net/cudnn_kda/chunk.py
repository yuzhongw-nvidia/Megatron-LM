# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""``chunk_kda_cudnn``: FLA's ``chunk_kda`` call signature on cuDNN's KDA kernels.

``KimiDeltaAttention`` calls its kernel through FLA's keyword surface (see the call in
``kda.py``): ``[B, T, H, D]`` operands, raw gate logits with ``use_gate_in_kernel`` and the
bounded gate options, fp32 post-sigmoid ``beta``, a packed ``cu_seqlens`` under THD and FLA's
``cp_context`` under chunkwise context parallelism.  This entry point accepts the same arguments,
maps them onto cudnn-frontend's native operators (``ops.py``) and returns FLA's
``(output, final_state)`` pair.  Without ``cp_context`` it is cuDNN's autograd op on the flattened
THD batch; with ``cp_context`` it is the chunkwise-CP composition of ``cp.py``.
"""

from __future__ import annotations

from typing import Optional

import torch

from . import ops as _ops
from .cp import chunk_kda_cudnn_cp

HAVE_CUDNN_KDA = _ops.is_available()


def cudnn_kda_unavailable_reason() -> Optional[str]:
    """``None`` when cudnn-frontend >= 1.32.0 with the KDA operators is importable, else why not."""
    return _ops.unavailable_reason()


def chunk_kda_cudnn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    lower_bound: Optional[float] = None,
    state_v_first: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_cpu: Optional[torch.Tensor] = None,
    cp_context=None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    cp_state_producer: str = "cudnn",
    **unused,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """KDA chunked forward (autograd) on cuDNN kernels with FLA's ``chunk_kda`` conventions.

    Args mirror ``fla.ops.kda.chunk_kda``; ``A_log`` / ``dt_bias`` are the gate parameters FLA
    takes through ``**kwargs``.  ``cp_state_producer`` selects how the chunkwise-CP boundary
    states are produced (``"cudnn"``: cuDNN summaries; see ``cp.register_boundary_state_producer``).
    Not supported: ``initial_state`` / ``output_final_state`` together with ``cp_context``
    (FLA rejects them too), ``return_intermediate_states``, fp32 io.
    """
    if unused:
        unsupported = {key for key in unused if unused[key] not in (None, False)}
        if unsupported:
            raise NotImplementedError(
                f"chunk_kda_cudnn: unsupported arguments {sorted(unsupported)}"
            )
    if q.dim() != 4:
        raise ValueError(f"chunk_kda_cudnn expects [B, T, H, D] operands, got {tuple(q.shape)}")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"chunk_kda_cudnn: io dtype must be bf16/fp16, got {q.dtype}")
    B, T, H, K = q.shape
    V = v.shape[-1]
    if cp_context is not None and cu_seqlens is None and B != 1:
        raise ValueError("chunk_kda_cudnn: chunkwise CP needs a single packed sequence (B == 1)")
    if cu_seqlens is not None and B != 1:
        raise ValueError("chunk_kda_cudnn: packed cu_seqlens inputs must have B == 1")

    # THD flattening: a [B, T, ...] batch becomes B sequences of T tokens.
    if cp_context is not None:
        # FLA replaces the caller's cu_seqlens with the rank-local ones of the context (chunk.py);
        # Megatron passes the global packed offsets alongside the context.
        cu = cp_context.cu_seqlens.to(torch.int32)
    elif cu_seqlens is not None:
        cu = cu_seqlens.to(torch.int32)
    else:
        cu = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=q.device)
    thd = lambda t: t.reshape(B * T, t.shape[2], t.shape[3])
    q3, k3, v3 = thd(q), thd(k), thd(v)
    g3, gate_kwargs = _ops.prepare_gate(
        thd(g),
        A_log,
        dt_bias,
        use_gate_in_kernel=use_gate_in_kernel,
        lower_bound=lower_bound,
        heads=g.shape[2],
        head_dim=g.shape[3],
    )
    beta2 = beta.reshape(B * T, beta.shape[2])
    if beta2.dtype not in (torch.float32, q.dtype):
        beta2 = beta2.float()
    scale_value = float(scale) if scale is not None else K**-0.5
    flags = {
        "use_qk_l2norm_in_kernel": bool(use_qk_l2norm_in_kernel),
        "use_beta_sigmoid_in_kernel": bool(use_beta_sigmoid_in_kernel),
        "allow_neg_eigval": bool(allow_neg_eigval),
    }

    if cp_context is not None and getattr(cp_context, "is_cp_enabled", True):
        if initial_state is not None or output_final_state:
            raise NotImplementedError(
                "chunk_kda_cudnn: initial_state / output_final_state are not supported "
                "with cp_context"
            )
        out = chunk_kda_cudnn_cp(
            q3,
            k3,
            v3,
            g3,
            beta2,
            gate_kwargs.pop("a_log", None),
            gate_kwargs.pop("dt_bias", None),
            cu,
            cp_context,
            scale=scale_value,
            flags=flags,
            gate_kwargs=gate_kwargs,
            producer=cp_state_producer,
        )
        return out.reshape(B, T, H, V), None

    ops = _ops.ops()
    result = ops.kimi_delta_attention(
        q3,
        k3,
        v3,
        g3,
        beta2,
        cu,
        scale=scale_value,
        initial_state=_ops.fla_state_to_cudnn(initial_state, state_v_first),
        output_final_state=bool(output_final_state),
        **flags,
        **gate_kwargs,
    )
    out, final_state = result[0], result[1]
    out = out.reshape(B, T, H, V)
    if not output_final_state:
        return out, None
    return out, _ops.cudnn_state_to_fla(final_state, state_v_first)
