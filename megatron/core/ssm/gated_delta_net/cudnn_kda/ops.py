# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Thin access layer over cudnn-frontend's native KDA operators.

cudnn-frontend >= 1.32.0 ships Kimi Delta Attention as graph-API nodes with Python engines
(``cudnn.linear_attention.ops.kda``): the chunked forward/backward (``kimi_delta_attention``,
``kda_fwd``, ``kda_bwd``) and, new in 1.32.0, the two per-span summaries a chunkwise context
parallel needs (``kimi_delta_attention_summary`` / ``kimi_delta_attention_summary_bwd``).  This
module keeps the import lazy (the package must import without cudnn-frontend), checks the
version once, and holds the operand conventions shared by the no-CP and the CP paths:

* THD layout ``[total_tokens, heads, dim]`` with int32 ``cu_seqlens``; a ``[B, T, ...]`` batch
  is flattened to ``B`` sequences of ``T`` tokens.
* io dtype bf16/fp16 for q/k/v (and the output); ``g`` fp32; ``beta`` fp32 or io dtype;
  ``a_log`` fp32 ``[HO]`` and ``dt_bias`` fp32 ``[HO, K]`` (FLA keeps ``dt_bias`` as ``[HO*K]``).
* recurrent states ``[N, HO, V, K]`` (V-major); FLA's layout is ``[N, HV, K, V]``.
"""

from __future__ import annotations

import functools
from typing import Any, Optional

import torch

MIN_CUDNN_FRONTEND = (1, 32, 0)
_REQUIRED_OPS = (
    "kimi_delta_attention",
    "kda_fwd",
    "kda_bwd",
    "kimi_delta_attention_summary",
    "kimi_delta_attention_summary_bwd",
)


def _parse_version(text: str) -> tuple[int, ...]:
    parts = []
    for piece in str(text).split(".")[:3]:
        digits = ""
        for ch in piece:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


@functools.lru_cache(maxsize=1)
def _probe() -> tuple[Any, Optional[str]]:
    """Import cudnn-frontend once; return ``(ops module, None)`` or ``(None, reason)``."""
    try:
        import cudnn  # cudnn-frontend
    except ImportError as e:  # pragma: no cover - environment dependent
        return None, f"cudnn-frontend is not installed ({e})"
    version = getattr(cudnn, "__version__", "0")
    if _parse_version(version) < MIN_CUDNN_FRONTEND:
        return None, (
            f"cudnn-frontend {version} is too old; the KDA backend needs >= "
            f"{'.'.join(str(v) for v in MIN_CUDNN_FRONTEND)} (native KDA and the CP summary ops)"
        )
    try:
        from cudnn.linear_attention.ops import kda as ops
    except ImportError as e:  # pragma: no cover - environment dependent
        return None, f"cudnn-frontend {version} has no linear_attention KDA ops ({e})"
    missing = [name for name in _REQUIRED_OPS if not hasattr(ops, name)]
    if missing:
        return None, f"cudnn-frontend {version} lacks {missing}"
    return ops, None


def unavailable_reason() -> Optional[str]:
    """``None`` when the cuDNN KDA operators can be used, else a human-readable reason."""
    return _probe()[1]


def is_available() -> bool:
    """Whether cudnn-frontend >= 1.32.0 with the native KDA operators is importable."""
    return _probe()[0] is not None


def ops():
    """The ``cudnn.linear_attention.ops.kda`` module; raises ``ImportError`` with the reason."""
    mod, reason = _probe()
    if mod is None:
        raise ImportError(f"cuDNN KDA backend unavailable: {reason}")
    return mod


MIN_CUTE_DSL = (4, 7, 0)


def device_unavailable_reason() -> Optional[str]:
    """``None`` when this GPU and CuTe DSL can run cuDNN's KDA engines, else the reason.

    cudnn-frontend 1.32's KDA engines (``kda_frost`` / ``kda_summary_frost``) serve Blackwell
    GPUs (compute capability 10.x) and need nvidia-cutlass-dsl >= 4.7.0; the other engines serve
    a forward-only subset, so a decline must surface at module construction rather than in the
    first backward.  Returns ``None`` without a CUDA device (CPU-only construction stays possible).
    """
    if not torch.cuda.is_available():
        return None
    major, _minor = torch.cuda.get_device_capability()
    if major != 10:
        return (
            "cuDNN's KDA engines need a Blackwell GPU (compute capability 10.x), "
            f"found compute capability {major}.{_minor}"
        )
    try:
        import cutlass  # nvidia-cutlass-dsl

        dsl_version = getattr(cutlass, "__version__", None)
    except ImportError:
        return "cuDNN's KDA engines need nvidia-cutlass-dsl >= 4.7.0 (CuTe DSL), which is missing"
    if dsl_version is not None and _parse_version(dsl_version) < MIN_CUTE_DSL:
        return f"cuDNN's KDA engines need nvidia-cutlass-dsl >= 4.7.0, found {dsl_version}"
    return None


def prepare_gate(
    g: torch.Tensor,
    A_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    *,
    use_gate_in_kernel: bool,
    lower_bound: Optional[float],
    heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, dict]:
    """Map FLA's gate arguments onto the native op.

    FLA's ``chunk_kda`` takes raw gate logits when ``use_gate_in_kernel`` is set and applies
    ``A_log`` / ``dt_bias`` itself: the bounded ("safe") gate
    ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`` when ``lower_bound`` is given, else
    ``-exp(A_log) * softplus(g + dt_bias)``.  cuDNN fuses the bounded form (``safe_gate`` with
    ``a_log`` / ``dt_bias`` / ``gate_lower_bound``); the softplus form is reproduced here in
    fp32 torch (differentiable), and ``use_gate_in_kernel=False`` means ``g`` already is the
    log decay.  Returns the fp32 ``g`` to hand to cuDNN and the gate keyword arguments.
    ``g`` must be 3-D THD ``[total_tokens, heads, head_dim]``.
    """
    g32 = g.float()
    if not use_gate_in_kernel:
        return g32, {}
    if lower_bound is not None:
        kwargs = {"safe_gate": True, "gate_lower_bound": float(lower_bound)}
        if A_log is not None:
            kwargs["a_log"] = A_log.float().reshape(heads)
        if dt_bias is not None:
            kwargs["dt_bias"] = dt_bias.float().reshape(heads, head_dim)
        return g32, kwargs
    # Unbounded gate: no fused form in cuDNN; mirror the shim and FLA's `fused_kda_gate`.
    if dt_bias is not None:
        g32 = g32 + dt_bias.float().reshape(heads, head_dim)
    g32 = torch.nn.functional.softplus(g32)
    if A_log is not None:
        g32 = -A_log.float().exp().reshape(heads, 1) * g32
    else:
        g32 = -g32
    return g32, {}


def fla_state_to_cudnn(
    state: Optional[torch.Tensor], state_v_first: bool
) -> Optional[torch.Tensor]:
    """FLA ``[N, HV, K, V]`` (``[N, HV, V, K]`` if ``state_v_first``) -> cuDNN ``[N, HO, V, K]``."""
    if state is None:
        return None
    state = state.float()
    return state.contiguous() if state_v_first else state.transpose(-1, -2).contiguous()


def cudnn_state_to_fla(
    state: Optional[torch.Tensor], state_v_first: bool
) -> Optional[torch.Tensor]:
    """Inverse of :func:`fla_state_to_cudnn` (fp32 like FLA's final state)."""
    if state is None or state.numel() == 0:
        return None
    return state.contiguous() if state_v_first else state.transpose(-1, -2).contiguous()
