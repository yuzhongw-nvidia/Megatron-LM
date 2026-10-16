# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Backend mode and CuTe DSL capability predicates for the internal GDR backend."""

from __future__ import annotations

import os
from typing import Any

import torch

from .kernels.fused_gdr_cp_cute.backend import _has_module
from .metadata import _CHUNK_SIZE

_BACKEND_ENV = "MCORE_GDN_INTERNAL_BACKEND"
_FUSED_BWD_HEADS = 64
_FUSED_BWD_HEAD_DIM = 128


def _fused_bwd_support_reason(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    A: torch.Tensor | None = None,
    do: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
) -> str | None:
    """Return why the verified latest-main fused backward cannot run.

    The forward calls this with the five input tensors (and ``cu_seqlens``) to decide
    whether to save the fused-backward state; the backward additionally checks ``A``,
    ``do`` and ``dht``, which only exist at backward time.
    """
    if q.dtype != torch.bfloat16:
        return "fused backward requires bf16 inputs"
    activations = (k, v) if do is None else (k, v, do)
    if any(tensor.dtype != q.dtype for tensor in activations):
        return "q, k, v, and do dtypes differ"
    if q.ndim != 4 or tuple(q.shape[2:]) != (_FUSED_BWD_HEADS, _FUSED_BWD_HEAD_DIM):
        return "fused backward requires BTHD with H=64 and D=128"
    if any(tensor.shape != q.shape for tensor in activations):
        return "q, k, v, and do shapes differ"
    if any(not tensor.is_contiguous() for tensor in (q, *activations)):
        return "q, k, v, and do must be contiguous"
    scalar_shape = q.shape[:-1]
    if g.shape != scalar_shape or beta.shape != scalar_shape:
        return "g and beta must match q's BTH shape"
    if A is not None and (A.dtype != torch.bfloat16 or A.shape != (*scalar_shape, _CHUNK_SIZE)):
        return "A must be bf16 with shape BTH64"
    if cu_seqlens is None:
        num_sequences = q.shape[0]
        if num_sequences < 1:
            return "dense fused backward requires a positive batch size"
        if q.shape[1] < 1:
            return "dense fused backward requires a positive sequence length"
    else:
        if q.shape[0] != 1:
            return "packed fused backward requires batch size 1"
        if (
            cu_seqlens.dtype != torch.int32
            or not cu_seqlens.is_contiguous()
            or cu_seqlens.numel() < 2
        ):
            return "cu_seqlens must be contiguous int32 with at least two offsets"
        num_sequences = cu_seqlens.numel() - 1
    if dht is not None and (
        dht.dtype != torch.float32
        or dht.shape != (num_sequences, _FUSED_BWD_HEADS, _FUSED_BWD_HEAD_DIM, _FUSED_BWD_HEAD_DIM)
    ):
        return f"dht must be float32 with shape ({num_sequences}, 64, 128, 128)"
    return None


def _backend_mode() -> str:
    mode = os.environ.get(_BACKEND_ENV, "auto").lower()
    if mode not in {"auto", "cute", "fla"}:
        raise ValueError(f"{_BACKEND_ENV} must be auto|cute|fla, got {mode!r}")
    return mode


def _cutedsl_support_reason(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool,
    use_beta_sigmoid_in_kernel: bool,
    allow_neg_eigval: bool,
    state_v_first: bool,
    cu_seqlens: torch.Tensor | None,
    cp_context: object | None,
    kwargs: dict[str, Any],
) -> str | None:
    if not _has_module("cutlass") or not _has_module("cuda.bindings.driver"):
        return "CuTe DSL runtime is not installed"
    if not q.is_cuda:
        return "inputs are not CUDA tensors"
    if any(tensor.device != q.device for tensor in (k, v, g, beta)):
        return "q, k, v, g, and beta must be on the same CUDA device"
    if not torch.version.cuda or int(torch.version.cuda.split(".", 1)[0]) < 13:
        return "CUDA 13 or newer is required"
    if torch.cuda.get_device_capability(q.device)[0] != 10:
        return "device is not SM100"
    if q.dtype not in (torch.bfloat16, torch.float16):
        return f"dtype {q.dtype} is not supported"
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return "q, k, and v dtypes differ"
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return "q, k, and v must use BTHD layout"
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        return "q, k, and v must be contiguous"
    if q.shape[:2] != k.shape[:2] or q.shape[:2] != v.shape[:2]:
        return "q, k, and v token dimensions differ"
    if q.shape[2] != k.shape[2] or q.shape[2] != v.shape[2]:
        return "grouped value attention is not yet enabled for the CuTe path"
    if q.shape[-1] != 128 or k.shape[-1] != 128 or v.shape[-1] != 128:
        return "head dimensions must all equal 128"
    if tuple(g.shape) != tuple(beta.shape) or tuple(g.shape) != tuple(q.shape[:3]):
        return "g and beta shapes must match q's BTH dimensions"
    if initial_state is not None or output_final_state:
        return "initial or final state is not yet supported"
    if use_qk_l2norm_in_kernel:
        return "in-kernel QK L2 normalization is not supported"
    if use_beta_sigmoid_in_kernel or allow_neg_eigval:
        return "in-kernel beta transformation is not supported"
    if state_v_first:
        return "V-first state layout is not supported"
    if (
        cp_context is not None
        and _fused_bwd_support_reason(q=q, k=k, v=v, g=g, beta=beta, cu_seqlens=cu_seqlens)
        is not None
    ):
        return "context parallel fused backward requires bf16 BTHD with H=64 and D=128"
    extra_kwargs = dict(kwargs)
    if extra_kwargs.get("use_gate_in_kernel", False):
        return "in-kernel gate activation is not supported"
    for name in ("use_gate_in_kernel", "head_first", "use_cute"):
        if extra_kwargs.get(name) is False:
            extra_kwargs.pop(name)
    for name in ("A_log", "dt_bias", "g_input"):
        if extra_kwargs.get(name) is None:
            extra_kwargs.pop(name, None)
    if extra_kwargs:
        return f"extra options are not supported: {sorted(extra_kwargs)}"
    if cp_context is None and cu_seqlens is not None and q.shape[0] != 1:
        return "packed variable length inputs require batch size 1"
    return None
