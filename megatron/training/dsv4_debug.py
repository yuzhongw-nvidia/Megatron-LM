# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Lightweight DSv4 THD/SBHD debug logging.

This module is inert unless ``MCORE_DSV4_DEBUG=1`` is set. It records compact
JSONL tensor signatures around the GPT training batch/loss path so THD and SBHD
runs can be compared without dumping full input tensors.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch


def _enabled() -> bool:
    return os.environ.get("MCORE_DSV4_DEBUG", "0").lower() in {"1", "true", "yes", "on"}


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", "0"))


def _iteration() -> int | None:
    try:
        from megatron.training import get_args

        curr = getattr(get_args(), "curr_iteration", None)
        return None if curr is None else int(curr) + 1
    except Exception:
        return None


def _debug_dir() -> Path | None:
    if not _enabled():
        return None
    override = os.environ.get("MCORE_DSV4_DEBUG_DIR")
    if override:
        return Path(override)
    try:
        from megatron.training import get_args

        save_dir = getattr(get_args(), "save", None)
    except Exception:
        save_dir = None
    if not save_dir:
        return None
    return Path(save_dir) / "dsv4_debug"


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def _tensor_hash(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().contiguous().cpu().reshape(-1)
    return hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()


def _tensor_summary(tensor: torch.Tensor | None, *, with_hash: bool = True) -> dict[str, Any] | None:
    if tensor is None:
        return None
    detached = tensor.detach()
    summary: dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": int(detached.numel()),
    }
    if detached.numel() == 0:
        return summary

    flat = detached.reshape(-1)
    if detached.is_floating_point() or detached.is_complex():
        values = detached.float()
        summary.update(
            {
                "sum": float(values.sum().item()),
                "mean": float(values.mean().item()),
                "std": float(values.std(unbiased=False).item()),
                "min": float(values.min().item()),
                "max": float(values.max().item()),
                "abs_max": float(values.abs().max().item()),
                "l2": float(torch.linalg.vector_norm(values).item()),
                "nan_count": int(torch.isnan(values).sum().item()),
                "inf_count": int(torch.isinf(values).sum().item()),
                "finite_count": int(torch.isfinite(values).sum().item()),
            }
        )
    else:
        values = flat.to(torch.int64)
        summary.update(
            {
                "sum": int(values.sum().item()),
                "min": int(values.min().item()),
                "max": int(values.max().item()),
            }
        )
    sample_count = min(16, flat.numel())
    summary["first"] = _jsonable(flat[:sample_count].detach().cpu().tolist())
    summary["last"] = _jsonable(flat[-sample_count:].detach().cpu().tolist())
    if with_hash and detached.numel() <= 4_000_000:
        summary["sha256"] = _tensor_hash(detached)
    return summary


def _packed_seq_summary(packed_seq_params: Any) -> dict[str, Any] | None:
    if packed_seq_params is None:
        return None
    fields = [
        "qkv_format",
        "max_seqlen_q",
        "max_seqlen_kv",
        "cu_seqlens_q",
        "cu_seqlens_kv",
        "cu_seqlens_q_padded",
        "cu_seqlens_kv_padded",
        "local_cp_size",
    ]
    out: dict[str, Any] = {}
    for field in fields:
        if not hasattr(packed_seq_params, field):
            continue
        value = getattr(packed_seq_params, field)
        out[field] = _tensor_summary(value) if isinstance(value, torch.Tensor) else _jsonable(value)
    return out


def _write(kind: str, payload: dict[str, Any]) -> None:
    directory = _debug_dir()
    if directory is None:
        return
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "time": time.time(),
        "rank": _rank(),
        "iteration": _iteration(),
        **payload,
    }
    path = directory / f"rank{_rank():05d}.{kind}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(payload), sort_keys=True) + "\n")


def log_batch(
    *,
    tokens: torch.Tensor | None,
    labels: torch.Tensor | None,
    loss_mask: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    packed_seq_params: Any,
    padding_mask: torch.Tensor | None,
) -> None:
    if not _enabled():
        return
    _write(
        "batch",
        {
            "event": "forward_step_batch",
            "tokens": _tensor_summary(tokens),
            "labels": _tensor_summary(labels),
            "loss_mask": _tensor_summary(loss_mask),
            "attention_mask": _tensor_summary(attention_mask, with_hash=False),
            "position_ids": _tensor_summary(position_ids),
            "padding_mask": _tensor_summary(padding_mask),
            "packed_seq_params": _packed_seq_summary(packed_seq_params),
        },
    )


def log_forward_output(output_tensor: torch.Tensor | None) -> None:
    if not _enabled():
        return
    _write(
        "forward",
        {
            "event": "forward_step_output",
            "output_tensor": _tensor_summary(output_tensor),
        },
    )


def log_loss(
    *,
    loss_mask: torch.Tensor | None,
    output_tensor: torch.Tensor | None,
    loss: torch.Tensor,
    num_tokens: torch.Tensor,
) -> None:
    if not _enabled():
        return
    _write(
        "loss",
        {
            "event": "loss_func",
            "loss_mask": _tensor_summary(loss_mask),
            "output_tensor": _tensor_summary(output_tensor),
            "loss": _tensor_summary(loss),
            "num_tokens": _tensor_summary(num_tokens),
        },
    )
