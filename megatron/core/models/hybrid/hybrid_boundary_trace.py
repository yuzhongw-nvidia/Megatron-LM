# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Opt-in tensor sidecars for narrowing hybrid-stack TP parity failures.

The tracer is intentionally independent of the training logger.  It is enabled
only when ``MCORE_HYBRID_BOUNDARY_TRACE_PATH`` is set and writes at most the
configured number of occurrences for selected ranks, decoder layers, and
boundary stages.  This keeps full-size diagnostic runs bounded while allowing
offline TP1/TP2 tensor reconstruction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

_ALLOWED_STAGES = frozenset({"incoming", "aggregated", "new_partial"})


@dataclass(frozen=True)
class _TraceConfig:
    output_dir: Path
    global_ranks: frozenset[int]
    decoder_layers: frozenset[int]
    stages: frozenset[str]
    max_occurrences: int


_CONFIG_INITIALIZED = False
_CONFIG: Optional[_TraceConfig] = None
_COUNTS: dict[tuple[int, int, str], int] = {}


def _parse_nonnegative_int_set(value: str, name: str) -> frozenset[int]:
    try:
        result = frozenset(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError(f"{name} must contain comma-separated integers, got {value!r}") from error
    if not result or any(item < 0 for item in result):
        raise ValueError(f"{name} must contain non-negative integers, got {value!r}")
    return result


def _load_config() -> Optional[_TraceConfig]:
    output_path = os.environ.get("MCORE_HYBRID_BOUNDARY_TRACE_PATH", "").strip()
    if not output_path:
        return None

    ranks = _parse_nonnegative_int_set(
        os.environ.get("MCORE_HYBRID_BOUNDARY_TRACE_GLOBAL_RANKS", "0"),
        "MCORE_HYBRID_BOUNDARY_TRACE_GLOBAL_RANKS",
    )
    layers = _parse_nonnegative_int_set(
        os.environ.get("MCORE_HYBRID_BOUNDARY_TRACE_DECODER_LAYERS", "0,1,2,3"),
        "MCORE_HYBRID_BOUNDARY_TRACE_DECODER_LAYERS",
    )
    stages = frozenset(
        item.strip()
        for item in os.environ.get(
            "MCORE_HYBRID_BOUNDARY_TRACE_STAGES", "incoming,aggregated,new_partial"
        ).split(",")
        if item.strip()
    )
    unknown_stages = stages - _ALLOWED_STAGES
    if not stages or unknown_stages:
        raise ValueError(
            "MCORE_HYBRID_BOUNDARY_TRACE_STAGES must be a non-empty subset of "
            f"{sorted(_ALLOWED_STAGES)}, got {sorted(stages)}"
        )
    max_occurrences_text = os.environ.get("MCORE_HYBRID_BOUNDARY_TRACE_MAX_OCCURRENCES", "1")
    try:
        max_occurrences = int(max_occurrences_text)
    except ValueError as error:
        raise ValueError(
            "MCORE_HYBRID_BOUNDARY_TRACE_MAX_OCCURRENCES must be a positive integer, "
            f"got {max_occurrences_text!r}"
        ) from error
    if max_occurrences <= 0:
        raise ValueError(
            "MCORE_HYBRID_BOUNDARY_TRACE_MAX_OCCURRENCES must be positive, "
            f"got {max_occurrences}"
        )
    return _TraceConfig(
        output_dir=Path(output_path),
        global_ranks=ranks,
        decoder_layers=layers,
        stages=stages,
        max_occurrences=max_occurrences,
    )


def _get_config() -> Optional[_TraceConfig]:
    global _CONFIG_INITIALIZED, _CONFIG
    if not _CONFIG_INITIALIZED:
        _CONFIG = _load_config()
        _CONFIG_INITIALIZED = True
    return _CONFIG


def reset_hybrid_boundary_trace_state() -> None:
    """Clear lazy environment parsing and occurrence counts (primarily for tests)."""
    global _CONFIG_INITIALIZED, _CONFIG
    _CONFIG_INITIALIZED = False
    _CONFIG = None
    _COUNTS.clear()


def record_hybrid_boundary(layer: int, stage: str, tensor: torch.Tensor) -> Optional[Path]:
    """Persist one selected decoder boundary tensor and return its path."""
    if layer < 0:
        raise ValueError(f"layer must be non-negative, got {layer}")
    if stage not in _ALLOWED_STAGES:
        raise ValueError(f"unsupported hybrid boundary stage {stage!r}")
    if not torch.is_tensor(tensor):
        raise TypeError(f"hybrid boundary value must be a tensor, got {type(tensor).__name__}")

    config = _get_config()
    if config is None:
        return None
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if (
        rank not in config.global_ranks
        or layer not in config.decoder_layers
        or stage not in config.stages
    ):
        return None
    if tensor.is_cuda and torch.cuda.is_current_stream_capturing():
        return None

    key = (rank, layer, stage)
    occurrence = _COUNTS.get(key, 0)
    if occurrence >= config.max_occurrences:
        return None

    stored = tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    payload = {
        "rank": rank,
        "layer": layer,
        "stage": stage,
        "occurrence": occurrence,
        "original_dtype": str(tensor.dtype).removeprefix("torch."),
        "stored_dtype": "bfloat16",
        "shape": list(stored.shape),
        "tensor": stored,
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / (
        f"hybrid_boundary_rank{rank}_layer{layer}_{stage}_occ{occurrence}.pt"
    )
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, output_path)
    _COUNTS[key] = occurrence + 1
    return output_path


def load_hybrid_boundary_trace(path: Path | str) -> dict:
    """Load and validate one sidecar produced by :func:`record_hybrid_boundary`."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    tensor = payload.get("tensor")
    if not torch.is_tensor(tensor):
        raise ValueError(f"{path}: sidecar does not contain a tensor")
    if payload.get("stored_dtype") != "bfloat16" or tensor.dtype != torch.bfloat16:
        raise ValueError(f"{path}: unsupported stored dtype {payload.get('stored_dtype')!r}")
    if list(tensor.shape) != payload.get("shape"):
        raise ValueError(f"{path}: tensor shape does not match sidecar metadata")
    return payload
