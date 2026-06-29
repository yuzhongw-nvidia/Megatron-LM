# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Router-internal tensor dumping for DSv4 THD/SBHD numerical debug.

This module is inert unless ``MCORE_DSV4_ROUTER_INTERNAL_DEBUG`` is enabled.
Training code enables it for selected iterations, and router code calls
``dump_router_internal_tensors`` at specific internal boundaries.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import torch


def _truthy(value: str | None) -> bool:
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


def router_internal_logging_enabled() -> bool:
    return _truthy(os.environ.get("MCORE_DSV4_ROUTER_INTERNAL_DEBUG"))


def should_log_router_internal(iteration: int) -> bool:
    if not router_internal_logging_enabled():
        return False
    iterations = os.environ.get("MCORE_DSV4_ROUTER_INTERNAL_ITERATIONS", "").strip()
    if iterations:
        requested = {int(item.strip()) for item in iterations.split(",") if item.strip()}
        return iteration in requested
    interval = int(os.environ.get("MCORE_DSV4_ROUTER_INTERNAL_INTERVAL", "0") or "0")
    return True if interval <= 0 else iteration % interval == 0


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", "0"))


def _parse_layers() -> set[int] | None:
    layers = os.environ.get("MCORE_DSV4_ROUTER_INTERNAL_LAYERS", "").strip()
    if not layers:
        return None
    return {int(item.strip()) for item in layers.split(",") if item.strip()}


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


class _RouterInternalLogger:
    def __init__(self, save_dir: str, iteration: int) -> None:
        root = os.environ.get("MCORE_DSV4_ROUTER_INTERNAL_SAVE_DIR", "").strip()
        self._save_dir = Path(root or save_dir)
        self._iteration = iteration
        self._rank_name = f"rank{_rank():05d}"
        self._layers = _parse_layers()
        self._tensor_names = {
            item.strip()
            for item in os.environ.get("MCORE_DSV4_ROUTER_INTERNAL_TENSOR_NAMES", "").split(",")
            if item.strip()
        }
        self._call_counts: dict[tuple[int, str], int] = {}

    def _should_dump_layer(self, layer_index: int) -> bool:
        return self._layers is None or layer_index in self._layers

    def dump(
        self,
        layer_number: int | None,
        route_type: str,
        tensors: dict[str, torch.Tensor | None],
    ) -> None:
        layer_index = -1 if layer_number is None else layer_number - 1
        if not self._should_dump_layer(layer_index):
            return

        key = (layer_index, route_type)
        call_index = self._call_counts.get(key, 0)
        self._call_counts[key] = call_index + 1

        tensor_dir = (
            self._save_dir
            / "router_internal_tensors"
            / f"iter_{self._iteration:07d}"
            / self._rank_name
        )
        event_dir = self._save_dir / "router_internal" / f"iter_{self._iteration:07d}"
        tensor_dir.mkdir(parents=True, exist_ok=True)
        event_dir.mkdir(parents=True, exist_ok=True)

        tensor_rows: list[dict[str, Any]] = []
        for name, tensor in tensors.items():
            if tensor is None:
                continue
            if self._tensor_names and name not in self._tensor_names:
                continue
            file_name = (
                f"layer{layer_index:02d}_{_safe_name(route_type)}_"
                f"call{call_index:04d}_{_safe_name(name)}.pt"
            )
            torch.save(tensor.detach().cpu(), tensor_dir / file_name)
            tensor_rows.append(
                {
                    "name": name,
                    "dtype": str(tensor.dtype),
                    "shape": list(tensor.shape),
                    "tensor_file": file_name,
                }
            )

        event = {
            "iteration": self._iteration,
            "rank": self._rank_name,
            "module": f"decoder.layers.{layer_index}.mlp.router_internal",
            "layer_index": layer_index,
            "route_type": route_type,
            "call_index": call_index,
            "tensors": tensor_rows,
        }
        with (event_dir / f"{self._rank_name}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")


_CURRENT_LOGGER: _RouterInternalLogger | None = None


def enable_router_internal_logging(save_dir: str, iteration: int) -> None:
    global _CURRENT_LOGGER
    if not router_internal_logging_enabled():
        return
    _CURRENT_LOGGER = _RouterInternalLogger(save_dir, iteration)


def disable_router_internal_logging() -> None:
    global _CURRENT_LOGGER
    _CURRENT_LOGGER = None


def dump_router_internal_tensors(
    layer_number: int | None,
    route_type: str,
    tensors: dict[str, torch.Tensor | None],
) -> None:
    logger = _CURRENT_LOGGER
    if logger is None:
        return
    logger.dump(layer_number, route_type, tensors)
