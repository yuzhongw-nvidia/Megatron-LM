# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""File-backed DSv4 MoE router replay debug helper.

This helper is intentionally inert unless ``MCORE_DSV4_ROUTER_REPLAY_MODE`` is
set to ``record`` or ``replay``.  It records the per-router top-k expert indices
for the current microbatch and replays them by matching a hash of valid tokens,
labels, and positions.  The signature-based lookup avoids assuming SBHD and THD
place the same sample on the same data-parallel rank.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction

_CURRENT_SIGNATURE: str | None = None
_CURRENT_ITERATION: int | None = None
_PREPARED_REPLAY = False


def _mode() -> str | None:
    mode = os.environ.get("MCORE_DSV4_ROUTER_REPLAY_MODE", "").strip().lower()
    return mode if mode in {"record", "replay"} else None


def _root_dir() -> Path | None:
    directory = os.environ.get("MCORE_DSV4_ROUTER_REPLAY_DIR", "").strip()
    return Path(directory) if directory else None


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


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    cpu = tensor.detach().contiguous().cpu().reshape(-1)
    return cpu.view(torch.uint8).numpy().tobytes()


def _masked_flat(tensor: torch.Tensor | None, mask: torch.Tensor) -> torch.Tensor | None:
    if tensor is None:
        return None
    flat_tensor = tensor.detach().reshape(-1)
    flat_mask = mask.detach().reshape(-1).to(torch.bool)
    if flat_tensor.numel() != flat_mask.numel():
        return None
    return flat_tensor[flat_mask]


def _batch_signature(
    *,
    tokens: torch.Tensor | None,
    labels: torch.Tensor | None,
    loss_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
) -> tuple[str | None, int]:
    if tokens is None or labels is None or loss_mask is None:
        return None, 0
    valid_mask = loss_mask.detach().reshape(-1).to(torch.bool)
    hasher = hashlib.sha256()
    valid_tokens = _masked_flat(tokens, valid_mask)
    valid_labels = _masked_flat(labels, valid_mask)
    valid_positions = _masked_flat(position_ids, valid_mask)
    for name, tensor in (
        ("tokens", valid_tokens),
        ("labels", valid_labels),
        ("position_ids", valid_positions),
    ):
        if tensor is None:
            continue
        hasher.update(name.encode("utf-8"))
        hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
        hasher.update(str(tensor.dtype).encode("utf-8"))
        hasher.update(_tensor_bytes(tensor))
    return hasher.hexdigest(), int(valid_mask.sum().item())


def _iteration_dir(root: Path, iteration: int) -> Path:
    return root / f"iter_{iteration:07d}"


def _route_path(root: Path, iteration: int, signature: str) -> Path:
    return _iteration_dir(root, iteration) / f"sig_{signature}.pt"


def _events_path(root: Path) -> Path:
    return root / f"rank{_rank():05d}.events.jsonl"


def _write_event(root: Path, payload: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    event = {
        "time": time.time(),
        "rank": _rank(),
        **payload,
    }
    with _events_path(root).open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True, default=str) + "\n")


def _enable_record() -> None:
    RouterReplay.clear_global_indices()
    RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)


def _enable_replay(path: Path) -> None:
    payload = torch.load(path, map_location="cpu")
    topk_indices = payload["topk_indices"]
    RouterReplay.clear_global_indices()
    RouterReplay.set_replay_data(topk_indices)
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)


def prepare_router_replay(
    *,
    tokens: torch.Tensor | None,
    labels: torch.Tensor | None,
    loss_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
) -> None:
    """Prepare router record/replay for the current forward microbatch."""
    global _CURRENT_ITERATION, _CURRENT_SIGNATURE, _PREPARED_REPLAY

    mode = _mode()
    root = _root_dir()
    if mode is None or root is None:
        return

    iteration = _iteration()
    signature, valid_tokens = _batch_signature(
        tokens=tokens,
        labels=labels,
        loss_mask=loss_mask,
        position_ids=position_ids,
    )
    if iteration is None or signature is None:
        _write_event(
            root,
            {
                "event": "skip_prepare",
                "mode": mode,
                "reason": "missing_iteration_or_signature",
                "iteration": iteration,
            },
        )
        return

    _CURRENT_ITERATION = iteration
    _CURRENT_SIGNATURE = signature
    _PREPARED_REPLAY = False

    if mode == "record":
        _enable_record()
        _write_event(
            root,
            {
                "event": "record_prepare",
                "iteration": iteration,
                "signature": signature,
                "valid_tokens": valid_tokens,
                "num_router_instances": len(RouterReplay.global_router_replay_instances),
            },
        )
        return

    path = _route_path(root, iteration, signature)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing router replay file for iteration {iteration}, signature {signature}: {path}"
        )
    _enable_replay(path)
    _PREPARED_REPLAY = True
    _write_event(
        root,
        {
            "event": "replay_prepare",
            "iteration": iteration,
            "signature": signature,
            "valid_tokens": valid_tokens,
            "path": str(path),
            "num_router_instances": len(RouterReplay.global_router_replay_instances),
        },
    )


def save_router_replay_record() -> None:
    """Persist recorded top-k indices for the current forward microbatch."""
    global _CURRENT_ITERATION, _CURRENT_SIGNATURE

    if _mode() != "record":
        if _mode() == "replay" and _PREPARED_REPLAY:
            RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)
        return

    root = _root_dir()
    if root is None or _CURRENT_ITERATION is None or _CURRENT_SIGNATURE is None:
        return

    recorded = RouterReplay.get_recorded_data()
    missing = [idx for idx, tensor in enumerate(recorded) if tensor is None]
    if missing:
        raise RuntimeError(f"Missing recorded router replay tensors for router indices {missing}")

    topk_indices = [tensor.detach().cpu() for tensor in recorded]
    iteration_dir = _iteration_dir(root, _CURRENT_ITERATION)
    iteration_dir.mkdir(parents=True, exist_ok=True)
    path = _route_path(root, _CURRENT_ITERATION, _CURRENT_SIGNATURE)
    tmp_path = path.with_suffix(f".rank{_rank():05d}.tmp")
    torch.save(
        {
            "iteration": _CURRENT_ITERATION,
            "rank": _rank(),
            "signature": _CURRENT_SIGNATURE,
            "topk_indices": topk_indices,
            "shapes": [list(tensor.shape) for tensor in topk_indices],
            "dtypes": [str(tensor.dtype) for tensor in topk_indices],
        },
        tmp_path,
    )
    os.replace(tmp_path, path)
    _write_event(
        root,
        {
            "event": "record_save",
            "iteration": _CURRENT_ITERATION,
            "signature": _CURRENT_SIGNATURE,
            "path": str(path),
            "num_router_tensors": len(topk_indices),
            "shapes": [list(tensor.shape) for tensor in topk_indices],
        },
    )
    RouterReplay.clear_global_router_replay_action()


def cleanup_router_replay() -> None:
    """Clear per-iteration replay state after forward/backward completes."""
    global _CURRENT_ITERATION, _CURRENT_SIGNATURE, _PREPARED_REPLAY
    if _mode() is None:
        return
    RouterReplay.clear_global_router_replay_action()
    RouterReplay.clear_global_indices()
    _CURRENT_ITERATION = None
    _CURRENT_SIGNATURE = None
    _PREPARED_REPLAY = False
