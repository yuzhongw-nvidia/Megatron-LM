# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""DSA indexer top-k record/replay helpers for DSv4 numerical debug.

This module is inert unless the training helper enables a replay action. It
records/replays DSA indexer top-k indices in call order for the current
microbatch. The file-backed signature matching lives in
``megatron.training.dsv4_indexer_replay_debug``.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

import torch


class DSAIndexerReplayAction(Enum):
    """Actions for DSA indexer top-k replay."""

    RECORD = "record"
    REPLAY_FORWARD = "replay_forward"
    REPLAY_BACKWARD = "replay_backward"


class DSAIndexerReplay:
    """Global call-order replay state for DSA indexer top-k tensors."""

    _action: Optional[DSAIndexerReplayAction] = None
    _target_indices: List[torch.Tensor] = []
    _recorded_indices: List[torch.Tensor] = []
    _replay_forward_pos: int = 0
    _replay_backward_pos: int = 0

    @classmethod
    def set_replay_data(cls, topk_indices: List[torch.Tensor]) -> None:
        cls._target_indices = topk_indices
        cls._replay_forward_pos = 0
        cls._replay_backward_pos = 0

    @classmethod
    def get_recorded_data(cls) -> List[torch.Tensor]:
        return cls._recorded_indices

    @classmethod
    def set_action(cls, action: DSAIndexerReplayAction) -> None:
        cls._action = action
        if action == DSAIndexerReplayAction.RECORD:
            cls._recorded_indices = []
        elif action == DSAIndexerReplayAction.REPLAY_FORWARD:
            cls._replay_forward_pos = 0
        elif action == DSAIndexerReplayAction.REPLAY_BACKWARD:
            cls._replay_backward_pos = 0

    @classmethod
    def clear_action(cls) -> None:
        cls._action = None

    @classmethod
    def clear_indices(cls) -> None:
        cls._target_indices = []
        cls._recorded_indices = []
        cls._replay_forward_pos = 0
        cls._replay_backward_pos = 0

    @classmethod
    def apply(cls, topk_indices: torch.Tensor) -> torch.Tensor:
        """Record or replace one DSA indexer top-k tensor."""
        if cls._action == DSAIndexerReplayAction.RECORD:
            cls._recorded_indices.append(topk_indices.detach())
            return topk_indices

        if cls._action == DSAIndexerReplayAction.REPLAY_FORWARD:
            if cls._replay_forward_pos >= len(cls._target_indices):
                raise RuntimeError(
                    "DSA indexer replay exhausted forward tensors: "
                    f"position={cls._replay_forward_pos}, available={len(cls._target_indices)}"
                )
            target = cls._target_indices[cls._replay_forward_pos]
            cls._replay_forward_pos += 1
            return cls._validate_and_move(target, topk_indices)

        if cls._action == DSAIndexerReplayAction.REPLAY_BACKWARD:
            if cls._replay_backward_pos >= len(cls._target_indices):
                raise RuntimeError(
                    "DSA indexer replay exhausted backward tensors: "
                    f"position={cls._replay_backward_pos}, available={len(cls._target_indices)}"
                )
            target = cls._target_indices[cls._replay_backward_pos]
            cls._replay_backward_pos += 1
            return cls._validate_and_move(target, topk_indices)

        return topk_indices

    @staticmethod
    def _validate_and_move(target: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        target = target.to(device=like.device, dtype=like.dtype)
        if tuple(target.shape) != tuple(like.shape):
            if target.numel() != like.numel():
                raise RuntimeError(
                    "DSA indexer replay tensor shape mismatch: "
                    f"target={tuple(target.shape)} current={tuple(like.shape)}"
                )
            target = target.reshape_as(like)
        return target
