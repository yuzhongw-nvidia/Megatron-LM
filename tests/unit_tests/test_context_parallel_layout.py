# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.context_parallel_layout import (
    get_cp_rank_partition_indices,
    validate_cp_partition_layout,
)
from megatron.core.models.common.embeddings.rope_utils import get_pos_emb_on_this_cp_rank
from megatron.core.utils import get_pretrain_batch_on_this_cp_rank


class _FakeCPGroup:

    def __init__(self, size: int, rank: int):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


@pytest.mark.parametrize(
    ("layout", "rank", "expected"),
    [
        ("zigzag", 0, [0, 3]),
        ("zigzag", 1, [1, 2]),
        ("contiguous", 0, [0, 1]),
        ("contiguous", 1, [2, 3]),
    ],
)
def test_cp_rank_partition_indices(layout, rank, expected):
    index = get_cp_rank_partition_indices(cp_size=2, cp_rank=rank, cp_partition_layout=layout)
    assert index.tolist() == expected


def test_invalid_cp_partition_layout():
    with pytest.raises(ValueError, match="cp_partition_layout"):
        validate_cp_partition_layout("interleaved")


@pytest.mark.parametrize(
    ("layout", "rank", "expected"),
    [
        ("zigzag", 0, [[0, 1, 2, 3, 12, 13, 14, 15]]),
        ("zigzag", 1, [[4, 5, 6, 7, 8, 9, 10, 11]]),
        ("contiguous", 0, [[0, 1, 2, 3, 4, 5, 6, 7]]),
        ("contiguous", 1, [[8, 9, 10, 11, 12, 13, 14, 15]]),
    ],
)
def test_pretrain_batch_cp_partition_layout(monkeypatch, layout, rank, expected):
    cp_group = _FakeCPGroup(size=2, rank=rank)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: group.size())
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: group.rank())

    batch = {"tokens": torch.arange(16).view(1, 16)}
    out = get_pretrain_batch_on_this_cp_rank(
        batch, cp_group=cp_group, cp_partition_layout=layout
    )

    assert out["tokens"].tolist() == expected


def test_rope_cp_partition_layout():
    cp_group = _FakeCPGroup(size=2, rank=1)
    pos_emb = torch.arange(16).view(16, 1, 1, 1)

    zigzag = get_pos_emb_on_this_cp_rank(pos_emb, seq_dim=0, cp_group=cp_group)
    contiguous = get_pos_emb_on_this_cp_rank(
        pos_emb, seq_dim=0, cp_group=cp_group, cp_partition_layout="contiguous"
    )

    assert zigzag.flatten().tolist() == [4, 5, 6, 7, 8, 9, 10, 11]
    assert contiguous.flatten().tolist() == [8, 9, 10, 11, 12, 13, 14, 15]
