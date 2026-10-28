# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU unit tests for get_cp_slice_for_thd's contiguous CP partitioning.

These cover the numerically load-bearing contiguous equal-split and the
partition_total_tokens tail-padding that the Kimi-K3 cp_partition_mode="contiguous"
packed-THD recipes rely on. The zigzag branch delegates to Transformer Engine and is
covered by the existing THD correctness tests; it is not exercised here. The CP group is
a lightweight stub so the tests run on CPU without a process group.
"""

import pytest
import torch

from megatron.core.datasets.data_schedule_utils import get_cp_slice_for_thd


class _StubCPGroup:
    """Minimal stand-in for a CP process group: only size()/rank() are used."""

    def __init__(self, size, rank):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def _make_batch(total_tokens):
    # One packed document spanning the whole buffer (cu_seqlens_padded[-1] == total_tokens).
    cu = torch.tensor([0, total_tokens], dtype=torch.int32)
    return {
        "tokens": torch.arange(total_tokens, dtype=torch.int64),
        "position_ids": torch.arange(total_tokens, dtype=torch.int64),
        "labels": torch.arange(total_tokens, dtype=torch.int64),
        "loss_mask": torch.ones(total_tokens, dtype=torch.float32),
        "padding_mask": torch.zeros(total_tokens, dtype=torch.bool),
        "cu_seqlens_padded": cu,
    }


@pytest.mark.parametrize("cp_size", [2, 4])
def test_contiguous_equal_split_assigns_one_block_per_rank(cp_size):
    total_tokens = 8 * cp_size
    local = total_tokens // cp_size
    keys = ["tokens", "position_ids", "labels", "loss_mask", "padding_mask"]
    for rank in range(cp_size):
        batch = _make_batch(total_tokens)
        get_cp_slice_for_thd(
            batch, _StubCPGroup(cp_size, rank), keys=keys, cp_partition_mode="contiguous"
        )
        expected = torch.arange(rank * local, (rank + 1) * local, dtype=torch.int64)
        assert batch["tokens"].tolist() == expected.tolist()
        assert batch["labels"].tolist() == expected.tolist()
        assert batch["tokens"].numel() == local


def test_contiguous_pads_to_partition_total_tokens_before_split():
    # Real tokens = 6, target per-rank length 4 over cp_size=2 -> pad to 8 then split.
    cp_size, rank, real = 2, 1, 6
    partition_total_tokens = 8
    batch = _make_batch(real)
    get_cp_slice_for_thd(
        batch,
        _StubCPGroup(cp_size, rank),
        keys=["tokens", "padding_mask"],
        cp_partition_mode="contiguous",
        partition_total_tokens=partition_total_tokens,
    )
    # Rank 1 owns rows [4, 8): tokens 4,5 are real, 6,7 are padding (value 0).
    assert batch["tokens"].tolist() == [4, 5, 0, 0]
    # padding_mask pads with True (padding positions).
    assert batch["padding_mask"].tolist() == [False, False, True, True]


def test_contiguous_requires_divisible_total_tokens():
    batch = _make_batch(7)  # 7 not divisible by cp_size=2
    with pytest.raises(RuntimeError, match="divisible by cp_size"):
        get_cp_slice_for_thd(
            batch, _StubCPGroup(2, 0), keys=["tokens"], cp_partition_mode="contiguous"
        )


def test_cp_size_one_is_a_noop_without_target():
    batch = _make_batch(5)
    before = batch["tokens"].clone()
    get_cp_slice_for_thd(
        batch, _StubCPGroup(1, 0), keys=["tokens"], cp_partition_mode="contiguous"
    )
    assert batch["tokens"].tolist() == before.tolist()
