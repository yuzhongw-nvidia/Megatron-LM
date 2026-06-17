# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.context_parallel_layout import (
    ContextParallelLayout,
    get_context_parallel_layout_chunk_indices,
    get_required_cp_sequence_layout_for_layer,
    get_thd_context_parallel_rank_indices,
)
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_stage_input_cp_sequence_layout,
)


class _PipelineLayout:

    def __init__(self, offset):
        self.offset = offset

    def get_layer_offset(self, **_kwargs):
        return self.offset


def _token_ranges(*spans):
    return [token for start, end in spans for token in range(start, end)]


def test_context_parallel_layout_chunk_indices():
    assert get_context_parallel_layout_chunk_indices(4, 2, "zigzag").tolist() == [2, 5]
    assert get_context_parallel_layout_chunk_indices(4, 2, "contiguous").tolist() == [4, 5]


def test_thd_context_parallel_rank_indices_match_per_sequence_chunk_order():
    cu_seqlens = torch.tensor([0, 16, 40])

    assert get_thd_context_parallel_rank_indices(
        cu_seqlens, 2, 0, "zigzag"
    ).tolist() == _token_ranges((0, 4), (12, 16), (16, 22), (34, 40))
    assert get_thd_context_parallel_rank_indices(
        cu_seqlens, 2, 1, "zigzag"
    ).tolist() == _token_ranges((4, 12), (22, 34))
    assert get_thd_context_parallel_rank_indices(cu_seqlens, 2, 0, "contiguous").tolist() == list(
        range(0, 20)
    )
    assert get_thd_context_parallel_rank_indices(cu_seqlens, 2, 1, "contiguous").tolist() == list(
        range(20, 40)
    )


@pytest.mark.parametrize("layout", ["zigzag", "contiguous"])
def test_thd_context_parallel_rank_indices_cover_all_tokens_once(layout):
    cu_seqlens = torch.tensor([0, 32, 96, 128])
    cp_size = 4

    rank_indices = [
        get_thd_context_parallel_rank_indices(cu_seqlens, cp_size, rank, layout)
        for rank in range(cp_size)
    ]

    assert [indices.numel() for indices in rank_indices] == [32, 32, 32, 32]
    assert torch.cat(rank_indices).sort().values.tolist() == list(range(128))


@pytest.mark.parametrize("layout", ["zigzag", "contiguous"])
def test_thd_context_parallel_rank_indices_ignore_duplicate_boundaries(layout):
    compact_cu_seqlens = torch.tensor([0, 16, 40])
    padded_cu_seqlens = torch.tensor([0, 16, 40, 40, 40])

    for rank in range(2):
        assert torch.equal(
            get_thd_context_parallel_rank_indices(padded_cu_seqlens, 2, rank, layout),
            get_thd_context_parallel_rank_indices(compact_cu_seqlens, 2, rank, layout),
        )


def test_thd_context_parallel_rank_indices_reject_uneven_chunks():
    with pytest.raises(ValueError, match="divisible"):
        get_thd_context_parallel_rank_indices(torch.tensor([0, 10]), 2, 0, "zigzag")


def test_thd_context_parallel_rank_indices_reject_decreasing_boundaries():
    with pytest.raises(ValueError, match="nondecreasing"):
        get_thd_context_parallel_rank_indices(torch.tensor([0, 16, 8]), 2, 0, "zigzag")


def test_thd_context_parallel_rank_indices_reject_unknown_layout():
    with pytest.raises(ValueError, match="Unsupported"):
        get_thd_context_parallel_rank_indices(torch.tensor([0, 16]), 2, 0, "interleaved")


def test_required_layout_rejects_unknown_layer_type():
    with pytest.raises(ValueError, match="Cannot determine CP sequence layout"):
        get_required_cp_sequence_layout_for_layer(object(), SimpleNamespace(cp_comm_type=None))


def test_gated_delta_net_chunkwise_layout_plan_follows_linear_attention_pattern():
    config = SimpleNamespace(
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=2,
        linear_cp_mode="chunkwise",
        num_layers=4,
        pipeline_model_parallel_layout=None,
        pipeline_model_parallel_size=1,
    )

    assert (
        get_experimental_attention_variant_stage_input_cp_sequence_layout(config)
        == ContextParallelLayout.CONTIGUOUS
    )

    config.pipeline_model_parallel_layout = _PipelineLayout(offset=2)
    assert (
        get_experimental_attention_variant_stage_input_cp_sequence_layout(config)
        == ContextParallelLayout.ZIGZAG
    )


def test_gated_delta_net_headwise_layout_plan_preserves_zigzag():
    config = SimpleNamespace(
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=[1, 0],
        linear_cp_mode="headwise",
        num_layers=2,
        pipeline_model_parallel_layout=None,
        pipeline_model_parallel_size=1,
    )

    assert (
        get_experimental_attention_variant_stage_input_cp_sequence_layout(config)
        == ContextParallelLayout.ZIGZAG
    )
