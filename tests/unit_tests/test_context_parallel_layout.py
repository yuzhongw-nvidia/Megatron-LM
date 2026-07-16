# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace
from typing import List, Tuple

import pytest
import torch

import megatron.core.context_parallel_layout as context_parallel_layout
from megatron.core.context_parallel_layout import (
    build_thd_cp_partition_route,
    build_cp_partition_mode_plan,
    convert_cp_partition_mode_nested,
    get_context_parallel_layout_chunk_indices,
    get_or_build_thd_cp_partition_route,
    get_required_cp_partition_mode_for_layer,
    get_thd_context_parallel_rank_indices,
)
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_stage_input_cp_partition_mode,
)
from megatron.core.models.hybrid.hybrid_layer_allocation import (
    get_hybrid_stage_input_cp_partition_mode,
)


class _PipelineLayout:

    def __init__(self, offset):
        self.offset = offset

    def get_layer_offset(self, **_kwargs):
        return self.offset


class IdentityOp:
    pass


class GatedDeltaNet:
    pass


class _FakeGroup:

    def __init__(self, size, rank):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def _token_ranges(*spans):
    return [token for start, end in spans for token in range(start, end)]


_CpuThdLayoutSegment = Tuple[int, int, int]


def _cpu_compact_thd_cu_seqlens_to_list(cu_seqlens: torch.Tensor) -> List[int]:
    if cu_seqlens.dim() != 1:
        raise ValueError(f"cu_seqlens must be 1-D, got shape {tuple(cu_seqlens.shape)}.")

    cu = cu_seqlens.detach().to(device="cpu", dtype=torch.long).tolist()
    if not cu or cu[0] != 0:
        raise ValueError(f"cu_seqlens must start at 0, got {cu_seqlens}.")

    compact_cu: List[int] = [cu[0]]
    prev = cu[0]
    for value in cu[1:]:
        if value < prev:
            raise ValueError(f"cu_seqlens must be nondecreasing, got {cu_seqlens}.")
        if value != prev:
            compact_cu.append(value)
        prev = value
    return compact_cu


def _cpu_validate_thd_route_partitioning(cu: List[int], cp_size: int) -> None:
    total_tokens = cu[-1]
    if total_tokens % cp_size != 0:
        raise ValueError(
            f"Contiguous CP partitioning requires total_tokens={total_tokens} "
            f"to be divisible by cp_size={cp_size}."
        )

    chunk_divisor = 2 * cp_size
    bad_seq_lens = [
        seq_end - seq_start
        for seq_start, seq_end in zip(cu[:-1], cu[1:])
        if (seq_end - seq_start) % chunk_divisor != 0
    ]
    if bad_seq_lens:
        raise ValueError(
            "All packed sequence lengths must be divisible by "
            f"2 * cp_size ({chunk_divisor}) for zigzag CP layout conversion, "
            f"got {bad_seq_lens}."
        )


def _cpu_build_thd_layout_segments(
    cu: List[int],
    cp_size: int,
    cp_rank: int,
    cp_partition_mode: str,
) -> Tuple[List[_CpuThdLayoutSegment], int]:
    total_tokens = cu[-1]
    if cp_partition_mode == "contiguous":
        part_len = total_tokens // cp_size
        if part_len == 0:
            return [], 0
        return [(cp_rank * part_len, part_len, 0)], part_len

    if cp_partition_mode != "zigzag":
        raise ValueError(f"Unsupported context-parallel partition mode {cp_partition_mode!r}.")

    segments: List[_CpuThdLayoutSegment] = []
    local_start = 0
    for seq_start, seq_end in zip(cu[:-1], cu[1:]):
        seq_len = seq_end - seq_start
        chunk_len = seq_len // (2 * cp_size)
        first_chunk = cp_rank
        second_chunk = 2 * cp_size - cp_rank - 1
        segments.append((seq_start + first_chunk * chunk_len, chunk_len, local_start))
        segments.append((seq_start + second_chunk * chunk_len, chunk_len, local_start + chunk_len))
        local_start += 2 * chunk_len

    return segments, local_start


def _cpu_intersect_thd_layout_segments(
    source_segments: List[_CpuThdLayoutSegment],
    target_segments: List[_CpuThdLayoutSegment],
) -> List[Tuple[int, int, int]]:
    intersections: List[Tuple[int, int, int]] = []
    source_index = 0
    target_index = 0
    while source_index < len(source_segments) and target_index < len(target_segments):
        source_global_start, source_len, source_local_start = source_segments[source_index]
        target_global_start, target_len, target_local_start = target_segments[target_index]
        source_global_end = source_global_start + source_len
        target_global_end = target_global_start + target_len

        overlap_start = max(source_global_start, target_global_start)
        overlap_end = min(source_global_end, target_global_end)
        if overlap_start < overlap_end:
            intersections.append(
                (
                    source_local_start + overlap_start - source_global_start,
                    target_local_start + overlap_start - target_global_start,
                    overlap_end - overlap_start,
                )
            )

        if source_global_end <= target_global_end:
            source_index += 1
        else:
            target_index += 1

    return intersections


def _cpu_append_range(rows: List[int], start: int, length: int) -> None:
    rows.extend(range(start, start + length))


def _cpu_row_list_is_identity(rows: List[int]) -> bool:
    return all(row == index for index, row in enumerate(rows))


def _cpu_row_list_to_tensor(rows: List[int]) -> torch.Tensor:
    if not rows:
        return torch.empty(0, dtype=torch.long)
    return torch.tensor(rows, dtype=torch.long)


def _cpu_build_thd_cp_partition_route(
    cu_seqlens: torch.Tensor,
    cp_size: int,
    cp_rank: int,
    source_partition_mode: str,
    target_partition_mode: str,
):
    cu = _cpu_compact_thd_cu_seqlens_to_list(cu_seqlens)
    _cpu_validate_thd_route_partitioning(cu, cp_size)

    source_segments_by_rank: List[List[_CpuThdLayoutSegment]] = []
    source_lengths: List[int] = []
    target_segments_by_rank: List[List[_CpuThdLayoutSegment]] = []
    target_lengths: List[int] = []
    for rank in range(cp_size):
        source_segments, source_length = _cpu_build_thd_layout_segments(
            cu, cp_size, rank, source_partition_mode
        )
        target_segments, target_length = _cpu_build_thd_layout_segments(
            cu, cp_size, rank, target_partition_mode
        )
        source_segments_by_rank.append(source_segments)
        source_lengths.append(source_length)
        target_segments_by_rank.append(target_segments)
        target_lengths.append(target_length)

    local_source_segments = source_segments_by_rank[cp_rank]
    local_target_segments = target_segments_by_rank[cp_rank]

    send_rows_list: List[int] = []
    input_split_sizes: List[int] = []
    for dst_rank in range(cp_size):
        intersections = _cpu_intersect_thd_layout_segments(
            local_source_segments, target_segments_by_rank[dst_rank]
        )
        intersections.sort(key=lambda item: item[1])
        input_split_size = 0
        for source_row, _, length in intersections:
            _cpu_append_range(send_rows_list, source_row, length)
            input_split_size += length
        input_split_sizes.append(input_split_size)

    recv_rows_list: List[int] = []
    output_split_sizes: List[int] = []
    for src_rank in range(cp_size):
        intersections = _cpu_intersect_thd_layout_segments(
            source_segments_by_rank[src_rank], local_target_segments
        )
        intersections.sort(key=lambda item: item[1])
        output_split_size = 0
        for _, target_row, length in intersections:
            _cpu_append_range(recv_rows_list, target_row, length)
            output_split_size += length
        output_split_sizes.append(output_split_size)

    return SimpleNamespace(
        local_source_length=source_lengths[cp_rank],
        local_target_length=target_lengths[cp_rank],
        send_rows=_cpu_row_list_to_tensor(send_rows_list),
        recv_rows=_cpu_row_list_to_tensor(recv_rows_list),
        input_split_sizes=input_split_sizes,
        output_split_sizes=output_split_sizes,
        send_rows_are_identity=_cpu_row_list_is_identity(send_rows_list),
        recv_rows_are_identity=_cpu_row_list_is_identity(recv_rows_list),
    )


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


def test_thd_contiguous_rank_indices_allow_uneven_sequence_lengths():
    cu_seqlens = torch.tensor([0, 10, 18])

    assert get_thd_context_parallel_rank_indices(cu_seqlens, 2, 0, "contiguous").tolist() == list(
        range(0, 9)
    )
    assert get_thd_context_parallel_rank_indices(cu_seqlens, 2, 1, "contiguous").tolist() == list(
        range(9, 18)
    )


@pytest.mark.parametrize(
    ("source_layout", "target_layout"), [("zigzag", "contiguous"), ("contiguous", "zigzag")]
)
@pytest.mark.parametrize(
    ("cu_seqlens", "cp_size"),
    [
        (torch.tensor([0, 16, 40]), 2),
        (torch.tensor([0, 32, 96, 128]), 4),
        (torch.tensor([0, 32, 96, 128, 128, 128]), 4),
    ],
)
def test_thd_cp_partition_route_reassembles_target_layout(
    source_layout, target_layout, cu_seqlens, cp_size
):
    source_indices = [
        get_thd_context_parallel_rank_indices(cu_seqlens, cp_size, rank, source_layout)
        for rank in range(cp_size)
    ]
    target_indices = [
        get_thd_context_parallel_rank_indices(cu_seqlens, cp_size, rank, target_layout)
        for rank in range(cp_size)
    ]
    routes = [
        build_thd_cp_partition_route(cu_seqlens, cp_size, rank, source_layout, target_layout)
        for rank in range(cp_size)
    ]
    for route in routes:
        assert route.send_rows_are_identity == torch.equal(
            route.send_rows, torch.arange(route.send_rows.numel(), dtype=route.send_rows.dtype)
        )
        assert route.recv_rows_are_identity == torch.equal(
            route.recv_rows, torch.arange(route.recv_rows.numel(), dtype=route.recv_rows.dtype)
        )
    send_buffers = [
        source_indices[rank].index_select(0, routes[rank].send_rows) for rank in range(cp_size)
    ]

    for dst_rank in range(cp_size):
        recv_chunks = []
        for src_rank in range(cp_size):
            src_route = routes[src_rank]
            send_offset = sum(src_route.input_split_sizes[:dst_rank])
            send_len = src_route.input_split_sizes[dst_rank]
            recv_chunks.append(send_buffers[src_rank].narrow(0, send_offset, send_len))
        recv_buf = torch.cat(recv_chunks, dim=0)
        out = torch.empty(routes[dst_rank].local_target_length, dtype=recv_buf.dtype)
        out.index_copy_(0, routes[dst_rank].recv_rows, recv_buf)
        assert torch.equal(out, target_indices[dst_rank])


@pytest.mark.parametrize(
    ("source_layout", "target_layout"), [("zigzag", "contiguous"), ("contiguous", "zigzag")]
)
@pytest.mark.parametrize(
    ("cu_seqlens", "cp_size"),
    [
        (torch.tensor([0, 16, 40]), 2),
        (torch.tensor([0, 32, 96, 128]), 4),
        (torch.tensor([0, 32, 96, 128, 128, 128]), 4),
        (torch.tensor([0, 64, 192, 256]), 8),
        (torch.tensor([0]), 4),
    ],
)
def test_thd_cp_partition_route_matches_cpu_range_oracle(
    source_layout, target_layout, cu_seqlens, cp_size
):
    for rank in range(cp_size):
        actual = build_thd_cp_partition_route(
            cu_seqlens, cp_size, rank, source_layout, target_layout
        )
        expected = _cpu_build_thd_cp_partition_route(
            cu_seqlens, cp_size, rank, source_layout, target_layout
        )

        assert actual.local_source_length == expected.local_source_length
        assert actual.local_target_length == expected.local_target_length
        assert actual.input_split_sizes == expected.input_split_sizes
        assert actual.output_split_sizes == expected.output_split_sizes
        assert actual.send_rows_are_identity == expected.send_rows_are_identity
        assert actual.recv_rows_are_identity == expected.recv_rows_are_identity
        assert torch.equal(actual.send_rows.cpu(), expected.send_rows)
        assert torch.equal(actual.recv_rows.cpu(), expected.recv_rows)


def test_thd_cp_partition_route_cache_reuses_same_microbatch_route():
    packed_seq_params = SimpleNamespace(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 16, 40]),
        cu_seqlens_q_padded=None,
        cp_partition_route_cache=None,
    )
    cp_group = _FakeGroup(size=2, rank=0)

    route = get_or_build_thd_cp_partition_route(
        packed_seq_params, cp_group, "zigzag", "contiguous"
    )
    same_route = get_or_build_thd_cp_partition_route(
        packed_seq_params, cp_group, "zigzag", "contiguous"
    )
    reverse_route = get_or_build_thd_cp_partition_route(
        packed_seq_params, cp_group, "contiguous", "zigzag"
    )

    assert same_route is route
    assert reverse_route is not route
    assert len(packed_seq_params.cp_partition_route_cache) == 2


def test_thd_context_parallel_rank_indices_reject_decreasing_boundaries():
    with pytest.raises(ValueError, match="nondecreasing"):
        get_thd_context_parallel_rank_indices(torch.tensor([0, 16, 8]), 2, 0, "zigzag")


def test_thd_context_parallel_rank_indices_reject_unknown_layout():
    with pytest.raises(ValueError, match="Unsupported"):
        get_thd_context_parallel_rank_indices(torch.tensor([0, 16]), 2, 0, "interleaved")


def test_convert_cp_partition_mode_nested_recurses_over_tensor_containers(monkeypatch):
    calls = []

    def fake_convert(tensor, cp_group, **kwargs):
        calls.append((tensor, cp_group, kwargs))
        return tensor + 10

    monkeypatch.setattr(context_parallel_layout, "convert_cp_partition_mode", fake_convert)
    cp_group = object()
    cu_seqlens = torch.tensor([0, 8])
    untouched = object()
    value = (torch.tensor([1]), [None, untouched, torch.tensor([2])])

    converted = convert_cp_partition_mode_nested(
        value,
        cp_group,
        source_partition_mode="zigzag",
        target_partition_mode="contiguous",
        seq_dim=lambda tensor: tensor.dim() - 1,
        cu_seqlens=cu_seqlens,
    )

    assert torch.equal(converted[0], torch.tensor([11]))
    assert converted[1][0] is None
    assert converted[1][1] is untouched
    assert torch.equal(converted[1][2], torch.tensor([12]))
    assert [call[1] for call in calls] == [cp_group, cp_group]
    assert [call[2]["seq_dim"] for call in calls] == [0, 0]
    assert all(call[2]["cu_seqlens"] is cu_seqlens for call in calls)


def test_required_partition_mode_rejects_unknown_layer_type():
    with pytest.raises(ValueError, match="Cannot determine CP partition mode"):
        get_required_cp_partition_mode_for_layer(object(), SimpleNamespace(cp_comm_type=None))


def test_build_cp_partition_mode_plan_requires_stage_entry_layout():
    config = SimpleNamespace(
        context_parallel_size=2, dynamic_context_parallel=False, cp_comm_type=None
    )

    with pytest.raises(ValueError, match="cp_stage_entry_partition_mode"):
        build_cp_partition_mode_plan([], config, None, owner_name="TestBlock")


def test_build_cp_partition_mode_plan_tracks_exit_layout():
    config = SimpleNamespace(
        context_parallel_size=2,
        dynamic_context_parallel=False,
        cp_comm_type=None,
        linear_cp_mode="chunkwise",
    )

    entry, plan, exit_layout = build_cp_partition_mode_plan(
        [IdentityOp(), GatedDeltaNet()], config, "zigzag", owner_name="TestBlock"
    )

    assert entry == "zigzag"
    assert plan == [None, "contiguous"]
    assert exit_layout == "contiguous"


def test_build_cp_partition_mode_plan_skips_layer_inspection_without_cp():
    config = SimpleNamespace(context_parallel_size=1, dynamic_context_parallel=False)

    entry, plan, exit_layout = build_cp_partition_mode_plan(
        [object(), object()], config, None, owner_name="TestBlock"
    )

    assert entry == "zigzag"
    assert plan == [None, None]
    assert exit_layout == "zigzag"


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
        get_experimental_attention_variant_stage_input_cp_partition_mode(config)
        == "contiguous"
    )

    config.pipeline_model_parallel_layout = _PipelineLayout(offset=2)
    assert (
        get_experimental_attention_variant_stage_input_cp_partition_mode(config)
        == "zigzag"
    )


def test_gated_delta_net_headwise_layout_plan_uses_contiguous():
    config = SimpleNamespace(
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=[1, 0],
        linear_cp_mode="headwise",
        num_layers=2,
        pipeline_model_parallel_layout=None,
        pipeline_model_parallel_size=1,
    )

    assert (
        get_experimental_attention_variant_stage_input_cp_partition_mode(config)
        == "contiguous"
    )


def test_hybrid_stage_input_layout_follows_previous_sensitive_layer():
    config = SimpleNamespace(experimental_attention_variant=None, linear_cp_mode="chunkwise")

    assert get_hybrid_stage_input_cp_partition_mode(config, "M-G", 0) == "zigzag"
    assert get_hybrid_stage_input_cp_partition_mode(config, "M-G", 2) == "zigzag"
    assert get_hybrid_stage_input_cp_partition_mode(config, "M-G", 3) == "contiguous"


def test_hybrid_stage_input_layout_uses_future_layer_before_first_sensitive_layer():
    config = SimpleNamespace(experimental_attention_variant=None, linear_cp_mode="chunkwise")

    assert get_hybrid_stage_input_cp_partition_mode(config, "-G", 0) == "contiguous"


def test_hybrid_stage_input_layout_handles_dsv4_symbols():
    config = SimpleNamespace(
        experimental_attention_variant="dsv4_hybrid", linear_cp_mode="chunkwise"
    )

    assert get_hybrid_stage_input_cp_partition_mode(config, "D-E", 0) == "contiguous"
    assert get_hybrid_stage_input_cp_partition_mode(config, "C-E", 0) == "contiguous"
