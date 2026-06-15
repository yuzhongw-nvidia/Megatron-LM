# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from typing import List, Literal, Optional, Tuple

import torch

CPPartitionLayout = Literal["zigzag", "contiguous"]
VALID_CP_PARTITION_LAYOUTS = ("zigzag", "contiguous")


def validate_cp_partition_layout(cp_partition_layout: str) -> CPPartitionLayout:
    """Validate and normalize a context-parallel sequence partition layout."""
    if cp_partition_layout not in VALID_CP_PARTITION_LAYOUTS:
        raise ValueError(
            f"cp_partition_layout must be one of {VALID_CP_PARTITION_LAYOUTS}, "
            f"got {cp_partition_layout!r}."
        )
    return cp_partition_layout


def get_cp_rank_partition_indices(
    cp_size: int,
    cp_rank: int,
    cp_partition_layout: str,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return the two global chunk indices owned by one CP rank.

    Both supported layouts split the global sequence into ``2 * cp_size`` chunks,
    so the local sequence length remains identical. They differ only in which two
    chunks each rank owns.
    """
    cp_partition_layout = validate_cp_partition_layout(cp_partition_layout)
    if cp_partition_layout == "zigzag":
        indices = [cp_rank, 2 * cp_size - cp_rank - 1]
    else:
        indices = [2 * cp_rank, 2 * cp_rank + 1]
    return torch.tensor(indices, dtype=torch.int64, device=device)


def get_thd_cp_rank_partition_indices(
    cu_seqlens: torch.Tensor,
    total_tokens: int,
    cp_size: int,
    cp_rank: int,
    cp_partition_layout: str,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return packed-THD token indices owned by one CP rank.

    ``cu_seqlens`` describes the padded packed sequence spans in the global
    packed token buffer. Each span is split into ``2 * cp_size`` chunks, then
    the same layout rule as SBHD is applied per packed sequence:

    * ``zigzag``: rank ``r`` owns chunks ``[r, 2 * cp_size - r - 1]``.
    * ``contiguous``: rank ``r`` owns chunks ``[2 * r, 2 * r + 1]``.

    The returned indices can be used directly with ``index_select`` along the
    packed token dimension.
    """
    cp_partition_layout = validate_cp_partition_layout(cp_partition_layout)
    if device is None:
        device = cu_seqlens.device
    total_tokens = int(total_tokens)

    if cp_size == 1:
        return torch.arange(total_tokens, dtype=torch.int64, device=device)

    if cu_seqlens.dim() == 2:
        assert cu_seqlens.shape[0] == 1, (
            f"THD cu_seqlens with a batch dimension must have shape [1, n], "
            f"got {tuple(cu_seqlens.shape)}."
        )
        cu_seqlens = cu_seqlens[0]
    assert cu_seqlens.dim() == 1, f"THD cu_seqlens must be 1-D, got {cu_seqlens.dim()}-D."
    assert int(cu_seqlens[-1].item()) == total_tokens, (
        f"THD total_tokens ({total_tokens}) must match cu_seqlens[-1] "
        f"({int(cu_seqlens[-1].item())})."
    )

    chunk_indices = get_cp_rank_partition_indices(
        cp_size, cp_rank, cp_partition_layout, device=cu_seqlens.device
    ).tolist()
    cu_seqlens_list = cu_seqlens.tolist()
    index_parts: List[torch.Tensor] = []
    for start, end in zip(cu_seqlens_list[:-1], cu_seqlens_list[1:]):
        seq_len = int(end - start)
        assert seq_len % (2 * cp_size) == 0, (
            f"THD packed sequence length ({seq_len}) must be divisible by "
            f"2 * cp_size ({2 * cp_size}) for {cp_partition_layout!r} CP layout."
        )
        chunk_len = seq_len // (2 * cp_size)
        for chunk_idx in chunk_indices:
            chunk_start = int(start) + int(chunk_idx) * chunk_len
            index_parts.append(
                torch.arange(
                    chunk_start, chunk_start + chunk_len, dtype=torch.int64, device=device
                )
            )

    if not index_parts:
        return torch.empty(0, dtype=torch.int64, device=device)
    return torch.cat(index_parts, dim=0)


def zigzag_to_contiguous_chunks(
    x: torch.Tensor, cp_group: torch.distributed.ProcessGroup, seq_dim: int = 0
) -> torch.Tensor:
    """Permute chunks across CP ranks from zigzag to contiguous layout.

    In the zigzag attention-load-balanced layout, rank ``r`` holds global chunks
    ``[r, 2*cp-r-1]``. In the contiguous-time layout, rank ``r`` holds
    ``[2r, 2r+1]``. The permutation is at chunk granularity, so one all-to-all
    routes each chunk to its target rank without materializing the full sequence.
    """
    return convert_cp_partition_layout(
        x,
        cp_group=cp_group,
        seq_dim=seq_dim,
        source_layout="zigzag",
        target_layout="contiguous",
    )


def contiguous_to_zigzag_chunks(
    x: torch.Tensor, cp_group: torch.distributed.ProcessGroup, seq_dim: int = 0
) -> torch.Tensor:
    """Permute chunks across CP ranks from contiguous to zigzag layout."""
    return convert_cp_partition_layout(
        x,
        cp_group=cp_group,
        seq_dim=seq_dim,
        source_layout="contiguous",
        target_layout="zigzag",
    )


def convert_cp_partition_layout(
    x: torch.Tensor,
    cp_group: Optional[torch.distributed.ProcessGroup],
    seq_dim: int,
    source_layout: str,
    target_layout: str,
) -> torch.Tensor:
    """Convert a local tensor between supported CP partition layouts."""
    source_layout = validate_cp_partition_layout(source_layout)
    target_layout = validate_cp_partition_layout(target_layout)
    if source_layout == target_layout:
        return x

    cp_size = cp_group.size() if cp_group is not None else 1
    if cp_size == 1:
        return x

    if source_layout == "zigzag" and target_layout == "contiguous":
        return _zigzag_contiguous_chunk_swap(x, cp_group, seq_dim, to_contiguous=True)
    if source_layout == "contiguous" and target_layout == "zigzag":
        return _zigzag_contiguous_chunk_swap(x, cp_group, seq_dim, to_contiguous=False)

    raise ValueError(f"Unsupported CP layout conversion: {source_layout!r} -> {target_layout!r}.")


def _zigzag_contiguous_chunk_swap(
    x: torch.Tensor,
    cp_group: Optional[torch.distributed.ProcessGroup],
    seq_dim: int,
    to_contiguous: bool,
) -> torch.Tensor:
    """Single-all-to-all chunk permutation between zigzag and contiguous layouts."""
    cp_size = cp_group.size() if cp_group is not None else 1
    if cp_size == 1:
        return x
    cp_rank = cp_group.rank()

    if seq_dim != 0:
        x = x.movedim(seq_dim, 0)
    x = x.contiguous()

    seq_len_local = x.size(0)
    assert seq_len_local % 2 == 0, (
        f"zigzag/contiguous chunk swap requires an even local sequence length, "
        f"got {seq_len_local}."
    )
    chunk_len = seq_len_local // 2

    def _rank_to_chunks(rank: int, in_zigzag: bool) -> Tuple[int, int]:
        if in_zigzag:
            return (rank, 2 * cp_size - rank - 1)
        return (2 * rank, 2 * rank + 1)

    def _chunk_to_dest(chunk_idx: int, target_zigzag: bool) -> Tuple[int, int]:
        if target_zigzag:
            if chunk_idx < cp_size:
                return chunk_idx, 0
            return 2 * cp_size - chunk_idx - 1, 1
        return chunk_idx // 2, chunk_idx % 2

    source_in_zigzag = to_contiguous
    target_in_zigzag = not to_contiguous

    local_chunk_indices = _rank_to_chunks(cp_rank, source_in_zigzag)
    local_dests = [_chunk_to_dest(c, target_in_zigzag) for c in local_chunk_indices]

    local_slot_order = sorted(range(2), key=lambda s: local_dests[s])
    local_chunks = [x[:chunk_len], x[chunk_len:]]
    send_buf = torch.cat([local_chunks[s] for s in local_slot_order], dim=0).contiguous()

    input_split_chunks = [0] * cp_size
    for dst_rank, _ in local_dests:
        input_split_chunks[dst_rank] += 1

    output_split_chunks = [0] * cp_size
    recv_dst_slots_per_source: List[List[int]] = [[] for _ in range(cp_size)]
    for src in range(cp_size):
        src_chunks = _rank_to_chunks(src, source_in_zigzag)
        src_dests = [_chunk_to_dest(c, target_in_zigzag) for c in src_chunks]
        src_slot_order = sorted(range(2), key=lambda s: src_dests[s])
        for s in src_slot_order:
            dst_rank, dst_slot = src_dests[s]
            if dst_rank == cp_rank:
                output_split_chunks[src] += 1
                recv_dst_slots_per_source[src].append(dst_slot)

    input_split_sizes = [n * chunk_len for n in input_split_chunks]
    output_split_sizes = [n * chunk_len for n in output_split_chunks]

    from megatron.core.tensor_parallel import all_to_all

    recv_buf = all_to_all(cp_group, send_buf, output_split_sizes, input_split_sizes)

    target_slots: List[Optional[torch.Tensor]] = [None, None]
    offset = 0
    for src in range(cp_size):
        for dst_slot in recv_dst_slots_per_source[src]:
            target_slots[dst_slot] = recv_buf[offset : offset + chunk_len]
            offset += chunk_len
    assert all(t is not None for t in target_slots), "Incomplete chunk reassembly in CP swap"

    out = torch.cat(target_slots, dim=0)
    if seq_dim != 0:
        out = out.movedim(0, seq_dim)
    return out.contiguous()
