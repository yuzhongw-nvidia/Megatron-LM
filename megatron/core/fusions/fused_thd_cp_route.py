# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Triton fused builder for THD context-parallel partition routes."""

from typing import Optional
from unittest.mock import MagicMock

import torch
from packaging import version

from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl

    if version.parse(triton.__version__) < version.parse("3.4.0") and not torch.cuda.is_available():
        HAVE_TRITON = False
    else:
        HAVE_TRITON = tl.constexpr(version.parse(triton.__version__) >= version.parse("2.0.0"))
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()


_MODE_CONTIGUOUS = 0
_MODE_ZIGZAG = 1

FUSED_THD_CP_ROUTE_SUPPORTED_CP_SIZES = (2, 4, 8)
FUSED_THD_CP_ROUTE_MAX_LOCAL_LENGTH = 16384
FUSED_THD_CP_ROUTE_MAX_CU_SEQLENS = 129


def _mode_to_int(cp_partition_mode: str) -> Optional[int]:
    if cp_partition_mode == "contiguous":
        return _MODE_CONTIGUOUS
    if cp_partition_mode == "zigzag":
        return _MODE_ZIGZAG
    return None


@triton.jit
def _find_seq_idx_from_global(
    cu_ptr,
    global_token,
    NUM_CU_SEQLENS: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    lo = tl.full((BLOCK_SIZE,), 0, tl.int64)
    hi = tl.full((BLOCK_SIZE,), NUM_CU_SEQLENS - 1, tl.int64)
    for _ in tl.static_range(0, SEARCH_STEPS):
        mid = (lo + hi + 1) // 2
        boundary = tl.load(cu_ptr + mid)
        take_upper = boundary <= global_token
        lo = tl.where(take_upper, mid, lo)
        hi = tl.where(take_upper, hi, mid - 1)
    return lo


@triton.jit
def _find_seq_idx_from_local_row(
    cu_ptr,
    local_row,
    CP_SIZE: tl.constexpr,
    NUM_CU_SEQLENS: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    lo = tl.full((BLOCK_SIZE,), 0, tl.int64)
    hi = tl.full((BLOCK_SIZE,), NUM_CU_SEQLENS - 1, tl.int64)
    for _ in tl.static_range(0, SEARCH_STEPS):
        mid = (lo + hi + 1) // 2
        boundary = tl.load(cu_ptr + mid) // CP_SIZE
        take_upper = boundary <= local_row
        lo = tl.where(take_upper, mid, lo)
        hi = tl.where(take_upper, hi, mid - 1)
    return lo


@triton.jit
def _local_to_global(
    cu_ptr,
    local_row,
    rank: tl.constexpr,
    TOTAL_TOKENS: tl.constexpr,
    CP_SIZE: tl.constexpr,
    MODE: tl.constexpr,
    NUM_CU_SEQLENS: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    if MODE == 0:
        part_len = TOTAL_TOKENS // CP_SIZE
        return rank * part_len + local_row

    seq_idx = _find_seq_idx_from_local_row(
        cu_ptr, local_row, CP_SIZE, NUM_CU_SEQLENS, SEARCH_STEPS, BLOCK_SIZE
    )
    seq_start = tl.load(cu_ptr + seq_idx)
    seq_end = tl.load(cu_ptr + seq_idx + 1)
    chunk_len = (seq_end - seq_start) // (2 * CP_SIZE)
    local_base = seq_start // CP_SIZE
    intra_seq_local = local_row - local_base
    in_first_slot = intra_seq_local < chunk_len
    chunk = tl.where(in_first_slot, rank, 2 * CP_SIZE - rank - 1)
    offset = tl.where(in_first_slot, intra_seq_local, intra_seq_local - chunk_len)
    return seq_start + chunk * chunk_len + offset


@triton.jit
def _global_to_owner_local(
    cu_ptr,
    global_token,
    TOTAL_TOKENS: tl.constexpr,
    CP_SIZE: tl.constexpr,
    MODE: tl.constexpr,
    NUM_CU_SEQLENS: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    if MODE == 0:
        part_len = TOTAL_TOKENS // CP_SIZE
        owner_rank = global_token // part_len
        local_row = global_token - owner_rank * part_len
        return owner_rank, local_row

    seq_idx = _find_seq_idx_from_global(
        cu_ptr, global_token, NUM_CU_SEQLENS, SEARCH_STEPS, BLOCK_SIZE
    )
    seq_start = tl.load(cu_ptr + seq_idx)
    seq_end = tl.load(cu_ptr + seq_idx + 1)
    chunk_len = (seq_end - seq_start) // (2 * CP_SIZE)
    seq_offset = global_token - seq_start
    chunk = seq_offset // chunk_len
    offset = seq_offset - chunk * chunk_len
    in_first_half = chunk < CP_SIZE
    owner_rank = tl.where(in_first_half, chunk, 2 * CP_SIZE - chunk - 1)
    local_slot = tl.where(in_first_half, 0, 1)
    local_base = seq_start // CP_SIZE
    local_row = local_base + local_slot * chunk_len + offset
    return owner_rank, local_row


@triton.jit
def _build_thd_cp_route_kernel(
    cu_ptr,
    send_rows_ptr,
    recv_rows_ptr,
    input_splits_ptr,
    output_splits_ptr,
    flags_ptr,
    route_lengths_ptr,
    TOTAL_TOKENS: tl.constexpr,
    CP_SIZE: tl.constexpr,
    CP_RANK: tl.constexpr,
    LOCAL_LENGTH: tl.constexpr,
    SOURCE_MODE: tl.constexpr,
    TARGET_MODE: tl.constexpr,
    NUM_CU_SEQLENS: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    side = tl.program_id(0)
    rows = tl.arange(0, BLOCK_SIZE)
    row_mask = rows < LOCAL_LENGTH
    safe_rows = tl.minimum(rows, LOCAL_LENGTH - 1).to(tl.int64)

    prefix = tl.full((), 0, tl.int64)
    identity = tl.full((), 1, tl.int32)

    for peer in tl.static_range(0, CP_SIZE):
        if side == 0:
            global_token = _local_to_global(
                cu_ptr,
                safe_rows,
                peer,
                TOTAL_TOKENS,
                CP_SIZE,
                TARGET_MODE,
                NUM_CU_SEQLENS,
                SEARCH_STEPS,
                BLOCK_SIZE,
            )
            owner, mapped_row = _global_to_owner_local(
                cu_ptr,
                global_token,
                TOTAL_TOKENS,
                CP_SIZE,
                SOURCE_MODE,
                NUM_CU_SEQLENS,
                SEARCH_STEPS,
                BLOCK_SIZE,
            )
            valid = row_mask & (owner == CP_RANK)
            compact_pos = tl.cumsum(valid.to(tl.int64), 0) - 1
            split_size = tl.sum(valid.to(tl.int64), 0)
            tl.store(send_rows_ptr + prefix + compact_pos, mapped_row, mask=valid)
            tl.store(input_splits_ptr + peer, split_size)
        else:
            global_token = _local_to_global(
                cu_ptr,
                safe_rows,
                peer,
                TOTAL_TOKENS,
                CP_SIZE,
                SOURCE_MODE,
                NUM_CU_SEQLENS,
                SEARCH_STEPS,
                BLOCK_SIZE,
            )
            owner, mapped_row = _global_to_owner_local(
                cu_ptr,
                global_token,
                TOTAL_TOKENS,
                CP_SIZE,
                TARGET_MODE,
                NUM_CU_SEQLENS,
                SEARCH_STEPS,
                BLOCK_SIZE,
            )
            valid = row_mask & (owner == CP_RANK)
            compact_pos = tl.cumsum(valid.to(tl.int64), 0) - 1
            split_size = tl.sum(valid.to(tl.int64), 0)
            tl.store(recv_rows_ptr + prefix + compact_pos, mapped_row, mask=valid)
            tl.store(output_splits_ptr + peer, split_size)

        expected_row = prefix + compact_pos
        mismatch = valid & (mapped_row != expected_row)
        identity = tl.where(tl.sum(mismatch.to(tl.int64), 0) == 0, identity, 0)
        prefix += split_size

    tl.store(route_lengths_ptr + side, prefix)
    tl.store(flags_ptr + side, identity)


def _is_supported_fused_route(
    cu_seqlens: torch.Tensor,
    cp_size: int,
    source_mode: int,
    target_mode: int,
    *,
    device: torch.device,
) -> bool:
    if not HAVE_TRITON:
        return False
    if source_mode == target_mode:
        return False
    if device.type != "cuda" or cu_seqlens.device.type != "cuda":
        return False
    if cp_size not in FUSED_THD_CP_ROUTE_SUPPORTED_CP_SIZES:
        return False
    if cu_seqlens.numel() > FUSED_THD_CP_ROUTE_MAX_CU_SEQLENS:
        return False
    total_tokens = int(cu_seqlens[-1].item())
    local_length = total_tokens // cp_size
    if local_length == 0 or local_length > FUSED_THD_CP_ROUTE_MAX_LOCAL_LENGTH:
        return False
    return True


def build_fused_thd_cp_partition_route(
    cu_seqlens: torch.Tensor,
    cp_size: int,
    cp_rank: int,
    source_partition_mode: str,
    target_partition_mode: str,
    *,
    device: torch.device,
):
    """Build a THD CP route with one Triton kernel, or return ``None`` if unsupported.

    ``cu_seqlens`` is expected to be compacted, validated, and resident on ``device``.
    The caller owns validation errors; unsupported but valid inputs fall back to the
    eager route builder by receiving ``None`` from this function.
    """
    source_mode = _mode_to_int(source_partition_mode)
    target_mode = _mode_to_int(target_partition_mode)
    if source_mode is None or target_mode is None:
        return None
    if not _is_supported_fused_route(
        cu_seqlens, cp_size, source_mode, target_mode, device=device
    ):
        return None

    total_tokens = int(cu_seqlens[-1].item())
    local_length = total_tokens // cp_size
    num_cu_seqlens = cu_seqlens.numel()
    block_size = triton.next_power_of_2(local_length)
    search_steps = max(1, (num_cu_seqlens - 1).bit_length())

    send_rows = torch.empty(local_length, device=device, dtype=torch.long)
    recv_rows = torch.empty(local_length, device=device, dtype=torch.long)
    input_split_sizes_dev = torch.empty(cp_size, device=device, dtype=torch.long)
    output_split_sizes_dev = torch.empty(cp_size, device=device, dtype=torch.long)
    flags_dev = torch.empty(2, device=device, dtype=torch.int32)
    route_lengths_dev = torch.empty(2, device=device, dtype=torch.long)

    _build_thd_cp_route_kernel[(2,)](
        cu_seqlens,
        send_rows,
        recv_rows,
        input_split_sizes_dev,
        output_split_sizes_dev,
        flags_dev,
        route_lengths_dev,
        TOTAL_TOKENS=total_tokens,
        CP_SIZE=cp_size,
        CP_RANK=cp_rank,
        LOCAL_LENGTH=local_length,
        SOURCE_MODE=source_mode,
        TARGET_MODE=target_mode,
        NUM_CU_SEQLENS=num_cu_seqlens,
        SEARCH_STEPS=search_steps,
        BLOCK_SIZE=block_size,
    )

    route_lengths = route_lengths_dev.cpu().tolist()
    assert route_lengths == [local_length, local_length]
    flags = flags_dev.cpu().tolist()

    from megatron.core.context_parallel_layout import ThdCPPartitionRoute

    return ThdCPPartitionRoute(
        source_partition_mode=source_partition_mode,
        target_partition_mode=target_partition_mode,
        cp_size=cp_size,
        cp_rank=cp_rank,
        local_source_length=local_length,
        local_target_length=local_length,
        send_rows=send_rows,
        recv_rows=recv_rows,
        input_split_sizes=input_split_sizes_dev.cpu().tolist(),
        output_split_sizes=output_split_sizes_dev.cpu().tolist(),
        send_rows_are_identity=bool(flags[0]),
        recv_rows_are_identity=bool(flags[1]),
    )
