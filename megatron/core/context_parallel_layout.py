# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Context parallel sequence partition-mode helpers."""

from typing import Any, List, Literal, Optional, Tuple

import torch


CpPartitionMode = Literal["zigzag", "contiguous"]
DEFAULT_CP_PARTITION_MODE: CpPartitionMode = "zigzag"


def normalize_cp_partition_mode(cp_partition_mode: Optional[str]) -> Optional[CpPartitionMode]:
    """Return a canonical CP partition mode string, preserving ``None`` requirements."""
    if cp_partition_mode is None:
        return None
    if cp_partition_mode in ("zigzag", "contiguous"):
        return cp_partition_mode
    raise ValueError(f"Unsupported context-parallel partition mode {cp_partition_mode!r}.")


def get_context_parallel_layout_chunk_indices(
    cp_size: int, cp_rank: int, cp_partition_mode: str
) -> torch.Tensor:
    """Return the two global chunk indices owned by this CP rank in ``cp_partition_mode``."""
    cp_partition_mode = normalize_cp_partition_mode(cp_partition_mode)
    if cp_partition_mode is None:
        raise ValueError("A concrete context-parallel partition mode is required.")
    if cp_size < 1:
        raise ValueError(f"cp_size must be >= 1, got {cp_size}.")
    if not 0 <= cp_rank < cp_size:
        raise ValueError(f"cp_rank must be in [0, {cp_size}), got {cp_rank}.")

    if cp_partition_mode == "zigzag":
        return torch.tensor([cp_rank, 2 * cp_size - cp_rank - 1], dtype=torch.long)
    if cp_partition_mode == "contiguous":
        return torch.tensor([2 * cp_rank, 2 * cp_rank + 1], dtype=torch.long)
    raise ValueError(f"Unsupported context-parallel partition mode {cp_partition_mode!r}.")


################################################################################
# Layer-to-CP-partition-mode mapping
################################################################################
#
# ``None`` is a meaningful result here: it means the module is token-layout
# agnostic and preserves whichever CP partition mode it receives.  It must not
# be used as the fallback for an unrecognized module type; unknown types should
# fail loudly so new layer implementations add an explicit partition-mode policy.


def get_required_cp_partition_mode_for_layer(
    layer: Any, config: Any, *, cp_comm_type: Optional[str] = None
) -> Optional[CpPartitionMode]:
    """Return the CP partition mode required by a layer or attention-like module.

    The helper intentionally uses light duck-typing instead of importing concrete
    modules, because several of those modules already import this file.
    """
    if cp_comm_type is None:
        cp_comm_type = getattr(config, "cp_comm_type", None)

    if layer is None:
        raise ValueError("Cannot determine CP partition mode for None.")

    module_name = layer.__class__.__name__
    if hasattr(layer, "self_attention"):
        return get_required_cp_partition_mode_for_layer(
            layer.self_attention, getattr(layer, "config", config), cp_comm_type=cp_comm_type
        )
    if module_name in {"IdentityOp", "IdentityFuncOp"}:
        return None
    if module_name == "GatedDeltaNet":
        mode = getattr(config, "linear_cp_mode", "chunkwise")
        if mode in {"chunkwise", "headwise"}:
            return "contiguous"
        raise ValueError(f"Unsupported GatedDeltaNet linear_cp_mode: {mode!r}.")
    if module_name in {"DSv4HybridAttention", "DSv4HybridSelfAttention"}:
        return "contiguous"

    # Preserve current standard-attention behavior.  Ring/P2P needs zigzag for
    # causal load balancing, and TE A2A currently still expects zigzag input.
    # ``cp_comm_type`` is deliberately part of this policy surface so TE A2A can
    # switch to contiguous here once the backend stops requiring zigzag.
    del cp_comm_type
    if module_name in {
        "SelfAttention",
        "CrossAttention",
        "MultiLatentAttention",
        "MLASelfAttention",
        "FusedMLASelfAttention",
        "AbsorbedMLASelfAttention",
    }:
        return "zigzag"
    raise ValueError(
        f"Cannot determine CP partition mode for layer/module type {module_name!r}."
    )


def get_thd_context_parallel_rank_indices(
    cu_seqlens: torch.Tensor, cp_size: int, cp_rank: int, cp_partition_mode: str
) -> torch.Tensor:
    """Return global THD token indices owned by one CP rank in a layout.

    Args:
        cu_seqlens: Global packed-sequence cumulative lengths before CP partitioning.
        cp_size: Context-parallel group size.
        cp_rank: Context-parallel rank.
        cp_partition_mode: Either ``"zigzag"`` or ``"contiguous"``.

    The returned indices are ordered exactly as the rank-local THD tensor is stored.
    ``"zigzag"`` follows Megatron's per-sequence load-balanced chunk order; ``"contiguous"``
    partitions the flattened packed THD buffer into rank-contiguous spans.
    """
    cp_partition_mode = normalize_cp_partition_mode(cp_partition_mode)
    if cp_partition_mode is None:
        raise ValueError("A concrete context-parallel partition mode is required.")
    if cp_size < 1:
        raise ValueError(f"cp_size must be >= 1, got {cp_size}.")
    if not 0 <= cp_rank < cp_size:
        raise ValueError(f"cp_rank must be in [0, {cp_size}), got {cp_rank}.")
    if cu_seqlens.dim() != 1:
        raise ValueError(f"cu_seqlens must be 1-D, got shape {tuple(cu_seqlens.shape)}.")

    cu = cu_seqlens.to(dtype=torch.long)
    if cu.numel() == 0 or cu[0].item() != 0:
        raise ValueError(f"cu_seqlens must start at 0, got {cu_seqlens}.")

    if torch.any(torch.diff(cu) < 0):
        raise ValueError(f"cu_seqlens must be nondecreasing, got {cu_seqlens}.")

    nonduplicate_boundaries = torch.ones(cu.numel(), device=cu.device, dtype=torch.bool)
    nonduplicate_boundaries[1:] = cu[1:] != cu[:-1]
    cu = cu[nonduplicate_boundaries]

    total_tokens = int(cu[-1].item())
    positions = torch.arange(total_tokens, device=cu.device, dtype=torch.long)
    if total_tokens == 0:
        return positions

    seq_lens = torch.diff(cu)
    if cp_partition_mode == "contiguous":
        if total_tokens % cp_size != 0:
            raise ValueError(
                f"Contiguous CP partitioning requires total_tokens={total_tokens} "
                f"to be divisible by cp_size={cp_size}."
            )
        part_len = total_tokens // cp_size
        rank_start = cp_rank * part_len
        return positions[rank_start : rank_start + part_len]

    chunk_divisor = 2 * cp_size
    if torch.any(seq_lens % chunk_divisor != 0):
        raise ValueError(
            "All packed sequence lengths must be divisible by "
            f"2 * cp_size ({chunk_divisor}) for zigzag CP layout conversion, "
            f"got {seq_lens}."
        )

    seq_idx = torch.bucketize(positions, cu[1:], right=True)
    global_starts = cu[:-1]
    pos_in_seq = positions - global_starts[seq_idx]
    chunk_lens = (seq_lens // chunk_divisor)[seq_idx]
    chunk = pos_in_seq // chunk_lens
    offset = pos_in_seq - chunk * chunk_lens

    owner = torch.where(chunk < cp_size, chunk, 2 * cp_size - chunk - 1)
    local_slot = torch.where(chunk < cp_size, torch.zeros_like(chunk), torch.ones_like(chunk))

    local_starts = (global_starts // cp_size)[seq_idx]
    local_pos = local_starts + local_slot * chunk_lens + offset

    rank_mask = owner == cp_rank
    rank_positions = positions[rank_mask]
    rank_local_pos = local_pos[rank_mask]
    return rank_positions[torch.argsort(rank_local_pos)]


def zigzag_to_contiguous_chunks(
    x: torch.Tensor,
    cp_group: torch.distributed.ProcessGroup,
    seq_dim: int = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Permute CP chunks from Megatron zigzag layout to contiguous-time layout.

    SBHD tensors have two equal chunks per rank along ``seq_dim`` and use a
    chunk-level all-to-all. THD tensors pass global ``cu_seqlens`` and use one
    packed-token all-to-all over the whole local THD tensor.
    """
    if cu_seqlens is not None:
        return _zigzag_contiguous_thd_swap(
            x,
            cp_group,
            seq_dim,
            cu_seqlens,
            source_partition_mode="zigzag",
            target_partition_mode="contiguous",
        )
    return _zigzag_contiguous_chunk_swap(x, cp_group, seq_dim, to_contiguous=True)


def contiguous_to_zigzag_chunks(
    x: torch.Tensor,
    cp_group: torch.distributed.ProcessGroup,
    seq_dim: int = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Inverse of :func:`zigzag_to_contiguous_chunks`."""
    if cu_seqlens is not None:
        return _zigzag_contiguous_thd_swap(
            x,
            cp_group,
            seq_dim,
            cu_seqlens,
            source_partition_mode="contiguous",
            target_partition_mode="zigzag",
        )
    return _zigzag_contiguous_chunk_swap(x, cp_group, seq_dim, to_contiguous=False)


def convert_cp_partition_mode(
    x: torch.Tensor,
    cp_group: Optional[torch.distributed.ProcessGroup],
    *,
    source_partition_mode: str,
    target_partition_mode: str,
    seq_dim: int = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    sequence_parallel: bool = False,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    tp_cp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """Convert a sequence tensor between CP zigzag and contiguous layouts.

    With sequence parallel enabled, the baseline path gathers the full CP-local
    sequence on each TP rank, performs the CP layout conversion, then scatters
    back to the original SP sharding.  ``tp_cp_group`` is accepted for the
    future direct TPxCP all-to-all implementation.
    """
    del tp_cp_group

    source_partition_mode = normalize_cp_partition_mode(source_partition_mode)
    target_partition_mode = normalize_cp_partition_mode(target_partition_mode)
    if source_partition_mode is None or target_partition_mode is None:
        raise ValueError(
            "source_partition_mode and target_partition_mode must be concrete partition modes."
        )
    if source_partition_mode == target_partition_mode:
        return x

    cp_size = cp_group.size() if cp_group is not None else 1
    if cp_size == 1:
        return x

    if sequence_parallel and tp_group is not None and tp_group.size() > 1:
        from megatron.core.tensor_parallel.mappings import (
            gather_from_sequence_parallel_region,
            scatter_to_sequence_parallel_region,
        )

        moved = x.movedim(seq_dim, 0) if seq_dim != 0 else x
        gathered = gather_from_sequence_parallel_region(moved, group=tp_group)
        converted = _convert_cp_partition_mode_full_sequence(
            gathered,
            cp_group,
            source_partition_mode=source_partition_mode,
            target_partition_mode=target_partition_mode,
            seq_dim=0,
            cu_seqlens=cu_seqlens,
        )
        scattered = scatter_to_sequence_parallel_region(converted, group=tp_group)
        return scattered.movedim(0, seq_dim).contiguous() if seq_dim != 0 else scattered

    return _convert_cp_partition_mode_full_sequence(
        x,
        cp_group,
        source_partition_mode=source_partition_mode,
        target_partition_mode=target_partition_mode,
        seq_dim=seq_dim,
        cu_seqlens=cu_seqlens,
    )


def get_packed_seq_params_cp_partition_cu_seqlens(
    packed_seq_params: Optional[Any],
) -> Optional[torch.Tensor]:
    """Return THD cumulative sequence lengths used for CP layout conversion."""
    if packed_seq_params is None or getattr(packed_seq_params, "qkv_format", None) != "thd":
        return None
    return (
        packed_seq_params.cu_seqlens_q_padded
        if packed_seq_params.cu_seqlens_q_padded is not None
        else packed_seq_params.cu_seqlens_q
    )


def convert_hidden_states_cp_partition_mode(
    hidden_states: Optional[torch.Tensor],
    cp_group: Optional[torch.distributed.ProcessGroup],
    *,
    source_partition_mode: str,
    target_partition_mode: str,
    packed_seq_params: Optional[Any] = None,
    sequence_parallel: bool = False,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    tp_cp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> Optional[torch.Tensor]:
    """Convert model hidden states between CP sequence layouts."""
    if hidden_states is None:
        return None
    dynamic_cp_group = (
        getattr(packed_seq_params, "cp_group", None) if packed_seq_params is not None else None
    )
    return convert_cp_partition_mode(
        hidden_states,
        dynamic_cp_group if dynamic_cp_group is not None else cp_group,
        source_partition_mode=source_partition_mode,
        target_partition_mode=target_partition_mode,
        seq_dim=0,
        cu_seqlens=get_packed_seq_params_cp_partition_cu_seqlens(packed_seq_params),
        sequence_parallel=sequence_parallel,
        tp_group=tp_group,
        tp_cp_group=tp_cp_group,
    )


def _convert_cp_partition_mode_full_sequence(
    x: torch.Tensor,
    cp_group: Optional[torch.distributed.ProcessGroup],
    *,
    source_partition_mode: CpPartitionMode,
    target_partition_mode: CpPartitionMode,
    seq_dim: int,
    cu_seqlens: Optional[torch.Tensor],
) -> torch.Tensor:
    """Convert a tensor whose sequence dim contains the full CP-local sequence."""
    if source_partition_mode == "zigzag" and target_partition_mode == "contiguous":
        return zigzag_to_contiguous_chunks(x, cp_group, seq_dim=seq_dim, cu_seqlens=cu_seqlens)
    if source_partition_mode == "contiguous" and target_partition_mode == "zigzag":
        return contiguous_to_zigzag_chunks(x, cp_group, seq_dim=seq_dim, cu_seqlens=cu_seqlens)
    raise ValueError(
        f"Unsupported CP partition mode conversion "
        f"{source_partition_mode!r} -> {target_partition_mode!r}."
    )


def _zigzag_contiguous_thd_swap(
    x: torch.Tensor,
    cp_group: Optional[torch.distributed.ProcessGroup],
    seq_dim: int,
    cu_seqlens: torch.Tensor,
    source_partition_mode: str,
    target_partition_mode: str,
) -> torch.Tensor:
    """Single-all-to-all THD permutation between zigzag and contiguous layouts.

    The packed THD tensor stays packed: we first group local tokens by their
    target CP rank, exchange those groups once, then scatter received tokens
    back into the target rank-local order.
    """
    cp_size = cp_group.size() if cp_group is not None else 1
    if cp_size == 1:
        return x
    cp_rank = cp_group.rank()
    from megatron.core.tensor_parallel.mappings import all_to_all

    if seq_dim != 0:
        x = x.movedim(seq_dim, 0)
    x = x.contiguous()

    cu = cu_seqlens.to(device=x.device, dtype=torch.long)
    # TODO: Let a future CP layout scheduler precompute this routing once per
    # microbatch from immutable cu_seqlens and pass it through both THD swaps.
    # Do not cache it across microbatches because packed sequence boundaries change.
    source_by_rank = [
        get_thd_context_parallel_rank_indices(cu, cp_size, rank, source_partition_mode)
        for rank in range(cp_size)
    ]
    target_by_rank = [
        get_thd_context_parallel_rank_indices(cu, cp_size, rank, target_partition_mode)
        for rank in range(cp_size)
    ]

    local_source_indices = source_by_rank[cp_rank]
    local_target_indices = target_by_rank[cp_rank]
    if x.size(0) != local_source_indices.numel():
        raise ValueError(
            f"Local THD tensor length ({x.size(0)}) does not match {source_partition_mode} "
            f"rank-{cp_rank} partition length ({local_source_indices.numel()})."
        )

    total_tokens = int(cu[-1].item())
    target_owner = torch.empty(total_tokens, device=x.device, dtype=torch.long)
    target_local_pos = torch.empty(total_tokens, device=x.device, dtype=torch.long)
    for rank, indices in enumerate(target_by_rank):
        target_owner[indices] = rank
        target_local_pos[indices] = torch.arange(indices.numel(), device=x.device)

    local_target_owner = target_owner[local_source_indices]
    local_target_pos = target_local_pos[local_source_indices]

    send_parts: List[torch.Tensor] = []
    input_split_sizes: List[int] = []
    for dst_rank in range(cp_size):
        dst_mask = local_target_owner == dst_rank
        dst_rows = dst_mask.nonzero(as_tuple=False).flatten()
        if dst_rows.numel() > 0:
            dst_rows = dst_rows[torch.argsort(local_target_pos[dst_rows])]
            send_part = x.index_select(0, dst_rows)
        else:
            send_part = x.narrow(0, 0, 0)
        send_parts.append(send_part)
        input_split_sizes.append(send_part.size(0))
    send_buf = torch.cat(send_parts, dim=0).contiguous()

    output_split_sizes: List[int] = []
    recv_target_positions: List[torch.Tensor] = []
    for src_rank in range(cp_size):
        src_indices = source_by_rank[src_rank]
        src_to_this_rank = target_owner[src_indices] == cp_rank
        recv_global_indices = src_indices[src_to_this_rank]
        if recv_global_indices.numel() > 0:
            recv_positions = target_local_pos[recv_global_indices]
            recv_positions = recv_positions[torch.argsort(recv_positions)]
        else:
            recv_positions = local_target_indices.narrow(0, 0, 0)
        recv_target_positions.append(recv_positions)
        output_split_sizes.append(recv_positions.numel())

    recv_buf = all_to_all(cp_group, send_buf, output_split_sizes, input_split_sizes)

    out_shape = (local_target_indices.numel(),) + tuple(x.shape[1:])
    out = x.new_empty(out_shape)
    offset = 0
    for recv_positions in recv_target_positions:
        recv_len = recv_positions.numel()
        if recv_len > 0:
            out[recv_positions] = recv_buf[offset : offset + recv_len]
            offset += recv_len

    if seq_dim != 0:
        out = out.movedim(0, seq_dim)
    return out.contiguous()


def _zigzag_contiguous_chunk_swap(
    x: torch.Tensor,
    cp_group: Optional[torch.distributed.ProcessGroup],
    seq_dim: int,
    to_contiguous: bool,
) -> torch.Tensor:
    """Single-all-to-all chunk permutation between zigzag and contiguous layouts.

    Each rank holds exactly two chunks along ``seq_dim``. The mapping from
    local (rank, slot) to (rank, slot) in the target layout is deterministic
    and depends only on ``cp_size`` and ``cp_rank``, so we pack send data in
    destination-rank order and use one ``all_to_all_single`` with unequal
    splits to route each chunk to its target rank.
    """
    cp_size = cp_group.size() if cp_group is not None else 1
    if cp_size == 1:
        return x
    cp_rank = cp_group.rank()
    from megatron.core.tensor_parallel.mappings import all_to_all

    # Work with seq_dim at position 0.
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
        """Global chunk indices at (slot 0, slot 1) for this rank."""
        if in_zigzag:
            return (rank, 2 * cp_size - rank - 1)
        return (2 * rank, 2 * rank + 1)

    def _chunk_to_dest(chunk_idx: int, target_zigzag: bool) -> Tuple[int, int]:
        """Destination (rank, slot) for a given global chunk index in the target layout."""
        if target_zigzag:
            if chunk_idx < cp_size:
                return chunk_idx, 0
            return 2 * cp_size - chunk_idx - 1, 1
        return chunk_idx // 2, chunk_idx % 2

    source_in_zigzag = to_contiguous
    target_in_zigzag = not to_contiguous

    local_chunk_indices = _rank_to_chunks(cp_rank, source_in_zigzag)
    local_dests = [_chunk_to_dest(c, target_in_zigzag) for c in local_chunk_indices]

    # Pack the send buffer so chunks are ordered by (dst_rank, dst_slot).
    local_slot_order = sorted(range(2), key=lambda s: local_dests[s])
    local_chunks = [x[:chunk_len], x[chunk_len:]]
    send_buf = torch.cat([local_chunks[s] for s in local_slot_order], dim=0).contiguous()

    input_split_chunks = [0] * cp_size
    for dst_rank, _ in local_dests:
        input_split_chunks[dst_rank] += 1

    # Mirror every source rank's packing logic so we know which received chunk
    # belongs in which local target slot.
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

    recv_buf = all_to_all(cp_group, send_buf, output_split_sizes, input_split_sizes)

    # Reassemble local chunks in target-layout slot order.
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
