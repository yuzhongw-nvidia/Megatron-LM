# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor


@dataclass
class PackedSeqParams:
    '''
    parameters to TEDotProductAttention and fused rope kernels for the
    `thd` (packed) sequence format
    '''

    qkv_format: str = None
    cu_seqlens_q: Tensor = None
    cu_seqlens_kv: Tensor = None
    cu_seqlens_q_padded: Tensor = None
    cu_seqlens_kv_padded: Tensor = None
    max_seqlen_q: int = None
    max_seqlen_kv: int = None
    local_cp_size: int = None
    cp_group: dist.ProcessGroup = None
    total_tokens: int = None
    seq_idx: Tensor = None

    def __post_init__(self):
        """Pre-compute seq_idx for Mamba mixer CUDA graph compatibility.

        If total_tokens is 16 (for example), this method takes packed_seq_params.cu_seqlens_q_padded
        (or cu_seqlens_q) which is of the form [0, 5, 7, 11] and returns a tensor of the form
        [0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        which is [0]*(5-0) + [1]*(7-5) + [2]*(11-7) + [3]*(16-11)
        In the above example, there are three sequences in the pack.
        In general, the output has an additional sequence index (e.g. 0, 1, 2, 3) so that any tokens
        beyond the last padded input sequence are accounted for as an extra sequence. However, If
        cu_seqlens_q_padded[-1] == max_seqlen then this additional sequence index will not be
        included.
        """
        cu_seqlens = (
            self.cu_seqlens_q_padded if self.cu_seqlens_q_padded is not None else self.cu_seqlens_q
        )
        if isinstance(cu_seqlens, Tensor) and self.total_tokens is not None:
            total_tokens_tensor = torch.tensor(
                [self.total_tokens], dtype=cu_seqlens.dtype, device=cu_seqlens.device
            )
            # Example: [0, 5, 7, 11] -> [0, 5, 7, 11, 16]
            cu_seqlens_with_max = torch.cat([cu_seqlens, total_tokens_tensor])
            # Example: [0, 5, 7, 11, 16] -> [5, 2, 4, 5]
            seq_lengths = cu_seqlens_with_max[1:] - cu_seqlens_with_max[:-1]
            # Clamp to non-negative: cu_seqlens_q_padded may not be strictly
            # monotonic when context parallelism slices sequences across ranks,
            # or when padded cumulative lengths exceed total_tokens (e.g. the
            # appended total_tokens sentinel is smaller than cu_seqlens[-1]
            # due to padding). In either case the diff can go negative, which
            # causes torch.repeat_interleave to fail.
            seq_lengths = seq_lengths.clamp(min=0)
            # Example: [5, 2, 4, 5] -> [0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3]
            self.seq_idx = (
                torch.repeat_interleave(
                    torch.arange(seq_lengths.numel(), device=cu_seqlens.device), seq_lengths
                )
                .to(torch.int32)
                .unsqueeze(0)  # Add a batch dimension
            )

def resolve_cp_group(
    static_cp_group: dist.ProcessGroup, packed_seq_params: PackedSeqParams = None
) -> dist.ProcessGroup:
    """Return the dynamic CP group from packed_seq_params when available, else the static one.

    Dynamic CP assigns a per-microbatch CP group that may differ from the
    process-group stored at model construction time.  This helper centralises
    the resolution logic used by GPTModel, GatedDeltaNet, and MTP layers.
    """
    if packed_seq_params is not None and packed_seq_params.cp_group is not None:
        return packed_seq_params.cp_group
    return static_cp_group


def _pad_seq_tensor(t: Optional[Tensor], target_len: int) -> Optional[Tensor]:
    """Pad a [..., seq] tensor to ``target_len`` along the last dim with zeros.

    Asserts the actual length does not exceed ``target_len``: an oversize input
    would silently desync the captured graph from replay shapes.
    """
    if t is None:
        return None
    actual_len = t.shape[-1]
    assert actual_len <= target_len, (
        f"Sequence-length tensor (last dim = {actual_len}) exceeds target "
        f"({target_len}); refusing to silently truncate. Increase "
        f"--max-seqlen-per-dp-cp-rank or filter overlong samples upstream."
    )
    if actual_len == target_len:
        return t
    return F.pad(t, (0, target_len - actual_len), value=0)


def _round_up_to_alignment(value: int, alignment: int) -> int:
    assert alignment > 0, f"Packed sequence padding alignment must be > 0, got {alignment}."
    return ((value + alignment - 1) // alignment) * alignment


def pad_sequence_for_thd(
    tokens: Optional[Tensor],
    labels: Optional[Tensor],
    loss_mask: Optional[Tensor],
    position_ids: Optional[Tensor],
    packed_seq_params: PackedSeqParams,
    alignment: int,
) -> Tuple[
    Optional[Tensor],
    Optional[Tensor],
    Optional[Tensor],
    Optional[Tensor],
    PackedSeqParams,
    Optional[Tensor],
]:
    """Pad packed THD tensors to an alignment without changing sequence metadata.

    This appends padding tokens to the token-like tensors after packing/CP slicing.
    It intentionally keeps ``cu_seqlens`` unchanged so the original sequence
    boundaries are preserved. The returned padding mask marks the appended tokens
    for MoE auxiliary loss/routing paths.

    Returns:
        Padded (tokens, labels, loss_mask, position_ids, packed_seq_params, padding_mask)
        padding_mask: [1, target_len] bool tensor, True at padding positions.
    """

    actual_T = None
    mask_device = None
    for candidate in (tokens, labels, loss_mask, position_ids):
        if candidate is not None:
            actual_T = candidate.shape[-1]
            mask_device = candidate.device
            break
    actual_T_is_local = actual_T is not None
    if actual_T is None:
        assert packed_seq_params.cu_seqlens_q is not None, (
            "packed_seq_params.cu_seqlens_q must be available to derive padding_mask "
            "when tokens/labels/loss_mask/position_ids are all None."
        )
        actual_T = int(packed_seq_params.cu_seqlens_q[-1].item())
        mask_device = packed_seq_params.cu_seqlens_q.device

    from megatron.core import parallel_state

    cp_size = (
        packed_seq_params.local_cp_size
        if packed_seq_params.local_cp_size is not None
        else parallel_state.get_context_parallel_world_size()
    )
    cp_rank = parallel_state.get_context_parallel_rank() if cp_size > 1 else 0

    if cp_size > 1:
        from megatron.core.extensions.transformer_engine import get_thd_partitioned_indices

        if actual_T_is_local:
            local_actual_T = int(actual_T)
            local_target_T = _round_up_to_alignment(local_actual_T, alignment)
        else:
            local_actual_T = int(
                get_thd_partitioned_indices(
                    (
                        packed_seq_params.cu_seqlens_q_padded
                        if packed_seq_params.cu_seqlens_q_padded is not None
                        else packed_seq_params.cu_seqlens_q
                    ),
                    int(actual_T),
                    cp_size,
                    cp_rank,
                ).numel()
            )
            local_target_T = _round_up_to_alignment(local_actual_T, alignment)
    else:
        local_actual_T = int(actual_T)
        local_target_T = _round_up_to_alignment(local_actual_T, alignment)

    padded_params = PackedSeqParams(
        qkv_format=packed_seq_params.qkv_format,
        cu_seqlens_q=packed_seq_params.cu_seqlens_q,
        cu_seqlens_kv=packed_seq_params.cu_seqlens_kv,
        cu_seqlens_q_padded=packed_seq_params.cu_seqlens_q_padded,
        cu_seqlens_kv_padded=packed_seq_params.cu_seqlens_kv_padded,
        max_seqlen_q=packed_seq_params.max_seqlen_q,
        max_seqlen_kv=packed_seq_params.max_seqlen_kv,
        local_cp_size=packed_seq_params.local_cp_size,
        cp_group=packed_seq_params.cp_group,
        total_tokens=local_target_T if cp_size == 1 else None,
    )

    tokens = _pad_seq_tensor(tokens, local_target_T)
    labels = _pad_seq_tensor(labels, local_target_T)
    loss_mask = _pad_seq_tensor(loss_mask, local_target_T)
    position_ids = _pad_seq_tensor(position_ids, local_target_T)
    padding_mask = torch.arange(local_target_T, device=mask_device).unsqueeze(0) >= local_actual_T

    return tokens, labels, loss_mask, position_ids, padded_params, padding_mask
