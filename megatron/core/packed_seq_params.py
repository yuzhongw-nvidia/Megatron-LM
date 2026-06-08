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


def _cu_seqlens_with_padded_total(
    cu_seqlens: Optional[Tensor], target_total_tokens: int
) -> Optional[Tensor]:
    if cu_seqlens is None:
        return None
    assert cu_seqlens.numel() > 0, "cu_seqlens must contain at least the zero offset."
    current_total_tokens = int(cu_seqlens[-1].item())
    assert current_total_tokens <= target_total_tokens, (
        f"Packed cu_seqlens total ({current_total_tokens}) exceeds padded token total "
        f"({target_total_tokens}). Increase --pad-packed-seq-alignment or reduce packed "
        f"sequence length."
    )
    if current_total_tokens == target_total_tokens:
        return cu_seqlens
    padded = cu_seqlens.clone()
    padded[-1] = target_total_tokens
    return padded


def _max_seqlen_with_padded_total(
    cu_seqlens: Optional[Tensor], max_seqlen: Optional[int], target_total_tokens: int
) -> Optional[int]:
    if cu_seqlens is None:
        return max(max_seqlen or 0, target_total_tokens)
    if cu_seqlens.numel() <= 1:
        return max_seqlen
    last_seq_start = int(cu_seqlens[-2].item())
    padded_last_seq_len = target_total_tokens - last_seq_start
    return max(max_seqlen or 0, padded_last_seq_len)


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
    """Pad packed THD tensors to an alignment.

    This follows the original ``pad_thd_for_cuda_graph`` structure: determine
    the current packed token count, pad token-like tensors along the sequence
    dimension, return updated ``PackedSeqParams``, and create a padding mask.
    The original ``cu_seqlens`` values are preserved, while
    ``cu_seqlens_*_padded[-1]`` is extended to the padded token total.

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
    if actual_T is None:
        assert packed_seq_params.cu_seqlens_q is not None, (
            "packed_seq_params.cu_seqlens_q must be available to derive padding_mask "
            "when tokens/labels/loss_mask/position_ids are all None."
        )
        actual_T = int(packed_seq_params.cu_seqlens_q[-1].item())
        mask_device = packed_seq_params.cu_seqlens_q.device

    metadata_cu_seqlens = (
        packed_seq_params.cu_seqlens_q_padded
        if packed_seq_params.cu_seqlens_q_padded is not None
        else packed_seq_params.cu_seqlens_q
    )
    if metadata_cu_seqlens is not None:
        actual_T = max(int(actual_T), int(metadata_cu_seqlens[-1].item()))
    target_T = _round_up_to_alignment(int(actual_T), alignment)

    padded_params = PackedSeqParams(
        qkv_format=packed_seq_params.qkv_format,
        cu_seqlens_q=packed_seq_params.cu_seqlens_q,
        cu_seqlens_kv=packed_seq_params.cu_seqlens_kv,
        cu_seqlens_q_padded=_cu_seqlens_with_padded_total(
            (
                packed_seq_params.cu_seqlens_q_padded
                if packed_seq_params.cu_seqlens_q_padded is not None
                else packed_seq_params.cu_seqlens_q
            ),
            target_T,
        ),
        cu_seqlens_kv_padded=_cu_seqlens_with_padded_total(
            (
                packed_seq_params.cu_seqlens_kv_padded
                if packed_seq_params.cu_seqlens_kv_padded is not None
                else packed_seq_params.cu_seqlens_kv
            ),
            target_T,
        ),
        max_seqlen_q=_max_seqlen_with_padded_total(
            (
                packed_seq_params.cu_seqlens_q_padded
                if packed_seq_params.cu_seqlens_q_padded is not None
                else packed_seq_params.cu_seqlens_q
            ),
            packed_seq_params.max_seqlen_q,
            target_T,
        ),
        max_seqlen_kv=_max_seqlen_with_padded_total(
            (
                packed_seq_params.cu_seqlens_kv_padded
                if packed_seq_params.cu_seqlens_kv_padded is not None
                else packed_seq_params.cu_seqlens_kv
            ),
            packed_seq_params.max_seqlen_kv,
            target_T,
        ),
        local_cp_size=packed_seq_params.local_cp_size,
        cp_group=packed_seq_params.cp_group,
    )

    tokens = _pad_seq_tensor(tokens, target_T)
    labels = _pad_seq_tensor(labels, target_T)
    loss_mask = _pad_seq_tensor(loss_mask, target_T)
    position_ids = _pad_seq_tensor(position_ids, target_T)
    padding_mask = torch.arange(target_T, device=mask_device).unsqueeze(0) >= actual_T

    return tokens, labels, loss_mask, position_ids, padded_params, padding_mask
