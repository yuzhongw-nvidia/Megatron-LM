# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Packed-sequence metadata helpers for CP partition-mode tracking."""

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from megatron.core.packed_seq_params import PackedSeqParams


def get_packed_seq_params_cp_partition_cu_seqlens(
    packed_seq_params: Optional["PackedSeqParams"],
) -> Optional[torch.Tensor]:
    """Return THD cumulative sequence lengths used for CP layout conversion.

    ``packed_seq_params=None`` represents the ordinary SBHD path. Only THD
    metadata carries global packed-token boundaries.
    """
    if packed_seq_params is None or getattr(packed_seq_params, "qkv_format", None) != "thd":
        return None
    return (
        packed_seq_params.cu_seqlens_q_padded
        if packed_seq_params.cu_seqlens_q_padded is not None
        else packed_seq_params.cu_seqlens_q
    )


def finalize_packed_seq_params(
    packed_seq_params: Optional["PackedSeqParams"],
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    *,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    tp_cp_group: Optional[torch.distributed.ProcessGroup] = None,
    needs_layout_conversion: bool = True,
) -> Optional["PackedSeqParams"]:
    """Resolve CP metadata and prebuild the THD layout routes for a microbatch.

    The CP-only route is always prebuilt; the fused TP x CP route next to it only when
    the TP groups are supplied (see ``prebuild_thd_cp_partition_routes``).

    Args:
        packed_seq_params: Packed-sequence metadata for the microbatch.
        cp_group: Caller-provided static context-parallel process group. A
            per-microbatch group stored in ``packed_seq_params`` takes precedence.
        tp_group: Tensor-parallel process group. Only matters under sequence
            parallelism: only then are the module inputs sequence-parallel shards whose
            layout conversion exchanges them over the TP x CP group, so pass the TP
            groups only when set and the prebuild builds the fused TP x CP route as well.
            A route the prebuild did not anticipate is built lazily from the cached host
            ``cu_seqlens`` without an extra device-to-host copy.
        tp_cp_group: The TP x CP process group that pairs with ``cp_group``.
        needs_layout_conversion: Build routes only for batches that will convert layouts.
            Sample-level inter-document masking must not build per-document THD routes.

    Returns:
        The finalized packed-sequence metadata, or ``None`` when no metadata was provided.
    """
    if packed_seq_params is None:
        return None

    # Keep these imports local: routes depends on this module for metadata access.
    from megatron.core.packed_seq_params import resolve_cp_group

    from .routes import prebuild_thd_cp_partition_routes

    cp_group = resolve_cp_group(static_cp_group=cp_group, packed_seq_params=packed_seq_params)
    if needs_layout_conversion:
        prebuild_thd_cp_partition_routes(
            packed_seq_params=packed_seq_params,
            cp_group=cp_group,
            tp_group=tp_group,
            tp_cp_group=tp_cp_group,
        )
    return packed_seq_params
