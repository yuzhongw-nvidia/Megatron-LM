# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared preparation of packed microbatch metadata before model execution."""

from megatron.core.context_parallel_layout import finalize_packed_seq_params
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_config import TransformerConfig


def prepare_packed_seq_params(
    packed_seq_params: PackedSeqParams | None,
    config: TransformerConfig,
    *,
    capacity: int | None = None,
    local_tokens: int | None = None,
) -> PackedSeqParams | None:
    """Finalize CP routes and prepare enabled attention layouts outside graph capture.

    Args:
        packed_seq_params: Metadata for this microbatch, or None for unpacked input.
        config: The model's transformer configuration. Its sequence_parallel flag
            selects the fused TP x CP route when the process groups support it.
        capacity: Global physical token capacity when it differs from the supplied
            sequence boundaries (for example, raw metadata on a middle PP stage).
            Dynamic packed graphs otherwise use the configured fixed capacity.
        local_tokens: Physical token count on this CP rank before TP sequence sharding,
            obtained from the batch tensors. Required for packed linear attention
            unless the caller supplies its global physical capacity.

    Returns:
        The supplied metadata with its CP group and per-microbatch routes prepared.
    """
    packed_seq_params = finalize_packed_seq_params(
        packed_seq_params, sequence_parallel=getattr(config, "sequence_parallel", False)
    )
    if (
        packed_seq_params is not None
        and packed_seq_params.qkv_format == "thd"
        and _model_has_linear_attention(config)
    ):
        cp_size = packed_seq_params.cp_group.size()
        total_seq_len = local_tokens * cp_size if local_tokens is not None else capacity
        if total_seq_len is None:
            raise ValueError("Supply local_tokens or capacity for packed linear attention.")

        # These were per-layer _resolve_cu_seqlens checks. Validate each microbatch
        # once on the CPU, for every CP size/mode; Q/KV equality stays in the layer.
        for name in ("q", "kv"):
            padded = getattr(packed_seq_params, f"cu_seqlens_{name}_padded")
            cu_seqlens = (
                padded if padded is not None else getattr(packed_seq_params, f"cu_seqlens_{name}")
            )
            cu_cpu = cu_seqlens.detach().to(device="cpu", copy=True)
            total_cu = cu_cpu[-1].item()
            if total_cu != total_seq_len:
                raise ValueError(
                    f"GDN/KDA: cu_seqlens_{name}[-1]={total_cu} does not match "
                    f"total_sequence_length={total_seq_len}."
                )
            seq_lengths = cu_cpu[1:] - cu_cpu[:-1]
            if (seq_lengths % cp_size != 0).any():
                raise ValueError(
                    "All per-sequence lengths in cu_seqlens must be divisible by "
                    f"cp_size={cp_size}, but got lengths: {seq_lengths.tolist()}"
                )

    if packed_seq_params is None or not getattr(config, "dsa_cp_balance_indexer", False):
        return packed_seq_params

    from megatron.core.transformer.cuda_graph_config import cuda_graph_captures_attention
    from megatron.core.transformer.experimental_attention_variant.cp_balanced_indexer import (
        prebuild_balanced_layouts,
    )

    dynamic_packs = getattr(config, "dsa_cp_balance_indexer_graph_dynamic_packs", False)
    if capacity is None and dynamic_packs:
        capacity = config.max_seqlen_per_dp_cp_rank * config.context_parallel_size
    prebuild_balanced_layouts(
        packed_seq_params,
        cp_group=packed_seq_params.cp_group,
        pad_alignment=config.pad_packed_seq_alignment,
        capacity=capacity,
        graphs_enabled=cuda_graph_captures_attention(config),
        graph_dynamic_packs=dynamic_packs,
    )
    return packed_seq_params


def _model_has_linear_attention(config: TransformerConfig) -> bool:
    """Check the decoder/MTP layer pattern before preparing FLA-specific metadata."""
    pattern = getattr(config, "hybrid_layer_pattern", None)
    if pattern is not None:
        return "G" in pattern or "K" in pattern
    if getattr(config, "experimental_attention_variant", None) not in (
        "gdn",
        "gated_delta_net",
        "kda",
    ):
        return False
    if getattr(config, "is_hybrid_model", False):
        return True
    frequency = getattr(config, "linear_attention_freq", None)
    return frequency > 1 if isinstance(frequency, int) else bool(frequency and any(frequency))
