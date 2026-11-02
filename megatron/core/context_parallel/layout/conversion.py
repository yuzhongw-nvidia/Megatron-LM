# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tensor operations for converting between CP partition modes."""

import warnings
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple, Union, cast

import torch

from megatron.core.tensor_parallel.mappings import all_to_all
from megatron.core.utils import nvtx_range

from .manager import _redistribute_layout
from .metadata import get_packed_seq_params_cp_partition_cu_seqlens
from .routes import (
    build_thd_cp_partition_route,
    build_thd_tp_cp_partition_route,
    get_thd_cp_partition_route,
    get_thd_tp_cp_partition_route,
    resolve_tp_cp_group_rank_by_logical_rank,
)
from .types import CpPartitionMode, ThdCpRoute

if TYPE_CHECKING:
    from megatron.core.packed_seq_params import PackedSeqParams


def _cuda_graph_captures_attention(config: Any) -> bool:
    """Return whether the configured training graph includes the attention module."""
    cuda_graph_impl = getattr(config, "cuda_graph_impl", "none")
    if cuda_graph_impl == "full_iteration":
        return True
    if cuda_graph_impl not in ("local", "transformer_engine"):
        return False
    cuda_graph_modules = getattr(config, "cuda_graph_modules", ())
    if isinstance(cuda_graph_modules, str):
        cuda_graph_modules = (cuda_graph_modules,)
    return not cuda_graph_modules or any(
        getattr(module, "name", module) == "attn" for module in cuda_graph_modules
    )


class CpPartitionModeConverter:
    """Convert tensors across one CP layout edge."""

    def __init__(
        self,
        *,
        packed_seq_params: Optional["PackedSeqParams"],
        source_partition_mode: CpPartitionMode,
        target_partition_mode: CpPartitionMode,
        config: Any,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        tp_cp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        self.cp_group = cp_group
        self.packed_seq_params = packed_seq_params
        self.source_partition_mode = source_partition_mode
        self.target_partition_mode = target_partition_mode
        self.config = config
        self.tp_group = tp_group
        self.tp_cp_group = tp_cp_group
        if (
            self.conversion_needed
            and getattr(self.packed_seq_params, "qkv_format", None) == "thd"
            and _cuda_graph_captures_attention(self.config)
        ):
            # TODO(yuzhongw): use GIN to make native NCCL all-to-all split sizes replayable
            # before allowing THD CP layout routes inside CUDA graph capture.
            raise ValueError(
                "CUDA graph capture that includes attention is not supported for THD CP layout "
                f"conversion: source={self.source_partition_mode!r}, "
                f"target={self.target_partition_mode!r}."
            )

    @property
    def conversion_needed(self) -> bool:
        """Return whether this edge needs a real layout conversion."""
        return (
            self.source_partition_mode != self.target_partition_mode
            and self.cp_group is not None
            and self.cp_group.size() > 1
        )

    def assert_no_dense_attention_inputs(
        self,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> None:
        """Reject dense attention tensors when this edge would reorder tokens."""
        if not self.conversion_needed:
            return
        if attention_mask is not None:
            self._raise_unsupported_dense_attention(
                "an explicit attention_mask", hidden_states=hidden_states
            )
        if attention_bias is not None:
            self._raise_unsupported_dense_attention("attention_bias", hidden_states=hidden_states)

    def convert(
        self,
        value: Any,
        *,
        seq_dim: Union[int, Callable[[torch.Tensor], int]] = 0,
        sequence_parallel: bool = False,
    ) -> Any:
        """Convert a tensor or nested tensor container across this layout edge."""
        if not self.conversion_needed or value is None:
            return value
        # Nested values may contain optional tensors; traverse containers while
        # preserving their original shape.
        if isinstance(value, tuple):
            return tuple(
                self.convert(part, seq_dim=seq_dim, sequence_parallel=sequence_parallel)
                for part in value
            )
        if isinstance(value, list):
            return [
                self.convert(part, seq_dim=seq_dim, sequence_parallel=sequence_parallel)
                for part in value
            ]
        if not torch.is_tensor(value):
            return value

        resolved_seq_dim = seq_dim(value) if callable(seq_dim) else seq_dim
        # Sequence-parallel THD shards travel over the fused TP x CP route when the
        # process groups allow it; only the composed fallback needs the CP-only route.
        thd_tp_cp_partition_route = None
        if sequence_parallel and self.tp_group is not None and self.tp_group.size() > 1:
            thd_tp_cp_partition_route = get_thd_tp_cp_partition_route(
                self.packed_seq_params,
                self.source_partition_mode,
                self.target_partition_mode,
                cp_group=self.cp_group,
                tp_group=self.tp_group,
                tp_cp_group=self.tp_cp_group,
            )
        thd_cp_partition_route = None
        if thd_tp_cp_partition_route is None:
            thd_cp_partition_route = get_thd_cp_partition_route(
                self.packed_seq_params, self.source_partition_mode, self.target_partition_mode
            )
        converted = convert_cp_partition_mode(
            x=value,
            source_partition_mode=self.source_partition_mode,
            target_partition_mode=self.target_partition_mode,
            seq_dim=resolved_seq_dim,
            cu_seqlens=get_packed_seq_params_cp_partition_cu_seqlens(self.packed_seq_params),
            sequence_parallel=sequence_parallel,
            cp_group=self.cp_group,
            tp_group=self.tp_group,
            tp_cp_group=self.tp_cp_group,
            thd_cp_partition_route=thd_cp_partition_route,
            thd_tp_cp_partition_route=thd_tp_cp_partition_route,
        )
        if self.packed_seq_params is not None:
            self.packed_seq_params.cp_partition_mode = self.target_partition_mode
        return converted

    def _raise_unsupported_dense_attention(
        self, tensor_name: str, *, hidden_states: Optional[torch.Tensor]
    ) -> None:
        hidden_shape = tuple(hidden_states.shape) if hidden_states is not None else None
        raise NotImplementedError(
            "Changing CP partition mode with "
            f"{tensor_name} is not supported yet: "
            f"source={self.source_partition_mode!r}, "
            f"target={self.target_partition_mode!r}, "
            f"qkv_format={getattr(self.packed_seq_params, 'qkv_format', None)!r}, "
            f"hidden_shape={hidden_shape}."
        )


def convert_module_input_tensors_cp_partition_mode(
    *,
    hidden_states: torch.Tensor,
    packed_seq_params: Optional["PackedSeqParams"],
    target_partition_mode: CpPartitionMode,
    sequence_parallel: bool,
    config: Any,
    source_partition_mode: Optional[CpPartitionMode] = None,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    tp_cp_group: Optional[torch.distributed.ProcessGroup] = None,
    attention_mask: Optional[torch.Tensor] = None,
    attention_bias: Optional[torch.Tensor] = None,
    key_value_states: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[CpPartitionModeConverter]]:
    """Convert a module's rank-local sequence tensors to a target CP layout.

    This helper performs the common "entry conversion" pattern used by modules
    that need to consume a different CP layout than their caller supplied.  It
    returns a converter for the opposite edge so the module output can be
    converted back to the original input layout.
    """
    if cp_group is None or cp_group.size() <= 1:
        return hidden_states, None

    if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
        source_partition_mode = packed_seq_params.cp_partition_mode
    elif source_partition_mode is None:
        source_partition_mode = getattr(config, "cp_partition_mode", None)
    if source_partition_mode is None:
        raise ValueError(
            "The module input CP partition mode is required before layout conversion when "
            "context parallelism is active. Pass source_partition_mode or set "
            "config.cp_partition_mode."
        )
    if source_partition_mode == target_partition_mode:
        return hidden_states, None

    input_to_target_converter = CpPartitionModeConverter(
        cp_group=cp_group,
        packed_seq_params=packed_seq_params,
        source_partition_mode=source_partition_mode,
        target_partition_mode=target_partition_mode,
        config=config,
        tp_group=tp_group,
        tp_cp_group=tp_cp_group,
    )
    input_to_target_converter.assert_no_dense_attention_inputs(
        attention_mask=attention_mask, attention_bias=attention_bias, hidden_states=hidden_states
    )
    if key_value_states is not None:
        raise NotImplementedError(
            "Changing CP partition mode with cross-attention key/value states is not supported "
            f"yet: source={source_partition_mode!r}, target={target_partition_mode!r}."
        )
    hidden_states = input_to_target_converter.convert(
        value=hidden_states, seq_dim=0, sequence_parallel=sequence_parallel
    )

    target_to_input_converter = CpPartitionModeConverter(
        cp_group=cp_group,
        packed_seq_params=packed_seq_params,
        source_partition_mode=target_partition_mode,
        target_partition_mode=source_partition_mode,
        config=config,
        tp_group=tp_group,
        tp_cp_group=tp_cp_group,
    )
    return (hidden_states, target_to_input_converter)


def convert_cp_partition_mode(
    x: torch.Tensor,
    *,
    source_partition_mode: Optional[str],
    target_partition_mode: Optional[str],
    seq_dim: int = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    sequence_parallel: bool = False,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    tp_cp_group: Optional[torch.distributed.ProcessGroup] = None,
    thd_cp_partition_route: Optional[ThdCpRoute] = None,
    thd_tp_cp_partition_route: Optional[ThdCpRoute] = None,
) -> torch.Tensor:
    """Convert a sequence tensor between CP zigzag and contiguous layouts.

    SBHD tensors use one unified all-to-all-v redistribution path over CP or
    TPxCP. THD tensors use their packed-token CP route; when sequence
    parallelism shards the packed sequence, the shards are exchanged directly
    over the TPxCP group with one all-to-all-v. The composed TP gather -> CP
    all-to-all -> TP scatter path only remains for process groups that cannot
    be fused (no ``tp_cp_group``, or a CP group that is not the one the TPxCP
    group was built from, as with dynamic context parallelism).
    """

    if source_partition_mode == target_partition_mode:
        return x

    cp_size = cp_group.size() if cp_group is not None else 1
    if cp_size == 1:
        return x
    assert cp_group is not None

    if source_partition_mode not in ("zigzag", "contiguous") or target_partition_mode not in (
        "zigzag",
        "contiguous",
    ):
        cp_rank = cp_group.rank() if cp_group is not None else 0
        raise ValueError(
            f"Unsupported CP partition mode conversion "
            f"{source_partition_mode!r} -> {target_partition_mode!r}; "
            f"shape={tuple(x.shape)}, seq_dim={seq_dim}, cp_size={cp_size}, cp_rank={cp_rank}."
        )
    source_layout = cast(CpPartitionMode, source_partition_mode)
    target_layout = cast(CpPartitionMode, target_partition_mode)

    if cu_seqlens is None:
        moved = x.movedim(seq_dim, 0) if seq_dim != 0 else x
        converted = _redistribute_layout(
            input_=moved,
            cp_group=cp_group,
            source_layout=source_layout,
            target_layout=target_layout,
            sequence_parallel=sequence_parallel,
            tp_group=tp_group,
            tp_cp_group=tp_cp_group,
        )
        return converted.movedim(0, seq_dim).contiguous() if seq_dim != 0 else converted

    if sequence_parallel and tp_group is not None and tp_group.size() > 1:
        group_rank_by_logical_rank = resolve_tp_cp_group_rank_by_logical_rank(
            cp_group, tp_group, tp_cp_group
        )
        if group_rank_by_logical_rank is not None:
            return _redistribute_thd_tp_cp_layout(
                x=x,
                cp_group=cp_group,
                tp_group=tp_group,
                tp_cp_group=tp_cp_group,
                group_rank_by_logical_rank=group_rank_by_logical_rank,
                seq_dim=seq_dim,
                cu_seqlens=cu_seqlens,
                source_partition_mode=source_layout,
                target_partition_mode=target_layout,
                thd_tp_cp_partition_route=thd_tp_cp_partition_route,
            )
        return _redistribute_thd_layout_via_tp_gather(
            x=x,
            cp_group=cp_group,
            tp_group=tp_group,
            seq_dim=seq_dim,
            cu_seqlens=cu_seqlens,
            source_partition_mode=source_layout,
            target_partition_mode=target_layout,
            thd_cp_partition_route=thd_cp_partition_route,
        )

    return _redistribute_thd_layout(
        x=x,
        cp_group=cp_group,
        seq_dim=seq_dim,
        cu_seqlens=cu_seqlens,
        source_partition_mode=source_layout,
        target_partition_mode=target_layout,
        thd_cp_partition_route=thd_cp_partition_route,
    )


def _redistribute_thd_layout_via_tp_gather(
    x: torch.Tensor,
    cp_group: torch.distributed.ProcessGroup,
    tp_group: torch.distributed.ProcessGroup,
    seq_dim: int,
    cu_seqlens: torch.Tensor,
    source_partition_mode: CpPartitionMode,
    target_partition_mode: CpPartitionMode,
    thd_cp_partition_route: Optional[ThdCpRoute],
) -> torch.Tensor:
    """Composed TP gather -> CP all-to-all -> TP scatter conversion of an SP THD shard.

    Only used when the TP and CP groups cannot be fused into one TP x CP
    all-to-all (see :func:`convert_cp_partition_mode`).
    """
    from megatron.core.tensor_parallel.mappings import (
        gather_from_sequence_parallel_region,
        scatter_to_sequence_parallel_region,
    )

    warnings.warn(
        "THD CP layout conversion with sequence parallelism uses the naive "
        "TP gather -> CP all-to-all -> TP scatter fallback because the TP and CP "
        "process groups cannot be fused into one TP x CP all-to-all (no tp_cp_group, or "
        "a CP group that is not part of the TP x CP group, e.g. dynamic context "
        "parallelism).",
        RuntimeWarning,
        stacklevel=3,
    )
    moved = x.movedim(seq_dim, 0) if seq_dim != 0 else x
    # This gather is only used to run a duplicated CP layout permutation before
    # scattering back to SP shards. Its backward must split, not reduce-scatter;
    # otherwise every TP rank contributes the same full-sequence gradient.
    gathered = gather_from_sequence_parallel_region(
        input_=moved, tensor_parallel_output_grad=False, group=tp_group
    )
    converted = _redistribute_thd_layout(
        x=gathered,
        cp_group=cp_group,
        seq_dim=0,
        cu_seqlens=cu_seqlens,
        source_partition_mode=source_partition_mode,
        target_partition_mode=target_partition_mode,
        thd_cp_partition_route=thd_cp_partition_route,
    )
    scattered = scatter_to_sequence_parallel_region(input_=converted, group=tp_group)
    return scattered.movedim(0, seq_dim).contiguous() if seq_dim != 0 else scattered


def _pack_thd_cp_route_send_buffer(
    x: torch.Tensor, send_index: Optional[torch.Tensor]
) -> torch.Tensor:
    if send_index is None:
        return x
    return x.index_select(0, send_index)


def _scatter_thd_cp_route_recv_buffer(
    recv_buf: torch.Tensor, recv_index: Optional[torch.Tensor], out_shape: Tuple[int, ...]
) -> torch.Tensor:
    if recv_index is None:
        return recv_buf
    out = recv_buf.new_empty(out_shape)
    if recv_index.numel() > 0:
        out.index_copy_(0, recv_index, recv_buf)
    return out


def _select_thd_route_direction(
    route: ThdCpRoute, source_partition_mode: str, target_partition_mode: str
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], list, list]:
    """Return (send_index, recv_index, input_split_sizes, output_split_sizes)."""
    if source_partition_mode == "zigzag" and target_partition_mode == "contiguous":
        return (
            route.zigzag_index,
            route.contiguous_index,
            route.zigzag_split_sizes,
            route.contiguous_split_sizes,
        )
    if source_partition_mode == "contiguous" and target_partition_mode == "zigzag":
        return (
            route.contiguous_index,
            route.zigzag_index,
            route.contiguous_split_sizes,
            route.zigzag_split_sizes,
        )
    raise ValueError(
        f"Unsupported CP partition mode conversion "
        f"{source_partition_mode!r} -> {target_partition_mode!r} for THD route."
    )


def _exchange_thd_layout_with_route(
    x: torch.Tensor,
    route: ThdCpRoute,
    communication_group: torch.distributed.ProcessGroup,
    seq_dim: int,
    source_partition_mode: str,
    target_partition_mode: str,
    *,
    nvtx_scope: str,
    rank_description: str,
) -> torch.Tensor:
    """Differentiable pack -> all-to-all-v -> scatter of a packed THD tensor.

    ``route`` describes the exchange from this rank's point of view over
    ``communication_group``. Every op is differentiable, so the backward is the
    inverse exchange: the gradient is packed by the receive index, sent back with
    the swapped split sizes and scattered by the send index.
    """
    conversion_name = f"{source_partition_mode}_to_{target_partition_mode}"
    with nvtx_range(f"cp_layout/thd/{nvtx_scope}/{conversion_name}"):
        if seq_dim != 0:
            x = x.movedim(seq_dim, 0)
        x = x.contiguous()

        send_index, recv_index, input_split_sizes, output_split_sizes = _select_thd_route_direction(
            route, source_partition_mode, target_partition_mode
        )
        local_source_length = sum(input_split_sizes)
        local_target_length = sum(output_split_sizes)

        if x.size(0) != local_source_length:
            raise ValueError(
                f"Local THD tensor length ({x.size(0)}) does not match {source_partition_mode} "
                f"{rank_description} partition length ({local_source_length})."
            )
        if local_target_length != x.size(0):
            raise ValueError(
                "THD CP layout conversion must preserve the local token count, "
                f"got source={local_source_length}, target={local_target_length}, "
                f"{rank_description}, "
                f"source_layout={source_partition_mode!r}, "
                f"target_layout={target_partition_mode!r}."
            )

        with nvtx_range(f"cp_layout/thd/pack/{conversion_name}"):
            send_buf = _pack_thd_cp_route_send_buffer(x=x, send_index=send_index)
            if not send_buf.is_contiguous():
                send_buf = send_buf.contiguous()

        with nvtx_range(f"cp_layout/thd/all_to_all/{conversion_name}"):
            recv_buf = all_to_all(
                group=communication_group,
                input_=send_buf,
                output_split_sizes_=output_split_sizes,
                input_split_sizes=input_split_sizes,
            )

        with nvtx_range(f"cp_layout/thd/scatter/{conversion_name}"):
            out_shape = (local_target_length,) + tuple(x.shape[1:])
            out = _scatter_thd_cp_route_recv_buffer(
                recv_buf=recv_buf, recv_index=recv_index, out_shape=out_shape
            )

        if seq_dim != 0:
            out = out.movedim(0, seq_dim)
        return out.contiguous()


def _redistribute_thd_layout(
    x: torch.Tensor,
    cp_group: Optional[torch.distributed.ProcessGroup],
    seq_dim: int,
    cu_seqlens: torch.Tensor,
    source_partition_mode: str,
    target_partition_mode: str,
    thd_cp_partition_route: Optional[ThdCpRoute] = None,
) -> torch.Tensor:
    """Single-all-to-all THD permutation between zigzag and contiguous layouts.

    The packed THD tensor stays packed: we first group local tokens by their
    target CP rank, exchange those groups once, then scatter received tokens
    back into the target rank-local order.
    """
    cp_size = cp_group.size() if cp_group is not None else 1
    if cp_size == 1:
        return x
    assert cp_group is not None
    cp_rank = cp_group.rank()

    route = thd_cp_partition_route
    if route is None:
        route = build_thd_cp_partition_route(
            cu_seqlens=cu_seqlens, cp_size=cp_size, cp_rank=cp_rank, device=x.device
        )
    return _exchange_thd_layout_with_route(
        x=x,
        route=route,
        communication_group=cp_group,
        seq_dim=seq_dim,
        source_partition_mode=source_partition_mode,
        target_partition_mode=target_partition_mode,
        nvtx_scope="swap",
        rank_description=f"cp_size={cp_size}, cp_rank={cp_rank}",
    )


def _redistribute_thd_tp_cp_layout(
    x: torch.Tensor,
    cp_group: torch.distributed.ProcessGroup,
    tp_group: torch.distributed.ProcessGroup,
    tp_cp_group: torch.distributed.ProcessGroup,
    group_rank_by_logical_rank: Tuple[int, ...],
    seq_dim: int,
    cu_seqlens: torch.Tensor,
    source_partition_mode: str,
    target_partition_mode: str,
    thd_tp_cp_partition_route: Optional[ThdCpRoute] = None,
) -> torch.Tensor:
    """Fused TP x CP permutation of one sequence-parallel THD shard.

    The shard moves directly from its ``(tp_rank, cp_rank)`` owner to the target
    layout's owners with a single all-to-all-v over ``tp_cp_group``, replacing
    the TP gather -> CP all-to-all -> TP scatter composition. Both compositions
    apply the same token permutation, so the results are identical.
    """
    cp_size, cp_rank = cp_group.size(), cp_group.rank()
    tp_size, tp_rank = tp_group.size(), tp_group.rank()

    route = thd_tp_cp_partition_route
    if route is None:
        route = build_thd_tp_cp_partition_route(
            cu_seqlens=cu_seqlens,
            cp_size=cp_size,
            cp_rank=cp_rank,
            tp_size=tp_size,
            tp_rank=tp_rank,
            group_rank_by_logical_rank=group_rank_by_logical_rank,
            device=x.device,
        )
    return _exchange_thd_layout_with_route(
        x=x,
        route=route,
        communication_group=tp_cp_group,
        seq_dim=seq_dim,
        source_partition_mode=source_partition_mode,
        target_partition_mode=target_partition_mode,
        nvtx_scope="tp_cp_swap",
        rank_description=(
            f"cp_size={cp_size}, cp_rank={cp_rank}, tp_size={tp_size}, tp_rank={tp_rank}"
        ),
    )
