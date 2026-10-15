# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Chunk metadata caches shared by the internal GDR backend's forward and backward paths."""

from __future__ import annotations

import weakref
from collections import OrderedDict
from dataclasses import dataclass

import torch
from fla.ops.utils.index import prepare_chunk_indices

_CHUNK_SIZE = 64
_DENSE_CHUNK_METADATA_CACHE_LIMIT = 64


@dataclass
class _PackedChunkMetadata:
    """Chunk metadata derived from one packed ``cu_seqlens`` tensor.

    ``chunk_offsets`` is always materialized; ``chunk_indices`` is filled on the first
    request that needs it, so the single-local-sequence CP window (which only consumes the
    offsets) never pays for the indices.
    """

    owner: weakref.ReferenceType
    version: int
    chunk_size: int
    chunk_offsets: torch.Tensor
    chunk_indices: torch.Tensor | None = None


@dataclass(frozen=True)
class _DenseChunkMetadata:
    cu_seqlens: torch.Tensor
    chunk_offsets: torch.Tensor


_packed_chunk_metadata_cache: dict[
    tuple[int, tuple[int, int] | None], _PackedChunkMetadata
] = {}
_dense_chunk_metadata_cache: OrderedDict[
    tuple[str, int | None, int, int, int, tuple[int, int] | None], _DenseChunkMetadata
] = OrderedDict()


def _clear_packed_chunk_metadata_cache_for_test() -> None:
    _packed_chunk_metadata_cache.clear()


def _clear_dense_chunk_metadata_cache_for_test() -> None:
    _dense_chunk_metadata_cache.clear()


def _current_stream_cache_key(tensor_or_device: torch.Tensor | torch.device) -> tuple[int, int] | None:
    device = (
        tensor_or_device.device
        if isinstance(tensor_or_device, torch.Tensor)
        else torch.device(tensor_or_device)
    )
    if device.type != "cuda":
        return None
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
        device = torch.device("cuda", device_index)
    stream = torch.cuda.current_stream(device)
    return (device_index, int(stream.cuda_stream))


def _build_chunk_offsets(cu_seqlens: torch.Tensor) -> torch.Tensor:
    sequence_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    chunks_per_sequence = (sequence_lengths + _CHUNK_SIZE - 1) // _CHUNK_SIZE
    chunk_offsets = torch.empty_like(cu_seqlens)
    chunk_offsets[0] = 0
    torch.cumsum(chunks_per_sequence, dim=0, out=chunk_offsets[1:])
    return chunk_offsets


def _packed_chunk_metadata(
    cu_seqlens: torch.Tensor | None, *, include_chunk_indices: bool = True
) -> _PackedChunkMetadata | None:
    """Return the cached chunk metadata of ``cu_seqlens`` (keyed by identity, version, stream)."""
    if cu_seqlens is None:
        return None
    stream_key = _current_stream_cache_key(cu_seqlens)
    key = (id(cu_seqlens), stream_key)
    version = cu_seqlens._version
    cached = _packed_chunk_metadata_cache.get(key)
    if (
        cached is None
        or cached.owner() is not cu_seqlens
        or cached.version != version
        or cached.chunk_size != _CHUNK_SIZE
    ):
        chunk_indices = (
            prepare_chunk_indices(cu_seqlens, _CHUNK_SIZE) if include_chunk_indices else None
        )
        chunk_offsets = _build_chunk_offsets(cu_seqlens)

        def remove_cached_owner(owner_ref):
            current = _packed_chunk_metadata_cache.get(key)
            if current is not None and current.owner is owner_ref:
                _packed_chunk_metadata_cache.pop(key, None)

        cached = _PackedChunkMetadata(
            owner=weakref.ref(cu_seqlens, remove_cached_owner),
            version=version,
            chunk_size=_CHUNK_SIZE,
            chunk_offsets=chunk_offsets,
            chunk_indices=chunk_indices,
        )
        _packed_chunk_metadata_cache[key] = cached
    elif include_chunk_indices and cached.chunk_indices is None:
        cached.chunk_indices = prepare_chunk_indices(cu_seqlens, _CHUNK_SIZE)
    return cached


def prepare_validated_chunk_metadata(
    cu_seqlens: torch.Tensor | None,
    *,
    include_chunk_indices: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Prepare trusted packed metadata before entering the profiled GDR region."""
    metadata = _packed_chunk_metadata(cu_seqlens, include_chunk_indices=include_chunk_indices)
    if metadata is None:
        return None, None
    chunk_indices = metadata.chunk_indices if include_chunk_indices else None
    return chunk_indices, metadata.chunk_offsets


def _validate_validated_chunk_indices(
    validated_chunk_indices: torch.Tensor | None, cu_seqlens: torch.Tensor | None
) -> None:
    if validated_chunk_indices is None:
        return
    if (
        cu_seqlens is None
        or validated_chunk_indices.ndim != 2
        or validated_chunk_indices.shape[1] != 2
        or validated_chunk_indices.dtype != cu_seqlens.dtype
        or validated_chunk_indices.device != cu_seqlens.device
        or not validated_chunk_indices.is_contiguous()
    ):
        raise ValueError(
            "validated_chunk_indices must be contiguous with shape (num_chunks, 2) "
            "and match cu_seqlens dtype and device"
        )


def _device_cache_key(device: torch.device) -> tuple[str, int | None]:
    device = torch.device(device)
    device_index = device.index
    if device.type == "cuda" and device_index is None:
        device_index = torch.cuda.current_device()
    return (device.type, device_index)


def _dense_chunk_metadata(
    batch_size: int, seqlen: int, device: torch.device
) -> _DenseChunkMetadata:
    if batch_size < 1 or seqlen < 1:
        raise ValueError("dense batch size and sequence length must be positive")
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    key = (
        *_device_cache_key(device),
        batch_size,
        seqlen,
        _CHUNK_SIZE,
        _current_stream_cache_key(device),
    )
    cached = _dense_chunk_metadata_cache.get(key)
    if cached is not None and cached.cu_seqlens.device == device:
        _dense_chunk_metadata_cache.move_to_end(key)
        return cached
    if cached is not None:
        _dense_chunk_metadata_cache.pop(key, None)
    cu_seqlens = torch.arange(
        0, (batch_size + 1) * seqlen, seqlen, device=device, dtype=torch.int32
    )
    chunks_per_sequence = (seqlen + _CHUNK_SIZE - 1) // _CHUNK_SIZE
    chunk_offsets = torch.arange(
        0,
        (batch_size + 1) * chunks_per_sequence,
        chunks_per_sequence,
        device=device,
        dtype=torch.int32,
    )
    metadata = _DenseChunkMetadata(cu_seqlens=cu_seqlens, chunk_offsets=chunk_offsets)
    _dense_chunk_metadata_cache[key] = metadata
    if len(_dense_chunk_metadata_cache) > _DENSE_CHUNK_METADATA_CACHE_LIMIT:
        _dense_chunk_metadata_cache.popitem(last=False)
    return metadata
