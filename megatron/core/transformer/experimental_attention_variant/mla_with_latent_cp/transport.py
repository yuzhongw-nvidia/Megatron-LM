# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Differentiable explicit-group all-gather transport for MLA latent CP."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Iterator, Protocol

import torch
import torch.distributed as dist
from torch import Tensor

from .layout import PhaseSpec
from .utils import _require


@dataclass(frozen=True)
class PayloadLease:
    """A consumer-stream-ordered latent payload and its CP group-rank owner."""

    owner: int
    tensor: Tensor


class LatentCPTransport(Protocol):
    """Transport seam for owner-ordered latent CP payloads."""

    def iter_payloads(
        self,
        local_payload: Tensor,
        phase_plan: tuple[PhaseSpec, ...],
        consumer_stream: torch.cuda.Stream | None = None,
    ) -> Iterator[PayloadLease]:
        """Yield one payload lease per phase, ordered on the requested consumer stream."""
        ...


_COMMUNICATION_STREAM_LOCK = threading.RLock()
_COMMUNICATION_STREAMS: dict[tuple[int, int], torch.cuda.Stream] = {}


def _communication_stream(payload: Tensor) -> torch.cuda.Stream | None:
    """Return one process/device-local stream shared by latent CP collectives."""

    if not payload.is_cuda:
        return None
    device_index = payload.device.index
    _require(device_index is not None, "CUDA payload must have a concrete device")
    key = (os.getpid(), device_index)
    with _COMMUNICATION_STREAM_LOCK:
        stream = _COMMUNICATION_STREAMS.get(key)
        if stream is None:
            with torch.cuda.device(payload.device):
                stream = torch.cuda.Stream(device=payload.device)
            _COMMUNICATION_STREAMS[key] = stream
        return stream


@dataclass
class _PendingCollective:
    """Lifetime and readiness state for one asynchronous collective."""

    work: Any | None = None
    input_tensor: Tensor | None = None
    output_tensor: Tensor | None = None
    waited: bool = False

    def wait_on_consumer_stream(
        self, consumer_stream: torch.cuda.Stream | None = None
    ) -> None:
        """Order the collective output on exactly one consumer stream."""

        _require(not self.waited, "an asynchronous collective was waited twice")
        _require(self.work is not None, "asynchronous collective lost its work handle")
        if consumer_stream is None:
            self.work.wait()
        else:
            with torch.cuda.stream(consumer_stream):
                self.work.wait()
        self.work = None
        self.input_tensor = None
        self.output_tensor = None
        self.waited = True


def _launch_all_gather(
    payload: Tensor,
    cp_group: dist.ProcessGroup,
    cp_size: int,
    communication_stream: torch.cuda.Stream | None,
    pending: _PendingCollective,
) -> Tensor:
    """Launch one flattened rank-major all-gather without consuming its output."""

    _require(payload.ndim > 0 and payload.is_contiguous(), "payload must be contiguous")
    input_flat = payload.view(-1)
    gathered_flat = torch.empty(
        cp_size * input_flat.numel(), dtype=payload.dtype, device=payload.device
    )
    if communication_stream is not None:
        producer_stream = torch.cuda.current_stream(payload.device)
        with torch.cuda.stream(communication_stream):
            communication_stream.wait_stream(producer_stream)
            work = dist.all_gather_into_tensor(
                gathered_flat, input_flat, group=cp_group, async_op=True
            )
            payload.record_stream(communication_stream)
            gathered_flat.record_stream(communication_stream)
    else:
        work = dist.all_gather_into_tensor(
            gathered_flat, input_flat, group=cp_group, async_op=True
        )
    _require(work is not None, "asynchronous all-gather returned no work handle")
    pending.work = work
    pending.input_tensor = input_flat
    pending.output_tensor = gathered_flat
    return gathered_flat.view(cp_size, *payload.shape)


def _launch_reduce_scatter(
    gathered_gradient: Tensor,
    cp_group: dist.ProcessGroup,
    cp_size: int,
    payload_shape: tuple[int, ...],
    communication_stream: torch.cuda.Stream | None,
    pending: _PendingCollective,
) -> Tensor:
    """Reduce rank-major gathered gradients back to the owning CP rank."""

    _require(
        tuple(gathered_gradient.shape) == (cp_size, *payload_shape),
        "all-gather gradient shape disagrees with the original payload",
    )
    input_flat = gathered_gradient.contiguous().view(-1)
    output_elements = input_flat.numel() // cp_size
    output_flat = torch.empty(
        output_elements,
        dtype=gathered_gradient.dtype,
        device=gathered_gradient.device,
    )
    if communication_stream is not None:
        producer_stream = torch.cuda.current_stream(gathered_gradient.device)
        with torch.cuda.stream(communication_stream):
            communication_stream.wait_stream(producer_stream)
            work = dist.reduce_scatter_tensor(
                output_flat, input_flat, group=cp_group, async_op=True
            )
            input_flat.record_stream(communication_stream)
            output_flat.record_stream(communication_stream)
    else:
        work = dist.reduce_scatter_tensor(
            output_flat, input_flat, group=cp_group, async_op=True
        )
    _require(work is not None, "asynchronous reduce-scatter returned no work handle")
    pending.work = work
    pending.input_tensor = input_flat
    pending.output_tensor = output_flat
    return output_flat.view(payload_shape)


class _LatentAllGatherExchange(torch.autograd.Function):
    """One all-gather forward with its exact reduce-scatter-SUM backward."""

    @staticmethod
    def forward(
        ctx: Any,
        payload: Tensor,
        cp_group: dist.ProcessGroup,
        cp_size: int,
        communication_stream: torch.cuda.Stream | None,
        pending: _PendingCollective,
    ) -> Tensor:
        """Publish one rank-major gather while retaining an external wait handle."""

        ctx.cp_group = cp_group
        ctx.cp_size = cp_size
        ctx.payload_shape = tuple(payload.shape)
        ctx.communication_stream = communication_stream
        return _launch_all_gather(
            payload, cp_group, cp_size, communication_stream, pending
        )

    @staticmethod
    def backward(
        ctx: Any, gathered_gradient: Tensor
    ) -> tuple[Tensor, None, None, None, None]:
        """Sum every consumer's owner slice and return this rank's payload gradient."""

        pending = _PendingCollective()
        grad_payload = _launch_reduce_scatter(
            gathered_gradient,
            ctx.cp_group,
            ctx.cp_size,
            ctx.payload_shape,
            ctx.communication_stream,
            pending,
        )
        pending.wait_on_consumer_stream()
        return grad_payload, None, None, None, None


class AllGatherTransport:
    """One asynchronous gather, local bypass, and one first-remote dependency."""

    def __init__(self, cp_group: dist.ProcessGroup):
        self.cp_group = cp_group
        self.group_ranks = tuple(dist.get_process_group_ranks(cp_group))
        self.rank = dist.get_rank(cp_group)
        self.size = dist.get_world_size(cp_group)
        _require(len(self.group_ranks) == self.size, "invalid CP peer list")

    def iter_payloads(
        self,
        local_payload: Tensor,
        phase_plan: tuple[PhaseSpec, ...],
        consumer_stream: torch.cuda.Stream | None = None,
    ) -> Iterator[PayloadLease]:
        """Yield each payload after ordering the selected consumer behind its receive."""
        _require(len(phase_plan) == self.size, "phase-plan length must equal CP size")
        for phase_index, phase in enumerate(phase_plan):
            expected_owner = (self.rank - phase_index) % self.size
            _require(
                phase.phase == phase_index, "phase-plan indices must be contiguous"
            )
            _require(
                phase.owner == expected_owner,
                "phase-plan owner order disagrees with the CP group",
            )

        local_phase, *remote_phases = phase_plan
        if self.size == 1:
            yield PayloadLease(owner=local_phase.owner, tensor=local_payload)
            return

        pending = _PendingCollective()
        communication_stream = _communication_stream(local_payload)
        gathered = _LatentAllGatherExchange.apply(
            local_payload,
            self.cp_group,
            self.size,
            communication_stream,
            pending,
        )
        # The local phase consumes the original tensor, not the gathered local chunk.
        yield PayloadLease(owner=local_phase.owner, tensor=local_payload)

        # Every raw remote slice is first consumed by the one stream named here.
        # Projection outputs may subsequently fan out to alternating attention streams.
        pending.wait_on_consumer_stream(consumer_stream)
        for phase in remote_phases:
            yield PayloadLease(owner=phase.owner, tensor=gathered[phase.owner])
