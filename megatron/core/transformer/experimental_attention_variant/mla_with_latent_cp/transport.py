# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Differentiable collective/P2P transports for MLA latent CP."""

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
    """A consumer-stream-ordered latent payload and its original CP owner."""

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
    """Return one process/device-local stream shared by latent ring transports."""

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
class _PendingExchange:
    """CUDA readiness state for one prefetched receive."""

    works: tuple[Any, ...] = ()
    send_tensor: Tensor | None = None
    waited: bool = False

    def wait_on_consumer_stream(
        self, consumer_stream: torch.cuda.Stream | None = None
    ) -> None:
        """Order one prefetched receive on an explicit or current consumer stream."""

        _require(not self.waited, "a prefetched ring payload was consumed twice")
        if consumer_stream is None:
            for work in self.works:
                work.wait()
        else:
            with torch.cuda.stream(consumer_stream):
                for work in self.works:
                    work.wait()
        self.works = ()
        self.waited = True
        self.send_tensor = None


def _launch_ring_exchange(
    payload: Tensor,
    cp_group: dist.ProcessGroup,
    send_peer: int,
    receive_peer: int,
    communication_stream: torch.cuda.Stream | None,
    pending: _PendingExchange,
    wait_for_compute_stream: bool,
) -> Tensor:
    """Launch one explicit-group exchange, isolated from the attention stream."""

    receive = torch.empty_like(payload)
    operations = [
        dist.P2POp(dist.isend, payload, send_peer, group=cp_group),
        dist.P2POp(dist.irecv, receive, receive_peer, group=cp_group),
    ]
    if communication_stream is None:
        for work in dist.batch_isend_irecv(operations):
            work.wait()
        pending.send_tensor = payload
        return receive

    producer_stream = torch.cuda.current_stream(payload.device)
    with torch.cuda.stream(communication_stream):
        if wait_for_compute_stream:
            # current_stream() is the communication stream inside this context.
            communication_stream.wait_stream(producer_stream)
        works = tuple(dist.batch_isend_irecv(operations))
        _require(works, "a CUDA P2P exchange returned no work handle")
        payload.record_stream(communication_stream)
        receive.record_stream(communication_stream)
    pending.works = works
    pending.send_tensor = payload
    return receive


@dataclass
class _PendingCollective:
    """Lifetime and readiness state for one asynchronous all-gather."""

    work: Any | None = None
    input_tensor: Tensor | None = None
    output_tensor: Tensor | None = None
    waited: bool = False

    def wait_on_consumer_stream(
        self, consumer_stream: torch.cuda.Stream | None = None
    ) -> None:
        """Order the gather output on exactly one first-consumer stream."""

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


@dataclass
class _PendingDirectReverse:
    """Lifetime and readiness state for one fixed-order reverse P2P batch."""

    works: tuple[Any, ...]
    send_tensors: tuple[Tensor, ...]
    receive_tensor: Tensor
    waited: bool = False

    def wait_on_consumer_stream(
        self, consumer_stream: torch.cuda.Stream | None = None
    ) -> None:
        """Wait every actual batch Work on the backward caller stream."""

        _require(not self.waited, "a reverse P2P batch was waited twice")
        if consumer_stream is None:
            for work in self.works:
                work.wait()
        else:
            with torch.cuda.stream(consumer_stream):
                for work in self.works:
                    work.wait()
        self.works = ()
        self.send_tensors = ()
        self.waited = True


def _launch_direct_reverse(
    output_gradients: tuple[Tensor | None, ...],
    reverse_group: dist.ProcessGroup,
    group_ranks: tuple[int, ...],
    rank: int,
    payload_shape: tuple[int, ...],
    payload_dtype: torch.dtype,
    payload_device: torch.device,
    communication_stream: torch.cuda.Stream | None,
) -> tuple[Tensor, _PendingDirectReverse]:
    """Route remote-view gradients directly to owners in fixed phase order."""

    cp_size = len(group_ranks)
    _require(
        len(output_gradients) == cp_size - 1,
        "remote gradient count disagrees with CP size",
    )
    sends = tuple(
        (
            torch.zeros(payload_shape, dtype=payload_dtype, device=payload_device)
            if gradient is None
            else gradient.contiguous()
        )
        for gradient in output_gradients
    )
    _require(
        all(tuple(send.shape) == payload_shape for send in sends),
        "remote gradient shape disagrees with the original payload",
    )
    receives = torch.empty(
        (cp_size - 1, *payload_shape),
        dtype=payload_dtype,
        device=payload_device,
    )
    operations = []
    for phase, send in enumerate(sends, start=1):
        owner_peer = group_ranks[(rank - phase) % cp_size]
        consumer_peer = group_ranks[(rank + phase) % cp_size]
        operations.extend(
            (
                dist.P2POp(dist.isend, send, owner_peer, group=reverse_group),
                dist.P2POp(
                    dist.irecv,
                    receives[phase - 1],
                    consumer_peer,
                    group=reverse_group,
                ),
            )
        )

    if communication_stream is None:
        works = tuple(dist.batch_isend_irecv(operations))
    else:
        producer_stream = torch.cuda.current_stream(payload_device)
        with torch.cuda.stream(communication_stream):
            communication_stream.wait_stream(producer_stream)
            works = tuple(dist.batch_isend_irecv(operations))
            for send in sends:
                send.record_stream(communication_stream)
            receives.record_stream(communication_stream)
    _require(works, "a reverse P2P batch returned no work handle")
    return receives, _PendingDirectReverse(works, sends, receives)


class _LatentAllGatherDirectP2PExchange(torch.autograd.Function):
    """One all-gather forward with a fixed-order direct P2P backward."""

    @staticmethod
    def forward(
        ctx: Any,
        payload: Tensor,
        cp_group: dist.ProcessGroup,
        reverse_group: dist.ProcessGroup,
        group_ranks: tuple[int, ...],
        rank: int,
        communication_stream: torch.cuda.Stream | None,
        pending: _PendingCollective,
    ) -> tuple[Tensor, ...]:
        """Return owner-ordered remote views while retaining one gather Work."""

        cp_size = len(group_ranks)
        gathered = _launch_all_gather(
            payload, cp_group, cp_size, communication_stream, pending
        )
        ctx.set_materialize_grads(False)
        ctx.reverse_group = reverse_group
        ctx.group_ranks = group_ranks
        ctx.rank = rank
        ctx.payload_shape = tuple(payload.shape)
        ctx.payload_dtype = payload.dtype
        ctx.payload_device = payload.device
        ctx.communication_stream = communication_stream
        return tuple(gathered[(rank - phase) % cp_size] for phase in range(1, cp_size))

    @staticmethod
    def backward(ctx: Any, *output_gradients: Tensor | None) -> tuple[Any, ...]:
        """Send each phase gradient to its owner and sum received own-gradients."""

        caller_stream = (
            torch.cuda.current_stream(ctx.payload_device)
            if ctx.communication_stream is not None
            else None
        )
        received, pending = _launch_direct_reverse(
            output_gradients,
            ctx.reverse_group,
            ctx.group_ranks,
            ctx.rank,
            ctx.payload_shape,
            ctx.payload_dtype,
            ctx.payload_device,
            ctx.communication_stream,
        )
        pending.wait_on_consumer_stream(caller_stream)
        grad_payload = received.sum(dim=0)
        return grad_payload, None, None, None, None, None, None


class AllGatherDirectP2PTransport:
    """One async gather, local bypass, and fixed-order direct reverse P2P."""

    def __init__(
        self,
        cp_group: dist.ProcessGroup,
        reverse_group: dist.ProcessGroup | None = None,
    ):
        self.cp_group = cp_group
        self.group_ranks = tuple(dist.get_process_group_ranks(cp_group))
        self.rank = dist.get_rank(cp_group)
        self.size = dist.get_world_size(cp_group)
        _require(len(self.group_ranks) == self.size, "invalid CP peer list")
        self.reverse_group = cp_group if reverse_group is None else reverse_group
        reverse_ranks = tuple(dist.get_process_group_ranks(self.reverse_group))
        _require(
            reverse_ranks == self.group_ranks,
            "reverse CP communicator must preserve the forward CP rank order",
        )
        _require(
            dist.get_rank(self.reverse_group) == self.rank, "CP group ranks disagree"
        )

    def iter_payloads(
        self,
        local_payload: Tensor,
        phase_plan: tuple[PhaseSpec, ...],
        consumer_stream: torch.cuda.Stream | None = None,
    ) -> Iterator[PayloadLease]:
        """Yield local data directly and remote gather views after one dependency."""

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
        remote_payloads = _LatentAllGatherDirectP2PExchange.apply(
            local_payload,
            self.cp_group,
            self.reverse_group,
            self.group_ranks,
            self.rank,
            communication_stream,
            pending,
        )
        yield PayloadLease(owner=local_phase.owner, tensor=local_payload)
        pending.wait_on_consumer_stream(consumer_stream)
        for phase, remote_payload in zip(remote_phases, remote_payloads, strict=True):
            yield PayloadLease(owner=phase.owner, tensor=remote_payload)


class _LatentRingExchange(torch.autograd.Function):
    """One explicit-group clockwise ring hop with the exact reverse backward hop."""

    @staticmethod
    def forward(
        ctx: Any,
        payload: Tensor,
        cp_group: dist.ProcessGroup,
        previous_peer: int,
        next_peer: int,
        communication_stream: torch.cuda.Stream | None,
        pending: _PendingExchange,
        wait_for_compute_stream: bool,
    ) -> Tensor:
        """Prefetch the preceding owner's payload on the communication stream."""
        ctx.cp_group = cp_group
        ctx.previous_peer = previous_peer
        ctx.next_peer = next_peer
        ctx.communication_stream = communication_stream
        return _launch_ring_exchange(
            payload,
            cp_group,
            next_peer,
            previous_peer,
            communication_stream,
            pending,
            wait_for_compute_stream,
        )

    @staticmethod
    def backward(
        ctx: Any, grad_receive: Tensor
    ) -> tuple[Tensor, None, None, None, None, None, None]:
        """Route the received-payload gradient through the reverse ring hop."""
        grad_receive = grad_receive.contiguous()
        pending = _PendingExchange()
        grad_payload = _launch_ring_exchange(
            grad_receive,
            ctx.cp_group,
            ctx.previous_peer,
            ctx.next_peer,
            ctx.communication_stream,
            pending,
            True,
        )
        pending.wait_on_consumer_stream()
        return grad_payload, None, None, None, None, None, None


class P2PRingTransport:
    """One-hop-prefetched P2P transport with an explicit reverse autograd ring."""

    def __init__(self, cp_group: dist.ProcessGroup):
        self.cp_group = cp_group
        self.group_ranks = tuple(dist.get_process_group_ranks(cp_group))
        self.rank = dist.get_rank(cp_group)
        self.size = dist.get_world_size(cp_group)
        _require(len(self.group_ranks) == self.size, "invalid CP peer list")
        self.previous_peer = self.group_ranks[(self.rank - 1) % self.size]
        self.next_peer = self.group_ranks[(self.rank + 1) % self.size]

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
                "phase-plan owner order disagrees with the P2P ring",
            )

        payload = local_payload
        pending: _PendingExchange | None = None
        communication_stream = (
            _communication_stream(local_payload) if self.size > 1 else None
        )
        for phase_index, phase in enumerate(phase_plan):
            if pending is not None:
                pending.wait_on_consumer_stream(consumer_stream)

            next_payload: Tensor | None = None
            next_pending: _PendingExchange | None = None
            if phase_index + 1 < self.size:
                next_pending = _PendingExchange()
                next_payload = _LatentRingExchange.apply(
                    payload,
                    self.cp_group,
                    self.previous_peer,
                    self.next_peer,
                    communication_stream,
                    next_pending,
                    phase_index == 0,
                )

            yield PayloadLease(owner=phase.owner, tensor=payload)
            if next_payload is not None and next_pending is not None:
                payload = next_payload
                pending = next_pending
