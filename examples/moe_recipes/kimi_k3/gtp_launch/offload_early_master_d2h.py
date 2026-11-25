# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Early device-to-host copy of offloaded master weights (Kimi-K3 GTP recipe).

With chunked optimizer-state offload Megatron keeps every selected master weight on the GPU
from the start of the optimizer step until the next training step has staged its parameters,
and only then copies all of them to pinned host memory in one go
(``ChunkedOptimizerStateOffloader._schedule_master_d2h``, called from
``offload_for_forward``). Right after that call the level-1 timer of the forward pass
synchronizes the device, so every rank waits for its own copy (full model: 32-42 GiB,
0.19-0.47 s per step) and then, in the timer's barrier, for the slowest rank's copy
(0.34-0.50 s per step in all at 8K context), with nothing else running.

A master weight is final as soon as the optimizer has stepped the state chunk that holds it;
everything that follows until the release only reads it. This module therefore splits the
copy from the release, without modifying any Megatron file:

  * right after a state chunk has been stepped (where the offloader queues the chunk's state
    copy), the chunk's masters are also copied to their pinned host buffers, on the same
    transfer stream, while the GPU tensors stay bound and readable. These copies overlap the
    remaining chunks of the optimizer step;
  * at release time the copy of a master is skipped when it was copied early and has not been
    touched since (same storage, same tensor version); it is only rebound to the host buffer,
    exactly as before. Any other master is copied as before.

The bytes that reach host memory are the same, so training is bit-identical; only the time at
which they are copied changes. What the version check cannot see is a write through
``param.data`` between the optimizer step and the release. Nothing in the Kimi-K3
configuration does that (``qk_clip`` is off), and the verify mode checks it at run time:

  MCORE_EARLY_MASTER_D2H         0 (default) | 1
  MCORE_EARLY_MASTER_D2H_VERIFY  N (default 0): at the first N releases of every offloader (-1:
                                 always) compare each early host copy with the GPU tensor bit
                                 for bit before releasing it; a mismatch is fatal. Blocking and
                                 slow, meant for the first steps of a run (before the measured
                                 window) and for correctness runs.

Enable through ``kimi_k3_gtp_entry.py``.
"""

import os

import torch

COUNTS = {
    "early_copies": 0,
    "early_bytes": 0,
    "skipped_at_release": 0,
    "copied_at_release": 0,
    "verified": 0,
    "releases": 0,
}


def enabled():
    value = os.environ.get("MCORE_EARLY_MASTER_D2H", "0")
    if value not in ("0", "1"):
        raise ValueError(f"MCORE_EARLY_MASTER_D2H must be 0 or 1, got {value!r}")
    return value == "1"


def verify_releases():
    return int(os.environ.get("MCORE_EARLY_MASTER_D2H_VERIFY", "0"))


def install():
    """Patch ``ChunkedOptimizerStateOffloader`` in place. Idempotent."""
    from megatron.core.optimizer.cpu_offloading import chunked_optimizer_state_offload as module

    cls = module.ChunkedOptimizerStateOffloader
    if getattr(cls, "_xm_installed", False):
        return cls
    original_step = cls.step
    original_schedule_state_d2h = cls._schedule_state_d2h
    verify_limit = verify_releases()

    def master_tensors(self, param):
        """``(state_key, gpu_tensor, version_owner)`` of the CUDA-bound masters of ``param``."""
        if id(param) in self._explicit_master_param_ids and param.data.is_cuda:
            yield None, param.data, param

    def announce(self, text):
        if not getattr(cls, "_xm_announced", False):
            cls._xm_announced = True
            rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "?"))
            print(f"[EARLY MASTER D2H] rank={rank} {text}", flush=True)

    def early_copy(self, params):
        """Copy the masters of ``params`` to their host buffers; the GPU binding stays."""
        early = self.__dict__.setdefault("_xm_early", {})
        transfers = [
            (param, state_key, gpu_tensor, owner)
            for param in params
            if id(param) in self._selected_param_ids
            for state_key, gpu_tensor, owner in master_tensors(self, param)
        ]
        if not transfers:
            return
        self._d2h_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._d2h_stream):
            for param, state_key, gpu_tensor, owner in transfers:
                cpu_tensor = self._cpu_buffer_for_master(param, gpu_tensor)
                cpu_tensor.copy_(gpu_tensor, non_blocking=True)
                gpu_tensor.record_stream(self._d2h_stream)
                early[(id(param), state_key)] = (gpu_tensor.data_ptr(), owner._version, cpu_tensor)
                COUNTS["early_copies"] += 1
                COUNTS["early_bytes"] += gpu_tensor.numel() * gpu_tensor.element_size()
        announce(self, f"first early master copy ({len(transfers)} tensors of one state chunk)")

    def _schedule_state_d2h(self, params):
        original_schedule_state_d2h(self, params)
        if getattr(self, "_xm_in_step", False):
            early_copy(self, params)

    @torch.no_grad()
    def step(self):
        self._xm_in_step = True
        try:
            return original_step(self)
        finally:
            self._xm_in_step = False

    def _schedule_master_d2h(self):
        """The original release, minus the copies that were already made and are still valid."""
        if not self._selected_params:
            return
        self._validate_master_storage()
        self._wait_master_h2d()

        early = self.__dict__.setdefault("_xm_early", {})
        transfers = []
        for param in self._selected_params:
            for state_key, gpu_tensor, owner in master_tensors(self, param):
                record = early.get((id(param), state_key))
                current = (
                    record is not None
                    and record[0] == gpu_tensor.data_ptr()
                    and record[1] == owner._version
                    and record[2] is self._cpu_master.get(param)
                )
                transfers.append((param, state_key, gpu_tensor, current))
        early.clear()
        if not transfers:
            return

        releases = self.__dict__.get("_xm_releases", 0)
        self._xm_releases = releases + 1
        COUNTS["releases"] += 1
        verify = verify_limit < 0 or releases < verify_limit
        if verify:
            # Blocking: wait for the early copies, then compare every one of them on the GPU.
            self._d2h_stream.synchronize()
            mismatched = []
            for param, state_key, gpu_tensor, current in transfers:
                if current:
                    host = self._cpu_master[param].to(gpu_tensor.device, non_blocking=True)
                    if not torch.equal(host, gpu_tensor):
                        mismatched.append(tuple(gpu_tensor.shape))
                    del host
                    COUNTS["verified"] += 1
            if mismatched:
                raise RuntimeError(
                    f"early master D2H: {len(mismatched)} host copies differ from the GPU master at "
                    f"release time (first shapes {mismatched[:3]}); something wrote a master weight "
                    "after its state chunk was stepped without changing its tensor version"
                )

        self._master_weights_resident = False
        self._d2h_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._d2h_stream):
            for param, state_key, gpu_tensor, current in transfers:
                cpu_tensor = self._cpu_buffer_for_master(param, gpu_tensor)
                if current:
                    COUNTS["skipped_at_release"] += 1
                else:
                    cpu_tensor.copy_(gpu_tensor, non_blocking=True)
                    COUNTS["copied_at_release"] += 1
                gpu_tensor.record_stream(self._d2h_stream)
                if state_key is None:
                    param.data = cpu_tensor
                else:
                    self.optimizer.state[param][state_key] = cpu_tensor

    cls.step = step
    cls._schedule_state_d2h = _schedule_state_d2h
    cls._schedule_master_d2h = _schedule_master_d2h
    cls._xm_installed = True
    return cls
