# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Launch wrapper of the Kimi-K3 GTP 1M-context recipe.

Runs an unmodified Megatron entry point (``pretrain_hybrid.py`` of the checkout on
``PYTHONPATH``) with three launch-side changes. None of them changes arithmetic, precision
or which checks run, and no file of the checkout is modified:

1. ``MCORE_NUMA_MEMBIND=gpu``: bind the rank's host memory to the NUMA node of its GPU
   before torch is imported, so that pinned buffers (optimizer-state and activation offload)
   do not end up on the other socket.
2. Deferred validation (always on, ``MCORE_DEFERRED_VALIDATION_MODE=passthrough`` switches
   it off): Megatron's NaN/Inf checks of the loss and of every gradient bucket normally turn
   their result into a Python bool on the spot, one blocking host/GPU synchronisation per
   check. The rerun state machine can record such checks and validate them later
   (``capture_deferred_validations`` / ``run_deferred_validations``; Megatron uses it for the
   full-iteration CUDA graph). The wrapper applies that facility to the eager
   forward-backward function: every check still runs with the same value, message and fatal
   flag, after the forward-backward pass returns and before the optimizer update.
3. ``MCORE_EARLY_MASTER_D2H=1``: with chunked optimizer-state offload, copy a state chunk's
   master weights to host memory right after the chunk is stepped instead of all at once
   before the next forward pass (``offload_early_master_d2h.py``). Same bytes, earlier.

Environment:
  MCORE_NUMA_MEMBIND               off (default) | gpu
  MCORE_DEFERRED_VALIDATION_MODE   deferred (default) | passthrough
  MCORE_EARLY_MASTER_D2H           0 (default) | 1
  MCORE_EARLY_MASTER_D2H_VERIFY    N (default 0; -1 = always): bitwise check of the early
                                   copies at the first N releases of every offloader
  MCORE_DEFERRED_VALIDATION_ENTRY  script to run, relative to the Megatron checkout
                                   (default: pretrain_hybrid.py)

Usage (one process per GPU, started by torch.distributed.run; see render_launcher.py):
  python -m torch.distributed.run ... examples/moe_recipes/kimi_k3/gtp_launch/kimi_k3_gtp_entry.py ARGS
"""

import functools
import os
import runpy
import sys

VALIDATION_KEY = "eager_training"
MODES = ("deferred", "passthrough")
MEMBIND_MODES = ("off", "gpu")
MPOL_BIND = 2
# set_mempolicy system call numbers (glibc has no wrapper for it).
_SET_MEMPOLICY_NR = {"x86_64": 238, "aarch64": 237}


def _rank():
    return os.environ.get("RANK", os.environ.get("LOCAL_RANK", "?"))


def _mode():
    mode = os.environ.get("MCORE_DEFERRED_VALIDATION_MODE", "deferred")
    if mode not in MODES:
        raise ValueError(f"MCORE_DEFERRED_VALIDATION_MODE must be one of {MODES}, got {mode!r}")
    return mode


def _wrap_forward_backward(forward_backward_func, mode="deferred"):
    from megatron.core.full_cuda_graph import FullCudaGraphWrapper
    from megatron.core.rerun_state_machine import get_rerun_state_machine

    if mode == "passthrough":
        return forward_backward_func

    if isinstance(forward_backward_func, FullCudaGraphWrapper):
        # The full-iteration CUDA graph wrapper already defers validations itself.
        return forward_backward_func

    @functools.wraps(forward_backward_func)
    def deferred_validation_forward_backward(*args, **kwargs):
        rerun_state_machine = get_rerun_state_machine()
        with rerun_state_machine.capture_deferred_validations(VALIDATION_KEY):
            result = forward_backward_func(*args, **kwargs)
        try:
            # Raises here (before the optimizer update) if any captured check fails.
            rerun_state_machine.run_deferred_validations(VALIDATION_KEY)
        finally:
            rerun_state_machine.clear_deferred_validations(VALIDATION_KEY)
        return result

    return deferred_validation_forward_backward


def install():
    """Wrap the factory the training and evaluation loops use to get their step function.

    Returns the root of the Megatron checkout that was imported."""
    import megatron.training.training as training

    original_factory = training.get_forward_backward_func
    mode = _mode()

    @functools.wraps(original_factory)
    def get_forward_backward_func(*args, **kwargs):
        return _wrap_forward_backward(original_factory(*args, **kwargs), mode)

    training.get_forward_backward_func = get_forward_backward_func
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(training.__file__))))


def _parse_cpulist(text):
    """Parse a kernel CPU or node list such as ``0-71,109-143`` into a set of ints."""
    values = set()
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            low, high = part.split("-", 1)
            values.update(range(int(low), int(high) + 1))
        else:
            values.add(int(part))
    return values


def _gpu_numa_node(gpu_index, nvidia_smi="nvidia-smi", sysfs="/sys/bus/pci/devices"):
    """NUMA node of a GPU, from its PCI device in sysfs."""
    import subprocess

    listing = subprocess.run(
        [nvidia_smi, "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    buses = {}
    for line in listing.splitlines():
        index, _, bus = line.partition(",")
        buses[int(index)] = bus.strip().lower()
    if gpu_index not in buses:
        raise RuntimeError(
            f"[NUMA MEMBIND] GPU index {gpu_index} not in nvidia-smi listing {sorted(buses)}"
        )
    bus = buses[gpu_index]
    # nvidia-smi prints an 8-digit PCI domain, sysfs uses 4 digits.
    candidates = [bus, bus[4:]] if len(bus.split(":")[0]) == 8 else [bus]
    for name in candidates:
        path = os.path.join(sysfs, name, "numa_node")
        if os.path.exists(path):
            with open(path) as handle:
                return int(handle.read().strip())
    raise RuntimeError(f"[NUMA MEMBIND] no sysfs PCI device for GPU {gpu_index} (bus {bus})")


def _local_gpu_index(local_rank=None):
    """Index of this rank's GPU as nvidia-smi numbers it."""
    if local_rank is None:
        local_rank = int(os.environ["LOCAL_RANK"])
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = [d.strip() for d in visible.split(",") if d.strip()]
    if devices and all(d.isdigit() for d in devices):
        return int(devices[local_rank])
    return local_rank


def _memory_numa_nodes(sysfs="/sys/devices/system/node"):
    """NUMA nodes that have memory. A machine can list more nodes than sockets (the GB300
    nodes report node0-node1 with memory and further memory-less nodes); a memory policy can
    only name nodes with memory (binding to a memory-less node is EINVAL)."""
    path = os.path.join(sysfs, "has_memory")
    if not os.path.exists(path):
        path = os.path.join(sysfs, "online")
    with open(path) as handle:
        return sorted(_parse_cpulist(handle.read()))


def _syscall_number(table, machine=None):
    import platform

    machine = machine or platform.machine()
    if machine not in table:
        raise RuntimeError(f"[NUMA MEMBIND] no system call number known for machine {machine!r}")
    return table[machine]


def _set_mempolicy_bind(node, libc=None, machine=None):
    """set_mempolicy(MPOL_BIND, {node}) for the calling thread; threads and processes created
    afterwards inherit it."""
    import ctypes

    libc = libc or ctypes.CDLL(None, use_errno=True)
    words = node // 64 + 1
    mask = (ctypes.c_ulong * words)()
    mask[node // 64] = 1 << (node % 64)
    # The kernel reads maxnode - 1 bits of the mask.
    result = libc.syscall(
        _syscall_number(_SET_MEMPOLICY_NR, machine), MPOL_BIND, mask, 64 * words + 1
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(
            error,
            f"[NUMA MEMBIND] set_mempolicy(MPOL_BIND, node {node}) failed: {os.strerror(error)}",
        )


def bind_memory_to_numa_node():
    """MCORE_NUMA_MEMBIND=gpu: allocate this rank's host memory only on the NUMA node its GPU
    hangs off.

    At full scale part of a rank's pinned host buffers ends up on the other NUMA node (the
    default policy falls back to it when the local node has no free page), and the ranks whose
    optimizer state or offloaded activations sit there copy them more slowly; the whole job
    waits for the slowest rank. With MPOL_BIND the kernel reclaims page cache on the local node
    instead of falling back; if the node cannot hold the rank's memory the allocation fails
    rather than spilling. Must run before torch is imported so that every thread inherits the
    policy. Fails loudly rather than leaving the rank unbound.
    """
    mode = os.environ.get("MCORE_NUMA_MEMBIND", "off")
    if mode not in MEMBIND_MODES:
        raise ValueError(f"MCORE_NUMA_MEMBIND must be one of {MEMBIND_MODES}, got {mode!r}")
    if mode == "off":
        return None
    gpu_index = _local_gpu_index()
    node = _gpu_numa_node(gpu_index)
    memory_nodes = _memory_numa_nodes()
    if node not in memory_nodes:
        raise RuntimeError(
            f"[NUMA MEMBIND] GPU NUMA node {node} has no memory (nodes with memory: {memory_nodes})"
        )
    _set_mempolicy_bind(node)
    print(
        f"[NUMA MEMBIND] rank={os.environ.get('RANK', '?')} "
        f"local_rank={os.environ.get('LOCAL_RANK', '?')} gpu={gpu_index} gpu_numa_node={node} "
        f"mode={mode} memory bound to node {node}",
        flush=True,
    )
    return node


def main():
    # Before anything imports torch: threads created later inherit the memory policy.
    bind_memory_to_numa_node()
    megatron_root = install()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import offload_early_master_d2h

    early_master_d2h = offload_early_master_d2h.enabled()
    if early_master_d2h:
        offload_early_master_d2h.install()
    verify = offload_early_master_d2h.verify_releases()
    print(
        f"[EARLY MASTER D2H] rank={_rank()} "
        f"{'installed, verify releases ' + str(verify) if early_master_d2h else 'off'}",
        flush=True,
    )
    script = os.path.join(
        megatron_root, os.environ.get("MCORE_DEFERRED_VALIDATION_ENTRY", "pretrain_hybrid.py")
    )
    print(
        f"[DEFERRED VALIDATION] rank={_rank()} mode={_mode()} wrapping "
        f"get_forward_backward_func; running {script}",
        flush=True,
    )
    sys.argv[0] = script
    # The script's own directory comes first on sys.path when it is run directly.
    sys.path.insert(0, os.path.dirname(script))
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
