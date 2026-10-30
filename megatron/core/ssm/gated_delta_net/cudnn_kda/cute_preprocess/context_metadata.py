# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Chain metadata the CuTe DSL CP dispatcher reads from FLA's ``FLACPContext``.

The fused kernels exchange state through symmetric memory, so every rank must make the same
dispatch decision (``"fused"`` / ``"noop"`` / refuse) from the same immutable data.  PR #6511
attached the GLOBAL packed offsets (``global_cu_seqlens_cpu`` / ``global_num_seqs``) plus a few
private memo fields to the context object with :func:`prepare_cp_context_metadata`, called from
the attention module with the model config and the global ``cu_seqlens``.  The cuDNN KDA
producer only sees the context, whose ``build_cp_context`` keeps the rank-local offsets, so
:func:`ensure_cp_context_metadata` reconstructs the global offsets with one ``all_gather_object``
over the CP group the first time a context object is seen, and attaches them.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from megatron.core.transformer.enums import CudaGraphModule

_READY_ATTR = "_cute_preprocess_metadata_ready"


def _gdn_is_cuda_graphed(config) -> bool:
    """Return whether this GDN attention region is included in CUDA-graph capture."""
    impl = getattr(config, "cuda_graph_impl", "none")
    if impl == "none":
        return False
    if impl == "full_iteration":
        return True
    modules = getattr(config, "cuda_graph_modules", None)
    return not modules or CudaGraphModule.attn in modules


_GLOBAL_CU_SEQLENS_UNSET = object()


def _logical_global_cu_seqlens_cpu(cu_seqlens: torch.Tensor | None) -> torch.Tensor | None:
    """Return a detached CPU copy of logical global packed offsets."""
    if cu_seqlens is None:
        return None
    if cu_seqlens.dim() != 1:
        raise ValueError("CuTe CP global cu_seqlens must be one-dimensional.")

    cpu = cu_seqlens.detach()
    if cpu.device.type != "cpu":
        cpu = cpu.to(device="cpu")
    if cpu.dtype != torch.long:
        cpu = cpu.to(dtype=torch.long)
    if cpu.numel() < 2:
        raise ValueError("CuTe CP global cu_seqlens requires at least two entries.")

    logical_entries = cpu.numel()
    final_offset = int(cpu[-1])
    while logical_entries > 1 and int(cpu[logical_entries - 2]) == final_offset:
        logical_entries -= 1
    cpu = cpu[:logical_entries].clone()

    if int(cpu[0]) != 0:
        raise ValueError("CuTe CP global cu_seqlens must start at zero.")
    if bool(((cpu[1:] - cpu[:-1]) <= 0).any()):
        raise ValueError("CuTe CP global cu_seqlens must be strictly increasing.")
    return cpu


def _same_cpu_tensor_values(lhs: torch.Tensor | None, rhs: torch.Tensor | None) -> bool:
    """Value equality of two optional CPU tensors (``None`` only equals ``None``)."""
    if lhs is None or rhs is None:
        return lhs is rhs
    return lhs.shape == rhs.shape and lhs.dtype == rhs.dtype and torch.equal(lhs, rhs)


def prepare_cp_context_metadata(
    context: object,
    *,
    config: object | None = None,
    global_num_sequences: int | None = None,
    global_cu_seqlens: object = _GLOBAL_CU_SEQLENS_UNSET,
) -> object:
    """Attach the CuTe CP dispatcher's hints to an FLA CP context (PR #6511's helper, verbatim).

    Sets ``global_num_seqs`` / ``global_cu_seqlens_cpu``, bumps ``_cutedsl_metadata_generation``
    and clears the chain / window memos when either changed, and always refreshes
    ``_cutedsl_cuda_graph_enabled`` from ``config`` (``None`` means no CUDA graphs).
    """
    if global_cu_seqlens is _GLOBAL_CU_SEQLENS_UNSET:
        global_cu_seqlens_cpu = getattr(context, "global_cu_seqlens_cpu", None)
    else:
        global_cu_seqlens_cpu = _logical_global_cu_seqlens_cpu(global_cu_seqlens)

    if global_cu_seqlens_cpu is not None:
        inferred_num_sequences = int(global_cu_seqlens_cpu.numel() - 1)
        if global_num_sequences is None:
            global_num_sequences = inferred_num_sequences
        elif int(global_num_sequences) != inferred_num_sequences:
            raise ValueError("CuTe CP global_num_sequences does not match global cu_seqlens.")

    if global_num_sequences is not None:
        global_num_sequences = int(global_num_sequences)
        if global_num_sequences < 1:
            raise ValueError("CuTe CP metadata requires at least one sequence.")

    metadata_changed = getattr(
        context, "global_num_seqs", None
    ) != global_num_sequences or not _same_cpu_tensor_values(
        getattr(context, "global_cu_seqlens_cpu", None), global_cu_seqlens_cpu
    )
    if metadata_changed:
        context.global_num_seqs = global_num_sequences
        context.global_cu_seqlens_cpu = global_cu_seqlens_cpu
        context._cutedsl_metadata_generation = (
            getattr(context, "_cutedsl_metadata_generation", 0) + 1
        )
        context._cutedsl_chain_memo = {}
        context._cutedsl_window_memo = {}
    context._cutedsl_cuda_graph_enabled = _gdn_is_cuda_graphed(config)
    return context


def gather_global_cu_seqlens(context: object) -> torch.Tensor:
    """Reconstruct the GLOBAL packed offsets from every rank's local ones (collective).

    FLA's ``get_cp_cu_seqlens`` gives rank ``r`` the local offsets
    ``clamp(global, r*part, (r+1)*part) - r*part`` with duplicates removed: the vector always
    runs ``[0, ..., part]``, its interior entries are document boundaries strictly inside the
    rank's token share, and a boundary exactly at ``r*part`` is visible only as
    ``is_first_rank`` on rank ``r`` (the previous rank then has ``is_last_rank``).  Gathering
    ``(local offsets, is_first_rank)`` from every rank therefore recovers the global vector
    exactly, up to tokens FLA itself drops when the total is not divisible by the world size
    (it floors ``part``; the reconstruction ends at ``world_size * part`` like every rank's view).

    Every rank must call this (``all_gather_object`` over ``context.group``).  Cost is one
    small host collective per context object; :func:`ensure_cp_context_metadata` memoizes it.
    """
    group = context.group
    world_size = dist.get_world_size(group)
    cu_cpu = getattr(context, "cu_seqlens_cpu", None)
    source = cu_cpu if cu_cpu is not None else context.cu_seqlens
    local = [int(x) for x in source.tolist()]
    if len(local) < 2 or local[0] != 0 or local[-1] <= 0:
        raise ValueError(f"CuTe CP context has malformed local cu_seqlens {local}")
    payload = (local, bool(getattr(context, "is_first_rank", False)))
    gathered: list = [None] * world_size
    dist.all_gather_object(gathered, payload, group=group)

    part = local[-1]
    offsets = {0, part * world_size}
    for r, (cu, first) in enumerate(gathered):
        if cu[0] != 0 or cu[-1] != part:
            raise RuntimeError(
                f"CuTe CP ranks disagree on the token share: rank {r} has local cu_seqlens "
                f"{cu} while this rank's share is {part} tokens"
            )
        offsets.update(r * part + x for x in cu[1:-1])
        if first and r > 0:
            offsets.add(r * part)
    return torch.tensor(sorted(offsets), dtype=torch.long)


def ensure_cp_context_metadata(context: object, *, config: object | None = None) -> object:
    """Attach the dispatcher metadata once per context object (collective on the first call).

    Uses the context's own ``global_cu_seqlens_cpu`` when a caller already attached one (PR
    #6511's attention module does), else reconstructs it with :func:`gather_global_cu_seqlens`.
    The decision to gather is a pure function of the context object every rank built the same
    way, so the collective is entered by all ranks together.
    """
    if getattr(context, _READY_ATTR, False):
        return context
    if getattr(context, "global_cu_seqlens_cpu", None) is None:
        prepare_cp_context_metadata(
            context, config=config, global_cu_seqlens=gather_global_cu_seqlens(context)
        )
    else:
        prepare_cp_context_metadata(context, config=config)
    setattr(context, _READY_ATTR, True)
    return context
