# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""One-shot diagnostics for prepared, pre-clip optimizer gradients."""

import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from megatron.core import parallel_state
from megatron.core.transformer.fsdp_dtensor_checkpoint import handle_experts_in_state_dict
from megatron.core.utils import unwrap_model


def _optimizer_coordinates() -> dict[str, int]:
    """Return the active parallel ranks and world sizes."""

    return {
        "global_rank": torch.distributed.get_rank(),
        "tp_rank": parallel_state.get_tensor_model_parallel_rank(),
        "tp_size": parallel_state.get_tensor_model_parallel_world_size(),
        "dp_rank": parallel_state.get_data_parallel_rank(with_context_parallel=True),
        "dp_size": parallel_state.get_data_parallel_world_size(with_context_parallel=True),
        "ep_rank": parallel_state.get_expert_model_parallel_rank(),
        "ep_size": parallel_state.get_expert_model_parallel_world_size(),
        "edp_rank": parallel_state.get_expert_data_parallel_rank(),
        "edp_size": parallel_state.get_expert_data_parallel_world_size(),
        "etp_rank": parallel_state.get_expert_tensor_parallel_rank(),
        "etp_size": parallel_state.get_expert_tensor_parallel_world_size(),
        "pp_rank": parallel_state.get_pipeline_model_parallel_rank(),
        "pp_size": parallel_state.get_pipeline_model_parallel_world_size(),
    }


def _iter_leaf_optimizers(optimizer, path: tuple[int, ...] = ()):
    """Yield optimizer leaves together with their stable child-index path."""

    children = getattr(optimizer, "chained_optimizers", None)
    if children:
        for child_index, child in enumerate(children):
            yield from _iter_leaf_optimizers(child, (*path, child_index))
        return
    yield path, optimizer


def _optimizer_membership(optimizer) -> tuple[dict[int, dict[str, Any]], list[str]]:
    """Map prepared optimizer tensors to their optimizer implementation and group."""

    membership: dict[int, dict[str, Any]] = {}
    errors = []
    for path, leaf in _iter_leaf_optimizers(optimizer):
        raw_optimizer = getattr(leaf, "optimizer", None)
        if raw_optimizer is None:
            continue
        for group_index, group in enumerate(getattr(raw_optimizer, "param_groups", ())):
            group_metadata = {}
            for key in (
                "is_expert_parallel",
                "lr",
                "lr_mult",
                "momentum",
                "wd_mult",
                "weight_decay",
            ):
                value = group.get(key)
                if isinstance(value, (bool, int, float, str)) or value is None:
                    group_metadata[key] = value
            for parameter in group.get("params", ()):
                parameter_id = id(parameter)
                entry = {
                    "leaf_path": list(path),
                    "wrapper_type": type(leaf).__name__,
                    "optimizer_type": type(raw_optimizer).__name__,
                    "group_index": group_index,
                    "group": group_metadata,
                }
                if parameter_id in membership and membership[parameter_id] != entry:
                    errors.append(
                        "prepared optimizer parameter appears in multiple leaves: "
                        f"{membership[parameter_id]} and {entry}"
                    )
                membership[parameter_id] = entry
    return membership, errors


def _canonical_name_to_parameter(model) -> tuple[dict[str, torch.nn.Parameter], list[str]]:
    """Build canonical parameter names in one pass, including global expert indices."""

    unwrapped_chunks = [unwrap_model(model_chunk) for model_chunk in model]
    errors = []
    if parallel_state.get_pipeline_model_parallel_world_size() != 1:
        errors.append("gradient fingerprint currently requires pipeline parallel size 1")

    name_to_parameter = {}
    multiple_chunks = len(unwrapped_chunks) > 1
    for chunk_index, model_chunk in enumerate(unwrapped_chunks):
        prefix = f"model_chunk{chunk_index}." if multiple_chunks else ""
        for name, parameter in model_chunk.named_parameters():
            canonical_name = prefix + name
            if canonical_name in name_to_parameter:
                errors.append(f"duplicate local parameter name: {canonical_name}")
            name_to_parameter[canonical_name] = parameter

    num_experts = None
    if unwrapped_chunks:
        config = getattr(unwrapped_chunks[0], "config", None)
        num_experts = getattr(config, "num_moe_experts", None)
    name_to_parameter = handle_experts_in_state_dict(name_to_parameter, num_experts)
    return name_to_parameter, errors


def _tensor_statistics(
    gradient: torch.Tensor, weight: torch.Tensor
) -> tuple[dict[str, Any], list[str]]:
    """Compute compact scalar statistics without copying a full tensor to the host."""

    errors = []
    gradient_flat = gradient.detach().reshape(-1)
    weight_flat = weight.detach().reshape(-1)
    if gradient_flat.numel() == 0:
        return {
            "numel": 0,
            "sum": 0.0,
            "l1": 0.0,
            "sq_sum": 0.0,
            "min": 0.0,
            "max": 0.0,
            "weight_sq_sum": 0.0,
            "grad_weight_dot": 0.0,
        }, ["encountered an empty prepared gradient"]
    if gradient_flat.shape != weight_flat.shape:
        errors.append(
            f"prepared gradient shape {tuple(gradient.shape)} differs from "
            f"master weight shape {tuple(weight.shape)}"
        )

    gradient_l2 = torch.linalg.vector_norm(gradient_flat, ord=2, dtype=torch.float64)
    gradient_scalars = torch.stack(
        (
            gradient_flat.sum(dtype=torch.float64),
            torch.linalg.vector_norm(gradient_flat, ord=1, dtype=torch.float64),
            gradient_l2,
            gradient_flat.min().double(),
            gradient_flat.max().double(),
        )
    )
    gradient_sum, gradient_l1, gradient_l2_value, gradient_min, gradient_max = (
        gradient_scalars.cpu().tolist()
    )

    weight_l2_value = torch.linalg.vector_norm(weight_flat, ord=2, dtype=torch.float64).item()
    grad_weight_dot = None
    if gradient_flat.shape == weight_flat.shape and gradient_flat.device == weight_flat.device:
        grad_for_dot = gradient_flat
        weight_for_dot = weight_flat
        if grad_for_dot.dtype != torch.float32:
            grad_for_dot = grad_for_dot.float()
        if weight_for_dot.dtype != torch.float32:
            weight_for_dot = weight_for_dot.float()
        grad_weight_dot = torch.dot(grad_for_dot, weight_for_dot).item()
    else:
        errors.append(
            f"cannot compute grad/weight dot product across devices "
            f"{gradient_flat.device} and {weight_flat.device}"
        )

    statistics = {
        "numel": gradient_flat.numel(),
        "sum": gradient_sum,
        "l1": gradient_l1,
        "sq_sum": gradient_l2_value * gradient_l2_value,
        "min": gradient_min,
        "max": gradient_max,
        "weight_sq_sum": weight_l2_value * weight_l2_value,
        "grad_weight_dot": grad_weight_dot,
    }
    if not all(
        value is None or math.isfinite(value) for key, value in statistics.items() if key != "numel"
    ):
        errors.append("prepared gradient statistics contain a non-finite value")
    return statistics, errors


def _aggregate_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate independent tensor shards using additive scalar moments."""

    numel = sum(record["stats"]["numel"] for record in records)
    sq_sum = sum(record["stats"]["sq_sum"] for record in records)
    weight_sq_sum = sum(record["stats"]["weight_sq_sum"] for record in records)
    grad_weight_dots = [record["stats"]["grad_weight_dot"] for record in records]
    grad_weight_dot = (
        None if any(value is None for value in grad_weight_dots) else sum(grad_weight_dots)
    )
    return {
        "numel": numel,
        "sum": sum(record["stats"]["sum"] for record in records),
        "l1": sum(record["stats"]["l1"] for record in records),
        "l2": math.sqrt(max(sq_sum, 0.0)),
        "sq_sum": sq_sum,
        "min": min(record["stats"]["min"] for record in records),
        "max": max(record["stats"]["max"] for record in records),
        "weight_l2": math.sqrt(max(weight_sq_sum, 0.0)),
        "weight_sq_sum": weight_sq_sum,
        "grad_weight_dot": grad_weight_dot,
        "signed_mean": (
            sum(record["stats"]["sum"] for record in records) / numel if numel else 0.0
        ),
        "mean_abs": (sum(record["stats"]["l1"] for record in records) / numel if numel else 0.0),
        "rms": math.sqrt(max(sq_sum, 0.0) / numel) if numel else 0.0,
    }


def _canonicalize_records(payloads: list[dict[str, Any]]):
    """Combine TP shards and select one copy of replicated prepared gradients."""

    records_by_name = defaultdict(list)
    errors = []
    warnings = []
    for payload in payloads:
        errors.extend(
            f"rank {payload['coordinates']['global_rank']}: {message}"
            for message in payload["errors"]
        )
        for record in payload["records"]:
            records_by_name[record["name"]].append(record)

    canonical = []
    for name in sorted(records_by_name):
        records = sorted(
            records_by_name[name],
            key=lambda record: (
                record["coordinates"]["tp_rank"],
                record["coordinates"]["ep_rank"],
                record["coordinates"]["global_rank"],
            ),
        )
        sample = records[0]
        metadata = sample["metadata"]
        coordinates = sample["coordinates"]
        is_expert = not metadata["allreduce"]
        partition_dim = metadata["partition_dim"]
        is_tp_sharded = not is_expert and isinstance(partition_dim, int) and partition_dim >= 0
        is_expert_tp_sharded = is_expert and metadata["expert_tp"] and coordinates["etp_size"] > 1

        if is_tp_sharded or is_expert_tp_sharded:
            selected = records
            aggregation = "tp_shards" if is_tp_sharded else "expert_tp_shards"
            expected_count = coordinates["tp_size"] if is_tp_sharded else coordinates["etp_size"]
        else:
            selected = [
                min(
                    records,
                    key=lambda record: (
                        record["coordinates"]["tp_rank"],
                        record["coordinates"]["global_rank"],
                    ),
                )
            ]
            aggregation = "expert_owner" if is_expert else "tp_replicated"
            expected_count = 1 if is_expert else coordinates["tp_size"]

        if len(records) != expected_count:
            warnings.append(
                f"{name}: aggregation={aggregation} has {len(records)} records, "
                f"expected {expected_count}"
            )
        dense_owner_dp_ranks = sorted(
            {
                record["coordinates"]["dp_rank"]
                for record in records
                if record["metadata"]["allreduce"]
            }
        )
        owner_aligned = not dense_owner_dp_ranks or len(dense_owner_dp_ranks) == 1
        if not owner_aligned:
            warnings.append(
                f"{name}: dense TP peers use different LayerWise DP owners "
                f"{dense_owner_dp_ranks}"
            )

        replica_spread = {}
        if aggregation == "tp_replicated" and len(records) > 1:
            for field in ("sum", "l1", "sq_sum", "min", "max", "grad_weight_dot"):
                values = [
                    record["stats"][field]
                    for record in records
                    if record["stats"][field] is not None
                ]
                replica_spread[field] = max(values) - min(values) if values else None

        canonical.append(
            {
                "name": name,
                "aggregation": aggregation,
                "stats": _aggregate_statistics(selected),
                "metadata": metadata,
                "source_global_ranks": [record["coordinates"]["global_rank"] for record in records],
                "owner_dp_ranks": dense_owner_dp_ranks,
                "owner_edp_ranks": sorted(
                    {record["coordinates"]["edp_rank"] for record in records}
                ),
                "owner_aligned_across_tp": owner_aligned,
                "replica_spread": replica_spread,
                "optimizer": sample["optimizer"],
                "local_shapes": [record["local_shape"] for record in records],
            }
        )
    if not canonical:
        errors.append("no prepared LayerWise FP32 master gradients were captured")
    return canonical, errors, warnings


@torch.no_grad()
def dump_pre_clip_optimizer_grad_fingerprint(
    *, model, optimizer, iteration: int, found_inf: bool, output_path: str
) -> None:
    """Write one canonical full-model view of prepared LayerWise gradients.

    The hook runs after optimizer.prepare_grads() and before clipping. It reads only
    locally owned FP32 masters, avoiding invalid portions of reduce-scattered model grad
    buffers. The output is intended for controlled TP1/TP2 diagnostics.
    """

    coordinates = _optimizer_coordinates()
    local_errors = []
    records = []
    try:
        membership, membership_errors = _optimizer_membership(optimizer)
        local_errors.extend(membership_errors)
        name_to_parameter, name_errors = _canonical_name_to_parameter(model)
        local_errors.extend(name_errors)
        for name, model_parameter in sorted(name_to_parameter.items()):
            master_parameter = getattr(model_parameter, "main_param", None)
            if master_parameter is None or master_parameter.grad is None:
                continue
            statistics, statistic_errors = _tensor_statistics(
                master_parameter.grad, master_parameter
            )
            local_errors.extend(f"{name}: {message}" for message in statistic_errors)
            optimizer_entry = membership.get(id(master_parameter))
            if optimizer_entry is None:
                local_errors.append(f"{name}: FP32 master is absent from optimizer leaves")
                optimizer_entry = {
                    "leaf_path": [],
                    "wrapper_type": None,
                    "optimizer_type": None,
                    "group_index": None,
                    "group": {},
                }
            partition_dim = getattr(model_parameter, "partition_dim", None)
            if hasattr(partition_dim, "item"):
                partition_dim = partition_dim.item()
            records.append(
                {
                    "name": name,
                    "local_shape": list(master_parameter.shape),
                    "gradient_dtype": str(master_parameter.grad.dtype).removeprefix("torch."),
                    "weight_dtype": str(master_parameter.dtype).removeprefix("torch."),
                    "metadata": {
                        "allreduce": bool(getattr(model_parameter, "allreduce", True)),
                        "tensor_model_parallel": bool(
                            getattr(model_parameter, "tensor_model_parallel", False)
                        ),
                        "expert_tp": bool(getattr(model_parameter, "expert_tp", False)),
                        "sequence_parallel": bool(
                            getattr(model_parameter, "sequence_parallel", False)
                        ),
                        "partition_dim": partition_dim,
                        "partition_stride": getattr(model_parameter, "partition_stride", None),
                        "is_qkv": bool(getattr(model_parameter, "is_qkv", False)),
                        "managed_by_layer_wise": bool(
                            getattr(model_parameter, "is_managed_by_layer_wise_optimizer", False)
                        ),
                    },
                    "coordinates": coordinates,
                    "optimizer": optimizer_entry,
                    "stats": statistics,
                }
            )
    except (AssertionError, AttributeError, KeyError, RuntimeError, TypeError, ValueError) as error:
        local_errors.append(f"local capture failed with {type(error).__name__}: {error}")

    payload = {"coordinates": coordinates, "errors": local_errors, "records": records}
    world_size = torch.distributed.get_world_size()
    gathered_payloads = [None] * world_size if coordinates["global_rank"] == 0 else None
    torch.distributed.gather_object(payload, gathered_payloads, dst=0)

    root_error = None
    if coordinates["global_rank"] == 0:
        try:
            canonical, errors, warnings = _canonicalize_records(gathered_payloads)
            result = {
                "schema_version": 1,
                "scope": "prepared_layerwise_fp32_master_gradients_before_clipping",
                "iteration": iteration,
                "found_inf": bool(found_inf),
                "world_size": world_size,
                "topology": coordinates,
                "raw_record_count": sum(
                    len(rank_payload["records"]) for rank_payload in gathered_payloads
                ),
                "canonical_parameter_count": len(canonical),
                "errors": errors,
                "warnings": warnings,
                "canonical": canonical,
                "rank_payloads": gathered_payloads,
            }
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(f"refusing to overwrite existing fingerprint: {destination}")
            temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
            with temporary.open("x", encoding="utf-8") as output:
                json.dump(result, output, sort_keys=True, separators=(",", ":"))
                output.write("\n")
            os.replace(temporary, destination)
            if errors:
                root_error = f"gradient fingerprint structural errors: {errors[:8]}"
            print(
                "MCORE_PRE_CLIP_GRAD_FINGERPRINT_GATE="
                f"{'FAIL' if errors else 'PASS'} path={destination} "
                f"raw_records={result['raw_record_count']} "
                f"canonical_params={result['canonical_parameter_count']} "
                f"warnings={len(warnings)} errors={len(errors)}",
                flush=True,
            )
        except (
            AssertionError,
            AttributeError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            root_error = f"{type(error).__name__}: {error}"
            print(f"MCORE_PRE_CLIP_GRAD_FINGERPRINT_GATE=FAIL error={root_error}", flush=True)

    status = torch.tensor(
        [1 if root_error else 0],
        dtype=torch.int32,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    torch.distributed.broadcast(status, src=0)
    if status.item():
        raise RuntimeError(
            root_error or "rank 0 failed to materialize the prepared-gradient fingerprint"
        )
