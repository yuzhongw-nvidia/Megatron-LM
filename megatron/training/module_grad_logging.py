# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Module-level forward/backward summary logging for DSv4 debug.

This logger is inert unless ``MCORE_DSV4_MODULE_GRAD_DEBUG`` is enabled.  It is
intended for one-step numerical debugging, where recording compact summaries at
module boundaries is enough to locate the first backward divergence without
dumping full activation tensors.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
from collections import defaultdict
from typing import Any, Iterable

import torch

from megatron.core.utils import unwrap_model


def _truthy(value: str | None) -> bool:
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


def module_grad_logging_enabled() -> bool:
    return _truthy(os.environ.get("MCORE_DSV4_MODULE_GRAD_DEBUG"))


def should_log_module_grads(iteration: int) -> bool:
    if not module_grad_logging_enabled():
        return False
    iterations = os.environ.get("MCORE_DSV4_MODULE_GRAD_ITERATIONS", "").strip()
    if iterations:
        requested = {int(item.strip()) for item in iterations.split(",") if item.strip()}
        return iteration in requested
    interval = int(os.environ.get("MCORE_DSV4_MODULE_GRAD_INTERVAL", "0") or "0")
    return True if interval <= 0 else iteration % interval == 0


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", "0"))


def _iter_named_tensors(obj: Any, prefix: str = "") -> Iterable[tuple[str, torch.Tensor]]:
    if isinstance(obj, torch.Tensor):
        yield prefix or "tensor", obj
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_named_tensors(value, next_prefix)
        return
    if isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            next_prefix = f"{prefix}.{idx}" if prefix else str(idx)
            yield from _iter_named_tensors(value, next_prefix)
        return
    if dataclasses.is_dataclass(obj):
        for field in dataclasses.fields(obj):
            next_prefix = f"{prefix}.{field.name}" if prefix else field.name
            yield from _iter_named_tensors(getattr(obj, field.name), next_prefix)


_CURRENT_VALID_MASK: torch.Tensor | None = None


def _numeric_summary(tensor: torch.Tensor, prefix: str = "") -> dict[str, Any]:
    detached = tensor.detach()
    out: dict[str, Any] = {}
    if detached.numel() == 0:
        return out
    if detached.is_floating_point() or detached.is_complex():
        values = detached.float()
        finite = torch.isfinite(values)
        safe_values = values[finite] if finite.any() else values.reshape(-1)[:0]
        out.update(
            {
                f"{prefix}sum": float(values.sum().item()),
                f"{prefix}mean": float(values.mean().item()),
                f"{prefix}std": float(values.std(unbiased=False).item()),
                f"{prefix}min": float(values.min().item()),
                f"{prefix}max": float(values.max().item()),
                f"{prefix}abs_max": float(values.abs().max().item()),
                f"{prefix}l2": float(torch.linalg.vector_norm(values).item()),
                f"{prefix}nan_count": int(torch.isnan(values).sum().item()),
                f"{prefix}inf_count": int(torch.isinf(values).sum().item()),
                f"{prefix}finite_count": int(finite.sum().item()),
            }
        )
        if safe_values.numel() > 0:
            out[f"{prefix}finite_mean"] = float(safe_values.mean().item())
            out[f"{prefix}finite_abs_max"] = float(safe_values.abs().max().item())
    else:
        values = detached.reshape(-1).to(torch.int64)
        out.update(
            {
                f"{prefix}sum": int(values.sum().item()),
                f"{prefix}min": int(values.min().item()),
                f"{prefix}max": int(values.max().item()),
            }
        )
    return out


def _valid_view(tensor: torch.Tensor) -> tuple[torch.Tensor, int] | None:
    if _CURRENT_VALID_MASK is None or tensor.dim() == 0:
        return None
    mask = _CURRENT_VALID_MASK.detach().reshape(-1).to(torch.bool)
    if mask.numel() == 0:
        return None
    detached = tensor.detach()
    if detached.numel() == mask.numel():
        return detached.reshape(-1)[mask], int(mask.sum().item())
    if detached.dim() >= 2 and detached.numel() % detached.shape[-1] == 0:
        rows = detached.reshape(-1, detached.shape[-1])
        if rows.shape[0] == mask.numel():
            return rows[mask], int(mask.sum().item())
    return None


def _summary(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach()
    out: dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": int(detached.numel()),
    }
    out.update(_numeric_summary(detached))
    valid_result = _valid_view(detached)
    if valid_result is not None:
        valid, valid_tokens = valid_result
        out["valid_numel"] = int(valid.numel())
        out["valid_tokens"] = valid_tokens
        out.update(_numeric_summary(valid, prefix="valid_"))
    return out


def _sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "tensor"


def _summaries(obj: Any, prefix: str) -> list[dict[str, Any]]:
    return [
        {"path": name, **_summary(tensor)}
        for name, tensor in _iter_named_tensors(obj, prefix)
        if tensor is not None
    ]


class ModuleGradLogger:
    """Capture module boundary summaries with forward and backward hooks."""

    def __init__(self, save_dir: str):
        self._save_dir = save_dir
        pattern = os.environ.get(
            "MCORE_DSV4_MODULE_GRAD_NAME_REGEX",
            r"(decoder|mtp)\..*(self_attention|mlp|router|experts|shared_experts|"
            r"linear_|layernorm|norm|core_attention)",
        )
        self._name_regex = re.compile(pattern)
        self._max_events = int(os.environ.get("MCORE_DSV4_MODULE_GRAD_MAX_EVENTS", "20000"))
        self._dump_tensors = _truthy(os.environ.get("MCORE_DSV4_MODULE_GRAD_DUMP_TENSORS"))
        dump_name_pattern = os.environ.get("MCORE_DSV4_MODULE_GRAD_DUMP_NAME_REGEX", r"$^")
        dump_path_pattern = os.environ.get("MCORE_DSV4_MODULE_GRAD_DUMP_PATH_REGEX", r".*")
        self._dump_name_regex = re.compile(dump_name_pattern)
        self._dump_path_regex = re.compile(dump_path_pattern)
        self._hooks = []
        self._events: list[dict[str, Any]] = []
        self._tensor_dumps: list[tuple[str, torch.Tensor]] = []
        self._call_counters: defaultdict[tuple[str, str], int] = defaultdict(int)

    def _module_class(self, module: torch.nn.Module) -> str:
        cls = module.__class__
        return f"{cls.__module__}.{cls.__name__}"

    def _append_event(self, event: dict[str, Any]) -> None:
        if len(self._events) < self._max_events:
            self._events.append(event)

    def _call_index(self, module_name: str, phase: str) -> int:
        key = (module_name, phase)
        idx = self._call_counters[key]
        self._call_counters[key] += 1
        return idx

    def _should_dump_tensor(self, module_name: str, tensor_path: str) -> bool:
        return (
            self._dump_tensors
            and self._dump_name_regex.search(module_name) is not None
            and self._dump_path_regex.search(tensor_path) is not None
        )

    def _summaries_with_optional_dump(
        self,
        obj: Any,
        prefix: str,
        module_name: str,
        phase: str,
        call_index: int,
    ) -> list[dict[str, Any]]:
        rows = []
        for tensor_path, tensor in _iter_named_tensors(obj, prefix):
            if tensor is None:
                continue
            row = {"path": tensor_path, **_summary(tensor)}
            if self._should_dump_tensor(module_name, tensor_path):
                file_name = (
                    f"{len(self._tensor_dumps):05d}_{phase}_"
                    f"{_sanitize_filename(module_name)}_call{call_index}_"
                    f"{_sanitize_filename(tensor_path)}.pt"
                )
                row["tensor_file"] = file_name
                self._tensor_dumps.append((file_name, tensor.detach().cpu()))
            rows.append(row)
        return rows

    def _make_forward_hook(self, model_chunk_name: str, module_name: str, module_class: str):
        def hook(_, args, kwargs, output):
            call_index = self._call_index(module_name, "forward")
            self._append_event(
                {
                    "phase": "forward",
                    "model_chunk": model_chunk_name,
                    "module": module_name,
                    "module_class": module_class,
                    "call_index": call_index,
                    "tensors": [
                        *self._summaries_with_optional_dump(
                            args if isinstance(args, tuple) else (args,),
                            "args",
                            module_name,
                            "forward",
                            call_index,
                        ),
                        *self._summaries_with_optional_dump(
                            kwargs, "kwargs", module_name, "forward", call_index
                        ),
                        *self._summaries_with_optional_dump(
                            output, "output", module_name, "forward", call_index
                        ),
                    ],
                }
            )

        return hook

    def _make_backward_hook(self, model_chunk_name: str, module_name: str, module_class: str):
        def hook(_, grad_input, grad_output):
            call_index = self._call_index(module_name, "backward")
            self._append_event(
                {
                    "phase": "backward",
                    "model_chunk": model_chunk_name,
                    "module": module_name,
                    "module_class": module_class,
                    "call_index": call_index,
                    "tensors": [
                        *self._summaries_with_optional_dump(
                            grad_input, "grad_input", module_name, "backward", call_index
                        ),
                        *self._summaries_with_optional_dump(
                            grad_output, "grad_output", module_name, "backward", call_index
                        ),
                    ],
                }
            )

        return hook

    def register_hooks(self, model: list[torch.nn.Module]) -> None:
        assert len(self._hooks) == 0
        for model_chunk_id, model_chunk in enumerate(model):
            model_chunk_name = f"model_chunk{model_chunk_id}"
            unwrapped_model_chunk = unwrap_model(model_chunk)
            for module_name, module in unwrapped_model_chunk.named_modules():
                if not module_name or not self._name_regex.search(module_name):
                    continue
                module_class = self._module_class(module)
                self._hooks.append(
                    module.register_forward_hook(
                        self._make_forward_hook(model_chunk_name, module_name, module_class),
                        with_kwargs=True,
                    )
                )
                self._hooks.append(
                    module.register_full_backward_hook(
                        self._make_backward_hook(model_chunk_name, module_name, module_class)
                    )
                )

    def remove_hooks(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def save(self, iteration: int) -> None:
        if not self._events and not self._tensor_dumps:
            return
        out_dir = os.path.join(
            self._save_dir, "module_grads", f"iter_{iteration:07d}"
        )
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"rank{_rank():05d}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for event in self._events:
                f.write(json.dumps(event, sort_keys=True) + "\n")
        if self._tensor_dumps:
            tensor_dir = os.path.join(
                self._save_dir,
                "module_grad_tensors",
                f"iter_{iteration:07d}",
                f"rank{_rank():05d}",
            )
            os.makedirs(tensor_dir, exist_ok=True)
            if _CURRENT_VALID_MASK is not None:
                torch.save(
                    _CURRENT_VALID_MASK.detach().cpu(),
                    os.path.join(tensor_dir, "valid_mask.pt"),
                )
            for file_name, tensor in self._tensor_dumps:
                torch.save(tensor, os.path.join(tensor_dir, file_name))
        self._events.clear()
        self._tensor_dumps.clear()
        self._call_counters.clear()


_LOGGER: ModuleGradLogger | None = None


def enable_module_grad_logging(model: list[torch.nn.Module], save_dir: str) -> None:
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = ModuleGradLogger(save_dir)
    _LOGGER.register_hooks(model)


def disable_module_grad_logging() -> None:
    assert _LOGGER is not None
    _LOGGER.remove_hooks()


def save_module_grads(iteration: int) -> None:
    assert _LOGGER is not None
    _LOGGER.save(iteration)


def set_module_grad_valid_mask(loss_mask: torch.Tensor | None) -> None:
    global _CURRENT_VALID_MASK
    if not module_grad_logging_enabled() or loss_mask is None:
        _CURRENT_VALID_MASK = None
        return
    _CURRENT_VALID_MASK = loss_mask.detach().to(torch.bool)


def clear_module_grad_valid_mask() -> None:
    global _CURRENT_VALID_MASK
    _CURRENT_VALID_MASK = None
