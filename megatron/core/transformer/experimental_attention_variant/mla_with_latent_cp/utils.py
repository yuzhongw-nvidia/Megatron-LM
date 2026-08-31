# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared errors, qualification constants, and numerical helpers for MLA latent CP."""

from __future__ import annotations

import math
from typing import Final, TypeAlias

import torch
from torch import Tensor

from megatron.core.transformer.enums import AttnBackend

CUDNN_FRONTEND_SOURCE_REV: Final[str] = "0a14b7181d129d30e7bad34b8c3ed0a0c995e23d"
"""Immutable source revision used to implement and qualify the cuDNN Graph adapter."""

QualifiedBackendTuple: TypeAlias = tuple[AttnBackend, str, str, tuple[int, int]]

# This feature is fail-closed. These are the complete, exact tuples backed by the checked-in
# qualification contract; wildcards, version ranges, and runtime overrides are unsupported.
QUALIFIED_BACKEND_CONFIGS: Final[tuple[QualifiedBackendTuple, ...]] = (
    (AttnBackend.fused, "1.22.1", "9.21.0", (9, 0)),
    (AttnBackend.fused, "1.26.0", "9.25.0", (10, 0)),
    (AttnBackend.flash, "4.0.0b11", "flash-attn-4==4.0.0b11", (10, 0)),
)


class LatentCPError(RuntimeError):
    """Base error for the experimental latent-CP implementation."""


class BackendNotQualifiedError(LatentCPError):
    """Raised when the exact backend/package/device tuple is not qualified."""


class BackendPlanNotSupportedError(LatentCPError):
    """Raised before P2P when a public backend reports that a phase plan is unsupported."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"MLAWithLatentCP: {message}")


def scatter_upper_phase(
    output: Tensor, lse: Tensor, back_indices: Tensor, local_tokens: int
) -> tuple[Tensor, Tensor]:
    """Functionally scatter an upper rectangular phase into full local Q rows."""

    output_full = torch.zeros(
        (local_tokens, *output.shape[1:]), dtype=torch.float32, device=output.device
    ).index_copy(0, back_indices, output.float())
    lse_full = torch.full(
        (local_tokens, lse.size(1)), -torch.inf, dtype=torch.float32, device=lse.device
    ).index_copy(0, back_indices, lse.float())
    return output_full, lse_full


def _merge_attention_forward_impl(
    output_a: Tensor,
    lse_a: Tensor,
    output_b: Tensor,
    lse_b: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute one FP32 online-softmax merge and retain its two weights."""

    merged_lse = torch.logaddexp(lse_a, lse_b)
    valid_a = torch.isfinite(lse_a) & torch.isfinite(merged_lse)
    valid_b = torch.isfinite(lse_b) & torch.isfinite(merged_lse)
    delta_a = torch.where(
        valid_a, lse_a - merged_lse, torch.full_like(lse_a, -torch.inf)
    )
    delta_b = torch.where(
        valid_b, lse_b - merged_lse, torch.full_like(lse_b, -torch.inf)
    )
    weight_a = torch.exp(delta_a)
    weight_b = torch.exp(delta_b)
    merged_output = output_a * weight_a.unsqueeze(-1) + output_b * weight_b.unsqueeze(
        -1
    )
    return merged_output, merged_lse, weight_a, weight_b


@torch.compile(fullgraph=True)
def _compiled_merge_attention_forward(
    output_a: Tensor,
    lse_a: Tensor,
    output_b: Tensor,
    lse_b: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Fuse the CUDA online-softmax merge into one compiled region."""

    return _merge_attention_forward_impl(output_a, lse_a, output_b, lse_b)


def _merge_attention_backward_impl(
    grad_output: Tensor,
    grad_lse: Tensor,
    output_a: Tensor,
    output_b: Tensor,
    merged_output: Tensor,
    weight_a: Tensor,
    weight_b: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute the analytical gradients for one FP32 partial merge."""

    grad_output_a = grad_output * weight_a.unsqueeze(-1)
    grad_output_b = grad_output * weight_b.unsqueeze(-1)
    output_term_a = torch.sum(grad_output * (output_a - merged_output), dim=-1)
    output_term_b = torch.sum(grad_output * (output_b - merged_output), dim=-1)
    grad_lse_a = weight_a * (grad_lse + output_term_a)
    grad_lse_b = weight_b * (grad_lse + output_term_b)
    return grad_output_a, grad_lse_a, grad_output_b, grad_lse_b


@torch.compile(fullgraph=True)
def _compiled_merge_attention_backward(
    grad_output: Tensor,
    grad_lse: Tensor,
    output_a: Tensor,
    output_b: Tensor,
    merged_output: Tensor,
    weight_a: Tensor,
    weight_b: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Fuse the CUDA analytical merge backward into one compiled region."""

    return _merge_attention_backward_impl(
        grad_output,
        grad_lse,
        output_a,
        output_b,
        merged_output,
        weight_a,
        weight_b,
    )


class _AttentionPartialMerge(torch.autograd.Function):
    """Merge two FP32 attention partials with an analytical backward."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        output_a: Tensor,
        lse_a: Tensor,
        output_b: Tensor,
        lse_b: Tensor,
    ) -> tuple[Tensor, Tensor]:
        merge = (
            _compiled_merge_attention_forward
            if output_a.is_cuda
            else _merge_attention_forward_impl
        )
        merged_output, merged_lse, weight_a, weight_b = merge(
            output_a, lse_a, output_b, lse_b
        )
        ctx.save_for_backward(output_a, output_b, merged_output, weight_a, weight_b)
        return merged_output, merged_lse

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: Tensor,
        grad_lse: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        output_a, output_b, merged_output, weight_a, weight_b = ctx.saved_tensors
        merge_backward = (
            _compiled_merge_attention_backward
            if grad_output.is_cuda
            else _merge_attention_backward_impl
        )
        return merge_backward(
            grad_output,
            grad_lse,
            output_a,
            output_b,
            merged_output,
            weight_a,
            weight_b,
        )


def merge_attention_partials(
    output_a: Tensor, lse_a: Tensor, output_b: Tensor, lse_b: Tensor
) -> tuple[Tensor, Tensor]:
    """Stable FP32 online-softmax merge for two attention partials."""

    _require(
        output_a.dtype == output_b.dtype == torch.float32,
        "partial outputs must be FP32",
    )
    _require(lse_a.dtype == lse_b.dtype == torch.float32, "partial LSE must be FP32")
    return _AttentionPartialMerge.apply(output_a, lse_a, output_b, lse_b)


def merge_attention_partial_rows(
    output_a: Tensor,
    lse_a: Tensor,
    output_b: Tensor,
    lse_b: Tensor,
    row_indices: Tensor | None,
    row_slice: tuple[int, int] | None,
) -> tuple[Tensor, Tensor]:
    """Merge a row subset without materializing full-size scatter buffers when contiguous."""

    if row_indices is None:
        return merge_attention_partials(output_a, lse_a, output_b, lse_b)
    if row_slice is None:
        scattered_output, scattered_lse = scatter_upper_phase(
            output_b, lse_b, row_indices, output_a.size(0)
        )
        return merge_attention_partials(
            output_a, lse_a, scattered_output, scattered_lse
        )

    start, stop = row_slice
    _require(
        0 <= start <= stop <= output_a.size(0), "partial row slice is out of range"
    )
    _require(stop - start == output_b.size(0), "partial row slice has the wrong length")
    merged_rows, merged_lse_rows = merge_attention_partials(
        output_a[start:stop], lse_a[start:stop], output_b, lse_b
    )
    return (
        torch.cat((output_a[:start], merged_rows, output_a[stop:]), dim=0),
        torch.cat((lse_a[:start], merged_lse_rows, lse_a[stop:]), dim=0),
    )


def pack_phase_key_value(
    expanded: Tensor,
    k_rope: Tensor,
    qk_head_dim: int,
    v_head_dim: int,
) -> tuple[Tensor, Tensor]:
    """Pack contiguous backend K/V tensors from interleaved MLA projection output."""

    k_content = expanded[..., :qk_head_dim]
    value = expanded[..., qk_head_dim : qk_head_dim + v_head_dim]
    key = torch.cat(
        (k_content, k_rope.unsqueeze(1).expand(-1, expanded.size(1), -1)), dim=-1
    )
    return key, value.contiguous()


@torch.compile(fullgraph=True)
def _compiled_pack_phase_key_value(
    expanded: Tensor,
    k_rope: Tensor,
    qk_head_dim: int,
    v_head_dim: int,
) -> tuple[Tensor, Tensor]:
    """Fuse backend K/V packing and its autograd graph on CUDA."""

    return pack_phase_key_value(expanded, k_rope, qk_head_dim, v_head_dim)


def fused_pack_phase_key_value(
    expanded: Tensor,
    k_rope: Tensor,
    qk_head_dim: int,
    v_head_dim: int,
) -> tuple[Tensor, Tensor]:
    """Use compiled K/V packing on CUDA and the same eager formula elsewhere."""

    pack = _compiled_pack_phase_key_value if expanded.is_cuda else pack_phase_key_value
    return pack(expanded, k_rope, qk_head_dim, v_head_dim)


def _cudnn_backward_proxy_impl(
    partial_output: Tensor, grad_output: Tensor, grad_lse: Tensor
) -> tuple[Tensor, Tensor]:
    """Compute cuDNN's corrected output proxy with pure tensor operations."""

    partial_output = partial_output.float()
    grad_output = grad_output.float()
    grad_lse = grad_lse.float()
    norm2 = torch.sum(grad_output * grad_output, dim=-1)
    threshold = math.sqrt(torch.finfo(torch.float32).tiny)
    safe = torch.isfinite(norm2) & (norm2 >= threshold) & torch.isfinite(grad_lse)
    denominator = torch.where(safe, norm2, torch.ones_like(norm2))
    coefficient = torch.where(safe, grad_lse / denominator, torch.zeros_like(grad_lse))
    raw_correction = coefficient.unsqueeze(-1) * grad_output
    correction = torch.where(
        safe.unsqueeze(-1), raw_correction, torch.zeros_like(raw_correction)
    )
    return partial_output - correction, grad_output


@torch.compile(fullgraph=True)
def _compiled_cudnn_backward_proxy(
    partial_output: Tensor, grad_output: Tensor, grad_lse: Tensor
) -> tuple[Tensor, Tensor]:
    """Fuse the CUDA cuDNN backward correction into one compiled region."""

    return _cudnn_backward_proxy_impl(partial_output, grad_output, grad_lse)


def cudnn_backward_proxy(
    partial_output: Tensor, grad_output: Tensor, grad_lse: Tensor
) -> tuple[Tensor, Tensor]:
    """Construct cuDNN's corrected BF16 o/dO inputs in FP32.

    For safe rows, dot(G_i, O_corr) encodes the missing LSE gradient in public sdpa_backward.
    Zero and tiny rows use zero correction.
    """

    proxy = (
        _compiled_cudnn_backward_proxy
        if partial_output.is_cuda
        else _cudnn_backward_proxy_impl
    )
    return proxy(partial_output, grad_output, grad_lse)
