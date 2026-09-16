# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Independent native-column references, including the Hybrid LM-head path."""

import json

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.tensor_parallel import layers, mappings
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.internal


@pytest.mark.parametrize("tp,sequence_parallel", [(1, False), (2, False), (2, True)])
@pytest.mark.parametrize("mode", ["trainable", "fused_wgrad", "frozen"])
def test_native_column_tp_dgrad_rounds_after_collective(tp, sequence_parallel, mode, monkeypatch):
    """Fixed cotangents expose partial rounding independently of softmax/optimizer."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=tp)
    try:
        rank = parallel_state.get_tensor_model_parallel_rank()
        group = parallel_state.get_tensor_model_parallel_group()
        generator = torch.Generator(device="cuda").manual_seed(423)
        # The actual Hybrid training recipe has a singleton microbatch dimension.
        full_input = torch.randn(64, 1, 128, device="cuda", generator=generator).bfloat16()
        full_weight = (torch.randn(4096, 128, device="cuda", generator=generator) * 0.1).bfloat16()
        full_probe = (torch.randn(64, 1, 4096, device="cuda", generator=generator) * 0.1).bfloat16()
        local_weight = full_weight.chunk(tp, dim=0)[rank].contiguous()
        local_probe = full_probe.chunk(tp, dim=-1)[rank].contiguous()
        reference_output = torch.nn.functional.linear(full_input.double(), local_weight.double())
        reference_dgrad = full_probe.double().matmul(full_weight.double()).bfloat16()
        reference_wgrad = (
            local_probe.flatten(0, 1).double().t().matmul(full_input.flatten(0, 1).double())
        )
        if sequence_parallel:
            local_input = full_input.chunk(tp, dim=0)[rank].contiguous()
            reference_dgrad = reference_dgrad.chunk(tp, dim=0)[rank]
        else:
            local_input = full_input

        collectives = []
        original_all_reduce = torch.distributed.all_reduce
        original_reduce_scatter = layers.dist_reduce_scatter_func

        def all_reduce(tensor, *args, **kwargs):
            collectives.append(("all_reduce", tensor.dtype))
            return original_all_reduce(tensor, *args, **kwargs)

        def reduce_scatter(output, tensor, *args, **kwargs):
            collectives.append(("reduce_scatter", output.dtype, tensor.dtype))
            return original_reduce_scatter(output, tensor, *args, **kwargs)

        monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
        monkeypatch.setattr(layers, "dist_reduce_scatter_func", reduce_scatter)
        monkeypatch.setattr(mappings, "dist_reduce_scatter_func", reduce_scatter)
        results = []
        for enabled in (False, True):
            config = TransformerConfig(
                num_layers=1,
                hidden_size=128,
                num_attention_heads=4,
                tensor_model_parallel_size=tp,
                sequence_parallel=sequence_parallel,
                bf16=True,
                params_dtype=torch.bfloat16,
                perform_initialization=False,
                gradient_accumulation_fusion=mode == "fused_wgrad",
                tp_reduce_in_fp32=enabled,
            )
            model = layers.ColumnParallelLinear(
                128,
                4096,
                config=config,
                init_method=lambda tensor: tensor,
                bias=False,
                gather_output=False,
                tp_group=group,
            )
            with torch.no_grad():
                model.weight.copy_(local_weight)
            model.weight.requires_grad_(mode != "frozen")
            if mode == "fused_wgrad":
                model.weight.main_grad = torch.zeros_like(model.weight, dtype=torch.float32)
                model.weight.grad_added_to_main_grad = False
            x = local_input.detach().clone().requires_grad_(True)
            output, bias = model(x)
            assert bias is None and output.dtype == torch.bfloat16
            torch.testing.assert_close(output.double(), reference_output, atol=8e-3, rtol=8e-3)
            collectives.clear()
            output.backward(local_probe)
            assert x.grad.dtype == torch.bfloat16
            expected_dtype = torch.float32 if enabled and tp > 1 else torch.bfloat16
            if tp > 1:
                expected = (
                    ("reduce_scatter", expected_dtype, expected_dtype)
                    if sequence_parallel
                    else ("all_reduce", expected_dtype)
                )
                assert collectives == [expected]
            else:
                assert not collectives
            error = (x.grad.float() - reference_dgrad.float()).norm()
            grad_weight = None
            if mode != "frozen":
                grad_weight = (
                    (model.weight.main_grad if mode == "fused_wgrad" else model.weight.grad)
                    .detach()
                    .clone()
                )
                tolerance = 1e-5 if mode == "fused_wgrad" else 8e-3
                torch.testing.assert_close(
                    grad_weight.double(), reference_wgrad, atol=tolerance, rtol=tolerance
                )
            results.append((output.detach().clone(), x.grad.detach().clone(), grad_weight, error))

        baseline, candidate = results
        torch.testing.assert_close(candidate[0], baseline[0], rtol=0, atol=0)
        if mode != "frozen":
            torch.testing.assert_close(candidate[2], baseline[2], rtol=0, atol=0)
        if tp == 1:
            # TP1 also uses bounded FP32 accumulation. Validate its precision
            # against the oracle, not the disabled path's GEMM ordering.
            assert candidate[3] / reference_dgrad.float().norm() < 4e-4
        else:
            assert candidate[3] < baseline[3] * 0.1, (candidate[3], baseline[3])
        print(
            "NATIVE_COLUMN_FP32_DGRAD "
            + json.dumps(
                dict(
                    tp=tp,
                    sequence_parallel=sequence_parallel,
                    mode=mode,
                    rank=rank,
                    baseline_error=float(baseline[3]),
                    fp32_reduce_error=float(candidate[3]),
                    reference="independent FP64 matmul rounded once to BF16",
                )
            ),
            flush=True,
        )
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("tp,sequence_parallel", [(1, False), (2, False), (2, True)])
@pytest.mark.parametrize("mode", ["trainable", "fused_wgrad", "frozen"])
def test_native_column_large_vocabulary_dgrad(tp, sequence_parallel, mode):
    """Long softmax-like contractions must stay accurate at TP1 and TP2.

    A few dense probability cotangents with a dominant negative target term
    reproduce the long-accumulation error of a vocabulary projection. Zero
    padding retains a realistic GEMM row count without an expensive FP64
    reference for every row. Both SP ranks receive nonzero reference rows.
    """
    Utils.initialize_model_parallel(tensor_model_parallel_size=tp)
    try:
        rank = parallel_state.get_tensor_model_parallel_rank()
        group = parallel_state.get_tensor_model_parallel_group()
        generator = torch.Generator(device="cuda").manual_seed(1923)
        length, vocab, hidden = 2048, 163840, 128
        full_weight = (
            torch.randn(vocab, hidden, device="cuda", generator=generator) * 0.02
        ).bfloat16()
        full_input = (
            torch.randn(length, 1, hidden, device="cuda", generator=generator) * 0.1
        ).bfloat16()
        active = torch.cat(
            (torch.arange(16, device="cuda"), torch.arange(1024, 1040, device="cuda"))
        )
        logits = torch.randn(32, vocab, device="cuda", generator=generator)
        probe = logits.softmax(dim=-1)
        targets = torch.randint(vocab, (32,), device="cuda", generator=generator)
        probe[torch.arange(32, device="cuda"), targets] -= 1
        probe = probe.bfloat16()
        del logits
        reference = torch.zeros(length, 1, hidden, device="cuda", dtype=torch.bfloat16)
        reference[active, 0] = (probe.double() @ full_weight.double()).bfloat16()
        full_probe = torch.zeros(length, 1, vocab, device="cuda", dtype=torch.bfloat16)
        full_probe[active, 0] = probe
        local_weight = full_weight.chunk(tp, dim=0)[rank].contiguous()
        local_probe = full_probe.chunk(tp, dim=-1)[rank].contiguous()
        local_input = full_input
        if sequence_parallel:
            local_input = full_input.chunk(tp, dim=0)[rank].contiguous()
            reference = reference.chunk(tp, dim=0)[rank].contiguous()
        results = []
        for enabled in (False, True):
            config = TransformerConfig(
                num_layers=1,
                hidden_size=hidden,
                num_attention_heads=4,
                tensor_model_parallel_size=tp,
                sequence_parallel=sequence_parallel,
                bf16=True,
                params_dtype=torch.bfloat16,
                perform_initialization=False,
                gradient_accumulation_fusion=mode == "fused_wgrad",
                tp_reduce_in_fp32=enabled,
            )
            model = layers.ColumnParallelLinear(
                hidden,
                vocab,
                config=config,
                init_method=lambda tensor: tensor,
                bias=False,
                gather_output=False,
                tp_group=group,
            )
            with torch.no_grad():
                model.weight.copy_(local_weight)
            model.weight.requires_grad_(mode != "frozen")
            if mode == "fused_wgrad":
                model.weight.main_grad = torch.zeros_like(model.weight, dtype=torch.float32)
                model.weight.grad_added_to_main_grad = False
            x = local_input.detach().clone().requires_grad_(True)
            output, bias = model(x)
            assert bias is None and output.dtype == torch.bfloat16
            output.backward(local_probe)
            assert x.grad.dtype == torch.bfloat16 and torch.isfinite(x.grad).all()
            relative_error = (
                (x.grad.double() - reference.double()).norm() / reference.double().norm()
            ).item()
            wgrad = None
            if mode != "frozen":
                wgrad = (
                    (model.weight.main_grad if mode == "fused_wgrad" else model.weight.grad)
                    .detach()
                    .clone()
                )
            results.append((output.detach(), x.grad.detach().clone(), wgrad, relative_error))
        baseline, candidate = results
        torch.testing.assert_close(candidate[0], baseline[0], atol=0, rtol=0)
        if mode != "frozen":
            torch.testing.assert_close(candidate[2], baseline[2], atol=0, rtol=0)
        # Independent FP64 reference, rounded once at the BF16 activation boundary.
        # The disabled path records the regression magnitude without relying on
        # a particular vendor's GEMM algorithm being inaccurate on every GPU.
        assert candidate[3] < 4e-4, (tp, sequence_parallel, mode, candidate[3])
        print(
            "NATIVE_COLUMN_LARGE_VOCAB "
            + json.dumps(
                dict(
                    rank=rank,
                    tp=tp,
                    sequence_parallel=sequence_parallel,
                    mode=mode,
                    baseline_relative_l2=baseline[3],
                    candidate_relative_l2=candidate[3],
                    reference="independent FP64 softmax-like contraction rounded once to BF16",
                )
            ),
            flush=True,
        )
    finally:
        Utils.destroy_model_parallel()


def test_native_column_fp32_dgrad_partial_final_tile():
    """A vocabulary that does not fill the final tile retains all its terms."""
    generator = torch.Generator(device="cuda").manual_seed(184)
    weight = torch.randn(4128, 128, device="cuda", generator=generator).bfloat16()
    probe = torch.randn(32, 1, 4128, device="cuda", generator=generator).bfloat16()
    expected = probe.double() @ weight.double()
    actual = layers._linear_dgrad_in_fp32(probe, weight)
    assert actual.dtype == torch.float32 and actual.shape == expected.shape
    torch.testing.assert_close(actual.double(), expected, atol=5e-4, rtol=2e-5)
