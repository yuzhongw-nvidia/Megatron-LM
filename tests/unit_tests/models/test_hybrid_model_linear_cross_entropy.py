# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""HybridModel with the fused linear cross-entropy output layer
(``--cross-entropy-loss-fusion --cross-entropy-fusion-impl linear``)."""

import pytest
import torch

from megatron.core import tensor_parallel
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.linear_cross_entropy import LinearCrossEntropyModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.utils import get_device_arch_version
from tests.unit_tests.test_utilities import Utils

_SEQ_LEN = 32
_MICRO_BATCH = 2
_VOCAB_SIZE = 128
# Attention and MLP layers only: exercises the LM head without any Mamba kernel.
_PATTERN = "*-*-"


def _build_model(impl: str) -> HybridModel:
    """Build a small hybrid model; identical seeds give identical parameters per impl."""
    model_parallel_cuda_manual_seed(123)
    torch.manual_seed(123)
    config = TransformerConfig(
        num_layers=len(_PATTERN),
        hidden_size=256,
        num_attention_heads=8,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl=impl,
    )
    return HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=_VOCAB_SIZE,
        max_sequence_length=_SEQ_LEN,
        hybrid_layer_pattern=_PATTERN,
    ).cuda()


def _batch() -> dict:
    generator = torch.Generator(device="cuda").manual_seed(7)
    input_ids = torch.randint(
        0, _VOCAB_SIZE, (_MICRO_BATCH, _SEQ_LEN), device="cuda", generator=generator
    )
    labels = torch.randint(
        0, _VOCAB_SIZE, (_MICRO_BATCH, _SEQ_LEN), device="cuda", generator=generator
    )
    position_ids = torch.arange(_SEQ_LEN, device="cuda").unsqueeze(0).repeat(_MICRO_BATCH, 1)
    attention_mask = torch.ones(
        _MICRO_BATCH, 1, _SEQ_LEN, _SEQ_LEN, dtype=torch.bool, device="cuda"
    )
    return dict(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        labels=labels,
    )


class TestHybridModelLinearCrossEntropy:
    def setup_method(self, method):
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_output_layer_class_follows_the_fusion_impl(self):
        native = _build_model("native")
        assert not native.fuse_linear_cross_entropy
        assert type(native.output_layer) is tensor_parallel.ColumnParallelLinear

        fused = _build_model("linear")
        assert fused.fuse_linear_cross_entropy
        assert isinstance(fused.output_layer, LinearCrossEntropyModule)
        # Same parameter set and initialization as the plain output layer.
        for (name_n, p_n), (name_f, p_f) in zip(
            native.named_parameters(), fused.named_parameters()
        ):
            assert name_n == name_f
            assert torch.equal(p_n, p_f)

    @pytest.mark.skipif(
        get_device_arch_version() != 10,
        reason="fused linear cross entropy kernels require GPU architecture 10",
    )
    def test_fused_loss_and_grads_match_native(self):
        batch = _batch()
        native = _build_model("native")
        fused = _build_model("linear")

        loss_native = native(**batch)
        loss_fused = fused(**batch)
        # Both return the per-token loss as [b, s].
        assert loss_native.shape == (_MICRO_BATCH, _SEQ_LEN)
        assert loss_fused.shape == loss_native.shape
        # The native path rounds the logits to bf16 before the fp32 cross entropy; the fused kernel
        # keeps its logits chunks in fp32, so agreement is at the bf16 level, not bitwise.
        torch.testing.assert_close(loss_fused.float(), loss_native.float(), atol=2e-2, rtol=2e-2)

        loss_native.sum().backward()
        loss_fused.sum().backward()
        grad_native = native.output_layer.weight.grad
        grad_fused = fused.output_layer.weight.grad
        assert grad_fused is not None and torch.isfinite(grad_fused).all()
        torch.testing.assert_close(grad_fused.float(), grad_native.float(), atol=1e-2, rtol=1e-2)
        # The decoder receives gradients through the fused loss as well.
        first_decoder_param = next(p for n, p in fused.named_parameters() if "decoder" in n)
        assert first_decoder_param.grad is not None
        assert torch.isfinite(first_decoder_param.grad).all()

    def test_logits_are_still_returned_without_labels(self):
        fused = _build_model("linear")
        batch = _batch()
        batch.pop("labels")
        logits = fused(**batch, runtime_gather_output=True)
        assert logits.shape == (_MICRO_BATCH, _SEQ_LEN, _VOCAB_SIZE)
