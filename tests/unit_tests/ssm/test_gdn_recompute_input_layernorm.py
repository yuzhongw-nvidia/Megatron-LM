# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""GDN/KDA support for the TransformerLayer input-layernorm recompute under FP8/FP4.

With "layernorm" in recompute_modules and FP8/FP4 enabled, TransformerLayer calls
``self_attention.set_for_recompute_input_layernorm()`` so that the Transformer Engine linears
reading the (discarded and later recomputed) layernorm output save their original input instead
of a quantized copy. These tests check which projections the GDN-family layers mark.
"""

import dataclasses

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _init_model_parallel(tp_size):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("transformer_engine.pytorch")
    pytest.importorskip("fla")
    if Utils.world_size < tp_size:
        pytest.skip(f"needs at least {tp_size} ranks")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp_size, pipeline_model_parallel_size=1
    )
    model_parallel_cuda_manual_seed(123)


@pytest.fixture
def model_parallel():
    _init_model_parallel(1)
    yield
    Utils.destroy_model_parallel()


@pytest.fixture
def model_parallel_tp2():
    _init_model_parallel(2)
    yield
    Utils.destroy_model_parallel()


def _build(variant, submodule_overrides=None, **overrides):
    kwargs = dict(
        hidden_size=256,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=4,
        linear_num_value_heads=16,
        num_layers=1,
        normalization="RMSNorm",
        use_cpu_initialization=True,
        num_attention_heads=4,
        activation_func=F.silu,
        bf16=True,
        params_dtype=torch.bfloat16,
        gradient_accumulation_fusion=False,
        experimental_attention_variant=variant,
        linear_attention_freq=[1],
        linear_cp_mode="headwise",
        transformer_impl="transformer_engine",
    )
    kwargs.update(overrides)
    if variant == "kda":
        kwargs["linear_num_key_heads"] = kwargs["linear_num_value_heads"]
    config = TransformerConfig(**kwargs)
    spec = get_experimental_attention_variant_module_spec(config=config)
    submodules = spec.submodules
    if submodule_overrides:
        submodules = dataclasses.replace(submodules, **submodule_overrides)
    groups = ProcessGroupCollection(
        tp=parallel_state.get_tensor_model_parallel_group(),
        cp=parallel_state.get_context_parallel_group(),
    )
    return spec.module(config, submodules=submodules, layer_number=1, pg_collection=groups)


def _saves_original_input(module):
    return getattr(module, "save_original_input", False) is True


def test_gdn_marks_a_plain_input_projection(model_parallel):
    # A GDN whose input layernorm lives outside the layer, as in TransformerLayer specs with a
    # separate input_layernorm: in_proj is a plain TE linear that reads the layernorm output.
    gdn = _build("gdn", submodule_overrides={"in_proj": TEColumnParallelLinear})
    assert not _saves_original_input(gdn.in_proj)

    gdn.set_for_recompute_input_layernorm()

    assert _saves_original_input(gdn.in_proj)
    assert not _saves_original_input(gdn.out_proj)


def test_gdn_leaves_a_fused_norm_linear_input_projection_alone(model_parallel):
    # The default GDN spec fuses the input layernorm into in_proj (TE LayerNormLinear), which
    # reads the residual stream and has no save_original_input switch; the hook must not raise.
    gdn = _build("gdn")
    assert not hasattr(gdn.in_proj, "save_original_input")

    gdn.set_for_recompute_input_layernorm()

    assert not hasattr(gdn.in_proj, "save_original_input")


@pytest.mark.parametrize(
    "f_lora_rank,gate_lora_rank", [(None, None), (32, None), (None, 32), (32, 32)]
)
def test_kda_marks_every_projection_reading_hidden_states(
    model_parallel, f_lora_rank, gate_lora_rank
):
    kda = _build("kda", kda_f_lora_rank=f_lora_rank, kda_gate_lora_rank=gate_lora_rank)

    kda.set_for_recompute_input_layernorm()

    expected = ["in_proj", "beta_proj"]
    if not kda.use_legacy_fused_projections:
        expected.append("f_proj" if f_lora_rank is None else "f_a_proj")
        expected.append("g_proj" if gate_lora_rank is None else "g_a_proj")
    for name in expected:
        assert _saves_original_input(getattr(kda, name)), name
    for name in ("out_proj", "f_b_proj", "g_b_proj"):
        module = getattr(kda, name, None)
        if module is not None:
            assert not _saves_original_input(module), name


def test_kda_leaves_beta_proj_alone_under_sequence_parallel(model_parallel_tp2):
    kda = _build("kda", tensor_model_parallel_size=2, sequence_parallel=True, kda_f_lora_rank=32)

    kda.set_for_recompute_input_layernorm()

    assert _saves_original_input(kda.in_proj)
    assert _saves_original_input(kda.f_a_proj)
    assert _saves_original_input(kda.g_proj)
    # beta_proj reads an all-gathered copy of the hidden states under sequence parallelism.
    assert not _saves_original_input(kda.beta_proj)
