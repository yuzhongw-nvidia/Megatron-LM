# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

from megatron.training.datasets.data_samplers import _needs_identity_collate


def _args(**overrides):
    defaults = {
        "dynamic_context_parallel": False,
        "sequence_packing_scheduler": None,
        "use_vanilla_collate_fn": False,
        "use_varlen_dataset": False,
        "varlen_bshd_validation": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_varlen_bshd_validation_uses_default_collate():
    assert (
        _needs_identity_collate(
            _args(use_varlen_dataset=True, varlen_bshd_validation=True)
        )
        is False
    )


def test_varlen_thd_uses_identity_collate():
    assert _needs_identity_collate(_args(use_varlen_dataset=True)) is True


def test_sequence_packing_scheduler_uses_identity_collate():
    assert _needs_identity_collate(_args(sequence_packing_scheduler="dp_balanced")) is True


def test_dynamic_context_parallel_uses_identity_collate():
    assert _needs_identity_collate(_args(dynamic_context_parallel=True)) is True
