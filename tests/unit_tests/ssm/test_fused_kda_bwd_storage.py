# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Static resource contract of the fused KDA backward kernel."""

import pytest

pytest.importorskip("cutlass")

from megatron.core.ssm.gated_delta_net.internal_gdn_backend.kernels.fused_kda_bwd_cute import (  # noqa: E402
    storage,
)


def test_shared_storage_fits_sm100_opt_in_limit():
    budget = storage.get_layout_budget()
    assert budget.smem_bytes <= 232_448
    assert budget.tmem_columns == 512
    assert budget.threads == 384


def test_tmem_ranges_are_phase_disjoint():
    storage._validate_tmem_ranges()
    ranges = {name: (begin, end) for name, begin, end, _, _ in storage.TMEM_RANGES}
    assert ranges["dh"] == (0, 128)
    assert ranges["gate"] == (448, 512)
    assert ranges["ring"] == (ranges["slot_a"][0], ranges["slot_b"][1])
    assert all(end <= storage.TMEM_COLUMNS for _, end in ranges.values())


def test_overlapping_live_ranges_are_rejected(monkeypatch):
    bad = storage.TMEM_RANGES + (("clobber", 0, 32, "wy", "wy"),)
    monkeypatch.setattr(storage, "TMEM_RANGES", bad)
    with pytest.raises(AssertionError, match="clobber"):
        storage._validate_tmem_ranges()
