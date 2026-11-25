# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Hybrid Muon dispatch ported from (Apache-2.0):
# https://github.com/fanshiqing/Megatron-LM/commit/6c77e623b582b024a3a6434ce62223b0a7b20fc0

import pytest
import torch

from megatron.core.optimizer import (
    OptimizerConfig,
    _bucket_emerging_param_groups,
    _get_param_groups,
)
from megatron.core.optimizer import emerging_optimizers as eo_mod
from megatron.core.optimizer.emerging_optimizers import HAVE_EMERGING_OPTIMIZERS, TensorParallelMuon
from megatron.core.optimizer.optimizer_config import muon_modes_are_hybrid
from tests.unit_tests.test_utilities import Utils

if HAVE_EMERGING_OPTIMIZERS:
    from megatron.core.optimizer.layer_sharded_muon import LayerShardedMuon
else:
    LayerShardedMuon = None
pytestmark = pytest.mark.skipif(not HAVE_EMERGING_OPTIMIZERS, reason="emerging_optimizers required")


class _DispatchModelCfg:
    num_attention_heads = 8
    num_query_groups = 2
    kv_channels = 16


class _DispatchChunk:
    config = _DispatchModelCfg()


class _HybridPGStub:
    """Duck-typed ProcessGroupCollection with a distinct sentinel per field, so the tests can
    assert WHICH domain's groups each bucket's optimizer received."""

    def __init__(self):
        self.gtp_remat = object()
        self.expt_gtp_remat = object()
        self.tp = object()
        self.expt_tp = object()


def _hybrid_cfg(tp_mode, expert_tp_mode=None, split_qkv=False) -> OptimizerConfig:
    """A valid muon config (``__post_init__`` enforces the layer_sharded requirements)."""
    return OptimizerConfig(
        optimizer='muon',
        use_layer_wise_distributed_optimizer=True,
        muon_tp_mode=tp_mode,
        muon_expert_tp_mode=expert_tp_mode,
        muon_split_qkv=split_qkv,
    )


def _hybrid_group(is_expert: bool, is_expert_parallel=None, optimizer=None):
    """One param group shaped like _get_param_groups output under hybrid modes: the semantic
    ``is_expert`` marker plus the communication flag, which coincide except at EP=1."""
    group = {
        'params': [torch.nn.Parameter(torch.randn(8, 8))],
        'is_expert_parallel': is_expert if is_expert_parallel is None else is_expert_parallel,
        'is_expert': is_expert,
    }
    if optimizer is not None:
        group['optimizer'] = optimizer
    return group


def _create_bucket(cfg, is_expert, pg_collection=None):
    """Build one bucket the way _get_megatron_emerging_optimizer does: the bucket flag is
    passed explicitly, never re-inferred from the groups."""
    optimizer, init_state_fn = eo_mod._create_emerging_optimizer(
        cfg,
        [_hybrid_group(is_expert)],
        'muon',
        [_DispatchChunk()],
        pg_collection,
        is_expert=is_expert,
    )
    assert init_state_fn is eo_mod._EMERGING_OPTIMIZERS['muon'].init_state_fn
    return optimizer


def test_hybrid_routes_buckets_to_different_classes():
    cfg = _hybrid_cfg('auto', 'layer_sharded')
    assert muon_modes_are_hybrid(cfg)

    dense_opt = _create_bucket(cfg, is_expert=False)
    assert type(dense_opt) is eo_mod.TensorParallelMuon
    assert dense_opt.tp_mode == 'auto', "dense bucket must honor muon_tp_mode"

    expert_opt = _create_bucket(cfg, is_expert=True)
    assert isinstance(expert_opt, LayerShardedMuon)
    # 'layer_sharded' is the registry selector; the class receives the bitwise reference
    # mode for its delegated (fallback/degenerate) paths.
    assert expert_opt.tp_mode == 'duplicated'


def test_hybrid_expert_bucket_gets_expert_domain_groups():
    pgs = _HybridPGStub()
    expert_opt = _create_bucket(
        _hybrid_cfg('auto', 'layer_sharded'), is_expert=True, pg_collection=pgs
    )
    assert expert_opt.gtp_remat_group is pgs.expt_gtp_remat
    assert expert_opt.tp_group is pgs.expt_tp

    # Reverse hybrid: the DENSE bucket's LayerShardedMuon keeps the dense domain.
    dense_opt = _create_bucket(
        _hybrid_cfg('layer_sharded', 'duplicated'), is_expert=False, pg_collection=pgs
    )
    assert isinstance(dense_opt, LayerShardedMuon)
    assert dense_opt.gtp_remat_group is pgs.gtp_remat
    assert dense_opt.tp_group is pgs.tp


def test_reverse_hybrid_expert_bucket_is_tensor_parallel_muon():
    expert_opt = _create_bucket(_hybrid_cfg('layer_sharded', 'duplicated'), is_expert=True)
    assert type(expert_opt) is eo_mod.TensorParallelMuon
    assert expert_opt.tp_mode == 'duplicated'


@pytest.mark.parametrize("tp_mode", ['layer_sharded', 'auto'])
def test_expert_mode_follows_dense_by_default(tp_mode):
    """An unset expert mode is NOT hybrid: both bucket flags resolve to the single mode's
    class, and the registry path (no flag) is what every group takes."""
    cfg = _hybrid_cfg(tp_mode)
    assert not muon_modes_are_hybrid(cfg)
    expected = LayerShardedMuon if tp_mode == 'layer_sharded' else eo_mod.TensorParallelMuon
    for is_expert in (False, True):
        optimizer = _create_bucket(cfg, is_expert)
        assert isinstance(optimizer, expected), f"tp_mode={tp_mode}, is_expert={is_expert}"


def test_hybrid_expert_bucket_tolerates_split_qkv():
    """Under hybrid modes split-QKV may stay on for the dense (TensorParallelMuon) bucket;
    the expert LayerShardedMuon bucket, which owns no QKV, gets it forced off instead of
    tripping the constructor reject."""
    cfg = _hybrid_cfg('auto', 'layer_sharded', split_qkv=True)
    cfg.muon_split_qkv_per_head = True
    dense_opt = _create_bucket(cfg, is_expert=False)
    assert dense_opt.split_qkv is True, "dense bucket must keep split-QKV"
    assert dense_opt.split_qkv_per_head is True
    expert_opt = _create_bucket(cfg, is_expert=True)
    assert isinstance(expert_opt, LayerShardedMuon)
    assert expert_opt.split_qkv is False


def test_explicit_expert_mode_equal_to_dense_is_not_hybrid():
    cfg = _hybrid_cfg('auto', 'auto')
    assert not muon_modes_are_hybrid(cfg)
    optimizer = _create_bucket(cfg, is_expert=True)
    assert type(optimizer) is eo_mod.TensorParallelMuon
    assert optimizer.tp_mode == 'auto'


def test_muon_builders_default_to_the_dense_bucket():
    """The bucket flag is the builders' single input. Without it (the registry's call) they
    describe the dense bucket, i.e. the non-hybrid behavior; with it, the expert bucket."""
    cfg = _hybrid_cfg('auto', 'layer_sharded')
    assert eo_mod._muon_bucket_tp_mode(cfg, is_expert=False) == 'auto'
    assert eo_mod._muon_bucket_tp_mode(cfg, is_expert=True) == 'layer_sharded'
    assert eo_mod._muon_config_to_cls(cfg) is eo_mod.TensorParallelMuon
    assert eo_mod._muon_config_to_cls(cfg, is_expert=True) is LayerShardedMuon

    dense_kwargs = eo_mod._muon_registry_config_to_kwargs(
        cfg, [_DispatchChunk()], pg_collection=None
    )
    assert dense_kwargs['tp_mode'] == 'auto'
    assert 'gtp_remat_group' not in dense_kwargs

    pgs = _HybridPGStub()
    expert_kwargs = eo_mod._muon_registry_config_to_kwargs(
        cfg, [_DispatchChunk()], pgs, is_expert=True
    )
    assert expert_kwargs['tp_mode'] == 'duplicated'  # LayerShardedMuon's delegated-path mode
    assert expert_kwargs['gtp_remat_group'] is pgs.expt_gtp_remat
    assert expert_kwargs['tp_group'] is pgs.expt_tp
    assert expert_kwargs['split_qkv'] is False

    # Reverse hybrid: the expert bucket is a TensorParallelMuon running the expert mode.
    rev_kwargs = eo_mod._muon_registry_config_to_kwargs(
        _hybrid_cfg('layer_sharded', 'distributed'), [_DispatchChunk()], pgs, is_expert=True
    )
    assert rev_kwargs['tp_mode'] == 'distributed'
    assert 'gtp_remat_group' not in rev_kwargs


def test_expert_bucket_is_muon_only():
    """Only the muon entry produces expert buckets; passing the flag for another entry is a
    programming error, not a silently ignored override."""
    with pytest.raises(AssertionError, match="muon entry"):
        eo_mod._create_emerging_optimizer(
            _hybrid_cfg('auto', 'layer_sharded'),
            [_hybrid_group(True)],
            'adaptive_muon',
            [_DispatchChunk()],
            None,
            is_expert=True,
        )


def test_bucketing_splits_on_the_two_expert_notions():
    """_bucket_emerging_param_groups: the communication flag splits only off the layer-wise
    path; the semantic flag splits the primary Muon optimizer only under hybrid modes; scalar
    groups never split on it. The EP=1 routed expert (allreduce=True, so not expert-parallel)
    is the corner case the semantic flag exists for."""
    dense = _hybrid_group(False)
    routed = _hybrid_group(True)  # EP>1: both flags set
    routed_ep1 = _hybrid_group(True, is_expert_parallel=False)  # EP=1, matching sizes
    scalar = _hybrid_group(False, optimizer='adam')
    groups = [dense, routed, routed_ep1, scalar]

    # Layer-wise + hybrid: the comm split is collapsed, the semantic split applies to muon.
    buckets = _bucket_emerging_param_groups(
        groups, 'muon', use_layer_wise=True, hybrid_muon_modes=True
    )
    assert set(buckets) == {('muon', False, False), ('muon', False, True), ('adam', False, False)}
    assert buckets[('muon', False, True)] == [routed, routed_ep1]
    assert buckets[('muon', False, False)] == [dense]

    # Layer-wise, not hybrid: one bucket per optimizer name, as before this flag existed.
    buckets = _bucket_emerging_param_groups(
        groups, 'muon', use_layer_wise=True, hybrid_muon_modes=False
    )
    assert set(buckets) == {('muon', False, False), ('adam', False, False)}

    # Plain path + hybrid: the comm flag splits too; the EP=1 expert shares the dense comm
    # groups (grad stats over mp) but still gets the expert mode.
    buckets = _bucket_emerging_param_groups(
        groups, 'muon', use_layer_wise=False, hybrid_muon_modes=True
    )
    assert set(buckets) == {
        ('muon', False, False),
        ('muon', True, True),
        ('muon', False, True),
        ('adam', False, False),
    }


class TestHybridMuonExpertIdentity:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        Utils.initialize_model_parallel()
        yield
        Utils.destroy_model_parallel()

    def test_get_param_groups_splits_routed_experts_at_ep1(self):
        """With EP=1 and matching dense/expert sizes, routed experts carry ``allreduce=True``
        like dense weights (``is_expert_parallel=False``). The semantic routed-expert marker
        still gives them their own groups under hybrid modes; the shared expert stays with
        the dense weights; without the predicate the grouping is untouched."""
        model = torch.nn.ModuleDict(
            {
                'dense': torch.nn.Linear(8, 8, bias=False),
                'experts': torch.nn.Linear(8, 8, bias=False),
                'shared_experts': torch.nn.Linear(8, 8, bias=False),
            }
        )
        model.experts.weight.expert_tp = True  # the tag _get_megatron_emerging_optimizer stamps
        cfg = _hybrid_cfg('auto', 'layer_sharded')

        groups = _get_param_groups(
            [model], cfg, {}, expert_param_fn=lambda p: getattr(p, 'expert_tp', False)
        )
        assert [(g['is_expert_parallel'], g['is_expert']) for g in groups] == [
            (False, False),
            (False, True),
        ]
        assert [id(p) for p in groups[0]['params']] == [
            id(model.dense.weight),
            id(model.shared_experts.weight),
        ]
        assert [id(p) for p in groups[1]['params']] == [id(model.experts.weight)]

        plain = _get_param_groups([model], cfg, {})
        assert len(plain) == 1
        assert 'is_expert' not in plain[0]


def test_hybrid_offload_prefetch_waits_for_each_child_step():
    from types import SimpleNamespace

    from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer

    calls = []

    class Child:
        def __init__(self, name):
            self.name = name

        def prefetch_optimizer_master_weights_for_step(self):
            calls.append((self.name, 'masters'))

        def prefetch_optimizer_state_for_step(self):
            calls.append((self.name, 'state'))

    optimizer = object.__new__(LayerWiseDistributedOptimizer)
    optimizer.config = SimpleNamespace(
        chunked_optimizer_state_offload=True, optimizer_state_offload_fraction=1.0
    )
    optimizer.chained_optimizers = [Child('dense'), Child('expert')]
    optimizer._managed_optimizer_state_offload_indices = (0, 1)
    optimizer.prefetch_optimizer_state_for_gradient_finalization()
    assert calls == [('dense', 'masters'), ('expert', 'masters'), ('dense', 'state')]
    calls.clear()
    optimizer._before_child_step(0)
    assert calls == [('dense', 'state')]
    calls.clear()
    optimizer._before_child_step(1)
    assert calls == [('expert', 'state')]
