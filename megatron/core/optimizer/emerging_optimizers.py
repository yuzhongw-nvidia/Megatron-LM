# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Hybrid Muon dispatch ported from (Apache-2.0):
# https://github.com/fanshiqing/Megatron-LM/commit/6c77e623b582b024a3a6434ce62223b0a7b20fc0

"""Emerging optimizer registry.

To add a new emerging optimizer:
  1. Define its optimizer class (or import it).
  2. Write its ``_<name>_init_state_fn`` and ``_<name>_config_to_kwargs``.
  3. Add an ``EmergingOptimizerEntry`` to ``_EMERGING_OPTIMIZERS`` at the bottom.
"""

import hashlib
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, NamedTuple, Optional, get_args

import torch
from torch.optim.optimizer import ParamsT

from megatron.core.process_groups_config import ProcessGroupCollection, resolve_gtp_remat_group
from megatron.core.utils import (
    get_emerging_optimizers_version,
    get_pg_rank,
    get_pg_size,
    is_emerging_optimizers_min_version,
    log_single_rank,
)

from .optimizer_config import ParamKey, ParamPredicate, muon_effective_expert_tp_mode

try:
    from emerging_optimizers import registry
    from emerging_optimizers import utils as eo_utils
    from emerging_optimizers.orthogonalized_optimizers import (
        AdaptiveMuon,
        OrthogonalizedOptimizer,
        get_muon_scale_factor,
    )
    from emerging_optimizers.orthogonalized_optimizers.muon_utils import (
        NSCoeffT,
        newton_schulz,
        newton_schulz_tp,
    )

    # It is necessary to import optimizers for the registry to work.
    from emerging_optimizers.scalar_optimizers import Lion  # pylint: disable=unused-import

    try:
        from emerging_optimizers.soap import SOAP  # pylint: disable=unused-import
    except ModuleNotFoundError as error:
        # Emerging-Optimizers >= 0.5.0a0 moved SOAP to ``legacy_soap`` (same registry
        # name). Without this fallback the import error disables every optimizer of
        # the package, including Muon.
        if error.name != 'emerging_optimizers.soap':
            raise
        from emerging_optimizers.legacy_soap import SOAP  # pylint: disable=unused-import

    HAVE_EMERGING_OPTIMIZERS = True
except ImportError:
    HAVE_EMERGING_OPTIMIZERS = False
    OrthogonalizedOptimizer = object
    AdaptiveMuon = object


logger = logging.getLogger(__name__)

# newton_schulz_tp() gained the use_syrk kwarg in emerging_optimizers 0.4.0. Earlier releases
# expose use_syrk on the non-TP newton_schulz() only, so 0.3.x still rejects it here. Spelled
# ".dev0" so pre-release builds of that line are accepted too, matching how the TE minimums
# elsewhere in the tree are written.
_SYRK_MIN_EO_VERSION = "0.4.0.dev0"


def get_supported_coefficient_types() -> tuple[str, ...]:
    """Return the coefficient types supported by the installed emerging_optimizers.

    Reads the members of the ``NSCoeffT`` Literal type so that new types
    added upstream are automatically available without code changes here.
    """
    assert (
        HAVE_EMERGING_OPTIMIZERS
    ), "emerging_optimizers >= 0.2 is required for NSCoeffT. Please install or upgrade it."
    return get_args(NSCoeffT)


def validate_coefficient_type(coefficient_type: str) -> None:
    """Raise ``ValueError`` if *coefficient_type* is not supported."""
    supported = get_supported_coefficient_types()
    if coefficient_type not in supported:
        raise ValueError(
            f"Unsupported muon coefficient type '{coefficient_type}'. "
            f"Supported types: {supported}"
        )


# ===========================================================================
# Registry dataclass and public API
# ===========================================================================


def _eopt_init_state_fn(opt, config=None):
    """Initialize emerging optimizer state for torch_dist checkpoint format."""
    for group in opt.param_groups:
        # Checkpoint init needs state for all parameters, including those without grads yet.
        opt._init_group(group, skip_non_grad_params=False)


def _default_param_overrides_factory() -> Dict[ParamKey, Dict[str, Any]]:
    """Default param overrides: route non-linear/embedding params to Adam."""
    return {
        ParamKey(
            predicate=ParamPredicate(name="nonlinear_or_embedding", fn=_is_nonlinear_or_embedding)
        ): {'optimizer': 'adam'}
    }


@dataclass
class EmergingOptimizerEntry:
    """Everything needed to create and configure an emerging optimizer.

    Attributes:
        optimizer_cls: The torch optimizer class (default when config_to_cls is None).
        init_state_fn: Lazily initialises optimizer state (needed for checkpoint formats).
        config_to_kwargs: ``(config, model_chunks, pg_collection) -> dict`` of constructor kwargs.
        config_to_cls: Optional ``(config) -> type`` callable. When provided, overrides
            ``optimizer_cls`` so the instantiated class can vary based on config (e.g.
            selecting ``LayerShardedMuon`` when ``muon_tp_mode == 'layer_sharded'``).
        default_param_overrides: Per-parameter config overrides applied automatically
            (e.g. route non-linear params to Adam).
    """

    optimizer_cls: type
    init_state_fn: Callable = _eopt_init_state_fn
    config_to_kwargs: Callable | None = None
    config_to_cls: Callable | None = None
    default_param_overrides: Dict[ParamKey, Dict[str, Any]] = field(
        default_factory=_default_param_overrides_factory
    )


def _create_emerging_optimizer(
    config, param_groups, eopt_name, model_chunks, pg_collection, *, is_expert: bool = False
):
    """Instantiate an emerging optimizer and return it with its init_state_fn.

    ``is_expert`` marks the expert bucket of a hybrid dense/expert Muon configuration (see
    ``_get_megatron_emerging_optimizer``). Only the ``muon`` entry produces such buckets, and
    its builders take the flag so the bucket gets the class, mode and process-group axes of
    the expert weight family. Every other bucket goes through the entry's registered
    builders unchanged.
    """
    entry = _EMERGING_OPTIMIZERS[eopt_name]
    if is_expert:
        assert (
            eopt_name == 'muon'
        ), f"expert buckets exist only for the muon entry, got {eopt_name!r}"
        eopt_kwargs = _muon_registry_config_to_kwargs(
            config, model_chunks, pg_collection, is_expert=True
        )
        optimizer_cls = _muon_config_to_cls(config, is_expert=True)
    else:
        if entry.config_to_kwargs is not None:
            eopt_kwargs = entry.config_to_kwargs(config, model_chunks, pg_collection)
        else:
            eopt_kwargs = _default_adam_based_eopt_config_to_kwargs(
                eopt_name, config, model_chunks, pg_collection
            )
        optimizer_cls = (
            entry.config_to_cls(config) if entry.config_to_cls is not None else entry.optimizer_cls
        )
    optimizer = optimizer_cls(param_groups, **eopt_kwargs)
    return optimizer, entry.init_state_fn


# ===========================================================================
# Shared helpers
# ===========================================================================


def _is_nonlinear_or_embedding(param):
    """True for parameters that should NOT use the emerging optimizer."""
    return getattr(param, 'is_embedding_or_output_parameter', False) or len(param.shape) != 2


def _is_muon_excluded(param):
    """True for parameters that should use the scalar optimizer instead of Muon."""
    return not getattr(param, 'use_muon', True) or _is_nonlinear_or_embedding(param)


def _get_qkv_split_shapes(model_cfg, split_qkv_per_head: bool = False) -> list[int]:
    """Compute fused QKV split shapes from logical attention layout metadata."""
    if hasattr(model_cfg, 'projection_split_shapes'):
        if split_qkv_per_head:
            return list(model_cfg.per_head_split_shapes) * model_cfg.num_groups
        return list(model_cfg.projection_split_shapes)

    query_projection_size = (
        model_cfg.num_attention_heads // model_cfg.num_query_groups * model_cfg.kv_channels
    )
    if split_qkv_per_head:
        num_query_heads_per_group = model_cfg.num_attention_heads // model_cfg.num_query_groups
        per_group_shapes = [model_cfg.kv_channels] * num_query_heads_per_group
        if getattr(model_cfg, 'attention_output_gate', False):
            per_group_shapes += [model_cfg.kv_channels] * num_query_heads_per_group
        per_group_shapes += [model_cfg.kv_channels, model_cfg.kv_channels]
        return per_group_shapes * model_cfg.num_query_groups
    if getattr(model_cfg, 'attention_output_gate', False):
        return [
            query_projection_size,
            query_projection_size,
            model_cfg.kv_channels,
            model_cfg.kv_channels,
        ]
    return [query_projection_size, model_cfg.kv_channels, model_cfg.kv_channels]


def _localize_qkv_split_shapes(
    global_split_shapes: list[int], local_start: int, local_rows: int
) -> tuple[list[int], bool]:
    """Intersect global per-head split sizes with a rank-local contiguous row range.

    Returns:
        The physical rank-local split sizes and whether every intersected head is complete.
    """
    local_stop = local_start + local_rows
    local_split_shapes = []
    all_heads_complete = True
    head_start = 0
    for head_rows in global_split_shapes:
        head_stop = head_start + head_rows
        overlap_start = max(head_start, local_start)
        overlap_stop = min(head_stop, local_stop)
        if overlap_start < overlap_stop:
            overlap_rows = overlap_stop - overlap_start
            local_split_shapes.append(overlap_rows)
            all_heads_complete &= overlap_rows == head_rows
        head_start = head_stop

    if sum(local_split_shapes) != local_rows:
        raise RuntimeError(
            f"Muon per-head QKV local range [{local_start}, {local_stop}) is outside "
            f"the global split shape with {sum(global_split_shapes)} rows"
        )
    return local_split_shapes, all_heads_complete


def _qkv_split_groups_are_complete(
    split_shapes: list[int], local_start: int, local_rows: int
) -> bool:
    """Return whether a local row range contains only complete fused QKV groups."""
    split_width = sum(split_shapes)
    if split_width <= 0:
        raise ValueError(f"Muon QKV split shapes must sum to a positive size: {split_shapes}")
    return local_start % split_width == 0 and local_rows % split_width == 0


# ===========================================================================
# Registry – populated below only when emerging_optimizers is installed.
# ===========================================================================

_EMERGING_OPTIMIZERS: Dict[str, EmergingOptimizerEntry] = {}


# ===========================================================================
# Muon
# ===========================================================================


# tp_mode="auto" selects tp_mode per weight (Dense/GTP weights only)
_AUTO_TP_MODES = ("duplicated", "distributed")


@dataclass(frozen=True)
class HardwareProfile:
    """HW spec for different GPU. Use for cost model when selecting TP mode.
    Bandwidths are UNIDIRECTIONAL."""

    bf16_peak_tflops: float  # dense, fp32_matmul_prec = "medium" for now
    bw_intra_gbps: float  # collectives staying inside one NVLink domain
    bw_inter_gbps: float  # collectives crossing domains, over the fabric


_PROFILES = {
    # Keys are matched as a substring of the reported device name ("NVIDIA GB200").
    # Both bandwidths are PER GPU. Only run on GB200 & GB300 for now.
    # TODO: May need to add other HW Spec
    "GB200": HardwareProfile(bf16_peak_tflops=2500.0, bw_intra_gbps=900.0, bw_inter_gbps=100.0),
    "GB300": HardwareProfile(bf16_peak_tflops=2500.0, bw_intra_gbps=900.0, bw_inter_gbps=100.0),
}


def _hardware_profile() -> Optional[HardwareProfile]:
    """Profile for the local GPU, or None when the hardware is not in the registry."""
    try:
        name = torch.cuda.get_device_properties(0).name
    except Exception:  # noqa: BLE001 - no CUDA device: fall back, do not fail
        return None
    return next((prof for key, prof in _PROFILES.items() if key in name), None)


def _select_tp_mode(
    m: int,
    n: int,
    group_size: int,
    steps: int,
    use_syrk: bool,
    elem_size: int,
    communication_crosses_domain: bool,
    profile: Optional[HardwareProfile] = None,
    candidates: tuple[str, ...] = _AUTO_TP_MODES,
) -> str:
    """Cost model for per weight tp_mode selection. Mirrors the op sequence in
    scaled_orthogonalize_fn_with_gtp_remat -- keep in sync.
    """
    min_dim, max_dim = min(m, n), max(m, n)
    m_partitioned = (
        m // group_size
    )  # dist orthogonalizes the [n, m/group_size] shard; transpose is forced
    gram = 1 if use_syrk else 2  # SYRK halves the two gram ops
    # Per NS step: gram X@X.T + gram A@A + GEMM B@X.
    flops = {
        "duplicated": steps
        * (gram * (min_dim * min_dim * max_dim + min_dim**3) + 2 * min_dim * min_dim * max_dim),
        "distributed": steps * (gram * (n * n * m_partitioned + n**3) + 2 * n * n * m_partitioned),
    }
    if profile is None:
        return "duplicated" if communication_crosses_domain else min(candidates, key=flops.get)

    ring_fraction = (
        group_size - 1
    ) / group_size  # ring: each rank moves (group_size-1)/group_size of the buffer
    num_bytes = {
        "duplicated": m * n * elem_size * ring_fraction,  # one all-gather
        "distributed": steps * 2 * n * n * elem_size * ring_fraction,  # gram all-reduce per step
    }
    bw = (profile.bw_inter_gbps if communication_crosses_domain else profile.bw_intra_gbps) * 1e9
    peak = profile.bf16_peak_tflops * 1e12
    cost = {mode: flops[mode] / peak + num_bytes[mode] / bw for mode in candidates}
    return min(candidates, key=cost.get)


def offload_chunk_partition_key(param: torch.Tensor) -> bool:
    """Chunk partition of ``param`` for chunked optimizer-state offload: expert-TP or not.

    Expert-tensor-parallel parameters are gathered by the batched duplicated-GTP Muon step
    over the expert GTP remat group, whose ranks are different data-parallel ranks and so own
    different dense parameters (the LayerWise optimizer shards dense parameters across data
    parallelism). Keeping expert parameters in chunks of their own makes every rank of that
    group pack them identically, so the batches, and the all-gathers they issue, match.
    The LayerWise optimizer passes this key when it enables the offload of a Muon child.
    """
    return bool(getattr(param, "expert_tp", False))


class TensorParallelMuon(OrthogonalizedOptimizer):
    """Tensor Parallel Muon optimizer."""

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.01,
        use_decoupled_weight_decay: bool = True,
        split_qkv: bool = False,
        split_qkv_per_head: bool = False,
        is_qkv_fn: Callable[[torch.Tensor], bool] | None = None,
        qkv_split_shapes: list[int] | None = None,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        pg_collection: Optional[ProcessGroupCollection] = None,
        tp_mode: Literal["blockwise", "duplicated", "distributed", "auto"] = "duplicated",
        use_syrk: bool = False,
        ns_batch_size: int = 1,
        ns_batch_max_mb: int = 2048,
        ns_gather_ahead: int = 0,
    ) -> None:
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}")
        if use_syrk and not is_emerging_optimizers_min_version(_SYRK_MIN_EO_VERSION):
            raise ValueError(
                f"use_syrk requires emerging_optimizers >= {_SYRK_MIN_EO_VERSION}, but "
                f"{get_emerging_optimizers_version()} is installed. Upgrade "
                "emerging_optimizers or drop --muon-use-syrk."
            )
        if split_qkv_per_head and not split_qkv:
            raise ValueError("split_qkv_per_head requires split_qkv=True")
        if ns_batch_size > 1:
            # Same floor as LayerShardedMuon: batched (3-D) newton_schulz and, with SYRK, the
            # batched SYRK kernel. Imported lazily; layer_sharded_muon imports this module.
            from megatron.core.optimizer.layer_sharded_muon import _validate_ns_config

            _validate_ns_config(use_syrk, ns_batch_size)
        elif ns_batch_size < 1:
            raise ValueError(f"ns_batch_size must be at least 1, got {ns_batch_size}")
        if ns_batch_max_mb < 1:
            raise ValueError(f"ns_batch_max_mb must be at least 1, got {ns_batch_max_mb}")
        if ns_gather_ahead < 0:
            raise ValueError(f"ns_gather_ahead must be non-negative, got {ns_gather_ahead}")

        def scaled_orthogonalize_fn(
            grad: torch.Tensor,
            tp_group: torch.distributed.ProcessGroup,
            partition_dim: int | None = None,
            tp_mode_this_group: str = tp_mode,
            scale_shape: tuple[int, int] | None = None,
        ) -> torch.Tensor:
            log_single_rank(
                logger,
                logging.DEBUG,
                f'Orthogonalizing grad with {num_ns_steps} steps, '
                f'{coefficient_type} coefficient, '
                f'{scale_mode} scale mode, extra_scale_factor={extra_scale_factor}',
            )
            if scale_shape is None:
                size = [grad.size(-2), grad.size(-1)]
                if partition_dim is not None:
                    size[partition_dim] *= get_pg_size(tp_group)
            else:
                # This overrides only the final Muon scalar; NS still uses grad's physical
                # shape and partition metadata for its collectives.
                size = scale_shape
            if partition_dim is not None and get_pg_size(tp_group) == 1:
                # A one-rank TP group holds the whole matrix already. newton_schulz_tp would
                # still all-gather it (one NCCL kernel plus a concat copy per parameter) in
                # "duplicated" mode; the single-rank Newton-Schulz is the same computation.
                partition_dim = None
            # Only forward the kwarg when enabled; older emerging_optimizers do not
            # accept it at all, and __init__ has already rejected use_syrk on those.
            ns_kwargs = {"use_syrk": True} if use_syrk else {}
            orth_grad = newton_schulz_tp(
                grad,
                steps=num_ns_steps,
                coefficient_type=coefficient_type,
                tp_group=tp_group,
                partition_dim=partition_dim,
                tp_mode="duplicated" if tp_mode_this_group == "blockwise" else tp_mode_this_group,
                **ns_kwargs,
            )
            scale_factor = get_muon_scale_factor(size[0], size[1], mode=scale_mode)
            # newton_schulz_tp returns a fresh tensor, so scale it in place. Two separate
            # multiplies keep the rounding of the former ``orth_grad * a * b`` expression; a
            # multiply by exactly 1.0 is an identity and is skipped.
            orth_grad.mul_(scale_factor)
            if extra_scale_factor != 1.0:
                orth_grad.mul_(extra_scale_factor)
            return orth_grad

        self.pg_collection = pg_collection
        self.tp_mode = tp_mode
        self.split_qkv = split_qkv
        self.split_qkv_per_head = split_qkv_per_head
        self.is_qkv_fn = is_qkv_fn
        self.qkv_split_shapes = qkv_split_shapes
        # For the tp_mode="auto" cost model (_resolve_tp_mode / _select_tp_mode).
        self.num_ns_steps = num_ns_steps
        self.use_syrk = use_syrk
        self.elem_size = 2 if fp32_matmul_prec == "medium" else 4  # bf16 vs tf32/fp32
        # For the batched duplicated-GTP step (see step / _step_gtp_batches).
        self.coefficient_type = coefficient_type
        self.scale_mode = scale_mode
        self.extra_scale_factor = extra_scale_factor
        self.fp32_matmul_prec = fp32_matmul_prec
        self.ns_batch_size = ns_batch_size
        self.ns_batch_max_bytes = ns_batch_max_mb << 20
        # Batched duplicated-GTP gathers issued this many batches ahead of the Newton-Schulz
        # that consumes them, on a side stream (0 = in order on the compute stream).
        self.ns_gather_ahead = ns_gather_ahead
        self._gather_stream: Optional[torch.cuda.Stream] = None
        # Batched duplicated-GTP steps taken so far: (batches, matrices). Tests and probes
        # read it to confirm the batched path ran at all.
        self.gtp_batch_counts = (0, 0)
        # Batch-plan hashes exchanged with the GTP remat peers but not yet verified
        # (see _exchange_gtp_batch_plans / verify_gtp_batch_plans).
        self._pending_gtp_plan_checks: List[tuple] = []
        # (GTP remat group, plan hash) -> exchanges issued so far for that plan.
        self._gtp_plan_exchanges: Dict[tuple, int] = {}
        self._tp_mode_cache: Dict[tuple, str] = {}
        self._hw_profile = _hardware_profile() if tp_mode == "auto" else None
        self._warned_qkv_split_disabled = False
        self._warned_distributed_qkv_fallback = False

        weight_decay_method = "decoupled" if use_decoupled_weight_decay else "l2"
        # Use explicit class call instead of super() so that subclasses with
        # multiple inheritance (e.g. TensorParallelAdaptiveMuon) don't route
        # through an intermediate class that doesn't accept scaled_orthogonalize_fn.
        OrthogonalizedOptimizer.__init__(
            self,
            params,
            lr,
            momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            weight_decay_method=weight_decay_method,
            fp32_matmul_prec=fp32_matmul_prec,
            scaled_orthogonalize_fn=scaled_orthogonalize_fn,
        )

    @staticmethod
    def _all_gather_tensor(t, group, dim):
        """All-gather equal-size shards of ``t`` over ``group`` and concat along ``dim``.

        The shards land in one ``[group_size, *t.shape]`` buffer. Along dim 0 that buffer *is*
        the concatenation, so the gathered matrix needs no extra copy; other dims still
        concatenate once.
        """
        group_size = get_pg_size(group)
        try:
            backend = torch.distributed.get_backend(group)
        except Exception:  # not a real process group (tests fake the collective)
            backend = None
        if backend != torch.distributed.Backend.NCCL:
            # all_gather_into_tensor is NCCL-only; gloo (CPU tests) keeps the list form.
            shards = [torch.empty_like(t) for _ in range(group_size)]
            torch.distributed.all_gather(shards, t.contiguous(), group)
            return torch.cat(shards, dim=dim)
        gathered = torch.empty((group_size,) + tuple(t.shape), dtype=t.dtype, device=t.device)
        torch.distributed.all_gather_into_tensor(gathered, t.contiguous(), group)
        if dim == 0:
            return gathered.view(group_size * t.size(0), *t.shape[1:])
        return torch.cat(gathered.unbind(0), dim=dim)

    @staticmethod
    def _strip_pad(t, pad_length):
        """Drop the trailing ``pad_length`` rows of dim 0 (no-op if ``pad_length == 0``)."""
        return t[:-pad_length] if pad_length else t

    @staticmethod
    def _restore_pad(t, pad_length):
        """Re-append ``pad_length`` zero rows to dim 0 (no-op if ``pad_length == 0``)."""
        return torch.nn.functional.pad(t, (0, 0, 0, pad_length)) if pad_length else t

    # ------------------------------------------------------------------------------------------
    # Batched duplicated-GTP step.
    #
    # In duplicated mode every GTP-sharded 2-D weight costs one all-gather plus one whole-matrix
    # Newton-Schulz, serialized on the compute stream by the per-parameter loop of
    # OrthogonalizedOptimizer.step. A MoE model has thousands of identically shaped expert
    # weights per rank, so that loop is bound by kernel launches (tens of thousands of small
    # GEMMs) and by one latency-bound all-gather per weight. With ``ns_batch_size > 1`` the
    # step below groups same-shape eligible weights, all-gathers each group with ONE
    # collective, and runs one batched (3-D) Newton-Schulz per group. With ``ns_gather_ahead``
    # the gathers run on a side stream that many batches ahead of their Newton-Schulz, so
    # they overlap the GEMMs of the batches before them; by default they run in order on the
    # compute stream. (An earlier "side-stream gathers run 4-12x longer under the GEMMs"
    # measurement was taken while the offload chunks split the expert weights differently on
    # each rank, so the peers' gathers were mismatched and each one waited for the peer's next
    # collective; it says nothing about the overlap itself.)
    # Weights that need anything else (TP sharding, split-QKV,
    # blockwise/distributed GTP, no GTP) keep the per-parameter path unchanged.
    # ------------------------------------------------------------------------------------------

    class _GtpBatchKey(NamedTuple):
        """Weights that share a key can be stacked into one batched Newton-Schulz."""

        gtp_remat_group: Any
        rows: int  # local (sharded) rows
        cols: int
        pad_length: int
        dtype: torch.dtype

    def _gtp_batch_key(self, p: torch.Tensor) -> Optional["TensorParallelMuon._GtpBatchKey"]:
        """The batch key of ``p``, or None when ``p`` must take the per-parameter path.

        Eligible: a 2-D GTP-sharded weight whose resolved mode is ``duplicated`` and whose
        gathered matrix carries no further TP sharding, so the whole-matrix Newton-Schulz
        needs no collective of its own and only the GTP all-gather and the reshard remain.
        The GTP group and mode come from ``_resolve_gtp_remat_mode``, the resolver
        ``scaled_orthogonalize_fn_with_gtp_remat`` uses for the per-parameter path.
        """
        if p.dim() != 2 or self.pg_collection is None:
            return None
        if self.split_qkv and self.is_qkv_fn(p):  # type: ignore[misc]
            return None
        gtp_remat_group, gtp_active, _, mode = self._resolve_gtp_remat_mode(p)
        if not gtp_active or mode != "duplicated":
            return None
        is_expert = getattr(p, 'expert_tp', False)
        tp_group = self.pg_collection.expt_tp if is_expert else self.pg_collection.tp
        partition_dim = getattr(p, "partition_dim", None)
        if partition_dim is not None and partition_dim != -1:
            if tp_group is not None and get_pg_size(tp_group) > 1:
                return None  # TP-sharded after the GTP gather: newton_schulz_tp owns it
        return self._GtpBatchKey(
            gtp_remat_group, p.shape[0], p.shape[1], getattr(p, 'pad_length', 0), p.dtype
        )

    def _momentum_input(self, p: torch.Tensor, group: dict) -> torch.Tensor:
        """Weight decay + momentum update of ``p``; returns the tensor Newton-Schulz consumes.

        Same statements, same order as OrthogonalizedOptimizer.step.
        """
        grad = p.grad
        state = self.state[p]
        self._apply_weight_decay_inplace(p, grad, group["lr"], group["weight_decay"])
        state["momentum_buffer"].lerp_(grad, 1 - group["momentum"])
        if self.nesterov:
            return grad.lerp(state["momentum_buffer"], group["momentum"])
        return state["momentum_buffer"]

    def _step_param(self, p: torch.Tensor, group: dict) -> None:
        """The per-parameter body of OrthogonalizedOptimizer.step for one weight."""
        grad = self._momentum_input(p, group)
        with eo_utils.fp32_matmul_precision(self.fp32_matmul_prec):
            group_kwargs = {k: v for k, v in group.items() if k != "params"}
            orth_grad = self.orthogonalize(p, grad, **group_kwargs)
        self.pre_weight_update_fn_inplace(p, orth_grad)
        p.add_(orth_grad, alpha=-group["lr"])
        self.post_weight_update_fn_inplace(p)

    @torch.no_grad()  # type: ignore[misc]
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """One optimizer step; batches same-shape duplicated-GTP weights when enabled."""
        if self.ns_batch_size <= 1:
            return OrthogonalizedOptimizer.step(self, closure)
        if closure is not None:
            raise ValueError("closure is not supported")

        plain: List[tuple[torch.Tensor, dict]] = []
        buckets: Dict[TensorParallelMuon._GtpBatchKey, List[tuple[torch.Tensor, dict]]] = {}
        for group in self.param_groups:
            self._init_group(group)
            for p in group["params"]:
                if p.grad is None:
                    continue  # pragma: no cover
                key = self._gtp_batch_key(p)
                if key is None:
                    plain.append((p, group))
                else:
                    buckets.setdefault(key, []).append((p, group))

        # The per-parameter weights first: their work overlaps the first batch's gathers.
        for p, group in plain:
            self._step_param(p, group)

        batches: List[tuple[TensorParallelMuon._GtpBatchKey, List[tuple[torch.Tensor, dict]]]] = []
        for key, items in buckets.items():
            gathered_bytes = (
                get_pg_size(key.gtp_remat_group) * key.rows * key.cols * key.dtype.itemsize
            )
            batch_size = max(1, min(self.ns_batch_size, self.ns_batch_max_bytes // gathered_bytes))
            for start in range(0, len(items), batch_size):
                batches.append((key, items[start : start + batch_size]))
        self.verify_gtp_batch_plans(wait=False)
        self._exchange_gtp_batch_plans(batches)
        self._step_gtp_batches(batches)
        return None

    # ------------------------------------------------------------------------------------------
    # Batch-plan guard.
    #
    # Every batch is one collective over the GTP remat group, so the peers must form the same
    # batches in the same order. Nothing in NCCL enforces that: with mismatched sizes a ring
    # all-gather neither errors nor hangs, it streams the next collective's bytes into the
    # previous one, so each gather finishes one batch late and the gathered momentum is wrong
    # without any symptom but a slow optimizer step. (The chunked optimizer-state offload hit
    # exactly this: it stepped the optimizer per chunk and packed a different number of
    # expert weights into a chunk on each data-parallel rank until it partitioned the chunks
    # by expert_tp.) The guard below all-gathers an 8-byte hash of each call's batch plan per
    # group, issued ahead of the batches on the same stream, and checks the hashes once the
    # gather has completed, so a mismatch raises within a few calls instead of corrupting the
    # update silently. Checks are deferred so the host keeps running ahead of the GPU.
    #
    # The plans are static (the same weights in the same chunks every step), so each distinct
    # plan is exchanged _GTP_PLAN_EXCHANGES_PER_PLAN times per group and then no more: the
    # steady-state step pays no collective for the guard. Skipping is decided from the local
    # plan alone, which is safe because peers that agree on the plans skip the same calls,
    # and peers that do not were already reported by the first exchange of the plan they
    # disagree on.
    # ------------------------------------------------------------------------------------------

    _GTP_PLAN_EXCHANGES_PER_PLAN = 2

    @staticmethod
    def _gtp_batch_plan_hash(plan: List[tuple]) -> int:
        """A stable 63-bit hash of one group's ordered batch plan."""
        digest = hashlib.blake2b(repr(plan).encode(), digest_size=8).digest()
        return int.from_bytes(digest, "little", signed=True)

    def _all_gather_plan_hash(self, gathered: torch.Tensor, local: torch.Tensor, group) -> None:
        """All-gather the ``[1]`` int64 plan hash into ``gathered`` (``[group_size]``)."""
        self._all_gather_into(gathered, local, group)

    def _exchange_gtp_batch_plans(self, batches) -> None:
        """Queue one small all-gather per GTP remat group carrying this call's plan hash."""
        plans: Dict[Any, List[tuple]] = {}
        for key, items in batches:
            plans.setdefault(key.gtp_remat_group, []).append(
                (key.rows, key.cols, key.pad_length, str(key.dtype), len(items))
            )
        if not plans:
            return
        device = batches[0][1][0][0].device
        for group, plan in plans.items():
            value = self._gtp_batch_plan_hash(plan)
            exchanged = self._gtp_plan_exchanges.get((group, value), 0)
            if exchanged >= self._GTP_PLAN_EXCHANGES_PER_PLAN:
                continue  # confirmed against the peers; see the guard's note above
            self._gtp_plan_exchanges[(group, value)] = exchanged + 1
            size = get_pg_size(group)
            if device.type == "cuda":
                # Stage through pinned host memory with non-blocking copies: a pageable
                # ``torch.tensor(..., device=cuda)`` and a ``.tolist()`` each synchronize the
                # current stream, and this runs once per step() call (per offload chunk), so
                # either would stall the host behind the GPU at every chunk boundary.
                local_host = torch.empty(1, dtype=torch.int64, pin_memory=True)
                local_host[0] = value
                local = local_host.to(device, non_blocking=True)
                gathered = torch.empty(size, dtype=torch.int64, device=device)
                self._all_gather_plan_hash(gathered, local, group)
                peers = torch.empty(size, dtype=torch.int64, pin_memory=True)
                peers.copy_(gathered, non_blocking=True)
                event = torch.cuda.Event()
                event.record()
                # ``local`` / ``gathered`` stay referenced until the event has completed so the
                # caching allocators cannot hand their blocks to another stream's work.
                self._pending_gtp_plan_checks.append(
                    (value, peers, event, plan, (local_host, local, gathered))
                )
            else:
                local = torch.tensor([value], dtype=torch.int64, device=device)
                gathered = torch.empty(size, dtype=torch.int64, device=device)
                self._all_gather_plan_hash(gathered, local, group)
                self._pending_gtp_plan_checks.append((value, gathered, None, plan, ()))

    def verify_gtp_batch_plans(self, wait: bool = True) -> None:
        """Check the exchanged batch-plan hashes against the local ones.

        With ``wait=False`` only the exchanges whose gather has completed are checked (the
        rest stay pending); ``wait=True`` synchronizes and checks them all. Raises
        ``RuntimeError`` on the first mismatch: the ranks of a GTP remat group formed
        different batches, so their batched gathers would not line up.
        """
        remaining = []
        for value, gathered, event, plan, keepalive in self._pending_gtp_plan_checks:
            if event is not None:
                if wait:
                    event.synchronize()
                elif not event.query():
                    remaining.append((value, gathered, event, plan, keepalive))
                    continue
            peers = gathered.tolist()
            if any(v != value for v in peers):
                bad = [i for i, v in enumerate(peers) if v != value]
                raise RuntimeError(
                    "Batched Muon step: the GTP remat peers did not form the same batches. "
                    f"Local plan of {len(plan)} batches (rows, cols, pad, dtype, size), first: "
                    f"{plan[:3]}; group ranks with another plan: {bad}. Every rank of a GTP "
                    "remat group must see the same eligible weights in the same order in each "
                    "step() call (with chunked optimizer-state offload, every chunk)."
                )
        self._pending_gtp_plan_checks = remaining

    def _all_gather_into(self, out: torch.Tensor, t: torch.Tensor, group) -> None:
        """All-gather ``t`` over ``group`` along dim 0 into the preallocated ``out``."""
        try:
            backend = torch.distributed.get_backend(group)
        except Exception:  # not a real process group (tests fake the collective)
            backend = None
        if backend != torch.distributed.Backend.NCCL:
            # all_gather_into_tensor is NCCL-only; gloo (CPU tests) keeps the list form. The
            # row chunks of a contiguous ``out`` are contiguous, so no copy follows.
            torch.distributed.all_gather(
                list(out.chunk(get_pg_size(group), dim=0)), t.contiguous(), group
            )
            return
        torch.distributed.all_gather_into_tensor(out, t.contiguous(), group)

    def _gather_gtp_batch(self, key, items, compute_stream=None):
        """Momentum update of a batch, then ONE all-gather of the stacked shards.

        The batch's momentum shards are stacked into a single ``[batch, rows, cols]`` tensor
        and all-gathered with one collective, so a batch of 8 weights costs one NCCL kernel
        instead of 8 latency-bound ones. Returns ``(stacked, gathered, done)``: the stacked
        input, the rank-major ``[gtp_remat_size * batch, rows, cols]`` result that
        ``_apply_gtp_batch`` reorders, and the event the consumer must wait on (None when the
        gather ran in order on ``compute_stream``/CPU).

        With ``compute_stream`` given the gather is issued on the side stream
        ``self._gather_stream`` after an event recorded on the compute stream (so it sees the
        momentum update), which lets it overlap the Newton-Schulz of the batches before it.
        Every rank issues the same gathers in the same order on the same communicator, so the
        stream placement cannot reorder collectives across ranks.
        """
        stacked = torch.stack([self._momentum_input(p, group) for p, group in items])
        gathered = torch.empty(
            (get_pg_size(key.gtp_remat_group) * len(items), key.rows, key.cols),
            dtype=key.dtype,
            device=stacked.device,
        )
        if compute_stream is None:
            self._all_gather_into(gathered, stacked, key.gtp_remat_group)
            return stacked, gathered, None
        if self._gather_stream is None:
            # High priority so the NCCL kernel's CTAs are scheduled at the next GEMM wave
            # boundary instead of queueing behind the whole Newton-Schulz of the previous
            # batch; it does not preempt resident blocks.
            self._gather_stream = torch.cuda.Stream(priority=-1)
        ready = torch.cuda.Event()
        ready.record(compute_stream)
        with torch.cuda.stream(self._gather_stream):
            self._gather_stream.wait_event(ready)
            self._all_gather_into(gathered, stacked, key.gtp_remat_group)
            done = torch.cuda.Event()
            done.record(self._gather_stream)
        return stacked, gathered, done

    def _apply_gtp_batch(self, key, items, gathered: torch.Tensor) -> None:
        """Batched whole-matrix Newton-Schulz on ``gathered``; reshard and apply the updates.

        Same operations as the duplicated branch of ``scaled_orthogonalize_fn_with_gtp_remat``
        plus ``scaled_orthogonalize_fn`` with ``partition_dim=None``, applied to a
        ``[batch, rows, cols]`` stack instead of one matrix at a time. ``gathered`` is the
        rank-major ``[gtp_remat_size * batch, rows, cols]`` buffer of ``_gather_gtp_batch``;
        the per-weight whole matrices are ``[batch, gtp_remat_size * rows, cols]``.
        """
        gtp_remat_size = get_pg_size(key.gtp_remat_group)
        gathered = (
            gathered.view(gtp_remat_size, len(items), key.rows, key.cols)
            .transpose(0, 1)
            .reshape(len(items), gtp_remat_size * key.rows, key.cols)
        )
        x = gathered[:, : -key.pad_length] if key.pad_length else gathered
        ns_kwargs = {"use_syrk": True} if self.use_syrk else {}
        with eo_utils.fp32_matmul_precision(self.fp32_matmul_prec):
            orth = newton_schulz(
                x, steps=self.num_ns_steps, coefficient_type=self.coefficient_type, **ns_kwargs
            )
        orth.mul_(get_muon_scale_factor(orth.size(-2), orth.size(-1), mode=self.scale_mode))
        if self.extra_scale_factor != 1.0:
            orth.mul_(self.extra_scale_factor)
        if key.pad_length:
            orth = torch.nn.functional.pad(orth, (0, 0, 0, key.pad_length))
        start = get_pg_rank(key.gtp_remat_group) * key.rows
        for (p, group), full in zip(items, orth.unbind(0)):
            update = full[start : start + key.rows].contiguous()
            self.pre_weight_update_fn_inplace(p, update)
            p.add_(update, alpha=-group["lr"])
            self.post_weight_update_fn_inplace(p)

    def _step_gtp_batches(self, batches) -> None:
        """Run the batches: one gather, one batched Newton-Schulz, the updates, per batch.

        With ``ns_gather_ahead > 0`` (and CUDA) the gather of batch ``i + ns_gather_ahead`` is
        issued on the side stream before the Newton-Schulz of batch ``i`` runs, so up to that
        many gathers are in flight under the GEMMs; the buffers of a batch stay referenced
        until its Newton-Schulz consumed them. ``ns_gather_ahead == 0`` is the in-order path:
        every gather on the compute stream right before its Newton-Schulz.
        """
        if not batches:
            return
        ahead = self.ns_gather_ahead
        compute_stream = (
            torch.cuda.current_stream() if ahead > 0 and batches[0][1][0][0].is_cuda else None
        )
        if compute_stream is None:
            ahead = 0
        # Gathers issued but not yet consumed, oldest first.
        inflight: List[tuple] = []
        for i in range(min(ahead, len(batches))):
            inflight.append(self._gather_gtp_batch(*batches[i], compute_stream))
        for i, (key, items) in enumerate(batches):
            if ahead == 0:
                stacked, gathered, done = self._gather_gtp_batch(key, items)
            else:
                stacked, gathered, done = inflight.pop(0)
                if i + ahead < len(batches):
                    inflight.append(self._gather_gtp_batch(*batches[i + ahead], compute_stream))
                if done is not None:
                    compute_stream.wait_event(done)
            self._apply_gtp_batch(key, items, gathered)
            self.gtp_batch_counts = (
                self.gtp_batch_counts[0] + 1,
                self.gtp_batch_counts[1] + len(items),
            )
            # ``stacked`` was read and ``gathered`` written by the side stream; both were
            # allocated on the compute stream, which has now waited for that work, so freeing
            # them here cannot hand their memory to an earlier side-stream user.
            del stacked, gathered

    def _resolve_tp_mode(self, m: int, n: int, group_size: int) -> str:
        """Cached per-shape mode for tp_mode="auto", dense (GTP) weights only.

        communication_crosses_domain=False always: this is only called for dense weights
        (see scaled_orthogonalize_fn_with_gtp_remat), and GTP stays inside one NVLink domain.
        """
        key = (m, n, group_size)
        if key not in self._tp_mode_cache:
            self._tp_mode_cache[key] = _select_tp_mode(
                m,
                n,
                group_size,
                self.num_ns_steps,
                self.use_syrk,
                self.elem_size,
                communication_crosses_domain=False,
                profile=self._hw_profile,
            )
            log_single_rank(
                logger,
                logging.INFO,
                f"muon tp_mode=auto (dense): ({m}, {n}) group_size={group_size} -> "
                f"{self._tp_mode_cache[key]}",
            )
        return self._tp_mode_cache[key]

    def _resolve_gtp_remat_mode(self, p: torch.Tensor, qkv_split: bool = False) -> tuple:
        """How Muon handles ``p`` along GTP_remat: ``(group, active, size, mode)``.

        ``group`` is the GTP_remat group of ``p``'s weight family (None without process
        groups), ``active`` whether ``p`` is actually sharded over it, ``size`` the group
        size when active and 1 otherwise, and ``mode`` the resolved NS mode: ``self.tp_mode``,
        with ``auto`` resolved per weight. ``qkv_split`` pins ``auto`` to duplicated, the only
        mode that can honor a split. The per-parameter path and the batched duplicated-GTP
        step both resolve through here, so they cannot disagree on a weight's mode.
        """
        # TODO: Clean up code that determines if parameter is a MoE layer and which TP group to use
        is_expert = getattr(p, 'expert_tp', False)
        gtp_remat_group = (
            resolve_gtp_remat_group(self.pg_collection, is_expert) if self.pg_collection else None
        )

        # Parameters with is_gtp_weight_remat=False are not sharded along the
        # GTP process group, and do not require all-gathering prior to
        # orthogonalization.
        gtp_active = (
            gtp_remat_group is not None
            and get_pg_size(gtp_remat_group) > 1
            and getattr(p, 'is_gtp_weight_remat', False)
        )
        gtp_remat_size = get_pg_size(gtp_remat_group) if gtp_active else 1

        mode = self.tp_mode
        if mode == "auto":
            if qkv_split:
                # Only duplicated can honor a split; the cost model below doesn't know
                # that, so pin it here instead of letting the shape decide.
                mode = "duplicated"
            else:
                # Scoped to dense (GTP) weights for now; expert weights keep today's default.
                mode = (
                    self._resolve_tp_mode(p.shape[0] * gtp_remat_size, p.shape[1], gtp_remat_size)
                    if gtp_active and not is_expert
                    else "duplicated"
                )
        return gtp_remat_group, gtp_active, gtp_remat_size, mode

    def scaled_orthogonalize_fn_with_gtp_remat(
        self, p, grad, tp_group, partition_dim, qkv_split_shapes=None
    ):
        """Orthogonalize a (possibly GTP-sharded) momentum, then reshard.

        When GTP is inactive this is a plain passthrough to ``scaled_orthogonalize_fn``.
        Otherwise, ``mode`` (``self.tp_mode``, or resolved per-weight when
        ``self.tp_mode == "auto"``) controls how GTP sharding is handled:

        - **blockwise**: orthogonalize the local GTP shard independently, no collective.
        - **duplicated**: all-gather over GTP, run whole-matrix NS (TP-aware), reshard.
        - **distributed**: distribute NS over GTP via small-Gram all-reduce. When both
          GTP and TP are active, NS is distributed over the larger group to minimize
          redundant compute; the smaller group is all-gathered beforehand.

        GTP_remat may pad dim 0 for alignment (see gtp_remat_shard_dim0). blockwise and
        duplicated strip the padding before calling scaled_orthogonalize_fn and restore it
        after, since every rank holds a uniform, fully-reconstructed tensor by then.
        distributed keeps the padding needed by its collective, but applies Muon's scale
        factor using the unpadded logical matrix shape.

        ``qkv_split_shapes`` (when set) runs Newton-Schulz on q, k and v separately. The
        split needs the whole matrix, so under GTP it is available on the duplicated path
        only -- blockwise and distributed fall back to whole-matrix NS and warn once.
        """
        if qkv_split_shapes is not None and not qkv_split_shapes:
            raise ValueError(
                "qkv_split_shapes must be None (no split) or non-empty; got an empty sequence"
            )

        gtp_remat_group, gtp_active, gtp_remat_size, mode = self._resolve_gtp_remat_mode(
            p, qkv_split=qkv_split_shapes is not None
        )

        def scaled_orthogonalize_fn_with_qkv_split(grad):
            """Orthogonalize `grad`, splitting [q|k|v] first when `qkv_split_shapes` is not None.

            `grad` is always GTP-unsharded -- GTP is inactive, or the caller has already
            gathered. A GTP row shard would cut q/k/v mid-boundary.
            """
            if qkv_split_shapes is None:
                return self.scaled_orthogonalize_fn(
                    grad, tp_group, partition_dim, tp_mode_this_group=mode
                )
            qkv_rows = sum(qkv_split_shapes)
            if grad.size(0) % qkv_rows != 0:
                raise RuntimeError(
                    f"Muon QKV split shape mismatch: grad_shape={tuple(grad.shape)}, "
                    f"split_shapes={qkv_split_shapes}"
                )
            num_query_groups = grad.size(0) // qkv_rows
            cols = grad.size(-1)
            qkv_grads = torch.split(
                grad.view(num_query_groups, qkv_rows, cols), qkv_split_shapes, dim=1
            )
            qkv_grads = [
                self.scaled_orthogonalize_fn(
                    g.reshape(-1, cols), tp_group, partition_dim, tp_mode_this_group=mode
                ).view(num_query_groups, -1, cols)
                for g in qkv_grads
            ]
            return torch.cat(qkv_grads, dim=1).view(grad.shape)

        if not gtp_active:
            return scaled_orthogonalize_fn_with_qkv_split(grad)

        gtp_rank = get_pg_rank(gtp_remat_group)
        pad_length = getattr(p, 'pad_length', 0)

        if qkv_split_shapes is not None and mode != "duplicated":
            # Fall back, not raise: these configs ran before the qkv fix, so an abort
            # here would break live jobs.
            if not self._warned_qkv_split_disabled:
                self._warned_qkv_split_disabled = True
                debug_name = getattr(p, '_debug_name', '')
                log_single_rank(
                    logger,
                    logging.WARNING,
                    f"Muon: split-QKV is disabled for GTP-sharded qkv weights under "
                    f"tp_mode='{mode}'"
                    + (f" (auto-resolved from '{self.tp_mode}')" if self.tp_mode == "auto" else "")
                    + (f", first seen on {debug_name}" if debug_name else "")
                    + "; blockwise/distributed keep the rows sharded, so the q/k/v "
                    "boundaries are not available on any rank. These weights get "
                    "whole-matrix Newton-Schulz, which is NOT the update rule the same "
                    "model gets at TP1. Pass --muon-tp-mode duplicated for a "
                    "layout-invariant qkv update, or --muon-no-split-qkv to make the "
                    "whole-matrix rule explicit at every layout.",
                )
            qkv_split_shapes = None

        if mode == "blockwise":
            # Local block NS on this rank's GTP row-shard (shape [M/gtp_remat_size, K]):
            # partition_dim=None makes scaled_orthogonalize_fn run a plain Newton-Schulz on
            # the shard with no GTP/TP collective. pad_length can exceed one shard's row
            # count, so only the overlap between this rank's shard and the trailing padded
            # rows of the full tensor is this rank's own padding.
            shard_size = grad.size(0)
            ranks_from_end = gtp_remat_size - 1 - gtp_rank
            local_pad_length = min(shard_size, max(0, pad_length - ranks_from_end * shard_size))
            if local_pad_length == shard_size:
                # Entirely padding: grad is exact zero, and NS(0) = 0.
                return torch.zeros_like(grad)
            result = self.scaled_orthogonalize_fn(
                self._strip_pad(grad, local_pad_length), tp_group, None, tp_mode_this_group=mode
            )
            return self._restore_pad(result, local_pad_length)

        if mode == "duplicated":
            # All-gather over GTP (dim 0), strip/restore padding exactly (every rank now
            # holds the same padded tensor), orthogonalize the whole matrix
            # (scaled_orthogonalize_fn handles any TP sharding per tp_mode), reshard dim 0.
            gathered_grad = self._all_gather_tensor(grad, gtp_remat_group, 0)
            result = scaled_orthogonalize_fn_with_qkv_split(
                self._strip_pad(gathered_grad, pad_length)
            )
            result = self._restore_pad(result, pad_length)
            reshard_size = result.size(0) // gtp_remat_size
            return result[gtp_rank * reshard_size : (gtp_rank + 1) * reshard_size].contiguous()

        # distributed: NS via the small-Gram all-reduce (no redundant full-matrix NS).
        # QKV splitting has already fallen back to this unsplit fused layout above, so
        # pad_length describes grad's full dim-0 layout here. Keep the physical rows for
        # the collective, but exclude them from Muon's shape-based scale.
        scale_shape = None
        if pad_length:
            if grad.shape != p.shape:
                raise RuntimeError(
                    "Distributed GTP Muon padding requires the momentum and parameter "
                    f"layouts to match, got grad={tuple(grad.shape)} and p={tuple(p.shape)}"
                )
            size = [grad.size(-2) * gtp_remat_size - pad_length, grad.size(-1)]
            if partition_dim is not None:
                size[partition_dim] *= get_pg_size(tp_group)
            scale_shape = (size[0], size[1])

        # A momentum with both TP and GTP as sharding axes takes two communication steps: an
        # all-gather that eliminates one axis, then the Gram all-reduce that distributes NS over
        # the other. With GTP as the only sharding axis, the Gram all-reduce is the only
        # communication needed. partition_dim is what says whether TP is a sharding axis here,
        # the same signal scaled_orthogonalize_fn and newton_schulz_tp key off.
        needs_two_step_communication = (
            partition_dim is not None and tp_group is not None and get_pg_size(tp_group) > 1
        )

        if not needs_two_step_communication:
            # GTP is the only sharding axis: distribute NS over it on the local dim-0 row shard.
            return self.scaled_orthogonalize_fn(
                grad,
                gtp_remat_group,
                partition_dim=0,
                tp_mode_this_group=mode,
                scale_shape=scale_shape,
            )

        # GTP + TP: distributed NS can only operate over one (group, dim) at a
        # time. Distribute over the larger group so that the NS GEMMs are sharded
        # across more ranks (less redundant compute), and all-gather the smaller
        # group to eliminate its sharding beforehand.
        tp_size = get_pg_size(tp_group)
        if gtp_remat_size >= tp_size:
            smaller_group, smaller_dim = tp_group, partition_dim
            larger_group, larger_dim = gtp_remat_group, 0
        else:
            smaller_group, smaller_dim = gtp_remat_group, 0
            larger_group, larger_dim = tp_group, partition_dim

        gathered_grad = self._all_gather_tensor(grad, smaller_group, smaller_dim)
        orthogonalized_grad = self.scaled_orthogonalize_fn(
            gathered_grad,
            larger_group,
            larger_dim,
            tp_mode_this_group=mode,
            scale_shape=scale_shape,
        )
        shard_size = orthogonalized_grad.size(smaller_dim) // get_pg_size(smaller_group)
        reshard_rank = get_pg_rank(smaller_group)
        return orthogonalized_grad.narrow(
            smaller_dim, reshard_rank * shard_size, shard_size
        ).contiguous()

    def _warn_distributed_qkv_fallback(self):
        """Warn once when a QKV layout cannot use distributed Newton-Schulz."""
        if self.tp_mode != "distributed" or self._warned_distributed_qkv_fallback:
            return
        log_single_rank(
            logger,
            logging.WARNING,
            "muon_tp_mode='distributed' is not supported for per-head or fragmented "
            "QKV splitting; falling back to non-TP Newton-Schulz.",
        )
        self._warned_distributed_qkv_fallback = True

    def _gather_qkv_grad(self, p, grad, tp_group, expected_rows):
        """Reconstruct a fused QKV gradient and record how to restore its local shard."""
        gathered_grad = grad
        tp_slice = None
        if gathered_grad.shape[0] != expected_rows:
            partition_dim = getattr(p, "partition_dim", None)
            if partition_dim != 0 or tp_group is None:
                raise RuntimeError(
                    f"Muon QKV split shape mismatch: grad_shape={tuple(gathered_grad.shape)}, "
                    f"expected_rows={expected_rows}, partition_dim={partition_dim}"
                )
            tp_size = get_pg_size(tp_group)
            if gathered_grad.shape[0] * tp_size != expected_rows:
                raise RuntimeError(
                    "Muon QKV split cannot reconstruct the global tensor: "
                    f"local_grad_shape={tuple(gathered_grad.shape)}, tp_size={tp_size}, "
                    f"expected_rows={expected_rows}"
                )
            tp_rank = get_pg_rank(tp_group)
            tp_local_rows = gathered_grad.shape[0]
            shards = [torch.empty_like(gathered_grad) for _ in range(tp_size)]
            torch.distributed.all_gather(shards, gathered_grad, tp_group)
            gathered_grad = torch.cat(shards, dim=0)
            tp_slice = (tp_rank, tp_local_rows)

        if gathered_grad.shape[0] != expected_rows:
            raise RuntimeError(
                "Muon QKV split shape mismatch after gathering: "
                f"grad_shape={tuple(gathered_grad.shape)}, expected_rows={expected_rows}"
            )
        return gathered_grad, tp_slice

    @staticmethod
    def _restore_local_qkv_grad(gathered_grad, tp_slice):
        """Restore the TP dim-0 shard recorded by ``_gather_qkv_grad``."""
        if tp_slice is not None:
            tp_rank, tp_local_rows = tp_slice
            gathered_grad = gathered_grad[tp_rank * tp_local_rows : (tp_rank + 1) * tp_local_rows]
        return gathered_grad.contiguous()

    def _gather_gtp_remat_qkv_grad(self, p, grad, tp_group):
        """Reconstruct a fused QKV gradient from its GTP_remat row shards.

        GTP_remat shards dim 0 of the TP-local weight, padded at the end for alignment, without
        regard for q/k/v or head boundaries. Return the unpadded TP-local gradient and how to
        restore this rank's shard, or ``(grad, None)`` when ``p`` is not GTP_remat-sharded.
        """
        if not getattr(p, 'is_gtp_weight_remat', False) or not self.pg_collection:
            return grad, None
        gtp_remat_group = resolve_gtp_remat_group(
            self.pg_collection, getattr(p, 'expert_tp', False)
        )
        if gtp_remat_group is None or get_pg_size(gtp_remat_group) <= 1:
            return grad, None

        global_split_shapes = getattr(p, "qkv_split_shapes_global", None)
        if global_split_shapes is None:
            raise RuntimeError("Muon QKV split under GTP_remat requires global split shapes")
        tp_size = get_pg_size(tp_group) if tp_group is not None else 1
        if sum(global_split_shapes) % tp_size != 0:
            raise RuntimeError(
                "Muon QKV split under GTP_remat cannot derive the TP-local layout: "
                f"global_split_shapes={global_split_shapes}, tp_size={tp_size}"
            )
        tp_local_rows = sum(global_split_shapes) // tp_size

        gtp_remat_size = get_pg_size(gtp_remat_group)
        shards = [torch.empty_like(grad) for _ in range(gtp_remat_size)]
        torch.distributed.all_gather(shards, grad.contiguous(), gtp_remat_group)
        gathered_grad = torch.cat(shards, dim=0)
        if gathered_grad.shape[0] < tp_local_rows:
            raise RuntimeError(
                "Muon QKV split under GTP_remat reconstructed too few rows: "
                f"gathered_shape={tuple(gathered_grad.shape)}, expected_rows={tp_local_rows}"
            )
        gtp_remat_slice = (get_pg_rank(gtp_remat_group), grad.shape[0], gathered_grad.shape[0])
        return gathered_grad[:tp_local_rows], gtp_remat_slice

    @staticmethod
    def _restore_gtp_remat_qkv_grad(grad, gtp_remat_slice):
        """Restore the padded GTP_remat row shard recorded by ``_gather_gtp_remat_qkv_grad``."""
        if gtp_remat_slice is None:
            return grad
        gtp_rank, shard_rows, padded_rows = gtp_remat_slice
        if grad.shape[0] < padded_rows:
            grad = torch.nn.functional.pad(grad, (0, 0, 0, padded_rows - grad.shape[0]))
        return grad[gtp_rank * shard_rows : (gtp_rank + 1) * shard_rows].contiguous()

    def _orthogonalize_split_qkv(self, grad, split_shapes, orthogonalize_fn):
        """Split and reconstruct Megatron's interleaved fused QKV update."""
        if grad.ndim != 2:
            raise RuntimeError(f"Muon QKV gradient must be 2D, got {grad.ndim}D")
        if not split_shapes or any(size <= 0 for size in split_shapes):
            raise RuntimeError(f"Muon QKV split shapes must be positive: {split_shapes}")
        split_width = sum(split_shapes)
        if self.split_qkv_per_head:
            if grad.shape[0] != split_width:
                raise RuntimeError(
                    f"Muon per-head QKV split shape mismatch: grad_shape={tuple(grad.shape)}, "
                    f"split_shapes={split_shapes}"
                )
            if len(set(split_shapes)) == 1:
                # A 3D input selects Emerging-Optimizers' batched Newton-Schulz path.
                head_rows = split_shapes[0]
                return orthogonalize_fn(grad.view(len(split_shapes), head_rows, -1)).view_as(grad)
            return torch.cat(
                [orthogonalize_fn(head) for head in torch.split(grad, split_shapes, dim=0)], dim=0
            )

        if grad.shape[0] % split_width != 0:
            raise RuntimeError(
                f"Muon QKV split shape mismatch: grad_shape={tuple(grad.shape)}, "
                f"split_shapes={split_shapes}"
            )
        num_query_groups = grad.shape[0] // split_width
        grouped_grad = grad.view(num_query_groups, split_width, -1)
        projection_grads = torch.split(grouped_grad, split_shapes, dim=1)
        projection_grads = [
            projection.reshape(-1, grad.shape[-1]) for projection in projection_grads
        ]
        projection_grads = [
            orthogonalize_fn(projection).view(num_query_groups, -1, grad.shape[-1])
            for projection in projection_grads
        ]
        return torch.cat(projection_grads, dim=1).view_as(grad)

    def _orthogonalize_qkv_per_head(self, p, grad, tp_group):
        """Orthogonalize every Q, gate, K, and V head independently.

        Split sizes may describe complete heads in the local tensor or the global fused
        QKV tensor. For a global layout, reconstruct TP dimension 0 before splitting so
        heads crossing rank boundaries remain complete.
        """
        self._warn_distributed_qkv_fallback()
        local_split_shapes = getattr(p, "qkv_split_shapes", None)
        heads_are_complete = getattr(p, "qkv_split_heads_are_complete", None)
        use_local_layout = heads_are_complete is True or (
            heads_are_complete is None
            and local_split_shapes is not None
            and sum(local_split_shapes) == grad.shape[0]
        )
        if use_local_layout:
            qkv_split_shapes = local_split_shapes
        else:
            qkv_split_shapes = getattr(p, "qkv_split_shapes_global", None)
            if qkv_split_shapes is None:
                qkv_split_shapes = self.qkv_split_shapes
        if qkv_split_shapes is None:
            raise RuntimeError("Muon per-head QKV split requested but qkv_split_shapes is not set")
        if not qkv_split_shapes or any(size <= 0 for size in qkv_split_shapes):
            raise RuntimeError(
                f"Muon per-head QKV split shapes must be positive: {qkv_split_shapes}"
            )

        expected_rows = sum(qkv_split_shapes)
        gathered_grad, tp_slice = self._gather_qkv_grad(p, grad, tp_group, expected_rows)

        gathered_grad = self._orthogonalize_split_qkv(
            gathered_grad,
            qkv_split_shapes,
            lambda head_grad: self.scaled_orthogonalize_fn(
                head_grad,
                tp_group=None,
                partition_dim=None,
                tp_mode_this_group="duplicated" if self.tp_mode == "auto" else self.tp_mode,
            ),
        )

        return self._restore_local_qkv_grad(gathered_grad, tp_slice)

    def _orthogonalize_global_qkv(self, p, grad, tp_group, split_shapes):
        """Orthogonalize projections after reconstructing their global QKV layout."""
        self._warn_distributed_qkv_fallback()
        global_split_shapes = getattr(p, "qkv_split_shapes_global", None)
        if global_split_shapes is None:
            raise RuntimeError("Muon global QKV split requires global split shapes")
        expected_rows = sum(global_split_shapes)
        if expected_rows % sum(split_shapes) != 0:
            raise RuntimeError(
                f"Muon global QKV layout does not contain complete query groups: "
                f"global_split_shapes={global_split_shapes}, split_shapes={split_shapes}"
            )

        gathered_grad, tp_slice = self._gather_qkv_grad(p, grad, tp_group, expected_rows)
        gathered_grad = self._orthogonalize_split_qkv(
            gathered_grad,
            split_shapes,
            lambda projection_grad: self.scaled_orthogonalize_fn(
                projection_grad,
                tp_group=None,
                partition_dim=None,
                tp_mode_this_group="duplicated" if self.tp_mode == "auto" else self.tp_mode,
            ),
        )
        return self._restore_local_qkv_grad(gathered_grad, tp_slice)

    def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Orthogonalize the momentum.

        Args:
            p: The parameter tensor. It is necessary to pass param tensor in addition to
                momentum because a lot of information is only available in the param tensor,
                attributes for example.
            grad: The momentum tensor.

        Returns:
            The orthogonalized gradient tensor.
        """
        # TODO(deyuf): switch to group
        if self.pg_collection:
            tp_group = (
                self.pg_collection.expt_tp
                if getattr(p, 'expert_tp', False)
                else self.pg_collection.tp
            )
        else:
            tp_group = None
        partition_dim = None if self.tp_mode == "blockwise" else getattr(p, "partition_dim", None)
        if partition_dim == -1:
            partition_dim = None

        if self.split_qkv and self.is_qkv_fn(p):  # type: ignore[misc]
            # Preserve dev's explicit split-QKV contract for every TP mode. GTP shards
            # cut through projection boundaries, so reconstruct before splitting rather
            # than falling back to unsplit NS for blockwise/distributed GTP.
            grad, gtp_slice = self._gather_gtp_remat_qkv_grad(p, grad, tp_group)
            if self.split_qkv_per_head:
                grad = self._orthogonalize_qkv_per_head(p, grad, tp_group)
                return self._restore_gtp_remat_qkv_grad(grad, gtp_slice)

            qkv_split_shapes = getattr(p, "qkv_split_shapes", None)
            if qkv_split_shapes is None:
                qkv_split_shapes = self.qkv_split_shapes
            if not qkv_split_shapes:
                raise RuntimeError("Muon QKV split requested but qkv_split_shapes is not set")
            if getattr(p, "qkv_split_groups_are_complete", None) is False:
                grad = self._orthogonalize_global_qkv(p, grad, tp_group, qkv_split_shapes)
                return self._restore_gtp_remat_qkv_grad(grad, gtp_slice)

            grad = self._orthogonalize_split_qkv(
                grad,
                qkv_split_shapes,
                lambda projection_grad: self.scaled_orthogonalize_fn(
                    projection_grad,
                    tp_group,
                    partition_dim,
                    tp_mode_this_group="duplicated" if self.tp_mode == "auto" else self.tp_mode,
                ),
            )
            return self._restore_gtp_remat_qkv_grad(grad, gtp_slice)

        return self.scaled_orthogonalize_fn_with_gtp_remat(p, grad, tp_group, partition_dim)


class TensorParallelAdaptiveMuon(TensorParallelMuon, AdaptiveMuon):
    """Tensor Parallel Adaptive Muon optimizer.

    This class extends Muon by adding AdamW-style or NorMuon-style second moment
    accumulation after orthogonalization. This idea was first explored in D.E. Carlson,
    E. Collins, Ya-Ping Hsieh, L. Carin, and V. Cevher. *Preconditioned spectral
    descent for deep learning.* In Advances in neural information processing systems 28 (2015).
    The step() method is overridden to include second moment normalization logic.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr: Learning rate.
        momentum: The exponential decay rate for momentum.
        nesterov: Whether to use Nesterov momentum.
        weight_decay: Weight decay coefficient.
        use_decoupled_weight_decay: Whether to use decoupled weight decay.
        split_qkv: Whether to split QKV weights for orthogonalization.
        split_qkv_per_head: Whether to orthogonalize individual Q, gate, K, and V heads.
        is_qkv_fn: Function to determine if a tensor is a QKV weight.
        qkv_split_shapes: Shapes for splitting QKV weights.
        fp32_matmul_prec: Precision for FP32 matrix multiplication.
        coefficient_type: The type of coefficient set to use for the Newton-Schulz iteration.
        num_ns_steps: The number of iteration steps to use in the Newton-Schulz iteration.
        scale_mode: The type of scale factor to use for the update.
        extra_scale_factor: The additional scale factor to use for the update.
        pg_collection: Process group collection for distributed training.
        tp_mode: Tensor parallel mode ("blockwise", "duplicated", "distributed", or "auto").
        use_syrk: Whether to use the Triton SYRK kernel for the Gram matrix in
            Newton-Schulz. Requires emerging_optimizers >= 0.4.0.
        moment2_method: Method for second moment accumulation ("adamuon" or "normuon").
        beta2: The exponential decay rate for second moment.
        eps: Small constant for numerical stability.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.01,
        use_decoupled_weight_decay: bool = True,
        split_qkv: bool = False,
        split_qkv_per_head: bool = False,
        is_qkv_fn: Callable[[torch.Tensor], bool] | None = None,
        qkv_split_shapes: list[int] | None = None,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        pg_collection: Optional[ProcessGroupCollection] = None,
        tp_mode: Literal["blockwise", "duplicated", "distributed", "auto"] = "duplicated",
        use_syrk: bool = False,
        ns_batch_size: int = 1,
        ns_batch_max_mb: int = 2048,
        ns_gather_ahead: int = 0,
        moment2_method: Literal["adamuon", "normuon"] = "adamuon",
        beta2: float = 0.95,
        eps: float = 1e-8,
    ) -> None:
        # ns_batch_size / ns_batch_max_mb are accepted so the shared ``muon_*`` config mapping
        # can pass them, but AdaptiveMuon.step owns this class's step, so they are inert here.
        TensorParallelMuon.__init__(
            self,
            params,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            use_decoupled_weight_decay=use_decoupled_weight_decay,
            split_qkv=split_qkv,
            split_qkv_per_head=split_qkv_per_head,
            is_qkv_fn=is_qkv_fn,
            qkv_split_shapes=qkv_split_shapes,
            fp32_matmul_prec=fp32_matmul_prec,
            coefficient_type=coefficient_type,
            num_ns_steps=num_ns_steps,
            scale_mode=scale_mode,
            extra_scale_factor=extra_scale_factor,
            pg_collection=pg_collection,
            tp_mode=tp_mode,
            use_syrk=use_syrk,
            ns_batch_size=ns_batch_size,
            ns_batch_max_mb=ns_batch_max_mb,
            ns_gather_ahead=ns_gather_ahead,
        )
        self.scale_mode = scale_mode
        self.extra_scale_factor = extra_scale_factor
        self.moment2_method = moment2_method

        for group in self.param_groups:
            group.setdefault("beta2", beta2)
            group.setdefault("eps", eps)

    @torch.no_grad()  # type: ignore[misc]
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Step function"""
        return AdaptiveMuon.step(self, closure)


def _kwargs_from_config(optimizer_cls: type, prefix: str, config) -> Dict[str, Any]:
    """Match ``optimizer_cls.__init__`` parameters to config attributes.

    For each init parameter, looks for ``{prefix}_{name}`` on *config* first,
    then falls back to ``{name}`` (unprefixed).  ``self`` and ``params`` are
    always skipped.
    """
    skip_params = {"self", "params"}
    sig = inspect.signature(optimizer_cls.__init__)
    kwargs: Dict[str, Any] = {}
    for name in sig.parameters:
        if name in skip_params:
            continue
        prefixed = f"{prefix}_{name}"
        if hasattr(config, prefixed):
            kwargs[name] = getattr(config, prefixed)
        elif hasattr(config, name):
            kwargs[name] = getattr(config, name)
    return kwargs


def _muon_bucket_tp_mode(config, is_expert: bool) -> str:
    """NS mode of one Muon bucket: ``muon_tp_mode`` for the dense bucket, the effective expert
    mode (:func:`muon_effective_expert_tp_mode`) for the expert bucket.

    The bucket flag is the single input every ``muon`` builder keys on, so class, kwargs and
    process-group axes cannot disagree.
    """
    if is_expert:
        return muon_effective_expert_tp_mode(config)
    return getattr(config, 'muon_tp_mode', 'duplicated')


def _muon_config_to_cls(config, is_expert: bool = False) -> type:
    """The ``muon`` registry entry's class.

    ``LayerShardedMuon`` when the bucket's mode is ``'layer_sharded'`` (a registry-level class
    selector, not a TensorParallelMuon runtime mode), ``TensorParallelMuon`` otherwise. The
    registry calls this without ``is_expert``, which is the dense bucket and thus the
    non-hybrid behavior; the expert bucket of a hybrid configuration passes ``is_expert=True``.
    """
    if _muon_bucket_tp_mode(config, is_expert) == 'layer_sharded':
        from megatron.core.optimizer.layer_sharded_muon import LayerShardedMuon

        return LayerShardedMuon
    return TensorParallelMuon


def _muon_config_to_kwargs(config, model_chunks, pg_collection) -> Dict[str, Any]:
    """Convert OptimizerConfig to TensorParallelMuon constructor kwargs.

    Shared by ``adaptive_muon`` (which layers its own kwargs on top), so it never
    dispatches on ``muon_tp_mode='layer_sharded'``; that lives in
    :func:`_muon_registry_config_to_kwargs`.
    """
    kwargs = _kwargs_from_config(TensorParallelMuon, "muon", config)
    kwargs["is_qkv_fn"] = lambda p: getattr(p, "is_qkv", False)
    kwargs["qkv_split_shapes"] = _get_qkv_split_shapes(
        model_chunks[0].config, split_qkv_per_head=kwargs.get("split_qkv_per_head", False)
    )
    kwargs["pg_collection"] = pg_collection
    return kwargs


def _muon_registry_config_to_kwargs(
    config, model_chunks, pg_collection, *, is_expert: bool = False
) -> Dict[str, Any]:
    """``config_to_kwargs`` for the ``muon`` registry entry.

    TensorParallelMuon kwargs, plus LayerShardedMuon's own when
    :func:`_muon_config_to_cls` selects it (the same helper the entry's
    ``config_to_cls`` uses, so class and kwargs cannot disagree).

    ``is_expert`` selects the expert bucket of a hybrid dense/expert configuration
    (:func:`_create_emerging_optimizer`): its mode is the effective expert mode and its
    LayerShardedMuon defaults to the expert (expt_gtp_remat, expt_tp) axes. The registry
    itself calls this with the default, i.e. for the dense bucket.
    """
    kwargs = _muon_config_to_kwargs(config, model_chunks, pg_collection)
    cls = _muon_config_to_cls(config, is_expert)
    if cls is TensorParallelMuon:
        if is_expert:
            # The reflective builder read config.muon_tp_mode; the expert bucket runs its own.
            kwargs["tp_mode"] = _muon_bucket_tp_mode(config, is_expert)
        return kwargs
    kwargs.update(_kwargs_from_config(cls, "muon", config))
    # DSV4's TensorParallelMuon supports per-head QKV splitting. Expert layer
    # sharding has no QKV split path and its constructor does not accept this flag.
    if kwargs.pop("split_qkv_per_head", False) and not is_expert:
        raise ValueError("LayerShardedMuon does not implement per-head split-QKV.")
    # 'layer_sharded' selected the class; it is not a TensorParallelMuon mode, so the
    # delegated (empty-homes fallback) path runs the bitwise reference mode instead.
    kwargs["tp_mode"] = "duplicated"
    # No config attr for these: the axes come from the collection. An expert bucket defaults
    # to the expert axes; per-group NS-home wiring (_wire_layer_sharding_ns_homes) assigns
    # domains group by group regardless, this keeps the fallback consistent.
    if is_expert:
        kwargs["gtp_remat_group"] = getattr(pg_collection, "expt_gtp_remat", None)
        kwargs["tp_group"] = getattr(pg_collection, "expt_tp", None)
        kwargs["split_qkv"] = False
    else:
        kwargs["gtp_remat_group"] = getattr(pg_collection, "gtp_remat", None)
        kwargs["tp_group"] = getattr(pg_collection, "tp", None)
    return kwargs


def _adaptive_muon_config_to_kwargs(config, model_chunks, pg_collection) -> Dict[str, Any]:
    """Convert OptimizerConfig to TensorParallelAdaptiveMuon constructor kwargs."""
    kwargs = _muon_config_to_kwargs(config, model_chunks, pg_collection)
    kwargs.update(_kwargs_from_config(TensorParallelAdaptiveMuon, "adaptive_muon", config))
    return kwargs


def _default_adam_based_eopt_config_to_kwargs(
    eopt_name, config, model_chunks, pg_collection
) -> Dict[str, Any]:
    """Convert OptimizerConfig to default emerging optimizer constructor kwargs."""
    kwargs = _kwargs_from_config(registry.get_optimizer_cls(eopt_name), eopt_name, config)
    kwargs["betas"] = (config.adam_beta1, config.adam_beta2)
    return kwargs


# -----------------------------------------------------------------------
# Register emerging optimizers
# -----------------------------------------------------------------------
_EMERGING_OPTIMIZERS.update(
    {
        'muon': EmergingOptimizerEntry(
            optimizer_cls=TensorParallelMuon,
            init_state_fn=_eopt_init_state_fn,
            config_to_kwargs=_muon_registry_config_to_kwargs,
            config_to_cls=_muon_config_to_cls,
            default_param_overrides={
                ParamKey(predicate=ParamPredicate(name="muon_excluded", fn=_is_muon_excluded)): {
                    'optimizer': 'adam'
                }
            },
        ),
        "adaptive_muon": EmergingOptimizerEntry(
            optimizer_cls=TensorParallelAdaptiveMuon,
            init_state_fn=_eopt_init_state_fn,
            config_to_kwargs=_adaptive_muon_config_to_kwargs,
            default_param_overrides={
                ParamKey(predicate=ParamPredicate(name="muon_excluded", fn=_is_muon_excluded)): {
                    'optimizer': 'adam'
                }
            },
        ),
    }
)

# Register soap with default config
# TODO(skyw): register all emerging optimizers.
if HAVE_EMERGING_OPTIMIZERS:
    for eopt_name in registry.get_optimizer_name_list():
        if eopt_name in _EMERGING_OPTIMIZERS:
            # skip already registered local versions, e.g. TensorParallel versions.
            continue
        _EMERGING_OPTIMIZERS[eopt_name] = EmergingOptimizerEntry(
            optimizer_cls=registry.get_optimizer_cls(eopt_name)
        )
