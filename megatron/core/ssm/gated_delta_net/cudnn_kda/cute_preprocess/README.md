# CuTe DSL CP pre-process for the cuDNN KDA backend

Optional, experimental `BoundaryStateProducer` for `linear_cp_mode="chunkwise"` on the cuDNN
KDA backend (`megatron/core/ssm/gated_delta_net/cudnn_kda`). It replaces the default
`CudnnSummaryProducer` (cuDNN per-span summaries + `all_gather` + transition fold) with the
fused CuTe DSL kernels of NVIDIA/Megatron-LM PR #6511: one warp-specialised tcgen05 launch per
direction that scans this rank's boundary sequence, pushes its summary strips to the consumer
ranks through PyTorch symmetric memory, folds the predecessors' (forward) / successors'
(backward) strips in-kernel and returns the merged boundary state.

## Selecting it

```python
from megatron.core.ssm.gated_delta_net.cudnn_kda import cute_preprocess

if cute_preprocess.is_available():
    cute_preprocess.register()          # registers the "cute" producer
    out, _ = chunk_kda_cudnn(..., cp_context=ctx, cp_state_producer="cute")
else:
    print(cute_preprocess.unavailable_reason())
```

Nothing in the cuDNN backend imports this package by itself, and importing it imports neither
cutlass nor flash-linear-attention. The producer is selected explicitly; there is no
environment-variable knob and no silent fallback: an envelope miss raises
`NotImplementedError` with the reason (`backend.get_cutedsl_rejection_reasons()` keeps a
count).

## Envelope

* SM100-class GPU (compute capability 10.x: B200 / GB200 / GB300), NCCL CP group of 2..8 ranks,
  `torch.distributed._symmetric_memory`, cuda-python (`cuda.bindings.driver`).
* One packed sequence per rank shard (`B == 1`, THD), `K == V == 128`, `HV % 4 == 0`, `HV == H`
  (no grouped value attention), bf16 q/k/v, fp32 gate, chunk size 64.
* No CUDA-graph capture: both producer methods refuse at capture time
  (`torch.cuda.is_current_stream_capturing()`), since the kernels spin on symmetric-memory flags
  and the producer runs host-side collectives. The dispatcher's config-driven check
  (`backend._cuda_graphs_enabled`) is inert on this path because the context metadata is attached
  without a model config.
* `use_qk_l2norm_in_kernel`, `use_beta_sigmoid_in_kernel` and `allow_neg_eigval` unset: the
  producer composes FLA's chunk kernels, which take L2-normalised q/k and post-sigmoid beta
  (Megatron's KDA module hands them over that way).
* The bounded ("safe") gate needs `a_log` and `-5 <= gate_lower_bound < 0` (FLA's range); the
  unbounded softplus gate and a caller-supplied log decay go through `chunk_local_cumsum`.

## Toolchain caveat

The engines (`fused_ws.py`, `bwd_ws.py`, `protocol.py`) were written against
nvidia-cutlass-dsl 4.5.x; importing them under 4.8.x fails (`cutlass.cute.core.ThrMma` is
gone). cudnn-frontend 1.32, which the cuDNN KDA backend requires, needs the DSL >= 4.7, so in an
environment where the cuDNN backend runs this producer cannot, until the kernels are ported.
`unavailable_reason()` reports exactly that. The first launch of each wrapper is a multi-second
`cute.compile`, entered by all ranks in lockstep right after a rank-consistent handshake.

## What the producer computes

For each call the producer slices the cuDNN-convention THD operands to the boundary sequence
(forward: the LAST local sequence, `cu_list[-2:]`; backward: the FIRST, `cu_list[:2]`, as FLA's
own producer does) and runs FLA 0.5.x Triton kernels on that slice as one dense sequence:

| FLA function | role | output |
| --- | --- | --- |
| `fla.ops.kda.gate.kda_gate_chunk_cumsum` | bounded gate + chunk-local cumsum (log2, `scale=RCP_LN2`) | `gk` fp32 `[1, T, H, K]` |
| `fla.ops.utils.chunk_local_cumsum` | chunk-local cumsum of an already-applied log decay | `gk` fp32 `[1, T, H, K]` |
| `fla.ops.kda.chunk_intra.chunk_kda_fwd_intra` (-> `recompute_w_u_fwd`) | WY representation and pre-gated keys/queries | `w`, `u`, `kg` (and `qg` with `disable_recompute=True`) bf16 |
| `fla.ops.kda.chunk_bwd.chunk_kda_bwd_dAv` | intra-chunk term `dv = Aqk @ do` the backward pre-process consumes | `dv` bf16 `[1, T, HV, V]` |

Forward: `kg, w, u, gk` go to `backend.chunk_gated_delta_rule_fwd_h_pre_process_cutedsl`
(`windowed=True`); the fp32 `[N, HV, K, V]` result (slot 0 live) is transposed to cuDNN's
`[N, HO, V, K]` with `ops.fla_state_to_cudnn`, or `None` is returned on `is_first_rank`.
Backward: `qg, kg, w, do, dv, gk, scale` go to `..._bwd_dhu_pre_process_cutedsl`; slot `N - 1`
is live, `None` on `is_last_rank`. Both calls are collective and every rank must make them.

The fused kernels decide `fused` / `noop` identically on every rank from the GLOBAL packed
offsets. FLA's `FLACPContext` only carries the rank-local ones, so the first call on a context
object reconstructs them with one `all_gather_object` over the CP group
(`context_metadata.gather_global_cu_seqlens`) and attaches PR #6511's metadata
(`prepare_cp_context_metadata`).

## Files

| file | origin |
| --- | --- |
| `_host.py` | PR #6511, byte-identical copy (FLA MIT header) |
| `fused_ws.py`, `bwd_ws.py`, `protocol.py`, `symm.py` | PR #6511 code re-formatted with this repository's black settings (line length 100) and given one-line pylint docstrings; no semantic change, FLA MIT header kept |
| `backend.py` | PR #6511 dispatcher; env knobs removed, misses raise, `windowed=True` added |
| `context_metadata.py` | PR #6511 `prepare_cp_context_metadata` + the global-offset reconstruction |
| `producer.py` | `CuTePreprocessProducer` (the protocol implementation) |
| `__init__.py` | `is_available()`, `unavailable_reason()`, `register()` |

Launch counters: `backend.get_cutedsl_fused_launch_count()`,
`get_cutedsl_fused_bwd_launch_count()`, `get_cutedsl_cp_noop_count()`.
