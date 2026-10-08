# GDN and KDA Elementwise Fusion

GDN-family layers have two independent, opt-in `TransformerConfig` options:

| Option | Fused operations | Default |
| --- | --- | --- |
| `gdn_pre_gated_delta_rule_fusion` | Causal convolution, SiLU, layout transforms, Q/K L2 normalization, head expansion, beta and decay preparation | `False` |
| `gdn_gated_output_norm_fusion` | Output RMSNorm and SiLU (GDN) or sigmoid (KDA) gating after the gated delta rule | `False` |

Both options support the `gdn` and `kda` attention variants, including the
deprecated `gated_delta_net` alias for GDN. They can be enabled independently
or together. KDA retains its existing pre-GDR fusion and low-rank projection
paths. The linear-attention recurrence and output projection retain their
existing implementations. The post-GDR implementation is adapted from Layali Rashid's
[output-gating fusion in PR #7368](https://github.com/NVIDIA/Megatron-LM/pull/7368).
Its post-GDR kernels operate on local tensor shapes and strides, including
multi-batch projection views and heads redistributed by context parallelism.

For example, configure a supported BF16 GDN model with:

```python
config = TransformerConfig(
    # ... model dimensions and other training options ...
    experimental_attention_variant="gdn",
    gdn_pre_gated_delta_rule_fusion=True,
    gdn_gated_output_norm_fusion=True,
)
```

For KDA, set `experimental_attention_variant="kda"` and
`gdn_gated_output_norm_fusion=True`. Select `gdn_pre_gated_delta_rule_fusion`
independently to use the existing KDA preprocessing fusion.
KDA uses sigmoid gating regardless of the convolution activation. Its fused
forward computes `RMSNorm(x) * sigmoid(gate)`; backward uses
`sigmoid(gate) * (1 - sigmoid(gate))` for the gate derivative. Both variants share
the layout-aware kernel and select their gate formula at compile time. Existing
KDA head-count and head-dimension constraints still apply.

## Post-GDR requirements

The post-GDR fusion checks its requirements on **every forward**, including
selective output-norm recomputation. When enabled, unsupported inputs raise
`ValueError`; there is no silent fallback. Disable the option to use the
existing unfused path.

The supported configuration requires:

- `deterministic_mode=False`;
- an `RMSNorm` output normalization module; GDN also requires SiLU/Swish activation;
- nonempty CUDA BF16 or FP16 recurrence output;
- matching output and gate shapes `[batch, sequence_length, local_heads, head_dim]`
  with a power-of-two head dimension;
- a gate in the activation dtype or FP32 and a contiguous BF16, FP16 or FP32
  RMSNorm weight of length `head_dim`, all on the same CUDA device.

The gate may be a strided view into the input projection. Fusion consumes
that view directly, including separate batch and sequence strides; it does
not materialize a contiguous copy. Strided recurrence outputs are supported
as well. Outputs and returned input gradients are contiguous in logical order.
Context parallelism retains the existing communication before and after this
local operation, and the existing GDN/TP/CP shape constraints still apply. The checks
inspect tensor metadata without synchronizing CUDA. Post-GDR fusion can
process packed sequences because normalization and gating operate per token;
the existing GDN packed-sequence checks still apply.

`MCORE_GDN_FUSION` is not used. Select the feature through
`TransformerConfig.gdn_gated_output_norm_fusion` (or the corresponding
`--gdn-gated-output-norm-fusion` training argument).

## Execution and numerical behavior

The training `forward` runs pre-GDR preprocessing, the recurrence, gated output
normalization, and the output projection in `_forward_compute`. Inference keeps
its existing dispatch. Selective `gdn_norm_out` recomputation retains its existing
checkpoint lifecycle.

The post-GDR kernels preserve the activation-dtype RMSNorm materialization boundary
before the FP32 gating multiply, and round the gradient entering RMSNorm to the
activation dtype. They support first-order autograd only.
Floating-point operation ordering can differ from the unfused path;
correctness tests do not establish bitwise equivalence or training
convergence. Enabling either fusion is incompatible with deterministic mode.


## Packed boundary selection and runtime validation

GDN/KDA `_resolve_cu_seqlens` only selects padded boundaries when available,
otherwise the actual boundaries. It takes no token capacity, CP size, or strict
validation argument and performs no value checks or host transfers. This policy
is the same for CP1, headwise CP, and chunkwise CP.

Q/KV boundary equality remains a layer runtime check controlled by
`strict_runtime_validation`, including with a prepared FLA CP context. KDA also
checks that both boundary arrays contain at least one sequence. These checks
are not relocated to microbatch construction. In particular, `always` retains
the synchronization from CUDA Q/KV equality; disabling strict validation removes
that runtime check. Existing CP route preparation keeps its own layout checks.

The original resolver checks run once in `prepare_packed_seq_params` for every
packed microbatch containing GDN/KDA, regardless of CP size or mode: each selected
Q/KV boundary endpoint must match the physical input token count, and every sequence
length must be divisible by the microbatch's effective CP size. These checks use CPU
snapshots during batch preparation, independently of the layer's strict flag.
Callers supply `local_tokens` from the CP-local batch tensors before TP sequence
sharding, or an explicit global `capacity` on fixed-shape middle pipeline stages.
The expected length is never inferred from the boundary tensor being checked.

## Packed chunkwise-CP preparation

The training helper `prepare_packed_seq_params` builds FLA metadata once per
packed chunkwise-CP microbatch for models containing GDN/KDA, including hybrid
MTP. It stores one derived `PackedSeqParams.fla_cp_context`, shared by the layers.
The preparation helper reuses the CPU Q boundaries from the checks above.
`prepare_linear_attention_cp` lives beside `prepare_packed_seq_params` in
`megatron/training/utils/packed_seq_utils.py`; no additional core module is needed.

Call `megatron.training.utils.packed_seq_utils.prepare_packed_seq_params` with
the model config and physical token count before model execution. Integrations
that construct their own metadata must validate the physical boundaries and
provide the FLA context before entering packed chunkwise-CP layers. A fresh temporary CPU snapshot avoids stale FLA identity-cache
hits when CUDA boundary buffers are reused. Keep a microbatch's metadata unchanged
through backward, and prepare again for the next microbatch. FLA may retain the
snapshot in its own bounded cache; only its derived context is stored on the batch.

Context construction and its CPU/GPU transfers occur before the model, even when
FLA caching is disabled. With strict runtime validation disabled, the prepared
layer entrance performs no metadata H2D/D2H, scalar readback, or host synchronization.
Headwise permutations, SBHD context caching, and later recurrence chunk/backward
indices are unchanged.
