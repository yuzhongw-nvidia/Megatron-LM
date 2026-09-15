# Tensor-parallel Muon parameter metadata

Muon uses a parameter's logical matrix shape when orthogonalizing its gradient.
In `duplicated` TP mode, a parameter with `partition_dim=0` or `1` is gathered
across the tensor-parallel group before Newton-Schulz; an unsharded parameter
uses `partition_dim=-1`. The `tensor_model_parallel` flag alone does not select
this optimizer path.

A duplicated TE projection owns a complete weight on every TP rank. TE receives
`parallel_mode=None` for this projection, but its default parameter metadata
can retain `partition_dim=0`. The MCore wrappers therefore explicitly restore
all three replicated-parameter attributes after TE initializes the parameters:

- `tensor_model_parallel=False`
- `partition_dim=-1`
- `partition_stride=1`

This applies to both `TELinear` in duplicated mode and
`TERMSNormDuplicatedLinear`. Expert/data-parallel reduction attributes retain
their existing meanings. Without this normalization, duplicated Muon can
concatenate repeated copies of an already complete matrix and compute an update
for the wrong matrix shape.

`test_duplicated_telinear_muon_update_matches_unsharded_reference` in
`tests/unit_tests/test_emerging_optimizers.py` exercises a real TE parameter
and real Newton-Schulz updates at TP2 against a complete-matrix TP1 reference,
including both tall and wide weight matrices.
`test_duplicated_linear_has_replicated_tensor_parallel_metadata` in
`tests/unit_tests/transformer/moe/test_latent_moe_layer.py` covers both wrappers.

## Newton-Schulz arithmetic precision

`--muon-fp32-matmul-prec` accepts PyTorch's `medium`, `high`, and `highest`
settings. The default remains `medium`. The earlier CLI choices incorrectly
included `low` and omitted `highest`; `low` is not a PyTorch precision setting.

Use `highest` when diagnosing update sensitivity: the pinned Emerging Optimizers
implementation keeps Newton-Schulz arithmetic in FP32 for this setting, while
`medium` casts its normalized input to BF16. This changes arithmetic precision;
it does not change the logical TP matrix or per-head splitting domain. Training
trajectory parity still requires an end-to-end comparison.

## Interleaved TP matrix layouts

A gated MLP's first projection uses `partition_stride=2`: each TP rank owns its
local gate rows followed by its local up rows. The checkpoint/global matrix
stores all gate rows before all up rows. Concatenating rank-local gradients in
rank order therefore permutes the global matrix's rows.

Duplicated Muon reconstructs strided parameters in global order before calling
Newton-Schulz, then restores the local strided update. This applies to both
partition axes. It preserves the mathematical update and avoids a TP-dependent
GEMM reduction order; in BF16 NS, equivalent row permutations can otherwise
produce measurably different updates even from exactly the same gradient.
QKV layouts retain their existing projection/per-head handling. Blockwise and
distributed modes retain their existing behavior.

`test_muon_strided_tp_update_matches_global_matrix` compares real TP2 updates
against the upstream NS API on the full global matrix for tall/wide shapes,
both partition axes, and `medium`/`highest` arithmetic. It requires exact equality
after restoring the corresponding local shard.
