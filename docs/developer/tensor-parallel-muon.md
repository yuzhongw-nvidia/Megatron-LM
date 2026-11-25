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

## Layer-sharded Muon with chunked optimizer-state offload

Use `--optimizer muon --use-distributed-optimizer --muon-tp-mode layer_sharded`
with `--muon-no-split-qkv --chunked-optimizer-state-offload` and a positive
`--optimizer-state-offload-fraction`. Disable `muon_split_qkv_per_head` in
inherited recipes as well. Layer-sharded Newton-Schulz currently operates on
whole matrices, so a numerical or timing comparison must also disable
split-QKV in the duplicated reference.

The offloader steps the optimizer on one state chunk at a time. Dense/expert
process groups are resolved from the `is_expert_parallel` marker, which
survives chunk repacking, rather than the group's position in the temporary
list. Newton-Schulz homes are LPT-balanced within each call; exchange plans
are cached by parameter identities and process-group domain. The duplicated
optimizer's cross-step gather prefetch hook is inert on the layer-sharded
path. The state-schema check accepts exactly `TensorParallelMuon` and
`LayerShardedMuon`, and continues to reject AdaptiveMuon.

Existing coverage is in `tests/unit_tests/optimizer/test_layer_sharded_muon.py`,
`tests/unit_tests/optimizer/test_layer_sharded_e2e_parity.py`, and
`tests/unit_tests/test_optimizer_state_offloading.py`. The distributed parity
cases require four GPUs and include dense TP2/GTP2, expert ETP2/EGTP2, and
Kimi TP1/GTP4/EGTP4.

## Separate dense and routed-expert Muon modes

`--muon-tp-mode auto --muon-expert-tp-mode layer_sharded` selects
TensorParallelMuon for dense weights and LayerShardedMuon for routed MoE expert
weights. An unset expert mode inherits the dense mode. Shared experts stay in the
dense family. The override requires Muon, a model with experts, and the layer-wise
distributed optimizer when either mode is layer-sharded.

Semantic expert identity (`param.expert_tp`) is independent of the gradient
communication marker (`is_expert_parallel`): at EP1 with matching dense/expert TP
and GTP sizes, expert parameters can use dense gradient communication while still
requiring the expert optimizer mode.

Dense QKV/per-head splitting remains available with dense `auto`; the expert
optimizer receives splitting disabled since routed experts have no QKV weights.
Dense `layer_sharded` continues to require unsplit matrices. Expert sharding
uses expert GTP/TP process groups, and the existing offloader dispatches each
chunk to its owning base optimizer. Use EGTP greater than one to exercise
cross-rank expert layer sharding in a smoke test.

`tests/unit_tests/optimizer/test_hybrid_muon_dispatch.py` checks class selection,
process-group domains, dense split-QKV preservation, and EP1 semantic expert
identity. GPU end-to-end and chunked-offload validation must accompany a new
training recipe; constructor tests alone do not establish numerical parity.

Hybrid modes with chunked offload create two state managers. Initialization occurs once
per child. Gradient finalization stages only the first child's state; subsequent
children prefetch immediately before their own update. Transfer streams remain shared
and dense/expert parameter ownership is unchanged. End-to-end hybrid offload coverage
uses dense auto with per-head splitting and expert layer sharding on TP1/GTP4/EGTP4.
