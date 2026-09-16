# FP32 tensor-parallel sums in BF16 training

A row-parallel forward GEMM and a column-parallel input-gradient GEMM each
produce a partial sum. Rounding each partial to BF16 before communication adds
an extra rounding boundary that is absent at TP1. Casting an already-rounded
partial to FP32 inside the collective cannot recover the lost information.

`TransformerConfig.tp_reduce_in_fp32` (CLI `--tp-reduce-in-fp32`) requests an
FP32 GEMM output for these partials and keeps that dtype through all-reduce or
reduce-scatter. The result is cast to BF16 after communication completes.
Input/weight GEMM dtypes and weight-gradient accumulation are unchanged.
The option is disabled by default and leaves TP1 arithmetic unchanged.

`TELinear` passes this option only when TE owns the TP communication; replicated
and explicit expert-communication projections keep their existing behavior.
`TELayerNormColumnParallelLinear` passes the same option to TE's independent
fused backward, which casts the completed input-gradient sum back to BF16
before the original normalization backward.

## Native vocabulary projection

Hybrid models use `tensor_parallel.ColumnParallelLinear` for both the LM and
MTP output heads even when the transformer layers use TE. This native layer
must honor the same option: its input gradient sums over vocabulary shards.
It uses a BF16-by-BF16 TE GEMM with FP32 output, followed by FP32 all-reduce or
reduce-scatter and a final BF16 cast. Forward logits and weight-gradient
accumulation, including fused/deferred accumulation, keep their existing paths.

Frozen column weights are supported too. With sequence parallelism, the
all-gather and backward reduce-scatter stay inside the custom autograd
function so autograd cannot cast a local partial before communication.
Projections with explicit expert communication or disabled gradient reduction
retain their original behavior. Native row-parallel layers are outside this
option's current scope; transformer row projections use TE.

## Supported execution

- Ordinary BF16 training with Transformer Engine.
- Sequence-parallel reduce-scatter or non-SP all-reduce.
- Expert tensor parallelism of one for MoE models.
- No FP8/FP4, Userbuffers overlap, or symmetric all-reduce.

A TE build with the explicit `tp_reduce_in_fp32` constructor argument is
required for both `Linear` and `LayerNormLinear`. MCore checks this interface
and raises an actionable error for unsupported builds. No installed source
rewriting or process-global function patching is used by this implementation.

This reduces one source of TP-dependent rounding; it does not promise bitwise
training equality. KDA's complete-sequence decay projection has a separate
contract described in `kda-tensor-parallel-projections.md`.
