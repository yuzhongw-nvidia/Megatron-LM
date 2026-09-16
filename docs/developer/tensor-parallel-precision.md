# FP32 tensor-parallel sums in BF16 training

A row-parallel forward GEMM and a column-parallel input-gradient GEMM each
produce a partial sum. Rounding each partial to BF16 before communication adds
an extra rounding boundary that is absent at TP1. Casting an already-rounded
partial to FP32 inside the collective cannot recover the lost information.

`TransformerConfig.tp_reduce_in_fp32` (CLI `--tp-reduce-in-fp32`) requests an
FP32 GEMM output for these partials and keeps that dtype through all-reduce or
reduce-scatter. The result is cast to BF16 after communication completes.
Input/weight GEMM dtypes and weight-gradient accumulation are unchanged.
The option is disabled by default. Both TE and native paths use bounded
contractions at TP1 too, as described below. Both variants must be rerun from
the same checkpoint when this option or its accumulation implementation changes.

`TELinear` passes this option only when TE owns the TP communication; replicated
and explicit expert-communication projections keep their existing behavior.
`TELayerNormColumnParallelLinear` passes the same option to TE's independent
fused backward, which casts the completed input-gradient sum back to BF16
before the original normalization backward.

## Bounded TE contractions

Returning FP32 from a long BF16 GEMM does not eliminate internal contraction
error. Fixed-input KDA and dense MLP projections still show TP-dependent errors
before the final BF16 cast. TE row forward and column input gradients therefore
use contraction tiles of at most 512 BF16 elements and add their FP32 outputs
before the FP32 collective. Fused LayerNormLinear uses the same helper for its
independent column dgrad. A shorter final tile is supported.

The option applies this algorithm at TP1 as well. It preserves the BF16 operand
and activation boundaries, FP32 communication, column forward, row dgrad, and
weight-gradient paths. The cost is additional GEMM launches, contiguous slices
and FP32 additions. This opt-in correctness path reduces accumulation error;
it does not promise bitwise equality across arbitrary TP partition layouts.

## Native vocabulary projection

Hybrid models use `tensor_parallel.ColumnParallelLinear` for both the LM and
MTP output heads even when the transformer layers use TE. This native layer
must honor the same option: its input gradient sums over vocabulary shards.
It uses BF16-by-BF16 TE GEMMs with FP32 outputs, followed by FP32 all-reduce or
reduce-scatter and a final BF16 cast. Forward logits and weight-gradient
accumulation, including fused/deferred accumulation, keep their existing paths.

An FP32 GEMM output alone does not resolve accumulation error along a very
large contraction dimension. In a fixed-input 163840-word projection, full
and TP2-sharded GEMMs still differed before BF16 rounding. Native column input
gradients therefore split the vocabulary contraction into tiles of at most
4096 elements, and add those FP32 GEMM outputs in FP32 before communication.
The final tile can be shorter. Operands remain BF16. This bounds the length of
each GEMM accumulation without changing logits, weight gradients or collective
shapes. Small contractions need only one GEMM.

The same algorithm runs at TP1 when the option is enabled: keeping TP1's long
contraction would retain its larger accumulation error. Numerical comparisons
must rerun both variants from the same checkpoint. Fixed-size tiles reduce
error but do not guarantee bitwise equality across arbitrary TP partitions.
The tradeoff is extra GEMM launches, contiguous operand slices and FP32 output
additions; this opt-in correctness path does not promise unchanged throughput.

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

## FP32 normalization parameters and gradients

`TransformerConfig.normalization_in_fp32` (CLI `--normalization-in-fp32`)
addresses a separate rounding boundary: TE returns norm parameter gradients in
the parameter dtype. A BF16 norm rounds each local sequence/head-shard partial
before MCore accumulates gradients or reduces sequence-parallel parameters.
FP32 DDP gradient buffers cannot recover that lost precision.

When enabled, `TENorm` allocates gamma/beta in FP32 and evaluates normalization
in FP32, then casts the output to the input activation dtype. Native AMP is
disabled inside this norm only. Fused `TELayerNormColumnParallelLinear` and
`TERMSNormDuplicatedLinear` request the corresponding explicit TE API: norm
arithmetic and gamma/beta gradients remain FP32, while the linear activation
boundary and linear weights retain their BF16 paths. The completed linear
input gradient is BF16 before conversion to FP32 for norm backward.

Norm parameters use `mark_keep_in_fp32` so `Float16Module` preserves their
dtype. Parameter names, checkpoint keys, and sequence-parallel metadata remain
unchanged. Existing BF16 checkpoint values load into FP32 norm parameters.
This option is disabled by default. Enabling it changes norm arithmetic and
parameter updates at TP1 too, so numerical comparisons must rerun both TP1 and
TP2 from the same checkpoint. It does not promise bitwise training equality.

Supported norms are LayerNorm and RMSNorm in ordinary BF16 TE training, without
FP8/FP4, TP communication overlap, symmetric all-reduce, or fused residual
RMSNorm. A TE build exposing `normalization_in_fp32` on `LayerNormLinear` is
required for fused projections. TE validates FP32 norm parameters separately
from linear parameters; the ordinary parameter dtype checks remain active.
