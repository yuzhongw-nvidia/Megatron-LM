# KDA tensor-parallel projection precision

KDA uses a replicated low-rank F-decay down projection followed by a
column-parallel up projection. The down projection controls a recurrent decay
gate, so small changes in its BF16 output can grow across the sequence.

With sequence parallelism, evaluating the replicated down projection on each
rank's sequence slice changes the GEMM shape and its accumulation order. It also
splits the weight-gradient GEMM across ranks. Identical weights and token values
can therefore produce different rounded gates when the TP degree changes.

## Complete-sequence down projection

`KimiDeltaAttention._project_f_latent` gathers the sequence before `f_a_proj`,
then scatters the small latent output back to the sequence-parallel layout
expected by `f_b_proj`. Its autograd contract is:

- The input gather's backward splits the complete input gradient.
- The latent scatter's backward gathers the complete latent gradient.
- Each rank computes the complete replicated down-projection weight gradient.

Consequently `f_a_proj` parameters have `sequence_parallel=False`. Applying
another sequence-parallel parameter-gradient all-reduce would multiply this
gradient by TP. The weight remains replicated, with unchanged checkpoint keys,
shapes, and Muon metadata. TP1 and execution without SP use the original GEMM.
This follows the complete-sequence replicated-gradient contract already used by
KDA's beta projection.

`test_kda_sequence_parallel_projection.py` checks independent complete-matrix
forward/input/weight gradients at TP1 and TP2, and exercises the actual KDA
forward path to verify that the projection receives the complete sequence.
These tests establish the projection's arithmetic and reduction contract;
end-to-end training trajectory acceptance is a separate requirement.

The FP32 projection cases use BF16-representable operands and FP64 matrix
references. This isolates the TP gradient contract from TE's independent TF32
GEMM policy and does not require an external TF32 environment override.
