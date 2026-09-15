# Hybrid Attention Residuals

`AttnResHybridLayer` treats each hybrid entry as one depth sublayer. It computes
the depth aggregation from completed block sources and, inside a block, the
current partial sum. The sublayer consumes that aggregation, while its
bias/dropout/residual operation adds the raw branch output to the partial sum.
At a block boundary the residual is zero. MTP entries retain their incoming
partial and attend over the completed trunk sources.

These are separate inputs: the aggregated activation is not the residual to
accumulate. Reconstructing a branch as `(aggregation + branch) - aggregation`
introduces rounding and cancellation in BF16, including in backward. Hybrid
entries therefore accept an explicit residual override. The TransformerLayer
path validates that exactly one attention or MLP sublayer is active, and retains
the normal normalization, selective recomputation, bias/dropout fusion, and
offloading paths. Mamba entries use the same residual ownership convention.
Parameter names and checkpoint layouts are unchanged.

The mathematical reference is Block AttnRes in
[Attention Residuals, Section 3.2](https://arxiv.org/pdf/2603.15031).
`tests/unit_tests/ssm/test_hybrid_attention_residual_precision.py` checks the
wrapper against independent PyTorch aggregation and accumulation equations,
using small branch values at Kimi-K3's hidden size. It covers FP32/BF16,
attention/MLP/Mamba entries, fused/unfused residual addition, and block/MTP
positions, including source and parameter gradients.
