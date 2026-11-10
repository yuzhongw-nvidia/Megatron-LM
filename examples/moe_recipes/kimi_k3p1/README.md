# Kimi-K3.1 Blackwell recipes

Kimi-K3.1 is Kimi-K3 with one structural change: the full-attention layers
(every fourth decoder block and the last one) use DeepSeek-style Sparse
Attention (DSA) with a top-k indexer instead of dense MLA. In the hybrid layer
pattern the `+` (MLA) entries become `D`. The KDA layers, the latent MoE,
SitU-GLU and the attention residuals are those of Kimi-K3.

This page covers what is specific to Kimi-K3.1. The launch procedure, the
environment variables and the Kimi-K3 measurements are in the
[Kimi-K3 guide](../kimi_k3/README.md).

**Status: the 9-layer proxy recipe is the measured configuration; the full
model recipe has not been run.** Both use TP1 instead of Kimi-K3's TP4, because
the fused DSA kernels need all 96 attention heads on one rank (see
[head counts](#dsa-kernel-backends-and-head-counts)), and both keep 8192 tokens
per rank. The full recipe applies the proxy's layout to the 93-block model and
passes argument and config validation; whether it fits in memory is untested.

## Configurations

| Recipe | GPUs | TP/PP/EP/CP | MBS/GBS/SL | Decoder blocks | DSA layers | Run |
|---|---:|---|---|---:|---:|---|
| [Kimi-K3.1, packed THD 256K](gb300/mxfp8_THD256K_256GPU_TP1PP4EP64CP32_muon.yaml) | 256 | 1/4/64/32 | 1/64/262144 | 93 | 24 | No |
| [Kimi-K3.1 9L proxy, packed THD 64K](../kimi_k3p1_proxy_9layer/gb300/mxfp8_THD64K_64GPU_TP1PP1EP64CP8_muon.yaml) | 64 | 1/1/64/8 | 1/256/65536 | 9 | 2 | Yes |

Each configuration has one more DSA layer in its MTP depth (`/DE`). Both
recipes use mock data.

## Settings that differ from Kimi-K3

- `experimental_attention_variant: dsa`, with `D` in place of `+` in
  `hybrid_layer_pattern`.
- DSA indexer from the Kimi-K3.1 model config: 32 heads of dimension 128,
  top-k 2048, per-token selection without key pooling, an index computed by
  every DSA layer, no indexer loss, and no Hadamard rotation
  (`dsa_indexer_rotate_activation: false`; the option defaults to true).
- TP1 without sequence parallelism, and a CP size that keeps 8192 tokens per
  rank: CP8 at 64K and CP32 at 256K.
- `cp_partition_mode: contiguous` for the whole stack. It is the native layout
  of the KDA layers; the DSA layers convert their down-projected latents to the
  zigzag layout internally and convert the attention output back.
  `cp_comm_type: all_gather`, because DSA gathers the latent keys across the
  CP ranks.
- `kv_channels` is not set. With 96 heads and a hidden size of 7168 the heads
  do not divide the hidden size, and for MLA models the value is derived from
  `qk_head_dim + qk_pos_emb_head_dim`.
- The MLA latent CP settings of Kimi-K3 (`mla_latent_cp`,
  `hierarchical_context_parallel_sizes`) are dropped; DSA does not use them.

## DSA kernel backends and head counts

`dsa_kernel_backend` selects the fused sparse-attention kernels. Without a
fused kernel DSA falls back to a reference path that materializes the dense
query-key scores, which does not fit in memory at these sequence lengths. What
matters is the number of attention heads on one rank, 96 divided by the TP
size:

| TP | Heads per rank | `cudnn` | `tilelang` |
|---:|---:|---|---|
| 1 | 96 | Yes, with the cudnn-frontend pinned in the [image](#image) | Yes |
| 2 | 48 | No | Yes (kernel tests only) |
| 3 | 32 | Yes, padded to 64 heads (not run for this model) | No |
| 4 | 24 | No | No |

- `cudnn` runs 64 heads, multiples of 128 and divisors of 64 (padded to 64)
  through FlashMLA, and 96 heads through cudnn-frontend's own forward. With an
  indexer loss it does not take 96 heads.
- `tilelang` accepts any head count from 16 up. Its kernels tile 24 and 32
  heads 32 wide, and that tiling fails the reference-parity test once top-k
  spans more than one 64-key block.
- TileLang is not part of the image. The TileLang measurement below added
  `tilelang==0.1.12` on top of it and set `dsa_kernel_backend: tilelang`.

Kimi-K3's TP4 layout would leave 24 heads on a rank, which neither backend
runs. That is why these recipes use TP1.

## Image

The recipes use their own [Dockerfile](Dockerfile), built on top of the
Kimi-K3 image:

```bash
# Run from the Megatron-LM repository root.
docker build --platform=linux/arm64 \
    -f examples/moe_recipes/kimi_k3/Dockerfile -t megatron-kimi-k3:blackwell .
docker build --platform=linux/arm64 \
    --build-arg KIMI_K3_IMAGE=megatron-kimi-k3:blackwell \
    -f examples/moe_recipes/kimi_k3p1/Dockerfile -t megatron-kimi-k3p1:blackwell .
```

| Component | Version / revision |
|---|---|
| Base | the Kimi-K3 image |
| cuDNN frontend | `e55745ffce9371f082b9dad452f74d5807cdb0e6`, the head of NVIDIA/cudnn-frontend#2474 (temporary pin; replace with the release that contains it) |
| CUTLASS DSL | `4.8.1` (CUDA 13), required by that cuDNN frontend |
| FlashMLA | `b7643bd54521f563b839b98289b5cd048c062ba2` (`nv_dev`) |

The image is separate from the Kimi-K3 one because CUTLASS DSL 4.8 removed
`cute.make_fragment`. FlashAttention-4 4.0.0b11, which Kimi-K3's MLA latent CP
requires, and Megatron's fused linear cross-entropy kernels still call it.
Kimi-K3.1 runs neither. DSL 4.8 also moved `ThrMma` and `ThrCopy` out of
`cutlass.cute.core`; the Dockerfile rewrites those references in
FlashAttention-4 and quack-kernels so that they still import.

The Dockerfile has not been built with Docker as a whole. Its CUTLASS DSL,
cuDNN frontend and source-patch steps were run as written on top of an image
that already held FlashMLA at the pinned commit, and the measurements below ran
in the result.

## Measured throughput

The 9-layer proxy recipe as written, on 64 GB300 GPUs: 20 optimizer steps,
median over steps 4 to 20. Each step holds 256 mock sequences of 32768 to
65536 tokens, about 10.9 million tokens.

| `dsa_kernel_backend` | Step time | Throughput | Peak allocated | Peak device memory |
|---|---:|---:|---:|---:|
| `cudnn` (the recipe) | 20.8 s | 702.6 TFLOP/s/GPU | 113.4 GiB | 129.2 GiB |
| `tilelang` | 33.3 s | 437.9 TFLOP/s/GPU | 113.4 GiB | 129.8 GiB |

Both backends ran back to back on the same nodes. Their training loss agrees
within 1.7e-3 over the 20 steps. The TileLang throughput is the same FLOPs per
step divided by its step time; that run predates the DSA accounting on the
hybrid FLOPs path and was not repeated.

At the kernel level, sparse attention for one rank of this configuration (8192
queries, 65536 keys, top-k 2048, 96 heads) takes 27.0 ms forward plus backward
on `cudnn` and 112.9 ms on `tilelang`.

## Not validated

- The full 93-block recipe: it has not been run, and its memory use with TP1
  is unknown.
- Real data. The measured runs used mock data.
- Numerical parity of the contiguous layout against zigzag at CP greater than 1
  for the DSA layers.
