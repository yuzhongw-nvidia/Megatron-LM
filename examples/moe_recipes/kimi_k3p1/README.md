# Kimi-K3.1 Blackwell recipes

Kimi-K3.1 replaces the full-attention MLA layers in Kimi-K3 with top-k sparse
attention (DSA). The full decoder has 69 KDA and 24 DSA blocks, plus one DSA/MoE
MTP depth. The 9-block decoder has seven KDA and two DSA blocks, plus MTP.
The entry point for both is `pretrain_hybrid.py`.

## GTP at 256K context

The [256-GPU GTP recipe](gb300/mxfp8_THD256K_256GPU_GTP64EP64CP128_muon.yaml)
uses 64 dense and four expert weight shards with an effective context-parallel
group of 128, with no MTP layer. K3.1 DSA uses all-gather for context parallelism
and keeps its absorbed K/V up-projection whole on each tensor-parallel rank. The original K3
MLA recipe uses a hierarchical latent exchange. To launch K3.1, set
`RECIPE=examples/moe_recipes/kimi_k3p1/gb300/mxfp8_THD256K_256GPU_GTP64EP64CP128_muon.yaml`
when submitting the [K3 GTP launcher](../kimi_k3/GTP_1M_TRAINING_GUIDE.md).

## Full-model throughput on GB300

| Packed THD length | GPUs | TP/PP/EP/CP | GBS | Tokens per CP rank | Median step | Median TFLOP/s/GPU | Status |
|---|---:|---|---:|---:|---:|---:|---|
| [64K](gb300/mxfp8_THD64K_256GPU_TP1PP4EP64CP16_muon.yaml) | 256 | 1/4/64/16 | 256 | 4,096 | 69.4620 s | 638.0 | Completed 50 steps |
| [256K](gb300/mxfp8_THD256K_256GPU_TP1PP4EP64CP64_muon.yaml) | 256 | 1/4/64/64 | 64 | 4,096 | — | — | Cancelled before allocation, zero steps |

The 64K run on November 25 used Megatron-LM
`46467f8deca440c724653e1b17443038e5f3e1a8` with the improved cuDNN KDA
implementation, frontend 1.33.0 and CUTLASS DSL 4.8.1. The table gives
separate medians of step time and logged throughput over iterations 6–50.
Peak allocated device memory was 218.0 GiB. Its mock documents each fill one
64K sequence. Both full-model recipes specify 16,777,216 tokens per optimizer
step, MBS1, PP4, EP64 and ETP1. The 256K YAML records the submitted CP64
configuration. There is no completed full-model 256K timing for this release.

## 9-block proxy on GB300

The [64K decoder proxy](../kimi_k3p1_proxy_9layer/gb300/mxfp8_THD64K_64GPU_TP1PP1EP64CP8_muon.yaml)
uses 64 GPUs, TP1/PP1/EP64/CP8, MBS1/GBS256 and packed 32K–64K mock
documents. It has 8,192 tokens per CP rank. Medians below cover steps 4–20,
before profiler collection.

| Run | Date | Median step | Median TFLOP/s/GPU |
|---|---|---:|---:|
| PyTorch profiler recipe, 25 steps | 2026-11-24 | 19.8631 s | 733.9 |
| Nsight Systems capture, through step 22 | 2026-11-25 | 19.9400 s | 734.6 |

Both runs executed `dd62d69aa50e` with the optimized KDA module applied;
the module matched the replay revision `46467f8deca4`. The older 20-step
proxy comparison below used the earlier frontend and KDA configuration.

## Attention and data settings

All three recipes use `kda_backend: cudnn` with chunkwise, contiguous CP and
`dsa_kernel_backend: cudnn` with latent `all_gather`. TP1 keeps all 96 query
heads on one rank. DSA uses 32 indexer heads of dimension 128, top-k 2048,
per-token selection, zero indexer loss and no Hadamard rotation. The YAML
passes `--no-dsa-indexer-rotate-activation` explicitly. The hybrid pattern
uses `D` for DSA and `/DE` for its MTP depth. The rest of the training stack
uses the K3 MoK, MXFP8 and Muon settings. [K3 runtime and launch guide](../kimi_k3/README.md)

## DSA kernel backends and head counts

`dsa_kernel_backend` selects the fused sparse-attention kernels. Without a
fused kernel DSA falls back to a reference path that materializes the dense
query-key scores, which does not fit in memory at these sequence lengths. What
matters is the number of attention heads on one rank, 96 divided by the TP
size:

| TP | Heads per rank | `cudnn` | `tilelang` |
|---:|---:|---|---|
| 1 | 96 | Yes, with the frontend 1.33.0 in the [image](#image) | Yes |
| 2 | 48 | No | Yes (kernel tests only) |
| 3 | 32 | Yes, padded to 64 heads (not run for this model) | No |
| 4 | 24 | No | No |

- `cudnn` runs 64 heads, multiples of 128 and divisors of 64 (padded to 64)
  through FlashMLA, and 96 heads through cudnn-frontend's own forward. With an
  indexer loss it does not take 96 heads.
- `tilelang` accepts any head count from 16 up. Its kernels tile 24 and 32
  heads 32 wide, and that tiling fails the reference-parity test once top-k
  spans more than one 64-key block.
- TileLang is not part of the image. The historical TileLang measurement below added
  `tilelang==0.1.12` on top of it and set `dsa_kernel_backend: tilelang`.

A TP4 layout would leave 24 heads on a rank, which neither backend
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

The shared parent pins cuDNN frontend 1.33.0, CUTLASS DSL 4.8.1 and the
FA4/Quack compatibility changes. Its H96 sparse forward serves the published TP1
recipes. The derived public Dockerfile also builds FlashMLA at
`b7643bd54521f563b839b98289b5cd048c062ba2` for the head counts that dispatch
to FlashMLA. The November measurements used the same recorded ARM64 image digest
as K3. The public Dockerfiles have not been rebuilt for this update.
[Container versions and measured image](../kimi_k3/README.md#build-the-container)

## Earlier DSA backend comparison

The earlier proxy used FLA KDA and cuDNN frontend `e55745ffce93` on 64 GB300 GPUs for 20 optimizer steps,
median over steps 4 to 20. Each step holds 256 mock sequences of 32768 to
65536 tokens, about 10.9 million tokens.

| `dsa_kernel_backend` | Step time | Throughput | Peak allocated | Peak device memory |
|---|---:|---:|---:|---:|
| `cudnn` | 20.8 s | 702.6 TFLOP/s/GPU | 113.4 GiB | 129.2 GiB |
| `tilelang` | 33.3 s | 437.9 TFLOP/s/GPU | 113.4 GiB | 129.8 GiB |

Both backends ran back to back on the same nodes. Their training loss agrees
within 1.7e-3 over the 20 steps. The TileLang throughput is the same FLOPs per
step divided by its step time; that run predates the DSA accounting on the
hybrid FLOPs path and was not repeated.

At the kernel level, sparse attention for one rank of this configuration (8192
queries, 65536 keys, top-k 2048, 96 heads) takes 27.0 ms forward plus backward
on `cudnn` and 112.9 ms on `tilelang`.

The reported throughput uses mock data. The earlier numerical checks of the DSA
kernels do not cover end-to-end real-data convergence or parity of the contiguous
and zigzag DSA layouts across CP ranks.
