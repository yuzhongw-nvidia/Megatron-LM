# Kimi-K3 Blackwell recipes

These recipes describe the Kimi-K3 text backbone and a nine-decoder-block
proxy, including model parameters, runtime arguments and a shared ARM64
container build. Use the MCore checkout containing these recipes and
`pretrain_hybrid.py` as the training entry point.

**Status: MLA latent CP is enabled in the 512K mock-data and 64K real-data
recipes. Recorded 512K runs on 2026-10-12 achieved about 765 TFLOP/s/GPU for
the full model on 256 GB300 GPUs and 902 TFLOP/s/GPU for the 9L proxy on 64
GPUs. A separate 9L 64K real-data check qualified the pure-a2a attention
path; see the validation scope below.**

## Known limitations on this main-based branch

- **Context-parallel layout.** These recipes drive CP through the single-knob
  `cp_partition_mode` (the attention module converts its own input to zigzag)
  and deliberately leave `linear_cp_layout` unset, which keeps the legacy path
  and lets Attention Residuals run with CP>1. Do not set `linear_cp_layout` /
  `attention_cp_layout` here — doing so opts into the per-layer CP layout
  manager, which is mutually exclusive with Attention Residuals.

## Configurations

| Recipe | GPUs | TP/PP/EP/CP/ETP | MBS/GBS/SL | MLA CP communication | Decoder blocks / HybridModel entries | AttnRes block entries |
|---|---:|---|---|---|---|---:|
| [Kimi-K3 GB300, packed THD 512K](gb300/mxfp8_THD512K_256GPU_TP4PP4EP64CP16_muon.yaml) | 256 | 4/4/64/16/1 | 1/32/524288 | a2a8 + p2p2 | 93 / 186 | 24 |
| [Kimi-K3 9L proxy GB300, packed THD 512K](../kimi_k3_proxy_9layer/gb300/mxfp8_THD512K_64GPU_TP4PP1EP64CP16_muon.yaml) | 64 | 4/1/64/16/1 | 1/32/524288 | a2a8 + p2p2 | 9 / 18 | 2 |
| [Kimi-K3 GB300, packed THD 64K](gb300/mxfp8_THD64K_256GPU_TP4PP4EP64CP2_muon.yaml) | 256 | 4/4/64/2/1 | 1/256/65536 | a2a2 | 93 / 186 | 24 |
| [Kimi-K3 9L proxy GB300, packed THD 64K](../kimi_k3_proxy_9layer/gb300/mxfp8_THD64K_64GPU_TP4PP1EP64CP2_muon.yaml) | 64 | 4/1/64/2/1 | 1/256/65536 | a2a2 | 9 / 18 | 2 |

The 512K recipes have dense DP=1 (TP4 x CP16 = 64 GPUs per pipeline stage);
the 64K recipes have dense DP=8 (TP4 x CP2 x DP8 = 64). All have expert DP=1,
32 microbatches per DP rank and 16,777,216 tokens per optimizer step. Counts
in the table exclude the one MTP prediction depth, which adds an MLA+MoE
block (`/+E`) to every configuration.

- Model dimensions: hidden size 7168, 96 attention heads, 896 routed experts,
  top-16 routing, MoE latent size 3584 and expert FFN size 3072.
- The proxy retains the full model's widths and expert count. Its main pattern
  contains seven KDA blocks, two MLA blocks, one dense MLP and eight MoE MLPs.
- The full model uses four pipeline stages with 24/24/24/21 decoder blocks
  (48/48/48/42 HybridModel entries); MTP runs on the last stage. The pipes in
  `hybrid_layer_pattern` define this layout.
- `attn_res_block_layers` counts individual HybridModel entries. An attention
  operation and its following MLP count as two entries.
- All use MXFP8 parameter gather, per-head Muon in duplicated TP mode,
  compiled AttnRes, pre-GDR fusion, gated output-norm fusion and HybridEP.
- Selective recomputation covers `layernorm`, `gdn`, `moe` and `mlp`.
  Fine-grained activation offloading covers `core_attn` and `attn_proj`, with
  `NVTE_CPU_OFFLOAD_V1=1` selecting TE's activation-only CPU offload path.
  MLA up-projection and core-attention recomputation must remain disabled
  with these latent CP configurations.
- Chunked optimizer-state offload moves states to CPU between GPU optimizer
  computations. The chunk size is 256 MiB and the offload fraction is 1.0.
- The `dp_balanced` scheduler pads each buffer to
  `max_seqlen_per_dp_cp_rank` x CP: 32768 x 16 = 524288 tokens at 512K, or
  32768 x 2 = 65536 at 64K. The HybridEP chunk size is 128 tokens.
- The packed buffer uses a contiguous CP partition, matching KDA's native
  chunkwise layout. MLA handles the attention layout internally; the
  sequence-parallel conversion combines TP gathering and CP redistribution
  into one TP x CP all-to-all.
- `mla_latent_cp: true` exchanges compressed latent KV and reconstructs K/V
  for each attention step. At 512K, `cp_comm_type: a2a+p2p` and
  `hierarchical_context_parallel_sizes: [8, 2]` split the 24 heads per TP rank
  across eight a2a ranks, leaving three heads per rank and a two-step latent
  ring. The forward ring uses one stream (`mla_latent_cp_two_stream` is off).
  At 64K, `cp_comm_type: a2a` uses two CP ranks, 12 heads per rank and one
  causal attention step without a ring. Both use `attention_backend: flash`
  with FlashAttention-4 4.0.0b11.
- The 512K recipes use fixed-length mock documents and forced balanced
  routing. The 64K recipes use LongAlign-10k and Quantile Balancing without
  forced routing. All use a 50-step cosine schedule with a peak learning
  rate of 1e-5. MTP and per-head Muon remain experimental settings.

### NCCL memory settings

All recipes set `NCCL_LL128_BUFFSIZE=1048576` and `NCCL_PROTO=^LL128` to
reduce communication-buffer memory; the 614400-byte LL128 buffer used before
2026-10-08 breaks the RoCE transport's connection setup on GB300 NVL72
systems. The proxy keeps `NCCL_BUFFSIZE=1048576`; the 256-GPU pipeline recipe
uses `NCCL_BUFFSIZE=2097152`, because 1 MiB SIMPLE buffers halve the
inter-rack pipeline point-to-point bandwidth. All recipes also set
`strict_runtime_validation_frequency: once_per_microbatch`, which runs the
hybrid layers' strict runtime input checks on the first instance of each
layer type per microbatch instead of in every layer (the per-layer checks
cost about 3.8 ms of host work and four stream synchronizations per KDA
forward, 0.5% of the 512K step). The shared
[NCCL communicator configuration](nccl/tp_dp_cp_max_ctas16.yaml) sets
`max_ctas: 16` only for `tp_dp_cp`, the TP x CP all-to-all group
(64 ranks at 512K, eight at 64K).
Other process groups retain NCCL's default communicator settings.

`nccl_communicator_config_path` is relative to the Megatron-LM repository
root. Run the launch from that directory on every node. These settings were
measured with the TP x CP group inside one GB300 NVLink domain; reassess them
when changing the parallel layout or interconnect.

## Measured throughput

The recorded 512K latent CP runs on 2026-10-12 used NCCL 2.32.3, HybridEP
10d4dd7 with the RUNPATH fix, and FlashAttention-4 4.0.0b11. The full model
used 256 GB300 GPUs (64 nodes, four GPUs each) for 10 iterations; the proxy
used 64 GPUs (16 nodes) for 20. The measurements override the recipes with
`--train-iters N --eval-iters 1`. Nsight Systems profiled the last two
iterations, one rank per pipeline stage for the full model and rank 0 for
the proxy. The snapshot statistics below exclude the first iteration and
include the remaining profiled iterations.

| Metric | Full model, 256 GPUs | 9L proxy, 64 GPUs |
|---|---:|---:|
| Mean iteration time | 158.21 s (steps 2-10) | 65.59 s (steps 2-20) |
| Mean throughput | 764.5 TFLOP/s/GPU | 901.6 TFLOP/s/GPU |
| Median throughput | 767.5 TFLOP/s/GPU | 903.3 TFLOP/s/GPU |
| Tokens per iteration | 16.8M (32 x 524288) | 16.8M (32 x 524288) |
| First iteration | 2330 s | 526 s |
| Peak allocated GPU memory (rank 0) | 163.8 GB | 69.0 GB |

Both runs completed without errors and include one MTP depth. The first
iteration includes data construction, compilation and communication setup.
The preceding 2026-10-09 measurements with standard MLA CP on the fixed
container were 169.4 s / 714.0 TFLOP/s/GPU for the full model and
70.05 s / 844.2 TFLOP/s/GPU for the proxy. These are recorded experiments;
no additional training run was performed for this configuration update.

The 9L 64K pure-a2a latent CP path was checked separately on LongAlign-10k
for 32 continuation steps on 2026-10-13, against the same-code TE p2p/cuDNN
reference. Maximum absolute LM-loss difference was 2.69e-4, below the 1e-3
threshold; maximum MTP-loss difference was 1.58e-4. That check loaded the
same checkpoint with `--finetune`, disabled router load balancing, and used
the `alltoall` dispatcher. It validates the attention change under those
settings, not the complete HybridEP/Quantile Balancing recipe. No full-model
64K latent CP result is claimed here.

The 512K throughput runs use mock data and do not establish long-context
convergence. Full-model 1M-context training on 256 GPUs remains future work.

## Build the container

All four YAMLs reference the shared [Dockerfile](Dockerfile) through
`DEPENDENCIES.dockerfile_path`, relative to the Megatron-LM repository root.
One image serves all four recipes. Build on an ARM64 builder with Docker/BuildKit:

```bash
# Run from the Megatron-LM repository root.
docker build --platform=linux/arm64 \
    -f examples/moe_recipes/kimi_k3/Dockerfile -t megatron-kimi-k3:blackwell .
```

The public build uses the following fixed components:

| Component | Version / revision |
|---|---|
| NVIDIA PyTorch base | `nvcr.io/nvidia/pytorch:26.04-py3` |
| NCCL runtime and headers | `2.32.3-1+cuda13.4` (CUDA toolkit and PyTorch unchanged) |
| TransformerEngine | `0a2ebe942b9f6a66aa0292502966e73114a7b912` |
| cuDNN frontend | `5ba9432e652e0b4df3aaca85cf3f95fe6bb7375a` |
| Flash Linear Attention / fla-core | `0.5.2` |
| FlashAttention-4 | `4.0.0b11` (required by the MLA latent CP split forward/backward API) |
| Transformers | `5.16.1` |
| NVIDIA Resiliency Extension | `0.6.0` |
| Emerging-Optimizers | `a44d1f83a4950b445f2a77ca71177c5f033d45d5` |
| causal-conv1d | `4f6ae4e26ae5fe8af9372f8d312ab25cc4595223` (1.6.2.post1) |
| HybridEP | `10d4dd7377d5bce900fbb4b80cce863764892b95` (hybrid-ep branch) |
| CUTLASS DSL | `4.5.2` (CUDA 13) |
| Apache TVM FFI | `0.1.11` |

The Dockerfile explicitly installs `flash-attn-4==4.0.0b11` and checks the
installed version. MLA latent CP uses its split forward/backward API and
rejects other versions. This matches the recorded benchmark image.

NCCL's runtime and development packages are pinned together. The Dockerfile
checks the library actually resolved by the dynamic linker with
`ncclGetVersion`; PyTorch's build-time NCCL version alone does not identify
the loaded runtime. The refreshed benchmark image passed single-node and
two-node GPU collective checks with NCCL 2.32.3.

HybridEP revisions before 17cfb81 can retain a recycled pinned-host token
count during blocking permute/unpermute, causing intermittent illegal memory
accesses at long sequence lengths. The pinned 10d4dd7 revision fixes that
lifetime issue. The public build sets `HYBRID_EP_MULTINODE=0`; place each EP64
group within one NVLink domain. The benchmark image was built with multinode
support enabled but used the same within-domain EP placement. Cross-domain
EP requires a suitable communication build and a separate placement review.

All recipes enable `gdn_pre_gated_delta_rule_fusion`, which requires the
C++ convolution backward from causal-conv1d. The Dockerfile builds it from
the pinned source revision against the NGC CUDA/PyTorch installation.
TE targets SM100a and SM103a; its optional NCCL EP extension stays disabled
because these recipes use HybridEP.

A `/opt/venv` environment inherits NGC's system packages and installs recipe
dependencies locally. The venv is first on `PATH`; `pip check` remains a
build requirement, and the build includes `pytest` for NGC's triton-kernels.
Run with the image's default Python environment.

The hybrid-ep build links libibverbs / libmlx5 from the NGC base image's
source build of rdma-core (`/workspace/rdma-core/build/lib`) and records that
directory as the extension's RUNPATH. Megatron imports `deep_ep` at module
level before NCCL creates its first communicator, so that rdma-core loads
first, NCCL's network plugin then loads the system rdma-core as well, its devx
HCA query fails and NCCL silently falls back to NET/Socket for every transfer
that leaves the NVLink domain (pipeline parallelism across racks ran at about
3 GB/s). The Dockerfile strips the RUNPATH after the HybridEP install and
asserts that no `rdma-core` entry remains. The benchmark image received the
same `patchelf --remove-rpath` step on 2026-10-08 (marker
`/etc/kimi_k3_container_patches.txt`); the 2026-10-09 measurements used that
image.

The original public ARM64 build and dependency checks passed in September.
The October measurements used the existing benchmark image after its NCCL
upgrade and the RUNPATH patch, with FlashAttention-4 4.0.0b11 installed. The
public Dockerfile, including the explicit FA4 pin, has not been rebuilt from
scratch as part of this recipe synchronization.

## Prepare the launch

The YAML has the recipe schema `DEPENDENCIES` / `ENV_VARS` / `ARGS`; it is
not the training entry point's `--yaml-cfg` schema. Export `ENV_VARS` and
translate `ARGS` into command-line flags. Lists become multiple arguments;
`true` becomes a flag; `false` and `null` are omitted.

The following example writes a launch script from any recipe. Run it
inside the built container with the checkout as the working directory,
Python/PyYAML available and `OUTPUT_PATH` set to a writable, container-visible
directory. The 512K recipes use mock data and need no checkpoint. The 64K
recipes load
`zai-org/LongAlign-10k` with `SFTTokenizer`, use a training-only split, and
cache data under `${OUTPUT_PATH}/data_cache/longalign_10k`. For a continuation
check, supply a matching checkpoint with `--load CHECKPOINT --finetune`.
The recipes enable one MTP depth: a checkpoint without MTP weights requires
removing `mtp_num_layers` and the trailing `/+E` from `hybrid_layer_pattern`
in the YAML before rendering the launcher.

```bash
export RECIPE=examples/moe_recipes/kimi_k3_proxy_9layer/gb300/mxfp8_THD512K_64GPU_TP4PP1EP64CP16_muon.yaml
: "${OUTPUT_PATH:?Set OUTPUT_PATH to the run output directory}"
export OUTPUT_PATH
python - <<'PYTHON' > run-kimi-k3.sh
import os
import shlex
import yaml

with open(os.environ["RECIPE"]) as stream:
    recipe = yaml.safe_load(stream)
print("#!/usr/bin/env bash\nset -euo pipefail")
for key, value in recipe["ENV_VARS"].items():
    print(f"export {key}={shlex.quote(str(value))}")
args = []
for key, value in recipe["ARGS"].items():
    if value is None or value is False:
        continue
    args.append("--" + key.replace("_", "-"))
    if value is not True:
        values = value if isinstance(value, list) else [value]
        args.extend(os.path.expandvars(str(item)) for item in values)
launcher = (
    'exec python -m torch.distributed.run --nnodes="${NNODES:?}" '
    '--nproc-per-node="${GPUS_PER_NODE:-4}" --node-rank="${NODE_RANK:?}" '
    '--master-addr="${MASTER_ADDR:?}" --master-port="${MASTER_PORT:-29500}" '
    'pretrain_hybrid.py '
)
print(launcher + shlex.join(args) + ' "$@"')
PYTHON
```

The launcher uses `python -m torch.distributed.run` to keep the worker
processes in the image's default Python environment.

Launch `bash run-kimi-k3.sh` once per node through your scheduler, with the
checkout and output directory mounted at the same paths on every node.
Provide `NNODES`, `GPUS_PER_NODE`, `NODE_RANK`, `MASTER_ADDR` and
`MASTER_PORT` from the allocation. With four GPUs per node, use 64 nodes for
the full recipe and 16 nodes for the proxy. Each node must use the same
recipe, image and rendezvous address; `NODE_RANK` must be unique.

`DYNAMO_CACHE_SIZE_LIMIT=64` accommodates the different depth-source counts
in the compiled AttnRes path. Keep it when launching any of these configurations.
