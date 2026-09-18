# Kimi-K3 Blackwell recipes

These recipes describe the Kimi-K3 text backbone and a nine-decoder-block
proxy, including model parameters, runtime arguments and a shared ARM64
container build. Use the MCore checkout containing these recipes and
`pretrain_hybrid.py` as the training entry point.

**Status: both 512K configurations completed 20-step mock-data runs on GB300
with NCCL 2.32.3 and HybridEP 10d4dd7 on 2026-10-06. The full model achieved
643 TFLOP/s/GPU on 256 GPUs and the 9L proxy 843 TFLOP/s/GPU on 64 GPUs.
These recipes replace the earlier 256K configurations.**

## Configurations

| Recipe | GPUs | TP/PP/EP/CP/ETP | MBS/GBS/SL | Decoder blocks / HybridModel entries | AttnRes block entries |
|---|---:|---|---|---|---:|
| [Kimi-K3 GB300, packed THD 512K](gb300/mxfp8_THD512K_256GPU_TP4PP4EP64CP16_muon.yaml) | 256 | 4/4/64/16/1 | 1/32/524288 | 93 / 186 | 24 |
| [Kimi-K3 9L proxy GB300, packed THD 512K](../kimi_k3_proxy_9layer/gb300/mxfp8_THD512K_64GPU_TP4PP1EP64CP16_muon.yaml) | 64 | 4/1/64/16/1 | 1/32/524288 | 9 / 18 | 2 |

Both recipes have dense DP=1 (TP4 x CP16 = 64 GPUs per pipeline stage),
expert DP=1 and 32 microbatches per optimizer step. Each step processes
16,777,216 tokens (32 x 524288). Counts in the table exclude the one MTP
prediction depth, which adds an MLA+MoE block (`/+E`) to both configurations.

- Model dimensions: hidden size 7168, 96 attention heads, 896 routed experts,
  top-16 routing, MoE latent size 3584 and expert FFN size 3072.
- The proxy retains the full model's widths and expert count. Its main pattern
  contains seven KDA blocks, two MLA blocks, one dense MLP and eight MoE MLPs.
- The full model uses four pipeline stages with 24/24/24/21 decoder blocks
  (48/48/48/42 HybridModel entries); MTP runs on the last stage. The pipes in
  `hybrid_layer_pattern` define this layout.
- `attn_res_block_layers` counts individual HybridModel entries. An attention
  operation and its following MLP count as two entries.
- Both use MXFP8 parameter gather, per-head Muon in duplicated TP mode,
  compiled AttnRes, pre-GDR fusion, gated output-norm fusion and HybridEP.
- Selective recomputation covers `layernorm`, `gdn`, `moe` and `mlp`.
  Fine-grained activation offloading covers `core_attn` and `attn_proj`, with
  `NVTE_CPU_OFFLOAD_V1=1` selecting TE's activation-only CPU offload path.
  Attention/MLP norm offloading and MLA up-projection recomputation are not
  enabled in these measured configurations.
- Chunked optimizer-state offload moves states to CPU between GPU optimizer
  computations. The chunk size is 256 MiB and the offload fraction is 1.0.
- Every mock document contains exactly 524288 tokens. The `dp_balanced`
  scheduler packs one document per buffer and pads to
  `max_seqlen_per_dp_cp_rank` x CP = 32768 x 16 = 524288 tokens, giving each
  microbatch a static shape. The HybridEP chunk size is 128 tokens.
- The packed buffer uses a contiguous CP partition, matching KDA's native
  chunkwise layout. MLA converts to zigzag internally. The sequence-parallel
  conversion combines TP gathering and CP redistribution into one TP x CP
  all-to-all, avoiding repeated conversions around the KDA layers.
- The recipes use mock data with forced balanced routing, a 50-step cosine
  schedule and a peak learning rate of 1e-5. MTP and per-head Muon remain
  experimental settings. Mock-data loss is not a convergence result.

### NCCL memory settings

Both recipes set `NCCL_LL128_BUFFSIZE=614400`, `NCCL_PROTO=^LL128` and
`NCCL_BUFFSIZE=1048576` to reduce communication-buffer memory. The shared
[NCCL communicator configuration](nccl/tp_dp_cp_max_ctas16.yaml) sets
`max_ctas: 16` only for `tp_dp_cp`, the 64-rank TP x CP all-to-all group.
Other process groups retain NCCL's default communicator settings.

`nccl_communicator_config_path` is relative to the Megatron-LM repository
root. Run the launch from that directory on every node. These settings were
measured with the TP x CP group inside one GB300 NVLink domain; reassess them
when changing the parallel layout or interconnect.

## Measured throughput

Both configurations completed 20 iterations on 2026-10-06 with the settings
above and the refreshed NCCL 2.32.3 / HybridEP 10d4dd7 runtime. The full model
used 256 GB300 GPUs (64 nodes, four GPUs each), and the proxy used 64 GB300
GPUs (16 nodes). The measurements override the recipe with `--train-iters 20
--eval-iters 1`; no profiler was enabled. The first iteration is excluded
from the steady-state statistics.

| Metric | Full model, 256 GPUs | 9L proxy, 64 GPUs |
|---|---:|---:|
| Mean iteration time (steps 2-20) | 188.0 s | 70.15 s |
| Iteration-time range (steps 2-20) | 187.2-189.3 s | 69.7-70.9 s |
| Mean throughput (steps 2-20) | 643.4 TFLOP/s/GPU | 843.0 TFLOP/s/GPU |
| Median throughput (steps 2-20) | 643.7 TFLOP/s/GPU | 844.7 TFLOP/s/GPU |
| Tokens per iteration | 16.8M (32 x 524288) | 16.8M (32 x 524288) |
| First iteration | 1645 s | 470 s |

Both runs completed without errors. The first iteration includes mock-data
construction, compilation and communication setup. The results include one
MTP depth. The proxy uses the sequence-parallel MTP padding-mask fix; the
full model uses the same relevant training settings with MTP on its final
pipeline stage. GPU memory savings come from activation recomputation,
attention offloading and smaller NCCL buffers.

Separate 64K real-data continuation checks of the CP-layout and memory
changes kept the maximum LM-loss difference below 1e-3 over 32 steps.
The 512K throughput runs use mock data and do not establish long-context
convergence. Full-model 1M-context training on 256 GPUs remains future work.

## Build the container

Both YAMLs reference the shared [Dockerfile](Dockerfile) through
`DEPENDENCIES.dockerfile_path`, relative to the Megatron-LM repository root.
One image serves both recipes. Build on an ARM64 builder with Docker/BuildKit:

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
| Transformers | `5.16.1` |
| NVIDIA Resiliency Extension | `0.6.0` |
| Emerging-Optimizers | `a44d1f83a4950b445f2a77ca71177c5f033d45d5` |
| causal-conv1d | `4f6ae4e26ae5fe8af9372f8d312ab25cc4595223` (1.6.2.post1) |
| HybridEP | `10d4dd7377d5bce900fbb4b80cce863764892b95` (hybrid-ep branch) |
| CUTLASS DSL | `4.5.2` (CUDA 13) |
| Apache TVM FFI | `0.1.11` |

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

Both recipes enable `gdn_pre_gated_delta_rule_fusion`, which requires the
C++ convolution backward from causal-conv1d. The Dockerfile builds it from
the pinned source revision against the NGC CUDA/PyTorch installation.
TE targets SM100a and SM103a; its optional NCCL EP extension stays disabled
because these recipes use HybridEP.

A `/opt/venv` environment inherits NGC's system packages and installs recipe
dependencies locally. The venv is first on `PATH`; `pip check` remains a
build requirement, and the build includes `pytest` for NGC's triton-kernels.
Run with the image's default Python environment.

The original public ARM64 build and dependency checks passed in September.
The October measurements used the existing benchmark image after its NCCL
upgrade; the updated public Dockerfile has not been rebuilt as part of this
recipe synchronization.

## Prepare the launch

The YAML has the recipe schema `DEPENDENCIES` / `ENV_VARS` / `ARGS`; it is
not the training entry point's `--yaml-cfg` schema. Export `ENV_VARS` and
translate `ARGS` into command-line flags. Lists become multiple arguments;
`true` becomes a flag; `false` and `null` are omitted.

The following example writes a launch script from either recipe. Run it
inside the built container with the checkout as the working directory,
Python/PyYAML available and `OUTPUT_PATH` set to a writable, container-visible
directory. The checkpoint path can be supplied later with `--load`.

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
in the compiled AttnRes path. Keep it when launching either configuration.
