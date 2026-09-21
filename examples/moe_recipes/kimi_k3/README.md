# Kimi-K3 Blackwell recipes

These recipes describe the Kimi-K3 text backbone and a nine-decoder-block
proxy, including their model parameters, runtime arguments and container
build instructions. Use the MCore checkout containing these recipes and
`pretrain_hybrid.py` as the training entry point.

**Status: the ARM64 container build and its `pip check` passed on 2026-09-17.
A single-node GB300 software-stack probe of that image passed on 2026-09-18
except for the missing causal-conv1d package, which the Dockerfile now installs;
with causal-conv1d added the probe passed in full and a 29-decoder-block
single-node proxy of this architecture trained for 50 iterations on the image.
The full recipe was measured on 256 GB300 GPUs (163 s per iteration, 498
TFLOP/s/GPU) and the 9L proxy on 64 GB300 GPUs (61 s per iteration, 661
TFLOP/s/GPU); see below.**

## Configurations

| Recipe | GPUs | TP/PP/EP/CP/ETP | MBS/GBS/SL | Decoder blocks / HybridModel entries | AttnRes block entries |
|---|---:|---|---|---|---:|
| [Kimi-K3 GB300, packed THD 256K](gb300/mxfp8_THD256K_256GPU_TP4PP4EP64CP16_muon.yaml) | 256 | 4/4/64/16/1 | 1/64/262144 | 93 / 186 | 24 |
| [Kimi-K3 9L proxy GB200, packed THD 256K](../kimi_k3_proxy_9layer/gb200/mxfp8_THD256K_64GPU_TP4PP1EP64CP16_muon.yaml) | 64 | 4/1/64/16/1 | 1/64/262144 | 9 / 18 | 2 |

Both recipes have dense DP=1 (TP4 x CP16 = 64 GPUs per pipeline stage; the
proxy is a single stage), expert DP=1 and 64 microbatches per optimizer
step. Counts in the table exclude the one MTP prediction depth, which adds an
MLA+MoE block (`/+E`) to both recipes.

- Model dimensions: hidden size 7168, 96 attention heads, 896 routed experts,
  top-16 routing, MoE latent size 3584 and expert FFN size 3072.
- The proxy keeps the full model's widths and expert count. Its main pattern
  contains seven KDA blocks, two MLA blocks, one dense MLP and eight MoE MLPs.
- The full model uses four pipeline stages with 24/24/24/21 decoder blocks
  (48/48/48/42 HybridModel entries); MTP runs on the last stage. The pipes in
  `hybrid_layer_pattern` define this layout.
- `attn_res_block_layers` counts individual HybridModel entries. An attention
  operation and its following MLP count as two entries.
- Both use MXFP8 parameter gather, per-head Muon, duplicated Muon TP mode,
  compiled AttnRes, pre-GDN fusion, HybridEP and selective recomputation of
  `gdn`, `moe`, `mlp` and `mla_up_proj`.
- Chunked optimizer-state offload moves states to CPU between GPU optimizer
  computations. The chunk size is 256 MiB and the offload fraction is 1.0.
- Both recipes train packed THD sequences of 262144 tokens: every mock
  document is exactly 262144 tokens, the `dp_balanced` scheduler packs one
  document per buffer, and buffers are padded to `max_seqlen_per_dp_cp_rank`
  x CP = 16 x 16384 = 262144 tokens so every micro-batch has a static shape.
  The buffer is split across the 16 CP ranks in zigzag order; KDA runs in
  chunkwise CP mode and the MLA layers use the fused attention backend. The
  HybridEP chunk size (`NUM_OF_TOKENS_PER_CHUNK_*`) is 128, which divides the
  16384-token per-rank count. The loss is computed per token.
- The recipes use mock data with forced balanced routing, a 50-step cosine
  schedule and a peak learning rate of 1e-5. MTP and per-head Muon are
  experimental settings inherited from the reference configuration.

## Measured throughput

Both recipes ran for 20 iterations with Megatron-LM at the commit that
introduced these configurations: the full model on 256 GB300 GPUs (64 nodes,
4 GPUs each) and the proxy on 64 GB300 GPUs (16 nodes; the proxy recipe
targets GB200 but was measured on GB300). Steps 1-5 (warmup) and the last
two steps (profiler capture, garbage collection) are excluded from the
medians.

| Metric | Full model, 256 GPUs | 9L proxy, 64 GPUs |
|---|---:|---:|
| Median iteration time (steps 6-18) | 163.4 s (162.8-164.3 s) | 60.7 s (59.8-61.5 s) |
| Median throughput | 498 TFLOP/s/GPU | 661 TFLOP/s/GPU |
| Tokens per iteration | 16.8M (64 x 262144) | 16.8M (64 x 262144) |
| First iteration (mock dataset construction, AttnRes compilation, HybridEP setup) | 1736 s | 485 s |
| Peak allocated memory | 156.5 / 148.8 / 149.4 / 141.9 GB on the first rank of pipeline stages 0-3; 172.4 GB over all ranks | 62.0 GB on rank 0; 67.0 GB over all ranks |

No iteration was skipped and no NaN occurred in either run. The measurement
container carried the component revisions listed under "Build the container"
except for HybridEP, where it used revision d28bd67; the Dockerfile has since
moved the pin to 10d4dd7, because revisions before 17cfb81 fail intermittently
with an illegal memory access once the per-rank token count per dispatch grows
(observed at 524288 packed tokens with CP16, i.e. 32768 tokens per rank; see
DeepEP issue 756). At 262144 tokens both runs are attention-bound: the MLA attention
kernels and the context-parallel ring traffic that feeds them take the
largest share of GPU time, followed by the MXFP8 GEMMs and the HybridEP
dispatch/combine. The gap between the proxy's 661 and the full model's 498
TFLOP/s/GPU comes from the four-stage pipeline: its bubble, the inter-stage
transfers and the heavier last stage, which carries the MTP block and the
output layer.

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

Both recipes enable `gdn_pre_gated_delta_rule_fusion`, which calls the C++
depthwise-convolution backward of causal-conv1d. Without that package the KDA
layer raises an `ImportError` in its first forward pass. No prebuilt
causal-conv1d wheel matches the NGC 26.04 base, so the Dockerfile builds the
pinned revision from source; nvcc 13 includes the SM100 and SM103 cubins.

TE targets both SM100a and SM103a. The NGC PyTorch installation supplies
PyTorch and CUDA; the extensions build against that installation. A
`/opt/venv` environment inherits those system packages and installs the recipe
dependencies locally. This lets pip upgrade Python dependencies without
uninstalling Debian-managed packages such as `python3-yaml`. The venv is first
on `PATH`, and `pip check` remains a build requirement. The build also installs
`pytest`, a declared dependency of NGC's `triton-kernels`. Run with the image's
default Python environment.

This is a new public build definition assembled from the recorded Kimi-K3
dependency revisions and the public reference Dockerfiles. The original
benchmark image's complete Dockerfile was not recorded. This definition has
passed a native ARM64 build, including dependency consistency checks.

A software-stack probe on one GB300 node (four GPUs) checked the 2026-09-17
image: the default Python is the `/opt/venv` environment, `pip check` passes
at runtime, every component above is installed at the listed revision, the
whole stack imports, TE MXFP8 and Flash Linear Attention kernels run on the
device, and the Megatron Core unit tests for KDA, quantile balancing, HybridEP,
latent MoE, AttnRes/MTP, SiTU-GLU and MXFP8 parameter gather with Muon ran on
it. The fused pre-GDN tests were skipped because causal-conv1d was missing;
the Dockerfile now installs it.

With causal-conv1d installed on top of that image, the same probe passed in
full: the fused pre-GDN/KDA tests ran, and the only remaining failures were an
unrelated upstream test typo. A 29-decoder-block single-node proxy of this
architecture (KDA, MLA, latent MoE with HybridEP, MTP, compiled AttnRes, fused
pre-GDN, MXFP8 parameter gather, per-head Muon, chunked optimizer-state
offload) then trained for 50 iterations on one GB300 node with mock data: no
errors, no NaN or skipped iterations, a decreasing loss and constant memory.

TE is built with `NVTE_WITH_NCCL_EP=0`: its optional NCCL EP extension uses
a NCCL device API that does not match the NGC 26.04 base. These recipes select
HybridEP as the MoE dispatcher backend.

The HybridEP build follows the public DeepSeek-V4 recipe and targets one
NVLink domain per EP group. Place each group of 64 EP ranks within one
GB200/GB300 NVLink domain. The full recipe needs four such groups, one per
pipeline stage. EP groups spanning separate NVLink domains require a
multinode-enabled communication build and a corresponding placement review.
This HybridEP revision provides no dense top-k routing metadata, so Megatron
Core uses the bool routing map with it.

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
export RECIPE=examples/moe_recipes/kimi_k3_proxy_9layer/gb200/mxfp8_THD256K_64GPU_TP4PP1EP64CP16_muon.yaml
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
