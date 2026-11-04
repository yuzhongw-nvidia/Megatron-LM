# Kimi-K3 Blackwell recipes

These decoder-only recipes use `pretrain_hybrid.py`. The full model has 93 decoder
blocks (69 KDA and 24 MLA) and one additional MLA/MoE multi-token prediction (MTP)
depth. The 9-block proxy has seven KDA blocks, two MLA blocks and one MTP depth.
Both use the full widths, 96 attention heads and 896 routed experts with top-16 routing.

## Full-model throughput on GB300

| Packed THD length | GPUs | TP/PP/EP/CP | GBS | Median step | Median TFLOP/s/GPU | Measurement |
|---|---:|---|---:|---:|---:|---|
| [64K](gb300/mxfp8_THD64K_256GPU_TP1PP4EP64CP8_muon.yaml) | 256 | 1/4/64/8 | 256 | 58.8662 s | 877.9 | 2026-11-25, cuDNN KDA |
| [256K](gb300/mxfp8_THD256K_256GPU_TP1PP4EP64CP32_muon.yaml) | 256 | 1/4/64/32 | 64 | 94.5596 s | 860.5 | 2026-11-25, cuDNN KDA |
| [1M](gb300/mxfp8_THD1M_512GPU_TP4PP4EP64CP32_muon.yaml) | 512 | 4/4/64/32 | 32 | 246.27 s | 813 | **Historical only** · 2026-10-28, FLA KDA, `5d15f3590` |

The two November measurements ran Megatron-LM
`46467f8deca440c724653e1b17443038e5f3e1a8` with the improved cuDNN KDA
implementation, cuDNN frontend 1.33.0 and CUTLASS DSL 4.8.1. Step time and logged
TFLOP/s/GPU are separate medians over iterations 6–50 of 50. Each mock document
fills one packed sequence. At both lengths the global batch contains 16,777,216
tokens per optimizer step, with 8,192 tokens per CP rank. Peak allocated device
memory was 219.9 GiB at 64K and 220.0 GiB at 256K.

**The 1M result predates the current environment.** Its last measured revision was
`5d15f3590ec760928d408bcb389313f4b3e8f761`. The October run used FLA KDA, the
MoK `d59188a` image, TP4 with sequence parallelism, 32,768 tokens per CP rank and
33,554,432 tokens per step. The medians cover steps 2–5 of five. The published
1M YAML retains that parallel layout and selects cuDNN KDA. Its performance with
frontend 1.33.0 remains unmeasured.

## 9-block proxy

| Recipe | GPUs | TP/PP/EP/CP | GBS | Data | Median step | Median TFLOP/s/GPU |
|---|---:|---|---:|---|---:|---:|
| [64K benchmark](../kimi_k3_proxy_9layer/gb300/mxfp8_THD64K_64GPU_TP1PP1EP64CP8_muon.yaml) | 64 | 1/1/64/8 | 256 | Packed 32K–64K mock | 20.7224 s | 775.1 |
| [512K benchmark](../kimi_k3_proxy_9layer/gb300/mxfp8_THD512K_64GPU_TP4PP1EP64CP16_muon.yaml) | 64 | 4/1/64/16 | 32 | Fixed-length mock | 60.07 s¹ | 984.4¹ |
| [64K continuation](../kimi_k3_proxy_9layer/gb300/mxfp8_THD64K_64GPU_TP4PP1EP64CP2_muon.yaml) | 64 | 4/1/64/2 | 256 | LongAlign-10k | — | — |

The 64K benchmark reports unprofiled steps 4–20 of 25 on November 24. The
executed checkout was `dd62d69aa50e` with the optimized KDA module applied.
Its file matched the replay revision `46467f8deca4` byte for byte. The proxy and
full-model data distributions differ.

¹ Historical FLA KDA measurement on `d223af3ebac4ee07f1e9dc57194c147dddb6ef0e`
with NVLS disabled, median over steps 2–18. The cuDNN KDA 512K recipe has no new timing.

## Runtime configuration

- KDA uses `kda_backend: cudnn`, `linear_cp_mode: chunkwise` and contiguous
  packed THD shards. FLA remains the implementation of attention residuals,
  including their fused output RMSNorm.
- Dense MLA exchanges latent KV with FlashAttention-4. At TP1 the 64K full model
  uses all-to-all over eight ranks. The 256K full model adds a four-way ring
  (`a2a+p2p` with group sizes `[8, 4]`). Both use MBS1, ETP1 and four
  pipeline stages with 24/24/24/21 decoder blocks. MTP runs on the last stage.
- The benchmark recipes use forced balanced expert routing, MXFP8 parameter
  gather, per-head Muon, the MoK routed-expert megakernel with native shared
  experts, selective recomputation of GDN/MoE/MLP, and chunked optimizer-state
  offload. The full model fuses the elementwise MLA output gate and offloads
  core attention and the attention projection. The 64K proxy YAML records the
  switches used in its own timing run.
- The full models set `NCCL_BUFFSIZE=2097152`. The proxies use `1048576`.
  All use `NCCL_LL128_BUFFSIZE=1048576` and `NCCL_PROTO=^LL128`. Launch
  from the repository root so the shared communicator config resolves.
  The `tp_dp_cp` group uses `max_ctas: 16`. [Communicator configuration](nccl/tp_dp_cp_max_ctas16.yaml)

## Earlier numerical validation

The fused-AttnRes change was checked on the 9L 64K
real-data protocol (LongAlign-10k, 32 continuation steps from a shared
checkpoint, router load balancing off, `alltoall` dispatcher): maximum
absolute LM-loss difference against the compiled reference 1.74e-4, MTP
2.35e-4, below the 1e-3 threshold (2026-10-13). The MoK megakernel was
compared with the native MoE path on the 9L 64K recipe from random
initialization over 32 steps: per-step LM-loss differences grow to 1.62e-3
at step 20 and re-converge to 8.0e-5 at step 32 with identical gradient-norm
trajectories (2026-10-20); a same-configuration rerun floor for that
from-scratch protocol has not been established, so this is a trajectory
check rather than a parity pass. The rebased code base matched the
pre-rebase one on the 9L 64K real-data protocol with a maximum LM-loss
difference of 1.9e-4 (2026-10-26). The earlier checks of the latent CP
attention path (2.69e-4 on 2026-10-13) remain valid. No full-model 64K
real-data result is claimed here; the mock-data throughput runs do not
establish long-context convergence.

## Build the container

Build the shared KDA/MLA environment on an ARM64 builder. The November
measurements used a published ARM64 image with manifest digest
`sha256:5991a37ee5019e514c064e01bb69d49bcc796e8eaed2d513688e3f22b2ff66ea`.
The [Dockerfile](Dockerfile) pins the public build inputs.

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
| cuDNN frontend | `1.33.0` (CPython 3.12 ARM64 wheel, SHA-256 `5ab2417b891b126d7cfdb11066a65a791a715bfbe53113502ecfdf0de1216abe`) |
| Flash Linear Attention / fla-core | `0.5.2` |
| FlashAttention-4 | `4.0.0b11` (required by the MLA latent CP split forward/backward API) |
| Quack kernels | `0.4.1` |
| Transformers | `5.16.1` |
| NVIDIA Resiliency Extension | `0.6.0` |
| Emerging-Optimizers | `a44d1f83a4950b445f2a77ca71177c5f033d45d5` |
| causal-conv1d | `4f6ae4e26ae5fe8af9372f8d312ab25cc4595223` (1.6.2.post1) |
| HybridEP | `10d4dd7377d5bce900fbb4b80cce863764892b95` (hybrid-ep branch) |
| Mixture-of-Kittens (MoK) | `d59188aab3c9d5f92dd197d9997226ba3a43c701` (ThunderKittens 1c3920d9 with the MXFP8 MN-major patch; built for sm_100a and sm_103a) |
| CUTLASS DSL | `4.8.1` (CUDA 13) |
| Apache TVM FFI | `0.1.11` |

The measured environment preserves the FA4 split forward/backward API and
applies two DSL 4.8 adaptations to FA4 and Quack: `ThrMma` and `ThrCopy`
move to `cutlass.cute`, and register allocation uses `cute.make_rmem_tensor`.
The frontend wheel supplies the KDA summaries and FROST state-chain routines.

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

The published measurements used the recorded image digest. The public
Dockerfile was synchronized with those package versions and patches and has
not been rebuilt as part of this recipe update.

## Prepare the launch

The YAML has the recipe schema `DEPENDENCIES` / `ENV_VARS` / `ARGS`; it is
not the training entry point's `--yaml-cfg` schema. Export `ENV_VARS` and
translate `ARGS` into command-line flags. Lists become multiple arguments;
`true` becomes a flag; `false` and `null` are omitted. A JSON-valued mapping is encoded as one argument.

The following example writes a launch script from any recipe. Run it
inside the built container with the checkout as the working directory,
Python/PyYAML available and `OUTPUT_PATH` set to a writable, container-visible
directory. The benchmark recipes use mock data and `NullTokenizer`. The
separate TP4/CP2 64K continuation example uses LongAlign-10k and
`SFTTokenizer`, with a training-only split and a cache under
`${OUTPUT_PATH}/data_cache/longalign_10k`. For a continuation
check, supply a matching checkpoint with `--load CHECKPOINT --finetune`.
The recipes enable one MTP depth: a checkpoint without MTP weights requires
removing `mtp_num_layers` and the trailing `/+E` from `hybrid_layer_pattern`
in the YAML before rendering the launcher.

```bash
export RECIPE=examples/moe_recipes/kimi_k3_proxy_9layer/gb300/mxfp8_THD64K_64GPU_TP1PP1EP64CP8_muon.yaml
: "${OUTPUT_PATH:?Set OUTPUT_PATH to the run output directory}"
export OUTPUT_PATH
python - <<'PYTHON' > run-kimi-k3.sh
import json
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
    if isinstance(value, dict):
        args.append(json.dumps(value, separators=(",", ":")))
    elif value is not True:
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
the full 64K and 256K recipes, 128 nodes for the 1M recipe and 16
nodes for the proxy.
Each node must use the same recipe, image and rendezvous address;
`NODE_RANK` must be unique.

`DYNAMO_CACHE_SIZE_LIMIT=64` accommodated the different depth-source counts
of the compiled AttnRes path; the FLA path does not depend on it, and the
recipes keep the export so earlier launch scripts remain valid.
