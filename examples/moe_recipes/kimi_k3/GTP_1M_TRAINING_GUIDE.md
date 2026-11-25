# Kimi-K3 on GTP at 1M context: training guide

This guide explains how the full Kimi-K3 text backbone (93 decoder blocks, 896 experts) is
trained on 256 GB300 GPUs with Generalized Tensor Parallelism (GTP) at a context length of
1,048,576 tokens, one sequence per microbatch, and how to launch that run. It documents the
configuration of one measured run; it is not a general-purpose tuned recipe.

**Status (2026-11-22, Megatron `659174d21`): 855.6 TFLOP/s/GPU as printed, 56.4 s per
optimizer step of 4,194,304 tokens, 167.6 GiB peak allocated per GPU, 20 steps on mock
data.** This is a single run on a single allocation. [Measured results](#measured-results)
explains what the number covers, and
[What was and was not verified](#what-was-and-was-not-verified) explains how that launch
differs from the scripts in this directory.

| File | Purpose |
|---|---|
| [gb300/mxfp8_THD1M_256GPU_GTP64EP64CP128_muon.yaml](gb300/mxfp8_THD1M_256GPU_GTP64EP64CP128_muon.yaml) | Recipe: the environment and the 169 training flags of the measured run |
| [gtp_launch/kimi_k3_gtp_entry.py](gtp_launch/kimi_k3_gtp_entry.py) | Entry point: wraps `pretrain_hybrid.py` with three launch-side changes |
| [gtp_launch/offload_early_master_d2h.py](gtp_launch/offload_early_master_d2h.py) | Early host copy of offloaded master weights (used by the entry point) |
| [gtp_launch/hybridep_overlay_env.sh](gtp_launch/hybridep_overlay_env.sh), [hybridep_combine_prob_tma_patch.py](gtp_launch/hybridep_combine_prob_tma_patch.py) | Per-node patched copy of HybridEP's JIT kernel header |
| [gtp_launch/render_launcher.py](gtp_launch/render_launcher.py) | Turns the recipe YAML into a per-node launch script |
| [gtp_launch/node_launch.sh](gtp_launch/node_launch.sh), [submit_256gpu.sbatch](gtp_launch/submit_256gpu.sbatch) | Per-node task and an example Slurm submission |

## Parallel configuration

| Dimension | Value | Flag |
|---|---|---|
| GPUs | 256 (64 nodes x 4 GB300, four NVL72 racks of 16 nodes) | |
| Tensor / pipeline parallel | 1 / 1 | `tensor_model_parallel_size`, `pipeline_model_parallel_size` |
| GTP dense weight shards | 64 | `tensor_parallel_num_weight_shards` |
| GTP expert weight shards | 4 | `expert_tensor_parallel_num_weight_shards` |
| GTP sequence shards | 64 | `tensor_parallel_num_sequence_shards` |
| Context parallel, configured | 2 | `context_parallel_size` |
| Context-parallel group at run time | 128 = 64 sequence shards x 2 | |
| MLA latent CP communication | all-to-all over 32 ranks, then a ring of 4 | `cp_comm_type: a2a+p2p`, `hierarchical_context_parallel_sizes: [32, 4]` |
| Expert parallel | 64 (896 experts, 14 per rank) | `expert_model_parallel_size` |
| Data parallel | 2 | derived |
| Sequence length | 1,048,576 (8,192 tokens per GPU) | `seq_length`, `max_seqlen_per_dp_cp_rank` |
| Microbatch / global batch | 1 / 4 (two microbatches per DP rank per step) | `micro_batch_size`, `global_batch_size` |
| Tokens per optimizer step | 4,194,304 | |

Megatron prints the resolved layout at start-up. The measured run printed:

```text
using world size: 256, data-parallel size: 2, resolved context-parallel group size: 128,
hierarchical context-parallel sizes: [32, 4], tensor-model-parallel size: 1,
pipeline-model-parallel size: 1
```

### What the dimensions mean here

- **No TP, no PP.** Every GPU runs every layer on its own part of the sequence. There are no
  pipeline stages and no tensor-parallel activations; GTP splits the weights, and GTP
  sequence shards together with CP split the tokens.
- **GTP weight shards.** Dense weights are stored in 64 shards, one on each rank of a
  64-rank group. A layer's weights are all-gathered right before the layer runs, and its
  gradients are reduce-scattered afterwards. The all-gather carries MXFP8 weights
  (`fp8_param_gather`); gradients are reduced in bf16 with an fp32 accumulator
  (`gtp_remat_reduce_scatter_with_fp32_accumulation`). Expert weights are first split by
  EP64 and then into 4 shards. `gtp_remat_opt_in_modules: moe_latent_proj` adds the latent
  projections of the MoE layers to the sharded weights. The all-gathers and reduce-scatters
  run on NCCL user buffers (`gtp_remat_nccl_ub`, `gtp_expert_remat_nccl_ub`) with NVLink
  SHARP.
- **GTP sequence shards.** The same 64-rank group also splits the token sequence into 64
  parts. Combined with the configured CP of 2, this gives a context-parallel group of 128
  ranks: the 1M sequence is cut into 128 contiguous pieces of 8,192 tokens.
- **KDA layers** (69 of the 93 blocks) run chunkwise on the contiguous partition
  (`cp_partition_mode: contiguous`, `linear_cp_mode: chunkwise`). FLA's CP kernels merge
  the state across ranks.
- **MLA layers** (24 blocks) use latent CP (`mla_latent_cp`): ranks exchange the compressed
  KV latent instead of the expanded K/V. On the 128-rank group the exchange is
  hierarchical: an all-to-all inside each group of 32 consecutive ranks (3 of the 96 heads
  per rank), then a ring of 4 steps between those groups. The attention kernels are
  cuDNN's (`attention_backend: fused`); the FlashAttention-4 path of the 512K recipes is
  not used.
- **Residual stream layout.** `linear_cp_layout: contiguous` keeps the residual stream in
  the contiguous layout, so KDA layers need no layout conversion; each MLA layer converts
  its own input to zigzag and back. `cp_contiguous_padded_qkv_seqlens` gives the contiguous
  view the padded sequence lengths. On this branch Attention Residuals can run with a
  layout manager that never converts the residual stream; the limitation described in
  [README.md](README.md) applies to the TP/PP recipes.
- **MoE.** EP64 with the HybridEP flex dispatcher, grouped GEMM and permute fusion. GDN,
  MoE and MLP activations are recomputed (`recompute_modules: [gdn, moe, mlp]`), and
  attention activations are offloaded to the host
  (`offload_modules: [core_attn, attn_proj]`).
- **Optimizer.** Muon with per-head QKV. Dense matrices use the layer-wise distributed mode
  (`muon_tp_mode: auto`), and expert matrices use the layer-sharded mode
  (`muon_expert_tp_mode: layer_sharded`). Optimizer states live on the host and visit the
  GPU in 1 GiB chunks (`chunked_optimizer_state_offload`).
- **Triton kernels.** `triton_autotune_mode: pinned` gives every rank the same kernel
  configuration. Kernels listed in `triton_autotune_config_invariant` keep the
  configuration Triton selects by timing. The list includes FLA's cross-rank merge kernel,
  whose pinned default is about 25 times slower at 127 ranks crossed (18.15 ms against
  0.73 ms per call, with identical values).

### Placement on the machine

Each group is a block of consecutive ranks, and `torch.distributed.run` numbers ranks node
by node. The node order of the launch therefore decides which GPUs share a group:

| Group | Size | Ranks (first block) | Hardware |
|---|---:|---|---|
| GTP weight / sequence-shard group | 64 | 0-63 | one rack (one NVLink domain) |
| Expert-parallel group | 64 | 0-63 | one rack |
| Latent CP all-to-all stage | 32 | 0-31 | half a rack |
| Context-parallel group | 128 | 0-127 | two racks |
| Data-parallel replicas | 2 | 0-127, 128-255 | two racks each |

This has two consequences:

- Allocate four blocks of 16 nodes, each inside one rack (`--segment=16` with Slurm), and
  give the nodes of a rack consecutive node ranks. HybridEP in the public image is built
  for a single NVLink domain; an EP group that spans racks does not work with it.
- Inside each CP group, the latent ring and the KDA state exchange cross the rack boundary.
  With TransformerEngine's K/V ring in place of the latent exchange, the cross-rack ring
  took about 70 % of a 67 s step at 1M (one microbatch). The latent ring reduced the step
  to 47 s and its `a2a+p2p` mode to 37 s, before the later changes that are part of this
  recipe.

## Launching

### 1. Image and checkout

Use the image built from [Dockerfile](Dockerfile) (see [README.md](README.md)). The measured
run used an image with the same component revisions (TransformerEngine `0a2ebe9`, cuDNN
frontend `5ba9432`, FLA 0.5.2, HybridEP `10d4dd7`, NCCL 2.32.3, Emerging-Optimizers
`a44d1f8`). The checkout must be visible at the same path on every node.

### 2. Slurm

[gtp_launch/submit_256gpu.sbatch](gtp_launch/submit_256gpu.sbatch) is an example for Slurm
with the pyxis container plugin:

```bash
MEGATRON_DIR=/path/to/Megatron-LM CONTAINER_IMAGE=/path/to/image.sqsh \
    sbatch --account=<account> examples/moe_recipes/kimi_k3/gtp_launch/submit_256gpu.sbatch
```

The script sorts the job's host names, which keeps the nodes of a rack together when the
rack is part of the name, and starts [node_launch.sh](gtp_launch/node_launch.sh) once per
node inside the container. Arguments after the script name are appended to the training
command line, for example `--train-iters 50`.

### Run the full model at 256K

Use 32 context-parallel ranks for 262,144 tokens to retain the 1M recipe's 8,192
tokens per rank. On the same 256 GPUs, keep 64 dense weight shards, four expert
weight shards, and EP64. Set the GTP sequence shards to 32 and the configured CP
size to one. Each 64-rank dense weight group then processes two independent samples.
MLA uses one all-to-all over the 32 CP ranks, with three of the 96 attention heads
per rank after the exchange.

```bash
MEGATRON_DIR=/path/to/Megatron-LM CONTAINER_IMAGE=/path/to/image.sqsh \
    sbatch --account=<account> examples/moe_recipes/kimi_k3/gtp_launch/submit_256gpu.sbatch \
    --seq-length 262144 --max-position-embeddings 262144 --max-seqlen-per-dp-cp-rank 8192 \
    --tensor-parallel-num-sequence-shards 32 --context-parallel-size 1 \
    --cp-comm-type a2a --hierarchical-context-parallel-sizes 32 \
    --global-batch-size 16 \
    --varlen-mock-dataset-config-json '{"mode":"distribution","type":"lognormal","format":"thd","min_seq_len":262144,"max_seq_len":262144,"mean_seq_len":262144,"lognormal_sigma":1.1}'
```

The resolved CP size is 32. The 256 GPUs process eight independent samples per
microbatch (four data-parallel replicas × two batch shards). Global batch size 16
gives two microbatches and 4,194,304 tokens per optimizer update, matching the
1M reference run.

### 3. Any other scheduler

Run the following once per node, inside the container, with the repository root as the
working directory:

```bash
python examples/moe_recipes/kimi_k3/gtp_launch/render_launcher.py \
    examples/moe_recipes/kimi_k3/gb300/mxfp8_THD1M_256GPU_GTP64EP64CP128_muon.yaml > run.sh
NNODES=64 NODE_RANK=<0..63> MASTER_ADDR=<host of node rank 0> bash run.sh
```

`run.sh` exports the recipe's `ENV_VARS`, puts the repository root on `PYTHONPATH`, sources
the HybridEP overlay and starts `torch.distributed.run` with four processes on the entry
point. `NODE_RANK` must follow the rack order described above.

### Time budget

In the measured run, the first forward pass started 5 minutes after the job started. The
first step took 259 s (kernel compilation, autotuning, communicator set-up) and later steps
about 56 s; the 20-step job ran for 29 minutes in total. The example requests one hour.
Submit with `--no-requeue`; otherwise Slurm restarts the 64-node job on its own after a
node failure.

### Variants

| Variant | Change | Recorded result |
|---|---|---|
| One microbatch per step | `--global-batch-size 2` | 790.6 TFLOP/s/GPU, 30.50 s per step |
| With the MTP depth of the reference recipes | In the recipe: append `/+E` to `hybrid_layer_pattern`, add `mtp_num_layers: 1` and `mtp_loss_scaling_factor: 0.1` | 845.5 TFLOP/s/GPU, 59.16 s per step |
| Real data | Replace `mock_data`, the mock dataset JSON and `moe_router_force_load_balancing` (see the 64K recipes) | not run |

## Launch-side changes

The measured run did not start `pretrain_hybrid.py` directly: four changes were applied
around it at launch. None of them changes arithmetic, precision or which checks run, and
none modifies a file of the checkout. Because they are part of the recorded configuration,
they are included in this directory.

1. **Deferred validation of the NaN/Inf checks** (`kimi_k3_gtp_entry.py`). Megatron checks
   the loss and the gradient norm of every bucket for NaN/Inf, and each check makes the
   host wait for the GPU. With GTP's per-layer buckets this adds up to hundreds of
   synchronisations per backward pass. The entry point routes the eager forward-backward
   function through the rerun state machine's own deferred-validation facility (the one
   the full-iteration CUDA graph uses), so that all checks are evaluated together, after
   the forward-backward pass and before the optimizer update.
2. **Host memory bound to the GPU's NUMA node** (`MCORE_NUMA_MEMBIND=gpu`). A GB300 node has
   two sockets with two GPUs each. Without the binding, part of a rank's pinned host
   buffers ends up on the other socket; that rank's offload copies run slower, and all
   other ranks wait for it.
3. **Early host copy of offloaded master weights** (`MCORE_EARLY_MASTER_D2H=1`). With
   chunked optimizer-state offload, the master weights are normally copied to the host all
   at once before the next forward pass. The module instead copies the master weights of
   each chunk right after the chunk is stepped, so the copy overlaps the rest of the
   optimizer step. `MCORE_EARLY_MASTER_D2H_VERIFY=N` compares the early copies bit for bit
   at the first N releases.
4. **HybridEP JIT header patch** (`hybridep_overlay_env.sh patched`). With 14 experts per
   rank, the probability slice of the backward combine is not a multiple of 16 bytes, and
   HybridEP `10d4dd7` copies it with scalar loads in its producer thread. The patch copies
   the enclosing 16-byte-aligned window with one TMA bulk copy instead; the accumulated
   values are the same floats. The script works on a per-node copy of the installed
   `deep_ep` package and fails if an anchor does not match exactly once.

To switch them off individually, use `MCORE_DEFERRED_VALIDATION_MODE=passthrough`,
`MCORE_NUMA_MEMBIND=off`, `MCORE_EARLY_MASTER_D2H=0` and
`render_launcher.py --hybridep-overlay installed`.

The four changes were introduced on the 8K-context configuration of the same model, in
three steps, each measured against a control on the same allocation (one job per step):
the HybridEP patch together with deferred validation (597.5 -> 682.0 TFLOP/s/GPU), then
the early master copy (681.7 -> 686.3), then the memory binding (676.4 -> 706.9). Their
contribution to the 1M step time was not measured separately.

## Measured results

Full model, 256 GB300 GPUs, 1M context, mock data, forced balanced routing, 20 optimizer
steps per run, Megatron `659174d21`. The statistics cover steps 6-20, because the first
steps include compilation and autotuning. "TFLOP/s/GPU" is the value Megatron prints with
`log_throughput`, computed from its FLOPs formula; within a run, the product of step time
and printed throughput is constant across steps to about 0.01 %.

| Run | Step time, median (mean) | TFLOP/s/GPU at the median (mean) step time | Tokens per step | Peak allocated / device used per GPU |
|---|---:|---:|---:|---:|
| Two microbatches per step (this recipe) | 56.37 s (56.45 s) | 855.6 (854.4) | 4,194,304 | 167.6 / 221.7 GiB |
| One microbatch per step | 30.50 s (30.77 s) | 790.6 (783.6) | 2,097,152 | 165.4 / 219.5 GiB |
| Two microbatches + one MTP depth | 59.16 s (59.44 s) | 845.5 (841.5) | 4,194,304 | 169.7 / 224.1 GiB |

A GB300 has 276.6 GiB of device memory. The mean is the throughput a longer run would
see; the median is less sensitive to individual slow steps. Step times scatter in one
direction only: most steps are close to the fastest one, and a few are slower. Python's
garbage collection runs after every tenth step (`manual_gc_interval: 10`), and its cost
is counted in the following step.

How to read these numbers:

- Each row is one run on its own allocation. There is no same-allocation A/B between the
  rows and no repeat of any row; differences of 1-2 % between runs are within what a
  different set of nodes can produce.
- The runs use mock data with fixed-length documents and forced balanced routing. They
  measure throughput and memory, not convergence. With the router's own routing the expert
  load is not balanced, and the step time will differ.
- The headline recipe has no MTP depth, whereas the reference recipes in this directory
  enable one. With the MTP depth, the step is 5.0 % longer and the printed throughput is
  1.2 % lower.
- Loss values on mock data say nothing about the model.

## What was and was not verified

- **The training flags are the measured ones.** The launch script that `render_launcher.py`
  renders from the recipe was compared with the command line of the measured run: 169
  flags, identical in order and value. The run's W&B flags are left out.
- **The launch scripts are not the ones that ran.** The measurement was launched by an
  external toolkit. Its entry script carried the three mechanisms of `kimi_k3_gtp_entry.py`
  together with diagnostics (see below) and experiment switches that were off.
  `kimi_k3_gtp_entry.py` contains only those three mechanisms;
  `offload_early_master_d2h.py` and `hybridep_combine_prob_tma_patch.py` are copies of the
  files that ran. The entry point and the HybridEP overlay in this directory were run once,
  on one GB300 node with a small proxy of the model (four GPUs, four steps, 2026-11-25).
  In that run, host memory was bound on all four ranks, the early master copies were
  verified bit for bit at every release (`MCORE_EARLY_MASTER_D2H_VERIFY=-1`), and the
  first-step losses were equal to those of the same proxy under the toolkit's entry
  script. `render_launcher.py` was only run outside a cluster. `node_launch.sh` and
  `submit_256gpu.sbatch` were not run, and neither was the full model under this entry
  point. Expect to adapt the Slurm example to your site.
- **Not carried over.** The environment of the measured run listed two source-level patches
  of TransformerEngine's own CP ring attention. This configuration does not call that code
  (the MLA layers use the latent CP path), so the patches are omitted. The diagnostics of
  the measured run are not included either: a per-rank memory report in each of the first
  two steps, and a watchdog that dumps stacks when a forward-backward call exceeds 240 s.
- **Precision.** The full model was not compared against a TP/PP reference at 1M. Whether a
  launch of this recipe is numerically equivalent to the reference recipes has to be
  established separately (see the continuation check described in [README.md](README.md)).
- **Scale.** Other sequence lengths, GPU counts or interconnects need their own layout: the
  group sizes above are tied to four racks of 16 nodes.
