#!/bin/bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Example Slurm submission of the Kimi-K3 GTP 1M recipe: 64 GB300 nodes (4 GPUs each) in
# four NVL72 racks. Written for Slurm with the pyxis container plugin (--container-*
# options of srun); adapt account, partition, QoS and the container options to your site.
#
#   MEGATRON_DIR=/path/to/Megatron-LM CONTAINER_IMAGE=/path/or/registry/image \
#       sbatch --account=<account> examples/moe_recipes/kimi_k3/gtp_launch/submit_256gpu.sbatch
#
#SBATCH --job-name=kimi_k3_gtp_1m
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
# Blocks of 16 nodes, each inside one rack (one NVLink domain).
#SBATCH --segment=16
# Start-up and the first step take about 10 minutes, each later step about 1 minute.
#SBATCH --time=01:00:00
# A node failure must not start the job again by itself.
#SBATCH --no-requeue
set -euo pipefail

: "${MEGATRON_DIR:?Megatron-LM checkout, visible at the same path on every node}"
: "${CONTAINER_IMAGE:?training image (see examples/moe_recipes/kimi_k3/Dockerfile)}"
RECIPE="${RECIPE:-examples/moe_recipes/kimi_k3/gb300/mxfp8_THD1M_256GPU_GTP64EP64CP128_muon.yaml}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-${MEGATRON_DIR}:${MEGATRON_DIR},/tmp:/tmp}"

# Rank order: host names sorted, which keeps the 16 nodes of a rack together when the rack
# is part of the host name (rack ID before node ID). Check the list if your names differ.
KIMI_K3_HOSTS="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | sort | paste -sd, -)"
export KIMI_K3_HOSTS
echo "rank order of the nodes: ${KIMI_K3_HOSTS}"

srun --ntasks-per-node=1 \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${CONTAINER_MOUNTS}" \
  --container-workdir="${MEGATRON_DIR}" \
  bash examples/moe_recipes/kimi_k3/gtp_launch/node_launch.sh "${RECIPE}" "$@"
