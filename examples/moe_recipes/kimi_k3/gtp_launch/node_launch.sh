#!/usr/bin/env bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Per-node task of a Kimi-K3 GTP launch: one copy on every node, inside the training
# container, with the Megatron-LM repository root as working directory.
#
#   KIMI_K3_HOSTS=host0,host1,... bash examples/moe_recipes/kimi_k3/gtp_launch/node_launch.sh \
#       RECIPE.yaml [extra training flags]
#
# KIMI_K3_HOSTS lists the nodes of the job in rank order, the same list on every node. The
# node's position in it is its torchrun node rank; the first host is the rendezvous master.
# Keep the nodes of one NVLink domain (rack) next to each other in the list: the EP group, the
# GTP weight-shard group and the all-to-all stage of the latent CP are blocks of consecutive
# ranks (see ../GTP_1M_TRAINING_GUIDE.md).
set -euo pipefail

recipe="${1:?usage: node_launch.sh RECIPE.yaml [extra training flags]}"
shift
IFS=, read -r -a hosts <<< "${KIMI_K3_HOSTS:?comma-separated host list in rank order}"
me="$(hostname -s)"
NODE_RANK=""
for index in "${!hosts[@]}"; do
  if [ "${hosts[${index}]%%.*}" = "${me}" ]; then
    NODE_RANK="${index}"
  fi
done
if [ -z "${NODE_RANK}" ]; then
  echo "node_launch.sh: host ${me} is not in KIMI_K3_HOSTS" >&2
  exit 1
fi
export NODE_RANK
export NNODES="${#hosts[@]}"
export MASTER_ADDR="${hosts[0]}"
echo "node_launch.sh: host=${me} node_rank=${NODE_RANK} nnodes=${NNODES} master=${MASTER_ADDR}"

launcher="$(mktemp /tmp/run-kimi-k3-gtp.XXXXXX)"
python examples/moe_recipes/kimi_k3/gtp_launch/render_launcher.py "${recipe}" > "${launcher}"
exec bash "${launcher}" "$@"
