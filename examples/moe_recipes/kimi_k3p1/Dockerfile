# syntax=docker/dockerfile:1.7
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Kimi-K3.1 training dependencies for GB200/GB300 (ARM64): the Kimi-K3 image plus the
# kernels that DSA sparse attention needs.
#
# Build the Kimi-K3 image first (../kimi_k3/Dockerfile), then run from the Megatron-LM
# repository root:
#   docker build --platform=linux/arm64 \
#       --build-arg KIMI_K3_IMAGE=megatron-kimi-k3:blackwell \
#       -f examples/moe_recipes/kimi_k3p1/Dockerfile -t megatron-kimi-k3p1:blackwell .
#
ARG KIMI_K3_IMAGE=megatron-kimi-k3:blackwell
FROM ${KIMI_K3_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# The parent provides cuDNN frontend 1.33.0, the H96 DSA path and cuDNN KDA.
# Keep FlashMLA available for head counts that select its sparse forward.
ARG FLASH_MLA_COMMIT=b7643bd54521f563b839b98289b5cd048c062ba2

# SM100 only; the SM90 kernels are not built.
RUN git clone https://github.com/deepseek-ai/FlashMLA.git /opt/FlashMLA && \
    cd /opt/FlashMLA && git checkout "${FLASH_MLA_COMMIT}" && \
    git submodule update --init --recursive && \
    unset PIP_CONSTRAINT && \
    FLASH_MLA_DISABLE_SM90=1 NVCC_THREADS=16 \
    CFLAGS="-I/usr/local/cuda/include/cccl" CXXFLAGS="-I/usr/local/cuda/include/cccl" \
    python -m pip install --no-build-isolation --no-cache-dir .

RUN python -m pip check && \
    python -c 'from importlib.metadata import version; assert version("nvidia-cudnn-frontend") == "1.33.0"; assert version("nvidia-cutlass-dsl") == "4.8.1"; print("FlashMLA", version("flash_mla"))'
WORKDIR /workspace/Megatron-LM
