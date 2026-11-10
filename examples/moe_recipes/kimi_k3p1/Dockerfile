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
# This is a separate image rather than a change to the Kimi-K3 one. The cudnn-frontend
# revision below needs CuTe DSL >= 4.8, and DSL 4.8 removed cute.make_fragment, which two
# things the Kimi-K3 recipes run still call: FlashAttention-4 4.0.0b11 (MLA latent CP) and
# Megatron's fused linear cross-entropy kernels. Kimi-K3.1 runs neither of them.
ARG KIMI_K3_IMAGE=megatron-kimi-k3:blackwell
FROM ${KIMI_K3_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# TODO: temporary pin. This commit is the head of NVIDIA/cudnn-frontend#2474, which adds
# the SM100 sparse-attention forward for 96 heads that Kimi-K3.1 needs. Once that PR is
# merged and cudnn-frontend has released it, replace this commit with that release.
ARG CUDNN_FRONTEND_COMMIT=e55745ffce9371f082b9dad452f74d5807cdb0e6
# FlashMLA nv_dev. The cuDNN DSA backend requires it and runs its sparse forward for 64
# heads and multiples of 128; other head counts go to cudnn-frontend.
ARG FLASH_MLA_COMMIT=b7643bd54521f563b839b98289b5cd048c062ba2

# cudnn-frontend at this revision requires nvidia-cutlass-dsl >= 4.8 and builds with
# nanobind instead of pybind11.
RUN unset PIP_CONSTRAINT && python -m pip install --no-cache-dir \
        'nanobind>=3.0.1,<4' 'nvidia-cutlass-dsl[cu13]==4.8.1'

RUN unset PIP_CONSTRAINT && python -m pip install --no-cache-dir \
        --no-build-isolation \
        "nvidia-cudnn-frontend[cutedsl] @ git+https://github.com/NVIDIA/cudnn-frontend.git@${CUDNN_FRONTEND_COMMIT}"

# CuTe DSL 4.8 moved ThrMma and ThrCopy from cutlass.cute.core to cutlass.cute.
# FlashAttention-4 4.0.0b11 and quack-kernels 0.4.1 still name the old location, and
# TransformerEngine imports flash_attn.cute, so `import transformer_engine` fails until
# the references are rewritten. Kimi-K3.1 only needs these two packages to import.
RUN SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" && \
    grep -rlE 'cute\.core\.(ThrMma|ThrCopy)\b' \
        "${SITE}/flash_attn/cute" "${SITE}/quack" --include='*.py' | \
        xargs sed -i -E 's/\bcute\.core\.(ThrMma|ThrCopy)\b/cute.\1/g' && \
    ! grep -rqE 'cute\.core\.(ThrMma|ThrCopy)\b' \
        "${SITE}/flash_attn/cute" "${SITE}/quack" --include='*.py'

# SM100 only; the SM90 kernels are not built.
RUN git clone https://github.com/deepseek-ai/FlashMLA.git /opt/FlashMLA && \
    cd /opt/FlashMLA && git checkout "${FLASH_MLA_COMMIT}" && \
    git submodule update --init --recursive && \
    unset PIP_CONSTRAINT && \
    FLASH_MLA_DISABLE_SM90=1 NVCC_THREADS=16 \
    CFLAGS="-I/usr/local/cuda/include/cccl" CXXFLAGS="-I/usr/local/cuda/include/cccl" \
    python -m pip install --no-build-isolation --no-cache-dir .

RUN python -m pip check && \
    python -c 'from importlib.metadata import version; assert version("nvidia-cutlass-dsl") == "4.8.1", version("nvidia-cutlass-dsl"); assert version("flash-attn-4") == "4.0.0b11"; print("cudnn-frontend", version("nvidia-cudnn-frontend"), "| flash_mla", version("flash_mla"))' && \
    rm -rf /root/.cache/pip
WORKDIR /workspace/Megatron-LM
