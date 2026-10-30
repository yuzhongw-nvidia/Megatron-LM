# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Derived from flash-linear-attention; see this package's __init__.py for the
# inline MIT license notice and upstream contributor link.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Upstream contributors:
# https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# The MIT notice above covers the FLA-derived dispatcher and engines (backend.py, _host.py,
# protocol.py, symm.py, fused_ws.py, bwd_ws.py); this module, producer.py and
# context_metadata.py are NVIDIA code.

"""Optional CuTe DSL boundary-state producer for the cuDNN KDA chunkwise-CP path.

Ported from NVIDIA/Megatron-LM PR #6511 (``internal_gdn_backend/kernels/fused_gdr_cp_cute``):
warp-specialised tcgen05 (SM100) kernels that compute the cross-rank boundary states of a
chunkwise context-parallel step and exchange them through PyTorch symmetric memory in a single
launch per direction, in place of cuDNN's per-span summaries + ``all_gather`` + transition fold
(``cp.CudnnSummaryProducer``).  :func:`register` makes it selectable as
``cp_state_producer="cute"``; the cuDNN backend never imports this package on its own, and
importing it imports neither cutlass nor flash-linear-attention.

Toolchain caveat: the kernels were written against nvidia-cutlass-dsl 4.5.x, and importing them
under 4.8.x fails (``cutlass.cute.core.ThrMma`` is gone).  cudnn-frontend 1.32 -- the release the
cuDNN KDA backend requires -- needs the DSL >= 4.7, so in an environment where the cuDNN backend
runs, this producer cannot run until the kernels are ported to the newer DSL.  It is shipped as
an optional, experimental optimisation; :func:`unavailable_reason` names the blocker.

Envelope (``backend._is_supported``): SM100-class GPU (compute capability 10.x), NCCL CP group of
size 2..8, one packed sequence (``B == 1``), ``K == V == 128``, ``HV % 4 == 0`` and ``HV == H``,
bf16 operands with the fp32 gate cumsum, chunk size 64, no CUDA-graph capture (the producer
refuses at capture time), and the in-kernel q/k L2-norm / beta-sigmoid / negative-eigenvalue
options unset.  Anything else raises ``NotImplementedError`` -- there is no silent fallback.
"""

from __future__ import annotations

import functools
from typing import Optional

PRODUCER_NAME = "cute"

__all__ = ["PRODUCER_NAME", "is_available", "register", "unavailable_reason"]


@functools.lru_cache(maxsize=1)
def _probe() -> Optional[str]:
    """``None`` when the producer can run here, else a human-readable reason (cached)."""
    from . import backend as _backend

    if not _backend._cutlass_runtime_available():
        return (
            "nvidia-cutlass-dsl is not importable (needs cutlass.cute, cutlass.pipeline, "
            "cutlass.utils.blackwell_helpers)"
        )
    if not _backend._has_module("cuda.bindings.driver"):
        return "cuda-python (cuda.bindings.driver) is not installed"
    if not _backend._has_module("torch.distributed._symmetric_memory"):
        return "this torch build has no torch.distributed._symmetric_memory"
    if not _backend._has_module("fla.ops.kda"):
        return "flash-linear-attention (fla.ops.kda) is not installed"
    try:
        import cutlass

        version = getattr(cutlass, "__version__", "unknown")
    except Exception as e:  # pragma: no cover - environment dependent
        return f"import cutlass failed: {e!r}"
    try:
        from . import bwd_ws, fused_ws  # noqa: F401  (imports the engines)
    except Exception as e:  # pragma: no cover - environment dependent
        return (
            f"the CuTe CP engines do not import under nvidia-cutlass-dsl {version} (they were "
            f"written against 4.5.x; 4.8.x removed cutlass.cute.core.ThrMma): {e!r}"
        )
    import torch

    if not torch.cuda.is_available():
        return "CUDA is not available"
    major, minor = torch.cuda.get_device_capability()
    if major != 10:
        return f"needs an SM100-class GPU (compute capability 10.x), found {major}.{minor}"
    return None


def unavailable_reason() -> Optional[str]:
    """``None`` when the CuTe producer can be used on this process, else why not."""
    return _probe()


def is_available() -> bool:
    """``True`` when the CuTe producer can run on this process (see :func:`unavailable_reason`)."""
    return _probe() is None


def register() -> None:
    """Register :class:`producer.CuTePreprocessProducer` as the ``"cute"`` boundary-state producer.

    After this, ``chunk_kda_cudnn(..., cp_state_producer="cute")`` (or
    ``cp.chunk_kda_cudnn_cp(..., producer="cute")``) uses the fused CuTe pre-process.  The producer
    is constructed per call and raises with :func:`unavailable_reason` when it cannot run.
    """
    from megatron.core.ssm.gated_delta_net.cudnn_kda.cp import register_boundary_state_producer

    from .producer import CuTePreprocessProducer

    register_boundary_state_producer(PRODUCER_NAME, CuTePreprocessProducer)
