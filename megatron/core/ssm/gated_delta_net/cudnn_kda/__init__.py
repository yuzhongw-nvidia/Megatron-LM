# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""cuDNN backend for Kimi Delta Attention (``kda_kernel_backend="cudnn"``).

Runs the KDA chunked recurrence on cudnn-frontend's native kernels (>= 1.32.0, Blackwell
SM100+ engines) behind FLA's ``chunk_kda`` call signature, including ``linear_cp_mode="chunkwise"``
through cuDNN's per-span summary primitives.  Importing this package does not require
cudnn-frontend (its absence is probed, not an error); ``HAVE_CUDNN_KDA`` /
``cudnn_kda_unavailable_reason()`` report whether the backend can be used.
"""

from .chunk import HAVE_CUDNN_KDA, chunk_kda_cudnn, cudnn_kda_unavailable_reason
from .cp import register_boundary_state_producer

__all__ = [
    "HAVE_CUDNN_KDA",
    "chunk_kda_cudnn",
    "cudnn_kda_unavailable_reason",
    "register_boundary_state_producer",
]
