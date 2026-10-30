"""Host launcher for the SM100 fused KDA backward (development interface)."""

import threading

import torch

_CACHE = {}
_LOCK = threading.Lock()
_BT = 64


def _chunk_offsets(cu_seqlens):
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    chunks = (lengths + _BT - 1) // _BT
    return torch.cat([chunks.new_zeros(1), chunks.cumsum(0)]).to(torch.int32)


def fused_kda_bwd(
    q, k, v, gc, beta, aqk, akk, h, do, dht, cu_seqlens, scale, *, debug=False, trace=False
):
    """Run the fused KDA backward on packed inputs.

    Shapes follow ``docs/fused-gdr-to-kda-port-analysis_cc.md`` section 2.
    Returns ``(dq, dk, dv, dg, dbeta, dh0)`` and, with ``debug=True``, a sixth
    FP32 tensor of per-chunk intermediates ``[C, H, 10, 64, 128]``.  With
    ``trace=True`` the last element is an int64 ``[4 roles, 8 chunks, 16]``
    tensor of ``%globaltimer`` phase stamps from CTA 0.
    """
    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda
    from cutlass.cute.runtime import from_dlpack

    from .kernel import DEBUG_SLOTS, FusedKdaBwdKernel

    heads = q.shape[2]
    num_sequences = cu_seqlens.numel() - 1
    chunk_offsets = _chunk_offsets(cu_seqlens)
    num_chunks = int(chunk_offsets[-1])
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    dg = torch.empty_like(gc)
    dbeta = torch.empty_like(beta)
    dh0 = torch.empty_like(dht)
    debug_shape = (num_chunks, heads, len(DEBUG_SLOTS), _BT, 128) if debug else (1, 1, 1, 1, 1)
    debug_out = torch.zeros(debug_shape, device=q.device, dtype=torch.float32)
    trace_out = torch.zeros((4, 8, 16) if trace else (1, 1, 1), device=q.device, dtype=torch.int64)

    tensors = (
        q,
        k,
        v,
        gc,
        beta,
        aqk,
        akk,
        h,
        do,
        dht,
        cu_seqlens,
        chunk_offsets,
        dq,
        dk,
        dv,
        dg,
        dbeta,
        dh0,
        debug_out,
        trace_out,
    )
    args = [from_dlpack(t, assumed_align=16) for t in tensors]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    key = (heads, num_sequences, debug, trace, tuple(t.shape for t in tensors))
    with _LOCK:
        compiled = _CACHE.get(key)
        if compiled is None:
            kernel = FusedKdaBwdKernel(
                heads=heads, num_sequences=num_sequences, debug=debug, trace=trace
            )
            compiled = cute.compile(kernel, *args, cutlass.Float32(scale), stream)
            _CACHE[key] = compiled
    compiled(*args, cutlass.Float32(scale), stream)
    outputs = (dq, dk, dv, dg, dbeta, dh0)
    if debug:
        outputs += (debug_out,)
    if trace:
        outputs += (trace_out,)
    return outputs
