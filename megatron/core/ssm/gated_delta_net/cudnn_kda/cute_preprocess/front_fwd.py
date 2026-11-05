# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fused CuTe DSL forward WY front-half for the eat-raw KDA CP pre-process.

One kernel replaces FLA's forward glue (``kda_gate_chunk_cumsum`` +
``chunk_kda_fwd_intra``: chunk cumsum, intra-chunk ``A`` build, ``solve_tril``,
``recompute_w_u``) that the ``cute`` producer runs on the boundary window before
the fused CP scan launch.  It consumes the RAW window operands and emits the
operands the scan kernel already takes (``w`` / ``u`` / ``kg`` / ``gk``), so it
drops in behind ``CuTeRawPreprocessProducer`` exactly where the torch
``eat_raw_ref.front_half_fwd`` reference sits -- same oracle, same interface.

STATUS: first draft, correctness-first and NOT yet trace-compiled (the authoring
host has no CuTe DSL toolchain and no GPU; see the worklog).  It is a SIMT
(no-tensor-core) kernel: one chunk of one value head per persistent-CTA item,
every A / solve / w / u reduction done in registers over shared memory.  That is
slow but keeps the first-draft surface small -- no tcgen05 / TMEM / swizzle /
pipeline -- so the GB300 compile-iterate loop converges on the engine, then the
tensor-core + sub-chunk(16) rewrite follows (tracked in the worklog).

MATH (per value head hv; one 64-token chunk; all in the log2 gate domain):

  gk[c,d]  = RCP_LN2 * sum_{c'<=c} ld[c',d]          (chunk-local cumsum)
  A[c,i]   = -beta_c * sum_d k[c,d] k[i,d] 2^(gk[c,d]-gk[i,d])   (c>i, else 0)
  Akk      = (I - stril(A))^{-1}                      (solve_tril, col i *= beta_i)
  w[c,d]   = sum_i Akk[c,i] (k[i,d] 2^gk[i,d])
  u[c,d]   = sum_i Akk[c,i] v[i,d]
  kg[c,d]  = k[c,d] 2^(gk[last,d] - gk[c,d])

``2^(gk[c,d]-gk[i,d])`` is evaluated as one ``exp2`` of the difference, which for
``c > i`` is <= 1 -- so this SIMT build is overflow-safe directly, without the
sub-chunk(16) referencing the factorised tensor-core build needs (that strategy,
required when ``A`` is formed as ``(k 2^g) @ (k 2^-g)^T``, is kept in
``eat_raw_ref._intra_A`` for the tensor-core rewrite).
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import torch
from cutlass.cute.runtime import make_ptr

from ._host import _raw_stream
from .protocol import SmemAllocator

BT = 64  # chunk size (kernel contract)
RCP_LN2 = 1.4426950408889634  # 1 / ln(2): natural log-decay -> log2
NB_SYNC = 1  # named-barrier id for the full-CTA sync (0 is reserved for sync_threads)


class FrontHalfFwdWS:
    """Host wrapper + SIMT kernel computing ``(w, u, kg, gk)`` from raw window operands.

    ``K == V == 128`` (the cute backend envelope), ``HV == H`` (no grouped value
    attention), ``B == 1`` THD.  One instance per ``(HV, K)`` geometry; built and
    cached by :func:`front_half_fwd_cute`.
    """

    def __init__(self, HV: int, K: int, device=None):
        assert K == 128, f"front-half kernel needs K == 128, got {K}"
        self.HV = HV
        self.K = K
        self.V = K
        self.io_dtype = cutlass.BFloat16
        self.num_threads = K  # 128 threads: thread d owns key-dim column d
        if device is None:
            device = torch.cuda.current_device()
        n_sm = torch.cuda.get_device_properties(device).multi_processor_count
        self.n_ctas = n_sm

    # ------------------------------------------------------------------ host
    @cute.jit
    def __call__(
        self,
        pK: cute.Pointer,  # [T, HV, K] bf16
        pV: cute.Pointer,  # [T, HV, V] bf16
        pBeta: cute.Pointer,  # [T, HV] fp32
        pLd: cute.Pointer,  # [T, HV, K] fp32 (gate log-decay, pre-cumsum)
        pW: cute.Pointer,  # [T, HV, K] bf16 out
        pU: cute.Pointer,  # [T, HV, V] bf16 out
        pKg: cute.Pointer,  # [T, HV, K] bf16 out
        pGk: cute.Pointer,  # [T, HV, K] fp32 out
        T: cutlass.Int32,
        stream,
    ):
        HV, K, V = self.HV, self.K, self.V
        bf16 = self.io_dtype
        f32 = cutlass.Float32
        mK = cute.make_tensor(pK, cute.make_layout((T, HV, K), stride=(HV * K, K, 1)))
        mV = cute.make_tensor(pV, cute.make_layout((T, HV, V), stride=(HV * V, V, 1)))
        mBeta = cute.make_tensor(pBeta, cute.make_layout((T, HV), stride=(HV, 1)))
        mLd = cute.make_tensor(pLd, cute.make_layout((T, HV, K), stride=(HV * K, K, 1)))
        mW = cute.make_tensor(pW, cute.make_layout((T, HV, K), stride=(HV * K, K, 1)))
        mU = cute.make_tensor(pU, cute.make_layout((T, HV, V), stride=(HV * V, V, 1)))
        mKg = cute.make_tensor(pKg, cute.make_layout((T, HV, K), stride=(HV * K, K, 1)))
        mGk = cute.make_tensor(pGk, cute.make_layout((T, HV, K), stride=(HV * K, K, 1)))

        @cute.struct
        class SharedStorage:
            """fp32 working tiles for one chunk of one head."""

            sk: cute.struct.MemRange[f32, BT * K]  # keys
            sv: cute.struct.MemRange[f32, BT * V]  # values
            sgk: cute.struct.MemRange[f32, BT * K]  # log2 gate cumsum
            sA: cute.struct.MemRange[f32, BT * BT]  # A then (I-L)^-1 (strictly lower)
            sbeta: cute.struct.MemRange[f32, BT]
            srow: cute.struct.MemRange[f32, BT]  # snapshot of one A row for solve_tril

        self.shared_storage = SharedStorage

        self.front_kernel(mK, mV, mBeta, mLd, mW, mU, mKg, mGk, T).launch(
            grid=[self.n_ctas, 1, 1], block=[self.num_threads, 1, 1], stream=stream
        )

    # ---------------------------------------------------------------- kernel
    @cute.kernel
    def front_kernel(
        self,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mBeta: cute.Tensor,
        mLd: cute.Tensor,
        mW: cute.Tensor,
        mU: cute.Tensor,
        mKg: cute.Tensor,
        mGk: cute.Tensor,
        T: cutlass.Int32,
    ):
        """One persistent-CTA item == one 64-token chunk of one value head."""
        HV, K, V = self.HV, self.K, self.V
        f32 = cutlass.Float32
        bf16 = self.io_dtype
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        d = tidx  # thread d owns key/value column d (K == V == num_threads)

        smem = SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sk = storage.sk.get_tensor(cute.make_layout((BT, K), stride=(K, 1)))
        sv = storage.sv.get_tensor(cute.make_layout((BT, V), stride=(V, 1)))
        sgk = storage.sgk.get_tensor(cute.make_layout((BT, K), stride=(K, 1)))
        sA = storage.sA.get_tensor(cute.make_layout((BT, BT), stride=(BT, 1)))
        sbeta = storage.sbeta.get_tensor(cute.make_layout(BT))
        srow = storage.srow.get_tensor(cute.make_layout(BT))
        barrier = pipeline.NamedBarrier(barrier_id=NB_SYNC, num_threads=self.num_threads)

        nt = (T + (BT - 1)) // BT
        n_items = HV * nt
        item = bidx
        while item < n_items:
            hv = item % HV
            n = item // HV
            t0 = n * BT
            nvalid = cutlass.min(T - t0, BT)

            # ---- load this chunk's column d (all BT rows) into smem (fp32) ----
            # Trace-based DSL: dynamic branches are `if`/`else` blocks, not ternaries.
            for c in cutlass.range(BT, unroll=1):
                tok = t0 + c
                kv = f32(0.0)
                vv = f32(0.0)
                lv = f32(0.0)
                if c < nvalid:
                    kv = mK[tok, hv, d].to(f32)
                    vv = mV[tok, hv, d].to(f32)
                    lv = mLd[tok, hv, d].to(f32)
                sk[c, d] = kv
                sv[c, d] = vv
                sgk[c, d] = lv  # ld; cumsummed in place below
            if d < BT:
                bv = f32(0.0)
                if d < nvalid:
                    bv = mBeta[t0 + d, hv].to(f32)
                sbeta[d] = bv
            barrier.arrive_and_wait()

            # ---- chunk-local cumsum of ld -> gk (log2), per column d ----
            run = f32(0.0)
            for c in cutlass.range(BT, unroll=1):
                run = run + sgk[c, d]
                sgk[c, d] = run * f32(RCP_LN2)
                if c < nvalid:
                    mGk[t0 + c, hv, d] = sgk[c, d]  # gk output (fp32)
            barrier.arrive_and_wait()

            # ---- A[c,i] = -beta_c * sum_d k[c,d] k[i,d] 2^(gk[c,d]-gk[i,d]), c>i ----
            # 4096 entries over num_threads: thread owns entries e = tidx, tidx+NT, ...
            for e in cutlass.range_constexpr(BT * BT // self.num_threads):
                flat = tidx + e * self.num_threads
                c = flat // BT
                i = flat % BT
                val = f32(0.0)
                if c > i:
                    acc = f32(0.0)
                    for dd in cutlass.range(K, unroll=1):
                        acc = acc + sk[c, dd] * sk[i, dd] * cute.math.exp2(
                            sgk[c, dd] - sgk[i, dd], fastmath=True
                        )
                    val = -sbeta[c] * acc
                sA[c, i] = val
            barrier.arrive_and_wait()

            # ---- solve_tril: (I - stril(A))^{-1} - I, forward substitution ----
            # row i gains sum_{k in (j, i)} A_old[i,k] * A[k,j] for each j < i.
            for ii in cutlass.range(BT - 1, unroll=1):
                i = ii + 1  # rows 1 .. BT-1
                if d < BT:
                    srow[d] = sA[i, d]  # snapshot old row i
                barrier.arrive_and_wait()
                j = d
                if j < i:
                    acc = srow[j]
                    for kk in cutlass.range(BT, unroll=1):
                        if (kk > j) & (kk < i):
                            acc = acc + srow[kk] * sA[kk, j]
                    sA[i, j] = acc
                barrier.arrive_and_wait()

            # ---- w / u / kg, per column d (Akk = (solve + I), column i scaled by beta_i) ----
            gk_last = sgk[nvalid - 1, d]  # thread owns column d, so this is a local read
            for c in cutlass.range(BT, unroll=1):
                if c < nvalid:
                    kd_c = sk[c, d] * cute.math.exp2(sgk[c, d], fastmath=True)
                    wv = sbeta[c] * kd_c  # diagonal Akk[c,c] = beta_c
                    uv = sbeta[c] * sv[c, d]
                    for i in cutlass.range(BT, unroll=1):
                        if i < c:
                            a_ci = sA[c, i] * sbeta[i]  # Akk[c,i] = solve[c,i] * beta_i
                            wv = wv + a_ci * (sk[i, d] * cute.math.exp2(sgk[i, d], fastmath=True))
                            uv = uv + a_ci * sv[i, d]
                    mW[t0 + c, hv, d] = wv.to(bf16)
                    mU[t0 + c, hv, d] = uv.to(bf16)
                    mKg[t0 + c, hv, d] = (
                        sk[c, d] * cute.math.exp2(gk_last - sgk[c, d], fastmath=True)
                    ).to(bf16)
            barrier.arrive_and_wait()  # reuse smem for the next item

            item = item + self.n_ctas


@functools.lru_cache(maxsize=8)
def _wrapper(HV: int, K: int, device_index: int) -> FrontHalfFwdWS:
    return FrontHalfFwdWS(HV, K, device=device_index)


def front_half_fwd_cute(
    k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, gate_kwargs: dict
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused forward front-half: raw window operands -> ``(w, u, kg, gk)``.

    Drop-in for :func:`eat_raw_ref.front_half_fwd` (the torch oracle): same
    inputs (THD window ``k`` / ``v`` ``[T, HV, .]``, raw gate operand ``g``,
    ``beta`` ``[T, HV]``, ``gate_kwargs``) and the same ``[1, T, HV, .]`` outputs
    (``w`` / ``u`` / ``kg`` bf16, ``gk`` fp32).  The bounded/plain gate is applied
    on the host (one elementwise, not the glue this fuses); the chunk cumsum, WY
    ``A`` / ``solve_tril`` and ``w`` / ``u`` / ``kg`` run in the kernel.
    """
    from .eat_raw_ref import _apply_gate

    t, hv, kdim = k.shape
    vdim = v.shape[-1]
    assert kdim == vdim == 128, "front-half kernel needs K == V == 128"
    device = k.device
    ld = _apply_gate(g, gate_kwargs).contiguous()  # [T, HV, K] fp32 log-decay
    k_c = k.contiguous()
    v_c = v.contiguous()
    beta_c = beta.float().contiguous()
    w = torch.empty((t, hv, kdim), dtype=torch.bfloat16, device=device)
    u = torch.empty((t, hv, vdim), dtype=torch.bfloat16, device=device)
    kg = torch.empty((t, hv, kdim), dtype=torch.bfloat16, device=device)
    gk = torch.empty((t, hv, kdim), dtype=torch.float32, device=device)

    wrapper = _wrapper(hv, kdim, device.index)
    bf16, f32 = cutlass.BFloat16, cutlass.Float32

    def p(tensor, dtype):
        return make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)

    stream = cuda.CUstream(_raw_stream(device.index))
    wrapper(
        p(k_c, bf16),
        p(v_c, bf16),
        p(beta_c, f32),
        p(ld, f32),
        p(w, bf16),
        p(u, bf16),
        p(kg, bf16),
        p(gk, f32),
        cutlass.Int32(t),
        stream,
    )
    return (w.unsqueeze(0), u.unsqueeze(0), kg.unsqueeze(0), gk.unsqueeze(0))
