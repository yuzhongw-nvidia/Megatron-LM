# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fused CuTe DSL backward WY front-half for the eat-raw KDA CP pre-process.

Backward sibling of :mod:`front_fwd`.  One kernel replaces FLA's backward glue
(``kda_gate_chunk_cumsum`` + ``chunk_kda_fwd_intra`` + ``chunk_kda_bwd_dAv``) the
``cute`` producer runs on the backward boundary window, emitting the operands the
fused CP backward scan already takes: ``qg = q*2^gk``, ``kg`` and ``w`` (the same
WY front-half as the forward), ``dv = Aqk^T @ do`` and ``gk``.  It drops in behind
``CuTeRawPreprocessProducer.final_state_grad`` where the torch
``eat_raw_ref.front_half_bwd`` oracle sits -- same inputs, same ``[1,T,HV,.]``
outputs.

STATUS: first draft, correctness-first SIMT, NOT yet trace-compiled (no cutlass
/ GPU on the authoring host) -- for the GB300 compile-iterate loop, like
:mod:`front_fwd`.  The WY half (cumsum, ``A``, ``solve_tril``, ``w``, ``kg``) is
identical to the forward; this adds ``qg`` (elementwise) and the intra-chunk
attention ``Aqk`` with ``dv = Aqk^T @ do``.

MATH (per value head hv; one 64-token chunk; log2 gate domain):

  gk[c,d]   = RCP_LN2 * cumsum_c ld[c,d]
  A/solve/w/kg                                  (identical to front_fwd)
  qg[c,d]   = q[c,d] 2^gk[c,d]
  Aqk[c,j]  = scale * sum_d q[c,d] k[j,d] 2^(gk[c,d]-gk[j,d])     (j<=c, else 0)
  dv[c,v]   = sum_{j>=c} Aqk[j,c] do[j,v]                         (= Aqk^T @ do)

``2^(gk_c-gk_j)`` for ``j<=c`` is <= 1 (direct exp2 of the difference), so Aqk is
overflow-safe without sub-chunking, like the forward A.  ``dv = Aqk^T @ do`` (the
transpose FLA folds into its chunk_kda_bwd_dAv load) is the convention verified
against autodiff of naive_chunk_kda to machine precision (test_kda_cute_eat_raw).
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

BT = 64
RCP_LN2 = 1.4426950408889634
NB_SYNC = 1


class FrontHalfBwdWS:
    """Host wrapper + SIMT kernel computing ``(qg, kg, w, dv, gk)`` from raw window operands.

    ``K == V == 128``, ``HV == H``, ``B == 1`` THD.  One instance per ``(HV, K)``.
    """

    def __init__(self, HV: int, K: int, device=None):
        assert K == 128, f"front-half kernel needs K == 128, got {K}"
        self.HV = HV
        self.K = K
        self.V = K
        self.io_dtype = cutlass.BFloat16
        self.num_threads = K
        if device is None:
            device = torch.cuda.current_device()
        self.n_ctas = torch.cuda.get_device_properties(device).multi_processor_count

    # ------------------------------------------------------------------ host
    @cute.jit
    def __call__(
        self,
        pQ: cute.Pointer,  # [T, HV, K] bf16
        pK: cute.Pointer,  # [T, HV, K] bf16
        pDo: cute.Pointer,  # [T, HV, V] bf16
        pBeta: cute.Pointer,  # [T, HV] fp32
        pLd: cute.Pointer,  # [T, HV, K] fp32 (gate log-decay, pre-cumsum)
        pQg: cute.Pointer,  # [T, HV, K] bf16 out
        pKg: cute.Pointer,  # [T, HV, K] bf16 out
        pW: cute.Pointer,  # [T, HV, K] bf16 out
        pDv: cute.Pointer,  # [T, HV, V] bf16 out
        pGk: cute.Pointer,  # [T, HV, K] fp32 out
        T: cutlass.Int32,
        scale: cutlass.Float32,
        stream,
    ):
        HV, K, V = self.HV, self.K, self.V
        f32 = cutlass.Float32
        mQ = cute.make_tensor(pQ, cute.make_layout((T, HV, K), stride=(HV * K, K, 1)))
        mK = cute.make_tensor(pK, cute.make_layout((T, HV, K), stride=(HV * K, K, 1)))
        mDo = cute.make_tensor(pDo, cute.make_layout((T, HV, V), stride=(HV * V, V, 1)))
        mBeta = cute.make_tensor(pBeta, cute.make_layout((T, HV), stride=(HV, 1)))
        mLd = cute.make_tensor(pLd, cute.make_layout((T, HV, K), stride=(HV * K, K, 1)))
        mQg = cute.make_tensor(pQg, cute.make_layout((T, HV, K), stride=(HV * K, K, 1)))
        mKg = cute.make_tensor(pKg, cute.make_layout((T, HV, K), stride=(HV * K, K, 1)))
        mW = cute.make_tensor(pW, cute.make_layout((T, HV, K), stride=(HV * K, K, 1)))
        mDv = cute.make_tensor(pDv, cute.make_layout((T, HV, V), stride=(HV * V, V, 1)))
        mGk = cute.make_tensor(pGk, cute.make_layout((T, HV, K), stride=(HV * K, K, 1)))

        @cute.struct
        class SharedStorage:
            """fp32 working tiles for one chunk of one head."""

            sq: cute.struct.MemRange[f32, BT * K]
            sk: cute.struct.MemRange[f32, BT * K]
            sdo: cute.struct.MemRange[f32, BT * V]
            sgk: cute.struct.MemRange[f32, BT * K]
            sA: cute.struct.MemRange[f32, BT * BT]  # A then (I-L)^-1 (strictly lower)
            sAqk: cute.struct.MemRange[f32, BT * BT]  # lower-incl-diag q-k attention
            sbeta: cute.struct.MemRange[f32, BT]
            srow: cute.struct.MemRange[f32, BT]

        self.shared_storage = SharedStorage

        self.front_kernel(mQ, mK, mDo, mBeta, mLd, mQg, mKg, mW, mDv, mGk, T, scale).launch(
            grid=[self.n_ctas, 1, 1], block=[self.num_threads, 1, 1], stream=stream
        )

    # ---------------------------------------------------------------- kernel
    @cute.kernel
    def front_kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mDo: cute.Tensor,
        mBeta: cute.Tensor,
        mLd: cute.Tensor,
        mQg: cute.Tensor,
        mKg: cute.Tensor,
        mW: cute.Tensor,
        mDv: cute.Tensor,
        mGk: cute.Tensor,
        T: cutlass.Int32,
        scale: cutlass.Float32,
    ):
        """One persistent-CTA item == one 64-token chunk of one value head."""
        HV, K, V = self.HV, self.K, self.V
        f32 = cutlass.Float32
        bf16 = self.io_dtype
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        d = tidx  # thread d owns column d (K == V == num_threads)

        smem = SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sq = storage.sq.get_tensor(cute.make_layout((BT, K), stride=(K, 1)))
        sk = storage.sk.get_tensor(cute.make_layout((BT, K), stride=(K, 1)))
        sdo = storage.sdo.get_tensor(cute.make_layout((BT, V), stride=(V, 1)))
        sgk = storage.sgk.get_tensor(cute.make_layout((BT, K), stride=(K, 1)))
        sA = storage.sA.get_tensor(cute.make_layout((BT, BT), stride=(BT, 1)))
        sAqk = storage.sAqk.get_tensor(cute.make_layout((BT, BT), stride=(BT, 1)))
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

            # ---- load column d (all BT rows) into smem (fp32) ----
            for c in cutlass.range(BT, unroll=1):
                tok = t0 + c
                qv = f32(0.0)
                kv = f32(0.0)
                dov = f32(0.0)
                lv = f32(0.0)
                if c < nvalid:
                    qv = mQ[tok, hv, d].to(f32)
                    kv = mK[tok, hv, d].to(f32)
                    dov = mDo[tok, hv, d].to(f32)
                    lv = mLd[tok, hv, d].to(f32)
                sq[c, d] = qv
                sk[c, d] = kv
                sdo[c, d] = dov
                sgk[c, d] = lv  # ld; cumsummed below
            if d < BT:
                bv = f32(0.0)
                if d < nvalid:
                    bv = mBeta[t0 + d, hv].to(f32)
                sbeta[d] = bv
            barrier.arrive_and_wait()

            # ---- cumsum ld -> gk (log2) + qg = q * 2^gk ----
            run = f32(0.0)
            for c in cutlass.range(BT, unroll=1):
                run = run + sgk[c, d]
                sgk[c, d] = run * f32(RCP_LN2)
                if c < nvalid:
                    mGk[t0 + c, hv, d] = sgk[c, d]
                    mQg[t0 + c, hv, d] = (sq[c, d] * cute.math.exp2(sgk[c, d], fastmath=True)).to(
                        bf16
                    )
            barrier.arrive_and_wait()

            # ---- A[c,i] = -beta_c * sum_d k[c,d] k[i,d] 2^(gk_c-gk_i), c>i (same as fwd) ----
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
            # ---- Aqk[c,j] = scale * sum_d q[c,d] k[j,d] 2^(gk_c-gk_j), j<=c, else 0 ----
            for e in cutlass.range_constexpr(BT * BT // self.num_threads):
                flat = tidx + e * self.num_threads
                c = flat // BT
                j = flat % BT
                val = f32(0.0)
                if j <= c:
                    acc = f32(0.0)
                    for dd in cutlass.range(K, unroll=1):
                        acc = acc + sq[c, dd] * sk[j, dd] * cute.math.exp2(
                            sgk[c, dd] - sgk[j, dd], fastmath=True
                        )
                    val = scale * acc
                sAqk[c, j] = val
            barrier.arrive_and_wait()

            # ---- solve_tril on sA (same as fwd) ----
            for ii in cutlass.range(BT - 1, unroll=1):
                i = ii + 1
                if d < BT:
                    srow[d] = sA[i, d]
                barrier.arrive_and_wait()
                j = d
                if j < i:
                    acc = srow[j]
                    for kk in cutlass.range(BT, unroll=1):
                        if (kk > j) & (kk < i):
                            acc = acc + srow[kk] * sA[kk, j]
                    sA[i, j] = acc
                barrier.arrive_and_wait()

            # ---- w / kg (Akk = solve + I, col i *= beta_i); no u in the backward ----
            gk_last = sgk[nvalid - 1, d]
            for c in cutlass.range(BT, unroll=1):
                if c < nvalid:
                    kd_c = sk[c, d] * cute.math.exp2(sgk[c, d], fastmath=True)
                    wv = sbeta[c] * kd_c
                    for i in cutlass.range(BT, unroll=1):
                        if i < c:
                            a_ci = sA[c, i] * sbeta[i]
                            wv = wv + a_ci * (sk[i, d] * cute.math.exp2(sgk[i, d], fastmath=True))
                    mW[t0 + c, hv, d] = wv.to(bf16)
                    mKg[t0 + c, hv, d] = (
                        sk[c, d] * cute.math.exp2(gk_last - sgk[c, d], fastmath=True)
                    ).to(bf16)

            # ---- dv[c,v] = sum_{j>=c} Aqk[j,c] * do[j,v]  (= Aqk^T @ do); thread owns column v=d
            for c in cutlass.range(BT, unroll=1):
                if c < nvalid:
                    acc = f32(0.0)
                    for j in cutlass.range(BT, unroll=1):
                        if (j >= c) & (j < nvalid):
                            acc = acc + sAqk[j, c] * sdo[j, d]
                    mDv[t0 + c, hv, d] = acc.to(bf16)
            barrier.arrive_and_wait()  # reuse smem for the next item

            item = item + self.n_ctas


@functools.lru_cache(maxsize=8)
def _wrapper(HV: int, K: int, device_index: int) -> FrontHalfBwdWS:
    return FrontHalfBwdWS(HV, K, device=device_index)


def front_half_bwd_cute(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    gate_kwargs: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused backward front-half: raw window operands -> ``(qg, kg, w, dv, gk)``.

    Drop-in for :func:`eat_raw_ref.front_half_bwd`: same inputs (THD window ``q`` /
    ``k`` ``[T, HV, K]``, raw gate ``g``, ``beta`` ``[T, HV]``, ``do`` ``[T, HV, V]``,
    ``scale``, ``gate_kwargs``) and the same ``[1, T, HV, .]`` outputs.  The gate is
    applied on the host (one elementwise); cumsum + WY + qg + Aqk/dv run in the
    kernel.
    """
    from .eat_raw_ref import _apply_gate

    t, hv, kdim = q.shape
    vdim = do.shape[-1]
    assert kdim == vdim == 128, "front-half kernel needs K == V == 128"
    device = q.device
    ld = _apply_gate(g, gate_kwargs).contiguous()
    q_c = q.contiguous()
    k_c = k.contiguous()
    do_c = do.contiguous()
    beta_c = beta.float().contiguous()
    qg = torch.empty((t, hv, kdim), dtype=torch.bfloat16, device=device)
    kg = torch.empty((t, hv, kdim), dtype=torch.bfloat16, device=device)
    w = torch.empty((t, hv, kdim), dtype=torch.bfloat16, device=device)
    dv = torch.empty((t, hv, vdim), dtype=torch.bfloat16, device=device)
    gk = torch.empty((t, hv, kdim), dtype=torch.float32, device=device)

    wrapper = _wrapper(hv, kdim, device.index)
    bf16, f32 = cutlass.BFloat16, cutlass.Float32

    def p(tensor, dtype):
        return make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)

    stream = cuda.CUstream(_raw_stream(device.index))
    wrapper(
        p(q_c, bf16),
        p(k_c, bf16),
        p(do_c, bf16),
        p(beta_c, f32),
        p(ld, f32),
        p(qg, bf16),
        p(kg, bf16),
        p(w, bf16),
        p(dv, bf16),
        p(gk, f32),
        cutlass.Int32(t),
        cutlass.Float32(scale),
        stream,
    )
    return (qg.unsqueeze(0), kg.unsqueeze(0), w.unsqueeze(0), dv.unsqueeze(0), gk.unsqueeze(0))
