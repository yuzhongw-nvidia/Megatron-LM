# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tensor-core forward WY front-half for the eat-raw KDA CP pre-process.

The SIMT :mod:`front_fwd` kernel is correct (passed CP2 parity on GB300) but slow
by construction -- every A / solve / w / u reduction is a register loop, so the
batched-torch and SIMT paths both regress KDA step time (+65% vs FLA at the GDR
level).  This kernel does the two heaviest reductions -- the ``solve_tril``
(6 squarings) and ``w`` / ``u`` -- on the tensor cores, through the one
both-operands-from-smem SM80 warp-MMA gemm validated in isolation against torch
(worklog ``gemm_ss_probe.py``: A@B^T / A@B, (64,64,64)/(64,64,128)/(64,128,64),
rel-L2 ~3e-8 on Blackwell).  The intra-chunk ``A`` build stays SIMT for now (it is
overflow-safe via the direct ``exp2`` of the gate difference and needs the
sub-chunk-referenced block gemms to tensor-core-ise; that is the next step).

Numerics: bf16 operands, fp32 MMA accumulate.  The high solve powers (``A^4``
onward, entries ``<1e-4``) are ~``I`` so the 6-factor Neumann product
``(I-A)^{-1} = prod_i (I + A^(2^i))`` keeps its precision from the first two
factors; if CP2 parity needs more, the bf16 hi/lo 3-pass (``protocol._fold_gemms``)
drops in per gemm.

Standard matmuls (``p@p``, ``inv@(I+p)``, ``Akk@X``) feed the MMA's ``A@B^T`` with
the second operand's transpose MATERIALISED contiguous (SIMT), staying on the
``transpose=False`` ldmatrix path the probe validated (not a strided view).

Selected by ``MCORE_KDA_CUTE_RAW_FUSED_FWD=1`` (the SIMT kernel stays available
under ``MCORE_KDA_CUTE_RAW_SIMT=1``); drop-in for ``eat_raw_ref.front_half_fwd``.
First GPU-assembly draft: the gemm primitive is validated, the assembly (smem
chaining / transposes / barriers) is for the compile-iterate loop.  Each gemm's
output is anchored to ``tc_gemm_sim.py``.
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import torch
from cutlass.cute.nvgpu import warp
from cutlass.cute.runtime import make_ptr

from ._host import _raw_stream
from .protocol import SmemAllocator

BT = 64
RCP_LN2 = 1.4426950408889634
NB_SYNC = 1
F32 = cutlass.Float32


class FrontHalfFwdTC:
    """Host wrapper + tensor-core kernel: ``(w, u, kg, gk)`` from raw window operands.

    ``K == V == 128``, ``HV == H``, ``B == 1`` THD.  One instance per ``(HV, K)``.
    """

    def __init__(self, HV: int, K: int, device=None):
        assert K == 128, f"front-half kernel needs K == 128, got {K}"
        self.HV = HV
        self.K = K
        self.V = K
        self.io_dtype = cutlass.BFloat16
        self.num_threads = 128  # 4 warps for the SM80 tiled MMA; also K == 128 for the SIMT phases
        if device is None:
            device = torch.cuda.current_device()
        self.n_ctas = torch.cuda.get_device_properties(device).multi_processor_count

    # ------------------------------------------------------------------ host
    @cute.jit
    def __call__(
        self,
        pK: cute.Pointer,
        pV: cute.Pointer,
        pBeta: cute.Pointer,
        pLd: cute.Pointer,
        pW: cute.Pointer,
        pU: cute.Pointer,
        pKg: cute.Pointer,
        pGk: cute.Pointer,
        T: cutlass.Int32,
        stream,
    ):
        HV, K, V = self.HV, self.K, self.V
        bf16 = self.io_dtype
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
            """fp32 load/cumsum tiles + bf16 matrix scratch for the MMA chain."""

            sk: cute.struct.MemRange[F32, BT * K]
            sv: cute.struct.MemRange[F32, BT * V]
            sgk: cute.struct.MemRange[F32, BT * K]
            sbeta: cute.struct.MemRange[F32, BT]
            # bf16 [BT,BT] matrices (A / power / inverse / I+p / transpose scratch)
            sAb: cute.struct.Align[cute.struct.MemRange[bf16, BT * BT], 1024]
            sP: cute.struct.Align[cute.struct.MemRange[bf16, BT * BT], 1024]
            sPn: cute.struct.Align[cute.struct.MemRange[bf16, BT * BT], 1024]
            sInv: cute.struct.Align[cute.struct.MemRange[bf16, BT * BT], 1024]
            sIp: cute.struct.Align[cute.struct.MemRange[bf16, BT * BT], 1024]
            sTt: cute.struct.Align[cute.struct.MemRange[bf16, BT * BT], 1024]
            # bf16 [BT,K] operand/result tiles for w/u (sXb beta-scaled X, sScr gemm result,
            # sXbt the materialised transpose [K,BT])
            sXb: cute.struct.Align[cute.struct.MemRange[bf16, BT * K], 1024]
            sScr: cute.struct.Align[cute.struct.MemRange[bf16, BT * K], 1024]
            sXbt: cute.struct.Align[cute.struct.MemRange[bf16, K * BT], 1024]

        self.shared_storage = SharedStorage
        self.front_kernel(mK, mV, mBeta, mLd, mW, mU, mKg, mGk, T).launch(
            grid=[self.n_ctas, 1, 1], block=[self.num_threads, 1, 1], stream=stream
        )

    # ------------------------------------------------- the validated gemm_ss
    @cute.jit
    def _gemm_ss(self, tiled_mma, tc_a, tc_b, sA, sB, M: cutlass.Constexpr, N: cutlass.Constexpr):
        """acc[M,N] = sA[M,Kc] @ sB[N,Kc]^T, both from smem (worklog gemm_ss_probe)."""
        tidx, _, _ = cute.arch.thread_idx()
        thr_mma = tiled_mma.get_slice(tidx)
        acc = cute.make_rmem_tensor(thr_mma.partition_shape_C((M, N)), F32)
        for i in cutlass.range_constexpr(cute.size(acc)):
            acc[i] = F32(0.0)
        thr_a = tc_a.get_slice(tidx)
        thr_b = tc_b.get_slice(tidx)
        sA_src = thr_a.partition_S(sA)
        sB_src = thr_b.partition_S(sB)
        rA = thr_mma.make_fragment_A(thr_mma.partition_A(sA))
        rB = thr_mma.make_fragment_B(thr_mma.partition_B(sB))
        rA_cv = thr_a.retile(rA)
        rB_cv = thr_b.retile(rB)
        nkt = cute.size(rA, mode=[2])
        for kt in cutlass.range_constexpr(nkt):
            cute.copy(tc_a, sA_src[None, None, kt], rA_cv[None, None, kt])
            cute.copy(tc_b, sB_src[None, None, kt], rB_cv[None, None, kt])
            cute.gemm(tiled_mma, acc, rA[None, None, kt], rB[None, None, kt], acc)
        return acc, thr_mma

    # ---------------------------------------------------------------- kernel
    @cute.kernel
    def front_kernel(self, mK, mV, mBeta, mLd, mW, mU, mKg, mGk, T: cutlass.Int32):
        """One persistent-CTA item == one 64-token chunk of one value head."""
        HV, K, V = self.HV, self.K, self.V
        f32 = F32
        bf16 = self.io_dtype
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        d = tidx

        smem = SmemAllocator()
        st = smem.allocate(self.shared_storage)
        sk = st.sk.get_tensor(cute.make_layout((BT, K), stride=(K, 1)))
        sv = st.sv.get_tensor(cute.make_layout((BT, V), stride=(V, 1)))
        sgk = st.sgk.get_tensor(cute.make_layout((BT, K), stride=(K, 1)))
        sbeta = st.sbeta.get_tensor(cute.make_layout(BT))
        sAb = st.sAb.get_tensor(cute.make_layout((BT, BT), stride=(BT, 1)))
        sP = st.sP.get_tensor(cute.make_layout((BT, BT), stride=(BT, 1)))
        sPn = st.sPn.get_tensor(cute.make_layout((BT, BT), stride=(BT, 1)))
        sInv = st.sInv.get_tensor(cute.make_layout((BT, BT), stride=(BT, 1)))
        sIp = st.sIp.get_tensor(cute.make_layout((BT, BT), stride=(BT, 1)))
        sTt = st.sTt.get_tensor(cute.make_layout((BT, BT), stride=(BT, 1)))
        sXb = st.sXb.get_tensor(cute.make_layout((BT, K), stride=(K, 1)))
        sScr = st.sScr.get_tensor(cute.make_layout((BT, K), stride=(K, 1)))
        sXbt = st.sXbt.get_tensor(cute.make_layout((K, BT), stride=(BT, 1)))
        bar = pipeline.NamedBarrier(barrier_id=NB_SYNC, num_threads=self.num_threads)

        # one SM80 tiled MMA + ldmatrix tiled copies serve every gemm (shape-general)
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(bf16, f32, (16, 8, 16)), (4, 1, 1), permutation_mnk=(64, 16, 16)
        )
        ld_atom = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), bf16)
        tc_a = cute.make_tiled_copy_A(ld_atom, tiled_mma)
        tc_b = cute.make_tiled_copy_B(ld_atom, tiled_mma)

        nt = (T + (BT - 1)) // BT
        n_items = HV * nt
        item = bidx
        while item < n_items:
            hv = item % HV
            n = item // HV
            t0 = n * BT
            nvalid = cutlass.min(T - t0, BT)

            # ---- load col d, cumsum ld -> gk, store gk  [SIMT, from front_fwd] ----
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
                sgk[c, d] = lv
            if d < BT:
                bv = f32(0.0)
                if d < nvalid:
                    bv = mBeta[t0 + d, hv].to(f32)
                sbeta[d] = bv
            bar.arrive_and_wait()
            run = f32(0.0)
            for c in cutlass.range(BT, unroll=1):
                run = run + sgk[c, d]
                sgk[c, d] = run * f32(RCP_LN2)
                if c < nvalid:
                    mGk[t0 + c, hv, d] = sgk[c, d]
            bar.arrive_and_wait()

            # ---- A build (SIMT direct, overflow-safe) -> sAb (bf16, strictly lower) ----
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
                sAb[c, i] = val.to(bf16)
            bar.arrive_and_wait()

            # ---- solve: sInv = (I-A)^{-1} (full inverse) via 6 squarings [tensor-core] ----
            # sInv = I + A ; sP = A.  The [BT,BT] matrices have only BT (64) columns, so
            # their column-d SIMT ops must be guarded to threads d < BT (threads BT..2BT-1
            # still join every barrier + every gemm, which needs all 128 for the 4 warps).
            if d < BT:
                for c in cutlass.range(BT, unroll=1):
                    sP[c, d] = sAb[c, d]
                    sInv[c, d] = sAb[c, d]
                sInv[d, d] = (sInv[d, d].to(f32) + f32(1.0)).to(bf16)  # +I on the diagonal
            bar.arrive_and_wait()
            for _it in cutlass.range_constexpr(5):
                # sPn = sP @ sP : materialise sP^T (contiguous) then gemm_ss(sP, sP^T)
                if d < BT:
                    for c in cutlass.range(BT, unroll=1):
                        sTt[d, c] = sP[c, d]  # sTt = sP^T
                bar.arrive_and_wait()
                acc, thr_mma = self._gemm_ss(tiled_mma, tc_a, tc_b, sP, sTt, BT, BT)
                tC = thr_mma.partition_C(sPn)
                for i in cutlass.range_constexpr(cute.size(acc)):
                    tC[i] = acc[i].to(bf16)
                bar.arrive_and_wait()
                # sIp = I + sPn ; sTt = sIp^T
                if d < BT:
                    for c in cutlass.range(BT, unroll=1):
                        sIp[c, d] = sPn[c, d]
                    sIp[d, d] = (sIp[d, d].to(f32) + f32(1.0)).to(bf16)
                bar.arrive_and_wait()
                if d < BT:
                    for c in cutlass.range(BT, unroll=1):
                        sTt[d, c] = sIp[c, d]
                bar.arrive_and_wait()
                # sInv = sInv @ (I + sPn)
                acc, thr_mma = self._gemm_ss(tiled_mma, tc_a, tc_b, sInv, sTt, BT, BT)
                tC = thr_mma.partition_C(sInv)
                for i in cutlass.range_constexpr(cute.size(acc)):
                    tC[i] = acc[i].to(bf16)
                bar.arrive_and_wait()
                # sP <- sPn
                if d < BT:
                    for c in cutlass.range(BT, unroll=1):
                        sP[c, d] = sPn[c, d]
                bar.arrive_and_wait()
            # sInv is now the full inverse (I - A)^{-1}

            # ---- w = sInv @ Xb ; u = sInv @ Vb  [tensor-core], kg [SIMT] ----
            gk_last = sgk[nvalid - 1, d]
            # Xb[i,d] = beta_i * k[i,d] * 2^gk[i,d]
            for c in cutlass.range(BT, unroll=1):
                sXb[c, d] = (sbeta[c] * sk[c, d] * cute.math.exp2(sgk[c, d], fastmath=True)).to(
                    bf16
                )
            for c in cutlass.range(BT, unroll=1):
                sXbt[d, c] = sXb[c, d]  # sXbt[K,BT] = Xb^T
            bar.arrive_and_wait()
            acc, thr_mma = self._gemm_ss(tiled_mma, tc_a, tc_b, sInv, sXbt, BT, K)
            tC = thr_mma.partition_C(sScr)
            for i in cutlass.range_constexpr(cute.size(acc)):
                tC[i] = acc[i].to(bf16)
            bar.arrive_and_wait()
            for c in cutlass.range(BT, unroll=1):
                if c < nvalid:
                    # sInv is the FULL inverse, so w = Akk @ X = sInv @ Xb (no extra +Xb).
                    mW[t0 + c, hv, d] = sScr[c, d]
                    mKg[t0 + c, hv, d] = (
                        sk[c, d] * cute.math.exp2(gk_last - sgk[c, d], fastmath=True)
                    ).to(bf16)
            bar.arrive_and_wait()
            # u: Vb[i,d] = beta_i * v[i,d]
            for c in cutlass.range(BT, unroll=1):
                sXb[c, d] = (sbeta[c] * sv[c, d]).to(bf16)
            for c in cutlass.range(BT, unroll=1):
                sXbt[d, c] = sXb[c, d]
            bar.arrive_and_wait()
            acc, thr_mma = self._gemm_ss(tiled_mma, tc_a, tc_b, sInv, sXbt, BT, V)
            tC = thr_mma.partition_C(sScr)
            for i in cutlass.range_constexpr(cute.size(acc)):
                tC[i] = acc[i].to(bf16)
            bar.arrive_and_wait()
            for c in cutlass.range(BT, unroll=1):
                if c < nvalid:
                    mU[t0 + c, hv, d] = sScr[c, d]  # u = sInv @ Vb
            bar.arrive_and_wait()

            item = item + self.n_ctas


@functools.lru_cache(maxsize=8)
def _wrapper(HV: int, K: int, device_index: int) -> FrontHalfFwdTC:
    return FrontHalfFwdTC(HV, K, device=device_index)


def front_half_fwd_tc(
    k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, gate_kwargs: dict
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tensor-core forward front-half: raw window operands -> ``(w, u, kg, gk)``.

    Drop-in for :func:`eat_raw_ref.front_half_fwd` / :func:`front_fwd.front_half_fwd_cute`.
    The gate is applied on the host; cumsum + A + solve + w/u run in the kernel
    (solve and w/u on the tensor cores).
    """
    from .eat_raw_ref import _apply_gate

    t, hv, kdim = k.shape
    vdim = v.shape[-1]
    assert kdim == vdim == 128, "front-half kernel needs K == V == 128"
    device = k.device
    ld = _apply_gate(g, gate_kwargs).contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    beta_c = beta.float().contiguous()
    w = torch.empty((t, hv, kdim), dtype=torch.bfloat16, device=device)
    u = torch.empty((t, hv, vdim), dtype=torch.bfloat16, device=device)
    kg = torch.empty((t, hv, kdim), dtype=torch.bfloat16, device=device)
    gk = torch.empty((t, hv, kdim), dtype=torch.float32, device=device)

    wrapper = _wrapper(hv, kdim, device.index)
    bf16, f32 = cutlass.BFloat16, F32

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
    return w.unsqueeze(0), u.unsqueeze(0), kg.unsqueeze(0), gk.unsqueeze(0)
