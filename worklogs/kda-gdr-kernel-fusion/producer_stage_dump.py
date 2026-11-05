"""Producer-stage dump for the matches_single_rank TC parity failure.

Monkeypatches ``CuTeRawPreprocessProducer.initial_state`` so that, inside the
REAL collective CP2 run, it reports -- on the actual seq=128 model activations,
per rank -- (a) the boundary k-window's per-token L2 norms (confirm the producer
really receives L2-normalised keys), (b) the three front-half implementations
(tensor-core / SIMT / torch oracle) computed on the SAME window and their
pairwise rel-L2 + NaN/Inf, and (c) stats of the fused-scan boundary state the
producer returns.  The scan is still called exactly once (the real pipeline).

The point: front-half divergence on random driver inputs was ~1e-3 (benign), but
matches_single_rank fails at cosine 0.11.  This localises whether, on the real
data, the TC front-half itself diverges (edge case the random driver missed) or
whether the front-half agrees and the divergence is purely in the scan / collective.

Usage (coordinator, GPU, 2-rank CP2 test):

    MCORE_KDA_CUTE_RAW_FUSED_FWD=1 \
      python -c "import worklogs.kda_gdr_kernel_fusion.producer_stage_dump as d; d.install(); \
                 import pytest, sys; \
                 sys.exit(pytest.main(['-s','-x', \
                   'tests/unit_tests/ssm/gated_delta_net/test_kda_cudnn.py::test_cudnn_chunkwise_cp_matches_single_rank', \
                   '-k','sbhd-mbs1 and cute_raw']))"

Run it twice -- once as above (TC front-half) and once adding
MCORE_KDA_CUTE_RAW_SIMT=1 (SIMT front-half) -- and diff the DUMP lines.  (The
module path uses whatever import name resolves; simplest is to copy this file next
to the test or run from the worklog dir and adjust the import.)
"""
from __future__ import annotations

import sys

import torch


def _st(name, t):
    """One-line finite/stat summary of a tensor."""
    tf = t.detach().float()
    nan = int(torch.isnan(tf).sum())
    inf = int(torch.isinf(tf).sum())
    fin = tf[torch.isfinite(tf)]
    amax = float(fin.abs().max()) if fin.numel() else float("nan")
    nrm = float(fin.norm()) if fin.numel() else float("nan")
    return f"{name}: shape={tuple(t.shape)} dtype={t.dtype} norm={nrm:.4e} max={amax:.4e} nan={nan} inf={inf}"


def _rel(a, b):
    a = a.detach().float()
    b = b.detach().float()
    return (a - b).norm().item() / (b.norm().item() + 1e-30)


def install():
    """Patch the producer in place.  Call once before running the test."""
    from megatron.core.ssm.gated_delta_net.cudnn_kda.cute_preprocess import producer as P
    from megatron.core.ssm.gated_delta_net.cudnn_kda.cute_preprocess.eat_raw_ref import (
        front_half_fwd as _oracle,
    )
    from megatron.core.ssm.gated_delta_net.cudnn_kda.cute_preprocess.front_fwd import (
        front_half_fwd_cute as _simt,
    )

    try:
        from megatron.core.ssm.gated_delta_net.cudnn_kda.cute_preprocess.front_fwd_tc import (
            front_half_fwd_tc as _tc,
        )
    except Exception as exc:  # noqa: BLE001
        _tc = None
        _tc_err = repr(exc)

    def _rank():
        try:
            return torch.distributed.get_rank()
        except Exception:  # noqa: BLE001
            return 0

    def _emit(msg):
        print(f"[DUMP r{_rank()}] {msg}", file=sys.stderr, flush=True)

    def initial_state(self, k, v, g, beta, cu, cu_list, cp_context, gate_kwargs, flags):
        if torch.cuda.is_current_stream_capturing():
            raise NotImplementedError(
                "the CuTe raw CP preprocess producer cannot run under CUDA-graph capture"
            )
        chosen = self._front_half()
        P._check_envelope(k, v, flags)
        P.ensure_cp_context_metadata(cp_context)

        bos, eos = int(cu_list[-2]), int(cu_list[-1])
        kw = k[bos:eos]
        vw = v[bos:eos]
        gw = g[bos:eos]
        bw = beta[bos:eos]
        T = eos - bos
        knorm = kw.detach().float().norm(dim=-1)
        _emit(
            f"WINDOW bos={bos} eos={eos} T={T} nt={(T + 63) // 64} "
            f"|k_i| min={float(knorm.min()):.4e} max={float(knorm.max()):.4e} "
            f"chosen_front_half={getattr(chosen, '__name__', chosen)}"
        )
        _emit(_st("  k_win", kw) + " | " + _st("v_win", vw) + " | " + _st("beta_win", bw))

        # Compute all three front-halves on the identical window (local, cheap).
        refs = {}
        try:
            refs["oracle"] = _oracle(kw, vw, gw, bw, gate_kwargs)
        except Exception as exc:  # noqa: BLE001
            _emit(f"  oracle front-half RAISED {exc!r}")
        try:
            refs["simt"] = _simt(kw, vw, gw, bw, gate_kwargs)
        except Exception as exc:  # noqa: BLE001
            _emit(f"  simt front-half RAISED {exc!r}")
        if _tc is not None:
            try:
                refs["tc"] = _tc(kw, vw, gw, bw, gate_kwargs)
            except Exception as exc:  # noqa: BLE001
                _emit(f"  tc front-half RAISED {exc!r}")
        else:
            _emit(f"  tc front-half UNAVAILABLE: {_tc_err}")

        names = ("w", "u", "kg", "gk")
        for tag, tup in refs.items():
            for nm, t in zip(names, tup):
                _emit("  " + _st(f"{tag}.{nm}", t))
        if "tc" in refs and "simt" in refs:
            d = " ".join(f"{nm}={_rel(refs['tc'][i], refs['simt'][i]):.3e}" for i, nm in enumerate(names))
            _emit(f"  REL-L2 tc-vs-simt  {d}")
        if "tc" in refs and "oracle" in refs:
            d = " ".join(f"{nm}={_rel(refs['tc'][i], refs['oracle'][i]):.3e}" for i, nm in enumerate(names))
            _emit(f"  REL-L2 tc-vs-oracle {d}")
        if "simt" in refs and "oracle" in refs:
            d = " ".join(f"{nm}={_rel(refs['simt'][i], refs['oracle'][i]):.3e}" for i, nm in enumerate(names))
            _emit(f"  REL-L2 simt-vs-oracle {d}")

        # Real pipeline: use the chosen front-half output for the one real scan.
        w, u, kg, gk = chosen(kw, vw, gw, bw, gate_kwargs)
        state = P._backend.chunk_gated_delta_rule_fwd_h_pre_process_cutedsl(
            k=kg, w=w, u=u, g=None, gk=gk, bg=None, v=None,
            chunk_size=P.CHUNK_SIZE, state_v_first=False,
            cu_seqlens=cu, context=cp_context, windowed=True,
        )
        _emit(_st("SCAN state (fla layout)", state))
        if cp_context.is_first_rank:
            _emit("is_first_rank -> return None")
            return None
        out = P._ops.fla_state_to_cudnn(state, state_v_first=False)
        _emit(_st("RETURN boundary (cudnn layout)", out))
        return out

    P.CuTeRawPreprocessProducer.initial_state = initial_state
    _emit("producer_stage_dump installed")
