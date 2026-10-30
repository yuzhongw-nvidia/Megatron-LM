# Fused KDA backward ported from the fused GDR backward: design and plan

## Status and scope

- Worktree `.worktrees/fused-kda-bwd-cc`, branch `codex/fused-kda-bwd-cc`, based on
  `yuzhongw-nvidia:yuzhongw/internal-gdr-backend` at `b5416e299`.
  That base carries the SM100 fused GDR backward
  (`internal_gdn_backend/kernels/fused_gdr_bwd_cute`) but not the KDA module or adapter;
  KDA integration is therefore a later milestone that depends on rebasing onto, or merging,
  the KDA adapter branch.
- This document supersedes the plan section of
  [the earlier analysis](fused-gdr-to-kda-port-analysis.md) for this branch and resolves the
  three blocking items of [its review](fused-gdr-to-kda-port-analysis-claude-review.md)
  (anchor safety, resource ledger, schedule wording). Those two files live in the older
  `fused-kda-bwd` worktree; they are referenced, not copied.
- Done so far: an executable per-chunk specification
  (`tests/unit_tests/ssm/kda_fused_bwd_reference.py`) with CPU tests against an independent
  FP64 autograd oracle, and a precision study on that specification. No device code exists
  yet; no GPU accuracy or performance number in this document is a kernel result.

## 1. Decisions

| Topic | Decision |
| --- | --- |
| Packaging | New package `kernels/fused_kda_bwd_cute`, starting from copies of the GDR `launcher.py`, `layouts.py`, `storage.py`, `tcgen05_ws.py` and the pipeline/role helpers of `kernel.py`. GDR files are not modified. |
| Inputs | Consume the FLA forward residuals: BF16 `Aqk` and `Akk`, BF16 chunk states `h`, FP32 log2 chunk-local gate cumsum `gc`. The GDR operations `p` (recompute QKᵀ) and `at` (KKᵀ gram) disappear. |
| Gate gradient | Output `dg` is FLA's `dg` after its chunk-local reverse cumsum (natural-log units). The reverse cumsum is fused; `kda_gate_bwd` (A_log/dt_bias) stays outside. |
| Orientation | Every token × key-channel result (dq, dk, intra partials) is accumulated channel-major: M = 128 key channels on TMEM lanes, N = 64 tokens. Per-channel factors, the diagonal fallback and the dg reverse cumsum become per-thread work. Value-side tiles (X, dv_new, v_new, D) stay token-major as in GDR. |
| Intra anchors | 16-token blocks. Lower products anchor at a block's last row, upper products at its first row; off-diagonal factors are ≤ 1 for any gate. Diagonal blocks reuse the same MMA and fall back per channel to exact evaluation when the block span exceeds 96 (log2). |
| Precision | BF16 MMA operands everywhere except the anchored intra token operands, which are BF16 high/low pairs (two MMAs). Justified by the study in §5. |
| Gate tile | FP32 `gc` lives in TMEM (64 columns, channel-major), not shared memory. |

## 2. Interface contract

Low-level entry (packed THD, mirroring `fused_gdr_bwd`):

| Tensor | Shape / dtype | Meaning |
| --- | --- | --- |
| `q`, `k`, `v`, `do` | `[1, N, H, 128]` BF16 | token inputs and output gradient |
| `gc` | `[1, N, H, 128]` FP32 | chunk-local inclusive cumsum of the natural-log gate × log2(e) (FLA's `g` inside `chunk_kda_bwd`) |
| `beta` | `[1, N, H]` FP32 | |
| `aqk` | `[1, N, H, 64]` BF16 | FLA `Aqk`: causal (diagonal included) gated QKᵀ, **already multiplied by `scale`** |
| `akk` | `[1, N, H, 64]` BF16 | FLA `Akk = (I + L)⁻¹` with `L_ij = β_i Σ_d k_id k_jd 2^(G_id−G_jd)` (strictly lower); β enters through `w = Akk (k⊙β⊙Γ)`, `u = Akk (v⊙β)` |
| `h` | `[1, C, H, 128, 128]` BF16 | chunk-start states, `[K, V]` layout, global chunk order |
| `dht` | `[B, H, 128, 128]` FP32 | final-state gradient |
| `cu_seqlens` | `[B + 1]` int32 | positive lengths; tails need not be multiples of 64 |

Outputs: `dq`, `dk`, `dv` BF16 `[1, N, H, 128]`; `dg` FP32 `[1, N, H, 128]` (post reverse cumsum);
`dbeta` FP32 `[1, N, H]`; `dh0` FP32 `[B, H, 128, 128]`. The adapter decides whether `dh0`
is returned to autograd; the kernel always produces it, which does not change Megatron's
autograd interface by itself (review item P2-3).

Tail chunks extend `gc` from the last valid row and zero invalid token rows before use, as in
GDR; every store is predicated by the valid row count.

## 3. Per-chunk mathematics and operation map

Notation per chunk (64 rows, reverse order): `G` = `gc` tile, `Γ = 2^G`, `G_L` = last valid row,
`S` = chunk-start state, `dS` = gradient of the chunk-end state (initially `dht`). All exponents
below are differences `G_a − G_b` with `a` at or after `b`, hence ≤ 0, except where noted.
`reference.py` implements exactly this list and is the normative definition.

1. Operand tiles (BF16): `kΓ = k⊙Γ`, `kgL = k⊙2^(G_L−G)`, `qΓs = scale·q⊙Γ`.
2. `X = kΓ S`; `Δ = v − X` (BF16, replaces v in place); `vn = (Akk·diag β) Δ`.
3. `dvn = kgL dS + Aqkᵀ dO`.
4. `D = Akkᵀ dvn`; `dv = β⊙D`; `dβ += rowsum(D⊙Δ)`.
5. `dAqk = scale · dO vnᵀ` (causal, valid); `dAkk = (dvn Δᵀ)·diag β` (strictly lower);
   `dL = −Akkᵀ dAkk Akkᵀ` (strictly lower).
6. State-path gradients: `dq_s = Γ⊙(scale·dO Sᵀ)`, `dk_s = 2^(G_L−G)⊙(vn dSᵀ)`,
   `dk_w = −Γ⊙((β⊙D) Sᵀ)`.
7. Intra: `dq_i += Σ_{j≤i} dAqk_ij k_j 2^(G_i−G_j)`; `R_i = Σ_{j<i} dL_ij k_j 2^(G_i−G_j)`,
   `dk_row = β⊙R`, `dβ += Σ_d k⊙R`; `dk_col_j = Σ_{i≥j} (dAqk_ij q_i + dL_ij β_i k_i) 2^(G_i−G_j)`.
8. `dg = q⊙dq + k⊙(dk_w + dk_row − dk_col − dk_s)`; row `G_L` additionally receives
   `Σ_i k⊙dk_s + Γ_L⊙Σ_v (S⊙dS)`; then chunk-local reverse cumsum.
9. `dS ← Γ_L⊙dS + qΓsᵀ dO − (k⊙β⊙Γ)ᵀ D`.

Operation table (A × B → accumulator). "ch-major" = M 128 channels × N 64 tokens; "tok" = M 64
tokens × N 128 values packed in 64 TMEM columns; "sq" = 64 × 64.

| KDA op | Math | Orientation | GDR origin | Change |
| --- | --- | --- | --- | --- |
| `x_state` | `kΓ · S` | tok | `u_state` | A becomes the scaled tile `kΓ` |
| `dvn_state` | `kgL · dS` | tok | `dvprime_state` | gate moves from a row epilogue into the A operand |
| `dvn_aqk` | `Aqkᵀ · dO` | tok | `dvprime_pg` | A is the loaded `Aqk`, no P-gamma epilogue |
| `vn` | `(Akk·diag β) · Δ` | tok | `vprime` | β folded into the 64×64 operand |
| `d` | `Akkᵀ · dvn` | tok | `dv` | unchanged shape |
| `dakk` | `dvn · Δᵀ` | sq | `dag` | epilogue: β column scale, strict mask |
| `daqk` | `dO · vnᵀ` | sq | `dpg` | epilogue: scale, causal mask |
| `inv_left`, `inv_right` | `−Akkᵀ dAkk Akkᵀ` | sq | `da_left`, `da_right` | unchanged |
| `dq_state` | `S · dOᵀ` | ch-major | `dq_state` | transposed; epilogue `·Γ·scale` per element |
| `dk_state` | `dS · vnᵀ` | ch-major | `dk_state` | transposed; epilogue `·2^(G_L−G)` |
| `dk_w` | `S · Dᵀ` | ch-major | `dk_state_g` | separate ring slot; epilogue `·(−βΓ)` |
| `dh_k` | `(kβΓ)ᵀ · D` | M128 × N128 | `dh_k` | gate on the K side (A), raw D on the V side |
| `dh_q` | `(qΓs)ᵀ · dO` | M128 × N128 | `dh_q` | gate on the K side; raw `dO` replaces `dO·γ` |
| `intra_lower_q[J]` | `k̃ᵀ[:,J] · dAqk[:,J]ᵀ` | ch-major, K = 16 | `dq_dp` | per-block anchor; hi/lo pair |
| `intra_lower_k[J]` | `k̃ᵀ[:,J] · dL[:,J]ᵀ` | ch-major, K = 16 | `dk_da` (part) | per-block anchor; hi/lo pair |
| `intra_upper[I]` | `q̃ᵀ[:,I]·dAqk[I,:] + (βk)~ᵀ[:,I]·dL[I,:]` | ch-major, K = 16 | `dk_dp` + `dk_da` (part) | both products accumulate one partial |
| — | — | — | `p`, `at` | removed (Aqk loaded; dg reduced per channel) |

The per-chunk decay of `dH` (`Γ_L` per key row) is a per-lane scale in the state seed, the same
place GDR applies its scalar `g_last`.

## 4. Numerical safety (review item P1-1)

A single 64-token anchor is unsafe, as the review showed (`63 × 5 / ln 2 ≈ 454` in log2).
The scheme here never needs one:

- **State path.** Only `2^G` and `2^(G_L−G)` appear, both ≤ 1 because KDA gates are ≤ 0 and
  `G` is a chunk-local cumsum.
- **Off-diagonal intra blocks.** For a lower product with source block `J` and anchor `c_J` =
  `G` at the last row of `J`: the operand `x_j 2^(c_J−G_j)` (`j ∈ J`) and the epilogue factor
  `2^(G_i−c_J)` (`i` after `J`) are both ≤ 1 for **any** gate. Upper products use the first row
  of the source block and are symmetric. Underflow of the operand only drops contributions
  smaller than `2^-126` relative to the operand magnitude.
- **Diagonal blocks.** The same MMA also produces the diagonal block, but its epilogue factor
  `2^(G_i−c_J)` grows with the block span `G_first − G_last`. When the span of channel `d` is
  below 96, both factors stay inside the normal FP32/BF16 range, and the fast result is used.
  Otherwise, the thread that owns lane `d` recomputes that channel's 16 × 16 triangle exactly,
  using `2^(G_i−G_j)` with `i ≥ j`, which is ≤ 1. Because channels are TMEM lanes, any inf or
  NaN from an unsafe channel stays in that lane's partial and is overwritten before use.
- **Masking.** Epilogues never multiply a factor > 1 by a structurally zero partial: rows
  before the source block, and invalid tail rows, are skipped by predication rather than by
  multiplying by zero.
- **Default contract.** `kda_safe_gate=False` with no lower bound is the Megatron default, so
  the fallback must exist; with `safe_gate` (|g| ≤ 5) the span is ≤ 108 and the fallback is rare
  but not impossible, so the check stays.

CPU evidence (`tests/unit_tests/ssm/test_kda_fused_bwd_reference.py`, 7 tests, all passing):
the specification matches the FP64 recurrent autograd oracle to 1e-9 for dense, packed
`[17, 64, 65]` and tail inputs, for per-token gates down to −20 (a 64-token span of ≈ 1800 in
log2), and in the uniform-channel test, where the channel sum of `dg` equals the scalar-gate
gradient through the general path. With FP32-range operands, the anchored diagonal with fallback matches
to 1e-5 on mixed strong/weak channels. Disabling the fallback makes the result non-finite.

## 5. Precision study

Relative L2 error against the FP64 oracle, `K = V = 128`, `T = 256`, BF16 forward residuals,
all non-intra operand tiles rounded to BF16 as planned (`.runs/precision_cpu.py`):

| Intra operands (token / square) | gates −0.3…−0.01: dg | gates −3…−0.5: dg | gates −8…−2: dg | other grads |
| --- | ---: | ---: | ---: | ---: |
| FP32 / FP32 | 2.3e-3 | 2.3e-3 | 1.7e-3 | 1.5e-3 – 2.6e-3 |
| BF16 / BF16 | 4.0e-3 | 1.5e-2 | 3.5e-2 | unchanged |
| BF16 / FP32 | 3.9e-3 | 1.5e-2 | 3.5e-2 | unchanged |
| FP32 / BF16 | 2.4e-3 | 2.4e-3 | 1.7e-3 | unchanged |
| BF16 hi/lo / BF16 | 2.4e-3 | 2.4e-3 | 1.7e-3 | unchanged |

GB200 check at the main shape (`check_kda_fused_bwd_reference_gpu.py`; `B=2, T=8192, H=64`,
FP64 oracle on heads 0–2, seed 0). FLA BF16 `chunk_kda` and the planned kernel numerics both
receive FLA's own BF16 residuals. Values are relative L2 / max-abs error against the oracle:

| Gates | Path | dq | dk | dv | dg | dβ | dh0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| −0.3…−0.01 | FLA | 3.11e-3 / 4.7e-3 | 3.15e-3 / 0.147 | 3.22e-3 / 9.9e-3 | 2.79e-3 / 2.2e-2 | 2.78e-3 / 5.4e-2 | 1.74e-3 / 2.1e-3 |
| | plan | 3.33e-3 / 4.9e-3 | 3.33e-3 / 0.113 | 2.92e-3 / 9.9e-3 | 2.92e-3 / 1.5e-2 | 2.42e-3 / 4.2e-2 | 1.75e-3 / 3.3e-3 |
| −3…−0.5 | FLA | 2.82e-3 / 3.3e-3 | 2.83e-3 / 0.158 | 2.89e-3 / 9.9e-3 | 2.33e-3 / 5.2e-3 | 2.37e-3 / 4.3e-2 | 1.73e-3 / 7.7e-4 |
| | plan | 3.28e-3 / 5.4e-3 | 3.26e-3 / 0.087 | 2.85e-3 / 9.9e-3 | 2.78e-3 / 4.3e-3 | 2.33e-3 / 3.4e-2 | 1.65e-3 / 8.1e-4 |
| −8…−2 | FLA | 2.53e-3 / 3.4e-3 | 2.54e-3 / 0.135 | 2.87e-3 / 9.9e-3 | 1.95e-3 / 8.4e-4 | 2.35e-3 / 4.3e-2 | 1.66e-3 / 1.1e-4 |
| | plan | 2.59e-3 / 4.3e-3 | 2.58e-3 / 0.115 | 2.85e-3 / 9.9e-3 | 2.04e-3 / 7.1e-4 | 2.33e-3 / 3.4e-2 | 1.72e-3 / 1.2e-4 |

The plan is within 1.19× of FLA's relative L2 error on every gradient, and its max-abs error
is within 1.6×. The kernel accuracy envelope for M3/M4 is therefore: per gradient, relative L2
≤ 1.25× and max-abs ≤ 2× FLA's error against the same oracle, on these three gate regimes and
at least two seeds.

The anchored token operand is the sensitive one: `dg`'s reverse cumsum cancels the `+q⊙dq` and
`−k⊙dk_col` contributions of each token pair, and the two sides are rounded through different
anchors. A high/low BF16 pair for that operand alone recovers FP32 accuracy. The square tiles
(`dAqk`, `dL`) can stay single BF16. The floor with exact arithmetic and BF16 residuals is
1e-4 – 1e-3. These are CPU numbers for the specification; the kernel must reproduce them
(milestone M0 adds the FLA comparison on GPU).

## 6. Resource ledger (review item P1-2)

The GDR shared storage is 226,944 B of the 232,448 B opt-in limit. Translating it literally to
KDA, with a 32 KiB FP32 gate tile and six scaled K-side tiles, needs about 291 KB, so the
layout below replaces, aliases or moves allocations. Sizes are bytes; all MMA tiles are BF16.

| Buffer | Bytes | Contents over the chunk (in order) |
| --- | ---: | --- |
| `sQ`, `sK` | 2 × 16,384 | raw q, k (TMA); live until the dg epilogue |
| `sV` | 16,384 | v (TMA) → Δ = v − X → dv store staging |
| `sDo` | 16,384 | dO (TMA) |
| `sAqk` | 8,192 | Aqk (TMA) → dAqk |
| `sAkk` | 8,192 | Akk (TMA) |
| `sSq` | 8,192 | Akk·diag β → dAkk → inverse middle term → dL |
| `sH` | 32,768 | S (TMA) |
| `sDS` | 32,768 | BF16 dS snapshot → anchored token pairs (intra) |
| `sKA` | 16,384 | kΓ → kβΓ → anchored token pair half |
| `sKB` | 16,384 | kgL → anchored token pair half |
| `sQA` | 16,384 | qΓs → anchored token pair half |
| `sT21` | 16,384 | dvn → vn → D → anchored token pair half |
| vectors, reductions, barriers | ≈ 4,096 | β (4-head TMA vector), dβ partials, mbarriers |
| **Total** | **≈ 225,300** | about 7 KB below the limit |

Notes:

- Removed against GDR: the dQ staging tile (dq/dk/dg are stored by the channel-major warps
  directly; a warp's 32 lanes write 32 contiguous channels), the `dO·γ` tile, the `W` tile (β is
  folded into `Akk`), `sTmp23` (dh_k reads raw D and uses `kβΓ`), and all FP32 gate vectors.
- At the intra phase, the live buffers are `sQ`, `sK`, `sAqk`, `sSq`, and the six anchored
  half-tiles (96 KiB), which alias `sDS`, `sKA`, `sKB`, `sQA` and `sT21`. All of those buffers
  are dead by then: their last readers are `dvn_state`, `dk_state`, `dk_w` and `dh_k`. TMA
  destinations are never aliased, so next-chunk prefetch is not delayed by intra work.

TMEM (512 columns, one 512-column allocation; FP32; 64 tok columns or N ch-major columns per
tile):

| Range | Phases | Contents |
| --- | --- | --- |
| 0–127 | whole chunk | dH (128 key lanes × 128 value columns) |
| 128–191 | whole chunk | gate `gc` channel-major (lane = channel, column = token) |
| 192–255 | state → intra | dq channel-major accumulator |
| 256–319 | state → intra | dk channel-major accumulator |
| 320–383 | state/WY | tok slot A: X → vn |
| 384–447 | state/WY | tok slot B: dvn → D → `dk_w` (ch-major, after D is in `sT21`) |
| 448–511 | WY | sq slots: dAqk, dAkk/inverse chain (2 × 32 columns, `ws` packing as in GDR) |
| 320–511 | intra | R accumulator (64) + two-slot intra partial ring (2 × 64) |

Phase-overlap correctness is enforced the GDR way: a static `TMEM_RANGES` table with
`(first column, last column, first phase, last phase)` and an import-time disjointness check,
plus a matching SMEM liveness table. Writing those two tables and their tests is M1's first
deliverable; the sizes above are planning numbers until `SharedStorage.size_in_bytes()` and
the cubin resource report confirm them.

Gate data volume (review P1-2): `gc` and `dg` are each 512 MiB at `B=2, T=8192, H=64, K=128`
(1 GiB of FP32 traffic versus 4 MiB for GDR's scalar gate). At about 7 TB/s that is roughly
0.15 ms of HBM time by itself. Exponent work is about 6 `exp2` per (token, channel) per chunk
(Γ, 2^(G_L−G), four anchors), about 49 K MUFU operations per chunk, or about 0.2 ms at full
MUFU rate if not overlapped. These are model numbers, not measurements.

## 7. Warp roles and schedule (review item P1-3)

The port keeps GDR's CTA-per-(sequence, head) grid, reverse chunk walk, TMEM-resident dH, one
MMA-issue warp, one TMA warp, and the pipeline/barrier helpers (`_producer_at`,
`_consumer_at`, reuse barriers). It rebuilds the phases, operands and buffer lifetimes for
KDA; it does not promise stage-for-stage reuse of GDR's 16 phases.

| Warps | Role |
| --- | --- |
| 0–3 (CH) | channel-major: state seed (dS snapshot, Γ_L decay, `Σ S⊙dS`), gate tile into TMEM, scaled K-side tiles, dq/dk/dg/intra epilogues, diagonal fallback, dg reverse cumsum, dq/dk/dg/dh0 stores |
| 4–7 (TK) | token-major: Δ, vn/dvn/D materialization, dv store, row dβ terms, square-tile epilogues and the inverse chain |
| 8 | tcgen05 MMA issue |
| 9 | TMA loads (q, k, v, dO, Aqk, Akk, h, β) |
| 10–11 | spare; candidates for gate prefetch and output stores after profiling |

Phase order within a chunk: seed and tiles → `x_state`, `dvn_state` → `dvn_aqk`, `vn` → `d`,
`dakk`, `daqk` → `dq_state`, `dk_state`, `dk_w`, `dh_q`, `dh_k`, inverse chain → anchored tiles
→ intra ring → epilogues and stores. Next-chunk TMA loads are released buffer by buffer, as
in GDR.

## 8. CP and integration boundaries (review item P2-3)

- CP preprocessing and collectives stay outside; they produce `dht` (and the local `h` when it
  is recomputed). One primary compute launch is the target; the CP preprocessing launches
  and any duplicated local `dv`/state work are timed separately and included in full-backward
  and E2E numbers.
- Integration requires the KDA adapter, which is not on this base branch. The test harness
  therefore drives the kernel with FLA forward residuals directly until the adapter is
  available here.
- Strict mode must report fallback reasons and launch counts, so a silent FLA fallback cannot
  pass a fused-path test.

## 9. Milestones and gates

Each milestone ends with tests that fail on the previous code and pass on the new code. A
review of the diff comes before the next milestone starts.

| ID | Deliverable | Gate |
| --- | --- | --- |
| M0 | Specification (done) + GPU harness: FLA `chunk_kda_bwd` vs FP64 oracle vs specification on B200/GB200 at `B=2, T=8192, H=64` with selected heads | Specification error ≤ FLA error per gradient (or within a fixed envelope recorded in the repository) |
| M1 | Package skeleton: storage/TMEM tables with static checks, layouts for the new variants, launcher with compile key, TMA of all inputs, MMA warp issuing `x_state`/`dvn_state`/`dvn_aqk`/`vn`/`d` | Debug dumps of X, dvn, vn, D match the specification (single head, T = 64, 128, 130) |
| M2 | Full state/WY path: squares, inverse chain, `dq_state`, `dk_state`, `dk_w`, `dh_q`, `dh_k`, recurrence, dv/dβ/dh0 | `dv`, `dh0`, `dβ`(state part) match the specification; multi-chunk and packed tails |
| M3 | Intra path: anchored hi/lo tiles, ring, epilogues, diagonal fallback, dg reverse cumsum | All six gradients match the specification and the oracle; strong-decay and mixed-channel fallback tests |
| M4 | Main shape: `B=2, T=8192/8191, H=64`, packed `[17, 64, 65]`, `compute-sanitizer` memcheck/racecheck | Accuracy envelope from M0; zero stack/local spills; kernel time vs FLA and GDR (CUDA events over ≥ 20 interleaved samples, JIT excluded) |
| M5 | Adapter integration (after the KDA adapter is on this branch), CP2/CP4 with real cross-rank state | Strict-mode launch counters; CP tests that actually exchange state |

Performance targets are set after M4 measures the first complete kernel. The historical
≈ 1.34 ms (state-only) and ≈ 10 ms (standalone intra) prototypes from the older worktree are not
baselines for this kernel.

### Implementation status

| Artifact | State |
| --- | --- |
| `tests/unit_tests/ssm/kda_fused_bwd_reference.py` and its CPU tests | Done; 7 tests pass |
| `tests/unit_tests/ssm/check_kda_fused_bwd_reference_gpu.py` (M0) | Done; results in §5 |
| `kernels/fused_kda_bwd_cute/layouts.py` | Written; trace-time validation runs inside the kernel compile |
| `kernels/fused_kda_bwd_cute/storage.py` | Written; TMEM disjointness and SMEM budget checks at import |
| `kernels/fused_kda_bwd_cute/kernel.py`, `launcher.py` | v1 complete: all six gradients, packed tails, diagonal fallback; correct on GB200 (below) |
| `tests/unit_tests/ssm/check_kda_mma_probe_gpu.py` | Passes: six operand orientations and both TMEM readback patterns match torch |
| `tests/unit_tests/ssm/check_kda_fused_bwd_kernel_gpu.py` | Stage-by-stage debug-dump comparison against the specification |

v1 kernel results on GB200 (FP64 oracle; relative L2 error):

| Case | dq | dk | dv | dg | dβ | dh0 |
| --- | --- | --- | --- | --- | --- | --- |
| T=64, H=2 | 2.75e-3 | 3.03e-3 | 3.24e-3 | 2.55e-3 | 3.58e-3 | 1.75e-3 |
| packed [200, 67], H=2 | 2.90e-3 | 3.20e-3 | 3.05e-3 | 2.39e-3 | 2.25e-3 | 1.74e-3 |
| packed [17, 64, 65], H=4 | 2.71e-3 | 3.10e-3 | 3.12e-3 | 2.23e-3 | 2.94e-3 | 1.63e-3 |
| T=256, gates −20…−5 (fallback) | 2.77e-3 | 3.12e-3 | 2.99e-3 | 2.08e-3 | 1.71e-3 | 1.62e-3 |

Main shape `B=2, T=8192, H=64`, FLA residuals, heads 0–2 against the FP64 oracle. Each cell gives
the kernel's error, then FLA's in parentheses:

| Gates | dq | dk | dv | dg | dβ | dh0 |
| --- | --- | --- | --- | --- | --- | --- |
| −0.3…−0.01 | 3.12e-3 (3.11e-3) | 3.13e-3 (3.15e-3) | 2.95e-3 (3.22e-3) | 2.72e-3 (2.79e-3) | 2.67e-3 (2.78e-3) | 1.74e-3 (1.74e-3) |
| −3…−0.5 | 2.88e-3 (2.81e-3) | 2.88e-3 (2.82e-3) | 2.87e-3 (2.90e-3) | 2.36e-3 (2.36e-3) | 2.34e-3 (2.36e-3) | 1.69e-3 (1.73e-3) |
| −8…−2 | 2.87e-3 (2.52e-3) | 2.86e-3 (2.52e-3) | 2.87e-3 (2.88e-3) | 2.36e-3 (1.99e-3) | 2.34e-3 (2.35e-3) | 1.89e-3 (1.79e-3) |

All cells meet the §5 envelope: relative L2 at most 1.19× FLA's, and max-abs at most 1.6× FLA's.

Median GPU time (CUDA events, warmed, JIT excluded): **v1 kernel 31.2 ms; FLA
`chunk_kda_bwd` 14.0 ms**. The two do not cover identical work: FLA's time includes its
recomputation of `h`, while the kernel consumes the saved `h`. v1 is correctness-first and has
not been optimized. Its phases are fully serialized, the tile builders and epilogues use
per-element shared/global accesses, and the intra fold does twelve TMEM read-modify-write
passes per chunk. M4 is now performance work, guided by Nsight Compute.

### Optimization log (M4, in progress)

All timings use the same main-shape harness. Accuracy is unchanged across these steps: same
errors against the oracle to the printed digits.

| Step | Evidence | Change | Kernel time |
| --- | --- | --- | ---: |
| v1 | nsys: 34.75 ms per launch (`profile/kda_bwd_nsys_v1`) | — | 31.2 ms |
| v2 | ncu: 251 M local-memory spill requests, 87 % no-eligible cycles; the unrolled exact-diagonal code was if-converted, so it ran on every chunk and missed in the instruction cache (`profile/kda_bwd_ncu_v1`) | Gate column in TMEM, not registers; the exact diagonal is a rare, warp-uniform path with runtime loops; dβ reduced per 32-token part | 15.95 ms |
| v3 | `%globaltimer` phase trace: the TK warpgroup idle ~70 µs per chunk; 8 K contended shared atomics in TK phase C (`profile/kda_bwd_trace_v2`) | Channel-major work split by token half across both warpgroups (seed split by value half); quad row-reductions instead of atomics | 15.79 ms |
| v4 | trace: TK's phase A became the critical path | Masks rebalanced (CH masks Aqk; tail-row zeroing only for partial chunks) | 14.71 ms (FLA 13.98 ms) |
| v5 | trace: per-element column accesses in tile building | Gate-scaled tiles built in a token-row mapping: 128-bit swizzled shared loads/stores and 128-bit gate loads; vectorized state seed; one dβ reduction per fold | 10.13 ms |
| v6 | trace: epilogue serialized across halves; TK per-element phases | Concurrent two-half epilogue (cumsum carry added in registers); TK uses stmatrix / 32-bit fragment copies (GDR pattern) and 128-bit `dv` row stores | 8.42 ms |
| v7 | trace: fold dominated by serialized TMEM waits | Fold keeps each target in registers across its four partials, with batched TMEM loads; vectorized triangular masks | 8.04 ms |
| v8 | strong gates (−8…−2) took 19.7 ms: the exact diagonal fallback runs almost everywhere, and its inner loop waited on an L2 load per term | Fallback loops reordered (runtime loop over source tokens with TMEM gate loads, 16 independent register sums); span limit raised to 112 so `safe_gate` (|g| ≤ 5) never falls back | **8.06 / 8.03 / 16.7 ms** for weak / mid / strong gates (FLA 13.9 ms in all three) |
| v9 | strong gates: the fallback still evaluated one `exp2` per term and waited on a TMEM gate load per source token | Fallback splits the decay at the block's middle token in FP32 (each half span < 100, so factors stay within 2^±100): 32 `exp2` per block, gates from the register column, square rows by 128-bit shared loads, masked pairs skipped at compile time, one FMA per term. Spans too wide to split keep the per-term loop | **8.10 / 8.08 / 12.0 ms** (FLA 13.9 ms) |
| v10 | ncu (`profile/kda_bwd_ncu_v11`): 85 M local-memory spill requests (67 % of L1 sectors) at the 168-register cap; the fold's per-element factor and dβ lines stall on `long_scoreboard` | Fold with deferred per-token scaling: 2^(G_t − G_lo[b]) = 2^(G_t − G_up[tb]) · 2^(G_up[tb] − G_lo[b]) (lower) and 2^(G_up[b] − G_t) = 2^(G_lo[tb] − G_t) · 2^(G_up[b] − G_lo[tb]) (upper). Every factor except the diagonal one is ≤ 1, so the loop keeps only the unscaled sum live and needs 2 `exp2` per ring slot; the per-token factor, β, the dβ partials and the exact blocks are applied once per target. The epilogue reuses the dq registers for the cumsum | **7.11 / 7.04 / 9.74 ms** |
| v11 | ncu (`profile/kda_bwd_ncu_v12`): 58 M spill requests; the accumulator was spilled around the never-taken exact branch, and 32 serial warp reductions for dβ | Each target is added into TMEM before the rare exact branch, which does its own TMEM update with branch-local registers; dβ uses a butterfly transpose-reduce (31 shuffles, one shared atomic per lane) | **6.65 / 6.66 / 9.68 ms** |
| v12 | trace: TK wrote dAqk and T, ran its phase-C half, then wrote dL; CH idled ~4 µs before ready-E | TK writes only T before ready-D (all MMA D needs), its phase-C half, then dAqk; CH writes dL after its phase-C half | ready-E 1.2 µs earlier per chunk; end-to-end within noise (6.74 / 6.77 / 9.71 ms) |
| v13 | SASS histogram (`profile/kda_bwd_ncu_v12`): FMUL 194 M vs MUFU 41 M; the precise `exp2` wraps each `ex2` in range-reduction code | `exp2(..., fastmath=True)` (hardware `ex2.approx`, ~2 ulp; every factor feeds a BF16 product or FP32 sum) | **6.52 / 6.49 / 9.34 ms** (FLA 14.1 ms on this node); errors identical to the printed digits |
| v14 | trace: the epilogue took 14 µs of a 42.7 µs chunk; it loaded 96 TMEM values and kept 32 suffix sums live across a barrier at the register cap | Epilogue streams 8 tokens at a time from TMEM (`tcgen05.ld/st .x8`); the lower half parks its local dg suffix sums in its consumed dq columns and adds the carry after the exchange | **6.19 / 6.15 / 9.08 ms** (FLA 13.9 ms); epilogue 10 µs, chunk period 37 µs |
| v15 | trace stamps inside the epilogue: the TK half's first pass took 7.8 µs vs 3.7 µs for CH; ncu: `LDS` lines stalled on `long_scoreboard` (operands reloaded from local memory), 20 M `LDL` right after mbarrier waits | Tried 320 threads without `setmaxregister` (5.79 / 5.79 / 8.73 ms). `ptxas -v` on the dumped PTX showed why it barely helped: every SM sub-partition has a 16 K-register file, ten warps put three warps on two sub-partitions, and ptxas caps each thread at 16384 / (3 × 32) → 168 registers (1488 B spill stores). The earlier `setmaxregister` split (CH 240 / TK 160 / other 56) starved the TK warpgroup | — |
| v16 | ptxas spill report per configuration (local `ptxas -v` on the dumped PTX) and timing: 384 threads with CH 232 / TK 232 / other 40 halves the spills (584 B stores, 1848 B loads); 240 / 240 / 24 is slower (the MMA and TMA warps spill) | 384 threads, `setmaxnreg` 232 / 232 / 40 (the sub-partition budget of 504) | **5.58 / 5.53 / 8.42 ms** |
| v17 | trace: the epilogue halves were unbalanced because the upper half also wrote dg in its first pass | Both halves park their dg suffix sums in their consumed dq columns and write dg after the exchange (upper half's carry is zero) | 5.52 / 5.58 / 8.48 ms (epilogue 8.1 µs) |
| v18 | trace: ~1.7 µs from FREE to the next chunk's phase A while all 112 KB of inputs reload | v, dO, Akk and h (72 KB) are prefetched once MMA phase D has consumed them (`_E_LOAD_EARLY`); only q, k and Aqk (40 KB) wait for FREE | **5.54 / 5.52 / 8.28 ms** (FLA 14.0 ms); phase A starts 1.3 µs earlier, chunk period 35.9 µs |
| v19 | GDR comparison (`profile/gdr_vs_kda_v1/REPORT.md`): KDA executes 3.2× GDR's instructions at 1.6× the cycles per instruction; phase C made three TMEM passes per half before the ring | The state-path gate scalings (dq: scale·2^G, dk⁺: −2^G, dk⁻: 2^(G_last−G) with its dg term) move into the fold finalize, which already loads each target and the gate column: one TMEM pass per target instead of two, and less work before ready-E | **5.11 / 5.01 / 7.51 ms** |
| v20 | per-line instruction counts: integer/address math 332 M vs GDR's 103 M; per-element `s_k[t, d]`, `dq[0, t, head, d]` and `gc[0, t, head, d]` recompute swizzled shared addresses and 64-bit offsets with runtime strides | Channel-column shared loads through eight per-residue bases (`_column_bases`: the SW128 XOR depends only on row % 8), and global gate loads / dq, dk, dg stores through per-thread base pointers with compile-time row strides | **4.81 / 4.82 ms** weak / mid (epilogue 6.5 → 2.5 µs); strong 7.83 ms (re-measured; ~4 % slower than v19, under study) |
| — | tried: hoisting all gate loads of the tile builders ahead of the arithmetic; prefetching the next chunk's gate column during the epilogue | No gain (phase A unchanged in the trace) and a regression (+5 %: 40 more registers live across the loop), both reverted | — |
| v21 | trace: about 1.9 µs from FREE to phase A while q, k and Aqk reload | The gate loads and the state seed (which read neither) run before the late-load wait | about 1 % (re-baseline on another node: 4.81 / 4.87 / 8.04 → 4.77 / 4.83 / 8.08 ms) |
| v22 | the four 64×64 squares (T, dAqk, dL, dAkk·β) were written to shared memory one swizzled BF16 element at a time | `stmatrix` from the TMEM-load fragment, as for the token tiles | **4.68 / 4.65 / 7.84 ms** |
| v23 | trace: CH idles about 6.6 µs per chunk while TK prepares the phase-B and phase-C operands | TK publishes ready-B before building −(kβG) (MMA B does not read it) and ready-C before staging and storing dv and seeding dβ (MMA C reads neither) | **4.60 / 4.57 / 7.54 ms** (interleaved A/B, `.runs/bench.py`) |
| v24 | ncu (`profile/kda_bwd_ncu_v29`, weak gates): named-barrier waits are the largest stall after the role mbarrier waits; the fold ran a 128-thread barrier per ring partial (12 per chunk) before one thread released the slot | Each consumer warp releases the slot itself after its TMEM loads (`ring_empty` counts 8 warp arrivals) | **4.55 / 4.53 / 7.43 ms** (interleaved A/B) |
| — | tried: L2 prefetch of the next chunk's gate tile (one line per thread); a third ring slot in the gate column's TMEM (gate column reloaded into registers for the fold) | Both slower: +5 % (+25 % strong gates; the trace of CTA 0 looked faster, so likely code generation) and +5 %. The deeper ring did not shorten the ring phase, so the fold is bound by its own work (mainly the three per-target finalizations), not by ring round trips | — |

At v20 the kernel is about 2.9× faster than FLA's full KDA backward for weak and mid gates and
1.8× faster for gates of −8…−2 per token, where every diagonal 16×16 block takes the CUDA-core
fallback. The remaining strong-gate cost (about 21 µs per chunk) could go away with a
tensor-core diagonal (mid-block anchored operands), but that needs SMEM and TMEM room for the
extra operand tiles and partials.

At v18 a chunk takes about 36 µs in CTA 0 (weak gates, `profile/kda_bwd_trace_v2`), and the
critical path is still element-wise consumer work:
- phase A (gate column, state seed, gate-scaled tiles), about 4 µs;
- token-major B/C preparation (TK), about 5 µs;
- the channel-major phase C halves (state-path scaling, anchored intra tiles, dL), 6–8 µs;
- the intra ring and fold, about 6 µs;
- the epilogue, about 8 µs.

All MMAs together take about 3 µs. Remaining local-memory traffic (about 0.6 KB of spill
stores per thread) is mostly loop-invariant shared-memory addresses of the token-tile copy
partitions, reloaded after each mbarrier wait; it is a few percent of the chunk. Next
candidates:
1. Phase C and phase A instruction count: one gate exponential per element shared between
   the dq and dk scalings, and fewer per-element global gate loads in the tile builders.
2. The epilogue: stage dq and dk in the freed intra-operand tiles and store whole rows
   instead of 2-byte per-thread stores.
3. A tensor-core diagonal for strong gates.

v1 deviations from the §6 ledger: the 64×64 squares use ordinary (non-`ws`) MMAs with 64 TMEM
columns each, borrowing the not-yet-live `dk_minus` and token slots. Since v2 the gate column
lives in TMEM columns 448–511, as §6 planned.

The v1 kernel uses one-shot mbarrier edges per phase (consumers → MMA "ready", MMA → consumers
"done") and reloads the next chunk only after both consumer warpgroups release the current
chunk. This is deliberately simpler than GDR's per-buffer reuse edges; per-buffer release and
cross-chunk overlap come after correctness (M4).

Toolchain on the GB200 nodes: the NGC `pytorch:26.04-py3` image (local `.sqsh`), plus
`nvidia-cutlass-dsl==4.5.0` (the GDR kernel needs `cute.nvgpu.OperandMajorMode`, which is not
in the image's 4.4.1) and `nvidia-resiliency-ext==0.6.0` (needed to import `megatron.core`),
installed into a worktree-local `.deps` directory. FLA is imported without `fla/__init__`
because that file requires `transformers`.

## 10. External references

No public fused, chunked KDA backward for SM100 was found, so there is nothing to port directly.
The closest work, for cross-checking ideas only:

- [cuLA](https://gitee.com/geniusxxx/cuLA) (inclusionAI): CuTe DSL/CUTLASS linear-attention
  kernels with a Blackwell KDA path; its tensor-core use requires `safe_gate=True`. Its KDA
  backward is reported to be a hybrid, with a native WY stage and the reverse-state stage still
  in FLA Triton.
- [arXiv 2608.12700](https://arxiv.org/pdf/2608.12700): describes a native tcgen05/TMEM
  reverse-state backward for the gated linear recurrence family.
- [FlashInfer KDA backward](https://docs.flashinfer.ai/api/kda_backward.html): a recurrent (not
  chunked) SM100 training backward.
- [FlashKDA](https://github.com/gmh5225/FlashKDA) (Moonshot AI): forward only.

The GDR kernel itself is a CuTe DSL derivative of QwenLM FlashQLA's Blackwell fused GDR
backward (see the `kernel.py` header); the KDA package inherits that attribution.

## 11. Open risks

- **Register pressure in CH warps.** One lane per channel holds a 64-token column of
  partials. The epilogues must process 16-token column slices to stay under the per-warpgroup
  register limit without spills.
- **Diagonal fallback cost.** If real gates trip the 96 span often, the CUDA-core triangle
  (136 multiply-adds per channel per block) becomes visible. Its frequency must be measured
  on real Kimi/KDA gate statistics before any optimization.
- **`dβ` cross-channel reduction.** `Σ_d k⊙R` is a reduction across the 128 CH threads per
  token. The plan uses warp shuffles plus a 4-warp SMEM combine; it may move to an MMA against
  a ones vector if it shows up in profiles.
- **SMEM headroom (about 7 KB).** Any extra staging (for example an FP32 gate prefetch buffer)
  must replace something in the ledger.
- **GDR baseline accuracy.** The older worktree reported GDR/FLA baseline accuracy failures
  on some real-gate cases. A "near-GDR performance" claim needs a GDR baseline that passes
  the same accuracy guard in the same environment.
