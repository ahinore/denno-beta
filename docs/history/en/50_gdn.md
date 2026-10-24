# Volume 6 GDN (Gated DeltaNet) — formulas, kernels, chunked scan

> [Index](README.md) / Previous: [Volume 5 prefill performance](40_prefill_perf.md) / Next: [Volume 7 Quality](60_quality.md)
> Japanese original: [../50_gdn.md](../50_gdn.md)

GDN is the recurrent part of the Qwen3.5 family's hybrid layers (3:1); it is not a block of matrix products but an
operation that "updates the state one token at a time". This volume lays out in chronological order the record of how we
copied that formula, why we moved from the single-shot kernel to the chunked 3-pass, and how we finally found that the
bottleneck was not "arithmetic" but "local memory bandwidth".
Model integration, the GGUF loader and output quality are outside the scope of this volume.
Dates are the commit author dates in JST.

## What this volume covers

- The history of copying GDN's delta rule in stages — CPU reference → GPU single-shot kernel → chunked 3-pass — and the numeric contract at each stage.
- That prefill's GDN was bounded not by arithmetic but by **local memory bandwidth** (including the record that both of the 2 options tried in s192 were rejected).
- That moving the beta/alpha projection to the GPU failed twice (s78/s79 not adopted, s96 removed), and was finally solved by the CPU-side data layout.
- How the attempt to reduce launches (s89/s90) did not hold structurally, and the goal moved to the "reduce synchronization" direction (P15/P45).

## Background knowledge

- GDN's state S is **a function of the entire sequence** and cannot be sliced by position. "Cut the state partway and reuse it" as with KV does not work, and
  prefix reuse is limited to a single point at the prompt boundary.
- One step of the delta rule is a **linear map** on S, `S ← eg·(I − βkkᵀ)·S + βkvᵀ`, so the dependency across chunks can be
  decomposed into "local transition inside a chunk (parallel)" and "state propagation between chunks (serial)". This is the basis of the chunked scan.
- Numeric discipline: on the CPU path, "token-sequential == batch prefill is bit-identical". GPU GDN initially had the contract "bit identity not required, determinism only", but
  from s241 on, more changes claim bit identity. Which position each entry takes is stated in the entry.

## Reading order

The body is chronological (2026-10-14 → 10-24). Readers in a hurry can pick up P19 → s64 → s67 → s72 (the device-residency flow) →
P15 and s192 (settling the bandwidth bound) → s241/s243 → P36 → P40/P43 (the instruments) → P45 in that order, and read
"the target of the speedup moving from arithmetic to bandwidth and synchronization" as a single thread.

## Phases covered

| phase | date | commit | one-line summary |
|---|---|---|---|
| P19 | 2026-10-14 | `0bbfa79e` | From a CPU reference and a single-shot kernel to the chunked 3-pass |
| GDN GQA head mapping | 2026-10-16 | `e2660908` | Settle the head mapping from blocked to strided |
| s61c | 2026-10-19 | `b25529d3` | Split attn_gdn into projection and scan so it can be measured |
| s64 | 2026-10-19 | `8e7ff24d` | Break down 1 scan call into alloc/launch/sync/d2h |
| s67 | 2026-10-19 | `b36df11e` | Reuse the work area across layers and halve the alloc |
| s72 | 2026-10-19 | `4119dce4` | Keep y on the device and chain it through to out_proj |
| s78/s79 | 2026-10-19 | `85a94052` | Prototype a GPU kernel for the beta/alpha projection (not adopted) |
| s89/s90 | 2026-10-19 to 20 | `ddf7201a` | Folding 2 launches structurally does not hold |
| s96 | 2026-10-20 | `55586f03` | Delete the already-rejected GPU gdn_proj path |
| P15 | 2026-10-23 | `c0b19d07` | Remove the host wait between passes and turn 3 syncs into 1 |
| s241 | 2026-10-23 | `0d9faef7` | Fuse pass 1's decay and dot product into 1 pass |
| s243 | 2026-10-23 | `8503925c` | Set the fused loop's unroll to 8 and cut registers |
| P36 | 2026-10-23 | `3e288aa2` | pass 1's 4-lane version has no effect in measurement |
| P40 | 2026-10-23 | `bf88ab26` | Isolate the 3 kernels' real time with CUDA events |
| P43 | 2026-10-23 | `88e9e86d` | The reason KPROF printed nothing on the device was the output point |
| P45 | 2026-10-24 | `d19aa70b` | Take the sync after the 3 passes out of the device output path |

## Gap between the ceiling (theoretical values) and measurements

| item | theoretical value or upper bound | measured | cause of the gap |
|---|---|---|---|
| GDN pass 1's local memory transfer (27B q4, 1 layer) | estimated 265 GB/layer. 147 ms/layer at 1.8 TB/s (conversion, estimated) | 138 ms/layer (P15, n=14,043, average of 48 layers) | bandwidth-bound. The estimate and the measurement nearly agree, so cutting it requires a structural change of the .cu |
| the actual arithmetic of 1 GDN layer | 132 GFLOP/layer. 4.4 ms/layer at q4 GEMM's effective 30 TFLOPS (conversion, estimated) | 138 ms/layer (same as above) | efficiency ≈3%. There is 30x of headroom in the arithmetic, so making the arithmetic faster does not help |
| 4-lane version of pass 1 (P36) | local access instructions 1/4, stack frame 0 B, 128 regs (design values) | 9.69 → 9.88 s (27B q4, 16k, median of 4) | occupancy dropped to 1 block/SM and cancelled the gain |
| pass 1's fusion + unroll (s241/s243) | no material (only the reduction rates recorded) | GDN pre −5.8% (65k) → a further −1.8% | the only condition is "65k" in the commit body; there is no record of the model or the number of runs |
| removing the `ctx.sync()` after the 3 passes (P45) | 10–100 ms/req = 0.03–0.3% of 35.8 s (estimated) | not measured | what the sync contains is the 3-pass GPU execution itself. What disappears is only the drain |
| GDN scan phase (27B q4, 65,000 chars prefill) | no material | 6,965 ms = 19.4%, of which `gdn_pf_sync` 6,887 ms (99%) | what the sync contains (the split of pass 1/2/3) is separated by the P40/P43 instruments. The on-device values are not measured |

---

## Chronology

### P19: GDN's CPU reference and single-shot kernel, up to the chunked 3-pass (2026-10-14, `0bbfa79e`)

**Intent** — Running Qwen3.5's hybrid layers requires GDN's scan. GDN is a sequential recurrence, so it cannot be routed into the existing GEMMs;
at the CPU-reference stage it was about 90 ms/token for prefill and about 20 ms/token for decode (0.8B, CPU token-sequential). This becomes the next bottleneck.

**What changed** — `0bbfa79e` (the CPU reference in gdn.rs) / `cb9d30bb` (lib registration) / `feab1c8b` (`GdnCore` split and the GDN slot of the KV).
The formula is `g = ssm_a(negative) × softplus(alpha + dt_bias)` → `S ← S·exp(g)` → `sk = Σk·S`, `d = β(v − sk)`, `S += k⊗d`, `o = Σq·S`.
q/k are L2-normalized, the output is a gated RMS norm (`rmsnorm(o)·w·silu(z)`), and the input side has a causal depthwise conv (d_conv=4) + SiLU.
We copied llama.cpp's `build_delta_net_autoregressive` (the sequential AR form) 1:1.

**Stumble** — The sequential AR form is mathematically equivalent to the chunked form but rounds in a different order. Therefore we do not require bit identity with llama.cpp
(the contract is "same input → same output + determinism"). For the GPU version we used a configuration of 1 block = 1 head and thread = a column of S (dv), made
all the delta-rule updates (sk / outer product / output) complete inside the thread, and removed inter-thread reductions. Determinism is
guaranteed by "fixed launch configuration, no atomic / no warp reduction, sums in fixed order".

**What we learned** — The single-shot kernel gave y maxabs 3.353e-8 and ssm maxabs 7.451e-9 (margin against the relaxation bound of 1e-3), and the conv state was bit-identical.
E2E went 5.11 s → 2.09 s. But with 1 block/head it is token-sequential, so it does not scale on a long prefill.
So we added the chunked 3-pass: K1 `dno_gdn_chunk_pre` (chunk-parallel, W = the local AR end state starting from zero, M = the transition from the identity matrix),
K2 `dno_gdn_chunk_state` (chunk-serial, `Sinit[ci+1] = M[ci]·Sinit[ci] + W[ci]`), K3 `dno_gdn_chunk_scan` (chunk-parallel, sequential AR inside a chunk).
We route to chunked when `n ≥ 2048`, and share the split `chunk_plan(n, 2048, 1024, 16)` with the CPU reference.
A single chunk is bit-identical to the sequential AR; multiple chunks are mathematically equivalent with only a different rounding order.

**What we learned / remaining issues** — The basis for numerical safety is that "k is L2-normalized (‖k‖≈1), β∈(0,1) and eg=exp(g)<1, so the spectral norm of
each factor is ≤1, M becomes a contraction and the rounding error is not amplified". The on-device E2E effect of chunked is not measured
(because a GPU binary rebuild is required).

**What worked next** — The frame "GDN does not need bit identity, only determinism" splits later decisions in two. s241/s243 became
changes that claim bit identity, and s192's 4-lane plan was held back as needing "to give up bit identity".

### GDN GQA head mapping: settling the convention, and the period when the PTX was left behind (2026-10-16, `e2660908`)

**Intent** — Settle the mapping convention for the k/q head that v-head `h` references. This is not about performance but about the convention, yet the
mapping is 1 line of index arithmetic, and the same expression appears in 2 CPU sites and 3 GPU kernels. If they disagree, the CPU and the GPU compute different things.

**What changed** — `e2660908` / `crates/denno-engine/src/gdn.rs` / `crates/denno-engine/kernels/gdn.cu`.
`kh = h * nk / nv` (blocked, equivalent to HF's `repeat_interleave`) → `kh = h % nk` (strided, equivalent to llama.cpp's `ggml_repeat`).
The GPU side is 3 sites: `dno_gdn_scan` / `dno_gdn_chunk_pre` / `dno_gdn_chunk_scan`. The load-time guard
(configurations where n_v_heads is not an integer multiple of n_k_heads are rejected. 0.8B 1x, 9B 2x, 27B 3x) stayed as it was.

**Stumble** — At this point **we did not regenerate the PTX** (it was an environment without nvcc). That is, gdn.cu is strided while
`kernels_gdn.ptx` stayed blocked, and we temporarily created a state where the GPU path runs on the old convention. Regeneration was a separate commit
(`fc78586d`), and it takes the procedure of first confirming that recompiling the pre-change source matches the committed PTX by sha256.

**What we learned** — The agreement was cos ≥ 0.99984 on 9B (median 0.99998) and cos 0.99999 on 27B (all tokens).
Performance is not measured (1 line of index arithmetic, so no change expected = estimate). 0.8B has nk == nv so both conventions are identical and it is unaffected.
We made the diagnostics (per-layer `[LCHK]` checksum / `[GDUMP]` GDN substage dump) permanent behind env gates.

**What worked next** — We became able to say per layer "which substage moved", and from then on we could mechanically attribute the diff lines
in later GDN changes (s241 onward). The risk of the GPU and PTX conventions drifting apart also remained as a standard regeneration procedure.

### s61c: splitting `attn_gdn` into projection and scan (2026-10-19, `b25529d3`)

**Intent** — In the T5 profile, prefill's `attn_gdn` accounted for 28.5 s. This single bucket
mixed the beta/alpha projection (`apply_bt_rows` → P17i's GEMM) and the GDN scan body (`gdn_scan_batch`: DevBuffer allocation + H2D/D2H + sync), so
we could not separate the effect of T5b's first lever (making the scan device-resident) from "is the projection side already running on the GPU".

**What changed** — `b25529d3` / `crates/denno-engine/src/prof.rs` / `crates/denno-engine/src/qwen35.rs`.
`ATT_GDN` stays as the parent bucket, and we added display-only children `ATT_GDN_PROJ` (id `100*ATTN+2`) / `ATT_GDN_SCAN` (`+3`) (`ATT_CLS_N` 2→4).
**No change to the numeric path or the branches.**

**Stumble** — The double-counting trap. If we extend the child-to-parent accumulation to the raw id (`attn_core`) as well, `attn_core` gets double counted
(we actually hit this during implementation). The invariants are `attn_core == attn_gdn + attn_full`, `attn_gdn == gdn_proj + gdn_scan` (both time and call count).

**What we learned** — We split only the 3 sites in prefill (beta projection, alpha projection → PROJ; scan → SCAN).
We did not split `ATT_GDN` in the decode chain because beta/alpha and the scan run in a single fused device call
(splitting them would require touching the numeric path and would exceed the "measurement only" scope). Tests +1, 299→300 passed.
Lesson: **do not pin the absolute value of a shared counter** such as a total row in a test. What should be pinned is the relation "the sum of the breakdown equals the parent".

**What worked next** — With this split, `gdn_proj` appeared as 96 call / 25.3 s ≒ 263 ms/call, and we reached the
defect of s62 where "n<64 was silently falling into the single-thread path" (details in `40_prefill_perf.md`).

### s64: breakdown measurement of 1 GDN scan call (2026-10-19, `8e7ff24d`)

**Intent** — After s63, the largest remaining item in prefill was `d2h_state` 10,604 ms / 168 call (28%).
However, a single `gdn_scan` call mixed "buffer allocation + input H2D → launch → `cuCtxSynchronize` → D2H + host reflection", and
we could only guess where the weight was.

**What changed** — `8e7ff24d` / `crates/denno-engine/src/cuda.rs` / `crates/denno-engine/src/prof.rs`.
We split the internals of `gdn_scan_impl` (the n < 2048 path; decode n==1 is also here) and `gdn_scan_chunked_impl` (the 3-pass for n ≥ 2048) into
the 4 intervals alloc / launch / sync / d2h. Each interval's elapsed time is read at the entry of the next interval, so the sum of the breakdown ≒ the outer `gdn_scan`.
The new `prof::Seq` (`begin`/`lap`/`end`) is **a pass-through that does not even call `Instant::now()` when PROFILE is disabled**.

**Stumble** — To avoid double counting, we limited the accumulation target to the dedicated table `GDN_PF_NS/CALLS` and the `gdn_pf_*` ids (`10*D2H+9..12`),
and added nothing at all to the existing raw ids, `d2h_state` / `attn_gdn` / `gdn_scan`, or the D2H breakdown 0..8. We ran 1 request on the CPU
(v6_2b, int8, in=302) and confirmed that the GPU-only buckets were 0 ms / 0 call and the `gdn_pf_*` rows were not printed.

**What we learned** — The measured 63 ms/call has not been obtained in the GPU-less development environment (not measured). Three facts were settled from the code before measuring:
(1) the arguments of `gdn_scan_batch` have no device buffer for the weights, and the 4 of `wconv`/`wa`/`wdt`/`wnorm` are **H2D'd again on every call**
(26 layers × 2 requests = 48 call). (2) "Reuse the state DevBuffer outside the layer loop" does not work in prefill — the layer loop walks `li`
once each in ascending order, so the same (li, size) never comes a second time. (3) `gdn_scan` always calls `ctx.sync()`, so the pipeline breaks and
it has a structure that absorbs the wait for the preceding GEMM to finish.

**What worked next** — s67's `GdnScratch` (cross-layer reuse) and s72's `y_dst` chain use these (2)(3) directly as their design basis.
The isolation "what can be reused is only scratch whose size is fixed and proportional to n" became the implementation condition as it was.

### s67: reusing the GDN scan work area across layers (2026-10-19, `b36df11e`)

**Intent** — In s64's breakdown, `gdn_scan` 3.15 s consisted of alloc 1.04 s / sync 1.09 s / d2h 0.96 s. alloc was 48 call at 21.7 ms/call, and
the content was **16 `DevBuffer::from_slice`** (input 6 + weights 4 + chunk 3 + output 3, total ≈180 MB) being `cuMemAlloc`'d per layer and
thrown away. n is constant within a request and the shapes are fixed by the model, yet we rebuilt them every time.

**What changed** — `b36df11e` / `crates/denno-engine/src/cuda.rs`. We added `GdnScratch` (thread_local, grow-only), allocate
the 16 of the chunked scan only on the first call, and overwrite them with `copy_from` afterwards. Same convention as `SCRATCH` (for maybe_gpu_linear).

**Stumble** — We had to isolate the implicit dependency on zero initialization. `w`/`m`/`sinit`/`y` are written in full by each pass so they need no zero fill, and
the **zero H2D (≈60 MB/call)** that the old implementation did with `from_slice(&vec![0.0; …])` disappeared as well. On the other hand only `conv_out`/`ssm_out`
keep the convention "write only in the final chunk block", so they are still initialized to zero (behavior unchanged). The 4 weights change contents per layer, so
they are swapped with `copy_from` (hundreds of KB, so the transfer is negligible).

**What we learned** — `gdn_pf_alloc` 1,044 ms → 526 ms (the remainder is pure H2D transfer ≈11 ms/call), `gdn_scan` 3,126 → 2,593 ms,
`attn_gdn` 4,209 → 3,664 ms, prefill 23.7 s → 23.1 s, 240 → 246 tok/s. 231 passed / 0 failed, 3 GPU anchors MATCH.
The numbers changed only because "the lifetime of the buffers got longer"; the written contents, kernel arguments, launch configuration and computation order are unchanged.

**What worked next** — The distinction "scratch can be reused but state cannot" became the premise as it was for s72's `y_dst` and P15's sync removal.
The pattern of moving the lifetime out of the layer loop instead of removing the allocation itself took hold.

### s72: keeping GDN's y on the device and chaining it through to out_proj (2026-10-19, `4119dce4`)

**Intent** — s70 made the FFN device-resident, but GDN's attention block still had the same round trip. The scan D2H'd y
(n×4096 = 47 MB), and the out_proj right after H2D'd the same y back. That wasted **2.2 GB/request** round-tripping 94 MB per layer over 24 layers.

**What changed** — `4119dce4` / `crates/denno-engine/src/cuda.rs` / `crates/denno-engine/src/model/qwen35.rs`.
We added `y_dst: Option<u64>` to `gdn_scan_chunked_impl`; when specified, the scan writes y directly into the chain slot (`CH_PFH`) and
drops only the conv/ssm state to the host. `gdn_scan_batch_oproj` is "scan → out_proj (int8 GEMM) → D2H the result once"
as 1 function. In `forward_prefill`'s GDN path, if this holds it skips the whole existing path (D2H y → `apply_bt_rows_into`).
We also stopped the zero fill in `DevBuffer::to_vec` (D2H directly into the uninitialized region then `set_len`; a few ms for 47 MB).

**Stumble** — All the preconditions (weight upload, slot allocation, ctx) must be checked **before** the scan.
The scan updates conv/ssm, so failing partway and falling back would **advance the state twice**. This ordering became a design constraint.

**What we learned** — It goes through the same `f_quant` + the same `f_gemm` (identical int8 accumulation), so it is **bit-identical**, and all 3 GPU anchors (1b / 2b / full 1b)
MATCH. prefill 11.34 s → 10.11 s (502 → 562 tok/s), −78% / 4.6x from the 46.2 s at the start of the session.

**What worked next** — The pattern "when chaining another kernel behind a kernel that updates state, hoist all the precondition checks to the front" was
reused as it was in P15/P45's sync removal.

### s78/s79: prototyping a GPU kernel for GDN's beta/alpha projection → not adopted (2026-10-19, `85a94052`)

**Intent** — In the most recent profile, prefill's `gdn_proj` was 5.4 s (5,866 tok). We tried the GPU once in s77, but
the shape check compared `w.len()` with `k*tokens` so it was always false and it fell to the CPU: **dead code**, and the timing was a no-op closure too.
s78 rebuilt it from the kernel, and s79 got the build through.

**What changed** — `85a94052` (kernel) / `e52df695` (opt-in gate) / `98c83cbb` (tests).
New kernel `dno_matvec_rows_f32` (`crates/denno-engine/kernels/kernels_elem.cu`). It computes `[m,k] × [n,k]^T → [m,n]` with
1 block = 1 output (i,j), and uses no shared memory, no atomics and no warp reduction (how determinism is made).
On the host side: the shape gate `gdn_proj_shape_ok`, the non-normal precondition check (`gdn_proj_min_row_sqsum`), 1 call = 1 H2D + 1 D2H,
and the 4-way split `D2H_GDN_PROJ_{H2D,KERNEL,SYNC,D2H}`. We isolated it behind **opt-in `DENNO_GPU_GDN_PROJ=1` (default off)**,
and the default path stays the existing `apply_bt_rows_into` (CPU, bit-invariant).

**Stumble** — 3 things were broken on the device. (1) The numbers did not match: the maximum absolute difference from the CPU was on the order of 1.0e2 (k=4096 nv=32 m=4096), not within ulp.
Determinism (bit-identical for the same input twice) holds, so the computation itself disagrees (we suspect the passing of grid/dimensions and the weight layout).
(2) A runtime error on the k=1 shape → `LMV32_BROKEN` latches and every subsequent call falls to the CPU. (3) Because the opt-in gate's judgment comes before
the non-normal precondition check, with the gate off we get `GdnProjMiss::Disabled` and cannot observe `Denorm` (the ordering of diagnostics).

**What we learned** — **Measured on the real GPU, the kernel is not correct, so we do not use it by default.** We kept it isolated as opt-in and
kept default off. 312 passed / 0 failed / 18 ignored. The GPU probe `cuda_gdn_proj_f32_bit_exact_vs_cpu` does not claim bit identity despite its name;
it only records determinism and the deviation (the name is historical). As a by-product, we repaired an existing breakage where 7 test files had
not followed the addition of `AppState.temperature_cap` and did not compile (tests only).

**What worked next** — The same shape (an F32 projection with large m and small k) switched direction to being solved by the CPU-side data layout rather than the GPU, and
s66's transposed layout (group-major `[g][d][lane]`) gave 5.0x (details in `40_prefill_perf.md`).
This GPU path itself is removed in s96.

### s89/s90: folding GDN's scan into 2 launches does not hold (2026-10-19 to 20, `ddf7201a`)

**Intent** — The chunked scan is 3 launches + 3 syncs: pass 1 (W/M) → pass 2 (serial propagation of Sinit) → pass 3 (sequential AR).
Folding pass 2 into chunk 0 of pass 3 should have made it 2 launches.

**What changed** — `ddf7201a` / `crates/denno-engine/src/cuda.rs` / `crates/denno-engine/kernels/gdn.cu`.
We implemented `dno_gdn_chunk_scan_hoist`, placing the chunk-boundary state in graph memory `s_rows[t]`.
**As a conclusion `route_hoist` is always false** — the hoist version is never run even once.

**Stumble** — There are 2 reasons, and both come down to "the writer of the row `ci*chunk_len − 1` that a folded chunk ci ≥ 1 reads is not in the same launch".
(1) **GDN layers are not contiguous**: full attention layers sit in between, and a full layer issues no GDN launch.
By the time layer i reads `s_rows`, the launch that wrote it does not exist. (2) **The head of an interval always breaks on the host**: each layer's scan interval is
`[lo, n)`, and the host state matches the interval boundary every time `cache.advance(n)` happens. Adjacent intervals can only be connected through the host.
Therefore the 3-launch path is the only correct path. We keep the bookkeeping and the invariant tests, but **a healthy ledger is not a basis for the path to run**.

**What we learned** — Measured (cold A/B, 5,686 token, 3 alternating rounds): OLD `b502e34` = 13.362 / 13.408 / 13.404 s,
NEW `98c83cb` = 13.345 / 13.377 / 13.375 s → **−0.03 s, no gain** (does not reach the acceptance condition of 0.3 s).
The 3 anchors matched, and decode was unchanged at 16.4 ms/token. We also learned where the remaining cost goes: of `gdn_scan` 1,987 ms,
1,100 ms is waiting on the sync (= the actual GPU work) and 520 ms is activation H2D, so there is no resident lever left inside the scan.

**What worked next** — The goal moved from "reduce launches" to "reduce synchronization". From then on this was treated as `d2h_state`
(6,147 ms = 38.9% of wall), leading to P15/P45's sync removal.

### s96: deleting the already-rejected GPU gdn_proj path (2026-10-20, `55586f03`)

**Intent** — The GPU path for prefill's GDN beta/alpha projection prototyped in s78/s79 (the `DENNO_GPU_GDN_PROJ=1` opt-in) had
been settled as numerically broken by s79's on-device measurement. It remained as dead code with default off, so we delete the Rust side.

**What changed** — `55586f03` / `crates/denno-engine/src/cuda.rs` / `crates/denno-engine/src/model/qwen35.rs` / `crates/denno-engine/src/prof.rs`.
Deleted: `gdn_proj_f32_into` / `gdn_proj_opt_in` / `gdn_proj_shape_ok` / `gdn_proj_min_row_sqsum` / `gdn_proj_gpu_calls` /
`gdn_proj_cpu_denorm_rows` / `gdn_proj_last_miss`, `GdnProjMiss`, the `GDN_PROJ_*` constants, the 3 corresponding tests, and
`model/qwen35.rs`'s `gdn_proj_into_gpu()` plus the call branch in prefill. Callers go directly through the existing CPU path
(`apply_bt_rows_into` → `matvec_rows_bt_into`). `prof.rs` **loses nothing** — we keep the id reservations
(renumbering would break comparisons with past profile logs). The table keeps printing 0.

**Stumble** — We left the `.cu` / `.ptx`. PTX regeneration requires nvcc, so removing the `.cu` and the `.ptx` at the same time would break the build convention.
The kernel is not shared with other paths, but we left the deletion as work on a GPU machine (there is no reference from the Rust side any more).

**What we learned** — The numbers behind the deletion: with k=4096 nv=32 m=33, `out[0]` was **−17.205 vs −8.410** (CPU reference), maximum absolute difference ~1e2.
A near-2x difference cannot be explained by rounding; it is an implementation bug. The relative error also does not satisfy R1 (1e-6) of NUMERIC_POLICY.
The basis for the default path being unchanged: what we removed is only the branch that "`return`s only when the opt-in is set and all the GPU gates pass", and
by default it was always computed with `apply_bt_rows_into`. The computation order, the rounding and the buffers change in no way (bit-invariant).
`cargo check --features cuda --tests` = exit 0, engine 242 passed / 17 ignored, server all green.

**What worked next** — "Delete an opt-in with default off as soon as it is found to be numerically broken" became the convention.
P36's 4-lane version took the same shape (opt-in, default off), but we keep it because its numbers are correct and it merely has no effect.

### P15: removing the host wait between passes of the GDN chunk scan (2026-10-23, `c0b19d07`)

**Intent** — With 27B q4 at 65k, `sl_lay_gdn_scan` 6,613.7 ms (10.5%, 48 layers = 138 ms/layer) and `sl_lay_norm_in` 1,855.2 ms (2.9%, 64 layers = 29 ms/layer).
The starting point was checking the instruction's 2 hypotheses: "is it 1 launch per chunk" and "is the host ordering bad".

**What changed** — `c0b19d07` / `crates/denno-engine/src/cuda.rs`. We no longer stop the host between pass 1 → pass 2 → pass 3.
The 3 passes are issued to the same stream, so FIFO guarantees the order in which pass 2 reads pass 1's W/M and pass 3 reads pass 2's Sinit.
The only synchronization point is "the 1 point where the host reads the result", and we kept 1 so that `DENNO_PROFILE`'s breakdown (`D2H_GDN_PF_*`) keeps
projecting the 3-pass execution time as sync (if a display-only interval becomes 0 the scan looks free). We change not one of the kernel, arguments, grid, block, shmem,
chunk split, issue order and buffers → the output bits are unchanged.

**Stumble** — Both hypotheses do not hold. The chunk loop was on the device side from the start (chunk = blockIdx.x), and there are 0 host round trips per chunk.
3 launches per layer makes the launch cost 15 us = 0.01%. The identity of the 138 ms/layer was **local memory bandwidth inside the kernel**:
pass 1 places `float Wcol[256] + Mcol[256]` in local memory per thread (dk is a runtime argument, so 256 elements = 2,048 B/thread), and
reads and writes it row by row → estimated 265 GB/layer. 147 ms at 1.8 TB/s, matching the measured 138 ms.

**What we learned** — `gdn_scan` goes from 3 launch/layer 3 sync/layer (144/req) → 3 launch/layer 1 sync/layer (48/req).
`ctx.sync()` is not `cuStreamSynchronize` but `cuCtxSynchronize` (a drain of the whole context), so
the device-wide drain drops by 96 times per request. Prediction (not measured on device): `sl_lay_gdn_scan` 138 → 138 ms/layer
(the interval is the kernel execution time, and the removed round trip is under 0.1 ms per layer), wall 63.05 s → −3–15 ms (0.005–0.02%).
316 passed / 36 ignored, CPU gate 279 passed. The actual arithmetic of the same layer is 132 GFLOP, and at q4 GEMM's effective 30 TFLOPS that is 4.4 ms
= 3% efficiency. It does not move on the host side.

**What worked next** — The reading "a phase value is a split of wall, not kernel time" (KPI §5.1) was settled, and
the need for P40's CUDA event measurement was born here. Cutting the 138 ms/layer for real requires the .cu
(put the state arrays in shmem / split pass 1 into W-only and M-only to remove the 265 GB of local round trips. Prospect 138 → 10–20 ms/layer).

### s241: fusing pass 1's decay and dot product into 1 pass (2026-10-23, `0d9faef7`)

**Intent** — s192 had settled that pass 1's bottleneck was not the "capacity" of local memory but **the local access instructions themselves**.
The 2 options that reduce capacity (shrink the arrays to 128 elements / W in 128 registers and M in shared memory) were already rejected with "no change" and "13% worse" respectively
(stack frame 2,080 B → 1,056 B / 0 B, prefill 3,161 token at 9.75 / 9.77 / 11.02 s). The remaining direction is to reduce the instruction count.

**What changed** — `0d9faef7` / `crates/denno-engine/kernels/gdn.cu` / `kernels_gdn.ptx` (regenerated).
In pass 1 we fused the application of the decay and the dot product into 1 pass. The diff is 1,945 lines of .cu and 1,438 lines of PTX.

**Stumble** — The change is large, and following the diff in review is hard. "bit-identical" is the claim in the commit heading, but
the body has neither the reason the rounding order does not change nor a record of the anchors matching (by P36 the practice had become to state "anchor 3/3 MATCH" explicitly).

**What we learned** — At 65k, GDN pre **−5.8%**, bit-identical. The only condition is "65k" in the commit body, and there is no record of the
model, quantization, number of runs, or whether it is median or min (not recorded). There is no record of the test count either.

**What worked next** — The same fused loop became the target of s243's unroll. s192's plan "if we give up bit identity, 1 column can be
shared over 4 lanes" is still held back, and it is tried as opt-in in P36.

### s243: setting the fused loop's unroll to 8 (2026-10-23, `8503925c`)

**Intent** — The loop fused in s241 still has a high ratio of loop control and address arithmetic. Unrolling amortizes it.

**What changed** — `8503925c` / `crates/denno-engine/kernels/gdn.cu` (8 lines) / `kernels_gdn.ptx` (regenerated).
The unroll of pass 1's fused loop is now 8. Registers went 44 → 40.

**Stumble** — Unrolling trades against register pressure, and more registers would drop occupancy. Here it went 44 → 40 regs, i.e. **decreased**, so
that worry did not arise. This is the opposite direction from s192's register version, which was 255 regs and 13% worse.

**What we learned** — pre a further **−1.8%**, bit-identical. The conditions (model/quantization/number of runs/median or min) are
not recorded in the commit body (not recorded).

**What worked next** — Unknown from the record. However, s192/P36's observation "optimizations that add registers can be cancelled by occupancy" and
the fact that here the regs decreased and it worked straightforwardly form a pair, which is circumstantial evidence for P36's interpretation (what cancelled the gain was occupancy).

### P36: pass 1's 4-lane version (DK=128) — no effect in measurement (2026-10-23, `3e288aa2`)

**Intent** — s192 held back the plan "if 1 column is shared over 4 lanes, the local access instructions become 1/4 but the order of the sum changes";
we implement it as an opt-in with DK=128 fixed and settle it by measurement.

**What changed** — `3e288aa2` / `crates/denno-engine/kernels/gdn.cu` / `kernels_gdn.ptx` (regenerated).
We added `dno_gdn_chunk_pre_l4`. The old version had 1 column = 1 thread and placed `Wcol`/`Mcol` (2,080 B) in local memory, while
the new kernel shares a column over 4 lanes holding 32 rows each, giving stack frame 0 B and 128 regs (the cap via launch_bounds), and
the local access instructions become 1/4. opt-in `DENNO_GDN_LANES=4`, default is the old kernel (off).

**Stumble** — **No effect in measurement**: with 27B q4, 16k, median of 4, `DENNO_GDN_LANES=1` (default) 9.69 s and
`DENNO_GDN_LANES=4` 9.88 s, rather a slight increase. Occupancy dropping to 1 block/SM (512 threads) cancels the gain.
That is, the amount of local memory was not the bottleneck (consistent with s192's conclusion).

**What we learned** — We also checked the breakdown: of the `gdn_scan` phase 6,965 ms, `gdn_pf_sync` is 6,887 ms (99%), alloc 55 ms / launch 41 ms / d2h 8 ms.
**What the sync contains (the split of the 3 kernels) is not measured**, and the next task was stated to be host-side event measurement (`DENNO_GDN_KPROF`).
Anchor 3/3 MATCH after the PTX regeneration as well.

**What worked next** — P40's CUDA event measurement became the direct next step. The 4-lane version was kept with default off
(the numbers are correct and it merely has no effect, so its nature differs from s96's deletion target).

### P40: isolating the real time of the chunked 3 kernels with CUDA events (2026-10-23, `bf88ab26`)

**Intent** — For 27B q4's 65,000 chars prefill, the `sl_lay_gdn_scan` phase = 6,965 ms (19.4%), but with `DENNO_PROFILE`'s `gdn_pf_*` it
only breaks down as far as `sync` (13,774.9 ms / 2 requests). The split of pass 1/2/3 and the possibility that "the sync is absorbing the remainder of the upstream `gdn_proj` GEMM"
(P27's phase attribution) remained unmeasured.

**What changed** — `bf88ab26` / `crates/denno-engine/src/cuda.rs`. Only when `DENNO_GDN_KPROF=1`, we record
4 events (`CU_EVENT_DEFAULT` = timing enabled) in `gdn_scan_chunked_impl`, and right after the existing synchronization (the 1 after the 3 passes) finishes we read the
3 intervals with `cuEventElapsedTime` and accumulate them in ns into static `AtomicU64`s. The output is 1 line at dump time and at exit:
`[GDN_KPROF] pre=<ms> state=<ms> scan=<ms> calls=<n> total=<ms>`.

**Stumble** — `kernel_invariants`'s `gdn_kernel_shared_layout_matches_host_smem` was red. Because P36 added pass 1's 4-lane version
(`dno_gdn_chunk_pre_l4`), there are now 2 kernels with the same `sc[]` allocation. We updated the expected value from 1 → 2 and
made it green again (we did not touch the `.cu`). **Measurement only**: we change not one of the launch function, arguments, grid/block, issue order and synchronization position, and
the accumulation is written only into display-only statics (it does not flow into the numeric path or the anchor values). If `DENNO_GDN_KPROF` is unset, `kprof_begin()`
simply returns `None` and no event creation or measurement happens (1 branch per site).

**What we learned** — 330 passed / 0 failed / 40 ignored (baseline 327 + 3 measurement tests), `kernel_invariants` 11 passed, non-cuda `--lib` 286 passed.
The output of the existing `DENNO_PROFILE` / `DENNO_PF_DBG` is unchanged when the env is unset. **The on-device ms split of pre/state/scan is not measured**
(this environment has no GPU).

**What worked next** — The intervals were defined as "kernel execution + gaps between issues", including the host-side stalls between passes, so
the next step was settled: once measurement reveals the dominant pass, aim at that pass's lane/chunk split.
Since P36's 4-lane version is already settled as having no effect, "the dominant one may not be pass 1" was recorded at the same time.

### P43: why `[GDN_KPROF]` produced not a single line on the device (the output point depended on the dump) (2026-10-23, `88e9e86d`)

**Intent** — On the device at 65k we set `DENNO_PROFILE=1 DENNO_GDN_KPROF=1 DENNO_FC_KPROF=1`, yet `[FC_KPROF]` appeared twice and
`[GDN_KPROF]` 0 times. When the instrument goes silent the next decision cannot be made.

**What changed** — `88e9e86d` / `crates/denno-engine/src/cuda.rs` / `crates/denno-engine/src/prof.rs`.
We call `gdn_kprof_report_at_prefill_end()` from `pf_timing_read()` (end of prefill, after the final row's D2H). It reads only display-only statics, so
**it adds not one synchronization, launch or event creation**. When the dump path (`DENNO_PROFILE`) is alive the summary side prints the same line, so
we do not print it on the prefill side (avoiding a double line). The line is extended to `[GDN_KPROF] pre=<ms> state=<ms> scan=<ms> calls=<n> miss=<n> total=<ms>`.

**Stumble** — P40's hypothesis "there is no sync after pass 3, so NOT_READY" was **refuted**. The `ctx.sync()` after the 3 passes is still
in `gdn_scan_chunked_impl` after s173/P15, so `finish()`'s read happens after completion. The actual cause was **the output point**: P40
emitted the line only from `prof::summary()` and the thread-exit hook, and production startup unsets `DENNO_PROFILE`, so
`prof::enabled()` is false → no dump, and the worker thread does not exit either so the exit hook does not run. `[FC_KPROF]` was emitted
directly from inside prefill with `eprintln!`, so it came out with 1 env (this difference is what the asymmetry of the observation was).

**What we learned** — `miss=<n>` is the number of calls whose interval could not be read and was discarded (not finished, negative value, missing event, event creation failure).
With the env ON the line appears even when `calls=0`, so "silently nothing comes out" does not happen. 337 passed (kprof_tests 3 → 5), non-cuda `--lib` 291 passed,
`kernel_invariants` 19 passed (pinning in source the wiring to the end of prefill and "the output point does not call sync/launch/memcpy").
The on-device pre/state/scan ms and the actual miss values are not measured.

**What worked next** — "Design the instrument all the way to where it comes out" became the convention. P45's deferred read (moving the read to the next existing synchronization and
counting unfinished ones as misses) uses this miss rule as it is.

### P45: taking the `ctx.sync()` after the 3 passes out of the device output path (2026-10-24, `d19aa70b`)

**Intent** — The `ctx.sync()` after the 3 passes is `cuCtxSynchronize` (a drain of the whole context) and 48 of them remained per request.
In prefill's residual-resident path y is not read on the host, so this one is unnecessary. The synchronization we removed is only 1 kind at 1 site.

**What changed** — `d19aa70b` / `crates/denno-engine/src/cuda.rs`. We removed `gdn_scan_chunked_impl`'s `ctx.sync()` after the 3 passes
only from the prefill residual-resident path (`y_dst = Some`), and left it in the host output path (`y_dst = None`) (the 1 before reading y with `to_vec`).

**Stumble** — We had to assemble 3 safety arguments. (1) The 3 passes and the following o_proj / the next layer's rmsnorm and GEMM are issued in order
to the same stream (legacy NULL), so FIFO suffices. (2) The state write-back is `cuMemcpyDtoDAsync` on the same stream.
(3) The next layer's beta/alpha H2D is `cuMemcpyHtoD` (a host-synchronous copy), so the buffer reuse order is preserved too.
The original reason for the 1 removed call was "project the 3-pass execution time into `gdn_pf_sync`" = display only (the same basis on which s173/P15 removed the 2 between passes).

**What we learned** — The numbers and the issue sequence are unchanged (no change to kernel / arguments / grid / block / shmem / issue order; `.cu` / `.ptx` / PTX regeneration untouched).
The expected reduction is **10–100 ms/req = 0.03–0.3% of 35.8 s (estimated, not measured)**. Most of `gdn_pf_sync` 13,774.9 ms (48 calls/req) is
the 3-pass GPU execution itself (the phase total is 99% of wall, and s192 settled that pass 1 is local-memory bound), so
it is not "6.5 s disappears". What can be cut is `sync − (pre+state+scan)`, and the guide for the upper bound is the measured `gdn_pf_launch` 20 ms/req.
The structural gain is that the device-wide drain disappears per layer (it does not drag in concurrent requests). 339 passed (+1),
`kernel_invariants` 21 passed (+2). As a side effect, the residual-resident path's `gdn_pf_sync` / `gdn_pf_d2h` rows become 0 ms / 0 calls (display only).

**What worked next** — The list of synchronizations we kept (chain_pf_h2_join's event_sync / the host output path's `ctx.sync()` / `PfSink::Host`'s D2H wait /
the i8 path `gdn_scan_batch_oproj`'s D2H wait / the synchronous D2H of `DENNO_PF_OVERLAP=0`) remained as
the implementation list of "synchronize only at the point where the host reads the result". What works next on the GDN side is the bandwidth itself (P15's prospect 138 → 10–20 ms/layer), and
this is a structural change of the .cu, so it is not started.

---

## Rejected approaches

Judgments that looked reasonable at the time but were discarded later. We place only references to the corresponding
entries in the body.

### Fusing W/M into 2 passes — folding pass 2 into pass 3 to get 2 launches (s89/s90 `ddf7201a`)
- What we tried: of the chunked 3 launches + 3 syncs, fold pass 2 (the serial propagation of Sinit) into chunk 0 of pass 3 to get 2 launches (`dno_gdn_chunk_scan_hoist`).
- Measured: cold A/B, 5,686 token, 3 alternating rounds, OLD `b502e34` 13.362 / 13.408 / 13.404 s, NEW `98c83cb` 13.345 / 13.377 / 13.375 s → **−0.03 s**. Does not reach the acceptance of 0.3 s.
- Why we discarded it: the writer of the row `ci*chunk_len − 1` that a folded chunk ci ≥ 1 reads is not in the same launch. GDN layers are not contiguous and full layers sit in between, and the head of an interval always breaks on the host — with these 2, 3 launches is the only correct path. **`route_hoist` is always false** (the hoist version never runs even once).
- Record: body s89/s90. We kept only the bookkeeping and the invariant tests, and afterwards moved to the "reduce synchronization" direction (P15/P45).

### The 2 array options for pass 1 — the direction of reducing capacity (s192, recorded in body s241)
- What we tried: 2 options to reduce the capacity of local memory. (a) shrink the arrays to 128 elements, (b) W in 128 registers and M in shared memory.
- Measured: prefill 3,161 token at 9.75 / 9.77 / 11.02 s (default / (a) / (b)). The stack frame went 2,080 B → 1,056 B / 0 B. The register version was 255 regs and **13% worse**.
- Why we discarded it: (a) no change, (b) cancelled by occupancy. The bottleneck was settled to be **the number of access instructions, not the capacity** of local memory (s192).
- Record: body s241 (quoted as s192's result). The direction afterwards changed to fusion that reduces the instruction count (s241/s243).

### Lane splitting of pass 1 (4-lane version, DK=128) (P36 `3e288aa2`)
- What we tried: share 1 column over 4 lanes to make the local access instructions 1/4 (`dno_gdn_chunk_pre_l4`, opt-in `DENNO_GDN_LANES=4`).
- Measured: 27B q4, 16k, median of 4: 9.88 s with 4 lanes against 9.69 s default, a slight increase. stack frame 0 B and 128 regs met the design values.
- Why we discarded it: occupancy dropped to 1 block/SM (512 threads) and cancelled the gain. It became the corroboration that the amount of local memory is not the bottleneck. The numbers are correct so we kept it with default off (its nature differs from s96's deletion target).
- Record: body P36. Anchor 3/3 MATCH after the PTX regeneration as well.

### Fusing pass 1's W/M update from 2 passes into 1 (P49)
- What we tried: in pass 1's W step, W and M were written in separate i-ascending loops; we folded them into 1 loop and accumulate `skv` / `mk` alternately at the same i (`#pragma unroll 8`; under `-fmad=false` the f32 addition order is unchanged).
- Measured: 27B q4, 1 request of the 65k class, with `DENNO_GDN_KPROF=1`, the scan phase's pre went **4,385 → 6,377 ms (+47%)** and the 65k wall **38.41 → 40.18 s**. The numbers matched (same wide md5) but the time increased.
- Why we discarded it: it got slower even though we reduced the loop count to 1. We see it as register pressure rising because 1 iteration keeps the accumulators for 2 arrays alive at once, and combined with `unroll 8` the scheduling collapsed (estimated). This is the opposite direction from s241's "3 arrays in 1 loop" working, and we learned that **pass 1 is not uniformly favorable to fusion**.
- Record: the branch was discarded. The same lesson reads well as a pair with body s241 / s243 (the side where 2 arrays and unroll 8 worked).

### The GPU kernel for GDN's beta/alpha projection (s78/s79 `85a94052` → removed in s96 `55586f03`)
- What we tried: build `dno_matvec_rows_f32`, which computes `[m,k] × [n,k]^T → [m,n]` with 1 block = 1 output, isolate it behind opt-in `DENNO_GPU_GDN_PROJ=1`, and try it on the device.
- Measured: with k=4096 nv=32 m=33, `out[0]` was **−17.205 vs −8.410** (CPU reference), maximum absolute difference ~1e2. A runtime error on the k=1 shape → every subsequent call falls to the CPU.
- Why we discarded it: determinism holds but the computation itself disagrees (an implementation bug that rounding cannot explain). It remained as dead code with default off, so s96 deleted the Rust side. `.cu` / `.ptx` were left because PTX regeneration is required.
- Record: body s78/s79 and s96. The same shape was solved by the CPU-side transposed layout (s66, `40_prefill_perf.md`).

### P15's 2 hypotheses — per-chunk launches and host ordering (P15 `c0b19d07`)
- What we tried: check the cause of `sl_lay_gdn_scan` 138 ms/layer with 2 hypotheses, "is it 1 launch per chunk" and "is the host ordering bad".
- Measured: the chunk loop was on the device side from the start (chunk = blockIdx.x) and there are 0 host round trips per chunk. The launch cost of 3 launches per layer is 15 us = 0.01%.
- Why we discarded it: neither holds. The identity of the 138 ms/layer is the local memory bandwidth inside the kernel (estimated 265 GB/layer, 147 ms at 1.8 TB/s, matching the measured 138 ms).
- Record: body P15. Only the rejection of the hypotheses remains, and cutting it requires a structural change of the .cu (prospect 138 → 10–20 ms/layer).

### P43's hypothesis — NOT_READY because the sync after pass 3 is missing (P43 `88e9e86d`)
- What we tried: chase the cause of `[GDN_KPROF]` producing not a single line on the device with P40's hypothesis "there is no sync after pass 3, so the interval cannot be read".
- Measured: `[FC_KPROF]` appeared twice and `[GDN_KPROF]` 0 times. The `ctx.sync()` after the 3 passes was still there, so the read happened after completion.
- Why we discarded it: the cause was the output point. P40 emitted the line only from `prof::summary()` and the thread-exit hook, and production startup unsets `DENNO_PROFILE`, so no dump appeared and the worker thread did not exit either. We changed it to emit directly from the end of prefill.
- Record: body P43. "Design the instrument all the way to where it comes out" became the convention.
