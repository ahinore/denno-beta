# Volume 5 prefill performance — the bottleneck was not the arithmetic but the round trips and the copies

> [Index](README.md) / Previous: [Volume 4 Decode performance](30_decode_perf.md) / Next: [Volume 6 GDN](50_gdn.md)
> Japanese original: [../40_prefill_perf.md](../40_prefill_perf.md)

Dates are the commit author dates in JST.

## What this volume covers

- The slowness of prefill was not the amount of arithmetic but the **host round trip for every single linear-layer call** and the **copy of the return value**. 13,252-token cold prefill of 1.5B int8 went 395.8 s → 12.2 s, and 5,686 tokens of 9B int8 went 46.2 s → 10.11 s.
- The bottleneck moved "O(m²) CPU attention" → "CPU GEMM" → "transfer + synchronization structure" → "result copy" → "activation round trip". At each stage we re-measured where the wall was and moved on.
- Before things got faster, situations where **the measurement lies** came up repeatedly: a false hotspot (T5a), a path that silently fell back to a single thread (s62), a bucket that points at a whole block (s84), an overwritten measurement label (s87).
- We keep 2 withdrawn attempts: P17b's single-shot contiguous pinned D2H (measured 2.2x worse) and T5a's opt-in gate (2.85x regression by default).

## Background knowledge

- prefill is the path that runs a forward pass with n>1 in one go (`Qwen35::forward` → `forward_prefill`). The difference from decode is that the row count m of the matrices becomes the token count.
- Numeric contract: the int8 dot products are accumulated in i32, where associativity holds, so changing tiles, thread counts or row concatenation stays bit-identical. Only attention was relaxed in P17g (determinism is maintained).
- The phase buckets of `DENNO_PF_DBG` are **a split of wall time**, not a split of kernel time (settled in P27). In this volume we do not read a phase's ms as "the cost of that kernel".
- Notation: we state measured / conversion (estimated) / not measured explicitly. We do not quote a phase's % as a current value.

## Reading order

The entries are in chronological order. For readers in a hurry: (1) the ceiling table below → (2) s63 (result copy) → (3) s66 / s68 (layout and copy path) →
(4) s70 (activation round trip) → (5) the T5a revert; picking up only the failure and measurement stories first works well. P17k is a 6-step buildup, but each step can be read independently.

## Phases covered

| phase | date | commit | one-line summary |
|---|---|---|---|
| P17b | 2026-10-13 | `197a06c` | Connect the int8 linear layers to the real path and decide where prefill stays on the CPU |
| P17i | 2026-10-13 | `f2c18315` | Move long prefill's linear layers to the GPU with a tiled GEMM |
| P17j | 2026-10-14 | `cededce6` | Replace the O(m²) CPU attention with a flash-style kernel |
| P17k | 2026-10-14 | `4bf11f5e` | Make prefill resident in 6 stages starting from the FFN block |
| B1 | 2026-10-15 | `455dbe0a` | Move rmsnorm to the GPU and make the inside of the layer device-resident |
| Variant B | 2026-10-16 | `875dd3cd` | Parallelize CPU prefill with rayon and keep bit identity |
| s48 | 2026-10-17 | `a2e5368c` | qwen35's full attention in 1 kernel launch |
| T5a revert | 2026-10-19 | `083218d6` | Revert a gate built on a false hotspot |
| s62 / T5b | 2026-10-19 | `d9fef26c` | Fix the n<64 projection silently falling into the single-thread path |
| s63 | 2026-10-19 | `6d34d90e` | The bottleneck was not the GPU but the 21.8 s return-value copy |
| s66 | 2026-10-19 | `ac34b1ea` | The slowness of the small-n projection was a side effect of the transposed layout |
| s68 | 2026-10-19 | `b682c3f5` | The D2H through pinned staging was eating 5x the GEMM body |
| s69 | 2026-10-19 | `1f588b4f` | Remove the "via pinned + to_vec" from the chain D2H |
| s70 | 2026-10-19 | `9ec23f3c` | Make prefill's FFN a device-resident chain and cut activation round trips from 4 to 2 |
| s71 | 2026-10-19 | `91b54265` | Restore the direct D2H of s69 that s70 had rolled back |
| s73 | 2026-10-19 | `57405fe9` | Parallelize full attention's prepare over tokens |
| s74 | 2026-10-19 | `e0ea110b` | int8 GEMM's per-thread tile to 8x16 |
| s75 | 2026-10-19 | `5bca9b7e` | Fix smem alignment and row coverage and recover |
| s80 / s81 | 2026-10-19 | `b69e0190` | Concatenate the MLP's gate+up at load time into 1 GEMM |
| s84 | 2026-10-19 | `66698f68` | Show that rmsnorm's "0.4 s" was a misnomer for the whole block |
| s87 / s88 | 2026-10-20 | `e547749d` | Make the phase accounting additive and measure d2h_state per issue site |
| P14 | 2026-10-23 | `c4c6f569` | Fold the FFN's gate/up into 1 GEMM |
| P20 | 2026-10-23 | `5f222949` | Write the full layer's q directly into pinned staging |
| P21 | 2026-10-23 | no commit | Choose the q4 GEMM row tile from the projection's n (implementation discarded) |
| P25 | 2026-10-23 | no commit | Settle the breakdown of ffn_down and decide to add nothing |
| P27 | 2026-10-23 | no commit | Audit from the mechanism what the phase accounting does not measure |
| P28 | 2026-10-23 | `3c383a7f` | Measure full_core's host side and move pointless allocations |
| P29 | 2026-10-23 | `86c44535` | Fix 4 points on how to read the phase accounting in the docs |
| P30 | 2026-10-23 | `70675d32` | Move PF's env to the server startup side and kill the empty runs |
| P31 | 2026-10-23 | `b5f97db8` | Pin the static invariants of the prefill chain without a GPU |
| P34 | 2026-10-23 | `a9b91394` | Turn the smem cast scanner's known limitations into tests |
| P36 | 2026-10-23 | `9e7de184` | Make the scanner take in the host launch block shape |
| P39 | 2026-10-23 | `255e147f` | Make the scanner take in the launch grid and judge blockIdx rows |
| P41 | 2026-10-23 | `1068bd0d` | Add a device that splits the full_core phase into 4 intervals |
| P44 | 2026-10-23 | `c7b0991e` | D2H V directly into the host KV cache rows instead of back into scratch |
| P46 | 2026-10-24 | `2db6448d` | Chunk the full layer's att round trip and overlap D2H/H2D with the gate |
| P47 | 2026-10-24 | `b44ba551` | Move full_qk's D2H destination to pinned staging |
| P48 | 2026-10-24 | `6f5f4a3c` | Remove the per-row intermediate Vec from prepare |
| P50 | 2026-10-24 | `e841db48` | Break down tail and show that att's H2D is already pinned |
| P51 | 2026-10-24 | `ec7d4f31` | AVX2 16 lanes for GDN's host beta/alpha projection |
| P52 | 2026-10-24 | `b6e3fa27` | Widen the q4 GEMM's per-thread tile to 8x8 |

## Gap between the ceiling (theoretical values) and measurements

| item | theoretical value or upper bound | measured | cause of the gap |
|---|---|---|---|
| PCIe H2D | 12–25 GB/s (a placeholder value from s170, estimated) | 14.2 GB/s (the transfer bench of P17k step 4) | the bandwidth itself is near the limit. Capped by small transfers and synchronization |
| D2H (pinned only) | 14.3 GB/s (s68's measurement = the limit of this environment) | 19.5 ms / 279 MB | — |
| D2H (pinned → heap memcpy) | 14.3 GB/s | 79.8 ms / 3.5 GB/s | the memcpy is 4.7 GB/s |
| D2H (pageable direct) | 13.9 GB/s (s68) | 20.1 ms / 13.9 GB/s | no difference = going through pinned was double work |
| prefill's round-trip bytes | 41.8 GB @12,850 tok (arithmetic from the shapes, estimated) → 1.7–3.5 s = 2.7–6.2% of wall | wall 56.3–61.8 s (as of s170) | round trips are no longer the wall |
| int8 GEMM | no material for the theoretical TOPS (s74 only states "an order of magnitude below the theoretical performance") | effective 10.6 TOPS (s74, 8x8 tile) | too many smem loads per dp4a and spills |
| GEMM effective (f32 equivalent) | no material | 13.4 TFLOP/s (s63, including GEMM+D2H+sync) / 14.2 TFLOPS (s170's 3-way split, estimated) | transfers and synchronization are included in the denominator |
| TFLOP/s by phase | no material | ffn_gate about 16 / gdn_proj about 7.6 TFLOP/s (the conversion of KPI §5, estimated) | gdn_proj is lower because the projections are split |
| rmsnorm | 3.1 GB of reads and writes → about 3 ms at 1 TB/s, about 31 ms even at 100 GB/s (s84's estimate, estimated) | not measured (invisible from the host) | does not reach the 0.25 s decision line, so not optimized |
| single-shot contiguous pinned D2H (P17b) | predicted −0.37 s/request (estimated) | 9.78 → 22.14 s (+772 ms/layer) | concatenating 3 shots into 1 serialized the transfers |

### P17b: connecting the int8 linear layers to the real path and the line "prefill stays on the CPU" (2026-10-13, `197a06c`)

**Intent** — Connect the int8 kernel built in P17 to the real path of every int8 linear layer, and confirm bit identity and the speedup end to end. The interest on the prefill side was "may we use the GEMV-type kernel (1 warp per output) for m>1".

**What changed** — Hooked `maybe_gpu_linear` in `crates/denno-engine/src/cuda.rs` into the int8 paths of `quant::Linear::apply_bt`
/ `apply_batch_bt` / `apply`. Zero changes to model-side code. Weights stay resident in VRAM keyed by uid; on the host side,
pinned buffers + the seriality of the default stream cut synchronization to once per linear layer.

**Stumble** — (1) When we keyed the weight registry by the buffer pointer of `Quant8Linear.w`, freeing and reallocating the Vec reused the address,
and a collision where weights with different contents landed at the same address showed up in the tests → we assign a uid at construction time and fixed it. (2) GEMV re-reads the weights in proportion to the row count m, so it is weak at prefill:
**measured 0.39x at m=128**. → We left `m > DENNO_CUDA_MAX_M` (default 16) on the CPU.

**What we learned / remaining issues** — e2e 200-token greedy went CPU 9.08 s → GPU 2.66 s (3.4x), decode 22.1 → 75.8 tok
/s. Per op (600 decode steps + prefill total) was TOTAL 28,591 → 9,686 ms, and the linear layers shrank from 86% →
62% while attn_core became the largest op (28.6%). **prefill's linear layers were deliberately on the CPU at this point**, and speeding up long prefill had not been started.
The synchronization reduction was about 40 µs → about 28 µs per linear layer.

**What worked next** — This m threshold became the basis of P17i's GEMM path (`dno_gemm_i8` at m ≥ 256). Reapplying the same
"batch the transfers" idea to the q4 family was the 2026-10-23 reapplication of P17b (`7e9ea37d`, full-layer q|k|v as 1 contiguous pinned
D2H), but **on the real machine it got worse, 9.78 → 22.14 s (+772 ms/layer), and was reverted (`f55c707c`)**. The prior prediction was −0.37 s/request
(`REPORT_P17B.md`). Concatenating 3 shots into 1 serializes the transfer.

### P17i: moving long prefill's int8 linear layers to the GPU with a tiled GEMM (2026-10-13, `f2c18315`)

**Intent** — Cold prefill of 1.5B int8 at t≈2810 was about 100 s. In the profile, 83% of prefill was the CPU GEMM of the
linear layers (gate_up 47 s, down 29 s).

**What changed** — `dno_gemm_i8` in `kernels/kernels_int8.cu` (64×64 tile, 4×4 outputs per thread, dp4a, A/B tiles into shared
memory). Dispatch uses `m > max_m() && m >= gemm_min_m()` (default 256) for GEMM, `m <= 16` for the existing
GEMV, and **17–255 stays on the CPU** (at small m the per-layer synchronization costs more than the CPU GEMM).

**Stumble** — We first nailed down the basis for bit identity: int8 dot products are accumulated in i32, where associativity holds, so they do not depend on tiling or
order. On the f32 side we use the same `((dot as f32) * ascale) * scale[j]` as GEMV. When k is not a multiple of 4/32 we handle it with zero padding at the tile boundaries.
The tests covered 8 shapes including the tile boundaries 63/64/65/100/128/130 and k=1533/17911, and confirmed **bit identity with the GEMV kernel on the same input**.

**What we learned / remaining issues** — 1.5B int8, longreq2 (t=2827 / max 256): cold **about 100 s → 32.4 s**,
warm (prefix cache hit) 3.70 s, short text (t=38, 2 tok) 11.1 ms (identical to P17h = the old
CPU path because of the GEMM threshold). The cold prefill response matched the CPU prefill exactly. PTX sha256 `9021673e…`. The remaining bottleneck moved to the O(m²)
CPU attention (next entry).

**What worked next** — "Integer accumulation stays bit-identical whatever the order" became the common premise of every later GEMM tile change (P17k step 6, s74 / s75).

### P17j: GPU offload of long prefill attention (flash-style online softmax) (2026-10-14, `cededce6`)

**Intent** — Of the TOTAL 395.8 s of 13,252-token cold prefill, attn_core was 334 s (84.5%).
After P17i put the linear layers on the GPU, the O(m²) CPU attention dominated on its own. The procedure document's next candidate, "sampler kernel",
measured 0.001% and was dropped.

**What changed** — `dno_attention_prefill` in `kernels/kernels_elem.cu`. grid=(heads,
ceil(n/PF_BQ)), block=(dh), dynamic shmem, PF_BQ=16 / PF_BK=32, causal, exp via `dno_expf`
(glibc reproduction). K/V reads P17g's GPU KV mirror ([t_total, kv_dim] row-major) directly. Determinism is fixed by ascending tile order, ascending
kk within a tile, and output dimension d owned by a single thread (no atomics).

**Stumble** — (1) The exactness contract is the same as P17g: "relax attention only, keep determinism". The difference from the CPU reference is maxabs
1.49e-7. (2) A latent flake in the test infrastructure surfaced: tests that swap the global `GPU_ATTN` flag raced under parallel execution → serialized with `test_lock()`.
Because `chain_upload` shares pinned memory, 2 consecutive uploads without an intervening sync overwrite before the DMA completes → we added
`ctx.sync()` on the test side (the production path has a download sync between uploads, so there is no problem).

**What we learned / remaining issues** — For 13,252-token cold: **wall 395.8 → 58.6 s, attn_core 334 → 11.8 s**. For 3,352-token
cold: TOTAL 15.8 s, attn_core 1.03 s (6.5%). The remaining bottleneck was gate_up_proj at 49.8%. Outside the constraints dh ≥
PF_BQ and shmem ≤ 48 KB we fall back to the CPU. PTX sha256 `c41520a9…` (compiled twice and compared).

**What worked next** — The design difference "prefill attention completes in 1 launch, so we can fall back without redoing the whole forward pass"
became the pattern for the per-layer fallback of B1 / s48.

### P17k: making prefill's linear layers device-resident — 6 steps starting from the FFN block (2026-10-14, `4bf11f5e` → `35421798`)

**Intent** — After P17j the remainder of the 13,252-token cold run (PROFILE TOTAL 73.6 s) was gate_up_proj 36.0 s /
attn_core 12.1 s / swiglu 8.2 s / down 5.2 s. Moreover **GPU utilization during prefill was almost 0%**
(spikes only), and wall time was dominated not by kernels but by CPU work + transfers + the synchronization structure. In the bench, 1 m=13,252 gate_up took 508 ms and the
output was 949 MB.

**What changed** — We split the commits by stage. (1) `4bf11f5e`: `dno_swiglu_int` + `maybe_gpu_ffn` chain
gate_up GEMM → swiglu → quantization → down GEMM, with only the 81 MB down output going D2H. (2) `870bb871`:
`maybe_gpu_oproj` reads CH_ATT directly, and attention's exp goes `dno_expf` → `exp2f`. (3) `99d29406`:
invert the grid from (heads, row blocks) → (row blocks, heads), smem row stride dh+1, PF_BQ 16→48 / PF_BK
32→8. (4) `67e243ad`: `dno_rope_qkv_pf` ropes q into a contiguous region and writes k/v directly into KV mirror rows. (5) `d2b81878`:
GEMM to 128×128 tiles, 8×8 per thread, smem stride 33 B. (6) `35421798`: software pipeline of the K/V tiles and
2r×2k quad dot.

**Stumble** — (1) Even with exp2f, attn_core went 11.86 → 11.11 s, essentially unchanged → "expf is the bottleneck" was wrong; the rest was the transfer + synchronization structure.
(2) The mystery of error 700 (invalid address) once smem exceeded ~43 KB was actually **the host constants `ATTN_PF_BQ/BK` left at 16/32, out of sync with the kernel
(48/8)** (grid about 4x too large + insufficient smem → out-of-bounds write). (3) After a source change the PTX sha256 was identical to before, and we deployed
with the constant mismatch → E2E became nondeterministic garbage. Since then md5 comparison is mandatory at synchronization. (4) Qwen2.5-1.5B carries QKV bias, and
the initial "bias-free only" gate fell back for every layer → we implemented `[qb|kb|vb]` concatenated bias addition.

**What we learned / remaining issues** — The wall of 13,252-token cold went 77.9 (before step 1) → 33.6 → 30.7 → 24.5 → 17.6 → 13
.0 → **12.2 s** (step 7 was cold twice). qkv_proj 4,511 → 0.7 ms, rope 2,754 → 0.4 ms,
down_proj 5,009 → 1,147 ms. The attention kernel alone went 368 → 112.4 → 69.3 ms/call. maxabs
vs CPU is 1.79e-7 (n=37) / 1.97e-7 (n=2056), determinism maintained. What remains is attn_core 3.5 s (47%) and
rmsnorm 2.3 s.

**What worked next** — The "eliminate round trips with device residency" pattern fed B1's whole-prefill chain, and the findings on smem bank conflicts and grid order
fed the s74 / s75 GEMM tiles. The host/kernel constant mismatch became the motive for s75's `static_assert` and the static inspection tests.

### B1: moving prefill's rmsnorm to the GPU and the device-resident hidden-state chain (2026-10-15, `455dbe0a`)

**Intent** — After P17k, rmsnorm was 2.3 s of what remained. Moreover rmsnorm was serial with h's H2D (81 MB/layer), and per layer
the round trips of h/proj/down amounted to 81 MB × 4 = 324 MB (about 9 GB/request at 13k prefill).

**What changed** — Added `forward_prefill_gpu` and restructured prefill into "embedding H2D once → everything inside the layer stays device-resident → the final row
D2H once". `dno_add_rmsnorm` fuses the residual addition and writes back into CH_X in place, so the residual
stream is a single buffer across layers. Added `maybe_gpu_ffn_dev` / `maybe_gpu_oproj_dev` / `chain_download_at`
(partial D2H from an offset within a slot). **No new kernels** (reuses the existing PTX, sha256 unchanged).

**Stumble** — Consistency on mid-way failure. The chain is all-or-nothing: on failure we redo the whole forward pass on the old path (KV writes are idempotent and advance happens only
once at the end, so it is safe). Unsupported structures (Qwen3 QK-Norm, some fp32, qkv bias not arranged as [qb|kb|vb]) and kernel constraint violations are rejected by an up-front gate,
avoiding pointless GPU_BROKEN firings.

**What we learned / remaining issues** — rmsnorm / GEMM / rope / swiglu / residual use the same kernel and the same operation order, so they are **bit-identical**
(only attention falls in P17j's relaxed class). `cuda_prefill_chain_bit_exact_vs_cpu` compares post-rope q, the K/V rows, the final
hn and the final residual bitwise against the CPU reference. **The wall on the real machine is not measured** — after the chain, per-op accounting is absorbed into the final sync and loses meaning,
so it was recorded that measurement has to be done on a wall basis.

**What worked next** — "Do not drop to the host inside a region that completes within the layer" became the direct precedent for s70 on the 9B side (the prefill FFN chain).

### Variant B: parallelizing CPU prefill with rayon (keeping the bit-identical contract) (2026-10-16, `875dd3cd`)

**Intent** — prefill of a large system+tools prompt (n≈2000–2500) was in the tens of seconds, with GPU utilization near 0% and only 1 CPU core
at 100%. The cause was not the GPU side but the sequential CPU implementation (the O(n²) token-by-token full attention and `matmul_bt_i8`).

**What changed** — Made `tensor.rs::matmul_bt_i8` use rayon (the dynamic quantization of the input is row-parallel, the GEMM body is parallel over 32-channel j blocks,
output writes use `SendPtr`, remainder blocks are sequential). In `qwen35.rs` we split `full_attn_core` into `full_attn_prepare`
(KV writes + QK-Norm + partial interleaved RoPE, sequential in token order) and `full_attn_attend_head`
(1 head, read-only), and made prefill's attend parallel over (i, head). rmsnorm, residual addition and SwiGLU are also row-parallelized.

**Stumble** — `matmul_bt_i8` only had a comment saying "parallelized" but was actually sequential `chunks_exact` (a side finding of REPORT_NEXT16).
We narrowed the condition for parallelization to one thing — **do not change the accumulation order of any output element** — and parallelized only along i/j. scores became task-local scratch to avoid sharing.

**What we learned / remaining issues** — 9B int8, n=2480 / max_tokens 4 / seed 42: 16 threads **111.6 s**, 1 thread **568.4 s
** (5.09x), the output the same md5 `f9b7f6be`. The CPU reference md5s (`feea9444` / `da0b1f57`) are unchanged for both 16T and
1T. A prototype per operation stage (outside the repository, same CPU) was 9.2–10.3x for attention, 8.2–8.9x for the matmul equivalent, 5.6x for SwiGLU,
and bit-identical to the sequential version in every configuration. `.cu` / PTX unchanged.

**What worked next** — The criterion "parallelization must not change the accumulation order of output elements" was reused as it was for s62's small-n path and s73's prepare parallelization.

### s48: batching qwen35 prefill's full attention on the GPU (2026-10-17, `a2e5368c`)

**Intent** — qwen35's `forward_prefill` always ran full attention on the CPU with rayon attend, and of the 33.6 s at
n≈2.5k, 13–18 s was estimated to be this O(n²) mass.

**What changed** — Keeping prepare (KV writes + QK-Norm + RoPE, token-sequential) as it was, we replaced attend with 1 launch of P17j's
`dno_attention_prefill`. qwen35 has dh=256 and needs 84.8 KB of dynamic smem (BQ=64 /
BK=8), so we take the opt-in limit with `cuDeviceGetAttribute` and use `cuFuncSetAttribute` to obtain permission to exceed the 48 KB default.
The gate is `n >= gemm_min_m()` (default 256), and `DENNO_GPU_ATTN=0` restores every path to the old behavior. `.cu` / PTX unchanged.

**Stumble** — In environments where the opt-in cannot be taken the limit stays at 48 KB and qwen35's prefill attention silently falls back to the CPU
(llama's dh=128 = 43.8 KB is unaffected). Based on this "silent fallback" we unified the 4 smem comparison gates from a fixed 48 KB to
the post-opt-in limit. The n≥256 gate keeps short text of 1b (m=14) on CPU attend, so the bits are unchanged.

**What we learned / remaining issues** — Numerically only attention is in the relaxed class (maxabs<1e-3). **The wall on the real machine is not measured**, and
the expected effect n≈2.5k 33.6 s → 16–22 s is an estimate. The 2b GPU anchors may change, so re-acquiring them remained as human work. The test is 1 `#[ignore]` test
(opt-in success, gate firing, approximate agreement, determinism).

**What worked next** — This opt-in mechanism became the premise for the attention shapes of the q4 family, where 27B also has dh=256.

### T5a revert: an opt-in gate built on a false hotspot was a 2.85x regression by default (2026-10-19, `083218d6` → revert `bbe82d75`)

**Intent** — T4's profile made it look like "prefill's projections are all on the CPU (30.1 s)", so we added an opt-in gate named
`DENNO_GPU_PREFILL_GEMM`.

**What changed** — `083218d6` replaced prefill's call with `apply_bt_rows_maybe_gpu`, designed to take the old path when the gate is unset.
**The premise was false, so we reverted it in `bbe82d75`** (the crates return byte-identical to T4's `1e51741`). Only the measurement fix
(the `via` column) was kept in a separate commit.

**Stumble** — (1) T4's `via` column was wrong: `time_full` did not pass `on_gpu` and always recorded false, so even when it ran on the
GPU it displayed `cpu`. (2) In fact `quant.rs::apply_bt` (Q8, m>1) calls `maybe_gpu_linear`, and at
`m > max_m() && m >= gemm_min_m()` (=256) it was already inside P17i's GEMM. (3) The replacement called `tensor::matmul_bt_i8`
directly, bypassing `maybe_gpu_linear` = **a genuine CPU fallback**.

**What we learned / remaining issues** — prefill wall on the real machine (int8 9B, 5,6xx tok cache miss, fixed seed, single instance):

| build | env | prefill wall |
|---|---|---|
| `083218d6` | gate unset (default) | 202–203 s (3 runs) / PF_BENCH 200.8 s |
| `083218d6` | `DENNO_GPU_PREFILL_GEMM=1` | 70.7 / 65.3 s / PF_BENCH 71.4 s (79.7 tok/s) |
| `1e51741` (T4) | — | PF_BENCH 70.9 s (80.2 tok/s) |

The default was a **2.85x regression**, and gate on was merely a reproduction of the old behavior. The 3 anchors with the gate on were MATCH = GPU GEMM really is bit-identical.
In s61's follow-up we made the `via` column measured (`time_n_ev` + OR accumulation of `cuda::Probe`), fixed the denominator of `tok/call` to
`CALLS_ROWS`, and added 4 regression tests (295 → 299 passed).

**What worked next** — "Before optimizing based on a profile column, check whether that column is measured or an assertion". This lesson led directly to the review of
s62 (silent single-thread fallback) and s84 (the bucket is the whole block).

### s62 / T5b: the n<64 projection was silently falling into the single-thread path (2026-10-19, `d9fef26c`)

**Intent** — In s61c's separated measurement, prefill's `gdn_proj` (beta/alpha projection) was 96 calls / 25.3 s ≒ 263
ms/call. For a k=4096 × n=32 GEMM this value is beyond the range of tuning. Following T5a's lesson, we pinned down the path before changing anything.

**What changed** — Added `matvec_rows_bt_small_n` to `tensor.rs`. Output channels are parallelized with rayon, and each task walks only its own
weight rows (`JG=2`), while the batch (i) direction is run with `MB=8` SIMD lanes. Each (i,j) keeps a strictly sequential
f32 accumulation over d.

**Stumble** — The reality of the path: `beta_w`/`alpha_w` are fixed to `QuantMode::Fp32` (making them int8 breaks the delta-rule gate), so
they go through the Fp32 branch of `Linear::apply_bt_rows` → `tensor::matvec_rows_bt` and **never reach the GPU seam**.
"The projections are already on the GPU" holds for qkv/gate/out but did not hold for beta/alpha. The root cause is that `matvec_rows_bt_vec` splits output channels with
`par_chunks_exact` in `JB=64` widths, and with n=32 < 64 it returns **0 chunks**, so the parallel body never runs at all.
All channels fell into the single-threaded remainder path (j-outer/i-inner striding over 745 MB), measured **2.8 GFLOPS**. Nothing failed — it was correct but slow, so it stayed
hidden for a long time.

**What we learned / remaining issues** — CPU 9B, same shape (k=4096 n=32 m=2843): `gdn_proj` **263 ms/call → 0.73
ms/call (7.2x)**, PROFILE's `gdn_proj` 1,514.9 ms / 288 calls → 211.2 ms / 288 calls.
The md5s of the 3 anchors match the pre-change build exactly. Tests pin bit identity for n=1..63 (+1 → 301 passed). **The existing tests only used
n≥16 and had never hit this 0-chunk condition** — the direct cause of the miss. Moving it to the GPU was only proposed, as it requires a new fp32 GEMM kernel.

**What worked next** — s66 shows that this small-n path's transposed layout is the next bottleneck.

### s63: prefill's bottleneck was not the GPU but the "result copy" (2026-10-19, `6d34d90e`)

**Intent** — The request was a full profile. Measured, of the 46.2 s prefill of 9B int8 / 5,686 tokens, **GPU computation was only 5.9 s
(12.7%)**, and the largest item was the return-value copy (`to_vec`) of `linear_gemm` at 21.8 s (47%).

**What changed** — Added `matvec_rows_bt_into` / `matvec_bt_into` / `matvec_bt_i8_into` /
`matvec_rows_bt_i8_into` / `matmul_bt_i8_into` to `tensor.rs`, and threaded `_into` through `cuda.rs::linear_gemm_into` /
`maybe_gpu_linear_into` and `quant.rs::Linear::apply_bt_rows_into`.
prefill in `qwen35.rs` allocates its 16 work buffers once, outside the layer loop.

**Stumble** — `full_attn_attend_head` accumulates its output with `*a += pv * b`. The old implementation allocated `vec![0.0]` per layer, so it was implicitly
zero-initialized, but reusing the region leaves the previous layer's values. `qwen38_27b_shape_prefill_batch_bit_exact` failed, and a **buffer-by-buffer
bisect (6 builds)** identified `att`. The fix is a parallel fill of `att`. **Any place that implicitly relies on "a new Vec is zero" will break
when you switch to reuse.**

**What we learned / remaining issues** — We cross-checked 4 measurement systems (PF buckets, a temporary 6-way split patch of `linear_gemm`, 0.5 s sampling, process
CPU time). For 5,686-token warm: result copy 21.8 s (188 ms/call, n=12288, a new
279 MB Vec, effective 1.48 GB/s, page faults 68k/call), GPU actual work 5.9 s (78.5 TFLOP / 13.4 TFLOP/s),
input copy 1.7 s (13.7 GB/s), H2D/quantization/launch/D2H launch 0.018 s (400 calls total). SM utilization 11.6%,
process CPU 88.3 s / wall 45.5 s = 1.94 cores. After the fix **prefill 46.2 → 28.1 s (123 → 202 tok/s)**,
result copy about 2.5 s. decode 300 tokens 10.05 s (33.6 ms/token), **unchanged** (decode does not go through
`_into`).

**What worked next** — In the post-fix breakdown the largest item became `d2h_state`, which led to s67's reuse of the GDN work area and s68's shorter copy path.
The numbers SM utilization 11.6% and 1.94 cores were used repeatedly afterwards as evidence that "the GPU is idling".

### s66: the small-n projection's body was a side effect of the transposed layout (2026-10-19, `ac34b1ea`)

**Intent** — In the s63 / s64 / s65 measurements, `gdn_proj` (the F32 projection of β/α, 96 calls) was 56 ms/call and 0.75 GFLOP = **13 GFLOPS
**. Even though it uses SIMD it performed at single-core level.

**What changed** — Changed the transpose destination of `matvec_rows_bt_small_n_impl` from `at[d * mp + i]` to **group-major
`[g][d][lane]`** (`at[(g*k + d)*MB + l] = a[(g*MB + l)*k + d]`). The transpose itself is also parallelized per group
with rayon (the write regions are disjoint).

**Stumble** — Because 8 lanes are bundled, group g reads `at[g*8 + d*mp]` = **8 floats scattered at intervals of mp (= the whole batch length)
in the d direction**. One group walks the entire `k*mp` (2843×4096×4 ≈ 46 MB) region using only 8 floats per 64 B line,
and that repeats for mp/8 = 356 groups. It was breaking the CPU's cache/prefetch rather than the GPU, equivalent to an effective 0.8 GB/s.
After the fix, one group's working set becomes a contiguous `k*MB` = 128 KB region (it fits in L2).

**What we learned / remaining issues** — For 5,686-token warm, `gdn_proj` **5,405 → 1,083 ms (5.0x)**,
prefill **28.1 → 23.7 s (202 → 240 tok/s)**. −49% from the 46.2 s at session start. The transpose is only data placement, and
the ascending-d sequential accumulation of each (i,j) changes not a single instruction = **bit-identical**. 3 anchors MATCH. The largest remaining item is gate_up 7.38 s.
`matvec_rows_bt_vec_impl` (the n ≥ 64 path) keeps the same `[d][i]` layout, so the same kind of headroom remains.

**What worked next** — "Changing the layout alone gives 5x" became, alongside s68's "changing the copy path alone gives 1.56x", the pattern of the work in this period.

### s68: the D2H through pinned staging was eating 5x the GEMM body (2026-10-19, `b682c3f5`)

**Intent** — After s67, the largest item of prefill's 23.1 s was `gate_up_proj` 7.38 s (64 calls, 114 ms/call).
Reproducing every stage of 1 call in a micro-bench, against GEMM body 12 ms, quantization 6 ms and H2D 7 ms, there was **D2H 20 ms + a host-to-host copy of 60 ms
**.

**What changed** — In `cuda.rs::linear_gemm_into`, issue H2D directly from the caller's `a` slice (removing the `copy_from_slice` into the pinned
`hx`), and issue D2H directly into the caller's `dst` (removing the pinned `ho` + `copy_from_slice`).
Transfer contents, kernel arguments and operation order unchanged.

**Stumble** — The idiom "download into pinned and then `copy_from_slice` into a Rust buffer" is double work in this environment and 4x slower. Measured on 279 MB, average of 3
runs:

| route | ms | effective |
|---|---|---|
| D2H → pinned + `memcpy` → heap (old) | 79.8 | 3.5 GB/s |
| D2H → pageable direct (new) | 20.1 | 13.9 GB/s |
| D2H → pinned only | 19.5 | 14.3 GB/s |

The pinned→heap `memcpy` delivered only 60 ms = **4.7 GB/s**.

**What we learned / remaining issues** — `gate_up_proj` **7.38 → 2.87 s** (114 → 44 ms/call), prefill **23.1 → 14.8 s
(246 → 384 tok/s)**. −68% / 3.1x from the 46.2 s at session start. 3 anchors MATCH, tests 231
passed. The remaining breakdown is `d2h_state`, `attn_core`, `gate_up`, and **the next target is still D2H** — `chain_download`
(attention output 93 MB) and o_proj's `to_vec` were still going through the same pinned route.

**What worked next** — The measured unit costs 13.9 / 14.3 GB/s became the reference values for later judgments of "is the transfer the wall" (s170's round-trip inventory, P41's design memo).

### s69: removing the "via pinned + to_vec" from the chain D2H (2026-10-19, `1f588b4f`)

**Intent** — s68 removed the double work in the GEMM path, but the attention output fetch `chain_download` and o_proj's return value
(`maybe_gpu_oproj`) still did "D2H into pinned → duplicate to the heap with `to_vec`". We were writing the 93 MB output twice every time.
(The primary record had this heading duplicated twice, so this volume merges them into 1 entry.)

**What changed** — `chain_download` / `chain_download_at` in `cuda.rs` allocate the output `Vec` with `with_capacity`,
D2H directly into that uninitialized region, and `set_len` after completion. `maybe_gpu_oproj` also drops the pinned `ho` and
D2Hs directly into the returned `Vec`.

**Stumble** — Only the destination changed; neither the arithmetic nor the issue order was touched. Because completion must be guaranteed before `set_len`, we made it a condition not to move the synchronization.

**What we learned / remaining issues** — prefill **14.80 → 14.50 s (384 → 392 tok/s)**. The difference is 0.30 s,
a different order of magnitude from s68's 1.56x — for attention output 93 MB × 16 layers ≈ 1.5 GB (conversion), the absolute amount of the removed copy is small.
3 anchors MATCH, tests 231 passed.

**What worked next** — Having eliminated the extra host-side copies entirely made it visible that the remaining cost was **the round trip of the activations themselves**, which led directly to s70's FFN residency.

### s70: making prefill's FFN a device-resident chain — activation round trips from 4 to 2 (2026-10-19, `9ec23f3c`)

**Intent** — Once s68 / s69 removed the extra host-side copies, the remaining cost became the activation round trip itself. Per layer (m=2843, e=4096,
ff=6144) it is H2D 47 MB (x) + D2H 140 MB (gate/up) + H2D 47 MB (post-swiglu) + D2H 47 MB
(down) = **281 MB**. The device-resident chain was wired only for decode (m=1), and prefill passed straight through.

**What changed** — `chain_gemm_grouped_pf` (grouped int8 GEMM for prefill; launches `dno_gemm_i8` and
writes into `out_slot`, no sync), `chain_swiglu_rows` (`dno_swiglu_int` for m>1), `chain_mlp_prefill`
(H2D the rmsnormed x → gate/up → swiglu → down → D2H directly into the caller's buffer), and a new CH_PFH slot.
We replaced the FFN part of `qwen35::forward_prefill`, falling back to the old path when not `quant8()` / m<256 / on failure.

**Stumble** — The existing `chain_linear_grouped` assumes decode and, when `m > max_m()` (=16), gives up and launches the
GEMV kernel, so it could not be used as it was. The existing `chain_swiglu` used the gate/up block-separated layout and was m=1 only.
Both are structured so that "reusing the existing resident chain for prefill" is impossible, and a pf version had to be built separately.

**What we learned / remaining issues** — GEMM uses the same kernel and the same integer accumulation, and swiglu uses the same
f32 operation sequence as the CPU's `gv / (1.0 + (-gv).exp()) * uv` (`dno_expf` + `__fdiv_rn` / `__fadd_rn` / `__fmul_rn`), so **bit-identical**
(3 anchors MATCH). `gate_up_proj` 5.74 → 3.52 s, `down_proj` 2.81 → 1.01 s, and the CPU's
1.66 s of swiglu moved to the GPU. prefill **14.50 → 11.58 s (392 → 491 tok/s)**, activation transfer per layer **281 → 94 MB
**. −75% / 4.0x from the 46.2 s at session start. What remains is `attn_core` (GDN scan, gdn_proj, attn_full) and
qkv.

**What worked next** — The decision to "build the pf version of the chain separately" reappears in the same shape in the second half as P14 (folding gate|up into 1 GEMM on q4).

### s71: restoring s69's direct D2H (2026-10-19, `91b54265`)

**Intent** — The patch for s70 was generated from **`cuda.rs` as of s68**, so it rolled back s69's changes (the direct D2H of chain_download / chain_download_at /
maybe_gpu_oproj). In other words s69 was cancelled by the commit right after it, and the 11.58 s measured in s70 was a value
**without s69**.

**What changed** — Reapplied s69's 3 spots on top of s70 (removing the `pin_dn` / `ho` staging).

**Stumble** — Because the patch was generated from a scratch working copy, the previous commit's diff was silently dropped. The build and the tests passed and the numbers were correct, so
what was lost was only performance — a kind of accident you cannot notice without measuring. s69's effect (0.30 s) remained on top of s70 as it was.

**What we learned / remaining issues** — prefill **11.58 → 11.34 s (491 → 502 tok/s)**, 3 anchors
MATCH, tests 231 passed.

**What worked next** — As prevention we made it a rule that "the input of a patch-generating script must always come from the file at the repository HEAD". It came to be treated as the second example, in the same
family as T5a's "the measurement lies", of a regression that stays behind silently.

### s73: parallelizing full attention's prepare over tokens (2026-10-19, `57405fe9`)

**Intent** — `qk_norm` 650 ms + `rope` 491 ms were running on a single CPU thread. One row is about 5 µs, but 5,686 rows × 8 layers exceeded 1
second.

**What changed** — In the full attention path of `qwen35::forward_prefill`, QK-Norm → RoPE →
KV cache row writes are now parallel over rows (tokens). Each row's write destination is disjoint (q rows, KV pos rows), and the operation sequence for one row is the same as the token-sequential path.
prof accounts it all into the `qknorm` bucket (equal to the sum of the parts).

**Stumble** — Even though the operation sequence did not change, **the output of the 2 short anchors changed** (determinism maintained, the full anchor unchanged). The cause was not identified and is
tracked as T9. We confirmed that needle recall on long text (4,671 tokens) is normal. Re-baselined as relaxation class R1 (f32 reduction order)
following the procedure in `docs/NUMERIC_POLICY.*.md`: v6_1b `c61ad85b…` → `47df93d7…`, v6_2b `9e17d563…` →
`9b20de09…`, v6_1b_full `af6eb6ab…` unchanged. The rebaselining was run twice to confirm determinism, and then `--write`.

**What we learned / remaining issues** — prefill **10.11 → 9.65 s (562 → 589 tok/s)**. Independent of thread count (the same md5 with `RAYON_NUM_THREADS=1/8/16`
). However, re-measuring in s75 **did not reproduce the drift and returned to the old baseline values** — the natural interpretation is that a 1 ulp-level perturbation flipped a token at a near-tie in sampling, and
the possibility that parallel prepare reads uninitialized memory remained unverified in T9.

**What worked next** — The first case in which the procedure "if the numbers move, rebaseline and move on" was actually applied. At the same time, the danger of moving the baseline without identifying the cause was also kept as a record.

### s74: widening int8 GEMM's per-thread tile to 8x16 (2026-10-19, `e0ea110b`) — PTX not generated

**Intent** — `dno_gemm_i8` uses 128x128 tiles, 256 threads and 8x8 outputs per thread, so for 1 `__dp4a`
(4 multiply-adds) only 1 B load from smem applies, and the inner loop is dp4a 144 = 16 b64 loads + 4 a32 loads.
The effective **10.6 TOPS** is an order of magnitude below int8's theoretical performance.

**What changed** — Output per thread 8x8 → **8x16**, 1 block = 128 rows × 256 columns with **64 threads**
(block 16x4). The dp4a/B-load ratio doubles from 8 → 16. We chose 256 columns so that accumulators (8x16 = 128) times the thread count
(64) keeps 8192 outputs/block. The 4 elements in the k direction stay as i32 packs and use `__dp4a` directly. Weights are zero-padded to 256 rows when uploaded
(recorded in `GpuW.npad`). The host launch shape was updated in all 11 places.

**Stumble** — Because the accumulators double, other registers that the old version was spilling (a/b packs, address arithmetic, loop unrolling) had to be rolled back.
Padding can be asymmetric between A and B because A rows are shared by 2 B tiles, and zeroing only the weight side means no contribution. Every shape of the real models is a multiple of 256, so
the VRAM growth is zero (extra rows are allocated only for shapes that are not multiples).

**What we learned / remaining issues** — int8 dot products are exact in i32, so the dot is identical whatever the tile shape, thread count or accumulator split.
On the f32 side we keep the order `((float)dot * ascale) * scale[j]`, so bit identity is the expectation (the difference allowed by relaxation class R2 is also not expected to appear).
**`kernels_int8.ptx` was not regenerated** (there is no nvcc in this build environment), and **the GPU path still runs the old PTX (8x8 kernel)**.
The old kernel truncates at `gj < n` for column j, so the numbers are correct: **it does not regress, but the speedup does not fire either**. Performance is **not measured**.

**What worked next** — The practice of explicitly recording the state "new host code and old kernel coexist until the PTX is regenerated" was born. This state became the premise of s75's instant
death (error 716).

### s75: fixing dno_gemm_i8's shared-memory alignment and row coverage (2026-10-19, `5bca9b7e`)

**Intent** — s74's kernel died instantly on the real GPU. In the first int8 GEMM, `cuMemcpyDtoHAsync` returned error 716 → permanent
CPU fallback (PF wall 171.68 s / 33.1 tok/s, 17.8x slower than the baseline 9.65 s / 589 tok/s).

**What changed** — (1) `GEMM_SP` from 33 (`GEMM_TK+1`) to **36** (`GEMM_TK+4`). Because it is a multiple of 4 but not of 32,
the 8-row difference between adjacent threads = 288 B = 72 words, 72 mod 32 = 8, so the banks shift by 8. (2) Returned the block to
`(16,16)` = 256 threads and introduced `GEMM_BX` / `GEMM_BY` to cover the 128x256 tile exactly. (3) 5
`static_assert`s. (4) Unified the host launch shape in 10 places to `(GEMM_BX, GEMM_BY, 1)`.

**Stumble** — There were 2 independent causes. (1) The row stride of the shared stage is not aligned to a 4-byte boundary: the kernel accesses with a 4-byte pointer
cast `*(int*)&sA[tid][c]`, but the stride 33 is odd, so every row becomes unaligned. It becomes a sticky error 716 and every subsequent D2H /
sync fails. compute-sanitizer's diagnosis is `Invalid __shared__ write of size 4 bytes / Access at 0x421 is misaligned`
. (2) The block shape disagreed with the tile dimensions: 64 threads × 8 rows covers only 32 rows, so **rows 32..127 of the 128-row tile are written by nobody**
(about 75% of the output uninitialized). **The CPU tests stayed green**, and this class of defect can only be caught on the GPU or by static inspection.

**What we learned / remaining issues** — Regenerated the PTX with nvcc 12.9 (md5 b5c926ea → fe254249), 3 anchors match.
The second warm run was PF wall 8.71 s / 653 tok/s (−9.7% from s73's 9.65 s), but **this was overstated** — the first run carries the
PTX JIT and weight residency. Re-running the cold A/B, s73 `57405fe9` was 14.222 / 14.221 s (399.8 tok/s) and
s75 `5bca9b7e` was 13.407 / 13.429 s (424.1 / 423.4 tok/s), so **the contribution is −5.7% (0.80 s)**.
From then on, a cold single run is the official value and warm values are used only for relative comparison within the same session. CPU tests 233 passed.

**What worked next** — Recording "corrections of the measurement basis" in HISTORY became established. The static inspection
test that pulls in the `.cu` with `include_str!` and crudely evaluates the `#define`s was made permanent after confirming its effectiveness with mutation tests (it fails with any of `GEMM_SP = GEMM_TK+1` / `GEMM_BY = 4` / `sA[tid][x*3]`)
.

### s80 / s81: concatenating the MLP's gate+up at load time into 1 GEMM (2026-10-19, `b69e0190`)

**Intent** — prefill's MLP projected gate, up and down separately, and on the GPU path gate and up made 2 round trips.
`dno_gemm_i8` is independent per row (output channel) and just walks n in tiles, so concatenating the gate and up weights along the row direction lets 1
GEMM compute both.

**What changed** — `Layer.gate_up_w` (`[2*ff, e]`, row order `[all gate rows | all up rows]`) is generated once at model load with
`concat_rows()` (a new uid is assigned). prefill's int8 path calls the concatenated weight once with `q8_apply_bt_rows_into()`, and
SwiGLU has a branch for the concatenated layout ([all g columns | all u columns]). `chain_mlp_prefill()` receives the single `gate_up_w`
and derives `ff = n/2`. The decode chain's GATE_UP is also unified. The fp32 path keeps 2 calls as before.

**Stumble** — Without assigning a new uid, the concatenated weight collides with the original on the GPU device cache key (the same class as P17b's pointer-key accident).
Also, holding a concatenated copy per layer increases resident memory, so we made it once at load and require no host-side concatenated copy afterwards.

**What we learned / remaining issues** — The basis for bit identity is that "int8 quantization scales are per row, and the integer dot product and accumulation are row-independent", so
concatenating rows changes neither each row's quantized values nor the accumulation order. On the fp32 side the accumulation of `matvec_rows_bt` is closed per (i, j)
and therefore does not depend on row order. 3 tests pin bit identity for both int8 and fp32, at m=1 /
m=3, of prefill logits and GDN state (engine lib 232 passed). The cost is increased resident memory (about 1 GB for int8 /
about 4 GB equivalent for fp32). **We did not run the GPU, and performance is not measured**.

**What worked next** — The same reasoning (row concatenation alone keeps bit identity) opened the way to fusing `qkv + z` and `beta + alpha` on the attention side.
That fusion was implemented in s83 and reverted the same day (`aa973eaf`). On the q4 family, P14 takes the same shape.

### s84: measuring rmsnorm — "0.4 s" was a misnomer for the whole block (2026-10-19, `66698f68`) — not optimized

**Intent** — T8's question was "should rmsnorm be fused with the residual addition". The decision line is 0.25 s.

**What changed** — Added to `prof.rs` an `Ambient` scope (a thread-local parent bucket; inner samples are added both to the parent and to their own bucket) and row-level hooks
`norm_rows` / `norm_rows_end` (reading `Instant` only once per layer), plus the display rows `norm` / `norm_other` /
`block` / `block_rest`. Off by default, it is a true no-op with zero clock reads. In `qwen35.rs` we wrapped each layer's
attention block and FFN block in prefill with the parent scope and fired the event after the norm call.

**Stumble** — Measured (9B, 5,686 tokens, PF wall 13.733 s / 414 tok/s), the `rmsnorm` row was 11,761.8 ms
(43.3%) and its child `norm` was **0.0 ms / calls 0**. (1) The `rmsnorm` row is not the normalization but **the whole layer block**, and the
`attn_core` / `gate_up_proj` / `qkv_proj` / `down_proj` / `o_proj` inside it are double-counted.
(2) The `norm` counter never fired once — prefill's int8 path runs on the device-resident chain, so the normalization does not appear as a host-side function call.
Therefore "0.0 ms" is evidence of "invisible from the host", not of "free". (3) Because of the double counting the table cannot be summed (`TOTAL 37,922.7 ms` >
`[prefill] total 27,182.8 ms` > wall 13.733 s). The earlier "rmsnorm 0.4 s" was also a product of the same parent-scope placement.

**What we learned / remaining issues** — The basis for not optimizing is not a measurement but an upper-bound estimate: normalization is an elementwise op that reads and writes 1 hidden state per layer,
9B's 5,866 tok × 4,096 dims × 16 layers = 1.54 GB, about 3.1 GB of reads and writes. At GPU bandwidth 1 TB/s that is about 3 ms, and even dropping to 100 GB/s it is
about 31 ms, 0.25% or less of the 13.2 s prefill — **an order of magnitude away from the 0.25 s decision line** (all of these are conversions/estimates).
Step 2 (fusion with the residual addition, batching in the token direction) is not carried out. decode is unchanged at 16.37 ms/token, 3 anchors MATCH.

**What worked next** — "Do not wrap a whole block in a parent scope" became the direct motive for s87's rebuild of the accounting. The convention of showing a counter that never fires (calls=0) in the table as it is was also settled
here.

### s87 / s88: making the profile accounting additive and measuring d2h_state per issue site (2026-10-20, `e547749d` / `26c6b4a3`)

**Intent** — s84's `Ambient` was "1 parent + also added to the children", so the table was not additive, and `d2h_state` mixed several issue sites into one bucket so "where it is heavy"
could not be read. The second rebuild of the instrumentation, neither of which touches the numeric path.

**What changed** — s87 (`e547749d`): `AMBIENT` from a single slot to a stack. Unified the time writes into `add_outer_ns`. Fixed the double counting of calls /
toks (the count was doubled because `owners` includes one's own bucket). Collapsed samples are also counted as children,
making `self + children = parent`. Added the **`self` column (7th) = parent − children**, and the phase total and cumulative % are computed from the sum of `self`.
s84's per-row norm hooks and the layer-block `Ambient(RMSNORM)` wrapper were removed. s88 (`26c6b4a3`): added `D2H_SRC_NS[phase][site]`
(**display only** — it is not added into any raw bucket, so no double counting occurs). Added `mark_d2h_src_ns` / `add_d2h_src_ns_only` /
`d2h_src_on()`, and the prefill table now shows the top 10 rows per issue site (ms / calls / us per call).

**Stumble** — Just 2 places writing time doubled every sample (the whole thing was inflated without anyone noticing). Because counts and times were split into different APIs, "who accumulates the count"
became ambiguous, and we actually stepped on the trap of doubling the count by calling both. The tests use a global lock + a `#[cfg(test)]` forced flag (it does not write env, so it does not disturb parallel tests) +
`reset()` at entry, removing order dependence.

**What we learned / remaining issues** — GPU measurement: GPU_VERIFY PASS and 3 anchors MATCH = the instrumentation is numerically a no-op. Because per-site **ranking did not exist before this
commit**, the pre-fix values are only the `d2h_state` bucket totals (prefill: CPU machine 4,832.3 ms / 160 calls,
GPU machine 6,165 ms / 1,140 calls). The first real ranking comes from the measurements of later builds. The verification run confirmed that the new buckets move (for 1 decode:
`d2h_src_other` 88.9 ms / 548 calls, `d2h_src_attn_pf` 9.3 ms / 8 calls, `d2h_src_gdn_state` 0.0 ms
/ 432 calls, `d2h_src_linear` 0.0 ms / 2,386 calls) — this is an observation of 1 decode, not prefill's ranking.

**What worked next** — The convention "display-only tables are not added into the raw buckets" made it possible to add breakdowns without changing the accounting. The second half's P25 (settling ffn_down's breakdown) is read on top of this additive
foundation.

---

### P14: folding the FFN's gate|up into 1 GEMM (2026-10-23, `c4c6f569`)

**Intent** — One FFN tile is 4 launches: gate GEMM + up GEMM + swiglu + down GEMM. gate and up read the same input `pact`, with
no synchronization between them. Making these 2 into 1 reduces launches. 56 launches per layer → 42, and 3,584 → 2,688 over the whole request (64 layers) (−896).

**What changed** — `c4c6f569` / `crates/denno-engine/src/cuda.rs` / `crates/denno-engine/src/model/qwen35.rs`.
`gate_up_w` (= 1 row-concatenated buffer [all gate rows | all up rows], already present at load) is GEMMed once with n=2ff and written into `CH_GA` as [gate|up] within the row.
The following swiglu stays `dno_swiglu_int`. The kernel, accumulation order, grid and tile width are unchanged, so `CH_DOWN` is bit-identical. `DENNO_PF_FFN_FUSED=0`
restores the previous 2 calls.

**Stumble** — Only gate and up can be folded; swiglu is already inside the same chain (no sync between them), so it stops at 4 → 3 launches. So that it does not break silently when the premise fails,
we added `pf_ffn_fused_ready` (all Q4, shapes consistent, gate/up registered as row ranges of the concatenated side) and made it fail closed to the old path when not satisfied. The weights are
`q4_register_row_alias`ed at the head of `prefill_layer_dev` (before the individual gate/up are loaded into VRAM) and
share the same single buffer as the decode chain, so VRAM and H2D do not increase.

**What we learned / remaining issues** — The prediction is `sl_lay_ffn_gate` 10,042.6 → ≈10,040 ms (Δ −2 ms), and `sl_lay_swiglu` 210.4 and
`sl_lay_ffn_down` 11,100.0 are unchanged (all values from the n=14,043 / wall 63.05 s run, not measurements of the current build).
Because the MAC count, total block count and tile width are unchanged and only 896 launches disappear, the ms gain is expected to be almost 0 (3–6 µs per launch).
The real-machine A/B and the md5s of the 8-question anchors are not measured (no GPU). Gates: `--lib` 279 passed, `--features cuda --lib`
320 passed / 37 ignored (+4).

**What worked next** — "Launch count is not the bottleneck" is confirmed twice within the same volume. P21 selects the GEMM shape and gets 0 ms on 27B,
and P51 estimates the issuance of 1,344 launches/req at ≈9 ms. The i8-side equivalent (gate and up in 1 launch) is in s80/s81 of this volume.

### P20: writing the full layer's attention q directly into pinned staging (2026-10-23, `5f222949`)

**Intent** — The full layer's `sl_lay_full_core`
is "host prepare → q H2D → 1 attention launch → att D2H → host sigmoid gate → att H2D", and the q written by prepare was placed once into the host's `buf_q` and then copied into pinned staging (345 MB/layer).
We first judged the 2 branches in the instruction: (a) heads are not looped on the host — attention is already 1 launch/layer, and the kernel with `grid.y = heads = 24`
reads `blockIdx.y` as the head. (b) There is only 1 host synchronization between attention and o_proj, and
it cannot be dropped because it is the D2H completion wait for the host to read 345 MB of att and apply the sigmoid gate (dropping it requires an n-token version of the gate kernel = `.cu`).

**What changed** — `5f222949` / `crates/denno-engine/src/model/qwen35.rs`. Changed prepare's write destination from `buf_q`
to q's pinned staging, and made the H2D source that pinned buffer as it is. The launch table is unchanged: `sl_lay_full_qk`
is 3 GEMMs (grid 220×96 / 220×8 / 220×8, block 16×16) + D2H 805 MB + 1 sync, and `sl_lay_full_core` is
1 `dno_attention_prefill` (grid 220×24, block 256 threads, smem 84,800 B,
BQ=64/BK=8) + att D2H 345 MB → gate → H2D 345 MB, while `sl_lay_full_oproj` is 1 GEMM. The 5 launches per full layer are the same before and after.

**Stumble** — Removing the q copy barely moves the phase. The bottleneck is the 650-fold re-read of K/V by the attention kernel (bandwidth), and that is `.cu`'s job.
The launch contribution is 5 launches × ≈5 µs = 25 µs/layer, which does not reach the "bottleneck" the instruction speaks of. We kept a branch that redoes prepare into `buf_q`
for the path that falls back to CPU attend (KV rows are overwrites of the same values, hence idempotent).

**What we learned / remaining issues** — The prediction (not a measurement; CPU measured unit cost 21.1 ms/layer × 16 layers) is `sl_lay_full_core`
10,695.3 → 10,355–10,540 ms (−0.16 to −0.34 s), and `sl_lay_full_qk` unchanged.
The values are bit-identical — the kernel, grid, block, smem, arguments, H2D byte count and issue order are unchanged, and what changed is only "whether the H2D source is the copy in `buf_q` or
the pinned buffer prepare wrote directly". The row for `5f22294` in `docs/KPI.md`
is 3,161 tokens 9.69 s / 12,850 tokens 38.79–40.35 s (cold, with a nonce every time, median of 3, first run discarded).
The real-machine ms, the byte identity of 2 E2E runs, and the execution of the new path are not measured.

**What worked next** — The line "what can be removed inside a phase is only the host's serial copies; transfers and kernels do not move unless you touch `.cu`" became the
selection criterion for P28 (unread allocations), P44 (the pure V copy) and P46 (serialization of the round trip).

### P21: choosing the q4 prefill GEMM row tile from the projection's n (2026-10-23, no commit)

**Intent** — Because `sl_lay_gdn_proj` was said to be 18.0 s, we first doubted the premise.
The instruction's "18.0 s / 7.6 TFLOP/s" is a measurement from before P13 (the n=14,043 / wall 63.05 s run), and the same KPI is 39.0–42.18 s after P12+P13.
Rebuilding the 18.0 s breakdown from measured unit costs gives host beta/alpha 9.75 s + h2's D2H 4.9 s + GEMM 3.67 s = 18.3 s (1.7% off the measured 18,015.8 ms), and
81% was not GEMM.

**What changed** — Added `q4_pf_tile` / `pf_gdn_tile` to `crates/denno-engine/src/cuda.rs` and choose the row tile from the projection's n. The occupancy lower bound
`(t/64)·⌈n/128⌉ ≥ 4·sm`, the L2 upper bound `A(t·k·4) + B(n·(k/32)·20) ≤ 64 MiB`, and `sm = 0` returns the previous value.
`DENNO_PF_GEMM_SHAPE=0` restores the fixed tile. **This implementation was never committed**, and was discarded as WIP in the next P25's STEP 0 (`git restore`
of 1 file). The related docs commit is `86c44535` (P29 noted 18.0 s as a historical value).

**Stumble** — The instruction's direction (lower the tile for narrow n) is about GEMV with a 1-dimensional grid; GEMM has a 2-dimensional grid, so the sign was reversed.
Since `blocks = (tile/64)·⌈n/128⌉`, lowering the tile also lowers the block count and makes SM occupancy worse. For narrow n, raising the tile is correct.
27B's qkv (n=10,240, 80 column blocks, 1,280 blocks) and z (n=6,144, 48 column blocks, 768 blocks) already have 4.5 waves or more at tile 1,024, so
the selection never fires.

**What we learned / remaining issues** — On 27B before == after, and
28 launches/layer, 1,344 launches/request and 28,160 blocks/layer are all unchanged, with a predicted 0 ms. Only projections with n ≤ 4,096 fire.
We also nailed down why the GEMM does not lose on shape: 1 block = A 64 rows × B 128 columns × all of k, and the arithmetic intensity 48.76 FLOP/B depends on neither n nor the tile.
There is no mechanism by which `gdn_oproj` and `ffn_gate`, with the same kernel and the same block work, run at 30.8 TFLOP/s and 31.9 TFLOP/s while
only qkv/z drop to 11–13 TFLOP/s.

**What worked next** — With the bottleneck settled as host beta/alpha + the h2 copy, the next move narrowed to "SIMD headroom in the host projection" and "moving beta/alpha to the device".
The former was carried out as P51's AVX2 16-lane version.

### P25: settling the breakdown of `sl_lay_ffn_down` (measurement only) (2026-10-23, no commit)

**Intent** — Because `sl_lay_ffn_down` was said to be 11.1 s, we first checked whether the instruction's rule "if host or copy is 20% or more of the phase, add exactly 1 point" applies.
The instructed order was: if it does not apply, add nothing.

**What changed** — Nothing. Checking the interval `[PF_PH_FFN_DOWN event → next event]` row by row, what goes in is only 1 launch of `dno_gemm_q4_0`,
with H2D 0 B, D2H 0 B, 0 `ctx.sync()`, 0 allocations and 0 quantizations. Because it is the residual-resident path (`PfSrc::Slot` / `PfSink::Slot`),
`dst = None`, and the only branch in `chain_mlp_prefill_q4_impl` that calls sync is not reached. The prologue (`weights_for_q4` ×3,
`chain_ensure_buf` ×4) is before the tile loop, so it falls into the previous phase. The condition did not hold, so 0 lines of code and no commit. The related docs commit is `70675d32`
(P30 moved the env placement that was in this procedure to the server startup side).

**Stumble** — The breakdown is 100% GEMM, yet with the same kernel, the same block work and the same arithmetic intensity (24.38 MAC/B), only down is 14.4 TFLOP/s while
gate|up is 31.9 and o_proj is 30.8. Accumulating at the reference rates, down should take 5.02–5.20 s, and the 5.9–6.1 s gap from the
measured bucket of 11,100 ms remains on the device side. There are 3 candidates: (1) wave quantization from having only 640 blocks in 1 launch (3.76 blocks/SM
against 170 SMs, upper bound of the efficiency difference 1.5x = at most 3.5 s); (2) because all blocks synchronize on the same barrier in a 1-wave launch,
the single-buffer k loop cannot hide latency; (3) being the phase at the end of the layer, all host delay lands here. We got as far as "1 nsys run is enough for the isolation".

**What we learned / remaining issues** — 11,100.0 ms and each rate are values from the s170 run (n=14,043, wall 63.05 s), from the host-bound build before P12/P13.
Whether the current build still shows the same 11.1 s is not measured, and we left the prediction "if it has dropped to 5–6 s then down is at the kernel ceiling and there is no lever in this phase". The tile (1,024) is the only
lever that can test wave quantization without `.cu`, but we decided not to touch it until the isolation is done (we only confirmed the property that bit identity is preserved because it changes only how rows are grouped).

**What worked next** — The suspicion "reading a bucket as kernel time lies" became the entry point of the next P27 accounting audit.

### P27: auditing the phase accounting — what `DENNO_PF_DBG` does not measure (2026-10-23, no commit)

**Intent** — P25 could only say "100% GEMM", and what was inside that 100% (kernel time or gaps) was not settled. To avoid misjudging, we audited the accounting mechanism itself.

**What changed** — Nothing (0 lines of code changed, 0 commits). The related docs commit is `86c44535` (P29 fixed this audit's 4 conclusions in the docs).

**Stumble** — `pf_phase()` only places events on the default stream, and `pf_timing_read()` accumulates `cuEventElapsedTime(ev[k], ev[k+1])` into
`phase_of[k]`. Each interval is therefore "the execution of the kernels issued in that interval + GPU idle inside the interval + host delay", and the
11 buckets are a complete split of wall but **not a split of kernel time**. Whether `sl_lay_ffn_down`'s
11.1 s is "a kernel at 12.39 ms/launch" or "kernel 6 ms + gap 6 ms" cannot in principle be decided by this mechanism.
Moreover the attribution is systematically off by 1 phase: phase boundaries are placed at the head of an issue group, and `weights_for_q4` / `chain_ensure_buf` / upload precede it, so they fall into the previous phase (FFN's prologue → `norm_post`, full's qkv prologue → `norm_in`,
o_proj prologue and att upload → `full_core`, the last tile's down interval → the next layer's prologue).

**What we learned / remaining issues** — We could show it with an example: the same `dno_add_rmsnorm` is 29.0 ms/layer in `norm_in` and ≤ 4.0 ms/layer in `norm_post`, a difference of 7x or more.
s170's `norm_in` is not kernel time. In a cold run the H2D of the 13.21 GB of weights falls into the phase at its call position and inflates `norm_post`
by ≈0.5–0.6 s — which is also the reason `prefill_bench` discards the first run.
s170's table is the sum of 9 rows 61,999.6 ms (98.33%) + tail 567.4 ms, with almost no "unexplainable wall", but in a
host-bound build a slow phase absorbs its own delay, so this is not evidence that the accounting is right.
The constraint obtained from the 3,161-token wall goes only as far as "what remains in down is ≈1.0 s (2.6 s if every floor is taken optimistically)".

**What worked next** — The convention "a bucket is a split of wall" was fixed in the docs in P29/P30, and
from then on every number in this volume is read together with "which build, which n, who measured it". P41's `DENNO_FC_KPROF` was designed, in light of
this reflection, as a device that re-splits the interval into 4.

### P28: measuring the host side of `sl_lay_full_core` first (2026-10-23, `3c383a7f`)

**Intent** — Following the instruction's "measure first", we tabulated the phase's launches / transfers / host costs. The phase is device bound (the attention kernel dominates,
1.15 GB/layer of transfers), and
the host side is prepare 44.9 ms + sigmoid gate 45.4 ms = 90.3 ms/layer (n=14,043, 1.45 s/request over 16 full layers).
Both are "necessary data movement", and removing them requires moving the gate to the device (`.cu`).

**What changed** — `3c383a7f` / `crates/denno-engine/src/model/qwen35.rs` only (+24/−3). Within the phase, every request
`fit_prefill_buf(buf_q)` and `fit_prefill_buf(buf_att)` were mmapping + zero-filling 2 × 345 MB, but the
`AttnPlace::GpuDev` path never reads these 2 buffers (allocations >128 KB do not go through malloc's arena, so we were paying page faults every time). We moved the allocations to their points of
use (before prepare when `!q_in_pin`, before CPU attend's zero fill, before GpuHost's `copy_from_slice`).

**Stumble** — Pinning down the live path took effort. With `DENNO_GPU_ATTN` default ON + `attn_prefill_gpu_supported` (judges `n >= gemm_min_m()` = default 256, and dh=256 has opted in for 84,800 B of smem), n=14,043 is GPU attend. The residual-resident path has
`host_att_needed=false`, so `chain_pf_att_inplace` runs as `AttnPlace::GpuDev` = P12's pinned direct-send path as it is.
The zero fill of `buf_att` is only inside `if !att_done`, so P12's removal is preserved — after confirming these 3 points in the code we deleted the "unread allocation".

**What we learned / remaining issues** — Measured (no GPU needed, `rustc -O`, real 27B shape): prepare 44.9 ms/layer (3,198 ns/token),
gate 45.4 ms/layer (3,233 ns/token, of which exp is 12.1 ms and the rest is reading and writing 690 MB = 21 GB/s).
At n=12,850 it is 38.2 + 43.1 = 81.2 ms/layer, and at n=3,161 11.1 + 11.2 = 22.2 ms/layer. The bandwidth calibration is r+w 41 GB/s on the same 345 MB copy.
The predicted effect (measured unit cost × 2 buffers) is −0.24 s/request @14,043, −0.25 s @12,850, −0.06 s @3,161.
The values do not change by a single bit (every path used writes all elements immediately before use). The real-machine cold A/B is not measured (no GPU). Tests are unchanged at `--features cuda --lib`
321 passed / 37 ignored.

**What worked next** — A pattern for hunting "allocations and zero fills paid every time inside a phase" formed, and P44's V copy, P47's reduction of the `buf_qg`/`buf_k` fits,
and P48's per-row Vec followed.

### P29: fixing how to read the phase accounting in the docs (2026-10-23, `86c44535`)

**Intent** — P27's audit result existed only in a work report, and `docs/PREFILL_PATH.md` still read `sl_lay_*`'s ms as "the cost of that phase".
The same misreading would break the next decision, so we fixed the docs first.

**What changed** — `86c44535` / `docs/PREFILL_PATH.md` (+114/−2) / `docs/KPI.md` (+16/−0). At the head of `PREFILL_PATH.md`
we placed a new section "How to read phase accounting (settled in P27)", consolidating 4 points: (a) buckets are a split of wall, not kernel time,
(b) attribution is off by 1 phase (with file:line for the boundary markers of all 11 buckets), (c) the `norm_post` inflation in cold runs is the reason the first run is discarded,
(d) s170's % are stated explicitly as historical values. In `KPI.md` we placed the same 4 points as a summary in §5.1 and referenced them from the existing "discard the first run" rule.

**Stumble** — Reading all of `scripts/prefill_bench.py`, the first-run discard was not implemented (it takes the median over all runs, so with `runs=2` the slower one becomes the median).
We wrote honestly in the document that "discarding is an operational rule; the script itself does not discard". We did not add a single existing number — the only new things are 2 arithmetic results
(`11,100.0 ms ÷ 896 launch = 12.39 ms/launch`, `13.21 GB / 18 GB/s = 0.734 s`).

**What we learned / remaining issues** — s170's % has a denominator more than 6x different (63.05 s vs the current 9.69 s / 38.79–40.35 s), so
we embedded in the body that it cannot be read as a share of the current HEAD. Found but not fixed: the env placement `DENNO_PF_DBG=1 python3 scripts/prefill_bench.py …`
remained in 3 places (env is read once by the server at startup, so it does not take effect on the python side). This was fixed in P30.

**What worked next** — The convention for quoting "a phase's ms" in this volume (bucket name + conditions + whether it is a historical value) was fixed here, and P41's `[FC_KPROF]`
and P50's tail breakdown are read in the same framework.

### P30: moving PF's env to the server startup side (2026-10-23, `70675d32`)

**Intent** — Fix the 3 places P29 left as "found but not fixed". The form `DENNO_*_=1 python3 scripts/prefill_bench.py …` does not reach serve,
because bench is only an HTTP client. The purpose is to eliminate the cause of measurements coming back empty.

**What changed** — `70675d32` / `docs/PREFILL_PATH.md` / `docs/KPI.md`, 2 files only (49 insertions / 10 deletions). We pinned the basis in the implementation:
`DENNO_PF_DBG` (OnceLock, `cuda.rs:11921-11924`), `DENNO_Q4_M_TILE` (`cuda.rs:3504-3510`),
`DENNO_PF_OVERLAP` (`cuda.rs:11702-11705`) and `DENNO_PF_FFN_FUSED` (`cuda.rs:5446-5449`) are all read only once at serve startup. We rewrote the 3 blocks into the form "start the server, wait for health, send the bench, grep the server's log".

**Stumble** — There was 1 more cause of empty results. The `[PF_DBG]` lines go to serve's stderr, so `2>&1 | grep` on the bench output yields not a single line.
We moved the grep target to the server log. Also, ①②'s A/B differ in env, so **the server must be restarted twice**, whereas the original form only sent twice to the same serve.

**What we learned / remaining issues** — Not a single number changed (`65000 2` / `65000 3` / `DENNO_PF_OVERLAP=0` /
`DENNO_PF_FFN_FUSED=0` stay as they were). Incidentally we closed the duplicate `## 4` in `KPI.md` and changed the title of `PREFILL_PATH.md`
from "as of s163" to "s163-s175". We decided not to build an env forwarding mechanism into `scripts/prefill_bench.py` (it is a value read once at startup, so
forwarding it does nothing unless the server is restarted). Behavior on the real machine is not verified (no GPU).

**What worked next** — From then on the A/B procedure was unified as "env goes on the server startup side", and an empty measurement can be explained by "the server was not restarted".

### P31: pinning the prefill chain's static invariants without a GPU (2026-10-23, `b5f97db8`)

**Intent** — Every change in this volume touches "launch arguments, block/grid, buffer sizes, phase marker positions". To avoid breaking them while we cannot run the GPU,
we first thickened the static invariants that read and inspect the source.

**What changed** — `b5f97db8` / `crates/denno-engine/tests/kernel_invariants.rs`
only (2312 → 3251 lines, 939 net additions, 95 assert sites). Added 4: `chain_mlp_prefill_q4` / `chain_gdn_prefill_q4` / residual residency
(`CH_PF_RES`) / PF phase markers. `.cu` / `.ptx` / product code unchanged.

**Stumble** — One premise was wrong. The FFN's 3 phases are not pushed once per layer but **per 1,024-row tile** (as `pf_record`'s doc says).
We rewrote the tests to pin "there is 1 place that pushes", "the FFN is inside the tile loop", "the GDN is outside the loop" and "the phase sequence of a single layer is ascending in id".
The mistakes caught are 4 classes: (a) mismatch between `Q4_M_TILE`
and `Q4_GM`, a regression that allocates an intermediate buffer for the whole prompt (m) (2.7 GiB at 13k rows), a regression that drops the remainder-merged tail from `cap`;
(b) swapping the launch arguments of chain and per-op (nib/sc, dk/dv) and changing only one of block/grid; (c) reading the residual before writing it, a dropped residual addition for `Some(CH_O)`,
passing the wrong slot to the next layer, a premise check placed after the fact (double addition); (d) double-pushing / missing / reordered phase markers.

**What we learned / remaining issues** — `--features cuda --lib` 321 passed / 37 ignored, identical to the baseline, and `--test kernel_invariants`
is 15 passed (11 existing + 4 new). The 4 new ones detected all 11 mutation patterns (temporarily breaking `cuda.rs` / `qwen35.rs` and restoring them).
Not measured: GPU execution, anchors, ms/token (no GPU).

**What worked next** — This test group became the receptacle for P34/P36/P39's scanner extensions and P52's block change, and the pattern "pin the Rust-side launch shape and the kernel-side premise as a pair" was formed.

### P34: turning the smem cast scanner's known limitations into tests (2026-10-23, `a9b91394`)

**Intent** — The smem cast scanner of P3-3 (s104/s105) has 3 recorded limitations, and of those we close the ones that can be decided statically with tests. Without touching `.cu`,
the purpose is to reduce missed alignment/bound violations of 4-byte/8-byte casts of shared memory.

**What changed** — `a9b91394` / `crates/denno-engine/tests/kernel_invariants.rs` only. `.cu` / `.ptx` / product code / `cuda.rs` unchanged.

**Stumble** — Limitation 2 (float2 alignment): the old version judged float2 with `% 4` as a 4-byte type and also ignored the alias's base offset. We gave `SmemObj` a `pre8`
(the base's byte remainder mod 8) and `dyn_base`, and made `eval_mod` (an unknown identifier is 0 only when multiplied by 0) compute the remainder of symbolic offsets like
`PF_BK * (dh + 1)`. Limitation 3 (indirect casts): the old version only looked at `*(T*)`, so it caught not a single instance of the form `int *p = (int*)&sA[r][c];`.
We added `indirect_casts4`, register pointer variables, and run `*(int*)p` through the same alignment/bound judgment.
Limitation 1 (row upper bound): for rows resolved to a number we now check "row < declared row count" and "flat end ≤ total element count", but when the row comes from threadIdx / blockIdx there is no
static upper bound in `.cu`.

**What we learned / remaining issues** — Substituting the hard limit 1024 makes `ty * 8 + u <= 8191` always "exceed" for a 128-row declaration = a false positive, so
we left this as an open item (counterexample: `*(const int*)&sA[ty * 8 + u][c0]`). The 3 real-kernel instances (`corr_sm[rc * 2]` / 2 `s_sm`
) are green as 8-byte anchors (8B / 32B). Self-verification is 1 new test (the 16th), pinning that all 4 broken fixtures (float2 column step `kk += 1`, alias base
`sh + 1`, 2 indirect casts at odd columns, row 9 outside the declaration) are reported while the correct side (`kk += 2` / `sh + 2` / column 8 / row 7) stays green.
The detection counts are unchanged at 9 casts / 9 symbolic-ok / 0 unresolved. `--lib` 321 passed, `kernel_invariants` 16 passed.

**What worked next** — The open item "how to obtain an upper bound when the row is symbolic" became 2 stages: P36 (take it from the host launch block) and P39 (take it from the grid).

### P36: making the scanner take in the host launch block shape (2026-10-23, `9e7de184`)

**Intent** — Close P34's last remaining limitation (no upper bound for symbolic rows) by taking it from the launch block in `cuda.rs`. If `.cu` has no static basis, the launch side's block dimensions are basis enough.

**What changed** — `9e7de184` / `crates/denno-engine/tests/kernel_invariants.rs` only. `launch_map_from_cuda`
blanks out comments and string literals (line counts and line numbers unchanged), builds field→kernel name from the `module_get_function` correspondence, and reduces every `.launch_async(..)` to
`kernel name -> [(cuda.rs line, (block.x, block.y, block.z))]`. Constants are resolved by evaluating `const NAME: u32|usize|i32|i64 = EXPR;`
without `as` casts. The result on the current tree is 61 call sites / 18 kernels / 10 that cannot be resolved to a kernel name (function-pointer variables like `f` / `f_c` / `func`
, and fields not in `module_get_function`'s table).

**Stumble** — Unless the rule is sound, false positives make the scanner unusable. We mark `Resolved` only when all launch sites agree on a completely static block, and `Unresolved` for symbolic dimensions
(`(dh as u32, 1, 1)`), disagreement between sites (`dno_matvec_rows_bt_i8` has 2 sites, 256 and 128) and no site. Row expressions are
evaluated by `symbolic_row_bound` with interval arithmetic (`threadIdx.[xyz]` is 0..block-1, `blockDim.[xyz]`, `#define` constants, proven loop bounds, the
enclosing `if (X < N)` guard). We report a violation only when `max` can be said to be reachable; otherwise we accumulate into `row_bound_unresolved` with a reason and do not go green.

**What we learned / remaining issues** — Incidentally we found 2 defects in the scanner itself (both fixed, existing asserts unchanged): (1) the smem name table was per file, so
`dno_gemm_q4_0`'s `sA[64][32]` overwrote `dno_gemm_i8`'s `sA[128][36]` and the row count looked like 64 → made it a table per kernel body.
(2) Because column anchors were resolved file-wide, another kernel's loop variable turned the `const int c = x * 4;` anchor into 1 → limited the search to the kernel body and
use the last declaration before the cast line for local bindings. In the real kernels, 6 instances in `kernels_int8.cu` now resolve from the launch block:
9 casts / 9 symbolic-ok / 6 row upper bounds from the launch block / 0 undecidable / 0 violations. `kernel_invariants`
is 17 passed (1 new, 4 mutations). *This is different from the GDN 4-lane version (`3e288aa2`); see `50_gdn.md` for that one.

**What worked next** — The idea "take the basis of the bound from outside" extended as it was into P39's grid intake.

### P39: making the scanner take in the host launch grid and judge blockIdx rows (2026-10-23, `255e147f`)

**Intent** — Close what P36 left: "`blockIdx.*` in a row expression never gives an upper bound (the grid argument is not read)".

**What changed** — `255e147f` / `crates/denno-engine/tests/kernel_invariants.rs` only. `launch_map_from_cuda` reads both the grid (2nd argument) and the block (3rd argument) from the same
`.launch_async(f, grid, block, shmem, args)` (`LaunchSite { line, grid, block }`). No new regular-expression machinery was added. The result is unchanged: 61 call sites / 18 kernels / 10 unresolved.

**Stumble** — For the real kernels, 18/18 grids could not be resolved statically (`(gmx, gmy, 1)` / `(m as u32, 1, 1)` / `grid` /
`(n.div_ceil(ATTN_PF_BQ) as u32, heads as u32, 1)` / `(nv as u32, 1, 1)` — all variables, kernel arguments or function calls).
Therefore 0 rows are decided from the grid in today's real kernels, and
**the moment a row that uses blockIdx appears it becomes unresolved with the reason "the grid cannot be resolved statically"** (no guessing and no fallback to the hard limit).
Requiring the grid uniformly would turn today's 6 green cases red, so we added a dependency-driven gatekeeper: only row expressions that use blockIdx require the grid (dependencies through a local binding
like `const int r = blockIdx.y * BD + ...` are also detected with an in-progress marker).

**What we learned / remaining issues** — The rules are `max(blockIdx.d) = grid[d] - 1` and `max(threadIdx.d) = block[d] - 1`. For mixed terms
(`blockIdx.y * BD + threadIdx.y`) we take `hi = Σhi` as the sum of independent terms (blockIdx is the block number and
threadIdx is the number within the block — different dimensions, so the maxima can be attained simultaneously → we can still claim a violation while keeping `exact`). For products we do not make this assumption: for variables we take the endpoint product as an overestimate and
set `exact = false`, falling to unresolved. Self-verification is 1 new test (the 18th) with 9 fixtures and 5 mutations (blockIdx always None, falling unresolved grids to the hard limit 1024, allowing grid disagreement, making the bound
`grid[d]`, making the product `exact` — the test fails for all of them). The real-kernel counts are unchanged at 9 casts / 9 symbolic-ok / 6 / 0 undecidable / 0 violations,
and `kernel_invariants` is 18 passed.

**What worked next** — Unknown from the records (after this, P41 onwards moved to the measurement side and the scanner was not extended further).

### P41: isolating the breakdown of the full_core phase with host-side events (2026-10-23, `1068bd0d`)

**Intent** — `sl_lay_full_core`'s bucket bundles "the kernels issued inside the interval + host stalls" into one, so it is impossible to tell whether the
attention body (device) or the host prepare / gate (host) is heavier. Based on P27's conclusion, we add a device that splits the inside of the interval.

**What changed** — `1068bd0d` / `crates/denno-engine/src/cuda.rs` only (+276/−1). Only when `DENNO_FC_KPROF=1`, place events at the
**existing launch / sync positions** and split one full layer into the 4 intervals prep / attn / d2h / tail, printing one line at the end of
prefill (OFF by default; unset is 1 branch and no events are created either). Because it uses a separate event pool and a separate env, the output of `DENNO_PROFILE` / `DENNO_PF_DBG`
is unchanged.

**Stumble** — `dno_attention_prefill` fuses QK scores + online softmax + PV into 1 launch, so it cannot physically be split into
qk and av. Following the instruction's "if it cannot be taken physically you may fall back to 2–3 intervals", we set up attn/d2h instead of qk/av.
The interval definitions are prep = phase FULL_CORE → just before the attn launch (host prepare's QK-Norm/RoPE/KV writes + KV mirror H2D + q's pinned H2D),
attn = just before the launch → just after the launch, d2h = just after the launch → completion of `chain_pf_att_inplace`'s `cuMemcpyDtoHAsync`,
tail = D2H completion → phase FULL_OPROJ. Calls where the 5 points are not all present (e.g. a layer that fell back to CPU attend) are not read as an interval.

**What we learned / remaining issues** — The output is
`[FC_KPROF] prep=<ms> attn=<ms> d2h=<ms> tail=<ms> calls=<n> total=<ms> (<ms>/token)`.
At the same time we wrote the design for making full_qk's D2H pinned in `docs/FULL_QK_D2H_PINNED.md` (no implementation).
The design memo's conclusion: this tree's HEAD is P17b's single-shot contiguous pinned D2H, exactly the shape that measured 9.78 → 22.14 s (+772 ms/layer) on the real machine and was reverted.
So the implementation is "return to 3 launches + pinned destination + allocate the pool once outside the layer loop", without changing the copy count, issue order or GEMM.
The reduction is −26 ms/request at s68's measured unit costs (pageable 13.9 / pinned 14.3 GB/s), or
−1.08 s/request back-calculated from the PF_DBG bucket (6.5 GB/s), and **the two are incompatible** → we will settle it with 2 measurement points after implementing (unverified). Tests are
`--features cuda --lib` 321 passed / 37 ignored, `kernel_invariants` 18 passed. The actual `[FC_KPROF]`
output and GPU execution are not measured (no GPU).

**What worked next** — These 4 intervals became the common yardstick of P44/P46/P48/P50.
P44 mapped the 4 intervals onto the code and settled that "the contents of prep and tail are 1 host loop each + 1–2 H2Ds each", and
P50 measured tail at 634 ms in the same intervals and showed that the gate is the main cause.

### P44: D2H-ing the full layer's V directly into the host KV cache rows instead of back into scratch (2026-10-23, `c7b0991e`)

**Intent** — Mapping P41's 4 intervals onto the code, the contents of `prep` and `tail` were only 1 rayon host loop each and 1–2 PCIe H2Ds each, with
almost no allocation / zero fill / extra copy left (already removed in P12 / P17k / P20 / P28).
Of those, exactly 1 remained that could be removed completely without changing the numbers: the pure copy of V rows that prepare does every layer.

**What changed** — `c7b0991e` / `crates/denno-engine/src/model/qwen35.rs` only (+210/−45, of which 79 lines are tests).
Split `pf_prepare_row` into `pf_prepare_qk_row` (the QK-Norm + RoPE body for q/k) + the v copy, and made the v argument of `pf_prepare_into`
an `Option` (`None` = V is already in the cache row). In the full branch of `prefill_layer_dev` we stop calling `fit_prefill_buf(buf_v)` and
pass the cache rows `[pos0..pos0+n)` of `cache.kv_bufs(li, n)` to `chain_qkv_prefill_q4_dev`'s `out_v`.
Because V is not transformed by a single bit in prepare, making the D2H destination the cache row itself removes the `n·kv_dim·4 B` read + write entirely.

**Stumble** — The measured n disagreed with the instruction's 65k. Back-calculating from the interval numbers, n≈18k–22k is the most consistent (from both attn 130.8 ms/layer and d2h 24.6 ms/layer).
The structure is the same under either reading (prep/tail are "1 host loop + 1.5 H2Ds"), so we gave estimates with n=20,000 as primary and n=65,000 as secondary. We left a request to include `n=` on the `[FC_KPROF]` line in the next measurement.

**What we learned / remaining issues** — Measured (CPU only, 16 threads, minimum of 4 rounds, real 27B shape): the prepare loop 42.0 → 37.5 ms/layer (n=14,043, −10.7%),
197.5 → 172.9 ms/layer (n=65,000, −12.5%). The bytes removed are n·kv_dim·4 × 2 = 508 MiB/layer at 65k and 8.1 GiB/request over 16 layers,
effective 21.6 GB/s (in an environment where an equally sized copy gives r+w 41 GB/s). The estimated reduction is 0.02–0.09 s/request (a guess).
The values do not change by a single bit — what was removed is 1 pure copy, the byte sequence D2H writes is identical and only the destination changes. q/k's operation sequence is the same code, and the
launch function, arguments, grid/block/shmem, issue order and sync positions and counts are also unchanged. Tests 292 → 293 (+1:
`pf_prepare_v_in_cache_matches_copy_path_bit_exact`).
However 46% cannot be solved this way: the remaining dominant terms are the host RoPE/QK-Norm, the host sigmoid gate, q's H2D and att's D2H/H2D, none of which can be removed without touching `.cu`.
GPU execution, anchor md5 and cold A/B are not measured.

**What worked next** — Learning that "everything left is `.cu`'s job" made it possible to move the next target from "byte count" to "serialization" (P46's chunking).

### P46: chunking the full layer's att round trip and overlapping D2H/H2D with the gate (2026-10-24, `2db6448d`)

**Intent** — tail (1,013.8 ms total over 16 layers = 63.4 ms/layer) has only 4 steps inside, with no copies, allocations or per-row Vec churn.
What remains is (a) 1 host sigmoid gate, (b) 1 att H2D, (c) o_proj's prologue (µs). Previously it was
`D2H everything → ctx.sync() → gate everything → H2D everything`, during which the device worked only 49.2 ms and the host waited for the whole D2H.

**What changed** — `2db6448d` / `crates/denno-engine/src/cuda.rs` (+330/−27) /
`crates/denno-engine/src/model/qwen35.rs` (+15/−3). Split att into chunks of row blocks and proceed `D2H(k) complete → gate(k) → H2D(k)`.
Controlled by `DENNO_PF_ATT_PIPE` (unset = enabled, limit 8; `0` = the previous single round trip; `N` = limit N).
`K = clamp(ceil(bytes/128 MiB), 2, 8)`, and under 128 MiB / 1 row / flag OFF gives K=1 = the old path itself. Changed the signature of `gate` from
`FnOnce(&mut [f32])` → `FnMut(row offset, row block)`. Events are grow-only, reused across layers, `CU_EVENT_DISABLE_TIMING`.

**Stumble** — The gate is 3 passes (att read + read of qg's gate half + att write), measured on CPU at 24.2 GB/s = 81% of pure read, and removing `exp`
takes 76.1 → 53.3 ms, so it is 70% memory bound and 30% exp. Reducing the pass count requires changing the expression, and turning `/` into a reciprocal multiply or approximating `exp`
breaks bit identity. So the judgment was that only the serialization can be removed. Only when H2D fails midway does `CH_ATT` become partially updated, so we finish gating the remaining chunks and then restore everything with 1 synchronous H2D
(so the caller's fallback does not gate twice). When `event_sync` is unavailable we substitute `ctx.sync()`, and on drivers without
`event_create` we fall to K=1 (fail closed).

**What we learned / remaining issues** — D2H and H2D have the same byte count and the same pinned buffer, so `H ≈ D = 24.6 ms/layer`, and from `tail = G + H` the gate is
`G ≈ 63.4 − 24.6 = 38.8 ms/layer`.
The second run (d2h 454.3 / tail 1,009.7) gives 34.7 / 28.4 = 55% / 45%, the same ratio.
The estimated reduction is 38.8 ms/layer → 0.62 s/request (65k, 1.7% of the 35.8 s wall), `tail` 63.4 → 24.6 ms/layer (−61%), `d2h+tail`
88.0 → 49.2 ms/layer (−44%). **All estimates** (we cannot run the GPU).
Basis for bit identity: the sum of (pinned offset, device offset, byte count) over all chunks covers the same region in the same direction as the old single shot, each element is applied exactly once with the same expression, and the rows are disjoint.
3 tests (`att_pipe_chunk_bounds_cover_rows_once` with 10 values of m × 7 limits,
`att_pipe_gate_is_bit_exact_across_chunk_counts` with n ∈ {1,2,3,5,8,13} × all chunk counts).
`--features cuda --lib` 341 passed (+3).

**What worked next** — The structure "the gate is longer than D2H so it cannot be fully hidden" was confirmed by measurement in P50, and the floor of tail was settled as the gate, not the transfer.

### P47: moving full_qk's D2H destination to pinned staging (2026-10-24, `b44ba551`)

**Intent** — Move P41's design memo into an implementation. `chain_qkv_prefill_q4_dev`'s D2H destination for q/k was the caller's pageable `Vec`, so move it to a thread-local pinned pool.

**What changed** — `b44ba551` / `crates/denno-engine/src/cuda.rs` /
`crates/denno-engine/tests/kernel_invariants.rs`. Added `pin_pf_qkv` (grow-only, 1 buffer split into 2 slices for q/k) and made the return value
a `PfQkView` so the consumers (prepare / attend in `full_attn_host_side`) read those slices directly.
Zero copies into a Vec — pinned→heap is measured at 4.7 GB/s in this measurement environment, so copying would add a new 2.7 s cost over 12.88 GB/request.
v's destination stays as in P44 (host KV cache rows).

**Stumble** — There were 4 readings of the reduction, and they contradicted each other. s68's measured unit costs (pageable 13.9 / pinned 14.3 GB/s) give −26 ms,
back-calculation from the PF_DBG bucket (6.5 GB/s) gives −1.08 s, the instruction's "~6 GB/s upper bound" gives −1.25 s, and "about 1.0 s @65k" gives −0.10 s.
For the same 12.88 GB they span 927 ms to 2,147 ms, more than 2x apart. So we took "a shape that loses under any of them" — keep 3 copies, the issue order and the GEMM, and change only the destination —
and decided to settle the reduction with 2 measurement points on the real machine. Unlike P17b (merging 3 shots into 1 and measuring +772 ms/layer, reverted), we keep 3 launches and do not concatenate.

**What we learned / remaining issues** — Basis for unchanged numbers: not a single device-side instruction changed (the function, arguments and grid/block of the 3 GEMMs, `CH_QKV`'s layout and `q_off`/
`k_off`, the destination offsets and issue order of the 3 D2Hs, 1 `ctx.sync()`). What changed is only the D2H destination address, and the byte sequence the consumer reads is identical to before.
Fallback is fail-closed: over the limit (default 1 GiB, `DENNO_PF_QKV_PIN_MAX`) or on `cuMemHostAlloc`
failure, we allocate not a byte of the pool and fall back to the old path of D2H-ing directly into the caller's `Vec` (no `GPU_BROKEN`; the reason is written to stderr once).
`DENNO_PF_QKV_PIN_DBG=1` confirms in 1 line whether the pinned path was used. As a side effect, on layers where pinned is effective the `buf_qg`/`buf_k`
fits (747.8 MB of mmap + zero fill) disappear. Tests: `--lib` 292 passed, `--features cuda` 342 passed,
`kernel_invariants` 22 passed. GPU code not executed.

**What worked next** — Unknown from the records (P48 immediately after goes back to the host side of prep, and the real-machine A/B of this pinned change is still outstanding).

### P48: removing the per-row intermediate Vec from the full layer's prepare (2026-10-24, `6f5f4a3c`)

**Intent** — P41's `prep` is 68 ms/layer. We first split it into "host prepare 36.0 ms (53%) + transfers 32 ms (47%)" and
looked for work on the host side that could be discarded. The transfers are q's pinned H2D (345 MB) and the KV mirror H2D (k+v 115 MB), and by the interval definition both are inside prep.

**What changed** — `6f5f4a3c` / 1 file, `crates/denno-engine/src/model/qwen35.rs`. One row of `pf_prepare_qk_row` goes from
"go through the `Vec` returned by `rmsnorm` and `copy_from_slice`" to "write directly into the destination (`q_r` / `k_row`)" (`rmsnorm_into`, a 10-line helper).
Per row, 29 allocations (q 24 + k 4 + `k_i.to_vec()` 1), 28 KiB of writes + 28 KiB of reads, and 2 copies of k disappear.
At n=14,043 that is 407 thousand allocations/layer and 6.51 million/request over 16 layers going to 0.

**Stumble** — 86% of the host prepare 36.0 ms/layer is q's single QK-Norm, and
its 25.1 ms is about the same as "the time to copy the same byte count (qg's q half 345 MB read + q 345 MB write) with no arithmetic" = it is stuck at the bandwidth floor.
The arithmetic's share is only 5.8 ms. Therefore the measured effect of "reusing the per-row Vec" is −0.5 to −1.6 ms/layer in the norm step (all 5 runs the same sign, typically −0.7), and
0 ± 0.5 ms/layer for prepare as a whole (whole comparisons have ±1 ms resolution due to ordering effects). This is consistent with the record that P12 tried the same change and did not take it because "it only moves −1.6 ms/layer".
The idea of summing the squares of 4 heads together (−0.9 ms) and the NT store idea (−2.8 ms, but a plain copy with the same NT store becomes slower, plus 32 B alignment and `sfence`
hazards) were not taken.

**What we learned / remaining issues** — The estimated reduction is −0.7 ms/layer → −0.011 s/request over 16 layers (estimated), `prep` 68.0 → 67.3 ms/layer. Small, but
allocations in the hot path go to 0 and the intermediate buffer's traffic also goes to 0. Basis for bit identity: the sum of squares uses the same expression and the same order (sequential f32 accumulation, ascending index, no reassociation), `inv` is the same expression,
and the output is the same 2 left-associated multiplications. The test `p48_prepare_norm_into_bit_exact_vs_legacy` confirms bit identity against the old implementation as a reference (real model shape,
`pos0 = 7`, with and without the RoPE table). The remaining conclusion is the line "a task that seriously cuts `prep` is a task that can touch `.cu`; improving how the host side is written is worth the remaining 1–4%". The real machine's
`[FC_KPROF]`, anchors and cold A/B are not carried out.

**What worked next** — The same way of measuring "the bandwidth floor" (compare against a copy with all arithmetic removed) was reused for P50's evaluation of the gate's 3 passes.

### P50: breaking down tail and showing that att's H2D is already pinned (2026-10-24, `e841db48`)

**Intent** — After P46 the tail is 634 ms / 16 layers = 39.6 ms/layer (27B q4, n=14,043, 1 request).
The instruction was "if tail is dominated by a pageable att H2D, apply the P47 pattern (pinned staging + fail-closed)", so we first settle what is inside.

**What changed** — `e841db48` is docs only (adds `docs/P50_FULL_CORE_TAIL.md`).
**Not a single line of code changed** — there was nothing to replace.

**Stumble** — tail contains only 3 things, with 0 allocations and 1 conversion (the gate).
(1) The sigmoid output gate (host) is 3 passes = att read 345.1 + read of qg's gate half-column 345.1 + att write 345.1 = 1,035.4 MB/layer,
and 39.6 ms is equivalent to 26.1 GB/s. Even in a CPU-only measurement (16 threads) the same 3 passes are 40.1–40.9 ms/layer at 25.3–25.8 GB/s,
642–654 ms over 16 layers = 101–103% of the measured tail. (2) att's H2D is 345.1 MB/layer, and since the
D2H of the same 345.1 MB is 24.6 ms/layer we can back-calculate pinned 14.0 GB/s, so
H2D can also be at most 24.6 ms/layer = 394 ms = only 62% of tail. Reading the whole tail as 1 transfer gives 8.7 GB/s,
which is 1.6x slower than the measured pageable 13.9 GB/s — a single transfer cannot explain it.

**What we learned / remaining issues** — att's H2D **already comes from pinned memory**. `chain_pf_att_inplace` makes the D2H destination and the H2D source the same
`pins[li*3+2]` (345.1 MB, `cuMemHostAlloc` PORTABLE, grow-only per layer), and the fallback `chain_pf_upload_att`
also copies into pinned before H2D. A pageable att H2D does not exist in the current code, so making it pinned removes 0 bytes and the estimated reduction is 0 ms (estimated).
P46's "tail → 394 ms" is the lower bound where "the gate is completely hidden by D2H";
in reality the gate (35–40 ms/layer) is longer than D2H (24.6 ms/layer) so it is not fully hidden — before = gate 35–39 + H2D 24.6 = 60–63 (measured 63.4),
after = gate 35 + the last chunk 4.1 = 39.1 (measured 39.6), and both close. The instruction's "att is about 230 MB/layer" is wrong; with `heads*dh = 6,144`
in f32 it is 345.1 MB/layer (230 MB is for 4,096 columns). If we cut further it would be moving the gate to the device (`.cu` + PTX regeneration) for ≈80 ms/layer = 1.3 s/request (estimated), or stopping the per-layer pinned att (5.52 GB → 345.1 MB,
wall unchanged).

**What worked next** — Unknown from the records (after this, P51 moves to the GDN side and P52 to the kernel side).

### P51: AVX2 16 lanes for GDN's host beta/alpha projection (2026-10-24, `ec7d4f31`)

**Intent** — `sl_lay_gdn_proj` is 4,657 ms/request.
Dividing the whole bucket by the GEMM's FLOPs gives 24.3 TFLOP/s, which was read as "the GEMM is slow", but 21% of the bucket is not GEMM, so the denominator was simply too large.
The GEMM is 3.67–3.73 s (consistent with the same run's `gdn_oproj` measurement of 30.3 TFLOP/s), and
the bottleneck is the host terms — h2's D2H 0.767 s + host beta/alpha 3.567 s = 4.33 s.

**What changed** — `ec7d4f31` / `crates/denno-engine/src/tensor.rs`. AVX2 16-lane version of the inner loop of
`matvec_rows_bt_pair_into` (the host projection that computes GDN's beta/alpha in 1 pass over h2). `MB=16`, and all blocks with `cols == 4 && nr == 32`
go to `pair_group_avx2`. One group of 16 rows is held in 2 `__m256`, with 8 accumulators for 4 channels in registers. Multiply and add are separate instructions
(`_mm256_mul_ps` + `_mm256_add_ps`, not fused into FMA). The row count (32 rows/block), the transpose tile, the block count and the weight re-read count are unchanged, and
remainder blocks and remainder channels fall back to the old loop. aarch64 stays on the old path.

**Stumble** — Previously 1 block of 32 rows was run as `MB=8` lanes × `GB=4` groups, and the indices of the accumulators `acc[g][t][l]` were runtime values (`ng` / `cols`), so every MAC involved a stack load/store and only 186 GFLOP/s came out.
Measured (real 27B shape m=14,043 / k=5,120 / n=48+48, 16 threads, best of 5,
HEAD and the changed version measured back to back on the same machine): 74.32 → 30.14 ms/layer (186 → 458 GFLOP/s, 2.47x), 3.567 → 1.447 s over 48 layers.
The breakdown is the transpose tile construction ≈14.6 ms/layer (the lower bound for reading h2's 287 MB is ≈12 ms) + 1 pass over the channel groups ≈0.63 ms × 24 passes.

**What we learned / remaining issues** — After the change copy + host = 2.21 s < GEMM 3.73 s, so the bucket is GEMM-bound again.
Estimated 4.66 → ≈4.05 s (−0.6 s/request, −1.6% of the 38.05 s wall), floor GEMM 3.73 s. Basis for unchanged numbers: each output element starts from `acc = 0` and stays ascending in
`d`, the AVX2 version lays lanes only along `l` (rows), and multiply and add are separate instructions so there are 2 roundings, identical to the old scalar computation. What changes is only the order in which (i,j) are processed. Tests are `--lib`
293 passed, `--release --features cuda --lib` 343 passed, `kernel_invariants` 22 passed. A real-machine `[PF_DBG]` 1 run (`sl_lay_gdn_proj` dropping to ≈4.0 s and `sl_lay_gdn_scan` not moving) is not measured on its own.
In the integrated build (P47/P48/P50/P51/P52), `GPU_VERIFY.sh` is 3/3 anchors MATCH, the md5s of wide 8 questions × 150 tokens are also identical to before the change, and the
65k wall is 34.95 s (38.05 s before integration). However P51's own contribution is not isolated (P52 accounts for −1.8 to 2.2 s, the rest is the sum of P47/P48/P50).

**What worked next** — Unknown from the records (see `50_gdn.md` for GDN's mathematical side). With the remaining floor now GEMM 3.73 s, the order was organized as: what works next is either the tail-wave A/B of `DENNO_Q4_M_TILE=2048`
or moving beta/alpha to the device.

### P52: widening the q4 GEMM's per-thread tile to 8x8 (2026-10-24, `b6e3fa27`)

**Intent** — prefill's GEMM is dominated by `dno_gemm_q4_0` in almost every phase. Keeping the shape 1 block = A 64 rows × B 128 columns × all of k with
25 KB of smem, we widen only the tile each thread holds to reduce shared-memory loads.

**What changed** — `b6e3fa27` / `crates/denno-engine/kernels/kernels_int8.cu` /
`crates/denno-engine/src/cuda.rs` / `crates/denno-engine/tests/kernel_invariants.rs`.
Block from (16,16) = 256 threads to (16,8) = 128 threads,
and per-thread work from 4 rows × 8 columns (32 acc) to 8 rows × 8 columns (64 acc).
Shared-memory loads per element drop from 32 FFMA / 12 loads to 64 FFMA / 16 loads (2.7 → 4.0). For the B tile expansion, `tid`
becomes the column as it is (0..127), with the low 4 bits of a word being element `wi*4+e` and the high 4 bits element `16+wi*4+e`, the same correspondence as the old 2-way split. On the Rust side,
the block in `q4_dispatch_tile` and `chain_gemm_q4_pf` becomes `(16, 8, 1)`. The PTX was regenerated in the following `4d9c28c7`
(nvcc 12.9 / compute_75 / `-fmad=false -O3`, PTX diff +5,535 lines / −2,863 lines).

**Stumble** — The block change only works as a pair between Rust and the kernel, so fixing only one side either covers rows/columns twice or leaves rows uncovered. `kernel_invariants` was updated to pin, as a pair, "the block is
`(16,8,1)`" and "the kernel assumes `ty * 16 + tx`, `const int col = tid;` and `acc[8][8]`".
The grid expression (`m.div_ceil(64)`, `n.div_ceil(128)`) and `Q4_GM=64` / `Q4_GN=128` / smem are unchanged.

**What we learned / remaining issues** — The accumulation order is unchanged (ascending `b` → ascending `c`, each output element written exactly once, rows disjoint), so bit identity is preserved — the order of
`acc[r][j] += av[r] * bv[j]` for each output element is the same in the 4-row and 8-row versions, and only the range of `r` grew. Registers grow to 64 acc + `av[8]` + `bv[8]`,
so the effect on occupancy also needs confirmation on the real machine. Tests are green for `--lib` / `--features cuda` / `kernel_invariants`.

**Measured (27B Q4_0, 65k class, cold, run 1 discarded because it includes the weight upload, median of 3 runs)**:

| arm | 65k wall | note |
|---|---|---|
| before (canonical) | 38.44 s | the deployed binary |
| before (rebuilt) | 38.17 s | rebuilt from the same source. Build-to-build variation ±0.3 s |
| **P52** | **36.26 s** | minimum 36.09 s |
| before (rebuilt, 2nd) | 37.93 s | the same arm re-measured alternately |

The difference across the 4 arms measured alternately is **−1.8 to 2.2 s (about −5%)**. The output md5 of wide 8 questions × 150 tokens (temp 0) is
identical between before the change and P52 (`e36644365c6746b5ee1a06720d5ad4a1`), and `GPU_VERIFY.sh`'s 3 anchors are also 3/3 MATCH,
so the numeric path has not moved.

**What worked next** — The last entry of this volume. What remains is the 2.55 s mass of `full_oproj` 15.9 / `full_qk` 15.7 TFLOP/s
(the other phases are around 30), with −1.3 s of headroom to the ceiling. We are at the stage of suspecting something other than the tile (transpose, bandwidth).

---

## Rejected approaches

Judgments that looked reasonable at the time but were discarded later. We place only references to the corresponding
entries in the body.

### Leaving prefill on the CPU (the line drawn in P17b `197a06c` / P17i `f2c18315`)
- What we tried: decide the range of int8 linear layers to put on the GPU. GEMM for `m > max_m()` and `m >= 256`, GEMV for `m <= 16`, and 17–255 in between on the CPU.
- Measured value: cold prefill of 1.5B int8 at t≈2810 was about 100 s, and 83% of prefill was the CPU GEMM of the linear layers (gate_up 47 s, down 29 s).
- Why we dropped it: at small m the per-layer synchronization costs more than the CPU GEMM, so 17–255 stayed on the CPU (we did not take the option of pushing it to the GPU).
- Record: body P17b / P17i.

### sampler kernel (dropped in P17j, `cededce6`)
- What we tried: turning the sampler into a kernel, which was the procedure document's next candidate.
- Measured value: measured 0.001%. Of the TOTAL 395.8 s of 13,252-token cold, attn_core was 334 s (84.5%) and would pay off first.
- Why we dropped it: the ratio is equal to 0, and attention should be loaded first.
- Record: body P17j.

### T5a's opt-in gate (2.85x regression by default, `083218d6` → revert `bbe82d75`)
- What we tried: add `DENNO_GPU_PREFILL_GEMM` based on the profile "prefill's projections are all on the CPU (30.1 s)".
- Measured value: against the default 202–203 s (3 runs) / PF_BENCH 200.8 s, gate on was 70.7 / 65.3 s / PF_BENCH 71.4 s. T4's `1e51741` was PF_BENCH 70.9 s.
- Why we dropped it: the premise was false. The `via` column simply did not pass `on_gpu` and always recorded `cpu`, and it was already inside the GEMM path. The replacement bypassed `maybe_gpu_linear` and became a genuine CPU fallback.
- Record: body T5a revert. The crates were returned to byte-identical with `1e51741`, and only the measurement fix (the `via` column) and 4 regression tests were kept.

### How int8 GEMM's thread count and smem stride were chosen (s74 `e0ea110b` → s75 `5bca9b7e`)
- What we tried: widen the per-thread tile from 8x8 → 8x16 and make 1 block 64 threads (16x4).
- Measured value: instant death on the real machine (`cuMemcpyDtoHAsync` error 716) → permanent CPU fallback with PF wall 171.68 s / 33.1 tok/s (17.8x slower than the baseline 9.65 s / 589 tok/s). After the fix, cold went 14.222 → 13.407 s (−5.7%).
- Why we dropped it: 64 threads × 8 rows covers only 32 rows, so rows 32..127 of the 128-row tile are written by nobody (about 75% of the output uninitialized). In addition the smem stride 33 is not aligned to a 4-byte boundary. We kept the decision to widen the tile itself and dropped the way the thread count and stride were chosen in s75 (256 threads, stride 36).
- Record: body s74 / s75. The CPU tests stayed green, and this class of defect can only be caught on the GPU or by static inspection.

### The idea of fusing rmsnorm with the residual addition (dropped in s84, `66698f68`)
- What we tried: T8's question. The decision line is 0.25 s.
- Measured value: normalization is 9B's 5,866 tok × 4,096 dims × 16 layers = 1.54 GB, about 3.1 GB of reads and writes. About 3 ms at 1 TB/s, about 31 ms even at 100 GB/s (both conversions/estimates). 0.25% or less of the 13.2 s prefill.
- Why we dropped it: it does not reach the 0.25 s decision line by an order of magnitude. The earlier value "rmsnorm 0.4 s" was a product of a measurement that wrapped the whole layer block in a parent scope.
- Record: body s84. Step 2 (fusion, batching in the token direction) was decided not to be carried out.

### The idea of choosing the q4 GEMM row tile from the projection's n (P21, no commit)
- What we tried: choose the row tile from the projection's n with `q4_pf_tile` / `pf_gdn_tile` (the occupancy lower bound and L2 upper bound expressions).
- Measured value: the breakdown of `sl_lay_gdn_proj` 18.0 s is host beta/alpha 9.75 s + h2's D2H 4.9 s + GEMM 3.67 s = 18.3 s, 1.7% off the measured 18,015.8 ms. 81% was not GEMM.
- Why we dropped it: the implementation was not committed and was discarded as WIP in the next P25's STEP 0 (`git restore` of 1 file). What remained is the note "18.0 s is a measurement from before P13" (`86c44535`).
- Record: body P21.

### The idea of adding 1 point to `sl_lay_ffn_down` (P25, no commit)
- What we tried: check whether the rule "if host or copy is 20% or more of the phase, add exactly 1 point" applies.
- Measured value: inside the interval there is only 1 launch of `dno_gemm_q4_0`, with H2D 0 B, D2H 0 B, 0 `ctx.sync()`, 0 allocations and 0 quantizations.
- Why we dropped it: the condition did not hold, so 0 lines of code and no commit.
- Record: body P25.

### The idea of dropping the synchronization between attention and o_proj (premise refuted in P20, `5f222949`)
- What we tried: judge whether the synchronization can be removed from the full layer's `sl_lay_full_core` (the host loop over heads, and the synchronization between attention and o_proj).
- Measured value: heads are not looped on the host (attention is already 1 launch/layer, `grid.y = heads = 24`). There is only 1 synchronization, the D2H completion wait for reading 345 MB of att and applying the sigmoid gate.
- Why we dropped it: dropping it requires an n-token version of the gate kernel (`.cu`). We did not take the removal of the synchronization and attacked instead from the side of "reduce the round trip itself" in P44 / P46.
- Record: body P20.

### The idea of eliminating prepare's and the sigmoid gate's host cost (P28, `3c383a7f`)
- What we tried: can the host side of `sl_lay_full_core` (prepare 44.9 ms + sigmoid gate 45.4 ms = 90.3 ms/layer) be eliminated.
- Measured value: n=14,043, 1.45 s/request over 16 full layers. The phase is device bound and the attention kernel dominates (1.15 GB/layer of transfers).
- Why we dropped it: both are necessary data movement, and eliminating them requires moving the gate to the device (`.cu`). Instead we moved the pointless 2 × 345 MB allocation (mmap + zero fill) per request to its point of use.
- Record: body P28.

### The idea of making att's H2D pinned (no target in P50, `e841db48`)
- What we tried: verify the instruction "if tail is dominated by a pageable att H2D, apply the P47 pattern".
- Measured value: the tail after P46 is 634 ms / 16 layers = 39.6 ms/layer (27B q4, n=14,043, 1 request).
- Why we dropped it: there was nothing to replace. Not a line of code was changed; only docs were added.
- Record: body P50.
