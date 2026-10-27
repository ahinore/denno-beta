# Volume 4 decode performance — from SIMD to a device-resident chain, and rejections closed by arithmetic

> [Index](README.md) / Previous: [Volume 3 Model support](20_models.md) / Next: [Volume 5 Prefill performance](40_prefill_perf.md)
> Japanese original: [../30_decode_perf.md](../30_decode_perf.md)

A record of what we measured, what we added, and what we discarded in order to make decode (the path that generates one token at a time) faster.
It begins with CPU SIMD, then moves through batching, GPU offload of attention, the device-resident chain,
and ends with the verdict that "Rust alone cannot get there" and a run of rejections closed with numbers, laid out in chronological order.
Dates are the commit author dates in JST.
`119dd92` (P8) / `1bbffff` (P9) / `4e2428f` (P10) / `16319c7` (P11) / `55c1de0` (P12) are
from the predecessor repository and do not exist in this repository.

## What this volume covers

- The order in which decode's bottleneck moved: "linear layers (bandwidth) → attention → host round trip → kernel body".
- **What we kept inside the bit-identical contract and where we relaxed it** each time we made things faster (the only relaxation is attention in P17g).
- The breakdown of 16.4 ms/token for 9B int8: the arithmetic showing that the bus is 2% or less and 98% is GPU computation.
- The 5 rejected proposals (s118/s119/s120/s121/s122) and the numbers that justified rejecting them.
- Why 27B q4 went from 75.89 to 32.42 ms/token (removal of 392 per-op linear D2H per step).

## Background knowledge

- **decode** = one step of autoregression. The entire weight set is read once per token, so it tends to be bandwidth-bound.
- **exactness contract** = the classification of how much numerical agreement is required (bit-identical / the relaxation classes R1 to R4).
  In this volume only the attention scores and softmax are relaxed, in P17g; rmsnorm, int8 GEMV, swiglu,
  expf, rope, QK-Norm and fp32 matvec stay bit-identical to the end.
- **device-resident chain** = a structure that concatenates the layer loop of one step inside the GPU and leaves the only
  synchronization at the end of the step, as a single logits D2H. The llama family uses P17d to P17h, the qwen35 family uses C4.
- The models are Qwen2.5-0.5B/1.5B, Qwen3.5-9B (int8) and 27B (q4). The conditions for each number are given in each entry.

## Reading order

Read chronologically. For readers in a hurry, the excerpt is **P8 → P9/P10 → P12 → P17d to P17h → C4 → s49/s50 →
s101/s108 → s157 → s174**. For the record of rejections (s118 to s122), if you only want the conclusions read them right after s101/s108;
reading s160 right after s157 shows that "the same numbers are being denied from a different angle".

## Phases covered

| phase | date | commit | one-line summary |
|---|---|---|---|
| P8 | 2026-10-12 | `119dd92` | Cut the sampler's fixed cost of 3.5 ms/step with a partial sort |
| P9 | 2026-10-12 | `1bbffff` | SIMD for the fp32 batch matvec, bit-identical without using FMA |
| P10 | 2026-10-12 | `4e2428f` | Parallelize batch attention over session × KV head |
| P11 | 2026-10-12 | `16319c7` | Fix mojibake by holding back U+FFFD in streaming detok |
| P12 | 2026-10-12 | `55c1de0` | probs@V to a row-fused sgemm and direct strided access |
| P17d | 2026-10-13 | `121aa3a` | Move elementwise ops to the GPU and make decode 1 step resident |
| P17e | 2026-10-13 | `c889336` | Make KV demand-based and growable, slots from 4 to 32 |
| P17f | 2026-10-13 | `d7d126e` | Drop the contiguity copy for scores and use strided access |
| P17g | 2026-10-13 | `bf2a3c6` | Move attention to the GPU and relax the exactness contract only here |
| P17h | 2026-10-13 | `b090d03` | Move rope to the GPU and reduce step synchronization to 1 logits |
| B2 | 2026-10-15 | no commit | Hoist the V re-read in probs@V and cut LDS by 3.7x |
| B3 | 2026-10-15 | `401b8f92` | Consolidate rope for b>1 into 1 kernel and cut CPU work |
| C4 | 2026-10-16 | `18c97fe2` | Put qwen35's decode 1 step on a device-resident chain |
| s49 | 2026-10-17 | `769aa4a` | Move the per-step 2.4 GB read of the fp32 lm_head to the GPU |
| s50 | 2026-10-17 | `fca2d93` | Delay the GDN state D2H until the point where it is needed |
| s54 | 2026-10-19 | `b635b820` | Split d2h_state into 6 buckets so the breakdown can be read |
| s101 / s108 | 2026-10-19 | `96f34fe0` | Ledger the host round trips of 1 step and show that 12 ms is unrealistic |
| s118 | 2026-10-20 | no commit | Key truncation cannot be proven numerically invariant by static analysis |
| s119 | 2026-10-20 | no commit | Design 3 speculative decoding variants and derive the break-even by arithmetic |
| s120 | 2026-10-20 | `506b515a` | Measure the acceptance rate of n-gram speculation and close variant B |
| s121 | 2026-10-20 | `568afa64` | Close swiglu epilogue fusion with a 90 ms upper bound by arithmetic |
| s122 | 2026-10-20 | `4dff66e7` | Refute key truncation executably with a regression test |
| s157 / B20 stage 1 | 2026-10-21 | `2701cf3` | Put 27B q4's linear layers on the resident chain and remove 392 round trips/step |
| s160 | 2026-10-22 | no commit | Refute the premise of a gate_up fixed cost by static audit |
| s174 / P16 | 2026-10-23 | `c3e53517` | Fold the 2 beta/alpha fp32 matvec launches into 1 |
| P-MOE-F12 | 2026-10-27 | `b43d66e` | Fused AVX2 GEMV for Q4_K in the MoE expert path |
| P-MOE-F13 | 2026-10-27 | `5828846` | Fused GEMV for Q8_0 (attention projections and shared expert) |
| P-MOE-F14 | 2026-10-27 | `58ccd86` | Fused GEMV for Q5_1 (`ffn_down_exps`) |
| P-MOE-F15 | 2026-10-27 | `74e41d3` | Parallelize the GEMV row direction and expose in-layer phases |
| P-MOE-F16 | 2026-10-27 | `a42e144` | Parallelize the GDN delta rule over heads |
| P-MOE-F17 | 2026-10-27 | `a469224` | Stop-token decode did not append the stop token to `tokens` |
| P-MOE-G1 | 2026-10-27 | `6d3fc13` | Wire ExpertCache into the MoE expert path (sealed at a default of 0) |
| P-MOE-G2 | 2026-10-27 | `e4e5547` | Wire the QSA indexer into the full attention path (`DENNO_QSA=1`) |
| P-MOE-G3 / G4 | 2026-10-27 | not merged | Widening to AVX-512 only gains −12 ms in mix for Q8_0 |
| P-MOE-G5 | 2026-10-27 | `3dc3612` | Parallelize the expert loop with rayon, 355 → 278 ms |
| P-MOE-G6 | 2026-10-27 | not merged | Keeping dense weights resident in RAM makes no difference |
| P-MOE-G7 | 2026-10-27 | no commit | Flattening expert × row-block parallelism is within the noise |
| P-MOE-G8 | 2026-10-27 | not merged | Head parallelism for attention moves attn 90 → 93 ms |
| P-MOE-G9 | 2026-10-27 | no commit | madvise read-ahead for the selected experts makes no difference |
| P-MOE-G10 | 2026-10-27 | not merged | Software prefetch in the GEMV row loop makes no difference |
| H1 | 2026-10-27 | no commit | Reusing GPU scratch takes 302 → 313 ms |

## Gap between the ceiling (theoretical values) and measurements

| item | theoretical value or upper bound | measured | cause of the gap |
|---|---|---|---|
| bus transfer of 9B int8 decode 1 step | 0.199–0.333 ms (2,852,864 B/step, computed) | 16.4 ms/token (accepted value) | transfer is 2% or less. The remaining 98% is GPU computation |
| P2-1 target for 9B int8 decode | 12 ms/token (target value) | 16.4 ms/token | the Rust-only levers total 0.18 ms/step (about 1%) |
| lower bound of weight reads for 27B q4 decode | 9.3 ms/token (16.7 GB ÷ peak ~1.79 TB/s, converted (estimated)) | 32.42 ms/token (median of 3) | effective bandwidth is a fraction of peak + scan/attention/launch |
| 27B q4 decode (before B20 stage 1) | same as above | 75.89 ms/token (median of 3) | per-op linear D2H within the step was 392 times/step |
| attention of 1.5B int8 long-context decode | no data | CPU 18.52 s → GPU 4.23 s (t≈7700, 128 tok) | on short text (t≈130) synchronization dominates and no difference appears |
| launch reduction for 27B q4 decode (P16) | 610 launches/token (−7.3% from 658) | 0.05–0.10 ms/token is **prediction only, not measured** | because enqueue overlaps with GPU execution |

---

### P8: rewriting the sampler (2026-10-12, `119dd92`)

**Intent** — In the P7.5 profile the sampler remained as a **fixed cost of about 3.5 ms/step**
(3.7% in fp32, 9.9% in int8). The softmax over the entire 152k vocabulary and the top-p sort ran every step, so
the faster the linear layers became, the larger its relative weight grew.

**What changed** — `crates/denno-engine/src/sampler.rs`. For top_k we use `select_nth_unstable` + a partial sort;
top_p alone is computed over the top 2048 subset and falls back to the old full path only when the boundary is hit;
plain temperature (the production default) drops the sort for two O(n) passes with scratch reuse.
**The greedy path is completely unchanged** (temperature=0 goes straight to argmax).

**Stumble** — If the top_k tie-break diverges from the old implementation, the baseline output moves. We wrote a
bit-identical reference-implementation test at the same time and pinned it. Separately, the temperature path changes
the RNG-to-token mapping relative to the old implementation, so we **documented it as a specification change**
and made greedy invariance and invariance of the 0.5B baseline output the conditions (CI ✅).

**What we learned / remaining issues** — the sampler went from 3.5 to **0.66 ms/step (5.3x)**, and its share
from fp32 3.7% → 0.7%, int8 9.9% → 2%. Tests were 98 all green (+5). After this the sampler dropped out of the top of
the profile and the next bottleneck moved to the linear layers and attention.

**What worked next** — The line "do not touch the greedy path" became the premise that let the A/B tests of
P9/P10/P12 compare while sharing the baseline output. Leaving the temperature-path incompatibility as a documented
specification change also paid off in the later discussion of baseline updates (P17f).

### P9: SIMD for the fp32 batch kernel (2026-10-12, `1bbffff`)

**Intent** — With P6's continuous batching at b=4, the fp32 path **transitioned to compute-bound**.
`solo` is 15.4 tok/s (64.7 ms/step), but on the batch side the per-token cost of the linear layers stayed at
61.0 ms, so throughput did not grow with width.

**What changed** — `matvec_rows_bt` in `crates/denno-engine/src/tensor.rs`.
**Lanes are laid only along i (the 4 sessions in the batch) and j (ILP over 4 output channels), not along k** —
the k accumulation of each output element stays strictly sequential, using only separate multiply and add instructions
with no FMA (rounding kept separate). This preserves bit identity with `matvec_bt` for single decode. Activations are
transposed to `[k][mp]` with zero padding so that lane loads are contiguous (m*k is at most a few tens of KB, so the
transpose cost is negligible). Two implementations, aarch64 NEON and x86 SSE, plus a scalar fallback.

**Stumble** — The bit-identity test shapes were insufficient, so we extended them to m∈{1..8}, k∈{1,4,65,896} and
including n remainders. In e2e **throughput stalled at 19.4 tok/s (21.8 without the profiler)**, not reaching the
microbenchmark's bandwidth floor (width 4 = the solo step time). We estimated the causes as the average batch width
dropping to **2.15** (sessions with output lengths of 45 to 83 tokens finish early) and memory-system contention on the
real server (interference between activations/KV and the weight stream). **Both of these are estimates; the isolation has
not been measured.**

**What we learned / remaining issues** — 106 tests all green; on hardware, fp32 and int8 both show 4 sessions
concurrently == solo exactly. The kernel alone is **3.92x** on gate_up (k=896, n=4864) and **5.86x** on down
(k=4864, n=896) (vs solo ×4). In-server per-token linear cost went 61.0 → 47.5 ms (1.28x), almost matching the
1.36x throughput ratio — **the linear layers are no longer the main bottleneck**. What remains is (1) the drop in
average batch width, (2) memory-system contention, (3) attention/prefill (the part that does not grow with width).

**What worked next** — The measurement discipline of "judge by in-server per-token cost, not by the kernel-only ratio"
was established here, and continued in the same form through P10's interleaved A/B, P12's per-op profile and s157's
fixed KPI protocol. Batching attention was explicitly named as the next candidate.

### P10: parallelizing attention (session × KV head) (2026-10-12, `4e2428f`)

**Intent** — In the P9 profile attn_core was 5.5% under batching. Since the linear layers had improved to 1.28x,
attention was the largest of the remaining "parts that do not grow with width".

**What changed** — We parallelized the attention of `forward_batch` with rayon. The task unit is
**(session i × KV head kvh)** (at most 32 tasks at 0.5B), and each task holds a non-overlapping slice of `att`.
The sequence of operations inside a task is exactly the same as the single-decode (n==1) path (sgemm_acc → softmax →
matmul, same shapes), and since there is no numerical interaction between sessions/KV heads, **bit identity is
trivially preserved**. KV reads are shared through the new `KvCache::kv_bufs_ref(&self)` (read-only). The single-decode
path and the prefill path are unchanged (the discipline of not touching the reference path).

**Stumble** — The first before/after comparison produced new 41.2 tok/s vs old 21.8 tok/s (2x), but this was
**measurement contamination**. Other inference workloads running continuously on the same host use the CPU, so
background load varies with time of day and absolute values from distant points in time cannot be compared. Switching to
interleaved A/B settled the difference at +4%. The lesson is "performance measurement on the same host must always be
new/old interleaved + the same workload".

**What we learned / remaining issues** — With 2 interleaved pairs, batch 4×256 throughput was **33.4 / 33.1 tok/s**
(average 33.2) vs old 31.6 / 32.2 (average 31.9) = **+4%**, with the new code ahead in both pairs. solo 256 was
17.3 tok/s, unchanged. This is physically consistent too (attn_core is 5.5% under batching, so recovering most of that
wall time gives a few percent). 106 tests all green, 4 sessions concurrently == solo exactly on hardware, and CI (x86)
passed as well.

**What worked next** — Interleaved A/B, so as not to misread "+4%" as 2x, became the default practice for every later
comparison (P12's 3.1 to 3.4%, the ON/OFF of the P17 series, s157's KPI). Keeping a concrete instance of measurement
contamination together with its numbers also helped detect the later case of "thinking we had flipped a flag while
hitting the same server" (P12).

### P11: fixing byte-split characters in streaming detok (2026-10-12, `16319c7`)

**Intent** — Not a speed story. It started when the gpt2 exactness test failed with mojibake, re-downloading the model
did not fix it (the same file, 548,105,171 B; English raw generation was fine), and we found the cause was not the model.
**The reason it belongs in this volume** is that the separation established here — "text path only, numeric path
unchanged" — became the criterion by which later speed A/B tests could be judged by text diff.

**What changed** — In `emit_delta_and_check_stop`, tokens whose tail would become U+FFFD are **held back**, and flushed
together as a delta at the step where the character completes. The new `flush_stream` flushes the held-back part at the
end of generation (EOS/max_tokens) (both step_chat and step_chat_batch; the EOS part is excluded from decode).

**Stumble** — There were 2 causes. (1) The CI exactness job only pulled Qwen, so the gpt2 test had **never once run**
("model not cached → skip") (the baseline string remained from the conditions at the time). (2) With byte-level BPE,
multibyte characters split at token boundaries, unfinished byte sequences become U+FFFD in decode, and mixing that into
already-committed text shifts the subsequent byte diffs and drops characters. Verification used HF transformers (fp32,
greedy) as the reference implementation on the same prompt, and denno's output matched exactly.

**What we learned / remaining issues** — 99+7 unit tests all green, 5 exactness tests all green (aarch64), and in CI
gpt2 actually ran on x86 and was green. The numeric path is unchanged (confirmed by measurement: baseline output and
4 sessions concurrently == solo). **Remaining issue**: holding back in streaming delays emission, so the perceived
first-token speed drops slightly (not measured).

**What worked next** — The pattern "green is not evidence that the path ran" appears once here (skipped because the
model was not cached). P17g rediscovered the same pattern as a "path gate", and in C4 the tests evolved into explicitly
verifying a Some return. The separation of the text path and the numeric path was also fixed at this point.

### P12: row fusion of probs@V in decode attention + direct strided access (2026-10-12, `55c1de0` / `f3dea3b`)

**Intent** — The development environment migration was complete, so in a clean environment with no load from other
workloads we quantified the remaining bottlenecks since P9 (drop in average batch width / memory-system contention /
prefill interruption). Per op at int8 W=4, gate_up 29.5% was followed by **attn_core 19.4%**, down 18.0% and lm_head
10.8%, so in int8 attention was the bottleneck after the linear layers (KV contiguity copy + 7 per-(kvh,g) m=1 sgemms +
23 allocations per task, with the t×dh buffer above the mmap threshold causing mmap/munmap every step).

**What changed** — (1) `tensor::matmul_rows_bstrided` — probs@V went from per-g matmul (m=1) × group plus a vh
contiguity copy to **1 row-fused sgemm + direct strided access into the cache** (no copy).
(2) New `attention_decode_kvhead` — `forward()` (n==1) and `forward_batch()` were unified to go through the same
function. (3) Removed the qg copy and reused kh/scores from thread_local scratch (removing mmap churn).
(4) The scores side keeps the previous path.

**Stumble** — matrixmultiply **switches its SIMD kernel by stride and row count, so the numbers change**. On the scores
side (the path passing b in the transposed layout rs=1/cs=k), neither row fusion (rows=m) nor strided access was
bit-identical (measured). On the probs@V side (row-major b), row fusion + strided access was bit-identical (measured) —
that is, **only the V side could keep the bit-identical discipline**. We also hit 2 problems on the measurement side.
(1) We started the A/B servers under different binary names, but the pkill pattern did not match the old name, so
**all 12 runs hit the same server and produced invalid data** (fixed on the 2nd attempt, checking the quant mode log
every time). (2) The per-op profile's `grep -A14 … | tail -13` **cut off the first line (the largest op)**, and we
misread attn as 48.9% (correctly gate_up 29.5% / attn 19.4%).

**What we learned / remaining issues** — In the clean environment, with 2 interleaved rounds and slot 8, the A/B gave
fp32 W=4 26.8/26.8 → 27.7/27.7 tok/s (**+3.4%**), int8 W=4 126.4/122.5 → 130.4/126.2 (**+3.1%**) and
int8 W=8 119.0/113.5 → 121.3/119.3 (**+3.4%**). Per op (int8 W=4, 1022 tok, profiler ON) attn_core went 1,543 →
**1,193 ms (−22.6%)**, TOTAL 7,942 → 7,796 ms. Tests were 100+7 all green plus `decode_attention_probs_v_bitexact` +
5 exactness tests (bit-identical including the baseline output). **Remaining issue**: a fundamental optimization of the
scores side (k-direction SIMD / packed reuse) is incompatible with the bit-identical discipline and requires a decision
on whether updating the baseline output is acceptable — this is resolved in P17f when approval to "change the baseline"
is obtained.

**What worked next** — The understanding that "the numbers change depending on which kernel is selected" became the
foundation for P17f's design (baseline update + HF reference verification) and for P17g's decision to relax attention.
The 2 measurement failures (wrong process, truncated first line) became the premise for how per-op tables have been
read since.

### P17d: GPU elementwise ops + the decode chain (2026-10-13, `121aa3a`)

**Intent** — In the P17c profile, swiglu 5.1% and host round trips in various places remained. Two items:
(1) GPU kernels for elementwise ops (rmsnorm/residual/swiglu), (2) turning decode 1 step into a pipeline with hidden
states resident on the device. Because CPU attention uses `matrixmultiply::sgemm` (SIMD/FMA, with an
implementation-dependent accumulation order), moving attention to the GPU would require a decision to relax the
exactness contract → we deferred it and did this phase, which needs no such decision, first.

**What changed** — In `kernels/kernels_elem.cu`: `dno_add_rmsnorm` (the sum of squares is computed by 1 thread in the
same sequential order as the CPU, reproducing `1/(ss+eps).sqrt()` → `x*inv*w` left-associative), `dno_swiglu` and
`dno_expf_probe`. Compilation uses `-fmad=false` plus explicit `__*_rn` to control contraction. In
`crates/denno-engine/src/cuda.rs` and `model/llama.rs`, the chain "embedding H2D → [rmsnorm → qkv → (CPU:
rope/attention/KV) → o_proj → rmsnorm → gate/up → swiglu → down] × layers → final norm → lm_head → logits D2H".

**Stumble** — **Exactly reproducing glibc expf was the hardest part.** swiglu's `exp(-v)` has to be bit-identical to
Rust's `f32::exp`. glibc e_expf uses f64 internal arithmetic + a 32-entry table + a cubic polynomial and can be
reproduced on the GPU, but **the actual CPU binary is the FMA-contracted version** (multiarch IFUNC selects the FMA
variant), and a straightforward sequential implementation differed by 1 ULP in the final f32 (measured at x=32.564632f,
the only mismatch in a 455M-value sweep). We disassembled with gcc -O2 -mfma, identified that the contraction spans
**5 places**, reproduced it with `__fma_rn`, and obtained bit identity over 455,078,062 values. Two more: the `f32::exp`
loop used for comparison can be replaced by libmvec through LLVM auto-vectorization, so we pinned it to a scalar call
with `#[inline(never)]`. With `cargo build -p denno-cli` the cuda feature does not propagate to the CLI, and **a binary
without the feature prints no GPU log and silently runs on the CPU**.

**What we learned / remaining issues** — GPU tests 12/12, no regression across 106 CPU tests, and with Qwen2.5-1.5B
int8 the GPU serve and CPU serve responses were exactly identical (2 prompts, byte-identical apart from msg id).
**At 200 tok, GPU 2.1 s vs CPU 6.6 s (3.2x); at 150 tok, 1.46 s vs 6.6 s (4.5x)**. The remaining CUDA candidates are a
fused attention kernel (requires a relaxation decision), paged KV, and a sampler kernel.

**What worked next** — Showing with numbers that the direction "remove host round trips" works became the direct
premise for P17g/P17h's synchronization removal and for C4's chain design. The expf lesson (the contract depends on the
host's glibc build) also foreshadowed the PTX/source mismatch found in P17f.

### P17e: demand-based KV allocation (growable cache + reservation) (2026-10-13, `c889336`)

**Intent** — P15's KV budget assumed "1 slot = preallocating the full context width", and Qwen2.5-1.5B reserved about
1.9 GB per slot and about 7.5 GB just before startup with the default 4 slots. Real sessions are mostly a few hundred
to a few thousand tokens, so we make it **grow according to the amount written** and raise the slot count.
paged KV (block splitting) requires a large structural change, so we deferred it and made it demand-based while keeping
the contiguous layout. The reason it belongs in the decode volume is that the slot count is directly the upper bound on
concurrent decode width.

**What changed** — `crates/denno-engine/src/kv_cache.rs`. Initial allocation is 0, and the buffers of all layers grow by
doubling according to the amount written (cap max_len, amortized O(1), existing rows preserved by realloc). The layout
stays "a contiguous buffer whose first len rows are used", so the numeric path including attention is exactly the same
as the old implementation. `DEFAULT_BATCH_SLOTS` goes from 4 to **32**, and the default is computed as "the number of
slots for which 4096 tokens of guide reservation per session × bytes_per_token fits in the KV budget" (at 1.5B int8 with
bpt 57,344 B, 235 MB of reservation per slot → 32 slots for a 12 GB budget). `install_session` registers prompt +
max_tokens as a reservation, and overruns are rejected with 400.

**Stumble** — **Models that advance len via `advance()` alone (dummy models) could skip capacity allocation.**
Demand-based allocation makes it a contract that "a model writing KV must reserve capacity with `kv_bufs(layer, n)`
before writing", and we adapted the 3 dummy models and the server's test model to it. The old implementation was
"advance alone moves the snapshot" (everything preallocated), so the tests hid this contract.

**What we learned / remaining issues** — Qwen2.5-1.5B int8's RSS right after startup went from **about 9.9 GB to about
2.5 GB** (the old implementation preallocated 4 slots × 1.88 GB). e2e (200 tok) was GPU 2.15 s vs CPU 9.1 s with exactly
identical GPU/CPU responses. We also confirmed on hardware that a server started with a 1 MB budget correctly rejects
reservation overruns with 400. CPU tests 110 ok, GPU tests 12/12, no wire test regressions.
**Not measured**: throughput of decode with 32 slots filled (width 8 or more) has not been measured.

**What worked next** — Raising the width limit from 4 to 32 removed P12's constraint that "W=8 needs 8 slots", so later
batch A/B tests could vary width without worrying about the slot count. The design of "closing off OOM with
reservations" is the same idea as C4's per-step bulk allocation (the fix for the segfault and the nondeterminism).

### P17f: fundamental optimization of the scores side (baseline update, user approval) (2026-10-13, `d7d126e`)

**Intent** — The homework left in P12, where we concluded "the scores side is incompatible with the bit-identical
discipline". With user approval ("please go ahead and change the baseline"), we chose the path of **updating the
numeric baseline while keeping determinism (same shape → same result)**. Following the operating rule for baseline
updates, we verified against the HF transformers reference implementation before applying it.

**What changed** — decode scores: per-g `sgemm_acc(m=1, kh contiguity copy)` × group → 1 `sgemm_scores_kvstrided(m=group,
direct strided access to kc)`. Removed the kh contiguity copy (t_total×dh × kv_heads×layers/step of memory movement) and
KH_SCRATCH. Prefill also removed all qg/kh/vh contiguity copies and uses direct strided access to the Q row buffer and
the kv buffer.

**Stumble** — **Not bit-identical to the previous path (as expected)**. Rather than stopping there, we confirmed that
the greedy output text was unchanged in all 6 existing baseline tests (including 0.5B fp32 "1+1は?" and "日本の首都は?",
GPT-2 and Qwen3), and that the HF reference gives an exact match for 0.5B fp32 200-token greedy, and then applied it.
Separately, the committed `kernels_elem.ptx` **did not match** the result of compiling the `.cu` with
`nvcc -arch=compute_75 -fmad=false` (committed in P17d without checking the hash). That was the cause of the GPU expf
test showing 1 mismatch in 455,078,062; replacing it with the recompiled version restored 12/12.

**What we learned / remaining issues** — Performance on 1.5B int8 200 tok e2e: **CPU warm 9.10 s → 8.46 s**,
**GPU chain 2.15 s → 1.57 s** (the chain's attention improved automatically through the same function). Determinism is
maintained by the single/batch decode bit-identity tests and the prefix cache exactness test. **Remaining issue**: the
baseline moved, so later comparisons can only be made "between the same baselines" (the anchor md5s need re-baselining).
**Lesson: artifacts (.ptx) must always be hash-compared against a build in the verification environment before
committing.**

**What worked next** — The procedure of "accepting a numerically-changing optimization in 2 stages, reference
implementation and text invariance" was established, and P17g's attention relaxation (accepted with maxabs 2.98e-8 and
matching greedy sequence) passed with exactly the same pattern. Hash comparison of PTX settled in B2/B3 as the procedure
"first verify that recompiling the pre-change source matches the committed PTX".

### P17g: GPU offload of decode attention (2026-10-13, `bf2a3c6`)

**Intent** — In the P17c profile attn_core became the largest op (32.7%), and CPU-side attention was the last big
remaining item for GPU offload. With user approval ("a CPU/GPU mismatch is acceptable"), we **relax the exactness
contract for attention only** and move it to the GPU. The bit-identical contracts for the linear layers (int8) / expf /
swiglu / rmsnorm are not changed.

**What changed** — In `kernels/kernels_elem.cu`, `dno_attention` (grid=(heads), block=(dh), 1 block handles 1 q head,
GQA with kvh=h/group, scores → softmax (exp via `dno_expf`) → probs@V, shared memory dh*4 B). In `cuda.rs`, a GPU KV
mirror (the host KvCache is authoritative, with a per-session copy on the device keyed by `kv_id()`, missing rows bulk
H2D'd in the next chain_attention, LRU eviction when `DENNO_GPU_KV_MB` (default 8192 MB) is exceeded). Applied to both
the solo and batch paths, adopting only when all sessions succeed and redoing everything on the CPU if even 1 fails
(preventing mixing), so single/batch bit identity is maintained.

**Stumble** — **The production Qwen2.5-1.5B has QKV bias** (from safetensors), so the solo chain had **never once been
triggered** since P17d (P17f's GPU e2e speedup came from per-op GPU linear). The initial implementation gated on
chain_compatible, so GPU attention was not triggered either; we identified this by adding temporary logging, removed the
gate, and integrated it into the per-op path as well. **Lesson: "green" is not evidence that the path ran — verify the
path gate once against the actual model structure.**

**What we learned / remaining issues** — With 1.5B int8 long-context decode (t≈7700, 128 tok), **CPU attention 18.52 s
→ GPU attention 4.23 s (4.4x)**. On short text (t≈130) ON/OFF is about the same (synchronization-bound). Tests: kernel
vs CPU reference maxabs **2.98e-8** + GPU/CPU greedy sequence match, GPU 14/14, CPU 110, 6 exactness baselines passed.
**Remaining issues**: (1) prefill of long prompts is slow (about 100 s at t≈2810, a pre-existing problem unrelated to
this work); (2) GPU/CPU e2e responses become non-identical because of the relaxation (`DENNO_GPU_ATTN=0` still allows
bit-identical comparison as before); (3) the effect is thin on short text (rope/synchronization remain).

**What worked next** — (3) became the direct motivation for P17h. Also, the pattern of "always provide an escape hatch"
(`DENNO_GPU_ATTN=0`) became the practice for every GPU path after P17g.

### P17h: GPU offload of rope + a synchronization-free decode chain (2026-10-13, `b090d03`)

**Intent** — Resolve P17g's remaining issue (3) (thin effect on short text = rope/synchronization remain). rope is not
subject to the exactness relaxation, so we move it to the GPU **while keeping bit identity mandatory**.

**What changed** — The `dno_rope_qkv` kernel (bias addition to qkv + rope in 1 kernel.
grid=(heads + 2*kv_heads), block=(dh); block b < heads is q, the next is k, the rest is v with bias only).
The bit-identical design computes cos/sin on the host (the same process's glibc sinf/cosf) with **the same formula** as
CPU `rope_row`, uploads the table `[cos(half)|sin(half)]` once per step, and the device side applies it using only
mul.rn/add.rn/sub.rn with `-fmad=false`. The bias addition happens before rope (same order as the CPU).
Furthermore, the qkv linear output stays resident on the device, rope/attention also complete inside the device, and the
roped k/v is escaped by asynchronous D2H to a pinned staging area, with the reflection into the host KvCache done only
once after the step's final sync.

**Stumble** — Removing the in-step sync introduces a new **overwrite conflict** on pinned buffers. We dropped ping-pong,
switched to a dedicated slot per (layer × purpose), and avoided it by id assignment. On the applied path,
`forward_batch` must **delegate b==1 to the solo chain** (so that it affects the server's actual decode path), so we
added a dual path that falls back to the conventional batch path on failure. We removed the qkv bias condition from
chain_compatible (dno_rope_qkv supports bias).

**What we learned / remaining issues** — With 1.5B int8, short text (t≈38, 2 tok) **11.0 ms** (compared with the batch
path's 14.4 ms with `DENNO_GPU_ATTN=0`; the synchronization reduction effect), long-text decode (t≈7657, 128 tok,
**warm**) **4.23 s → 3.88 s (1.09x)**. Cold prefill is unchanged (about 366 s, a pre-existing problem). Tests:
`cuda_solo_chain_bias_rope_gpu_matches_cpu` **explicitly verifies activation of forward_decode_gpu via a Some return**,
GPU 16/16, CPU 110, 6 exactness baselines passed. The PTX sha256 was compared against the recompiled result.

**What worked next** — The skeleton "1 sync per step" was completed here, and C4 (qwen35)'s chain reused the same
skeleton as is. The 11.0 ms figure for short text later became the starting point for the judgment that "in short text
only synchronization is left" (the s101/s108 inventory).

### B2: V accumulation optimization of attn_core — row chunking of probs@V (2026-10-15, no commit)

> Related commit: `401b8f92` (B3, the same `kernels_elem.ptx` series). B2's own commit cannot be identified from the records.

**Intent** — After P17k the remaining bottleneck was attn_core 3,514 ms (the largest op, cold prefill of 13,252 tokens).
It surfaced as the largest op because B1 eliminated rmsnorm. We carry out P17k's "remaining implementation candidate" 1,
"reducing V re-reads in the V accumulation (row chunking)". The condition is that it is only an internal kernel
restructuring and **the numbers are bit-identical** (the same output as the old kernel).

**What changed** — Only the probs@V accumulation part of `dno_attention_prefill` in `kernels/kernels_elem.cu`.
The old implementation re-read `v_sm[kk]` once per row (PF_BQ=64) inside the row loop, issuing per thread per tile
s 512 + v 512 + corr 64 smem load instructions. We hoisted `v_sm[kk]` out of the row loop, interleaved 2 rows for reuse,
and changed `s_sm`/`corr_sm` to float2 (8 B) loads → **s 256 + v 8 + corr 32** (v re-reads 64 → 1, LDS instruction count
reduced by about 3.7x).

**Stumble** — Guaranteeing bit identity. We kept the FMA chain into acc[r] "fixed r, kk ascending" exactly the same as
the old implementation (the inner part of `fmaf(p.y, v1, fmaf(p.x, v0, acc))` is kk and the outer is kk+1), and compared
the old and new loops in host C++ with 128 threads × 64 rows × random values (including subnormals, ±0 and maximum
values), confirming all bits identical at both -O0/-O2. Alignment is also a condition: since the start of an s_sm row
is at 32 B steps and the dynamic smem base is 16 B aligned, float2 access alignment always holds (kk is at even steps).

**What we learned / remaining issues** — As a side effect of the PTX static analysis, the old PTX spilled acc[PF_BQ] to
local memory (ld.local 81 / st.local 33), whereas the new implementation has **0 spills**, and dynamic smem loads in the
V accumulation went from ~1,088 to ~296 instructions/thread/tile. The other kernels are byte-identical to the old PTX
(the difference is local to this kernel). **Not measured**: kernel-only timing of `cuda_attn_pf_bench` (n=t=13252) (the
improvement from the old 69.3 ms/call) and E2E wall. BK 8→16 needs ~54 KB of smem and is opt-in, and carries the risk of
occupancy dropping from 2 blocks/SM to 1 blocks/SM, so we left it for a hardware A/B decision.

**What worked next** — The order "count the smem load instructions before writing, and fix bit identity first with a C++
simulation" was reused as is in B3 and P16 (folding beta|alpha into 1 launch).

### B3: GPU rope for b>1 batch decode (2026-10-15, `401b8f92`)

**Intent** — B3 of the P17 series' remaining issues. rope in the batch path (b>1) was applied per session on the CPU via
`tensor::rope_row`, so powf/cos/sin ran session × (heads+kv_heads) × half × layers × steps times. Because
"synchronization is hidden under batching", the effect is mainly reduced CPU work and it was considered low priority.

**What changed** — New `dno_rope_qk_batch` (grid=(heads + kv_heads, b), block=(dh), no shared; q is `q[i, b*dh..]`, k is
`k[i, kh*dh..]`, in-place row-major, table is `[b, dh]`). In `cuda.rs`, `batch_rope_qk` (device buffers for q/k + a
`[q|k]` concatenated pinned staging, H2D → kernel → sync → D2H → sync → host write-back). On failure it returns false
with the host q/k unchanged, and the caller redoes it with CPU rope (no GPU/CPU mixing). The table is cached on an exact
match of (poss, dh, theta), and from the 2nd layer onward in a step's layer loop the generation + H2D are skipped.

**Stumble** — rope is not subject to relaxation, so bit identity is mandatory. The table is generated by the host with
the same formula as `tensor::rope_row` (the same process's glibc powf/cosf/sinf), and the kernel uses only mul/sub
(-fmad=false → the PTX is only mul.rn/sub.rn/add.rn). Verification was a host C++ simulation over 8 shapes × 4 theta ×
3 pos × 40 trials, with **5,427,840 element comparisons all bit-identical at both -O0/-O2 (-ffp-contract=off)**.
Shape guards (b ≤ 65535, dh ≤ 1024, heads+kv_heads ≤ 65535) silently fall back to the CPU when not satisfied.

**What we learned / remaining issues** — CPU tests all green (lib 147 unchanged); the PTX was regenerated after first
verifying that recompiling the pre-change source matched the committed PTX (B2's sha).
**Not measured**: the wall of b>1 multi-session E2E (the reduction in the rope phase). Since per-op accounting became
meaningless after B1, the reduction can only be produced as wall.

**What worked next** — The pattern "build the table on the host and do only the 4 basic operations on the device" was
ported as is to C4's QK-Norm + partial interleaved rope (`dno_qknorm_rope_pi`).

### C4: device-resident decode chain for qwen35 (2026-10-16, `18c97fe2`)

**Intent** — Run qwen35 (9B/27B)'s decode 1 step (n==1) on a device-resident chain with the same skeleton as the llama
family's P17b-g/P17h solo chain. qwen35 has GDN layers (state has 2 faces, conv/ssm), so a mirror of the GDN state was
needed in addition to KV. The design followed `DESIGN_C4.md`.

**What changed** — `forward_decode_chain` in `model/qwen35.rs` (GDN layers branch to `chain_q35_gdn_scan`, full
attention layers to `chain_q35_fullattn`; tried at n==1 in `forward()` and returns immediately if Some, otherwise the
conventional per-op path). 3 kinds in `kernels/kernels_elem.cu`: `dno_matvec_f32` (dedicated to the beta/alpha control
projections, 1 thread per output, sequential over k, bit-identical to `tensor::matvec_bt`), `dno_qknorm_rope_pi`
(QK-Norm + partial interleaved rope in 1 launch, thread 0 computing the sequential sum of squares identical to the CPU)
and `dno_attn_gate_mul`. GDN state is held in 2 ping-pong faces, and the new state is reflected into the host cache
after staged async D2H → the logits download sync (**the host state is always authoritative**).

**Stumble** — After 192 green CPU verifications, the first request on hardware **killed the process (segfault, reproduced
3/3, fault addr with the same tail pattern every time)**. The cause was that the pinned buffer for the GDN staged D2H was
allocated for only 1 layer (the grow-only `ensure_pin` judged that it would fit from the 2nd layer onward and did not
reallocate → the DtoH of the 2nd GDN layer wrote outside the pinned region). The fix is to allocate all GDN layers at
once at the start of the step. Next came **nondeterminism where all runs of the same payload produced different output**
(4 requests, all different text, speed normal at 27.9 ms/step). The cause was that mid-step grow reallocation of CH_QKV /
CH_ATT / aq **freed the in-flight buffers of the immediately preceding launch**, which fired reliably at layer 0 of the
first step. The fix extended the "allocate the full size first" rule to all buffers within a step (no change to the
numeric path).

**What we learned / remaining issues** — After the segfault fix, 9B was **27.9 ms/step (2.8x from the old 77 ms)**.
After the nondeterminism fix we left a procedure to confirm md5 identity across 4 runs of the same payload on hardware
(not carried out). CPU tests are 205 passed. **Remaining issue**: the output when the chain is active can change with the
k-path fix, so anchors need to be re-acquired (human action).

**What worked next** — The premise "green on the CPU can still die on the GPU" was confirmed here, and subsequent GPU
changes came in a set of 3: (1) explicit verification of activation (Some return), (2) lifetime audit of in-flight
buffers, (3) single-sourcing byte counts in pure functions. s50's delta approach and s157's q4 chain ride on this
skeleton.

### s49: GPU offload of the fp32 lm_head (2026-10-17, `769aa4a`)

**Intent** — The serve default is fp32 mode (the default for `--quant`), and in that mode neither the decode chain
(int8 only) nor `maybe_gpu_linear` (Q8 only) is triggered, so lm_head (vocab 151936 × e 4096 ≈ 2.4 GB) remained a CPU
matvec. **That meant a 2.4 GB weight read every step.**

**What changed** — New `maybe_gpu_matvec_f32` in `cuda.rs`, reusing C4's chain control-projection `dno_matvec_f32` as is
(no change to `.cu` / PTX). The launch is **block=(1,1,1), grid=(n,1,1)** — the kernel has all threads in the block
perform the same computation, so using the chain's block=128 convention at vocab scale would multiply redundant work by
128. Weights are synchronously H2D'd once with a key unique per model load.

**Stumble** — An accounting error was found. s45's hardware measurement showed the decode lm_head bucket at
**46.3% = 19.6 ms/step**, but in the old accounting the chain path's lm_head bucket absorbed the final `chain_download`
(all staged async D2H + the sync waiting for the whole step's GPU completion), so it was **over-attribution that included
the sync wait**. We added a D2H op (id 14) to `prof.rs` and separated it (observation only).

**What we learned / remaining issues** — No numerical effect (the GPU path is all-bits identical to `tensor::matvec_bt`:
separate rounding of multiply and add, sequential over d, 1 block = 1 output). CPU tests 253 passed / 18 ignored, the
same as s48. **Not measured**: decode ms/step of fp32 serve. The expected effect (after the initial 2.4 GB H2D, h 16 KB +
logits 0.6 MB/step) is an **estimate**, and re-measuring on hardware was left as a human action.

**What worked next** — The measurement hole "when a bucket absorbs a sync, you read that op's cost too high" remained on
record, and s54's finer measurement and s101/s108's inventory (from the era when d2h_logits looked like 13.6 ms) closed
the same hole by name.

### s50: making the decode D2H a delta (the GPU GDN mirror becomes authoritative) (2026-10-17, `fca2d93`)

**Intent** — The C4 chain returned the GDN new state (conv/ssm) to the host via staged async D2H at the end of every
step. At 9B this was **~75 MB/step** (an **estimate** noted in the implementation comment, not a measurement) and had
become the dominant term of decode. Since there is a 2-face ping-pong mirror, the next step always reads from the mirror
— the host only needs it in 2 places: (1) the next prefill, (2) CPU fallback when the chain is not active.

**What changed** — Added `gdn_pending: Option<usize>` to `KvMirror` (while Some(pos), the host state may be stale).
Added a `gdn_pending.is_none()` condition to the LRU eviction candidates (if the mirror disappears before the flush,
recovery is impossible). Removed the staged D2H from `chain_q35_gdn_scan` and added `chain_q35_gdn_flush_pending`, which
does a synchronous D2H only "at the point where the host state is actually needed". Removed the old
`chain_q35_gdn_stage_*` and `q35_gdn_stage_bytes`.

**Stumble** — Position consistency. Per layer, if `gdn_pos == target` the cur face, if `== target+1` the 1-cur face (the
input face of a step that failed mid-layer), if `None` the chain did not run this layer and the host is current;
anything else is treated as inconsistent, pending is discarded and None is returned. The D2H happens while holding the
mirror lock (releasing it lets LRU eviction remove the source). None means "unrecoverable" and the caller drops the
request — it does not silently continue with a stale state.

**What we learned / remaining issues** — No numerical effect (the transfer is the bits as they are, no arithmetic; the
ping-pong read face is unchanged). CPU tests 253 passed, the same as s49. **Not measured**: the reduction in decode
ms/step. The effect of removing ~75 MB/step is an **estimate**, and running the ignored tests on hardware and
re-measuring are human actions.

**What worked next** — The pattern of delaying only the transfer while keeping the "host is authoritative" contract
became the material for s54's D2H breakdown (kv_flush / gdn_flush) and s101/s108's inventory (transfer is 2% or less of
the whole), leading to the judgment "cutting D2H does not make decode faster".

### s54: finer-grained measurement of d2h_state (2026-10-19, `b635b820`)

**Intent** — Split the `d2h_state` bucket remaining after s50's delta work so that its breakdown can be read from
measurements. The purpose is to identify the per-step D2H breakdown of decode, and to make it possible to judge whether
reduction is feasible by "the ratio of d2h_logits to d2h_kv_flush". **Measurement only, behavior unchanged.**

**What changed** — Without changing the raw ids in `prof.rs`, we added the 6 buckets `D2H_LOGITS` / `D2H_KV_STAGE` /
`D2H_KV_FLUSH` / `D2H_GDN_FLUSH` / `D2H_KV_DELAYED_FLUSH` / `D2H_ATTN_PF` (10*D2H + i = 140..145). `summary()` prints the
breakdown (ms / calls / cum%) as indented lines directly under `d2h_state`. We replaced the caller ids in `qwen35.rs`
(decode's logits download → `D2H_LOGITS`, `chain_q35_kv_stage_flush` → `D2H_KV_FLUSH`, prefill attention output →
`D2H_ATTN_PF`).

**Stumble** — We fixed the structure first by reading the code: in the decode chain, `d2h_state` is **2 calls** per step —
(1) `ctx.sync()` of `chain_download(CH_LOGITS, vocab)` absorbs the completion wait of the in-flight DMA (the per-layer
staged async D2H), (2) `chain_q35_kv_stage_flush` reflects the full-layer k/v rows to the host. The full layer's
`cuMemcpyDtoHAsync` only issues asynchronously and has no synchronization point, so the transfer time is all accounted to
(1). **Therefore the ratio of these 2 calls alone cannot separate "transfer" from "GPU wait"** — adding the breakdown
still requires a different instrument for the next level of isolation.

**What we learned / remaining issues** — At 9B, 16 layers × 2 × kv_dim(1024) × 4 B = **128 KB/step** for the KV flush
part (arithmetic). No numerical effect (1 branch passes through when prof is disabled, and the telemetry raw ids are
unchanged), CPU tests 261 passed, the same as s53. **Not measured**: the breakdown measurement on hardware (GPU). The 3
reserved buckets (`D2H_KV_STAGE` / `D2H_GDN_FLUSH` / `D2H_KV_DELAYED_FLUSH`) are unused at present.

**What worked next** — We found that this breakdown alone was not enough, and moved to the s101/s108 inventory of
"counting host round trips from the code rather than from bucket names". The ledger in `docs/DECODE_PATH.md` is an
extension of this.

### s101 / s108: inventory of the decode path — 12 ms/token is not reachable with Rust alone (2026-10-19, `96f34fe0`)

**Intent** — decode is 16.4 ms/token and P2-1's target is 12 ms/token. Where the time was disappearing could only be
inferred from bucket names (there was an era when `d2h_logits` looked like 13.6 ms/step). We counted every host round
trip of 1 step from the code and fixed it as a ledger in `docs/DECODE_PATH.md`. **This is not measured. It is arithmetic
derived from the code, the byte counts and s68's bandwidth.**

**What changed** — `docs/DECODE_PATH.md` (new, ledger) and, as a by-product of the instrument, `chain_download_site(slot)`
(a pure function). Logits are classified as `D2H_SRC_LOGITS` and everything else as `LINEAR`, so they do not fall into
`OTHER` (display only; the numeric path is not touched).

**Stumble** — We redid how to read the buckets 3 times. (1) `d2h_logits` looking like 13 ms was because it measured "the
wait until that step's GPU work finished", not PCIe (s51 removing the per-step D2H of KV rows changed nothing, 16.4 vs
16.5 ms, which is the same fact seen from the other side). (2) The remaining accounting hole (the H2D inside chain_rmsnorm
and the embedding upload appearing under the rmsnorm/embed measurement wrappers) we only recorded in the document.
(3) The same reading also revealed that "45 block" in `docs/TASKS.md` was wrong (5,686 tok is 89 blocks with PF_BQ=64,
with 10 rows of remainder).

**What we learned / remaining issues** — The 9B int8 chain has **only 1 sync** per step, 0 per-layer syncs, 0 in-step
allocations, 0 weight uploads, and 0 KV mirror re-seeds (`have == pos0`). The amount on the bus is H2D 1,785,856 B +
D2H 73,728 B + D2H 993,280 B = **2,852,864 B**, which at the measured bandwidths (pinned D2H 14.3 GB/s, H2D 14.2 GB/s,
pageable conservatively 8.6 GB/s) is **0.199–0.333 ms**. That is, of the 16.4 ms the bytes in motion are **2% or less**,
and the rest is GPU computation. The levers and their upper bounds (all computed values) are L1 ≤ 0.125, L2a ≤ 0.05,
L2b ≤ 0.07, L3 ≤ 0.05, L4 up to ~5% depending on startup overhead, and L5 (the kernel bodies of the GDN scan / attention /
GEMM) is **the remaining 98%**. **Verdict: there is no lever left that reaches 12 ms/token with Rust alone** (the best
Rust-only combination is L1 + L3 at about 0.18 ms/step = about 1%, smaller than the noise floor of the bus).
**The target of 16.4 → 12 ms/token has not been achieved.**

**What worked next** — The discipline of not writing "it got faster" was made explicit in this section, and every later
rejection (s118 to s122) uses this ledger's numbers as upper bounds. The remaining work on the Rust side was narrowed to
"removing the serial parts around the GPU" (s174/P16's launch reduction is the last one).

### s118: causal key range truncation is not numerically invariant (2026-10-20, no commit)

> Related commit: `3eab3629` (docs only). **0 lines of code change, no commit.**

**Intent** — P1-2 candidate 4. Verification of the idea that changing the tile-loop start of `dno_attention_prefill` from
0 to `jstart = floor((pos0+i0)/PF_BK)*PF_BK` reduces the amount of computation, since each row of block b only needs
keys `j <= pos0+i`. **The reason it stays in this volume** is that the reason for rejection was neither "bit identity" nor
"invisible at R3" but a new kind: **static analysis cannot prove numerical invariance**.

**What changed** — Nothing (design study only). The kernel in question is on the prefill side, but the type of judgment is
the same as decode's attention optimization: "the reach of the online-softmax accumulator".

**Stumble** — The premise does not hold. `jend = min(t_total, pos0+i0+PF_BQ)` does not depend on `n`, and the score
reads K/V for all `j < t_total`. When `t_total` exceeds `pos0+i0+PF_BQ` (the tail block of prefill), the kernel
deliberately computes a screen of `PF_BQ` width beyond the block's causal boundary. Those pairs do not affect the output
or the tile's `m` because of the `-INFINITY` mask, but what they do affect is the **representable range** of the
online-softmax accumulator `acc`. If earlier tiles have accumulated up to `acc = sum_j exp(s_j - m_run)` (each term ≤ 1),
the rounding unit of the FMAs that remain afterwards becomes `2^-23 * acc`. Truncation removes that accumulation.

**What we learned / remaining issues** — We identified an example where a difference appears at row granularity: with
`pos0=0, i0=64` (`jstart=64`), row 64's first tile in HEAD is `[0,8)` with 8 valid columns, whereas after truncation it
is `[64,72)` with only 1 valid column at `j=64`, so the number of columns until reaching the max changes from 8 to 1. The
magnitude of the difference is `2^-23` relative only in cases where the dot product with `d=255` cancels down to an O(1)
quotient; it is invisible at R3 (absolute 1e-3) but visible in the anchor md5 (compared to 1e-7). **"The mask path adds
exactly +0.0f" can be confirmed in the code, but "the same bits without that +0.0f" is a different proposition** and
cannot be proven by static analysis → do not commit. If it is ever introduced, a guard that explicitly skips rows with
`m_run == -INFINITY` is needed first (`exp((-inf)-(-inf)) = NaN` → `l = NaN` → output `0 * NaN = NaN`).

**What worked next** — The line "do not ship what cannot be proven" was drawn, and it motivated turning the same idea
into an **executable refutation** (a test) in s122. s117's block count error is also corrected here.

### s119: design proposal for speculative decoding — zero implementation (2026-10-20, no commit)

> Related commit: `623359bc` (design document only). **Not a single byte of `crates/` was changed.**

**Intent** — s115's inventory closed with "decode's only synchronization is the logits D2H, and removing it requires a
multi-token forward + `.cu`", so we design that "multi-token forward". The deliverables are 1 design document
(`docs/SPEC_DECODE_DESIGN.md`) and records only.

**What changed** — Nothing. Step 1 = the breakdown of 1 step (decomposition of 16.4 ms, the places that assume `n == 1`),
Step 2 = the correctness contract, implementation cost and break-even p of variants A / A-lite / B, Step 3 = a comparison
table.

**Stumble** — Variant A (verify in place) **cannot be bit-identical**. When 1 forward produces position `k+1`, the rmsnorm
/ attention of row 1 is affected by "other tokens in the same buffer", so **the thing being compared (row 1's logits)
itself changes from the sequential case**, leaving no means of inspection. It is the same kind of difference as s118 (the
reach of the online-softmax `acc`), a "path-dependent difference" that does not fit within the R1 bracket either. Even
with `λ = 1.00` (an optimistic assumption) the break-even is `p > 0.48` (including a 15 ms reject). Variant A-lite (using
the existing `n ≧ 2` path as a 2-token prefill) can keep bit identity, but 1 step is **computed at 18.3 ms**
(`λ = 1.115`) with a break-even of **p* = 0.885** — **rejected for speed purposes**.

**What we learned / remaining issues** — Only variant B (n-gram speculation) does not change the structure of 1 step
(draft = normal step, verify = normal step, reject = normal step). Bit identity and determinism stay under the existing
contract, `.cu` / anchors are unchanged, and the break-even is **p* = 0.0006**. Whether it works is determined solely by
"the rate at which the 12-gram of the immediately preceding context has the same continuation" = p, and **measuring that
needs no GPU**. At p = 0.5 / 0.7 / 0.9 the effective cost is 10.9 / 9.7 / 8.6 ms/token (**a hypothetical table**, not
measurements). Rollback exists as a mechanism (`KvCache::truncate` + `bump_epoch()` discards the GPU GDN state), but its
actual cost is **computed at 15 ms** (KV 210 MB + GDN 5.8 MB at 14.2 GB/s), paying 1 step's worth per reject.
**Not measured**: `λ`, the acceptance rate p, the actual cost of a reject, and the `n == 2` step time.

**What worked next** — Reducing the problem to the form "measuring p decides whether variant B is viable without a GPU"
led directly to s120's measurement (and rejection). Variant A-lite's prediction of "running at 2.6 ms/step per token"
contradicts existing measurements, so we kept it as 1 counter-measurement (not carried out).

### s120: measured the acceptance rate p of n-gram speculation and closed it (2026-10-20, `506b515a`)

**Intent** — Since s119 wrote that "the success of variant B is determined solely by the rate p at which the continuation
matches the previous one when a 12-gram context reappears, and that measurement needs no GPU", we build a tool and
measure. **Not a single byte of `crates/` or `kernels/` was changed.**

**What changed** — `scripts/ngram_oracle.py` (standard library only, network forbidden, no model weights needed,
randomness only from a fixed seed for the synthetic corpus). It scans the positions of context length n once in ascending
order, counts as opportunities only the positions where "the context appeared in the past", and counts matches of the
following k tokens. It takes the **immediately preceding occurrence** (matching the implementation, which searches the
context backwards; taking the 2nd or earlier occurrence would be too lenient). Tests are 29 in
`scripts/test_ngram_oracle.py`.

**Stumble** — There is no tokenizer. Tokenization is done 2 ways, `words` (runs of ASCII alphanumerics / a single symbol
/ character-class boundaries for Japanese) and `bytes` (UTF-8 byte sequences), and **neither is the real tokenizer**.
The corpus is also (a) a proxy = `docs/*.ja.md` + `docs/*.en.md` + `crates/**/*.rs` (58 documents), (b) synthetic
tool-call-like (fixed seed), and **we did not go looking for real prompt logs**. We stated "proxy / synthetic" in the
output and the document.

**What we learned / remaining issues** — Measured p (n = 12, with denominators): proxy words 1,235,606 tokens / 368,831
opportunities gives **p_1 = 0.2984** (110,072 matches), p_2 = 0.1521, p_3 = 0.1216, p_4 = 0.0970. Proxy bytes 2,667,516 /
955,830 gives **p_1 = 0.1265**, p_2 = 0.0588. Synthetic words 226,194 / 205,686 gives **p_1 = 0.1156**, p_2 = 0.0286.
Shortening the context does not raise it (proxy words n=2 gives p_1 = 0.3316, n=4 gives 0.3400). **Verdict: variant B is
closed.** Looking only at the ratio to p* = 0.0006 there is 190 to 550 times the margin, but variant B pays 2 steps (draft
step + verify step) for 1 accepted token, so the effective cost is `(2S + C_d)/(1 + p_1)`; with `S` = 16.4 ms and
p_1 = 0.2984 that is **25.3 ms/token = +54% over sequential**. To win it needs `p_1 > 1 + 2 C_d / S` = **1.00001**
(impossible with C_d ≥ 0), and **no amount of measuring p overturns that**.

**What worked next** — The form of rejection "if the break-even formula does not depend on the value of p, measuring p
does not change the verdict" became established as the same "closed by arithmetic" type as s121 (upper bound 90 ms <
acceptance 300 ms). **Remaining issue**: p for real workloads (model output) is not measured (work on the operations
side).

### s121: the upper bound of swiglu epilogue fusion is 90 ms — closed, not implemented (2026-10-20, `568afa64`)

**Intent** — P1-4 "fuse swiglu into the epilogue of the gate+up GEMM" has the acceptance criterion "at least 0.3 s faster
in a cold A/B". We computed the prior question "can the upper bound reach 0.3 s" from the code, byte counts and
bandwidth. The target is prefill's MLP pass, but it stays here as the same "derive an upper bound by arithmetic and close
it" type as s101/s108.

**What changed** — Nothing (not a single byte changed in `crates/` or `kernels/`; neither a CUDA build nor a server start
was done). We only identified the path: `chain_mlp_prefill` = H2D of the normed x → gate_up GEMM → `dno_swiglu_int` →
down GEMM → 1 D2H of the layer output. The gate_up output is an interleaved `[m, 2ff]` f32 layout, the activation is
`[m, ff]` f32 and does not leave the device. swiglu runs 32 times per request.

**Stumble** — Both premises of the estimate were stale. (1) The upper bound of `rmsnorm` 413 ms is a value from before
s85, and s85 removed the `Ambient` parent scope that double-counted the whole layer block on the rmsnorm row. The norm is
host processing of 5,686 rows per layer and is not on the GPU, so a GEMM epilogue cannot reduce a single byte of it.
(2) The supporting row (gate_up_proj 2,283 ms / 64 calls / 14.4%) does not match any record elsewhere in this document
and is 1.65x larger than s70's value for the same bucket. We used it **only as a published upper-bound value**, and based
the verdict on the ratio to that row.

**What we learned / remaining issues** — 5,686 rows × 12,288 f32 = 279.5 MB per layer. The standalone swiglu touches reads
of 17.887 GB + writes of 8.943 GB = **26.830 GB**, and from the measured 64 calls 2,283 ms that is an **effective
11.75 GB/s**. Fusion can eliminate the 17.887 GB re-read of gate_up and up to the 8.943 GB write when the activation is
written into the gate stripe (the activation that the down GEMM reads cannot be eliminated). In bandwidth terms,
17.887 GB at the ~1.79 TB/s peak is **10 ms**, at a pessimistic 100 GB/s it is **179 ms**, and at 200 GB/s it is **89 ms**
→ upper bound **X = 90 ms**. **Verdict: closed. Upper bound 90 ms < 300 ms.** Moreover, "the activation round trip that
fusion was supposed to make worthwhile" was already eliminated in s70 (281 MB → 94 MB). The numeric class is R1, and
keeping the formula makes it bit-identical — **what blocks it is not numerics but byte counts**.

**What worked next** — The confirmation procedure "fusing removes neither a kernel launch nor a synchronization"
(1 launch on an asynchronous stream, with the wait only after the following GEMM) was used as is in s174/P16's launch
counting. **Not measured**: cold A/B (no plan, since the issue is closed).

### s122: refuted causal key range truncation executably and closed it (2026-10-20, `4dff66e7`)

**Intent** — s122's handoff instructed a one-file `.cu` edit on the basis that "starting the tile loop from
`jstart = ((pos0+i0)/PF_BK)*PF_BK` is numerically identical". Before starting the implementation we re-checked the
premise and found **the premise false**; applying it as-is would break attention in all prefill.
**Not a single byte of `.cu` was changed** (sha256 unchanged before and after).

**What changed** — `crates/denno-engine/tests/attn_prefill_jstart.rs` (5 tests) and replacement of the description in
`docs/TASKS.md`. It is a regression test so that the false premise cannot be derived again.

**Stumble** — The kernel's causal boundary is **per row**, not per block. `jend = min(t_total, pos0 + i0 + PF_BQ)` is an
upper bound matched to the block's **worst row**, and how far is actually read is decided by the mask `ja <= pos0 + ia`.
That is, row r reads up to `j <= pos0 + i0 + r`. `pos0 + i0` is the **upper bound** of row 0, not the **lower bound** of
the block. Furthermore, with a causal mask, "a tile where all rows are masked" only ever appears **at the tail**, so
raising the loop start cannot represent such a tile.

**What we learned / remaining issues** — CPU measurement (5 tests): the kernel's mask matches the CPU reference (499,095
(row, key) pairs). For the cold shape (n = 5,686, pos0 = 0) at block `i0 = 64`, **4,032 pairs** of the range to be
truncated `[0, 64)` are not masked (keys that should be reached). Across all 89 blocks, **15,983,616 pairs** disappear.
Scanning 291 blocks, **the number of all-masked tiles is 0** → jstart-style truncation does not exist. The magnitude of
the numeric difference in an fp32 toy with `dh = 8` is max |att_full − att_truncated| = **0.466**, 466 times R3
(absolute 1e-3) — not within the relaxation bracket, let alone bit identity. s118 had reached the same conclusion by
static analysis, but an incorrect premise (`pos0 + i0` and below is fully masked) remained in `docs/TASKS.md`, so it was
nearly re-executed. **With this, all candidates (1-4) of P1-2 are rejected with numbers.**

**What worked next** — The policy "do not end a rejection at static analysis; leave it as an executable test" was
established here. The fact that s118's estimate (invisible at R3) rested on a false premise is also shown by the order of
magnitude of the numbers. **Not measured**: GPU measurement, PTX regeneration and cold A/B are all unnecessary (source
unchanged).

### s157 / B20 stage 1: moving 27B q4's linear layers onto the device-resident chain (2026-10-21, `2701cf3`)

**Intent** — 27B q4 decode went through the per-op wrapper `maybe_gpu_linear_q4_into`. Because it takes host slices as
arguments, **1 step included 392 linear D2H transfers and synchronizations**, and GPU util stayed at 40%. We put q4 on the
same entry point that 9B int8 already uses (the chain), and remove the per-op round trips from the layer loop.

**What changed** — `chain_linear_q4_device` in `cuda.rs` (enqueues the existing `dno_gemv_q4_0c{C}` using only
device-resident buffers. The kernel, arguments, grid `(n.div_ceil(8*C), m, 1)` and block `(256,1,1)` are identical to the
per-op path, with no host slice and no synchronization). `chain_linear` becomes a dispatcher for int8 / q4, and returns
None unless the group is all-Q8 or all-Q4, leaving the caller on per-op (slower but not wrong). `DENNO_GPU_Q4_CHAIN` is
OFF only when `"0"` (so a typo does not silently change the path). `.cu` / `.ptx` unchanged.

**Stumble** — Numerical agreement cannot be produced on the CPU, so it had to be left to the GPU gate. We fixed in tests
only the boundaries of flag parsing (`"0"` / `""` / `"off"` / `" 0"` / unset) and the rejection of empty and mixed groups.
Also, entering the chain moves 27B's GDN scan / attention to the device versions of the functions that 9B uses (which do
not take host slices), so we had to make explicit that **the changed scope is not limited to the linear layers**.

**What we learned / remaining issues** — Measured (GPU gate, not predicted values): 27B q4 decode **75.89 → 32.42
ms/token** (fixed protocol, median of 3, temperature 0, max_tokens 200. TTFT 0.205 / 0.202 s, wall 6.689 / 6.685 s /
200 token. run1 is cold at 49.30). 9B int8 is **16.24 ms/token, unchanged** (gen2 3.248 s / 200 token). Anchors
**3/3 MATCH**, GPU_VERIFY PASS. In PROFILE (40 token), `d2h_src_linear` is **898 calls at 0.0 ms**, and `sl_lin` is
**0 call/step** (previously 392). `bench_all.sh` showed no regression (the +15% on m=300 gate_up is a known fluctuation).
Response verification was exactly identical across 2 runs.

**What worked next** — The acceptance metric was defined as "the number of per-op linear D2H within a step" (0 on the
chain path, baseline 392, target < 5), and it was fixed in `docs/DECODE_PATH.md` §6.1 that this is a different metric from
the total number of D2H issues (measured about 140/step, all asynchronous with 0 ms of host wait). The separation of
these 2 metrics became the basis for concluding "there is no fixed cost left" in the s160 audit.

### s160: the gate_up "fixed cost" of 27B q4 decode rejected by static audit (2026-10-22, no commit)

> Related commit: `2701cf3` (B20 stage 1). **0 lines of code change, 0 tests added.**

**Intent** — G1's premise was "1 gate_up call in serve has a fixed cost of about 29.5 ms (allocation, activation
quantization, H2D, synchronization, copy-out), and that is the main cause of decode's 32.42 ms/token". We check this
premise by static audit. **The reason it stays in this volume** is that the rejection became not "do not implement" but a
warning that "misreading a measurement makes you chase a fixed cost that does not exist".

**What changed** — Nothing. The evidence was left in `REPORT_G1.md` (gitignored). The path was identified as `forward`
n==1 → `forward_decode_chain` → `chain_linear` → `chain_linear_q4_device`, and the issue is **only 1 kernel**. Only the
per-op path, unit tests and bench call `maybe_gpu_linear_q4_into`, and decode's chain does not reach it. The profile
bucket name (`sl_lay_ffn_gate` exists only inside the chain) was the evidence for the path.

**Stumble** — The premise is stale. Of the 6 stages of the fixed cost, **5 do not exist structurally**: allocation is
grow-only and 0 times in steady state (`chain_q35_step_reserve*` at the start of the step takes the maximum first),
activation quantization is 0 because the q4 GEMV reads the activation as f32, H2D is 0 because CH_H is device-resident,
issuing is once, synchronization is 0 times (the step's only sync is the logits D2H at the end), and copy-out is 0 times
(the output stays in CH_GA and is read by swiglu). The 4 priorities in the HANDOFF had also already been implemented in
`2701cf3`.

**What we learned / remaining issues** — 29.5 ms/call does not describe the current path: gate_up is 64 layers = 64 calls
per token, so 29.5 ms/call would be **1,888 ms/token**, which conflicts with the measured 32.42 by a factor of **58**.
The m=21.5 in the record is prefill's average m (decode is m=1), and the `lin_sync_wait` 89% was the wait on
`cuCtxSynchronize` = a value waiting for **the kernel execution itself** (not a fixed cost). Comparing with the weight
byte share, gate_up is 111.4 MB per layer × 64 layers = 7.13 GB, **42.7%** of the 16.7 GB of weights on the card, and the
measured share of 28.4% is **smaller** than the byte share (it flows faster than the step average). Candidates for the
28.4% are (E1) the first-touch weight H2D falling inside the span (7.13 GB / 41 step = 174 MB/step, 14.5 ms/step at H2D
12 GB/s), or (E2) the issue blocking on device backpressure so that the span absorbs GPU execution time, and **neither
can be eliminated on the Rust side**. The discrimination procedure was left in the REPORT (not carried out).

**What worked next** — The line from s49/s54/s101 of "do not read a bucket's ms as the op's cost" was completed here.
Adding measurement spans was shelved (5 of the 6 stages are 0, and doing so would mean rewriting the sum of the table's
exclusive entries and the COVERAGE assertion, contradicting "do not redesign"). The only homework left is that if
measurements of the 6 stages are needed, design them in a separate task starting from how ticks are held.

### s174 / P16: folding GDN's beta|alpha fp32 matvec into 1 launch (2026-10-23, `c3e53517`)

**Intent** — The launches of 1 decode token are 257 of `dno_gemv_q4_0c4` and 129 of `dno_add_rmsnorm`. After statically
determining "merge adjacent same-shape gemvs within a layer into 1 launch", **q4 GEMV cannot be folded**. There is only 1
foldable pair, and it was the GDN layer's beta|alpha (2 launches of `dno_matvec_f32`).

**What changed** — `GdnLayer::ba_w` now holds a row-concatenated `[2*nv, e]` built only once at load time (row order [all
beta rows | all alpha rows]), and `cuda.rs` gained `Q35_TAG_BA` and `chain_q35_ctl_f32_ba`. The same `dno_matvec_f32`,
the same argument order, the same block, with only grid.x going from n → 2n. `chain_q35_gdn_scan` uses 1 launch when
ba_w exists and the shapes match, and the previous 2 launches otherwise (fail closed).

**Stumble** — Confirming why they cannot be folded was the main work. A layer's 4 launches (qkv → o → gate_up → down) are
**serial** on the residual stream, and between every adjacent pair there is always 1 dependent kernel (between qkv|o =
scan/attn, between o|gate_up = the residual addition of `dno_add_rmsnorm`, between gate_up|down = swiglu, between layer i's
down and layer i+1's qkv = `dno_add_rmsnorm`). A kernel takes 1 input pointer, so only pairs with "the same input slot and
the same k" can be made into 1 launch. q|k|v was already consolidated into 1 launch in s166 (337 → 257).

**What we learned / remaining issues** — The launch count for 1 decode token goes `dno_matvec_f32` **96 → 48**, all
kernels **658 → 610 (−7.3%)**, and per request (150 token) 98,700 → 91,500. `dno_gemv_q4_0c4` 257 and
`dno_add_rmsnorm` 129 are unchanged. Numerically, because the reduction is row-independent and sequential, concatenating
rows leaves each row's accumulation order unchanged = **bit-identical** (fixed on the CPU side by the test
`f32_beta_alpha_concat_rows_match_parts`). VRAM is unchanged; the host grows by the concatenated copy, +94 MB (27B).
**Predicted ms**: 48 fewer launches × 1 to 2 us = 0.05 to 0.10 ms/token, measured 0.00 to 0.10 ms
(17.49–17.51 → 17.45–17.51 ms/token, ≤0.6%) — **not measured on hardware**. Tests are 321 passed / 37 ignored.

**What worked next** — The conclusion was reached that "the room for reducing launches is statically exhausted by
counting the dependencies of the residual stream". Reducing 257 further requires fusing residual addition + rmsnorm +
gemv (a `.cu` change), which is an extension of the direction whose upper bound was computed at 90 ms in s121.
**Remaining issue**: the hardware ms/token, the 3 anchors and nsys have not been carried out.

---

### P-MOE-F12..F17: making Flash-Next (MoE) decode 11.8x faster with per-type fused GEMV and parallelization (2026-10-27, `b43d66e`..`a469224`)

**Goal** — The newly supported MoE model (`qwen4exp`, 48 layers, 512 experts, top-10) took
**6827 ms** for a one-token forward, 17x slower than the 390 ms/token CPU reference
implementation. Most of it was the MoE expert GEMV, which expanded one row at a time to f32
and then did a sequential dot product (no SIMD at all).

**What went in (measured at each step)**

| step | commit | change | forward |
|---|---|---|---|
| start | — | — | 6827 ms |
| F12 | `b43d66e` | Fused AVX2+FMA GEMV for Q4_K (FMA straight from the 144 B block) | 5214 ms |
| F13 | `5828846` | Fused GEMV for Q8_0 + in-layer phase profiling (`DENNO_Q4_PROFILE=1`) | 1755 ms |
| F14 | `58ccd86` | Fused GEMV for Q5_1 | 1036 ms |
| F15 | `74e41d3` | Parallelize the GEMV row direction (feature `par`, on by default) | 601 ms |
| F16 | `a42e144` | Parallelize the GDN delta rule over heads | **579 ms** |

**Why it worked, and why the first step did not**

- The Q4_K kernel alone gave only 1.3x. The reason is that this model's quantization is
  **mixed**: the attention projections (`attn_qkv` / `attn_gate` / `attn_output`) and the
  shared expert are **Q8_0**, `ffn_down_exps` is **Q5_1** (43 of 48 layers), and only
  `ffn_gate_exps` / `ffn_up_exps` are Q4_K. Adding one kernel per type took 3 steps to 6.5x,
  and parallelization added another 1.8x.
- The kernel alone went from 2281 to 164 ns/row for Q4_K. Maximum difference from an f64
  truth was 1.05e-5 for the fused Q8_0 path (6.10e-5 scalar) and 2.38e-7 for the fused Q5_1
  path (2.15e-6 scalar): the fused versions are **more** accurate. For Q5_1, `w = d*q + m`
  lets us split `dot(w, x) = d*dot(q, x) + m*sum(x)`.
- Row parallelism keeps every element's accumulation order (rows are independent), so it
  stayed bit-identical and gave 1.7x on 16 threads. Head parallelism is the same story
  (heads are independent). The 4-token greedy sequence `[271, 248068, 198, 760]` never moved.
- The `MADV_WILLNEED` prefetch had no effect (5197 vs 5214 ms).

**The remaining bottleneck** — After parallelization the per-layer split is attn 3.72 ms
(gdn 1.29 of it) / **moe 4.63 ms** / mix 1.50 ms. The MoE path reads 10.1 MB per layer at
2.2 GB/s, which is about the NVMe sequential speed (measured 2.8 GB/s; 26 GB/s for ranges
already in page cache). The model is 104 GB against 60 GB of RAM, so the page cache cannot
hold it and reads fall through to disk every time. It is **I/O bound, not compute bound**.

**Side product** — Running `cargo test --release --all-targets` on a clean clone surfaced an
existing defect: `decode_greedy_stop` did not append the stop token to `tokens` (it also
failed at `0f5610c`). The stop branch broke out before calling `step()`, so F17 now runs the
forward and then stops.

**Not verified** — The 27B (Q4_0) wide 8-prompt md5 and the GPU anchors have not been
re-measured because the production server is down. Row and head parallelism do not change
accumulation order, so the numbers should be unchanged, but this gets confirmed after restart.

---

### P-MOE-G1..G5: wiring ExpertCache and QSA, and parallelizing the expert loop with rayon (2026-10-27, `6d3fc13`..`3dc3612`)

All numbers are CPU, real weights, 1 forward of 5 tokens, after warm-up, median of 3 alternating rounds.

**Goal** — F17's 579 ms was I/O-bound rather than compute-bound, with a per-layer breakdown of attn 3.72 ms
(of which gdn 1.29) / moe 4.63 ms / mix 1.50 ms. The model is 104 GB against only 60 GB of RAM, so the expert
reads fall to disk every time. We tried 3 directions: (G1) lift that with an explicit cache, (G2) reduce the
cell count by selecting keys with QSA, (G5) parallelize the computation itself.

**What went in (measured at each step)**

| step | commit | change | forward |
|---|---|---|---|
| right after F17 | — | — | 579 ms |
| G1 + G2 | `6d3fc13` / `e4e5547` | wire ExpertCache and the QSA indexer into the paths | 355 ms (attn 93 / moe 185 / mix 56) |
| G5 | `3dc3612` | compute the selected experts' contributions h_e with rayon | **278 ms** (attn 84 / moe 121 / mix 56) |

- G1's ExpertCache is `DENNO_EXPERT_CACHE_MB`. It **lost badly to parallel mmap in measurement, so it is
  sealed off at a default of 0** (disabled in the default configuration).
- G2's QSA is enabled with `DENNO_QSA=1`. It only passes the selected cell list to `attend_visible`; the dense
  path goes through `attend_impl(None)` and keeps the same loop expressions and the same rounding order as
  before G2. The output is exactly identical to dense.
- G5 is `DENNO_MOE_PAR` (default 1). Only the mutually independent h_e are parallelized, and the weighted
  accumulation stays sequential in index order, so the addition order is unchanged = **bit-identical**
  (all `to_bits()` equal between `par` 0 and 1 on a synthetic GGUF).

**Why it worked, and why the first step did not** — Most of the reduction is in G5's moe, 185 → 121 ms (−64 ms).
The 9-token greedy run goes 5769 → 4515 ms. Expert contributions are independent of each other, so raising the
parallelism changes no element's sequence of operations and the numerical contract did not have to be relaxed.
attn also dropped from 93 to 84 ms, but G5 touches only the moe expert loop, so that difference cannot be
explained by G5 (the per-layer breakdown may be measured differently at each step; the material gives no
explanation).

**Numerical agreement** — 16/16 tokens agree with the reference implementation (llama.cpp). GREEDY4 is
`[271, 248068, 198, 760]`. In the F series only the first 4 tokens agreed and it diverged at the 9th, so this
agreement is for a configuration with QSA enabled (estimate).

**Remaining bottleneck** — The 278 ms breakdown is attn 84 / moe 121 / mix 56, and moe is still the largest.
The 5 further attempts to cut it (G6 to G10) and 1 GPU-side attempt (H1) were all rejected by measurement
(see the rejected catalog below).

## Rejected approaches

Judgments that looked reasonable at the time but were discarded later. We place only references to the corresponding
entries in the body; the numbers are taken from the body and from the records in `docs/TASKS.md`.

### s118: causal key range truncation — static analysis cannot prove numerical invariance (no commit)
- What we tried: raise the tile-loop start of `dno_attention_prefill` to `jstart` so that keys before the causal boundary are not read.
- Measured value: the magnitude of the difference is `2^-23` relative in cases where the dot product with `d=255` cancels. Invisible at R3 (absolute 1e-3), visible in the anchor md5 (compared to 1e-7).
- Why we dropped it: it can be read from the code that the mask path adds `+0.0f`, but "the same bits without it" is a different proposition that static analysis cannot prove. We do not ship what cannot be proven.
- Record: body s118, related commit `3eab3629` (docs only).

### s119: speculative decoding variants A / A-lite (no commit)
- What we tried: design and comparison of 3 speculation variants using multi-token forward (A = verify in place / A-lite = use the existing n≧2 path as a 2-token prefill / B = n-gram).
- Measured value: variant A-lite is **computed at 18.3 ms** per step (`λ = 1.115`) with a break-even acceptance rate of **0.885**. Variant A has no means of comparison because row 1's logits themselves change from the sequential case.
- Why we dropped it: variant A cannot be bit-identical, and variant A-lite does not reach break-even for speed purposes. The surviving variant B was also closed in s120.
- Record: body s119, related commit `623359bc` (design document only).

### s120: n-gram speculation variant B (rejected by measurement, `506b515a`)
- What we tried: measure the rate p at which the continuation matches when the preceding 12-gram context reappears, on a proxy corpus and a synthetic corpus.
- Measured value: p_1 = 0.2984 on proxy words (368,831 opportunities), 0.1265 on proxy bytes, 0.1156 on synthetic words. The effective `(2S + C_d)/(1 + p_1)` with S = 16.4 ms and p_1 = 0.2984 is **25.3 ms/token = +54% over sequential**.
- Why we dropped it: the winning condition is `p_1 > 1 + 2 C_d / S` = 1.00001, which no value of p overturns. Since the break-even formula does not depend on p, measuring p does not change the verdict.
- Record: body s120, `scripts/ngram_oracle.py` and 29 tests. p for real workloads is not measured.

### s121: swiglu epilogue fusion (rejected by arithmetic, `568afa64`)
- What we tried: upper-bound calculation of whether applying swiglu in the epilogue of the gate+up GEMM can eliminate the activation re-read.
- Measured value: what can be eliminated is the 17.887 GB re-read (+ 8.943 GB of writes into the stripe). 10 ms at peak bandwidth, 89 ms at 200 GB/s, 179 ms at 100 GB/s → upper bound **X = 90 ms**. Acceptance is a cold A/B of 0.3 s.
- Why we dropped it: upper bound 90 ms < acceptance 300 ms. Moreover, the activation round trip that motivated fusion had already been reduced from 281 MB to 94 MB in s70.
- Record: body s121. Not a byte was implemented, and no cold A/B was carried out (closed issue).

### s122: executable refutation of causal key range truncation (`4dff66e7`)
- What we tried: run s118's idea on a CPU model and count the (row, key) pairs that the truncation drops.
- Measured value: for the cold shape (n = 5,686, pos0 = 0) at block i0 = 64, **4,032 pairs** of the skipped range `[0, 64)` are not masked. 15,983,616 pairs across all 89 blocks. The difference in the fp32 toy is max 0.466, 466 times R3.
- Why we dropped it: the premise ("below `pos0 + i0` is fully masked") is false. With a causal mask, tiles that are fully masked only ever appear at the tail, so raising the loop start cannot represent them.
- Record: body s122, `crates/denno-engine/tests/attn_prefill_jstart.rs` (5 tests). Not a byte of `.cu` was changed.

### P1-2 candidates 1 to 3 (all rejected with numbers, `docs/TASKS.md` §P1-2)
- Candidate 1: `__ldg` for K/V reads (s112 `b5f1f37` + `78e7d08`). Measured **-0.9%** and reverted (`ac73729`). The bucket is not DRAM-bound, so the cache hint does not help.
- Candidate 2: models of traffic, smem and occupancy for (BQ 64, BK 16) / (128, 8) / (128, 16) / (32, 16) / BQ 256 / the kv head loop (s116, no code change). None of the shapes reaches the 0.3 s acceptance. K/V reads are 273.2 GB per request against an actual 186 MB, a factor of 1,466, but the measurements are 147 GB/s in pass 1 and 433 GB/s in pass 2, so it is not DRAM-bound, and the range of 1 kv head fits in the 126 MB L2. BQ 128 / BK 8 makes the bytes 0.522x but also halves the warps (149.6 KiB, 1 block/SM), and BQ 256 (283.1 KiB) exceeds the 227 KiB smem limit.
- Candidate 3: kernel rewrite (per-kv-head transposed tiles / moving online softmax to a flash-style tile, s117, design only). (a) exceeds the limits (227 KiB, 255 regs) with 283.1 KiB smem and 256 floats/thread; (b) is within budget (124.5 KiB, about 64 regs) but amounts to only 14.3–45.3 ms = 1.5–4.8% of the 942 ms bucket.
- Candidate 4: causal key range truncation → the s118 / s122 sections above. With this, P1-2 is closed with all candidates rejected with numbers.

### s160: chasing the gate_up "fixed cost" of 27B q4 decode (no commit)
- What we tried: static audit of the premise "1 gate_up call has a fixed cost of about 29.5 ms and is the main cause of 32.42 ms/token".
- Measured value: 29.5 ms/call × 64 layers = 1,888 ms/token, which conflicts with the measured 32.42 by a factor of **58**. Of the 6 stages of the fixed cost, 5 (allocation, activation quantization, H2D, issuing, synchronization, copy-out) are structurally 0 times. Against gate_up's byte share of 42.7%, the measured share is 28.4%, smaller than the byte share.
- Why we dropped it: the m=21.5 in the record is prefill's average m and decode is m=1. `lin_sync_wait` 89% is a value waiting for the kernel execution itself, not a fixed cost. We rejected the way of reading the measurement itself, which chases a fixed cost that does not exist.
- Record: body s160, related commit `2701cf3`.

### P-MOE-G3 / G4: widening to AVX-512 (2026-10-27, not merged)
- What we tried: widen the Q4_K / Q8_0 / Q5_1 GEMV to 512 bits and process 2 rows in 1 instruction. Row r keeps its lanes exactly where the AVX2 version puts them, so the values stay bit-identical to AVX2.
- Measured value: **Q4_K is decode-bound and shows no gain. Q8_0 gains −12 ms in mix only.** Per kernel alone (n_in = 2560, n = 640) Q8_0 goes 253.6 → 133.1 ns/row (1.91x) and Q5_1 goes 220.8 → 192.8 ns/row (1.14x).
- Why we dropped it: only mix's 12 ms moves in the forward, and the main attn / moe buckets do not move. As long as moe is I/O-bound, widening the kernel does not reduce the wait for rows to arrive (estimate).
- Record: the work survives only as a WIP commit (`909cb18`) and was not merged. The values are `to_bits()` identical to the AVX2 version.

### P-MOE-G6: keeping dense weights resident in RAM (2026-10-27, not merged)
- What we tried: hold every `GgufLinear` other than the experts in an owned Vec instead of mmap, to avoid page eviction (`DENNO_PIN_DENSE=1`).
- Measured value: forward is 363–371 ms against 349–372 ms, **no difference**.
- Why we dropped it: dense is about 1 GB, and keeping it resident does not make the per-token reads faster. The difference drowns in the overlap of the ranges.
- Record: unmerged work commit `4fc94da`. The byte ranges are identical on and off (a copy of the same range), and `to_bits()` equality is fixed by tests.

### P-MOE-G7: flattening expert × row-block parallelism (no commit)
- What we tried: flatten G5's expert parallelism and F15's row parallelism from 2 levels into 1, evening out the load difference between experts.
- Measured value: moe **124 → 110 ms**. But the 9-token greedy run is **4515 against 4438 ms**, a difference within the noise.
- Why we dropped it: the per-layer moe figure improves by 14 ms but does not show up in the per-token total. Acceptance is on total time, so it is not taken.
- Record: no commit.

### P-MOE-G8: head parallelism for attention (2026-10-27, not merged)
- What we tried: parallelize the `attend` head loop with rayon (`DENNO_ATTN_PAR`). Heads are independent, so whichever thread takes one, each element's sequence of products and sums is unchanged.
- Measured value: attn **90 → 93 ms** (slightly worse).
- Why we dropped it: attn is already filled by GEMV row parallelism, and against 24 heads the work for 1 head is too small, so the fork-join cost wins.
- Record: unmerged work commit `089dbea`. `to_bits()` identical to the sequential path, and the parallel branch actually running is checked by a counter.

### P-MOE-G9: madvise read-ahead for the selected experts (no commit)
- What we tried: read the ranges of the selected experts ahead of time with `MADV_WILLNEED`.
- Measured value: moe **123 against 123 ms**.
- Why we dropped it: no difference. Same conclusion as `MADV_WILLNEED` in the F series (5197 against 5214 ms).
- Record: no commit.

### P-MOE-G10: software prefetch in the GEMV row loop (2026-10-27, not merged)
- What we tried: issue `_mm_prefetch` on the head of row r+3 just before processing row r (`DENNO_GEMV_PF`). The PF-disabled monomorphization keeps not 1 prefetch instruction.
- Measured value: moe **125 against 124 ms**.
- Why we dropped it: no difference. The row access is fully sequential, so the hardware prefetcher is already doing the job (estimate).
- Record: unmerged work commit `f9df39d`. `to_bits()` equality on and off and the number of prefetches issued (off = 0, on = `n_out - PF_DIST`) are fixed by tests.

### H1: reusing GPU scratch (no commit)
- What we tried: reuse the GPU-side scratch buffers instead of allocating them again.
- Measured value: **302 → 313 ms** (there is no record of what this total covers).
- Why we dropped it: it got worse. The material gives no breakdown of the regression.
- Record: no commit.
