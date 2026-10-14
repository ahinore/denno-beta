# denno-beta Development History

> Architecture details: see the [Architecture Guide](ARCHITECTURE.en.md).
> This note is the **development log**. New features are appended at the end in this format:
>
> **Entry format**: under `### Pn: Feature (date, commit)`, record **Implementation** / **Tests** / **Profile** / **Next up**. Completed items are reflected in the candidate list under "Next up".

**Updated**: 2026-10-13
**Repository**: https://github.com/ahinore/denno-beta (main, `3ff6175` synced)

---

## 1. Development Environment and Deployment

```
[dev machine: x86_64 Linux (everything inside Docker)]
├─ denno-beta container (port 8080)            … LLM API server (production)
│    └─ model cache: ai_work/denno-beta/data/.cache
└─ dev container … dev agent + dev denno (8090) / denno-view (8093)
     └─ /work/denno-beta (= host ai_work/denno-beta, git-managed)

Dev flow: agent implements → cargo test → exactness check → commit
        → GitHub sync → CI (build+test+exactness)
        → production container docker compose build && up -d → health/generation tests
```

- CI: `.github/workflows/ci.yml` — build+test (push/PR), exactness (push only, downloads real models). **All green on x86-64-v3**
- GitHub push: currently synced via the Windows repository checkout. Container-direct push awaits deploy key registration
- Reference output (exactness): "1+1は?" → `1+1は1です。これは数学的定理で、1と1を加えることによって得られます。` (seed 42, temp 0, fp32). Confirmed identical on the CI x86 runner

## 2. Timeline (early → P2: summary)

| Stage | Content |
|---|---|
| Initial implementation | single binary (cli/engine/server), GPT-2/Qwen2.5-0.5B, Anthropic API compatible + SSE, exact decoding (seed 42) |
| Optimization #1 | `target-cpu=native`, attention scalar loop → sgemm, decode matvec output-split parallelism → prefill 88s→24s, decode 1s/token→0.1s |
| P1 | KV cache + LRU prefix cache (single entry) → subsequent prefills ≈0 (1519→11 tokens) |
| P2 | multi-entry LRU prefix cache (largest-LCP selection, 32k token budget) → `reuse 47/48` even in alternating sessions |

## 3. Phase Records

### P3: int8 weight-only quantization ✅ (2026-10-11, commit `89aae74`)

- **Implementation**: per-channel scale quantization + dequant-in-matvec kernel (new `quant.rs`, 430 lines). fp32 default fully preserved; active only with `--quant int8`
- **Tests**: 77 tests green. fp32 exactness unchanged; int8 identical across 3 runs (deterministic)
- **Profile**: speed flat (14.4 → 13.9 tok/s). Dequantization overhead cancels the bandwidth savings; the value is halved memory (weights ~1GB → ~0.5GB)
- **Next**: SIMD int8 dot product (→ done in P5), continuous batching (→ P4)

### P4: Continuous batching ✅ (2026-10-11, commits `f65281f` → `4e9058a` → `e3ad3ca`)

- **Implementation**: request scheduler (`scheduler.rs`) — bounded queue (503 when full) + engine stepper advanced by `step_chat` (1 forward). New prefills interrupt decode; multiple sessions rotate via KV suspend/resume (`MAX_ACTIVE_SESSIONS=4`)
- **Tests**: 86 tests green. **Each output is bit-identical to a solo run even under interleaving** (verified with KvSensitiveModel). CI (x86) 3 consecutive full greens
- **Profile**: none taken. Instead found a real bug: the worker held the mutex almost continuously, starving the HTTP handler's `engine.lock()` by hundreds of ms → fixed by caching the model name in AppState
- **Next**: int8 SIMD kernel (→ P5), true parallel decode (→ P6 candidate)

### P5: int8 SIMD kernel ✅ (2026-10-12, commit `eec5624`)

- **Implementation**: redesigned P3's "dequantize weights and multiply in fp32" into **"dynamic activation quantization + true i8×i8 integer dot product"**. aarch64 NEON `vdotq_s32` / x86 AVX2 `_mm256_madd_epi16`, switched by cfg
- **Tests**: 91 tests green (+5). i32 accumulation's associativity makes SIMD/scalar bit-identical (property test). Also verified on the CI x86 AVX2 path. Deterministic within int8 (3 runs, same md5), fp32 fully unchanged
- **Profile**: kernel-only 5.4–12.9× (lm_head 5.39ms→1.0ms), e2e decode **~14.4 → ~51 tok/s**
- **Next**: true parallel decode (P6)

### P6: Continuous batching (batched parallel decode) ✅ (2026-10-12, commit `52f5742`)

- **Implementation**: KV cache changed from single to a **slot pool** (`caches: Vec<KvCache>`, `DENNO_BATCH_SLOTS` default 4). Sessions stay resident in their own slots until completion (old suspend/restore rotation removed), and one decode step of all resident sessions is combined into **a single `forward_batch`**. Kernel layer: new multi-column matvec `matvec_rows_bt`/`matvec_rows_bt_i8` — **each element's k-direction accumulation order exactly matches a standalone matvec** (no sgemm, which would change accumulation order = bit-identical even in fp32). rayon row-block parallelism shares raw pointers via a `SendPtr::write` method (an edition-2021 disjoint-closure-capture workaround). Scheduler converted to continuous batching (admit while slots free, prefill-priority interleaving, batched decode `step_chat_batch`, 503 when full)
- **Tests**: **106 tests green** (engine 99 + server 7). Contract tests that `forward_batch` matches solo decode bit-for-bit for B∈{1,2,3} over multiple steps, alternating-session exactness, all passing on CI x86-64 too. **On hardware**: 0.5B with 4 concurrent sessions, fp32/int8 both solo==batch fully identical (including repeated runs), reference output unchanged
- **Profile** (0.5B, 4 requests × 256 cap, same machine): sequential 604 tokens/20.6s → batch 604 tokens/14.8s. Aggregate throughput **fp32 13.9→18.7 tok/s (1.35×), int8 38.8→66.2 tok/s (1.7×)**. 131 steps consumed 272 tokens across 4 sessions = batching fully working. **At b=4 the regime shifts from bandwidth-bound to compute-bound** (fp32: 72ms→110ms per step, int8: 27ms→36ms) = the weight stream runs once but FLOPs are 4×. Next breakthrough would be order-preserving SIMD widening in the k direction
- **Size dependence measured** (3B int8, b=2, 128 cap): solo 12.8 tok/s → batch aggregate 20.5 tok/s (**1.6×**). Batch step time almost unchanged (solo 78ms → 81ms at b=2) = **3B still weight-bandwidth-bound** (weights 3GB/step vs relatively small FLOPs). Confirmed the prediction "larger models benefit from batching more directly". 3B also verified solo==batch bit-identical
- **Design note**: `release_session` keeps KV rows and `cached_prompts[slot]` in place so the next session with the same prefix can reuse them zero-copy with truncate only (reproducing the old single-cache behavior). Eviction goes to the prefix store (`contains_prompt` avoids duplicates). Prefill processes the remaining prompt in one forward (splitting would change GEMM accumulation order = exactness NG)
- **Next**: sampler optimization (→ P8, completed before P6 work began)

### P7: Large Qwen model support ✅ (2026-10-12, commit `9fb18ab`)

- **Implementation**: sharded safetensors (`model.safetensors.index.json` + download all shards / tensor name resolution, new `WeightStore`), untied lm_head (`tie_word_embeddings: false`)
- **Tests**: 93 tests green (+2). 0.5B exactness fully unchanged. **0.5B/1.5B/3B/7B all answered normally ("1+1は2です。" etc.)**
- **Profile** (by size, 20-core CPU):

| Model | fp32 | int8 | Notes |
|---|---|---|---|
| 0.5B | 14.4 tok/s | ~51 tok/s | production |
| 1.5B | — | — | runs unchanged on the existing binary |
| 3B | ~2 tok/s | ~4.7 tok/s (2.4×) | sharded checkpoint + untied |
| 7B | ~1.2 tok/s | ~2.1 tok/s (1.8×) | weights 28GB, RAM 121GB with room to spare |

- **Next**: profiler introduction for bottleneck visibility (→ P7.5)

### P7.5: Profiler + bottleneck visualization ✅ (2026-10-12, commit `40e209d`)

- **Implementation**: lightweight self-profiler `prof.rs`. Enabled with `DENNO_PROFILE=1`; on request completion logs accumulated time per phase at INFO. Zero overhead when disabled, never touches the numeric path
- **Tests**: 93 tests still green. exactness reference output unchanged (profiler disabled). CI ✅, deployed
- **Profile**: below (later measurements appended here in this format)

#### Profiling method constraints

Inside Docker, `perf` (denied by perf_event_paranoid=4), `valgrind/callgrind` (SIGILL on aarch64 native code), and `gdb` (ptrace forbidden) are all unavailable → **in-code measurement (self-instrumentation) is the only proper way**

#### Measurement conditions

Qwen2.5-0.5B, one long generation request (cumulative incl. prefill).

**fp32 decode time breakdown** (800 tokens total, 74.8s overall):

```mermaid
pie title fp32 decode time breakdown (74.8s total)
    "gate_up_proj (MLP)" : 30.8
    "down_proj (MLP)" : 19.0
    "qkv_proj (Attention)" : 16.7
    "o_proj (Attention)" : 12.9
    "attn_core (softmax etc.)" : 10.1
    "lm_head" : 5.7
    "sampler" : 3.7
    "others (swiglu/rope/rmsnorm/detok)" : 1.1
```

**int8 decode time breakdown** (461 tokens to EOS, 16.5s overall):

```mermaid
pie title int8 decode time breakdown (16.5s total)
    "gate_up_proj (MLP)" : 22.2
    "qkv_proj (Attention)" : 17.6
    "attn_core (softmax etc.)" : 17.6
    "o_proj (Attention)" : 14.3
    "down_proj (MLP)" : 12.5
    "sampler" : 9.9
    "lm_head" : 3.5
    "others (swiglu/rope/rmsnorm/detok)" : 2.4
```

#### What it tells us

1. **Linear layers (matvec) dominate** — ~80% in fp32. Decode is weight-bandwidth-bound as expected. int8+SIMD compresses this to ~70% and yields ~4.5× speed
2. **attn_core share grows in int8** (10%→18%) — it surfaces relatively as linear layers speed up. KV cache reads + scales/softmax. Grows further with long contexts
3. **sampler is a fixed ~3.5ms/step cost** — 3.7% in fp32, ~10% in int8. Softmax + top-p sort over the full 152k vocabulary is the bottleneck
4. swiglu/rope/rmsnorm/embed total ~1% — numerically confirmed "no optimization needed"

- **Next**: sampler optimization (top candidate) / resume P6 (true parallel decode)

### P8: Sampler optimization ✅ (2026-10-12, commit `119dd92`)

- **Implementation**: responding to the P7.5 profile (fixed ~3.5ms/step sampler cost), `sampler.rs` fully rewritten. top_k=`select_nth_unstable`+partial sort (with a reference-implementation test that ties break bit-identically to the old code) / top_p-only = compute over the top-2048 subset with fallback to the old full path on boundary hits / plain temperature (production default) = sort-free two-pass O(n) + scratch reuse. **The greedy path is completely unchanged** (temperature=0 goes straight to argmax)
- **Tests**: 98 tests green (+5 sampler tests including bit-identical reference checks). **Declared as a spec change**: the temperature path's RNG→token mapping differs from the old implementation (greedy unchanged). 0.5B greedy reference output unchanged, CI ✅
- **Profile**: sampler 3.5ms→**0.66ms/step (5.3×)**. Share 3.7%→0.7% (fp32), 9.9%→2% (int8)
- **Next**: P6 continuous batching (→ completed as P6)

### P9: fp32 batch kernel SIMD (order-preserving lane parallelism) ✅ (2026-10-12, commit `1bbffff`)

- **Implementation**: SIMDized the fp32 batch kernel `matvec_rows_bt` after P6 found "b=4 shifts to compute-bound". **Lanes run only along i (4 sessions in the batch) and j (4 output channels for ILP), never along k** — each output element's k accumulation stays strictly sequential (only separate multiply and add instructions, no FMA = rounding kept separate) → bit-identical to solo decode (`matvec_bt`) is preserved. Activations are transposed to `[k][mp]` with zero padding to keep lane loads contiguous (m*k is at most tens of KB, transposition cost negligible). aarch64 NEON / x86 SSE (`_mm_mul_ps`) dual implementations + scalar fallback for other targets
- **Tests**: 106 tests green. Bit-identity test shapes extended (m∈{1..8}, k∈{1,4,65,896}, with remainders). CI exactness ✅ on the x86 SSE path too. **On hardware**: fp32/int8 both 4 concurrent sessions == solo fully identical, reference output unchanged
- **Profile**: below

#### Profile

**① Server per-op** (`DENNO_PROFILE=1`, 0.5B fp32, solo/batch measured in separate sessions)

| op | solo: ms (calls) | solo share | batch: ms (calls) | batch share |
|---|---:|---:|---:|---:|
| gate_up_proj | 1,642 (3,600) | 33.8% | 5,316 (6,432) | 37.0% |
| down_proj | 1,196 (1,800) | 24.7% | 3,152 (3,216) | 22.0% |
| qkv_proj | 828 (5,400) | 17.0% | 2,161 (9,648) | 15.0% |
| o_proj | 502 (1,800) | 10.3% | 1,358 (3,216) | 9.5% |
| lm_head | 405 (75) | 8.3% | 1,256 (134) | 8.7% |
| attn_core | 187 (1,800) | 3.8% | 785 (3,216) | 5.5% |
| rope/sampler/rmsnorm/detok | 42 | 0.9% | 142 | 2.3% |
| **TOTAL** | **4,854** | 100% | **14,362** | 100% |

- **solo**: 75tok/4.88s = **15.4 tok/s** (64.7ms/step), linear layers total **94.2%**
- **batch**: 279tok/130 steps, average batch width **2.15** (early-finishing sessions with outputs of 45–83 tokens), aggregate **19.4 tok/s** (21.8 without the profiler)
- **Per-token linear-layer cost 61.0→47.5ms (1.28×)**. Attention unchanged per token (2.5→2.8ms) = wider within a step but the same per token
- The in-server batch linear layers don't reach the microbenchmark bandwidth floor (width 4 = solo step time) → attributed to the drop in average width (2.15) + memory-system contention in the real server (activation/KV interfering with the weight stream)

```mermaid
pie showData title P9 fp32 batch (4x256) per-op time
    "gate_up_proj" : 5316
    "down_proj" : 3152
    "qkv_proj" : 2161
    "o_proj" : 1358
    "lm_head" : 1256
    "attn_core" : 785
    "others" : 142
```

**② Kernel-only microbenchmark** (0.5B shapes, m=4): gate_up(k=896, n=4864) **3.92×** / down(k=4864, n=896) **5.86×** (vs 4× standalone `matvec_bt`). Shipped as an `#[ignore]` test `bench_matvec_rows_bt_m4` (`cargo test --release bench_matvec -- --ignored --nocapture`)

- **e2e limiting factors identified** (profile analysis): the batch aggregate factor (fp32 1.36×) almost matches the per-token linear improvement (1.28×) — linear layers are no longer the main bottleneck. In theory width 4 could cut per-token linear cost to ~1/4 (61→~15ms), so **the remaining bottlenecks are ① average batch width drop (completion-time skew, avg 2.15), ② in-server memory-system contention, ③ attention/prefill (unchanged per token, doesn't grow with width)**. Next levers: session-lifetime flattening + attention batching
- **Next**: attention batching (P10 candidate) / b=4 measurements for 3B/7B

### P10: Attention parallelization (session × KV head) ✅ (2026-10-12, commit `4e2428f`)

- **Implementation**: `forward_batch`'s attention parallelized with rayon. Task granularity is **(session i × KV head kvh)** (up to 32 tasks at 0.5B); each task owns a non-overlapping slice of `att`. The op sequence inside a task is exactly the solo decode (n==1) path (sgemm_acc→softmax→matmul, same shapes) and there is no numeric interaction between sessions/KV heads, so **bit-identity is trivially preserved**. KV reads use the new `KvCache::kv_bufs_ref(&self)` (read-only) shared accessor. Solo decode and prefill paths untouched (discipline: don't touch the reference paths)
- **Tests**: 106 tests green (the fp32/int8 batch==solo bit-identity tests directly cover the parallel path; test model is GQA-shaped). Hardware: fp32/int8 both 4 concurrent == solo fully identical. CI 38240311689 ✅ (passes on x86 too)
- **Profile** (interleaved A/B measurement, see lessons below):

  | Measurement | new (P10) | old (P9) |
  |---|---:|---:|
  | batch 4×256 aggregate | **33.4 / 33.1 tok/s** (avg 33.2) | 31.6 / 32.2 tok/s (avg 31.9) |
  | solo 256 | 17.3 tok/s (unchanged) | same |

  - **P10 effect: aggregate +4%** (new code won in both interleaved pairs, reproducible). Physically consistent: P9's profile had attn_core at 5.5% in batch; parallelization recovers most of that wall time
  - **Measured-pollution example**: a naive before/after comparison showed new 6.78s/41.2tok/s vs old 12.99s/21.8tok/s (2×!), but interleaved A/B showed only +4% — **other resident inference workloads on the same host use the CPU**, so background load varies by time of day and distant absolute values are incomparable
  - Lesson: **on the same host, always measure performance with new/old interleaved + same workload** (script `p10_ab2.sh`)
- **Next**: analysis of remaining bottlenecks (average batch width drop, memory contention) / b=4 for 3B/7B / quantifying prefill interruption impact

### P11: Streaming detok byte-split character fix + gpt2 exactness baseline refresh ✅ (2026-10-12, commit `16319c7`)

- **Trigger**: gpt2 exactness test failing with garbled output. Re-downloading the model produced the **same file** (548,105,171B, English raw generation normal) and the failure persisted → not the model
- **Cause 1 (stale baseline)**: CI's exactness job only pulled Qwen, so the gpt2 test was "model not cached → skip" and **had never actually run**. The baseline string dated from the original conditions
- **Cause 2 (streaming detok bug)**: in byte-level BPE (GPT-2), multibyte characters split across token boundaries and incomplete byte sequences decode to U+FFFD. Letting that U+FFFD into the confirmed text shifts subsequent byte diffs and drops characters (the "??の??は?" symptom)
- **Verification (reference implementation)**: installed HF transformers (fp32, greedy) in a container and compared the same prompt → **fully matches denno's output** (GPT-2 base repeats the input with the fallback-template prompt = the echo is plain greedy behavior)
- **Implementation**: `emit_delta_and_check_stop` — **hold back** tokens whose tail is U+FFFD and flush them together once the character completes. New `flush_stream` — flushes held tokens at generation end (EOS/max_tokens) in both step_chat/step_chat_batch (EOS excluded from decode). **Text path only; the numeric path is untouched** (outside profiling scope, performance-neutral — confirmed by the reference output and 4-session solo==batch)
- **Tests**: unit 99+7 all green + exactness 5 all green. **CI 38244751670 ✅** — workflow now pulls gpt2, and **the gpt2 test actually runs and passes on x86** (baseline proven to match the reference implementation). Deployed; reference output unchanged

### P12: Remaining bottleneck analysis + decode attention optimization (probs@V row merge + direct strided reference) ✅ (2026-10-12, commits `55c1de0`/`f3dea3b`)

- **Trigger**: with the dev environment migration complete, quantified P9's remaining bottlenecks (average batch width drop / memory contention / prefill interruption) in a clean environment with no other workload load
- **Analysis 1 (batch-width scaling)**: concurrent firing × 256-token cap (natural EOS termination; token counts differ per width, so **compare aggregate tok/s**). fp32 scales linearly up to W=8 (16.2→~50); int8 **tops out at W=4** (149→132–135, slightly down at W=8). Slots are `DENNO_BATCH_SLOTS` (default 4); a true width-8 batch needs 8 (default 4 gives two-way queuing)
- **Analysis 2 (per-op profile)**: fp32 W=4 has linear layers + lm_head at ~85% (bandwidth-bound as expected). int8 W=4: gate_up 29.5% / **attn_core 19.4%** / down 18.0% / lm_head 10.8% — **in int8, attention is the second bottleneck after linear layers** (KV contiguous copy + m=1 sgemm ×7 per (kvh,g) + 23 allocations/task; the t×dh buffer crosses the mmap threshold and mmaps/munmaps every step)
- **Finding (kernel-swap pitfall)**: matrixmultiply **picks different SIMD kernels by stride and row count, changing the numbers**. The scores side (passing b in transposed layout rs=1/cs=k) is not bit-identical with either row merging (rows=m) or strided reference (confirmed empirically). The probs@V side (row-major b) **is bit-identical with row merge + strided reference** (measured) → only the V side can honor the bit-exact discipline
- **Implementation**: ①`tensor::matmul_rows_bstrided` — probs@V as per-g matmul (m=1) × group + vh contiguity copy → **one row-merged sgemm with direct strided cache reference** (no copy) ②new `attention_decode_kvhead` — forward() (n==1) and forward_batch() now share one function ③qg copy removed; kh/scores reused in thread_local scratch (mmap churn removed) ④scores side keeps the old path (sgemm_acc (m=1) + kh contiguity copy — proven numerically necessary)
- **Tests**: unit 100+7 green + `decode_attention_probs_v_bitexact` (row-merge+stride vs old per-row, to_bits comparison) + **exactness 5 all green (bit-identical including reference output)**
- **A/B measurements** (clean environment, old 1a7fee6 vs P12, 2 interleaved rounds, 8 slots):

#### Profile

| Config | old (1a7fee6) | P12 (55c1de0) | diff |
|---|---:|---:|---:|
| fp32 W=4 | 26.8 / 26.8 tok/s | 27.7 / 27.7 tok/s | **+3.4%** |
| int8 W=4 | 126.4 / 122.5 tok/s | 130.4 / 126.2 tok/s | **+3.1%** |
| int8 W=8 | 119.0 / 113.5 tok/s | 121.3 / 119.3 tok/s | **+3.4%** |

| per-op (int8 W=4, 1022tok, profiler ON) | old ms | P12 ms |
|---|---:|---:|
| attn_core | 1,543 | **1,193 (-22.6%)** |
| gate_up_proj | ~2,343 | 2,432 |
| down_proj | 1,430 | 1,480 |
| lm_head | 854 | 869 |
| TOTAL | 7,942 | 7,796 |

- **Measurement lessons**: ①A/B servers were started under different binary names (`/tmp/denno_old`/`/tmp/denno_p12`), but the pkill pattern `denno serve` **did not match `denno_old serve`, so all 12 runs hit the same server — invalid data** (fixed in round 2 by checking the quant-mode log every time) ②`grep -A14 … | tail -13` on per-op profiles **truncates the top line (largest op)** — the first analysis misread attn as 48.9% (the gate_up row was cut; the truth: 29.5% gate_up / 19.4% attn)
- **Next**: fundamental scores-side optimization (k-direction SIMD/pack reuse) is incompatible with the bit-exact discipline → **needs a decision on whether updating the reference output is acceptable**. b=4 measurements for 3B/7B. CUDA optimization (§8)

### P13: Inference visualization (denno-trace) telemetry foundation + denno-view relay ✅ (2026-10-13, commit `69663e6`)

- **Trigger**: the user-requested inference visualization feature (§11 of the architecture guide). First stage (L0 flow / L1 statistics) of streaming the call stack, hidden layers, and KV state to a separate server for real-time display
- **Implementation**: ①new `denno-telemetry` crate (**zero deps / std only** — engine, server, and viewer (incl. WASM) share one format). Frame format v1 = `"DNTL"` (4B) + version (1B) + kind (1B) + payload, integers as LEB128 varint, probabilities as f16 (self-implemented RNE). kinds: ServerInfo/SessionAdmit/SessionDone/Prefill/Token/Step/OpSummary/KvUsage/Gap/TopK ②engine capture hooks (install/release/step_chat/step_chat_batch; all guarded by `telem::enabled()` and read-only, never touching the numeric path) ③prof.rs op timing also accumulates into telemetry (delegation via lib.rs avoids a direct prof→telemetry dependency) ④WS endpoint `/v1/telemetry/ws` in denno-server (binary frames, ServerInfo right after connect; hub→broadcast relay thread; 503 when telemetry disabled) ⑤new `denno-view` crate — WS client to the upstream denno (2s retry on disconnect), multiplex relay to multiple browsers, `VIEW_RECORD` replay recording ([u32 len][bytes] append format; input for the P14 scrubber), simple live dashboard (HTML+JS decoder; WASM in P14)
- **Design principles implemented** (§11): ①fully decoupled from inference (capture only emits) ②never make inference wait (hub is a bounded SyncSender 4096 + try_send; on full → drop + Gap frame marking the loss) ③OFF by default (without `DENNO_TELEMETRY` no capture code runs at all = zero cost)
- **Tests**: unit 100 + telemetry integration 1 + server 7 + telemetry crate 5, all green. The integration test covers OFF→ON lifecycle (SessionAdmit/Prefill/Token/TopK/KvUsage/OpSummary/SessionDone) + **generation output fully identical ON/OFF** (testutil dummy model) + all frames encode→decode reversible. exactness 5 all green. **JS decoder also verified** (all 10 frame kinds generated by Rust decoded in node → exact match, including f16 2.75/0.5/0.25 and u64 varint)
- **e2e on hardware** (local denno-dev:8090 + denno-view:8093): with telemetry ON the reference output is **bit-identical** (usage 34/26). denno-view logs `server: denno-dev (quant=fp32, slots=4, ctx=32768)` / `session 1 admit (slot 0, prompt 34 tok)` / `session 1 done (reason 0, 26 tok)`, **Gap 0**. Dashboard HTML served OK
- **2 bugs found in e2e**: ①**f16 decode subnormal path exponent off by 2^14** (found by a round-trip test; 6.1e-5→0x03FF→mangles into 0.999; after the k-loop the f32 exponent is 113-k) ②**idle KvUsage frame flood** — the scheduler worker keeps polling `step_chat_batch` with empty batches while idle, so emitting KvUsage on empty batches produced thousands of frames/sec → hub full → Gap flood (fixed: no emission on empty batches; idle polling itself is existing design)
- **CI**: 38275707105 ✅ (build+test+exactness). Deployed; reference output bit-identical (production runs telemetry OFF)
- **Next**: P14 (WASM viewer: wasm-bindgen + canvas/WebGL2, replay scrubber) / P15 (L2 tensor focus)

### P14: denno-view inference-flow animation (call graph + layer panel + KV/variables) ✅ (2026-10-13, commit `8befe3f`, JS version; WASM port pending)

- **Trigger**: the user-specified denno-view spec — a model-driven program call graph in the center (prompt drops down, functions light up in order, values pass as pulses), a stack of model layers on the right (lighting up when referenced), and the KV cache + hidden variables on the left (lighting up on stores with values shown)
- **Telemetry extension**: frames extended with the data needed to light the call graph sequentially — ①`LayerRun { layer, ops }` (one layer's op sequence bundled into one frame in execution order, 24 frames/token) ②`ServerInfo.layers` ③`Prefill.tokens` (for the drop animation). Implemented with a thread-local layer buffer + `prof::time_layer`
- **denno-view UI**: 3 columns (center = call graph + implementation-code panel / left = KV slots + variables + op timing / right = model structure). The browser side keeps the **never-make-inference-wait** design with a playback queue + decimation
- **Verification**: all tests green + exactness 5/5 + JS decoder 12 kinds matched + e2e reference output **bit-identical** with Gap 0 (509 LayerRun frames). CI 38298659275
- **P14.1–P14.3 feature additions (2026-10-13, commits `6b198f2`/`e34e411`/`06df5ed`)**: ①layer panel fix — view keeps ServerInfo/KvUsage snapshots so browsers opened later render immediately + layer rows show op names/measured ms ②token text display — `VIEW_TOKENIZER` → `/tokenizer.json` serving, BPE inverse map + streaming UTF-8 decoding correctly restores multibyte characters across token boundaries; input/output tokens shown as text chips ③built-in chat — view's `POST /chat` relays to the upstream denno `/v1/messages`, and sending a message drives the animation automatically
- **P14.4 stepping code + model structure (2026-10-13, commit `91bbb6a`)**: implementation-code panel in the center-left (pseudo-code; the running line advances like a debugger PC + `li = N/24` watch), right panel rebuilt as **model structure** (embed→decoder×24[ATT+MLP chips]→final_norm→lm_head, chips lighting individually per LayerRun)
- **P14.4b–k UI polish (details omitted, commits `62acfc8`–`5e64eef`)**: layout fixes (gaps/equal split/pane separation/section dividers), magnifying cone (code panel = magnified callout of the model_forward node), token placement (input above the graph, output below), fly-dots to KV/variables (SVG fly-lines), UI fully in English, heading "denno (LLM Server)", speed slider (low/high, fastest by default, low = watchable slow motion), ServerInfo model_id added (real model ID display). **The never-make-inference-wait design held throughout**
- **P14.4l–q code panel accuracy + real-code toggle (commits `091da66`–`3ff6175`)**: ①pseudo-code execution order fixed to match the real implementation — KV writes are inside the loop (right after rope), attn_core reads the cache (`&kv[li][..pos+1]`), prefill branch (p1-p4) added. During prefill the cursor walks p1→p2 (bulk KV write)→layer cascade→p3 ②**pseudo⇔real code toggle** ("real code ▸" button, pseudo default) — scheduler.rs/engine.rs/llama.rs embedded at compile time via `include_str!` + served at `/code/src` (the displayed code is always the same generation as the running binary), pseudo↔real mapped by content anchors so cursor/state lighting/fly-lines work in both modes ③**usage sites** of the KV cache and hidden x shown as fly-lines (KV→attn read, x→rmsnorm/final_norm)
- **P14.4r chat output synced to emit_token (commit `b1e6ecb`)**: the built-in chat used to display the whole reply at once regardless of the animation speed. Now **each token is appended to the chat bubble when its Token frame plays** (with speed=low the output display slows down accordingly). The session is captured at the first SessionAdmit after sending, and the bubble is finalized with usage once all output tokens are played. Fallbacks to showing the full reply (4s watchdog) when the tokenizer is missing, telemetry is off, or Token frames are dropped. Also fixed a latent bug where tokText was called twice per token (consuming the streaming UTF-8 decoder twice). Verified: JS syntax OK, dev server rebuilt+restarted, chat reply OK, p143_check CHAT_OK (19 token frames)
- **P15 Qwen3 support (commit `e67eeb2`)**: **Qwen3 = Qwen2 + QK-Norm** — q_norm/k_norm.weights applied as per-head_dim RMSNorm after the QKV bias, before RoPE (`qknorm_rows_inplace`, new `qk_norm` profiler op). Batch/solo decode bit-identity covered by a unit test (`qk_norm_batch_matches_solo_decode_bit_exactly`). **KV memory budget** (`DENNO_KV_BUDGET_MB`, default 12288MB): explicit env vars win, slots first + prefix cache with the remainder (min 2048 tokens guaranteed), small-KV models keep their previous configuration entirely — fixes the ~47GB KV preallocation SIGKILL on Qwen3-0.6B. **minijinja-contrib pycompat**: registers the Python-style str methods (startswith etc.) that HF chat templates rely on, so Qwen3's thinking template renders. Exactness: Qwen3-0.6B greedy output verified fully identical to an HF transformers (fp32) reference implementation + CI now pulls Qwen3-0.6B
- **P16 replay scrubber (commit `5def63e`)**: playback UI for the recorded frame stream in denno-view. Server side: `VIEW_REPLAY=/path` makes `GET /replay` serve the record ([u32 len][bytes] concatenation) **read per request** (recording can continue while replaying; a record truncated mid-write by a killed process is tolerated — the parser reads up to the last complete frame). Browser side: scrubber bar in the header (replay/live toggle, play/pause, jump-to-start, step −1/+1, slider seek, position display). **Seek rebuilds state by fast-forwarding from frame 0** (snapshot frames ServerInfo/KvUsage via the normal path; animation frames Token/LayerRun/TopK/Prefill state-only, transient effects skipped — the `RP_FFWD` flag). Playback reuses the live pacing (queue+drain); live WS frames are ignored while in replay mode. Verified: JS syntax OK, record→`/replay` byte-identical (same md5), 146/300-frame parse check, UI serving check
- **Remaining**: ①WASM port (wasm-bindgen for the decoder + state machine; JS version stays as the working implementation) ②L2 tensor focus (P18; hidden-variable values are currently L1 statistics only — KV length/top-k/tokens)

### P17: CUDA int8 offload kernels (bit-exact to the CPU int8 path) ✅ (2026-10-13, commit `11281dd`)

> User direction: "establish CUDA optimization before other model families". Foundation of migration step 1 (linear layers) from §8. The L2 tensor focus previously labeled P17 in the candidate list moves down to **P18**.

- **Implementation** (`denno-engine/src/cuda.rs`, feature `"cuda"`): the CUDA Driver API is resolved at runtime via **dlopen** (zero dependency crates; builds without the feature — CI / GPU-less environments — are completely unaffected). The kernel `kernels_int8.cu` is **compiled to compute_75 PTX and committed** (`nvcc -ptx -arch=compute_75 -fmad=false`; the driver JIT runs it on all newer archs sm_86/89/120 — verified on an RTX 5090 sm_120 / driver 13010). Kernels: ①per-row dynamic activation quantization (dno_quantize_rows, 1 block/row, shared-memory maxabs reduction) ②int8 GEMV/GEMM (dno_matvec_rows_bt_i8, 1 warp per output, lanes sweep k coalesced + warp i32 reduction)
- **Bit-identity design** (CPU: `tensor::quantize_row_i8` / `matvec_rows_bt_i8`): ①maxabs involves no rounding so the sweep order is irrelevant; `roundf` = `f32::round` (half-away-from-zero); the ±127 clamp follows the same rule ②i8×i8→i32 accumulation is **integer, so associativity holds** — the output is independent of warp-reduction/lane-split order (this is why it matches the CPU AVX2/sdot kernels bit-for-bit) ③the output scale multiplies `(float)dot * ascale[i] * scale[j]` left-to-right (same associativity as the CPU)
- **GPU runtime environment** (new): a **denno-gpu container** on the dev host (nvidia/cuda:12.9.1-devel-ubuntu22.04 + rustc 1.99.0, `--gpus all`; /work/denno-beta binds the same host tree as the existing dev container; builds isolated via `CARGO_TARGET_DIR`). RTX 5090 = 32GB / sm_120 (requires CUDA 12.8+, so the existing 12.6 image could not be used)
- **Verification**: `cuda_quantize_rows_bit_exact_vs_cpu` / `cuda_linear_int8_bit_exact_vs_cpu` — across decode m=1, batch decode m>1 and prefill GEMM, **to_bits fully identical** on the RTX 5090 for real model shapes (Qwen2.5-1.5B) including lm_head 1536×151936
- **Microbenchmark** (weights resident in VRAM, decode m=1, activations H2D every call): **lm_head 18.6×** (5.46ms→0.294ms, effective 794 GB/s) / down 11.1× / gate_up 2.6× / qkv·o 0.4–0.6× — small layers are dominated by the per-call fixed overhead (launch + sync + H2D/D2H ≈ 40µs). gate_up at batch m=8 reaches 4.1×
- **CI**: added `cargo check -p denno-engine --features cuda` (compile-only, no GPU needed)
- **Next** (P17b: engine integration): weights resident in VRAM + a **device-resident activation pipeline** (one sync per step → removes the small-layer fixed overhead) → attention fusion kernel → sampler → paged KV (in §8.4 order)

### P17b: CUDA engine integration (resident weights + pinned async pipeline) ✅ (2026-10-13, commit `197a06c`)

> Wires the P17 kernels into the real path of every int8 linear layer and proves bit-identity plus 3.4× e2e. The first phase to confirm §8.3.5 — "with int8, GPU offload does not change the output" — at the e2e level.

- **A single funnel as the integration point**: hooking `maybe_gpu_linear` into the int8 paths of `quant::Linear::apply_bt` / `apply_batch_bt` / `apply` is all it takes. Every int8 linear layer of llama/gpt2 (qkv, o, gate/up, down, lm_head) flows through these three functions, so **zero model-side code changes**. A `None` return falls back to the existing CPU kernels (both are bit-identical, so the output is unchanged)
- **The resident-weight registry is keyed by uid**: the first attempt keyed on the buffer pointer of `Quant8Linear.w`, but a test exposed a collision — **Vec free→realloc reuses addresses, so a different weight set can appear at the same address** (happens between different Linears of identical size and shape). `Quant8Linear` now gets a `uid` assigned at construction (Clone shares the uid safely: a clone is a deep copy of identical content), and the registry uploads on first use keyed by uid. A debug_assert also checks the shape match per uid
- **Pinned + async, one sync per pipeline**: H2D(x, async) → quantize kernel → GEMV kernel → D2H(out, async) → `cuCtxSynchronize` ×1 per linear layer. P17 synced on every copy and launch (×4, ≈40µs fixed); with pinned host buffers (`cuMemHostAlloc(PORTABLE)`) + default-stream ordering the overhead drops to **≈28µs** (qkv 0.041→0.033ms, lm_head 0.286→0.220ms @ decode m=1)
- **Decode-only offload**: GEMV-style kernels (1 warp per output) re-read the weights proportionally to m, so they lose to prefill GEMMs (**measured 0.39× at m=128**). `m > DENNO_CUDA_MAX_M` (default 16) stays on CPU — prefill performance is untouched; only decode and batch decode (m ≤ threshold) run on the GPU
- **Verification**: ①pipeline bit-exactness via apply_bt (4 shapes, including lm_head 1536×151936) ②whole-forward GPU offload bit-exactness on the tiny model (prefill + batch decode vs solo) ③the 106 CPU tests regress to zero changes (without the cuda feature) ④**e2e**: Qwen2.5-1.5B int8 — CPU serve vs GPU serve produce fully identical response text and usage (2 prompts, 200-token greedy; concurrent batch decode of 2 requests also matches solo)
- **e2e measurement**: 200-token greedy generation **CPU 9.08s → GPU 2.66s (3.4×)**. VRAM usage is lm_head 234MB + layer weights ≈ just over 1GB
- **Per-op profile comparison** (`DENNO_PROFILE=1`, 200-token greedy ×3 requests, 600 decode steps + prefill cumulative):

  | op | CPU ms | CPU % | CUDA ms | CUDA % | speedup |
  |---|---:|---:|---:|---:|---:|
  | lm_head | 3,049 | 10.7% | 281 | 2.9% | **10.9×** |
  | gate_up | 11,685 | 40.9% | 2,340 | 24.2% | **4.99×** |
  | down | 6,108 | 21.4% | 1,432 | 14.8% | 4.26× |
  | o | 1,594 | 5.6% | 590 | 6.1% | 2.70× |
  | qkv | 2,051 | 7.2% | 1,614 | 16.7% | 1.27× |
  | attn_core | 3,190 | 11.2% | 2,767 | 28.6% | 1.15× |
  | swiglu | 554 | 1.9% | 420 | 4.3% | 1.32× |
  | other | ≈361 | 1.3% | ≈243 | 2.5% | 1.49× |
  | **TOTAL** | **28,591** | 100% | **9,686** | 100% | **2.95×** |

  Decode 22.1→**75.8 tok/s** (e2e 3.4×). **The linear layers shrink from 86% to 62% and attn_core becomes the top op (28.6%)** — clearly identifying attention fusion as the next bottleneck. qkv stays at 1.27× (3 split calls × 50,400 — fixed overhead dominates) → fused in P17c

### P17c: Fused GPU linear groups (qkv / gate+up single call) ✅ (2026-10-13, commit `7bf9eb0`)

> Resolves the "qkv split into 3 calls × 50,400 invocations (600 steps × 28 layers × 3) fixed overhead" identified in the P17b profile. Linear layers that consume the same input are fused into one GPU call; e2e improves 19%.

- **Fusion = row concatenation**: the weight rows of q/k/v (or gate/up) are concatenated in part order into a single buffer, registered and computed under a uid-vector cache key, and the output is split by column blocks. Per-row arithmetic (quantization rule, i32 accumulation, scale multiply order) is exactly identical to the individual calls, so **bit-exactness is guaranteed structurally** (no new arithmetic rules introduced). The host-side concat buffer is dropped after upload, so no steady-state RAM growth. When the GPU is disabled/errored or a part is fp32, it falls back to the previous separate 3+2 calls
- **Discovery: CUDA current context is thread-local (the biggest lesson of this phase)**: libtest spawns a **new thread per test even with `--test-threads=1`**, so calling `cuCtxSetCurrent` once on the initializing thread leaves `cuMemAlloc` failing on other threads with **error 201 INVALID_CONTEXT**. Worse, the failure is silently absorbed by GPU_BROKEN→CPU fallback, so "GPU tests green" no longer means "ran on the GPU" (only the new fusion test `expect`ed on None, which exposed it — the P17b model-integration test could have been passing as CPU-vs-CPU). Fixed by calling `CudaCtx::make_current()` at every GPU entry point; this also removes a latent bug in the multithreaded server
- **Verification**: ①fused vs separate-call bit-exactness (qkv-type 3 parts with differing n + gate/up-type 2 parts, m=1/4) ②whole-forward bit-exactness on the tiny model (prefill + batch decode) ③the 106 CPU tests regress to zero changes ④**e2e**: Qwen2.5-1.5B int8 200-token greedy — **text and usage fully identical** to CPU serve (3 requests, usage 49 in/200 out)
- **e2e measurement**: 200-token generation **GPU 2.638s → 2.146s (−19%)**, matching the theoretical prediction (overhead removed 84µs/layer/step × 28 layers × 600 steps ≈ 0.47s/request → 2.64−0.47=2.17s). CPU 10.67s → **4.96×**. Weight uploads: grouped 56 (=28 layers × 2 groups) + separate 57 (=down 28 + o 28 + lm_head 1), fully consistent with the call breakdown
- **Post-fusion profile re-acquisition** (`DENNO_PROFILE=1`, same workload as the P17b section — the pre-fusion distribution is the P17b table):

  | op | P17b GPU ms | P17c GPU ms | change |
  |---|---:|---:|---:|
  | qkv | 1,614 | 610 | **−62%** |
  | gate_up | 2,340 | 1,850 | **−21%** |
  | attn_core | 2,767 | 2,623 | −5% |
  | swiglu | 420 | 405 | −4% |
  | lm_head | 281 | 275 | −2% |
  | down | 1,432 | 1,426 | ±0% |
  | o | 590 | 591 | ±0% |
  | **TOTAL** | **9,686** | **8,021** | **−17%** |

  The CPU side re-measures at TOTAL 28,532ms (nearly unchanged from P17b's 28,591ms). Op-sum ratio **2.95×→3.56×**, decode **75.8→96.1 tok/s**, e2e (CPU 9.07s → GPU 2.08s) **4.36×**. The reductions match theory (qkv: 2 calls' worth 941ms/3req theoretical vs −1,004ms measured; gate_up: 470ms vs −489ms). **attn_core becomes even more dominant (28.6%→32.7%)** — attention fusion is confirmed as the next bottleneck
- Remaining CUDA candidates: attention fusion kernel (attn_core 32.7% = top GPU op), paged KV, sampler kernel

### P17d: Elementwise GPU kernels + decode chain (device-resident hidden state) ✅ (2026-10-13, commit `121aa3a`)

> The P17c profile left swiglu (5.1%) and host round-trips in place, so this phase (1) GPU-izes the elementwise ops (rmsnorm/residual/swiglu) and (2) turns one solo decode step into a device-resident pipeline. CPU attention uses `matrixmultiply::sgemm` (SIMD/FMA accumulation order is implementation-defined), so moving attention to GPU requires relaxing the exactness contract → deferred; this phase was done first since it needs no decision.

- **Elementwise kernels (`kernels_elem.cu`)**: `dno_add_rmsnorm` (residual add + RMSNorm; the sum of squares is computed sequentially in one thread in the same order as CPU, then `1/(ss+eps).sqrt()` → `x*inv*w` left-associated), `dno_swiglu` (silu + up multiply), `dno_expf_probe` (test entry). Compiled with `-fmad=false` plus explicit `__*_rn` intrinsics so contraction is fully controlled
- **Exact glibc expf replication (the hardest part of P17d)**: swiglu's `exp(-v)` must match Rust `f32::exp` (= glibc expf) bit-for-bit. glibc's e_expf uses f64 internal arithmetic + the `__exp2f_data` table (32 entries) + a cubic polynomial, all reproducible on GPU — but the **actual CPU binary is the FMA-contracted build** (the multiarch IFUNC selects the FMA variant), which differs from a naive sequential implementation by 1 ULP in the final f32 (measured at x=32.564632f: the single mismatch out of a 455M-value sweep). Experiments with gcc -O2 -mfma plus disassembly identified **5 contraction sites** (1. `kd = fma(InvLn2N, xd, SHIFT)`, 2. `r = fma(InvLn2N, xd, -kd)`, 3-5. the three polynomial mul+add pairs); replicating them with `__fma_rn` achieves **bit-exactness on 455,078,062 values** (specials + full exponent scan 0x3C..0xC2 at mantissa step 5 + random bits). Note: this contract depends on the host's glibc build (FMA variant selection)
- **Decode chain**: one solo decode step (n==1) becomes a device-resident pipeline: "embed H2D → [rmsnorm(+residual) → qkv(grouped) → (CPU: rope/attention/KV) → o_proj → rmsnorm(+residual) → gate/up(grouped) → swiglu → down] × layers → final norm → lm_head → logits D2H". Host round-trips are only qkv and logits, with **one sync per layer (waiting on the qkv D2H) while the GPU pre-executes the next layer's FFN during CPU attention**. Linears use the same P17b/c kernels and uid registry, rmsnorm/swiglu replicate the CPU op order → bit-exactness is structurally preserved. Unsupported structures (bias / Qwen3 QK-Norm / partially fp32) or runtime errors fall back via GPU_BROKEN to the existing path (k/v writes are idempotent and advance happens once at the end, so retry is safe)
- **Verification**: (1) GPU tests 12/12 (new: 455M-value expf bit-exact + whole-chain bit-exact), (2) CPU 106 tests unchanged, (3) **e2e**: GPU serve vs CPU serve on Qwen2.5-1.5B int8 produce **identical responses** (2 prompts; byte-identical except the random msg id), (4) measured: 200-token request GPU 2.1s vs CPU 6.6s (**3.2×**); 150-token 1.46s vs 6.6s (**4.5×**)
- Lessons: (1) the `f32::exp` loop used as the test's reference can be auto-vectorized by LLVM into **libmvec** (the vector expf, which differs from scalar by 1 ULP) — pin it to scalar calls via `#[inline(never)]`. (2) `cargo build -p denno-cli` does **not propagate the cuda feature** into the CLI binary (`-p denno-cli -p denno-engine --features cuda` is required) — a feature-less binary runs silently on CPU with no GPU logs
- Remaining CUDA candidates: attention fusion kernel (needs the exactness-contract relaxation decision = deferred), paged KV, sampler kernel

### P17e: Demand-based KV allocation (growing cache + reservation hard guarantee) ✅ (2026-10-13, commit `c889336`)

> The P15 KV memory budget assumed "one slot = the full context preallocated"; on Qwen2.5-1.5B that is ~1.9GB per slot, and the default 4 slots preallocated ~7.5GB before serving. Since real sessions mostly use a few hundred to a few thousand tokens, this phase switches to **growing buffers by the amount actually written**, which allows many more slots. Fully paged KV (block-based) is a larger structural change and stays deferred; the design keeps the **contiguous layout intact** while making allocation demand-based.

- **KvCache rewrite**: the old implementation preallocated `[max_len, kv_dim]` at slot creation (one slot = full ctx width × bytes_per_token). The new one **allocates 0 initially** and grows all layers' buffers geometrically (doubling, capped at max_len, amortized O(1), realloc preserves written rows). The memory layout stays "contiguous buffer with the first `len` rows in use", so **every numeric path including attention is completely identical to the old implementation (bit-identical)** — a structural change that never touches the exactness contract
- **Slot count recomputed**: `DEFAULT_BATCH_SLOTS` changed 4 → **32 (cap)**; the default is derived from "one session's tentative reservation of 4096 tokens (`SESSION_RESERVE_TOKENS`) × bytes_per_token fits the KV budget". On Qwen2.5-1.5B int8 (bpt 57,344B) that is 235MB/slot → 32 slots within the 12GB budget. Because KV is demand-based, **the pre-serve allocation is 0**
- **Hard guarantee at install**: `install_session` registers `prompt + max_tokens` as a KV reservation and verifies (1) the reservation fits the slot's `max_len`, (2) the sum of all busy slots' reservations × bytes_per_token fits `DENNO_KV_BUDGET_MB`. Exceeding either returns 400 (`invalid_request_error`) — closing the weakness of grow-on-demand ("OOM can strike mid-generation") with reservations
- **Measured**: startup RSS on Qwen2.5-1.5B int8 **~9.9GB → ~2.5GB** (the old build preallocated 4 slots × 1.88GB). e2e (200 tok): GPU 2.15s vs CPU 9.1s, **GPU/CPU responses fully identical** (byte-identical except the msg id). A server started with a 1MB budget correctly rejected the over-reservation request with 400 on real hardware
- **Verification**: CPU tests 110 ok (kv_cache unit tests expanded for growth/preservation/snapshot/restore), GPU tests 12/12, denno-server wire tests unchanged
- Lesson: **models (dummy models) that advance `len` via `advance()` alone could skip capacity allocation** — demand-based allocation introduces the contract "a model writing KV must call `kv_bufs(layer, n)` before writing", and the 3 dummy models plus denno-server's test model were adapted to it. The old build made "snapshot works after advance alone" pass silently because everything was preallocated, which hid the contract from tests
- Remaining CUDA candidates: attention fusion kernel (needs the exactness-contract relaxation decision = deferred), paged KV, sampler kernel

### P17f: Fundamental scores-side optimization (baseline update, user-approved) ✅ (2026-10-13, commit `d7d126e`)

> The scores-side transposed-layout optimization deferred since P12 was implemented with user approval ("change the baseline"). matrixmultiply switches SIMD kernels based on strides/row counts, so numerics change — but **determinism (same shapes → same results) is preserved**. Per the baseline-update rule, verified against the HF transformers reference before landing.

- **Decode scores**: per-g `sgemm_acc(m=1, kh contiguous copy)` × group → a single `sgemm_scores_kvstrided(m=group, kc strided reference)`. Eliminates the kh contiguous copy (t_total×dh × kv_heads×layers per step) and KH_SCRATCH
- **Prefill**: all qg/kh/vh contiguous copies removed; scores computed with strided references into the Q row buffer and the KV buffers. probs@V now uses `matmul_rows_bstrided_par` (row-chunked parallel) reading v in place
- **Numerics**: intentionally not bit-identical to the P12 path (as expected). However **the greedy output text of all 6 stored baselines is unchanged** (0.5B fp32 "1+1", "日本の首都は?", GPT-2, Qwen3 included). **HF verification: 0.5B fp32 200-token greedy generation is identical to the transformers reference** (prefecture enumeration prompt)
- **Determinism**: solo/batch decode bit-exactness and prefix-cache exactness tests all pass (identical shapes always produce identical results)
- **Performance** (1.5B int8, 200-token e2e): CPU warm 9.10s → **8.46s**; GPU chain 2.15s → **1.57s** (the chain's CPU attention goes through the same function and improves automatically)
- **Fix (discovered)**: the committed `kernels_elem.ptx` **did not match** compiling the committed `.cu` with `nvcc -arch=compute_75 -fmad=false` (committed at P17d without a hash check); this caused a 1/455,078,062 mismatch in the GPU expf test. Replaced with a fresh compile; GPU tests back to 12/12. **Lesson: always hash-verify generated artifacts (.ptx) against the validated build before committing**
- Verification: engine CPU 110 ok + workspace green + GPU cuda_ 12/12 + e2e GPU/CPU responses fully identical (18091/18092)

### P17g: GPU offload of decode attention (dno_attention kernel + GPU KV mirror) ✅ (2026-10-13, commit `bf2a3c6`)

> With user approval ("CPU/GPU mismatch is acceptable"), only the attention exactness contract is relaxed to enable GPU offload. The bit-exactness contracts for linear layers (int8) / expf / swiglu / rmsnorm are unchanged.

- **dno_attention kernel**: grid=(heads), block=(dh); one block handles one q head (GQA: kvh=h/group). scores → softmax (exp uses dno_expf to minimize the CPU delta) → probs@V. Shared memory dh*4 bytes
- **GPU KV mirror**: the host KvCache stays authoritative; a decode-chain copy is kept per session on the device (keyed by `KvCache::kv_id()`, no collisions since caches are not cloned). Missing rows are bulk-uploaded H2D on the next chain_attention; rollback/evict re-seed automatically via the len comparison. LRU eviction beyond `DENNO_GPU_KV_MB` (default 8192MB)
- **Applied on both paths**: solo (forward/forward_decode_gpu) and batch (forward_batch/chain_attention_batch). **GPU attention only consumes already-biased q/k/v and is independent of bias/QK-Norm**, so it does not depend on chain_compatible (the solo chain condition). Adopted only when all sessions succeed; any failure redoes all sessions with CPU attention (no mixing) → the solo/batch bit-identity contract (P6) is preserved
- **Key finding**: production Qwen2.5-1.5B **has QKV biases** (from safetensors), so the solo chain had never fired since P17d (the P17f GPU e2e speedup came from per-op GPU linears). The initial implementation gated GPU attention on chain_compatible, so it never engaged either; identified by temporary logging, then the gate was removed and the per-op path also wired up. **Lesson: "green" does not prove a path fires — verify path gates once against the real model structure**
- **`DENNO_GPU_ATTN=0`**: escape hatch that disables GPU attention and restores the bit-exact CPU attention path
- Tests: `cuda_attention_kernel_vs_cpu_close` (kernel vs CPU reference, maxabs 2.98e-8) and `cuda_gpu_attention_forward_vs_cpu_close` (GPU/CPU greedy token match + small logits tolerance). GPU cuda_ **14/14**, CPU **110**, exactness **6 criteria** pass
- **Performance** (1.5B int8, long-context decode, t≈7700, 128 tokens): CPU attention **18.52s → GPU attention 4.23s** (**4.4x**). At short contexts (t≈130) ON/OFF are nearly equal (sync-bound); the benefit concentrates at long context
- **Known remaining items**: ① long-prompt prefill is slow (~100s at t≈2810; pre-existing issue unrelated to P17g, needs investigation) ② GPU/CPU e2e responses now differ due to the relaxed attention (start the GPU side with `DENNO_GPU_ATTN=0` to restore bit-exact comparison) ③ GPU attention gains little at short context (rope/syncs remain — next candidate)

### P17h: GPU offload of rope + sync-free decode chain (dno_rope_qkv + device-resident qkv) ✅ (2026-10-13, commit `b090d03`)

> Resolves P17g remaining item ③ (little gain at short context = rope/syncs remain). rope is NOT part of the relaxed contract, so it is GPU-offloaded while **staying bit-exact**.

- **dno_rope_qkv kernel**: performs qkv bias addition + rope in one kernel. grid=(heads + 2*kv_heads), block=(dh). Blocks b < heads handle q heads (bias+rope), heads ≤ b < heads+kv_heads handle k heads (bias+rope), the rest handle v heads (bias only). Bias is a single concatenated [qb | kb | vb] buffer (NULL when absent)
- **Bit-exact design**: cos/sin are computed on the host (glibc sinf/cosf of the same process) with the **same formula** as CPU `rope_row`, and uploaded as a `[cos(half)|sin(half)]` table once per step (`chain_rope_prepare`). Device-side arithmetic uses `-fmad=false` mul.rn/add.rn/sub.rn (same rounding as Rust scalar ops). Bias is added before rope (same order as the CPU path). Verified bit-exact by `cuda_rope_qkv_bitexact_vs_cpu` at pos0=0/1/7/63, with and without bias
- **Removing per-step syncs**: the qkv linear output stays device-resident (no CPU download); rope/attention complete on-device. Roped k/v are asynchronously D2H-copied into a pinned staging buffer, and the host KvCache (always authoritative) is updated once per step after the final sync (logits download) via `chain_kv_stage_flush`. Pinned buffers abandon ping-pong in favor of **dedicated slots per (layer × purpose)** (with no syncs, async DMA overwrites are avoided by id assignment)
- **Applied path**: forward_batch **delegates b==1 to the solo chain (forward_decode_gpu)** (so P17h reaches the server's actual decode path; falls back to the batch path on failure). The qkv-bias condition is removed from chain_compatible (dno_rope_qkv supports bias)
- Tests: `cuda_solo_chain_bias_rope_gpu_matches_cpu` (bias-bearing Qwen2.5-like structure; **asserts the chain fires** via Some return + CPU greedy match). GPU cuda_ **16/16**, CPU **110**, exactness **6 criteria** pass, 0 errors in the serve log
- **Performance** (1.5B int8, dev 8090): short request (t≈38, 2 tokens) **11.0ms** (vs 14.4ms on the `DENNO_GPU_ATTN=0` batch path — sync-removal effect); long decode (t≈7657, 128 tokens, warm) **4.23s → 3.88s** (1.09x). Cold prefill unchanged (~366s; pre-existing issue)
- **PTX**: sha256 `a78c645c23b4e0d17504c705c23fb4fbbf10ba0737a0ca3c12bd65c342fe10e5` (verified against a fresh `nvcc -arch=compute_75 -fmad=false -ptx` compile)

### P17i: GPU offload of int8 linears for long prefill (tiled GEMM, dno_gemm_i8) ✅ (2026-10-13, commit `f2c1831`)

> Resolves the post-P17h remaining issue "cold prefill unchanged (~100s at t≈2810)". Profiling had shown 83% of prefill time in CPU GEMMs of the linears (gate_up 47s, down 29s), so the `maybe_gpu_linear` gate **m > DENNO_CUDA_MAX_M (default 16) → CPU** was replaced with a GEMM path.

- **dno_gemm_i8 kernel** (kernels_int8.cu): 64×64 tiles, 1 thread computes 4×4 outputs, dp4a, A/B tiles in shared memory. **Bit-exactness rationale**: integer accumulation is associative, so tiling/ordering cannot change the result; the f32 side uses the same order as the GEMV kernel, `((dot as f32) * ascale) * scale[j]`. When k is not a multiple of 4/32, zero padding at tile boundaries handles the edge correctly
- **Dispatch**: GEMM path when `m > max_m() && m >= gemm_min_m()` (default 256, `DENNO_CUDA_GEMM_MIN_M`); `m <= max_m()` keeps the GEMV path; **17..255 stays on CPU** (for small m the per-layer syncs cost more than the CPU GEMM; short prompts are fastest on CPU). `DENNO_CUDA_GEMM=0` disables the GEMM path (escape hatch)
- **Tests**: `cuda_gemm_i8_bit_exact_vs_cpu` (8 shapes: tile edges 63/64/65/100/128/130, k not a multiple of 4 such as 1533/17911; **also bit-exact vs the GEMV kernel on the same input** — verifying order-invariance of integer accumulation). GPU cuda_ **17/17**, CPU **110**, exactness **6 criteria** pass
- **Performance** (1.5B int8, dev 8090, longreq2 t=2827/max 256): **cold ~100s → 32.4s**; warm (prefix cache hit) 3.70s (same as P17h 3.88s = no decode regression); short request (t=38, 2 tokens) **11.1ms (identical to P17h** — below the threshold, short prompts stay on the CPU path)
- **Bit-exactness in production**: the cold-prefill response is **identical** to P17h (CPU prefill) — compared via the longreq2 output text. 0 errors in the serve log
- **PTX** (kernels_int8): sha256 `9021673e1a10214eaf11d1bfe1a3a697ee7b6dde10d5f0f0c5be0a39f10c02da` (verified against a fresh `nvcc -ptx -arch=compute_75 -fmad=false -O3` compile)
- Known observation: responses on a prefix-cache hit (warm) may differ from cold — **pre-existing behavior unrelated to P17i** (prefill does not run at all on a warm hit)

### P17j: GPU offload of long-prefill attention (flash-style online softmax, dno_attention_prefill) ✅ (2026-10-14)

> Measured profile finding: in an ~8000-token cold prefill, **attn_core took 334s of 395.8s total (84.5%)** (after P17i GPU-offloaded the linears, the O(m²) CPU attention dominates). The playbook's next candidate "sampler kernel" measured 0.001% and was dropped; this item was done instead. The exactness contract is the same frame as P17g (**relaxed numerics for attention only, user-approved; determinism maintained**).

- **dno_attention_prefill kernel** (kernels_elem.cu): flash-style online softmax. grid=(heads, ceil(n/PF_BQ)), block=(dh), dynamic shmem=(PF_BQ+2·PF_BK)·dh+PF_BQ·PF_BK+2·PF_BQ floats (≤48KB; PF_BQ=16/PF_BK=32). Block (h, qt) owns query rows [qt·PF_BQ, +PF_BQ) of q head h. kvh=h/group, kh_off=kvh·dh. scores[i,j]=q·K·scale (causal, j=0..pos0+i), exp via dno_expf (glibc replica). K/V are read directly from the P17g GPU KV mirror ([t_total, kv_dim] row-major, same layout as the host KvCache). **Determinism**: fixed accumulation order — tiles ascending, kk ascending within a tile, output dim d owned by thread tid=d (no atomics). Constraints: dh≥PF_BQ (a block owns PF_BQ rows + output dim tid) and shmem≤48KB (dh≤152) — verified on the host; non-qualifying shapes fall back to CPU
- **Host side**: `chain_attention_prefill` (cuda.rs) — uploads q ([n, heads·dh], rope/QK-Norm already applied), syncs the KV mirror, launches the kernel, and writes the result to CH_ATT. Returning None = CPU fallback. Since prefill completes within the attention block, fallback works **without re-running the whole forward and without touching host state** (a design difference from the P17g decode chain). A `#[cfg(feature="cuda")]` hook was added to llama.rs's prefill attention path
- **Tests**: `cuda_attention_prefill_vs_cpu_reference` — **close match** against the CPU reference path (sgemm_scores_kvstrided+causal+softmax+matmul_rows_bstrided_par; maxabs 1.49e-7), plus ①determinism (two launches on the same input are bit-identical) ②causal check (rewriting future K/V leaves past rows bit-identical; rows in the changed window do change). GPU cuda_ **18/18** (stable across 5 consecutive parallel runs), CPU **110**, exactness **6 criteria**, workspace all green
- **Performance** (1.5B int8, dev 8090, longreq400 **input_tokens 13,252 measured**; the earlier "~8000 tokens" labeling was wrong): **~13k-token cold: 395.8s → 58.6s wall** (PROFILE cumulative TOTAL 71.5s = combined with the preceding longreq (3,352tok) request), **attn_core 334s → 11.8s**, response is sane ("OK"). 3,352-token cold: TOTAL 15.8s, attn_core 1.03s (6.5%). **The remaining bottleneck is gate_up_proj 49.8% → next candidate (investigated in P17k, see below)**
- **PTX** (kernels_elem): sha256 `c41520a9c7b50b4f2aaea0c6fe6b2083c8361efd87e7a997999b89de1ad57971` (compiled twice inside the container with `nvcc -ptx -arch=compute_75 -fmad=false -O3` and cross-checked)
- **Test-infrastructure fixes (a flake latent since P17g was surfaced and resolved)**: ①tests that toggle the global `GPU_ATTN` flag raced with tests that depend on the flag's state (bit-exact vs close-match changes with the flag) under parallel execution — `test_lock()` now serializes the flag-toggling / flag-dependent cuda tests (poisoning recovered via `unwrap_or_else(PoisonError::into_inner)`) ②`chain_upload` shares a pinned buffer (pin_x), so **two consecutive uploads without a sync in between** overwrite the pinned buffer before the first DMA completes, corrupting the device-side input (`cuda_chain_elem_bit_exact_vs_cpu` flaked under parallel execution; production paths always have a download sync between uploads so they are unaffected — added `ctx.sync()` in the test)

### P17k: device-resident prefill linear layers — FFN block (gate_up → swiglu → down) ✅ (2026-10-14)

> Remaining bottleneck of the 13,252-token cold prefill after P17j (single request, PROFILE TOTAL 73.6s): gate_up_proj 36.0s (49.0%) / attn_core 12.1s (16.4%) / swiglu 8.2s / down_proj 5.2s / qkv_proj 4.5s / rope 2.7s / o_proj 2.4s / rmsnorm 2.3s. **GPU utilization during prefill is near 0%** (spikes only) — wall time is dominated by CPU-side work + transfers + the sync structure, not kernels.

- **Investigation** (first half of 2026-10-14, `cuda_gemm_i8_bench_prefill_shapes`, #[ignore]): one call at m=13,252 — gate_up (k=1536, n=17920) **508ms** (output 949MB!), down (k=8960, n=1536) 125ms, qkv (k=1536, n=2048) 65ms. Via the production path (linear_gemm: Scratch+pinned) 28 consecutive calls degrade to **845ms/call** (+66% vs single — suspected throttling/memory pressure under sustained load)
- **Structural problem**: `linear_gemm` round-trips the **entire [m,n] fp32 output to the host every call (949MB D2H) + to_vec (949MB Vec alloc+copy)**, then H2Ds it back as the next layer's input. One gate_up call moves ~1.9GB host⇔device
- **Implementation (steps 1+2: device-resident FFN)**:
  - New kernel **`dno_swiglu_int`** (kernels_elem): swiglu reading directly from the [m, 2ff] interleaved layout (per-row [gate|up]) — a variant that consumes the grouped GEMM output without D2H. Identical op sequence to dno_swiglu (dno_expf + `__fdiv_rn` + `__fadd_rn` + `__fmul_rn`), so it is **bit-exact** vs CPU (silu_inplace + elementwise product)
  - **`maybe_gpu_ffn` (cuda.rs)**: gate_up GEMM → swiglu_int → act quantize → down GEMM, all chained **on device**, with only the down output [m, e] (81MB) going D2H. Eliminates the 949MB D2H + to_vec + gate/up split copies + CPU swiglu + H2D entirely. Scratch gains act/aq2/asc2/out2. Profiling attributes to GATE_UP/SWIGLU/DOWN (queue wait lands in DOWN, which contains the sync)
  - llama.rs prefill mlp block gets a `#[cfg(feature="cuda")]` hook (on success: residual add + `continue`; on failure/out-of-scope: the existing path). **Every stage is bit-exact vs CPU** (GEMM: integer accumulation; quantize/swiglu: proven identical ops) — no relaxation needed
  - **Test**: `cuda_ffn_bit_exact_vs_cpu_reference` — **bit-exact** vs the CPU reference (grouped matvec → silu_inplace + product → down matvec, with the same per-row gate split as the existing prefill path). GPU cuda_ **20/20** (stable across 3 runs), CPU **110**, exactness **6 checks**
  - **Performance** (13,252-token cold): **wall 77.9s → 33.6s, PROFILE TOTAL 73.6s → 29.1s**. gate_up_proj 36.0s → **0.6ms**, swiglu 8.2s → **0.2ms** (both gone). FFN block effectively 1.77s/layer → 0.18s/layer (**~10x**). Response sane ("OK"). Remaining: attn_core 12.1s (41%) / down 5.1s (actually includes the gate_up+swiglu queue wait) / qkv 4.5s / rope 2.7s / o_proj 2.4s
  - **PTX** (kernels_elem): sha256 `da87a462ef27bf7b71fbe5f5b1a0701a36807ba5b2651cb0f576ce90461d100a` (adds dno_swiglu_int, recompile-verified)
- **Step 3: device continuity after attention + exp2f (✅ 2026-10-14, commit 870bb87)**:
  - **`maybe_gpu_oproj` (cuda.rs)**: reads att directly from the CH_ATT device slot written by `chain_attention_prefill` and runs o_proj on device (quantize → dno_gemm_i8 → only the [m, e] output goes D2H). Removes the 81MB att D2H + to_vec + re-H2D round trip per layer. o_proj GEMM is in the bit-exact frame, so no relaxation. llama.rs prefill hook: after a successful GPU attention it tries the GPU o_proj (on failure only, it falls back to chain_download + the CPU path)
  - **exp2f in `dno_attention_prefill`**: `dno_expf` (f64-based exact glibc replica, 1/64 DP rate on desktop GPUs) → `exp2f((x-mx)*LOG2E_F)` (hardware op MUFU.EX2). Within the approved attention-relaxation frame; **determinism maintained** (same input → same result). Diff vs CPU reference: **maxabs 1.79e-7** (nearly unchanged from 1.49e-7 with expf). The decode `dno_attention` kernel stays on dno_expf (bit-exact frame)
  - **Finding: attn_core was NOT expf-bound** — exp2f left attn_core at 11.86s → 11.11s (nearly unchanged). GPU SM utilization sampled 30/30 at 0% during the request (7257MiB), and concurrent nvidia-smi sampling during the GEMM bench confirmed that **nvidia-smi reports 0% SM utilization during DMA transfers** → the remaining ~400ms/layer of attn_core is transfer + sync structure
  - **pinned H2D experiment**: switching the attention q upload (81MB) from a synchronous pageable `copy_from` to pinned staging + async H2D → **attn_core 11.1s → 11.3s (unchanged)**. The H2D itself is not the bottleneck (the change is kept; the async upload is groundwork for the next step)
  - **Performance** (13,252-token cold): **PROFILE TOTAL 29.1s → 26.4s, wall 33.6s → 30.7s**. o_proj 2.37s → **0.49s**. Remaining: attn_core 11.1s (42%) / down 5.0s (includes queue wait) / qkv 4.5s / rope 2.8s / rmsnorm 2.3s. Tests: GPU cuda_ **20/20**, exactness **6**, response OK
  - **PTX**: sha256 `528563f5a8e5ac59372a4f37051378affb615980ab8163fb48c13b7e3754cbc0`
- **Step 4: attn_core breakdown + 3.3x prefill attention kernel (✅ 2026-10-14, commit 99d2940)**:
  - **Diagnostic benches** (`cuda_transfer_bench` and `cuda_attn_pf_bench`, both #[ignore]): raw transfer bandwidth is 14.2GB/s H2D / 14.3GB/s pinned D2H (pageable D2H 8.6GB/s) — transfers account for only ~25-35ms per layer, while **the attention kernel alone measured 368ms/call at the request shape (n=t=13252, 14 heads)** (x28 layers = 10.3s) — nearly all of attn_core was the kernel itself
  - **Three fixes** (all inside the approved attention-relaxation frame, determinism maintained): ①grid swapped from (heads, row-blocks) to **(row-blocks, heads)** — with x as the fastest dimension, concurrently running blocks share one head's 27MB K/V slice in L2 (the old layout had a ~380MB working set across 14 heads and was DRAM-bound; numerically unchanged) ②**padded the q/k/v smem row stride to dh+1** — dh=128 is a multiple of 32, so the score-dot smem reads suffered **32-way bank conflicts** (~1/32 effective smem bandwidth); layout-only change, numerically identical ③**PF_BQ 16→48 / PF_BK 32→8** — K/V re-read traffic is inversely proportional to BQ (~314GB→104GB), with smem kept in the empirically safe range
  - **Kernel-only measurements**: 368ms → 149.5ms (padding) → **112.2ms/call (3.3x)**
  - **Performance** (13,252-token cold): **attn_core 11.1s → 5.12s, PROFILE TOTAL 26.4s → 20.5s, wall 30.7s → 24.5s**. Remaining bottlenecks: down 5.1s (incl. queue wait) / qkv 4.5s / attn_core 5.1s / rope 2.8s / rmsnorm 2.3s. maxabs vs CPU unchanged at 1.79e-7, GPU cuda_ **22/22** (incl. the 2 benches), exactness **6**, response correct ("OK")
  - **Empirical caveat**: dynamic smem is stable up to ~35KB but the kernel faults with an illegal address (**error 700**) above ~43KB (reproduced with BQ=64/BK=8 and BQ=48/BK=16; cause unresolved). Treat BQ=48/BK=8 (34.9KB) as the practical ceiling
  - **PTX**: sha256 `28f50332e4f2262dcaa5ae1e8caa6470f8c998d08b31fecd7f43c1253c91e9bc`
- **Step 5: device continuity of qkv → rope → K/V write → attention (✅ 2026-10-14)**:
  - **Why it did not fire at first (initial E2E stayed at 24.7s)**: diagnostic logs showed `qkv dev gate: qb=true kb=true vb=true` — **Qwen2.5-1.5B carries QKV biases** (the config has no explicit `attention_bias`, but `q/k/v_proj.bias` exists in the safetensors), so the original "no-bias only" gate fell back on every layer. Fixed by implementing a **concatenated [qb|kb|vb] bias add (before rope)** in `dno_rope_qkv_pf` (same order as the CPU path's add_bias→rope_rows, bit-exact)
  - **Per-layer device flow**: h uploaded once (H2D) after rmsnorm → qkv GEMM (dno_gemm_i8, device-resident) → **dno_rope_qkv_pf** (grid=(heads+2·kv_heads, n), block=(dh)) ropes q into a contiguous qout buffer and ropes k/v directly into rows (pos0+i) of the GPU KV mirror → attention reads device q with the step-4 kernel → only K/V rows [pos0..t_total) are D2H'd (27MB/layer). The old host qkv 122MB D2H + q 81MB H2D + CPU rope are fully eliminated
  - **Performance** (13,252-token cold): **qkv_proj 4,511ms → 0.7ms, rope 2,754ms → 0.4ms, PROFILE TOTAL 20.4s → 12.7s, wall 24.5-24.7s → 17.6s**. Remaining bottlenecks: down 5.0s (39%) / attn_core 4.7s / rmsnorm 2.3s. maxabs unchanged at 1.79e-7, GPU cuda_ **23/23** (new rope_qkv_pf bit-exact test), response correct ("OK")
  - **Test infrastructure fix**: the test-only GPU mirror key collided with real KvCache kv_ids (global sequence), polluting other tests (failed only in full runs) → switched to the sentinel key `u64::MAX-1`
  - **Infra fix (denno-server)**: the PROFILE logger emitted empty blocks ~250k times/sec while idle, growing the serve log **~470GB/hour** (3rd disk-full incident). A timed-out investigation grep also kept holding the deleted 506GB log, adding ~700GB of invisible usage (df−du gap; recovered via lsof +L1 → kill -9). Root fix: PROFILE output limited to one line after the last session completes (`done.iter().any(|&d| d)` gate)
  - **PTX**: sha256 `21fc893194c8f68ad2b11311e471dc55b9558772d5332aea83feb32e18cecfdd`
- **Step 6: GEMM kernel improvement — 128×128 tiles + smem stride padding (✅ 2026-10-14)**:
  - **Problem with the old kernel**: with 64×64 tiles and a 32B smem row stride, the **sB row reads (adjacent threads on different rows) all hit the same bank and serialized 16-way** (32B stride is a multiple of 32 banks), and compute density between syncs was low (16 dp4a/thread/chunk)
  - **Fix**: **128×128 tiles, 8×8 outputs per thread** + smem stride padded to `GEMM_TK+1` (33B — adjacent threads' 8-row gap = 264B = 66 words → banks 2 apart, conflict-free) + A re-reads halved. **Integer accumulation structurally guarantees bit-exactness** (dp4a and the f32 epilogue are identical to the old kernel)
  - **Performance**: GEMM bench (m=8000): down 125ms → **48.1ms (2.6x)**, qkv 65 → 33.5ms, gate_up 508 → 260ms, bit-exact 8 cases OK. **E2E (13,252-token cold): down_proj 5,009ms → 1,147ms (4.4x), o_proj 494 → 272ms, PROFILE TOTAL 12.7s → 8.4s, wall 17.6s → 13.0s**. GPU cuda_ **23/23**, response correct ("OK")
  - **PTX (kernels_int8)**: sha256 `a008361079c039166edcc68219d256fba8f97d8506f649ab271c11ed87ff2c51`
- **Step 7: prefill attention kernel optimization + host constant mismatch fix (✅ 2026-10-14)**:
  - **Critical latent bug (since step 4)**: host constants `ATTN_PF_BQ/BK` were still **16/32** while the kernel had moved to 48/8. (1) the grid was `ceil(n/16)` — **~4x too many blocks**, with i0 >= n invalid blocks running the full time loop for nothing; (2) dynamic smem was computed on the 16/32 basis, and **when it fell below the kernel's requirement the kernel wrote out of bounds → error 700 (illegal address)** — the true cause of the long-standing "~43KB+ triggers error 700" mystery. Unified to 64/8
  - **Kernel changes (kernels_elem.cu)**: (4) software pipelining of K/V tiles (next tile prefetched into registers, hiding global-load latency under compute; numerically unchanged), (5) 4-way unrolled score dot (independent chains, fixed combine order), (6) **2r×2k quad dot** (each thread owns 4 pairs, sharing q/k smem loads → score-phase smem reads halved from 2 to 1 per FMA). (4)+(5) alone gave only ~1% — **the kernel is smem-bandwidth bound**; (6) took the bench from 90.2 → **69.3ms/call (1.3x)**. A 2×dh (256-thread) block was tried with no effect (confirming the bandwidth bound)
  - **Performance** (13,252-token cold): **attn_core 4,517ms → 3,514ms, PROFILE TOTAL 8.4s → 7.4s, wall 13.0s → 12.2s / 11.96s (two cold runs)**. Bench (n=t=13252, 14 heads): 112.4ms → **69.3ms (1.62x)**. Response correct ("OK", input 13252)
  - **Tests**: GPU cuda_ **24/24** (new `cuda_attention_prefill_vs_cpu_large` for large n — n=2056, multiple row-blocks × many tiles, approximate match + determinism. The n=37-only tests left the prefetch nearly idle: a 1-block launch). maxabs 1.79e-7 (n=37) / 1.97e-7 (n=2056), **determinism preserved**
  - **Lesson (workflow)**: after a source edit, the compiled PTX sha256 came out **identical to the previous build** (the sync hadn't landed) and the host/kernel constant mismatch shipped → E2E produced non-deterministic garbage (the grid computed only part of the rows, leaving uninitialized att to flow downstream). **Syncs now require md5 verification on both sides**
  - **PTX (kernels_elem)**: sha256 `efc3a7113af791aaf86ddf0110f82a36424879e58b380a0c5d31bc330eababaa`
- **Remaining candidates (step 8 onward)**:
  1. **attn_core 3.5s (remaining 47%)**: reduce v re-reads in the V accumulation (row chunking); larger BK (>48KB smem needs opt-in via `cuFuncSetAttribute`)
  2. **GPU rmsnorm**: rmsnorm 2.3s (~20ms/layer) sits serially before the h upload (81MB H2D). Generating h on device in the prefill path would remove that upload too (elementwise ops are already bit-exact)

### P19: Qwen3.5 support (GGUF loader + Gated DeltaNet hybrid) through B2 model integration + E2E ✅ (2026-10-14, commits `49ac073`/`0bbfa79`/`cb9d30b`/`feab1c8`)

> Origin: supporting Qwen3.8-27B (UD-IQ2_XXS mixed-quant GGUF) → stepwise implementation of the Qwen3.5 hybrid architecture (**Gated DeltaNet 3:1 + full attention**). Stages: A = GGUF+dequant foundation → B = GDN CPU reference → B2 = model integration + E2E (this section) → GPU offload → 9B → 27B scale.

- **GGUF foundation (`49ac073`)**: ① GGUF container reader (`gguf.rs`, mmap, header/KV/tensor info) ② dequant for **18 GGML k-quant/IQ types** (`kquant.rs` + generated `kquant_tables.rs` via `ref/gen_kquant_tables.py`) ③ **bit-exact verification vs llama.cpp** with an LCG fixture (16 blocks × 18 types, all matching) + real-GGUF verification over 320 tensors. The table file is strictly generated (no manual edits). The initial commit failures were caused by `.gitignore`'s `data/` rule excluding the fixture (fixed with negation patterns)
- **GDN CPU reference (`0bbfa79`/`cb9d30b`)**: 1:1 port of llama.cpp `delta-net-base.cpp` `build_delta_net_autoregressive` (sequential AR form). Math: g = ssm_a (negative) × softplus(alpha + dt_bias) → S ← S·exp(g) → delta rule (sk=Σk·S, d=β(v−sk), S+=k⊗d, o=Σq·S), q/k L2-normalized, output = gated RMS norm (rmsnorm(o)·w·silu(z)), causal depthwise conv + SiLU. softplus follows the ggml convention. **Bit-exactness vs llama.cpp is not required** (the denno contract is same-input→same-output + determinism; the sequential AR form is mathematically equivalent to the chunked form)
- **B2 model integration (`feab1c8`)**: `model/qwen35.rs` (`model_type: qwen3_5` dispatch) — 24 layers = 18 GDN + 6 full attention (3:1) · full-attention layers use fused Q+gate (attn_output_gate) · partial MRoPE (interleaved [11,11,10], equivalent to standard RoPE for text) · QK-Norm · sigmoid output gate · tied embeddings (vocab 248,320) · **GGUF (incl. Unsloth Dynamic mixed quants) → dequant → re-quantize into the existing Fp32/Int8 pipeline** (no new GEMM needed). Large matrices use the existing `Linear`; tiny f32 tensors (conv1d/ssm_a/ssm_dt/ssm_norm) go through a `GdnCore<'a>` reference view + `forward_token_core`. GDN state lives in per-layer GDN slots of the KvCache (conv buffer + SSM matrix)
- **Three bugs found and fixed during implementation**:
  1. **KV capacity bug**: `cache.len()` does not advance until forward() finishes, so per-token `kv_bufs(li,1)` panics out-of-range on the 2nd token → pre-allocate capacity for all tokens at the top of forward()
  2. **Recurrent state × prefix reuse** (most important): KV live-reuse `truncate` / prefix-entry restore invalidates GDN conv/SSM state (recurrent state is not position-sliceable) → measured **nondeterminism where the same seed produced different outputs**. Added `Model::is_recurrent()`; recurrent models reuse KV only on pure extension (lcp == current length), otherwise fall back to full prefill
  3. **int8 quality degradation on control signals**: a per-tensor quantization-error probe (test included) showed all tensors at 0.8–1.5% (normal range) with the worst on `ssm_beta/ssm_alpha` (control signals passing through sigmoid/softplus) → these are **kept Fp32 even in Int8 mode** (< ~1MB across 24 layers, negligible cost)
- **Tests**: gdn lib 4/4 (independent oracle match, batched-prefill vs token-by-token bit-exact, determinism) · real-GGUF layer-0 (Q8_0/UD-IQ2_XXS both, decay exp(g)=0.0086–0.9995 confirming multi-timescale behavior) · `qwen35_model` (load/forward/determinism 2-run bit-exact/**cache continuity forward(3)+forward(1)==forward(4) bit-exact**) · lib 114 passed
- **E2E verification** (GPU dev environment, both fp32 and int8; UD-IQ2_XXS is the default pick so the mixed-quant path is the production path): Japanese QA (correct answer), English haiku, 307-token summarization all coherent · greedy (temp=0) correct on both int8 and fp32 · **identical output across 3 consecutive requests at seed 42**. prefill ~90ms/token, decode ~20ms/token (CPU token-by-token) → GPU offload is the next bottleneck
- **Next**: GDN/prefill GPU offload → 9B (untied) → 27B

## 4. Candidate List

### Qwen3.5 main line (final goal: Qwen3.8-27B on UD-IQ2_XXS)

- [ ] **GDN/prefill GPU offload** (top priority): currently CPU token-by-token at prefill ~90ms/token, decode ~20ms/token. First port the sequential AR kernels, then consider a chunked-scan form
- [ ] **Qwen3.5-9B real E2E**: 32 layers = 24 GDN + 8 full, untied, nearly identical head layout (model download needed). Final validation before 27B
- [ ] **Qwen3.8-27B scale**: 64 layers = 48 GDN + 16 full, int8 required. GPU offload is a prerequisite (CPU sequential prefill exceeds practical time)
- [ ] **GDN state snapshots in the prefix cache** (optional): recurrent models currently always full-prefill (correctness first). Speeds up multi-turn chats
- [ ] **rmsnorm GPU offload** (P17k leftover, 2.3s) / **attn_core V accumulation** (leftover 3.51s, largest op)

### Earlier candidates

- [x] ~~**sampler optimization**: softmax + top-p sort over the 152k vocab cost a fixed ~3.5ms/step (found in P7.5)~~ → **done in P8** (0.66ms/step, greedy unchanged)
- [x] ~~**P6 resume: true parallel decode**: batched matrix form for concurrent session decode~~ → **done in P6** (slot pool + forward_batch, fp32 1.35× / int8 1.7× @b=4). **Remaining: b=4 shifts to compute-bound, so k-direction order-preserving SIMD widening** (SDOT etc., pairwise adds that keep accumulation order) could further improve batch efficiency
- [x] ~~**Batch efficiency improvement (P6 addendum)**: k-direction SIMD of `matvec_rows_bt` (order-preserving)~~ → **done in P9**. What was implemented was not k-direction but **i-direction (batch-lane) parallelism** (k-direction SIMD is impossible in fp32 because accumulation order changes; i-direction keeps each element's k accumulation sequential while computing 4 sessions simultaneously = bit-identity preserved). Kernel 3.9–5.9×. Per the profile, the next e2e bottlenecks are **average batch width drop (completion skew) + attention (per-session processing) + memory-system contention**
- [x] ~~**Attention batching (P10 candidate)**: `forward_batch`'s attention core was a per-session sequential loop~~ → **done in P10** (rayon parallel over session × KV head, bit-identity preserved). Effect: **aggregate +4%** (interleaved A/B; attn_core was 5.5% in batch, matching the theoretical cap). softmax's max subtraction/sum are session-independent so determinism held under batching too
- [x] ~~gpt2 exactness test failure (garbled text)~~ → **done in P11** (not fixed by re-downloading. Causes: ①the test was skipped in CI so the baseline went stale ②streaming detok byte-split character bug. Baseline updated to match the HF transformers reference + detok hold/flush + CI pulls gpt2)
- [x] ~~**Remaining bottleneck analysis**: from the P9 profile — ①average batch width drop ②in-server memory contention ③prefill interruption~~ → **done in P12** (quantified in a clean environment. fp32 scales to W=8 / int8 tops at W=4. int8 per-op: gate_up 29.5% / attn 19.4% / down 18.0% / lm_head 10.8%. Attention overhead (V-side merge + copies + allocations) reduced in P12. Fundamental scores-side optimization is incompatible with bit-exactness — the reference-update decision remains)
- [x] ~~**Inference visualization (denno-trace) P13: telemetry foundation**: fully decouple observation from the inference process (details in Architecture Guide §11). `denno-telemetry` crate (versioned flat format, f16/varint, Gap frames) + denno-side capture (L0 flow / L1 statistics: prof hooks, phase transitions, sampler top-k, KV usage, output tokens) + WS endpoint (`/v1/telemetry/ws`) + denno-view server skeleton. Principles: ①visualization not inside inference (denno only streams) ②bounded channel + drop-allowed so inference never waits ③OFF by default + read-only capture compatible with bit-exactness (all-green exactness as the gate)~~ → **done in P13** (commit `69663e6`. ON/OFF outputs fully identical, e2e Gap 0, JS decoder all 10 kinds matched. Details in the P13 section)
- [x] ~~**Inference visualization P14: viewer (L0/L1)**: flow diagram, token stream, KV usage grid, sampler top-k view, replay recording + scrubber~~ → **done in P14 (JS version first, commit `8befe3f`)**: the user-specified animation UI (model-driven program + implementation-code panel + layer panel + KV/variables + built-in chat). **WASM port and replay scrubber remain** (details in the P14 section)
- [ ] **Inference visualization P18: L2 tensor streaming** (formerly P17 — CUDA optimization took P17, so this moves down): focus protocol (viewer specifies session × layer × tensor kind; denno streams only that combination). Hidden-state heatmap, attention probability map (which past tokens a token attends to), raw K/V viewer. Budget: ≤5% for the focused session (f16 + narrowing)
- [ ] **b=4 measurements for 3B/7B**: measurement only. If still bandwidth-bound, wider batches should scale linearly (measure with interleaved A/B). Requires pulling large models into the cache (`denno pull`)
- [x] ~~**CUDA optimization**: the analysis is in Architecture Guide §8 (linear→cuBLAS, attention fusion, sampler kernel, paged KV). The int8 route can keep CPU/GPU bit-identity. P6's slot pool + forward_batch structure is isomorphic to GPU continuous batching and ports directly~~ → **step 1 done in P17** (commit `11281dd`: int8 linear-layer kernels implemented **bit-exactly** against the CPU int8 path, verified on an RTX 5090, lm_head 18.6×. Hand-written kernels via dlopen + the Driver API (zero deps) rather than cuBLAS, so exactness is guaranteed structurally). **Engine integration done in P17b** (commit `197a06c`: uid-keyed resident-weight registry + pinned async pipeline + decode-only offload; 3.4× e2e with fully identical responses — details in the P17b section). **Fused qkv/gate_up GPU calls done in P17c** (commit `7bf9eb0`: linear layers sharing the same input fused into one call, e2e 4.96× — details in the P17c section). **Elementwise GPU kernels + decode chain done in P17d** (commit `121aa3a`: glibc expf replicated exactly including the FMA contraction sites, device-resident hidden-state pipeline — details in the P17d section). **KV switched to demand-based allocation in P17e** (commit `c889336`: slot preallocation ~7.5GB → 0 at startup, slots 4→32, install-time reservation hard guarantee — details in the P17e section). **Attention kernel fusion / paged KV remain** — details in the P17 section)
- [ ] **GitHub push automation**: direct container push after deploy key registration
- [ ] tool use (Anthropic API tools parameter)
- [ ] Llama-family variants (models without GQA, RoPE scaling)
- [ ] **Other model families**: DeepSeek series (MoE + shared experts), GLM series, gpt-oss (attention sink + sliding window + MXFP4). ~~Qwen3 (QK-Norm)~~ → **done in P15** (thinking-mode switch and MoE variants not covered)
- [ ] **Visualizing each family's unique mechanisms**: MoE routing (which experts fired per token + router probabilities), MLA latent-compressed KV (cache-size comparison vs regular MHA), attention sink (attention concentrating on the first token), sliding-window band pattern (attention mask visualization), thinking-mode token flow
- [ ] sliding window attention (long-context KV memory reduction)
- [ ] multi-model serving (several models in one process)

## 5. Lessons (Design Decisions on Record)

- **Determinism first**: every optimization/parallelization is verified against the reference output for "same input → same output". Now CI-enforced (verified on x86 too)
- **A test skipped in CI is the same as never having run** (P11 lesson): `#[ignore]` + early return on missing model cache means a stale baseline goes unnoticed forever. Update baselines by checking against a reference implementation (HF transformers etc.) and add model pulls to CI so the test actually runs
- **matrixmultiply picks SIMD kernels by stride/row count, changing the numbers** (P12 lesson): optimizations that change GEMM shape/stride can only be adopted after bit-identity is confirmed by measurement (to_bits comparison). The scores side (transposed layout) fails both row-merge and stride; the probs@V side (row-major) passes both — the same library behaves differently per path. Keep the unit tests forever
- **Measurement script lessons** (P12): ①pkill patterns must match the actual command line (`denno serve` doesn't match `/tmp/denno_old serve` → all 12 A/B runs hit the same server — invalid data). Confirm the server identity each time via the startup `quant mode` log ②`tail -n` on per-op profiles truncates the top (largest op) line — reading shares without the largest op misleads
- **Adding observation hooks to always-polling loops causes floods** (P13 lesson): the scheduler worker keeps running `step_chat_batch` on empty batches while idle, so emitting KvUsage regardless of batch content produces thousands of frames/sec → hub full → Gap flood. **The emission condition must be "a meaningful event happened"** (no emission on empty batches) — a telemetry design principle. After adding any observation, always check frame rates in a hardware e2e
- **Hand-rolled f16 pitfalls** (P13): decoding subnormals is easier to get wrong than encoding (RNE rounding/subnormal/Inf/NaN) — the exponent was off by 2^14 and 6.1e-5 mangled into 0.999. Known-value tests need **both directions**: "expected-value table for encode" + "known values for decode (0x0001/0x0400/0x03FF)". Round-trip (relative error) tests catch one-direction bugs but subnormal round-trip error is large, so known values are the reliable check
- **CI lessons**: `target-cpu=native` is dangerous on shared CI (rust-cache shares artifacts across runners; SIGILL on different CPU generations). Use a fixed target on CI (x86-64-v3); native optimization only on self-hosted production builds
- **Mutex starvation lesson**: a thread acquiring a std::sync::Mutex (unfair) in a near-continuous loop can starve other threads for hundreds of ms. Cache immutable data referenced by handlers outside the lock
- **The reality of quantization**: "quantize weights only, multiply in fp32 with dequant" doesn't get faster (proven in P3). Dynamic activation quantization + true integer ops + SIMD is what delivers the big speedup (P5, ~3.5×). Integer i32 accumulation is associative, so SIMD keeps determinism
- **Profiling inside containers**: perf (kernel restriction), valgrind (SIGILL on aarch64 native code), gdb (ptrace forbidden) are all unavailable → lightweight self-instrumentation (`prof.rs`) is the optimal answer. env-var gating + no touching the numeric path keeps exactness intact
- **Agent-driven development operations**: sessions break at a 40-turn cap, so alternating "implementation (agent) → verification/commit (agent or human)" is stable. The test suite makes handovers easy. **Never run multiple agents on the same working tree simultaneously** (P6 conflict lesson)
- **Performance measurement on a shared host requires interleaved A/B (P10 lesson)**: other resident inference workloads use the same host's CPUs + the production server always runs, so background load varies a lot by time of day (the same code measured 8.4s↔16s). **Distant before/after comparisons are invalid** — rebuild new/old alternately and measure the same workload, comparing averages. A naive comparison that looked like a "2× improvement" was +4% in interleaved A/B (script `p10_ab2.sh`)
- **Prefix cache design**: the key is "pick the entry with the largest LCP" (picking the most recent always misses in alternating sessions)
- **Batching and determinism (the core of P6)**: fp32 sgemm changes accumulation order → not bit-identical, so "make each element's k-direction accumulation order in the multi-column matvec identical to a standalone matvec" is how the exactness contract is realized. If the sampler gives each session an independent RNG, batch composition has zero effect on each session's output (verified on hardware: 4 sessions solo==batch). **Batching helps bandwidth-bound decode, but growing b shifts to compute-bound** (weight stream once, FLOPs × b). Extending the factor requires k-direction SIMD widening, which coexists with the "order-preserving" constraint just like i32 accumulation (P5)
- **Order-preserving SIMD is possible in fp32 (the core of P9)**: k-direction lanes are impossible, but **splitting lanes along i (batch) and j (ILP) keeps each element's k accumulation sequential** → NEON/AVX2 gives 4× parallelism while bit-identical. Key points: ①avoid FMA and use separate `vmul`+`vadd` (keep product rounding independent), ②transpose activations to [k][4] for contiguous lane loads, ③avoid dependency-chain stalls with ILP across "independent chains of multiple output channels" (splitting d across multiple accumulators would change order — not allowed). int8 is order-free anyway thanks to i32 associativity (P5's SDOT applies as-is) — only fp32 carries this special constraint
- **Rust NLL pitfall**: `Vec<&mut dyn FnMut>` (a Vec of reference-typed elements) extends the borrow region to the implicit drop position → the simplest fix is an API that takes `Box<dyn FnMut>` directly. When sharing raw pointers with rayon, edition-2021 disjoint closure capture captures per-field (`cp.0`) and raises E0277, so capture the whole struct via a method like `SendPtr::write(self, idx, v)`
- **Model porting walls**: if the architecture is the same (qwen2), models differing only in dimension run unchanged. Walls are checkpoint format (sharding) and weight sharing (tied/untied) — "loader-side differences" (proven in P7)
- **Production/dev separation**: production on port 8080 (denno-beta container), dev testing on 8090. Production is not stopped for long periods
- **Known model quirks**: "1たす1は?" (kanji numerals) produces broken output — a model weakness, not a bug. "1+1は?" is fine
- **KV preallocation is determined by slot count × max context** (P15 lesson): Qwen3-0.6B reaches ~9.4GB per KV slot even with GQA (kv_heads 8) — 28 layers × 40960 ctx — and tried to allocate ~47GB (4 slots + prefix), dying with SIGKILL (exit 137). It wasn't OOM but preallocation — when "memory seems available but the process dies", suspect KV preallocation. `resolve_kv_budget` (pure function + unit tests) keeps the footprint within budget while explicit env vars take priority
- **HF chat templates depend on Python str methods** (P15 lesson): Qwen3's template uses `startswith` — minijinja's unknown-method fallback fails silently, so it's easy not to notice the template isn't being applied. The official fix is registering `minijinja_contrib::pycompat::unknown_method_callback` via `set_unknown_method_callback`. **Verify template rendering by checking the first token of the output** (a thinking template starts with `<think>`)
- **int8 associativity enables CPU/GPU bit-identity** (core of P17): i32 accumulation is independent of addition order, so no matter how the GPU splits warps/lanes/tiles, it matches the CPU SIMD kernels bit-for-bit. What must match: ①the same quantization rule (maxabs involves no rounding so its order is irrelevant; CUDA `roundf` = Rust `f32::round`; the ±127 clamp identical) ②the associativity of the scale multiplies (`(float)dot * ascale * scale[j]` left-to-right). fp32 changes with reduction order and has no such guarantee — per §8.3.5, int8 is the definitive GPU exactness route. **To use CUDA with zero dependencies: dlopen + the Driver API** (no link-time dependency; builds without the feature are untouched in CI / GPU-less environments). Commit the kernel as compute_75 PTX and let the driver JIT handle it (no rebuild needed even on new archs like sm_120). **The first wall of GPU porting is not kernel throughput but the fixed overhead of small ops** (launch + sync + H2D/D2H ≈ 40µs per call) — a device-resident pipeline (one sync per step) is the mandatory design at integration time
- **A green GPU test is not evidence the GPU ran** (P17c lesson): CUDA runtime errors are **silently absorbed** by GPU_BROKEN→CPU fallback, so a "GPU result vs CPU result" comparison test can pass as CPU-vs-CPU without the GPU ever running. Tests that must exercise the GPU path should `expect` on the returned None. Also, **the CUDA current context is thread-local** (libtest spawns a new thread per test even with `--test-threads=1`), so a shared context must call `cuCtxSetCurrent` at every GPU entry point — setting it once on the initializing thread makes cuMemAlloc fail on other threads with error 201
- **Debugging examples (early days)**: sgemm stride design mistake → SIGSEGV; discovery of Qwen2's QKV bias (optional load); Hermes' `max_tokens: 65535` clamp; missing SSE `event:` lines causing infinite retries; idle disconnects from missing keep-alive during prefill
