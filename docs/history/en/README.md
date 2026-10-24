> Japanese original: [../README.md](../README.md)

# denno-beta development history (index)

Start here and walk into the volumes. They are ordered so that **read top to bottom, they show what was built,
where it got stuck, and how it was made faster**. Inside a volume the order is chronological; across volumes the
order runs "features → models → performance → quality → verification".

- The volumes are the readable text. **The numbers and the narrative are authoritative in the volumes.**
- The old `docs/HISTORY.ja.md` / `docs/HISTORY.en.md` (4,007 lines, stopping at P41) were folded into these
  8 volumes and **deleted**. The raw commit-by-commit log can still be read out of git history (point at the
  parent of the commit that deleted them).
- All 137 phases, date-ordered with their commits, are in [`90_phase_index.md`](90_phase_index.md).

## Volumes

| Volume | What it tells you | Main phases |
|---|---|---|
| 1 [`01_bootstrap.md`](01_bootstrap.md) | Taking a CPU-only engine to usable speed. Why bit-identical output became the contract | initial work → P12 |
| 2 [`10_api_features.md`](10_api_features.md) | The Anthropic Messages–compatible server. tools / streaming / prefix cache / stop conditions | P11, tool use, s44–s54 |
| 3 [`20_models.md`](20_models.md) | What "supporting a model" actually means. GGUF, Qwen3.5, 27B, Q4_0, kquant | P3–P48 (model work) |
| 4 [`30_decode_perf.md`](30_decode_perf.md) | Removing *waiting* from one-token-at-a-time generation. D2H, launch fusion, gemv bandwidth | P8–P20 |
| 5 [`40_prefill_perf.md`](40_prefill_perf.md) | Removing *round trips* from reading the prompt. Phase accounting, chaining, GEMM tiles | P17b–P52 |
| 6 [`50_gdn.md`](50_gdn.md) | Gated DeltaNet: the maths, the kernels, the numbers. Where bit-identity was given up | P19, s61c, s192 and others |
| 7 [`60_quality.md`](60_quality.md) | Measured isolation of broken output. Fake turns, repetition collapse, thinking modes, stop conditions | s53–s56b, P4-1/P4-4 |
| 8 [`70_infra.md`](70_infra.md) | How it was measured, and when a number can be trusted. Anchors, cold A/B, measurement windows | everything from P7.5 on |
| Appendix A [`90_phase_index.md`](90_phase_index.md) | Every phase in date order (machine-generated index) | 137 phases |
| Appendix B [`80_licenses.md`](80_licenses.md) | Ledger of the external implementations consulted and their licences | P19 |

## The whole picture

denno is an LLM inference engine written in Rust. It keeps a CPU reference implementation and CUDA kernels on
**the same numeric path**, with a contract that moving to the GPU does not change the output
(`../../NUMERIC_POLICY.ja.md`). It targets the Qwen3.5 family (Gated DeltaNet hybrids) and runs 9B int8 and
27B q4 from the same binary. The API is Anthropic Messages–compatible and is used from a local desktop app.

Only three prerequisites are needed to read this.

- **prefill** (reading the prompt) and **decode** (generating one token at a time) are different jobs with
  different bottlenecks. Always be conscious of which one a number belongs to.
- An **anchor** is an md5 baseline that proves "same input, same output". Change the numeric path and the
  anchor moves. Deciding whether it is allowed to move is the centre of this development.
- The **cold / warm** distinction. The second run of the same prompt hits the prefix cache, so every speed
  claim must be checked as either "a single cold run" or "a relative comparison within one session".

## Reading order

1. **These three first**: `01_bootstrap.md` (what was built) → `20_models.md` (what was made to run) →
   `30_decode_perf.md` (how it was made faster).
2. **Only want performance**: `30_decode_perf.md` and `40_prefill_perf.md`. The table of numbers is in
   `../../KPI.md`.
3. **Only want the catalogue of stumbles**: the `## Rejected approaches` section at the end of each volume,
   in order. That is where the density is.
4. **Interested in quality and failure modes**: `60_quality.md` (fake turns, repetition collapse, thinking
   modes, stop conditions).
5. **How it was measured**: `70_infra.md` (profilers, anchors, cold A/B, running measurement windows).
6. **Want a commit**: [`90_phase_index.md`](90_phase_index.md) (all 137 phases, date-ordered, with commits).

## Timeline (by arc)

Four eras, each with its **representative arcs** one line at a time. `Effect` lists only the KPI that
**moved** in that arc. `—` means no recorded number. The denominators (model, quantisation, token count) are
in the body of each volume.

Every one of the 137 phases, one line each in date order, is in [`90_phase_index.md`](90_phase_index.md).

### Era 1: Standing it up (CPU → int8 → batching)

| phase | date | What was done (one line) | Volume | Effect |
|---|---|---|---|---|
| initial – P2 | –2026-10-11 | Engine skeleton, tokenizer, sampler, CLI and server | 01 | — |
| P3 | 2026-10-11 | int8 weight-only quantisation | 01 | — |
| P4 | 2026-10-11 | Continuous batching | 01 | — |
| P5 | 2026-10-12 | int8 SIMD kernels | 01 | — |
| P6 | 2026-10-12 | Batch-parallel decode | 01 | — |
| P7 | 2026-10-12 | Support for the large Qwen model | 01 | — |
| P7.5 | 2026-10-12 | Profiler and bottleneck visualisation | 70 | — |
| P8 | 2026-10-12 | Sampler optimisation | 01 | — |
| P9 | 2026-10-12 | SIMD for the fp32 batch kernel (order-preserving lane parallelism) | 01 | — |
| P10 | 2026-10-12 | Parallel attention (session × KV head) | 01 | — |
| P11 | 2026-10-12 | Streaming detok byte-split fix; exactness criterion updated | 01 | — |
| P12 | 2026-10-12 | Decode attention optimisation (probs@V row fusion) | 30 | — |

### Era 2: Going to the GPU (int8 kernels → residency → Qwen3.5)

| phase | date | What was done (one line) | Volume | Effect |
|---|---|---|---|---|
| P13 / P14 | 2026-10-13 | Inference visualisation (denno-trace / denno-view) | 70 | — |
| P17 | 2026-10-13 | CUDA int8 offload kernels (bit-identical to CPU) | 30 | — |
| P17b | 2026-10-13 | CUDA engine integration (resident weights + pinned async) | 30 | — |
| P17c–P17d | 2026-10-13 | GPU linear-layer fusion, elementwise ops on GPU, decode chain | 30 | — |
| P17e | 2026-10-13 | Demand-driven KV allocation (growable cache) | 30 | — |
| P17f–P17h | 2026-10-13 | scores-side optimisation, decode attention on GPU, rope on GPU | 30 | — |
| P17i–P17j | 2026-10-13–14 | Linear layers for long prefill / attention on GPU | 40 | — |
| P17k | 2026-10-14 | Prefill FFN made device-resident | 40 | — |
| P19 | 2026-10-14 | Qwen3.5 support (GGUF loader + GDN hybrid) | 20 / 50 | — |
| P19-B2.5 | 2026-10-14 | GitHub sync and CI fixes | 70 | — |
| P19-A1–A2 | 2026-10-14–15 | Qwen3.5 on GPU, GDN scan GPU kernel, 9B real E2E | 20 / 50 | — |
| A2.5 | 2026-10-15 | 9B output quality (repetition penalty) | 60 | — |
| A3 | 2026-10-15 | Scaling to Qwen3.8-27B | 20 | — |
| A1-r2 / A4 | 2026-10-15 | GDN chunked scan; GDN state snapshot in the prefix cache | 50 | — |
| B1–B3 | 2026-10-15 | Prefill rmsnorm on GPU, V accumulation in attn_core, rope on GPU for b>1 | 40 | — |

### Era 3: Breaking it, fixing it, making it measurable

| phase | date | What was done (one line) | Volume | Effect |
|---|---|---|---|---|
| 8095 GPU E2E regression investigation | 2026-10-15 | Record of investigating the 9B GPU prefill breakdown | 40 | — |
| tool use support | 2026-10-16 | Anthropic-compatible tools / tool_choice | 10 | — |
| 27B structure check, output-quality investigation | 2026-10-16 | Out-gate hypothesis refuted, MTP layers excluded, comparison against llama.cpp | 20 | — |
| GDN GQA head mapping fix | 2026-10-16 | Root cause of the 27B output breakdown identified and fixed | 20 / 50 | — |
| C4 | 2026-10-16 | Device-resident decode chain for qwen35 (GDN state GPU mirror) | 30 / 50 | — |
| tool_choice constrained decoding | 2026-10-16 | Token-level constraints for `any` / `tool` | 10 | — |
| Two C4 bugs | 2026-10-16 | segfault (write outside the pinned region) and non-deterministic output (realloc while in flight) | 30 | — |
| streaming tool_use | 2026-10-16 | SSE assembler | 10 | — |
| Interrupted-request hang audit, disconnect detection | 2026-10-16 | busy-spin fix, generation stops on abort, default max_tokens cap | 10 / 60 | — |
| epoch verification | 2026-10-16 | Root fix of the 2b two-value branch (content generation of the KV mirror) | 30 | — |
| Plan B (rayon prefill) | 2026-10-16 | Parallelising CPU prefill (bit-identity preserved) | 40 | 33.6 s → see the volume for the measurement |
| s44 | 2026-10-16 | Hardening post-disconnect handling for non-streaming requests | 10 | — |
| s46 | 2026-10-17 | Fixing the prefix cache that never hit once | 10 / 40 | 2nd conversation turn 1.41 s |
| s47 / s52 | 2026-10-17–18 | thinking off / brief / full, and accepting the map form | 60 | — |
| s48 / s49 | 2026-10-17 | Batching prefill full attention on GPU; fp32 lm_head on GPU | 40 / 30 | prefill 33.6 → 27.2 s (s48) |
| s50 / s51 / s54 | 2026-10-17–19 | Cutting decode D2H (GDN mirror as authority, deferred flush, split accounting) | 30 / 70 | the d2h_state breakdown became visible |

### Era 4: Rebuilding the measurement, then making it fast (the real prefill work)

| phase | date | What was done (one line) | Volume | Effect |
|---|---|---|---|---|
| s53 / s53b / s53c | 2026-10-19 | Stop conditions strengthened → misfire → fix → brief rebuilt | 60 | misfire 23 tok → normal 1,024 tok |
| s56 / s57 | 2026-10-19 | Three opt-in repetition controls | 60 | — |
| T3 / T5a | 2026-10-19 | Cutting off partial prefix reuse; prefill GEMM opt-in withdrawn | 40 / 70 | T5a regressed 2.85x by default |
| s61 / s61c / s62 | 2026-10-19 | via column measured; attn_gdn split; small-n projection defect fixed | 70 / 40 | gdn_proj 263 → 0.73 ms/call |
| s63 | 2026-10-19 | Prefill was limited by the result copy (`to_vec` removed) | 40 | 46.2 → 28.1 s, 123 → 202 tok/s |
| s64–s73 | 2026-10-19 | Inside a GDN call, transposed layout, direct D2H, chaining, parallel full attention | 40 / 50 / 70 | s68: 23.1 → 14.8 s and more |
| s74–s81 | 2026-10-19 | int8 GEMM tiles, shared-memory fix, gate+up concatenated at load | 40 | see the volume |
| s84 / s87–s90 | 2026-10-19 | rmsnorm measured (not optimised); profile accounting made additive; GDN 2-launch refuted | 70 / 50 | — |
| s96–s110 | 2026-10-19 | Dead GPU paths deleted; GPU-free regression (needle, 10-turn conversation); kernel invariant tests; cold A/B driver; dashboard | 70 / 60 | — |
| s118–s122 | 2026-10-19–20 | Speculative decoding, causal truncation, swiglu epilogue **refuted by measurement and closed** | 30 / 40 | see the rejected-approaches catalogues |
| s123–s127 | 2026-10-20 | Running 27B natively in Q4_0; moving production 8095 to 27B q4; fixing the VRAM creep | 20 / 40 | 18.4 GB / 180 ms/token (s123) |
| s151–s159 | 2026-10-21–22 | Profile accounting on the serve path (B12–B20); kquant TQ1_0 / IQ1_M; `denno info` detail | 70 / 20 | KPI 32.42 ms/token (B20 stage 1) |
| s163–s165 | 2026-10-22 | Q4 FFN → GDN layers → residual stream made device-resident (the real prefill work) | 40 | 40.10 → 13.87 s (3,161 tokens) |
| s168–s180 | 2026-10-22–23 | `denno info` quantisation types, P3-3 scanner extension, memory work (P35 / P37 / P38 / P42), measurement work (P40 / P43 / P41) | 20 / 70 | P42: host RAM −4,370.4 MB |
| P12 / P13 / P14 / P16 / P19 / P20 | 2026-10-22–23 | Prefill att round trip removed; GDN beta/alpha fused; FFN fused; decode beta\|alpha fused; full-layer q written straight to pinned | 30 / 40 / 50 | 13.55 → 9.59 s (3,161 tokens, 1fde630) |
| P17b / P49 / P52 | 2026-10-23 | q\|k\|v contiguous D2H regressed 2.2x and was withdrawn; GDN pass fusion rejected at +47%; q4 GEMM 8x8 adopted | 40 | 65k 38.17 → 36.26 s (P52) |
| P47 / P48 / P50 / P51 | 2026-10-23 | full_core tail analysis, q4 GEMM bandwidth, `denno info` layer count | 40 / 70 / 20 | see the volumes |
| integration and deployment | 2026-10-24 | The integrated P47–P52 build went to production (16k 8.44 s / 65k 34.95 s) | 40 / 70 | 16k 8.44 s, 65k 34.95 s |

> This arc table is the **skeleton of the story**. That no phase is missing can be checked against the
> 137 lines of `90_phase_index.md` (sum of each volume's `## Phases covered` = 7 + 13 + 27 + 25 + 41 + 16 + 8).

## Volume summaries

- **`01_bootstrap.md`** — Taking a CPU-only engine to usable speed with int8 quantisation, continuous
  batching, SIMD and parallel attention. The pre-GPU design decisions (why bit-identity became the contract)
  are here.
- **`10_api_features.md`** — The Anthropic Messages–compatible server. tools / tool_choice / streaming /
  disconnect detection / prefix cache / max_tokens. Also a record of how breakage was repaired.
- **`20_models.md`** — Getting Qwen3.5 (GDN hybrid) and the 27B to run. The GGUF loader, the head-mapping
  bug, native Q4_0, kquant, `denno info`. The volume that shows **what supporting a model really means**.
- **`30_decode_perf.md`** — One token at a time from 33.50 to 17.28–17.32 ms/token. Cutting D2H, fusing
  launches, gemv bandwidth. The theme is **removing waiting**.
- **`40_prefill_perf.md`** — Reading the prompt from 40.10 s / 162.3 s down to 9.59 s / 39.0 s. Building phase
  accounting, chaining, deleting round trips. The theme is **removing round trips**.
- **`50_gdn.md`** — Gated DeltaNet: implementation and performance. Why bit-identity is hard here and where it
  was given up. The negative result on local-memory boundedness (s192) is kept here.
- **`60_quality.md`** — When the output broke, narrowing down "under exactly which conditions does it break"
  by measurement. Fake turns, repetition collapse, thinking modes, stop conditions, needle recall, 10-turn
  conversation. The theme is the constraint **fix it without changing the defaults** (opt-in, no-op by
  default), and it includes the apparatus that runs quality checks without a GPU.
- **`70_infra.md`** — The history of measurement and verification. `DENNO_PROFILE` accounting, `GPU_VERIFY`,
  anchors, cold A/B, running measurement windows, the list of pitfalls. The volume that tells you **when a
  number can be trusted**.

## How to read the numbers

1. **Read the denominator first.** Model (9B int8 / 27B q4), token count, cold or warm. Never put numbers with
   different denominators side by side and call it "N times faster".
2. **prefill is official only as cold, nonce-carrying, median** (`scripts/prefill_bench.py`). The second run of
   the same prompt hits the prefix cache and cannot be used for comparison. 3,161 tokens and 12,850 tokens are
   different numbers.
3. **decode is ms/token.** Discard one warm-up run. Do not mix TTFT with generation speed.
4. **Separate estimated from measured.** In this index and the volumes, a value measured inside the container
   is written "measured (container)"; a GPU-side expectation is written "estimated". Never quote an estimate
   as a settled value.
5. **Do not quote `DENNO_PROFILE` / `DENNO_PF_DBG` percentages as KPIs.** PROFILE changes the thing it
   measures (with SAMPLE=1 it inflates to 2.7–4x the measured value). PF_DBG's phases are **a partition of
   wall time**, not of kernel time. For ratios, use GPU-side time from nsys.
6. **The absolute value of bench m=1 is L2-relative** (it reads the same buffer repeatedly). Decide with
   relative change plus the real-machine KPI.
7. **The anchor md5** is the evidence that "the output did not change". If the numbers improve but the anchor
   moved without explanation, treat it as a failure. If it must move, re-baseline explicitly following
   `../../NUMERIC_POLICY.ja.md`.
