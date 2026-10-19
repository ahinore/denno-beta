# TASKS — denno-beta task queue

State at the time of writing: HEAD `b2b1343`, pushed to `origin/main`.
Prefill (5,686-token payload, cold A/B protocol) = **13.18 s / 431 tok/s**.
Decode = **16.4 ms/token** (gen2, 200 tokens).
Anchors: `c61ad85b` / `9e17d563` / `af6eb6ab`.

Measurement rules (see HISTORY "measurement-basis correction"):
cold single-shot (server start → one request, prefix cache off, alternating A/B) is the
official prefill number. Warm second runs are only for relative comparison inside one
session. Every GPU change must pass `GPU_VERIFY.sh` with the three anchors.

Every task below assumes: pathspec-only commits, one task = one commit, determinism is
absolute, and the numeric classes R1–R4 in `docs/NUMERIC_POLICY.*.md` are the only
permitted relaxations.

## P0 — in flight

### T8: rmsnorm cost and fusion
- Status: running (dev s84). Step 1 adds a profile bucket; step 2 optimizes only if the
  norm costs >= 0.25 s per prefill.
- Acceptance: bucket measured and recorded; if optimized, row-sum order unchanged and
  anchors MATCH; otherwise closed with the measurement as the reason.

## P1 — prefill, ordered by expected gain

Estimates come from the last full profile (s64-era, scaled for s68–s81 changes). T8's
bucket will refresh part of it; re-measure before committing to any number.

### P1-1: GDN scan residency / pass reduction
- Where: `cuda.rs` (`gdn_scan_chunked_impl`, `gdn_scan_batch_oproj`, `chain_*`), `kernels/gdn.cu`.
- Why: the scan was ~3.1 s of a 40.9 s profile and still moves state through host memory.
- Idea: keep the state on device across the whole GDN block; cut the 3-pass chunked scan
  to 2 or fold the final normalization into the last pass. Heads are independent, so
  head-parallel splits are numerically free.
- Estimated: 0.5–1.0 s. Risk: medium (kernel change -> PTX regen -> GPU_VERIFY).
- Acceptance: cold A/B >= 0.3 s faster, anchors MATCH, decode not worse than 3 %.

### P1-2: full-attention prefill kernel occupancy
- Where: `dno_attention_prefill` (kernels), `full_attn_attend_gpu`.
- Why: attn_core is the largest profiled bucket; attn_full was ~1.2 s. The T7b-2 failure
  showed this kernel is sensitive to how much data it has to re-read (attn_core went
  4.0 s -> 10.1 s when the fused layout forced strided extraction).
- Idea: raise occupancy (smaller BK, more blocks), check L2 reuse of K/V across heads.
- Estimated: 0.3–0.8 s. Risk: medium; the attention contract allows 1e-3 abs (R3).

### P1-3: d2h_state / sync reduction
- Where: `cuda.rs` chains, `prof.rs` `d2h_state`.
- Why: after s68–s72 the round trips collapsed, but the profile still shows sync-heavy
  buckets. Each sync serializes CPU and GPU.
- Idea: count syncs per layer, then batch the remaining D2H into one transfer per block.
- Estimated: 0.3–0.6 s. Risk: low-medium (transfer only, bit-exact expected).

### P1-4: swiglu / elementwise fusion
- Where: `dno_swiglu` + the residual/normalization passes.
- Why: swiglu was ~0.85 s; it is a pure elementwise pass over the activation buffer.
- Idea: fuse swiglu into the gate+up GEMM epilogue (the fusion already concatenates
  gate and up, so the epilogue has both halves in registers).
- Estimated: 0.2–0.4 s. Risk: low; bit-exactness needs the same multiply order.

## P2 — decode (separate axis)

### P2-1: decode 16.4 ms/token -> 12 ms
- Where: C4 device-resident chain, per-layer syncs, GDN state mirror.
- Why: decode is now the user-visible latency for chat; 5,686-token prefill is a one-off.
- Idea: remove the per-layer D2H for the GDN state (the s50 deltization already defers it),
  keep logits on device until the final token, then measure the true sync count.
- Estimated: 20–25 % of decode time. Risk: medium — the s50 work showed this area is
  easy to get subtly wrong (state authority).

## P3 — correctness and debt

### P3-1: remove or fix the rejected GPU gdn_proj kernel
- Status: `DENNO_GPU_GDN_PROJ` is an opt-in that produces wrong values (max abs diff ~1e2;
  m=33 gives -17.205 vs -8.410; k=1 latches `LMV32_BROKEN`).
- Action: delete it (preferred) or fix the grid/index plumbing and `av[n-2]` underflow.
  Dead code with a known-broken kernel is a trap for the next reader.
- Acceptance: code gone or fixed + tested; no default-path change.

### P3-2: T9 — why s73 moved two short anchors
- Status: open. The drift did not reproduce on the s75 build (anchors back to canonical),
  so it is not a shipping blocker.
- Hypothesis: s73's parallel prepare changed the number of `kv_bufs()` calls per layer
  from n to 1; if that call has a side effect (allocation, mirror invalidation) the
  prefill path may differ. Also check `full_attn_qknorm`'s returned buffer for
  uninitialized tail.
- Acceptance: a definite answer, or a documented "not reproducible" with the experiment
  list. No code change required if the anchors stay canonical.

### P3-3: extend the GPU-less kernel invariant tests
- Status: s75 added a test that parses `kernels_int8.cu` `#define`s and checks stride and
  block-shape invariants. `gdn.cu` and the elem kernels have the same class of risk
  (4-byte casts on shared memory, block vs tile coverage).
- Acceptance: the same static checks exist for every kernel that does smem casts.

### P3-4: fold the cold A/B into GPU_VERIFY
- Status: the protocol is documented in HISTORY but run by hand.
- Action: `GPU_VERIFY.sh` gains a `--ab <old-binary>` mode that does the alternating cold
  rounds and prints the delta. Removes the human step and the unit mismatch that produced
  the inflated 8.71 s figure.
- Acceptance: one command produces "old A/B new, delta %, anchors PASS".

## P4 — productisation, never validated end to end

### P4-1: 9B long-conversation stability
- Why: the original complaint ("collapses after a few turns") was traced to
  `temperature = 1.0` and fixed with `--temperature-cap 0.7`, but no multi-turn
  regression exists.
- Action: script a 10-turn conversation against 8095, check for repetition collapse and
  template fragments; record the transcript as a fixture.
- Acceptance: a repeatable check that fails if the cap regresses.

### P4-2: fp32 serve evaluation
- Why: the int8 path is measured, the fp32 path is not. The fp32 lm_head GPU work (s49)
  exists but no wall-clock number.
- Acceptance: cold prefill + decode numbers for fp32 at 9B.

### P4-3: 27B Q4_0 quality
- Why: 27B loads and runs deterministically but output quality (greedy collapse) is
  unexplained. UD-IQ2_XXS remains impossible (IQ1_M unsupported).
- Action: compare against llama.cpp on the same GGUF at the same seed, per-layer, to find
  where the collapse starts.
- Acceptance: either a fix or a documented root cause.

### P4-4: long-context regression test
- Why: the 4,671-token needle test was a one-off during the s73 investigation.
- Action: make it a script with a fixed payload and expected answer; run it in GPU_VERIFY.
- Acceptance: needle recall passes on every verified build.

## P5 — tooling and ops

### P5-1: dashboard shows cold A/B and decode
- Why: `pf_best.txt` now holds a cold number but the graph still mixes eras.
- Action: parse the A/B lines from the progress log, plot prefill and decode on separate
  axes, mark the protocol change with a vertical line.

### P5-2: docs sync
- `README.md` and `docs/api.md` still describe pre-s68 behaviour in places
  (round-trip counts, the old "PF wall" basis).

## Suggested order

1. T8 (in flight) -> P1-1 GDN scan -> P1-2 attention -> P1-4 swiglu fusion.
2. In parallel on the CPU-only agent: P3-1 (delete the dead kernel), P3-3, P4-1, P4-4.
3. P2-1 decode once P1 has plateaued; P3-4 the moment a second A/B is needed by hand.