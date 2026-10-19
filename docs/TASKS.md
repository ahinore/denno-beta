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

Estimates below were refreshed at s91 from the last full profile (cold, 5,686 tok,
`DENNO_PROFILE=1`, prefill wall 13.75 s) and are quoted as measured ms / share of the wall.
Round-tripping "Estimated" numbers from the s64-era profile are gone.

### P1-1: GDN scan residency / pass reduction
- Where: `cuda.rs` (`gdn_scan_chunked_impl`, `gdn_scan_batch_oproj`, `chain_*`), `kernels/gdn.cu`.
- Status: **closed (2026-10-19) - fold not implementable, measured zero gain** (s90 attempt
  `ddf7201`, closed in s91).
- Reason: the 2-launch fold needs the boundary row a chunk reads to be written by a
  *different* launch, which this forward cannot provide — the GDN layers are non-adjacent,
  and `cache.advance(n)` after every forward forces the state back to the host per layer.
  The scan's own cost is 1.99 s = 1.10 s sync wait (real GPU work, not idle) + 0.52 s H2D of
  activations, so no residency lever is left inside the scan.
- Measured: cold A/B, 5,686-token payload, 3 alternating rounds — OLD 13.362 / 13.408 /
  13.404 s vs NEW 13.345 / 13.377 / 13.375 s = **-0.03 s, no gain** (acceptance was
  >= 0.3 s faster). Anchors MATCH, decode unchanged at 16.4 ms/token.
- Remaining scan cost (`gdn_scan` 1,987 ms) is booked under P1-3, not here.

### P1-2: full-attention prefill kernel occupancy
- Where: `dno_attention_prefill` (kernels), `full_attn_attend_gpu`.
- Why: `attn_full` measures 942 ms (16 calls = 8 full-attention layers, two passes each),
  6.9 % of the wall. The T7b-2 failure showed this kernel is sensitive to how much data it
  has to re-read (attn_core went 4.0 s -> 10.1 s when the fused layout forced strided
  extraction).
- Idea: raise occupancy (smaller BK, more blocks), check L2 reuse of K/V across heads.
- Estimated: not yet re-derived from a per-kernel profile; 942 ms is the ceiling for the
  whole bucket. Risk: medium; the attention contract allows 1e-3 abs (R3).

### P1-3: d2h_state / sync reduction
- Where: `cuda.rs` chains, `prof.rs` `d2h_state`.
- Status: **closed (2026-10-19) - no separable lever in the cold path** (s92 inventory, no
  code change).
- Why the bucket is big: `d2h_state` measures 6,147 ms (38.9 % of the wall), the largest
  bucket. Children: `gdn_pf_sync` 1,095 ms / 96 calls, `gdn_pf_alloc` 521 ms / 48,
  `d2h_attn_pf` 491 ms / 8, `d2h_src_other` 491 ms / 8. Each sync serializes CPU and GPU.
- What the inventory found (s92):
  - `gdn_pf_alloc` 521 ms is *not* per-call device allocation on this path. The cold
    5,686-token request runs the chunked route (`n >= 2048`), which already reuses
    `GdnScratch` across layers since s67 (`s.ensure(...)` + `copy_from` only; no
    `DevBuffer::from_slice` in the per-layer region). What remains inside the bucket is the
    zero-fill plus the H2D of activations, i.e. real transfer time, so recycling an
    allocation cannot remove it.
  - `gdn_pf_sync` 96 calls = chunked 3 passes x 24 GDN layers (72, genuine dependencies
    pass1->pass2->pass3 and host reads) + the trailing `ctx.sync()` of
    `gdn_scan_batch_oproj` x 24 (waiting for the asynchronous D2H of `out`).
  - Batching the D2H per block is rejected: the prefill layer loop is strictly sequential
    (`qwen35.rs:1208`) and a full layer's attention output is consumed immediately by that
    layer's o_proj and the residual add, so accumulating several layers' outputs changes
    the layer inputs and breaks bit-exactness.
  - The single-token route (`n < 2048`, `gdn_scan_impl`) does build its 13 buffers per layer
    with `from_slice`. That is a separate task; it does not ride the cold measurement.
- Why the 1.6 s once looked separable: the s91 estimate treated `gdn_pf_alloc` as allocation
  cost; the s92 code read disproved it.
- Test counts corrected to what s92 actually ran (the older 250 / 37 could not be traced to
  a command, so both figures are kept with their command instead of guessing):
  - `CARGO_TARGET_DIR=/tmp/denno-target cargo test -p denno-engine` -> 242 passed / 0 failed /
    18 ignored (default build, no `cuda` feature; the GPU tests are the ignored ones).
  - `CARGO_TARGET_DIR=/tmp/denno-target cargo test -p denno-server` -> 77 passed / 0 failed /
    0 ignored.
  - `CARGO_TARGET_DIR=/tmp/denno-target cargo check -p denno-engine --features cuda --tests`
    -> exit 0.
- Remaining from P1-1: `GdnStateCache` (`cuda.rs:954`) has no wired-in call path as of s90.
  Decide whether to use it or delete it (tracked under P3).

### P1-4: swiglu / elementwise fusion
- Where: `dno_swiglu` + the residual/normalization passes.
- Why: elementwise work is now `gate_up_proj` 2,283 ms / 14.4 % (64 calls), `qkv_proj`
  1,534 ms / 9.7 %, `down_proj` 999 ms / 6.3 %, `rmsnorm` 413 ms / 2.6 %, `qk_norm`
  108 ms / 0.7 %, `o_proj` 175 ms / 1.1 %. Only `rmsnorm` is a pure elementwise pass in
  that list.
- Idea: fuse swiglu into the gate+up GEMM epilogue (the fusion already concatenates
  gate and up, so the epilogue has both halves in registers).
- Estimated: bounded by `rmsnorm` 413 ms until the projection epilogues are instrumented
  separately. Risk: low; bit-exactness needs the same multiply order.

## P2 — decode (separate axis)

### P2-1: decode 16.4 ms/token -> 12 ms
- Where: C4 device-resident chain, per-layer syncs, GDN state mirror.
- Why: decode is now the user-visible latency for chat; 5,686-token prefill is a one-off.
- Idea: remove the per-layer D2H for the GDN state (the s50 deltization already defers it),
  keep logits on device until the final token, then measure the true sync count.
- Estimated: 20–25 % of decode time. Risk: medium — the s50 work showed this area is
  easy to get subtly wrong (state authority).

## P3 — correctness and debt

### P3-1: remove or fix the rejected GPU gdn_proj kernel — CLOSED (s96)
- Status: **closed** for the Rust side. The opt-in (`DENNO_GPU_GDN_PROJ`) produced wrong
  values (max abs diff ~1e2; m=33 gives -17.205 vs -8.410; k=1 latches `LMV32_BROKEN`),
  so the whole path was deleted in s96: `cuda::gdn_proj_f32_into` and its helpers, the
  `GdnProjMiss`/`DENNO`-style plumbing, `model/qwen35.rs::gdn_proj_into_gpu` and its
  prefill branch, and the three tests that covered them.
- Default path: unchanged (beta/alpha always went through `apply_bt_rows_into` with the
  gate off; it is now called unconditionally — bit-identical).
- Remaining (GPU machine only): delete `dno_matvec_rows_f32` from `kernels/kernels_elem.cu`
  and regenerate `kernels_elem.ptx`. Nothing in Rust references it; PTX regeneration needs
  nvcc on a GPU host, so it is left as a human step.
- Profile ids `D2H_GDN_PROJ_*` / `ATT_GDN_PROJ` stayed reserved on purpose (renumbering
  would break historical profile comparisons).

### P3-2: T9 — why s73 moved two short anchors — CLOSED (2026-10-19, s101)
- Status: **closed (2026-10-19, s101) — not reproducible; cause is the s73 build's int8 GEMM**.
- Evidence (CPU path, not the GPU anchors): s73 (57405fe) and its parent 4119dce were built
  in release for the CPU path (GGUF `local/qwen35-9b-cpu`) and served the same anchor
  payloads; the three anchor text md5s are identical across both revisions
  (v6_1b 788221d7 / v6_2b f5603b67 / v6_1b_full 0184c481). These are CPU-path values,
  **not** the GPU anchor values in `data/verify/anchors_baseline.txt`.
- `DENNO_DEBUG_LAYER_CHKSUM=1` ([LCHK]) gave **136 / 136 identical lines** (emb, L00..L31
  attn+ffn, final).
- Hypotheses rejected: `kv_bufs(li, n)` has no side effect (`ensure(len+n)` only, no `len`
  or epoch advance; n calls == 1 call); `full_attn_qknorm` has no uninitialised tail on the
  exercised path (`k_row` is fully initialised, `buf_att` explicitly zero-filled since s63).
- The s73 -> s75 source delta is `kernels_int8.cu` + `cuda.rs` only, and s75 fixes s74's
  unwritten-row int8 GEMM defect. Conclusion: the drift came from that broken kernel (argmax
  flips on near-ties), not from the parallelised prepare. No code change needed.
- Remaining: the GPU-side re-measurement of "not reproducible" is owned by the main session
  (this container has no GPU).

### P3-3: extend the GPU-less kernel invariant tests
- Status: **implemented (s104/s105) — closed.** `tests/kernel_invariants.rs` now carries a
  smem-cast scanner over every kernel file. It builds a shared-memory table from
  `__shared__` / `extern __shared__` declarations plus the alias chain
  (`float *s_sm = v_sm + ..`; dynamic region -> cap = i64::MAX), then classifies every
  4-byte cast (`*(int*)&sA[r][c]` and `reinterpret_cast<const float2*>(&s_sm[..])`) as
  numeric-ok / symbolic-row-ok / known-bad / unresolved, plus a 2-D reference pass
  (row-stride times column alignment, out-of-declaration rows panic).
  Self-check: deliberately broken defines (stride 35, column 9, tail past the array) must
  still be reported, so a regex that matches nothing cannot stay green.
- Acceptance: the same static checks exist for every kernel that does smem casts -> met for
  `kernels_int8.cu` (6 casts), `kernels_elem.cu` (float2 casts), `gdn.cu`.
- Known limits (recorded, not fixed): the row bound of a symbolic row is not checked (only
  the column); row-alias form `const int *row = &sA[r];` does not exist in the tree and its
  assertion is pinned to 0; float2 casts through a dynamic alias with a symbolic column stay
  `unresolved` (alignment cannot be judged statically) — they are covered by
  `elem_float2_casts_are_aligned` instead; the scanner is text-based, so a cast written
  through an intermediate pointer variable is invisible.
- `NO .cu/.ptx change`: the task was tests only, and any real violation found must be
  reported to the main session (a .cu edit needs PTX regeneration + GPU_VERIFY).

### P3-4: fold the cold A/B into GPU_VERIFY
- Status: **implemented (s102) — one command prints old/new medians and the delta %**.
  `scripts/cold_ab.py` runs the alternating cold-single-shot protocol (server start ->
  one request with a cold prefix cache, old/new in alternating rounds), prints the per-round
  lines, a table, per-binary medians and `delta = (new - old) / old * 100`, and records the
  payload sha256 and the server-reported `input_tokens`. It reports only; the supervisor
  applies the ">= 0.3 s faster" acceptance rule.
- Action (original): `GPU_VERIFY.sh` gains a `--ab <old-binary>` mode that does the alternating cold
  rounds and prints the delta. Removes the human step and the unit mismatch that produced
  the inflated 8.71 s figure.
- Acceptance: one command produces "old A/B new, delta %, anchors PASS".
- s102: implemented as `scripts/cold_ab.py` (stdlib only; `--dry-run`, `--self-test`,
  `--json`). Tests: `python3 scripts/test_cold_ab.py` (32 tests, CPU only). GPU_VERIFY.sh
  stays gitignored and is not edited by the dev; wiring it in is a human step
  (snippet in REPORT_S102.md).

## P4 — productisation, never validated end to end

### P4-1: 9B long-conversation stability — CHECKER DONE (s103), real run is a human step
- Why: the original complaint ("collapses after a few turns") was traced to
  `temperature = 1.0` and fixed with `--temperature-cap 0.7`, but no multi-turn
  regression exists.
- Action: script a 10-turn conversation against 8095, check for repetition collapse and
  template fragments; record the transcript as a fixture.
- What exists now (s103): `scripts/long_conversation_check.py` drives 10 turns against an
  Anthropic-style `/v1/messages`, appending each assistant reply so the conversation
  actually grows, with detectors for repetition collapse (`repeat_line`,
  `repeat_sentence`, `long_line`), chat-template leakage (`template_marker`), fabricated
  user turns (`fake_turn`), empty replies (`empty`) and truncation (`max_tokens`,
  reported only). Exit 0 clean / 2 hard failure / 3 infrastructure; ports 8080 and 8090
  are refused; `--dry-run` and `--self-test` open no socket. Fixed prompts live in
  `scripts/fixtures/long_conversation_turns.json`; `--out` writes a transcript that
  `--replay` re-judges to the same verdict (the verdict is deterministic, the sampled
  text is not). 67 CPU tests in `scripts/test_long_conversation_check.py`.
- Cap regression (s107): the s97 checker (`longconv_check.py`) was retired in s107, but
  its `--expect-cap` was ported first, because P4-1's whole point is to fail when the cap
  is removed. Pass `--temperature 1.0 --expect-cap 0.7` to assert the serve clamps to the
  cap; a request or a server-accepted temperature above the cap exits 2. Nothing is sent
  when `--temperature` is omitted, so the default run cannot mask a regression.
- The exact command to run the 10-turn check is in `scripts/README.md`
  ("The long-conversation run (P4-1)"); `scripts/run_cpu_checks.sh` runs the CPU half.
- Still open: the real 10-turn run against a serve started with `--temperature-cap 0.7`
  on a GPU host is a HUMAN STEP (the agent container has no GPU and must not drive
  servers). Until that transcript exists, the cap itself is unverified end to end.
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

### P4-4: long-context regression test — CHECKER DONE (s98/s100), GPU_VERIFY wiring is a human step
- Why: the 4,671-token needle test was a one-off during the s73 investigation.
- Action: make it a script with a fixed payload and expected answer; run it in GPU_VERIFY.
- What exists now (s98/s100): `scripts/needle_check.py` builds a deterministic
  filler + needle payload from `scripts/needle_fixture.json`, tries the OpenAI-compatible
  `/v1/chat/completions` and falls back to `/v1/messages` on 404, and judges the reply by
  the expected passphrase. Exit 0 recall ok / 1 recall failed (regression) / 2 transport or
  parse failure, so an infrastructure problem is never mistaken for a recall failure.
  `scripts/needle_selftest.py` proves those three codes against a fake server on CPU.
  See `scripts/README.md` for the table of scripts and their exit codes.
- Still open, and it is a HUMAN STEP: wiring the checker into `GPU_VERIFY`. `GPU_VERIFY`
  is gitignored and is not edited by the agent, so the step is to add this line to it and
  run it on a GPU host:

      python3 scripts/needle_check.py --base-url http://127.0.0.1:8095 \
          --json-out /tmp/needle_<build>.json || echo "NEEDLE RECALL REGRESSED"

  A non-zero exit must fail the verify run: 1 is a real recall regression, 2 means the
  check could not run (do not record 2 as a pass).
- Acceptance: needle recall passes on every verified build.

## P5 — tooling and ops

### P5-1: dashboard shows cold A/B and decode
- Why: `pf_best.txt` now holds a cold number but the graph still mixes eras.
- Action: parse the A/B lines from the progress log, plot prefill and decode on separate
  axes, mark the protocol change with a vertical line.

### P5-2: docs sync — DONE (2212c4a, s101)
- `README.md` and `docs/api.md` still describe pre-s68 behaviour in places
  (round-trip counts, the old "PF wall" basis).
- Done in 2212c4a (s101): `README.md` + `docs/api.md` synced with the current implementation.

## Suggested order

1. P1-2 attention (942 ms bucket) -> P1-4 swiglu fusion. P1-3 is closed: the s92 inventory found no separable lever.
   (P1-1 and P1-3 are closed as of s91/s92: neither has a lever left in the cold path.)
2. In parallel on the CPU-only agent: P3-1 (delete the dead kernel), P3-3, P4-1, P4-4.
3. P2-1 decode once P1 has plateaued; P3-4 the moment a second A/B is needed by hand.