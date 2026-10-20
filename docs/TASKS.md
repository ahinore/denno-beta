# TASKS — denno-beta task queue

State at the time of writing: HEAD `b2b1343`, pushed to `origin/main`.
Prefill (5,686-token payload, cold A/B protocol) = **13.18 s / 431 tok/s**.
Decode = **16.4 ms/token** (gen2, 200 tokens).
Anchors: `c61ad85b` / `9e17d563` / `af6eb6ab`.

**Plateau accepted by the operator (2026-10-20).** Prefill stays at ~13.2-13.4 s cold
(13.61 s measured on this host today, which runs ~3% slower) and decode at 16.4 ms/token.
Every P1 and P2 candidate is closed with a measurement, so **no new prefill/decode
performance work is queued**: new work on that axis needs an explicit operator decision
(it would mean speculative decoding plus an anchor rebase). Priority moves to P4 (product
questions) and to finishing the human GPU steps recorded under each task.

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
- Status (2026-10-20): **P1-2 fully closed - candidates 1-4 all rejected with numbers. No
  `.cu` edit survived; nothing awaits a PTX regen or a cold A/B.**
  - Candidate 1 (`__ldg` on the K/V reads, s112 `b5f1f37` + `78e7d08`): **-0.9 %, reverted**
    (`ac73729`); the bucket is not DRAM-bound, so a cache hint cannot pay.
  - Candidate 2 (tile/holding variants of the existing kernel, s116, no code change):
    a traffic + smem/occupancy model over (BQ 64, BK 16), (BQ 128, BK 8), (BQ 128, BK 16),
    (BQ 32, BK 16), plus BQ 256 and a one-block-per-kv-head loop. **No variant reaches the
    0.3 s acceptance.**
    - Geometry: 16 query heads / 4 kv heads / group 4 / dh 256, 8 full-attention layers;
      `dno_attention_prefill` runs grid `(n/64, heads)`, 256 threads, 84,800 B smem =
      2 blocks/SM, 58.9 ms/call.
    - K/V reads for one 5,686-token request = 273.2 GB against 186 MB of real K/V
      = 1,466x duplication, but measured throughput is only 147 GB/s (pass 1) and
      433 GB/s (pass 2) of a ~1.79 TB/s peak, i.e. **not DRAM-bound**.
    - One kv head's whole range is 22.2 MB (pass 1) / 44.4 MB (pass 2) and fits the 126 MB
      L2, so the duplicated bytes are mostly L2 hits and the DRAM floor is ~0.3 ms.
    - BK 16/32 remove no bytes at all (1.000x) and BK 32 drops occupancy to 1 block/SM;
      (BQ 128, BK 8) halves the bytes (0.522x) but halves the warps hiding the latency
      (149.6 KiB, 1 block/SM) for tens of ms of gain; (BQ 128, BK 16) is worse than that;
      BQ 256 (283.1 KiB) and the kv-head loop (282-538 KiB, and 256 floats/thread spills
      past the 255-register limit) exceed the 227 KiB smem cap or spill.
  - Candidate 3 (open, s117 as design-only): the only direction the model leaves is a
    **kernel rewrite** — K/V shared per kv head via a transposed tile, or online softmax
    moved onto an MMA/flash-style tile. Both need `.cu` work, a PTX regen and a real cold
    A/B, so the dev must first prove the resource budget (smem <= 227 KiB, registers <= 255,
    occupancy >= 2 blocks/SM) and may answer "not feasible" with numbers and no commit.
    - Result (s117): both variants answer **not feasible for the 0.3 s acceptance**. (a) needs
      283.1 KiB smem and 256 floats/thread (caps: 227 KiB, 255 regs). (b) fits the budget
      (124.5 KiB, ~64 regs) but is bounded at 14.3-45.3 ms = 1.5-4.8 % of the 942 ms bucket,
      because the whole kernel issues only 606.6 M lane-ops (0.50 ms at 1 instr/scheduler/cycle)
      while it measures 942 ms = 1,900x that ceiling; the cost is L2 latency with too few warps,
      which mma does not fix. The committed PTX also targets `sm_75`, where mma is unavailable.
  - Candidate 4 (open, s118): **causal key-range truncation** in the existing kernel. Row r of
    block b only needs keys `j <= pos0 + i0 + r`, so every tile whose upper bound is below
    `pos0 + i0` is fully masked for the whole block (s117: 52.8 % of all sm-tiles, the rest
    half-masked on average, ~75 % of pass-1 key work). Starting the tile loop plus the register
    prologue at `jstart = (pos0 + i0) / PF_BK * PF_BK` (aligned down) removes that work and is
    numerically identical: a fully masked tile adds exactly 0 to `acc`, and leaves `m_run` /
    `l_run` unchanged. One `.cu` edit, PTX regen + cold A/B by the supervisor.
    - **Result (s122): rejected - the premise above is false, so the change is not numerically
      identical.** Row r of block b reads keys `j <= pos0 + i0 + r` *because of the mask*; that
      is its upper bound, not a lower bound. Keys below `pos0 + i0` are exactly the ones the
      early rows of the block must attend to, so the proposed `jstart` deletes live work.
      - `jend = min(t_total, pos0 + i0 + PF_BQ)` (`kernels_elem.cu:342`) spans the **worst-case**
        row of the block; the mask `ja <= pos0 + ia` / `jb <= pos0 + ia` (`:449-452`) then lets
        row `i0+r` read up to `pos0 + i0 + r`. There is no "tile below `pos0 + i0` is fully
        masked" set - for a causal mask the fully-masked tiles are always a **suffix**, so a
        loop-start change cannot express them at all.
      - Measured with the CPU model in `crates/denno-engine/tests/attn_prefill_jstart.rs`
        (cold `n = 5,686`, `pos0 = 0`, block `i0 = 64`, `jstart = 64`): **4,032** (row,key) pairs
        inside the skipped range `[0,64)` are *not* masked. Across all 89 blocks 15,983,616
        live pairs would be dropped. Across 291 blocks scanned, **0** tiles were fully masked,
        hence 0 form a maskable prefix.
      - The kernel's mask was first confirmed to equal the CPU reference
        (`full_attn_attend_head` at `t_total = pos0+i+1`) on 499,095 (row,key) pairs, so this is
        not an artefact of the model.
      - Magnitude: a `dh = 8` fp32 toy gives max |att_full - att_truncated| = **0.466**, i.e.
        466x outside the R3 relaxation bound (1e-3), let alone the bit-identical contract.
      - This matches s118 (`docs/HISTORY.*.md`), which had already reached the same conclusion
        from a static reading; s122 adds the executed guard so the false premise cannot be
        re-derived from this file again. No `.cu` change, no PTX regen, no cold A/B needed.
    BK 32 stays rejected: `(64 + 2*32)*257 + 64*32 + 128 = 35,072 floats = 140,288 B` per block,
    so two blocks/SM would need 274 KiB against the 227 KiB cap.

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
- Status: **closed (2026-10-20) - the upper bound is 90 ms (the 17.887 GB of bytes any fusion
  can still remove, at ~200 GB/s; bandwidth bracket 10-179 ms), i.e. below the 0.3 s
  acceptance, so the fusion is not implemented** (s121, docs only, no code, no `.cu`).
  Reason: the pass is dominated by bytes a fusion cannot delete (the activation read by the
  down GEMM; absorbing that write needs the down GEMM in the same launch and a 48 KB f32 row
  does not fit beside the 128x256 tile), and s70's `chain_mlp_prefill` already removed the
  host round trips. The `rmsnorm` 413 ms ceiling was a pre-s85 `Ambient` artifact.
  file:line map, traffic table and arithmetic: `docs/HISTORY.ja.md` / `docs/HISTORY.en.md`,
  s121 section.

## P2 — decode (separate axis)

### P2-1: decode 16.4 ms/token -> 12 ms
- Where: C4 device-resident chain, per-layer syncs, GDN state mirror.
- Why: decode is now the user-visible latency for chat; 5,686-token prefill is a one-off.
- Status: **closed (2026-10-20) — the only host sync left in decode is the logits D2H, and it
  cannot be removed without changing the sampled token sequence** (s115 inventory, no commit).
- What the inventory found (s115, file:line in `crates/denno-engine/src/model/qwen35.rs`):
  decode 1 token = `forward_decode_chain` (qwen35.rs:1805-2036) and it carries **exactly one**
  host sync per step:
  - `chain_download(CH_LOGITS, vocab)` (qwen35.rs:2024-2026) -> `ctx.sync()` at cuda.rs:2646.
    This is the only sync. It waits for every kernel and every staged D2H issued earlier in the
    step.
  - `chain_sync_pending()` (qwen35.rs:2020-2023) adds one more sync **only when
    `prof::enabled()`** (an observation point, not a numeric path).
  - `chain_q35_kv_stage_flush` (cuda.rs:5459) and `cache.advance(1)` are pure host work (no
    sync). The per-layer K/V rows are staged async D2H into a pinned buffer (cuda.rs:5070-5102)
    and complete under the logits sync.
  - GDN state already uses the s50 deferred-flush mirror (cuda.rs:5163-5430); the per-step
    ~75 MB D2H is gone and `chain_q35_gdn_flush_pending` runs only at prefill start or on the
    CPU fallback. No residency lever is left there.
  - The per-op syncs (cuda.rs:1524 / 1988 / 2104 / 2289 / 3029 / 3998 ff.) are on paths the
    decode chain does not take unless it falls back.
- Why the remaining cut is not implementable: the engine samples on the host after every
  forward (engine.rs:871 and the unconditional `sess.logits = logits` at engine.rs:904), so the
  logits must be on the host **before** the next token is appended to the sequence. Keeping
  logits device-resident and deferring the D2H to the next step does not overlap anything (the
  next step cannot start without the token), and removing the D2H for non-final tokens requires
  speculative multi-token forward plus device-side per-position logits plus mid-chain KV/GDN
  mirror bookkeeping — out of scope for a bit-exact change and it needs `.cu` work.
- Attempted and rejected in s115: a deferred-logits scheme (`chain_download_deferred`,
  `chain_wait_deferred_logits`, `pin_logits`, thread-local `decode_tail_last_hint`, returning
  `Vec::new()` from intermediate `forward_decode_chain` calls). It does not build as a design
  (an empty return loses `sess.logits` at engine.rs:904) and cannot pay off while sampling
  stays strictly serial. Work in progress is parked as a patch outside the repo
  (`~/ai_work/s115_p2-1_wip.patch`); the tree was restored to the pre-s115 state.
- Consequence: decode stays at ~16.4 ms/token (GPU-compute bound). The decode axis is closed
  unless a multi-token speculative decoder is designed as its own task (would need `.cu` +
  anchor re-baseline, i.e. an operator decision).
- Speculation follow-up (s119 design + s120 measurement): `docs/SPEC_DECODE_DESIGN.md` compares
  three plans. Plan A (in-place verification) cannot be bit-exact; A-lite (drive the existing
  `n >= 2` path) breaks even only at p* = 0.885 and is rejected for speed. Plan B (n-gram /
  in-prompt lookup, host-side draft, one ordinary step per draft/verify/reject) keeps the
  existing bit-exact and deterministic contracts with **p\* = 0.0006**, so its verdict rests
  only on p = "the 12-gram context recurs and continues the same way".
- **Plan B measured (s120, `scripts/ngram_oracle.py`, CPU only, no model weights):**
  p_1 = 0.2984 (110,072 / 368,831) on the **proxy** corpus (`docs/*.ja.md` + `docs/*.en.md` +
  `crates/**/*.rs`, 58 files, 1,235,606 tokens, approximate word split) and
  p_1 = 0.1265 (120,950 / 955,830) with the raw-UTF-8-byte tokenizer (2,667,516 tokens); the
  **synthetic** tool-call corpus gives p_1 = 0.1156 (23,776 / 205,686). k = 2..4 fall off fast
  (proxy words 0.1521 / 0.1216 / 0.0970; synthetic 0.0286 / 0.0171 / 0.0064), and shrinking the
  context does not help (proxy words n = 2: p_1 = 0.3316, p_2 = 0.2134). Both tokenizers are
  approximations, not the model tokenizer, and **the corpus is a proxy, not model output**;
  the real-workload p is **not measured**.
- **Decision (s120): plan B is closed on the measured p.** p* = 0.0006 is not close to being
  the binding constraint here (every measured p_1 is 190-2,600x above it: 0.1156/0.0006 = 193
  to 0.3316/0.0006 = 553), but the *useful* break-even is, and the arithmetic from the s119
  formulas says the measured range cannot reach it:
  - plan B costs a **draft step + a verify step** per accepted token, so with `S` = 16.4 ms
    (measured) the effective time is `(2S + C_d) / (1 + p_1)`. At p_1 = 0.2984 that is
    32.80 / 1.2984 = **25.3 ms/token**, i.e. **+54 % vs plain decode**, not a gain.
  - it only wins when `p_1 > 1 + 2 C_d / S` = **1.00001**, i.e. **never** for any C_d >= 0.
    The p at which the 12 ms/token target is met is `(1 + p_1)/32.8 >= 1/12` -> p_1 >= 1.73,
    which p cannot reach (p <= 1).
  - the invariant behind both numbers: a draft step that can be rejected buys nothing unless
    it costs **less than a normal step**. Measuring p cannot change that; only making the draft
    cheaper than `S` can.
- Therefore the honest statement is: **n-gram speculation (plan B) is closed (measured p).**
  The measurement was still worth running - it rules out the one plan that preserved the
  bit-exact contract, and it says the decode axis has no host-side lever left. Reopening it
  would need a draft cheaper than `S` (an approximate draft model, or folding draft+verify into
  one forward), which is plan A's territory and carries the numeric-class cost s119 rejected.
- Next measurement, only if that changes: a **real response token log** (token ids from the
  server) at n = 2 and n = 12 to confirm the proxy numbers on model output. Providing that log
  is an operator task (privacy/network policy forbids fetching one here); the real-workload p
  is **not measured**.

## P3 — correctness and debt

### P3-1: remove or fix the rejected GPU gdn_proj kernel — CLOSED (s96)
- Status: **closed** for the Rust side. The opt-in (`DENNO_GPU_GDN_PROJ`) produced wrong
  values (max abs diff ~1e2; m=33 gives -17.205 vs -8.410; k=1 latches `LMV32_BROKEN`),
  so the whole path was deleted in s96: `cuda::gdn_proj_f32_into` and its helpers, the
  `GdnProjMiss`/`DENNO`-style plumbing, `model/qwen35.rs::gdn_proj_into_gpu` and its
  prefill branch, and the three tests that covered them.
- Default path: unchanged (beta/alpha always went through `apply_bt_rows_into` with the
  gate off; it is now called unconditionally — bit-identical).
- `.cu`/PTX half: **done (2026-10-20, supervisor)**. `dno_matvec_rows_f32` and its two
  helpers (`DNO_MVROWS_TILE`, `dno_mvrows_acc_rev`) were deleted from
  `kernels/kernels_elem.cu` (73 lines, nothing else referenced them) and
  `kernels_elem.ptx` was regenerated on the GPU host with
  `nvcc -ptx -arch=compute_75 -fmad=false -O3` (sha256 `cbe5533e` -> `7abee571`; the
  pre-edit source recompiled to `cbe5533e`, i.e. the committed PTX reproduced exactly
  before the deletion). `GPU_VERIFY.sh` = RESULT: PASS, anchors c61ad85b / 9e17d563 /
  af6eb6ab all MATCH, decode gen2 16.4 ms/token unchanged, so the dead kernel was
  genuinely dead.
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
- Result (main session, 2026-10-20, GPU host): **fp32 at 9B is not viable on this host.**
  `serve --quant fp32` (the default mode) keeps full f32 weights: 48 GB RSS for the 9B
  model. The card has 32,607 MiB, so fp32 cannot be GPU-resident here, and the CPU load
  took 25 s. The 5,686-token prefill did not finish: the client hit its 300 s timeout
  (19.0 tok/s measured mid-flight) and the request was still running at ~15 min when it
  was killed, so the decode number behind it never started (one request slot, no
  preemption on a cold request). Practical conclusion: fp32 prefill at 9B on this CPU is
  more than 6x slower than int8+GPU before it even completes, so there is no useful
  wall-clock fp32 number to publish from this host. Measuring it needs either a 48 GB
  card or a smaller model.

### P4-3: 27B Q4_0 quality
- Why: 27B loads and runs deterministically but output quality (greedy collapse) is
  unexplained. UD-IQ2_XXS remains impossible (IQ1_M unsupported).
- Action: compare against llama.cpp on the same GGUF at the same seed, per-layer, to find
  where the collapse starts.
- Acceptance: either a fix or a documented root cause.
- Result (main session, 2026-10-20, GPU): **GPU E2E re-verified — the CPU fix holds on the
  GPU path too.** No download was needed: the uniform Q4_0 GGUF was already inside the
  container at `data/.cache/denno/hub/models--local--qwen35-27b/Qwen3.8-27B-Q4_0.gguf`
  (15 GB, plus tokenizer and config), so `serve local/qwen35-27b --quant int8` resolves
  directly. Load took 90 s, VRAM 31,977 / 32,607 MiB (co-resident nothing else).
  Answers: "The capital of France is **Paris**." twice (the two replies are text-identical;
  the JSON files differ only because they carry per-request timings), and
  "日本の首都は**東京**です。" followed by a correct explanation that Japan has no explicit
  statutory capital. That reply stopped at `max_tokens=128` after 85.6 s = 669 ms/token, so
  27B decode is ~40x the 9B int8 rate and should not be used for chat without a budget.
- Anchors: **27B stays verified by answer only** (decided 2026-10-20). The two historical 27B
  md5s (`1a4ea71a…` / `b4e58cb3…`) predate the GQA fix and are dropped rather than rebased:
  the standing 3-anchor set is the 9B production model, GPU_VERIFY switches to it on a bare
  launch, and adding a 27B arm would mean a second anchor protocol for a model that is not
  the production target. Reproduce the 27B check with `p43_27b.sh` in the operator tools
  directory if the GPU path needs re-confirming.

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
- Result (main session, 2026-10-20): run against the canonical build on 8095
  (`python3 scripts/needle_check.py --base-url http://127.0.0.1:8095`, inside the
  container): payload 24,113 chars, `usage.input_tokens` 4,640, request 11.57 s,
  answer `QX-4471`, verdict MATCH, **exit 0**. So the checker works end to end on a real
  build. `GPU_VERIFY` now also passes with 3/3 anchors on the same build. Note: the
  `denno-gpu` container had no python3, so the main session installed `python3-minimal`
  (3.10.12) — the line above now runs as written; the remaining step is only adding it to
  the gitignored `GPU_VERIFY` script.

## P5 — tooling and ops

### P5-1: dashboard shows cold A/B and decode — DASHBOARD DONE (s109), era wired in (s110), GPU run is a human step
- Why: `pf_best.txt` now holds a cold number but the graph still mixes eras.
- Action: parse the A/B lines from the progress log, plot prefill and decode on separate
  axes, mark the protocol change with a vertical line.
- Done in s109: the progress log is free-form prose, so it is NOT parsed. The
  machine-readable source is `scripts/cold_ab.py --json`, and `scripts/dashboard.py`
  builds the dashboard from a directory of those JSON files plus an optional `PF_BENCH`
  log. One self-contained HTML file, inline SVG only (no script, CDN, font or image
  reference), so it opens offline. Two panels with separate y axes — **prefill (s)** =
  cold A/B `wall` seconds of the successful measurements plus the `PF wall=<sec>` lines,
  **decode (ms/token)** = `wall * 1000 / tokens` from the PF lines — and x = measurement
  order (index). `--protocol-change LABEL` draws a red dashed vertical line where that
  era/label starts (repeatable), which is what splits the eras. A panel with fewer than
  2 points prints "not enough points" instead of a line. Output is byte-identical for the
  same inputs; exit 0 ok / 1 unusable input (bad JSON, missing dir, no points — the file
  name is always in the message) / 2 usage error, and no partial `--out` is left behind.
- Shape note (nothing is guessed): `cold_ab.py --json` writes `stats.old.median`,
  `stats.new.median`, `stats.delta_pct` and the per-round `measurements[*].wall`, and it
  has **no decode number**. Decode therefore comes only from the PF lines.
- Era gap CLOSED (s110): `cold_ab.py --era LABEL` now writes a top-level `"era"` key into
  the `--json` payload, so the dashboard reads a real era instead of guessing from the
  file name. It is metadata only — rounds, ports, payload sha256 and the printed report
  are untouched, and `--json` without `--era` writes byte-identical output to HEAD
  (verified by running both). The label is 1–64 characters with no control characters
  (edges trimmed); a bad value is an argparse usage error (exit 2, nothing measured).
  `--era` without `--json` is a no-op, not an error (same as the printing-only flags).
  The dashboard precedence is unchanged: `era` -> `label` -> file name, and
  `parse_cold_ab_payload()` still reports which one it used in `era_source`
  (`era` / `label` / `filename`), so pre-s110 JSONs keep working.
- CPU half is covered: `python3 scripts/dashboard.py --self-test` (21 checks) and
  `python3 scripts/test_dashboard.py -v` (21 tests, including
  `TestEraRoundTrip`: a real `cold_ab.py --json --era` run against a fake server, read
  back with `era_source == "era"`); `python3 scripts/test_cold_ab.py -v` (44 tests).
  Both suites run inside `scripts/run_cpu_checks.sh` as stages
  `dashboard_selftest` / `dashboard_tests` / `cold_ab_tests` (still 11 stages).
- Still open, and it is a HUMAN STEP: there is no real A/B JSON in the repo and the
  agent container has no GPU, so no real chart exists yet. On a GPU host, after each
  cold A/B run, collect the JSON and build the page:

      # one JSON per measurement, e.g. /tmp/cold_ab_runs/s110_<build>.json
      python3 scripts/cold_ab.py --old <old-bin> --new <new-bin> \
          --json /tmp/cold_ab_runs/s110_<build>.json --era s110
      PF_BENCH.sh > /tmp/pf_bench_<build>.log
      python3 scripts/dashboard.py --cold-ab-dir /tmp/cold_ab_runs \
          --pf-bench-log /tmp/pf_bench_<build>.log --max-tokens 32 \
          --protocol-change <era-where-the-protocol-changed> \
          --out /tmp/dashboard.html

  Pass `--era <label>` on every cold A/B run: `--protocol-change <label>` only draws a
  line where a run actually carries that era, so a JSON without `--era` puts the label
  in the file name instead. `--max-tokens 32` is the PF_BENCH.sh payload size
  (`max_tokens=32`). The PF line has no per-line `output_tokens`, so decode is derived
  from that fixed count; a log line that carries `output_tokens=<n>` is used as-is.
- Acceptance: one HTML page where prefill and decode are on separate axes and the
  protocol change is a visible vertical line.
- Result (main session, 2026-10-20): **first real chart.** A 3-round cold A/B
  (s81 `b502e34` vs HEAD `506b515`, payload sha256 `dead4c91...`, 5,686 input tokens)
  gave old median 13.633 s vs new 13.610 s = **-0.17%**, i.e. parity since s81 on a
  machine that measures ~3% slower than yesterday. The JSON was rendered with
  `python3 scripts/dashboard.py --cold-ab-dir <dir> --out p51_real.html`
  (4,729 bytes, inline SVG only, real values 13.60-13.63 s). The JSON and the page are
  kept outside the repo (an operator-side tools directory), because the `--json` payload
  records host-specific binary paths. Still open: no decode point on the chart, since a
  cold A/B carries no decode number and the only decode source is a `PF_BENCH` line.

### P5-2: docs sync — DONE (2212c4a, s101)
- `README.md` and `docs/api.md` still describe pre-s68 behaviour in places
  (round-trip counts, the old "PF wall" basis).
- Done in 2212c4a (s101): `README.md` + `docs/api.md` synced with the current implementation.

## Suggested order

1. ~~P1-2 attention (942 ms bucket), candidate 4 = causal key-range truncation~~ → **rejected in
   s122** (and already in s118): the kernel's causal bound is per row, so the "fully masked
   tiles below `pos0 + i0`" set does not exist. **P1-2 is now fully closed** - candidates 1-4
   are all rejected with numbers, and P1-1, P1-3, P1-4 were closed in s91/s92/s121. There is no
   open prefill lever left; the remaining 942 ms would need a per-row key range (the "kernel
   rewrite" s117 bounded at 1.5-4.8 % of the bucket), so prefill work should stop here.
2. In parallel on the CPU-only agent: P3-1 (delete the dead kernel), P3-3, P4-1, P4-4.
3. P2-1 decode once P1 has plateaued; P3-4 the moment a second A/B is needed by hand.