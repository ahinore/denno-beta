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
  file:line map, traffic table and arithmetic: `docs/history/30_decode_perf.md`, s121 section.

### P1-5: the phase-lever queue P24-P28 (2026-10-22/23) - four measurement closures, one implementation

This queue asked, phase by phase, whether the host side of a `DENNO_PF_DBG` interval could be
removed (P24 asked the same question of the decode launch inventory). P24-P27 were all closed
with **zero code changes** (measurement only) and P28 is the only commit. Phase names and ms
are the s170 table (n=14,043, 64 layers, wall ~62.9 s) as quoted in REPORT_P26.md /
REPORT_P28.md; `[M]` = measured, `[R]` = arithmetic over measured shapes, `[A]` = assumption,
the same labels REPORT_P27.md uses. **A reader must not re-open any of P24-P27 as a host-side
lever: the measurement says the lever is not there.**

#### P24 - decode launch inventory (s177, REPORT_P24.md)
- Status: **CLOSED as measurement-only - no code change, no commit.**
- Measured: **610 kernel launches/token** = 48 GDN layers x 9 + 16 full layers x 11 + `lm_head`
  + final rmsnorm; per token 2 H2D (20,736 B), 33 D2H (1,124,352 B), 16 D2D, 0 memset,
  1 host memcpy. Host fixed cost on the critical path **0.28 ms/token = 1.6 %** of 17.3 ms
  (greedy argmax 0.149 ms, measured on this container's CPU).
- Ruled out (**host-side lever absent**): the kernel times sum to ~17.9 ms/token against a
  **17.28-17.32 ms/token** wall, so the GPU is busy for the whole step and there is no gap to
  fill. No adjacent same-shape GEMV pair is left (qkv|z, q|k|v, gate_up and beta|alpha are
  already 1 launch each), the only sync is the logits D2H (removing it changes the sampled
  sequence), and both H2D are position-dependent. The remaining candidates (rmsnorm parallel
  reduction, qknorm q/k fusion, pinned logits) need `.cu` or a new numeric contract.

#### P25 - `sl_lay_ffn_down` (dev2 worktree, REPORT_P25.md)
- Status: **CLOSED as measurement-only - no code change, no commit.**
- Measured: the phase contains exactly one thing, the down `dno_gemm_q4_0` launch - 14
  launches/layer (13 x 1,024 rows + 1 x 731, tile 1,024), 896 launches and 160.2 TFLOP per
  request at n=14,043; H2D 0 B, D2H 0 B, `ctx.sync()` 0, alloc 0, quantize 0. The bucket is
  **100 % GEMM**.
- Ruled out (**device-only region**): the "host or copy >= 20 % of the phase" condition is
  structurally 0, not small - there is no host work inside this phase to remove, and the
  prologue (`weights_for_q4`, `chain_ensure_buf`) sits before the phase boundary (see P27).
  What is left is device-side: down runs at **14.4 TFLOP/s** where gate|up is 31.9 and o_proj
  30.8, so **5.9-6.1 s** of the 11,100 ms bucket is unexplained inside the kernel; wave
  quantisation (640 blocks/launch) bounds that at 1.5x = ~3.5 s.
- Consequence: the 11,100 ms is the s170 **host-bound** build, so the current value is a
  measurement question (best estimate 4.6-6.7 s [R], see P27), not an implementation one.

#### P26 - `sl_lay_full_qk` and `sl_lay_gdn_oproj` (s178, REPORT_P26.md)
- Status: **CLOSED as measurement-only - no code change, no commit.**
- Measured: `gdn_oproj` = **1** `dno_gemm_q4_0` launch/layer (m=14,043, k=6,144, n=5,120,
  grid (220,40,1) = 8,800 blocks), H2D 0, D2H 0, sync 0, host copy 0 - the phase is the o_proj
  GEMM itself. `full_qk` = 3 launches/layer + 3 **pageable** D2H (805.3 MB/layer, 12.88 GB per
  request at 14,043) + 1 mandatory `ctx.sync()`; host fixed cost 3-6 us/layer = 0.05-0.1 ms per
  request.
- Ruled out (**device-only region** for `gdn_oproj`, **host-side lever absent** for `full_qk`):
  gdn_oproj is device-complete with nothing to delete; full_qk's only remaining lever is the
  transfer shape, and folding the 3 D2H into 1 is exactly P17b - measured **9.78 -> 22.14 s**
  (+772 ms/layer) and reverted - so the 3 pageable copies stay.
- Caveat recorded, not resolved: s170's phase % and P13's 11 TFLOP/s cannot both be true
  (`gdn_oproj` at 2.2 % would require 45 TFLOP/s). It changes the priority order, not the
  verdict.

#### P27 - phase-accounting audit (dev2 worktree, REPORT_P27.md)
- Status: **CLOSED as measurement-only - no code change, no commit.**
- Measured: `pf_phase()` only records events on the default stream, so each bucket is
  "kernels issued in the interval + GPU idle + host lag", **not kernel time**; the 11 buckets
  do split the wall completely (s170: 9 printed rows sum to 61,999.6 ms = 98.33 %, tail
  567.4 ms = 0.9 %) but that is not an attribution of the time.
- Ruled out (**attribution shifted one phase** / **cold-run inflation**): the phase boundary
  sits at the head of an issue group, so `weights_for_q4`, `chain_ensure_buf` and uploads fall
  into the *previous* phase (FFN prologue -> `norm_post`, full qkv prologue -> `norm_in`,
  o_proj prologue and att upload -> `full_core`). Warm runs make that us-level (< 10 ms
  total); the cold run pays the 13.2 GB first-time weight H2D and inflates `sl_lay_norm_post`
  by ~0.5-0.6 s [R], which is why `scripts/prefill_bench.py` discards the first request. No
  accounting defect was found in the host code, hence no commit.
- Consequence: the s170 table is not a table of the current build. At 3,161 tokens the known
  floors leave `down <= 1.0-2.6 s`; at 12,850 only the joint bound `down + full_core =
  14.2-17.0 s` survives. One command settles it: `DENNO_PF_DBG=1 python3
  scripts/prefill_bench.py 65000 3`.

#### P28 - `sl_lay_full_core`: unused `buf_q` / `buf_att` allocation moved to the use sites
- Status: **IMPLEMENTED - commit `3c383a7`** (`crates/denno-engine/src/model/qwen35.rs`,
  +24/-3). Only the *position* of two allocations changed.
- What changed: `full_attn_host_side` called `fit_prefill_buf(buf_att)` and
  `fit_prefill_buf(buf_q)` at the top of every request (2 x 345 MB at n=14,043), but the live
  path - GPU attend = `AttnPlace::GpuDev` - never reads either buffer. The two fits moved to
  the three places that do read them: before prepare writes `buf_q` (`!q_in_pin`), before the
  zero fill inside `if !att_done`, and before `copy_from_slice` in `full_attn_attend_gpu`.
  `fit_prefill_buf` itself, the zero-fill rule, the arithmetic, the transfer bytes, the issue
  order, the kernel and grid/block/smem are unchanged; every reader writes all elements
  immediately before use, so the output is bit-identical.
- Measured (this container, CPU): one `fit_prefill_buf` of 345 MB costs 118.95 / 118.28 ms
  (n=14,043), 126.19 / 121.21 ms (12,850), 30.45 / 30.46 ms (3,161) = **237.23 / 247.40 /
  60.91 ms per request** for the pair. The host side of `full_core` itself is prepare
  44.91 ms/layer + sigmoid gate 45.40 ms/layer = **90.32 ms/layer = 1.445 s/request** at
  14,043, which is 13.5 % of the **10,695.3 ms** phase; the phase is device bound (attention
  kernel ~60 %, transfers ~14 %).
- **PREDICTED / NOT MEASURED**: -0.24 s/request at 14,043, -0.25 s at 12,850, -0.06 s at 3,161
  (2-3 % of the `full_core` phase, ~0.6 % of the 38.79-40.35 s wall at 12,850), plus ~690 MB
  less RSS at 14,043. **GPU-only caveat**: the live path is `GpuDev`, which this container
  cannot execute (`chain_ctx()` returns None), so the cold A/B and the RSS check are still a
  human step. If the cold A/B does not move, reverting is harmless because the values are
  identical.
- The only >= 0.3 s candidate left in `full_core` is an n-token sigmoid gate kernel (`.cu`);
  the existing `dno_attn_gate_mul` is single-token (14,043 launches). The 45.40 ms/layer host
  gate and the 690 MB/layer att round trip would go together. That is main-session `.cu` work.

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
  -> corrected (2026-10-23): "IQ1_M unsupported" is stale. IQ1_M (type 29) is decoded —
  enum `kquant.rs:40`, `from_id` `kquant.rs:72`, decode arm `kquant.rs:624` — implemented
  in 2344f67 (s157, Task K1 commit 2), with golden-value tests in
  `tests/dequant_ref.rs` (`iq1_m_matches_hand_computed_block`, `tests/dequant_ref.rs:303`).
  So IQ1_M is no longer what blocks UD-IQ2_XXS. Whether that 6.8 GB file loads end to end
  has not been re-verified here (this container holds only the 27B Q4_0 GGUF).
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

### P4-5: Q4_0 native weight path — DONE (s123)
- Why: int8 mode expands every GGUF to fp32 and re-quantises to one byte per element, so a
  4.5-bit source still costs 8 bits in VRAM (27B: 25,709 MB, only 5.3 GB left on a 32,607 MiB
  card). UD-IQ2_XXS cannot even load (TQ1_0 / IQ1_M are not decoded in `kquant.rs`), and it
  would not have helped in int8 mode anyway. The lever was to keep the GGUF 4-bit blocks as
  they are.
  -> corrected (2026-10-23): the parenthetical is stale. TQ1_0 (type 34) and IQ1_M (type 29)
  are both decoded in `kquant.rs` — enums `kquant.rs:42` / `kquant.rs:40`, `from_id` arms
  `kquant.rs:74` / `kquant.rs:72`, decode arms `kquant.rs:591` / `kquant.rs:624` — implemented
  in 2344f67 (s157), so the loader's type gate accepts both ids (`gguf.rs:178-179`) and the
  GGUF path dequantises through `kquant::dequant_row` (`gguf.rs:238`). "Cannot even load" was
  true when this section was written (s123); it is no longer. Loading UD-IQ2_XXS itself is
  still unverified in this container (no such GGUF here), so read this as "the decoder gap is
  gone", not "the file is measured".
- What was implemented (opt-in, `--quant q4` / `--quant q4_0`; 9B int8 and the anchors are
  untouched):
  - `Linear::Q4(Q4Linear)` holds the Q4_0 blocks verbatim: 32 elements = f16 scale (2B) +
    16B nibbles = 18B, dequantised as `y = d * (q - 8)` with `d` allowed to be negative.
    `Q4Linear::quantize` follows ggml's `quantize_row_q4_0_ref` (including its saturation at
    15 when the block extreme is negative, which caps the error at one block scale there).
  - CPU path `tensor::matvec_rows_bt_q4_into` (rayon over rows, in-order f32 accumulation).
  - Two kernels in `kernels_int8.cu`: `dno_gemv_q4_0` (1 warp = 1 output row, 1 block = 8 rows)
    and `dno_gemm_q4_0` (1 block = 16 rows x 256 cols, 34 KB smem). Both read an SOA layout
    built at upload time: `nib` = one uint4 per block (16B aligned), `sc` = one f32 per block.
  - `cuda::maybe_gpu_linear_q4{,_into}` mirrors the int8 seam (weight registry keyed by
    `Q4Linear.uid`, permanent latch on error). `m > 16` goes to the tiled GEMM; the int8
    `gemm_min_m` (256) is deliberately not applied because the Q4 CPU fallback is a plain
    loop, so GPU GEMM wins even at m=17. Prefill at m=17..255 was measured at seconds/token
    before that change and milliseconds after.
  - `concat_rows` gained a Q4 arm so the gate+up fusion (s80) still applies.
- Numbers (27B, uniform Q4_0 source, GPU, 2026-10-20):
  - VRAM **18,369 MiB** (int8: 31,977) — about 16.7 GB of weights, so ~13 GB stays free.
  - decode **178-186 ms/token** (80 tokens, prefix cached, three runs 16.30 / 14.89 / 14.23 s);
    int8 was 669 ms/token, so 3.6x. Three runs produced identical text (deterministic).
  - prefill **34 tok/s** (1,240 tokens in 36.38 s).
  - short prompts: "The capital of France is" -> `Paris` (4 tokens, 1.14 s once warm),
    "Name three colours." -> 14 tokens 4.45 s, Japanese capital -> `日本の首都は東京です。` 9
    tokens 3.73 s. First request after launch pays the 16.7 GB upload (10.9 s) on top.
  - correctness contract: CPU vs GPU maxabs 9.5e-6 (GEMV) and **0 (GEMM, bit-exact)** at
    k=256 n=64 for m=1/7/16/300; Q4 vs fp32-on-dequantised-weights <= 1e-5; 2nd run bit-identical.
- Tests: 246 CPU lib tests pass (5 new Q4 unit tests + 1 concat test), plus the ignored GPU test
  `cuda::tests::cuda_q4_linear_matches_cpu`. GPU_VERIFY on the same tree: 3/3 anchors MATCH,
  RESULT: PASS, 9B decode unchanged at 16.4 ms/token.
- Reproduce: `p44_q4.sh` (launch + VRAM), `p44b_q4_ask.sh` (answers), `p44c_q4_dec.sh`
  (decode timing + determinism), `p44d_q4_prefill.sh` (prefill) in the operator tools directory.
- Still open:
  - `kquant.rs` has no TQ1_0 (type 29) or IQ1_M (type 24) decoder, so IQ2-class files still
    cannot be loaded. With the native path in place that is now the only thing between the
    6.8 GB UD-IQ2_XXS and a 24 GB headroom, and it is a separate piece of work.
    -> corrected (2026-10-23): wrong on both counts. (1) The type IDs are TQ1_0 = 34 and
    IQ1_M = 29; type 24 is I8 (`kquant.rs:69-74`, and `tests/dequant_ref.rs:358` asserts
    `from_id(24) == None`). (2) Both decoders exist as of 2344f67 (s157): enums
    `kquant.rs:42` / `kquant.rs:40`, block size 256 at `kquant.rs:89-90`, type sizes 54 B / 56 B
    at `kquant.rs:119` / `kquant.rs:117`, decode arms `kquant.rs:591` / `kquant.rs:624`. So
    "IQ2-class files still cannot be loaded" is no longer true for this reason, and this bullet
    is closed — the P4-5 residue that remains is the pass-through item below (blocks are
    re-quantised from fp32 rather than taken verbatim from the file).
  - Blocks are re-quantised from fp32 today (GGUF -> fp32 -> Q4_0), which is within one block
    scale of the file's own blocks. Passing the file's blocks straight through needs the tensor
    type plumbed into `Linear`, and would also skip the 100 GB fp32 expansion at load.
  - The CPU Q4 path is a correctness fallback, not a fast path.

### P4-6: Q4 native path held whole-prompt scratch - DONE (main session, 2026-10-20)
- Symptom: with 27B on `--quant q4`, a long prompt pushed VRAM from 18,369 MiB to
  32,073 MiB; it never came back down, and the next request died with
  `CUDA cuMemAlloc が失敗: error 2` and fell back to CPU (seconds/token).
  10,821 tokens cost 299 s; a 2,000-token prompt cost 42-106 s.
- Cause: `q4_dispatch` H2D'd the whole activation tile and ran one GEMV/GEMM with
  `m = prompt length`. Its buffers live in `SCRATCH`, a thread_local, grow-only set
  sized `(m*k + m*n)*4 B`. With k=n=17,408 and m=10,821 that is ~1.5 GB per thread
  set, kept forever. This is a memory shape bug, not a weight leak: the Q4 upload
  lines are 401 in total and do not repeat per request (counted).
- Fix: `q4_dispatch` walks `m` in fixed tiles of `Q4_M_TILE = 256` rows and calls a
  new `q4_dispatch_tile`, so the scratch is bounded by the tile instead of the
  prompt. Rows are independent and a row's accumulation order is unchanged, so the
  result is bit-identical to the untiled path; the GPU test
  `cuda_q4_linear_matches_cpu` covers m = 1/7/16/300 (300 crosses the boundary).
- Measured (27B q4, 10,820-token prompt, `vram_probe.py`): before 32,073 MiB and
  the next request failed with cuMemAlloc error 2; after 28,389 MiB in 306 s and
  the request completes. A 3,020-token prompt: 21,135 MiB in 93 s.
- Diagnostics kept (env-gated, no effect by default): `DENNO_MEM_DBG=1` logs every
  `DevBuffer` alloc/free with the free/total MiB and the caller location
  (`#[track_caller]`), and an alloc failure now reports the requested size and the
  free MiB. This is how the shape bug was pinned; keep it for the next one.
- Follow-up: the 0.9 MB/token that remained at this point was mostly the KV mirror
  waste, fixed in P4-7 (28,387 -> 21,603 MiB at 10,820 tokens); what is left is
  ~0.29 MB/token of GDN scan and qkv/rope chain scratch.

### P4-7: bound the n-sized prefill scratch - DONE (main session, 2026-10-20)
- Target was: after a 16k-token prompt, VRAM should be within ~1 GB of the 3k case.
- The dominant term was **not** the GDN scan. `with_mirror_ext` allocated the GPU KV
  mirror for **all** `n_layers` whenever any layer grew the cap, so 27B (64 layers =
  48 GDN + 16 full) kept K/V buffers for 48 layers that never write KV: 128 buffers
  x 64 MiB = 8.1 GB at a 10,880-token prompt. The growth also used
  `next_power_of_two`, allocating 16,384 rows for 10,880 tokens (1.51x).
- Fix: the cap growth only resets the per-layer cursors; each layer's K/V buffer is
  now ensured when that layer is actually used (`alloc_kv` calls), and the cap rounds
  up in 1,024-token steps instead of powers of two. A buffer that is re-allocated
  drops its `len` so the next use re-seeds from the host (never a partial seed).
- Measured (27B q4, `vram_probe.py`):
  | prompt | before P4-6 | after P4-6 | after P4-7 |
  |---|---|---|---|
  | 70 | 18,377 MiB | 18,377 MiB | 18,377 MiB |
  | 3,020 | 21,505 MiB | 21,135 MiB | 19,471 MiB |
  | 10,820 | 32,073 MiB (next request OOM) | 28,387 MiB | 21,603 MiB |
- Consequence: `DENNO_MAX_PROMPT_TOKENS` goes from 12,288 to 32,768 (the guard still
  rejects longer prompts with a 400 instead of OOM-ing into the CPU fallback).
- Remaining n-proportional terms are ~0.29 MB/token (GDN scan scratch ~0.11, qkv/rope
  chain ~0.07, KV mirror 16 layers x 2 x 4 KB). A 64k prompt would still need ~40 GB:
  chunking the GDN scan and the chain buffers is the next lever if that is needed.
- Note: this is why 9B long contexts never worked either - the wasted mirror was sized
  by layer count, not by prompt.

### P4-10: Q4 prefill GEMM micro-tile and GDN state reuse - DONE (main session, 2026-10-20)
- Scope: 27B q4 is the production model on 8095. Prefill sat at 34 tok/s and decode at
  178-186 ms/token, which is what made a long Hermes turn feel slow.
- `dno_gemm_q4_0` (`kernels_int8.cu`) had one thread per (row, 16-column group): 32 FMA
  per 20 shared loads, and its `sB[Q4_GN][32]` staging row was 32 floats wide, so every
  lane of a warp touched the same shared bank (32-way conflict).
- Two changes, both order preserving:
  1. `sB[Q4_GN][33]` (+1 pad) with the column assignment `col = tx + 16 * v`, so the
     shared address spreads over banks instead of repeating `c mod 32`.
  2. A 4-row x 8-column micro-tile per thread: 256 threads cover 64 rows x 128 columns
     per block, 12 loads per 32 FMA (0.375 against 0.625 before). Shared memory 25 KB.
- The accumulation order is untouched (ascending `b`, then `c` inside each 32-element
  block), so the GPU GEMM stays bit-identical to the CPU reference: the ignored GPU test
  reports `q4 m=300: maxabs=0e0` while the GEMV sizes stay at 4.8e-6 / 7.6e-6 / 9.5e-6.
- Decode: `gdn_scan_impl` allocated 13 device buffers for every layer of every token.
  It now borrows the thread-local `GDN_SCRATCH` that the chunked path already used, and
  the `conv_out` / `ssm_out` zero fill moved from a host `Vec` upload to
  `cuMemsetD32Async` (`DevBuffer::fill_zero`).
- Measured (27B q4, cold single request, prefix cache off, `vram_probe.py 300`):
  | stage | prefill 3,020 tok | decode, 128 tok warm |
  |---|---|---|
  | baseline (c51d6a3) | 93.3 s (32 tok/s) | 186 ms/token |
  | bank padding + micro-tile | 43.4 s cold / 36.7 s warm (82 tok/s) | - |
  | + GDN buffer reuse | 36.9 s | 169.9 ms/token |
  | + device zero fill | 36.9 s | 152.7 ms/token |
- Next decode levers, from the same profile: `d2h_state` is still 50.8 ms/token because
  the GDN state round-trips host and device for every layer of every token (647 KB each
  way, six synchronous copies per layer), and `gate_up_proj` is 41 ms/token at about
  2.4 GB/s effective, i.e. latency bound rather than bandwidth bound. Both need the
  device-resident state (`GDN_STATE`, the s89 scaffolding, still unused) and a GEMV
  rewrite, not more tuning of the current kernels.

### P4-11: GDN state stays on the device - DONE (main session, 2026-10-20)
- Scope: the biggest decode item left over from P4-10. `gdn_scan_impl` and
  `gdn_scan_chunked_impl` uploaded the conv / ssm state from the host and downloaded
  the result back for every layer of every token: 647 KB each way, six synchronous
  copies per layer, 50.8 ms/token on 27B q4.
- `GdnStateCache` (the s89 scaffolding) is now the authoritative copy:
  - `gdn_state_begin(kv, li, ...)` returns the device pointers for the layer and seeds
    them from `KvCache::gdn` only once, when the layer is not known to be in sync.
  - `gdn_state_commit(li, ...)` copies `conv_out` / `ssm_out` back with
    `cuMemcpyDtoDAsync`; nothing crosses the bus in the steady state.
  - `gdn_state_take(kv, li)` brings one layer back to the host (snapshot, reset,
    CPU fallback) and invalidates it, so the next scan re-seeds from the host.
  - The cache is keyed by the KV-cache id as well as the layer. Slots are reused and
    the first id is 0, so the key is `Option<u64>`, not `u64` with 0 as "unused".
  - Layer vectors resize on first use (`touch_layer`): the layer count is only known
    when the first GDN layer runs.
- Three bugs surfaced while landing it, each caught by the anchor suite:
  1. Chunked pass 3 still read the scratch `conv_in`, which the device-resident path no
     longer fills, so the last chunk's conv history was stale. It now reads the state
     buffer.
  2. The decode chain (`chain_q35_gdn_scan`) seeds its own mirror from the *host* state
     when its position jumps. With the host no longer updated by the scan, that seed
     read a stale value and every anchor text changed (deterministically, so it looked
     like a contract violation rather than a race). It now seeds through
     `gdn_state_take`, falling back to the host only when the device has no valid copy.
  3. `attach_gdn_slot` replaced host state without invalidating the device copy.
- `DENNO_GDN_DEVSTATE=0` restores the old host round-trip for bisecting; the default is
  the device-resident path. With it set, the three anchors reproduce the s130 hashes
  exactly, which is what proved the plumbing itself is numerically neutral.
- Measured (27B q4, `DENNO_PROFILE=1`, prefix cache off, `bin_canonical` = candidate):
  | stage | prefill 3,020 tok | decode, 128 tok |
  |---|---|---|
  | P4-10 (48b3a2b) | 36.9 s (82 tok/s) | 152.7 ms/token |
  | P4-11 device state | 36.3 s (83 tok/s) | 126.7-126.8 ms/token |
  Anchor suite 3/3 MATCH. VRAM after one request 19,669 MiB (19,473 before), i.e. the
  48-layer state costs about 200 MiB including allocation granularity.
- Next decode lever is unchanged: `gate_up_proj` runs at about 2.4 GB/s effective and
  needs a GEMV rewrite; with 126.7 ms/token it is now a larger share of the step.

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

### P5-3: production server on 8095 is 27B (q4) - DONE (main session, 2026-10-20)
- Why: the Q4_0 native weight path (P4-5) put 27B at 18.4 GB and 180 ms/token, so the
  production endpoint no longer has to stay on the 9B int8 build for speed.
- Launch: `bash scripts/prod_serve_27b.sh` (repo-relative, no host paths inside). It stops
  any running `serve local/qwen35*`, then runs
  `target/bin_canonical serve local/qwen35-27b --port 8095 --host 0.0.0.0 --seed 42
  --quant q4 --repetition-penalty 1.05 --temperature-cap 0.7`.
- Tuning: `DENNO_BATCH_SLOTS=4` and `DENNO_PREFIX_CACHE_MAX_TOKENS=32768`. The default 24
  slots x 4,096 tokens cost 12.9 GB of KV, which exceeds the 12,288 MB budget and forces
  the prefix cache down to its 2,048-token floor; with 4 slots the prefix cache holds
  32,768 tokens (4,295 MB) instead. Measured after one request: 18,367 MiB of 32,607 MiB.
- 9B and 27B cannot coexist (18.0 + 18.4 > 32.6 GB), so 8095 serves exactly one of them.
- GPU_VERIFY: it still kills the server, builds, and measures the three anchors on the
  9B int8 build (unchanged protocol), but its three tail restarts now call a `prod_up`
  helper that runs `scripts/prod_serve_27b.sh` when it exists and only falls back to
  `bin_canonical` + 9B args otherwise, so a verify run no longer leaves production on 9B.
  `GPU_VERIFY.sh` itself stays untracked.
- Client side: the Hermes desktop provider entry named for 9B was renamed to `denno-27b`
  (`api http://<gpu-host>:8095/v1`, `transport anthropic_messages`,
  `default_model local/qwen35-27b`). The first pin of 32,768 was rejected outright by
  Hermes Agent, which enforces `MINIMUM_CONTEXT_LENGTH = 64_000`
  (`agent/agent_init.py:_enforce_minimum_context`) and refused to build the agent, so the
  entry is pinned at 65,536 - the smallest window Hermes accepts.
- Practical note: 65,536 is a legality floor, not a comfort zone. Prefill measured
  34 tok/s and decode 180 ms/token, so a 10.8k-token prompt already costs 5 minutes
  (observed 4m59s) and VRAM peaks at 29,795 MiB with the transient prefill buffers, i.e.
  about 2.8 GB of headroom left. Keep desktop sessions short and start a new chat instead
  of letting the history grow.

### P5-4: per-layer benches instead of the 6-minute serve loop - DONE (main session, 2026-10-20)
- Full-model measurement (start 27B, send 3,020 tokens, read the profile) costs about
  six minutes per data point, which is too slow to iterate on a kernel. Each bottleneck
  is now measured in isolation at its real 27B shape, with synthetic weights in the same
  Q4_0 block layout: `docker exec denno-gpu bash /work/denno-beta/scripts/bench_all.sh`
  builds the two examples and prints the whole table in about 20 s. It leaves 8095 alone.
- `crates/denno-engine/examples/bench_q4_line.rs` - one Q4 linear layer.
  `--k/--n/--layers/--iters`, `--raw` skips the f32 stage (needed for `lm_head`, whose
  715 MB of Q4 weights would otherwise cost 5 GB of temporary f32).
- `crates/denno-engine/examples/bench_gdn.rs` - one GDN scan at nk 16 / nv 48 / dk dv 128
  / conv 4, for n = 1, 8, 32, 300. Checks determinism (same input twice, bit equal) and
  compares against `gdn::forward_token_core`.
- Both print a CPU-reference `maxabs`, so a fast-but-wrong kernel shows up as text.
- Baseline (27B q4, HEAD `db47d2a`, one A/B-free run, `--iters` default):
  | layer | m or n | ms/call | rate | x layers |
  |---|---|---|---|---|
  | gate_up | m=1 | 0.103 | not measured | 6.6 ms/token (x64) |
  | gate_up | m=2 | 0.141 | not measured | 9.0 ms/token (x64) |
  | gate_up | m=8 | 0.511 | not measured | 32.7 ms/token (x64) |
  | gate_up | m=64 | 2.349 | not measured | n/a |
  | gate_up | m=300 | 16.976 | not measured | n/a |
  | down_proj | m=1 | 0.069 | not measured | 4.4 ms/token (x64) |
  | down_proj | m=8 | 0.088 | not measured | 5.6 ms/token (x64) |
  | down_proj | m=64 | 2.222 | not measured | n/a |
  | down_proj | m=300 | 6.912 | not measured | n/a |
  | lm_head | m=1 | 0.659 | not measured | 0.7 ms/token (x1) |
  | GDN scan | n=1 | 0.147 | not measured | 7.1 ms/token (x48) |
  | GDN scan | n=8 | 0.306 | not measured | n/a |
  | GDN scan | n=32 | 0.815 | not measured | n/a |
  | GDN scan | n=300 | 5.745 | not measured | n/a |
  - Shapes: gate_up k5120 n34816; down_proj k17408 n5120, both x64 layers;
    lm_head k5120 n248320 x1 with `--raw`; GDN scan nk16 nv48 dk/dv128 conv4 x48 layers.
  - Reproduce with `bash scripts/bench_all.sh` on a GPU host (the script builds the two
    examples with `--features cuda` and prints the same table).
  - `down_proj m=300` is noisy: repeated runs were observed between 5.72 and 6.912.
    Treat 6.912 as the top of that range, not a stable value.
- Rate columns are **not measured** on this run: `bench_all.sh` without `--iters`
  prints ms/call, and GB/s / GFLOP/s were not recorded for `db47d2a`. Do not quote the
  old 240.9 / 213.2 / 254.1 GB/s figures with the new ms/call.
- Effect on the real model (27B q4 on 8095): decode went from 105.3 to **58.5 ms/token**.
  The gain is in the small-m GEMV path (B1), not in the GEMM rows: m=1/m=2/m=8 improved
  by roughly 4x / 5.7x / 6.3x while m=64 and m=300 stayed at 2.349 / 16.976 ms.
- Superseded baseline (**do not use**): the row set 0.416 / 0.807 / 3.216 / 2.596 /
  16.877 / 0.235 / 2.815 / 0.143 / 5.746, measured at `d949195`. It was replaced by B1
  (commit `59e9ce9`, PTX `0f0f4ae1`). It is kept only as history: the old table's
  reading ("m=2 doubles the cost", "m=8 worse than m=64") no longer describes the code.
- What the split shows now: the plain m=1 GEMV is no longer the loss (0.103 ms), and
  m=2 / m=8 scale roughly linearly (0.141 / 0.511). The prefill GEMM at m=300
  (16.976 ms) is unchanged by B1 and stays the open prefill item (B2, and the B3 note
  below); do not attribute the decode gain to it.
- Iteration cost is 12 s of build plus 5-20 s of run, with no server restart and no long
  prompt. A full-model run plus `GPU_VERIFY` stays the acceptance step, not the inner loop.

### P5-5: GDN chunk parallelism (Task B3) - CLOSED, target dropped, no speedup
- Status: **closed (s136, no code change).** The n=300 GDN row (5.745 ms/call) is **not**
  improved; the target ("n=300 about 4 ms") is dropped. Nothing was committed for B3
  attempt 1, and the two B2 attempts that followed were reverted (`2ab333d`, `db47d2a`).
- Reason 1 - there is no chunk to parallelise at n=300:
  - `GDN_CHUNK_MIN_TOKENS = 2048` (`scan_schedule::CHUNK_MIN_TOKENS`), so the routing is
    `n >= 2048` -> chunked path, otherwise -> single-shot `dno_gdn_scan`.
  - `chunk_plan(n, 2048, 1024, 16)` gives **n_chunks = 1** for n = 1 / 8 / 32 / 300, i.e.
    those four rows all run the single-shot kernel. Splitting the head across chunks or
    splitting work per chunk cannot help a kernel that has exactly one chunk.
- Reason 2 - the single-shot kernel is SM-starved and packing makes it worse:
  - `dno_gdn_scan` uses **1,120 B of local memory per thread** (`Scol[256]` is an
    indexed array, so it is spilled to local, not kept in registers), 66 regs/thread,
    1 barrier. One 128-thread block therefore wants ~143 KB of local, so residency is
    **one block per SM**.
  - The launch is 48 blocks (one per head) = **48 of ~170 SMs**; roughly 72% of the chip
    is idle for the whole scan.
  - Packing several heads into one block lowers the block count, which is the wrong
    direction: pack 2 -> 24 blocks, **pack 4 -> 12 blocks**. Fewer SMs with work, plus
    more register-heavy heads per SM. This was prototyped and discarded on the occupancy
    arithmetic, before any measurement.
- Reason 3 - no bit-exact route to more blocks:
  - Splitting `dv` across blocks keeps each column's accumulation order but also splits
    the `o_inv` 128-term sum and the `q_inv` / `k_inv` dk-term sums, which the kernel
    keeps as a single ascending sequential sum in thread 0 to match the CPU reference.
    That moves GDN output, i.e. the three 9B anchors would need re-baselining.
  - Lowering the chunk threshold so n=300 runs as 2 chunks raises the block count to 96,
    but multi-chunk routes through `W` / `M` (`Sinit[ci+1] = M[ci]*Sinit[ci] + W[ci]`),
    which is a different rounding order: the repo's own
    `gdn::tests::gdn_chunked_matches_ar_relaxed` asserts only `maxabs < 1e-3`, not bit
    equality. It is also a numerics change for every request at n >= 300, not a
    speed-only change.
- Why the target is dropped rather than re-specified for the production shape:
  - Production 3,020-token prefill already takes the **chunked** path: **3 chunks**
    (1,007 each), **144 blocks** on pass 1 / pass 3 and **48 blocks** on pass 2. The
    n=300 gate row is a bench shape, not a production shape.
  - So the bench row measures a kernel path that production 3,020 does not use, and the
    chunked path is already spread over 144 blocks. There is no bit-exact route from 48
    blocks to ~144 blocks inside the single-shot kernel.
- Not measured: the split of the GPU's extra ~17 us/token at n=300 between the 4-pass
  S-column walk in local memory and the two 128-term scalar sums. A CPU-reference
  stand-in was measured (release, `scan_chunked_core`, nc=1): n=1 69.3 ms, n=8 81.7 ms,
  n=32 128.2 ms, n=300 657.0 ms (2.19 ms/token), but that is a CPU model, not the GPU
  kernel breakdown.
- Explicitly **no speedup is claimed** for B3, and none was measured.

## Open levers after P24-P28 (2026-10-23)

One question only: what is still open? Nothing new is added here - every line points at a task
that already exists above. P24-P27 closed the host-side queue for the prefill phases they
measured, so the host side is not an open lever for those phases.

**(a) Device-side `.cu` work - main session only** (needs a `.cu` edit, PTX regeneration and
`GPU_VERIFY.sh`)
- P1-2 candidate 3: the `dno_attention_prefill` kernel rewrite (K/V shared per kv head via a
  transposed tile, or online softmax on a flash-style tile). Bounded at 1.5-4.8 % of the
  942 ms bucket, and the committed PTX targets `sm_75`, where mma is unavailable.
- P28 residue: an n-token sigmoid output-gate kernel - the existing `dno_attn_gate_mul` is
  single-token. This is the only >= 0.3 s candidate left in `full_core` (the 45.40 ms/layer
  host gate and the 690 MB/layer att round trip would go together).
- P25 residue: double buffering the k loop of `dno_gemm_q4_0` - only if the re-read in (b)
  still reports `sl_lay_ffn_down` at ~10 s. Wave quantisation is the `.cu`-free half of the
  same question and is the `DENNO_Q4_M_TILE` sweep, i.e. (b).

**(b) GPU measurement - main session only**
- P27 §5a: `DENNO_PF_DBG=1 python3 scripts/prefill_bench.py 65000 3` (discard req=1). That one
  command decides whether P25's residue still exists and what `full_core` is now.
- P25 §5b: the `DENNO_Q4_M_TILE=2048` / `4096` sweep (bit-identical, no rebuild) - only after
  the line above still reports ~10 s.
- P26 §6 / P28 §6: one nsys trace (`cuda_gpu_kern_sum,cuda_gpu_mem_time_sum`) to replace the
  [R]/[A] transfer unit prices with measured GB/s, and to see whether `cuMemHostAlloc` runs
  once per layer (16 x 345 MB pinned).
- P28 §6: the cold A/B of `sl_lay_full_core` and the RSS check (the -0.24 s is PREDICTED, not
  measured), then anchors 3/3 and decode ms/token.
- The human steps already recorded under P3-2, P4-1, P4-4 and P5-1.

**(c) Operator decision required**
- P5-5 reason 3: splitting `dv` across blocks (the 4-lane GDN split) keeps each column's
  accumulation order but splits the `o_inv` / `q_inv` / `k_inv` sums, which moves the GDN
  output - the anchors would need `ANCHOR_REBASE.sh` plus an approved numeric relaxation.
  This is the same class as P2-1's speculative decode ("anchor re-baseline, i.e. an operator
  decision"). No agent can take this decision.

**(d) CPU-only work a dev container can do**
- P4-5 residue: the GGUF blocks are still re-quantised from fp32 today (GGUF -> fp32 -> Q4_0)
  instead of being passed through verbatim, so load still pays the fp32 expansion (see the
  "Still open" bullets under P4-5). The `TQ1_0` / `IQ1_M` decoder item that used to sit here
  is done — `kquant.rs:591` / `kquant.rs:624`, 2344f67 (s157) — and was removed on 2026-10-23.
- P3-3 known limits: extend the smem-cast scanner (the row bound of a symbolic row, float2
  casts through a dynamic alias with a symbolic column, casts through an intermediate
  pointer).
- Measurement-only closures of the P24-P27 kind: any further phase that can be settled from
  code + arithmetic + a CPU microbenchmark, with no `.cu` and no GPU.
- Docs sync (this item): TASKS.md and KPI.md were the last files out of step with the
  P24-P28 closures.

## Suggested order

1. ~~P1-2 attention (942 ms bucket), candidate 4 = causal key-range truncation~~ → **rejected in
   s122** (and already in s118): the kernel's causal bound is per row, so the "fully masked
   tiles below `pos0 + i0`" set does not exist. **P1-2 is now fully closed** - candidates 1-4
   are all rejected with numbers, and P1-1, P1-3, P1-4 were closed in s91/s92/s121. There is no
   open prefill lever left; the remaining 942 ms would need a per-row key range (the "kernel
   rewrite" s117 bounded at 1.5-4.8 % of the bucket), so prefill work should stop here.
2. In parallel on the CPU-only agent: P3-1 (delete the dead kernel), P3-3, P4-1, P4-4.
3. P2-1 decode once P1 has plateaued; P3-4 the moment a second A/B is needed by hand.