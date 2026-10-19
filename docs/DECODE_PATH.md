# Decode path inventory — where does 16.4 ms/token go?

> Scope: the C4 device-resident decode chain for Qwen3.5-9B int8 (`n == 1`),
> i.e. `Qwen35::forward_decode_chain` (`crates/denno-engine/src/model/qwen35.rs:1804`).
> This document is a **code inventory plus arithmetic**, not a measurement. Every
> number in it is either read out of the source or computed from bytes and the
> transfer bandwidth measured in s68 (`docs/HISTORY.en.md`, "Pinned staging made the
> D2H five times costlier"). No GPU was involved in writing it.
>
> Status: s108 (measurement and decision document only). No behaviour change; the
> Rust side is untouched except for a `prof` label fix that is display-only.

## 1. What one decode step does

The chain is one host round trip at the two ends (embedding in, logits out) and
everything between is enqueued on the CUDA **default stream**:

1. per step: rope table upload, four buffer reservations, embedding row upload
2. per layer (64): rmsnorm -> qkv (+z for GDN layers) -> [GDN scan | full attention]
   -> o_proj -> rmsnorm -> gate_up -> swiglu -> down
3. final norm + lm_head
4. **the step's only sync**: `chain_download(CH_LOGITS)` (`cuda.rs:2622`,
   `cuMemcpyDtoHAsync` + `cuCtxSynchronize`)
5. host bookkeeping: `chain_q35_kv_stage_flush` copies the staged k/v rows into the
   host `KvCache`, then `cache.advance(1)`

Because every launch and every copy goes to the *same* stream, the GPU needs no
dependency sync anywhere inside the step. Step *k* ends with the full logits on the
host, so step *k+1* starts with an empty device queue.

## 2. Every host<->device round trip, per step

Model dimensions used below are the 9B ones documented in `docs/ARCHITECTURE.en.md`
and in `cuda.rs`'s C4 comments: e = 4096, d = 36 layers (27 GDN + 9 full), vocab ≈
248k, GDN nk=16 / nv=48 / dk=dv=128 / d_conv=4, full attention kv_heads × dh = 1024
(kv_dim), full-attention heads ≈ 32 → hd = 4096.
Transfers listed as **B/step** with "d" as a variable where a formula is clearer.

| # | site (file:line) | API | kind | count/step | bytes/call | bytes/step | prof bucket |
|---|---|---|---|---|---|---|---|
| 1 | `qwen35.rs:1859` -> `cuda.rs:4833` `chain_q35_rope_pi_upload` | `cuMemcpyHtoDAsync` | H2D | 1 | n_rot×4 (nest/2 × 2 f32; ≤ 256 B, all full layers share it) | ≤ 256 B | none |
| 2 | `qwen35.rs:1870-1876` `chain_q35_step_reserve*` | `cuMemAlloc_v2`/`cuMemFree_v2` | alloc | 0 in the steady state | CH_QKV 4·max(16384, 9216) = 64 KiB, CH_ATT 16 KiB, aq 16 KiB, kv_stage d×2×1024×4 = 294,912 B | 0 in the steady state | none |
| 3 | `qwen35.rs:1879` -> `cuda.rs:2566` `chain_upload(CH_X)` | `cuMemcpyHtoDAsync` | H2D | 1 | e×4 = 16,384 B | 16,384 B | none |
| 4 | `cuda.rs:2755` (inside `chain_rmsnorm`) | `cuMemcpyHtoDAsync` | H2D | 3d (attn_ln1, ln2, final) | e×4 = 16,384 B | 3d×16,384 B = 1,769,472 B | none (only the timing wrapper `RMSNORM` sees the launch) |
| 5 | `cuda.rs:5021-5022` `copy_from_at` (full layers only) | `cuMemcpyHtoD` (**sync**) | H2D | 0 in the steady state (only when the mirror lost rows) | pos0×1024×4 per tensor | 0 | lands in `o_proj` (`::OPROJ`) via the timing wrapper of the calling layer |
| 6 | `cuda.rs:5078-5096` (full layers only) | `cuMemcpyDtoHAsync` ×2, **no sync** | D2H | 2×9 = 18 | kv_dim×4 = 4096 B (k, v) | 73,728 B | counted in `d2h_kv_stage` (asynchronous, so it carries no time) |
| 7 | `cuda.rs:5062` (full layers only) | `cuMemcpyDtoDAsync` | D2D (not host traffic) | 9 | 4096 B | 36,864 B | none |
| 8 | `cuda.rs:2646` (inside `chain_download_inner`, called from `qwen35.rs:2024`) | `cuMemcpyDtoHAsync` + `cuCtxSynchronize` | D2H + **the only sync** | 1 | vocab×4 | vocab×4 (≈ 993,280 B for 248,320) | `d2h_logits_copy` (download + sync). The wait part is pre-attributed by the extra probe sync: `d2h_logits_wait` (prof only), `d2h_logits_pfhold` on the first step |
| 9 | `qwen35.rs:2031` -> `cuda.rs:5459` `chain_q35_kv_stage_flush` | none (host memcpy) | host | 1 | 18×4096 B copied out of the pinned stage | 73,728 B host copy | `d2h_kv_flush` |

Not in the steady-state step: `chain_q35_gdn_flush_pending` (`cuda.rs:5365`, 27 layers
× `MemcpyDtoH` of conv 19,456 B + ssm 196,608 B ≈ 5.8 MB) runs **once per request**,
at the start of the next prefill or on a CPU fallback, not per decode step.
`chain_q35_kv_stage_begin` (`cuda.rs:5333`) calls `ensure_pin`, which does not grow
after the first call.

### 2.1 What the step does *not* do (each was checked against the code)

- **No per-layer sync.** `chain_rmsnorm`, `chain_linear_grouped`, `chain_swiglu`,
  `chain_q35_gdn_scan` and `chain_q35_fullattn` only enqueue; none of them calls
  `ctx.sync()` or `cuMemcpy` on the host. The GDN state is not copied back per step
  (s50 deltization, `cuda.rs:5310-5314`).
- **No per-step allocation.** All four `chain_q35_step_reserve*` requests match the
  step-start reservation, so `Scratch::ensure` is a no-op and `DevBuffer::from_slice`
  never runs in the step body (`q35_w_dev` fills its cache on the first step only).
- **No weight upload per step.** int8 weights live in the process-wide uid registry
  (`weights_for_group`, `cuda.rs:1680`), fp32 GDN/QK-Norm weights in the per-model
  `q35_w_cache` (`cuda.rs:4780`).
- **The KV mirror does not re-seed per step.** `with_mirror` only issues the H2D of
  rows `[have, pos0)` when `m.len[li] < t_total`. Each decode step writes row `pos0`
  on the device (k via the QK-Norm+rope kernel straight into the mirror,
  `cuda.rs:5037-5052`; v via D2D, `cuda.rs:5062`) and sets `m.len[li] = t_total`, so
  the next step finds `have == pos0` and copies nothing. The mirror is the authority;
  the host cache is a shadow filled by row 6 and flushed by row 9.

## 3. Breaking 16.4 ms/step into transfer, sync and compute

Transfer time computable from the table above. Bandwidth figures are the s68
measurements on this host (pinned D2H 14.3 GB/s, H2D 14.2 GB/s; `flagAsync` targets
are pageable, so 8.6 GB/s is the conservative number for D2H):

| traffic | bytes/step | time at 14.3 GB/s | time at 8.6 GB/s |
|---|---|---|---|
| H2D, rows 1+3+4 (rope table, embed, 3d norm weights) | 1,785,856 | 0.125 ms | 0.208 ms |
| D2H, row 6 (staged k/v rows, asynchronous) | 73,728 | 0.005 ms | 0.009 ms |
| D2H, row 8 (logits) | 993,280 | 0.069 ms | 0.116 ms |
| **total transfer** | **2,852,864** | **0.199 ms** | **0.333 ms** |
| host copy, row 9 (not GPU time) | 73,728 | — | ~0.002 ms at 4.7 GB/s |

So **at most ~0.33 ms of the 16.4 ms (2.0 %) is actually moving bytes**, and the whole
figure is independent of the transfer library. Run 6's asynchronous D2H is ordered
before the attention launch, so even that is hidden under compute.

The 16.4 ms therefore decomposes as:

- **(a) transfer: ≤ 0.33 ms** — all of it in rows 3/4/8; rows 6/9 are scheduling or
  duplicated work (section 4).
- **(b) unavoidable dependency sync: ≤ 0.07 ms** — exactly one, the logits D2H at the
  end of the step, and even that is at most 0.07 ms of copy; the sync before it would
  wait for the lm_head GEMV only *if* the rest of the step were already finished. It
  never is: the sync is the point where the step's GPU work completes, so what a wall
  clock puts in `d2h_logits_copy` is mostly (c).
- **(c) GPU compute: the rest, ≈ 16.0 ms (98 %)** — 64 layers × (2 rmsnorm +
    3-4 GEMMs + 1 swiglu + 1 attention/scan kernel) plus lm_head. This is the number
    the device would report as busy, modulo scheduling.

The earlier reading of `d2h_logits` (~13.6 ms/step in the 29-33 ms era) is consistent
with this: the bucket contains the one sync of the step, so it measures "how long the
GPU still needs", which is compute, not PCIe. s51 confirming that removing the
per-step KV-row D2H changed nothing (16.4 vs 16.5 ms) is the same statement from the
other side — rows 6+9 are 0.005 ms of bus time.

## 4. Candidate levers and their ceilings

Ceilings are computed in ms/step from the table; nothing here is measured on a GPU.

| lever | work | what is removed | ceiling (ms/step) | share of the 4.4 ms gap to P2-1 |
|---|---|---|---|---|
| L1: drop the 3d per-layer norm-weight H2D, keep the device copy of the norm weights | Rust only (`chain_rmsnorm` currently uploads `w` into the tail of the output buffer on every call) | 3d×16,384 B = 1.77 MB/step | 0.125 (byte bound, 0.208 at 8.6 GB/s) | 2.8 % |
| L2a: `chain_download` at the end of the step into the pinned stage already used for k/v, then flush | Rust only, bit-identical, judged above the "one sync, no dependency" line (see §5) | nothing on the bus; removes the pageable 3.6 KB allotment per step and gives the D2H pinned bandwidth | ≤ 0.05 | 1 % |
| L2b: keep logits on device, sample on device or D2H only the top-k | Rust only but `f32` logits and the sampler is bit-exact by contract (R-class review), so it is a numeric change; a device-side top-k needs a `.cu` | 0.9-1.0 MB/step of D2H | ≤ 0.07 | 1.6 % |
| L3: compute the rope table on the device instead of building it per step on the host | `.cu` (new kernel argument) | a host `powf`/`cos`/`sin` chain of n_rot elements per step. For n_rot = 128 that is 64 pairs, i.e. tens of microseconds of `libm` on the hottest decode path — the only per-step host *compute* in the chain | n_rot-dependent; estimate ≤ 0.05 for n_rot = 128 (transcendentals not measured here), plus a launch | ≤ 1 % |
| L4: fuse rmsnorm into the consuming GEMM quantize | `.cu`, numerically safe (the norm is replicated exactly, the fusion only moves the read) | 2d+1 kernel boundaries and 2d+1 reads of a 16 KB row | governed by launch overhead; for 3d+1 ≈ 109 launches at ~4-8 us each a fusing pass could plausibly reach the low hundreds of microseconds | up to ~5 % |
| L5: GDN scan / attention / GEMM kernel work | `.cu` | — | 98 % of the step sits behind these | the only lever with enough room for 4.4 ms |

Rows 1/3 (rope table and embedding upload) are 16.6 KB/step; there is nothing to win
there.

## 5. Verdict

**No Rust-only lever remains that can reach P2-1's 12 ms/token.** The best Rust-only
set is L1 + L3 (≈ 0.18 ms/step, ~1 %), and both are below the noise floor of an 8.6-14.3
GB/s host bus; L2a is a plumbing improvement for the *next* step (pinned even/odd
buffers) rather than a decode lever.

- The 16.4 ms is **GPU-bound**: one sync per step, no per-layer sync, no per-step
  allocation, and only ~0.3 ms of bus traffic. **P2-1 requires kernel work**
  (`.cu` + PTX regeneration + `GPU_VERIFY`, a human step on the GPU host).
- What Rust can still do is remove the *serial* parts around the GPU: the rope table
  per step (§4 L3) is the largest genuinely per-step host cost, and the logits
  download (L2a) is the only per-step host allocation left. Both are small.
- Two accounting gaps are worth fixing in `prof` (display only, done in this session):
  the H2D inside `chain_rmsnorm` and the embedding upload were attributed to the
  `rmsnorm`/`embed` timing wrappers, which look like compute; the GDN-state recovery
  and the prefill attention download were both filed under `d2h_src_other`. The next
  on-device profile should show the norm-weight upload as `d2h_src_linear` so L1's
  1.77 MB/step is visible without reading this document.

### 5.1 Things that are *not* on this path (checked, so they need no re-check)

- The GDN state is not transferred per step (s50). The 5.8 MB recovery happens once per
  request.
- The KV mirror does not re-seed per step; `chain_q35_fullattn`'s `have < pos0` branch
  is only taken after a rollback/restore.
- `cache.kv_bufs(0, 1)` in `forward` does not grow anything after prefill, so it is
  not a hidden allocation in the step.
- `chain_q35_step_reserve*` sizes match the in-chain `ensure` requests, so the step
  body never frees a buffer that an in-flight kernel is writing to.