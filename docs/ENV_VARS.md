# DENNO_* Environment Variables Reference

This page lists the user-facing `DENNO_*` environment variables that
`denno-cli`, `denno-server`, `denno-engine`, `denno-telemetry`, and
`denno-view` read at runtime. Every default below was checked directly
against the source (not against older comments) when this page was written.

Scope notes:

- Only stable, user-facing knobs are documented. The engine also has a large
  set of internal tuning / experiment gates (prefill micro-stages, kernel
  selection A/B flags, `DENNO_PROFILE` sub-counters, etc.). Those are not part
  of the public surface and may change or disappear between releases.
- "unset" in the *Default* column means the variable has no built-in default;
  the code either treats an unset value as "feature off" or derives a value
  from context (card capacity, KV budget, model metadata).
- Boolean flags are generally enabled by `1`/`true`/`yes`/`on` (case
  insensitive where the source normalises it). Where a flag is enabled by
  default, the *disable* spelling is given.
- The serving variables apply to the `denno serve-flash` path unless marked
  `denno serve`.

## Serving and security

| Variable | Default | Meaning |
| --- | --- | --- |
| `DENNO_FLASH_HOST` | `127.0.0.1` | Bind host for `denno serve-flash`. |
| `DENNO_FLASH_PORT` | `8098` | Bind port. `0` lets the OS choose a free port. Range `0..=65535`. |
| `DENNO_FLASH_DIR` | unset | Model directory to load (required unless supplied another way). |
| `DENNO_FLASH_NAME` | `qwen3.8-flash-next` | Model name advertised by `/v1/models`. |
| `DENNO_FLASH_MAX_CTX` | `4096` | Maximum context length in tokens. |
| `DENNO_FLASH_MAX_TOKENS` | `512` | Default generated-token cap when a request omits `max_tokens`. |
| `DENNO_FLASH_MODEL_REUSE` | enabled | Reuse the decoder across requests. `0`/`false`/`no` rebuilds the model each request (slower, re-uploads weights). |
| `DENNO_FLASH_PREFIX_SPLICE` | enabled | Splice the cached prefix when the new prompt extends it. `0` disables and falls back to full prefill. |
| `DENNO_MTP_PATH` | unset | MTP / drafter weights path. Unset disables speculative decode. |
| `DENNO_STEER_RECIPE` | unset | Steering recipe name loaded at start. |
| `DENNO_STEER_DIR` | `/work/serve8098/steer` (code default) | Directory holding steering `*.dns` snapshots. |
| `DENNO_AUTO_CONFIG` | unset | `1`/`true`/`on`/`yes`/`auto` derives the VRAM/RAM budget from the machine. Invalid spellings are ignored as "off". |
| `DENNO_KV_SLOTS` | `2` | KV cache slot count, `1..=8`. `0`, typos, or out-of-range fall back to `2`. CLI flag wins over the env. |
| `DENNO_MAX_CTX` | deprecated | **Deprecated.** Not read for context length; use `DENNO_FLASH_MAX_CTX`. The CLI `doctor` warns when it is set. |
| `DENNO_API_KEYS` | unset (no auth) | `denno serve`: comma-separated accepted API keys. Empty/unset means the API is open. Values are trimmed, de-duplicated, and never logged. |
| `DENNO_CORS_ORIGIN` | loopback only | `denno serve`: `*` allows any origin; otherwise a comma-separated origin allowlist. Unset permits loopback (`localhost`, `127.0.0.1`, `[::1]`) on any port only. |
| `DENNO_MAX_BODY_MB` | `16` | `denno serve`: maximum request body size in MiB. Clamped to `1..=4096`. |
| `DENNO_RATE_LIMIT_RPM` | `0` (disabled) | `denno serve`: allowed requests per minute across inference endpoints. `0`, unset, or invalid disables limiting. |
| `DENNO_RATE_LIMIT_BURST` | `rpm / 60` (min `1`) | `denno serve`: token-bucket burst capacity. |
| `DENNO_MAX_QUEUE_DEPTH` | `32` | `denno serve`: maximum queued requests. Clamped to `1..=65536`; CLI flag wins. |
| `DENNO_REQUEST_TIMEOUT_SECS` | `0` (off) | `denno serve`: per-request timeout in seconds. `0` disables. Capped at `86400`. |
| `DENNO_DRAIN_TIMEOUT_SECS` | `10` | `denno serve`: graceful-shutdown drain wait, in seconds. |
| `DENNO_DRAIN_TIMEOUT_MS` | unset | Drain wait in milliseconds; lower priority than `DENNO_DRAIN_TIMEOUT_SECS`. Also honoured by `serve-flash` shutdown. |
| `DENNO_MODEL_ALIASES` | unset | `denno serve`: comma-separated `alias` or `alias=target` entries. An `alias=target` entry is ignored unless `target` matches the running model. |
| `DENNO_SERVER_METRICS` | enabled | `denno serve`: `0`/`false`/`no` stops Prometheus `/metrics` recording. The `/metrics` endpoint remains available. |

## Model / VRAM and RAM budget

| Variable | Default | Meaning |
| --- | --- | --- |
| `DENNO_VRAM_TOTAL_MB` | unset (detect card) | Override the total VRAM in MiB used for budget planning. `auto`, `0`, empty, or non-numeric means "detect". CLI `--vram-budget` takes priority. |
| `DENNO_VRAM_RESERVE_MB` | `1024` | Headroom (MiB) reserved for CUDA context, JIT, and fragmentation when computing usable VRAM. |
| `DENNO_VRAM_BUDGET_CARD` | enabled | Enforce a card-derived expert budget cap in addition to env budgets. `0`/`false`/`off`/`no` reverts to env-only budgets. |
| `DENNO_GPU_WEIGHT_MB` | `0` (unlimited) | Cap (MiB) for the GPU weight registry. Unset, empty, `0`, or invalid means unlimited (no eviction). |
| `DENNO_GPU_KV_MB` | unset (derived) | Explicit GPU KV budget in MiB. Unset derives it from the model and card. |
| `DENNO_KV_BUDGET_MB` | `12288` | Combined KV budget (MiB) used to size batch slots and the prefix cache when they are not set explicitly. |
| `DENNO_EXPERT_VRAM_MB` | unset (off) | Resident expert pool budget on GPU, in MiB. Unset / `0` disables the explicit pool. With `DENNO_VRAM_BUDGET_CARD` this acts as a lower bound. |
| `DENNO_EXPERT_VRAM_DYNAMIC` | off | `1` scales the expert pool with context length. Experimental; off by default. |
| `DENNO_EXPERT_RAM_MB` | unset (off) | Resident expert pool budget in host RAM, in MiB. `auto` derives it from `MemAvailable` minus headroom. `0`/unset disables. |
| `DENNO_EXPERT_CACHE_MB` | `0` (off) | Per-matrix expert cache size in MiB (`0` disables; forward then uses mmap + `prefetch()`). Three matrices are cached, so residency is about `3 * MB`. |
| `DENNO_EXPERT_CACHE_CAP_MB` | unset (unlimited) | Total cap (MiB, all three matrices) for the expert cache. When `DENNO_EXPERT_CACHE_MB` exceeds it, the cache is scaled down proportionally. |
| `DENNO_MMAP_WEIGHTS` | off | `1`/`true`/`yes`/`on` loads weights via mmap instead of materialising them on the heap. Other values keep the heap path. |
| `DENNO_Q4_PROFILE` | unset (off) | Presence-based: enables Q4 dequant/cache profiling counters. Any value turns it on. |
| `DENNO_MAX_PROMPT_TOKENS` | `32768` | Maximum prompt length in tokens; longer prompts are rejected with `400`. |
| `DENNO_MAX_SYSTEM` | unset | Truncate the system prompt to the first `N` characters before prefill. |
| `DENNO_STRIP_TOOLS` | off | `1`/`true`/`yes` drops client `tools` definitions before prefill (they can be thousands of tokens). |
| `DENNO_GGUF_FILE` | unset | Force a specific GGUF file name when a model directory holds several. |
| `DENNO_GGUF_VALIDATE_LAYOUT` | off | Validate every tensor's byte range against the file length on open. `0` explicitly disables. |

## CUDA tuning

These only take effect in CUDA-enabled builds. They are all optional; the
defaults are tuned for decode on a single device.

| Variable | Default | Meaning |
| --- | --- | --- |
| `DENNO_CUDA` | unset (off) | `1`/`true`/`on` enables the CUDA acceleration gate. Unset keeps the CPU path. |
| `DENNO_CUDA_DEVICE` | `0` | Ordinal of the CUDA device to use. |
| `DENNO_CUDA_MAX_M` | `16` | Largest row count `m` that is offloaded for decode/batch decode. Larger `m` (prefill GEMMs) stays on CPU. |
| `DENNO_CUDA_GEMM` | enabled | `0` disables the tiled GPU GEMM path for prefill. |
| `DENNO_CUDA_GEMM_MIN_M` | `256` | Smallest `m` that triggers the tiled GPU GEMM. `0` means "always once above `DENNO_CUDA_MAX_M`". |
| `DENNO_BF16_FAST` | enabled | `0`/`false`/`off` disables the fused BF16 FMA kernel used for MoE mix down/up. |
| `DENNO_BF16_EXACT` | off | `1`/`on`/`true`/`yes` forces the exact sequential BF16 path instead of the FMA kernel. |
| `DENNO_F32_EXACT` | off | `1`/`on`/`true`/`yes` forces the exact sequential F32 (router) path. |
| `DENNO_VNNI` | enabled | `0` disables the AVX-VNNI int8 kernel on supported CPUs. |
| `DENNO_QSA` | unset (off) | `1` enables the QSA attention kernel. |

## Prefill and KV cache

| Variable | Default | Meaning |
| --- | --- | --- |
| `DENNO_BATCH_SLOTS` | `32` | Upper bound on KV cache slots in `denno serve`; the effective count is also limited by the KV budget. Explicit values beat the budget. |
| `DENNO_PREFIX_CACHE_MAX_TOKENS` | derived from KV budget | Token cap for the multi-entry LRU prefix cache. Explicit values beat the budget. |
| `DENNO_KV_SLOT_TOK` | context length | Per-slot token cap for the multi-slot path. Values below `8`, `0`, unset, or invalid mean "up to the context length". |
| `DENNO_KV_RADIX` | off | `1`/`true`/`yes`/`on` selects slots via a radix (path-compressed trie) index with prefix-priority eviction. Only effective with 3 or more slots. |
| `DENNO_KV_SNAPSHOT_DIR` | unset (off) | Directory for KV snapshot persistence. Unset disables snapshots. |
| `DENNO_KV_SNAPSHOT_MB` | `512` | Disk quota for KV snapshots in MiB. `0` means unlimited. |
| `DENNO_PF_CHUNK` | auto | Overrides the prefill layer-loop chunk size (highest priority). Unset uses `1536` on the streaming path and `512` otherwise. |
| `DENNO_PREFILL_BATCH_CHUNK` | streaming `1024`, otherwise `512` | Tokens per prefill batch chunk when `DENNO_PF_CHUNK` is unset. Clamped to `1..=65536` (streaming) or `1..=512`. |
| `DENNO_PREFILL_BATCH` | off | `1` enables layer-first batched prefill. Other values leave it off. |
| `DENNO_PREFILL_WORK_MB` | `0` (off) | Reserve the given MiB as prefill working area before a request. `0`/unset disables the reservation. |
| `DENNO_PF_POOL_MB` | `2688` | Upper bound (MiB) borrowed from the prefill workspace pool per request. |
| `DENNO_PF_PREAD_THREADS` | `16` | Number of threads doing `pread` copies during expert prefetch. (An old in-source comment said 32; the constant is 16.) |

## Observability and profiling

| Variable | Default | Meaning |
| --- | --- | --- |
| `DENNO_TELEMETRY` | off in the library; `denno serve` treats unset as on | `0` disables the telemetry frame stream. In the library, an unset value means off; the server normalises unset to enabled. |
| `DENNO_TELEM_LAZY` | enabled | Skip building top-8 / layer-activity / arch frames when no viewer is subscribed. `0`/`off`/`false` always builds them (A/B). |
| `DENNO_TELEMETRY_TOPK` | off | `1` sends the top-k expert frame. Any other value keeps it off. |
| `DENNO_TELEMETRY_ARCH` | off | Enables the architecture frame. Requires `DENNO_TELEM_EXPERT_SKIP` support for expert-skip data. |
| `DENNO_TELEM_EXPERT_SKIP` | off | `1` collects expert-skip statistics (also requires `DENNO_TELEMETRY_ARCH`). |
| `DENNO_TELEMETRY_ACT` | off | Enables activation (top-k activation) frames. |
| `DENNO_TELEMETRY_ACT_TOPK` | `64` | Number of top activations per frame when `DENNO_TELEMETRY_ACT` is on. Invalid values fall back to `64`. |
| `DENNO_PROFILE` | off | Enables per-step profiling output. `0` disables; any other value enables. |
| `DENNO_PROFILE_SAMPLE` | `1` (every step) | Profile every `N`-th step. Clamped to `1..=1000000`; invalid values fall back to `1`. |
| `DENNO_DEV_PROFILE` | off | `1` enables device-side profiling counters. Only `1` enables it. |
| `DENNO_SEG_PROF` | off | Enables the host-side segment profiler (VRAM lookup / H2D / launch / sync / D2H phases). |

## Verifying a variable

Every name on this page appears in `crates/`. To check a single one against
the source:

```
grep -rn '"DENNO_FLASH_PORT"' crates/
```

If a variable is not listed here, it is either an internal experiment gate
(not stable) or no longer read by the code.
