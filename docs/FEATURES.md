# Feature Matrix

One page that answers "what does this build actually do?". It covers the wire
protocols, every `denno` subcommand, the four `denno-view` pages, and the
observability surfaces. Each row was checked with `grep` against `crates/` on
this tree; the commands used are listed in [Verifying this document](#verifying-this-document).
Defaults are the values in the code. Examples bind to loopback only.

Two server front ends exist and they do **not** expose the same feature set:

- `denno serve <repo>` — the multi-protocol server for the 27B-class engine
  (default `127.0.0.1:8080`).
- `denno serve-flash` — the minimal server for Qwen3.8-Flash-Next
  (`qwen4exp`), with model-control and steering admin routes
  (default `127.0.0.1:8098`).

`serve-flash` answers the Anthropic route set plus its admin routes; it returns
`404` for the OpenAI chat/completions routes and all Ollama routes. Use `serve`
when a client needs OpenAI or Ollama compatibility.

## 1. Wire protocols

| Protocol | Method and path | Streaming | `serve` | `serve-flash` |
|---|---|---|---|---|
| Anthropic Messages | `POST /v1/messages` | JSON or SSE (`stream: true`) | yes | yes |
| Anthropic token count | `POST /v1/messages/count_tokens` | n/a | yes | no |
| OpenAI chat | `POST /v1/chat/completions` | JSON or SSE | yes | no |
| OpenAI completions | `POST /v1/completions` | JSON or SSE | yes | no |
| OpenAI models | `GET /v1/models` | n/a | yes | yes |
| OpenAI model by id | `GET /v1/models/{id}` | n/a | yes | no |
| Ollama tags | `GET /api/tags` | n/a | yes | no |
| Ollama version | `GET /api/version` | n/a | yes | no |
| Ollama chat | `POST /api/chat` | JSON or NDJSON | yes | no |
| Ollama generate | `POST /api/generate` | JSON or NDJSON | yes | no |
| Ollama show / ps | `POST /api/show`, `GET /api/ps` | n/a | yes | no |

On `serve`, all routes accept an optional API key, sent either as `x-api-key`
or as an `Authorization` header with a bearer token. Configure it with
`--api-key`, `--api-keys`, or `DENNO_API_KEYS`. Authentication is off unless a
key is set. `serve-flash` has no authentication. Overload responses carry
`Retry-After` (Anthropic `503`, OpenAI `429`, Ollama `503`).

## 2. CLI subcommands

`denno` has twelve subcommands. Only `chat`, `bench`, and `abliterate` need a
running server; the rest work directly on local files.

| Subcommand | Purpose | Needs a server | Machine-readable output |
|---|---|---|---|
| `denno pull <repo>` | Download a model from Hugging Face into the local cache | no | n/a |
| `denno models` | List cached models | no | n/a |
| `denno info <repo>` | Show resolved model information from `config.json` | no | n/a |
| `denno generate <repo> --prompt …` | Single-shot generation without a server (debugging) | no | n/a |
| `denno serve <repo>` | Multi-protocol HTTP server (Anthropic, OpenAI, Ollama) | no (it is the server) | n/a |
| `denno serve-flash` | Qwen3.8-Flash-Next HTTP server with admin routes | no (it is the server) | `--print-config` |
| `denno chat` | Interactive REPL against a running server | yes | n/a |
| `denno bench` | Reproducible pp/tg benchmark with JSON report and regression gate | yes | `--json`, `--compare`, `--compare-json` |
| `denno abliterate` | Residual capture, direction fit, `.dns`/recipe export, OFF/ON eval | yes | `--out-recipe`, `--dry-run` |
| `denno inspect <path>` | Read only the GGUF header: architecture, quant histogram, VRAM estimate | no | `--json` |
| `denno tokenize --dir <dir>` | Tokenize text using only tokenizer metadata | no | `--json` |
| `denno doctor` | Preflight: CPU SIMD, RAM, CUDA/Level Zero presence, missing GGUF shards | no | `--json` |

Notable flags per subcommand:

| Subcommand | Flags that change behaviour |
|---|---|
| `serve` | `--host`, `--port`, `--name`, `--api-key`, `--api-keys`, `--max-tokens`, `--quant`, `--prefix-cache-tokens`, `--max-queue-depth`, `--request-timeout-secs`, `--log-format`, `--cors-origin`, `--max-body-mb` |
| `serve-flash` | `--dir`, `--host`, `--port`, `--max-ctx`, `--name`, `--max-tokens`, `--kv-slots`, `--mtp-path`, `--steer-recipe`, `--vram-budget`, `--config`, `--print-config` |
| `chat` | `--url`, `--api` (`anthropic` or `openai`), `--system`, `--max-tokens`, `--temperature`, `--model`, `--api-key`, `--timeout` |
| `bench` | `--url`, `--api`, `--pp`, `--tg`, `--depths`, `--rounds`, `--json`, `--no-flush`, `--compare`, `--fail-on-regress-pct`, `--compare-json` |
| `abliterate` | `--pos-file`, `--neg-file`, `--eval-file`, `--capture-mode`, `--alpha`, `--out-dir`, `--out-recipe`, `--embed-snapshots`, `--dry-run` |
| `inspect` | `--json`, `--expert-cache-mib` |
| `tokenize` | `--dir`, `--text`, `--stdin`, `--json` |
| `doctor` | `--dir`, `--json` |

## 3. Viewer pages (`denno-view`)

The viewer listens on port 8093 by default (`VIEW_PORT`, all interfaces) and
relays the upstream telemetry stream from `VIEW_UPSTREAM`. All pages share
`GET /ws` on the viewer's own origin, so no CORS setup is needed.

| Path | What it shows | Extra requirements |
|---|---|---|
| `/` | Dashboard: live rates, KV/VRAM/latency, op timing, input tokens, op graph with code pane, API examples | — |
| `/2d` | Architecture map with interventions, steering, presets, selector DSL, abliteration workbench | `/api/*` REST proxy to upstream |
| `/3d` | 3D topology built from the model: layers, experts, KV band, prediction head, chat | `DENNO_TELEMETRY_TOPK=1` for sampler top-8 |
| `/chat` | Minimal chat page, `POST /chat` relays to upstream `/v1/messages` | upstream server |
| `/ws` | Telemetry WebSocket, proxied from upstream `/v1/telemetry/ws` | `DENNO_TELEMETRY` on the server |
| `/api/*` | REST proxy to the upstream server | upstream server |
| `/replay` | Recorded frame stream (`VIEW_REPLAY`) | a recording file |
| `/capture.bin` | Concatenated frame stream (`VIEW_CAPTURE`); `/3d?replay=1` uses it | a capture file |
| `/frames.mjs`, `/dash.mjs`, `/view2d.mjs`, `/arch2d.mjs`, `/nodes2d.mjs`, `/hud2d.mjs`, `/keys3d.mjs`, `/chat.mjs` | Pure-logic JS modules served as static assets | — |

Offline use: `VIEW_RECORD` appends received frames to a file, `VIEW_REPLAY`
serves them at `/replay`, and the `/2d` trace box can record and re-load up to
2,048 frames in the browser without touching inference.

## 4. Observability

| Surface | Where | Enable / disable | Notes |
|---|---|---|---|
| Prometheus metrics (`serve`) | `GET /metrics` | `DENNO_SERVER_METRICS` (default on; `0` freezes recording) | No auth; text format `version=0.0.4`; no `prometheus` crate |
| Prometheus metrics (`serve-flash`) | `GET /metrics` | `DENNO_METRICS` (default on) | Separate implementation in `serve_metrics.rs` |
| JSON access log | stdout, one line per request | `serve --log-format json` | Fields `ts_ms`, `method`, `path`, `status`, `duration_ms` |
| Text access log | stdout | `serve --log-format text` (default) | Two lines per request |
| Telemetry stream | `GET /v1/telemetry/ws` | `DENNO_TELEMETRY` (on for `serve-flash`, opt-in for `serve`) | Binary `DNTL` v1 frames, 17 kinds; drop-safe bounded channel |
| NDJSON traces | `denno-telemetry` encode/decode API | n/a | Same frames as JSON lines for offline trace files |
| Preflight | `denno doctor` | n/a | CPU SIMD, RAM, CUDA driver presence, Level Zero presence, split-GGUF gaps; does not run the GPU |
| Health / readiness | `GET /health`, `GET /ready` | n/a | No auth; `/ready` returns `503` while draining |
| Graceful drain | SIGTERM / SIGINT | `DENNO_DRAIN_TIMEOUT_MS` (default 10 s) | Waits for in-flight generation before exit |

`serve` metrics series: `denno_model_info`, `denno_requests_in_flight`,
`denno_requests_queued`, `denno_requests_completed_total`,
`denno_requests_aborted_total{reason}`, `denno_prompt_tokens_total`,
`denno_completion_tokens_total`, `denno_kv_cache_read_tokens_total`,
`denno_kv_reuse_ratio`, `denno_prefix_cache_hits_total`,
`denno_prefix_cache_misses_total`, `denno_prefix_cache_reused_tokens_total`,
`denno_prefix_cache_hit_rate`, `denno_ratelimit_rejections_total{reason}`,
`denno_ttft_seconds`, `denno_tpot_seconds`,
`denno_http_requests_total{endpoint,status}`.

`serve-flash` metrics add expert-residency and per-token timing series:
`denno_last_decode_ms_per_token`, `denno_last_prefill_tokens_per_second`,
`denno_expert_vram_hit_rate`, `denno_expert_disk_read_bytes_total`,
`denno_expert_vram_used_bytes`, `denno_expert_vram_budget_bytes`,
`denno_expert_ram_tier_bytes`, `denno_expert_ram_tier_experts`.

Telemetry frame kinds (v1): `SERVER_INFO`, `SESSION_ADMIT`, `SESSION_DONE`,
`PREFILL`, `TOKEN`, `STEP`, `OP_SUMMARY`, `KV_USAGE`, `GAP`, `TOP_K`,
`LAYER_RUN`, `LAYER_ACT`, `ROLE_ACT`, `EXPERT_RESIDENCY`, `TIER_STATS`,
`ROUTER_TOPK`, `EXPERT_SKIP`. Kind numbers are append-only and never reused.
The activation/architecture families are gated (`DENNO_TELEMETRY_ACT`,
`DENNO_TELEMETRY_TOPK`, `DENNO_TELEMETRY_ARCH`, `DENNO_TELEM_EXPERT_SKIP`).

## 5. Serving and operations

| Capability | `serve` | `serve-flash` | Control |
|---|---|---|---|
| API key auth | yes | no | `--api-key`, `--api-keys`, `DENNO_API_KEYS` |
| CORS | configurable | always `*` | `--cors-origin`, `DENNO_CORS_ORIGIN` (`serve` only) |
| Request body limit | yes | no | `--max-body-mb`, `DENNO_MAX_BODY_MB` (default 16 MiB) |
| Queue-depth overload | yes | no | `--max-queue-depth`, `DENNO_MAX_QUEUE_DEPTH` (default 32) |
| Per-request wall clock | yes | no | `--request-timeout-secs`, `DENNO_REQUEST_TIMEOUT_SECS` (default off) |
| Token-bucket rate limit | yes | no | `DENNO_RATE_LIMIT_RPM`, `DENNO_RATE_LIMIT_BURST` (default off) |
| Prefix / KV cache stats | yes | yes | `GET /v1/cache/stats`; `DELETE /v1/cache` drops entries |
| Multi-slot prefix cache | yes | yes (`--kv-slots >= 3`) | `--prefix-cache-tokens`, `DENNO_KV_SLOTS` |
| Model control | no | yes | `/v1/model/shape`, `/v1/model/choices`, `POST /v1/model/select`, `/v1/limits` |
| Interventions | no | yes | `GET/POST/DELETE /v1/intervene`, `/v1/experts/stats`, `POST /v1/abort` |
| Steering | no | yes | `/v1/steer*` record, analyze, apply, export/import, recipes, snapshots |
| Determinism | fixed `--seed` | n/a | Same seed gives the same output; int8 quant keeps same-seed determinism |

## Verifying this document

Run these from the repository root. Each command reads only source and docs.

```bash
# Routes registered by the multi-protocol server
grep -n '\.route(' crates/denno-server/src/lib.rs crates/denno-server/src/ollama.rs

# Routes answered by the flash server
grep -no '"/v1/[a-z_/{}]*"\|"/api/[a-z_/{}]*"\|"/healthz\?"' crates/denno-cli/src/serve_flash.rs

# CLI subcommands and their flags
grep -n 'enum Commands' -A 310 crates/denno-cli/src/main.rs

# Viewer pages and static assets
grep -n '\.route(' crates/denno-view/src/main.rs

# Metric names
grep -n '"denno_[a-z_]*"' crates/denno-server/src/metrics.rs crates/denno-cli/src/serve_metrics.rs

# JSON access-log fields
grep -n '"ts_ms"\|"duration_ms"' crates/denno-server/src/logging.rs

# Telemetry frame kinds
grep -n '^| kind |' docs/TELEMETRY_PROTOCOL.md
```

Documentation hygiene: this file contains no email addresses and no
non-loopback addresses. The only host references are `127.0.0.1` and
`localhost`, which the examples need in order to run.
