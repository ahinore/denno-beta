# denno Quick Reference

One page for the things you need most often: which guide to read, which
subcommand to run, which route to call, and which `DENNO_*` variable to set.

Every command, route, flag, and environment variable on this page was checked
with `grep` against `crates/` on this tree. Defaults are the values in the
code. Examples bind to loopback only.

## Documentation index

| Guide | Read it for |
|---|---|
| [ARCHITECTURE.en.md](ARCHITECTURE.en.md) / [ARCHITECTURE.ja.md](ARCHITECTURE.ja.md) | Model internals, tensor layout, per-stage dataflow |
| [DETAILS.en.md](DETAILS.en.md) / [DETAILS.ja.md](DETAILS.ja.md) | Model dimensions, environment variables, other model families |
| [API_COOKBOOK.md](API_COOKBOOK.md) | Copy-pasteable examples for the Anthropic, OpenAI, and Ollama wire protocols |
| [api.md](api.md) | API reference notes (request/response shapes, error envelope) |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Running the two HTTP front ends, auth, CORS, body limits, shutdown, health probes |
| [PERFORMANCE_TUNING.md](PERFORMANCE_TUNING.md) | VRAM budgets, expert cache, prefill/decode knobs, auto-config |
| [BENCHMARKING.md](BENCHMARKING.md) | Reproducible `denno bench` runs and regression gating |
| [OBSERVABILITY.md](OBSERVABILITY.md) | `/metrics`, JSON access logs, `denno doctor`, telemetry traces |
| [TELEMETRY_PROTOCOL.md](TELEMETRY_PROTOCOL.md) | Binary frame format shared by server, viewer, and `denno-telemetry` |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Startup failures, slow requests, crashes, empty viewer |
| [VIEWER.en.md](VIEWER.en.md) / [VIEWER.ja.md](VIEWER.ja.md) | `denno-view` pages, selector syntax, telemetry controls |
| [ABLITERATION_GUIDE.md](ABLITERATION_GUIDE.md) | Residual capture, ablation, steering recipes, OFF/ON evaluation |
| [QUANT_SUPPORT_MATRIX.md](QUANT_SUPPORT_MATRIX.md) | Which GGUF quantization types are implemented and tested |
| [NUMERIC_POLICY.en.md](NUMERIC_POLICY.en.md) / [NUMERIC_POLICY.ja.md](NUMERIC_POLICY.ja.md) | Rounding and determinism rules for kernels |
| [DECODE_PATH.md](DECODE_PATH.md) / [PREFILL_PATH.md](PREFILL_PATH.md) | Decode and prefill hot paths, per-operator notes |
| [KPI.md](KPI.md) | Throughput and latency targets |
| [TASKS.md](TASKS.md) | Task log and measurement protocol |

## CLI cheat sheet

`denno` has two server front ends. `serve` is the multi-protocol server for the
27B-class engine; `serve-flash` is the minimal server for
Qwen3.8-Flash-Next (`qwen4exp`). Model-management and client subcommands work
against either.

| Subcommand | Purpose | Typical use |
|---|---|---|
| `denno pull <repo>` | Download a model from Hugging Face into the local cache | One-time setup |
| `denno models` | List cached models | Check what is available offline |
| `denno info <repo>` | Show resolved model information | Verify config/quantization before serving |
| `denno generate <repo> --prompt …` | Single-shot generation (debugging, no server) | Sanity-check weights and sampler |
| `denno serve <repo>` | Start the Anthropic-compatible multi-protocol HTTP server | Production 27B serving |
| `denno serve-flash` | Start the Qwen3.8-Flash-Next HTTP server | Flash-Next serving |
| `denno chat` | Interactive REPL against a running server | Manual testing of either API |
| `denno bench` | Reproducible pp/tg benchmark against a running server | Throughput and regression checks |
| `denno abliterate` | Record residuals, fit a direction, export `.dns`/recipe, evaluate OFF/ON | Steering/ablation workflow |
| `denno inspect <path>` | Read only the GGUF header: architecture, quant histogram, VRAM estimate | Preflight without loading weights |
| `denno tokenize --dir <dir>` | Tokenize text using only tokenizer metadata | Debug prompts and chat templates |
| `denno doctor` | Check CPU SIMD, RAM, CUDA/Level Zero presence, missing GGUF parts | First thing to run on a new host |

### Server flags

| Command | Flag | Default | Meaning |
|---|---|---|---|
| `serve` | `--host` | `127.0.0.1` | Bind address |
| `serve` | `--port` | `8080` | Bind port |
| `serve` | `--name` | repo id | Advertised model id; clients send this as `model` |
| `serve` | `--api-key` | unset | Require `x-api-key` / `Authorization: Bearer` |
| `serve` | `--api-keys` | unset | Comma-separated keys; union with `--api-key` and `DENNO_API_KEYS` |
| `serve` | `--max-tokens` | `256` | Default generation cap |
| `serve` | `--quant` | `fp32` | Weight quantization mode (`fp32` or `int8`) |
| `serve` | `--prefix-cache-tokens` | context length | Multi-entry prefix-cache budget |
| `serve` | `--max-queue-depth` (alias `--max-queue`) | `32` | Requests allowed to wait behind the one in flight |
| `serve` | `--request-timeout-secs` | `0` (off) | Wall-clock limit per request |
| `serve` | `--log-format` | `text` | `text` or one-JSON-line-per-request `json` |
| `serve` | `--cors-origin` | loopback only | `*` or comma-separated allowed origins |
| `serve` | `--max-body-mb` | `16` | Request body limit (MiB); overflow returns 413 |
| `serve-flash` | `--dir` | `DENNO_FLASH_DIR` | Split-GGUF directory |
| `serve-flash` | `--host` / `--port` | `127.0.0.1` / `8098` | Bind address and port |
| `serve-flash` | `--max-ctx` | `4096` | KV/GDN context capacity |
| `serve-flash` | `--name` | `qwen3.8-flash-next` | Advertised model id |
| `serve-flash` | `--max-tokens` | `512` | Per-request generation cap |
| `serve-flash` | `--kv-slots` | `2` | Context slots; `>= 3` enables the LCP/LRU prefix cache |
| `serve-flash` | `--mtp-path` | `DENNO_MTP_PATH` | Optional MTP (nextn) head safetensors |
| `serve-flash` | `--steer-recipe` | `DENNO_STEER_RECIPE` | Startup steering recipe (JSON) |
| `serve-flash` | `--vram-budget` | `DENNO_AUTO_CONFIG` | `auto` or MiB override; fills only unset `DENNO_*` values |
| `serve-flash` | `--config` | unset | TOML config file |
| `serve-flash` | `--print-config` | off | Print the resolved config and exit |

### Client flags

| Command | Flag | Default | Meaning |
|---|---|---|---|
| `chat` | `--url` | `http://127.0.0.1:8098` | Server base URL |
| `chat` | `--api` | `anthropic` | `anthropic` (`/v1/messages`) or `openai` (`/v1/chat/completions`) |
| `chat` | `--max-tokens` | `512` | Per-turn cap |
| `chat` | `--temperature` | `0.7` | Sampling temperature |
| `chat` | `--api-key` | unset | Credential forwarded to the server |
| `bench` | `--url` | `http://127.0.0.1:8098` | Server base URL |
| `bench` | `--pp` | `512,2048,8192` | Prompt-processing token counts |
| `bench` | `--tg` | `128` | Generation token count |
| `bench` | `--depths` | `0,2048,8192` | Context depths |
| `bench` | `--rounds` | `3` | Repetitions; median is reported |
| `bench` | `--json <path>` | unset | Write a JSON report |
| `bench` | `--compare <base> <new>` | unset | Compare two JSON reports without running |
| `bench` | `--fail-on-regress-pct` | unset | Exit non-zero when a metric regresses beyond this percentage |
| `abliterate` | `--pos-file` / `--neg-file` | required | One prompt per line; `#` starts a comment |
| `abliterate` | `--alpha` | `1.0` | Steering strength |
| `abliterate` | `--capture-mode` | `all` | `all` or `last_prompt_token` |
| `abliterate` | `--out-recipe` | unset | Write a portable recipe JSON |
| `inspect` | `--json` | off | Machine-readable output |
| `inspect` | `--expert-cache-mib` | `4096` | Expert-cache allowance for the VRAM estimate |
| `tokenize` | `--text` / `--stdin` | one required | Input text |
| `tokenize` | `--json` | off | Machine-readable output |
| `doctor` | `--dir` | unset | Also check a split-GGUF directory |
| `doctor` | `--json` | off | Machine-readable output |

## HTTP routes

`serve` implements the OpenAI, Anthropic, Ollama, and operations/admin groups
below. `serve-flash` implements the Anthropic routes, `/v1/models`,
`/v1/telemetry/ws`, health/metrics, the model-control routes, and the
steering/admin routes — it returns `404` for the OpenAI chat
completions/completions routes and all Ollama routes.

### OpenAI

| Method | Path | Notes |
|---|---|---|
| `GET` | `/v1/models` | Model list; `model_dir` and weight mtime are included (both servers) |
| `GET` | `/v1/models/{id}` | Single model object (`serve` only) |
| `POST` | `/v1/chat/completions` | Chat completions, streaming and non-streaming |
| `POST` | `/v1/completions` | Legacy text completions |

### Anthropic

| Method | Path | Notes |
|---|---|---|
| `POST` | `/v1/messages` | Messages API, non-streaming and SSE |
| `POST` | `/v1/messages/count_tokens` | Token count without generating (`serve` only) |

### Ollama

| Method | Path | Notes |
|---|---|---|
| `GET` | `/api/tags` | Installed-model list |
| `GET` | `/api/version` | Version string |
| `POST` | `/api/chat` | Chat facade |
| `POST` | `/api/generate` | Generate facade |

### Operations and admin

| Method | Path | Notes |
|---|---|---|
| `GET` | `/health` | Liveness; no auth |
| `GET` | `/ready` | Readiness; `503` while draining |
| `GET` | `/metrics` | Prometheus text; absent when metrics are disabled |
| `GET` | `/v1/telemetry/ws` | Binary telemetry WebSocket |
| `GET` | `/v1/cache/stats` | Prefix-cache hit/miss/entry stats (both servers) |
| `GET` | `/v1/cache` | Cache stats (`serve-flash`) |
| `DELETE` | `/v1/cache` | Drop retained prefix entries; in-flight work is untouched |
| `GET` | `/v1/limits` | Advertised context and token limits (`serve-flash`) |
| `GET` | `/v1/model/shape` | Model topology description (`serve-flash`) |
| `GET` | `/v1/model/choices` | List selectable models (`serve-flash`) |
| `POST` | `/v1/model/select` | Switch the active model (`serve-flash`) |
| `GET`/`POST`/`DELETE` | `/v1/intervene` | Read, set, or clear expert/block interventions (`serve-flash`) |
| `GET` | `/v1/experts/stats` | Expert-residency counters (`serve-flash`) |
| `POST` | `/v1/abort` | Abort the in-flight generation (`serve-flash`) |
| `GET` | `/v1/steer` | Steering state (`serve-flash`) |
| `POST` | `/v1/steer/record`, `/v1/steer/record/stop` | Start/stop activation recording (`serve-flash`) |
| `POST` | `/v1/steer/apply` | Apply a direction (`serve-flash`); `DELETE` clears it |
| `POST` | `/v1/steer/analyze` | Fit a direction from recorded snapshots (`serve-flash`) |
| `GET`/`POST` | `/v1/steer/export`, `/v1/steer/import` | Move directions between processes (`serve-flash`) |
| `POST` | `/v1/steer/recipe/export`, `/v1/steer/recipe/import` | Move recipes (`serve-flash`) |
| `DELETE` | `/v1/steer/snapshots/{name}` | Delete one snapshot (`serve-flash`) |

### Viewer (`denno-view`)

| Method | Path | Notes |
|---|---|---|
| `GET` | `/` | Dashboard: rates, KV/VRAM, op graph, API examples |
| `GET` | `/2d` | 2D architecture map, interventions, ablation workbench |
| `GET` | `/3d` | 3D topology view with chat |
| `GET` | `/ws` | Viewer WebSocket proxying upstream telemetry |
| `GET` | `/chat` / `POST` | Chat page and relay to upstream `/v1/messages` |
| `GET` | `ANY /api/*` | REST proxy to the upstream server |
| `GET` | `/frames.mjs`, `/dash.mjs`, `/view2d.mjs`, `/nodes2d.mjs`, `/hud2d.mjs`, `/keys3d.mjs` | JS modules |
| `GET` | `/replay`, `/capture.bin` | Recorded frame stream (`VIEW_REPLAY`, `VIEW_CAPTURE`) |

## Environment variables

Resolution order for `serve-flash` is CLI flag > `DENNO_*` > `--config` file >
built-in default. Empty or whitespace-only values count as unset.

### Serving and HTTP

| Variable | Default | Effect |
|---|---|---|
| `DENNO_API_KEYS` | unset | Comma-separated accepted API keys (`serve`); union with `--api-key`/`--api-keys` |
| `DENNO_CORS_ORIGIN` | loopback only | `*` or comma-separated origins (`serve`) |
| `DENNO_MAX_BODY_MB` | `16` | Request body limit in MiB (`serve`) |
| `DENNO_MAX_QUEUE_DEPTH` | `32` | Queue depth before overload responses (`serve`) |
| `DENNO_REQUEST_TIMEOUT_SECS` | `0` (off) | Per-request wall-clock limit (`serve`) |
| `DENNO_PREFIX_CACHE_MAX_TOKENS` | context length | Prefix-cache token budget (`serve`) |
| `DENNO_MODEL_ALIASES` | unset | Extra `model` aliases, `alias` or `alias=target`, comma-separated |
| `DENNO_DRAIN_TIMEOUT_MS` | `10000` | How long shutdown waits for in-flight generation |
| `DENNO_SERVER_METRICS` | on | `0`/`false`/`no` removes `/metrics` (`serve`) |
| `DENNO_METRICS` | on | `0`/`false`/`no` removes `/metrics` (`serve-flash`) |
| `DENNO_PROFILE` | off | `1` emits one-line slot-release/profile logs (`serve`) |

### Flash-Next server

| Variable | Default | Effect |
|---|---|---|
| `DENNO_FLASH_DIR` | unset | Split-GGUF directory (`--dir`) |
| `DENNO_FLASH_HOST` | `127.0.0.1` | Bind address |
| `DENNO_FLASH_PORT` | `8098` | Bind port |
| `DENNO_FLASH_MAX_CTX` | `4096` | Context capacity |
| `DENNO_FLASH_NAME` | `qwen3.8-flash-next` | Advertised model id |
| `DENNO_FLASH_MAX_TOKENS` | `512` | Per-request generation cap |
| `DENNO_KV_SLOTS` | `2` | Context slots; `>= 3` enables the prefix cache |
| `DENNO_MTP_PATH` | unset | MTP/nextn head safetensors; unset means no MTP |
| `DENNO_STEER_RECIPE` | unset | Startup steering recipe |
| `DENNO_AUTO_CONFIG` | unset | `1`/`auto` derives unset defaults from VRAM and RAM |
| `DENNO_MODEL_CHOICES` | unset | File listing selectable model ids |
| `DENNO_MODEL_SELECT` | unset | File holding the current selection |
| `DENNO_TELEM_LAZY` | on | Skip top-k/activation/arch frame construction while no telemetry subscriber is connected; `0`/`off`/`false` always builds them |

### Telemetry and viewer

| Variable | Default | Effect |
|---|---|---|
| `DENNO_TELEMETRY` | on | `0` disables the telemetry channel and thread |
| `DENNO_TELEMETRY_ACT` | off | `1` sends per-layer neuron activations; `DENNO_TELEMETRY_ACT_TOPK` sets K (default 64) |
| `DENNO_TELEMETRY_TOPK` | off | `1` sends sampler top-8 and entropy |
| `DENNO_TELEMETRY_ARCH` | off | `1` sends role activations and router top-k |
| `DENNO_TELEM_EXPERT_SKIP` | off | `1` also sends skipped-expert frames (needs `DENNO_TELEMETRY_ARCH`) |
| `VIEW_UPSTREAM` | `ws://127.0.0.1:8080/v1/telemetry/ws` | Upstream telemetry WebSocket |
| `VIEW_PORT` | `8093` | Viewer listen port |
| `VIEW_RECORD` | unset | Append received frames to this file |
| `VIEW_REPLAY` | unset | Serve a recording at `/replay` |
| `VIEW_CAPTURE` | unset | Serve a frame stream at `/capture.bin` |
| `VIEW_TOKENIZER` | unset | `tokenizer.json` used to render token text |

### Performance (27B path)

| Variable | Default | Effect |
|---|---|---|
| `DENNO_EXPERT_VRAM_MB` | `0` (off) | Cap for resident experts in VRAM (MiB) |
| `DENNO_GPU_WEIGHT_MB` | `0` (unlimited) | Dense-weight registry byte cap; least-recently-used entries are evicted |
| `DENNO_VRAM_RESERVE_MB` | `1024` | Headroom subtracted from the card budget |
| `DENNO_EXPERT_GPU` | off | Run supported expert GEMV on the GPU |
| `DENNO_EXPERT_IQ_GPU` | off | Enable the IQ-family expert VRAM path |
| `DENNO_EXPERT_GPU_MIN_HIT` | `0.5` | Recent-hit threshold below which the CPU path is used (0 = always GPU) |
| `DENNO_EXPERT_RAM_MB` | unset | Host-RAM expert cache size |
| `DENNO_KV_RADIX` | off | Radix-tree KV reuse when slots allow it |
| `DENNO_Q4_PROFILE` | off | Emit per-request layer-phase profile lines (`serve-flash`) |

The full knob list, with measured values and caveats, is in
[PERFORMANCE_TUNING.md](PERFORMANCE_TUNING.md).

## Common recipes

```bash
# 1. Download, inspect, and serve a 27B-class model on all protocols
denno pull <org>/<model>
denno inspect /path/to/model.gguf
denno serve <org>/<model> --host 127.0.0.1 --port 8080 --name demo

# 2. Serve Flash-Next and attach the viewer
DENNO_TELEMETRY=1 denno serve-flash \
  --dir /path/to/qwen38-flash-next-gguf --port 8098 --max-ctx 65536
VIEW_UPSTREAM=ws://127.0.0.1:8098/v1/telemetry/ws \
  VIEW_TOKENIZER=/path/to/tokenizer.json \
  denno-view            # open http://127.0.0.1:8093/ and /3d

# 3. Measure and gate
denno bench --url http://127.0.0.1:8098 --json /tmp/bench.json
denno bench --compare /tmp/base.json /tmp/bench.json --fail-on-regress-pct 5

# 4. Preflight
denno doctor --dir /path/to/qwen38-flash-next-gguf
```

## Conventions worth remembering

- `serve` binds `127.0.0.1:8080` by default; `serve-flash` binds
  `127.0.0.1:8098`; `denno-view` binds port `8093`.
- Authentication is off unless a key is configured. When configured,
  `serve` accepts `x-api-key` and `Authorization: Bearer`.
- `/health` and `/ready` never require auth. `/metrics` does not either, because
  scrapers carry no key.
- `serve-flash` answers only the Anthropic route set plus the model/steering
  admin routes; use `serve` when a client needs OpenAI or Ollama compatibility.
- `denno bench` is a client: it never loads weights, so numbers belong to the
  server process.
- Telemetry is on by default for `serve-flash`; enable it explicitly with
  `DENNO_TELEMETRY=1` for `serve`.
