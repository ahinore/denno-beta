# Deployment Guide

This guide covers the operational surface of the two HTTP front ends:

| Binary / subcommand | Model family | Implementation |
|---|---|---|
| `denno serve <repo>` | 27B family (`denno-engine`) | `crates/denno-server/` (axum) |
| `denno serve-flash` | Qwen3.8-Flash-Next (`qwen4exp`) | `crates/denno-cli/src/serve_flash.rs` (hand-rolled HTTP) |

Every flag, environment variable, route, and header named below exists in
`crates/` at the time of writing. Section 6 shows the commands used to check
that; run them again after touching the CLI or the server.

`denno serve` and `denno serve-flash` are separate code paths. They share the
`/v1/messages` request shape but not their configuration, authentication, or
probe set. Do not assume a flag works on the other subcommand.

---

## 1. `denno serve` (27B family, axum)

### 1.1 Command line

```console
$ denno serve <repo> \
    --host 127.0.0.1 \
    --port 8080 \
    --name my-model \
    --api-key <KEY> \
    --max-queue-depth 32 \
    --request-timeout-secs 300 \
    --log-format json \
    --cors-origin "https://127.0.0.1:5173"
```

`<repo>` is the positional model id; the model must already be in the local
cache (`denno pull <repo>`), otherwise startup fails with `not cached`.

| Flag | Default | Meaning |
|---|---|---|
| `<repo>` (positional) | required | Model id; also the default advertised name |
| `--host <HOST>` | `127.0.0.1` | Bind address |
| `--port <PORT>` | `8080` | Bind port |
| `--name <NAME>` | `<repo>` | Model id advertised to clients |
| `--api-key <KEY>` | none | Require auth; no environment-variable equivalent |
| `--max-tokens <N>` | `256` | Server default for requests that omit `max_tokens` |
| `--seed <N>` | none | Default sampler seed (fixed seed makes output deterministic) |
| `--prefix-cache-tokens <N>` | `DENNO_PREFIX_CACHE_MAX_TOKENS`, else context length | Token budget for the multi-entry prefix cache |
| `--quant <fp32\|int8>` | `fp32` | Weight quantization mode |
| `--max-queue-depth <N>` | `32` | Accepted-but-waiting requests; excess gets an overload response |
| `--request-timeout-secs <S>` | `0` (off) | Wall-clock limit per request; `0` disables |
| `--repetition-penalty <F>` | none | Default repetition penalty (requests can override) |
| `--temperature-cap <F>` | none | Upper clamp on the effective temperature |
| `--log-format <text\|json>` | `text` | Access-log format; any other value is a usage error |
| `--cors-origin <ORIGINS>` | loopback only | `*` or a comma-separated origin allowlist |

`--max-queue-depth` also accepts the visible alias `--max-queue`.

### 1.2 Shutdown

`SIGTERM` and `SIGINT` do not terminate the process immediately:

1. Readiness flips to draining, and new generation requests are rejected with
   an overload response (Anthropic `503`, OpenAI `429`, Ollama `503`, all with
   `Retry-After`).
2. In-flight work is allowed to finish for up to `DENNO_DRAIN_TIMEOUT_MS`
   (default `10000`, clamped to `0..=600000`; `0` means "do not wait").
3. The listener closes and the process exits.

Size the orchestrator's termination grace period above the drain timeout.
`/health` keeps answering during drain; `/ready` returns `503` (see section 4).

---

## 2. `denno serve-flash` (Qwen3.8-Flash-Next)

### 2.1 Flags

```console
$ denno serve-flash --dir <model-dir> --host 127.0.0.1 --port 8098
```

| Flag | Default | Meaning |
|---|---|---|
| `--dir <PATH>` | unset | Directory with the split GGUF shards |
| `--host <HOST>` | `127.0.0.1` | Bind address |
| `--port <PORT>` | `8098` | Bind port (`0` lets the OS pick) |
| `--max-ctx <N>` | `4096` | Context length (KV / GDN state capacity) |
| `--name <NAME>` | `qwen3.8-flash-next` | Advertised model id |
| `--max-tokens <N>` | `512` | Generation cap when the request omits `max_tokens` |
| `--mtp-path <PATH>` | unset | MTP (nextn) head safetensors; `config.json` must sit beside it |
| `--kv-slots <N>` | `2` | Context slots; `3+` enables the LCP + LRU prefix cache; `0` means unset |
| `--steer-recipe <PATH>` | unset | Steering recipe JSON applied at startup |
| `--vram-budget <auto\|MiB>` | unset | Derive unset `DENNO_*` defaults from VRAM, or override total VRAM |
| `--config <PATH>` | unset | TOML-subset config file (section 2.2) |
| `--print-config` | off | Print the resolved configuration and exit without serving |

`--dir` has no built-in default. When it is not given on the command line, on
the environment, or in the config file, startup fails with an error explaining
the three ways to set it.

### 2.2 TOML config file and precedence

`--config <PATH>` accepts a deliberately small TOML subset, parsed by
`crates/denno-cli/src/config.rs` (no `serde`/`toml` dependency):

- one `key = value` per line; `[table]` sections are rejected
- values are `"..."` (with `\n \t \r \\ \"` escapes), `'...'` (literal), or a
  decimal integer
- `#` starts a comment that runs to end of line
- blank lines are ignored
- unknown keys, wrong types, out-of-range integers, and duplicate keys are
  rejected with the file name and line number

Accepted keys, in the order `--print-config` prints them:

`dir`, `host`, `port`, `max_ctx`, `name`, `max_tokens`, `mtp_path`,
`kv_slots`, `steer_recipe`, `vram_budget`

Ranges enforced while loading and while resolving: `port` `0..=65535`,
`max_ctx` `1..=4294967295`, `max_tokens` `1..=4294967295`,
`kv_slots` `0..=8`. `kv_slots = 0` is treated as "unset" and normalizes to the
default of `2`; `--print-config` annotates that case.

Example:

```toml
# flash.toml
dir = "./models/flash-next"
host = "127.0.0.1"
port = 8098
max_ctx = 8192
name = "flash-next"
max_tokens = 1024
kv_slots = 4
```

Resolution order for every key, highest first:

1. command-line flag (only when explicitly typed; clap defaults are not used)
2. `DENNO_*` environment variable
3. config file (`--config`)
4. built-in default

Environment variables: `DENNO_FLASH_DIR`, `DENNO_FLASH_HOST`,
`DENNO_FLASH_PORT`, `DENNO_FLASH_MAX_CTX`, `DENNO_FLASH_NAME`,
`DENNO_FLASH_MAX_TOKENS`, `DENNO_MTP_PATH`, `DENNO_KV_SLOTS`,
`DENNO_STEER_RECIPE`, `DENNO_AUTO_CONFIG` (maps to `vram_budget`).

An environment variable that is empty or whitespace-only counts as unset.
Numeric values from the environment are range-checked after resolution, with
the variable name included in the error message.

Use `--print-config` to see the effective value and its origin before starting:

```console
$ denno serve-flash --config flash.toml --port 9001 --print-config
# denno serve-flash resolved config (cli > env > config > default)
dir = "./models/flash-next"  # source: config /path/to/flash.toml
host = "127.0.0.1"  # source: config /path/to/flash.toml
port = 9001  # source: cli
...
```

`--print-config` performs the same type and range validation as a real
startup, then exits without binding a socket. It is safe to run in CI.

---

## 3. Network and security settings

### 3.1 Bind address

Both subcommands default to `127.0.0.1`. Binding to a non-loopback address is
an explicit act: pass `--host 0.0.0.0` (or a specific interface address) and put
the process behind a reverse proxy or firewall. Nothing else in either server
opens a socket.

### 3.2 API key authentication (`denno serve` only)

`--api-key <KEY>` enables authentication. Accepted request headers:

- `x-api-key: <KEY>`
- `Authorization: Bearer <KEY>`

A missing or mismatched key returns `401` with an `authentication_error` body.
There is no environment-variable form of the key and no way to configure
multiple keys; pass the secret via the command line and keep it out of shell
history (`--api-key "$(cat ...)"` still lands in the process list; prefer your
supervisor's secret injection).

Unauthenticated routes: `/health`, `/ready`, `/metrics`. Every other route
(`/v1/models`, `/v1/messages`, `/v1/messages/count_tokens`,
`/v1/chat/completions`, `/v1/completions`, `/api/tags`, `/api/version`,
`/api/chat`, `/api/generate`, `/v1/telemetry/ws`, `/v1/cache/stats`,
`/v1/cache`) requires the key when one is configured.

`denno serve-flash` has no API key support. It must be protected by network
policy or a reverse proxy that performs authentication.

### 3.3 CORS (`denno serve` only)

Resolution order: `--cors-origin` > `DENNO_CORS_ORIGIN` > built-in default.

- unset or empty: only loopback origins (`127.0.0.1`, `localhost`, `[::1]`)
  from any port are allowed
- `*`: any origin is allowed
- comma-separated list: exactly those origins are allowed, and the loopback
  default is turned off, so include loopback entries explicitly if browser
  tooling needs them

Preflight (`OPTIONS` with `Access-Control-Request-Method`) is answered by the
CORS layer before routing. Allowed request headers include `content-type`,
`x-api-key`, `authorization`, `anthropic-version`, `x-request-id`. Exposed
response headers: `x-request-id`, `x-denno-ttft-ms`, `x-denno-decode-tps`.

`denno serve-flash` always sends `Access-Control-Allow-Origin: *` and has no
CORS flag. Do not expose it to untrusted browsers.

### 3.4 Queue depth and request timeout (`denno serve` only)

Queue depth:

- resolution: `--max-queue-depth` > `DENNO_MAX_QUEUE_DEPTH` > `32`
- clamped to `1..=65536`; `0` is raised to `1`
- a request that cannot be admitted gets an overload response with
  `Retry-After` (Anthropic `503`, OpenAI `429`, Ollama `503`)
- `/ready` and `/metrics` report the effective limit (`limits.max_queue_depth`
  and `denno_requests_queued`)

Request timeout:

- resolution: `--request-timeout-secs` > `DENNO_REQUEST_TIMEOUT_SECS` > `0`
- `0` disables the wall-clock limit; the maximum accepted value is `86400`
- on expiry the generation is aborted, the KV slot is released, and an error
  response is returned
- the HTTP handler waits slightly longer than the limit (limit + 30 s) so the
  scheduler's own timeout is the one that fires
- aborted requests are counted as `denno_requests_aborted_total{reason="timeout"}`

With the timeout disabled, the handler's hard ceiling is 900 s per
non-streaming request; a request that exceeds it returns `500`. Set an explicit
timeout when clients can send long generations.

`denno serve-flash` serializes generation through a single worker and has no
queue-depth or request-timeout knobs. Excess concurrent connections wait in the
job channel; apply client-side and proxy timeouts instead.

---

## 4. Probes, metrics, tracing, and access logs

### 4.1 `denno serve`

| Route | Auth | Semantics |
|---|---|---|
| `GET /health` | none | Liveness. Always `200` with `{"status":"ok"}` while the process runs |
| `GET /ready` | none | Readiness. `200` only when the model is loaded, the worker is alive, and the server is not draining; otherwise `503` |
| `GET /metrics` | none | Prometheus text exposition, always `200` |

`/ready` body (same shape for `200` and `503`):

```json
{
  "status": "ready",
  "ready": true,
  "draining": false,
  "model_loaded": true,
  "worker_alive": true,
  "model": "my-model",
  "context_length": 32768,
  "pending": 0,
  "in_flight": 1,
  "aborts": {"completed": 12, "timeout": 0, "disconnected": 0, "error": 0},
  "limits": {"max_queue_depth": 32, "request_timeout_secs": 300}
}
```

`status` is `ready`, `loading`, or `draining`. During shutdown the server keeps
`/health` at `200` while `/ready` returns `503`, so a liveness probe does not
restart the pod mid-drain and a readiness probe removes it from the endpoint
list first.

Use `/health` for liveness and `/ready` for readiness. Never point a readiness
probe at `/health`: it stays `200` even while the model is still loading.

Prometheus: scrape `GET /metrics`. Series naming, histogram buckets, and
example PromQL are in `docs/OBSERVABILITY.md`. The recording gate is
`DENNO_SERVER_METRICS` (default on; `0`, `false`, or `no` disables recording,
while `/metrics` keeps returning `200` with frozen values).

Request tracing: every response, including SSE and NDJSON, carries
`x-request-id`. A client-supplied `x-request-id` is sanitized (printable ASCII
only, `"` and `\` removed, truncated to 128 bytes) and reused; if nothing usable
remains the server generates `req-<12 hex><8 hex>`. The value appears in every
access-log line, so client logs and server logs join on it.

Completed non-streaming JSON responses that report token usage also carry:

- `x-denno-ttft-ms` — wall-clock from acceptance to the complete response
- `x-denno-decode-tps` — `output_tokens / seconds` including prefill (a lower
  bound on true decode speed)

Access logs: `--log-format json` emits one JSON line per request to the
`denno_access` tracing target with `ts_ms`, `method`, `path`, `status`,
`duration_ms`, and `request_id`. The default `text` format emits two lines
(`METHOD path rid=...` then `-> status rid=...`). There is no environment
variable switch. Logging is asynchronous and bounded: under backpressure lines
are dropped and a single `denno-logger: N log line(s) dropped` line is emitted.
Use `denno_http_requests_total` when exact counts matter.

### 4.2 `denno serve-flash`

| Route | Semantics |
|---|---|
| `GET /health` and `GET /healthz` | `200` with `{"ok":true,"model":"<name>"}` |
| `GET /ready` | `200` while serving, `503` after a shutdown signal sets draining |
| `GET /v1/models` | Advertised model, context length, capabilities |
| `GET /metrics` | Prometheus text; when `DENNO_METRICS` is `0`/`false`/`no` the route does not exist (`404`) |
| `POST /v1/messages` | Anthropic Messages API, JSON or SSE |
| `GET /v1/limits` | Generation limits (clients read the cap here instead of hard-coding) |
| `GET /v1/model/shape` | Layer/MoE shape for the viewer |
| `GET\|POST\|DELETE /v1/intervene` | Expert/block/residual intervention |
| `GET /v1/telemetry/ws` | WebSocket telemetry stream for the viewer |

`/ready` body: `status` (`ready` or `draining`), `ready`, `draining`,
`model_loaded` (always `true`, because the model is built before the listener
binds), `model`, `context_length`, `in_flight`.

Shutdown follows the same pattern as `denno serve`: `SIGTERM`/`SIGINT` set
draining, `POST` requests get `503` with `overloaded_error`, `GET` probes keep
working, and generation waits up to `DENNO_DRAIN_TIMEOUT_MS` (default 10 s).

The flash `/metrics` series are a different set from `denno serve`
(vLLM-style names, `model_name` label); see
`crates/denno-cli/src/serve_metrics.rs`.

---

## 5. Pre-flight checklist

Run these on the target host before starting the server.

1. Cache the model (or copy the shard directory).

   ```console
   $ denno pull <repo>
   ```

2. Check the model directory and the host with `doctor`. It reads only file
   presence, `/proc`, and CPUID; it never initializes CUDA or runs the GPU.

   ```console
   $ denno doctor --dir <model-dir> --json
   ```

   Exit codes: `0` OK, `1` model check failed (missing shard, zero-byte shard,
   or no `.gguf` at all), `2` usage error. The report is printed even on
   failure. Machine consumers should check `.model.ok` and
   `.model.missing_shards` / `.model.zero_byte_shards` rather than the exit code
   alone. Top-level keys are `cpu`, `memory`, `gpu`, `model`, `recommendations`;
   `recommendations` lists the `denno serve-flash` flags and environment
   variables suggested for the detected hardware.

3. Confirm the effective configuration.

   ```console
   $ denno serve-flash --config flash.toml --print-config
   ```

4. Confirm the bind address and port are what the proxy expects, and that the
   API key (if any) is injected by the supervisor rather than baked into an
   image layer.

5. Decide the probe wiring: liveness `/health`, readiness `/ready`. Both
   subcommands serve these; both keep `/health` at `200` and flip `/ready` to
   `503` during drain.

6. Set the termination grace period above `DENNO_DRAIN_TIMEOUT_MS`
   (default 10 s) for both subcommands.

7. Confirm the metrics gate and scrape path: `/metrics`, with
   `DENNO_SERVER_METRICS` (`denno serve`) or `DENNO_METRICS` (`denno serve-flash`)
   left at its default unless recording is intentionally disabled.

---

## 6. Verification

The facts in this document are checked against the sources with:

```console
# flags (both subcommands)
$ grep -n "arg(long" crates/denno-cli/src/main.rs

# serve-flash config keys, env vars, and defaults
$ grep -n "DENNO_FLASH_\|DENNO_MTP_PATH\|DENNO_KV_SLOTS\|DENNO_STEER_RECIPE\|DENNO_AUTO_CONFIG" crates/denno-cli/src/config.rs

# queue depth / request timeout resolution and limits
$ grep -n "MAX_QUEUE_DEPTH_ENV\|REQUEST_TIMEOUT_SECS_ENV\|DEFAULT_MAX_QUEUE_DEPTH\|MAX_QUEUE_DEPTH_LIMIT\|MAX_REQUEST_TIMEOUT_SECS" crates/denno-server/src/lib.rs

# CORS resolution and header names
$ grep -n "CORS_ORIGIN_ENV\|CORS_ALLOW_HEADERS\|CORS_EXPOSE_HEADERS" crates/denno-server/src/lib.rs

# routes
$ grep -n '\.route("' crates/denno-server/src/lib.rs
$ grep -n '"/health"\|"/metrics"' crates/denno-cli/src/serve_flash.rs

# readiness, drain, request id, access log
$ grep -n "serves_requests\|DEFAULT_DRAIN_TIMEOUT_MS\|MAX_REQUEST_ID_LEN\|access_log_json" \
    crates/denno-server/src/lib.rs crates/denno-server/src/lifecycle.rs crates/denno-server/src/logging.rs

# doctor exit codes and report shape
$ grep -n "model.ok\|bail!" crates/denno-cli/src/doctor.rs
```

If any of these greps stops matching, update this guide in the same change.
