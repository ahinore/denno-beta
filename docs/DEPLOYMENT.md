# Production Deployment, Security & Operations Checklist

This is the operational checklist for running denno in production. It covers the
three processes, their bind defaults, the TLS reverse proxy in front of them,
authentication and CORS, request-size and rate limits, graceful shutdown,
Prometheus and viewer observability, and the `denno doctor` pre-flight check.

Every flag, route, status code, and `DENNO_*` variable below was checked against
`crates/` while this page was written. Section 9 lists the commands used; run
them again after changing the CLI, the server, or the viewer.

## 0. Processes and default exposure

| Process | Implementation | Default listen address | Inbound TLS | Built-in auth |
|---|---|---|---|---|
| `denno serve <repo>` | `crates/denno-server/` (axum) | `127.0.0.1:8080` | no | optional API key |
| `denno serve-flash` | `crates/denno-cli/src/serve_flash.rs` (hand-rolled HTTP/1.1) | `127.0.0.1:8098` | no | none |
| `denno-view` | `crates/denno-view/` (axum) | `0.0.0.0:8093` | no | none |

None of the listeners terminates TLS. The `rustls` dependency in `denno-view`
is outbound only (the upstream WebSocket and `/chat` proxy). Production TLS
therefore belongs at a reverse proxy, and authentication that denno does not
provide belongs there too.

`denno serve` and `denno serve-flash` are separate implementations. They share
the `/v1/messages` request shape but not their flags, middleware, or defaults.
Do not assume a setting works on the other subcommand.

`denno-view` binds all interfaces by default and has no authentication. Put it
behind the proxy (with authentication) or restrict the published port; see
section 6.

---

## 1. Pre-flight: model cache and `denno doctor`

Run these on the target host before starting a server.

1. Cache the model for `denno serve` (the positional `<repo>` must already be
   in the local cache):

   ```console
   $ denno pull <repo>
   ```

2. Check the host and, for `serve-flash`, the split GGUF directory:

   ```console
   $ denno doctor --dir <model-dir> --json
   ```

   `denno doctor` reads file presence, `/proc`, CPUID, `statvfs`, and
   `getrlimit` only. It never initializes CUDA and never launches `nvidia-smi`.

   Exit codes:

   | Code | Meaning |
   |---|---|
   | `0` | Checks passed (or `--dir` was omitted, so no model check ran) |
   | `1` | The model directory check failed (missing shard, zero-byte shard, or no `.gguf`) |
   | `2` | Usage error (clap) |

   The JSON report is printed even when the model check fails. Automation
   should read the report rather than relying on the exit code alone:

   - top-level keys: `cpu`, `memory`, `gpu`, `model`, `disk`, `fd_limits`,
     `env`, `recommendations`
   - `model.ok` is the overall model verdict; `model.missing_shards` and
     `model.zero_byte_shards` name the failures; `model.dir` is `null` when
     `--dir` was omitted
   - `env` lists the effective `DENNO_*` variables; values whose names look
     secret (`*API_KEY*`, `*SECRET*`, `*TOKEN*`, `*PASSWORD*`, `*AUTH*`) are
     replaced with `<redacted>`
   - `recommendations` is a list of human-readable `serve-flash` flag and
     `DENNO_*` suggestions for the detected hardware

3. Confirm the resolved `serve-flash` configuration without binding a socket:

   ```console
   $ denno serve-flash --config flash.toml --print-config
   ```

   `--print-config` performs the same type and range validation as a real
   startup, prints each value with its origin (`cli`, `env`, `config`, or
   `default`), and exits.

4. Confirm the model id, bind address, and port are what the proxy expects.

5. Decide the secret injection path. Prefer the supervisor's environment-file
   mechanism over command-line arguments: `--api-key` and `--api-keys` are
   visible in the process list.

---

## 2. Binding and TLS reverse proxy

### 2.1 Bind address

Both servers default to loopback (`127.0.0.1`). Keep that default and let the
reverse proxy connect over loopback. Binding to a non-loopback address is an
explicit act:

```console
$ DENNO_API_KEYS="$(cat /run/secrets/denno_keys)" \
    denno serve <repo> --host 0.0.0.0 --port 8080
```

`--host 0.0.0.0` on `denno serve` must be paired with authentication
(`--api-key`, `--api-keys`, or `DENNO_API_KEYS`). `denno serve-flash` has no
authentication, so exposing it beyond loopback requires proxy-level
authentication and request limits.

### 2.2 Reverse proxy requirements

The proxy is responsible for TLS termination, client authentication, request
body limits, and connection timeouts. The minimum configuration for
`denno serve`:

- forward to `127.0.0.1:8080` (or the configured port);
- `client_max_body_size` at least as large as `--max-body-mb` (default 16 MiB),
  otherwise the proxy returns its own 413 before denno sees the request;
- `proxy_read_timeout` and `proxy_send_timeout` above the generation limit. If
  `--request-timeout-secs` is `0` (disabled), non-streaming requests can wait up
  to 900 s, so a 60 s proxy timeout is wrong;
- disable response buffering for SSE. denno already sends
  `x-accel-buffering: no` and `cache-control: no-store` on streaming responses,
  but `proxy_buffering off` is still required in nginx;
- upgrade WebSocket connections on `/v1/telemetry/ws` if the viewer is used;
- restrict `/metrics` to the Prometheus scraper. It is unauthenticated and
  exposes the model name, request counts, token counts, and latency
  distributions;
- restrict `/v1/telemetry/ws` to the viewer. It is unauthenticated when
  telemetry is enabled and streams inference internals.

An nginx example (replace the certificate paths and upstream port):

```nginx
server {
    listen 443 ssl;
    server_name _;

    ssl_certificate     /etc/denno/tls/fullchain.pem;
    ssl_certificate_key /etc/denno/tls/privkey.pem;

    client_max_body_size 16m;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_buffering off;
        proxy_read_timeout 900s;
        proxy_send_timeout 900s;
    }

    location = /metrics {
        allow 127.0.0.1;
        deny all;
        proxy_pass http://127.0.0.1:8080;
    }

    location = /v1/telemetry/ws {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 3600s;
    }
}
```

Put the API key check at the proxy as well if browsers or untrusted clients can
reach it. If the proxy strips or overwrites `x-api-key` / `Authorization`, the
server-side key check will fail with `401`.

For `denno serve-flash`, point the proxy at `127.0.0.1:8098` and apply the same
TLS, authentication, body-size, and timeout rules. There is no server-side body
limit on that path: `read_req` allocates from `Content-Length` directly, so the
proxy must cap request bodies.

For `denno-view`, see section 6; it needs proxy authentication more than the
servers do because it has none of its own.

---

## 3. `denno serve` (27B family, axum)

### 3.1 Command line

```console
$ DENNO_API_KEYS="$(cat /run/secrets/denno_keys)" \
  denno serve <repo> \
    --host 127.0.0.1 \
    --port 8080 \
    --name my-model \
    --max-body-mb 16 \
    --max-queue-depth 32 \
    --request-timeout-secs 300 \
    --log-format json \
    --cors-origin "<allowed-origin>"
```

`<repo>` is the positional model id. The weight directory must already be in
the local cache (`denno pull <repo>`), otherwise startup fails with `not cached`
before the listener binds.

| Flag | Default | Meaning |
|---|---|---|
| `<repo>` (positional) | required | Model id; also the default advertised name |
| `--host <HOST>` | `127.0.0.1` | Bind address |
| `--port <PORT>` | `8080` | Bind port |
| `--name <NAME>` | `<repo>` | Model id advertised to clients |
| `--api-key <KEY>` | none | Accepted key; comma-separated values also work |
| `--api-keys <KEYS>` | none | Comma-separated accepted keys; union with `--api-key` and `DENNO_API_KEYS` |
| `--max-tokens <N>` | `256` | Parsed for CLI compatibility but not passed to the server; see the note below |
| `--seed <N>` | none | Default sampler seed (fixed seed makes output deterministic) |
| `--prefix-cache-tokens <N>` | `DENNO_PREFIX_CACHE_MAX_TOKENS`, else context length | Token budget for the multi-entry prefix cache |
| `--quant <fp32\|int8>` | `fp32` | Weight quantization mode |
| `--max-queue-depth <N>` | `32` | Accepted-but-waiting requests; alias `--max-queue` |
| `--request-timeout-secs <S>` | `0` (off) | Wall-clock limit per request; `0` disables; capped at `86400` |
| `--repetition-penalty <F>` | none | Default repetition penalty (requests can override) |
| `--temperature-cap <F>` | none | Upper clamp on the effective temperature |
| `--log-format <text\|json>` | `text` | Access-log format; any other value is a usage error |
| `--cors-origin <ORIGINS>` | loopback only | `*` or a comma-separated origin allowlist |
| `--max-body-mb <MB>` | `16` | Request body limit in MiB; clap requires `>= 1` |

`--max-tokens` note: the flag is accepted (clap default `256`) but the server
does not use it. Requests that omit `max_tokens` get `DEFAULT_MAX_TOKENS`
(`1024`), and explicit values are clamped to `4096` by `resolve_max_tokens`.
Size client requests accordingly; do not rely on `--max-tokens` to lower the
server default.

### 3.2 Authentication

Authentication is enabled when at least one key is configured. The effective
key list is the union of:

- `--api-key` (comma-separated values are accepted),
- `--api-keys` (comma-separated),
- `DENNO_API_KEYS` (comma-separated).

The list is trimmed, de-duplicated, and stored as one comma-separated value.
Keys are never logged; the startup summary prints only the count. If all three
sources are empty, authentication is disabled and every API is open (the
historical default).

Accepted request headers:

```text
x-api-key: <key>
Authorization: Bearer <key>
```

`bearer` is matched case-insensitively. If both headers are present, one match
is enough. Missing or mismatched credentials return `401` with:

```json
{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}
```

Unauthenticated routes, by design:

| Route | Reason |
|---|---|
| `GET /health` | liveness probes do not carry keys |
| `GET /ready` | readiness probes do not carry keys |
| `GET /metrics` | Prometheus scrapers do not carry keys |
| `GET /v1/telemetry/ws` | viewer stream; returns `503` when telemetry is disabled |

Every other route requires the key when one is configured, including
`/v1/models`, `/v1/models/{id}`, `/v1/messages`,
`/v1/messages/count_tokens`, `/v1/chat/completions`, `/v1/completions`,
`/api/tags`, `/api/version`, `/api/show`, `/api/ps`, `/api/chat`,
`/api/generate`, `/v1/cache/stats`, and `/v1/cache`.

`denno serve-flash` has no API key support. Protect it at the proxy.

### 3.3 CORS

Resolution order: `--cors-origin` > `DENNO_CORS_ORIGIN` > built-in default.

| Value | Behaviour |
|---|---|
| unset or empty | only loopback origins (`http://localhost[:port]`, `https://localhost[:port]`, `http://127.0.0.1[:port]`, `https://127.0.0.1[:port]`, `http://[::1][:port]`, `https://[::1][:port]`) |
| `*` | any origin (`Access-Control-Allow-Origin: *`) |
| comma-separated list | exactly those origins; the loopback default is disabled, so add loopback entries explicitly |

Origins are compared after lowercasing and removing a trailing slash.
Preflight (`OPTIONS` plus `Access-Control-Request-Method`) is answered with
`204` by the CORS layer before routing and does not consume a rate-limit token.
Allowed methods are `GET, POST, OPTIONS`. The default allowed header list is
`content-type, x-api-key, authorization, anthropic-version, x-request-id`;
requested headers are echoed back. Exposed response headers are
`x-request-id`, `x-denno-ttft-ms`, `x-denno-decode-tps`. Credentials
(`Access-Control-Allow-Credentials`) are not enabled.

### 3.4 Body-size limit

Resolution order: `--max-body-mb` > `DENNO_MAX_BODY_MB` > `16` MiB. The value is
clamped to `1..=4096` MiB.

The limit is enforced twice:

1. a declared `Content-Length` above the limit is rejected before the body is
   read;
2. bodies without a trustworthy length are cut off by the axum body limit while
   streaming.

Oversized requests get `413` with a JSON envelope:

```json
{"type":"error","error":{"type":"invalid_request_error","message":"request body too large: the configured limit is N bytes"}}
```

The message contains only the configured byte limit, not the request content.
Set the proxy's `client_max_body_size` at or above this value.

### 3.5 Rate limit

Rate limiting is opt-in and controlled only by environment variables (there is
no CLI flag). Both are read once at startup.

| Variable | Default | Meaning |
|---|---|---|
| `DENNO_RATE_LIMIT_RPM` | unset = disabled | Allowed requests per minute; `0`, unset, or invalid disables limiting |
| `DENNO_RATE_LIMIT_BURST` | `max(1, rpm/60)` | Token-bucket capacity |

Only inference endpoints consume tokens:

`/v1/messages`, `/v1/messages/count_tokens`, `/v1/chat/completions`,
`/v1/completions`, `/api/chat`, `/api/generate`.

`/health`, `/ready`, and `/metrics` bypass the limiter. A trailing slash is
treated as the same path. When the bucket is empty the response is `429` with
`Retry-After` (at least 1 second) and an OpenAI-shaped body, even for Anthropic
and Ollama paths:

```json
{"error":{"message":"...","type":"rate_limit_error","param":null,"code":"rate_limit_exceeded"}}
```

Plan client retries around `Retry-After`; do not retry immediately.

`denno serve-flash` has no rate limiter. Enforce it at the proxy.

### 3.6 Queue depth and request timeout

Queue depth:

- resolution: `--max-queue-depth` > `DENNO_MAX_QUEUE_DEPTH` > `32`;
- clamped to `1..=65536`; `0` becomes `1`;
- the scheduler executes one generation at a time; the rest wait in the bounded
  queue;
- queue overflow is protocol-specific and always carries `Retry-After: 1`:
  - Anthropic (`/v1/messages`): `503` `overloaded_error`
  - OpenAI (`/v1/chat/completions`, `/v1/completions`): `429`
    `rate_limit_exceeded`
  - Ollama (`/api/chat`, `/api/generate`): `503`

Request timeout:

- resolution: `--request-timeout-secs` > `DENNO_REQUEST_TIMEOUT_SECS` > `0`;
- `0` disables the wall-clock limit; the maximum accepted value is `86400`;
- on expiry the generation is aborted, the KV slot is released, and the request
  is counted as `denno_requests_aborted_total{reason="timeout"}`;
- the handler waits `limit + 30 s` so the scheduler's own timeout fires first;
- with the timeout disabled, the non-streaming handler's hard wait is `900 s`;
  exceeding it returns `504` for that request path.

Set an explicit timeout when clients can request long generations. Otherwise a
single long request occupies the only worker and the rest wait in the queue.

### 3.7 Graceful shutdown (`SIGTERM` / `SIGINT`)

Shutdown does not terminate the process immediately:

1. readiness flips to draining; new generation requests are rejected with
   `503` (Anthropic `overloaded_error`, OpenAI and Ollama `503`);
2. in-flight work is allowed to finish for up to the drain timeout;
3. the listener closes and the process exits.

Drain timeout resolution:

| Variable | Default | Range |
|---|---|---|
| `DENNO_DRAIN_TIMEOUT_SECS` | `10` | `0..=600` seconds; wins over the ms variable |
| `DENNO_DRAIN_TIMEOUT_MS` | unset | `0..=600000` ms; read only when the seconds variable is absent |

`0` means "do not wait". `/health` keeps answering `200` during drain; `/ready`
returns `503`. Set the orchestrator's termination grace period above the drain
timeout, otherwise the process is killed before the drain finishes.

`denno serve-flash` follows the same signal flow: `GET` probes keep working,
non-`GET` requests get `503` with `overloaded_error`, and generation is given up
to `DENNO_DRAIN_TIMEOUT_MS` (default `10 s`). It does not read
`DENNO_DRAIN_TIMEOUT_SECS`.

### 3.8 Probes, metrics, tracing, and access logs

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

`status` is `ready`, `loading`, or `draining`. Use `/health` for liveness and
`/ready` for readiness; never point a readiness probe at `/health`, which stays
`200` even while the model is loading or the server is draining.

Prometheus: scrape `GET /metrics` with
`Content-Type: text/plain; version=0.0.4; charset=utf-8`. Series naming,
histogram buckets, and example PromQL are in `docs/OBSERVABILITY.md`. Key
series:

- `denno_model_info{model=...}`
- `denno_requests_in_flight`, `denno_requests_queued`
- `denno_requests_completed_total`, `denno_requests_aborted_total{reason}`
- `denno_prompt_tokens_total`, `denno_completion_tokens_total`
- `denno_kv_cache_read_tokens_total`, `denno_kv_reuse_ratio`
- `denno_ttft_seconds`, `denno_tpot_seconds`
- `denno_http_requests_total{endpoint,status}`

The recording gate is `DENNO_SERVER_METRICS` (default on). `0`, `false`, or `no`
stops recording HTTP counters, tokens, and histograms, while `/metrics` keeps
returning `200` with frozen values. The `in_flight` and `queued` gauges and the
abort counters still reflect the scheduler at scrape time. The variable is read
once; changing it requires a restart.

Request tracing: every response, including SSE and NDJSON, carries
`x-request-id`. A client-supplied value is sanitized (printable ASCII only,
`"` and `\` removed, truncated to 128 bytes) and reused; if nothing usable
remains the server generates `req-<12 hex><8 hex>`. The value appears in every
access-log line, so client and server logs join on it.

Completed non-streaming JSON responses that report token usage also carry:

- `x-denno-ttft-ms` — wall-clock from acceptance to the complete response;
- `x-denno-decode-tps` — `output_tokens / seconds`, including prefill (a lower
  bound on true decode speed).

Access logs: `--log-format json` emits one JSON line per request to the
`denno_access` tracing target with `ts_ms`, `method`, `path`, `status`,
`duration_ms`, and `request_id`. The default `text` format emits two lines
(`METHOD path rid=...`, then `-> status rid=...`). There is no environment
switch for the format. Logging is asynchronous and bounded: under backpressure
lines are dropped and a single `denno-logger: N log line(s) dropped` line is
emitted. Use `denno_http_requests_total` when exact counts matter.

### 3.9 Telemetry WebSocket

`GET /v1/telemetry/ws` is a WebSocket stream for `denno-view`. It is
unauthenticated. When `DENNO_TELEMETRY` is unset or `0` in the dense server,
the route returns `503` with `telemetry disabled (set DENNO_TELEMETRY=1)`.
When enabled it sends a `ServerInfo` frame first, then inference frames. A slow
viewer only drops frames; it does not block inference.

---

## 4. `denno serve-flash` (Qwen3.8-Flash-Next)

### 4.1 Flags

```console
$ denno serve-flash --dir <model-dir> --host 127.0.0.1 --port 8098
```

| Flag | Default | Meaning |
|---|---|---|
| `--dir <PATH>` | unset | Directory with the split GGUF shards; required unless supplied by env or config |
| `--host <HOST>` | `127.0.0.1` | Bind address |
| `--port <PORT>` | `8098` | Bind port; `0` lets the OS choose |
| `--max-ctx <N>` | `4096` | Context length (KV / GDN state capacity) |
| `--name <NAME>` | `qwen3.8-flash-next` | Advertised model id |
| `--max-tokens <N>` | `512` | Generation cap when the request omits `max_tokens` |
| `--mtp-path <PATH>` | unset | MTP (nextn) head safetensors; `config.json` must sit beside it |
| `--kv-slots <N>` | `2` | Context slots; `3+` enables the LCP + LRU prefix cache; `0` means unset |
| `--steer-recipe <PATH>` | unset | Steering recipe JSON applied at startup |
| `--vram-budget <auto\|MiB>` | unset | Derive unset `DENNO_*` defaults from VRAM, or override total VRAM |
| `--config <PATH>` | unset | TOML-subset config file (section 4.2) |
| `--print-config` | off | Print the resolved configuration and exit without serving |
| `--strict-env` | off | Exit instead of starting when startup `DENNO_*` diagnostics produce any warning |

`--dir` has no built-in default. When it is not given on the command line, in
the environment, or in the config file, startup fails with an error explaining
the three ways to set it.

### 4.2 TOML config file and precedence

`--config <PATH>` accepts a deliberately small TOML subset, parsed by
`crates/denno-cli/src/config.rs` (no `serde`/`toml` dependency):

- one `key = value` per line; `[table]` sections are rejected;
- values are `"..."` (with `\n \t \r \\ \"` escapes), `'...'` (literal), or a
  decimal integer;
- `#` starts a comment that runs to end of line;
- blank lines are ignored;
- unknown keys, wrong types, out-of-range integers, and duplicate keys are
  rejected with the file name and line number.

Accepted keys, in the order `--print-config` prints them:

`dir`, `host`, `port`, `max_ctx`, `name`, `max_tokens`, `mtp_path`,
`kv_slots`, `steer_recipe`, `vram_budget`.

Ranges enforced while loading and resolving: `port` `0..=65535`, `max_ctx`
`1..=4294967295`, `max_tokens` `1..=4294967295`, `kv_slots` `0..=8`.
`kv_slots = 0` is treated as unset and normalizes to the default `2`;
`--print-config` annotates that case.

Resolution order for every key, highest first:

1. command-line flag (only when explicitly typed; clap defaults are not used);
2. `DENNO_*` environment variable;
3. config file (`--config`);
4. built-in default.

Environment variables: `DENNO_FLASH_DIR`, `DENNO_FLASH_HOST`,
`DENNO_FLASH_PORT`, `DENNO_FLASH_MAX_CTX`, `DENNO_FLASH_NAME`,
`DENNO_FLASH_MAX_TOKENS`, `DENNO_MTP_PATH`, `DENNO_KV_SLOTS`,
`DENNO_STEER_RECIPE`, `DENNO_AUTO_CONFIG` (maps to `vram_budget`).

An environment variable that is empty or whitespace-only counts as unset.
Numeric values from the environment are range-checked after resolution, with
the variable name included in the error message.

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

### 4.3 Security differences from `denno serve`

- no API key support;
- no CORS allowlist: every response carries `Access-Control-Allow-Origin: *`;
- no request body limit: `Content-Length` is trusted and allocated directly;
- no rate limiter, queue-depth knob, or request-timeout knob;
- generation is serialized through a single worker, so excess concurrent
  connections wait in the job channel;
- the config and viewer routes (`/v1/intervene`, `/v1/model/select`,
  `/v1/steer/*`, `/v1/cache`) are unauthenticated.

Expose it only behind a proxy that provides authentication, a body-size cap,
rate limiting, and timeouts.

### 4.4 Probes, metrics, and shutdown

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

`/metrics` is gated by `DENNO_METRICS` (default on); `0`, `false`, or `no`
removes the route entirely and it falls through to `404`. Series names are a
different (vLLM-style) set from `denno serve`; see
`crates/denno-cli/src/serve_metrics.rs`.

Shutdown: `SIGTERM`/`SIGINT` set draining, `POST` requests get `503` with
`overloaded_error`, `GET` probes keep working, and generation waits up to
`DENNO_DRAIN_TIMEOUT_MS` (default `10 s`).

---

## 5. Prometheus and log pipeline

Minimum production wiring:

1. scrape `127.0.0.1:8080/metrics` (or the configured port) from the
   Prometheus agent on the same host, or through a proxy rule restricted to the
   scraper;
2. alert on `denno_requests_queued > 0` sustained, `denno_requests_aborted_total{reason="timeout"}`
   increasing, and `up == 0`;
3. ship the process's stderr/stdout access log; `--log-format json` is one
   object per request and joins to client logs on `x-request-id`;
4. do not treat the access log as an audit log: it is bounded and may drop
   lines under backpressure.

`serve-flash` exposes `/metrics` under the same path but with a different
series set and the `DENNO_METRICS` gate. Scrape it at its own port.

---

## 6. `denno-view` observability and exposure

`denno-view` is an operator UI, not part of the inference API. It has no
authentication and binds `0.0.0.0:8093` regardless of any flag (the port is set
by `VIEW_PORT`; there is no host setting). Restrict it with the proxy or with
the container's published ports. Do not expose it on an untrusted network: its
`/api/*` route proxies the upstream REST API, its `/code/src` route serves
server/engine source, and its `/chat` route relays to the upstream
`/v1/messages`.

Observability endpoints:

| Route | Semantics |
|---|---|
| `GET /healthz` | Minimal liveness: always `200` with `{"schema":1,"service":"denno-view","status":"ok","ok":true}`; does not inspect the upstream |
| `GET /status` | Viewer health and upstream telemetry freshness as JSON; returns `200` while the viewer process can answer |
| `GET /ws` | Viewer's own WebSocket relay to `VIEW_UPSTREAM` |
| `GET /metrics` | Not implemented |

`GET /status` body:

```json
{
  "schema": 1,
  "service": "denno-view",
  "status": "ok",
  "ok": true,
  "upstream": {
    "connected": true,
    "last_frame_age_ms": 120,
    "stale": false,
    "stale_after_ms": 5000
  },
  "frames": 12345,
  "bytes": 6789012
}
```

`upstream.connected` is `false` when the upstream WebSocket is down.
`upstream.stale` is `true` when no frame has arrived for more than 5000 ms,
including the case where no frame has ever arrived. Use `/healthz` for process
liveness and `/status` for "is the viewer still receiving telemetry"; a
`/status` response with `stale: true` is not a process failure.

Because there is no `/metrics`, scrape `/status` with an HTTP probe and alert on
`upstream.stale` or `upstream.connected == false`, not on HTTP status alone.

Environment variables: `VIEW_UPSTREAM` (default
`ws://127.0.0.1:8080/v1/telemetry/ws`), `VIEW_PORT` (default `8093`),
`VIEW_RECORD`, `VIEW_REPLAY`, `VIEW_CAPTURE`, `VIEW_TOKENIZER`.

Telemetry prerequisites:

- `serve-flash`: telemetry is on by default; `DENNO_TELEMETRY=0` disables it.
- `denno serve`: set `DENNO_TELEMETRY=1`; unset or `0` leaves
  `/v1/telemetry/ws` at `503`.
- `DENNO_TELEMETRY_TOPK=1` is required for the top-8 token display
  (`/2d` token probabilities and the `/3d` prediction head).
- `DENNO_TELEMETRY_ARCH=1` enables `RoleAct` / `RouterTopK` frames;
  `DENNO_TELEM_EXPERT_SKIP=1` additionally sends expert-skip data.
- `DENNO_TELEMETRY_ACT=1` enables layer activation frames, with
  `DENNO_TELEMETRY_ACT_TOPK` controlling K (default `64`).

The viewer's chat proxy does not forward `x-api-key` or `Authorization`. If the
upstream server requires a key, chat through the viewer returns `401`; point
the browser at the server directly or front both with a proxy that injects the
key.

---

## 7. Operations checklist

Before first start:

- [ ] `denno doctor --dir <model-dir> --json` exits `0` and `model.ok` is true.
- [ ] Model is cached (`denno pull <repo>`) or the split GGUF directory is
      complete.
- [ ] Bind address is loopback (`127.0.0.1`) and the proxy is the only ingress.
- [ ] TLS certificate and key are installed at the proxy.
- [ ] Proxy `client_max_body_size` is at least `--max-body-mb`.
- [ ] Proxy read/send timeouts exceed the request timeout (or are generous when
      it is disabled).
- [ ] Proxy disables buffering for SSE and upgrades `/v1/telemetry/ws` if used.
- [ ] `--api-key` / `--api-keys` / `DENNO_API_KEYS` are injected by the
      supervisor, not baked into an image layer.
- [ ] `--cors-origin` is an explicit allowlist if browsers use the API.
- [ ] `DENNO_RATE_LIMIT_RPM` and `DENNO_RATE_LIMIT_BURST` match the expected
      client concurrency.
- [ ] `--max-queue-depth` and `--request-timeout-secs` are set for the expected
      generation length.
- [ ] `DENNO_DRAIN_TIMEOUT_SECS` (dense) or `DENNO_DRAIN_TIMEOUT_MS` (flash) is
      below the orchestrator's termination grace period.
- [ ] Metrics gate is left at default unless recording is intentionally
      disabled, and `/metrics` is restricted.
- [ ] `denno-view` is behind the proxy or its port is not published.

After start:

- [ ] `/health` returns `200`.
- [ ] `/ready` returns `200` with `draining: false`.
- [ ] `/metrics` returns `200` and includes `denno_model_info`.
- [ ] A test request with no key returns `401` when authentication is enabled.
- [ ] A request with an oversized body returns `413` with a JSON envelope.
- [ ] `denno-view` `/healthz` returns `200` and `/status` reports
      `upstream.connected: true`, `upstream.stale: false` when telemetry is on.

Before shutdown:

- [ ] Send `SIGTERM`, not `SIGKILL`.
- [ ] Confirm `/ready` flips to `503` while `/health` stays `200`.
- [ ] Confirm the process exits within the drain timeout plus proxy grace.

---

## 8. What denno does not provide

Do not plan around features that are not in the code:

- no inbound TLS, certificate loading, or automatic certificate management;
- no user accounts, sessions, RBAC, or audit log;
- no multi-tenant isolation: one model and one generation worker per process;
- no API key support in `denno serve-flash` or `denno-view`;
- no configurable body limit or rate limit in `denno serve-flash`;
- no CORS allowlist in `denno serve-flash`;
- no `/metrics` endpoint in `denno-view`.

Those gaps are the reverse proxy's responsibility.

---

## 9. Verification

The facts in this document are checked against the sources with:

```console
# serve flags and parsed-but-unused --max-tokens
$ grep -n "arg(long\|max_tokens: _max_tokens" crates/denno-cli/src/main.rs

# serve-flash flags, config keys, env vars, and defaults
$ grep -n "arg(long" crates/denno-cli/src/main.rs
$ grep -n "DENNO_FLASH_\|DENNO_MTP_PATH\|DENNO_KV_SLOTS\|DENNO_STEER_RECIPE\|DENNO_AUTO_CONFIG" crates/denno-cli/src/config.rs

# auth, CORS, body limit, queue depth, request timeout
$ grep -n "API_KEYS_ENV\|DENNO_API_KEYS\|CORS_ORIGIN_ENV\|MAX_BODY_MB_ENV\|MAX_QUEUE_DEPTH_ENV\|REQUEST_TIMEOUT_SECS_ENV" crates/denno-server/src/lib.rs

# rate limiting and its response envelope
$ grep -n "RATE_LIMIT_RPM_ENV\|RATE_LIMIT_BURST_ENV\|RATE_LIMIT_INFERENCE_PATHS\|RATE_LIMIT_BYPASS_PATHS" crates/denno-server/src/ratelimit.rs

# drain timeout precedence
$ grep -n "DRAIN_TIMEOUT_SECS_ENV\|DRAIN_TIMEOUT_MS_ENV\|resolve_drain_timeout" crates/denno-server/src/lifecycle.rs

# metrics gate and series names
$ grep -n "ENV_ON\|denno_http_requests_total\|denno_requests_queued" crates/denno-server/src/metrics.rs

# routes
$ grep -n '\.route("' crates/denno-server/src/lib.rs
$ grep -n '"/health"\|"/ready"\|"/metrics"\|"/v1/limits"' crates/denno-cli/src/serve_flash.rs
$ grep -n '\.route("' crates/denno-view/src/main.rs

# viewer bind, status, and stale threshold
$ grep -n "VIEW_PORT\|VIEW_UPSTREAM\|STALE_AFTER_MS\|0.0.0.0" crates/denno-view/src/main.rs

# doctor report shape, exit path, and env redaction
$ grep -n "pub struct Report\|pub struct EnvReport\|pub struct FdLimitReport\|REDACTED\|bail!" crates/denno-cli/src/doctor.rs

# serve-flash has no body limit on Content-Length
$ grep -n "fn read_req\|vec!\[0u8; content_length\]" crates/denno-cli/src/serve_flash.rs
```

If any of these stop matching, update this guide in the same change.
