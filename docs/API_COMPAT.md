# API Compatibility Matrix & Client Migration Guide

This document is the single-page reference for the three wire protocols that
`denno serve` exposes, and for the exact places where each one differs. Every
route, status code, header, and field name below was checked directly against
`crates/denno-server/src/` (`lib.rs`, `openai.rs`, `ollama.rs`,
`ratelimit.rs`). If this document and the code disagree, the code is right —
please fix this file.

The `crates/denno-server/tests/docs_consistency_1249.rs` gate scans this file
(and the rest of `docs/`) for leaked private addresses and host paths, so keep
examples on loopback.

## 1. Route inventory

`denno serve` (the axum router in `crates/denno-server/src/lib.rs`) listens on
`127.0.0.1:8080` by default. All routes below belong to that binary.

| Method | Path | Protocol | Streaming | Auth | Notes |
| --- | --- | --- | --- | --- | --- |
| GET | `/health` | ops | no | no | `{"status":"ok"}`, liveness only |
| GET | `/ready` | ops | no | no | 200 when serving, 503 while loading/draining |
| GET | `/metrics` | ops | no | no | Prometheus text format 0.0.4 |
| GET | `/v1/models` | OpenAI | no | yes | single-model list |
| GET | `/v1/models/{id}` | OpenAI | no | yes | 404 for an unknown id |
| POST | `/v1/messages` | Anthropic | SSE | yes | Messages API |
| POST | `/v1/messages/count_tokens` | Anthropic | no | yes | `{"input_tokens":N}` |
| POST | `/v1/chat/completions` | OpenAI | SSE | yes | Chat Completions |
| POST | `/v1/completions` | OpenAI | SSE | yes | legacy raw-prompt completions |
| GET | `/api/tags` | Ollama | no | yes | model list |
| GET | `/api/version` | Ollama | no | yes | `{"version":"0.1.0-denno"}` |
| POST | `/api/chat` | Ollama | NDJSON | yes | stream defaults to true |
| POST | `/api/generate` | Ollama | NDJSON | yes | `raw:true` skips the chat template |
| POST | `/api/show` | Ollama | no | yes | model metadata |
| GET | `/api/ps` | Ollama | no | yes | resident models |
| GET | `/v1/cache/stats` | denno ext | no | yes | prefix-cache counters |
| DELETE | `/v1/cache` | denno ext | no | yes | flush prefix cache |
| GET | `/v1/telemetry/ws` | denno ext | WebSocket | no | separate stream, not part of API compat |

Routes that are **not** implemented by `denno serve` and answer `501` through
the fallback are listed in section 6.

`denno serve-flash` (the hand-written server in
`crates/denno-cli/src/serve_flash.rs`) is a different, smaller route set:
`/health`, `/ready`, `/metrics`, `POST /v1/messages`, `GET /v1/models`, and the
`/v1/steer/*` extensions. Everything else in this document is about
`denno serve`.

## 2. Authentication

Authentication is **off by default**. It turns on only when at least one key is
configured (`--api-key`, `--api-keys`, or `DENNO_API_KEYS`, comma-separated).

When enabled, any request to an authenticated route must present one of:

- `x-api-key: <key>`
- `Authorization: Bearer <key>`

Both headers are accepted on every protocol; if both are present, a match on
either one is enough. Comparison is constant-time. There is no
protocol-specific requirement — in particular, `anthropic-version` is listed in
the CORS allow-list but is **not** required or validated.

A failed check returns `401` with the protocol's own error envelope:

```json
// Anthropic: /v1/messages, /v1/messages/count_tokens
{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}

// OpenAI: /v1/* (except /v1/messages*)
{"error":{"message":"invalid x-api-key","type":"authentication_error","param":null,"code":null}}

// Ollama: /api/*
{"error":"invalid x-api-key"}
```

`/health`, `/ready`, and `/metrics` never require authentication.

## 3. Common response headers

These are applied by the outermost middleware layers and therefore appear on
every protocol, including streaming and error responses:

| Header | Value / condition |
| --- | --- |
| `x-content-type-options` | `nosniff` on all responses |
| `x-request-id` | echoed if the client sent a sane ASCII value, otherwise generated; always present |
| `x-denno-ttft-ms` | completed non-streaming JSON responses only |
| `x-denno-decode-tps` | completed non-streaming JSON responses only |
| `cache-control` | `no-store` on SSE, every inference path, and `/metrics` |
| `x-accel-buffering` | `no` on SSE |
| `retry-after` | set on `429` and on overloaded `503` responses |

Content type behaviour:

- Successful `application/json` and `text/*` responses get `; charset=utf-8`
  appended (unless a charset is already present). So a normal JSON body is
  served as `application/json; charset=utf-8`.
- Error responses keep `application/json` with **no** charset.
- SSE is `text/event-stream` unchanged (no charset).
- Ollama streams are `application/x-ndjson`.
- `/metrics` is `text/plain; version=0.0.4; charset=utf-8`.

CORS: `Access-Control-Allow-Methods` is `GET, POST, OPTIONS`;
`Access-Control-Expose-Headers` is
`x-request-id, x-denno-ttft-ms, x-denno-decode-tps`. Allowed request headers
include `content-type, x-api-key, authorization, anthropic-version,
x-request-id`. Preflight is answered by the CORS layer before routing, so it
returns `204` even for a path that would otherwise be `404`.

## 4. Protocol reference

### 4.1 Anthropic Messages — `/v1/messages`

Request fields accepted: `model`, `messages`, `system` (string or block
array), `max_tokens`, `max_completion_tokens` (alias), `temperature`, `top_p`,
`top_k`, `stop_sequences`, `stream`, `tools`, `tool_choice`, `thinking`, plus
the denno extensions `repetition_penalty`, `frequency_penalty`,
`presence_penalty`, `no_repeat_ngram_size`.

- `max_tokens` is technically required by the Anthropic spec but is **not**
  rejected when missing: the server applies `DEFAULT_MAX_TOKENS` (1024).
  Explicit values are clamped at `MAX_TOKENS_CLAMP` (4096); `0` means
  "prefill only".
- The advertised `model` must match the served model name or one of its
  registered aliases, otherwise `404 not_found_error`.
- `thinking` accepts `"off" | "brief" | "full"` or the Anthropic map form
  `{"type":"enabled"|"disabled", ...}`. The default is brief.

Non-streaming response:

```json
{
  "id": "msg_...",
  "type": "message",
  "role": "assistant",
  "model": "<served name>",
  "content": [{"type": "text", "text": "..."}],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 0
  }
}
```

`stop_reason` is one of `end_turn`, `max_tokens`, `stop_sequence`, or
`tool_use`. When a tool call is emitted, `content` becomes
`[text block (if non-empty)] + [tool_use blocks]` and `stop_reason` is
`tool_use` with `stop_sequence: null`.

Streaming (`"stream": true`) uses SSE **with named events**. Each frame carries
an `event: <type>` line and a `data: <json>` line. The sequence is:

1. `message_start` — full message skeleton with `usage` (including
   `cache_read_input_tokens` / `cache_creation_input_tokens`), `stop_reason:
   null`.
2. `content_block_start` — `index: 0`, `content_block: {"type":"text","text":""}`.
3. `ping` — `{"type":"ping"}`.
4. `content_block_delta` — `delta: {"type":"text_delta","text":"..."}` for text;
   tool calls use `content_block_start` (`type: tool_use`) plus
   `input_json_delta` fragments.
5. `content_block_stop`.
6. `message_delta` — `delta.stop_reason`, `delta.stop_sequence`,
   `usage.output_tokens`.
7. `message_stop`.

Keep-alive frames are emitted as `event: ping` every 2 seconds during long
prefills.

### 4.2 Anthropic token counting — `/v1/messages/count_tokens`

Request body is the same shape as `/v1/messages` (minus `stream`). The server
runs the same normalization and tokenizer but performs no prefill and touches
no KV cache. Response:

```json
{"input_tokens": 123}
```

The value matches what the generation path reports as `usage.input_tokens` for
the same inputs. This endpoint is subject to the same auth and drain checks as
`/v1/messages`.

### 4.3 OpenAI Chat Completions — `/v1/chat/completions`

Request fields: `model`, `messages`, `max_tokens`, `max_completion_tokens`,
`temperature`, `top_p`, `n`, `stop`, `stream`, `stream_options`, `tools`,
`tool_choice`, `response_format`, `logprobs`, `top_logprobs`, `seed`.

OpenAI-specific limits:

- `n` must be 1 or absent; any other value is `400`.
- `response_format` supports `{"type":"text"}`, `{"type":"json_object"}`, and
  `{"type":"json_schema", ...}`. `json_object` / `json_schema` are implemented
  as a system-prompt instruction, not constrained decoding on this route.
- `stream_options.include_usage: true` adds `"usage": null` to every chunk and
  emits one final chunk with empty `choices` and a populated `usage`.
- `top_logprobs` is only meaningful with `logprobs: true`; out-of-range values
  are rejected with a parameter-tagged `400`.

Non-streaming response:

```json
{
  "id": "chatcmpl_...",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "<served name>",
  "system_fingerprint": "...",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "..."},
    "logprobs": null,
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
}
```

`finish_reason` is `stop`, `length`, or `tool_calls`. With tools active and a
call emitted, `message.tool_calls[]` is populated with
`{"id","type":"function","function":{"name","arguments"}}` where `arguments`
is a JSON **string**, and `message.content` becomes `null` when there is no
text.

Streaming (`"stream": true`) uses SSE **without named events**: frames have
only a `data:` line. The first chunk has
`delta: {"role":"assistant","content":""}`, subsequent chunks carry
`delta.content` (or tool-call deltas), the final content chunk carries
`finish_reason`, an optional usage chunk follows, and the stream ends with a
literal `data: [DONE]`. Keep-alive is a comment line (`: ping`) every 2
seconds, not a data frame.

### 4.4 OpenAI legacy Completions — `/v1/completions`

Accepts a raw `prompt` (string or single-element array) and does **not** apply
the chat template. It shares the scheduler, KV cache, and engine with the chat
route. Response `object` is `text_completion`; SSE chunks also use
`object: text_completion` and end with `[DONE]`. Multi-prompt batches and
token-id arrays are `400`; `n` other than 1 is `400`.

### 4.5 OpenAI model listing — `/v1/models`, `/v1/models/{id}`

```json
{
  "object": "list",
  "data": [{
    "id": "<served name>",
    "object": "model",
    "type": "model",
    "display_name": "<served name>",
    "created": 1700000000,
    "owned_by": "denno",
    "context_window": 0,
    "context_length": 0,
    "max_context_length": 0,
    "aliases": [],
    "capabilities": {
      "anthropic": true, "openai": true, "ollama": true,
      "tools": true, "streaming": true, "count_tokens": true,
      "telemetry": false
    }
  }],
  "has_more": false
}
```

The single model object is reused verbatim by `GET /v1/models/{id}`. Extra
informational keys (`model_id`, `model_dir`, `quant`, `slots`, `layers`) are
present but are denno extensions. `{id}` accepts the served name or a
registered alias; anything else is a `404` in the OpenAI envelope.

### 4.6 Ollama — `/api/chat`, `/api/generate`

Request fields (`/api/chat`): `model`, `messages`, `stream`, `options`,
`format`, `think`, `tools`, `keep_alive`.
`/api/generate` additionally accepts `prompt`, `system`, and `raw`.

Key differences from the other two protocols:

- **`stream` defaults to `true`** (Anthropic and OpenAI default to `false`).
- Streaming is **NDJSON** (`application/x-ndjson`), one JSON object per line,
  not SSE. There is no `[DONE]` sentinel; the last line has `"done": true`.
- Generation options use Ollama names inside `options`: `temperature`,
  `top_p`, `num_predict`, `seed`, `stop`, etc. Unknown option keys are ignored.
- `num_predict <= 0` is treated as "unspecified" and falls back to the server
  default.
- `format: "json"` or a JSON-schema object is applied as a system prompt
  instruction; other values are ignored.
- `think: false` disables thinking; truthy values leave the template default.
- History roles in `/api/chat` are `system`, `user`, `assistant`. A `tool`
  role in history is `400`.
- `/api/generate` runs the chat template unless `"raw": true`.

`/api/chat` streaming line (one per delta):

```json
{"model":"<served name>","created_at":"2026-01-01T00:00:00Z","message":{"role":"assistant","content":"..."},"done":false}
```

Final line adds timing and token counters:

```json
{"model":"<served name>","created_at":"2026-01-01T00:00:00Z","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","total_duration":0,"load_duration":0,"prompt_eval_count":0,"prompt_eval_duration":0,"eval_count":0,"eval_duration":0}
```

Durations are nanoseconds. `load_duration` and `prompt_eval_duration` are
always `0` because the engine does not expose those breakdowns;
`eval_duration` approximates the whole request. `done_reason` is `stop` or
`length`.

Non-streaming (`"stream": false`) returns the same fields as a single JSON
object: `message` (chat) or `response` (generate) plus the `done: true` block.

### 4.7 Ollama metadata — `/api/tags`, `/api/show`, `/api/ps`

`/api/tags` returns:

```json
{
  "models": [{
    "name": "<served name>",
    "model": "<served name>",
    "modified_at": "2026-01-01T00:00:00Z",
    "size": 0,
    "digest": "<64 hex chars>",
    "details": {"parent_model":"","format":"gguf","family":"denno","families":["denno"],"parameter_size":"","quantization_level":"fp32"}
  }]
}
```

`size` is `0` (the engine does not expose weight bytes). `digest` is a
deterministic display identifier derived from the model name, **not** a hash
of the weights.

`/api/show` takes `{"model": "..."}` (older clients send `{"name": "..."}`;
`model` wins when both are present). Missing name is `400`; an unknown model
is `404`. It returns `license`, `modelfile`, `parameters` (empty string),
`template` (empty string), `details`, `model_info`, `capabilities`
(`["completion"]`), and `modified_at`.

`/api/ps` always reports the single resident model with `size: 0`,
`size_vram: 0`, and an `expires_at` about 100 years in the future (denno never
unloads the model).

## 5. Feature matrix

| Feature | Anthropic `/v1/messages` | OpenAI `/v1/chat/completions` | OpenAI `/v1/completions` | Ollama `/api/chat` | Ollama `/api/generate` |
| --- | --- | --- | --- | --- | --- |
| Streaming transport | SSE with `event:` names | SSE, `data:` only | SSE, `data:` only | NDJSON | NDJSON |
| Stream default | false | false | false | **true** | **true** |
| Auth headers | `x-api-key` / `Bearer` | `x-api-key` / `Bearer` | same | same | same |
| Tool calling | native `tools` + `tool_choice` | OpenAI `tools` + `tool_choice` | not supported | OpenAI-form `tools` | not supported |
| Tool output shape | `tool_use` blocks | `tool_calls[]` (arguments string) | n/a | `message.tool_calls[]` (arguments object) | n/a |
| Structured output | via `thinking` only | `response_format` (prompt hint) | no | `format` (prompt hint) | `format` (prompt hint) |
| `n > 1` | n/a | `400` | `400` | n/a | n/a |
| logprobs | no | yes | yes | no | no |
| Usage fields | `input_tokens`, `output_tokens`, cache counters | `prompt_tokens`, `completion_tokens`, `total_tokens` | same | `prompt_eval_count`, `eval_count` | same |
| Stop reasons | `end_turn` / `max_tokens` / `stop_sequence` / `tool_use` | `stop` / `length` / `tool_calls` | `stop` / `length` | `stop` / `length` | `stop` / `length` |
| Error envelope | `{"type":"error","error":{...}}` | `{"error":{...}}` | `{"error":{...}}` | `{"error":"..."}` | `{"error":"..."}` |

## 6. Endpoints intentionally not implemented

Two OpenAI-compatible paths are recognized but not implemented. They return
`501 Not Implemented` with the **OpenAI** envelope, regardless of method:

```json
{
  "error": {
    "message": "POST /v1/embeddings is not implemented: ...",
    "type": "api_error",
    "param": null,
    "code": "not_implemented"
  }
}
```

| Path | Reason / suggested alternative |
| --- | --- |
| `/v1/embeddings` | no embedding model is loaded; use `/v1/chat/completions` or `/v1/completions` |
| `/v1/responses` | the Responses API is not implemented; use `/v1/chat/completions` (or `/v1/messages`) |

A single trailing slash is tolerated for these two paths
(`/v1/embeddings/` also returns `501`), but not for arbitrary deeper paths.

There is also no `/api/embed`, `/api/pull`, `/api/create`, or `/api/delete`.

## 7. Error envelopes and status codes

The envelope depends on which protocol owns the path. The mapping is driven by
the path prefix, not by a header:

- `/api` and `/api/*` → Ollama
- `/v1/messages` and `/v1/messages/*` → Anthropic
- everything else → OpenAI

| Status | Anthropic | OpenAI | Ollama |
| --- | --- | --- | --- |
| `400` | `{"type":"error","error":{"type":"invalid_request_error","message":"..."}}` | `{"error":{"message":"...","type":"invalid_request_error","param":...,"code":null}}` | `{"error":"..."}` |
| `401` | `authentication_error` | `authentication_error` | plain message |
| `404` (unknown model) | `not_found_error` | `invalid_request_error` | plain message |
| `405` (known route, wrong method) | **OpenAI envelope** `method_not_allowed` | OpenAI envelope | OpenAI envelope |
| `404` (unknown route) | **OpenAI envelope** `unknown_route` | OpenAI envelope | OpenAI envelope |
| `408` (body read timeout) | `api_error` | `server_error` | plain message |
| `413` (body too large) | **Anthropic envelope for all protocols** | Anthropic envelope | Anthropic envelope |
| `429` (rate limit) | `rate_limit_error` + `code`/`param` + `Retry-After` | `rate_limit_error` + `code: rate_limit_exceeded` | plain message |
| `500` | `api_error` | `server_error` | plain message |
| `503` (draining / overloaded) | `overloaded_error` (+ `Retry-After` when queue-full) | `server_error` | plain message (+ `Retry-After` when queue-full) |
| `504` (engine timeout) | `api_error` | `server_error` | plain message |

Three quirks are worth calling out because they surprise proxy authors:

1. **`404` unknown route and `405` method-not-allowed always use the OpenAI
   envelope**, even for `/api/*`. They are produced by the router fallback,
   outside any protocol handler. A client parsing `{"error":"..."}` from an
   Ollama path will not get that shape here.
2. **`413` always uses the Anthropic-shaped envelope** (`{"type":"error",
   "error":{"type":"invalid_request_error",...}}`), because it is emitted by
   the body-limit middleware before the protocol handler runs.
3. **Queue-full is `503` for Anthropic and Ollama but `429` for OpenAI chat
   completions**, because each protocol's convention for "capacity, retry
   later" differs. All three set `Retry-After`.

Rate limiting, when enabled via `DENNO_RATE_LIMIT_RPM` / `DENNO_RATE_LIMIT_BURST`,
applies only to the six inference paths (`/v1/messages`,
`/v1/messages/count_tokens`, `/v1/chat/completions`, `/v1/completions`,
`/api/chat`, `/api/generate`). Probe and metrics routes are exempt.

## 8. Client migration guide

### 8.1 From OpenAI to denno

Point the OpenAI SDK's `base_url` at `http://127.0.0.1:8080/v1` and set
`api_key` to a configured denno key (or any non-empty string when auth is
off). `POST /v1/chat/completions`, `GET /v1/models`, tools, SSE streaming, and
`stream_options.include_usage` work as expected.

Watch for:

- `n > 1` is `400`; issue separate requests instead.
- `response_format: {"type":"json_schema"}` is a prompt instruction. Validate
  the output on the client if strict schema adherence matters.
- `/v1/embeddings` and `/v1/responses` are `501`; route embeddings elsewhere.
- Legacy `/v1/completions` does not apply a chat template, so `prompt` is
  tokenized verbatim.

### 8.2 From Anthropic to denno

Point the Anthropic SDK at `http://127.0.0.1:8080` without a `/v1` suffix
change (the SDK appends `/v1/messages`). Set `x-api-key` (or
`Authorization: Bearer`). `anthropic-version` is accepted but not validated.

Watch for:

- `max_tokens` may be omitted; the server applies 1024 rather than returning
  `400`. Set it explicitly if you rely on a larger budget (values are clamped
  at 4096).
- SSE frames carry real `event:` names, so standard Anthropic event parsers
  work. Keep-alive is `event: ping`.
- `count_tokens` returns only `{"input_tokens": N}` — there is no
  `cache_creation_input_tokens` in that response.
- Tool calls surface as `tool_use` blocks with a generated `tool_use` id.
  `stop_reason` is `tool_use` and `stop_sequence` is `null`.

### 8.3 From Ollama to denno

Point an Ollama client's host at `http://127.0.0.1:8080`. `GET /api/tags`,
`GET /api/version`, `POST /api/chat`, `POST /api/generate`, `POST /api/show`,
and `GET /api/ps` are implemented. Streaming is NDJSON and defaults on, as
Ollama clients expect.

Watch for:

- `size` / `size_vram` are `0`, and `digest` is a display identifier rather
  than a weights hash. Do not use it for integrity checking.
- `keep_alive` is accepted and ignored; the model is always resident.
- Pull/create/delete/embed operations are not implemented.
- Non-streaming `done: true` responses still include timing fields; timings
  are approximate (`load_duration` and `prompt_eval_duration` are `0`).
- Error bodies for `/api/*` are `{"error":"..."}` for handler-level errors,
  but router-level `404` / `405` use the OpenAI envelope (see section 7).

## 9. Fact-check appendix

| Claim | Source |
| --- | --- |
| Route table, layer order, fallbacks | `crates/denno-server/src/lib.rs` (`router_with_request_timeouts`) |
| Protocol chosen by path prefix | `crates/denno-server/src/lib.rs` (`protocol_for_path`) |
| Anthropic error envelope / `ApiError` | `crates/denno-server/src/lib.rs` (`IntoResponse for ApiError`) |
| `DEFAULT_MAX_TOKENS = 1024`, `MAX_TOKENS_CLAMP = 4096` | `crates/denno-server/src/lib.rs` |
| Auth headers and constant-time compare | `crates/denno-server/src/lib.rs` (`check_auth`, `constant_time_eq`) |
| 404 / 405 / 501 envelopes | `crates/denno-server/src/lib.rs` (`route_not_found`, `method_not_allowed`, `unimplemented_openai_hint`) |
| 413 envelope | `crates/denno-server/src/lib.rs` (`payload_too_large_response`) |
| Header hygiene, charset, SSE headers | `crates/denno-server/src/lib.rs` (`header_hygiene`) |
| Anthropic SSE event sequence | `crates/denno-server/src/scheduler.rs`, `stream_asm.rs`, `lib.rs` (`to_event`) |
| `message_json` / usage counters | `crates/denno-server/src/lib.rs` (`message_json`) |
| OpenAI response, SSE, `[DONE]`, usage | `crates/denno-server/src/openai.rs` |
| OpenAI error envelope and type mapping | `crates/denno-server/src/openai.rs` (`OaError`) |
| Ollama NDJSON, final line, timings | `crates/denno-server/src/ollama.rs` (`NdjsonStream`, `final_line`, `done_fields`) |
| Ollama metadata handlers | `crates/denno-server/src/ollama.rs` (`tags`, `show`, `ps`) |
| Rate-limit paths and error type/code | `crates/denno-server/src/ratelimit.rs` |
| `/metrics` content type | `crates/denno-server/src/lib.rs` (`metrics_endpoint`) |
