# denno API Cookbook

Copy-pasteable examples for the three wire protocols that `denno` speaks.
Every route, header, field, and CLI flag below was checked against
`crates/denno-server/src/{lib,openai,ollama,json_schema}.rs` and
`crates/denno-cli/src/main.rs`. If an example and the code disagree, the code is
right — please fix this file.

All examples bind to loopback (`127.0.0.1:8098`). Replace the model directory
and advertised model name with your own; nothing here needs a real credential.

## 0. Two server binaries, two route sets

| Binary | Command | Routes | `--vram-budget` |
| --- | --- | --- | --- |
| `denno serve` | full multi-protocol server (27B-class GGUF cache) | `/v1/messages`, `/v1/messages/count_tokens`, `/v1/chat/completions`, `/v1/completions`, `/api/tags`, `/api/version`, `/api/chat`, `/api/generate`, `/v1/models`, `/health`, `/ready`, `/v1/telemetry/ws` | no |
| `denno serve-flash` | Qwen3.8-Flash-Next (`qwen4exp`) minimal server | `POST /v1/messages`, `GET /v1/models`, `GET /health`, `GET /ready`, `/v1/telemetry/ws` | yes |

`serve-flash` answers `404 not_found_error` for the OpenAI and Ollama routes.
Use `denno serve` when you need those.

Build once:

```bash
cargo build --release --bin denno
# binary: ./target/release/denno (or $CARGO_TARGET_DIR/release/denno)
```

## 1. Starting the server

### 1.1 Full multi-API server

```bash
# 1) download / list / inspect the model (cache under the denno model dir)
./target/release/denno pull Qwen/Qwen3-4B-Instruct-2507
./target/release/denno models
./target/release/denno info Qwen/Qwen3-4B-Instruct-2507

# 2) serve on loopback. --name is what clients must send in "model".
./target/release/denno serve Qwen/Qwen3-4B-Instruct-2507 \
  --host 127.0.0.1 --port 8098 --name demo
```

Useful `serve` flags (all verified in `main.rs`):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--host` | `127.0.0.1` | bind address |
| `--port` | `8080` | bind port (this cookbook uses 8098 explicitly) |
| `--name` | repo id | advertised model id; must match `model` in requests |
| `--api-key` | unset | when set, require `x-api-key` or `Authorization: Bearer <key>` |
| `--max-queue` | `32` | queued requests beyond the one in flight; overflow = 503 |
| `--seed` | unset | default sampling seed |
| `--quant` | `fp32` | `fp32` or `int8` |
| `--prefix-cache-tokens` | unset | prefix-cache token cap |
| `--repetition-penalty` | unset | default repetition penalty |
| `--temperature-cap` | unset | clamp requested `temperature` |

### 1.2 Flash server with `--vram-budget auto`

`--vram-budget auto` (or a MiB number) asks the engine to derive `DENNO_*`
defaults from the card's VRAM, host RAM, and model size. Explicitly set `DENNO_*`
variables always win. The same behaviour is available via `DENNO_AUTO_CONFIG=1`.

```bash
./target/release/denno serve-flash \
  --dir /path/to/qwen3.8-flash-next-gguf \
  --host 127.0.0.1 --port 8098 \
  --name demo \
  --vram-budget auto
```

On a machine without CUDA (CPU-only host) VRAM probing fails and the plan is
skipped with a single stderr line:

```text
[auto-config] skip: VRAM を検出できない ...
```

That is non-fatal. Pass `--vram-budget 24576` (MiB) to force a plan without
probing. When the plan is applied it prints one line starting with
`[auto-config] plan vram=...`.

#### Small-card tiers (10 / 12 / 16 GB)

`DENNO_VRAM_TOTAL_MB=<MiB>` overrides the detected total VRAM without touching
the CLI, so the same plan can be exercised on a bigger card (the lab uses this to
validate the 10 / 12 / 16 GB tiers). Precedence: numeric `--vram-budget` >
`DENNO_VRAM_TOTAL_MB` > probe. `auto` / empty / `0` mean "do not override".

Tier boundaries use a 256 MiB slack (`VRAM_TIER_SLACK_MB`) so real cards whose
reported total is slightly under the nominal size land in the right tier:
RTX 4070 reports 12,282 MiB, RTX 4080 16,376, RTX 5070 Ti 16,303. Without the
slack a 12 GB card stays "below 12 GB", keeps the default P45 gate (0.5) and
thrashes experts between CPU and GPU.

On 10-16 GB cards the dense budget is raised to `DENSE_WORKING_SET_MB` (5,632
MiB), the per-token dense working set (attention / ssm / gate / gdn / head /
ple). Below it the dense LRU turns over every layer and attention falls back to
the CPU: measured decode on UD-IQ3_XXS with a 12 GB budget was 355.8 ms/tok at
4,505 MiB of dense weights and 18.0 ms/tok at 5,600. The expert budget for those
tiers is `usable - dense`, so the plan fits the card.

| nominal VRAM | `DENNO_GPU_WEIGHT_MB` | `DENNO_EXPERT_VRAM_MB` | `MIN_HIT` | pread / chunk | measured decode |
| ---: | ---: | ---: | --- | --- | --- |
| 8 GB | 2,867 | 5,160 | default 0.5 | 8 / 1024 | not measured |
| 10 GB | 5,632 | 3,584 | default 0.5 | 8 / 1024 | 28.8 ms/tok (Q4_K_XL) |
| 12 GB | 5,632 | 5,632 | 0 | 12 / 1536 | 25.4 ms/tok (Q4_K_XL), ~18.0 (IQ3_XXS) |
| 16 GB | 6,144 | 9,216 | 0 | 12 / 1536 | 20.7 ms/tok (Q4_K_XL), 14.5 (IQ3_XXS) |
| 24 GB | 9,420 | 16,957 | 0 | default | not measured |
| 32 GB | 12,000 | 22,000 | 0 | default | 13.0 ms/tok (Q4_K_XL), 9.8-11.1 (IQ3_XXS) |

Decode figures are lab measurements on an RTX 5090 with the 32 GB card
simulated down through these budgets (lab t886a/b/c, `decode` scenario). The
12 GB IQ3_XXS number is from a split identical to the plan within 32 MiB; the
10 / 12 / 16 GB rows are the same binary that computes the table above.
The `24 GB` row is the formula only, no measurement.

### 1.3 Readiness and liveness

`/health` is liveness and never requires auth. `/ready` is readiness: 200 only
when the model is loaded, the worker is alive, and the server is not draining;
503 otherwise.

```bash
curl -sS http://127.0.0.1:8098/health
# denno serve:       {"status":"ok"}
# denno serve-flash: {"ok":true,"model":"demo"}

curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8098/ready
# 200

curl -sS http://127.0.0.1:8098/ready
```

`denno serve` `/ready` body:

```json
{
  "status": "ready",
  "ready": true,
  "draining": false,
  "model_loaded": true,
  "worker_alive": true,
  "model": "demo",
  "context_length": 4096,
  "pending": 0,
  "in_flight": 0
}
```

Wait for readiness in a script:

```bash
for i in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8098/ready)
  [ "$code" = "200" ] && break
  sleep 1
done
[ "$code" = "200" ] || { echo "server never became ready"; exit 1; }
```

### 1.4 Auth

`--api-key secret` makes every API route require either header:

```bash
curl -sS http://127.0.0.1:8098/v1/models \
  -H 'x-api-key: secret'
# or
curl -sS http://127.0.0.1:8098/v1/models \
  -H 'Authorization: Bearer secret'
```

`/health` and `/ready` stay unauthenticated for probes. `/v1/telemetry/ws`
requires `DENNO_TELEMETRY=1` on the server, otherwise it returns 503.

## 2. OpenAI-compatible API

`base_url` for SDKs is `http://127.0.0.1:8098/v1`. Any non-empty API key works
when the server was started without `--api-key`.

### 2.1 `POST /v1/chat/completions` (non-streaming)

```bash
curl -sS http://127.0.0.1:8098/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "demo",
    "messages": [
      {"role": "system", "content": "You are terse."},
      {"role": "user", "content": "Name the capital of France."}
    ],
    "max_tokens": 64,
    "temperature": 0
  }'
```

Response (fields match the OpenAI schema):

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1760000000,
  "model": "demo",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Paris."},
    "logprobs": null,
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 21, "completion_tokens": 3, "total_tokens": 24}
}
```

`finish_reason` is `stop`, `length`, or `tool_calls`. `max_tokens` and
`max_completion_tokens` are accepted; the effective value defaults to 1024 and
is clamped to 4096. An unknown `model` returns 404 `invalid_request_error`.

### 2.2 Streaming (`stream: true`)

```bash
curl -sN http://127.0.0.1:8098/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "demo",
    "messages": [{"role": "user", "content": "Count to five."}],
    "max_tokens": 48,
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

Each SSE line is `data: <json>`. The first chunk carries `delta.role`, later
chunks carry `delta.content`, the final chunk carries `finish_reason`, then a
usage-only chunk when `include_usage` is true, then `data: [DONE]`.

```text
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"demo","choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}]}
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"demo","choices":[{"index":0,"delta":{"content":"One"},"logprobs":null,"finish_reason":null}]}
...
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"demo","choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"stop"}]}
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"demo","choices":[],"usage":{"prompt_tokens":9,"completion_tokens":5,"total_tokens":14}}
data: [DONE]
```

### 2.3 Tool calling

```bash
curl -sS http://127.0.0.1:8098/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "demo",
    "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
    "max_tokens": 256,
    "temperature": 0,
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"],
          "additionalProperties": false
        }
      }
    }],
    "tool_choice": "auto"
  }'
```

A tool call response has `content: null` and a `tool_calls` array whose
`function.arguments` is a JSON **string**:

```json
{
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_...",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"city\":\"Tokyo\"}"}
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

Send the result back as a `role: "tool"` message, then ask again:

```bash
curl -sS http://127.0.0.1:8098/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "demo",
    "messages": [
      {"role": "user", "content": "What is the weather in Tokyo?"},
      {"role": "assistant", "content": null, "tool_calls": [
        {"id": "call_1", "type": "function",
         "function": {"name": "get_weather", "arguments": "{\"city\":\"Tokyo\"}"}}
      ]},
      {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp_c\": 17}"}
    ],
    "max_tokens": 128
  }'
```

Notes, all verified in `openai.rs`:

- `tool_choice` accepts `"auto"`, `"none"`, `"required"` (both map to the
  server's forced mode), and
  `{"type":"function","function":{"name":"get_weather"}}`. Using `tool_choice`
  without `tools` is a 400.
- `"required"` / a named function enable constrained decoding of the tool-call
  scaffold. With `"auto"` the model emits a `<tool_call>` XML block that the
  server extracts; malformed XML is left in the text rather than invented.
- History `arguments` may be a JSON string or an object; both normalise to the
  same input. `id` / `tool_call_id` are accepted but not rendered into the
  prompt (the chat template does not reference them).

### 2.4 Structured output (`response_format`)

`{"type":"text"}` is the default. `{"type":"json_object"}` appends a
"single JSON object only" instruction. `{"type":"json_schema"}` appends the
normalised schema itself:

```bash
curl -sS http://127.0.0.1:8098/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "demo",
    "messages": [{"role": "user", "content": "Extract: Ada, 36, Berlin."}],
    "max_tokens": 128,
    "temperature": 0,
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "person",
        "strict": true,
        "schema": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "city": {"type": "string"}
          },
          "required": ["name", "age", "city"],
          "additionalProperties": false
        }
      }
    }
  }'
```

Read the answer from `choices[0].message.content` and parse it as JSON on the
client. The server validates the schema at request time:

- `json_schema.name` is required and non-empty; `schema` is required;
  `strict` must be a boolean.
- `strict: true` additionally requires every object node with `properties` to
  declare `additionalProperties: false` and to list all properties in
  `required`. `oneOf` / `anyOf` / `allOf` / `not` / `if` / `then` / `else` /
  `patternProperties` / `$defs` / `definitions` / `prefixItems` /
  `unevaluatedProperties` / `unevaluatedItems` / `dependentSchemas` are 400.
- `$ref` is always a 400 (no resolver).
- Schema size limit 16 KiB, nesting limit 16.

The schema is a **prompt instruction, not a token mask**. `denno` does not
build a JSON-Schema logit mask, so a weak model can still emit non-conforming
JSON. Validate on the client; use `strict: true` to reject malformed schemas
early, not to guarantee conforming output.

### 2.5 `POST /v1/completions` (legacy, raw prompt)

No chat template. `prompt` is a string or a one-element string array; batches
and token-id arrays are 400.

```bash
curl -sS http://127.0.0.1:8098/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "demo",
    "prompt": "def fib(n):\n    ",
    "max_tokens": 48,
    "temperature": 0,
    "stop": ["\n\n"]
  }'
```

Response:

```json
{
  "id": "cmpl-...",
  "object": "text_completion",
  "created": 1760000000,
  "model": "demo",
  "choices": [{"text": "if n < 2: return n", "index": 0, "logprobs": null, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 7, "completion_tokens": 9, "total_tokens": 16}
}
```

`stream: true` yields `object: "text_completion"` chunks plus `[DONE]`.
`echo: true` prefixes the prompt to `text`.

### 2.6 Python (`openai` SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8098/v1", api_key="local")

resp = client.chat.completions.create(
    model="demo",
    messages=[{"role": "user", "content": "Name the capital of France."}],
    max_tokens=64,
    temperature=0,
)
print(resp.choices[0].message.content)

stream = client.chat.completions.create(
    model="demo",
    messages=[{"role": "user", "content": "Count to five."}],
    max_tokens=48,
    stream=True,
    stream_options={"include_usage": True},
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()

# legacy completions
cmpl = client.completions.create(
    model="demo", prompt="def fib(n):\n    ", max_tokens=48, temperature=0
)
print(cmpl.choices[0].text)
```

## 3. Anthropic-compatible API (`denno serve`)

Base URL `http://127.0.0.1:8098`. `anthropic-version` is not inspected, but
sending it keeps SDKs happy. Auth is `x-api-key` (or Bearer) when `--api-key`
is set.

### 3.1 `POST /v1/messages` (non-streaming)

```bash
curl -sS http://127.0.0.1:8098/v1/messages \
  -H 'Content-Type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{
    "model": "demo",
    "max_tokens": 64,
    "temperature": 0,
    "system": "You are terse.",
    "messages": [{"role": "user", "content": "Name the capital of France."}]
  }'
```

```json
{
  "id": "msg_...",
  "type": "message",
  "role": "assistant",
  "model": "demo",
  "content": [{"type": "text", "text": "Paris."}],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 21,
    "output_tokens": 3,
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 21
  }
}
```

`stop_reason`: `end_turn`, `max_tokens`, `stop_sequence`, `tool_use`.
`max_tokens` defaults to 1024 and is clamped to 4096; `0` means prefill only.
`model` must equal the advertised name (404 `not_found_error` otherwise).

Supported fields: `model`, `max_tokens`, `max_completion_tokens`, `messages`
(content as string or text blocks), `system` (string or text blocks),
`temperature`, `top_p`, `top_k`, `stop_sequences`, `repetition_penalty`,
`frequency_penalty`, `presence_penalty`, `no_repeat_ngram_size`, `stream`,
`tools`, `tool_choice`, `thinking`. Unknown fields are ignored.

`thinking` defaults to `"brief"`. `"off"` disables the brief instruction;
`{"type":"enabled"}` and `{"type":"disabled"}` (Hermes-style maps) both map to
`off` in the current implementation.

Tool use (Anthropic shape: `input_schema`, response blocks of `type:
"tool_use"` with a JSON `input` object):

```bash
curl -sS http://127.0.0.1:8098/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "demo",
    "max_tokens": 256,
    "temperature": 0,
    "messages": [{"role": "user", "content": "Weather in Tokyo?"}],
    "tools": [{
      "name": "get_weather",
      "description": "Current weather for a city",
      "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
      }
    }],
    "tool_choice": {"type": "tool", "name": "get_weather"}
  }'
```

### 3.2 Streaming

```bash
curl -sN http://127.0.0.1:8098/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "demo",
    "max_tokens": 64,
    "stream": true,
    "messages": [{"role": "user", "content": "Count to five."}]
  }'
```

Event sequence (each SSE frame has `event: <type>` and `data: <json>`):

```text
event: message_start        # message skeleton + usage (input/cache counts)
event: content_block_start  # index 0, {"type":"text","text":""}
event: ping
event: content_block_delta  # repeated, {"delta":{"type":"text_delta","text":"..."}}
event: content_block_stop
event: message_delta        # {"delta":{"stop_reason":...,"stop_sequence":null},"usage":{"output_tokens":N}}
event: message_stop
```

### 3.3 `POST /v1/messages/count_tokens`

Same request shape as `/v1/messages`; `model` and `messages` are required,
`system` / `tools` / `tool_choice` / `thinking` are honoured, `max_tokens` is
ignored. It uses the same normalisation and tokenizer as generation but does
not prefill or touch the KV cache.

```bash
curl -sS http://127.0.0.1:8098/v1/messages/count_tokens \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "demo",
    "messages": [{"role": "user", "content": "Name the capital of France."}]
  }'
```

```json
{"input_tokens": 11}
```

If generation holds the engine lock, the call retries briefly and then returns
503 `overloaded_error`; retry later.

`count_tokens` is not available on `denno serve-flash` (404).

### 3.4 `cache_control`

Anthropic `cache_control` markers are **accepted and ignored**. There is no
explicit cache-breakpoint API; KV prefix reuse is automatic and reported in
`usage`:

```bash
curl -sS http://127.0.0.1:8098/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "demo",
    "max_tokens": 32,
    "system": [
      {"type": "text", "text": "Long shared system prompt ...",
       "cache_control": {"type": "ephemeral"}}
    ],
    "messages": [{"role": "user", "content": "Summarise in one line."}]
  }'
```

Read the split from the response:

- `usage.cache_read_input_tokens` — prefix tokens served from the KV cache
  (prefill skipped)
- `usage.cache_creation_input_tokens` — newly prefetched tokens
- `input_tokens = cache_read + cache_creation`

In streaming mode the same three values appear in
`message_start.message.usage`.

### 3.5 Python (`anthropic` SDK)

```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://127.0.0.1:8098",
    api_key="local",
)

msg = client.messages.create(
    model="demo",
    max_tokens=64,
    temperature=0,
    system="You are terse.",
    messages=[{"role": "user", "content": "Name the capital of France."}],
)
print(msg.content[0].text)

with client.messages.stream(
    model="demo",
    max_tokens=64,
    messages=[{"role": "user", "content": "Count to five."}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
print()

n = client.messages.count_tokens(
    model="demo",
    messages=[{"role": "user", "content": "Name the capital of France."}],
)
print(n.input_tokens)
```

`count_tokens` needs a server built with the same tokenizer path; the value
matches `usage.input_tokens` for the same normalised request.

## 4. Ollama-compatible API (`denno serve`)

All routes are NDJSON for streaming and plain JSON otherwise. The model field
may be empty; a non-empty value must match the advertised name or you get 404
`model '<name>' not found, try pulling it first`.

### 4.1 `GET /api/tags` and `GET /api/version`

```bash
curl -sS http://127.0.0.1:8098/api/tags
curl -sS http://127.0.0.1:8098/api/version
```

```json
{
  "models": [{
    "name": "demo",
    "model": "demo",
    "modified_at": "2026-01-01T00:00:00Z",
    "size": 0,
    "digest": "…64 hex…",
    "details": {
      "parent_model": "",
      "format": "gguf",
      "family": "denno",
      "families": ["denno"],
      "parameter_size": "",
      "quantization_level": "fp32"
    }
  }]
}
```

```json
{"version": "0.1.0-denno"}
```

`digest` is a deterministic display identifier derived from the model name, not
a hash of the weights.

### 4.2 `POST /api/chat`

`stream` defaults to **true** (Ollama semantics). Set `false` for one JSON
object.

```bash
curl -sS http://127.0.0.1:8098/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "demo",
    "messages": [
      {"role": "system", "content": "You are terse."},
      {"role": "user", "content": "Name the capital of France."}
    ],
    "stream": false,
    "options": {"temperature": 0, "num_predict": 64}
  }'
```

```json
{
  "model": "demo",
  "created_at": "2026-01-01T00:00:00Z",
  "message": {"role": "assistant", "content": "Paris."},
  "done": true,
  "done_reason": "stop",
  "total_duration": 123456789,
  "load_duration": 0,
  "prompt_eval_count": 21,
  "prompt_eval_duration": 0,
  "eval_count": 3,
  "eval_duration": 123456789
}
```

Streaming is one JSON object per line, `"done": false` for deltas and a final
`"done": true` line carrying the counters:

```bash
curl -sN http://127.0.0.1:8098/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "demo",
    "messages": [{"role": "user", "content": "Count to five."}],
    "options": {"num_predict": 48}
  }'
```

```text
{"model":"demo","created_at":"...","message":{"role":"assistant","content":"One"},"done":false}
...
{"model":"demo","created_at":"...","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","total_duration":...,"load_duration":0,"prompt_eval_count":9,"prompt_eval_duration":0,"eval_count":5,"eval_duration":...}
```

`tools` are not supported on `/api/chat`; a `role: "tool"` message is a 400.
`done_reason` is `stop` or `length`. Durations are nanoseconds;
`prompt_eval_duration` and `load_duration` are 0 because the engine does not
expose a prefill/load split.

### 4.3 `POST /api/generate`

Same rules; the payload is `prompt` + optional `system` / `raw`.

```bash
curl -sN http://127.0.0.1:8098/api/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "demo",
    "prompt": "Count to five.",
    "system": "You are terse.",
    "options": {"temperature": 0, "num_predict": 48, "stop": ["\n\n"]}
  }'
```

Streaming lines use `"response"` instead of `"message"`:

```text
{"model":"demo","created_at":"...","response":"One","done":false}
...
{"model":"demo","created_at":"...","response":"","done":true,"done_reason":"stop",...}
```

`"raw": true` bypasses the chat template and sends `prompt` verbatim (the same
raw path as `/v1/completions`); `system` is then ignored. `"stream": false`
returns a single object with `"response"` and the same trailing counters.

`options` fields understood: `temperature`, `top_p`, `num_predict`, `stop`.
`num_predict <= 0` is treated as "unset" and the server default (1024, clamped
to 4096) applies. Other keys (`num_ctx`, `seed`, ...) are ignored.

## 5. `denno bench`

`denno bench` drives a **running** server over SSE and reports reproducible
pp/tg numbers. It works with `denno serve` (both `--api` values) and
`denno serve-flash` (`--api anthropic` only; the OpenAI route does not exist
there). The default URL is exactly the loopback port used above.

```bash
./target/release/denno bench --url http://127.0.0.1:8098 --api anthropic
```

Key flags (verified in `main.rs`):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--url` | `http://127.0.0.1:8098` | target base URL |
| `--api` | `anthropic` | `anthropic` (`/v1/messages`) or `openai` (`/v1/chat/completions`) |
| `--pp` | `512,2048,8192` | prompt-processing token counts |
| `--tg` | `128` | generated tokens per tg case |
| `--depths` | `0,2048,8192` | context depths |
| `--rounds` | `3` | repetitions per case; the median is reported |
| `--json` | unset | write the full JSON report to a file |
| `--no-flush` | off | skip the between-round KV flush (cache-warm run) |
| `--flush-tokens` | `0` = auto | flush prompt size (auto = context - 64) |
| `--seed` | `801` | prompt/calibration seed |
| `--det-tokens` | `16` | tokens per determinism check |
| `--reuse-depth` | `2048` | context length for the turn-2 KV-reuse probe |
| `--timeout` | `600` | per-request HTTP read timeout, seconds |

Example with a JSON artefact:

```bash
./target/release/denno bench \
  --url http://127.0.0.1:8098 --api openai \
  --pp 512,2048 --depths 0,4096 --tg 64 --rounds 2 \
  --json bench-report.json
```

### 5.1 Reading the Markdown summary

```text
## denno bench — demo @ http://127.0.0.1:8098 (api=anthropic, rounds=3)

context_length: 8192

| test | depth | tokens | ttft_ms | pp_tok_s | tpot_ms | tg_tok_s |
|---|---|---|---|---|---|---|
| pp512 | 0 | 512 | 41.203 | 12426.11 | - | - |
| pp2048 | 2048 | 4096 | 512.884 | 7987.42 | - | - |
| tg128 | 0 | 9 | 12.410 | - | 18.221 | 54.88 |

Turn-2 KV reuse TTFT: 3.102 ms (cold 210.441 ms, 67.84x) at depth 2048
Determinism: md5 `…` verified across 3 rounds (2048 tokens, 16 generated)
```

- `pp<pp>` rows: `ttft_ms` is the time to first token, treated as prefill time;
  `pp_tok_s = total_tokens / ttft`. `total_tokens` is `pp + depth`.
- `tg<tg>` rows: `ttft_ms` is prefill for the calibrated prompt, `tpot_ms` is
  the per-round time-per-output-token (median across rounds), `tg_tok_s` is the
  per-round decode rate, median across rounds. `tokens` here is `prompt_tokens`.
- `Turn-2 KV reuse`: TTFT of a second turn sharing the prefix versus a cold
  first turn; `speedup = cold / reuse`.
- `Determinism`: md5 of a fixed-seed generation repeated across rounds. With a
  fixed `--seed` and unchanged inputs, it must say `verified`; `MISMATCH` means
  the run was not deterministic (concurrency, `--no-flush` interactions, or a
  different effective sampler) — re-run before trusting the numbers.
- `errors:` lists per-case failures (usually 503 queue-full, drain, or a
  timeout). A failed case prints `error` in the table instead of numbers;
  it is not silently dropped.

### 5.2 `--json` report fields

```json
{
  "url": "http://127.0.0.1:8098",
  "api": "anthropic",
  "model": "demo",
  "context_length": 8192,
  "rounds": 3,
  "created_unix": 1760000000,
  "pp": [{"pp_tokens": 512, "depth": 0, "total_tokens": 512,
          "ttft_ms": 41.2, "pp_tok_s": 12426.1,
          "ttft_runs_ms": [41.1, 41.3, 41.2], "error": null}],
  "tg": [{"depth": 0, "gen_tokens": 128, "prompt_tokens": 9,
          "ttft_ms": 12.4,
          "tpot_ms": {"n": 3, "mean": 18.3, "median": 18.221,
                       "min": 17.9, "max": 19.1, "stddev": 0.28,
                       "p90": 18.9, "p99": 19.1},
          "itl_ms": {"n": 381, "mean": 18.2, "median": 18.1,
                      "min": 16.4, "max": 41.0, "stddev": 0.9,
                      "p90": 18.9, "p99": 21.4},
          "tg_tok_s": 54.88, "runs_tok_s": [54.1, 54.9, 55.6],
          "error": null}],
  "turn2": {"depth": 2048, "cold_ttft_ms": 210.4,
            "reuse_ttft_ms": 3.1, "speedup": 67.84, "error": null},
  "determinism": {"tokens": 2048, "gen_tokens": 16, "md5": "…",
                  "verified": true, "round_md5": ["…", "…", "…"], "error": null},
  "errors": []
}
```

Use `pp_tok_s` for prefill throughput and `tg_tok_s` for decode throughput;
`tpot_ms` and `itl_ms` tell you the latency distribution, which the single
rate hides. Compare runs only at equal `--rounds`, `--pp`, `--depths`, and
`--seed`.

## 6. Common errors

| Status | `type` | Cause | Fix |
| --- | --- | --- | --- |
| 404 | `not_found_error` / `invalid_request_error` | `model` does not match the advertised name | send the `--name` value or omit `model` on Ollama routes |
| 400 | `invalid_request_error` | malformed body, empty `messages`, bad `tool_choice`, invalid `response_format` | check the `param` field in the error |
| 401 | `authentication_error` | `--api-key` set and header missing/wrong | send `x-api-key` or `Authorization: Bearer` |
| 503 | `overloaded_error` / `server_error` | queue full, draining, or `count_tokens` could not get the engine lock | retry later; raise `--max-queue` only if latency is acceptable |
| 503 | text `telemetry disabled` | `/v1/telemetry/ws` without `DENNO_TELEMETRY=1` | enable telemetry on the server |

Readiness probing should use `/ready`; `/health` stays 200 while draining and
is only a liveness signal.

## 7. Verification checklist

Re-check this document against the source before changing examples:

```bash
# routes
grep -n '\.route(' crates/denno-server/src/lib.rs

# OpenAI request fields / streaming / response shape
grep -n 'pub struct ChatCompletionRequest' -A 18 crates/denno-server/src/openai.rs
grep -n 'pub struct CompletionRequest' -A 16 crates/denno-server/src/openai.rs
grep -n '\[DONE\]' crates/denno-server/src/openai.rs

# Anthropic SSE event order
grep -n 'message_start\|message_delta\|message_stop' crates/denno-server/src/scheduler.rs

# count_tokens
grep -n 'async fn count_tokens' -A 45 crates/denno-server/src/lib.rs

# Ollama DTOs and NDJSON shape
grep -n 'pub struct ChatRequest\|pub struct GenerateRequest\|struct OllamaOptions' -A 12 crates/denno-server/src/ollama.rs
grep -n 'fn final_line' -A 30 crates/denno-server/src/ollama.rs

# JSON Schema strict rules
grep -n 'UNSUPPORTED_STRICT_KEYWORDS\|MAX_SCHEMA_BYTES' crates/denno-server/src/json_schema.rs

# CLI flags
grep -n 'vram-budget\|struct BenchOptions\|--no-flush' crates/denno-cli/src/main.rs crates/denno-cli/src/bench.rs
```

If a route disappears, delete its section rather than keeping a stale example.
