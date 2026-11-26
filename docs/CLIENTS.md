# Client Integration Examples

Copy-pasteable clients for the three compatibility surfaces denno serves:
OpenAI, Anthropic, and Ollama. Everything here targets loopback only
(`127.0.0.1`); replace the port, not the host, unless the server was started
with `--host`.

The wire contract is [`openapi.yaml`](openapi.yaml); field-level behaviour is in
[`api.md`](api.md) and longer recipes are in [`API_COOKBOOK.md`](API_COOKBOOK.md).
This page is the short path from "server is up" to "my SDK talks to it".

## 1. Which port, which routes

| Process | Default bind | OpenAI routes | Anthropic routes | Ollama routes |
|---|---|---|---|---|
| `denno serve <repo>` | `127.0.0.1:8080` | yes | yes | yes |
| `denno serve-flash` | `127.0.0.1:8098` | no (404) | `/v1/messages` only | no (404) |

`denno serve` is the multi-API axum server (`crates/denno-server`).
`denno serve-flash` only implements `/health`, `/ready`, `/metrics`,
`/v1/messages` and `/v1/steer/*`. It does not implement
`/v1/messages/count_tokens`.

The OpenAI, Anthropic and Ollama routes on `denno serve` share the same
scheduler, chat template and KV prefix cache, so the same prompt returns the
same text through any of the three wire formats. `serve-flash` is a separate
implementation for the flash architecture; only the Anthropic wire format
overlaps.

## 2. Before the first request

Start the server and pin a model name. `--name` is what clients send as
`model`; if it is omitted the advertised name defaults to the model repo name
for `denno serve`, and to `qwen3.8-flash-next` for `denno serve-flash`.

```bash
denno serve Qwen/Qwen2.5-0.5B-Instruct --name demo --port 8080
# or, for the flash path (Anthropic wire format only):
denno serve-flash --dir /path/to/gguf --name demo --port 8098
```

Wait for readiness, then discover the model name instead of guessing it:

```bash
until curl -fsS http://127.0.0.1:8080/ready >/dev/null; do sleep 1; done

curl -sS http://127.0.0.1:8080/v1/models
curl -sS http://127.0.0.1:8080/api/tags
```

`/v1/models` returns one object. Its `id` is the canonical name, and `aliases`
lists the other names the OpenAI and Anthropic routes accept:

```json
{
  "object": "list",
  "data": [{
    "id": "demo",
    "object": "model",
    "display_name": "demo",
    "context_length": 4096,
    "aliases": ["Qwen/Qwen2.5-0.5B-Instruct"],
    "capabilities": {"openai": true, "anthropic": true, "ollama": true}
  }],
  "has_more": false
}
```

The Ollama routes are stricter: send exactly the name shown by
`GET /api/tags` (the `name` / `model` field). Aliases are not resolved there.

### Authentication

No auth is required when the server is started without a key. With
`--api-key secret` (or `--api-keys a,b` / `DENNO_API_KEYS`), every API route
accepts either header:

```bash
curl -sS http://127.0.0.1:8080/v1/models -H 'x-api-key: secret'
curl -sS http://127.0.0.1:8080/v1/models -H 'Authorization: Bearer secret'
```

`/health`, `/ready` and `/metrics` stay unauthenticated for probes. The SDKs
below send the right header for you (`Authorization: Bearer` for OpenAI,
`x-api-key` for Anthropic); use any non-empty placeholder when auth is off.

### CORS

The default policy allows any loopback origin (`http://localhost:<port>`,
`http://127.0.0.1:<port>`, `http://[::1]:<port>`). Browser clients on another
origin need `DENNO_CORS_ORIGIN` on the server; native Node and Python clients
are unaffected.

## 3. curl

### 3.1 OpenAI chat completions

```bash
curl -sS http://127.0.0.1:8080/v1/chat/completions \
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

Streaming (SSE, terminated by `data: [DONE]`):

```bash
curl -sN http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "demo",
    "messages": [{"role": "user", "content": "Count to five."}],
    "max_tokens": 48,
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

`max_tokens` defaults to 1024 when omitted and is clamped to 4096 when larger.
`/v1/completions` takes a raw `prompt` and skips the chat template.

### 3.2 Anthropic messages

```bash
curl -sS http://127.0.0.1:8080/v1/messages \
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

Streaming event names: `message_start`, `content_block_start`,
`content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`,
plus `ping` and `error`.

Token counting without prefill:

```bash
curl -sS http://127.0.0.1:8080/v1/messages/count_tokens \
  -H 'Content-Type: application/json' \
  -d '{"model": "demo", "messages": [{"role": "user", "content": "Hello"}]}'
# {"input_tokens": 7}
```

The value matches `usage.input_tokens` of the equivalent `/v1/messages` call.

### 3.3 Ollama

```bash
curl -sS http://127.0.0.1:8080/api/tags
curl -sS http://127.0.0.1:8080/api/version

curl -sS http://127.0.0.1:8080/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "demo",
    "messages": [{"role": "user", "content": "Name the capital of France."}],
    "stream": false,
    "options": {"temperature": 0, "num_predict": 64}
  }'

curl -sN http://127.0.0.1:8080/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"model": "demo", "prompt": "Count to five.", "stream": true}'
```

Ollama's `stream` defaults to **true**; set `"stream": false` for one JSON
object. Streaming bodies are NDJSON (`application/x-ndjson`), one object per
line, the last line carries `"done": true`, `done_reason`, `eval_count` and the
Ollama duration fields (nanoseconds).

## 4. Python: OpenAI SDK

```bash
pip install openai
```

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key=os.environ.get("DENNO_API_KEY", "local"),
    timeout=120.0,
)

resp = client.chat.completions.create(
    model="demo",
    messages=[
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Name the capital of France."},
    ],
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

# Legacy raw prompt (no chat template)
cmpl = client.completions.create(
    model="demo", prompt="def fib(n):\n    ", max_tokens=48, temperature=0
)
print(cmpl.choices[0].text)
```

Notes:

- `base_url` must include `/v1`; the SDK appends `/chat/completions`.
- The SDK retries 429 and 5xx by default and honours `Retry-After`, which the
  server sends on rate-limited responses.
- `model` may be the canonical name or any entry of `aliases`. An unknown name
  is a 404 with an OpenAI error envelope.

## 5. Node: OpenAI SDK

```bash
npm install openai
```

```js
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://127.0.0.1:8080/v1",
  apiKey: process.env.DENNO_API_KEY ?? "local",
  timeout: 120_000,
});

const resp = await client.chat.completions.create({
  model: "demo",
  messages: [
    { role: "system", content: "You are terse." },
    { role: "user", content: "Name the capital of France." },
  ],
  max_tokens: 64,
  temperature: 0,
});
console.log(resp.choices[0].message.content);

const stream = await client.chat.completions.create({
  model: "demo",
  messages: [{ role: "user", content: "Count to five." }],
  max_tokens: 48,
  stream: true,
});
for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content ?? "");
}
process.stdout.write("\n");
```

CommonJS works the same with `const OpenAI = require("openai")`. Set
`maxRetries: 0` if you want to observe raw 429 responses in tests.

## 6. Python: Anthropic SDK

```bash
pip install anthropic
```

```python
import os
import anthropic

client = anthropic.Anthropic(
    base_url="http://127.0.0.1:8080",   # no /v1 here
    api_key=os.environ.get("DENNO_API_KEY", "local"),
    timeout=120.0,
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

The server does not inspect `anthropic-version`; the SDK sends its default
header anyway. `denno serve-flash` (:8098) implements `/v1/messages` but not
`/v1/messages/count_tokens`; drop the `count_tokens` block there. Everything
else in this section works against both processes.

## 7. Node: Anthropic SDK

```bash
npm install @anthropic-ai/sdk
```

```js
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic({
  baseURL: "http://127.0.0.1:8080",
  apiKey: process.env.DENNO_API_KEY ?? "local",
  timeout: 120_000,
});

const msg = await client.messages.create({
  model: "demo",
  max_tokens: 64,
  temperature: 0,
  system: "You are terse.",
  messages: [{ role: "user", content: "Name the capital of France." }],
});
console.log(msg.content[0].text);

const stream = client.messages.stream({
  model: "demo",
  max_tokens: 64,
  messages: [{ role: "user", content: "Count to five." }],
});
for await (const event of stream) {
  if (event.type === "content_block_delta" && event.delta.type === "text_delta") {
    process.stdout.write(event.delta.text);
  }
}
process.stdout.write("\n");

const n = await client.messages.countTokens({
  model: "demo",
  messages: [{ role: "user", content: "Name the capital of France." }],
});
console.log(n.input_tokens);
```

## 8. Ollama CLI

The Ollama CLI talks to whatever `OLLAMA_HOST` points at, so no local Ollama
daemon is needed. Use the `denno serve` port (8080), not the flash port.

```bash
export OLLAMA_HOST=127.0.0.1:8080

ollama list                  # GET /api/tags
ollama show demo             # POST /api/show
ollama ps                    # GET /api/ps
ollama run demo "Name the capital of France."
```

`ollama run demo` (no prompt) opens the interactive REPL, which streams through
`POST /api/chat`. Generation options can be passed inline:

```bash
ollama run demo "Count to five." --verbose
```

`ollama list` reports `size: 0` and a synthetic `digest`; both are display
values, not weight metadata. `/api/show` returns the model name, context
length, layer count, KV slot count and quantization; `/api/ps` always reports
the single loaded model with `size_vram: 0` because the engine does not expose
weight bytes.

The CLI's pull path is not implemented: `ollama pull` and `ollama rm` fail, and
a name that does not match the advertised model gets
`model '<name>' not found, try pulling it first`. Use `ollama list` to get the
exact name.

## 9. Errors

| Status | Envelope type | Typical cause |
|---|---|---|
| 400 | `invalid_request_error` | malformed JSON, empty `messages`, bad `response_format` / `tool_choice`, negative `max_tokens` |
| 401 | `authentication_error` | `--api-key` is set and the header is missing or wrong |
| 404 | `not_found_error` (Anthropic) / `invalid_request_error` (OpenAI) | `model` does not match the served name or an alias |
| 404 | Ollama error with "try pulling it first" | Ollama route was given a name other than the advertised one |
| 429 | `rate_limit_error` (OpenAI envelope) | local rate limit; retry after `Retry-After` seconds |
| 503 | `overloaded_error` / `server_error` | draining, queue full, or engine unavailable |

OpenAI and Anthropic error bodies carry `error.type` and `error.message`;
Ollama errors carry a single `error` string. `400` bodies include `param` when
a specific field is at fault.

## 10. Verifying this page against the source

```bash
# routes and their implementations
grep -n '\.route(' crates/denno-server/src/lib.rs
grep -n 'fn admin_router' -A 6 crates/denno-server/src/ollama.rs

# default ports / model name flag
grep -n 'default_value_t = 8080\|待ち受けポート' crates/denno-cli/src/main.rs

# auth headers
grep -n 'x-api-key\|authorization' crates/denno-server/src/lib.rs | head

# Ollama model matching (exact name, no aliases)
grep -n 'fn check_model' -A 8 crates/denno-server/src/ollama.rs

# OpenAI alias matching
grep -n 'fn model_matches' -A 3 crates/denno-server/src/openai.rs
```

If a route or default changes, update the table in section 1 first; every
example below it depends on the port and route set.
