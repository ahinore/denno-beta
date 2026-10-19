# denno-beta

[![CI](https://github.com/ahinore/denno-beta/actions/workflows/ci.yml/badge.svg)](https://github.com/ahinore/denno-beta/actions/workflows/ci.yml)

English | [日本語](README.ja.md)

An Anthropic Messages API–compatible LLM server written in Rust.
The goal is to understand how LLMs work (tokenization → forward pass → sampling → decode loop) through code, so the **inference kernel is implemented entirely from scratch** (no existing LLM libraries such as candle). Structural reference: [ashhart/TensorFold](https://github.com/ashhart/TensorFold)

- **Single binary** `denno` (CLI `pull/models/info/generate/serve` + axum HTTP server + inference engine). CPU-only by default, with optional GPU offload via a `--features cuda` build
- Supported models: GPT-2 124M, Llama-family (RMSNorm+RoPE+SwiGLU+GQA) = **Qwen2.5-0.5B–7B-Instruct**, **Qwen3-0.6B** (QK-Norm, thinking mode), **Qwen3.5 (Gated DeltaNet hybrid, via the GGUF loader path)** — with chat templates
- **Exact decoding**: fixed seed + temperature 0 → bit-identical output for the same input. A discipline kept intact across optimizations and parallelization (continuous batching / SIMD), enforced by exactness tests in CI
- **CUDA offload** (`--features cuda`): both the decode and prefill chains run on hand-written CUDA kernels with weights/KV/hidden states resident in VRAM (13,252-token prefill: 395.8s → 12.2s, P17k-era measurement — not the current performance). Numerical contract: linear layers/rope/rmsnorm/swiglu are bit-identical to CPU; only attention has an approved relaxation window (see [Architecture §8](docs/ARCHITECTURE.en.md))
- Continuous batching (decode of multiple sessions combined into one forward, KV slot pool), prefix cache (largest-LCP selection, with GDN state snapshots), int8 quantization (SIMD integer kernel, half the weight memory), repetition penalty
- **Measured figures** (all on Qwen3.5-9B, `--quant int8`, GPU): cold single-shot is the official prefill number — server start → one request, prefix cache off, alternating A/B. As of s100 it is **13.18 s / 431 tok/s** (prefill) and **16.4 ms/token** (decode), with anchors `c61ad85b` / `9e17d563` / `af6eb6ab`. Source: [docs/TASKS.md](docs/TASKS.md) header and the "measurement-basis correction" note in the [Development History](docs/HISTORY.en.md). Warm second runs are only comparable within one session
- **Inference visualization (denno-trace)**: telemetry that never makes inference wait (`DENNO_TELEMETRY=1`, off by default, read-only) + a separate visualization server `denno-view`. Watch the model-driven program animation (stepped implementation code, layer panel, KV/variables) in the browser

## Usage (Qwen example)

```bash
# 1. download the model (into the HF hub cache ~/.cache/denno/hub)
cargo run --release -- pull Qwen/Qwen2.5-0.5B-Instruct

# 2. start the server (fixed seed for exact decoding)
cargo run --release -- serve Qwen/Qwen2.5-0.5B-Instruct --name denno --port 8080 --seed 42

# 3. call the API (Anthropic Messages format)
curl http://127.0.0.1:8080/v1/messages \
  -H "content-type: application/json" \
  -d '{"model":"denno","max_tokens":128,
       "messages":[{"role":"user","content":"1+1=?"}]}'
```

You can also generate directly via the CLI:

```bash
cargo run --release -- generate Qwen/Qwen2.5-0.5B-Instruct --prompt "1+1=?" --temperature 0 --seed 42
cargo run --release -- generate openai-community/gpt2 --prompt "The capital of France is" --raw --seed 42
```

It also works as-is with the official Anthropic Python SDK:

```python
import anthropic
c = anthropic.Anthropic(base_url="http://127.0.0.1:8080", api_key="sk-test")
print(c.messages.create(model="denno", max_tokens=128,
      messages=[{"role":"user","content":"Hello"}]).content[0].text)
```

## Starting the visualization (denno-view)

```bash
# 1. start the server with telemetry on (same command plus DENNO_TELEMETRY=1)
DENNO_TELEMETRY=1 cargo run --release -- serve Qwen/Qwen2.5-0.5B-Instruct --name denno --port 8080 --seed 42

# 2. in another terminal, start denno-view
#    (defaults to the telemetry WS at ws://127.0.0.1:8080, listens on :8093)
VIEW_TOKENIZER=~/.cache/denno/hub/models--Qwen--Qwen2.5-0.5B-Instruct/tokenizer.json \
  cargo run --release -p denno-view

# 3. open http://localhost:8093/ in a browser and chat in the built-in chat box —
#    the model-driven program, layer panel, and KV/variable animations play along
```

**Demo** — while the built-in chat generates, the model-driven program steps through the implementation code and the layer / KV / variable panels animate in sync:

![denno-view animation demo](docs/assets/denno_view.gif)

`denno-view` environment variables: `VIEW_UPSTREAM` (default `ws://127.0.0.1:8080/v1/telemetry/ws`), `VIEW_PORT` (default 8093), `VIEW_RECORD` (record frames to a file for replay), `VIEW_TOKENIZER` (tokenizer.json path for token text display).

## Documentation

- [Architecture (English)](docs/ARCHITECTURE.en.md) / [アーキテクチャ解説(日本語)](docs/ARCHITECTURE.ja.md) — design principles, inference flow, KV cache, compute kernels, model differences, visualization architecture, usage
| [Development History (English)](docs/HISTORY.en.md) / [開発履歴(日本語)](docs/HISTORY.ja.md) — implementation records P1–P17 (CUDA) / P19 (Qwen3.5) / the A–B, C and s-series up to s101, measurement data, lessons learned
- [API spec](docs/api.md)

## License

Dual licensed under MIT or Apache License 2.0 — see [LICENSE-MIT](LICENSE-MIT) / [LICENSE-APACHE](LICENSE-APACHE).
