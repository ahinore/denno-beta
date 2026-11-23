# denno-beta

[![CI](https://github.com/ahinore/denno-beta/actions/workflows/ci.yml/badge.svg)](https://github.com/ahinore/denno-beta/actions/workflows/ci.yml)

English | [日本語](README.ja.md)

An Anthropic Messages API–compatible LLM server written in Rust. Every inference kernel, CPU and CUDA, is written from scratch; no existing library such as candle is used. The main target today is **Qwen3.8-Flash-Next** (a sparse MoE with 512 experts): its 84–111 GB of weights run on a PC with 32 GB of VRAM and 64 GB of RAM.

- Experts are cached in VRAM, then RAM, then SSD, and only the ones the router picks are pulled up
- At temperature 0 the same input always gives the same output (checked in CI)
- A browser viewer, [denno-view](docs/VIEWER.en.md), shows inference as it runs
- An Intel Arc Pro B60 port is in progress on branch `arc-port`

## Current performance

Qwen3.8-Flash-Next, one request at a time over `/v1/messages`. All values are warm medians, and the output md5 is the same on every run.

| Machine | Weights | Decode (short chat) | Prefill 3,826 tokens | Prefill 14,018 tokens |
|---|---|---|---|---|
| RTX 5090 32 GB | Q3 (UD-IQ3_XXS) | **15.3 ms/tok (65 tok/s)** | **2.44 s** | **6.0–6.3 s** |
| RTX 5090 32 GB | Q4 (UD-Q4_K_XL) | 18.2–19.3 ms/tok (53 tok/s) | 9.2–9.8 s | 25.2–25.9 s |
| RTX 3080 10 GB | Q3 (UD-IQ3_XXS) | 116 ms/tok (8.6 tok/s) | 9.4 s | 27.3 s |

- **RTX 5090:** Ryzen 7 9800X3D, 64 GB RAM, measured 2026-11-17 on `8ae66f334`.
  - Decode after reading 14k tokens is 16.1–16.5 ms/tok on Q3 and 27–29 ms/tok on Q4.
  - Q4 is 111 GB and does not fit in RAM, so every prefill reads part of the experts from the SSD.
- **RTX 3080:** Core i9-13900K, 64 GB RAM, Docker on WSL2 on Windows 11, measured 2026-11-18.
  - The weights do not all fit in RAM, so timings move a lot with what is left in the page cache. The values are the best warm runs.
  - The three values are the best runs of different settings; no single setting gets all three (see "GPUs with less VRAM" below).
  - The build used is a development version newer than `main`.
- **Intel Arc Pro B60 24 GB:** branch `arc-port` decodes Q3 at 41 ms/tok as of 2026-11-18.
- The per-stage breakdown and older measurements are in [docs/DETAILS.en.md](docs/DETAILS.en.md).

## Usage

### 1. Get the weights

Use the 2026-11-13 release (revision `f5d175de`) of [`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF). No login is needed.

| Weights | Size | RAM, roughly |
|---|---|---|
| Q3: `UD-IQ3_XXS` (default) | 83.6 GB | 64 GB |
| Q4: `UD-Q4_K_XL` | 111.3 GB | runs on 64 GB; more RAM is faster |

```bash
pip install -U huggingface_hub
hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --revision f5d175de2cf7f720d45d6064019a81fcf056e986 \
  --include "UD-IQ3_XXS/*" --local-dir ~/models/qwen38-flash-next-1113
```

For Q4, use `--include "UD-Q4_K_XL/*"`. `--dir` reads every `*.gguf` in a folder, so keep one set of weights per folder.

### 2. Build and start

```bash
cargo build --release --features cuda -p denno-cli -p denno-engine

DENNO_CUDA=1 DENNO_PREFILL_BATCH=1 DENNO_PREFILL_BATCH_CHUNK=512 DENNO_STRIP_TOOLS=1 \
DENNO_START_PREFETCH=1 DENNO_GPU_WEIGHT_MB=12000 \
DENNO_EXPERT_GPU=0 DENNO_EXPERT_BATCH=1 DENNO_EXPERT_IQ_GPU=1 \
DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0 \
DENNO_GEMV_V3=1 DENNO_MIX_GPU=1 DENNO_ROUTER_GPU=1 \
DENNO_MOE_BATCH_PTX=crates/denno-engine/kernels/kernels_moe_batch.ptx \
./target/release/denno serve-flash \
  --dir ~/models/qwen38-flash-next-1113/UD-IQ3_XXS \
  --port 8098 --max-ctx 65536 --max-tokens 4096
```

- These are the settings for the RTX 5090 (32 GB of VRAM) in the table above.
- For Q4, point `--dir` at `UD-Q4_K_XL` and set `DENNO_START_PREFETCH=0`.

### 3. Call it

```bash
curl http://127.0.0.1:8098/v1/messages \
  -H "content-type: application/json" \
  -d '{"model":"qwen3.8-flash-next","max_tokens":256,
       "messages":[{"role":"user","content":"1+1=?"}]}'
```

```python
import anthropic
c = anthropic.Anthropic(base_url="http://127.0.0.1:8098", api_key="sk-test")
print(c.messages.create(model="qwen3.8-flash-next", max_tokens=256,
      messages=[{"role":"user","content":"Hello"}]).content[0].text)
```

- `"stream": true` returns SSE. `GET /v1/models` and `GET /health` are also there.
- There is no `/v1/chat/completions` yet.
- Requests are handled one at a time. A continued conversation reuses the earlier read, so the reply starts in about 0.5 s.

### GPUs with less VRAM (example: RTX 3080 10 GB)

- Replace `compute_90` in `crates/denno-engine/kernels/build_ptx.sh` with your GPU's value (`compute_86` for a 3080) and rebuild the PTX.
- Set `DENNO_GPU_WEIGHT_MB=5000`, `DENNO_EXPERT_GPU_MIN_HIT=0` and `RAYON_NUM_THREADS=8`. On a 13900K, 24 threads was slower.
- For the fastest decode, leave the experts on the CPU (no `DENNO_EXPERT_IQ_GPU`).
- For the fastest prefill, set `DENNO_EXPERT_IQ_GPU=1` and `DENNO_EXPERT_VRAM_MB=800` to `1500`.

The full list of environment variables, and `serve` for the other models (GPT-2, Qwen2.5, Qwen3.5-27B), are in [docs/DETAILS.en.md](docs/DETAILS.en.md).

## Documentation

- [Details](docs/DETAILS.en.md): model layout, per-stage breakdown, environment variables, other models
- [denno-view](docs/VIEWER.en.md): starting the viewer, the 3D and 2D views, telemetry frames
- [Architecture](docs/ARCHITECTURE.en.md), [numeric policy](docs/NUMERIC_POLICY.en.md)
- [Development history](docs/history/en/README.md), [tasks and measurement protocol](docs/TASKS.md), [API spec](docs/api.md)

## License

Code: MIT or Apache License 2.0 ([LICENSE-MIT](LICENSE-MIT) / [LICENSE-APACHE](LICENSE-APACHE)).

Weights are not in this repository and keep their own licenses. Qwen3.8-Flash-Next is under the [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE); pass the bundled `LICENSE` along with any copy. Reference implementations are credited in [docs/history/80_licenses.md](docs/history/80_licenses.md) (llama.cpp for the k-quant formats and the GDN formulas; the Qwen model sources and [TensorFold](https://github.com/ashhart/TensorFold) for the structure).
