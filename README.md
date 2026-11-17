# denno-beta

[![CI](https://github.com/ahinore/denno-beta/actions/workflows/ci.yml/badge.svg)](https://github.com/ahinore/denno-beta/actions/workflows/ci.yml)

English | [日本語](README.ja.md)

An Anthropic Messages API–compatible LLM server written in Rust.
The goal is to understand how LLMs work (tokenization → forward pass → sampling → decode loop) through code, so the **inference kernel is implemented entirely from scratch** — no existing LLM libraries such as candle. Structural reference: [ashhart/TensorFold](https://github.com/ashhart/TensorFold).

- **Single binary** `denno` — CLI (`pull` / `models` / `info` / `generate` / `serve` / `serve-flash`), axum HTTP server, inference engine. CPU only by default; a `--features cuda` build adds hand-written CUDA kernels.
- **Models**
  - GPT-2 124M
  - Llama family (RMSNorm + RoPE + SwiGLU + GQA): Qwen2.5-0.5B–7B-Instruct, Qwen3-0.6B (QK-Norm, thinking mode)
  - Qwen3.5-27B (Gated DeltaNet hybrid, loaded from GGUF, Q4_0)
  - **Qwen3.8-Flash-Next** (arch `qwen4exp`) — sparse MoE; its own serving path, `serve-flash`
- **Exact decoding**: fixed seed + temperature 0 → bit-identical output for the same input, kept intact across every optimization (continuous batching, SIMD, CUDA), enforced by exactness tests in CI.
- **CUDA offload**: decode and prefill both run on hand-written kernels with weights / KV / hidden states resident in VRAM. Numerical contract: linear layers, rope, rmsnorm and swiglu are bit-identical to CPU; only attention has an approved relaxation window ([Architecture §8](docs/ARCHITECTURE.en.md), [numeric policy](docs/NUMERIC_POLICY.en.md)).
- Continuous batching (multiple sessions in one forward, KV slot pool), prefix cache (largest-LCP selection, with GDN state snapshots), int8 quantisation for the 27B path, repetition / frequency / presence penalties, no-repeat n-gram.
- **Expert tiering** (Flash-Next): each of the 512 experts is cached in VRAM, then RAM, then read from the GGUF on SSD; the router's top-10 per layer pulls the ones it needs back up. Every tier has a budget, so the 84–111 GB model runs on 32 GB of VRAM and 64 GB of RAM.
- **Intel GPU port (in progress)**: a Level Zero + hand-written OpenCL C backend for the Arc Pro B60 lives on branch `arc-port` (not in `main` yet).
- **Inference telemetry + viewer**: telemetry that never makes inference wait (`DENNO_TELEMETRY`, on by default on the `serve-flash` path and off on `serve`, read-only) and a separate `denno-view` server with a 2D dashboard (`/`) and a 3D view (`/3d`) of the real layer structure.

## Qwen3.8-Flash-Next at a glance

Values read from the GGUF header, not from the model card:

| Item | Value |
|---|---|
| Layers | 48 — **12 full attention (QSA, `(il+1) % 4 == 0`) + 36 GDN**, `full_attention_interval = 4` |
| Experts | 512 per layer, **top-10** selected per token, plus a shared expert |
| Widths | hidden 2560, expert FFN 640, context 262144 |
| Attention | 24 query heads / 2 KV heads, head dim 256; indexer 4 heads x 128 |
| PLE | n-gram embedding layer at **layer index 1 only** (one injection per token): 16 heads = (ngram 3 − 1) × 8, 16 tables, about 28.8 GB kept on SSD |
| Weights (default) | unsloth `UD-IQ3_XXS` (2026-11-13 release), split GGUF, 83.6 GB — experts mix IQ3_XXS / IQ2_S / IQ4_NL / IQ4_XS per layer, attention / ssm output / shared expert mostly Q6_K, hyper connection and PLE Q8_0, the 28.8 GB PLE table (Engram) IQ4_NL |
| Weights (larger) | `UD-Q4_K_XL` from the same repo (2026-11-13 release), 111.3 GB; it does not fit in RAM, so part of the experts is read from the SSD every time |

`serve-flash` keeps the dense weights resident and treats the experts as a cache. The serving path is deliberately minimal: `/v1/messages` only (`/v1/chat/completions` is not implemented), one request at a time, with a prefix cache that reuses the longest shared prefix — continuing an 805-token conversation drops from 204.7 s to 5.7 s.

## Measured

### Qwen3.8-Flash-Next (measured 2026-11-17)

One request at a time over HTTP `/v1/messages`. The server is restarted for every measurement; the first condition runs on a cold page cache and is not used as the baseline (2 warm conditions × 2 rounds). "Cold" prefill means no conversation prefix cache. Output is temperature 0 and gives the same md5 on every run.

Machines:

- **rtx5090**: GeForce RTX 5090 32 GB (PCIe Gen5 x16), Ryzen 7 9800X3D, 64 GB RAM, NVMe SSD. Code `8ae66f334` (the parent of this README commit)
- **arc**: Intel Arc Pro B60 24 GB (PCIe Gen4 x8), Ryzen 7 8700G, 64 GB RAM, NVMe SSD. Branch `arc-port` at `a4f891d28`

| Machine / weights | Decode (short chat) | Decode (after 14k) | Prefill 3,826 tokens | Prefill 14,018 tokens | 2nd turn, time to first token |
|---|---|---|---|---|---|
| rtx5090 / UD-IQ3_XXS (default) | **15.3 ms/tok (65 tok/s)** | 16.1–16.5 ms/tok (61 tok/s) | **2.44 s (1,570 tok/s)** | **6.00–6.25 s (2,300 tok/s)** | 0.45–0.51 s |
| rtx5090 / UD-Q4_K_XL | 18.2–19.3 ms/tok (53 tok/s) | 27–29 ms/tok (35 tok/s) | 9.2–9.8 s (400 tok/s) | 25.2–25.9 s (550 tok/s) | 0.50–0.60 s |
| arc / UD-IQ3_XXS (reference only, text breaks) | 56 ms/tok (18 tok/s) | 100 ms/tok | 29.4–29.8 s (130 tok/s) | 107.8 s (130 tok/s) | — |
| arc / swift15 IQ3_S (old default) | 88–90 ms/tok (11 tok/s) | 115 ms/tok | 15.1–15.4 s (250 tok/s) | 62.6 s (220 tok/s) | — |

- On rtx5090, decode is the median over 128 output tokens. A second run of the same chat (experts already in VRAM) drops to 13 ms/tok (77 tok/s).
- UD-Q4_K_XL is 111 GB and does not fit in RAM; a 3,826-token prefill reads about 44 GB from the SSD every time (capped at 5.95 GB/s).
- The first request after a server start runs on a cold page cache: 3.8 s for 3,826 tokens and 7.8 s for 14k on UD-IQ3_XXS.
- The Intel GPU build on arc is still being ported. Each process gets 3,826 tokens → a short question → the same question again; decode is taken from the short question once its experts are in VRAM (all 3 rounds agree).
  - UD-IQ3_XXS runs fast but the text breaks: the answer to the 28-token question emits the role name `user` at token 44 and stops. The types that changed in the new model are not handled right yet.
  - swift15 IQ3_S gives correct text, but the third request of a process (the second short question) faults the GPU every time (`ZE_RESULT_ERROR_DEVICE_LOST`), so its decode comes from the first short question (experts still being uploaded).

Where the time goes, rtx5090 / UD-IQ3_XXS (warm, measured on the development branch one step before `8ae66f334`):

| | Total | Breakdown |
|---|---|---|
| Decode | 16.3 ms/tok | experts 4.73 (including 2.33 for the 8 % of experts computed on the CPU) / mix (hyper connection) 4.73 / ssm (GDN) 3.52 / full attention 1.47 / head 0.73 / shared + router 0.80 |
| Prefill 3,826 | 2.45 s | experts 1.32 s (1.09 s waiting for uploads) / ssm 0.62 / other (PLE etc.) 0.24 / mix 0.14 / full attention 0.14 |
| Prefill 14k | 6.44 s | ssm 2.61 s / experts 1.88 / other 1.05 / full attention 0.88 |

### Other models (older measurements)

| Model | Prefill | Decode |
|---|---|---|
| Qwen3.5-27B, Q4_0 | 9.24 s @ 16k ctx / 38.03 s @ 65k ctx | 17.3 ms/token |
| Qwen3.5-9B, int8 | 13.18 s / 431 tok/s (5,686 tok) | 16.4 ms/token |

Older numbers, rejected experiments and the protocol that defines "official" live in [docs/TASKS.md](docs/TASKS.md) and the [development history](docs/history/en/README.md).

## Usage (Qwen3.8-Flash-Next)

Flash-Next is a sparse MoE model served by its own path, `serve-flash`. It speaks the same `/v1/messages` API as everything else; `serve-flash` exists because the model has no `Engine` implementation behind `serve` — the 512-expert layout and the PLE tables need their own loader.

### Getting the weights

The GGUF is on the Hugging Face Hub, public — no login, no acceptance gate:

| | |
|---|---|
| Repo | [`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) (quantised from [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)) |
| Quant (default) | `UD-IQ3_XXS`, 4 shards, 83.6 GB. The tested release is revision `f5d175de` (2026-11-13) |
| License | Qwen Community License 1.0 — `LICENSE` ships with the weights; keep it with any copy you pass on |

```bash
pip install -U huggingface_hub
hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --revision f5d175de2cf7f720d45d6064019a81fcf056e986 \
  --include "UD-IQ3_XXS/*" --local-dir ~/models/qwen38-flash-next-1113
```

That leaves `~/models/qwen38-flash-next-1113/UD-IQ3_XXS/` holding:

```
Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf   10.9 MB  (metadata only)
Qwen3.8-Flash-Next-UD-IQ3_XXS-00002-of-00004.gguf   49.9 GB
Qwen3.8-Flash-Next-UD-IQ3_XXS-00003-of-00004.gguf    4.9 GB
Qwen3.8-Flash-Next-UD-IQ3_XXS-00004-of-00004.gguf   28.8 GB  (PLE table)
```

- `--dir` reads every `*.gguf` in a directory in sorted order, so keep one quant per folder. Sizes above are decimal GB.
- With more RAM you can take the larger `UD-Q4_K_XL` (111.3 GB) the same way: use `--include "UD-Q4_K_XL/*"` and add `DENNO_START_PREFETCH=0` when starting.
- The first shard of the 11-13 release (metadata only) has no trailing padding to the 32-byte alignment. denno reads it from 2026-11-16 on (commit `42500e5`).

```bash
# 1. build the CUDA path (the feature lives on denno-engine, not on the binary crate)
cargo build --release --features cuda -p denno-cli -p denno-engine

# 2. fetch the weights (command above), then point --dir at the shard directory

# 3. start the server (the settings used for the numbers above; for 32 GB of VRAM)
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

Call it with the Anthropic Messages format — `stream: true` gives you SSE:

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

| Endpoint | What it does |
|---|---|
| `POST /v1/messages` | Messages API. JSON, or SSE when `stream: true` (`message_start` / `content_block_delta` / `message_delta` / `message_stop`) |
| `GET /v1/models` | the advertised model id (`--name`, default `qwen3.8-flash-next`) |
| `GET /health` | liveness |
| `GET /v1/telemetry/ws` | telemetry frames for `denno-view` |

`/v1/chat/completions` is not implemented (404), and requests are handled one at a time: the model holds a single KV slot, so a second caller waits for the first to finish. A prefix cache is kept between requests — continuing an 805-token conversation drops from 204.7 s to 5.7 s.

Environment knobs that matter in practice:

| Variable | Effect |
|---|---|
| `DENNO_CUDA=1` | enable the GPU path (without it the same build runs on the CPU, roughly an order of magnitude slower) |
| `DENNO_PREFILL_BATCH=1`, `DENNO_PREFILL_BATCH_CHUNK=512` | batched GPU GEMM for prefill (chunk default 8; 512 measured best, above 1024 regresses) |
| `DENNO_GPU_WEIGHT_MB`, `DENNO_EXPERT_VRAM_MB` | VRAM budget for dense weights and for the expert cache; if the dense weights fill the first budget, the experts stay on the CPU and decode roughly halves in speed |
| `DENNO_EXPERT_IQ_GPU=1`, `DENNO_EXPERT_GPU_MIN_HIT=0` | keep IQ-type experts in VRAM and compute them on the GPU / send all of them to the GPU, with no hit-rate floor |
| `DENNO_EXPERT_GPU=1`, `DENNO_EXPERT_BATCH=1` | place the selected experts on the GPU (older path) / batch them |
| `DENNO_START_PREFETCH=1` | read the experts and the PLE table into the page cache before listening (startup grows by a dozen seconds, the first request gets faster) |
| `DENNO_GEMV_V3=1`, `DENNO_MIX_GPU=1`, `DENNO_ROUTER_GPU=1` | run the decode GEMV, mix and router on the GPU |
| `DENNO_STRIP_TOOLS=1` | drop the `tools` definitions from the prompt (a 40-tool request costs 6,979 prompt tokens, or 842 without them) |
| `DENNO_MAX_SYSTEM=N` | keep only the first N characters of the system prompt |
| `DENNO_TELEMETRY`, `DENNO_TELEMETRY_ACT=1`, `DENNO_TELEMETRY_ARCH=1` | telemetry (on by default here; `0` disables) / per-layer activation / structure frames |

### Other models (the dense path, `serve`)

`serve` covers GPT-2, the Llama family (Qwen2.5-0.5B–7B-Instruct, Qwen3-0.6B) and Qwen3.5-27B from GGUF, with continuous batching, prefix cache and int8 quantisation:

```bash
# 1. download into the HF hub cache (~/.cache/denno/hub)
cargo run --release -p denno-cli -- pull Qwen/Qwen2.5-0.5B-Instruct

# 2. start the server (fixed seed for exact decoding)
cargo run --release -p denno-cli -- serve Qwen/Qwen2.5-0.5B-Instruct --name denno --port 8080 --seed 42

# 3. single-shot generation, no server needed
cargo run --release -p denno-cli -- generate Qwen/Qwen2.5-0.5B-Instruct --prompt "1+1=?" --temperature 0 --seed 42
cargo run --release -p denno-cli -- generate openai-community/gpt2 --prompt "The capital of France is" --raw --seed 42
```

## Starting the visualization (denno-view)

```bash
# 1. start the server (serve-flash has telemetry on by default; DENNO_TELEMETRY=0 disables it)
DENNO_CUDA=1 ./target/release/denno serve-flash \
  --dir ~/models/qwen38-flash-next-1113/UD-IQ3_XXS --port 8098 --max-ctx 65536 --max-tokens 1024

# 2. in another terminal, start denno-view against that server
#    (VIEW_PORT defaults to 8093)
VIEW_UPSTREAM=ws://127.0.0.1:8098/v1/telemetry/ws \
  cargo run --release -p denno-view

# 3. open http://localhost:8093/ for the dashboard, or /3d for the 3D view
```

With the dense `serve` path instead, pass `DENNO_TELEMETRY=1` when starting it, and point `VIEW_TOKENIZER` at the model's `tokenizer.json` so the viewer can turn token ids into text (the GGUF path carries its own tokenizer, and without `VIEW_TOKENIZER` the viewer shows raw ids).

**Demo** — the 3D view (`/3d`) running a live conversation against Flash-Next: the prompt is tokenised in the input rail, the router pulls experts up from SSD into VRAM, the layer columns light up per neuron, and the sampler's top-8 candidates sit at the prediction head while the chosen token flies back to the chat box:

https://github.com/user-attachments/assets/3eb62bce-bfc8-44e7-adc8-bd95d60360f1

*27 s, 1256x618, 5 fps, 2.2 MB. The player above is a GitHub attachment, because GitHub's markdown renderer will not play a video that lives in the repository; the same file is checked in at [`docs/assets/denno_3d.mp4`](docs/assets/denno_3d.mp4).*

The 3D view (`/3d`) is built from the model's real topology rather than an illustration: 48 layers along Z, each with its mixer (12 QSA / 36 GDN), router, 512 experts laid out 32x16, shared expert and GR read/write, with a hybrid-block frame every 4 layers. Experts sit at the height of the tier that currently holds them (VRAM / RAM / SSD) and move between tiers as the router selects them; the KV band shows prefix-cache reuse, the prompt tokens, and the decode read position; the prediction head shows the real top-8 candidates from the sampler, and the chosen token flies back to the chat box. Keys: `f` toggles a quaternion-based free-flight mode (WASD + mouse, Shift to accelerate, Q/E up/down, Esc back to orbit).

The 2D view (`/2d`) drives interventions (`/v1/intervene`) and steering from a diagram laid out like Qwen's official architecture figure. The 48-layer column sits on the left. The 12 hybrid blocks are detailed on the right: each layer shows the 4 expanded-residual streams, GR Read / Write, the hidden states, Gated DeltaNet / Qwen Sparse Attn, and the MoE with its router, shared expert and 512 experts, plus the N-gram input on layer 2. The prediction head and MTP sit on top, and the embeddings at the bottom. Wheel to zoom, drag to pan, double-click to zoom onto a part, `0` to see everything; detail follows the zoom (one thumbnail per layer when zoomed out, single experts and their numbers when zoomed in). Click selects, Shift+drag selects a range. The older tile view is still there behind the `wall` button (or `?view=wall`). The tree on the right lists the same parts as `hybrid block → layer → residual / mixer / MoE → expert`. Its check boxes are the selection, "switch off / on checked" applies it, and checking a group selects everything below it. A selector expression above the tree picks nodes:

```
expert: count == 0 and layer >= 40              # experts in late layers that never fired
expert: count >= 0.5 * lmax(count)              # experts fired at least half as often as the busiest in their layer
expert: rate > 0.05 | top 5 by count per layer  # the 5 busiest per layer
mixer: kind == "qsa" and layer % 8 == 3; moe: layer in [5, 9, 17]
```

An expression is `[types:] condition [| top N by expr [per layer|hybrid]] [| limit N]`, joined with `;` (union). Types are `expert mixer moe shared ple stream block layer hybrid residual ffn experts *`; without them every switchable kind is searched. Conditions use the values `layer hybrid pos index kind type count rate rank act w acc topk off sel vram skip tokens n_expert n_used n_hc` with `+ - * / % ^`, chained comparisons (`0 <= layer < 8`), `and or not`, `in a..b` and `in [..]`. Functions are `abs min max sqrt log exp floor ceil round clamp`, plus the per-layer aggregates `lmax lmin lmean lsum` and the whole-model `gmax gmin gmean gsum`. `count` is the number of activations since the viewer was reset (clear residual): for an expert, how often the router put it in the top-k; for a block, how many tokens had a non-zero value for its role. The full grammar is at the top of `crates/denno-view/src/nodes2d.mjs`.

`denno-view` environment variables: `VIEW_UPSTREAM` (default `ws://127.0.0.1:8080/v1/telemetry/ws`), `VIEW_PORT` (default 8093), `VIEW_RECORD` (append frames to a file), `VIEW_REPLAY` (serve a recording at `/replay`), `VIEW_CAPTURE` (serve a captured frame stream at `/capture.bin`), `VIEW_TOKENIZER` (tokenizer.json for token text). Its `/chat` endpoint proxies to the upstream server, so the built-in chat box needs no CORS setup.

## Telemetry frames

`DNTL` magic, 1-byte version, 1-byte kind, then a varint-packed payload. Kind numbers are stable across branches; retired numbers are not reused.

| Kind | Frame | Meaning |
|---|---|---|
| 1 | `ServerInfo` | model id, quant, slots, context, layers |
| 2 / 3 | `SessionAdmit` / `SessionDone` | KV slot install / release |
| 4 | `Prefill` | one prefill forward: token count, ns, **the prompt token ids** |
| 5 | `Token` | one decoded token (id + position) |
| 6 | `Step` | one batched decode forward (width, ns) |
| 7 | `OpSummary` | accumulated ns per op |
| 8 | `KvUsage` | per-slot KV usage |
| 9 | `Gap` | frames dropped because a subscriber fell behind |
| 10 | `TopK` | sampler top-8 (probability as f16 bits), chosen token, entropy |
| 11 | `LayerRun` | ops executed inside one layer, in order |
| 12 | `LayerAct` | residual-stream activation: the abs-value top-K neuron indices per layer |
| 13 | `RoleAct` | per-role RMS (attn / gdn / router / shared expert / PLE) |
| 14 | `ExpertResidency` | which experts of a layer are in VRAM |
| 15 | `TierStats` | VRAM / RAM / SSD occupancy, 1 Hz |
| 16 | `RouterTopK` | the experts the router actually picked, with gate weights |

Everything is sent on a channel that drops rather than blocks, so a slow viewer can never slow inference down.

## Documentation

- [Architecture (English)](docs/ARCHITECTURE.en.md) / [アーキテクチャ解説(日本語)](docs/ARCHITECTURE.ja.md) — design principles, inference flow, KV cache, compute kernels, model differences, visualization architecture
- [Development history](docs/history/en/README.md) (English) / [日本語](docs/history/README.md) — eight volumes plus an index: bootstrap, API features, models, decode performance, prefill performance, GDN, output quality, measurement & verification, with measured numbers and the approaches that were rejected
- [Tasks and measurement protocol](docs/TASKS.md), [KPI definitions](docs/KPI.md)
- [API spec](docs/api.md)

## License

Code: dual licensed under MIT or Apache License 2.0 — see [LICENSE-MIT](LICENSE-MIT) / [LICENSE-APACHE](LICENSE-APACHE).

Weights are not distributed with this repository and keep their own licenses — Qwen3.8-Flash-Next is under the [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE) (use, modification and redistribution are permitted; the notice must travel with copies, and a Model-as-a-Service offering needs a separate license from Qwen). Reference implementations are attributed in [docs/history/80_licenses.md](docs/history/80_licenses.md): llama.cpp (MIT) for the k-quant block formats and the GDN formulas, and the Qwen model sources for their architecture.
