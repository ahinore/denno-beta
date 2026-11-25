# Details (model layout, environment variables, other models)

[Back to README](../README.md) | [日本語](DETAILS.ja.md)

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

## Where the time goes

Where the time goes, rtx5090 / UD-IQ3_XXS (warm, measured on the development branch one step before `8ae66f334`):

| | Total | Breakdown |
|---|---|---|
| Decode | 16.3 ms/tok | experts 4.73 (including 2.33 for the 8 % of experts computed on the CPU) / mix (hyper connection) 4.73 / ssm (GDN) 3.52 / full attention 1.47 / head 0.73 / shared + router 0.80 |
| Prefill 3,826 | 2.45 s | experts 1.32 s (1.09 s waiting for uploads) / ssm 0.62 / other (PLE etc.) 0.24 / mix 0.14 / full attention 0.14 |
| Prefill 14k | 6.44 s | ssm 2.61 s / experts 1.88 / other 1.05 / full attention 0.88 |

## serve-flash environment variables


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
| `DENNO_TELEMETRY_TOPK=1` | emit the sampler's top-8 (id, probability, entropy) as TopK frames (OFF by default; needed by the /2d token-probs panel and the 3D candidate list) |
| `DENNO_KV_SLOTS=N`, `DENNO_KV_SLOT_TOK` | number of context slots (default 2 = the legacy main + aux; 3 or more uses LCP matching + LRU) / per-slot token cap on the N-slot path (default: context length). The CLI flag is `--kv-slots` |

## Other models (the dense path, `serve`)

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

## Other models (older measurements)

| Model | Prefill | Decode |
|---|---|---|
| Qwen3.5-27B, Q4_0 | 9.24 s @ 16k ctx / 38.03 s @ 65k ctx | 17.3 ms/token |
| Qwen3.5-9B, int8 | 13.18 s / 431 tok/s (5,686 tok) | 16.4 ms/token |

Older numbers, rejected experiments and the protocol that defines "official" live in [docs/TASKS.md](TASKS.md) and the [development history](history/en/README.md).
