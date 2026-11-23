# denno-view (inference visualization)

[Back to README](../README.md) | [日本語](VIEWER.ja.md)

## Starting

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

*27 s, 1256x618, 5 fps, 2.2 MB. The player above is a GitHub attachment, because GitHub's markdown renderer will not play a video that lives in the repository; the same file is checked in at [`docs/assets/denno_3d.mp4`](assets/denno_3d.mp4).*

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
