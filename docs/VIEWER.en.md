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

## Abliteration workbench (`/2d` → Abliterate)

The `abliterate` box in the right-hand control pane is the A/B contrast workbench for directional ablation. It builds on the residual-stream steering engine (`crates/denno-engine/src/model/steer.rs`) and the `/v1/steer` endpoints.

1. Paste two prompt sets (one prompt per line): Set A is the direction you want to remove, B the one you want to keep. `load sample` fills a small pair.
2. `capture A & B` records `ablit_A`, then `ablit_B`: it clears the KV cache before every prompt so the whole prompt is prefilled, runs each prompt through `/chat` with `max_tokens: 1, temperature: 0`, stops the recording after each set, and finally calls `/v1/steer/analyze`. The progress line shows `A 3/8…` and `cancel` stops between prompts (the partial set is still saved).
3. The separation chart shows `sep_score = ||mu_A - mu_B|| / (||mu_A|| + ||mu_B|| + 1e-12)` for every layer. Click or drag on the chart to pick the layer range `[from, to)`, or press `use recommended` (the contiguous half-max window around the peak, excluding layer 0 and the final layer).
4. `apply abliteration` sends `POST /v1/steer/apply` with `terms: [{ablit_A, +1}, {ablit_B, -1}]`, `mode: "project"`, the chosen layers and `alpha` (0.0 – 2.0, step 0.05, default 1.0). `reset / clear` sends `DELETE /v1/steer/apply`.
5. `compare off vs on` runs one test prompt with no steering, then with the current range applied, and shows both answers side by side. The previous steering state is restored afterwards.
6. `colour: ablit` (also the `ablit` chip in the header) paints the 48-layer column and the 12 hybrid blocks by per-layer separation.
7. `download .dns` / `download .json` fetch `GET /v1/steer/export?name=ablit_A` (binary DNSTEER1 or JSON with a base64 body). The `import` field posts the file back to `POST /v1/steer/import`: a `.dns` file is sent as `application/octet-stream`, a JSON file as a JSON direction payload.

Endpoints added for this (`crates/denno-cli/src/serve_flash.rs`):

- `POST /v1/steer/analyze` — body `{pos, neg}` or `{terms:[{snapshot,coef}...]}` (positive coefficients form the A side, negative the B side). Returns `{pos, neg, n_layer, peak_layer, recommended_range:[from,to], layers:[{layer, diff_norm, norm_pos, norm_neg, cos_sim, sep_score}]}`.
- `GET /v1/steer/export?name=<snap>[&format=json]` — the stored DNSTEER1 snapshot, or the same bytes base64-packed in JSON.
- `POST /v1/steer/import[?name=<new>]` — a DNSTEER1 body or a JSON payload (`data_base64` or a `mean` array). Shape and model checks match the loader; bad input is a 400.

`POST /v1/steer/record` also accepts `capture_mode`: `"all"` (the default, unchanged) averages every recorded token, while `"last_prompt_token"` records only the residual at the final prefill token (`nt - 1`) per layer and ignores the generated decode tokens. Any other value is a 400. `GET /v1/steer` reports the active `capture_mode`. The Abliterate panel defaults its `capture mode` selector to `last_prompt_token` (difference-in-means on the instruction boundary).

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
