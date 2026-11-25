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

# 3. open http://localhost:8093/ for the dashboard, /2d for the map, /3d for the 3D view
```

With the dense `serve` path instead, pass `DENNO_TELEMETRY=1` when starting it, and point `VIEW_TOKENIZER` at the model's `tokenizer.json` so the viewer can turn token ids into text (the GGUF path carries its own tokenizer, and without `VIEW_TOKENIZER` the viewer shows raw ids).

## Pages

| Path | What it is |
|---|---|
| `/` | dashboard: live rates, KV/VRAM/latency, op timing, input tokens, op graph with code pane, API examples |
| `/2d` | architecture map with interventions, steering, presets, and the abliteration workbench |
| `/3d` | 3D view built from the model topology: layers, experts, KV band, prediction head, chat |

All three pages connect to `GET /ws` on the viewer's own origin, which relays the upstream telemetry stream. `/2d` also proxies REST through `/api/*` to the upstream server, and every page's chat box posts to `/chat`, which relays to the upstream `/v1/messages` (so no CORS setup is needed).

**Demo** — the 3D view (`/3d`) running a live conversation against Flash-Next: the prompt is tokenised in the input rail, the router pulls experts up from SSD into VRAM, the layer columns light up per neuron, and the sampler's top-8 candidates sit at the prediction head while the chosen token flies back to the chat box:

https://github.com/user-attachments/assets/3eb62bce-bfc8-44e7-adc8-bd95d60360f1

*27 s, 1256x618, 5 fps, 2.2 MB. The player above is a GitHub attachment, because GitHub's markdown renderer will not play a video that lives in the repository; the same file is checked in at [`docs/assets/denno_3d.mp4`](assets/denno_3d.mp4).*

## Dashboard (`/`)

- Header: server info, active sessions, dropped-frame (`gap`) counter, a `speed` slider (0.5–8, replay playback speed), and replay transport: `replay` toggles live/replay, `▶` play/pause, `⏮` to start, `◂` / `▸` step, plus a position slider (enabled when `VIEW_REPLAY` is set).
- Left: KV cache slots, `hidden x (residual)`, `logits top-k`, `entropy (nat)`, `chosen token`, per-op timing for the last step, live rates for decode / prefill / VRAM over a rolling 90 s window, KV / VRAM / latency utilisation, and API examples for the OpenAI-compatible `/v1/chat/completions` and Anthropic-compatible `/v1/messages` endpoints with a copy button and a ready-made `curl`.
- Centre: input tokens for the last prompt, the op graph, and a code pane that switches between pseudo-code and the real implementation (`real code ▸` / `◂ pseudo code`).

Keyboard: `2` opens `/2d`, `3` opens `/3d`, `R` reloads the page. The shortcuts are ignored while typing in an input or a content-editable element.

## 2D view (`/2d`)

The 2D view drives interventions (`/v1/intervene`) and steering from a diagram laid out like Qwen's official architecture figure. The 48-layer column sits on the left. The 12 hybrid blocks are detailed on the right: each layer shows the 4 expanded-residual streams, GR Read / Write, the hidden states, Gated DeltaNet / Qwen Sparse Attn, and the MoE with its router, shared expert and 512 experts, plus the N-gram input on layer 2. The prediction head and MTP sit on top, and the embeddings at the bottom.

Navigation: wheel to zoom, drag to pan, double-click to zoom onto a part, `0` to see everything, `+` / `=` and `-` to zoom, `Esc` to clear the selection. Detail follows the zoom (one thumbnail per layer when zoomed out, single experts and their numbers when zoomed in). Click selects, Shift+drag selects a range. The older tile view is still there behind the `wall` button (or `?view=wall`).

Header:

- `panes`: `chat`, `tree`, `ctrl` toggle the three side panes.
- `colour`: `per token` (last token), `accumulated` (since the last clear), `ablit` (per-layer separation score from the Abliterate tab). The `ablit` chip and the panel's `colour` button are the same switch.
- `model`: selecting a model restarts the server with that model (`POST /v1/model/select`); the page is unusable until the load finishes.
- `epoch` and `cache` show the telemetry epoch and the current context cache; the banner states whether an intervention or steering is active.

The tree on the right lists the same parts as `hybrid block → layer → residual / mixer / MoE → expert`. Its check boxes are the selection, "switch off / on checked" applies it, and checking a group selects everything below it. A selector expression above the tree picks nodes:

```
expert: count == 0 and layer >= 40              # experts in late layers that never fired
expert: count >= 0.5 * lmax(count)              # experts fired at least half as often as the busiest in their layer
expert: rate > 0.05 | top 5 by count per layer  # the 5 busiest per layer
mixer: kind == "qsa" and layer % 8 == 3; moe: layer in [5, 9, 17]
```

An expression is `[types:] condition [| top N by expr [per layer|hybrid]] [| limit N]`, joined with `;` (union). Types are `expert mixer moe shared ple stream block layer hybrid residual ffn experts *`; without them every switchable kind is searched. Conditions use the values `layer hybrid pos index kind type count rate rank act w acc topk off sel vram skip tokens n_expert n_used n_hc` with `+ - * / % ^`, chained comparisons (`0 <= layer < 8`), `and or not`, `in a..b` and `in [..]`. Functions are `abs min max sqrt log exp floor ceil round clamp`, plus the per-layer aggregates `lmax lmin lmean lsum` and the whole-model `gmax gmin gmean gsum`. `count` is the number of activations since the viewer was reset (clear residual): for an expert, how often the router put it in the top-k; for a block, how many tokens had a non-zero value for its role. The full grammar is at the top of `crates/denno-view/src/nodes2d.mjs`.

Selector bar buttons: `select` replaces the selection with the match (`Ctrl+Enter` in the textarea), `+ add` adds the match (`Shift+Enter`), `− remove` subtracts it, `reveal` opens the tree at the first match.

Control panes:

- `token probs`: the sampler's top-8 distribution and entropy per token. Requires `DENNO_TELEMETRY_TOPK=1` on the server (default off). The strip holds recent tokens; hover or click pins one and shows its breakdown.
- `intervention`: counts of experts / blocks / streams currently off, `switch off selected`, `switch on selected`, `clear selection`, `clear all off`.
- `steer`: record a snapshot (`record` / `stop`), the snapshot list, `layers [from,to)`, `alpha`, `mode` (`subtract` / `project`), `terms`, `apply`, `clear apply`.
- `preset`: `export` the checked nodes, selector expression, off set and steering as JSON; import a preset file, then `apply loaded` to send its intervention and restore the selection, or `clear loaded` to drop it locally.
- `abliterate`: the A/B abliteration workbench and batch OFF/ON evaluation. See [`ABLITERATION_GUIDE.md`](ABLITERATION_GUIDE.md).
- `state`: KV main/aux, expert residency (VRAM / RAM / mmap / coverage), `clear KV cache` (`DELETE /v1/cache`, the next chat prefills from scratch) and `clear residual` (zero the per-token / accumulated display).
- `requests`: the last 14 REST calls with status.

Telemetry strip: `ttft`, `tok/s`, `vram hit`, `kv reuse`, `mtp accept`, plus grid / cell / ws / frames / tokens / last / vram / skip counters. The legend's `vram` checkbox draws a dot for experts that stay resident in VRAM.

Chat: `Enter` sends, `Shift+Enter` adds a newline, `stop` aborts generation (`POST /v1/abort`), `clear chat` clears the local log (use `clear KV cache` to drop the server context), and a `続きを生成` button continues a truncated answer.

Keyboard on `/2d`: `0` fit, `+` / `=` zoom in, `-` zoom out, `Esc` clear selection, `Ctrl+Enter` / `Shift+Enter` in the selector box.

## 3D view (`/3d`)

The 3D view is built from the model's real topology rather than an illustration: 48 layers along Z, each with its mixer (12 QSA / 36 GDN), router, 512 experts laid out 32x16, shared expert and GR read/write, with a hybrid-block frame every 4 layers. Experts sit at the height of the tier that currently holds them (VRAM / RAM / SSD) and move between tiers as the router selects them; the KV band shows prefix-cache reuse, the prompt tokens, and the decode read position; the prediction head shows the real top-8 candidates from the sampler (requires `DENNO_TELEMETRY_TOPK=1`), and the chosen token flies back to the chat box.

Panels: model stats (current layer and mixer kind, neurons lit, fps, VRAM / RAM / DISK expert counts, telemetry state, prefill / decode counters, tier stats, PLE, token and KV positions, prompt text, top-1 probability / chosen token / entropy), the legend, a 48-layer strip (click a bar or use `[` / `]`), the layer inspector (layer, mixer kind, expert summary, activation bar), and the keyboard legend.

Without telemetry the view runs a simulated demo (`mode: demo (routing simulated)`); when frames arrive it switches to live mode. `?replay=1` loads a recorded frame stream from `/capture.bin` (set `VIEW_CAPTURE`) instead of opening a websocket. The chat box uses `/chat`; `?api=http://host:8098` connects it directly to a server instead.

Keyboard on `/3d`:

| Key | Action |
|---|---|
| `1` | camera: Overview |
| `2` | camera: Front (residual lanes head-on) |
| `3` | camera: Top (layer order along Z) |
| `4` | camera: Side (expert grid from the side) |
| `5` | camera: Reset |
| `[` / `]` | layer inspector: previous / next layer |
| `p` | toggle the fps overlay |
| `f` | fly mode (WASD + mouse, Shift to accelerate, Q/E up/down, `Esc` back to orbit) |
| `space` | pause / resume the demo |
| `c` | toggle the chat panel |
| `?` | toggle this keyboard legend (`Esc` closes it) |

Mouse: drag rotates, wheel zooms.

## Environment variables

`denno-view`:

- `VIEW_UPSTREAM` — upstream telemetry websocket (default `ws://127.0.0.1:8080/v1/telemetry/ws`).
- `VIEW_PORT` — listen port (default 8093).
- `VIEW_RECORD` — append received frames to a file.
- `VIEW_REPLAY` — serve a recording at `/replay` (enables the dashboard replay transport).
- `VIEW_CAPTURE` — serve a concatenated frame stream at `/capture.bin` (`/3d?replay=1`).
- `VIEW_TOKENIZER` — `tokenizer.json` used for token text.

`serve-flash` telemetry controls:

| Variable | Default | Effect |
|---|---|---|
| `DENNO_TELEMETRY` | on | `0` disables the telemetry channel and thread entirely |
| `DENNO_TELEMETRY_ACT` | off | `1` sends `LayerAct` (top-K neuron indices per layer); `DENNO_TELEMETRY_ACT_TOPK` sets K (default 64) |
| `DENNO_TELEMETRY_TOPK` | off | `1` sends `TopK` (sampler top-8 + entropy) for `/2d` token probs and the 3D prediction head |
| `DENNO_TELEMETRY_ARCH` | off | `1` sends `RoleAct` and `RouterTopK` |
| `DENNO_TELEM_EXPERT_SKIP` | off | `1` also sends `ExpertSkip` (experts the router wanted but intervention disabled); requires `DENNO_TELEMETRY_ARCH` |

### Layer drilldown (task 857)

Clicking a layer box in the left column (or one of the layer panels on the right) opens a compact drilldown in the bottom-left corner of the map. It lists the experts the layer's router put in its top-k with their gating weights (raw value; the bar is normalised to the largest one), a VRAM hit / CPU miss badge (`?` until `ExpertResidency` arrives, `skip` for experts an intervention cut), and a sparkline of how often each expert fired over the last 16 tokens (one vertical tick per token it was selected). `×` or Esc closes it; clicking another layer switches to that layer. Clicking a layer in the old `wall` view opens the same drawer. hit / miss is only meaningful with `DENNO_EXPERT_GPU=1` and `DENNO_EXPERT_VRAM_MB>0`; when the telemetry has not arrived the drawer says "waiting for RouterTopK" / "waiting for ExpertResidency" and shows no guessed numbers.

## Abliteration workbench (`/2d` → Abliterate)

The `abliterate` box in the right-hand control pane is the A/B contrast workbench for directional ablation. It builds on the residual-stream steering engine (`crates/denno-engine/src/model/steer.rs`) and the `/v1/steer` endpoints.

1. Paste two prompt sets (one prompt per line): Set A is the direction you want to remove, B the one you want to keep. `load sample` fills a small pair.
2. Pick `capture mode`: `last_prompt_token` (default; only the final prefill token per layer, difference-in-means on the instruction boundary) or `all` (average every prefill and decode token).
3. `capture A & B` records `ablit_A`, then `ablit_B`: it clears the KV cache before every prompt so the whole prompt is prefilled, runs each prompt through `/chat` with `max_tokens: 1, temperature: 0`, stops the recording after each set, and finally calls `/v1/steer/analyze`. The progress line shows `A 3/8…` and `cancel` stops between prompts (the partial set is still saved).
4. The separation chart shows `sep_score = ||mu_A - mu_B|| / (||mu_A|| + ||mu_B|| + 1e-12)` for every layer. Click or drag on the chart to pick the layer range `[from, to)`, or press `use recommended` (the contiguous half-max window around the peak, excluding layer 0 and the final layer).
5. `apply abliteration` sends `POST /v1/steer/apply` with `terms: [{ablit_A, +1}, {ablit_B, -1}]`, `mode: "project"`, the chosen layers and `alpha` (0.0 – 2.0, step 0.05, default 1.0). `reset / clear` sends `DELETE /v1/steer/apply`.
6. `eval off vs on` is the batch comparison: paste 1–10 prompts in `eval prompts`, then it runs every prompt with no steering (OFF), applies the current `[from, to)` / `alpha` (`project`), runs them again (ON), and renders a compact table `prompt | OFF | ON | len delta`. The progress line reads `eval 2/5 (off)…` / `eval 2/5 (on)…` and `cancel` stops between prompts. The output length is `usage.output_tokens` (character count when a server does not report it), each response is marked `[refuse]` when it looks like a refusal, and changed responses are highlighted. The previous single-rule apply state is restored on completion and on cancel.
7. `colour: ablit` (also the `ablit` chip in the header) paints the 48-layer column and the 12 hybrid blocks by per-layer separation.
8. `download .dns` / `download .json` fetch `GET /v1/steer/export?name=ablit_A` (binary DNSTEER1 or JSON with a base64 body). The `import` field posts the file back to `POST /v1/steer/import`: a `.dns` file is sent as `application/octet-stream`, a JSON file as a JSON direction payload.
9. `save recipe (.json)` posts the current `[from, to)` / `alpha` (A +1 / B −1 / `project`) to `POST /v1/steer/recipe/export` and downloads `<name>.recipe.json`. Tick `embed snapshots` to make the file self-contained: both snapshots are then carried as base64 DNSTEER1 inside `snapshots`, so the recipe needs no `$DENNO_STEER_DIR` on the target machine.
10. `load recipe (.json)` validates the file locally (same rules as the server), then applies it with `POST /v1/steer/recipe/import?apply=1` and syncs the layer range, alpha slider and name to the recipe. Embedded snapshots are loaded into memory only; nothing is written to disk.
11. `stack` combines several concept vectors. `add vector` appends a row (up to 4); each row carries a pos / neg snapshot pair, a `mode` (`project` / `subtract`), its own layer range `[from, to)` and its own `alpha`. Unticking the box on the left drops the row from the request. `apply stack` sends only the enabled rows in one `POST /v1/steer/apply` as `rules: [{terms:[{pos,+1},{neg,-1}], layers, alpha, mode}]` (a new row starts with the range currently selected on the chart). The server applies the rules one by one in the written order, so two rules over the same layer each project. It is a separate button from the single-vector `apply abliteration`, and the legacy body (`terms` / `layers` / `alpha` / `mode`) is unchanged.

The full workflow, recipe schema, and verification checklist are in [`ABLITERATION_GUIDE.md`](ABLITERATION_GUIDE.md).

Endpoints added for this (`crates/denno-cli/src/serve_flash.rs`):

- `POST /v1/steer/apply` — either the legacy `{terms:[{snapshot,coef}], layers:[from,to], alpha, mode}` or the stacked `{rules:[{terms,layers,alpha,mode}]}`. `rules` cannot be combined with `terms` / `layers` / `alpha` / `mode` (400). Up to 8 rules, each with its own layer range, alpha and mode, applied in order. The `apply` object returned by `GET /v1/steer` also carries `rules` (an empty array means the legacy single rule).
- `POST /v1/steer/analyze` — body `{pos, neg}` or `{terms:[{snapshot,coef}...]}` (positive coefficients form the A side, negative the B side). Returns `{pos, neg, n_layer, peak_layer, recommended_range:[from,to], layers:[{layer, diff_norm, norm_pos, norm_neg, cos_sim, sep_score}]}`.
- `GET /v1/steer/export?name=<snap>[&format=json]` — the stored DNSTEER1 snapshot, or the same bytes base64-packed in JSON.
- `POST /v1/steer/import[?name=<new>]` — a DNSTEER1 body or a JSON payload (`data_base64` or a `mean` array). Shape and model checks match the loader; bad input is a 400.
- `POST /v1/steer/recipe/export` — body `{name?, description?, embed_snapshots?}` exports the active steering, or pass `terms` / `layers` / `alpha` / `mode` to export a configuration without applying it. Returns the versioned recipe JSON (`version: 1`).
- `POST /v1/steer/recipe/import[?apply=1]` — a recipe JSON (or `{recipe, apply}`). Validates it and, when `apply` is set, applies it. Embedded snapshots land in memory only.

A recipe is `{version: 1, name, description?, model_id?, mode, alpha, layer_from, layer_to, terms: [{snapshot, coef}], snapshots?: {name: base64-DNSTEER1}}`. Validation is strict (unknown field, non-finite alpha/coef, empty terms, `layer_from > layer_to`, malformed embedded snapshot, model id mismatch at apply time are all errors). `denno serve-flash --steer-recipe <path>` (or `DENNO_STEER_RECIPE`) loads and applies one at startup and logs a single line: `[flash] steer recipe loaded: <name> mode=<mode> alpha=<alpha> layers=<from>..<to>`. Without the flag or the env var nothing changes.

A recipe only holds a single rule, so `save recipe` while a stacked (`rules`) apply is active makes `POST /v1/steer/recipe/export` return a 400 instead of silently collapsing the stack.

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
| 17 | `ExpertSkip` | experts the router wanted but intervention disabled |

Everything is sent on a channel that drops rather than blocks, so a slow viewer can never slow inference down.
