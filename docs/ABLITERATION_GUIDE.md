# Abliteration and steering guide

How to measure a concept direction on the residual stream, remove it, package the
result as a portable recipe, and verify that quality did not collapse. This guide
matches the current `serve-flash` + `denno-view` code; every flag and endpoint
below exists today.

Related: [`VIEWER.en.md`](VIEWER.en.md) for the viewer itself,
[`ARCHITECTURE.en.md`](ARCHITECTURE.en.md) for the model internals.

## 1. Concept

Abliteration ("directional ablation") removes a behaviour direction from the
residual stream instead of retraining weights. The engine keeps a named
**snapshot**: the mean residual vector for every layer (and every
hyper-connection stream) over a set of prompts. Two snapshots give a direction:

```
v = mean(A) - mean(B)
```

`A` is the set you want to remove (for example prompts that trigger a refusal),
`B` the set you want to keep (harmless prompts with a similar surface form).
Subtracting the difference-in-means direction from the residual stream at a
chosen layer range suppresses the behaviour while leaving most other
capabilities intact.

Two application modes exist (`mode` in the API):

| mode | update | effect |
|---|---|---|
| `project` | `h <- h - alpha * (h . v_hat) * v_hat` | removes only the component along `v`; direction of `h` elsewhere is untouched |
| `subtract` | `h <- h - alpha * v` | subtracts the raw mean difference; can shift the norm |

`project` is the safer default and the one the `/2d` workbench uses. The
operation is applied independently per hyper-connection stream, with `v_hat`
normalised per stream.

Key knobs:

- `layers: [from, to)` — half-open layer range. Early layers carry syntax and
  token identity, late layers carry output formatting; the interesting window is
  usually in the middle. Start from the recommended range (section 4).
- `alpha` — strength. `0` is a no-op, `1` is the plain difference. The workbench
  slider spans `0.0 .. 2.0` step `0.05`; the API accepts any finite value.
- `terms: [{snapshot, coef}]` — a linear combination. For A/B ablation:
  `[{ablit_A, +1}, {ablit_B, -1}]`.

Snapshots live on the server, not in the browser. `GET /v1/steer` lists them
(`snapshots: [{name, tokens, created, layers}]`) alongside `capture_mode`,
`recording` and the active `apply`.

## 2. Capture modes

`POST /v1/steer/record` starts a recording. The body is strict: only `name` and
`capture_mode` are allowed.

```json
{"name": "ablit_A", "capture_mode": "last_prompt_token"}
```

- `name`: `A-Z a-z 0-9 _ -`, at most 64 characters, must not start with `.`.
- `capture_mode`:
  - `"all"` (default) — average every recorded residual: all prefill tokens plus
    every decoded token. This is the historical behaviour.
  - `"last_prompt_token"` — record only the residual at the final prefill token
    (`nt - 1`) per layer and ignore decode tokens. This is difference-in-means on
    the instruction boundary, which is what you normally want for abliteration:
    the generated continuation would otherwise dominate the mean.
  - any other value is a `400`.

The active mode is reported as `capture_mode` by `GET /v1/steer`, and it stays
visible after `POST /v1/steer/record/stop` (the last used value is kept).

Stop with `POST /v1/steer/record/stop`. The response carries
`saved: {name, tokens, created, layers}`. Starting a recording while another is
active is an error, not an implicit stop. `DELETE /v1/steer/snapshots/<name>`
removes a stored snapshot.

`DENNO_STEER_DIR` selects where snapshots (`<name>.dns`) are stored. Set it
explicitly if you run more than one experiment; the default is process-local.

## 3. Probe extraction

The recipe for a clean probe set:

1. Write both prompt sets, one prompt per line. Keep the two sets matched in
   style and length where possible; a length or topic confound becomes part of
   the direction.
2. For each prompt, clear the context cache (`DELETE /v1/cache`), then send the
   prompt with `max_tokens: 1` and `temperature: 0`. Clearing first forces a full
   prefill, so every prompt token (and the final prefill token) is actually seen;
   a cached prefix would otherwise be skipped and silently drop out of the mean.
3. Record into `ablit_A`, stop, then record into `ablit_B`, stop.
4. Analyse with `POST /v1/steer/analyze`:

```json
{"pos": "ablit_A", "neg": "ablit_B"}
```

or, for combinations, `{"terms": [{"snapshot": "ablit_A", "coef": 1}, ...]}`
(positive coefficients form the A side, negative the B side).

Response:

```json
{
  "pos": "ablit_A",
  "neg": "ablit_B",
  "n_layer": 48,
  "peak_layer": 21,
  "recommended_range": [14, 30],
  "layers": [
    {"layer": 0, "diff_norm": 0.0, "norm_pos": 0.0, "norm_neg": 0.0,
     "cos_sim": 0.0, "sep_score": 0.0}
  ]
}
```

`sep_score` is

```
||mu_A - mu_B|| / (||mu_A|| + ||mu_B|| + 1e-12)
```

and is what the separation chart draws. The recommended range is the contiguous
half-max window around the peak `sep_score`, excluding layer 0 and the final
layer; a flat or all-zero profile falls back to `[1, n_layer - 1)`.

`/2d` performs this whole sequence with `capture A & B` (progress `A 3/8…`,
`cancel` stops between prompts and keeps what was captured). The `capture mode`
selector defaults to `last_prompt_token`.

## 4. Choosing layers and alpha

- Start with `use recommended`, then widen the range by a layer or two if the
  effect is weak.
- Sweep `alpha` in `0.25 / 0.5 / 0.75 / 1.0`. Above `1.0` the projection starts
  removing unrelated content; the slider allows up to `2.0` for experiments, not
  as a target.
- Keep `project`. Reserve `subtract` for cases where the direction is known to be
  nearly orthogonal to the residual mean.
- Prefer the smallest range and alpha that pass the OFF/ON evaluation in
  section 6. A larger ablation is not a better ablation.

## 5. Portable recipe

A recipe is a single-file steering configuration. Schema (`version: 1`):

```json
{
  "version": 1,
  "name": "no_refusal",
  "description": "optional free text",
  "model_id": "flash-next-UD-IQ3_XXS",
  "mode": "project",
  "alpha": 0.5,
  "layer_from": 14,
  "layer_to": 30,
  "terms": [
    {"snapshot": "ablit_A", "coef": 1.0},
    {"snapshot": "ablit_B", "coef": -1.0}
  ],
  "snapshots": {
    "ablit_A": "<base64 DNSTEER1>",
    "ablit_B": "<base64 DNSTEER1>"
  }
}
```

Fields:

- `version` must be `1`.
- `name` is required and non-empty; `description` and `model_id` are optional.
- `mode` is `project` or `subtract`; `alpha` and every `coef` must be finite.
- `layer_from <= layer_to <= n_layer`; terms must be non-empty (at most 16).
- `snapshots` is optional. When present, each name maps to a base64 DNSTEER1
  payload; embedded snapshots are used in memory only and never written to disk.
- Validation is strict: unknown fields, non-finite numbers, empty terms,
  `layer_from > layer_to`, malformed embedded snapshots, and a `model_id`
  mismatch at apply time are all errors.

Export:

- `POST /v1/steer/recipe/export` with `{name?, description?, embed_snapshots?}`
  exports the active configuration. Pass `terms` / `layers` / `alpha` / `mode`
  to export a configuration that is not applied. If a stacked (`rules`) apply is
  active the endpoint returns `400` instead of silently collapsing the stack.
- `/2d` `save recipe (.json)` downloads `<name>.recipe.json`; tick
  `embed snapshots` for a self-contained file.

Import:

- `POST /v1/steer/recipe/import` validates only.
- `POST /v1/steer/recipe/import?apply=1`, or a wrapped body
  `{"recipe": {...}, "apply": true}`, validates and applies.
- A bad recipe is a `400`; nothing is applied on validation failure.

Startup:

```
denno serve-flash --dir <model-dir> --steer-recipe /path/to/recipe.json
```

`DENNO_STEER_RECIPE` is the environment-variable equivalent (the flag wins).
Exactly one log line is printed:

```
[flash] steer recipe loaded: <name> mode=<mode> alpha=<alpha> layers=<from>..<to>
```

Without the flag or the variable, startup behaviour is unchanged.

`/2d` `load recipe (.json)` validates locally with the same rules, applies with
`?apply=1`, and syncs the layer range, alpha and name to the recipe. Embedded
snapshots stay in memory.

## 6. Batch OFF/ON evaluation in `/2d`

`eval off vs on` is the quality check that belongs to every recipe change.

1. Paste 1–10 prompts in `eval prompts` (one per line). Use prompts that are
   *not* the capture prompts: a direction fitted and evaluated on the same text
   will look better than it is.
2. The run clears the apply state (OFF), clears the context cache before each
   prompt, and sends each prompt with a 96-token budget.
3. It re-applies the current `[from, to)` / `alpha` in `project` mode (ON) and
   repeats the prompts, again clearing the cache before each one.
4. The table shows `prompt | OFF | ON | len delta`. Output length is
   `usage.output_tokens`, or the character count when the server does not report
   usage. Responses that look like refusals are flagged `[refuse]`, and changed
   rows are highlighted.
5. `cancel` stops between prompts. The previous single-rule apply is restored on
   completion and on cancel. A stacked (`rules`) apply is not restored by this
   panel — re-apply it with `apply stack` if needed.

Read the table for:

- refusals disappearing in the ON column (the intended effect),
- length collapse or repeated boilerplate (over-ablation),
- prompts that were already answered in OFF and got worse in ON (collateral
  damage).

## 7. Recommended verification workflow

1. Baseline: run the OFF set and record it (the eval table keeps OFF for you).
2. Capture A/B with `last_prompt_token`, clear the cache before every prompt.
3. Analyse, take the recommended range, then apply with `alpha` 0.5 as a first
   candidate.
4. Run OFF/ON on held-out prompts. Accept only if refusals drop without length
   collapse or topic drift.
5. Sweep `alpha` and the layer range one variable at a time; keep the smallest
   setting that passes.
6. Re-run a broader capability check outside the viewer (your normal benchmark
   or a fixed evaluation prompt set) before shipping. The viewer table is a
   smoke test, not a benchmark.
7. Save the recipe with `embed snapshots` only when it must travel to a machine
   without the snapshot directory; embedded weights make the file large.
8. Set `model_id` in shipped recipes so a mismatched model is refused instead of
   silently applying a direction measured on another checkpoint.
9. Version recipes with the model, capture mode, prompt sets and alpha in the
   name or description. Keep the capture prompt sets with the experiment, not in
   the repository if they contain private text.

## 8. API quick reference

| Method and path | Purpose |
|---|---|
| `GET /v1/steer` | status: `capture_mode`, `recording`, `snapshots`, `apply` |
| `POST /v1/steer/record` | start recording `{name, capture_mode?}` |
| `POST /v1/steer/record/stop` | stop and save the snapshot |
| `DELETE /v1/steer/snapshots/<name>` | delete a snapshot |
| `POST /v1/steer/apply` | legacy `{terms, layers, alpha, mode}` or stacked `{rules: [...]}` |
| `DELETE /v1/steer/apply` | clear steering |
| `POST /v1/steer/analyze` | `{pos, neg}` or `{terms}` → per-layer separation |
| `GET /v1/steer/export?name=<snap>[&format=json]` | download DNSTEER1 or JSON |
| `POST /v1/steer/import[?name=<new>]` | upload DNSTEER1 or JSON direction |
| `POST /v1/steer/recipe/export` | produce a recipe JSON |
| `POST /v1/steer/recipe/import[?apply=1]` | validate (and apply) a recipe |

`POST /v1/steer/apply` details:

- legacy body: `{"terms": [{"snapshot": "ablit_A", "coef": 1.0}],
  "layers": [14, 30], "alpha": 0.5, "mode": "project"}`
- stacked body: `{"rules": [{"terms": [...], "layers": [a, b], "alpha": 0.5,
  "mode": "project"}, ...]}` — at most 8 rules, each with its own range, alpha
  and mode, applied in the written order. `rules` cannot be combined with
  `terms` / `layers` / `alpha` / `mode` (`400`). Rules over the same layer each
  project, so effects accumulate.
- `GET /v1/steer` reports `apply.rules` (an empty array means the legacy single
  rule).

Common `400` causes: missing snapshot name, snapshot shape not matching the
model, `from > to`, `to > n_layer`, non-finite `alpha` / `coef`, empty `terms`,
unknown JSON fields, and a recipe exported while a stack is active.
