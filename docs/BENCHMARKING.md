# Benchmarking and Regression Gating

How to measure a running `denno` server reproducibly, store baselines, and fail CI
when a change regresses throughput or breaks determinism.

`denno bench` is a client: it drives an already-running server over SSE and reads
`usage` counters from the responses. It never loads weights itself, so the numbers
belong to the server process, not to the CLI invocation.

- Works with `denno serve` (`--api anthropic` and `--api openai`).
- Works with `denno serve-flash` with `--api anthropic` only; that server has no
  OpenAI route.
- Every request in the benchmark uses `temperature = 0` (greedy), so the numbers
  are not sampled from a distribution.
- `denno bench` itself needs no GPU. On a CPU-only host it measures whatever
  backend the server was started with.

Related documents: `docs/API_COOKBOOK.md` (§5 has an annotated end-to-end example
and the full Markdown/JSON walkthrough), `docs/PERFORMANCE_TUNING.md` (§4 covers
the determinism knobs that interact with these numbers).

## 1. Build and start a target

```bash
cargo build --release -p denno-cli
./target/release/denno serve <model> --port 8098
```

`serve` takes the model as a positional argument; `--port` defaults to 8080.
The default target of `denno bench` is `http://127.0.0.1:8098` (the
`serve-flash` default), so when benchmarking `serve` on its default port either
start it with `--port 8098` or pass `--url http://127.0.0.1:8080` explicitly.

Wait for readiness before measuring. Use `/ready`, not `/health`: `/health` stays
200 while the server is draining and is only a liveness signal.

## 2. Running a benchmark

```bash
./target/release/denno bench \
  --url http://127.0.0.1:8098 --api anthropic \
  --pp 512,2048,8192 --tg 128 --depths 0,2048,8192 \
  --rounds 3 --seed 801 \
  --json bench.json
```

Everything below is verified against `crates/denno-cli/src/main.rs` (flag
definitions) and `crates/denno-cli/src/bench.rs` (execution and output).

| Flag | Default | Meaning |
| --- | --- | --- |
| `--url` | `http://127.0.0.1:8098` | base URL of the running server |
| `--api` | `anthropic` | `anthropic` (`/v1/messages`) or `openai` (`/v1/chat/completions`) |
| `--pp` | `512,2048,8192` | prompt-processing token counts (comma separated) |
| `--tg` | `128` | generated tokens per tg case |
| `--depths` | `0,2048,8192` | context depths in tokens (comma separated) |
| `--rounds` | `3` | repetitions per case; the median is reported |
| `--json` | unset | write the full JSON report to this path |
| `--markdown` | on | print the Markdown summary (it is on by default; pass it explicitly when combining with `--json`) |
| `--no-flush` | off | skip the between-round KV/prefix-cache flush |
| `--flush-tokens` | `0` = auto | size of the flush prompt (auto = context length - 64) |
| `--seed` | `801` | base seed for prompt synthesis and calibration |
| `--det-tokens` | `16` | generated tokens in the determinism check |
| `--reuse-depth` | `2048` | context length for the turn-2 KV-reuse probe |
| `--timeout` | `600` | per-request HTTP read timeout, seconds |
| `--compare <BASE> <NEW>` | unset | compare two JSON reports instead of running a benchmark |
| `--fail-on-regress-pct <N>` | unset | with `--compare`, treat a drop worse than N % as a regression and exit non-zero |

Notes on flag behaviour that are easy to get wrong:

- `--markdown` cannot be switched off. It is a boolean flag whose default is
  already `true`; there is no `--no-markdown`. Redirect stdout if you only want
  the JSON file.
- `--json` writes pretty-printed JSON and prints `wrote <path>` on stderr. It does
  not replace the Markdown output.
- `--seed` does **not** get sent to the server, and no `seed` field is added to
  the request body. It only seeds the synthetic prompt generator and the token
  calibration, so that two runs send byte-identical prompts.
- `--no-flush` makes the run cache-warm: later rounds can reuse the previous
  round's prefix. That is a different measurement, not a different noise level.
  Never mix `--no-flush` into one side of a comparison only.

### 2.1 What is measured

The case matrix is `--pp` x `--depths` for prefill and `--depths` for decode.
Prompts are synthetic but deterministic, and each is calibrated: the client
binary-searches prompt length until the server's `usage.input_tokens` matches the
target, because the server's tokenizer is the only authority.

- **pp**: calibrated prompt of `pp + depth` tokens, one generated token. The time
  to first token is treated as prefill time, and
  `pp_tok_s = total_tokens / ttft`. The reported `ttft_ms` is the median over
  rounds and `pp_tok_s` is derived from that median.
- **tg**: calibrated prompt of `depth` tokens (at least 1), then `--tg` generated
  tokens. `ttft_ms` is prefill for that prompt. `tg_tok_s` is the median over
  rounds of `(output_tokens - 1) / decode_span`, and `tpot_ms` is the median over
  rounds of `decode_span / (output_tokens - 1)`. `itl_ms` pools every inter-token
  interval across rounds and reports mean/median/min/max/stddev/p90/p99.
- **turn-2 KV reuse**: at `--reuse-depth`, a cold first turn, then a second turn
  that shares the prefix and asks for one more sentence.
  `speedup = cold_ttft_ms / reuse_ttft_ms`.
- **determinism**: a calibrated prompt of `max(256, --det-tokens)` tokens,
  generated `--det-tokens` tokens, repeated `--rounds` times. Each round's output
  text is hashed with MD5. `verified` is true only when every round produced the
  same hash and all rounds completed.

Between rounds the client pushes the prefix cache out with a throwaway request
sized at the context length minus 64 tokens (or the largest `--pp` plus 1024 when
the server does not advertise a context length). Flush failures are recorded in
`errors` and do not abort the run.

## 3. The JSON report

`--json` writes a single object. Fields not listed here do not exist.

```json
{
  "url": "http://127.0.0.1:8098",
  "api": "anthropic",
  "model": "demo",
  "context_length": 8192,
  "rounds": 3,
  "created_unix": 1760000000,
  "pp": [
    {"pp_tokens": 512, "depth": 0, "total_tokens": 512,
     "ttft_ms": 41.2, "pp_tok_s": 12426.1,
     "ttft_runs_ms": [41.1, 41.3, 41.2], "error": null}
  ],
  "tg": [
    {"depth": 0, "gen_tokens": 128, "prompt_tokens": 9, "ttft_ms": 12.4,
     "tpot_ms": {"n": 3, "mean": 18.3, "median": 18.221, "min": 17.9,
                 "max": 19.1, "stddev": 0.28, "p90": 18.9, "p99": 19.1},
     "itl_ms": {"n": 381, "mean": 18.2, "median": 18.1, "min": 16.4,
                "max": 41.0, "stddev": 0.9, "p90": 18.9, "p99": 21.4},
     "tg_tok_s": 54.88, "runs_tok_s": [54.1, 54.9, 55.6], "error": null}
  ],
  "turn2": {"depth": 2048, "cold_ttft_ms": 210.4, "reuse_ttft_ms": 3.1,
            "speedup": 67.84, "error": null},
  "determinism": {"tokens": 2048, "gen_tokens": 16, "md5": "<hex>",
                  "verified": true, "round_md5": ["<hex>", "<hex>", "<hex>"],
                  "error": null},
  "errors": []
}
```

- `context_length` is `null` when the server did not report one.
- A failed case keeps its row and sets `error` to a string; it is not dropped.
  `pp_tok_s` / `tg_tok_s` / `ttft_ms` remain 0 in that row, so consumers should
  check `error` before trusting a rate.
- `errors` collects per-case failures (calibration, flush, stream errors) plus a
  one-line reason for each failed case.
- `determinism.md5` is the first round's hash; `round_md5` holds every round.
  A `MISMATCH` is visible either as `verified: false` with differing hashes or as
  a short `round_md5` when a round errored.
- `created_unix` is wall-clock seconds. It is metadata, not part of the case key,
  and `--compare` deliberately ignores it.

## 4. Comparing against a baseline

`--compare` does not run a benchmark, so it needs no server and no model. It reads
two JSON reports written by an earlier `--json` run.

```bash
./target/release/denno bench \
  --compare baseline.json candidate.json \
  --fail-on-regress-pct 5
```

Semantics, as implemented in `compare_reports` / `run_compare`:

- Exactly two paths are required. `--compare` is declared with two positional
  values, so a single path is a usage error before any file is read.
- Rows are matched by case identity, not by list position: pp by
  `(pp_tokens, depth)`, tg by `(gen_tokens, depth)`, turn-2 as a single row
  (depth is taken from whichever report has it), determinism by `md5`.
- Only three metrics are judged, and for all of them **higher is better**:
  `pp_tok_s`, `tg_tok_s`, and turn-2 `speedup`.
- `delta = (new - base) / base * 100`. If `base` is 0 or either value is not
  finite, the row gets no verdict (`-`) and is never counted as a regression.
- With a threshold `N`, `delta <= -N` is `REGRESS`, `delta >= +N` is `improve`,
  and everything else is `ok`. `N = 0` is degenerate: a 0.00 % delta satisfies
  `delta <= -0` and is reported as `REGRESS`. Use a positive threshold.
- A negative `N` is normalised to its absolute value, but clap parses `-5` as a
  flag, so pass it as `--fail-on-regress-pct=-5`.
- Without `--fail-on-regress-pct`, deltas and a verdict of `ok` are still printed
  but nothing is judged; the run only fails on a determinism mismatch.
- Cases present in only one report are `missing`. They are counted and printed
  but do **not** fail the run — this keeps a shrunken matrix visible without
  breaking CI on an intentional scope change.
- A determinism MD5 mismatch fails the run regardless of the threshold, even when
  every throughput row is flat and `regressions: 0`.
- Exit status is 0 on success. Any regression, determinism mismatch, unreadable
  file, or parse error exits non-zero (via `anyhow`, status 1) with an
  `Error: bench compare: ...` line on stderr. The Markdown table is printed before
  the failure, so the CI log still contains the diagnosis.
- Output is the Markdown table on stdout, followed by a short ANSI summary that
  lists only `REGRESS` rows and the determinism verdict. Setting `NO_COLOR=1`
  strips the escape sequences; unset it to keep colour on a terminal.
- If `model` differs between the two reports, a warning line is added to the
  Markdown. It is a warning, not a failure; gate on it in CI if your workflow
  requires a single model per baseline.
- `rounds` from both reports are printed in the header. The comparison does not
  enforce equal rounds — check the header, because 1-round and 3-round medians are
  not the same measurement.

Example table for a 20 % prefill drop at a 5 % threshold:

```text
### denno bench compare

- base: `demo` http://127.0.0.1:8098 (rounds=3)
- new: `demo` http://127.0.0.1:8098 (rounds=3)

| test | metric | base | new | delta | status |
|---|---|---|---|---|---|
| pp512 @ d0 | pp_tok_s | 1000.000 | 800.000 | -20.00% | REGRESS |

| check | base | new | delta | status |
|---|---|---|---|---|
| turn-2 KV reuse @ d2048 | speedup | 2.000 | 1.980 | -1.00% | ok |
| determinism md5 | <hex> | <hex> | - | MATCH |

regression threshold: 5.00%; regressions: 1, missing: 0
```

The process then exits 1 and stderr carries
`Error: bench compare: 1 regression(s), 0 missing, determinism Some(true)`.

## 5. CI gating recipe

Store one JSON report per tracked configuration. Keep the file small by using a
stable, explicit case list rather than the defaults.

```bash
# One-off: produce the baseline on the reference machine and commit it as an artefact.
./target/release/denno bench --url http://127.0.0.1:8098 \
  --pp 512,2048 --tg 64 --depths 0 --rounds 5 --seed 801 \
  --json bench-baseline.json

# Every candidate build: same flags, same seed, same depths, same rounds.
./target/release/denno bench --url http://127.0.0.1:8098 \
  --pp 512,2048 --tg 64 --depths 0 --rounds 5 --seed 801 \
  --json bench-candidate.json

# Gate. Non-zero exit fails the job.
./target/release/denno bench \
  --compare bench-baseline.json bench-candidate.json \
  --fail-on-regress-pct 5
```

Rules that make the gate meaningful:

1. Compare only runs with the same `--pp`, `--depths`, `--tg`, `--rounds`, and
   `--seed`. A different matrix shows up as `missing`, which does not fail, so a
   mismatched baseline can silently pass.
2. Compare only the same server build flags (`--quant`, `--name`, cache and
   queue settings) and the same backend. A backend change is a new baseline.
3. Run the determinism check with `DENNO_Q4_DETERMINISTIC=1` on both sides if
   MD5 equality is part of the gate (see §6).
4. Keep the baseline on the same machine class. Absolute tok/s is not portable.
5. Pick the threshold from observed noise, not from zero. Run the same commit
   twice against itself and use a threshold above the largest observed swing;
   with the default `--rounds 3`, a few percent of jitter is normal.

## 6. Determinism controls

`denno bench` always sends `temperature = 0` for pp, tg, turn-2, and the
determinism check. On top of that, the engine has to be deterministic, and that is
not the default.

- `DENNO_Q4_DETERMINISTIC=1` (also accepts `true`, `on`, `yes`, case-insensitive).
  Default off. When off, the CPU Q4_K/Q5_K fused kernels accumulate in a different
  lane order than the scalar path, and expert residency changes between rounds can
  flip the lowest bits, so the same prompt yields a different MD5. Turn it on for
  both the baseline run and the candidate run, and leave it off in normal serving.
  The flag changes numerics, so never compare a run with it against one without.
- Prefill chunk width also changes the accumulation order. If you gate on MD5,
  pin `DENNO_PF_CHUNK` (and `DENNO_PF_CHUNK_BALANCE` if it is used) identically on
  both sides.
- `--seed` fixes prompt synthesis, not sampling. Two runs with different `--seed`
  values are different measurements even at `temperature = 0`.
- `--no-flush` changes cache state across rounds. Keep it on the same side of both
  runs, or leave it off.
- `--quant int8` keeps same-seed determinism but is not bit-identical to `fp32`.
  A `fp32` baseline cannot be used to gate an `int8` build.
- Interpreting the JSON: `determinism.verified = true` means every round
  completed and all `round_md5` entries match. `false` means either a differing
  hash or a short `round_md5` from an errored round; check `determinism.error`
  and the top-level `errors` array before blaming the kernels.
- `--compare` re-checks MD5 equality across the two files. A mismatch is a hard
  failure with no threshold, and the Markdown row reads `MISMATCH`.

## 7. Noisy-neighbour caveats

- **First round includes one-time work.** Weight upload, allocator warm-up, and
  first-touch page faults land in round 1. Use `--rounds 3` or more and rely on
  the median, never on a single run.
- **Shared hosts move.** Other processes on the same CPU, NUMA effects, and
  cgroup throttling change tok/s without changing the build. Record the host
  class with the baseline; do not compare across machines.
- **The flush request is traffic.** By default the client sends a request close
  to the context length between rounds. With a small `--request-timeout-secs` or
  `--max-queue-depth` on the server, long pp cases can time out or get a 503
  overload response. Raise those limits for the benchmark server and expect
  occasional `errors` entries otherwise.
- **Concurrent clients invalidate the measurement.** Any other request during a
  round perturbs TTFT, TPOT, and cache state. Benchmark on an idle server.
- **Queueing looks like slowness.** A 503 / overloaded response becomes a case
  `error`, not a low rate. Check `errors` before interpreting a delta; a run with
  errors is not a valid baseline.
- **Drain and restart.** A server that is draining or restarting between rounds
  produces missing or errored cases. Wait for `/ready` again and re-run.
- **Medians hide tails.** `pp_tok_s` is derived from median TTFT; `itl_ms` carries
  p90/p99. A build can keep the median and still regress the tail, so gate on both
  when latency matters.
- **Turn-2 reuse is sensitive to cache state.** A cold cache, a different
  `--reuse-depth`, or an intervening flush changes `speedup` dramatically. Treat
  it as a separate signal, and do not compare it across different `--rounds`.

## 8. Verifying this document

Re-run these against the source before changing examples:

```bash
# flag definitions and defaults
grep -n 'Bench {' -A 55 crates/denno-cli/src/main.rs

# --compare contract and exit path
grep -n 'fail_on_regress_pct\|run_compare\|render_compare_ansi' crates/denno-cli/src/main.rs

# comparison semantics, verdicts, thresholds
grep -n 'fn compare_reports\|fn mk_row\|fn pct_delta\|exit_fail' crates/denno-cli/src/bench.rs

# JSON report structs
grep -n 'pub struct BenchReport' -A 24 crates/denno-cli/src/bench.rs
grep -n 'pub struct PpRow' -A 15 crates/denno-cli/src/bench.rs
grep -n 'pub struct TgRow' -A 20 crates/denno-cli/src/bench.rs
grep -n 'pub struct Determinism\|pub struct Turn2' -A 15 crates/denno-cli/src/bench.rs

# determinism knobs
grep -rn 'DENNO_Q4_DETERMINISTIC' crates/denno-engine/src/model/moe_expert.rs
grep -rn 'DENNO_PF_CHUNK' crates/denno-engine/src/auto_config.rs crates/denno-engine/src/model/exp_pf.rs

# end-to-end CLI contract test
grep -n 'fn ' crates/denno-cli/tests/bench_compare_1064.rs
```

Run the CLI contract test when touching `--compare`:

```bash
cargo test --release -p denno-cli --test bench_compare_1064
```

If a flag disappears, delete it from the table rather than keeping a stale
example.
