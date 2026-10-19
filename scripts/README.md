# scripts/ — CPU-only checkers and drivers

Every script here runs on a CPU-only host with the Python standard library only
(`urllib`, `http.server`, `unittest`, `subprocess`). Nothing here needs `requests`,
a GPU, Docker, or a running server to *self-test*: the checkers that talk to a
server do so only when you point them at one, and their self-tests use a throwaway
in-process fake server on an ephemeral port.

Two rules that hold everywhere in this directory:

- Ports **8080** (serve default) and **8090** (production) are refused by the
  scripts that take a `--base-url`. Use 8095 or anything else.
- The scripts **report**; they never start, restart or stop a server.

## Scripts

| Script | Purpose | How to self-test | Exit codes | Needs GPU host? | Serves |
|---|---|---|---|---|---|
| `long_conversation_check.py` | Drives a scripted 10-turn conversation at an Anthropic-style `/v1/messages`, appending each reply so the conversation really grows. Flags repetition collapse (`repeat_line`, `repeat_sentence`, `long_line`), chat-template leakage (`template_marker`), fabricated user turns (`fake_turn`), empty replies (`empty`), truncation (`max_tokens`, reported only) and, with `--expect-cap`, a `--temperature-cap` regression. | `python3 scripts/long_conversation_check.py --self-test` (no socket); `--dry-run` prints the 10 request bodies and sends nothing | 0 clean / 2 hard failure or cap regression / 3 infrastructure (connection, HTTP != 200, malformed body, unreadable fixture) | Yes, for a real run against 8095 | P4-1 |
| `test_long_conversation_check.py` | `unittest` suite for `long_conversation_check.py`: detectors, verdict determinism, the temperature-cap path, and the full request path against a fake server. | `python3 scripts/test_long_conversation_check.py -v` | 0 all pass / 1 at least one test fails (unittest convention) | No | P4-1 |
| `needle_check.py` | Long-context needle recall: builds a filler + needle payload, tries `/v1/chat/completions` and falls back to `/v1/messages` on 404, and checks the answer against the expected passphrase. | `python3 scripts/needle_selftest.py` (separate file); the script itself has no `--self-test` | 0 recall ok / 1 recall failed (= regression) / 2 transport or parse failure (never confused with 1) | Yes, for a real run | P4-4 |
| `needle_selftest.py` | Self-test for `needle_check.py`: starts a fake server and verifies each exit code, the 404 fallback, and payload determinism. | `python3 scripts/needle_selftest.py -v` | 0 all pass / 1 at least one check fails | No | P4-4 |
| `needle_fixture.json` | Filler, needle and expected answer for `needle_check.py`. | Read by the script and its self-test | — | — | P4-4 |
| `cold_ab.py` | Cold single-shot A/B driver: alternates two binaries cold-start, sends one fixed request each, and reports median wall time and delta %. | `python3 scripts/cold_ab.py --self-test`; `--dry-run` prints the payload sha256 and the round plan without starting anything | 0 ok / 2 failure or guard trip / 3 infrastructure | Yes (real measurement) | P3-4 |
| `test_cold_ab.py` | `unittest` suite for `cold_ab.py`: round plan, median/delta maths, payload determinism, safety guards, and the full path against a fake server. | `python3 scripts/test_cold_ab.py -v` | 0 all pass / 1 at least one test fails | No | P3-4 |
| `dashboard.py` | P5-1 dashboard: reads every `*.json` from `cold_ab.py --json` in a directory plus an optional `PF_BENCH` log, and writes one self-contained HTML file (inline SVG, no script/CDN/fonts) with **prefill (s)** and **decode (ms/token)** on two separate axes, x = measurement order. `--protocol-change LABEL` draws a vertical marker where that era starts (repeatable); a panel with fewer than 2 points says "not enough points". Decode points come from the PF lines (`wall * 1000 / tokens`, tokens = the line's `output_tokens` else `--max-tokens`); the cold A/B JSON has no decode number, so it never invents one. Output is byte-identical for the same inputs. | `python3 scripts/dashboard.py --self-test` | 0 ok / 1 unusable input (bad JSON, missing dir, no points; no partial `--out` is left) / 2 usage error | No (the chart is CPU-only; the *data* needs a GPU host) | P5-1 |
| `test_dashboard.py` | `unittest` suite for `dashboard.py`: byte-identical output on two runs, both panels with their own units, marker only when `--protocol-change` matches, the single-point "not enough points" path with no NaN/Inf, malformed JSON naming the file (exit 1), one point per PF line, and `--help`. | `python3 scripts/test_dashboard.py -v` | 0 all pass / 1 at least one test fails | No | P5-1 |
| `fixtures/long_conversation_turns.json` | The fixed 10 prompts for `long_conversation_check.py`. Do not reorder or renumber; append only, and say why in `docs/TASKS.md`. | Asserted by `test_long_conversation_check.py` | — | — | P4-1 |

## Running the whole GPU-free gate

```sh
scripts/run_cpu_checks.sh
```

Runs, in order: the kernel invariant tests, the engine lib tests, the server tests
and every Python self-test below. It stops at the first failure, prints one line per
stage plus a final `PASS`/`FAIL`, and exits 0 only if everything passed.

- Exit: 0 all stages passed / 1 a stage failed / 2 usage or setup problem
  (no `cargo` or no `python3` in `PATH`).
- `CARGO_TARGET_DIR` is honoured and defaults to `/tmp/denno-test-target`, because
  `target/debug` in a bind-mounted checkout can be owned by another user (EACCES).
- `PYTHON=/path/to/python` selects the interpreter (default `python3`).
- Stateless: no network, no server start/stop, no process inspection. Per-stage
  logs go to `$CARGO_TARGET_DIR/run_cpu_checks_logs/<stage>.log`; on failure it
  prints the last 40 lines of the failing stage.
- `scripts/run_cpu_checks.sh --list` prints the stages without running anything.

| Stage | Command |
|---|---|
| `kernel_invariants` | `cargo test -p denno-engine --test kernel_invariants` |
| `engine_lib` | `cargo test -p denno-engine --lib` |
| `server` | `cargo test -p denno-server` |
| `cold_ab_selftest` | `python3 scripts/cold_ab.py --self-test` |
| `cold_ab_tests` | `python3 scripts/test_cold_ab.py` |
| `longconv_selftest` | `python3 scripts/long_conversation_check.py --self-test` |
| `longconv_tests` | `python3 scripts/test_long_conversation_check.py` |
| `needle_selftest` | `python3 scripts/needle_selftest.py` |
| `needle_help` | `python3 scripts/needle_check.py --help` |
| `dashboard_selftest` | `python3 scripts/dashboard.py --self-test` |
| `dashboard_tests` | `python3 scripts/test_dashboard.py` |

This covers the CPU half of P4-1, P4-4, P5-1 and the P3-4 / P5-1 driver tests. It
cannot cover the GPU half (a real 8095 run, the cold A/B measurement); those stay
human steps.

## The long-conversation run (P4-1)

The self-tests prove the detectors, not the model. The real run needs a GPU host
with a serve started using `--temperature-cap 0.7`:

```sh
# collapse detection only (temperature is not sent; the server's cap applies)
python3 scripts/long_conversation_check.py --base-url http://127.0.0.1:8095 \
    --out /tmp/lc.json

# cap regression detection: ask for 1.0 and fail if the server does not clamp to 0.7
python3 scripts/long_conversation_check.py --base-url http://127.0.0.1:8095 \
    --temperature 1.0 --expect-cap 0.7 --out /tmp/lc_cap.json

# re-judge a stored transcript without a server (same verdict, deterministic)
python3 scripts/long_conversation_check.py --replay /tmp/lc_cap.json --expect-cap 0.7
```

`--temperature` is only sent when you pass it. The default sends none, so the
serve-side cap stays in control and the run can't mask a regression by
overriding it.