# Testing and Quality Gates

How to run every test layer in this workspace, what each layer actually proves,
and why the gates are invoked the way they are.

The short version: all CPU layers are runnable from a clean checkout with a Rust
toolchain and Node.js; the model-dependent exactness layer is opt-in and needs
cached weights; GPU kernels are compile-checked here but executed only on a GPU
host.

## 1. Quick reference

| Layer | Command | Needs |
|---|---|---|
| Workspace unit + integration tests | `cargo test --release --workspace -- --test-threads=1` | Rust toolchain |
| One crate | `cargo test --release -p denno-server -- --test-threads=1` | Rust toolchain |
| One integration file | `cargo test --release -p denno-engine --test kernel_invariants` | Rust toolchain |
| Viewer JS unit tests | `node --test 'crates/denno-view/tests/*.test.mjs'` | Node.js (no `npm install`) |
| CPU-only gate script | `scripts/run_cpu_checks.sh` | Rust + `python3` |
| CUDA compile check | `cargo check -p denno-engine --features cuda` | Rust toolchain, no GPU |
| Exactness (real weights) | `cargo test --release -p denno-engine --test exactness -- --ignored --nocapture` | Cached models (see section 6) |
| GPU kernel tests | not on a CPU-only host | GPU + `--features cuda` + `DENNO_CUDA=1` |

Pass/fail counts are per test binary. To see them without running:
`cargo test --release -p denno-engine -- --list`.

## 2. Where the tests live

| Crate | Unit tests | Integration tests |
|---|---|---|
| `denno-telemetry` | in `src/` (`#[cfg(test)]`) | `tests/`: wire golden bytes, codec robustness/fuzz, NDJSON parity, hub stress, trace I/O |
| `denno-engine` | `src/lib.rs` and module tests, including kernel-side unit tests | `tests/`: `kernel_invariants` (static `.cu` checks, no GPU), quant/dequant references, model-free forward paths, plus `#[ignore]`d exactness and GPU files |
| `denno-server` | in `src/` | `tests/`: one file per HTTP contract area (OpenAI, Anthropic, Ollama, auth, CORS, limits, scheduler, streaming, metrics, lifecycle) |
| `denno-cli` | `src/main.rs` | `tests/`: help, config parsing, inspect, tokenize, ablation, bench comparison; some spawn the real binary |
| `denno-view` | in `src/` | `tests/*.rs` check the served assets and the HTTP surface; `tests/*.test.mjs` are Node unit tests for the pure JS modules |

`denno-view` deserves a note: it is a binary crate, so its Rust tests cannot
import the router. The Rust files therefore either `include_str!` the served
HTML/JS and assert the DOM ids and bindings line up (`ui813_dash_3d.rs` and
friends), or start the real binary on an ephemeral loopback port and speak raw
HTTP to it (`http_contract_949.rs`, `http_security_1209.rs`). The numeric logic
lives in the `.mjs` modules and is tested by `node --test`.

Server tests bind `127.0.0.1:0`, never a fixed port, so they do not collide with
a `denno serve` you happen to have running.

## 3. Why `--release`

Always run the gate with `--release`:

- The assertions compare exact generated text and exact output-token counts.
  Release is the profile that ships; keeping the gate on the same profile avoids
  "passes in debug, differs in release" surprises.
- The workspace sets `opt-level = 3` and `lto = "thin"` for release, and
  `.cargo/config.toml` adds `-C target-cpu=native`. Engine tests are slow enough
  in debug that people stop running them.
- The first `--release` build after a clean checkout compiles every test binary
  (about 140 binaries) with thin LTO. That is the slow part; subsequent runs
  only rebuild what changed.

`CARGO_TARGET_DIR` is honoured. In a bind-mounted checkout, `target/` can be
owned by another user and produce `EACCES`; point the build elsewhere:

```bash
CARGO_TARGET_DIR=/tmp/denno-test-target cargo test --release -p denno-engine --lib
```

## 4. Why `--test-threads=1`

The gate runs with one test thread on purpose. This is not a workaround for a
flaky test; it is required by how several tests observe process-global state.

- **Environment variables.** Engine tests set and remove process environment
  variables (`std::env::set_var` / `remove_var`) to exercise config resolution:
  `DENNO_EXPERT_VRAM_MB`, `DENNO_GPU_WEIGHT_MB`, `DENNO_STEER_DIR`,
  `DENNO_CUDA_GEMM_MIN_M`, and others in `auto_config.rs`, `steer.rs`,
  `expert_vram.rs`, `body_limit_946.rs`, `drain_timeout_1006.rs`. Environment is
  per process, not per test, so parallel tests overwrite each other and read
  values another test installed.
- **Once-per-process decisions.** Some behaviour is latched in a `OnceLock` or
  a `static` the first time it is observed (profiling enablement in
  `failure_release_log.rs`, forced flags in `mem_audit.rs` and `pf_pool.rs`).
  A second test cannot change it, and parallel scheduling makes which test wins
  nondeterministic.
- **Process-wide counters.** Tests that assert "one counter increment per
  request" share atomics with every other test in the binary
  (`model/qwen4exp/tests.rs`: `--test-threads=1` is what makes the counter
  meaningful).
- **Single-device GPU work.** CUDA tests share one device and one context; they
  are not safe to overlap. They are also skipped on a CPU-only host, but the
  serial rule stays for the GPU gate.
- **CI parity.** `.github/workflows/ci.yml` runs exactly
  `cargo test --workspace -- --test-threads=1`, so a local green run means the
  same scheduling as CI.

`cargo test` still runs different test binaries one at a time, so files that
touch the same global state in different binaries do not race each other.
`--test-threads=1` serializes tests inside each binary, which is where the
shared state lives.

Do not "fix" an order-dependent failure by dropping this flag. Fix the test so
it saves and restores the global it touches, or move the assertion into a
process of its own.

## 5. The viewer JS tests

The `.mjs` modules are dependency-free ES modules imported directly from
`crates/denno-view/src/`. `node --test` is the whole runner; there is no
`package.json`, no install step, and no browser.

```bash
# from the repository root: files must be named explicitly
node --test 'crates/denno-view/tests/*.test.mjs'

# or let node discover them from the crate directory
cd crates/denno-view && node --test
```

Both forms run all 30 test files (536 tests as of the snapshot below).

Pitfall: on Node.js 22, passing the directory itself
(`node --test crates/denno-view/tests/`) treats it as a module path and exits
with `Cannot find module .../tests`. Use the glob (quoted, so node expands it)
or `cd` into the crate. The file header comment in some tests says
`node --test crates/denno-view/tests/`; prefer the two forms above.

The Rust and JS halves of `denno-view` cover different things and both are
needed: the Rust files prove the assets are wired to the modules and served
correctly, the JS files prove the numeric/state logic in those modules.

## 6. Exactness tests (real weights)

`crates/denno-engine/tests/exactness.rs` compares generated text against
literal reference strings produced by an independent implementation
(HF transformers, fp32, greedy). Every test is marked `#[ignore]` because it
needs model files, and CI runs it in a separate job.

| Test | Environment variable | Default location |
|---|---|---|
| `qwen_1plus1_matches_baseline` | `DENNO_QWEN_DIR` | `~/.cache/denno/hub/models--Qwen--Qwen2.5-0.5B-Instruct` |
| `qwen_prefix_cache_keeps_exact_output` | `DENNO_QWEN_DIR` | same |
| `qwen_exact_decoding_is_reproducible` | `DENNO_QWEN_DIR` | same |
| `qwen_int8_mode_is_deterministic` | `DENNO_QWEN_DIR` | same |
| `gpt2_japan_capital_matches_baseline` | `DENNO_GPT2_DIR` | `~/.cache/denno/hub/models--openai-community--gpt2` |
| `qwen3_qknorm_matches_baseline` | `DENNO_QWEN3_DIR` | `~/.cache/denno/hub/models--Qwen--Qwen3-0.6B` |
| `qwen35_9b_multiturn_prefix_cache_exact` | `QWEN35_9B_DIR` | none; skipped unless set |

(The module header lists the first three variables; `DENNO_QWEN3_DIR` and
`QWEN35_9B_DIR` are read by their tests and are not in that header.)

Run one file, or one test, with the harness arguments after `--`:

```bash
cargo test --release -p denno-engine --test exactness -- --ignored --nocapture
cargo test --release -p denno-engine --test exactness -- --ignored --nocapture qwen_1plus1
```

What the tests pin:

- The exact output string and `StopReason` for a fixed prompt, temperature 0,
  seed 42. Multibyte truncation at `max_tokens` is asserted literally (GPT-2
  ends in U+FFFD), so a tokenizer change is visible.
- Determinism: the same request twice returns the same text and token count.
- Prefix cache does not change bytes: a cold engine and a warm engine with a
  restored KV prefix produce identical output, and multi-turn 9B runs must hit
  `PrefixHit::Entry` while still matching a cold full-prefill reference.
- `QuantMode::Int8` is deterministic; it is not required to equal fp32
  (rounding differs by design).

**A skipped test still reports as passed.** When the model directory is absent,
`model_dir()` prints `skip: model not cached ...` and returns; the harness
counts that as a pass. `0 failed` therefore does not prove exactness ran. Check
the output with `--nocapture` and confirm there are no `skip:` lines, or set the
directories explicitly.

Exact strings are tied to the numerics of the build. CI sets
`RUSTFLAGS="-C target-cpu=x86-64-v3"` so every runner enables the same FMA
baseline; the local `.cargo/config.toml` uses `target-cpu=native`. Do not
rebaseline a reference string from a differently tuned build, and do not loosen
an assertion to make a change pass. The allowed drift classes and the
rebaselining procedure are in [`NUMERIC_POLICY.en.md`](NUMERIC_POLICY.en.md)
(Japanese: [`NUMERIC_POLICY.ja.md`](NUMERIC_POLICY.ja.md)).

CI runs the exactness job on pushes to the main branch only, after
`denno pull Qwen/Qwen2.5-0.5B-Instruct` and `denno pull Qwen/Qwen3-0.6B`, via:

```bash
cargo test --release --workspace -- --ignored
```

## 7. Ignored benches and GPU tests

There are about 100 ignored tests in the workspace. `exactness.rs` is only one
group; the rest are kernel microbenchmarks and GPU comparisons. Three rules:

- `#[ignore]` is not "dead code". Keep them compiling: `cargo test --release`
  builds every test binary even when the tests are filtered out, so a broken
  ignored test fails the gate at compile time.
- GPU test files are wrapped in `#[cfg(feature = "cuda")]` and additionally
  require `DENNO_CUDA=1`; without the feature the file contains no tests, and
  without the variable the tests skip. On a CPU-only host they must not be run
  at all.
- The CUDA feature can still be compile-checked without a GPU, because the
  kernels are prebuilt PTX and `libcuda` is resolved at runtime by `dlopen`:

```bash
cargo check -p denno-engine --features cuda
```

This is what CI runs. It catches signature and constant drift; it proves
nothing about kernel results.

Benchmarks are not a gate. `docs/BENCHMARKING.md` covers `denno bench` and
baseline storage; `scripts/bench_all.sh` drives the ignored benchmark examples,
which need the appropriate hardware.

## 8. The Python checkers

`scripts/` contains standalone checkers and their `unittest` suites. They use
only the Python standard library and their self-tests use in-process fake
servers, so they need no GPU and no running `denno`:

```bash
scripts/run_cpu_checks.sh          # every stage, stop at first failure
scripts/run_cpu_checks.sh --list   # print the stage table
```

The script runs, in order: `kernel_invariants`, the engine library tests, the
server tests, then the Python self-tests and suites for the cold A/B driver, the
long-conversation checker, the needle checker, the dashboard, and the n-gram
oracle. Logs go to `$CARGO_TARGET_DIR/run_cpu_checks_logs/` (default target is
`/tmp/denno-test-target`). See [`scripts/README.md`](../scripts/README.md) for
the per-script contract, exit codes, and the reserved ports.

Note that this script invokes `cargo test` without `--release` and without
`--test-threads=1`. It is a fast, dependency-light smoke gate for the Python
tooling; use the workspace release command in section 1 as the authoritative
Rust gate.

## 9. CI

`.github/workflows/ci.yml` has two jobs:

1. `build-test` (every push and pull request)
   - `cargo build --workspace --release`
   - `cargo test --workspace -- --test-threads=1`
   - `cargo check -p denno-engine --features cuda`
2. `exactness` (pushes to the main branch only)
   - pulls Qwen2.5-0.5B and Qwen3-0.6B
   - `cargo test --release --workspace -- --ignored`

CI sets `RUSTFLAGS="-C target-cpu=x86-64-v3"` instead of `native` for two
reasons: the build cache is shared between runners, and a native binary built
for one CPU generation can raise `SIGILL` on another; and x86-64-v3 includes
FMA, which keeps floating-point behaviour close to the exactness references.
Locally, leave `.cargo/config.toml` alone unless you are reproducing CI.

## 10. Measured snapshot

Taken on a CPU-only host with rustc 1.99.0 and Node.js 22.22.1, after a full
`cargo test --release --workspace -- --test-threads=1` and the Node run. Counts
grow with every task; treat them as an order-of-magnitude check, not a contract.

| Crate | Passed | Failed | Ignored |
|---|---:|---:|---:|
| `denno-cli` | 339 | 0 | 1 |
| `denno-engine` | 1536 | 0 | 99 |
| `denno-server` | 663 | 0 | 0 |
| `denno-telemetry` | 73 | 0 | 0 |
| `denno-view` (Rust) | 45 | 0 | 0 |
| **Total (143 test binaries)** | **2656** | **0** | **100** |

`node --test` on `crates/denno-view/tests/*.test.mjs`: 30 files, 536 passed,
0 failed. The ignored count includes the 7 exactness tests; none of them ran in
this snapshot because no model directories were configured.

## 11. Before opening a pull request

1. `cargo test --release --workspace -- --test-threads=1` is green.
2. If `crates/denno-view/src/*.mjs` changed,
   `node --test 'crates/denno-view/tests/*.test.mjs'` is green.
3. If engine code near CUDA kernels, PTX constants, or `kernels/*.cu` changed,
   `cargo check -p denno-engine --features cuda` is green.
4. If a numeric path, tokenizer, chat template, or prefix-cache path changed,
   run the exactness file with the model directories set and confirm the tests
   actually executed (no `skip:` lines). If an output legitimately changed,
   follow `NUMERIC_POLICY.en.md` instead of editing the literal.
5. If `scripts/` changed, run its self-tests or `scripts/run_cpu_checks.sh`.
6. Report the pass/fail/ignored counts from the run, not just "tests pass".

## 12. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Cannot find module '.../tests'` from node | directory passed to `node --test` on Node 22 | use the quoted glob or `cd crates/denno-view && node --test` |
| `EACCES` writing `target/` | bind-mounted checkout owned by another user | set `CARGO_TARGET_DIR` |
| A test passes alone, fails in the suite | shared env var or `OnceLock` | run with `--test-threads=1`; if it still fails, the test leaks global state, fix the test |
| Exactness "passes" instantly | model not cached, test skipped | set the directory variable and rerun with `--nocapture` |
| First release build takes many minutes | thin LTO over ~140 test binaries | expected; use a persistent `CARGO_TARGET_DIR` |
| Harness flag ignored | arguments on the wrong side of `--` | cargo flags first, then `--`, then test-harness flags (`--test-threads`, `--ignored`, `--nocapture`) |
| CUDA check fails without a GPU | it should not; only `cargo check` is run | `cargo check`, not `cargo test --features cuda` |
