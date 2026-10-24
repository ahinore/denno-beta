> Japanese original: [../70_infra.md](../70_infra.md)

> [Index](README.md) / Previous: [Volume 7 Quality](60_quality.md) / Appendix: [all phases](90_phase_index.md)

# Volume 8 — Measurement, verification, operation: when a number can be trusted

> Sources: the development records (decision log, current-value cards), `../../KPI.md`,
> `../../NUMERIC_POLICY.ja.md`. Every number is a measurement that exists in the record; wherever this
> volume says "estimated", it is an estimate. Internal details (host names, IPs, container names, user names,
> absolute paths) are not written in this volume.

## The question this volume answers

**This project rebuilt its idea of "when a number can be trusted" several times.** What should be measured,
and what must not be measured? How is "it got faster" proven? That history is this volume.

There are two ways to read it.

- If you are about to measure the same engine: §1 (the instruments) → §6 (the list of pitfalls) → §7 (the
  acceptance procedure)
- If you want to quote past numbers: read §5 (when a number can be trusted) and stop there.

---

## 1. The instruments

| Instrument | What it answers | Status |
|---|---|---|
| `GPU_VERIFY.sh` | Did the output change (3 anchors) + decode speed | in use |
| `ANCHOR_REBASE.sh` | Is it deterministic (restart and measure twice) · re-baselining the anchors | in use |
| `scripts/prefill_bench.py` | Cold prefill measurement (nonce, median over repeats) | in use (the official procedure) |
| `PF_BENCH.sh` | Quick prefill measurement (5,686-token class) | auxiliary |
| cold A/B driver | Alternately starting two builds and comparing them | in use |
| `DENNO_PROFILE` | Per-phase breakdown of the serve path | conditional (§6.2) |
| `DENNO_PF_DBG` | Prefill wall time split by phase | conditional (§6.3) |
| `DENNO_FC_KPROF` / `[GDN_KPROF]` | Real kernel time for a specific phase | conditional |
| nsys | Per-kernel GPU time, H2D/D2H counts and bytes | in use (relative comparisons) |
| ncu | Inside a kernel (SM occupancy, instructions) | **unavailable** (permissions, §4.4) |
| ptxas `-v` / `cuobjdump` / `nvdisasm` / `compute-sanitizer` | Registers, stack, SASS, invalid accesses | in use |
| Kernel invariant tests | Statically checking the assumptions of a `.cu` without a GPU | in use |
| needle / long-conversation regression | Quality regression (the part that runs without a GPU) | in use |
| dashboard | KPI movement per generation | data added by hand |

---

## 2. The pillars of verification: anchors and the wide comparison

### 2.1 What an anchor is

The md5 of a result generated from a fixed input and a fixed seed. **Same binary, same input, always the same
md5.** A change that touches the numeric path is not accepted unless these three (1b / 2b / full) match.

- The anchor baselines live in `data/verify/anchors_baseline.txt`. **They are not in the repository** (they are
  environment-dependent).
- Whenever an anchor moves, the reason must be explained. A move that cannot be explained is treated as a
  **failure**.
- The anchor has moved a number of times. There are only three kinds of reason:
  1. A numeric-relaxation class (R1–R4) was deliberately adopted (e.g. batching attention on the GPU).
  2. The prompt itself changed (e.g. the default thinking mode changed to brief).
  3. **An unexplained move, such as reading uninitialised memory** (e.g. there was a drift that did not
     reproduce on the next build. It is still open; if it reproduces, stop and investigate).

### 2.2 The wide comparison

Three anchors cannot catch every input, so the practice is combined with **dumping 8 questions × 150 tokens to
a `wide` file and comparing it with `cmp` for an exact match**. One byte different is a mismatch.

- Accepting a speed improvement requires **both** "anchors 3/3 MATCH" and "wide identical".
- If the output token count changes substantially, suspect a logic regression rather than numeric error
  (`NUMERIC_POLICY` §5).

### 2.3 How determinism is checked

Two runs in the same process are not enough. **Restart the process and measure twice** (`ANCHOR_REBASE.sh`
check mode). There has been a bug where "only the first request returns a different value" (KV mirror
generation mismatch, an in-step reallocation freeing the buffer of an in-flight kernel), so the measurement
**includes the first request right after start-up**.

---

## 3. How speed measurement evolved

### 3.1 Phase 1: "the second run on the same server" was the official number (wrong)

For a long time the same prompt was run twice and the second run was taken as the official value. That run
**hits the prefix cache**, so it cannot be used to compare prefill. It was the main cause of ±30% scatter.

- Concretely, one build recorded "−9.7%" which, re-measured cold, was −5.7%.
- A warm second run is only usable for a relative comparison within the same session.

### 3.2 Phase 2: single cold run + alternating A/B (the official procedure)

The current official procedure:

- **prefill**: `scripts/prefill_bench.py <characters> <repeats>`. **The argument is characters**, not tokens
  (e.g. `16000` ≈ 3,161 tokens, `65000` ≈ 12,850 tokens). A nonce is mixed into the prompt so the prefix cache
  is always missed, and the median of several runs is taken.
- **Comparing builds**: stop the server → start it → one request, across two builds, **alternately**, for
  3 rounds. Within one binary the scatter is about 1.5 ms, so anything smaller is judged "no change".
- **decode**: discard one warm-up, take the median ms/token. Do not mix TTFT with generation speed.
- **Acceptance floor**: a prefill improvement must be at least 0.3 s (cold). Anything smaller is not adopted,
  because it cannot be told apart from measurement error.

### 3.3 Running "measurement windows"

There is only one GPU, so measurement is declared as a **window** and made exclusive.

- Before starting, write a 【declaration】 into the decision log stating the purpose, the port used, the
  expected duration and the exit condition.
- While a window is open, no other role builds, runs GPU_VERIFY, benches, or advances the shared tree.
- On closing, write 【declaration released】 and the result (measured values and whether it was adopted).
  **The role that opened the window decides whether to adopt.**
- Forgetting to close a window leaves the other roles stopped in "waiting". It has happened several times, so
  the supervising side checks on every pass whether a declaration is still valid on paper.

---

## 4. Caveats per instrument (what measurement has established)

### 4.1 The absolute values of `DENNO_PROFILE` are unusable

The measurement changes the thing being measured. Without sampling (every span) it inflates to **2.7–4x** the
measured value (e.g. 137 ms/token where the real value is 33.5 ms/token). Even with sampling (taking a span
once every N steps) about 10% overhead remains.

→ **Do not use absolute values. Look at ratios only.** And when looking at a ratio, corroborate it with
GPU-side time from nsys.

### 4.2 The profile accounting has been rebuilt repeatedly

The problem "the sum of the phases does not explain the wall time" came up again and again. What was
accumulated in response:

- Measure a `via` column (CPU path or GPU path).
- Fix the denominator of `tok/call` (do not confuse call count with token count).
- Define the remainder as "total − named phases" (`sl_lin_other` / `sl_step_other`).
- Split the measurement points themselves (`d2h_state` divided into logits / kv_flush / gdn_flush / attn_pf).

**Lesson**: leave "other" alone and the real bottleneck hides inside it. Always give the remainder a name.

### 4.3 The phase percentages of `DENNO_PF_DBG` are not KPIs

The phase buckets are **a complete partition of wall time**, not a partition of kernel time. A bucket mixes
"kernels issued in that interval + GPU idle + host delay". So "ffn_down is 17.6%" does not mean "the ffn_down
kernel used 17.6% of the time".

- Do not put phase percentages into a KPI table.
- Always state the denominator (that run's wall time). Do not quote the percentage of a run whose denominator
  is 6x different as a current value.

### 4.4 ncu is unavailable

Looking inside a kernel (SM occupancy, instruction mix) requires ncu, which this environment refuses with
`ERR_NVGPUCTRPERM`. Lifting that needs a host-side configuration change and recreating the container, during
which the server is down. **The decision is to leave it until maintenance.**

What can be used instead: `ptxas -v` (registers, stack, spills), `cuobjdump -sass`, `nvdisasm`,
`compute-sanitizer`, and interval measurement with host-side CUDA events.

### 4.5 nsys

It gives per-kernel GPU time plus H2D/D2H counts and bytes. **Use it for relative comparisons; take absolute
values from real-machine KPIs.** CUPTI activity runs without special permissions.

### 4.6 m=1 in bench is L2-relative

It reads the same buffer repeatedly, so it measures the state once loaded into L2. **Usable for finding the
bandwidth ceiling, not for predicting real-machine speed.** Decide with relative change plus the real-machine
KPI.

---

## 5. When a number can be trusted (six checks before quoting)

1. **Denominator**: model (9B int8 / 27B q4), quantisation, token count, cold or warm.
2. **Was it cold**: the prefix cache was not hit (nonce present, or the first request after start-up).
3. **Median or single run**: how many runs. Was the warm-up discarded.
4. **Measured or estimated**: a value measured inside the container (no GPU) is "measured (container)" and is
   not a GPU-side value. GPU-side values are written "estimated".
5. **Which instrument**: PROFILE / PF_DBG / nsys / real-machine wall. Never mix numbers from different
   instruments and call it "N times".
6. **Did the anchor match**: for a change that touched the numeric path, if the anchor moved without
   explanation, it is a failure even if it got faster.

---

## 6. The list of pitfalls (only the ones that recurred)

### 6.1 Builds and trees

- Do not run several agents on the shared tree at once. Builds must use separate `CARGO_TARGET_DIR`s.
- Starting work while the shared tree's HEAD is stale causes rework at integration time.
  **Check that the base is current before starting.**
- Do not use `git add -A`. Name the pathspecs. There was an accident that pulled in 43,249 files (build
  artefacts and internal documents) and required rewriting history. Internal documents are excluded via
  `.gitignore`.
- Do not commit internal documents (work reports, handovers, verification scripts, verification data), because
  they contain IPs, host names and user names. A personal-information grep before push is mandatory.

### 6.2 Running the server

- **Stop the server before replacing the binary.** A running binary cannot be overwritten (Text file busy).
- Stopping with `pgrep -f` or by matching `for p in /proc/*` cmdline makes **your own command line match the
  pattern and kills you** (exit 143). Take the pid first and kill it explicitly.
- Fix one port as production. Two instances on the same GPU run out of VRAM, fall back to CPU and become
  "strangely slow" (measured above 400 s).
- After swapping the server for a measurement, **always restart it with the production arguments and check
  health**. Leaving it stopped is the most expensive outcome.

### 6.3 Running the agents (development side)

- Passing a Japanese prompt into a container turns it into `?`. Write prompts in ASCII.
- When the turn limit arrives, an agent stops without committing. **Tell it every time: commit as you
  approach the limit.**
- Start in the order "send the script → check the md5 → launch → check the process". Chaining several stdin
  payloads into one ssh leaves the second empty.
- A background launch can die instantly without being noticed. Always write the exit code to a log.

### 6.4 Contaminating the measurement environment

- A resident process on the GPU (screen streaming, another python, etc.) pins utilisation at 38–47% and
  contaminates the numbers. Take bench runs in a window where utilisation is 0–5%.
- Do not run other builds or another bench during a measurement (check load average and available memory
  first).
- The practice is to look at uptime and free memory immediately before measuring. When things are suddenly
  slow, suspect that first.

---

## 7. The acceptance procedure (order matters)

If an earlier step fails, do not measure the next one. The order is kept so that the attribution of numbers
does not break down.

1. **Build + CPU gate**: confirm the target commit is in HEAD and no uncommitted implementation is mixed in,
   then `cargo build --release` and `cargo test --release --lib`.
2. **GPU_VERIFY**: anchors 3/3 MATCH, `RESULT: PASS`, health ok. Confirm afterwards that it was restarted with
   the production arguments.
3. **Real-machine decode**: discard the warm-up, take the median. Must not be worse than the baseline.
4. **Real-machine prefill**: cold, nonce, median. Respect the character-count argument (measuring with a small
   argument makes it incomparable to the baseline).
5. **Bit-identity**: the wide comparison.
6. **Adoption**: if a number changes, always record the old md5 → the new md5. A commit that touches a `.cu`
   requires regenerating the PTX (`nvcc -ptx -arch=compute_75 -fmad=false -O3`).

**What must not be done**:

- Committing without measuring the anchors.
- Loosening a test or a tolerance in order to pass verification (a change on the test side goes in a separate
  commit with the reason written down).
- Running acceptance measurements on a tree with uncommitted sources mixed in (the numbers stop having an
  owner).
- Re-baselining the anchors for speed or refactoring purposes (re-baselining only when a numeric relaxation is
  adopted).

---

## 8. Rejected approaches in this volume (what was dropped on the operations and measurement side)

- **Making the second run of the same prompt the official prefill value** — it hits the prefix cache. Re-measured
  cold, −9.7% was −5.7%. Afterwards, single cold runs plus alternating A/B became the official procedure.
- **Talking about speed with the absolute values of `DENNO_PROFILE`** — the measurement itself inflates by
  2.7–4x. Use ratios only.
- **Putting the phase percentages of `DENNO_PF_DBG` into a KPI table** — they are a partition of wall time, not
  of kernel time.
- **Looking inside kernels with ncu** — refused for permissions. Lifting it needs the container recreated
  (server down).
- **Stopping the server with `pgrep -f`** — matches your own command line and kills you (exit 143). Take the
  pid first.
- **Committing everything with `git add -A`** — pulled in 43,249 files and required rewriting history.
- **Starting a new task on a stale base of the shared tree** — causes rework at integration. Check the base
  before starting.
