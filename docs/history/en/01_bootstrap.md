# Volume 1 Bootstrap — from a single binary to CI

> [Index](README.md) / Next: [Volume 2 API features](10_api_features.md)
> Japanese original: [../01_bootstrap.md](../01_bootstrap.md)

Period covered: 2026-10-11 (initial implementation) through the completion of P2. This is the oldest stretch of the work in scope, with neither GPU nor quantization yet.
Dates are commit author dates in JST.

## What this volume covers

- The single-binary layout (cli / engine / server), and why Anthropic Messages API compatibility was chosen first
- The consequences of making exact decoding (fixed seed, greedy, same input → same output) the baseline from the start
- The 2 stages of the KV cache and the prefix cache (P1 = single entry → P2 = multi-entry LRU)
- Standing up Docker deployment and CI, and why the build target CPU was pinned to x86-64-v3

## Background

Rust / Cargo workspace, the transformer forward pass and KV cache, the Anthropic Messages API and SSE, Docker multi-stage builds.
Quantization, CUDA, and GGUF do not appear in this volume (they start in volume 3).

## Reading order

Read in the order: initial implementation → first contact with a real model → first speed work → P1 (KV/prefix cache) → Docker → P2 and CI → CI fix.
If you only want the summary of the numbers, the table in `docs/KPI.md` is enough. This volume is that table with each row opened up, intent and stumbles included.

## Phases covered

| phase | date | commit | one-line summary |
|---|---|---|---|
| Initial implementation | 2026-10-11 | `7f85369` | the skeleton of a single binary (cli / engine / server) with Anthropic Messages API compatibility |
| Real-model bring-up | 2026-10-11 | `b718e59` | fixed the GEMM stride and the QKV bias of the Qwen2 family; GPT-2 / Qwen2.5-0.5B pass |
| First speed work | 2026-10-11 | `67280c9` | attention moved to sgemm, `target-cpu=native`, SSE event names and ping |
| P1 | 2026-10-11 | `831246b` | reuse of the KV cache across requests and a single-entry prefix cache |
| Docker deployment | 2026-10-11 | `f1c3bde` | multi-stage build and automatic model fetch at startup |
| P2 / CI | 2026-10-11 | `6faf889` | 48 tests, multi-entry LRU prefix cache, CI on push |
| CI fix | 2026-10-11 | `37cc6b0` | pinned `target-cpu` to `x86-64-v3` in response to SIGILL |

---

### Initial implementation: a single binary and the Anthropic-compatible API (2026-10-11, `7f85369`)

- **Intent**: We wanted a "readable" LLM server for learning purposes, self-contained in 1 repository. Since we were not bringing in an existing inference engine and were going to write the forward pass and the sampler ourselves, we narrowed the target architectures to 2 lines, GPT-2 and the Qwen2/Llama family. The client we used at the time spoke the Anthropic Messages API, so we matched the compatible side, and to drive the same engine from both the CLI and the server we split the code into 3 crates (cli / engine / server).

- **What changed**
  - `7f85369` — 22 files / +4,276 lines.
  - `crates/denno-engine/src/{load,tensor,tokenizer,sampler}.rs`, `model/{gpt2,llama,mod}.rs`.
  - `crates/denno-server/src/lib.rs`, `docs/PLAN.md`, `docs/api.md`.

- **Stumble**
  - Symptom: running a real model produced output with no meaning. Streaming was also not interpreted correctly by the client.
  - Cause: the first commit only went as far as "the shape works", and there was no means of judging numeric correctness. Because we fixed the API shape first, getting a real model through and the differences in the SSE spec were left unresolved.
  - How we got out: the numeric side was closed in the next commit, the SSE side 2 commits later, separately (the rest of this volume).

- **What we learned / remaining issues**
  - Splitting the engine into a crate lets the same forward pass be driven from both the CLI and the server, which makes verification easy to run.
  - On the other hand there was no standard for "correctness", and the only way to check whether the output matched was to eyeball it every time.

- **What worked next**: the structure "the same engine can be driven directly from the CLI" became the foundation for the later exactness tests (running a synthetic model with no real model) and for bit comparison against the CPU reference.

---

### Real-model bring-up — GEMM stride and QKV bias (2026-10-11, `b718e59`)

- **Intent**: The initial implementation was shape only, and running a real model did not produce matching numbers. We first set the goal of getting the 2 models, GPT-2 and Qwen2.5-0.5B, through end to end.

- **What changed**
  - `b718e59` — +27 / -3.
  - `crates/denno-engine/src/load.rs` — fixed the stride interpretation in GEMM.
  - `crates/denno-engine/src/model/llama.rs` — added the QKV bias of the Qwen2 family.

- **Stumble**
  - Symptom: real-model output had no meaning. Neither a type check nor a panic fired.
  - Cause: (1) the stride interpretation of the weights disagreed with the transpose direction, (2) the Qwen2 family carries a bias on QKV, but the implementation assumed no bias.
  - How we got out: we ran GPT-2 and Qwen2.5-0.5B side by side. With only one of them, the missing bias does not surface on GPT-2.

- **What we learned / remaining issues**
  - Architecture differences (presence of bias, tied/untied) stay invisible until 2 lines of real models are run.
  - From then on, "when a new architecture is added, run it E2E with the smallest model" became the default.

- **What worked next**: the 9B untied lm_head mix-up and the 27B MTP layer contamination (volume 3) were all defects of the same shape: "the shape works but the values are wrong". The habit of "running 2 lines" from this point carried straight over.

---

### First speed work and client compatibility — sgemm attention / native target / SSE (2026-10-11, `67280c9`)

- **Intent**: With a real model running, the next stage was to reach usable speed. Prefill and decode at the time were scalar loops, and waiting for 1 request was painful. At the same time we had to close the problem of clients not being able to connect (SSE).

- **What changed**
  - `67280c9` — +162 / -27.
  - attention from a scalar loop to sgemm (`model/llama.rs`), `target-cpu=native` in `.cargo/config.toml`.
  - SSE event names and ping keep-alive in `crates/denno-server/src/lib.rs`, addition of `examples/bench_gemm.rs`.

- **Stumble**
  - Symptom: the client does not interpret the stream, or it cuts off partway.
  - Cause: the SSE event names did not match the Anthropic format, and there was no ping to fill the idle time.
  - How we got out: we matched the event names to the spec and added the ping keep-alive. The commit subject says "works with Hermes agent", and this is where it first connected to a real client.

- **What we learned / remaining issues**
  - The measured speed survives only as "prefill 88 s → 24 s, decode 1 s/token → 0.1 s" in the summary table of HISTORY §2. The model, prompt length, and thread count are not stated, so it cannot be used as a conditional comparison.
  - `target-cpu=native` was the optimal answer at this point, but it later became a cause of CI failures (the end of this volume).

- **What worked next**: the constraint that "it got faster" can only be claimed conditionally became the starting point of the later measurement discipline (interleaved A/B, single cold run, anchor md5).

---

### P1: KV cache and a single-entry prefix cache (2026-10-11, `831246b`)

- **Intent**: Rebuilding the same system prompt and history every turn was the biggest waste. The goal was to keep the KV across requests and make the next prefill nearly 0.

- **What changed**
  - `831246b` — +76 / -12.
  - KV reuse across requests in `engine.rs`, addition of `kv_cache.rs`.
  - output-split parallelism for the decode matvec in `tensor.rs`, and the corresponding change in `model/llama.rs`.

- **Stumble**
  - Symptom: the 2nd run of the same conversation is fast, but the moment a different conversation arrives it is back to the original.
  - Cause: the prefix cache was "single entry", so a new prompt discarded the previous entry.
  - How we got out: this limitation was left in place until P2 turned it into a multi-entry LRU. The parallelization of the decode matvec splits output elements and does not change the accumulation order, so the numbers are unchanged.

- **What we learned / remaining issues**
  - The 2nd and later prefills dropped to the equivalent of 1,519 tokens → 11 tokens (summary table of HISTORY §2; the model and prompt are not stated).
  - The remaining issue was that "a single entry does not work for alternating sessions".

- **What worked next**: the policy "if the KV does not have to be discarded, do not discard it" continued consistently into the later KV budget design (demand-based allocation, slot pool) and into how state is preserved for recurrent models (the GDN snapshot in volume 3).

---

### Docker deployment — multi-stage build and automatic model fetch (2026-10-11, `f1c3bde`)

- **Intent**: We wanted to stop carrying around a locally built binary and make the server come up with `docker compose up`. Model fetching was also pulled to startup.

- **What changed**
  - `f1c3bde` — +73 / -0.
  - `Dockerfile` (multi-stage build), `docker-compose.yml`.
  - `docker/entrypoint.sh` (automatic model fetch at startup).

- **Stumble**
  - Symptom: every model change forces the image to be rebuilt.
  - Cause: build artifacts and the model cache were in the same image layer.
  - How we got out: separated the build stage from the runtime stage and moved the model to a volume. The entrypoint checks whether the model is already fetched and skips the download.

- **What we learned / remaining issues**
  - Once deployment is 1 command, "we shipped it to production" is fixed to mean "we rebuilt the image and restarted it".
  - At this point deployment and automated tests were still not connected.

- **What worked next**: the idea of confining settings to a startup script carried straight over into the later production startup script (the form that pins quantization, KV budget, and the temperature cap with flags).

---

### The P1 test suite, the P2 multi-entry LRU prefix cache, and CI (2026-10-11, `6faf889`)

- **Intent**: The effect of P1 could be measured, but there was no means of noticing when it broke. We stacked tests up to 48, and at the same time widened the prefix cache from a single entry to a multi-entry LRU and ran it in CI on push.

- **What changed**
  - `6faf889` — 14 files / +1,613 / -17.
  - `prefix_cache.rs` (new, 295 lines), `kv_cache.rs` (+210), `engine.rs` (+389), `tensor.rs` (+250).
  - `testutil.rs` (new, 83 lines), `tests/exactness.rs` (new, 123 lines), `.github/workflows/ci.yml`.

- **Stumble**
  - Symptom: with multiple entries, a rule is needed for "which entry to use". The tests cannot run exactness in an environment without a real model.
  - Cause: there was no selection rule, and CI has no model cache.
  - How we got out: choose the entry with the largest longest common prefix (LCP), with LRU eviction under a 32k token budget. On the test side we placed a synthetic model in `testutil.rs` and separated out a layer that verifies determinism only.

- **What we learned / remaining issues**
  - Running 2 sessions alternately now stays effective up to `reuse 47/48` (summary table of HISTORY §2).
  - The exactness reference output is "1+1は?" → `1+1は1です。これは数学的定理で、1と1を加えることによって得られます。` (seed 42, temperature 0, fp32). The same output was confirmed on the CI x86 runner.

- **What worked next**: the 2-tier arrangement of "a layer that looks only at determinism with no real model" and "a layer that looks at string equality with a real model" became the pattern for every regression test afterwards. This reference string later also turned into the side that exposed the streaming detok bug (volume 2).

---

### CI SIGILL fix — pinning the build target to x86-64-v3 (2026-10-11, `37cc6b0`)

- **Intent**: CI started failing depending on the runner. We wanted to close the cause and get back to the same verdict on every push.

- **What changed**
  - `37cc6b0` — +11 / -0.
  - `.github/workflows/ci.yml` only. Pinned `target-cpu` from `native` to `x86-64-v3` and updated the cache key.

- **Stumble**
  - Symptom: CI dies with SIGILL (illegal instruction).
  - Cause: `rust-cache` shares build artifacts across runner CPU generations, so a binary built for a newer CPU is executed on an older runner.
  - How we got out: pin the build target to the lowest common denominator instruction set — dropped `native` for `x86-64-v3`.

- **What we learned / remaining issues**
  - The cache is fast, but it implicitly fixes the CPU features. Using `native` on the dev machine and getting reproducibility in CI do not coexist.
  - From then on, "all green on x86-64-v3 in CI" became the premise for pass/fail.

- **What worked next**: the recognition that "same input, same output" also has to be guaranteed on the machine side led to running the exactness job. The later decision to keep anchor md5 values separately for GPU and CPU is an extension of this.

---

## Rejected approaches

Decisions that looked sound at the time but were dropped later. The reasons for rejection are written with the facts observable at that time.

### Keeping `target-cpu=native` as the CI build target (rejected in `37cc6b0`)
- Why adopted: using the dev machine's instruction set as-is is fast. At the time of this volume it was the optimal answer.
- Observation: CI dies with SIGILL (illegal instruction). Because `rust-cache` shares build artifacts across runner CPU generations, a binary for a newer CPU is executed on an older runner.
- Rejected: pinned the build target to the lowest common denominator, `x86-64-v3`. We learned that `native` on the dev machine and reproducibility in CI do not coexist.

### Leaving the prefix cache at a single entry (rejected in `6faf889`)
- Why adopted: if the KV can be kept across requests, the 2nd prefill is nearly 0. We judged that was enough to start with.
- Observation: the 2nd run of the same conversation is fast, but the moment a different conversation arrives it is back to the original. Because the previous entry is discarded, it does not work for alternating sessions.
- Rejected: moved to a multi-entry LRU in P2 (choose the largest LCP, evict under a 32k token budget). Reached `reuse 47/48` with 2 alternating sessions.

### Reporting speedups without stating the conditions (rejected with `67280c9` and its record)
- The form in use: leaving only the numbers "prefill 88 s → 24 s" and "decode 1 s/token → 0.1 s".
- Observation: the model, prompt length, and thread count are not stated, so it cannot be used as a conditional comparison. The same conditions cannot be reproduced afterwards.
- Rejected: moved to a discipline that always attaches interleaved A/B, a single cold run, and anchor md5 as conditions.

### Judging output correctness by eye (rejected at the time of the initial implementation)
- The form in use: judging correctness by whether the text reads as language when a real model is run.
- Observation: against output with no meaning, we could not say what was different. Because neither a type check nor a panic fires, the defect does not surface.
- Rejected: moved to the 2-tier arrangement of a layer that looks only at determinism with a synthetic model (`testutil.rs`) and a layer that looks at matching the reference string with a real model (exactness).
