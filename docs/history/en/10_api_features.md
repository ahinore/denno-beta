# Volume 2 API features — tool use, thinking, stop conditions, disconnect detection

> [Index](README.md) / Prev: [Volume 1 Bootstrap](01_bootstrap.md) / Next: [Volume 3 Model support](20_models.md)
> Japanese original: [../10_api_features.md](../10_api_features.md)

Period covered: 2026-10-12 through 2026-10-19. A record of the features that went into the server layer (`denno-server`) and the engine's text post-processing.
Model-structure topics are in volume 3, and speed and kernel topics are split off into separate volumes.
Dates are commit author dates in JST.

## What this volume covers

- The design that added tool use / tool_choice while keeping Anthropic Messages API compatibility (prompt conversion, extraction, constrained decoding)
- SSE assembly (streaming tool_use), and the limits of how far disconnect detection and aborting can go
- How thinking control changed 3 times — "string → map accepted → the meaning of the default changed" — and why
- How to add stop conditions (role-label stop, holdback) and 3 kinds of repetition suppression while keeping the default bits unchanged

## Background

The request/response shapes of the Anthropic Messages API (content blocks, `stop_reason`, the SSE event sequence), token boundaries in byte-level BPE,
prompt rendering with a chat template (Jinja). Every change in this volume is either "post-processing of the text stream" or "masking of logits", and none of them touches the numerics of the forward pass.

## Reading order

In the order: P11 (streaming detok) → tool use → tool_choice constraint → streaming tool_use → disconnect detection → max_tokens default →
non-stream disconnect and logging → thinking (s47 → s52 → s53c) → stop conditions → repetition suppression → temperature cap.
Read the 3 thinking entries strictly in order (the later ones change the meaning of the earlier ones).

## Phases covered

| phase | date | commit | one-line summary |
|---|---|---|---|
| P11 | 2026-10-12 | `1a7fee6` | hold-back of U+FFFD in streaming detok and an update of the gpt2 exactness criterion |
| tool use | 2026-10-16 | `751dc10` | accepting tools / tool_choice, chat template conversion, `<tool_call>` extraction |
| tool_choice 拘束 | 2026-10-16 | `a16b203` | forcing `any` / `tool` at token level with a logits mask |
| streaming tool_use | 2026-10-16 | `05e5f8f` | an SSE assembler that assembles `<tool_call>` across delta boundaries |
| 切断検知 | 2026-10-16 | `e7d3e7a` | watching SSE `is_closed()` at step boundaries and aborting generation |
| max_tokens 既定 | 2026-10-16 | `9636337` | applying 4096 only when unspecified (`max_completion_tokens` is accepted too) |
| s44 非ストリーム切断 | 2026-10-17 | `08e7c9b` | the closed check in respond, and non-blocking stdout logging |
| s47 thinking 制御 | 2026-10-17 | `4e89c11` | accepting `thinking: off / brief / full` and passing it to template variables |
| s52 thinking マップ | 2026-10-18 | `b9d29f8` | accepting the `{"type":"enabled"}` form (`budget_tokens` is not read) |
| s53c brief の意味変更 | 2026-10-18 | `e7246b8` | brief becomes "thinking disabled + concise instruction", map `enabled` treated as off |
| s53 停止条件 | 2026-10-19 | `f21a985` | holdback for the role-label stop and a default max_tokens of 1024 |
| s56/s57 反復抑止 | 2026-10-19 | `b3eaf71` | no_repeat_ngram / frequency / presence added as opt-in |
| temperature 上限 | 2026-10-19 | `47e1928` | clamping the effective temperature with `--temperature-cap` |

---

### P11: fixing the byte-split characters in streaming detok and updating the gpt2 exactness criterion (2026-10-12, `1a7fee6`)

- **Intent**: The gpt2 exactness test was failing on mojibake, and re-fetching the model did not fix it. We needed to isolate the cause and also bring the test criterion itself up to date.

- **What changed**
  - `1a7fee6` — hold-back handling in streaming detok (`emit_delta_and_check_stop`) and the new `flush_stream`.
  - Added gpt2 model fetching to the CI workflow (the exactness job).

- **Stumble**
  - Symptom: the output was mojibake, like "??の??は?". The model file was identical (548,105,171 B), and raw English generation was normal.
  - Cause: two of them. (1) The exactness job in CI only fetched Qwen, so the gpt2 test never actually ran — it was always "model not cached → skip" (the reference string had been left as it was at the time). (2) In byte-level BPE a multi-byte character is split at token boundaries, and an incomplete byte sequence becomes U+FFFD. Mixing that U+FFFD into already-committed text shifts the byte diff afterwards and characters go missing.
  - Way out: hold back tokens whose tail becomes U+FFFD, and at the step where the character completes, emit them together as a diff. At the end of generation (EOS / max_tokens) `flush_stream` emits what was held back. We cross-checked against a reference implementation (HF transformers, fp32, greedy) and confirmed that denno's output matched exactly.

- **What we learned / remaining issues**
  - 99 + 7 unit tests and 5 exactness cases were green. The gpt2 test actually ran in CI and went green (a demonstration that the reference string matches the reference implementation).
  - The numeric path is unchanged (text path only). We confirmed this with the reference output and with a measured 4 concurrent sessions == solo.

- **What worked next**: the holdback idea of "hold it, commit it, then emit" was reused as-is for the later role-label stop (late in this volume) and for boundary tracking in streaming tool_use.

---

### tool use support — tools / tool_choice compatible with the Anthropic Messages API (2026-10-16, `751dc10`)

- **Intent**: Clients were built on the assumption of tool use, and a 400 when sending `tools` was a blocker in real operation. We took the policy of completing this in the API layer only and not touching the model's numeric path.

- **What changed**
  - `751dc10` — `crates/denno-server/src/lib.rs` (parsing, validation, extraction, response assembly), `tokenizer.rs` (a tools argument for `apply_chat_template`), `engine.rs` (`begin_chat_with_tools`).
  - Added a tool use section to `docs/api.md`.

- **Stumble**
  - Symptom: requests with `tools` returned 400. There was also no means of pulling a tool call out of the model output.
  - Cause: accepting the request side, passing the `tools` variable to the chat template, and structured extraction from the output were all unimplemented.
  - Way out: convert validated tools into the Qwen/HF form (`{"type":"function","function":{...}}`) and pass them to the template's `tools` variable. On the history side, convert assistant `tool_use` blocks into `<tool_call>` XML conforming to the official template, and user `tool_result` into `<tool_response>`. On the response side, extract only complete `<tool_call>` blocks and remove **only the blocks that were extracted successfully** from the text (a parse failure or an unfinished block stays as text — a defence for the case where the model outputs a format example).

- **What we learned / remaining issues**
  - With real-machine curl we confirmed generation with tools (in=277) → a 2nd turn returning tool_result (in=332) → `tool_choice: none` producing the same output as no tools. Determinism over 2 consecutive runs on the tools path also matched.
  - The no-tools path passes null to the template, so rendering is bit-identical. Remaining issues were streaming (tools + stream rejected with 400) and forcing `any`/`tool`.

- **What worked next**: the rule "remove only the blocks that were extracted successfully" became the criterion for the later semantics of streaming tool_use (an unfinished block is text).

---

### tool_choice constrained decoding — a token-level constraint on `any` / `tool` (2026-10-16, `a16b203`)

- **Intent**: We accepted `tool_choice` but did not make it take effect (`any`/`tool` behaved the same as auto). To serve the use case of making the model pick a tool name reliably, we force it with a logits mask immediately before sampling.

- **What changed**
  - `a16b203` — new `crates/denno-engine/src/toolmask.rs` (`ToolConstraint` / `ToolMask` state machine).
  - `engine.rs` (`ChatSession.tool_mask`, `apply` immediately before sample and `feed` after sample), `tokenizer.rs` (`piece_bytes`), `server/lib.rs` and `scheduler.rs` (wiring the constraint).

- **Stumble**
  - Symptom: with the mask in place, EOS came out in the middle of a block and broke it / the allowed set became empty and all probabilities went to 0.
  - Cause: the handling of EOS and the degradation when the required spelling does not exist in the vocabulary were undefined. Also, if the state machine cannot advance on the sampled token, it becomes an infinite loop.
  - Way out: while the constraint is unsatisfied, remove EOS from the allowed set and allow it again once CLOSE completes. If the allowed set is empty, warn and drop the mask. If `feed` cannot make a transition, drop the state to Done and release the constraint (preventing a dead path).

- **What we learned / remaining issues**
  - The states are Watch → Head (forcing `\n<function=` one byte at a time) → Name (only prefixes of allowed names) → Body (free, watching for CLOSE) → Done (no mask). The allowed set is recomputed and cached only when the state changes.
  - For a conforming script, with and without the constraint are bit-identical. The unconstrained path has no mask, so it is bit-identical to before. Verification of `any`/`tool` on a real model was not performed (a remaining issue at the time).

- **What worked next**: the design of forcing with "nothing but -inf substitution on logits" was aligned with no_repeat_ngram in repetition suppression (late in this volume) on the same application point and the same ordering discipline.

---

### streaming tool_use support — the SSE assembler (2026-10-16, `05e5f8f`)

- **Intent**: Because tools + stream was rejected with 400, clients could not use tool conversations in streaming mode. We port the same semantics as the non-streaming `extract_tool_calls` to the delta stream.

- **What changed**
  - `05e5f8f` — new `crates/denno-server/src/stream_asm.rs` (`StreamAssembler`).
  - `scheduler.rs` (`Run.asm`), `lib.rs` (removed the 400 for tools + stream), `engine.rs` (`on_deltas` becomes `Box<dyn FnMut(&str) + '_>`).

- **Stumble**
  - Symptom: the OPEN/CLOSE of `<tool_call>` was split at delta boundaries and the leading part leaked out as text. A borrow error (`E0505`) prevented compilation.
  - Cause: taking the assembler out of Run and passing it to the callback collides with the `'static` requirement and with the move. The leading part of an OPEN that arrives split was emitted as committed text.
  - Way out: track it in the `scan` buffer and do not emit committed text until the tail can no longer be a prefix of OPEN. We closed the borrow region of the callback inside a block and separated it from the move that returns `asms` to Run. We also matched the trim convention to the non-streaming one, and fixed in a unit test that for any split width it matches the remaining text of `extract_tool_calls(full text)`.

- **What we learned / remaining issues**
  - 10 unit tests for `stream_asm.rs` were green (all splits of chunks 1-100, all byte boundaries of OPEN/CLOSE, hold of a partial OPEN prefix, ascending index for multiple blocks). We replaced `tools_with_stream_is_rejected` with `tools_with_stream_is_accepted` and verified SSE integrity. workspace 215 passed.
  - Remaining issue: confirmation of tool-call streaming on a real model (at the time, verification went as far as a synthetic vocabulary).

- **What worked next**: the shape "hold the assembler in Run and take it in and out at step boundaries" landed in the same place as the step-boundary sweep of disconnect detection. The structure in which the scheduler's Run is where conversation state lives was fixed here.

---

### Client disconnect detection and generation abort (SSE stream) (2026-10-16, `e7d3e7a`)

- **Intent**: Even when a client abandoned the connection the server kept generating, and a measured generation of over 13 minutes occupied the single worker so that every subsequent request queued. We detect disconnects and abort, for SSE only.

- **What changed**
  - `e7d3e7a` — `scheduler.rs` (`Run::client_gone`, the disconnect sweep, factoring out into `admit_job`).
  - New `crates/denno-server/tests/client_disconnect.rs` (reproducing a disconnect with a real TCP socket).

- **Stumble**
  - Symptom: after a disconnect `in_flight` stayed at 1 and did not decrease. In phases where not a single delta is sent (during prefill, in the queue) it can never be detected.
  - Cause: we were swallowing the send failure inside the delta callback (`let _ = blocking_send`), and that was the only place with a chance to look at connection state.
  - Way out: the SSE response body is a `ReceiverStream`, so when the client disconnects, hyper detects a broken pipe while writing SSE events / pings and drops the body, which makes the channel's `Sender::is_closed()` true. The worker checks this closure at **each step boundary** of decode / prefill, removes the Run in question and calls `release_session`.

- **What we learned / remaining issues**
  - 4 tests fix "disconnect during generation", "disconnect before the first token", "receiver-side drop first", and "the normal case". We confirmed that the first 2 FAIL on the code before the fix. workspace 220 passed.
  - Limit: abort is per step boundary, so it does not work in the middle of one forward (a prefill of thousands of tokens). Non-stream was excluded at the time on the grounds that "there is no chance to observe connection state".

- **What worked next**: the question "couldn't non-stream ride the same sweep?" led to the next entry (s44), where that premise gets corrected.

---

### Default max_tokens upper bound for chat, applied only when unspecified (2026-10-16, `9636337`)

- **Intent**: A thinking model kept generating for a very long time, making the single-worker server keep subsequent requests waiting. Guarantee an upper bound on the request side before generation starts.

- **What changed**
  - `9636337` — `crates/denno-server/src/lib.rs` (`DEFAULT_MAX_TOKENS = 4096`, `resolve_max_tokens`, `MessagesRequest.max_tokens` becomes `Option<usize>`).
  - New `crates/denno-server/tests/max_tokens_default.rs`.

- **Stumble**
  - Symptom: omitting `max_tokens` returned 400 (missing required field). Including it did not make the limit effective.
  - Cause: in the Anthropic spec `max_tokens` is required, and the implementation read it as required too. The concept of a default did not exist.
  - Way out: make `max_tokens` optional and apply 4096 only when unspecified. Accept the OpenAI-compatible alias `max_completion_tokens` too, and prefer `max_tokens` when both are present. Respect an explicit value (including 0). The application point is the single place in the handler where `GenParams` is built, so it applies equally to both the non-stream and SSE paths.

- **What we learned / remaining issues**
  - The basis for the default 4096 is the same value as the engine-side KV reservation per session, which lines up with the premise of the KV budget design.
  - 7 tests (directly observing `usage.output_tokens` on a model that does not emit EOS) were green. workspace 227 passed.

- **What worked next**: the rule "default when unspecified, respect when explicit" was maintained as-is in the later change to a default of 1024 (s53). Leaving it in a shape where only the default value could be swapped paid off.

---

### Hardening the post-disconnect handling of non-stream /v1/messages (s44) (2026-10-17, `08e7c9b`)

- **Intent**: After a non-stream disconnect the completion log was missing, and there were cases where the server stopped responding to new connections (wedge) once the log consumer died. We kill these as 2 independent problems.

- **What changed**
  - `08e7c9b` — `scheduler.rs` (added the respond closed check to `Run::client_gone`).
  - New `crates/denno-server/src/logging.rs` (`NonBlockingStdout`, a bounded queue of 4096 lines), `tests/nonstream_disconnect.rs`, `tests/frozen_log_robustness.rs`.

- **Stumble**
  - Symptom 1: a non-stream disconnect was not detected and kept blocking the single worker.
  - Symptom 2: when the log consumer (the far end of the pipe) froze, the whole server went to CPU 0% and `/health` stopped responding.
  - Cause: (1) the s38 premise that "hyper does not observe the connection until the response is sent" was wrong; in reality it watches the read side while the response future is pending, and on disconnect the handler future is dropped along with it. (2) Synchronous stdout logging blocked forever in `write()` once the pipe buffer (64 KB) filled up, and stopped every thread while holding the global lock of `std::io::stdout()`.
  - Way out: (1) use `respond.is_closed()` on the oneshot held by the handler as an indirect observation of the disconnect, and put it on the same sweep as SSE. (2) Put log lines into a bounded queue and have only 1 dedicated thread write to stdout. On overflow, discard and count, and report it in 1 line after recovery.

- **What we learned / remaining issues**
  - 6 tests were green (verified that `/health` and subsequent requests complete even under a frozen sink, and that log lines were handled by "discard rather than block"). workspace 237 passed.
  - The lesson that remained is that "logging has to be best effort". A logging path in an inference server can become the bottleneck.

- **What worked next**: after this, a separate accident of the "log flood breaks production" type happened with measurement output (prof) as well. It led to the discipline of always considering an upper bound and a discard rule together when adding an output point.

---

### thinking control — implementing `thinking`: `off` | `brief` | `full` (s47) (2026-10-17, `4e89c11`)

- **Intent**: We wanted to control the amount of thinking of a thinking model from the request side. It is an extension that the Anthropic API does not have, but since the client could already send its own fields, we accept it as an optional top-level field.

- **What changed**
  - `4e89c11` — `tokenizer.rs` (`enable_thinking: Option<bool>` on the `apply_chat_template` family), `engine.rs` (the same argument on `begin_chat_with_tools`), `server/lib.rs` (the `Thinking` enum, `parse_thinking`, `brief_system`), `scheduler.rs` (propagation to `Job`).
  - Added `ThinkingSimModel` to `testutil.rs`, and a template with the same `enable_thinking` branch as the real model.

- **Stumble**
  - Symptom: thinking starts even with off / the behaviour when the template variable is undefined is unreadable.
  - Cause: the real template refers to it in the form `{% if enable_thinking is defined and enable_thinking is false %}`, so not passing the variable results in "thinking ON". We needed to distinguish the presence of the variable from its truth value.
  - Way out: pass it as a ctx variable only when specified (None = no variable = identical to the previous path). The generation prompt with off becomes an empty think block, and, like the real model's chat_template, there are 3 newlines just before the end tag (confirmed by expanding the real file with python3).

- **What we learned / remaining issues**
  - We fixed in tests that off = full + 2 extra tokens for `<think>` and `</think>`. brief at the time was soft control — "keep thinking ON and inject a short-answer instruction at the start of system" — and we stated explicitly that it was not hard control.
  - 11 tests were green, workspace 253 passed. Confirming the effect of brief on a real machine was a remaining issue.

- **What worked next**: the design "not passing the variable = bit-identical to the previous path" took hold as the pattern for adding features without breaking existing anchors. The later s52 / s53c also swap only the meaning on top of this pattern.

---

### Accepting the thinking map form — taking the Anthropic thinking map for client compatibility (s52) (2026-10-18, `b9d29f8`)

- **Intent**: Because the client sends thinking in map form `{"type":"enabled","budget_tokens":8000}`, the s47 parser that accepted only strings returned 400 and E2E did not pass. thinking alone was the blocker.

- **What changed**
  - `b9d29f8` — `server/lib.rs` (`MessagesRequest.thinking` becomes `Option<serde_json::Value>`, `parse_thinking` looks at the whole value).
  - A thinking section in `docs/api.md` (both forms, the ignored `budget_tokens`, the conditions for 400).

- **Stumble**
  - Symptom: 400 with the map form. The string form passes.
  - Cause: the parser's input type was `Option<String>`, and the object was dropped at the serde stage.
  - Way out: change to `Option<Value>`, keep String as before, branch on `type` for Object (`enabled` → brief, `disabled` → off, unknown → 400), and return 400 for other types as well (numbers, booleans, arrays). `budget_tokens` is not read (to keep it deterministic).

- **What we learned / remaining issues**
  - The caller side is unchanged with a compatible signature, and the subsequent brief injection and the `enable_thinking` path are completely identical to s47. 3 tests + 1 existing test updated, workspace 256 passed.
  - Remaining issue: re-confirmation of the client E2E on a real machine (at the time, as far as the wire test on the CPU path).

- **What worked next**: the policy "accept widely, interpret deterministically" (do not reject unknown fields, do not read fields that are not used for interpretation) was carried over to the validation of the repetition suppression parameters later.

---

### Changing the meaning of thinking=brief — disabling thinking and treating map enabled as off (s53c) (2026-10-18, `e7246b8`)

- **Intent**: In measurement, the 9B kept writing meta-analysis under "Thinking Process:" endlessly and never reached an answer. Because brief was "thinking ON + instruction", the instruction was not working.

- **What changed**
  - `e7246b8` — `server/lib.rs` (brief also becomes `enable_thinking=Some(false)`, `THINKING_BRIEF_INSTRUCTION` becomes "回答は簡潔に、数文でまとめてください。", map `enabled` → `Off`).
  - `tests/thinking_control.rs` and `docs/api.md` followed.

- **Stumble**
  - Symptom: even with brief (the default) long thinking ran and no answer came back.
  - Measured: in a comparison of thinking ON + instruction, **all 4 candidates** continued English meta-analysis under "Thinking Process:" and never reached an answer (max_tokens exhausted). Meanwhile off + a concise instruction completed in 2.0 s. We determined that **the instruction alone does not fix it**.
  - Cause: because brief was soft control (the instruction only), it did not actually constrain the model's generation.
  - Way out: change brief to "thinking disabled + concise instruction" and treat the map `enabled` as off as well. In other words, the default became effectively "do not think".

- **What we learned / remaining issues**
  - Because the meaning of the default changes, we updated the reference side of the existing wire tests (the engine-direct execution path has no system, so it is unaffected).
  - As a side effect, the operation became "thinking does not run unless `full` is sent explicitly". This later combined with the temperature problem (late in this volume) and produced another symptom.

- **What worked next**: the procedure "when changing a default, update the reference side of every test that depends on that default at the same time" took hold. The change here became part of the grounds for the temperature cap decision.

---

### Strengthening stop conditions — role-label stop and a default max_tokens of 1024 (s53) (2026-10-19, `f21a985`)

- **Intent**: When the next speaker header appears in the generated text, the model goes on writing another role's sentences as-is (a fake turn). We add a stop equivalent to llama.cpp's reverse prompt. At the same time we lower the default generation length from 4096 to 1024.

- **What changed**
  - `f21a985` — `crates/denno-engine/src/engine.rs` (`sent_len`, `StopStatus`, `find_stop_marker`, `line_label_at`, `tail_is_line_end`, `partial_holdback`, holdback release in `flush_stream`).
  - `server/lib.rs` (`DEFAULT_MAX_TOKENS` to 1024, `MAX_TOKENS_CLAMP = 4096` clamping an explicit value at the upper bound).

- **Stumble**
  - Symptom: intermediate strings like `u` / `us` / `use` leak into the SSE delta. It stops wrongly on a word like `userland`.
  - Cause: because the marker crosses token boundaries, we emit the string before it is committed. The line-label check and the prefix check did not agree on how they look at word boundaries.
  - Way out: add a holdback that does not emit while the tail could be a prefix of the marker, and make `sent_len` (bytes emitted so far) an invariant. At the end of generation `sent_len == text.len()` always holds, so the final text of a normal response is identical to before. `userland` is held back and then treated as a non-stop.

- **What we learned / remaining issues**
  - The longest marker byte length is `MAX_STOP_MARKER_LEN = 12` (`<|im_start|>`). It is also used for the holdback upper-bound calculation. It is combined with the client-specified `stop_sequences` by "the earliest stop position in the text".
  - Known limit: the prefix path does not look at word boundaries, so `userland` after a newline can still stop (after the holdback). We accepted this as the safe-side behaviour. 5 tests added, workspace 261 passed.

- **What worked next**: the design "hold the emitted boundary as a single integer" is the same idea as the later state management of disconnect detection and the SSE assembler, and became a model for fixing the invariants of the text path in tests.

---

### 3 opt-in kinds of repetition suppression — no_repeat_ngram / frequency / presence (s56/s57) (2026-10-19, `b3eaf71`)

- **Intent**: The repetition penalty introduced for the 9B quality improvement was not enough on its own to suppress repetition collapse in long text. We add the OpenAI-compatible frequency / presence and the llama.cpp / HF-compatible no_repeat_ngram as **additional opt-in**.

- **What changed**
  - `b3eaf71` — `sampler.rs` (`FreqPresencePenalty`, `context_counts`, `apply_freq_presence_penalty`, `apply_no_repeat_ngram`, `AntiRepetition`).
  - `engine.rs` (3 fields on `GenParams`, `apply_session_penalty` as the common entry point for sequential and batch), validation in `server/lib.rs`, 3 CLI flags.
  - `a9dd857` — the serve default repetition penalty flag (avoiding the collapse of 9B int8 + tools).

- **Stumble**
  - Symptom: enabling them by default changes existing output. Swapping the order breaks the bit contract.
  - Cause: because the 4 kinds of correction apply to the same logits in sequence, the application order determines numeric identity.
  - Way out: fix the application order to "raw logits → repetition → frequency → presence → no_repeat_ngram (-inf) → temperature → top-k → top-p". All are disabled by default, and when `AntiRepetition::is_noop()` is true we skip both context construction and counts allocation (even the cost of the default path is unchanged).

- **What we learned / remaining issues**
  - The **internal** order of frequency and presence is observable in f32, so we killed it with a test. On the other hand, placing no_repeat_ngram before or after frequency/presence is unobservable in f32 (the -inf mask overwrites the result of the preceding stage), so we confirmed it survives a mutation test as well. Therefore "put it last" is guaranteed by code structure and documentation, not by a test (an honest note).
  - workspace 290 passed (+26 from the base 264). We left the quality evaluation of whether repetition collapse is actually suppressed as impossible without a real GPU.

- **What worked next**: the method of "proving the default no-op by bits" became the standard check for the later changes to quantization and the load path (volume 3). The order "add opt-in with default off, then measure" was also fixed here.

---

### Adding a temperature cap to serve (`--temperature-cap`) (2026-10-19, `47e1928`)

- **Intent**: Isolation of the case where conversation with the 9B collapses in 2〜3 turns. Looking at the history of real sessions, Chinese was mixed in on turn 2, and in another session fragments of Polish and Thai appeared. We identify the cause and stop it on the server side.

- **What changed**
  - `47e1928` — `denno serve --temperature-cap <f32>`. Clamps the effective temperature to `min(requested value, c)`, and also caps the sampler default of 1.0 when the request omits temperature.
  - Negative values and non-finite values are rejected at startup. 4 unit tests.

- **Stumble**
  - Symptom: multilingual mixing. We suspected the cache, but with it enabled/disabled the output over 5 turns and 10 turns was byte-identical, alternating 2 conversations that share a prefix produced no mixing (`reuse 1007/1049`), and streaming was identical to non-streaming.
  - Cause: sampling. The client's spec always attaches `temperature: 1` when thinking is enabled, while denno treats the thinking map `enabled` as off (s53c). **Thinking is turned off but only T=1 remains**. When we reproduced it with the equivalent of a real session (system 11,949 characters, 26 tools, 7.6k tokens in total), it collapsed on turns 2〜3 at T=1.0, while all 4 turns were normal at T=0.7 and T=0.0.
  - Way out: put an upper-bound flag on the server side so that the effective value is capped even when the client sends T=1. If the flag is unspecified the requested value passes through and is bit-identical to before (the anchors are unchanged).

- **What we learned / remaining issues**
  - Verification: started with `--repetition-penalty 1.05 --temperature-cap 0.7` and resent the 7.6k prompt sending T=1.0 → no collapse, and the same response as specifying T=0.7 directly. On a start without the flag, 3 GPU anchors were MATCH.
  - Remaining issue: the real fix is a client-side patch not to attach `temperature: 1` to a non-Anthropic endpoint, but it is untouched. Because it may disappear with a client update, we put the server-side cap in first.

- **What worked next**: the isolation procedure "before suspecting the cache, reproduce with the same payload and only the conditions varied" took hold. The later 27B quality investigation (volume 3) uses the same order: first reproduce on CPU, then compare against a reference implementation.

---

## Rejected approaches

Judgements that looked sound at the time but were discarded later. The reasons for rejection are written as the facts observable at that point.

### Implementing brief as "thinking ON + short-answer instruction" (rejected in s47 `4e89c11` → s53c `e7246b8`)
- Why adopted: we thought an instruction at the start of system alone could lead to an answer without changing the meaning of thinking. We stated explicitly that it was not hard control.
- Observation: in measurement the 9B kept writing meta-analysis under "Thinking Process:" endlessly and never reached an answer. The instruction does not constrain generation, so it does not work once thinking runs.
- Rejected: changed brief to "thinking disabled + concise instruction" and treated the map `enabled` as off as well. The default became effectively "do not think". As a side effect, thinking stopped running unless `full` was sent explicitly.

### Judging the stop marker by the offset from the start of the chunk (rejected in s53 `f21a985`)
- The form in use: look for the marker at a relative position from the start of the supplied chunk.
- Observation: it stops wrongly on `system` / `user` in the body (a word like `userland`). Because the marker crosses token boundaries, `u` / `us` / `use` leak into the delta. The line-label check and the prefix check did not agree on how they look at word boundaries.
- Rejected: moved to a holdback with the bytes emitted so far, `sent_len`, as the invariant. We kept the limit that the prefix path does not look at word boundaries (accepted as the safe side).

### Excluding non-stream disconnects (rejected in disconnect detection `e7d3e7a` → s44 `08e7c9b`)
- Why adopted: we took it that "hyper does not observe the connection until the response is sent", so non-stream has no chance to observe it.
- Observation: the premise was wrong. It watches the read side while the response future is pending, and on disconnect the handler future is dropped along with it. In practice a case appeared where `in_flight` stayed at 1 and did not decrease after a disconnect.
- Rejected: used `respond.is_closed()` on the oneshot held by the handler as an indirect observation of the disconnect, and put it on the same sweep as SSE.

### Rejecting tools + stream with 400 (rejected in tool use `751dc10` → streaming tool_use `05e5f8f`)
- Why adopted: there was no means of porting the non-streaming extractor to a delta stream, so we narrowed the scope. The rejection was fixed in a test (`tools_with_stream_is_rejected`).
- Observation: clients cannot use tool conversations in streaming mode. With an SSE assembler in place, a unit test showed that for any split width it matches the remaining text of `extract_tool_calls(full text)`.
- Rejected: removed the 400 and accept with `StreamAssembler`. Replaced the test with `tools_with_stream_is_accepted`.
