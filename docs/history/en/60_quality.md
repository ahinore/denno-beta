# Volume 7 Quality isolation — reducing broken output to its conditions

> [Index](README.md) / Prev: [Volume 6 GDN](50_gdn.md) / Next: [Volume 8 Measurement, verification, operations](70_infra.md)
> Japanese original: [../60_quality.md](../60_quality.md)

Period covered: 2026-10-15 through 2026-10-22. A record of how, when generation quality broke, we narrowed down by measurement which conditions alone reproduce it,
fixed it without changing the default output, and pinned the failure mode into a rig that runs without a GPU.
The implementation of API accepted formats, wiring and stop conditions is in Volume 2 `10_api_features.md`; determinism, anchors and numeric tolerance are in
`70_infra.md`. This volume does not duplicate the implementation description; it deals only with the measurements and decisions of isolation.
Dates use the date recorded in the commit author date as is (timezone as recorded in the commit.
Some commits change date when converted to JST, so we do not convert).

## The question this volume answers

- From a report that "the output breaks", how do we narrow down the conditions that reproduce it? What do we fix and what do we vary so that it falls to a single condition?
- When we introduce a fix, how do we manage without changing the default output (opt-in, default None, bit no-op)?
- Can we make quality judgement runnable on a machine without a GPU? Where is the boundary at which the parts that cannot be measured are left as "not measured"?

## Background

In greedy decoding (temperature 0), a slight difference in logits changes the choice of word and then branches the whole subsequent context.
Therefore a "quality" report does not reproduce unless it is first decomposed into decoding conditions (temperature, sampler, penalties) and prompt conditions (system length, number of tools).
Rendering of the chat template, token boundaries of byte-level BPE, and the application order of logits processing are also background.
The changes in this volume centre on logits post-processing and text post-processing; they do not touch the forward numeric path.

## Reading order

A2.5 (the body of repetition penalty) → s53c (changing the meaning of brief) → s53/s53b (stop false positives) → s56/s57 (opt-in repetition suppression) →
s56b (isolating the 9B + tools breakage and serve defaults) → s98/s100 (needle recall) → s103/s107 (10-turn conversation) →
s162 (long context into the default gate). The first 3 are "the decision whether to change the default meaning"; the last 5 are
"pinning the failure mode into a rig". s53c and s53/s53b are the same week's story, but the former is thinking and the
latter is text post-processing; the layer they touch is different.

## Phases covered

| phase | date | commit | one-line summary |
|---|---|---|---|
| A2.5 | 2026-10-15 | `4fada3c` | added repetition penalty HF-compatible and default no-op (1.1 is appropriate, 1.15/1.3 are excessive) |
| s53c | 2026-10-18 | `e7246b8` | changed brief from "thinking ON + prompt sentence" to "thinking disabled + concise instruction" (all 4 candidate prompt sentences failed) |
| s53/s53b | 2026-10-18 / 10-19 | `b1f5655` / `f21a985` | fixed the false positives of the role-label stop (23 tok / 44 tok) with a true line-start criterion |
| s56/s57 | 2026-10-19 | `b3eaf71` | added no_repeat_ngram / frequency / presence as opt-in (default no-op) |
| s56b | 2026-10-19 | `a9dd857` | isolated the breakage at 6,223 tok + 9 tools, added serve default flag (production 1.05) |
| s98/s100 | 2026-10-20 | `a0afc03` / `7543158` | needle recall regression checker and filler correction (about 4,656 tok is estimated) |
| s103/s107 | 2026-10-20 | `65e4546` / `cef9384` / `1a063af` | 10-turn conversation checker, retirement of the s97 version, single-command CPU gate |
| s162 | 2026-10-22 | `5931c7a` | long context (3,000 / 4,096 tok) regression into the default gate (4 added, +0.01 s) |

---

## A2.5: Adding repetition penalty as opt-in (2026-10-15, `4fada3c`)

**Intent**: Qwen3.5-9B, in greedy decoding at temperature 0, repeats the same phrase and the sentence does not move forward.
The first thing to suspect is the rendering of the chat template, but we verified that separately, and **the template conformed to the official one with no mismatch**
(confirmed it is not the cause of the repetition). As the remaining option, we add the standard means of penalising already-seen tokens.
But we do not change the default output by 1 byte.

**Stumble**: Our guess at the strength was off. `penalty` 1.15 and 1.3 over-suppressed.
1.1 was appropriate, and this matched llama.cpp's default. The other stumble is that "the moment you add it, the default changes".
We implemented `penalty == 1.0` as a bit-exact no-op and limited exposure to an optional field on the request and a
CLI flag. On the server's 2 decode paths (sequential and batched) we made both pass through the same function immediately before sampling.

**What we learned**: With 2 payloads of 9B int8, 1.1 broke the loop and preserved the thinking structure (the output md5
matched over 2 consecutive runs and after a restart = determinism preserved). But this is **suppression of the symptom**, not an explanation of the
numeric cause of the repetition degeneration. A comparison against running the same GGUF under llama.cpp remained. We also did not
decide to "make the default 1.1". Changing the default changes the output of every request and breaks the premises of the anchors and the CPU reference md5.

**What worked next**: Keep the default at None (unspecified) and shift to specifying it explicitly in operations. Final quality evaluation is
left as human work on a real GPU. This order — "add opt-in, then measure" — became the pattern for the 3 repetition-suppression options that follow and for the
serve default flag.

---

## s53c: Changing brief to "thinking disabled + concise instruction" (2026-10-18, `e7246b8`)

**Intent**: thinking was introduced with 3 modes: off / brief / full. brief is the default, and at the time it was
soft control: "keep thinking ON, and inject a short-answer prompt sentence at the head of system". Having thinking run by default is a
breeding ground for quality accidents, so we inspect the meaning of the default itself.

**Stumble**: The prompt sentence did not fix it. **We tried 4 candidate prompt sentences, and in every one the 9B wrote the "Thinking Process:"
meta-analysis endlessly and never reached an answer**. Shortening the system made no difference, so it was not the fault of a long system.
The cause is the kind of control: a prompt sentence does not constrain generation. Once thinking starts running, the strength of the wording has no effect on the result.
The wording of the 4 candidates is not preserved in the repository. What the record has is only 1 from s47 that was in the code until just before s53c
("Keep thinking to a few short sentences and write the answer right away.") and 1 after the replacement.
What remains is only the result: none of the 4 candidates reached an answer.
The way out was to change brief to **disable thinking** as off does, and then inject the concise instruction.
We also aligned the Anthropic map's `enabled` to be treated as off (the accepted format itself is Volume 2).

**What we learned**: Because the meaning of the default changes, the reference side of the wire tests that depend on the default had to be updated at the same time
(the engine-direct execution family has no system, so no effect). As a side effect, the operation became "thinking does not run unless full is explicitly
sent". This later combined with the problem "thinking is turned off but only temperature 1 remains" (the temperature-side handling is Volume 2's `--temperature-cap`).

**What worked next**: We abandon the premise that "generation can be controlled with a prompt sentence" and shift to hard control such as template variables.
Quality is judged not by what we made the model write but by **whether it reached an answer**. Every phase after this in the volume is
written with this view.

---

## s53/s53b: The role-label stop false-positiving, and its fix (2026-10-18 / 10-19, `b1f5655` / `f21a985`)

**Intent**: When the next speaker header appears in the generated text, the model starts writing text for a different role as is (fake turn).
We add, in text post-processing, a stop equivalent to llama.cpp's reverse prompt, and at the same time lower the default generation length from 4096 to
1024. We do not touch the numeric path.

**Stumble**: It false-positived. **In measurement it cut at 23 tok and 44 tok** (the record does not preserve the model name and quantization).
Because it stops while the body is still short, long answers do not come out. The cause is the criterion of the decision: we performed the stop decision against "the part not yet emitted". When a chunk boundary falls
in the middle of a word, `system` / `user` in the body looks like a "standalone label at line start". In addition, because the
marker straddles token boundaries, `u` / `us` / `use` leaked into the delta before it was confirmed.
There were 2 ways out: (1) make the decision take 2 arguments, **the whole text + the unsent boundary**, and find line start on the `\n` basis of the whole
text; (2) add holdback that does not emit while the tail can be a prefix of the marker, and make `sent_len` (the number of bytes already sent)
an invariant. After the fix it was **normal up to 1,024 tok** (no false stop until it reaches the default generation length).

**What we learned**: Without the invariant "hold the sent boundary in a single integer", line-start judgement does not hold.
Another point: the prefix path does not look at word boundaries, so it can still stop after holdback on `userland` after a newline.
We accept this because it is on the safe side (it stops too often) and recorded it as a limitation. The tests pinned 12-byte boundaries, CRLF, full-width colon, and
non-stop when not at line start. In git author date, the fix `b1f5655` is 10-18 and the feature consolidation
`f21a985` is 10-19; the fix landed first.

**What worked next**: Unify the decision functions of the text path to always take both "the whole" and "the boundary" as arguments.
The later SSE assembler and disconnect detection are written with the same idea (hold the boundary in a single integer).

---

## s56/s57: Adding 3 kinds of repetition suppression as opt-in (2026-10-19, `b3eaf71`)

**Intent**: A single repetition penalty cannot suppress repetition collapse in long contexts. We add OpenAI-compatible
`frequency_penalty` / `presence_penalty` and llama.cpp / HF-compatible `no_repeat_ngram_size` as
**additional opt-ins**. The reason not to make them default is simple: all 4 change the logits.
Changing the default changes the output of every request and breaks the premises of the anchors and the CPU reference md5.
All defaults are disabled and the output stays bit-identical to before.

**Stumble**: The 4 kinds of correction apply to the same logits in sequence, so **the application order becomes a bit contract**.
We fixed the order as "raw logits → repetition → frequency → presence → no_repeat_ngram (-inf) → temperature → top-k →
top-p". Here an honest limitation appeared. The **internal** order of frequency and presence is observable in f32
(subtracting separately changes the rounding), so it can be killed by a test. On the other hand, placing no_repeat_ngram before or after frequency/presence
is not observable in f32 (the -inf mask overwrites the result of the preceding subtraction). It also survived mutation testing.
That is, "place it last" cannot be pinned by a test.

**What we learned**: Default no-op required a test pinning that the logits are bit-invariant over
all combinations of disabled values (`1.0` / `0.0` / `-0.0` / `n<=1`). If `is_noop()` is true, context construction and counts allocation are also
skipped, so even the cost of the default path does not change. The remaining issue is quality evaluation: whether repetition collapse is actually suppressed
cannot be judged without a real GPU (human work). Interaction with quantization is also unevaluated.

**What worked next**: Orders that cannot be observed are guaranteed by code structure and documentation rather than tests. This compromise is
treated the same as "do not pretend to have measured what cannot be measured" in `docs/NUMERIC_POLICY.ja.md`. Do not change the default, and do not
break the order of adding opt-in and then measuring on the real machine.

---

## s56b: Reducing the 9B + 9 tools breakage to its conditions (2026-10-19, `a9dd857`)

**Intent**: In a real session, the 9B response showed a **fake `user` turn** and **`.showError warning` contamination**.
Rather than ending this with "the model is bad", we identify the conditions that reproduce it and make it avoidable on the operations side.

**Stumble**: The first hypothesis (it breaks because the context is long) was wrong. We split the conditions as follows.

- Fixed: model (Qwen3.5-9B int8), decoding (temperature 0 = greedy), prompt body, seed.
- Varied: system length, number of tools, size of the tools schema, presence/absence of `repetition_penalty`.

The result is summarised in the following single line. **It broke only at 6,223 tok (system about 2.6k + 9 tools about 3.6k)**.
In the same 6k-class context it was normal without tools, and normal with tools if the schema was shrunk.
That is, the breakage condition is not "length" alone but the combination "long system + 9 tools".
With `--repetition-penalty 1.05` the same prompt returned a normal short answer.

**What we learned**: We cannot make the countermeasure the default. Making the default 1.05 changes sampling for every request and
breaks the premises of the anchors and the CPU reference md5. So we put a default-value flag in serve and applied it
**only when the request does not specify `repetition_penalty`**. A startup without the flag stays `None` and the
output is bit-identical to before. Values of 0 or below and non-finite values are rejected at startup. The production startup script specifies 1.05 explicitly.
Note that 1.05 is a weaker value than the 1.1 judged appropriate in A2.5, and the context of the decision differs (A2.5 was 2 short payloads,
here a 6,223 tok tools context).

**What worked next**: Since the breakage condition depends on "the number of tools", reducing tools or shrinking the schema is the
real client-side fix. The server side is kept as a safety valve. This isolation procedure
(list what to fix first, then vary one at a time) was used in the same order in the later 27B quality investigation.

---

## s98/s100: Making needle recall a regression without a GPU (2026-10-20, `a0afc03` / `7543158`)

**Intent**: "Breaks in long context" type bugs had no reproduction procedure so far. Long-context needle recall
ended as a one-off disposable test (4,671 token). We turn it into a one-command script so that the same judgement comes back every time the build changes.

**Stumble**: Unless the filler is fixed, the payload differs every time and the regression cannot be compared. We assemble the filler from an
LCG and 6 word lists fixed in the fixture, and pinned that the same fixture yields a byte-identical payload by matching the real request bodies of
2 child processes. The needle is placed in an English sentence on a topic unrelated to the filler
(warehouse / clerical work), so the answer cannot come from the local context. The judgement is word-boundary-aware, and
a confusing neighbour like `QX-44715` is not regarded as a successful recall (a simple substring match would pass it).
In s100, to match the measured chars/token = 5.1785, `filler_chars` was corrected once from 21,400 to 24,000.
The payload is 24,113 chars, and **about 4,656 token is estimated**, not measured.

**What we learned**: The exit code had to be split into 3 values. 0 = recall OK / 1 = recall failure (regression) /
2 = communication or interpretation failure. This is so that "could not measure" is not misreported as "broken"; 2 is not mixed into 1.
The token count takes only the `usage` the server returns; if it is not returned, we write "not measured" and do not invent a number.
The detector itself was verified with 25 tests against a fake server (normal, wrong answer, neighbour, omission, HTTP 500, no connection,
non-JSON, 404 fallback, determinism, standard library only).

**What worked next**: Recall against a real server and the real token count are left as human work. What the checker can prove is only that
"the detector fires correctly on a given string" and "it works correctly against a fake server"; the
model's actual behaviour is not measured. We drew this boundary the same way in s103/s107.

---

## s103/s107: Making a 10-turn conversation a script (2026-10-20, `65e4546` / `cef9384` / `1a063af`)

**Intent**: There were reports that 9B conversation breaks in 2 or 3 turns, but no script reproduced how it breaks. We run a fixed 10 turns and
**append the reply to actually grow the conversation** (turn N has 2N-1 messages).

**Stumble**: The s97 prototype had 2 defects. It had only 4 detectors and could not detect the temperature-cap regression.
In s107, before retiring the s97 version, we ported `--expect-cap` (fail if the accepted temperature exceeds the cap) to the new checker.
The other stumble is the handling of temperature: if we send temperature by default, we cannot detect the serve-side cap regression.
Therefore regression detection is done with the explicit specification `--temperature 1.0 --expect-cap 0.7`. The threshold was pinned from both sides
(23 characters does not fire / 24 characters fires).

**What we learned**: The detectors became 7 kinds. `repeat_line` (the same line of 24 characters or more 3 times in a row),
`repeat_sentence` (the final sentence 3 times in a row), `long_line` (in a reply of 400 characters or more, 1 line is 60% or more),
`template_marker` (a standalone `user` / `assistant` / `<|im_start|>` etc. at line start), `fake_turn` (a fake `user`
block after a blank line), `empty`, `max_tokens`. The last one is **report-only and does not fail** (a long answer is not a collapse).
Exit codes are 0 = no collapse / 2 = collapse / 3 = infrastructure failure. Ports 8080 (serve default) and 8090 (production) are
refused before connecting, and the script does not start or stop the server. In s107 all stages were combined into 1 command
(`scripts/run_cpu_checks.sh`: kernel invariants → engine lib → server → each self-test. It stops at the first failure and
returns 0 only when everything passes).

**What worked next**: Running 10 turns on a real model is human work. The script thresholds are premised on adjustment after looking at the real machine's response distribution;
the initial values are values verified against the fake server. The script may only be appended to; reordering is forbidden.

---

## s162: Putting long-context regression into the default gate (2026-10-22, `5931c7a`)

**Intent**: The default gate all ran with `DUMMY_CONTEXT = 64`. The paths that only appear in the 3,020 token
prefill that production serves and in multi-turn conversation (the path where the KV cache grows beyond its initial capacity,
the prefix cache budget and eviction on long prompts, the context capacity boundary) had never been
executed in the default gate.

**Stumble**: Running long context needs a test model. We added `LongKvSensitiveModel::with_context(n)`
(context 4096, 1 layer, kv_dim 4, 32 B/token), and `forward` delegates to the existing `KvSensitiveModel`,
passing the KV row formula and the logits assembly through the same code as the 64 token version. We also separated overlap with existing tests one by one
(tests that looked only at text and token count at 64 token or below, tests that asserted the primitive API with a minimal `max_len`, etc.).

**What we learned**: The 4 added tests are all in the default gate (no `#[ignore]`), and lib was 274 passed / 0.27〜0.29 s
(the 4 added alone are 0.01 s). Because kv_dim is 4, the KV is 128 KB even at 4,096 token, so there was no need to shrink the length.
On the other hand, in 45 iterations of the whole lib, 8 times 1 test in `prof::tests` failed (flaky from sharing a global counter.
The 4 added failed 0 times out of 45). Test isolation was handled in a separate phase (s163).

**What worked next**: Long context on real models (27B q4 / 9B int8), contexts above 4,096, and GDN's long-context prefix cache are
left as not measured. What the rig in this volume can prove is only that "the detector fires under the given conditions" and
that "the long-context path is actually taken in the default gate".

---

## Rejected approaches

These looked valid at the time but were discarded later. The reason for rejection is written with the facts observable at that time.

### Controlling brief with only a prompt sentence (rejected in s47 → s53c `e7246b8`)
- Reason adopted: we thought we could lead to an answer with only an instruction at the head of system, without changing the meaning of thinking. We had stated it was not hard control.
- Observation: **we tried 4 candidate prompt sentences, and none reached an answer**. The 9B wrote the "Thinking Process:" meta-analysis endlessly, and shortening the system made no difference.
- Rejected: changed brief to "thinking disabled + concise instruction" and treated the map `enabled` as off too. A prompt sentence does not constrain generation, so it has no effect once thinking runs.

### Performing the stop decision against "the unsent part" (rejected in s53 → s53b `b1f5655`)
- Form in use: pass `text[sent_len..]` and treat its head as the line start of the text.
- Observation: **it cut at 23 tok / 44 tok**. When a chunk boundary falls in the middle of a word, `system` / `user` in the body looks like a standalone label at line start. After the fix it was normal up to 1,024 tok.
- Rejected: made the decision take 2 arguments, "the whole text + the unsent boundary", and find line start on the whole `\n` basis. The point that the prefix path does not look at word boundaries was left as the safe side.

### Making the repetition penalty default 1.05 / 1.1 (rejected in A2.5 `4fada3c` / s56b `a9dd857`)
- Form we nearly adopted: since it fixes quality, just make it the default.
- Observation: changing the default changes sampling for every request and breaks the premises of the anchors and the CPU reference md5. In the 6,223 tok tools context, 1.05 made the breakage disappear, but that is about a specific condition.
- Rejected: the default stays `None` (unspecified). It is specified explicitly through the serve startup flag and the request. A startup without the flag is bit-invariant.

### Adopting penalty 1.15 / 1.3 (rejected in A2.5 `4fada3c`)
- Reason adopted: repetition was strong, so a stronger penalty looked more reliable.
- Observation: 1.15 and 1.3 over-suppressed. 1.1 was appropriate and matched llama.cpp's default.
- Rejected: recorded 1.1 as the appropriate value, and use the even weaker 1.05 in the operational 6,223 tok context.

### The reading that "the length of the 6k context is the cause of the breakage" (rejected in s56b `a9dd857`)
- Form we nearly adopted: the simple reading that it breaks when the context length exceeds a threshold.
- Observation: in the same 6k-class context it was normal without tools, and normal with tools if the schema was shrunk. What broke was only **the combination of 6,223 tok (system about 2.6k + 9 tools about 3.6k)**.
- Rejected: changed the countermeasure from "cut the length" to "look at the number of tools and the schema". Looking only at length gets the reproduction condition wrong.

### Pinning the application position of no_repeat_ngram by a test (rejected in s56/s57 `b3eaf71`)
- Form we nearly adopted: kill the order of the 4 penalties entirely by mutation testing.
- Observation: the internal order of frequency and presence is observable in f32, but for no_repeat_ngram the -inf mask overwrites the preceding result, so no difference appears in f32 whichever side it is placed on. It also survived mutation testing.
- Rejected: guarantee "place it last" by code structure and documentation rather than a test. Do not pretend to have measured an order that cannot be measured.

### Making the needle judgement a simple substring match (rejected in s98 `a0afc03`)
- Form we nearly adopted: recall succeeds if the expected string is contained.
- Observation: a confusing neighbouring string like `QX-44715` also matched, passing output that did not recall.
- Rejected: made the judgement word-boundary-aware. The detector pinned, with 25 tests against a fake server, that neighbour, wrong answer and omission are each failed.

### Merging exit code 2 (could not measure) into 1 (broken) (rejected in s98 / s103)
- Form we nearly adopted: all failures are non-zero, so merge into 1.
- Observation: reporting a communication failure or non-JSON as a "regression" misreports a build that does not need fixing as broken.
- Rejected: needle uses 0 / 1 / 2 and the 10-turn conversation uses 0 / 2 / 3. Do not mix infrastructure failure with quality regression.

### The 10-turn conversation checker sending temperature by default (rejected in s103/s107)
- Form we nearly adopted: send temperature explicitly from our side to get closer to a real session.
- Observation: the serve-side temperature cap is a startup setting, so if the checker overwrites it, the cap regression cannot be detected.
- Rejected: by default do not send temperature. Regression detection is done with the explicit specification `--temperature 1.0 --expect-cap 0.7`, and fail if the accepted temperature exceeds the cap.

---

Date basis: the date of each phase is the date recorded in the commit author date, used as is (no JST conversion).
