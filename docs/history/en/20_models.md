# Volume 3 Model support — from GPT-2 to the 27B hybrid

> [Index](README.md) / Prev: [Volume 2 API features](10_api_features.md) / Next: [Volume 4 Decode speed improvements](30_decode_perf.md)
> Japanese original: [../20_models.md](../20_models.md)

Period covered: 2026-10-11 through 2026-10-23. A record of adding supported models, adding quantization formats, and running the 27B output breakdown to ground.
Speed, kernel and measurement topics are in other volumes. Here we deal only with "which weights we read, in what shape, and how we interpret them".
Dates are commit author dates in JST.

## What this volume covers

- The order in which supported models were added (GPT-2 → Qwen2.5 → larger Qwen → quantization → GGUF → Qwen3.5 hybrid → 27B)
- How to verify a new quantization format (bit-identical against a reference implementation, fixtures, cross-checking the type ids)
- The process of isolating the 27B output breakdown in the order "structure → numeric path → comparison against the reference implementation", and the fact that GDN's GQA head mapping was the root cause
- How to change the way weights are held (resident f32 table, row-chunk quantization, raw Q4_0 blocks) while keeping the numerics unchanged

## Background

The GGUF container and ggml quantization blocks (k-quant / IQ families), Gated DeltaNet (the recurrent state of the delta rule),
GQA head mapping, chat templates. There are 2 kinds of numeric contract: "bit-identical (R1)" and "determinism + output validity (relaxed)";
GDN belongs to the latter, GEMM and quantization belong to the former.

## Reading order

P3 → P7 → P19 (foundation → GDN reference → model integration) → 9B (A2 / A2.5) → 27B (A3 / A4) → GPU regression investigation →
27B structure verification, quality investigation, GQA fix → prefix cache support for recurrent models → Q4_0 native →
model info display (s156) → quantization decoders (K1) → stricter display (s158) → pinning bit-identical behaviour (K2 / K3) →
GGUF fallback (s164) → dominant type (s168) → layer count (s175) → how weights are held (P35 / P37 / P38 / P42).
The 3 27B entries (structure verification, quality investigation, GQA fix) are one continuous story, so read them together.

## Phases covered

| phase | date | commit | one-line summary |
|---|---|---|---|
| P3 | 2026-10-11 | `ffabb50` | int8 weight-only quantization (memory halved, speed flat) |
| P7 | 2026-10-12 | `b6c6c5d` | loading split safetensors and untied lm_head |
| P19 groundwork | 2026-10-14 | `49ac073` | GGUF container reader and decoding of 18 k-quant / IQ types |
| P19-B | 2026-10-14 | `0bbfa79` | CPU reference for Gated DeltaNet |
| P19-B2 | 2026-10-14 | `feab1c8` | Qwen3.5 model integration and E2E |
| A2 | 2026-10-15 | `9183e7e` | the 9B GDN k-head convention and resolving untied lm_head |
| A2.5 | 2026-10-15 | `4fada3c` | repetition penalty added HF-compatible and a no-op by default |
| A3 | 2026-10-15 | `646c22a` | synthetic shape test for the 27B and correcting the KV budget |
| A4 | 2026-10-15 | `bf2c90b` | GDN state snapshot for the prefix cache |
| GPU E2E regression investigation | 2026-10-15 | `d409d4f` / `ed09722` / `9dc6fc9` | explaining the 9B GPU prefill breakdown as drift amplification, and writing down the regression criteria |
| 27B structure check | 2026-10-16 | `203deed` | rejection of the out-gate hypothesis, `expect_ne` and MTP exclusion |
| 27B quality investigation | 2026-10-16 | `7b07870` | CPU isolation and comparison against the llama.cpp reference |
| GQA head mapping fix | 2026-10-16 | `e266090` | changing blocked to strided makes the 27B answer correctly |
| s46 | 2026-10-17 | `262bd85` | prefix cache hits for recurrent models (split prefill) |
| Q4_0 native path | 2026-10-20 | `c51d6a3` | the 27B runs at 18.4 GB / 178-186 ms/token |
| s156 | 2026-10-21 | `ca5e44a` | `denno info` reads the dimensions from `text_config` |
| K1 | 2026-10-21 | `2344f67` | adding TQ1_0 / IQ1_M decoders and correcting the type ids |
| s158 | 2026-10-22 | `c95b44c` | unknown becomes `?`, and rope / attention / dtype / quantization are displayed |
| K2 | 2026-10-22 | `f6f03fd` | TQ1_0 / IQ1_M pinned bit-identical against the reference implementation and moved into the default gate |
| K3 | 2026-10-22 | `45a9696` | Q4_0 edge blocks and the accumulation order of the CPU gemv pinned |
| s164 | 2026-10-22 | `8a0525b` | filling the dimensions from a GGUF-only directory |
| s168 | 2026-10-22 | `73d2e10` | `quant:` reports the dominant type other than F32 |
| s175 | 2026-10-23 | `d787f66` | layer count aligned to the inference trunk (block_count − nextn) |
| P35 | 2026-10-23 | `98dd8e9` | judging raw-block passthrough on real data (allowed types 0) |
| P37 | 2026-10-23 | `1bc1702` | bounding the fp32 scratch area at load time (10/10 bytes identical) |
| P38 | 2026-10-23 | `cf917e5` | proof that the table can be replaced by per-row dequantization |
| P42 | 2026-10-23 | `955690e` | replacing the resident 5,086 MB `token_embd` with per-row dequantization |

---

### P3: int8 weight-only quantization (2026-10-11, `ffabb50`)

- **Intent**: We wanted to halve memory. At the time the fp32 weights were about 1 GB even for 0.5B, so there was no room to load a larger model.

- **What changed**
  - `ffabb50` — per-channel scale quantization and a matvec kernel with dequantization folded in (new `quant.rs`, 430 lines).
  - fp32 stayed the default; it is only enabled with `--quant int8`.

- **Stumble**
  - Symptom: the expected speedup did not appear. 0.5B went 14.4 → 13.9 tok/s, essentially flat.
  - Cause: the implementation dequantized the weights and then accumulated in fp32, so the dequantization overhead cancelled the bandwidth savings.
  - Way out: at this point we gave up on speed and put the value in halving memory (about 1 GB → about 0.5 GB). The true integer dot product is solved later by the SIMD kernels.

- **What we learned / remaining issues**
  - 77 tests green. fp32 exactness unchanged; int8 was completely identical across 3 runs (deterministic).
  - The remaining issue was "a form that does not pay dequantization every time" = dynamic quantization of activations + an i8×i8 integer dot product.

- **What worked next**: the separation "quantization is about memory, speed is a different topic" took hold. Even in the later Q4_0 native path, VRAM was the goal first and speed came as a consequence.

---

### P7: support for larger Qwen models — split safetensors and untied lm_head (2026-10-12, `b6c6c5d`)

- **Intent**: We wanted to extend the path that worked for 0.5B to 1.5B / 3B / 7B. The 7B class does not fit in a single file, and some models do not share lm_head with the embedding.

- **What changed**
  - `b6c6c5d` — loading split safetensors (resolving `model.safetensors.index.json`, fetching all shards, resolving tensor names. New `WeightStore`).
  - The path for a non-tied lm_head (`tie_word_embeddings: false`).

- **Stumble**
  - Symptom: the 7B could not be loaded / the 3B output was broken.
  - Cause: the weights are split across index.json + multiple shards, which a loader that assumes a single file cannot resolve. For the 3B, lm_head is a separate tensor but the embedding was being reused.
  - Way out: look up the tensor name → shard mapping from index.json and resolve through `WeightStore`. When untied, prefer the `lm_head` tensor.

- **What we learned / remaining issues**
  - 93 tests green (+2). 0.5B exactness unchanged. Normal responses confirmed for all of 0.5B / 1.5B / 3B / 7B.
  - Measured by size (20-core CPU): 0.5B is fp32 14.4 tok/s and int8 about 51 tok/s, 3B is about 2 → about 4.7 tok/s (2.4×), 7B is about 1.2 → about 2.1 tok/s (1.8×). The 7B weights are 28 GB, which is comfortable with 121 GB of RAM.

- **What worked next**: the point "resolve correctly from the config whether it is untied" recurs as-is in the later 9B logits wipeout (middle of this volume). And for the 9B the config was in a wrapper format, so the key was in a different place.

---

### P19 foundation: GGUF container reader and decoding of 18 k-quant / IQ types (2026-10-14, `49ac073`)

- **Intent**: The 27B distribution format was GGUF (mixed quantization). The safetensors path could not reach it, so we first had to build GGUF reading and decoding of the quantization blocks.

- **What changed**
  - `49ac073` — `gguf.rs` (an mmap-based container reader. Header / KV / tensor info).
  - `kquant.rs` and the generated `kquant_tables.rs` (generator `ref/gen_kquant_tables.py`). Decoding of 18 k-quant / IQ types.
  - Bit-identical verification against llama.cpp with LCG fixtures (all 16 blocks × 18 types match), plus verification on a real GGUF with 320 tensors.

- **Stumble**
  - Symptom: the fixtures could not be committed. We wanted to hand-edit `kquant_tables.rs`.
  - Cause: `data/` in `.gitignore` also excluded the fixture directory. The distinction between generated and hand-written code was also vague.
  - Way out: add a negation to `.gitignore` so only the fixtures pass. We restricted `kquant_tables.rs` to generated output and stated the policy that hand editing is forbidden.

- **What we learned / remaining issues**
  - All 18 types bit-identical with llama.cpp. The 320 tensors of the real GGUF could also be verified.
  - Remaining issue: every time a type contained in a mixed quantization is added, this verification has to be run again (the later K1/K2/K3 are exactly that).

- **What worked next**: "link the reference implementation and use its output as the expected value" became the standard method around quantization. Pinning the md5 of the fixtures means that, whenever the generator is touched, we can confirm every time that the existing part has not changed.

---

### P19-B: CPU reference for Gated DeltaNet (2026-10-14, `0bbfa79`)

- **Intent**: Qwen3.5 is a hybrid of Gated DeltaNet and full attention, and there was no code that handled GDN's recurrent state. First we build 1 reference implementation and take it as the truth for the integration.

- **What changed**
  - `0bbfa79` — new `gdn.rs`. A 1:1 port of llama.cpp's `build_delta_net_autoregressive` (sequential AR form).
  - `cb9d30b` — module registration in `lib.rs`.

- **Stumble**
  - Symptom: the formulas can be followed, but it is not decided how far it should match llama.cpp.
  - Cause: the GDN update is mathematically equivalent in the chunked form and the sequential AR form, but the rounding order does not match. Requiring bit-identical would constrain the implementation.
  - Way out: we stated explicitly that GDN is outside the bit-identical contract (denno's contract is "same input → same output" + determinism). We write it in the sequential AR form and treat equivalence with the chunked form as a mathematical claim.

- **What we learned / remaining issues**
  - The update is g = ssm_a (negative) × softplus(alpha + dt_bias), S ← S·exp(g), the delta rule (sk = Σk·S, d = β(v − sk), S += k⊗d, o = Σq·S), q/k L2-normalized, output is a gated RMS norm (rmsnorm(o)·w·silu(z)), causal depthwise conv + SiLU. softplus follows the ggml convention.
  - Tests are 4 in the gdn lib (agreement with an independent oracle, bit-identical between batch prefill and token-by-token, determinism). We also checked layer-0 of a real GGUF and saw that the decay exp(g) has multiple time scales from 0.0086 to 0.9995.

- **What worked next**: by declaring the exception "this layer does not require bit-identical" up front, the later design of the GPU kernels and the chunked scan became free. At the same time, a criterion was born for isolating GDN's numeric differences as "within the relaxed contract".

---

### P19-B2: qwen35 model integration and E2E (2026-10-14, `feab1c8`)

- **Intent**: With GGUF and the GDN reference implementation in place, assemble Qwen3.5 as 1 model and run it end to end on the real model.

- **What changed**
  - `feab1c8` — new `model/qwen35.rs` (dispatch on `model_type: qwen3_5`).
  - 24 layers = 18 GDN + 6 full attention (3:1), full attention has Q+gate fused, partial MRoPE (interleaved [11,11,10]), QK-Norm, sigmoid output gate, tied embeddings (vocab 248,320).
  - GGUF (including mixed quantization) → dequant → requantize into the existing Fp32/Int8 pipeline, so that no new GEMM is created. The tiny f32 tensors (conv1d / ssm_a / ssm_dt / ssm_norm) use the reference view `GdnCore` + `forward_token_core`. The GDN state lives in the GDN slots of the KV cache (conv buffer + SSM matrix).

- **Stumble**
  - Symptom 1: out-of-range panic on the second token.
  - Symptom 2: the output changes even with the same seed (non-determinism).
  - Symptom 3: quality drops in int8.
  - Cause: (1) `cache.len()` does not advance until forward finishes, so allocating per token is not enough. (2) Live reuse of the KV and prefix entry restore invalidate GDN's conv/SSM state (a recurrent state cannot be sliced by position). (3) The per-tensor quantization error was 0.8〜1.5 % on all tensors, which is the normal range, but the worst were `ssm_beta` / `ssm_alpha` — control signals that pass through sigmoid / softplus.
  - Way out: (1) allocate for all tokens at once at the start of forward. (2) add `Model::is_recurrent()` and, for a recurrent model, reuse only pure extension (lcp == len); otherwise do a full prefill. (3) keep the control signals in Fp32 even in Int8 mode (under 1 MB for 24 layers, so the cost is negligible).

- **What we learned / remaining issues**
  - Tests are 4 gdn plus `qwen35_model` (load / forward / bit-identical determinism over 2 runs / cache continuity forward(3)+forward(1) == forward(4) bit-identical), with lib 114 passed.
  - E2E: Japanese QA, an English haiku, and a 307-token summary were coherent. With greedy (temp 0) both int8 and fp32 answered correctly, and seed 42 × 3 consecutive requests were completely identical. Prefill about 90 ms/token, decode about 20 ms/token (CPU token-by-token), and the next bottleneck was moving to the GPU.

- **What worked next**: the finding "a model with recurrent state breaks the premise of the prefix cache" became the starting point for the later GDN state snapshot (A4) and for split prefill at the template boundary (s46).

---

### A2: real E2E for Qwen3.5-9B — the GDN k-head convention and untied lm_head (2026-10-15, `9183e7e`)

- **Intent**: Extend the path that worked for 0.8B to 9B (nk=16 / nv=32, untied, 32 layers = 24 GDN + 8 full). The output broke at the points where the structure differs from 0.8B.

- **What changed**
  - `9183e7e` — the v-head → k-head mapping in `gdn.rs` and `gdn.cu`, and `tie_word_embeddings` resolution in `qwen35.rs`.

- **Stumble**
  - Symptom: the 9B output is garble (a meaningless run of characters). The hidden states match the numpy reference for layers 0〜31, yet only the output is broken.
  - Cause 1: the k-head that v-head `h` refers to was computed as `h % nk`. For 0.8B nk == nv so it is harmless, but for 9B it is a wrong mapping.
  - Cause 2: the 9B config.json is in the VLM wrapper format, and `tie_word_embeddings: false` is at the **top level** and not inside `text_config`. The loader looked only at `text_config` and did `unwrap_or(true)`, so it treated it as tied and used the embedding instead of lm_head, wiping out the logits.
  - Way out: cause 1 changed to `h * nk / nv` (equivalent to HF / llama.cpp blocked tiling). Layer 0 numpy verification showed an error difference of blocked 0.27 % / interleaved 73 %, which settled it. For cause 2, fall back to the top level when it cannot be resolved from `text_config`.

- **What we learned / remaining issues**
  - Verification: comparison against the numpy reference implementation matched all 32 layers for a single fp32 token (max relative error 1e-4), logits matched for 3 sequential tokens, sequential and batch prefill were bit-identical, and llama.cpp on the same GGUF produced normal output (isolating it as an engine-specific bug).
  - E2E (int8): normal thinking output. md5 `90fe7e58…` was identical on 2 consecutive runs and after a server restart.
  - **Note**: the blocked convention introduced here turned out to be wrong when strided was found correct in the later GQA head mapping fix (late in this volume), and it was reversed. The 9B had merely been "tolerating" blocked.

- **What worked next**: "the hidden states are right but only the output is broken" = suspect the final stage (lm_head / output gate) became an established isolation. The 27B investigation also suspected this shape first (and it was rejected).

---

### A2.5: repetition penalty — dealing with greedy repetition degeneration in the 9B (2026-10-15, `4fada3c`)

- **Intent**: The 9B sometimes repeated the same phrase under greedy at temperature 0 and never reached an answer. First suspect the chat template, then deal with it on the sampling side.

- **What changed**
  - `4fada3c` — `sampler::apply_repetition_penalty` (HF-compatible, deterministic). Exposed as `GenParams.repetition_penalty`, an optional field of `/v1/messages`, and the CLI `--repetition-penalty`.

- **Stumble**
  - Symptom: it does not escape the thinking loop. It is unclear whether the template or the numerics are to blame.
  - Cause: the template followed the official one and had no mismatch (we expanded the whole chat_template and ran real CPU tests with minijinja + pycompat). So the cause is on the generation side.
  - Way out: for the unique already-seen tokens in the context (prompt + generated), apply `logit / penalty` if `logit > 0` and `logit * penalty` if `logit <= 0`, exactly once (the same rule as HF's `RepetitionPenaltyLogitsProcessor`). Place it before temperature / top-k / top-p (on the raw logits). It uses no RNG, so greedy is fully deterministic. `penalty == 1.0` is a bit-exact no-op.

- **What we learned / remaining issues**
  - The 9B int8 E2E (penalty 1.1) escaped the loop. md5 `4551f00f…` was identical on 2 consecutive runs and after a restart. penalty 1.15 / 1.3 over-suppress; 1.1 is the right value (it matches the llama.cpp default).
  - Added a permanent regression test `qwen35_9b_int8_penalty_determinism` (determinism + repetition-loop detection with a most-frequent 4-gram canary).
  - Remaining issue: the root numeric cause of the repetition degeneration is unidentified. The handling of thinking output is also unsettled.

- **What worked next**: the pattern "fix the application position and prove the default no-op by bits" was established here, and the later 3 kinds of repetition suppression (another volume) were built on top of it.

---

### A3: scaling to Qwen3.8-27B — synthetic shape test and KV budget (2026-10-15, `646c22a`)

- **Intent**: We wanted to check with structure and budget, before placing the real GGUF, whether the 27B (64 layers = 48 GDN + 16 full, untied) can run on the existing path.

- **What changed**
  - `646c22a` — a synthetic 27B shape test in `qwen35.rs` (a minimal GGUF v3 writer, F32 only, that generates a small model preserving the structural ratios inside the test).
  - A GDN head ratio guard, and dropping KV buffer allocation for GDN layers.

- **Stumble**
  - Symptom: the 27B config has a key `output_gate_type: "swish"` that the 9B does not have, and its interpretation is not settled. The KV budget estimate was also 4× too large.
  - Cause: cross-checking 3 reference implementations (transformers main / vLLM / llama.cpp master) showed that **none of them consumes this key**. On the KV side, KV buffers were being allocated and counted even for GDN layers (layers that do not write KV).
  - Way out: the full attention output gate is sigmoid and GDN's z gate is silu (= swish), which matches the existing behaviour → the numeric path is unchanged. When the key is present we log info, and an unknown value becomes a warn. For KV we exclude GDN slot layers from allocation, from `bytes_per_token`, and from the snapshot targets.

- **What we learned / remaining issues**
  - KV went from 512 KB/token to 128 KB/token for the 27B, and from 256 KB to 64 KB for the 9B (1/4 of the real amount).
  - Budget estimate (int8, KV budget default 12 GB): linear layers int8 about 24.6 GB + embed fp32 about 5.1 GB + lm_head int8 about 1.3 GB ≈ about 31 GB (host RAM). With KV at 128 KB/token, a 32K prefill = 4 GB, 24 slots at the default budget, and a single-request cap of about 98K tokens. The full width of ctx 262,144 needs about 34 GB of KV, which exceeds the default budget, so the budget has to be raised.
  - Tests are lib 128 passed (7 new). The UD-IQ2_XXS GGUF is about 7.3 GB (single file) and every quantization format it uses is dequant-supported — however the IQ1_M type id was misidentified at the time and is corrected later in K1.

- **What worked next**: the technique "fix the contract with a synthetic model that preserves the structural ratios even without the real thing" became the foundation for all the 27B work (structure verification, split prefill, GQA fix).

---

### A4: GDN state snapshot for the prefix cache (2026-10-15, `bf2c90b`)

- **Intent**: Up to A3, a recurrent model did not use prefix cache entry restore and always did a full prefill. Copying only the KV does not bring back the GDN state and produces silent non-determinism. We want to skip prefill in multi-turn.

- **What changed**
  - `bf2c90b` — `kv_cache.rs` (`gdn_snapshot` / `gdn_restore`, bit copy), `prefix_cache.rs` (state bytes converted to tokens in the LRU weight), `engine.rs` (change of eviction timing).

- **Stumble**
  - Symptom: it is not decided where to save the state.
  - Cause: the GDN state (conv + SSM) is a function of the whole sequence and cannot be saved "per token position".
  - Way out: save exactly 1 copy of the state at the prompt boundary (right after prefill, before decode), paired with the KV snapshot. Eviction moves from completion time (`release_session`) to right after prefill, and KV rows are saved at exactly the prompt length. The restore condition is only "the new prompt contains the entry prompt's full length as a prefix". On a partial match we do not even restore the KV alone but fall back to a full prefill — we structurally never create a path that restores only the KV and leaves the recurrent state inconsistent.

- **What we learned / remaining issues**
  - Tests +13 (lib 147). As a **control experiment** we confirmed that restoring only the KV (without restoring the GDN state) changes the output — the test really does verify the state restore (a teeth check).
  - The continued forward after restore is bit-identical to a full prefill (it follows from cache continuity and the bit copy of the state). snapshot / restore is a pure memory copy, with no change to the numeric path.
  - Remaining issue: multi-turn E2E measurement on the real machine. For the 27B the state is about 150 MB per entry, so the effect on the prefix cache budget has to be looked at.

- **What worked next**: the understanding "for a model with state, the cache premise (cuttable by position) breaks" leads to s46's split prefill (a problem of not hitting for a different reason).

---

### GPU E2E regression investigation — the Qwen3.5-9B GPU prefill breakdown (2026-10-15, `d409d4f` / `ed09722` / `9dc6fc9`)

- **Intent**: A report that only the 9B GPU server produces broken output while the CPU server is normal. Settle whether it is a code regression, the environment, or numeric instability. There was no code change; only the investigation record and the written criteria remain.

- **What changed**
  - `d409d4f` — added the investigation record to HISTORY.
  - `ed09722` — the record of the drift injection experiment.
  - `9dc6fc9` — added the GPU-side anchors and wrote down the regression criteria.

- **Stumble**
  - Symptom: broken from the first token on the GPU. Even with `max_tokens=1` it disagrees with the CPU. 5 consecutive runs completely identical (determinism preserved).
  - Cause: isolation showed that the prefill path was broken. However, when we diffed all changes since A2 file by file and function by function in the GPU path files, **none of the changes reaches the Qwen3.5 GPU prefill path**. Running the same payload on a CPU build gave md5 `3a0ddf1c…`, completely identical to the CPU server. The only remaining explanation was "the numeric difference on the GPU is amplified by greedy".
  - Way out: identify the first diverging layer with a per-layer checksum (`[LCHK]`, env gated) → relative ssq difference of about 4e-7 at the attn of L00 (the first GDN layer). The embedding and the full attention layers were bit-identical. Next we injected a deterministic perturbation into the CPU build and confirmed that even eps=1e-7 (smaller than the observed drift) splits the final argmax when injected at L00. The split destination was most often the runner-up at the final position `8160 ("Here")` (6 of 9 configs). The top-2 are `90700 ("Thinking")` and `8160` with a logit difference of 2.03, a near-tie.

- **What we learned / remaining issues**
  - The relative ssq difference of the hidden states grows chaotically by 3〜4 orders of magnitude, from about 1e-5 at 32 layers to O(0.1). The earlier the source layer, the larger the amplification.
  - Judgement: the GPU observation is explained by "the instability of temperature 0 greedy (a fragility in the specification)", and there is no evidence that the drift of the GDN GPU kernel is abnormally large. H1 (a build from another source) and H3 (model corruption) are rejected.
  - Implication: the GPU E2E regression judgement for a GDN model uses not "bit-identical with the CPU" but **(a) determinism on the GPU (b) validity of the output content**. The anchors for the new canonical payload are CPU `3a0ddf1c…` / `95199a33…`, GPU `1a4ea71a…` / `b4e58cb3…` (each identical on 2 consecutive runs).

- **What worked next**: with the pass/fail criteria for a model that contains layers that cannot be bit-identical written down, later numeric-changing fixes such as the GQA fix could proceed together with an anchor update.

---

### 27B structure verification and load-time guards — rejecting the out-gate hypothesis + MTP exclusion (2026-10-16, `203deed`)

- **Intent**: Kill the hypothesis that the cause of the 27B output breakdown is "misreading the structure". At the same time, add load-time guards so that an unexpected structure is not read silently.

- **What changed**
  - `203deed` — `expect_ne` in `qwen35.rs` (compare the GGUF header shapes against the expected values derived from the config before dequant), and exclusion of the MTP (nextn) layer.
  - `tests/qwen35_27b_header.rs` (verifies only the header of a real 27B GGUF, `#[ignore]` + graceful skip when the env var is unset).

- **Stumble**
  - Symptom: only the 27B output is broken. We cannot rule out a wrong interpretation of the structure.
  - Cause: verifying the hypothesis required a 3-way cross-check (transformers' `modeling_qwen3_5.py`, llama.cpp's `qwen35.cpp`, and the real GGUF headers of 0.8B / 9B / 27B in Q4_0). The result: the 27B full attention structure matches the current engine exactly. The 9B being square with `heads×dh == e` (4096) is a coincidence, and the non-square 0.8B (1024→2048) and 27B (5120→6144) are the correct case for the default path. The position of the out-gate is also common to the 3 sizes (sigmoid after attention, before o_proj).
  - Way out: reject the hypothesis and narrow the investigation to drift in the numeric path / quality factors. The guard was defined as "before dequantizing a layer tensor, compare the shape in the header against the expected value derived from the config; ignore a trailing 1; reject a dimension higher than expected".

- **What we learned / remaining issues**
  - MTP handling: the part where GGUF `block_count` exceeds `num_hidden_layers` is skipped only if it matches `nextn_predict_layers` and the excess block has `nextn.eh_proj` and no `ssm_a` (27B: `block_count=65` = trunk 64 + 1). An inconsistency is a load rejection. Reading nextn as a normal layer would break the semantics.
  - Tests are 4 synthetic + the real header verification (PASS on the real file). workspace 188 passed.
  - Real machine: UD-IQ2_XXS cannot start because IQ1_M decoding is unsupported → the uniform Q4_0 version (16.06 GB) loaded successfully, VRAM 30.2 GB / 32.6 GB, warm decode about 0.9 s/step, md5 identical over 3 runs `1aeac03a…`. However the **output quality is broken** (greedy loops), and we move on with it unexplained.

- **What worked next**: `expect_ne` and the MTP exclusion later became the premise for "outputting the layer count from a GGUF-only directory" (s164 / s175). Having understood at this point why block_count is 65 paid off.

---

### 27B output quality investigation — CPU isolation and comparison against the llama.cpp reference (2026-10-16, `7b07870`)

- **Intent**: With the structure confirmed correct, the next question is whether it is the numeric path or the quality of the GGUF itself. Run the same GGUF under the same conditions in the reference implementation and compare.

- **What changed**
  - `7b07870` — made the diagnostic test `tests/qwen35_27b_logit_probe.rs` permanent (`#[ignore]` + graceful skip when the env var is unset, findings recorded in the header).

- **Stumble**
  - Symptom: greedy echoes prompt fragments + loops on `assistant` / template fragments.
  - Cause: first we reproduced it on the CPU, rejecting the GPU drift amplification theory. repetition penalty 1.1 did not fix it either (the loop merely turned into an early end_turn = the logits distribution itself is wrong). The GGUF header matches the config exactly and all the types used are supported. The chat template was also byte-identical to the 9B's.
  - Way out: built mainline llama.cpp (`59d5fad`, with `qwen35.cpp` support) on the CPU and compared the same GGUF with the same question. llama.cpp answered correctly (`Paris is the capital of France.`). Furthermore, comparing the distribution at generation step 0 on the same raw prompt (17 tokens, tokenization also matching), llama.cpp's top-1 is `The` (logprob ≈ 0, more than 10 nat ahead of the runner-up) while this engine's top-1 is `What` (logit 15.79) with `The` at top-2 (14.15, difference 1.63). This engine's top-1 is not in llama.cpp's top-5.

- **What we learned / remaining issues**
  - What the probe showed: the greedy 24 steps for `"What is the capital of France?"` **echo the user's question token by token completely** over steps 0〜6, emit `<|im_end|>` at step 7 (logit 19.25, runner-up 16.5) and stop, then loop on template fragments. The prefill itself is bit-identical (determinism preserved).
  - Conclusion: the GGUF quality candidate is rejected. **The 27B breakdown is specific to this engine's 27B path**, and it has already diverged at generation step 0 (= the logits of the final prefill position). The remaining candidates are GQA q24/kv4 (group 6), GDN v-head 48 (repeat 3), and the tensor correspondence of the real weights.
  - The 9B anchor reproduced (`3a0ddf1c…`), confirming that the 9B path is sound.

- **What worked next**: "compare the step 0 distribution on the same input as the reference implementation" led directly to identifying the first diverging layer in the next GQA fix. Keeping the probe as a test instead of discarding it also paid off.

---

### Qwen3.5 GDN GQA head mapping fix — from blocked to tiled repeat (2026-10-16, `e266090`)

- **Intent**: Among the remaining candidates, settle the correspondence of the k/q heads that the GDN v-head refers to. Kill the root cause of the 27B output breakdown.

- **What changed**
  - `e266090` — `forward_token_core` and `scan_chunked_core` in `gdn.rs`, and the 3 kernels in `kernels/gdn.cu` (`dno_gdn_scan` / `dno_gdn_chunk_pre` / `dno_gdn_chunk_scan`).
  - The convention comments in `qwen35.rs` and the load-time guard message.

- **Stumble**
  - Symptom: the logits at the final prefill position of the 27B differ from the reference implementation. It is not known at which stage it diverges.
  - Cause: we prepared a per-layer checksum tool on the llama.cpp side and compared; the first divergence is in the attn substage of the GDN layers (full attention, embedding, FFN and final all match). Dumping the GDN substage per token showed the divergence at the scan output `o`, while everything before it (qkv through the k/v projections) matched. In other words, only the head correspondence inside the scan is wrong.
  - Way out: recompute with the head mapping changed, and confirm that **only strided (`kh = h % nk`, equivalent to llama.cpp's `ggml_repeat`) matches** (9B: cos ≥ 0.99984, median 0.99998; 27B: cos 0.99999 on all tokens). The blocked (`h * nk / nv`, equivalent to HF's `repeat_interleave`) introduced in A2 diverged. We unified the 2 CPU sites + 3 GPU sites to strided.

- **What we learned / remaining issues**
  - The new test `gdn_gqa_head_mapping_is_tiled` (synthetic GDN, dk=dv=2, nk=2 / nv=4) pins the strided convention. Assertions guarantee that the input cannot be confused with blocked. workspace 189 passed.
  - E2E: the 27B (uniform Q4_0, int8) answers correctly (`The capital of France is **Paris**.`, md5 `a021580f…` identical on 2 consecutive runs), and Japanese too (`1483c4fe…`). The 9B has new baselines `feea9444…` / `da0b1f57…`.
  - **The PTX was not regenerated** (there is no nvcc in this environment), so the GPU path still runs the old blocked PTX. Regeneration and GPU E2E re-verification are for a human.
  - Remaining verification: HF uses `repeat_interleave` (blocked) while GGUF + llama.cpp are cyclic. For both to be right, the GGUF conversion would have to be reordering the k/q rows, but the converter has not been checked directly. This engine reads only GGUF, so the correctness of the convention for that GGUF is settled by measurement.

- **What worked next**: the case "a one-line misinterpretation such as head mapping breaks the output of the whole model" became the discipline of "cross-check not only the shapes but also the semantics against the reference implementation". `[LCHK]` and `[GDUMP]` are kept permanently behind env gates and reused in later investigations.

---

### s46: prefix cache never hits once on a recurrent model (2026-10-17, `262bd85`)

- **Intent**: Even though A4 added paired restore of the GDN state, there was an observation that on the real model (Qwen3.5) the prefix cache never hits even once. Determine the cause and fix it.

- **What changed**
  - `262bd85` — `tokenizer.rs` (`apply_chat_template_base`), `engine.rs` (`ChatSession.gen_prompt_start`, split prefill), `prefix_cache.rs` (longest selection in `best_full_match`).

- **Stumble**
  - Symptom: a full prefill every time even in multi-turn. `best_full_match` never succeeds.
  - Cause: the chat template ends the prompt with the generation prompt every time. In the next turn's rendering that part is replaced by the assistant message body, so turn_{n+1}'s prompt does not contain turn_n's prompt as a token prefix (confirmed on the real 9B: turn2's prompt branches at the last 2 tokens of turn1, `<think>\n`). An A4 entry keyed on the whole prompt will never match.
  - Way out: add a function that returns the "base" rendered with `add_generation_prompt: false` (up to just before the generation prompt = the history part that also appears as-is in the next turn), and split the prefill into 2 forwards `[base][generation prompt]` only when the base actually becomes a token prefix of the prompt. Evict the KV + GDN snapshot at the base boundary and do not create an entry for the whole prompt (do not accumulate entries that will never hit). When there are several full-match candidates, choose the longest.

- **What we learned / remaining issues**
  - Split contract: `forward(a) + forward(b)` is bit-identical to `forward(a++b)` (GEMM is an i32 accumulation that is row-independent, and GDN / attention are token-sequential, so splitting does not change the accumulation order).
  - Tests +5 (synthetic model) + an `#[ignore]` test on the real 9B (3 turns hit an Entry and each turn is completely identical to a cold full prefill. The reference is obtained by running in reverse order — in forward order the reference side would hit the cache too. 85.8 s to pass). workspace 242 passed.
  - Additional stumble: the first version of the test compared snapshots at different positions wholesale and was structurally invalid. We fixed it to the correct contract (prefix comparison at the base boundary + for GDN, comparison at the same position after the split completes). The failure output before the fix showed that the overlapping region was an exact prefix match, confirming it was a test defect rather than the numeric path.
  - Remaining issue: measuring the multi-turn effect on a real GPU.

- **What worked next**: the design "take the cache key not at the whole prompt but at a boundary that also appears in the next turn" carried over as-is into the verification that base detection does not break when thinking=off (the s47 test).

---

### Q4_0 native path — the 27B runs at 18.4 GB / 178-186 ms/token (2026-10-20, `c51d6a3`)

- **Intent**: On the int8 path the 27B needed VRAM 31,977 MiB and 669 ms/token of decode, which was not practical. We stop the double work of expanding the GGUF 4-bit blocks to fp32 and repacking them into int8, and instead hold the weights as Q4_0 blocks and run GEMV / GEMM with dedicated kernels.

- **What changed**
  - `c51d6a3` — `Linear::Q4` and `Q4Linear`. A block is 32 elements = f16 scale 2 B + nibbles 16 B = 18 B. Element c (0..15) is the low nibble and c+16 the high nibble, `y = d*(q-8)`, and `d` can be negative (the same rule as `ggml`'s `quantize_row_q4_0_ref`).
  - CPU reference `matvec_rows_bt_q4_into`; on the GPU `dno_gemv_q4_0` / `dno_gemm_q4_0`. The layout is SOA (`nib` 16 B aligned uint4, `sc` f32). Sequential f32 accumulation, so deterministic.
  - A weight registry dedicated to Q4, and a Q4 branch in `concat_rows` (keeping the gate+up fusion).

- **Stumble**
  - Symptom: the speed premise breaks on the CPU fallback path.
  - Cause: the Q4 CPU fallback is a plain loop, and the premise of `gemm_min_m` for int8 (that it lands on the tiled GEMM) does not apply.
  - Way out: always route m > 16 through the tiled GEMM, and do not apply `gemm_min_m` to the Q4 CPU fallback.

- **What we learned / remaining issues**
  - Numeric contract: the difference from the CPU reference is GEMV maxabs 9.5e-6 (within relative 1e-4), and GEMM 0 (bit-identical). Determinism held over 3 consecutive runs.
  - Measured (27B, seed 42): int8 is VRAM 31,977 MiB / 669 ms/token, the Q4_0 native is **18,369 MiB / 178-186 ms/token / prefill 34 tok/s**.
  - Tests are lib 246 plus `cuda_q4_linear_matches_cpu` (m=1/7/16/300, `--ignored`), anchors 3/3 MATCH.
  - Remaining issue: TQ1_0 and IQ1_M decoding are unsupported so UD-IQ2_XXS cannot be loaded. The path that passes GGUF blocks through without expanding them is also unimplemented. Note that the type numbers recorded at this point (TQ1_0 = 29 / IQ1_M = 24) are wrong and are corrected later in K1.

- **What worked next**: the policy "hold the quantization format as it is" became the premise for the later P35 (judging raw-block passthrough) and P42 (holding token_embd as raw blocks). The idea of stopping the double work was itself right; the problem was that "the requantized values do not match the original".

---

### s156: `denno info` reads the dimensions from `text_config` (2026-10-21, `ca5e44a`)

- **Intent**: Qwen3.5's config.json is a multimodal wrapper, and the language model dimensions exist only in the nested `text_config`. Because we looked only at the top level, layers, hidden, vocab and context were all displayed as 0.

- **What changed**
  - `ca5e44a` — the lookup order in `crates/denno-cli/src/main.rs` (`text_config` / `llm_config` / `language_config` / top level).
  - Support for a directory argument (if the repo argument is an existing directory, read `config.json` directly instead of going through the hub cache).

- **Stumble**
  - Symptom: `layers: 0` and so on for the 27B / 9B. `family` also becomes the wrapper's `qwen3_5`.
  - Cause: the reference location was hard-wired to the top level. In a wrapper-format config the language model values are one level down.
  - Way out: decide the lookup order and resolve `model_type` in the same order, so that `qwen3_5_text` is displayed. The flat aliases (`n_layer` / `n_embd` / `n_ctx`) are as before.

- **What we learned / remaining issues**
  - Verification: `local/qwen35-27b` = 64 / 5120 / 248320 / 262144, `local/qwen35-9b` = 32 / 4096 / 248320 / 262144 (both with family `qwen3_5_text`).
  - denno-cli 5 passed, denno-engine lib 262 passed / 1 ignored, no regression.
  - Remaining issue: there is still no distinction between 0 and "unknown" (the next s158 makes it `?`).

- **What worked next**: the knowledge "in a wrapper-format config the values are nested" connected to the realization that the later 9B `tie_word_embeddings` problem (A2) has the same root. By closing the lookup order into a single function, the later display additions (s158) could be built on top of it.

---

### K1: adding TQ1_0 / IQ1_M decoders to kquant (2026-10-21, `2344f67`)

- **Intent**: Make IQ2-family and ternary GGUFs decodable at least on the CPU path. The goal is to be able to load UD-IQ2_XXS.

- **What changed**
  - `2344f67` — decoders for `block_tq1_0` (54 B / 256 elements) and `block_iq1_m` (56 B / 256 elements) in `kquant.rs`.
  - 5 self-contained tests in `tests/dequant_ref.rs`.

- **Stumble**
  - Symptom: implementing with the plan's type numbers (TQ1_0 = 29 / IQ1_M = 24) makes a real IQ1_M be read as TQ1_0 (silent corruption).
  - Cause: the numbers in the plan differed from upstream.
  - Way out: cross-checked against both llama.cpp's `ggml.h` and `gguf-py/gguf/constants.py`, and adopted `GGML_TYPE_I8 = 24` / `GGML_TYPE_IQ1_M = 29` / `GGML_TYPE_TQ1_0 = 34`. All 3 are pinned by tests (`from_id(24)` stays unsupported and returns `None`).

- **What we learned / remaining issues**
  - TQ1_0 packs 5 ternary values into 1 byte in base-3 (digit extraction giving the same result as `((u8)(b * 3^n) * 3) >> 8`). IQ1_M reads `scales[8]` as `u16[4]` and splits the f16 scale across 4 high nibbles (there is no independent `d`).
  - Tests are bit-identical against hand-computed reference blocks (54 / 56 raw bytes and 256 expected values), the type numbers and block size / type size, no panic on degenerate blocks (all zeros, scale 0, f16 NaN / ±Inf, smallest subnormal), and unsupported IDs staying `None`.
  - In a scratch outside the repository, confirmed bit-identical against llama.cpp's `dequantize_row_tq1_0` / `dequantize_row_iq1_m` for 16 blocks (4096 elements). We also reproduced the existing harness's fixture md5 `36acb59a…`, so the toolchain matches too.
  - Remaining: loading a real GGUF has not been done. Updating `ref/dequant_ref.c` and the fixtures is out of scope (the next K2).

- **What worked next**: the discipline "cross-check type ids against upstream, not against the plan" became the premise for later quantization tasks. Getting a number wrong corrupts silently, so pinning all 3 by test paid off.

---

### s158: `denno info` shows unknown as `?` and displays rope / attention / dtype / quantization (2026-10-22, `c95b44c`)

- **Intent**: Unknown dimensions were displayed as `0`, so an unimplemented value could not be distinguished from a genuine 0. Output the structural information from config.json, and if there is a GGUF, also show the most frequent tensor type in 1 line.

- **What changed**
  - `c95b44c` — the `denno-cli` display (`layers` / `hidden` / `vocab` / `context` become `Option<u64>`), and the rope, attention, dtype and quantization lines.
  - Open `GgufFile` only when there is a `*.gguf` directly under the directory, and show the most frequent tensor type as a human-readable name + numeric id in 1 line.

- **Stumble**
  - Symptom: `0` cannot be read as either "unknown" or "really 0". Qwen3.5 also puts `rope_theta` in an unusual place.
  - Cause: the display type was fixed to numeric, and there was no layer absorbing key aliases (another place for the same fact).
  - Way out: values that are absent are `?`, and only when `quantization_config` is absent, `none`. As aliases, if `rope_scaling` has no `type` then `rope_type`, and `rope_theta` is also read from `rope_parameters.rope_theta` (not fabricating a value — if it is absent it stays `?`). For the GGUF most frequent value, on a tie the smaller ggml type id is chosen (so it does not change between runs).

- **What we learned / remaining issues**
  - Measured (CPU only): `denno info data/qwen35-9b` = 32 / 4096 / 248320 / 262144, `rope_theta: 10000000` (from `rope_parameters`), `attn_heads: 16` / `kv_heads: 4` / `head_dim: 256`, `quant: Q8_0 (8)`. `data/qwen35-test` = 24 / 1024 / 248320 / 262144, `attn_heads: 8` / `kv_heads: 2` / `head_dim: 256`. Both have no `torch_dtype` and no `rope_scaling` in the config, so `?` (`?` means "absent", not 0).
  - denno-cli 13 passed (was 5). We did not snapshot the whole output; we cut the resolution and the stringification into functions and verified those.
  - Not measured: the test counts of engine / server, the GPU anchors, and the behaviour on the real machine (structurally unchanged because the numeric path was not touched, but not measured).

- **What worked next**: the policy "do not write a guessed default value" was maintained in the GGUF fallback (s164) and the dominant type decision (s168). Stricter display later became the basis for finding the 27B layer-count mismatch (s175).

---

### K2: pinning TQ1_0 / IQ1_M bit-identical against the reference implementation and moving the bitexact test into the default gate (2026-10-22, `f6f03fd`)

- **Intent**: The 2 types added in K1 were not in the fixtures, so they had never been checked against the reference implementation even once. Rebuild the fixtures with 20 types, remove `#[ignore]`, and put them into the default gate.

- **What changed**
  - `f6f03fd` — added sections for TQ1_0 (type 34) and IQ1_M (type 29) after the existing 18 types in the generator `ref/dequant_ref.c`.
  - Updated `tests/data/dequant_ref.bin` to 20 sections / 267,381 bytes / md5 `fe45c5bf…`. `ref/README.md` (regeneration procedure and attribution).

- **Stumble**
  - Symptom: if the expected values are computed by hand, the test itself rubber-stamps a misinterpretation of the block structure.
  - Cause: IQ1_M refers to a large table called `iq1s_grid`, so carrying it ourselves carries the risk of a transcription error.
  - Way out: we used as expected values **the output of calling llama.cpp's `dequantize_row_tq1_0` / `dequantize_row_iq1_m` linked directly**. The block structs use `block_tq1_0` / `block_iq1_m` from `ggml-common.h` via `sizeof`, and for `iq1s_grid` the table that `ggml-quants.c` pulls in with `GGML_COMMON_IMPL_C` worked as-is (no extra initialization or stub needed).

- **What we learned / remaining issues**
  - Verification of the generation pipeline: regenerating the old fixture with the generator from before adding the 2 types gave md5 `36acb59a…`, a complete match (232,815 bytes identical, excluding the section count). After adding the types, the byte strings of the existing 18 sections are unchanged.
  - Tests are 6 passed (all 20 types BIT-EXACT. TQ1_0 / IQ1_M are 4096 elements each). We prepared 4 failure messages (no fixture / section count mismatch / missing required type / truncated fixture).
  - **Optimization-level dependence (measured)**: generating with `-O0` makes IQ3_XXS's f32 output differ from `-O2` (the bytes from 168337 on differ at 0x7f / 0xff). The committed fixture was generated with `-O2`, so regeneration must always use `-O2`. TQ1_0 / IQ1_M were identical under `-O0` and `-O2`.
  - License: the source of reference is llama.cpp's `ggml-quants.c` / `ggml-common.h` (MIT). This repository is MIT OR Apache-2.0. The generator only links and calls the functions, with no code taken in, and the fixture is output data.

- **What worked next**: "make the expected value the reference implementation's own output" took hold as the only honest method around quantization. The discovery of the optimization-level dependence stayed in the README as a precondition for regenerating the fixture.

---

### K3: pinning the Q4_0 edge blocks bit-identical against the reference implementation, and pinning the accumulation order of the CPU gemv (2026-10-22, `45a9696`)

- **Intent**: The production 27B weights are Q4_0 (ggml type 2). The K2 fixture already had a pseudorandom Q4_0 section that had been checked, but there is no guarantee that 16 random blocks necessarily include "negative d", "all-zero nibbles" and "maximum nibbles". Add the boundary values deterministically and pin them.

- **What changed**
  - `45a9696` — added `emit_q4_0_edge()` to the generator and one `Q4_0_EDGE` section (21 sections in total / 268,278 bytes / md5 `eeff5cb0…`).
  - 2 tests (`q4_0_dequant_bitexact_vs_llama_cpp_edge_blocks`, `q4_gemv_accumulation_order_is_sequential_in_block_order`).

- **Stumble**
  - Symptom: we thought we were creating edge cases, but the K2 random fixture already contained Q4_0 (the premise was wrong).
  - Cause: we planned on the premise "there is no Q4_0 section".
  - Way out: after correcting the premise, we changed to a policy of adding only the boundary values that randomness cannot guarantee. 6 blocks (d = -1.5 with low 0..15, d = +0.5 with all nibbles 0 → q = -8, d = +0.5 with all nibbles 15 → q = +7, d = -0.75 with all 0xFF, d = +2.0 with different upper and lower patterns, d = +0.0 where the product is -0.0). All the d values are exact in f16.

- **What we learned / remaining issues**
  - Pipeline verification: regenerating with the generator from before K3 gave md5 `fe45c5bf…`, a complete match, and after the addition not a single byte of the raw bytes or the f32 output of the existing 20 sections changed.
  - Pinning the accumulation order: one output for k = 96 (3 blocks) is pinned to `0xca7b15f7`. We also confirm at the same time that the pairwise sum of the same term sequence `0xca7b15f9`, the 2-accumulator sum `0xca7b15fa`, the reverse order `0xca7b15f9`, and the single rounding (fma equivalent) `0xca7b15f5` are **all different values**, so any change of order or fusion must fail. The implementation, as read, was already in the documented order and the change was 0 lines.
  - Gate: engine lib 270 passed / 1 ignored. Only the first run had `prof::tests::attn_core_is_split_into_gdn_and_full_rows` fail, but it passed in a standalone run and on re-run (a known flaky caused by a shared global counter). dequant_ref 7 passed, server lib 37 passed.
  - Not measured: the accumulation order and bit-identity of the Q4_0 path on the GPU, and re-verification of types other than Q4_0 (Q4_1 / Q5_0 and so on).

- **What worked next**: the policy "pin the order in a test even when the implementation is already correct" became the grounds for claiming byte identity in the later P37 (row-chunk quantization).

---

### s164: `denno info` GGUF fallback (2026-10-22, `8a0525b`)

- **Intent**: In a GGUF-only directory without config.json, `denno info` failed and could not output a single dimension. Fill them in from the GGUF metadata.

- **What changed**
  - `8a0525b` — `gguf_dims(dir)` and `info_resolve(dir)` in `denno-cli`. Added `GgufValue::as_array_len()` to `gguf.rs` additively.

- **Stumble**
  - Symptom: it exits with `config.json を開けません` and the line itself is not printed.
  - Cause: dimension resolution was fixed to config.json, and there was no path that reads GGUF.
  - Way out: read `general.architecture`, then `<arch>.block_count` → layers, `<arch>.embedding_length` → hidden, `<arch>.context_length` → context. The arch name is assembled from the metadata value (not hard-coded). vocab is the array length of `tokenizer.ggml.tokens`, and if absent `<arch>.vocab_size`. If config.json exists it wins, and GGUF fills only the missing parts (`.or()` per dimension).

- **What we learned / remaining issues**
  - Measured (GGUF-only directory): `layers: 65` / `hidden: 5120` / `vocab: 248320` / `context: 262144`, `family: ?`.
  - **Note (measured)**: `qwen35.block_count` in this file is 65, while `num_hidden_layers` in the config.json of the same model is 64. The difference of 1 is `nextn_predict_layers = 1` (the MTP layer), which the GGUF block_count includes. Because we output block_count as-is per the spec, only the layer count deviated from the engine's actual layer count (corrected in the next s175).
  - denno-cli 13 → 18 passed, engine lib 274 passed / 1 ignored, server lib 37 passed, dequant_ref 7 passed.

- **What worked next**: the understanding "GGUF metadata can stand in for config.json, but the semantics are not the same" was used directly in the handling of 65 vs 64 layers (s175).

---

### s168: the `quant:` line of `denno info` reports the quantization type (2026-10-22, `73d2e10`)

- **Intent**: For the 27B q4 GGUF it displayed `quant: F32 (0)`. F32 is where rmsnorm / bias live, and in a real file the number of those exceeds the quantized weights (the 27B Q4_0 file has F32 456 / Q4_0 352 / Q5K 48 / Q4_1 8 / Q6K 1 / Q8_0 1). The most frequent value over all tensors lets F32 win.

- **What changed**
  - `73d2e10` — added `dominant_tensor_type()` and replaced the vote of `gguf_common_type()` from `most_common_type()`.
  - The meaning of `most_common_type()` (most frequent, ties go to the smaller ggml type id = deterministic) and the display format `<name> (<id>)` are unchanged.

- **Stumble**
  - Symptom: F32 appears on the line asking for the kind of quantization.
  - Cause: "most frequent" did not carry the meaning "quantization type". F32 is structurally always present in a fixed number.
  - Way out: if there is at least 1 non-F32 tensor, exclude F32 from the vote. A GGUF with only F32 stays F32 as before (do not make it an empty vote). 0 tensors is `None` → `?`.

- **What we learned / remaining issues**
  - denno-cli 18 → 23 passed. We confirmed that reverting the change makes `gguf_quant_value_skips_f32_storage_tensors` fail with `left: "F32 (0)"` / `right: "Q4_0 (2)"` (confirming the test has teeth).
  - Real machine (CPU, real file): `quant: F32 (0)` → `quant: Q4_0 (2)` (no other lines changed).
  - `layers` (65 vs 64) and `family: ?` are deliberately unchanged (awaiting an operational decision).

- **What worked next**: the judgement "the display shows only facts, but which statistic to take must match the meaning" became the same kind of point as the later decision on the layer count (inference trunk).

---

### s175: aligning the layer count of `denno info` to the inference trunk for a GGUF-only repository (2026-10-23, `d787f66`)

- **Intent**: Re-checking the premise, HEAD was already filling the dimensions from GGUF (the "0 is displayed" observation was from a build before s164). The only real bug left was the layer count.

- **What changed**
  - `d787f66` — `gguf_dims` only. When there is no config.json (or the layers key is missing), the GGUF-derived layer count becomes `<arch>.block_count - <arch>.nextn_predict_layers`. If there is no `nextn_predict_layers`, block_count as-is. An abnormal value where the subtraction is 0 or less becomes `None` = `?`.

- **Stumble**
  - Symptom: `layers: 65` in a GGUF-only directory. The engine actually runs 64 layers.
  - Cause: the 27B GGUF has `block_count = 65` (inference trunk 64 + MTP 1). The engine takes the config's `num_hidden_layers` as the truth and skips the excess `blk.64` as a nextn layer, so the number of layers it runs is 64. Only the display said 65.
  - Way out: subtract so that the display matches "the inference trunk the engine runs". If one wants to see the raw block_count, the right thing is to add another line (e.g. `mtp: 1`); we left that alone as out of scope.

- **What we learned / remaining issues**
  - Tests: engine cuda build lib 321 passed / 37 ignored (unchanged), denno-cli 23 → 25 passed (+2: the MTP subtraction, and the abnormal value nextn == block_count, no nextn, config precedence).
  - Real machine (CPU, real file): `layers: 65` → `layers: 64` in a GGUF-only directory (the hidden / vocab / context / quant lines unchanged). A directory with config.json stays 32, and a config.json + MTP GGUF also stays at the config's 64.
  - Not measured: GPU anchors, ms/token, nsys (structurally unchanged because not one line of the numeric path was changed, but not measured).

- **What worked next**: the judgement "when the same fact has 2 representations, which one to display is decided by operational meaning" remained as a point to carry over to the index side as well.

---

### P35: judging raw-block passthrough for GGUF Q4_0 on real data (2026-10-23, `98dd8e9`)

- **Intent**: For the remaining item of the Q4_0 native path, "stop the GGUF → fp32 → Q4_0 requantization and stack the raw blocks as they are", first check on a real file whether it is bit-identical.

- **What changed**
  - `98dd8e9` — `Q4Linear::from_gguf_blocks` in `quant.rs` (a validating copy; fail-closed with `None` if the shape does not match), and the env gates `gguf_passthrough_*` (allowed types = currently 0).
  - `GgufHeader` in `gguf.rs` (a header view without mmap. It goes through the same parsing functions as `GgufFile`, so there is one set of parsing rules), and routed the 10 2-D linear weights plus the untied `output.weight` in `qwen35.rs` through the gate.

- **Stumble**
  - Symptom: stacking the raw blocks changes the values.
  - Cause: the Q4_0 blocks in the real file were not produced by the rule of `ggml`'s `quantize_row_q4_0_ref`. Dumping 1 block gave `d = 4.543304e-3` and nibble range 1..12 with `max|q-8| = 7`, but the ref rule always maps the element with the largest absolute value to nibble 0 (`|q-8| = 8`). Therefore the requantized `d'` becomes `max|y|/-8 = d*7/8`, and the ratio was 1.1426〜1.1433 on all 14 tensors (= 8/7 = 1.142857, the width of f16 rounding).
  - Way out: accept that they do not match and leave the allowed-type list empty. Even with the gate ON it goes through the conventional fp32 path, so not a single bit of the numerics changes.

- **What we learned / remaining issues**
  - Measured (an `#[ignore]` proof test, CPU): parse only the header (10,996,700 B) without mmap, read the tensors 4 MB at a time on row boundaries, and compare `dequant_row` → `Q4Linear::quantize` byte by byte (peak a few tens of MB, 5.1 s in total). Q4_0 is **NOT identical** (352 tensors / 12.441 GiB, 16 sampled, 2 matching = `token_embd.weight` and `blk.64.ffn_up.weight`). F32 456, Q4_1 8, Q5K 48, Q6K 1 and Q8_0 1 are out of scope because they are not in the Q4_0 layout.
  - Predicted effect (**estimated, not measured**): if Q4_0 were allowed, the fp32 expansion that could be saved is 88.47 GiB (12.441 GiB × 4 B/elem ÷ 0.5625 B/elem = 7.111×. It is a per-tensor temporary allocation, not simultaneous residency. The largest is `token_embd.weight` at 5.086 GB). From the measured rate of the proof test, 0.77 GiB / 5.1 s, the CPU side of loading would drop by about 82 s. However, since the allowed types are 0, the actual saving is 0.
  - Tests: engine cuda build lib 323 passed / 38 ignored.

- **What worked next**: the fact "requantization changes the values" is later reinterpreted by P38 as "dequantization does not change them". Not confusing the two led to the 5 GB reduction of token_embd.

---

### P37: bounding the fp32 scratch area at load time for large 2-D weights (2026-10-23, `1bc1702`)

- **Intent**: Loading `token_embd.weight` (1.271 G elements for the 27B) built the f32 of the whole tensor (5.086 GB), built 1 more copy with `embed.clone()` for the tied lm_head (a further 5.086 GB), and only then quantized. This tensor alone peaked at about 10.9 GB. It was the main cause of a 60 GB host being OOM-killed while the server was resident — a load-time memory problem, not a numeric one.

- **What changed**
  - `1bc1702` — `Q4Linear::zeros` / `quantize_rows_into` and `Quant8Linear::zeros` / `quantize_rows_into` in `quant.rs`. The body of the existing `quantize` was extracted into the form "`zeros` then call `quantize_rows_into(0, ..)` once", so old and new go through the same code (byte identity is structurally guaranteed).
  - Row-chunk-driven `gguf_rows_quantized_chunked`, `ChunkedQuantStats`, the env gate `DENNO_GGUF_CHUNKED_QUANT` (default ON; `0` / `false` / `no` / `off` for the conventional path), and `chunk_rows_for`.

- **Stumble**
  - Symptom: chunking the quantization risks a block being cut in the middle of a row and changing the values.
  - Cause: whether the row and block boundaries coincide depends on the quantization type.
  - Way out: cut on row units (a row = k elements). Rows are contiguous in GGUF (dim0 = k is innermost), so a row boundary is a block boundary (verify `k % blck_size == 0`, and fall back to the conventional path if not satisfied). Both the Q4_0 32-element block and the K-family 256-element superblock close within a row. The chunk width is the number of rows where one chunk's f32 is 16 MiB (4 M elements) (k=5120 → 819 rows, 6144 → 682 rows, 17408 → 240 rows).

- **What we learned / remaining issues**
  - Measured (proof test, CPU): comparing 10 tensors gave **10/10 bytes identical** (`token_embd` Q4_0, `output` Q6K, `ffn_up` Q4_0, `ffn_gate` Q4_0, `ffn_down` Q4_1, `attn_q` Q4_0, `attn_qkv` Q4_0, `attn_v` Q4_0, `attn_output` Q4_0, `ssm_out` Q5K). The int8 side comparison also matched on all 5. The peak f32 scratch (computed value) is 20.21 MB chunked vs 5,085.59 MB overall (252×). 23.5 s.
  - Predicted effect (**estimated, not measured**): the load-time peak for `token_embd` goes from 10.887 GB to 5.820 GB, about 5.07 GB less. For ffn / attn it is at most 356.5 MB → 19.1 MB per tensor. Load time and RSS on the real machine are out of scope.
  - Tests: engine cuda build lib 327 passed / 39 ignored.

- **What worked next**: the distinction "show the peak of the scratch area as a computed value, and take measurements separately" was used as-is in the later P42 REPORT. The fact that the 5.086 GB resident table remained is recorded as an issue at this point.

---

### P38: deciding whether the 5.086 GB f32 table of `token_embd` can be replaced by per-row dequantization (2026-10-23, `cf917e5`)

- **Intent**: After P37 the f32 gather table of `token_embd.weight` (5.086 GB) was still resident. It is the largest host RAM item in the loader, so decide with numbers whether it can be removed. There is no change to product code, only analysis and a proof test.

- **What changed**
  - `cf917e5` — the proof test `gguf::tests::q4_row_dequant_matches_embed_table_real_gguf` (`#[ignore]`, CPU, seek + read_exact without mmap).

- **Stumble**
  - Symptom: we had read P35's "the Q4_0 blocks in this file cannot be reproduced by requantization (`d' = d*7/8`)" as "the table cannot be replaced".
  - Cause: confusing **requantization** (f32 → Q4_0 encoding) with **dequantization** (Q4_0 → f32). The table is built by merely dequantizing the file's blocks.
  - Way out: confirmed that `dequant_row` closes within a block (it carries no state between blocks) and that `row_bytes = 160 × 18 = 2880` is an integer multiple of the 32-element block (row boundary = block boundary). Therefore dequantizing a single row is the same computation of the same 160 blocks over the same bytes, and the f32 bits match.

- **What we learned / remaining issues**
  - Measured: `token_embd.weight` k=5120 / n=248320 / Q4_0 (raw 715.2 MB, table 5,085.6 MB). Comparing 40 rows (0, 1, 2, the chunk boundaries 818 / 819 / 820 / 1637〜1639, evenly spaced rows, 248318 / 248319) across 32 chunks gave **40/40 match**. Furthermore a sweep over all 248,320 rows gave **difference 0** (3.30 s). All comparisons are on the f32 bit pattern.
  - Cost (measured on this CPU with a warm page cache): 1 row = a 2,880 B pread + `dequant_row`, giving 3.09 µs/token for B=1, 24.93 µs for B=8 and 103.91 µs for B=32. Pure dequantization is 1.9 µs/row; the rest is pread and allocation. The peak scratch is 23.4 KB. The 3.09 µs/token for B=1 being 0.018 % of the 17.30 ms/token KPI is **arithmetic**, not a GPU measurement.
  - Judgement: bit-identical, so it is worth implementing. The resident reduction is 5,085.6 MB if rows are dequantized from the mmap'ed file region (page cache = reclaimable), or 4,370.4 MB if a 715.2 MB copy of the raw blocks is made resident. A row cache is unnecessary (even 64 rows is 1.31 MB).
  - **The Q4 copy of `lm_head` cannot be the source** — that one is requantized (P35's allowed types stay empty) and is not the file's blocks, so dequantizing it would not be bit-identical.
  - Tests: engine cuda build lib 327 passed / 40 ignored. Note that `tests/kernel_invariants.rs::gdn_kernel_shared_layout_matches_host_smem` was red on pristine HEAD (pre-existing, unrelated).

- **What worked next**: the order "write the proof test first and decide the implementation from its result" determined the design of the next P42 as-is. The sketch (`EmbedRows::{F32, Blocks}`, the 3 gather sites, a gate defaulting to ON) became the shape of the implementation.

---

### P42: replacing the resident 5,086 MB f32 table of `token_embd` with per-row dequantization (2026-10-23, `955690e`)

- **Intent**: In the host RSS of the 27B q4 serve, the f32 gather table of `token_embd.weight` occupied 5,085,593,600 B. P38 proved on a real file that per-row dequantization is bit-identical, so we implement it.

- **What changed**
  - `955690e` — `embed_blocks_and_linear_chunked` in `gguf.rs` (allocate the file's own raw blocks `n*row_bytes` instead of the f32 table, and build the quantized copy for the tied lm_head with the same row chunks as before) and the eligibility check `embed_blocks_eligible` (type is Q4_0, `k % 32 == 0`, `nbytes == n*row_bytes`, `vocab*e == elems`).
  - Replaced `embed: Vec<f32>` in `qwen35.rs` with `EmbedTable { F32, Q4Rows { ... } }` and moved the readers (decode 1 row / prefill n rows / chain upload 1 row / test helpers) to `row()` / `row_into()`. Restore is `kquant::dequant_row_into` (new, same body as `dequant_row`), with no requantization at all.

- **Stumble**
  - Symptom: replacing the table risks the Fp32 path (tied fp32 lm_head) silently allocating 5 GB.
  - Cause: the readers of the table are scattered across 3 places (decode / prefill / GPU chain upload), and the premise differs per path.
  - Way out: make the `F32` variant byte-identical to the current `&w[i*e..][..e]` and the fail-closed destination when the premise breaks. R4 / R5 (tied fp32 lm_head, full-row matvec) are unreachable when `quant != Fp32`, so they panic with a reason if called with `Q4Rows` (do not silently allocate 5 GB). Default ON; `DENNO_EMBED_ONDEMAND=0|false|no|off` returns to the conventional f32 table.

- **What we learned / remaining issues**
  - Grounds for bit identity: for the 27B Q4_0 (k=5,120 / n=248,320 / row_bytes=2,880) we compared all 248,320 rows of `EmbedTable::Q4Rows` against the loader's chunk dequant and got a `to_bits()` difference of 0 (measured 3.19 s, resident 715.2 MB, pread 0.66 s). The unit tests on synthetic blocks include row 0, the last row, and rows crossing chunk boundaries, and also pin agreement with the `F32` variant.
  - Tests: engine cuda build lib 335 passed / 41 ignored (baseline 330 + 5 new), `--test kernel_invariants` 11 passed, non-cuda lib 291 passed.
  - **Unverified (estimated)**: the host resident reduction of **-4,370.4 MB** is the difference in allocated bytes (5,085,593,600 → 715,161,600 B), not an RSS measurement. The effect on the KPI (decode +0.003 ms/token = 0.018 %, prefill +9.8 ms @3,161 tokens / +39.7 ms @12,850 tokens = 0.10 %) is a value predicted from P38's 3.09 µs/token and has not been measured on the real machine. The 1-row restore on the product path is 1.95 µs/row measured on the CPU, which is smaller than the 3.09 µs used for the prediction (the prediction is on the conservative side). Re-acquiring the anchor md5s has not been done.

- **What worked next**: the distinction "for a change that reduces residency, show the reduction in allocated bytes and measure RSS and the KPI separately" took hold. The pattern of safely landing a change that does not touch the numeric path, with a proof of bit identity and a fail-closed design, was completed here.

---

## Rejected approaches

Judgements that looked sound at the time but were discarded later. The reasons for rejection are written as the facts observable at that point.

### The blocked head mapping of A2 (rejected in A2 `9183e7e` → the GQA fix `e266090`)
- Why adopted: the k-head that a v-head refers to was computed as `h * nk / nv` (equivalent to HF's `repeat_interleave`). Layer 0 numpy verification showed an error difference of blocked 0.27 % / interleaved 73 %, which settled blocked as correct.
- Observation: the logits at the final prefill position of the 27B diverge from the reference implementation. A per-token dump of the GDN substage showed that only the scan output `o` differs. Only strided (`h % nk`, equivalent to llama.cpp's `ggml_repeat`) matches (9B: cos ≥ 0.99984, 27B: cos 0.99999).
- Rejected: unified the 2 CPU sites + 3 GPU sites to strided. The 9B had merely been "tolerating" blocked, and it was wrong as a convention.

### Treating `output_gate_type: "swish"` as a change to the numeric path (rejected in A3 `646c22a`)
- The form we nearly adopted: since the key exists only in the 27B config, interpret it as the output gate being implemented differently from the 9B and change the path.
- Observation: cross-checking 3 reference implementations (transformers main / vLLM / llama.cpp master) showed that none of them consumes this key. The full attention output gate is sigmoid and GDN's z gate is silu (= swish), which matched the existing behaviour.
- Rejected: no change to the numeric path. When the key is present we only log info, and an unknown value only warns.

### Explaining the 27B breakdown by the GPU drift amplification theory (rejected in the GPU E2E investigation → the 27B quality investigation `7b07870`)
- Why adopted: for the 9B, the per-layer checksum showed a relative ssq difference of about 4e-7 at L00, which greedy amplification explained. We tried to read the 27B in the same framework.
- Observation: first, running the same payload on the CPU reproduced the breakdown for the 27B too. Before any amplification, it already differed from the reference implementation at generation step 0 (= the logits of the final prefill position) (llama.cpp's top-1 is `The`, this engine's is `What` at logit 15.79 with `The` at top-2). The GGUF quality and the chat template had already been rejected.
- Rejected: for the 27B we switched to the order "reproduce on the CPU → compare the step 0 distribution on the same input as the reference implementation", which reached the GDN head mapping. Drift amplification remains as the explanation of the 9B GPU event.

### The reading that "Q4_0 passthrough is impossible" in P35 (rejected in P35 `98dd8e9` → P38 `cf917e5`)
- The reading adopted: stacking the raw blocks changes the values (`d' = d * 7/8`, ratio 1.1426〜1.1433), so the Q4_0 blocks of this file cannot be used as they are. The allowed-type list was left empty.
- Observation: P38 showed that this is about **requantization** (f32 → Q4_0 encoding) and not about **dequantization** (Q4_0 → f32). The table is built by merely dequantizing the file's blocks, with difference 0 over all 248,320 rows (the 40/40 sample comparison also matched).
- Rejected: withdrew "the table cannot be replaced". P42 replaced it with per-row dequantization, taking residency from 5,085.6 MB to 715.2 MB.

### The single-file premise of P7 (rejected in P7 `b6c6c5d`)
- The premise in use: the weights fit in 1 `*.safetensors` and can be looked up by tensor name from there.
- Observation: the 7B class is split into `model.safetensors.index.json` + multiple shards, which a loader premised on a single file cannot resolve (the 7B cannot be loaded). As the same kind of mistake, the 3B was reusing the embedding even though lm_head is a separate tensor.
- Rejected: moved to looking up the tensor name → shard mapping from index.json and resolving through `WeightStore`. When untied, prefer the `lm_head` tensor.
