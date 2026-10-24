# All phases in date order

> [Index](README.md) / Prev: [Volume 8 Measurement, verification, operations](70_infra.md)
> Japanese original: [../90_phase_index.md](../90_phase_index.md)

This document is a **mechanically built index**. It takes the `## Phases covered` table of each volume,
lines them up and re-sorts them by date (137 rows). **The authoritative content is the body of each volume**;
this is a table for looking up "when, in which commit, and in which volume it is written".

- The same phase can appear in two volumes (P11 is API and decode, s53c is the accepted form of the API and the isolation of quality).
  This is not duplication but a description from a different angle, so both rows are kept.
- Dates are the commit author date. Commits from the predecessor repository (P8–P12 in Volume 4) do not exist in this repository.
- No numbers are written here. Numbers are in the body of each volume and in `docs/KPI.md`.

### Period 1: Bootstrap — running on CPU alone

| Date | phase | commit | What was done | Vol |
|---|---|---|---|---|
| 2026-10-11 | Initial implementation | `7f85369` | Single binary (cli / engine / server) and the skeleton of an Anthropic Messages API-compatible server | 01 |
| 2026-10-11 | Real-model bring-up | `b718e59` | Fixed the GEMM stride and the QKV bias of the Qwen2 family; GPT-2 / Qwen2.5-0.5B now pass | 01 |
| 2026-10-11 | First speed work | `67280c9` | Turned attention into sgemm, `target-cpu=native`, SSE event names and ping | 01 |
| 2026-10-11 | P1 | `831246b` | Reuse of the KV cache across requests and a single-entry prefix cache | 01 |
| 2026-10-11 | Docker deployment | `f1c3bde` | Multi-stage build and automatic model fetch at startup | 01 |
| 2026-10-11 | P2 / CI | `6faf889` | 48 tests, multi-entry LRU prefix cache, CI on push | 01 |
| 2026-10-11 | CI fix | `37cc6b0` | After SIGILL, pinned `target-cpu` to `x86-64-v3` | 01 |
| 2026-10-11 | P3 | `ffabb50` | int8 weight-only quantization (memory halved, speed flat) | 20 |
| 2026-10-12 | P11 | `1a7fee6` | U+FFFD holdback in streaming detok and an update of the gpt2 exactness criterion | 10 |
| 2026-10-12 | P7 | `b6c6c5d` | Loading split safetensors and the untied lm_head | 20 |
| 2026-10-12 | P8 | `119dd92` | Trimming the sampler's fixed cost of 3.5 ms/step with a partial sort | 30 |
| 2026-10-12 | P9 | `1bbffff` | SIMD fp32 batch matvec, bit-identical without using FMA | 30 |
| 2026-10-12 | P10 | `4e2428f` | Parallelized batch attention over session × KV head | 30 |
| 2026-10-12 | P11 | `16319c7` | Fixed mojibake with U+FFFD holdback in streaming detok | 30 |
| 2026-10-12 | P12 | `55c1de0` | probs@V to row-fused sgemm and direct strided access | 30 |

### Period 2: To the GPU — from int8 kernels to Qwen3.5

| Date | phase | commit | What was done | Vol |
|---|---|---|---|---|
| 2026-10-13 | P17d | `121aa3a` | Moved elementwise ops to the GPU and made one decode step resident | 30 |
| 2026-10-13 | P17e | `c889336` | Made the KV cache demand-based and growable, slots from 4 to 32 | 30 |
| 2026-10-13 | P17f | `d7d126e` | Dropped the contiguous copy of scores and read them strided | 30 |
| 2026-10-13 | P17g | `bf2a3c6` | Moved attention to the GPU and relaxed the exactness contract only here | 30 |
| 2026-10-13 | P17h | `b090d03` | Moved rope to the GPU, one sync per step at logits | 30 |
| 2026-10-13 | P17b | `197a06c` | Connected int8 linear to the real path and decided the CPU line for prefill | 40 |
| 2026-10-13 | P17i | `f2c18315` | Moved the linear layers of long prefill to the GPU with a tiled GEMM | 40 |
| 2026-10-14 | P19 foundation | `49ac073` | GGUF container reader and decoding of 18 k-quant / IQ types | 20 |
| 2026-10-14 | P19-B | `0bbfa79` | CPU reference of Gated DeltaNet | 20 |
| 2026-10-14 | P19-B2 | `feab1c8` | Qwen3.5 model integration and E2E | 20 |
| 2026-10-14 | P17j | `cededce6` | O(m²) CPU attention to a flash-style kernel | 40 |
| 2026-10-14 | P17k | `4bf11f5e` | Made prefill resident in 6 stages starting from the FFN block | 40 |
| 2026-10-14 | P19 | `0bbfa79e` | From CPU reference and a single-shot kernel to a chunked 3-pass | 50 |
| 2026-10-15 | A2 | `9183e7e` | Resolution of the 9B GDN k-head convention and of the untied lm_head | 20 |
| 2026-10-15 | A2.5 | `4fada3c` | Added repetition penalty HF-compatible and a no-op by default | 20 |
| 2026-10-15 | A3 | `646c22a` | Synthetic shape test for 27B and correction of the KV budget | 20 |
| 2026-10-15 | A4 | `bf2c90b` | GDN state snapshot into the prefix cache | 20 |
| 2026-10-15 | GPU E2E regression investigation | `d409d4f` / `ed09722` / `9dc6fc9` | Explained the 9B GPU prefill failure as drift amplification and wrote down the regression criterion | 20 |
| 2026-10-15 | B2 | no commit | Hoisted the V re-read of probs@V and cut LDS by 3.7× | 30 |
| 2026-10-15 | B3 | `401b8f92` | Combined rope for b>1 into one kernel and cut CPU work | 30 |
| 2026-10-15 | B1 | `455dbe0a` | Moved rmsnorm to the GPU and made the inside of a layer device-resident | 40 |
| 2026-10-15 | A2.5 | `4fada3c` | Added repetition penalty HF-compatible and a no-op by default (1.1 is appropriate, 1.15/1.3 are excessive) | 60 |

### Period 3: Break it, fix it, make it measurable

| Date | phase | commit | What was done | Vol |
|---|---|---|---|---|
| 2026-10-16 | tool use | `751dc10` | Accepting tools / tool_choice, chat template conversion, `<tool_call>` extraction | 10 |
| 2026-10-16 | tool_choice constraint | `a16b203` | Force `any` / `tool` at token level with a logits mask | 10 |
| 2026-10-16 | streaming tool_use | `05e5f8f` | Assemble a `<tool_call>` spanning delta boundaries in the SSE assembler | 10 |
| 2026-10-16 | disconnect detection | `e7d3e7a` | Look at SSE `is_closed()` at step boundaries and abort generation | 10 |
| 2026-10-16 | max_tokens default | `9636337` | Apply 4096 only when unspecified (also accept `max_completion_tokens`) | 10 |
| 2026-10-16 | 27B structure validation | `203deed` | Rejection of the out-gate hypothesis, `expect_ne` and MTP exclusion | 20 |
| 2026-10-16 | 27B quality investigation | `7b07870` | CPU isolation and comparison against the llama.cpp reference | 20 |
| 2026-10-16 | GQA head mapping fix | `e266090` | Changed blocked to strided so that 27B answers correctly | 20 |
| 2026-10-16 | C4 | `18c97fe2` | One decode step of qwen35 into a resident device chain | 30 |
| 2026-10-16 | Plan B | `875dd3cd` | Parallelized CPU prefill with rayon and kept bit-identity | 40 |
| 2026-10-16 | GDN GQA head mapping | `e2660908` | Settled the head mapping from blocked to strided | 50 |
| 2026-10-17 | s44 non-stream disconnect | `08e7c9b` | Closed-connection check in respond and non-blocking stdout log | 10 |
| 2026-10-17 | s47 thinking control | `4e89c11` | Accept `thinking: off / brief / full` and pass it to template variables | 10 |
| 2026-10-17 | s46 | `262bd85` | Prefix cache hit for a recurrent model (split prefill) | 20 |
| 2026-10-17 | s49 | `769aa4a` | Moved the 2.4 GB read per step of the fp32 lm_head to the GPU | 30 |
| 2026-10-17 | s50 | `fca2d93` | Delay the D2H of GDN state until the point it is needed | 30 |
| 2026-10-17 | s48 | `a2e5368c` | Full attention of qwen35 in a single kernel | 40 |
| 2026-10-18 | s52 thinking map | `b9d29f8` | Accept the `{"type":"enabled"}` form (do not read `budget_tokens`) | 10 |
| 2026-10-18 | s53c meaning change of brief | `e7246b8` | brief to "thinking disabled + concise instruction", map `enabled` treated as off | 10 |
| 2026-10-18 | s53c | `e7246b8` | brief from "thinking ON + steering sentence" to "thinking disabled + concise instruction" (all 4 steering candidates failed) | 60 |
| 2026-10-18 | s53/s53b | `b1f5655` / `f21a985` | Fixed false positives of the role-label stop (23 tok / 44 tok) with a true line-start criterion | 60 |

### Period 4: Rebuilding measurement — phase accounting and verification devices

| Date | phase | commit | What was done | Vol |
|---|---|---|---|---|
| 2026-10-19 | s53 stop conditions | `f21a985` | Holdback of the role-label stop and default max_tokens 1024 | 10 |
| 2026-10-19 | s56/s57 repetition suppression | `b3eaf71` | Added no_repeat_ngram / frequency / presence as opt-in | 10 |
| 2026-10-19 | temperature cap | `47e1928` | Clamp the effective temperature with `--temperature-cap` | 10 |
| 2026-10-19 | s54 | `b635b820` | Split d2h_state into 6 parts so that the breakdown can be read | 30 |
| 2026-10-19 | s101 / s108 | `96f34fe0` | Ledgered the host round-trips of one step and showed that 12 ms is unrealistic | 30 |
| 2026-10-19 | T5a revert | `083218d6` | Reverted a gate built from a false hotspot | 40 |
| 2026-10-19 | s62 / T5b | `d9fef26c` | Fixed projections with n<64 silently falling back to a single thread | 40 |
| 2026-10-19 | s63 | `6d34d90e` | The bottleneck was not the GPU but a 21.8 s return-value copy | 40 |
| 2026-10-19 | s66 | `ac34b1ea` | The slowness of small-n projections was a side effect of the transposed layout | 40 |
| 2026-10-19 | s68 | `b682c3f5` | D2H through pinned was eating 5× the GEMM body | 40 |
| 2026-10-19 | s69 | `1f588b4f` | Removed the pinned path + to_vec from chain D2H | 40 |
| 2026-10-19 | s70 | `9ec23f3c` | Made FFN a resident chain, activation round-trips from 4 to 2 | 40 |
| 2026-10-19 | s71 | `91b54265` | Recovered s70 having rolled back s69 | 40 |
| 2026-10-19 | s73 | `57405fe9` | Parallelized the prepare of full attention over tokens | 40 |
| 2026-10-19 | s74 | `e0ea110b` | int8 GEMM per-thread tile to 8x16 | 40 |
| 2026-10-19 | s75 | `5bca9b7e` | Fixed smem alignment and row coverage to recover | 40 |
| 2026-10-19 | s80 / s81 | `b69e0190` | Concatenated gate+up at load time into a single GEMM | 40 |
| 2026-10-19 | s84 | `66698f68` | Showed that the "0.4 s" of rmsnorm was a misnomer for the whole block | 40 |
| 2026-10-19 | s61c | `b25529d3` | Split attn_gdn into projection and scan so that it can be measured | 50 |
| 2026-10-19 | s64 | `8e7ff24d` | Split one scan call into alloc/launch/sync/d2h | 50 |
| 2026-10-19 | s67 | `b36df11e` | Reused the workspace across layers and halved alloc | 50 |
| 2026-10-19 | s72 | `4119dce4` | Kept y on the device and chained it through to out_proj | 50 |
| 2026-10-19 | s78/s79 | `85a94052` | Prototype of GPU kernels for the beta/alpha projections (not adopted) | 50 |
| 2026-10-19 | s89/s90 | `ddf7201a` | Folding two launches does not hold structurally | 50 |
| 2026-10-19 | s56/s57 | `b3eaf71` | Added no_repeat_ngram / frequency / presence as opt-in (no-op by default) | 60 |
| 2026-10-19 | s56b | `a9dd857` | Isolated the collapse at 6,223 tok + 9 tools and added serve default flags (1.05 in production) | 60 |
| 2026-10-20 | Q4_0 native path | `c51d6a3` | 27B runs at 18.4 GB / 178–186 ms/token | 20 |
| 2026-10-20 | s118 | no commit | Key truncation cannot be proven numerically invariant by static analysis | 30 |
| 2026-10-20 | s119 | no commit | Designed 3 plans for speculative decoding and computed the break-even arithmetically | 30 |
| 2026-10-20 | s120 | `506b515a` | Measured the acceptance rate of n-gram speculation and closed Plan B | 30 |
| 2026-10-20 | s121 | `568afa64` | Closed the 90 ms upper bound of swiglu epilogue fusion arithmetically | 30 |
| 2026-10-20 | s122 | `4dff66e7` | Disproved key truncation executably with a regression test | 30 |
| 2026-10-20 | s87 / s88 | `e547749d` | Made phase accounting additive and split D2H by issue point | 40 |
| 2026-10-20 | s96 | `55586f03` | Deleted the already-rejected GPU gdn_proj path | 50 |
| 2026-10-20 | s98/s100 | `a0afc03` / `7543158` | Needle-recall regression checker and filler correction (about 4,656 tok is an estimate) | 60 |
| 2026-10-20 | s103/s107 | `65e4546` / `cef9384` / `1a063af` | 10-turn conversation checker, retirement of the s97 version, one-command CPU gate | 60 |
| 2026-10-21 | s156 | `ca5e44a` | `denno info` reads dimensions from `text_config` | 20 |
| 2026-10-21 | K1 | `2344f67` | Added TQ1_0 / IQ1_M decoders and corrected the type ids | 20 |
| 2026-10-21 | s157 / B20 stage 1 | `2701cf3` | Put the 27B q4 linears on the resident chain and removed 392 round-trips/step | 30 |

### Period 5: Cutting round-trips — the main line of prefill and production deployment

| Date | phase | commit | What was done | Vol |
|---|---|---|---|---|
| 2026-10-22 | s158 | `c95b44c` | Unknowns as `?`, and display of rope / attention / dtype / quantization | 20 |
| 2026-10-22 | K2 | `f6f03fd` | Fixed TQ1_0 / IQ1_M bit-identical to the reference implementation, into the default gate | 20 |
| 2026-10-22 | K3 | `45a9696` | Fixed the Q4_0 edge block and the accumulation order of the CPU gemv | 20 |
| 2026-10-22 | s164 | `8a0525b` | Fill in dimensions from a directory containing only GGUF | 20 |
| 2026-10-22 | s168 | `73d2e10` | Emit `quant:` with the dominant type other than F32 | 20 |
| 2026-10-22 | s160 | no commit | Refuted the premise of a fixed gate_up cost by static audit | 30 |
| 2026-10-22 | s162 | `5931c7a` | Long-context (3,000 / 4,096 tok) regression into the default gate (4 more tests, +0.01 s) | 60 |
| 2026-10-23 | s175 | `d787f66` | Match the layer count to the inference trunk (block_count − nextn) | 20 |
| 2026-10-23 | P35 | `98dd8e9` | Judged raw-block passthrough on real data (0 allowed types) | 20 |
| 2026-10-23 | P37 | `1bc1702` | Capped the fp32 temporary area at load time (10/10 bytes identical) | 20 |
| 2026-10-23 | P38 | `cf917e5` | Proof that row-wise dequantization can replace the table | 20 |
| 2026-10-23 | P42 | `955690e` | The 5,086 MB resident `token_embd` to row-wise dequantization | 20 |
| 2026-10-23 | s174 / P16 | `c3e53517` | Fold the two fp32 matvecs of beta and alpha into one | 30 |
| 2026-10-23 | P14 | `c4c6f569` | Fold FFN gate/up into a single GEMM | 40 |
| 2026-10-23 | P20 | `5f222949` | Write q of the full layer directly into pinned staging | 40 |
| 2026-10-23 | P21 | no commit | Choose the row tile of the q4 GEMM from the n of the projection (implementation discarded) | 40 |
| 2026-10-23 | P25 | no commit | Settled the breakdown of ffn_down and decided to add nothing | 40 |
| 2026-10-23 | P27 | no commit | Audited from the mechanism what phase accounting does not measure | 40 |
| 2026-10-23 | P28 | `3c383a7f` | Measured the host cost of full_core and moved a wasted allocation | 40 |
| 2026-10-23 | P29 | `86c44535` | Fixed 4 points of how to read phase accounting in docs | 40 |
| 2026-10-23 | P30 | `70675d32` | Fixed the PF env on the server-launch side and eliminated a no-op | 40 |
| 2026-10-23 | P31 | `b5f97db8` | Fixed the static invariants of the prefill chain without a GPU | 40 |
| 2026-10-23 | P34 | `a9b91394` | Turned the known limits of the smem cast scanner into checks | 40 |
| 2026-10-23 | P36 | `9e7de184` | The scanner takes in host launch block shapes | 40 |
| 2026-10-23 | P39 | `255e147f` | The scanner takes in the grid and judges blockIdx rows | 40 |
| 2026-10-23 | P41 | `1068bd0d` | Added a device that splits the full_core phase into 4 intervals | 40 |
| 2026-10-23 | P44 | `c7b0991e` | D2H V directly into the KV cache row instead of back to scratch | 40 |
| 2026-10-23 | P15 | `c0b19d07` | Removed the host wait between passes, 3 syncs to 1 | 50 |
| 2026-10-23 | s241 | `0d9faef7` | Fused the decay and the inner product of pass 1 into a single pass | 50 |
| 2026-10-23 | s243 | `8503925c` | Unrolled the fused loop by 8 and cut registers | 50 |
| 2026-10-23 | P36 | `3e288aa2` | The 4-lane version of pass 1 showed no effect in measurement | 50 |
| 2026-10-23 | P40 | `bf88ab26` | Split the wall time of the 3 kernels with CUDA events | 50 |
| 2026-10-23 | P43 | `88e9e86d` | The reason KPROF did not appear on the real machine was the output point | 50 |
| 2026-10-24 | P46 | `2db6448d` | Chunked the att round-trips and overlapped D2H/H2D with the gate | 40 |
| 2026-10-24 | P47 | `b44ba551` | The D2H destination of full_qk to pinned staging | 40 |
| 2026-10-24 | P48 | `6f5f4a3c` | Removed the per-row intermediate Vec from prepare | 40 |
| 2026-10-24 | P50 | `e841db48` | Broke down the tail and showed that the H2D of att is already pinned | 40 |
| 2026-10-24 | P51 | `ec7d4f31` | AVX2 16-lane host projection of GDN beta/alpha | 40 |
| 2026-10-24 | P52 | `b6e3fa27` | Widened the per-thread tile of the q4 GEMM to 8x8 | 40 |
| 2026-10-24 | P45 | `d19aa70b` | Removed the sync after the 3 passes from the device output path | 50 |
