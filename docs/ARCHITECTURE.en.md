# denno-beta Architecture Guide

> An Anthropic Messages API–compatible LLM server written in Rust.
> The goal is to understand how LLMs work (tokenization → forward pass → sampling → decode loop) through code, so the **inference kernel is implemented entirely from scratch** (no existing LLM libraries such as candle).
> Repository: https://github.com/ahinore/denno-beta
> Structural reference: ashhart/TensorFold (a similar server written in Zig)
> Development history and measurement data: see the [Development History](HISTORY.en.md)

---

## 1. Purpose and Design Principles

- An LLM server written in Rust with a **fully hand-written inference kernel** (Anthropic Messages API compatible)
- The goal is understanding LLM internals through code → no existing inference libraries
- Single binary, exact decoding with a fixed seed (same seed + temperature 0 → **bit-identical output**). Runs on CPU by default; CUDA GPU offload is optional via a `--features cuda` build (§8; linear layers etc. stay bit-identical to CPU)
- The topmost discipline: optimizations and parallelization must never break "same input → same output"
- Operations policy: **keep the host machine clean; everything runs inside Docker**

## 2. Overall Architecture

```mermaid
flowchart LR
    subgraph CLI["denno-cli (single binary)"]
        A[pull / models / info / generate / serve]
    end
    subgraph Server["denno-server (axum)"]
        B["POST /v1/messages<br/>(non-streaming + SSE)"]
        C["GET /v1/models, /health"]
        S[scheduler<br/>bounded queue + continuous batching]
    end
    subgraph Engine["denno-engine (inference core)"]
        D[tokenizer<br/>HF tokenizers + chat template]
        E["model<br/>GPT-2 / Qwen2 (Llama-family) / Qwen3.5 (GDN)"]
        F[KvCache<br/>slot pool + prefix cache]
        G["Sampler<br/>greedy / temperature / top-p / top-k<br/>repetition penalty (A2.5)"]
        T[tensor<br/>sgemm / norm / rope]
        Q[quant<br/>int8 + SIMD i8×i8]
        U["cuda<br/>GPU offload (--features cuda)"]
        P[prof<br/>self-profiler]
        TL[telemetry<br/>capture hooks]
    end
    subgraph View["denno-view (visualization relay, separate process)"]
        V[dashboard<br/>WS multiplex + /chat proxy]
    end
    A --> B --> S --> D --> E --> T
    E --> F
    E --> G
    E --> Q
    E --> U
    TL -.->|observation only| W2[GET /v1/telemetry/ws]
    W2 --> V
```

- A cargo workspace built around the **single binary** `denno` (cli/server/engine + telemetry), plus the visualization relay `denno-view` running as a separate process
- Runs Qwen2.5-0.5B–7B-Instruct, GPT-2 124M, and **Qwen3.5 (GDN, via the GGUF loader path, §12)**. CPU-only by default; a `--features cuda` build keeps weights/KV/hidden states resident in VRAM with kernel offload (§8)
- Inference is guarded by `std::sync::Mutex<Engine>`, and the **scheduler steps the engine one forward at a time**, letting new prefills interrupt decoding (P4; converted to continuous batching in P6)

```
denno-beta/
├── Cargo.toml                 # workspace (profile.dev opt-level=2)
├── .cargo/config.toml         # rustflags = ["-C", "target-cpu=native"]
├── crates/
│   ├── denno-engine/src/
│   │   ├── tensor.rs          # matmul, normalization, RoPE, activations
│   │   ├── quant.rs           # int8 quantization + SIMD i8×i8 dot products (NEON/AVX2)
│   │   ├── kv_cache.rs        # KV cache (truncate for prefix reuse)
│   │   ├── prefix_cache.rs    # multi-entry LRU prefix cache
│   │   ├── sampler.rs         # temperature/top-k/top-p sampling + repetition penalty (A2.5)
│   │   ├── load.rs            # safetensors loading (single/sharded, F16/BF16→f32)
│   │   ├── tokenizer.rs       # tokenizer.json + minijinja chat template
│   │   ├── engine.rs          # begin/step_chat(): prefill→decode stepper
│   │   ├── cuda.rs            # CUDA kernel dispatch + GPU chains (feature "cuda", §8)
│   │   ├── gguf.rs            # GGUF container/KV/tensor reading (§12)
│   │   ├── kquant.rs          # GGML k-quant/IQ dequant (bit-exact with llama.cpp)
│   │   ├── kquant_tables.rs   # auto-generated tables for the above (do not hand-edit)
│   │   ├── prof.rs            # self-profiler (DENNO_PROFILE=1)
│   │   └── model/
│   │       ├── mod.rs         # Model trait + load_model dispatch
│   │       ├── gpt2.rs        # GPT-2 (LayerNorm + GELU)
│   │       ├── llama.rs       # Llama/Qwen2 (RMSNorm + RoPE + SwiGLU + GQA)
│   │       ├── qwen35.rs      # Qwen3.5 hybrid (GDN + full attention, §12)
│   │       └── gdn.rs         # Gated DeltaNet math core (sequential AR form)
│   ├── denno-server/src/
│   │   ├── lib.rs             # axum HTTP server + SSE + telemetry WS
│   │   └── scheduler.rs       # bounded queue + continuous batching worker
│   ├── denno-telemetry/src/   # telemetry frame format (zero deps, DNTL + f16/varint)
│   ├── denno-view/src/        # visualization relay server (dashboard + WS multiplex + /chat)
│   └── denno-cli/src/         # clap subcommands + HF download
└── docs/                      # API spec, architecture guide / dev history (JA & EN)
```

## 3. Inference Flow: Request to Response

```mermaid
flowchart TB
    REQ["POST /v1/messages"] --> AUTH{api key check<br/>model name check}
    AUTH -->|"queue full"| E503["503 overloaded_error"]
    AUTH --> TOK["tokenizer<br/>apply_chat_template → encode"]
    TOK --> SUB["scheduler.submit_chat<br/>bounded queue (mpsc sync_channel)"]
    SUB -->|"queue full"| E503
    SUB --> W["worker (single thread)"]
    W --> ADMIT[admit: begin_chat<br/>chat template / max_tokens clamp]
    ADMIT --> PC{prefix cache<br/>largest LCP entry?<br/>(recurrent: full prompt match only)}
    PC -->|"hit"| TR["reuse KV (truncate) · recompute only the difference<br/>recurrent restores KV+GDN state as a pair (no truncate)"]
    PC -->|"miss"| PF["prefill the whole prompt<br/>(weights × input GEMM)"]
    TR --> LIVE[live session]
    PF --> LIVE
    LIVE --> LOOP["step_chat × N<br/>(engine mutex one forward at a time)"]
    LOOP --> PULL["pull newly arrived requests between steps<br/>(admit when a slot is free)"]
    PULL --> BATCH["decode: combine all resident sessions<br/>into a single forward (P6)"]
    BATCH --> STOP{stop condition?<br/>EOS / max_tokens / stop_seq}
    STOP -->|"continue"| SAMP["sampler<br/>greedy / temperature / top_p / top_k<br/>repetition penalty (A2.5)<br/>fixed seed for determinism"]
    SAMP --> LOOP
    STOP -->|"stop"| OUT[ChatOutput<br/>text / stop_reason / usage]
    OUT --> RESP["JSON or SSE<br/>(event: lines + 2-second keep-alive)"]
```

### Key points

- **Prefill** (forward the whole prompt at once) and **decode** (one token at a time) form the same two-stage structure as real products. Prefill is batched GEMM; decode is matvec
- **Queue and 503**: admission uses a bounded channel. Requests arriving while the running set + waiting queue are full immediately get `503 overloaded_error`. Concurrent sessions are capped at `MAX_ACTIVE_SESSIONS = 4`
- **Prefix cache**: from the prompt token lists and KV snapshots, pick the entry with the **largest LCP (longest common prefix)** and reuse the matching KV portion. Entries evicted by LRU when the token budget (default 32k) is exceeded. Choosing "largest LCP" rather than "most recent" is the key to hits in alternating-session workloads
- **step_chat granularity**: the engine mutex is held for one forward only. New admits/prefills can interrupt during decode, and decode of all resident sessions is combined into a single forward (§6.1)
- **Determinism guarantees**: the sampler is seed-fixed and the accumulation order of every dot product is fixed. Same input → bit-identical output. int8 mode differs from fp32 due to activation quantization, but is deterministic within int8
- **max_tokens clamping**: max_tokens exceeding the context would be a 400 in the Anthropic API, but for learning purposes it is clamped to fit the context and generation continues

## 4. KV Cache (kv_cache.rs)

Transformer attention needs "the K (keys) and V (values) of all past tokens". Instead of recomputing everything for each generated token, each layer's K/V are cached and **only the new token is appended**.

```rust
pub struct KvCache {
    n_layers: usize,   // 24 (Qwen2.5-0.5B)
    kv_dim: usize,     // kv_heads(2) × head_dim(64) = 128
    max_len: usize,    // 32768
    len: usize,        // number of tokens written
    k: Vec<Vec<f32>>,  // per-layer [max_len, kv_dim] contiguous buffers
    v: Vec<Vec<f32>>,
}
```

- `advance(n)`: after writing n tokens, advance `len`
- `truncate(len)`: roll back to `len` (for the prefix cache. The rolled-back region is stale, but attention only reads `0..len+n`, so it is never read and gets overwritten by the next forward)
- Allocated in bulk at startup and reused across requests (no re-allocation)
- **Slot pool**: the engine holds `caches: Vec<KvCache>` (env `DENNO_BATCH_SLOTS`, default 4); a session occupies its slot until completion (P6 continuous batching. The KV rows and cached_prompts in a slot are kept after completion so the next session with the same prefix can reuse them by truncation only)
- **Size**: per token = 24 layers × 128 dims × 2 (K+V) × 4 bytes ≈ **24.6 KB/token**. About 800 MB at full context. Thanks to Qwen2's **GQA**, K/V are stored for only 2 heads, shared by the 14 q-heads

## 5. Compute Kernels (tensor.rs / quant.rs)

Heavy GEMMs are delegated to `matrixmultiply::sgemm` (AVX2/AVX512 capable), parallelized with rayon.

| Function | Role |
|---|---|
| `matmul(a, b, m, k, n)` | c = a@b (b: [k,n] row-major) |
| `matmul_bt(a, b, m, k, n)` | c = a@b^T (b: [n,k] nn.Linear layout) |
| `matvec_bt(a, b, k, n)` | m=1 only. **Parallelized by output columns** (below) |
| `sgemm_acc(...)` | c += α·a@b^T (for attention scores) |
| `rmsnorm / layernorm` | normalization |
| `rope_rows` | RoPE (rotate_half style) |
| `silu / gelu / softmax / add_bias` | elementwise ops |

### Decode parallelization

Prefill matmuls have large m and can split along rows of A, but **decode is an m=1 matvec** with no room to split along A. Instead, split by output columns (= rows of B):

```rust
c.par_iter_mut().enumerate().with_min_len(64).for_each(|(j, cj)| {
    *cj = dot(a, &b[j*k..][..k]);   // output j is independent, so this parallelizes
});
```

### Batch decode kernels and "order-preserving SIMD" (P6/P9)

The batch kernels `matvec_rows_bt` (fp32) / `matvec_rows_bt_i8` (int8), which combine decode of multiple sessions into one forward, carry a contract: **the k-direction accumulation of every output element must be bit-identical to a standalone matvec**. How the SIMD lanes are laid out is critically important:

| Direction | fp32 | int8 |
|---|---|---|
| **k** (dot product length) | ❌ not allowed — partial sums reorder → not bit-identical | ⭕ allowed — i32 accumulation is associative (SDOT) |
| **i** (batch sessions) | ⭕ allowed (P9) — each element's k accumulation stays sequential while 4 lanes run in parallel | (SDOT suffices) |
| **j** (output channels) | ⭕ allowed — independent chains used for ILP | ⭕ allowed |

Structure of the P9 fp32 kernel:

```text
Transpose activation a[m,k] into [k][mp] (zero-padded)
→ process 4 output channels × 4 batch lanes simultaneously:
   acc[t] += vld1q(at + d*mp) * dup(row_t[d])   // separate vmul + vadd instructions
Each lane's d accumulation is strictly sequential, no FMA (multiplication and addition rounding kept separate)
→ every row is bit-identical to matvec_bt
```

The kernel alone achieves **3.9–5.9×** vs 4 standalone matvecs (0.5B shapes, m=4). Batch-decode linear layers are back to weight-bandwidth-bound (same as solo). The remaining bottleneck in batch mode is attention itself (per-session sequential processing).

### int8 quantization + SIMD kernel (quant.rs)

- Enabled with `--quant int8` (opt-in). fp32 default is completely unchanged
- Linear-layer weights are quantized offline at load time with per-channel scales (**embeddings always stay fp32**)
- At decode time **activations are dynamically quantized** and a true i8×i8 integer dot product runs on SIMD
  - aarch64: NEON `vdotq_s32` (FEAT_DotProd) / x86: AVX2 `_mm256_madd_epi16` (switched via `target_arch` cfg)
  - i32 accumulation is associative, so SIMD and the scalar reference are **bit-identical** (determinism preserved)
- Effect: 5.4–12.9× vs fp32 in kernel benchmarks; e2e decode roughly 14.4 → ~51 tok/s

### Profiler (prof.rs)

Enabled with `DENNO_PROFILE=1`. On request completion, logs accumulated time per phase (embed/rmsnorm/rope/qkv/attn/oproj/gate_up/swiglu/down/lm_head/sampler/detok). Zero overhead when disabled, never touches the numeric path. Measurement data: P7.5 section of the [Development History](HISTORY.en.md).

## 6. HTTP Server (denno-server)

axum + tokio. Anthropic Messages API compatible.

### Endpoints
- `POST /v1/messages` — non-streaming / SSE streaming (`"stream": true`)
- `GET /v1/models`, `GET /health`
- Auth: `x-api-key` header (optional)
- Errors in Anthropic format: `{"type":"error","error":{"type":"...","message":"..."}}`

### SSE event sequence (Anthropic-compatible)

```
event: message_start        → metadata + usage (input_tokens)
event: content_block_start  → empty text block
event: ping
event: content_block_delta  ×N → {"delta":{"type":"text_delta","text":"..."}}
event: content_block_stop
event: message_delta        → stop_reason (end_turn/max_tokens/stop_sequence)
event: message_stop
```

Implementation notes:
- **Every event needs an `event: <type>` line**. Without it, Anthropic-wire clients fail to parse immediately and retry forever
- **2-second keep-alive**: during prefill the event stream goes quiet, and clients would time out without it
- Generation runs in `tokio::task::spawn_blocking`, streamed to the SSE response via an mpsc channel

### 6.1 Continuous batching (scheduler.rs + engine.rs, P6)

A single worker thread advances a **set of resident sessions** together. The KV cache is a **slot pool** (`DENNO_BATCH_SLOTS`, default 4); a session occupies its slot until completion (the old suspend/restore rotation is gone).

```mermaid
flowchart TB
    REQ["new request"] --> Q["bounded queue<br/>(503 when full)"]
    Q --> W{"free slot?<br/>active < MAX_ACTIVE"}
    W -->|yes| ADM["install_session<br/>pick a free slot<br/>(prefer the largest-LCP slot)"]
    W -->|no| WAIT["wait in queue"]
    ADM --> PRE{"prefilled?"}
    PRE -->|no| PF["step_chat: remaining prompt<br/>in a single forward"]
    PRE -->|yes| DEC
    PF --> DEC["step_chat_batch:<br/>sample + termination check per session,<br/>forward combined into one forward_batch"]
    DEC --> DONE{"all done?"}
    DONE -->|no| W
    DONE -->|"some sessions finished"| REL["release_session:<br/>free the slot (the KV snapshot into the prefix store<br/>is taken right after prefill since A4)<br/>KV rows and cached_prompts stay<br/>→ next session reuses them zero-copy"]
    REL --> W
```

Exactness rests on two legs:
1. **Kernel** — `matvec_rows_bt`/`matvec_rows_bt_i8` keep the k-direction accumulation order of every output element identical to a standalone matvec (`matvec_bt`; sgemm is not used). rayon splits by row blocks, so each element's addition sequence is unchanged
2. **Sampler** — each session owns an independent RNG, so output is bit-identical to a solo run no matter which sessions run concurrently (verified on real hardware for fp32/int8 with 4 concurrent sessions = solo)

Throughput at 0.5B: fp32 1.35× / int8 1.7× (b=4). At b=4, 0.5B transitions to compute-bound with the old fp32 kernel, but P9's order-preserving SIMD put batch linear layers back to weight-bandwidth-bound (solo-equivalent). Meanwhile 3B int8 (b=2) keeps step time almost unchanged (78ms→81ms) while aggregating 1.6× = still bandwidth-bound. **Larger models benefit from batching more directly**. The remaining bottleneck in batch mode is attention itself (per-session sequential processing).

## 7. Model Differences

### 7.1 Family detection

`load_model()` reads `model_type` from `config.json` to pick the family. The common interface is just the `Model` trait (`forward(tokens, cache) → logits`). The engine, scheduler, and server never know the family.

```mermaid
flowchart LR
    A[config.json] -->|"model_type = gpt2"| B[Gpt2<br/>model/gpt2.rs]
    A -->|"model_type = qwen2"| C[LlamaQwen2<br/>model/llama.rs]
    A -->|"GGUF: arch = qwen35"| Q35["Qwen35<br/>model/qwen35.rs (§12)"]
    A -->|other| D[error<br/>unsupported model_type]
    B --> E[Model trait<br/>forward / new_cache / eos_ids / context]
    C --> E
    Q35 --> E
```

### 7.2 Architecture comparison

| Item | GPT-2 (`gpt2.rs`) | Qwen2 / Llama-family (`llama.rs`) |
|---|---|---|
| Positional encoding | learned absolute positions `wpe` [ctx, e] | **RoPE** (rope_theta, default 1M) |
| Normalization | LayerNorm (weight+bias) | **RMSNorm** (no bias, eps) |
| Activation | GELU | **SwiGLU** (gate/up split, silu×down) |
| Attention | MHA (3e output by one Conv1D `c_attn`, with bias) | **GQA** (q/k/v as separate Linears, no bias) |
| Q/K/V heads | Q=KV (all heads) | KV head count from config (`0.5B`: Q14/KV2, `7B`: Q28/KV4) |
| lm_head | tied (uses transposed `wte`) | switched by config `tie_word_embeddings` (3B/7B untied → separate `lm_head.weight`) |
| EOS tokens | one | **possibly multiple** (two read from generation_config) |
| Weight layout | `[out, in]` (Conv1D style) → `bt` kernels | `[in, out]` → regular matvec kernels |
| Embedding | `wte` always fp32 (for gather) | likewise always fp32 (not quantized even in int8 mode) |

### 7.3 Layer-internal differences

**One GPT-2 layer:**

```mermaid
flowchart TB
    X[x input] --> N1["LayerNorm(weight + bias)"]
    N1 --> CAT["c_attn: one Conv1D outputs concatenated Q,K,V<br/>[n, 3e] in one shot (with bias)"]
    CAT --> KVW["write K, V to the KV cache<br/>(all heads)"]
    KVW --> ATT["causal attention<br/>per-position × per-head loops:<br/>q·k → softmax → Σ w·v"]
    ATT --> PROJ["c_proj (linear) + bias"]
    X -->|"residual +"| ADD1[(+)]
    PROJ --> ADD1
    ADD1 --> N2["LayerNorm(weight + bias)"]
    N2 --> FC["c_fc: expand to 4e"]
    FC --> ACT["GELU (activation)"]
    ACT --> PROJ2["c_proj: project back to e"]
    ADD1 -->|"residual +"| ADD2[(+)]
    PROJ2 --> ADD2
```

**One Qwen2 (Llama-family) layer:**

```mermaid
flowchart TB
    X2[x input] --> N1B["rmsnorm (no bias)"]
    N1B --> QKV["q/k/v computed as three separate Linears<br/>(0.5B has bias on q/k/v)"]
    QKV --> R["apply RoPE<br/>q for all heads / k for kv_heads"]
    R --> KVW2["write K, V to the KV cache<br/>(kv_heads only = GQA)"]
    KVW2 --> ATTB["GQA attention<br/>scores = q@k^T computed as one sgemm per kv-head × group<br/>→ causal mask → softmax → probs@V also a matmul"]
    X2 -->|"residual +"| ADD1B[(+)]
    ATTB --> OP["o_proj (linear)"]
    OP --> ADD1B
    ADD1B --> N2B["rmsnorm (no bias)"]
    N2B --> GU["gate Linear + up Linear"]
    GU --> SW["silu(gate) × up (SwiGLU)"]
    SW --> DOWN["down Linear"]
    ADD1B -->|"residual +"| ADD2B[(+)]
    DOWN --> ADD2B
```

**Code-level differences:**

| Processing | GPT-2 | Qwen2/Llama-family |
|---|---|---|
| Normalization | LayerNorm (weight+bias) | rmsnorm (no bias) |
| Q/K/V | **one matmul, concatenated output** | **three separate Linears** |
| Position injection | none (added at embedding) | **RoPE rotation right before attention** |
| Attention core | scalar loops (per position) | **sgemm matmul** (GQA) |
| MLP | 4e expand → GELU → project | gate/up → silu×up → down |

After the last layer: both families do a final normalization → lm_head (weights × hidden state) for logits.

### 7.4 GPT-2 forward flow (program order)

What `gpt2.rs`'s `forward(tokens, cache)` executes, in order. **The prompt (prefill) and each generated token run the same function**; only the token count n differs (prefill: whole prompt; decode: 1).

```mermaid
flowchart TB
    TOK["token list tokens[n]"] --> EMB["embedding: x[i] = wte[tok_i] + wpe[pos_i]<br/>(positions read from a learned table)"]
    EMB --> L["▼ 24-layer loop ─────────────────"]
    L --> LN1["layernorm(x) — weight+bias"]
    LN1 --> QKV["c_attn: one Conv1D outputs [n, 3e] in one shot<br/>(Q,K,V concatenated, with bias) → add_bias"]
    QKV --> KVW["write K, V to the KV cache<br/>(all heads = e dims)"]
    KVW --> ATT["causal attention:<br/>per-position × per-head loops:<br/>q·k → softmax → Σ w·v<br/>(double loop of scalar dots)"]
    ATT --> PROJ["c_proj + bias → add to x (residual)"]
    PROJ --> LN2["layernorm(x)"]
    LN2 --> MLP["mlp: expand to 4e → GELU → project → add to x (residual)"]
    MLP -->|"next layer / loop end"| L
    L --> FIN["final layernorm"]
    FIN --> LOG["logits = h @ wte^T<br/>(tied: uses the transposed embedding matrix)"]
```

**Characteristics**: QKV comes out of one big matmul (a single GEMM = GPU-friendly). Positions are added at the embedding stage, so attention has no positional processing. The attention core is a straightforward "per-position, per-head" loop structure.

### 7.5 Qwen2/Llama-family forward flow (program order)

`llama.rs`'s `forward(tokens, cache)`. Same skeleton of "embedding → layers → logits", but very different inside.

```mermaid
flowchart TB
    TOK["token list tokens[n]"] --> EMB["embedding: x[i] = embed[tok_i]<br/>(no position added yet)"]
    EMB --> L["▼ 24-layer loop ─────────────────"]
    L --> N1["rmsnorm(x) — no bias"]
    N1 --> QKV3["q/k/v computed as three separate Linears<br/>(0.5B has bias on q/k/v — unusual for Llama-family)"]
    QKV3 --> ROPE["RoPE: inject relative position into q and k as rotations<br/>(no absolute position table)"]
    ROPE --> KVW["write K, V to the KV cache<br/>(kv_heads only = GQA)"]
    KVW --> ATT["GQA attention (one sgemm):<br/>double loop over kv-head × group, but<br/>scores = q@k^T computed as a matmul<br/>→ causal mask → softmax → probs@V also a matmul"]
    ATT --> OPROJ["o_proj → add to x (residual)"]
    OPROJ --> N2["rmsnorm(x)"]
    N2 --> MLP["gate_proj + up_proj → silu(gate)×up<br/>(SwiGLU) → down_proj → add to x (residual)"]
    MLP -->|"next layer / loop end"| L
    L --> FIN["final rmsnorm"]
    FIN --> LOG["logits = lm_head(h)<br/>(untied: separate weights / tied: embed^T)"]
```

**Characteristics**:
- **Position is injected in a different place** — GPT-2 adds it at embedding; Qwen2 rotates q/k via RoPE right before attention
- **QKV is three separate Linears** (standard Llama-family shape). But 0.5B has bias on q/k/v (normally absent in Llama-family; ignoring it destroys the output)
- **Attention is GQA + matmul** — kv-heads are shared by a group of q-heads, and scores are delegated to sgemm (`sgemm_acc`). Structurally and performance-wise different from GPT-2's scalar loops
- **MLP is SwiGLU** — two Linears + elementwise product + one Linear (vs GPT-2's one Linear + GELU + one Linear)

### 7.6 Decode step shared by both models (engine.rs)

Model differences live only inside `forward` above. The stepping loop and sampling are shared:

```
one step of step_chat():
  apply_repetition_penalty       ← A2.5: only when requested (unique context tokens; bit-identical no-op when disabled)
  sampler.sample(logits)        ← greedy / temperature / top-k / top-p (fixed seed)
  → EOS / stop_sequences check
  → stream the diff of tok.decode(all generated)
  → model.forward(&[next], cache)   ← next token (prefill passes &prompt_ids[lcp..] in bulk)
```

The design only changes the n passed to `forward` between prefill and decode. Linear layers switch automatically: GEMM when n>1, output-split parallel matvec (`apply_bt`) when n=1.

From P6 on, `step_chat_batch` is used when multiple sessions run concurrently, and linear layers use the multi-column `apply_batch_bt` (→ `matvec_rows_bt`) to combine into one forward (§6.1). Each element's accumulation order matches the n=1 matvec, so solo-identical bit-exact output is preserved.

### 7.7 Qwen2.5 size differences

| Model | tied | shards | fp32 memory | Implementation notes |
|---|---|---|---|---|
| 0.5B | tied | single file | ~1GB (weights) | — |
| 1.5B | tied | single file | ~3GB | Just bigger dims. **Runs unchanged on the existing binary** |
| 3B | **untied** | **sharded** (index.json + 2 shards) | ~12GB | ✅ supported: sharded checkpoint + untied lm_head. fp32 ~2 tok/s, int8 ~4.7 tok/s |
| 7B | untied | sharded | ~28GB | ✅ verified with the same code: fp32 ~1.2 tok/s, int8 ~2.1 tok/s |

- Per-token KV cache size = `2 (K,V) × n_layer × n_kv_head × head_dim × 4 bytes`. At 32k tokens: 0.5B ≈ 805 MB (basis of the prefix cache token budget); 1.5B ≈ 1.9 GB; 7B ≈ the same as 0.5B thanks to GQA (4 KV heads)
- Large models are weight-bandwidth-bound at decode, so `--quant int8` + the SIMD kernel pays off most (3B: ~2.4× fp32; 7B: ~1.8×)

### 7.8 Model loading flow

```mermaid
flowchart LR
    PULL["denno pull REPO"] --> HUB["download into the HF hub cache<br/>(single model.safetensors)"]
    PULL -->|"3B/7B: index.json"| SHARD["download all shards"]
    SERVE["denno serve REPO"] --> CFG[read config.json]
    CFG --> MT{"model_type?"}
    MT -->|gpt2| G2[Gpt2::load]
    MT -->|qwen2| LQ[LlamaQwen2::load]
    G2 --> Q{quant mode?}
    LQ --> Q
    Q -->|fp32| FP[keep weights as fp32]
    Q -->|int8| I8["quantize linear layers offline with per-channel scales<br/>(embeddings stay fp32)"]
    FP --> READY[hand to the engine as the Model trait]
    I8 --> READY
```

`WeightStore` handles single and sharded checkpoints transparently, resolving shards via the `model.safetensors.index.json` mapping.

## 8. CUDA Optimization (implemented — P17 through B3)

Enabled by a `--features cuda` build. The P17 series implemented GPU offload of both the decode and prefill paths, and P19-A1 extended it to Qwen3.5 (GDN). **The 13,252-token cold prefill went from 395.8s → 12.2s** (at P17k), and B1 additionally removed the rmsnorm and host round trips structurally. All kernels are hand-written (no cuBLAS); committed `.ptx` files are managed by the procedure "first verify that recompiling the pre-change source reproduces the committed PTX sha256, then regenerate."

### 8.1 GPU path overview

The basic structure keeps **weights, the KV mirror, and hidden states resident in VRAM**, while the CPU handles scheduler/tokenizer/sampler. Kernel dispatch is centralized in `cuda.rs`:

| Path | Composition |
|---|---|
| decode chain (P17b/P17c/P17d/P17g/P17h) | embedding → rmsnorm → rope (`dno_rope_qkv`) → qkv GEMV (int8) → attention (`dno_attention` + GPU KV mirror) → o_proj → swiglu → down, chained with **hidden-state device residency and no syncs**. Linear layers group qkv / gate+up calls |
| prefill chain (P17i/P17j/P17k+B1) | Long prefill uses tiled int8 GEMM (`dno_gemm_i8`) + Flash-style online softmax (`dno_attention_prefill`) + resident FFN block (gate_up → swiglu → down). **B1's `forward_prefill_gpu` (Llama-family)** also moves rmsnorm (`dno_add_rmsnorm`, fused with residual add) to GPU and restructures into "one embedding H2D → everything resident on device within a layer → one D2H for the last row" (removing host round trips of 81MB × 4/layer ≈ 9GB/request; only KV-mirror-row D2H and the final row remain) |
| batched decode (B3) | rope for b>1 offloaded to `dno_rope_qk_batch` (host-generated tables + `(poss,dh,theta)` cache; on failure the host buffers stay untouched and it falls back to CPU) |
| Qwen3.5 (GDN, P19-A1/A1-r2) | Linear auto-offload + GDN scan kernels (decode n==1 single-shot; prefill n≥2048 3-pass chunked, §12.6). GDN state is VRAM-resident |

- **Demand-based KV allocation (P17e)**: the GPU KV mirror uses grow-on-demand allocation + advance reservation (`kv_bufs(layer,n)`) as a hard guarantee
- **all-or-nothing contract**: if a chain fails mid-way, the whole forward is redone on the CPU path (KV writes are idempotent; no GPU/CPU mixing)
- **Measured**: Qwen3.5-9B int8 cold prefill ~28s on CPU → 2.09s on GPU (285 in/64 out, A1) → 13,252 tokens in 12.2s (P17k)

### 8.2 Numerical contract (exactness)

| Operation | Contract |
|---|---|
| Linear layers (int8 GEMV/GEMM), rmsnorm, rope, swiglu, residual add | **Bit-identical to CPU** (int8 i32 accumulation is associative + fixed operation order; `-fmad=false` prevents fma contraction) |
| attention (decode `dno_attention` / prefill `dno_attention_prefill`) | **Approved relaxation window** (maxabs < 1e-3, cos > 0.999, user-approved) |
| GDN scan | bit-identity not required (contract is only "same input → same output + determinism"; `expf` etc. allowed) |

- **E2E regression criteria**: models dominated by linear layers are judged by "bit-identical to CPU"; GDN models by "**(a) determinism on GPU (same payload → same output) and (b) validity of the output content**". On near-tie prompts even temperature-0 greedy can flip the argmax through chaotic amplification across 32 layers of a tiny within-contract drift (~4e-7 observed) — see the "8095 GPU E2E regression investigation" section in the [Development History](HISTORY.en.md)
- Existing contracts — sequential == batched prefill bit-identity, solo == batch decode — hold on the GPU path too

### 8.3 What stays on CPU

- tokenizer / detok (HF tokenizers, CPU-friendly)
- scheduler, HTTP (scheduler.rs, lib.rs) — outside CUDA porting
- the engine's step_chat loop body (though its content becomes a sequence of kernel launches)
- sampler (RNG and top-p sorting stay on CPU, preserving the §5 determinism contract)

## 9. Usage

```bash
# download the model (into the HF hub cache ~/.cache/denno/hub)
cargo run --release -- pull Qwen/Qwen2.5-0.5B-Instruct

# start the server (fixed seed for exact decoding)
cargo run --release -- serve Qwen/Qwen2.5-0.5B-Instruct --name denno --port 8080 --seed 42

# generate directly via the CLI (with/without chat template)
cargo run --release -- generate Qwen/Qwen2.5-0.5B-Instruct --prompt "1+1は?" --temperature 0 --seed 42
cargo run --release -- generate openai-community/gpt2 --prompt "The capital of France is" --raw --seed 42
```

### Starting the visualization (denno-view)

```bash
# 1. start the server with telemetry on (same command plus DENNO_TELEMETRY=1)
DENNO_TELEMETRY=1 cargo run --release -- serve Qwen/Qwen2.5-0.5B-Instruct --name denno --port 8080 --seed 42

# 2. in another terminal, start denno-view
#    (defaults to the telemetry WS at ws://127.0.0.1:8080, listens on :8093)
VIEW_TOKENIZER=~/.cache/denno/hub/models--Qwen--Qwen2.5-0.5B-Instruct/tokenizer.json \
  cargo run --release -p denno-view

# 3. open http://localhost:8093/ in a browser and chat in the built-in chat box —
#    the model-driven program, layer panel, and KV/variable animations play along
```

`denno-view` environment variables: `VIEW_UPSTREAM` (default `ws://127.0.0.1:8080/v1/telemetry/ws`), `VIEW_PORT` (default 8093), `VIEW_RECORD` (record frames to a file for replay), `VIEW_TOKENIZER` (tokenizer.json path for token text display).

Being Anthropic Messages API compatible, it works as-is with the official SDKs and Anthropic-compatible clients (a Python SDK example is in the README).

## 10. Known Limitations

- No tool use (tool_use/tool_result content blocks) → no agent features, plain conversation only
- One model per process (multi-model serving is a future candidate). Concurrent decode of multiple sessions is already handled by continuous batching (P6)
- The viewer is the JS version (WASM port, replay scrubber, and L2 tensor display remain)
- Large-model download verification and performance are recorded in the [Development History](HISTORY.en.md)

## 11. Inference Visualization (denno-trace) — Implemented Architecture (P13/P14)

Since denno is a learning server, it can display the **program flow, hidden-layer states, and KV state** during inference in real time (P13 built the telemetry foundation, P14 the animation UI). Three design principles:

1. **Fully decoupled from inference** — visualization logic never enters the inference process. denno only streams observations
2. **Never make inference wait** — bounded channel + drop-allowed (when the consumer falls behind, frames are dropped and a gap marker is emitted). Waiting/blocking is forbidden
3. **Zero cost when OFF** — off by default (without `DENNO_TELEMETRY` the capture code never even runs). Observation is read-only copies only, **never touching the numeric path** (compatible with the bit-exact discipline)

### 11.1 Overall structure

```mermaid
graph LR
  subgraph denno["denno server (inference kernel)"]
    K["capture hooks (prof/forward/sampler/kv_cache)"] --> R["ring buffer + dedicated forwarding thread"]
    R --> W["WS endpoint /v1/telemetry/ws (binary frames)"]
  end
  W -->|"WebSocket (f16/varint flat format)"| V["denno-view server (separate process)"]
  V --> B["browser UI (HTML+JS version; WASM port pending)"]
  V --> F["replay recording (VIEW_RECORD, for later scrubbing)"]
  V --> C["built-in chat (POST /chat → upstream /v1/messages proxy)"]
```

- **denno (capture side)**: observation points in `prof.rs` op timing and forward/sampler/kv_cache/engine lifecycle (read-only, never touching the numeric path), streamed over WebSocket via a dedicated thread. One WS route on axum (`/v1/telemetry/ws`)
- **denno-view (relay server)**: WS client to the upstream denno (2-second retry on disconnect), multiplex relay to multiple browsers, frame recording to a file via `VIEW_RECORD` (replay material), built-in chat (sending a message drives the animation), `/tokenizer.json` (token text display), `/code/src` (implementation-code display, embedded at compile time via include_str!)
- **Viewer (UI)**: currently the HTML+JS version (the binary telemetry decoder is also implemented in JS and verified against Rust-generated frames). Porting to wasm-bindgen + canvas/WebGL2 remains

### 11.2 Capture levels (3 tiers, on demand)

| Level | Content | Traffic | Use |
|---|---|---|---|
| L0 flow (always on) | phase transitions (prefill/decode), op enter/exit (logical call stack rebuilt from event nesting), session lifecycle, output token stream | tens of B/step, **<0.5%** | program flow diagram, op timeline |
| L1 statistics | hidden-state norms, KV usage (layer×slot grid), logits top-k (8) + entropy, sampler choices | hundreds of B/step, **1–2%** | state overview, token distribution |
| L2 tensors (focus-limited, P15 candidate) | hidden-state vectors of a chosen session×layer (f16), attention probabilities (t_total vector), raw K/V | **one session + narrowing** ≤5% | heatmaps, tensor inspector |

- **P13 implemented L0/L1, and P14 the animation UI on top**. L2 focus is not implemented yet (the plan: the viewer sends "show me layer 14 attention of this session" and denno streams just that combination)
- L2 vectors are **rounded to f16 for transfer** (observation only, no effect on the numeric path, so outside the bit-exact discipline; halves bandwidth)

### 11.3 Wire format (denno-telemetry crate)

- New crate `denno-telemetry`: shared frame types + encode/decode (both the server and viewer use it)
- Frames: `magic(4B) + version(u8) + kind(u8) + session_id(u32) + seq(u32) + payload(len-prefixed)`
- Payload is a **flat structure + LEB128 varint + f16 tensors** (no JSON in the inference hot path — serialization leans zero-copy)
- A `Gap { lost: u32 }` frame explicitly marks loss (never silently skipped)
- Versioned (ready for future fields; the viewer skips unknown kinds)

### 11.4 denno-side implementation (P13)

- `denno-telemetry` crate (zero deps / std only): frame types + encode/decode. Server and viewer share the same format
- kinds: `ServerInfo / SessionAdmit / SessionDone / Prefill / Token / Step / OpSummary / KvUsage / Gap / TopK` (+ P14's `LayerRun` — one layer's op sequence bundled into one frame in execution order. `ServerInfo.layers` and `Prefill.tokens` were also added)
- Capture points: ① inside prof op timing (one emit line per existing hook) ② engine phase transitions and session admit/eos ③ after sampler (top-k) ④ kv_cache advance (usage; **not emitted for empty batches** — idle polling would flood frames) ⑤ after detok (output tokens)
- All capture is **read-only references after computation** (hidden states copied right after forward completes; never mid-flight)
- WS: axum `route("/v1/telemetry/ws", get(ws_handler))` — sends ServerInfo right after connect and relays via a hub→broadcast thread (503 when telemetry is disabled)

### 11.5 Viewer (denno-view) screen layout (implemented)

Three columns. The browser side also uses a playback queue + decimation so it **never makes inference wait** (the server side uses a bounded channel + drops).

| Area | What it shows |
|---|---|
| Center: implementation-code panel | Pseudo-code of the model-driven program. The running line advances like a debugger's PC, with the prefill branch and KV writes (right before attn_core) in true order. "real code ▸" switches to the actual scheduler.rs/engine.rs/llama.rs source (include_str! embedding + content-anchor mapping) |
| Center: model-driven program | denno serve→scheduler→engine.step_chat→model.forward{embed→layers→final_norm→lm_head}→sampler→detok. Active functions light up and "value" pulse dots travel between nodes. Input tokens drop from the top into embed, output tokens line up below. Fly-lines for KV/variable writes (green), reads (pink), and consumption (blue) |
| Left: KV slots + variables + op timing | KV slot cards (busy/token count/usage bar), hidden x/logits top-k/entropy/chosen token, measured ms per op |
| Right: model structure | embed→decoder×24 (ATT+MLP chips)→final_norm→lm_head. Chips light up individually per LayerRun, layer rows show measured op ms |
| Bottom: chat | Built-in chat (POST /chat proxy). Sending a message drives the animation |
| Header | speed slider (low = watchable slow motion / high ≈ real time, default high), model ID display |

- Not implemented: chrome-tracing-style op timeline, tensor inspector, attention view, replay scrubber (future candidates along with L2)
- 60fps strategy: decode into typed arrays → draw only the visible range → `requestAnimationFrame` decimation. When falling behind, decimate the display (data is buffered, never dropped; only rendering is reduced)

### 11.6 Performance budget and determinism

- L0 <0.5% / L1 1–2% / L2 ≤5% for the focused session (f16 + narrowing + bounded channel). Over budget → drop frames and report via Gap (inference is never slowed down)
- With telemetry OFF the code path effectively doesn't exist, and **exactness tests and the reference output are unchanged**. Even when ON it's read-only so numbers don't change (though A/B measurements are taken with telemetry OFF)

---
Development history, verification data, and future plans: [Development History](HISTORY.en.md)
## 12. Qwen3.5 hybrid architecture (Gated DeltaNet) and the GGUF loader (P19)

> The third model family, added in P19. On top of the existing safetensors + `model_type` dispatch route, two new elements arrived: a **GGUF loader route** and **layers with recurrent state (Gated DeltaNet)**. Measurement data: [Development History](HISTORY.en.md), P19 section.

### 12.1 Layer layout of Qwen3.5 (3:1 hybrid)

Qwen3.5 consists of two layer types throughout (0.8B: 24 layers = **18 GDN + 6 full attention**, 3:1 period. 27B: 64 layers = 48+16):

- **GDN layers (Gated DeltaNet)**: instead of attention, past context is memorized by a **recurrent linear state SSM** (Delta-Rule family). No KV cache; a fixed-size state matrix is carried instead
- **full attention layers**: GQA attention like previous Qwen2/3 (with QK-Norm, partial MRoPE, sigmoid output gate). Only these layers use the KV cache

### 12.2 GDN numerics (1 token, head h)

```
g[h]   = ssm_a[h] * softplus(alpha[h] + ssm_dt[h])   # ssm_a is negative (= -exp(A_log))
S      <- S * exp(g)                                  # decay (per-head time scales)
sk[c]  = sum_r k[r] * S[r][c]
d[c]   = beta[h] * (v[c] - sk[c])                     # beta = sigmoid(W_beta*x) ∈ (0,1)
S[r][c]+= k[r] * d[c]                                 # delta-rule state update
o[c]   = sum_r q[r] * S[r][c]
output  = rmsnorm(o) * w * silu(z)                    # gated RMS norm (z from attn_gate)
```

- q/k are **L2-normalized** beforehand (x/sqrt(Σx²+eps); q additionally 1/sqrt(head_k_dim))
- softplus follows the ggml convention (`x > 20 ? x : log(1+exp(x))`)
- The implementation is a 1:1 port of llama.cpp `delta-net-base.cpp`'s `build_delta_net_autoregressive` (sequential AR form). Bit-identity with llama.cpp is not required — only the denno contract (same input → same output + determinism) is guaranteed (the sequential AR form and the chunked form are mathematically equivalent)
- Code layout: `gdn.rs` has `GdnCore<'a>` (reference views of the tiny f32 tensors: conv1d/ssm_a/ssm_dt/ssm_norm) + `forward_token_core` (math core). Larger matrices (qkv/gate/beta/alpha/out) reuse the existing `Linear` → `model/qwen35.rs` assembles the layers

### 12.3 GGUF loader route (new in P19)

Previously HF safetensors was the only route; P19 added a **second weights route in GGUF format**:

```
*.gguf (mmap) → gguf.rs (container/KV/tensor info)
  → kquant.rs + kquant_tables.rs (auto-generated): dequant for 18 GGML k-quant/IQ types
     (bit-exact with llama.cpp: LCG fixtures, 16 blocks × 18 types all matched)
  → dequantized f32 weights re-quantized into the existing Fp32/Int8 pipeline
     (no new GEMM needed; Linear::new(w, k, n, mode) used as-is)
```

- Even mixed per-layer quantization like Unsloth Dynamic (UD-IQ2_XXS etc.) becomes identical to the existing route after dequantization
- **`ssm_beta`/`ssm_alpha` stay Fp32 even in Int8 mode**: control signals passed through sigmoid/softplus; ~1% per-row int8 noise breaks the output (beta directly scales the whole delta rule, alpha the decay rate)
- `pick_gguf`: selects from .gguf files in the directory, preferring UD-IQ2_XXS (`DENNO_GGUF_FILE` for explicit selection)
- **Load-time shape guard (`expect_ne`)**: before dequantizing any layer tensor, its GGUF header shape (ne) is checked against config-derived expectations (trailing 1s ignored). Unexpected structures (presence/absence of full-attn q|gate fusion, out-gate structure, mixed-in MTP layers) are rejected with a clear load-time error instead of being silently misread until the logits break (a product of the 27B out-gate structure investigation)
- **MTP (nextn) layer exclusion**: the portion of GGUF `block_count` exceeding config `num_hidden_layers` is skipped as nextn (MTP) layers only when it matches `nextn_predict_layers` and each extra block carries `nextn.eh_proj` without `ssm_a` (27B: block_count=65 = trunk 64 + 1). Inconsistencies are rejected at load — reading a nextn layer as an ordinary layer would corrupt the semantics
- **The 27B full-attention structure is common to 0.8B/9B** (settled 2026-10-16): attn_q=[e, 2×heads×dh] fused (per-head [q|gate]) · out-gate = sigmoid after attention, before o_proj · attn_output=[heads×dh, e] non-square. 9B is square only by coincidence (heads×dh == e); the engine's default path is correct for all three sizes (cross-checked against transformers modeling_qwen3_5 / llama.cpp qwen35.cpp / the real GGUF headers)

### 12.4 Recurrent-state management (KvCache extension and engine constraints)

GDN state is not "per token position" but a **function of the whole sequence**, imposing new constraints on KV-cache reuse design:

- KvCache gains a **GDN slot** (`gdn: Vec<Option<(conv_buf, ssm_matrix)>>`, per layer), allocated via `attach_gdn_slot`, zero-cleared by `reset()`
- `Model::is_recurrent()` (default false, true only for qwen35) added. For recurrent models:
  - **Prefix cache entry restoration is possible as a paired KV + GDN-state restore (A4)** — entries store a **GDN state snapshot taken at the prompt boundary (right after prefill, before decode)** paired with the KV rows (`KvCache::gdn_snapshot()`/`gdn_restore()`, a bit-exact copy). Since the state is a function of the whole sequence, it is valid at exactly one position: restoration only happens when the new prompt fully contains the entry prompt as a prefix (`best_full_match`). On a partial match even the KV-only restore is refused and the request falls back to a full prefill, so **there is structurally no path that restores the KV only and leaves the recurrent state inconsistent**
  - KV live reuse (truncate) is **safe only for pure extension (lcp == current length)**; truncating mid-sequence invalidates GDN state and falls back to full prefill
  - The snapshot is evacuated into the entry **right after prefill** (the prefill branch of engine `step_chat`); at completion (`release_session`) the state has already advanced with the generated tokens, so it cannot be evacuated there. The GDN state's byte size is converted to a token count via `bytes_per_token` and counted into the prefix-cache LRU budget (A4)
  - Ignoring this constraint causes **nondeterminism where the same seed changes the output** (observed and fixed in P19). The A4 restore path is test-guaranteed to be **bit-identical to a full prefill after restore** (qwen35 synthetic 27B shape + engine tests; a contrast experiment also confirms that a KV-only restore does change the output)

### 12.5 Determinism / exactness contract (P19 additions)

- Model level: forward determinism (2 runs bit-identical) + **cache continuity (forward(3)+forward(1) == forward(4) bit-identical)**
- Only dequant requires bit-exactness with llama.cpp (the GDN sequential math does not)
- GDN state, like KV, "grows monotonically within a session", so it inherits the existing KV guarantees (P6's solo==batch, P17e's demand-based allocation contract `kv_bufs(layer,n)` called ahead)

### 12.6 GDN GPU kernels and chunked scan (P19-A1 / A1-r2)

The GDN scan GPU kernel (`kernels/gdn.cu` + `kernels_gdn.ptx`) keeps **one block per head (grid = nv) with each thread holding one column of S**, so every delta-rule update completes within a thread (no cross-thread reduction). Determinism policy: fixed launch configuration, ascending-order sequential sums, no atomics / warp reductions. GDN falls under the relaxed contract (no bit-identity required), so `expf` etc. are allowed and CPU/GPU numerical agreement is verified within the relaxed bounds (maxabs < 1e-3, cos > 0.999).

Route selection (`cuda::gdn_scan_batch` is the single entry point, requires dk == dv / d_conv == 4):

- **decode (n==1)**: single-shot kernel `dno_gdn_scan`, 1 launch. Same kernel and launch configuration as batch prefill, so on the GPU the token-by-token and batch GDN outputs are bit-identical
- **prefill (1 < n < 2048)**: the same single-shot kernel, 1 launch (sequential token scan)
- **long prefill (n ≥ 2048)**: 3-pass **chunked scan** (`dno_gdn_chunk_pre` / `dno_gdn_chunk_state` / `dno_gdn_chunk_scan`):
  1. K1 (chunks in parallel): per-chunk local transitions W[ci] (end state of a local AR scan starting from S=0) and M[ci] (transition matrix propagated column-wise from the identity)
  2. K2 (chunks serial): serial propagation of the chunk-start states Sinit (Sinit[0]=S_in, Sinit[ci+1] = M[ci]·Sinit[ci] + W[ci])
  3. K3 (chunks in parallel): sequential AR scan within each chunk from Sinit[ci] → y and the final state

- Chunking is shared with the CPU reference implementation via `chunk_plan(n, 2048, 1024, 16)` (`gdn::scan_chunked_core`): chunk_tokens=1024, up to 16 chunks, remainder concentrated in the last chunk (all chunks non-empty)
- The chunked path is **GPU-only** (never called from the CPU forward paths), so the CPU contract "sequential == batch prefill bit-identical" is unchanged. A single chunk is bit-identical to sequential AR; multiple chunks are mathematically equivalent (only rounding order differs). The GPU chunked path is also deterministic (same input → bit-identical), covered by tests
- Mathematical basis: one delta-rule step is a **linear** map on S, S ← eg·(I − βkkᵀ)·S + βkvᵀ, so the cross-chunk dependency decomposes into "local transition computation" and "state propagation". k is L2-normalized (‖k‖≈1), β ∈ (0,1), eg = exp(g) < 1 — every factor has spectral norm ≤ 1, so M is a contraction and rounding error is not amplified
