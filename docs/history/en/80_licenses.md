> Japanese original: [../80_licenses.md](../80_licenses.md)

> [Index](README.md) / Previous: [Volume 8 Measurement and verification](70_infra.md) / Next: [Appendix A all phases](90_phase_index.md)

# Appendix B — Reference implementations and licence attribution

This repository is written with reference to external implementations. This page is the single ledger of
**which one was consulted, how far it was followed, and under which licence**. The same content is stated in
the header of each source file: this is the ledger, the headers are the fulfilment.

| Source | Licence | How it is used | Verdict |
|---|---|---|---|
| llama.cpp `ggml-quants.c` (`dequantize_row_*`) | MIT (Copyright (c) 2023-2026 The ggml authors) | **Faithful port** (bit-identical contract) | OK — the MIT notice obligation is fulfilled in the `kquant.rs` header |
| kquant tables from `ggml-common.h` (`kquant_tables.rs` is generated) | same | **Verbatim extraction of table values** (`gen_kquant_tables.py`) | OK — the copyright notice is embedded in the generated file's header (regenerated) |
| llama.cpp `delta-net-base.cpp` / `qwen35.cpp` (GDN maths) | same | Port of the formulae and operation order (Rust reimplementation, no verbatim copy) | OK — attribution in the `gdn.rs` header |
| llama.cpp `delta-net-base.cpp` (GDN maths → CUDA kernel) | same | The same formulae ported to CUDA (`kernels/gdn.cu`). GDN is under the no-bit-identity contract, so `expf` etc. are used and only determinism is guaranteed | OK — MIT notice in the `gdn.cu` header |
| GGUF format specification (`gguf.cpp`) | same | Reference to the format specification only (the parser is our own) | OK — attribution in the `gguf.rs` header |
| Quantized block definitions in `ggml-common.h` (byte layout of Q8_0 / Q5_1 / Q4_K / Q5_K) | MIT (Copyright (c) 2023-2026 The ggml authors) | Reference for the block layout (the batched GEMM for prefill is our own CUDA implementation) | OK — attribution in the `kernels/kernels_gemm.cu` header |
| llama.cpp `src/models/qwen4exp.cpp` (commit `d854d98`, fetched 2026-10-26) | MIT (Copyright (c) 2023-2026 The ggml authors) | Reference for the Qwen4-family formulae (the PLE n-gram index, hyper-connections, the QSA indexer, the MoE routing order). Reimplemented in Rust / CUDA; no verbatim copy | OK — attribution in the headers of the port targets (`moe.rs` / `ple.rs` / `hyper.rs` / `indexer.rs`) |
| Qwen3.5 model (configuration and weights) | Apache-2.0 | Configuration values consulted; GGUF weights loaded | OK — attribution in the `qwen35.rs` header |
| Qwen3.8-Flash-Next (configuration and weights, `qwen4exp` architecture) | Qwen Community License 1.0 | Configuration values consulted; GGUF weights loaded for local inference only. The weights are not redistributed | OK — the copyright notice is kept in this appendix. Not used to run a Model as a Service |

**Conclusion** — All of it is compatible with this repository's licence (MIT OR Apache-2.0). The MIT-derived
parts (the kquant port and the table extraction) keep the upstream copyright notice in the relevant files, as
the MIT notice obligation requires. There is no GPL contamination. The licences of the model weights
(Apache-2.0 / Qwen Community License 1.0) apply to the weights themselves and do not extend to the code in
this repository. The weights are not redistributed.

**Practice** — When a new reference implementation is brought in, add it to **both** this table and the header
of the relevant file. One of the two is not enough as a ledger: fixing only the table and forgetting the header
makes the attribution disappear when a file is taken out on its own.

> This table was moved here from "reference implementations and licence check (2026-10-14)" in the old
> `docs/HISTORY.ja.md`, when the development history was reorganised into volumes. The content is unchanged
> (the verdicts, the licences, the distinction in how each is used); only the rows were tidied for reading.
