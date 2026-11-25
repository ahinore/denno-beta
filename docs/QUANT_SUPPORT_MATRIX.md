# 量子型サポートマトリクス (task 858)

GGML の量子型ごとに、どの段 (stage) が GPU カーネルを持ち、どこが CPU に落ちるかを
1 枚にまとめたもの。型の一覧の**唯一の定義**は
`crates/denno-engine/src/quant_matrix.rs` の定数で、この文書と
`quant_matrix::tests` がその内容を固定する。

- 対象バージョン: `br858` (task 858)
- CPU 参照: `kquant::dequant_row_into` / `kquant::dequant_block_into` (全 25 型)
- 数値方針: `docs/NUMERIC_POLICY.ja.md`。GPU カーネルのミラーは f64 真値に対して
  絶対 1e-4 + 相対 1e-5 (prefill だけは FMA 融合 + 加算順が違うため契約が緩い)。

## 1. 段と入口

| 段 | 入口 (コード) | 型の判定 |
| --- | --- | --- |
| CPU 参照 (decode / prefill 共通) | `kquant::dequant_row_into` | `kquant::dequant_supported` = 全 25 型 |
| dense decode GEMV (m=1) | `cuda::maybe_gpu_gguf_gemv_into` → `CudaCtx::moe_gemv_dev_with_pref` | `cuda::gguf_row_bytes` (→ `quant_matrix::DENSE_GPU`) |
| dense prefill GEMM (m=nt) | `cuda::maybe_gpu_gguf_gemm_into` → `CudaCtx::moe_gemm_dev` | `cuda::gguf_row_bytes` + `quant_matrix::dense_gpu_supported` |
| expert decode (VRAM 常駐, K 系 4 型) | `expert_vram::acquire` → `CudaCtx::moe_gemv_dev_buf` | `CudaCtx::moe_gemv_dev_ready` = `gguf_row_bytes` |
| expert decode (IQ 系) | `model::exp_iq` → `kernels_exp_iq*.cu` | `exp_iq::slot` (→ `quant_matrix::EXPERT_IQ`) |
| expert prefill (K 系) | `cuda::moe_pf` / `model::exp_pf` | `exp_pf::slot` の K スロット (→ `PREFILL_EXPERT_K`) |
| expert prefill (IQ 系) | `cuda::moe_pf` の TC grouped GEMM | `exp_pf::slot` の IQ スロット (→ `EXPERT_IQ`) |

`gguf_row_bytes` は decode GEMV・prefill GEMM・expert の VRAM 常駐が**共通で**
通るので、ここが表とずれると「GPU を試みて必ず落ちる = 静かに CPU へ戻る」経路に
なる。task 858 で `quant_matrix::DENSE_GPU` を単一の定義にし、テストで固定した。

## 2. マトリクス

凡例: `G` = GPU カーネルあり / `C` = CPU (スカラー参照) / `-` = この段では扱わない。

| 型 | id | dense decode | dense prefill | expert decode | expert prefill |
| --- | ---: | :---: | :---: | :---: | :---: |
| F32 | 0 | C | C | - | - |
| F16 | 1 | C | C | - | - |
| Q4_0 | 2 | C | C | C | C |
| Q4_1 | 3 | C | C | C | C |
| Q5_0 | 6 | C | C | C | C |
| Q5_1 | 7 | **G** | **G** | **G** | **G** |
| Q8_0 | 8 | **G** | **G** | **G** | **G** |
| Q2_K | 10 | C | C | C | C |
| Q3_K | 11 | C | C | C | C |
| Q4_K | 12 | **G** | **G** | **G** | **G** |
| Q5_K | 13 | **G** | **G** | **G** | **G** |
| Q6_K | 14 | C | C | C | **G** |
| IQ2_XXS | 16 | C | C | C | C |
| IQ2_XS | 17 | C | C | C | C |
| IQ3_XXS | 18 | C | C | C | **G** |
| IQ1_S | 19 | C | C | C | C |
| IQ4_NL | 20 | C | C | C | **G** |
| IQ3_S | 21 | C | C | C | **G** |
| IQ2_S | 22 | C | C | **G** | **G** |
| IQ4_XS | 23 | C | C | **G** | **G** |
| IQ1_M | 29 | C | C | C | C |
| BF16 | 30 | C | C | - | - |
| TQ1_0 | 34 | C | C | C | C |
| MXFP4 | 39 | C | C | C | C |
| Q2_0 | 42 | C | C | **G** | **G** |

補足:

- **Q6_K は非対称**。expert の prefill (`kernels_exp_k_pf.cu` /
  `kernels_exp_k_tc.cu`) にはカーネルがあるが、dense の decode / prefill と
  expert の decode には無い。既定モデルの `token_embd` は Q6_K で dense ではない
  ため実害は無いが、Q6_K の dense 投影を持つモデルでは decode も prefill も CPU。
- **Q6_K の expert decode** も CPU (IQ 経路にも K 経路にも 6 型スロットが無い)。
- **IQ 系の dense** は CPU。prefill では `gemv_batch` が行列を f32 に開いて
  トークンブロックで使い回す (`dequant_rows_f32`、作業域 O(n·k) の一時領域)。
- F32 / F16 / BF16 は `GgufLinear::gemv` の CPU 参照で、expert としては現れない。

## 3. ゲート (env)

| 経路 | ゲート | 既定 |
| --- | --- | --- |
| dense prefill GEMM | `DENNO_GPU_GEMM` | on (`0` で CPU) |
| dense prefill TC GEMM (Q8_0 / Q4_K / Q5_K) | `DENNO_PF_TC2_GEMM` / `DENNO_PF_TC_GEMM` / `DENNO_PF_TC_SMALLN` | on |
| dense decode GEMV カーネル | `DENNO_GEMV_V3` / `DENNO_GEMV_V5` / `DENNO_GEMV_FP16` | v5 既定 on、fp16 off |
| expert decode K 系 (VRAM 常駐) | `DENNO_EXPERT_GPU=1` + `DENNO_EXPERT_VRAM_MB>0` | off |
| expert decode IQ 系 | `DENNO_EXPERT_IQ_GPU=1` | off |
| expert prefill IQ TC | `DENNO_PF_TC_EXP` | on |
| expert prefill K TC | `DENNO_PF_TC_EXP_K` | on |

## 4. 監査の結論 (task 858)

1. **想定外のフォールバックは見つからなかった。** どの段も型の判定は明示的な
   whitelist で、外れた型は `kquant::dequant_supported` の CPU 参照に落ちる
   (fail closed)。全 25 型が CPU でデコードできることは
   `quant_matrix::tests::every_type_has_cpu_reference_and_gpu_types_are_subset`
   が固定する。
2. **表が 4 か所に散っていた** (dense の 4 型が 3 つの独立した `match` に
   書かれていた)。dense の 4 型を `quant_matrix::DENSE_GPU` に集約し、
   `gguf_row_bytes` と `gemv_batch_inner` がそこを通るようにした。値・分岐は不変。
3. **型ごとの数値パリティのテストが Q5_K だけ抜けていた**
   (`fp16_gemv_mirror_error_stats` の 4 ケースは Q4_K / Q5_1 / Q8_0 / Q8_0)。
   `dense_quant_gemv_mirror_within_numeric_policy` を足して 4 型すべてを
   f64 真値 + 余弦で固定した。
4. prefill の K 系 / IQ 系の索引パリティは既存の
   `pf_exp_k_tests` / `pf_exp_tc_tests` / `pf_exp_ktc_tests` が型ごとに持つ。
   expert decode の IQ 系は `exp_iq::tests::mirror_matches_dequant_row` が持つ。

## 5. 確認方法 (GPU 無しのコンテナで回る)

```
CARGO_TARGET_DIR=/tmp/target858 cargo test --release -p denno-engine \
  --features cuda --lib -- quant_matrix --test-threads=1
```

GPU のある環境では、実カーネルとのビット一致 / 誤差を次で見る (どれも `#[ignore]`):

- `cuda::tests::cuda_moe_gemv_v2_bit_exact_vs_cpu` (dense decode の 4 型)
- `cuda::tests::cuda_gemv_fp16_matches_cpu` (fp16 経路)
- `cuda::tests::cuda_gemm_q8_0_bit_exact_vs_cpu` ほか prefill の TC/GEMM 系
