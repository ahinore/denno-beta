# ref/ — llama.cpp 参照実装との突き合わせ用ツール

kquant の dequant 実装 (`crates/denno-engine/src/kquant.rs`) が llama.cpp の
`dequantize_row_*` と **1 bit も違わない**ことを検証するためのリファレンス生成器。

- `dequant_ref.c` — ggml-quants.c を直接リンクして擬似ランダムブロック列を
  デコードし、(生バイト, f32 出力) をバイナリにダンプする。
- 出力は `crates/denno-engine/tests/data/dequant_ref.bin` に commit 済み。
  テスト `kquant_dequant_bitexact_vs_llama_cpp`
  (`crates/denno-engine/tests/dequant_ref.rs`) が既定でこれを読む。

期待値は必ず**参照実装の出力**から作る。手計算・近似・別実装での代用は禁止
(このファイルを読んだ人が同じ手順を再現できることが契約)。

## 1. ビルド

llama.cpp のチェックアウトを `<llama.cpp>` とする。ggml-quants.c が
`GGML_COMMON_IMPL_C` で ggml-common.h を取り込むため、iq1s_grid などの
テーブルは ggml-quants.c 側に入る (追加のリンクは不要)。

```
gcc -O2 \
    -I<llama.cpp>/ggml/include -I<llama.cpp>/ggml/src \
    -o /tmp/dequant_ref \
    ref/dequant_ref.c <llama.cpp>/ggml/src/ggml-quants.c -lm
```

`ggml_abort` / `ggml_type_name` / `ggml_type_size` / `ggml_blck_size` /
`ggml_row_size` は dequant 経路では未使用なので、`dequant_ref.c` の中で
最小スタブとして定義している (ggml.c はリンクしない)。

**注意 (実測)**: 最適化レベルを変えると f32 出力が変わる型がある。
`-O0` でビルドすると IQ3_XXS の出力が `-O2` と一致しない (byte 168337 以降)。
commit 済み fixture は `-O2` で生成したものなので、**`-O2` で生成すること**。
TQ1_0 / IQ1_M は `-O0` と `-O2` で出力が一致することを確認済み (2026-10-22)。

## 2. 再生成

```
# 1) 生成 (stderr に各型のバイト数と要素数が出る)
/tmp/dequant_ref /tmp/dequant_ref_new.bin

# 2) 新しい fixture でテストを回す (差し替える前に必ず)
DEQUANT_REF=/tmp/dequant_ref_new.bin \
    cargo test -p denno-engine --release --test dequant_ref -- --nocapture

# 3) 全型 BIT-EXACT を確認してから差し替え
cp /tmp/dequant_ref_new.bin crates/denno-engine/tests/data/dequant_ref.bin

# 4) 既定経路でも通ることを確認
cargo test -p denno-engine --release --test dequant_ref
```

`DEQUANT_REF` は再生成時の比較用の上書きで、既定は
`tests/data/dequant_ref.bin` (テストの cwd は `crates/denno-engine`)。
fixture が無い / セクション数が 21 でない / Q4_0・Q4_0_EDGE・TQ1_0・IQ1_M を含まない
場合はテストが理由を明示して落ちる。

Q4_0_EDGE だけは LCG ではなく `emit_q4_0_edge()` が 6 ブロックを明示的に作る
(負の d / 全ゼロのニブル / 最大ニブル / 負の d + 最大ニブル / 上下別パターン /
d = +0.0)。このセクションの期待値も `dequantize_row_q4_0` の出力そのもので、
手計算はしない。ブロックの d は f16 で厳密な値
(0xBE00 = -1.5 / 0x3800 = 0.5 / 0xBA00 = -0.75 / 0x4000 = 2.0 / 0x0000 = +0.0)。

## 3. 出力形式 (すべて LE)

```
u32 n_sections
各 section:
  u32 name_len, name bytes (ASCII, NUL 無し)
  u32 n_bytes,  raw bytes (量子化ブロック列そのまま)
  u32 n_f32,    f32 * n_f32
```

- 生バイトは決定論的 LCG (`0x12345678` 起点) で作る。型ごとに 16 ブロック。
  乱数系列は型ごとにリセットしない (セクション順に連続)。
  例外は末尾の `Q4_0_EDGE` で、こちらは `emit_q4_0_edge()` が明示的に組む
  (6 ブロック・LCG を消費しない)。
- **既存セクションの順序とバイト列は変えない**。新しい型は末尾に追加する
  (そうしないと過去の検証結果と比較できなくなる)。

## 4. 現在のカバレッジ (21 セクション / 20 型)

| セクション | ブロック | 要素数 | 由来 |
|---|---|---|---|
| Q8_0, Q4_0, Q4_1, Q5_0, Q5_1 | 32 | 512 | P19 (2026-10-14) |
| Q2_K, Q3_K, Q4_K, Q5_K, Q6_K | 256 | 4096 | P19 |
| IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S | 256 | 4096 | P19 |
| IQ1_S, IQ4_XS | 256 | 4096 | P19 |
| IQ4_NL | 32 | 512 | P19 |
| TQ1_0 (ggml type 34) | 256 | 4096 | K2 (2026-10-22) |
| IQ1_M (ggml type 29) | 256 | 4096 | K2 |
| Q4_0_EDGE (type 2 のエッジ列) | 32 | 192 | K3 (2026-10-22) |

型 ID は `ggml.h` の `enum ggml_type` が正 (I8=24 / IQ1_M=29 / TQ1_0=34)。

## 5. ライセンス

- 参照元: llama.cpp / ggml `ggml/src/ggml-quants.c` および `ggml-common.h`
  (iq1s_grid テーブル)。ライセンスは **MIT**
  (Copyright (c) 2023-2026 The ggml authors)。
- 本リポジトリ: **MIT OR Apache-2.0**。
- 本ディレクトリの生成器は llama.cpp の dequantize 関数を**そのままリンクして
  呼ぶ**だけで、コードの取り込み (移植) は行っていない。移植版は
  `crates/denno-engine/src/kquant.rs` にあり、同ファイルのヘッダに MIT の
  著作権表示を置いている (MIT の表示義務を満たす)。
- GPL のコードは一切持ち込まない。fixture (`dequant_ref.bin`) は
  参照実装の**出力データ**であり、コードではない。
- 新しい参照実装を足すときは `docs/history/80_licenses.md`
  「参照実装とライセンス帰属」の表にも追記する。

## 6. 実測記録 (2026-10-22, Task K2 / K3)

- 使用した llama.cpp: `/work/llama.cpp` HEAD `59d5fad`
- 生成器は既存 18 型について旧 fixture を**完全再現**
  (md5 `36acb59aefdb6fd10d2a0fbcaf532fac`。先頭のセクション数 u32 を除く
  232815 バイトが旧 fixture と一致)。
  → 生成パイプライン自体が決定論的であることの確認になる。
- 新 fixture: 20 セクション / 267381 バイト
  (md5 `fe45c5bfee2d0678a5da1f70b86d258e`)。
- `cargo test -p denno-engine --release --test dequant_ref` で
  **20 型すべて BIT-EXACT** (TQ1_0 / IQ1_M を含む)。

### K3 (Q4_0 のエッジ列)

- K3 前の生成器で再生成した fixture は commit 済みの 20 セクション版と
  **md5 一致** (`fe45c5bfee2d0678a5da1f70b86d258e`)。
- Q4_0_EDGE 追加後: **21 セクション / 268278 バイト**
  (md5 `eeff5cb0c11a73db7972e264fae5c2b4`)。既存 20 セクションの生バイトと
  f32 出力は全セクション比較で**不変**。Q4_0_EDGE は 108 生バイト + 192 f32。
- `cargo test -p denno-engine --release --test dequant_ref` で
  **7 passed / 0 failed / 0 ignored** (Q4_0 / Q4_0_EDGE とも BIT-EXACT)。
- 累積順の固定は `crates/denno-engine/src/tensor.rs` の lib テスト
  `q4_gemv_accumulation_order_is_sequential_in_block_order`
  (`matvec_rows_bt_q4_into` の k = 96 の 1 出力を `0xca7b15f7` に固定)。
