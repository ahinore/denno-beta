ADOPT: yes t416b t416c

# task 416 (dev3) — UD-Q4_K_XL の prefill expert を **K 系の tensor core grouped GEMM** で
# (14k cold 25.2 -> 18.0 s = -28.6%、既定モデルは md5 全一致)

branch: **br416** = 開始時の int-auto `d9a87bc` (merge: br410) から。
**br412 は取り込んでいない** (`ADOPT: no` で全部が既定 off のゲートと seg の
診断だけ。この task の変更はそのどれにも依存しない)。終端でも int-auto は
`d9a87bc` のまま (動いていない) なので merge は無い。

数値はすべて GPU ラボの実測 (`PROFILE=1`、交互 2 ラウンド)。lab 名と測った SHA:
**t416a = `d9a87bc`** (UD、depth8k、3 COND、baseline と段のプロファイル)、
**t416b = `a2bf046`** (UD、depth8k、4 COND、`PTX=exp_k_tc`)、
**t416c = `eea10b3`** (既定モデル `flash-next-UD-IQ3_XXS`、depth8k、4 COND、
`PTX=` 無し = コミット済み PTX)。3 ラウンド。

## 0. 結論 (5 行)

1. **採る (`ADOPT: yes`)。** UD-Q4_K_XL の **14k cold prefill は 17.98 s** 対
   baseline 25.18 / 25.22 / 24.96 s = **-28.6%**。タスク目標 (-5%) の 5.7 倍。
2. **3,826 cold も悪化しない。** 壁はもう GPU ではなく 44.4 GB の pread
   (`copy_s`) で決まる。experts 段の gpu_busy は 4,700.6 -> **430.5 ms**
   (11 分の 1) で、読みが 7.2 s のラウンドは 8.76 s 対 baseline 9.13/9.17 s。
   読みが 12.6 s に振れたラウンドだけ 14.68 s になった (ディスクの振れ)。
3. **やったこと**: `kernels_exp_k_tc.cu` (10 本) を追加し、prefill の選択 expert
   グループを IQ 系と同じ 64 トークン x 64 出力行の bf16 mma で回す。
   ゲート `DENNO_PF_TC_EXP_K` は lab の結果で**既定 on** にした。
4. **既定モデルは無変化。** 4 COND x 6 文脈の md5 が**全部 SAME**
   (`edaa84ad / 1938059f / b42d79d2 / 12fc5c36 / 672fec07 / 98e879d2`)、
   decode 13.2-13.4、3,826 2.34-2.53 s、14k 5.71-6.45 s、errors 0。
   既定モデルの expert は IQ 系なので `ktc_shapes_ok` が偽 = **構造的に不発**
   (`[pf-ktc]` の行も t416c のログに出ていない)。
5. **UD の decode は不変** (16.3 対 16.4 ms/tok)。このカーネルは prefill 専用で、
   decode は seg 経路のまま。md5 は c2000 の 2 文脈だけ DIFF (bf16 丸め)、
   c8000 の 2 文脈は SAME。

## 1. 前提の測り直しと、なぜこの塊を選んだか (Do 1 / Do 2)

### 1.1 現行 int-auto の UD-Q4_K_XL (`d9a87bc`、t416a)

| 条件 | 実測 |
| --- | --- |
| warm short decode | b1 16.5 / b2 16.2 ms/tok |
| 3,826 cold (c2000_cold) | b1 9.15 / b2 10.68 s |
| 14k cold (c8000_cold) | b1 25.22 / b2 25.18 s |
| 14k 後 decode | 38.6 / 33.8 ms/tok |
| errors / text | 0 / 健全 |

### 1.2 段のプロファイル (t416a の x0 = 捨て COND、`DENNO_PF_LM_DBG=1`)

14k cold の台帳 (`[q4ledger-pf]`、`tok=14018 wall=27045.4ms`):

| 段 | wall | 割合 | 内訳 (busy / exposed) |
| --- | ---: | ---: | --- |
| experts | 20,330.6 ms | 75.2% | **18,339.5 / 1,991.1** |
| ssm | 3,162.8 | 11.7% | 1,314.8 / 1,848.0 |
| other | 2,403.5 | 8.9% | 171.4 / 2,232.1 (ple 631 / embed 323) |
| full | 1,123.0 | 4.2% | 639.6 / 483.4 |
| mix | 25.5 | 0.1% | 881.7 / 0.0 |

3,826 cold (`tok=3826 wall=12916.9ms`、プロファイラ込み) は experts 10,820.3 ms の
うち **busy 4,761.3 / exposed 6,058.9** で、`[pf-host] up_wait=10231.9` と
`[exp-pf-stream] up_MB=44,409 pread_calls=42,582` から**読み (pread) が律速**。

**選んだ理由**: 14k の最大の塊は experts 段の GPU (18.34 s = 壁の 67.8%) で、
これは「まだ誰も触っていない」形だった。根拠は 2 つ。

- 6,728,640 行 (rows_per_expert=215.3) x 3 行列 x 2560 x 640 MAC =
  **3.3e13 MAC = 6.6e13 FLOP** を 18.34 s で回しており、**3.6 TFLOPS**
  = 5090 の fp32 (104.8) の **3.4%** しか出ていない。
- 現行の m 行 GEMV (`kernels_exp_k_pf.cu`) は 1 block = 8 出力行なので、1 expert
  あたりの活性の読み直しは gate 176 MB + up 176 MB + down 176 MB = **528 MB**
  (m=215、e=2560、ff=640)。重みは 1 expert あたり 2 MB なので**活性律速**で、
  31,247 expert で **16.5 TB** を読んでいる計算になる。同じ形の IQ 系は
  `kernels_exp_iq_tc.cu` (task 250、既定 on) で既に 25.1 -> 9.9 s にしていた。

## 2. 変えたこと

1. **`kernels_exp_k_tc.cu` (新規、10 エントリ)**。`kernels_exp_iq_tc.cu` と
   同じ body (TC_BM=64 / TC_BN=64 / TC_BK=32 / TC_THREADS=256、bf16 wmma
   16x16x16、FP32 アキュムレート、重みは smem に 1 回だけデコード、x は
   64 行タイルで 1 回) で、**デコードだけ** K 系 5 型に差し替えた。
   各式は `kernels_exp_k_pf.cu` の `kpf_w8_*` と 1:1 (レーンの `lane*8` を
   タイル内の `j*8` = 連続 8 要素に置き換えただけ)。
   - Q4_K / Q5_K: `get_scale_min_k4` の 6 bit 展開 + `d*sc*q - dmin*m`
   - Q6_K: 128 要素の半分割 + i8 スケール
   - Q5_1: `d*(q + 16*hb) + m` (下ニブル/上ニブル + qh の 32 bit)
   - Q8_0: `d*i8`
2. **cuda.rs**: K 版のモジュールを起動時に 1 回読み (`DENNO_EXP_KTC_PTX` で
   差し替え可)、**引数の数 (gu 10 / dn 9) を ABI ゲートで確認**してから使う
   (合わなければ従来 GEMV へ fail closed)。`ktc_shapes_ok` (純関数) と
   `TcSet` (IQ / K の関数引き) を足し、`pf_launch_group_tc` は
   `(gu_fn, dn_fn)` を受ける形にした。
3. **exp_pf.rs**: ゲート `DENNO_PF_TC_EXP_K` (純関数 `tc_exp_k_from_value`)。
   最初は既定 off で入れ、lab t416b の結果で**既定 on** にした
   (`DEFAULT_TC_EXP_K_ON`)。明示的な偽値 (`0`/`off`/`false`/`no`) で切り戻せる。
4. **発火の証拠**: 最初に K 版を選んだとき 1 行
   `[pf-ktc] K 系の tensor core グループ GEMM を使う (gate/up=Q4K down=Q5_1 k=2560/640 n=640/2560)`。
   実機の型は **gate/up = Q4_K、down = Q5_1、ff=640** (`ff=1536` と書いた
   見積りは誤り。実測で 640 と分かった)。
5. **build_ptx.sh**: target `exp_k_tc` (`fmad`、`compute_90`) を足した
   (`targets_all` にも 1 語)。PTX は lab t416b の生成物をそのまま commit。

## 3. 数値の扱い

- **値は変わる。** 重みを bf16 に丸め、活性も bf16 に丸めて mma する
  (アキュムレートは FP32、行ごとの和は k 昇順のまま)。prefill の射影は
  2026-10-28 のオペレーター指示で**ビット一致契約の外**で、IQ 系の TC と
  同じ扱い。decide の seg 経路・decode の値は 1 ビットも変わらない。
- 決定性: 木の形は固定 (タイル幅 32 の逐次 + FP32 の mma)。同じ入力なら同じ値。
- md5 (t416b、UD): `a` / `b` は SAME、`c2000_cold` `f9557457 -> d6cec0ff`、
  `c2000_turn2` `fbaf5055 -> c519179d` が DIFF (意図した丸め)、
  `c8000_cold` `20779eab` と `c8000_turn2` `e6398d57` は **SAME**。
- md5 (t416c、既定モデル): 4 COND x 6 文脈が**全部 SAME**。

## 4. lab t416b: UD-Q4_K_XL (sha `a2bf046`、`ptx=exp_k_tc`)

`load_before=1.98`、`[pf-ktc]` の発火行あり (両ラウンド)。

| cond | env | decode ms/tok | 3,826 cold r1 / r2 | 14k cold | 14k 後 decode |
| --- | --- | ---: | --- | ---: | ---: |
| x0 (捨て) | 基準 | 17.4 | 14.46 / 10.08 | 24.96 | 39.1 |
| b1 | 基準 | 20.4 | 9.13 / 9.17 | 25.22 | 38.6 |
| ktc | `PF_TC_EXP_K=1` | **16.3** | **14.68 / 8.76** | **17.98** | 35.0 |
| b2 | 基準 | 16.4 | 11.44 / 9.93 | 25.18 | 33.8 |

- **14k cold は -28.6%** (17.98 対 25.18/25.22/24.96)。3 本の baseline が
  ±0.5% に収まっているので、この差はノイズではない (14k は depth8k の仕様で
  **ラウンド 1 にしか流れない**ため、COND あたり 1 本の測定)。
- **3,826 cold は読み律速**。round 2 は ktc 8.76 s 対 b1 9.17 s = -4.5% だが、
  round 1 は ktc の pread が 12.61 s に振れて 14.68 s。同じ COND の 2 ラウンドで
  壁が 5.9 s 動いており、**ディスクの振れが COND 間の差より大きい**
  (`[exp-pf-stream]` の copy_s は x0 12.30 / b1 7.19 / ktc 12.61 / b2 9.53 /
  2 ラウンド目 7.31 / 7.26 / 7.20 / 7.97)。
- experts 段の gpu_busy は **4,700.6 -> 430.5 ms** (3,826、b1 対 ktc round 2)、
  **18,339.5 -> 432** (14k、t416a x0 対 ktc)。起動数は 15,043 -> 12,217
  (14k) / 5,445 -> 4,095 (3,826) = 群あたり「expert 数 x 3 本」-> 「2 本」。
- 3,826 の `mix` 段は 565 -> 260 ms に縮んだ (GPU が空くので待ちが減る)。
- decode は 16.3 対 16.4 ms/tok (b1 の 20.4 は、そのラウンドの md5 欄に `a:` が
  2 本ある = 再送があった回で、外れ値と見る)。
- errors 0。text は全 COND 健全 (§7)。

### VERDICT (t416b)

```
VERDICT b1 vs x0 OK decode +17.3% prefill -11.9% md5 DIFF errors 0
VERDICT ktc vs x0 OK decode -6.3% prefill -10.4% md5 DIFF errors 0
VERDICT b2 vs x0 OK decode -5.8% prefill -9.6% md5 DIFF errors 0
VERDICT_DEPTH b1 c2000_cold vs x0 decode -76.8% prefill -25.4%
VERDICT_DEPTH b1 c8000_cold vs x0 decode -1.3% prefill +1.0%
VERDICT_DEPTH ktc c2000_cold vs x0 decode -33.8% prefill -4.5%
VERDICT_DEPTH ktc c8000_cold vs x0 decode -10.6% prefill -28.0%
VERDICT_DEPTH b2 c8000_cold vs x0 decode -13.7% prefill +0.9%
VERDICT baseline x0 errors 0 text ok
```
(VERDICT は x0 比。x0 は捨て COND なので、判定は本文の baseline b1/b2 との
比較で読む。`ktc c8000_cold -28.0%` が x0 24.96 に対する数字。)

## 5. lab t416c: 既定モデル (sha `eea10b3`、`ptx=none` = コミット済み PTX)

| cond | env | decode ms/tok | 3,826 cold | 14k cold |
| --- | --- | ---: | ---: | ---: |
| x0 (捨て) | 基準 | 13.4 | 5.49 | 7.99 |
| b1 | 基準 | 13.4 | 2.53 | 6.45 |
| ktc | `PF_TC_EXP_K=1` (明示 on) | 13.2 | 2.34 | 5.71 |
| b2 | 基準 | 13.2 | 2.38 | 5.93 |

- **md5 は 4 COND x 6 文脈の全部が SAME。** ゲートを明示 on にしても既定モデルは
  1 バイトも変わらない (IQ 系なので `ktc_shapes_ok` が偽、`[pf-ktc]` の行も無し)。
- decode 13.2-13.4、3,826 2.34-2.53 s、14k 5.71-6.45 s で、差はノイズの内側
  (ktc が最速に見えるが、ゲートが発火していない以上ただの順序効果)。
- errors 0、text 健全。

### VERDICT (t416c)

```
VERDICT b1 vs x0 OK decode +0.0% prefill +0.0% md5 SAME errors 0
VERDICT ktc vs x0 OK decode -1.5% prefill -0.9% md5 SAME errors 0
VERDICT b2 vs x0 OK decode -1.2% prefill -0.9% md5 SAME errors 0
VERDICT_DEPTH ktc c2000_cold vs x0 decode -21.4% prefill -57.4%
VERDICT_DEPTH ktc c8000_cold vs x0 decode -10.9% prefill -28.5%
VERDICT_DEPTH b2 c8000_cold vs x0 decode -12.5% prefill -25.8%
VERDICT baseline x0 errors 0 text ok
```

## 6. 見積り (推定。実測ではない)

- 6.6e13 FLOP を 18.34 s で回していたのが 3.4% の効率で、律速は活性の再読み
  (16.5 TB)。タイル化で活性の読みは 8 分の 1 (2.1 TB) になり、実測 -28.6% は
  「重み 2 MB + 活性 2.1 TB を 5090 の DRAM/L2 で流す」形に近づいたと読む。
- 14k の残りは gpu_busy の内訳で ssm (1.31 s) / full (0.64 s) / other (2.23 s
  exposed) / ple (0.63 s)。次に効くのは ssm と other で、TC 化で experts が
  0.43 s になった以上、**prefill の最大の塊はもう expert ではない**。
- 3,826 の床は 44.4 GB の pread。読みが 7.2 s で安定する限り、これ以上 GPU を
  速くしても壁は 8.7 s 付近で止まる (推定)。
- 3,826 の読みが 12.6 s に振れた回の原因 (staging のスロット数 / page cache) は
  この task では切り分けていない。同じ COND の 2 ラウンドで 5.9 s 動くので、
  次の task で 3,826 を判定に使うなら copy_s を必ず併記すべき。

## 7. text (抜粋)

- UD ktc (c2000_turn2、md5 c519179d):
  `"ご提示いただいた文書は、LLM推論エンジン（`denno-engine）における **Prefill（事前計算フェーズの性能最適化の歴史と技術的深掘り**を記録したものです。\n\n以下に、文書の構造、主要な発"` (健全。baseline は「歴史と教訓を詳細に記録したもの」で、語が 2 つ違うだけ)
- UD ktc (b、md5 87d8a97c):
  `"Rust の「所有権（Ownership）とライフタイム（Lifetime）は、Rust を学ぶ上で最も重要であり、かつ最も難しい概念です。\n\nしかし、これらは「メモリ管理をコンパイラが自動的に行"` (98 文字、健全)
- 既定モデル: 4 COND とも `"# B-Trees: A Comprehensive Explanation..."` /
  上と同じ Rust の文ではじまり、md5 も SAME。

## 8. テスト (最終状態)

- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`:
  **1,293 passed / 0 failed / 103 ignored**
- `... --test kernel_invariants`: **120 passed / 0 failed**
- cuda 無しの CI 相当: `cargo build --workspace --release` 通過、
  `cargo test --workspace -- --test-threads=1` **rc=0 / failed 0**
- 追加 5 本 (cuda::pf_exp_ktc_tests):
  - `ktc_kernel_index_mirror_matches_cpu_reference`: 5 型 x 512 要素について
    `.cu` の索引算術を Rust へ写し、`kquant::dequant_row_into` と要素ごとに
    比較 (相対 1e-5)。**GPU 無しで索引の取り違えを落とす**ためのテスト。
  - `ktc_shapes_ok_pins_the_contract` / `tc_exp_k_gate_defaults_on` /
    `ktc_tile_constants_match_the_cu` / `ktc_ptx_has_the_ten_entries`
    (コミット済み PTX の 10 エントリと引数の数 10 / 9)。
- `.cu` はこのコンテナで nvcc が使えないので、構文と PTX は lab の生成に
  依存する (t416b で `ptx=exp_k_tc` の生成と実機 JIT が通っている)。

## 9. GPU で確認してほしいこと / 残し

1. 既定 on にした。切り戻しは `DENNO_PF_TC_EXP_K=0` (純関数テストで固定)。
2. 次に効くのは 14k の ssm (1.31 s、既に task 397/407 の領分) と other 段の
   exposed 2.23 s (ple 0.63 / embed 0.32)。experts はもう 0.43 s。
3. 3,826 は読み律速 (44.4 GB)。次に 3,826 を判定に使うときは copy_s を併記する。
