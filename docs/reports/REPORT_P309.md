ADOPT: yes t309b t309c

# REPORT P309 — UD-Q4_K_XL の prefill expert を K 系カーネルで GPU へ (3,826 cold -42.8%)

branch: br309 = int-auto 0217194 (開始時)。lab: **t309a** (UD、depth、PROFILE=1、
sha d9894b1、PTX=exp_k_pf) / **t309b** (UD、depth8k、PROFILE=1、sha 6c704cd、
committed PTX) / **t309c** (既定モデル、depth、PROFILE=1、sha 4068cd3 = 既定 on)。
最後に int-auto が c8b41a2 (br308) へ動いたので merge した (衝突なし、両側を残した)。

## 0. 結論

- **UD-Q4_K_XL の 3,826 cold prefill は 29.75 -> 17.01 s (-42.8%)** (t309b、baseline 4 本の
  median 29.75 対 kwc 2 本 16.09/17.93 の median 17.01)。target の 30% を満たす。
- **14k cold は 109.1 -> 59.22 s (-45.7%)** (t309b、b0t 109.99 / b1 108.18 対 kwc 59.22)。
  errors 0、text sane (引用は 5 節)。
- **既定モデル (swift15-abl-IQ3_S) は 3 COND で md5 4 本すべて一致・decode +0.4%・
  3,826 cold 3.12 s** (t309c)。int-auto の参照 (tchk0217a) の md5
  (65ededc3 / f18c75fc / d4b6eefb / 8ff9345f) とも一致。
- 変えたのは **prefill の選択 expert を K 系 (Q4_K/Q5_K/Q6_K/Q5_1/Q8_0) でも GPU の
  m 行カーネルで回す**こと。新しい `.cu` (`kernels_exp_k_pf.cu`、5 本) と、
  既存のストリーミング経路 (ステージング + グループのダブルバッファ) への配線。
  ゲート `DENNO_PF_EXP_K` は lab の結果で**既定 on**。行数の少ない expert は CPU に
  残す (`DENNO_PF_EXP_K_MIN_ROWS`、既定 8)。
- 採否の分かれ目は**ホスト→pinned の写し**だった。K 系の層では pinned バウンスの
  write-combined を**型で**やめる (`DENNO_PF_STREAM_WC` 未設定時の既定を K 系だけ
  false) と、写しが 10.3/11.3 s で安定する。WC のままだと 10.3/36.3 s と振れ、
  2 本目の median が baseline を上回った (t309a の 29.58 対 28.57)。

## 1. 前提の測り直し (task の Do 1)

int-auto 0217194 + 既定 env (DENNO_EXPERT_IQ_GPU=1 VRAM=22000 MIN_HIT=0
GPU_WEIGHT_MB=12000)、PROFILE=1 の実測 (lab t309a/t309b の b0t/b0/b1)。

3,826 cold prefill の台帳 (t309b b0t_1):

| 段 | 実測 | 割合 |
| --- | --- | --- |
| experts (CPU、flat MoE) | 92,509 ms (14k) / 24.1-25.9 s (3,826) | 84.1% (14k) / 85% (3,826) |
| other | 11,510 ms (14k) / 2.8 s (3,826) | 10.5% / 16% |
| ssm | 2,276 ms (14k) | 2.1% |
| full / mix / shared / router | 1,515 / 1,095 / 856 / 224 ms (14k) | 1.4 / 1.0 / 0.8 / 0.2% |

REPORT_P301 の構図 (experts が最大、14k で 84.6%) と同じ。experts は 3,826 で
7.22 TMAC / 24.5 s = 295 GMAC/s、14k で 26.4 TMAC / 92.5 s = 285 GMAC/s で、
P301 が「CPU の定式上限」と書いた水準のまま。**ただし内訳は計算だけではない**:
同じ条件の繰り返しで experts が 24.0-25.3 s と ±5% 動き、`[exp-pf-stream]` の
up_MB (44.4 GB) と stage_copy の比から、**mmap からのページフォルト (disk/page
cache) が 3-4 GB/s で効いている**ことが今回わかった (7 節)。

## 2. 変えたこと

### 2.1 K 系の m 行カーネル (`d9894b1`)

`kernels_exp_k_pf.cu` を追加。`kernels_exp_iq_pf.cu` と同じ形:

- 1 起動で 1 expert の m 行を受け、重みを 1 回デコードして m 行へ配る
  (batched GEMV)。1 warp = 2 出力行、1 block = 8 行、`PF_MR=16` 行ずつ。
- 256 要素ブロックの型 (Q4_K 144 B / Q5_K 176 B / Q6_K 210 B) は
  **1 レーン = そのブロックの 8 要素** (lane*8..+7)。
- 32 要素ブロックの型 (Q5_1 24 B / Q8_0 34 B) は **1 レーン = 1 ブロック**、
  1 warp = 1 行、1 block = 4 行。down の k=640 は 256 で割れないのでこの形が要る
  (実際 UD の down は Q5_1)。
- デコードは ggml の式そのまま: Q4_K/Q5_K は `d*sc*q - dmin*m`
  (`get_scale_min_k4` の 6 bit 展開、Q5_K は qh の 5 ビット目を
  `qh[要素] >> サブブロック` で足す)、Q6_K は 128 要素の半分割 + i8 スケール、
  Q5_1 は `d*q5 + m`、Q8_0 は `d*i8`。
- `build_ptx.sh` に `exp_k_pf` を追加 (targets_all と spec、`-fmad` あり =
  prefill の射影はビット一致契約の外)。
- PTX は lab t309a の `PTX=exp_k_pf` で生成したものを入れた (`6c704cd`)。
  5 本とも引数 9 個で .cu と一致 (テストで固定)。

### 2.2 配線とゲート (`d9894b1`、既定 off で入れて lab 後に on)

- `exp_pf`: スロットを IQ 6 + K 5 = 11 に拡張 (`slot` / `kernel_name` /
  `tile_of` / `launch_plan`)。K 系の PTX は**別モジュール**として起動時に読み、
  読めなければそのスロットは null のまま = その型は CPU へ fail closed
  (`moe_gemv_iq_pf_ready` がゲートと関数の有無を見る)。
- ゲート `DENNO_PF_EXP_K` (純関数 `k_from_value` + `OnceLock`、
  `DENNO_EXPERT_IQ_GPU=1` の内側)。**lab 後に既定 on** (`4068cd3`)。
- 行数の床 `DENNO_PF_EXP_K_MIN_ROWS` (既定 8、`0` で床なし)。K 系のときだけ、
  `moe_experts_iq_pf_batch` の中で行数が床未満の expert を CPU の flat に残し、
  GPU の加重和 (`out` を上書き) の上へ足す。実測した UD は 1 chunk の
  rows_per_expert が 129.5 (3,826) / 215.8 (14k) なので床は発火しない
  (短い prefill は既存の `DENNO_PF_STREAM_MIN_TOK=512` で従来どおり CPU)。
- `qwen4exp`: prefill の判定を `iq_expert_mode` から `pf_expert_mode` へ。
  IQ は従来どおり (先に short-circuit)、K 系はゲート + ストリーミング + 型が
  揃ったときだけ真。

### 2.3 write-combined を型で変える (`4068cd3`)

`stream_wc_for_from(v, pf_on, is_k)`: 明示値はそのまま、**未設定のときだけ**
IQ 系は WC (従来)、K 系は通常 pinned。型はストリーミング層の先頭で決めて
`PfStream` に持たせる (env は 1 回だけ読む既存の形を壊さない)。

## 3. 数値の扱い

- **丸めが変わる**。K 系カーネルは 1 行を 32 レーンで分割して木で足す (decode の
  `kernels_seg_k.cu` とも CPU の逐次ともビット一致しない)。prefill の射影は
  ビット一致契約の外 (policy: speed over strict bit identity)。
- 決定性: t309b の kwc と kpg は転送経路が違うのに md5 が
  `b=ffbf4402 / c2000_cold=d6cec0ff / c2000_turn2=17278c2d / c8000_cold=e6d464d8`
  で一致 = 値はホスト側の経路に依らない。t309a の k も同じ md5。
- md5 (t309b): `a` は SAME、`b` / `c2000_cold` / `c2000_turn2` / `c8000_cold` が
  DIFF (K 系の丸め)。`c8000_turn2` は kwc 42ac2301 / b1 63ee88d3 で、
  **baseline の b0t と b1 でも違う** (P301 4 節と同じ、14k 後の decode は
  同じ構成でも再現しない回がある)。
- 既定モデルは IQ 系なので `pf_expert_mode` が `iq_expert_mode` を先に返し、
  新しい分岐に入らない。WC の既定も IQ では従来のまま。

## 4. lab の数字 (すべて実測)

### 4.1 t309a — UD、depth、PROFILE=1、sha d9894b1 (+ PTX 生成)

| cond | 3,826 cold (2 本) | stage_copy | experts |
| --- | --- | --- | --- |
| b0t | 29.88 / 27.87 | - | 25.1 s |
| b0 | 27.71 / 29.16 | - | 24.1 s |
| k (K gate、WC 既定) | **16.59 / 42.57** | 10.32 / 36.33 s | 12.32 / 38.42 s |
| b1 | 28.05 / 29.36 | - | 24.1 s |

`VERDICT k vs b0t OK decode +5.5% prefill +0.0% md5 DIFF errors 0`
(median 29.58 対 28.88 = 負け)。**2 本目だけ写しが 3.5 倍遅い**のが原因で、
同じ 44.4 GB が 1.2 GB/s。ここから「WC の書き込みが冷えたページで詰まる」と
推定して t309b で WC を振った。

### 4.2 t309b — UD、depth8k、PROFILE=1、sha 6c704cd (committed PTX)

| cond | 3,826 cold (2 本) | 14k cold | stage_copy 3,826 | stage_copy 14k |
| --- | --- | --- | --- | --- |
| b0t | 29.87 / 29.63 | 109.99 | - | - |
| kwc (K gate + WC=0) | **16.09 / 17.93** | **59.22** | 10.14 / 11.32 s | 35.31 s |
| kpg (K gate + SRC=page) | 57.10 / 54.45 | 80.11 | 52.01 / 49.92 s | 61.21 s |
| b1 | 29.41 / 41.51 | 108.18 | - | - |

- **3,826: baseline 4 本の median 29.75 -> kwc の median 17.01 = -42.8%**。
  b1 の 2 本目 (41.51 s) は host が詰まった回で、baseline 自体も ±40% 振れる。
- **14k: 109.1 -> 59.22 s = -45.7%** (14k は round 1 だけなので 1 本ずつ)。
- `VERDICT kwc vs b0t OK decode +2.9% prefill -11.0% md5 DIFF errors 0`
- `VERDICT_DEPTH kwc c2000_cold vs b0t decode +60.3% prefill -42.8%`
- `VERDICT_DEPTH kwc c8000_cold vs b0t decode -34.6% prefill -46.2%`
- `VERDICT_DEPTH kpg c2000_cold vs b0t decode -18.4% prefill +87.5%`
- `VERDICT_DEPTH kpg c8000_cold vs b0t decode -31.4% prefill -27.2%`
- `VERDICT baseline b0t errors 0 text ok`

`page` (src=page、pinned を介さず mmap から直接 H2D) は 3,826 で +87.5% と負けた。
pageable の H2D が 1 GB/s 台まで落ちるため (`probe_GBs=1.0`)。**採らない。**
- t309b は採否を決めた SHA (6c704cd) で、kwc は `DENNO_PF_STREAM_WC=0` を**明示**して
  測った。最終 SHA (4068cd3) は K 系のとき同じ値を**既定**にする変更だけで、
  t309b の kwc と起動形・数値経路は同一 (純関数の真理値表テストで固定)。
  lab の上限 3 ラウンドを使い切ったので、最終 SHA で UD を測り直してはいない。

**decode の 2% 判定**: warm short decode (a1/a2、prompt 28、out 128) は
kwc 20.7/18.6/20.5/18.6 ms/tok (median 19.6)、b0t 20.0/18.3/20.1/18.5 (19.2)、
b1 20.2/18.8/20.3/18.3 (19.5)。kwc は b0t 比 +2%、b1 比 +0.5% で、P301 4.3 節と
同じく**この指標は 2% を判定できる分解能が無い** (nt=28 の prefill は
`stream_min_tok=512` に届かず、新しい分岐に一度も入らない。decode のループにも
変更は無い)。
`VERDICT_DEPTH kwc c2000_cold ... decode +60.3%` は kwc_2 の 1 本
(153 ms/tok) で、同じ回の baseline b1_2 も prefill 41.51 s (他は 29.4-29.9) と
詰まっていた。kwc_1 は 49 ms/tok で b0t_1 の 63 ms/tok より速い。
同じ現象は P301 4 節にも記録がある (14k 後の decode は再現しない回がある)。

### 4.3 t309c — 既定モデル (swift15-abl-IQ3_S)、depth、sha 4068cd3 (既定 on)

| cond | warm short decode | 3,826 cold | md5 (a / b / cold / turn2) |
| --- | ---: | ---: | --- |
| off (`DENNO_PF_EXP_K=0`) | 17.9 | 3.12 | 65ededc3 / f18c75fc / d4b6eefb / 8ff9345f |
| on (既定) | 17.9 | 3.12 | 同上 (完全一致) |
| offb (`=0`) | 17.9 | 3.13 | 同上 (完全一致) |

`VERDICT on vs off OK decode +0.4% prefill +0.0% md5 SAME errors 0`
`VERDICT offb vs off OK decode +0.4% prefill +0.0% md5 SAME errors 0`
`VERDICT baseline off errors 0 text ok`

int-auto の参照 (tchk0217a) と md5 が一致 = 既定モデルは何も変わっていない。

## 5. text (TXTD/TXTB、t309b、そのまま引用)

- kwc 3,826 cold の続き: 「ご提示いただいた文書は、LLM推論エンジン
  （`denno-engine）における **Prefill（事前計算フェーズの性能最適化の歴史と
  教訓を詳細に記録したものです。...」
- kwc 14k cold の続き: 同上の書き出しで、段落構成も b0t と同等。
- TXT (a) は "# B-Trees: A Comprehensive Explanation ..."、TXTB は
  「Rust の「所有権（Ownership）とライフタイム（Lifetime）...」で、b0t と同じ書式。
  TXTB の kwc は 2 文目が「メモリ管理をコンパイラが自動でや...」と語が変わる
  (丸めで 1 トークンが入れ替わる。P301 と同じ性質)。
- errors は全 COND 0。

## 6. `other` は何か (task の Do 2 の後半)

14k の `other` 11,510 ms (10.5%) の中身を、台帳の副区間と `pre-kprof` で割った
(t309b b0t_1、すべて実測):

| 内訳 | ms | other 比 |
| --- | ---: | ---: |
| PLE (`pre-kprof` の ple、台帳 sub は 1,026) | 2,793 | 24% |
| embed | 572 | 5% |
| combine (H2D + カーネル + 同期) | 321 | 3% |
| norm ほか | 1 | 0% |
| **未分類 (層ループのホスト仕事)** | **7,823** | **68%** |

- 未分類の正体は「6 つの名前付き区間 (mix / router / shared / full / ssm /
  experts) の外側で層ループが回すホスト時間」で、14,018 トークン × 48 層 =
  672,864 token-layer で割ると **11.6 us/token/層** (task 文の 15-18 us と同じ桁)。
- 中身の候補は、層ループの前後 (res の受け渡し、hc の入出力、KV のホスト側
  ミラーへの追記、layout の詰め直し) と、attention 入口のうち FULL/SSM の
  区間に入っていない部分。**特定はまだ** (層ループに区間を足して測る必要がある)。
- 14k の kwc は同じ other が 1,314 ms (pre-kprof) まで減る。experts が GPU へ
  移って層ループの待ちが減り、PLE/embed の 2.8 s + 0.5 s が残る形になる。

## 7. わかったこと (次の担当者へ)

1. **UD の prefill experts は計算律速ではなかった。** CPU の experts 24-25 s の
   裏で 133 GB (48 層 × 3 chunk × 295 expert × 3.13 MB) を mmap から読んでおり、
   実効 5.5 GB/s。GPU 経路は VRAM 常駐 (ヒット 32-48%) の分だけ読む量が
   44.4 GB に減るので、**同じ 4 GB/s 台の読みでも 12 s で済む**。
2. **pinned バウンスの write-combined は「冷えたページ」で 3.5 倍遅くなる。**
   WC は CPU の書き込みをキャッシュに載せないので、フォールト待ちのレイテンシが
   そのままスループットに出る。K 系だけ WC を外すと 10.3/11.3 s で安定する。
3. 残る最大のコストは **stage_copy (10-11 s / 44.4 GB = 4 GB/s)**。ここを倍に
   できれば 3,826 は 12 s を切る。候補: グループを跨いだ先読み
   (`MADV_WILLNEED` を写しスレッドの外から)、ステージングの 2 枚を
   もっと深くする、`cuMemcpyHtoDAsync` を複数ストリームへ散らす。
4. `DENNO_PF_STREAM_SRC=page` はこのホストでは負け (pageable H2D が 1 GB/s 台)。
   `DENNO_PF_EXP_K_MIN_ROWS` の床は実形では発火しない (129-216 行/expert)。

## 8. テスト

- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`
  **1199 passed / 0 failed / 101 ignored** (int-auto c8b41a2 merge 後)
- `cargo test --release -p denno-engine --features cuda --test kernel_invariants`
  **95 passed / 0 failed**
- 追加したテスト:
  - `cuda::pf_exp_k_tests::k_kernel_index_mirror_matches_cpu_reference` —
    5 型の索引算術を .cu の行から 1 対 1 で写し、`kquant::dequant_row_into` と
    512 要素すべてで一致 (レーンの取りこぼし・nibble の上下・qh のビット位置・
    Q6_K の面分割を固定)。**GPU 無しで検証できるのはここまで**。
  - `cuda::pf_exp_k_tests::ptx_entry_params_counts_the_committed_k_ptx` —
    committed PTX の 5 本が引数 9 個 (ABI)。
  - `exp_pf::tests::k_slots_names_and_plans` — スロット・カーネル名・タイル・
    launch_plan (640 / 2,560 の実形)・row_bytes の割り切り。
  - `exp_pf::tests::k_gate_truth_table` — ゲートと床の真理値表、WC の型別既定。
  - `qwen4exp::lm_dev_tests::pf_expert_mode_k_truth_table`。
- .cu は g++ スタブで構文検査のみ (このコンテナに nvcc が無い)。PTX は lab の
  nvcc (compute_75) が生成したものを commit した。

## 9. GPU で確認してほしいこと

1. 本番 (8098) の UD-Q4_K_XL で 3,826 と 14k を 2 回ずつ。台帳の experts が
   24-25 s -> 12-13 s (3,826) / 92.5 -> 45 s (14k) になり、`[exp-pf-stream]` の
   `copy_s` が 10-11 s で安定すること (WC の既定が K 系で false になっている)。
   `DENNO_PF_EXP_K=0` で従来 (experts 24 s) に戻ること。
2. 冷えた page cache (別モデルを 1 回流した直後) でも 42 s に戻らないこと。
   t309a の k_2 は WC のままだったので 42.57 s だった。
3. 14k の `other` (7.8 s の未分類) の特定: 層ループの前後に台帳の区間を足して、
   PLE/embed/combine 以外が何かを測る。次のタスクの候補。
