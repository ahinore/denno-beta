ADOPT: yes t358b

# REPORT P358 — UD-Q4_K_XL: prefill の expert pread を 32 本へ (3,826 -14.6%、14k -15.0%)

branch: **br358** = int-auto `c1532cf` から。**br354 は merge していない**
(`ensure_ctx_current` はスレッドを変える実験の保険で、本 task はスレッドを
変えないので不要)。作業中に int-auto は動かなかった (最後に fetch して
`c1532cf` のまま = behind 0)。lab は 3 ラウンド: **t358a** (`1ed372b`、
pinned バウンスの仮説) / **t358b** (`027692f`、pread 32 本 → 採否を決めた回) /
**t358c** (`bfb1eed`、既定モデルの確認)。数値はすべて GPU ラボの実測。

## 0. 結論 (3 行)

1. **採る。** `DENNO_PF_PREAD_THREADS` の既定を 0 (rayon 共有プール 14 本) から
   **32 本**にした。lab t358b で **3,826 cold 中央値 12.12 → 10.35 s (-14.6%)**、
   **14k cold 30.25 → 25.71 s (-15.0%)**。機序も出ている: `[exp-pf-stream]` の
   `copy_s` (44.4 GB / 98.2 GB の pread) が 10.2 → 8.2 s / 14.2 → 11.8 s。
2. **pinned バウンスの共有化 (`DENNO_PF_PIN_STAGE`) は採らない。** task 318 の
   thread_local 版を共有 1 本に直して測ったが (t358a)、`other` は 3,826 で
   542 → 567-579 ms、14k で 1,507 → 1,480 ms と**動かない**。この木の `other` は
   もう 3.3 GB/s 出ていて、pageable の仮説 (1.67 GB/s) は古い数字だった。
   ゲートは既定 off のまま残す (切り戻し不要、真理値表はテストで固定)。
3. **UD の prefill は 3,826 が I/O 律速 (44.4 GB の pread、5.4 GB/s)、14k が
   パイプライン律速** (GPU busy 18.3 s / 壁 25.3 s、`slotwait_s` 5.5 s)。
   decode は 18.9 ms/tok (a1) で、内訳は experts 6.7 / mix 5.3 / gate 3.6 /
   gdn 3.5 / router 1.7 / full 1.5 = 残りは .cu の仕事 (下の §7)。

## 1. 前提: base の測り直し (task の Do 1)

int-auto `c1532cf` + タスク指定の 4 env、`PROFILE=1`、UD-Q4_K_XL。lab の
`[pre-kprof]` / `[q4ledger-pf]` / `[exp-pf-stream]` から (t358a/t358b の baseline):

| 段 | 3,826 cold | 14k cold |
| --- | --- | --- |
| 壁時計 | 10.5-12.8 s | 26.9-31.2 s |
| experts (pread + GPU) | 8.2 s | 20.5 s |
| うち `copy_s` (pread) | 7.9-10.4 s (44.4 GB) | 11.7-14.3 s (98.2 GB) |
| `other` | 0.54 s (1.81 GB = 3.3 GB/s) | 1.51 s (5.89 GB = 3.9 GB/s) |
| ssm / mix / full | 0.63 / 0.46 / 0.14 s | 3.56 / 0.02 / 1.30 s |
| PLE (`[pre-kprof]` ple) | 0.76 s | 2.52 s |

- **3,826 は pread が壁の 76-82%**。`[exp-pf-stream]` は
  `layers=48 groups=679 experts=14202 up_MB=44447.0`、`hit_rate=0.322`。
  14,202 本は全部 distinct (同じ expert を 2 回上げていない) ので、
  **バイト数はこれ以上減らせない** (VRAM 22 GB の上限は P215/P248 で確定済み)。
- **14k は `up_MB=98201.8` / `copy_s` 11.7-14.3 s に対して GPU busy 18.3 s**で、
  壁 25.3 s のうち `slotwait_s=5.5 s` (生産側がスロット待ち)。consumer 律速。
- warm short decode (a1/a2) は 2.3-2.6 s / 128 tok = 18-20 ms/tok。t358b の
  pt32_1 の a1 (2,414 ms / 128 = 18.9 ms/tok) の `[q4prof] decode` は
  per-layer mix 0.110 / attn 0.105 / moe 0.178 ms、内訳は ms/128 tok で
  moe router 212.5 / experts 858.0 / shared 23.6、attn gate 454.6 / gdn 452.7 /
  full 188.3 = **per token で experts 6.7 / mix 5.3 / gate 3.6 / gdn 3.5 /
  router 1.7 / full 1.5**。この 6 本で 17.3 ms/tok になる。

## 2. 変えたこと (3 コミット)

| コミット | 何 | ゲート | 既定 | 判定 |
| --- | --- | --- | --- | --- |
| `1ed372b` | pinned バウンスを **共有 1 本** (grow-only、2 の冪切り上げ) に。`DENNO_PF_PIN_MIN_KB` で窓の下限を env 化 | `DENNO_PF_PIN_STAGE` | off | **採らない** (t358a: 壁が動かない) |
| `027692f` | seg 経路の結果 D2H を `to_vec` → `to_slice` (毎層 10 KB の Vec 確保をやめる) | なし (値は不変) | — | 残す (中立。§6) |
| `bfb1eed` | `DENNO_PF_PREAD_THREADS` の既定 0 → **32**、`0` は明示的な切り戻し | `DENNO_PF_PREAD_THREADS` | **32** | **採る** (t358b) |

### 2.1 なぜ 32 本が今効くのか (task 325 との違い)

task 325 は 14 本と 32 本を比べて「3,826 は copy 7.70 → 7.20 s だが壁時計は
動かない」として既定 0 を残した。その時点の 3,826 の壁は 10.3-14.9 s で、
experts 以外 (PLE 0.77 s、層優先の前処理) がまだ重く、pread の 1.6 s の差が
壁に埋もれていた。task 353/357 が PLE と層優先を削った後は、**壁に占める
pread の比率が 76-82%** になり、同じ 32 本がそのまま壁に出る (t358b)。

## 3. lab の結果

### 3.1 t358a (`1ed372b`) — pinned バウンス: **負け**

`SCENARIO=depth8k`、4 COND × 2 ラウンド。baseline (b0t) の 14k は page cache が
冷えていて 58.21 s なので採否に使わない (b0t は捨て COND)。

| cond | 3,826 cold (r1/r2) | 14k cold (r1) | warm short decode | md5 |
| --- | --- | --- | --- | --- |
| b0t | 15.79 / 11.45 | 58.21 | 46/20, 56/20 ms/tok | SAME |
| pin (`PIN_STAGE=1`) | 10.39 / 9.69 | 25.29 | 21/20, 18/18 | SAME |
| pins (`+MIN_KB=4`) | 10.48 / 10.14 | 25.42 | 20/20, 21/18 | SAME |
| b1 | 10.30 / 12.24 | 26.87 | 20/20, 18/18 | SAME |

- `[pf-pin]` は発火している: 3,826 の 1 リクエストで `calls=93 MB=889.0 grows=1`。
  それでも **`other` は 542 → 567/579 ms**、3,826 の壁は b1 と同等。
- 14k の -5.9% は **ホストの外乱** (b1_1 は load 6.48、pin_1 は load 9.44 で
  走っており、`copy_s` が 13.25 → 11.53 s と動いた = ディスクの速度差)。
  同じ run の同じコード (b0t/b1) が 3,826 で 15.79 → 10.30 s と 35% 振れる回で、
  1 本の差は読めない。→ **採らない。**

### 3.2 t358b (`027692f`) — pread 32 本: **勝ち**

`SCENARIO=depth8k`、5 COND × 2 ラウンド、捨て COND (x0) を先頭・baseline を末尾。

| cond | 3,826 cold (r1/r2) | 14k cold (r1) | `copy_s` 3,826 | warm short decode (a1) |
| --- | --- | --- | --- | --- |
| x0 (14 本) | 12.81 / 11.54 | 30.25 | 10.37 / 9.26 s | 3.12 / 2.62 s |
| **pt32** | **10.83 / 9.87** | **25.71** | **8.61 / 7.74 s** | 2.57 / 2.53 |
| pt48 | 9.84 / 11.04 | 34.65 | 7.41 / 8.58 s | 2.53 / 2.54 |
| pin4 | 10.44 / 13.81 | 26.05 | 8.27 / 10.41 s | 2.57 / 2.56 |
| b1 (14 本) | 12.12 / 18.28 | 31.18 | 9.96 / 14.12 s | 2.54 / 2.88 |

- **3,826 中央値: baseline 12.12 (12.81/11.54/12.12/18.28 のうち外れの 18.28 を
  除いた 3 本の中央値) → pt32 10.35 = -14.6%**。両ラウンドで同方向
  (12.81→10.83、11.54→9.87)。
- **14k: 30.25 / 31.18 → 25.71 = -15.0%** (14k は round 1 だけなので 1 本ずつ。
  x0 と b1 の 2 本が 30.25/31.18 で揃っている)。
- **`copy_s` が機序**: 44.4 GB が 10.2 → 8.2 s (4.4 → 5.4 GB/s、baseline 4 本の
  中央値 10.17 対 pt32 の中央値 8.18)、98.2 GB が 14.2 → 11.8 s (6.9 → 8.3 GB/s)。
- 48 本は 3,826 で最速 (7.41 s) だが 14k で 14.11 s と負けた (同じ回の
  c8000_cold の decode が 94 ms/tok で、ホストが詰まった回)。**32 を採る。**
- decode は a1/a2 の生秒が 2.3-2.6 s で 14 本と差なし (分解能 1 ms/tok = 5%)。
- md5 は 5 COND × 2 ラウンド × 全 request で **同一**
  (`a:737a73af b:87d8a97c c2000_cold:f9557457 c2000_turn2:9cbe03ca
  c8000_cold:27c0c9dc c8000_turn2:d88fd518`)、errors 0、text sane (§8)。

### 3.3 t358c (`bfb1eed`) — 既定モデル (swift15-abl-IQ3_S): **何も変わらない**

`SCENARIO=depth`、3 COND × 2 ラウンド。x0/b1 は `DENNO_PF_PREAD_THREADS=0`
(= int-auto の既定)、d32 は env 無し = 新しい既定 32。

| cond | warm short decode | 3,826 cold | c2000_turn2 | md5 |
| --- | ---: | ---: | ---: | --- |
| x0 (=0) | 16.6 ms/tok | 2.51 s | 0.40 s | 65ededc3 / f18c75fc / a91e17da / 5a8f5488 |
| d32 (既定 32) | 16.6 | 2.51 | 0.40 | 同上 (完全一致) |
| b1 (=0) | 16.6 | 2.50 | 0.40 | 同上 |

`VERDICT d32 vs x0 OK decode -0.2% prefill -0.8% md5 SAME errors 0`。
md5 の 4 本 (`65ededc3 / f18c75fc / a91e17da / 5a8f5488`) は、REPORT_P357 が
int-auto の既定モデルで記録した値と**同じ**。既定モデルは `iq_expert_mode` が
真で `pf_expert_mode` の K 系分岐に入らないが、pread の本数はストリーム共通
なので、この 1 ラウンドで「変わらない」を確認した。

## 4. 数値の扱い

- `DENNO_PF_PREAD_THREADS` は**読みの並列度だけ**を変える。pread が書くバイト列は
  同一なので、値は変わらない (t358b の md5 が全 COND で SAME なのがその証拠)。
- 決定性: 同じ COND の 2 ラウンドで md5 が一致。3 ラウンド合計で md5 が
  DIFF した COND は 1 つも無い (t358a/b/c)。
- 本 task は .cu / .ptx を 1 行も変えていない (`git diff int-auto..HEAD` は
  cuda.rs / exp_pf.rs / serve_flash.rs の Rust だけ)。
- ホストの load は 1.5-9.4 と大きく振れた。1 本の差では読まず、同じ run の
  複数 COND と 2 ラウンドの中央値で判定した (§3.2)。

## 5. テスト

- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`
- `cargo test --release -p denno-engine --features cuda --test kernel_invariants`
- 追加/更新したテスト:
  - `cuda::task318_tests::pin_window_and_min_kb_truth_table` (窓 1 MiB..=256 MiB、
    `DENNO_PF_PIN_MIN_KB` の解釈)
  - `cuda::task318_tests::pin_grow_is_grow_only_and_rounded` (grow-only + 2 の冪)
  - `exp_pf::tests::pread_threads_rounds_and_defaults` (既定 32、`0` は切り戻し)
- 注: `--features cuda` を付けない `--lib` は HEAD で既にビルドが通らない
  (int-auto でも同じ。`crate::cuda` が cfg の外から参照されている)。

## 6. 残したが効かなかったもの

- `DENNO_PF_PIN_STAGE` の共有バウンス (`1ed372b`): 既定 off のまま残す。
  P318 の thread_local 版 (+59%) を共有 1 本に直したが、この木では `other` が
  もう 3.3 GB/s 出ていて伸びない (t358a)。真理値表と grow のテストは残した。
- seg 経路の D2H を `to_vec` → `to_slice` (`027692f`): 値は不変で、毎層 10 KB の
  `Vec` 確保が消える。decode の分解能 (1 ms/tok) では差は見えない (t358b の
  pin4 が pt32 と同等)。

## 7. GPU で確認してほしいこと / 次の一手

1. **本番 (8098) の UD-Q4_K_XL で 3,826 / 14k を 2 回ずつ**。`[exp-pf-stream]` の
   `copy_s` が 8.2 s / 11.8 s 前後になり、壁が 10.4 s / 25.7 s 前後になること。
   `DENNO_PF_PREAD_THREADS=0` で従来 (12.1 s / 30.3 s) に戻ること。
2. **14k の `slotwait_s` 5.5 s は未解決**。consumer (GPU) 側が律速なので、
   ステージングの枚数 (`PF_STAGE_SLOTS = 3`、定数) を増やしても効かない見込み
   (推定)。効かせるには .cu 側で kernel を速くするか、VRAM を増やす
   (22 GB の上限は P215/P248 で確定済み) しかない。
3. **decode の残りは .cu の仕事**。warm short decode 18.9 ms/tok (a1) の内訳は
   experts 6.7 (K 型 seg 経路、`[seg-prof]` で GPU kernel 65 us/層 + host 34 us/層) /
   mix 5.3 (97 呼び出し × 5 起動) / gate 3.6 / gdn 3.5 / router 1.7 / full 1.5。
   P354 §8 の (a) gate+up の融合、(c) silu+sum を down に畳む、が候補。
4. 3,826 の pread は 5.4 GB/s で、ディスクの実力 (page cache が効いた回で
   8.3 GB/s) にはまだ届かない。**pread の先読み深さ** (`DENNO_PF_RA_GROUPS`) を
   振る余地はある (本 task では測っていない)。

## 8. text (そのまま引用)

- TXT (UD、t358b pt32): `# B-Trees: A Comprehensive Explanation\n\n## 1. What is a
  B-Tree?\n\nA **B-tree (Balanced Tree) is a self-balancing tree data structure that
  maintains sorted data and allows searches, sequential access, insertions, and
  deletions in logarithmic time.`
- TXTB: 「Rust の「所有権（Ownership）とライフタイム（Lifetime）は、Rust を学ぶ上で
  最も重要であり、かつ最も難しい概念です。」(全 COND 同一)
- TXTD: 「ご提示いただいた文書は、LLM推論エンジン（`denno-engine）における
  **Prefill（事前計算フェーズの性能最適化の歴史と教訓を詳細に記録したものです。」
- errors は 3 ラウンドとも全 COND 0。

## 9. lab の VERDICT 行 (そのまま)

```text
# t358a (1ed372b、UD-Q4_K_XL、depth8k)
VERDICT pin vs b0t OK decode -45.8% prefill -44.6% md5 SAME errors 0
VERDICT pins vs b0t OK decode -44.8% prefill -47.3% md5 SAME errors 0
VERDICT b1 vs b0t OK decode -47.0% prefill -47.3% md5 SAME errors 0
VERDICT baseline b0t errors 0 text ok

# t358b (027692f、UD-Q4_K_XL、depth8k)
VERDICT pt32 vs x0 OK decode -9.4% prefill -39.6% md5 SAME errors 0
VERDICT pt48 vs x0 OK decode -11.7% prefill -39.6% md5 SAME errors 0
VERDICT pin4 vs x0 OK decode -10.6% prefill -40.4% md5 SAME errors 0
VERDICT b1 vs x0 OK decode -4.0% prefill -40.9% md5 SAME errors 0
VERDICT baseline x0 errors 0 text ok

# t358c (bfb1eed、swift15-abl-IQ3_S、depth) — 既定モデル
VERDICT d32 vs x0 OK decode -0.2% prefill -0.8% md5 SAME errors 0
VERDICT b1 vs x0 OK decode -0.5% prefill -0.8% md5 SAME errors 0
VERDICT_DEPTH d32 c2000_cold vs x0 decode -0.4% prefill +0.0%
VERDICT_DEPTH d32 c2000_turn2 vs x0 decode +0.5% prefill +0.0%
VERDICT_DEPTH b1 c2000_cold vs x0 decode +0.0% prefill -0.2%
VERDICT_DEPTH b1 c2000_turn2 vs x0 decode -0.5% prefill +0.0%
VERDICT baseline x0 errors 0 text ok
```

**注意**: lab の VERDICT は**先頭の COND を基準**にするので、t358a/t358b の
`b0t`/`x0` は捨て COND (page cache が冷えた 1 本目) の数字を含む。採否は
§3.2 の同じ run の baseline 2 本 (x0/b1) と、`[exp-pf-stream]` の copy_s で読んだ。
