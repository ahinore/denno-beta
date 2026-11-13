ADOPT: yes t285c

# REPORT P285 — IQ3_S decode: GDN の beta/alpha を GPU へ、router の top-k を挿入法へ

br285 = int-auto `c5cb2939c` (task 282 の merge 後)。br281 は**採っていない**が、この
タスクの 4 つのうち `DENNO_EXP_SCRATCH` が br281 の変更なので `br281` を merge して
取り込んだ (1 つ目の commit の親。衝突なし)。int-auto はこの作業中に動かなかった
(最終確認でも `c5cb2939c` = merge 不要)。

## 0. 結論

**ADOPT: yes。** 既定を 4 つ変えた (すべて env で切り戻せる)。

| gate | 前 | 後 | 単独の実測 (t285a) |
| --- | --- | --- | --- |
| `DENNO_GDN_BA_GPU` | off | **on** | decode -2.5% |
| `DENNO_DEC_ROUTER_TOPK_FAST` (新規) | — | **on** | decode -1.0% |
| `DENNO_EXP_SCRATCH` | off | **on** | (ノイズ床以下、bit 一致) |
| `DENNO_GEMV_V5_LANES` | 8 | **32** | decode -1.0% |

t285c (depth8k、base / all_a / allv5_a / all_b / allv5_b / base2 の交互 6 COND、
各 2 ラウンド): 短い decode は base `18.8` -> **all (`18.0` / `18.0`) -4.0 / -4.4%**、
allv5 (v5 lane も 32) **`17.9` / `17.9` -5.0 / -4.8%**、base2 `18.8` (+0.2%、この回の
ノイズ床)。prefill は短い点 +0.8%、14k は base2 と同値。errors 0、text 健全。
**最終ツリーの既定 = allv5** (t285c の `-5.0 / -4.8%` がそのまま最終状態)。

## 1. 前提 (測り直した値)

int-auto の短い decode は task 文のとおり 18.1〜18.3 ms/tok (t285a の base は 18.9、
この回の最初の COND バイアスを含む)。**per-stage は lab の `[q4ledger]` を
そのまま読んだ** (t285a base_1/base_2 の暖まった窓):

| 区間 | wall | うち GPU | host cpu | 起動/同期 |
| --- | --- | --- | --- | --- |
| mix | 3.75-3.80 | 2.72-2.74 | 0.16-0.20 | 388 起動 / 97 同期 |
| router | 1.32-1.33 | 0.71-0.72 | **0.38-0.39** | 48 起動 / 48 同期 |
| shared | 0.63-0.64 | 0 (非同期) | 0.26 | 192 起動 |
| full | 1.70 | 1.19 | 0.32 | 76 起動 / 18 同期 |
| ssm | 4.27-4.41 | 0 (窓の外) | **0.88-0.96** | 125 起動 |
| experts | 3.5-4.4 | — | — | 層一括 |
| head | 0.74 | 0.70-0.71 | 0.03 | 1 起動 |

= **ホスト側で大きいのは ssm の 0.9 ms (beta/alpha の GEMV) と router の 0.4 ms
(top-k の選択)**。GPU 側 (mix 2.7 / full 1.2 / experts / head 0.7) は task 275/278/282
の後でどれも「その形の実測最良帯域」の近くに居るので、このタスクは**ホスト仕事**を
取った (task 文の「waits, launches, host work between stages」)。

## 2. 変えたこと (4 つ、すべて env ゲート)

### 2.1 `DENNO_GDN_BA_GPU` を既定 on (task 269 の実装)

ssm 層の beta/alpha (BF16・n=96・491 KB/層) をホストの GEMV で回していたのを、
GDN submission の中で GPU の `dno_matvec_bf16_r` 1 発に置き換える。36 層 x 21.8 us
= **0.785 ms/token** が消える (重みは 18 MB で VRAM 常駐、v5 の lane 対象外)。

- 丸め順が変わる (1 行 = 1 ブロック 256 スレッドの固定木) → **md5 DIFF は想定内**。
  決定論は保つ。切り戻しは `DENNO_GDN_BA_GPU=0`。
- task 269 では目標 8% に届かず既定 off のままだった。今の base は他の段が縮んで
  同じ 0.785 ms の相対価値が上がっている、というのが既定 on にする理由。
- `gate_audit` の台帳に `DENNO_GDN_BA_GPU` -> `kernels_elem.ptx`
  (`dno_matvec_bf16_r` / `dno_matvec_f32_r`) を足した (PTX が欠けたら起動ログに WARN)。

### 2.2 `DENNO_DEC_ROUTER_TOPK_FAST` (新規・既定 on) — decoder の top-k を挿入法へ

decode の router は `select_topk` = `select_nth_unstable_by` 版で、ne=512 / k=8 の
1 行に 4.2 us かかっていた (task 195 の相対マイクロベンチ)。prefill 用に既にある
**挿入法** `select_topk_into_fast` (0.68 us/行) を decode でも使う。

- 選択規則は同一 (値の降順・同値は小さい index・出力は index 昇順) なので**値は
  1 ビットも変わらない**。NaN 入り・k > 32 は従来版へ落ちる (既存の実装のまま)。
- 発火は `DEC_ROUTER_TOPK_FAST_CALLS` (テスト専用カウンタ) で数える。

### 2.3 `DENNO_EXP_SCRATCH` を既定 on (br281)

`moe_experts_iq_gpu_batched` の `h = vec![0.0f32; nseg*e]` (実形 40 KB) を、sumdev 経路
(既定 on) では 1 度も読まないので確保しない。値は不変 (48 層で 1.9 MB/token の
ゼロ埋めが消える)。P281 §6 が「次の decode タスクが既定 on を前提にしてよい」と
申し送っていた項目で、bit 一致なので積んでも risk が無い。

### 2.4 `DENNO_GEMV_V5_LANES` の既定を 8 -> 32 (br281 の実装)

v5 GEMV の lane 上限。Q8_0 k=6144 (`per_unit`=96) で lanes 8 -> 32 = block 64 -> 256 に
なり、1 レーンの依存ロード連鎖が 12 -> 3 対になる。**丸め順だけの変更**で、t281a /
t285a / t285c の md5 はすべて SAME (all と allv5 の md5 は完全一致)。切り戻しは
`DENNO_GEMV_V5_LANES=8`。

## 3. lab (IQ3_S、depth8k、PROFILE=1、各 COND 2 ラウンド)

### t285a — SHA `fca5e3b1b`、6 COND (base / ba / topk / v5l / all / base2)

| COND | 短い decode | prefill | md5 | 中身 |
| --- | --- | --- | --- | --- |
| base (最初) | 18.9 | 0.59 s | 基準 | 4 ゲートとも off |
| ba | 18.4 | 0.60 | DIFF | beta/alpha GPU |
| topk | 18.7 | 0.59 | SAME | 挿入法 |
| v5l | 18.7 | 0.59 | SAME | lanes 32 |
| all | **18.0** | 0.59 | DIFF | ba + topk + scratch |
| base2 | 19.0 | 0.59 | 基準 | 4 ゲートとも off |

`VERDICT ba vs base OK decode -2.5% prefill +0.8% md5 DIFF errors 0`
`VERDICT topk vs base OK decode -1.0% prefill -0.8% md5 SAME errors 0`
`VERDICT v5l vs base OK decode -1.0% prefill +0.0% md5 SAME errors 0`
`VERDICT all vs base OK decode -4.5% prefill +0.0% md5 DIFF errors 0`
`VERDICT base2 vs base OK decode +0.6% prefill -0.8% md5 SAME errors 0`

台帳の裏取り (同じ回): `ssm` の wall 4.27-4.57 -> **3.55-3.72**、ssm の host cpu
0.88-1.10 -> **0.06-0.15** (ba の COND で既にこの値)、`router` の cpu 0.39 -> **0.21**
(topk の COND)、`mix` / `full` / `experts` / `head` は変わらない。

### t285b — 同じ SHA、4 COND (base / all / allv5 / base2)

base 18.8 / all 19.0 / allv5 **17.9** / base2 20.0。**この回は使わない**: `base2` が
`base` より +6.4% (同じコード) で、走行中にホストが重くなった (load_before 3.05)。
`all` の +1.0% も base2 との比較では -5.4% で、ドリフトに埋もれている。

### t285c — 同じ SHA、6 COND を交互に (base / all_a / allv5_a / all_b / allv5_b / base2)

| COND | 短い decode | prefill | md5 |
| --- | --- | --- | --- |
| base (最初) | 18.8 | 0.59 s | 基準 |
| all_a | 18.0 (-4.0%) | 0.60 | DIFF |
| allv5_a | **17.9 (-5.0%)** | 0.60 | DIFF |
| all_b | 18.0 (-4.4%) | 0.60 | DIFF |
| allv5_b | **17.9 (-4.8%)** | 0.60 | DIFF |
| base2 (最後) | 18.8 (+0.2%) | 0.60 | 基準 |

`VERDICT all_a vs base OK decode -4.0% prefill +0.8% md5 DIFF errors 0`
`VERDICT allv5_a vs base OK decode -5.0% prefill +0.8% md5 DIFF errors 0`
`VERDICT all_b vs base OK decode -4.4% prefill +0.8% md5 DIFF errors 0`
`VERDICT allv5_b vs base OK decode -4.8% prefill +0.8% md5 DIFF errors 0`
`VERDICT base2 vs base OK decode +0.2% prefill +0.8% md5 SAME errors 0`

深い点 (`VERDICT_DEPTH`、c8000_turn2 = 14k の次): base 18.9 / base2 18.4 /
all_a 17.9 / all_b 16.4 / allv5_a 18.0 / allv5_b 16.3 ms/tok。**同コードの base と
base2 が 18.9 と 18.4 で、深い点は順序バイアスが ±3% 残る**ので「悪化していない」
以上は読めない。c2000_cold の prefill が COND 間で 3.4〜8.5 s と動くのも同じ理由
(前の COND が expert / page cache を温める)。短い点の prefill は 0.59 -> 0.60 s
(+0.8%、2% 以内)、14k の prefill は `all` 8.36 s = `base2` 8.36 s で不変 (t285a)。

本文 (`allv5_a`、t285c): "A **B-tree** (specifically, the **B+ tree) is the
fundamental data structure used by most modern relational databases (such as MySQL,
PostgreSQL, Oracle, and SQL Server)..." / "Rust の「所有権（Ownership）とライフタイム
（Lifetimes）は、Rust の最も重要な概念であり...」" — 健全。

## 4. 数値の扱い

- `DENNO_GDN_BA_GPU`: **意図した丸め順の変更** (beta/alpha の縮約が固定の木になる)。
  md5 は 6 本文すべて DIFF。決定論・行チャンク不変・par/seq 一致は触っていない。
- `DENNO_DEC_ROUTER_TOPK_FAST` / `DENNO_EXP_SCRATCH`: **1 ビットも変わらない**
  (選択規則が同一 / 未読の確保をやめるだけ)。lab の md5 も SAME。
- `DENNO_GEMV_V5_LANES=32`: 丸め順だけの変更。t285a の `v5l` は SAME、
  t285c の `all` と `allv5` は md5 が**完全に一致**。
- 「md5 DIFF」は ba の丸めだけで、`all` と `allv5` の出力は同一。

## 5. 見積り (推定) と実測

| 変更 | 見積り | 実測 |
| --- | --- | --- |
| beta/alpha を GPU へ | -0.7 ms (-3.7%) | **-0.5 ms (-2.5%)**、台帳の ssm は -0.7〜-0.9 ms |
| router top-k 挿入法 | -0.2 ms (-1.1%) | **-0.17 ms (-1.0%)** (router cpu 0.39 -> 0.21) |
| expert 40 KB 確保除去 | -0.05 ms | ノイズ床以下 (bit 一致) |
| v5 lanes 32 | -0.1 ms (-0.5%) | **-0.1 ms (-0.6%)** (18.0 -> 17.9) |
| 4 つ積む | -1.05 ms (-5.5%) | **-0.9 ms (-4.9%)** (t285c の allv5) |

単独 COND の ba が -2.5% と小さいのは、t285a の 2 番目の COND で最初のバイアスが
残っているため (台帳では -0.7〜-0.9 ms 出ている)。4 つの和 (-4.0%) より積んだ方
(-4.9%) が良いのも同じ理由で、各単独値は ±0.5% のノイズを含む。

## 6. テスト

- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`
  → **1177 passed / 0 failed** (100 ignored)
- `cargo test --release -p denno-engine --features cuda --test kernel_invariants`
  → **94 passed / 0 failed**
- 追加/更新:
  - `dec_router_topk_fast_matches_select_nth` — 挿入法と select_nth 版で MoE の出力が
    **ビット一致**し、分岐が実際に通る (カウンタ) ことを固定。
  - `dec_router_topk_fast_truth_table` / `exp_scratch_gate_truth_table` (既定 on) /
    `gdn_ba_gpu_truth_table` (既定 on) / `v5_lanes_cap_table` (既定 32)。
  - `gate_audit` の台帳に 1 行追加 (`every_listed_gate_is_on_by_default` と
    `listed_ptx_exist_with_entries` が新しい既定を検査する)。

## 7. GPU で確認してほしいこと

1. `DENNO_GDN_BA_GPU=1` で `[q4attn]` の `gate` が `gdn` とほぼ同じ値になる
   (= ホストの beta/alpha が消えた証拠)。`[q4ledger] ssm` の cpu が 0.9 -> 0.1 ms/tok。
2. md5 が base と DIFF になるのは ba の丸めだけ。`DENNO_GDN_BA_GPU=0` に戻すと
   base の md5 (`a:82ab0c06 b:624e0d4d ...`) に戻ること。
3. **残る risk**: `ba_gpu_weight` が VRAM を確保できないと `gdn_launch_ba` が Err を
   返し、`gdn_decode_submit` の契約どおり**その要求は落ちる** (型の fail-closed は
   呼び出し側で済ませてあるが、確保失敗はここでは拾っていない)。36 層で 18 MB なので
   実測 6 条件 x 2 ラウンドで errors 0 だったが、VRAM が厳しい条件では
   `DENNO_GDN_BA_GPU=0` で切り戻すのが安全。

## 8. merge

int-auto は `c5cb2939c` のまま (この作業中に動かず、最終確認でも同じ) = merge 不要。
`build_ptx.sh` の `targets_all` は触っていない (新しいカーネルを足していない。
ba は既存の `kernels_elem.ptx`、v5 lanes は既存の `kernels_moe_v5.ptx`)。
