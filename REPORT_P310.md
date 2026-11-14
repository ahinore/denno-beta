ADOPT: no t310a t310b

# REPORT P310 — IQ3_S decode: shared expert を v5 の GEMV にしても -2.5% (target 4% に届かず)

branch: br310 = int-auto `d6f914e97` (= br307 の merge 後) から。br307 は既に int-auto に
入っていたので merge していない。作業中に int-auto は動かず (最後に fetch し直して
`d6f914e97` のまま)、merge は無い。lab 2 ラウンド: **t310a** (`d7103e021`、5 COND、
`DENNO_DEC_SH_V5` の 1/2 を切り分け、先頭 COND に診断 env) / **t310b**
(`d7103e021`、5 COND、base と候補を交互に 2 サンプルずつ)。数値はすべて GPU ラボの実測。

## 0. 結論 (3 行)

1. **採らない。** decode の GEMV で最後まで v3/v2 を直接引いていた **shared expert の
   gate/up/down** を v5 にすると、`shared` 行 0.86 → 0.61 ms/token・`sh_wait`
   0.56 → 0.04、step 14.47 → 14.10 ms/token = **-2.5%** (t310b、2 ラウンド 3 サンプル、
   床 0.2%)。**target の 4% には届かない。** 14k 後は -1.6% (c8000_turn2)、
   prefill は不変 (c2000_cold 2.93-2.95 s / c8000_cold 7.39-7.44 s)。
2. 効いた機序は明確: shared expert の 3 本 (Q8_0 1.74 MB x 3) の **GPU 時間が
   join の待ちそのもの**だった (`sh_wait` が 0.56 → 0.04)。task 307 の
   `DENNO_GGUF_GEMV_V5` (dense) と同じ「v3 と v5 で 1.9 倍割れている」構図で、
   **shared expert だけが v5 を引いていなかった** (full 層と GDN は v5 優先)。
3. 残りは host 側に 4% を賄える山が無い (1 節)。次はカーネル。特に **mix 2 GB/token
   = 840 GB/s / head 590-960 GB/s / experts は既に約 1 TB/s** で、mix と head は
   bandwidth の 5 割前後しか出ていない。

## 1. 前提の測り直し (task の Do 1): 現在の int-auto の per-stage

`PROFILE=1` の台帳 (`[q4ledger]`、ms/token、warm a2 = 28 token の 2 本目 128 tok)。
t310b の base (2 サンプル) と t310a の base/base2。値は 4 サンプルの幅。

| 区間 | base (実測) | 中身 |
| --- | --- | --- |
| step | 14.38-14.69 | 1 トークン全体 |
| mix | 3.59-3.63 | gpu 2.4 / launch 0.48 (339 発) / cpu 0.36 |
| experts | 3.44-3.50 | gpu 2.5 (約 1 TB/s) / launch 0.25 (4 発/層) / host ~0.7 |
| ssm | 3.37-3.39 | GDN の完了待ち (同期 D2H 3.1) |
| full | 1.46-1.50 | gpu 1.01 (12 層) |
| shared | 0.86-0.89 | **`sh_begin` 0.42-0.46 + `sh_wait` 0.56-0.58** |
| head | 0.73 | gpu 0.70 (413-675 MB) |
| loop | sync 1.94-2.03 | expert 段の join (47 同期、43 us/層) |

- 合計 ≈ GPU 9.9 + host/待ち 4.2。**host 側で消せるのは起動 1.3 ms
  (964 発 x 1.35 us) + `sh_begin` 0.42 + 小さな台帳だけ**で、4% (0.58 ms) を
  1 個で賄える山が無い (P299 の結論と同じ)。task 307 が router (-0.9) / v5 (-0.16) /
  norm host (-0.22) を取った後の残り。
- **台帳の `loop` 行は当てにならない (計器のバグ)**: `moe_ffn_quant` が
  `enter(LOOP)` の内側で `enter(EXPERTS)` を開き、その後 `drop(_l_loop)` するので
  `CUR` が EXPERTS から LOOP へ巻き戻り、**expert 段の起動と同期が `loop` に付く**
  (`loop` の wall は 0.01 のまま)。`experts` の `cpu` 列が同期待ちを含むのはこのため。
  数値の合計は正しいが、行の帰属は読むときに注意が要る (今回はそこから
  「experts の同期 = 43 us/層 = GPU 2.5 ms とほぼ同じ」と確認した)。

## 2. 変えたこと (1 本のゲート)

### `DENNO_DEC_SH_V5` — shared expert の GEMV も v5 を先に引く (既定 off)

- `CudaCtx::plan_shared_gemv` (v5 → v3/v2 の順、形が合わなければ従来) を足し、
  `shared_expert_begin` (4 発を side stream に積む経路) と `shared_gemv_begin`
  (gate/up だけの旧経路) の 2 箇所から使う。
- v5 と v3 は**引数の並びが同一** (w, x, out, n_in, n_out, rb, rows_per_block, lanes)
  なので `launch_moe_gemv_on` は計画をそのまま積める。新しい PTX は要らない
  (`kernels_moe_v5.ptx` は既にコミット済み・既定 on の full/GDN が使用中)。
- `1` = gate/up/down の 3 本とも v5、`2` = gate/up だけ v5 (down は k=640 で
  `v5_lanes_div(10, 32) = 2` = block 16 スレッドに落ちる形のため)。既定 = 0 (従来)。
- 数値: v5 は 4 アキュムレータなので丸め順が変わる (決定性は保つ)。**実測は
  md5 が全 COND・全ラウンドで SAME** (2 節の 3 と 4 を見よ)。

## 3. lab

### 3.1 t310a (SHA d7103e021、5 COND、load_before 1.72)

`SCENARIO=depth8k PROFILE=1`。先頭 COND (toss) に診断 env
(`DENNO_Q4_TLINE=1 DENNO_GDN_DEC_KPROF=1 DENNO_MIX_KPROF=1`) を載せたが、**失敗**:
`[q4tline]` は `gpu_busy=0.00 launches=0` (イベントが unavailable) で、
さらに PLE が 2.9 → 199.8 ms に膨らんで toss だけ 19.0 ms/tok になった。
`DENNO_Q4_TLINE` は decode の計器としては使えない (窓が埋まる/CUPTI 無し)。

| cond | 中身 | decode 中央値 | a2 (s) | c2000_turn2 | c8000_turn2 | step (a2) |
| --- | --- | --- | --- | --- | --- | --- |
| toss | base + 診断 env | 19.0 | 1.98 / 1.97 | 20.2 | 19.5 | — |
| base | baseline | 17.8 | 1.86 / 1.86 | 17.7 | 17.7 | 14.41 |
| shv5 | `SH_V5=1` (3 本) | 16.9 | 1.81 / 1.82 | 15.1 | 15.5 | 14.07 |
| shv5gu | `SH_V5=2` (gate+up) | 17.0 | 1.81 / 1.82 | 15.2 | 15.5 | 14.00 |
| base2 | baseline | 17.3 | 1.86 / 1.86 | 15.3 | 15.9 | 14.38 |

- a2 (同一プロンプト・warm) は base/base2 が **1.86 / 1.86 (床 0.0%)** に対して
  shv5/shv5gu が 1.81-1.82 = **-2.3%**。step でも 14.38-14.41 → 14.00-14.07 = -2.4%。
- 集計の「decode 中央値」は文脈の違う要求を混ぜるので、base 17.8 / base2 17.3
  (2.8%) と床が広い。採否は a2 と step で読んだ。
- 3 本とも v5 (=1) より gate+up だけ v5 (=2) がわずかに良い (14.07 対 14.00)。
  down は k=640 で lane が 2 に落ちるためで、予想どおり。
- この回の VERDICT は先頭の toss (診断 env 付き) が基準なので読めない。

```
VERDICT base vs toss OK decode -6.4% prefill +0.8% md5 SAME errors 0
VERDICT shv5 vs toss OK decode -11.1% prefill -0.8% md5 SAME errors 0
VERDICT shv5gu vs toss OK decode -10.5% prefill -0.8% md5 SAME errors 0
VERDICT base2 vs toss OK decode -8.8% prefill -0.8% md5 SAME errors 0
VERDICT_DEPTH shv5gu c8000_turn2 vs toss decode -20.5% prefill -11.5%
VERDICT_DEPTH base2 c8000_turn2 vs toss decode -18.5% prefill -11.5%
VERDICT baseline toss errors 0 text ok
```

### 3.2 t310b (SHA d7103e021、5 COND、load_before 1.69) — 採否を読む回

COND = toss / base / sh2 / base2 / sh2b (base と候補を交互に 2 サンプルずつ)。
この回は host が静かで prefill も揃った (c2000_cold 2.93-2.95 s、
c8000_cold 7.39-7.44 s、全 COND 同値)。

| cond | decode 中央値 | c2000_turn2 | c8000_turn2 | step (a2, 2 ラウンド) | shared | sh_wait |
| --- | --- | --- | --- | --- | --- | --- |
| toss | 17.3 | 15.3 | 17.1 | — | — | — |
| base | 17.1 | 15.3 | 15.7 | 14.42 / 14.47 | 0.86 | 0.56 |
| sh2 | 17.0 | 15.0 | 15.5 | **14.07 / 14.11** | **0.61** | **0.04** |
| base2 | 17.2 | 15.4 | 15.8 | 14.69 / 14.45 | 0.89 | 0.58 |
| sh2b | 17.1 | 15.1 | 15.5 | **14.08 / 14.10** | 0.61 | 0.04 |

```
VERDICT base vs toss OK decode -1.4% prefill -0.8% md5 SAME errors 0
VERDICT sh2 vs toss OK decode -2.0% prefill +0.0% md5 SAME errors 0
VERDICT base2 vs toss OK decode -0.7% prefill -0.8% md5 SAME errors 0
VERDICT sh2b vs toss OK decode -1.1% prefill -0.8% md5 SAME errors 0
VERDICT_DEPTH base c2000_cold vs toss decode -1.2% prefill -0.2%
VERDICT_DEPTH base c2000_turn2 vs toss decode +0.3% prefill +0.0%
VERDICT_DEPTH base c8000_cold vs toss decode -7.7% prefill -18.6%
VERDICT_DEPTH base c8000_turn2 vs toss decode -8.2% prefill +0.0%
VERDICT_DEPTH sh2 c2000_cold vs toss decode -1.9% prefill +0.0%
VERDICT_DEPTH sh2 c2000_turn2 vs toss decode -1.5% prefill +0.0%
VERDICT_DEPTH sh2 c8000_cold vs toss decode -9.5% prefill -18.6%
VERDICT_DEPTH sh2 c8000_turn2 vs toss decode -9.6% prefill +0.0%
VERDICT_DEPTH base2 c2000_cold vs toss decode -1.2% prefill +0.3%
VERDICT_DEPTH base2 c2000_turn2 vs toss decode +0.5% prefill +0.0%
VERDICT_DEPTH base2 c8000_cold vs toss decode -7.7% prefill -18.6%
VERDICT_DEPTH base2 c8000_turn2 vs toss decode -7.8% prefill +0.0%
VERDICT_DEPTH sh2b c2000_cold vs toss decode -1.5% prefill -0.2%
VERDICT_DEPTH sh2b c2000_turn2 vs toss decode -1.0% prefill +0.0%
VERDICT_DEPTH sh2b c8000_cold vs toss decode -9.5% prefill -18.1%
VERDICT_DEPTH sh2b c8000_turn2 vs toss decode -9.1% prefill +0.0%
VERDICT baseline toss errors 0 text ok
```

- VERDICT の基準は先頭 COND (toss) で、toss だけが先頭バイアスで遅い
  (c8000_cold 9.08 s、c8000_turn2 17.1)。**採否は toss ではなく
  base/base2 と sh2/sh2b の対で読む** (下の表)。

- **短い decode: step 14.42/14.47 (base) + 14.69/14.45 (base2) に対して
  sh2/sh2b が 14.07/14.11/14.08/14.10 = -2.5%** (床: base と base2 の最良が
  14.42 対 14.45 = 0.2%)。
- 14k 後 (c8000_turn2): base 15.7 / base2 15.8 に対して 15.5 / 15.5 = **-1.6%**。
  2k 後: 15.3 / 15.4 → 15.0 / 15.1 = -2.0%。
- prefill: 全 COND 同値 (上記)。errors 0。
- md5 は t310b の全 COND・全ラウンドで SAME
  (`a:65ededc3 b:f18c75fc c2000_cold:d8706fac c2000_turn2:94f77974
  c8000_cold:e2292eb6 c8000_turn2:3cf204ea`)。
- テキスト (sh2): `TXT A **B-tree** (specifically, the **B+ tree) is the
  fundamental data structure used by most modern relational databases (such as
  MySQL, PostgreSQL, Oracle, and SQL Server).` /
  `TXTB Rust の「所有権（Ownership）とライフタイム（Lifetimes）は、Rust の最も
  重要な概念であり、同時に初心者が最もつまずきやすい部分です。` /
  `TXTD ご提示いただいた文書は、LLM推論エンジン（おそらく \`denno-engine）の
  **Prefill（事前計算）フェーズにおけるパフォーマンス最適化の記録です。` — base と同一。

### 3.3 baseline の md5 が t307c / t309c の参照からずれている (この task の変更ではない)

t310a / t310b の **base** の md5 は c2000_cold `d8706fac` / c2000_turn2
`94f77974` / c8000_turn2 `3cf204ea` で、t307c / t309c / tchk0217a の参照
(`d4b6eefb` / `8ff9345f` / `cc2c0e8f`) と違う (a / b / c8000_cold は一致)。
int-auto が `0179e968a` → `d6f914e97` (br307 の merge) へ進んだ時点で入った差で、
**この task の変更とは無関係** (5 COND x 2 ラウンドの 10 サンプルが全部同じ値 =
決定的)。307 (decode の mix chain) と 308 (prefill の router / full の devres) の
merge で長い文脈の丸めが動いた可能性がある。**監督は 307 の merge 後の
長文脈 md5 を一度確認したほうがよい。**

## 4. 採否 (task の規則: target 達成・他が 2% 超悪化なし・テキスト健全)

- 短い decode **-2.5%**、14k 後 -1.6%、prefill 不変、errors 0、テキスト健全。
  **target の 4% に届かないので ADOPT: no。** ゲートは既定 off のまま
  (`DENNO_DEC_SH_V5=1` / `=2` で入る)。
- このゲート自体は筋の通った -2.5% なので、**監督が部分採用したければ
  `dec_sh_v5_mode` の既定を 2 にする 1 行**で入る (数値は上のとおり)。

## 5. 数値の扱い

- 丸め: v5 は 4 アキュムレータで加算するので v3 とは順序が違う (決定性は保つ)。
  実測の md5 は全 COND で SAME、テキストも同一。よって「丸めは変わったが
  出力トークンは変わらなかった」。
- 効果の読み方: ラボの集計「decode 中央値」は文脈の違う要求 (a1 cold / b 48 tok /
  14k cold / turn2) を混ぜるので、**同じプロンプトの warm a2 と台帳の step で読む**。
  t310b の床は 0.2% (base 14.42 対 base2 14.45) で、-2.5% は床の 12 倍。
- 測っていない数字は書いていない。

## 6. テスト

最終 commit の内容 (int-auto `d6f914e97` の上、merge 不要) で:

- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`
  → 1207 passed / 0 failed / 102 ignored。新規の真理値表
  `dec_sh_v5_gate_truth_table` を含む。
- `cargo test --release -p denno-engine --features cuda --test kernel_invariants`
  → 95 passed / 0 failed。
- 既知の赤 (`qwen4exp_decode spec_batch_stops_at_eos`、`qwen4exp_params` の
  validate 2 件) は踏んでいない。
- 注意: `model::pf_host::tests::nested_stage_guard_counts_once` が 1 回だけ落ちた
  (単独実行と再実行では緑。この task の変更とは無関係の順序依存)。

## 7. GPU で確認してほしいこと

- `DENNO_DEC_SH_V5=1/2` を本番の長い文脈で回して md5 とテキストが変わらないこと
  (lab の depth8k では SAME)。
- br307 の merge 後の長文脈 md5 (`c2000_cold` / `c2000_turn2` / `c8000_turn2`) が
  参照からずれている件 (3.3)。回帰なら 307 x 308 の merge を疑う。
- 次に効くのはカーネル: mix (2 GB/token を 840 GB/s) と head (590-960 GB/s)。
  experts は既に約 1 TB/s なので、ここを触るなら mix/head の tile と vector 幅。
