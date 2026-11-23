ADOPT: yes t576b t576c

# task 576 (dev1) — 既定モデルの decode: 5 番目に大きい `shared` 段は
# **GPU ではなくホストの投入で決まっていた** (0.58 ms のうち sh_wait 0.05、
# 1 層 4 発 x 48 層 = 192 本/token の起動)。gate + up + silu*mul を 1 発に
# 畳んで **shared 段 -21/-23%、step -1.8/-2.0%、decode 中央値 10.0 -> 9.9
# (-2.0%)**、md5 は全 COND 全 6 文脈 SAME、errors 0。

branch **`br576`** = 開始時の `int-auto` **`f9d9b0c3e`** (merge: br572)。
br572 は既に int-auto に merge 済みだったので取り込み不要。終端で int-auto を
fetch し直したが **`f9d9b0c3e` のまま** (追加 merge なし)。

数値はすべて GPU ラボの実測 (`SCENARIO=depth8k`、`PROFILE=1`、
`MODEL=flash-next-UD-IQ3_XXS`)。baseline COND = int-auto の既定 +
`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0
DENNO_GPU_WEIGHT_MB=12000`。先頭に捨て COND、末尾に baseline の繰り返し。

lab 名と SHA:
- **t576a** = `f9d9b0c3e` (**nsys の診断**。`SCENARIO=decode`、COND x0/b/k)。
  コード変更なし = 現在の int-auto そのもの。
- **t576b** = `b764ed19c` (ゲート既定 off、`PTX=sh_gu`)。COND x0 / b / **g** / b2。
- **t576c** = `de1c9aa4a` (既定 on + 収録 PTX)。COND x0 / **on** / **off** (`=0`) / on2。

## 0. 結論 (6 行)

1. **採る (`ADOPT: yes`)。** 既定 `DENNO_DEC_SH_GU` を on にした (切り戻し `=0`)。
2. **段で 2 ラウンド以上 2% 以上**を満たす: `shared` 0.58 -> 0.45 (-22%、t576b) /
   0.58-0.60 -> 0.46 (-21/-23%、t576c の off 対 on)。他段は 1% 超の悪化なし。
3. **decode 中央値は -2.0%** (t576b b 10.0 -> g 9.8、t576c off 10.1 -> on/on2 9.9)。
   step は 9.97/10.01 -> 9.85/9.81 (t576b)、10.06/10.06 -> 9.88/9.86 (t576c)。
4. **値はビット一致** (丸め順を 1 つも変えていない): md5 は t576b / t576c の
   全 COND・全 6 文脈で SAME。狙って変えた丸めは無い。
5. prefill は不変 (0.56-0.57 -> 0.55-0.56 s)、14k 後も悪化なし
   (c8000_turn2 on 11.1/11.1 vs off 11.2/11.2)、errors 0。
6. **次の一手**は expert の `sum`/`silu` と ssm の `scan` (どれも .cu の iq/gdn
   カーネル側、§6)。

## 1. 現在地の段プロファイルと、なぜ `shared` を選んだか

出処: t576b の `b` (warm short decode の 2 番目の `[q4ledger]` ブロック、
ms/token・0.1 ms 刻み)。t576a の nsys と t576c の `off` も同じ値。

| 段 | ms/tok | 直近に誰が触ったか (REPORT) |
| --- | ---: | --- |
| experts | **2.88** | P388 (ミス CPU を起動に重ねる)、P526/P534/P562 (seg ゲート)、P564/P568 (中身の内訳、残りは .cu) |
| ssm | **2.82** | P496 (dn8 + rows 掃引)、P504 (out を dn7)、P392c (入力をデバイスから) |
| mix | **1.90** | P418 (dn5 で -11%)、P431 (dn7)、P464 (NOWAIT) |
| full | **0.95** | **P572 (今回の直前。KV ミラーの埋め戻しで -23%)** |
| **shared** | **0.58** | **P319 (`DENNO_DEC_SH_V5=2` で -2.4%)。以後 250 タスク誰も触っていない** |
| head | 0.46 | P427/P504 (dn6/dn7 = 帯域天井 1.57 TB/s) |
| router | 0.16 | P364 (mix の鎖の末尾へ) |

**選んだ理由**: 大きい順に experts / ssm / mix / full だが、experts・ssm・mix は
残りが「.cu のカーネル中身」(P568・P496 §8・P418 の天井メモ)、full は P572 が
直したばかり。`shared` だけが **5 番目の大きさで、中身がまだ計器で割られていない**
唯一の段だった。しかも割ってみると **GPU ではなくホストで決まっていた** (§2)。
`DENNO_SH_BEGIN_CACHE` (P307) は測って負けて既定 off、`DENNO_DEC_SH_V5` (P319) が
最後の変更で、その後の知見 (P572 の「待ちではなく投入が壁」) を持ち込める段だった。

**やらなかったこと (測って負けた/天井と分かっている)**: ssm out への dn8
(P496 の d8a が base と同等で d8r8 が勝ち)、ssm qkv (1.33 TB/s = DRAM の 74%)、
mix の dn7/up4f (P418 の 297 GB/s 天井メモ)、full の wo (553 GB/s、dn6/dn7/dn8 は
P427/P500/P504 が掃引済み)、expert の予測・先読み (task 337)。

## 2. なぜ shared は「投入」で決まっているのか (t576a の nsys が決め手)

[`q4ledger`] の `shared` 行 (warm):

| COND | shared wall | launch/cnt | sync | cpu | sh_wait |
| --- | ---: | --- | ---: | ---: | ---: |
| b (t576b) | 0.58 | **0.24/192** | 0.00 | 0.20 | **0.05** |
| g (t576b) | **0.45** | **0.12/96** | 0.00 | 0.20 | 0.05 |

`sh_wait` が 0.05 しかない = ホストが GPU を待っていない。段の wall は
**4 発/層の起動と H2D/D2H を積むホスト仕事** (48 層 x 4 = 192 本/token) そのもの。

t576a の nsys (CUDA API 集計、b COND 全体):

| API | 回数 | 合計 | 平均 |
| --- | ---: | ---: | ---: |
| cuEventSynchronize | 134,834 | 1.94 s | 14.4 us |
| cuMemcpyDtoH_v2 (同期) | 16,944 | 1.18 s | 69.4 us |
| cuCtxSynchronize | 7,822 | 0.86 s | 110 us |
| **cuLaunchKernel** | **365,650** | **0.82 s** | **2.23 us** |
| cuEventRecord | 283,809 | 0.41 s | 1.44 us |
| cuMemcpyHtoDAsync | 115,975 | 0.40 s | 3.44 us |

起動は 1 本 2.2 us。192 本/token を 96 本にすれば -0.21 ms が期待でき、
実測 -0.13 ms (0.58 -> 0.45) と桁が合う。

**nsys で分かった副産物 (次のタスクの材料)**: `[dec-kprof-avg]` の per-op gpu= は
やはり 3-6 us/op 過大だった。実測のカーネル時間は
`dno_dec_silu_mul` 0.87 us (表は 3.3)、`dno_dec_exp_sum` 2.16 (表 4.2)、
`dno_mix_gemv_q8_0_dn7` 10.7 (mix down と ssm out の平均。表は 11.7/16.0)、
`dno_mix_upmix_q8_up4f` 4.94 (表 9.2)、`dno_gdn_scan_sig2` 10.7 (表 12.6)、
`dno_mix_gemv_q8_0_dn8` 30.7 (表 33.6)。**silu と sum を畳んで得られるのは
0.36 ms ではなく 0.1 ms 程度**なので、P572 §6 の「silu+sum で -3.5%」は
過大である (推定の取り下げ)。

## 3. 変えたこと

1. **新カーネル** `kernels/kernels_sh_gu.cu::dno_dec_sh_gateup_silu` (Q8_0)。
   1 スレッドが gate 行 r と up 行 r の内積を **同じ x** で取る (x の読みが半分)。
   積和の演算列は `kernels_moe_v5.cu::dno_gemv_q8_0_v5` と同一
   (アキュムレータ 4 本 / `j & 3` / ブロック 0 -> 1 / shfl_down 1,2,4...)、
   epilogue の silu は `kernels_dec_epi.cu::dno_dec_silu_mul` と同じ式
   `x / (1 + __expf(-x))`、`act = silu(g) * u`。**ビット一致**。
   grid = ceil(n/rows)、block = rows*lanes は v5 の計画 (実形 80 x 64) と同じ。
2. **ホスト** (`cuda.rs::shared_expert_enqueue`): gate / up / silu の 3 発を
   1 発に置換え。門番は「gate と up が Q8_0 / 同じ k, n / 同じ v5 計画
   (rows, lanes, rb, 関数) / `sh_gu_shape_ok` / PTX が引数 9 本」。
   満たさなければ**従来の 3 発**へ落ちる (fail closed)。
   1 層 4 発 -> 2 発 = 96 本/token の起動が消える。
3. ゲート `DENNO_DEC_SH_GU` (**既定 on**、明示的な偽値 `""`/`0`/`off`/`false`/`no`
   だけ off = P572 の `DENNO_FULL_DEC_MIRROR_FILL` と同じ流儀)。
4. 診断 `[sh-gu] fused=<回数> fallback=<回数> ptx=ok|none` (ゲート on のときだけ 1 行)。
   t576b の g は `fused=6144 fallback=0` = 48 層 x 128 トークン全部が融合経路。
5. `build_ptx.sh` に target `sh_gu` と `targets_all` の末尾 (他ブランチとの衝突検出)。
   `kernel_invariants` の 8 本が `targets_all` の末尾文字列を固定していたので
   `sh_gu)` へ更新し、収録 PTX の 9 引数も見るようにした。

## 4. 数値

### 4.1 段 (ms/token、`[q4ledger]`、同じ run の中の対)

| run | COND | shared | launch/本数 | sh_begin | step | layer (/tok) |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| t576b | b | 0.58 | 0.24/192 | 0.41 | 9.97 | — |
| t576b | b2 | 0.58 | 0.24/192 | 0.41 | 10.01 | — |
| t576b | **g** | **0.45** | **0.12/96** | **0.28** | **9.85** | — |
| t576c | **on** | **0.46** | **0.12/96** | 0.29 | **9.88** | **9.362** |
| t576c | **off** | 0.58 | 0.24/192 | 0.41 | 10.06 | 9.540 |
| t576c | **on2** | **0.46** | **0.12/96** | 0.29 | **9.86** | **9.343** |
| t576c | off2 | 0.60 | 0.25/192 | 0.42 | 10.06 | 9.536 |

他の段は t576b / t576c で動いていない: mix 1.89-1.90、router 0.16-0.17、
full 0.94-0.96、ssm 2.81-2.86、experts 2.84-2.89、head 0.46。

### 4.2 decode 中央値 (RESULT.md)

| run | 基準 | 候補 | 差 |
| --- | ---: | ---: | ---: |
| t576b | b 10.0 / b2 10.0 | g 9.8 | **-2.0%** |
| t576c | off 10.1 / off2 10.1 | on 9.9 / on2 9.9 | **-2.0%** |

`VERDICT g vs x0 OK decode -1.6% prefill +0.9% md5 SAME errors 0` (t576b)
`VERDICT on vs x0 OK decode +0.0%` / `off +1.6%` / `on2 -0.4%` (t576c、
x0 は既定 on の捨て COND なので on と一致する = 既定が効いている証拠)

14k / prefill (t576c、同じ run): c8000_cold 5.11-5.38 (位置ドリフトの内側)、
**c8000_turn2 on 11.1 / 11.1 vs off 11.2 / 11.2 (-0.9%)**、
prefill 0.55 / 0.56 / 0.55 / 0.55 s = 不変。errors 0。

### 4.3 text (t576c `on` の実出力、抜粋)

`# B-Trees: A Comprehensive Explanation` /
`## 1. What is a B-Tree?` / `A **B-tree (Balanced Tree) is a self-balancing
tree data structure that maintains sorted data and allows searches,
sequential access, insertions, and deletions in logarithmic time.` —
健全。`TXTB` / `TXTD` も P572 と同じ本文。
md5 は a:edaa84ad b:1938059f c2000_cold:9b28e018 c2000_turn2:8b02d5c3
c8000_cold:672fec07 c8000_turn2:199f23b3 で **t576b / t576c の全 COND SAME**。

## 5. テスト (このブランチ)

- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`
  → **1420 passed / 0 failed / 103 ignored** (新規 5 本)
- `--test kernel_invariants` → **153 passed / 0 failed**
- cuda 無し: `cargo build --workspace --release` rc=0、
  `cargo test --workspace --release -- --test-threads=1` は
  `denno-server` の `admit_error_message::send_error_logs_engine_cause` が
  1 回だけ落ち、**同じ木で再実行すると 7 passed** (既知の flaky。原因は
  このタスクの変更と無関係 = 受付エラーのタイミング試験)。
- 追加したテスト:
  - `sh_gu_gate_truth_table` / `sh_gu_shape_truth_table` (門番の真理値表、
    実形 k=2560 / n=640 / rows=8 / lanes=8 が通ること)
  - `shgu_word_and_byte_rows_agree_and_partition_covers_once`: `.cu` の索引算術を
    **生バイト読み**と **17 ワード読み**の 2 通りに写し、実形で全行・全レーンが
    **1 ビットも違わない**こと、レーン分割が 40 対を 1 回ずつ覆うことを固定
  - `shgu_silu_formula_matches_epilogue`: silu の式
  - `sh_gu_fusion_is_wired`: カーネル名・ゲート名・起動関数・`build_ptx.sh` の
    target / targets_all・**収録 PTX の引数 9 本** (nvcc の実出力を読む)
- 分岐が発火した証拠: lab の `[sh-gu] fused=6144 fallback=0`
  (= 1 トークン 48 層 x 128 トークン) と `[q4ledger] shared` の起動 192 -> 96 本。

## 6. 次にやるなら (推定、測っていない)

1. **expert の `sum` (0.87 us/層) と `silu` (0.85 us/層) を畳む**。nsys の実測では
   2 本で 1.7 us/層 = 0.08 ms/token (0.8%) しかない。P572 §6 の「-3.5%」は
   per-op の過大評価だったので**取り下げる**。畳むなら `dno_exp_gemv_*` の
   epilogue に silu を入れる形 (.cu)。**費用対効果は低い**。
2. **ssm の `scan`** (nsys 10.7 us/層 = 0.39 ms = 4%) は `threadIdx.x == 0` の
   直列スカラー列 (P577)。14k prefill でも ssm busy の 54%。ここが decode で
   残る最大の .cu 対象。ただし P496 §8 も「ba を scan に畳む」は負けている。
3. **同期の本数**そのもの。nsys で cuEventSynchronize が 134,834 回 (平均 14.4 us)、
   cuCtxSynchronize 7,822 回 (110 us)。`sync_floor` は 2-5 us なので、空振りの
   同期を 100 本消せば 0.2-0.5 ms になる。今回は起動 (cuLaunchKernel 0.82 s) を
   削ったが、待ちの側は手つかず。
4. `shared` の `down` も `launch` 側に 48 本残っている。down は入力を
   gate/up の出力 (`dact`) に持つので、`down` を**次の層の 1 発に畳む**手は
   無いが、silu を down のカーネル頭で計算すれば `dact` の 1 発を消せる
   (推定 -0.05 ms、0.5%)。今回は入れていない。

## 7. GPU で確認してほしいこと

- 本番形 (serve_flash、長い文脈) で `[sh-gu] fused=` の増加と `fallback=0`。
- `DENNO_DEC_SH_GU=0` で `[q4ledger] shared` の起動が 192 本/token に戻ること
  (切り戻しの確認)。
- 実モデル (swift15 IQ3_S) の shared も Q8_0 640/640/2560 なので同じ経路に乗る。
  ただし hc mix が BF16 のモデルでは shared の量子型が違う可能性がある
  (`[q4ledger] shared` の wall が 0.58 のままなら不活性 = 門番で落ちている)。
