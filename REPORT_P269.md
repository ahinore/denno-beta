ADOPT: no t269c

# REPORT P269 — IQ3_S decode の内訳を測り直し、いちばん大きい「GPU が遊ぶ塊」を取る

(br269 = br266 の上。br266 は int-auto の直系の子で、既定の挙動は int-auto と同じ。
 merge は fast-forward で、このタスクの変更は全部その上に載っている。)

## 1. 前提と測り方 (Do の 1)

lab `t269a` (SHA 136249eef = br269 の開始点, scenario=depth8k, PROFILE=1,
model=swift15-abl-IQ3_S, COND = toss / base / evt / base2)。`evt` は base に
`DENNO_GDN_DEC_KPROF=1` を足しただけで、**CUDA イベントで層の中の
GPU 時間 / ホスト時間 / 起動数 / 同期数 / 隙間**が読める。台帳は `[q4ledger]`
(PROFILE=1) と `[q4prof]` と `[dec-exp-kprof]` / `[exp-iq-kprof]`。

基準 (base COND、短い decode): **19.4 ms/tok**、14k の次 18.1 / 18.0 ms/tok、
prefill 25.14 s @14,018・7.29 s @3,826。md5 は全 COND SAME、errors 0、
`toss`/`base`/`base2` は 0.2% 以内で揃っている (この回のノイズ床)。

### 1.1 1 トークンの台帳 (`[q4ledger]`, a1 = 128 トークン, hit_rate 0.99)

`step` = 16.93 ms/tok。区間は重なりを含むので、合計は step より少し大きい。

| 区間 | wall | うち GPU 待ち | launch | sync 回数 | h2d | d2h |
| --- | --- | --- | --- | --- | --- | --- |
| mix | 3.54 | 2.58 | 0.53 | 97 | 3.97 MB | 0.99 MB |
| ssm | 4.15 | 0 | 0.17 | 0 | 1.30 MB | 0.37 MB |
| experts | 4.32 | 0 | 0 | 0 | 0 | 0 |
| full | 1.62 | 1.15 | 0.10 | 18 | 0.29 MB | 0.34 MB |
| router | 1.25 | 0.66 | 0.07 | 48 | 0.49 MB | 0.10 MB |
| head | 0.73 | 0.70 | 0 | 1 | 0.01 MB | 0.99 MB |
| shared | 0.59 | 0 | 0.24 | 0 | 0.49 MB | 0.49 MB |
| 合計 | 16.2 | — | 1021 起動/tok | **211 同期/tok** | 14.2 MB | 3.8 MB |

14k の次 (c8000_turn2): step 17.94、mix 3.59 / ssm 4.19 / experts 4.56 /
full 2.24 / router 1.26 / head 0.73、同期 206/tok。

**GPU は 12.9 ms ぶん実際に回っている** (下の表の kern+copy の合計)。
16.9 ms のうち 76% で、残り 4 ms が「ホストが次を積めず GPU が遊ぶ」時間。

### 1.2 層の中のイベント表 (`[dec-kprof-avg]`, us/層)

短い decode / 3,800 の次 / 14,000 の次。

| 種 | wall | kern | copy | gap (GPU 遊び) | wait (ホスト待ち) | launch |
| --- | --- | --- | --- | --- | --- | --- |
| mix (1 回) | 55.1 / 55.7 / 51.2 | 33.2 / 33.1 / 33.1 | 3.5 / 2.9 / 2.7 | 7.8 / 9.7 / 7.2 | 31.5 / 32.0 / 28.2 | 9.9 / 10.0 / 9.7 |
| gdn (ssm 層) | 110.6 / 107.5 / 107.1 | 84.6 / 84.6 / 85.5 | 19.0 / 16.0 / 14.6 | 2.7 / 2.6 / 2.7 | 0 / 0 / 0 | 8.4 / 8.3 / 8.5 |
| full (full 層) | 226.3 / 142.0 / 203.6 | 139.1 / 115.4 / 178.7 | 17.4 / 0 / 0 | 51.2 / 13.6 / 12.8 | 137.7 / 104.9 / 164.7 | 15.0 / 15.8 / 16.5 |
| router (1 回) | 43.1 / 29.7 / 28.3 | 7.5 / 7.7 / 7.7 | 0 | 0 | 2.5 | 2.5 |

mix 1 回の中身 (短い decode、us/起動):

| 起動 | 形 | GPU | 備考 |
| --- | --- | --- | --- |
| `dno_matvec_bf16_r_silu` (down) | 320 block | 9.3 | 6.55 MB → 704 GB/s |
| `dno_mix_norm_inj` | **grid=1**, block=1024 | **10.3** | 正規化 + inject (1 SM) |
| `dno_mix_up_bf16` (up) | 1280 block | 7.8 | 6.55 MB → 840 GB/s |
| `dno_hc_mixed_f32` | **grid=10** | **5.8** | e=2560 の足し込み |
| D2H mixed | 41 KB | 3.5 | |

ssm 1 層の中身: `dno_gemv_q8_0_v5` 29.8 (31.5 MB → 1,057 GB/s)、`q4k_v5` 21.9、
`q5k_v5` 31.0、`dno_gdn_scan_sig2` 12.7、**D2H は host=86.5 us のブロッキング**
(GPU の鎖 95 us を待つ)。起動とコピーの内訳: H2D 8.1、D2H 10.9。

expert (48 層): `gate`(gate+up 融合) 40.9 / `up` 1.5 / `silu` 3.4 / `down` 16.3 /
`d2h` 5.8 / `sum` 4.2 = 1 層 68〜72 us → **3.3 ms/tok**。型別の実効帯域は

| 型 (g/u/d) | 本数 | us/本 | GB/s |
| --- | --- | --- | --- |
| IQ2_S | 18,683 | 7.5 | **261** |
| IQ3_XXS | 17,396 | 4.6 | 476 |
| IQ3_S | 11,205 | 7.6 | **305** |
| IQ4_XS | 1,247 | 4.5 | 598 |

ヒット率はリクエストで 0.865〜0.99。ミス expert の CPU 計算は 0.46〜1.68 ms/tok で、
GPU の後ろに一部しか隠れない (t266r4 の 0.915 では experts 区間が 4.32 → 5.93)。

### 1.3 どこが取れるか

- **GPU 側**: いちばん大きいのは ssm 3.7 / mix 3.5 / experts 3.3 / full 1.9 ms。
  ssm の GEMV は 1,057 GB/s でもう天井、mix の 2 本も 704/840 GB/s、full は
  v3 と v5 の混在 (49.2 と 26.2 us の同型比較)、experts は IQ3_S/IQ2_S が
  305/261 GB/s で **同じカーネルの IQ3_XXS/ IQ4_XS (476/598) の 6 割**。
  mix の小さい 2 本 (norm_inj 10.3 + hc_mixed 5.8 = 97 呼び出しで 1.56 ms/tok) は
  起動・占有で決まっていて、t216 が融合版 (UPMIX) を作っている。
- **ホスト側 (GPU が遊ぶ)**: いちばん大きい 1 本は **ssm の beta/alpha 射影**。
  `A_GATE − A_GDN` = **0.785 ms/token** (36 層 × 21.8 us)。重みは BF16 で
  491 KB/層、ホスト側は rayon の行チャンクが `max(32)` のため 96 行で 3 スレッド
  しか使わず、実測 22 GB/s。ここは GPU が丸ごと遊んでいる。

## 2. 変えたこと (Do の 2)

**ssm の beta/alpha 射影を GDN submission の中で GPU で回す** (`DENNO_GDN_BA_GPU`)。
`forward_token_gpu` がホストで回していた F32/BF16 GEMV をやめ、`dno_matvec_bf16_r`
(または `dno_matvec_f32_r`) を xba の beta/alpha 節域へ 1 発で書き、scan は
従来どおりその節域を読む (カーネル・引数・発行順は変えない)。

- 数値: 縮約が固定の木になるので CPU の逐次/FMA と**丸め順が違う** (決定論は保つ)。
  ゲートで戻せる (`DENNO_GDN_BA_GPU=0` が既定)。
- 型は BF16 だった (lab t269b で判明)。BF16 -> f32 は正確なので重みの値は動かない。
  重みは BF16 のまま VRAM に常駐 (36 層で 18 MB)。
- **fail closed は呼び出し側で行う**: submit の中で失敗させると
  `gdn_gpu_scan` がプロセス全体で不使用になり、ssm が CPU 経路に落ちて 35% 遅くなる
  (t269b で実際に踏んだ)。型が合わなければ `gdn_ba_units()` が空を返し、従来の
  ホスト計算のままになる。

## 3. lab

### t269a (測定のみ、SHA 136249eef)

上の 1.1 / 1.2 の表。base 19.4 ms/tok、`evt` (イベント計測) も 19.4 で
計測の overhead は検出できない。

### t269b (SHA 4d40048f6)

toss / base / ba / upm / both / base2。

| COND | 短い decode | 14k の次 | 判定 |
| --- | --- | --- | --- |
| base | 19.4 | 18.1 | — |
| ba | **26.2** | 24.8 | +35.8% (遅い、md5 SAME) |
| upm | 20.4 | 19.1 | +5.7% (遅い、md5 SAME) |
| both | 27.2 | 25.5 | +40.6% |

`ba` はログに `gdn ba: 型/行数が想定外 (Bf16, n=96)` が出て submit が失敗し、
`gdn_gpu_scan` が不使用になって ssm が CPU 経路に落ちていた (だから md5 も SAME)。
修正は 3 章のとおり。`upm` (`DENNO_DEC_MIX_UPMIX=2`、t216c では -2%) はこの
ビルドでは +5.7% で、**採らない**。

### t269c (SHA 1ede42f40) — 最終

toss / base / ba / iqv1 / base2。`iqv1` = `DENNO_EXP_IQ_V2=0` (expert GEMV を
v1 に戻す。型別の帯域差が v2 のせいかを見る)。

| COND | 短い decode | 14k cold | 14k の次 | prefill 14k | 判定 (vs toss) |
| --- | --- | --- | --- | --- | --- |
| toss | 19.4 | 18.1 | 18.2 | 25.33 s | (基準) |
| base | 19.5 | 18.2 | 18.1 | 25.06 s | +0.4% |
| **ba** | **18.8** | **18.1** | **18.7** | 25.26 s | **-3.0%** |
| iqv1 | 20.0 | 19.1 | 19.1 | 25.12 s | +3.4% |
| base2 | 19.4 | 18.2 | 18.3 | 25.08 s | +0.2% |

`VERDICT ba vs toss OK decode -3.0% prefill +1.6% md5 DIFF errors 0`
`VERDICT_DEPTH ba c8000_cold vs toss decode +0.3%`
`VERDICT_DEPTH ba c8000_turn2 vs toss decode +2.6%`

- **ba は短い decode で -3.4% (base 19.5 → 18.8)**。a2 は 17 → 16 ms/tok。
  md5 は 6 本とも DIFF = 意図した丸め順の変更 (beta/alpha の縮約が木になる)。
  本文は健全 (`TXT ba`: B-tree / B+tree の説明、`TXTB ba`: Rust の所有権と
  ライフタイムの説明)。errors 0。
- 14k は横ばい (cold 18.1 は base と同じ、の次 18.7 は +2.6%)。
- **`iqv1` は +3.4% で v1 の方が遅い** = expert GEMV の型別帯域差は v2 のせいではない。
- 注意: ba は丸めが変わるので出力長が変わる (c2000_cold 120 → 84 トークン、
  c8000_cold 109 → 80)。深い点の ms/tok は**別のトークン数での中央値**なので、
  深い点の比較は ±2% の範囲でしか読めない。
- `t269b` の `upm` (`DENNO_DEC_MIX_UPMIX=2`) は +5.7% で、このビルドでは
  t216c の -2% を再現しない。採らない (既定 off のまま)。

## 3.1 採否

**ADOPT: no t269c。** 目標「短い decode と 14k の次が両方 8% 以上」に対して
短い -3.4%、14k の次 +2.2% (base2 比) で届いていない。-3.4% は本物だが
8% の 4 割強で、しかも深い点は横ばい〜微減。`DENNO_GDN_BA_GPU` は**既定 off**
のまま残す (コードとテストは残す。次のタスクが前提にしてよい)。

**残りの差 (8% に届かない理由)**: 内訳は 1.3 のとおりで、decode の 76% は
すでに GPU が実際に回っている。GPU 側の 4 本 (ssm 3.7 / mix 3.5 / experts 3.3 /
full 1.9 ms) はどれも「その形で観測された最良の帯域」の 1.2〜3 倍の範囲に居る。
唯一 3 倍以上離れているのが expert GEMV の IQ3_S (305 GB/s) と IQ2_S (261 GB/s)
で、同じカーネルの IQ3_XXS は 476 GB/s 出ている。ここを縮めるには 1 行 2 MB を
32 レーンで割る現在の形 (kernels_exp_iq) を書き換える必要があり、本タスクの
残り時間では着手していない (`iqv1` の +3.4% で v1 が遅いことは確認済み)。

## 4. 数値の扱い

- 丸め順が変わるのは `DENNO_GDN_BA_GPU=1` のときだけ (既定 off)。オフでは
  ホストの GEMV がそのまま走り、値は 1 ビットも変わらない。
- 決定性・行チャンク不変・par/seq 一致は触っていない。
- md5 の比較は lab の TXT/TXTB/TXTD を読んで行う (SAME でも本文を確認する)。

## 5. 見積り (推定) と実測

- 着手前の見積り: beta/alpha の GPU 化で -0.6 ms/token 前後 (-3%)。
  **実測 -0.7 ms/token (19.5 → 18.8) でほぼ見積りどおり**。
- 8% には届かない。残る差の主因は expert GEMV の IQ3_S / IQ2_S が
  305 / 261 GB/s (同じカーネルの IQ3_XXS は 476) で、ここは 1 本 2 MB の
  行を 32 レーンで割る現在の形を変える (新しいカーネル) 必要がある。
  見積りではここが 1.3 ms/token ぶんあり、900 GB/s まで出れば -8% 相当になる。

## 5.1 merge

`int-auto` (2a7dbf71e) を取り込んだ。衝突は 2 箇所で、どちらも意味のある統合:

- `expert_vram.rs` の `pub use gpu::{...}`: 両側の追加 (br266 の
  `fallback_breakdown` と int-auto の `acquire_host` / `keys` /
  `touch_many_count`) を**全部残した**。
- `qwen4exp.rs` の起動ログ 1 行: br266 側が「旧 4 型カーネル gemv/gemv2/gemm」を
  出していたが、その経路は task 261 の最終 commit で撤回済みなので int-auto 側
  (cap_mb + IQ ゲート + 上げ直し) を採った。`expert_kernel_status()` の呼び出しも
  落とした (関数は残っている)。

`build_ptx.sh` の targets_all は衝突しなかった (このタスクはカーネルを足していない)。

## 6. テスト

- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`
- `cargo test --release -p denno-engine --features cuda --test kernel_invariants`
- 追加: `gdn_ba_units_tile_beta_alpha_rows` (単位が beta_alpha の全行を隙間なく
  1 回ずつ覆う = task 140 の罠を固定) / `gdn_ba_gpu_truth_table` (ゲートの解釈)。

## 7. GPU で確認してほしいこと

1. `DENNO_GDN_BA_GPU=1` で `[q4attn]` の `gate` が `gdn` とほぼ同じ値になる
   (ホストの beta/alpha が消えた証拠)。
2. そのとき md5 が DIFF になる場合は、値の差が丸めだけか (logits の max abs 差)。
