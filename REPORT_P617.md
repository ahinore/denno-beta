ADOPT: yes t617a t617b t617c

# task 617 (dev2) — 既定モデル (flash-next-UD-IQ3_XXS) の prefill:
# **hc inject が F32 重みのため、4 行に対して 16 行タイルの GEMM で回っていた**
# (nsys t613a: `dno_gemm_f32_mt` 960 呼び出し 315 us = 302 ms)。重みだけ BF16 に
# 丸めて、BF16 の旧モデルが task 198 から使っている k 分割カーネル
# (`dno_mix_inj2_bf16`、1 ブロック 1 トークン) へ載せた (`DENNO_PF_INJ_F32_BF16`)。
# **14k の gpu_busy 4,145 -> 3,968 ms (-4.3%、4 サンプルで -176/-183)、
# mix 段の GPU 656.8 -> 478.7 ms (-27.1%)。** 壁は 14k 5.01 -> 4.85 s (-3.1%)、
# 3,826 2.12 -> 2.07 s (-2.4%)、decode 不変 (9.8-9.9 ms/tok)。
# カーネルも PTX も 1 バイトも変えていない (既存の既定 on 経路へ F32 重みを載せた)。

branch **`br617`** = 開始時の `int-auto` **`7bc7f90f1`** (merge: br609)。
br613 は REPORT のみ (コード変更 0 行) なので merge していない。
終端で int-auto を **`6a8597af0`** -> **`d5a97b487`** へ進めて 2 回 merge した (§7)。

数値はすべて GPU ラボの実測 (`SCENARIO=depth8k`、`PROFILE=1`、
`MODEL=flash-next-UD-IQ3_XXS`、baseline COND = タスク指定の env
`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0
DENNO_GPU_WEIGHT_MB=12000`、全 COND に `DENNO_PF_MIX_KPROF=1`、先頭に捨て COND)。
lab 名と SHA: **t617a = `67eff9e95`** (`x0 b g b2`、load_before 1.98)、
**t617b = `abda534bb`** (`x0 b g g2 b2`、load_before 1.48)、
**t617c = `3dfcf5b85`** (最終ツリー、既定 on と `=0` の対、load_before 1.91)。

## 0. 結論 (7 行)

1. **採る (`ADOPT: yes`)。** inject の段 (mix の GPU) が **-178.1 ms (-27.1%)**、
   2 ラウンドとも再現。14k の **gpu_busy は 7 サンプルで -176..-183 ms (-4.3%)**。
2. **壁は 14k -3.1% / 3,826 -2.1%** (3 run の 8-14 サンプルの中央値)。
   タスクの目標 (2.5%) を 14k 側で満たし、3,826 も悪化していない。
   **ただし t617a 単独では 14k -1.4%** だった (§8 のばらつき)。
   最終ツリー (既定 on) と切り戻しの対は t617c で取り、14k -2.8% / 3,826 -1.7%、
   `=0` で md5 が旧値に戻ることを確認した (§3.3)。
3. **選んだ理由**: P613 が nsys で「唯一 1 本で 2.5% を取れる可能性がある」と
   書いた 302 ms の `dno_gemm_f32_mt` の正体が、**inject の形 (n=4) と
   カーネルのタイル (16 行) の不一致**で、しかも **BF16 用の速いカーネルは
   既に既定 on で載っている** (task 198)。.cu を触らずに直せる唯一の場所。
   P572/P576/P581/P609 の「次の手」のどれでもない (全部 .cu か構造変更)。
4. **数値は丸めだけ変わる** (inject の重みを BF16・仮数 8 bit に丸める)。
   md5 は `c2000_cold` / `c8000_cold` が **SAME**、`turn2` の 2 つが DIFF。
   本文は同じ内容 (§6)。
5. ゲートは `DENNO_PF_INJ_F32_BF16` (**既定 on**、`=0` でビット一致に戻る)。
   発火は `[pf-mix-kprof]` の **`inj2_calls=672`** (14k) / `192` (3,826) で数える。
6. 壁が -3% で止まる理由は §4 の実測: 空いた GPU 時間が **expert 上げの待ち**
   (`up_wait`) に移る。壁 = GPU + 上げの床の和。
7. decode は 9.8-9.9 ms/tok で不変、14k 後 decode も 12.0-12.1 (§3.3)、errors 0。

## 1. fresh な段プロファイル (Do 1)

### 1.1 14k、t617a の b (baseline、`[q4ledger-pf] tok=14018`)

wall 4,919.8 / **gpu_busy 4,140.6** / exposed_host 1,405.0 (sum 5,545.6 overlap -625.8)

| 段 | wall ms | busy ms | exposed ms |
| --- | ---: | ---: | ---: |
| ssm | 2,098.2 | **1,319.8** | 778.4 |
| experts | 1,754.2 | **1,424.4** | 329.7 |
| full | 730.3 | **553.4** | 176.8 |
| mix | 31.0 | **656.8** | 0.0 |
| other | 306.1 | 186.0 | 120.1 |

`[pf-host] up_wait=1,310.2`、`h2d=3,108.0MB`、`[exp-pf-stream] up_MB=28,015.6`。
t609c (P613 が引いた run) は 5,147.5 / busy 4,142.7 なので、**この木の既定は
P613 の時点から動いていない** (busy 4,140.6 = 一致)。

`sub other` の中身 (この run): `iter_ple`、`res_up`、`pf_teardown` は P609 の
run と同じ桁。新しい段は無い。

### 1.2 既に誰かが触った段 (task 500 以降、このモデル)

REPORT_P502..P613 (既定モデル分) の「触った段」を並べる。**今回の inject は
ここに無い**。

| 段 | 触った REPORT | 結果 |
| --- | --- | --- |
| experts 上げ (staging) | P530 P534 P538 P546 P550 P554 P558 P566 P570 | 読みの形・上げスレッド・pool 分割は採用、読みを大きくするのは負け |
| experts 先読み | P609 (`DENNO_PF_ER_PAGE_HIT`) | page cache 速度で先読みを止める (採用、-4.2%) |
| experts の中 (GEMM) | なし (P613 が nsys の per-kernel まで) | **ncu 未計測** |
| ssm scan | P397 P402 P407 (5 案) | 全滅 (wy6/6p +45/+51%、wy7 -0.8%、wy7q +10%) |
| ssm 投影 | P407 (`PF_TC_GEMM2`) | ssm 分は `GDN_TC1` として既定 on |
| full (KV ミラー) | P505 P553 P561 P572 | NORESIZE / DEFER / RESERVE / MIRROR_FILL を採用 |
| mix (hc の down/up) | P585-P601 | Q8_0 の TC 経路 |
| **mix (hc の inject)** | **なし** | **本タスク (-27%)** |
| other (PLE / 残差 / teardown) | P349 P565 P581 P609 | PLE pipe、RES_PIN、RES_MT、er page hit |

## 2. 変えたこと (1 本、env ゲート)

`crates/denno-engine/src/cuda.rs` のみ (+ テスト 2 本)。

1. `pf_weight` の **F32 腕**に分岐を足した。`want_bf16 && pf_inj_f32_bf16_on()
   && mix_inj2_on() && pf_inj_bf16_shape(k, n)` (k>0・1<=n<=8) のとき、重みを
   **BF16 (RNE)** に丸めた写しを新しい常駐表 `PF_INJ_BF_W` へ上げて
   `PfWKind::Bf16` を返す。`pf_mix` の GEMM 選択は既存の順序で
   `mix_inj2_ok(kk, nn, nt, kind == Bf16)` を先に見るので、そのまま
   `dno_mix_inj2_bf16` (k を 256 スレッドへ分割・1 ブロック 1 トークン・
   固定木縮約) に載る。**カーネル / PTX / build_ptx.sh は 1 バイトも変えていない。**
2. `f32_bytes_to_bf16_bytes` (RNE、NaN は inf に化けさせない) を純関数で追加。
3. 発火の計数 `PF_INJ2_CALLS` を足し、`[pf-mix-kprof]` の行に `inj2_calls=` を出す
   (tag == "inject" のときだけ数える)。

既定モデルの hc inject は **F32・k = hc*n_embd = 10,240・n = hc = 4**。
従来は F32 のまま `dno_gemm_f32_mt` (1 ブロック 16 行 x 32 トークン) に渡り、
**16 行のうち 12 行が空回り**していた (1 呼び出し 315 us、nsys t613a)。
BF16 だった旧モデルは task 198 でこの形を k 分割カーネルへ移しており、
**その経路が F32 のテンソルには開いていなかった**のが穴。

## 3. lab の結果

### 3.1 段 (`mix` の GPU と 14k の gpu_busy)

| run / COND | 14k wall ms | gpu_busy ms | mix busy ms | up_wait ms | h2d MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| t617a b | 4,919.8 | 4,140.6 | **656.8** | 1,310.2 | 3,108.0 |
| t617a g | 4,853.0 | **3,965.3** | **478.7** | 1,397.3 | 2,883.0 |
| t617a b2 | 4,922.7 | 4,146.3 | 656.8 | 1,310.9 | 3,108.0 |
| t617b b | 5,120.8 | 4,144.3 | - | 1,305.1 | 3,108.0 |
| t617b g | 4,825.7 | **3,967.9** | - | 1,405.6 | 2,883.0 |
| t617b g2 | 4,851.0 | **3,972.0** | - | 1,350.6 | 2,883.0 |
| t617b b2 | 5,093.1 | 4,155.3 | - | 1,300.1 | 3,108.0 |

- **gpu_busy: baseline 4,140.6 / 4,146.3 / 4,144.3 / 4,155.3 (中央 4,145.3)
  -> g 3,965.3 / 3,967.9 / 3,972.0 (中央 3,967.9) = -177.4 ms (-4.28%)。**
- mix 段の busy は t617a の台帳で **656.8 -> 478.7 = -178.1 ms (-27.1%)**
  (gpu_busy の差と 1 ms 以内で一致 = 削ったのは全部 inject)。
- `[pf-mix-kprof]` の `inj2_calls=` が g だけ **672** (14k、48 層 x 2 mix x 7 チャンク)
  / **192** (3,826)。b 側は 0 (行が出ない)。672 x 265 us = 178 ms で上の差と一致。
- 他の段は t617a の台帳で `full` 553.4 -> 553.3 / `ssm` 1,319.8 -> 1,321.3 /
  `experts` 1,424.4 -> 1,426.0 / `other` 186.0 -> 186.0 (**すべて ±0.1%**)。
- `h2d` も 225 MB 減った (3,108.0 -> 2,883.0 MB、`other` の 267 本 x 0.84 MB)。
  **中身は未確定** (`pf-pin` は b/g で同値 552+4,644 本。F32 の inject 重みは
  1 本 160 KB なので候補だが、本数が合わない)。採否には使っていない。

### 3.2 壁 (prefill 秒、生値)

| 用途 | baseline (b, b2, b_2, b2_2) | g, g2 | 中央値の差 |
| --- | --- | --- | ---: |
| 14k cold (t617a) | 4.930 / 4.930 | 4.860 | -1.4% |
| 14k cold (t617b) | 5.130 / 5.100 | 4.830 / 4.860 | -4.8% |
| 14k cold (t617c、off 側) | 5.250 / 4.934 | 4.951 (既定 on) | -2.8% |
| **14k cold (6 off 側 + 4 on)** | 中央 **5.017** | 中央 **4.860** | **-3.1%** |
| 3,826 cold (t617a) | 2.160 / 2.110 / 2.118 / 2.118 | 2.082 / 2.067 | |
| 3,826 cold (t617b) | 2.12 / 2.13 / 2.13 / 2.12 | 2.09 / 2.07 / 2.07 / 2.07 | |
| 3,826 cold (t617c) | 2.76 / 2.14 / 2.18 / 2.12 | 2.13 / 2.10 | |
| **3,826 cold (14 off 側 + 8 on)** | 中央 **2.12** | 中央 **2.076** | **-2.1%** |

### 3.3 t617c — 最終ツリー (ゲート既定 on) と切り戻しの対

`3dfcf5b85` (REPORT と merge を含む最終ツリー) をラボで回した。**既定が on** なので
`off` COND だけ `DENNO_PF_INJ_F32_BF16=0` で落とした (`x0 off on off2`、2 ラウンドずつ)。

| COND | 14k wall ms | 14k gpu_busy | 3,826 (r1 / r2) s | md5 (cold / turn2) |
| --- | ---: | ---: | --- | --- |
| off (`=0`) | 5,250.4 | 4,142.9 | 2.76 / 2.14 | 旧 (`672fec07` / `199f23b3`) |
| on (既定) | **4,951.2** | **3,965.4** | **2.13 / 2.10** | 新 (`672fec07` / `7c72f4ed`) |
| off2 (`=0`) | 4,933.5 | 4,145.5 | 2.18 / 2.12 | 旧 |

- **既定 on で `inj2_calls=672` が出る** (on COND のみ)。`off` 側は 0。
- gpu_busy **4,142.9 / 4,145.5 -> 3,965.4 (-177 / -180 ms)**。t617a/t617b と同じ値
  (**通算 7 サンプルで -176..-183 ms**)。
- 壁は 14k off の中央 5,092 -> **4,951 (-2.8%)**、3,826 off の中央 2.15 ->
  **2.115 (-1.7%)**。この run の先頭は非常に冷えており (x0 の 3,826 が 8.19 s)、
  off の round 1 はまだ温まり途中 (2.76) だった。
- **`=0` で md5 が旧値に完全に戻る** (切り戻しが効く証拠)。
- decode は 9.8-9.9 ms/tok、14k 後 decode も off 12.4 / on 12.5 / off2 12.1 で差なし。

### 3.4 decode (悪化していないこと)

- warm short decode (RESULT の中央値): t617a x0 9.9 / b 9.8 / g 9.8 / b2 9.9、
  t617b 全部 **9.9**。
- 14k 後 decode (ms/tok、1 サンプル/COND): t617a b 11.9 / g 12.4 / b2 12.0、
  t617b b 12.2 / g 12.0 / g2 12.1 / b2 12.1。
  **b の 4 サンプル 11.9-12.2、g の 3 サンプル 12.0-12.4** で、t617a の
  g=12.4 だけが外れ値 (同じ run の b は 11.9)。中央値では差なし。
- このタスクは prefill の `pf_mix` だけを触っており、decode の hc mix は
  別経路 (`dno_pf_inj` / `mix_inj_r4`) なので構造的に不変。

## 4. なぜ壁が -3% で止まるか (実測からの計算)

14k の壁 = `gpu_busy + exposed_host`。GPU を 177 ms 削ったのに、
**`up_wait` が 1,310 -> 1,398 ms (+88) へ伸びた** (t617b でも 1,305 -> 1,406)。
層ループの中で expert の上げと GPU 計算は直列なので、GPU が速くなると
次の層の上げ待ちが長くなる。P613 の式 (壁 = GPU 4,142 + 上げ 1.38 s) の
GPU 側を 4.3% 削った分は、上げ側の待ちに一部吸われる。
**上げの床 (28 GB x 3 往復 / 61 GB/s = 1.38 s) を動かさない限り、
ここから先は 1 本で 2.5% にならない。** t617a は baseline が速い run
(4.92 s) だったので壁の差が -1.4% に縮み、t617b は baseline が遅い run
(5.12 s) で -4.8% になった。**g の壁は 3 サンプルで 4.826-4.853 (±0.3%) と
安定しており、baseline の方が run 間で 4.92-5.12 と振れる** (page cache と load)。

注 (§2 の副産物): h2d が 225 MB 減っている (`other` の 267 本 x 0.84 MB)。
**中身は未確定** (`pf-pin` は b/g で同値。F32 の inject 重み 160 KB x 96 本を
常駐表から引くようになった分が候補だが、本数が合わない)。次の担当が追う価値は
あるが、壁への寄与は 10 ms 級 (推定) で、採否には使っていない。

## 5. 残っている手 (次の担当へ、推定は「推定」と書く)

1. **expert IQ tensor core GEMM** (14k の GPU 40%、1,425 ms): P613 §5-1 のまま。
   23 TMAC/s が bf16 の何 % かは ncu のカウンタでしか決まらない。
   **本タスクでは測っていない** (ラウンドを採用の確認に使った)。
2. **scan の直列スカラー列** (ssm busy の 54%、~770 ms): .cu (P576 §6-2, P581 §6-2)。
3. **残差をデバイスに残す** (P581 §6-1、推定 -133 ms = 2.6%): 構造変更。
4. **上げの床 1.38 s** (28 GB x 3 往復): 壁の 28%。往復を減らすには
   mmap の pinned 登録しかなく、層あたり 100 ms 超 (推定、P613 §3)。
5. **h2d が 225 MB 減った件の中身** (§4 の注。採否には使っていない)。
6. 本タスクで削ったのは GPU 側なので、**上げの床を削る変更とは独立に効く**
   (両方入れれば加算される見込み、推定)。

## 6. 本文 (健全性)

t617a g_1_c8000_cold (md5 `672fec07` = **b と SAME**):

> ご提示いただいた文書は、LLM推論エンジン（`denno-engine）における
> **Prefill（事前計算）フェーズの性能最適化の歴史を詳細に記録したものです。

t617a g_1_c8000_turn2 (md5 `7c72f4ed`、b は `199f23b3` = DIFF):

> ご提示いただいた文書は、LLM推論エンジン（`denno-engine）における
> **Prefill（事前計算）処理の性能最適化の歴史を詳細に記録したものです。
> 以下に、この文書の技術的詳細を、時系列と技術的洞察に基づいて深掘り...

t617b g_1_c2000_turn2 (md5 `ac9314a6`、b は `8b02d5c3`):

> ご提示いただいた文書は、LLM推論エンジン（`denno-engine）における
> **Prefill（事前計算）フェーズの性能最適化の歴史を詳細に記録したものです。
> 以下に、文書の構造と技術的な深掘りした解説を提供します。

`!!!!` / 空 / プロンプトの反復なし。RESULT の text 欄は全 COND `looks ok`、
errors 0。

**md5 が DIFF な理由 (意図した丸め)**: inject の重みを BF16 (仮数 8 bit) に
丸めたので、inject の値が相対 1e-3 程度動き、sigmoid を通って hc combine の
倍率が 1e-4 相対で変わる。prefill の最初のトークン (cold 2 文脈) は同じで、
続きの turn2 で語尾が変わる。演算の意味・決定性は変わらない
(縮約は固定木で、実行ごとに同一)。**`DENNO_PF_INJ_F32_BF16=0` で md5 は
完全に元へ戻る** (t617a/b の b COND が従来 md5)。

## 7. merge とテスト

- 終端で int-auto を **`6a8597af0` -> `d5a97b487` (merge: br592)** と 2 回進めて
  merge した (1 回目は REPORT の docs 移動 + origin-main-1123、2 回目は br592 の
  `kernels_gdn_scan3` / `gate_audit`)。cuda.rs と kernel_invariants.rs は衝突なし、
  `targets_all` の衝突も無い (br592 は gdn_scan3 を足したが `targets_all` の行は
  別)。**merge のたびにゲートを回し直した** (下の本数は最終値)。
- 既定 on のゲート台帳 (`gate_audit::DEFAULT_ON_GATES`) には行を足していない。
  このゲートが使う PTX は `kernels_mix_inj2.ptx` / `dno_mix_inj2_bf16` で、
  `DENNO_PF_MIX_INJ2` の行が同じ組を持っている (台帳の注記の
  「同じファイルなら行を分けない」に従う)。PTX が欠ければその行が WARN を出す。
- この REPORT はリポジトリ直下 (`REPORT_P617.md`)。docs への移動は次の担当に任せる。
- **最終ツリー (`3dfcf5b85`) をラボで回した (t617c)**: 既定 on で `inj2_calls=672` が
  出て、gpu_busy が off の 4,142.9 / 4,145.5 から 3,965.4 へ落ち、`=0` で md5 が
  旧値に戻る (§3.3)。**merge 後・既定 on のままで採否の数字が再現している。**
- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`
  -> **1427 passed / 0 failed / 103 ignored** (`CARGO_TARGET_DIR=/tmp/t617`、br592 merge 後)
- `cargo test --release -p denno-engine --features cuda --test kernel_invariants
  -- --test-threads=1` -> **155 passed / 0 failed** (新規 1 本。br592 の 1 本を含む)
- cuda feature 無し: `cargo build --workspace --release` rc=0。
  `cargo test --workspace --release -- --test-threads=1` -> **63 本の
  test binary 全部 ok、rc=0** (P613 で落ちていた `denno-server` の
  `admit_error_message::slot_exhaustion_is_transient_retry_succeeds` も緑)。
- 新規テスト (発火を数える): ゲート真理値表 / inject の形の判定 /
  `f32_bytes_to_bf16_bytes` の RNE・NaN・端数 / `PF_INJ2_CALLS` の計数 /
  kernel_invariants の配線 (`pf_inj_f32_bf16_is_wired`)。
  ラボでは `inj2_calls=672` で分岐の発火を数えた。

## 8. 数値の扱い / 前提が崩れた点

- §1/§3 は GPU ラボの実測。§4 の ms と % は実測からの計算。「推定」だけが見積り。
- **14k はラウンド 1 にしか出ない = 1 COND 1 サンプル/COND** で、COND の位置で
  最大 ±7% 動く (P613 と同じ)。だから 3 run 取って中央値で読んだ (§3.2)。
  **t617a 単独では 14k -1.4%**、t617b では -4.8%、t617c では -2.8%。
  run 間で動いたのは baseline (4.93 vs 5.13 vs 5.25/4.93) で、
  **on 側は 4.83-4.96 と安定** (GPU を削った分だけ上げの揺れに鈍い)。
- t617a の b_1 (3,826) は **load=14.83** の回で 2,160.8 ms と外れた
  (b_2 は 2,108.6)。中央値には両方入れてある。
- `[pf-mix-kprof]` の `inj` 区間 (イベント計測) は**プールが尽きて出なかった**
  (b も g も `inj` 行なし)。段の証拠は台帳の `mix` busy と `inj2_calls=` で取った。
- decode の判定は warm short decode (各 COND 2 ラウンドの中央値) を使った。
  14k 後 decode は 1 サンプル/COND で ±1 ms 動く (§3.3)。

## 9. GPU で確認してほしいこと

1. **14k のサンプルは 3 run で 10 本取った** (off 側 6 / on 側 4)。中央値で -3.1%。
   ばらつきは baseline 側だけで、on 側は 4.83-4.96 と安定している (§8)。
   これ以上を詰めるより、`gpu_busy` (-177 ms) と `inj2_calls=` で読むのが早い。
2. **本番形 (serve_flash) の長い prefill** で `inj2_calls=` が
   層数 x 2 x チャンクになること、`DENNO_PF_INJ_F32_BF16=0` で 0 に戻ること。
3. **UD-Q4_K_XL で退行しないこと**: あのモデルの inject も F32 (k=10,240 / n=4)
   なので同じ経路に載る。ディスク律速なので壁は動かない見込み (推定)。
   1 リクエストで `inj2_calls` と `up_wait` を見ればよい。
4. decode の `[mix-qchain]` が不変であること (このタスクは prefill の `pf_mix`
   だけを触っている)。

## 10. VERDICT 行 (lab の出力そのまま)

t617a (`67eff9e95`):

```text
VERDICT b vs x0 OK decode -0.4% prefill +0.9% md5 SAME errors 0
VERDICT g vs x0 OK decode -0.4% prefill +0.0% md5 DIFF errors 0
VERDICT b2 vs x0 OK decode +0.0% prefill -0.9% md5 SAME errors 0
VERDICT_DEPTH b c8000_cold vs x0 decode -6.1% prefill -45.5%
VERDICT_DEPTH g c8000_cold vs x0 decode -2.3% prefill -46.3%
VERDICT_DEPTH b2 c8000_cold vs x0 decode -5.3% prefill -45.5%
VERDICT_DEPTH b c2000_cold vs x0 decode -69.0% prefill -75.2%
VERDICT_DEPTH g c2000_cold vs x0 decode -69.2% prefill -75.9%
VERDICT baseline x0 errors 0 text ok
```

t617b (`abda534bb`):

```text
VERDICT b vs x0 OK decode +0.0% prefill +0.0% md5 SAME errors 0
VERDICT g vs x0 OK decode +0.4% prefill -1.8% md5 DIFF errors 0
VERDICT g2 vs x0 OK decode +0.4% prefill -0.9% md5 DIFF errors 0
VERDICT b2 vs x0 OK decode +0.0% prefill +0.0% md5 SAME errors 0
VERDICT_DEPTH b c8000_cold vs x0 decode +0.0% prefill +2.6%
VERDICT_DEPTH g c8000_cold vs x0 decode -1.6% prefill -3.4%
VERDICT_DEPTH b c2000_cold vs x0 decode +0.4% prefill +0.0%
VERDICT_DEPTH g c2000_cold vs x0 decode +0.0% prefill -2.1%
VERDICT baseline x0 errors 0 text ok
```

(`vs x0` は捨て COND との比較。x0 は 14k と 3,826 が cold なので差が大きく出るだけで、
採否の判定には使っていない。判定は §3.1/§3.2 の b/b2/off との対で読む。)

t617c (`3dfcf5b85`、最終ツリー。x0 と on が既定 on、off が `=0`):

```text
VERDICT off vs x0 OK decode -0.4% prefill +0.0% md5 DIFF errors 0
VERDICT on vs x0 OK decode +0.4% prefill -0.9% md5 SAME errors 0
VERDICT off2 vs x0 OK decode +0.0% prefill -0.9% md5 DIFF errors 0
VERDICT_DEPTH off c8000_cold vs x0 decode +0.0% prefill -15.3%
VERDICT_DEPTH on c8000_cold vs x0 decode +0.8% prefill -20.1%
VERDICT_DEPTH off2 c8000_cold vs x0 decode -2.3% prefill -20.5%
VERDICT_DEPTH off c2000_cold vs x0 decode -11.2% prefill -52.3%
VERDICT_DEPTH on c2000_cold vs x0 decode -12.3% prefill -58.9%
VERDICT_DEPTH off2 c2000_cold vs x0 decode -12.3% prefill -58.2%
VERDICT baseline x0 errors 0 text ok
```
