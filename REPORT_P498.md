ADOPT: no t498c

# REPORT P498 — UD-Q4_K_XL: 現 int-auto の段プロファイルを取り直した。warm short decode
# 13.4 ms/token の 39% は experts 段で、その中身は **1 トークン 4.1 ms の層あたり固定費
# (85 us/層) + ミス 1 本あたり 57 us** にきれいに分かれる。層の固定費のうち名前が付いて
# いて削れそうだった最後の 1 つ (`[seg-prof] d2h=13.3-14.8 us/層` = 10 KB の pageable な
# 同期 D2H、0.64-0.71 ms/token = decode の 5%) を pinned バウンスに替えたが、
# **分岐は 100% 発火 (6048 本/リクエスト、fallback 0) して d2h も decode も動かなかった**。
# `DENNO_SEG_SUM_PIN` は既定 off のまま (採らない)。目標 (UD 3% / prefill 5%) は未達。

branch: **br498** = 開始時の int-auto **e0746f3** (= merge: br490)。`br494` は
int-auto に未 merge だったが中身は `REPORT_P494.md` だけ (コード変更 0) なので
取り込んでいない (コードは同一)。終端で int-auto を fetch し直したら
**890fc66 (merge: br496)** に動いていたので merge した (衝突なし。br496 は既定モデルの
GDN decode 用 `kernels_mix_dn8` で、こちらとは別経路)。

lab 名 (すべて `tree=denno-dev3` / `MODEL=flash-next-UD-Q4_K_XL` / `SCENARIO=depth8k` /
`PROFILE=1` / `ptx=none`) / baseline COND = タスク文のとおり `DENNO_EXPERT_IQ_GPU=1
DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0 DENNO_GPU_WEIGHT_MB=12000`:
- **t498a** = `6b1fe78` (コードは e0746f3 と同一) / `load_before=3.14` /
  COND: x0 (捨て) b **f0** (`DENNO_EXP_UP_FILL=0`) **f8** (`DENNO_EXP_UP_FILL_MULT=8`) b2
- **t498b** = `6b1fe78` / `load_before=1.79` /
  COND: x0 b **sp** (`DENNO_SEG_PROF=1 DENNO_EXP_KPROF=1`、診断) b2
- **t498c** = `968bc4d` / `load_before=20.29` (高い) /
  COND: x0 b **sp** (`DENNO_SEG_SUM_PIN=1`) **spq** (`DENNO_SEG_SUM_PIN=1
  DENNO_SEG_PROF=1`) b2

lab は 3 ラウンドで上限。**採るゲートは無く、既定値も変えていない**
(足したのは既定 off の `DENNO_SEG_SUM_PIN` と診断行だけ)。

## 0. 結論 (7 行)

1. **採らない (`ADOPT: no`)。** 目標 (UD 3,826/14k cold prefill -5%、または UD の
   warm short decode -3%) に届く変更は無い。今回作った `DENNO_SEG_SUM_PIN` は
   仮説 (「10 KB の pageable D2H は driver のステージングで遅い」) が**外れ**、
   既定 off のまま残す (切り戻し不要、既定の経路は 1 命令も変わらない)。
2. **warm short decode の段 (t498a の a2、128 token、hit 0.960)**: step **13.33** =
   experts **5.15** (39%) / ssm **2.99** (22%) / mix **2.24** (17%) / full **1.27** (10%) /
   shared 0.61 / head 0.61 / router 0.19。**最大の段は experts**。
3. **experts 段は 2 項に分かれる**: ミス本数を変えた 5 点 (hit 0.680-0.962) で
   `experts ≈ 4.1 ms/token + 57 us x ミス本数/token`。ミス 1 本 = 2.9 MB を 57 us で
   読む = **51 GB/s** (DRAM の実効上限) なので**ミスの読みは床**。残るのは
   **層あたり 85 us の固定費** (48 層 x 85 us = 4.1 ms)。
4. **その固定費の内訳 (`[seg-prof]`、t498b、定常)**: 1 層あたり lookup 1.5 +
   table 0.7 + h2d 2.9 + **launch 9.1** + sync 1-37 + **d2h 13.3-14.8** us。
   d2h は e*4 = 10 KB の同期 D2H で **0.7 GB/s**、48 層/token で 0.64-0.71 ms
   (decode の 5%)。ここだけが「名前が付いていてホスト側で削れる」最後の項目だった。
5. **pinned バウンスは効かなかった (t498c)**: 専用 64 KiB pinned へ DMA してから
   ホストへ写す経路を足し、**6048 本/リクエスト (= 48 層 x 126 token) で 100% 発火、
   fallback 0** を確認したが、`[seg-prof] d2h` は **12.0-17.0 us/層** (base 12.6-15.1)
   で変わらず、warm decode も **13.46 (sp) 対 13.33 (b) / 13.43 (b2) = +1.0%
   (分解能 0.1 ms の範囲)**。**13 us はドライバの pageable ステージングではなく、
   同期コピー自体 (ストリームの drain) の固定費**だった。task 356 が専用ストリームで
   6.9 us/copy まで下げても壁が動かなかったのと同じ理由で、ここは閉じた。
6. **`DENNO_EXP_UP_FILL` は UD では逆効果 (t498a)**: off で short decode
   **14.1 -> 15.9 (+13%)**、8x で 14.0 (−0.7% = 分解能)。ミス読みと mmap の取り合いを
   疑った仮説は**否定** (fill を止めるとミスが増える方が大きい)。既定は触らない。
7. **prefill は前回 (P494) の床のまま**: 14k cold の壁 22.08 s のうち
   experts **17,517 ms (79.3%、全部ホスト = ディスク)**、ssm 1,862 / full 891 /
   other 1,783。読んだ量 `up_MB=52,923.7` / `copy_s=16.85` = **3.1 GB/s**
   (この日のディスク。P452 の 5.8-6.2 GB/s の日もある)。3,826 も 43.5 GB を
   14.2-16.2 s = 2.7-3.1 GB/s。**読むバイトは最小 (P494 §1.2)・窓は満杯**で、
   ホスト側に 5% の余地は無い。

## 1. なぜこの段を選んだか (Do 1 / Do 2)

### 1.1 warm short decode の段 (t498a の a2 = 2 本目の 128 token 要求)

| 段 | b (round 1) | b2 (round 2) | 中身 |
| --- | ---: | ---: | --- |
| experts | **5.15** | 5.20 | `cpu=5.15` = ミス expert の CPU GEMV のみ (launch/sync/h2d/d2h 0) |
| ssm | 2.97 | 2.99 | GDN decode (GPU)。sync 0 (待ちは次の段に回る) |
| mix | 2.24 | 2.25 | Q8_0 の hc mix。qchain が 4656/4656 で発火 (GPU) |
| full | 1.27 | 1.28 | flash decoding 12 層。gpu 0.78 + sync 0.86/15 |
| shared | 0.63 | 0.62 | 共有 expert。`sh_begin=0.44` (P307 で cache 化は効かず棄却済み) |
| head | 0.61 | 0.61 | 語彙 GEMV。gpu 0.56 (516 MB を 905 GB/s = VRAM 律速) |
| router | 0.18 | 0.18 | |
| **step** | **13.33** | **13.43** | `[flash] decode=1.72s (13 ms/tok)` |

ミス本数: `[exp-iq] lookups=61440 hits=59082 hit_rate=0.962` -> 19.6 本/token。
`[dec-up-w] submitted=464 done=464 dma=...` (fill の上げ直し) も同時に走っている。

### 1.2 experts 段 = 固定費 + ミス本数 x 57 us

同じ run の中の 5 点 (すべて a1/a2 の 128 token、`[q4ledger] experts` と
`[exp-iq] hit_rate`):

| hit_rate | ミス本/token | experts ms/token |
| ---: | ---: | ---: |
| 0.962 | 19.6 | 5.15 |
| 0.933 | 32.9 | 5.57-5.64 |
| 0.837 | 80 | 8.30-8.60 |
| 0.771 | 112.5 | 12.73-12.85 |
| 0.680 | 157 | 12.76-12.91 |

傾き 57 us/本 (2.9 MB / 57 us = **51 GB/s** = このホストの DRAM 実効上限)、
切片 **4.1 ms/token = 85 us/層**。P408 (t408b) の層内訳 (sync 36-49 / miss 35-73 /
d2h 8.2 / launch 6.9 / h2d 1.9 / lookup 0.9 us) と整合する。

つまり **decode の 39% は「層あたり 85 us の固定費」**で、そのうち名前が付いていて
ホスト側 (Rust) から削れるのは d2h (13.8) と launch (9.1) だけ。launch は 5 起動の
融合 = `.cu` で、task 412 が**測って棄却済み** (`DENNO_SEG_GU_FUSE` で -0.6%)。
残ったのが d2h 13.8 us/層 = 0.66 ms/token (5%) で、これを今回試した。

### 1.3 選ばなかった段と、その理由

- **ssm 2.99 (22%)**: GDN の完了待ち + 状態の同期 D2H。`DENNO_PF_GDN_WB_ASYNC` は
  P397、decode の D2H 非同期化は共通ルールの「Already rejected」に入っている。
- **mix 2.24 (17%)**: qchain が 100% 発火済み (P282)。`DENNO_MIX_INJ_R4` (P484) も
  入っている。
- **full 1.27 / head 0.61**: どちらも GPU 律速 (head は VRAM 905 GB/s)。
  カーネルは `.cu` = このタスクでは不可。
- **prefill**: §0-7 のとおりディスクの床。P494 の結論を再現しただけ。

## 2. 変えたこと (すべて既定 off)

1. **`DENNO_SEG_SUM_PIN`** (新規、既定 off): seg 層一括経路の和 (e*4 = 10 KB) の D2H を、
   専用の 64 KiB pinned バウンス (`cuMemHostAlloc` は初回 1 回だけ) へ `cuMemcpyDtoH`
   してからホストへ写す。失敗 (pinned が取れない / CUDA 無し / 転送エラー) は false を
   返して従来の pageable `to_slice` へ丸ごと落ちる。値は同じバイト列。
   `DENNO_PF_PIN_STAGE` (P358/P369 が prefill で棄却した共有バウンス) とは別経路・
   別ゲート (1 MiB 床も窓も通さない)。
2. 診断行 2 本: 起動 `[flash] seg sum pin: on|off`、リクエスト終端
   `[seg-pin] calls=.. fallbacks=..` (ゲート on のときだけ)。分岐が発火した証拠を
   ログに残すため。
3. テスト 4 本 (`model::seg_prof::tests::t498_*` 2 本、`cuda::task498_tests` 2 本)。

**既定値・既定の経路は変えていない。** ゲート off では `seg_sum_pin_on()` が false で
`if` が短絡し、`outb.to_slice(&mut out[..e])` がそのまま走る (分岐 1 回だけ増える)。

## 3. lab の結果

### 3.1 t498a (現 int-auto の段プロファイル + fill の A/B)

| COND | short decode (ms/tok) | 3,826 cold (s) | 14k cold (s) | 8k 後 decode |
| --- | ---: | ---: | ---: | ---: |
| x0 (捨て) | 14.4 | 18.45 | 20.89 | 22.7 |
| b (base) | **14.1** | 16.23 | 22.08 | 18.3 |
| f0 (fill off) | **15.9 (+13%)** | 15.72 | 19.62 | 17.5 |
| f8 (fill 8x) | 14.0 | 25.27 (外れ) | 20.66 | 18.8 |
| b2 (base) | 14.1 | 16.20 | 22.92 | 24.1 |

- **fill off は short decode を 13% 悪化**させた (ミスが増える分が先に効く)。
  8x は分解能の中。`[exp-iq]` の hit_rate も b 0.960 -> f0 0.906 に落ちている。
- 3,826 / 14k の prefill は**この run のディスクが遅い** (3.0 GB/s) ので採否に使わない。

### 3.2 t498b (層の内訳、診断のみ)

`[seg-prof]` 定常 (UD、48 層、warm) — **us/層**:

| lookup | table | h2d | launch(5) | sync | d2h |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.5 | 0.6 | 2.9 | 9.1 | 1-37 | **13.3-14.8** |

GPU: gate ~300-430 / up ~350-460 / down ~290-340 GB/s、1 層 60-75 us。
(`miss=` と `alloc=` の列は GPU 待ちを含む包括区間なので使っていない。)

### 3.3 t498c (本命の A/B)

`load_before=20.29` の高い回。それでも baseline 同士 (b / b2) は揃っている。

| COND | short decode | warm a2 step | experts | `[seg-pin]` | `[seg-prof] d2h` |
| --- | ---: | ---: | ---: | --- | --- |
| x0 (捨て) | 15.0 | — | — | — | — |
| b (base) | **14.1** | **13.33** | 5.15 | — | — |
| sp (pin) | 14.0 | **13.46** | 5.27 | **calls=6048 fallbacks=0** | 12.0-17.0 |
| spq (pin+prof) | 14.4 | 13.5 | 5.3 | calls=... | 12.0-17.0 |
| b2 (base) | 14.0 | 13.43 | 5.20 | — | (off) 12.6-15.1 (t498b) |

- **分岐は 100% 発火** (`calls=6048` = 48 層 x 126 token、`fallbacks=0`)。
- **d2h は 1 us も縮まない** (12.0-17.0 対 base 12.6-15.1)。pinned へ DMA しても
  10 KB の同期コピーは 13 us かかる = コストは**ドライバのステージングではなく、
  同期コピーが null ストリームを drain する固定費**。
- decode は **13.46 対 13.33 / 13.43 = +1.0%** (0.1 ms 刻みの分解能と同じ)。
  8k 後 decode (b 18.3 / sp 18.3) と 14k cold prefill (b 20.44 / sp 21.64 /
  b2 21.89 = 揺れ) も差なし。

### 3.4 VERDICT 行

```text
t498a: VERDICT b vs x0 OK decode -2.4% prefill -14.4% md5 DIFF errors 0
       VERDICT f0 vs x0 OK decode +10.3% prefill -7.2% md5 DIFF errors 0
       VERDICT f8 vs x0 OK decode -2.7% prefill -13.7% md5 DIFF errors 0
       VERDICT b2 vs x0 OK decode -2.4% prefill -13.7% md5 DIFF errors 0
t498b: (診断のみ。sp は SEG_PROF+EXP_KPROF のため decode 23.3 = 参考値)
t498c: VERDICT b vs x0 OK decode -6.5% prefill -6.5% md5 DIFF errors 0
       VERDICT sp vs x0 OK decode -6.8% prefill -17.4% md5 DIFF errors 0
       VERDICT spq vs x0 OK decode -4.4% prefill -14.5% md5 DIFF errors 0
       VERDICT b2 vs x0 OK decode -7.0% prefill -16.7% md5 DIFF errors 0
```

### 3.5 既定モデルのラウンドは取っていない (ラウンド上限)

lab は 3 ラウンド (t498a/b/c) で上限に達した。今回の変更は**既定 off** で、off の
ときの経路は `if crate::model::seg_prof::seg_sum_pin_on()` が短絡して
`outb.to_slice(&mut out[..e])` がそのまま走る (分岐 1 回だけ増える)。
**既定モデルと UD のゲート off の計算経路は 1 命令も変わらない**ので、
「既定モデルが変わっていない」はラウンドではなくコードの一致で示す。加えて
t498a/b/c の長い 3 本の md5 は全 COND SAME (§4)。

## 4. md5 と出力テキスト

- 長い 3 本 (`c2000_cold` 73510e1f / `c2000_turn2` 9e32c460 / `c8000_cold` 47c833e2) は
  **3 run x 全 COND で SAME**。
- 48 token の `b` と `c8000_turn2` の md5 は**同じ env の COND 同士でも揺れる**
  (t498a: x0 `b:87d8a97c` 対 b2 `b:ffbf4402`。t498c: b `b:87d8a97c` 対 sp
  `b:1938059f` / b2 `b:87d8a97c`)。P490/P494 が記録した既存の揺れで、
  **このタスクの変更 (既定 off) とは無関係**。
- 出力 (t498a の `TXTD`、3,826 の回答、全 COND 同文):

```text
ご提示いただいた文書は、LLM推論エンジン（`denno-engine）における
**Prefill（事前計算フェーズの性能最適化の歴史と教訓を詳細に記録したものです。
以下に、文書の構造と主要な技術的詳細を、論理的な流れに沿っ
```

- 短い `TXTB` (全 COND 同趣旨): 「Rust の「所有権（Ownership）とライフタイム
  （Lifetime）は、Rust を学ぶ上で最も重要であり、かつ最も難しい概念です。…」

## 5. 数値の扱い

- すべて GPU ラボの実測 (`SCENARIO=depth8k`、`PROFILE=1`)。マイクロベンチは取って
  いない (このコンテナは GPU 無し)。
- lab の**ディスクは 3 GB/s の日**だった (t498a/b/c とも `copy_s/up_MB` で 3.0-3.1
  GB/s)。prefill の 5% はこの日には原理的に読めない (P452 の速い日で 5.8-6.2 GB/s)。
- `experts = 4.1 + 0.057 x miss` の 2 項モデルは**同じ run の 5 点**から出した推定で、
  「ミスの読み = 51 GB/s = 床」「固定費 = 85 us/層」の結論は P408 の層内訳と整合する。
- **推定**: 固定費の残り (launch 9.1 + sync 1-37 us/層) を削るには 5 起動の融合
  (`.cu`) が要る。task 412 が -0.6% で棄却しており、今回のタスクは `.cu` 不可。

## 6. テスト

最終 HEAD (int-auto 890fc66 を merge 済み) で:

```text
CARGO_TARGET_DIR=/tmp/t498 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
同 --test kernel_invariants
CARGO_TARGET_DIR=/tmp/t498nc cargo build --workspace --release            (cuda 無し)
同 cargo test --workspace --release --no-fail-fast -- --test-threads=1
```

結果:

```text
cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1353 passed; 0 failed; 103 ignored
cargo test --release -p denno-engine --features cuda --test kernel_invariants
  -> 144 passed; 0 failed
cargo build --workspace --release            (cuda 無し) -> OK
cargo test --workspace --release --no-fail-fast -- --test-threads=1 -> 失敗 0
```

新規テスト 4 本は緑: `model::seg_prof::tests::t498_seg_sum_pin_truth_table` /
`t498_seg_pin_counts_accumulate` / `cuda::task498_tests::seg_pin_copy_rejects_out_of_range` /
`seg_pin_copy_falls_back_without_cuda` (最後の 1 本は **GPU の無いこのコンテナでは
fallback 側**を通る = fail closed の証拠。pinned 経由の実発火は lab の
`[seg-pin] calls=6048 fallbacks=0` で確認した)。

## 7. 次の担当者へ

1. **UD の warm short decode は「experts 段の固定費 85 us/層」が本体** (4.1 of 13.4 ms)。
   名前の付いた内訳は launch 9.1 / d2h 13.8 / h2d 2.9 / lookup+table 2.2 us/層。
   d2h は今回 pinned でも動かなかったので、**残るのは launch の融合だけ**で `.cu` が要る。
2. **ミスの読み (57 us/本 = 51 GB/s) は床**。ミス本数を減らす道は VRAM 常駐で、
   P435/P452 が prefill を壊すと測って閉じている。
3. **prefill は P494 のとおりディスクの床**。判定は `[exp-pf-stream] copy_s` が
   6 GB/s 以上出た回だけで行う。3 GB/s の日は 5% 差が埋もれる。
4. `DENNO_SEG_SUM_PIN` は**既定 off のまま**残した (発火は 100% するが効かない)。
   もし将来「同期 D2H の drain を消す」ことを狙うなら、コピーを非同期にして
   待ちを後ろへ回すしかない (task 356 が専用ストリームで試して 0% = 棄却済み)。
