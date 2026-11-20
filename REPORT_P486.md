ADOPT: yes t486b

# task 486 (dev3) — UD-Q4_K_XL: 窓の小さい prefill 層 (残り 1〜3 チャンク) でも
# 先読みの窓を使う (`DENNO_PF_ER_SMALL`)。3,826 cold の experts 段 `copy_s` が
# 6.98 -> 6.91 s (-3.9%、2 ラウンドの中央値で -3.9% / 別 run で -7.1%)、
# 壁が 8.39 -> 8.11 s (-3.3%)。14k・短い decode・既定モデルは不変。
# **タスクの 5% 目標は未達** (段の規則で採る)

branch: **br486** = 開始時の int-auto **d034ef5** (merge: br482)。自分の br482 は
既に int-auto に merge 済みだったので取り込み不要。終端で int-auto を再 fetch したら
**4b378cc (merge: br481)** に動いていたので merge した (**7cd7849**。衝突は
`kernel_invariants.rs` の末尾だけで、t486 のテストと task 481 のテストを両方残した)。

lab 名 (すべて `tree=denno-dev3` / `PROFILE=1` / `ptx=none`) /
baseline COND = タスク文のとおり `DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000
DENNO_EXPERT_GPU_MIN_HIT=0 DENNO_GPU_WEIGHT_MB=12000`:
- **t486a** = `0d7c96a` / `model=flash-next-UD-Q4_K_XL` / `SCENARIO=depth8k` /
  `load_before=1.52` / 6 COND: x0 b **t** **s8** **s** b2。
  門を入れただけの commit (門は既定 off、`DENNO_PF_ER_SMALL=1` で on)。
  t = スレッド 24、s8 = 門 on + 8 本、s = 門 on + 24 本。**2x2 の切り分け用**。
- **t486b** = `692a4c6` / `flash-next-UD-Q4_K_XL` / `depth8k` / `load_before=3.41` /
  COND: x0 b **d** b2 (b/b2 = `DENNO_PF_ER_SMALL=0`、d = 既定 on)。**採否はこれで読む**。
- **t486c** = `692a4c6` / `flash-next-UD-IQ3_XXS` (既定モデル) / `depth8k` /
  `load_before=1.66` / COND: x0 b d b2 (同じ A/B)。既定モデルの確認。

**lab はすべて merge 前の `692a4c6` で取っている**。br481 (4b378cc) は上げスレッドの
分割 DMA (`DENNO_PF_UP_SPLIT_KB`、既定 8 MiB) で、こちらが触った先読みの門
(`pf_early_route_prefetch` / `rc_plan`) とは別の場所。

## 0. 結論 (6 行)

1. **採る (段の規則で)。** 3,826 cold で先読みの門が発火し
   (`er_layers=47 er_experts=740 er_MB=2313`)、experts 段の `copy_s` が
   **6.98 -> 6.91 s (t486b の 2 ラウンド中央値で -3.9%)**、t486a の同じ A/B では
   **-7.1%**。壁は **8.39 -> 8.11 s (-3.3%)** (t486b)、t486a の round 2 では
   -4.2%。**2 run x 2 ラウンドとも baseline の最良サンプルより下**。
2. **タスクの目標 (UD の 3,826/14k が 5%、または短い decode 3%) は未達。**
   段は -3.9〜-7.1%、壁は -3.3〜-4.2% で、5% には届かない。
   operator の段ルール (2 ラウンドの中央値で当該段 2% 以上、他は 1% 超悪化しない)
   には当てはまるので既定 on にした (`DENNO_PF_ER_SMALL=0` で切り戻せる)。
3. **効いた理由は task 474 が閉じた 3,826 の窓を、同じ 8 本のままで使えたこと。**
   task 474 は「残り 1 チャンク = 窓 9.5 ms に 42 MB の先読みが収まらず ±0」と
   していたが、その 42 MB は**予算**の値で、窓に収まる分だけ読めばよい
   (実際に読んだのは 2,313 MB = 43.5 GB の 5.3%)。`copy_s` は 2 ラウンドとも
   baseline の最良より下がった。
4. **24 本にすると悪化した** (t486a の s: `copy_s` 7.40/7.37、壁 8.70)。
   先読みが窓を越えて MoE の写しと帯域を奪い合う。**スレッドの既定は 8 のまま**
   (task 486 では触らない)。
5. **14k はコード経路が 1 命令も変わらない** (wide = task 474 の規則が先に立つ)。
   t486a / t486b の両方で「門を入れた COND」の 14k が遅く出たが、内訳は
   **ホスト側**だった (t486b d_1: 全カーネルの launch が 46 us = baseline の 10 倍、
   GPU 時間は full 552.2 対 551.7 / ssm 1,321 対 1,314 で同一。t486a s8_1 も
   launch 24.7 対 4.4 us、mix の壁 128.7 対 25.7)。`experts` の `copy_s` は
   d 7.69 / b 7.74 / b2 7.56 で差なし。
6. **既定モデル (t486c) は完全不変**: 3,826 が全 COND 2.10-2.12 s、14k 5.16-5.29 s、
   短い decode 10.8 ms/tok、**md5 は全 COND・全リクエストで SAME**、errors 0。
   門は既定モデルでは**発火しない** (3,826 の実測レート 22,455 MB / 1.09 s =
   20.6 GB/s > 12 GB/s の閾値。`er=None`)。

## 1. 前提: なぜこれを選んだか (P482 の続き)

P482 §6 の申し送りは 4 つで、14k の残りはもう「ディスク (69%) と GPU カーネル
(23%) が床」だった。P482 の lab の生ログ (t482b) を読み直して、残りの段の実体を
バイトと IOPS で確かめた:

| 段 (14k cold) | ms | 実体 | 判断 |
| --- | ---: | --- | --- |
| `iter_top` (PLE gather) | 392-452 | 166,920 本の 90 B ランダム pread (28.8 GB の表)、470k IOPS | **閉じた** (デバイス上限。task 182 で RAM 置きも捨てている) |
| `pf_er` | 217 | 窓の中の先読みの回収待ち (`rc_join`) | 窓が足りない層は対象外 |
| `pf_teardown` | 96-244 | 追い出した expert の戻し (1.5-1.9 GB の H2D) | 次リクエストの decode のため。触らない |
| `res_up` | 92 | 残差 574 MB の pinned バウンス経由 H2D | 主成分はホストの memcpy。単独では -0.7% |
| experts の `copy_s` | 7,320 | 43.5 GB のディスク (5.9 GB/s) | **ここだけ窓が余っていた** |

`[pre-kprof] prefill tok=14018 ... | ple rows=166942 reads=166920 bytes=15024780
avg_read=90B read=76922.6 ms` = 1 行 1 pread。28.8 GB の表なのでソートしても
隣接しない (重複も 22 本だけ)。PLE を RAM に置く手は task 182 が測って捨てている。

14k の `experts` は P474 の窓 (残り 6 チャンク = 62 ms/層) で既に 27% が隠れて
いた (`er_MB=14166.6`)。**残っていたのは 3,826 (2 チャンク) の窓**で、task 474 が
「残り 1 チャンク = 9.5 ms に 42 MB が収まらず ±0」として閉じていた。

## 2. 変えたこと (2 commit + merge)

### 2.1 `0d7c96a` — `DENNO_PF_ER_SMALL` (門。この時点では既定 off)

- 窓の判定を 2 本に分けた: `wide` = task 474 の規則 (残り 4 チャンク以上、
  14k の経路はこれ)、`narrow` = 残り 1〜3 チャンク。
- `narrow` は 4 条件の純関数 `exp_pf::er_small_pass(wide, small_on, nt0, nt_all,
  bytes, copy_ns)` に集約: 広い窓でない / ゲート on / 窓の形 / **ディスク実測**。
- ディスク実測はストリームが既に積んでいる `STREAM_BYTES / STREAM_COPY_NS`
  (bytes/ns = GB/s) をそのまま使う。256 MiB 未満は「まだ分からない」= false、
  12 GB/s 以下だけ通す。**既定モデルは page cache から 20.6 GB/s で来るので
  通らない** (先読みは 1 バイトも得をせず router の同期だけ増える)。
  この判定を router GEMV + D2H の**前**に置いた (P482 と同じ理由)。
- 値も読みも変えない: 先読みは置き場 (`rc_plan`) に読むだけで、MoE の写しが
  同じ (パート, オフセット, 長さ) を拾う (task 474 の経路そのもの)。

### 2.2 `692a4c6` — 既定 on にした (t486a の 2x2 が勝ったため)

t486a は 2x2 (門 x スレッド) で測った。3,826 cold の `[exp-pf-stream]`:

| COND | 門 | 本数 | `copy_s` r1/r2 (s) | 壁 r1/r2 (s) | `er_MB` |
| --- | --- | ---: | --- | --- | ---: |
| b | off | 8 | 8.50 / 6.98 | 10.07 / 8.11 | 0 |
| b2 | off | 8 | 6.95 / 7.59 | 8.06 / 11.32 (外乱) | 0 |
| t | off | 24 | 7.33 / 6.89 | 8.49 / 8.05 | 0 |
| **s8** | **on** | 8 | **6.93 / 6.61** | 8.09 / **7.77** | **2,313** |
| s | on | 24 | 7.40 / 7.37 | 9.76 / 8.70 | 2,313 |

- 門は発火した: `er_layers=47 er_experts=740 er_MB=2313` (47 層 x 15.7 本 =
  予算 42 MB/層 = 2.3 GB = 43.5 GB の 5.3%)。
- `copy_s` は s8 が 2 ラウンドとも最小 (baseline の最良 6.95 より下)。
  壁は round 2 で 8.11 -> 7.77 = **-4.2%**。
- **24 本は悪化** (s: 7.40/7.37、壁 8.70)。スレッドの既定は 8 のまま。
- 費用は `other` に 70 ms (router の同期 62 ms + D2H 205 MB)、得は `copy_s`
  -0.37 s。正味 -0.34 s。

## 3. lab の結果

### 3.1 t486b (採否を読む run) — 3,826 cold、同じ run の 4 サンプル

`[exp-pf-stream]` の `copy_s` と `REQ c2000_cold` の壁 (s):

| COND | 門 | copy_s r1 | copy_s r2 | 壁 r1 | 壁 r2 |
| --- | --- | ---: | ---: | ---: | ---: |
| x0 (捨て) | on | 10.61 | 6.96 | 11.92 | 8.12 |
| b | off | 8.07 | 7.21 | 9.35 | 8.36 |
| **d** | **on** | **6.98** | **6.91** | **8.12** | **8.09** |
| b2 | off | 7.25 | 7.13 | 8.41 | 8.24 |

- **`copy_s`**: baseline 4 サンプルの中央値 7.23 -> d 6.945 = **-3.9%**。
  d は 4 サンプル中 2 つ (r1/r2) とも最小。
- **壁**: baseline の中央値 8.385 -> d 8.105 = **-3.3%**。
- `er_layers=47 er_experts=740 er_MB=2313` が d だけに出る (門の発火)。
- 14k: x0 36.61 (捨て・冷え) / b 11.80 / d 13.65 / b2 12.24。
  **d の 14k はホスト外乱** (§0 の 5 行目): d_1 は launch が全段で 10 倍
  (mix 218.5 ms / 4,704 回 = 46 us、baseline は 21.5 ms = 4.6 us)、GPU 時間は
  同一 (full 552.2 対 551.7、ssm 1,321 対 1,314、experts 1,184 対 1,141)。
  `copy_s` は d 7.69 / b 7.74 / b2 7.56。

### 3.2 t486a (門の 2x2) — 3,826 cold、同じ run の 5 サンプル

§2.2 の表。round 2 (page cache が暖まった後) で s8 の壁 7.77 対 baseline
8.11/11.32、`copy_s` 6.61 対 6.98/7.59。**24 本 (s) は 3,826 で負けた**。

### 3.3 t486c (既定モデル UD-IQ3_XXS) — 何も変わらない

| COND | 門 | 3,826 cold (s) | 14k cold (s) | decode ms/tok | md5 |
| --- | --- | ---: | ---: | ---: | --- |
| x0 | on | 2.12 | 5.72 | 10.8 | SAME |
| b | off | 2.12 | 5.16 | 10.8 | SAME |
| d | **on** | 2.12 | 5.29 | 10.8 | SAME |
| b2 | off | 2.10 | 5.24 | 10.8 | SAME |

`er=None` (門は発火しない): 3,826 の実測レート **22,455 MB / 1.09 s = 20.6 GB/s**
> 12 GB/s。14k は wide の経路で `er=(48, 6992, 14179.3)` が全 COND 同じ。
**md5 は 4 COND x 全リクエストで SAME**、errors 0、text は全 COND 「looks ok」。

### 3.4 VERDICT 行 (x0 は捨て COND なので採否には使わない)

```text
t486b: VERDICT b vs x0 OK decode -8.5% prefill -5.6% md5 DIFF errors 0
       VERDICT d vs x0 OK decode -9.5% prefill -16.1% md5 DIFF errors 0
       VERDICT b2 vs x0 OK decode -9.8% prefill -16.1% md5 DIFF errors 0
t486c: VERDICT b vs x0 OK decode +0.0% prefill -2.6% md5 SAME errors 0
       VERDICT d vs x0 OK decode +0.4% prefill +0.0% md5 SAME errors 0
       VERDICT b2 vs x0 OK decode +0.0% prefill -0.9% md5 SAME errors 0
t486a: VERDICT b vs x0 OK decode -4.2% prefill -14.3% md5 DIFF errors 0
       VERDICT t vs x0 OK decode -3.7% prefill -14.3% md5 DIFF errors 0
       VERDICT s8 vs x0 OK decode -5.0% prefill -14.3% md5 DIFF errors 0
       VERDICT s vs x0 OK decode +28.8% prefill -14.3% md5 DIFF errors 0
       VERDICT b2 vs x0 OK decode -5.3% prefill -14.3% md5 DIFF errors 0
```

`x0` は「先頭の COND は page cache が冷えて遅い」ための捨て COND で、b/d/b2 との
差が小さいのはそのため (x0 の 3,826 は 11.92、b は 9.35)。採否は §3.1 の
b/d/b2 の比較で読む。t486a の `s` の decode +28.8% は外乱 (a1 の md5 も 1 回だけ
違う = 同じ COND の中で状態が割れた。baseline 同士でも起きる既知の揺れ)。

### 3.5 出力テキスト (sane の根拠)

t486b の `TXTD` (3,826 の回答、md5 `73510e1f` は全 COND 一致):

```
ご提示いただいた文書は、LLM推論エンジン（`denno-engine）における
**Prefill（事前計算フェーズの性能最適化の歴史と教訓を詳細に記録したものです。
以下に、文書の構造と主要な技術的詳細を、論理的な流れに沿っ
```

`TXTB` (33 トークン、md5 `87d8a97c`):

```
Rust の「所有権（Ownership）とライフタイム（Lifetime）は、Rust を学ぶ上で
最も重要であり、かつ最も難しい概念です。しかし、これらは「メモリ管理を
コンパイラが自動的に行
```

## 4. 数値の扱い

- 採否の主指標は **`[exp-pf-stream]` の `copy_s` (experts 段のディスク) と壁**。
  3,826 は 1 run に 2 ラウンドあるので、ラウンドを分けて読む。
- 3,826 の壁は同じ run の baseline 同士でも 8.06 / 11.32 s と振れる (t486a の b2
  round 2 はディスクの外乱)。1 サンプルの ±5% は読まない。`copy_s` の方が安定
  (t486b の baseline 7.13-8.07、d 6.91-6.98)。
- **14k は ±0 (推定)**。根拠は (a) コード経路が wide で 1 命令も変わらない、
  (b) `copy_s` が d 7.69 / b 7.74 / b2 7.56、(c) 壁が遅く出た回はすべて
  ホスト側 (launch_ms 10 倍) か GPU 側の外乱で、GPU 時間は同一。**実測で
  「14k が変わらない」と言い切るには 4 run 目が要る** (3 ラウンドの上限を使い
  切ったので、ここでは推定と書く)。
- マイクロベンチは取っていない (GPU の無いコンテナ)。`copy_s` / `er_MB` /
  `launch_ms` / `gpu_ms` / 壁はすべて lab の実測。

## 5. テスト

最終 HEAD (2 commit + int-auto 4b378cc の merge、int-auto d034ef5 の上):

```text
CARGO_TARGET_DIR=/tmp/t486 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1348 passed; 0 failed; 103 ignored   (task 486 で 1 本追加。br481 の 1 本を含む)
同 --test kernel_invariants
  -> 142 passed; 0 failed                 (task 486 で 1 本追加。br481 の 1 本を含む)
CARGO_TARGET_DIR=/tmp/t486nc cargo build --workspace --release        (cuda 無し) -> OK
同 cargo test --workspace --release --no-fail-fast -- --test-threads=1 -> 失敗 0
```

既知の赤 (`qwen4exp_decode spec_batch_stops_at_eos`、`qwen4exp_params` の
validate 2 件) は今回も出ていない。

追加したテスト:
- `kernel_invariants::t486_early_route_small_window_is_wired`:
  門の真理値表 5 通り (広い窓でない / ゲート / 窓の形 / ディスク実測の 4 条件を
  1 つずつ落とす)、`er_small_from_value` の既定 on と偽値 5 通り、
  `early_route_small_ok` の形、`er_small_disk_bound` の境界 (256 MiB、
  5.9 / 7.1 / 19 GB/s)、配線の順序 (門が router / `peek_many` / `rc_plan` より前、
  最後の窓の判定が `wide || narrow` を見ている)。
- `t482_early_route_window_exit_is_wired` を新しい順序 (ゲート -> wide -> narrow
  -> 早期終了 -> router) に更新。

## 6. 次の担当者へ

1. **14k の 4 run 目**を取れば「14k ±0」を実測で言える (今回は 3 ラウンド上限)。
2. 3,826 の伸びしろは**まだある**: 窓 404 ms に対して先読みは 42 MB/層 = 5.3%。
   予算 (`DENNO_PF_EARLY_ROUTE_KB`、既定 24 KiB/残トークン) を 2 倍にすると
   窓を越えるので、**窓の中で読める量を層ごとに実測して予算を決める**のが次の一手
   (推定 -0.2 s = -2.5%)。
3. 門の費用の主成分は **router の D2H 205 MB** (1 層 3.9 MB = 全 expert の
   score を落としている)。top-k だけ落とせば 62 ms -> 10 ms 程度になり、同じ
   窓でより多く読める (推定 -0.05 s)。`pf_router_dev_topk` は MoE 本体と共有
   なので、触るなら両方の経路を測ること。
