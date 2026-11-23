ADOPT: no t510a t510b

# REPORT P510 — UD-Q4_K_XL: prefill の「選択 expert 全体 / VRAM 常駐から回せたぶん /
# 上げたぶん」をログで分離できるようにした。実測すると **常駐の再利用は既に厳密に
# 効いていて、読むバイト数 = 選択 − 常駐** だった。残るのは容量だけで、
# その容量は P435/P452（VRAM 予算）と P458（所有 RAM tier）で棄却済み。
# **タスク目標（5% / 3%）に届く変更は無い。**

branch: **br510** = 開始時の int-auto **890fc66** (= merge: br496)。
終端で int-auto を fetch し直したが **890fc66 のまま** なので merge は無い。

lab 名 (すべて `tree=denno-dev3` / `SCENARIO=depth8k` / `PROFILE=1` / `ptx=none`) /
baseline COND はタスク文のとおり `DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000
DENNO_EXPERT_GPU_MIN_HIT=0 DENNO_GPU_WEIGHT_MB=12000`:
- **t510a** = `f390db5` / `MODEL=flash-next-UD-Q4_K_XL` / `load_before=14.69` /
  COND: x0 b b2（3 本とも同じ env。x0 は捨て）
- **t510b** = `f390db5` / `MODEL=flash-next-UD-IQ3_XXS`（既定）/ `load_before=1.86` /
  COND: x0 b b2

lab は **2 ラウンドで止めた**（3 ラウンド目に残す候補が全部「測って棄却済み」だったため。§5）。

## 0. 結論 (7 行)

1. **採らない (`ADOPT: no`)。** タスクの狙い（routing 統計で常駐を選び、3,826/14k
   の cold prefill でディスクから読むバイトを減らす）は、**読む側の仕組みが既に
   その通りに動いていて、残りは容量だけ**だった。容量は P435/P452/P458 で棄却済み。
2. **変わったのは計器だけ**（`[exp-pf-stream]` に `res_exp` / `res_MB` / `req_exp` /
   `req_MB` を追加）。`up_MB` は「ステージングへ上げたぶん」しか数えないので、
   VRAM 常駐が実際に効いているかをログだけで読めなかった。数値経路は 1 命令も変えない。
3. **実測（UD、3,826 cold、b_1 / b_2）**: 選択 **65,651.2 MB**（20,952 本）、
   VRAM 常駐から回せた **22,124.3 / 22,072.6 MB（7,043 / 7,024 本）**、
   上げた（= ディスク/置き場から読んだ）**43,526.9 / 43,578.6 MB**、
   `pread` **43,739.9 / 43,803.8 MB**。**65,651.2 − 22,124.3 = 43,526.9**（一致）。
4. **実測（UD、14k cold）**: 選択 **71,436.4 MB**、常駐 **18,692.6〜18,902.5 MB
   （6,015〜6,096 本）**、上げた **52,533.9〜52,743.8 MB**、`pread` **53,898.4 MB**。
   こちらも **71,436.4 − 18,902.5 = 52,533.9**（一致）。
5. **つまり常駐の再利用は既に厳密**（1 本も二重に読んでいない）。14k で常駐が
   18.9 GB に落ちるのは**追い出しではなく被覆率**（VRAM は 21,999 MiB のまま満杯で、
   14k の選択集合が広いので重なりが 33.7% → 26.5% に下がる）。
6. **先読みの置き場 (P474/P486/P490) はディスクのバイトを減らさない。**
   14k で `ra_MB=11,163.9` / `ra_hit_MB=9,799.3`（`ra_pct=88`）だが、
   `pread_MB` は `up_MB` とほぼ同じ 53.9 GB。置き場は**読む時刻を移すだけ**で、
   読む量は変わらない（タスク文の「earlier ではなく fewer」はここで閉じている）。
7. **既定モデル (t510b) は不変**: decode 10.6 ms/tok、3,826 cold 1.89〜1.90 s、
   14k cold 5.04〜5.85 s、**md5 は全 COND・全リクエストで SAME**、errors 0。
   既定モデルでも常駐は 20.3 GB（9,940 本）/ 18.4 GB（9,070 本）で同じ形。

## 1. なぜこの段を選んだか (Do 2)

- 段プロファイルは P506 の再測定どおり、prefill は **壁の 90-93% が expert の読み**
  (3,826 で `up_MB=43,515` / `copy_s=14.8-18.3`、14k で `up_MB=52,743.8`)。
  decode は experts 5.07 / ssm 2.85 / mix 2.21 / full 1.27 ms/token で、
  ホスト側に残った名前付きの項目が無い (P506 §1)。
- タスク文が名指しした「routing 統計で常駐を選び、ディスクから読むバイトを減らす」
  は、**読む量そのもの**を触る唯一の残った方向なので、そこを選んだ。
- ただし着手前にコードを読んで、**常駐の再利用は既に実装済み**（prefill の
  `iq_pf_stream_layer` が `peek_many` のヒットを `PfStreamEntry{dev: Some}` として
  ステージングから外す。`items_of` は `en.host` のある expert だけを積む）と分かった。
  そこで「実装が効いているか」を数字で確かめる計器を先に入れた（§2）。結果が §0-3/4。

## 2. 変えたこと (`f390db5`)

- `crates/denno-engine/src/model/exp_pf.rs`: `STREAM_RES_EXPERTS` / `STREAM_RES_BYTES`
  / `STREAM_ALL_EXPERTS` / `STREAM_ALL_BYTES` と `note_stream_residency`、
  `stream_residency()`。`stats_reset` で 0 に戻す。`[exp-pf-stream]` の末尾に
  ` res_exp= res_MB= req_exp= req_MB=` を足す。
- `crates/denno-engine/src/model/qwen4exp.rs`: `iq_pf_stream_layer` で entries を
  組んだ直後に、`dev.is_some()` の本数と全体の本数を積む。1 expert のバイト数は
  `row_bytes(gate,e)*ff + row_bytes(up,e)*ff + row_bytes(down,ff)*e`。
- **env ゲートは無い**（計器だけで、既定の経路は 1 命令も変わらない。新しい
  ログ欄が出るだけ）。数値経路・丸め順・アンカーは不変。
- 計器のコストは 1 層あたり `entries` の線形走査 1 回（≤ 512 本）で、
  `[seg-prof]` の 1 層 65-70 us に対して無視できる（推定 1 us 未満/層）。

## 3. lab の結果

### 3.1 t510a（UD-Q4_K_XL、`load_before=14.69`）

| リクエスト | prefill s | 選択 req_MB | 常駐 res_MB (本) | 上げ up_MB | pread_MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3,826 cold (b_1) | 16.42 | 65,651.2 | 22,124.3 (7,043) | 43,526.9 | 43,739.9 |
| 3,826 cold (b_2) | 16.48 | 65,651.2 | 22,072.6 (7,024) | 43,578.6 | 43,803.8 |
| 14k cold (x0_1, 捨て) | 24.01 | 71,436.4 | 18,692.6 (6,015) | 52,743.8 | - |
| 14k cold (b_1) | 41.75 | 71,436.4 | 18,902.5 (6,096) | 52,533.9 | 53,898.4 |
| 14k cold (b2_1) | 19.87 | 71,436.4 | 18,692.6 (6,015) | 52,743.8 | - |

`VERDICT b vs x0 OK decode -1.7% prefill -4.8% md5 DIFF errors 0`
`VERDICT b2 vs x0 OK decode -1.9% prefill -3.2% md5 DIFF errors 0`

- 短い decode は 13.9-14.1 ms/tok（P506 の 14.0 と同じ）。**壁は使えない**:
  `load_before=14.69` で 14k が 19.87 / 24.01 / 41.75 s と 2 倍振れた（P506 §7-3 の
  警告どおり）。採否はバイトの台帳（負荷に依存しないカウント）で読んだ。
- `req_exp` が `experts` より大きい（20,952 / 22,795 対 16,780）のは、層優先 prefill が
  1 層をトークンのタイルに割るため。同じ (層, expert) が複数タイルで数えられる。

### 3.2 t510b（既定モデル UD-IQ3_XXS、`load_before=1.86`）

| リクエスト | prefill s | 選択 req_MB | 常駐 res_MB (本) | 上げ up_MB | pread_MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3,826 cold (b_1) | 1.89 | 42,723.7 | 20,268.2 (9,940) | 22,455.5 | - |
| 14k cold (b_1) | 5.85 | 46,446.7 | 18,431.1 (9,070) | 28,015.6 | 18,465.0 |

`VERDICT b vs x0 OK decode +0.0% prefill -0.9% md5 SAME errors 0`
`VERDICT b2 vs x0 OK decode +0.0% prefill +0.0% md5 SAME errors 0`

- 3 COND とも decode 10.6 ms/tok、3,826 1.89〜1.90 s、**md5 が全リクエストで SAME**。
  計器は既定モデルの数値も出力も変えていない。
- 既定モデルは 14k でも 5 s 台（UD の 20 s と違ってディスク律速ではない。
  83.6 GB でページキャッシュに収まる範囲が広い）。

### 3.3 出力テキスト（健全性）

t510a の `TXTD`（14k の回答）は P498/P506 と同じ趣旨の日本語要約
（「ご提示いただいた文書は、LLM推論エンジン（denno-engine）における Prefill の
性能最適化の歴史と教訓を詳細に記録したものです…」）。`TXTB`（Rust の所有権）も同趣旨。
t510b は全 COND で md5 SAME = バイト一致。

## 4. 数値の扱い

- すべて GPU ラボの実測。このコンテナは GPU 無しなのでマイクロベンチは取っていない。
- `req_MB` / `res_MB` / `up_MB` はカウント由来（1 expert のバイト数 × 本数）で、
  負荷に依存しない。壁と `copy_s` は t510a では負荷で 2 倍振れたので**採否に使っていない**。
- `pread_MB` はファイル読みの実バイト。`ra_MB` は先読みの置き場へ入れたバイトで、
  `pread` と二重に数えられる（置き場は pread の結果を保持するだけ）。
- 帯域・秒は lab の値をそのまま。見積りは「推定」と書いた。

## 5. 次の担当者へ（なぜ 3 ラウンド目を使わなかったか）

タスクの方向（ディスクから読むバイトを減らす）に対して、残る手は**容量**しかない。
測って棄却済み:

1. **VRAM 予算を上げる** = P435「26 GB で decode -3.4% だが 14k cold +8.5%」、
   P452「decode だけ上げると 23.5 GB の refill で decode +7.6%」。今回の台帳でも
   常駐は 22,000 MiB で満杯（UD 3,826 で 7,043 本 = 22.1 GB）と確認できた。
   4,000 MiB 増やすと 14k の被覆は 26.5% → 約 32%（推定）、読む量 -7.6%（推定）だが、
   P435 の実測が逆に 8.5% 悪い。
2. **prefill から VRAM へ積む** (`DENNO_EXPERT_IQ_PF_UPLOAD_MB`) = t151b で
   decode +17.3%（prefill が decode の作業集合を奪う）。既定 0 のまま。
3. **所有する RAM tier** = P458 で wall +20%。**PLE 表の RAM 常駐** = P506 で 28.8 GB、
   decode +234%。**過去セッションの事前ロード** = 共通ルールで棄却済み。
4. **先読みの窓** = P474/486/490 で満杯。今回 `ra_pct=88` を確認。しかも置き場は
   `pread` を減らさない（§0-6）。

したがって次の一手は P506 §5 と同じく `.cu`（seg 層一括 GEMV の帯域、gate 420 /
up 450 / down 345 GB/s = HBM の 20-26%）しかない。本タスクは `.cu` 不可。

## 6. テスト

```
CARGO_TARGET_DIR=/tmp/t510 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1350 passed; 0 failed; 103 ignored
CARGO_TARGET_DIR=/tmp/t510 cargo test --release -p denno-engine --features cuda --test kernel_invariants
  -> 144 passed; 0 failed
```

新規テスト `exp_pf::tests::stream_residency_counts_resident_and_all_separately`
（常駐ぶんと全体を別々に積み、`stats_reset` で 0 に戻ることを見る）。

## 7. GPU で確認してほしいこと

1. 本タスクは**数値経路を変えていない**ので、「既定モデルが変わっていない」は
   t510b（md5 全 SAME）とコード一致の両方で示した。
2. prefill の採否を判定する回は `load_before` が 2 以下のときに取ってほしい
   （t510a は 14.69 で壁が 2 倍振れた）。
3. 「ディスクから読むバイト」を今後も追うなら、`[exp-pf-stream]` の
   `req_MB / res_MB / up_MB / pread_MB` を並べて読むのが最短（本タスクで追加）。
