ADOPT: yes t271a

# REPORT P271 — /2d に expert tier を出し、切り替え / SIGTERM で使用回数を保存する

branch: br271 (br268 から。int-auto 89b7f8a は br268 に merge 済みで、最終確認でも
89b7f8a のまま = 追加 merge なし)。commit: d9812f4 + 最終。

## 0. 結論

- **採る**。viewer と endpoint のテストは緑、使用回数はモデル切り替えと SIGTERM の
  両方で保存される (テストで発火を確認)。lab t271a は gates off / on で md5 全一致・
  errors 0、decode は悪化していない。
- decode の計算経路は 1 命令も変えていない。変わったのはリクエスト終端の計器
  (`exp_iq` の 3 つの atomic load) と、終了経路の保存だけ。

## 1. 前提 (task 文と lab の値。測り直していない)

- lab: SCENARIO=depth8k、PROFILE=1、MODEL=swift15-abl-IQ3_S (既定)、ptx=none。
  baseline COND = int-auto 既定 + `DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000
  DENNO_EXPERT_GPU_MIN_HIT=0 DENNO_GPU_WEIGHT_MB=12000`。
- task 263 の `GET /v1/experts/stats` は tier の本数・MB・覆盖率・ファイル・最終保存
  までだった。task 271 で mmap tier と「直前のリクエストの tier 別ヒット率」を足した。
- 使用回数の記録そのもの (`DENNO_EXPERT_STATS_DIR` / `DENNO_EXPERT_IQ_STATS_FILE`) は
  既定 off のまま。RAM tier (`DENNO_EXPERT_RAM_MB`) も既定 off のまま
  (task 268 / t268c が IQ3_S で +68% として棄却済み。ここでは触っていない)。

## 2. 変えたこと

### 2.1 endpoint: mmap tier と直前のリクエストのヒット率 (serve_flash.rs)

- `vram` / `ram` に加えて `mmap: {experts, mb}` を返す。mmap は**記録された
  (回数 > 0) expert のうち VRAM にも RAM にも載っていない本数**で、MB は
  `expert_bytes / (n_layer * n_expert)` を掛けた値。
- `last_request: {lookups, vram:{hits,hit_rate}, ram:{...}, mmap:{...}}` を返す。
  出所は `exp_iq` の参照/VRAM ヒットと `expert_ram` の VRAM ミス転送元
  (`miss_ram` / `miss_mmap`、task 268)。どちらもリクエスト終端で 0 に戻るので、
  戻す直前に `record_last_tier_hits()` が写す。率は lookups で丸めるので
  合計が 1 を超えない。参照 0 件 (prefill だけで終わったリクエスト) では
  前の値を残す。
- `exp_iq::lookups_hits()` を足した (整形せず生の 2 値)。

### 2.2 /2d の state 欄 (denno-view)

- state 欄に `exp vram` / `exp ram` / `exp mmap` (本数 · MB · 直前の last %)、
  `exp cover` (vram/ram/total の %)、記録ファイルと最終保存時刻を出した。
  整形は view2d.mjs の純関数 (`expertStatsView` / `expertTierText` /
  `expertCoverageText`) に置き、index2d.html は DOM へ書くだけ。
- 読み直し: 起動時、チャットの各リクエストの後、5 秒ごとの poll、モデル切り替えが
  完了した直後 (reload の前)。404 の古いサーバーでは欄を `?` にして理由を出す。
- モック店 (`createMockStore`) に `GET /v1/experts/stats` を足した (形から決まる
  固定値。MB は 1 本 3,072,000 B で計算)。

### 2.3 終了時の保存 (serve_flash.rs)

- `save_stats_on_stop(why)` = `exp_stats::save_now()` (一時ファイル → rename の
  原子的な保存) + 1 行ログ。呼ぶ場所は 2 つ:
  - `POST /v1/model/select` が切り替えを決めた時点 (202 を返す前)。既存の
    終了ループ側でもう一度呼ぶが、保存済みなら `save_now` は何もしない。
  - SIGTERM。専用スレッドで `tokio::signal::unix::signal(SIGTERM)` を待ち、
    通常の文脈で保存して `exit(0)` する。既定の SIGTERM は即死なので、
    20 秒ごとの保存の谷間の回数が消えていた。
- `StatsSaveOnExit` (Drop) は `process::exit` では走らないので、実質はこの 2 経路が
  唯一の終了時保存になる。

## 3. 数値の扱い

- decode / prefill の数値経路は不変。丸め順も変えていない (md5 全一致が根拠)。
- 新しい env ゲートは足していない。viewer / endpoint は読み取り、保存は終了時のみ。
- 既存の `DENNO_EXPERT_STATS_DIR` を on にすると、起動プラン (task 263) と
  記録が動く。lab の on COND はこれだけを足した。

## 4. lab t271a (実測。lab の数字はそのまま)

env (全 COND 共通): `DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000
DENNO_EXPERT_GPU_MIN_HIT=0 DENNO_GPU_WEIGHT_MB=12000`。
on COND だけ `DENNO_EXPERT_STATS_DIR=/tmp/p271stats` を足した。
最初の b0 は捨て COND、最後の b2 は baseline の繰り返し。

| COND | decode ms/tok (median) | prefill s (median) | errors | text | md5 |
| --- | --- | --- | --- | --- | --- |
| b0 (捨て) | 19.4 | 0.61 | 0 | looks ok | a:82ab0c06 b:624e0d4d c2000_cold:b2057e53 c2000_turn2:3852ffec c8000_cold:44178e66 c8000_turn2:44634658 |
| stats (gate on) | 18.5 | 0.57 | 0 | looks ok | 同上 (全部一致) |
| b2 (baseline 繰り返し) | 19.4 | 0.61 | 0 | looks ok | 同上 (全部一致) |

VERDICT stats vs b0 OK decode -4.8% prefill -6.5% md5 SAME errors 0
VERDICT b2 vs b0 OK decode -0.2% prefill +0.0% md5 SAME errors 0

長文 (median):

| COND | c2000_cold | c2000_turn2 | c8000_cold | c8000_turn2 |
| --- | --- | --- | --- | --- |
| b0 prefill s / decode ms/tok | 3.33 / 20.3 | 0.41 / 18.4 | 17.15 / 19.6 | 0.49 / 19.7 |
| stats | 3.28 / 19.8 | 0.41 / 18.5 | 16.42 / 18.8 | 0.49 / 18.1 |
| b2 | 3.34 / 20.0 | 0.41 / 17.7 | 16.44 / 18.8 | 0.49 / 18.2 |

- decode は目標の「1% 以内」を判定できる分解能がこのマシンに無い。同じバイナリ・
  同じ env の baseline 同士 (b2 vs b0) でも c8000_turn2 は -7.5% 動いた。
  stats は -4.8% (短縮方向) で、悪化は無い。prefill も -6.5% (短縮方向)。
- 出力テキストは 3 COND とも sane、md5 は 6 リクエスト全部で一致 (SAME)。
- stats COND のログで保存が実際に走っている: `[exp-stats] on
  file=/tmp/p271stats/dnnoexps-IQ3_S-50292326400.bin`、リクエスト終端ごとに
  `[exp-stats] saved ... counted=9578 cells bytes=98341`。endpoint の材料になる
  `[exp-iq] lookups=61440 hits=31561 hit_rate=0.514` も同 COND で出ている。

## 5. テスト (すべて CPU、このコンテナ)

- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`:
  1142 passed / 0 failed / 100 ignored。
- `cargo test --release -p denno-engine --features cuda --test kernel_invariants`:
  90 passed / 0 failed。
- `cargo test --release -p denno-cli --features denno-engine/cuda -- --test-threads=1`:
  75 passed / 0 failed / 1 ignored。
  - 新規 `model_select_saves_expert_stats`: 記録先を一時ディレクトリにして 1 セル
    数え、`POST /v1/model/select` が 202 を返した時点でファイルができ、セルが
    読めることを確認 (保存後 dirty=false)。
  - 新規 `sigterm_saves_stats_before_exit`: このテストバイナリを子プロセスで
    `--ignored` 起動し、子が自分へ `kill -TERM` を送る。子は保存して exit(0)、
    親は終了コード 0・`SIGTERM` のログ・保存ファイルのセルを確認する
    (ignored の 1 本はこの子テスト)。
  - 既存 `experts_stats_json_has_every_tier` に mmap / last_request のキーを追加。
    新規 `last_tier_hits_split_vram_ram_mmap` は 10 参照 / VRAM 6 / RAM 3 /
    mmap 1 の分解と、率が lookups を超えない丸めを固定する。
- `cargo build --release -p denno-view` + `cargo test --release -p denno-view`:
  5 passed (HTML/JS の include_str! が通る)。
- `node --test crates/denno-view/tests/`: 74 passed / 0 failed (新規 3 本:
  モックの形と MB、実 API 形の整形、404/欠け/範囲外の率の耐性)。

## 6. 見積り (測っていない)

- endpoint の追加コスト: 1 リクエストあたり 24,576 セル (48x512) の走査 1 回と
  atomic load 3 回。**推定** 0.1 ms 未満で、decode の 19 ms/tok に対して無視できる。
- SIGTERM の待ちスレッドは idle で、decode 中もロックを取らない (受信後に
  `save_now` が STATE のロックを 1 回取るだけ)。

## 7. GPU で確認してほしいこと

1. 実機の `/2d` state 欄に vram / ram / mmap の本数・MB と last % が出ること。
   RAM tier は t268c の理由で既定 off のままなので、ram は 0 本・mmap が大半になる
   はず (on にするなら `DENNO_EXPERT_RAM_MB` を別途)。
2. 8098 の `serve_loop.sh` が SIGTERM で止めるとき、`[flash] SIGTERM: ...` と
   `[exp-stats] saved ...` が出て、次の起動の `[exp-plan]` がそのファイルを読むこと。
   終了コードは 0 にした (SIGTERM の既定 143 を待つ監督なら要相談)。
3. モデル切り替え (models.conf) で 202 の直前に保存され、切り替え後の初回
   リクエストの `[exp-plan]` が前モデルのファイルを読まないこと (ファイル名は
   ディレクトリ名 + expert バイトで分かれる)。
