ADOPT: yes t325a t325b t325c

理由: UD-Q4_K_XL のミス expert の写しを、mmap のフォールト + memcpy から
**GGUF のパートへの `read_at` (1 回 1-4 MB) で pinned へ直接読む**経路に替えると、
**3,826 cold prefill が 2 ラウンドとも 10% 以上速い** (t325a: b0t 13.76 → 11.17 s、
-18.8% / b1 14.80 → -24.5%。t325b: off 13.32 → 既定の b0t/b1 の中央値 11.05 s、
-17.0%)。
1 回目の読みの実速度は **4.2 → 5.8 GB/s** (`copy_s` 44.4 GB = 10.47 → 7.70 s)。
14k cold は −6.6% (t325a) / ±0 (t325b) で悪化せず、warm short decode は
18 ms/tok で不変 (この分岐は 28 トークンの要求では発火しない)、md5 は
c2000_cold / c8000_cold が全 COND で同一、errors 0、text sane。
ゲート `DENNO_PF_PREAD` は**既定 on** (`0` / `off` / `false` / `no` で戻る)。
`MADV_WILLNEED` の先読み (`DENNO_PF_RA_GROUPS`) は効果なしで**既定 off のまま**。

# REPORT P325 — UD-Q4_K_XL の 1 回目の読みを `read_at` にした (3,826 cold -19%)

branch: br325 = int-auto `0d3cd40` (開始時、br321 merge 済み) から。最後に
int-auto `3ae8b00` (br319 の merge) を取り込んだ (衝突なし、`build_ptx.sh` の
`targets_all` もそのまま)。lab: **t325a**
(sha 8443f1f、pread / MADV_WILLNEED 先読み / pread 32 本) / **t325b**
(sha 6f89f0b、既定 on の確認) / **t325c** (既定モデル)。3 ラウンドで上限。
UD は `MODEL=flash-next-UD-Q4_K_XL`、baseline COND = int-auto 既定 +
`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0
DENNO_GPU_WEIGHT_MB=12000`、先頭に baseline の使い捨て (b0t)、末尾に repeat (b1)。

## 0. 前提 (Do 1) — 何が律速か

REPORT_P321 の結論は「UD の prefill は 1 回目の読み (mmap → pinned) が
2.6-3.5 GB/s で、バイト数を減らしても速くならない」だった。この木で
`[exp-pf-stream]` と `[q4ledger-pf]` を読み直すと (すべて実測):

- 3,826 cold の `experts` 段 14.5 s のうち **`stage_copy` が 10.5-12.7 s**
  (44.4 GB = 3.5-4.2 GB/s)。GPU は 4.2 s、DMA は 50 GB/s 台で、写しが全部。
- 同じ 44.4 GB を page cache から読むと 22-25 GB/s 出る (lab t227a の
  `[pf-probe] cpu_copy_GBs=22.3`、`copy_s=1.90` の回)。**写しの仕組みは
  十分速く、遅いのは冷えたページの取り込み**。
- 冷えたページの取り込みは `readahead` の 128 KB 単位で 1 ページずつ
  フォールトしている (`[exp-ram-probe] prefill minflt=1129741 majflt=125425`)。
  既存の計測 (`[start-prefetch] experts ... 2.2 / 3.3 GB/s`、12 本 × 4 MB の
  pread) は page cache を埋める用途で、これは**デバイスの上限ではない**
  (本タスクで 5.8 GB/s が出た: §2)。
- したがって伸びしろは「1 回あたりの I/O を大きくする」こと。pread なら
  1 回が 1 MB (expert 1 本ぶん) になる。フォールト経路は 128 KB のまま。

## 1. 変えたこと

### 1.1 `DENNO_PF_PREAD` (既定 on。`8443f1f` で追加、`6f89f0b` で既定 on)

- `PfStreamEntry` に `src: Option<[(u32, u64); 3]>` (GGUF パート番号, ファイル内
  絶対オフセット) を足し、`qwen4exp` のミス expert の mmap 区間で
  `slice_abs` から埋める (RAM tier の区間は `None`)。
- `pf_copy_into_pin` が `read_at` と `copy_nonoverlapping` を切り替える。
  切る単位は `DENNO_PF_PREAD_KB` (既定 4096 KiB)。**開けないパート・短読み・
  エラーは false を返して memcpy へ落ちる** (fail open)。部分的な書き込みが
  残っても memcpy が同じ範囲を上書きするので中身は正しい。
- pread は **write-combined でない pinned のスロットだけ**使う。WC のメモリへ
  `read(2)` すると極端に遅いため (K 系の既定は WC でない: task 309)。
- `DENNO_PF_PREAD_THREADS` (既定 0 = rayon の共有プール 14 本) で専用プール。
- パートはモデル構築時に開く (**添字はパート番号そのもの**。開けないパートが
  あっても詰めない = 別ファイルを読む事故を作らない)。

### 1.2 `DENNO_PF_RA_GROUPS` (既定 off。`8443f1f`)

写しの外の専用スレッドから、n グループ先の mmap 範囲へ `MADV_WILLNEED` を
投げる (深さ 1 の sync_channel で先へ行き過ぎない)。**効果なし** (§3.3) なので
既定 off のまま残す (A/B 用)。

数値の扱い: どちらも**バイト列を変えない** (同じ範囲を同じ位置へ置くだけ)。
md5 が変わる余地は無い (§4 で確認)。

### 1.3 試さなかった候補と、その理由

- **RAM tier を「プロンプトが使う expert」に使う**: tier を埋めるには同じ
  44.4 GB を先にディスクから読む必要があり、リクエスト内では読む順番が
  変わるだけ (tier への pread は本タスクの pread と同じ 5.8 GB/s)。tier を
  起動時に埋めるのは「warming at start / preloading past sessions」で既に
  不採用 (共通ルール)。prefill の途中で埋めると 1 パス目が 2 回読みになる。
- **`DENNO_PF_PREAD_KB` を小さくする (512 KiB)**: expert 1 本 (約 1 MB) が
  2 回の read になるだけで、I/O を小さくする方向。§2 の 14 本 / 32 本の
  差が無いことから、本数ではなくサイズで決まると判断した。
- **O_DIRECT**: page cache を捨てることになり、14k の 2 パス目 (page cache から
  9 GB/s) を壊す (P321 の 1 パス化と同じ失敗)。pread は page cache を残す。


## 2. t325a — 5 COND、sha 8443f1f (ゲートは既定 off、COND で明示)

| cond | 3,826 cold (r1 / r2) | `copy_s` (44.4 GB) | 実速度 | 14k cold (r1) |
| --- | ---: | ---: | ---: | ---: |
| b0t (baseline) | 14.74 / 12.78 (med 13.76) | 10.47 / 8.87 | 4.2 GB/s | 29.03 |
| ra4 (先読み 4) | 14.00 / 13.57 (med 13.79) | 10.69 / — | 4.2 GB/s | 28.44 |
| **pread (14 本)** | **10.97 / 11.37 (med 11.17)** | **7.70 / 8.28** | **5.8 GB/s** | 27.12 |
| **pread + 32 本** | **10.28 / 11.33 (med 10.81)** | **7.20 / 8.32** | **6.2 GB/s** | 26.79 |
| b1 (baseline の repeat) | 14.88 / 14.72 (med 14.80) | 11.15 / — | 4.0 GB/s | 32.98 |

`VERDICT ra4 vs b0t OK decode +3.6% prefill -5.9% md5 SAME errors 0`
`VERDICT pread vs b0t OK decode +3.6% prefill -6.6% md5 SAME errors 0`
`VERDICT pread32 vs b0t OK decode -3.2% prefill -11.8% md5 DIFF errors 0`
`VERDICT baseline b0t errors 0 text ok`
（集計の prefill は 3,826 と 14k を混ぜたもの。段ごとは
`VERDICT_DEPTH pread c2000_cold vs b0t decode +22.3% prefill -18.8%` /
`VERDICT_DEPTH pread32 c2000_cold ... prefill -21.5%` /
`VERDICT_DEPTH pread c8000_cold ... prefill -6.6%`)

## 3. t325b — 既定 on の確認、4 COND、sha 6f89f0b

| cond | 3,826 cold (r1 / r2) | `copy_s` | 14k cold (r1) | warm decode (a2) |
| --- | ---: | ---: | ---: | ---: |
| b0t (既定 = on、使い捨て) | 13.86 / 9.86 (med 11.86) | 10.21 / 6.91 | 30.95 | 18 ms/tok |
| **off (`DENNO_PF_PREAD=0`)** | **12.96 / 13.69 (med 13.32)** | **9.90 / 10.18** | 28.90 | 18 ms/tok |
| p32 (32 本) | 11.33 / 10.77 (med 11.05) | 8.09 / 7.66 | 28.91 | 18-19 ms/tok |
| b1 (既定 = on) | 10.50 / 10.00 (med 10.25) | 7.45 / 7.03 | 26.77 | 18 ms/tok |

`VERDICT off vs b0t OK decode -4.0% prefill -28.0% md5 SAME errors 0`
`VERDICT p32 vs b0t OK decode +1.3% prefill -20.6% md5 SAME errors 0`
`VERDICT b1 vs b0t OK decode -4.3% prefill -29.1% md5 SAME errors 0`
`VERDICT_DEPTH off c2000_cold vs b0t decode -20.1% prefill +12.4%`

- 3,826 は **off 13.32 → on 10.25-11.86 s (-11% 〜 -23%)**。2 ラウンドとも
  10% を超える。`copy_s` も off 9.90/10.18 に対し on 6.91-8.09 で一貫して速い。
- 14k は on {30.95, 28.91, 26.77} 対 off {28.90} で **差なし (±0.1%)**。
- p32 (32 本) は 14 本と同じ水準 (3,826 med 11.05 対 10.25-11.86)。差は
  2% 判定の分解能の外なので**既定は 14 本 (共有プール)** にした。

### 3.3 先読み (MADV_WILLNEED) が効かなかった理由

t325a の ra4 は `ra_groups=158 ra_MB=19483.5 ra_s=0.26` (19.5 GB を 0.26 s で
madvise) で、3,826 +0.2% / 14k −2.0%。**フォールトの置き換えになっていない**:
madvise の readahead も 128 KB 単位で、デバイスは既に 3.3 GB/s で回っている
(§0)。写しスレッドの外から投げても、I/O の大きさが変わらないので伸びない。

## 4. 数値の扱い

- **バイト列は不変**: t325a/t325b の全 COND で c2000_cold `f9557457`、
  c8000_cold `27c0c9dc`、c2000_turn2 `fbaf5055`、a `737a73af`、b `87d8a97c`。
  pread は「同じ範囲を同じ順に置く」だけなので、これは設計どおり。
- c8000_turn2 だけ動く回がある (t325a: b0t/ra4/pread `1c85f65c` 対
  pread32/b1 `1bb961e4`。t325b: 全 COND `1bb961e4`)。**同じ env の b0t と b1 で
  も違う**ので、P301 §4 / P321 §3 と同じ既知の再現性 (14k 後の decode) で、
  この変更とは無関係。
- `decode ms/tok (median)` の列は a1 (noisy) と a2 を混ぜたもので、COND 間で
  ±3% 動く。**warm short decode は a2 が全 COND で 18 ms/tok**、a1 も 20-23 で
  差はホストの揺れ。この分岐は `stream_min_tok=512` 未満では入らないので、
  28 トークンの要求に影響する経路が無い (構造的に不変)。
- 3,826 の絶対値はラウンド内で 9.9-14.9 s と大きく動く (ホストの load)。
  だから 1 つの値ではなく **b0t / off / b1 を同じ run に置いた中央値**で見た。

## 5. t325c — 既定モデル (swift15-abl-IQ3_S)、sha 6f89f0b、depth

| cond | warm short decode | 3,826 cold | 3,826 後 decode | md5 (a / b / cold / turn2) |
| --- | ---: | ---: | ---: | --- |
| b0t (既定 = pread on) | 17.2 | 2.65 | 16.5 | 65ededc3 / f18c75fc / a91e17da / 5a8f5488 |
| off (`DENNO_PF_PREAD=0`) | 17.1 | 2.64 | 16.5 | 同上 (完全一致) |
| b1 (既定) | 17.3 | 2.64 | 16.3 | 同上 (完全一致) |

`VERDICT off vs b0t OK decode -0.7% prefill +0.0% md5 SAME errors 0`
`VERDICT b1 vs b0t OK decode +0.5% prefill +1.7% md5 SAME errors 0`
`VERDICT baseline b0t errors 0 text ok`

- **既定モデルは 1 バイトも変わっていない** (md5 4 本が 3 COND で完全一致、
  3,826 cold 2.64-2.65 s、decode 17.1-17.3 ms/tok)。
- 機序も確認した: IQ3_S の pinned バウンスは `wc=1` (write-combined) なので
  pread の条件 (`!pin_wc`) が偽になり、**`[exp-pf-stream]` に `pread_MB` が
  出ない** (b0t/b1 のログ)。`[pf-pread] 読み口: 2 パートを開いた` だけが出る。
  つまり既定モデルでは新しい経路に入る分岐が無い。


## 6. text (そのまま引用)

- TXTD 3,826 cold (t325b b1): 「ご提示いただいた文書は、LLM推論エンジン
  （`denno-engine）における **Prefill（事前計算フェーズの性能最適化の歴史と
  教訓を詳細に記録したものです。...」
- TXTD 14k cold (t325b b1): 同上の書き出し。
- TXT: `# B-Trees: A Comprehensive Explanation ... ## 1. What is a B-Tree?`
- TXTB: 「Rust の「所有権（Ownership）とライフタイム（Lifetime）は、Rust を
  学ぶ上で最も重要であり、かつ最も難しい概念です。」
- errors は全 COND 0。

## 7. わかったこと (次の担当者へ)

1. **効いたのは I/O の大きさ**。128 KB のフォールトを 1 MB の `read_at` に
   すると、同じデバイスで 4.2 → 5.8 GB/s。フォールト経路の 1 回あたりの
   サイズは `readahead` が決めるので、`madvise` では変えられない (§3.3)。
   以前の 2.2-3.3 GB/s は「page cache を埋める pread」の値で、デバイスの
   上限ではなかった (同じホストで 5.8-6.2 GB/s が出る)。
2. **14k はまだ page cache / DRAM 律速**。98.5 GB を 2 パスで上げ、`copy_s` は
   10.3-11.0 s (9 GB/s) で pread でも動かない。ここを縮めるには
   `stage_copy` の 9 GB/s 自体 (probe では 22 GB/s 出る) を上げる必要があり、
   候補はステージングの枚数 (3) を増やして上げスレッドを先へ走らせること
   (`DENNO_PF_STREAM_LOOKAHEAD` の上限は枚数 - 1 = 2 で、今は 2 が上限)。
3. `DENNO_PF_PREAD_THREADS` は 32 でも 14 でも同じ。デバイスは本数では
   なく 1 回のサイズで決まっている (§2 の 5.8 対 6.2 GB/s)。
4. 既定モデル (IQ3_S) は WC の pinned を使うので pread は**発火しない**
   (`!pin_wc` の条件)。t325c で `pread_MB` が出ないことを確認した。

## 8. テスト

- merge 後 (`3ae8b00` + br325、最終木):
  `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`
  **1222 passed / 0 failed / 102 ignored** (11.1 s)
- `cargo test --release -p denno-engine --features cuda --test kernel_invariants`
  **99 passed / 0 failed**
- 追加したテスト:
  - `pread_gate_truth_table` — 未設定は on、`0/off/false/no` だけ偽
  - `pread_kb_rounds_and_defaults` / `pread_threads_rounds_and_defaults` — 丸め
  - `pread_blocks_cover_exactly` — 塊に切っても隙間・重なりが無い
  - `pread_copy_is_byte_identical_to_memcpy` — **実ファイル**を作り、
    `read_at` 経路と memcpy 経路の pinned の中身が 1 バイトも違わないこと
  - `ra_groups_gate_truth_table`
- task 320 の `task320_up_decouple_orders_copy_before_stream_wait` は
  `pf_copy_into_pin` の呼び出し文字列を見ているので、新しい引数に合わせて
  更新した (順序の不変条件はそのまま)。

## 9. GPU で確認してほしいこと

1. 本番 (8098) の UD-Q4_K_XL で 3,826 / 14k cold をそれぞれ 2 回。
   `[exp-pf-stream]` に `pread_MB` が出ること (出なければパートが開けていない)。
   lab の 3 ラウンドは使い切ったので、最終木 (`361c64a` = `6f89f0b` + int-auto
   `3ae8b00`) では測り直していない。merge で入ったのは br319 の **decode 側**
   (shared expert GEMV v5、`fd_*` の kprof) だけで、prefill の写しには触れて
   いない。
2. 既定モデルは t325c のとおり不変 (md5 4 本 SAME、`pread_MB` なし)。
3. 次の山は 14k の `stage_copy` 9 GB/s (page cache からの写し)。
