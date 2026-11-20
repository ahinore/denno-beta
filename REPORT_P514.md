ADOPT: no t514a t514b t514c

# REPORT P514 — UD-Q4_K_XL のホスト側 (page cache / RAM tier / 追い出し) を実測した。
# **page cache は既に expert の 67-70% (49-51 GB / 73.45 GB) をリクエスト間に保持していて、
# 3,826 cold のディスク読みは 43.5 GB ではなく 17.0 GB だった** (残りは page cache)。
# 追い出しているのは expert の pread 列そのもの。落とす方策 (DENNO_PF_FADV_END) は
# 3,826 で prefill +31% / ディスク読み +65% と**明確に悪い**。既定 off のまま棄却。
# タスク目標 (5% / 3%) に届く変更は無い。

branch: **br514** = 開始時の int-auto **890fc66** (= merge: br496) + **br510 の 2 本**
(e3fc826 と f390db5 = `[exp-pf-stream]` の常駐台帳)。br510 は int-auto に未 merge だったので
br514 に含めた (タスク文の「必要なら merge してよい」に該当。数値経路は br510 も不変)。
終端で int-auto を fetch し直したが **890fc66 のまま**なので merge は無い。

lab 名 (すべて `tree=denno-dev3` / `SCENARIO=depth8k` / `PROFILE=1` / `ptx=none` / SHA `9b62b80`) /
baseline COND はタスク文のとおり `DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000
DENNO_EXPERT_GPU_MIN_HIT=0 DENNO_GPU_WEIGHT_MB=12000`。どの COND にも `DENNO_EXPERT_RAM_PROBE=1`
(計器) を足した。

- **t514a** = `MODEL=flash-next-UD-Q4_K_XL` / `load_before=2.34` / COND: x0 b c b2 (4)
- **t514b** = `MODEL=flash-next-UD-Q4_K_XL` / `load_before=1.74` / COND: x0 b c (2 ラウンド目)
- **t514c** = `MODEL=flash-next-UD-IQ3_XXS` (既定) / `load_before=1.30` / COND: x0 b c

`c` = baseline + **`DENNO_PF_FADV_END=1`** (本タスクで足したゲート。リクエスト終端に
expert 範囲へ `posix_fadvise(DONTNEED)`)。

## 0. 結論 (7 行)

1. **採らない (`ADOPT: no`)。** 測った結果、タスク文の前提「ディスクから 52.7 GB を毎回
   読んでいる」は**このビルドでは既に成り立っていない**。3,826 cold の実ディスク読みは
   **17.0-17.5 GB**、14k cold は **21.8-24.0 GB**。残りは page cache が返している。
2. **リクエスト間の page cache 常駐 (実測、b のリクエスト終端)**: expert **49.1-51.2 GB /
   73.45 GB (67-70%)**、モデル全体 **53.6-56.1 GB / 106.2 GB (51-53%)**。
   差の **4.5 GB が dense / PLE**。MemAvailable は 55.6-57.9 GB で、ホスト RAM は
   page cache が使い切っている (空きを作る余地は無い)。
3. **何が追い出しているか**: expert の pread 列そのもの。1 リクエストで page cache へ
   入るのは 3,826 で **43.5 GB**、14k で **52.9 GB** = 59 GiB のホスト RAM の 74% / 90%。
   つまり 1 本の 14k リクエストで page cache はほぼ総入れ替えになる。lab の load は
   1.3-2.3 で、8098 のリクエストも 0 (他のプロセスのせいではない)。
4. **二重持ちは無い** (baseline)。RAM tier も置き場も off (`DENNO_EXPERT_RAM_MB` 未設定)
   なので、expert は page cache か VRAM のどちらか1箇所にしか無い。常駐 51 GB +
   非 expert 4.5 GB = 55.5 GB でホスト RAM とほぼ一致し、余分な写しは見えない。
5. **落とす方策は負け**: `DENNO_PF_FADV_END=1` は 3,826 cold で prefill **15.7 -> 20.5 s
   (+31%)**、ディスク読み **17.0 -> 28.1 GB (+65%)**、3,826 の次の turn2 の decode
   **18.0-20.8 -> 77.6 ms/tok (3.7-4.3 倍)**。14k は変化なし (20.9 / 20.5 対 21.4-22.8 s、
   もともと全部読み直すので差が出ない)。**page cache に残っていることが効いている**
   という直接の証拠で、この方向は閉じた。
6. **既定モデルは不変** (t514c): 全 COND・全リクエストで **md5 SAME** (a:edaa84ad
   c2000_cold:9b28e018 c8000_cold:672fec07)、decode 10.6 ms/tok、errors 0。
   既定モデルでは `c` がさらに悪い (3,826 cold 5.48 -> 11.62 s) = 83.6 GB のモデルは
   expert の大半が page cache に載るので、落とすと即座に損になる。
7. **残っている段はディスクではない**: 3,826 の `copy_s` は 14.3-14.6 s で、その中の
   ディスクは 17.0 GB (実効 1.2 GB/s)。**43.5 GB を 3.0 GB/s で写している**のが壁で、
   page cache 側の工夫では動かない (P510 の「残りは .cu」と同じ結論に、別の計器で到達)。

## 1. なぜこの段を選んだか (Do 2)

- P510 の台帳で prefill の壁の 90-93% は expert の読みで、その内訳は
  「選択 65.7 GB − VRAM 常駐 22.1 GB = 43.5 GB を上げる」だった。ここまでは
  「読む量 = 選択 − 常駐」で厳密。
- 本タスクの指示は「ディスクから読むバイトを減らす」なので、まず**実際にディスクから
  何バイト読んでいるか**を測る計器を入れた (§2)。`up_MB` (= ステージングへ上げた量) は
  ディスク読みではなかった (`/proc/self/io` の `read_bytes` で分離)。
- そのうえで page cache / RAM tier / 追い出しの 3 点を測り、追い出しを止める方策を試した。

## 2. 変えたこと (`9b62b80`)

- `crates/denno-engine/src/model/expert_ram.rs`
  - `[exp-ram-probe]` に `/proc/self/io` の差分 **`io_rchar_MB` / `io_read_MB`** を追加。
    `io_read_MB` がタスク文の disk bytes (page cache で済んだぶんは入らない)。
  - モデル全パートの mmap を覚えて **`file_resident`** (dense / PLE 込みの常駐 MiB) も出す。
    `expert_resident` との差が「expert 以外が page cache に残っている量」。
  - 新ゲート **`DENNO_PF_FADV_END`** (既定 off): リクエスト終端に expert テンソル範囲へ
    `posix_fadvise(DONTNEED)`。1 行 `[exp-pc-end]` を出し、`DENNO_EXPERT_RAM_PROBE=1` の
    ときは落とす前後の `expert_resident` / `file_resident` も並べる。
- `crates/denno-engine/src/model/qwen4exp.rs`: `install_part_map` (全パートの mmap) と、
  `fadv_end_enabled()` でも expert 範囲を登録する条件。
- `crates/denno-cli/src/serve_flash.rs`: リクエスト終端 (統計を出す関数) で
  `expert_ram::drop_expert_pages_end(tag)` を呼ぶ。
- **数値経路は 1 命令も変えない**。既定で変わるのはログ欄だけ (`DENNO_EXPERT_RAM_PROBE=1`
  のときのみ)。`DENNO_PF_FADV_END` は既定 off。

### この環境の `fadvise(DONTNEED)` の挙動 (方策の設計根拠、実測)

`/tmp/pc_test2.py` (64 MiB のファイル) で確認した:

| 状態 | drop 後の常駐 |
| --- | --- |
| pread で page cache に入れただけ (未フォールト) | 65,536 KiB -> **0 KiB** |
| mmap でフォールトした | 65,536 KiB -> **65,536 KiB (落ちない)** |

つまり `fadvise(DONTNEED)` は **mapcount 0 のページだけ**を落とす。だから
「リクエスト終端に expert 範囲を落とす」は pread の読み残しだけを消し、mmap で使った
ページは残す。この性質は単体テスト `fadvise_drops_unmapped_but_not_mapped_pages` で固定した。

なお **mmap してフォールトしたページは落とせない**ので、「VRAM 常駐 expert の page cache を
落として他へ回す」という案は原理的に効かない (それが実装できたとしても 0 バイトしか
落ちない)。page cache の常駐を減らす手段はこの環境には無い、が本タスクの実測結論。

## 3. lab の結果

### 3.1 t514a / t514b (UD-Q4_K_XL)

3,826 cold prefill (壁 / `[exp-pf-stream]` の `copy_s` / 実ディスク読み):

| COND | prefill s | copy_s | up_MB | io_read_MB | ディスク比率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| b (a, r1) | 15.83 | 14.56 | 43,515.1 | 17,547.9 | 40.3% |
| b2 (a, r2) | 15.58 | 14.30 | 43,527.9 | 17,220.5 | 39.6% |
| b (b, r1) | 15.54 | 14.32 | 43,532.9 | 17,037.5 | 39.1% |
| b (a, r2=2 本目) | 15.77 | - | - | - | - |
| c (a) | 20.57 | 19.33 | 43,515.1 | 28,220.8 | 64.9% |
| c (b) | 20.55 | - | - | 28,136.2 | 64.7% |

14k cold prefill:

| COND | prefill s | copy_s | up_MB | io_read_MB | ディスク比率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| b (a) | 22.78 | 18.67 | 52,743.8 | 21,770.1 | 41.3% |
| b2 (a) | 21.66 | 17.62 | 52,545.6 | 23,974.5 | 45.6% |
| b (b) | 21.44 | 16.63 | 52,880.5 | 23,699.6 | 44.8% |
| c (a) | 20.92 | 16.88 | 52,743.8 | 24,726.2 | 46.9% |
| c (b) | 20.46 | - | - | 23,984.0 | 45.4% |

- **3,826 は c が +31% 遅い** (20.5 対 15.7 s、3 サンプルずつ)。ディスク読みは +65%。
  14k は c のほうが僅かに速いが、これは 14k がもともと全部読み直すため
  (ディスク比率 41-47% で c と差が無い)。**3,826 の差は page cache の損失そのもの**。
- 3,826 の turn2 の decode は b が 18.0 / 20.8 ms/tok、c が **77.6 / 77.7 ms/tok**。
  c2000_cold の decode も b 20.8-23.5 対 c 77.6-77.7。**落とした expert を次の
  リクエストがディスクから読み直している**。`[q4prof]` の `experts` も b 1,867 ms
  対 c 3,283 ms。
- `[exp-ram-probe]` の常駐 (リクエスト終端): b は expert 49.1-51.6 GB (67-70%) /
  file 53.6-56.1 GB、c は drop 直後に expert 30.7-37.2 GB まで下がり、次の大きな
  リクエストで 51-53 GB まで戻る。**落ちるのは 13-19 GB で、残りは unmapped でない**
  (§2 の表と一致)。
- `VERDICT b vs x0 OK decode +0.0% prefill +0.0% md5 DIFF` (t514b)、
  `VERDICT c vs x0 OK decode +19.0% prefill +10.2% md5 DIFF`。x0 は捨て COND の
  あとの温かい状態なので b と同等 = **baseline は計器を足しても変わっていない**。

### 3.2 t514c (既定モデル UD-IQ3_XXS)

| COND | decode ms/tok | 3,826 cold s | 14k cold s | md5 |
| --- | ---: | ---: | ---: | --- |
| x0 | 10.6 | 6.95 | 8.06 | a:edaa84ad c2000_cold:9b28e018 c8000_cold:672fec07 |
| b | 10.6 | 5.48 | 7.60 | すべて SAME |
| c | 10.6 | 11.62 | 7.72 | すべて SAME |

- **md5 が全 COND・全リクエストで SAME**、errors 0、decode 10.6 ms/tok。計器もゲートも
  既定経路の数値を変えていない。
- 既定モデルでは `c` が 3,826 で 2.1 倍遅い。83.6 GB のモデルは expert の大半が
  page cache に載るので、終端に落とすと次のリクエストが全部ディスクになる。

### 3.3 出力テキスト (健全性)

t514a/b の `TXTD` (14k の回答) は P498/P506/P510 と同じ趣旨の日本語要約
(「ご提示いただいた文書は、LLM推論エンジン（denno-engine）における Prefill（事前計算フェーズの
性能最適化の歴史と教訓を詳細に記録したものです…」)。`TXTB` (Rust の所有権) も同趣旨。
t514c は全 COND で md5 SAME = バイト一致。

## 4. 数値の扱い

- すべて GPU ラボの実測。このコンテナは GPU 無しなのでマイクロベンチは取っていない
  (`fadvise` の挙動だけは `/tmp/pc_test2.py` でこのコンテナ上の実測)。
- `io_read_MB` は `/proc/self/io` の `read_bytes` の差 = **ディスクから読んだ実バイト**。
  `up_MB` / `pread_MB` は「要求した量」なので、その差が page cache の寄与。
  `read_bytes` は page fault 由来の読みも含むが、本経路の主因は pread。
- `expert_resident` / `file_resident` は mincore の常駐で、**mapcount 0 のページも含む**
  “page cache に載っている量”。落とせる量 (§2) とは別物。
- 壁時計は lab の値をそのまま。3,826 は 3 サンプルで安定 (15.5-15.8 / 20.5)、14k は
  1 サンプル/COND なので ±10% の幅を見込む。見積りは「推定」と書いた。

## 5. なぜ RAM tier を作り直さなかったか (新しい理由があるかの検討)

タスク文の例示「ホット expert をホスト RAM に 1 回だけ置く RAM tier を、その範囲の page cache を
落として作る」は、**この実測では成立しない**:

1. ホスト RAM は **page cache が既に使い切っている** (MemAvailable 55.6-57.9 GB は全部
   reclaimable な page cache)。owned tier は page cache を**置き換える**だけで、
   合計の被覆は増えない (59 GiB という容量は動かない)。
2. しかも **落とせない**: mmap でフォールトしたページは `fadvise(DONTNEED)` で消えない
   (§2)。tier の範囲を page cache から落とすには mapping を壊す必要があり、
   それはモデルの読み口そのものを壊す。
3. 実測でも tier 相当の「落とす」は +31% / +65% のディスク読み。容量を増やせない以上、
   置き換えは必ず損になる。P458 の「owned RAM tier で wall +20%」と同じ結論を、
   バイトの台帳で裏付けた形。

## 6. 次の担当者へ

- **ディスクはもう壁ではない** (3,826 で 17 GB / 14.56 s の `copy_s` のうち 1.2 GB/s)。
  壁は **43.5 GB を 3.0 GB/s で写す staging 経路**。t325a の pread 採用時は 5.8-6.2 GB/s
  だったので、いまは**同じ経路が半分の速度**しか出ていない (モデルが 104 -> 111 GB に
  変わった / ディスク読みが混ざる)。ここを詰めるのが次の一手で、`.cu` か
  staging の並列度 (pread スレッドは 32 で飽和済み = t424) の見直しになる。
- 「読むバイトを減らす」は **page cache が既に 60% 削っている**。残り 40% を削るには
  容量が要るが、ホスト RAM は満杯で、VRAM 側の容量は P435/P452 で棄却済み。
- 計器は `[exp-ram-probe]` の `io_rchar_MB / io_read_MB / expert_resident / file_resident`
  を並べて読むのが最短。`DENNO_PF_FADV_END=1` は「何が落とせるか」を見る診断として使える。

## 7. テスト

```
CARGO_TARGET_DIR=/tmp/t514 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1354 passed; 0 failed; 103 ignored
CARGO_TARGET_DIR=/tmp/t514 cargo test --release -p denno-engine --features cuda --test kernel_invariants
  -> 145 passed; 0 failed
CARGO_TARGET_DIR=/tmp/t514b cargo build --release -p denno-cli --features denno-engine/cuda  -> ok
CARGO_TARGET_DIR=/tmp/t514b cargo build --release --workspace                              -> ok
```

新規テスト: `expert_ram::tests` の 4 本
(`fadvise_drops_unmapped_but_not_mapped_pages` / `proc_io_see_pread_bytes` /
`part_map_is_used_by_file_resident` / `end_drop_is_inert_without_the_gate`) と、
`kernel_invariants::t514_host_page_cache_instruments_and_end_drop_are_wired`。

## 8. GPU で確認してほしいこと

1. 本タスクは**数値経路を変えていない** (既定モデル t514c の md5 全 SAME とコード一致)。
2. 「ディスク読み」を今後も追うなら `[exp-ram-probe] io_read_MB` を読む (本タスクで追加)。
   `up_MB` と並べると page cache の寄与がそのまま出る。
3. prefill の採否を判定する回は `load_before` が 2 以下のときに取ってほしい
   (t514a は 2.34、t514b は 1.74 で、3,826 は 3 サンプルとも 15.5-15.8 s で安定した)。
