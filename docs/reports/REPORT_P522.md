ADOPT: no t522a t522b t522c t522d

# REPORT P522 — UD-Q4_K_XL: **読みを大きくする (隣接区間の併合 + pinned をファイル順) は
# 3,826 cold prefill を +39% 遅くした。** 素のディスクは自前プローブで **O_DIRECT 4 MiB x 8 本
# = 4.6 GB/s** 出るのに、この buffered+pinned 経路では **0.93 MB x 32 本 (3.4 GB/s) が既に天井**
# で、1.64 MB へ畳むと **copy_s 14.4 -> 21.3 s (+48%)** と逆に遅くなった。
# ゲート 2 本は既定 off で残し、計器 (読みの形の台帳) だけ既定 on。既定モデルは md5 全 SAME。

branch: **br522** = 開始時の int-auto **cdcd6b1** (= merge: br518。自分の br518 は merge 済み
だったので取り込み不要) + 本タスクの 3 本。終端で int-auto を fetch し直したら
**0a7ed94 (merge: br504)** に動いていたので merge した (両側そのまま、`build_ptx.sh` の衝突なし)。

lab 名 (すべて `tree=denno-dev3` / `PROFILE=1` / `ptx=none`) /
baseline COND = タスク文のとおり `DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000
DENNO_EXPERT_GPU_MIN_HIT=0 DENNO_GPU_WEIGHT_MB=12000`:

- **t522a** = `ab6d2e3` / `MODEL=flash-next-UD-Q4_K_XL` / `SCENARIO=depth8k` / `load_before=1.51` /
  COND: x0 b **ld** (ld = baseline + `DENNO_DISK_PROBE=1`)。1 ラウンド目: fresh profile と
  ディスクの素の速さと読みの形。
- **t522b** = `8c24fac` / 同モデル / `load_before=5.85` / COND: x0 b **m m16** b2
  (m = `DENNO_PF_PREAD_MERGE=1`、m16 = + `DENNO_PF_PREAD_THREADS=16`)。probe に 2 MiB を
  足した版。**併合だけでは 1 本も畳めない**ことが分かった回 (下記 §3.2)。
- **t522c** = `64c28f8` / 同モデル / `load_before=2.27` / COND: x0 b **f f8** b2
  (f = `DENNO_PF_PIN_FILE_ORDER=1 DENNO_PF_PREAD_MERGE=1`、f8 = + `DENNO_PF_PREAD_THREADS=8`)。
  本命の A/B。
- **t522d** = `64c28f8` / `MODEL=flash-next-UD-IQ3_XXS` (既定) / `SCENARIO=depth` /
  `load_before=1.70` / COND: x0 b b2。既定モデルの確認。

lab は 4 リクエスト使った (A/B 3 回 + 既定モデル 1 回)。数はすべて GPU ラボの実測。

## 0. 結論 (6 行)

1. **採るゲートは無い (`ADOPT: no`)。** 併合は狙いどおり発火した (pread の回数
   **46,800 -> 26,400 = -44%**、1 本の平均 **0.93 -> 1.64 MB**、読むバイトは不変) が、
   **copy_s が 14.4 -> 21.3 s (+48%)、3,826 cold の壁が 15.8 -> 21.9 s (+39%)**。
   4 本の baseline (16.33/14.39/14.18/14.63) に対して f は 20.09/21.28、
   f8 は 20.55/21.75 と、ラウンドをまたいで同じ向きに出た。
2. **素のディスクはまだ余力がある** (自前プローブ、O_DIRECT、条件ごとに別の 512 MiB 窓):
   1 MiB **3.5** / 2 MiB **4.3** / 4 MiB **4.6** / 16 MiB **4.4 GB/s** (8 本)。
   我々の expert 読みは 1 本 0.93 MB で **3.4 GB/s** = ちょうど 1 MiB の天井。
3. **読みの形 (実測の台帳、3,826)**: 45,107 区間 / 42,030 MB。隣り合う区間の隙間は
   **0 が 38%**、1 MiB 超が **40%**。連続しているところだけ畳んでも 1.64 MB 止まりで、
   隙間まで読むとバイトが増える (§3.3)。
4. **それでも速くならなかった**。O_DIRECT のプローブは「大きいほど速い」と言うのに、
   pinned への buffered pread は逆。**帯域ではなく、buffered 経路固有の何か** (推定:
   1 回の pread の中でカーネルが小分けに発行するため、並列本数が減るほど損をする) が効いている。
   切り分けは §6 に残した。
5. **既定モデル (t522d) は不変**: decode 10.5/10.5/10.6 ms/tok、3,826 cold 1.88-2.00 s、
   **md5 全 COND・全リクエスト SAME**、errors 0、text ok。台帳は既定 on だが数値は動かない。
6. ゲート 2 本 (`DENNO_PF_PIN_FILE_ORDER` / `DENNO_PF_PREAD_MERGE`) は**既定 off で残した**
   (未設定で従来と 1 命令も変わらない。既定 off のままなので切り戻し作業は無い)。計器
   `DENNO_PF_RANGE_LEDGER` (既定 on) と `DENNO_DISK_PROBE` (既定 off) は次に読む形を
   測るために残す。

## 1. 前提: fresh profile と、なぜこの段を選んだか (Do 1 / Do 2)

t522a の b (UD-Q4_K_XL、`PROFILE=1`):

- warm 短 decode **14.1 ms/tok** (lab の負荷で 13.9-14.2)。3,826 cold **15.38 s**、
  14k cold **20.95 s** (どちらも 2 サンプルの中間値)。
- prefill の段は **experts (ディスク) が 82-84%** で、ssm/full/mix は数 % (P494 の
  17,537 / 21,277 ms と同じ比率)。`copy_s` は 3,826 で 14.4-16.3 s、14k で 15.3-16.1 s。
- **この段を選んだ理由**: タスクの Do 2 が「バイトを減らす」か「速く読む」の 2 択に絞って
  いて、バイト側は P494 が **16,778 本 x 3.14 MB = 52.7 GB の 1 パス = 最小**と確定済み。
  残るのは **読み方 (1 回の長さ・順序・経路)** だけだった。P494 は「先読みの窓は使い切り」
  まで示しているので、窓ではなく**pread そのものの形**を測ることにした。
- 前回までに測って棄却済みのもの (再試行していない): `DENNO_PF_PREAD_THREADS` 48/64
  (task 424)、4 MiB への束ね (task 424)、ページキャッシュ経由の先読み (task 470)、
  `DENNO_EXPERT_VRAM_MB` > 22000、expert を飛ばす/prune 系。

## 2. 変えたこと

### 2.1 計器 (commit 1/3)

- **`DENNO_DISK_PROBE`** (既定 off、診断専用): モデル読み込みの直後に、**自分で書いたプローブ**が
  GGUF の最大パート (49.7 GB) を読む。条件は
  **buffered / O_DIRECT x 1/2/4/16 MiB x 1/8/32 本 x seq/scatter の 24 条件**で、
  条件ごとに**別の 512 MiB 窓**を読む (ページキャッシュの再利用を避ける)。
  O_DIRECT は `open(2)` を `extern "C"` で直接呼び、4096 境界の作業バッファへ読む。
  出力は `[disk-probe] ... GB/s=` の 1 行/条件。
- **`DENNO_PF_RANGE_LEDGER`** (既定 on、計器のみ): `pf_copy_into_pin` が実際に要求した
  `(パート, 絶対オフセット, 長さ)` を 1 グループ (= 1 回の MoE 呼び出し) 単位で並べ替え、
  **隣接区間の隙間を 0 / 64 KiB / 256 KiB / 1 MiB / それ以上**の 5 階級で数える。
  `[exp-pf-stream]` の末尾に `rn= rMB= rspan_MB= gap0= gap64k= gap256k= gap1m= gapgt=`。
  **読みの計画には触らない** (並べ替えは台帳の中だけ)。

### 2.2 本命 (commit 2/3, 3/3、どちらも既定 off)

- **`DENNO_PF_PREAD_MERGE`**: 隣り合う読み区間を 1 本の `pread` に畳む。条件は
  **pinned 側もファイル側も連続**していること (`dst_a+len_a == dst_b` かつ
  `part_a == part_b` かつ `off_a+len_a == off_b`)。読むバイトも書く位置も 1 バイトも変えない。
  読めなかったときの memcpy の転送元は先頭区間のものを残す (mmap は連続なので正しい)。
- **`DENNO_PF_PIN_FILE_ORDER`**: `pf_slot_prepare` で `(expert, tensor)` を
  **(パート, オフセット) 昇順**に並べ、`ptrs` (デバイス側) と `parts` (ホスト側) を同じ順で埋める。
  デバイスへ渡るバイト列は同じで、**置き場所の順序だけ**が変わる。
  これが無いと併合は 1 本も成立しない (既定の並びは gate/up/down が expert 順で、
  ファイル上は gate テンソルの次の expert が続くため)。t522b がその証拠:
  `pread_calls` 46,828 -> 46,779 でほぼ不変。

## 3. 数値

### 3.1 素のディスク (t522b x0、24 条件、2 ラウンドで同値)

| 1 回の長さ | 8 本 scatter | 32 本 scatter | 1 本 seq |
| --- | ---: | ---: | ---: |
| 1 MiB | **3.53 / 3.56** | 3.43 / 3.44 | 1.68 |
| 2 MiB | **4.26 / 4.32** | 4.09 / 4.13 | 2.06 |
| 4 MiB | **4.63 / 4.67** | 4.51 / 4.28 | 2.91 |
| 16 MiB | 4.46 / 4.45 | 4.50 / 4.46 | 4.12 |

- **天井は 4 MiB x 8 本の 4.6 GB/s**。1 本ずつだと 16 MiB まで大きくしないと出ない。
- **expert の実形 (0.93 MB) は 1 MiB の 3.4-3.5 GB/s** = 我々の copy 段の実効 3.0-3.4 GB/s と
  ほぼ一致。**「今の読みは 1 MiB の天井に張り付いている」**が、4 MiB まで上げれば
  +30% の余地がある、というのがこのプローブの結論。
- この日のディスクは 3.8-4.7 GB/s。P494 の日は 3.0 GB/s だった。**日差が 1.5 倍**ある。

### 3.2 読みの形 (t522a b、台帳の実測)

| リクエスト | 区間数 rn | 要求 MB | 平均 | gap0 | gap64k | gap256k | gap1m | gapgt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3,826 | 45,107 | 42,030 | 0.93 MB | 16,971 (38%) | 0 | 0 | 4,389 | 18,113 |
| 14k | 45,024 | 40,982 | 0.91 MB | 18,254 (40%) | 0 | 0 | 3,493 | 16,498 |

- **隙間は二極**: 完全に連続 (38-40%) か、1 MiB 超 (36-40%)。中間がほぼ無い。
  → 「小さい隙間だけ埋めて大きくする」策は**このモデルでは選べない**。
- 台帳の rMB は pread に回った分だけで、先読みの置き場 (RAM) から来た分は含まない
  (`rMB` + `ra_MB` = `pread_MB` で一致)。

### 3.3 併合 + ファイル順 (t522c、本命の A/B)

3,826 cold prefill (s、ラウンド別) と `copy_s`:

| COND | prefill r1 | prefill r2 | copy_s r1 | copy_s r2 | pread_calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| b (base) | 18.42 | 15.67 | 16.33 | 14.39 | 46,834 / 46,781 |
| b2 (base) | 15.42 | 15.91 | 14.18 | 14.63 | 46,793 / 46,786 |
| **f** (併合+ファイル順) | **21.33** | **22.53** | **20.09** | **21.28** | **26,467 / 26,367** |
| **f8** (+ 8 本) | **21.80** | **23.51** | **20.55** | **21.75** | 26,409 / 26,384 |

- **併合は確かに発火している** (`pread_calls` -44%、1 本 1.64 MB)。バイトは不変
  (`up_MB` 43,534 vs 43,533)。**それでも copy_s が +48%、壁が +39%**。
- 14k は**この回のディスクが暴れて判定不能**: c8000_cold が b_1 **42.63 s**、
  x0_1 19.62、b2_1 21.52、f 26.50、f8 44.97。1 サンプル/COND なので使わない。
- md5: UD はラウンド間で動く (P518 §3.5 と同じ既知の性質)。**安定している a / c2000_cold /
  c8000_cold は全 COND SAME**、errors 0、text は全 COND `looks ok`
  (`TXTD`: 「ご提示いただいた文書は、LLM推論エンジン（`denno-engine）における
  Prefill（事前計算）フェーズの性能最適化...」)。

### 3.4 既定モデル (t522d)

| COND | decode ms/tok | 3,826 cold s | c2000_cold decode | md5 |
| --- | ---: | ---: | ---: | --- |
| x0 | 10.5 | — | 14.7 | edaa84ad / 1938059f / 9b28e018 / 8b02d5c3 |
| b | 10.5 | 2.00 | 12.9 | 同じ |
| b2 | 10.6 | 1.88 | 12.8 | 同じ |

**md5 が全 COND・全リクエストで SAME**、errors 0。P518 の値 (10.6 / 1.87) と同じ。
台帳 (既定 on) は既定モデルでも `rn=` を出すが数値は動かない。

## 4. 判定 (ADOPT: no の理由)

- 目標 (3,826/14k で 5%、短い decode で 3%) には**遠い**。3,826 cold は**逆に +39%**。
  運用規則 (段で 2 ラウンド 2% 以上勝ち / 他が 1% 超悪化しない) も満たさない。
- 併合の**機構は実測で確認**できた (calls -44%、バイト不変、1 本 1.64 MB)。それで
  遅くなるのだから、**「要求を大きくすれば速い」はこの経路では成り立たない**。
- ゲートは既定 off のまま残す (未設定なら従来と 1 命令も変わらない)。
  **計器 2 つは残す**: 台帳は既定 on (数値不変、次に読む形を見るのに要る)、
  プローブは既定 off (lab で 1 回回せばディスクの天井が分かる)。

## 5. テスト

```
CARGO_TARGET_DIR=/tmp/t522m cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1359 passed; 0 failed; 103 ignored   (int-auto 0a7ed94 を merge した後の最終ツリー)
CARGO_TARGET_DIR=/tmp/t522m cargo test --release -p denno-engine --features cuda --test kernel_invariants
  -> 145 passed; 0 failed
CARGO_TARGET_DIR=/tmp/t522m cargo build --release -p denno-engine --features cuda -> ok
CARGO_TARGET_DIR=/tmp/t522nc cargo build --workspace --release + cargo test --workspace --release
  -> cuda feature 無しの側も緑 (CI が赤くならないことの確認)
```

新規テスト (純関数):
- `cuda::...::task522_gap_bucket_and_probe_plan_truth_table` (隙間の階級、ゲートの解釈、
  条件表 24 個の重複なし、窓が重ならない、scatter が置換になっている)
- `task522_merge_adjacent_runs_truth_table` (両側連続のときだけ畳む / 連続 3 本は 1 本 /
  合計バイト不変)
- `task522_pin_order_file_order_truth_table` (全 (i, m) が 1 回ずつ、ファイル順、RAM tier は後ろ)
- `task522_range_ledger_counts_gaps` (同じパートの中の隙間だけを数える)

**緑は分岐が発火した証拠ではない**ので、lab の台帳で発火を数えた:
`rn=` は t522a/b/c の全 COND で出て、`pread_calls` は t522b の m で
**46,828 -> 46,779 (変化なし)** = 併合が効いていない証拠、t522c の f で
**46,834 -> 26,467 (-44%)** = 発火した証拠。

## 6. 次の担当者へ (切り分けの案)

1. **「大きい buffered pread が遅い」の正体**。O_DIRECT では 4 MiB が最速なのに、
   pinned への buffered では 1.64 MB が 0.93 MB より 48% 遅い。候補:
   (a) **並列本数**が効いている (1 グループ 96 本 -> 55 本。同時発行は 32 本のまま) —
   なら `DENNO_PF_PREAD_THREADS` を増やす方向が正しいが、f8 (8 本) も同じく遅いので
   本数だけでは説明できない。
   (b) buffered の大きな pread は**カーネル内で小分けに発行**され、1 呼び出しの中で直列化する。
   → `DENNO_PF_PREAD_KB` を 512-1024 KiB に**下げる**と 1 呼び出しが小さくなり並列度が上がる
   (今は 4 MiB で、実区間 0.93 MB には効いていない設定)。**未測定。次はここから。**
   (c) O_DIRECT で 2-4 MiB を読む経路 (整列区間だけ、bounce 無し) — 天井 4.6 GB/s に届く可能性。
2. **併合 + ファイル順のコードは既定 off で残っている**。再挑戦するなら
   `DENNO_PF_PIN_FILE_ORDER=1 DENNO_PF_PREAD_MERGE=1` で同じ A/B が 1 リクエストで回る。
3. **台帳 (`DENNO_PF_RANGE_LEDGER`) は既定 on**。`rn=` / `gap0=` / `gapgt=` で読みの形が
   そのまま読める。`DENNO_PF_RANGE_LEDGER=0` で止まる。
4. **ディスクの日差が 1.5 倍** (今回 3.8-4.7、P494 の日 3.0 GB/s)。3,826/14k の
   1 サンプル/COND は判定に使わない。`SCENARIO=depth8k` の 14k はラウンド 1 にしか
   流れない (2 サンプル欲しければ別リクエスト)。
5. 既定モデルは page cache に収まるので、この段の改善は既定モデルにはほぼ効かない
   (t522d で不変を確認済み)。

## 7. GPU で確認してほしいこと

1. 既定モデル (t522d) の md5 が全 SAME なのは**計器だけ**を入れたことの証拠。
   数値経路は 1 命令も変えていない (ゲート 2 本は既定 off)。
2. `DENNO_PF_PREAD_KB` を 512/1024 に下げた A/B (未測定、§6 の (b))。
3. O_DIRECT を expert の実区間 (0.93 MB) で使ったときの実効 (プローブでは 1 MiB 3.5 GB/s
   = buffered と同等なので、天井を出すには 2 MiB 以上が要る)。
