ADOPT: yes t546b t546c

# REPORT P546 — UD-Q4_K_XL の 3,826 cold prefill が **-7〜-11%**:
# `copy_s` の 43.6 GB のうちディスクから来る 17-21 GB を **page cache を通さず
# O_DIRECT + 整列バウンス**で読む (`DENNO_PF_PREAD_ODIRECT_BOUNCE`、既定 on)。
# ディスクの実効 2.19 -> 2.48 GB/s、壁 9.2-10.2 -> 8.7-9.5 s。decode と既定モデルは不変。

branch: **br546** = 開始時の int-auto **7b9b2e9 (merge: br520)**。
br542 は int-auto に未 merge だったが、**このタスクでは使わない** (P542 の
`DENNO_PF_SPLIT_IO` は既定 off のまま br542 に残す。下の §2 に、なぜ P542 の
次の一手を採らなかったかを台帳の数字で書いた)。終端で int-auto を fetch し直したら
**65a3bf6 (merge: br565)** に動いていたので merge した (衝突なし。br565 は
prefill の res_up で、こちらは staging の読み口)。

lab 名 (すべて `tree=denno-dev3` / `PROFILE=1` / `ptx=none`):

| lab | SHA | MODEL / SCENARIO | COND | 用途 |
| --- | --- | --- | --- | --- |
| **t546a** | `7b9b2e9` (= int-auto) | UD-Q4_K_XL / depth | x0 b **diag** b2 | fresh な段プロファイル (`DENNO_MIX_KPROF=1 DENNO_SEG_PROF=1`) |
| **t546b** | `76cc1e1` | UD-Q4_K_XL / depth8k | x0 b **odb** b2 | ゲートの A/B (既定 off) |
| **t546c** | `2db7769` | UD-Q4_K_XL / depth8k | x0 b **off** b2 | 既定 on の確認 (off = 明示 0) |
| **t546d** | `2db7769` | **UD-IQ3_XXS (既定)** / depth | x0 b off b2 | 既定モデルが不変なこと |

baseline COND = タスク文のとおり `DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000
DENNO_EXPERT_GPU_MIN_HIT=0 DENNO_GPU_WEIGHT_MB=12000`。`odb` = それ + 
`DENNO_PF_PREAD_ODIRECT_BOUNCE=1`、`off` = 明示 0 (task 546 前の経路)。
3 ラウンド + 既定モデル 1 本 (P534 と同じ理由: 既定モデルは別 MODEL なので
同じリクエストに入らない)。

## 0. 結論 (7 行)

1. **採る (`ADOPT: yes t546b t546c`)。** 3,826 cold が t546b で **-8.6 / -7.1%**
   (b 10,220 / 9,401 -> odb 9,197 / 8,730 ms)、t546c (既定 on) で
   **off 10,240 (9,660 / 10,820) 対 on 9,140 (9,540 / 8,670 / 9,280 / 8,990) = -10.8%**。
   目標 (-5%) を 3 ラウンドで満たした。14k cold も -11.3 / -10.1%。
2. **効いたのは「ディスク読みを page cache に通さない」こと。** 43.6 GB の pread のうち
   ディスクから来るのは 16.9-21.2 GB で、これを buffered pread で読むと
   (a) ページキャッシュへの write、(b) readahead、(c) 常駐 (49-51 GB) の追い出し、
   が付いてくる。`mincore` で「載っていない」と分かった区間だけ O_DIRECT +
   4096 整列バウンスにすると、`copy_s` は 8.80 -> 7.79 / 8.00 -> 7.37 s、
   ディスク実効 2.19 -> 2.48 GB/s (**+13%**)。**読むバイト数は同じ**
   (io_read 19.3/16.9 GB 対 19.5/17.6 GB)。
3. **発火は台帳で数えた** (緑は証拠ではない): `odb_MB=18597 / odb_n=23927 /
   odb_fail=0` (t546b)、`20638/26336`、`16938/21895`、`19390/24957`、`17876/23106` (t546c)。
   `nc_miss_MB` と `odb_MB` が一致 = 「載っていない」の予測どおりだけを回している。
   `off` と既定モデルは **行ごと出ない** (発火 0)。
4. **decode は不変。** 13.2 ms/tok が t546b の b/odb/b2、t546c の b/off/b2 で同値
   (13.1-13.4 の帯)。`gpu_busy` も 1,117 ms で 4 COND 同値。
5. **既定モデルは 1 命令も変わらない** (t546d)。decode 32.6/32.5/32.4/32.5、
   3,826 cold 30.30/30.18/30.04/29.86 s、**md5 が全 COND・全コンテキスト SAME**、
   `odb_MB` は 1 行も出ない (既定モデルは staging で pread を使わない = `src` を持たない)。
   ただしこの回の絶対値 (decode 32.5 ms/tok、3,826 が 30 s) は履歴 (10.4 / 2.1 s) と
   桁が違う。**4 COND が全部同じなので「変わらない」の判定はできる**が、
   この日のホスト/ディスクが別の仕事で埋まっていた可能性がある (推定、原因は未特定)。
6. **値は 1 バイトも変えない。** O_DIRECT で読むのは同じファイルの同じバイトで、
   写す先も同じ。長い 2 本 (`c2000_cold` / `c8000_cold`) の md5 は全 COND SAME。
   短い `b` / `c2000_turn2` だけ DIFF するのは P490/P494/P526/P534 が記録した
   **既存の decode の揺れ** (同じバイナリの COND 間でも変わる。t546c の `off` と
   `b2` は同じ env で `c8000_turn2` が一致)。
7. `.cu` / `.ptx` は触っていない (ホストの読み口だけ)。

## 1. fresh な段プロファイルと、なぜこの段を選んだか (Do 1 / Do 2)

### 1.1 現 int-auto (7b9b2e9) の段 (t546a / t546b の b)

warm short decode (a2、128 token) は **13.2 ms/tok**。`[q4ledger]` と
`[q4prof]` (b_1、2 リクエスト x 128 token の合計を 256 で割った値):

| 段 | ms/token | 中身 / 出所 |
| --- | ---: | --- |
| experts | **4.5-6.5** | `[seg-prof]` GPU 50-57 us/層 (gate 33-36 + up 2-5 + silu 1.5-3 + down 9-14 + sum 3-4)、ホスト 36-40 us/層 (d2h 16-20 / h2d 10-14 / launch 5.5 / lookup 1.3 / table 0.6)。VRAM のヒット率で 2 ms 振れる |
| ssm | 2.77 | gate 1.39 + gdn 1.38 (GPU、P392/496/504 が既に 4 レーン化) |
| mix | 2.3-2.8 | `[mix-kprof]` 97 呼び出し/token、down/up とも **GPU 100%** (Q8_0、21-46 us/呼) |
| full | 0.6-1.3 | flashdec (P520 が -6.2%) |
| shared | 0.09-0.69 | gpu 律速 |
| head | 0.23 | GPU |
| ple / router | 0.22 / 0.35 | |

prefill の 1 リクエスト (b_1、3,826 cold、`[q4ledger-pf]`):

`wall=10,532 ms / experts 9,659 (91.7%) / ssm 380 / mix 278 / other 171 / full 45`、
`gpu_busy=1,117 ms` で**ホスト待ちが 90%**。`[exp-pf-stream]` は
`up_MB=43,586 copy_s=9.19 ready_s=9.32 slotwait_s=0.00`、
`[exp-ram-probe] io_read_MB=20,707`、`pread_MB=43,640 pread_calls=55,906`。
14k cold は `wall=14,903 / experts 10,166 (68%) / other 1,931 / ssm 1,885`。

### 1.2 なぜ staging の「読み口」か

- decode の残りは**全部 GPU 律速**で、`.cu` か CUDA graph しかない。直近の実測:
  seg GPU 50-57 us/層 (P534 が down を 30-33 -> 16-18 us/層にして -6.2% を取った後)、
  ssm/mix は起動あたりの実時間待ち (P392/P496/P502/P504/P526 が
  「消しても壁が動かない」を 3 回再現)、d2h は P502 が機構ごと閉じた。
- 一方 3,826 cold は **壁の 92% が staging の copy** で、そのうち
  **16.9-21.2 GB がディスク**。ここだけが「まだ 1.3 倍の余地がある」と
  実測で分かっていた (t522 の O_DIRECT プローブ: 4 MiB x 8 本で 4.6 GB/s、
  t530 の buffered 0.93 MB x 32 本で 3.4 GB/s。実リクエストは 2.19-2.48 GB/s)。
- 既に測られて棄却済みの手は全部よけた: 塊 (32 -> 128/256 MiB、P538/P542)、
  pread の本数 (32/48/64、t424/P542)、pread を細かくする (384 KiB と 1 MiB の組、
  P530/P538)、隣接区間の併合 (P522)、プール分離 (P542)、
  `NC_FIRST` / `O_DIRECT` の整列窓 (P530)。**残っていたのは「page cache を
  経由しない」という 1 点だけ**だった (P530 は O_DIRECT を整列窓だけで試し、
  dst とファイル位置が 4096 を法として一致せず 1 バイトも読めなかった)。

### 1.3 P542 の「次の担当者へ」を採らなかった理由 (測らずに台帳で落とした)

- **バウンス 3 枚** (P542 §7-1、予想 -11%): 落ちる。P542 自身のログで
  base (wide 32 MiB) の `ready_s - copy_s` は **0.12-0.14 s** で
  `slotwait_s=0.00`。1.1-1.2 s の隙間は wide=128 MiB の COND **だけ**に出ており、
  その wide=128 は base に負けている (3,826 の中央値 9.46-10.04 対 9.22)。
  上限 1.2%、目標 5% に届かない。
- **`DENNO_PF_UP_DECOUPLE=1` の測り直し** (P542 §7-2): 同上。待ちが
  0.12 s しか無いので上限 1.2%。
- **`DENNO_PF_SPLIT_IO` (br542) の続き** (resident で 2 プールに分ける):
  §2 の機構 (mincore で分ける) は今回 **pread の側**で実装した。分ける先が
  memcpy プールではなく O_DIRECT の読み口になった、というのが今回の差分。

## 2. 変えたこと

1. **`DENNO_PF_PREAD_ODIRECT_BOUNCE`** (新規、**既定 on**):
   `pf_copy_into_pin` で `mincore(2)` の判定 (`range_resident`) を
   「載っていない」について回し、**載っていない区間だけ**
   `read_part_ptr_od_bounce` で読む。読みは
   `od_bounce_shape(off, len, 4096)` が返す整列した `(start, total, delta)` で
   thread-local のバウンス (スレッドごと 1 枚、足りなければ伸ばす) へ O_DIRECT で
   読み、`delta` から `len` バイトを `dst` へ写す。**載っている区間は従来の
   buffered pread** (O_DIRECT にすると page cache のヒットを捨てて全部ディスクになる)。
   失敗 (整列不能 / 短読み / 読み口が無い) は buffered pread へ落ちる。
2. `PARTS_OD` (O_DIRECT の読み口) を `odirect_on() || odb_on()` で開く。
3. 台帳: `[exp-pf-stream]` の末尾に `odb_MB` / `odb_n` / `odb_skip_MB` /
   `odb_fail` / `odb_ms` (発火が無ければ行ごと出ない)。`note_staging_group` も
   odb で回す (grp_miss_s / grp_hit_s が読める。t546b の odb は
   `grp_miss_s=7.74 / 7.30` 対 `copy_s=7.79 / 7.37` = 写しの時間はほぼ全部
   「ミスを含むグループ」)。
4. テスト 4 本: `task546_od_bounce_shape_covers_and_aligns` /
   `task546_odb_gate_truth_table` (lib)、`t546_odirect_bounce_is_wired`
   (kernel_invariants、静的配線)、t538 の同時本数テストを 2 対へ更新。

## 3. lab の結果

### 3.1 t546b (`76cc1e1`、ゲート既定 off、UD-Q4_K_XL、depth8k)

| COND | decode ms/tok | 3,826 cold r1 / r2 | 14k cold (r1) | `copy_s` r1 / r2 | `io_read_MB` r1 / r2 | `odb_MB` |
| --- | ---: | --- | ---: | --- | --- | --- |
| x0 (捨て) | 13.3 | 12.72 / 9.38 | 16.90 | 11.30 / 8.00 | 26.4 / 17.0 | (on) 24.3 |
| b | 13.2 | **10.22 / 9.40** | **13.69** | 8.80 / 8.00 | 19.3 / 16.9 | — |
| **odb** | 13.2 | **9.20 / 8.73** | **12.14** | **7.79 / 7.37** | 19.5 / 17.6 | 18.6 / 16.7 |
| b2 | 13.2 | 22.98 / 9.79 | 14.25 | 21.60 / 8.47 | 17.2 / 16.9 | — |

```text
VERDICT b vs x0 OK decode -1.2% prefill -4.8% md5 DIFF errors 0
VERDICT odb vs x0 OK decode -0.9% prefill -5.6% md5 DIFF errors 0
VERDICT b2 vs x0 OK decode -0.9% prefill -6.5% md5 DIFF errors 0
```

- 3,826 の中央値: b 9,810 / **odb 8,965 (-8.6%)** / b2 9,790 (22.98 は
  page cache が冷えた外れ。同じ日の 1 本目に近い回で、P542 も記録している)。
- **14k は 1 サンプル/COND** (README のとおり depth8k は 14k をラウンド 1 に
  しか流さない): 13,690 -> 12,140 = **-11.3%**。
- 読むバイトは変わらない (`io_read_MB` 19.3 対 19.5)。**速くなったのは
  ディスクの実効**: 19.3/8.80 = 2.19 -> 19.5/7.79 = **2.50 GB/s**。

### 3.2 t546c (`2db7769`、既定 on、UD-Q4_K_XL、depth8k)

| COND | decode ms/tok | 3,826 cold r1 / r2 | 14k cold (r1) | `copy_s` r1 / r2 | `odb_MB` r1 / r2 |
| --- | ---: | --- | ---: | --- | --- |
| x0 (捨て) | 13.2 | 10.46 / 8.57 | 13.48 | 8.96 / 7.19 | 24.3 / 17.4 |
| b (既定 on) | 13.4 | **9.54 / 8.67** | **12.17** | 8.08 / 7.31 | 20.6 / 16.9 |
| off (明示 0) | 13.2 | 9.66 / 10.82 | 13.54 | 8.28 / 9.44 | — |
| b2 (既定 on) | 13.1 | **9.28 / 8.99** | **12.30** | 7.90 / 7.64 | 19.4 / 17.9 |

```text
VERDICT b vs x0 OK decode +1.2% prefill -0.8% md5 DIFF errors 0
VERDICT off vs x0 OK decode +0.0% prefill -7.8% md5 DIFF errors 0
VERDICT b2 vs x0 OK decode -0.9% prefill -7.0% md5 DIFF errors 0
```

- 3,826 の中央値: **on 9,140** (9.54/8.67/9.28/8.99) 対 **off 10,240**
  (9.66/10.82) = **-10.8%**。
- 14k: on 12.17/12.30 対 off 13.54 = **-9.6%**。
- **既定 on で発火している** (`odb_MB` が 4 COND 中 4 本に出て `odb_fail=0`、
  `off` は行ごと出ない)。env を立てない既定経路でも `[pf-pread] O_DIRECT の
  読み口: 4 パート` が出る。

### 3.3 t546d (`2db7769`、既定モデル flash-next-UD-IQ3_XXS、depth)

| COND | decode ms/tok | 3,826 cold | md5 | `odb_MB` |
| --- | ---: | ---: | --- | --- |
| x0 (捨て) | 32.6 | 30.30 | edaa84ad / 1938059f / b42d79d2 / 2081cdeb | — |
| b (既定 on) | 32.5 | 30.18 | **全部 SAME** | 出ない |
| off | 32.4 | 30.04 | **全部 SAME** | 出ない |
| b2 (既定 on) | 32.5 | 29.86 | **全部 SAME** | 出ない |

```text
VERDICT b vs x0 OK decode -0.2% prefill -1.2% md5 SAME errors 0
VERDICT off vs x0 OK decode -0.6% prefill -1.2% md5 SAME errors 0
VERDICT b2 vs x0 OK decode -0.4% prefill -0.6% md5 SAME errors 0
```

- **`odb_MB` は 4 COND とも 1 行も出ない** = 既定モデルは staging で pread を
  使わない (`src` を持たない) ので、このゲートは構造的に不活性。
  `[pf-pread]` の行だけは出る (読み口は開く) が、読む経路に入らない。
- **絶対値の注意**: この回は decode 32.5 ms/tok / 3,826 cold 30 s で、
  履歴 (10.4 / 2.1 s) と桁が違う。4 COND が全部同値・md5 全 SAME なので
  「このゲートで変わっていない」ことは言えるが、**この日のホストが遅かった理由は
  分けていない** (推定: 他の仕事の同居。`load_before=2.17` なので load ではない)。

## 4. 出力テキスト (健全性)

- UD の 3,826 (t546c の b、既定 on):

```text
ご提示いただいた文書は、LLM推論エンジン（`denno-engine）における
**Prefill（事前計算）フェーズの性能最適化の歴史を詳細に記録したものです。
以下に、文書の構造と主要な技術的詳細を、論理的な流れに沿っ...
```

- 14k (t546c の b) も同趣旨で、`off` / `b2` と同文。`errors 0` は
  4 リクエスト全部・全ラウンド。
- md5: UD の長い 2 本 (`c2000_cold` 73510e1f / `c8000_cold` 47c833e2) は
  **t546b/t546c の全 COND で SAME**。短い `b` / `c2000_turn2` の DIFF は
  既存の揺れ (t542 も同じ。P490/P494/P526/P534 が記録)。

## 5. 数値の扱い

- すべて GPU ラボの実測。このコンテナは GPU 無しなのでマイクロベンチは取っていない。
- 壁は `[q4ledger-pf] close wall` (リクエストの実時間)、`copy_s` / `io_read_MB` は
  `[exp-pf-stream]` / `[exp-ram-probe]` の実測。割り算で出した実効 GB/s は
  本文に「割り算」と書いた。
- run を跨いだ比較はしない。各ラウンドの b / b2 (同じ env) と比べた。
  t546b は b と b2 が -8% 割れた回 (10.22 対 9.40 と 22.98) があるので、
  **odb が b と b2 の両方より速いこと**を採否の根拠にした
  (odb 9.20/8.73 対 b 10.22/9.40 と b2 9.79)。t546c は on/off を交互に 2 回ずつ。
- 14k は 1 サンプル/COND なので、-11.3 / -9.6% は 3,826 の判定の補強としてだけ使う。

## 6. 見積り (どこまで行けるか / 次はどこか)

- **残っている床**: ディスク 17-21 GB を 2.5 GB/s で読む = 7.0-8.4 s。
  t522 の O_DIRECT プローブは 4 MiB x 8 本で 4.6 GB/s だったので、
  **読み幅を 4 MiB にすると +30-50% の余地がある (推定)**。
  今は `DENNO_PF_PREAD_KB` が wide で 1 MiB 既定、区間は 0.93 MB が 38% なので
  1 本の pread は 0.93 MB のまま → 4 MiB にしても本数は増えない。
  **隣接区間の併合 (t522 で棄却) を O_DIRECT 側でだけ**やるのが次の形
  (page cache を汚さないので t522 の「併合で +48%」の理由が消える可能性がある、推定)。
- `odb_ms=70-81 s` は 32 スレッドの積算で、壁ではない (1 本 0.93 MB を
  2.6 MB/ms = 2.6 GB/s で読んでいる計算)。バウンスの memcpy は
  20.7 GB で ~1.0-1.5 s 推定だが、ディスク待ちの裏に入っている
  (`copy_s` が減っているので表に出ていない)。
- **既定モデルには効かない** (`src` を持たない)。IQ 系で staging に pread を
  使うモデルが出たら別途測ること。

## 7. テスト

```text
CARGO_TARGET_DIR=/tmp/t546  cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1400 passed; 0 failed; 103 ignored      (int-auto 65a3bf6 (merge: br565) を merge した後の最終ツリー)
CARGO_TARGET_DIR=/tmp/t546  cargo test --release -p denno-engine --features cuda --test kernel_invariants
  -> 153 passed; 0 failed
CARGO_TARGET_DIR=/tmp/t546nc cargo build --workspace --release                          -> ok
CARGO_TARGET_DIR=/tmp/t546nc cargo test  --workspace --release -- --test-threads=1      -> 失敗 0
```

- 追加したテスト 3 本 (`task546_od_bounce_shape_covers_and_aligns` /
  `task546_odb_gate_truth_table` / `t546_odirect_bounce_is_wired`) と、
  t538 の同時本数テストの更新 (enter/leave が 2 対、buffered pread の
  直前/直後を見る)。
- **緑は発火の証拠ではない**ので lab の台帳で数えた: §0-3 のとおり
  `odb_MB` が 5 COND に出て `odb_fail=0`、`off` と既定モデルは 0。
- `.cu` / `.ptx` は触っていない (`PTX=` 不要)。

## 8. GPU で確認してほしいこと

1. **本番の連続リクエスト**で、`odb_MB` が「3,826 で 17-21 GB」のまま出ること、
   および **2 本目以降の 3,826 cold が今より遅くなっていない**こと。
   O_DIRECT は page cache を汚さないので、2 本目の常駐は**増える方向**に
   働くはず (推定) だが、実測は 3,826 を 2 回連続で流したときだけ分かる
   (この lab の depth8k は 3,826 を 1 回しか流さない)。
2. `DENNO_PF_PREAD_ODIRECT_BOUNCE=0` で t546c の `off` (9.66 / 10.82) に戻ること
   (切り戻しの確認)。
3. 既定モデルで `[exp-pf-stream]` に `odb_MB` が今後も出ないこと (§3.3 で確認済み)。
4. ディスクが 2 倍動く日があるので、`copy_s` だけで採否を読まないこと。
   壁 (`[q4ledger-pf] close wall`) と `io_read_MB` を組で読む。
