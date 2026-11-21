ADOPT: yes t530c

# REPORT P530 — UD-Q4_K_XL の staging copy が半分の速さだった理由は
# **「1 本の pread が 0.93 MB のままで、グループの barrier が最後の 1 本のディスク待ちを待つ」**。
# `DENNO_PF_PREAD_KB` を 4 MiB から **384 KiB** に下げると 3,826 cold が
# **15.9 -> 11.8 s (-26%)**、14k cold が **20.0 -> 16.0 s (-20%)**、`copy_s` は
# 14.9 -> 10.6 s (**2.93 -> 4.12 GB/s**) で、ディスクから読むバイトは 17.0 GB で不変。
# 512 KiB だけは回によって 2 倍以上へ跳ねるので採らない。O_DIRECT (bounce 無し) は
# **1 バイトも対象にならない** (0/43.5 GB) ので閉じた。

branch: **br530** = 開始時の br522 (7eaff58 = int-auto 0a7ed94 + task 522 の 3 本)。
br522 はまだ int-auto に入っていないが、task 522 の台帳 (`rn=`/`gap0=`) を土台に
するので**そのまま土台にした**。さらに **br514 を merge した** (`c189f05`):
task 文が使えと言う `[exp-ram-probe]` の `io_rchar_MB` / `io_read_MB` /
`file_resident` が int-auto に無く、br514 (ADOPT: no で未 merge) にしか無いため。
終端で int-auto を fetch し直したが **0a7ed94 のまま** (merge する物は無し)。

lab 名 (すべて `tree=denno-dev3` / `PROFILE=1` / `ptx=none`) /
baseline COND = タスク文のとおり `DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000
DENNO_EXPERT_GPU_MIN_HIT=0 DENNO_GPU_WEIGHT_MB=12000` + `DENNO_EXPERT_RAM_PROBE=1`
(io_ の欄を読むため。全 COND 共通):

- **t530a** = `e90430f` / `MODEL=flash-next-UD-Q4_K_XL` / `SCENARIO=depth8k` /
  `load_before=2.99` / COND: x0 b **k5 k1 nf** b2
  (k5 = `DENNO_PF_PREAD_KB=512`、k1 = 1024、nf = `DENNO_PF_PREAD_NC_FIRST=1`)。
- **t530b** = `db0c14f` / 同モデル / `load_before=4.17` / COND: x0 b **k5 k3 k2** b2
  (k3 = 384、k2 = 256)。512 の両隣を挟む回。
- **t530c** = `3a6f421` / `MODEL=flash-next-UD-IQ3_XXS` (既定) / `SCENARIO=depth` /
  `load_before=4.27` / COND: x0 b(4096) k2(256) b2(4096)。既定モデルの確認。

lab は 3 リクエスト (上限 3)。

## 0. 結論 (6 行)

1. **採る (`ADOPT: yes t530c`)。** 変えたのは `DENNO_PF_PREAD_KB` の**既定だけ**
   (4096 -> 384 KiB)。3,826 cold **15.9 -> 11.8 s (-26%)**、14k cold
   **20.0 -> 16.0 s (-20%)**、既定モデルは不変 (§3.6)。目標 (5%) を大きく超える。
2. **壁の正体 (Do 2 の (d))**: staging の `copy_s` 15.6 s のうち **15.1 s (97%) が
   「ディスク読みを 1 本でも含むグループ」**で、そうでないグループは 0.3 s だけ。
   グループは barrier (全タスク完了まで H2D を積まない) なので、4 MiB 設定では
   実区間 0.93 MB が**切られず 1 タスク**、1 グループ 46 本のうち 40% がミスで、
   **最後の 1 本のディスク待ちで 31 スレッドが遊ぶ**。
3. **(a) 小さいほど速い** (3,826 cold、t530a/b の実測): 4 MiB 16.0-17.2 /
   1024 KiB 16.0 / 512 KiB 13.0-13.1 (**だが t530b で 12.1 と 30.5 に分裂**) /
   **384 KiB 11.86 / 11.82** / 256 KiB 11.83 / 11.66。384 と 256 は 2 ラウンド
   揃うので 384 (呼び出し回数が少ない方) を既定にした。
4. **バイトは 1 バイトも減っていない** (`io_read_MB` 17,352 -> 17,028、`up_MB`
   43,527 不変、`pread_MB` 43,759 不変)。速くなったのは**並列度**:
   `pread_calls` 46,798 -> 138,255 (3.0 倍)、`copy_s` 14.86 -> 10.58 s、
   ディスクの実効 1.14 -> 1.61 GB/s。
5. **(c) O_DIRECT (bounce 無し) は閉じた**: `od_win_MB=0.0` / `od_win_calls=0` が
   t530b の**全 COND・全ラウンド**。dst (pinned の中の位置) とファイル位置が
   4096 を法として一致する区間が 1 つも無い (設計上そうなる)。
6. **(b) 同時本数は 32 のままで足りた**。`DENNO_PF_PREAD_THREADS` は触っていない
   (48/64 は task 424 で棄却済み)。**ミスを先に並べる `DENNO_PF_PREAD_NC_FIRST` は
   発火したが効かない** (t530a nf: 16.71 / 17.55 対 b 17.14 / 17.24、並べ替えは
   22,388 本動いた)。**計器 `DENNO_PF_NC` は既定 off に戻した** (mincore 45,000 回で
   約 330 ms = `copy_s` の 2-3%)。

## 1. 前提: fresh profile と、なぜこの段を選んだか (Do 1 / Do 2)

t530a の b (UD-Q4_K_XL、`PROFILE=1`、3,826 cold):

- warm 短 decode **14.1 ms/tok** (13.9-14.2)、3,826 cold **17.14 s** (`copy_s` 15.83)、
  14k cold **22.16 s** (`copy_s` 17.76)。P522 の b (15.4-18.4) と同じ帯。
- `[exp-pf-stream]`: `up_MB=43526.9` `copy_s=15.83` (**2.75 GB/s**)、
  `pread_MB=43746.9 pread_calls=46786`、`rn=45033 rMB=41938.2`
  (1 区間 0.93 MB、task 522 の台帳と同じ形)。
- `[exp-ram-probe]`: `io_read_MB=17352.0` = **17.0 GB だけディスク**、残り 26.5 GB は
  page cache。**だから壁はディスク帯域ではなく「43.5 GB を pinned へ写す段」**で、
  task 522 の §6 が「次はここ」と残した段そのもの。バイトを減らす方向は P494
  (1 パスが最小) と P514 (page cache が既に 60% 削っている) で閉じている。
- 前回までに棄却済みで**再試行していない**もの: `DENNO_PF_PREAD_THREADS` 48/64
  (t424)、4 MiB への束ね (t424)、隣接区間の併合 (t522)、ページキャッシュ経由の
  先読み (t470)、`DENNO_EXPERT_VRAM_MB` > 22000、expert の prune 系。

## 2. 変えたこと

### 2.1 計器 (commit 1/3、2/3 — 数値経路は変えない)

- `[exp-pf-stream]` の末尾に task 530 の欄 (`exp_pf::staging_stats`):
  - `nc_judged=` / `nc_miss=` / `nc_miss_MB=` / `nc_unknown=` / `nc_ms=`
    — `mincore(2)` で mmap のページが page cache に載っているかを予測した結果。
  - `grp=` / `grp_miss=` / `grp_miss_s=` / `grp_hit_s=`
    — 1 回の `pf_copy_into_pin` を 1 グループとして、**ミスを含むグループ**と
    含まないグループの時間を別々に積む (Do 2 (d) の判定材料)。
  - `nc_ord_grps=` / `nc_ord_moved=` — 並べ替えゲートが何本動かしたか。
  - `od_win_MB=` / `od_win_calls=` / `od_ok_MB=` / `od_fail=`
    — bounce 無しで O_DIRECT に回せる整列区間の量 (ゲート off でも数える)。
- 計器 `DENNO_PF_NC` (**採否の判断後に既定 off**): 上の予測を回す。

### 2.2 ゲート 2 本 (どちらも既定 off のまま)

- **`DENNO_PF_PREAD_NC_FIRST`**: グループ内のタスクを「page cache に載っていない
  区間が先」に並べ替える。バイト列は変えない (順序だけ)。→ 発火したが効かない。
- **`DENNO_PF_PREAD_ODIRECT`**: 整列した内側だけ `O_DIRECT` の別 fd で読む
  (bounce 無し。head/tail は buffered)。窓は
  `odirect_window(dst, at, len, 4096)` = 「`dst ≡ at (mod 4096)` のときだけ」。
- 追加した純関数 (テスト対象): `odirect_window` / `nc_first_order` /
  `range_resident` (mincore) / ゲートの解釈 3 本。

### 2.3 既定の変更 (採否の commit)

- `DEFAULT_PREAD_KB`: **4096 -> 384**。
- `DEFAULT_NC`: true -> **false** (計器の税 2-3% を残さない)。

## 3. 数値 (すべて GPU ラボの実測)

### 3.1 (a) 1 回の長さ (3,826 cold = `c2000_cold`、14k cold = `c8000_cold`)

| COND (PREAD_KB) | t530a 3,826 r1 / r2 | t530b 3,826 r1 / r2 | 14k (r1) | `copy_s` 3,826 (GB/s) |
| --- | --- | --- | --- | --- |
| 4096 (旧既定) | 17.14 / 17.24 | 16.14 / 15.85 | 22.16 / 19.96 | 15.83-14.86 (2.8-2.9) |
| 1024 | 16.04 / 16.03 | — | 20.56 | 14.74 (3.0) |
| 768 | — | — | — | — |
| 512 | 13.00 / 13.14 | **12.12 / 30.51** | **33.56** | 11.72 / 29.2 |
| **384** | — | **11.86 / 11.82** | **15.96** | **10.58 (4.12)** |
| 256 | — | 11.83 / 11.66 | 15.07 | 10.58 (4.12) |

- baseline の repeat (b2、4096) は t530b で 15.86 / 15.63 = b と一致 (位置の効果なし)。
- **512 KiB だけ分裂する** (t530b の 2 ラウンド目 30.51 s、14k 33.56 s)。
  推定: 512 KiB はカーネルの readahead の 1 窓と同じ長さで、回によって二重読みが
  起きる。384 / 256 は 2 ラウンド揃う。
- 1024 KiB は「0.93 MB を切らない」設定なので 4 MiB とほぼ同じ (実測も -6% 止まり)。

### 3.2 (b) 同時本数とバイト

| COND | pread_calls | pread_MB | io_read_MB | up_MB | copy_s |
| --- | ---: | ---: | ---: | ---: | ---: |
| b (4096) | 46,798 | 43,758.6 | 17,352.0 | 43,526.9 | 14.86 |
| k3 (384) | 138,255 | 43,751.9 | 17,027.6 | 43,515.1 | 10.58 |
| k2 (256) | 181,522 | 43,802.4 | 17,166.0 | 43,581.5 | 10.58 |

- **バイトは不変**。変わったのは同時に飛ぶ本数だけ。ディスクの実効は
  17.0/14.86 = **1.14 GB/s -> 17.0/10.58 = 1.61 GB/s**。
- `DENNO_PF_PREAD_THREADS` は 32 のまま (触っていない)。

### 3.3 (c) O_DIRECT、bounce 無し

- t530b の**全 COND・全ラウンドで `od_win_MB=0.0` `od_win_calls=0`**
  (ゲート off でも数える計器)。dst は pinned の中の連続オフセット、`at` は
  ファイル位置で、**4096 を法として一致しない**ので窓が取れない。
- bounce を作れば使えるが、43.5 GB の memcpy が増えるだけで、今回の勝ち
  (並列度) と両立しない。**この方向は閉じた**。

### 3.4 (d) page cache ヒットとディスク読みは 1 グループの中で直列化するか

t530a b_1 の実測:

| COND | grp | grp_miss (割合) | grp_miss_s | grp_hit_s |
| --- | ---: | ---: | ---: | ---: |
| b (4096) | 5,538 | 4,725 (**85%**) | **15.56** | 0.27 |
| k3 (384) | 5,541 | 4,255 (77%) | **10.15** | 0.43 |
| k2 (256) | 5,528 | 4,324 (78%) | 10.18 | 0.40 |

- **`copy_s` の 97% が「ミスを含むグループ」の中**。ヒットだけのグループは 0.3 s。
  つまり写しの時間は「ディスク待ちのグループ」の時間そのもので、グループは
  barrier なので最後の 1 本が全体を決める。これが半分の速さの正体。
- mincore の予測 (b_1): `nc_judged=44,956` のうち `nc_miss=19,358` (43%)、
  `nc_miss_MB=18,200` / `up_MB=43,527` = **42% がディスク行き**。
  `io_read_MB=17,352` とほぼ一致 (予測は当たっている)。
- **`DENNO_PF_PREAD_NC_FIRST` は発火したが効かない** (t530a nf: `nc_ord_moved=22,388`、
  3,826 cold 16.71 / 17.55 対 b 17.14 / 17.24)。ミスを前に出しても、barrier が
  最後の 1 本を待つ構造は変わらない。**切る数を増やす方 (a) が正しかった**。

### 3.5 decode と md5

- t530a の decode 中央値: b 14.1、k5 13.9、k1 13.9、nf 13.9、x0 14.4 ms/tok。
  `VERDICT k5 vs x0 ... decode -3.5%`。t530b: b 13.9、k3 13.9、k2 13.8。
- md5 は UD ではラウンド間で動く (P518/P522 と同じ既知の性質)。`a:a` /
  `c2000_cold` / `c8000_cold` は**全 COND で SAME**、`errors 0`、
  テキストは全 COND `looks ok` (TXTD: 「ご提示いただいた文書は、LLM推論エンジン
  （`denno-engine）における **Prefill（事前計算）フェーズの性能最適化の歴史と教訓を
  詳細に記録したものです。…」)。

### 3.6 既定モデル (t530c、`MODEL=flash-next-UD-IQ3_XXS`、`SCENARIO=depth`)

| COND | decode ms/tok | 3,826 cold r1 / r2 | md5 |
| --- | ---: | --- | --- |
| x0 (新既定 384) | 10.4 | **5.26** (最初の COND、page cache が冷たい) / 1.89 | edaa84ad / 1938059f / 9b28e018 / 8b02d5c3 |
| b (4096) | 10.4 | 1.88 / 1.89 | 同じ |
| k2 (256) | 10.4 | 1.88 / 1.87 | 同じ |
| b2 (4096) | 10.4 | 1.88 / 1.89 | 同じ |

- **全 COND・全リクエストで md5 SAME**、errors 0、text ok。
  `VERDICT k2 vs x0 OK decode +0.4% prefill +0.0% md5 SAME`。
- 既定モデルは 83.6 GB で page cache に載るので、この段の変更は素通りする
  (P522 の t522d と同じ)。b2 (4096) と x0 (384) の差は 1.88 対 1.89 s = ノイズ。
- ここだけ `DENNO_PF_NC` が既定 off のビルドで回っている (t530a/b は on で、
  330 ms の税込み)。それでも UD の順位は変わらない。

### 3.7 lab の VERDICT 行 (抜粋)

t530a (`e90430f`): `VERDICT b vs x0 OK decode -2.2% prefill -9.2% md5 DIFF errors 0` /
`VERDICT k5 vs x0 OK decode -3.5% prefill -12.3% md5 DIFF errors 0` /
`VERDICT k1 vs x0 OK decode -3.5% prefill -9.2% md5 DIFF errors 0` /
`VERDICT nf vs x0 OK decode -3.3% prefill -10.8% md5 DIFF errors 0` /
`VERDICT b2 vs x0 OK decode -1.9% prefill -10.0% md5 DIFF errors 0` /
`VERDICT_DEPTH k5 c2000_cold vs x0 decode -50.6% prefill -30.5%` /
`VERDICT_DEPTH nf c2000_cold vs x0 decode -52.6% prefill -8.9%` /
`VERDICT baseline x0 errors 0 text ok`。

t530b (`db0c14f`): `VERDICT b vs x0 OK decode -51.8% prefill -30.6% md5 DIFF errors 0` /
`VERDICT k5 vs x0 ... -52.4% / -31.2%` / `VERDICT k3 vs x0 ... -52.0% / -31.2%` /
`VERDICT k2 vs x0 ... -52.1% / -31.8%` / `VERDICT b2 vs x0 ... -52.0% / -31.8%` /
`VERDICT_DEPTH k3 c2000_cold vs x0 decode -11.7% prefill -36.9%` /
`VERDICT_DEPTH k3 c8000_cold vs x0 decode -25.0% prefill -36.7%` /
`VERDICT_DEPTH k2 c2000_cold vs x0 decode -12.4% prefill -37.4%` /
`VERDICT_DEPTH k2 c8000_cold vs x0 decode -46.8% prefill -40.2%` /
`VERDICT_DEPTH b c2000_cold vs x0 decode -10.2% prefill -14.7%` /
`VERDICT_DEPTH b2 c2000_cold vs x0 decode -12.8% prefill -16.0%` /
`VERDICT_DEPTH k5 c8000_cold vs x0 decode -34.1% prefill +33.1%` (512 の分裂) /
`VERDICT baseline x0 errors 0 text ok`。
(x0 は最初の COND で page cache が冷たいので、判定は b と b2 の間で読む。)

t530c (`3a6f421`、既定モデル): `VERDICT b vs x0 OK decode +0.8% prefill +0.0% md5 SAME errors 0` /
`VERDICT k2 vs x0 OK decode +0.4% prefill +0.0% md5 SAME errors 0` /
`VERDICT b2 vs x0 OK decode +0.4% prefill +0.9% md5 SAME errors 0` /
`VERDICT baseline x0 errors 0 text ok`。

## 4. 判定

- **`ADOPT: yes t530c`**。3,826 cold -26% / 14k cold -20% は目標 (5%) を大きく超え、
  decode も悪くない (-3.5%)、既定モデルも悪くない (§3.6)。運用規則 (2 ラウンド
  2% 以上) を満たす。
- 採るのは **`DEFAULT_PREAD_KB = 384`** の 1 定数だけ。ゲート 2 本は既定 off で
  残す (未設定なら従来と 1 命令も変わらない)。計器は既定 off。
- 512 KiB と O_DIRECT は採らない (§3.1、§3.3)。

## 5. テスト

```
CARGO_TARGET_DIR=/tmp/t530 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1368 passed; 0 failed; 103 ignored
CARGO_TARGET_DIR=/tmp/t530 cargo test --release -p denno-engine --features cuda --test kernel_invariants
  -> 147 passed; 0 failed
CARGO_TARGET_DIR=/tmp/t530nc cargo build --workspace --release -> ok
CARGO_TARGET_DIR=/tmp/t530nc cargo test --workspace --release -- --test-threads=1 -> ok
```

新規テスト:
- `task530_odirect_window_truth_table` (整列の位相が同じときだけ窓が取れる / 端数 / 0 長)
- `task530_nc_first_order_is_a_stable_partition`
- `task530_gates_and_staging_stats` (既定 off、台帳の書式と reset)
- `task530_range_resident_sees_touched_memory` (mincore が自分のヒープを常駐と答える)
- `pread_kb_rounds_and_defaults` に `DEFAULT_PREAD_KB == 384` を追加
- `kernel_invariants::t530_staging_copy_instruments_and_gates_are_wired`
  (予測 → 並べ替え → 写し → グループ記録の順、O_DIRECT が buffered へ戻れる位置、
  出口 2 つに記録があること)

**緑は発火の証拠ではない**ので、lab の台帳で数えた: `nc_ord_moved=22,388` (nf)、
`pread_calls` 46,798 -> 138,255 (384 で切れている)、`od_win_calls=0` (O_DIRECT は
発火しない = 閉じた)、`nc_ms≈330` (計器の費用)。

## 6. 次の担当者へ

1. **残っている段は「ディスクの実効 1.6 GB/s」そのもの**。並列度を上げて
   1.14 -> 1.61 GB/s まで来たが、素のディスクは 3.4-4.6 GB/s (t522) ある。
   次は「グループの barrier を外す」= ミスのグループだけ H2D を後ろへ回す、
   2 グループ先まで写し終えてから積む、など。`grp=` の計器がそのまま使える
   (`DENNO_PF_NC=1` で戻る)。
2. 384 と 256 はほぼ同じ (差は 1-2%)。256 にすると呼び出しが 18 万回になるので
   既定は 384 のままが無難。
3. `DENNO_PF_PREAD_NC_FIRST` と `DENNO_PF_PREAD_ODIRECT` は既定 off で残した。
   前者は再挑戦の価値なし (barrier が原因)、後者は bounce 無しでは 0 バイト。
4. `SCENARIO=depth8k` の 14k はラウンド 1 にしか流れない。1 サンプル/COND は
   判定に使わない (今回は 3 リクエストとも 14k は r1 のみ)。

## 7. GPU で確認してほしいこと

1. 既定モデル (t530c) で 384 KiB が 4096 KiB より悪くないこと (§3.6)。
2. `DENNO_PF_PREAD_KB=512` の分裂 (12.1 s と 30.5 s) の再現性。再現するなら
   「readahead の窓と同じ長さ」の推定が確かめられる。
3. ディスクの日差が大きい (P494 の日 3.0 GB/s、P522 の日 3.8-4.7 GB/s)。
   3,826/14k の 1 サンプル/COND は判定に使わない。
