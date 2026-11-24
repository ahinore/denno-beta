ADOPT: yes t594b t594c

UD-Q4_K_XL の **decode の experts 段**で、seg 層の prologue (約 10.8 KB) の
H2D を pageable の同期 `cuMemcpyHtoD` から pinned + 同一ストリームへの非同期 DMA に変えた
(`DENNO_SEG_H2D_PIN`、既定 on)。experts 段は **4 ラウンド全部で -4.5% 〜 -8.7%** (4.26/4.02 ->
3.89/3.84、4.10/4.14 -> 3.75/3.82 ms/token)、短い decode の壁は t594b で **-3.4% / -2.8%**、
t594c で **-1.7% / -1.8%**。他の段は動かず、既定モデルは分岐が 1 回も発火せず (calls=0)
数値も md5 も完全に同一。**タスクの目標 (3%) は t594b では満たしたが t594c では届かず、
2 run の平均は -2.5%**。採否はオペレーター規則 (段で 2 ラウンド 2% 以上 + 他が 1% 超
悪化しない + 本文健全) で決めた。

# REPORT P594 — UD-Q4_K_XL: decode の seg 層 prologue H2D を pinned + 非同期 DMA に

branch **`br594`** = 開始時の `int-auto` **`42192e3`** (`merge: br668`)。作業中も終端でも
int-auto は `42192e3` のまま (fetch し直して確認。merge する物は無い)。push 無し、
docker 無し、GPU 直叩き無し。`.cu` / `.ptx` は触っていない (新カーネル無し)。

lab 名と SHA:
- **t594a** = `42192e3` (段プロファイル)。MODEL=flash-next-UD-Q4_K_XL、SCENARIO=depth8k、
  PROFILE=1、COND `x0 b k s b2` (`k` = `DENNO_GDN_DEC_KPROF=1`、`s` = `DENNO_SEG_PROF=1`)。
  `load_before=1.45`、rc=0。
- **t594b** = `11db77a` (ゲート既定 off の A/B)。COND `x0 b p ps b2`
  (`p` = `DENNO_SEG_H2D_PIN=1`、`ps` = `+DENNO_SEG_PROF=1`)。`load_before=3.86`、rc=0。
- **t594c** = `8ada7cb` (既定 on の確認と切り戻しの対照)。COND `x0 d off d2`
  (`d`/`d2` = 既定 on、`off` = `DENNO_SEG_H2D_PIN=0`)。`load_before=1.89`、rc=0。
- **t594d** = `8ada7cb` (既定モデル、**Do 3 の 1 ラウンド**)。MODEL=flash-next-UD-IQ3_XXS、
  SCENARIO=depth、COND `x0 d off d2`。`load_before=18.43`、rc=0。

## 0. 結論 (6 行)

1. **採る (`ADOPT: yes`)。** `DENNO_SEG_H2D_PIN` を既定 on にした (切り戻し `=0`)。
2. **段の勝ちは 4 ラウンド全部で再現**: experts 段が t594b `4.10 / 4.14 -> 3.75 / 3.82`
   (-8.5% / -7.7%)、t594c で `4.26 / 4.02 -> 3.89 / 3.84` (d、-8.7% / -4.5%) と
   `3.91 / 3.79` (d2、-8.2% / -5.7%)。同じ run の中の対で、他段は動いていない
   (mix 2.15 / ssm 2.70-2.74 / full 0.94-0.95 / shared 0.48-0.50 / head 0.46)。
3. **短い decode の壁**: t594b `22.53 / 22.67 -> 21.77 / 22.03` (**-3.4% / -2.8%**、
   `[q4ledger] tok=128 total wall=` の同型リクエスト)、t594c `22.81 / 22.40 ->
   22.42 / 21.99` (**-1.7% / -1.8%**、`off` 対 `d`)。**2 run の平均 -2.5%**。
   タスク目標の 3% は t594b で満たし t594c では届かない (run 間の差は §4)。
4. **既定モデル (UD-IQ3_XXS) は構造的に不変**。`[seg-h2d] calls=0` = seg 層の prologue 経路を
   1 回も通っていない (既定モデルの expert は IQ 型で `[exp-iq]` 経路)。decode 9.8 ms/tok、
   prefill 0.57 s が `d` / `off` / `d2` で同一、md5 は 4 COND x 4 文脈すべて SAME。
5. **本文は健全・errors 0・md5 SAME** (UD の t594b/c は全 6 文脈 SAME。表の `b:` の
   1 本だけ既存の非決定でぶれる。P646 §6 と同じ現象で、この変更のせいではない)。
6. **prefill の壁 (3,826 / 14k cold) はこの run では判定できない** (§4)。変更は prefill の
   ステージング経路に触らないが、lab のディスクが同じ run の中で 1.5〜3 倍動いた
   (t594c の x0 が 3,826 で 9.6 s -> 18.1 s、14k が 9.9〜12.8 s)。**悪化した証拠は無い**。

## 1. なぜ seg 層 prologue の H2D を選んだか (fresh な段プロファイル)

### 1.1 段 (t594a の b、UD-Q4_K_XL、`[q4ledger-pf]`)

| 段 | 3,826 cold wall 7,387.8 ms | share | 14k cold wall 10,984.2 ms | share |
| --- | ---: | ---: | ---: | ---: |
| **experts** | **6,476.2** | **87.7%** | **7,729.6** | **70.4%** |
| ssm | 366.5 | 5.0% | 1,756.4 | 16.0% |
| mix | 284.3 | 3.8% | 49.5 | 0.5% |
| other | 219.0 | 3.0% | 812.5 | 7.4% |
| full | 41.9 | 0.6% | 636.1 | 5.8% |

3,826 の experts は `[pf-host] up_wait=6,137.9ms` = **ステージングの写し待ち (ディスク)** で、
`[exp-pf-stream]` の O_DIRECT 実読み 18,388.6 MB / 23,691 本 = 0.776 MB/本 を `ready_s=5.76 s`
で流して **3.19 GB/s**。t522 のプローブ (O_DIRECT 1 MiB x 8 本) の天井 3.5 GB/s の **91%**。
P550 / P566 / P570 がこの段を 3 方向 (読みの大きさ・併合・グループ数) で測っており、
**残りは 9% で -5% は取れない** (P566 §2.2)。だから prefill の壁は今回追わない。

14k の ssm 1,756 ms は GPU busy 1,313 ms で、P577 が特定した scan の直列和 = **`.cu` が要る**
(このタスクは `.cu` 禁止)。mix / other / full は単独で 7% 以下。

### 1.2 decode の段 (t594a の b、短い decode 11.4 ms/tok、`[q4ledger]`)

| 段 | ms/token | share |
| --- | ---: | ---: |
| experts | 4.87 | 41% |
| ssm | 2.70 | 23% |
| mix | 2.18 | 18% |
| full | 0.94 | 8% |
| shared | 0.50 | 4% |
| head | 0.46 | 4% |
| router | 0.18 | 2% |

experts の中身 (`[seg-prof]`、温まった短い decode、48 層/token):

| 区間 | us/層 | 48 層で |
| --- | ---: | ---: |
| **h2d (prologue)** | **8.0** | **0.38 ms** |
| launch (3 起動) | 4.4 | 0.21 ms |
| lookup + table | 1.6 | 0.08 ms |
| sync (GPU 待ち) | 20 | 0.96 ms |
| d2h (和 10 KB) | 7.2 | 0.35 ms |
| miss (CPU GEMV、GPU と重ね打ち) | 65 | 3.1 ms |
| GPU (gate 21.7 + up 1.4 + silu 0.4 + down 14.4 + sum 3.7) | 41.6 | 2.0 ms |

### 1.3 既に測られていたもの / 選んだ理由

- **miss (3.1 ms = 25%)**: P646 が lab t638a の選択ログを再生し、**入り口 (admission) の変更で
  取れる上界は Belady で +3.4 pt = decode の 2.8% (推定)** と出して棄却済み。ミス 1 本は
  57 us で、これは expert 重み ~2.9 MB を RAM から読む時間 (メモリ律速) なので、
  カーネル側でも削れない。
- **sync / d2h**: P526 が `DENNO_SEG_SUM_PIN` と `DENNO_SEG_SUM_ASYNC` を 2 ラウンド測り、
  「d2h は縮むが壁に出ない」で既定 off のまま。**再試行しない**。
- **ssm (2.70 ms)**: P592 が scan の第 3 版 (`DENNO_GDN_SCAN3`) を既定 on にした後。
  残りは GPU 時間 (scan_sig3 11.7 + dn8 35.9 + dn7 14.0 + matvec_f32_r 8.7 us/層) で
  **`.cu` が要る**。
- **h2d 8.0 us/層だけが「まだ誰も触っていない大きいホスト区間」**だった。P546 は
  「h2d 10-14 us/層」と数字だけ残して中身を追っておらず、P526 の 2 本は D2H 側の話。
  pageable の同期 `cuMemcpyHtoD` が 10.8 KB で 1.3 GB/s しか出ないこと、その待ちが
  **ホストのミス計算の前に直列に挟まっている**ことが t594a の `[seg-prof]` で見えた。
  `DENNO_SEG_H2D_PIN` という新しい理由 (pinned + 同一ストリームの DMA) がある。

## 2. 変えたこと

- `crates/denno-engine/src/model/seg_prof.rs`: ゲート `DENNO_SEG_H2D_PIN` (**既定 on**、
  `=0` で切り戻し)、台帳 (calls / fallbacks)、リクエスト終端の `[seg-h2d]` 行、テスト 2 本。
- `crates/denno-engine/src/cuda/moe_pf.rs`: seg 層の prologue を **pinned バウンスへ直接組み**
  (同じ `put_u64s` / `put_u32s` / `put_f32s` を pinned の `u8` スライスに回す)、
  `cuMemcpyHtoDAsync` を **カーネルと同じ null ストリーム**へ出す。送るバイト数・内容・
  宛先は従来経路と同一。pinned は thread-local の grow-only 1 本で、次の層がここへ書くのは
  和カーネルの後のイベントを待った後なので、前の DMA は完了している。
  確保失敗・DMA 失敗・ゲート off は従来の pageable 同期経路へ落ちる (fail closed)。
- `crates/denno-cli/src/serve_flash.rs`: リクエスト終端に `[seg-h2d] calls= fallbacks=` を出す
  (分岐が発火した証拠。0 なら採否に使わない)。

数値経路は 1 命令も変えていない (バイト列が同じ)。丸め順も不変。

## 3. 実測 (すべて GPU ラボ)

### 3.1 t594b (`11db77a`、ゲートは既定 off、`p` で明示 on)

同じ run、同じリクエスト列、hit 率も同一 (`0.962` / uploads `464`、b_1 と p_1 が完全一致)。

| 指標 | b (r1 / r2) | p (`DENNO_SEG_H2D_PIN=1`) | 差 |
| --- | ---: | ---: | ---: |
| short decode `tok=128 total wall` | 22.53 / 22.67 | **21.77 / 22.03** | **-3.4% / -2.8%** |
| experts 段 ms/token | 4.10 / 4.14 | **3.75 / 3.82** | **-8.5% / -7.7%** |
| mix / ssm / full / shared / head | 2.15 / 2.74 / 0.95 / 0.49 / 0.46 | 2.15 / 2.71 / 0.95 / 0.48 / 0.46 | 不変 |
| `[seg-prof] h2d` us/層 (ps) | 8.0 | **0.0** (table 0.5 -> 2.2) | -8 us 純減 |
| md5 / errors / text | — | 全 6 文脈 SAME | 0 / looks ok |

`[seg-h2d] calls=6050 fallbacks=0`。

### 3.2 t594c (`8ada7cb`、既定 on の確認と `=0` の対照)

| 指標 | off (`=0`) r1 / r2 | d / d2 (既定 on) | 差 |
| --- | ---: | ---: | ---: |
| short decode `tok=128 total wall` | 22.81 / 22.40 | d 22.42 / 21.99、d2 22.23 / 21.89 | **-1.7% / -1.8%** (d)、-2.5% / -2.3% (d2) |
| experts 段 ms/token | 4.26 / 4.02 | d 3.89 / 3.84、d2 3.91 / 3.79 | **-8.7% / -4.5%**、-8.2% / -5.7% |

`[seg-h2d] calls=42448 fallbacks=0` (d_1)、`fallbacks` は 1 回も出ていない。

### 3.3 t594d (既定モデル UD-IQ3_XXS、depth、Do 3 の 1 ラウンド)

| COND | decode ms/tok | prefill s | md5 |
| --- | ---: | ---: | --- |
| x0 | 9.9 | 0.57 | 基準 |
| d (既定 on) | 9.8 | 0.57 | **全 SAME** |
| off | 9.8 | 0.57 | **全 SAME** |
| d2 | 9.8 | 0.57 | **全 SAME** |

`[seg-h2d] calls=0` (4 COND とも) = **seg 経路に入っていない**ので構造的に不変。
`VERDICT d vs x0 OK decode -0.8% prefill +0.9% md5 SAME errors 0`。

## 4. 数値の扱い

- 段・decode・prefill の値はすべて lab の実測。**同じ run の中の対** (`b` vs `p`、
  `off` vs `d`) でだけ比べた。run をまたいだ絶対値は比べていない。
- short decode の -3.4% と -1.7% の差は **計測点のばらつき**。`total wall` は
  同一リクエスト・同一トークン列の和で、hit 率も同一なので、差は h2d の 8 us/層
  (0.38 ms/token = 3.1%) のうち実際に壁へ出た分が run で違うため。t594c は
  全体にノイズが大きく (`off` の 3 番目のリクエストが 61.98 ms、x0 の 3,826 が
  18.1 s)、experts 段の -4.5% 側の丸めが出ている。
- **prefill (3,826 / 14k) はこの 2 run では判定できない**。t594c の 14k cold は
  `off` 9.93 s / `d` 12.78 s / `d2` 10.47 s、t594b は `p` 20.39 s (既知の
  ディスク 2 倍化、P566 §5)。x0 の 3,826 が 9.6 s -> 18.1 s と動いた run なので、
  14k の +29% は変更の効果ではないと判断する。**変更は prefill のステージング
  (batch 経路) に触らず、`[seg-h2d]` の増分も decode の 48 層/token と一致**している。
- 目標判定: **3% は t594b で満たし (-3.4% / -2.8%)、t594c では届かない (-1.7%)。
  2 run の中央値で -2.5%。** 採否はオペレーター規則 (段で 2 ラウンド 2% 以上 +
  他が 1% 超悪化しない + 本文健全) に当てはまるので採った。目標未達を明記する。

## 5. テスト

```text
CARGO_TARGET_DIR=/tmp/t594   cargo test  --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1432 passed; 0 failed; 103 ignored      (int-auto は 1430。+2 は今回のゲートのテスト)
CARGO_TARGET_DIR=/tmp/t594   cargo test  --release -p denno-engine --features cuda --test kernel_invariants -- --test-threads=1
  -> 155 passed; 0 failed
CARGO_TARGET_DIR=/tmp/t594nc cargo build --workspace --release           -> ok
CARGO_TARGET_DIR=/tmp/t594nc cargo test  --workspace --release -- --test-threads=1
  -> 1559 passed; 0 failed
```

既知の赤 (`qwen4exp_decode spec_batch_stops_at_eos`、`qwen4exp_params` の validate 2 件) は
本タスクでも同じ (触っていない)。追加した 2 本は「緑の証拠」ではなく規則の固定:
`t594_seg_h2d_tests::seg_h2d_pin_truth_table` (既定 on、明示的な偽値だけ off) と
`seg_h2d_counters_accumulate` (台帳が呼んだ分だけ増える)。
**分岐の発火は lab の `[seg-h2d] calls=6050 / 42448 / 0` で数えた** (既定モデルは 0)。

## 6. GPU で確認してほしいこと

1. 既定 on の最終木 (`8ada7cb`) で、UD-Q4_K_XL の短い decode が -2% 前後で再現するか
   (t594c は -1.7%、t594b は -3.4%)。**同じ run に `off` を置いて対で読む**。
2. 3,826 / 14k cold の prefill が悪化していないこと。この run ではディスクのドリフトで
   読めなかった。`[seg-h2d]` の増分が decode の 48 層/token 相当に留まることも見てほしい。
3. 次の一手: decode の残りは experts の miss CPU GEMV (3.1 ms = 25%、メモリ律速) と
   ssm の GPU カーネル (scan_sig3 / dn8、`.cu` が要る)。**`.cu` の許可が要る**。
