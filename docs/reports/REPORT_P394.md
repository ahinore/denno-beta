ADOPT: yes t394a t394c

# REPORT P394 — UD-Q4_K_XL: 終端の戻しを N 本で重ねる / 次の層を先に読む

branch: **br394** = 開始時の int-auto `04ab8eb` から。**br390 は merge していない**
(br390 の中身は `DENNO_PF_RESTORE_PAR` = 戻すページの先読みで `ADOPT: no`。
ページは既に page cache にあったので効かず、既定 off の死んだゲートだけ。
今回の対象は同じ場所だが、先読みではなく「同期転送を N 本で重ねる」方向なので
br390 の分岐は要らない)。作業中に int-auto は動かず (`04ab8eb` のまま)、
終端の fetch でも変わらなかったので merge は無い。

数値はすべて GPU ラボの実測 (`SCENARIO=depth8k`、`PROFILE=1`、`ptx=none`)。
lab 名と SHA:
**t394a = `18c7de8`** (UD-Q4_K_XL、5 COND: x0 / b1 / r8 / r16 / b2)、
**t394b = `e09e1af`** (UD-Q4_K_XL、5 COND: x0 / b1 / lp / lpm / b2)、
**t394c = `e09e1af`** (既定モデル flash-next-UD-IQ3_XXS、4 COND: x0 / b1 / mt / b2)。

## 0. 結論

1. **採る 1 本**: `DENNO_PF_RESTORE_MT` (層優先 prefill の終端で戻す expert を
   N 本のスレッドで重ねる)。戻しの段は 3,826 で **70/72 → 55/53 ms**
   (2 ラウンドとも -21〜-25%)、14k で **549 → 213 ms**、既定モデルの 14k で
   **121 → 71 ms**。**既定 on** にした (切り戻しは `DENNO_PF_RESTORE_MT=0`)。
2. **落とす 1 本**: `DENNO_PF_LAYER_PRE` (次の層の expert 全体を先に読む)。
   発火はした (48 層 / 77,018 MB / 144 回) が、3,826 の round 2 で
   b1 copy_s 7.36 / lp 7.25 / b2 7.20 と**区別できない**。層の全数は 512 本で
   routing は 296 本 (58%) しか使わないため、全数読みは 1.73 倍の無駄読みになる。
   既定 off のまま残した (A/B 用)。
3. **task の目標 (UD prefill -5%) は満たしていない。** 3,826 cold の壁は
   9.1-9.7 s で、床は **44.4 GB の pread = 7.11 s (6.24 GB/s)**。これは P325 が
   測ったこのデバイスの上限 6.2 GB/s と一致する。14k の床は `gpu_busy` 21.53 s。
   採ったのはオペレーター規則 (段で 2% 以上 x 2 ラウンド、他は 1% 以上悪化しない)
   で、壁への寄与は -0.2% (3,826) / -1.3〜-2.1% (14k)。
4. **既定モデルは無傷** (t394c): 3,826 2.42-2.44 s、14k 6.03-6.51 s、
   decode 15.5-15.6 ms/tok で 4 COND 同一、**md5 は全リクエストで SAME**。

## 1. 前提: 段の読み直し (Do 1 / Do 2)

新しい実測を取る前に、直近の実測 (lab t390b = SHA 0c9fdea、REPORT_P390) の
生ログを段ごとに開き直した (`/work/gpu_lab/done/t390b/b1_1.log`)。int-auto は
t390b のあと br388 (decode のミス CPU を起動と重ねる) と br393 の merge だけが
入っていて、どちらも prefill のホスト経路を触っていない
(`git diff --stat b7f3e55..04ab8eb` は `cuda.rs` と `exp_iq.rs` の decode 側だけ)。

| 段 (UD) | 3,826 cold | 14k cold | 中身 |
| --- | ---: | ---: | --- |
| 壁 | 9.13-12.02 s | 25.4-29.9 s | — |
| `gpu_busy` | 5,527.9-5,530.4 ms | 21,530.9-21,593.4 ms | 全 COND 同一 |
| experts | 7.39-9.26 s | 19.06-21.48 s | `up_wait` = 44.4 GB / 98.9 GB の pread |
| other | 529-728 ms | 2,015-4,008 ms | ple / embed / res_up / **戻し** |
| ssm | 465-468 ms | 3,272-3,457 ms | 露出はキュー待ち (P393 §6) |
| full | 143-147 ms | 1,106-1,394 ms | KV ミラー |
| `[pf-pool]` の戻し | **289 本 / 854 MiB** | **1,057 本 / 3,310 MiB** | 終端で LRU へ戻す |

既定モデルの段 (t394c、mt=1 の回): 3,826 は wall 2,440.3 ms
(mix 292.2 / ssm 457.4 / experts 1,327.2 / other 234.4 / full 129.0)、
14k は wall 6,429.2 ms (ssm 2,457.4 / experts 1,911.3 / other 1,101.8 /
full 937.2 / mix 21.4)。`[pf-restore]` は 14k で 827 本 / 1,636 MiB。

### なぜこれを取ったか

- 3,826 の壁は **44.4 GB の pread** (expert の全重み 1 回ぶん)。デバイスの
  実測上限 6.2 GB/s (P325) に対し既に 5.1-6.3 GB/s で回っている = 読む速さでは
  もう縮まない。
- 14k の壁は `gpu_busy` 21.53 s (GPU 74%)。`.cu` は本 task では触れない。
- 残るホスト仕事は 2 つ: **終端の戻し** (P390 §9.1 が名指しした未着手) と、
  **attention のあいだ遊ぶディスク** (層優先は routing が決まるまで読めない)。

## 2. 変えたこと (2 本、どちらも env ゲート 1 本の内側)

### 2.1 `DENNO_PF_RESTORE_MT` (既定 on) + `DENNO_PF_RESTORE_THREADS` (既定 8)

層優先 prefill の終端で追い出した expert を戻す `pf_restore_experts` を、
専用 rayon プールの N 本で回す。1 本ぶんは
`expert_vram::acquire_deferred` (転送を LRU のロックの外で行う版。task 214 が
背景の上げ直しで使っている)。プールの各スレッドで 1 回だけ
`make_current_on_this_thread` と `enable_thread_upload_stream` を呼ぶ
(CUDA の current context はスレッドローカル。忘れると最初の確保が error 201 で
落ちて 1 本も載らない = task 146 / task 214 の穴)。張れなければ直列へ落ちる。
VRAM へ渡るバイト列・LRU の内容は 1 本ずつの経路と同一 (順序と重なりだけ変わる)。

### 2.2 `DENNO_PF_LAYER_PRE` (既定 off)

層ループの先頭で**次の層の gate/up/down 全体**を Linux の `readahead(2)` で
ページキャッシュへ積む。範囲は `ExpertW::slice_abs` (= mmap 経路の `slice` と
同じ `ExpertSlice`) から作るので、あとで `pread` が読む範囲と必ず一致する。
ユーザー側の写しも VRAM もステージングも増やさない。

### 2.3 台帳

- `[pf-restore] n 本 / MiB を戻した / (戻せず) / ms / mt= / threads=`
- `[exp-pf-stream]` の末尾に `lpre_layers= / lpre_MB= / lpre_calls=`

## 3. lab の結果

### 3.1 t394a (UD-Q4_K_XL、x0 / b1 / r8 (MT 8 本) / r16 (MT 16 本) / b2)

`load_before=1.87`。14k は round 1 のみ、3,826 は 2 ラウンドの中央値。

| cond | 3,826 cold | 14k cold | 短い decode | 戻し 3,826 | 戻し 14k |
| --- | ---: | ---: | ---: | ---: | ---: |
| x0 (捨て) | 10.77 | 27.97 | 19.1 | 70 / 77 ms | 1,944 ms |
| b1 | 9.48 | 25.84 | 18.5 | 70 / 72 ms | 549 ms |
| **r8** | **9.43** | **25.31** | 18.3 | **55 / 53 ms** | **213 ms** |
| **r16** | **9.46** | **24.82** | 18.4 | **59 / 57 ms** | **177 ms** |
| b2 | 9.66 | 25.13 | 18.3 | 89 / 70 ms | 267 ms |

- **戻しの段は r8 / r16 の 2 ラウンドとも速い** (3,826 で -21〜-25%)。
  14k も 1 サンプルで -61%。オペレーター規則 (段で 2% 以上 x 2 ラウンド、
  他は 1% 以上悪化しない) を満たす。
- 壁は r8 vs b1 で 3,826 -0.5% / 14k -2.1%、r8 vs b2 で 3,826 -2.4% /
  14k +0.7%。**位置効果 (±2-3%) の中**で、壁としては読めない。
- `[exp-pf-stream] copy_s` 3,826: b1 7.83 / 7.11、r8 7.65 / 7.14、b2 8.22 / 7.12。
  最良 7.11 s = 6.24 GB/s でデバイス上限 (P325 の 6.2 GB/s)。14k: b1 12.57、
  r8 12.18、b2 11.48。
- decode は 18.3-18.5 ms/tok で全 COND 同一。md5 は
  `a:737a73af b:87d8a97c c2000_cold:f9557457 c2000_turn2:fbaf5055
  c8000_cold:27c0c9dc` が 5 COND x 2 ラウンドで同一、`c8000_turn2` だけ揺れる
  (UD の既知の性質、P386 §8)。errors 0、text sane。

### 3.2 t394b (UD-Q4_K_XL、x0 / b1 / lp (先読み) / lpm (先読み + MT) / b2)

`load_before=1.86`。**この回はディスクが不調で、b1 の 14k は copy_s 32.61 s
(通常 12-13 s)、lpm の round 2 の 3,826 は copy_s 23.55 s** だった。
そのため 14k は判定に使えない (壁: b1 45.06 / lp 26.38 / lpm 26.19 / b2 25.41 s)。

3,826 cold (copy_s / 壁、ms):

| cond | round 1 | round 2 |
| --- | --- | --- |
| x0 | 10.00 / 12,393 | 8.06 / 10,014 |
| b1 | 8.56 / 10,570 | 7.36 / 10,086 |
| lp | 7.73 / 9,722 | 7.25 / 9,271 |
| lpm | 7.11 / 9,092 | 23.55 / 25,496 (ディスク不調) |
| b2 | 9.14 / 11,372 | 7.20 / 9,170 |

- **round 2 では lp (9,271) は b2 (9,170) と区別できない。** round 1 だけ見ると
  lp が最速だが、b2 の round 1 が遅い (11,372) ためで、中央値の -8% は
  位置効果。**先読みは効いていない。**
- 台帳: `lpre_layers=48 lpre_MB=77017.9 lpre_calls=144` (発火はした)。
  1 層 1,604 MB を投げているが、`[exp-pf-stream]` の実績は 1 層あたり
  296 本 = 925 MB。**1.73 倍の無駄読み** (層の全数は 512 本)。
- md5 は t394a と同じ 5 本が全 COND 同一、errors 0、text sane。

### 3.3 t394c (既定モデル flash-next-UD-IQ3_XXS、x0 / b1 / mt / b2)

`load_before=1.83`。

| cond | 3,826 cold | 14k cold | decode | md5 |
| --- | ---: | ---: | ---: | --- |
| x0 | 2.42 | 6.03 | 15.6 | SAME |
| b1 | 2.42 | 6.51 | 15.5 | SAME |
| **mt (既定 on)** | **2.44** | **6.43** | 15.5 | **SAME** |
| b2 | 2.44 | 6.40 | 15.5 | SAME |

- mt vs b1/b2 は 3,826 +0.8%/+0.0%、14k -1.2%/+0.5%、decode ±0.1 ms/tok。
  **悪化なし**。md5 は 4 COND 全 8 リクエストで同一 (`edaa84ad / 1938059f /
  b42d79d2 / 2081cdeb / 672fec07 / bc4aaf54`)。
- `[pf-restore]` (14k、827 本): mt=0 121 ms → mt=1 **71 ms** (-41%)。
- 既定モデルの戻しは 14k でしか発火しない (3,826 では追い出し 0 本)。

## 4. text (そのまま引用)

- t394a r8 の 3,826 cold: 「prefill の律速は計算量ではなく、線形層ごとの
  ホスト往復と戻り値コピーであった。計測の誤認や撤回された最適化を経て、
  GPU常駐化と転送経路の改善により、1.5Bモデルで約32倍の高速化を達成した。
  最終的に、PCIe帯域の上限に近づき、往復がボトルネックではなくなった状態へ至った。」
- t394a r8 の 14k cold: 「prefillの律速は計算量ではなく、線形層ごとのホスト往復と
  戻り値コピーであった。pinned経由の二度手間を排除し、FFNをデバイス常駐化する
  ことで、1.5Bモデルで395.8秒から12.2秒へ高速化した。」
- t394b lpm の 3,826 cold は t394a r8 と同一の文 (md5 f9557457)。
- TXT (B-tree) / TXTB (Rust の所有権) も全 COND で同じ形。既定モデル (t394c) は
  md5 が 4 COND 全部 SAME なので text も同一。

## 5. 分かったこと (次の担当者へ)

1. **3,826 の壁の床は 44.4 GB / 6.2 GB/s = 7.11 s。** 現在の壁 9.1-9.7 s との差
   (2-2.6 s) は「attention (phase 1) のあいだ routing が決まらず読めない」こと
   と、終端の戻し・ple で決まる。**読む速さではもう縮まない。**
2. **層の expert は全数要らない。** `[exp-pf-stream] experts=14192 / 48 層 =
   296 本` に対し層の全数は 512 本 (`lpre_MB` 77 GB / 48 層 / 3.13 MB = 512)。
   routing は 58% しか使わないので、**全数を先に読む手は無駄読みで負ける**
   (t394b)。予測で絞る道は task 337 が否定済み。この枝は閉じる。
3. **14k の床は `gpu_busy` 21.53 s + ホストの直列 (ple 0.53 / embed 0.12-0.34 /
   戻し 0.21-0.55 / tail 0.1) = 約 22.5 s。** 現在 25.1-25.3 s との差 2.6-2.8 s は
   expert の流し込みの層境界のドレイン (`slotwait_s` 5.3-5.8 s、
   `up_wait` 17.1-17.7 s)。ここを詰めるには層の先頭グループの転送を routing より
   前に始めるしかないが、それは 2 の理由で無駄読みになる。
4. **戻しの並列化は効くが小さい** (壁の 1-2%)。段は 2 割以上速いので既定 on に
   した。8 本と 16 本は 3,826 で 55/53 と 59/57、14k で 213 と 177 で、
   差は 1 サンプルの範囲。8 本 (既定) で十分。
5. **この環境のディスクは時々 1.9-3 GB/s まで落ちる** (t394b の b1 14k 32.61 s、
   lpm round 2 23.55 s)。判定に使う回は `copy_s` を必ず読むこと。

## 6. テスト

- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`:
  **1,273 passed / 0 failed / 103 ignored**
- `... --test kernel_invariants`: **116 passed / 0 failed**
- 新規: `restore_mt_gate_truth_table` / `layer_pre_gate_truth_table` /
  `t394_restore_mt_is_wired` (配線 + 台帳)。既存の `t242_...` は切り出した
  `pf_restore_one` を見るように直した。
- `.cu` / `.ptx` は触っていない。

## 7. GPU で確認してほしいこと

1. 既定 on にしたのは戻しの並列化 1 本。`DENNO_PF_RESTORE_MT=0` で切り戻せる。
   既定の prefill 経路は「戻しの転送を N 本で重ねる」以外 1 バイトも変わらない
   (t394c の既定モデルは md5 まで同一)。
2. `DENNO_PF_LAYER_PRE` は既定 off のまま (測って負け)。
3. 3,826 の床はディスク (6.2 GB/s)、14k の床は GPU (21.5 s)。次の一手は
   「層の先頭グループの転送を routing 前に始められる形」= expert を層単位で
   VRAM に置く形を作るしかない (推定)。無駄読みを避けるには routing が要るので、
   層の全数を置ける VRAM (925 MiB/層) が前提になる。
