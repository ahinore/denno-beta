ADOPT: yes t397a t397b t397c

# task 397 (dev2) — 既定モデルの prefill: GDN scan (wy4) のチャンク内の従属連鎖を削る (wy5)

branch `br397` = 開始時の `int-auto` `04ab8eb8b`。**br393 は int-auto に merge 済み**
(`b7f3e5528`) だったので merge していない。作業中に int-auto は
`04ab8eb8b` -> `8aaf48807` (br392 / br394) へ動いたので終端で merge した
(衝突なし。`targets_all` は末尾に足したので当たらなかった)。

数値はすべて GPU ラボの実測 (`SCENARIO=depth8k`、`PROFILE=1`、
`model=flash-next-UD-IQ3_XXS`)。lab 名と測った SHA:
**t397a = `86c4779c0`** (5 COND、`ptx=gdn_pf_wy5`、wy5 の初期版) /
**t397b = `1fdb6c140`** (4 COND、`ptx=gdn_pf_wy5`、wy5 の最終版) /
**t397c = `813993198`** (5 COND、`ptx=none` = 収録した PTX、既定 on の確認)。
3 ラウンドで上限。**採用したのは t397b / t397c で測った最終版**。

## 0. 結論 (7 行)

- **採る 1 本**: `DENNO_PF_SCAN_WY5` (prefill GDN scan の WY カーネルの作り直し)。
  **既定 on** にした。切り戻しは `DENNO_PF_SCAN_WY5=0` (wy4 に戻る)。
- **ssm 段の GPU busy が -14.4% / -14.5%** (3,826 / 14k)。同じ run の中で
  on と off を交互に測った t397c で **433.3 -> 371.2 / 371.9 / 371.3 / 371.3 ms**
  (3,826)、**1,557.7 -> 1,331.2 / 1,330.6 / 1,331.3 / 1,330.3 ms** (14k)。
  ベースラインは 3 run で 432.8-433.4 / 1,552.7-1,558.9 と安定。
- 壁は **3,826 で -2.5%** (t397c: `off` 2,437.4 -> on 2,380.0 / 2,385.9 / 2,368.6 ms)、
  **14k は解像できない** (同じ run の on が 5,739-6,265 ms に散る。位置効果と
  `other` 段のホスト揺れ ±400 ms)。14k の `gpu_busy` は 4,675.5 -> 4,450.4 ms
  (**-4.8%**) なので、GPU 仕事は確実に減っている。
- **decode は不変** (15.5-15.6 ms/tok、3 run の全 COND)、errors 0、text sane (§5)。
  md5 は **turn2 の 2 本だけ DIFF** (`c2000_turn2` / `c8000_turn2`)、cold の 2 本は
  SAME。理由は §2.1 (A/G と P/QP の和の木を変えた = 意図した丸め差)。
- **落とした 1 本**: `DENNO_PF_GDN_WB_ASYNC` (状態の書き戻しを async D2H +
  自ストリームの待ちにする)。ssm の exposed が 901.0 -> 897.8 / 901.9 ms で
  1 ms も動かず、**既定 off のまま**残した (A/B 用)。
- 目標 (壁で 2.5%) は 3,826 で満たし、14k は「悪化していない」とは言えるが
  ±10% のノイズの中では差を出せない。採否はオペレーター規則
  (段で 2% 以上 x 2 ラウンド、他は 1% 以上悪化しない) で決めた: **3 ラウンドとも
  ssm 段 -5.6% / -14.4% / -14.3% で規則を通っている**。

## 1. なぜ scan を取ったか (Do 1: 現行 int-auto の取り直し)

### 1.1 段のプロファイル (14k cold、`[q4ledger-pf]`、ms)

t397c の `off` (= int-auto の既定 = wy4) と t397a / t397b の b1/b2 (同じ):

| 段 | wall | gpu busy | exposed | 中身 |
| --- | ---: | ---: | ---: | --- |
| ssm | 2,458.1 | **1,557.7** | 900.4 | scan 936 + qkv 251 + z 159 + out 153 + ba 60 (t389a の kprof) |
| experts | 1,886.3 | 1,425.0 | 462 | 27.4 GB の流し込み (P385 で床) |
| full | 879.0 | 638.8 | 241 | bf16 TC で 45 TFLOPS 級 (余白小) |
| mix | 21.0 | 882.6 | 0 | 別ストリームで重なる |
| other | 758.0 | 172.0 | 586 | ホスト (ple 214 / res_up / expert の戻し) |

3,826 cold (t397c の `off`): ssm 455.0 / **433.3** / 21.0、experts 1,336 / 530 / 806、
mix 294 / 243 / 52、full 127 / 89 / 38、other 226 / 47 / 178。

### 1.2 選んだ理由

- **ssm の GPU 仕事が最大** (14k で gpu_busy の 33%、3,826 で 32%) で、その
  60% が GDN scan カーネル 1 本 (t389a の `[gdn-pf-kprof]` 14k: scan 935.7 ms /
  qkv 251.1 / z 158.6 / out 153.3 / ba 59.8、合計 1,559 = `ssm busy` と一致)。
- scan は **1.5-3 TFLOPS** (ピークの 2-3%) しか出ておらず、P311 が v 列 4 分割
  (wy4) まで、P379 §7.3 が「GDN scan (14k 936 ms = gpu_busy の 20%) は `.cu` の
  許可が要る」と**次の担当者に残していた**。本タスクは「新しいカーネルは
  PTX= で」と `.cu` を許可しているので、ここを取った。
- 他の段は閉じている: experts は DRAM 3 ストリームの床 (P385)、full は TC で
  効率が良く (45 TFLOPS)、mix は別ストリームで重なる (882 ms の busy が
  wall 21 ms)、`other` は P393 で PLE / res_up を削った残り。

## 2. 変えたこと (新カーネル 1 本 + ゲート 2 本)

### 2.1 `kernels_gdn_pf_wy5.cu` (`DENNO_PF_SCAN_WY5`、**既定 on**)

wy4 の写しで、**直したのはチャンク内の従属連鎖 4 点**だけ。チャンク長 C=8、
v 列 4 分割 (192 ブロック)、256 スレッド、共有メモリ 7,393 float (29,572 B) は
wy4 と同一。カーネル引数も 17 個で同じ (`gdn_pf_wy_launch` を共有)。

1. **A/G** (旧: 1 スレッド = 1 組で 128 要素の直列ドット積 x 2 本、36 組を
   36 スレッドが担当) -> **1 ワープ = 1 組**。レーンが 4 要素ずつ積んでから
   `__shfl_down_sync` 5 段。従属連鎖 128 -> 9。和の木が変わるので**丸めが wy4 と
   違う**。
2. **P/QP** (旧: 2 アキュムレータで 128 段) -> **4 アキュムレータ**
   (`r` を 4 本ずつ、最後に 1 回だけ足す)。従属連鎖 128 -> 32。ここも丸めが変わる。
3. **conv の入力**: チャンクの 8 トークンぶんを先に読む (コンパイル時に固定した
   8 回の独立ロード)。窓の前進は `tt < len` の内側だけ = wy4 と同じ量。
   演算列は同一で値は wy4 と一致。
4. **o と状態の更新** (t397b で追加、**値は wy5 と同一**):
   - `o = QP + (G の下三角) D` は D が確定済みで行が独立なのに、wy4 は 32 スレッドで
     8 行を直列に回していた -> `(i, c)` の全要素を 1 スレッドずつに (j 昇順は同じ)。
   - 状態の更新は 1 要素ごとに `rl[t]` / `Dc[t][c]` を読み直していた -> `t` を外に
     回して 1 回ずつ読み、16 本の `r` をレジスタに持つ。積は `rl*Kc*Dc` の
     左からの順のまま (`-fmad=false`)。1 スレッドが持つ `r` の集合は wy4 の
     `idx += THREADS` と同じ。
- 形の前提は wy4 と同じ (`gdn_pf_wy5_ok` = `gdn_pf_wy4_ok`)。PTX が無い / 形が
  違う / opt-in に失敗は wy4 -> v3 -> v2 -> v1 -> CPU へ fail closed。
- **A/G の組の列挙が wy4 と同じ集合を 1 回ずつ覆う**ことは len=1..8 でミラー検証した
  (組 `(i,j)` j<=i の全数)。カーネルは g++ スタブで構文検査、レジスタは lab の
  `[kattr]` で `num_regs=95 local_bytes=0` (wy4 は 96。spill なし)。

### 2.2 `DENNO_PF_GDN_WB_ASYNC` (既定 off、**採らず**)

状態の書き戻し (`conv 窓 + ssm 状態`) を `cuMemcpyDtoHAsync` (pinned) +
`pf_sync_own_stream` (null ストリームだけの待ち) にする。14k は層の最終チャンクで
36 回、`cuMemcpyDtoH` がコンテキスト全体 (mix の別ストリーム / expert の上げ
ストリーム) を待っている、という読みだったが、**t397a の実測で ssm の exposed は
901.0 (wy4) -> 897.8 (wb) / 901.9 (b2) で動かなかった**。失敗時は従来の同期
コピーでやり直す実装なので、既定 off のまま A/B 用に残す (コードは生きている)。

## 3. 数値 (lab、同じ run の中の対)

### 3.1 ssm 段の GPU busy (ms) — 採否の判定に使った段

| run | SHA | 3,826 base | 3,826 wy5 | 14k base | 14k wy5 |
| --- | --- | ---: | ---: | ---: | ---: |
| t397a | `86c4779c0` | 433.1 / 433.4 | 409.0 (-5.6%) | 1,557.3 / 1,558.9 | 1,469.4 (-5.7%) |
| t397b | `1fdb6c140` | 433.3 / 433.1 | **370.9 (-14.4%)** | 1,554.5 / 1,557.9 | **1,330.9 (-14.5%)** |
| t397c | `813993198` | 433.3 (off) | **371.2 / 371.9 / 371.3 / 371.3** | 1,557.7 (off) | **1,331.2 / 1,330.6 / 1,331.3 / 1,330.3** |

t397c は**同じ run の中で** `off` (明示的に wy4) と既定 on を交互に置いたので、
この差がそのままゲートの効果。on の 4 サンプルは ±0.5 ms (0.1%) で揃っている。

### 3.2 他の段と壁 (t397c、14k cold)

| 段 (ms) | off (wy4) | b1 / b2 / wb (wy5) |
| --- | ---: | ---: |
| ssm wall / busy / exposed | 2,458 / 1,558 / 900 | 2,243 / 1,331 / 912 |
| full busy | 639 | 639 / 638 / 640 |
| experts busy | 1,425 | 1,425 / 1,426 / 1,426 |
| mix busy | 883 | 884 / 884 / 883 |
| close gpu_busy | 4,675.5 | 4,450.4 / 4,449.7 / 4,451.7 |
| close wall | 6,002.4 | 6,204.3 / 6,052.0 / 5,739.5 |
| decode (ms/tok) | 15.6 | 15.6 / 15.6 / 15.5 |

**ssm 以外の GPU 段は 1 つも動いていない** (±1 ms)。`other` (ホスト) は
758 (off) / 1,188・1,045・719 (on) と **±400 ms 揺れる** — 14k の壁が
on と off で読めないのはこれと位置効果 (§4)。

3,826 cold の壁 (t397c): off 2,437.4 -> b1 2,380.0 / wb 2,368.6 / b2 2,385.9 ms
(**-2.3 〜 -2.8%**)。同じ run の `close gpu_busy` は 1,342.1 -> 1,281.5 / 1,282.6 /
1,282.6 (-4.5%)。

### 3.3 落とした wb (t397a、14k)

| | ssm wall | ssm busy | ssm exposed | other exposed |
| --- | ---: | ---: | ---: | ---: |
| b1 / b2 (wy4) | 2,455.1 / 2,460.8 | 1,557.3 / 1,558.9 | 897.8 / 901.9 | 1,302.0 / 555.4 |
| wb | 2,458.6 | 1,557.5 | 901.0 | 1,016.2 |

`other` の exposed は b1 1,302 / b2 555 ともともと揺れる帯なので、wb の 1,016 は
「差ではない」。**ssm の露出は 1 ms も動かず、採らない**。

## 4. 数値の扱いと見積り

- **段の数字が判定**。壁は 1 COND 1 サンプル (depth8k の 14k は round 1 だけ) で、
  同じ run の中でも 14k が 5,739-6,265 ms (9%) に散る。この機械の位置効果
  (P389/P393 の 20%) と `other` 段のホスト揺れ (±400 ms) が原因で、**2.5% は
  壁では解像できない**。3,826 は 2,368-2,437 (2.9%) で、こちらは辛うじて読める。
- 見積り (推定): wy4 -> wy5 で 1 チャンクあたり 14.8 -> 12.6 us (14k の
  scan 936 -> 795 ms 相当)。t397a (A/G + P/QP + 先読みのみ) が -5.7%、
  t397b (o と状態更新を追加) が -14.4% なので、**残りの伸びの大半は
  o の 8 倍並列と状態更新の読み直し削減**だった (A/G の直列ドット積は
  思ったより小さかった — 推定)。
- 14k の `gpu_busy` -4.8% に対して壁が動かないのは、14k の壁が
  `experts` の流し込み (exposed 462) と `other` のホスト仕事 (586) で決まって
  いるため (推定ではなく台帳の内訳そのもの)。scan をもう 800 ms 削っても
  壁は 2.5% しか動かない計算になる。

## 5. text (sane の証拠)

t397c の b1 (既定 = wy5) の c8000_cold (14k cold の応答、`b1_1_c8000_cold.json`):

> prefillの遅延は計算量ではなく、ホスト往復とメモリコピーが律速であった。
> 計測の誤認や静かな退行を排除しつつ、デバイス常駐化と転送経路の最適化を段階的に実施した。
> その結果、1.5Bモデルで395.8秒から12.2秒へ、9Bモデルで46.2秒から10.11秒へと大幅な高速化を達成した。

`off` (wy4) の c8000_cold と**同じ文**。他の COND も同じ形で、「!!!!」や空出力は無い。

## 6. 分かったこと (次の担当者へ)

1. **`[gdn-pf-kprof]` はこの木では数が足りない。** 14k の 1 リクエストで
   `qkv_gemm 68.5ms/72 | scan 87.8ms/36` しか出ない (チェーンは 36 層 x 7 チャンク =
   252 回。`[q4ledger-pf]` の `ssm` h2d 1080 回 = 252 x 4 が正しい)。
   t389a (11-17 朝の木) では 252 回ぶん全部出て `ssm busy` と一致したので、
   途中のタスクでイベントのプールが壊れた可能性がある。**段の判定は
   `[q4ledger-pf]` の busy を使うこと**。
2. **wy4 / wy5 はどちらも 1.5-3 TFLOPS で、まだ余白がある。** 直した 4 点で
   14.4% だった。残っているのは (a) 前進代入 (32 スレッド + 8 行の直列)、
   (b) `P/QP` が 8 ワープで S を 8 回読み直すこと (128 KB/チャンク)、
   (c) 7 バリア/チャンク。チャンク長 C=16 は P311 が直列 A/G のままで +5% だったが、
   A/G がワープ並列になった今は C=16 の A/G 固定費が 1/5 になっているので**もう一度
   測る価値がある** (推定)。
3. **14k の壁はもう scan では動かない** (§4)。14k で 2.5% を取るには
   `other` のホスト仕事 (586 ms) か expert の流し込み (462 ms の露出) しかない。
4. 3,826 も `experts` の露出 806 ms が壁の 34% を占めるので、ssm を削っても
   壁は -2.5% で頭打ち (実測)。ssm の残り (371 ms) を全部 0 にしても -14% が上限。

## 7. テスト

int-auto `8aaf48807` を merge した後の最終 HEAD で:

```
cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1280 passed / 0 failed / 103 ignored
cargo test --release -p denno-engine --features cuda --test kernel_invariants
  -> 117 passed / 0 failed
cargo test --release -p denno-engine --lib -- --test-threads=1        (cuda feature 無し)
  -> 949 passed / 0 failed / 55 ignored
cargo test --release -p denno-engine --test kernel_invariants          (cuda feature 無し)
  -> 113 passed / 0 failed
```

追加した 3 本:
- `task397_scan_wy5_gate_and_shape` (lib) — ゲートの真理値表 (既定 on・明示的な偽値だけ
  off)、形の前提 (dv=128 / n=0 を弾く)、共有メモリが wy4 と同じ 7,393 float
  (区画のオフセットが 1 つずつ wy4 と一致)、直した 4 点が .cu に居ること。
- `task397_gdn_wb_async_gate_defaults` (lib) — wb ゲートの真理値表 (既定 off)。
- `task397_gdn_pf_wy5_kernel_is_wired` (kernel_invariants) — カーネル名 / 17 引数 /
  `#define` / smem の区画が重ならないこと / 起動形 (`GDN_PF_WY4_BLOCKS` と
  `GDN_PF_WY5_SMEM_BYTES`) / 選択順 (wy5 -> wy4) / fail closed の判定に wy5 が
  入っていること / `build_ptx.sh` の target と targets_all の**末尾** /
  wb の実装が `memcpy_dtoh_async` + `pf_sync_own_stream` で同期コピーを含まないこと。
- **発火の証拠**は lab の段の数字 (§3.1)。GPU 無しではカーネルは走らないので、
  .cu は g++ スタブで構文だけ、数値の正しさは lab の text と md5 で見た。
- 既知の赤 (`qwen4exp_decode spec_batch_stops_at_eos`、`qwen4exp_params` の validate 2 件)
  は触っていない (この 4 本では 0 failed なので、いまは赤が出ていない)。

## 8. GPU で確認してほしいこと (監督へ)

1. 既定 on の wy5 は `DENNO_PF_SCAN_WY5=0` で wy4 に戻せる。wy4 の PTX はそのまま。
2. **PTX は lab の生成物を収録した** (`crates/denno-engine/kernels/kernels_gdn_pf_wy5.ptx`、
   `done/t397b/ptx/` と md5 一致)。t397c は `PTX=` 無しでこの PTX を測り、
   md5 が t397b と一致した (12fc5c36 / 98e879d2) ので、収録物と lab の生成物は
   同じ挙動。
3. 3,826 の壁 -2.5% は 1 サンプル対 3 サンプルなので、実機で 2,380 ms 前後に入るか
   を見てほしい。14k は scan 以外が壁なので、期待しないでほしい。
4. `[gdn-pf-kprof]` の数が合わない件 (§6-1) は、次のタスクで誰かが見るべき。

## 9. VERDICT 行

t397a (`86c4779c0`、5 COND、ptx=gdn_pf_wy5):
```
VERDICT b1 vs x0 OK decode -0.3% prefill +1.5% md5 SAME errors 0
VERDICT wy5 vs x0 OK decode +0.3% prefill +0.8% md5 DIFF errors 0
VERDICT wb vs x0 OK decode +0.0% prefill +0.8% md5 SAME errors 0
VERDICT b2 vs x0 OK decode +0.0% prefill +1.5% md5 SAME errors 0
VERDICT baseline x0 errors 0 text ok
```

t397b (`1fdb6c140`、4 COND、ptx=gdn_pf_wy5):
```
VERDICT b1 vs x0 OK decode +2.0% prefill +0.0% md5 SAME errors 0
VERDICT wy5 vs x0 OK decode +0.3% prefill -0.7% md5 DIFF errors 0
VERDICT b2 vs x0 OK decode +1.0% prefill -0.7% md5 SAME errors 0
VERDICT baseline x0 errors 0 text ok
```

t397c (`813993198`、5 COND、ptx=none = 収録 PTX):
```
VERDICT b1 vs x0 OK decode +0.0% prefill +0.8% md5 SAME errors 0
VERDICT off vs x0 OK decode +0.0% prefill +0.8% md5 DIFF errors 0
VERDICT wb vs x0 OK decode -0.3% prefill +1.5% md5 SAME errors 0
VERDICT b2 vs x0 OK decode +0.0% prefill +0.0% md5 SAME errors 0
VERDICT_DEPTH b1 c8000_cold vs x0 decode -5.1% prefill -1.0%
VERDICT_DEPTH off c8000_cold vs x0 decode -5.7% prefill -4.3%
VERDICT_DEPTH wb c8000_cold vs x0 decode +0.0% prefill -8.5%
VERDICT_DEPTH b2 c8000_cold vs x0 decode -6.3% prefill -3.5%
VERDICT baseline x0 errors 0 text ok
```

t397a の 3,826 は `wy5` の露出が 24 -> 238 ms に移って壁は +11% に見えた (t397a の
wy5 は A/G + P/QP + 先読みだけの初期版)。t397b の 3,826 はそのリクエストだけ
ページキャッシュが冷えて (`majflt=232,879`、`copy_s=4.55 s` vs 1.15-1.49 s)
壁が 5.99 s になったので、**3,826 の壁は t397c の 1 本 (off 2,437 -> on 2,368-2,386)
だけを読んでほしい**。
