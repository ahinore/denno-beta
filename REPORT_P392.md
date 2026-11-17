ADOPT: yes t392c (DENNO_GDN_X_MIX 既定 on) / t392b の DENNO_GDN_BA_PAR は ADOPT: no

# REPORT P392 — UD-IQ3_XXS decode: GDN の入力 cur を mix のデバイス出力から直接読む

branch `br392` = 開始時の `int-auto` **04ab8eb8b** (merge: br388。自分の br388 は
既に merge 済みだったので取り込み不要)。作業中に int-auto は動かなかった
(最終確認でも `04ab8eb8b` = merge 不要)。数値はすべて GPU ラボの実測で、
lab 名と測った SHA は **t392a = 04ab8eb8b** (プロファイル) /
**t392b = a2eec8ebe** (ba の重ね打ち A/B) / **t392c = 598a3f0d7** (x-mix の A/B)。
`RESULT.md` 2 行目は 3 run とも
`model=flash-next-UD-IQ3_XXS ptx=none`。

## 0. 結論

1. **採る (`ADOPT: yes`): `DENNO_GDN_X_MIX` を既定 on にした。** ssm 層の入力
   `cur` は直前の attn mix が作った `mixed` そのもので、mix の Q8_0 鎖はその
   コピーを**デバイスにも残している** (`dm`)。GDN の submission は同じバイト列を
   H2D で上げ直さず、そのポインタを qkv GEMV / beta-alpha matvec の入力に使う。
   層あたり 10.4 KB の pageable 同期 H2D (実測 5.6 us) が丸ごと消える。
   - **ssm 段 (同一 run の `[q4ledger]`、ms/token)**: round 1 は base
     3.69/3.52/3.71/3.40/3.58 (**中央値 3.58**) -> xm 3.51/3.41/3.57/3.32/3.44
     (**3.44 = -3.9%**)、round 2 は base 3.53 -> xm 3.43 (**-2.8%**)、
     base2 は 3.53 / 3.52。**2 ラウンドとも 2% 超**(オペレーター規則)。
   - **decode の中央値 15.6 -> 15.4 (-1.3%)**、prefill 0.66 -> 0.67 (不変)、
     errors 0、md5 は全 COND・全コンテキストで **SAME**、text 健全 (§4)。
     task の目標 (decode 2%) には届いていない。
   - 発火の証拠: `[gdn-x-mix] no_h2d_layers=4608 h2d_layers=0` (両ラウンド・
     全リクエスト)。`[q4ledger] ssm` の h2d は **1.29MB/37 -> 0.92MB/1**
     (残る 1 本は層 0 の状態シード)。
2. **採らない (`ADOPT: no`): `DENNO_GDN_BA_PAR`** (beta/alpha の matvec を
   専用ストリームで qkv GEMV と重ねる)。狙いは 1 層 6.0 us だが、**重ねると
   qkv GEMV が +1.2 us 遅くなり、鎖のカーネル時間は 89.2 -> 89.3 us で不変**。
   ba の実測は 6.0 -> 3.0 us に半減したのに総量が変わらない = 2 本は同じ資源
   (単一ウェーブの load/store 発行) を取り合っている。さらにイベント 2 本ぶんで
   launch が 10.1 -> 11.1 us 増え、ssm は 4 リクエストすべてで +0.05〜+0.19 ms
   悪化した。ゲートは既定 off のまま残す (§3.1)。
3. **却下の理由も数値で残した**: GDN の 1 層は
   `kern 89.6 us (qkv 2x35.2 + scan 13.2 + ba 6.0) + copy 16.4 + gap 5 + launch 10`
   で、**ba 以外は全部が直列依存**。qkv と out の GEMV は 1 本ずつ (qkv+gate は
   連結済み)、scan は qkv と ba の後、out は scan の後。つまり ssm には
   「独立な仕事」が ba しか無く、それは重ねても縮まない (t392b の実測)。

## 1. 前提: どこを削ることにしたか (task の Do 1/2)

既定モデル (flash-next-UD-IQ3_XXS) の **t392a の取り直し** (int-auto 04ab8eb8b、
`SCENARIO=depth8k PROFILE=1`、ベースライン COND = 22000/12000/`GPU_MIN_HIT=0`、
load_before 1.67、4 COND x 2 ラウンド):

| cond | decode ms/tok | prefill s | md5 | errors |
| --- | ---: | ---: | --- | ---: |
| x0 (捨て) | 15.5 | 0.67 | 全 SAME | 0 |
| base | 15.6 | 0.65 | 全 SAME | 0 |
| kp (KPROF 3 種) | 15.6 | 0.66 | 全 SAME | 0 |
| base2 | 15.6 | 0.67 | 全 SAME | 0 |

段の台帳 (base の短い decode の窓 = a2、`[q4ledger]`、step 15.56 ms/token):

| 段 | ms/tok | 中身 |
| --- | ---: | --- |
| experts | 4.45 | cpu 4.45 (ミス CPU 2.24 + IQ 鎖の起動 4 本/層) |
| mix | 4.36 | cpu 3.69 / launch 0.35/242 |
| **ssm** | **3.52** | launch 0.20/144、GPU は 1 層 89.6 us |
| full | 1.41 | sync 0.98 / gpu 0.97 |
| head | 0.73 | gpu 0.70 (帯域) |
| shared | 0.63 | sh_begin 0.43 |
| router | 0.18 | — |
| loop | 0.01 | sync 2.18/49 (層端の待ち) |

14k 後 (c8000_turn2) は mix 3.97 / ssm 3.45 / experts 3.28 / full 2.12 / head 0.73 /
shared 0.61 で step 15.45。

**選んだ理由**: 大きい 2 本 (experts 4.45 / mix 4.36) は既に床にいる。
experts のミス CPU は 38 本/token の重みを **2 回** (上げ直しの pread と CPU GEMV)
DRAM から読んでいて 43 GB/s ≒ DRAM の実効上限、mix の鎖は投入律速 (P382:
launch -194 本 = -0.43 ms、t384a: CUDA graph は負け、t372c: ホスト仕事は待ちに
隠れる)。その次に大きい **ssm 3.52** は、t392a のカーネル別内訳
(`[dec-kprof-avg] gdn`) を取ったところ **1 層 115 us のうち 16.4 us がコピー**
(pageable の同期 H2D 5.6 + 最後の D2H 10.4) で、ここだけが「同じ値を別の
置き場から読む」で消せる無駄だった。それが `DENNO_GDN_X_MIX`。

## 2. 変えたこと (env ゲート 2 本)

### 2.1 `DENNO_GDN_X_MIX` (既定 on → t392c で採った)

- `maybe_gpu_mix_q_chain_into` の鎖が同期で終わった直後に、`mixed` のデバイス
  ポインタと要素数を控える (`cuda::gdn_x_in_mix_arm`、Atomics 2 語)。
  `zc_out` のときは実際に書いた先 (`omap.dev`) を控える。
- `gdn_decode_submit` は控えを **1 回だけ**取り出し (`gdn_x_in_mix_take`、
  要素数が違えば捨てる)、あれば `cur` を xba へ H2D せず、そのポインタを
  qkv GEMV / ba matvec の入力にする。beta/alpha は従来どおり xba の側。
- **fail closed の条件**: 鎖が不発 (BF16 以外の別経路・prefill)、要素数違い、
  `ba_units` が空のモデル (β/α をホストで計算する = staging に詰める必要がある)、
  `DENNO_GDN_X_MIX=0`。どれも従来の H2D 経路へ落ちる。
- 発火の証拠: `[gdn-x-mix] no_h2d_layers=N h2d_layers=M` をリクエスト終端で 1 行。

### 2.2 `DENNO_GDN_BA_PAR` (既定 off のまま = 却下)

- `gdn_launch_ba` を `gdn_launch_ba_on(ctx, stream, ..)` にし、ba を専用
  ストリーム (CU_STREAM_NON_BLOCKING + イベント 2 本) へ出して scan の前に join。
- 値は変えない (同じカーネル・同じ入力)。**実測は ssm が全リクエストで悪化**
  したので既定 off。将来「GEMV を細くする」以外の理由で再挑戦するなら、
  このゲートと `[gdn-ba-par]` の行はそのまま使える。

## 3. lab の結果

すべて `SCENARIO=depth8k PROFILE=1`、ベースライン COND は
`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0
DENNO_GPU_WEIGHT_MB=12000`。最初の COND は捨て、最後に同じものを置いた。

### 3.1 t392b (`a2eec8ebe`、6 COND… 実際は 5 COND x 2 ラウンド、load_before 2.57)

| cond | env | decode ms/tok | prefill s | md5 |
| --- | --- | ---: | ---: | --- |
| x0 | (捨て) | 16.1 | 0.72 | 全 SAME |
| base | — | 15.7 | 0.69 | 全 SAME |
| **par** | `GDN_BA_PAR=1` | 16.2 | 0.68 | 全 SAME |
| parp | `GDN_BA_PAR=1` + `GDN_DEC_KPROF=1` | 15.8 | 0.68 | 全 SAME |
| base2 | — | 15.6 | 0.68 | 全 SAME |

```text
VERDICT base  vs x0 OK decode -2.4% prefill -4.2% md5 SAME errors 0
VERDICT par   vs x0 OK decode +0.7% prefill -4.9% md5 SAME errors 0
VERDICT parp  vs x0 OK decode -1.5% prefill -5.6% md5 SAME errors 0
VERDICT base2 vs x0 OK decode -2.9% prefill -5.6% md5 SAME errors 0
VERDICT baseline x0 errors 0 text ok
```

ssm 段 (4 リクエスト、ms/token): base 3.76/3.53/3.71/3.41、**par
3.81/3.72/3.87/3.59**、base2 3.69/3.51/3.70/3.40 = par が全点で悪い。
`[dec-kprof-avg] gdn` (parp、1 層): wall 115.8 / kern **89.3** (ba **3.0**、
qkv 36.4、scan 13.2) / copy 16.4 / gap 5.4 / launch **11.1**。base (t392a の kp)
は kern 89.6 (ba 6.0、qkv 35.2)、launch 10.1。
つまり **重ね打ちは成立している (ba 6.0 -> 3.0) が、qkv が +1.2 us 吸って
総量が変わらない**。`[gdn-ba-par] par_layers=4608 fallback=0`。

### 3.2 t392c (`598a3f0d7`、5 COND x 2 ラウンド、load_before 1.03)

| cond | env | decode ms/tok | prefill s | md5 |
| --- | --- | ---: | ---: | --- |
| x0 | (捨て) | 15.8 | 0.67 | 全 SAME |
| base | — | 15.6 | 0.66 | 全 SAME |
| **xm** | `GDN_X_MIX=1` | **15.4** | 0.67 | 全 SAME |
| xmb | `GDN_X_MIX=1` + `GDN_BA_PAR=1` | 16.5 | 0.67 | 全 SAME |
| base2 | — | 15.6 | 0.67 | 全 SAME |

```text
VERDICT base  vs x0 OK decode -1.2% prefill -2.2% md5 SAME errors 0
VERDICT xm    vs x0 OK decode -2.2% prefill -0.7% md5 SAME errors 0
VERDICT xmb   vs x0 OK decode +4.7% prefill -0.7% md5 SAME errors 0
VERDICT base2 vs x0 OK decode -1.0% prefill +0.0% md5 SAME errors 0
VERDICT baseline x0 errors 0 text ok
```

ssm 段 (ms/token、5 リクエスト + round 2 の 6 リクエスト):

| cond | round 1 | 中央値 | round 2 | 中央値 |
| --- | --- | ---: | --- | ---: |
| base | 3.69 3.52 3.71 3.40 3.58 | 3.58 | 3.69 3.51 3.70 3.41 3.53 3.44 | 3.53 |
| **xm** | 3.51 3.41 3.57 3.32 3.44 | **3.44 (-3.9%)** | 3.51 3.41 3.56 3.32 3.46 3.35 | **3.43 (-2.8%)** |
| base2 | 3.72 3.52 3.72 3.41 3.53 | 3.53 | 3.69 3.52 3.72 3.41 3.54 3.44 | 3.52 |
| xmb | 3.82 3.66 3.79 3.50 3.57 | 3.66 | — | — |

他の段は xm が base と同じか僅かに良い (round 2 の短い窓: mix 5.04 対 5.12、
full 1.42 対 1.44、head 0.73 対 0.74、shared 0.69 対 0.69)。
`[gdn-x-mix] no_h2d_layers=4608 h2d_layers=0`。

### 3.3 14k 後の decode について (注意)

t392c の 14k 点は base 14.8 / xm **16.9** / base2 14.5 ms/tok で、xm が遅く出た。
**これはゲートのせいではない**と、同じ run の台帳で切り分けた:

- xm の 14k 窓は **名前の付いた段がすべて base 以下** (ssm 3.39 対 3.47、mix 3.97
  対 3.98、full 2.12 = 2.12、experts 3.33 対 3.26、head 0.73 対 0.74)。
  段の合計は xm 14.41 / base 14.50 ms/token。
- 差は**台帳に入らない区間** (PLE inject / embed) に全部出ている: その残差は
  base/base2 でも同じ run の別リクエストで **+1.5〜+3.0 ms/token** まで跳ねる
  (base_1 の b と 14k、base2_1 の b と 14k)。14k decode の PLE pread は
  1.1〜2.4 ms/token で run 間・順序で倍ほど動く (MEMORY / P368 の申し送り)。
- 14k は **1 COND 1 サンプル** (lab は round 1 だけ 14k を回す) なので、
  この 1 点で採否を読まない。段の台帳 (同一 run) が根拠。

## 4. 出力テキスト (sane) と md5

t392c の xm から:

- `TXT`: `# B-Trees: A Comprehensive Explanation\n\n## 1. What is a B-Tree?\n\nA **B-tree (Balanced Tree) is a self-balancing tree data structure that maintains sorted data and allows searches, sequential access, insertions, and deletions in logarithmic time.`
- `TXTB`: `Rust の「所有権（Ownership）とライフタイム（Lifetime）は、Rust を学ぶ上で最も重要であり、かつ最も難しい概念です。\n\nしかし、これらは「メモリを安全に管理するためのルール**です。`
- `TXTD` (14k 後): `ご提示いただいた文書は、LLM推論エンジン（\`denno-engine）における **Prefill（事前計算）フェーズの性能最適化の歴史を詳細に記録した技術レポートです。`

md5 は t392c の全 COND・全コンテキストで
`a:edaa84ad b:1938059f c2000_cold:b42d79d2 c2000_turn2:2081cdeb
c8000_cold:672fec07 c8000_turn2:bc4aaf54` (60 リクエストすべて SAME、base と
1 バイトも違わない)。errors は全 COND 0。

## 5. 数値の扱い

- **値は 1 つも変えない**。同じバイト列 (直前の mix の出力) を、H2D した写しでは
  なく元のデバイスバッファから読むだけ。md5 が全 SAME なのはその証拠。
- 採否は同一 run の `[q4ledger] ssm` (段) を主、decode の中央値を従にした
  (0.1 ms/tok 刻み)。t392b は load_before 2.57 で絶対値が高く、x0 と base の
  間にも -2.4% の drift がある回なので、run 内の比だけで読んだ。
- t392c の decode 中央値 15.6 -> 15.4 は **-1.3%** で、目標 2% には届かない。
  ssm 段は 2 ラウンドとも 2% 超なので、オペレーター規則 (段で 2% 以上を
  2 ラウンド + 他が 1% 超悪化なし) で既定 on にした。
- 既定 on の確認 run は回していない (lab ラウンドの上限 3 回を、task が要求する
  プロファイル 1 回 + A/B 2 回で使い切った)。既定の切り替えは env 解釈の
  1 行で、測った `=1` と同一のコード経路。

## 6. テスト

関門 (`--features cuda`、最終 HEAD で実行):

```text
CARGO_TARGET_DIR=/tmp/t392 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1276 passed; 0 failed; 103 ignored
同 --test kernel_invariants
  -> 115 passed; 0 failed
```

(P388 の最終 HEAD は 1271 + 115。足した 5 本が増えている。)
足したテスト (5 本、`cuda::tests`):
- `gdn_ba_par_gate_truth_table` / `gdn_ba_par_line_counts_fired_layers`
- `gdn_x_mix_gate_truth_table` (既定 on・明示的な偽値だけ off)
- `gdn_x_mix_take_is_one_shot_and_checks_len` (1 回で消える / 要素数違いは捨てる)
- `gdn_x_mix_line_counts_layers`

## 7. GPU で確認してほしいこと / 次のタスクへの申し送り

1. 既定 on のビルドで `[gdn-x-mix] no_h2d_layers=<48 x 層数>` が出ること
   (`DENNO_GDN_X_MIX=0` で行が出ず、`[q4ledger] ssm` の h2d が 1.29MB/37 に戻る)。
2. **ssm の残りは 1 層 89.6 us の GPU 時間**で、qkv (35.2) + scan (13.2) + out
   (35.2) が直列。ここを縮めるには (a) `.cu` で scan を out に融合する、
   (b) GEMV の形を変える (単一ウェーブ・load/store 発行律速なので lanes では
   動かない — t281 / t392b)、のどちらか。ホスト側でできることはもう無い。
3. `DENNO_GDN_BA_PAR` は既定 off の診断用として残した。**同じ形の「独立な
   カーネルを重ねる」は今後も効かない**見込み (ba 6.0 -> 3.0 でも総量不変)。
   重ね打ちを試すなら、まず `[dec-kprof-avg]` の `kern` 合計が減るかを見る。
4. 次に decode で削るなら **experts のミス経路の DRAM 交通量** (重みを 2 回
   読んでいる: 上げ直しの bounce と CPU GEMV)。ミス CPU 2.24 ms/token の
   うち半分は「2 回目の読み」なので、同じ pinned バッファから CPU GEMV が
   読めれば 1 ms/token 級。t392a の台帳が根拠。
