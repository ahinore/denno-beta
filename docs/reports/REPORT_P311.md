ADOPT: no t311a t311b t311c

# task 311 — IQ3_S prefill: GDN scan を WY の v 列 4 分割で -28.6% (wall は -2.9% / -4.3%、目標 -5% に届かず)

branch `br311` は開始時の `int-auto` (`c8b41a29e` = `merge: br308`) から。
作業中に int-auto は `d6f914e97` (tasks 307/309) へ動いたので merge した (§1)。
lab 3 ラウンド: **t311a** (`7e6a8fca6`、5 COND、wy4 + 既知の wy32)、
**t311b** (`4142b4ad0`、5 COND、wy4 + wy8)、**t311c** (`77c8ddf72`、5 COND、
wy4 + wy16)。数値はすべて GPU ラボの実測。

## 0. 結論 (3 行)

- **採らない (目標未達)。** 3,826 cold は同じ run の baseline 2.945 s (中央値)
  → 2.86 s (**-2.9%**、t311b)、14k cold は 7.40 → 7.08 s (**-4.3%**、t311b)。
  目標 -5% に 3,826 は 2.1 ポイント、14k は **0.7 ポイント**届かない。
- 取ったのは 1 つ: **`DENNO_PF_SCAN_WY4`** — ssm 層の GDN スキャン (task 228 の
  WY/UT 形式チャンク版) を **v 列で 4 分割**した新カーネル `dno_gdn_pf_wy4`。
  1 ブロック = 1 ヘッド (48 ブロック) だったのを (1 ヘッド, v 列 32 本) の
  192 ブロックにする。`scan_pf` が 3,826 で **359.0 → 256.5 ms**、14k で
  **1315.9 → 938.2 ms**(どちらも **-28.6%**)。**md5 は wy32 と全出力で一致**
  (移植が要素ごとに同じ順序であることの実測)。
- 効かなかった 2 つを測って落とした: **8 分割 (wy8)** は A/G と conv の重複が
  8 倍になって `scan_pf` 413.6 ms (wy4 より 61% 遅い)、**チャンク長 16 (wy16)**
  は A/G が C² で効いて 270.5 ms。3 ラウンドを使い切ったのでこれ以上は測って
  いない。既定は off のまま (§2)。

## 1. 前提

- **読んだもの**: `REPORT_P308.md` (と それが参照する P296 / P239 / P227 / P231)。
  P308 の結論は「3,826 の 2.968 s の内訳は 転送 1.31 s (DDR 下限) + ssm 619 +
  mix 262 + full 220 + PLE 147 + combine 65」。P296 は「転送は 3 往復
  67.8 GB / 51 GB/s の床、ssm/mix/full は .cu の仕事」と締めている。
  **この 2 つが今回の出発点**で、ssm のうち `scan_pf` (359 ms = 3,826 の 12%)
  が 0.2 TFLOPS 級 (P296 §5.2) と一番効率が悪いので、そこを取った。
- 基準 COND = タスク指定の 4 本 (`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000
  DENNO_EXPERT_GPU_MIN_HIT=0 DENNO_GPU_WEIGHT_MB=12000`) = int-auto の既定。
  先頭に捨て COND (`x0`)、末尾に baseline の繰り返し (`base2`)。
- **int-auto の移動**: 開始時 `c8b41a29e` → 作業中に `d6f914e97`
  (`merge: br307`、tasks 307/309 が入った)。タスクの指示どおり merge して
  両側を残した (`build_ptx.sh` の `targets_all` で衝突。`gdn_pf_wy4` と
  `exp_k_pf` の両方を残した)。**307 は mix / router / GEMV 系の変更で、
  今回の scan とは独立**なので wy4 の差はそのまま乗る見込み (推定) だが、
  merge 後の再測はしていない (ラウンド上限)。以下の数値はすべて merge 前の
  `c8b41a29e` 系の木で測ったもの。
- ホスト負荷 (`load_before`): t311a 2.02 / t311b 1.91 / t311c 1.71。
  **t311a の wy4 r1** と **t311c の wy4/wy16 r1** は `ssm` の `cpu` 列が
  218 ms に跳ねた外れ値 (同じ COND の r2 は 24 ms)。採否は同じ run の
  2 ラウンドが揃った値で読んだ。

## 2. 変えたこと (4 コミット)

1. `7e6a8fca6` **`kernels_gdn_pf_wy4.cu` (新規) + `DENNO_PF_SCAN_WY4`** —
   WY のチャンクスキャンを 1 ブロック = (1 ヘッド, v 列 32 本) に分割する。
   状態は dk × 32 = 16 KB を共有メモリに置く (smem 全体 7,393 float = 29,572 B、
   48 KB 既定の内側)。チャンク長 C = 8 は wy32 と同じ。q/k の conv・L2 正規化・
   A/G (`k_i·k_j`, `q_i·k_j`) は 4 ブロックが同じ値を計算する (v3 の conv と
   同じ重複。値は同じ)。数値は wy32 と**要素ごとに同じ順序**で、変わるのは
   状態の添字 (128 列 → 32 列 + `voff`) だけ。
2. `4142b4ad0` **wy4 の PTX を収録** + **wy8** (8 分割, 384 ブロック) を追加。
3. `77c8ddf72` **wy16** (4 分割のまま C = 16) を追加。
4. `221ab8786` **wy8 / wy16 を削除** — どちらも wy4 より遅かった (§3)。
   残るのは `DENNO_PF_SCAN_WY4` (既定 off) と `kernels_gdn_pf_wy4.ptx` だけ。

ホスト側: PTX ローダを `gdn_wy_load` に共通化 (wy32 / wy4 で共有)、
`gdn_pf_wy_launch` に起動形状 (vblocks / threads / smem) を引数化 (wy32 は
1 / 512 / 91,012 B のまま)、選択順は **wy4 → wy32 → v3 → v2 → v1**。

## 3. 数値 (lab、すべて実測)

### scan_pf (台帳の段。ms)

| COND | 3,826 r1 / r2 | 14k |
| --- | ---: | ---: |
| base (v3) | 359.0 / 360.2 | 1315.9 |
| base2 (v3) | 360.1 / 359.6 | 1315.5 |
| wy (wy32, 48 ブロック) | 400.4 / 400.5 | 1465.7 |
| **wy4 (192 ブロック)** | **256.5 / 257.1** | **938.2–938.7** |
| wy8 (384 ブロック) | 413.6 / 413.1 | 1518.2 |
| wy16 (192 ブロック, C=16) | 270.5 / 270.3 | 988.7 |

- **wy4 は v3 より -28.6%** (3,826 / 14k とも同じ率)。wy32 は v3 より 11% 遅く、
  t231a の実測 (392.8) を再現した。
- **wy8 は wy4 より 61% 遅い**。8 分割では 1 ブロックの状態仕事が半分になる
  代わりに、A/G (`2·C²·dk`) と q/k conv (`2·dk + dv` チャネル) の重複が
  8 倍になる。固定費が勝つ。
- **wy16 も wy4 より 5% 遅い**。C を上げると A/G の仕事が C² で増える
  (per-token で +8%) ので、バリアが半分になっても回収できない。

### wall (prefill の実測、s)

t311a (`7e6a8fca6`、load 2.02):

| COND | 3,826 r1 / r2 | 14k |
| --- | --- | ---: |
| x0 (捨て) | 9.79 / 2.96 | 11.82 |
| base | 2.984 / 2.991 | 7.453 |
| **wy4** | 3.095 (外れ値) / **2.910** | **7.148** |
| wy | 3.013 / 3.035 | 7.587 |
| base2 | 2.981 / 2.978 | 7.451 |

t311b (`4142b4ad0`、load 1.91):

| COND | 3,826 r1 / r2 | 14k |
| --- | --- | ---: |
| x0 (捨て) | 6.19 / 2.96 | 10.99 |
| base | 3.70 / 2.95 | 9.18 |
| **wy4** | **2.88 / 2.84** | **7.08** |
| wy8 | 3.19 / 3.00 | 7.59 |
| base2 | 2.95 / 2.94 | 7.40 |

t311c (`77c8ddf72`、load 1.71):

| COND | 3,826 r1 / r2 | 14k |
| --- | --- | ---: |
| base | 4.99 / 2.97 | 10.22 |
| **wy4** | 3.61 (外れ値) / **2.86** | 8.94 (位置バイアス) |
| **wy16** | 3.10 / 2.87 | 7.17 |
| base2 | 3.06 / 2.99 | 7.47 |

- **3,826: base2 2.945 → wy4 2.86 = -2.9%** (t311b)。t311a は r2 どうしで
  2.978 → 2.910 (-2.3%)、t311c は 2.99 → 2.86 (-4.3%)。**3 ラウンドの中央値は
  -2.9%**。
- **14k: base2 7.40 → wy4 7.08 = -4.3%** (t311b)。t311a も 7.451 → 7.148
  (-4.1%)。**2 ラウンドとも -4.1〜-4.3%**。t311c の wy4 8.94 は、直前の
  base が 10.22 だった位置 (page cache が冷えている) の外れ値。
- decode: 17.9 → 18.0 ms/tok (+0.3%)。14k 後 decode も 17.3 → 17.3。
- errors は全 COND 0、text は全 COND `looks ok`。

### 目標の判定

| 条件 | 目標 | 実測 | 判定 |
| --- | --- | --- | --- |
| 3,826 cold | -5% 以上 | **-2.9%** (t311b) | **未達** (2.1 pt) |
| 14k cold | 悪化しない | **-4.3%** | OK (ただし -5% に 0.7 pt 届かず) |
| 短 decode | +2% 以内 | +0.3% | OK |
| errors / text | 0 / sane | 0 / sane | OK |
| md5 | 意図した丸めだけ DIFF | wy4 = wy32 (全出力一致)、v3 とは DIFF | OK |

## 4. 機序 — なぜ 4 分割だけが効くか

- `ssm` の GPU 時間は 534.7 → 431.6 ms (3,826)、1957.3 → 1558.2 ms (14k)。
  **差は `scan_pf` の差とほぼ同じ** (103 ms / 377 ms)。wall の差 (-80 / -305 ms)
  は、scan 以外の段が同じだけ残るので少し小さい。
- wy32 は 48 ブロック (170 SM の 28%) で、1 ブロックが dk × dv = 128 × 128 の
  状態を共有メモリに持ち、P/QP (`2·C·dv·dk`) と状態更新 (`C·dk·dv`) を
  dv = 128 列ぶん回す。4 分割すると 1 ブロックのその仕事が 1/4 になる
  (192 ブロックで SM をほぼ埋める)。
- ただし分割しても減らない仕事が 2 つある: **A/G** (`2·C²·dk`、列に依存しない
  ので 4 ブロックが同じ計算を重複) と **q/k の conv** (同じく 128 + 128 チャネル
  を各ブロックが持つ)。8 分割にするとこの固定費が 2 倍になり、状態仕事の
  減少を上回る (413 ms)。C = 16 も A/G が C² で増える (270 ms)。
- **残る `scan_pf` 256 ms はまだ 1.5 TFLOPS 級**で余白がある。1 チャンクあたり
  P/QP と状態更新で `3·dv·dk` = 12,288 MAC を必ず払い、バリアが 7 個ある。
  次に削るならこの 2 つ (§8)。

## 5. なぜ 5% に届かないか

3,826 の wall 2.86 s の内訳 (t311b の `[q4ledger-pf]` と `[pf-host]`):

| 段 | ms | 律速 |
| --- | ---: | --- |
| expert 転送 (写し 908.7 + H2D 待ち 409.2) | 1,318 | **DDR 3 往復 67.8 GB / 51 GB/s の床** |
| ssm (scan 256 / qkv 142 / out 44) | 442 | GPU |
| mix | 264 | GPU (205) |
| full | 208 | GPU (130) |
| PLE 注入 | 142 | ホスト |
| combine | 58 | GPU |
| その他 (embed / accum 同期 / plan) | ~230 | |

- 転送は P227 / P296 が 4 経路とも棄却した床で、今回も触っていない。
- **scan を 0 にしても -9%** なので、scan の -28.6% では wall -2.9% が上限。
  14k は scan の比率が高い (938 / 7,080 = 13%) ので -4.3% まで来たが、
  あと 0.7 ポイント足りない。
- 3,826 で 5% を取るには、scan の残り 256 ms をもう一段 (150 ms 以下) 削るか、
  PLE + combine + mix/full の露出ホスト (合計 ~270 ms) を崩す必要がある。

## 6. md5 とテキスト

- **md5 が wy4 / wy8 / wy32 で全 6 出力一致** (t311a と t311b)。これは
  カーネルの写しが**要素ごとに同じ順序**であることの実測で、v3 とは
  縮約の木が違うので DIFF (wy32 を既定 off のまま残している理由と同じ)。
  wy16 はチャンク境界が変わるので wy4 とも DIFF。
- 変わった md5 (t311b): c2000_cold `d8706fac` → `a91e17da`、
  c2000_turn2 `94f77974` → `5a8f5488`、c8000_cold `e2292eb6` → `e2292eb6` (SAME)、
  c8000_turn2 `3cf204ea` → `561c6da8`。
- 本文 (wy4 の 14k cold、TXTD):
  「ご提示いただいた文書は、LLM推論エンジン（おそらく \`denno-engine）の
  **Prefill（事前計算）における性能最適化の記録です。…### 1. 核心的な結論」
  — base と同じ書き出しで、内容も同じ。TXT (B-tree) / TXTB (Rust の所有権) は
  全 COND で同一。

## 7. テスト

```
CARGO_TARGET_DIR=/tmp/t311 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1,209 passed / 0 failed / 102 ignored   (int-auto d6f914e97 の merge 後)
CARGO_TARGET_DIR=/tmp/t311 cargo test --release -p denno-engine --features cuda --test kernel_invariants -- --test-threads=1
  -> 96 passed / 0 failed
```
追加した 4 本:
- `pf311_scan_wy4_gate_defaults` / `pf311_scan_wy4_shape_gate` /
  `pf311_scan_wy4_smem_matches_source` — 真理値表 (既定 off、明示的な偽値だけ
  off)・形のゲート (dv == 4 × 32)・smem の区画 (重なりなし、29,572 B、
  最後の区画の終端 = WY4_TOTAL)。
- `task311_gdn_pf_wy4_kernel_is_wired` (kernel_invariants) — カーネル名 /
  17 引数 (wy32 と同じ ABI) / 4 分割で 128 列 / smem の並び / 起動形 /
  build_ptx.sh の target。
**発火の証拠**は lab のログ: `scan_pf` が 359 → 256 ms に落ち、md5 が wy32 と
一致した (v3 のままなら 359 で wy32 とは DIFF のまま)。
未検証: このコンテナには nvcc が無いので、PTX は lab が生成したものを収録した
(g++ スタブで .cu の構文検査だけ行った)。

## 8. GPU で確認してほしいこと / 次にやるなら

1. 実機 (8098) の 3,826 で `ssm scan_pf` が 359 → 256 ms、14k で 1316 → 938 ms
   になること。`DENNO_PF_SCAN_WY4=1` で有効 (既定 off)。
2. **採らない判断**: 目標 -5% に 14k で 0.7 ポイント届かない。既定 off のまま
   残したので、切り戻しは不要。
3. 次に削るなら `scan_pf` の残り 256 ms。候補は (a) P/QP と状態更新の
   `3·dv·dk`/チャンクを減らす (S_0 をチャンクごとに読み直さず、状態を
   レジスタに持つ v3 型の配置と組み合わせる)、(b) A/G の重複を v 分割側で
   共有する (4 ブロックが同じ A/G を計算している)。
4. 14k は -4.3% なので、**mix の mixed D2H** (14k で 6.9 GB のうち full 12 層
   ぶん 1.7 GB は誰も読まない) と組み合わせれば 5% に届く可能性がある (推定。
   3,826 で 37.9 ms / 14k で 135.9 ms が露出している)。

## 9. lab の VERDICT 行

t311a (`7e6a8fca6`):
```
VERDICT base vs x0 OK decode +0.4% prefill +0.0% md5 SAME errors 0
VERDICT wy4 vs x0 OK decode +1.1% prefill -0.8% md5 DIFF errors 0
VERDICT wy vs x0 OK decode +0.4% prefill +0.0% md5 DIFF errors 0
VERDICT base2 vs x0 OK decode +0.9% prefill -0.8% md5 SAME errors 0
VERDICT_DEPTH base c8000_cold vs x0 decode +0.6% prefill -1.6%
VERDICT_DEPTH wy4 c8000_cold vs x0 decode +0.0% prefill -5.5%
VERDICT_DEPTH base2 c8000_cold vs x0 decode +0.0% prefill -1.6%
VERDICT baseline x0 errors 0 text ok
```

t311b (`4142b4ad0`):
```
VERDICT base vs x0 OK decode -1.7% prefill -4.0% md5 SAME errors 0
VERDICT wy4 vs x0 OK decode -1.3% prefill -4.8% md5 DIFF errors 0
VERDICT wy8 vs x0 OK decode -1.9% prefill -4.0% md5 DIFF errors 0
VERDICT base2 vs x0 OK decode -0.9% prefill -4.0% md5 SAME errors 0
VERDICT_DEPTH base c8000_cold vs x0 decode -16.2% prefill -16.5%
VERDICT_DEPTH wy4 c8000_cold vs x0 decode -16.8% prefill -35.6%
VERDICT_DEPTH base2 c8000_cold vs x0 decode -16.2% prefill -32.7%
VERDICT baseline x0 errors 0 text ok
```

t311c (`77c8ddf72`):
```
VERDICT base vs x0 OK decode -0.9% prefill -3.2% md5 SAME errors 0
VERDICT wy4 vs x0 OK decode +0.0% prefill -4.0% md5 DIFF errors 0
VERDICT wy16 vs x0 OK decode -0.7% prefill -4.0% md5 DIFF errors 0
VERDICT base2 vs x0 OK decode -0.7% prefill -3.2% md5 SAME errors 0
VERDICT_DEPTH base c8000_cold vs x0 decode +0.6% prefill -13.5%
VERDICT_DEPTH wy4 c8000_cold vs x0 decode -11.6% prefill -24.4%
VERDICT_DEPTH wy16 c8000_cold vs x0 decode -7.1% prefill -39.3%
VERDICT_DEPTH base2 c8000_cold vs x0 decode -10.5% prefill -36.8%
VERDICT baseline x0 errors 0 text ok
```
(VERDICT はどれも最初の COND = 冷えて負荷のある x0 との比較なので、採否は
同じ run の base2 との対で読んだ — §3 の表。)
