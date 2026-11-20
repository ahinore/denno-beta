ADOPT: no t526a t526b t526c

# REPORT P526 — UD-Q4_K_XL: 層あたりの固定費 (seg 経路) に効く 3 つを、いまの int-auto で一緒に測り直した。
# 融合 (`DENNO_SEG_GU_FUSE`) は狙いどおり発火する (起動 880.9 -> 785.0/token = -2 x 48 層、
# 層の GPU は gate+up+silu 39 us -> 36 us) が、壁は **-0.9% (t526a) と -4.4% (t526b)** で
# 2 ラウンドとも 2% に届かない。D2H の 2 つ (`DENNO_SEG_SUM_PIN` / `DENNO_SEG_SUM_ASYNC`) は
# d2h を 7.5 -> 2.5-3.8 us/層 に縮めるが壁に 1% も出ない (P498/P502 と同じ)。
# **採らない。3 つとも既定 off のまま** (既定経路は 1 命令も変わらない)。

branch: **br526** = 開始時の int-auto **890fc66 (merge: br496)**。`br502` (br498 を含む) を
merge した (DENNO_SEG_SUM_PIN + DENNO_SEG_SUM_ASYNC。どちらも既定 off)。br412 は int-auto
から 1,000 コミット以上離れているので、**融合カーネルと起動分岐だけを手で載せた** (下 §2)。
終端で int-auto を fetch し直したが **890fc66 のまま** なので merge は無い。

lab 名 (すべて `tree=denno-dev3` / `SCENARIO=depth8k` / `PROFILE=1`) / baseline COND =
タスク文のとおり `DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0
DENNO_GPU_WEIGHT_MB=12000`:

| lab | SHA | MODEL | PTX | load_before | COND |
| --- | --- | --- | --- | ---: | --- |
| **t526a** | `fa9df8f` | flash-next-UD-Q4_K_XL | `seg_k` | 2.26 | x0(捨) b **gu** **pin** **asy** **all** |
| **t526b** | `a7378ba` | flash-next-UD-Q4_K_XL | none (コミット済 PTX) | 1.77 | x0(捨) b gu pin asy all (全部 `DENNO_SEG_PROF=1` 付き) |
| **t526c** | `a7378ba` | **flash-next-UD-IQ3_XXS** (既定) | none | 2.13 | x0(捨) b gu all |

## 0. 結論 (6 行)

1. **採らない (`ADOPT: no`)。** 3 つの同時 (**all**) は t526a で **-1.2%** (13.60 -> 13.44 ms/tok)、
   t526b で -3.4% (14.46 -> 13.87) だが、t526b は **b 自身が同じ env の x0 より 3.0% 遅い回**
   (14.46 対 14.03) で、その回の差は読めない。**2 ラウンドとも 2% に届かない**ので規則
   (medians of at least two rounds by at least 2 percent) を満たさない。
2. **融合 (`DENNO_SEG_GU_FUSE`) は狙った枝が発火している。** 起動 **880.9 -> 785.0/token**
   = **-95.9 = 2 起動 x 48 層** (t526b の `[q4ledger] launches=`)。`[seg-prof]` の
   層の GPU は gate 17.9 + up 18.2 + silu 3.2 = **39.3 us** -> **融合 1 本 33.5-33.9 us**
   (+ up/silu の残り 1.8-3.0) で、**-3〜-5 us/層**。ホストの起動は 6.8-7.5 -> 4.3-4.5 us/層。
3. それでも壁が動かないのは **P412 と同じ理由** (§5): seg 層の壁は GPU (76 us) と
   ミス CPU/待ちの重なりで決まっており、起動 2 本と GPU 3 us は `sync` (17-44 us/層) の
   余白に吸われる。**理論値でも -0.26 ms/token (GPU) + -0.12 ms (起動) = -0.38 ms = -2.7%
   が上限**、実測はその半分以下。
4. **D2H の 2 つは d2h だけ縮めて壁に出ない。** `DENNO_SEG_SUM_PIN`: d2h **7.3-10.9 -> 7.5-7.7 us/層**
   (この run では縮まない。P498 の 12-17 -> 1.1-1.5 とは条件が違う)、`DENNO_SEG_SUM_ASYNC`:
   **7.3-10.9 -> 2.5-3.8 us/層** (-4〜-5 us/層 = -0.2 ms/token 相当)。decode は
   asy 13.60 -> 13.48 (t526a) / 14.46 -> 13.90 (t526b) で、b との差は 1% 内外。
   **`all` で pin は calls=0** (async が先に写す) なので、3 つ同時は実質 gu+asy。
5. **既定モデル (t526c) は 1 命令も変わらない。** `DENNO_SEG_GU_FUSE=1` / `all` でも
   **起動数が 832.9/token で b と同一**、decode 10.6 = 10.6、**md5 全コンテキスト SAME**。
   既定モデルの expert は IQ3_XXS 系で seg K 経路に乗らないため、融合カーネルは 1 回も
   起動しない (ゲートは不活性)。**これはラウンドの実測で示した** (P506 のコード一致ではない)。
6. **したがって 3 つとも既定 off のまま**残す (`all` を既定にしても 2% に届かない)。
   次の一手は P506 §5 のとおり **seg 層一括 GEMV の帯域 (`.cu`)** で、本タスクの結果は
   その判断を変えない (§6)。

## 1. なぜこの 3 つを選んだか (Do 1 / Do 2) と、測り直す理由

- タスク文のとおり、P506 の段プロファイルでは **experts が warm short decode の 39%** で
  最大。P498 がその中身を **層の固定費 約 85 us/層 (4.1 ms/token) + ミス 57 us/回** に割り、
  **固定費のうち起動 2 本と D2H は「測って棄却」のまま残っていた**。
- 棄却時の前提が変わった: task 412 の時点は **ミス CPU が約 292 us/回 x 約 18 回/token** で
  experts 段の主役だった。いまは **ミス 57 us** まで落ちた (P498) ので、
  **層の固定費の相対比が上がった**。これが「もう一度だけ測る」新しい理由。
- br412 は int-auto から遠い (task 412 は 775429c 起点) ので **merge せず、diff を読んで
  融合の部分だけ手で載せた**。`DENNO_SEG_DN_ROW` (-43%) と `DENNO_SEG_K8` (-0.6%) は
  **載せていない** (前者は指定どおり除外、後者は本タスクの対象外)。

## 2. 変えたこと

1. `crates/denno-engine/kernels/kernels_seg_k.cu`: `dno_seg_gemv_gu_{q4k,q5k,q8_0,q5_1}` を
   **追加のみ** (既存 6 本は 1 バイトも変えない)。1 warp = `SK_GU_ROWS`=2 行 x (gate, up) を
   回し、`silu(g)*u` を act へ直接書く。行ごとの加算順は `dno_seg_gemv_k_*` と同じ
   (ブロック昇順のアキュムレータ 1 本 + 同じ `sk_reduce` の木) なので **行の値はビット一致**。
   `dno_seg_gemv_dn_*` / `dno_seg_gemv_k8_*` は載せていない。
2. `crates/denno-engine/src/cuda.rs`:
   - `SegKKernels` に `gu: [Option<CUfunction>; 4]` (古い PTX では `None` = 従来 3 起動へ落ちる)。
   - `SEG_K_GU_NAMES` / `SEG_WARPS=4` / `SEG_GU_ROWS=2` / `DENNO_SEG_GU_FUSE` (純関数 + OnceLock、
     **既定 off**)。
   - `moe_seg_layer_sum` の中: `ty_g == ty_u && rb_g == rb_u` かつゲート on のとき
     gate/up/silu の 3 起動を **1 起動**に置き換える (`if fuse_gu.is_none() { ... }` で従来経路は
     そのまま残す)。
3. `br502` を merge (P498 の `DENNO_SEG_SUM_PIN` と P502 の `DENNO_SEG_SUM_ASYNC`。
   どちらも既定 off、`serve_flash.rs` の起動行と終端カウンタ付き)。
4. テスト 3 本: `seg_fuse_gate_default_off` / `seg_fuse_shape_matches_cu`
   (cu の `#define SK_GU_ROWS` と Rust の `SEG_GU_ROWS` の一致) /
   `seg_fuse_ptx_has_new_entries` (PTX の 4 エントリと既存 4 本の残存)。
5. コミット済みの `kernels_seg_k.ptx` は **lab t526a が PTX=seg_k で再生成したもの**。
   差分は **3,879 行の追加のみ (削除 0)** = `dno_seg_gemv_gu_*` 4 本だけ。

## 3. lab の結果

### 3.1 t526a (`fa9df8f`、UD-Q4_K_XL、ptx=seg_k、2 ラウンド交互)

| COND | 短い decode ms/tok (28 tok -> 128 out、4 サンプル中央値) | 3,826 cold | 14k cold | md5 (長い 3 本) |
| --- | ---: | ---: | ---: | --- |
| x0 (捨て) | 14.03 | 17.77 | 23.03 | — |
| b | 13.60 | 16.80 | 19.59 | SAME |
| gu | **13.48 (-0.9%)** | 15.77 | 22.04 | SAME (長い 3 本) |
| pin | 13.60 (0.0%) | 15.68 | 20.54 | SAME |
| asy | **13.48 (-0.9%)** | 16.62 | 19.94 | SAME |
| all | **13.44 (-1.2%)** | 16.22 | 19.85 | SAME |

```text
t526a: VERDICT b vs x0 OK decode -2.5% prefill -13.4% md5 SAME errors 0
       VERDICT gu vs x0 OK decode -2.7% prefill -13.4% md5 DIFF errors 0
       VERDICT pin vs x0 OK decode -1.9% prefill -12.7% md5 DIFF errors 0
       VERDICT asy vs x0 OK decode -3.3% prefill -12.7% md5 DIFF errors 0
       VERDICT all vs x0 OK decode -3.3% prefill -11.2% md5 DIFF errors 0
```

(x0 は捨て COND = 最初の 1 本は必ず遅い。採否は **b と比べる**。md5 の DIFF は
48 token の `b` と `c2000_turn2` だけで、これは P490/P494/P498 が記録した **既存の揺れ**
(同じバイナリの COND 間でも変わる)。長い 3 本 `c2000_cold` / `c8000_cold` / `c2000_turn2` は
全 COND SAME。)

### 3.2 t526b (`a7378ba`、コミット済 PTX、全 COND `DENNO_SEG_PROF=1`)

| COND | 短い decode ms/tok | launches/token | d2h us/層 | 層の GPU (gate+up+silu) | 起動 us/層 |
| --- | ---: | ---: | ---: | ---: | ---: |
| x0 (捨て) | 14.03 | 880.9 | — | — | — |
| b | 14.46 | 880.9 | 7.3-10.9 | 39.3 (17.9 + 18.2 + 3.2) | 6.8-7.5 (5 本) |
| gu | **13.83 (-4.4%)** | **785.0 (-95.9)** | 8.4-8.8 | **33.5-33.9 (融合 1 本)** | 4.3-4.5 (3 本) |
| pin | 14.02 (-3.0%) | 880.9 | 7.5-7.7 | 38.0 | 7.2-7.3 |
| asy | 13.90 (-3.9%) | 880.9 | **2.5-3.8** | 39.0 | 6.9-7.2 |
| all | **13.87 (-4.1%)** | **785.0** | **2.8-3.5** | 35.5-36.0 | 4.3-4.9 |

- **b は同じ env の x0 より 3.0% 遅い** (14.46 対 14.03)。x0 を基準にすると gu -1.4% /
  all -1.2% で、**t526a と同じ帯**。この run の b 対比 (-4%) は b 側の遅れを含む。
- `[seg-prof]` の生値 (tok=651-653、48 層):
  - b: `launch=6.8-7.5/5 sync=23.8-43.8 d2h=7.3-10.9 | gate=17.9 up=18.2 silu=3.2 down=33.6 sum=3.1`
  - gu: `launch=4.3-4.5/3 sync=16.8-43.7 d2h=8.4-8.8 | gate=33.9 up=1.0 silu=0.7 down=34.3 sum=4.7`
  - pin: `d2h=7.5-7.7` / asy: `d2h=2.5-3.8` (`[seg-async] calls=... fallbacks=0` が全ラウンド)
  - all: `launch=4.3-4.9/3 d2h=2.8-3.5`、`[seg-pin] calls=0` (async が先に写す)。
- **融合が発火した証拠** は 3 つ揃っている: 起動数 -95.9/token、`[seg-prof]` の up/silu 区間が
  1-2 us に潰れる、ptx.log に `dno_seg_gemv_gu_*` を nvcc が吐いた。

### 3.3 t526c (既定モデル flash-next-UD-IQ3_XXS、1 ラウンド)

| COND | decode ms/tok | launches/token | md5 |
| --- | ---: | ---: | --- |
| x0 (捨て) | 10.6 | 832.9 | edaa84ad ... |
| b | 10.6 | 832.9 | 全部 SAME |
| gu | 10.6 | **832.9 (b と同一)** | 全部 SAME |
| all | 10.6 | **832.9** | 全部 SAME |

`VERDICT gu vs x0 OK decode +0.0% prefill -1.8% md5 SAME errors 0` /
`VERDICT all vs x0 OK decode +0.0% prefill -1.8% md5 SAME errors 0`。
**ゲートを立てても起動数が 1 本も減らない = seg K 経路に乗っていない = 既定モデルは不変。**

### 3.4 出力テキスト (健全性)

- UD の gu / all の 3,826 回答 (TXTD) は b と同じ趣旨の日本語:
  「ご提示いただいた文書は、LLM推論エンジン（`denno-engine）における **Prefill（事前計算
  フェーズの性能最適化の歴史と教訓を詳細に記録したものです。」(以降も b と同文)
- 短い TXTB も同趣旨 (Rust の所有権とライフタイムの説明)。`errors 0` は全 COND 全ラウンド。

## 4. 数値の扱い

- すべて GPU ラボの実測。このコンテナは GPU 無しなのでマイクロベンチは取っていない。
- 短い decode は **`decode` 秒 / out token** で計算した (REQ 行の ms/tok は整数丸めなので使わない)。
  1 COND = 2 ラウンド x (a1, a2) の 4 サンプル。
- **run を跨いだ比較はしない** (b と gu は必ず同じ run の中)。t526b の b は x0 より 3% 遅く、
  run 内でも ±2% 動く。**2% の採否を 1 サンプルで決めない**ための §3.2 の注記。
- `[seg-prof]` の `miss=` と `alloc=` は GPU 待ちを含む包括区間なので使っていない (P498 §3.2 と同じ)。
- 帯域 (GB/s) は `[seg-prof]` の実バイト / GPU 実時間。

## 5. 見積り (なぜ 2% に届かないか)

- 融合 1 本は **GPU -3〜-5 us/層**、**起動 -2.5 us/層** = 48 層で **-0.26 ms + -0.12 ms**。
  decode 14 ms に対して **-2.7% が理論上限**。ただし seg 層の壁は
  `GPU 76 us` と `sync 17-44 us` + ミス CPU の**重なり**で決まっており、GPU と起動の
  短縮はその待ちの余白に吸われる (P412 の「GPU の時間は 1 us も減らなかった」の再現。
  今回は silu の 3 us が消えた分だけ GPU が減った)。実測 -0.9〜-1.4% はこの範囲。
- D2H の短縮 (asy: -4〜-5 us/層 = -0.2 ms/token) も同じ余白に入る。**同期コピーが
  背景の上げ直しの待ちを吸収している** という P506 §5-2 の読みを裏づけた。
- pin が P498 の 1.1-1.5 us/層 まで縮まないのは、この run の d2h が既に 7.5 us/層で
  あったため (P498 の 12-17 us/層 は task 496 前の値)。**ゲートの実装は生きている**
  (`[seg-pin] calls>0 fallbacks=0`) が、縮める余地が小さかった。

## 6. 次の担当者へ

1. **seg 経路の層の固定費はこれで打ち止め** (起動 5 -> 3、D2H 非同期/pinned、ミス CPU の重ね打ち)。
   残るのは **GPU の 76 us/層** で、内訳は gate 17.9 / up 18.2 / silu 3.2 / down 33.6 / sum 3.1。
   帯域は gate 445-473 / up 460-469 / down 338-351 GB/s = 5090 の 20-26%。
   **P506 §5-1 と同じ結論: 3 行列を 1 起動にして down を n 方向に割る `.cu`**。
   本タスクの融合で「1 起動にすると GPU も 3 us 減る」ことは実測できたので、
   down 33.6 us を n 方向に割る版の見込みは立つ (25 us/層なら decode -0.4 ms)。
2. `DENNO_SEG_GU_FUSE` は **既定 off のまま**残した。2% の線に対して -1% 前後で、
   規則を満たさない。次の `.cu` タスクで down まで融合するなら、そのとき一緒に
   A/B して採否を決めるのが筋 (単独では 2% に届かない)。
3. `DENNO_SEG_SUM_PIN` / `DENNO_SEG_SUM_ASYNC` は **既定 off のまま**。d2h は縮むが壁に出ない
   (P498 / P502 の再現)。`all` で pin は calls=0 になるので、**両方 on にする意味は無い**。
4. prefill はこの 3 ラウンドとも 3,826 cold 15.7-17.8 s / 14k cold 19.6-23.0 s (UD) で、
   ディスク律速のまま (P506 §0-6)。本タスクの変更は prefill に 1 命令も触れていない
   (§3.1 の prefill 差は run 内のノイズ)。

## 7. テスト

```text
CARGO_TARGET_DIR=/tmp/t526 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1361 passed; 0 failed; 103 ignored
CARGO_TARGET_DIR=/tmp/t526 cargo test --release -p denno-engine --features cuda --test kernel_invariants
  -> 144 passed; 0 failed
CARGO_TARGET_DIR=/tmp/t526b cargo build --workspace --release          (cuda feature 無し)
  -> 成功
CARGO_TARGET_DIR=/tmp/t526b cargo test --workspace --release -- --test-threads=1   (cuda feature 無し)
  -> 成功
```

既知の赤 (`qwen4exp_decode spec_batch_stops_at_eos`、`qwen4exp_params` の validate 2 件) は
このフィルタでは出ていない。`.cu` は g++ の構文スタブで `-fsyntax-only` を通した
(変更前のファイルも同じスタブで通ることを先に確認。**レジスタ数・smem・本当の PTX は未検証**)。

## 8. GPU で確認してほしいこと

1. 次の `.cu` タスクでは **down (33.6 us/層、345 GB/s) を n 方向に割る / gate+up+down を
   1 起動にする** のを見てほしい。本タスクで「seg 層の起動融合だけで -2 x 48 本、GPU -3 us/層」は
   実測済みなので、down が 25 us/層になれば decode -0.4 ms (-3%) が射程に入る。
2. `DENNO_SEG_GU_FUSE=1` を単独で既定にするかは、**負荷の低い回 (load_before 1.5 以下) で
   b と gu を 4 ラウンド以上**回して決めてほしい。本タスクの 2 ラウンドは -0.9% と -4.4% に
   割れており、この精度では 2% を判定できない。
