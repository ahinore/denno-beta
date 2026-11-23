ADOPT: no t480b

# REPORT P480 — 既定モデル decode: expert GEMV の形を 2 方向に振った。
# **どちらも採らない** (下+和の 1 起動化は down が 16.7 -> 31.2 us/層、
# 1 行 4 warp 版は gate+up が 21.9 -> 35.9 us/層)。
# 生き残った知見は「experts 段の壁は起動本数ではなく、**行あたりの並列度**」。

branch `br480` = 開始時の `int-auto` **8455128ec** (= merge: br465)。br476 は
`ADOPT: no` (既定 off のまま) なので取り込んでいない。終端で fetch し直したら
int-auto が **b89a08a0e (merge: br474)** に動いていたので merge した
(`318487fe2`、衝突なし。br474 は prefill 側で、このタスクの decode とは接しない)。

数値はすべて GPU ラボの実測 (`SCENARIO=depth8k`、`PROFILE=1`、モデルは
`flash-next-UD-IQ3_XXS`、ベースライン COND = int-auto の既定 +
`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0
DENNO_GPU_WEIGHT_MB=12000`、先頭に捨て COND、末尾に base の繰り返し)。
lab 名と測った SHA: **t480a = `d421692e2`** (down+sum 融合の A/B、`PTX=exp_iq_ds`)、
**t480b = `023fbf7ff`** (1 行 4 warp 版の A/B、`PTX=exp_iq_ds`)。
`RESULT.md` 2 行目は 2 run とも `model=flash-next-UD-IQ3_XXS`、`load_before` は
1.60 / 8.68 (t480b はホストが重い回。base と base2 が一致しているので比較は成立)。

## 0. 結論 (7 行)

1. **採らない (両方)。** (a) down GEMV と重み付き和を 1 起動に畳む `ds` は
   **down 16.7 -> 31.2 us/層 (+87%)**、experts 段 2.88 -> 3.9 ms/token、
   decode 11.1 -> 11.9 (+6.6%)。`ds8` (8 行/ブロック) でも 28.8 us で同じ負け。
   (b) 1 行を 4 warp で分担する `w4` は **gate+up 21.9 -> 35.9 us/層 (+64%)**、
   decode 11.1 -> 17.1 (+53%)。こちらは**値も壊れた** (下記 §4.2) ので、
   ゲート+up の遅さだけを読んでも負けは動かない。
2. **負けの理由は数字で説明できる。** experts の GEMV は
   `1 行 = 1 warp` で **1 warp が同時に持つ未完了ロードが 1-2 本**。
   ds は warp 数を 1/10 にし、w4 は warp を 4 倍にしたが 1 warp の仕事を
   1/4 にした (ブロックの 256 要素を 32 レーン x 8 で既に使い切っているので、
   行を割っても 1 warp あたりのロード本数は増えない)。
   **どちらも「warp 数 x warp あたりの未完了ロード」を増やせていない**
   (P476 §7-1 の勘定と同じ)。実測の帯域は 446-619 GB/s で変わらないか悪化。
3. **起動を 1 本消す価値より、形を崩す損失の方が大きい。** `sum` は 4.2 us/層、
   `silu` は 3.3 us/層 (2,560 要素 / 20 KB のカーネル) で、これはほぼ
   **起動とイベント計測の固定費**。畳めば 7.5 us/層 (0.36 ms/token) だが、
   畳む形 (1 warp に複数行 / 1 行を複数 warp) はどちらも 3-9 us/層を失う。
4. **既定は変えていない。** 2 経路とも**コードごと取り除いた** (既定 off の
   死んだ経路を残さない。測った数字はこの REPORT に残す)。したがって
   `br480` のツリーは int-auto + この REPORT だけで、モデルの挙動は不変。
5. **段の新しい実測 (t480a base、warm の 128 token、ms/token)**: mix 2.58 /
   ssm 2.79 / experts 2.88 / full 1.18 / shared 0.63 / head 0.60 / router 0.17 /
   combine 0.04 / sample 0.02、**step 11.08**。P472 の段 (mix 2.57 / ssm 2.79 /
   experts 2.89 / full 1.18 / shared 0.63 / head 0.60 / router 0.17、step 11.07) と
   **同じ**。int-auto は decode の段構成を変えていない。
6. **experts 段の内訳 (t480a base、warm、[dec-exp-kprof]、us/層)**:
   gate+up 21.9 / up 1.6 / silu 3.3 / down 16.7 / d2h 0.4 / sum 4.2 = 48.1。
   型ごとの帯域 ([exp-iq-kprof]) は IQ2_S 446 / Q2_0(down) 385 / IQ3_XXS 499 /
   IQ4_XS 619 GB/s。**experts の GPU は 48 層で 2.3 ms/token、段は 2.88** =
   残り 0.58 は起動と同期の隙間。
7. **次に効きそうなのは mix 段の鎖** (2.58 ms/token のうちカーネルは
   0.77 ms = 16 us/層。48 層 x 5 起動/token で 1.8 ms/token が鎖の待ち)。
   experts は形を 2 方向に振って両方負けたので、**同じ方向の再試行はしない**。

## 1. なぜこれを選んだか (Do 1 / Do 2)

- Do 1 の再プロファイル (上の §0-5) で、3 つの大きい段は experts 2.88 /
  ssm 2.79 / mix 2.58 でほぼ並ぶ。**この 1 か月で触られていないのは experts の
  GEMV の形**で、P476 §7-4 が「down は行長 512 の短い GEMV で 1 重みあたり
  gate+up より 1.6 倍遅い」「sum は 10 ブロック = 起動律速で、畳むには down の
  列優先化が要る」と**名指しで次の候補**に挙げていた。
- なので 1 つ目はその指示どおり **down + sum の 1 起動化** (ds)。
- 2 つ目は同じ測定から**逆向きの仮説**を立てた: 同じモデルの mix / ssm の GEMV は
  `1 ブロック 1 行 (128-256 スレッド)` で 700-900 GB/s 出ており、experts の
  `1 行 1 warp` は 446-619 GB/s。差は**行あたりの並列度**ではないか →
  `1 行 4 warp` (w4) を試した。

## 2. 変えたこと (どちらも取り除いた)

### 2.1 ds: down GEMV と重み付き和を 1 起動に (t480a)

- `kernels_exp_iq_ds.cu` (新規、`PTX=exp_iq_ds`) に 6 型ぶんの
  `dno_exp_gemv_ds_<type>`。**1 出力行を 1 warp が nseg (=10) 個ぶん**まとめて
  計算し、`out[j] = sum_t w[t] * dot(down_t 行 j, act_t)` を t 昇順の 1 本の
  アキュムレータで書く。1 行の内積は v2 と同じレーン割り・同じアキュムレータ
  4 本・同じ warp の木、和は `dno_dec_exp_sum` と同じ「積 1 回 + 和 1 回」
  (`__fmul_rn` / `__fadd_rn` を明示) = **ビット一致**。
- `rows` (1 ブロックの行数) は起動形で 4 / 8 を選べる (`blockDim = 32*rows`)。
- 出力は `DENNO_DEC_EXP_SUM_ZC` と同じ pinned のホスト (または VRAM の作業域) へ
  直接書くので、D2H の経路は変えていない。
- 門は新 env `DENNO_DEC_EXPDOWN=ds|ds8` と既存 env の値 `DENNO_DEC_EXPGU=ds|ds8`
  の両方 (lab の COND が新しい env 名を落とすことがあるため。t294b)。

### 2.2 w4: 1 行を 4 warp で分担 (t480b)

- 同じ `kernels_exp_iq_ds.cu` に 6 型ぶんの `dno_exp_gemv_w4_<type>` を追加。
  1 ブロック = 4 行 x 4 warp = 512 スレッド。行の内積は warp ごとに v2 と同じ木で
  縮約し、block 内は warp 番号の昇順に 1 本のアキュムレータで足す
  (**丸め順が変わる** = md5 DIFF になり得る)。
- 引数は v2 と同じ 8 本なので、ホストは関数と blockDim だけ差し替える
  (gate+up 融合 / 非融合の gate・up / down の 3 箇所)。
- 門は `DENNO_DEC_EXPW4=1` と `DENNO_DEC_EXPGU=w4`。

### 2.3 取り除いた理由

どちらも目標に届かず、しかも**形の変更が本質的に負け**なので、既定 off のまま
残すと同じ試行が繰り返される。`git checkout int-auto -- <5 files>` + 新規 .cu の
削除で **int-auto とバイト同一**に戻した (§4 の数字だけを残す)。

## 3. 数値の扱い

- ds: v2 の down と `dno_dec_exp_sum` の合成と**ビット一致**を狙って書き、
  実測でも **3 run 全 COND 全コンテキストで md5 SAME**。狙いどおりだった
  (遅いだけで正しい)。
- w4: 丸め順が変わる (行の内積を 4 本に割る) ので md5 DIFF を想定していた。
  実際は**値が壊れた** (§4.2) ので、丸めの議論以前の不具合が混ざっている。
  不具合は 2 つで、どちらも私の実装漏れ: (a) gate+up 融合の segment 表の
  `out_off` (up 側は `+ng`) を無視して `obase[row]` へ書いた、
  (b) IQ4_NL / Q2_0 は 1 ブロックを 2 / 4 レーンで分担する型なのに、
  w4 の本体が「全レーンが同じブロック」の前提 (LPB=1) で x のオフセットを
  計算していた。**gate+up の遅さ (21.9 -> 35.9 us/層) は (a) では説明できない**
  (仕事量は同じ) ので、形そのものの負けと読んだ。

## 4. lab の結果

### 4.1 t480a (`d421692e2`、ds / ds8、`PTX=exp_iq_ds`)

| cond | decode ms/tok | prefill s | c8000_turn2 | md5 |
| --- | ---: | ---: | ---: | --- |
| c0 (捨て) | 11.2 | 0.56 | 15.2 | SAME |
| base | 11.1 | 0.58 | 12.1 | SAME |
| ds | **11.9** | 0.57 | 12.7 | SAME |
| ds8 | **11.8** | 0.56 | 12.6 | SAME |
| base2 | 11.2 | 0.56 | 12.0 | SAME |

```text
VERDICT base vs c0 OK decode -0.7% prefill +2.7% md5 SAME errors 0
VERDICT ds vs c0 OK decode +6.6% prefill +1.8% md5 SAME errors 0
VERDICT ds8 vs c0 OK decode +5.6% prefill +0.0% md5 SAME errors 0
VERDICT base2 vs c0 OK decode +0.0% prefill +0.0% md5 SAME errors 0
```

`[exp-ds] ds_layers=6029 rows=4 fallback=0` = 分岐は狙いどおり発火した
(6144 層のうち h10 の 6029 層。残りはミスがある層 = nseg≠10 で 2 起動へ)。

**証拠 (`[dec-exp-kprof]`、warm の 6144 層、us/層)**:

| 段 | base | ds (4 行) | ds8 (8 行) |
| --- | ---: | ---: | ---: |
| gate+up | 21.8 | 22.2 | 22.2 |
| silu | 3.2 | 3.3 | 3.3 |
| down (ds は和込み) | 16.6 | **31.2** | **28.8** |
| sum | 4.1 | 0.5 | 0.5 |
| 合計 | 46.3 | 57.7 | 55.3 |

生成した PTX (`done/t480a/ptx/kernels_exp_iq_ds.ptx`) に `ld.local` / `st.local` は
**0 本** = レジスタのスピルではない。warp 数が 2,560 (base は 25,600) に落ち、
1 warp が 10 本のロード列を持っても未完了ロードの総数が半減するため。

### 4.2 t480b (`023fbf7ff`、w4、`PTX=exp_iq_ds`)

| cond | decode ms/tok | prefill s | c8000_turn2 | md5 |
| --- | ---: | ---: | ---: | --- |
| c0 (捨て) | 11.1 | 0.56 | 12.8 | SAME |
| base | 11.1 | 0.57 | 12.6 | SAME |
| w4 | **17.1** | 0.71 | 17.3 | **DIFF** |
| base2 | 11.1 | 0.57 | 12.4 | SAME |

```text
VERDICT base vs c0 OK decode +0.0% prefill +0.9% md5 SAME errors 0
VERDICT w4 vs c0 FAIL decode +53.3% prefill +26.5% md5 DIFF errors 0
VERDICT base2 vs c0 OK decode -0.4% prefill +1.8% md5 SAME errors 0
```

`[exp-w4] w4_launches=12282 wpr=4` = 分岐は発火した。`[dec-exp-kprof]` の warm は
**gate+up 35.9 us/層 / down 119.5 us/層** (base は 21.9 / 16.7)。
**`[exp-iq] hit_rate 0.835 -> 0.425`** (t480b の base vs w4) = 壊れた値でルータの
選択が変わり、VRAM のヒット率が落ちた。したがって decode +53% のうち
どこまでが形のせいかは切り分けられない。**切り分けられるのは gate+up の
+64% だけ**で、これが形の負けの根拠。テキストは `TXTB w4` が
`R  2 2 2 2 2 2 2 ...` で **BROKEN** (base は健全。下記 §5)。

## 5. 出力テキスト (健全性)

- t480a: 全 COND・全コンテキストで md5 SAME、errors 0。base の `TXTD`:
  `ご提示いただいた文書は、LLM推論エンジン（`denno-engine`）における **Prefill（事前計算）フェーズの性能最適化の歴史を詳細に記録した技術レポートです。`
- t480b: base/base2 は健全 (上の `TXTD` と同じ)。w4 は `TXTB` が `R  2 2 2 2 2 2...`、
  `TXTA` の a1/a2 が別 md5 (`31ed8740` / `d807cd0b`) = 非決定的。**採らない**
  根拠の 1 つ。

## 6. テスト

```text
CARGO_TARGET_DIR=/tmp/t480f cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1343 passed; 0 failed; 103 ignored
同 --test kernel_invariants
  -> 139 passed; 0 failed
```
(最終ツリーは int-auto + REPORT のみ。実験中に足した ds/w4 用のテスト 5 本は
コードごと取り除いたので、本数は int-auto と同じ。`.cu` は g++ のスタブで構文検査、
レーン割りとブロック被覆は Python ミラーで v2 と要素一致を確認した。)

## 7. 次への申し送り (測って駄目だったもの = 再試行しない)

1. **down + sum の 1 起動化は再試行しない。** 1 warp に nseg 行を持たせると
   warp 数が 1/10 になり、10 本のロード列を持っても未完了ロードが半減する
   (down 16.6 -> 31.2 us/層)。8 行/ブロックでも 28.8。
2. **1 行を複数 warp に割るのは再試行しない。** 256 要素のブロックを
   32 レーン x 8 で既に使い切っているので、行を割っても 1 warp のロード本数は
   増えず、縮約の追加だけが残る (gate+up 21.9 -> 35.9 us/層)。
3. **experts の GEMV は形を変えても速くならない** (446-619 GB/s が天井)。
   次に触るなら **バイト数** (読む重みを減らす) か、**層あたりの起動/同期の数**。
4. **mix 段が次の本命**: 2.58 ms/token のうちカーネルは 16 us/層 (down 8.3 +
   up 7.6) だけで、残り ~38 us/層 は鎖の待ちと起動。48 層 x 5 起動/token。
   まず `DENNO_GDN_DEC_KPROF=1` の `[dec-kprof-avg]` で 1 呼び出しの段
   (mix_norm / down / up / mixed / inj / h2d) を実測してから設計すること。
5. **ssm 段**も 2.79 ms/token のうちカーネル 1.7 ms で、~30 us/層が隙間
   (P476 §7-3 の D2H の形)。ここも「起動を減らす」ではなく「待ちを減らす」側。
