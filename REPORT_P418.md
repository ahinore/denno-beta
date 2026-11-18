ADOPT: yes t418b t418c

# REPORT P418 — 既定モデル decode: mix の down GEMV を「1 ブロック 1 行・1 レーン 1 対」に

branch `br418` = 開始時の `int-auto` **ff1a1eccb** (merge: br416)。作業中に int-auto は
**ad1cfe1b5** (merge: br420 = `DENNO_PF_POOL_K_PLAN` 既定 on) へ動いたので merge した
(衝突なし、`build_ptx.sh` も無傷)。**br414 は merge していない**: int-auto に未 merge
だが、本タスクの対象 (mix の down) とは別で、要らなかった (br414 の `attn_dec2` は
lab の `targets_all` 末尾を動かすので、かえって衝突源になる)。
数値はすべて GPU ラボの実測 (lab 名と測った SHA):
**t418a = ff1a1eccb** (取り直しの前提 + `[dec-kprof-avg]` の段内訳)、
**t418b = 4235e7d96** (PTX=mix_dn5、ゲートで A/B)、
**t418c = 2c23e4905** (収録した PTX を既定で測る。`ptx=none`、`DENNO_MIX_QK=v5` が旧コード)。
`RESULT.md` 2 行目は 3 run とも `model=flash-next-UD-IQ3_XXS`、t418b だけ `ptx=mix_dn5`。

## 0. 結論 (5 行)

1. **採る。** 短い decode が t418b: base 13.2 / base2 13.2 → **dn5 12.9 (-2.3%)**、
   t418c: v5 (旧) 13.2 → **base 12.9 / base2 12.8 / kp 12.8 (-2.3〜-3.0%)** の 2 ラウンド。
   14k 後 (c8000_turn2) も t418b **15.4/14.8 → 14.4**、t418c **14.8 → 14.5** で -2.0〜-2.7%。
2. 段の動きは mix だけ: `[q4ledger]` の mix が **3.97-3.98 → 3.53-3.56 ms/token
   (-10.7〜-11.3%)**。full 2.13 / ssm 3.37-3.39 / experts 3.53-3.56 / head 0.73 /
   router 0.18 / shared 0.62 は 1 つも動いていない (0.03 ms 以内)。prefill は 0.56 s で不変。
3. 理由はカーネルの並列度だった。lab t418a の `[dec-kprof-avg]` (band 0) で mix の
   down は **19.4 us / 3.48 MB = 180 GB/s**、grid 160 x block 64 = 10,240 スレッド
   (1 レーンが 5 対を逐次ループ)。同じ band の inject (`dno_matvec_f32_r`、512 x 256 =
   131,072 スレッド) は 13 MB を 7.1 us で読んでいた。**同じ 3.48 MB を 320 ブロック
   x 160 スレッドで 1 対ずつ読む**新カーネルにしたら **15.3 us (-21%)** になり、
   その差がそのまま mix 段 → decode に出た (t418c の kp)。
4. md5 は 3 run の全 COND・全コンテキストで **SAME** (`a:edaa84ad b:1938059f
   c2000_cold:b42d79d2 c2000_turn2:12fc5c36 c8000_cold:672fec07 c8000_turn2:98e879d2`)、
   errors 0。合計順は v5 と変えたが、f32 の丸めの差は出力トークンを 1 つも動かさなかった。
5. 発火の証拠: `[mix-qchain] ... dn5=` を足した。t418c の base は **dn5=12416**
   (= 97 呼び出し x 128 トークン)、`DENNO_MIX_QK=v5` は dn5=0、t418b の base も dn5=0。
   既定は `(Dn5, V5)`、切り戻しは `DENNO_MIX_QK=v5` (または `1` / `both`)。

## 1. 前提: どこを削ることにしたか (task の Do 1/2)

### 1.1 t418a = int-auto ff1a1eccb の取り直し (depth8k、PROFILE=1、load 2.37)

| cond | short ms/tok | c2000_turn2 | c8000_cold | c8000_turn2 | prefill c8000_cold |
| --- | ---: | ---: | ---: | ---: | ---: |
| x0 (捨て) | 13.2 | 17.9 | 22.4 | 20.0 | 9.17 s |
| base | 13.2 | 14.3 | 15.7 | 15.2 | 6.08 s |
| base2 | 13.3 | 14.3 | 15.4 | 14.8 | 5.95 s |

14k turn2 の `[q4ledger]` (ms/token、base/base2): **mix 3.97-4.09** / full 2.13 /
ssm 3.38 / experts 3.53-3.66 / head 0.73 / router 0.18 / shared 0.62 / step 14.76-15.14。

### 1.2 段の内訳 (`DENNO_GDN_DEC_KPROF=1` の `[dec-kprof-avg]`、band 0 = 短い decode)

mix の窓 (1 層 1 回、48.4 窓/token) は wall 89-114 us で、中身は

| op | grid x block | gpu us | 備考 |
| --- | --- | ---: | --- |
| `dno_gemv_q8_0_v5` (down) | **160 x 64** | **19.4** | 3.48 MB = **180 GB/s** |
| `dno_matvec_f32_r` (inject) | 512 x 256 | 7.1 | 13 MB = 帯域なり (これが比較対象) |
| `dno_mix_upmix_q8_0` (up) | 427 x 128 | 10.6 | 3.48 MB = 328 GB/s |
| H2D@a / D2H@a | — | 10.0 / 5.9 | xn 41 KB と mixed 10 KB |

### 1.3 なぜ down を選んだか

- mix は 15.1 ms/token のうち **3.97-4.09 (27%) で最大の段**。他の段は P388/P392/P397/
  P410/P414 で .cu 融合・重ね打ちが入っていて「床」(ssm は GPU 完走待ち、experts は
  IQ の表引き、full は flashdec の計算鎖、head は 963 GB/s の帯域)。
- その mix の中で **down だけが極端に並列度が低い** (10,240 スレッド)。同じ band で
  131,072 スレッドの inject は帯域なりに出ているので、**カーネルの中身ではなく
  スレッド数**が原因だと読めた (推定)。v5 の形 (`lanes=32 / rows=2`) は
  n_out=320 で決まるので、`DENNO_MIX_QK` の既存の値 (blk/r8/seg) では増やせない。
- P382 は「mix は投入 (launch) 律速で、カーネルを速くしても効かない」と書いたが、
  それは**小さなカーネルを畳んだ**話で、3.48 MB を 180 GB/s で読む本体には手が
  入っていなかった。今回それを確かめた (実際、GPU 4.1 us の短縮が wall 4.6 us/call
  として出た = この段は GPU が効く)。

## 2. 変えたこと

### 2.1 `kernels_mix_dn5.cu` — `dno_mix_gemv_q8_0_dn5`

- **grid = n_out (320)、block = 32 * warps (160 = 5 warp)**。1 ブロック = 1 行、
  `pair = warp * 32 + lane` で**1 レーンがちょうど 1 対 (68 B = 64 要素) を 1 回だけ読む**。
  ループも先読みも無い (全レーンが同時に 1 回のロードを出す)。
- 積和は v5 と同じ「Q8_0 の 2 ブロックを 17 ワードで読む」「d0/d1 の f16 展開」
  「`(int8 -> float) * d * x` を `__fmul_rn` / `__fadd_rn` で 1 演算 1 丸め」「4 アキュムレータ」。
- 縮約は warp 内の木 (offset 16,8,4,2,1) → warp ごとの部分和を共有メモリ → スレッド 0 が
  **固定順 (warp 0..4 の逐次加算)** = 決定的。**v5 と違うのは合計順だけ**。
- 門番は純関数 `mix_dn5_warps(n_in)`: Q8_0 / `n_in % 64 == 0` / 対の数が 32 の倍数 /
  warp 数 <= 8。実形 10,240 → 5 warp。合わない形 (up の n_in=320 など) と PTX 欠けは
  v5 へ落ちる (fail closed)。

### 2.2 ゲートと配線

- 既存の `DENNO_MIX_QK` に値 `dn5` / `d5` を足した (**新しい env 名を作らない**)。
  lab の COND は新しい env 名を落とすので、これは効く (t418b で実際に届いた)。
- **既定を `(Dn5, V5)` にした** (down だけ dn5、up は v5 + `DENNO_MIX_UPQ8`)。
  切り戻しは `DENNO_MIX_QK=v5` (または `1` / `both`)、`0` は task 266 より前。
- 発火の証拠: `[mix-qchain]` 行に `dn5=` を足した。t418c base = 12416、v5 = 0。
- `gate_audit` の既定 on 台帳に `kernels_mix_dn5.ptx` / `dno_mix_gemv_q8_0_dn5` の行を
  足した (PTX が欠けると静かに v5 へ落ちる事故を起動 1 行で見る)。
- `build_ptx.sh` に target `mix_dn5` (nofmad) を足し、`targets_all` は**末尾ではなく
  `mix_upmix_q8` の直後**に入れた (末尾は他ブランチが append する場所なので動かさない)。

## 3. lab の結果 (3 ラウンド)

すべて `SCENARIO=depth8k PROFILE=1`、ベースライン COND は
`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0
DENNO_GPU_WEIGHT_MB=12000`。最初の COND は捨て、最後に baseline の繰り返し。

### 3.1 t418b (`4235e7d96`、PTX=mix_dn5、gate off = 旧コード)

| cond | short ms/tok | c2000_cold | c2000_turn2 | c8000_cold | c8000_turn2 | prefill c8000_cold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| x0 (捨て) | 13.2 | 15.2 | 14.4 | 16.8 | 15.4 | 7.47 s |
| base (v5) | 13.2 | 15.0 | 14.3 | 15.8 | 15.4 | 6.68 s |
| **dn5** | **12.9** | **14.6** | **13.9** | **14.9** | **14.4** | 5.97 s |
| base2 (v5) | 13.2 | 14.8 | 14.2 | 15.5 | 14.8 | 5.64 s |

```text
VERDICT base  vs x0 OK decode +0.0% prefill -1.7% md5 SAME errors 0
VERDICT dn5   vs x0 OK decode -2.4% prefill +0.0% md5 SAME errors 0
VERDICT base2 vs x0 OK decode +0.0% prefill +0.0% md5 SAME errors 0
VERDICT_DEPTH dn5 c8000_turn2 vs x0 decode -6.6% prefill -7.3%
VERDICT_DEPTH base c8000_turn2 vs x0 decode +0.0% prefill -5.5%
VERDICT baseline x0 errors 0 text ok
```

段 (14k turn2、ms/token): mix **3.97 / 3.95 → 3.54**、step 15.30 / 14.80 → 14.36。
他の段は不変 (full 2.13、ssm 3.38-3.39、experts 3.52-3.57、head 0.73)。

### 3.2 t418c (`2c23e4905`、収録した PTX を既定で測る。load 11.19 で他タスクと重なった)

| cond | short ms/tok | c2000_cold | c2000_turn2 | c8000_cold | c8000_turn2 | prefill |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| x0 (捨て) | 12.8 | 15.4 | 14.1 | 15.5 | 14.5 | 0.56 s |
| **v5 (旧)** | **13.2** | 15.0 | 14.3 | 15.5 | **14.8** | 0.56 s |
| base (dn5) | 12.9 | **14.5** | 13.9 | **15.0** | **14.5** | 0.56 s |
| kp (dn5 + kprof) | 12.8 | 15.6 | 13.8 | 16.1 | 14.5 | 0.56 s |
| base2 (dn5) | 12.8 | 14.6 | 13.9 | 14.9 | 14.5 | 0.56 s |

```text
VERDICT v5    vs x0 OK decode +3.4% prefill +0.0% md5 SAME errors 0
VERDICT base  vs x0 OK decode +0.6% prefill +0.0% md5 SAME errors 0
VERDICT kp    vs x0 OK decode +0.0% prefill +0.0% md5 SAME errors 0
VERDICT base2 vs x0 OK decode +0.0% prefill +0.0% md5 SAME errors 0
VERDICT_DEPTH v5 c8000_turn2 vs x0 decode +2.7% prefill +2.0%
VERDICT_DEPTH base c8000_cold vs x0 decode -3.1% prefill -10.3%
VERDICT_DEPTH base c8000_turn2 vs x0 decode +0.0% prefill -2.0%
VERDICT baseline x0 errors 0 text ok
```

段 (14k turn2): mix **3.98 → 3.53 / 3.54 / 3.56**、step 14.82 → 14.38-14.41。
kp の `[dec-kprof-avg]` (band 0) で新カーネルは **grid 320 x block 160、gpu 15.3 us**
(v5 は 19.4)、窓の wall は 89-97 us (v5 は 96-114)。`[mix-qchain] dn5=12416` (v5 は 0)。

### 3.3 採否の判定 (task の規則)

- 目標「短い decode か 14k 後が 2% 以上」: 短い decode **-2.3% (t418b) / -2.3〜-3.0% (t418c)**、
  14k 後 **-2.0〜-2.7%**。2 ラウンドとも 2% を超えた。
- 「他が 1% を超えて悪化しない」: 段は mix 以外すべて ±0.03 ms 以内、prefill は 0.56 s で
  同値 (c8000_cold は 6.68/5.64 → 5.97/5.66 で、run 内の順序バイアスの範囲)。
- errors 0、md5 全 SAME、text 健全 (§4)。 => **ADOPT: yes**、既定 on (2/n で実施済み)。

## 4. 出力テキスト (t418c の base = 既定 dn5)

- `TXT`: `# B-Trees: A Comprehensive Explanation\n\n## 1. What is a B-Tree?\n\nA **B-tree (Balanced Tree) is a self-balancing tree data structure that maintains sorted data and allows searches, sequential access, insertions, and deletions in logarithmic time.`
- `TXTB`: `Rust の「所有権（Ownership）とライフタイム（Lifetime）は、Rust を学ぶ上で最も重要であり、かつ最も難しい概念です。\n\nしかし、これらは「メモリを安全に管理するためのルール**です。`
- `TXTD` (14k 後): `ご提示いただいた文書は、LLM推論エンジン（`denno-engine）における **Prefill（事前計算）処理の性能最適化の記録です。`
- v5 の COND と 1 バイトも違わない (md5 6 コンテキスト全部 SAME)。

## 5. 数値の扱い

- **値は丸めだけ変えた** (合計順: v5 は「1 レーンが 5 対を積んでから 32 レーンの木」、
  dn5 は「1 対ずつ 32 レーンの木 → 5 warp の逐次」)。積和そのものの式・1 演算 1 丸め・
  Q8_0 の読み方は同一。差は相対 1e-7 程度で、md5 は 6 コンテキスト全部 SAME だった。
- 採否は速度で決めた (§3.3)。lab の数字はすべて実測、見積りは §7 にだけ書く。
- t418a の x0 はページキャッシュが冷えて 20-22 ms/tok だったので基準に使っていない
  (base/base2 と、t418b/t418c の同一 run 内の対で判定した)。

## 6. テスト

関門 (merge 後 8f8ea9429 のツリーで実行):

```text
CARGO_TARGET_DIR=/tmp/t418 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1299 passed; 0 failed; 103 ignored
同 --test kernel_invariants
  -> 121 passed; 0 failed
CARGO_TARGET_DIR=/tmp/t418nc cargo build --workspace --release   -> ok (cuda feature 無し)
同 cargo test --workspace --release -- --test-threads=1          -> rc=0
  (denno-server の admit_error_message が 1 回だけ負荷でタイムアウトした。
   単独で再実行すると 7 passed。この task の変更は engine の mix GEMV だけで無関係)
```

既知の赤 (`qwen4exp_decode spec_batch_stops_at_eos` / `qwen4exp_params` の validate 2 件)
は範囲外で触っていない。

足したテスト (3 本):
- `cuda::tests::mix_dn5_gate_and_shape` — `DENNO_MIX_QK` の真理値表 (既定 (Dn5, V5)、
  `v5`/`1`/`both` で (V5, V5) へ戻る) と `mix_dn5_warps` の門番 (10240 → 5 warp、
  32 の倍数でない / 9 warp / up の 320 は None)。
- `cuda::tests::mix_dn5_cpu_mirror_matches_v5_and_reference` — 実形 (10240 x 8 行) の
  Q8_0 をテスト内で組み、**dn5 の合計順**と v5 の合計順を CPU で写して、f64 の真値に
  対して 1e-4 + 1e-5|ref|、互いに 1e-3 + 1e-5|b| で一致すること + 決定的であることを固定。
- `kernel_invariants::task418_mix_dn5_kernel_is_wired` — `.cu` の名前・引数 6 本・
  1 レーン 1 対の式・warp の逐次縮約 / cuda.rs の定数と起動形 (block = 32 * warps) /
  既定 (Dn5, V5) と切り戻し / build_ptx.sh の target と targets_all (末尾は動かさない) /
  収録 PTX の entry と引数 6 本。

## 7. なぜ効いたか / 次への申し送り

1. 効いた理由は**並列度**。同じ 3.48 MB を 10,240 スレッド (180 GB/s) から
   51,200 スレッド (227 GB/s) に増やして 19.4 → 15.3 us。まだ 227 GB/s なので、
   天井は `x` の再読み (全 320 行が同じ 41 KB を読む = 13 MB/token の L2 トラフィック) と
   推定する。
2. **次の一手 (推定、未実測)**: 1 ブロックが 2〜4 行を持ち、`x` の 64 要素を共有メモリに
   1 回だけ置いて行で使い回す。320 ブロック → 160/80 ブロックになるが、x のトラフィックが
   13 MB → 6.5/3.3 MB に減る。3.48 MB を 1.4 TB/s で読めれば 2.5 us 級で、いまの 15.3 us
   からさらに -8〜-10 us/call (decode で -0.6〜-0.8 ms/token) が見込める。
   ただしブロック数が減るので並列度とのトレードオフを測る必要がある。
3. `up` (`dno_mix_upmix_q8_0`、427 x 128、10.6 us = 328 GB/s) も同じ形の問題を残している。
   silu + hc_mixed を融合したまま行を増やす改造が要る (推定 -4〜-6 us/call)。
4. **lab の制約 (再確認)**: `DENNO_MIX_QK` のような**既存 env の新しい値**は COND で届く
   (t418b の dn5 は届いた)。新しい env 名は届かない (P414)。既定 off のゲートを測るときは
   「既存 env に値を足す」のが一番安い。

## 8. GPU で確認してほしいこと

1. 既定のビルドで `[gates] 既定 on の PTX 監査: 31 本のゲートを確認 / 欠落 0 本` が
   1 行出て、`[mix-qchain] ... dn5=` が 0 でないこと (PTX が読めていれば v5 へ落ちない)。
2. `DENNO_MIX_QK=v5` で `dn5=0` に戻り、decode が 13.2 ms/tok 側へ戻ること (切り戻し)。
3. 本番 (8098) で短い decode が 20-26 ms/tok の範囲に収まり、長い文脈でも text が
   崩れないこと (md5 は lab の 6 コンテキストで SAME だが、実運用の分布は見ていない)。
