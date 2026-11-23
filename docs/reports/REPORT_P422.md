ADOPT: yes t422b t422c

# REPORT P422 — 既定モデル decode: Q8_0 GEMV を「1 ブロック rows 行・1 レーン 1 対」に
# (短い decode 12.8 -> 12.0 ms/tok = -7.0%、14k 後 14.7 -> 13.8 = -5.4%)

branch `br422` = 開始時の `int-auto` **c5bfab6f9** (merge: br418。自分の br418 は
既に merge 済みだったので取り込み不要)。終端で int-auto が **8cef37b60**
(merge: br415 = prefill の exp_pf / linear_gguf) へ動いたので merge した
(衝突なし、`build_ptx.sh` の `targets_all` も無傷)。最終 HEAD は 126a7b030 + REPORT。
数値はすべて GPU ラボの実測 (lab 名と測った SHA):
**t422a = c5bfab6f9** (前提の取り直し + `[dec-kprof-avg]` の段内訳)、
**t422b = 56a609dbe** (PTX=mix_dn6、4 案 + base/base2 の A/B)、
**t422c = 8aed5bd7b** (収録した PTX を既定で測る。`ptx=none`、5 COND)。
`RESULT.md` 2 行目は t422a/t422c が `ptx=none`、t422b が `ptx=mix_dn6`、
3 run とも `model=flash-next-UD-IQ3_XXS`。

## 0. 結論 (6 行)

1. **採る。** 新カーネル `dno_mix_gemv_q8_0_dn6` (1 ブロック rows 行 / 1 レーン 1 対 /
   x をレジスタに 1 回) を 2 か所の既定にした:
   **mix の down (rows=2)** と **GDN の qkv/out (rows=2)**。
2. **短い decode (中央値、同じ run の対)**: t422b base 12.9 / base2 12.9 →
   dn6 rows=4 **12.6 (-2.1%)** / rows=2 **12.5 (-3.1%)**。
   t422c base 12.8 / base2 12.9 → dn6 (既定) **12.5 (-2.7%)**、
   dn6 + GDN **12.0 (-7.0%)**。生の秒でも round1/round2 が
   base 1.64/1.64 s → dn6 1.60/1.61 s → dn6g2 **1.53/1.54 s** (128 トークン)。
3. **14k 後 (c8000_turn2)**: t422b base 14.8 / base2 14.4 → rows=2 **14.2**、
   t422c base 14.7 / base2 14.6 → dn6 **14.2 (-3.4%)** / dn6g2 **13.8 (-5.4%)**。
4. **段の動きは狙った 2 段だけ** (`[q4ledger]` 14k turn2、ms/token):
   **mix 3.55/3.57 → 3.35 → 3.36 (-6%)**、**ssm 3.40/3.40 → 3.37 → 2.75 (-19%)**。
   full 2.13 / experts 3.52 / head 0.73 は不変 (±0.03)。
5. **md5 は全 COND・全 6 文脈で SAME** (`a:edaa84ad b:1938059f c2000_cold:b42d79d2
   c2000_turn2:12fc5c36 c8000_cold:672fec07 c8000_turn2:98e879d2`)、errors 0、
   text 健全 (§4)。**mix の down は dn5 と 1 ビットも変わらない** (同じ積和・同じ
   縮約順) ので、変わるのは GDN の合計順 (丸めだけ) だが、出力トークンは動かなかった。
6. 発火の証拠: `[mix-qchain]` に `dn6=` を足した。t422c の base は
   `dn5=12416 dn6=0`、dn6 は `dn5=0 dn6=12416`、dn6g2 は **`dn6=21632`**
   (= mix 12416 + GDN の qkv/out 9216)。`[gates] 既定 on の PTX 監査: 32 本 /
   欠落 0 本`。

## 1. 前提: t422a = int-auto c5bfab6f9 の取り直し (depth8k、PROFILE=1、load 10.49)

| cond | short ms/tok | c2000_turn2 | c8000_cold | c8000_turn2 | prefill c8000_cold |
| --- | ---: | ---: | ---: | ---: | ---: |
| x0 (捨て) | 12.8 | 14.0 | 15.6 | 15.0 | 6.95 s |
| base | 12.8 | 14.0 | 15.3 | 14.6 | 6.04 s |
| kp (+kprof) | 12.8 | 14.0 | 18.2 | 15.9 | 6.83 s |
| base2 | 12.8 | 13.8 | 15.0 | 14.5 | 5.67 s |

`[q4ledger]` (kp、14k turn2): mix 3.52 / experts 5.09 (kprof の上乗せ込み) /
ssm 3.39 / full 2.13 / head 0.73 / shared 0.62 / router 0.18 / step 15.92。

`[dec-kprof-avg]` (band 0。mix は 1 窓 96.9 us、うち kern 29.5 / copy 16.5 /
gap 18.5 / host steps 19.0):

| op | grid x block | gpu us | 中身 |
| --- | --- | ---: | --- |
| `dno_mix_gemv_q8_0_dn5` (mix down) | 320 x 160 | **15.4** | 重み 3.48 MB + **x 13.1 MB** = 16.6 MB |
| `dno_gemv_q8_0_v5` (GDN qkv/out) | 320 x 256 | **35.4 x 2/層** | 重み 27.9 MB、1 レーンが 5 対を逐次 (787 GB/s) |
| `dno_mix_upmix_q8_0` (mix up) | 427 x 128 | 10.8 | 3.48 MB |
| `dno_matvec_f32_r` (router) | 512 x 256 | 7.0 | gap 22.5 |
| H2D@a / D2H@a | — | 9.7 / 5.9 | 起動の固定費が支配的 |

## 2. なぜこれを選んだか (Do 1 / Do 2)

- 段は experts / mix / ssm が 3.4-3.5 ms/token で並ぶ。どれも既に手が入っている
  (388 が experts のミス GEMV の重ね打ち、418 が mix の down、392 が ssm の
  `GDN_X_MIX`、406 が expert の上げ直し予算)。
- 段の内訳を op まで降りると、**残っているのは Q8_0 GEMV の形**だった:
  dn5 は「行ごとに x 41 KB を読み直す」ので x が 13.1 MB (重みの 3.8 倍)、
  GDN の v5 は「1 レーンが 5 対を逐次ループ」で 787 GB/s。どちらも 418 が
  mix の down で解いたのと同じ問題 (並列度・再読み) で、418 の申し送り
  「x を共有して複数行を 1 ブロックで」の直系。
- 起動の固定費 (H2D 9.7 / D2H 5.9 / 各カーネル ~7 us) はもう削れないので、
  **op を増やさずに 1 カーネルの中の並列度と再読みを直す**方針にした。

## 3. 変えたこと

### 3.1 `kernels_mix_dn6.cu` — `dno_mix_gemv_q8_0_dn6(w, x, out, n_in, n_out, row_bytes, rows)`

- **grid = ceil(n_out / rows)、block = rows * 32 * warps**
  (warps = ceil(n_in / 64 / 32)、上限 8、rows * 32 * warps <= 1024)。
- スレッド `(r, wid, lane)` は対 `pair = wid*32 + lane` を担当し、行 `blockIdx*rows + r`
  を読む。**x の 64 要素はレジスタに 1 回だけ載せ、同じレーンが rows 行で使い回す**。
- 積和 (`acc[j&3]` に 1 演算 1 丸め、d0/d1 の f16 展開、Q8_0 の 17 ワード読み) と
  縮約 (warp の木 16..1 → warp ごとの部分和 → **warp 0..warps-1 の逐次加算**) は
  dn5 と同一。**rows=1 で撃てば dn5 と同じ値**、rows>1 でも行ごとの演算列が同じ
  なので **dn5 と 1 ビットも変わらない** (CPU ミラーで固定)。
- 門番は純関数 `mix_dn6_plan(n_in, n_out, want_rows)`: Q8_0 / `n_in % 64 == 0` /
  warps <= 8 / rows は 1024 スレッドに丸める。合わない形と PTX 欠けは v5 へ
  (fail closed)。

### 3.2 ゲートと既定

- **mix の down**: 既存の `DENNO_MIX_QK` に値 `dn6` / `d6` / `dn6r1` / `dn6r2` /
  `dn6r4` / `dn6r6` / `dn6r8` を足した (新しい env 名は作らない = lab に届く)。
  **既定を `(Dn6 rows=2, V5)` にした**。切り戻しは `dn5` (task 418) / `v5` / `0`。
- **GDN の qkv/out**: 既存の `DENNO_GDN_DEC_GEMV` に値 `dn6` / `dn6r2` 等を足し、
  **既定を dn6 (rows=2)** にした (`GDN_DEC_GEMV_DN6_DEFAULT`)。v5 の計画は
  形が合わない単位の受け皿として残す。切り戻しは `v5`。
- 発火: `[mix-qchain]` 行に `dn6=` を足した。`gate_audit` の既定 on 台帳に
  `kernels_mix_dn6.ptx` を足した。
- `build_ptx.sh` に target `mix_dn6` (nofmad) を足し、`targets_all` は
  **末尾ではなく `mix_dn5` の直後**に入れた (末尾は他ブランチが append する場所)。

## 4. lab の結果 (3 ラウンド)

すべて `SCENARIO=depth8k PROFILE=1`、ベースライン COND は
`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0
DENNO_GPU_WEIGHT_MB=12000`、最初の COND は捨て、最後に baseline の繰り返し。
lab は各 COND を 2 回ずつ回すので、下表の中央値は 2 ラウンド分。

### 4.1 t422b (`56a609dbe`、PTX=mix_dn6、load 1.92)

| cond | short ms/tok | c2000_turn2 | c8000_cold | c8000_turn2 | prefill c8000_cold |
| --- | ---: | ---: | ---: | ---: | ---: |
| x0 (捨て) | 12.9 | 14.0 | 16.7 | 15.5 | 7.30 s |
| base (dn5) | 12.9 | 13.9 | 15.3 | 14.8 | 5.96 s |
| **m6 (rows=4)** | **12.6** | 13.8 | 15.1 | 14.5 | 5.93 s |
| **m6r2 (rows=2)** | **12.5** | 13.8 | 14.7 | **14.2** | 5.92 s |
| g6 (GDN rows=6) | 13.1 | 14.1 | 15.3 | 14.6 | 5.64 s |
| base2 (dn5) | 12.9 | 13.8 | 15.0 | 14.4 | 5.64 s |

```text
VERDICT base  vs x0 OK decode +0.0% prefill -1.7% md5 SAME errors 0
VERDICT m6    vs x0 OK decode -2.1% prefill +0.0% md5 SAME errors 0
VERDICT m6r2  vs x0 OK decode -2.4% prefill +0.0% md5 SAME errors 0
VERDICT g6    vs x0 OK decode +2.1% prefill -2.6% md5 SAME errors 0
VERDICT base2 vs x0 OK decode +0.0% prefill -1.7% md5 SAME errors 0
VERDICT baseline x0 errors 0 text ok
```

段 (14k turn2、ms/token、base / m6 / m6r2 / g6 / base2): mix
**3.58 / 3.38 / 3.34 / 3.54 / 3.56**、ssm 3.39 / 3.38 / 3.38 / **3.65** / 3.39、
full 2.13 / 2.12 / 2.13 / 2.12 / 2.13、experts 3.53 / 3.51 / 3.50 / 3.50 / 3.50、
step 15.25 / 15.06 / 14.67 / 15.27 / 14.90。
→ mix の down は rows=2 が最良 (-6.7%)。**GDN rows=6 は ssm を +7.7% 悪化**させた
(ブロック数 427 で 1 ブロック 960 スレッド = 大きすぎる) ので、rows を 2 にして
t422c で測り直した。

### 4.2 t422c (`8aed5bd7b`、収録した PTX を既定で測る。`ptx=none`、load 5.56)

| cond | short ms/tok | c2000_turn2 | c8000_cold | c8000_turn2 | prefill c8000_cold |
| --- | ---: | ---: | ---: | ---: | ---: |
| x0 (捨て、dn5) | 12.9 | 14.8 | 15.7 | 14.9 | 7.11 s (cache 冷) |
| base (dn5) | 12.8 | 14.0 | 15.4 | 14.7 | 5.93 s |
| **dn6 (既定、mix rows=2)** | **12.5** | 13.8 | 14.8 | **14.2** | 5.93 s |
| **dn6g2 (上の GDN dn6r2)** | **12.0** | **13.0** | **14.4** | **13.8** | 5.64 s |
| base2 (dn5) | 12.9 | 14.0 | 15.1 | 14.6 | 5.76 s |

```text
VERDICT base  vs x0 OK decode -0.6% prefill -1.7% md5 SAME errors 0
VERDICT dn6   vs x0 OK decode -2.7% prefill -0.9% md5 SAME errors 0
VERDICT dn6g2 vs x0 OK decode -7.0% prefill -2.6% md5 SAME errors 0
VERDICT base2 vs x0 OK decode -0.3% prefill -2.6% md5 SAME errors 0
VERDICT baseline x0 errors 0 text ok
```

生の秒 (128 トークンの a1、round1 / round2): base 1.64 / 1.64 s、
dn6 1.60 / 1.61 s (-2.4% / -1.8%)、dn6g2 **1.53 / 1.54 s (-6.7% / -6.1%)**。

段 (14k turn2、base / dn6 / dn6g2 / base2): **mix 3.55 / 3.35 / 3.36 / 3.57**、
**ssm 3.40 / 3.37 / 2.75 / 3.40**、full 2.13 / 2.13 / 2.14 / 2.13、
experts 3.52 / 3.50 / 3.53 / 3.53、step 15.33 / 14.77 / 14.33 / 15.05。

### 4.3 採否の判定 (task の規則)

- 目標「短い decode か 14k 後が 2% 以上」: 短い decode は **-2.7% (t422c dn6)**、
  両方を入れて **-7.0%**。14k 後も -3.4% / -5.4%。t422b の同じ段の対でも
  -2.1% / -2.4% (rows=4 / rows=2) で、**2 ラウンド (t422b と t422c、各 2 回) とも
  2% を超えた**。
- 「他が 1% を超えて悪化しない」: 段は mix と ssm だけが動き、full / experts /
  head / prefill は不変。prefill の中央値は 0.56-0.57 s の丸め幅 (14k cold は
  5.64-5.93 s で base の振れの内側)。
- errors 0、md5 全 SAME、text 健全 (§5)。 => **ADOPT: yes**、既定 on (3.2)。
- **GDN 側だけは 1 リクエスト (t422c) の実測**である点を明記する。rows=6 は
  t422b で悪化、rows=2 は t422c で ssm -19%。同じカーネル・同じ理由 (ブロック数と
  1 レーン 1 対) で、ssm 段の台帳は同じ run の base / base2 が 3.40 / 3.40 で
  揃っているので、ノイズではないと判断した。次タスクで 2 本目の確認をしてほしい。

## 5. 出力テキスト (t422c の dn6g2 = 最終既定と同じ経路)

- `TXT`: `# B-Trees: A Comprehensive Explanation\n\n## 1. What is a B-Tree?\n\nA **B-tree (Balanced Tree) is a self-balancing tree data structure that maintains sorted data and allows searches, sequential access, insertions, and deletions in logarithmic time.`
- `TXTB`: `Rust の「所有権（Ownership）とライフタイム（Lifetime）は、Rust を学ぶ上で最も重要であり、かつ最も難しい概念です。\n\nしかし、これらは「メモリを安全に管理するためのルール**です。`
- `TXTD` (14k 後): `ご提示いただいた文書は、LLM推論エンジン（`denno-engine）における **Prefill（事前計算）フェーズの性能最適化の歴史を詳細に記録した技術レポートです。`
- base (dn5) / dn6 / dn6g2 / base2 で 1 バイトも違わない (md5 6 コンテキスト全部 SAME)。

## 6. 数値の扱い

- **mix の down は値が変わらない** (dn5 と同じ積和・同じ縮約順。CPU ミラー
  `mix_dn6_cpu_mirror_is_bit_identical_to_dn5` が rows=1/2/4/6 でビット一致を固定)。
- **GDN の qkv/out は合計順が変わる** (v5 は「1 レーンが 5 対を積んでから 32 レーンの
  木」、dn6 は「1 対ずつ 32 レーンの木 → warp の逐次」)。差は丸めだけで、
  出力トークンは 6 文脈で 1 つも動かなかった (md5 SAME)。
- 採否は速度で決めた (§4.3)。lab の数字はすべて実測、見積りは §8 にだけ書く。
- t422c の x0 は c2000_cold の prefill が 5.40 s (cache 冷) だったので基準に使わず、
  同じ run の base / base2 との対で判定した。

## 7. テスト

関門 (merge 後 126a7b030 のツリーで実行):

```text
CARGO_TARGET_DIR=/tmp/t422 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1301 passed; 0 failed; 103 ignored
同 --test kernel_invariants
  -> 122 passed; 0 failed
CARGO_TARGET_DIR=/tmp/t422nc cargo build --workspace --release   -> ok (cuda feature 無し)
同 cargo test --workspace --release -- --test-threads=1          -> 下の §7.1
```

既知の赤 (`qwen4exp_decode spec_batch_stops_at_eos` / `qwen4exp_params` の validate 2 件)
は範囲外で触っていない。

足したテスト (3 本) と直したテスト (4 本):
- `cuda::tests::mix_dn6_gate_and_shape` — `DENNO_MIX_QK` / `DENNO_GDN_DEC_GEMV` の
  真理値表 (既定は mix が dn6 rows=2、GDN が dn6 rows=2、`dn5` / `v5` で切り戻し)、
  `mix_dn6_warps` (10240 → 5 warp、5120 → 3、320 → 1、288 対 → None)、
  `mix_dn6_plan` (1024 スレッドへの丸め)。
- `cuda::tests::mix_dn6_cpu_mirror_is_bit_identical_to_dn5` — 実形 (10240 x 8 行) の
  Q8_0 をテスト内で組み、**dn6 (rows=1/2/4/6) の合計順を CPU で写して dn5 と
  ビット一致**すること、f64 の真値に 1e-4 + 1e-5|ref| で一致することを固定。
- `kernel_invariants::task422_mix_dn6_kernel_is_wired` — `.cu` の名前・引数 7 本・
  1 レーン 1 対・x の 1 回読み・共有メモリの縮約 / cuda.rs の定数と起動形
  (block = 32 * warps * rows) / 既定 (Dn6, V5) と GDN の既定 dn6 / GDN の起動点
  2 か所 / build_ptx.sh の target と targets_all (末尾は動かさない) /
  収録 PTX の entry と引数 7 本。
- 既定が変わったので、既存の 4 本 (`mix_qk_gate_defaults_on`、
  `task286_mix_gemv_v5_is_wired`、`task418_mix_dn5_kernel_is_wired`、
  `task422` の GDN 既定) の期待値を dn6 に更新した。

### 7.1 cuda 無しの workspace テスト (merge 後 126a7b030 のツリー)

```text
CARGO_TARGET_DIR=/tmp/t422nc cargo build --workspace --release  -> rc=0
同 cargo test --workspace --release -- --test-threads=1
  -> rc=0 (denno-engine 959 passed / 79 passed + 各 crate のテスト全部 ok、失敗 0)
```

## 8. なぜ効いたか / 次への申し送り

1. 効いた理由は **ブロック数と 1 レーン 1 対**。
   - mix の down: dn5 は 320 ブロックが同じ x 41 KB を読む (x 13.1 MB)。dn6 は
     2 行/ブロック = 160 ブロックで x が半分 + 行ごとの再読みが L1 に収まる。
     mix 段 -6%。
   - GDN の qkv/out: v5 は 320 ブロック x 256 スレッドで 1 レーンが 5 対を逐次。
     dn6 は **1280 ブロック x 160 スレッド**で 1 レーン 1 対。ssm 段 -19%。
     rows=6 (427 ブロック x 960 スレッド) は悪化したので、**ブロック数を増やす
     方向が正しい** (推定: 1 ブロック 960 スレッドは 1 SM に 2 ブロックしか
     載らず、レイテンシを隠す warp が足りない)。
2. **次の一手 (推定、未実測)**: 同じ形の Q8_0 GEMV がまだ 3 か所ある。
   `full` の qkv/wo (`dno_gemv_q8_0_v5` 320 x 256、30.7-33.8 us) と
   mix の up (`dno_mix_upmix_q8_0` 427 x 128、10.8 us = 322 GB/s)。
   前者は `plan_moe_gemv_v5` の `ROWS = 8` を 2 に落とすだけで試せる (推定 -3%〜)。
3. H2D@a 9.7 us / D2H@a 5.9 us / router の gap 22.5 us は起動の固定費で、
   カーネルを速くしても消えない。mix の窓 96.9 us のうち 16.5 us がコピー、
   18.5 us が gap。ここを削るには**鎖そのものを短くする** (op を減らす) しかない。
4. **lab の制約 (再確認)**: 既存 env の新しい値 (`DENNO_MIX_QK=dn6`,
   `DENNO_GDN_DEC_GEMV=dn6`) は COND で届く。今回は新しい env 名を 1 つも
   作らずに済ませた。
