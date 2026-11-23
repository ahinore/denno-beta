ADOPT: yes t484c

# REPORT P484 — 既定モデル decode: hc inject の GEMV を 4 行同時の F32 FMA にした。
# 短い decode 11.1 -> 10.8 ms/tok (-2.7%)、8k 後 12.1 -> 11.7 (-3.3%)、
# 14k 系 (c8000) も -3.8 〜 -4.9%。prefill と md5 は不変。

branch `br484` = 開始時の `int-auto` **b89a08a0e** (= merge: br474)。
`br480` は int-auto に未 merge だったが、中身は `REPORT_P480.md` だけ
(実験コードは task 480 の 3/n で削除済み) なので取り込んだ (fast-forward)。
終端でも int-auto は b89a08a0e のまま (動いていなかった)。

数値はすべて GPU ラボの実測 (`SCENARIO=depth8k`、`PROFILE=1`、モデルは
`flash-next-UD-IQ3_XXS`、ベースライン COND = int-auto の既定 +
`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0
DENNO_GPU_WEIGHT_MB=12000`、先頭に捨て COND、末尾に base の繰り返し)。
lab 名と SHA: **t484a = `150989326`** (現 int-auto の per stage profile)、
**t484b = `ed62b87cc`** (A/B、門は既定 off)、**t484c = `47e913457`**
(門を既定 on にした最終ツリーの確認)。`RESULT.md` 2 行目は 3 run とも
`model=flash-next-UD-IQ3_XXS`、`load_before` は 2.00 / 1.66 / 1.78。

## 0. 結論 (6 行)

1. **採る。** mix 段のホスト仕事で最大だった **hc inject の GEMV (16.1 us/回、
   97 呼び出し/token = 1.56 ms/token = decode の 14%)** を、行をまたいで x を
   共有する 4 行同時の FMA カーネルに置き換えた (**1.00 us/回**)。
   丸めだけ変わる (差 2.3e-6、|ref| ≈ 1.0)。
2. **短い decode 11.1 -> 10.8 ms/tok (-2.7%、t484b)、10.8 対 base2 11.1 /
   base 11.2 (t484c で -2.7 〜 -3.6%)。** 深い点は t484c で
   c2000 13.0 (base 13.7) / c8000_cold 12.5 (base 13.1) / c8000_turn2 11.7
   (base 12.1) で **-3.8 〜 -5.1%**。2 ラウンドが同じ向き・同じ大きさ。
3. **効いたのは 1.5 ms のうち 0.6 ms だけ。** 残りは GPU の鎖の待ちに隠れていた
   (`[q4ledger]` の mix: cpu 2.14 -> 1.52、wall 2.73 -> 2.10 ms/token)。
   つまり現状の mix 段は **もうホスト CPU 律速ではない** (下の §5)。
4. **`[mix-qchain]` に `inj4=` を足した。** 発火は 12288 = 128 token x 96 で
   100% (miss 0)。分岐が本当に通った証拠。n=5 など形が合わないときは
   何も書かずに従来経路へ落ちる (テストで固定)。
5. **既定 on、切り戻しは `DENNO_MIX_INJ_R4=0`。** 値は FMA と 8 レーン縮約で
   丸めが変わる (md5 は全 COND SAME だった。差が小さく logits の argmax が
   動かないため)。モデルの挙動を丸め以上に変えるものではない。
6. **次は mix の GPU 側**(1 呼び出し down 11.7 us + up 9.0 us = 2.0 ms/token で、
   3.48 MB を 297 / 387 GB/s)。ここは .cu を触るしかない (§6)。

## 1. なぜこれを選んだか (Do 1)

- t484a の profile で、decode の段は **experts 2.88 / ssm 2.79 / mix 2.58**
  だった。experts と ssm は P480 / P476 / P392 で形を振って決着済み
  (「experts の壁は起動本数ではなく行あたりの並列度」「ssm は待ちの形」)。
- mix 段の内訳を `[dec-kprof-avg]` の host step で見ると
  **`mix_inj` 16.1 us/回** が飛び抜けて大きい (`mix_norm` 3.3、`mix_h2d_pack` 0.8、
  `mix_pack` 0.5)。97 呼び出し/token で 1.56 ms/token。
- `[mix-kprof]` の warm 行は `gemv inj calls=12288 ms=169.5 gpu=0 ty=-`。
  `ty=-` は F32 (ggml id 0 = F32。`[mix-kprof]` の `ty_name` は 0 を「未設定」と
  表示する)。実形は **F32 の 4 行 x 10,240 = 160 KB**。
- dev の実測 (`examples/inj_gemv_bench.rs`、commit しない) でも、現行の
  1 行ずつ逐次和は **16.7 us**、4 行同時 FMA は **0.98 us**。lab と dev が一致。
- P84/P86 は hc **down/up** を 2 行カーネルにしたが、**inject は 1 行のまま**
  だった (n=4 で行並列の門番 512 行に届かない)。ここが唯一の取り残し。

## 2. 変えたこと

### 2.1 `expert_gemv_f32_rows_small` (moe_expert.rs)

- 既存の `gemv_f32_rows_fma_avx2<G, A>` を **G = n_out (1..=4) / A = 1** で回す
  入口を足した。1 ブロック = 1 x ベクトルで、G 本の独立な FMA チェーンが
  同じ x を共有する (x の再読みが 4 分の 1)。
- **A = 2 (k を 2 分割) は駄目だった**: 内側ループの `acc[g][a]` が動的添字に
  なってスタックへ落ち、3.18 us。A = 1 で 1.00 us (行ごとの独立チェーンだけで
  FMA レイテンシは隠れる)。この理由はコードの doc に残した。
- 形が合わない (n_out > 4 / 長さ違い / AVX2+FMA 無し / `DENNO_F32_EXACT=1`) は
  **何も書かずに false**。呼び出し側は従来の `gemv_label` にそのまま落ちる。

### 2.2 `do_inj` から呼ぶ (qwen4exp.rs)

- `inject_gemv_r4(w, xn, &mut v)` を先に試し、false なら従来の
  `kp_step!("mix_inj", w.gemv_label(...))`。成功時は
  `set_last_linear_ran_on_gpu(false)` を立てて `[mix-kprof]` の gpu= を汚さない。
- 門は新 env `DENNO_MIX_INJ_R4` (既定 on、`0`/`off`/`false`/`no` だけ off) と、
  **lab の COND が新しい env 名を落とす回の逃げ道**として既存 `DENNO_MIX_QK` の
  値に `inj4` が含まれていれば on (t294b の経験)。t484b では**両方**を COND に
  入れたので、どちらが届いても発火する (実際は新 env も届いた)。
- 発火の証拠 `[mix-qchain] ... inj4=N` を足した。

## 3. 数値の扱い

- 値は **FMA + 8 レーン縮約**なので classic の逐次和と丸め順が違う。
  dev 実測の max|diff| は 2.3e-6 (|ref| 0.99) で、既定の許容
  (絶対 1e-4 + 相対 1e-5) の内側。テストはその許容で classic と比べている。
- **lab の md5 は 5 コンテキスト x 3 run すべて SAME** (差が小さく argmax が
  動かないため)。テキストも全 COND `looks ok`。
- 決定性: 行ごとに独立・k 昇順の 1 本のチェーンなので、同じ入力なら同じ値。
  行チャンク分割 (テストの n = 1..4) でも同じ。

## 4. lab の結果

### 4.1 t484b (`ed62b87cc`、門は既定 off、A/B)

| cond | decode ms/tok | prefill s | c2000_cold | c2000_turn2 | c8000_cold | c8000_turn2 | md5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| c0 (捨て) | 11.2 | 0.58 | 13.5 | 12.3 | 13.1 | 12.3 | SAME |
| base | 11.1 | 0.58 | 13.4 | 12.2 | 13.0 | 12.1 | SAME |
| inj4 | **10.8** | 0.57 | **13.0** | **11.8** | 13.1 | 12.0 | SAME |
| kp (inj4+kprof) | 10.8 | 0.56 | 14.1 | 12.0 | 13.6 | 11.9 | SAME |
| base2 | 11.1 | 0.59 | 13.4 | 12.2 | 13.2 | 12.3 | SAME |

```text
VERDICT base vs c0 OK decode -0.7% prefill -0.9% md5 SAME errors 0
VERDICT inj4 vs c0 OK decode -3.8% prefill -2.6% md5 SAME errors 0
VERDICT base2 vs c0 OK decode -0.7% prefill +0.9% md5 SAME errors 0
```

base と base2 が 11.1 で一致しているので、c0 (11.2) ではなく base を基準に読む:
**短い decode 11.1 -> 10.8 (-2.7%)**、c2000_cold -3.0%、c2000_turn2 -3.3%、
c8000_turn2 -0.8%、c8000_cold +0.8% (この点は base2 も 13.2 で ±1.5% の揺れ)。

証拠:
- `[mix-qchain] on calls=12416 hits=12416 qk=12416 ... inj4=12288` (kp run)。
  12288 = 96 呼び出し x 128 token で **inject の全回**。
- `[q4ledger]` (base -> inj4): mix **wall 2.73 -> 2.10、cpu 2.14 -> 1.52**、
  ssm 2.87 -> 3.07、full 1.17 -> 1.26、experts 3.63 -> 3.65。
  mix が縮んだぶん、それまで mix の陰に隠れていた GPU 待ちが ssm/full に
  見えるようになった (合計は 24.09 -> 23.31 ms/token)。
- `[q4prof]` の layer 時間 11.442 -> 11.050 ms/token (-3.4%)。

### 4.2 t484c (`47e913457`、門を既定 on にした最終ツリー)

| cond | decode ms/tok | prefill s | c2000_cold | c2000_turn2 | c8000_cold | c8000_turn2 | md5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| c0 (捨て) | 13.3 | 0.66 | 43.2 | 30.7 | 13.8 | 12.9 | SAME |
| base (=`DENNO_MIX_INJ_R4=0`) | 11.2 | 0.57 | 13.7 | 12.3 | 13.1 | 12.1 | SAME |
| on (既定) | **10.8** | 0.56 | **13.0** | **11.7** | **12.5** | **11.7** | SAME |
| base2 (=0) | 11.1 | 0.57 | 13.3 | 12.1 | 13.0 | 12.3 | SAME |

```text
VERDICT base vs c0 OK decode -15.7% prefill -13.0% md5 SAME errors 0
VERDICT on vs c0 OK decode -18.6% prefill -14.5% md5 SAME errors 0
VERDICT base2 vs c0 OK decode -16.3% prefill -13.0% md5 SAME errors 0
VERDICT baseline c0 errors 0 text ok
```

c0 は page cache が冷えた捨て COND (43.2 ms/tok) なので基準にしない。
base/base2 を基準にすると **短い decode -2.7 〜 -3.6%、c2000_cold -2.3 〜 -5.1%、
c2000_turn2 -3.3 〜 -4.9%、c8000_cold -3.8 〜 -4.6%、c8000_turn2 -3.3 〜 -4.9%**。
prefill は 0.57 -> 0.56 (c8000_cold 5.27/5.22 -> 5.21) で変わらない。
**既定 on のツリーが A/B と同じ勝ちを再現した。**

## 5. 出力テキスト (健全性)

t484c の `on` (既定 on) の `TXTD`:
`ご提示いただいた文書は、LLM推論エンジン（`denno-engine`）における **Prefill（事前計算）フェーズの性能最適化の歴史を詳細に記録した技術レポートです。`
base / base2 と同一の md5 (`edaa84ad` ほか 5 コンテキスト全部)、errors 0。

## 6. テスト

```text
CARGO_TARGET_DIR=/tmp/t484f cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1345 passed; 0 failed; 103 ignored   (既定 on のツリーで実行)
同 --test kernel_invariants
  -> 139 passed; 0 failed
(関門の外) CARGO_TARGET_DIR=/tmp/t484nocuda cargo build --workspace --release  -> ok
同 cargo test --workspace --release -- --test-threads=1                      -> 全 suite ok
同 cargo test --release -p denno-engine --lib -- --test-threads=1
  -> 968 passed; 0 failed; 55 ignored   (cuda feature 無しのビルドも壊していない)
```

足したテスト 2 本 (どちらも既定 on の真理値表を含む):
- `mix_inj_r4_gate_truth_table`: 未設定/空は on、`0`/`off`/`false` だけ off、
  `DENNO_MIX_QK` の `inj4` は明示 off より強い。
- `mix_inj_r4_fires_and_matches_classic`: n = 1..4 で**発火** (計数が増える) し、
  classic の逐次和と許容内で一致。ゲート off では**1 バイトも書かない**。
  n = 5 は不発で false。

dev のベンチ (`examples/inj_gemv_bench.rs`、未追跡・commit しない):
`k=10240 n=4: q8_fast=4.24 us | q8_2row=1.84 us | f32_classic=16.73 us |
f32_rows4=0.99 us | expert_gemv_f32_rows_small(実装)=1.00 us`、
`vs classic max|diff|=2.265e-06 (|ref|=9.863e-01)`。

## 7. 次への申し送り

1. **mix 段はもうホスト律速ではない。** inject (1.56 ms/token) を消したあとの
   内訳は GPU が down 11.7 + up 9.0 = 20.7 us/回 (2.0 ms/token) で、
   `[q4ledger]` の mix wall 2.10 のうち ~1.2 ms が **待ち** (defer しない
   FFN の mix 48 回 x 20 us)。ホストはもう 1 ms を切った。
2. **次に mix を削るなら GPU カーネル**: 3.48 MB を 297 GB/s (down dn7) /
   387 GB/s (up u4f) でしか読めていない。RTX 5090 で 3.48 MB は
   L2 (96 MB) に載るので、形を変えれば 600-800 GB/s が狙える (推定)。
   dn5 -> dn6 -> dn7 とホスト側の形は振り切ったので、**.cu で 1 ブロックが
   複数行を受けて x を共有する**方向しか残っていない (P480 §7-3 と同じ結論)。
3. **FFN の mix の待ちを消すのは P472 で済み** (ffn まで defer すると中立)。
   同じ方向は再試行しない。
4. **inject 側で残っているもの**: `mix_norm` 3.3 us/回 (40 KB の読み書きで
   帯域なり) と `mix_h2d_pack` 0.8 us。どちらも床に近い。
5. **experts 3.65 / ssm 3.07 ms/token** は P480 / P476 の結論どおり
   「形」では動かない。触るなら重みのバイト数か、層あたりの同期の数。
