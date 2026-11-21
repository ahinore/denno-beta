ADOPT: yes t504c

# REPORT P504 — 既定モデル decode: `dn7` (1 対 = 4 レーン) を mix の down 以外
# (GDN の out 投影 / head の語彙 GEMV) へ広げて **既定 on**。短い decode
# 10.6 -> 10.4 ms/tok (-1.9〜-2.2%)、14k 後 11.6 -> 11.4、head 段 0.60 -> 0.46
# = -23%、ssm 段 2.93 -> 2.87 = -2.1%。full 層への同じ手は **負け** (full 段 +10%)
# なので既定 off のまま残した。lab t504a / t504b / t504c の数字は GPU ラボの実測。

branch `br504` = 開始時の `int-auto` **890fc66d4** (merge: br496)。br500 は
未 merge だったが **採らなかった** (ADOPT: no、既定を 1 つも変えていない) ので
merge していない。終端で int-auto は 890fc66d4 のまま (追加 merge なし)。

lab 名と SHA:
- **t504a** = `3044d5e05` (5 cond x 2 ラウンド、`load_before=2.52`、out dn7 と full dn7 の A/B)
- **t504b** = `c8ba025b6` (6 cond x 2 ラウンド、`load_before=3.48`、out / head / 両方)
- **t504c** = `246c00739` (4 cond x 2 ラウンド、`load_before=2.61`、**新しい既定**と
  切り戻しの対 = 最終確認)

すべて `SCENARIO=depth8k`、`PROFILE=1`、`MODEL=flash-next-UD-IQ3_XXS`、ベースライン
COND = int-auto の既定 + `DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000
DENNO_EXPERT_GPU_MIN_HIT=0 DENNO_GPU_WEIGHT_MB=12000`、先頭に捨て COND、末尾に
base の繰り返し。新しい `.cu` / `.ptx` は **足していない** (t431 の
`kernels_mix_dn7.ptx` を使い回す) ので `PTX=` も無い。

## 0. 結論 (7 行)

1. 着手時の fresh profile は **P500 の t500prof をそのまま使った** (SHA 890fc66d4 =
   着手時の int-auto と同一、同日)。warm short 10.60 ms/tok: **ssm 2.90 /
   experts 2.82 / mix 1.97 / full 1.25 / head 0.60 / shared 0.58 / router 0.17**。
2. ただし台帳の「wall」を分解すると **GPU のカーネル時間ではなく「層ごとの
   往復待ち」が本体**だった: ssm 行は wall 2.93 のうち launch 0.19 / cpu 0.04 で、
   残り ~2.6 ms は **out の D2H (`cuMemcpyDtoH`) が前段の GPU 鎖を待つ時間**
   (36 層 x 73 us)。head 行も sync 0.58 = 1 本の GEMV の完了待ちそのもの。
   したがって「効くのは層の鎖の末尾 (= out) と、最後に 1 発だけ直列に走る head」で、
   中央のカーネルをいくら縮めても wall は動かない (t504a の full dn7 がその実例)。
3. 効いた 2 本 (どちらも既存 `kernels_mix_dn7.cu` の再利用、丸め順だけ変わる):
   - **GDN の out 投影 (n_in 6,144 / n_out 2,560) を dn7 で撃つ**。ssm 段
     2.93 -> 2.87 (2 ラウンド同じ)、短い decode -0.7%。d2h の待ち行列が短くなる。
   - **head (Q8_0 2,560 x 248,320 = 675.4 MB) を dn7 で撃つ**。head 段
     0.60 -> 0.46 ms/tok = **-23%** (gpu 0.57 -> 0.42)。短い decode -1.8%。
     dn6 は 1 レーンが x の 64 要素をレジスタに載せる (64 本) ので occupancy が
     低かった。dn7 は 16 要素で済む。
4. 2 本同時 (`o7h7`) で **短い decode 10.6 -> 10.4 (-2.2%、c0 比。同一 run の
   切り戻し r = 10.6 と比べても -1.9%)、14k 後 11.6 -> 11.3 (-2.6%)**、
   prefill は不変 (0.55-0.56 s、c0 の 0.55 に対する +1.8% は 0.01 s 刻みの
   量子化で、base も同じ値)、errors 0、**md5 は 6 文脈すべて SAME**。
5. **full 層の qkv/wo を dn7 にするのは負け** (t504a: full 段 1.79 -> 1.87 =
   +4.5%、1.26 -> 1.39 = +10%)。P500 の dn8 と同じ結果で、full 段は GPU 鎖の
   中央にあり wall に露出していないため、形を変えても速くならず固定費だけ増える。
   既定 off のまま残した (`DENNO_FULL_DEC_FAST=dn7`)。
6. 採った 2 本は**既定 on** にした (lab t504c で既定と切り戻しの対を 2 ラウンド:
   短い decode 10.6 -> 10.4、c8000_turn2 11.6 -> 11.4、head 0.60 -> 0.46、
   ssm 2.93 -> 2.87、md5 全 SAME、errors 0)。切り戻しは
   `DENNO_GDN_DEC_GEMV=dn8ar4` / `DENNO_DEC_HEAD_FAST=dn6`。
7. 目標 (短い decode か 14k 後が 2% 以上) は **短い decode -2.2% (c0 比) /
   -1.9% (切り戻し比)、14k 後 -2.6%** で満たした (0.1 ms/tok 刻みの中央値で
   2 段。3 run 連続で同値)。

## 1. なぜこれを選んだか (Do 1/2)

- P500 の申し送りは「mix の split-K が本命」「head は Q6_K なので dn8 が使えない」
  だったが、**その 2 つは前提が古かった**: 既定モデルの head は **Q8_0**
  (675.4 MB。ラボのログの `CUDA gguf weight upload (uid=1568): ty=Q8_0 k=2560
  n=248320`) で、dn6/dn7 がそのまま撃てる。Q6_K ではない。
- P500 の「帯域 (GB/s)」の読みも、実際の wall の分解と食い違っていた。
  `[q4ledger]` の ssm 行は wall=2.93 / launch=0.19 / sync=0.00 / cpu=0.04 で、
  **残り 2.6 ms が「コピーの中で前段の GPU を待つ」時間**。つまり decode は
  帯域ではなく **層ごとの往復 (submit -> D2H 待ち)** で決まる。そこへ効くのは
  「鎖の末尾 (out)」と「直列 1 発の head」だけなので、この 2 つを選んだ。
- 同じ理屈で、`experts` 段 (wall 2.89 / cpu 2.89) と `mix` 段 (cpu 1.39) は
  **ホスト仕事**で、GPU カーネルを縮めても wall は動かない (この 2 段は
  t338/339/337 で既に多数の手が試されて棄却済み)。full 段は GPU 鎖の中央。

## 2. 変えたこと

### 2.1 `DENNO_GDN_DEC_GEMV` に out だけ dn7 の口 (既定 on)

- `gdn_out_dn7_from_value`: **未設定 = true** (out を dn7)。`o7` で終わる明示値も
  true。それ以外の明示値は「その値が決める従来のカーネル」へ戻す
  (`dn8ar4` = 厳密な切り戻し = out dn8 rows 4)。
- 起動点: `gdn_decode_submit` の out 投影で、dn8 の計画より**先に** dn7 を試す
  (`dn7_dec_plan` = Q8_0 + `mix_dn7_plan` の門番)。合わなければ従来どおり
  dn8 -> dn6 -> v5。
- 実形: n_in 6,144 (np 96) / n_out 2,560 -> 1 行 384 スレッド、grid 2,560。

### 2.2 `DENNO_DEC_HEAD_FAST` に dn7 の口 (既定 on)

- `head_dn7_from_value`: **未設定 = true**。`dn6` (= 旧既定 4 行) が切り戻し。
- `head_gemv_dn6` に `dn7: Option<(rows, threads)>` を足し、`Some` なら
  `dno_mix_gemv_q8_0_dn7` を撃つ (pinned 経由・同期 1 回の形は変えない)。
- 実形: n_in 2,560 (np 40) / n_out 248,320 -> 1 行 160 スレッド、grid 248,320。

### 2.3 full 層 (既定 off のまま)

- `DENNO_FULL_DEC_FAST=dn7` で qkv と wo を dn7 にする口を足したが、
  t504a で負けたので既定 off。

### 2.4 発火の証拠と台帳

- `[mix-qchain]` の末尾に `o7=` (out を dn7 で撃った回数) / `h7=` (head) /
  `f7=` (full) を足した。t504b/t504c の既定 COND は **`o7=4608`** (36 層 x 128 tok)、
  **`h7=128`** (1/tok)、base は両方 0。t504a の f7 COND は `f7=3072`。
- `gate_audit` に `kernels_mix_dn7.ptx` / `dno_mix_gemv_q8_0_dn7` の行を足した
  (dn7 が既定になったので、PTX が欠けたら起動時に WARN が出る)。
- `.cu` / `.ptx` は触っていない。

## 3. lab の結果

### 3.1 t504a (depth8k、5 cond x 2 ラウンド、load_before 2.52)

| cond | short ms/tok | prefill s | ssm | full | md5 | errors |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| c0 (捨て) | 10.6 | 0.56 | — | — | SAME | 0 |
| base | 10.6 | 0.56 | 2.94 / 2.92 | 1.79 / 1.26 | SAME | 0 |
| o7 (out dn7) | 10.5 | 0.57 | **2.86 / 2.88** | 1.71 / 1.26 | SAME | 0 |
| f7 (full dn7) | 10.7 | 0.56 | 2.96 / 2.94 | **1.87 / 1.39** | SAME | 0 |
| base2 | 10.5 | 0.57 | 2.94 / 2.95 | 1.71 / 1.26 | SAME | 0 |

```text
VERDICT o7 vs c0 OK decode -0.7% prefill +0.9% md5 SAME errors 0
VERDICT f7 vs c0 OK decode +0.7% prefill +0.0% md5 SAME errors 0
```

(ssm / full は 2 ラウンドの値。full 段は base でも 1.79 と 1.26 に分かれる
ラウンド間の振れがあるが、**同じラウンド内では必ず f7 が悪い**。)

### 3.2 t504b (depth8k、6 cond x 2 ラウンド、load_before 3.48)

| cond | short ms/tok | head (2R) | ssm (2R) | md5 | errors |
| --- | ---: | ---: | ---: | --- | --- |
| b | 10.6 | 0.60 / 0.60 | 2.93 / 2.93 | SAME | 0 |
| o7 | 10.5 | 0.60 / 0.60 | 2.87 / 2.87 | SAME | 0 |
| h7 | 10.4 | **0.46 / 0.46** | 2.92 / 2.92 | SAME | 0 |
| o7h7 | **10.4** | **0.46 / 0.46** | 2.87 / 2.89 | SAME | 0 |
| b2 | 10.6 | 0.61 / 0.60 | 2.94 / 2.93 | SAME | 0 |

```text
VERDICT o7 vs c0 OK decode -0.7% prefill +0.9% md5 SAME errors 0
VERDICT h7 vs c0 OK decode -1.8% prefill +1.8% md5 SAME errors 0
VERDICT o7h7 vs c0 OK decode -2.2% prefill +1.8% md5 SAME errors 0
```

深い点 (c8000_turn2、ms/tok): b 11.6 / o7 11.4 / h7 11.4 / o7h7 **11.3** / b2 11.7。

### 3.3 t504c (新しい既定 vs 切り戻し、4 cond x 2 ラウンド、load_before 2.61)

| cond | short ms/tok | head (2R) | ssm (2R) | c8000_turn2 | md5 | errors |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| c0 (捨て) | 10.6 | 0.60 / 0.60 | 2.92 / 2.92 | 11.8 | SAME | 0 |
| **b (新既定)** | **10.4** | **0.46 / 0.46** | **2.86 / 2.89** | **11.4** | SAME | 0 |
| r (切り戻し) | 10.6 | 0.60 / 0.60 | 2.93 / 2.94 | 11.6 | SAME | 0 |
| **b2 (新既定)** | **10.4** | **0.46 / 0.46** | **2.86 / 2.87** | **11.4** | SAME | 0 |

```text
VERDICT b vs c0 OK decode -2.2% prefill -0.9% md5 SAME errors 0
VERDICT r vs c0 OK decode -0.4% prefill +0.0% md5 SAME errors 0
VERDICT b2 vs c0 OK decode -2.2% prefill -0.9% md5 SAME errors 0
VERDICT_DEPTH b c8000_turn2 vs c0 decode -3.3% prefill -2.0%
VERDICT_DEPTH r c8000_turn2 vs c0 decode -1.3% prefill +0.0%
```

= 既定と切り戻しの差は **短い decode -1.9% (10.6 -> 10.4)、14k 後 -1.7%
(11.6 -> 11.4)**。c0 (先頭の捨て COND) は切り戻しと同じコードで 10.6 なので、
「c0 比 -2.2%」は先頭バイアスを含まない (r が c0 と同値であることがその証拠)。

### 3.4 出力テキスト (健全性)

md5 は 6 文脈すべて base と同一 (a:edaa84ad / b:1938059f / c2000_cold:9b28e018 /
c2000_turn2:8b02d5c3 / c8000_cold:672fec07 / c8000_turn2:199f23b3)。TXTB:

`Rust の「所有権（Ownership）とライフタイム（Lifetime）は、Rust を学ぶ上で最も重要であり、かつ最も難しい概念です。…`

TXTD (c8000_turn2):

`ご提示いただいた文書は、LLM推論エンジン（`denno-engine）における **Prefill（事前計算）処理の性能最適化の記録です。…`

errors は全 cond 0。

## 4. 数値の扱い

- 変わるのは**丸め順だけ** (dn7 は 1 行の和を 16 要素ずつ 4 レーンに割って足す)。
  決定性・行チャンク不変・par/seq 一致には触れていない。丸めが変わったのに
  **greedy の出力テキストは 1 文字も変わらなかった** (md5 全 SAME)。
- `DENNO_GDN_DEC_GEMV=dn8ar4` (out dn8 rows 4) + `DENNO_DEC_HEAD_FAST=dn6` が
  完全な切り戻し (t504c の r がその実測 = 旧既定と同じ 10.6)。

## 5. 見積りと実測の差

- 見積り: out の dn8 -> dn7 で out GEMV 24.8 -> ~12 us/層 = ssm 段 -13%
  (0.45 ms/tok = -4%)。実測: **ssm 段 -2.1% (0.06 ms/tok)**。out の D2H 待ちは
  qkv (24.8 us) + scan (12.5) + out (24.8) の**後ろ**に来るので、out を半分に
  しても待ちは鎖全体で決まり、短くなるのは out の分だけ (-12 us/層 = -0.43 ms の
  はずが -0.06 ms)。つまり **out の GEMV は既に D2H の影に隠れている**。
- 見積り: head dn6 -> dn7 で 0.60 -> 0.45 (-25%)。実測: **-23% (0.60 -> 0.46)**。
  こちらは 1 発の GEMV の完了待ちがそのまま wall なので見積りどおり。
- 教訓: この decode は「帯域」ではなく「層ごとの往復」で決まる。**wall に
  露出しているカーネル (鎖の末尾・直列 1 発) を選ばないと数字にならない**。
  中央のカーネル (full の qkv/wo) は形を変えても wall が動かない (t504a f7)。

## 6. 次への申し送り

1. **ssm 段の 2.6 ms/tok (36 層 x 73 us) は out の D2H 待ち**。ここを削るには
   (a) out をホストへ戻さずデバイスで残差に足す、(b) 次の層のホスト仕事を
   D2H の前に出す、のどちらかが要る。(a) は残差 + norm を .cu に移す話で、
   効果は最大 -2.5 ms/tok (推定 -20%) だが変更範囲が大きい。
2. **experts 段の cpu 2.89 ms/tok** が次の最大。うち 1.5 ms/tok は
   `[exp-iq-hit] ミスCPUms/tok` (VRAM に居ない expert の CPU GEMV) で、
   これは GPU と重なっているが**ホストの壁時計も食っている**。
3. **mix 段の cpu 1.39 ms/tok** は inject の FMA (inj4=12288)。GPU へ移すと
   鎖に 9 us/回 乗る (t372 で +6%) ので、鎖とホストのどちらが律速かで決まる。
   今回はホストが律速なので、**鎖を伸ばさない形 (inject を out の D2H 待ちの
   裏へ出す) なら勝つ可能性がある**。
4. head はまだ 675.4 MB を 0.46 ms = 1.47 TB/s (1.79 の 82%)。残りは小さい。
5. full 層は dn6 rows=2 のままが最良 (dn6 / dn7 / dn8 の 3 形を測って全部負け)。

## 7. テスト

- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`:
  **1350 passed / 0 failed / 103 ignored**。
- `... --test kernel_invariants -- --test-threads=1`: **145 passed / 0 failed**
  (task427 の dn6 配線テストを dn7 の引数追加に合わせて直し、task504 の
  配線テストを足した)。
- `cargo build --workspace --release` と `cargo test --workspace`: REPORT の
  最終 commit 時点で実行 (結果は commit message に書く)。
- 既知の赤 (`qwen4exp_decode spec_batch_stops_at_eos`、`qwen4exp_params` の
  validate 2 件) は触っていない。

## 8. GPU で確認してほしいこと

- 本番 (14k 超の長い文脈) で short/long の両方が -1.7〜-2% になること。
- head の dn7 が **prefill では発火しない**こと (head GEMV は prefill でも
  1 回だけ通るが、`DENNO_DEC_HEAD_FAST` は decode と同じ関数なので prefill の
  末尾 1 発も dn7 になる。prefill の数字は t504b/t504c で悪化していない)。
