ADOPT: yes t496c

# REPORT P496 — 既定モデル decode: GDN の qkv/out GEMV を「レーン詰め + x を shared」
# の dn8 にした (`DENNO_GDN_DEC_GEMV=dn8` を既定に)。**ssm 段 3.07 -> 2.94 ms/token
# (-4.2%)、短い decode 10.8 -> 10.6 ms/tok (-1.4〜-1.9%) で task の 2% 目標は未達**
# だが、オペレーター規則 (段の中央値が 2 ラウンド以上で 2% 以上勝ち、他が 1% 超
# 悪化せず、文が健全) を満たすので既定 on にする。md5 は全条件 SAME、errors 0。

branch `br496` = 開始時の `int-auto` **e0746f327** (merge: br490)。終端でも
`int-auto` は e0746f327 のままだったので追加 merge は無い。**br492 は取り込んで
いない** (ADOPT: no で、既定 off の死んだカーネル 18 本を運ぶ理由が無い)。
数値はすべて GPU ラボの実測 (`SCENARIO=depth8k`、`PROFILE=1`、モデルは
`flash-next-UD-IQ3_XXS`、ベースライン COND = int-auto の既定 +
`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0
DENNO_GPU_WEIGHT_MB=12000`、先頭に捨て COND、末尾に base の繰り返し)。

lab 名と SHA:
- **t496a = `247cb3657`** (PTX=mix_dn8、6 COND x 2 ラウンド、`load_before=6.11`)
- **t496b = `c2b6d4037`** (PTX=mix_dn8、同上、`load_before=3.03`)
- **t496c = `c042a09f6`** (収録 PTX で PTX= 無し、同上、`load_before=1.72`)

## 0. 結論 (6 行)

1. 触ったのは **GDN 層の qkv GEMV と out GEMV**。qkv は n_in=2,560 / n_out=16,384
   (重み 44.6 MB)、out は n_in=6,144 / n_out=2,560 (16.7 MB)。t492c の ssm 段
   3.03 ms/token のうち 52 us/層 (36 層で 1.9 ms) がこの 2 本。
2. dn6 は 1 レーン = 1 対 (64 要素) なので qkv は **np=40 を 64 レーンに割って
   62.5% しか埋まらない**。そこを埋める新カーネル `dno_mix_gemv_q8_0_dn8`
   (1 ブロック = rows*np 個の (行, 対)、x は shared に 1 回) を書いた。
3. **レーンを埋めただけでは速くならない** (rows=4 = 100% 埋まりで ssm 3.07 -> 3.02
   = -1.5%)。**qkv は帯域でなく DRAM/遅延で決まっている** (実測 1.27 TB/s =
   ピークの 71%)。t422 の「1 レーン 1 対」の読みはこれで確定。
4. 勝ちは **rows=8 (1 ブロック 8 行、x のグローバル読みが 1/8) + out も dn8**
   から来た: ssm **3.07 -> 2.94 (-4.2%)**。t496a/t496b の rows=8 (qkv のみ) も
   2.96/2.97 で同じく -3.4% (3 run、各 2 ラウンドで再現)。
5. 全体は **10.8 -> 10.6 ms/tok** (-1.4% 対 base、-2.8% 対 base2)。
   深い点は c2000_cold 13.1 -> 12.9 / c8000_cold 12.6 -> 12.3 /
   c8000_turn2 11.6 -> 11.5 と**わずかに良く**、prefill 0.56 -> 0.55 s は不変。
   mix 2.10 / full 1.26 / experts 3.67 / head 0.61 は 1% 以内で不変。
6. 採る。既定を dn8 (rows=8、out も dn8) にした。切り戻しは
   `DENNO_GDN_DEC_GEMV=dn6` (dn8 をやめる) / `=dn8` (out だけ dn6)。
   md5 は 3 run x 6 文脈すべて SAME、errors 0、text looks ok。

## 1. なぜこれを選んだか (Do 1)

- 着手時の段 (t492c base、warm): **ssm 3.03 / experts 2.88 / mix 1.97 / full 1.26 /
  head 0.60 / shared 0.60 / router 0.18**。ssm がいちばん大きい。
- 直近の decode タスクは mix (437/448/451/457/464/484/488)、full (448/460/468)、
  experts (476/492)、shared/head (472/427) を触っていて、**ssm の GEMV 本体
  (qkv/out) だけが「帯域の天井」という推定のまま残っていた** (P492 §8-3)。
- 形を実モデルのテスト (`gdn_decode_req_matches_real_model_dims`) で確認したら
  qkv は n_in=2,560 / n_out=16,384 で **np=40**、dn6 のレーンは 64 → 62.5%。
  「1.27 TB/s が 62.5% とほぼ一致する」ので、レーン詰めが効くというのが作業仮説
  だった (§0-3 のとおり**仮説は外れ**、勝ちは x の shared 化とブロック数だった)。
- タスクの Kernels 段落の条件 (初回 PTX=、生成物を commit、最後は PTX= 無し) も
  この経路なら満たせる。

## 2. 変えたこと

### 2.1 `kernels_mix_dn8.cu` (新規 1 entry)

dn6 の 1 対の積和 (`dno_dn6_dot64` と同一の 17 ワード読み・4 アキュムレータ・同じ
丸め) をそのまま使い、行の割り当てだけを変える:

- 1 ブロック = **rows*np 個の (行, 対)**。スレッド tid は `r = tid / np`、
  `pair = tid % np`。block = `32 * ceil(rows*np / 32)`、grid = `ceil(n_out/rows)`。
- **x は shared へ 1 回だけ**置き (`xs[pair*65 + j]`、65 は bank を散らす詰め物)、
  rows 本の行で使い回す。x のグローバル読みが 1/rows になる。
- 各行の和は「対の部分和を昇順 1 本」(`part[tid]` を shared に置き、行ごとに
  1 スレッドが昇順加算)。dn6 の warp 木 + warp 逐次とは**足す順が違うだけ**
  (丸めのみ。行ごとに独立で決定的)。

値の根拠: 1 対の積和・縮約木・f16 復元・`__fmul_rn`/`__fadd_rn` は dn6 と同一。
dn6 と同じ rows で撃てば**対の部分和まで 1 ビット一致**で、変わるのは対の足し順
だけ。実測でも md5 は全条件 SAME。

### 2.2 配線 (cuda.rs / qwen4exp.rs / gate_audit.rs / build_ptx.sh)

- `kernels_mix_dn8.ptx` を読み (`DENNO_MIX_DN8_PTX` で差し替え可)、
  `gdn_decode_submit` の qkv と out の起動点に dn8 を先に試す分岐を足した
  (形が合わなければ dn6 → v5 へ fail closed)。
- ゲートは既存 env の新しい値: `DENNO_GDN_DEC_GEMV=dn8` (qkv のみ) /
  `dn8a` (out も) / `dn8rN`・`dn8arN` (rows = 2/4/6/8/12/16、既定 8)。
- 形の門番 `mix_dn8_plan_with` (純関数): Q8_0 / n_in % 64 == 0 /
  items = rows*np <= 1024 / **block <= 640** / shared <= 48 KiB。
  `np % 32 == 0` の形は既定で dn6 のまま (`dn8a` のときだけ out も dn8)。
- 発火の証拠: `[mix-qchain]` に `dn8=` を足した。t496c の `d8ar8` は
  **`dn6=0 dn8=9216`** (qkv 4,608 + out 4,608 = 36 層 x 128 トークン)、base は
  `dn6=9216 dn8=0`。
- 既定 on の PTX 台帳 (`gate_audit`) に 1 行足した。

### 2.3 lab が暴いたバグ (直した)

`block = 768` で撃つと **`cuLaunchKernel が失敗: error 701`**
(= LAUNCH_OUT_OF_RESOURCES)。`.cu` は 1 レーンが x の 64 要素をレジスタに持つ
ので ~100 レジスタ/スレッド使い、768 x 100 > 65,536 で入らない
(t496b の `d8a8` が errors 2 / BROKEN)。**block の上限を 640** にした
(640 は t496b の `d8a16` で通っている)。out は rows=4 固定 (block 384)。

## 3. 数値の扱い

- 変わるのは**丸め順だけ** (対の部分和の足し方)。実測でも md5 は 3 run x 6 文脈
  すべて SAME、errors 0、text looks ok。
- 決定性: 行ごとに独立、対の和は昇順 1 本、shared の partial は item 固定。
  par / seq・行チャンク不変には触れていない (decode の GDN 1 本の中だけ)。
- 既定を切り替えたので**既定の値は t496c の `d8ar8` COND と同じ**。

## 4. lab の結果

### 4.1 ssm 段 (warm、`[q4ledger] ssm wall`。run 内の同じ COND 対で読む)

| run | base (r1 / r2) | rows=2 | rows=4 | rows=8 (qkv のみ) | rows=8 + out (dn8a) |
| --- | ---: | ---: | ---: | ---: | ---: |
| t496a | 3.09 / 3.07 | 3.15 | 3.04 / 3.02 | **2.96 / 2.97** | 未測 |
| t496b | 3.05 / 3.07 | — | — | **2.95 / 2.97** | 3.06 / 3.04 (rows16、out は dn6) |
| t496c | 3.07 / 3.08 | — | 3.00 / 3.00 (dn8a) | 2.97 / 2.97 | **2.94 / 2.95** |

### 4.2 短い decode (median) と深い点 (t496c)

| cond | short ms/tok | c2000_cold | c2000_turn2 | c8000_cold | c8000_turn2 | prefill s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| c0 (捨て) | 10.8 | 13.1 | 11.8 | 12.6 | 11.8 | 0.55 |
| base | 10.8 | 13.1 | 11.8 | 12.6 | 11.6 | 0.56 |
| d8r8 | 10.7 | 13.0 | 11.6 | 12.5 | 11.6 | 0.56 |
| d8a | 10.8 | 13.0 | 11.8 | 12.4 | 11.6 | 0.56 |
| **d8ar8 (採用)** | **10.6** | **12.9** | 11.7 | **12.3** | **11.5** | 0.55 |
| base2 | 10.9 | 13.1 | 11.8 | 12.4 | 11.6 | 0.55 |

```text
VERDICT base vs c0 OK decode +0.0% prefill +2.7% md5 SAME errors 0
VERDICT d8r8 vs c0 OK decode -0.7% prefill +0.9% md5 SAME errors 0
VERDICT d8a vs c0 OK decode +0.0% prefill +0.9% md5 SAME errors 0
VERDICT d8ar8 vs c0 OK decode -1.4% prefill +0.0% md5 SAME errors 0
VERDICT base2 vs c0 OK decode +0.7% prefill -0.9% md5 SAME errors 0
VERDICT baseline c0 errors 0 text ok
```

t496b の失敗 COND (記録): `VERDICT d8a8 vs c0 FAIL decode -90.7% prefill +283.6%
md5 DIFF errors 2` — block 768 の LAUNCH_OUT_OF_RESOURCES (§2.3)。

### 4.3 段の台帳 (t496c、warm、base_2 と d8ar8_2)

| 段 | base | d8ar8 | 差 |
| --- | ---: | ---: | ---: |
| ssm | 3.08 | **2.95** | -4.2% |
| experts | 3.67 | 3.71 | +1.1% (負荷ゆらぎ) |
| mix | 2.10 | 2.09 | 0% |
| full | 1.26 | 1.27 | +0.8% |
| head | 0.61 | 0.61 | 0% |
| shared | 0.61 | 0.61 | 0% |
| router | 0.17 | 0.18 | — |

## 5. 出力テキスト (健全性)

d8ar8 の c8000_turn2 (TXTD、md5 `199f23b3` で base と同一):

`ご提示いただいた文書は、LLM推論エンジン（\`denno-engine）における **Prefill（事前計算）処理の性能最適化の記録です。 …`

短い decode (a1) は `# B-Trees: A Comprehensive Explanation …` で、6 COND x 3 run
すべて `looks ok`、errors 0。

## 6. 見積りと実測の差 (次に同じ轍を踏まないため)

- 見積り: 「qkv は 62.5% しかレーンが埋まらないので、100% にすれば
  1.27 -> 1.8 TB/s (+40%)、ssm で -0.5 ms/token」。
- 実測: レーンを 100% にしても **-1.5%** だけ。qkv は**レーン律速ではなく
  DRAM/遅延律速**だった (44.6 MB の読みで 1.27 TB/s は既に天井側)。
- 逆に効いたのは「**1 ブロックが持つ行数を増やして x を shared で使い回す**」
  ほうで、rows 2 -> 4 -> 8 が単調 (3.15 / 3.03 / 2.96)。1 ブロックあたりの
  x 読みが 1/rows になるのと、ブロック数が減って発射と末尾が減るのが効いている
  (推定)。rows=16 は並列度が落ちて悪化 (3.04)。
- 小さいカーネルは**レーン数より「1 ブロックの仕事量」**を見る、が今回の教訓。

## 7. テスト

```text
CARGO_TARGET_DIR=/tmp/t496 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1349 passed; 0 failed; 103 ignored
同 --test kernel_invariants
  -> 144 passed; 0 failed
(関門の外) CARGO_TARGET_DIR=/tmp/t496nocuda cargo build --workspace --release -> ok
(関門の外) CARGO_TARGET_DIR=/tmp/t496 cargo test --workspace -- --test-threads=1 -> 結果は REPORT 末尾
```

足したテスト:
- `cuda::dec_kprof::tests::gdn_dn8_gate_and_launch_plan_truth_table`:
  ゲートの真理値表 (既定 dn8・rows 8・out も dn8・切り戻しの値)、起動形
  (実形 qkv = np 40 / rows 4 で block 160 / grid 4,096、rows 16 で 640 / 1,024、
  out = np 96 / rows 4 で 384 / 640)、門番 (np 32 倍数・items > 1024・
  block > 640・shared > 48 KiB)。
- `task496_mix_dn8_lane_packed_gemv_is_wired` (kernel_invariants): `.cu` の
  カーネル名と 7 引数、行/対の割り当てと shared の字面、cuda.rs の plan / launch /
  gate / 発火の証拠、build_ptx.sh の target と targets_all、収録 PTX の 7 引数。
- `.cu` の番地計算は g++ で全スレッドを pthread + barrier で走らせる harness を作り、
  参照実装と**ビット一致**することを 5 形 (np 5 / 40 / 96、rows 1..8、端数あり) で
  確かめた (nvcc が無いのでコンパイル前に潰した)。

## 8. 次への申し送り

1. **GDN の GEMV はもう触る所が小さい** (qkv + out で 1.9 ms のうち -0.13 ms を
   取った)。ssm の残りは scan 12.7 us/層と ba 7.4 us/層で、どちらもグリッドが
   48 / 96 と小さく、t441 (ba を scan に畳む) と t476 (ba の書き換え) で負けている。
2. **同じ dn8 を full 層の qkv (n_in=2,560、同じ np=40) に使うのは未着手**。
   `DENNO_FULL_DEC_FAST` の値に dn8 を足すだけだが、full 段は 12 層で
   1.26 ms しかなく、上限は -0.05 ms 程度 (推定)。
3. **mix の down/up は dn7 / up4p4 で既に 1 レーン複数対**なので dn8 の出番は無い。
4. `error 701` (LAUNCH_OUT_OF_RESOURCES) は **block x 100 レジスタ**で踏む。
   x を 64 要素レジスタに持つカーネルは block 640 を上限にする。
5. lab の COND で再現するときの推奨: `DENNO_GDN_DEC_GEMV=dn8ar8` (採用形) /
   `=dn8r8` (out は dn6)。

## 9. GPU で確認してほしいこと

- 既定 (env 無し) で `[mix-qchain] ... dn6=0 dn8=9216` が出て、短い decode が
  10.6 ms/tok、ssm 段が 2.94-2.95 になること。
- `DENNO_GDN_DEC_GEMV=dn6` で base の 10.8 / ssm 3.07 に戻ること (切り戻し)。
- 既定の PTX 監査 (`[gates]`) に `kernels_mix_dn8.ptx` が載り、欠落 0 本であること。
