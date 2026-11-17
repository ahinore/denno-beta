ADOPT: no t281b

# REPORT P281 — IQ3_S decode の次: GDN beta/alpha の単スレッド、v5 GEMV の lane 上限、expert 段の無駄な確保

br281 = int-auto `f1d7413e5` (br278 の merge 後)。task 277 の br277 は **採否 no** で
int-auto に入っていない (v4 は v3 より 17% 遅い) ので、そのコードは前提にしていない。
int-auto はこの作業中に動かなかった (最終確認でも `f1d7413e5` = merge 不要)。

## 0. 結論

**ADOPT: no。** 目標「短い decode が 5% 以上速い」に対して、2 つの lab
(t281a / t281b) の最良は **-2.5%** (t281b の `stack` 18.9 -> 18.4 ms/tok)。
内訳は **BF16 GEMV の小 n 行並列が単独で -1.9%** (bit 一致・prefill 不変)、
v5 lane 上限と expert 段の確保除去がそれぞれ -0.4% (ノイズ床 0.4% 程度)。
5% には 0.45 ms/token 足りない。3 つのゲートはすべて**既定 off** のまま残す
(`DENNO_BF16_SMALLN_PAR` は次の decode タスクが既定 on を前提にしてよい、
根拠は §6)。

## 1. 前提 (測り直した値)

task 文の lab `tint_9c5184c20` と `done/tint_f1d7413e5` はどちらも短い decode
18.2 ms/tok、prefill 0.56 s。**int-auto は task 277 以降 decode が動いていない**
ので、今回のベースはそのまま 18.2〜18.9 (run 間で 0.5 程度動く)。

`[q4ledger]` の 1 トークン (t277c の toss = v3 / norm_par が既定 on になる前):
mix 3.84 / ssm 4.28 / experts 5.96 / full 1.65 / router 1.36 / shared 0.64 /
head 0.74。ssm の内訳は 1 層あたり GPU 85 us + **ホスト 27 us** で、ホスト側は
ほぼ全部が GDN の beta/alpha 射影 (t269a の `[q4ledger] ssm` の cpu 列 = 0.98 ms/tok)。

**今回いちばん大きかった誤解の修正**: この beta/alpha は F32 ではなく
**BF16・n=96 (491 KB/層)** で、`GgufLinear::gemv` の既定の行しきい値
(`PAR_MIN_ROWS` = 512 行) に掛からないため **rayon に 1 タスクも渡らず
単スレッド**で走っていた (実測 22 GB/s)。

## 2. 変えたこと (3 つ、すべて既定 off のゲート)

### 2.1 `DENNO_BF16_SMALLN_PAR` — 小 n の BF16 GEMV を行並列に乗せる

`GgufLinear::gemv_impl_par` の BF16 高速経路で、`row_par_min == 0` (既定呼び出し)
かつ `n <= 256` のとき行しきい値を 8 行にする (`bf16_default_row_min`)。
n=96 なら 12 タスク = 8 スレッド。**行は独立・行内の d 昇順累積は不変**なので
値はビット一致。ssm の 36 層それぞれで効く。

### 2.2 `DENNO_GEMV_V5_LANES` — v5 GEMV の lane 上限

`plan_moe_gemv_v5` の上限を env から (`parse_v5_lanes` / `v5_lanes_cap`、既定 8 =
従来)。v5 の lane は「per_unit が割り切れる最大の 2 の冪」なので、効くのは
Q8_0 k=6144 (per_unit 96) だけ = ssm の out 投影 (16.7 MB/層、560 GB/s)。
k=2560 (per_unit 40) の qkv / full / head は 16/32 でも 8 のまま動かない。
**丸めは変わる** (レーン鎖の長さと lane 木の段数)。

### 2.3 `DENNO_EXP_SCRATCH` — sumdev 経路の 40 KB ゼロ埋め

`moe_experts_iq_gpu_batched` は毎層 `h = vec![0.0; nseg*e]` (実形 40 KB) を確保するが、
`DENNO_DEC_EXP_SUMDEV` (既定 on) の経路は `h` を 1 度も読まない。on では確保を
遅延させる。**値は不変** (48 層で 1.9 MB/token のゼロ埋めが消える)。

## 3. lab

### t281a — SHA `da3442d23`、depth8k、6 COND (toss / v5l16 / v5l32 / scr / all / base2)

短い decode (a1/a2 の min、ms/tok) と 14k の次:

| COND | a1 | a2 | 14k の次 | prefill 14k | md5 |
| --- | --- | --- | --- | --- | --- |
| toss (最初) | 18.98 | 16.25 | 22.42 | 13.58 s | 基準 |
| v5l16 | 18.75 | 16.17 | 18.52 | 10.35 s | SAME |
| v5l32 | 18.75 | 16.17 | 17.27 | 8.36 s | SAME |
| scr | 18.75 | **16.09** | 17.50 | 8.38 s | SAME |
| all | **18.59** | **16.02** | 18.44 | 10.06 s | SAME |
| base2 | 18.83 | 16.25 | 17.50 | 8.35 s | 基準 |

`VERDICT` 行は median で `v5l32 -0.5% / scr -0.5% / all -1.1%` (base2 比)。
**ssm の wall は base2 4.43 -> v5l32 4.38 / all 4.40 ms/tok** しか動かず、
v5 lane の見積り (-11 us/層) は外れだった (16.7 MB の読みは lane を増やしても
560 GB/s のまま = 律速は warp 数ではない)。

### t281b — SHA `371ffd665`、depth8k、4 COND (toss / bf / stack / base2)

| COND | a1 | a2 | b | c2000 の次 | 14k の次 | prefill 14k | md5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| toss (最初) | 18.83 | 16.17 | 20.62 | 16.95 | 18.75 | 8.50 s | 基準 |
| bf | 18.52 | 15.86 | 20.42 | 16.72 | **16.95** | 9.98 s | SAME |
| stack | **18.44** | **15.70** | **20.21** | **16.72** | **16.80** | 8.31 s | SAME |
| base2 | 18.91 | 16.17 | 20.62 | 16.95 | 17.50 | 8.34 s | 基準 |

`VERDICT bf vs toss OK decode -1.9% prefill +0.0% md5 SAME errors 0`
`VERDICT stack vs toss OK decode -2.1% prefill +0.0% md5 SAME errors 0`
`VERDICT base2 vs toss OK decode +0.4% prefill +0.0% md5 SAME errors 0` (ノイズ床)
`VERDICT_DEPTH bf c8000_turn2 vs toss decode -9.6% prefill +0.0%`
`VERDICT_DEPTH stack c8000_turn2 vs toss decode -10.4% prefill +0.0%`
`VERDICT baseline toss errors 0 text ok`

- **bf の -1.9% は短い点の 3 窓 (a1/a2/b) と 14k の次で全部同じ向き**。
  base2 (同じバイナリ・同じ env) は toss に対して +0.4% なので、床の外にある。
- `bf` の 14k cold prefill 9.98 s は 1 サンプルの外れ (同じコードの `stack` が
  8.31、`base2` 8.34、t281a の同じ条件も 8.35-8.38)。prefill は動いていない。
- md5 は **4 COND x 6 本すべて SAME** (t281a の v5l32 も含む) = BF16 の行並列は
  ビット一致、v5 lane の丸め変更は本文を変えない範囲。
- 本文 (`stack`、t281b): "A **B-tree** (specifically, the **B+ tree** variant,
  which is the standard in modern databases) is a self-balancing tree data
  structure ..." / "Rust の「所有権（Ownership）」と「ライフタイム（Lifetime）」は、
  Rust がメモリセーフティ（メモリ安全性）をコンパイル時に保証するための最も
  重要な概念です。" — 健全。

## 4. 数値の扱い

- `DENNO_BF16_SMALLN_PAR` / `DENNO_EXP_SCRATCH` は **on でも値が 1 bit 変わらない**
  (行は独立・h は未読)。lab の md5 も SAME。
- `DENNO_GEMV_V5_LANES=16/32` は**意図した丸め変更** (レーン鎖と lane 木の段数)。
  それでも 6 本の md5 は SAME だった。
- 決定性・行チャンク不変・par/seq 一致・アンカーはどれも触っていない
  (行並列は「行をどのタスクに割るか」だけ、v5 は lane の本数だけ)。

## 5. 見積り (推定) と実測

| 変更 | 見積り | 実測 |
| --- | --- | --- |
| BF16 small-n 行並列 | -0.5 ms (-3%) | **-0.36 ms (-1.9%)** |
| v5 lane 32 (ssm out) | -0.4 ms (-2.2%) | **-0.07 ms (-0.4%)** |
| expert 40 KB 確保除去 | -0.1 ms | **-0.07 ms (-0.4%)** |
| 3 つ積む | -1.0 ms (-5.5%) | **-0.47 ms (-2.5%)** |

BF16 は 22 GB/s -> 推定 3 倍を狙ったが実測は 1.6 倍止まり (8 スレッドでも
491 KB/層は L2/DRAM の立ち上がりが支配的)。

## 6. 引き継ぎ (5% に足りない 0.45 ms の居場所)

1. **`DENNO_BF16_SMALLN_PAR` は既定 on にしてよい** (-1.9%、bit 一致、prefill
   不変、t281b の 4 窓で同方向)。既定にすると短い decode が 18.9 -> 18.5。
2. **ssm の GDN はもうホスト側が細い**。残りは GPU の 3 本
   (q8_0_v5 29.8 x 2 + scan 12.7 us/層) で、qkv は 1,057 GB/s = 天井。
   out 投影の 560 GB/s は lane を 4 倍にしても動かなかった (t281a) = 次の手は
   「1 行を 2 warp で受ける」以外に無いが、それは task 277 が expert で失敗した形。
3. **experts の 40 KB は誤差**。`[q4ledger] experts` の 5〜6 ms のうち
   GPU 2.5 ms / ミス CPU 2.8 ms が本体で、ミスは VRAM 22 GB のヒット率で決まる。
4. mix (3.8 ms) は task 216 が「機構の変更では 0.3-1%」「down は uint4 化でも
   512 スレッド化でも縮まない」を実測済み。**触っていない**。
5. 5% を狙うなら残るのは **full 層 (1.65 ms、12 層 x 137 us) の 2 本の GEMV**
   (260-370 GB/s) か、**1 トークン 1,021 起動の固定費** (CUDA graph / device
   dispatch = 禁止リスト) だけ。前者は qkv が 2〜3 unit に割れている
   (`qkv_units` の unit ごとに 1 起動) ので、unit を 1 本に畳むカーネルが要る。

## 7. テスト

- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`
  → **1172 passed / 0 failed** (100 ignored)
- `cargo test --release -p denno-engine --features cuda --test kernel_invariants`
  → **93 passed / 0 failed**
- 追加: `cuda::tests::v5_lanes_cap_table` / `v5_lanes_shape_table`、
  `qwen4exp::tests::exp_scratch_gate_truth_table`、
  `linear_gguf::tests::bf16_smalln_par_gate_and_thresholds`、
  kernel_invariants の `gemv_v5_kernels_have_the_v4_abi` を新しい配線に更新。
- cuda feature 無しの `cargo build -p denno-cli` は **int-auto の時点で既に赤**
  (`qwen4exp.rs` の `expert_vram::ExpertShape` が cuda 限定、ほか E0433 が 20 数件)。
  今回の変更箇所 (cuda.rs:6090 付近 / linear_gguf.rs:190-240 / qwen4exp.rs:6280・
  12655・12732) では出ていない。

## 8. merge

int-auto は `f1d7413e5` のまま (この作業中に動かず)。`build_ptx.sh` の
`targets_all` は触っていない (新しいカーネルを足していない)。
