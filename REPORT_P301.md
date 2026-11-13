ADOPT: yes t301a t301b t301c

# REPORT P301 — UD-Q4_K_XL: hc mix の Q8_0 GEMM をテンソルコアへ (3,826 cold prefill -22.2%)

branch: br301 = int-auto 999ce35 (br297 merge 済み。作業中に int-auto は動かず、最後の
fetch でも 999ce35 のまま)。lab: t301a (UD、depth8k、SHA e60f3e0 = ゲート既定 off)、
t301b (UD、depth8k、SHA 29b34d2 = ゲート既定 on)、t301c (既定モデル、depth、SHA ca87b52)。

## 0. 結論

- **UD-Q4_K_XL の 3,826 cold prefill は 38.46 -> 29.94 s (-22.2%)** (t301b の b1/b0
  median 39.33/38.46 対 new 29.94)。target の 15% を満たす。t301a も同じ向き
  (b0/b1 median 49.42 -> 32.11 s、ただしこの回は host の load が高く baseline が
  38.22-60.63 s と振れた)。
- **14k cold prefill は 138.31/139.66 -> 110.38 s (-20.2% / -21.0%)**、**mix の
  項目は 31.04 -> 1.09 s (-96.5%)** (t301b)。14k も sane なテキストで完走。
- **既定モデル (swift15-abl-IQ3_S) は 3 COND で md5 4 本すべて一致・decode +0.4%・
  3,826 cold 3.42/3.42/3.41 s** (t301c)。tint_999ce3575 (int-auto の参照) の md5
  とも一致 = 既定モデルは何も変わっていない。
- 変えたのは 1 か所だけ。prefill の hc mix (`pf_mix`) の down / up は、BF16 の
  ときだけ sn3 のテンソルコア GEMM を使っていた。UD-Q4_K_XL の mix は **Q8_0**
  なので従来の量子 GEMM (1 起動 16 行 x 128 トークン、16 スレッド) に落ちていた。
  ここを sn3 の Q8_0 入口に繋いだ (`DENNO_PF_MIX_Q8TC`、**既定 on**)。
- errors 0、text sane (引用は 5 節)。md5 は `c2000_cold` / `c2000_turn2` /
  `c8000_turn2` が DIFF、`a` / `b` / `c8000_cold` は SAME (sn3 の bf16 丸め。
  turn2 が変わるのは cold prefill の KV が変わるため)。**expert の項目は
  14k で 93,312.8/93,267.2 -> 93,364.8 ms と不変** = 変えたのは mix だけである
  ことが台帳で確認できる。

## 1. 前提の測り直し (task の Do 1)

t301a の b0/b1 (int-auto 999ce35 + ゲート off、PROFILE=1) の 3,826:

| 段 | 実測 (b0_1 / b1_2) | 割合 |
| --- | --- | --- |
| experts (CPU、flat MoE) | 28,207 / 25,937 ms | 68-70% |
| mix (GPU、Q8_0 量子 GEMM) | 8,494 / 8,462 ms | 21-22% |
| other | 3.1 s (t297a と同じ) | 8% |
| ssm / full / shared / router | 0.67 / 0.22 / 0.25 / 0.06 s | 3% |

t297a (39.78 s) と同じ構図で、**mix が依然 2 番目に大きい塊 (21%)**、experts が
最大 (68%)。P297 の次の手 1 がこの mix。experts は P297 4.1 のとおり CPU の定式
上限なので触っていない (4.4 節)。

## 2. 変えたこと

### 2.1 mix の Q8_0 GEMM を sn3 へ (`e60f3e0`、既定 off で入れて lab 後に on)

- `pf_mix` の gemm closure は、BF16 の重みだけ `dno_gemm_sn3_bf16_w` (BM=64 /
  BN=64 / BK=32、bf16 mma) に回していた (task 222、lab t222b で既定 on)。
  量子 (Q8_0) は `f_gemm_q` の従来カーネル (1 起動 16 行 x 128 トークン、
  16 スレッド、1 呼び出し 35 起動) のままだった。
- 同じ `kernels_gemm_sn3.ptx` には **Q8_0 の入口** (`dno_gemm_sn3_q8_0_w`) が
  既にあり、shared expert で常用している。mix の down (k=10240 n=320) と
  up (k=320 n=10240) は sn3 の前提 (n_in % 32 == 0、n_out % 64 == 0、
  nt >= 64) を満たすので、型で分岐する match にして Q8_0 も同じ入口へ回した。
- ゲートは新設の `DENNO_PF_MIX_Q8TC` (純関数 + `OnceLock`、`=0` で従来経路)。
  `e60f3e0` では既定 off、lab t301a の結果を見て `29b34d2` で既定 on にした。
  next turn (19 トークン) と inject (n=4) は前提を満たさず従来経路のまま。
- .cu / .ptx は触っていない (既存カーネルへの配線だけ)。

### 2.2 テストの配線チェックを追随 (`ca87b52`)

`kernel_invariants::pf222_sn3_gates_are_wired_and_default_on` は cuda.rs の文字列
(`if pf_mix_tc_on()` / `&& kind == PfWKind::Bf16`) を直接見ていたので、2 アームの
match と「行バイト数を型から渡す」形に合わせ、新ゲートも既定 on の一覧に足した。

## 3. 数値の扱い

- **丸めが変わる**。sn3 は x と重みを bf16 に丸めて mma し、アキュムレートは
  FP32・加算順は k 昇順 (prefill の射影はビット一致契約の外)。従来の f32 SIMT
  カーネルとは値が変わる。Q8_0 の重み (int8) は bf16 で正確に表せるが、
  活性 x の bf16 丸めで 1 要素 ~0.4% の相対誤差が乗る。これは task 222 が
  本番モデルの mix (BF16) で採ったのと同じ丸めで、同じカーネルの同じ経路。
- md5 (t301b): a / b / c8000_cold は SAME、c2000_cold と c2000_turn2、
  c8000_turn2 が DIFF。turn2 が変わるのは cold prefill の KV が変わったため。
- 決定性: t301a と t301b で new の c2000_cold はどちらも f9557457、c8000_cold は
  534fda63 で、run をまたいで一致 = 決定論的。
- t301b の b1 vs b0 が md5 DIFF なのは c8000_turn2 だけ (17b38dde 対 e6398d57)。
  14k cold は両方 534fda63 なので、**14k のあとの decode は同じ構成でも再現しない**
  ことがある (ゲートとは無関係。t301a では 6 本すべて SAME だった)。

## 4. lab の数字 (すべて実測)

### 4.1 t301a — UD-Q4_K_XL、depth8k、PROFILE=1、SHA e60f3e0

| cond | warm short decode | 3,826 cold (2 本) | 14k cold |
| --- | ---: | --- | ---: |
| b0 (ゲート off) | 22.5 | 40.54 / 58.84 (median 49.69) | 143.45 |
| new (既定 off 時点で =1) | 22.3 | **33.96 / 30.26 (median 32.11)** | **115.26** |
| b1 (ゲート off) | 21.9 | 60.63 / 38.22 (median 49.42) | 143.61 |

`VERDICT new vs b0 OK decode -1.0% prefill -10.0% md5 DIFF errors 0`
`VERDICT b1 vs b0 OK decode -2.8% prefill +3.2% md5 SAME errors 0`
`VERDICT_DEPTH new c2000_cold vs b0 decode -11.5% prefill -35.4%`
`VERDICT_DEPTH new c8000_cold vs b0 decode -23.5% prefill -19.7%`
`VERDICT baseline b0 errors 0 text ok`

この回は host の load が 2.9 -> 11.6 と動き、baseline の 3,826 が 38.22-60.63 s と
振れた (experts の cpu が 25.9-48.0 s)。14k は 143.45/143.61 と安定。

### 4.2 t301b — UD-Q4_K_XL、depth8k、PROFILE=1、SHA 29b34d2 (既定 on の最終 SHA)

| cond | 3,826 cold (2 本) | 14k cold | mix 3,826 | mix 14k | experts 14k |
| --- | --- | ---: | ---: | ---: | ---: |
| b0 (off) | 39.53 / 39.12 | 139.66 | 8,499.5 / 8,477.8 | 31,038.3 | 93,312.8 |
| new (既定) | **29.65 / 30.23** | **110.38** | **316.5 / 304.6** | **1,089.8** | 93,364.8 |
| b1 (off) | 39.31 / 37.61 | 138.31 | 8,472.1 / 8,461.8 | 31,050.9 | 93,267.2 |

- **目標対比 (b1 median 38.46 -> new median 29.94) = -22.2%**。b0 median 39.33 対比
  でも -23.9%。
- 台帳の mix は 8.47 s -> 0.31 s (3,826)、31.04 -> 1.09 s (14k)。**experts の
  項目は 3 条件で 93.31/93.36/93.27 s と同一** = 変更が mix に閉じている。
  14k の差 29.28 s は mix の差 29.95 s とほぼ一致する。
- 14k の内訳 (new): experts 93.36 s (84.6%) / other 11.36 s (10.3%) /
  ssm 2.34 / full 1.13 / shared 0.87 / mix 1.09 / router 0.22 s。
- `VERDICT new vs b0 OK decode +13.9% prefill +7.3% md5 DIFF errors 0`
- `VERDICT b1 vs b0 OK decode +3.3% prefill -0.6% md5 DIFF errors 0`
- `VERDICT_DEPTH new c2000_cold vs b0 decode -18.1% prefill -23.9%`
- `VERDICT_DEPTH new c8000_cold vs b0 decode +20.3% prefill -21.0%`
- `VERDICT_DEPTH b1 c2000_cold vs b0 decode -19.0% prefill -2.2%`
- `VERDICT_DEPTH b1 c8000_cold vs b0 decode -14.8% prefill -1.0%`
- `VERDICT baseline b0 errors 0 text ok`

### 4.3 warm short decode について (採否規則の「2% を超えて悪化しない」)

lab の表の `decode ms/tok (median)` は t301b で new 24.6 / b0 21.6 (+13.9%) と出るが、
**この指標は変更の影響を受けないし、2% を判定できる分解能も無い**。

- warm short decode の実体は a1 / a2 (prompt 28、out 128)。この prefill は nt=28 で、
  sn3 の下限 (nt >= 64) に届かないので**新しい分岐に一度も入らない**。decode の
  ループにも変更は無い。
- 要求ごとの ms/tok (a1/a2、2 run x 4 本) を並べると、b0 20.5/21.6/23.4/18.7/
  23.1/18.4/20.0/18.8 (median 20.3)、b1 21.8/22.0/21.8/18.4/21.8/18.6/20.5/22.7
  (median 21.8)、**new 23.2/18.6/21.4/18.6/20.4/18.8/20.2/28.8 (median 20.3)**。
  差は new_2_a2 の 1 本 (28.8 ms/tok) で、同じバイナリの b0 と b1 でも +3.3%
  (21.6 対 22.3) 開く。母集団の median は b0 と new が同値。
- t301a の同じ指標は new 22.3 対 b0 22.5 (-1.0%)。既定モデル (t301c) は
  off 17.9 / on 18.0 / offb 17.9 ms/tok (+0.4%、同一バイナリの offb でも同じ幅)。

### 4.4 14k の最大の塊 (experts) を触らなかった理由

- 14k cold の 84.6% は routed experts の CPU 計算で、3,826 と同じ定式
  (8 expert x 3 行列 x 640 x 2560 = 1 トークン・1 層 39.3 MMAC) を 48 層回す。
  実測は t301b の b1: 14k が 26.46 TMAC / 93.27 s = **284 GMAC/s**、3,826 が
  7.22 TMAC / 28.21 s = **256 GMAC/s**。
- 同じ形をこのコンテナで回した P297 のマイクロベンチは 387.5 ms/層 = 388 GMAC/s
  なので、**実機の prefill はマイクロベンチより 26-34% 遅い** (差分は 3,826 で
  推定 9.6 s、14k で推定 24 s)。GEMM の外 (ページフォルト、活性の量子化、
  expert ごとの gather) に何かが残っている可能性があるが、このコンテナでは
  実モデルを動かせないので、原因の特定は次のタスクの実測が要る。
- 定式そのものを変える案 (CPU の水平加算を削る) は P297 4.1 で -6.7% しか出ない
  と実測済み。CPU の expert GEMM を GPU へ移す案は P266 (pack 付き grouped GEMM が
  4.45 倍遅い) と common rules で棄却済みで、pack を廃した転送カーネルは新しい
  .cu と実機検証が要る (このコンテナに nvcc が無い)。

### 4.5 t301c — 既定モデル (swift15-abl-IQ3_S)、depth、PROFILE=1、SHA ca87b52

| cond | warm short decode | 3,826 cold | md5 (a / b / cold / turn2) |
| --- | ---: | ---: | --- |
| off (`=0`) | 17.9 | 3.42 | 65ededc3 / f18c75fc / d4b6eefb / 8ff9345f |
| on (既定) | 18.0 | 3.42 | 同上 (完全一致) |
| offb (`=0`) | 17.9 | 3.41 | 同上 (完全一致) |

`VERDICT on vs off OK decode +0.4% prefill -0.8% md5 SAME errors 0`
`VERDICT offb vs off OK decode +0.2% prefill +0.8% md5 SAME errors 0`
`VERDICT baseline off errors 0 text ok`

既定モデルの mix は BF16 なので新しいアームを通らない (bf16 のアームは task 222 の
まま)。md5 は tint_999ce3575 (int-auto 999ce35 の参照 run) とも同じ。

## 5. text (TXTD、new、そのまま引用)

- 3,826 cold の続き (t301b): 「ご提示いただいた文書は、LLM推論エンジン
  （`denno-engine）における **Prefill（事前計算フェーズの性能最適化の歴史と
  教訓を詳細に記録したものです。..."
- 14k cold の続き (t301b): 「ご提示いただいた文書は、LLM推論エンジン
  （`denno-engine）における **Prefill（事前計算）フェーズの性能最適化の歴史を
  詳細に記録したものです。..."
- TXT (a) は "# B-Trees: A Comprehensive Explanation ..."、TXTB は
  「Rust の「所有権（Ownership）とライフタイム（Lifetime）...」で、b0 と同じ書式。
  b0 と new で同じ段落から始まり、内容も同等 (md5 が違うのは数値の丸めだけ)。

## 6. テスト

- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`
  **1189 passed / 0 failed / 101 ignored**
- `cargo test --release -p denno-engine --features cuda --test kernel_invariants`
  **95 passed / 0 failed**
- 追加: `pf_mix_q8_tc_gate_truth_table` (既定 on と偽値の真理値表、強制上書きの往復)、
  `pf_mix_q8_tc_ud_shapes` (3,826 のチャンク 1,913 と 14k の 2,047 で down/up が
  sn3 の前提を満たすこと、19 トークンと inject n=4 は落ちること)。
  `kernel_invariants` の配線チェックは新しい 2 アームに追随 (`ca87b52`)。
- 既知の赤 (`qwen4exp_decode spec_batch_stops_at_eos`、`qwen4exp_params` の
  validate 2 件) はこのツリーでは緑 (task 297 と同じ)。

## 7. GPU で確認してほしいこと / 次の手

1. 本番 (8098) の UD-Q4_K_XL で 3,826 と 14k を 2 回ずつ。台帳の `mix` が
   8.5 s -> 0.3 s / 31 s -> 1.1 s になり、`experts` の項目が変わらないこと。
   `DENNO_PF_MIX_Q8TC=0` で従来 (mix が 10 倍以上) に戻ること。
2. 残りの最大は experts (3,826 で 68%、14k で 84.6%) と `other` (14k で 11.4 s)。
   experts は CPU の定式上限、`other` は未分類のホスト時間 (1 トークン・1 層あたり
   15-18 us) で、中身の特定から始める必要がある。
3. 14k のあとの decode は同じ構成でも md5 が変わる回がある (4 節)。2% 判定に
   使うなら同一 COND を 4 本以上並べた run が要る。
