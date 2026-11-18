ADOPT: yes t419b

# task 419 (dev2) — 既定モデルの prefill: 残差 `other` を閉じ、PLE の gather を 128 本に

branch `br419` = 開始時の `int-auto` **`8cef37b60`** (merge: br415)。私の br415 は既に
int-auto に入っていたので merge は不要だった。作業中に int-auto は
**`a8a92bb33`** (merge: br422 = GDN qkv/out と mix down の dn6) へ動いたので終端で
merge した (§6)。

数値はすべて GPU ラボの実測 (`SCENARIO=depth8k`、`PROFILE=1`、
`model=flash-next-UD-IQ3_XXS`、`ptx=none`)。lab 名と測った SHA:
**t419a = `676b0a94e`** (3 COND、残差の内訳を測る計器を足した回、load 15.21)、
**t419b = `676b0a94e`** (5 COND、PLE の本数の掃引、load 1.77)、
**t419c = `78b73f535`** (6 COND、IQ 系 expert の行数の床、load 2.21)。
3 ラウンド (上限いっぱい)。

## 0. 結論 (6 行)

1. **採る (`ADOPT: yes t419b`)。ただし task の目標 (壁で 2.5%) は満たしていない。**
   採ったのはオペレーター規則 (段で 2% x 2 ラウンド、他は 1% 以上悪化しない) で、
   段は **`[pre-kprof]` の ple**: 14k 415.3 / 414.2 -> **381.4 / 382.9 ms (-7.9%)**、
   3,826 は 4 サンプルの中央値 215.5 -> **197.4 ms (-8.4%)** (t419b、2 COND x
   2 ラウンド、
   全部 `DENNO_START_DROP_CACHE=1`)。既定を `DENNO_PF_PLE_THREADS=128` にした。
2. **壁への寄与は 14k で -0.4%** (b1/b2 の平均 7.02 -> 6.99 s)、**3,826 は 0**
   (壁は expert の読みで決まる)。decode 12.8-12.9 ms/tok で不変、md5 は全 COND
   全コンテキストで **SAME**、errors 0、text sane (§8)。
3. **なぜこの 2 つか (task の Do 2)**: t419a の台帳で **壁の内訳が初めて閉じた**。
   14k は **GPU のカーネル 4,251 ms (58%) + expert の読み待ち 2,638 ms (36%) +
   PLE の gather 416 ms (6%)**、3,826 は **expert の読み待ち 6,849 ms (83%)**。
   ホスト側に大きな遊びはもう無い (層ループの 1 回転でどの区間にも入らない時間は
   14k 361 ms / 3,826 133 ms で、そのほぼ全部が PLE 層の残差往復と注入)。
   その中で **まだ誰も触っていない PLE の gather の本数**と、
   **IQ 系 expert の行数の床 (§5)** を取った。
4. **変えたこと**: (a) `DENNO_PF_PLE_THREADS` の既定 96 -> **128**、
   (b) `DENNO_PF_EXP_IQ_MIN_ROWS` (新規、**既定 0**) で IQ 系 expert にも行数の床を
   置けるようにした (t419c で **採らず**)。残差を測る計器 (`[q4ledger-pf] iter` と
   副区間 6 本) も入れた。
5. **値は 1 bit も変えていない**: t419a/t419b/t419c の全 COND・全文脈で md5 は SAME
   (`edaa84ad / 1938059f / b42d79d2 / 12fc5c36 / 672fec07 / 98e879d2`)。
   PLE の本数は読む順序を変えない (行は独立) ので、丸めも変わらない。
6. **分かったこと (次の担当者へ)**: 既定モデルの prefill はもう**ホスト側ではない**。
   14k の壁 6.2 s (暖) の内訳は (**推定**、t419a の段から) GPU カーネル ~4.25 s +
   expert の読み ~1.7 s + gather ~0.42 s。**残る大きな的は GPU の 4 段**
   (`experts` 1,416 / `ssm` 1,316 / `mix` 881 / `full` 638 ms) で、どれも
   カーネルを触るしかない。

## 1. Do 1: 基準の取り直しと、残差の正体 (t419a)

### 1.1 段の台帳 (`[q4ledger-pf]`、t419a b2、ms)

| 段 | 3,826 wall / busy | 14k wall / busy |
| --- | ---: | ---: |
| mix | 238.8 / 242.2 | 21.4 / **880.8** |
| full | 129.0 / 88.8 | 1,006.0 / **638.0** |
| ssm | 418.2 / 360.3 | **2,192.6 / 1,316.0** |
| experts | **7,083.8 / 516.8** | **3,139.0 / 1,415.9** |
| other | 418.5 / 47.4 | 962.2 / 171.6 |
| close wall | 8,288.3 | 7,321.3 |
| GPU busy 合計 | 1,255.4 | 4,422.3 |

- **3,826 はディスク**: `[exp-pf-stream]` は 10,938 本 / **22,425.6 MB** を読み
  `copy_s=6.93 s` (3.2 GB/s。速い回の上限は P325/P394 の 6.2 GB/s)、
  `[pf-host] up_wait=6,848.8 ms` = 壁の 83%。
- **14k も 3 割はディスク**: 13,616 本 / **27,910.5 MB**、`copy_s=2.71 s`、
  `up_wait=2,638.3 ms`。こちらは page cache が効いて 10.3 GB/s 出ている。
- **`DENNO_START_DROP_CACHE=1` の副作用**: 全 COND でモデルをディスクから読み直すので、
  3,826 の絶対値は task の状態 (2.50 s) と比べられない (同じ run の中だけで比べる)。
  DROP_CACHE 無しの現在の int-auto は tint_b23e2ebf8 の実測
  (3,826 cold 2.38-2.50 s / 14k cold 5.68-6.17 s、decode 15.1-15.2 ms/tok)。

### 1.2 残差 `other` を割った (計器を足した)

`other` は「壁時計 − 名前付き区間の和」の残差で、t415 の引き継ぎでは 14k 1,038 ms の
うち名前が付いていたのは 390 ms だけだった。task 419 で**回転ごとの uncovered** を
直接積む計器 (`q4ledger::iter_begin/iter_end`) と副区間 6 本を足して閉じた:

```
[q4ledger-pf] iter n=336 total=3580.8ms uncovered=360.9ms (1074.0us/iter)   # 14k
[q4ledger-pf] iter n=96  total=919.0ms  uncovered=133.0ms (1385.2us/iter)  # 3,826
[q4ledger-pf] sub other ... ple=213.8 res_up=150.4 pf_teardown=86.1 combine=4.1 res_park=2.5 inj=0.7 pf_setup=0.3 attn_pre=0.0
```

- **uncovered 361 ms (14k) の正体は PLE 層の残差往復 (`res_up` の層ループ内ぶん
  ~100 ms) + PLE の注入 (`sub ple` 213.8 ms のうち層ループ内)**。gather 自体は
  96 本の rayon ワーカーで走るので `sub` にはほとんど乗らない (別スレッドの
  thread_local)。
- `pf_teardown` (追い出した expert の戻しと解放) は 14k で 86 ms、3,826 で 0.8 ms
  (3,826 は追い出し 0 本)。
- 層ループの壁 (= 3,826 919 ms / 14k 3,581 ms) のうち **79% は GPU** で、
  ホストが詰まっているわけではなかった。**これで「other の未計測 ~990 ms」は閉じた。**

### 1.3 選んだ理由と、外した理由

- **PLE の gather (14k 416 ms / 3,826 213 ms)**: 1 行 90 B のランダム pread が
  167k 回 (14k)。本数を増やす方向は task 357 が 64/96/128 を測って 128 で
  386 ms まで下がると書いていたが、**既定は外乱 1 回を理由に 96 のまま**だった。
  t419b で 2 COND x 2 ラウンド取り直して 128 が両方の長さで -8% と確定したので採った。
- **IQ 系 expert の行数の床**: 行の少ない expert を CPU に残す床は task 309 が
  入れたが、条件が `!iq_expert_mode(lw)` なので **K 系 (UD-Q4_K_XL) にしか効かない**。
  既定モデルの expert は IQ3_XXS / IQ2_S / IQ4_NL / IQ4_XS / type 42 の混在で、
  1-2 行しか使わない expert にも 2.05 MB を上げていた。既定 0 のゲートを足して
  t419c で 16/32/64 を掃引した (§5)。**結果は採らず。**
- 外したもの: ssm の残り (397 wy5 / 402 / 407 が詰めた)、experts の TC (250/416/411)、
  `full` の flashdec 分割 (P196/P201 の玉。カーネルが要る)、mix の GEMM
  (sn3 の Q8_0 TC は既定 on で、1 呼び出し 1.3 ms = 10 TFLOPS が実力)、
  埋め込み (task 415 が既定 on にした)、expert の先読み (337/394 が否定)、
  VRAM 上限 (規則で禁止)。**`.cu` / `.ptx` は触っていない。**

## 2. 変えたこと

### 2.1 `DENNO_PF_PLE_THREADS` の既定 96 -> 128

`exp_pf::DEFAULT_PLE_THREADS` を 128 にした (丸めの上限も 128 のまま)。切り戻しは
`DENNO_PF_PLE_THREADS=96`。読む行・綴じ方は変えないので値はビット一致 (§0-5)。

### 2.2 `DENNO_PF_EXP_IQ_MIN_ROWS` (新規、**既定 0**)

`exp_pf`: `IQ_MIN_ROWS_ENV` / `DEFAULT_IQ_MIN_ROWS` (0) / `iq_min_rows_from_value` /
`iq_min_rows`。`qwen4exp::moe_experts_iq_pf_batch` のストリーム分割で、IQ モードの
ときは IQ の床を読む (K 系は従来どおり `k_enabled() && min_rows > 0`)。床 > 0 の
ときだけ分割が立つ。床未満の expert は CPU の flat 経路 (`moe_experts_flat`) で
計算し、GPU の和の後に足す (K 系の既存の分割と同じ順序)。**既定 0 は従来と同じ経路。**

### 2.3 診断の計器 (値には触れない)

- `q4ledger`: 副区間 6 本 (`pf_setup` / `pf_teardown` / `attn_pre` / `mix_keep` /
  `comb_at` / `iter`) と `iter_begin` / `iter_end` / `named_wall_ns`、
  `[q4ledger-pf] iter n=.. total=.. uncovered=..` 行。
- 層ループの入口・出口・attn 入口に副区間のガード。off (PROFILE 無し) は `Instant` 1 本も取らない。

## 3. 数値 (lab、同じ run の中の対)

### 3.1 t419b (`676b0a94e`、5 COND、load 1.77)

| COND | env | 3,826 cold s (r1 / r2) | 14k cold s | 3,826 ple ms | 14k ple ms | decode |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| x0 (捨て) | — | 8.28 / 8.37 | 7.51 | 215.9 / 213.5 | 417.5 | 12.8 |
| b1 | — (96 本) | 8.32 / 8.35 | 7.04 | 213.0 / 223.0 | 415.3 | 12.9 |
| t192 | `PLE_THREADS=192` | 8.33 / 8.31 | **6.99** | **196.6 / 195.0** | **381.4** | 12.8 |
| t256 | `PLE_THREADS=256` | 8.33 / 8.31 | **6.99** | **198.2 / 198.7** | **382.9** | 12.8 |
| b2 | — (96 本) | 8.36 / 8.32 | 7.01 | 216.1 / 214.8 | 414.2 | 12.8 |

- **丸めの上限が 128 なので t192 と t256 は両方 128 本** (2 サンプルずつ取ったのと同じ)。
- **ple 段**: 14k 414.75 -> 382.15 = **-7.9%**、3,826 215.45 -> 197.4 = **-8.4%**
  (それぞれ b1/b2 の 4 サンプルと t192/t256 の 4 サンプルの中央値)。
  2 COND x 2 ラウンドで同じ向き・同じ大きさ。
- **壁**: 14k 7.02 (b1/b2 の平均) -> 6.99 = **-0.4%**。3,826 は 8.32 -> 8.32 = 0。
- decode は 12.8-12.9 ms/tok で不変、md5 は全 COND SAME、errors 0。
- **なぜ ple 段 -8% が壁 -0.4% か**: gather 416 ms のうち減ったのは 34 ms で、
  壁の残りは expert の読み (2.6 s) と GPU (4.4 s)。段の規則で採る。

### 3.2 t419c (`78b73f535`、6 COND、load 2.21) — IQ 系 expert の行数の床

| COND | env | 3,826 cold s | 14k cold s | decode |
| --- | --- | ---: | ---: | ---: |
| x0 (捨て) | — | 8.38 | 7.26 | 12.8 |
| b1 | — | 8.32 | 7.05 | 12.8 |
| iq16 | `IQ_MIN_ROWS=16` | **12.43** | **13.38** | 12.8 |
| iq32 | `IQ_MIN_ROWS=32` | 8.32 | **7.02** | 12.8 |
| iq64 | `IQ_MIN_ROWS=64` | 8.34 | **7.02** | 12.8 |
| b2 | — | 11.05 (外れ) | 7.01 | 12.9 |

- **床 16 は大幅に悪化** (3,826 +49%、14k +90%)。CPU の flat 経路に回すと、
  回した expert の数だけ損をする (1 本 2 MB の読みより高い)。**この枝は閉じた。**
- 床 32 / 64 は baseline と同じ (14k 7.02 対 b1 7.05 / b2 7.01)。
- **md5 は iq16 でも SAME** = CPU 側の寄与を足す経路は値を持ち込まない
  (丸め順は変わりうるが、この範囲では出力トークンは動かない)。
- b2 の 3,826 8.32 -> 11.05 は位置/外乱の外れ (14k は 7.01 で正常、iq32/64 も 8.32)。
  同じ run の b1 と iq32/iq64 を対にして読んだ。
- 採否: **`ADOPT: no`。既定 0 のまま** (ゲートは残す。切り分け用)。

## 4. 数値の扱いと見積り

- **`DENNO_START_DROP_CACHE=1` を全 COND に付けた** (task 415 の t415c と同じ手)。
  付けないと 14k は COND の位置で 13% 動く。付けると 3,826 は 8.3 s に揃うが、
  絶対値は「毎回ディスクから読み直す」ぶん大きい。**採否は同じ run の対で読む。**
- **見積り (推定)**: PLE の本数を 128 にすると 14k で -34 ms、3,826 で -18 ms。
  壁への寄与は 14k -0.4%、3,826 0。**task の目標 (2.5%) には届かない。**
- **分かったこと**: 既定モデルの prefill は **GPU 58% / ディスク 36% / gather 6%**
  (14k、暖) で、ホスト側の残差はもう 1 s を切っている。2.5% を取るには
  `experts` / `ssm` / `mix` / `full` のカーネルか、expert のバイト数を減らすしかない
  (バイトは VRAM 上限の規則で閉じている)。

## 5. テスト

int-auto `a8a92bb33` (merge: br422) を merge した後の最終 HEAD で:

```
CARGO_TARGET_DIR=/tmp/t419 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1306 passed / 0 failed / 103 ignored
CARGO_TARGET_DIR=/tmp/t419 cargo test --release -p denno-engine --features cuda --test kernel_invariants
  -> 124 passed / 0 failed  (task 415 の 1 本 + task 419 の 1 本 + br422 の 1 本を含む)
CARGO_TARGET_DIR=/tmp/t419-nocuda cargo build --workspace --release            (cuda feature 無し)
  -> ok
CARGO_TARGET_DIR=/tmp/t419-nocuda CUDA_VISIBLE_DEVICES=-1 cargo test --workspace --release -- --test-threads=1
  -> (結果は §5.1)
```

追加した 3 本:
- `iter_uncovered_is_named_wall_complement` (lib) — 回転の uncovered が
  「経過 − 名前付き区間の増分」であること、`[q4ledger-pf] iter` 行が出ること、
  reset で 0 に戻ること。
- `iq_min_rows_truth_table` (lib) — 新しい床の既定 0 と解釈 (未設定/解析不能/0 は
  床なし、空白は落とす)。床の判定は K 系と同じ関数を使う。
- `task419_iq_min_rows_is_wired_into_the_stream_split` (kernel_invariants) —
  env 名 / 既定 0 / 読み口 / `moe_experts_iq_pf_batch` が IQ モードで IQ の床を読み、
  K 系の床と `cpu_be` / `moe_experts_flat` を残していること。
- 既存の `ple_threads_rounds_and_defaults` と kernel_invariants の既定値検査を
  128 に更新した。

### 5.1 cuda 無しの workspace テスト

```
CARGO_TARGET_DIR=/tmp/t419-nocuda CUDA_VISIBLE_DEVICES=-1 \
  cargo test --workspace --release -- --test-threads=1
  -> 63 スイート / 1,479 passed / 0 failed (exit 0)
```
既知の赤 (`qwen4exp_decode spec_batch_stops_at_eos`、`qwen4exp_params` の validate 2 件)
はこの木では出ていない (0 failed)。

## 6. merge / 終端

終端で int-auto **`a8a92bb33`** (merge: br422 = GDN qkv/out と mix down の dn6) を
merge した。**衝突は無かった** (br422 は `cuda.rs` に新しいカーネルとゲートを足し、
`kernel_invariants.rs` の末尾にテストを 1 本足していたが、私の追記とは別の場所だった)。
`build_ptx.sh` の `targets_all` は私は触っていないので衝突しない。
merge 後の HEAD で上の 4 つの関門を回し直した (§5)。

## 7. GPU で確認してほしいこと (監督へ)

1. 既定は `DENNO_PF_PLE_THREADS=128`。切り戻しは `=96`。
2. 本番で見てほしいのは **14k 級の長い prefill の ple 段** (`[pre-kprof]` の ple)。
   暖かいと 415 -> 381 ms。3,826 でも 215 -> 197 ms。
3. `DENNO_PF_EXP_IQ_MIN_ROWS` は既定 0 のまま (採らなかった)。触らないこと。

## 8. text (sane の証拠)

t419b の b1、c8000_cold (14k cold の応答、`TXTD`):

> ご提示いただいた文書は、LLM推論エンジン（`denno-engine）における **Prefill（事前計算）処理の性能最適化の記録です。\n\n以下に、この文書の技術的詳細を、**「問題の特定」、「解決プロセス」、「主要な発見と教

`x0` / `b1` / `t192` / `t256` / `b2` の 5 COND とも同じ文で、md5 も全 COND 一致
(`672fec07` 系)。「!!!!」や空出力は無い。t419c の 6 COND も同じ (§9)。

## 9. VERDICT 行

t419b (`676b0a94e`、5 COND、ptx=none、`t192`/`t256` は丸めで 128 本):
```
VERDICT b1 vs x0 OK decode +0.3% prefill +0.0% md5 SAME errors 0
VERDICT t192 vs x0 OK decode +0.0% prefill +1.8% md5 SAME errors 0
VERDICT t256 vs x0 OK decode +0.0% prefill +0.9% md5 SAME errors 0
VERDICT b2 vs x0 OK decode +0.0% prefill +0.0% md5 SAME errors 0
VERDICT_DEPTH b1 c2000_cold vs x0 decode +0.3% prefill +0.1%
VERDICT_DEPTH b1 c8000_cold vs x0 decode -6.8% prefill -6.3%
VERDICT_DEPTH t192 c2000_cold vs x0 decode +0.0% prefill -0.1%
VERDICT_DEPTH t192 c8000_cold vs x0 decode -8.0% prefill -6.9%
VERDICT_DEPTH t256 c2000_cold vs x0 decode +0.0% prefill -0.1%
VERDICT_DEPTH t256 c8000_cold vs x0 decode -7.4% prefill -6.9%
VERDICT_DEPTH b2 c2000_cold vs x0 decode +0.0% prefill +0.2%
VERDICT_DEPTH b2 c8000_cold vs x0 decode -7.4% prefill -6.7%
VERDICT baseline x0 errors 0 text ok
```
(`vs x0` の 14k 行は使わない: x0 は捨ての 1 本目で、b1 以降が **baseline なのに
-6.3%** と出ている = 位置効果。§3.1 の同じ run の対で読む。)

t419c (`78b73f535`、6 COND、`iq16`/`iq32`/`iq64` = `DENNO_PF_EXP_IQ_MIN_ROWS`):
```
VERDICT b1 vs x0 OK decode +0.0% prefill -1.7% md5 SAME errors 0
VERDICT iq16 vs x0 OK decode +0.0% prefill -1.7% md5 SAME errors 0
VERDICT iq32 vs x0 OK decode +0.0% prefill -1.7% md5 SAME errors 0
VERDICT iq64 vs x0 OK decode +0.0% prefill -2.6% md5 SAME errors 0
VERDICT b2 vs x0 OK decode +0.3% prefill -1.7% md5 SAME errors 0
VERDICT baseline x0 errors 0 text ok
```
(iq16 の悪化は `vs x0` の行ではなく §3.2 の同じ run の絶対値で読む: 3,826 12.43 /
14k 13.38 s 対 b1 8.32 / 7.05 s。)
