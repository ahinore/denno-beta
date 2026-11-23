ADOPT: yes t465b

# task 465 (dev2) — 既定モデル prefill: full attention の K/V を bf16 で読む
# (14k cold の `full` 段 GPU -13.4% / 同段 wall -7.3〜-7.6%、全体は -1.6% / ±0%)

branch `br465` = 開始時の `int-auto` **`70982c7dc`** (= merge: br461。自分の br461 は
既に merge 済みだったので取り込み不要)。数値はすべて GPU ラボの実測
(`SCENARIO=depth8k`、`PROFILE=1`、`model=flash-next-UD-IQ3_XXS`)。
lab 名と測った SHA: **t465a = `303be51eb`** (4 COND、ptx=attn_pf4)、
**t465b = `303be51eb`** (同じ要求の 2 回目)、**t465c = `28eace2c4`** (PTX を
commit し、既定 on のまま PTX= 無しで回した確認)。最後に `int-auto`
**`2615ddccd`** (merge: br468) を merge した (§6)。

## 0. 結論 (7 行)

1. **採る。** prefill の full attention が読む K/V を f32 から **bf16** にした
   (`DENNO_PF_ATTN_KVBF16`、既定 on)。v3 は元から smem へ入れるときに
   `__floats2bfloat162_rn` で bf16 に丸めているので、**値は 1 ビットも変わらない**。
2. **14k cold の `full` 段 GPU が 638.3 / 639.5 -> 552.7 / 552.7 ms
   (-13.4% / -13.6%、2 ラウンド)**。同段の wall は 826.7 / 833.8 -> 766.4 / 770.3 ms
   (-7.3% / -7.6%)。
3. **全体の壁時計は 14k で -1.6% (t465a の base 対) / -3.1% (t465c の base 対) /
   -1.7% (t465c の base2 対)**。**task の target (2.5%) は満たしていない**
   (t465c の base 対だけが届くが、base2 対は -1.7%)。`full` は 14k の壁の 16%
   しか無く、残りは experts (転送律速・床) と ssm (GEMM + スキャン) で埋まっている。
4. 採否は**オペレーター規則 (段で 2 ラウンド ≥2%)** で決めた: `full` 段は
   2 ラウンドとも -7% 超で、他の段の GPU 時間は不変、md5 は全 COND・全
   コンテキスト SAME、decode 不変、text sane。
5. **機序**: v3 は 1 クエリ行あたりプレフィックス全体の K/V を f32 で読む
   (14k・12 層で約 900 GB = 540 ms で 1.67 TB/s、HBM 帯域 1.79 TB/s に張り付き)。
   bf16 にすると読みが半分 (約 450 GB) になり、変換命令も消える。
6. **sm_120 の動的 smem 上限は 99 KB** なので、クエリタイルを 64 行から広げる案
   (K/V 読みを 2 倍の行で割る) は**取れない** (q_sm だけで 33.8 KB、128 行だと
   67.6 KB で p/s/stg が入らない)。だから読む側の型を変えた。
7. 切り戻しは `DENNO_PF_ATTN_KVBF16=0` (v3 と同じ値)。PTX 未生成 / 入口なし /
   smem の opt-in 失敗 / 行の順が想定外、のどれかで v3 へ fail closed。

## 1. Do 1 / Do 2: 現在の int-auto の段プロファイルと、選んだ段

### 1.1 段プロファイル (t465a の base_1、既定 env、14k)

| 段 (14k) | wall ms | GPU busy ms | 備考 |
| --- | ---: | ---: | --- |
| ssm | 2,096.4 | 1,318.9 | scan 651 + GEMM 610 + ba 60 (t447d の kprof) |
| experts | 1,836.9 | 1,421.7 | 転送 1.39 s (28.0 GB) が床、GPU は 74 TFLOPS |
| full | 826.7 | 638.3 | **attention 本体 約 540 ms** (qkv/out の GEMM 約 100 ms) |
| other | 317.7 | 185.8 | res_up 89 + pf_teardown 88 + embed 27 |
| mix | 21.9 | 659.3 | 完全に隠れている |

3,826 は experts 1,276 (うち転送待ち 1,039 = 22.4 GB / 20.6 GB/s の床)、ssm 382、
mix 266、full 110。

### 1.2 なぜ full attention を取ったか (Do 2)

- 大きい段のうち **experts は転送律速で床** (task 296/320/324/337/428/435/449/452)、
  **ssm は scan が 3 タスク (397/402/407/441/447) で掘られている**、mix も
  451/457/438 で一巡している。
- **full attention (prefill) は task 230/244 (v3 カーネル) 以降だれも触っていない**
  うえ、14k の壁の 16% を占める。しかも実測が **1.67 TB/s = HBM 帯域の 93%**
  で、読みを減らせば効く形だった。
- 選んだ理由を 1 行で: **触られていない最大の段で、律速が「読むバイト数」だと
  実測で分かっていて、しかも丸めを変えずにバイト数を半分にできるから。**

## 2. 変えたこと

### 2.1 `kernels_attn_pf4.cu` (新規、2 カーネル)

- `dno_attention_prefill4`: **v3 (`dno_attention_prefill3`) と同一**のスコア mma /
  マスク / exp2 / PV mma / 書き出し / 1/l で、**K/V の引数だけ
  `const __nv_bfloat16 *`**。タイル (BQ=64 / BK=32 / QD=264 / PD=40 / 256 スレッド /
  smem 89,600 B) は v3 と同じ。
- `dno_kv_f32_to_bf16`: f32 の K/V の行 `[r0, r0+n)` を bf16 へ落とす。
  丸めは v3 が smem へ書いていたのと同じ `__floats2bfloat162_rn`。

### 2.2 ホスト側 (`cuda.rs`)

- f32 ミラー (既存) から新チャンクの行だけを bf16 の作業域 (層ごとに 1 本を使い回す、
  14k で 2 x 14.3 MB) へ落としてから v4 を撃つ。層優先 prefill は 1 層のチャンクを
  続けて回すので、`pos0` が前回の続きなら差分だけ。`pos0 == 0` は層の先頭。
  想定外の順 (逐次 prefill・確保し直し) は 0 から落とし直す (読み手は常に
  `[0, pos0+nt)` しか見ないので、常に正しい)。
- 変換は 1 層 1 チャンクあたり 6 us 級 (14k で 2 M 要素 x 12 MB)。84 回で 1 ms 未満。
- ゲート `DENNO_PF_ATTN_KVBF16` (**既定 on**)。`gate_audit` の既定 on 台帳に 1 行足した。
- 撃てないときは v3 -> v2 -> 従来へ fail closed (値はどれも同じ丸め順)。

## 3. 数値 (lab、同じ run の中の対)

### 3.1 `full` 段 (14k cold、2 ラウンド)

| run | base wall / GPU | dev wall / GPU | base2 wall / GPU |
| --- | ---: | ---: | ---: |
| t465a | 826.7 / 638.3 | **766.4 / 552.7** | 827.8 / 638.5 |
| t465b | 833.8 / 639.5 | **770.3 / 552.7** | 828.8 / 637.3 |
| t465c | 828.5 / 637.8 | **766.9 / 552.7** | 828.6 / 638.5 |

GPU は **-13.3 / -13.4 / -13.6%**、wall は **-7.4 / -7.3 / -7.6%** (各 run の
base / base2 対)。**3 ラウンドとも dev は 552.7 ms で同じ値**。

### 3.2 壁時計 (14k cold、1 ラウンド 1 サンプル)

| run | base | dev | base2 |
| --- | ---: | ---: | ---: |
| t465a | 5,099.5 | **5,016.4 (-1.6%)** | 6,202.2 (外れ) |
| t465b | 5,864.6 (外れ) | **5,124.3** | 5,102.4 (+0.4%) |
| t465c | 5,211.4 | **5,048.9 (-3.1%)** | 5,134.9 (-1.7%) |

**14k は 1 COND 1 サンプルで、同じバイナリの base/base2 が 5,099.5 /
6,202.2 (t465a) と 5,864.6 / 5,102.4 (t465b) と大きく振れる** (t465a の base2 は
`ple` 1,121.6 ms、t465b の base は experts の exposed +450 ms のホスト詰まり)。
採否は振れない `full` 段 (§3.1) で読んだ。

### 3.3 3,826 cold (`full` 段が小さいので差はほぼ無い)

| run | base | dev | base2 |
| --- | ---: | ---: | ---: |
| t465a r2 | 2,108.9 (full 107.9 / GPU 88.9) | 2,104.9 (full 108.0 / GPU 82.5) | 2,120.6 (full 107.8 / GPU 89.2) |
| t465b r2 | 2,110.3 (full 108.2 / GPU 89.0) | 2,114.6 (full 112.9 / GPU 82.6) | 2,118.4 (full 108.6 / GPU 89.2) |
| t465c r2 | 2,124.8 (full 109.6 / GPU 89.2) | 2,115.0 (full 109.5 / GPU 82.7) | 2,119.8 (full 110.0 / GPU 89.2) |

3,826 でも `full` 段の GPU は **-7%** (89.2 -> 82.7)。段が小さい (109 ms) ので
全体は **-0.5% (dev 2,115.0 対 base 2,124.8) / -0.2% (対 base2 2,119.8)** で
実質不変 (変換カーネルのぶん +10 ms はこの段に乗っている)。

### 3.4 decode と md5 / text

- decode は全 COND・両 run で **11.3 ms/tok** (t465a の x0 だけ 11.4)。
  8k 後の turn2 も 12.3-12.6 で差なし。
- md5 は **全 COND・全 6 コンテキストで SAME** (`a:edaa84ad` / `b:1938059f` /
  `c2000_cold:9b28e018` / `c2000_turn2:8b02d5c3` / `c8000_cold:672fec07` /
  `c8000_turn2:199f23b3`)。**丸めを変えないので DIFF が 1 つも無い**。
- text は全 COND `looks ok`。dev (t465c の c2000_cold) の書き出し:
  「prefillの遅延は計算量ではなく、線形層ごとのホスト往復と戻り値コピーが律速で
  あった。計測の誤認や撤回された最適化を経て、GPU常駐化やメモリ転送の最適化により、
  1.5Bモデルで約32倍の高速化を達成した。」(base と同じ文)

### 3.5 t465c (PTX を commit し、既定 on のまま PTX= 無しで回した確認)

`ptx=none` = ツリーに入れた `kernels_attn_pf4.ptx` が読まれる。結果は §3.1-3.4 の
とおり **3 ラウンドで同じ値** (`full` GPU 552.7 ms) で、commit した PTX と
既定 on の経路がそのまま効いている。`[pf-attn4]` の起動ログも dev COND にだけ出る。

## 4. 見積りと、やらないと決めたこと

- 見積り (推定、実測ではない): K/V の読みが 900 -> 450 GB になるので、HBM 律速なら
  540 ms が 270 ms 前後になる。**実測は -85.6 / -86.8 ms (-13.4% / -13.6%)** で、
  見積りより小さい。bf16 にしても **450 GB / 552.7 ms = 0.81 TB/s** で帯域の
  半分以下なので、残りは mma と softmax の計算・レイテンシ (25% 占有率) が
  効いている。**次の一手はカーネルの mma 効率**で、読みの削減ではもう出ない。
- クエリタイル 128 行 (K/V 読みがさらに半分) は **sm_120 の動的 smem 上限 99 KB で
  取れない** (q_sm 67.6 KB + k/v 33.8 KB + p/s/stg 26 KB = 145 KB)。この GPU では
  64 行 x 32 キーが上限に近い。
- decode 側 (flash-dec2、task 448) は触っていない。decode の K/V 読みは
  full 段 1.76 ms/tok で既に 828 GB/s なので、bf16 化の余地は別タスク。
- **初回だけ PTX の JIT が ~200 ms かかる** (t465a の kvbf16_1 の 3,826 が
  +182 ms)。2 回目以降 (同じプロセス / ドライバのディスクキャッシュ) は 0。
  task 461 の `kernels_pf_ple.ptx` と同じ性質で、新しい PTX を足すたびに起きる。

## 5. 数値の扱い

- 先頭 COND (`x0`) は page cache が冷えるので採否に使っていない (`vs x0` の
  VERDICT 行も使っていない)。t465b の x0_1 は 14k 7,397 ms / 3,826 2,116 ms で、
  14k は基準にならない。
- 14k は 1 COND 1 サンプル (§3.2 のとおり base/base2 が最大 21% 振れる)。
  **採否は 2 ラウンドとも再現した `full` 段の GPU / wall で読んだ。**
- 3,826 は 2 ラウンド x 2 サンプルで、round 2 の対 (dev 2,104.9 / 2,114.6 対
  base 2,108.9 / 2,110.3、base2 2,120.6 / 2,118.4) を読んだ。
- 変換カーネルの GPU 時間は `full` 段に乗る (+10 ms @3,826、実質 0 @14k)。

## 6. テスト (最終 HEAD = int-auto `2615ddccd` の merge 後)

```
CARGO_TARGET_DIR=/tmp/t465 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1339 passed / 0 failed / 103 ignored
CARGO_TARGET_DIR=/tmp/t465 cargo test --release -p denno-engine --features cuda --test kernel_invariants -- --test-threads=1
  -> 137 passed / 0 failed
CARGO_TARGET_DIR=/tmp/t465nc cargo build --workspace --release
  -> rc=0
CARGO_TARGET_DIR=/tmp/t465nc cargo test --workspace --release -- --test-threads=1
  -> 63 バイナリ / 1494 passed / 0 failed
```

merge の競合は 2 ファイルだけで、どちらも両側が `targets_all` の末尾に別名を
足したもの。**両側を残して統合**した (task 464 の `attn_dec4` + 自分の `attn_pf4`):

```
targets_all=(... mix_dn7 mix_up4 attn_dec2 pf_elem attn_dec4 pf_ple attn_pf4)
```

- `crates/denno-engine/kernels/build_ptx.sh` 28 行目を上記 1 行に。
- `crates/denno-engine/tests/kernel_invariants.rs` の末尾 assert 6 箇所
  (10766 / 12293 / 12410 / 12605 / 12699 / 12993 行) を同じ文字列に。
- int-auto 側の `targets.contains("attn_dec4")` と自分の
  `contains("attn_pf4")` / PTX ABI 検査はどちらもそのまま通る (137 passed)。
- `build_ptx.sh.rej` / `cuda.rs.rej` は task 457 時の古い失敗 hunk で、
  今回の merge とは無関係。未追跡のまま (commit しない)。

追加したテスト:
- `cuda::tests::pf_attn_kvbf16_gate_truth_table` — 既定 on、明示的な真値だけ on
  (切り戻しは `0` / `off` など)。
- `kernel_invariants::task465_pf_attn_v4_is_v3_with_bf16_kv` — v4 のタイルが v3 と
  同一であること、smem の見積り (89,600 B < 99 KB)、host のミラー定数、K/V の
  引数が bf16 で 8 バイトのコピーで読まれること (f32 経路が残っていないこと)、
  変換カーネルの丸め、v3 と同じ mma / exp2 / 行最大、build_ptx.sh の target と
  targets_all、**PTX の ABI (attention 12 引数 / 変換 7 引数)**。
- `targets_all` の末尾 assert 6 箇所を `... pf_elem pf_ple attn_pf4)` へ更新
  (int-auto merge 後は §6 の統合文字列)。
- `gate_audit::DEFAULT_ON_GATES` に 1 行 (PTX と 2 入口の実在検査が走る)。

## 7. VERDICT 行

t465b (`303be51eb`、4 COND、ptx=attn_pf4):
```
VERDICT base vs x0 OK decode +0.0% prefill -0.0% md5 SAME errors 0
VERDICT kvbf16 vs x0 OK decode +0.0% prefill -0.0% md5 SAME errors 0
VERDICT base2 vs x0 OK decode +0.3% prefill +0.9% md5 SAME errors 0
VERDICT_DEPTH base c8000_cold vs x0 decode -2.0% prefill -20.7%
VERDICT_DEPTH kvbf16 c8000_cold vs x0 decode -5.4% prefill -30.7%
VERDICT_DEPTH base2 c8000_cold vs x0 decode -4.8% prefill -31.1%
VERDICT baseline x0 errors 0 text ok
```
(`vs x0` は捨て COND 相手なので採否に使っていない。採否は §3.1 の同じ run の対。)

t465a (`303be51eb`、4 COND、ptx=attn_pf4): 全て OK / errors 0 / md5 SAME。
`full` 段 14k は base 826.7 / 638.3、**dev 766.4 / 552.7**、base2 827.8 / 638.5。

t465c (`28eace2c4`、4 COND、ptx=none = commit した PTX、dev は既定 on):
```
VERDICT base vs x0 OK decode -0.7% prefill +0.0% md5 SAME errors 0
VERDICT dev vs x0 OK decode +0.0% prefill +0.0% md5 SAME errors 0
VERDICT base2 vs x0 OK decode -0.7% prefill +0.9% md5 SAME errors 0
VERDICT baseline x0 errors 0 text ok
```
(base/dev/base2 は 14k 5,211.4 / **5,048.9** / 5,134.9 ms、`full` 段 GPU は
637.8 / **552.7** / 638.5 ms。)

## 8. GPU で確認してほしいこと (監督へ)

1. **既定 on**。切り戻しは `DENNO_PF_ATTN_KVBF16=0` (v3・同じ値)。
2. `kernels_attn_pf4.ptx` は t465a の lab が `bash build_ptx.sh attn_pf4` で
   生成したものをそのまま commit した (2,371 行、`.entry dno_attention_prefill4`
   (12 引数) / `dno_kv_f32_to_bf16` (7 引数))。
3. **全体の壁時計は 14k で -1.6% (t465a) / -3.1% (t465c の base 対) / -1.7%
   (同 base2 対)** で、target (2.5%) には安定しては届いていない。
   段の規則 (2 ラウンド -7%) で採った。14k の残りは experts (転送の床) と
   ssm (scan 651 + GEMM 610) で、どちらも既存タスクで掘られている。
4. 新しい PTX を足したので、**プロセスごとに最初の prefill だけ JIT が ~200 ms**
   かかる (2 回目以降は 0)。常駐サーバでは 1 回きり。
