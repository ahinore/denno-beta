ADOPT: yes t361b

# REPORT P361 — IQ3_S decode: ミスした expert の CPU 寄与 (平坦化の行幅 64 -> 16)

branch: **br348** のまま (task 361 の指示 2「Stay on branch br348」に従った。br361 は切って
いない)。int-auto は `17e938604` (merge: br347) のまま動かず、merge は不要だった。
task 348 の未 commit 差分 (moe_expert.rs / qwen4exp.rs) は
`/work/backups/stop20261116-denno-beta.patch` とバイト一致で残っていたので、そこから続けた。

## 0. 結論 (3 行)

1. **採ったのは 1 本**: ミス経路の平坦化 (`expert_contribs_flat_decode_into`) の
   行チャンクを **64 -> 16** にした (`DENNO_EXP_MISS_ROWS`、既定 16、`=64` で切り戻し)。
   lab t361b の `[exp-iq-hit]` = ミス CPU 時間 (ms/tok) で、窓ごとの比の中央値が
   **r1 -5.8% / r2 -2.4%** (ミス 1 本以下の窓で -2〜-12%、ミス 3 本の窓では 0%)。
   decode 中央値は 16.6 (b0) / 16.7 (b1、b0 と同 env) / 16.7 (r16) で**分解能以下**、
   prefill +0.9%、md5 は 5 COND 全 SAME、errors 0。値は行の切り方だけで**ビット一致**。
   task の目標 (decode 2%) には届いていない。オペレーター規則 (段で 2% 以上を
   2 ラウンド) で採った。task 342 の hc32 と同じ立場。
2. **task 348 が書いていた 2 本はどちらも不採用**:
   (a) `DENNO_EXP_MISS_ROWS=auto` (位相の全行数 / スレッド数 = 92/183/256 行) は
   **ミス段で +4.0% (r1) / +7.5% (r2)** と悪化 (大きい塊ほど末尾の 1 タスク待ちが効く)。
   経路ごと撤去した。(b) IQ3_S m=1 の **AVX-512F 版**は実形 2560x640 で AVX2 版の
   **0.56x** (1T 4.59 -> 2.56 GB/s、8T 31.2 -> 18.4 GB/s、min of 30) で、こちらも撤去。
   IQ3_S の dequant は 32 要素あたり 8 本の格子 (`IQ3S_GRID`) の**スカラーロード**が
   本体で、FMA を半分にしても縮まない (IQ4_NL が 1.17x になったのは格子が 16 エントリで
   zmm の VPERMD に載ったため)。
3. **新しい事実 (次のタスクの材料)**: ミスの CPU 計算は**GPU の裏に隠れていない**。
   `DENNO_EXP_MISS_FLAT=0` (ミス 1 本 = 1 タスク = 1 スレッド) にすると decode が
   **16.6 -> 20.5 ms/tok (+23%)**、ミス CPU が 2.82 -> 6.43 ms/tok (+128%) になる。
   つまり (i) ミスは critical path に居る、(ii) 1 -> 14 スレッドで **2.3x** まで縮む、
   (iii) 14 スレッド内の**行の割り方**はほぼ効かない (そこはもう帯域/レイテンシの壁)。
   ミス段の総量 (2.8 ms/tok) を削るには、行幅ではなく **読むバイト数**か
   **GPU との重なり**を動かす必要がある。

## 1. 前提 (task 348 の残骸と、測り直した base)

- int-auto `17e938604`、IQ3_S (swift15-abl)、baseline COND = 既定 + `DENNO_EXPERT_IQ_GPU=1
  DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0 DENNO_GPU_WEIGHT_MB=12000`。
- 未 commit 差分の中身は 2 つだった: ① `DENNO_EXP_MISS_ROWS` (auto) ② `DENNO_IQ3S_AVX512`
  (zmm カーネル + 一致テスト + ベンチ)。どちらも測定前で、REPORT も無かった。
- base の decode は 16.6-16.7 ms/tok (P342/P347 の 16.6-16.8 と同じ帯)。
- ミス経路の現状の内訳 (`[exp-iq-hit]`、lab t361b b0、定常窓 a2 = レイヤ 6117 /
  ミス 5338 = 0.87 本/層): ミス CPU **2.82-2.86 ms/tok**、うち 1 本あたり 68-70 us。
  `[q4ledger] experts` の wall は同じ窓で **5.48 ms/tok** で、`[exp-iq-kprof]` の
  バイト数 / GB/s から出す GPU 側 (約 2.4 ms/tok = 114 GB/128 tok を 300-460 GB/s) との
  和にほぼ等しい = **直列**。

## 2. 変えたこと

### 2.1 `crates/denno-engine/src/model/qwen4exp.rs`

- `MISS_FLAT_ROWS = 16` と `DENNO_EXP_MISS_ROWS` (未設定 16 / 数字で上書き / 0 と
  壊れた値は既定)。ミス経路 (`expert_miss_contribs_into` → 平坦化) だけに掛ける。
  **mmap の k=10 経路 (flat_new) は 64 のまま** (`DECODE_FLAT_ROWS`)。10 本の
  ミスでは 64 が最良で、16 にするとタスクが 2,400 本になって固定費が勝つ
  (開発コンテナの実測 390 -> 441 us)。
- `exp_miss_rows_report()`: 起動行用の 1 行 (下記)。
- task 348 の `auto` 経路 (`flat_auto_rows` / `rows == 0` = 自動) は**撤去**。
  `expert_contribs_flat_decode_into` の契約は元どおり (0 行はエラー) に戻した。
- マイクロベンチ `bench_miss_flat_rows` を追加 (開発コンテナ用、`#[ignore]`)。
  Q4_K/Q5_1 の実形 2560x640 の疎 GGUF でミス 1/2/3/10 本 × 行幅 8..256 を交互に測る。
  値のビット一致を測る前に全組み合わせで確かめる。

### 2.2 `crates/denno-engine/src/model/moe_expert.rs`

- task 348 の AVX-512F 版カーネル (`gemv_iq3s_avx512` / `iq3s_w16_zmm` / `hsum512`) と
  ゲート・テスト・ベンチを**削除**。IQ3_S の dispatch に「なぜ書いたが捨てたか」
  (測定値と、格子がスカラーである理由) を残した。既定の挙動は変わらない。

### 2.3 `crates/denno-cli/src/serve_flash.rs`

- 起動時に 1 行 `[flash] exp miss rows: <16 (既定) | N (DENNO_EXP_MISS_ROWS)>`。
  新しい env 名がラボの COND から届いたかをログで確かめるため (t294b で 2 本
  届かなかった前例がある)。t361a/t361b のログで実際に届いていることを確認済み。

## 3. 数値の扱い

- 行幅は**行の切り方だけ**を変える。1 行の計算は `gemv_rows_seq` (行範囲入口) で、
  行は独立・累積順は行内で閉じるので、どの行幅でも寄与は**ビット一致**する
  (テスト `decode_flat_row_chunk_is_independent` が 7/32/64/96/1000 行で確認、
  ベンチは 8/16/32/46/64/92/128/256 行で確認)。lab の md5 も 5 COND 全 SAME。
- モデルの挙動 (選ばれる expert、丸め順) は変えていない。既定を変えたのは
  「タスクの切り方」1 つだけ。

## 4. Lab

`SCENARIO=depth8k PROFILE=1`、model `swift15-abl-IQ3_S`、PTX=none。各 COND 2 ラウンド
(交互)。比較は同じ run の baseline COND だけ。

### 4.1 t361a (`c397a153b`、3 COND、load_before 2.69)

COND = base0 (捨て) / mrauto (`DENNO_EXP_MISS_ROWS=auto`) / base1 (繰り返し)。

| cond | decode ms/tok | prefill s | errors | text | md5 |
| --- | --- | --- | --- | --- | --- |
| base0 | 16.7 | 0.58 | 0 | looks ok | 全 SAME |
| mrauto | 16.6 | 0.59 | 0 | looks ok | 全 SAME |
| base1 | 16.6 | 0.59 | 0 | looks ok | 全 SAME |

```text
VERDICT mrauto vs base0 OK decode -0.2% prefill +0.9% md5 SAME errors 0
VERDICT base1 vs base0 OK decode -0.5% prefill +0.9% md5 SAME errors 0
```

この run の `[exp-iq-hit]` ミス CPU (定常窓 a2、ms/tok): base0 2.898 / 2.844、
mrauto 2.963 / 2.934、base1 2.833 / 2.843。**auto は +3〜4% 悪い** (同じ env の
base0/base1 の差は 0.2%)。decode の中央値は動かない (16.7 / 16.6 / 16.6 で、
同じ env の base0 と base1 が 0.1 ms ずれる = 分解能 0.6%)。

### 4.2 t361b (`c397a153b`、5 COND、load_before 1.67)

COND = b0 (捨て) / mr (`=auto`) / r16 (`=16`) / slow (`DENNO_EXP_MISS_FLAT=0`) /
b1 (繰り返し)。全 COND に `DENNO_Q4_PROFILE=1`。

| cond | decode ms/tok | prefill s | errors | text | md5 |
| --- | --- | --- | --- | --- | --- |
| b0 | 16.6 | 0.58 | 0 | looks ok | 全 SAME |
| mr | 16.7 | 0.57 | 0 | looks ok | 全 SAME |
| r16 | 16.7 | 0.59 | 0 | looks ok | 全 SAME |
| slow | **20.5** | 0.58 | 0 | looks ok | 全 SAME |
| b1 | 16.7 | 0.58 | 0 | looks ok | 全 SAME |

```text
VERDICT mr vs b0 OK decode +0.5% prefill -1.7% md5 SAME errors 0
VERDICT r16 vs b0 OK decode +0.2% prefill +0.9% md5 SAME errors 0
VERDICT slow vs b0 OK decode +23.0% prefill -0.9% md5 SAME errors 0
VERDICT b1 vs b0 OK decode +0.2% prefill +0.0% md5 SAME errors 0
```

長文脈 (prefill s / decode ms/tok、中央値): b0 2.53/15.8・0.39/14.8・6.64/16.4・0.46/15.3、
r16 2.52/15.8・0.39/14.9・6.43/16.2・0.45/15.2、b1 2.52/15.9・0.39/14.8・6.43/16.2・
0.46/15.2、mr 2.52/15.9・0.39/14.7・6.45/17.7・0.45/16.4、slow 2.53/20.0・0.39/17.9・
6.42/18.6・0.46/17.1。

### 4.3 ミス段 (`[exp-iq-hit]` の ミスCPUms/tok、窓ごと)

窓 = 1 リクエストの decode。`層/ミス本` は b0_1 の値。r2 に無い窓は、その COND の
r2 が短い並びで切れたため (RESULT の Raw に無い)。値は ms/tok。

| 窓 (層 / ミス / 本/層) | b0 r1 / r2 | **r16 r1 / r2** | mr r1 / r2 | slow r1 / r2 |
| --- | --- | --- | --- | --- |
| a1 4119 / 12860 / 3.1 | 5.787 / 5.850 | 5.885 / 5.851 | 6.209 / 6.307 | 10.055 / 10.209 |
| a2 6117 / 5338 / 0.87 | 2.820 / 2.864 | **2.810 / 2.784** | 2.934 / 2.976 | 6.428 / 6.687 |
| b 2293 / 2999 / 1.3 | 1.637 / 1.622 | 1.552 / 1.592 | 1.703 / 1.717 | 3.735 / 3.646 |
| c2000_cold 4029 / 2776 / 0.7 | 1.585 / 1.591 | 1.528 / 1.485 | 1.599 / 1.624 | 4.411 / 4.425 |
| c2000_turn2 6135 / 2887 / 0.47 | 1.694 / 1.714 | 1.585 / 1.634 | 1.672 / 1.654 | 4.714 / 4.744 |
| c8000_cold 4412 / 2201 / 0.5 | 1.359 / 1.235 | 1.216 / - | 1.309 / - | 2.975 / - |
| c8000_turn2 6142 / 1735 / 0.28 | 1.192 / 1.107 | 1.048 / - | 1.144 / - | 3.069 / - |
| 6144 層 / 595 / 0.1 | 0.453 / 0.418 | 0.411 / 0.410 | 0.462 / 0.438 | 1.051 / 1.089 |

窓ごとの比 (同じラウンドの b0 対比) の中央値: **r16 -5.8% (r1) / -2.4% (r2)**、
mr +1.5% (r1) / +4.4% (r2)。r16 は 14 比較中 12 で良く、同値 1 (a1 r2)、負けは
a1 の +1.7% (r1) だけ (a1 はミス 3.1 本/層の唯一の窓 = 端数が元から無い)。
同じ env の b0 と b1 の差は最大 1.7% (w2 r2) なので、r16 の -2.4% はその外。
**ミスが 1 本以下の窓で -2〜-12%、ミス 3 本の窓では 0%** という形で、行幅の効き方
(64 行 = 20 タスク / 14 スレッド = 2 巡で 6 スレッド遊ぶ、が 16 行 = 80 タスクで
端数が消える) と一致する。

### 4.4 `slow` が示したこと (ミスは隠れていない)

`DENNO_EXP_MISS_FLAT=0` はミスを「1 expert = 1 タスク」に戻す (ミス 1 本の層では
**1 スレッド**が 3 行列を順に回す)。結果:

- decode 16.6 -> **20.5 ms/tok (+23.0%)**、ミス CPU 2.82 -> **6.43 ms/tok (+128%)**。
- つまり 1 スレッド -> 14 スレッドで **2.3x**。14 倍にはならない (帯域/レイテンシの壁)。
- そして**ミスを 2.3 倍遅くすると decode がその分だけ遅くなる** = ミスは GPU の裏に
  隠れていない。`[q4ledger] experts` の wall 5.48 が GPU 約 2.4 + ミス 2.86 の和に
  なることとも一致する (overlap 経路は既定 on だが、実測では重なっていない。
  アップロードの DMA が同じホストメモリを読んでいるためと推定)。

## 5. テスト

```
cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1231 passed; 0 failed; 103 ignored
cargo test --release -p denno-engine --features cuda --test kernel_invariants
  -> 102 passed; 0 failed
```

`CARGO_TARGET_DIR=/tmp/t348`。追加したテスト: `exp_miss_rows_value` (env の解釈・
既定 16・0 と壊れた値は既定へ・起動行の文字列)。更新したテスト:
`decode_flat_row_chunk_is_independent` (auto を消したので「0 行はエラー」に戻した)。

開発コンテナのマイクロベンチ (`bench_miss_flat_rows`、Q4_K/Q5_1 の実形、min of 20):
14T で k=1 は 16 行 52.9 us / 64 行 53.3 us (ほぼ同じ)、k=2 は 95.6 / 90.0、
k=3 は 136.6 / 124.7、k=10 は 440.8 / 390.0 と**16 行が k>=2 で悪い**。実機の
ミス分布 (0.87 本/層、k=1 が最多) では効くが、この proxy では再現しない。
採否は lab の実測 (4.3) で決めた。

## 6. GPU で確認してほしいこと

1. 本番 8098 の起動行に `[flash] exp miss rows: 16 (既定)` が出ること。
   出ていれば既定が効いている (以前のバイナリは 64)。
2. 実運用のテキストが変わらないこと (値はビット一致なので md5 は不変)。
3. もし decode が 1% でも悪く見える環境があれば `DENNO_EXP_MISS_ROWS=64` で
   切り戻して A/B してほしい (切り戻しは env 1 つ)。

## 7. 次の手 (やっていないこと)

- ミス段 2.8 ms/tok の削り代は「行の割り方」ではなく**バイト数**と**重なり**:
  (a) `DENNO_EXPERT_GPU_MIN_HIT=0` のアップロード (21 本/token = 41 MB/token の
  H2D) と同じホストメモリを読んでいるので、アップロードを止めた A/B で
  どこまで縮むかを見る価値がある (task 337 は「予算 16 が最良」としているが、
  それは decode 全体の話で、ミス段の単価ではない)。
  (b) `sum_mode` (`DENNO_DEC_EXP_SUMDEV`) の寄与和は今回の run では使われていない
  (`batch_ok=0`)。
- IQ3_S の 14 スレッド時の壁 (2.3x で頭打ち) は、格子 (`IQ3S_GRID`) のスカラー
  ロード 8 本 / 32 要素。AVX-512 の gather で 16 レーンずつ引く案は未試行
  (Zen 5 の gather は 4-5 cycle/命令。今回は時間切れ)。
- `DENNO_EXP_MISS_ROWS` は 8 行も試せる (今回の sweep は lab の 1 ラウンドを
  使う必要があるので、8 と 32 の比較は次のタスク)。
