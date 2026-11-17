ADOPT: yes t286c

# REPORT P286 — UD-Q4_K_XL decode: mix の GEMV を v5 形 (68 B の対読み) に

branch: br286 (int-auto `c5cb293` から。br282 は既に `c5cb293` で merge 済み
だったので取り込み不要)。lab: t286a (内訳) / t286b (A/B) / t286c (採否の既定) /
t286d (既定モデル IQ3_S)。既定を変えたゲート: `DENNO_MIX_QK` の既定
`(Blk, R8)` → `(V5, V5)` (task 282 の形は `=blk` で残す)。

## 0. 結論

- **採る**。UD-Q4_K_XL の warm 短 decode (128 tok) は**同じ run の
  baseline COND 比で -11〜-21%**。生の値 (128 tok の a1/a2、2 ラウンド計 4 本):
  t286b は baseline 23〜28 → v5 **20〜21**、t286c は baseline 23〜25 → **20〜21**。
  RESULT.md の decode 列 (48 tok の b リクエストも混ざる集計) では t286b
  `v5 21.4 vs base2 27.1 (-21%)`、t286c `on 22.6 vs base2 24.6 (-8.1%)`。
  どちらの run でも「同じ run の a2 同士」は **20 vs 23 = -13%**。
- 効いたのは 1 つだけ: **mix の down / up GEMV を既存の `dno_gemv_q8_0_v5`
  に替えた**こと。PTX は既にコミット済み (`kernels_moe_v5.ptx`、GDN の decode
  GEMV と同じカーネル) で、**新しいカーネルも .cu の変更も無い**。
- 内訳 (`[dec-kprof-avg]` の 1 カーネル平均、3.48 MB の行列):
  down `rowblk` 31.9 us → v5 **17.8 us**、up `r8` 39.7 us → v5 **11.8 us**。
  96 呼び出し/token なので GEMV だけで -4.2 ms/tok。
- prefill は 0.81〜0.85 s で baseline と同じ (この経路は prefill では撃たない)。
  errors 0。既定モデル (IQ3_S) は t286d で decode 18.8 ms/tok / md5 SAME /
  text ok / errors 0 が `=blk` と完全に一致 (5.)。

## 1. 前提と内訳 (lab t286a、commit `c5cb293` = 着手時の int-auto)

SCENARIO=depth8k、PROFILE=1、MODEL=flash-next-UD-Q4_K_XL。baseline COND =
タスク指定の 4 本 (`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000
DENNO_EXPERT_GPU_MIN_HIT=0 DENNO_GPU_WEIGHT_MB=12000`)。

`[q4prof] decode` (128 tok、`on` 相当) の 1 トークン内訳:

| 項目 | ms/tok | 中身 |
| --- | --- | --- |
| mix | **8.7** | 48 層 x 2 回 = 96 呼び出し、1 回 181 us/層 (= 2 呼び出し) |
| moe experts | 7.0 | seg 経路 (t278)。router 1.8 / shared 0.2 が別 |
| attn | 6.2 | GDN 36 層 + full 12 層 |
| ple | 0.03〜3.0 | ばらつく (ページキャッシュ) |
| head + norm + sample | 0.9 | |
| 合計 | 23.9 | [q4prof] の total と一致 |

`[dec-kprof-avg] mix band=0` (1 窓 = mix 1 回、3101 窓):

| op | us | 備考 |
| --- | --- | --- |
| H2D@a (xn 40 KB) | 6.8 | ホスト側の前処理が hin=26.2 us |
| K `dno_mix_gemv_q8_0_rowblk` (down 10240→320) | **31.8** | grid 320 x block 256 |
| K `dno_mix_lo_silu` | 4.9 | |
| K `dno_mix_gemv_q8_0_r8` (up 320→10240) | **39.8** | grid 320 x block 128 |
| K `dno_hc_mixed_f32` | 6.5 | |
| D2H@a (mixed 10 KB) | 6.5 | |

つまり mix は**ほぼ全部が GPU のカーネル時間**で、その 83% が 2 本の GEMV
(71.6 us/呼び出し、96 回で 6.9 ms/tok)。3.48 MB を 109 / 87 GB/s でしか
読めていない。同じ 3.48 MB を v5 形 (GDN の qkv) は 374 GB/s で読んでいた
(`[dec-kprof-avg] op K dno_gemv_q8_0_v5 grid=320 block=64 gpu=37.2`) ので、
形を v5 に合わせるのを第一候補にした。

## 2. 変えたこと (2 コミット。ホスト側だけ)

### 2.1 `MixGemv::V5` と `mix_v5_plan` (`83bf5b1`)

`DENNO_MIX_QK` に `v5` / `v5d` (down だけ) / `v5u` (up だけ) / `ups` (up だけ
seg) を足し、既存の `dno_gemv_q8_0_v5` を mix の down / up で撃てるようにした。

- 形は純関数 `mix_v5_shape(n_in, n_out)` が決める:
  - down (10240→320): `lanes=32` (1 行 = 1 warp、5 対/レーン)、2 行/ブロック
    (block 64)、grid 160。
  - up (320→10240): `lanes=1` (1 レーン = 1 行、5 対)、128 行/ブロック
    (block 128)、grid 80。
- v5 の制約 (`nb = n_in/32` が偶数、`np = nb/2` が lanes で割り切れる) を
  満たさない形は `None` を返して task 282 の (Blk, R8) へ落ちる (fail closed)。
- v5 は引数 **8 本固定**。既存の `launch_moe_gemv_on` は `lanes=1` のとき 6 本
  しか渡さない (rows_per_block / lanes が未定義値になる) ので、mix 用に
  `launch_mix_v5` を置いて常に 8 本渡す。

### 2.2 既定を `(V5, V5)` に (`9ff968b`)

- `DENNO_MIX_QK` 未設定 = `(V5, V5)`。task 282 の形は `=blk` (または `r8`)、
  旧 GEMV は `=0`。
- gate_audit の台帳: `DENNO_MIX_QK` の既定が使うのは `kernels_moe_v5.ptx` の
  `dno_gemv_q8_0_v5` になったので、そちらを見るようにした
  (`kernels_seg_k.ptx` は `DENNO_SEG_LAYER` の項目が引き続き見ている)。
- kernel_invariants: 台帳の期待を v5 に更新し、`task286_mix_gemv_v5_is_wired`
  を追加。

## 3. 数値の扱い

- v5 は 68 B (= 2 ブロック = 64 要素) を 1 レーンが担当し、アキュムレータ 4 本
  → `dno_v5_reduce` の固定木。**決定的**だが、task 282 の (Blk, R8) とも旧
  GEMV とも丸め順が違う。
- lab t286b/t286c の md5: `a` / `b` / `c2000_cold` は baseline と **SAME**。
  `c2000_turn2` は DIFF (t286b の `v5` は `fe926aab` = task 282 以前の既定と
  同じ値、t286c の `on` も同じ `fe926aab`)。greedy の分岐で、本文は sane (4.)。
- `DENNO_MIX_QK=0` (旧 GEMV) と `=blk` (task 282) は 1 ビットも変えない経路。

## 4. 実測 (lab)

### 4.1 t286a (commit `c5cb293`、depth8k、PROFILE=1) — 内訳

COND: b0 / kprof (`DENNO_GDN_DEC_KPROF=1`) / sprof (`DENNO_SEG_PROF=1`
`DENNO_MIX_KPROF=1`) / base / base2。内訳は 1. のとおり。

| COND | short decode ms/tok (RESULT) | prefill s |
| --- | --- | --- |
| b0 (捨て) | 30.9 | 1.40 |
| kprof | 25.4 | 0.81 |
| sprof | 27.3 | 0.82 |
| base | 27.0 | 0.93 |
| base2 (繰り返し) | 25.7 | 0.82 |

生の 128 tok の値は 23〜28 ms/tok (base2 の a2 が 23)。

### 4.2 t286b (commit `83bf5b1`、depth8k) — A/B

COND: b0 / k0 (kprof) / v5d / ups / v5 / base2。

| COND | short decode (RESULT) | 生の a1/a2 (r1 | r2) | prefill s |
| --- | --- | --- | --- |
| b0 (捨て) | 26.3 | 25,23 | 28,24 | 0.82 |
| k0 (kprof) | 25.1 | 25,23 | 25,23 | 0.83 |
| v5d (down だけ v5) | 25.7 | 24,23 | 27,23 | 0.85 |
| ups (up だけ seg) | 23.9 | 24,21 | 24,21 | 0.88 |
| **v5 (両方向)** | **21.4** | **21,21** | **21,20** | 0.83 |
| base2 (繰り返し) | 27.1 | 28,24 | 25,26 | 0.83 |

`[dec-kprof-avg]` の GEMV 1 本の平均 (us):

| 形 | down (10240→320) | up (320→10240) |
| --- | --- | --- |
| task 282 の既定 | `rowblk` 31.9 (grid 320 x 256) | `r8` 39.7 (grid 320 x 128) |
| v5 | **17.8** (grid 160 x 64) | **11.8** (grid 80 x 128) |
| seg (up のみ試した) | - | 12.0 (grid 2560 x 128) |

up は「1 行 1 warp にすると warp 数が 8 倍」で 39.7 → 12.0 になり、v5
(1 レーン 1 行 = 68 B の対読み) はさらにわずかに速い。down は v5 の
「1 行 1 warp + 17 ワード読み」が効いた (31.9 → 17.8)。

### 4.3 t286c (commit `9ff968b`、既定 on、depth8k) — 採否

COND: b0 (`DENNO_MIX_QK=blk`) / on (既定 = v5) / onp (on +
`DENNO_GDN_DEC_KPROF=1`) / base2 (`=blk`)。

| COND | short decode (RESULT) | 生の a1/a2 (r1 | r2) | prefill s | errors |
| --- | --- | --- | --- | --- |
| b0 (捨て) | 44.1 | 56(cold),33 | 25,23 | 1.14 | 0 |
| **on (新しい既定)** | **22.6** | 24,20 | 21,20 | 0.85 | 0 |
| onp (on + kprof) | 21.4 | 21,20 | 21,20 | 0.84 | 0 |
| base2 (繰り返し) | 24.6 | 24,23 | 25,23 | 0.84 | 0 |

VERDICT: `on vs b0 FAIL decode -48.8% prefill -25.0% md5 DIFF errors 0`
/`base2 vs b0 FAIL decode -44.3% prefill -26.3% md5 SAME errors 0`
/`VERDICT_DEPTH on c2000_cold vs b0 decode -19.7% prefill +0.6%` /
`VERDICT_DEPTH on c2000_turn2 vs b0 decode -20.7% prefill -9.6%`。
(b0 は最初の COND で、`a1` が prefill 5.83 s の冷えた 1 発 = 56 ms/tok。
採否の比較は **on vs base2** と生の a1/a2 で読む。b0 は捨て。)

採否の数字を素直に並べる:

| 比較 | 差 |
| --- | --- |
| t286b RESULT: v5 21.4 vs base2 27.1 | **-21.0%** |
| t286c RESULT: on 22.6 vs base2 24.6 | -8.1% |
| t286b 生 a2: 20/21 vs base2 24/26 | -13〜-20% |
| t286c 生 a2: 20/20 vs base2 23/23 | **-13.0%** |
| t286c 長文 (c2000_cold の decode): on 49.7 vs base2 55.0 | -9.6% |
| t286c 長文 (c2000_turn2): on 35.3 vs base2 38.5 | -8.3% |

RESULT の decode 列は 48 token の b リクエストも混ぜた集計なので差が圧縮される
(t286c の on は生 20〜21 なのに列は 22.6)。128 tok の a1/a2 同士では 2 run
とも **20 vs 23 = -13%** で、目標 (-10%) を満たす。長文でも -8〜-10%。

### 4.4 本文 (sane の確認)

`TXT on` (t286c): `# B-Trees: A Comprehensive Explanation\n\n## 1. What Is a
B-Tree?\n\nA **B-tree** is a self-balancing, sorted tree data structure that
maintains sorted data and allows searches, sequential access, insertions, and
deletions in **O(log n)** time. Unlike binary search trees (BSTs), B-trees are
**wide** ...` (b0 / base2 と同一の書き出し)。

md5 が DIFF した `c2000_turn2` (`on`): 「ご提示いただいた文書「第5巻 prefill の
速度改善」の要点を、技術的な背景と具体的な改善プロセスに基づいて詳細に解説します。
### 1. 核心的な発見：律速要因は「計算」ではなく「I/Oと同期」この巻の最大の発見は、
prefill（プロンプトの初期処理）におけるボトルネックが、GPU/CPUの演算能力
（FLOPS/TOPS）ではなく、**ホスト（CPU）とデバイス（GPU）間の往復通信
（Round-trip）およびメモリコピー**にあったという点です。」
(base2 は同じ内容で語尾だけ違う。丸め差で greedy が分岐した。破綻ではない。)

### 4.5 c8000 (14,018 tok) について

ラボの c8000 リクエストは**プロンプトが用意できず**、`prompt=3973 reuse=3954`
の c2000 を再送して空文字を返すことが多い (md5 `d41d8cd9` = 空)。t286a /
t286b / t286c の b0 と base2 でも同じなので**ラボ側の事象**で、この変更の
せいではない。t286b の `v5d` だけは 14,018 tok で走り、prefill 213.08 s /
decode 53.0 ms/tok だった。

## 5. 既定モデル (swift15-abl-IQ3_S) は変わらない

`MODEL=swift15-abl-IQ3_S`、SCENARIO=decode の 1 ラウンド (t286d、commit
`9ff968b`)。COND は b0 (`DENNO_MIX_QK=blk` = task 282 の形) / on (既定 = v5) /
base2 (`=blk`)。

| COND | decode ms/tok | prefill s | md5 | errors |
| --- | --- | --- | --- | --- |
| b0 (捨て) | 19.0 | 0.61 | a:82ab0c06 b:624e0d4d | 0 |
| **on (新しい既定)** | **18.8** | **0.61** | a:82ab0c06 b:624e0d4d | 0 |
| base2 (繰り返し) | 18.8 | 0.60 | a:82ab0c06 b:624e0d4d | 0 |

VERDICT: `on vs b0 OK decode -1.4% prefill -0.8% md5 SAME errors 0` /
`base2 vs b0 OK decode -1.2% prefill -1.6% md5 SAME errors 0`。
**on と base2 は同じ値** (b0 との差は最初の COND のバイアス)。md5 は 2 本とも
SAME、text ok、errors 0 = 既定モデルは 1 ビットも変わらない。

経路としても、IQ3_S の mix は BF16 で `quant_gpu_ty_ok(Bf16)` が false なので
量子型 chain の門番に入らず (`mix_v5_plan` も Q8_0 以外は `None`)、
`DENNO_MIX_QK` の値によらず 1 命令も変わらない。

## 6. テスト

- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`
  -> **1172 passed / 0 failed / 100 ignored**
- `cargo test --release -p denno-engine --features cuda --test kernel_invariants`
  -> **95 passed / 0 failed**
- 追加: `cuda::tests::mix_v5_shape_matches_the_ud_mix` (UD の実形 = lanes 32/1・
  拒否形・block_dim/lanes の不変量)、`mix_qk_gate_defaults_on` に新しい値、
  `kernel_invariants::task286_mix_gemv_v5_is_wired` (8 引数 ABI / 68 B の対読み /
  `mix_v5_shape` と `launch_mix_v5` / 既定 v5 と `=blk` の逃げ道)。

## 7. GPU で見てほしいこと (残り)

- 27B 以外の K 型モデル (例: 9B Q8_0) で `DENNO_MIX_QK` の v5 の効果。mix が
  Q8_0 なら載る。
- 次の候補 (このタスクでは取らなかった): down の v5 はまだ 195 GB/s
  (3.48 MB / 17.8 us)。`rowblk` を 1024 スレッドにした形か、down も seg 形
  (1 行 1 warp) にすると、もう数 us 縮む可能性がある (推定 -0.5 ms/tok)。
- `dno_gemv_q8_0_v3` (full 層の qkv、55.6 us) も v5 に替えられる可能性がある
  (推定 -0.2 ms/tok)。既定モデルにも効く経路なので、別タスクで A/B するのが
  よい。

## 8. 残したゲート

| env | 既定 | 意味 |
| --- | --- | --- |
| `DENNO_MIX_QCHAIN` | on | 量子型 mix の fused chain (off で従来のホスト経路) |
| `DENNO_MIX_QK` | **on (`V5, V5`)** | chain の GEMV。`blk` で task 282 の (Blk, R8)、`0` で旧 `plan_moe_gemv` |
| `DENNO_MIX_KPROF` | off | `[mix-kprof]` の区間計測 |
| `DENNO_GDN_DEC_KPROF` | off | `[dec-kprof-avg]` の op 別表 (mix の GEMV 内訳) |
| `DENNO_SEG_K_PTX` | (空) | seg_k の PTX パス差し替え (`=blk` のときだけ使う) |
| `DENNO_GEMV_V5_PTX` | (空) | v5 の PTX パス差し替え (既定の mix GEMV) |
