ADOPT: yes t278b t278c

# REPORT P278 — UD-Q4_K_XL decode の層一括 expert 経路から固定費を外す

branch: br278 (int-auto 2a7dbf7 = br271 の merge から)。lab: t278a / t278b / t278c
(3 ラウンド)。既定 on にしたゲート: `DENNO_SEG_LAYER` (新経路)、
`DENNO_SEG_K` (高速 GEMV、こちらも既定 on)。

## 0. 結論

- **採る**。UD-Q4_K_XL の warm 短 decode は同一 run の baseline 比で
  **65.9 -> 33.0 ms/tok (-49.9%、t278b)** / **65.5 -> 34.7 ms/tok (-47.0%、t278c)**。
  目標 (-25%) を大きく超えた。prefill は +1.2〜1.8% (ノイズ内)、errors 0、
  a/b の md5 は SAME、長文 2 ターン目だけ DIFF (丸め差。本文は sane)。
- 主因は 2 つあった: (1) 1 層 3 起動 / 2 同期 + ホスト往復 (silu と足し込み) の
  固定費、(2) 旧 GEMV カーネルが **1 行 = 1 スレッドの直列累積**で
  gate 53 / up 54 / down 92 GB/s しか出ないこと (5090 の VRAM は約 1.8 TB/s)。
- IQ 型 (既定モデル swift15-abl-IQ3_S) は `iq_mode` が先に立ち `seg_mode` に入らない
  ので、この変更の影響を受けない (下の 5. に根拠)。

## 1. 前提 (測り直していない。lab の実測)

task 文と lab t268b (`done/t268b/base_1.log`、UD-Q4_K_XL、decode、PROFILE=1、
base COND = `DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000
DENNO_EXPERT_GPU_MIN_HIT=0 DENNO_GPU_WEIGHT_MB=12000`):

| 項目 | 実測 (128 tok の decode) |
| --- | --- |
| decode 全体 | 64.8 ms/tok (base 65.9) |
| `[q4prof] moe experts` | 40.5 ms/tok |
| `[q4ledger] loop` | launches=143/tok syncs=96/tok (2/層) sync=23.2 ms/tok |
| 同 h2d | 235.73 MB/tok / 554 回 (= ミス 76.7 本 x 3.072 MB) |
| 同 d2h | 7.27 MB/tok / 143 回 (= 153 KB/層) |
| `sync_floor` | 2.4 us |

UD の 1 expert = 3,072,000 B (gate/up Q4_K 2560x640、down Q5_1 640x2560)。

## 2. 変えたこと

### 2.1 層を 1 待ちで閉じる (`cuda::moe_seg_layer_sum`)

1 層 = **prologue 1 回の H2D + 5 起動 + イベント 1 本の待ち + D2H e 要素**:

```
[ptr_g|ptr_u|ptr_d|seg_gu|seg_d|w|x] を 1 本のホストバッファに詰めて H2D 1 回
→ gate → up → silu_mul → down → 重み付き和 → イベント待ち → D2H(e)
```

- silu と和は既存の `kernels_dec_epi.ptx` (`dno_dec_silu_mul` / `dno_dec_exp_sum`)。
  和は `acc += w[t] * h[t*e+j]` を t 昇順に 1 アキュムレータで回すので、
  **ミスの無い層では従来のホスト足し込みと同じ演算列** (-fmad=false)。
- 待ちは `cuCtxSynchronize` ではなく **null ストリームのイベント** 1 本。
- prologue のレイアウトは純関数 `expert_batch::seg_stage_layout` (テストで固定)。

### 2.2 ミスは VRAM へ上げず CPU で回す (`qwen4exp::moe_experts_seg_gpu_batched`)

- `lookup_many` でヒットだけ集め、**ミスの `acquire` (1 本 3 MB の写し + H2D) をしない**。
  ミスは `expert_miss_contribs_into` (base と同じ 8 スレッド経路) で GPU の裏に回す。
- 上げ直しは層の後で `iq_warm_misses` に渡す = **非同期** (背景スレッド +
  非ブロッキングストリーム、既存の task 214 経路)。VRAM のヒット率は保つ。
- ヒット 0 本の層は `Contribs(全て None)` = 呼び出し側の CPU 経路 (1 層 30 MB の
  上げ直しを避ける)。VRAM 自体が無い (CUDA 無し / 予算 0) ときは旧経路へ落として
  従来のフォールバック契約をそのまま通す。
  カーネル / PTX / 形が合わなければ `Ok(None)` で旧経路へ (fail closed)。

### 2.3 高速 GEMV (`kernels_seg_k.cu` → `kernels_seg_k.ptx`)

`kernels_moe_batch.cu` は CPU とビット一致させるため 1 行を 64 レーンで写して
lane 0 が直列に足す。これを IQ 経路と同じ **1 行 = 1 warp (32 レーンで k 分割、
`__shfl_down_sync` の固定順で縮約)** にした 4 本を足した:
`dno_seg_gemv_k_q8_0` / `_q5_1` / `_q4k` / `_q5k`。block 128 (4 行) / smem 0 /
grid = (ceil(n_out/4), nseg, 1)。dequant は `kernels_moe_batch.cu` と同一。
**PTX は lab の nvcc で生成したものをコミット** (`kernels_seg_k.ptx`、md5
90930d7c...、t278b が測ったものと同一)。無い / 読めないときは旧カーネルへ
落ち、1 行 WARN を出す。`DENNO_SEG_K=0` で旧カーネルに戻せる。

### 2.4 計測 (`model/seg_prof.rs`、`DENNO_SEG_PROF=1`、既定 off)

層の区間 (VRAM 参照 / 表 / H2D / 起動 / 待ち / D2H / silu / 和 / ミス / 確保) と
カーネルの GPU 時間を CUDA イベントで測り、トークン先頭で 1 行出す。旧経路にも
同じ計器を入れた (t278a の `prof` COND が旧経路の内訳)。

## 3. 数値の扱い

- 下の数字はすべて lab の実測 (RESULT.md / 各 COND のログ)。
- 足し込みの順序が変わる (ヒットを先に全部、ミスを後) ので、ミスのある層では
  丸めが変わる。GEMV も 32 レーンの木になった。silu は `expf` の実装差。
  したがって **md5 は長文で DIFF になりうる** (t278c の c2000_turn2 が DIFF)。
  本文は sane (下に引用)。
- 旧経路 (ゲート off) の数値は 1 ビットも変えていない
  (`DENNO_SEG_LAYER=0` と `DENNO_SEG_K=0` で従来と同一のコード)。

## 4. 実測 (lab)

### 4.1 t278a (commit 1756d15、ptx=none) — 計測 + 新経路の骨格

COND: b0 / prof (旧 + `DENNO_SEG_PROF=1`) / segp (新 + prof) / chk (新 + GPU_CHECK) /
seg (新) / base2。SCENARIO=decode。

| COND | decode ms/tok | prefill s | md5 | errors |
| --- | --- | --- | --- | --- |
| b0 (最初 = 捨て) | 68.8 | 0.83 | SAME | 0 |
| prof (旧) | 66.6 | 0.82 | SAME | 0 |
| segp | 50.4 | 0.82 | SAME | 0 |
| chk | 89.5 | 0.82 | SAME | 0 |
| seg | 50.0 | 0.82 | SAME | 0 |
| base2 (基準) | 65.9 | 0.82 | SAME | 0 |

`[seg-prof]` (1 層あたり):

| 経路 | h2d | launch | sync | d2h | GPU gate | GPU up | GPU silu | GPU down | GPU sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 旧 (prof) | 25 us / 2 回 | 4.5 / 3 | **493 us / 2** | 33 / 2 | 176 | 173 | - | 140 | - |
| 新 (segp) | 2 us / 1 回 | 6.8 / 5 | 440 us / 1 | 8 / 1 | 180 | 172 | 2.4 | 132 | 3.4 |

旧経路の sync 493 us は**カーネル実行そのもの** (イベント合計 489 us と一致)。
「文脈同期が背景の転送を待っている」という私の当初の推定は**外れ**だった。
原因は直列累積で **gate 53 / up 54 / down 92 GB/s** しか出ていないこと。
1 待ち化の効果は 0.84 -> 0.50 ms/層 (base2 比 -24.1%)。

### 4.2 t278b (commit d9c941d、PTX=seg_k、SCENARIO=decode) — 高速 GEMV

| COND | decode ms/tok | prefill s | md5 | errors |
| --- | --- | --- | --- | --- |
| b0 | 66.2 | 0.83 | SAME | 0 |
| prof | 66.4 | 0.81 | SAME | 0 |
| segp | 33.2 | 0.81 | SAME | 0 |
| chk (GPU_CHECK) | 72.2 | 0.81 | SAME | 0 |
| seg | 33.0 | 0.81 | SAME | 0 |
| base2 (基準) | 65.9 | 0.81 | SAME | 0 |

VERDICT: `seg vs b0 OK decode -50.1% prefill -3.6% md5 SAME errors 0`
(= base2 比 **-49.9%**)。`chk vs b0 OK ... md5 SAME`。
`expert_gpu_check` は 20,474 層ぶん出て **max_abs <= 4.8e-7 / max_rel <= 3.1e-3**
(argmax は全層 1/1) = 新しい丸め順でも CPU と実用上同じ値。

`[seg-prof]` 新経路 (1 層あたり、tok=430):

| gate | up | silu | down | sum | 合計 GPU | 待ち (露出) | ミス CPU | gbs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 18.1 us | 18.6 | 2.4 | 35.0 | 3.7 | 78 us | 52 us | 40 us | gate 486 / up 471 / down 347 |

旧カーネルの GB/s (53 / 54 / 92) から **7〜9 倍**。1 層は 0.84 -> 0.11〜0.19 ms。

### 4.3 t278c (commit 4ffed61、**PTX= なし = コミットした PTX**、SCENARIO=depth)

b0 / seg (`DENNO_SEG_LAYER=1 DENNO_SEG_PROF=1`) / base2。

| COND | decode ms/tok | prefill s | md5 | errors |
| --- | --- | --- | --- | --- |
| b0 | 65.8 | 0.82 | SAME | 0 |
| seg | 34.7 | 0.84 | a/b SAME、c2000_turn2 だけ DIFF | 0 |
| base2 (基準) | 65.5 | 0.83 | SAME | 0 |

VERDICT: `seg vs b0 OK decode -47.3% prefill +1.8% md5 DIFF errors 0` /
`VERDICT_DEPTH seg c2000_cold vs b0 decode -50.3% prefill -5.7%`。
base2 比では decode **-47.0%**、prefill +1.2%。

長文 (3,973 tok の 2 ターン目) の decode は base2 112.0 -> seg 52.6 ms/tok
(-53%)、cold は 159.2 -> 73.8 (-54%)。prefill は 50.53 (base2) / 51.48 (seg) /
54.61 (b0) で、base 2 本のばらつき (±4%) の中。

コミットした `kernels_seg_k.ptx` が実際に使われたことは `[seg-prof]` の
gate 464〜482 GB/s で確認した (旧カーネルなら 53 GB/s になる)。

### 4.4 本文 (sane の確認)

`TXT seg` (a、t278c): `# B-Trees: A Comprehensive Explanation\n\n## 1. What Is a
B-Tree?\n\nA **B-tree** is a self-balancing, sorted tree data structure ...`

md5 が DIFF した c2000_turn2 の本文 (seg):
「ご提示いただいた文書「第5巻 prefill の速度改善」の要点を、技術的な背景と具体的な
改善プロセスに基づいて詳細に解説します。### 1. 核心的な発見：律速要因は「計算」では
なく「I/Oと同期」この巻の最大の発見は、prefill（プロンプトの初期処理）における
ボトルネックが、GPU/CPUの演算能力（FLOPS/TOPS）ではなく、**ホスト（CPU）とデバイス
（GPU）間の往復通信（Round-trip）とメモリコピー**にあったという点です。」
(base2 は同じ書き出しで「prefill（プロンプト処理）の遅延が...」と続く。内容は同じで
語尾だけ違う = 丸め差で greedy が分岐した。破綻ではない。)

## 5. 既定モデル (swift15-abl-IQ3_S) が変わらない根拠

lab のラウンド上限 (3) を使い切ったので、既定モデルの実測は取っていない。
代わりに**経路が届かない**ことをコードで示す:

- 新経路の入口は `if use_gpu && seg_mode` だけで、`seg_mode = !iq_mode &&
  seg_layer_enabled()`。既定モデルは expert が全部 IQ 型で `DENNO_EXPERT_IQ_GPU=1`
  なので `iq_mode == true` → `seg_mode == false`。
- 仮に `iq_mode` が false でも `moe_seg_layer_sum` は `moe_gemv_seg_ready` が
  false (IQ 型は `gguf_row_bytes` / `moe_batch_slot` を持たない) なので `Ok(false)`
  で旧経路へ落ちる。GPU を 1 命令も発行しない (cpu_miss も呼ばない)。
- 旧経路 (`moe_gemv_seg_run`) の変更は `DENNO_SEG_PROF` off で `stamp()` が `None`
  を返す計器だけ。カーネル・引数・順序は同一。
- `fast_batched = iq_batched.or(seg_batched)` は、IQ 経路が `Some` の層で
  旧 K 型経路を呼ばない (元の `if use_gpu && !iq_mode` と同じ)。

## 6. テスト

- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`
  -> 1168 passed / 0 failed / 100 ignored (int-auto 0566be1 = dev2 の task 279 を
  merge した後。merge 前は 1147)
- `cargo test --release -p denno-engine --features cuda --test kernel_invariants`
  -> 93 passed / 0 failed (merge 前は 91)
- 追加: `expert_batch::tests::seg_layer_gate_defaults_on` /
  `seg_stage_layout_is_packed_and_aligned`、`seg_prof::tests::*`、
  `cuda::tests::seg_k_gate_defaults_on`、
  `kernel_invariants::task278_seg_k_kernels_are_wired` (カーネル名 4 本 /
  引数 8 個 / warp 縮約 / smem を使わない / build_ptx.sh の target)、
  `gate_audit` の台帳に `DENNO_SEG_LAYER` -> `kernels_seg_k.ptx` を追加
  (既定 on のゲートの PTX 欠落を起動時に WARN する既存の監査に乗る)。

## 7. GPU で見てほしいこと (残り)

- 既定モデルでの 1 リクエスト (`DENNO_SEG_LAYER` 未設定 = 既定 on) の decode /
  md5 が、task 278 以前と 1 ビットも変わらないこと。経路は届かない (5.) が、
  実測 1 本で締めるとよい。
- 27B (swift15) 以外の K 型モデル (例: 9B Q8_0) で `DENNO_SEG_K` の効果。

## 8. 残したゲート

| env | 既定 | 意味 |
| --- | --- | --- |
| `DENNO_SEG_LAYER` | on | 1 待ち経路 (0/off/false で旧 3 起動 / 2 同期) |
| `DENNO_SEG_K` | on | 高速 GEMV (0 で旧 kernels_moe_batch、CPU とビット一致) |
| `DENNO_SEG_PROF` | off | `[seg-prof]` の区間計測 |
| `DENNO_SEG_K_PTX` | (空) | 高速 GEMV の PTX パス差し替え |
