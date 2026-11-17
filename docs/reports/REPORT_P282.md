ADOPT: yes t282c

# REPORT P282 — UD-Q4_K_XL decode: hc mix の fused chain と GEMV 2 本

branch: br282 (int-auto f1d7413 = task 278 の merge から。int-auto はその後動いて
いないので merge は無し)。lab: t282a / t282b / t282c (3 ラウンド)。
既定を変えたゲート: `DENNO_MIX_QCHAIN` (off → **on**)、`DENNO_MIX_QK` (新規、既定
`down=rowblk / up=r8`)。

## 0. 結論

- **採る**。UD-Q4_K_XL の warm 短 decode は同じ run の既定 (off) 33.4 → **26.5
  ms/tok (-20.7%)**、繰り返し baseline (base2 34.0) 比 **-22.1%**。目標 (-15%) を
  超えた。prefill は **0.81 → 0.81 s (0.0%)**、errors 0、text sane。md5 は
  a / b / c2000_cold が SAME、c2000_turn2 だけ DIFF (丸め差。下に本文)。
- 効いたのは 3 つ: (1) 量子型 mix の fused chain (task 266 の既定 off の経路) を
  既定 on にしたこと、(2) chain の GEMV を **1 行 1 ブロック / 1 warp 8 行**の
  2 本に替えたこと、(3) `hc_inject` のホスト GEMV を GPU の待ちに隠したこと。
- `[q4prof]` の mix は **0.337 → 0.181 ms/層** (16.2 → 8.7 ms/tok)。decode の
  残りは moe experts 6.6 / attn 5.9 / router 1.9 ms/tok。
- **prefill は chain を撃たない** (decode 専用)。このモデルの prefill は 1 トークン
  ずつ `hc_mix_quant_fast` を通るため、chain を使うと prefill の GPU 仕事の後ろに
  同期が並んで 6〜10% 遅くなる (t282a/t282b で実測)。ガードを入れて 0% に戻した。

## 1. 前提 (測り直していない。task 278 の REPORT と lab の実測)

- task 278 の後、int-auto の UD 短 decode は 33 ms/tok。内訳 (t278c、128 tok):
  mix 16.2 / moe experts 6.6 / attn 5.9 / router 1.9 / head+norm 1.3 ms/tok。
  mix が最大で、1 トークン 96 呼び出し x 7 MB = 672 MB を CPU の DRAM
  41 GB/s で流していた (ホストでは天井)。
- UD の mix: down = [320, 10240] Q8_0、up = [10240, 320] Q8_0、inject = [4, 10240]
  **F32**。`DENNO_MIX_GPU=1` は lab の本番 env で常に付くが、量子型の mix は
  `DENNO_MIX_QUANT_GPU` (既定 off) が無いと GPU に載らない。

## 2. 変えたこと (3 コミット。すべて env ゲートの内側)

### 2.1 量子型 mix の fused chain を既定 on (`DENNO_MIX_QCHAIN`)

task 266 が作って既定 off だった経路 (down → lo_silu → up → hc_mixed を 1 同期) を
既定 on にした。task 266 の時点では decode 64 ms/tok で expert が 40 ms あり、
mix の -2 ms は -2.9% にしかならなかった (採否 no)。task 278 の後は mix が最大の
項目なので同じコードが効く。

- 待ちを `cuCtxSynchronize` から null ストリームのイベント 1 本に変えた
  (task 278 の層一括経路と同じ理由: 背景の上げ直しストリームを待たない)。
- **prefill では撃たない**。`forward_prefill_inner` がスレッドローカルのガードを
  立て、`hc_mix_quant_fast` の門番が `!in_prefill_seq()` を見る。decode の
  `forward_token_at` はこのガードの外なので影響を受けない。

### 2.2 `hc_inject` の GEMV を GPU の待ちに隠す

UD の `hc_*_inject` は F32 (4 行 x 10240) で、1 呼び出し 16.9 us がホストで直列に
走っていた (`[mix-kprof] inj`)。chain を積んだあと・待つ前に移すと GPU の
0.14 ms の裏に隠れる。値はビット一致 (順序だけの変更)。鎖が不発のときは
従来位置で計算し直す。

### 2.3 mix の GEMV を 2 本追加 (`DENNO_MIX_QK`)

t282b の `[dec-kprof-avg]` (mix 1 窓 = 1 層) の実測:

| 項目 | us |
| --- | --- |
| kern 合計 | 117.2 |
| うち `dno_gemv_q8_0_v3` (1 本) | **53.8** |
| `dno_hc_mixed_f32` | 6.0 |
| `dno_mix_lo_silu` | 4.3 |
| copy (H2D+D2H) | 14.2 |
| launch | 19.0 |
| host (norm 7.5 / inj 19.5) | 27.0 |

従来の `plan_moe_gemv` は `lanes=2 / rows=8 / blockDim=16` で、down は GPU 全体で
640 スレッドしか使わず、3.5 MB に 54 us (65 GB/s)。`kernels_seg_k.cu` に 2 本
足した (PTX は lab の nvcc で生成、`PTX=seg_k`):

| カーネル | 形 | 起動形 |
| --- | --- | --- |
| `dno_mix_gemv_q8_0_rowblk` | 1 行 = 1 ブロック 256 スレッド (k を 8 warp に分割、部分和 8 本を固定順で加算) | grid = 行数、smem 32 B |
| `dno_mix_gemv_q8_0_r8` | 1 warp = 8 行 (1 行が短い形用。固定費を 8 分の 1 に) | grid = ceil(行数/32) |

`DENNO_MIX_QK` の値: 未設定 = `(Blk, R8)` / `0` = 旧 GEMV 両方 / `down` / `up` /
`seg` (task 278 の 1 行 1 warp 版)。

**t282a の seg_k (1 行 1 warp) は mix では負けだった** (decode 33.8 vs base 33.0、
mix 0.365 ms/層)。up が 1 行 320 要素 = warp が 10 反復しかしないため、行アドレス /
segment 表 / warp 縮約 / store の固定費が 340 B に対して重い。r8 はその固定費を
8 分の 1 にする。

## 3. 数値の扱い

- rowblk の縮約は「warp 内 `sk_reduce` の固定木 → 8 warp の部分和を thread 0 が
  0..7 の順で加算」で**決定的**。1 行 1 スレッドの逐次和とは丸め順が違う。
- r8 は 1 行 1 warp 版と同一の木 (8 本のアキュムレータは互いに独立)。
- したがって **md5 は長文で DIFF になりうる** (t282c の c2000_turn2 が DIFF)。
  本文は sane (下に引用)。
- `DENNO_MIX_QCHAIN=0` で従来のホスト経路、`DENNO_MIX_QK=0` で旧 GEMV に戻る。
  どちらも 1 ビットも変えない経路。

## 4. 実測 (lab)

### 4.1 t282a (decode、commit 2eff672、ptx=none) — 最初の A/B

COND: b0 / base / q0 (chain + 旧 GEMV) / q1 (chain + seg_k 両方向) / base2。

| COND | decode ms/tok | prefill s | md5 |
| --- | --- | --- | --- |
| b0 (捨て) | 33.2 | 0.85 | SAME |
| base | 33.0 | 0.81 | SAME |
| q0 | **29.6 (-10.3% vs base)** | 0.89 (+10%) | a DIFF |
| q1 | 33.8 (+2.4%) | 0.94 (+16%) | a DIFF |
| base2 | 32.9 | 0.80 | SAME |

`[q4prof]` の mix: q0 0.285 / q1 0.365〜0.370 ms/層。seg_k が負け、prefill が悪化。

### 4.2 t282b (depth、commit de51d5c、ptx=none) — 内訳と候補

COND: b0 / base / c1 (chain + 旧 GEMV + inject 隠し) / c2 (down だけ seg_k) /
cd (c1 + `DENNO_GDN_DEC_KPROF=1`) / base2。

| COND | decode ms/tok | prefill s |
| --- | --- | --- |
| b0 (捨て) | 39.3 | 1.15 |
| base | 34.5 | 0.84 |
| c1 | **28.9 (-16.2% vs base、-14.5% vs base2)** | 0.89 (+6%) |
| c2 | 33.5 | 0.93 |
| cd | 29.0 | 0.89 |
| base2 | 33.8 | 0.83 |

**prefill の +6% が問題**だったので chain を decode 専用にした (2.1)。
`[dec-kprof-avg]` の 54 us/GEMV もここで分かった。

### 4.3 t282c (depth、commit ee0403c、**PTX=seg_k**、既定 on) — 採否

COND: b0 / off (`DENNO_MIX_QCHAIN=0`) / on (既定 = 採否後の形) / qk0
(`DENNO_MIX_QK=0`) / base2。

| COND | decode ms/tok | prefill s | errors | md5 |
| --- | --- | --- | --- | --- |
| b0 (捨て) | 34.8 | 0.95 | 0 | SAME |
| **off** (いまの既定) | **33.4** | 0.81 | 0 | SAME |
| **on** (新しい既定) | **26.5 (-20.7% vs off、-22.1% vs base2)** | **0.81 (0.0%)** | 0 | a/b/cold SAME、turn2 DIFF |
| qk0 (chain + 旧 GEMV) | 31.3 (-6.3% vs off) | 0.83 | 0 | turn2 DIFF |
| base2 (繰り返し) | 34.0 | 0.82 | 0 | SAME |

VERDICT: `on vs b0 OK decode -23.8% prefill -14.2% md5 DIFF errors 0` /
`off vs b0 OK decode -3.9% prefill -14.7% md5 SAME errors 0`。
(b0 は最初の COND でバイアスが乗る。採否の比較は **on vs off / base2** で読む。)

長い文脈 (3,973 tok の 2 ターン目):

| COND | c2000_cold decode | c2000_turn2 decode | c2000_cold prefill |
| --- | --- | --- | --- |
| off | 67.5 | 50.6 | 50.66 |
| **on** | **54.7 (-19.0%)** | **40.9 (-19.2%)** | 50.75 (+0.2%) |
| base2 | 67.0 | 52.9 | 51.19 |

確認:
- `[mix-qchain] on calls=12416 hits=12416 frac=1.000 qk=12416` (decode)。
  **prefill は `hits=0`** (ガードが効いている)。
- `[q4prof]` の mix: 0.181 ms/層 = 8.7 ms/tok (off は 0.337 = 16.2)。
- コミットした `kernels_seg_k.ptx` (md5 5e46a0559d0f26c7996250e1884e7f21) が
  実際に使われたことは `qk=12416` (chain と同数) で確認した。既存 4 本の
  エントリは再生成で 1 バイトも変わっていない (diff は追加のみ、+1126 行)。

### 4.4 本文 (sane の確認)

`TXT on` (t282c、a): `# B-Trees: A Comprehensive Explanation\n\n## 1. What Is a
B-Tree?\n\nA **B-tree** is a self-balancing, sorted tree data structure that
maintains sorted data and allows searches, sequential access, insertions, and
deletions in **O(log n)** time. Unlike binary search trees (BSTs), B-trees are
**wide** (many children per node) and **shallow** ...` (off / base2 と同一の書き出し)。

md5 が DIFF した c2000_turn2 (`on`):
「ご提示いただいた文書「第5巻 prefill の速度改善」の要点を、技術的な背景と具体的な
改善プロセスに基づいて詳細に解説します。### 1. 核心的な発見：律速要因は「計算」
ではなく「I/Oと同期」この巻の最大の発見は、prefill（プロンプトの初期処理）に
おけるボトルネックが、GPU/CPUの演算能力（FLOPS/TOPS）ではなく、**ホスト（CPU）と
デバイス（GPU）間の往復通信（Round-trip）およびメモリコピー**にあったという点です。」
(off は同じ内容で「Round-trip）とメモリコピー」= 助詞だけ違う。丸め差で greedy が
分岐した。破綻ではない。)

## 5. 既定モデル (swift15-abl-IQ3_S) が変わらない根拠

lab のラウンド上限 (3) を UD に使い切ったので、既定モデルの実測は取っていない
(task 278 と同じ扱い)。代わりに**経路が届かない**ことをコードで示す:

- IQ3_S の mix は BF16。`quant_gpu_ty_ok(Bf16)` は false なので、量子型 chain の
  門番 (`mix_qchain_enabled() && quant_gpu_ty_ok(down.ty()) && quant_gpu_ty_ok(up.ty())`)
  に入らない。BF16 は先に chain2 (`DENNO_DECODE_CHAIN2_MIX`) で閉じる。
- `hc_inject` を隠すフックは chain の中だけで呼ばれる。`qchain_will` が false の
  モデルでは `inj_early` の位置も従来のまま。
- 新しい 2 カーネルは chain からしか撃たない。`DENNO_MIX_QCHAIN=0` と同じ経路。
- `[mix-qchain]` の行は **calls が 0 なら出さない**ようにしたので、既定モデルの
  ログも 1 バイト変わらない。
- テストで固定: `qwen4exp::mix_qchain_tests::qchain_types_are_the_block_four`
  (BF16 / F32 / F16 / IQ 系は chain の対象外)、
  `prefill_guard_turns_the_qchain_gate_off`。

## 6. テスト

- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`
  -> **1171 passed / 0 failed / 100 ignored**
- `cargo test --release -p denno-engine --features cuda --test kernel_invariants`
  -> **94 passed / 0 failed**
- 追加: `cuda::tests::mix_qk_gate_defaults_on` (種別表) /
  `mix_qk_stage_layout_is_packed_and_aligned` (前置き 48 B のレイアウト)、
  `qwen4exp::mix_qchain_tests::prefill_guard_turns_the_qchain_gate_off` /
  `qchain_line_is_quiet_without_calls`、
  `kernel_invariants::task282_mix_gemv_kernels_are_wired` (カーネル名 2 本 /
  引数 8・6 個 / grid 32 行・行数 / sk_reduce の木 / 台帳)、
  `gate_audit` の台帳に `DENNO_MIX_QCHAIN` -> `kernels_mix_dev.ptx` と
  `DENNO_MIX_QK` -> `kernels_seg_k.ptx` を追加。
- 注意: `kernels_seg_k.ptx` は lab で再生成したものを入れた (このコンテナに nvcc は
  無い)。中間コミット ee0403c では gate_audit の 2 本が「PTX に新しい entry が
  無い」で赤かった。PTX を入れた最終コミットで緑。

## 7. GPU で見てほしいこと (残り)

- 既定モデル (IQ3_S) で 1 リクエスト。decode / md5 が task 278 以前と 1 ビットも
  変わらないこと (経路は届かない (5.) が、実測 1 本で締めるとよい)。
- 27B 以外の K 型モデル (例: 9B Q8_0) で `DENNO_MIX_QCHAIN` の効果。
  mix が Q8_0 なら chain が載る。

## 8. 残したゲート

| env | 既定 | 意味 |
| --- | --- | --- |
| `DENNO_MIX_QCHAIN` | **on** | 量子型 mix の fused chain (off で従来のホスト経路) |
| `DENNO_MIX_QK` | **on** (`Blk,R8`) | chain の GEMV 種別。`0` で旧 `plan_moe_gemv` |
| `DENNO_MIX_KPROF` | off | `[mix-kprof]` の区間計測 |
| `DENNO_GDN_DEC_KPROF` | off | `[dec-kprof-avg]` の op 別表 (mix の GEMV 内訳) |
| `DENNO_SEG_K_PTX` | (空) | seg_k の PTX パス差し替え (`DENNO_MIX_QK` も同じ file) |
