ADOPT: no t266r1b

理由: 目標 (UD-Q4_K_XL の warm short decode -25%、cold prefill -30%) に届かない。
decode は最良の組み合わせでも -2.9% (lab t266r1b: base 64.5 → mix 62.6 ms/tok)、
prefill の m 行 expert GEMM は **4.45 倍遅い** (lab t266r2: 41.98 → 186.83 s / 2,000
token の cold)、SSD 読みの先出しは **0%** (t266r1b の pref 64.6、t266r3 の
mread 64.2 対 base 64.5)。ゲートはすべて既定 off のまま (既定の挙動は不変)。

# REPORT P266 — UD-Q4_K_XL: Q8_0 mix の融合 chain / prefill の m 行 expert GEMM / SSD 読みの先出し

## 前提 (task 266 の記述との差)

- 前提の数字は REPORT_P261.md と一致した。decode 64〜66 ms/tok、うち expert の
  SSD 読みが約 40 ms、mix (Q8_0) が br261 の逐次 GPU 経路で 10〜15 ms。
- **lab の SCENARIO=depth の cold プロンプトは 2,000 token** だった (c2000_cold)。
  task の言う 3,826 token はこのシナリオでは出ない。prefill の比較は
  2,000 token の実測で行い、3,826 は推定 (2,000 の約 1.9 倍) とした。
- br261 は br266 に merge 済み。生かしたのは (a) 量子型 4 種の診断
  (`DENNO_TENSOR_TYPES` / `[exp-iq]` の内訳)、(b) `DENNO_MIX_QUANT_GPU`
  (mix の Q8_0 を decode GEMV で GPU に載せる経路。REPORT_P261 で decode -2.8%)。
  **比較の base COND にはこの `DENNO_MIX_QUANT_GPU=1` を含めた** (br261 の -2.8% を
  含んだ状態から測る = 目標 25% に対して保守側)。
- prefill の m 行 expert GEMM は **P40 の grouped GEMM が既にあった**
  (`kernels_gemm_grouped.cu` + `model/moe_grouped_gpu.rs`、Q8_0 / Q5_1 / Q4_K /
  Q5_K 専用 = このモデルの型そのもの)。新規カーネルは書かず、既定 off のゲート
  `DENNO_MOE_GROUPED_GPU` をこのモデルで初めて測った (過去の REPORT に測定例なし)。

## 1. item 1: 量子型 mix の融合 chain (`DENNO_MIX_QCHAIN`、既定 off) — **-2.9%**

- 新カーネル `dno_mix_lo_silu` (kernels_mix_dev.cu): `lo_apply` = `silu(v*inv_hc)`。
  expf は kernels_elem.cu の `dno_expf` (glibc 逐語再現) を写したので CPU と
  **ビット一致**。`dno_hc_mixed_f32` と合わせて、丸めは従来のホスト経路と同一。
- `maybe_gpu_mix_q_chain_into` (cuda.rs): xn を H2D → down (量子 GEMV, デバイス内)
  → lo_silu → up (量子 GEMV) → hc_mixed → D2H mixed。**1 呼び出し 4 起動 / 1 同期**
  (従来は 2 往復 = 2 同期 + ホストの lo_apply / hc_mixed)。
- 起動だけを行う `CudaCtx::moe_gemv_dev_ptrs` を追加 (既存 `plan_moe_gemv` を共有
  するので v2 / v3 / fp16 のどれでも同じ形で積める)。
- 発火: `[mix-qchain] on calls=2689 hits=2689 frac=1.000` (1 リクエストあたり。
  lab t266r1b。calls=12416 は a2 の長い系列)。**100% 発火**。

| lab t266r1b (decode, UD-Q4_K_XL, 2 ラウンド交互) | decode ms/tok | md5 |
| --- | --- | --- |
| toss (捨て) | 65.4 | a:71a7197c b:9be3fa06 |
| base | 64.5 | SAME |
| **mix** | **62.6 (-2.9% vs base)** | SAME |
| pref (item 3 予測版) | 64.6 (+0.2%) | SAME |
| both | 63.0 | SAME |
| base2 (繰り返し) | 64.2 | SAME |

`VERDICT mix vs toss OK decode -4.3% prefill -9.5% md5 SAME errors 0` (toss 基準)。
mix は base 比 -2.9%、base2 比 -2.5%。md5 は全 COND で SAME = 値は動いていない。

## 2. item 2: prefill の m 行 expert GEMM (`DENNO_MOE_GROUPED_GPU`) — **4.45 倍遅い**

既存の P40 grouped GEMM は「1 層の選択 expert の重みを 1 本に詰めてから、16x16 の
タイルで m 行をまとめて回す」形で、task の求める形 (kernels_exp_iq_pf と同じ) そのもの。
型の関門 `block_type_ok` (4 型以外は配車前に落とす) だけ足して測った。

| lab t266r2 (depth, UD-Q4_K_XL) | cold prefill (2,000 tok) | 次の turn の prefill | decode ms/tok |
| --- | --- | --- | --- |
| base | 41.98 s | 0.94 s | 64.1 |
| **grp** (`DENNO_MOE_GROUPED_GPU=1`) | **186.83 s (+345%)** | 2.91 s | 64.4 |
| best (grp+mix+pref) | 224.21 s | 3.44 s | 79.0 |

`VERDICT grp vs toss` は md5 DIFF (prefill は丸め契約の外なので想定内。ただし
decode のテキストまで変わる = prefill の差が波及する)。**推定**の原因は、層ごとに
選択 expert の重み (実測 414 MiB/層、`[mem-audit] grouped packed`) をホストへ
詰め直してから H2D する二重のコピーで、prefill 全体では約 20 GB の pack +
約 20 GB の H2D が読み (約 20 GB) の上に乗る。カーネル自体の速度ではない
(1 層あたりの GPU 時間は小さい)。対策するなら「pack を経由せず mmap のまま
grouped GEMM に食わせる」カーネルが要る (今回は .cu を増やさない方針で見送り)。

## 3. item 3: SSD 読みの先出し — **0%** (2 通り試して両方不加)

### 3a. 1 トークン前の選択で先読み (`DENNO_DEC_PREF`、既定 off)

decode のトークン先頭で、前のトークンの同じ層の選択のうち VRAM に無い expert を
背景スレッド (2 本) で `madvise(MADV_WILLNEED)`。**1 本も積まれなかった**。
診断 (`[dec-pref] tokens=128 sels=6144 snap/tok=480 skip_res=61440 sub=0`) が理由を
示している: **直前のトークンが使った expert は必ず LRU に入っている** (ミスは
その場で VRAM へ上げるため)。つまり「予測 = 直前の選択」は構造的に空振りする。
ミスするのは「直前に使われなかった長尾の expert」なので、当てる手掛かりが無い。

### 3b. ミスのページを H2D の前に並列で読む (`DENNO_DEC_MISS_READ`、既定 off)

`moe_experts_gpu_batched` の acquire 直前に、ミスする expert の 3 行列を
`DENNO_DEC_PREF_CHUNKS` (既定 2) 分割して背景 2 スレッドで触り、ページを
page cache に載せる (メインスレッドはそのまま H2D。同じページのフォルトは
ページロックで 1 回に畳まれる)。

| lab t266r3 (decode, UD-Q4_K_XL) | decode ms/tok | md5 |
| --- | --- | --- |
| toss | 64.3 | a:71a7197c b:9be3fa06 |
| base | 64.5 | SAME |
| mread (chunks=2) | 64.2 | SAME |
| mread4 (chunks=4) | 64.5 | SAME |
| best (mix+mread) | 64.3 (※ mix は下記の理由で不発) | SAME |
| base2 | 64.2 | SAME |

`[dec-pref] miss_sub=9899 bytes=31120MB` = 1 リクエストで 31 GB を実際に触ったが、
decode の時間は動かない。**推定**: 読みは既にディスク帯域 (3.6 GB/s) で回っており、
並列化しても総バイト数が同じなので縮まない。0.86 ms/本は「3.07 MB / 3.6 GB/s」で
あって、レイテンシでもキュー深さでもない (31 GB を 0.8 s で触れたのは、その大半が
page cache に載っていたため = ディスク律速ですらない局面がある)。
※ t266r3 の `mix` は hits=0: この SHA は mix_dev の新 PTX を収録する**前**なので
  `dno_mix_lo_silu` が無く、chain が fail closed で従来経路に落ちた (想定どおり)。

## 4. 既定モデル (swift15-abl-IQ3_S) は不変 — lab t266r4

| cond | decode ms/tok | prefill s | md5 |
| --- | --- | --- | --- |
| base | 19.5 | 0.61 | a:82ab0c06 b:624e0d4d |
| all (MIX_QUANT_GPU + MIX_QCHAIN + DEC_MISS_READ) | 19.4 | 0.61 | **SAME** |
| base2 | 19.3 | 0.61 | SAME |

`VERDICT all vs base OK decode -0.8% prefill +0.0% md5 SAME errors 0` (-0.8% はノイズ)。
`[mix-qchain] hits=0` (mix は BF16 なので量子 chain に入らない)、`[dec-pref]` は出ない
(IQ expert は IQ 経路で acquire しない)。既定の数値・md5 は不変。

## 5. テキスト (lab t266r1b base、sane)

`TXT base: "# B-Trees: A Comprehensive Explanation\n\n## 1. What Is a B-Tree?\n\nA **B-tree** is a self-balancing, sorted tree data structure ..."`
`TXTB base: "Rust の「所有権（Ownership）」と「ライフタイム（Lifetime）」は、Rust がメモリセーフティ…"`
全 COND で errors 0、text "looks ok"。

## 6. 何が残ったか (次のタスクへの申し送り)

1. **mix の融合 chain は -2.9% で確実に効く** (md5 SAME、`[mix-qchain] frac=1.000`)。
   目標 25% には足りないので既定 off のまま。単独で採るならこのゲートだけ on に
   すればよい (他モデルには無害: BF16 mix では 1 度も入らない)。
2. **decode の 40 ms は「容量」か「帯域」の問題**。予測 (直前トークン) は原理的に
   無効 (skip_res=480/480)、並列読みも効かない (t266r3)。残る手は
   (a) VRAM 容量 (26 GB で -6.7% が上限・P261 で測定済み)、(b) ホスト RAM の
   階層 (task 263 の領分)、(c) 帯域そのもの (Gen5 x16 でも SSD は 3.6 GB/s のまま)。
3. **grouped prefill は pack をやめないと勝てない**。今回は 4.45 倍遅いという
   実測だけ残った (二重コピーが支配的、というのは推定)。

## 数値の扱い

- item 1: CPU とビット一致 (md5 SAME を lab で確認)。
- item 2: prefill は契約外。decode の md5 も変わる (prefill の差が波及)。
- item 3: 読む順番だけ。md5 SAME。
- 目標に対する比較は、すべて自分の run の base COND (toss/base/base2) に対して。

## テスト (CPU 関門)

- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`
  → 1096 passed / 0 failed (追加: exp_prefetch 6 本、mix_qchain 3 本、
  moe_grouped の型関門 1 本)。
- `--test kernel_invariants` → 88 passed / 0 failed。

## 使った lab (4 本。うち 1 本は PTX のコンパイル失敗で測定なし)

| name | 内容 | 結果 |
| --- | --- | --- |
| t266r1 | PTX=mix_dev の 1 回目 | PTX_FAILED (cstdint 漏れ) → 修正 |
| t266r1b | decode, UD, PTX=mix_dev, 6 COND | mix -2.9%、pref 0% |
| t266r2 | depth, UD, 6 COND | grouped prefill +345%、pref 0% |
| t266r3 | decode, UD, 6 COND | mread 0% |
| t266r4 | decode, 既定モデル, 3 COND | 不変 (md5 SAME) |
