# REPORT P53 — q4 GEMM の `full_oproj` / `full_qk` が遅い理由

作業: `/work/denno-dev2` / branch `p53` (親 `d4ee7d90` = `sMain`)。
`.cu` は 1 行も触っていない (PTX 再生成は不要)。実測は行っていない (GPU 非搭載)。

## 0. 結論 (先に)

1. **`full_oproj` は既に天井にいる。** 与えられた 0.44 s と実コードの形 (k=6,144 / n=5,120 /
   16 層) から 32.1 TFLOP/s。指示書の 15.9 は **層数を 16 ではなく 8 で割った値**と一致する
   (14.14 TFLOP / 2 = 7.07 TFLOP、7.07/0.44 = 16.1)。詰める余地は無い。
2. **`full_qk` の 15.7 TFLOP/s は GEMM の効率ではない。** この位相は
   「3 発の GEMM + **805.3 MB/層の D2H** + sync 1 回」で、D2H が 0.90 s を占める。
   GEMM だけなら 1.10 s (30 TFLOP/s) で、1.10 + 0.90 = 2.00 s ≒ 実測 2.11 s (差 0.11 s は
   sync と端ブロック)。**GEMM は他の位相と同じ 30 TFLOP/s で回っている。**
3. したがって「天井まで詰めると −1.3 s」は成立しない。残っているのは
   PCIe 上の D2H 0.90 s で、そのうち **k/v GEMM の後ろに隠せる 0.16 s (推定)** だけを
   このタスクで実装した (`.cu` 不要・数値不変)。残りは attention のデバイス化 (`.cu`) が前提。

## 1. 形状の確定値 (すべてコードの行から)

27B (Qwen3.5-27B, Q4_0 ネイティブ)。dims はテスト用の合成シェイプの注記が実値を書いている
(`crates/denno-engine/src/model/qwen35.rs:4579-4596`): e=5,120 / heads=24 / kv_heads=4 /
dh=256 / ff=17,408 / 64 層 = **48 GDN + 16 full** (interval 4)。

| 投影 | 重みの形 (k → n) | 根拠 | m | 層数 | 呼び出し/req | 量子化 |
|---|---|---|---|---|---|---|
| q (q\|gate 融合) | 5,120 → 12,288 (= 2·24·256) | `qwen35.rs:804` `let q_total = heads * dh * 2;` / `:807` `q_w: lin_g(... &[eu, q_total as u64], e, q_total)` | 14,043 | 16 | 16 | Q4_0 (`Linear::Q4`) |
| k | 5,120 → 1,024 (= 4·256) | `qwen35.rs:808-813` `&[eu, (kv_heads * dh) as u64], e, kv_heads * dh` | 14,043 | 16 | 16 | Q4_0 |
| v | 5,120 → 1,024 | `qwen35.rs:814-819` 同上 | 14,043 | 16 | 16 | Q4_0 |
| o_proj | 6,144 (= 24·256) → 5,120 | `qwen35.rs:822-827` `o_w: lin_g(... &[(heads * dh) as u64, eu], heads * dh, e)` | 14,043 | 16 | 16 | Q4_0 |

- 起動: `cuda.rs:13542` `chain_qkv_prefill_q4_dev` が 3 発
  (`cuda.rs:13597-13603`: `chain_gemm_q4_pf(gwq, pin, pqkv, rows, qn, qcols)` →
  `(gwk, ... pqkv + q_off ...)` → `(gwv, ... pqkv + k_off ...)`)。
  o_proj は `cuda.rs:13712` `chain_oproj_prefill_q4` の
  `cuda.rs:13739` `chain_gemm_q4_pf(gw, pa, po, rows, ow.k, ow.n)` の 1 発。
- カーネル: `dno_gemm_q4_0` (`kernels/kernels_int8.cu:605`)。1 block = A 64 行 × B 128 列
  (`Q4_GM=64` `:600` / `Q4_GN=128` `:601`)、block=(16,8)=128 thread、1 thread 8×8=64 acc
  (P52/s249 で 4×8 → 8×8、`:597-599`)。grid は `(m.div_ceil(64), n.div_ceil(128))`
  (`cuda.rs:5740`、per-op 側も同一 `cuda.rs:3839`)。smem は sA 64×32×4 + sB 128×33×4
  = 24.9 KB (`static_assert` で 48 KB 未満を強制、`.cu:602`)。
- 位相の区切り: `cuda.rs:13591` `pf_phase(PF_PH_FULL_QK)` から
  `cuda.rs:13690` の `ctx.sync()` まで = **GEMM 3 発 + D2H 3 発 + sync 1 回**。

## 2. なぜこの 2 つだけ TFLOP/s が半分なのか (算術)

m = 14,043。FLOP は `2·m·n·k`。

| 位相 | FLOP/req | 実測 s | TFLOP/s |
|---|---|---|---|
| ffn_gate (n=34,816, k=5,120, 64 層) | 320.4 T | 10.06 | 31.8 |
| ffn_down (n=5,120, k=17,408, 64 層) | 160.2 T | 5.33 | 30.1 |
| gdn_oproj (n=5,120, k=6,144, 48 層) | 42.4 T | 1.40 | 30.3 |
| **full_qk (q+k+v, 16 層)** | **33.0 T** | **2.11** | **15.6** |
| **full_oproj (n=5,120, k=6,144, 16 層)** | **14.14 T** | **0.44** | **32.1** |

full_oproj は同形の gdn_oproj (30.3) と同じ帯域で、**GEMM として遅くない**。
15.9 という数字は FLOP を半分に見積もった値 (層数 8) で、0.44 s から逆算すると
16.1 TFLOP/s になる。

full_qk の 33.0 TFLOP を 30 TFLOP/s で割ると **1.10 s**。実測 2.11 s との差 **1.01 s** が
GEMM 以外 = D2H。バイト数:

- q = 14,043 × 12,288 × 4 = **690,241,536 B (690.2 MB)**
- k = 14,043 × 1,024 × 4 = 57,520,128 B (57.5 MB)
- v = 同上 57.5 MB (宛先はホスト KV キャッシュの行、P44)
- 合計 **805,281,792 B (805.3 MB)/層** (q+k は 747.8 MB = P47 の pinned プール上限内)

この環境の pinned D2H 実測単価は **14.3 GB/s** (`docs/history/40_prefill_perf.md:84`、
s68 の実測 = 上限。345.1 MB → 24.6 ms で 14.0 GB/s と逆算する記述が `:1015`)。
805.3 MB / 14.3 GB/s = **56.3 ms/層** (q だけで 48.3 ms)。16 層で **0.90 s**。

**1.10 s (GEMM) + 0.90 s (D2H) = 2.00 s**、実測 2.11 s。差 0.11 s は
`cuCtxSynchronize` の drain (1 回/層) と端ブロック (14,043 = 219·64 + 27 行)。

roofline で言い直すと: q の GEMM は 1.767 TFLOP を 58.9 ms で処理し、
A の再読み (grid.y=96 回 × 287 MB = 27.6 GB/layer) を含めて ≈470 GB/s を要求する。
HBM 帯域には余裕があり、律速は共有メモリ経由の FMA/LDS 比 (1 要素 64 FFMA / 16 LDS = 4.0)
= SM の FFMA:LDS 比 4.0 ちょうど。ここが 30 TFLOP/s の理由で、**n が小さいからでも
K が小さいからでも起動回数が多いからでもない** (k/v の n=1,024 は 9.8 ms/層しかない)。

## 3. 実装した変更

`crates/denno-engine/src/cuda.rs` のみ (`.cu` / `.ptx` は不変)。

1. `ChainScratch` に `qk_stream` / `qk_ev_q` を追加 (`:4507-4511`、初期化 `:4557-4558`)。
   P6 の h2 用一式 (`h2_stream` 等) とは別に持つ — 同じストリームだと h2 の D2H と直列化し、
   `h2_pending` の join 契約と衝突する。
2. `pf_qk_copy_begin()` (`:13483`): 非ブロッキングのコピーストリーム + イベントを作り、
   **既定ストリームの「いまこの時点」まで**をイベントで受けさせる。
   失敗 (フラグ OFF・driver API 欠落・作成失敗) はすべて `None` = 従来経路。
   `GPU_BROKEN` は立てない (`chain_err` を通さない)。
3. `chain_qkv_prefill_q4_dev` (`:13542`): q の GEMM 起動の**直後**・k/v の GEMM 起動の
   **前**に参加点を置き (`:13601`)、**q の D2H だけ**をコピーストリームへ (`:13666`)。
   k/v の D2H は既定ストリーム・宛先・発行順とも不変。`ctx.sync()` は 1 回のまま
   (`cuCtxSynchronize` = コンテキスト全体なのでコピーストリームも join される)。
4. フラグ: `DENNO_PF_QK_OVERLAP` (既定 ON、`parse_pf_qk_overlap` `:13449`)。
   `=0` / `false` / `off` で従来の直列へ戻る。上位の `DENNO_PF_OVERLAP=0` でも OFF。

### 数値不変の根拠

- カーネル・grid・引数・宛先・バイト数・発行順は 1 つも変えていない。変えたのは
  「q の `cuMemcpyDtoHAsync` の第 4 引数 (ストリーム)」だけ。
- q の D2H は `dst_q` の先頭 `q_bytes` へ、`CH_QKV` の先頭 `q_bytes` から落とす (従来同一)。
  コピーが読むのは q の GEMM が書く領域だけなので、k/v の GEMM が書く領域とは重ならない。
- ホストは従来どおり `ctx.sync()` の後に pinned ステージングを読む。ビット列は同一。

### 見積り (推定)

- 必ず隠れる: k/v の GEMM ≈9.8 ms/層 → 16 層で **≈0.16 s/request (推定)**。
- 上限: k/v の D2H (8.0 ms/層) も隠れれば **≈0.29 s/request (推定)**。
  k/v のコピーは q と同じコピーエンジンに並ぶので、実機の取り合い次第。
- 65k wall 35.8 s に対して 0.4〜0.8%。**小さい。** 1.3 s にはならない (§0/§2)。

## 4. 実装しなかった候補 (設計のみ)

### (a) q の GEMM と D2H を行タイルでパイプライン化 — 推定 −0.6〜0.7 s/request
q を 2,048 行 (Q4_GM=64 の倍数) のタイルに割り、tile t の GEMM 起動直後にイベントを打って
tile t の D2H をコピーストリームで流す。q のコピー 48.3 ms のうち最後の 1 タイル分
(≈6 ms) 以外が GEMM の裏に入り、位相は 107 ms/層 → ≈65 ms/層になる (推定)。
各行は互いに独立で、タイルはポインタオフセット (`a + r0*k*4`, `out + r0*n*4`) で渡すだけ
なので累積順は不変 = ビット一致。
**採らなかった理由**: `tests/kernel_invariants.rs` の
`prefill_full_qk_d2h_lands_in_pinned_staging` が「D2H は 3 発のまま・連結も分割も禁止」を
固定している (P17b の 1 発連結が実測 +772 ms/層で revert された経緯)。分割はこの検査を
緩めることになるので、テストを緩めない条件と両立しない。採るなら検査を
「q は行タイルごとに 1 発・合計バイト数は `rows*qcols*4` で不変」へ**置き換える**判断
(オペレーター) が要る。

### (b) attention / qknorm / rope のデバイス化 — 推定 −0.9 s/request
q/k/v の D2H 805.3 MB/層が丸ごと消える (§2)。`full_core` のホスト側 312 ms/層も同時に
減る。`.cu` の新規カーネル (qknorm+rope、attention、sigmoid ゲート) と PTX 再生成が前提で、
このコンテナでは実装も検証もできない。**本命はこれ**で、(a) はその暫定策。

### (c) カーネルのタイル拡大 (8×8 → 16×8 など) — 見送り
FMA/LDS 比は 4.0 (現状) → 5.33 に上がるが、acc 128 本 + av[16] + bv[8] で 152 本になり
spill の危険が高い (P52/s75 で int8 側は 128 acc まで詰めた実績はある)。nvcc が無いので
spill の有無を確かめられない。かつ GEMM は他の位相と同じ 30 TFLOP/s なので、
先に (b) で D2H を消すのが順序。

### (d) q/k/v を 1 発の GEMM に連結 — 見送り
`layer.qkv_fused` (行連結 [q|k|v]) は既に登録済み (`qwen35.rs:3088-3091`) なので
n=14,336 の 1 発にできるが、出力が行内 [q|k|v] になり、3 つの連続領域への D2H が
strided (2D) になる。`cuMemcpy2DAsync` の追加が必要で、FLOP もコピーバイトも不変。
減るのは launch 2 発 (≈20 µs) だけ。

## 5. CPU テスト

| コマンド | 結果 | 時間 |
|---|---|---|
| `cargo test -p denno-engine --lib` | 293 passed / 0 failed / 7 ignored | 1.03 s |
| `cargo test --release --features cuda --lib` | 345 passed / 0 failed / 42 ignored (+ 37 / + 5 の別バイナリ) | 0.85 s |
| `cargo test --release --features cuda --test kernel_invariants` | 23 passed / 0 failed | 0.01 s |
| `cargo build --release` | 成功 | 26.5 s |

足したテスト (**緩和はしていない**。既存 23 本はすべて不変で緑):

- `cuda::p53_qk_overlap_tests::pf_qk_overlap_flag_defaults_on` — フラグは既定 ON、
  明示的な `0`/`off`/`false` だけが OFF。
- `cuda::p53_qk_overlap_tests::pf_qk_d2h_bytes_match_27b_shape` — 27B の
  q/k/v の D2H バイト数 (690,241,536 / 57,520,128 / 57,520,128、合計 805,281,792) と
  q+k = 747,761,664 (P47 のプール) を固定。14.3 GB/s で q が 47〜50 ms/層に入ること。
- `kernel_invariants::prefill_full_qk_q_copy_overlaps_kv_gemm` — 参加点が
  q GEMM の後・k GEMM の前にあること / 参加点が 1 箇所 / 失敗時は
  `unwrap_or(null_mut())` で既定ストリームへ落ちること / **q の D2H だけ**が
  コピーストリームで k/v は既定ストリーム / D2H 3 発・`ctx.sync()` 1 回のまま /
  `sync` が `cuCtxSynchronize` (全ストリーム join) / ストリームが
  `CU_STREAM_NON_BLOCKING` / イベントを既定ストリームに打つ / h2 の一式を流用しない /
  `GPU_BROKEN` を立てない。

## 6. メインが実機でやる必要があること

1. **PTX 再生成: 不要。** `.cu` を触っていない (`git diff --stat` で確認できる)。
2. **`GPU_VERIFY.sh`**: アンカー 3/3 MATCH と wide 8 問 × 150 token (temp 0) の md5 が
   `e36644365c6746b5ee1a06720d5ad4a1` (P52 時点) と同一か。数値経路は動いていないはず。
3. **cold A/B (2 腕 × 3 run、1 run 目は重みアップロードを含むので破棄)**:
   - 腕 A: 既定 (`DENNO_PF_QK_OVERLAP` 未設定 = ON)
   - 腕 B: `DENNO_PF_QK_OVERLAP=0` (従来の既定ストリーム直列)
   見る値: 65k wall と `[PF_DBG]` の `full_qk` バケット。
   **期待 (推定)**: `full_qk` 2.11 s → 1.95 s 前後 (−0.16 s)。−0.29 s まで行けば上限。
   差が 0 なら、コピーエンジンが既に飽和していて隠せる余地が無いということ
   (その場合はこの変更を revert して (b) に進むのが正しい)。
4. 悪化したら revert する (AGENTS.md の規律)。判定は wall の中央値で、±0.3 s の
   ビルド間ばらつき (P52 の 4 腕) より大きい差が出たときだけ採用。

## 7. 次にやるなら

`full_qk` の残り 0.90 s は PCIe 上の D2H そのもの。順序は
(b) attention のデバイス化 (−0.9 s、`.cu`) > (a) q のパイプライン化 (−0.6 s、検査の置換が前提)
> (c) カーネルのタイル拡大 (GEMM は既に他位相と同じ 30 TFLOP/s なので最後)。
`full_oproj` は触る理由が無い (§2)。
