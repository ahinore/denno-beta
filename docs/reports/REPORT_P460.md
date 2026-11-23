ADOPT: yes t460c

# REPORT P460 — 既定モデル decode: flash decoding の分割を長さ依存にした (auto) と
# combine + 出力ゲートの融合カーネル。14k 後の full 段が **-7.9%** (3 run で再現)、
# decode 合計は 14k で -1.0 to -1.5% (task の 2% 目標は未達)

branch `br460` = 開始時の `int-auto` **f145a13a6** (= merge: br448)。br456 は
merge 済みだったので取り込み不要。終端で int-auto を fetch し直したら
**6f2237c69 (merge: br457)** に動いていたので merge した (**da200e789**、
build_ptx.sh の targets_all と kernel_invariants の末尾 assert 6 箇所 +
task 457 のテスト追加で衝突 → 両側を残し、targets_all の末尾は
`... attn_dec2 pf_elem attn_dec4)`)。merge は lab の後なので、lab が見たのは
**055357732** (このタスクの 3/n、挙動は merge 後も同じ: 差分は build_ptx.sh と
テストだけ) と **c42e62d8f** / **90fa96071**。

数値はすべて GPU ラボの実測 (`SCENARIO=depth8k`、`PROFILE=1`、モデルは
`flash-next-UD-IQ3_XXS`、ベースライン COND = int-auto の既定 +
`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0
DENNO_GPU_WEIGHT_MB=12000`、先頭に捨て COND):
**t460a = c42e62d8f** (PTX=attn_dec4)、**t460b = 90fa96071** (commit 済み PTX)、
**t460c = 055357732** (既定を auto + 融合に反転した最終確認)。
`RESULT.md` 2 行目は 3 run とも `model=flash-next-UD-IQ3_XXS`、t460a だけ
`ptx=attn_dec4`、`load_before` は 5.23 / 1.38 / 1.64。

## 0. 結論 (7 行)

1. **採る (段の規則で)。** flash decoding の 1 分割 key 数を文脈長で変える
   `DENNO_ATTN_FLASHDEC_CHUNK=auto` (規則: `clamp(ceil(len/110), 64, 128)`) を
   **既定**にした。14k (14,141 keys) では chunk 64 -> 128 になり、分割数
   `n_split` が 221 -> 111、block 数 442 -> 222 になって combine の読みが半減する。
2. **14k 後の full 段 ([q4ledger]、ms/token) は 1.76-1.79 -> 1.63-1.68 =
   -7.9% (中央値)**。3 run・のべ 13 サンプルで再現した (下の表)。同じ段を
   `[dec-kprof-avg]` で見ると combine 16.2 + gate_mul 3.8 = 20.0 us/層が
   融合カーネル `dno_attn_flashdec_cbg` **10.8 us/層**になり、full 段の
   カーネル合計は 138.7 -> 128.5 us/層 (-10.2)。
3. **新しい .cu を 1 本足した**: `kernels_attn_dec4.cu` の
   `dno_attn_flashdec_cbg` = `dno_attn_flashdec_combine` (部分和の縮約) +
   `dno_dec_gate_mul` (出力ゲート) を 1 起動にしたもの。縮約順 (fmaxf / exp2f /
   fmaf) もゲートの演算列 (`dno_expf(-g)` -> `1/(1+e)` -> 乗算) も既存と同一で、
   **値はビット一致** (実測 md5 は 3 run x 全 COND・全コンテキストで SAME)。
4. **decode 合計は目標に届かない**: 14k 後の decode は台帳合計で 12.41-12.51 ->
   12.24-12.35 ms/token (同じ run の base と比べて **-1.0 to -1.5%**、差は
   ちょうど full 段の -0.13 ms)。短い decode と 2k 後は 1% 以内で不変。
   task の Target「短い decode か 14k 後で 2%」は**未達**。
5. **採否は task の Why にある運用規則で決めた**: 「段で 2 ラウンド以上 2% 以上
   勝ち、他が 1% 以上悪化せず、text が健全なら目標未満でも既定 on」。今回は
   段で -7.9% (3 run)、他は全部 1% 以内、errors 0、md5 SAME、text 健全 -> **yes**。
6. **試して駄目だったもの (このタスクで測った)**: `DENNO_GDN_DEC_GEMV=dn6r1`
   (GDN の qkv/out を 1 行 1 組に) は ssm 段 2.84 vs base 2.84-2.96 で利得なし。
   `DENNO_FULL_DEC_FAST=dn6r1` (full の qkv/wo) も full 段 1.63 で combo と
   同一 = 利得なし。どちらも既定は変えていない。
7. **正直な制限**: 14k は depth8k の仕様で **round 1 にしか出ない**ので、1 つの
   lab run では 14k のサンプルは COND あたり 1 本。3 run を重ねて判定した。
   7k-13k の文脈 (chunk 65-128 になる帯) は lab に要求が無く**未実測**。
   規則は n_split >= 110 を保つので 2k (chunk 64 のまま) と同じ失敗はしないと
   考えるが、実測ではない。

## 1. 前提: int-auto f145a13a6 の段プロファイル (t456a = REPORT_P456 §1)

`[q4ledger]` (ms/token):

| stage | 短い decode | 14k 後 (c8000_turn2) |
| --- | ---: | ---: |
| mix | 2.79 | 2.92 |
| router | 0.17 | 0.18 |
| shared | 0.64 | 0.66 |
| full | 1.19 | 1.77 |
| ssm | 2.75 | 2.78 |
| experts | 2.89 | 3.51 |
| head | 0.60 | 0.61 |
| 合計 | 11.03 | 12.43 |

`[dec-kprof-avg]` の 14k full band (us/層): `flashdec_t` 66.5 + combine 16.2 +
qknorm_rope 5.0 + gate_mul 3.8 + dn6 23.6 x 2 = 138.7、`step fd_sync` 129.0
(ホストは層ごとに GPU を待つ = full 段はほぼ GPU 時間そのもの)。

## 2. なぜこれを選んだか (Do 1 / Do 2)

- 台帳の大きい順は experts 2.89 / mix 2.79 / ssm 2.75 / full 1.19。
  experts・mix・ssm は task 388/406/418/422/431/437/392/441 で各 2-4 回手が入り、
  カーネル単体も帯域か ALU の天井 (qkv dn6 1078 GB/s、experts 432-485 GB/s、
  mix dn7 370 / up4f 424 GB/s) に近い。**14k で伸びる唯一の段が full**
  (1.19 -> 1.77、+49%) で、前タスク P456 の §9-2 が「次の一手は分割数の
  長さ依存化」と明記していた (chunk 128 で 14k full -6.6%、2k +16% という
  両端の実測があり、規則だけが未実装だった)。
- P456 の第 3 版 (prefetch) が「ロードの形は律速でない」ことを示したので、
  同じ場所のロードをいじるのはやめ、**(a) 分割数**と**(b) 起動回数 (combine と
  gate の融合)** に絞った。どちらも task が挙げている「小さいカーネルの起動
  オーバーヘッド」に効く。

## 3. 変えたこと

### 3.1 分割の長さ依存化 (`cuda.rs`)

- 純関数 `attn_fdec_chunk_is_auto` / `q4_fdec_chunk_auto`:
  `chunk = clamp(ceil(len / 110), 64, 128)`、`110` は block 数 `2*n_split` が
  SM 170 を余裕で超える最小の分割数。
  14,018 -> 128 (n_split 110。t456c で測った最良と同じ)、3,965 / 2,000 -> 64
  (現状不変)、7,041 -> 65、上限 128 (40,000 でも)。
- `attn_flashdec_chunk_for(len)` を decode の 2 経路 (`attn_launch` の
  デバイス連鎖と `q4_attn_decode_fdec`) で使う。**prefill はこの関数を通らない**
  (`attn_launch` は decode 専用)。
- 既定 = auto (未設定・空)。切り戻しは `DENNO_ATTN_FLASHDEC_CHUNK=64`。

### 3.2 新カーネル `kernels_attn_dec4.cu` (`dno_attn_flashdec_cbg`)

- part を split 昇順に畳むループ (fmaxf / exp2f / fmaf の順) は
  `dno_attn_flashdec_combine` と 1 命令も変えず、最後に
  `dno_expf(-g)` -> `__fdiv_rn(1, __fadd_rn(1, e))` -> `__fmul_rn` を足すだけ。
  gate に nullptr を渡すと combine 単体と同じ値。引数 6 本
  (part / att / gate / group / dh / n_split)、grid = (group, n_kv) /
  block = (dh) で combine と同じ。
- ホスト: 新 env `DENNO_ATTN_FLASHDEC_CBG` (**既定 on**、明示的な偽値だけ off)。
  `attn_launch` はゲートのポインタを受けて融合版を選び、`(ptr, gated)` を返す。
  呼び出し側 (`fwd_full_dec_dev`) は `gated` のとき出力ゲートを**飛ばす**
  (二重掛けしない)。PTX が無い・ABI が違う・形が合わないときは従来の
  2 起動へ落ちる (fail-closed)。
- `build_ptx.sh` に target `attn_dec4` を末尾に追加 (-fmad=false)。PTX は
  lab t460a の nvcc が生成したもの (`done/t460a/ptx/kernels_attn_dec4.ptx`、
  compute_75、6 引数、`ld.local`/`st.local` 0 本) を commit した。
- 発火の証拠: `[attn-fdec4] PTX ok: dno_attn_flashdec_cbg nargs=6`、
  `[attn-fdec4] gate cbg_on=true/false gate_ok=...`、
  `[q4attn] ... | cbg calls=N` (base COND は 0、融合 COND は 1152-1536)。
- 計数 `q4attn_kprof::CBG_CALLS` を `[q4attn]` 行に足した。

## 4. 数値の扱い

- 融合版は**値がビット一致** (縮約順もゲートの演算列も変えない)。実測でも md5 は
  t460a/t460b/t460c の全 COND・全コンテキストで SAME
  (`edaa84ad / 1938059f / b42d79d2 / 12fc5c36 / 672fec07 / 98e879d2`)。
- したがって差分は「速いか遅いか」だけ。判定は decode の中央値 (0.1 ms 刻み)
  ではなく、**同一 run の `[q4ledger]` の段**と `[dec-kprof-avg]` のカーネル単体で
  読んだ (t460c の decode 中央値は x0/new/old/new2/old2 = 11.3/11.4/11.3/11.4/11.4 で
  何も読めない。experts 段の ±0.6 ms の揺れの方が大きい)。
- 14k は run 1 回につき COND あたり 1 サンプルしかない (depth8k の仕様)。
  t456c/t460a/t460b/t460c の 4 run を重ねて判定した。

## 5. lab の結果

### 5.1 14k 後の full 段 ([q4ledger]、ms/token、round 1 のみ)

| run | 既定 (chunk 64) | 新 (auto / auto+融合) |
| --- | ---: | ---: |
| t456c (P456、int-auto) | base 1.77 / base2 1.77 | ck (chunk 128 だけ) **1.65** |
| t460a (c42e62d8f) | x0 1.76 / base 1.79 / base2 1.77 / g1 1.79 | auto 1.68 / **cbg 1.65** |
| t460b (90fa96071) | x0 1.77 / base 1.76 / base2 1.77 | **combo 1.63** / kp 1.63 / fw1 1.63 |
| t460c (055357732) | old 1.76 / old2 1.76 | x0 1.63 / **new 1.63** / **new2 1.63** |

既定の中央値 1.77 (n=15) に対し新は **1.63** (n=9、最悪 1.68) = **-7.9%**。
14k の decode 合計 (同じ run の台帳の和) は t460b で base 12.41 / base2 12.51 ->
combo 12.24、t460c で old 12.49 / old2 12.37 -> new2 12.35 (new 12.89 は
experts 段が 4.13 に跳ねた外れ値)。**合計では -1.0 to -1.5%** で、段の差
(-0.13 ms) と一致する。

### 5.2 14k のカーネル内訳 (`[dec-kprof-avg]`、us/層、t460b の kp COND)

| カーネル | 既定 (t456a) | 新 (t460b kp) |
| --- | ---: | ---: |
| `dno_attn_flashdec_t` | 66.5 (grid 221x2, block 384) | 65.5 (grid **111**x2, block 384) |
| combine / cbg | 16.2 (`..._combine` 12x2) | **10.8** (`..._cbg` 12x2) |
| `dno_dec_gate_mul` | 3.8 | — (融合済み) |
| `dno_dec_qknorm_rope` | 5.1 | 5.0 |
| `dno_mix_gemv_q8_0_dn6` x2 | 23.5 x 2 | 23.6 x 2 |
| 合計 | 138.7 | **128.5** |
| `step fd_sync` (ホストの待ち) | 129.0 | 122.3 |

kp COND は kprof のイベント計測のぶん full 段の台帳値が 0.16 ms ほど大きく出る
(1.81) ので、絶対値はこの表の中だけで比べること。

### 5.3 短い decode / 2k / その他の段 (t460b、round 1 と 2)

| 段 | 既定 (base / base2) | 新 (combo) |
| --- | ---: | ---: |
| 短い full (r1/r2) | 1.19/1.20, 1.18/1.19 | 1.19/1.19 |
| 2k turn2 の full | 1.19, 1.19 | 1.18, 1.17 |
| ssm | 2.73-2.78 | 2.74-2.79 |
| mix | 2.77-2.81 | 2.76-2.87 |
| experts | 2.85-2.90 | 2.86-2.93 |
| shared / head / router | 0.63 / 0.60 / 0.17 | 同左 |
| prefill (14018 cold) | 5.52-5.53 s | 5.58 s |

どれも 1% 以内。prefill は `attn_launch` を通らないので構造上も不変。

### 5.4 VERDICT 行 (t460c = 既定を反転した run)

```text
VERDICT new vs x0 OK decode +0.3% prefill -1.7% md5 SAME errors 0
VERDICT old vs x0 OK decode +0.0% prefill -0.9% md5 SAME errors 0
VERDICT new2 vs x0 OK decode +0.7% prefill -1.7% md5 SAME errors 0
VERDICT old2 vs x0 OK decode +0.0% prefill -0.9% md5 SAME errors 0
VERDICT_DEPTH new c8000_turn2 vs x0 decode +6.1% prefill +4.0%
VERDICT_DEPTH old c8000_turn2 vs x0 decode +0.6% prefill +2.0%
VERDICT_DEPTH new2 c8000_turn2 vs x0 decode -0.6% prefill +2.0%
VERDICT_DEPTH old2 c8000_turn2 vs x0 decode -0.6% prefill +2.0%
VERDICT baseline x0 errors 0 text ok
```

x0 が新既定なので old との比較が「戻したとき」になる。`new` の +6.1% は
experts 段の外れ値 (3.49 -> 4.13) で、同じ既定の new2 は -0.6%。この計器
(0.1 ms 刻みの中央値) では 0.13 ms の差は読めないので、§5.1 の段で判定した。

t460b (両方のゲートを明示した run) の VERDICT:

```text
VERDICT base vs x0 OK decode -0.7% prefill +2.7% md5 SAME errors 0
VERDICT combo vs x0 OK decode +0.7% prefill +1.8% md5 SAME errors 0
VERDICT kp vs x0 OK decode -0.3% prefill +1.8% md5 SAME errors 0
VERDICT fw1 vs x0 OK decode -0.3% prefill +1.8% md5 SAME errors 0
VERDICT base2 vs x0 OK decode +0.3% prefill +1.8% md5 SAME errors 0
VERDICT_DEPTH combo c8000_turn2 vs x0 decode -0.6% prefill -3.8%
VERDICT_DEPTH base  c8000_turn2 vs x0 decode +1.2% prefill -3.8%
VERDICT_DEPTH base2 c8000_turn2 vs x0 decode (同 left out)
VERDICT baseline x0 errors 0 text ok
```

## 6. 出力テキスト (t460c の new COND)

- `TXT`: `# B-Trees: A Comprehensive Explanation\n\n## 1. What is a B-Tree?\n\nA **B-tree (Balanced Tree) is a self-balancing tree data structure that maintains sorted data and allows searches, sequential access, insertions, and deletions in logarithmic time.\n\n`
- `TXTB`: `Rust の「所有権（Ownership）とライフタイム（Lifetime）は、Rust を学ぶ上で最も重要であり、かつ最も難しい概念です。\n\nしかし、これらは「メモリを安全に管理するためのルール**です。`
- `TXTD` (14k 後): `ご提示いただいた文書は、LLM推論エンジン（`denno-engine）における **Prefill（事前計算）フェーズの性能最適化の歴史を詳細に記録した技術レポートです。`

x0 (既定) と new/new2 の TXT/TXTB/TXTD は同一、md5 も全コンテキストで SAME。

## 7. 採否の判定 (task の規則)

- Target「短い decode **か** 14k 後で 2%」: 短い decode は不変、14k 後は
  **-1.0 to -1.5%** (台帳合計)。=> **未達**。
- Why の運用規則「段で 2 ラウンド以上 2% 以上勝ち、他が 1% 以上悪化しない、
  text 健全 -> 目標未満でも既定 on」: 14k の **full 段 -7.9%** を 3 run で
  再現 (t456c/t460a/t460b/t460c)、他の段と prefill は 1% 以内、errors 0、
  md5 SAME、text 健全 (…6)。=> **満たす**。
- よって **ADOPT: yes (t460a/t460b/t460c)**。既定は auto + 融合 on。
  切り戻しは `DENNO_ATTN_FLASHDEC_CHUNK=64` と `DENNO_ATTN_FLASHDEC_CBG=0`。

## 8. テスト

関門 (cuda feature あり。int-auto は 6f2237c69 を merge 済み):

```text
CARGO_TARGET_DIR=/tmp/t460 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1326 passed; 0 failed; 103 ignored   (task 460 で 3 本追加。merge 前は 1320)
同 --test kernel_invariants
  -> 133 passed; 0 failed                  (task 460 で 1 本追加。merge 前は 130)
CARGO_TARGET_DIR=/tmp/t460nc cargo build --workspace --release   -> ok (cuda feature 無し)
同 cargo test --workspace --release -- --test-threads=1
  -> 全 crate ok / 失敗 0 (engine lib 962 passed、kernel_invariants 129、
     server 系も 0 failed。既知の flake slot_leak も今回は緑)
```

足したテスト:
- `cuda::task448_tests::task460_auto_chunk_rule` — `auto` の真理値表、長さ ->
  chunk の規則 (14,018 -> 128 / 2,000 -> 64 / 7,040 -> 64 / 7,041 -> 65 /
  上限 128)、`q4_fdec_split` と組み合わせて 14k は n_split 110、2k は 32 のまま。
- `task460_cbg_gate_truth_table` — 融合の門は既定 on、明示的な偽値だけ off。
- `task460_cbg_counters_exist` — `CBG_CALLS` と `FCBG_NARGS == 6`。
- `kernel_invariants::task460_flashdec_combine_gate_kernel_and_wiring` —
  `.cu` の entry / 6 引数 / **縮約の順が combine と同一** / **ゲートの演算列が
  `dno_dec_gate_mul` と同一** / **dno_expf の写しの定数が dec_dev と一致** /
  ホストの配線 (門・起動・二重掛けしないこと・計数・2 経路の chunk) /
  build_ptx target / コミット済み PTX の 6 引数。

`.cu` は g++ の CUDA スタブで `-fsyntax-only` を通した (dec2 も同じハーネスで
通ることを確認)。PTX は lab の nvcc が生成したものをそのまま commit した。

## 9. GPU で確認してほしいこと / 次への申し送り

1. 既定は auto + 融合 on。切り戻しは `DENNO_ATTN_FLASHDEC_CHUNK=64` /
   `DENNO_ATTN_FLASHDEC_CBG=0` (どちらも既存 env なので lab に届く)。
2. **7k-13k の文脈が未実測**。自動規則はその帯で chunk 65-128 (n_split 110) を
   使う。実測して悪ければ `DENNO_ATTN_FLASHDEC_CHUNK=64` で戻すか、
   `q4_fdec_chunk_auto` の下限を上げる。
3. 次の一手 (推定): 14k の full 段の残りは `flashdec_t` 65.5 us/層 (55 MB を
   ~1 TB/s) と `dn6` 47 us/層 (wo は 474 GB/s = qkv の 44%)。**full 層の wo の
   形 (k 4096 / n 2560) が GDN の out (k 6144 / n 2560、647 GB/s) と同じく
   低帯域**で、rows=1 では動かなかった (fw1)。ブロック数を増やす別案
   (1 ブロック 1 行 x 2 レーン並列など) が候補。combine は融合後 10.8 us/層まで
   下がったので、もう主役ではない。
4. `DENNO_GDN_DEC_GEMV=dn6r1` は ssm 段で利得なし (2.84 vs 2.84-2.96)。rows は
   2 のままが良い。
