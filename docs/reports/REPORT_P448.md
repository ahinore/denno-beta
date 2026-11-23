ADOPT: yes t448e

# REPORT P448 — 既定モデル decode: flash decoding の K/V を shared のタイルへ協調ロードした
# (14k 後の full 段 -10.7% (1.98 -> 1.76 ms/tok)、カーネル単体 86.3 -> 66.4 us/層 = 637 -> 828 GB/s)

branch `br448` = 開始時の `int-auto` **2506204e8** (= merge: br437)。**br441 は merge して
いない** (task 441 の GDN scan3 は `ADOPT: no` で既定 off、本タスクの対象 (full attention)
とは無関係なので不要。br441 の上で作りかけた差分は `git cherry-pick` で br448 へ移し、
br441 は元の `536b585e9` に戻した)。終端で int-auto を fetch し直したが
**2506204e8 のまま**だったので merge は無し (`git merge-base --is-ancestor` で確認)。
数値はすべて GPU ラボの実測 (`SCENARIO=depth8k`、`PROFILE=1`、
`model=flash-next-UD-IQ3_XXS`)。

## 0. 結論 (7 行)

1. **採る。** decode の full attention (flash decoding) の K/V を、block の全スレッドで
   **shared メモリのタイルへ 1 行 1 回だけ**協調ロードする第 2 版 (`dno_attn_flashdec_t`)
   を書いた。1 key あたりの演算列は v1 と 1 命令も変えていないので **md5 は全 COND・
   全コンテキストで SAME**。
2. **カーネル単体 (t448e、14k、1 層 1 回)**: `dno_attn_flashdec_t` **66.4 us**
   (55 MB を **828 GB/s**) ← v1 `dno_attn_flashdec` **86.3 us** (637 GB/s) = **-23%**。
   combine は 16.1 us で不変。
3. **段 (同一 run・同一計器、c8000_turn2、ms/token)**: **full 1.98 -> 1.76 (-11.1%)**。
   他は mix 2.95->2.94 / ssm 2.80->2.78 / experts 3.56->3.55 / shared 0.68->0.66 /
   head 0.61->0.62 (0.01 ms = 台帳の分解能) / router 0.18->0.18。
4. **壁 (decode の中央値)**: 14k 後 (`c8000_turn2`) が v1 **13.2 / 13.4** に対し
   tiled **13.0 / 12.7 / 12.8** (3 COND、平均 **-3.5%**、最良の対で -4.5%)。
   短い decode は 11.3-11.4 で不変、14k cold prefill 5.55-5.72 s で不変。
   目標「14k 後 2%」は満たす (最悪の対 base vs v1 でも -1.5%、段は -10.7%)。
5. **既定 on**。切り戻しは `DENNO_ATTN_FLASHDEC_TILE=0` (v1 のカーネルへ)。
   行数は `DENNO_ATTN_FLASHDEC_TILE=4..64` (既定 16)。
6. **なぜこれを選んだか**: t441a の段プロファイルで decode の大きい 3 段 (experts 2.89 /
   mix 2.80 / ssm 2.74) はいずれも task 337/388/406/418/422/431/437 と 184/392/441 が
   詰めており、14k で伸びるのは **full attention (+0.75 ms/token)** だけだった。
   `REPORT_P410` §0-7 が「14k の decode を本気で削るには flashdec のカーネル (分割の
   作り直し) が要る。P196/P201 が未着手と書いた玉」と明記しており、task 448 は .cu の
   新カーネルを許可している。v1 は 1 block = (split, kv head)・1 warp = 1 q head で、
   **block 内の 12 warp が同じ K/V 行を 12 回読む** (L1 任せ) + 1 warp の未完了ロードが
   2 key ぶんしかない (2 段パイプライン) ので、帯域の 37% で止まっていた。
7. **正直な制限**: t448e の `load_before` は **6.67** (高負荷) で、decode の中央値は
   0.1 ms 刻みなので揺れる (13.0 / 12.7 / 12.8)。採否は同一 run の**段の台帳**と
   **カーネル単体の時間** (どちらも 3-4 桁の差) で読んだ。full 段の残りは
   combine 16.1 us (24 block しか立たない) で、次に削るならそこ。

## 1. 前提: t441a の段 (int-auto 2506204e8、14k、`[dec-kprof-avg]`)

| band | op | 1 回あたり | 形 |
| --- | --- | ---: | --- |
| full | `dno_attn_flashdec` | **86.3 us** | grid 221x2, block 384 |
| full | `dno_attn_flashdec_combine` | 16.2 us | grid 12x2, block 256 |
| full | `dno_mix_gemv_q8_0_dn6` (qkv / wo) | 23.6 x 2 | grid 1280, block 192 |
| full | `dno_dec_qknorm_rope` / `dno_dec_gate_mul` | 5.0 / 3.8 | |
| full | `op sync` (fd_sync の待ち) | host 149.5 us/層 | |

KV は 1 層 55 MB (`len=14019 kv_dim=512 dh=256 group=12` = 2 kv head x 256 x K/V x f32)。
86.3 us は **637 GB/s** で、同じ run の qkv/wo (dn6) は 1.7 TB/s なので **2.7 倍の伸びしろ**。

## 2. 変えたこと

### 2.1 新カーネル `kernels_attn_dec2.cu` (= `dno_attn_flashdec_t`、11 引数)

- v1 との違いは **K/V の読み方だけ**: タイル (`tile` 行、既定 16) を block の全スレッドで
  float4 協調ロードして shared に置き、計算は shared から読む。同期はタイルごとに 2 回。
- **1 key あたりの演算列は v1 と同一** (`fmaf` の順、`shfl_xor_sync` 16..1、online
  softmax の分岐、`acc = acc*corr + p*v`)。分割 (`q4_fdec_split`) も部分和の置き場も
  combine も共通なので、**値は v1 とビット一致**する (実測 md5 SAME)。
- 動的 shared は `2 * tile * dh * 4 B` (dh 256 / tile 16 で 32 KiB、上限 48 KiB)。
  `fdt_shape_ok` が「dh が 4 の倍数・dh <= 512・smem <= 48 KiB」を門番にし、外れたら
  v1 へ落ちる。PTX の ABI は 11 引数を `ptx_entry_params` でモジュールロード前に検査。

### 2.2 ホスト側 (`cuda.rs`) — **2 箇所**に配線した (ここが今回の落とし穴)

- `q4_attn_decode` (decode の CPU 側 dispatch。短い文脈や非デバイス連鎖用)。
- **`CudaCtx::attn_launch`** (task 172/179 の**デバイス連鎖**。実サーバの full 層 decode は
  こちらを通る)。t448a-d はここを忘れて v1 だけが走り、差が 0 だった (§4)。
- ゲートは **既存 env の新しい値** `DENNO_ATTN_FLASHDEC_TILE`
  (`0`/`off`/`false`/`no` = v1、`4..=64` = その行数でタイル版、未設定 = 既定 on = 16)。
  `DENNO_ATTN_FLASHDEC2` (新名) と `DENNO_Q4_GPU_ATTN_KERNEL=tile` でも on にできる。
- 起動時に 1 行 `[attn-fdec] kernel=... fdt_on=... tile=... fdtv=3 (task 448)` を出し、
  COND の env が届いたか・ビルドが新しいかをログだけで判別できるようにした。
- 計数: `[q4attn]` の行に `fdt calls=... fallback=... tile=...` を足した。
- `build_ptx.sh` に target `attn_dec2` を**末尾**に足した (targets_all の末尾 assert 6 箇所を
  新しい末尾へ)。PTX は lab の nvcc で生成した `done/t448a/ptx/kernels_attn_dec2.ptx` を
  commit した (11 引数を検査、`st.shared.v4` と `.extern .shared` を含むことを確認)。

## 3. 数値の扱い

- **値はビット一致**。1 key の演算列を 1 命令も変えていない (テスト
  `task448_flashdec_tile_kernel_and_wiring` が v1 の `DNO_FD_KEY` とタイル版の key ループを
  識別子の揺れだけ落として**行単位で突き合わせる**)。実測でも md5 は 6 COND x
  6 コンテキストすべて SAME (`edaa84ad / 1938059f / b42d79d2 / 12fc5c36 / 672fec07 / 98e879d2`)。
- 決定性・固定分割・原子操作なしは v1 と同じ。行独立性 (q head) も同じ。

## 4. lab の結果

すべて `SCENARIO=depth8k PROFILE=1`、ベースライン COND は
`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0
DENNO_GPU_WEIGHT_MB=12000`。最初の COND (x0) は捨て、最後に baseline の繰り返し。

### 4.1 無効だった 4 run (すべて「新しいカーネルが 1 回も走っていない」)

| run | SHA | 何が起きたか |
| --- | --- | --- |
| t448a | 75d7771b4 | PTX=attn_dec2 は生成できたが、門を `DENNO_Q4_GPU_ATTN_KERNEL=tile` にしていた。**この env は届かず** v1 が走った (`fdt calls=0`、`fdec calls=1248`) |
| t448b | a6a63fefd | 門を新名 `DENNO_ATTN_FLASHDEC2=1` にしたが、**これも届かず** v1 のまま。同じ COND の `DENNO_ATTN_FLASHDEC_TILE=24` は届いた (tile=24 がログに出る) |
| t448c | 51e9ef229 | 門を `DENNO_ATTN_FLASHDEC_TILE` に移した (tile=16 は届いた) が、まだ `fdt calls=0`。原因: **decode の実経路は `q4_attn_decode` ではなくデバイス連鎖の `attn_launch`** で、そちらに門が無かった |
| t448d | b9f172a98 | 既定 on にした (`fdt_on=true` がログに出る) が同じ理由で `fdt calls=0`。ここで `attn_launch` の存在に気づいた |

lab の env について分かったこと (次に測る人向け): **名前に数字を含む env は COND から
落ちる回がある** (`DENNO_Q4_GPU_ATTN_KERNEL`、`DENNO_ATTN_FLASHDEC2` は届かず、
`DENNO_ATTN_FLASHDEC_TILE` は届いた)。既存 env の新しい値も、名前次第では届かない。
起動 1 行のログ (`[attn-fdec] ... fdtv=3`) を入れてからは、届いたかどうかが 1 見で分かる。

### 4.2 t448e = `4ce2ca675` (本採用、`ptx=none` = commit 済み PTX、load 1.09 -> 6.67)

`fdt calls=1152 / 2688 / 3936 / 5472`、`[attn-fdec2] PTX ok: ... nargs=11 tile=16`、
`[attn-fdec2] len=14019 n_split=220 chunk=64 dh=256 tile=16 smem=32KiB`。
v1 の COND (`DENNO_ATTN_FLASHDEC_TILE=0`) は `fdt calls=0` で `fdec calls=1248`。

**カーネル単体 (14k、`[dec-kprof-avg]` の full band、1 層 1 回)**

| run (COND) | kernel | gpu us | GB/s (55 MB) |
| --- | --- | ---: | ---: |
| kv1 (v1) | `dno_attn_flashdec` | **86.3** | 637 |
| kt (tile 16) | `dno_attn_flashdec_t` | **66.4** | **828** |
| 両方 | `dno_attn_flashdec_combine` | 16.1 | - |

**段の台帳 (c8000_turn2、ms/token、同じ計器の 2 COND)**

| stage | v1 (kv1) | tiled (kt) | 差 |
| --- | ---: | ---: | ---: |
| **full** | **1.98** | **1.76** | **-11.1%** |
| mix | 2.95 | 2.94 | -0.3% |
| ssm | 2.80 | 2.78 | -0.7% |
| experts | 3.56 | 3.55 | -0.3% |
| shared | 0.68 | 0.66 | -2.9% |
| head | 0.61 | 0.62 | +1.6% (0.01 ms) |
| router | 0.18 | 0.18 | 0 |

計器を付けない COND でも同じ向き: full (c8000_turn2) が **v1 1.97** に対し
**base 1.76 / base2 1.77** (-10.7%)。c8000_cold は v1 1.99 / kv1 2.16 に対し
base 1.77 / base2 1.77 / kt 1.93。c2000_turn2 は v1 1.21 / kv1 1.21 に対し
base 1.19 / base2 1.19 / kt 1.19 (-1.7%、文脈が短いので KV が少ない)。
短い prompt (28) は 1.36-1.38 で不変。

**decode の中央値 (ms/tok、RESULT.md)**

| cond | short | c2000_turn2 | c8000_cold | c8000_turn2 |
| --- | ---: | ---: | ---: | ---: |
| x0 (捨て) | 11.3 | 16.3 | 15.2 | 15.6 |
| base (tiled) | 11.3 | 12.7 | 13.7 | **13.0** |
| v1 | 11.3 | 12.5 | 13.8 | **13.2** |
| kv1 (v1+kprof) | 11.3 | 12.5 | 14.7 | **13.4** |
| kt (tiled+kprof) | 11.3 | 12.5 | 14.2 | **12.8** |
| base2 (tiled) | 11.4 | 12.4 | 13.4 | **12.7** |

```text
VERDICT base vs x0 OK decode +0.3% prefill -0.8% md5 SAME errors 0
VERDICT v1 vs x0 OK decode +0.3% prefill -2.5% md5 SAME errors 0
VERDICT kv1 vs x0 OK decode +0.3% prefill -2.5% md5 SAME errors 0
VERDICT kt vs x0 OK decode +0.3% prefill -2.5% md5 SAME errors 0
VERDICT base2 vs x0 OK decode +0.7% prefill -3.3% md5 SAME errors 0
VERDICT_DEPTH base c8000_turn2 vs x0 decode -16.5% prefill -8.8%
VERDICT_DEPTH v1 c8000_turn2 vs x0 decode -15.5% prefill -8.8%
VERDICT_DEPTH kv1 c8000_turn2 vs x0 decode -14.5% prefill -7.0%
VERDICT_DEPTH kt c8000_turn2 vs x0 decode -18.0% prefill -8.8%
VERDICT_DEPTH base2 c8000_turn2 vs x0 decode -18.5% prefill -8.8%
VERDICT baseline x0 errors 0 text ok
```

(x0 は最初の COND で page cache が冷たい。14k cold prefill は全 COND 5.55-5.72 s で
有意差なし。prefill は機構的に無関係 = decode のカーネルだけ。)

## 5. 出力テキスト (t448e、base COND)

- `TXT`: `# B-Trees: A Comprehensive Explanation\n\n## 1. What is a B-Tree?\n\nA **B-tree (Balanced Tree) is a self-balancing tree data structure that maintains sorted data and allows searches, sequential access, insertions, and deletions in logarithmic time.\n\n`
- `TXTB`: `Rust の「所有権（Ownership）とライフタイム（Lifetime）は、Rust を学ぶ上で最も重要であり、かつ最も難しい概念です。\n\nしかし、これらは「メモリを安全に管理するためのルール**です。`
- `TXTD` (14k 後): `ご提示いただいた文書は、LLM推論エンジン（`denno-engine）における **Prefill（事前計算）処理の性能最適化の記録です。...`
- md5 は 6 COND すべてで 8 コンテキスト全部 SAME (v1 と同じ)。

## 6. 採否の判定 (task の規則)

- 目標「短い decode **か** 14k 後が 2% 以上」: **14k 後で満たす** (中央値 3 COND 平均
  -3.5%、段は -10.7%)。短い decode は不変 (11.3-11.4)。
- 演算子の規則「自分の段で 2% x 2 ラウンド、他は 1% 以上悪化しない」: full 段
  **-11.1% / -10.7%** (2 つの独立した COND 対、さらに c8000_cold -11%)。他段は
  mix -0.3 / ssm -0.7 / experts -0.3 / shared -2.9 / router 0 で、head だけ
  0.61 -> 0.62 (+0.01 ms = 台帳の分解能、機構的にも無関係) → 悪化と見なさない。
- errors 0、md5 全 SAME、text 健全 (§5)。 => **ADOPT: yes t448e**、既定 on のまま。

## 7. テスト

関門 (cuda feature あり。int-auto は 2506204e8 のまま):

```text
CARGO_TARGET_DIR=/tmp/t448 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1317 passed; 0 failed; 103 ignored   (task 448 で 6 本追加: 1311 -> 1317)
同 --test kernel_invariants
  -> 129 passed; 0 failed                  (task 448 で 1 本追加: 128 -> 129。
     targets_all の末尾 assert を 5 箇所 attn_dec2 へ更新した)
CARGO_TARGET_DIR=/tmp/t448nc cargo build --workspace --release   -> ok (cuda feature 無し)
同 cargo test --workspace --release -- --test-threads=1          -> 失敗 0 (test result 行すべて ok)
```

足したテスト (7 本):
- `cuda::task448_tests::task448_kernel_value_truth_table` — `DENNO_Q4_GPU_ATTN_KERNEL` の
  新しい値 (`tile` / `fdt` / `fdec2` = タイル版、`fdec1` / `fdec` = v1) と既存値の不変。
- `task448_tile_gate_counts_one_arm` — 5 値のうちタイル版を選ぶ値の**個数を数える**
  (既定 on なら 4、off なら 1) + FORCE フック + `DENNO_ATTN_FLASHDEC2` /
  `DENNO_ATTN_FLASHDEC_TILE` の真理値表。
- `task448_tile_rows_parse` — tile 行数 (既定 16、4..=64 に丸め)。
- `task448_shape_gate_and_smem` — smem = 2*tile*dh*4、48 KiB の上限、dh の門。
- `task448_tile_force_hook` — テスト用の差し替え。
- `task448_counters_exist` — `FDT_CALLS` / `FDT_FALLBACKS` の計数が動く。
- `kernel_invariants::task448_flashdec_tile_kernel_and_wiring` — `.cu` の entry / 11 引数 /
  協調ロード / 同期 2 回 / **1 key の演算列が v1 と同一** / ホストの**2 箇所**の配線 /
  build_ptx target / commit 済み PTX の 11 引数。

既知の赤 (`qwen4exp_decode spec_batch_stops_at_eos` / `qwen4exp_params` の validate 2 件) は
範囲外で触っていない。`.cu` は g++ の CUDA スタブで `-fsyntax-only` を通した
(v1 も同じハーネスで通ることを先に確認)。

## 8. GPU で確認してほしいこと / 次への申し送り

1. 既定 on。切り戻しは `DENNO_ATTN_FLASHDEC_TILE=0` (1 行で v1)。行数は同 env の 4..64。
2. 残るのは **combine 16.1 us/層** (grid 12x2 = 24 block しか立たず 338 GB/s)。
   分割方向を 2 段に割れば 2 倍は射程 (丸め順が変わるので md5 は DIFF になる)。
   14k で -0.1 ms/token (推定 -0.8%)。
3. タイル版の次の一手は**倍バッファ** (ロードと計算の重なり)。今はタイルごとに
   `__syncthreads` 2 回で、828 GB/s は帯域の 50%。推定 -10 us/層 (full 段 -5%)。
   `tile` 24 (48 KiB ちょうど) も未測定。
4. lab の COND は**名前に数字を含む env を落とす**回がある (§4.1)。新しいゲートを
   足したら起動 1 行で必ず実値を出すこと。
5. full 層の decode は **`q4_attn_decode` ではなくデバイス連鎖の `attn_launch`** を
   通る。attention のカーネルを差し替えるときは両方に配線する (片方だけだと lab で
   差が 0 になる)。
