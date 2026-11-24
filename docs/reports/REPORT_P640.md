ADOPT: yes t640d

# REPORT P640 — default model, f16 decode by hardware cvt, FP32 exp / fmad where the contract allows, PTX arch

branch `br640` = 開始時の `int-auto` **d5a97b487** (2026-11-23)。終端で `int-auto` を
fetch し直したら **ac4f7ecf3 (merge: br617)** へ動いていたので merge した (衝突なし。
br617 は prefill inject の BF16 化で cuda.rs の別領域と kernel_invariants を触るだけ)。
lab は 4 本: **t640a** (8d1ef85ce、f16 の PTX 再生成と段プロファイル)、
**t640b** (7b914a0e3、arch120 の A/B)、**t640c** (fcf302975、A/B の再試行。失敗)、
**t640d** (124d4f7e7、step 2 の速度 A/B + 最終状態の確認、SCENARIO=decode)。
`RESULT.md` 2 行目は 4 本とも `model=flash-next-UD-IQ3_XXS`。

## 0. 結論 (7 行)

1. **step 2 (f16 -> f32 のハードウェア変換) を採る。** 30 本の .cu の手書きビット展開
   (非正規化数の while ループ付き) を `__half2float` に置き換えた。NaN / inf だけは
   旧式を残したので **全 65536 パターンで 1 ビットも変わらない**。ゲートは無し (指示どおり)。
2. **md5 は 6/6 SAME** (t640a 対 変更前の同一 commit の lab `tint_d5a97b487`)。
3. **速くもなった。** t640d の同一 run 内 A/B (旧変換の PTX を `DENNO_*_PTX` で差し替え)
   で decode 9.8 -> 10.0 ms/tok (**旧が +2.0% 遅い**)、experts 段 2.92/2.87 (新) 対
   3.03/2.98 (旧) = **-3.6%**。mix は ±0.5% (ノイズ)。
4. **step 3 (DENNO_EXPF32 / DENNO_FMAD) はゲートを足さない。** FP64 の `dno_expf` を
   持つ 11 本は全部ビット一致契約で、唯一の非ビット一致 (mix up) は task 437 で既に
   `__expf` 版 (`dno_mix_upmix_q8_up4f`) が既定になっている。fmad は PTX の差分が
   `.rn` 接尾辞の有無だけ (fma 命令 0 本のまま) で効果 0。表は §4。
5. **step 4 (compute_120) は採らない。** 12 本の主要カーネルを compute_120 で作り直した
   t640b は、段の台帳が arch75 と同一 (mix 1.88 対 1.87 / ssm 2.74 対 2.76 /
   experts 2.92 対 2.91 / full 0.98 対 0.97)、decode 中央値も 9.8 対 10.0 (ノイズ)。
   既定は compute_75 に戻した。
6. **lab の道具の落とし穴**: COND の `DENNO_*_PTX` に渡すパスは **/tmp/labsrc/...**
   (`/src/...` ではない)。t640b/t640c の oldf16 は /src を指して読めず、
   v2/dn6/upmix_q8 へ落ちて 19.3 ms/tok (+96%) になっていた (A/B として無効)。
   t640d で /tmp/labsrc に直して初めて A/B が成立した。
7. ゲート: lib / kernel_invariants / cuda 無し workspace すべて緑 (§6)。

## 1. 前提: t640a = int-auto d5a97b487 の段プロファイル (depth8k、PROFILE=1)

warm short decode の `[q4ledger]` (base、a2、128 トークン):

`tok=128 total=19.46ms launches=749.0/tok syncs=63.0/tok sync_floor=5.3us`

| 段 | ms/token | launch (ms/本数) | sync (ms/本数) |
| --- | ---: | --- | --- |
| mix | **1.87** | 0.33 / 242 | 0 / 0 |
| ssm | **2.76** | 0.19 / 144 | 0 / 0 |
| experts | **2.91** | 0 (GPU 側は `[exp-iq]` で別計器) | 0 / 0 |
| full | 0.97 | 0.10 / 72 | 0.75 / 12 |
| shared | 0.45 | 0.12 / 96 | 0 / 0 |
| head | 0.46 | 0.00 / 1 | 0.43 / 1 |
| router | 0.17 | 0 / 0 | 0 / 0 |

prefill (`[q4ledger-pf]`、1 リクエスト 1315.5 ms / 28 トークン、launches 297 =
1 項目 1 起動、syncs 297、sync_floor 13.7 us):

| 段 | ms | 割合 | 主な中身 |
| --- | ---: | ---: | --- |
| experts | 451.0 | 34.3% | 全部 CPU (`gpu=0.0`) |
| ssm | 249.6 | 19.0% | launch 0.55/72・sync 55.76/72 |
| mix | 179.6 | 13.7% | 全部 CPU |
| router | 67.1 | 5.1% | CPU |
| full | 64.1 | 4.9% | launch 0.18/24・sync 15.59/24 |
| shared | 58.2 | 4.4% | launch 1.12/144・sync 20.94/144 |

**何を選んだか / なぜ**: 1 トークンあたりの起動は 749 本・sync は 63 本で、重いのは
experts (2.91) と ssm (2.76) と mix (1.87)。experts の IQ 系 GEMV は f16 のスケールを
**手書きのビット展開 (非正規化数の while ループ付き)** で f32 にしており、これが
operator の挙動レビューの step 2 そのもの。experts 段は decode の 30% を占めるので、
ここを小さくする変更が一番効く。オペレータのレビューは main 21d020a (11-19) を読んだ
もので数値が今日より古いので、上は測り直した値。

## 2. step 2: f16 -> f32 をハードウェア変換にした

変えた 30 本 (すべて `#include <cuda_fp16.h>` を足した):
`kernels_exp_iq / exp_iq_pf / exp_iq_tc / exp_iq_v2 / exp_iq_v3 / exp_k_pf / exp_k_tc /
gemm / gemm_grouped / gemm_sn / gemm_sn2 / gemm_sn3 / gemm_tc / gemm_tc2 / mix_dn5 /
mix_dn6 / mix_dn7 / mix_dn8 / mix_up4 / mix_upmix_q8 / moe / moe_batch / moe_fp16 /
moe_v2 / moe_v3 / moe_v4 / moe_v5 / moe_v5_q5k / seg_k / sh_gu`。

新しい本体 (NaN / inf だけ旧式を残す):

     if ((h & 0x7c00u) == 0x7c00u) {   // inf / NaN は旧式のまま (ペイロード保存)
         sign = (h & 0x8000u) << 16;
         return __int_as_float(sign | 0x7f800000u | ((h & 0x3ffu) << 13));
     }
     return __half2float(__ushort_as_half((unsigned short)h));

`__half2float` を通すのは **NaN 以外の全パターン** (0 / 非正規化数 / 正規化数 / inf)。
NaN を特別扱いするのは、ハードウェアの cvt が NaN を **quiet 化する**ため:
旧実装は `0x7c01 -> 0x7f802000` (f32 では signaling NaN) を返しており、
numpy/CPU の拡張 (`0x7fc00000`) と 2044 パターンで違う。モデルの重みに NaN は無いが、
「1 ビットも変えない」を保証するために旧式を残した。

### 2.1 全 65536 パターンの照合 (テスト)

- `cuda::tests::f16_cvt_exhaustive` (CPU、GPU 不要): 旧実装と新実装を 0x0000..0xffff の
  全部で比較。不一致 0。非正規化数 2046 パターン・NaN 2046 パターンを通したことを
  数えて assert している (空振り検出)。代表値 (1.0 / 最大非正規化数 0x03ff /
  最小非正規化数 0x0001 = 0x33800000) も固定。
- `cuda::tests::f16_cvt_sources_use_hardware_cvt`: 30 本が `__half2float(__ushort_as_half`
  を含み、旧式の while ループが残っていないことを確認 (写し忘れ検出)。
- GPU 側: `kernels_elem.cu` に probe カーネル `dno_f16cvt_probe(cnt, first)` を足した
  (旧式と新式を 65536 パターンで突き合わせ、不一致数と最初の不一致を書く)。
  `cuda.rs::chain_f16cvt_probe` + テスト `f16_cvt_probe_on_gpu` から回す。
  **このコンテナには GPU が無いので skip される (未実行)**。GPU のある環境で
  `cargo test --release -p denno-engine --features cuda --lib f16_cvt_probe_on_gpu`
  を 1 回走らせると cnt=0 / first=0xffffffff が出るはず (§7)。

### 2.2 数値と速度 (lab 実測)

- **md5**: t640a の 3 COND すべて `a:edaa84ad b:1938059f c2000_cold:9b28e018
  c2000_turn2:8b02d5c3 c8000_cold:672fec07 c8000_turn2:199f23b3` = 変更前の
  `tint_d5a97b487` (同じモデル・同じ env) と **6/6 一致**。
- **PTX が小さくなった**: `kernels_exp_iq_v3.ptx` 981 -> 866 行、レジスタ
  `.b32 %r<213>` -> `%r<164>`、`.pred %p<58>` -> `%p<52>`。31 本のうち 30 本で
  削除 > 追加 (gemm だけ +2631/-2440 で微増)。
- **同一 run 内 A/B** (t640d、SCENARIO=decode、旧変換の PTX 3 本を `DENNO_*_PTX` で差し替え):

| cond | decode ms/tok (median) | experts 段 | mix 段 | ssm 段 |
| --- | ---: | ---: | ---: | ---: |
| x0 (捨て) | 9.8 | 2.92 / 2.86 | 1.89 / 1.86 | 2.74 / 2.72 |
| base (新) | 9.8 | 2.92 / 2.87 | 1.88 / 1.87 | 2.74 / 2.72 |
| oldf16 (旧変換) | 10.0 | **3.03 / 2.98** | 1.89 / 1.88 | 2.79 / 2.78 |
| base2 (新) | 9.8 | 2.92 / 2.87 | 1.88 / 1.86 | 2.73 / 2.72 |

  VERDICT: `base vs x0 +0.0%` / `oldf16 vs x0 +2.0%` / `base2 vs x0 +0.0%`、md5 は
  4 COND とも SAME、errors 0、text "looks ok"。
  → 新変換は遅くない (experts 段 -3.6%、decode -2.0%)。ssm 段の +2% は機構的に
  無関係 (触っていない) なので同じ run のドリフトと読む。

## 3. step 3 の前提: カーネルごとの契約と FP64 exp

`dno_expf` (glibc expf の FP64 再現) を持つ .cu は 11 本:
`mix_upmix_q8 / mix_up4 / mix_down / mix_dev / mix_dev3 / mix_inj3 / dec_dev /
attn_dec4 / elem / pf / (elem の probe)`。契約は .cu のヘッダ記載のとおり:

| カーネル (target) | ビルド | 数値契約 | FP64 exp | DENNO_EXPF32 / FMAD の利得 |
| --- | --- | --- | --- | --- |
| exp_iq / exp_iq_v2 / exp_iq_v3 / exp_iq_pf / exp_iq_tc / exp_k_pf / exp_k_tc / seg_k / gemm / gemm_grouped / gemm_tc / gemm_tc2 / gemm_sn / gemm_sn2 / gemm_sn3 / mix_pf / mix_split / pf_elem | fmad (**既に on**) | CPU と不一致 (1 行を 32 レーンに分割 / bf16 mma) | 無し (pf_elem は既に `__expf`) | 0 (FMA も fast exp も既に有効) |
| mix_up4 | nofmad | 不一致 (1 対 = 4 レーンで丸め順が変わる) | 有り → **既定の `up4f` は `__expf`** (task 437) | 0 (既に済み) |
| mix_dn5 / dn6 / dn7 / dn8 | nofmad | 不一致 (4/16 レーン分割) | 無し | fmad: PTX 差 0 行 (dn7) / `.rn` 接尾辞だけ (up4, upmix_q8) → 0 |
| mix_dev | nofmad | 不一致 (ツリー縮約。既定経路ではない) | 有り | 測定せず (下記) |
| mix_upmix_q8 / mix_down / mix_dev3 / mix_inj3 / dec_dev / attn_dec4 / elem / pf / pf_ple | nofmad | **ビット一致** (CPU の 1 演算 1 丸めに合わせる) | 有り | **0 (契約により触らない)** |
| attn_dec2 / attn_dec5 (flashdec) | nofmad | 緩和 (承認済み。累積順が違う) | 無し (既に `exp2f` = SFU) | 0 |
| gdn / gdn_scan2 / gdn_scan3 / gdn_pf_wy* | nofmad | 決定論のみ (CPU とビット一致しない) | 無し (素の `expf`) | fmad は**未測定** (下記) |

### 3.2 DENNO_FMAD の証拠 (PTX の実物)

同じ .cu を `-fmad=false` と `-fmad` (既定) で作り、t640a で両方を生成して diff した
(build_ptx.sh の `*_fm` target。結論が出たので target は削除):

| カーネル | diff 行数 | 中身 | `fma.rn.f32` 本数 |
| --- | ---: | --- | --- |
| mix_dn7 | **0** | 完全一致 | 0 -> 0 |
| mix_up4 | 128 | `sub.rn.f32` -> `sub.f32` 等 (既定の丸めと同じ) | 0 -> 0 |
| mix_upmix_q8 | 28 | 同上 | 0 -> 0 |

つまり「ビット一致契約の外にある」noFMAD カーネルでも、f32 の演算は全部
`__fadd_rn` / `__fmul_rn` / `__fma_rn` の intrinsic で書かれていて、`-fmad` は
**丸め指定の表記を変えるだけ**。命令列が変わらないので投入もしない → ゲートは足さない。

**未測定として残すもの**: `gdn` / `gdn_scan3` / `gdn_pf_wy*` (prefill の GDN) は
決定論のみの契約で、素の `expf` と素の `*`/`+` を含むため fmad の余地がある。
3 ラウンドの上限で測れていない。推定: prefill の ssm 段 (249.6 ms のうち GDN の
スカラー列) に数 % の余地。次に測るならこの 4 本で `*_fm` を作って prefill の
`gpu_ms`/`cpu_ms` を見る。

## 4. step 4: DENNO_PTX_ARCH=compute_120 は効果なし (採らない)

t640b = 12 本 (exp_iq_v3 / exp_iq_pf / exp_iq_tc / exp_k_tc / exp_k_pf / seg_k /
mix_dn7 / mix_up4 / gdn_scan3 / elem / pf / attn_dec4) を compute_120 で再生成して
ビルドし直した run。それ以外の 61 本は収録済みの compute_75 のまま。
比較は t640a (同じ 12 本が compute_75) との段の台帳 (同一モデル・同一 env・同一 SCENARIO):

| 段 (ms/token) | t640a compute_75 | t640b compute_120 | 差 |
| --- | ---: | ---: | ---: |
| mix | 1.87 | 1.88 | +0.5% |
| ssm | 2.76 | 2.74 | -0.7% |
| experts | 2.91 | 2.92 | +0.3% |
| full | 0.97 | 0.98 | +1.0% |
| short decode 中央値 | 10.0 / 9.9 | 9.8 / 9.8 | ノイズ |

機構の説明: 収録 PTX の `.target` は sm_75 でも、ドライバが sm_120 向けに JIT するので
命令選択はほぼ同じ。このコードは f32 の演算を全部 intrinsic で固定しており、
compute_120 で新たに使える命令 (cp.async / redux.sync 等) を nvcc が出す書き方に
なっていない。→ **既定は compute_75 に戻した** (`DENNO_PTX_ARCH=compute_120` で
いつでも試せる)。

## 5. lab の落とし穴 (次に読む人向け)

- COND の `DENNO_<K>_PTX=/...` に渡すパスは **`/tmp/labsrc/crates/denno-engine/kernels/...`**。
  `/src` は gen/train のシナリオのパスで、サーバのビルドは `/tmp/labsrc`。
  `/src` を指すと `PTX が読めない ... → v2 のまま` で**静かにフォールバック**し、
  A/B が別物になる (t640b/t640c の oldf16 は 19.3 ms/tok = +96%。これは
  「融合カーネルが効いている」証拠にしかならない)。
- `PTX=` で再生成した PTX は **収録済みのものだけがビルドツリーに置かれる**。
  新しい名前 (`*_legacy`, `*_a120`) は `done/<name>/ptx/` に来るだけなので、
  使いたいなら **リポジトリに収録して commit してから**次の run で env で指す。

## 6. テスト

- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`:
  **1430 passed / 0 failed / 103 ignored** (int-auto ac4f7ecf3 を merge した後)。
- `... --test kernel_invariants`: **155 passed / 0 failed** (merge 後)。
  merge 前の途中で 9 本赤にした (targets_all の末尾を検査する配線テスト) が、
  legacy target を targets_all から外して直した (commit 4/n)。
- `cargo test --release --workspace -- --test-threads=1` (cuda feature 無し):
  **rc=0、63 個のテストバイナリすべて ok (failed 0)**。
- `.cu` 30 本は g++ スタブ (`cuda_fp16.h` / `mma.h` / `cuda_bf16.h` の最小スタブ) で
  `-fsyntax-only` が通ることを確認 (変更前のファイルでも同じハーネスが通ることを対照に取った)。
  レジスタ数・smem・本物の PTX は nvcc でしか分からない。

## 7. GPU で確認してほしいこと

1. `cargo test --release -p denno-engine --features cuda --lib f16_cvt_probe_on_gpu`
   (GPU のあるマシンで 1 回)。期待: 何も panic せず、cnt=0 / first=0xffffffff。
   これが step 2 の「GPU 上での全パターン照合」の実測になる (このコンテナでは skip)。
2. prefill の GDN (gdn / gdn_scan3 / gdn_pf_wy*) の fmad 版を 1 ラウンド測る
   (`*_fm` target を戻して `-fmad` で作り、prefill の ssm 段 gpu_ms を見る)。

## 8. lab の名前と VERDICT

- t640a (8d1ef85ce, depth8k, ptx=31 本+3 本):
  `VERDICT base vs x0 OK decode +2.0% ... md5 SAME` / `base2 ... +0.0%`。
- t640b (7b914a0e3, depth8k, arch120 12 本):
  `VERDICT base vs x0 OK decode +0.0%` / `oldf16 ... +96.0%` (フォールバック。無効)。
- t640c (fcf302975, depth8k, ptx=none): `base +2.0%` / `oldf16 +96.0%` (同じ失敗) /
  `newf16 +0.0%` / `base2 +0.0%`。
- t640d (124d4f7e7, decode, ptx=none):
  `VERDICT base vs x0 OK decode +0.0% prefill +0.0% md5 SAME errors 0` /
  `VERDICT oldf16 vs x0 OK decode +2.0% prefill +0.9% md5 SAME errors 0` /
  `VERDICT base2 vs x0 OK decode +0.0% prefill +0.0% md5 SAME errors 0`。
- md5 (t640a と tint_d5a97b487 が同一):
  `a:edaa84ad b:1938059f c2000_cold:9b28e018 c2000_turn2:8b02d5c3
   c8000_cold:672fec07 c8000_turn2:199f23b3`。
- 出力テキスト: 4 run とも "looks ok" (errors 0)。md5 が DIFF になる変更は入れていない。
