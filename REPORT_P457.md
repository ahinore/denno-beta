ADOPT: yes t457c

# task 457 (dev2) — 既定モデル prefill: hc mix の norm (二乗和の分担) と mixed (f32 の sigmoid)

branch `br457` = 開始時の `int-auto` **`f145a13a6`**。終端でもう一度 fetch したが
`int-auto` は `f145a13a6` のままだったので merge は無し (`git log -1 int-auto` で確認)。
**br451 は merge していない**: task 451 の mix up 融合は `ADOPT: no` で既定 off、
本タスクの対象 (norm / mixed) とは無関係で、コードを足すと 3.5k 行の死んだ経路が
増えるだけなので要らない。

数値はすべて GPU ラボの実測 (`SCENARIO=depth8k`、`PROFILE=1`、
`model=flash-next-UD-IQ3_XXS`)。lab 名と測った SHA:
**t457a = `e88ba8c5701a`** (lab の `RESULT=ARCHIVE_FAILED`。ログも build.log も
残らず、測定は 1 つも無い。ラウンドに数えない)、
**t457b = `e88ba8c57e31fc04fee87f2a73856107b48be304`** (6 COND、ptx=pf_elem)、
**t457c = `87e057fc6c3ff776c776a1b84e85a307e5ab854e`** (5 COND、ptx=none = commit 済み
PTX)。t457b のあとゲートを既定 on にし、t457c は**既定 on のまま** (baseline の
COND だけ `=0` で切り戻して) 測った。

## 0. 結論 (7 行)

1. **採る。** prefill hc mix の 2 つの要素カーネルを差し替えた。
   (a) `dno_pf_rmsgroup_p`: 二乗和を 256 スレッドで分担 (従来は thread 0 が
   1 行 2,560 要素を逐次)。(b) `dno_pf_mixed_fs`: sigmoid の exp を
   `dno_pf_expf` (glibc expf の **double 精度**再現) → `__expf` (SFU)。
2. **mix 段の GPU busy は -24.6%** (`[q4ledger-pf]` の busy_ms、3,826、
   4 ラウンド同一): base 242.6 / 243.4 / 243.1 / 243.2 に対し
   both **183.5 / 183.2 / 183.5 / 183.4**。14k は 884.4 / 883.3 →
   **660.7 / 660.6 (-25.2%)**。内訳は norm 単独 -12.9% / fast sigmoid 単独 -11.6%
   で、ほぼ加算的。
3. **壁時計 14k cold (target)**: t457c で base2 5,700.6 → **both 5,336.3 ms
   (-6.3%)**、t457b で 5,612.5 → **5,344.5 (-4.8%)**。target の 2.5% を 2 つの
   run で満たす。3,826 cold は t457b 2,275.9 / 2,268.3 → 2,221.8 / 2,210.5
   (**-2.4%**)、t457c 2.27 → 2.23 s (-1.8%) で、**どちらも悪化していない**。
4. **他の段は動いていない** (3,826、t457c の base2 → both、GPU busy): full
   89.1 → 89.2 / ssm 361.8 → 361.9 / experts 530.8 → 529.0 / other 47.4 → 47.4。
   14k: full 638.8 → 640.0 / ssm 1321.6 → 1322.2 / experts 1426.2 → 1425.2 /
   other 172.1 → 171.8。**すべて 0.4% 以内**。
5. **decode は不変**: 短い decode 11.3-11.4 ms/tok (両 run、全 COND)。
   8k 後 turn2 は 12.5-12.7 (base 12.9-13.3、base2 12.8-13.0) で悪化なし。
6. **既定 on にした** (`DENNO_PF_MIX_NORMP` / `DENNO_PF_MIX_FASTSIG`、どちらも
   「明示的な偽値だけ off」)。切り戻しは `=0`。値は丸めが変わる (§3)。
7. **down は触らない**。3,826 の down 421.5 us は 588 MB を **1.4 TB/s** で流して
   いて、既に帯域の天井 (§4)。

## 1. Do 1 / Do 2: 現在の int-auto の段プロファイルと、選んだ段

### 1.1 段プロファイル (t457b base2 = ゲート off、実測)

| 段 (3,826) | wall ms | GPU busy ms |
| --- | ---: | ---: |
| experts | 1,280.0 | 530.4 |
| ssm | 413.4 | 362.0 |
| **mix** | 260.6 | **243.0** |
| other | 192.5 | 47.4 |
| full | 129.3 | 89.1 |
| 合計 (wall) | **2,275.9** | 1,272 |

| 段 (14k) | wall ms | GPU busy ms |
| --- | ---: | ---: |
| ssm | 2,235.2 | 1,323.0 |
| experts | 1,908.8 | 1,427.0 |
| **mix** | 20.9 | **884.4** |
| full | 882.8 | 640.5 |
| other | 564.7 | 172.2 |
| 合計 (wall) | **5,612.5** | 4,447 |

14k は GPU busy が壁の 79% で、**mix は wall 20.9 ms = 完全に非同期**(GPU が詰まって
いるだけ)。3,826 は GPU 56%。

### 1.2 なぜ mix を取ったか (Do 2)

- 14k の GPU の大きい 3 つは experts 1,427 / ssm 1,323 / **mix 884**。experts は
  428/432 が「写しの待ちは構造的」と結論し、ssm は 392/406/418/427 が wy4-wy7q まで
  掘っており、task 457 の State も再開を禁じている。**mix だけが「まだ手を付けて
  いない大きな段」**。
- mix の中で t451c の `[pf-mix-kprof]` (192 呼び出し平均) は norm 278.5 us /
  down 421.5 / lo 11.2 / 残り (up+mixed+inj) 約 552。**down 以外の 830 us が
  要素カーネル**で、t451 は「mix の DRAM を 27% 減らして 0%」= バイトではなく
  並列度・演算が律速と結論していた (§4 で norm の逐次和と fp64 expf を名指し)。
  その 2 つは**誰も測っていない**。
- 選んだ理由を 1 行で: **mix は 14k で 2 番目に大きい非同期 GPU 仕事で、その中の
  830 us/呼び出しが「逐次和」と「fp64 expf」という 2 つの明らかな直列ボトルネック
  だったから。**

## 2. 変えたこと (新カーネル 2 本、既存 .cu は触らない)

`crates/denno-engine/kernels/kernels_pf_elem.cu` (新規) + `build_ptx.sh` に
target `pf_elem` (fmad / compute_75、targets_all の末尾)。

### 2.1 `dno_pf_rmsgroup_p` — 二乗和を block で分担 (`DENNO_PF_MIX_NORMP`)

- 従来 (`dno_pf_rmsgroup`) は `if (threadIdx.x == 0) for i in 0..e` で 1 行
  2,560 要素を**1 スレッドの依存鎖**で舐め、128 スレッドは書き出しにしか使って
  いなかった (157 MB を 278.5 us = 564 GB/s)。
- 新: 1 ブロック 256 スレッド。(a) 各スレッドが自分の列 10 要素 (`tid + k*256`) を
  添字昇順に足す、(b) warp 内は `__shfl_down_sync` の木 (off=16..1)、(c) warp 間は
  shared の 8 本を番号順に足す、(d) 要素は**レジスタに持ったまま**書き出す
  (2 回目のグローバル読みが無い)。式と 1 演算 1 丸め
  (`__fadd_rn` / `__fmul_rn` / `__fdiv_rn` / `__fsqrt_rn`) は従来と同一。
  変わるのは二乗和の**丸めの順**だけ (決定的、木の形は固定)。
- 門番: `e <= 256 * 16 = 4,096`。実形は e=2,560 (10 要素/スレッド)。

### 2.2 `dno_pf_mixed_fs` — sigmoid を f32 の速い経路に (`DENNO_PF_MIX_FASTSIG`)

- 式は `dno_pf_mixed` と 1 文字も変えていない。違うのは
  `sigmoid = 1/(1+dno_pf_expf(-g))` を `__fdividef(1, 1+__expf(-g))` にしたことだけ。
- `dno_pf_expf` は glibc expf の**ビット一致**再現で、中身は `__fma_rn` の **double**
  演算 + double の表引き。この GPU の fp64 は 1/64 速で、mixed は 1 呼び出し
  あたり e*nt*hc = 19.6M 回 exp を呼ぶ (推定 200M fp64 op)。
- 値は exp の 2 ULP 程度 (重みの量子化誤差 1e-2 台より 5 桁小さい)。
- 門番: `hc <= 8` かつ `nt >= 32` (短い prefill は従来)。

### 2.3 fail closed と発火の証拠

PTX が無い / entry が無い / `.param` の数が違う (7 / 6) / 形が門番の外、のどれかで
**従来カーネルへ落ちる** (ゲート on でも既定と同じ値)。発火は 1 形につき 1 回
`eprintln!` で出る (`DENNO_PF_MIX_KPROF` 不要):

```
[pf-elem] normp grid=7652x1x1 threads=256 e=2560 rows=7652 hc=4 inv=tree
[pf-elem] mixed_fs grid=10x1913x1 threads=256 e=2560 hc=4 nt=1913 sigmoid=__expf
```

t457b/t457c の nrmp / both の COND だけに出て、base/base2 には 1 行も出ない
(= ゲート off の COND は従来経路をそのまま通った)。

## 3. 数値 (lab、同じ run の中の対)

### 3.1 mix 段の GPU busy (計器、`[q4ledger-pf]` の busy_ms)

| run | COND | 3,826 r1 / r2 | 14k (r1) |
| --- | --- | ---: | ---: |
| t457b | base | 242.6 / 243.4 | 881.6 |
| t457b | nrmp | 211.4 / 211.7 | 761.2 |
| t457b | fsig | 214.8 / 214.6 | 782.4 |
| t457b | both | **183.5 / 183.2** | **660.7** |
| t457b | base2 | 243.0 / 243.1 | 884.4 |
| t457c | base | 243.1 / 243.2 | 882.6 |
| t457c | nrmp | 211.3 / 211.4 | 760.6 |
| t457c | both | **183.5 / 183.4** | **660.6** |
| t457c | base2 | 242.9 / 243.0 | 883.3 |

t457b と t457c は**別のバイナリ・別のラウンドで 4 桁まで同じ**。norm -12.9%、
fast sigmoid -11.6%、両方で -24.6% (3,826) / -25.2% (14k)。

### 3.2 壁時計

| run | 3,826 cold (r1 / r2) | both | 差 |
| --- | ---: | ---: | ---: |
| t457b | 2,437.5 / 2,271.3 (base)、2,275.9 / 2,268.3 (base2) | 2,221.8 / 2,210.5 | **-2.4%** |
| t457c | 2.27 (base)、2.27 (base2、中央値) | 2.23 | **-1.8%** |

| run | 14k cold base | base2 | both | 差 |
| --- | ---: | ---: | ---: | ---: |
| t457b | 7,817.4 (page cache 冷、使わない) | 5,612.5 | **5,344.5** | **-4.8%** |
| t457c | 5,728.1 | 5,700.6 | **5,336.3** | **-6.3%** |

14k の採否は**同じ run の base2 との対**で読んだ (base は t457b で page cache が
冷えていて 7.8 s、これは task の注意書きどおり採否に使えない)。
3,826 の t457b base r1 (2,437.5) も外乱 (t451c の base r1 と同じ傾向)。

### 3.3 decode

短い decode は 11.3-11.4 ms/tok で全 COND 同値 (0.1 ms 刻みの分解能)。
8k 後 turn2: t457c base 12.9 / both 12.5 / base2 13.0、t457b base 13.3 /
both 12.5 / base2 12.8。**悪化していない**。

### 3.4 md5 と text

- `a` / `b` (短い 2 本) は**全 COND で SAME** (`edaa84ad` / `1938059f`)。
- `c2000_cold`: base/base2 `b42d79d2`、nrmp `b42d79d2` (SAME)、both `9b28e018` (DIFF)。
- `c8000_cold`: base/base2 `672fec07`、nrmp `c9f47aad` (DIFF)、**both は `672fec07`
  で SAME** (2 つの丸め変更がたまたま同じトークン列に戻る)。
- `c2000_turn2` (base `12fc5c36`) と `c8000_turn2` (base `98e879d2`) は DIFF。
- DIFF はすべて**設計どおりの丸めの変更** (§2)。モデルの挙動を変える種類の変更
  ではない (枝刈り・近似カーネルの類ではない)。
- text (t457c の 14k cold の応答、TXTD both):

  > ご提示いただいた文書は、LLM推論エンジン（`denno-engine）における **Prefill（事前計算）フェーズの性能最適化の歴史を詳細に記録したものです。\n\n以下に、文書の構造と技術的な深掘りした解説を提供します。

  base 側は「…**Prefill（事前計算）処理の性能最適化の記録です。\n\n以下に、この文書の技術的詳細を、**「問題の特定」、「解決プロセス」…」で、
  どちらも sane。`!!!!` / 空出力 / プロンプトの反復は無い (errors 0)。

## 4. 見積りと、やらないと決めたこと

- 見積り (推定、実測ではない): norm -150 us + mixed -250 us = -400 us/呼び出し
  → 3,826 で -77 ms (-3.3%)、14k で -269 ms (-4.8%)。**実測は -333 us/呼び出し
  (14k の mix busy -223.7 ms / 672 呼び出し) で、見積りの 83%**。
- **down は触らない**。3,826 の down 421.5 us は x の 5 回読み (n=320 を BN=64 で
  5 タイル) + 重みの 30 回読みで 588 MB を流しており、**1.4 TB/s = ほぼ帯域の天井**。
  split-k は x の再読みが split 倍に増えるので**逆効果**。BN を広げるのは
  アキュムレータのレジスタ (64x320 = 80 reg/スレッド) が無理。
- `pf-mix-kprof` の up / mixed / inj のイベントが 1 本も解決しない既知の不具合は
  未修理 (本タスクの範囲外)。今回の判定は台帳の `busy_ms` で足りた。

## 5. 数値の扱い

- 先頭 COND (`x0`) は page cache が冷える (t457b の 3,826 8.34 s、14k 7.30 s) ので
  採否に使っていない。`vs x0` の VERDICT 行も使っていない。
- **14k はラウンド 1 にしか出ない**。t457b では x0 (7.30) と base (7.82) が
  まだ冷えており、nrmp/fsig/both/base2 (5.34-5.61) とは 40% 違う。採否は
  **温まった後ろの 2 つ (both と base2) の対**で読んだ。base2 は both より後に
  走っているので、この対は **both に不利な向き** (base2 がより温かい) の比較。
- 3,826 は 2 ラウンド x 2 run で、base と base2 の一致は t457c で 2.27/2.27
  (中央値)。t457b の base r1 だけが外乱。
- 段の数字は n=1 のサンプルではなく**そのラウンドの積算** (4704 起動 = 672 呼び出し)。

## 6. VERDICT 行

t457b (`e88ba8c57e31fc04fee87f2a73856107b48be304`、6 COND、ptx=pf_elem):
```
VERDICT base vs x0 OK decode -0.7% prefill +0.9% md5 SAME errors 0
VERDICT nrmp vs x0 OK decode +0.0% prefill -1.7% md5 DIFF errors 0
VERDICT fsig vs x0 OK decode -0.3% prefill +0.0% md5 DIFF errors 0
VERDICT both vs x0 OK decode -0.3% prefill -0.9% md5 DIFF errors 0
VERDICT base2 vs x0 OK decode +0.3% prefill +0.0% md5 SAME errors 0
VERDICT_DEPTH both c8000_cold vs x0 decode -9.3% prefill -26.7%
VERDICT_DEPTH base2 c8000_cold vs x0 decode -8.6% prefill -23.2%
VERDICT baseline x0 errors 0 text ok
```
(`vs x0` は捨て COND 相手なので採否に使っていない。採否は §3 の同じ run の対。)

t457c (`87e057fc6c3ff776c776a1b84e85a307e5ab854e`、5 COND、ptx=none):
```
VERDICT base vs x0 OK decode -0.7% prefill +1.8% md5 SAME errors 0
VERDICT nrmp vs x0 OK decode -1.4% prefill +1.8% md5 DIFF errors 0
VERDICT both vs x0 OK decode -0.7% prefill +0.9% md5 DIFF errors 0
VERDICT base2 vs x0 OK decode -0.7% prefill +1.8% md5 SAME errors 0
VERDICT_DEPTH base c8000_cold vs x0 decode +0.0% prefill -5.8%
VERDICT_DEPTH nrmp c8000_cold vs x0 decode -0.6% prefill -7.7%
VERDICT_DEPTH both c8000_cold vs x0 decode -0.7% prefill -12.2%
VERDICT_DEPTH base2 c8000_cold vs x0 decode +0.7% prefill -6.2%
VERDICT baseline x0 errors 0 text ok
```

## 7. テスト (最終 HEAD = int-auto merge 後)

```
CARGO_TARGET_DIR=/tmp/t457b cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1323 passed / 0 failed / 103 ignored
CARGO_TARGET_DIR=/tmp/t457b cargo test --release -p denno-engine --features cuda --test kernel_invariants -- --test-threads=1
  -> 132 passed / 0 failed
CARGO_TARGET_DIR=/tmp/t457nc cargo build --workspace --release            (cuda feature 無し)
  -> ok
CARGO_TARGET_DIR=/tmp/t457nc cargo test --workspace --release -- --test-threads=1
  -> ok (workspace の全バイナリ。既知の赤 3 件を除く)
```

追加したテスト:
- `cuda::pf_elem_tests::pf_elem_gate_truth_table` — ゲートの真理値表
  (**明示的な偽値だけ off** = 既定 on)。切り戻しの env が効くことの確認。
- `pf_elem_normp_ok_bounds` / `pf_elem_fastsig_ok_bounds` / `pf_elem_real_shape_grid`
  — 形の門番と実形の grid (norm 7,652 / mixed 10 x 1,913)。
- `pf_elem_norm_tree_matches_reference` — 分担版の二乗和をホストで鏡化し、
  従来の逐次和と f64 真値に 1e-6 相対で一致 (木の順序の取り違え検出)。
- `pf_elem_fastsig_values` — fast sigmoid が f64 参照に 3e-7 相対、端点 (0/1) で
  壊れない。
- `kernel_invariants::task457_pf_elem_kernel_and_wiring` — .cu の `#define` と
  cuda.rs のミラー、build_ptx.sh の target、**PTX の ABI (引数 7 / 6 本)**、
  norm の 1 演算 1 丸めの式、mixed の式。
- `task457_pf_elem_gates_default_on` / `task457_pf_elem_off_path_keeps_old_kernels`
  — 既定 on と、off 側が従来カーネル (`k.rmsgroup` / `k.mixed`) であること。
- `targets_all` の末尾 assert 6 箇所を `... attn_dec2 pf_elem)` へ更新
  (task 397/407/418/422/437 のテスト。task 451 と同じ手当て)。

## 8. GPU で確認してほしいこと (監督へ)

1. **既定 on**。切り戻しは `DENNO_PF_MIX_NORMP=0` / `DENNO_PF_MIX_FASTSIG=0`
   (片方ずつ戻せる。t457b で片方ずつの効果も測ってある: norm -12.9% /
   fast sigmoid -11.6%)。
2. `kernels_pf_elem.ptx` は t457b が `e88ba8c57` から生成したものをそのまま
   commit した (2 entry / 引数 7・6 本 / `ex2.approx` 6 / `shfl.sync` 5 /
   `div.rn.f32` 2)。t457c はそれを測定に使った (ptx=none)。
3. 生成 AI の応答は 14k の 1 コンテキストで言い回しが変わる (md5 DIFF §3.4)。
   丸めの変更によるもので、text は sane。厳密に戻したいときは上記 env。
4. 次の候補 (推定): `mixed` はまだ 176 MB を ~440 us で流している (400 GB/s)。
   `__expf` で演算は消えたので、次は **mixed と up の融合** (t451 の up2 は
   expf が残っていたので効かなかったが、いまは違う) か、mixed を 1 スレッド
   2 要素にして起動数を半分にする方向。norm は 157 MB / 約 120 us (推定) で
   帯域なり。down は §4 のとおり触らない。
