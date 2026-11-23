ADOPT: yes t461d

# REPORT P461 — 既定モデル prefill: PLE 注入をデバイスで閉じる
# (3,826 cold prefill -4.6%、other 段 190.5 -> 85.3 ms / 14k は ±1% / decode 不変)

branch `br461` = 開始時の `int-auto` **`6f2237c69`** (= merge: br457。自分の br457 は
既に merge 済みだったので取り込み不要)。終端でもう一度 fetch し直したが `int-auto` は
`6f2237c69` のままだったので merge は無し (`git merge-base --is-ancestor` で確認)。
数値はすべて GPU ラボの実測 (`SCENARIO=depth8k`、`PROFILE=1`、`model=flash-next-UD-IQ3_XXS`)。
lab 名と測った SHA: **t461a = `3d911065b`** (dev COND が 500。下記 §6 の 1)、
**t461b = `9bc865c5a`** (dev COND が panic。§6 の 2)、
**t461c = `6ffa35300`** (dev COND が動いた。§3 の「直す前」)、
**t461d = `8e3747ca7`** (最終。4 COND、ptx=pf_ple)。
t461a / t461b は自前の不具合で **dev COND の測定が無い**ので、採否は t461c / t461d で読んだ。

## 0. 結論 (7 行)

1. **採る。** prefill の PLE 注入 (gate → conv+silu → 残差加算) を、res / key / value を
   ホストへ往復させずデバイスで閉じた (`DENNO_PF_PLE_DEV`、既定 on)。
2. **3,826 cold prefill が -4.6%** (同じ run の対、ledger の wall): base2 2,210.6 /
   2,216.0 ms に対し **2,143.3 / 2,106.3 ms** (t461d の 2 ラウンド、両方とも下)。
   RESULT 表の中央値でも base2 2.21 s → **2.12 s (-4.1%)**。target の 2.5% を満たす。
3. **機序**: `[q4ledger-pf]` の `other` 段が **190.5 → 85.3 ms (-105 ms)**。PLE 層
   (48 層中 2 層) が 1 チャンク層あたり res 82 MB を D2H → ホストのトークンループ →
   82 MB を H2D していた往復と、`ple` のホスト仕事 (14k で 236.6 ms) が消えた。
4. **14k は ±1%** (dev 5,303.6 対 base 5,432.9 / base2 5,280.8 ms = -2.4% / +0.4%)。
   内訳は `other` -83 ms と `experts` +87 ms で相殺 (§4.2)。14k の 1 ラウンドは
   サンプル 1 本で、この run の base と base2 の差 (2.9%) より小さい。
5. **decode は不変**: 短い decode 11.3 ms/tok (全 COND)、8k 後 12.3-13.7 (dev が
   いちばん良い側)、c2000 後 13.6-13.7。**errors 0、text sane** (§5)。
6. **md5 は 2 コンテキストだけ DIFF** (`c2000_turn2` / `c8000_turn2`)。原因は
   (a) 二乗和・内積の縮約が block の木になった、(b) sigmoid の exp を `__expf` に
   した、の 2 つで、どちらも丸めだけ (§3.3)。cold prefill の md5 は 4 本とも SAME。
7. **切り戻しは `DENNO_PF_PLE_DEV=0`**。形・PTX・entry・引数の数が外れたら従来の
   ホスト経路へ fail closed (値も既定と同じ)。

## 1. Do 1 / Do 2: 現在の int-auto の段プロファイルと、選んだ段

### 1.1 段プロファイル (t461c base_1、既定 env、実測)

| 段 (3,826) | wall ms | GPU busy ms |
| --- | ---: | ---: |
| experts | 1,261.9 | 527.8 |
| ssm | 379.9 | 360.4 |
| other | 192.9 | 47.3 |
| full | 116.8 | 88.8 |
| mix | 213.5 (t461d) | 242.9 |
| 合計 (wall) | **2,211.7** | 1,272 |

| 段 (14k) | wall ms | GPU busy ms |
| --- | ---: | ---: |
| ssm | 2,080.5 | 1,320.4 |
| experts | 1,837.3 | 1,421.7 |
| full | 891.3 | 637.4 |
| other | 602.2 | 171.4 |
| mix | 21.6 | 659.0 |
| 合計 (wall) | **5,432.9** | 4,442 |

### 1.2 なぜ PLE を取ったか (Do 2)

- 大きい段は experts (転送律速。task 296/320/324 が gpuwait / lookahead / group まで
  詰めており、22.4 GB @ 20.6 GB/s の DDR 3 往復が床) と ssm (task 186/228/231/296/311
  が wy4 まで、392/406/418/427 が decode 側) で、どちらも「まだ手を付けていない」とは
  言えない。task 457 で mix も一巡した。
- 残る**ホスト側**の大きい項目が `other` で、その内訳は
  `ple 236.6 ms + res_up 142.0 ms` (14k、t461d base) = **prefill の 7%**。どちらも
  PLE 層 (48 層中 2 層) の往復で、**データは元から VRAM にある** (res は device の
  残差、key / value は PLE 射影の出力)。ホストへ落として CPU で回して戻す理由が無い。
- 選んだ理由を 1 行で: **prefill の残る最大のホスト仕事が PLE 層の往復 (res 82 MB x2 +
  key/value の D2H、14k で 379 ms) で、しかも全部デバイス内で閉じられる形だったから。**

## 2. 変えたこと (新カーネル 2 本 + 既存の量子 GEMM の再利用)

`crates/denno-engine/kernels/kernels_pf_ple.cu` (新規) + `build_ptx.sh` に target
`pf_ple` (nofmad、targets_all の末尾)。既存の .cu / .ptx は触っていない。

### 2.1 `dno_pf_ple_gate` — 1 ブロック = 1 トークン (11 引数)

CPU 参照 (`ple_inject_into` / `ple_inject_batch_par` の段 1) と同じ式:

- `ss_k = Σ key²` / `ss_r = Σ res²` → `invk / invq = 1/sqrt(ss/e + eps)`
- `dot = Σ (key*invk*gk) * (res*invq*gq)`、`sv = dot/sqrt(e)`
- `gate = sigmoid(sign(sv) * sqrt(max(|sv|, 1e-6)))`
- `invc = 1/sqrt(Σ(value*gate)²/e + eps)`

縮約は warp の木 (`__shfl_down_sync` off=16..1) + warp 間 8 本の順次 (457 の norm と
同じ形)。1 演算 1 丸め (`__fadd_rn` / `__fmul_rn` / `__fmaf_rn` / `__fdividef`)。

### 2.2 `dno_pf_ple_conv` — 1 ブロック = (列タイル, ストリーム) (18 引数)

```
gated = value[t][d] * gate[t][s]
x     = gated * invc[t][s] * gc[s][d]                 (grouped RMS の出力)
conv  = silu(Σ_k wt[k][s][d] * x[t - (kern-1-k)*dil])
res[t][s][d] += gated + conv
```

- 列 (d) ごとにトークン方向へ**因果**に進む。直前 `hist = (kern-1)*dil` 個の x は
  スレッド局所の ring に持ち、チャンク前の履歴はデバイスの履歴バッファから読む
  (slot は CPU と同じ `(head + hist + t - back) % hist`、head は 0 = 線形)。
- 実形は kern=4 / dil=3 (lab の `[pf-ple] dev ... kern=4 hist=9`)、grid は
  (ceil(e/256)=10, hc=4) = 40 ブロック。
- 履歴は**線形 (先頭が最古) で書き戻す**。CPU 経路もチャンク終端で
  `ple_hist_to_linear` を通すため (§6 の 2 で踏んだ)。

### 2.3 射影 (key / value) は既存カーネルのままデバイス出力へ

既定モデルの `ple_key` / `ple_value` は **Q8_0** で、従来のホスト経路は
`pf_ple_proj` の量子 GEMM (`pf2_gemm`) を使う。デバイス経路も**同じ `pf2_gemm`** を
出力だけデバイスにして撃つ (入力 `flat` もデバイスにあるので H2D は 1 回)。
F32 / BF16 / F16 の目録のときは f32 GEMM のデバイス出力版 (`f32_gemm_dev_out`)。

### 2.4 同期は 1 層 1 回

履歴の H2D は層の最初のチャンクだけ、D2H (と役割 RMS の寄与) は層の**最後の
チャンク**だけ。`emit_role_act` は `end == nt_all` のときしか roles を読まないので
途中チャンクの寄与は捨ててよい。lab t461c では毎呼び出し同期していて 14k の
`experts` が +89 ms 悪化した (t461d で 1 層 2 回に減らした)。

### 2.5 fail closed と発火の証拠

PTX が無い / entry が無い / 引数の数が違う (11 / 18) / 形が門番の外 / 残差のデバイス域が
無い、のどれかで **従来のホスト経路へ落ちる** (ゲート on でも既定と同じ値)。
起動を始めた後の失敗だけ `Err` (res を書き換えたかもしれないので黙って落とさない)。
発火は 1 形につき 1 回 `eprintln!`:

```
[pf-ple] dev nt=1913 hc=4 e=2560 kern=4 hist=9 gate=dno_pf_ple_gate conv=dno_pf_ple_conv
```

t461c / t461d の dev COND だけに出て、base / base2 には 1 行も出ない。

## 3. 数値 (lab、同じ run の中の対)

### 3.1 壁時計 (ledger の wall、t461d)

| run | 3,826 r1 / r2 | 14k (r1) |
| --- | ---: | ---: |
| base | 2,445.4 (先頭で冷え) / 2,213.7 | 5,432.9 |
| dev | **2,143.3 / 2,106.3** | **5,303.6** |
| base2 | 2,210.6 / 2,216.0 | 5,280.8 |

RESULT.md の中央値 (c2000_cold): base 2.33 / **dev 2.12** / base2 2.21 s。
dev は base2 に対して **-4.1%**、ledger の 2 ラウンド対でも -3.2% / -4.7%。
t461c でも dev の 2 本目は 2,106.5 ms と同じ値だった (再現する)。

### 3.2 段 (t461d、14k)

| 段 | base | dev | base2 |
| --- | ---: | ---: | ---: |
| other | 602.2 | **453.7** | 537.0 |
| experts | 1,837.3 | 1,903.6 | 1,816.9 |
| full | 891.3 | 828.6 | 826.4 |
| ssm | 2,080.5 | 2,096.0 | 2,079.4 |

`other` は -83 ms (PLE の `ple` 236.6 と `res_up` 142.0 のうち往復ぶん)。
`experts` は +87 ms で、層の最後のチャンクで 1 回だけ待つ同期が expert の上げ
ストリームを 2 回 drain するぶん (推定)。14k では相殺して ±1%。

### 3.3 decode と md5 / text

- 短い decode は全 COND 11.3 ms/tok (0.1 ms 刻み)。8k 後 turn2 は dev 13.1 /
  12.3 で base (13.5 / 12.9)・base2 (13.3 / 12.6) より良い側。
- md5: `a` / `b` / `c2000_cold` / `c8000_cold` は 4 COND とも SAME
  (`edaa84ad` / `1938059f` / `9b28e018` / `672fec07`)。
  `c2000_turn2` と `c8000_turn2` だけ DIFF (§0-6 の丸め 2 箇所)。
- text (t461d の dev、TXTD c8000_cold): base と同じ書き出しで sane。

## 4. 見積りと、やらないと決めたこと

- 見積り (推定、実測ではない): PLE の往復 286 MB/チャンク層 (res 82 x2 + key/value 102
  + flat 20) と CPU のトークンループが消えるので、14k で -250 ms、3,826 で -100 ms。
  **実測は 3,826 で -105 ms (other)**、14k で -83 ms (other) + experts +87 = ±0。
  3,826 の見積りは当たり、14k は同期のぶんだけ目減りした (t461d で改善済み)。
- 14k の `experts` +87 ms は同期 2 回ぶん。1 層 1 回に減らしても残るので、次の候補は
  「同期を層の最後のチャンクから要求の最後まで遅らせる」(履歴を要求の終端で 1 回だけ
  D2H する) だろう (推定 -40〜80 ms)。
- **PLE 層の conv はカーネル 1 本で 40 ブロックしか立たない** (grid 10 x hc 4)。列方向に
  さらに割る (1 ブロック 128 列) と 80 ブロックになり、帯域が伸びる余地がある (推定)。
  今は往復が消えたぶんで足りているので触らない。
- task 451 / 457 が触った mix の up / mixed / down は今回触っていない。

## 5. 数値の扱い

- 先頭 COND (`x0`) は page cache が冷える (t461d の 3,826 2.22 s、14k 6.58 s) ので
  採否に使っていない。`vs x0` の VERDICT 行も使っていない。
- 14k はラウンド 1 にしか出ない = **1 COND 1 サンプル**。t461d の base と base2 は
  同じバイナリで 5,432.9 / 5,280.8 ms (2.9% 違う) なので、14k の ±1% は**この run の
  分解能より小さい**。3,826 は 2 ラウンド x 2 サンプルで、dev の 2 本は base2 の
  2 本より下に離れている (差 3-5%)。
- **t461c の dev_1 (3,826) は 2,282.9 ms と外れた**。同じバイナリの dev_2 は
  2,106.5 ms で、`other` のサブ項目 (バイト数・cpu) が dev_1 と完全に同じなのに
  wall だけ 167 ms 大きい。原因は**そのプロセスで最初の PLE 呼び出しでの PTX の
  JIT** (CUDA のディスクキャッシュが温まると消える。t461d の dev_1 は 2,143.3 ms で
  再現しない)。採否は t461d の 2 ラウンドで読んだ。

## 6. 途中で踏んだこと (lab の 2 ラウンドを消費した)

1. **t461a**: dev COND の全リクエストが 500 (`PLE の射影 (デバイス出力) に失敗`)。
   既定モデルの PLE の重みは Q8_0 で、f32 GEMM の口では型で落ちていた。量子型は
   `pf2_gemm` を再利用する形に直した (§2.3)。
2. **t461b**: dev COND が panic (`RefCell already borrowed`)。`pf_ple_note` を
   `PLE_DEV` を `borrow_mut` している内側で呼んでいた。borrow の外へ移した。
   同時に、履歴の書き戻しが ring のままだったのを線形に直した (CPU 経路は
   チャンク終端で `ple_hist_to_linear` を通すので、そのままだと次のチャンク /
   decode が履歴を 1 段ずらして読む)。**鏡テストを CPU の線形化と突き合わせる形に
   してあったので、この 2 つは lab の前に見つかるはずだった** (t461b の時点で
   鏡テストは ring 前提のままだった = テストが実装に合わせていた。以後は
   CPU の `ple_hist_to_linear` を呼んで比較している)。

## 7. VERDICT 行

t461d (`8e3747ca7`、4 COND、ptx=pf_ple):
```
VERDICT base vs x0 OK decode -0.3% prefill -2.6% md5 SAME errors 0
VERDICT dev vs x0 OK decode -0.3% prefill -2.6% md5 DIFF errors 0
VERDICT base2 vs x0 OK decode +0.0% prefill -2.6% md5 SAME errors 0
VERDICT_DEPTH base c2000_cold vs x0 decode -1.1% prefill +5.0%
VERDICT_DEPTH dev c2000_cold vs x0 decode -0.7% prefill -4.3%
VERDICT_DEPTH base c8000_cold vs x0 decode -7.3% prefill -17.5%
VERDICT_DEPTH dev c8000_cold vs x0 decode -9.9% prefill -19.5%
VERDICT baseline x0 errors 0 text ok
```
(`vs x0` は捨て COND 相手なので採否に使っていない。採否は §3.1 の同じ run の対。)

t461c (`6ffa35300`、4 COND、ptx=pf_ple): dev の 3,826 が 2,282.9 / 2,106.5 ms、
14k 5,298.8 ms、base 5,539.6 / base2 5,365.9。VERDICT は全て OK / errors 0
(md5 は t461d と同じ 2 コンテキストだけ DIFF)。

## 8. テスト (最終 HEAD = int-auto merge 後)

```
CARGO_TARGET_DIR=/tmp/t461b2 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1329 passed / 0 failed / 103 ignored
CARGO_TARGET_DIR=/tmp/t461b2 cargo test --release -p denno-engine --features cuda --test kernel_invariants -- --test-threads=1
  -> 133 passed / 0 failed
CARGO_TARGET_DIR=/tmp/t461nc cargo build --workspace --release            (cuda feature 無し)
  -> ok
CARGO_TARGET_DIR=/tmp/t461nc cargo test --workspace --release -- --test-threads=1
  -> ok (既知の赤 3 件を除く)
```

追加したテスト:
- `cuda::pf_ple_dev_tests::pf_ple_dev_gate_truth_table` — ゲートの真理値表
  (**既定 on** = 明示的な偽値だけ off)。
- `pf_ple_dev_shape_gate` — 形の門番 (実形 kern=2/3/4、hc<=8、e<=8192、
  hist == (kern-1)*dil、上限)。
- `pf_ple_dev_grid` — 起動形 (conv grid.x = ceil(e/256) = 10) と `#define` のミラー。
- `pf_ple_gate_mirror_matches_cpu_reference` — gate / invc の式を CPU の
  `grouped_rms_norm_into` + `rmsnorm_into` と 1e-5 相対で突き合わせ。
- `pf_ple_conv_mirror_matches_cpu_ring` — conv の索引算術 (back の向き / 履歴 ring /
  書き戻しの線形化) を CPU の `ple_conv_silu_t` + `ple_hist_push_ring` +
  `ple_hist_to_linear` と 1e-5 相対で突き合わせ (kern/dil 4 通り)。
- `pf_ple_conv_mirror_ring_head` — チャンク前の履歴の head を 0..hist-1 で振って同じ。
- `kernel_invariants::task461_pf_ple_kernel_and_wiring` — .cu の `#define` と cuda.rs の
  ミラー、カーネル名 (cuda.rs が読む名前)、式の文字列、build_ptx.sh の target、
  **commit 済み PTX の ABI (11 / 18 引数)**。
- `targets_all` の末尾 assert 6 箇所を `... attn_dec2 pf_elem pf_ple)` へ更新。

## 9. GPU で確認してほしいこと (監督へ)

1. **既定 on**。切り戻しは `DENNO_PF_PLE_DEV=0` (従来のホスト経路・同じ値)。
2. `kernels_pf_ple.ptx` は t461d が `8e3747ca7` から生成したものをそのまま commit
   した (2 entry / 11・18 引数 / `ex2.approx` 2 / `shfl.sync` 25 /
   `.entry dno_pf_ple_gate` / `dno_pf_ple_conv`)。
3. 生成 AI の応答は 2 つの turn2 コンテキストで言い回しが変わる (md5 DIFF §3.3)。
   丸めの変更によるもので text は sane。厳密に戻したいときは上記 env。
4. 14k は ±1% で、この run の base/base2 の差 (2.9%) より小さい。**14k で勝ちたい
   なら次の 2 つ** (推定): (a) 履歴 / 寄与の同期を要求の終端 1 回にする、
   (b) conv を列方向に割って 40 -> 80 ブロックにする。
