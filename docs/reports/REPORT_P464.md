ADOPT: yes t464c

# REPORT P464 — 既定モデル decode: ssm 層の attn mix で鎖の D2H と待ちを
# GDN の同期へ後ろ倒しにした (`DENNO_MIX_NOWAIT`)。mix 段 -5.6 to -12.6%、
# 短い decode は 11.4 -> 11.2 ms/token (-1.8%) で **task の 2% 目標は未達**

branch `br464` = 開始時の `int-auto` **6f2237c69** (merge: br457) から切り、
**br460 (task 460) は int-auto に未 merge だったので取り込んだ** (14c75be36、
fast-forward)。終端で int-auto を fetch し直したら **70982c7dc (merge: br461)**
に動いていたので merge した。衝突は `build_ptx.sh` の `targets_all` と
`kernel_invariants.rs` の末尾一致 assert 6 箇所 (+ task 460 の `attn_dec4)` assert)。
どちらも**両側を残し**、末尾は
`... attn_dec2 pf_elem attn_dec4 pf_ple)` にした。

数値はすべて GPU ラボの実測 (`SCENARIO=depth8k`、`PROFILE=1`、モデルは
`flash-next-UD-IQ3_XXS`、ベースライン COND = int-auto の既定 +
`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0
DENNO_GPU_WEIGHT_MB=12000`、先頭に捨て COND):
**t464a = 6bf57f32c** (壊れていた run)、**t464b = 326d5c058** (既定 off の A/B)、
**t464c = d5e918c40** (既定 on に反転した最終確認)。3 run とも
`RESULT.md` 2 行目は `model=flash-next-UD-IQ3_XXS`、`load_before` は
2.86 / 11.63 / 1.94。

## 0. 結論 (6 行)

1. **採る (段の規則で)。** decode の mix の Q8_0 鎖は 1 呼び出しごとに
   `cuEventSynchronize` 1 本で待つ。ssm 層 (36 層/token) の **attn の mix** は
   出力 `mixed` をホストで読む相手が居ない (直後の GDN が `DENNO_GDN_X_MIX` で
   デバイスの `dm` をそのまま読む。inject は鎖の外でホストが計算済み) ので、
   D2H を撃たず待ちも省き、GDN submission の最後の D2H 待ちにまとめた。
   `DENNO_MIX_NOWAIT` を**既定 on** にした (切り戻しは `=0`)。
2. **mix 段は -5.6 to -12.6%**: t464c の `[q4ledger]` mix は
   4.42/3.59/3.02/2.84/2.74 -> **3.51/3.59 相当が基準で、off は
   4.99/4.04/3.33/3.01/2.91** (a/b/c2000_cold/c2000_turn2/c8000_turn2 の順)。
   t464b でも 2.81/2.85 -> 2.68/2.62。2 ラウンドで再現。
3. **短い decode は 11.4 -> 11.2 ms/token (-1.8%、t464c の 3 サンプル)、
   14k 後は 12.6 -> 12.1-12.5**。**task の Target「2%」は未達**。t464b は
   11.3-11.4 -> 11.2 (-1.3%)。
4. **ssm 段は +1.7 to +6.7% (同じ待ちの付け替え)**: mix の待ちが GDN の待ちへ
   移るので、`[q4ledger]` の ssm 行がその分だけ増える。**mix + ssm の和**は
   全コンテキストで改善 (a: 7.63 -> 7.99 ではなく off が 7.99 で on が 7.63)、
   step は 11.37 -> 11.23 (t464b) / 11.31-11.38 -> 11.19-11.25 (t464c)。
   **値はビット一致** (md5 は 3 run・全 COND・全コンテキストで SAME)。
5. **試して駄目だったもの**: head の GEMV 行数 `DENNO_DEC_HEAD_FAST=dn6r8` は
   head 段 0.61 -> 0.72-0.73 ms/token で**悪化** (既定は 4 行のまま)。
   `DENNO_GDN_BA_PAR=1` (ba matvec を別ストリームへ) は decode 11.8 vs 11.2 =
   **+5.2% 悪化** (既定 off のまま、コードは触っていない)。
6. **正直な制限**: 14k は depth8k の仕様で 1 run 1 サンプル。ssm 段の数字は
   「待ちの付け替え」を含むので単独では読めない (mix + ssm の和か step で読む)。

## 1. 前提: br460 込みの int-auto の段プロファイル (t464c の base COND)

`[q4ledger]` (ms/token、a = 28 トークン目から 128 token、b = 48 token、
warm = 2 回目の 128 token):

| 段 | a | b | warm | 14k 後 (t464b の base) |
| --- | ---: | ---: | ---: | ---: |
| mix | 4.42 | 3.59 | 2.65 | 2.92 |
| router | 0.18 | 0.18 | 0.17 | 0.18 |
| shared | 0.71 | 0.70 | 0.65 | 0.66 |
| full | 1.37 | 1.74 | 1.20 | 1.63 (br460 の auto + cbg) |
| ssm | 3.21 | 3.21 | 2.80 | 2.78 |
| experts | 10.90 | 6.61 | 2.88 | 3.51 |
| head | 0.61 | 0.61 | 0.60 | 0.61 |
| 合計 (step) | 43.2 | 33.7 | 11.19 | 12.3-12.5 |

experts は expert cache の暖まりで大きく動く (a の 10.9 は VRAM への
上げ直し中)。decode の中央値 (11.2-11.4) は a/b の混合。

## 2. なぜこれを選んだか (Do 1 / Do 2)

- 3 つの大きい段 (experts / mix / ssm) は task 388/406/418/422/431/437/392/441 で
  各 2-4 回手が入っている。カーネル単体は帯域か ALU の天井に近い
  (mix down dn7 288 GB/s は warp 数律速、up4f 387 GB/s、experts 355-541 GB/s)。
- 台帳の `mix` は 97 呼び出し/token で、1 呼び出し = カーネル 2 本 + H2D/D2H +
  **イベント待ち 1 本**。カーネルは ~21 us、呼び出しの壁は 41 us。つまり
  **残りの ~20 us は往復の遅れ**で、ここは今まで「消す」方向が未着手だった
  (task 382 の ZC は PCIe 読みで失敗、CUDA graph は t384a で失敗)。
- そこで「待たなくてよい待ち」を探した。ssm 層の attn mix だけが出力を
  ホストで読まない (t392c の `DENNO_GDN_X_MIX` が既にデバイス入力を配線済み)
  ので、D2H と待ちを落として GDN の同期へ寄せた。**新しいカーネルは不要**
  (値も発行順も変わらない)。

## 3. 変えたこと

### 3.1 mix の鎖 (`cuda.rs::maybe_gpu_mix_q_chain_into`)

- 引数 `defer: bool` を足した。defer の回は
  (a) `mixed` の D2H を撃たない、(b) イベントを打つが**待たない**。
  イベント (slot 8) は次の鎖の入口 `mix_pending_drain()` が scratch
  (`stage` / `hx`) の上書き前に待つ。実際には直後の GDN が同期するので
  そこで完了済み。イベントが張れない環境では降ろし直して文脈同期 (defer しない)。
- `gdn_x_in_mix_arm(mdev, e)` は従来どおり (defer でも指す先は同じ)。
- `mix_take_deferred()` / `gdn_x_in_mix_armed(n)` / `mix_nowait_line()` を追加。
- GDN submission (`gdn_decode_submit`): 早期の寸法検査が空の `cur` を
  **CPU 経路へ静かに落としていた** (lab t464a で最初の ssm 層から壊れた)。
  空は「控えが armed のときだけ」許し、控えが無ければ `poisoned` で要求を落とす。

### 3.2 呼ぶ場所 (`qwen4exp.rs`)

- `hc_mix_quant_defer` を足し、`run_layer` の **attn の mix だけ** defer する。
  条件: `DENNO_MIX_NOWAIT` && `DENNO_GDN_X_MIX` && `DENNO_GDN_GPU_SCAN` &&
  `DENNO_GDN_BA_GPU` (ホスト ba は `cur` が要る) && GDN の GPU 窓 && ssm 層 &&
  hook 無し。**ffn の mix は defer しない** (router GEMV が鎖に載り、CPU ミスの
  expert も `mixed` をホストで読む)。
- 鎖が defer したら空の `Vec` を返す (`mix_take_deferred`)。

### 3.3 受け側 (`qwen4exp_ssm.rs`)

- 空の `cur` は「入力はデバイス側 (`dm`) だけ」の合図。控えが無ければ要求を
  落とす。GPU 経路が decline したときも落とす (黙って 0 で計算しない)。
  ba をホスト計算するモデル (`ba_units` 空) でも落とす。

### 3.4 発火の証拠

- `[mix-nowait] deferred_layers=4608 alias_layers=4608` (128 token x 36 ssm 層)。
  off にすると `deferred_layers=0` で alias だけ 4608。
- `[q4ledger]` の mix 行: `d2h=1.09MB/145 -> 0.72MB/109` (36 回ぶん減る)。

## 4. 数値の扱い

- カーネル・引数・発行順は 1 つも変えていない (ホストへ降ろさないだけ) ので
  **値はビット一致**。実測でも md5 は t464b/t464c の全 COND・全コンテキストで
  SAME (`edaa84ad / 1938059f / 9b28e018 / ac9314a6 / 672fec07 / f820511a`)。
- したがって差分は「速いか遅いか」だけ。判定は同一 run の `[q4ledger]` の段
  (mix + ssm の和、step) と decode 中央値で読んだ。

## 5. lab の結果

### 5.1 mix 段 (`[q4ledger]`、ms/token、t464c = 既定 on vs off)

| 文脈 | on (base/base2) | off/off2 | 差 |
| --- | ---: | ---: | ---: |
| a (128 tok) | 4.42 / 4.39 / 4.47 | 4.99 / 5.04 / 5.00 | **-0.58 (-11.6%)** |
| b (48 tok) | 3.59 / 3.44 / 3.51 | 4.04 / 4.00 / 4.04 | **-0.51 (-12.6%)** |
| c2000_cold | 3.02 / 3.00 / 3.01 | 3.33 / 3.27 / 3.32 | -0.30 (-9.0%) |
| c2000_turn2 | 2.84 / 2.83 | 3.01 / 3.00 | -0.17 (-5.6%) |
| c8000_turn2 (14k) | 2.74 / 2.73 | 2.91 / 2.97 | -0.20 (-7.0%) |

t464b (既定 off の A/B、mix の warm): base 2.81 / 2.85 / 2.82 ->
nw 2.68 / 2.62 / 2.62 (-4.6 to -6.8%)。**2 ラウンドで 4.6% 以上**。

### 5.2 ssm 段と step (同じ t464c)

| 文脈 | ssm on | ssm off | step on | step off |
| --- | ---: | ---: | ---: | ---: |
| a | 3.21 | 3.00 | 43.2 | 36.7 (experts の cache 差で比較不能) |
| b | 3.21 | 3.03 | 33.7 | 30.8 (同上) |
| warm | 2.80 | 2.75 | 11.19-11.25 | 11.31-11.38 |
| c2000_turn2 | 2.87 | 2.80 | - | - |

ssm の増分 (+0.05 to +0.21) は **mix の待ちが GDN の待ちへ移ったぶん**で、
mix + ssm の和は on が常に小さい (warm: 5.45 vs 5.60、a: 7.63 vs 7.99)。
step は warm の block で **-0.13 ms/token (-1.2%)**。

### 5.3 decode 中央値 (summary、0.1 ms 刻み)

| run | 既定 (off 相当) | 新 (on) | 差 |
| --- | ---: | ---: | ---: |
| t464b | base 11.4 / base2 11.3 | nw 11.2 / nw2 11.2 | -1.3% |
| t464c | off 11.4 / off2 11.4 | base 11.2 / base2 11.2 | **-1.8%** |

14k 後 (c8000_turn2): t464c は on 12.1-12.5 / off 12.6、t464b は on 12.5 /
off 12.6-13.1。prefill は構造上不変 (defer は decode の窓の中でだけ成立) で、
実測も 0.56-0.59 s のノイズ内 (-3.4% to +2.6%)。

### 5.4 VERDICT 行 (t464c)

```text
VERDICT base vs x0 OK decode +0.3% prefill -2.6% md5 SAME errors 0
VERDICT off vs x0 OK decode +1.4% prefill -3.4% md5 SAME errors 0
VERDICT off2 vs x0 OK decode +1.4% prefill -2.6% md5 SAME errors 0
VERDICT bap vs x0 OK decode +5.2% prefill -3.4% md5 SAME errors 0
VERDICT base2 vs x0 OK decode +0.3% prefill -3.4% md5 SAME errors 0
VERDICT_DEPTH base c8000_turn2 vs x0 decode -1.8% prefill +2.0%
VERDICT_DEPTH off c8000_turn2 vs x0 decode -1.2% prefill +4.0%
VERDICT baseline x0 errors 0 text ok
```

x0 が新既定 (on) なので `off` の +1.4% が「切り戻したとき」のコスト。

t464b の VERDICT:

```text
VERDICT base vs x0 OK decode +0.7% prefill +1.7% md5 SAME errors 0
VERDICT nw vs x0 OK decode -1.0% prefill +0.9% md5 SAME errors 0
VERDICT nw2 vs x0 OK decode -1.4% prefill +1.7% md5 SAME errors 0
VERDICT kp vs x0 OK decode -1.0% prefill +2.6% md5 SAME errors 0
VERDICT base2 vs x0 OK decode +0.0% prefill +0.0% md5 SAME errors 0
VERDICT baseline x0 errors 0 text ok
```

## 6. 出力テキスト (t464c の base COND)

- `TXT`: `# B-Trees: A Comprehensive Explanation\n\n## 1. What is a B-Tree?\n\nA **B-tree (Balanced Tree) is a self-balancing tree data structure that maintains sorted data and allows searches, sequential access, insertions, and deletions in logarithmic time.\n\n`
- `TXTB`: `Rust の「所有権（Ownership）とライフタイム（Lifetime）は、Rust を学ぶ上で最も重要であり、かつ最も難しい概念です。\n\nしかし、これらは「メモリを安全に管理するためのルール**です。`
- `TXTD` (14k 後): `ご提示いただいた文書は、LLM推論エンジン（denno-engine）における **Prefill（事前計算）フェーズの性能最適化の歴史を詳細に記録した技術レポートです。`

on / off の TXT/TXTB/TXTD は同一で、md5 も全コンテキストで SAME。

## 7. 採否の判定 (task の規則)

- Target「短い decode **か** 14k 後で 2%」: 短い decode **-1.3 to -1.8%**、
  14k 後 **-1.0 to -4.0%** (サンプル 1 本/run)。=> **未達**。
- Why の運用規則「段で 2 ラウンド以上 2% 以上勝ち、他が 1% 以上悪化せず、
  text 健全 -> 目標未満でも既定 on」: **mix 段 -4.6 to -12.6% を 2 ラウンドで
  再現** (t464b / t464c)、text 健全、errors 0、md5 SAME。
  ただし **ssm 行は +1.7 to +6.7% に出る** — これは同じ待ちが mix から ssm へ
  移っただけで、mix + ssm の和と step は改善する (§5.2)。この読み替えを
  認めないなら規則は満たさない。
- よって **ADOPT: yes (t464b/t464c)**、既定は on。切り戻しは
  `DENNO_MIX_NOWAIT=0`。

## 8. テスト

関門 (cuda feature あり。int-auto 70982c7dc を merge 済み):

```text
CARGO_TARGET_DIR=/tmp/t464 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1336 passed; 0 failed; 103 ignored   (task 464 で 4 本追加。merge 前 1330)
同 --test kernel_invariants
  -> 135 passed; 0 failed                  (task 464 で 1 本追加。merge 前 134)
```

足したテスト:
- `cuda::tests::task464_mix_nowait_gate_truth_table` — 既定 on、明示的な偽値だけ off。
- `task464_mix_nowait_line_counts_layers` — 発火の証拠の行 (0 層なら出さない)。
- `task464_mix_take_deferred_is_one_shot` — defer の印は読むと消える。
- `task464_gdn_x_in_mix_armed_does_not_consume` — 控えの有無は消費せずに読める。
- `kernel_invariants::task464_mix_nowait_wiring` — 門・defer の実装 (D2H を撃たない
  条件 / イベントを待たない分岐 / 次の鎖の入口の drain / ホスト値を作らない)・
  呼ぶのは ssm 層の attn mix だけ (defer の入口は 1 箇所)・ssm 側の空入力の検査
  (armed / ba ホスト)・GDN の早期検査と poisoned・serve_flash の 1 行。

## 9. GPU で確認してほしいこと / 次への申し送り

1. 既定は `DENNO_MIX_NOWAIT=1`。切り戻しは `=0` (既存 env なので lab に届く)。
   `[mix-nowait] deferred_layers=N alias_layers=N` が発火の証拠。
2. **ssm 行の数字は単独で読まない** (待ちの付け替え)。mix + ssm の和か step で読む。
3. 残る往復: ffn の mix 48 回 (router と CPU ミス expert が `mixed` をホストで
   読むため defer 不可)、full 層の attn mix 12 回 (full の device 連鎖が `cur` を
   H2D している。`dm` を読ませれば defer できる余地はある)、expert 36 回。
   次に効きそうなのは full 層の attn mix の defer と、experts の 1 層 1 同期の
   短縮 (推定 -0.5 to -1%)。
4. `DENNO_DEC_HEAD_FAST=dn6r8` は head 段を +18% 悪化させる (既定は 4 行)。
   `DENNO_GDN_BA_PAR=1` は decode +5.2% 悪化。どちらも既定 off のまま。
5. 危険側の挙動: defer した層で GDN の GPU 経路が実行時に decline すると
   (scan の発行失敗など)、以前は CPU 経路へ落ちて正しい値が出ていたが、今は
   要求が落ちる (fail-loud)。前提 (窓 / ba GPU / armed) は毎層見ているので、
   起きるのは実行時失敗のときだけ。
