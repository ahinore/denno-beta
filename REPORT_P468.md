ADOPT: yes t468c

# REPORT P468 — 既定モデル decode: full 層の attn mix も defer し、
# full attention の qkv GEMV が鎖の `mixed` をデバイスから直接読むようにした
# (14k 後の decode 12.3 -> 12.0 ms/token = -2.4%、短い decode と prefill は不変)

branch `br468` = 開始時の `int-auto` **70982c7dc** (merge: br461)。
**br464 は int-auto に未 merge だったので取り込んだ** (`--no-edit` で merge、衝突なし。
br464 の `DENNO_MIX_NOWAIT` (ssm 層の defer、既定 on) が前提)。終端でもう一度
fetch したが int-auto は **70982c7dc のまま**だった (`git merge-base --is-ancestor` で
確認、merge 不要)。

数値はすべて GPU ラボの実測 (`SCENARIO=depth8k`、`PROFILE=1`、モデルは
`flash-next-UD-IQ3_XXS`、ベースライン COND = int-auto の既定 + br464 +
`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0
DENNO_GPU_WEIGHT_MB=12000`、先頭に捨て COND)。lab 名と測った SHA:
**t468a = `c4dd33917`** (全リクエストが落ちた)、**t468b = `f4af7eb2b`**
(新しい文脈の最初のリクエストだけ落ちた)、**t468c = `a260b0607`** (A/B)。
3 run とも `RESULT.md` 2 行目は `model=flash-next-UD-IQ3_XXS`、`load_before` は
1.58 / 1.62 / 1.32。

## 0. 結論 (6 行)

1. **採る。** full 層 (12 層/token) の **attn の mix** も `DENNO_MIX_NOWAIT` の
   defer に載せ、直後の full attention の device 経路 (`dec_full_dev`) が鎖の
   出力 `mixed` を **qkv GEMV の入力にそのまま使う**ようにした。ホストは
   `mixed` を 1 度も読まない (D2H 無し・H2D 無し・イベント待ち無し)。
   値は同じバイト列なので **md5 は全 COND・全コンテキストで SAME**。
2. **14k 後の decode は -2.4%** (summary の中央値 12.3 -> 12.0。base 12.4/12.2 →
   新 12.0/12.0 で、4 サンプルが完全に分かれる)。**task の Target はこれで満たす**
   (短い decode と prefill は不変、errors 0、text 健全)。
3. **短い decode は 11.2 -> 11.2 (0.1 ms 刻みで差なし)**、prefill 0.57 -> 0.56-0.57 s
   (どちらも悪化していない)。段は warm で mix 2.630 -> 2.595 (-1.3%)、
   full 1.19 -> 1.18 (-0.9%)、14k で mix 2.75 -> 2.70 (-1.8%)、
   step 12.20 -> 12.01 (-1.6%、台帳の平均)。
4. **発火の証拠**: `[mix-nowait-full] layers=1104` (128 token x 12 層 - 最初の
   1 トークンぶん)。台帳の `mix` の D2H が **0.72MB/109 -> 0.60MB/97**、
   14k の `full` の H2D が **0.12MB/12 -> 0.00MB/0**、`mix` の H2D も
   11.22MB/99 -> 11.22MB/99 (入力は元から鎖が上げていない)。
5. **採る根拠は 14k の Target の枝**。段の規則 (2 ラウンドで 2% 以上) は
   **満たさない** (mix -1.3%、full -0.9%)。full 層は 97 回の mix のうち 12 回だけで、
   1 回あたりに消えるのは D2H 1 本 + イベント待ち ~5-8 us なので、天井が
   ~0.1 ms/token しかない。**残りの 48 回 (ffn の mix) は router と CPU ミス expert が
   `mixed` をホストで読むので defer できない** (次はここ)。
6. **正直な制限**: t468c の 14k は run あたり 1 サンプル x 2 ラウンド (depth8k の仕様)。
   台帳の step (0.1 ms 刻みでない) では -1.6%、summary の中央値では -2.4% で、
   **目標の 2% は「満たす」と「境界」の間**。判断は summary (lab の表) で行った。

## 1. なぜこれを選んだか (Do 1 / Do 2)

t468c の base COND の段プロファイル (`[q4ledger]`、ms/token、warm の 128 token):

| 段 | warm | 14k 後 |
| --- | ---: | ---: |
| mix | 2.63 | 2.72-2.79 |
| router | 0.17 | 0.18 |
| shared | 0.63 | 0.71 |
| full | 1.19 | 1.17-1.64 |
| ssm | 2.78 | 2.83-2.87 |
| experts | 2.88 | 3.18-3.60 |
| head | 0.60 | 0.60 |
| 合計 (step) | 11.14 | 12.11-12.38 |

experts (2.9-3.6) と ssm (2.8) は task 388/406/418/422/437/448/461 で各 2-4 回
手が入っていて、GPU 側は [exp-iq-kprof] の実測で 450-540 GB/s (帯域の 1/3)、
ssm は台帳の `gpu` が 0 で待ちの付け替えの塊。**カーネル単体をこれ以上速くするには
.cu の中身 (量子 dequant の内側) を触る必要があり、1 タスクで 2% を出すには
リスクが高い**。P464 の申し送り (「次に効きそうなのは full 層の attn mix の
defer」) が唯一、構造が分かっていて値も変わらない残りだった。

P464 の時点で full 層を defer できなかった理由は 1 つだけ: **full attention の
device 経路が `cur` (ホスト) を pinned に写して qkv GEMV の入力にしていた**から。
鎖は同じ `mixed` をデバイスに残している (`DENNO_GDN_X_MIX` が GDN にやっている
のと同じ控え)。そこを配線した。

## 2. 変えたこと

### 2.1 `cuda.rs::dec_full_dev` (入力の H2D を省く)

- `Call` に `x_dev: Option<u64>` を足した。`Some` のとき `forward_inner` は
  pinned への写しも `cuMemcpyHtoDAsync` も撃たず、そのポインタを qkv GEMV の
  x に使う。`qkv_in` の作業域も確保しない。
- `ok_shape` の `cur.len() == qkv_k` は `x_dev.is_some() || ...` に緩めた。
- `dev_input_ok(len)`: **ホストだけで判定できる**事前検査 (ゲート on /
  attention のゲート / `len` の範囲 / chain_ctx)。mix の前に defer の可否を決める。
- `mirror_ready(cache_id, n_layer, kv_dim, layer, gen, len)`: `mirror_append` の
  前提 (ミラーの種・容量・`have + 1 == len`) を**書かずに**評価する。種の無い
  新しい文脈では defer しない (t468b でこれを踏んだ、§6)。
- 発火の計数 `[mix-nowait-full] layers=N` (`x_dev` で H2D を省いた層数)。

### 2.2 `qwen4exp.rs::run_layer` (defer の判定)

```
let mix_defer = mix_nowait_on() && gdn_x_mix_on() && gdn_gpu_win::active() && hook.is_none()
    && match layer_kind(il) {
        Ssm => gdn_gpu_scan_on() && gdn_ba_gpu_on(),      // 従来のまま
        FullAttn => self.attn.takes_dev_input(il),        // 新規
        _ => false,
    };
```

`LayerAttn::takes_dev_input` (既定 `false`) を足し、`FullAttn` が
`mix_nowait_full_on() && gdn_x_mix_on() && dev_input_ok(len + 1) && mirror_ready(...)`
を返す。`SsmAttn` は full 層だけ委譲する (ssm 層は `false`)。

### 2.3 `qwen4exp_attn.rs` (空の入力 = デバイス入力)

- 入口の長さ検査 2 箇所 (`FullAttn::forward` / `forward_token_impl`) は
  **空を許す**。空 = 「入力は鎖がデバイスに残した `mixed`」の合図。
- `dev_forward` が `full_x_in_mix_take(n_embd)` で控えを取り、`Call.x_dev` に入れる。
  控えが無い / device 経路が走らない回は **要求を落とす** (空の入力で黙って
  CPU 経路に落ちて 0 を計算しない)。QSA 層も空では回せないので落とす (既定 off)。

### 2.4 門は既存 env の値

`DENNO_MIX_NOWAIT` の値で切る (新しい env 名は lab の COND が落とすことがある):
既定 = ssm + full、`=ssm` = ssm だけ (full の切り戻し)、`=0` / `=off` = 両方止める。

## 3. 数値の扱い

- カーネル・引数・発行順は 1 つも変えていない (入力の置き場所だけ)。同じバイト列
  なので **値はビット一致**。実測でも md5 は t468c の全 COND・全コンテキストで
  SAME (`edaa84ad / 1938059f / 9b28e018 / 8b02d5c3 / 672fec07 / 199f23b3`)。
- よって差分は「速いか遅いか」だけ。段は同一 run の `[q4ledger]`、全体は
  summary の decode 中央値 (0.1 ms 刻み) で読んだ。

## 4. lab の結果 (t468c)

### 4.1 summary (ms/token、0.1 ms 刻み)

| コンテキスト | base | base2 | full | full2 | 差 (中央値) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 短い decode (a/b/warm) | 11.2 | 11.2 | 11.2 | 11.2 | 0 |
| c2000_turn2 | 12.5 | 12.2 | 12.3 | 12.1 | -0.15 (-1.2%) |
| **c8000_turn2 (14k 後)** | **12.4** | **12.2** | **12.0** | **12.0** | **-0.3 (-2.4%)** |
| c8000_cold | 13.3 | 12.9 | 12.6 | 12.7 | -0.45 (-3.4%) |
| c2000_cold | 13.7 | 13.4 | 13.4 | 13.3 | -0.2 (-1.5%) |
| prefill (28 tok) s | 0.57 | 0.57 | 0.56 | 0.57 | 0 |

`full` の 2 サンプルは 14k でどちらも 12.0、base の 2 サンプルは 12.4 / 12.2。
**4 サンプルが群で分かれる** (0.1 ms 刻みの量子化を考えるとこれが最大の根拠)。

### 4.2 台帳の段 (`[q4ledger]`、ms/token、同一 run)

| 段 | base (2 run) | full (2 run) | 差 |
| --- | ---: | ---: | ---: |
| warm mix | 2.65 / 2.61 | 2.60 / 2.58 | **-1.3%** |
| warm full | 1.19 / 1.20 | 1.18 / 1.18 | -0.9% |
| warm step | 11.16 / 11.13 | 11.11 / 11.18 | ±0 |
| 14k mix | 2.72 / 2.79 | 2.67 / 2.74 | -1.8% |
| 14k full | 1.63 / 1.17 | 1.60 / 1.17 | -1.2% (h2d 0.12MB -> 0) |
| 14k step | 12.38 / 12.11 | 11.91 / 12.14 | -1.6% |

### 4.3 発火の証拠

```
[mix-nowait] deferred_layers=5712 alias_layers=4608
[mix-nowait-full] layers=1104          (128 token x 12 層 - 1 トークンぶん)
[q4ledger] mix  ... d2h=0.60MB/97      (base は 0.72MB/109)
[q4ledger] full ... h2d=0.00MB/0       (base は 0.12MB/12、14k)
```

### 4.4 VERDICT 行 (t468c、c0 = 捨て COND)

```text
VERDICT base vs c0 OK decode -2.1% prefill -0.9% md5 SAME errors 0
VERDICT full vs c0 OK decode -1.7% prefill -1.7% md5 SAME errors 0
VERDICT base2 vs c0 OK decode -1.7% prefill -0.9% md5 SAME errors 0
VERDICT full2 vs c0 OK decode -2.1% prefill -0.9% md5 SAME errors 0
VERDICT_DEPTH base c8000_turn2 vs c0 decode -8.6% prefill -45.2%
VERDICT_DEPTH full c8000_turn2 vs c0 decode -12.1% prefill -46.2%
VERDICT_DEPTH base2 c8000_turn2 vs c0 decode -10.3% prefill -46.2%
VERDICT_DEPTH full2 c8000_turn2 vs c0 decode -11.5% prefill -46.2%
```

c0 は page cache の冷たい捨て COND なので c0 との比較は読まない。base 同士でも
c8000_turn2 が -8.6% と -10.3% に割れており、**run 間の揺れは ±1% 程度**ある。

## 5. 出力テキスト (t468c、full COND の warm)

`TXTA` (`full_1_a1.json`、28 token のプロンプト + 128 token):
`# B-Trees: A Comprehensive Explanation\n\n## 1. What is a B-Tree?\n\nA **B-tree (Balanced Tree) is a self-balancing tree data structure that maintains sorted data and allows searches, sequential access, insertions, and deletions in logarithmic time.\n\n### Key Characteristics:\n- **Balanced**: All leaf nodes are at the same depth.\n- **Multi-way**: Each node can have more than`

`TXTD` (14k 後、`full_1_c8000_turn2.json`):
`ご提示いただいた文書は、LLM推論エンジン（`denno-engine）における **Prefill（事前計算）処理の性能最適化の記録です。\n\n以下に、この文書の技術的詳細を、**「問題の特定」、「最適化の段階的変遷」、「計測の罠と教訓**`

base と同一 (md5 `199f23b3`)。

## 6. 壊れた 2 run (次に同じ轍を踏まないための記録)

1. **t468a (`c4dd33917`)**: 全リクエストが `FullAttn::forward: 入力長 0 が n_embd =
   2560 と違う` で落ちた (md5 は空文字列 `d41d8cd9`)。長さ検査を
   `forward_token_impl` だけで緩め、**anyhow 版 `forward_token` の入口にもう 1 本
   同じ検査が残っていた**のを見落とした。
2. **t468b (`f4af7eb2b`)**: 新しい文脈の最初のリクエスト (a/b) だけ
   `入力をデバイスから読む回で device 経路が走らなかった` で落ちた。
   `dec_full_dev` は **ミラーの全行シード**を要求し、新しい文脈の最初の full 層は
   必ず従来経路へ落ちる。`mirror_ready` を足して、種が無い回は defer しない
   (= ホストの `cur` を使う従来経路) ようにした。c2000/c8000 は SAME だった。
3. どちらも**要求を落とす (fail loud)** ので、値が静かに狂うことはない。
   t468b の 2 つの失敗は「要求が落ちた」として JSON にはっきり出た。

## 7. テスト

関門 (cuda feature あり。int-auto 70982c7dc + br464 を merge 済み):

```text
CARGO_TARGET_DIR=/tmp/t468 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1338 passed; 0 failed; 103 ignored
同 --test kernel_invariants
  -> 136 passed; 0 failed
```

足したテスト (4 本):
- `cuda::tests::task468_mix_nowait_full_gate_truth_table` — 既定 on、`ssm` で
  full だけ off、偽値で両方 off、`full`/真値は ssm の defer も on のまま。
- `cuda::tests::task468_mix_nowait_full_line_counts_layers` — 証拠の行 (0 層/ゲート off は出さない)。
- `kernel_invariants::task468_full_mix_nowait_wiring` — 門が既存 env の値であること、
  `dec_full_dev` が `x_dev` で H2D を省くこと、`mirror_ready` の条件、full 層の
  defer が `takes_dev_input` を通ること、空入力の 2 箇所の検査と fail loud、
  `SsmAttn` の委譲 (ssm 層は借りない)、serve_flash の 1 行。
- `task464_mix_nowait_wiring` は defer の判定が層種別の `match` に変わったので
  期待文字列を更新した (ssm の前提は変えていない)。

## 8. GPU で確認してほしいこと / 次への申し送り

1. 既定は **`DENNO_MIX_NOWAIT` = ssm + full**。切り戻しは `=ssm` (full だけ止める)、
   全停止は `=0`。`[mix-nowait-full] layers=N` が発火の証拠。
2. **危険側の挙動**: defer した full 層で device 経路が実行時に decline すると
   (ミラーの種切れ・VRAM 不足など)、以前は CPU 経路で正しい値が出たが、今は
   要求が落ちる (fail loud)。`mirror_ready` で既知の原因は事前に弾いている。
3. 次に効きそうなのは **ffn の mix 48 回の defer**。今は router GEMV が鎖の末尾に
   載っていて `probs` をホストで読むため待ちが残る。expert の prologue の x も
   ホストから上げている (1 層 10 KB)。`DENNO_GDN_X_MIX` と同じく **expert GEMV が
   鎖の `mixed` をデバイスから読む**ようにすれば、48 回ぶんの D2H (~0.05 MB/token) と
   H2D 1 本/層が消える見込み (**推定 -0.5 to -1%**)。
4. `DENNO_GDN_DEC_KPROF=1` (t468a の kp COND) の `[dec-kprof-avg]` で見ると、full 層は
   1 層 `wall 181 us / kern 90 / copy 17 / gap 50 / wait 101 / launch 15 / steps 70`
   (prefill 混じり) で、**`fd_sync` が 1 回 75.6 us**。full 層の残りは attention の
   待ちが主で、カーネルは `dno_gemv_q8_0_v5` (qkv と wo で 320 blocks) が ~30 us ずつ。
