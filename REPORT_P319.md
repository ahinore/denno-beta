ADOPT: yes t319a t319b

# REPORT P319 — IQ3_S decode: near miss を回収。shared v5 (`DENNO_DEC_SH_V5=2`) を既定 on に (-2.7% / -2.4%)

branch: br319 = int-auto `6465bf554` + **br310 を merge** (`d7103e021` = `DENNO_DEC_SH_V5` の実装。
int-auto に無く、測るのに要った。br316 には入っていたが int-auto には未 merge) + 最後に
int-auto `0d3cd40da` (merge: br320) を merge。lab 2 ラウンド: **t319a** (`a0c729c5d`、5 COND) /
**t319b** (`a0c729c5d`、6 COND、診断付き)。数値はすべて GPU ラボの実測。

## 0. 結論 (3 行)

1. **採る。** P310 が「target 4% に届かない」で既定 off のまま残した `DENNO_DEC_SH_V5=2`
   (shared expert の gate/up GEMV を v5 で先に引く) は、task 319 の採否規則
   (2 ラウンドの中央値で 2% 以上 / 他が 1% 超悪化なし / テキスト健全) を満たす:
   短い decode **-2.7% (t319a) / -2.4% (t319b)**、14k 後 **-2.0%**、prefill 不変、
   md5 全 COND・全ラウンド SAME、errors 0。**既定を 2 にした** (戻すのは `DENNO_DEC_SH_V5=0`)。
2. 機序は P310 のとおりで、他は 1 つも動かない: `[q4prof] shared` 54.8 -> 24.2 ms/128tok
   (`sub shared` 0.86 -> 0.62 ms/token、`sh_wait` 0.52 -> 0.05)。`ssm` / `experts` /
   `full` / `mix` / `head` の台帳は不変。
3. 次の山も測った (P316 §8 の申し送り): full 層 dev 経路のホスト仕事は **1 層 39-49 us**。
   内訳は `fd_qkv` 8.4-10.4 / `fd_mirror` 7.1-9.1 / `fd_attn` 7.1-8.8 / `fd_prep` 5.3-8.0 /
   `fd_wo` 3.6-4.1 / `fd_gate` 3.4-4.0 / `fd_d2h` 2.9-3.6 / `fd_tail` 0.9-1.0 us で、
   **どれも 2% を取れる大きさが無い** (12 層で 0.5 ms/token、しかも短い文脈では
   dev 経路は **44% の層だけ**)。失敗理由は 100% が `mirror` (KV ミラーの全行シード待ち)。

## 1. 前提の測り直し (Do 1): 現在の int-auto の per-stage

`SCENARIO=depth8k PROFILE=1`。t319a の `base` (= int-auto の既定 + lab の 4 env、
`DENNO_DEC_SH_V5=0` と同値) の実測。

- 短い decode (a2 = 28 -> 128 warm、同一プロンプト): **1.86 s / 128 tok = 14.5 ms/token**。
  `[q4prof]` の `total=1812-2018 ms/128 tok`、`layer_exec=6144` (= 128 x 48 層)。
- 台帳 (`[q4ledger]`、14k の窓): `full` 2.19 (launch 0.11 / sync 1.92 で **12 同期**) /
  `ssm` 3.41 (launch 0.22 / 161 発) / `experts` 3.92 (cpu 列 = 起動と同期) /
  `shared` 0.86 (`sh_begin` 0.42 + `sh_wait` 0.52) / `head` 0.73 / `loop` sync 1.73 (47) /
  `router` 0.18。P310 の値と同じで、int-auto はこの task の間 (a0c729c5d で測った時点) で
  decode の構成が変わっていない。
- `[exp-iq]`: `lookups=61440 hits=59689 hit_rate=0.972 vram=21998MiB/34000MiB`
  = expert の VRAM 常駐は既定 env (`DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0`)
  で 97%。`[exp-iq-kprof]` は 4.1-4.8 us/本 361-580 GB/s (既知の天井)。
- md5 は t307c/t309c の参照からずれたまま (`c2000_cold:a91e17da c2000_turn2:5a8f5488
  c8000_turn2:561c6da8`。P310 §3.3 / P316 §1 と同じ差)。**この task の変更とは無関係**
  (t319a / t319b の全 COND・全ラウンドで同じ値 = 決定的)。`a` / `b` / `c8000_cold` は一致。

## 2. 変えたこと (Do 2)

### 2.1 `DENNO_DEC_SH_V5` の既定を 2 にした (採否)

- `parse_dec_sh_v5` の既定を `2` にした。`1` = gate/up/down の 3 本、`2` = gate/up だけ、
  明示的な偽値 (`0` / `off` / `false` / `no` / 空) = 従来 (v3/v2)、未設定・未知の値 = `2`。
- 真理値表テスト `dec_sh_v5_gate_truth_table` を新しい既定に合わせて書き直した
  (未設定 = 2、偽値 7 種 = 0、未知 4 種 = 2)。
- 新しい PTX は要らない (`kernels_moe_v5.ptx` は既にコミット済み)。
- 数値: v5 は 4 アキュムレータなので丸め順が変わる (決定性は保つ)。**t319a / t319b の
  5+6 COND x 2 ラウンドすべてで md5 SAME、テキストも同一**。

### 2.2 full 層のホスト内訳 `fd_*` を `[dec-kprof-avg]` に足した (診断、既定 off のまま)

- `dec_full_dev::forward_inner` の各段に `dec_kprof::note_since` を打った:
  `fd_qkv` (pin コピー + H2D + 重み/計画 + qkv GEMV 起動) / `fd_prep` (scratch + norm 上げ +
  rope 表 + qknorm 起動) / `fd_mirror` / `fd_attn` / `fd_gate` / `fd_wo` / `fd_d2h` /
  `fd_sync` / `fd_tail` (pin -> Vec の詰め替え)。
- `DENNO_GDN_DEC_KPROF=1` のときだけ動く (`on()` の読み 1 回が off のコスト = 数値経路は不変)。

## 3. lab

### 3.1 t319a (`a0c729c5d`、5 COND、load_before 1.92) — 採否を読む回

COND = toss(`SH_V5=0` + `GDN_DEC_KPROF=1 MIX_KPROF=1`) / base(`SH_V5=0`) / sh2(既定 = `SH_V5=2`) /
base2(`SH_V5=0`) / sh2b(既定)。toss だけ診断 env を載せ、先頭 COND のバイアスを捨てた
(toss の prefill だけ 9.23 / 10.25 s に膨らんでいる = 先頭バイアス)。

a2 (28 -> 128 の warm、**128 token の秒**。この run の decode 中央値は 17.0-17.4 ms/tok で
0.1 ms 刻みでは差が出ないため、秒で読む):

| cond | a2 r1 / r2 | c2000_turn2 r1 / r2 | c8000_turn2 (r1) | md5 |
| --- | --- | --- | --- | --- |
| base | 1.86 / 1.86 | 1.95 / 1.96 | 2.02 | SAME |
| sh2 | **1.82 / 1.80** | 1.93 / 1.93 | **1.98** | SAME |
| base2 | 1.87 / 1.86 | 1.95 / 1.96 | 2.01 | SAME |
| sh2b | **1.81 / 1.81** | 1.94 / 1.97 | **1.98** | SAME |

```text
VERDICT base vs toss OK decode -1.8% prefill +0.0% md5 SAME errors 0
VERDICT sh2 vs toss OK decode -2.7% prefill +0.0% md5 SAME errors 0
VERDICT base2 vs toss OK decode -1.3% prefill +0.0% md5 SAME errors 0
VERDICT sh2b vs toss OK decode -2.9% prefill -0.8% md5 SAME errors 0
VERDICT baseline toss errors 0 text ok
```

- **a2 の中央値: base 1.86 / base2 1.865 に対して sh2 1.81 / sh2b 1.81 = -2.7%。**
  base と base2 の床は 0.3% で、4 サンプルずつが完全に分離している
  (base 側 1.86/1.86/1.87/1.86 < sh2 側 1.82/1.80/1.81/1.81)。
- prefill: c2000_cold base 2.81/2.79 / base2 2.80/2.78 に対して sh2 2.79/2.78、
  c8000_cold base 6.78 / base2 6.76 に対して sh2 6.77 = **不変 (差は 0.1% 台)**。
- md5: 5 COND x 2 ラウンド全部 SAME (`a:65ededc3 b:f18c75fc c2000_cold:a91e17da
  c2000_turn2:5a8f5488 c8000_cold:e2292eb6 c8000_turn2:561c6da8`)。

### 3.2 t319b (`a0c729c5d`、6 COND、load_before 3.13) — 独立ラウンド + 診断

COND = toss(`SH_V5=0`) / base(`SH_V5=0`) / sh2(既定) / base2(`SH_V5=0`) / sh2b(既定) /
diag(`SH_V5=0` + `DENNO_FULL_DEC_DBG=1 DENNO_GDN_DEC_KPROF=1`)。

| cond | a2 r1 / r2 | c2000_turn2 r1 / r2 | c8000_turn2 (r1) | md5 |
| --- | --- | --- | --- | --- |
| base | 1.86 / 1.86 | 1.96 / 1.96 | 2.02 | SAME |
| sh2 | **1.81 / 1.84** | 1.93 / 1.93 | **1.98** | SAME |
| base2 | 1.89 / 1.86 | 1.96 / 1.95 | 2.02 | SAME |
| sh2b | **1.82 / 1.81** | 1.93 / 1.93 | **1.98** | SAME |
| diag | 1.86 / 1.86 | 1.97 / 1.97 | 2.01 | SAME |

```text
VERDICT base  vs toss OK decode +0.2% prefill +0.0% md5 SAME errors 0
VERDICT sh2   vs toss OK decode -0.9% prefill +0.0% md5 SAME errors 0
VERDICT base2 vs toss OK decode +0.0% prefill +0.0% md5 SAME errors 0
VERDICT sh2b  vs toss OK decode -0.2% prefill -0.8% md5 SAME errors 0
VERDICT diag  vs toss OK decode +0.0% prefill -0.8% md5 SAME errors 0
VERDICT baseline toss errors 0 text ok
```

- **a2 の中央値: base 1.86 / base2 1.86 に対して sh2 1.815 / sh2b 1.815 = -2.4%**
  (t319a の -2.7% と合わせて **2 ラウンドとも 2% 超**)。14k 後は 2.02 -> 1.98 = -2.0%、
  2k 後は 1.96 -> 1.93 = -1.5%。
- prefill: c2000_cold 2.79 / 2.79 -> 2.78 / 2.78、c8000_cold 6.78 / 6.76 -> 6.76 / 6.77 = 不変。
- md5 は 6 COND x 2 ラウンド全部 SAME。テキスト (A): `**B-tree** (specifically, the **B+ tree) is
  the fundamental data structure used by most modern relational databases` — base と同一。
- この回は load_before 3.13 で全体がわずかに揺れた (base2 の r1 が 1.89) が、a2 の 4 対 4 は
  まだ分離している (base 側の最良 1.86 > sh2 側の最悪 1.84 ではない: 1.84 が 1 本だけある。
  中央値で読む)。

### 3.3 まとめ (採否の判定)

| 指標 | t319a | t319b | 判定 |
| --- | --- | --- | --- |
| 短い decode (a2、2 ラウンドの中央値) | -2.7% | -2.4% | **規則 (2% 以上) を満たす** |
| 14k 後の decode (c8000_turn2) | -2.0% | -2.0% | 悪化なし |
| 2k 後の decode (c2000_turn2) | -1.0% | -1.5% | 悪化なし |
| prefill (c2000_cold / c8000_cold) | +0.0% / +0.0% | +0.0% / 0.0% | 悪化なし (1% 以内) |
| md5 / errors / text | SAME / 0 / ok | SAME / 0 / ok | 健全 |

## 4. full 層のホスト内訳 (P316 §8 の「80 us の正体」)

`DENNO_GDN_DEC_KPROF=1` のときだけ出る `host step` 表 (us は 1 回あたり、窓 = 1 層 1 回)。

| step | t319a band 0 (短い) | t319b band 0 (短い) | t319a band 3800 | t319a band 14000 |
| --- | --- | --- | --- | --- |
| `fd_qkv` | 10.4 | 8.4 | 7.8 | 7.8 |
| `fd_mirror` | 9.1 | 7.1 | 5.4 | 5.2 |
| `fd_attn` | 8.8 | 7.1 | 10.2 | 7.6 |
| `fd_prep` | 8.0 | 5.3 | 4.0 | 3.9 |
| `fd_wo` | 4.1 | 3.6 | 3.4 | 3.4 |
| `fd_gate` | 4.0 | 3.4 | 3.3 | 3.4 |
| `fd_d2h` | 3.6 | 2.9 | 2.7 | 2.6 |
| `fd_tail` | 1.0 | 0.9 | 0.8 | 0.8 |
| **ホスト合計** | **49.0** | **38.7** | **37.6** | **34.7** |
| `fd_sync` (GPU 待ち) | 67.9 | 75.7 | 103.8 | 165.7 |

- 1 層の wall は短い帯で 176-234 us (kern 94-134 / gap 49-74 / wait 82-120)、14k で
  197-219 us (kern 171-194 / gap 12-19 / wait 161-177)。**14k は GPU 律速**、
  短い帯はホストと GPU が半々。
- ホスト合計 39-49 us/層 x 12 層 = **0.5-0.6 ms/token (短い decode の 3-4%)** だが、
  **最大の 1 項でも 10 us/層 = 0.12 ms/token (0.8%)** なので、ここを削っても 2% には届かない。
  P316 が「80 us/層 の正体」と呼んだものは、この内訳 (合計 39-49) + 起動の固定費
  (launch 14-26 us) + `gap` の計測自身の固定費に分かれる。**80 us が 1 つの块ではない。**

### 4.1 短い文脈では dev 経路が 44% の層しか使われていない (新しい事実)

- band 0 の 768 窓 (12 層 x 64) のうち、`fd_*` が揃っているのは **336 窓 (44%)**。
  残り 432 窓は従来 (CPU) 経路で、そこでは `cache_push 0.8` と `gate_mul 13.0 us` が測れる。
- `DENNO_FULL_DEC_DBG=1` の数え上げ (t319b diag): `calls=1537 gate=0 shape=0 ctx=0
  mirror=1537 plan=0 other=0` = **失敗理由は 100% が `mirror`** (KV ミラーの
  「全行シードが要る」)。band 3800 / 14000 では 768/768 が dev 経路。
- **これが得か損かは未計測**。短い文脈では CPU 経路のほうが安い可能性がある
  (dev 経路の固定費は 1 層 ~190-200 us で、`fd_sync` 68 us を含む)。
  ledger の `full` は両経路の混合で 122 us/層 (P310)。次タスクで
  `DENNO_Q4_ATTN_MIRROR_MB` を振るか、ミラーのシードを 1 トークン目以外でも
  安く済ませる手を入れて比べるのが筋。

## 5. 採否 (task の規則)

- 短い decode **-2.7% (t319a) / -2.4% (t319b)** = 2 ラウンドの中央値で 2% 超。
  14k 後 -2.0%、prefill 不変、他に 1% 超の悪化なし、errors 0、md5 SAME、テキスト健全。
  → **ADOPT: yes。`DENNO_DEC_SH_V5` の既定を 2 にした** (この commit)。
- 切り戻し: `DENNO_DEC_SH_V5=0` (従来の v3/v2)。`=1` (down も v5) は P310 の実測で
  `=2` よりわずかに悪いので既定にしない。

## 6. 数値の扱い

- v5 GEMV は 4 アキュムレータで加算するので v3 とは丸め順が違う (決定性は保つ)。
  実測の md5 は t319a / t319b の全 COND・全ラウンドで SAME。よって「丸めは変わったが
  出力トークンは変わらなかった」。
- 効果の読み方: ラボ集計の decode 中央値は 0.1 ms/tok 刻みで、しかも文脈の違う要求を混ぜる。
  **同じプロンプトの warm a2 の秒**と 14k 後の秒で読んだ。床は base/base2 の 0.3%。
- 測っていない数字は書いていない。推定は「推定」と書いた。

## 7. テスト

最終 commit (int-auto `0d3cd40da` を merge した後) で:

- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`
  → **1216 passed / 0 failed / 102 ignored**。
- `cargo test --release -p denno-engine --features cuda --test kernel_invariants`
  → **99 passed / 0 failed** (merge 前の `a0c729c5d` では 97。br320 が 2 本足した)。
- 既知の赤 (`qwen4exp_decode spec_batch_stops_at_eos`、`qwen4exp_params` の validate 2 件)
  は踏んでいない。

## 8. GPU で確認してほしいこと

- `DENNO_DEC_SH_V5=0` で従来に戻ること (A/B の切り戻し) と、本番の長い文脈で
  md5 / テキストが変わらないこと (lab の depth8k では全 COND SAME)。
- **短い文脈で dev 経路が 44% しか使われない件** (4.1)。`DENNO_FULL_DEC_DBG=1` で
  `mirror` の数がそのまま出る。ここが本命の次の 1 つで、dev 経路と CPU 経路の
  1 層コストを直接比べる計器が要る (今回は `fd_*` を dev 側にしか入れていない)。
- full 層のホスト内訳は合計 39-49 us/層で、1 項最大 10 us。**ここから 2% は取れない**
  (推定: 全部消しても 0.6 ms/token = 4%、現実には 1% 台)。
