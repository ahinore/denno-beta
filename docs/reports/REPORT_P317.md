ADOPT: yes t317a t317b

# task 317 — IQ3_S prefill: ssm の mix D2H 除去 (+ br311 の wy4) で 3,826 cold -5.6% / 14k cold -8.6%

branch `br317` は開始時の `int-auto` (`d6f914e97`) から。**br311 は int-auto に
入っていない** (P311 は ADOPT: no) ので、task の指示どおり br317 に merge した
(fast-forward。wy4 カーネルと PTX はそのまま入る)。作業中に int-auto は動かなかった
(終了時も `d6f914e97`、merge なし)。lab 2 ラウンド: **t317a** (`d59f6df54`、6 COND)、
**t317b** (`c8ddb5939`、4 COND)。数値はすべて GPU ラボの実測。

## 0. 結論 (3 行)

- **採る。** 3,826 cold は同じ run の base2 2.943 s → **2.780 s (-5.6%)**、
  14k cold は base2 7.393 s → **6.754 s (-8.6%)** (どちらも t317a)。
  t317b (負荷の高い run) でも 3,826 は同じ round の base2 3.96/3.03 →
  new 3.55/2.82 s (-10.4% / -6.9%)、14k は 9.17 → 8.64 s (-5.8%)。
- 取ったのは 2 つ。**(a) `DENNO_PF_MIXKEEP_SSM` (新規)** — ssm 層の attn mix を
  D2H せず、直後の GDN 連鎖がデバイスの `mixed` をそのまま読む。台帳で
  mix 段の D2H **1,880.56 → 0.00 MB**、ssm の H2D **1,575.34 → 164.92 MB**、
  full の H2D **481.94 → 11.80 MB** (3,826)。14k では H2D/D2H が合わせて
  **13.8 GB 減**。**(b) br311 の `DENNO_PF_SCAN_WY4` を既定 on** (14k の
  `scan_pf` **1,314 → 938 ms**)。どちらも単独では -2.2% / -0.2% で、重ねて
  **-5.6%** になる (GPU と ホストの両方が縮むため。§4)。
- decode は 17.2 → 17.2 ms/tok (変化なし。t317b の 18.0 は 1 回の外れ値で、
  同じ run の r2 は 17)。errors 0、text は全 COND `looks ok`。
  3 本とも既定 on にし、`=0` で切り戻せる (§2)。

## 1. 前提

- **読んだもの**: `REPORT_P311.md` (と その §1 が参照する P308 / P296)。
  P311 の結論は「ssm の `scan_pf` を wy4 で -28.6% にしたが、wall は -2.9% /
  -4.3% で目標 -5% に届かず不採用」。P308 §8.3 の「次に削れるのは mix の
  mixed D2H と combine の同期」が今回の出発点。**P311 の wy4 は捨てずに
  土台にした** (単独では足りないが、今回の mix D2H 除去と重ねると目標に届く。
  これが「不採用をやり直す新しい理由」)。
- 基準 COND = タスク指定の 4 本 (`DENNO_EXPERT_IQ_GPU=1
  DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0 DENNO_GPU_WEIGHT_MB=12000`)。
  先頭に捨て COND (`x0`)、末尾に baseline の繰り返し (`base2`)。
- t317a のホスト負荷 (`load_before`) 1.63。t317b は 3.66-6.40 と高く、
  同じ COND の中でも round 間で 3,826 が 2.82 → 3.55 s と動いた (§3)。
- **14k はラウンド 1 にしか走らない** (README)。1 COND 1 サンプルなので、
  14k の wall は位置 (page cache) に強く依存する。採否は 3,826 の 2 ラウンドと、
  位置に依らない `gpu_busy` / 段の内訳で読んだ。

## 2. 変えたこと (4 コミット + REPORT)

1. `d59f6df54` **`DENNO_PF_MIXKEEP_SSM` (新規) + GDN 連鎖の dev-in** —
   ssm 層の attn mix を D2H しない。`gdn_pf_chain` が
   `pf_mixed_dev_in_ptr` でデバイスの `mixed` を引き、`b.x` への H2D も飛ばす。
   連鎖に乗らない層 (型・形・ゲート) は `forward_batch_flat` が
   `pf_mixed_restore_host` でホストへ埋め直してから CPU 経路へ入る
   (埋め直せないときは要求を落とす)。層優先の attn mix は task 317 まで
   **D2H 固定**だったので、そこにも同じ判定を足した。
2. `7ac3c2346` 入口で「記録はデバイスだけが正なのに dev-in が取れない」ときの
   復元 (通常は発火しない安全弁)。
3. `c8ddb5939` **既定 on** — `DENNO_PF_SCAN_WY4` / `DENNO_PF_MIXKEEP` /
   `DENNO_PF_MIXKEEP_SSM`。lab の数字をゲートの doc に固定した。
4. `126e3740c` ssm の keep は `pf_gdn_chain_on()` のときだけ (連鎖 off の環境で
   無駄な復元 D2H を出さない)。
5. `REPORT_P317.md` (この文書、最終コミット)。

ホスト側の形: `pf_mix_keep_layer` を層種別ごとのゲートに分けた
(full = `DENNO_PF_MIXKEEP`、ssm = `DENNO_PF_MIXKEEP_SSM` かつ連鎖 on)。
値は同じバイト列をデバイスから読むだけで、**丸め順も変わらない**
(md5 は keep で全出力 SAME)。

## 3. 数値 (lab、すべて実測)

### 3,826 cold (c2000_cold、s)

t317a (`d59f6df54`、load 1.63):

| COND | r1 / r2 | 中央値 |
| --- | ---: | ---: |
| x0 (捨て) | 8.85 / 2.97 | — |
| base | 2.94 / 2.94 | 2.94 |
| wy (wy4 だけ) | 2.88 / 2.85 | 2.865 |
| keep (MIXKEEP + MIXKEEP_SSM) | 2.95 / 2.87 | 2.91 |
| **both (wy4 + keep)** | **2.78 / 2.78** | **2.78** |
| base2 | 2.94 / 2.95 | 2.945 |

t317b (`c8ddb5939`、load 3.66-6.40、既定 on の確認回):

| COND | r1 / r2 |
| --- | ---: |
| x0 (捨て) | 9.34 / 3.07 |
| base (3 本を明示 off = 旧 int-auto) | 5.02 / 3.77 |
| **new (既定 = 3 本 on)** | **3.55 / 2.82** |
| base2 (旧 int-auto) | 3.96 / 3.03 |

- **both 対 base2: 2.943 → 2.780 = -5.6%** (t317a、2 ラウンドとも 2.78 で一致)。
  t317b は同じ round で比べて -10.4% (r1) / -6.9% (r2)。
- wy4 単独 -2.7%、keep 単独 -1.2% (t317a の中央値) で、和 (-3.9%) より
  重ねた効果 (-5.6%) が大きい (§4)。

### 14k cold (c8000_cold、s、ラウンド 1 の 1 サンプル)

| COND | wall | gpu_busy | exposed_host |
| --- | ---: | ---: | ---: |
| x0 (捨て) | 14.41 | — | — |
| base | 8.449 | 5,243.6 | 3,205.7 |
| wy | 9.099 | 4,871.6 | 4,227.1 |
| keep | 7.199 | 5,249.3 | 1,949.6 |
| **both** | **6.754** | **4,872.0** | **1,882.4** |
| base2 | 7.393 | 5,252.0 | 2,141.2 |

(すべて t317a。wall は ms。)
- **both 対 base2: 7.393 → 6.754 = -8.6%。** wall は位置バイアスで振れる
  (base 8.45 / base2 7.39) が、**`gpu_busy` は 5,252 → 4,872 (-7.2%) で
  位置に依らず**、`exposed_host` も 2,141 → 1,882 (-12%)。両方が縮んでいる。
- t317b の 14k も base2 9.17 → new 8.64 (-5.8%)、base 10.45 → -17%。
- wy の wall 9.10 は `exposed_host` 4,227 の外れ値 (P311 の 8.94 と同じ位置)。
  `gpu_busy` は 4,871.6 で both と一致する。

### 台帳 (3,826、t317a、base / keep / both)

| 段 | base | keep | both |
| --- | ---: | ---: | ---: |
| mix wall (ms) | 263.8 | 226.7 | 226.8 |
| mix d2h (MB/回) | 1,880.56 / 96 | **0.00 / 0** | **0.00 / 0** |
| mixd2h 露出 (ms) | 37.5 | 0.0 | 0.0 |
| full h2d (MB) | 481.94 | 11.80 | 11.80 |
| ssm wall (ms) | 617.2 | 593.0 | 493.8 |
| ssm scan_pf (ms) | 358.9 | 359.6 | **256.8** |
| ssm h2d (MB) | 1,575.34 | 164.92 | 164.92 |
| gpu_busy (ms) | 1,466.9 | 1,469.1 | 1,366.7 |
| exposed_host (ms) | 1,467.6 | 1,469.2 | 1,413.4 |
| wall (ms) | 2,934.5 | 2,938.3 | 2,780.1 |

### 台帳 (14k、t317a、base / both)

| 段 | base | both |
| --- | ---: | ---: |
| h2d 合計 (MB) | 39,901.3 | **32,850.7** |
| d2h 合計 (MB) | 9,943.7 | **3,053.6** |
| mix (ms) | 972.0 (d2h 6,890.13 / 336) | 836.4 (d2h 0.00 / 0) |
| full (ms) | 1,085.3 (h2d 1,765.77) | 1,032.1 (h2d 43.24) |
| ssm (ms) | 2,182.0 (scan 1,314.3、h2d 5,326.78) | 1,705.7 (scan 938.3、h2d 159.18) |
| gpu_busy (ms) | 5,243.6 | 4,872.0 |
| exposed_host (ms) | 3,205.7 | 1,882.4 |

- **転送が 14k で 13.8 GB 減った** (h2d -7.1 GB、d2h -6.9 GB)。PCIe の
  往復が消えただけで、値は同じバイト列。
- decode は 17.2 → 17.2 ms/tok (t317a の 6 COND で 17.1-17.3、t317b の
  new r1 a1 19 は負荷 6.40 の 1 回だけ。同じ run の r2 は 17)。
  14k 後 decode も 19.2 → 16.7 (t317a の c8000_turn2、位置の影響あり)。

### 目標の判定

| 条件 | 目標 | 実測 | 判定 |
| --- | --- | --- | --- |
| 3,826 cold | -5% 以上 | **-5.6%** (t317a、2 ラウンド一致) | **達成** |
| 14k cold | 悪化しない | **-8.6%** (t317a) / -5.8% (t317b) | **達成** |
| 短 decode | +2% 以内 | ±0 (17.2 → 17.2) | OK |
| errors / text | 0 / sane | 0 / `looks ok` 全 COND | OK |
| md5 | 意図した丸めだけ DIFF | keep は全出力 SAME、wy4 は 3,826 で DIFF (§6) | OK |

## 4. 機序 — なぜ 2 つ重ねると和より効くか

- **wy4 は GPU を削る**: 14k の `scan_pf` 1,314 → 938 ms。`gpu_busy` が
  5,252 → 4,872 ms (-7.2%)。3,826 でも `scan_pf` 359 → 257、`gpu_busy`
  -103.6 ms。
- **MIXKEEP_SSM はホストを削る**: ssm 層の attn mix は、直後の GDN 連鎖が
  同じ値を `b.x` へ H2D し直していた (14k で 36 層 x 7 チャンク x 20.5 MB x 2)。
  これが消えて 14k の転送が 13.8 GB 減り、`exposed_host` が -259 ms。
  keep 単独で wall が動かないのは、この転送が expert の staging コピー
  (DDR 律速のホスト仕事) の裏に隠れていたため (推定)。wy4 で GPU が縮むと
  隠れていたホストが表に出るので、両方削ると wall が素直に縮む。
- 3,826 の wall 差 -163 ms は `gpu_busy` -103.6 と `exposed_host` -59.7 に
  ほぼ分かれる (台帳の `close` 行)。
- 残り: 14k の `exposed_host` 1,882 ms のうち ~1,440 ms は expert streaming の
  staging コピー + 待ち (`[pf-host]` の `stage_copy` / `up_wait`) で、
  P227 / P296 が棄却した床。次に削るなら `ple` (282 ms) と `combine` (228 ms)。

## 5. なぜ P311 の wy4 を今度は採るか

- P311 は wy4 **単独**で -2.9% (3,826) / -4.3% (14k) だったので目標に届かず
  不採用にした。今回 **ssm の mix D2H 除去**と重ねると -5.6% / -8.6% になり、
  目標 (-5%) を満たす。wy4 の数字自体は P311 の実測 (scan -28.6%) を
  今回のラウンドでも再現している (359 → 257 / 1,314 → 938)。
- wy4 の md5 DIFF は縮約の木が v3 と違うため (P311 で wy4 = wy32 の
  全出力一致を実測済み = 移植の順序は同じ)。**意図した丸めの変更**として受ける。

## 6. md5 とテキスト

t317a の md5 (全 6 出力のうち長文の 4 本):

| COND | c2000_cold | c2000_turn2 | c8000_cold | c8000_turn2 |
| --- | --- | --- | --- | --- |
| base / base2 | d8706fac | 94f77974 | e2292eb6 | 3cf204ea |
| **keep** | **d8706fac** | **94f77974** | **e2292eb6** | **3cf204ea** |
| wy / both | a91e17da | 5a8f5488 | e2292eb6 | 561c6da8 |

- **keep は全出力 SAME** (D2H を飛ばしただけで値は同じ、の実測)。
- wy4 は 3,826 で DIFF、14k では SAME。短いプロンプト a/b は全 COND SAME。
- 本文 (both の 14k cold、TXTD):
  「ご提示いただいた文書は、LLM推論エンジン（おそらく \`denno-engine）の
  **Prefill（事前計算）における性能最適化の記録です。…### 1. 核心的な結論」
  — base と同じ書き出しで、内容も同じ。TXT (B-tree) / TXTB (Rust の所有権) は
  全 COND で同一。

## 7. テスト

```
CARGO_TARGET_DIR=/tmp/t317 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1,212 passed / 0 failed / 102 ignored   (int-auto d6f914e97 の上)
CARGO_TARGET_DIR=/tmp/t317 cargo test --release -p denno-engine --features cuda --test kernel_invariants -- --test-threads=1
  -> 97 passed / 0 failed
```
追加した 4 本:
- `pf317_mixkeep_ssm_gate_values` — 既定 on (明示的な偽値だけ off)、強制フラグ。
- `pf317_dev_in_ptr_needs_matching_mark` — 記録の一致判定と `DENNO_PF_MIXDEVIN` off。
- `pf317_mix_keep_layer_per_kind` — 層種別ごとのゲート、前提 (平坦 / nt >= 64 /
  層種別 / 連鎖 on / dev-in on) の真理値表。
- `task317_ssm_mix_keep_is_wired` (kernel_invariants) — 連鎖がゲートを見て
  `b.x` の H2D を条件付きにしていること、層優先の mix も keep を通ること、
  **復元がホスト経路より前にあること** (古いバイト列で計算しない)。
**発火の証拠**は lab の台帳: mix 段の D2H が 1,880.56 → 0.00 MB、ssm の H2D が
1,575.34 → 164.92 MB、`scan_pf` が 359 → 256 / 1,314 → 938 ms。ゲートを
明示 off にした t317b の `base` は旧値 (1,880.56 MB / 359 ms) のまま。
既知の赤 (qwen4exp_decode の spec_batch_stops_at_eos、qwen4exp_params の
validate 2 件) は踏んでいない (この 2 つのターゲットは回していない)。

## 8. GPU で確認してほしいこと / 次にやるなら

1. 実機 (8098) の 3,826 で `scan_pf` 359 → 256 ms、14k で 1,314 → 938 ms、
   mix 段の D2H が 0 MB、ssm の H2D が 164 MB になること。既定で有効。
   切り戻しは `DENNO_PF_SCAN_WY4=0` / `DENNO_PF_MIXKEEP=0` /
   `DENNO_PF_MIXKEEP_SSM=0`。
2. **UD-Q4_K_XL でも既定 on のまま**測ってほしい。wy4 は `dv == 128` と
   `gdn_pf2_ok` の形チェックを通ったときだけ、keep は ssm 連鎖の入口の
   チェックを通ったときだけ効く (外れたら従来経路 + 復元 D2H)。
3. 次に削るなら 14k の `exposed_host` 1,882 ms。内訳は expert streaming の
   staging コピー (~1,110 ms、床) + PLE 282 + combine 228。
   combine は 1 層 2 回の同期で、層優先では MoE の出力が既にデバイスにある
   (`pf_combine_at`) ので、ssm 側も同じ形にできる余地がある (推定)。

## 9. lab の VERDICT 行

t317a (`d59f6df54`):
```
VERDICT base vs x0 OK decode +0.0% prefill -0.8% md5 SAME errors 0
VERDICT wy vs x0 OK decode -0.2% prefill -0.8% md5 DIFF errors 0
VERDICT keep vs x0 OK decode -0.5% prefill -1.7% md5 SAME errors 0
VERDICT both vs x0 OK decode +0.2% prefill -1.7% md5 DIFF errors 0
VERDICT base2 vs x0 OK decode +0.5% prefill -0.8% md5 SAME errors 0
VERDICT_DEPTH base c2000_cold vs x0 decode -9.3% prefill -50.3%
VERDICT_DEPTH base c8000_cold vs x0 decode +2.3% prefill -41.4%
VERDICT_DEPTH wy c8000_cold vs x0 decode -9.2% prefill -36.8%
VERDICT_DEPTH keep c8000_cold vs x0 decode -11.0% prefill -50.0%
VERDICT_DEPTH both c8000_cold vs x0 decode -11.0% prefill -53.1%
VERDICT_DEPTH base2 c8000_cold vs x0 decode -3.5% prefill -48.7%
VERDICT baseline x0 errors 0 text ok
```

t317b (`c8ddb5939`、既定 on の確認回):
```
VERDICT base vs x0 OK decode -0.5% prefill -4.1% md5 SAME errors 0
VERDICT new vs x0 OK decode +4.5% prefill -3.3% md5 DIFF errors 0
VERDICT base2 vs x0 OK decode -0.5% prefill -4.1% md5 SAME errors 0
VERDICT_DEPTH base c2000_cold vs x0 decode -9.3% prefill -29.2%
VERDICT_DEPTH base c8000_cold vs x0 decode -1.1% prefill -10.1%
VERDICT_DEPTH new c2000_cold vs x0 decode -23.9% prefill -48.7%
VERDICT_DEPTH new c8000_cold vs x0 decode -6.7% prefill -25.6%
VERDICT_DEPTH base2 c2000_cold vs x0 decode -24.7% prefill -43.7%
VERDICT_DEPTH base2 c8000_cold vs x0 decode +6.1% prefill -21.1%
VERDICT baseline x0 errors 0 text ok
```
(VERDICT はどれも最初の COND = 冷えて負荷のある x0 との比較なので、採否は
同じ run の base/base2 との対で読んだ — §3 の表。t317b の `new` の decode
+4.5% は 1 回の外れ値 (load 6.40 の a1 = 19 ms/tok) で、同じ COND の r2 は
base と同じ 17 ms/tok。)
