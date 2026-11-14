ADOPT: yes t308b t308c

# task 308 — IQ3_S prefill: router をデバイスで 1 回 + full 層の出力をデバイスに残す (-5.2%)

branch `br308` は開始時の `int-auto` (`02171944a` = `merge: br300`) から。`br300` は
既に int-auto に入っていたので merge していない。作業中に int-auto は動かず
(最後に fetch し直して `02171944a` のまま)、merge は無い。lab 3 ラウンド:
**t308a** (`37a9036fd`、6 COND、ゲート既定 off で切り分け)、**t308b**
(`37a9036fd`、5 COND、両ゲート同時)、**t308c** (`605082678`、5 COND、既定 on の確認)。
数値はすべて GPU ラボの実測。

## 0. 結論 (3 行)

- **採る。3,826 cold prefill が base 3.128 → 2.968 s (-5.1%、t308b)、
  old 3.110 → new 2.945 s (-5.3%、t308c) で目標 (-5%) を満たす。** 14k cold は
  -5.7% (t308b) / -7.0% (t308c、host 負荷あり)、decode は短 ±0%、14k 後 ±0%、
  errors 0、テキスト sane。
- 取ったのは 2 つ。**(a)** 層優先 MoE の router (logits → softmax → top-k →
  再正規化) を `pf_router_dev_topk` (task 225 のカーネル) でタイル全トークン
  一括に回し、`probs [nt][n_expert]` を降ろさない `DENNO_PF_LM_ROUTER_DEV`。
  **(b)** full 層の出力投影のあと D2H せず、直後の combine に同じバッファを
  読ませる `DENNO_PF_FULL_DEVRES` (ssm 層の `DENNO_PF_ATTN_DEVRES` の full 版)。
- 台帳で機序が確認できる (14k): h2d 41,636.8 → 39,901.3 MB、
  d2h 12,987.8 → 9,943.7 MB (-4.78 GB)、launches 9,945 → 9,993
  (+48 = 48 層ぶんの router カーネル)。3,826 でも h2d -455.9 MB /
  d2h -830.9 MB で、これは「12 full 層 × 39.2 MB」と「48 層 × 7.84 MB の probs」
  にちょうど一致する。

## 1. 前提

- 基準 COND はタスク指定の 4 本 (`DENNO_EXPERT_IQ_GPU=1
  DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0
  DENNO_GPU_WEIGHT_MB=12000`) = int-auto の既定。
- タスク 300 の確認 (`DENNO_PF_PLE_REUSE=0` との比較) は t308a と t308b で取った:
  3,826 の 2 ラウンド目で base 3.13 → ple0 3.20 s (**PLE_REUSE が -1.9%**)。
  t300b の -1.5% と同じ向き・同じ桁で、int-auto に載っている値が再現した。
  14k は t308a で ple0 8.38 / base2 8.00 と出たが、下の page cache の理由で
  採否には使っていない (3,826 の 2 サンプルで足りる)。
- **int-auto の既定経路は層優先 (`DENNO_PF_LM_LAYER` は task 265 から既定 on)**
  なので、router のホスト仕事は `moe_ffn_quant_batch_lm_dev_at` の中にある
  (`probs` を作って `router_softmax_rows` + `router_topk_rows`)。
  チャンク優先の `DENNO_PF_ROUTER_DEV` (task 228、既定 on) は層優先を通らない。
- **page cache のドリフト**: t308a の 14k cold は
  x0 10.86 → base 10.71 → ple0 8.38 → full 9.63 → rdev 7.71 → base2 8.00 s で、
  COND の順に下がった (78 GB のモデルが 59 GiB の RAM に収まらないので、
  走るたびに expert のページが溜まる)。**14k は位置で 3 割動く**ので、
  t308a の 14k 列は COND 間の比較に使っていない (台帳のバイト数だけ使う)。
  t308b/t308c は base/base2 を両端に置いて 5 COND に減らし、ドリフトを抑えた。
- t308c は `load_before=6.40` (他のホスト仕事) で、**1 ラウンド目の値は全部
  壊れている** (x0 17.74 s、old 6.23 s)。採否は 2 ラウンド目 (同一 COND の
  2 回目) で読んだ。t308c の `new` の c8000_cold decode 38.9 ms/tok も同じ理由の
  外れ値 (new2 は 20.3、old/old2 は 20.9/20.0)。decode の採否は t308b を使う。

## 2. 変えたこと (2 コミット)

1. `37a9036fd` **`DENNO_PF_LM_ROUTER_DEV`** (新設) — 層優先 MoE の router を
   `crate::cuda::pf_router_dev_topk` で回す。logits は `dno_gemm_f32_ba`
   (無ければ `dno_gemm_f32_mt`)、softmax + top-k + 再正規化は
   `dno_router_softmax_topk` (`kernels_router225`、task 225 でコミット済み)。
   **カーネルは既存なので `.cu` / `.ptx` は触っていない** (PTX 再生成も無し)。
   降りるのは `sels` / `weights` / `rowsum` だけ。role RMS の `router` 値は
   カーネルの `rowsum` (選んだ k 個の index 昇順の逐次和 = ホストの下の和と同じ)
   を使う。`expert_mask` があるとき・`nt < 64`・形が前提外のときは従来の
   ホスト経路へ fail closed。
2. `37a9036fd` **`DENNO_PF_FULL_DEVRES`** (新設) — `pf_full_out` の出力投影の
   あと、`pf_attn_dev_requested()` が立っていれば D2H せず
   `pf_attn_dev_store(ob.ptr(), nt, e)` に記録し、直後の `pf_combine_dev` が
   同じバッファを読む。値は 1 ビットも変わらない (同じカーネル・同じ入力)。
   `roles` を取るときは要求が立たないので従来どおり D2H する。
3. `605082678` 両ゲートを**既定 on** に (切り戻しは `=0`)。真理値表テストも更新。

## 3. 数値 (lab、すべて実測)

### t308a (`37a9036fd`、ゲート off、6 COND、load_before=1.38) — 切り分け

3,826 cold (2 ラウンド目 = 暖まった後。1 ラウンド目は先頭ほど冷える):

| COND | 3,826 r2 | 14k (r1、位置バイアス有) | 台帳 d2h / h2d (14k) |
| --- | ---: | ---: | --- |
| x0 (捨て) | 3.13 | 10.86 | — |
| base | 3.13 | 10.71 | 12,987.8 / 41,636.8 MB |
| ple0 | 3.20 | 8.38 | 12,987.8 / 41,636.8 MB |
| **full** | **3.06** | 9.63 | **11,265.2 / 39,914.3 MB** |
| **rdev** | **3.00** | 7.71 | **11,666.3 / 41,623.8 MB** |
| base2 | 3.12 | 8.00 | 12,987.8 / 41,636.8 MB |

- `full` の h2d / d2h が**両方 -1,722 MB** = 12 full 層 × 143.5 MB (14k)。
  combine の H2D と層出力の D2H が消えたことの直接の証拠。
- `rdev` の d2h が -1,321 MB = 48 層 × 28.7 MB (probs) 弱。
  launches は 9,945 → 9,993 (+48)。
- **md5: `full` 単独は全出力 SAME、`rdev` 単独で DIFF**。つまり
  `DENNO_PF_FULL_DEVRES` は値が完全に同じで、`DENNO_PF_LM_ROUTER_DEV` だけが
  softmax の和の結合順で丸めを動かす。狙いどおりの切り分け。

### t308b (`37a9036fd`、5 COND、load_before=1.51) — 両ゲート同時

| COND | 3,826 r1 / r2 | 14k cold | 短 decode | 14k 後 decode |
| --- | --- | ---: | ---: | ---: |
| x0 (捨て) | 5.52 / 3.14 | 11.94 | 17.8 | 19.7 |
| base | 3.84 / 3.13 | 10.30 | 17.9 | 17.6 |
| ple0 | 3.77 / 3.19 | 10.05 | 17.9 | 16.2 |
| **both** | **3.00 / 2.94** | **7.54** | **17.9** | **17.3** |
| base2 | 3.13 / 3.13 | 8.00 | 18.0 | 17.3 |

- **3,826: base2 の台帳 3,128.5 ms (3,132.3 / 3,124.6) → both 2,968.0 ms
  (2,995.2 / 2,940.7) = -5.1%**。
- **14k: base2 7,995.2 → both 7,537.9 ms = -5.7%**。
- decode: 短 17.9 → 17.9 (±0%)、14k 後 17.3 → 17.3 (±0%)。
- 台帳 (3,826): base2 h2d 25,725.8 / d2h 3,700.6 / launches 3,459 →
  both 25,269.9 / 2,869.7 / 3,507。-455.9 MB は 12 full 層の combine の H2D
  (14k では 12 × 143.5 = 1,722 MB ちょうど。3,826 は最終チャンクが 242 トークン
  なので少し小さい)、-830.9 MB は層出力 455.9 + router の probs 376.0
  (48 層 × 7.84 MB)。syncs は 846 で不変。

### t308c (`605082678`、既定 on の確認、5 COND、load_before=6.40)

| COND | 3,826 r1 / r2 | 14k cold | 短 decode | 14k 後 decode |
| --- | --- | ---: | ---: | ---: |
| x0 (捨て = 新既定) | 17.74 / 3.70 | 13.89 | 18.0 | 19.6 |
| old (`=0 =0`) | 6.23 / 3.11 | 11.07 | 18.0 | 19.8 |
| **new (既定)** | 3.94 / 2.94 | 9.77 | 17.9 | 38.9 (負荷の外れ値) |
| **new2 (既定)** | 4.38 / 2.95 | 10.66 | 17.9 | 18.9 |
| old2 (`=0 =0`) | 4.37 / 3.11 | 10.91 | 17.9 | 19.1 |

- **3,826: old の台帳 3,109.5 ms (3,106.1 / 3,112.9) → new 2,944.8 ms
  (2,943.3 / 2,946.3) = -5.3%**。2 つの独立した COND が 1 ms 以内で一致。
- **14k: old 10,986.4 ms (11,065.1 / 10,907.7) → new 10,213.4 ms
  (9,764.6 / 10,662.2) = -7.0%** (負荷で ±5% 動く回なので方向だけ読む)。
- decode: 短 18.0 → 17.9、14k 後 19.1 → 18.9 (new2)。`new` の 38.9 は
  `load_before=6.40` の外れ値で、同じバイナリの new2 は 18.9。
- 台帳の h2d / d2h は t308b と 1 バイトまで同じ (25,269.9 / 2,869.7)。

### 目標の判定

| 条件 | 目標 | 実測 | 判定 |
| --- | --- | --- | --- |
| 3,826 cold | -5% 以上 | **-5.1% (t308b) / -5.3% (t308c)** | **達成** |
| 14k cold | 悪化しない | -5.7% (t308b) / -7.0% (t308c) | OK |
| 短 decode | +2% 以内 | ±0% (17.9 → 17.9) | OK |
| 14k 後 decode | +2% 以内 | ±0% (17.3 → 17.3)、t308c は -1% | OK |
| errors / text | 0 / sane | 0 / sane | OK |
| md5 | 意図した丸めだけ DIFF | full 単独 SAME、router 単独で DIFF | OK |

目標は 5% ちょうどなので余裕は 0.1-0.3 ポイントしかない。t308b と t308c が
同じ -5.1/-5.3% を出したので、ノイズではなく効果と判断した。

## 4. 機序 — どこが消えたか

3,826 の台帳 (t308b、base2 → both) の差 -160.5 ms の内訳 (推定ではなく
バイト数と launch 数から):

| 消えたもの | バイト / 回数 | 3,826 の時間 (推定) |
| --- | --- | --- |
| full 層出力の D2H + combine の H2D | 39.2 MB × 12 × 2 = 941 MB | ~20 ms (47 GB/s) |
| router の probs D2H | 7.84 MB × 48 = 376 MB | ~10 ms |
| ホストの softmax + top-k | 3,826 × 512 × 48 | ~120 ms |
| **追加**: router カーネル 48 回 | 48 launch | ~5 ms |

合計の見積りは -145 ms で、実測は -160.5 ms。`[q4prof] prefill ms moe
router=76.7 experts=83.5 ...` は t308a の base の値で、ホストの router 仕事は
1 プロンプトで 100 ms 級という task 300 の台帳 (router 86 / topk 58) と整合する。

## 5. md5 とテキスト

- **md5 DIFF は `DENNO_PF_LM_ROUTER_DEV` の丸めだけ**。理由: デバイスの
  softmax は「レーン内の昇順部分和 → 固定木の結合」で和を取り、ホストは
  「行の昇順逐次和」なので、分母が 1 ulp 級に動く。選択規則 (値の降順・同値は
  小さい index・出力は index 昇順) と再正規化の加算順はホストと同一。
  これは task 225/228 がチャンク優先で既に既定 on にしているカーネルと同じ。
- `DENNO_PF_FULL_DEVRES` 単独では md5 が全出力 SAME (t308a) = 値は不変。
- 変わった md5 (t308c): c2000_cold `d4b6eefb` → `d8706fac`、
  c2000_turn2 `8ff9345f` → `94f77974`、c8000_turn2 `cc2c0e8f` → `3cf204ea`。
  c8000_cold は `e2292eb6` のまま。
- 本文 (t308c new2 の 14k cold):
  「ご提示いただいた文書は、LLM推論エンジン（おそらく \`denno-engine）の
  **Prefill（事前計算）フェーズにおけるパフォーマンス最適化の記録です。
  「計算量ではなく、データ転送とメモリコピーがボトルネックであった」
  — base と同じ書き出しで内容も同じ。TXT / TXTB (英語の B-tree、Rust の
  所有権) も全 COND で同じ。errors は全 COND 0。

## 6. なぜ -5% で止まるか

3,826 の 2.968 s の内訳 (t308b both の台帳 + t308a の切り分け):
MoE の転送 ~1.31 s (copy 0.90 + wait 0.41)、ssm 619 ms、mix 262、full ~220、
PLE ~147、combine ~65、MoE のホスト仕事 ~60 (router のホスト分がほぼ消えた)、
その他 ~280。`gpu_busy` は 1,507 ms で**触っていない** (ホスト側と往復だけ)。

1. 転送は 3 往復が DDR 帯域で決まる (P227 で 4 経路とも棄却済み)。ここは
   このタスクでも触っていない。
2. ssm / mix / full は GPU 律速。カーネル側の仕事なので `.cu` を触らない
   このタスクの範囲外。
3. ホスト側で残っているのは combine の同期 (1 層 2 回)、PLE の res 往復、
   `mix` の mixed D2H (3,826 で 1.88 GB、full 層のぶんは今も降ろしている)、
   `full` の入力側。`DENNO_PF_MIXKEEP` を層優先に広げるのは task 230 の
   棄却 (14k +4.2%) と同じ形なので触っていない。

## 7. テスト

```
CARGO_TARGET_DIR=/tmp/t308 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1,194 passed / 0 failed / 101 ignored   (int-auto 02171944a の上)
CARGO_TARGET_DIR=/tmp/t308 cargo test --release -p denno-engine --features cuda --test kernel_invariants -- --test-threads=1
  -> 95 passed / 0 failed
```
追加した 3 本:
- `pf_full_devres_gate_values` / `pf_lm_router_dev_gate_values` — 真理値表
  (既定 on、明示的な偽値だけ off)。
- `pf308_full_devres_store_peek_contract` — `pf_attn_dev_store` /
  `pf_attn_dev_peek` / `pf_attn_dev_clear` の契約 (形が違えば false、
  clear で消える)。GPU 不要。
分岐が発火した証拠は lab のログの `[pf308]` 行 2 本
(full 層の出力をデバイスに残す / 層優先 router: logits + softmax + top-k を
デバイスで 1 回)。t308b の both_1.log と t308c の new_1.log で確認した。

## 8. GPU で確認してほしいこと / 次にやるなら

1. 実機 (8098) の 3,826 級で `[q4ledger-pf]` の d2h が 3,700.6 → 2,869.7 MB、
   h2d が 25,725.8 → 25,269.9 MB になること (lab と同じ)。
2. 3,826 cold の余裕は 0.1-0.3 ポイントしかない。実機で 5% を切るようなら
   `DENNO_PF_LM_ROUTER_DEV=0` で router だけ戻して切り分けてほしい
   (full は値が変わらないので残してよい)。
3. 次に削れるのは `mix` の mixed D2H (3,826 で 1.88 GB のうち full 12 層ぶん
   0.47 GB は誰も読まない) と、combine の同期 (1 層 2 回 × 48 層)。
   前者は `pf_mix_keep_layer` を層優先にも広げる形になるが、task 230 の
   14k +4.2% と同じ罠 (MIXDEVIN の記録とホストの古いバイト列) があるので、
   層優先の MoE が mixed をデバイスで読むことの確認から。

## 9. lab の VERDICT 行

t308a (`37a9036fd`):
```
VERDICT base vs x0 OK decode +0.2% prefill +1.6% md5 SAME errors 0
VERDICT ple0 vs x0 OK decode -3.6% prefill -1.6% md5 SAME errors 0
VERDICT full vs x0 OK decode -3.2% prefill +0.0% md5 SAME errors 0
VERDICT rdev vs x0 OK decode -3.6% prefill -1.6% md5 DIFF errors 0
VERDICT base2 vs x0 OK decode -3.4% prefill -1.6% md5 SAME errors 0
VERDICT_DEPTH base c8000_cold vs x0 decode -2.8% prefill -1.4%
VERDICT_DEPTH full c8000_cold vs x0 decode -12.7% prefill -11.3%
VERDICT_DEPTH rdev c8000_cold vs x0 decode -12.2% prefill -29.0%
VERDICT_DEPTH base2 c8000_cold vs x0 decode -12.7% prefill -26.3%
VERDICT baseline x0 errors 0 text ok
```
(VERDICT はどれも最初の COND = 冷えた x0 との比較。t308a の 14k は
page cache のドリフトで位置に強く依存するので、採否は 3,826 の 2 ラウンド目と
台帳のバイト数で読んだ。)

t308b (`37a9036fd`):
```
VERDICT base vs x0 OK decode +0.7% prefill +1.7% md5 SAME errors 0
VERDICT ple0 vs x0 OK decode +0.7% prefill +1.7% md5 SAME errors 0
VERDICT both vs x0 OK decode +0.9% prefill +0.8% md5 DIFF errors 0
VERDICT base2 vs x0 OK decode +1.1% prefill +0.8% md5 SAME errors 0
VERDICT_DEPTH both c2000_cold vs x0 decode -10.5% prefill -31.4%
VERDICT_DEPTH both c8000_cold vs x0 decode -21.5% prefill -36.9%
VERDICT baseline x0 errors 0 text ok
```

t308c (`605082678`、`load_before=6.40`):
```
VERDICT old vs x0 OK decode +0.2% prefill +0.8% md5 DIFF errors 0
VERDICT new vs x0 OK decode -0.4% prefill +0.0% md5 SAME errors 0
VERDICT new2 vs x0 OK decode -0.2% prefill +2.5% md5 SAME errors 0
VERDICT old2 vs x0 OK decode -0.4% prefill +1.6% md5 DIFF errors 0
VERDICT_DEPTH old c8000_cold vs x0 decode -0.5% prefill -20.3%
VERDICT_DEPTH new c8000_cold vs x0 decode -20.9% prefill -67.9%
VERDICT baseline x0 errors 0 text ok
```
(t308c の VERDICT は「冷えて負荷のある x0」との比較なので使っていない。
採否は 2 ラウンド目の生値 — 表 3.)
