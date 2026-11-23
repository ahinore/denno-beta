ADOPT: yes t490b

# task 490 (dev3) — UD-Q4_K_XL: 窓の**広い** prefill 層 (14k、残り 6 チャンク) の
# 先読み予算を 24 -> 32 KiB/token に上げた (`DENNO_PF_ER_KB_WIDE`)。
# 14k cold の experts 段 16,529.6 -> 15,887.7 ms (-2.6%、baseline 2 本の
# 中央値比)、壁 20.00 -> 19.33 s (-2.0%)。3,826・短い decode・既定モデルは不変。
# **タスクの 5% 目標は未達** (段の規則で採る)

branch: **br490** = 開始時の int-auto **aea1324** (= merge: br486)。
自分の br486 は既に int-auto に merge 済みだったので取り込み不要。
終端で int-auto を再 fetch したが **aea1324 のまま** (動いていなかった) ので
merge は無い。

lab 名 (すべて `tree=denno-dev3` / `PROFILE=1` / `ptx=none`) /
baseline COND = タスク文のとおり `DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000
DENNO_EXPERT_GPU_MIN_HIT=0 DENNO_GPU_WEIGHT_MB=12000`:
- **t490a** = `aea1324` / `model=flash-next-UD-Q4_K_XL` / `SCENARIO=depth8k` /
  `load_before=3.86` / COND: x0 b b2 (現 int-auto の段プロファイル。値は変えない)。
- **t490b** = `cc9ef38` / `flash-next-UD-Q4_K_XL` / `depth8k` / `load_before=2.13` /
  COND: x0 b **k32** **k40** b2 (門だけ入れた commit。k32/k40 =
  `DENNO_PF_ER_KB_WIDE=32/40`)。**採否はこれで読む**。
- **t490c** = `cc9ef38` / `flash-next-UD-IQ3_XXS` (既定モデル) / `depth8k` /
  `load_before=8.05` / COND: x0 b **d32** **d40** b2 (同じ A/B)。

lab は 3 ラウンドの上限を使い切った。**t490b と t490c は merge 前の `cc9ef38`**
(= 既定 off の commit) で取っている。t490c の後に既定を 32 へ上げた (`0652ebb`)
のは、既定 on の値が t490b の k32 と同じ経路であることをコードで確認したため
(既定を変えても `er_route_kb(wide, base, 32)` = 32 で、lab の COND と同値)。

## 0. 結論 (6 行)

1. **採る (段の規則で)。** 14k cold の experts 段 (ディスク) が
   **16,529.6 -> 15,887.7 ms (-2.6%)**、`copy_s` 15.88 -> 15.26 s (-2.6%)、
   壁 20.00 -> 19.33 s (-2.0%)。baseline は同じ run の b と b2 の 2 本で、
   **k32 は両方より下** (b 16,529.6 / b2 16,111.4)。
2. **効いた理由は task 486 が残した「窓は余っているが予算が尽きる」。**
   14k は 1 層の予算 288 MB に対し置き場へ入ったのは 259 MB (`ra_MB=12,700.8`
   / `er_MB=14,166.6` = 87%)。32 KiB にすると計画 18,880.6 MB、実際に入ったのは
   **14,026.8 MB** で、**頭打ちは窓** (1 層 69 ms) になった。40 KiB でも
   入った量は 14,099.0 MB でほぼ同じ = 窓が上限。
3. **3,826 は 1 バイトも変わらない** (設計どおり `narrow` は従来値)。
   `er_MB=2313.0` が k32/k40 でも同値、`copy_s` は 14.17-15.82 の範囲で
   baseline と区別できない。
4. **短い decode も不変**: 14.0 (k32/k40) 対 14.1 (b/b2) ms/tok。
   `ssm` / `full` / `gpu_busy` は全 COND 同値 (1,854.7-1,858.2 / 741-750 /
   3,864-3,877 ms)。**変わったのは experts 段のディスクだけ**。
5. **既定モデル (t490c) も不変**: 14k の壁 5.099 (d32) 対 baseline 5.207/5.011
   (中央値 5.109 = -0.2%)、3,826 1.89-2.04 s、decode 10.8 ms/tok、
   **md5 は全 COND・全リクエストで SAME**、errors 0。
   門は既定モデルでも**発火した** (`er_MB` 14,179 -> 18,465 / 21,464、
   `ra_ms` 796 -> 1,025 / 1,184) が、壁は動かない = 効かない側に費用が出ない。
6. **タスクの目標 (UD の 3,826/14k が 5%、または短い decode 3%) は未達。**
   -2.0% (壁) / -2.6% (段) で、5% には届かない。operator の段ルール
   (当該段 2% 以上、他は 1% 超悪化しない) に当てはまるので既定 on にした。
   切り戻しは `DENNO_PF_ER_KB_WIDE=0`。

## 1. 前提: なぜこれを選んだか (Do 1 / Do 2)

### 1.1 現 int-auto の段プロファイル (t490a、UD-Q4_K_XL)

14k cold (b2_1、`[q4ledger-pf]`、wall 20,446.8 ms。t490a は load 3.86 で
ディスクが遅い回だった。数値は同じ run の中だけで比べる):

| 段 | wall | GPU | 備考 |
| --- | ---: | ---: | --- |
| experts | 16,983.7 | 193.8 | `up_wait=16,411.8` (ディスク) |
| ssm | 1,859.1 | 1,314.1 | GPU 律速 (P392/P397 で決着済み) |
| full | 743.2 | 551.1 | flash decoding (P448/P460 で決着済み) |
| mix | 26.1 | 0.0 (busy 657.4) | 非同期 |
| other | 834.7 | 200.7 | iter_top 292 / pf_er 218 / teardown 113 / res_up 97 |
| **wall** | **20,446.8** | `gpu_busy=3,865.4` | `exposed_host=17,212.8` |

3,826 cold は wall 17,354.5 / experts 16,442.1 / `copy_s=16.09` /
`gpu_busy=1,119.6`。**両方とも壁の 80-95% はディスク**で、GPU は 6-19% しか
使っていない。decode の内訳 (warm 128 tok、b2_1) は
`moe experts=675.7 / router=80.4 / shared=21.9`、`attn gate=380.8 /
gdn=380.0 / full=163.8` ms/128 tok で、**experts 5.28 ms/tok が最大**だが
これは VRAM ミスの CPU GEMV で、P435/P442 が「メモリ律速、512 bit でも
ノイズ」として閉じている。

### 1.2 選んだ理由

`other` (14k で 835 ms = 4.1%) は P482 が分解済みで、内訳の最大は
`iter_top 292` (PLE gather、P182 で RAM 置きも捨てたデバイス上限) と
`pf_er 218` (先読みの入口)。**残るのは「窓を使い切れていない」こと**で、
P486 §6.2 がそのまま申し送りにしていた:

> 窓 404 ms に対して先読みは 42 MB/層 = 5.3%。予算を 2 倍にすると窓を越えるので、
> **窓の中で読める量を層ごとに実測して予算を決める**のが次の一手

P486 の 2x2 (t486a) は 3,826 の **narrow** 窓 (8.4 ms) を見ていた。14k の
**wide** 窓 (69 ms) は予算の方が先に尽きている (t490a: 計画 288 MB/層 に対し
回収 259 MB/層) ので、**wide と narrow で予算を分ける**のが最小の変更で
最大の残りを取る形だった。decode と 3,826 は上のように既に閉じている。

## 2. 変えたこと (2 commit)

### 2.1 `cc9ef38` — `DENNO_PF_ER_KB_WIDE` (門。この時点では既定 0 = 従来と同じ)

- `exp_pf::er_kb_wide_from_value` / `er_kb_wide` / `er_route_kb(wide, base,
  wide_kb)` を足した。`er_route_kb` は純関数で、`wide && wide_kb > 0` のときだけ
  別値、それ以外は従来の `DENNO_PF_EARLY_ROUTE_KB` (24)。
- `DENNO_PF_EARLY_ROUTE_MB` (1 層の明示上限) は従来どおり最優先。
- 配線は `pf_early_route_prefetch` の 1 か所だけ: `wide` を見て KiB を選び、
  `early_route_budget` へ渡す。**narrow の経路は 1 命令も変わらない**。
- 値も読みも変えない (先読みは置き場に読むだけで、MoE の写しが同じ
  (パート, オフセット, 長さ) を拾う = task 474 の経路そのもの)。

### 2.2 `0652ebb` — 既定を 32 KiB/token にした (t490b の k32 が勝ったため)

`DENNO_PF_ER_KB_WIDE=0` で task 474/486 の 24 に戻る。

## 3. lab の結果

### 3.1 t490b (採否を読む run) — UD-Q4_K_XL、14k cold

同じ run の 4 サンプル (`[q4ledger-pf]` の experts 段と `[exp-pf-stream]`):

| COND | 予算 | `er_MB` | `ra_MB` (回収) | copy_s (s) | experts 段 (ms) | 壁 (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| b | 24 | 14,166.6 | 12,700.8 | 15.88 | 16,529.6 | 20.00 |
| **k32** | **32** | **18,880.6** | **14,026.8** | **15.26** | **15,887.7** | **19.33** |
| k40 | 40 | 23,590.6 | 14,099.0 | 15.42 | 16,035.1 | 19.51 |
| b2 | 24 | 14,166.6 | 13,004.9 | 15.47 | 16,111.4 | 19.46 |

- baseline 2 本の中央値: experts 段 **16,320.5**、`copy_s` **15.675**、
  壁 **19,730**。k32 は **-2.6% / -2.6% / -2.0%**、**両方の baseline より下**。
- 40 KiB は計画だけ増えて回収は 14.10 GB で同じ = **窓が上限**。よって 32 を採る
  (計画が小さい方が置き場の取り合いが少ない)。
- `ra_pct=88` は 24 でも 32 でも同じ (回収した先読みの 88% が MoE に使われる)。
- `ssm` / `full` / `gpu_busy` / `pf_er` は全 COND 同値 (§0 の 4 行目)。

3,826 cold は `er_MB=2313.0` が k32/k40 でも同値 (`narrow` は従来値)。
`copy_s` は b 15.68/14.39、k32 15.08/14.35、k40 15.82/14.17、b2 14.20/14.19 で、
**k32 の 2 ラウンドは baseline の範囲の内側** (この run の 3,826 は
ディスクの揺れが 15.68 対 14.17 = 10% あり、±2% は読めない)。

### 3.2 t490c (既定モデル UD-IQ3_XXS) — 何も変わらない

| COND | 14k cold (s) | 3,826 cold (s) | decode ms/tok | md5 |
| --- | ---: | ---: | ---: | --- |
| x0 | 7.90 (捨て・冷え) | 6.59 | 10.8 | SAME |
| b | 5.207 | 2.04 | 10.8 | SAME |
| d32 | 5.099 | 1.90 | 10.8 | SAME |
| d40 | 5.096 | 1.90 | 10.8 | SAME |
| b2 | 5.011 | 1.89 | 10.8 | SAME |

`er_MB` は 14,179.3 (b) -> 18,465.0 (d32) / 21,464.0 (d40)、`ra_ms` は
795.6 -> 1,025.4 / 1,184.4 で、**門は発火している**。それでも壁は
baseline 中央値 5.109 に対し -0.2% (b2 の 5.011 とも -0.2% 以内)。
`md5` は 4 COND x 全リクエストで SAME、errors 0、text は全 COND 「looks ok」。
既定モデルは expert がページキャッシュ / RAM tier から来るので
(`ra_pct=0`)、先に読んでも 1 バイトも得をしない = 設計どおり。

### 3.3 VERDICT 行 (x0 は捨て COND なので採否には使わない)

```text
t490b: VERDICT b vs x0 OK decode -38.7% prefill -15.2% md5 DIFF errors 0
       VERDICT k32 vs x0 OK decode -39.3% prefill -17.2% md5 DIFF errors 0
       VERDICT k40 vs x0 OK decode -39.3% prefill -17.2% md5 DIFF errors 0
       VERDICT b2 vs x0 OK decode -38.7% prefill -17.9% md5 DIFF errors 0
t490c: VERDICT b vs x0 OK decode +0.0% prefill +0.9% md5 SAME errors 0
       VERDICT d32 vs x0 OK decode +0.0% prefill -0.9% md5 SAME errors 0
       VERDICT d40 vs x0 OK decode +0.0% prefill -0.0% md5 SAME errors 0
       VERDICT b2 vs x0 OK decode +0.0% prefill -0.9% md5 SAME errors 0
t490a: VERDICT b vs x0 OK decode -3.8% prefill -7.8% md5 DIFF errors 0
       VERDICT b2 vs x0 OK decode -2.9% prefill -9.3% md5 DIFF errors 0
```

t490b の `x0` は最初の COND で page cache が冷えており (14k の 1 ラウンド目が
16.51 s、2 ラウンド目は 19.47-20.00 s に遅くなる回)、採否は §3.1 の
b/k32/k40/b2 だけで読む。**t490b の md5 DIFF は変更のせいではない**:
`c2000_cold` (3,826 の回答) は 5 COND すべて `73510e1f` で SAME、DIFF が出たのは
`b` (48 トークン) と `c2000_turn2` だけで、**baseline 同士 (x0 と b2) でも違う**
(t486b / t490a でも同じ揺れ)。こちらの変更は読む場所を変えるだけで、
MoE へ渡るバイト列は同一。

### 3.4 出力テキスト (sane の根拠)

t490b の `TXTD` (3,826 の回答、md5 `73510e1f` は全 COND 一致):

```text
ご提示いただいた文書は、LLM推論エンジン（`denno-engine）における
**Prefill（事前計算フェーズの性能最適化の歴史と教訓を詳細に記録したものです。
以下に、文書の構造と主要な技術的詳細を、論理的な流れに沿っ
```

`TXTB` (33 トークン、k32):

```text
Rust の「所有権（Ownership）とライフタイム（Lifetime）は、Rust を学ぶ上で
最も重要であり、かつ最も難しい概念です。しかし、これらは「メモリ管理を
コンパイラが自動的に行
```

## 4. 数値の扱い

- 採否の主指標は **`[q4ledger-pf]` の experts 段 (ディスク) と `copy_s`**、
  副が壁。decode と 3,826 は同じ run の中の A/B。
- **14k の限界**: lab は 14k のプロンプトを**ラウンド 1 にしか流さない**ので、
  14k は COND あたり 1 サンプルになる。ここでは「同じ run の baseline 2 本
  (b, b2) の中央値」と比べ、k32 が**両方より下**であることを根拠にした。
  2 ラウンドの中央値ではない点は規則より弱いので明記する (3 ラウンドの上限を
  使い切ったため、4 本目は取れない)。
- この run のディスクは最初の COND が速く後半が遅い方向へ 20% 漂流した
  (x0 の 14k 16.51 s 対 b/b2 20.00/19.46 s)。**k32 は b の後**に走っているので、
  漂流は k32 に不利に働く方向。それでも b より下だった。
- マイクロベンチは取っていない (GPU の無いコンテナ)。`er_MB` / `ra_MB` /
  `copy_s` / 段の壁 / 壁はすべて lab の実測。
- **推定**: 窓が上限なので、この経路の残りは 14.0 GB の回収で打ち止め。
  これ以上の予算 (40/64 KiB) は計画が増えるだけ (§3.1)。

## 5. テスト

最終 HEAD (2 commit、int-auto aea1324 の上):

```text
CARGO_TARGET_DIR=/tmp/t490 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1348 passed; 0 failed; 103 ignored
同 --test kernel_invariants
  -> 143 passed; 0 failed        (task 490 で 1 本追加)
CARGO_TARGET_DIR=/tmp/t490nc cargo build --workspace --release          (cuda 無し) -> OK
同 cargo test --workspace --release --no-fail-fast -- --test-threads=1  -> 失敗 0
```

既知の赤 (`qwen4exp_decode spec_batch_stops_at_eos`、`qwen4exp_params` の
validate 2 件) は今回も出ていない。

追加したテスト:
- `kernel_invariants::t490_wide_window_budget_is_wired`: 既定 32 と
  `0` の切り戻し口、`er_kb_wide_from_value` の 6 通り (None / 空 / 0 / 40 /
  非数 / 上限越え)、`er_route_kb` の真理値表 (wide x 0/32、narrow x 0/32)、
  予算の実数 (32 KiB x 12,000 = 393,216,000 B、24 KiB = 294,912,000 B、
  明示 MB が最優先)、配線 (`wide` を見て選び、選んだ KiB を
  `early_route_budget` へ渡し、選択が計算より前にある)。

## 6. 次の担当者へ

1. **14k の 4 本目**を取れば、この A/B を「2 ラウンドの中央値」で言い直せる
   (今回は 3 ラウンド上限)。推定では -2.0 〜 -2.6% のまま。
2. **窓が上限**なので、予算をこれ以上上げても意味が無い。次は
   (a) 窓そのものを広げる = MoE のディスク待ちの間に GPU を遊ばせない
   (14k の `gpu_busy=3,865 ms` が丸ごと直列。層 L の先読みと層 L+1 の
   attention を重ねる形。ただし層優先の順序を作り直す大きな変更)、
   (b) `pf_er` の入口費用 218 ms (router GEMV + D2H) を削る、の 2 つ。
   (a) は 14k で最大 -19% (推定)、今回の -2% とは桁が違う。
3. 3,826 の `narrow` 予算 (24 KiB) は task 486 のまま。ここを動かすと
   窓 (8.4 ms) を越えて写しと帯域を奪い合う (t486a の 24 本)。
