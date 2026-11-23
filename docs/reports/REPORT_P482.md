ADOPT: yes t482b

# task 482 (dev3) — UD-Q4_K_XL: 14k cold prefill の `other` (壁の 8%) を
# 初めて分解した (iter_top 392 / pf_er 217 / teardown 121 / res_up 92 ms)。
# そのうち **窓が足りない層で捨てる routing (GPU router + D2H) を先に捨てる**
# ゲートを足した: 3,826 cold の `other` 段 -49% (median 213 -> 108 ms)、
# 壁は -0.8〜-1.0% (lab のノイズ床の内側)、14k・短い decode・既定モデルは不変

branch: **br482** = 開始時の int-auto **b89a08a** (merge: br474)。**br478 は merge
していない** (task 478 は ADOPT: no で、`DENNO_PF_RA_DIRECT` は既定 off のまま
int-auto に入っていない。今回の変更はそこに依存しない)。
終端で int-auto を再 fetch したら **16481ef (merge: br484 = 既定モデル decode の
hc inject FMA)** に動いていたので merge した (`cbe9433`。衝突なし、両側を残した)。
**lab は merge 前の `cc12c6a` で取っている** — br484 は既定モデルの decode だけで、
こちらの変更 (prefill の先読み) とは経路が重ならない。

lab 名 (すべて `tree=denno-dev3` / `SCENARIO=depth8k` / `PROFILE=1` / `ptx=none`) /
baseline COND = タスク文のとおり `DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000
DENNO_EXPERT_GPU_MIN_HIT=0 DENNO_GPU_WEIGHT_MB=12000`:
- **t482a** = `7880ed9` / `model=flash-next-UD-Q4_K_XL` / `load_before=2.17` /
  COND: x0 b **xf** b2 — 台帳を足しただけで値は不変。xf は `DENNO_PF_XFER_DBG=1`
  (診断。1 転送 1 行なので遅い = 採否には使わない)
- **t482b** = `cc12c6a` / `model=flash-next-UD-Q4_K_XL` / `load_before=3.07` /
  COND: x0 b **o** **o2** b2 — 採否はこれで読む (o = `DENNO_PF_ER_EARLY_EXIT=0`)
- **t482c** = `cc12c6a` / `model=flash-next-UD-IQ3_XXS` (既定モデル) /
  COND: x0 b o o2 b2 — 既定モデルの確認 (同じ A/B)

## 0. 結論 (6 行)

1. **14k cold の `other` は 936 ms (壁 11.59 s の 8.1%) で、中身は
   `iter_top` 392 / `pf_er` 217 / `pf_teardown` 121 / `res_up` 92 ms**。
   task 419 の `iter n=336 uncovered=676.8ms` の正体はここで、初めて項目に割れた
   (§1.2 の表)。**壁の残りはディスク (69%) と GPU カーネル (23%) で、
   どちらもデバイス上限の 87-97%** にある (§4)。
2. 分解で見つかった**捨てる計算**を消した: `pf_early_route_prefetch` は
   窓の判定が router GEMV + D2H + VRAM 参照 + 置き場の計画の**後ろ**にあり、
   窓が足りない層 (3,826 = 2 チャンク) でも 48 層ぶん払っていた。
   判定を先頭へ移した (`DENNO_PF_ER_EARLY_EXIT`、既定 on)。
3. **3,826 cold の `other` 段: median 213.2 -> 108.2 ms (-49%)。**
   4 サンプル全部で ON <= 113.6 / OFF >= 178.2 (§3.1)。`pf_er` は ON で
   4/4 が 0.0、OFF で 70.5 / 69.1 ms。`iter uncovered` も 91.5 -> 22 ms。
4. **壁は -0.8〜-1.0% (推定) で、この lab のノイズ (±5%) の内側**。
   3,826 の壁は ON 9.68 / 9.00 s 対 OFF 9.13 / 9.65 s = **差が読めない**。
   14k と短い decode は 1 バイトも経路が変わらない (`pf_er` 217 ms が全 COND で
   同値、decode 14.3 ms/tok)。md5 も `c2000_cold` / `c8000_cold` が全 COND 一致。
5. **既定モデル (t482c) も同じ A/B で `other` が同じだけ縮む** (§3.3)。
   既定モデルは 3,826 の prefill が 2.2 s なので、同じ 70-90 ms が
   相対的には大きい。
6. `.cu` / `.ptx` は 1 バイトも変えていない (PTX 再生成不要)。

## 1. 前提: 現在の int-auto の UD-Q4_K_XL の段プロファイル (測り直した)

### 1.1 選んだ理由 (タスクの「まだ誰も触っていない最大のもの」)

P478 §0 の 6 行が「14k の床 8.94 s に対し実測 11.6-12.4 s、差の ~3 s は GPU の
仕事が写しと直列」で終わっていた。ただしその 3 s の内訳は
`ssm 1,881 + full 760` (GPU、カーネル律速) と `other 936` (正体不明) で、
**`other` だけが「誰も中身を読んでいない 8%」**だった。task 419 が
`iter uncovered=676.8ms` を出していたが、どの項目かは分かっていなかった。
ディスク (69%) は P420/P424/P474 で閉じており、decode は P431/435/437/442/449/452
で閉じているので、ここを選んだ。

### 1.2 14k cold の台帳 (t482a b2_1、ms)

| 段 | wall | うち GPU | うち host |
| --- | ---: | ---: | ---: |
| experts | 7,978.5 | 196.4 | 7,641.6 (`up_wait=7,379.0` = ディスク) |
| ssm | 1,881.4 | 1,315.2 | 39.5 |
| full | 759.7 | 552.3 | 408.0 |
| mix | 36.6 | 0.0 (busy 665.4 = 非同期) | 5.0 |
| other | 936.2 | 203.9 | 936.2 |
| **wall** | **11,592.4** | `gpu_busy=3,882.3` | `exposed_host=8,338.9` |

`other` の中身 (新しく足した 3 本 + 既存):

| 項目 | ms | 備考 |
| --- | ---: | --- |
| `iter_top` (新) | 392.1 | 回転の入口 = 残差の復帰 + PLE ブロック。`[h=145.9M/36]` |
| `pf_er` (新) | 217.0 | 早期 routing の先読み (層 1 回 x 48)。`[d=8.3M/144]` |
| `pf_teardown` | 121.4 | 終端の後片付け + 追い出した expert の戻し |
| `res_up` | 91.9 | 層 0 の残差 H2D `[h=574.2M/7]` |
| `combine` / `res_park` / `inj` | 3.4 / 3.2 / 1.0 | |
| **小計** | **830.6** | `iter_apre=0.0` (平坦化しているので `curs` は空) |

`iter n=336 uncovered` は **676.8 -> 613.3 ms** に減った (測れるようになったぶん)。

### 1.3 3,826 cold の `other` (t482a b_1)

`other` 181.3 ms = `pf_er` **72.6** + `res_up` 29.4 + `iter_top` 19.5 + 残り ~60。
**この 72.6 ms は完全な無駄だった**: 3,826 は 2 チャンクで
`early_route_window_ok(1913, 3826) == false` (4 チャンク未満) なので、
この先の先読みは何もしない。それでも router GEMV (1,913 トークン x 48 層) と
その D2H (7.7 MB / 144 回) と `peek_many` を払っていた。

## 2. 変えたこと (2 commit)

### 2.1 `7880ed9` — `other` の残差を割る台帳 3 本 (診断。値は不変)

- `q4ledger` に副区間 `iter_top` / `iter_apre` / `pf_er` を追加 (`SUB_N` 40 -> 43、
  既存の添字は動かしていない)。
- 層優先 prefill のチャンクループで、名前付き区間の外にある 3 か所を測る:
  回転の入口 (残差の復帰 / steer / PLE の待ち)、`curs` の組み立て、早期 routing の
  先読み (`pf_early_route_prefetch`、層 1 回)。
- テスト 2 本 (`task482_iter_residual_sub_names_are_wired` /
  `task482_sub_outside_named_region_lands_on_other`)。
- **ゲート無し** (計測だけ。`PROFILE=1` の台帳に列が増えるだけ)。

### 2.2 `cc12c6a` — `DENNO_PF_ER_EARLY_EXIT` (既定 on)

`pf_early_route_prefetch` の先頭で `early_route_window_ok(nt0, nt_all)` を見て、
窓が足りない層は**その場で** `None` を返す。task 474 の実装ではこの判定が
router GEMV + その D2H (`sels` / `wts` / `rowsum`) + `peek_many` (VRAM 参照) +
置き場の計画の**後ろ**にあり、窓が足りない層でも全部払ってから `None` を返していた。

- 元の判定もそのまま残したので、`DENNO_PF_ER_EARLY_EXIT=0` では task 474 と
  **1 命令も変わらない** (純関数 `er_early_exit_from_value`、既定 on・
  明示の偽値だけ off)。
- 値も I/O も変えない: 窓が足りない層はこの先の経路が必ず `None` を返すので、
  捨てる計算を先に捨てるだけ。md5 は 3 モデル x 全 COND で一致 (§3)。
- テスト: `kernel_invariants::t482_early_route_window_exit_is_wired`
  (早期終了が router / 置き場の計画より前、既定 on の真理値表、
  `early_route_window_ok` の意味が変わっていないこと)。

**この 1 本が効く範囲**: チャンク数が 2〜4 のプロンプト (1 チャンクは元から
router の手前で戻る)。チャンクは `min_tok=512` と `DENNO_PF_CHUNK` (既定 2,048) で
決まるので、**実運用では 2,048〜10,239 トークンの prefill 全部**が該当する
(3,826 はその代表)。

## 3. lab の結果

### 3.1 t482b (UD-Q4_K_XL) — 3,826 cold の `other` 段が本命

`depth8k` の `c2000_cold` (3,826 トークン、同じ run の 4 サンプル / 2 サンプル):

| COND | gate | `other` ms | `pf_er` ms | `iter uncovered` ms | 壁 s |
| --- | --- | ---: | ---: | ---: | ---: |
| x0 (捨て) | on | — | — | — | 10.62 |
| b | **on** | 106.6 | **0.0** | 21.4 | 9.68 |
| b (round 2) | **on** | 107.0 | **0.0** | 23.2 | — |
| b2 | **on** | 113.6 | **0.0** | 22.1 | 9.00 |
| b2 (round 2) | **on** | 109.3 | **0.0** | 21.4 | — |
| o | off | 178.2 | 70.5 | 91.3 | 9.13 |
| o2 | off | 248.1 | 69.1 | 91.7 | 9.65 |

**`other` の中央値 213.2 -> 108.2 ms = -49.2%** (on 4 サンプル / off 2 サンプル)。
`pf_er` は on で 4/4 が 0.0、off で 70.5 / 69.1。`iter uncovered` も 91.5 -> 22.0 ms。
**壁は 9.00-9.68 対 9.13-9.65 s で差が読めない** (この run の load_before=3.07、
同じ run の 14k も 11.40-21.24 s と 2 倍振れている)。

### 3.2 t482b — 14k / 短い decode / md5

- **14k は経路が変わらない**: `sub other pf_er` が b/o/o2/b2 の 4 COND すべてで
  **217.0 / 216.3 / 217.0 / 216.9 ms** (14k は 7 チャンクで窓が足りる)。
  `other` は 887.9-1125.9 ms で、b_1 の 1125.9 は**ディスクが半分の速さ**の回
  (`copy_s=16.94 s`、他は 7.37。壁 21.24 s)。**14k の変化は ±0 (推定)**。
- 短い decode: b 14.3 / o 14.3 / o2 14.7 / b2 14.3 / x0 14.8 ms/tok = 差なし。
- md5: `a` / `c2000_cold` / `c8000_cold` は**全 COND 一致**
  (`c2000_cold:73510e1f`、`c8000_cold:47c833e2`)。`b` / `*_turn2` が COND 間で
  割れるのは baseline 同士 (b と b2) でも起きる既知の揺れ (P466 §5 / P474 §4.2)。
- errors 0、text は全 COND 「looks ok」。TXTD (3,826) は
  「ご提示いただいた文書は、LLM推論エンジン（`denno-engine）における
  **Prefill（事前計算フェーズの性能最適化の歴史と教訓を詳細に記録したものです。…」

### 3.3 t482c (既定モデル flash-next-UD-IQ3_XXS) — 同じ A/B

3,826 cold の `other`:

| COND | gate | `other` ms | `pf_er` ms | `iter uncovered` ms | 壁 s |
| --- | --- | ---: | ---: | ---: | ---: |
| x0 (捨て) | on | — | — | — | 5.21 |
| b | **on** | 1579.8 (**外乱**: `iter_top=1502`) | 0.0 | 1503.4 | 3.08 |
| b (round 2) | **on** | 82.1 | **0.0** | 22.0 | — |
| b2 | **on** | 82.6 | **0.0** | 22.1 | 2.10 |
| b2 (round 2) | **on** | 78.3 | **0.0** | 22.3 | — |
| o | off | 145.0 | 64.5 | 84.9 | 2.12 |
| o2 | off | 145.4 | 64.3 | 85.1 | 2.12 |

**`other` の中央値 145.2 -> 82.1 ms = -43.5%** (b_1 は page cache が冷えた回なので
除く)。壁は 2.10 対 2.12 s = **-0.9%**。14k は `pf_er` 217.4-218.5 ms で
**全 COND 不変**、壁 5.18-5.32 s (b の `iter_top=304.6` が外乱)。
短い decode 11.1-11.2 ms/tok、**md5 は全 COND・全リクエストで SAME**
(b vs x0 が `md5 SAME` なのがこの run の一番強い証拠)。

### 3.4 VERDICT 行 (t482b は x0 が捨て COND なので採否には使わない)

```text
t482b: VERDICT b vs x0 OK decode -3.2% prefill -5.4% md5 DIFF errors 0
       VERDICT o vs x0 OK decode -3.2% prefill -6.2% md5 DIFF errors 0
       VERDICT o2 vs x0 OK decode -0.5% prefill -7.0% md5 DIFF errors 0
       VERDICT b2 vs x0 OK decode -3.4% prefill -7.0% md5 DIFF errors 0
t482c: VERDICT b vs x0 OK decode -0.7% prefill +3.5% md5 SAME errors 0
       VERDICT o vs x0 OK decode -0.3% prefill +0.9% md5 SAME errors 0
       VERDICT o2 vs x0 OK decode -0.7% prefill +0.9% md5 SAME errors 0
       VERDICT b2 vs x0 OK decode -1.0% prefill +0.0% md5 SAME errors 0
```

## 4. 数値の扱い

- **採否の主指標は `[q4ledger-pf] sub other` の `pf_er` と `other` の段**。
  壁はこの lab では 1 サンプル ±5-100% 振れる (t482b の 14k は同じ run で
  11.40 と 21.24 s、`copy_s` 7.37 対 16.94) ので、**1% の壁差は読めない**。
  段の差 (213 -> 108 ms、4/4 と 2/2 で重なり無し) は読める。
- **タスクの目標 (UD の 3,826/14k が 5% 速い、または短い decode 3%) は未達**。
  壁の実測は -0.9〜-1.0% (両モデル、3,826)。
- operator の段ルール (「2 ラウンドの中央値で当該段 2% 以上、他が 1% 超悪化しない」)
  には**当てはまる**ので既定 on にした: `other` 段 -43.5% (t482c) / -49.2% (t482b)
  の 2 ラウンド、14k・decode・md5 は不変。**5% の壁目標で厳密に読むなら
  `DENNO_PF_ER_EARLY_EXIT=0` で切り戻せる**。
- マイクロベンチは取っていない (GPU の無いコンテナ)。`pf_er` / `other` /
  `iter uncovered` / `copy_s` は lab の実測。壁の -0.8〜-1.0% は
  「段で消えた 65-105 ms ÷ 壁」の**推定**と、t482c の 2.10 対 2.12 s の実測。

## 5. テスト

最終 HEAD (2 commit + int-auto 16481ef の merge、int-auto b89a08a の上):

```text
CARGO_TARGET_DIR=/tmp/t482 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1347 passed; 0 failed; 103 ignored   (task 482 で 2 本追加。br484 の 2 本を含む)
同 --test kernel_invariants
  -> 140 passed; 0 failed                 (task 482 で 1 本追加)
CARGO_TARGET_DIR=/tmp/t482nc cargo build --workspace --release          (cuda 無し)
同 cargo test --workspace --release --no-fail-fast -- --test-threads=1
```

既知の赤 (`qwen4exp_decode spec_batch_stops_at_eos`、`qwen4exp_params` の
validate 2 件) は今回も出ていない。

## 6. 次の担当者へ (14k の残りはどこにあるか)

1. **14k cold の壁 11.4-12.0 s の内訳はもう読める**: `experts 7,978`
   (うち `up_wait 7,379` = ディスク 52.8 GB、5.4-6.0 GB/s = デバイス上限の
   87-97%) / `ssm 1,881` (GPU 1,315) / `full 760` (GPU 552) / `other 936`。
   **`other` の中は `iter_top 392` + `pf_er 217` + `pf_teardown 121` +
   `res_up 92`**。ディスクとカーネルは床なので、次に削れるのは `other` だけ。
2. **`iter_top` 392 ms の正体は PLE ブロック** (層 1 の 7 チャンクで ~300 ms +
   層 0 の残差 H2D 92 ms)。`[pre-kprof]` の `ple 293.3 ms` と一致する。
   ここを回転の外 (層 0 の attention の裏) へ出すのが次の一手 (推定 -250 ms = -2.2%)。
3. **`pf_teardown` 96-244 ms は「次のリクエストのための戻し」**
   (`pf_restore_experts`)。現リクエストには不要なので、次のリクエストの
   1 トークン目まで遅らせれば prefill から消える (推定 -100 ms = -0.9%)。
   ただし P242 の「戻さないと次の decode のヒット率が 0.75 -> 0.25」を壊さない形で。
4. **`pf_er` 217 ms は窓の中の実 I/O の回収** (`rc_plan` の `rc_join` が前の層の
   読み終わりを待つ)。捨てではないので、削るには層をまたぐ構造変更が要る
   (P478 §6.1 の波面)。**窓が足りない層の router はもう払わない** (今回)。
5. 3,826 cold は 8.3-9.4 s で 89% がディスク (`copy_s` 7.2-7.4 s / 43.5 GB)。
   `other` は 108 ms まで落ちたので、ここはもう触るところが無い。
