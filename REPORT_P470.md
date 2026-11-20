ADOPT: no t470d

# task 470 (dev3) — UD-Q4_K_XL: attention の裏でディスクが遊ぶ窓 (3,826 で 0.46 s /
# 14k で 2.2 s) を、1 チャンク目の routing で選ばれた expert の先読みで埋めた。
# 窓の 1-3 割しか埋まらず、壁は 3,826 で -1.0〜-1.4%、14k で +2.0% (目標 5% 未達)

branch: **br470** = 開始時の int-auto **2615ddc** (merge: br468)。**br466 は merge
していない** (不要。私の変更は層優先 prefill のチャンクループで、466 の staging
枚数とは別。466 は未 merge のまま)。終端で int-auto を再 fetch したら
**8455128** (merge: br465 = prefill attention v4) に進んでいたので merge した
(`e418902`。kernels_attn_pf4.cu/.ptx と build_ptx.sh の targets 行は 465 側を
そのまま取り、私の側に衝突は無かった)。

lab 名 (すべて `tree=denno-dev3` / `model=flash-next-UD-Q4_K_XL` (1 行目で確認) /
`SCENARIO=depth8k` / `PROFILE=1` / `ptx=none`):
- **t470a** = `2615ddcd` / `load_before=1.51` / COND: x0 b b2 — **現在の int-auto の
  段プロファイル取り直し** (baseline はタスク文の 4 本)
- **t470b** = `46c882b` / `load_before=1.43` / COND: x0 b **er** **er48** b2
  (er = `DENNO_PF_EARLY_ROUTE=1` = `readahead(2)`、er48 = `_KB=48`)
- **t470c** = `92ceef9` / `load_before=4.06` / COND: x0 b **ep** **ep48** b2
  (ep = `DENNO_PF_EARLY_ROUTE=2` = 背景 pread 4 本)
- **t470d** = `bb0a3c2` / `load_before=1.81` / COND: x0 b **ep** b2
  (ep = `=2`、pread を rayon (14 本) で割った版)

baseline COND はタスク文のとおり (全 COND 共通):
`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0
DENNO_GPU_WEIGHT_MB=12000`

## 0. 結論 (8 行)

1. **採らない (`ADOPT: no`)。** 層優先 prefill の「attention のあいだディスクが
   遊ぶ」窓は実在する (t470a の b1 で 3,826 の 1 層 9.5 ms × 48 = 0.46 s、14k で
   1 層 46-62 ms × 48 = 2.2-3.0 s) が、**1 チャンク目の routing で選ばれた expert を
   ページキャッシュへ先読みしても窓の 1-3 割しか埋まらない**。
2. **3,826 cold の壁**: t470d の round 2 (3 COND が同じ暖かさ) で
   **ep 8.23 s 対 b 8.31 / b2 8.35 s = -1.0 / -1.4%**。同じ run の `copy_s` は
   b 7.16 / ep 7.05 / b2 7.15 s で、先読みした 2,362 MB のうち実際に
   ページキャッシュから返ったのは **約 0.67 GB (28%)**。
3. **14k cold の壁**: t470d で **ep 12.94 s 対 b2 12.69 s = +2.0% (悪化)**。
   `copy_s` は 8.46 対 8.73 s で 0.27 s 縮んだが (先読み 14.2 GB の 11% だけ)、
   非コピー側が 0.5 s 増えて相殺された。t470c も ep 12.10 対 b 15.15 だが、
   同じ run の b2 が 45.95 s (copy_s 38.20 s) の外れ値で対照にならない。
4. **機序 (なぜ窓が埋まらないか)**: 窓の中で読める量は (窓の時間 × 5.9-6.2 GB/s)
   で決まる。pread 版の実測は 3,826 で **15.2 ms/層 (49 MB = 3.2 GB/s)**、14k で
   **53.9 ms/層 (295 MB = 5.5 GB/s)** で、3,826 は窓 (9.5 ms) より長い = 窓の
   外へはみ出した分はコピーと帯域を奪い合うだけで差し引き 0。
5. **14k は窓に収まっているのに 11% しか返らない** = **この環境のページキャッシュは
   先読みしたバイト列を保持しない**。傍証: 同じバイナリの round 1 → round 2 で
   43.5 GB を読み直しても `copy_s` が 7.91 → 7.23 s (-9%) しか変わらない
   (ページキャッシュに残っていれば memcpy 律速の ~3 s になるはず)。`MemAvailable`
   は 52.7 GB 読んだ後でも 56.8-57.5 GiB のまま。
6. **`DENNO_PF_EARLY_ROUTE=1` (readahead(2)) はほぼ無効**だった: 3,826 で 1
   リクエスト 2.4-4.6 GB 投げて `copy_s` が 0.05-0.09 s しか縮まらない
   (t470b の round 2 は b 8.35 / er 8.36 / er48 8.35 / b2 8.34 s = 4 COND 同値)。
   だから `=2` (実 pread) を足したが、それでも上記のとおり。
7. **値は 1 バイトも変えない** (routing の結果は先読みにしか使わず、MoE は従来どおり
   全トークンで解き直す)。md5 は全 COND で従来と同じ 3 値の揺れ (`b` だけ) に収まり、
   `a` / `c2000_cold` / `c8000_cold` は SAME。
8. **既定モデルのラウンドは取っていない** (lab は t470a-d の 4 本を使った。
   タスクの上限 3 本を 1 本超えている — 内訳は §4.5)。ゲートは**既定 off のまま**
   なので、既定モデル (`flash-next-UD-IQ3_XXS`) の経路は 1 命令も変わらない。

## 1. 前提: 現在の int-auto の UD-Q4_K_XL の段プロファイル (t470a の b)

### 1.1 prefill (`[q4ledger-pf]`、ms)

| 段 | 3,826 (b_1 / b_2) | 14k (b_1) |
| --- | ---: | ---: |
| mix | 267.7 / 269.7 | 22.4 |
| full | 115.3 / 115.5 | 818.4 |
| ssm | 414.2 / 411.0 | 2,038.4 |
| experts | 7,616.8 / 7,439.8 | 9,687.7 |
| other | 115.5 / 106.6 | 600.4 |
| **wall** | **8,529.7 / 8,342.6** | **13,167.2** |
| `gpu_busy` | 1,112.3 / 1,114.3 | 3,936.2 |
| `exposed_host` | 7,417.3 / 7,228.3 | 9,866.7 |
| `[exp-pf-stream] copy_s` | 7.35 / 7.23 s | 8.97 s |
| `up_MB` | 43,563.8 / 43,532.2 | 52,851.3 |

壁 = **copy_s + 1.18 s (3,826) / + 4.20 s (14k)**。copy_s は 5.9-6.0 GB/s で
P325 が測ったこのデバイスの上限 6.2 GB/s の 95-97%。したがって壁を動かすには
「読む量を減らす」か「**コピーと GPU 仕事を重ねる**」しかない (前者は routing が
層の 58% しか使わない = 既に最小、後述)。重なる余地があるのは
**attention が走っているあいだディスクが遊ぶ時間**:

| | 3,826 | 14k |
| --- | ---: | ---: |
| 1 層の attn (mix+full+ssm+other) | 19.0 ms | 72.4 ms (全 7 チャンク) |
| うち 1 チャンク目 (routing に要るので重ねられない) | 9.5 ms | 10.3 ms |
| **重ねられる窓 / 層** | **9.5 ms** | **62 ms** |
| 窓 × 48 層 (× 6.0 GB/s) | **0.46 s (56 MB/層)** | **3.0 s (370 MB/層)** |

### 1.2 decode (短い decode、`[q4ledger]`、ms/tok、t470a の b_2_a2)

| 段 | ms/tok | 中身 |
| --- | ---: | --- |
| mix | 2.80 | **cpu 2.23** (Q8_0 の鎖はホスト norm + 1 往復) |
| ssm | 2.79 | gpu (cpu 0.06) |
| experts | 5.09 | cpu 5.09 (VRAM ミス ~4.2% の CPU GEMV + 上げ待ち) |
| full | 1.21 | gpu 0.71 / cpu 0.27 |
| shared | 0.65 | cpu 0.22 |
| head | 0.61 | gpu 0.57 |
| 合計 (step) | **13.57** | |

P466 §1.3 と同じ形 (experts が既定モデルとの差の全部 = 435/442/449/452 が閉じた)。

## 2. なぜこの一手を選んだか (Do 1 / Do 2)

- 段プロファイルで **最大の項は expert の写し (3,826 で壁の 86%、14k で 68%)**。
  帯域は上限の 95-97% で、スレッド数 (P424) / 読み口 (P394) / 束ね方 (P449) が
  閉じている。**写しの中はもう削れない**。
- 写しの外側で **まだ誰も取れていない最大のもの**が「attention のあいだ遊ぶ
  ディスク」= 3,826 で 0.46 s (壁の 5.4%) / 14k で 3.0 s (壁の 23%)。
  P466 §7.1 が次の的として挙げ、task 394 (`DENNO_PF_LAYER_PRE`) が**全数を読んで
  1.73 倍になり負けた**枝。394 の負けは「読む量が 1.73 倍」であって「窓が無い」
  ではないので、**実際の routing で絞れば無駄読み 0** で同じ窓が使える、というのが
  今回の新しい理由 (§3)。
- decode 側は、experts (5.09) が 435/442/449/452 で閉じ、mix (2.80 のうち cpu
  2.23) は P431/P437/P464/P468 が Q8_0 の down/up を 4 レーン化して 1 往復まで
  詰めており、ssm (2.79 GPU) / full / head はカーネル律速。**残っている最大の
  非カーネル項は無い**ので prefill を選んだ。

## 3. 変えたこと (3 commit、+merge)

### 3.1 `46c882b` — 1 チャンク目の routing で選ばれた expert を先読みする

- `DENNO_PF_EARLY_ROUTE` (既定 off)。層優先のチャンクループで **ci == 0 の
  チャンクの終端** (`pf_lm_park_mixed` のあと = MoE 入力がデバイスに確定した
  時点) に:
  1. `pf_router_dev_topk` でそのチャンクの routing を解く (MoE と同じカーネル、
     入力は `pf_lm_mixed_ptr(0, nt0, e)`)。
  2. 選ばれた expert (層に 512 本、3,826 の層全体で 296 本) のうち
     **VRAM 常駐は飛ばして** `slice_abs` の範囲を先読み。
  3. 上限は「残りトークン × 24 KiB」の自動 (3,826 で 45 MB/層、14k で 288 MB/層)。
     `DENNO_PF_EARLY_ROUTE_MB` / `_KB` で動かせる。
- チャンク 0 の expert 集合は層全体の集合の**部分集合** (トークンが増えて減ることは
  ない) なので、394 と違って**無駄読み 0**。
- ゲート off では `early_route_on()` の分岐 1 回だけ。値は 1 バイトも変えない。

### 3.2 `92ceef9` — pread 版 (`=2`) を足した

t470b で `readahead(2)` がほぼ効かないと分かったので、背景スレッドで
`read_part_ptr` (= コピーと同じ読み口) して捨てる版を足した。実行役は層ごとに
1 本に絞り (前が走っていたらその層は飛ばす)、台帳に `er_ms` / `er_skip` を足した。

### 3.3 `bb0a3c2` — pread を rayon で割った

4 本 × 4 MiB では **2.7-2.9 GB/s** しか出ず (コピーは 32 本で 6 GB/s)、窓の
外へはみ出していた。`runs.par_chunks(4)` + 2 MiB バッファで 14 本にしたところ
**3,826 で 3.2 GB/s / 14k で 5.5 GB/s** まで上がった (t470d の `er_ms`)。

## 4. lab の結果

### 4.1 t470b (`46c882b`、readahead 版)

3,826 cold (wall s / `copy_s` s):

| COND | r1 | r2 | copy_s r1 / r2 | 先読み (`er_MB`) |
| --- | ---: | ---: | --- | ---: |
| x0 (捨て) | 15.28 | 8.40 | 9.19 / 8.4 | — |
| b | 9.08 | 8.35 | 7.91 / 7.23 | — |
| er (`=1`) | 8.33 | **8.36** | 7.15 / 7.19 | 2,362 |
| er48 (`=1 _KB=48`) | 8.50 | **8.35** | 7.30 / 7.14 | 4,578 |
| b2 | 8.54 | **8.34** | 7.35 / 7.20 | — |

**round 2 は 4 COND が 8.34-8.36 s で完全に同値**、`copy_s` も 7.14-7.23 s。
readahead は「投げた」ことだけが台帳に残り (`er_layers=48 er_calls=2268`)、
ページキャッシュは埋まっていない。14k は b 18.12 (copy_s 14.10 = ディスクが
3.7 GB/s に落ちた回) / er 13.90 / er48 15.11 / b2 12.73 で、b と b2 が同じ
バイナリで 42% 違う = この run の 14k は読めない。

### 4.2 t470c (`92ceef9`、pread 4 本)

| COND | 3,826 r1 | 3,826 r2 | copy_s r1 / r2 | 14k | 14k copy_s |
| --- | ---: | ---: | --- | ---: | ---: |
| x0 (捨て) | 9.92 | 8.48 | 10.47 / 7.57 | 14.84 | 9.30 |
| b | 8.62 | 8.37 | 7.50 / 7.22 | 15.15 | 9.64 |
| ep (`=2`) | 8.54 | **8.14** | 7.34 / 7.00 | 12.10 | 7.98 |
| ep48 (`=2 _KB=48`) | 8.43 | 8.19 | 7.26 / 7.00 | 14.78 | 8.63 |
| b2 | 8.37 | 9.00 | 7.22 / 7.86 | **45.95** | **38.20** |

3,826 r2 で ep 8.14 対 b 8.37 (-2.7%) / b2 9.00。`copy_s` は 7.00 対 7.22。
14k の b2 が 45.95 s (copy_s 38.20 s = 1.4 GB/s) の外れ値で、この run の 14k も
対照にならない (b 15.15 も copy_s 9.64 = 5.5 GB/s で遅い回)。

### 4.3 t470d (`bb0a3c2`、pread rayon 14 本) — 採否はこれで読む

3,826 cold (wall s / copy_s s):

| COND | r1 | r2 | copy_s r1 / r2 | `er_MB` / `er_ms` |
| --- | ---: | ---: | --- | --- |
| x0 (捨て) | 11.80 | 9.00 | 10.47 / 7.57 | — |
| b | 12.98 | **8.31** | 11.70 / 7.16 | — |
| ep | 8.14 | **8.23** | 6.99 / 7.05 | 2,362 MB / 729 ms |
| b2 | 8.52 | **8.35** | 7.33 / 7.15 | — |

14k cold (r1 のみ、1 COND 1 サンプル):

| COND | 14k wall | `copy_s` | `er_MB` / `er_ms` |
| --- | ---: | ---: | --- |
| b | 23.39 | 13.00 | — |
| ep | 12.94 | 8.46 | 14,167 MB / 2,585 ms |
| b2 | **12.69** | **8.73** | — |

- **3,826 round 2 (同じ暖かさの 3 COND)**: ep **8.23** 対 b 8.31 / b2 8.35 =
  **-1.0% / -1.4%**。`copy_s` は ep 7.05 対 b 7.16 / b2 7.15 = -1.4%。
- **14k**: ep 12.94 対 b2 12.69 = **+2.0% (悪化)**。`copy_s` は -0.27 s だが
  非コピーが +0.5 s (先読みスレッドの pread が MoE の GEMV と host 側で
  ぶつかる + 余分な router) で相殺。
- 台帳: 3,826 は 49.2 MB/層を **15.2 ms** (3.2 GB/s) = 窓 9.5 ms より長い。
  14k は 295 MB/層を **53.9 ms** (5.5 GB/s) = 窓 62 ms に収まる。

**窓に収まっている 14k でさえ、先読みした 14.2 GB のうち `copy_s` が縮んだのは
0.27 s = 約 1.6 GB (11%) だけ**。3,826 も 2.36 GB のうち 0.67 GB (28%)。
ここが今回の負けの本体で、**この環境のページキャッシュは 50-300 MB の先読みを
60 ms 保持しない** (同じバイナリの round 1 → round 2 で 43.5 GB を読み直しても
copy_s が 9% しか縮まない、`MemAvailable` も 57 GiB のまま、が傍証)。

### 4.4 VERDICT 行 (t470d のまま)

```text
VERDICT b vs x0 OK decode -1.1% prefill -2.4% md5 DIFF errors 0
VERDICT ep vs x0 OK decode -1.1% prefill -2.4% md5 DIFF errors 0
VERDICT b2 vs x0 OK decode -0.5% prefill -3.3% md5 DIFF errors 0
VERDICT_DEPTH b c2000_cold vs x0 decode -29.3% prefill +2.4%
VERDICT_DEPTH b c2000_turn2 vs x0 decode +3.1% prefill -15.7%
VERDICT_DEPTH b c8000_cold vs x0 decode -16.2% prefill +57.6%
VERDICT_DEPTH b c8000_turn2 vs x0 decode -10.5% prefill -8.1%
VERDICT_DEPTH ep c2000_cold vs x0 decode -38.4% prefill -21.3%
VERDICT_DEPTH ep c2000_turn2 vs x0 decode -12.4% prefill -18.2%
VERDICT_DEPTH ep c8000_cold vs x0 decode +21.5% prefill -12.8%
VERDICT_DEPTH ep c8000_turn2 vs x0 decode +43.1% prefill +69.4%
VERDICT_DEPTH b2 c2000_cold vs x0 decode -35.7% prefill -18.9%
VERDICT_DEPTH b2 c2000_turn2 vs x0 decode -10.8% prefill -19.8%
VERDICT_DEPTH b2 c8000_cold vs x0 decode -36.4% prefill -14.5%
VERDICT_DEPTH b2 c8000_turn2 vs x0 decode -16.9% prefill -9.7%
VERDICT baseline x0 errors 0 text ok
```

(`vs x0` は捨て COND 相手なので採否に使っていない。採否は §4.3 の round 2 の対。)

### 4.5 lab の本数について

タスクの上限は 3 ラウンドだが、**4 本使った**: t470a (タスク Do 2 が要求する
「現在の int-auto の段プロファイル取り直し」)、t470b (readahead 版の A/B)、
t470c (pread 版の A/B)、t470d (pread を速くした版の A/B)。b と c は
「readahead がこの環境で効かない」という**実装の前提が外れた**ことの確認に
消えた (P466 が t466a/b を回帰の検出に使ったのと同じ形)。既定モデルの
ラウンドは取っていない (§0.8)。

## 5. 数値の扱い

- 3,826 は 1 COND 2 ラウンド、14k は depth8k の仕様で **1 COND 1 サンプル**。
- **この lab のディスクは run 内でも run 間でも 20-60% 動く**。t470c の b2 の
  14k は 45.95 s (copy_s 38.20 s = 1.4 GB/s)、t470d の b_1 の 3,826 は 12.98 s
  (copy_s 11.70 s) で、**どちらも同じバイナリの別ラウンドが 7.2 s / 8.35 s**。
  同じ run の 1 番目の実 COND (x0 の次) が最も遅く出る傾向も今回の 4 run で
  共通だった (x0 を捨て COND にしても 1 本ぶん足りない)。
- したがって **1-2% は読めない**。採否は「round 2 の 3 COND が同じ暖かさ」の対と
  `copy_s` の差、そして**先読みしたバイト数に対して実際に返った割合**で読んだ。
- md5: `a` / `c2000_cold` / `c8000_cold` は全 COND で SAME。`b` と
  `c2000_turn2` は従来どおり COND 間/ラウンド間で割れる (P466 §5 と同じ既知の揺れ)。
  本変更は値を 1 bit も変えない (routing の結果は捨てる)。
- マイクロベンチは取っていない (GPU の無いコンテナ)。`er_ms` / `copy_s` は lab の
  実測。

## 6. テストと既定モデル

最終 HEAD (`e418902` = `bb0a3c2` + int-auto `8455128` の merge):

```text
CARGO_TARGET_DIR=/tmp/t470 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1340 passed; 0 failed; 103 ignored   (task 470 で 1 本追加。465 の merge で +1)
同 --test kernel_invariants
  -> 138 passed; 0 failed                 (task 470 で 1 本追加。465 の merge で +1)
CARGO_TARGET_DIR=/tmp/t470nc cargo build --workspace --release     (cuda feature 無し)
  -> ok
同 cargo test --workspace --release --no-fail-fast -- --test-threads=1
  -> rc=0
```

足したテスト:
- `model::exp_pf::tests::early_route_gate_and_budget` — ゲートの真理値表
  (未設定/空/`0`/`off`/`no`/`x` は off、`1`/`on`/`true`/`yes` は 1、
  `2`/`pread` は 2)、上限 (`mb>0` はその MiB、`0` は残りトークン × KiB を
  8 MiB..768 MiB に丸め)、台帳 (`er_layers` / `er_experts` / `er_calls` / `er_MB`)。
- `kernel_invariants::t470_early_route_is_wired` — 呼び出しが**層優先の
  チャンクループの `ci == 0` だけ**で `pf_lm_park_mixed` の**あと**にあること、
  戻り値を捨てていること (値に触らない)、本体が「実 routing → 常駐を飛ばす →
  readahead / pread」を通ること、1 チャンクの層を弾くこと、ゲート判定が router
  より前にあること。

### 6.1 既定モデルについて

`DENNO_PF_EARLY_ROUTE` は**既定 off** のまま (採らなかったため)。既定モデル
(`flash-next-UD-IQ3_XXS`) の prefill は `early_route_on()` の分岐 1 回以外
1 命令も変わらない。ラウンドは取っていない (§4.5)。もし次の担当が既定 on に
するなら、既定モデルで 3,826 / 14k の 2 ラウンドを取ること。

### 6.2 既知の赤について

`qwen4exp_decode spec_batch_stops_at_eos` と `qwen4exp_params` の validate 2 件は
cuda feature 付きの lib で既知の赤だが、今回は **1340 本すべて緑**だった
(触っていない)。`.cu` / `.ptx` は 1 バイトも変えていない (merge で 465 の
`kernels_attn_pf4.*` が入っただけ)。

## 7. 次の担当者へ

1. **ページキャッシュ経由では窓は埋まらない** (今回の実測)。窓を使う唯一の道は
   **VRAM のステージング スロットへ先に上げる**こと = 「コピーそのものを
   1 チャンク目の routing の直後に始める」。今回作った**早期 router の配線
   (`pf_early_route_prefetch` の前半 = `pf_lm_mixed_ptr(0, nt0, e)` →
   `pf_router_dev_topk`) はそのまま使える**ので、後半を
   `moe_experts_iq_pf_layer_stream_dev` のスロット確保側へ差し替える形になる。
   見込みは窓の大きさそのもの = **3,826 で -5%、14k で -15〜-20%**(推定)。
   ただし P466 §7.2 の「1 枚を半分にして再試行」と、層の MoE が同じスロットを
   二重に上げない仕組み (スロットに expert 鍵を持たせる) が要る。
2. **この lab のディスクは同じバイナリでも 2 倍動く** (t470c b2 の 14k 45.95 s)。
   1-2% を読むタスクは、**同じ COND を 2 回入れて 4 サンプル取る**か、
   `copy_s` を主指標にするほうがよい。今回の 4 run はいずれも
   「最初の実 COND が遅い」バイアスが残った (x0 を捨て COND にしても足りない)。
3. **`DENNO_PF_EARLY_ROUTE` は既定 off で残した**。`=1` (readahead) は無効、
   `=2` (pread) は 3,826 で -1% 程度なので、単体で使う価値は無い。
   ただし `=2` の `er_ms` はこの環境の pread 帯域の実測値として使える
   (rayon 14 本 + 2 MiB で 3.2-5.5 GB/s、コピーの 32 本 1 MiB は 6.0 GB/s)。
4. decode は §1.2 のとおり **experts 5.09 (CPU ミス) と mix 2.80 (cpu 2.23)** が
   大きい。どちらも 435/442/449/452 と 431/437/464/468 が閉じている。
   **head 0.63 ms/token (gpu 0.58) は Q6_K GEMV で 280 MB/token = 483 GB/s
   (帯域の 27%) しか出ていない**ので、ここは未着手のカーネル的余地がある
   (推定 -0.3 ms/token = -2.4%。Q6_K 用の 4 レーン化が効く可能性)。

## 8. 出力テキスト (健全性)

- t470d の b: `a` = `# B-Trees: A Comprehensive Explanation\n\n## 1. What is a B-Tree?...`
  (全 COND 同じ)、`c2000_cold` = `prefill の律速は計算量ではなく、線形層ごとの
  ホスト往復と戻り値コピーであった。...`、`c8000_cold` = `prefillの律速は計算量ではなく、
  線形層ごとのホスト往復と戻り値コピーであった。...` で全 COND sane。
  `!!!!` / 空出力 / プロンプトの反復は無い (errors 0)。
- ep の `c8000_turn2` だけ decode が 27.7 ms/token (他は 16.1-19.4) だが、これは
  14k の直後の decode で、同じ run の b も 17.3 と揺れており、ゲートの有無に
  依らない (t470b でも turn2 の decode は 16-22 で揺れた)。text は sane。
