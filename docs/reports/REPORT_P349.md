ADOPT: no t349a t349b t349c

# task 349 — 冷えた page cache のあとの IQ3_S prefill

branch `br349` は開始時の `int-auto` `ef21be403` から。`br343` は int-auto に
merge されていなかったが、**merge していない** (task 343 は `ADOPT: no` で、
触るのは GDN のホスト書き戻しだけ。本タスクの cold 読みとは独立なので要らない)。
タスクの途中で int-auto が `6c19792cf` (merge: br342) へ動いたので、
最後に merge した (衝突なし。`build_ptx.sh` の `targets_all` はどちらも
触っていない)。数値はすべて GPU ラボの実測 (`SCENARIO=depth8k`, `PROFILE=1`,
`swift15-abl-IQ3_S`)。lab 名は t349a / t349b / t349c。

## 0. 結論 (5 行)

- **目標に届かない。** 14k cold の最良は **-5.1%** (9.24 → 8.77 s、bg ゲート)。
  15% には 1.4 s 足りない。`ADOPT: no`。3 本のゲートはすべて既定 off。
- **冷えた 14k の中身は PLE の gather と expert の 2 つだけ** (t349c の
  x0 が決定的: expert が暖まり PLE だけ冷えた状態で 14k 9,334 ms、
  `[pre-kprof] ple=1,853`)。暖まった 14k (同 b3) は 6,417 ms で `ple=43.7`。
- **`DENNO_PF_BG_WARM` は cold で大きい** (3,826 cold -28.3% / -31.4%、
  14k cold -4.5% / -5.1%) が、**暖まった 3,826 を +45.8% 悪くする**
  (2.51 → 3.66 s)。常駐が暖まっている通常時に効かないので採らない。
- **`DENNO_PF_STREAM_PREAD` は効かない** (3,826 -2.5%、14k +1.3%)。IQ3_S は
  もともと WC バウンスなので pread が一度も発火していなかったが、非 WC に
  落としても cold の実速度が上がらなかった (§2.1)。
- **`DENNO_PF_PLE_PIPE` は lab で一度も発火していない** (t349b の pp が
  baseline と同値)。層優先の経路 (`forward_prefill_batch_resident_lm`) が
  全チャンクの gather を層ループの前に直列で引いており、最初の実装は
  非層優先の経路にしか入っていなかった。**直した版 (7431cea95) は
  ラウンド切れで未測定** (§2.3)。

## 1. Do 1: 基準の取り直しと、cold の内訳

### 1.1 基準 (int-auto `ef21be403` + 既定 env、すべて `DENNO_START_DROP_CACHE=1`)

| lab | 3,826 cold (2 reps) | 14k cold | 3,826 warm decode |
| --- | --- | --- | --- |
| t349a b1 | 8.80 / 8.79 | 9.23 | 16.6 |
| t349a b2 | 8.80 / 8.76 | 9.21 | 16.6 |
| t349a b3 | 8.79 / 8.80 | 9.21 | 16.7 |
| t349b b1 | 8.83 / 8.81 | 9.26 | 16.6 |
| t349b b2 | 8.86 / 8.84 | 9.22 | 16.6 |
| t349b b3 | 8.85 / 8.82 | 9.24 | 16.7 |

**冷えた page cache の 3,826 は 8.79-8.86 s、14k は 9.21-9.26 s。**
タスクの前提 (warm 2.52 / 6.44 s、cold の最初の COND 8.5-13.8 s) と一致する。
`DENNO_START_DROP_CACHE=1` は毎回の server 起動で clean page を落とすので、
lab の「最初の COND だけ冷たい」偏りを全 COND に均せて、交互の A/B が取れる
(この使い方が本タスクのやり方。§6 に注意も書いた)。

### 1.2 段ごと (t349b b1 の 14k、`[q4ledger-pf]`、ms)

| 段 | wall | exposed | h2d | 備考 |
| --- | ---: | ---: | --- | --- |
| mix | 20 | 0 | 0 | |
| full | 1,151 | 359 | 43 MB | |
| ssm | 2,511 | 936 | 159 MB | GDN の状態 D2H 823.69 MB / 504 (cold でも不変) |
| experts | 2,945 | **1,453** | 114 MB | `[pf-host] up_wait=2,436` |
| other | 2,635 | **2,361** | 4,128.72 MB / 4,110 | 中身は §1.3 |
| close | 9,256 | — | — | gpu_busy 1,491 |

### 1.3 `other` の正体 = PLE の gather (t349c の x0 が決定的)

t349c は「前のラウンドの cold COND が expert を暖めたあと」に走ったので、
**x0 は expert が暖かく PLE だけ冷たい**状態になった:

| t349c の cond | 14k wall | `[pre-kprof]` ple | layer | experts 露出 |
| --- | ---: | ---: | ---: | ---: |
| x0 (expert 暖・PLE 冷) | 9,334 | **1,853** | 7,056 | 1,472 |
| b3 (全部 暖 = 定常) | 6,417 | **43.7** | 6,026 | 379 |

- 冷えた 14k の +2.9 s は **ple +1.81 s と experts +1.09 s** でほぼ全部。
  `other` の h2d 4,128.72 MB / 4,110 は cold でも暖でも同じ値で、
  **ページが cold のときだけ 1.6 GB/s** になる (PLE の行を含む 4 KB ページ)。
- PLE の gather は `[pre-kprof]` の 2 行目で読める: 14k で
  `rows=166750 reads=166728 bytes=15007500 avg_read=90B read=13220.6 ms`。
  1 行 90 B のランダム pread が **166,728 回**で、和の待ちが 13.2 s
  (= 8 本並列で 1.85 s の壁)。`ple.rs` の doc の「14k で約 2.7 s」と同方向。
- したがって **14k cold を縮めるには PLE の gather を何とかするしかない**
  (expert 側は bg で -1.1 s まで削ったが、それでも 15% には届かない)。

## 2. Do 2: 変えたこと (3 本、すべて既定 off)

### 2.1 `DENNO_PF_STREAM_PREAD` — IQ 系のストリームでも pread (t349a `pr`)

task 325 の pread は `!pin_wc` のスロットだけで発火する。IQ 系は
`stream_wc_for` の既定が WC なので、**IQ3_S では pread が一度も発火して
いなかった** (t343c の `[exp-pf-stream]` に `pread_` の欄が無い)。
ゲート on でストリームの pin だけ非 WC に落として read_at を使う。

| t349a | 3,826 cold | 14k cold |
| --- | --- | --- |
| baseline (b1/b2/b3 中央値) | 8.79 | 9.21 |
| pr | 8.58 / 8.57 (**-2.5%**) | 9.33 (**+1.3%**) |

発火はしている: pr の `[exp-pf-stream]` は `wc=1 spread=1` で、
`pread_MB=22653.4 pread_calls=33135` (3,826) / `pread_MB=28377.7
pread_calls=41547` (14k) = **上げたバイトが全部 read_at 経由**。
それでも cold の実速度が上がらない (3.19 GB/s のまま) ので、14k は悪化した。
task 325 の UD は 4.2 → 5.8 GB/s だったが、IQ3_S の WC memcpy はもともと
3.2 GB/s 出ており、非 WC の pinned への read(2) がそれを上回らない。

### 2.2 `DENNO_PF_BG_WARM` — 起動後の非同期温め (t349a `bg` / t349b `ppbg`)

`serve_flash` のモデル構築時に 1 本のスレッドを立て、`warm()` で page cache を
読む (join しない = 起動もリクエストも待たせない)。extra (PLE + ホスト重み、
2,523 MB) を先、expert を `DENNO_PF_BG_WARM_MB` (既定 24 GiB) まで後。
t349a の bg は 28,141 MB を 10.49 s (2.7 GB/s) で読んだ。

| lab / cond | 3,826 cold | 14k cold | 3,826 warm |
| --- | --- | --- | --- |
| t349a baseline | 8.79 | 9.21 | — |
| t349a bg | **6.04 / 6.56 (中央値 -28.3%)** | 8.80 (**-4.5%**) | — |
| t349b baseline | 8.84 | 9.24 | — |
| t349b ppbg (pipe + bg) | **6.06 / 6.06 (-31.4%)** | 8.77 (**-5.1%**) | — |
| t349c baseline (b1/b2/b3) | — | — | **2.51 / 2.51 / 2.51** |
| t349c bg | — | — | **3.66 (+45.8%)** |

- cold の 3,826 は 2 ラウンドとも -28% 前後で、**露出が消えたのは experts だけ**
  (6,323 → 3,537 ms)。`other` は 1,035 → 1,038 ms で動かない (bg の extra は
  PLE の**表全体**ではない。表は 28.80 GB で予算 4 GB に入らない)。
- **暖まった 3,826 は +45.8%** (2.51 → 3.66 s)。bg が 28 GB を読み直すぶん、
  すでに暖まっている cache を追い出して、最初の prefill と帯域を奪い合う。
  生産の定常は暖まっている側なので、これは致命的。`ADOPT: no` の主因。
- 14k の -4.5% / -5.1% は「experts の露出 -0.5 s」ぶんだけで、
  PLE の gather には効かない (bg の extra に PLE の表が入らない)。

### 2.3 `DENNO_PF_PLE_PIPE` — PLE の gather を広いプールで引く (未測定)

t349b の `pp` は **baseline と同値** (3,826 8.84/8.81、14k 9.23、
`[pre-kprof] ple=1,853.3`)。原因は lab のログで分かった: 層優先の経路
(`forward_prefill_batch_resident_lm`) は全チャンクの gather を**層ループの
前に直列で**引いており、最初の実装 (sync_channel のパイプ) は非層優先の
経路 (`forward_prefill_batch_resident`) にしか入っていなかった。

直した版 (`7431cea95`): 埋め込みは直列のまま、gather だけを 32 本の専用
プールで引く (`ple_par_parts` はゲート on のとき `current_num_threads()` を
上限 32 で見る。従来の上限は 16)。読む行・重複排除・綴じ方は同じなので
値は変えない。**ラウンド上限 (3) を使い切ったので未測定**。次に 1 本
測るならこれ (§5)。

## 3. なぜ目標に届かなかったか (数字)

- 目標は 14k cold を 9.22 → 7.84 s。必要 -1.38 s。
- cold の内訳は ple +1.81 s、experts +1.09 s、その他 ±0。
- expert 側は bg で -0.5 s まで削れたが、それ以上は cache の大きさで頭打ち
  (14k の写しは 28.38 GB で、ホストの page cache は RSS 15-20 GB を引くと
  43 GB 前後。PLE の表 28.80 GB と両方は載らない)。
- PLE 側は **IOPS 律速のランダム 90 B pread 166,728 回**。並列度を上げる
  (上限 16 → 32) のが唯一の残った手で、これも 1.85 → 0.6 s 程度の見込み
  (推定)。bg と合わせても 9.22 - 1.2 - 0.5 = 7.5 s (-19%) だが、
  bg は暖かい側で +46% なので併用できない。
- 運用者の規則 (2 ラウンドの中央値で 2% 以上) は **cold の 3,826 で bg が
  満たす** (-28.3% / -31.4%)。ただし「他が 1% 以上悪化しない」を
  **暖まった 3,826 の +45.8% が破る**ので、規則の側からも採れない。

## 4. 採否

- `ADOPT: no`。3 本とも**既定 off** のまま残す (数値を doc に書いた)。
  - `DEFAULT_STREAM_PREAD = false` / `DEFAULT_BG = false` /
    `DEFAULT_PLE_PIPE = false`。
  - off のときの木は 1 バイトも変わらない (pread の可否・温めの有無・
    gather の並列度のどれも、ゲートが偽なら従来の値)。
- 数値経路は変えていない。md5 は t349a / t349b の全 COND・全ラウンド SAME
  (`a91e17da` / `5a8f5488` / `e2292eb6` / `561c6da8`)、errors 0、text sane。

VERDICT (t349b、全行):

```
VERDICT b1 vs x0 OK decode +0.0% prefill +0.8% md5 SAME errors 0
VERDICT pp vs x0 OK decode +0.2% prefill +0.0% md5 SAME errors 0
VERDICT ppbg vs x0 OK decode +0.7% prefill +0.0% md5 SAME errors 0
VERDICT b2 vs x0 OK decode +0.0% prefill +0.0% md5 SAME errors 0
VERDICT b3 vs x0 OK decode +0.2% prefill -0.8% md5 SAME errors 0
VERDICT_DEPTH b1 c2000_cold vs x0 decode +0.0% prefill -0.2%
VERDICT_DEPTH b1 c2000_turn2 vs x0 decode +0.0% prefill +0.0%
VERDICT_DEPTH b1 c8000_cold vs x0 decode -2.8% prefill -0.9%
VERDICT_DEPTH b1 c8000_turn2 vs x0 decode -2.2% prefill +0.0%
VERDICT_DEPTH pp c2000_cold vs x0 decode -0.3% prefill -0.1%
VERDICT_DEPTH pp c2000_turn2 vs x0 decode -0.2% prefill +0.0%
VERDICT_DEPTH pp c8000_cold vs x0 decode -6.7% prefill -1.2%
VERDICT_DEPTH pp c8000_turn2 vs x0 decode -8.3% prefill -2.1%
VERDICT_DEPTH ppbg c2000_cold vs x0 decode -7.3% prefill -31.4%
VERDICT_DEPTH ppbg c2000_turn2 vs x0 decode -2.8% prefill -2.3%
VERDICT_DEPTH ppbg c8000_cold vs x0 decode -6.2% prefill -6.1%
VERDICT_DEPTH ppbg c8000_turn2 vs x0 decode -7.4% prefill -2.1%
VERDICT_DEPTH b2 c2000_cold vs x0 decode -0.6% prefill +0.2%
VERDICT_DEPTH b2 c2000_turn2 vs x0 decode +0.0% prefill +0.0%
VERDICT_DEPTH b2 c8000_cold vs x0 decode -6.7% prefill -1.3%
VERDICT_DEPTH b2 c8000_turn2 vs x0 decode -8.3% prefill -2.1%
VERDICT_DEPTH b3 c2000_cold vs x0 decode -0.3% prefill +0.0%
VERDICT_DEPTH b3 c2000_turn2 vs x0 decode -0.2% prefill +0.0%
VERDICT_DEPTH b3 c8000_cold vs x0 decode -6.2% prefill -1.1%
VERDICT_DEPTH b3 c8000_turn2 vs x0 decode -8.3% prefill +0.0%
VERDICT baseline x0 errors 0 text ok
```

VERDICT (t349c、bg の暖かい側):

```
VERDICT b1 vs x0 OK decode +0.5% prefill +1.7% md5 SAME errors 0
VERDICT bg vs x0 OK decode +0.5% prefill -0.8% md5 SAME errors 0
VERDICT b2 vs x0 OK decode +0.2% prefill +1.7% md5 SAME errors 0
VERDICT b3 vs x0 OK decode +0.2% prefill +1.7% md5 SAME errors 0
VERDICT_DEPTH b1 c2000_cold vs x0 decode +0.4% prefill +0.0%
VERDICT_DEPTH b1 c2000_turn2 vs x0 decode +0.5% prefill +0.0%
VERDICT_DEPTH b1 c8000_cold vs x0 decode -12.3% prefill -12.8%
VERDICT_DEPTH b1 c8000_turn2 vs x0 decode -8.5% prefill +0.0%
VERDICT_DEPTH bg c2000_cold vs x0 decode +8.6% prefill +45.8%
VERDICT_DEPTH bg c2000_turn2 vs x0 decode +5.6% prefill +1.2%
VERDICT_DEPTH bg c8000_cold vs x0 decode -5.3% prefill -16.3%
VERDICT_DEPTH bg c8000_turn2 vs x0 decode -1.9% prefill +0.0%
VERDICT_DEPTH b2 c2000_cold vs x0 decode +5.6% prefill +0.2%
VERDICT_DEPTH b2 c2000_turn2 vs x0 decode +5.0% prefill +0.0%
VERDICT_DEPTH b2 c8000_cold vs x0 decode -12.9% prefill -25.6%
VERDICT_DEPTH b2 c8000_turn2 vs x0 decode -8.0% prefill +0.0%
VERDICT_DEPTH b3 c2000_cold vs x0 decode +0.0% prefill +0.0%
VERDICT_DEPTH b3 c2000_turn2 vs x0 decode +0.3% prefill +0.0%
VERDICT_DEPTH b3 c8000_cold vs x0 decode -12.3% prefill -31.3%
VERDICT_DEPTH b3 c8000_turn2 vs x0 decode -8.5% prefill +0.0%
VERDICT baseline x0 errors 0 text ok
```

(t349c は cache が run の中で暖まっていくので、c8000_cold の COND 間比較は
位置の効果が乗る。3,826 は全 COND 2.51 s で安定しており、bg だけ 3.66 s。
bg の +45.8% はこの 3,826 の比較を指す。)

テキスト (t349b の `TXT` / `TXTB` / `TXTD`、全 COND 同一):

```
A **B-tree** (specifically, the **B+ tree) is the fundamental data structure used by most
modern relational databases (such as MySQL, PostgreSQL, Oracle, and SQL Server).
Rust の「所有権（Ownership）とライフタイム（Lifetimes）は、Rust の最も重要な概念であり、
同時に初心者が最もつまずきやすい部分です。
ご提示いただいた文書は、LLM推論エンジン（c2000プロジェクト）における、
**Prefill（事前計算）フェーズの性能改善プロセスを詳細に記録したものです。
```

## 5. GPU で確認してほしいこと / 次の一手

- **最優先: `DENNO_PF_PLE_PIPE=1` を `DENNO_START_DROP_CACHE=1` で 1 ラウンド。**
  `7431cea95` の 32 本プールが PLE の gather を 1.85 s からどこまで下げるか。
  見るのは `[pre-kprof]` の `ple=` と `[q4ledger-pf] other` の露出。
  期待は 0.6-0.9 s (推定)。15% には単独では届かない見込みなので、
  **bg を併用しない**こと (暖かい側で +46%)。
- PLE の gather は「1 行 90 B × 166,728 回」のランダム pread。表 28.80 GB を
  RAM に置く案と `DENNO_PLE_PAR` を上げる案は task 189 で不採用
  (暖かい側で -0.06%)。今回の新しい理由は **cold では ple が wall の 20%**
  という点で、そこだけを狙うなら「プロンプト全体の行を先にまとめて読む」
  より「並列度を上げる」ほうが筋 (行数は同じ、queue だけ深くする)。
- `DENNO_PF_BG_WARM` は**起動直後の 1 本目だけ**を速くしたいなら有効
  (3,826 cold -28%)。ただし暖まった状態で +46% なので、既定 on にはできない。
  「最初の 1 リクエストのときだけ読む」形にすれば両立する可能性はある
  (推定、未実装)。
- lab の注意: `DENNO_START_DROP_CACHE=1` を COND に置くと毎回の起動で
  page cache が落ちるので、cold の A/B が交互に取れる (本タスクの方法)。
  ただし**その前のラウンドの状態を引き継ぐ**ので、t349c のように
  「expert は暖かく PLE だけ冷たい」混ざった状態も起きる。cold を見る
  ラウンドは全 COND に DROP を入れること。
- `DENNO_PF_STREAM_PREAD` は削ってよい (効かない)。残すなら A/B 用。

## 6. テスト

```
CARGO_TARGET_DIR=/tmp/t349 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1,236 passed / 0 failed / 102 ignored   (int-auto 6c19792cf の 1,230 + 本タスクの 6 本)
CARGO_TARGET_DIR=/tmp/t349 cargo test --release -p denno-engine --features cuda --test kernel_invariants -- --test-threads=1
  -> 106 passed / 0 failed                   (int-auto 6c19792cf の 102 + 本タスクの 4 本)
```

- 足したテスト: `model::start_warm` の 4 本 (bg の env の解釈 / `cap_ranges` /
  空なら不発 / **実際に読む** (背景スレッドの台帳 `bg_bytes` が計画バイトに
  達する))、`model::exp_pf` の 2 本 (stream_pread と ple_pipe の真理値表)、
  `kernel_invariants` の 4 本 (stream_pread の配線 6 点、bg の背景性と順序、
  PLE パイプの配線と二重読み防止、PLE パイプの 32 本プール)。
- GPU 無しで走るのは判断と配線まで。**実機で発火した証拠**は lab の
  `[exp-pf-stream] spread=1` / `[pf-bg-warm] start …` と
  `[pf-bg-warm] [start-prefetch] experts 28,141.3 MB … in 10.49s (2.7 GB/s)`
  (t349a の bg)。PLE パイプは**発火の証拠がまだ無い** (未測定)。

### 6.1 既知の赤 (本タスクの外)

- `model::pf_host::tests::nested_stage_guard_counts_once` が**稀に落ちる**
  (HEAD で 9 回中 1 回)。中身は 1 ms の sleep 2 回の入れ子で
  `wall < 3 ms` を課すテストで、負荷が高いと sleep が伸びて
  `wall=5,651,925 ns` になる (このコンテナはホスト共有)。
  `pf_host.rs` は本タスクで触っていない (`int-auto..HEAD` の diff は空、
  最後の変更は task 254)。直さない (範囲外)。
- 上の 1,236 は**緑だった回**の数字。8 回の繰り返しで 7 回が
  1,236 passed / 0 failed、1 回が上記の 1 failed。
