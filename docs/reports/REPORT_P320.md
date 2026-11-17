ADOPT: yes t320a t320b t320c

# task 320 — IQ3_S prefill: expert の上げを専用スレッドへ出して 3,826 cold -5.4%

branch `br320` は開始時の `int-auto` (`ba85315db` = merge: br317) から。
**br311 は br317 に merge 済みで、br317 は int-auto に入っている** ので、base は
wy4 (`DENNO_PF_SCAN_WY4`) と MIXKEEP / MIXKEEP_SSM が既定 on の状態。br311 の
merge は不要だった。最後に int-auto `6465bf554` (merge: br318) を merge した
(衝突なし。`build_ptx.sh` の `targets_all` も両方残っているのを確認).
lab 3 ラウンド: **t320a** (`ba85315db`、4 COND)、**t320b** (`9d8717cdd`、5 COND)、
**t320c** (`1a658d35b`、5 COND)。数値はすべて GPU ラボの実測。

## 0. 結論 (3 行)

- **採る。** 3,826 cold は同じ run の base/base2 の中央値 **2.77 s → 2.62 s
  (-5.4%)** (t320b と t320c、各 2 ラウンド)、14k cold も **6.98 → 6.77 s
  (-3.0%、3 ラウンドの中央値)** で悪化なし。decode は 17.1-17.2 で変わらない。
- 取ったのは 1 つ。**`DENNO_PF_STREAM_UT` を既定 on** — expert の pinned 写しと
  DMA を専用スレッドへ出し、計算スレッドはカーネルを積むだけにする。台帳で
  `exposed_host` が 1,418 → 1,228 ms (3,826)、2,110 → 1,777 ms (14k) に縮み、
  `gpu_busy` は 1,367 → 1,375 ms で変わらない (縮んだのはホストの待ちだけ)。
- 先読み 2 (`DENNO_PF_STREAM_LOOKAHEAD`) と `DENNO_PF_UP_DECOUPLE` も試したが
  **どちらも追加の効果なし** (2.647 / 2.61 s)。両方とも既定 off のまま残した (§5)。

## 1. 前提

- **読んだもの**: `REPORT_P317.md` (と その §1 が参照する `REPORT_P311.md`)。
  P317 の「次に削るなら 14k の `exposed_host` 1,882 ms。内訳は expert streaming の
  staging コピー (~1,110 ms、床) + PLE 282 + combine 228」が今回の出発点。
  **P317 の数字をそのまま信じず、現 int-auto で PROFILE=1 を取り直した** (§3)。
- 基準 COND = タスク指定の 4 本 (`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000
  DENNO_EXPERT_GPU_MIN_HIT=0 DENNO_GPU_WEIGHT_MB=12000`)。先頭に捨て COND (`x0`)、
  末尾に baseline の繰り返し (`base2`)。`SCENARIO=depth8k`、`PROFILE=1`。
- 台帳の読み方: `[q4ledger-pf] close` の `wall = gpu_busy + exposed_host`。
  `exposed_host` は「GPU がアイドルでホストだけが動いていた時間」で、
  `[pf-host]` がその中の experts 段の内訳を出す。
- ホスト負荷: t320a `load_before=4.72`、t320b `1.93`、t320c `7.11`。
  **同じ COND でも run の最初のラウンドは 3.4 s 前後に出る** (t320c の base r1
  3.43 / r2 2.77、t320a の base r1 3.46 / r2 2.77)。採否は r2 同士と、
  t320b の clean な run を主に読んだ。

## 2. 変えたこと (2 コミット + merge + REPORT)

1. `9d8717cdd` **`DENNO_PF_STREAM_LOOKAHEAD` (新規、既定 1)** — 上げ専用スレッド
   への依頼を `gi + 1` から `gi + look` へ (1..=2)。1 つ先だけだと、計算スレッドが
   待つ `ev_up` は「1 反復前に依頼したグループ」のもので、その写しがまだ終わって
   いない。**結果は変わらなかった** (§4) ので既定 1 のまま。
2. `1a658d35b` **`DENNO_PF_STREAM_UT` を既定 on** + **`DENNO_PF_UP_DECOUPLE`
   (新規、既定 off)** — 後者はワーカーがホストで待つ対象を「pinned を前回読んだ
   DMA」だけにし、VRAM ステージングの再利用 (カーネルの完了) を
   `cuStreamWaitEvent` で上げストリームに守らせる。順序は 写し → stream wait → DMA。
   **効果なし** (§4)。`[exp-pf-stream]` に `dec=` を足した (発火の確認用)。
3. `9232a40e9` int-auto `6465bf554` の merge。
4. `REPORT_P320.md` (この文書、最終コミット)。

値の扱い: 2 本とも**デバイスへ渡るバイト列も丸め順も変えない** (写しと DMA を
どのスレッドがいつ積むかだけ)。md5 は全 COND で SAME (§6)。

## 3. 数値 (lab、すべて実測)

### 3,826 cold (c2000_cold、s。r1 / r2 と中央値)

| run | cond | r1 / r2 | 中央値 |
| --- | --- | --- | ---: |
| t320a (load 4.7) | base (ut=0) | 3.46 / 2.77 | 3.12 |
| t320a | ut=1 | 3.16 / 2.63 | 2.90 |
| t320a | base2 (ut=0) | 2.86 / 2.78 | 2.82 |
| t320b (load 1.9) | base (ut=0) | 2.77 / 2.79 | 2.78 |
| t320b | ut=1 | 2.64 / 2.65 | **2.645** |
| t320b | ut=1 + 先読み 2 | 2.65 / 2.65 | 2.65 |
| t320b | base2 (ut=0) | 2.80 / 2.79 | 2.795 |
| t320c (load 7.1) | base (ut=0) | 3.43 / 2.77 | 3.10 |
| t320c | **new = 既定 (ut=1)** | 2.60 / 2.60 | **2.60** |
| t320c | ut=1 + decouple | 2.61 / 2.61 | 2.61 |
| t320c | base2 (ut=0) | 2.76 / 2.76 | 2.76 |

- **既定 (ut=1) 対 ut=0: 2.77 → 2.62 s = -5.4%** (t320b と t320c の clean な
  7 サンプル / 4 サンプルの中央値)。r2 同士の対でも t320b -4.3% / t320c -5.8%。
- t320a (負荷の高い run) も r2 同士で 2.77 → 2.63 (-5.1%)、中央値では
  base2 2.82 → ut 2.90 (ut の r1 が 3.16 と出たため)。方向は同じ。

### 14k cold (c8000_cold、s、ラウンド 1 の 1 サンプル)

| run | base (ut=0) | ut=1 | 差 |
| --- | ---: | ---: | ---: |
| t320a (load 4.7) | 8.56 | 7.30 | -14.7% |
| t320b (load 1.9) | 6.80 | 6.75 (先読み 2) / 6.77 (ut=1) | -0.4% / -0.7% |
| t320c (load 7.1) | 6.98 | 6.66 | -4.6% |
| 中央値 | **6.98** | **6.77** | **-3.0%** |

(1 run 1 サンプルなので位置に弱い。base2 は 8.72 / 6.84 / 6.78。)

### 台帳 (3,826、t320c の base r2 と new r1)

| 項目 | base (ut=0) | new (既定) |
| --- | ---: | ---: |
| wall (ms) | 2,764 | 2,604 |
| gpu_busy (ms) | 1,367 | 1,375 |
| exposed_host (ms) | 1,398 | 1,228 |
| stage_copy (ms) | 900 | 0 (ワーカーへ) |
| up_wait (ms) | 410 | 1,111 |
| 写しの合計 copy_s | 0.90 | 1.08 |
| 実 H2D の探針 probe_GBs | 16.8 | 54.9 |
| 転送量 up_MB | 22,653 | 22,653 |

### 台帳 (14k、t320c の base と new)

| 項目 | base (ut=0) | new (既定) |
| --- | ---: | ---: |
| wall (ms) | 6,981 | 6,658 |
| gpu_busy (ms) | 4,870 | 4,880 |
| exposed_host (ms) | 2,110 | 1,777 |
| stage_copy / up_wait (ms) | 1,109 / 513 | 0 / 1,415 |
| copy_s | 1.11 | 1.32 |
| up_MB | 28,378 | 28,378 |

### 目標の判定

| 条件 | 目標 | 実測 | 判定 |
| --- | --- | --- | --- |
| 3,826 cold | -2.5% 以上 | **-5.4%** (中央値、2 ラウンド x 2 run) | **達成** |
| 14k cold | -1% 以内の悪化 | **-3.0%** (中央値) | OK |
| 短 decode | -1% 以内の悪化 | 17.1-17.2 (base と同じ、0.1 ms 刻み) | OK |
| errors / text | 0 / sane | 0 / `looks ok` 全 COND | OK |
| md5 | SAME (値は変えない) | 全 COND 同一 | OK |

## 4. 機序 — 何が縮んで、何が縮まなかったか

- **縮んだのはホストの待ちだけ。** 層優先の prefill は 1 プロンプトで 22.65 GB
  (3,826) / 28.38 GB (14k) の expert 重みを上げる。内訳は「page cache から pinned
  への写し (3,826 で 900 ms)」+「DMA (探針 17 GB/s)」で、直列だと 1,305 ms。
  専用スレッドに出すと写しと DMA が重なり、`exposed_host` が 1,398 → 1,228 ms。
  `gpu_busy` が動かない (1,367 → 1,375) ので、**速くなったのはホストの直列待ち**
  だと確定できる。
- **写しは DDR 律速の床。** 22.65 GB / 1.08 s = 21 GB/s で、読み (page cache) と
  書き (write-combined pinned) を合わせると DDR の実効帯域にほぼ張り付いている。
  P227 §2 の実測 (写し 21 GB/s 単独、DMA 57 GB/s 単独、重ねると DMA 27 GB/s) と
  同じ値。ここを超えるには**バイトを減らす**しかない (今回はやっていない)。
- **先読み 2 が効かない理由**: 依頼を深くしても、ワーカーがスロット再利用
  (`ev_use` = カーネルの完了) をホストで待つ限り写しはカーネルの後ろに付く。
  t320b で ut=1 2.645 / ut=1+先読み 2 2.65 と差が出なかったのはこれで説明できる。
- **decouple が効かない理由**: `ev_use` のホスト待ちを GPU 側の
  `cuStreamWaitEvent` に置き換えても、1 グループあたりの律速が
  「写し 3.1 ms 対 DMA 3.2 ms」の転送そのものなので、カーネル (1.7 ms/グループ)
  が転送の裏に入っても壁時計は動かない。t320c で dec 2.606-2.607 / new
  2.603-2.604 と一致した。**分離は無害 (`md5` SAME、errors 0) だが無効**。
- 残り: 3,826 の wall 2,604 ms の内訳は attn (mix 205 + full 130 + ssm 441 =
  776 ms GPU) + MoE 段 (転送 1,111 + カーネル 598) + PLE 143。MoE 段はまだ
  「転送とカーネルの和」に近く、理論床は転送 1,111 + attn 776 + PLE 143 ≈ 2,030 ms
  (推定)。

## 5. なぜ task 165 の「ut は効かない」を覆すか

- task 165 は **層優先になる前のチャンク優先**で測っていた (t165a: ut=1 102.10 s、
  ut=0 99.97 s、14k)。当時は 1 グループ = 「全部写す → 1 回 DMA → 待つ」で、
  依頼も 1 つ先だけ、ステージングも 512 MiB を 3 枚。今回は
  **層優先 (task 227/265) + グループ 64 + write-combined + 3 枚**の経路で、
  上げるバイトが 158.5 GB → 28.4 GB (14k) に減った後の話である。
  同じ「専用スレッド」でも、写しと DMA の比率 (3.1 : 2.5) と 1 グループの
  カーネル時間 (1.7 ms) が違うため、隠れる割合が変わった。
- 数値の裏付け: t165a の当時は `enq_s=9.53` (積み込みが前の DMA を待っていた)。
  今回の ut=1 は `enq_s=0.05`、`ready_s=0.27`、`wait_s=0.84` で、
  積み込みは詰まっていない。

## 6. md5 とテキスト

t320c の md5 (5 COND すべて同一):

| 出力 | md5 |
| --- | --- |
| a | 65ededc3 |
| b | f18c75fc |
| c2000_cold | a91e17da |
| c2000_turn2 | 5a8f5488 |
| c8000_cold | e2292eb6 |
| c8000_turn2 | 561c6da8 |

- **全 COND で SAME**。値も丸め順も変えないので、wy4 の丸め (P311) による
  ベースラインの DIFF (d8706fac → a91e17da) はそのまま。
- 本文 (new の 14k cold、TXTD):
  「ご提示いただいた文書は、LLM推論エンジン（おそらく \`denno-engine）の
  **Prefill（事前計算）における性能最適化の記録です。…### 1. 核心的な結論」 —
  base と同じ書き出しで内容も同じ。TXT (B-tree) / TXTB (Rust の所有権) も全 COND 同一。

## 7. テスト

```
CARGO_TARGET_DIR=/tmp/t320 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1,215 passed / 0 failed / 102 ignored   (int-auto 6465bf554 の merge 後)
CARGO_TARGET_DIR=/tmp/t320 cargo test --release -p denno-engine --features cuda --test kernel_invariants -- --test-threads=1
  -> 99 passed / 0 failed
```
追加した 2 本:
- `stream_gate_parses_values` に追記 — 先読みの解釈 (1..=2 に丸める) と
  `DENNO_PF_UP_DECOUPLE` (明示的な真値だけ on)。
- `task320_upload_thread_lookahead_is_wired` — 依頼ループが `gi + look` を
  ゲートから取ること、旧い `gi + 1` が残っていないこと、`PF_STAGE_SLOTS = 3` と
  上限 2 の対応。
- `task320_up_decouple_orders_copy_before_stream_wait` — 写し → `cuStreamWaitEvent`
  → DMA の順序 (逆にすると直列のままで、この task の効果が消える)。
**発火の証拠**は lab の台帳: `ut=1` の run は `[exp-pf-stream] ut=1`、
`stage_copy` が 0 になり `up_wait` が 406 → 1,111 ms、`probe_GBs` が 16.9 → 54.9。
`dec=1` の run は `dec=1` で、`copy_s` が 1.08 → 1.34 に増え、`wait_s` が
0.84 → 1.10 (GPU 側の待ちに移った)。既定 off の run は `ut=0 dec=0`。
既知の赤 (qwen4exp_decode の spec_batch_stops_at_eos、qwen4exp_params の
validate 2 件) は踏んでいない (この 2 つのターゲットは回していない)。

## 8. GPU で確認してほしいこと / 次にやるなら

1. 実機 (8098) の 3,826 で `exposed_host` 1,398 → 1,228 ms、14k で
   2,110 → 1,777 ms になること。既定で有効。切り戻しは `DENNO_PF_STREAM_UT=0`。
2. **UD-Q4_K_XL でも既定 on のまま**測ってほしい (K 系は `stream_wc_for` が
   write-combined を既定 off にする。写しの側が変わるので、同じ -5% は出ない
   可能性がある。推定)。
3. 次に削るなら MoE 段の「転送 + カーネル」の重なり (3,826 で 380 ms 程度、推定)。
   ただし写しは DDR 律速の床なので、バイトを減らす (VRAM 常駐 expert を prefill で
   使う) 以外に大きな余地はない。VRAM の作業域借用が LRU を追い出しており、
   `[exp-pf-stream]` のとおり今回は 1 バイトも再利用していない (22.65 GB = 全 expert)。
   P250 / P265 の追い出し階段を繰り返す危険があるので、着手するなら独立タスクで。

## 9. lab の VERDICT 行

t320a (`ba85315db`、4 COND):
```
VERDICT base vs x0 OK decode -0.7% prefill -0.8% md5 SAME errors 0
VERDICT ut vs x0 OK decode -0.9% prefill -0.8% md5 SAME errors 0
VERDICT base2 vs x0 OK decode -0.7% prefill -0.8% md5 SAME errors 0
VERDICT_DEPTH base c2000_cold vs x0 decode +1.8% prefill +9.9%
VERDICT_DEPTH base c2000_turn2 vs x0 decode -0.5% prefill +1.2%
VERDICT_DEPTH base c8000_cold vs x0 decode -19.2% prefill -12.7%
VERDICT_DEPTH base c8000_turn2 vs x0 decode -13.2% prefill -2.0%
VERDICT_DEPTH ut c2000_cold vs x0 decode +5.4% prefill +2.1%
VERDICT_DEPTH ut c2000_turn2 vs x0 decode +3.5% prefill +0.0%
VERDICT_DEPTH ut c8000_cold vs x0 decode -15.2% prefill -25.6%
VERDICT_DEPTH ut c8000_turn2 vs x0 decode -6.4% prefill -2.0%
VERDICT_DEPTH base2 c2000_cold vs x0 decode +0.0% prefill -0.5%
VERDICT_DEPTH base2 c2000_turn2 vs x0 decode -0.5% prefill +0.0%
VERDICT_DEPTH base2 c8000_cold vs x0 decode -16.2% prefill -11.1%
VERDICT_DEPTH base2 c8000_turn2 vs x0 decode -8.5% prefill -2.0%
VERDICT baseline x0 errors 0 text ok
```

t320b (`9d8717cdd`、5 COND):
```
VERDICT base vs x0 OK decode -0.7% prefill +1.7% md5 SAME errors 0
VERDICT ut1 vs x0 OK decode +0.7% prefill +0.8% md5 SAME errors 0
VERDICT ut2 vs x0 OK decode +1.4% prefill +0.8% md5 SAME errors 0
VERDICT base2 vs x0 OK decode +0.2% prefill +0.8% md5 SAME errors 0
VERDICT_DEPTH base c2000_cold vs x0 decode +0.0% prefill -0.4%
VERDICT_DEPTH base c2000_turn2 vs x0 decode +0.0% prefill +1.2%
VERDICT_DEPTH base c8000_cold vs x0 decode -7.7% prefill +0.0%
VERDICT_DEPTH base c8000_turn2 vs x0 decode -3.8% prefill +0.0%
VERDICT_DEPTH ut1 c2000_cold vs x0 decode +1.1% prefill -5.2%
VERDICT_DEPTH ut1 c2000_turn2 vs x0 decode +0.3% prefill +0.0%
VERDICT_DEPTH ut1 c8000_cold vs x0 decode -8.9% prefill -0.4%
VERDICT_DEPTH ut1 c8000_turn2 vs x0 decode -3.8% prefill +0.0%
VERDICT_DEPTH ut2 c2000_cold vs x0 decode +0.0% prefill -5.0%
VERDICT_DEPTH ut2 c2000_turn2 vs x0 decode +0.3% prefill +0.0%
VERDICT_DEPTH ut2 c8000_cold vs x0 decode -8.9% prefill -0.7%
VERDICT_DEPTH ut2 c8000_turn2 vs x0 decode -3.8% prefill +0.0%
VERDICT_DEPTH base2 c2000_cold vs x0 decode +0.7% prefill +0.2%
VERDICT_DEPTH base2 c2000_turn2 vs x0 decode +0.0% prefill +0.0%
VERDICT_DEPTH base2 c8000_cold vs x0 decode -8.3% prefill +0.6%
VERDICT_DEPTH base2 c8000_turn2 vs x0 decode -3.8% prefill +2.1%
VERDICT baseline x0 errors 0 text ok
```

t320c (`1a658d35b`、5 COND、既定 on の確認回):
```
VERDICT base vs x0 OK decode +0.9% prefill +0.8% md5 SAME errors 0
VERDICT dec vs x0 OK decode +0.7% prefill +0.0% md5 SAME errors 0
VERDICT new vs x0 OK decode +0.9% prefill +0.0% md5 SAME errors 0
VERDICT base2 vs x0 OK decode +0.5% prefill -0.8% md5 SAME errors 0
VERDICT_DEPTH base c2000_cold vs x0 decode +5.1% prefill +6.5%
VERDICT_DEPTH base c2000_turn2 vs x0 decode +4.3% prefill +0.0%
VERDICT_DEPTH base c8000_cold vs x0 decode -9.4% prefill +2.8%
VERDICT_DEPTH base c8000_turn2 vs x0 decode -9.1% prefill -4.2%
VERDICT_DEPTH dec c2000_cold vs x0 decode +0.7% prefill -10.3%
VERDICT_DEPTH dec c2000_turn2 vs x0 decode -0.3% prefill +0.0%
VERDICT_DEPTH dec c8000_cold vs x0 decode -9.4% prefill -1.9%
VERDICT_DEPTH dec c8000_turn2 vs x0 decode -12.9% prefill -4.2%
VERDICT_DEPTH new c2000_cold vs x0 decode +0.0% prefill -10.7%
VERDICT_DEPTH new c2000_turn2 vs x0 decode +0.3% prefill +0.0%
VERDICT_DEPTH new c8000_cold vs x0 decode -9.9% prefill -1.9%
VERDICT_DEPTH new c8000_turn2 vs x0 decode -12.9% prefill -4.2%
VERDICT_DEPTH base2 c2000_cold vs x0 decode +0.4% prefill -5.2%
VERDICT_DEPTH base2 c2000_turn2 vs x0 decode -0.3% prefill +0.0%
VERDICT_DEPTH base2 c8000_cold vs x0 decode -10.5% prefill -0.1%
VERDICT_DEPTH base2 c8000_turn2 vs x0 decode -13.4% prefill -4.2%
VERDICT baseline x0 errors 0 text ok
```
(VERDICT は「先頭の COND = 冷えて負荷のある x0」との比較なので、採否は同じ run の
base/base2 との対で読んだ — §3 の表。t320c の `base` r1 (3.43 s) と
`x0` r1 (3.21 s) は run 先頭の負荷で、r2 はどちらも 2.6-2.8 s。)
