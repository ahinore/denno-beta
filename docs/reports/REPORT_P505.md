ADOPT: yes t505b t505c

# task 505 (dev2) — 既定モデル prefill: full 層 KV のホストミラーへの積みから
# **ゼロ埋め**を消した (`DENNO_PF_KV_NORESIZE` を既定 on)。
# `full` 段 **-11.6% / -11.0%** (3,826 cold、2 ラウンド)、14k は -6.9% / -6.7%。
# **壁時計は動かない** (3,826 は 1.88 s、14k は 5.02-5.07 s が COND の位置で動く)。
# タスクの目標 (壁 -2.5%) は満たしていない。採ったのは運用規則の
# 「段で 2 ラウンド 2% 以上 + 他が 1% 以上悪化しない + 本文が健全」の側。

branch **`br505`** = 開始時の `int-auto` **`890fc66d4`** (= merge: br496)。
`br501` は **merge していない**: あちらは `ADOPT: no t501b` (層優先 MoE の写しを
層の頭へ移す 3 mode が全部負け) で、残したのは既定 off の予約コードだけで、
本タスクの対象 (`AttnCache::push_many`) と 1 行も重ならない。終端で int-auto を
fetch し直したが **`890fc66d4` のまま**なので merge は無い。

数値はすべて GPU ラボの実測 (`SCENARIO=depth8k`、`PROFILE=1`、
`model=flash-next-UD-IQ3_XXS`、baseline COND = int-auto の既定 +
`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0
DENNO_GPU_WEIGHT_MB=12000`、先頭に捨て COND、末尾に base の繰り返し)。

lab 名と SHA:
- **t505a** = `890fc66d4` (x0/b/b2、段プロファイルだけ。`load_before=5.81`)
- **t505b** = `07df79468` (x0/b/nz/np/b2。`load_before=2.23`) — noresize の採否
- **t505c** = `2c052967b` (x0/b/np/fl/b2。`load_before=11.89`) — 並列の伸ばしと flat の使い回し

## 0. 結論 (7 行)

1. **採る (`ADOPT: yes`)。ただし段の勝ちで、壁は動かない。**
   `full` 段の壁: 3,826 cold で **108.5 / 108.3 -> 95.5 / 95.6 ms** (b -> np、
   2 ラウンドとも同じ向き、**-11.6% / -11.0%**)、14k cold で
   840.3 -> 782.8 ms (**-6.9%**、t505b は 868.0 -> 780.9 で -10.0%)。
2. **機序**: `AttnCache::push_many` は伸ばすときに `resize(need, 0.0)` で新しい領域を
   **ゼロ埋め**してから `copy_from_slice` で上書きしていた。14k の full 層は
   12 層 x 7 チャンク x 8 MB = **689 MB** を積むので、同じページを 2 回書いていた。
   `kv_push` 副区間の実測: 14k **305.0 -> 241.0 ms** (b -> np)、
   3,826 **29.6 -> 16.6 ms** (**-44%**)。ゼロ埋めの memset と、その分のフォルトが消えた。
3. **14k の `full` 段の cpu 列 502 ms の正体がこれだった**: `kv_wait` (D2H の
   イベント待ち) が 126-129 ms、`kv_push` (ホストの写し) が 224-348 ms。
   残り ~100 ms は qkv の起動と rope 表。P369 の注記「689 MB を 2.3 GB/s」と一致する。
4. **並列の伸ばし (`DENNO_PF_KV_PUSH_PAR`) は小さく勝つが既定 off のまま**:
   t505b の np は**効いていなかった** (伸ばす経路が `extend_from_slice` 1 本のまま
   だったので nz と同じ 250.6 ms)。`reserve` + 空き領域へ直接書く形に直した t505c では
   3,826 で **25.3 (nz, t505b) -> 16.6 ms**、14k で **250.2 -> 241.0 ms**。
   ただし nz と np を同じ run で並べたラウンドが無い (上限 3 ラウンドを使い切った)
   ので、既定は off。**次のタスクで nz/np を同一 run の 2 ラウンドで測れば採れる。**
5. **PLE の `flat` の使い回し (`DENNO_PF_PLE_FLAT_REUSE`) は効果なし**:
   `iter_top` は 146.2 (fl) 対 144.4 (b2) ms。既定 off のまま残す。
6. **14k の壁はこの run でも読めない**: 同じコードの b と b2 が 6.52 / 5.03 (t505b)、
   5.67 / 5.02 (t505c) で、**COND の位置だけで 20-25% 動く** (P393 / P482 と同じ)。
   壁は採否に使っていない。3,826 は t505c の 2 ラウンドとも全 COND 1.88 s で、
   12 ms の段の差は 0.6% にしかならない。
7. **値は 1 バイトも変えない**: md5 は 3 run x 全 COND x 全 6 文脈で **SAME**、
   errors 0、decode 10.6 ms/tok (深い点 12.9 / 11.6-11.8) も不変 (§5)。

## 1. なぜこの段を取ったか (Do 1 / Do 2)

`t505a` (開始時の int-auto) の `[q4ledger-pf]` を撮り直した。3,826 は experts が
壁の 54%、14k は experts 40% / ssm 33% / full 14% / other 12%。

| 段 (14k cold、t505a b2) | wall ms | GPU busy ms | host ms | 中身 / これまでの担当 |
| --- | ---: | ---: | ---: | --- |
| experts | 2,251.1 | 1,416.1 | 835 (露出) | 上げ 28.0 GB / 18.4 GB はディスク (P481/P486/P490/P497/P501 で決着) |
| ssm | 1,912.5 | 1,316.3 | 596 | GPU 律速 (P392/P397/P402/P441/P447/P496) |
| **full** | **824.9** | 551.9 | **502.1 (cpu)** | **未着手**。`kv_wait` と `kv_push` |
| other | 681.6 | 201.5 | 480 | iter_top 266 / pf_er 218 / teardown 89 / res_up 97 (P482/P489) |
| mix | 39.9 | 657.7 (別 stream) | 8 | P457/P464/P484 |

| 段 (3,826 cold、t505a b2) | wall ms | GPU busy ms | 中身 |
| --- | ---: | ---: | --- |
| experts | 1,070.8 | 529.6 | `up_wait` 877.6 (DDR の床、P481/P497) |
| ssm | 406.7 | 360.5 | GPU 律速 |
| mix | 248.7 | 183.1 | 別ストリーム |
| other | 143.4 | 51.1 | |
| full | 111.2 | 82.4 | |

- 14k で **名前の付いた host 列のうち最大**が `full` の cpu 502 ms だった
  (experts の露出 835 ms はディスク/DDR の床で P481/P497/P501 が閉じている、
  `other` は P482 が中身を割って P489 が pf_er を閉じた)。ここを取った。
- コードを読んで、`full` の cpu の大半が `AttnCache::push_many` の 689 MB の写しで、
  伸ばすたびに `resize(need, 0.0)` のゼロ埋めを挟んでいることが分かった
  (P369 の注記に「689 MB、約 2.3 GB/s」とある)。同じページを 2 回書いている。

## 2. 変えたこと (`07df79468` / `2c052967b`、既定 on + 2 本の off ゲート)

### 2.1 `DENNO_PF_KV_NORESIZE` (**既定 on**、`=0` で切り戻し)

- `crates/denno-engine/src/model/qwen4exp_attn.rs`: `push_many` の中身を
  `push_many_opts(layer, k, v, n, noresize, par)` に分け、`noresize` のときは
  インプレースで書ける範囲を `copy_from_slice`、足りない末尾は
  `reserve` + 空き領域へ直接書いてから `set_len` する (ゼロ埋めをしない)。
- キャッシュに入るバイト列・長さ・順序は従来と同一。テスト
  `task505_push_many_noresize_and_par_match_resize` が 3 経路
  (resize / noresize / noresize+par) を突き合わせる (伸ばす回とインプレースの回、
  `clear` の後の積み直し、容量超過と短い入力のエラーまで)。

### 2.2 `DENNO_PF_KV_PUSH_PAR` (既定 off) — 写しを rayon で分ける

- `par > 1` かつ 1 チャンクの写しが 2^18 要素以上のときだけ分ける
  (分割の境目は要素数だけで決まる)。伸ばす経路も `reserve` してから
  空き領域を `par_chunks_mut` で分ける (`extend_from_slice` は 1 本だった)。
- テスト `task505_push_many_parallel_growth_matches_serial` が直列と一致することを固定。

### 2.3 `DENNO_PF_PLE_FLAT_REUSE` (既定 off) — PLE の `flat` を使い回す

- `qwen4exp.rs` の PLE デバイス経路が層チャンクごとに `Vec::with_capacity(nt * e)`
  で作っていた 21 MB を、スレッド局所の grow-only バッファの貸し出しにする
  (`DENNO_PF_PLE_REUSE` と同じ「全要素を書き直す」形)。**測って効かなかった**。

### 2.4 台帳 (診断。既定の経路は不変)

- `q4ledger` に副区間 **`kv_wait`** (D2H のイベント待ち) と **`kv_push`**
  (ホストの写し) を足した (`SUB_N` 43 -> 45、既存の添字は動かしていない)。
  差し込みは `pf_kv_wb_with` / `pf_kv_wb_copy_into` の待ちと `push_many` の写し。
- これが今回の採否の根拠。`full` 段の cpu 502 ms が「待ち 126」と「写し 305-348」
  に割れ、写しの側だけが消えたことが読める。

## 3. lab の結果

### 3.1 `full` 段と `kv_push` (採否の本体)

| run / COND | 3,826 full 壁 ms | 3,826 `kv_push` | 14k full 壁 ms | 14k `kv_push` |
| --- | ---: | ---: | ---: | ---: |
| t505b b | 109.9 | 29.8 | 868.0 | 347.9 |
| t505b **nz** | **105.0** | **25.3** | **780.9** | **250.2** |
| t505b np (par は未発火) | 105.6 | 25.7 | 795.6 | 250.6 |
| t505b b2 | 109.2 | 30.4 | 838.5 | 292.7 |
| t505c b | 108.5 / 108.3 | 29.6 / 29.4 | 840.3 | 305.0 |
| t505c **np** | **95.5 / 95.6** | **16.6 / 16.6** | **782.8** | **241.0** |
| t505c **fl** | 96.4 / 97.1 | 17.5 / 18.2 | **775.3** | **223.9** |
| t505c b2 | 108.9 / 108.1 | 30.0 / 29.3 | 831.6 | 284.7 |

- 3,826 は t505c の **2 ラウンドとも** b/b2 が 108.1-108.9、np が 95.5-95.6。
  `kv_push` は 29-30 -> 16.6 ms (**-44%**)。`kv_wait` は 35.3 ms で不変 (期待どおり)。
- 14k は 1 COND 1 サンプル。`kv_push` は 305.0 (b) / 284.7 (b2) に対し
  241.0 (np) / 223.9 (fl) で **-21% / -15% (対 b2)**。`full` 段も -6.7%。
- `fl` と `np` の差は `iter_top` の 1.8 ms で、flat の使い回しは効いていない。

### 3.2 壁 (採否に使っていない)

| run / COND | 3,826 cold r1 / r2 (s) | 14k cold (s) | decode 3,826 後 | decode 14k 後 |
| --- | --- | --- | ---: | ---: |
| t505b x0 | 2.66 / 1.88 | 8.16 | 13.0 | 13.3 |
| t505b b | 2.06 / 1.88 | 6.52 | 13.1 | 13.0 |
| t505b nz | 2.00 / 1.89 | 5.82 | 13.0 | 12.6 |
| t505b np | 1.92 / 1.88 | 5.08 | 13.0 | 12.2 |
| t505b b2 | 1.87 / 1.88 | 5.03 | 12.9 | 12.3 |
| t505c x0 | 2.53 / 1.88 | 7.88 | 13.1 | 13.5 |
| t505c b | 1.88 / 1.88 | 5.67 | 12.9 | 12.5 |
| t505c np | 1.88 / 1.88 | 5.07 | 12.9 | 12.2 |
| t505c fl | 1.88 / 1.88 | 5.05 | 12.9 | 12.7 |
| t505c b2 | 1.88 / 1.88 | 5.02 | 12.9 | 12.5 |

- 3,826 は t505c で **全 COND 1.88 s** (2 ラウンド)。12 ms の段の勝ちは 0.6%。
- 14k は COND の位置だけで 20-25% 動く (同じコードの b と b2 が 6.52 / 5.03)。
  np/fl は b より速く b2 と同帯 = **悪化はしていないが、勝ちも読めない**。
- t505c は `load_before=11.89` で、b の `iter_top` が 468.3 ms (np/fl/b2 は 140-146)
  という外乱が出た。外乱のある列は採否に使っていない。

### 3.3 VERDICT 行 (x0 は捨て COND なので `vs x0` は使わない)

```text
t505b: VERDICT b vs x0 OK decode +0.0% prefill +1.8% md5 SAME errors 0
       VERDICT nz vs x0 OK decode +0.0% prefill +0.9% md5 SAME errors 0
       VERDICT np vs x0 OK decode +0.4% prefill +0.0% md5 SAME errors 0
       VERDICT b2 vs x0 OK decode +0.0% prefill +1.8% md5 SAME errors 0
t505c: VERDICT b vs x0 OK decode +0.0% prefill -0.9% md5 SAME errors 0
       VERDICT np vs x0 OK decode -0.4% prefill +0.9% md5 SAME errors 0
       VERDICT fl vs x0 OK decode +0.0% prefill -0.9% md5 SAME errors 0
       VERDICT b2 vs x0 OK decode -0.4% prefill +0.0% md5 SAME errors 0
```

## 4. 数値の扱い

- 採否は**段** (`full` の壁と `kv_push`) で読んだ。3,826 は 2 ラウンド x 2 サンプル。
  14k は 1 サンプルなので、同じ run の b と b2 に挟まれた位置でだけ読んだ。
- 壁時計は使っていない (3,826 は全 COND 同一、14k は位置効果が 20-25%)。
- `kv_push` / `kv_wait` は新しい副区間の実測 (ホストの壁時計)。
- マイクロベンチは取っていない (GPU の無いコンテナ)。バイト数は
  `AttnCache` の行長 x 位置数から: 14k は 12 層 x 14,018 位置 x 2 (k/v) x 1,024 B
  = 689 MB。3,826 は 188 MB。
- t505c の `other` は b が外乱 (load 11.89) で 871.6 ms、np/fl/b2 が 560-595 ms。
  同じコードの np と fl でも `res_up` が 122.5 / 101.9 ms と開くので、
  この run の 5% 級の差は読まない。

## 5. 出力テキスト (健全性)

t505c の `fl` の 14k (`c8000_turn2`、md5 `199f23b3` で baseline と同一):

> ご提示いただいた文書は、LLM推論エンジン（`denno-engine）における **Prefill（事前計算）
> 処理の性能最適化の記録です。 以下に、この文書の技術的詳細を、**「問題の特定」、
> 「最適化の段階的変遷」、「計測の罠と…

3,826 の回答 (`c2000_turn2`) は `…Prefill（事前計算）フェーズの性能最適化の歴史を
詳細に記録した技術レポートです。`、短い decode は `# B-Trees: A Comprehensive
Explanation`、`Rust の「所有権（Ownership）とライフタイム（Lifetime）は…`。
**3 run の全 COND x 全 6 文脈で md5 が SAME**、`!!!!` / 空出力 / プロンプトの反復は無し、
errors 0。

## 6. テスト

```text
CARGO_TARGET_DIR=/tmp/t505 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1353 passed; 0 failed; 103 ignored     (task 505 で 4 本追加)
同 --test kernel_invariants
  -> 144 passed; 0 failed
(関門の外) CARGO_TARGET_DIR=/tmp/t505nc cargo build --workspace --release
```

- 追加したテスト: `task505_push_many_noresize_and_par_match_resize` /
  `task505_push_many_parallel_growth_matches_serial` /
  `task505_kv_push_gates_truth_table` / `task505_kv_wb_sub_names_are_wired`
  (+ `ple_reuse_scratch_contract` に flat_reuse の真理値表)。
- 既知の赤 (`qwen4exp_decode spec_batch_stops_at_eos`、`qwen4exp_params` の validate 2 件)
  は今回も出ていない。
- 測った木 (`2c052967b`) と最終 commit の差は「`DENNO_PF_KV_NORESIZE` の既定を on に
  した」1 点で、測った COND (nz / np) と同じ分岐に入る。

## 7. 次の担当者へ

1. **残っている手 (順に)**
   - `DENNO_PF_KV_PUSH_PAR` (並列の伸ばし) を **nz と np を同一 run の 2 ラウンド**で
     測る。3,826 の `kv_push` は 25.3 -> 16.6 ms、14k は 250 -> 241 ms と一貫して
     同じ向きだが、同じ run での対が無い。
   - `kv_wait` (**14k で 126-129 ms、3,826 で 35 ms**)。D2H のイベント待ちで、
     `pf_full_out` の後に待っている。ここを 1 層ぶん遅らせる (pinned 2 枚の
     ダブルバッファ) と、84 回 x 1.5 ms が GPU の裏へ隠れる (**推定 -100 ms**)。
   - `iter_top` (14k で 140-146 ms、3,826 で 20-22 ms)。PLE 層は 48 層中 2 層で、
     `flat` の使い回しでは動かなかった。残りは `ples_all` の gather 待ちと
     `pf_res_park_load`。`[pre-kprof]` と突き合わせて中身を割ること。
   - `pf_teardown` (14k で 90-105 ms) は「次のリクエストのための戻し」。
     次のリクエストの 1 トークン目まで遅らせれば prefill から消える (P482 §6-3)。
2. **`full` 段の残りは ~100 ms** (qkv の起動 + rope 表 + out の D2H)。`kv_push` を
   消したので、次に `full` を触るならそこ。
3. **計測の注意**: 14k の壁は COND の位置で 20-25% 動く (今回も b 6.52 / b2 5.03)。
   壁で採否を読まないこと。3,826 も 1.87-2.66 s と動くが、t505c のように
   `load_before` が低い run では 1.88 s に揃う。
