ADOPT: yes t357a t357b t357c

# task 357 — IQ3_S prefill、PLE gather の本数を 32 → 96 に

branch `br357` は開始時の `int-auto` `1ce57fb2c` (br353 merge 済み) から。
**br353 は既に int-auto に merge されていた** (merge commit `1ce57fb2c`) ので
merge は不要だった。int-auto は最後にもう一度 fetch したが `1ce57fb2c` のままで、
最終 merge も無し。数値はすべて GPU ラボの実測 (`SCENARIO=depth8k`, `PROFILE=1`,
`swift15-abl-IQ3_S`)。lab 名は t357a / t357b / t357c (3 ラウンド、上限 3)。

## 0. 結論 (5 行)

- **14k cold が 8.09 → 7.81 s (-3.5%)。3 ラウンドとも再現** (t357a -3.4% /
  t357b -3.5% / t357c -3.0%)。3,826 cold も 8.41 → 8.27 (-1.7%) / 8.40 → 8.30
  (-1.2%) / 8.50 → 8.37 (-1.5%) で**両段とも良くなった** (悪化なし)。
- 効いたのは **PLE gather の並列度**。`ple_par_parts` の上限を 32 本固定から
  `DENNO_PF_PLE_THREADS` (既定 **96**) にした。`[pre-kprof]` の ple は 14k で
  680 → 425 ms、3,826 で 330 → 216 ms。**層ループの前に直列で露出する**時間
  なので、そのまま wall が減る。
- 値は変えない (読む行・重複排除・綴じ方・デコードは同一)。md5 は t357a/b/c の
  **全 COND・全リクエストで SAME** (a91e17da / 5a8f5488 / e2292eb6 / 561c6da8)、
  errors 0、text sane。
- **暖かい page cache では変化なし** (t357c の warm 対: 32 本 2.51 / 6.43 s、
  96 本 2.51 / 6.44 s。ple は 13.5 → 14.1 / 53.9 → 52.8 ms)。つまりこの変更は
  「page cache が cold のとき」だけに効く。decode は 16.6 ms/tok (短) /
  21.0 (3,826 後) / 18.0 (14k 後) で全 COND 同一。
- 128 本は gather が 386 ms まで下がるが、t357b の 1 回目で 14k が 11.21 s に
  跳ねた (decode も 33.9 ms/tok = ホストの外乱) ので既定は 96 にした。

## 1. Do 1: 基準の取り直しと段ごとのプロファイル

### 1.1 基準の測り方 (task 353 と同じ)

ラボの depth8k は同じ server で 3,826 → 14k を続けて回すので、page cache が
暖まると 14k が 6.4 s まで落ちる (これは supervisor の tint の 2.50 / 6.43 s と
同じ数字)。cold の A/B を取るため、**全 COND に `DENNO_START_DROP_CACHE=1`** を
置いて毎回の server 起動で clean page を落とした。全 COND が同じ冷え方になる。

### 1.2 段ごと (t357a x0、`[q4ledger-pf]`、ms)

3,826 cold (wall 8,424): `[pre-kprof]` ple=330 / layer=8,000。

| 段 | wall | exposed | 備考 |
| --- | ---: | ---: | --- |
| experts | 6,934 | 6,387 | 82.3%。`[exp-pf-stream] up_MB=22,653 copy_s=6.73` |
| ssm | 602 | 162 | |
| other | 589 | 512 | うち ple 144 |
| full | 174 | 44 | |
| close | 8,424 | — | gpu_busy 1,396 / exposed_host 7,110 |

14k cold (wall 8,098): `[pre-kprof]` ple=677 / layer=7,059。

| 段 | wall | exposed | 備考 |
| --- | ---: | ---: | --- |
| experts | 2,949 | 1,450 | 36.4%。`up_MB=28,378 copy_s=2.45` |
| ssm | 2,516 | 936 | 31.1%。busy 1,577 (実 GPU 仕事) |
| other | 1,450 | 1,168 | 17.9%。**うち層ループの外の PLE gather 677** |
| full | 1,162 | 355 | |
| mix | 20 | 0 | GPU busy 759 (隠れている) |
| close | 8,098 | — | gpu_busy 4,899 / exposed_host 3,938 |

### 1.3 どの段を取るか (他を取らなかった理由)

- **3,826 cold は disk の床**: 22.65 GB の expert を 3.2-3.5 GB/s で読む 6.5-7.0 s
  が wall 8.4 s の 8 割。pread (`DENNO_PF_STREAM_PREAD`、t353a で -2.3%)、
  `MADV_WILLNEED` 先読み、32 本 (P325) はどれも ±2% に収まっている。P227/P309 で
  `STREAM_SRC=page|reg` も負け (MAP_SHARED は pin できない)。ここは触らない。
- **14k の experts (exposed 1,450)** は「WC pin への写し 2.45 s (11.6 GB/s) +
  DMA 0.49 s − GPU 1.49 s」で説明できる。WC は t165b/c で非 WC より良いと
  測られている。expert の集合は router の後でしか分からないので、これ以上
  前へ倒せない。
- **14k の ssm (2,516 / busy 1,577)** は実 GPU 仕事で、3,826 → 14k のスケール
  (601 → 2,516) がトークン数にほぼ比例する。
- 残るのは **層ループの前に直列で露出する PLE gather 677 ms**。task 353 で
  32 本にして 1,851 → 681 ms まで来たが、これは 1 行 90 B のランダム pread が
  レイテンシ律速で、**pread はカーネル内で待つので本数 = 同時に飛ぶ I/O 数**。
  32 本は 8C/16T の 2 倍でしかない。ここを取った。

## 2. Do 2: 変えたこと

### 2.1 `DENNO_PF_PLE_THREADS` (既定 96) — 唯一の変更

- `exp_pf`: `PLE_THREADS_ENV` / `DEFAULT_PLE_THREADS=96` /
  `ple_threads_from_value` (1..=128 に丸め、未設定・解析不能・0 は既定)。
- `qwen4exp::forward_prefill_batch_resident_lm`: `const PLE_PIPE_THREADS: usize = 32`
  を `exp_pf::ple_threads()` に。専用プールの本数がそのまま同時 pread 数になる。
- `ple::ple_par_parts`: pipe 分岐の上限 32 固定をやめ、`ple_threads()` と
  `current_num_threads()` の小さいほう (1..=128) に。
- 値は変えない: 読む行 (n-gram 索引)、昇順 + 重複排除、`gather_rows` の
  「連続 index をまとめる」規則、`dequant_row_into` はそのまま。並列度だけ。

| lab (cold、2 ラウンドの中央値) | 3,826 | 14k | ple 3.8k / 14k |
| --- | ---: | ---: | ---: |
| t357a baseline (x0/b1/b2 の中央値) | 8.41 | 8.09 | 330 / 677 ms |
| t357a p64 | 8.31 (-1.2%) | 7.90 (-2.4%) | 241 / 488 |
| t357a p96 | **8.24 (-2.1%)** | **7.82 (-3.4%)** | 216 / 425 |
| t357b baseline (x0/b1/b2) | 8.40 | 8.08 | — / 680 |
| t357b p96 | **8.30 (-1.2%)** | **7.80 (-3.5%)** | — / 425 |
| t357b p128 | 8.26 | 11.21 (外乱) | — / 386 |
| t357c off1 (32) | 8.50 | 8.06 | 330 / 676 |
| t357c on1 (既定 96) | **8.37 (-1.5% 対 off1)** | **7.82 (-3.0% 対 off1)** | 212 / 419 |
| t357c b2 (32、末尾 repeat) | 14.58 (外乱) | 8.05 | 329 / 681 |
| t357c warm 32 / 96 | 2.51 / 2.51 | 6.43 / 6.44 | 13.5/14.1 / 53.9/52.8 |

- 3 ラウンドとも **14k が -3.0% 以上**、3,826 も -1.2 〜 -2.1% で悪化なし。
- 外乱 2 件 (t357a b1 の 14k 15.79 s / ple 4,029 ms、t357c の最初と最後の
  COND が 3,826 10.57 / 14.58 s) は decode も同時に悪い (44.6 / 37.4 / 28.4
  ms/tok) のでホストの外乱。**中央値から外した**。t357c の on1 (8.37 / 7.82) は
  外乱の次の COND で、off1 と同じ条件の隣なので対で読める。
- 1080 H2D / 504 D2H / 1764 launch の ssm 段は本変更の前後で同じ値 (t357a x0 と
  p96 の `[q4ledger-pf]` で確認)。層ループの中は 1 バイトも変わっていない。

### 2.2 128 本を既定にしなかった理由

gather は 425 → 386 ms まで下がる (t357b p128)。しかし同じ COND の 14k が
11.21 s、decode 33.9 ms/tok で、ホストの外乱と区別できない 1 回だった。96 本は
3 ラウンド 4 回とも 7.80-7.82 s で安定しているので、**再現性を取って 96**。

## 3. 数値の扱い

- **バイト列は不変**。md5 は t357a (5 COND)、t357b (5 COND)、t357c (6 COND) の
  全リクエストで完全一致: `a:65ededc3` / `b:f18c75fc` / `c2000_cold:a91e17da` /
  `c2000_turn2:5a8f5488` / `c8000_cold:e2292eb6` / `c8000_turn2:561c6da8`。
- decode: 短 16.6 ms/tok (全 COND)、3,826 後 21.0、14k 後 18.0 (warm は 16.1-16.2)。
  0.1 ms/tok 刻みなので差は無い。層ループに触っていないので構造的にも不変。
- 冷えた状態の作り方は task 353 と同じ (`DENNO_START_DROP_CACHE=1` を全 COND)。
  これが無いと depth8k は COND の位置で 6.4 s まで暖まり、cold の A/B が取れない。
  暖かい側は t357c の w32/w96 対で「変わらない」ことを確認した。
- マイクロベンチは無し。lab の数値だけを実測として書いている。

## 4. 採否

- `ADOPT: yes` (lab t357a / t357b / t357c)。`DEFAULT_PLE_THREADS = 96` にした。
  切り戻しは `DENNO_PF_PLE_THREADS=32` (task 353 の値)。
- 目標 (どちらかが 2.5% 以上、他は悪化なし) は **14k cold -3.5% /
  3,826 cold -1.5%** で満たした。decode・md5・errors・text も条件内。
- 注意: この変更は **page cache が cold のときだけ**効く (暖かいと ple が
  54 ms なので差が出ない)。本番の「起動後の 1 本目」(8.5-13.8 s の側) が
  対象で、2 本目以降 (6.4 s) は変わらない。

VERDICT (t357a、全行):

```
VERDICT b1 vs x0 OK decode +0.2% prefill -1.7% md5 SAME errors 0
VERDICT p64 vs x0 OK decode +0.0% prefill +0.0% md5 SAME errors 0
VERDICT p96 vs x0 OK decode +0.2% prefill -0.8% md5 SAME errors 0
VERDICT b2 vs x0 OK decode +0.0% prefill -0.8% md5 SAME errors 0
VERDICT_DEPTH b1 c2000_cold vs x0 decode +0.3% prefill -0.2%
VERDICT_DEPTH b1 c2000_turn2 vs x0 decode -0.4% prefill +0.0%
VERDICT_DEPTH b1 c8000_cold vs x0 decode +133.0% prefill +94.9%
VERDICT_DEPTH b1 c8000_turn2 vs x0 decode +148.2% prefill +17.4%
VERDICT_DEPTH p64 c2000_cold vs x0 decode +0.0% prefill -1.3%
VERDICT_DEPTH p64 c2000_turn2 vs x0 decode -0.2% prefill -1.2%
VERDICT_DEPTH p64 c8000_cold vs x0 decode -0.6% prefill -2.5%
VERDICT_DEPTH p64 c8000_turn2 vs x0 decode +0.0% prefill +0.0%
VERDICT_DEPTH p96 c2000_cold vs x0 decode +0.0% prefill -2.2%
VERDICT_DEPTH p96 c8000_cold vs x0 decode -0.5% prefill -3.4%
VERDICT baseline x0 errors 0 text ok
```

VERDICT (t357b、128 本の確認、抜粋):

```
VERDICT_DEPTH p96 c2000_cold vs x0 decode +0.3% prefill -1.2%
VERDICT_DEPTH p96 c8000_cold vs x0 decode -4.0% prefill -4.1%
VERDICT_DEPTH p128 c2000_cold vs x0 decode +0.0% prefill -1.7%
VERDICT_DEPTH p128 c8000_cold vs x0 decode +80.3% prefill +37.9%
VERDICT baseline x0 errors 0 text ok
```

VERDICT (t357c、既定 on を逆側から確認 + warm 対、抜粋):

```
VERDICT off1 vs x0 OK decode +0.0% prefill +0.0% md5 SAME errors 0
VERDICT on1 vs x0 OK decode -0.2% prefill +0.0% md5 SAME errors 0
VERDICT w32 vs x0 OK decode +0.0% prefill +0.0% md5 SAME errors 0
VERDICT w96 vs x0 OK decode -0.2% prefill +0.0% md5 SAME errors 0
VERDICT b2 vs x0 OK decode +0.0% prefill +0.9% md5 SAME errors 0
VERDICT_DEPTH off1 c8000_cold vs x0 decode -7.3% prefill +3.2%
VERDICT_DEPTH on1 c8000_cold vs x0 decode -7.8% prefill +0.1%
VERDICT_DEPTH w32 c8000_cold vs x0 decode -17.3% prefill -17.7%
VERDICT_DEPTH w96 c8000_cold vs x0 decode -16.8% prefill -17.5%
VERDICT baseline x0 errors 0 text ok
```

(t357c の x0 は先頭の捨て COND で 3,826 が 10.57 s / decode 37.4 の外乱。
対比は `off1` (8.50 / 8.06) と `on1` (8.37 / 7.82) で取る。)

テキスト (t357a の p96、TXT と TXTD):

```
TXT  A **B-tree** (specifically, the **B+ tree) is the fundamental data structure
     used by most modern relational databases (such as MySQL, PostgreSQL, Oracle,
     and SQL Server).\n\nBelow is a detailed explanation of how it works and why
     it is the standard for
TXTD ご提示いただいた文書は、LLM推論エンジン（c2000プロジェクト）における、
     **Prefill（事前計算）フェーズの性能改善プロセスを詳細に記録したものです。
     以下に、文書の構造と技術的要点を、より詳細に解説します
```

## 5. 次の一手 (未実測・未実装)

- gather は 96 本で 425 ms 残る。**層ループの前に直列で露出している**のが
  構造的な無駄で、PLE 注入は層 1 なので層 0 (14k で約 150 ms) の裏には隠せる。
  生産スレッド + 層 1 で消費する形にすると 425 → 150-250 ms (推定) の余地。
  本タスクでは測っていない (round 3 を使い切った)。
- 3,826 の experts (22.65 GB / 6.4 s) は disk の床なので、bytes を減らす以外に
  無い。VRAM 常駐は初回リクエストでは空なので効かない。

## 6. テスト

```
CARGO_TARGET_DIR=/tmp/t357 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1238 passed / 0 failed / 103 ignored
CARGO_TARGET_DIR=/tmp/t357 cargo test --release -p denno-engine --features cuda --test kernel_invariants -- --test-threads=1
  -> 107 passed / 0 failed
```

- 追加: `model::exp_pf::tests::ple_threads_rounds_and_defaults` (丸めと既定 96)、
  `kernel_invariants::t357_ple_threads_gate_is_wired_to_the_pool` (env → プール
  の配線、32 本固定が消えたこと)。
- 更新: `kernel_invariants::t349_ple_pipe_widens_the_lm_gather_pool` (文字列検査を
  新しい配線に)。
- int-auto は最後にもう一度 fetch したが `1ce57fb2c` のままで merge は不要。
