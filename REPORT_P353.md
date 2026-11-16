ADOPT: yes t353a t353b

# task 353 — IQ3_S prefill、PLE の gather を冷えた page cache で広げる

branch `br353` は開始時の `int-auto` `17e938604` から。**`br349` は int-auto に
merge されていなかったので merge した** (`e436c437f`)。task 349 の 3 本のゲート
(ple pipe / stream pread / bg warm) はどれも既定 off のまま取り込み、そのうち
**ple pipe だけを本タスクで測って既定 on にした**。数値はすべて GPU ラボの実測
(`SCENARIO=depth8k`, `PROFILE=1`, `swift15-abl-IQ3_S`)。lab 名は t353a / t353b。

## 0. 結論 (5 行)

- **14k cold が 9.29 → 8.06-8.08 s (-13.2% / -13.8%)、3,826 cold が
  8.85 → 8.41-8.42 s (-5.0% / -5.2%)。2 ラウンドの中央値で 2 段とも再現。**
  目標 (2.5%) を両方超えた。decode は 16.6 → 16.6-16.8 ms/tok (0.1 ms 刻みの
  範囲内)、md5 は全 COND・全ラウンド SAME、errors 0、text sane。
- 効いたのは **層優先 prefill の先頭 PLE gather の並列度**。`ple_par_parts` の
  上限を 16 → 32 にし、専用の 32 本プールで引く (`DENNO_PF_PLE_PIPE`、br349 の
  `7431cea95` をそのまま使った)。1 行 90 B のランダム pread が IOPS 律速なので、
  queue を深くするだけで wall が落ちる。
- **`[pre-kprof]` の ple がそのまま減った**: 3,826 771.3 → 328.3 ms、
  14k 1,851.3 → 681.3 ms。層の時間 (8,000 / 7,059 ms) は動いていないので、
  差は全部 gather。
- **大きいチャンク (`DENNO_PF_CHAIN_BIG=1` + `DENNO_PF_POOL_BIG_CHUNK=4096`) は
  効かない** (14k でチャンク 2,047 → 4,096 が実際に発火して 9.21 s、baseline 9.21-9.29。
  ただし転送量は 28.36 → 28.80 GB と**増えた**)。turn2 の md5 も DIFF。
  既定 off のまま (この組合せを既定にする理由が無い)。
- **`DENNO_PF_STREAM_PREAD` は task 349 と同じ結果** (3,826 -2.3%、14k +1.0%)。
  2.5% に届かず、14k は 1% 悪化なので既定 off のまま。

## 1. Do 1: 基準の取り直しと、冷えた状態の作り方

### 1.1 基準 (int-auto `17e938604` + ラボ tint、`DENNO_START_DROP_CACHE=1`)

タスクの前提は「3,826 cold 7.3 s / 14k cold 25.0 s」だったが、これは
**page cache が完全に冷えた最初の 1 本**の数字で、ラボの depth8k は同じ server で
何度も prefill するため、そのままでは COND の位置で 2.5 s / 6.4 s まで暖まる
(tchkef21 の実測: x0 6.49 / 13.82 → base2 2.52 / 6.44)。

そこで task 349 と同じ方法で、**全 COND に `DENNO_START_DROP_CACHE=1` を置いて
毎回の server 起動で clean page を落とした**。すると COND 間のばらつきが消えて
交互の A/B が取れる:

| t353a の COND | 3,826 cold (2 reps の中央値) | 14k cold | decode |
| --- | --- | --- | --- |
| x0 (捨て) | 8.82 | 9.23 | 16.6 |
| b1 | 8.85 | 9.29 | 16.6 |
| b2 | 8.82 | 9.24 | 16.6 |

**冷えた 3,826 は 8.82-8.85 s、14k は 9.21-9.29 s で安定** (t349 の
8.79-8.86 / 9.21-9.26 と同じ)。タスクの 7.3 / 25.0 s とは桁が違うが、これは
「毎回 cold に落とす」と「1 回だけ cold」の違いで、A/B の基準としてはこちらが正しい。

### 1.2 段ごと (t353a b1 の 14k、`[q4ledger-pf]`、ms)

| 段 | wall | exposed | 備考 |
| --- | ---: | ---: | --- |
| mix | 20 | 0 | |
| full | 1,152 | 361 | |
| ssm | 2,530 | 954 | |
| experts | 2,287 | 791 | `[pf-host] up_wait=1,784` |
| other | 3,315 | 3,041 | 中身は PLE (`[pre-kprof] ple=1,851`) |
| close | 9,287 | — | gpu_busy 4,894 |

**cold の 14k は `other` (PLE の gather) が最大**で、`[pre-kprof]` の
`ple rows=166750 reads=166728 bytes=15007500 read=13133.0 ms` がその正体
(1 行 90 B を 166,728 回、8 本並列で 1.85 s の壁)。3,826 でも ple=771 ms で、
`experts` (2,287 ms) に次ぐ 2 番目だった。

## 2. Do 2: 変えたこと

### 2.1 `DENNO_PF_PLE_PIPE` — PLE gather を 32 本で引く (採用、既定 on)

br349 の `7431cea95` (未測定だった) をそのまま使い、既定を on にした。
層優先の経路 (`forward_prefill_batch_resident_lm`) は全チャンクの gather を
層ループの前に直列で引くので、パイプ (生産スレッド) ではなく **専用プールの
本数**が効く。`ple_par_parts` がゲート on のとき `current_num_threads()` を
上限 32 で見る (従来の `DENNO_PLE_PAR` は 16)。

| lab t353a | 3,826 cold | 14k cold | ple (3.8k / 14k) | decode |
| --- | --- | --- | --- | --- |
| b1 / b2 (baseline) | 8.85 / 8.82 | 9.29 / 9.24 | 771 / 1,851 ms | 16.6 |
| pl (`DENNO_PF_PLE_PIPE=1`) | **8.41 (-5.0%)** | **8.06 (-13.2%)** | 328 / 681 ms | 16.7 |

t353b は既定を on にした commit (`aea7854f1`) で、**逆側から**確認した
(on = 既定、off = `DENNO_PF_PLE_PIPE=0`):

| lab t353b | 3,826 cold | 14k cold | decode |
| --- | --- | --- | --- |
| x0 / on1 / on2 (既定 on) | 8.41 / 8.42 / 8.41 | 8.11 / 8.07 / 8.08 | 16.6 / 16.8 / 16.6 |
| off1 / off2 (`=0`) | 8.87 / 8.88 | 9.45 / 9.29 | 16.6 / 16.6 |

2 ラウンドの中央値: **3,826 が 8.85 → 8.41 / 8.42 (-5.0% / -5.2%)、
14k が 9.29 / 9.37 → 8.06 / 8.08 (-13.2% / -13.8%)**。

- 読む行・重複排除・綴じ方は同じなので **md5 は 8 本すべて SAME**
  (`a91e17da` / `5a8f5488` / `e2292eb6` / `561c6da8`)。t353b も全 COND SAME。
- レップ間のばらつきは 0.01-0.02 s。baseline の 2 レップも 8.85 / 8.82、
  9.29 / 9.21 で、**差はレップ間のばらつきの 20 倍以上**。
- decode は 16.6 (baseline) に対し on が 16.6-16.8。0.1 ms/tok 刻みなので
  on1 の +1.2% は刻み 1 つの差で、on2 は +0.0%、t353a の pl は +0.2%。
  `c2000_cold` の decode は全 COND 21.0 ms/tok で同一、`c8000_cold` も
  18.0/18.4 (on) 対 18.3/17.9 (off) で有意差なし。
- `[pre-kprof]` の read 合計は 3,826 で 6,098 → 9,991 ms、14k で 13,133 →
  18,392 ms と**増える** (同時に飛ぶ本数が増えるので和は増える)。壁は
  771 → 328 / 1,851 → 681 ms。IOPS 律速の裏取り。
- 暖かい側の害が無いこと: 同じ run の `c2000_turn2` / `c8000_turn2` (PLE の表は
  1 本目の gather で暖まっている) は全 COND で 0.42 / 0.46 s、decode も
  17.8-17.9 / 16.5-16.7 で on/off の差が無い。プール生成はリクエストごとに
  1-2 ms (推定) だが、長い prefill では誤差。

### 2.2 `DENNO_PF_STREAM_PREAD` — 不採用 (既定 off のまま)

| lab t353a | 3,826 cold | 14k cold |
| --- | --- | --- |
| baseline | 8.82-8.85 | 9.21-9.24 |
| pr (`DENNO_PF_STREAM_PREAD=1`) | 8.62 (-2.3%) | 9.32 (+1.0%) |

task 349 (既定 env) の -2.5% / +1.3% と同じ傾向。2.5% に届かず、14k が 1% 悪い。
`[exp-pf-stream]` に `spread=1` が出て pread は発火している (読むバイトは同じ)。

### 2.3 大きいチャンク — 不採用 (ゲートを足していない)

task 270 の `DENNO_PF_CHAIN_BIG=1` と `DENNO_PF_POOL_BIG_CHUNK=4096` を
組み合わせると、2048 の壁 (GDN の連鎖) を外したまま 14k のチャンクを
2,047 → 4,096 にできる。t279a は UD-Q4_K_XL でしか測っておらず、**IQ3_S の
ラボ tint (expert を VRAM に置く) では転送量が壁の 42%** なので、チャンク数を
減らす価値があるかを測った:

| lab t353a | 3,826 cold | 14k cold | `up_MB` (3.8k / 14k) | md5 |
| --- | --- | --- | --- | --- |
| baseline | 8.82-8.85 | 9.21-9.24 | 22,653 / 28,363 | SAME |
| ck | 8.82 (チャンクは 1,913 のまま) | 9.21 (4,096 に発火) | 22,653 / **28,796** | 14k turn2 が DIFF |

**チャンクを半分にしても転送量は減らなかった** (むしろ +1.5%)。expert の
アップロードは「チャンク数 × 128 本」ではなく、LRU のヒット率で決まる
(expert 本数 13,840 → 14,085)。よって既定にしない。md5 が変わるのも損。

## 3. なぜ効いたか / 残り

- cold の 14k は ple 1.85 s + experts 0.79 s 露出で、**ple が最大の単一項目**。
  gather は 1 行 90 B のランダム pread 166,728 回で、8 本では queue が浅い。
  32 本にすると同じ行数を深い queue で投げられる (ディスク側の IOPS が上限)。
- 32 本で 1.85 → 0.68 s。**まだ 0.68 s 残る**ので、48/64 本や
  `io_uring` の余地はある (未測定)。ただし本タスクの目標は達成した。
- 3,826 の残りは experts (転送 22.6 GB / 1.05-3.4 s)。チャンクを増やしても
  減らないことを §2.3 で確認したので、次は「層ごとに 1 回だけ上げる」構造側。

## 4. 採否

- `ADOPT: yes` (lab t353a / t353b)。`DEFAULT_PLE_PIPE = true` にした。
  切り戻しは `DENNO_PF_PLE_PIPE=0` (既定 on、`0` 系だけが偽)。
- 数値経路は変えていない (読む行・綴じ方・丸め順は同一)。md5 SAME がその証拠。
- `DENNO_PF_STREAM_PREAD` / `DENNO_PF_BG_WARM` は既定 off のまま (前者は本タスクで
  再確認、後者は task 349 の +45.8% のため触っていない)。
- lab は 2 ラウンド使った (上限 3)。3 本目は使っていない。

VERDICT (t353a、全行):

```
VERDICT b1 vs x0 OK decode +0.0% prefill +0.9% md5 SAME errors 0
VERDICT ck vs x0 OK decode +0.0% prefill -0.9% md5 DIFF errors 0
VERDICT pl vs x0 OK decode +0.2% prefill +0.0% md5 SAME errors 0
VERDICT pr vs x0 OK decode +0.0% prefill +0.9% md5 SAME errors 0
VERDICT b2 vs x0 OK decode -0.5% prefill +0.9% md5 SAME errors 0
VERDICT_DEPTH b1 c2000_cold vs x0 decode +0.3% prefill +0.3%
VERDICT_DEPTH b1 c2000_turn2 vs x0 decode +0.0% prefill +0.0%
VERDICT_DEPTH b1 c8000_cold vs x0 decode +1.2% prefill +0.7%
VERDICT_DEPTH b1 c8000_turn2 vs x0 decode +1.0% prefill +0.0%
VERDICT_DEPTH ck c2000_cold vs x0 decode +0.3% prefill +0.0%
VERDICT_DEPTH ck c2000_turn2 vs x0 decode +0.0% prefill +0.0%
VERDICT_DEPTH ck c8000_cold vs x0 decode +2.4% prefill -0.2%
VERDICT_DEPTH ck c8000_turn2 vs x0 decode +1.9% prefill +0.0%
VERDICT_DEPTH pl c2000_cold vs x0 decode +0.3% prefill -4.6%
VERDICT_DEPTH pl c2000_turn2 vs x0 decode -0.2% prefill +0.0%
VERDICT_DEPTH pl c8000_cold vs x0 decode +0.6% prefill -12.7%
VERDICT_DEPTH pl c8000_turn2 vs x0 decode +0.5% prefill +0.0%
VERDICT_DEPTH pr c2000_cold vs x0 decode +0.6% prefill -2.3%
VERDICT_DEPTH pr c2000_turn2 vs x0 decode +0.2% prefill +0.0%
VERDICT_DEPTH pr c8000_cold vs x0 decode +0.6% prefill +1.0%
VERDICT_DEPTH pr c8000_turn2 vs x0 decode +1.0% prefill +0.0%
VERDICT_DEPTH b2 c2000_cold vs x0 decode +0.3% prefill +0.3%
VERDICT_DEPTH b2 c2000_turn2 vs x0 decode -0.4% prefill +0.0%
VERDICT_DEPTH b2 c8000_cold vs x0 decode +0.6% prefill +0.1%
VERDICT_DEPTH b2 c8000_turn2 vs x0 decode +1.0% prefill +0.0%
VERDICT baseline x0 errors 0 text ok
```

VERDICT (t353b、既定 on を逆側から確認、全行):

```
VERDICT off1 vs x0 OK decode +0.2% prefill +0.9% md5 SAME errors 0
VERDICT on1 vs x0 OK decode +1.2% prefill +0.9% md5 SAME errors 0
VERDICT off2 vs x0 OK decode -0.2% prefill +0.9% md5 SAME errors 0
VERDICT on2 vs x0 OK decode +0.0% prefill +0.0% md5 SAME errors 0
VERDICT_DEPTH off1 c2000_cold vs x0 decode +0.3% prefill +5.5%
VERDICT_DEPTH off1 c2000_turn2 vs x0 decode +0.4% prefill +0.0%
VERDICT_DEPTH off1 c8000_cold vs x0 decode -6.7% prefill +16.5%
VERDICT_DEPTH off1 c8000_turn2 vs x0 decode -7.8% prefill +0.0%
VERDICT_DEPTH on1 c2000_cold vs x0 decode +0.0% prefill +0.2%
VERDICT_DEPTH on1 c2000_turn2 vs x0 decode +0.4% prefill -1.2%
VERDICT_DEPTH on1 c8000_cold vs x0 decode -7.8% prefill -0.5%
VERDICT_DEPTH on1 c8000_turn2 vs x0 decode -8.7% prefill +0.0%
VERDICT_DEPTH off2 c2000_cold vs x0 decode +0.6% prefill +5.7%
VERDICT_DEPTH off2 c2000_turn2 vs x0 decode +0.2% prefill +0.0%
VERDICT_DEPTH off2 c8000_cold vs x0 decode -8.3% prefill +14.5%
VERDICT_DEPTH off2 c8000_turn2 vs x0 decode -8.7% prefill +0.0%
VERDICT_DEPTH on2 c2000_cold vs x0 decode +0.6% prefill +0.1%
VERDICT_DEPTH on2 c2000_turn2 vs x0 decode +0.4% prefill +0.0%
VERDICT_DEPTH on2 c8000_cold vs x0 decode -6.1% prefill -0.4%
VERDICT_DEPTH on2 c8000_turn2 vs x0 decode -7.4% prefill +0.0%
VERDICT baseline x0 errors 0 text ok
```

テキスト (t353a の pl、2 レップ同一):

```
A **B-tree** (specifically, the **B+ tree) is the fundamental data structure used by most
modern relational databases (such as MySQL, PostgreSQL, Oracle, and SQL Server).
Rust の「所有権（Ownership）とライフタイム（Lifetimes）は、Rust の最も重要な概念であり、
同時に初心者が最もつまずきやすい部分です。
ご提示いただいた文書は、LLM推論エンジン（c2000プロジェクト）における、
**Prefill（事前計算）フェーズの性能改善プロセスを詳細に記録したものです。
```

## 5. GPU で確認してほしいこと / 次の一手

- 既定 on の確認は t353b で済み (on 8.41 / 8.08、off 8.87-8.88 / 9.29-9.45)。
  本番の 1 本目 (page cache が冷えた状態) で 14k が 9.2 → 8.1 s になるはず。
- 48/64 本 (`ple_par_parts` の上限) を試す価値はある。14k でまだ 0.68 s 残って
  いる。ただしプール生成はリクエストごとなので、短い prefill では発火しない
  (`ctx.len() > chunk` が条件)。
- 3,826 の experts (転送 22.6 GB) はチャンクでは削れない (§2.3。チャンクを
  14k で半分にしても転送量は増えた)。層ごとに 1 回だけ上げる構造
  (ステージングの使い回し) が次の候補 (未実装・未測定)。
- 注意: `DENNO_START_DROP_CACHE=1` は診断用で、全 COND を同じ冷え方に揃える。
  これが無いと depth8k は COND の位置で 2.5 / 6.4 s まで暖まり、cold の A/B が
  取れない (tchkef21 の 6.49 → 2.52 s)。

## 6. テスト

```
CARGO_TARGET_DIR=/tmp/t353 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1236 passed / 0 failed / 102 ignored
CARGO_TARGET_DIR=/tmp/t353 cargo test --release -p denno-engine --features cuda --test kernel_invariants -- --test-threads=1
  -> 106 passed / 0 failed
```

- int-auto は最後にもう一度 fetch したが `17e938604` のままで、merge は不要だった。
- 更新したテスト: `model::exp_pf::tests::ple_pipe_gate_truth_table` (既定 on、
  `0` / `off` / 解析不能だけ偽)、`kernel_invariants` の
  `t349_ple_pipe_is_wired_and_does_not_double_read` (既定の文字列)。
- 発火の証拠 (lab): t353a の pl で `[pre-kprof]` の ple が 771 → 328 ms /
  1,851 → 681 ms。t353b では既定 on のまま on の 3 COND が同じ値
  (8.41-8.42 / 8.07-8.11) を出し、`DENNO_PF_PLE_PIPE=0` の 2 COND だけが
  8.87-8.88 / 9.29-9.45 に戻った。ゲートが効いている証拠。
