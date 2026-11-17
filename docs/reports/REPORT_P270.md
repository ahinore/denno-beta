ADOPT: no t270b

# task 270 — 14k cold prefill の台帳と、次に大きいピース (expert の上げ)

ブランチ `br270` は開始時の `int-auto` (62a9ad16b) から。測定はすべて GPU ラボ
(SCENARIO=depth8k、PROFILE=1、MODEL=swift15-abl-IQ3_S)、3 ラウンド。

## 0. 前提 (読む順に)

- 測定した木 = `62a9ad16b` (開始時の int-auto) + このタスクの 2 コミット。
  候補はすべて env で切り、**既定は off** なので base COND は int-auto と同一。
- 測定後に `int-auto` が動いた (`89b7f8a44`: **task 265 のプールが演算子判断で
  既定 on**)。最後に fetch して merge 済み (衝突は
  `forward_prefill_inner` の chunk 呼び出し 1 箇所だけで、両側を残した)。
  したがって下の数値は**プール導入前**の int-auto のもの。プール入りの現在の
  int-auto では 14k の prefill は task 265 の実測 (t265a/c) で既に -34% であり、
  このタスクの候補 (既定 off) はその上に積んでいない。
- 基準 COND は タスク指定の 4 本:
  `DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0
  DENNO_GPU_WEIGHT_MB=12000`。各ラウンドの先頭に捨て COND (`toss`)、末尾に
  base の繰り返し (`base2`) を置いた。

## 1. 台帳 (先に表) — lab t270b の base2、3,826 / 14,018 cold

PROFILE=1 の `[q4ledger-pf]`。ms は合計、`busy` は起動ペアの GPU 実時間、
`exposed` は「壁時計 − GPU 実時間」= ホストが GPU を待たせている/idle の近似。

### 3,826 トークン cold (壁時計 7,310 ms、h2d 63.2 GB、d2h 9.05 GB)

| 段 | wall_ms | 割合 | gpu_ms | cpu_ms | h2d_MB | d2h_MB | busy_ms | exposed_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mix | 309.1 | 4.2% | 216.2 | 1.9 | 11.8 | 3,761.1 | 228.6 | 80.5 |
| router | 37.2 | 0.5% | 26.5 | 1.5 | 0.0 | 15.4 | 30.5 | 6.8 |
| shared | 195.3 | 2.7% | 27.7 | 124.4 | 0.0 | 1,880.6 | 32.9 | 162.4 |
| full | 223.7 | 3.1% | 130.4 | 42.9 | 482.0 | 658.2 | 133.2 | 90.5 |
| ssm | 645.6 | 8.8% | 542.6 | 31.0 | 1,581.3 | 353.0 | 552.4 | 93.2 |
| **experts** | **5,435.4** | **74.4%** | 395.1 | 2,962.3 | 58,270.1 | 1,880.6 | 4,124.0 | 1,311.4 |
| other | 463.6 | 6.3% | 72.9 | 463.6 | 2,873.6 | 502.2 | 76.9 | 386.7 |

- 内訳: `ssm` = scan_pf 358.8 / qkv_pf 150.3 / out_pf 46.9、
  `other` = PLE 146.5 / combine 154.0 / embed 7.7。
- `experts` の中身 (`[pf-host]`): stage_copy 2,199.1 / **ptrtab 1,884.3** /
  up_wait 357.2 / sync 399.0 / scratch 140.4 / d2h 128.8。
  上げた重みは 38.9 GB、`layers=96` (= 48 層 × 2 チャンク)、copy 2.42 s。

### 14,018 トークン cold (壁時計 25,103 ms、h2d 180.7 GB、d2h 33.3 GB)

| 段 | wall_ms | 割合 | gpu_ms | cpu_ms | h2d_MB | d2h_MB | busy_ms | exposed_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mix | 1,100.1 | 4.4% | 794.7 | 3.0 | 39.3 | 13,780.3 | 807.8 | 292.2 |
| router | 132.7 | 0.5% | 110.2 | 4.0 | 0.0 | 56.5 | 111.0 | 21.7 |
| shared | 681.4 | 2.7% | 109.6 | 428.9 | 0.0 | 6,890.1 | 114.8 | 566.6 |
| full | 1,214.8 | 4.8% | 850.3 | 188.5 | 1,765.8 | 2,411.5 | 853.6 | 361.2 |
| ssm | 2,292.8 | 9.1% | 2,012.7 | 81.0 | 5,344.6 | 1,176.7 | 2,015.3 | 277.5 |
| **experts** | **18,320.8** | **73.0%** | 1,338.9 | 7,852.4 | 163,474.8 | 7,541.9 | 15,162.0 | 3,158.8 |
| other | 1,360.0 | 5.4% | 292.3 | 1,360.0 | 10,048.2 | 1,418.5 | 279.2 | 1,080.8 |

- 内訳: `ssm` = scan_pf 1,314.9 / qkv_pf 540.4 / out_pf 169.5、
  `other` = combine 561.6 / PLE 357.4 / embed 29.5。
- `experts` の中身 (`[pf-host]`): stage_copy 6,064.1 / **ptrtab 8,477.3** /
  up_wait 531.0 / sync 1,339.5 / d2h 400.4 / scratch 487.5 / その他 926.8。
- 上げた重み **156.5 GB** (`[exp-pf-stream] up_MB=156470.8`, 76,249 本、
  `layers=432` = 48 層 × 9 チャンク、`rows_per_expert=87.0`、copy 6.06 s、
  probe 53.0 GB/s)。
- 突き合わせ: `close wall=25102.7 gpu_busy=19343.7 exposed_host=5758.9
  delta=0.0`。**GPU が 19.3 s 動き、5.8 s はホスト側で止まっている**。

**読み方**: 台帳の最大項目は expert の重み上げ (156 GB)。中身は
「mmap (ページキャッシュ) → pinned のバウンスに写す」(6.06 s、DRAM 律速:
156 GB × 読み書き 2 = 312 GB / 6.06 s = 51 GB/s) と、`ptrtab` の 8.5 s
(表の同期 H2D が null ストリームの前の仕事を待つ。ホストが 1 グループ先まで
しか進めない) で、この 2 つが **直列**になっている。

## 2. 変えたこと (2 コミット、どれも既定 off)

1. `27cff00ad` **`DENNO_PF_CHAIN_BIG`** — GDN prefill の連鎖を `nt >= 2048` でも。
   task 197 でチャンク 2048 が +39.6% だった理由は容量ではなく
   `gdn_pf_chain_shape_ok` の `nt < GDN_CHUNK_MIN_TOKENS (2048)` で連鎖が外れ、
   ssm が 3.96 → 24.7 s に落ちたこと。当時は単発 sigmoid スキャンしか無かったが、
   task 186/225 の v2/v3 はトークン長の上限を持たない (既定 v3)。規則を純関数
   `chain_nt_ok(.., big)` に切り出し、`gdn_pf_chain` は**投影を撃つ前**にスキャン
   カーネルの有無を確かめる (無ければ状態を触らずに降りる)。
2. `3092247f1` **`DENNO_PREFILL_WORK_MB` の配線** — task 256 が用意していた
   `expert_vram::reserve_work_area` は呼び出し側が無く、ノブが死んでいた。
   prefill の入口で予約する (既定 0 = 何もしない)。

## 3. ラボ 3 ラウンド

### t270a (27cff00ad): チャンクを上げるだけ

| COND | 3,826 cold | 14,018 cold | 14k 後 decode |
| --- | --- | --- | --- |
| toss/base2 (base) | 7.26 / 7.27 | 25.73 / 25.10 | 20.6 / 19.3 |
| c3k (chunk 3072 + CHAIN_BIG) | 7.38 | **39.44** | 20.7 |
| c4k (chunk 4096 + CHAIN_BIG) | 7.27 | **40.75** | 21.3 |

`VERDICT_DEPTH c3k c8000_cold vs toss decode +0.1% prefill +53.3%` /
`c4k c8000_cold decode +3.0% prefill +58.4%`。**原因**: 14k で
`pf_stage_alloc` が空き VRAM 不足で staging を取れず、streaming が**黙って
落ちた** (`[exp-pf-stream]` 行が出ない。experts 18.2 → 32.2 s、d2h 52.5 GB =
非ストリーミング経路の h の D2H)。ssm は 2,215 ms で連鎖は生きており、
2048 の壁自体は外れている (scan_pf 1,315 ms は 1536 のときと同一)。

### t270b (3092247f1): + 表の非同期 + 作業域の予約

| COND | 3,826 cold | 14,018 cold | 14k 後 decode | 上げた量 | copy_s | 予約で空けた expert |
| --- | --- | --- | --- | --- | --- | --- |
| base2 | 7.28 / 7.29 | 25.06 / 25.10 | 18.1 | 156.5 GB | 6.06 | — |
| c3k (3072, reserve 2600) | 7.12 / 7.12 | 24.31 / 24.35 | 18.4 / 19.6 | **103.0 GB** | 8.99 | 1.3〜2.5 GB |
| c4k (4096, reserve 3400) | 7.00 / 7.00 | 24.90 / 24.97 | 18.9 / 19.5 | 89.4 GB | 9.19 | 2.2〜2.6 GB |

`VERDICT_DEPTH c3k c2000_cold vs toss decode -0.4% prefill -2.5%` /
`c3k c8000_cold decode +8.5% prefill -3.0%` /
`c4k c8000_cold decode +4.8% prefill -0.6%` / errors 0 / text sane。
**狙い (14k -15%、3.8k -10%) に届かず、decode は +2% を超えた。**

- 上げた量は狙いどおり -34% (156.5 → 103.0 GB) になった。しかし
  **copy が 6.06 → 8.99 s に増えた**。作業域が VRAM を食って staging の
  1 枚が 170 → 62 MB に縮み、スロット 3 枚でホストが先へ進めず、
  「前のグループのカーネル完了待ち」が `stage_copy` に乗ったため
  (slotwait は ut 経路専用なので 0 のまま)。
- decode の後退は予約で expert を 1.3〜2.5 GB 追い出したぶん。許容 +2% は
  **1 GB 弱まで**で、チャンクを上げるのに要る追加は 1.2 GB 以上。ここが壁。
- `DENNO_PF_TAB_ASYNC=1` は 1,927/1,927 回すべて非同期で通った (ptrtab
  2,356 → 18.7 ms) が、待ちは `sync` (399 → 5,151 ms) へ移っただけで
  総額は変わらない。

### t270c (3092247f1): 表の非同期だけ (チャンク据え置き)

| COND | 3,826 cold | 14,018 cold | 14k 後 decode | md5 |
| --- | --- | --- | --- | --- |
| base2 | 7.28 / 7.31 | 25.12 / 25.23 | 18.1 | SAME |
| tab (TAB_ASYNC=1) | 7.24 / 7.22 | 24.97 / 24.96 | 18.1 | SAME |
| tab1g (+ STREAM_MB=1024) | 7.22 / 7.22 | 25.07 / 25.07 | 18.2 | SAME |

`VERDICT_DEPTH tab c8000_cold vs toss decode +0.0% prefill -1.0%` /
`tab1g c8000_cold decode +0.5% prefill -0.6%`。**-1%。**
台帳で `ptrtab` が 8.5 s に見えるのは「待っている相手 (前のグループの
カーネル)」がそこに計上されているだけで、非同期にすると同じ待ちが
`sync` に移る。壁はホストと GPU の直列化そのもの。

## 4. なぜ Target に届かないか (機序)

1. 156 GB の上げは **DRAM 3 往復/バイト** (ページキャッシュ読み → pinned 書き
   → DMA 読み) で、copy だけで 51 GB/s = 実質上限。バイトを減らす以外に
   速くする手が無い (task 227 の探針と同じ結論)。
2. バイト数 = 「チャンク枚数 × 層 × ミス率」。14k は 9 枚 (1536)。枚数を
   減らすには 1 チャンクを大きくするしかなく、その作業域 (mix の res/scratch +
   MoE の行バッファ + staging) は **expert の LRU から借りるしかない**。
3. 借りる量と decode のコストは実測で「1 GB で +1.5〜3%」。
   - 3.8k -10% には 2 枚 (2047 以下) で足りるが、実測 -2.3%。
   - 14k -15% には 5 枚以下 (chunk 3072 以上) が必要で、追加 1.2 GB 以上
     → decode +3.3〜8.5%。
4. task 265 のプール (作業域を長い prefill の間だけ借りる) は、まさにこの
   壁を正面から取った唯一の形で、prefill は -34%/-54% に到達した。
   演算子判断で既定 on になっている (int-auto 89b7f8a44)。このタスクの
   「プール以外で」という条件では、同じ効果を出す別解が見つからなかった。

## 5. 数値の扱い

- 数値経路: 2 コミットとも**既定 off**。`DENNO_PF_CHAIN_BIG=0` (既定) と
  `DENNO_PREFILL_WORK_MB` 未設定では、カーネル・引数・発行順・コピーは
  1 つも変わらない (t270c の base2 と tab/tab1g の md5 は 6 本とも SAME)。
- チャンクを変えた COND (t270a/b) は md5 が DIFF。トークンのまとめ方が
  変わるので、丸め順と LRU の履歴が変わる (意図した差分)。
- 3 ラウンドの数値はすべてラボの実測。見積りは「推定」と書いた。

## 6. テスト

- merge 後 (`89b7f8a44` + このブランチ) で:
  `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`
  → **1,120 通過 / 0 失敗 / 99 ignored**。
  `--test kernel_invariants` → **90 通過 / 0 失敗**。
- merge 前 (`3092247f1`) は 1,086 / 88。差は int-auto 側のテスト。
- 追加した契約テスト: `cuda::tests::chain_big_env_and_nt_rule`
  (ゲートの真理値表、nt < 2048 / >= 2048 / dk=256 / 床の 4 条件)。

## 7. GPU で確認してほしいこと

1. merge 後の既定 (`DENNO_PF_POOL` 既定 on) で 14k cold の台帳を取り直すと、
   task 265 の 16.5 s に対してこのブランチが **1% も悪化していない**こと
   (このタスクのコードは既定 off なので、差が出たら merge の取りこぼし)。
2. `DENNO_PF_CHAIN_BIG=1 DENNO_PF_CHUNK=3072` を **プール on の上**で測ると、
   プールの作業域とぶつからないか (プールは既に 2.7 GB 借りるので、
   たぶん両立しない。両立しないなら `DENNO_PF_CHAIN_BIG` は将来の掃除対象)。
3. `DENNO_PF_TAB_ASYNC=1` 単体は -1% (t270c)。既定 off のままでよいが、
   `ptrtab` の計上は「待ちの付け替え」なので、台帳を読むときは
   `stage_copy + ptrtab + up_wait + sync` の和で見るのが正しい。
