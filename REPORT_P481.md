ADOPT: yes t481b t481c

# task 481 (dev2) — 既定モデル prefill: expert 上げの「3 往復」を 2 往復にした
# (キャッシュありの小さいバウンス 2 枚を交互に使い、写しと DMA を重ねる)。
# 3,826 cold -7.5% / -10.4%、14k cold -2.9% / -3.0% (2 ラウンド)、decode 不変

branch `br481` = 開始時の `int-auto` **`b89a08a0e`** (merge: br474)。**br477 は merge
していない**: `ADOPT: no t477a t477b` で、残したのは計器 (`[pf-kvwb]` の 1 行) だけで、
本タスクの対象 (上げの写し) とは別の場所。終端で int-auto は **`d034ef56a`**
(merge: br482 = 窓が足りない層の捨て routing を消す) へ動いたので merge した (§7)。
br482 と本タスクは触る場所が違う (向こうは `pf_early_route_prefetch` の先頭、
こちらは上げスレッド `pf_up_run`)。

数値はすべて GPU ラボの実測 (`SCENARIO=depth8k`、`PROFILE=1`、
`model=flash-next-UD-IQ3_XXS`、`ptx=none`、ベースライン COND = int-auto の既定 +
`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0
DENNO_GPU_WEIGHT_MB=12000`、先頭に捨て COND)。lab 名と測った SHA:
**t481a = `53e283bcf`** (5 COND、分割経路が重みを壊した回)、
**t481b = `6b438a4e4`** (5 COND、修復後 1 ラウンド目)、
**t481c = `9d8459ccd`** (5 COND、既定 on にして off/on を 2 回ずつ)。
3 ラウンド (上限いっぱい)。

## 0. 結論 (6 行)

1. **採る (`ADOPT: yes t481b t481c`)。両方の目標を 2 ラウンドで満たした。**
   3,826 cold **-7.5% / -10.4%**、14k cold **-2.9% / -3.0%**、decode は不変
   (11.1-11.2 ms/tok)、md5 は全 COND・全 6 文脈 **SAME**、errors 0、text sane (§5)。
2. **取った段**: experts 段。3,826 の壁 2,119 ms のうち experts が 1,270 ms (60%) で、
   その中の**ホストの写しが 1,090 ms (壁の 51%)**。これは同じ run の計器で
   「動かすバイト数に対して時間が異常」な唯一の列だった (22.4 GB / 20.6 GB/s)。
3. **機序**: 上げスレッドは従来「page cache を読む → write-combined のバウンスへ書く
   → DMA がバウンスを読み直す」の **3 往復**で、DRAM の読みが 2 本あった。
   キャッシュありの pinned 2 枚 (8 MiB) を交互に使い、**写し(i+1) を DMA(i) と
   重ねる**と、DMA が読む行はまだ L3 (9800X3D は 96 MB) にあるので読み直しが
   DRAM に降りない。実測で **copy_s 1.09 → 0.73 s (-33%) / 1.31 → 0.90 s (-32%)**、
   `ready_s` 1.04 → 0.88 / 1.37 → 1.25 (§4)。
4. **値は 1 バイトも変えない**: デバイスへ渡るバイト列は 1 回写しと同一
   (`split_at_copy_reassembles_to_the_same_bytes` が分割の組み立てを固定)。
   md5 も 2 ラウンド全 COND SAME。
5. **既定 on** (`DENNO_PF_UP_SPLIT_KB=8192`)。切り戻しは `=0` / `off`。
   16 MiB も同じだけ効いた (3,826 1.96 / 14k 5.09) が、L3 に余裕のある 8 MiB を既定にした。
6. **lab t481a で踏んだ穴**: バウンスは依頼 (グループ) をまたいで使い回すので、
   再利用待ちを「この依頼の中で 2 個前か」で判定すると、**前の依頼の DMA がまだ
   読んでいるバウンスを次の依頼が上書きして重みが壊れる** (3,826 が 41 s、14k は空出力、
   md5 DIFF)。`used: [bool; 2]` で「過去に使ったか」を見る形に直した (§3.2)。

## 1. Do 1: 現在の int-auto の段プロファイル (t481b の base2、実測)

数値は t481b base2 = 同じ run の末尾 COND (`/work/gpu_lab/done/t481b/base2_1.log`)。

### 1.1 3,826 cold (壁 2,118.7 ms / GPU busy 1,222.5 ms)

| 段 | wall ms | GPU busy ms | 露出 ms | 中身 |
| --- | ---: | ---: | ---: | --- |
| experts | 1,269.7 | 531.4 | **738.4** | 上げ 22,455.5 MB / `copy_s` 1.09 (= 壁の 51%) |
| ssm | 345.5 | 361.8 | 0.0 | GPU 律速 |
| mix | 265.2 | 183.5 | 81.6 | 別ストリーム |
| other | 142.4 | 63.1 | 79.3 | res_up 25.6 |
| full | 95.9 | 82.6 | 13.3 | |
| router | 68.8 | - | - | |

### 1.2 14k cold (壁 5,166.7 ms / GPU busy 4,161.3 ms)

| 段 | wall ms | GPU busy ms | 露出 ms |
| --- | ---: | ---: | ---: |
| ssm | 1,951.3 | 1,321.8 | 629.5 |
| experts | 1,839.0 | 1,425.1 | 413.9 |
| full | 800.0 | 553.4 | 246.6 |
| other | 546.7 | 201.7 | 345.0 |
| mix | 29.7 | 659.4 | 0.0 (完全に非同期) |

- `[exp-pf-stream]` (base2): 3,826 は `up_MB=22455.5 copy_s=1.09 ready_s=1.04`、
  14k は `up_MB=28015.6 copy_s=1.31 ready_s=1.37`。
- **14k は GPU が 80.5% 詰まっている** (P477 と同じ)。expert の上げ (1.37 s) も
  GPU (1.43 s) とほぼ同じで、どちらも床。**3,826 は上げだけが床** (GPU は 42% しか
  使っていない)。

## 2. なぜこの段を取ったか (Do 2)

- 14k の 4 段 (ssm 1,322 / experts 1,425 / mix 659 / full 553 GPU ms) は、
  どれも既にカーネル側を詰めてある: experts は TC 化 (250/416/411) で
  **bf16 TC の約 70%**、attention は P465 の bf16 化、mix は sn3 (Q8_0 TC) で
  「1 呼び出し 10 TFLOPS が実力」、ssm は wy5/wy6/wy7q。**GPU の仕事はもう床**。
- ホスト側は P477 が「full のコピーは GPU の drain 待ち」「res_up も同じ」と閉じた。
- 残ったのは **3,826 の expert 上げ (22.4 GB / 20.6 GB/s = 壁の 51%)**。これは
  「転送経路そのもの」で、P239 (分割) / P325 (pread) / P357 (page/reg) が
  それぞれ別の角度から測って閉じていたが、**どれもバウンスを 1 枚のまま**だった。
  3 往復のうち DMA の読み直しだけが「まだ L3 に載っている」ことに誰も注目して
  いなかったので、そこを取った (P239 の分割は 1 枚のバッファ内の分割で、
  再利用距離が 65-100 MB = L3 を超えていた)。

## 3. 変えたこと

### 3.1 `DENNO_PF_UP_SPLIT_KB` (新規、**既定 8192 = 8 MiB**)

`exp_pf`: `UP_SPLIT_KB_ENV` / `DEFAULT_UP_SPLIT_KB` (8192) / `up_split_kb_from_value`
(未設定・空・解析不能は既定、明示的な `0`/`off` は 0、正値は 64..=65536 KiB) /
`up_split_bytes` / `note_up_split`。`[exp-pf-stream]` の末尾に `usplit=` (積んだ
サブ塊の数) と `usplit_kb=` を足し、リクエスト境界で 0 に戻す。

`cuda` (`pf_up_run`): 分割が有効なら**キャッシュあり** (`cuMemHostAlloc` の
PORTABLE のみ = write-combined ではない) の pinned 8 MiB を 2 枚持ち、塊ごとに
「写す → 非同期 H2D を積む → そのバウンスの完了イベントを記録」を繰り返す。
次に同じバウンスへ書く前に必ず前回の DMA の完了を待つ。最後にグループ全体の
`up_ev` を記録する (呼び出し側の待ち方は従来と同一)。既定 0 では従来の
「全部写して 1 回 DMA」と 1 命令も変わらない。

`pf_copy_into_pin_at`: `[start, start+len)` をバウンスの先頭へ**詰めて**写す
(既存の `pf_copy_into_pin_range` は絶対オフセットへ書く)。区間の切り方だけが
違い、デバイスへ渡るバイト列は 1 回写しと同一。

### 3.2 t481a で踏んだ穴 (commit 2/n で修復)

バウンス 2 枚は依頼 (グループ) をまたいで使い回す。最初の実装は待ち判定を
「この依頼の中で 2 個前か」(`i >= 2`) にしていたため、**前の依頼の最後の DMA**
(上げストリームには先読みぶんが積まれている) がまだ読んでいるバウンスを
次の依頼が上書きした。結果: 3,826 が **41.2 s** (experts 段 39.6 s、22 層が
CPU へ落ちた)、14k は**空出力** (md5 `d41d8cd9`)、md5 DIFF。
`PfUpSplit.used: [bool; 2]` を持たせ、一度でも DMA の転送元に使ったバウンスは
次に書く前に必ず `cuEventSynchronize` する形に直した (グループで戻さない)。

### 3.3 テスト (足した / 直した)

- `cuda::pf_stream_tests::split_at_copy_reassembles_to_the_same_bytes` (新規):
  隙間をまたぐ 4 区間を 24 バイトずつ 2 枚へ交互に詰めて繋ぐと、1 回写しと
  **1 バイトも変わらない** (整列の隙間を書かない)。数値不変の根拠。
- `exp_pf::tests::stream_gate_parses_values` を更新: `up_split_kb_from_value` の
  真偽表 (未設定 8192 / `0`・`off` で 0 / 下限 64 / 上限 65536)。
- `kernel_invariants::task481_up_split_dma_wiring` (新規): 分岐・分割経路・
  `used` の待ち・`usplit=` の計器・従来経路の残置を字面で固定する。

## 4. 数値 (lab、同じ run の中の対)

### 4.1 t481b (`6b438a4e4`、5 COND、load_before=2.01) — 修復後の 1 ラウンド目

| COND | 3,826 cold s | 14k cold s | decode ms/tok | copy_s 3,826 / 14k |
| --- | ---: | ---: | ---: | --- |
| x0 (捨て) | 2.26 | 7.46 (page cache が冷たい) | 11.1 | |
| base | 2.14 | 5.83 | 11.2 | |
| **sp8** (`UP_SPLIT_KB=8192`) | **1.96** | **5.03** | 11.2 | 0.80 / 0.94 |
| **sp16** (`=16384`) | **1.96** | 5.09 | 11.2 | 0.82 / 0.95 |
| base2 | 2.12 | 5.18 | 11.1 | 1.09 / 1.32 |

- **3,826 -7.5% (1.96 対 2.12)**、**14k -2.9% (5.03 対 5.18)**。decode は全 COND 同値。
- md5: **全 COND・全 6 文脈 SAME**、errors 0、text は全 COND `looks ok`。
- `usplit=2992` (3,826) / `3681` (14k) = 分割が発火した証拠。

### 4.2 t481c (`9d8459ccd`、5 COND、load_before=1.62) — 既定 on で 2 回ずつ

| COND | 3,826 cold s | 14k cold s | decode ms/tok | copy_s 3,826 / 14k |
| --- | ---: | ---: | ---: | --- |
| x0 (捨て、off) | 2.12 | 5.37 | 11.2 | 1.09 / 1.31 |
| off1 | 2.12 | 5.29 | 11.1 | 1.09 / 1.31 |
| **on1** | **1.91** | **5.15** | 11.2 | 0.73 / 0.90 |
| off2 | 2.12 | 5.22 | 11.2 | 1.09 / 1.32 |
| **on2** | **1.90** | **5.04** | 11.2 | 0.73 / 0.89 |

- **3,826 -10.4% (中央値 1.905 対 2.12)**、**14k -3.0% (5.10 対 5.25)**。
  off 同士 (2.12 / 2.12) と on 同士 (1.91 / 1.90) が揃っており、位置効果ではない。
- copy_s: 3,826 **1.09 → 0.73 (-33%)**、14k **1.31 → 0.90 (-32%)**。
  ready_s: 1.04 → 0.88、1.37 → 1.25。
- decode: 11.1-11.2 ms/tok (全 COND)、8k 後 turn2 は 12.1-13.0 で従来と同じ幅。
- md5: 全 COND・全 6 文脈 **SAME**、errors 0、text sane (§5)。

### 4.3 RESULT.md の VERDICT 行 (参考。`vs x0` は捨て COND 相手)

t481b: `VERDICT sp8 vs x0 OK decode +0.4% prefill -0.9% md5 SAME errors 0` /
`VERDICT sp16 vs x0 OK decode +0.7% prefill -1.8% md5 SAME errors 0` /
`VERDICT base2 vs x0 OK decode +0.0% prefill -0.9% md5 SAME errors 0`。
t481c: `VERDICT on1 vs x0 OK decode -0.7% prefill +2.7% md5 SAME errors 0` /
`VERDICT on2 vs x0 OK decode -0.7% prefill +0.9% md5 SAME errors 0` /
`VERDICT off1/off2 ... md5 SAME errors 0`。
(`vs x0` は捨て COND 相手なので採否には使っていない。同じ run の off と on の対で読んだ。)

## 5. 出力テキスト (健全性)

- t481c on1 の `c2000_cold` (3,826) 応答:
  > ご提示いただいた文書は、LLM推論エンジン（`denno-engine）における **Prefill（事前計算）フェーズの性能最適化の歴史を詳細に記録した技術レポートです。\n\n以下に、この文書の核心を、技術的背景、最適化の段階的変
- t481c on1 の `b` (Rust の所有権):
  > Rust の「所有権（Ownership）とライフタイム（Lifetime）は、Rust を学ぶ上で最も重要であり、かつ最も難しい概念です。\n\nしかし、これらは「メモリを安全に管理するためのルール**です。
- t481c on1 の `a` = `# B-Trees: A Comprehensive Explanation` (全 COND 同じ)。
- off1 と on1 の TXTD は同じ文で始まる (md5 も SAME)。`!!!!` / 空出力 / プロンプトの
  反復は無し (errors 0)。t481a の BROKEN は §3.2 の穴だけで、修復後に再発していない。

## 6. 数値の扱い

- 3,826 は各 COND 2 サンプル (depth8k の仕様)、14k は 1 COND 1 サンプル。
  t481c は off/on を 2 COND ずつ入れて 4 サンプルで読んだ (off 2.12/2.12、
  on 1.91/1.90 でばらつきは小さい)。
- `copy_s` はプロセス内の積算 (リクエスト境界で 0 に戻る) で、写しの実時間だけ
  (DMA の待ちは含まない)。`ready_s` はデータが揃うまでの時間。
- 14k の絶対値は run 間で 5.03-5.37 と振れる (page cache の温まり方)。採否は
  **同じ run の off との対**で読んだ (t481c: 5.25 対 5.10)。
- md5 SAME は「値が 1 ビットも変わっていない」ことの確認。

## 7. テスト (コードの最終 HEAD = この commit の親、int-auto `d034ef56a` を merge 済み)

```
CARGO_TARGET_DIR=/tmp/t481 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1348 passed / 0 failed / 103 ignored
CARGO_TARGET_DIR=/tmp/t481 cargo test --release -p denno-engine --features cuda --test kernel_invariants -- --test-threads=1
  -> 141 passed / 0 failed
CARGO_TARGET_DIR=/tmp/t481nc cargo build --workspace --release
  -> rc=0
CARGO_TARGET_DIR=/tmp/t481nc cargo test --workspace --release -- --test-threads=1
  -> rc=0 (FAILED / panicked 無し)
```

`.cu` / `.ptx` は 1 バイトも変えていない (PTX= も使っていない)。

## 8. GPU で確認してほしいこと (監督へ)

1. **既定の経路が変わっている**: `DENNO_PF_UP_SPLIT_KB` 未設定で分割が走る
   (既定 8192)。切り戻しは `DENNO_PF_UP_SPLIT_KB=0`。
2. 測ったのは `9d8459ccd` (br481 単体)。最終 HEAD は br482 を merge している
   (向こうは `pf_early_route_prefetch` の先頭、こちらは上げスレッドで別の場所)。
   merge 後の再測はラウンド上限 (3) のため取っていない。br482 は 3,826 の
   `other` を -49% するので、**合わせたときに -10% がどこまで残るかは未測**。
3. 次の一手: 3,826 の上げはまだ `ready_s` 0.88 s で、DRAM の読み 1 本ぶん
   (22.4 GB / 40 GB/s ≒ 0.56 s) には届いていない。残りは写しの並列度と
   サブ塊の大きさ (4 MiB / 2 MiB) の掃引で詰められる可能性がある (**推定**)。
   14k は GPU が 80.5% で、もうカーネル側しか残っていない (P477 と同じ)。
