ADOPT: yes t609c
# task 609 (dev2) — 既定モデル prefill: 14k cold で **-4.2%** (中央値)。
# `DENNO_PF_ER_PAGE_HIT` (既定 on): 窓の広い層の expert 先読みを、ストリームの実測が
# page cache 速度 (>12 GB/s) のときは止める。14k の pread 18,465 -> 333 MB、
# `ra_ms` 1,068.7 -> 40.2、`sub other` 810 -> 500 ms、`gpu_busy` は不変、
# 壁 5,277 -> 5,053 ms。3,826 は -1.1% (悪化なし)、decode 不変、md5 全 COND SAME。

branch **`br609`** = 開始時の `int-auto` **`302f07560`** (merge: br576)。
**終端でも int-auto は `302f07560` のまま** (fetch し直して確認。merge する物は無い)。
`br605` は `302f07560` で既に int-auto に入っているので merge 不要。

数値はすべて GPU ラボの実測 (`SCENARIO=depth8k`、`PROFILE=1`、
`MODEL=flash-next-UD-IQ3_XXS`、baseline COND = タスク指定の env
`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0
DENNO_GPU_WEIGHT_MB=12000`、先頭に捨て COND)。ラウンドは 3 本使い切った
(t609a 段プロファイル / t609b 汚染で捨て / t609c A/B)。

## 0. 結論 (6 行)

1. **採る (`ADOPT: yes t609c`)。** 14k cold の壁が baseline の中央値
   **5,277 ms** (5,277 / 5,453 / 5,196 の 3 本) から **5,053 ms**
   (5,148 / 4,959 の 2 本) へ **-4.2%** (平均 -4.8%)。
   目標 (3,826 か 14k が 2.5% 速い) を 14k 側で満たした。
2. **他は悪化していない。** 3,826 cold は 2.18 / 2.19 / 2.14 -> 2.18 / 2.13 s
   (中央値 -1.1%)、14k 後 decode 11.1-11.5 → 11.4-11.5 ms/tok、3,826 後
   11.3-11.4 → 11.3-11.4、短い decode 9.8-9.8 → 9.9-9.9、短い prefill 0.55 s 不変。
3. **GPU 仕事は 1 命令も変えていない。** `[q4ledger-pf] close` の `gpu_busy` は
   4,159.5 / 4,171.0 / 4,164.6 (base) 対 4,142.7 / 4,143.7 (gate) で ±0.3%。
   差は**露出ホスト時間** 1,737 / 1,884 / 1,647 → 1,641 / 1,451 ms から出ている。
4. **機構:** 14k の expert 28,015 MB のうち 18,465 MB を `pread` で
   「先読み」していたが、その中身は page cache (同じ行の `io_read_MB=782`)。
   先読みは `page cache -> rc_plan -> pinned -> DMA` の 5 往復で、しないときは
   3 往復で済む。`sub other` は base 883.3 / 752.0 / 793.6 → gate 645.4 / 355.8 ms
   (差 -310 ms = 先読みの計画・router 同期・readahead の join)。
5. 発火の証拠: `[exp-pf-stream]` に **`er_ph_layers=47`** (48 層中 47。
   1 層目は実測がまだ 256 MiB に満たないので先読みする)、`pread_MB` 18,465 →
   337.1 / 333.1、`ra_ms` 1,068.7 → 40.2 / 42.2、`er_layers` 48 → 1。
6. **ディスク律速のモデルは従来どおり先読みする** (UD-Q4_K_XL の 5.9-7.1 GB/s は
   閾値 12 GB/s 以下)。実測が薄いあいだ (256 MiB 未満) も先読みする側に倒す。

## 1. 段プロファイル (Do 1/2: 現在の int-auto を取り直した)

t609a (302f07560、PROFILE=1、捨て COND の次)。**3,826 cold**:
`wall=2,124.6 gpu_busy=1,203.7 exposed_host=958.4`
experts 1,317.9 (busy 525.6) / **ssm 382.0 (361.4)** / mix 284.0 (183.1) /
other 95.6 (51.0) / full 45.1 (82.6)。

**14k cold**: `wall=5,530.2 gpu_busy=4,149.0 exposed_host=2,004.3`
experts 2,114.4 (1,421.2) / **ssm 1,927.9 (1,317.7)** / full 707.4 (552.5) /
other 747.1 (201.0) / mix 33.5 (656.6)。
内訳: `sub other` = iter_top 305.4 (iter_ple 224.1 + iter_res 57.2) +
**pf_er 242.4** + pf_teardown 87.7 + res_up 68.2。
`[pf-host] up_wait=1,551.0`、`[exp-pf-stream] up_MB=28,015.6 copy_s=1.03
ready_s=1.55 pread_MB=18,328.5 ra_ms=1,110.0`、`io_read_MB=782`。

### なぜこれを選んだか

P577 が ssm の中を割り (scan 708 / qkv 244 / z 153 / out 153)、P605 が scan を
叩いて閉じた。experts は P481/P490/P546/P557 が DDR の床まで詰めており
(実測 3 往復 67.5 GB / 62 GB/s = 1.09 s = 3,826 の up_wait 1,073 ms と一致)、
mix は P585-P601 が 3 タスク、full/other は P553/P561/P565/P581 が触っている。
**残っていた最大の「誰も数字を説明していない」塊が、14k の
`other` の pf_er 242.4 ms と `pread_MB=18,465` (ra_ms 1,068.7) の正体**だった。
page cache から来ているバイトを、わざわざ RAM の置き場へ写して
5 往復で運んでいた (この経路は task 474 が「ディスクから読む層」のために
作ったもので、`early_route_window_ok` にはレートの条件が無い)。
`.cu`/`.ptx` を触らずに済む (ホスト側だけ) ことも選定理由。

## 2. 変えたこと (1 本、env ゲート)

`crates/denno-engine/src/model/exp_pf.rs`:
- `ER_PAGE_HIT_ENV` = `DENNO_PF_ER_PAGE_HIT`、既定 **on** (切り戻しは `=0`)。
- `er_stream_page_hit(bytes, copy_ns)`: 実測が `ER_SMALL_MIN_BYTES` (256 MiB)
  以上たまっていて、かつ `> ER_SMALL_MAX_GBS` (12 GB/s) のとき true。
  薄い実測と `copy_ns == 0` は false (先読みする側 = 従来どおり)。
- `er_page_hit_layers()` / `note_er_page_hit()`: 止めた層数を数え、
  `early_route_stats()` が `[exp-pf-stream]` の末尾へ `er_ph_layers=N` を出す
  (発火の証拠。**緑は分岐の証拠にならない**ので数を出す)。

`crates/denno-engine/src/model/qwen4exp.rs` (`pf_early_route_prefetch`):
- 窓の判定 (`wide` / `narrow`) の直後・**router の同期より前**で
  `wide && er_page_hit_on() && er_stream_page_hit(rate)` なら `None` を返す。
  router の GPU 仕事 (14k で 48 層ぶん) も払わない (task 482 と同じ位置)。

値は 1 バイトも変えない (先読みを飛ばすだけで、MoE は同じバイト列を mmap から
写す)。丸め順も変わらない。

## 3. 数値 (lab 実測・t609c、`load_before=14.47`)

| COND | 14k cold s | 3,826 cold s | 短い decode ms/tok | md5 |
| --- | ---: | ---: | ---: | --- |
| b (base) | 5.28 | 2.18 | 9.8 | SAME |
| b2 (base) | 5.46 | 2.19 | 9.8 | SAME |
| b3 (base) | 5.20 | 2.14 | 9.9 | SAME |
| **ph** (gate) | **5.15** | 2.18 | 9.8 | SAME |
| **ph2** (gate) | **4.96** | 2.13 | 9.9 | SAME |
| 中央値 base -> gate | 5,277 -> 5,053 ms (**-4.2%**) | 2.18 -> 2.155 s (-1.1%) | - | - |

台帳 (`[q4ledger-pf] tok=14018`):

| | b | b2 | b3 | ph | ph2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| wall ms | 5,277.2 | 5,453.3 | 5,195.5 | **5,147.5** | **4,959.1** |
| gpu_busy ms | 4,159.5 | 4,171.0 | 4,164.6 | 4,142.7 | 4,143.7 |
| exposed_host ms | 1,737.1 | 1,883.9 | 1,647.1 | **1,640.9** | **1,451.0** |
| `sub other` ms | 883.3 | 752.0 | 793.6 | **645.4** | **355.8** |
| up_wait ms | 1,397.2 | 1,365.6 | 1,366.4 | 1,382.4 | 1,310.3 |
| pread_MB | 18,465 | 18,465 | 18,465 | **337.1** | **333.1** |
| ra_ms | 1,068.7 | 1,056.4 | 1,046.2 | **40.2** | **42.2** |
| copy_s | 0.81 | 0.76 | 0.76 | 0.78 | 0.70 |
| ready_s | 1.40 | 1.36 | 1.36 | 1.38 | 1.31 |

`gpu_busy` が ±0.3% で動かないのに壁が -4% 動く = 差は全部ホスト側。
`copy_s` / `ready_s` (MoE の写しと待ち) はほぼ動かない (-1/-2%) ので、
**効いたのは写しの速さではなく、先読み自身の仕事 (計画・router 同期・
readahead の join) と、その 2 往復ぶんの DDR だった**。

## 4. 数値の扱い

- 採否は**同じ run の中の baseline COND (b / b2 / b3) と gate COND (ph / ph2)** の
  比較で読んだ (lab の `vs x0` は捨て COND との比較なので使っていない)。
- この run は `load_before=14.47` (他の dev のビルドと重なった) で、
  14k の 1 サンプルは COND 位置で ±7% 動く。gate の 2 本 (5.15 / 4.96) は
  どちらも base の 3 本 (5.20-5.46) より速い。**段の台帳 (gpu_busy 不変・
  exposed_host と `sub other` が -300 ms) が同じ向き**なので採った。
  きれいな環境で取り直すなら 14k だけで 1 リクエスト使うこと。
- 見積りは「推定」と書いた (36.6 GB の DDR)。それ以外は実測。
- 3,826 の -1.1% は誤差の内側 (gate は 3,826 では発火しない: `er_*` は 3,826 で
  0 のまま)。**悪化ではない**ことだけが言える。

## 5. 試して捨てたもの

- **t609b (302f07560、env だけの切り分け: er0 / pr0 / ep)**: この run は
  **page cache が冷えていた** (`[exp-pf-stream] copy_s=2.61-2.68 s` 対
  t609a の 1.03 s、`[exp-ram-probe] io_read` がディスク)。3,826 の壁が
  6.4-7.7 s (暖かいとき 2.1-2.2 s)、14k が 6.8-7.3 s で、COND 間の差は
  すべてディスクの当たり外れだった。**採否には使っていない** (ラウンドは
  1 本消費した)。
- その run の数字は傾向として er0 / pr0 / ep が b より速く見えるが、
  b (14k 7.32 s) だけが位置で外れていた (b2 6.91 s) ので信号ではない。
- ncu は P605 の t605a で `NO_REPORT` (ERR_NVGPUCTRPERM) だったので使っていない。

## 6. 本文 (健全性)

t609c の ph_1 c8000_cold (14k、md5 `672fec07` = baseline と同一):

> prefillの遅延は計算量ではなく、ホスト往復とメモリコピーが律速であった。
> 計測の誤認や静かな退行を排除しつつ、デバイス常駐化と転送経路の最適化を段階的に実施した。
> その結果、1.5Bモデルで395.8秒から12.2秒へ、9Bモデルで46.2秒から10.11秒へと大幅な高速化を達成した。

`!!!!` / 空 / プロンプトの反復なし、errors 0、md5 は全 6 COND × 全 6 文脈 SAME。

## 7. テスト (このコンテナで実行済み)

- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`
  → **1422 passed / 0 failed / 103 ignored**。
- `cargo test --release -p denno-engine --features cuda --test kernel_invariants
  -- --test-threads=1` → **153 passed / 0 failed**。
- `cargo build --workspace --release` (cuda 無し) → ok。
  `cargo test --workspace --release -- --test-threads=1` (cuda 無し) → 失敗 0。
- 追加テスト 2 本:
  - `model::exp_pf::tests::er_page_hit_gate_and_stats`: env の真理値表
    (未設定は既定 = on)、`er_stream_page_hit` の境界 (42 / 16.5 GB/s は止める、
    5.9 GB/s・256 MiB 未満・copy_ns=0 は止めない)、台帳 (`er_ph_layers=`) が
    `[exp-pf-stream]` の末尾に出ること、テスト上書きでゲートが切り替わること。
  - `model::qwen4exp::tests::pf609_er_page_hit_is_wired_into_the_prefetch`:
    `include_str!("qwen4exp.rs")` で `pf_early_route_prefetch` の本体が
    ゲート・判定・台帳の 3 つを呼び、**router の同期より前**にあり、
    計画ループより前にあることを固定 (配線忘れの再発防止)。
- `.cu` / `.ptx` は 1 バイトも触っていない。

## 8. 前提が崩れた点 (正直に)

- **`DENNO_PF_EARLY_ROUTE` (task 470/474/486/489) の前提が既定モデルには
  ずれていた。** 474/486 は「page cache の層では先読みは 1 バイトも得をしない」
  と書いているが、その判定 (`er_small_pass`) は**窓の狭い層にしか掛かって
  いなかった**。既定モデルの 14k は窓が広い (残り 6 チャンク) ので、
  12 GB/s を超える page cache 速度でも 18.5 GB を 5 往復で運んでいた。
  本タスクはその穴を塞いだだけで、470/474 がディスク向けに作った経路は残した。
- lab のラウンドは 3 本使い切った。うち 1 本 (t609b) は page cache が冷えて
  いて使えていない。**同じ日の別の dev の run が page cache を追い出した**
  ようで、t609b の `copy_s` は 2.6 s (ディスク律速) だった。
  次に測る人は `RESULT.md` の `copy_s` を最初に見ること
  (暖かいと 0.5-0.8 s、冷えると 2.6 s)。

## 9. GPU で確認してほしいこと

- 静かなラボで 14k cold を取り直す (1 リクエスト、`b` / `ph` / `b2` / `ph2`)。
  本タスクの -4.2% は `load_before=14.47` の run なので、きれいな環境で
  -2.5% 以上が再現するかを確かめてほしい。段の台帳 (`sub other` が -300 ms、
  `gpu_busy` 不変) が再現の判定に使える。
- UD-Q4_K_XL (ディスク律速) で **退行しない**ことの確認: あのモデルは
  5.9-7.1 GB/s なのでゲートは発火せず (`er_ph_layers` が出ない)、
  従来どおり先読みする。1 リクエストで足りる。
- 短い decode / 14k 後 decode が動いていないこと (本 run では不変)。

## 10. VERDICT 行 (t609c、lab の出力そのまま)

```text
VERDICT b vs x0 OK decode -0.4% prefill +0.0% md5 SAME errors 0
VERDICT ph vs x0 OK decode -0.4% prefill +0.0% md5 SAME errors 0
VERDICT b2 vs x0 OK decode -0.4% prefill +0.0% md5 SAME errors 0
VERDICT ph2 vs x0 OK decode +0.0% prefill +0.0% md5 SAME errors 0
VERDICT b3 vs x0 OK decode +0.0% prefill +0.0% md5 SAME errors 0
VERDICT_DEPTH b c8000_cold vs x0 decode -8.7% prefill -31.5%
VERDICT_DEPTH ph c8000_cold vs x0 decode -8.7% prefill -33.2%
VERDICT_DEPTH ph2 c8000_cold vs x0 decode -7.2% prefill -35.7%
VERDICT_DEPTH b c2000_cold vs x0 decode -10.3% prefill -36.1%
VERDICT_DEPTH ph2 c2000_cold vs x0 decode -11.4% prefill -37.5%
VERDICT baseline x0 errors 0 text ok
```

(`vs x0` は捨て COND との比較なので採否には使っていない。採否は §3 の
b / b2 / b3 対 ph / ph2。)
