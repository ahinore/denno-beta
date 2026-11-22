ADOPT: no t566a

# task 566 (dev3) — UD-Q4_K_XL の prefill: fresh な段プロファイルで残りを数え直し、
# 3,826 cold の壁の **87% は今も expert staging の写し = ディスク** (実効 3.1-3.2 GB/s、
# 0.78 MB x QD 20、1 MiB の天井 3.5 の 91%) であることを確認。**ラボは 2026-11-22
# 22:28 JST から HOLD** (comfyui が VRAM 30 GB を保持) で 1 ラウンドも走っておらず、
# **測定による採否はできていない**。次の 2 点を env だけで測れる形で要求に積んだ
# (t566a、queue=7 で待機): `DENNO_PF_PREAD_KB=384` と
# `DENNO_PF_STREAM_LOOKAHEAD=2 DENNO_PF_UP_WORKERS=3`。どちらも**今の構造
# (上げスレッド 2 本 + スレッドごとの 32 本 pool = rd_max 62) になってからは未測定**。

branch: **br566** = 開始時の int-auto **99d3858 (merge: br562)** から派生。
br562 は既に int-auto に入っていたので merge 不要。**コードは 1 行も変えていない**
(既定もゲートも不変)。終端の int-auto 再 fetch の結果は §7。

## 0. 結論 (6 行)

1. **採らない (`ADOPT: no t566a`)。** 理由は「負けた」ではなく **「測れていない」**:
   ラボは HOLD で、t566a は queue=7 のまま 1 度も走っていない (§5)。測っていない
   数字を実測として書かない規則に従い、既定は変えない。
2. **fresh な段プロファイルは取れた** (t562b = `e49eec9`、2026-11-22 20:33、UD-Q4_K_XL、
   depth8k、PROFILE=1)。この SHA と現 int-auto `99d3858` の差は `DENNO_SEG_GU16` の
   既定 on 化 (decode 専用) と REPORT だけで、**prefill の経路は同一** (§1.1)。
3. 3,826 cold の壁 6,520-7,382 ms の内訳: **experts 5,663-6,426 ms (87%)**、
   ssm 357-388 (5.3%)、mix 281-285 (3.9%)、other 171-242 (3.3%)、full 42-48 (0.6%)。
   experts の中身は GPU が 433 ms しか働かず、**exposed 5,229-5,993 ms = staging の
   写し待ち**。14k cold (11,620 ms) も experts 8,159 (70%) が最大で、次が ssm 1,898
   (GPU busy 1,319、`.cu` 無しでは触れない) と other 853。
4. **ディスクを台帳から割り直した**: 3,826 cold で O_DIRECT の実読み **15.9-18.7 GB /
   20.6-24.1k 本**、1 本 0.78 MB・4.8-5.2 ms = **149-162 MB/s/本**、平均同時 **19-22 本**、
   実効 **3.11-3.20 GB/s**。t522 のプローブ (O_DIRECT 1 MiB x 8 本 = 3.5) の **91%**。
   **残りは 9%** で、そこに -5% を取る余地は薄い (§2)。だから次の候補は
   「読みの形を変えずに同時本数だけ上げる」2 つに絞った (§3)。
5. どちらの候補も**値は 1 ビットも変えない** (同じバイトを同じ宛先へ写すだけ。
   カーネルの積む順も同じ) ので、採れた場合は md5 SAME が期待値。
6. 既定モデルは今回 1 ラウンドも流せていない (§5)。両候補とも `pread` を使う層だけに
   効く (`src` を持たない既定モデルはワーカー 0 固定) が、**確認は lab で必要**。

## 1. fresh な段プロファイル (Do 1 / Do 2)

### 1.1 使った実測と、それが現 int-auto を代表する根拠

ラボが HOLD なので新しい run は取れない。直近の同じ木の run を使った:

- **t562b** = SHA `e49eec9`、MODEL `flash-next-UD-Q4_K_XL`、SCENARIO `depth8k`、
  PROFILE=1、COND `x0 b p b2` の **b**(= タスク文の baseline 4 本そのもの)。
  測定時刻 2026-11-22 20:33 JST (HOLD の 2 時間前)。
- `git diff --stat e49eec9 99d3858` = `crates/denno-engine/src/cuda.rs` 33 行
  (`DENNO_SEG_GU16` の既定 on 化と doc) + `REPORT_P562.md` のみ。gu16 は decode の
  seg 層 (expert GEMV) だけを置き換えるので、**prefill の写し・段は同一コード**。

段の値 (`[q4ledger-pf]`、ms)。3,826 は 2 ラウンド (b の r1 / r2)、14k は r1 の 1 本:

| 段 | 3,826 cold r1 / r2 | share | 14k cold | share |
| --- | ---: | ---: | ---: | ---: |
| **experts** | **6,425.9 / 5,662.6** | **87.0 / 86.9%** | **8,158.9** | **70.2%** |
| ssm | 387.8 / 357.3 | 5.3% | 1,898.1 | 16.3% |
| mix | 285.1 / 281.4 | 3.9% | 26.5 (busy 658.4) | 0.2% |
| other | 241.5 / 170.6 | 3.3% | 853.2 | 7.3% |
| full | 41.7 / 47.9 | 0.6% | 683.1 | 5.9% |
| **wall** | **7,382.0 / 6,519.9** | | **11,619.7** | |

experts の中身 (同じ行の列): cpu 6,251.5 / 5,488.6、**exposed 5,993.2 / 5,228.9**、
gpu busy 432.7 / 433.7、sync 128.9 ms / 96 回。**GPU は壁の 6% しか働いていない。**

14k の内訳: ssm は gpu busy 1,318.7 / exposed 579.3、other は iter_top 366.4 /
iter_ple 270.0 / pf_er 242.8 / pf_teardown 125.3 / res_up 67.7 / sync 238.5 x 50。
段の和 12,251.7 対 壁 11,619.7 = **632 ms は 2 つの段に二重に乗っている**
(非同期の GPU busy を段の壁と別に積むため)。`iter uncovered` 613.2 ms はこの
二重計上の裏返しで、**実体のある穴ではない** (§2.3)。

### 1.2 既に誰かが触った段 (task 500 以降、このモデル)

- **experts = staging の写し**: P502 / P514 / P522 (隣接区間の併合 +39% で棄却、
  `PIN_FILE_ORDER`)、P530 (384 KiB、グループの barrier を特定)、P534 (decode down)、
  P538 (**wide 32 MiB + 組 1 MiB**、採用)、P542 (`SPLIT_IO` 既定 off)、
  P546 (**O_DIRECT + 整列バウンス**、採用、-7..-11%)、P550 (**読みの大きさ・併合・
  グループ数 = 3 方向とも棄却**、床は 2.5-3.0 GB/s)、P554 (**上げスレッド 2 本**、
  採用)、P558 (**上げスレッドごとの pread pool**、採用、rd_max 32 -> 62)、
  P562 (decode の seg 層)。
- **14k の ssm**: P397 (`SCAN_WY5`)、P407、P460 (flashdec の分割)、P577 (scan 708 ms
  と特定)、P581 (`RES_MT`)。**残りは GPU busy 1,319 ms で `.cu` が要る** (本タスクは
  `.cu` 禁止)。
- **14k の other / full**: P482 (sub other の分解)、P553 (`KV_DEFER` / `KV_PUSH_PAR`)、
  P561 (`KV_RESERVE`)、P565 (`RES_PIN`)、P581 (`RES_MT`)。既定モデル向けの手で、
  UD では効いていない (P558 §6)。
- decode は P534 / P562 が直したばかりで、壁はホスト律律速 (P562 §6)。

### 1.3 なぜこの段・この次元を選んだか

- 段で一番大きいのは今も **experts (3,826 で 87%、14k で 70%)** で、その中身は
  staging の写し。**他の段は 5% 以下**なので、5% の目標はここ以外に無い。
- 写しの**バイト数は削れない**ことを台帳で確かめた: 3,826 で必要量 `req_MB=65,651`、
  VRAM 常駐 `res_MB=22,136` が抜かれ、`up_MB=43,515` を上げる。O_DIRECT の実読み
  15.9-18.7 GB は「page cache に載っていない分」で、**専門家の重みは層ごとに
  1 回しか読まない** (rows_per_expert 131.8、専門家ヒット 13,913-13,929 = 全専門家)。
  つまり `up_MB` に重複は無い (P550 の「形を変えても動かない」と整合)。
- 残る次元は **同時本数**だけ。ところが P550 が「同時発行数」を試した時は pool が
  1 本共有 (32 本) で、**同時 32 本が上限**だった。P558 が pool をスレッドごとに
  分けて 62 本にした後、**「読みを細かくする」と「グループを 3 枚使う」は未測定**。
  これが「誰も測っていない」ではなく「**測ったが、その後に構造が変わった**」の 2 点で、
  タスクの規則どおり**変わった点 (rd_max 32 -> 62) を名指し**して測り直す (§3)。

## 2. 台帳から割り直したディスク (Do 2)

### 2.1 実測値 (t562b の b、`[exp-pf-stream]`)

| 項目 | 3,826 r1 | 3,826 r2 | 14k |
| --- | ---: | ---: | ---: |
| `up_MB` | 43,515.1 | 43,588.1 | 52,740.7 |
| `copy_s` (2 スレッドの積算) | 11.72 | 10.24 | 13.63 |
| `pread_MB` / `pread_calls` | 24,860.5 / 31,845 | 27,696.0 / 35,073 | 31,128.2 / 36,588 |
| `odb_MB` / `odb_n` (= mincore ミス) | 18,729.1 / 24,100 | 15,930.7 / 20,594 | 21,967.2 / 27,831 |
| `odb_ms` (スレッドの積算) | 125,787 | 98,679 | 152,167 |
| `rd_max` | 63 | 63 | 64 |
| `grp` / `grp_miss` / `grp_hit_s` | 1,674 / 1,573 / 0.04 | 1,679 / 1,517 / 0.16 | 1,976 / 1,796 / 0.29 |

### 2.2 割り算 (すべて上の実測からの計算)

- **1 本あたり**: 0.78 MB を `odb_ms / odb_n` = **5.22 / 4.79 ms** → **149 / 162 MB/s/本**。
- **平均同時本数**: `odb_ms / (copy_s / 2)` = **21.5 / 19.3 本** (rd_max は 63 なので
  枠は余っている。**グループの barrier のせいで平均は 20 本前後**)。
- **実効**: 18.73 / 5.86 = **3.20 GB/s**、15.93 / 5.12 = **3.11 GB/s** (14k も 3.22)。
- **天井**: t522 のプローブは O_DIRECT で 1 MiB x 8 本 = 3.5、2 MiB = 4.3、4 MiB =
  4.6 GB/s。**今の読み (0.78 MB) の天井 3.5 に対して 91%**。
- つまり **壁 = copy (5.1-5.9 s) + 他 (約 1.0-1.5 s)** で、copy はディスクの 91%。
  残り 9% を全部取っても -7% が上限 (推定)。**-5% は「上限の 7 割を取る」勝負**。

### 2.3 追わなかったもの (記録)

- `iter uncovered` 613.2 ms (14k): 段の和が壁より 632 ms 大きいのと同じ量で、
  非同期 GPU busy の二重計上の裏返し。**実体ではない**ので触らない。
- 14k の `mix` は段の壁 26.5 ms で `busy` 658.4 ms。**GPU は別の段の裏で働いており、
  壁には出ていない**ので対象外。
- `other` の `pf_er` 242.8 ms / `iter_ple` 270.0 ms / `iter_top` 366.4 ms は
  P553/P565 が内訳を取った後の残りで、**単独では 3% 未満**。今回は触らない。

## 3. 変えたこと (コードは 0 行、lab 要求だけ)

測定できないので既定は動かさない。**既存の env ゲート 3 本**で測れる 2 仮説を
t566a に積んだ (`SCENARIO=depth8k`、PROFILE=1、x0 / b / k3 / d3 / k3d3 / b2):

1. **`k3` = `DENNO_PF_PREAD_KB=384`** (既存ゲート、既定 1 MiB)。
   0.93 MB の実区間が 3 本に割れ、**グループあたりのディスク読みが 13 -> 33 本**に
   なる。P530 はこれで `copy_s` 14.9 -> 10.6 s (2.9 -> 4.1 GB/s) を出したが、当時は
   pool が 1 本共有で同時 32 本が上限だった。P550 (t550c) は現行に近い木で
   「変わらない」と書いたが、**その時も pool は共有** (P558 の前)。**変わった点 =
   rd_max 32 -> 62**。読みの総バイトは不変。
2. **`d3` = `DENNO_PF_STREAM_LOOKAHEAD=2 DENNO_PF_UP_WORKERS=3`** (どちらも既存ゲート、
   上限は 2 と 3 で、ステージング 3 枚に合わせてある)。**3 グループを同時に流す**ので
   平均同時本数 20 -> 30 前後になる。`LOOKAHEAD=2` は P379 が**既定モデル**で +2.4% と
   棄却したが、既定モデルは写しが **DRAM 律速 21 GB/s** でスロット再利用待ちが出る
   領域。UD は **ディスク律速 3.1 GB/s / GPU busy 6%** で待ちの構造が違う
   (`pf_slot_prepare` の grow 待ちは容量が足りている限り入らない)。
   **変わった点 = モデルの律速が DRAM -> ディスク**。
3. **`k3d3`** = 両方 (細かい読み x 3 グループ = 同時 90 本前後)。

見積り (推定、測っていない): k3 が 3.4-3.8 GB/s なら壁 -3..-9%、d3 は天井に近ければ 0、
同時本数律速なら -5..-10%。**両方とも値を変えない** (同じバイトを同じ宛先へ)。

## 4. ラボの状態 (なぜ「測れていない」か)

- `/work/gpu_lab/HOLD` (2026-11-22 22:28 JST): 「comfyui container holds 30 GB VRAM
  on rtx5090; lab stays on hold until the operator stops it」。**最後に走った run は
  2026-11-22 20:55 (t562c)** で、以降 `done/` は増えていない。
- t566a は `req/t566a.req` に置いてあり、`lab_wait.sh t566a` は **PENDING queue=7**。
  HOLD が解ければ 6 COND x 2 ラウンドで走る (推定 40-60 分)。
- 要求の中身は §3 のとおり。`PTX=` は付けていない (カーネルを足していない)。

## 5. 数値の扱い

- 本文の段の値・台帳の値は**すべてラボの実測** (t562b の生ログ `b_1.log` / `b_2.log`)。
  割り算 (§2.2) と「推定」と書いた見積りだけが計算値。
- t562b は HOLD 直前の run で、`model=flash-next-UD-Q4_K_XL`、`ptx=none`。
  **同じ run の b (r1) と b2 の差**が位置ドリフトの目安 (3,826 で 7,382 対 6,520 =
  -11.7%、P550 の「同じ run でも 14% 動く」と同水準)。**絶対値は run 間で比べない。**
- 目標 (3,826 か 14k の cold prefill -5%、または短 decode -3%) の判定は
  **していない** (測っていない)。decode は本タスクで触っていない。

## 6. 次にやること (HOLD が解けたら、この順で)

1. `t566a` をそのまま流す (既に queue)。判定は**同じ run の b と末尾 b2 の両方**に対し、
   3,826 cold と 14k cold の `[q4ledger-pf] tok=3826/14018 wall=` で行う。
   `rd_max` / `odb_ms` / `copy_s` が仮説どおり動いたかも台帳で数える
   (k3 なら `odb_n` が 2.5 倍、d3 なら `rd_max` が 90 前後)。
2. 勝ちが 2 ラウンド 2% 以上なら、**既定を 1 行ずつ変える**
   (`DEFAULT_STREAM_LOOKAHEAD` / `DEFAULT_UP_WORKERS` / `pread_kb_default`)。
   採った後の確認 run (`off` で切り戻した対照) と、既定モデルの 1 ラウンド
   (`SCENARIO=depth`、x0 / b / p / b2) を必ず取る。**既定モデルは UD の run より前に
   置く** (P562 §5-2 のページキャッシュの罠)。
3. どちらも負けたら、ディスクは天井 (91%) と結論して **staging を離れる**。
   その場合の次は 14k の `ssm` (1,898 ms、GPU busy 1,319) しかないが、`.cu` の
   許可が要る (scan の 4 分割は P311 で済んでいる)。

## 7. テスト

```text
CARGO_TARGET_DIR=/tmp/t566   cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1412 passed; 0 failed; 103 ignored
CARGO_TARGET_DIR=/tmp/t566   cargo test --release -p denno-engine --features cuda --test kernel_invariants -- --test-threads=1
  -> 153 passed; 0 failed
CARGO_TARGET_DIR=/tmp/t566nc cargo build --workspace --release      -> ok
CARGO_TARGET_DIR=/tmp/t566nc cargo test  --workspace --release -- --test-threads=1
  -> passed=1553 failed=0 (全バイナリの `test result:` の合計)
```

- **コードを 1 行も変えていない**ので、テストは int-auto の木そのものの緑。
- 既知の赤 (`qwen4exp_decode spec_batch_stops_at_eos`、`qwen4exp_params` の validate 2 件)
  は本タスクでも同じ (触っていない)。
- 終端で int-auto を再 fetch した結果: **99d3858 のまま** (merge する物は無い)。

## 8. GPU で確認してほしいこと

1. **t566a を流す** (`/work/gpu_lab/req/t566a.req`、TREE=denno-dev3、SHA=99d3858e2)。
   HOLD が解けたら `bash /work/gpu_lab/lab_wait.sh t566a`。
2. 見る値: 3,826 / 14k cold の壁 (同じ run の b と b2 の両方)、`[exp-pf-stream]` の
   `odb_n` / `odb_ms` / `copy_s` / `rd_max`、`[q4ledger-pf]` の experts 段。
   **値はビット一致 (`c2000_cold` = `73510e1f`、`c8000_cold` = `47c833e2`) が期待値。**
3. 勝ったら既定を変えるのは私の次のタスクでよい。切り戻しは
   `DENNO_PF_PREAD_KB=1024` / `DENNO_PF_STREAM_LOOKAHEAD=1` / `DENNO_PF_UP_WORKERS=2`。
