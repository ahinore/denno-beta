ADOPT: yes t518a t518b

# REPORT P518 — UD-Q4_K_XL: **層ごとの小さな同期コピー (4 KiB .. 1 MiB) を pinned バウンスへ**
# 通した。**3,826 cold prefill が 2 ラウンドとも -4.6%**、14k cold が -7.0 / -3.9%、
# cold の深い decode が -17〜-24%。warm 短 decode は -0.7% でタスク目標 (3%) には
# 届かず、prefill も 5% に僅かに届かない。-4.6% は運用規則
# (段で 2 ラウンド 2% 以上 / 他が 1% 超悪化しない / 本文が健全) を満たすので採った。

branch: **br518** = 開始時の int-auto **890fc66** + 本タスクの 2 本。**br514 は merge して
いない** (中身は page cache の計器と `DENNO_PF_FADV_END` の既定 off のゲートで、
本タスクには不要。P514 の結論は REPORT_P514.md から読める)。終端で int-auto を
fetch し直したら **f24a9d5 (merge: br505)** に動いていたので merge した (両側そのまま、
`build_ptx.sh` の衝突なし)。

## 0. 結論 (7 行)

1. **採る (`ADOPT: yes t518a t518b`)。** 層ごとの小さな同期コピー (< 1 MiB) を
   task 358 の共有 pinned バウンスに通すだけで、**3,826 cold prefill が
   16.47 -> 15.71 s (-4.6%)、14k cold が 21.4 -> 20.3 s (-5.1%)**。2 ラウンドとも
   同じ向き・同じ大きさ。既定モデルは **md5 全リクエスト SAME**・decode 10.6 ms/tok。
2. **採った理由は運用規則の側。** タスク目標 (3,826/14k で 5%、短い decode で 3%) は
   満たしていない (最大 -4.6%)。「2 ラウンドの中間値で 2% 以上・他が 1% 超悪化しない・
   本文が健全」を満たすので、既定 on にした。切り戻しは `DENNO_DEC_PIN_SMALL=0`。
3. **fresh profile (Do 1/Do 2)**: UD-Q4_K_XL の warm 短 decode は **12.7 ms/tok**
   (層の壁) で、段は **moe 6.14 (experts 5.32 + router 0.63 + shared 0.18) /
   attn 4.18 (ssm) / mix 2.21 / head 0.61**。experts が最大 (42%) で、その GPU 側は
   70 us/層 = **gate 440 / up 451 / down 352 GB/s**。**4 段すべてが m=1 GEMV の
   帯域律速**で、ホスト側に残っていたのは小さな同期コピーだけだった (§1)。
4. **なぜ小さなコピーか**: 1 リクエストで `[pf-pin]` が数える小さな往復は
   **3,826 で 12,572 回 / 134.8 MB**、decode で **131 回/token**。pageable の
   10 KB コピーは 11 us/回 (task 358 の実測) なので、3,826 で 0.1-0.3 s に相当する。
   実測の差 (-0.29 / -0.27 s) と桁が合う。
5. **task 318 の 1 MiB の床を外したのは decode では初**。P358 は prefill の大きな転送
   だけで判定して「壁が動かない」と棄却したが、**床の下 (4 KiB .. 1 MiB) は
   一度も測られていなかった**。ここが今回の当たり。
6. **warm 短 decode は動かない (-0.7%)**。段が帯域律速なので、ホストのコピーを
   速くしても壁は変わらない (予想どおり)。decode を 3% 動かすには `.cu` が要る (§6)。
7. **分岐は発火している**: `[pf-pin] calls=... MB=...` が c のログにだけ出る
   (b は 0 行、3,826 の 1 リクエストで 12,572 回)。既定モデルでも 4,668 回出て
   **数値と md5 は不変**。

## 1. なぜこの段・この変更を選んだか (Do 2)

直前の int-auto (890fc66) の UD-Q4_K_XL を PROFILE=1 で測り直した (t518a の b)。
warm 短 decode (28+128 tok、2 本目のリクエスト) の内訳:

| 段 | ms/tok | 割合 | 中身 (実測) |
| --- | ---: | ---: | --- |
| moe | **6.14** | 48% | experts 5.32 / router 0.63 / shared 0.18 |
| attn (ssm) | 4.18 | 33% | GDN の qkv+z / scan / out |
| mix | 2.21 | 17% | hc mix 2 回/層 (97 呼び出し/token) |
| head | 0.61 | 5% | 最終 GEMV の待ち (sync 0.58) |
| その他 (embed/ple/norm/sample/detok) | 0.10 | 1% | — |
| 合計 (層の壁) | **12.73** | — | outside 込みで 13.4 ms/tok |

- **experts の GPU 側は 70 us/層** (seg-prof を持つ別ブランチ t526b の同一モデル実測:
  gate 17.2 + up 16.8 + silu 3.3 + down 29.9 + sum 2.9)。**gate 440 / up 451 /
  down 352 GB/s** = 5090 の HBM の 20-25%。**帯域律速で、.cu 無しでは動かせない。**
- seg の経路は intercept を通らないので `[q4ledger] experts wall=5.32 cpu=5.32` と
  なり、台帳では中身が割れない (副区間 `exp_*` も seg 経路では積まれない)。
  ここは P278 の `DENNO_SEG_PROF` (既定 off、kprof のイベントで 3-4 倍に膨らむ)
  でしか割れない。**計器の穴として §6 に残した。**
- 帯域律速でない残りは**層ごとの小さな同期コピー**だけだった。実測の回数は
  `[pf-pin]` (本タスクで既定 on にした計器) が数える:
  - 3,826 の prefill: **12,572 回 / 134.8 MB** (平均 10.7 KB)。
  - decode: **131 回/token** (seg の prologue H2D 10.8 KB と結果 D2H 10 KB × 48 層、
    GDN の xba H2D 10.3 KB と out D2H 10 KB × 36 層、mix の H2D、full のミラー 65 KB)。
- pageable の 10 KB コピーは **11 us/回** (task 358 が e=2,560 で実測)。pinned の
  バウンスは task 358 で**共有 1 本 (grow-only)** になっていて、確保の固定費は
  プロセスで数回しか払わない。それなのに **task 318 の 1 MiB の床が残っていて、
  この帯が一度も pinned を通っていなかった**。これが選んだ理由。

## 2. 変えたこと

- `crates/denno-engine/src/cuda.rs`
  - 新ゲート **`DENNO_DEC_PIN_SMALL`** (**既定 on**、`0`/`false`/`no`/`off` で切り戻し)。
  - `with_pf_pin` の入口を「従来の窓 (`DENNO_PF_PIN_STAGE` + 床 1 MiB)」**or**
    「小さな往復の窓 (4 KiB .. 1 MiB 未満)」にした。**prefill の大きな転送の判定は
    1 文字も変えていない** (`pf_pin_window_ok` はそのまま)。
  - `DevBuffer::to_vec` (GDN decode の層ごとの out D2H が使う唯一の口) も pinned を
    通すようにした。ゲート off / 確保失敗 / 転送エラーは従来の pageable 経路で
    丸ごとやり直す (値は同じ)。
  - 純関数 `dec_pin_small_from_value` / `pf_pin_small_window_from` を切り出した。
- **数値経路は 1 命令も変えない** (同じバイト列を別の経路で写すだけ)。丸め順も
  変わらない。既定 off では分岐 1 回だったのが、既定 on で「窓の判定 2 つ」に
  なるだけ (コピー 1 回あたり)。
- 計器: `[pf-pin] calls= MB= grows= waits= wait_ms=` がリクエスト終端に出る
  (task 358 が用意していた行。ゲート on で 1 回でも通れば出る)。

### テスト

- `cuda::task318_tests::dec_pin_small_window_and_gate_truth_table` (新規):
  窓の真理値表 (4 KiB は入る / 4 KiB-1 は入らない / 1 MiB は入らない / ゲート off は
  何も入らない) と env の解釈 (未設定・空は on、`0`/`off`/`false`/`no` は off)。
  **ゲートを on にしても 1 発も入らない**、**prefill の大きな転送を巻き込む**という
  2 つの壊れ方をここで落とす。

## 3. lab の結果

lab 名 (すべて `tree=denno-dev3` / SHA **0081ba3** / `PROFILE=1` / `ptx=none`) /
baseline COND = タスク文のとおり `DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000
DENNO_EXPERT_GPU_MIN_HIT=0 DENNO_GPU_WEIGHT_MB=12000`:

- **t518a** = `MODEL=flash-next-UD-Q4_K_XL` / `SCENARIO=depth8k` / `load_before=3.03` /
  COND: x0 b c
- **t518b** = 同じ / `load_before=7.70` / COND: x0 b c (2 ラウンド目)
- **t518c** = `MODEL=flash-next-UD-IQ3_XXS` (既定) / `SCENARIO=depth` /
  `load_before=2.21` / COND: x0 b c

`c` = baseline + **`DENNO_DEC_PIN_SMALL=1`** (今回の変更)。

### 3.1 3,826 cold prefill (壁 s)

| COND | t518a r1 | t518a r2 | t518b r1 | t518b r2 | 中間値 |
| --- | ---: | ---: | ---: | ---: | ---: |
| b | 16.08 | 16.85 | 15.75 | 17.83 | **16.47** |
| c | 15.64 | 15.77 | 15.46 | 16.57 | **15.71** |

**-4.6%** (2 ラウンドとも同じ大きさ)。14k cold は **b 20.56 / 22.27 対
c 19.11 / 21.40 = -5.1%** (中間値)。

### 3.2 cold の深い decode (ms/tok)

| リクエスト | t518a b | t518a c | t518b b | t518b c |
| --- | ---: | ---: | ---: | ---: |
| c2000_cold | 28 / 30 | **22 / 21** | 22 / 30 | **22 / 21** |
| c2000_turn2 | 20 / 18 | 18 / 18 | 18 / 22 | 18 / 22 |
| c8000_cold | 32 | **19** | 31 | **29** |
| c8000_turn2 | 20 | 17 | 22 | 19 |

**c2000_cold が -17〜-24%** (中間値 28.8 対 21.8 / 25.9 対 21.5)。c8000 は 1 サンプル
/COND なので -6〜-39% と振れるが、向きは 2 ラウンドとも c が速い。**理由は
未解明**: cold の深い decode は expert のミス (CPU GEMV + ページフォルト) が支配的で、
そこの小さなコピーも pinned になるため、と推定している (確定はしていない)。

### 3.3 warm 短 decode (ms/tok、タスク目標の段)

| COND | t518a | t518b |
| --- | ---: | ---: |
| b (中間値) | 14.2 | 14.1 |
| c (中間値) | 14.1 | 14.0 |

**-0.7% (2 ラウンド同値)**。層の壁で見ると 12.73 -> 12.64 ms/tok。**目標の 3% には
届かない** (§6)。

### 3.4 既定モデル (t518c)

| COND | decode ms/tok | 3,826 cold s | c2000_cold decode | md5 |
| --- | ---: | ---: | ---: | --- |
| x0 | 10.6 | 1.87 | 13.8 | a:edaa84ad b:1938059f c2000_cold:9b28e018 c2000_turn2:8b02d5c3 |
| b | 10.6 | 1.87 | 12.9 | 同じ |
| c | 10.6 | 1.87 | 12.8 | 同じ |

- **md5 が全 COND・全リクエストで SAME**、errors 0、c2000_turn2 11.7 / 11.6 ms/tok。
  既定モデルでも `[pf-pin] calls=4668` が出る (= ゲートは効いている) のに数値は不変。

### 3.5 md5 について (言い訳のいらない範囲)

- c の変更は**同じバイト列を別の経路で写すだけ**なので、数値は変わりえない。
  既定モデル (t518c) の md5 が全 SAME なのが直接の証拠。
- UD の md5 は同じ COND の中でも**ラウンド間で動く** (`b` リクエストが
  t518a b:ffbf4402 / t518b b:ffbf4402 と c82883fe、c2000_turn2 が
  9e32c460 / fbaf5055 / c2f8eedb)。lab の prefix/再利用状態で出力が決定的でない
  ためで、b と c の差ではない。**安定している a / c2000_cold / c8000_cold は
  b と c で SAME。**

### 3.6 出力テキスト (健全性)

- `TXT`: `# B-Trees: A Comprehensive Explanation ... A **B-tree (Balanced Tree) is a
  self-balancing tree data structure that maintains sorted data and allows searches,
  sequential access, insertions` (b と c で同趣旨)。
- `TXTB`: 「Rust の「所有権（Ownership）とライフタイム（Lifetime）は、Rust を学ぶ上で
  最も重要であり、かつ最も難しい概念です。」(b と c で同趣旨)。
- `TXTD` (14k の回答、b と c で同趣旨): 「ご提示いただいた文書は、LLM推論エンジン
  （`denno-engine）における **Prefill（事前計算フェーズの性能最適化の歴史と教訓を
  詳細に記録したもの...」**。P498/P506/P510/P514 と同じ趣旨で、退行なし。
  `!!!!` のような壊れ方は 1 つも無い。

## 4. 数値の扱い

- すべて GPU ラボの実測。このコンテナは GPU 無しなのでマイクロベンチは取っていない。
- 壁時計は lab の値をそのまま。3,826 は 2 サンプル/COND/ラウンド、14k は
  1 サンプル/COND/ラウンドなので ±10% の幅を見込む (中間値で読んだ)。
- `[pf-pin]` の `calls` / `MB` は**本タスクの変更が通った回数とバイト数**で、
  負荷に依存しない。採否の主根拠は壁 (2 ラウンド同じ向き) と、この発火数。
- gbs (440/451/352) は t526b (別ブランチ、同じ UD モデル、`DENNO_SEG_PROF=1`) の
  実測。本タスクのツリーでは測っていない (seg-prof は kprof のイベントで膨らむ)。

## 5. なぜ 5% / 3% に届かなかったか

- **decode は帯域で決まっている。** experts 3.4 ms + ssm 2.6 ms + mix 1.7 ms +
  full 0.8 ms + head 0.6 ms = 約 9.1 ms が GPU の仕事で、壁 12.7 ms の 72%。
  残りは起動・待ち・小さなコピー。コピーを速くしても、GPU が詰まっていれば
  壁は動かない (§3.3)。
- **prefill の 5% には 0.4 ポイント足りない。** 3,826 の内訳は P514 のとおり
  「43.5 GB を staging へ写す」が支配的で、小さなコピーはその一部でしかない。
  ただし 2 ラウンドとも -4.6% と同じ大きさで出ているので、床を外した効果は
  本物である。

## 6. 次の担当者へ

1. **`down` (Q5_1) の seg カーネルが 352 GB/s しか出ていない** (gate Q4_K は 440)。
   同じ `moe_seg_layer_sum` の中で型だけ違う。ここを gate と同じ 440 にすると
   down が 29.9 -> 24 us/層 = **-0.28 ms/tok (decode -2.2%)**。`.cu` が要る。
2. **seg 経路の台帳の穴**: `[q4ledger] experts` は seg の起動を intercept しないので
   `cpu` 列に全部入り、副区間 (`exp_lookup` / `exp_miss` / `exp_sum` /
   `exp_upload`) も seg 経路では積まれない (P278 の `DENNO_SEG_PROF` は kprof の
   イベントで 3-4 倍に膨らむ)。次に experts 段を詰めるなら、まずここを直す
   (seg の 5 起動を `q4ledger::note_launch` に通し、`sub()` を積む)。
3. `DENNO_EXP_MISS_FLAT` (既定 on、`MISS_FLAT_ROWS=16`) は既にミスを行並列にして
   いる。**ミスの CPU 計算を「速くする」方向は閉じている** (P153/P361)。
4. 既定モデル (IQ 経路) は `gpu_layers=6144` で GPU が回るが、**UD (K 型) は
   `gpu_layers=0`** (`[exp-iq]` の行)。UD の decode の expert は seg 経路で回って
   いるので誤読しないこと (seg は `note_gpu_layer` を呼ばない)。
5. `DENNO_PF_PIN_STAGE` (1 MiB 以上も pinned) は prefill では P358 で棄却済み。
   今回の `DENNO_DEC_PIN_SMALL` は **1 MiB 未満だけ**なので別物。混ぜないこと。

## 7. テスト

```
CARGO_TARGET_DIR=/tmp/t518m cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1354 passed; 0 failed; 103 ignored   (int-auto f24a9d5 を merge した後の最終ツリー)
CARGO_TARGET_DIR=/tmp/t518m cargo test --release -p denno-engine --features cuda --test kernel_invariants
  -> 144 passed; 0 failed
CARGO_TARGET_DIR=/tmp/t518 cargo build --release -p denno-engine --features cuda -> ok
```

新規テスト: `cuda::task318_tests::dec_pin_small_window_and_gate_truth_table`。

## 8. GPU で確認してほしいこと

1. 本タスクは**数値経路を変えていない** (既定モデル md5 全 SAME + コード一致)。
   変わったのは「どのメモリ経由で写すか」だけ。
2. 既定 on のまま 3,826 / 14k の prefill を 1 ラウンド取って、-4.6% が再現するか
   (今回の採否は 0081ba3 = br505 を merge する前のツリーで測った。br505 は
   full 層 KV ミラーのゼロ埋めを消すので、UD の prefill にも効く可能性がある。
   **b と c の両方が速くなる方向なので、A/B の差は保たれると推定**している)。
3. `[pf-pin] calls=` がリクエスト終端に出ているか (出ていなければゲートが
   OFF か、窓の外)。`grows` は起動後に 2 で止まる (16 KB -> 128 KB)。
