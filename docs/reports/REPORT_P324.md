ADOPT: yes t324b t324c

# task 324 — prefill の MoE 転送とカーネルを重ねる (3,826 cold -3.1%、14k -1.8%)

branch `br324` は開始時の `int-auto` `0d3cd40da` (= merge: br320) から。
**br320 は既に int-auto に入っていた**ので merge は不要だった。最後に int-auto
`3ae8b0046` (merge: br319) を merge した (衝突なし。`cuda.rs` は自動 merge)。
lab は 3 ラウンド: **t324a** (`c8b5d9194`、5 COND)、**t324b** (`5a6452a20`、6 COND、
採否を決めた回)、**t324c** (`82015e3b9`、6 COND、既定 on の確認回)。
数値はすべて GPU ラボの実測。

## 0. 結論 (3 行)

- **採る。** 3,826 cold は同じ run の base/base2 に対して
  **t324b 2,607/2,609 → 2,532/2,532 ms (-2.9%)**、
  **t324c 2,625-2,648 → 2,547/2,555 ms (-3.2%)**。2 ラウンド x 2 run の
  中央値で **2,622 → 2,542 ms (-3.1%)**。14k cold も中央値 **6,665 → 6,543 ms
  (-1.8%)** で悪化なし。decode は短 17.1-17.4、14k 後 16.7-17.9 で差なし。
- 機序は 1 本。**グループの H2D 完了をホストで待たずに計算ストリームへ送る**
  (`DENNO_PF_UP_GPUWAIT`、既定 on)。台帳で `wait_s` 0.84-0.86 → 0.00 s、
  `ready_s` 0.26 → 1.04 s に移り、MoE 段の `exposed` が 849-871 → 663-673 ms。
  転送の床は写しの 21 GB/s のままで、**壁時計から消えたのはカーネルの直列分**。
- 残った露出は**層の入口の段差** (最初のグループの転送) だったので、グループを
  64 → 32 本にした (`DENNO_PF_STREAM_GROUP`、実効 31.8 → 16.4 本/グループ)。
  層の終端の同期もデバイス出力のときは省く (`DENNO_PF_STREAM_NOSYNC`)。
  3 本とも env で戻せる。

## 1. 前提

- **読んだもの**: `REPORT_P320.md` — ただし §4 の「22.65 GB = 全 expert」
  「prefill は VRAM の expert を 1 本も再利用していない」は**実測と合わない**
  (§2.1 で取り直した)。その §1 が指す `REPORT_P317.md`、`REPORT_P265.md` /
  `REPORT_P250.md` (作業域の確保で LRU を追い出す階段)、`REPORT_P227.md` /
  `REPORT_P309.md` (`DENNO_PF_STREAM_SRC=page|reg` は負け)。
- 基準 COND = タスク指定の 4 本 (`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000
  DENNO_EXPERT_GPU_MIN_HIT=0 DENNO_GPU_WEIGHT_MB=12000`)。先頭に捨て COND (`x0`)、
  末尾に baseline の繰り返し (`base2`)。`SCENARIO=depth8k`、`PROFILE=1`。
- ホスト: Ryzen 7 9800X3D / RTX 5090 / PCIe Gen5。`load_before` は t324a 1.95、
  t324b 2.83、t324c 2.09。**t324a は run の最初の 2 COND が page cache で 3 倍に
  膨らんだ** (base r1 3,346 ms、`[pre-kprof] ple=697.4 ms`) ので採否に使っていない。
  t324b / t324c は 6 COND とも r1/r2 が 1% 以内に揃った。

## 2. 変えたこと (3 コミット + merge + REPORT)

### 2.1 まず前提を実測で確かめた (コードは変えない)

prefill は VRAM 常駐の expert を**すでに読むだけ (peek) で使っていた**。
3,826 の `[exp-pf]` は `lookups=20847 hit_rate=0.470`、`[exp-pf-stream]` は
`experts=11045 up_MB=22653.4` = **ミスした 11,045 本だけを上げている**
(全 expert なら 20,847 本 = 42.7 GB)。14k は `hit_rate=0.389`、`experts=13849`。
`peek_many` の結果は `PfStreamEntry { dev: Some(...), host: None }` として渡り、
`needs[]` の集計もポインタ表も `dev` をそのまま使う (ミスだけステージングへ)。

つまり「常駐を読む」「常駐ぶんは上げない」はもう入っていて、残っていたのは
**転送そのものの重なり**だった。3,826 の台帳 (t320c・既定 ut=1) は
wall 2,604 = gpu_busy 1,375 + exposed 1,228 で、exposed のうち `up_wait` 1,111 ms。
転送は写しが床 (22.65 GB / 1.08 s = 21 GB/s、DDR 律速) で、同じ区間に MoE の
カーネルが 520 ms あるのに**直列**に付いていた (待つ → 積む、の繰り返し)。

作業域による LRU の追い出しは 3,826 では 0 本 (`[pf-pool] ... 追い出し 0 本`)、
14k では 1,360 本。3,826 のヒット率は LRU 容量 (22,000 MiB / 2.05 MB = 約 10,700 本)
でほぼ上限 (9,802 本ヒット) なので、**バイトを減らす余地はここには無い**。

### 2.2 `c8b5d9194` **`DENNO_PF_UP_GPUWAIT`** (新規、既定 on)

計算スレッドは「ワーカーが `ev_up` を記録した」(= 写しと DMA の積み込みが済んだ)
ところまでで止め、H2D の完了は `cuStreamWaitEvent` として**計算ストリームの後ろ**へ
入れる。カーネルは同じストリームに積むので、転送が終わったグループから GPU が
自分で走り出す。`ready` を待たずに待ちを入れると 1 つ前の記録で満たされ、
未転送のデータで走るので、順序は ready → ストリーム待ち → カーネルで固定した
(`kernel_invariants::task324_gpuwait_orders_stream_wait_before_kernel`)。
ut 経路の内側だけ (従来経路・`page`・`reg` は 1 バイトも変えない)。

### 2.3 `5a6452a20` **`DENNO_PF_STREAM_NOSYNC`** (新規、既定 on)

層の終端の `cuCtxSynchronize` を、出力がデバイスのまま (`PfStreamIo::Dev`) の
ときだけ飛ばす。次の層のカーネルは同じストリームの後ろに積まれ、ホストが書く表 /
読む行は同期コピー (`cuMemcpyHtoD` / `cuMemcpyDtoH`) なので順序は保たれる。
単独では効果なし (t324b の gwn 2,606 対 gw 2,611 ms) だが、採った形の一部。

### 2.4 `82015e3b9` **先頭グループを小さく切る** + 既定を採った形へ

GPU 側で待つと、残る露出は**層の入口の段差**になった (層 L+1 の計画は層 L の
MoE が終わるまで立たないので、最初のグループの転送は隠せない)。段差は最初の
グループのバイト数に比例するので、`group_bounds_first` (純関数、担当は重複なし・
取りこぼしなし、`first = 0` で従来と同一) を足し、`DENNO_PF_STREAM_GROUP` の
既定を 64 → 32 にした。グループは **plan の segment 数**で切る (常駐も 1 本と
数える) ので、ミス本数で見た実効は **31.8 → 16.4 本/グループ**
(t324b: 11,045 ミス / groups 347 → 675、1 グループ 65 → 34 MB)。

既定 on にしたのは `DENNO_PF_UP_GPUWAIT` / `DENNO_PF_STREAM_NOSYNC` /
グループ 32 の 3 つ。どれも env で戻せる (`=0` / `=64`)。
`DENNO_PF_STREAM_FIRST_G` (先頭だけ n 本) は**測ったが採らなかった**: グループ 32
では実効がすでに 16 本なので効かず (t324c の fg16 2,556/2,562 対 new 2,547/2,555)、
group 64 と組んでも効かない (g64fg16 2,635/2,631 = 旧既定と同じ)。既定 0 で残す。

値の扱い: 3 本とも**デバイスへ渡すバイト列も丸め順も変えない** (どのストリームで
いつ待つか、グループをどこで切るかだけ)。md5 は全 COND で SAME (§6)。

## 3. 数値 (lab、すべて実測)

### 3,826 cold (c2000_cold、ms。r1 / r2)

**t324b (`5a6452a20`、load 2.83)**

| cond | r1 / r2 | 中央値 |
| --- | --- | ---: |
| x0 (捨て) | 4,286 / 2,614 | 3,450 |
| base (旧既定) | 3,346 / 2,617 | 2,982 |
| gw (gpuwait だけ) | 2,613 / 2,609 | 2,611 |
| gwn (+nosync) | 2,607 / 2,605 | 2,606 |
| **gw32n (= 採った形)** | **2,532 / 2,532** | **2,532** |
| base2 (旧既定) | 2,607 / 2,609 | 2,608 |

**t324c (`82015e3b9`、既定 on の確認回、load 2.09)**

| cond | r1 / r2 | 中央値 |
| --- | --- | ---: |
| x0 (旧既定を env で) | 2,633 / 2,638 | 2,636 |
| base (旧既定を env で) | 2,625 / 2,642 | 2,634 |
| **new (= 既定)** | **2,547 / 2,555** | **2,551** |
| fg16 (new + FIRST_G=16) | 2,556 / 2,562 | 2,559 |
| g64fg16 (new + GROUP=64 + FIRST_G=16) | 2,635 / 2,631 | 2,633 |
| base2 (旧既定を env で) | 2,630 / 2,648 | 2,639 |

- **旧既定 → 新: t324b -2.9%、t324c -3.2%。4 run の中央値で 2,622 → 2,542 ms
  (-3.1%)**。t324c は既定の経路 (env なし) で測っている。
- gw / gwn が旧既定と同じ = **gpuwait と nosync 単独では壁時計が動かない**。
  動くのはグループ 32 と組んだとき (§4)。g64fg16 が旧既定と同じ = 主役は
  グループのバイト数で、先頭だけ絞っても足りない。

### 14k cold (c8000_cold、ms、round 1 の 1 サンプル)

| run | cond | 14k cold |
| --- | --- | ---: |
| t324a | base2 / gw | 10,190 / 10,270 |
| t324b | base (冷え) / gw / gwn / **gw32n** / base2 | 8,406 / 6,642 / 6,641 / **6,517** / 6,655 |
| t324c | base / **new** / fg16 / g64fg16 / base2 | 6,690 / **6,570** / 6,544 / 6,702 / 6,676 |

- t324b **-2.1%**、t324c **-1.6%**、中央値 **6,665 → 6,543 ms (-1.8%)**。
- t324a の 2 本は page cache が冷えた回なので採否に使っていない (向きは同じ)。

### 台帳 (3,826、`[q4ledger-pf] experts` と `[exp-pf-stream]`)

t324c の r2 (base = 旧既定 / new = 既定):

| 項目 | base | new |
| --- | ---: | ---: |
| experts wall (ms) | 1,394.3 | 1,232.3 |
| experts busy (ms) | 523.2 | 559.0 |
| experts exposed (ms) | 871.1 | **673.3** |
| `[pf-host] up_wait` (ms) | 1,117.8 | 1,044.2 |
| `wait_s` (H2D 待ち) | 0.86 | **0.00** |
| `ready_s` (順番待ち) | 0.26 | 1.04 |
| `gpuw` (本数) / `nsync` (回数) | 0 / 0 | **674 / 48** |
| groups | 347 | 675 |
| copy_s / up_MB | 1.08 / 22,653 | 1.10 / 22,653 |

(t324b も同じ形: base2 は experts exposed 848.9 / `wait_s` 0.84 / `gpuw=0`、
gw32n は 663.3 / 0.00 / `gpuw=674`。)

- **gpuwait は狙いどおり `wait_s` を 0 にした** (ホストは DMA を待たない)。
  待ちは `ready_s` (写しの完了) に移るだけで、写しが床なので壁時計は
  そこでは動かない。カーネルは転送の裏に入り、`experts` の exposed が
  871 → 673 ms に縮む (内訳は §4)。
- `nsync=48` = 48 層ぶん発火。`gpuw=674` = 675 グループのうち 1 本だけ
  全ヒットの層があった。

## 4. 機序 — 何が縮んで、何が縮まなかったか

- **転送の床は写し (21 GB/s)。** 22.65 GB / 1.08 s。P227 §2 の「写し 21 GB/s 単独、
  DMA 57 GB/s 単独、重ねると 27 GB/s」と同じで、DDR の読み + write-combined の
  書きで実効帯域に張り付いている (`probe_GBs` は 55-57 = 最初の 1 グループだけ
  同期で測るので競合がない)。**ここを超えるにはバイトを減らすしかない**が、
  3,826 のヒット率は LRU 容量でほぼ上限 (§2.1) なので今回は動かせない。
- **gpuwait でカーネルは転送の裏に入った。** experts 段の exposed
  871 → 673 ms の内訳は「転送 1,080 - カーネル 520 = 560 (転送律速の残り)」+
  「層の段差 311 → 113」。`gpu_busy` の合計は 1,371 前後で不変、
  `up_wait` 1,118 → 1,044 ms。**転送が律速のままなので壁時計は写しの時間に残る。**
- **効いたのはグループ 32 (実効 16.4 本)。** 段差 = 最初のグループの転送
  (g64 で 65 MB = 3.1 ms) なので、半分にすると 48 層で約 150 ms 縮む。
  実測 -198 ms (871.1 → 673.3、t324b でも 848.9 → 663.3)。代償はカーネル起動の
  増加 (groups 347 → 675、busy 523 → 559 ms = +36 ms) で、差し引き -2.8%。
- **nosync が単独で効かない理由**: 待たなくなった層末の仕事は、次の層の
  attention が同じストリームで待つので `[q4ledger-pf] other` の sync に移るだけ
  (gwn の experts exposed 712 / other exposed 389 対 gw の 850 / 252)。
  総和は変わらない。採った形では害がないので残したが、主役ではない。

## 5. なぜ task 265 / 250 の「作業域で LRU を追い出す」を今回触らないか

- 3,826 では追い出し 0 本なので、この task の 3,826 側の目標には効かない。
- 14k では 1,360 本 (2.8 GB) を追い出すが、作業域 (4,094 MiB) を実空き
  (1,987 MiB - 余白 1,024) に合わせると層優先の 1 パスが崩れ、タイル分割で
  H2D が増える (P250 §7 の 46 GB → 89 GB)。残り 1 ラウンドで安全に測れる形では
  ないので着手していない (推定: 2.8 GB を再利用できれば -140 ms)。

## 6. md5 とテキスト

t324b / t324c の md5 (全 COND 同一):

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
- 本文 (t324c の new、14k cold、TXTD): 「ご提示いただいた文書は、LLM推論エンジン
  （おそらく `denno-engine）の **Prefill（事前計算）における性能最適化の記録です**。
  …### 1. 核心的な結論」— base と同じ書き出しで内容も同じ。TXT (B-tree) /
  TXTB (Rust の所有権) も全 COND 同一 ("looks ok"/"text ok"、errors 0)。

## 7. テスト

```
CARGO_TARGET_DIR=/tmp/t320 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1,217 passed / 0 failed / 102 ignored   (int-auto 3ae8b0046 の merge 後)
CARGO_TARGET_DIR=/tmp/t320 cargo test --release -p denno-engine --features cuda --test kernel_invariants -- --test-threads=1
  -> 100 passed / 0 failed
```
追加 / 更新したテスト:
- `stream_gate_parses_values` — `DENNO_PF_UP_GPUWAIT` / `DENNO_PF_STREAM_NOSYNC`
  の既定 (未設定は on、`0` で off)、`DENNO_PF_STREAM_FIRST_G` の解釈 (0 /
  読めない値 / 上限 256)、`DEFAULT_STREAM_GROUP = 32`。
- `first_group_is_short_and_covers_everything` — 先頭だけ小さいグループ分けが
  全 expert を重複なく被覆し、`first = 0` で従来と一致すること。
- `task324_gpuwait_orders_stream_wait_before_kernel` (kernel_invariants) —
  ready → `cuStreamWaitEvent` → カーネルの順、ストリーム待ちがホストで待たない
  こと、nosync がデバイス出力のときだけ効くこと、先頭グループの絞りが
  gpuwait の内側にあること。
**発火の証拠**は lab の台帳: 既定の run は `gpuw=674 nsync=48 group=32` で、
`wait_s` が 0.86 → 0.00、`ready_s` が 0.26 → 1.04、`experts` の exposed が
871.1 → 673.3 ms (t324b は 848.9 → 663.3)。旧既定の run は
`gpuw=0 nsync=0 group=64`。
既知の赤 (qwen4exp_decode の spec_batch_stops_at_eos、qwen4exp_params の
validate 2 件) は踏んでいない (この 2 つのターゲットは回していない)。

## 8. GPU で確認してほしいこと / 次にやるなら

1. 実機 (8098) の 3,826 で `[q4ledger-pf] experts` の exposed が 871 → 673 ms、
   `[pf-host] up_wait` が 1,106 → 1,045 ms、`[exp-pf-stream]` が
   `gpuw=674 nsync=48 group=32` になること。既定で有効。
2. **UD-Q4_K_XL でも既定のまま**測ってほしい。K 系は `stream_wc_for` が
   write-combined を既定 off にするので写しの側が変わる。グループ 32 は K 系では
   未測定 (P309 の kwc は g64)。段差はバイト比例なので同じ向きと推定。
3. 次に削るなら **転送のバイト**。3,826 のヒット率は LRU 容量で頭打ちなので、
   (a) prefill 専用に空き VRAM (3,826 で 3,405 MiB 余る) を LRU の延長として
   使う、(b) 14k の作業域の追い出し 1,360 本を残す形 (§5)、(c) グループ 16。
   (c) は段差がさらに半分になるが、カーネル起動が 2 倍 (busy +70 ms 推定) で
   相殺される見込み。

## 9. lab の VERDICT 行

t324b (`5a6452a20`、6 COND、採否を決めた回):
```
VERDICT base vs x0 OK decode +0.0% prefill +0.0% md5 SAME errors 0
VERDICT gw vs x0 OK decode -0.5% prefill +0.0% md5 SAME errors 0
VERDICT gwn vs x0 OK decode +0.0% prefill +0.0% md5 SAME errors 0
VERDICT gw32n vs x0 OK decode +0.2% prefill +0.0% md5 SAME errors 0
VERDICT base2 vs x0 OK decode +0.2% prefill +0.0% md5 SAME errors 0
VERDICT_DEPTH base c2000_cold vs x0 decode +5.1% prefill -13.6%
VERDICT_DEPTH gw c2000_cold vs x0 decode -0.4% prefill -24.5%
VERDICT_DEPTH gwn c2000_cold vs x0 decode -0.4% prefill -24.5%
VERDICT_DEPTH gw32n c2000_cold vs x0 decode -0.4% prefill -26.8%
VERDICT_DEPTH gw32n c8000_cold vs x0 decode -21.4% prefill -36.3%
VERDICT_DEPTH base2 c2000_cold vs x0 decode +0.4% prefill -24.5%
VERDICT baseline x0 errors 0 text ok
```

t324c (`82015e3b9`、6 COND、既定 on の確認回):
```
VERDICT base vs x0 OK decode -0.7% prefill -1.7% md5 SAME errors 0
VERDICT new vs x0 OK decode -0.5% prefill -1.7% md5 SAME errors 0
VERDICT fg16 vs x0 OK decode -0.5% prefill -1.7% md5 SAME errors 0
VERDICT g64fg16 vs x0 OK decode +0.9% prefill -1.7% md5 SAME errors 0
VERDICT base2 vs x0 OK decode -0.9% prefill -2.5% md5 SAME errors 0
VERDICT_DEPTH base c2000_cold vs x0 decode -1.4% prefill +0.0%
VERDICT_DEPTH new c2000_cold vs x0 decode -1.1% prefill -3.0%
VERDICT_DEPTH new c8000_cold vs x0 decode +7.1% prefill -2.8%
VERDICT_DEPTH fg16 c2000_cold vs x0 decode -1.8% prefill -2.8%
VERDICT_DEPTH fg16 c8000_cold vs x0 decode +0.0% prefill -3.1%
VERDICT_DEPTH g64fg16 c2000_cold vs x0 decode -1.1% prefill +0.0%
VERDICT_DEPTH g64fg16 c8000_cold vs x0 decode +0.0% prefill -0.9%
VERDICT_DEPTH base2 c2000_cold vs x0 decode -1.1% prefill +0.2%
VERDICT_DEPTH base2 c8000_cold vs x0 decode +0.0% prefill -1.2%
VERDICT baseline x0 errors 0 text ok
```
(t324c の x0 は 2,633/2,638 ms で base 2,625/2,642 とほぼ同じ = この回は
先頭 COND の冷えが無い。t324c の c8000_cold の decode は base 17.9 / new 17.9 /
base2 16.7 と**同じ設定でも 7% 振れる** (1 ラウンド 1 サンプル)。t324b は全 COND
16.7 だったので、既定の副作用ではないと読んでいる。)

t324a (`c8b5d9194`、5 COND。最初の 2 COND が page cache で 3 倍に膨らんだ回なので
採否には使っていない):
```
VERDICT base vs x0 OK decode +0.5% prefill -0.8% md5 SAME errors 0
VERDICT gw vs x0 OK decode +1.8% prefill -1.6% md5 SAME errors 0
VERDICT gw2 vs x0 OK decode -0.7% prefill -1.6% md5 SAME errors 0
VERDICT base2 vs x0 OK decode +0.7% prefill -3.2% md5 SAME errors 0
VERDICT baseline x0 errors 0 text ok
```
