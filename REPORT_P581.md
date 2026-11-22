ADOPT: yes t581a t581b

# task 581 (dev2) — 既定モデル prefill: 残差 (`res`) の往復のホスト側を
# 専用 pinned バウンス + 1 MiB 塊の並列写しにした (`DENNO_PF_RES_MT`)。
# `res_up` が 3,826 で 47-49 -> 37-40 ms (-19%、2 ラウンド)、14k で
# 89.5 -> 72-73 ms (-19%)、写しは 11.7 -> 15.1 GB/s (このホストの memcpy
# 天井 18.8-19.9 の 80%)。壁は 3,826 で 2,125 -> 2,117-2,125 (動かない)、
# 14k は COND 位置のドリフト (±7%) の内側で読めない。decode 10.4 ms/tok
# 不変、md5 全 COND 全 6 文脈 SAME、errors 0。タスクの壁 2.5% は満たして
# いないので、オペレーター規則 (段で 2 ラウンド 2% 以上 + 他が 1% 超悪化
# しない + 本文健全) で既定 on。

branch **`br581`** = 開始時の `int-auto` **`f90016ee3`** (merge: br546)。
`br577` は **merge していない** (`f90016ee3` に既に無く、P577 の成果は
`ADOPT: no` なので必要が無かった。§1 で数値だけ引いている)。
終端で int-auto を fetch し直し、**`56998fc55` (merge: br554) まで 10 コミット
進んでいたので merge した** (両側保持、衝突なし。br554 は上げ専用スレッド
`DENNO_PF_UP_WORKERS` の話で、既定モデルでは decode/md5 不変と P554 が報告)。

数値はすべて GPU ラボの実測 (`SCENARIO=depth8k`、`PROFILE=1`、
`MODEL=flash-next-UD-IQ3_XXS`)。baseline COND = int-auto の既定 +
`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0
DENNO_GPU_WEIGHT_MB=12000`。先頭に捨て COND、末尾に base の繰り返し。

lab 名と SHA:
- **t581a** = `08d04d044` (ゲート既定 off)。COND: x0 (捨て) / b / g
  (`DENNO_PF_RES_XPROF=1`) / **m** (`DENNO_PF_RES_MT=1 DENNO_PF_RES_XPROF=1`)
  / b2、2 ラウンド交互。→ 内訳と A/B
- **t581b** = `efbc3cabf` (既定 on にした commit)。COND: x0 / b / **off**
  (`DENNO_PF_RES_MT=0`) / b2、2 ラウンド交互。→ 既定 on の確認と切り戻しの対照

## 0. 結論 (5 行)

1. **採る (`ADOPT: yes`)。** `res_up` は 3,826 で 2 ラウンド -19%、14k で
   -19% (1 サンプル) と再現した。他の段・decode・md5 は動いていない。
2. ただし**壁 (prefill の合計) は 3,826 で 0.5% ぶんしか動かない**:
   残差の往復は 3,826 の壁 2,125 ms の 2.2%、14k の 5,130 ms の 1.7% で、
   その 19% を削っても **-10 ms / -17 ms** にしかならない。**タスクの
   壁 2.5% は満たしていない。**
3. 壁の残りは GPU 時間と、P577 が特定した ssm の scan (14k で 708 ms、
   ssm busy の 54%、`threadIdx.x == 0` の直列スカラー列) が占める。
   scan は **.cu の変更**が要るので本タスクでは触れない。
4. 写しは **1 スレッド 11.7 -> 8 スレッド 15.1 GB/s** で、このホストの
   memcpy 天井 (574 MB の 1 スレッド 19.9 / 8 スレッド 18.8 GB/s) の 80%。
   **並列度ではもう伸びない。**
5. 次は「残差をデバイスに置いたままにする」(推定 -133 ms / 14k = 2.6%) か、
   scan のスカラー列 (推定 -560 ms)。前者は MoE の専門家がホスト側にある
   前提を崩すので構造変更 (§6)。

## 1. なぜ残差を選んだか (現在地の段プロファイル)

`int-auto` (`56998fc55`) の既定モデル。t581b の x0 (= 既定 on) の台帳から。

### 1.1 14k cold (14,018 トークン、wall 5,160.9 / gpu_busy 4,155.1)

| 段 | wall ms | この既定モデルで P500 以降に試したこと |
| --- | ---: | --- |
| ssm | 1,953.7 | **P577 が中を初めて割った** (scan 708.5 = 54% / qkv_gemm 244 / z 153 / out 153 / ba 60)。chunk x8/x16 と状態 smem 化は ±0.2% で棄却。scan の直列スカラー列が律速 = **.cu 変更が要る** |
| experts | 1,849.6 | 上げの往復 (P481)、先読み予算 (P490)、窓の捨て routing (P482)、ステージング (P557 棄却)、O_DIRECT (P546 採用)、上げ専用スレッド (P554 採用) |
| full | 772.7 | KV のゼロ埋め (P505)、KV 予約 (P561)、KV の待ちずらし (P553)。全部採用済み |
| other | 548.8 | 残差の pinned 往復 (**P565 採用** = `DENNO_PF_RES_PIN`)、`iter_top` の 2 分割 (P553)、`pf_er` は GPU 待ちと確定 (P489)。**残差のホスト側の写しだけが未計測** |
| mix | 36.0 | P569/P573 が測って閉じた (Q8_0 BM=128 +6.8%、k 2 分割 -1.9%) |

### 1.2 3,826 cold (wall 2,124.7 / gpu_busy 1,202.2)

mix 283.8 / ssm 380.9 / experts 1,322.2 / full 45.9 / other 91.8。

### 1.3 選んだ理由

P577 の次手 (scan) は .cu の変更でこのタスクでは不可。残る大きい段は
experts と full だが、どちらも P481-P561 で繰り返し測られ棄却/採用済み。
`other` の中の `res_up` だけが **P565 で「pinned へ通したのに 7.2 GB/s しか
出ない」と数字だけ残して中身 (写しと DMA の内訳) が未計測**だったので、
そこを割った。P565 も P577 も同じ結論 (壁 2.5% に届かず段で採る) を出して
いるので、この段で 2 ラウンド 2% 以上が出ればオペレーター規則で採れる。

## 2. 変えたこと

- `crates/denno-engine/src/cuda.rs` (`res_xfer` モジュール): 残差専用の
  pinned バウンス (共有バウンスのロックを取らない。育つときだけ確保 =
  3,826 の 39 MiB -> 14k の 143 MiB で 2 回) と、pageable -> pinned /
  pinned -> 宛先 Vec の写しを **1 MiB 塊の rayon 並列** (専用 8 スレッド
  プール、4 MiB 未満は直列) にする。DMA は同じ `cuMemcpyHtoD/DtoH`。
- `DENNO_PF_RES_MT` (既定 **on**、明示的な偽値だけ off)、切り戻しは `=0`。
- `DENNO_PF_RES_XPROF` (既定 **off**): 写しと DMA の ms を
  `[q4ledger-pf] resx` 行に出す計器。転送 0 回なら行を足さない。
- 値は変えない (バイト列の写しだけ) ので決定性も md5 も動かない。
  pinned が取れないときは従来の共有バウンス経路へ落ちる (値は同じ)。

## 3. 数値 (`res_up` = 残差の上げ、ms)

3,826 cold、同じ run の中で交互に:

| run | ゲート off | ゲート on | 差 |
| --- | --- | --- | --- |
| t581a r1 | b 48.6 / b2 47.0 | **m 36.9** | -22% |
| t581a r2 | b 46.9 / b2 47.0 | **m 39.4** | -16% |
| t581b r1 | off 47.1 | x0 40.2 / b 36.6 / b2 38.3 | -19% |
| t581b r2 | off 47.8 | x0 38.0 / b 38.6 / b2 38.2 | -19% |

14k cold (round 1 のみ = 1 サンプルずつ): off **89.5** / on **71.9 / 73.0 / 72.4**
(t581a は off 84.5 / 88.0 -> on 72.0)。

内訳 (t581a の m、`DENNO_PF_RES_XPROF=1`、14k、574 MB 上げ + 71 MB 下げ):
- 上げ: 写し 49.1 + DMA 19.8 -> **写し 37.9 + DMA 22.0** ms
- 下げ: DMA 6.9 + 写し 41.7 -> **6.9 + 5.3** ms (宛先 Vec の初回タッチが
  並列化で消えるため)

### 壁と decode

- 3,826 cold の壁: on 2,124.7 / 2,193.4 / 2,126.0 / 2,117.1 / 2,109.5、
  off 2,126.1 / 2,124.2。**差なし** (res_up の -10 ms は 0.5% 未満で、
  同じ COND のばらつき 0.9% の内側)。
- 14k cold の壁: t581b で x0 5,160.9 / off 5,131.9 / b2 5,085.4 = **読めない**
  (1 run の中でも COND の位置で ±7% 動く。t581a でも baseline が
  5,742.9 / 5,228.9 / 5,094.8 と 12% 開いた)。gpu_busy は 4,155.1 / 4,155.5 /
  4,155.7 と不変。
- decode: 10.4 ms/tok (`10.3` が 1 つ)、全 COND 同じ。RESULT の decode
  median も 10.3-10.4。
- errors 0、md5 は t581a / t581b の全 COND・全 6 文脈で SAME
  (`a:edaa84ad b:1938059f c2000_cold:9b28e018 c2000_turn2:8b02d5c3
  c8000_cold:672fec07 c8000_turn2:199f23b3`)。
- 本文 (t581b x0_1_a1): `# B-Trees: A Comprehensive Explanation` /
  `## 1. What is a B-Tree?` / `A **B-tree (Balanced Tree) is a self-balancing
  tree data structure that maintains sorted data and allows searches,
  sequential access, insertions, and deletions in logarithmic time.` /
  `### Key Characteristics: - **Balanced**: All leaf nodes are at the same
  depth.` — 健全。md5 SAME は「本文が同じ」の証拠でしかないので読んだ。

## 4. 採否

- タスクの Target (3,826 か 14k が 2.5% 速い) は **満たしていない** (壁は
  0.5% / 読めない)。
- オペレーター規則「段で 2 ラウンド 2% 以上 + 他が 1% 超悪化しない +
  本文健全」は満たす: `res_up` -19% (2 ラウンド 4 サンプル vs 対照 4 サンプル)、
  他の段と decode は不変、md5 SAME、errors 0。他が悪化した段は無い
  (`other` の和も 125.3 -> 107.0 / 113.7 と下がる)。
- したがって **既定 on** (`DEFAULT_PF_RES_MT = true`)。切り戻しは
  `DENNO_PF_RES_MT=0`。P565 の `DENNO_PF_RES_PIN` と同じ扱い。

## 5. テスト

- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`
  → **1405 passed / 0 failed** (追加 3 本: 並列写しが直列と同一バイト列 /
  内訳行は未使用なら出ない / 2 ゲートの真理値表。int-auto merge で +2)。
- `--test kernel_invariants` → **153 passed / 0 failed**。
- cuda 無し: `cargo build --workspace --release` ok、
  `cargo test --workspace --release -- --test-threads=1` 全スイート 0 failed。
- 分岐が発火した証拠: 真理値表テストが既定 on / `=0` で off を固定し、
  同一バイト列テストが並列経路を実際に通す (4 MiB 超を含む len で確認)。

## 6. 次にやるなら (推定、測っていない)

1. **残差をデバイスに置いたままにする。** `pf_upload_res` /
   `pf_download_res` は**チャンクごとに [nt][hc][e] をホストへ戻す**設計で、
   14k は 645 MB の往復。MoE の専門家がホスト側にあるため戻しているが、
   残差だけデバイスに残せるなら `res_up` + `iter_res` の 133 ms が消える
   = **14k で 2.6%** (推定)。専門家の入力をどう渡すか次第で、構造変更。
2. **scan の直列スカラー列** (P577 §7)。14k で推定 -560 ms (-10%) だが
   **.cu の変更**なのでタスク指定が要る。
3. `pf_teardown` が 14k で 87 ms 一定 (P565 の [pf-restore] 1,003 本 /
   1,975 MiB)。page cache の状態で 86-208 ms に振れると P565 が書いている。

## 7. GPU で確認してほしいこと

- `DENNO_PF_RES_MT=0` との対で 14k cold の `res_up` (期待 89 -> 72 ms) と
  壁 (期待 ±0.5%、読めないはず) を 2 ラウンド。
- 切り戻しが効くこと (`=0` で `res_up` が 89 ms 側へ戻る)。
- 実モデル (swift15 IQ3_S) では `res_up` のバイト数が違うので、
  このタスクの数字は既定モデルのものとして扱う。
