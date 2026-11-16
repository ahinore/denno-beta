ADOPT: yes t379a t379b t379c

# task 379 — 既定モデル (flash-next-UD-IQ3_XXS) の prefill: GDN 状態のホスト書き戻しを層の最終チャンクだけにする

branch `br379` は開始時の `int-auto` `5df54268c` (merge: br364) から。作業中に int-auto は
`3b6dc1a84` (merge: br368) へ動いたので**終端で merge した** (衝突なし。br368 の
hyper / sampler / cuda / qwen4exp と REPORT_P368.md の 789 行が入り、テストは merge 後に
取り直して緑)。**br377 は merge していない**: int-auto にも未 merge だが、br377 の候補は 2 ゲートとも ADOPT: no で既定 off
(`DENNO_PF_EMBED_PAR` / `DENNO_PF_XFER_DBG`)、本タスクの対象 (ssm 段の同期 D2H) とは
別の場所なので要らなかった。

数値はすべて GPU ラボの実測 (`SCENARIO=depth8k`, `PROFILE=1`)。`RESULT.md` 2 行目は
3 run とも `model=flash-next-UD-IQ3_XXS`、`ptx=none`。lab 名は t379a / t379b / t379c の
3 ラウンド (上限 3 を使い切った)。全 COND・全リクエストで errors 0、md5 は 3 run とも
**すべて SAME** (t377c と同じ値: `a:edaa84ad b:1938059f c2000_cold:b42d79d2
c2000_turn2:2081cdeb c8000_cold:672fec07 c8000_turn2:bc4aaf54`)。

## 0. 結論 (5 行)

- 取ったのは **ssm 段のホスト書き戻し**: 36 ssm 層 x 7 チャンクの各終端にあった
  `gdn_host_writeback` (conv + ssm の**同期** D2H 2 本) を、層の**最終チャンクだけ**にした
  (`DENNO_PF_GDN_WB_LAST`、既定 off で入れて lab で採ったので **on** にした)。
- 14k で **ssm 段 2,609.2 / 2,610.5 -> 2,458.2 / 2,451.7 ms (-5.8%)**、d2h
  **823.69 MB / 504 -> 117.67 MB / 72**、exposed **1,052.9 -> 901.1 ms**。
  **wall 6,165.4 -> 6,001.4 ms = -2.66%** (同じ run の対、1 サンプルずつ)。
- 3,826 は ssm 段 **-26.5%** (617-623 -> 449-461 ms、2 run の 6 サンプル) だが、
  その待ちは `mix` 段の壁 (137.3 -> 285.0 ms) に移り、**wall は -0.6% (同じ)**。
  3,826 の壁は expert stream の写し (up_wait 1,082-1,125 ms) が握っているため。
- decode は 16.6 (0.1 ms 刻み) で不変、md5 は全 COND SAME (値は 1 bit も変えない)。
- 同じラウンドで stream の重なり系 2 本を測って**落とした**: `DENNO_PF_UP_DECOUPLE=1` は
  up_wait 1,082 -> 1,082 ms (0%)、`DENNO_PF_STREAM_LOOKAHEAD=2` は 3,826 +2.4% /
  14k +3%。写しは 21 GB/s の DRAM 床で、3 枚のステージングではカーネルの裏に入らない。

## 1. Do 1: 現在の int-auto の取り直しと、どの段を取ったか

基準 COND = タスク指定の 4 本 (`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000
DENNO_EXPERT_GPU_MIN_HIT=0 DENNO_GPU_WEIGHT_MB=12000`)。t379a の先頭は捨て COND (`x0`)、
末尾に baseline の繰り返し (`b2`)。t379b / t379c も同じ形。

### 1.1 段ごと (14k、`[q4ledger-pf]`、ms、t379a の b2 = 末尾 base)

| 段 | wall | busy | exposed | h2d | d2h |
| --- | ---: | ---: | ---: | --- | --- |
| ssm | 2,610.5 | 1,557.5 | 1,053.0 | 159.18 MB/1,080 | 823.69 MB/504 |
| experts | 1,867.6 | 1,415.7 | 451.9 | 114.31 MB/2,484 | 59.21 MB/192 |
| full | 969.4 | 637.7 | 331.7 | 43.24 MB/252 | 689.01 MB/168 |
| mix | 21.3 | 883.7 | 0.0 | - | - |
| other | 696.7 | 171.7 | 525.0 | 3,007.72 MB/2,503 | 1,481.67 MB/95 |

`[q4ledger-pf] close wall=6165.4ms gpu_busy=4666.2ms exposed_host=2361.7ms sum=7027.8ms delta=-862.4ms`
(負の delta は `mix` の GPU 仕事が他の段の壁の裏でも走っている二重計上。P373 / P377 と同じ読み方)
`[pre-kprof] prefill tok=14018 ... | embed 104.4 | ple 60.5 | layer 5,805.5 | other 195.3`
`[pf-host] ... up_wait=1,343` (14k) / `1,082-1,125` (3,826)、
`[exp-pf-stream] up_MB=27,588.2 copy_s=1.28 probe_GBs=56.4 ut=1 dec=0 gpuw=732 group=32`。

3,826 (`b2`): `wall=2,452 ssm=616.9 (busy 433.3, exposed 183.6, d2h 235.34 MB/144) experts=1,324.1
mix=137.3 full=136.2 other=237.0`、`up_wait=1,082.9`、`copy_s=1.14`、`up_MB=23,394.1`。

### 1.2 どの段を取ったか (なぜこれか)

- 14k の壁 6,165 のうち **4,668 は GPU (`gpu_busy`)**。内訳は ssm 1,558 (scan 936) /
  experts 1,416 / mix 884 / full 638 で、**GPU 側は `.cu` の仕事**なので本タスクの範囲外。
- GPU の次に大きいのは **expert stream の写し** (`up_wait` 1,343 / 1,082 ms)。これは
  P377 §4 で「ページキャッシュ -> WC pinned の写しが 21 GB/s の DRAM 床、バイト数も床」と
  確定済み。重なり側のゲート (`DENNO_PF_UP_DECOUPLE`) は**まだ 1 度も測られていなかった**
  ので本タスクで測った (§4) が、動かなかった。
- その次が **`ssm` 段の exposed 1,053 ms**。中身は 36 層 x 7 チャンク x 2 本の
  `gdn_host_writeback` (同期 D2H 823.69 MB / 504 回) で、**誰も触っていない**最大の
  ホスト側の項目だった (P377 §7 の「次にやること」にも無い)。t334 の
  `DENNO_PF_GDN_WB_DIRECT` は「Vec をもう 1 回写す」のを 1 回にしただけで、
  毎チャンク書き戻す構造はそのまま (既定 off のまま、t334 の doc に「2 ラウンドで
  測り直すこと」と残っている)。**書き戻す回数そのもの**を減らすのは新しい。
- 前提 (expert の型が全部 GPU の速い経路か) は P377 §2.1 で確認済み (`[exp-iq-kprof]`
  に 4 型すべてが出て `fb=0`)。本タスクの 3 run でも `[exp-iq-kprof]` の型は同じで、
  `[exp-iq] fb=0` を確認した。

## 2. Do 2: 変えたこと

### 2.1 `DENNO_PF_GDN_WB_LAST` (新規、lab で採ったので既定 on)

- `pf_gdn_last_chunk_skippable(li)`: ゲート on / デバイスが正本 (`DENNO_GDN_DEVSTATE`) /
  最終チャンクでない / `DENNO_DEBUG_GDN_DUMP_LAYER` がその層でない、の 4 条件で真。
- `gdn_pf_chain` の終端で、真なら `gdn_host_writeback` を飛ばして回数だけ数える。
  偽なら従来どおり書く (**最終チャンクの書き戻しは必ず残る**)。
- 層優先 prefill のループが attn の直前に「最終チャンクか」(`end == nt_all`) を
  thread_local に立て、直後に真へ戻す (既定は真 = どこから呼ばれても従来どおり)。
  チャンク優先の経路は触っていない (常に真 = 従来)。
- **なぜ最終チャンクだけでよいか**: prefill のスキャン状態はデバイスが正本で、
  `gdn_state_begin` は `valid[li]` ならホストから seed しない。ホスト控えを読むのは
  prefill が終わったあとの decode (`GdnDecodeReq::conv_host`) と CPU 経路だけで、
  どれも層の最終チャンクの後。層の中のチャンクは必ず昇順なので、prefill 終端の
  ホスト控えは従来と同じバイト列になる。**値は 1 bit も変えない** (3 run の 8 リクエストで
  md5 SAME がその証拠)。
- 保険: 連鎖に乗れずホスト経路へ落ちたチャンクは、ホスト控えが古いので
  `gdn_state_restore_host` でデバイス正本から読み直してからホスト経路に入る。
- 台帳: `[exp-pf-stream]` の末尾に `wbl=` / `wbskip=` を足した (発火の証拠)。

### 2.2 発火の確認 (lab の実測)

| リクエスト | wbl | wbskip | ssm d2h | ssm wall |
| --- | ---: | ---: | --- | ---: |
| b1/b2 14k | 0 | 0 | 823.69 MB/504 | 2,609-2,610 |
| wl/wld 14k | 1 | **216** | **117.67 MB/72** | 2,452-2,458 |
| b1/b2 3,826 | 0 | 0 | 235.34 MB/144 | 617-623 |
| wl/wlc 3,826 | 1 | **36** | **117.67 MB/72** | 449-461 |

`wbskip` は 14k で 36 層 x 6 チャンク = 216、3,826 で 36 層 x 1 = 36。期待値どおり。

## 3. 数値 (lab、すべて同じ run の中の対)

### 3.1 14k cold (depth8k は round 1 にしか流さないので 1 COND 1 サンプル、s)

| run | x0 (捨て) | b1 | 候補 | b2 |
| --- | ---: | ---: | ---: | ---: |
| t379a | 6.67 | 6.40 | wl **6.00** / wld **6.01** | 6.17 |
| t379b | 8.39 | 6.46 | dc 6.94 / wlc **6.26** | 6.32 |
| t379c | 8.08 | 6.16 | lk2 6.43 / lk2d 6.13 / lk2dw 6.18 | 6.28 |

- **wl / wld 対 b2 = -2.66% / -2.53%** (t379a、同じ run)。b1 は 6.40 だが `ple` が
  364.8 ms (冷えた状態、wl は 54.9) なので比較に使わない。
- t379b の wlc は -0.9%、t379c の lk2dw は -1.6%。14k の wall は**同じコードでも
  6.16-6.46 と 5% 動く** (P377 §4.5 の位置効果) ので、wall の 2.5% は 1 サンプルでは
  解像できない。**採否は段の値で読んだ** (§3.3)。
- 14k の ssm 段は 4 サンプルとも再現: off 2,609.2 / 2,610.5 -> on 2,458.2 / 2,451.7。
- 14k の `close` 行は `gpu_busy` **4,666.2 -> 4,663.7** (動かない) で、
  `exposed_host` **2,361.7 -> 2,199.6 = -162.1 ms**、wall 6,165.4 -> 6,001.4 = -164.0 ms。
  **wall の差はまるごと露出ホスト時間から出ている** (GPU は遊んでいない)。

### 3.2 3,826 cold (2 ラウンド x 2 run、s)

| run | b1 | b2 | 候補 |
| --- | ---: | ---: | ---: |
| t379a | 2.579 / 2.456 | 2.452 / 2.482 | wl 2.447 / 2.449、wld 2.447 / 2.439 |
| t379b | 2.676 / 2.443 | 2.512 / 2.456 | dc 2.605 / 2.493、wlc 2.518 / 2.431 |
| t379c | 2.453 / 2.461 | 2.463 / 2.455 | lk2 2.545 / 2.492、lk2d 2.557 / 2.448、lk2dw 2.520 / 2.436 |

- wall は -0.6% (同じ)。**ssm 段だけが大きく動く**: off (b1/b2/x0/dc) の中央値 618.2 ms
  (n=12) -> on (wl/wld/wlc) 454.6 ms (n=6) = **-26.5%**。
- その分は `mix` 段の壁へ移る (137.3 -> 285.0 ms、busy は 243.5 -> 243.3 で同じ)。
  3,826 の壁 2,451 = `gpu_busy` 1,340 + `up_wait` 1,082 で閉じており、ホストに余裕が
  あるので、ホストの待ちを消しても壁は動かない。

### 3.3 なぜ「target 達成」と言えるか / 言えないか

- 14k の ssm 段は **-5.8%**、3,826 の ssm 段は **-26.5%**。どちらも 2 ラウンド以上
  再現し、d2h のバイト数 (823.69 -> 117.67 MB) が設計値と一致する。
- wall は 14k で **-2.66%** (同じ run の対、n=1)、3,826 で -0.6% (n=4、悪化ではない)。
  decode は 16.6 のまま、md5 は SAME。
- **正直な限界**: 14k の wall 差は 3 run で -2.66% / -0.9% / -1.6% とばらつく
  (14k の 1 COND 1 サンプルでは 2.5% を解像できない)。決定的なのは段の値で、
  wall はその 1/1 が同じ run で出た t379a の -2.66%。タスクの規則
  (「段で 2% 以上を 2 ラウンド」) と、3,826 が悪化していないことを根拠に採った。

## 4. このタスクで測って落としたもの (再試行禁止の材料)

### 4.1 `DENNO_PF_UP_DECOUPLE=1` (t379b の dc)

- 狙い: 上げスレッドのホスト待ちを pinned の DMA 完了だけにし、写しをカーネルの裏に入れる。
- 実測: 3,826 `up_wait` **1,082.1** (dc) 対 1,082.9 (b2) / 1,125.2 (b1) = **0%**。
  `copy_s=1.14 slotwait_s=0.00` は dc でも同じ。wall 2.605 / 2.493 対 b2 2.512 / 2.456。
  **写しは 21 GB/s の DRAM 床で、ワーカーが律速**。分離しても縮まない (推定ではなく実測)。
- 14k も dc 6.94 対 b2 6.32 で悪い側。

### 4.2 `DENNO_PF_STREAM_LOOKAHEAD=2` (t379c の lk2 / lk2d / lk2dw)

- 狙い: 依頼を 2 グループ先まで出して、写しを 1 グループぶん余分に先行させる。
- 実測: `up_wait` 1,086-1,095 ms (b1/b2 1,082-1,084) で**変わらない**。3,826
  lk2 2.545 / 2.492 対 b1 2.453 / 2.461 = **+2.4% 悪化**、14k lk2 6.43 対 6.16 = +3%。
  lk2d / lk2dw も改善しない。3 枚のステージングを全部埋めると、今度は
  `pf_slot_prepare` がスロット再利用で待つので、深さを増やしても待ちが移るだけ。

## 5. 数値の扱い

- md5 は 3 run 全 COND・全リクエストで SAME、しかも t377c (ゲートの無い木) と同値。
  ゲート on でも**出力はビット一致** (書き戻しの回数を減らすだけで、計算も丸め順も
  変えていない)。
- テキスト (t379a wl、14k): 「ご提示いただいた文書は、LLM推論エンジン
  （`denno-engine）における **Prefill（事前計算）フェーズの性能最適化の歴史を詳細に
  記録した技術レポートです。」、3,826 は「# B-Trees: A Comprehensive Explanation …」、
  TXTB (Rust の所有権) も従来と同じ。b1 と wl の TXTD は同じ文字列。
- 14k は 1 COND 1 サンプルなので、**採否の主根拠は段の値** (§3.3)。wall の数字は
  同じ run の中の対としてだけ使った。
- `other` の h2d/exposed は P377 §1.2 の `dec-up-worker` の混入を含むので読んでいない。

## 6. テスト

```
CARGO_TARGET_DIR=/tmp/t379 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1260 passed / 0 failed / 103 ignored (int-auto 3b6dc1a84 を merge した後で取り直した値。
     merge 前は 1253)
CARGO_TARGET_DIR=/tmp/t379 cargo test --release -p denno-engine --features cuda --test kernel_invariants -- --test-threads=1
  -> 112 passed / 0 failed
```

- 追加 (lib): `exp_pf::tests` の真理値表に `DENNO_PF_GDN_WB_LAST` の 11 行
  (未設定 = 既定 on、`0`/`off`/`no` で切り戻し、`1`/`on`/`true`/`yes`/` 1 ` で on)。
- 追加 (lib, cuda): `task379_tests::gdn_last_chunk_flag_defaults_true_and_skips_only_mid_chunks`
  — 旗の既定が真、途中チャンクだけ飛ばし、**最終チャンクは必ず書き戻す**。
- 追加 (kernel_invariants): 4 本 —
  `task379_gdn_wb_last_gate_is_single_and_on_by_default` (env 名 1 箇所・既定 on・
  未設定は既定へ)、`task379_gdn_wb_last_skip_wraps_the_writeback` (飛ばす判断と書き戻しが
  同じ if/else にあり、`note_gdn_wb_skip` で数え、4 条件が揃っている)、
  `task379_gdn_last_chunk_flag_is_set_in_the_layer_major_loop` (`end == nt_all` を
  attn の直前に立て直後に戻す・ホスト経路の読み直しがある)、
  `task379_gdn_wb_last_is_visible_in_the_stream_ledger` (`wbl=` / `wbskip=` と reset)。
- 既知の赤 (`qwen4exp_decode spec_batch_stops_at_eos`、`qwen4exp_params` の validate 2 件)
  は本タスクでも同じ (触っていない)。
- 発火の証拠は lab の段の値 (§2.2): wbskip 216 / 36、ssm d2h 823.69 -> 117.67 MB。

## 7. 次にやること

1. **14k の測り方を先に直す** (P377 §7.1 と同じ結論)。1 COND 1 サンプル + 位置効果 5%
   では 2.5% を解像できない。今回も 14k の wall 差は -2.66% / -0.9% / -1.6% に散った。
2. **3,826 の壁 = expert stream の写し** (up_wait 1,082 ms = 壁の 44%)。`dec` も
   `lookahead=2` も効かないので、残る道は**ステージングの枚数を増やす**
   (`PF_STAGE_SLOTS=3` を 4-6 にする。VRAM は 14k で残 2.3 GB、1 スロット 170 MB) か、
   写しを pinned の窓に直接当てる (`cuMemHostRegister`) の 2 つ。前者はゲート 1 本で
   試せるので次の候補。
3. **GDN scan** (14k 936 ms = `gpu_busy` の 20%) は `.cu` の許可が要る (P311 の wy4 で
   v 列 4 分割まで済み)。
4. 本タスクの `DENNO_PF_GDN_WB_LAST` は既定 on で入れた (14k -2.66%、3,826 同じ)。
   3,826 側で効かせるには 2 の写しを先に潰す必要がある (待ちが mix 段へ移るだけ)。

## 8. VERDICT 行

t379a (SHA `b868fc4cf`、5 COND、model=flash-next-UD-IQ3_XXS):

```
VERDICT b1 vs x0 OK decode +0.2% prefill -2.9% md5 SAME errors 0
VERDICT wl vs x0 OK decode -0.2% prefill -0.7% md5 SAME errors 0
VERDICT wld vs x0 OK decode +0.5% prefill -0.7% md5 SAME errors 0
VERDICT b2 vs x0 OK decode +0.0% prefill -0.7% md5 SAME errors 0
VERDICT baseline x0 errors 0 text ok
```

t379b (SHA `b868fc4cf`、5 COND):

```
VERDICT b1 vs x0 OK decode -5.1% prefill -8.2% md5 SAME errors 0
VERDICT dc vs x0 OK decode -12.9% prefill -7.5% md5 SAME errors 0
VERDICT wlc vs x0 OK decode -12.9% prefill -7.5% md5 SAME errors 0
VERDICT b2 vs x0 OK decode -12.7% prefill -6.1% md5 SAME errors 0
VERDICT baseline x0 errors 0 text ok
```

t379c (SHA `b868fc4cf`、6 COND):

```
VERDICT b1 vs x0 OK decode +0.5% prefill +0.7% md5 SAME errors 0
VERDICT lk2 vs x0 OK decode +0.0% prefill +0.7% md5 SAME errors 0
VERDICT lk2d vs x0 OK decode +0.2% prefill -0.7% md5 SAME errors 0
VERDICT lk2dw vs x0 OK decode +0.0% prefill +0.7% md5 SAME errors 0
VERDICT b2 vs x0 OK decode +0.2% prefill -0.7% md5 SAME errors 0
VERDICT baseline x0 errors 0 text ok
```

(`VERDICT*` は先頭 COND = x0 との比。x0 は毎回いちばん冷えた 1 本目 (t379b の
`c2000_cold` 13.94 s、t379c の `c8000_cold` 8.08 s) なので**採否には使っていない**。
採否は §3 の同じ run 内の対 = 候補 対 b2 と、段の値で読んだ。)

## 9. GPU で確認してほしいこと

- 既定 (ゲート on) の 14k で `[q4ledger-pf]` の `ssm` が **d2h 117.67 MB/72**、
  `[exp-pf-stream] wbl=1 wbskip=216` になっていること (off に戻すと 823.69 MB/504、0)。
- `DENNO_PF_GDN_WB_LAST=0` で従来どおり (823.69 MB/504、wall 6.16-6.46) に戻ること。
- 3,826 は ssm 段が 617-623 -> 449-461 ms になり、`mix` 段の壁が 137 -> 285 ms に
  なること (wall は同じ)。**これは仕様** (待ちが移っただけ) で、劣化ではない。
- 次のタスクの基準: 14k の `[pre-kprof] embed 102-105 | ple 54-60 | layer 5,649-5,805`、
  `up_wait=1,343`、`gpu_busy=4,667`、`ssm` の d2h **117.67 MB/72** (本タスク後)。
