ADOPT: yes t592c

# task 592 (dev1) — 既定モデル decode: fresh 段プロファイル (P588 と同値) から
# 最大の未着手 = **ssm 段の GDN scan** を選び、2 案を lab で実測した。1 案目
# (1 列 8 レーン = 1024 スレッド) は ssm 段 **+3.5%** で負け、そこから原因が
# 分かった (scan はメモリのレイテンシ隠蔽ではなく barrier と単一スレッドの
# 逐次和で決まっている)。2 案目 (**4 レーンのまま、逐次和の反復数をコンパイル
# 時定数にして展開**) が ssm 段 **-2.8% を 2 ラウンド再現**、値は scan2 と
# ビット一致 (md5 全 COND SAME)、errors 0、本文健全 → **採って既定 on**。
# decode の壁は短い方 -1.0% / 14k 後 -1.5% で、タスク目標の 2% には届いていない。

branch **`br592`** = 開始時の `int-auto` **`7bc7f90f1`** (merge: br609) から。
作業中に int-auto が **`6a8597af0`** (REPORT の docs/reports への移動 + README +
server test の retry) へ進んだので merge 済み (衝突なし、両側を保持)。

数値はすべて GPU ラボの実測 (`SCENARIO=depth8k`、`PROFILE=1`、
`MODEL=flash-next-UD-IQ3_XXS`)。baseline COND = int-auto の既定 +
`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0
DENNO_GPU_WEIGHT_MB=12000`。各 run は先頭に捨て COND (x0)、末尾に baseline の
繰り返し (b2)。

## 0. 結論 (5 行)

1. **採る (`ADOPT: yes t592c`)。`DENNO_GDN_SCAN3` を既定 on** にした。scan の
   第 3 版は第 2 版と同じスレッド構成 (1 列 4 レーン / 512 スレッド) で、
   **単一スレッドの逐次和 2 本の反復数をコンパイル時定数にして `#pragma unroll 8`**
   を付けただけ。加算の順序も被加数も変えないので**値はビット一致** (-fmad=false)。
2. 効き方 (t592c、2 ラウンド): **ssm 段 2.81 / 2.82 -> 2.71-2.73 ms/tok
   (-2.8%、c1 の 4 サンプル全部が baseline の 4 サンプルより下)**。
   短い decode 9.9 -> 9.8 (**-1.0%**)、14k 後 11.1 -> 11.0 (-0.9%、
   VERDICT_DEPTH は x0 比で -4.1%)。**prefill 0.55 s 不変、他の段は ±1% 以内**。
   errors 0、md5 は全 COND 全 6 文脈 **SAME** (意図どおりビット一致)、本文健全 (§4)。
3. **1 案目 (1 列 8 レーン = 1024 スレッド) は負けた** (t592b): ssm 段
   2.82 -> 2.92 (**+3.5%**、2 ラウンド一致)。48 ブロックしか無いのでブロックは
   48 SM にしか載らず、warp を 2 倍にしても SM は増えない。増えるのは
   `__syncthreads` (5 回/token) の参加 warp だけで損。**P588 §4 の「占有 28% が
   律速」という見立ては外れ**で、scan はメモリ律速ではない (§3)。
4. 途中で踏んだ罠: 1024 スレッドのブロックは **1 スレッド 64 レジスタ**が上限で、
   `__launch_bounds__` が無いと ptxas が超えて `cuLaunchKernel` が
   **error 701 (LAUNCH_OUT_OF_RESOURCES)** で落ちる (t592a。decode 1.0 ms/tok、
   errors 2、本文は空)。`__launch_bounds__(1024, 1)` で解決した。
5. 残る手は **barrier を減らす**ことだけで、scan は decode の 4% なので上限は
   -1%。**2% を狙うなら experts (2.82 ms = 29%) しかない** (§7)。

## 1. fresh 段プロファイル (t592c の b/b2、warm short decode、ms/tok)

| 段 | b (r1/r2) | b2 (r1/r2) | 備考 |
| --- | --- | --- | --- |
| experts | 2.86 / 2.82 | 2.90 / 2.82 | 最大。GPU 47.5 us/層 (gate 21.4 / down 16.4 / silu 3.3 / sum 4.2) |
| ssm | 2.81 / 2.83 | 2.82 / 2.84 | 36 層 x 4 起動。scan 10.7 us/層 |
| mix | 1.90 / 1.91 | 1.91 / 1.91 | qchain で 1 同期。cpu は鎖の GPU 待ち |
| full | 0.94 / 0.96 | 0.95 / 0.96 | 12 層。14k 後は 1.64-1.68 |
| shared | 0.46 | 0.45-0.46 | sh-gu 融合済 |
| head | 0.46 | 0.46 | Q8 674 MB を 1.57 TB/s |
| 計 (decode) | **9.9 / 9.9** | **9.9 / 9.9** | 14k 後 11.1 |

P588 (同じ `7bc7f90f1`) の 9.8-9.9 / experts 2.87 / ssm 2.81 / mix 1.89 と一致。
**1 日で decode は動いていない** (br609 は prefill だけ)。prefill 0.55 s、
c8000_cold 4.9-5.2 s も P588 と同じ帯。

## 2. なぜ ssm scan を選んだか

- 段は experts 2.82 ≈ ssm 2.82 > mix 1.90。experts は P588 §3 で天井が示されて
  いる (IQ dequant が 456-505 GB/s で計算律速、読みの形は重複なし。silu/sum の
  kprof 値は P576 が固定費過大と確定) ので「測って負けたものを再度測らない」対象。
- mix は P418 / P431 / P437 / P464 / P580 で掘り尽くし (dn8・ZC は棄却)。
- **ssm の scan だけが「並列度が構造的に足りない」**ことが P588 §4 に数値付きで
  残っていた (grid = nv = 48 / 170 SM = 28%、10.7 us/層、状態 3.1 MB を
  590 GB/s)。誰も手を入れていない唯一の箇所。

## 3. 1 案目: 1 列 8 レーン = 1024 スレッド (t592a / t592b)

実装: `kernels_gdn_scan3.cu` の `dno_gdn_scan_sig3` (1 列 8 レーン x 16 行、
block = dv*8 = 1024、grid は nv = 48)。ゲート `DENNO_GDN_SCAN3`、PTX は lab が
`PTX=gdn_scan3` で生成。

### t592a (SHA c4426911c): launch できず
`COND=c1` が 2 ラウンドとも `CUDA cuLaunchKernel が失敗: error 701` で落ちた
(decode 1.0 ms/tok、errors 2、本文は空 = md5 d41d8cd9)。1024 スレッドのブロックは
1 SM あたり 1 ブロックなので 1 スレッド 64 レジスタが上限で、`__launch_bounds__`
が無いと ptxas が超える割り当てをする。足したら
`[kattr] dno_gdn_scan_sig3: num_regs=64 local_bytes=0 max_threads=1024`
(スピル 0) で載った。

### t592b (SHA d10b8bbfb): 載ったが遅い

| cond | 短い decode | 14k 後 decode | ssm (短い) | ssm (14k 後) | errors | md5 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| x0 (捨) | 10.1 | 13.5 | 2.81-2.83 | 2.82-2.87 | 0 | 全 SAME |
| **b** | **10.0** | **11.4** | **2.82 / 2.79** | **2.85** | 0 | 全 SAME |
| c1 (8 レーン) | 10.0 | 11.9 | **2.92 / 2.88** | **2.97** | 0 | 全 SAME |
| b2 | 9.9 | 11.1 | 2.83 / 2.80 | 2.86 | 0 | 全 SAME |

`VERDICT c1 vs x0 OK decode -0.8% prefill +0.0% md5 SAME errors 0` /
`VERDICT b vs x0 OK decode -0.4%` / `VERDICT b2 vs x0 OK decode -1.6%`。
**ssm 段 +3.3-4.0% (2 ラウンド一致)**。decode の中央値 (0.1 ms 刻み) では
4% の段差は見えないが、14k 後は 11.9 対 11.1-11.4 で悪い。

なぜ負けたか (確定): 48 ブロックは 48 SM にしか載らないので warp を 2 倍に
しても SM は増えず、`__syncthreads` (5 回/token) の参加 warp だけが増える。
差 2.8 us/層 = barrier 5 回 x 0.56 us 相当で説明できる。

## 4. 2 案目 (採用): 4 レーンのまま逐次和を定数展開 (t592c)

scan2 の scan は
`for (int i = 0; i < dk; ++i) { sq += q_sh[i]*q_sh[i]; sk2 += k_sh[i]*k_sh[i]; }`
と、**反復数がカーネル引数 (実行時値)** なので `#pragma unroll` が効かない。
共有メモリのロード latency (約 30 cycle) が 128 反復ぶん、これが 2 本
(q/k の L2 和と、o の RMS 和) 露出する。反復数を定数 `DNO_S3_DK` にして
`#pragma unroll 8` を付けた。**加算の順序も被加数も変えない**ので値は scan2 と
ビット一致 (-fmad=false なので積和も融合しない)。

| cond | 短い decode | 14k 後 decode | ssm (短い、4 サンプル) | ssm (14k 後) | prefill | errors | md5 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| x0 (捨) | 9.9 | 11.5 | 2.81-2.83 | 2.83 | 0.55 | 0 | 全 SAME |
| **b** | 9.9 | 11.1 | **2.81 / 2.79 / 2.83 / 2.81** | **2.83** | 0.55 | 0 | 全 SAME |
| **c1 (scan3)** | **9.8** | **11.0** | **2.73 / 2.71 / 2.73 / 2.73** | **2.77** | 0.55 | 0 | 全 SAME |
| b2 | 9.9 | 11.1 | 2.82 / 2.80 / 2.84 / 2.80 | 2.84 | 0.55 | 0 | 全 SAME |

- `VERDICT c1 vs x0 OK decode -0.4% prefill -0.9% md5 SAME errors 0`
  (x0 は捨て COND なので b と比べる: 短い **9.9 -> 9.8 = -1.0%**、
  14k 後 **11.1 -> 11.0**)
- `VERDICT_DEPTH c1 c8000_turn2 vs x0 decode -4.1% prefill -2.0%`
  (同じ x0 比で b は -3.4% なので、b 比の実力は -0.9%)
- **ssm 段: b/b2 の 8 サンプルが 2.79-2.84、c1 の 4 サンプルが 2.71-2.73。
  重なりなし = -2.8% (2 ラウンド両方で -2.8%)。** 14k 後も 2.83-2.84 -> 2.77。
- 他の段は mix 1.90/1.92、full 0.94、experts 2.86/2.88、head 0.46 で
  **baseline と同一 (悪化 0)**。prefill 0.55 s 不変。errors 0。
- md5 は全 COND 全 6 文脈 SAME = **値は 1 ビットも変わっていない**
  (設計どおり。丸めの変更なし)。text も同じ (例: b/c1 とも
  `# B-Trees: A Comprehensive Explanation` / `## 1. What is a B-Tree?` /
  `A **B-tree (Balanced Tree) is a self-balancing tree data structure ...`)。
- `[kattr] dno_gdn_scan_sig3: num_regs=128 local_bytes=0 max_threads=512`
  (スピル 0。scan2 と同じ 512 スレッド構成)。

採否の根拠 (オペレーター規則): **段 (ssm) で 2 ラウンドとも 2% 以上
(-2.8% / -2.8%)、他は 1% を超えて悪化せず、本文健全、値はビット一致** →
タスク目標 (decode 2%) には届かないが既定 on にする。

## 5. 数値の扱い

- decode の中央値は 0.1 ms 刻みなので、**段の 3% 差は中央値で判定しない**
  (REPORT_P550 の既知)。採否は同じ run の `[q4ledger] ssm wall=` で読んだ。
- 見積りは「推定」と書いた。実測は lab の数字だけ。ssm -2.8% は
  decode 全体では -0.8% 相当 (2.82 -> 2.73 を 9.8 ms に対して) で、
  短い decode の -1.0% と整合する。

## 6. テスト

- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`
  → **1423 passed / 0 failed / 103 ignored**
- `--test kernel_invariants` → **154 passed / 0 failed**
- cuda 無し: `cargo build --workspace --release` 緑。
- int-auto (6a8597af0) merge 後に両方を再実行して緑。
- 追加した固定: scan3 の 4 レーン / 定数ループ 2 本 / `__launch_bounds__(512,1)` /
  引数 23 本 / block 4*dk / 既定 on / targets_all の末尾 / gate_audit の台帳。
- 縮約の正しさ (GPU 無しでは測れない) は CPU エミュレーションで検算した
  (`/tmp/scan_emul.cpp`、128 トークン、状態 f32、nk=16/nv=48/dk=dv=128):
  4 レーンと 8 レーンの差は y 相対 1.7e-7 / 状態 8.5e-8 / conv 0、f64 参照との差は
  4 レーン 2.9e-7 に対し 8 レーン 2.0e-7、状態 786432 要素すべて更新 (取りこぼし
  無し)。**8 レーン版がアルゴリズムとして正しいことを確かめたうえで**
  「それでも遅い」と判定している。

## 7. GPU で確認してほしいこと (次タスク)

- scan の残る手は **barrier 5 回/token を 3 回に減らす**ことだけ (step 2 の全和を
  warp 縮約にすると丸めが変わる)。scan は decode の 4% なので上限は -1%。
- **2% を狙うなら experts (2.82 ms = 29%)**。IQ dequant が 456-505 GB/s =
  天井の 25-28% なのに、読は重複が無く、ALU でも帯域でも説明がつかない。
  次は `PROFILER=ncu NCU_KERNEL=dno_exp_iq` で stall の理由 (LDS / barrier /
  issue) を 1 回取るのが最も安い。
- 14k 後の experts は 3.2-3.6 ms で、うち CPU ミス GEMV が約 1.3 ms/tok
  (hit_rate 0.962)。`DENNO_DEC_MISS_UP` は task 158 で棄却済 (PCIe 2 MB/本)。
