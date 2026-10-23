# GDN (Gated DeltaNet) — 数式・カーネル・chunked スキャン

GDN は Qwen3.5 系のハイブリッド層 (3:1) の再帰部分で、行列積の塊ではなく「1 トークンずつ状態を更新する」演算である。
この巻は、その数式をどう写したか、なぜ単発カーネルから chunked 3-pass へ移ったか、そして最終的に律速が
「演算」ではなく「ローカルメモリ帯域」だったと分かるまでの記録を時系列で並べる。
モデル統合・GGUF ローダ・出力品質の話は本巻の対象外。

## この巻で分かること

- GDN の delta rule を CPU 参照 → GPU 単発カーネル → chunked 3-pass と段階的に写した経緯と、各段の数値契約。
- prefill の GDN が演算ではなく**ローカルメモリ帯域**で律速されていたこと (s192 で試した 2 案がいずれも棄却された記録を含む)。
- beta/alpha 投影の GPU 化が 2 度失敗し (s78/s79 の不採用、s96 の削除)、最後は CPU 側のデータ配置で解けたこと。
- launch を減らす試み (s89/s90) が構造的に成立せず、「同期を減らす」方向 (P15/P45) へ目標が移った経緯。

## 前提知識

- GDN の状態 S は**シーケンス全体の関数**で、位置でスライスできない。KV のような「途中までの状態を切って再利用」が効かず、
  prefix 再利用もプロンプト境界の 1 点に限られる。
- delta rule の 1 ステップは S への**線形写像** `S ← eg·(I − βkkᵀ)·S + βkvᵀ` なので、chunk をまたぐ依存を
  「chunk 内のローカル遷移 (並列)」と「chunk 間の状態伝播 (直列)」に分解できる。これが chunked スキャンの根拠。
- 数値規律: CPU 経路は「トークン逐次 == バッチ prefill で bit 一致」。GPU の GDN は当初「bit 一致不要・決定論のみ」の契約だったが、
  s241 以降は bit 一致を主張する変更が増える。どちらの立場かは各エントリに書いた。

## 読む順序

本文は時系列 (2026-10-14 → 10-24)。急ぐ読者は P19 → s64 → s67 → s72 (デバイス常駐化の流れ) →
P15 と s192 (帯域律速の確定) → s241/s243 → P36 → P40/P43 (計測器) → P45 の順で拾うと、
「速くする対象が演算から帯域・同期へ移っていく」筋が 1 本で読める。

## 天井(理論値)と実測の差

| 項目 | 理論値または上限 | 実測 | 差の原因 |
|---|---|---|---|
| GDN pass 1 のローカルメモリ転送 (27B q4・1 層) | 推定 265 GB/層。1.8 TB/s で 147 ms/層 (換算・推定) | 138 ms/層 (P15・n=14,043・48 層の平均) | 帯域律速。推定と実測がほぼ一致するので、削るには .cu の構造変更が要る |
| GDN 1 層の実演算 | 132 GFLOP/層。q4 GEMM の実効 30 TFLOPS なら 4.4 ms/層 (換算・推定) | 138 ms/層 (同上) | 効率 ≈3%。演算には 30 倍の余裕があり、演算を速くしても効かない |
| pass 1 の 4 レーン化 (P36) | ローカルアクセス命令 1/4・stack frame 0 B・128 regs (設計値) | 9.69 → 9.88 s (27B q4・16k・4 回中央値) | occupancy が 1 block/SM まで落ちて利得を打ち消した |
| pass 1 の融合 + unroll (s241/s243) | 資料なし (削減率のみ記録) | GDN pre −5.8% (65k) → さらに −1.8% | 条件は commit 本文の「65k」だけで、モデル・run 数の記載が無い |
| 3 pass 後の `ctx.sync()` 削除 (P45) | 10〜100 ms/req = 0.03〜0.3% of 35.8 s (推定) | 未測定 | 同期の中身は 3 pass の GPU 実行そのもの。消えるのは drain の分だけ |
| GDN スキャン相 (27B q4・65,000 chars prefill) | 資料なし | 6,965 ms = 19.4%、うち `gdn_pf_sync` 6,887 ms (99%) | 同期の中身 (pass 1/2/3 の配分) は P40/P43 の計測器で分ける。実機値は未測定 |

---

## 時系列

### P19: GDN の CPU 参照と単発カーネル、chunked 3-pass まで (2026-10-14, `0bbfa79e`)

**意図** — Qwen3.5 のハイブリッド層を動かすには GDN のスキャンが要る。GDN は逐次再帰なので既存の GEMM へは流せず、
CPU 参照の時点で prefill 約 90 ms/token・decode 約 20 ms/token (0.8B・CPU トークン逐次) だった。ここが次の律速になる。

**変えたもの** — `0bbfa79e` (gdn.rs の CPU 参照) / `cb9d30bb` (lib 登録) / `feab1c8b` (`GdnCore` 分割と KV の GDN スロット)。
数式は `g = ssm_a(負値) × softplus(alpha + dt_bias)` → `S ← S·exp(g)` → `sk = Σk·S`, `d = β(v − sk)`, `S += k⊗d`, `o = Σq·S`。
q/k は L2 正規化、出力は gated RMS norm (`rmsnorm(o)·w·silu(z)`)、入力側に causal depthwise conv (d_conv=4) + SiLU。
llama.cpp の `build_delta_net_autoregressive` (逐次 AR 形式) を 1:1 で写した。

**つまずき** — 逐次 AR 形式は chunked 形式と数学等価だが丸め順が違う。したがって llama.cpp とのビット一致は要求しない
(契約は「同一入力→同一出力 + 決定論」)。GPU 化では 1 ブロック = 1 head・スレッド = S の列 (dv) という構成にして、
delta rule の全更新 (sk / 外積 / 出力) をスレッド内で完結させ、スレッド間 reduction を無くした。決定論は
「launch 構成固定・atomic / ワープ縮約なし・和は固定順序」で担保する。

**分かったこと** — 単発カーネルで y maxabs 3.353e-8・ssm maxabs 7.451e-9 (緩和枠 1e-3 に対し余裕)、conv 状態は bit 一致。
E2E は 5.11 s → 2.09 s。ただし 1 ブロック/head のトークン逐次なので長い prefill では伸びない。
そこで chunked 3-pass を追加した: K1 `dno_gdn_chunk_pre` (chunk 並列・W = ゼロ起点のローカル AR 終状態、M = 恒等行列からの遷移)、
K2 `dno_gdn_chunk_state` (chunk 直列・`Sinit[ci+1] = M[ci]·Sinit[ci] + W[ci]`)、K3 `dno_gdn_chunk_scan` (chunk 並列・chunk 内を逐次 AR)。
`n ≥ 2048` で chunked に routing し、分割は `chunk_plan(n, 2048, 1024, 16)` を CPU 参照と共有する。
単一 chunk は逐次 AR と bit 一致、複数 chunk は数学等価で丸め順のみ異なる。

**分かったこと / 残った課題** — 数値安全性の根拠は「k が L2 正規化 (‖k‖≈1)・β∈(0,1)・eg=exp(g)<1 なので各因子の
スペクトルノルム ≤1、M は縮小写像になり丸め誤差が増幅しない」こと。実機 E2E の chunked 効果は未測定
(GPU バイナリ再ビルドが要るため)。

**次に効いたこと** — 「GDN は bit 一致不要・決定論のみ」という枠が後の判断を二分する。s241/s243 は bit 一致を
主張する変更になり、s192 の 4 レーン案は「ビット一致を捨てる」必要があるとして保留された。

### GDN GQA head mapping: 規約の確定と、PTX を置き去りにした期間 (2026-10-16, `e2660908`)

**意図** — v-head `h` が参照する k/q head の写像規約を確定させる。性能の話ではなく規約の話だが、
写像は index 計算 1 行で、しかも CPU 2 箇所・GPU 3 カーネルに同じ式が入る。ここが食い違うと CPU と GPU で別の計算をする。

**変えたもの** — `e2660908` / `crates/denno-engine/src/gdn.rs` / `crates/denno-engine/kernels/gdn.cu`。
`kh = h * nk / nv` (blocked・HF `repeat_interleave` 相当) → `kh = h % nk` (strided・llama.cpp `ggml_repeat` 相当)。
GPU 側は `dno_gdn_scan` / `dno_gdn_chunk_pre` / `dno_gdn_chunk_scan` の 3 箇所。ロード時のガード
(n_v_heads が n_k_heads の整数倍でない構成は拒否。0.8B 1x・9B 2x・27B 3x) はそのまま。

**つまずき** — この時点で **PTX は再生成していない** (nvcc の無い環境だった)。つまり gdn.cu は strided、
`kernels_gdn.ptx` は blocked のまま、GPU 経路は旧規約で動く状態を一時的に作った。再生成は別コミット
(`fc78586d`) で、変更前ソースの再コンパイルがコミット済み PTX と sha256 一致することを先に確認する手順を踏んでいる。

**分かったこと** — 一致度は 9B で cos ≥ 0.99984 (中央値 0.99998)、27B で cos 0.99999 (全トークン)。
性能は未測定 (index 計算 1 行なので変化しない見込み = 見積り)。0.8B は nk == nv で両規約が同一なので影響しない。
診断 (`[LCHK]` 層別 checksum / `[GDUMP]` GDN サブステージダンプ) を env ゲート付きで恒久化した。

**次に効いたこと** — 「どのサブステージが動いたか」を層別に言えるようになり、以後の GDN 変更 (s241 以降) で
差分行の帰属を機械的に取れるようになった。GPU と PTX の規約がずれる危険も、再生成手順の定型として残った。

### s61c: `attn_gdn` を投影とスキャンに分離 (2026-10-19, `b25529d3`)

**意図** — T5 のプロファイルで prefill の `attn_gdn` が 28.5 s を占めていた。この 1 バケットに
beta/alpha の投影 (`apply_bt_rows` → P17i の GEMM) と GDN スキャン本体 (`gdn_scan_batch`: DevBuffer 確保 + H2D/D2H + sync) が
混ざっており、T5b の第一レバー (スキャンのデバイス常駐化) の効果と「投影側が既に GPU で走っているか」を切り離せなかった。

**変えたもの** — `b25529d3` / `crates/denno-engine/src/prof.rs` / `crates/denno-engine/src/qwen35.rs`。
`ATT_GDN` は親バケットとして残し、表示専用の子 `ATT_GDN_PROJ` (id `100*ATTN+2`) / `ATT_GDN_SCAN` (`+3`) を追加 (`ATT_CLS_N` 2→4)。
**数値経路・分岐は無変更**。

**つまずき** — 二重計上の罠。子から親へ積む処理を生 id (`attn_core`) にも波及させると `attn_core` が二重計上になる
(実装中に実際に踏んだ)。不変条件は `attn_core == attn_gdn + attn_full`、`attn_gdn == gdn_proj + gdn_scan` (時間・call 数とも)。

**分かったこと** — 分割したのは prefill の 3 箇所 (beta 投影・alpha 投影 → PROJ、スキャン → SCAN) だけ。
decode chain の `ATT_GDN` は beta/alpha とスキャンが 1 回の融合デバイス呼び出しで走るため分けていない
(分けるには数値経路に触る必要があり「計測のみ」の範囲を超える)。テスト +1 で 299→300 passed。
教訓: 総和行のような**共有カウンタの絶対値はテストで固定しない**。固定すべきは「細分の和が親と一致する」関係式のほう。

**次に効いたこと** — この分離で `gdn_proj` が 96 call / 25.3 s ≒ 263 ms/call と見え、s62 の
「n<64 が黙って単スレッド経路に落ちていた」欠陥に到達した (詳細は `40_prefill_perf.md`)。

### s64: GDN スキャン 1 call の内訳計測 (2026-10-19, `8e7ff24d`)

**意図** — s63 の後、prefill の残り最大項目は `d2h_state` 10,604 ms / 168 call (28%) だった。
ただし `gdn_scan` 1 call の中に「バッファ確保 + 入力 H2D → launch → `cuCtxSynchronize` → D2H + ホスト反映」が
混在しており、どこが重いかは推測するしかない状態だった。

**変えたもの** — `8e7ff24d` / `crates/denno-engine/src/cuda.rs` / `crates/denno-engine/src/prof.rs`。
`gdn_scan_impl` (n < 2048 経路・decode n==1 もここ) と `gdn_scan_chunked_impl` (n ≥ 2048 の 3-pass) の内部を
alloc / launch / sync / d2h の 4 区間に分割。区間の経過は次の区間の入口で読むので、細分の合計 ≒ 外側の `gdn_scan`。
新しい `prof::Seq` (`begin`/`lap`/`end`) は **PROFILE 無効時に `Instant::now()` すら呼ばない**素透し。

**つまずき** — 二重計上を避けるため、積む先は専用テーブル `GDN_PF_NS/CALLS` と `gdn_pf_*` id (`10*D2H+9..12`) だけに限定し、
既存の生 id・`d2h_state` / `attn_gdn` / `gdn_scan`・D2H 細分 0..8 には一切足さない。CPU で 1 リクエスト
(v6_2b・int8・in=302) を回し、GPU 専用バケットが 0 ms / 0 call、`gdn_pf_*` 行が非出力であることを確認した。

**分かったこと** — 63 ms/call の実測は GPU 非搭載の開発環境では取れていない (未測定)。実測前にコードから確定した事実は 3 つ:
(1) `gdn_scan_batch` の引数に重みのデバイスバッファが無く、`wconv`/`wa`/`wdt`/`wnorm` の 4 本を**呼び出しごとに H2D** し直している
(26 層 × 2 リクエスト = 48 call)。(2) 状態 DevBuffer の「層ループ外への再利用」は prefill では効かない — 層ループは `li` を
1 回ずつ昇順に回るので同じ (li, サイズ) の 2 回目が来ない。(3) `gdn_scan` は必ず `ctx.sync()` するためパイプラインが切れ、
前段 GEMM の完了待ちを吸い込む構造になっている。

**次に効いたこと** — s67 の `GdnScratch` (層またぎ再利用) と s72 の `y_dst` チェーンは、この (2)(3) を直接の設計根拠にしている。
「再利用できるのは n に比例するサイズ固定のスクラッチだけ」という切り分けがそのまま実装条件になった。

### s67: GDN スキャンの作業領域を層またぎで再利用 (2026-10-19, `b36df11e`)

**意図** — s64 の細分で `gdn_scan` 3.15 s の内訳は alloc 1.04 s / sync 1.09 s / d2h 0.96 s だった。alloc は 48 call で 21.7 ms/call、
中身は **16 本の `DevBuffer::from_slice`** (入力 6 + 重み 4 + chunk 用 3 + 出力 3・計 ≈180 MB) を層ごとに `cuMemAlloc` して
捨てていた分。n はリクエスト内で一定、形はモデルで固定なのに毎回作り直していた。

**変えたもの** — `b36df11e` / `crates/denno-engine/src/cuda.rs`。`GdnScratch` (thread_local・grow-only) を追加し、
chunked スキャンの 16 本を初回だけ確保して以後は `copy_from` で上書きする。`SCRATCH` (maybe_gpu_linear 用) と同じ規約。

**つまずき** — ゼロ初期化への暗黙依存を切り分ける必要があった。`w`/`m`/`sinit`/`y` は各 pass が全要素を書くのでゼロ埋め不要で、
旧実装が `from_slice(&vec![0.0; …])` で**ゼロを H2D していた分 (≈60 MB/call) も消えた**。一方 `conv_out`/`ssm_out` だけは
「最終 chunk ブロックのみ書き込み」の規約があるため従来どおりゼロで初期化する (挙動不変)。重み 4 本は層ごとに中身が変わるので
`copy_from` で差し替え (数百 KB なので転送は無視できる)。

**分かったこと** — `gdn_pf_alloc` 1,044 ms → 526 ms (残りは純粋な H2D 転送 ≈11 ms/call)、`gdn_scan` 3,126 → 2,593 ms、
`attn_gdn` 4,209 → 3,664 ms、prefill 23.7 s → 23.1 s・240 → 246 tok/s。231 passed / 0 failed、GPU アンカー 3 本 MATCH。
数値は「バッファの寿命が延びただけ」で、書き込み内容・カーネル引数・起動構成・演算順は不変。

**次に効いたこと** — 「スクラッチは再利用できるが状態はできない」という区別が、s72 の `y_dst` と P15 の同期削除で
そのまま前提になった。確保そのものを消すのではなく、寿命を層ループの外へ出すという型が定着した。

### s72: GDN の y をデバイスに残し out_proj までチェーン (2026-10-19, `4119dce4`)

**意図** — s70 で FFN をデバイス常駐化したが、GDN の注意ブロックは同じ往復を残していた。スキャンが y
(n×4096 = 47 MB) を D2H し、その直後の out_proj が同じ y を H2D して戻す。1 層あたり 94 MB、24 層で **2.2 GB/リクエスト**を
無駄に往復していた。

**変えたもの** — `4119dce4` / `crates/denno-engine/src/cuda.rs` / `crates/denno-engine/src/model/qwen35.rs`。
`gdn_scan_chunked_impl` に `y_dst: Option<u64>` を追加し、指定時はスキャンの y をチェーンスロット (`CH_PFH`) へ直接書いて
ホストへは conv/ssm 状態だけを落とす。`gdn_scan_batch_oproj` は「スキャン → out_proj (int8 GEMM) → 結果を 1 回だけ D2H」を
1 関数にしたもの。`forward_prefill` の GDN 経路は、これが成立すれば従来経路 (y を D2H → `apply_bt_rows_into`) を丸ごと飛ばす。
`DevBuffer::to_vec` のゼロ埋めもやめた (未初期化領域へ直接 D2H してから `set_len`・47 MB で数 ms)。

**つまずき** — 前提条件 (重みのアップロード・スロット確保・ctx) をすべてスキャン**前**に確認する必要がある。
conv/ssm はスキャンが更新するので、途中で失敗してフォールバックすると**状態が二重に進む**。この順序が設計の制約になった。

**分かったこと** — 同じ `f_quant` + 同じ `f_gemm` (同一 int8 累積) を通るため **bit 一致**、GPU アンカー 3 本 (1b / 2b / full 1b)
すべて MATCH。prefill 11.34 s → 10.11 s (502 → 562 tok/s)、セッション開始時の 46.2 s から −78% / 4.6 倍。

**次に効いたこと** — 「状態を更新するカーネルの後ろに別カーネルを連結するときは、前提検査を全部前へ出す」という型が、
P15/P45 の同期削除でそのまま再利用された。

### s78/s79: GDN beta/alpha 射影の GPU カーネル試作 → 不採用 (2026-10-19, `85a94052`)

**意図** — 直近のプロファイルで prefill の `gdn_proj` は 5.4 s (5,866 tok)。s77 で一度 GPU 化を試みたが、
形状チェックが `w.len()` を `k*tokens` と比べていたため常に false で CPU へ落ちる**死んだコード**で、計時も no-op クロージャだった。
s78 でカーネルから作り直し、s79 でビルドを通した。

**変えたもの** — `85a94052` (カーネル) / `e52df695` (opt-in ゲート) / `98c83cbb` (テスト)。
新カーネル `dno_matvec_rows_f32` (`crates/denno-engine/kernels/kernels_elem.cu`)。`[m,k] × [n,k]^T → [m,n]` を
1 ブロック = 1 出力 (i,j) で計算し、共有メモリ・atomic・warp 縮約を使わない (決定論の作り)。
ホスト側は `gdn_proj_shape_ok` の形状ゲート、非正規の事前検査 (`gdn_proj_min_row_sqsum`)、1 呼び出し = 1 H2D + 1 D2H、
`D2H_GDN_PROJ_{H2D,KERNEL,SYNC,D2H}` の 4 分割。**opt-in `DENNO_GPU_GDN_PROJ=1` (既定 off)** で隔離し、
既定経路は従来の `apply_bt_rows_into` (CPU・ビット不変)。

**つまずき** — 実機で 3 つ壊れていた。(1) 数値が合わない: CPU との絶対差は最大 1.0e2 級 (k=4096 nv=32 m=4096) で ulp の範囲ではない。
決定論 (同一入力 2 回でビット一致) は成立するので、計算そのものが食い違っている (grid/次元の受け渡し・重みの配置を疑う)。
(2) k=1 の形状で実行時エラー → `LMV32_BROKEN` が立ち、以降の呼び出しが全部 CPU へ落ちる。
(3) 非正規の事前検査より opt-in ゲートの判定が先に来るため、ゲート off では `GdnProjMiss::Disabled` になり
`Denorm` の観測ができない (診断の順序)。

**分かったこと** — **GPU 実機で測った結果、カーネルは正しくないため既定では使わない。** opt-in のまま隔離し、
既定 off を維持。312 passed / 0 failed / 18 ignored。GPU プローブ `cuda_gdn_proj_f32_bit_exact_vs_cpu` は名前と裏腹に
ビット一致を主張せず、決定論と偏差の記録だけを行う (名前は歴史的経緯)。副産物として、テスト 7 ファイルが
`AppState.temperature_cap` 追加に追随しておらずコンパイル不能だった既存破損を修復した (テストのみ)。

**次に効いたこと** — 同じ形状 (m が大きく k が小さい F32 投影) は GPU ではなく CPU 側のデータ配置で解く方向に切り替わり、
s66 の転置レイアウト (グループ優先 `[g][d][lane]`) で 5.0 倍になった (詳細は `40_prefill_perf.md`)。
この GPU 経路自体は s96 で削除される。

### s89/s90: GDN スキャンの 2 launch 折り畳みは成立しない (2026-10-19〜20, `ddf7201a`)

**意図** — chunked スキャンは pass 1 (W/M) → pass 2 (Sinit の直列伝播) → pass 3 (逐次 AR) の 3 launch + 3 sync。
pass 2 を pass 3 の chunk 0 に折り畳めば 2 launch になるはずだった。

**変えたもの** — `ddf7201a` / `crates/denno-engine/src/cuda.rs` / `crates/denno-engine/kernels/gdn.cu`。
`dno_gdn_chunk_scan_hoist` を実装し、chunk 境界の状態をグラフメモリ `s_rows[t]` に置く形にした。
**結論として `route_hoist` は常に false** — hoist 版を 1 回も走らせない。

**つまずき** — 理由は 2 つで、どちらも「折り畳んだ chunk ci ≥ 1 が読む行 `ci*chunk_len − 1` の書き手が同一 launch の中に居ない」
ことに帰着する。(1) **GDN 層が連続していない**: 間に full attention 層が入り、full 層は GDN の launch を出さない。
層 i の `s_rows` を読む時点で、それを書いた launch は存在しない。(2) **区間の先頭が必ずホストで切れる**: 各層のスキャン区間は
`[lo, n)` で、ホスト状態が `cache.advance(n)` のたびに区間境界と一致する。隣接区間はホスト経由でしか繋がらない。
したがって 3 launch 経路が唯一の正しい経路。帳簿と不変条件テストは残すが、**帳簿が健全であることは経路を走らせる根拠にはならない**。

**分かったこと** — 実測 (cold A/B・5,686 token・交互 3 ラウンド): OLD `b502e34` = 13.362 / 13.408 / 13.404 s、
NEW `98c83cb` = 13.345 / 13.377 / 13.375 s → **−0.03 s で利得なし** (受入条件の 0.3 s に届かない)。
アンカー 3 本は一致、decode は 16.4 ms/token で不変。残るコストの行き先も分かった: `gdn_scan` 1,987 ms のうち
1,100 ms が同期待ち (= 実際の GPU 仕事)、520 ms が活性化 H2D で、スキャンの中に残った常駐レバーは無い。

**次に効いたこと** — 目標が「launch を減らす」から「同期を減らす」へ移った。以後この分は `d2h_state`
(6,147 ms = 壁の 38.9%) として扱い、P15/P45 の同期削除に繋がる。

### s96: 却下済みの GPU gdn_proj 経路を削除 (2026-10-20, `55586f03`)

**意図** — s78/s79 で試作した prefill の GDN beta/alpha 射影の GPU 経路 (`DENNO_GPU_GDN_PROJ=1` の opt-in) は、
数値が壊れていることが s79 の実機計測で確定していた。既定 off のまま死んだコードとして残っていたので、Rust 側を削除する。

**変えたもの** — `55586f03` / `crates/denno-engine/src/cuda.rs` / `crates/denno-engine/src/model/qwen35.rs` / `crates/denno-engine/src/prof.rs`。
削除: `gdn_proj_f32_into` / `gdn_proj_opt_in` / `gdn_proj_shape_ok` / `gdn_proj_min_row_sqsum` / `gdn_proj_gpu_calls` /
`gdn_proj_cpu_denorm_rows` / `gdn_proj_last_miss`、`GdnProjMiss`、`GDN_PROJ_*` 定数、対応するテスト 3 本、
`model/qwen35.rs` の `gdn_proj_into_gpu()` と prefill の呼び出し分岐。呼び出し側は従来の CPU 経路
(`apply_bt_rows_into` → `matvec_rows_bt_into`) を直に通す。`prof.rs` は**何も消していない** — id 予約を残す
(番号を詰め直すと過去のプロファイルログとの比較が壊れる)。表は 0 で出続ける。

**つまずき** — `.cu` / `.ptx` は残した。PTX の再生成には nvcc が要るため、`.cu` と `.ptx` を同時に消さないとビルド規約が崩れる。
カーネルは他経路と共有していないが、削除は GPU 機での作業として残した (Rust 側からの参照はもう無い)。

**分かったこと** — 削除根拠の数値: k=4096 nv=32 m=33 で `out[0]` が **−17.205 対 −8.410** (CPU 参照)、最大絶対差 ~1e2。
倍近い差は丸めでは説明できず実装バグ。相対誤差も NUMERIC_POLICY の R1 (1e-6) を満たさない。
既定経路が不変である根拠: 消したのは「opt-in が立っていて、かつ GPU ゲートを全部通過したときだけ `return` する」分岐だけで、
既定では常に `apply_bt_rows_into` で計算されていた。演算順・丸め・バッファは一切変わらない (ビット不変)。
`cargo check --features cuda --tests` = exit 0、engine 242 passed / 17 ignored、server 全緑。

**次に効いたこと** — 「既定 off の opt-in は、数値が壊れていると分かった時点で消す」が規約になった。
P36 の 4 レーン版も同じ形 (opt-in・既定 off) を取ったが、こちらは数値が正しく効果が無いだけなので残している。

### P15: GDN chunk scan の pass 間ホスト待ちを削除 (2026-10-23, `c0b19d07`)

**意図** — 27B q4・65k で `sl_lay_gdn_scan` 6,613.7 ms (10.5%・48 層 = 138 ms/層)、`sl_lay_norm_in` 1,855.2 ms (2.9%・64 層 = 29 ms/層)。
指示の 2 仮説「chunk ごとに 1 launch ではないか」「ホスト順序が悪いのではないか」を確かめるのが起点。

**変えたもの** — `c0b19d07` / `crates/denno-engine/src/cuda.rs`。pass 1 → pass 2 → pass 3 の間でホストを止めない。
3 pass は同一ストリームへ発行するので、pass 2 が pass 1 の W/M を読み pass 3 が pass 2 の Sinit を読む順序は FIFO が保証する。
同期点は「ホストが結果を読む 1 点」だけにし、`DENNO_PROFILE` の細分 (`D2H_GDN_PF_*`) が 3 pass の実行時間を sync として
写し続けるよう 1 回は残した (表示専用区間が 0 になると scan がタダに見える)。カーネル・引数・grid・block・shmem・
chunk 分割・発行順・バッファは 1 つも変えない → 出力ビットは不変。

**つまずき** — 両仮説とも不成立。chunk ループは最初から全部デバイス側 (chunk = blockIdx.x) で、chunk ごとのホスト往復は 0 回。
1 層 3 launch で launch 代は 15 us = 0.01%。138 ms/層 の正体は**カーネル内のローカルメモリ帯域**だった:
pass 1 はスレッドごとに `float Wcol[256] + Mcol[256]` をローカルメモリに置き (dk が実行時引数なので 256 要素 = 2,048 B/スレッド)、
1 行ごとに読み書きする → 推定 265 GB/層。1.8 TB/s で 147 ms、実測 138 ms と一致する。

**分かったこと** — `gdn_scan` は 3 launch/層 3 sync/層 (144/req) → 3 launch/層 1 sync/層 (48/req)。
`ctx.sync()` は `cuStreamSynchronize` ではなく `cuCtxSynchronize` (コンテキスト全体の drain) なので、
デバイス全体の drain が 1 リクエストで 96 回減る。予測 (実機未計測): `sl_lay_gdn_scan` 138 → 138 ms/層
(区間はカーネル実行時間で、消えた往復は 1 層 0.1 ms 未満)、wall 63.05 s → −3〜15 ms (0.005〜0.02%)。
316 passed / 36 ignored、CPU ゲート 279 passed。同じ層の実演算は 132 GFLOP で、q4 GEMM の実効 30 TFLOPS なら 4.4 ms
= 効率 3%。ホスト側では動かない。

**次に効いたこと** — 「位相値は wall の分割でカーネル時間ではない」という読み方 (KPI §5.1) が確定し、
P40 の CUDA イベント計測の必要性がここで生まれた。実際に 138 ms/層 を削るには .cu が要る
(状態配列を shmem へ / pass 1 を W 専用・M 専用に分割して 265 GB のローカル往復を消す。見込み 138 → 10〜20 ms/層)。

### s241: pass 1 の decay と内積を 1 パスに融合 (2026-10-23, `0d9faef7`)

**意図** — s192 で pass 1 の律速がローカルメモリの「容量」ではなく**ローカルアクセス命令そのもの**と確定していた。
容量を削る 2 案 (配列を 128 要素へ縮小 / W をレジスタ 128 個・M を共有メモリ) は、それぞれ「変化なし」「13% 悪化」で棄却済み
(スタックフレーム 2,080 B → 1,056 B / 0 B、prefill 3,161 token で 9.75 / 9.77 / 11.02 s)。残る方向は命令数を減らすこと。

**変えたもの** — `0d9faef7` / `crates/denno-engine/kernels/gdn.cu` / `kernels_gdn.ptx` (再生成)。
pass 1 で decay の適用と内積を 1 パスに融合した。差分は .cu 1,945 行・PTX 1,438 行。

**つまずき** — 変更が大きく、レビューで差分を追うのが難しい。「bit 一致」はコミット見出しの主張だが、
丸め順が変わらない理由もアンカー一致の記載も本文には無い (後の P36 では「アンカー 3/3 MATCH」を明記する運用になった)。

**分かったこと** — 65k で GDN pre を **5.8% 削減**、bit 一致。条件はコミット本文の「65k」のみで、
モデル・量子化・run 数・median か min かの記載は無い (未記録)。テスト数の記載も無い。

**次に効いたこと** — 同じ融合ループが s243 の unroll 対象になった。s192 の「ビット一致を捨てれば 1 列を 4 レーンで
分担できる」案は依然保留のままで、これは P36 で opt-in として試される。

### s243: 融合ループの unroll を 8 に (2026-10-23, `8503925c`)

**意図** — s241 で融合したループはまだループ制御とアドレス計算の比率が高い。unroll でそれを償却する。

**変えたもの** — `8503925c` / `crates/denno-engine/kernels/gdn.cu` (8 行) / `kernels_gdn.ptx` (再生成)。
pass 1 の融合ループの unroll を 8 にした。レジスタは 44 → 40。

**つまずき** — unroll はレジスタ圧とのトレードオフで、増えれば occupancy を落とす。ここでは 44 → 40 regs と**減った**ので
その心配は起きなかった。s192 のレジスタ版が 255 regs で 13% 悪化したのとは逆方向の結果である。

**分かったこと** — pre をさらに **1.8% 削減**、bit 一致。条件 (モデル/量子化/run 数/median か min か) は
コミット本文に記載が無い (未記録)。

**次に効いたこと** — 記録からは不明。ただし「レジスタを増やす最適化は occupancy で相殺され得る」という s192/P36 の観測と、
ここで regs が減って素直に効いた事実が対になっており、P36 の解釈 (効果を打ち消したのは occupancy) の傍証になっている。

### P36: pass 1 の 4 レーン版 (DK=128) — 実測で効果なし (2026-10-23, `3e288aa2`)

**意図** — s192 で「1 列を 4 レーンで分担すればローカルアクセス命令が 1/4 になるが和の順序が変わる」として保留された案を、
DK=128 固定の opt-in として実装し、実測で決着させる。

**変えたもの** — `3e288aa2` / `crates/denno-engine/kernels/gdn.cu` / `kernels_gdn.ptx` (再生成済み)。
`dno_gdn_chunk_pre_l4` を追加。旧版は 1 列 = 1 スレッドで `Wcol`/`Mcol` (2,080 B) をローカルメモリに置いていたが、
新カーネルは列を 4 レーンで分担し 32 行ずつ持つので stack frame 0 B・128 regs (launch_bounds で上限) になり、
ローカルアクセス命令が 1/4 になる。opt-in `DENNO_GDN_LANES=4`、既定は旧カーネル (off)。

**つまずき** — **実測で効果なし**: 27B q4・16k・4 回中央値で `DENNO_GDN_LANES=1` (既定) 9.69 s、
`DENNO_GDN_LANES=4` 9.88 s と、むしろ微増。1 block/SM (512 スレッド) まで occupancy が落ちることが利得を打ち消している。
つまりローカルメモリ量は律速ではなかった (s192 の結論と一致)。

**分かったこと** — 内訳も確認: `gdn_scan` 相 6,965 ms のうち `gdn_pf_sync` が 6,887 ms (99%)、alloc 55 ms / launch 41 ms / d2h 8 ms。
**同期の中身 (3 カーネルの配分) は未計測**で、次の課題はホスト側のイベント計測 (`DENNO_GDN_KPROF`) と明示された。
PTX 再生成後もアンカー 3/3 MATCH。

**次に効いたこと** — P40 の CUDA イベント計測が直接の次手になった。4 レーン版は既定 off で残した
(数値は正しく効果が無いだけなので、s96 の削除対象とは性質が違う)。

### P40: chunked 3 カーネルの実時間を CUDA イベントで切り分ける (2026-10-23, `bf88ab26`)

**意図** — 27B q4・65,000 chars prefill の `sl_lay_gdn_scan` 相 = 6,965 ms (19.4%) の内訳が、`DENNO_PROFILE` の `gdn_pf_*` では
`sync` (13,774.9 ms / 2 リクエスト) までしか割れない。pass 1/2/3 の配分と「上流 `gdn_proj` GEMM の残りを sync が吸っている」
可能性 (P27 の位相帰属) が未計測のままだった。

**変えたもの** — `bf88ab26` / `crates/denno-engine/src/cuda.rs`。`DENNO_GDN_KPROF=1` のときだけ `gdn_scan_chunked_impl` に
4 本のイベント (`CU_EVENT_DEFAULT` = 計時有効) を打ち、既存の同期 (3 pass 後の 1 回) が済んだ直後に `cuEventElapsedTime` で
3 区間を読んで static な `AtomicU64` に ns で積算する。出力は dump 時と終了時に 1 行:
`[GDN_KPROF] pre=<ms> state=<ms> scan=<ms> calls=<n> total=<ms>`。

**つまずき** — `kernel_invariants` の `gdn_kernel_shared_layout_matches_host_smem` が赤だった。P36 が pass 1 の 4 レーン版
(`dno_gdn_chunk_pre_l4`) を足した時点で、同じ `sc[]` 割り当てを持つカーネルが 2 本になったため。期待値 1 → 2 に更新して
緑に戻した (`.cu` は触っていない)。**計測専用**で、launch の関数・引数・grid/block・発行順・同期位置は 1 つも変えず、
積算は表示用 static にしか書かない (数値経路・アンカー値には流れない)。`DENNO_GDN_KPROF` 未設定なら `kprof_begin()` が
`None` を返すだけでイベント生成も計測も起きない (各サイト分岐 1 回)。

**分かったこと** — 330 passed / 0 failed / 40 ignored (基準 327 + 計測 3 本)、`kernel_invariants` 11 passed、非 cuda `--lib` 286 passed。
既存の `DENNO_PROFILE` / `DENNO_PF_DBG` の出力は env 未設定時に不変。**実機の pre/state/scan の ms 配分は未測定**
(この環境は GPU 非搭載)。

**次に効いたこと** — 区間は「カーネル実行 + 発行の隙間」で pass 間のホスト側の詰まりも含む、と定義されたので、
実測で支配的な pass が判明したらその pass のレーン/チャンク分割を狙う、という次手が決まった。
P36 の 4 レーン版が効果なしと確定済みであることから「支配は pass 1 ではない可能性」も同時に記録された。

### P43: `[GDN_KPROF]` が実機で 1 行も出なかった原因 (出力点が dump 依存) (2026-10-23, `88e9e86d`)

**意図** — 実機 65k で `DENNO_PROFILE=1 DENNO_GDN_KPROF=1 DENNO_FC_KPROF=1` を付けたのに、`[FC_KPROF]` は 2 回出て
`[GDN_KPROF]` は 0 回だった。計測器が沈黙すると次の判断ができない。

**変えたもの** — `88e9e86d` / `crates/denno-engine/src/cuda.rs` / `crates/denno-engine/src/prof.rs`。
`pf_timing_read()` (prefill 末尾・最終行 D2H の後) から `gdn_kprof_report_at_prefill_end()` を呼ぶ。読むのは表示専用 static だけなので
**同期・launch・イベント生成を 1 つも足さない**。dump 経路 (`DENNO_PROFILE`) が生きているときは summary 側が同じ行を出すので
prefill 側では出さない (二重行の回避)。行は `[GDN_KPROF] pre=<ms> state=<ms> scan=<ms> calls=<n> miss=<n> total=<ms>` に拡張。

**つまずき** — P40 の仮説「pass 3 後の同期が無く NOT_READY」は**否定**された。3 pass 後の `ctx.sync()` は s173/P15 後も
`gdn_scan_chunked_impl` に残っているので、`finish()` の読み出しは完了後に行われている。実際の原因は**出力点**で、
P40 は `prof::summary()` とスレッド終了フックにしか行を出しておらず、本番起動は `DENNO_PROFILE` を unset するため
`prof::enabled()` が偽 → dump が出ず、ワーカースレッドも終了しないので終了フックも走らない。`[FC_KPROF]` は prefill 内から
`eprintln!` で直接出していたので env 1 本で出ていた (この差が観測の非対称の正体)。

**分かったこと** — `miss=<n>` は区間を読めずに捨てた call 数 (未完了・負値・イベント欠け・イベント生成失敗)。
env ON なら `calls=0` でも行が出るので「静かに何も出ない」は起こらない。337 passed (kprof_tests 3 → 5)、非 cuda `--lib` 291 passed、
`kernel_invariants` 19 passed (prefill 末尾への配線と「出力点が sync/launch/memcpy を呼ばない」をソースで固定)。
実機の pre/state/scan ms と miss の実値は未測定。

**次に効いたこと** — 「計測器は“出る場所”まで設計する」が規約になった。P45 の遅延読み (次の既存の同期へ読みを寄せ、
未完了は miss に数える) は、この miss 規則をそのまま使っている。

### P45: 3 pass 後の `ctx.sync()` をデバイス出力経路から外す (2026-10-24, `d19aa70b`)

**意図** — 3 pass 後の `ctx.sync()` は `cuCtxSynchronize` (コンテキスト全体の drain) で、1 リクエストあたり 48 回残っていた。
prefill の残差常駐経路では y をホストで読まないので、この 1 回は不要。削除した同期は 1 種類 1 箇所だけ。

**変えたもの** — `d19aa70b` / `crates/denno-engine/src/cuda.rs`。`gdn_scan_chunked_impl` の 3 pass 後の `ctx.sync()` を、
prefill 残差常駐経路 (`y_dst = Some`) からのみ外し、ホスト出力経路 (`y_dst = None`) には残した (y を `to_vec` で読む前の 1 回)。

**つまずき** — 安全根拠を 3 つ揃える必要があった。(1) 3 pass と後続の o_proj / 次の層の rmsnorm・GEMM は同一ストリーム
(legacy NULL) へ発行順に並ぶので FIFO で足りる。(2) 状態の書き戻しは同じストリームの `cuMemcpyDtoDAsync`。
(3) 次の層の beta/alpha H2D は `cuMemcpyHtoD` (ホスト同期コピー) なのでバッファ再利用の順序も保たれる。
削除した 1 回の元の理由は「3 pass の実行時間を `gdn_pf_sync` に写す」= 表示専用だった (s173/P15 が pass 間の 2 回を外したのと同じ根拠)。

**分かったこと** — 数値・発行列は不変 (kernel / 引数 / grid / block / shmem / 発行順は無変更、`.cu` / `.ptx` / PTX 再生成は未接触)。
期待削減量は **10〜100 ms/req = 0.03〜0.3% of 35.8 s (推定・実測ではない)**。`gdn_pf_sync` 13,774.9 ms (48 calls/req) の大半は
3 pass の GPU 実行そのもので (位相合計が wall の 99%・s192 で pass 1 がローカルメモリ律速と確定)、
「6.5 s が消える」ではない。削れるのは `sync − (pre+state+scan)` で、上限の目安は実測 `gdn_pf_launch` 20 ms/req。
構造的な利得はデバイス全体の drain が層ごとに消えること (並走リクエストを巻き込まない)。339 passed (+1)、
`kernel_invariants` 21 passed (+2)。副次効果として残差常駐経路の `gdn_pf_sync` / `gdn_pf_d2h` 行は 0 ms / 0 calls になる (表示のみ)。

**次に効いたこと** — 残した同期の一覧 (chain_pf_h2_join の event_sync / ホスト出力経路の `ctx.sync()` / `PfSink::Host` の D2H 待ち /
i8 経路 `gdn_scan_batch_oproj` の D2H 待ち / `DENNO_PF_OVERLAP=0` の同期 D2H) が、
「ホストが結果を読む点だけ同期する」の実装一覧として残った。GDN 側で次に効くのは帯域そのもの (P15 の見込み 138 → 10〜20 ms/層) で、
これは .cu の構造変更なので未着手。
