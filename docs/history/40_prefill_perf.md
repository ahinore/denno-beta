# 第5巻 prefill の速度改善 — 律速は計算ではなく往復とコピーだった

> [索引](README.md) / 前: [第4巻 decode の速度改善](30_decode_perf.md) / 次: [第6巻 GDN](50_gdn.md)

日付は commit の author date を JST で表す。

## この巻で分かること

- prefill の遅さは計算量ではなく、**線形層 1 呼び出しごとのホスト往復**と**戻り値コピー**だった。1.5B int8 の 13,252 token
cold は 395.8 s → 12.2 s、9B int8 の 5,686 token は 46.2 s → 10.11 s になった。
- 律速は「O(m²) の CPU attention」→「CPU GEMM」→「転送+同期構造」→「結果コピー」→「活性化の往復」へ移動した。各段で「今どこが壁か」
を測り直して進んでいる。
- 速くなる前に**計測が嘘をつく**局面が繰り返し出た: 偽のホットスポット (T5a)、黙って単スレッドに落ちる経路 (s62)、ブロック全体を指すバケット
(s84)、上書きされた計測ラベル (s87)。
- 撤回された試みを 2 つ残す: P17b の 1 発連続 pinned D2H (実測 2.2x 悪化) と T5a の opt-in ゲート (既定
2.85x 退行)。

## 前提知識

- prefill は n>1 の forward を 1 回で回す経路 (`Qwen35::forward` → `forward_prefill`)。
decode との違いは行列の行数 m がトークン数になること。
- 数値契約: int8 の積和は i32 で結合則が成立するので、タイル・スレッド数・行連結を変えてもビット一致する。attention だけは P17g で緩和済み
(決定論は維持)。
- `DENNO_PF_DBG` の位相バケットは **wall の分割**であってカーネル時間への分割ではない (P27 で確定)。本巻では位相の ms を「そのカーネルのコスト」
として読まない。
- 表記: 実測 / 換算(推定) / 未測定 を明記する。位相の % を現在値として引用しない。

## 読む順序

時系列に並んでいる。急ぐ読者は (1) 下の天井表 → (2) s63 (結果コピー) → (3) s66 / s68 (レイアウトとコピー経路) →
(4) s70 (活性化の往復) → (5) T5a 撤回 の順で、失敗と計測の話だけ先に拾うとよい。P17k は 6 ステップの積み上げだが、各段は独立して読める。

## 対象 phase 一覧

| phase | 日付 | commit | 一行要約 |
|---|---|---|---|
| P17b | 2026-10-13 | `197a06c` | int8 線形を実経路に接続し prefill の CPU 線引きを決める |
| P17i | 2026-10-13 | `f2c18315` | 長 prefill の線形層を tiled GEMM で GPU へ移す |
| P17j | 2026-10-14 | `cededce6` | O(m²) の CPU attention を flash 風カーネルへ |
| P17k | 2026-10-14 | `4bf11f5e` | FFN ブロックから 6 段階で prefill を常駐化 |
| B1 | 2026-10-15 | `455dbe0a` | rmsnorm を GPU 化し層内をデバイス常駐にする |
| 案 B | 2026-10-16 | `875dd3cd` | CPU prefill を rayon 並列化しビット一致を保つ |
| s48 | 2026-10-17 | `a2e5368c` | qwen35 の full attention をカーネル 1 発へ |
| T5a 撤回 | 2026-10-19 | `083218d6` | 偽のホットスポットから作ったゲートを revert |
| s62 / T5b | 2026-10-19 | `d9fef26c` | n<64 の投影が黙って単スレッドに落ちていたのを直す |
| s63 | 2026-10-19 | `6d34d90e` | 律速は GPU でなく戻り値コピー 21.8 s だった |
| s66 | 2026-10-19 | `ac34b1ea` | 小 n 投影の遅さは転置レイアウトの副作用だった |
| s68 | 2026-10-19 | `b682c3f5` | pinned 経由の D2H が GEMM 本体の 5 倍を食っていた |
| s69 | 2026-10-19 | `1f588b4f` | チェーン D2H の pinned 経由 + to_vec を廃止 |
| s70 | 2026-10-19 | `9ec23f3c` | FFN を常駐 chain にし活性化の往復を 4 回から 2 回へ |
| s71 | 2026-10-19 | `91b54265` | s70 が s69 を巻き戻していたのを復旧 |
| s73 | 2026-10-19 | `57405fe9` | full attention の prepare をトークン並列化 |
| s74 | 2026-10-19 | `e0ea110b` | int8 GEMM の per-thread タイルを 8x16 へ |
| s75 | 2026-10-19 | `5bca9b7e` | smem アラインメントと行カバレッジを直して復旧 |
| s80 / s81 | 2026-10-19 | `b69e0190` | gate+up をロード時に連結し 1 回の GEMM に |
| s84 | 2026-10-19 | `66698f68` | rmsnorm の「0.4 s」はブロック全体の誤称と示す |
| s87 / s88 | 2026-10-20 | `e547749d` | 位相会計を加法的にし D2H を発行点別に割る |
| P14 | 2026-10-23 | `c4c6f569` | FFN の gate/up を 1 回の GEMM に畳む |
| P20 | 2026-10-23 | `5f222949` | full 層の q を pinned ステージングへ直接書く |
| P21 | 2026-10-23 | commit なし | q4 GEMM の行タイルを投影の n から選ぶ (実装は破棄) |
| P25 | 2026-10-23 | commit なし | ffn_down の内訳を確定し何も入れないと決める |
| P27 | 2026-10-23 | commit なし | 位相会計が何を測っていないかを仕組みから監査 |
| P28 | 2026-10-23 | `3c383a7f` | full_core のホスト費用を測り無駄な確保を移す |
| P29 | 2026-10-23 | `86c44535` | 位相会計の読み方 4 点を docs に固定 |
| P30 | 2026-10-23 | `70675d32` | PF の env をサーバ起動側へ直し空振りを潰す |
| P31 | 2026-10-23 | `b5f97db8` | prefill チェーンの静的不変条件を GPU 無しで固定 |
| P34 | 2026-10-23 | `a9b91394` | smem キャスト走査器の既知の限界を検査化 |
| P36 | 2026-10-23 | `9e7de184` | 走査器がホスト起動 block 形状を取り込む |
| P39 | 2026-10-23 | `255e147f` | 走査器が grid を取り込み blockIdx 行を判定 |
| P41 | 2026-10-23 | `1068bd0d` | full_core 相を 4 区間に割る装置を入れる |
| P44 | 2026-10-23 | `c7b0991e` | V を scratch に戻さず KV キャッシュ行へ直接 D2H |
| P46 | 2026-10-24 | `2db6448d` | att 往復をチャンク化し D2H/H2D とゲートを重ねる |
| P47 | 2026-10-24 | `b44ba551` | full_qk の D2H 宛先を pinned ステージングへ |
| P48 | 2026-10-24 | `6f5f4a3c` | prepare から行ごとの中間 Vec を消す |
| P50 | 2026-10-24 | `e841db48` | tail を分解し att の H2D は既に pinned と示す |
| P51 | 2026-10-24 | `ec7d4f31` | GDN beta/alpha のホスト投影を AVX2 16 レーン化 |
| P52 | 2026-10-24 | `b6e3fa27` | q4 GEMM の per-thread タイルを 8x8 に広げる |

## 天井(理論値)と実測の差

| 項目 | 理論値または上限 | 実測 | 差の原因 |
|---|---|---|---|
| PCIe H2D | 12〜25 GB/s (s170 の置き値・推定) | 14.2 GB/s (P17k step 4 の転送ベンチ) | 帯域自体は上限近く。小口転送と同期で頭打ち |
| D2H (pinned のみ) | 14.3 GB/s (s68 の実測 = この環境の上限) | 19.5 ms / 279 MB | — |
| D2H (pinned → heap の memcpy) | 14.3 GB/s | 79.8 ms / 3.5 GB/s | memcpy が 4.7 GB/s |
| D2H (pageable 直) | 13.9 GB/s (s68) | 20.1 ms / 13.9 GB/s | 差なし = pinned 経由が二度手間だった |
| prefill の往復バイト | 41.8 GB @12,850 tok (形状からの算術・推定) → 1.7〜3.5 s = wall の 2.7〜6.2% | wall 56.3〜61.8 s (s170 時点) | 往復はもう壁ではない |
| int8 GEMM | 理論 TOPS は資料なし (s74 は「理論性能から 1 桁低い」とのみ記載) | 実効 10.6 TOPS (s74・8x8 タイル) | dp4a あたりの smem ロード過多と spill |
| GEMM 実効 (f32 換算) | 資料なし | 13.4 TFLOP/s (s63・GEMM+D2H+sync 込み) / 14.2 TFLOPS (s170 の 3 分解・推定) | 転送と同期を分母に含む |
| 位相別 TFLOP/s | 資料なし | ffn_gate 約 16 / gdn_proj 約 7.6 TFLOP/s (KPI §5 の換算・推定) | 投影を分けている分 gdn_proj が低い |
| rmsnorm | 3.1 GB の読み書き → 1 TB/s で約 3 ms・100 GB/s でも約 31 ms (s84 の見積り・推定) | 未測定 (ホストから見えない) | 判断線 0.25 s に届かず最適化せず |
| 1 発連続 pinned D2H (P17b) | 予測 −0.37 s/request (推定) | 9.78 → 22.14 s (+772 ms/層) | 3 発 → 1 発の連結で転送が直列化 |

### P17b: int8 線形層の実経路接続と「prefill は CPU のまま」の線引き (2026-10-13, `197a06c`)

**意図** — P17 で作った int8 カーネルを全 int8 線形層の実経路に接続し、e2e でビット一致と高速化を確かめる。prefill 側の関心は「GEMV 型カーネル
(1 warp/出力) を m>1 に使ってよいか」だった。

**変えたもの** — `crates/denno-engine/src/cuda.rs` の `maybe_gpu_linear` を `quant::Linear::apply_bt`
/ `apply_batch_bt` / `apply` の int8 パスにフック。モデル側のコードは変更ゼロ。重みは uid キーで VRAM 常駐、
ホスト側はピン留めバッファ + default stream の直列性で同期を 1 線形層 1 回に削った。

**つまずき** — ①重みレジストリのキーを `Quant8Linear.w` のバッファポインタにしたら、Vec の解放→再確保でアドレスが再利用され、
同一アドレスに別内容の重みが載る衝突がテストで露呈 → 構築時に uid を採番して解決。②GEMV は行数 m に比例して重みを読み直すため prefill に弱く、
**実測 m=128 で 0.39x**。→ `m > DENNO_CUDA_MAX_M` (既定 16) は CPU に残した。

**分かったこと / 残った課題** — e2e 200 tok greedy は CPU 9.08 s → GPU 2.66 s (3.4x)、decode 22.1 → 75.8 tok
/s。op 別 (600 decode step + prefill 累計) は TOTAL 28,591 → 9,686 ms で、線形層が 86% →
62% に縮み attn_core が最大 op (28.6%) になった。**prefill の線形層はこの時点で意図的に CPU** で、長 prefill の高速化は未着手。
同期削減は 1 線形層あたり約 40 µs → 約 28 µs。

**次に効いたこと** — この m 閾値が P17i の GEMM 経路 (m ≥ 256 で `dno_gemm_i8`) の土台になった。同じ「転送をまとめる」
発想を q4 系で再適用したのが 2026-10-23 の P17b 再適用 (`7e9ea37d`、full 層 q|k|v を 1 発の連続 pinned
D2H) だが、**実機で 9.78 → 22.14 s (+772 ms/層) と悪化して revert (`f55c707c`)**。事前の予測は −0.37 s/request だっ
た (`REPORT_P17B.md`)。3 発を 1 発に連結すると転送が直列化する。

### P17i: 長 prefill の int8 線形層を tiled GEMM で GPU へ (2026-10-13, `f2c18315`)

**意図** — 1.5B int8 の cold prefill が t≈2810 で約 100 s。プロファイルでは prefill の 83% が線形層の
CPU GEMM (gate_up 47 s・down 29 s) だった。

**変えたもの** — `kernels/kernels_int8.cu` に `dno_gemm_i8` (64×64 タイル・1 スレッド 4×4 出力・dp4a・A/B タイルを共有
メモリへ)。dispatch は `m > max_m() && m >= gemm_min_m()` (既定 256) で GEMM、`m <= 16` は従来
GEMV、**17〜255 は CPU のまま** (小 m では層あたりの同期が CPU GEMM より高くつく)。

**つまずき** — 先にビット一致の根拠を固めた: int8 の積和は i32 で結合則が成立するのでタイル化・順序に依存しない。f32 側は GEMV と同一の
`((dot as f32) * ascale) * scale[j]`。k が 4/32 の倍数でない場合もタイル境界の 0 パディングで処理する。テストはタイル境界
63/64/65/100/128/130 と k=1533/17911 を含む 8 形状で、**GEMV カーネルとも同入力でビット一致**を確認した。

**分かったこと / 残った課題** — 1.5B int8・longreq2 (t=2827 / max 256) で cold **約 100 s → 32.4 s**、
warm (prefix cache hit) 3.70 s、短文 (t=38・2 tok) 11.1 ms (P17h と同一 = GEMM 閾値で従来
CPU 経路)。cold prefill の応答は CPU prefill と完全一致。PTX は sha256 `9021673e…`。残る律速は O(m²) の
CPU attention に移った (次項)。

**次に効いたこと** — 「整数累積なら順序を変えてもビット一致」が、以降の GEMM タイル変更 (P17k step 6、s74 / s75) の共通前提になった。

### P17j: 長 prefill attention の GPU オフロード (flash 風 online softmax) (2026-10-14, `cededce6`)

**意図** — 13,252 token cold prefill の TOTAL 395.8 s のうち attn_core 334 s (84.5%)。
P17i で線形層を GPU に載せた後は、O(m²) の CPU attention が単独で支配していた。手順書の次候補「sampler カーネル」
は実測 0.001% で見送った。

**変えたもの** — `kernels/kernels_elem.cu` に `dno_attention_prefill`。grid=(heads,
ceil(n/PF_BQ))・block=(dh)、dynamic shmem、PF_BQ=16 / PF_BK=32、causal、exp は `dno_expf`
(glibc 再現)。K/V は P17g の GPU KV ミラー ([t_total, kv_dim] 行-major) を直接読む。決定論はタイル昇順・タイル内
kk 昇順・出力次元 d をスレッド専属にして固定 (原子操作なし)。

**つまずき** — ①exactness 契約は P17g と同じ「attention のみ緩和・決定論は維持」。CPU 参照との差は maxabs
1.49e-7。②テスト基盤の潜在フレークが顕在化した: グローバル `GPU_ATTN` フラグを差し替えるテストが並列実行で競合 → `test_lock()` で直列化。
`chain_upload` が pinned を共有するため、同期を挟まない連続 2 回 upload で DMA 完了前に上書きされる → テスト側に
`ctx.sync()` を追加 (本番経路は upload 間に download 同期が挟まるので無問題)。

**分かったこと / 残った課題** — 13,252 token cold で **wall 395.8 → 58.6 s・attn_core 334 → 11.8 s**。3,352 token
cold では TOTAL 15.8 s・attn_core 1.03 s (6.5%)。残りの律速は gate_up_proj 49.8%。dh ≥
PF_BQ・shmem ≤ 48 KB の制約外は CPU フォールバック。PTX は sha256 `c41520a9…` (2 回コンパイルして照合)。

**次に効いたこと** — 「prefill attention は 1 launch で完結するので forward 全体をやり直さずにフォールバックできる」
という設計差が、B1 / s48 の層単位フォールバックの型になった。

### P17k: prefill 線形層のデバイス常駐化 — FFN ブロックから 6 ステップ (2026-10-14, `4bf11f5e` → `35421798`)

**意図** — P17j 後の 13,252 token cold (PROFILE TOTAL 73.6 s) の残りは gate_up_proj 36.0 s /
attn_core 12.1 s / swiglu 8.2 s / down 5.2 s。しかも **prefill 中の GPU 使用率はほぼ 0%**
(スパイクのみ) で、壁時間はカーネルではなく CPU 処理 + 転送 + 同期構造が支配していた。ベンチでは m=13,252 の gate_up 1 回が 508 ms、
出力は 949 MB。

**変えたもの** — 段階ごとに commit を分けた。①`4bf11f5e`: `dno_swiglu_int` + `maybe_gpu_ffn` で
gate_up GEMM → swiglu → 量子化 → down GEMM を連結し、down 出力 81 MB だけ D2H。②`870bb871`:
`maybe_gpu_oproj` が CH_ATT を直接読み、attention の exp を `dno_expf` → `exp2f` に。③`99d29406`:
grid を (heads, 行ブロック) → (行ブロック, heads) に反転、smem 行ストライドを dh+1、PF_BQ 16→48 / PF_BK
32→8。④`67e243ad`: `dno_rope_qkv_pf` が q を rope して連続領域へ、k/v を KV ミラー行へ直書き。⑤`d2b81878`:
GEMM を 128×128 タイル・1 thread 8×8・smem ストライド 33 B に。⑥`35421798`: K/V タイルのソフトウェアパイプラインと
2r×2k クアッド dot。

**つまずき** — ①exp2f 化しても attn_core は 11.86 → 11.11 s でほぼ不変 →「expf 律速」は誤りで、残りは転送 + 同期構造だった。
②smem が ~43 KB 以上で error 700 (不正アドレス) になる謎は、**ホスト定数 `ATTN_PF_BQ/BK` が 16/32 のままカーネル
(48/8) と非同期**だったのが正体 (grid が約 4 倍過剰 + smem 不足で領域外書き)。③ソース変更後の PTX sha256 が変更前と同一で、
定数不一致のまま配備 → E2E が非決定な garbage に。同期は md5 照合を必須化した。④Qwen2.5-1.5B は QKV bias 持ちで、
当初の「bias 無しのみ」ゲートが全層フォールバックしていた → `[qb|kb|vb]` 連結 bias 加算を実装。

**分かったこと / 残った課題** — 13,252 token cold の wall は 77.9 (step 1 前) → 33.6 → 30.7 → 24.5 → 17.6 → 13
.0 → **12.2 s** (step 7 は 2 回コールド)。qkv_proj 4,511 → 0.7 ms、rope 2,754 → 0.4 ms、
down_proj 5,009 → 1,147 ms。attention カーネル単体は 368 → 112.4 → 69.3 ms/call。maxabs
vs CPU は 1.79e-7 (n=37) / 1.97e-7 (n=2056)、決定論維持。残りは attn_core 3.5 s (47%) と
rmsnorm 2.3 s。

**次に効いたこと** — 「デバイス常駐で往復を消す」型が B1 の prefill 全体チェーンに、smem バンク conflict と grid 順序の知見が
s74 / s75 の GEMM タイルに効いた。ホスト/カーネル定数の不一致は s75 の `static_assert` と静的検査テストの動機になった。

### B1: prefill の rmsnorm GPU 化と隠れ状態デバイス常駐チェーン (2026-10-15, `455dbe0a`)

**意図** — P17k 後の残りで rmsnorm が 2.3 s。しかも rmsnorm は h の H2D (81 MB/層) と直列で、層あたり
h/proj/down の往復が 81 MB × 4 = 324 MB 発生していた (13k prefill で約 9 GB/リクエスト相当)。

**変えたもの** — `forward_prefill_gpu` を新設し、prefill を「埋め込み 1 回 H2D → 層内は全部デバイス常駐 → 最終行 1 回
D2H」に再構成。`dno_add_rmsnorm` は residual 加算を融合し、in-place で CH_X に書き戻すので residual
stream は 1 本で層をまたぐ。`maybe_gpu_ffn_dev` / `maybe_gpu_oproj_dev` / `chain_download_at`
(スロット内オフセットから部分 D2H) を追加。**新規カーネル無し** (既存 PTX 再利用・sha256 不変)。

**つまずき** — 途中失敗時の整合。チェーンは all-or-nothing にして、失敗したら forward 全体を従来経路でやり直す (KV 書き込みは冪等・advance は最
後の 1 回なので安全)。非対応構造 (Qwen3 QK-Norm・一部 fp32・qkv bias が [qb|kb|vb] 揃っていない) とカーネル制約違反は事前ゲートで弾き、
GPU_BROKEN の無駄な発火を避けた。

**分かったこと / 残った課題** — rmsnorm / GEMM / rope / swiglu / residual は同一カーネル・同一演算順序なので**ビット一致**
(attention のみ P17j の緩和枠)。`cuda_prefill_chain_bit_exact_vs_cpu` で rope 後 q・K/V 行・最終
hn・最終 residual を CPU 参照とビット比較。**実機の wall 計測は未測定** — チェーン以降は op 計上が最終 sync に吸収されて意味を失うため、
wall 基準で測る必要があると記録された。

**次に効いたこと** — 「層内で完結する区間はホストに降ろさない」が、9B 側の s70 (prefill FFN チェーン) の直接の前例になった。

### 案 B: CPU prefill の rayon 並列化 (bit 一致契約を維持) (2026-10-16, `875dd3cd`)

**意図** — 大きな system+tools プロンプト (n≈2000〜2500) の prefill が 40 秒級で、GPU 使用率ほぼ 0%・CPU 1 コアだけ
100% という状態だった。原因は GPU 側ではなく CPU 側の逐次実装 (full attention の O(n²) トークン逐次と `matmul_bt_i8`) だった。

**変えたもの** — `tensor.rs::matmul_bt_i8` を rayon 化 (入力の動的量子化は行並列、GEMM 本体は j ブロック 32 チャネル並列、
出力書き込みは `SendPtr`、端数ブロックは逐次)。`qwen35.rs` は `full_attn_core` を `full_attn_prepare`
(KV 書き込み + QK-Norm + partial interleaved RoPE・トークン順に逐次) と `full_attn_attend_head`
(1 head・読み取り専用) に分割し、prefill の attend を (i, head) 並列に。rmsnorm・残差加算・SwiGLU も行並列化。

**つまずき** — `matmul_bt_i8` はコメントに「並列化」とあるだけで実際は逐次 `chunks_exact` だった (REPORT_NEXT16 の副次発見)。
並列化の条件は**出力要素ごとの累積順を変えない**こと 1 つに絞り、i/j 方向だけの並列にした。scores はタスクローカルスクラッチにして共有を避けた。

**分かったこと / 残った課題** — 9B int8・n=2480 / max_tokens 4・seed 42 で 16 スレッド **111.6 s**・1 スレッド **568.4 s
** (5.09x)、出力は同一 md5 `f9b7f6be`。CPU 参照 md5 (`feea9444` / `da0b1f57`) は 16T /
1T とも不変。演算ステージ別プロトタイプ (リポジトリ外・同一 CPU) は attention 9.2〜10.3 倍・matmul 相当 8.2〜8.9 倍・SwiGLU 5.6 倍で
、全構成で逐次版とビット一致。`.cu` / PTX は無変更。

**次に効いたこと** — 「並列化は出力要素の累積順を変えない」という基準が、s62 の小 n 経路と s73 の prepare 並列化でそのまま再利用された。

### s48: qwen35 prefill の full attention を GPU バッチ化 (2026-10-17, `a2e5368c`)

**意図** — qwen35 の `forward_prefill` は full attention を常に CPU の rayon attend で回しており、
n≈2.5k の 33.6 s のうち 13〜18 s がこの O(n²) 塊と推定されていた。

**変えたもの** — prepare (KV 書き込み + QK-Norm + RoPE・トークン逐次) はそのままに、attend を P17j の
`dno_attention_prefill` 1 発へ置き換え。qwen35 は dh=256 で動的 smem が 84.8 KB (BQ=64 /
BK=8) 必要なので、`cuDeviceGetAttribute` で opt-in 上限を取り `cuFuncSetAttribute` で 48 KB 既定を超える許可を取る。
ゲートは `n >= gemm_min_m()` (既定 256)、`DENNO_GPU_ATTN=0` で全経路を従来に戻せる。`.cu` / PTX 無変更。

**つまずき** — opt-in が取れない環境では上限 48 KB のままで、qwen35 の prefill attention は静かに CPU へ落ちる
(llama の dh=128 = 43.8 KB は影響なし)。この「静かなフォールバック」を前提に、smem 比較ゲート 4 箇所を 48 KB 固定から
opt-in 後の上限へ統一した。ゲート n≥256 により 1b (m=14) の短文は CPU attend のままで bit 不変。

**分かったこと / 残った課題** — 数値は attention のみ緩和枠 (maxabs<1e-3)。**実機の wall 計測は未測定**で、
期待効果 n≈2.5k で 33.6 s → 16〜22 s は推定。2b GPU アンカーは変化し得るため再取得が人間対応として残った。テストは `#[ignore]` の 1 本
(opt-in 成功・ゲート発動・近似一致・決定論)。

**次に効いたこと** — この opt-in 機構が、27B も dh=256 である q4 系の attention 形状の前提になった。

### T5a 撤回: 偽のホットスポットから作った opt-in ゲートが既定 2.85x 退行 (2026-10-19, `083218d6` → revert `bbe82d75`)

**意図** — T4 のプロファイルで「prefill の投影が全部 CPU (30.1 s)」に見えたため、`DENNO_GPU_PREFILL_GEMM` という
opt-in ゲートを追加した。

**変えたもの** — `083218d6` で prefill の呼び出しを `apply_bt_rows_maybe_gpu` に差し替え、ゲート未指定時は従来経路を通す設計。
**前提が偽だったので `bbe82d75` で revert** (crates は T4 の `1e51741` とバイト一致に戻る)。計測の修正
(`via` 列) だけは別 commit で残した。

**つまずき** — ①T4 の `via` 列が誤っていた: `time_full` が `on_gpu` を渡さず常に false で記帳するので、
GPU で走っていても `cpu` と表示された。②実際は `quant.rs::apply_bt` (Q8, m>1) が `maybe_gpu_linear` を呼び、
`m > max_m() && m >= gemm_min_m()` (=256) で P17i の GEMM に既に入っていた。③差し替え版は `tensor::matmul_bt_i8`
を直接呼ぶ形になり、`maybe_gpu_linear` を迂回 = **本物の CPU フォールバック**になっていた。

**分かったこと / 残った課題** — 実機 (int8 9B・5,6xx tok のキャッシュミス・固定 seed・単一インスタンス) の prefill
wall:

| build | env | prefill wall |
|---|---|---|
| `083218d6` | ゲート未指定 (既定) | 202〜203 s (3 回) / PF_BENCH 200.8 s |
| `083218d6` | `DENNO_GPU_PREFILL_GEMM=1` | 70.7 / 65.3 s / PF_BENCH 71.4 s (79.7 tok/s) |
| `1e51741` (T4) | — | PF_BENCH 70.9 s (80.2 tok/s) |

既定が **2.85x の退行**で、ゲート on は旧挙動の再現に過ぎなかった。ゲート on のアンカー 3 本は MATCH = GPU GEMM は本当にビット一致。
s61 の後処理で `via` 列を実測化 (`time_n_ev` + `cuda::Probe` の OR 集積)、`tok/call` の分母を
`CALLS_ROWS` に修正し、再発防止テスト 4 本を追加 (295 → 299 passed)。

**次に効いたこと** — 「プロファイルの列を根拠に最適化する前に、その列が実測か主張かを確認する」。この教訓が s62 (静かな単スレッド落ち) と
s84 (バケットがブロック全体) の見直しに直結した。

### s62 / T5b: n<64 の投影が黙って単スレッド経路に落ちていた (2026-10-19, `d9fef26c`)

**意図** — s61c の分離計測で prefill の `gdn_proj` (beta/alpha 投影) が 96 call / 25.3 s ≒ 263
ms/call。k=4096 × n=32 の GEMM でこの値はチューニングの域を超えている。T5a の教訓に従い、変更前に経路を確定させた。

**変えたもの** — `tensor.rs` に `matvec_rows_bt_small_n` を追加。出力チャネル方向を rayon で並列化し、
各タスクは担当する重み行 (`JG=2`) だけを走査、バッチ (i) 方向を `MB=8` レーンの SIMD で回す。各 (i,j) は d について厳密に逐次の
f32 累積のまま。

**つまずき** — 経路の実体: `beta_w`/`alpha_w` は `QuantMode::Fp32` 固定 (int8 にすると delta-rule のゲートが壊れる) なの
で `Linear::apply_bt_rows` の Fp32 分岐 → `tensor::matvec_rows_bt` を通り、**GPU seam には一切到達しない**。
「投影は既に GPU」は qkv/gate/out には当てはまるが beta/alpha には当てはまらなかった。根因は `matvec_rows_bt_vec` が出力チャネルを
`JB=64` 幅で `par_chunks_exact` 分割しており、n=32 < 64 では**0 チャンク**を返して並列本体が一度も走らないこと。
全チャネルが単スレッドの端数経路 (j-outer/i-inner で 745 MB をストライド走査) に落ち、実測 **2.8 GFLOPS**。何も失敗せず正しいが遅いだけなので長く
隠れた。

**分かったこと / 残った課題** — CPU 9B・同一形状 (k=4096 n=32 m=2843) で `gdn_proj` **263 ms/call → 0.73
ms/call (7.2 倍)**、PROFILE の `gdn_proj` 1,514.9 ms / 288 call → 211.2 ms / 288 call。
アンカー 3 本の md5 は変更前ビルドと完全一致。テストは n=1..63 のビット一致を固定 (+1 → 301 passed)。**既存テストが
n≥16 しか使っておらず、この 0 チャンク条件に一度も当たっていなかった**ことが見逃しの直接原因。GPU 化は fp32 GEMM カーネル新設が必要として提案のみ。

**次に効いたこと** — s66 で、この小 n 経路の転置レイアウトが次の律速だと分かる。

### s63: prefill の律速は GPU ではなく「結果コピー」だった (2026-10-19, `6d34d90e`)

**意図** — 依頼は全体プロファイル。計測すると 9B int8・5,686 token の prefill 46.2 s のうち **GPU 演算は 5.9 s
(12.7%)** しかなく、最大項目は `linear_gemm` の戻り値コピー (`to_vec`) で 21.8 s (47%) だった。

**変えたもの** — `tensor.rs` に `matvec_rows_bt_into` / `matvec_bt_into` / `matvec_bt_i8_into` /
`matvec_rows_bt_i8_into` / `matmul_bt_i8_into` を追加し、`cuda.rs::linear_gemm_into` /
`maybe_gpu_linear_into`、`quant.rs::Linear::apply_bt_rows_into` へ `_into` を貫通。
`qwen35.rs` の prefill は 16 本の作業バッファを層ループの外で 1 回だけ確保。

**つまずき** — `full_attn_attend_head` は出力を `*a += pv * b` で加算する。旧実装は層ごとに `vec![0.0]` を確保していたので暗黙に
ゼロ初期化されていたが、領域を使い回すと前層の値が残る。`qwen38_27b_shape_prefill_batch_bit_exact` が落ち、**バッファを 1 本ずつ戻す
bisect (6 ビルド)** で `att` と特定。修正は `att` の並列 fill。**「新しい Vec はゼロ」に暗黙依存している箇所は、
再利用に変えると必ず壊れる**。

**分かったこと / 残った課題** — 計測は 4 系統 (PF バケット・`linear_gemm` の 6 分割一時パッチ・0.5 s サンプリング・プロセス
CPU 時間) を突き合わせた。5,686 token warm で: 結果コピー 21.8 s (188 ms/呼び出し・n=12288・279 MB の新規
Vec・実効 1.48 GB/s・page fault 68k/呼び出し)、GPU 実働 5.9 s (78.5 TFLOP / 13.4 TFLOP/s)、
入力コピー 1.7 s (13.7 GB/s)、H2D/量子化/起動/D2H 起動 0.018 s (400 呼び出し合計)。SM 稼働率 11.6%、
プロセス CPU は 88.3 s / wall 45.5 s = 1.94 コア。修正後 **prefill 46.2 → 28.1 s (123 → 202 tok/s)**、
結果コピーは約 2.5 s。decode 300 token は 10.05 s (33.6 ms/token) で**変化なし** (decode は
`_into` を通らない)。

**次に効いたこと** — 修正後の内訳で最大が `d2h_state` になり、s67 の GDN 作業領域再利用と s68 のコピー経路短縮へ進んだ。
SM 稼働率 11.6%・1.94 コアという数字は「GPU は遊んでいる」の根拠として以後繰り返し使われた。

### s66: 小 n 投影の本体は転置レイアウトの副作用だった (2026-10-19, `ac34b1ea`)

**意図** — s63 / s64 / s65 の計測で `gdn_proj` (β/α の F32 投影・96 call) が 1 call 56 ms・0.75 GFLOP = **13 GFLOPS
**。SIMD を使っているのに単コア級しか出ていなかった。

**変えたもの** — `matvec_rows_bt_small_n_impl` の転置先を `at[d * mp + i]` から**グループ優先
`[g][d][lane]`** (`at[(g*k + d)*MB + l] = a[(g*MB + l)*k + d]`) に変更。転置自体もグループ単位に
rayon で並列化 (書き込み域は互いに素)。

**つまずき** — 8 レーン束ねのため、グループ g が読むのは `at[g*8 + d*mp]` = **d 方向に mp (=バッチ全長) 間隔で散らばる 8
float**。1 グループが `k*mp` (2843×4096×4 ≈ 46 MB) 全域を 64 B ラインあたり 8 float だけ使って歩き、
それが mp/8 = 356 グループ繰り返される。GPU ではなく CPU のキャッシュ/プリフェッチを壊していた形で、実効 0.8 GB/s 相当。
修正後は 1 グループの作業域が `k*MB` = 128 KB の連続領域 (L2 に乗る) になる。

**分かったこと / 残った課題** — 5,686 token warm で `gdn_proj` **5,405 → 1,083 ms (5.0 倍)**、
prefill **28.1 → 23.7 s (202 → 240 tok/s)**。セッション開始時の 46.2 s から −49%。転置はデータ配置だけ、
各 (i,j) の d 昇順逐次累積は 1 命令も変えていない = **ビット一致**。アンカー 3 本 MATCH。残った最大項目は gate_up 7.38 s。
`matvec_rows_bt_vec_impl` (n ≥ 64 経路) は同じ `[d][i]` レイアウトのままで、同種の余地が残る。

**次に効いたこと** — 「レイアウトを変えるだけで 5 倍」が、s68 の「コピー経路を変えるだけで 1.56 倍」と並ぶ、この時期の作業の型になった。

### s68: pinned ステージング経由の D2H が GEMM 本体の 5 倍を食っていた (2026-10-19, `b682c3f5`)

**意図** — s67 後、prefill 23.1 s の最大項目は `gate_up_proj` 7.38 s (64 call・114 ms/call)。
micro-bench で 1 call の全段を再現すると、GEMM 本体 12 ms・量子化 6 ms・H2D 7 ms に対し **D2H 20 ms + ホスト間コピー 60 ms
** が乗っていた。

**変えたもの** — `cuda.rs::linear_gemm_into` で H2D を呼び出し元の `a` スライスから直接発行 (pinned
`hx` への `copy_from_slice` を削除)、D2H を呼び出し元の `dst` へ直接発行 (pinned `ho` + `copy_from_slice` を削除)。
転送内容・カーネル引数・演算順は不変。

**つまずき** — 「pinned に落としてから Rust のバッファへ `copy_from_slice`」という常套句が、この環境では二度手間の上に 4 倍遅い。279 MB・3
回平均の実測:

| 経路 | ms | 実効 |
|---|---|---|
| D2H → pinned + `memcpy` → heap (旧) | 79.8 | 3.5 GB/s |
| D2H → pageable 直 (新) | 20.1 | 13.9 GB/s |
| D2H → pinned のみ | 19.5 | 14.3 GB/s |

pinned→heap の `memcpy` が 60 ms = **4.7 GB/s** しか出ていなかった。

**分かったこと / 残った課題** — `gate_up_proj` **7.38 → 2.87 s** (114 → 44 ms/call)、prefill **23.1 → 14.8 s
(246 → 384 tok/s)**。セッション開始時の 46.2 s から −68% / 3.1 倍。アンカー 3 本 MATCH、テスト 231
passed。残った内訳は `d2h_state`・`attn_core`・`gate_up` で、**次の標的は依然 D2H** — `chain_download`
(attention 出力 93 MB) と o_proj の `to_vec` が同じ pinned 経由のまま残っていた。

**次に効いたこと** — 13.9 / 14.3 GB/s という実測単価が、以降の「転送は壁か」の判断 (s170 の往復棚卸し、P41 の設計メモ) の基準値になった。

### s69: チェーン D2H の「pinned 経由 + to_vec」を廃止 (2026-10-19, `1f588b4f`)

**意図** — s68 で GEMM 経路の二度手間を消したが、attention 出力の取得 `chain_download` と o_proj の戻り値
(`maybe_gpu_oproj`) は「pinned へ D2H → `to_vec` でヒープへ複製」のままだった。93 MB の出力を毎回 2 度書いていた。
(一次記録ではこの見出しが 2 回重複していたため、本巻では 1 エントリに統合した。)

**変えたもの** — `cuda.rs` の `chain_download` / `chain_download_at` は出力 `Vec` を `with_capacity` で確保
し、その未初期化領域へ直接 D2H して完了後に `set_len`。`maybe_gpu_oproj` も pinned `ho` をやめ、戻り値の
`Vec` へ直接 D2H。

**つまずき** — 転送先だけの変更で、演算にも発行順にも触れていない。`set_len` の前にコピー完了を保証する必要があるため、同期の位置は動かさないことを条件にした。

**分かったこと / 残った課題** — prefill **14.80 → 14.50 s (384 → 392 tok/s)**。差は 0.30 s で、
s68 の 1.56 倍とは桁が違う — attention 出力 93 MB × 16 層 ≈ 1.5 GB (換算) では、消えたコピーの絶対量が小さい。
アンカー 3 本 MATCH、テスト 231 passed。

**次に効いたこと** — ホスト側の余分なコピーを消し切ったことで、残るコストが**活性化そのものの往復**だと見えるようになり、s70 の FFN 常駐化に直結した。

### s70: prefill の FFN をデバイス常駐チェーンに — 活性化の往復を 4 回から 2 回へ (2026-10-19, `9ec23f3c`)

**意図** — s68 / s69 でホスト側の余分なコピーを消すと、残るコストは活性化の往復そのものになった。1 層あたり (m=2843, e=4096,
ff=6144) は H2D 47 MB (x) + D2H 140 MB (gate/up) + H2D 47 MB (swiglu 後) + D2H 47 MB
(down) = **281 MB**。デバイス常駐チェーンは decode (m=1) 用にしか配線されておらず、prefill は素通りしていた。

**変えたもの** — `chain_gemm_grouped_pf` (prefill 用 grouped int8 GEMM。`dno_gemm_i8` を起動し
`out_slot` へ書き込み・同期なし)、`chain_swiglu_rows` (`dno_swiglu_int` を m>1 で)、`chain_mlp_prefill`
(rmsnorm 済み x を H2D → gate/up → swiglu → down → 呼び出し元バッファへ直接 D2H)、CH_PFH スロット新設。
`qwen35::forward_prefill` の FFN 部を差し替え、`quant8()` でない / m<256 / 失敗時は従来経路へ落ちる。

**つまずき** — 既存の `chain_linear_grouped` は decode 前提で `m > max_m()` (=16) なら諦めて
GEMV カーネルを起動するため、そのままでは使えなかった。既存 `chain_swiglu` は gate/up ブロック分離レイアウトで m=1 専用だった。
どちらも「既存の常駐チェーンを prefill に流用する」ことができない構造で、pf 版を別に立てる必要があった。

**分かったこと / 残った課題** — GEMM は同一カーネル・同一整数累積、swiglu は CPU の `gv / (1.0 + (-gv).exp()) * uv` と同一の
f32 演算列 (`dno_expf` + `__fdiv_rn` / `__fadd_rn` / `__fmul_rn`) なので**ビット一致**
(アンカー 3 本 MATCH)。`gate_up_proj` 5.74 → 3.52 s、`down_proj` 2.81 → 1.01 s、CPU の
swiglu 1.66 s は GPU へ。prefill **14.50 → 11.58 s (392 → 491 tok/s)**、1 層あたりの活性化転送 **281 → 94 MB
**。セッション開始時の 46.2 s から −75% / 4.0 倍。残りは `attn_core` (GDN スキャン・gdn_proj・attn_full) と
qkv。

**次に効いたこと** — 「pf 版チェーンを別に立てる」判断が、後半の P14 (q4 で gate|up を 1 回の GEMM に畳む) と同じ形として再登場する。

### s71: s69 の D2H 直行化を復旧 (2026-10-19, `91b54265`)

**意図** — s70 のパッチ生成元が **s68 時点の `cuda.rs`** で、s69 の変更 (chain_download / chain_download_at /
maybe_gpu_oproj の直接 D2H) を巻き戻していた。つまり s69 の次のコミットで s69 が打ち消されており、s70 で測った 11.58 s は
**s69 抜き**の値だった。

**変えたもの** — s70 の上に s69 の 3 箇所を再適用 (`pin_dn` / `ho` のステージングを削除)。

**つまずき** — パッチを scratch の作業コピーから生成していたため、直前 commit の差分が黙って落ちた。ビルドもテストも通り、数値も正しいので、
失われたのは性能だけ — 計測しなければ気づけない種類の事故。s69 の効果 (0.30 s) は s70 の上でもそのまま残っていた。

**分かったこと / 残った課題** — prefill **11.58 → 11.34 s (491 → 502 tok/s)**、アンカー 3 本
MATCH、テスト 231 passed。

**次に効いたこと** — 再発防止として「パッチ生成スクリプトの入力は必ずリポジトリの HEAD のファイルから取る」を規約化。T5a の「計測が嘘をつく」
と同系の、静かに残る退行の 2 例目として扱われるようになった。

### s73: full attention の prepare をトークン並列化 (2026-10-19, `57405fe9`)

**意図** — `qk_norm` 650 ms + `rope` 491 ms が CPU 単スレッドで走っていた。1 行分は 5 µs 程度だが、5,686 行 × 8 層で 1 秒
を超えていた。

**変えたもの** — `qwen35::forward_prefill` の full attention 経路で、QK-Norm → RoPE →
KV キャッシュ行書き込みを行 (トークン) 並列に。各行の書き込み先は互いに素 (q の行・KV の pos 行) で、1 行分の演算列はトークン逐次経路と同一。
prof は `qknorm` バケットに一括計上 (内訳の合算と同値)。

**つまずき** — 演算列を変えていないのに、**短いアンカー 2 本の出力が変わった** (決定論は維持・full アンカーは不変)。原因は未特定で
T9 として追跡。長文 (4,671 token) の needle recall が正常なことは確認した。緩和クラス R1 (f32 縮小順序) として
`docs/NUMERIC_POLICY.*.md` の手順で再基準化: v6_1b `c61ad85b…` → `47df93d7…`、v6_2b `9e17d563…` →
`9b20de09…`、v6_1b_full `af6eb6ab…` は不変。再基準化は 2 回実行して決定論を確認したうえで `--write`。

**分かったこと / 残った課題** — prefill **10.11 → 9.65 s (562 → 589 tok/s)**。スレッド数非依存 (`RAYON_NUM_THREADS=1/8/16`
で同一 md5)。ただし s75 で測り直すと**ドリフトは再現せず旧基準値に戻った** — 1 ulp 級の摂動がサンプリングのニアタイでトークンを反転させたと解釈するのが自然で、
並列 prepare が未初期化領域を読んでいる可能性は未検証のまま T9 に残った。

**次に効いたこと** — 「数値が動いたら再基準化して先へ進む」手順が実際に適用された最初の例。同時に、原因未特定のまま基準を動かす危険も記録として残った。

### s74: int8 GEMM の per-thread タイルを 8x16 へ (2026-10-19, `e0ea110b`) — PTX 未生成

**意図** — `dno_gemm_i8` は 128x128 タイル・256 thread・1 thread 8x8 出力で、1 回の `__dp4a`
(4 積和) に対して smem からの B ロードが 1 個分しか当たらず、内側ループは dp4a 144 = b64 ロード 16 + a32 ロード 4。
実効 **10.6 TOPS** は int8 の理論性能から 1 桁低い。

**変えたもの** — 1 thread の出力を 8x8 → **8x16**、1 ブロック = 128 行 × 256 列・**64 thread**
(block 16x4)。dp4a/B ロード比を 8 → 16 に倍増。256 列にしたのはアキュムレータ (8x16 = 128 個) とスレッド数
(64) の積で 8192 出力/ブロックを保つため。k 方向 4 要素は i32 パックのまま `__dp4a` を直接使う。重みは 256 行に 0 パディングしてアップロード
(`GpuW.npad` に記録)。ホストの launch 形状を全 11 箇所で更新。

**つまずき** — アキュムレータが倍になるので、旧版が spilling していた他のレジスタ (a/b パック・アドレス計算・ループ展開) をロールバックする必要があった。
パディングは A 行を 2 枚の B タイルで共有するため A/B 非対称でよく、重み側だけ常に 0 にすれば寄与しない。実モデルの形状はすべて 256 の倍数なので
VRAM 増はゼロ (倍数でない形状のときだけ余り行を確保)。

**分かったこと / 残った課題** — int8 の積和は i32 で厳密なので、タイル形状・スレッド数・アキュムレータ分割を変えても dot は同一。
f32 側も `((float)dot * ascale) * scale[j]` の順を維持するのでビット一致が期待値 (緩和クラス R2 で許容される差も出ない見込み)。
**`kernels_int8.ptx` は未再生成** (このビルド環境に nvcc が無い) で、**GPU 経路は旧 PTX (8x8 カーネル) のまま動く**。
旧カーネルは列 j の `gj < n` で打ち切るので数値は正しく、**退行はしないが高速化も発動しない**。性能は**未測定**。

**次に効いたこと** — 「PTX を再生成するまで新ホストコードと旧カーネルが同居する」状態を明示的に記録する慣行が生まれた。この状態が s75 の即死
(error 716) の前提になった。

### s75: dno_gemm_i8 の共有メモリ アラインメントと行カバレッジの修正 (2026-10-19, `5bca9b7e`)

**意図** — s74 のカーネルは GPU 実機で即死した。最初の int8 GEMM で `cuMemcpyDtoHAsync` が error 716 → 恒久
CPU フォールバック (PF wall 171.68 s / 33.1 tok/s、ベースライン 9.65 s / 589 tok/s の 17.8 倍遅い)。

**変えたもの** — ①`GEMM_SP` を 33 (`GEMM_TK+1`) から **36** (`GEMM_TK+4`) へ。4 の倍数かつ 32 の倍数でないので、
隣接 thread の 8 行差 = 288 B = 72 words、72 mod 32 = 8 で bank が 8 個ずつずれる。②block を
`(16,16)` = 256 thread に戻し `GEMM_BX` / `GEMM_BY` を導入してタイル 128x256 を過不足なく覆う。③`static_assert` 5
本。④ホストの launch 形状 10 箇所を `(GEMM_BX, GEMM_BY, 1)` に統一。

**つまずき** — 原因は独立に 2 件。①共有ステージの行ストライドが 4 バイト境界に揃っていない: カーネルは `*(int*)&sA[tid][c]` と 4 バイトのポインタキ
ャストでアクセスするのに、ストライド 33 は奇数なので行が進むたびに未アラインになる。sticky な error 716 になり以降の全 D2H /
sync が失敗する。compute-sanitizer の診断は `Invalid __shared__ write of size 4 bytes / Access at 0x421 is misaligned`
。②ブロック形状がタイル寸法と食い違っていた: 64 thread × 8 行 = 32 行しか覆わず、128 行タイルの **32..127 行が誰にも書かれない**
(出力の約 75% が未初期化)。**CPU テストは緑のまま**で、このクラスの欠陥は GPU か静的検査でしか捕まらない。

**分かったこと / 残った課題** — PTX を nvcc 12.9 で再生成 (md5 b5c926ea → fe254249)、アンカー 3 本一致。
ウォーム 2 回目は PF wall 8.71 s / 653 tok/s (s73 の 9.65 s から −9.7%) だったが、**これは過大**だった — 初回に
PTX の JIT と重み常駐が乗るため。cold A/B を取り直すと s73 `57405fe9` が 14.222 / 14.221 s (399.8 tok/s)、
s75 `5bca9b7e` が 13.407 / 13.429 s (424.1 / 423.4 tok/s) で、**寄与は −5.7% (0.80 s)**。
以後の比較は cold 単発を正式値とし、ウォーム値は同一セッション内の相対比較にのみ使う。CPU テスト 233 passed。

**次に効いたこと** — 「計測基準の訂正」を HISTORY に残す形が定着した。`include_str!` で `.cu` を取り込み `#define` を簡易評価する静的検査テ
ストは、変異テスト (`GEMM_SP = GEMM_TK+1` / `GEMM_BY = 4` / `sA[tid][x*3]` のいずれでも落ちる) で有効性を確認したうえで恒久化され
た。

### s80 / s81: MLP の gate+up をロード時に連結して 1 回の GEMM に (2026-10-19, `b69e0190`)

**意図** — prefill の MLP は gate・up・down を別々に投影しており、GPU 経路では gate と up で 2 往復していた。
`dno_gemm_i8` は行 (出力チャネル) ごとに独立で n をタイルで舐めるだけなので、gate と up の重みを行方向に連結すれば 1 回の
GEMM で両方計算できる。

**変えたもの** — `Layer.gate_up_w` (`[2*ff, e]`・行順 `[gate 全行 | up 全行]`) をモデルロード時に
`concat_rows()` で 1 回だけ生成 (新しい uid を採番)。prefill の int8 経路は `q8_apply_bt_rows_into()` で連結重み 1 呼
び出し、SwiGLU は連結レイアウト ([g 全列 | u 全列]) に合わせた分岐を持つ。`chain_mlp_prefill()` は `gate_up_w` 1 本を受け取り
`ff = n/2` を導出。decode chain の GATE_UP も統合。fp32 経路は従来どおり 2 呼び出しのまま。

**つまずき** — 連結重みは新しい uid を採番しないと GPU のデバイスキャッシュ鍵が重量で衝突する (P17b のポインタキー事故と同じ類型)。
また層ごとに連結コピーを持つと常駐メモリが増えるため、ロード時 1 回・以後はホスト側の連結コピー不要という形にした。

**分かったこと / 残った課題** — ビット一致の根拠は「int8 の量子化スケールは行ごと、整数ドット積と累積は行独立」なので、行を連結しても各行の量子化値も累積順も変わらないこと。
fp32 側も `matvec_rows_bt` の累積が (i, j) ごとに閉じているため行順に依存しない。テスト 3 本で int8 / fp32 の両方・m=1 /
m=3・prefill の logits と GDN 状態のビット一致を固定 (engine lib 232 passed)。コストは常駐メモリ増 (int8 で約 1 GB /
fp32 で約 4 GB 相当)。**GPU 実行はしておらず、性能は未計測**。

**次に効いたこと** — 同じ理屈 (行連結だけならビット一致) で attention 側の `qkv + z` と `beta + alpha` の融合へ進む道が開けた。
なおその融合は s83 で実装され、同日 revert されている (`aa973eaf`)。q4 系では P14 が同じ形を取る。

### s84: rmsnorm の実測 — 「0.4 s」はブロック全体の誤称だった (2026-10-19, `66698f68`) — 最適化せず

**意図** — T8 の問いは「rmsnorm を残差加算と融合すべきか」。判断線は 0.25 s。

**変えたもの** — `prof.rs` に `Ambient` スコープ (スレッドローカルの親バケット。内側のサンプルは親にも自分のバケットにも加算) と行単位フック
`norm_rows` / `norm_rows_end` (1 層 1 回だけ `Instant` を読む)、表示用の `norm` / `norm_other` /
`block` / `block_rest` 行を追加。既定 off はクロック読みゼロの真の no-op。`qwen35.rs` は prefill の各層
attention ブロックと FFN ブロックを親スコープで包み、norm 呼び出しの後にイベントを撃つ。

**つまずき** — 実測 (9B・5,686 token・PF wall 13.733 s / 414 tok/s) で `rmsnorm` 行が 11,761.8 ms
(43.3%)、その子 `norm` が **0.0 ms / calls 0**。①`rmsnorm` 行は正規化ではなく**層ブロック全体**で、中の
`attn_core` / `gate_up_proj` / `qkv_proj` / `down_proj` / `o_proj` が二重計上される。
②`norm` のカウンタが 1 回も発火していない — prefill の int8 経路はデバイス常駐チェーンで走るため、正規化はホスト側の関数呼び出しとして現れない。
したがって「0.0 ms」は「無料」ではなく「ホストから見えない」の証拠。③二重計上のため表は加算できない (`TOTAL 37,922.7 ms` >
`[prefill] total 27,182.8 ms` > wall 13.733 s)。以前の「rmsnorm 0.4 s」も同じ親スコープの切り方の産物だった。

**分かったこと / 残った課題** — 最適化しない判断の根拠は計測ではなく上限の見積り: 正規化は層ごとに隠れ状態 1 本を読み書きする要素演算で、
9B の 5,866 tok × 4,096 次元 × 16 層 = 1.54 GB、読み書きで約 3.1 GB。GPU 帯域 1 TB/s なら約 3 ms、100 GB/s に落ちても
約 31 ms で、13.2 s の prefill に対して 0.25% 以下 — **0.25 s の判断線には桁で届かない** (いずれも換算・推定)。
step 2 (残差加算との融合・トークン方向のバッチ化) は実施しない。decode は 16.37 ms/token で不変、アンカー 3 本 MATCH。

**次に効いたこと** — 「ブロック全体を親スコープで包んではいけない」が s87 の会計作り直しの直接の動機になった。発火しないカウンタ (calls=0) をそのまま表に出す規約もこ
こで定まった。

### s87 / s88: プロファイル会計の加法化と d2h_state の発行点別計測 (2026-10-20, `e547749d` / `26c6b4a3`)

**意図** — s84 の `Ambient` は「親 1 本 + 子へも合算」だったため表が加算的でなく、`d2h_state` は 1 本に複数の発行点が混ざって「どこが重いか」
を読めなかった。計測器の 2 段の作り直しで、どちらも数値経路には触れない。

**変えたもの** — s87 (`e547749d`): `AMBIENT` を単一スロットからスタックへ。時間の書き込みを `add_outer_ns` 1 本に統一。calls /
toks の二重計上を修正 (`owners` に自分のバケットが含まれるため件数が 2 倍になっていた)。collapsed な sample は子としても数え、
`self + 子 = 親` を作る。**`self` 列 (7 列目) = 親 − 子** を追加し、フェーズ総和と累積% は `self` の和で計算。
s84 の行単位 norm フックと層ブロックの `Ambient(RMSNORM)` 包みは撤去。s88 (`26c6b4a3`): `D2H_SRC_NS[phase][site]`
を追加 (**表示専用** — どの生バケットにも合算しないので二重計上が起きない)。`mark_d2h_src_ns` / `add_d2h_src_ns_only` /
`d2h_src_on()` を追加し、prefill の表に発行点別の上位 10 行 (ms / calls / us per call) を出す。

**つまずき** — 時間を書く場所が 2 箇所あっただけで全サンプルが 2 倍になっていた (気づかないまま全体が水増しされる)。件数と時間を別 API に分けたため「件数を誰が積むか」
が曖昧になり、両方呼ぶと件数が 2 倍になる罠も実際に踏んだ。テストはグローバルロック + `#[cfg(test)]` の強制フラグ (env を書かないので並走テストを揺らさない) +
入口 `reset()` にして、テスト順依存をなくした。

**分かったこと / 残った課題** — GPU 実測で GPU_VERIFY PASS・アンカー 3 本 MATCH = 計測器は数値 no-op。発行点別の**順位付けはこの
commit より前に存在しない**ため、修正前の値は `d2h_state` のバケット合計だけ (prefill: CPU 機 4,832.3 ms / 160 calls、
GPU 機 6,165 ms / 1,140 calls)。最初の実順位は以降のビルドの計測から得る。検証実行で新バケットが動くことは確認済み (decode 1 回で
`d2h_src_other` 88.9 ms / 548 calls・`d2h_src_attn_pf` 9.3 ms / 8 calls・`d2h_src_gdn_state` 0.0 ms
/ 432 calls・`d2h_src_linear` 0.0 ms / 2,386 calls) — これは decode 1 回の観測で、prefill の順位ではない。

**次に効いたこと** — 「表示専用の表は生バケットへ合算しない」という規約で、会計を変えずに内訳を足せるようになった。後半の P25 (ffn_down の内訳確定) はこの加法化され
た土台の上で読む。

---

### P14: FFN の gate|up を 1 回の GEMM に畳む (2026-10-23, `c4c6f569`)

**意図** — FFN の 1 タイルは gate GEMM + up GEMM + swiglu + down GEMM の 4 発で、gate と up は同じ入力 `pact` を読み、
間に同期が無い。この 2 発を 1 発にすれば launch が減る。1 層 56 発 → 42 発、リクエスト全体 (64 層) で 3,584 → 2,688 発 (−896)。

**変えたもの** — `c4c6f569` / `crates/denno-engine/src/cuda.rs` / `crates/denno-engine/src/model/qwen35.rs`。
`gate_up_w`(=[gate 全行 | up 全行] の行連結 1 本・ロード時に既にある) を n=2ff で 1 回 GEMM し、`CH_GA` へ行内 `[gate|up]` で書く。
続く swiglu は `dno_swiglu_int` のまま。カーネル・累積順・grid・タイル幅は不変なので `CH_DOWN` はビット同一。`DENNO_PF_FFN_FUSED=0`
で従来の 2 回呼び出しへ戻せる。

**つまずき** — 畳めるのは gate と up だけで、swiglu は既に同じチェーンの中 (間に sync 無し) なので 4 → 3 発で止まる。前提が崩れたときに静かに壊れないよう
`pf_ffn_fused_ready` (すべて Q4・形の整合・gate/up が連結側の行範囲として登録済み) を置き、満たさなければ従来経路へ fail closed にした。重みは
`prefill_layer_dev` の先頭 (個別の gate/up が VRAM へ載る前) で `q4_register_row_alias` し、
decode chain と同じ 1 本を共有するので VRAM と H2D は増えない。

**分かったこと / 残った課題** — 予測は `sl_lay_ffn_gate` 10,042.6 → ≈10,040 ms (Δ −2 ms)、`sl_lay_swiglu` 210.4 と
`sl_lay_ffn_down` 11,100.0 は不変 (いずれも n=14,043・wall 63.05 s の run の値で、現行ビルドの実測ではない)。
MAC 数・総ブロック数・タイル幅が不変で消えるのは launch 896 発だけなので、ms の利得はほぼ 0 の見込み (1 発 3〜6 µs)。
実機 A/B と 8 問アンカーの md5 は未計測 (GPU 非搭載)。ゲート: `--lib` 279 passed、`--features cuda --lib`
320 passed / 37 ignored (+4 本)。

**次に効いたこと** — 「launch 数は律速ではない」が同じ巻の中で 2 回確認される。P21 は GEMM 形状を選んでも 27B では 0 ms、
P51 は 1,344 launch/req の発行を ≈9 ms と見積もっている。i8 側の同種の話 (gate と up を 1 発に) は本巻の s80/s81 にある。

### P20: full 層 attention の q を pinned ステージングへ直接書く (2026-10-23, `5f222949`)

**意図** — full 層の `sl_lay_full_core`
は「ホスト prepare → q の H2D → attention 1 発 → att の D2H → ホスト sigmoid ゲート → att の H2D」で、
prepare が書いた q を一度ホストの `buf_q` に置いてから pinned ステージングへ写していた (345 MB/層)。
まず指示の 2 分岐を判定した: (a) heads はホストでループしていない — attention は既に 1 発/層で、`grid.y = heads = 24` のカーネルが
`blockIdx.y` を head として読む。(b) attention と o_proj の間のホスト同期は 1 回だけで、
これはホストが att 345 MB を読んで sigmoid ゲートを掛けるための D2H 完了待ちなので落とせない (落とすには n トークン版のゲートカーネル = `.cu` が要る)。

**変えたもの** — `5f222949` / `crates/denno-engine/src/model/qwen35.rs`。prepare の書き込み先を `buf_q`
から q の pinned ステージングへ変え、H2D の転送元をそのまま pinned にする。launch 表は不変: `sl_lay_full_qk`
は 3 GEMM (grid 220×96 / 220×8 / 220×8、block 16×16) + D2H 805 MB + sync 1、`sl_lay_full_core` は
`dno_attention_prefill` 1 発 (grid 220×24、block 256 スレッド、smem 84,800 B、
BQ=64/BK=8) + att D2H 345 MB → ゲート → H2D 345 MB、`sl_lay_full_oproj` は 1 GEMM。1 full 層 5 発は前後で同じ。

**つまずき** — q のコピーを消しても位相はほとんど動かない。律速は attention カーネルの K/V の 650 倍再読み (帯域) で、それは `.cu` の仕事。
launch の寄与は 5 発 × ≈5 µs = 25 µs/層で、指示書の言う「律速」には届かない。CPU attend へ落ちる経路のために prepare を `buf_q`
へやり直す枝は残した (KV 行は同一値の上書きなので冪等)。

**分かったこと / 残った課題** — 予測 (実測ではない。CPU 実測単価 21.1 ms/層 × 16 層) は `sl_lay_full_core`
10,695.3 → 10,355〜10,540 ms (−0.16〜−0.34 s)、`sl_lay_full_qk` は不変。
値はビット同一 — カーネル・grid・block・smem・引数・H2D バイト数・発行順は不変で、変えたのは「H2D の転送元が `buf_q` のコピーか、
prepare が直接書いた pinned か」だけ。`docs/KPI.md` の `5f22294`
の行は 3,161 token 9.69 s / 12,850 token 38.79〜40.35 s (cold・毎回 nonce 付き・3 回の中央値・1 回目は捨てる)。
実機の ms・E2E 2 回のバイト一致・新経路の実行は未計測。

**次に効いたこと** — 「位相の中で消せるのはホストの直列コピーだけで、転送とカーネルは `.cu` を触らないと動かない」という線引きが、
P28 (読まれない確保)・P44 (V の純コピー)・P46 (往復の直列化) の選別基準になった。

### P21: q4 prefill の GEMM 行タイルを投影の n から選ぶ (2026-10-23, commit なし)

**意図** — `sl_lay_gdn_proj` が 18.0 s と言われていたので、まず前提を疑った。
指示の「18.0 s / 7.6 TFLOP/s」は P13 以前の測定 (n=14,043・wall 63.05 s の run) で、同じ KPI は P12+P13 以降 39.0〜42.18 s。
18.0 s の内訳を測った単価で積み直すと host beta/alpha 9.75 s + h2 の D2H 4.9 s + GEMM 3.67 s = 18.3 s (実測 18,015.8 ms との差 1.7 %) になり、
81 % は GEMM ではなかった。

**変えたもの** — `crates/denno-engine/src/cuda.rs` に `q4_pf_tile` / `pf_gdn_tile` を足し、行タイルを投影の n から選ぶ。埋めの下限
`(t/64)·⌈n/128⌉ ≥ 4·sm`、L2 の上限 `A(t·k·4) + B(n·(k/32)·20) ≤ 64 MiB`、`sm = 0` は従来値を返す。
`DENNO_PF_GEMM_SHAPE=0` で固定タイルへ戻せる。**この実装は commit されず**、次の P25 の STEP 0 で WIP のまま破棄された (`git restore`
1 ファイル)。関連する docs commit は `86c44535` (P29 が 18.0 s を歴史的な値として注記した)。

**つまずき** — 指示の方向 (狭い n ではタイルを下げる) は 1 次元 grid の GEMV の話で、GEMM は 2 次元 grid なので符号が逆だった。
`blocks = (tile/64)·⌈n/128⌉` なので tile を下げると block 数も下がり SM の埋まりが悪くなる。狭い n では tile を上げるのが正しい。
27B の qkv (n=10,240・80 列ブロック・1,280 block) と z (n=6,144・48 列ブロック・768 block) は tile 1,024 で既に 4.5 波以上あり、
選択は発火しない。

**分かったこと / 残った課題** — 27B では before == after で、
1 層 28 launch・1 リクエスト 1,344 launch・1 層 28,160 block がすべて不変、予測は 0 ms。発火するのは n ≤ 4,096 の投影だけ。
GEMM が形で損していない根拠も固めた: 1 block = A 64 行 × B 128 列 × k 全体で、演算強度 48.76 FLOP/B は n にもタイルにも依らない。
同じカーネル・同じ block 仕事の `gdn_oproj` が 30.8 TFLOP/s、`ffn_gate` が 31.9 TFLOP/s で、
qkv/z だけ 11〜13 TFLOP/s に落ちる機構は無い。

**次に効いたこと** — 律速は host beta/alpha + h2 のコピーだと確定したので、次の一手は「ホスト投影の SIMD 余地」と「beta/alpha のデバイス化」に絞られた。
前者は P51 の AVX2 16 レーン化として実行された。

### P25: `sl_lay_ffn_down` の内訳を確定する (計測のみ) (2026-10-23, commit なし)

**意図** — `sl_lay_ffn_down` が 11.1 s と言われていたので、指示の規則「host または copy が位相の 20 % 以上なら 1 点だけ入れる」に当たるかを先に確かめた。
当たらなければ何も入れない、という順序が指示されていた。

**変えたもの** — なし。区間 `[PF_PH_FFN_DOWN のイベント → 次のイベント]` を 1 行ずつ確認した結果、入るのは `dno_gemm_q4_0` の launch 1 発だけで、
H2D 0 B・D2H 0 B・`ctx.sync()` 0 回・alloc 0 回・量子化 0 回だった。残差常駐経路 (`PfSrc::Slot` / `PfSink::Slot`) なので
`dst = None` になり、`chain_mlp_prefill_q4_impl` で唯一 sync を呼ぶ枝に到達しない。prologue (`weights_for_q4` ×3、
`chain_ensure_buf` ×4) はタイルループの前なので前の位相に落ちる。条件不成立なのでコード 0 行・commit なし。関連 docs commit は `70675d32`
(P30 がこの手順にあった env の付け方をサーバ起動側へ直した)。

**つまずき** — 内訳は GEMM 100 % なのに、同じカーネル・同じ block 仕事・同じ演算強度 (24.38 MAC/B) で down だけ 14.4 TFLOP/s、
gate|up は 31.9、o_proj は 30.8 だった。リファレンスのレートで積むと down は 5.02〜5.20 s で済むはずで、
実測バケット 11,100 ms との差 5.9〜6.1 s がデバイス側に残る。
候補は 3 つ: ①1 launch の block が 640 しかないことによる波の量子化 (SM 170 に対して 3.76 block/SM、
効率差の上限 1.5 倍 = せいぜい 3.5 s) ②1 波 launch で全 block が同じ barrier に同期するため、
single-buffer の k ループがレイテンシを隠せない ③層末尾の位相なのでホストの遅れが全部ここに落ちる。切り分けは nsys 1 回で足りる、というところまで。

**分かったこと / 残った課題** — 11,100.0 ms と各レートは s170 の run (n=14,043・wall 63.05 s) の値で、P12/P13 前のホスト律速ビルドのもの。
現行ビルドで同じ 11.1 s かは未測定で、「5〜6 s に落ちていれば down はカーネル天井で、この位相にレバーは無い」という予測を残した。tile (1,024) は波の量子化を試せる唯一の
`.cu` 不要レバーだが、切り分けが付くまで触らないと決めた (行のまとめ方だけを変えるのでビット一致は保たれる、という性質だけ確認)。

**次に効いたこと** — 「バケットをそのままカーネル時間として読むと嘘をつく」という疑いが、次の P27 の会計監査の入口になった。

### P27: 位相会計の監査 — `DENNO_PF_DBG` は何を測っていないか (2026-10-23, commit なし)

**意図** — P25 で「GEMM 100 %」までしか言えず、その 100 % の中身 (カーネル時間なのか隙間なのか) が決まらなかった。判断を誤らないために、会計の仕組みそのものを監査した。

**変えたもの** — なし (コード変更 0 行・commit 0 本)。関連 docs commit は `86c44535` (P29 がこの監査の結論 4 点を docs に固定した)。

**つまずき** — `pf_phase()` は既定ストリームにイベントを打つだけで、`pf_timing_read()` が `cuEventElapsedTime(ev[k], ev[k+1])` を
`phase_of[k]` に積む。したがって各区間は「その区間で発行したカーネルの実行 + 区間内の GPU アイドル + ホストの遅れ」で、
11 バケットは wall の完全な分割だが**カーネル時間への分割ではない**。`sl_lay_ffn_down`
の 11.1 s が「12.39 ms/launch のカーネル」なのか「カーネル 6 ms + 隙間 6 ms」なのかは、この仕組みでは原理的に決まらない。
さらに帰属が系統的に 1 位相ずれる: 位相境界は発行グループの先頭に置かれ、`weights_for_q4` / `chain_ensure_buf`/ upload はその前にあるので前の位相に落ちる (FFN の prologue → `norm_post`、full の qkv prologue → `norm_in`、
o_proj prologue と att upload → `full_core`、最終タイルの down 区間 → 次層の prologue)。

**分かったこと / 残った課題** — 実例で示せた: 同じ `dno_add_rmsnorm` が `norm_in` 29.0 ms/層、`norm_post` ≤ 4.0 ms/層で 7 倍以上違う。
s170 の `norm_in` はカーネル時間ではない。cold run では重み 13.21 GB の H2D がその呼び出し位置の位相に落ち、`norm_post`
が ≈0.5〜0.6 s 水増しされる — これが `prefill_bench` の 1 回目を捨てる理由でもある。
s170 の表は記録 9 行の和 61,999.6 ms (98.33 %) + tail 567.4 ms で「説明できない wall」がほぼ無いが、
ホスト律速ビルドでは遅い位相が自分の遅れを自分で吸うので、会計が合っていることの証拠にはならない。
3,161 token の wall から出る制約は「down に残るのは ≈1.0 s (全床を楽観端に取って 2.6 s)」まで。

**次に効いたこと** — 「バケットは wall の分割」という規約が P29/P30 で docs に固定され、
以後この巻の数値はすべて「どのビルドの・どの n の・誰が測ったか」を添えて読むことになった。P41 の `DENNO_FC_KPROF` は、
この反省を踏まえて区間を 4 つに割り直す装置として設計されている。

### P28: `sl_lay_full_core` のホスト側を先に測った (2026-10-23, `3c383a7f`)

**意図** — 指示書の「まず測る」に従い、位相の launch / 転送 / ホスト費用を表にした。位相は device bound (attention カーネルが支配的、
転送 1.15 GB/層) で、
ホスト側は prepare 44.9 ms + sigmoid ゲート 45.4 ms = 90.3 ms/層 (n=14,043・16 full 層で 1.45 s/リクエスト)。
この 2 つはどちらも「必要なデータ移動」で、消すにはゲートのデバイス化 (`.cu`) が要る。

**変えたもの** — `3c383a7f` / `crates/denno-engine/src/model/qwen35.rs` のみ (+24/−3)。位相の中で毎リクエスト
`fit_prefill_buf(buf_q)` と `fit_prefill_buf(buf_att)` が 2 × 345 MB を mmap + ゼロ埋めしていたが、
`AttnPlace::GpuDev` 経路はこの 2 本を一度も読まない (>128 KB の確保は malloc のアリーナを通らないので、ページフォルトを毎回払っていた)。確保を使用点
(`!q_in_pin` の prepare 前 / CPU attend のゼロ埋め前 / GpuHost の `copy_from_slice` 前) へ移した。

**つまずき** — live 経路の確定に手間が要った。`DENNO_GPU_ATTN` 既定 ON + `attn_prefill_gpu_supported` (`n >= gemm_min_m()`= 既定 256 を判定、dh=256 は smem 84,800 B を opt-in 済み) なので n=14,043 は GPU attend。残差常駐経路は
`host_att_needed=false` なので `chain_pf_att_inplace` が走る `AttnPlace::GpuDev` = P12 の pinned 直送り経路のまま。
`buf_att` のゼロ埋めは `if !att_done` の中だけなので P12 の削除は保たれている — この 3 点をコードで確認してから「読まれない確保」を消した。

**分かったこと / 残った課題** — 実測 (GPU 不要・`rustc -O`・27B 実シェイプ): prepare 44.9 ms/層 (3,198 ns/token)、
ゲート 45.4 ms/層 (3,233 ns/token。うち exp 12.1 ms、残りは 690 MB の読み書き = 21 GB/s)。
n=12,850 で 38.2 + 43.1 = 81.2 ms/層、n=3,161 で 11.1 + 11.2 = 22.2 ms/層。帯域の校正は同じ 345 MB のコピーで r+w 41 GB/s。
予測効果 (実測単価 × 2 本) は −0.24 s/request @14,043、−0.25 s @12,850、−0.06 s @3,161。
値は 1 ビットも変わらない (使う経路はどれも使用直前に全要素を書く)。実機の cold A/B は未計測 (GPU 非搭載)。テストは `--features cuda --lib`
321 passed / 37 ignored で不変。

**次に効いたこと** — 「位相の中で毎回払っている確保とゼロ埋め」を探す型ができ、P44 の V コピー、P47 の `buf_qg`/`buf_k` の fit 削減、
P48 の per-row Vec と続いていった。

### P29: 位相会計の読み方を docs に固定した (2026-10-23, `86c44535`)

**意図** — P27 の監査結果が作業レポートにしか無く、`docs/PREFILL_PATH.md` は `sl_lay_*` の ms を「その位相のコスト」として読ませる書き方のままだった。
同じ誤読が次の判断を壊すので、先に docs を直す。

**変えたもの** — `86c44535` / `docs/PREFILL_PATH.md` (+114/−2) / `docs/KPI.md` (+16/−0)。`PREFILL_PATH.md`
の冒頭に新節「位相会計の読み方 (P27 で確定)」を置き、(a) バケットは wall の分割でカーネル時間ではない、
(b) 帰属が 1 位相ずれる (11 バケット全部の境界マーカの file:line 付き)、(c) cold run の `norm_post` 水増しが 1 回目を捨てる理由、
(d) s170 の % を歴史的な値として明示、の 4 点を集約した。`KPI.md` には §5.1 として同じ 4 点の要約を置き、既存の「1 回目を捨てる」規則から参照させた。

**つまずき** — `scripts/prefill_bench.py` を全文読んだら、1 回目を捨てる処理は実装されていなかった (全 run の中央値で、`runs=2` では遅い方が中央値になる)。
ドキュメントには「捨てるのは運用規則で、スクリプト自身は捨てない」と正直に書いた。既存の数値は 1 つも増やしていない — 新しく出したのは 2 つの算術だけ
(`11,100.0 ms ÷ 896 launch = 12.39 ms/launch`、`13.21 GB / 18 GB/s = 0.734 s`)。

**分かったこと / 残った課題** — s170 の % は分母が 6 倍以上違う (63.05 s vs 現行 9.69 s / 38.79〜40.35 s) ので、
現行 HEAD の share として読めないことを本文に埋め込んだ。見つけたが直していないもの: `DENNO_PF_DBG=1 python3 scripts/prefill_bench.py …`
という env の付け方が 3 箇所に残っていた (env はサーバが起動時に 1 回読むので python 側では効かない)。これは P30 で直した。

**次に効いたこと** — この巻で「位相の ms」を引用するときの規約 (バケット名 + 条件 + 歴史的な値かどうか) がここで固定され、P41 の `[FC_KPROF]`
や P50 の tail 分解も同じ枠組みで読まれている。

### P30: PF の env をサーバ起動側へ直す (2026-10-23, `70675d32`)

**意図** — P29 が「見つけたが直していない」として残した 3 箇所を直す。`DENNO_*_=1 python3 scripts/prefill_bench.py …` の形は、
bench が HTTP クライアントにすぎないので serve に届かない。実測が空振りする原因を潰すのが目的。

**変えたもの** — `70675d32` / `docs/PREFILL_PATH.md` / `docs/KPI.md` の 2 本のみ (49 挿入 / 10 削除)。根拠を実装で固定した:
`DENNO_PF_DBG` (OnceLock、`cuda.rs:11921-11924`)、`DENNO_Q4_M_TILE` (`cuda.rs:3504-3510`)、
`DENNO_PF_OVERLAP` (`cuda.rs:11702-11705`)、`DENNO_PF_FFN_FUSED` (`cuda.rs:5446-5449`) はすべて serve 起動時に 1 回だけ読まれる。3 ブロックを「サーバを立てて health を待ち、bench を投げ、サーバのログを grep する」形に書き換えた。

**つまずき** — もう 1 つ空振りの原因があった。`[PF_DBG]` の行は serve の stderr に出るので、bench の出力を `2>&1 | grep` しても 1 行も出ない。
grep 先をサーバのログへ移した。また ①②の A/B は env が違うので **2 回サーバを立て直す**必要があり、元の形は同じ serve に 2 回投げるだけだった。

**分かったこと / 残った課題** — 数値は 1 つも変えていない (`65000 2` / `65000 3` / `DENNO_PF_OVERLAP=0` /
`DENNO_PF_FFN_FUSED=0` はそのまま)。ついでに `KPI.md` の `## 4` 重複を繰り上げ、`PREFILL_PATH.md`
の題を「s163 時点」から「s163-s175」へ直した。`scripts/prefill_bench.py` に env 転送機構は作らないと決めた (起動時に 1 回読む値なので、
転送しても立て直さない限り効かない)。実機での動作確認は未実施 (GPU 非搭載)。

**次に効いたこと** — 以後の A/B 手順が「env はサーバ起動側」で統一され、計測の空振りが「サーバを立て直していない」で説明できるようになった。

### P31: prefill チェーンの静的不変条件を GPU 無しで固定した (2026-10-23, `b5f97db8`)

**意図** — この巻の変更はどれも「launch の引数・block/grid・バッファの大きさ・位相マーカの位置」を触る。GPU を回せない状態でそれらを壊さないために、
ソースを読んで検査する静的不変条件を先に厚くする。

**変えたもの** — `b5f97db8` / `crates/denno-engine/tests/kernel_invariants.rs`
のみ (2312 → 3251 行・純追加 939 行・assert 95 箇所)。`chain_mlp_prefill_q4` / `chain_gdn_prefill_q4` / 残差常駐
(`CH_PF_RES`) / PF 位相マーカーの 4 本を追加。`.cu` / `.ptx` / プロダクトコードは無変更。

**つまずき** — 前提が 1 つ間違っていた。FFN の 3 位相は層に 1 回ではなく **1024 行タイルごとに押される** (`pf_record` の doc どおり)。
テストは「押す場所が 1 箇所」「FFN はタイルループの中」「GDN はループの外」「層 1 枚の位相列が id 昇順」を固定する形に直した。捕まえる間違いは 4 分類: (a) `Q4_M_TILE`
と `Q4_GM` の不一致・中間バッファをプロンプト全体 (m) で確保する退行 (13k 行で 2.7 GiB)・`cap` から端数併合 tail を落とす退行、
(b) chain と per-op の起動引数の取り違え (nib/sc・dk/dv) と block/grid の片側変更、(c) 残差を書く前の読み・`Some(CH_O)`
の残差加算落ち・次層へ渡すスロットの取り違え・前提検査の後置 (二重加算)、(d) 位相マーカーの二重押し/欠落/順序の入れ替え。

**分かったこと / 残った課題** — `--features cuda --lib` 321 passed / 37 ignored は基準と同一、`--test kernel_invariants`
は 15 passed (既存 11 + 新規 4)。新規 4 本は変異試験 11 通り (`cuda.rs` / `qwen35.rs` を一時的に壊して戻す) をすべて検出した。
測っていないこと: GPU 実行・アンカー・ms/token (GPU 非搭載)。

**次に効いたこと** — このテスト群が P34/P36/P39 の走査器拡張と P52 の block 変更の受け皿になり、「Rust 側の起動形状とカーネル側の前提を対で固定する」という型ができた。

### P34: smem キャスト走査器の既知の限界を検査化する (2026-10-23, `a9b91394`)

**意図** — P3-3 の smem キャスト走査器 (s104/s105) には記録済みの限界が 3 つあり、そのうち静的に決められるものをテストで閉じる。`.cu` を触らずに、
共有メモリの 4 バイト/8 バイト型キャストの整列・境界の見落としを減らすのが目的。

**変えたもの** — `a9b91394` / `crates/denno-engine/tests/kernel_invariants.rs` のみ。`.cu` / `.ptx`/ プロダクトコード / `cuda.rs` は無変更。

**つまずき** — 限界 2 (float2 の整列): 旧版は float2 も 4 バイト型として `% 4` で判定し、別名の基点オフセットも無視していた。`SmemObj` に `pre8`
(基点のバイト剰余 mod 8) と `dyn_base` を持たせ、`eval_mod` (未知の識別子は 0 倍のときだけ 0) で `PF_BK * (dh + 1)`
のような記号オフセットの剰余を求める形にした。限界 3 (間接キャスト): 旧版は `*(T*)` しか見ないので `int *p = (int*)&sA[r][c];` の形を 1 件も拾えなかった。
`indirect_casts4` を足し、ポインタ変数を登録して `*(int*)p` も同じ整列・境界判定に通す。
限界 1 (行の上限): 数値に解決できた行で「行 < 宣言行数」と「平坦端 ≤ 総要素数」を検査するようにしたが、行が threadIdx / blockIdx 由来のときは `.cu`
に静的な上限が無い。

**分かったこと / 残った課題** — ハード上限 1024 で代替すると `ty * 8 + u <= 8191` が 128 行の宣言に対して常に「越える」= 偽陽性になるため、
これは残件にした (反例: `*(const int*)&sA[ty * 8 + u][c0]`)。実カーネルの 3 件 (`corr_sm[rc * 2]` / `s_sm`
2 件) は 8 バイト錨 (8B / 32B) として緑。自己検証は新規テスト 1 本 (16 本目) で、壊した fixture 4 種 (float2 の列刻み `kk += 1`・別名基点
`sh + 1`・奇数 col の間接キャスト 2 件・宣言外の行 9) がすべて報告され、正しい側 (`kk += 2` / `sh + 2` / col 8 / 行 7) は緑のままであることを固定した。
検出件数は 9 casts / 9 symbolic-ok / 0 unresolved で不変。`--lib` 321 passed、`kernel_invariants` 16 passed。

**次に効いたこと** — 「行が記号のときの上限をどう取るか」という残件が、P36 (ホスト起動 block から取る) と P39 (grid から取る) の 2 段になった。

### P36: 走査器がホスト起動ブロック形状を取り込む (2026-10-23, `9e7de184`)

**意図** — P34 が残した最後の限界 (記号行の上限が無い) を、`cuda.rs` の起動ブロックから取って閉じる。`.cu` に静的根拠が無いなら、起動側の block 次元を根拠にすればよい。

**変えたもの** — `9e7de184` / `crates/denno-engine/tests/kernel_invariants.rs` のみ。`launch_map_from_cuda`
がコメントと文字列リテラルを潰し (行数・行番号は不変)、`module_get_function` の対応からフィールド→カーネル名を作り、全 `.launch_async(..)` を
`カーネル名 -> [(cuda.rs 行, (block.x, block.y, block.z))]` に落とす。定数は `const NAME: u32|usize|i32|i64 = EXPR;`
を `as` cast 抜きで評価して解決する。現行ツリーの結果は 61 呼び出し / 18 カーネル / 10 件がカーネル名に解決できない (`f` / `f_c` / `func`
のような関数ポインタ変数と、`module_get_function` の対応表に無いフィールド)。

**つまずき** — 規則を sound にしないと偽陽性で走査器が使えなくなる。全起動サイトが完全に静的な block で一致したときだけ `Resolved`、記号次元
(`(dh as u32, 1, 1)`)・サイト間の不一致 (`dno_matvec_rows_bt_i8` は 256 と 128 の 2 サイト)・サイト無しは `Unresolved` にした。行式は
`symbolic_row_bound` が区間演算で評価する (`threadIdx.[xyz]` は 0..block-1、`blockDim.[xyz]`、`#define` 定数、証明済みループ上限、
囲みの `if (X < N)` ガード)。違反として報告するのは `max` が到達可能と言えるときだけで、それ以外は理由付きで `row_bound_unresolved` に積み、緑にしない。

**分かったこと / 残った課題** — 副次的に走査器自身の欠陥が 2 件見つかった (どちらも修正済み・既存の assert は不変): (1) smem 名表がファイル単位だったため
`dno_gemm_q4_0` の `sA[64][32]` が `dno_gemm_i8` の `sA[128][36]` を上書きし、行数が 64 に見えていた → カーネル本体ごとの表にした。
(2) 列の錨をファイル全体で解いていたため、別カーネルのループ変数が `const int c = x * 4;` の錨を 1 に化けさせていた → 探索をカーネル本体に限定し、
ローカル束縛はキャスト行より前で最後の宣言を使う。実カーネルでは `kernels_int8.cu` の 6 件が起動ブロックから決まるようになり、
9 casts / 9 symbolic-ok / 起動ブロックからの行上限 6 件 / 判定不能 0 / 違反 0。`kernel_invariants`
は 17 passed (新規 1 本・変異 4 種)。※GDN の 4 レーン版 (`3e288aa2`) とは別物で、そちらは `50_gdn.md` を参照。

**次に効いたこと** — 「上限の根拠を外から取る」という発想が P39 の grid 取り込みにそのまま伸びた。

### P39: 走査器がホスト起動 grid を取り込み blockIdx 行を判定する (2026-10-23, `255e147f`)

**意図** — P36 に残した「行式の `blockIdx.*` は決して上限を与えない (grid 引数を読んでいない)」を閉じる。

**変えたもの** — `255e147f` / `crates/denno-engine/tests/kernel_invariants.rs` のみ。`launch_map_from_cuda` が同じ
`.launch_async(f, grid, block, shmem, args)` から grid (第 2 引数) と block (第 3 引数) の両方を読む
(`LaunchSite { line, grid, block }`)。新しい正規表現系は足していない。結果は不変で 61 呼び出し / 18 カーネル / 10 件が未解決。

**つまずき** — 実カーネルの grid は 18/18 が静的解決不能だった (`(gmx, gmy, 1)` / `(m as u32, 1, 1)` / `grid` /
`(n.div_ceil(ATTN_PF_BQ) as u32, heads as u32, 1)` / `(nv as u32, 1, 1)` — すべて変数・カーネル引数・関数呼び出し)。
したがって今日の実カーネルで grid から決まる行は 0 件で、
**blockIdx を使う行が現れた瞬間に「grid が静的に解決できない」という理由付きで unresolved になる** (推測もハード上限へのフォールバックもしない)。
一律に grid を要求すると今日緑の 6 件が赤になるので、grid を必要とするのは blockIdx を使う行式だけ、という依存駆動の門番を入れた (ローカル束縛
`const int r = blockIdx.y * BD + ...` 越しの依存も評価中の印で検出する)。

**分かったこと / 残った課題** — 規則は `max(blockIdx.d) = grid[d] - 1`、`max(threadIdx.d) = block[d] - 1`。混在項
(`blockIdx.y * BD + threadIdx.y`) は独立項の和として `hi = Σhi` を採る (blockIdx は block の番号、
threadIdx は block 内の番号で別の次元なので最大は同時に達成できる → `exact` を保ったまま違反を主張できる)。積はこの仮定を置かず、変数同士なら端点積の過大評価 +
`exact = false` にして unresolved に落とす。
自己検証は新規テスト 1 本 (18 本目) の fixture 9 種と変異 5 種 (blockIdx を常に None・grid 未解決をハード上限 1024 に落とす・grid 不一致を許す・上限を
`grid[d]` にする・積を `exact` にする、のすべてでテストが落ちる)。実カーネルの件数は 9 casts / 9 symbolic-ok / 6 件 / 判定不能 0 / 違反 0 で不変、
`kernel_invariants` は 18 passed。

**次に効いたこと** — 記録からは不明 (この後の P41 以降は計測側へ移り、走査器の拡張は続いていない)。

### P41: full_core 相の内訳をホスト側イベントで切り分ける (2026-10-23, `1068bd0d`)

**意図** — `sl_lay_full_core` のバケットは「区間内に発行したカーネル + ホストの詰まり」を 1 本にまとめているので、
attention 本体 (device) とホスト prepare / ゲート (host) のどちらが重いのか分からない。P27 の結論を踏まえ、区間の中を割る装置を入れる。

**変えたもの** — `1068bd0d` / `crates/denno-engine/src/cuda.rs` のみ (+276/−1)。`DENNO_FC_KPROF=1` のときだけ、
**既存の launch / sync の位置**にイベントを打って full 層 1 枚を prep / attn / d2h / tail の 4 区間に割り、
prefill 終了時に 1 行で出す (既定 OFF・未設定は分岐 1 回・イベントも作らない)。別イベントプール・別 env なので `DENNO_PROFILE` / `DENNO_PF_DBG`
の出力は不変。

**つまずき** — `dno_attention_prefill` は QK スコア + online softmax + PV が 1 launch に融合しているので、
qk と av には物理的に割れない。指示書の「物理的に取れなければ 2〜3 区間へ落としてよい」に従い、qk/av の代わりに attn/d2h を立てた。
区間の定義は prep = 位相 FULL_CORE → attn 発行直前 (ホスト prepare の QK-Norm/RoPE/KV 書き込み + KV ミラーの H2D + q の pinned H2D)、
attn = 発行直前 → 発行直後、d2h = 発行直後 → `chain_pf_att_inplace` の `cuMemcpyDtoHAsync` 完了、
tail = D2H 完了 → 位相 FULL_OPROJ。5 点そろわない呼び出し (CPU attend へ落ちた層など) は区間として読まない。

**分かったこと / 残った課題** — 出力は
`[FC_KPROF] prep=<ms> attn=<ms> d2h=<ms> tail=<ms> calls=<n> total=<ms> (<ms>/token)`。
同時に full_qk の D2H を pinned にする設計を `docs/FULL_QK_D2H_PINNED.md` に書いた (実装なし)。
設計メモの結論: このツリーの HEAD は P17b の 1 発連続 pinned D2H で、実機で 9.78 → 22.14 s (+772 ms/層) を測って revert された形そのもの。
だから実装は「3 発へ戻す + 宛先を pinned + プールは層ループの外で 1 回だけ確保」で、コピー回数・発行順・GEMM は変えない。
削減量は s68 の実測単価 (pageable 13.9 / pinned 14.3 GB/s) なら −26 ms/request、
PF_DBG のバケットから逆算 (6.5 GB/s) なら −1.08 s/request で**両立しない** → 実装後に 2 点計測で確定させる (未検証)。テストは
`--features cuda --lib` 321 passed / 37 ignored、`kernel_invariants` 18 passed。`[FC_KPROF]`
の実出力・GPU 実行は未計測 (GPU 非搭載)。

**次に効いたこと** — この 4 区間が P44/P46/P48/P50 の共通のものさしになった。
P44 は 4 区間をコードに写像して「prep と tail の中身はホストループ 1 本ずつ + H2D 1〜2 本ずつ」と確定させ、
P50 は同じ区間で tail を 634 ms と測ってゲートが主因だと示した。

### P44: full 層の V を scratch に戻さずホスト KV キャッシュ行へ直接 D2H する (2026-10-23, `c7b0991e`)

**意図** — P41 の 4 区間をコードに写像すると、`prep` と `tail` の中身は rayon のホストループ 1 本ずつと PCIe の H2D 1〜2 本ずつだけで、
アロケート / ゼロ埋め / 余分なコピーはほぼ残っていなかった (P12 / P17k / P20 / P28 で既に削られている)。
そのうち数値を変えずに完全に消せるものが 1 つだけ残っていた: prepare が毎層やっている V 行の純コピー。

**変えたもの** — `c7b0991e` / `crates/denno-engine/src/model/qwen35.rs` のみ (+210/−45、うちテスト 79 行)。
`pf_prepare_row` を `pf_prepare_qk_row` (q/k の QK-Norm + RoPE 本体) + v のコピーに分割し、`pf_prepare_into` の v 引数を
`Option` にした (`None` = V は既にキャッシュ行にある)。`prefill_layer_dev` の full 分岐で `fit_prefill_buf(buf_v)` をやめ、
`cache.kv_bufs(li, n)` のキャッシュ行 `[pos0..pos0+n)` を `chain_qkv_prefill_q4_dev` の `out_v` に渡す。
V は prepare で 1 ビットも変換されないので、D2H の宛先をキャッシュ行そのものにすれば `n·kv_dim·4 B` の read + write が丸ごと消える。

**つまずき** — 実測の n が指示の 65k と食い違った。区間の数字から逆算すると n≈18k〜22k が一番整合する (attn 130.8 ms/層 と d2h 24.6 ms/層 の両方から)。
どちらの読みでも構造は同じ (prep/tail は「ホストループ 1 本 + 1.5 本の H2D」) なので、推定は n=20,000 を主・n=65,000 を従として両方出した。次の計測では
`[FC_KPROF]` の行に `n=` を併記してほしい、と残した。

**分かったこと / 残った課題** — 実測 (CPU のみ・16 スレッド・4 ラウンドの最小値・27B 実シェイプ): prepare ループが 42.0 → 37.5 ms/層 (n=14,043・−10.7 %)、
197.5 → 172.9 ms/層 (n=65,000・−12.5 %)。消えたバイトは n·kv_dim·4 × 2 で 65k なら 508 MiB/層・16 層で 8.1 GiB/request、
実効 21.6 GB/s (同じ大きさのコピーで r+w 41 GB/s の環境)。推定削減は 0.02〜0.09 s/request (推測)。
値は 1 ビットも変わらない — 消したのは純コピー 1 本で、D2H が書くバイト列は同一・宛先だけが変わる。q/k の演算列は同一コードで、
launch の関数・引数・grid/block/shmem・発行順・同期の位置と回数も不変。テストは 292 → 293 (+1:
`pf_prepare_v_in_cache_matches_copy_path_bit_exact`)。
ただし 46 % はこれでは解けない: 残る支配項はホストの RoPE/QK-Norm・ホストの sigmoid ゲート・q の H2D と att の D2H/H2D で、どれも `.cu`
を触らない限り消せない。GPU 実行・アンカー md5・cold A/B は未計測。

**次に効いたこと** — 「残ったのは全部 `.cu` の仕事」と分かったことで、次は削る対象を「バイト数」から「直列化」へ移す判断ができた (P46 のチャンク化)。

### P46: full 層の att 往復をチャンク化し、D2H/H2D とゲートを重ねる (2026-10-24, `2db6448d`)

**意図** — tail (16 層合計 1,013.8 ms = 63.4 ms/層) の中身は 4 ステップだけで、コピーもアロケートも per-row Vec の churn も無かった。
残っているのは (a) ホストの sigmoid ゲート 1 本、(b) att の H2D 1 発、(c) o_proj の prologue (µs) だけ。従来は
`D2H 全体 → ctx.sync() → gate 全体 → H2D 全体` で、この間デバイスは 49.2 ms しか働いておらず、ホストは D2H の完了を丸ごと待っていた。

**変えたもの** — `2db6448d` / `crates/denno-engine/src/cuda.rs` (+330/−27) /
`crates/denno-engine/src/model/qwen35.rs` (+15/−3)。att を行ブロック単位のチャンクに割り、`D2H(k) 完了 → gate(k) → H2D(k)`
と進める。`DENNO_PF_ATT_PIPE` で制御 (未設定 = 有効・上限 8、`0` = 従来の 1 発往復、`N` = 上限 N)。
`K = clamp(ceil(bytes/128 MiB), 2, 8)` で、128 MiB 未満・1 行・フラグ OFF は K=1 = 従来経路そのもの。`gate` のシグネチャを
`FnOnce(&mut [f32])` → `FnMut(行オフセット, 行ブロック)` に変えた。イベントは grow-only・層をまたいで再利用・`CU_EVENT_DISABLE_TIMING`。

**つまずき** — ゲートは 3 パス (att read + qg のゲート半分 read + att write) で、CPU 実測 24.2 GB/s = 純 read の 81 %、`exp`
を外すと 76.1 → 53.3 ms なのでメモリ律速が 7 割・exp が 3 割。パス数を減らすには式を変えるしかなく、`/` を `*` 逆数にしたり `exp`
を近似したりするとビット一致が壊れる。したがって削れるのは直列化だけ、という判断になった。H2D が途中で失敗した場合だけ `CH_ATT` が部分更新になるので、
残りのチャンクをゲートし終えてから全体を 1 発の同期 H2D で戻す (呼び出し側のフォールバックが二重ゲートしない)。`event_sync` が使えない場合は `ctx.sync()` で代替し、
`event_create` が無い driver では K=1 に落ちる (fail closed)。

**分かったこと / 残った課題** — D2H と H2D は同じバイト数・同じ pinned なので `H ≈ D = 24.6 ms/層`、`tail = G + H` からゲート
`G ≈ 63.4 − 24.6 = 38.8 ms/層`。
2 回目の run (d2h 454.3 / tail 1,009.7) でも 34.7 / 28.4 = 55 % / 45 % で同じ比になった。
推定削減は 38.8 ms/層 → 0.62 s/request (65k・wall 35.8 s の 1.7 %)、`tail` は 63.4 → 24.6 ms/層 (−61 %)、`d2h+tail`
は 88.0 → 49.2 ms/層 (−44 %)。**すべて推定** (GPU を回せない)。
ビット一致の根拠: 全チャンクの (pinned オフセット, device オフセット, バイト数) の和が従来の 1 発と同じ領域を同じ向きに覆い、各要素は同じ式で 1 回だけ適用され、行は互いに素。
テスト 3 本 (`att_pipe_chunk_bounds_cover_rows_once` は m 10 種 × 上限 7 種、
`att_pipe_gate_is_bit_exact_across_chunk_counts` は n ∈ {1,2,3,5,8,13} × 全チャンク数でビット一致)。
`--features cuda --lib` 341 passed (+3)。

**次に効いたこと** — 「ゲートが D2H より長いので隠れ切らない」という構造が P50 で実測裏付けられ、tail の下限は転送ではなくゲートだと確定した。

### P47: full_qk の D2H 宛先を pinned ステージングへ移す (2026-10-24, `b44ba551`)

**意図** — P41 の設計メモを実装に移す。`chain_qkv_prefill_q4_dev` の q/k の D2H 宛先が呼び出し側の pageable な `Vec` だったので、
スレッドローカルの pinned プールへ移す。

**変えたもの** — `b44ba551` / `crates/denno-engine/src/cuda.rs` /
`crates/denno-engine/tests/kernel_invariants.rs`。`pin_pf_qkv` (grow-only・1 本を q/k の 2 スライスに分割) を足し、戻り値を
`PfQkView` にして消費者 (`full_attn_host_side` の prepare / attend) がそのスライスを直接読む。
Vec への写しは 0 — pinned→heap はこの計測環境で実測 4.7 GB/s なので、写すと 12.88 GB/request で +2.7 s の新規費用になる。
v の宛先は P44 のまま (ホスト KV キャッシュ行)。

**つまずき** — 削減量の読みが 4 つあり、互いに矛盾していた。s68 の実測単価 (pageable 13.9 / pinned 14.3 GB/s) なら −26 ms、
PF_DBG のバケットから逆算 (6.5 GB/s) なら −1.08 s、指示書の「~6 GB/s 上限」なら −1.25 s、「約 1.0 s @65k」なら −0.10 s。
同じ 12.88 GB に対して 927 ms〜2,147 ms と 2 倍以上開く。そこで「どれでも損をしない形」— コピー 3 発・発行順・GEMM を変えず宛先だけ変える — を採り、
削減量は実機 2 点計測で確定させることにした。P17b (3 発を 1 発に統合して +772 ms/層・revert) とは違い、3 発のまま・連結しない。

**分かったこと / 残った課題** — 数値不変の根拠: デバイス側は 1 命令も変えていない (GEMM 3 発の関数・引数・grid/block、`CH_QKV` のレイアウトと `q_off`/
`k_off`、D2H 3 発の宛先オフセットと発行順、`ctx.sync()` 1 回)。変わったのは D2H の宛先アドレスだけで、消費者が読むバイト列は従来と同一。
フォールバックは fail-closed: 上限 (既定 1 GiB・`DENNO_PF_QKV_PIN_MAX`) 超過と `cuMemHostAlloc`
失敗ではプールを 1 バイトも確保せず呼び出し側の `Vec` へ直接 D2H する従来経路に落ちる (`GPU_BROKEN` は立てない・理由を 1 回だけ stderr へ)。
`DENNO_PF_QKV_PIN_DBG=1` で pinned 経路が効いたかを 1 行確認できる。副産物として、pinned が効く層では `buf_qg`/`buf_k`
の fit (747.8 MB の mmap + ゼロ埋め) が消える。テストは `--lib` 292 passed、`--features cuda` 342 passed、
`kernel_invariants` 22 passed。GPU コードは未実行。

**次に効いたこと** — 記録からは不明 (直後の P48 は prep のホスト側へ戻り、この pinned 化の実機 A/B は残ったままになっている)。

### P48: full 層の prepare から行ごとの中間 Vec を消す (2026-10-24, `6f5f4a3c`)

**意図** — P41 の `prep` は 68 ms/層。まずこれを「ホスト prepare 36.0 ms (53 %) + 転送 32 ms (47 %)」に割り、
ホスト側に捨てられる仕事があるかを見た。転送は q の pinned H2D (345 MB) と KV ミラー H2D (k+v 115 MB) で、区間の定義上どちらも prep の中。

**変えたもの** — `6f5f4a3c` / `crates/denno-engine/src/model/qwen35.rs` の 1 ファイル。`pf_prepare_qk_row` の 1 行分を
「`rmsnorm` の戻り `Vec` を経由して `copy_from_slice`」から「宛先 (`q_r` / `k_row`) へ直接書く」形 (`rmsnorm_into`、10 行のヘルパ) へ。
1 行あたり 29 本の確保 (q 24 + k 4 + `k_i.to_vec()` 1)、28 KiB の write + 28 KiB の read、k の 2 回コピーが消える。
n=14,043 で 40.7 万本/層・16 層で 651 万本/request の確保が 0 になる。

**つまずき** — ホスト prepare 36.0 ms/層の 86 % は q の QK-Norm 1 本で、
その 25.1 ms は「同じバイト数 (qg の q 半分 345 MB read + q 345 MB write) を算術なしでコピーした時間」とほぼ同じ = 帯域の下限に張り付いている。
算術の取り分は 5.8 ms しかない。したがって「per-row Vec の再利用」の実測は norm ステップで −0.5〜−1.6 ms/層 (5 run すべて同符号・典型 −0.7)、
prepare 全体では 0 ± 0.5 ms/層 (全体比較は順序効果で分解能 ±1 ms) にとどまった。P12 が同じ変更を試して「−1.6 ms/層 しか動かない」と採らなかった記録と整合する。
2 乗和を 4 head まとめる案 (−0.9 ms) と NT store 案 (−2.8 ms だが同じ NT store で単純コピーは遅くなる・32 B 整列と `sfence`
の危険) は採らなかった。

**分かったこと / 残った課題** — 推定削減は −0.7 ms/層 → 16 層で −0.011 s/request (推定)、`prep` 68.0 → 67.3 ms/層。小さいが、
ホットパスから確保が 0 本になり中間バッファの流量も 0 になる。ビット一致の根拠: 2 乗和は同じ式・同じ順序 (f32 の逐次累積・添字昇順・再結合なし)、`inv` は同一式、
出力は同じ左結合の乗算 2 回。テスト `p48_prepare_norm_into_bit_exact_vs_legacy` が旧実装を参照実装としてビット一致を確認 (モデル実シェイプ・
`pos0 = 7`・RoPE 表あり/なし)。残った結論は「`prep` を本気で削るタスクは `.cu` を触れるタスクで、ホスト側の書き方の改善は残り 1〜4 %」という線引き。実機の
`[FC_KPROF]`・アンカー・cold A/B は未実施。

**次に効いたこと** — 同じ「帯域の下限」の測り方 (算術を全部外したコピーと比べる) が P50 のゲート 3 パスの評価に流用された。

### P50: tail を分解し、att の H2D は既に pinned だと示す (2026-10-24, `e841db48`)

**意図** — P46 の後で tail は 634 ms / 16 層 = 39.6 ms/層 (27B q4・n=14,043・1 リクエスト)。
指示は「tail が pageable な att H2D に支配されているなら P47 パターン (pinned ステージング + fail-closed) を適用せよ」だったので、まず中身を確定させる。

**変えたもの** — `e841db48` は docs のみ (`docs/P50_FULL_CORE_TAIL.md` を追加)。
**コードは 1 行も変えていない** — 置き換える対象が存在しなかった。

**つまずき** — tail の中身は 3 つだけで、アロケートは 0 件・変換はゲート 1 つ。
①sigmoid 出力ゲート (ホスト) は 3 パス = att read 345.1 + qg のゲート半列 read 345.1 + att write 345.1 = 1,035.4 MB/層、
39.6 ms は 26.1 GB/s 相当。CPU のみの実測 (16 スレッド) でも同じ 3 パスが 40.1〜40.9 ms/層・25.3〜25.8 GB/s、
16 層で 642〜654 ms = 実測 tail の 101〜103 %。②att の H2D は 345.1 MB/層で、
同じ 345.1 MB の D2H が 24.6 ms/層なので pinned 14.0 GB/s と逆算でき、
H2D も最大 24.6 ms/層 = 394 ms = tail の 62 % にしかならない。tail 全体を 1 発の転送と読むと 8.7 GB/s になり、
pageable の実測 13.9 GB/s より 1.6 倍遅い — 単一転送では説明できない。

**分かったこと / 残った課題** — att の H2D は**既に pinned から出ている**。`chain_pf_att_inplace` が D2H の宛先と H2D の転送元を同じ
`pins[li*3+2]` (345.1 MB・`cuMemHostAlloc` PORTABLE・層ごとに grow-only) にしており、フォールバックの `chain_pf_upload_att`
も pinned へ写してから H2D する。pageable な att の H2D は現行コードに存在しないので、pinned 化しても削れるバイトが 0、削減見積りは 0 ms (推定)。
P46 の「tail → 394 ms」は「ゲートが D2H に完全に隠れる」下限で、
実際はゲート (35〜40 ms/層) の方が D2H (24.6 ms/層) より長いので隠れ切らない — 前 = ゲート 35〜39 + H2D 24.6 = 60〜63 (実測 63.4)、
後 = ゲート 35 + 最後のチャンク 4.1 = 39.1 (実測 39.6) でどちらも閉じる。指示の「att は約 230 MB/層」は誤りで、`heads*dh = 6,144`
の f32 なので 345.1 MB/層 (230 MB は列数 4,096 のとき)。次に削るならゲートをデバイスへ (`.cu`+ PTX 再生成) で ≈80 ms/層 = 1.3 s/request (推定)、あるいは att の pinned を層ごとに持つのをやめる (5.52 GB → 345.1 MB、
wall は変わらない)。

**次に効いたこと** — 記録からは不明 (この後の P51 は GDN 側、P52 はカーネル側へ移っている)。

### P51: GDN beta/alpha のホスト投影を AVX2 16 レーン化する (2026-10-24, `ec7d4f31`)

**意図** — `sl_lay_gdn_proj` は 4,657 ms/request。
バケット全体を GEMM の FLOP で割ると 24.3 TFLOP/s になり「GEMM が遅い」と読まれていたが、バケットの 21 % は GEMM ではないので分母が過大なだけだった。
GEMM は 3.67〜3.73 s (同じ run の `gdn_oproj` 実測 30.3 TFLOP/s と一致) で、
律速はホスト項 — h2 の D2H 0.767 s + ホスト beta/alpha 3.567 s = 4.33 s。

**変えたもの** — `ec7d4f31` / `crates/denno-engine/src/tensor.rs`。`matvec_rows_bt_pair_into`
(GDN の beta/alpha を h2 の 1 走査で計算するホスト投影) の内側ループを AVX2 16 レーン化。`MB=16` にし、`cols == 4 && nr == 32`
の全ブロックを `pair_group_avx2` へ回す。1 グループ 16 行を 2 本の `__m256` で持ち、4 チャネル分の累積器 8 本をレジスタに置く。積と和は別命令
(`_mm256_mul_ps` + `_mm256_add_ps`、FMA に融合させない)。行数 (32 行/ブロック)・転置タイル・ブロック数・重みの再読み回数は不変で、
端数ブロックとチャネル端数は従来ループへ落ちる。aarch64 は従来経路のまま。

**つまずき** — 従来は 1 ブロック 32 行を `MB=8` レーン × `GB=4` グループで回し、累積器 `acc[g][t][l]` の添字が実行時値 (`ng` / `cols`) だったため、1 MAC ごとにスタックのロード/ストアが入って 186 GFLOP/s しか出ていなかった。
実測 (27B 実形 m=14,043 / k=5,120 / n=48+48・16 スレッド・best of 5、
HEAD と変更後を同じ機械で続けて計測): 74.32 → 30.14 ms/層 (186 → 458 GFLOP/s、2.47x)、48 層で 3.567 → 1.447 s。
内訳は転置タイルの構築 ≈14.6 ms/層 (h2 287 MB の読みが下限 ≈12 ms) + チャネル群 1 パス ≈0.63 ms × 24 パス。

**分かったこと / 残った課題** — 変更後は copy + host = 2.21 s < GEMM 3.73 s になり、バケットは GEMM 律速に戻る。
推定 4.66 → ≈4.05 s (−0.6 s/request、wall 38.05 s の −1.6 %)、床は GEMM 3.73 s。数値不変の根拠: 各出力要素は `acc = 0` から `d`
昇順のまま、AVX2 版は `l` (行) 方向にしかレーンを張らず、積と和が別命令なので丸めは 2 回で従来のスカラー計算と同一。変わるのは (i,j) を処理する順序だけ。テストは `--lib`
293 passed、`--release --features cuda --lib` 343 passed、`kernel_invariants` 22 passed。実機の `[PF_DBG]` 1 run (`sl_lay_gdn_proj` が ≈4.0 s に落ち、`sl_lay_gdn_scan` が動かないこと) は単独では未計測。
統合ビルド (P47/P48/P50/P51/P52) では `GPU_VERIFY.sh` がアンカー 3/3 MATCH、wide 8 問 × 150 token の md5 も変更前と同一、
65k wall は 34.95 s (統合前 38.05 s)。ただし P51 単独の寄与は切り分けていない (P52 が −1.8〜2.2 s、残りは P47/P48/P50 の合算)。

**次に効いたこと** — 記録からは不明 (GDN の数式面は `50_gdn.md` を参照)。残る床が GEMM 3.73 s になったので、次に効くのは `DENNO_Q4_M_TILE=2048`
の末尾波 A/B か beta/alpha のデバイス化、という順序が整理された。

### P52: q4 GEMM の per-thread タイルを 8x8 に広げる (2026-10-24, `b6e3fa27`)

**意図** — prefill の GEMM は `dno_gemm_q4_0` がほぼ全部の位相を占める。1 block = A 64 行 × B 128 列 × k 全体、
smem 25 KB という形は変えず、1 thread が持つタイルだけを広げて共有メモリのロード回数を減らす。

**変えたもの** — `b6e3fa27` / `crates/denno-engine/kernels/kernels_int8.cu` /
`crates/denno-engine/src/cuda.rs` / `crates/denno-engine/tests/kernel_invariants.rs`。
block を (16,16) = 256 thread から (16,8) = 128 thread へ、
1 thread の担当を 4 行 × 8 列 (32 acc) から 8 行 × 8 列 (64 acc) へ。
1 要素あたりの共有メモリロードが 32 FFMA / 12 load から 64 FFMA / 16 load に下がる (2.7 → 4.0)。B タイルの展開は `tid`
がそのまま列 (0..127) になり、1 語の下位 4 bit 群が要素 `wi*4+e`、上位 4 bit 群が要素 `16+wi*4+e` という旧 2 分割版と同じ対応。Rust 側は
`q4_dispatch_tile` と `chain_gemm_q4_pf` の 2 箇所の block を `(16, 8, 1)` へ。PTX は続く `4d9c28c7`
で再生成 (nvcc 12.9 / compute_75 / `-fmad=false -O3`、PTX の差分は +5,535 行 / −2,863 行)。

**つまずき** — block の変更は Rust とカーネルの対で効くので、片側だけ直すと出力の行/列が二重に覆われるか覆われない行が出る。`kernel_invariants` は「block が
`(16,8,1)` であること」と「カーネルが `ty * 16 + tx`・`const int col = tid;`・`acc[8][8]` を前提にしていること」を対で固定する形に更新した。
grid 式 (`m.div_ceil(64)`, `n.div_ceil(128)`) と `Q4_GM=64` / `Q4_GN=128` / smem は不変。

**分かったこと / 残った課題** — 累積順は不変 (`b` 昇順 → `c` 昇順、各出力要素は 1 回だけ書かれ、行は互いに素) なのでビット一致は保たれる — 各出力要素の
`acc[r][j] += av[r] * bv[j]` の順序は 4 行版と 8 行版で同じで、増えたのは `r` の範囲だけ。レジスタは acc 64 本 + `av[8]` + `bv[8]`
に増えるので occupancy への影響も実機で確認が要る。テストは `--lib` / `--features cuda` / `kernel_invariants` のいずれも緑。

**実測 (27B Q4_0・65k 級・cold・run 1 回目は重みのアップロードを含むため破棄・3 run の中央値)**:

| 腕 | 65k wall | 備考 |
|---|---|---|
| 変更前 (canonical) | 38.44 s | 配備済みバイナリ |
| 変更前 (再ビルド) | 38.17 s | 同一ソースから再ビルド。ビルド間のばらつき ±0.3 s |
| **P52** | **36.26 s** | 最小 36.09 s |
| 変更前 (再ビルド・2 本目) | 37.93 s | 同一腕を交互に測り直した値 |

4 腕を交互に並べた差は **−1.8〜2.2 s (約 −5%)**。wide 8 問 × 150 token (temp 0) の出力 md5 は
変更前と P52 で同一 (`e36644365c6746b5ee1a06720d5ad4a1`)、`GPU_VERIFY.sh` のアンカー 3 本も 3/3 MATCH
なので、数値経路は動いていない。

**次に効いたこと** — この巻の最後のエントリ。残っているのは `full_oproj` 15.9 / `full_qk` 15.7 TFLOP/s
(他の位相は 30 前後) の 2.55 s の塊で、天井まで −1.3 s の余地がある。タイルではなく別の理由 (転置・帯域) を疑う段階。

---

## 棄却カタログ

当時は妥当に見えたが後で捨てた判断。本文の該当エントリへの参照だけを置く。

### prefill を CPU のままにする線引き (P17b `197a06c` / P17i `f2c18315`)
- 何を試したか: int8 線形を GPU に載せる範囲を決める。GEMM は `m > max_m()` かつ `m >= 256`、GEMV は `m <= 16`、間の 17〜255 は CPU。
- 実測値: 1.5B int8・t≈2810 の cold が約 100 s で、prefill の 83% が線形層の CPU GEMM (gate_up 47 s・down 29 s)。
- なぜ捨てたか: 小 m では層あたりの同期が CPU GEMM より高くつくため、17〜255 は CPU のままにした (GPU に寄せる案は採らなかった)。
- 記録: 本文 P17b / P17i。

### sampler カーネル (P17j で見送り・`cededce6`)
- 何を試したか: 手順書の次候補だった sampler のカーネル化。
- 実測値: 実測 0.001%。13,252 token cold の TOTAL 395.8 s のうち attn_core が 334 s (84.5%) で先に効く。
- なぜ捨てたか: 比率が 0 に等しく、先に attention を載せるべきだった。
- 記録: 本文 P17j。

### T5a の opt-in ゲート (既定 2.85x 退行・`083218d6` → revert `bbe82d75`)
- 何を試したか: 「prefill の投影が全部 CPU (30.1 s)」というプロファイルを根拠に `DENNO_GPU_PREFILL_GEMM` を追加。
- 実測値: 既定 202〜203 s (3 回)・PF_BENCH 200.8 s に対し、ゲート on は 70.7 / 65.3 s・PF_BENCH 71.4 s。T4 の `1e51741` は PF_BENCH 70.9 s。
- なぜ捨てたか: 前提が偽。`via` 列が `on_gpu` を渡さず常に `cpu` と記帳していただけで、既に GEMM 経路に入っていた。差し替え版は `maybe_gpu_linear` を迂回して本物の CPU フォールバックになっていた。
- 記録: 本文 T5a 撤回。crates は `1e51741` とバイト一致に戻し、計測の修正 (`via` 列) と再発防止テスト 4 本だけを残した。

### int8 GEMM の thread 数と smem ストライドの選び方 (s74 `e0ea110b` → s75 `5bca9b7e`)
- 何を試したか: per-thread タイルを 8x8 → 8x16 に広げ、1 ブロックを 64 thread (16x4) にした。
- 実測値: 実機で即死 (`cuMemcpyDtoHAsync` error 716) → 恒久 CPU フォールバックで PF wall 171.68 s / 33.1 tok/s (ベースライン 9.65 s / 589 tok/s の 17.8 倍遅い)。修正後は cold で 14.222 → 13.407 s (−5.7%)。
- なぜ捨てたか: 64 thread × 8 行 = 32 行しか覆わず、128 行タイルの 32..127 行が誰にも書かれない (出力の約 75% が未初期化)。加えて smem ストライド 33 が 4 バイト境界に揃わない。タイルを広げる判断自体は残し、thread 数とストライドの選び方を s75 で捨てた (256 thread・ストライド 36)。
- 記録: 本文 s74 / s75。CPU テストは緑のままで、このクラスの欠陥は GPU か静的検査でしか捕まらない。

### rmsnorm を残差加算と融合する案 (s84 で見送り・`66698f68`)
- 何を試したか: T8 の問い。判断線は 0.25 s。
- 実測値: 正規化は 9B の 5,866 tok × 4,096 次元 × 16 層 = 1.54 GB、読み書きで約 3.1 GB。1 TB/s で約 3 ms、100 GB/s でも約 31 ms (いずれも換算・推定)。13.2 s の prefill に対して 0.25% 以下。
- なぜ捨てたか: 0.25 s の判断線に桁で届かない。「rmsnorm 0.4 s」という以前の値は、層ブロック全体を親スコープで包んだ計測の産物だった。
- 記録: 本文 s84。step 2 (融合・トークン方向のバッチ化) は実施しないと決めた。

### q4 GEMM の行タイルを投影の n から選ぶ案 (P21・commit なし)
- 何を試したか: `q4_pf_tile` / `pf_gdn_tile` で行タイルを投影の n から選ぶ (埋めの下限と L2 上限の式)。
- 実測値: `sl_lay_gdn_proj` 18.0 s の内訳は host beta/alpha 9.75 s + h2 の D2H 4.9 s + GEMM 3.67 s = 18.3 s で、実測 18,015.8 ms との差は 1.7%。81% は GEMM ではなかった。
- なぜ捨てたか: 実装は commit されず、次の P25 の STEP 0 で WIP のまま破棄された (`git restore` 1 ファイル)。残ったのは「18.0 s は P13 以前の測定」という注記 (`86c44535`)。
- 記録: 本文 P21。

### `sl_lay_ffn_down` に 1 点入れる案 (P25・commit なし)
- 何を試したか: 「host または copy が位相の 20% 以上なら 1 点だけ入れる」規則に当たるかの確認。
- 実測値: 区間内は `dno_gemm_q4_0` の launch 1 発だけで、H2D 0 B・D2H 0 B・`ctx.sync()` 0 回・alloc 0 回・量子化 0 回。
- なぜ捨てたか: 条件不成立なのでコード 0 行・commit なし。
- 記録: 本文 P25。

### attention と o_proj の間の同期を落とす案 (P20 で前提を否定・`5f222949`)
- 何を試したか: full 層の `sl_lay_full_core` から同期を外せるかの判定 (heads のホストループと、attention と o_proj の間の同期)。
- 実測値: heads はホストでループしていない (attention は既に 1 発/層、`grid.y = heads = 24`)。同期は 1 回だけで、att 345 MB を読んで sigmoid ゲートを掛けるための D2H 完了待ち。
- なぜ捨てたか: 落とすには n トークン版のゲートカーネル (`.cu`) が要る。同期の削除は採らず、P44 / P46 で「往復そのものを減らす」側から攻めた。
- 記録: 本文 P20。

### prepare と sigmoid ゲートのホスト費用を消す案 (P28・`3c383a7f`)
- 何を試したか: `sl_lay_full_core` のホスト側 (prepare 44.9 ms + sigmoid ゲート 45.4 ms = 90.3 ms/層) を消せるか。
- 実測値: n=14,043・16 full 層で 1.45 s/リクエスト。位相は device bound で、attention カーネルが支配的 (転送 1.15 GB/層)。
- なぜ捨てたか: どちらも必要なデータ移動で、消すにはゲートのデバイス化 (`.cu`) が要る。代わりに毎リクエスト 2 × 345 MB の無駄な確保 (mmap + ゼロ埋め) を使用点へ移した。
- 記録: 本文 P28。

### att の H2D を pinned 化する案 (P50 で対象なし・`e841db48`)
- 何を試したか: 「tail が pageable な att H2D に支配されているなら P47 パターンを適用せよ」という指示の検証。
- 実測値: P46 後の tail は 634 ms / 16 層 = 39.6 ms/層 (27B q4・n=14,043・1 リクエスト)。
- なぜ捨てたか: 置き換える対象が存在しなかった。コードは 1 行も変えず、docs だけを追加した。
- 記録: 本文 P50。
