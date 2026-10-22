# prefill 経路の構造と高速化 (s163 時点)

## 結論

prefill は **入力トークン全部を 1 回の forward で処理している**
(`Qwen35::forward` → `tokens.len() > 1` → `forward_prefill`)。遅さの原因は
バッチ化の欠如ではなく、**線形層 1 呼び出しごとのホスト往復**だった。

## 旧経路の往復構造

`q4_dispatch` は m を `Q4_M_TILE` 行のタイルに割り、タイルごとに
H2D → GEMM → D2H → sync を繰り返す (`SCRATCH` が thread_local かつ
grow-only のため、プロンプト長そのままの m を渡すと VRAM が積み上がるのが
理由)。FFN は 1 層あたり 5 往復:

| 段 | 転送 | サイズ (m=13000, e=5120, ff=17408) |
|---|---|---|
| h2 → gate/up | H2D | 266 MB |
| gate → host | D2H | 905 MB |
| up → host | D2H | 905 MB |
| hh → down | H2D | 905 MB |
| down → host | D2H | 266 MB |

13k token の prefill 161.4 s のうち `sl_lin` が **113.5 s (70%)**、
802 呼び出し = 141 ms/回。カーネル自体は 3,161 token の nsys で 7.9 s
(全体 47.7 s) に過ぎず、残りは PCIe とホスト側の待ちだった。

## 変更 (commit 125117c)

- `chain_mlp_prefill_q4` を追加 (`cuda.rs`)。`chain_mlp_prefill` (int8 版) と
  同じ形で、gate/up → swiglu → down をデバイス上で回し、D2H は down の 1 回だけ。
  行タイル 1024 で処理するので中間バッファはタイル分のみ。
- `Q4_M_TILE` 256 → 1024 (sync 回数が 1/4)。

## 変更 (s164・GDN 層)

- `chain_gdn_prefill_q4` を追加 (`cuda.rs`)。GDN 層を
  qkv/z 投影 → GDN chunk scan (3 launch) → o_proj までデバイス常駐で通し、
  D2H は o_proj の 1 回だけにする。qkv/z の GEMM 出力は GDN スクラッチ
  (`GDN_SCRATCH.q` / `.z`) へ直接書き、y もデバイス上に残す。
- 入力の出所だけを差し替える `GdnScanSrc` (Host / Dev) を導入。カーネル・
  grid・引数・chunk 分割 `plan(n)` は従来と同一。
- beta/alpha は従来どおりホストで計算した値を使う (n*nv*4 x2 = 5 MB なので
  往復は消していない)。GDN 状態の契約 (`gdn_state_begin` / `commit` /
  `gdn_host_writeback`) も不変。
- 発動条件: qkv_w / gate_w / out_w がすべて q4・形が一致・
  `n >= GDN_CHUNK_MIN_TOKENS` (2048)・`dk == dv`・`d_conv == 4`・
  端数タイルが `max_m()` 超 (per-op 経路が GEMV に落ちる形では発動しない)。
  それ以外は None → 従来経路へそのままフォールバック。

ビット一致の根拠:
- q4 GEMM は per-op 経路と**同一カーネル・同一グリッド・同一引数**
  (`dno_gemm_q4_0`)。行ごとに独立なので m の分割は 1 行の累積順を変えない。
- swiglu は `dno_swiglu` (CPU の `g/(1+exp(-g))*u` と同一演算列)。
- GDN スキャンは **n を分割しない** (chunk 分割は状態の持ち越しで決まるため、
  行タイルに割ると `plan(n)` が変わって数値が変わる)。

## 変更 (s165・残差ストリームの常駐 / P2)

- 残差 `h` を `CH_PF_RES` (デバイス) に置いたまま層を通す
  (`Qwen35::forward_prefill` → `prefill_layer_dev`)。層の中は
  `rmsnorm (デバイス) → 投影 (デバイス) → 残差加算 (次の rmsnorm の proj 引数)`。
- ホストが契約上必要とする転送だけが残る:
  GDN 層 = h2 の D2H 1 回 (beta/alpha のホスト計算) / full attention 層 =
  h2 の D2H 1 回 (q/k/v はホスト計算・KV はホストが正) + attention 出力の H2D 1 回 /
  FFN = 0 回。q/k/v/o の per-op 往復 (full 層) と FFN の 2 往復が消える。
- 単独の残差 add カーネルは PTX に無いので、加算は `dno_add_rmsnorm` の
  `proj` 引数に融合し、層末の加算を次の rmsnorm へ繰り延べる (追加カーネル 0)。
- `chain_attention_prefill` の `CH_ATT` は sigmoid ゲート前なので、o_proj の
  デバイス実行には**ホストでゲート済みの att** を 1 回だけ上げる。
- 発動条件は q4 全層 + `n > max_m()` + `e%32==0` + 端数タイル + 診断 OFF。
  詳しい契約・ビット一致の根拠・未計測の期待値は `REPORT_P2.md`。
- **未計測** (main が `scripts/prefill_bench.py` で測る)。アンカー一致も未確認。

## s166 (P4): beta/alpha のデバイス化は `dno_matvec_f32` では不可 (実装せず)

GDN 層に残っている h2 の D2H は「beta/alpha をホストで計算するため」だけなので、
decode の C4 (`chain_q35_ctl_f32`) と同じ形でデバイス計算に置き換えられれば
往復 12.6 GB (13k) が消える。**しかし既存 `dno_matvec_f32` では赤字になる。**

- カーネルは **m=1 専用**: `grid=(n,1,1)` / `block=(128,1,1)` で 1 launch = 入力ベクトル 1 本。
  行 (トークン) 方向のバッチ軸が無いので、prefill の n 行には **n launch** が要る
  (13k で 2 投影 × 48 層 × 12,850 = **1,233,600 launch**)。
- 1 launch の GPU 時間は k=e=5120 の**依存 FMA 連鎖**で決まる (PTX: 4 倍展開・
  `add.rn.f32` の直列鎖 → 4 cyc/element ≈ **12 µs**)。同一ストリームなので重ならない。
  → **+15 s (13k) / +3.6 s (3,161)**。
- 消せる現行コストの上限は **1.1〜1.7 s (13k)** (h2 D2H 12.6 GB + ホスト proj 0.6 s)。
  しかも両者とも n に線形で per-row 比は **device 1.15 ms vs host 0.13 ms (約 9 倍)** 一定
  → **勝てる n は存在しない** (「launch が少ないときだけ発動」のゲートは作れない)。
- 転置配置 (`h`=重み行 / `w`=h 行列) にすると launch は nv=48 まで落ちるが、出力が
  `[nv][n]` になりスキャンの `[n][nv]` と合わない上、1 launch が h 行列 263 MB を再読する。

必要なのは**行バッチ版カーネル** (1 出力要素 = 1 スレッド・d 昇順逐次和のまま
`grid=(nv, ceil(m/128))` にし、`out[r*n+j]` へ書く) で、`.cu` 追加 + PTX 再生成が前提
(見積り ≈10 ms/全層)。詳細・先行研究 (s62 / s78 / s79 / s96) は `REPORT_P4.md`。
**未計測** — 数字は静的解析による見積り。

## s167 (P6): h2 の D2H を qkv/z GEMM と重ねる (コピーストリーム・既定 OFF)

beta/alpha が要るのは chunk scan からで、その手前の qkv/z GEMM は h2 (デバイス) だけに
依存する。よって GDN 層を 2 位相に割り、

```
rmsnorm → { D2H h2 (専用コピーストリーム)  ||  qkv/z GEMM (既定ストリーム) }
        → join (ホストが h2 を読む直前の唯一の同期点) → beta/alpha (ホスト)
        → beta/alpha H2D → chunk scan → o_proj
```

の形にした (`chain_gdn_prefill_q4_dev_ovl`)。数値は不変 — 同じバイト列を同じ
バッファから戻し、ホスト計算の関数・引数・順序、カーネル・grid・引数・chunk 分割・
状態契約は 1 つも変えない (コピーと GEMM はどちらも `CH_PFH` を読むだけで独立)。

- コピーストリームは **非ブロッキング** (`CU_STREAM_NON_BLOCKING`) で作る。blocking で
  作ると legacy (既定) ストリームと暗黙に同期し、GEMM がコピー完了を待ってしまう。
  順序はイベントで明示する (既定ストリームで rmsnorm 完了 → コピーストリームが待つ →
  D2H → 完了イベント → ホストが join)。
- ゲートは `DENNO_PF_OVERLAP=1` (**既定 OFF**)。OFF・driver に `cuStreamCreate` 等が
  無い・stream/event/pinned の作成失敗は、従来と同一の順序 (同期 D2H → ホスト →
  GEMM/scan/o_proj) を通る。フラグ OFF ではストリームを 1 本も作らない。
- 効果の上限は「コピー + ホスト作業」を GEMM の裏へ移せる分 = P4 の上限
  (13k で 1.1〜1.7 s) から GEMM 時間を差し引いた分。GEMM が見積り (≈54 ms/層) より
  速いと隠し切れない。**未計測** (実機の A/B は main が行う)。
- 契約テスト: `cuda::p6_overlap_tests` (GPU 不要)。

## s170 (P9): 残っているホスト往復の棚卸しと次の常駐段の設計 (**設計のみ・未実装**)

27B q4 (e=5120, ff=17408, 64 層 = 48 GDN + 16 full, heads/kv/dh = 24/4/256,
att=6144, q 出力幅 12288, kv_dim=1024, vocab 248,320 — `config.json` を読んだ値) で、
残差常駐経路 (`res_dev`) に**まだ残っている**転送を全部数え、次の段の候補を 3 つに絞った。
**コード・`.cu`・`.ptx` は 1 バイトも変えていない。** 詳細な算術は `REPORT_P9.md`。

### 残っている往復 (12,850 token / 1 リクエスト)

| 層 | サイト | 向き | B/token/層 |
|---|---|---|---|
| GDN ×48 | `qwen35.rs:2105` → `cuda.rs:11101` `chain_download_at(CH_PFH)` (+`cuda.rs:3993` sync) | D2H | 20,480 |
| GDN ×48 | `cuda.rs:10361-10362` beta/alpha `copy_from` (pageable sync) | H2D | 384 |
| full ×16 | `cuda.rs:11473/11482/11491` q (fused gate 込み) / k / v | D2H | 49,152 + 4,096 + 4,096 |
| full ×16 | `cuda.rs:5111/5112` KV ミラー k / v (`copy_from_at`) | H2D | 8,192 |
| full ×16 | `cuda.rs:5133` q (norm+rope 後) | H2D | 24,576 |
| full ×16 | `qwen35.rs:977` → `cuda.rs:3925/3935` att (+sync) | D2H | 24,576 |
| full ×16 | `qwen35.rs:2195` → `cuda.rs:11400` att (ゲート後) | H2D | 24,576 |
| 全体 | `qwen35.rs:1493` → `cuda.rs:11356` embed | H2D | 20,480 (req 全体) |
| 末尾 | `qwen35.rs:1954/1957` 残差最終行 + pending、`qwen35.rs:1975` logits | D2H/H2D | 計 ≈1 MB (req 全体) |

合計 **41.8 GB @12,850 / 10.3 GB @3,161** (ESTIMATED — 形状からの算術)。
PCIe 12〜25 GB/s で **1.7〜3.5 s / 0.4〜0.9 s** = wall (56.3〜61.8 s / 13.86 s) の **2.7〜6.2 %**。
同期点は 1 リクエスト **227** (往復由来 83 + GDN chunk scan の 3 pass 由来 144)。

向きべつでは **D2H 29.5 GB / H2D 12.3 GB @12,850** (D2H が 71 %。3,161 では 7.25 / 3.02 GB)。
コピー発行数は **1 層あたり GDN 5** (h2 D2H 1 + beta/alpha H2D 2 + norm w H2D 2)、
**full 10** (qkv D2H 3 + ミラー H2D 2 + q H2D 1 + att D2H 1 + att H2D 1 + norm w H2D 2)、
**リクエスト全体で 403** (P2 前の 802 chain call の半分)。norm w の H2D は 20,480 B/call・
2 call/層 = 2.6 MB/req で合計には入れていない (無視できる)。

### 壁の 3 分解 (12,850, ESTIMATED)

```
wall 56.3 s (MEASURED) − PCIe 3.5 s (@12GB/s) − ホスト計算 8.9 s = カーネル 43.9 s
カーネル仕事量 24.33 GMAC/token → 312.6 GMAC = 625 TFLOP → 14.2 TFLOPS 実効
```

つまり **往復のバイトはもう壁ではない (≤6 %)**。残り 16 % は往復が強制している
**ホスト計算**で、その 91 % が 1 本 — GDN の beta/alpha 制御投影
(`matvec_rows_bt_into`, F32, k=5120, n=48)。**実測 84 ms/call × 96 call = 8.06 s @12,850**
(この開発コンテナの CPU。P4 の「ホスト proj 0.6 s」は再現しない = 13 倍の過小評価)。

### 消せない往復 (KV がホスト正)

`KvCache` (ホスト) が full 層 K/V の唯一の正で、decode / prefix snapshot / CPU attend が読む。
`with_mirror` (`cuda.rs:5106`) のミラーは**ホストの写し**。よって:

- **full 層の k/v D2H (post-RoPE) は消せない** — 8,192 B/token/層 = 1.68 GB (0.07〜0.14 s)。
  ホストで prepare するなら k/v 投影出力 57,344 B/token を戻すことになり、もっと高い。
- embed H2D 263 MB (埋め込み表がホスト RAM 5.1 GB)、末尾の残差最終行 40 KB + sync、
  logits D2H 993 KB も残る。**下限 1.94 GB = 0.08〜0.16 s。**

**q / o / att の往復は KV 権威と無関係** — q は KV に触らず、o_proj の出力は
デバイス常駐の残差に落ちるだけ。ホストに降ろしている理由は
「prepare (QK-Norm+RoPE) がホスト」「sigmoid 出力ゲートがホスト (`dno_attn_gate_mul` は
1 token 専用)」の 2 つだけなので、**KV 権威をデバイスへ移さずに消せる**:

- (a) q の prepare をデバイスへ (k/v はホストのまま) → q はデバイスから出ない。
  KV の向き・権威・`with_mirror` の意味は完全不変。**リスク最小**。
- (b) att ゲートをデバイスへ → att はデバイスから出ない。
- (c) prepare 全体をデバイスへ → ミラーが「prefill の producer」になり、ホストは
  ミラーから k/v 行を D2H して自分の KV に書く (権威はホストのまま・向きが逆)。
  **この場合だけ `have` / `epoch` の帳簿を「D2H 完了まで valid にしない」に変える必要**
  (途中失敗で欠けた行を valid と思うと次の decode が古い K/V を読む)。権威の移動ではない。

### 候補 (12,850 の期待秒 / リスク 順)

**① GDN の beta/alpha をデバイスで計算する (行バッチ版カーネル) — ≈8.4〜9.4 s (15〜17 %), リスク中**

- 継ぎ目: `cuda.rs:10951` の `host_ctrl` (`cuda.rs:11041` / `11102`) を
  `chain_gdn_ctrl_dev` に差し替え、`qwen35.rs:2158-2171` の closure (buf_h 写し + 2 回の
  `apply_bt_rows_into`) を丸ごと落とす。h2 の D2H・`pin_h2`・join も不要になる。
- 内訳: ホスト計算 8.06 s (MEASURED 単価) + D2H 12.87 GB (0.51〜1.07 s) + buf_h 写し 0.30 s
  + 48 sync。
- カーネル: `dno_matvec_f32` の行バッチ版を `kernels_elem.cu` に 1 本追加
  (1 出力 = 1 スレッド、`d` 昇順逐次 f32 和、`out[r*nv+j]`、grid=(nv, ceil(n/BLK)))。
- **ビット一致**: (i,j) ごとの `d` 昇順逐次和は `matvec_rows_bt_small_n_impl` と同一で、
  `-fmad=false` なら積と和が別丸め。既存 `dno_matvec_f32` はまさにその契約で
  `cuda_matvec_f32_lmhead_bit_exact_vs_cpu` が CPU 完全一致を固定している。行バッチ版は
  **添字だけ**変えるので 96 call 全部がビット一致。
- **VRAM +0** (入力は既存 `CH_PFH` 263 MB、出力は既存 `GDN_SCRATCH.beta/.alpha` 各 2.5 MB。
  どちらも最初の launch より前に ensure 済み)。重みを転置して上げる場合だけ +47 MB。
- 失敗モード: PTX 未再生成 → `None` を返して**今日の経路**へ落ちる (安全側)。
  n サイズの新バッファは 1 本も作らない (P4-6/P4-7 の教訓: 層ループ内の grow は
  発行済み launch の読み先を解放しうる)。副作用として P6 の 2 位相分割は q4 経路で
  死にコードになる (フラグを残すか撤去するかは operator 判断)。

**② att の sigmoid ゲートをデバイスで行う (n 行版) — ≈0.7〜1.2 s (1.2〜2.1 %), リスク低〜中**

- 継ぎ目: `qwen35.rs:2189-2197`。`full_attn_host_side` が `att_done` を返したときだけ
  D2H + ホストゲート + `chain_pf_upload_att` を飛ばし、`chain_pf_gate_att` 1 発 →
  `chain_oproj_prefill_q4`。`att_done == false` (CPU attend) は今日の経路のまま。
- カーネル: `kernels_elem.cu:742` の `dno_attn_gate_mul` の n 行版。式は既存そのまま
  `__fdiv_rn(*av, __fadd_rn(1.0f, dno_expf(-g)))`、添字だけ
  att = `CH_ATT + r*att + h*dh + d` / gate = `CH_QKV + r*qcols + h*2*dh + dh + d`。
  これは新規の発想ではなく、`qwen35.rs:925-927` のコメントが
  「`dno_attn_gate_mul` は 1 トークン専用なので D2H 後にホストで掛ける (C3 でデバイス連結する
  際に n トークン版カーネルを追加する)」と既に予告している作業そのもの。
- **ビット一致**: `dno_expf` は glibc `e_expf.c` の忠実な再実装で、ホストの `f32::exp` と
  ビット一致することが実機確認済み (`kernels_elem.cu:6-13`)。ホスト式は
  `*av /= 1.0 + (-gate[d]).exp()` (`qwen35.rs:1003`) = `div.rn(av, add.rn(1, exp))` で、
  既存カーネルの `__fdiv_rn` + `__fadd_rn` と同一。行を増やすだけなので 1 要素の演算列は不変。
  **`__fdividef` や逆数乗算に置き換えない** (丸めが変わる)。
- **VRAM +0** (`CH_ATT` を in-place。ゲートは `CH_QKV` の q 領域後半を読むだけで、
  そこは `chain_qkv_prefill_q4_dev` 以降だれも書かない)。
- 失敗モード: PTX 未生成 → **ゲート未適用のまま o_proj が走ると静かに壊れる**ので、
  `None` は必ず「D2H + ホストゲート + H2D」に落とす。`dno_expf` の |x|≥88 分岐を端折らない。
  なお att の D2H は `cuda.rs:3920` で 316 MB の Vec を層ごとに新規確保する形なので、
  実コストはバイト見積りより上。

**③ q の QK-Norm + RoPE をデバイスで行う — ≈0.6〜1.3 s (1.1〜2.2 %), リスク中**

- 継ぎ目: `cuda.rs:11473` の q D2H を落とし、`cuda.rs:5133` の q H2D を
  「デバイスの q を指す」に変える (カーネル・引数は同一、ポインタだけ差し替え)。
- カーネル: head 単位 rmsnorm (1 head = 1 block、和は `d` 昇順逐次 = `dno_add_rmsnorm` と
  同じ契約) + RoPE。**cos/sin 表はホストで 1 回作って上げる** (3.3 MB @12,850) ので、
  デバイス側は `x1*c - x2*s` / `x1*s + x2*c` の 2 回丸めだけ。
- **ビット一致**: 中〜強。rmsnorm は和の順序と `sqrt.rn`+`div.rn` をホストと同一にすれば一致
  (`rsqrt.approx` 不可)。RoPE は表をホストから持つので超越関数の一致問題が消える。
  危険は「q の rmsnorm が head 単位 (cols=256) で、既存 `dno_add_rmsnorm` が行単位
  (cols=e)」の点 — cols=256 で呼べるか (shmem/block) を先に確認する。
- **VRAM**: q 専用バッファなら **+316 MB** (@12,850)。`CH_QKV` に in-place で書くなら +0 だが
  attention カーネルに「q の head ストライド = 2·dh」を渡す引数追加が要る。
- 失敗モード: 表を層ごとに上げると 16 × 3.3 MB の無駄 → リクエスト先頭で 1 回。
  q_dev を層ループ内で grow させない。切り分けは `DENNO_DEBUG_LAYER_CHKSUM=1` の `L?? attn` 行。

**同時に拾える小物 (候補ではない)**: 末尾の `chain_download_at` 2 回 (`qwen35.rs:1954/1957`)
= sync 2 回を 1 回に。RoPE の `theta.powf(-2d/n_rot)` を (pos,d) ごとに計算し直している
(`qwen35.rs:204`、head ループの内側) ので表を 1 回作る (0.10 s @12,850, ESTIMATED。`.cu` 不要)。

### 前提の確かめ方 (この順序は「beta/alpha が 8 s」に依存している)

```
DENNO_PF_DBG=1 python3 scripts/prefill_bench.py 65000 2 2>&1 | grep -E "PF_DBG|sl_lay"
```

`sl_lay_gdn_proj` (`REPORT_P8.md:45`) は「h2 の D2H/join + ホスト beta/alpha + qkv/z GEMM」を
含むので、**候補①の上限がそのままこの行に出る**。予測 (このコンテナの CPU なら) は
168 ms/層。60 ms 前後なら P4 の 0.6 s が正しく、候補①は ≈1.2 s に縮むので ②③ を先にする。
`sl_lay_full_core` が 100 ms/層を超えるならホスト prepare が主因 → ③を先にする。

## 計測プロトコル (prefill)

**同じプロンプトの再送は prefix cache に当たるため、cold 計測には使えない。**
毎回ユニークな前置きを付けて送る (`scripts/prefill_bench.py`)。

```
python3 scripts/prefill_bench.py 16000 3   # 3,161 token 相当
python3 scripts/prefill_bench.py 65000 2   # 12,850 token 相当
```

判定は 3 回の中央値。起動直後の 1 回は重みの遅延アップロードを含むので捨てる。

## 実測 (8095・27B q4・cold)

| 条件 | 旧 | 新 | 倍率 |
|---|---|---|---|
| 3,161 token | 40.10 s | 24.21 s | 1.66x |
| 12,850 token | 162.3 s | 96.9〜109.1 s | 1.5〜1.7x |
| decode | 32.4 ms/token | 31.8〜35.6 ms/token | 変化なし |

`temperature 0` の生成 3 問は旧バイナリと 176 バイト完全一致。

## 残り

> s170 (P9) で下の 1・2 を定量化した (上の s170 節)。往復バイトは wall の 2.7〜6.2 % で、
> 残る 16 % は beta/alpha のホスト計算。次の一手は候補① (GDN 制御投影のデバイス化)。

1. ~~qkv / o_proj / GDN 投影 (`qkv_w` / `o_w` / `gate_w`(GDN) / `beta_w` /
   `alpha_w`) はまだ 1 回ごとに往復している。~~ s164 で q4 の GDN 層
   (qkv/z → scan → o_proj) は常駐化した。**未計測** — 期待値は
   13k prefill 97 s → 70 s 前後 (要 GPU 実測)。残っているのは
   - full attention 層の q/k/v → attention → o_proj (ホスト経由)
   - beta/alpha のホスト計算と H2D (5 MB/層) — **既存 `dno_matvec_f32` では
     デバイス化できない** (m=1 カーネル。s166/P4 の節と `REPORT_P4.md`)。行バッチ版
     カーネルの追加 (.cu + PTX 再生成) が前提条件。h2 の D2H 自体は s167/P6 で
     `DENNO_PF_OVERLAP=1` のときだけ qkv/z GEMM と重なる (既定 OFF・未計測)
   - embed の H2D と logits の D2H
2. 層の入力 (buf_h) と出力 (buf_out) をデバイスに留めれば、層をまたぐ
   往復も消せる (最終的に残るのは embed の H2D と logits の D2H だけ)。
3. アンカー (9B int8) は本変更が Q4 重み専用経路のため未実行。
   q4 経路の一致は上記のバイト一致で確認済み。

## 変更 (s168・行タイルを実測ノブに)

`Q4_M_TILE` は s124 以来 1024 固定だった (256 → 1024 は sync 回数 1/4 の実測)。
prefill が token 数に線形になった今、残るコストは q4 GEMM の重みストリーミングで、
重みは**行タイルごとに読み直される** (発行量 = グループ数 G × 層の重み)。

- `DENNO_Q4_M_TILE=512|1024|2048|4096|8192` (64 の倍数・既定 1024) で
  `q4_dispatch` / `chain_mlp_prefill_q4` / `chain_gdn_prefill_q4` の行タイルを
  再ビルド無しで切り替える。プロセス内で 1 回だけ読む (変更はサーバ再起動)。
- 端数タイルは `max_m()` (既定 16) 以下なら直前のタイルへ併合する
  (`q4_group_rows`)。GEMM の行ブロックは 64 行なので、端数を単独 launch すると
  1 発がほぼ空回りする。行は独立なので出力はビット一致。
- 中間バッファは 1 タイルの最大行数 (`tile + max_m()`) でループの前に確保する
  (併合で最終タイルだけ大きくなるため。途中 grow は先行 launch の読み先を解放する)。
- 13k token の launch 回数 (live 経路): 1024 → 3,856 / 2048 → 2,128 /
  4096 → 1,264。発行重み量は 173.5 GB → 54.5 GB (3.2x 減)。**未計測** —
  A/B 手順は `REPORT_P7.md` の 6 節。
