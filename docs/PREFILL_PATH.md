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
