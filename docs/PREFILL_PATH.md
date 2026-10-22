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

ビット一致の根拠:
- q4 GEMM は per-op 経路と**同一カーネル・同一グリッド・同一引数**
  (`dno_gemm_q4_0`)。行ごとに独立なので m の分割は 1 行の累積順を変えない。
- swiglu は `dno_swiglu` (CPU の `g/(1+exp(-g))*u` と同一演算列)。

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

1. qkv / o_proj / GDN 投影 (`qkv_w` / `o_w` / `gate_w`(GDN) / `beta_w` /
   `alpha_w`) はまだ 1 回ごとに往復している。同じ形で繋げば 13k で
   60〜70 s 台の見込み。残りは約 2/3。
2. 層の入力 (buf_h) と出力 (buf_out) をデバイスに留めれば、層をまたぐ
   往復も消せる (最終的に残るのは embed の H2D と logits の D2H だけ)。
3. アンカー (9B int8) は本変更が Q4 重み専用経路のため未実行。
   q4 経路の一致は上記のバイト一致で確認済み。
