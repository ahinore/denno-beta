# KPI の定義と現在値

この文書は decode / prefill の**判定に使う数値の定義**を固定する。記録だけ残っている
古い数値（測定条件が不明なもの）は判定に使わない。

## 1. 固定プロトコル（decode）

- 対象: 8095 の serve（27B は `--quant q4`、9B は int8）
- リクエスト: `temperature 0`・`max_tokens 200`・同一プロンプト・`stream: true`
- プロンプト: `List the integers from 1 to 400, one per line, then stop.`
- 指標:
  - TTFT = 最初の `content_block_delta` までの時間
  - decode = `(wall - TTFT) / output_tokens`
  - 3 回の中央値。**1 回目は重みの初回アップロードを含むので捨てる**（TTFT が秒単位で伸びる）
- GPU は他ワークロードの無い窓で測る（`nvidia-smi` の util が 0〜5%）

実装: `/opt/data/cache/scratch/kpi_1021.py`（コンテナへ `docker cp` して `python3 kpi_1021.py 8095`）。

## 2. 現在値

| 対象 | ビルド | decode (ms/token) | 備考 |
|---|---|---|---|
| 27B q4 | `7c1d688`（B20 段階 1 前） | **75.89** | 2026-10-21T13:47Z・3 回中央値・util 0%・137 W |
| 27B q4 | `2701cf3`（B20 段階 1） | **32.42** | 2026-10-21T14:25Z・3 回中央値。run1 は cold で 49.30（TTFT 7.306 s） |
| 9B int8 | `2701cf3` | **16.24** | GPU_VERIFY の gen2 = 3.248 s / 200 token。gen1 3.874 s は warmup 込み |
| 27B q4 | `8a0525b`（K5 時点 HEAD） | **32.43** | 2026-10-21T17:58Z・3 回中央値（GPU_VERIFY 再デプロイ後の 2 つ目の triple）・util 0%。生値 32.43 / 32.43 / 32.43 |
| 9B int8 | `8a0525b`（K5 時点 HEAD） | **16.24〜16.28** | GPU_VERIFY の gen2（同一ビルドの複数回で 16.24〜16.28 の幅） |
| 27B q4 | `8d2554e`（P7+P8 統合・`DENNO_FAST_REDUCE` 未設定 = 逐次) | **31.83〜32.04** | 2026-10-22T15:40Z・`dec_prof.py 200` の 2 回。ゲート OFF の退行が無いことの確認 |
| 27B q4 | `8d2554e`（`DENNO_FAST_REDUCE=1`） | **19.15〜19.43** | 2026-10-22T15:55Z・`dec_prof.py 200` の 2 回。逐次版の 1.65〜1.67 倍 |
| 27B q4 | `81ca995`（P10+P11+`dno_add_rmsnorm` 並列縮約・`DENNO_FAST_REDUCE=1`） | **17.20〜17.51** | 2026-10-22T17:20Z・`dec_prof.py 200` の 3 回。累積 1.95 倍 |

- 27B q4 の `2701cf3` の生値: TTFT 0.205 / 0.202 s、wall 6.689 / 6.685 s、out 200。
- 27B q4 の `8a0525b` は `2701cf3` の 32.42 と同水準（差 0.01）。アンカーは 3/3 MATCH
  （`c61ad85b` / `9e17d563` / `af6eb6ab`）。この行の triple の採り方は §3.1 の規則にもとづく。
- B20 段階 1 の効き方: 27B q4 の per-op 線形 D2H が 0 になった（PROFILE の `d2h_src_linear`
  898 call で 0.0 ms、`sl_lin` 0 call）。アンカー 3 本は MATCH（`c61ad85b` / `9e17d563` / `af6eb6ab`）。
- **s169 の `dno_matvec_f32` 並列リダクション**（`DENNO_FAST_REDUCE=1`）: beta/alpha の制御投影は
  1 token あたり 96 回・12.85 ms（decode の 38%）で、1 行の内積を 1 スレッドが逐次に回していた。
  1 行 1 ブロック・256 スレッド・warp shuffle の固定順（16,8,4,2,1）→ warp 合計を id 昇順、という
  atomic 無しの固定木に置き換えた（決定論は維持）。逐次版とは丸め順が違うため既定は OFF で、
  本番スクリプト `scripts/prod_serve_27b.sh` が明示的に ON にする。temp 0 の 3 問（80 token）と
  8 問 x 150 token の出力が逐次版と完全一致した。アンカーは q4 chain を通らないため不変（3/3 MATCH）。
- **prefill のタイルは 1024 のまま**（P7 で `DENNO_Q4_M_TILE` を追加）: 2,048 は 16k token で
  13.81 s（13.87 と同水準）だが、65k token では中間バッファが 274 MiB を要求して
  `cuMemAlloc error 2`（空き 233 MiB）で落ちた。既定 1,024 を変えない。
- 参考（測定条件が違うので判定に使わない）: 2026-10-21 の main session 実測は 161 token で
  72〜75、400 token で 87 ms/token。記録値 55.1 は条件が残っておらず再現しない。

## 3. 計測の運用規則（2026-10-21 の実測にもとづく）

§1 のプロトコル・式（`(wall - TTFT) / output_tokens`）・「1 回目を捨てる」規則の意味は変えない。
ここに固定するのは、**どの triple を正式値にするか**、**単発の変動を回帰と誤認しない条件**、
**GPU を巻き込まない手順**の 3 点だけ。

### 3.1 GPU_VERIFY の再デプロイ直後は 1 つ目の triple を捨て、2 つ目の triple の中央値を正式値にする

- 規則: 再デプロイ（= 重みの cold アップロード）の直後の **1 つ目の triple は丸ごと捨て、
  2 つ目の triple の中央値を正式値**として記録する。§1 の「1 回目を捨てる」は 1 つの triple の
  中の話で、こちらは triple 単位の規則。両方を適用する（= 捨てた triple の次で、さらに
  その triple の run1 を捨てる）。
- 実例 1（17:35Z の再デプロイ直後）: 1 つ目の triple は run1 の TTFT が 7.6 s に伸びて 49.94、
  run2 / run3 が 32.43 で、この triple の中央値は 32.43。それでも 1 つ目の triple は捨てる。
  2 つ目の triple は 32.43 / 32.43 / 32.43 で中央値 32.43 → 正式値 **32.43 ms/token**
  （§2 の `8a0525b` の行）。
- 実例 2（16:28Z）: cold 重みアップロードの名残で run2 42.83 が median になった。cold の影響は
  直後の 1 本に収まらないので、中央値が動いた triple をそのまま正式値にしない。
- この規則は「捨てる」対象を triple 単位に広げるだけで、条件（temperature 0・max_tokens 200・
  同一プロンプト・util 0〜5%）と式は §1 のまま。

### 3.2 `bench_all.sh` は行ごとに変動幅を持つ。単発の 3% 超では回帰と判定しない

2026-10-21 に実測した行ごとの範囲（単位は ms/call、`scripts/bench_all.sh` の出力そのまま）:

| 行 | m / n | 実測範囲 (ms/call) |
|---|---|---|
| gate_up | m=1 / m=2 / m=8 / m=64 / m=300 | 0.101〜0.109 / 0.137〜0.147 / 0.503〜0.528 / 2.316〜2.504 / 17.05〜17.23 |
| down_proj | m=1 / m=2 / m=8 / m=64 / m=300 | 0.054〜0.069 / 0.080〜0.089 / 0.248〜0.256 / 2.19〜2.35 / 6.55〜7.20 |
| lm_head | m=1 | 0.657〜0.664 |
| GDN | n=1 / n=8 / n=32 / n=300 | 0.143〜0.146 / 0.303〜0.311 / 0.806〜0.824 / 5.72〜5.83 |

- GDN の上記範囲は **det=OK・maxabs 2.44e-4** の測定。
- 判定規則:
  - **単発の 3% 超で回帰と判定しない**。同じ commit で 3 回測り、窓（util 0〜5 pct・
    クロック 2835〜2887 MHz）を揃えて比べる。
  - **m=1 は L2 相対**なので絶対値で判定しない。
  - m=64 / m=300 の down_proj は 6.55 と 7.20 の幅がある。**この行だけで合否を出さない**。
  - 範囲の外に出たときは、まず同じ commit で 3 回測り直して窓が揃っているかを確認する。

### 3.3 `bench_all.sh` は 8095 の常駐重みと VRAM で競合する

- 実例（2026-10-21T17:30Z）: 27B の重みが常駐（約 25.5 GB / 32.6 GB）した状態で `bench_all.sh`
  を回すと **serve が落ちる**。スクリプト自身が「小さな確保だけ」を前提にしているが、
  常駐重みがある窓ではその前提が崩れる。
- 回してよい窓: **8095 が lazy（1692 MiB・compute app 無し）** のとき。
- 回した直後に必ず health を確認する。落ちていたら `scripts/prod_serve_27b.sh` で復旧する
  （**スクリプトの引数・値は変更しない**）。
- serve が落ちていた区間の bench 値は採用しない（health 200 を確認してから数値を読む）。

### 3.4 health が 000 のときは、まず `ps aux | grep prod_serve` を見る

- health が **000**（接続不可）でも、再起動中なら故障ではない。2026-10-21 の **16:45Z** と
  **17:47Z** に同じ見かけを記録している。
- 手順: (1) `ps aux | grep prod_serve` で再起動中かを確認する → (2) `nvidia-smi` の compute apps
  と VRAM を併記して判断する → (3) 本当に落ちていれば §3.3 の復旧手順で戻す。
- health 000 を即「故障」として扱わない。再起動を重ねると常駐重みの再アップロードが起き、
  次の triple が cold になる（§3.1）。

## 4. 未了

- `64c1163` のバイナリを同プロトコルで測る比較（別 worktree のビルドが要る。dev の CPU と
  競合するので dev 停止窓で実施）。これで「55.1 がコード差か測定差か」が確定する。

## 4. prefill (27B q4・cold)

計測は `scripts/prefill_bench.py`（毎回ユニークな前置きを付けて prefix cache を外す）。
3 回の中央値、起動直後の 1 回は捨てる。**同じプロンプトの再送は cache に当たるので
cold 計測に使えない**（旧手順の `cache_probe.py` turn1 はこの理由でばらついた）。

| 条件 | `95f3cd8` | `125117c` (FFN) | `56c326d` (GDN) | `b3a8838` (残差) | `81ca995` (現行) | 合計 |
|---|---|---|---|---|---|---|
| 3,161 token (16,000 chars) | 40.10 s | 24.21 s | 16.41 s | 13.87 s | **13.55 s** | 2.96x |
| 12,850 token (65,000 chars) | 162.3 s | 96.9〜109.1 s | 65.86〜68.33 s | 56.93〜62.00 s | **55.23〜60.47 s** | 2.7x |

- `81ca995` の差分: dev1 P10（GEMV の列ブロック幅）で decode のみ 19.42 → 19.02、
  dev2 P11（RoPE 表の共有・GDN h2 写しの削除）で prefill 13.86 → 13.55、
  メインの `dno_add_rmsnorm` 並列縮約で decode 19.02 → 17.20。
- **内訳（`81ca995`・`DENNO_PF_DBG=1`・n=14,043・64 層・wall 63.05 s）**:

| 位相 | ms | % |
|---|---|---|
| `sl_lay_gdn_proj` | 18,015.8 | 28.6 |
| `sl_lay_ffn_down` | 11,100.0 | 17.6 |
| `sl_lay_full_core` | 10,695.3 | 17.0 |
| `sl_lay_ffn_gate` | 10,042.6 | 15.9 |
| `sl_lay_gdn_scan` | 6,613.7 | 10.5 |
| `sl_lay_full_qk` | 2,087.3 | 3.3 |
| `sl_lay_norm_in` | 1,855.2 | 2.9 |
| `sl_lay_gdn_oproj` | 1,379.3 | 2.2 |
| `sl_lay_swiglu` | 210.4 | 0.3 |
| `tail` (final+logits) | 561.0 | 0.9 |

- ffn_gate は約 16 TFLOP/s、gdn_proj は約 7.6 TFLOP/s（同じ換算）。投影を分けている分だけ
  gdn_proj の効率が悪い。attention core は 17% で、支配的ではないが無視できない。

- `125117c`: Q4 FFN (gate/up → swiglu → down) をデバイス常駐化。
- `56c326d`: GDN 層の qkv/z → chunk scan → o_proj をデバイス常駐化 (`35b5c40`)。
- `b3a8838`: 残差ストリーム `h` を層の間デバイスに置く (`b1e3494`) + decode の重み
  再転送を除去 (`ce21f1c`)。rmsnorm もデバイス (`CH_PF_RES`)。
- どれも q4 GEMM は per-op 経路と同一カーネル・同一引数、行分割のみ。
- 一致: `temperature 0` の生成 3 問が `95f3cd8` のバイナリと 176 バイト完全一致
  (`ea1fcdd1…`)。decode は 33.50 → 32.49 ms/token。

世代を重ねた現在値（すべて同じプロトコル・cold・nonce・中央値）:

| ビルド | 3,161 token | 12,850 token |
|---|---|---|
| `95f3cd8`（出発点） | 40.10 s | 162.3 s |
| `125117c` | 24.21 s | 96.9〜109.1 s |
| `56c326d` | 16.41 s | 65.86〜68.33 s |
| `b3a8838` | 13.87 s | 56.93〜62.00 s |
| `81ca995`（rmsnorm 並列化まで） | 13.55 s | 55.23〜60.47 s |
| `1fde630`（P12+P13） | **9.59 s** | **39.0〜41.8 s** |

- `1fde630` = P12（full attention の att 往復を pinned 直送りにし、GPU attend 時の
  `buf_att` ゼロ埋め 345 MB/層を廃止）＋ P13（GDN の beta/alpha を h2 の 1 回の走査に統合、
  `DENNO_PF_OVERLAP` を既定 ON）。decode は 17.44〜17.46 ms/token で変化なし。
- 3,161 token で 40.10 → 9.59 s（**4.18x**）、12,850 token で 162.3 → 39.0 s（**4.16x**）。
  アンカー 3/3 MATCH・lib テスト 316 passed・temp 0 の 8 問 × 150 token が全段階で完全一致。

- 内訳（`95f3cd8`・PROFILE・13k token）: prefill 161.4 s のうち `sl_lin` 113.5 s (70%)、
  802 呼び出し = 141 ms/回。カーネルは 3,161 token の nsys で 7.9 s / 47.7 s のみ。
- 一致: `temperature 0` の生成 3 問が旧バイナリと 176 バイト完全一致。
- 詳細は `docs/PREFILL_PATH.md`。
