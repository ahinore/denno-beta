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

- 27B q4 の `2701cf3` の生値: TTFT 0.205 / 0.202 s、wall 6.689 / 6.685 s、out 200。
- B20 段階 1 の効き方: 27B q4 の per-op 線形 D2H が 0 になった（PROFILE の `d2h_src_linear`
  898 call で 0.0 ms、`sl_lin` 0 call）。アンカー 3 本は MATCH（`c61ad85b` / `9e17d563` / `af6eb6ab`）。
- 参考（測定条件が違うので判定に使わない）: 2026-10-21 の main session 実測は 161 token で
  72〜75、400 token で 87 ms/token。記録値 55.1 は条件が残っておらず再現しない。

## 3. 未了

- `64c1163` のバイナリを同プロトコルで測る比較（別 worktree のビルドが要る。dev の CPU と
  競合するので dev 停止窓で実施）。これで「55.1 がコード差か測定差か」が確定する。

## 4. prefill (27B q4・cold)

計測は `scripts/prefill_bench.py`（毎回ユニークな前置きを付けて prefix cache を外す）。
3 回の中央値、起動直後の 1 回は捨てる。**同じプロンプトの再送は cache に当たるので
cold 計測に使えない**（旧手順の `cache_probe.py` turn1 はこの理由でばらついた）。

| 条件 | `95f3cd8` | `125117c` | 倍率 |
|---|---|---|---|
| 3,161 token (16,000 chars) | 40.10 s | **24.21 s** | 1.66x |
| 12,850 token (65,000 chars) | 162.3 s | **96.9〜109.1 s** | 1.5〜1.7x |

- 内訳（`95f3cd8`・PROFILE・13k token）: prefill 161.4 s のうち `sl_lin` 113.5 s (70%)、
  802 呼び出し = 141 ms/回。カーネルは 3,161 token の nsys で 7.9 s / 47.7 s のみ。
- 一致: `temperature 0` の生成 3 問が旧バイナリと 176 バイト完全一致。
- 詳細は `docs/PREFILL_PATH.md`。
