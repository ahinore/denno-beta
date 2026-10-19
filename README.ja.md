# denno-beta

[![CI](https://github.com/ahinore/denno-beta/actions/workflows/ci.yml/badge.svg)](https://github.com/ahinore/denno-beta/actions/workflows/ci.yml)

[English](README.md) | 日本語

Rustで書いたAnthropic Messages API互換のLLMサーバー。
LLMの動作原理(トークン化 → forward pass → サンプリング → デコードループ)をコードで理解することを目的に、**推論カーネルを完全自前実装**している(candle等の既存LLMライブラリは不使用)。構成の参考: [ashhart/TensorFold](https://github.com/ashhart/TensorFold)

- **単一バイナリ** `denno`(CLI `pull/models/info/generate/serve` + axum HTTPサーバー + 推論エンジン)。既定は CPU のみ、`--features cuda` ビルドで GPU オフロード
- 対応モデル: GPT-2 124M、Llama系(RMSNorm+RoPE+SwiGLU+GQA)= **Qwen2.5-0.5B〜7B-Instruct**、**Qwen3-0.6B**(QK-Norm、thinking モード対応)、**Qwen3.5(Gated DeltaNet ハイブリッド・GGUF ローダ経路)**(チャットテンプレート対応)
- **exact decoding**: seed固定・温度0で同一入力→ビット完全同一出力。最適化・並列化(連続バッチング/SIMD)をしても壊さない規律(CIでexactnessテストを担保)
- **CUDA オフロード**(`--features cuda`): decode/prefill 両チェーンを自前 CUDA カーネルで GPU 化(重み/KV/隠れ状態の VRAM 常駐・13,252 トークンの prefill 395.8s → 12.2s は P17k 時点の計測で現行性能ではない)。数値契約: 線形層/rope/rmsnorm/swiglu は CPU とビット一致、attention のみ緩和枠(詳細は[アーキテクチャ解説 §8](docs/ARCHITECTURE.ja.md))
- 連続バッチング(複数セッションのdecodeを1 forwardに統合・KVスロットプール)、prefix cache(LCP最大選択・GDN 状態スナップショット対応)、int8量子化(SIMD整数カーネル、重みメモリ半減)、repetition penalty
- **計測値**(いずれも Qwen3.5-9B・`--quant int8`・GPU):正式値は cold 単発(サーバー起動 → 1 リクエスト・prefix cache off・交互 A/B)。s100 時点で prefill **13.18 s / 431 tok/s**、decode **16.4 ms/token**、アンカー `c61ad85b` / `9e17d563` / `af6eb6ab`。出典: [docs/TASKS.md](docs/TASKS.md) 冒頭と[開発履歴](docs/HISTORY.ja.md)の「計測基準の訂正」節。ウォーム 2 回目は同一セッション内の相対比較にのみ使う
- **推論可視化(denno-trace)**: 推論を一切待たせない設計のテレメトリ(DENNO_TELEMETRY=1、既定OFF・read-only)+ 別プロセスの可視化サーバー `denno-view`。ブラウザでモデル駆動プログラムのアニメーション(実装コードのステップ実行・層パネル・KV/変数)が見られる

## 使い方(Qwenの例)

```bash
# 1. モデルのダウンロード(HF hub キャッシュ ~/.cache/denno/hub に入る)
cargo run --release -- pull Qwen/Qwen2.5-0.5B-Instruct

# 2. サーバー起動(seed 固定で exact decoding)
cargo run --release -- serve Qwen/Qwen2.5-0.5B-Instruct --name denno --port 8080 --seed 42

# 3. API を叩く(Anthropic Messages 形式)
curl http://127.0.0.1:8080/v1/messages \
  -H "content-type: application/json" \
  -d '{"model":"denno","max_tokens":128,
       "messages":[{"role":"user","content":"1+1は?"}]}'
```

CLI で直接生成することもできる:

```bash
cargo run --release -- generate Qwen/Qwen2.5-0.5B-Instruct --prompt "1+1は?" --temperature 0 --seed 42
cargo run --release -- generate openai-community/gpt2 --prompt "The capital of France is" --raw --seed 42
```

Python SDK(Anthropic公式)からもそのまま使える:

```python
import anthropic
c = anthropic.Anthropic(base_url="http://127.0.0.1:8080", api_key="sk-test")
print(c.messages.create(model="denno", max_tokens=128,
      messages=[{"role":"user","content":"こんにちは"}]).content[0].text)
```

## 可視化(denno-view)を起動する

```bash
# 1. サーバーをテレメトリONで起てる(DENNO_TELEMETRY=1 以外は同一コマンド)
DENNO_TELEMETRY=1 cargo run --release -- serve Qwen/Qwen2.5-0.5B-Instruct --name denno --port 8080 --seed 42

# 2. 別ターミナルで denno-view を起動
#    (既定で ws://127.0.0.1:8080 のテレメトリWSに接続し、:8093 で待受)
VIEW_TOKENIZER=~/.cache/denno/hub/models--Qwen--Qwen2.5-0.5B-Instruct/tokenizer.json \
  cargo run --release -p denno-view

# 3. ブラウザで http://localhost:8093/ を開き、内蔵チャットから話すと
#    モデル駆動プログラム・層パネル・KV/変数のアニメーションが連動して動く
```

**動作デモ** — 内蔵チャットで生成すると、モデル駆動プログラムが実装コードをステップ実行し、層/KV/変数パネルが連動してアニメーションします:

![denno-view 動作デモ](docs/assets/denno_view.gif)

`denno-view` の環境変数: `VIEW_UPSTREAM`(既定 `ws://127.0.0.1:8080/v1/telemetry/ws`)、`VIEW_PORT`(既定 8093)、`VIEW_RECORD`(フレームをファイル記録=リプレイ用)、`VIEW_TOKENIZER`(トークン文字表示用の tokenizer.json パス)。

## ドキュメント

- [アーキテクチャ解説(日本語)](docs/ARCHITECTURE.ja.md) / [Architecture (English)](docs/ARCHITECTURE.en.md) — 設計方針・推論フロー・KVキャッシュ・演算カーネル・モデルの違い・可視化の構成・使い方
- [開発履歴(日本語)](docs/HISTORY.ja.md) / [Development History (English)](docs/HISTORY.en.md) — P1〜P17(CUDA)/P19(Qwen3.5)/A〜B・C・s シリーズ(s101 時点)の実装記録・計測データ・学習メモ
- [API仕様](docs/api.md)

## ライセンス

MIT または Apache License 2.0 のデュアルライセンス — [LICENSE-MIT](LICENSE-MIT) / [LICENSE-APACHE](LICENSE-APACHE) を参照。
