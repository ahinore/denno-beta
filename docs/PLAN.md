# Rust製 Anthropic互換 LLMサーバー 実装計画

参考構造: [ashhart/TensorFold](https://github.com/ashhart/TensorFold)
(LSM inference engine: CLI `pull/models/info/serve` + `/v1` HTTP API + 推論エンジン。OpenAI chat completions と Anthropic Messages の両API、ストリーミング、prefix キャッシュ、exact decoding テストを持つ)

## 目的

LLMの動作原理(トークン化 → forward pass → サンプリング → デコードループ)を
コードで理解するため、小さいモデルを CPU で動かす最小限の Anthropic Messages API
サーバーを Rust で作る。まず「動く最小」を最短で作り、その後に TensorFold 的な
機能(prefix cache, テレメトリ, 量子化)を段階的に足す。

## 全体構造(TensorFold構成の対応付け)

```
tensorfold-native (Zig 単一バイナリ)      →  Rust cargo workspace (単一バイナリ)
├── CLI: pull / models / info / serve     →  crate `cli` (clap)
├── HTTPサーバー: /v1 API, SSE, auth     →  crate `server` (axum)
├── エンジン: checkpoint, tokenizer,     →  crate `engine`
│   kernels, KV cache, sampler
└── docs: api.md, exact decoding,        →  docs/ (PLAN, API仕様, テスト方針)
    memory profiling
```

## 機能一覧

### A. API層(必須 = MVP)

| 機能 | 詳細 |
| --- | --- |
| `POST /v1/messages` | Anthropic Messages API 本体。非ストリーミングから先に実装 |
| リクエスト項目 | `model`, `max_tokens`(必須), `messages[{role, content}]`, `system`(文字列のみ), `temperature`, `top_p`, `top_k`, `stop_sequences`, `stream`, `metadata` |
| レスポンス項目 | `id`(msg_接頭辞+ランダム), `type:"message"`, `role:"assistant"`, `content:[{type:"text", text}]`, `model`, `stop_reason`(`end_turn`/`max_tokens`/`stop_sequence`), `stop_sequence`, `usage:{input_tokens, output_tokens}` |
| エラー形式 | `{"type":"error","error":{"type":"...","message":"..."}}` + 対応するHTTPステータス。type: `invalid_request_error`(400), `authentication_error`(401), `not_found_error`(404), `request_too_large`(413), `rate_limit_error`(429), `api_error`(500), `overloaded_error`(529) |
| 認証 | `x-api-key` ヘッダ(+互換で `Authorization: Bearer`)。キーなし起動時は認証スキップ |
 | 必須ヘッダ処理 | `anthropic-version` の検証(緩く:未知でも警告なし受付でOK) |
| `GET /v1/models` | serve しているモデルIDの广告(TensorFoldも同様) |
| ヘルスチェック | `GET /health` |
| SSE ストリーミング | `stream:true` のとき `message_start` → `content_block_start` → `content_block_delta`(text_delta) ×N → `content_block_stop` → `message_delta`(stop_reason, usage) → `message_stop` + `ping` |

### B. エンジン層(必須 = MVP)

| 機能 | 詳細 |
| --- | --- |
| checkpoint 読込 | safetensors + `config.json`。`pull` でHFからダウンロード(sha256/size検証) |
| トークナイザ | HF `tokenizers` クレート(`tokenizer.json` をロード)。input/output token 数はここから |
| モデルフォワードパス | CPUのみ。最初は GPT-2(最小構成: LayerNorm, GELU, causal attention)。次に Llama系(RMSNorm, RoPE, SwiGLU, GQA) |
| KVキャッシュ | prefill(プロンプト一括)→ decode(1トークンずつ)。キャッシュは連続バッファ+長さ管理 |
| サンプリング | greedy(temperature=0)、temperature、top-k、top-p。確率的な場合は seed を固定して再現可能に |
| デコードループ | EOS / `max_tokens` / `stop_sequences` で停止。stop_sequence は末尾マッチで検出し、その直前までを返す |
| チャットテンプレート | `tokenizer_config.json` の chat_template を minijinja で適用(messages+system → プロンプト文字列) |
| コンテキスト制限 | prompt+output が window を超えたら `invalid_request_error` |

### C. CLI(TensorFold準拠)

| コマンド | 詳細 |
| --- | --- |
| `pull REPO` | HF から checkpoint をローカルキャッシュ(`~/.cache/<app>/hub`)へ。中断再開は後回し |
| `models` | キャッシュ済みで serve 可能な一覧 |
| `info REPO` | family / 形式 / context / メモリ見積り |
| `serve DIR` | モデルをロードして HTTP サーバー起動。`--host --port --name --api-key --max-tokens --temperature --top-p --top-k --seed --context` |

### D. 運用・品質(TensorFold思想: exact decoding)

- **再現性テスト**: 同一 checkpoint + 同一設定 + 同一 seed なら毎回同一出力(非ストリーミング/ストリーミング、単発/並列で一致)を CI で保証
- **テレメトリ**: `/metrics` or `/stats` で tokens/s、TTFT、KV使用量(最低限のログから開始)
- リクエストキュー: エンジンは1モデルで逐次実行なので、同時リクエストは直列化(将来は連続バッチング)

## クレート構成(予定)

```
anthropic-server-rs/
├── Cargo.toml            # workspace
├── crates/
│   ├── cli/              # エントリポイント、サブコマンド
│   ├── server/           # axum, API DTO, SSE, auth, エラー変換
│   └── engine/
│       ├── src/model/gpt2.rs      # family trait の最初の実装
│       ├── src/model/llama.rs     # Qwen2.5 等の Llama 系
│       ├── src/sampler.rs
│       ├── src/kv_cache.rs
│       └── src/tokenizer.rs
├── docs/
│   ├── PLAN.md           # 本ファイル
│   └── api.md            # 対応エンドポイント仕様
└── tests/                # exactness テスト、API e2e テスト
```

主要依存: `axum`, `tokio`, `serde`/`serde_json`, `tokenizers`, `safetensors`,
`minijinja`(chat template), `clap`, `thiserror`, `tracing`, `rand`,
`ndarray` + `rayon`(CPU 行列演算)。
※ candle/llm 系クレートは使わない(理解が目的のため forward pass は自前実装)。

## マイルストーン

### M0: 骨組み(0.5日)
- workspace + cli/server/engine の3クレート、`serve` が空応答の `POST /v1/messages` を返すところまで
- 受け基準: `cargo run -- serve --port 8080` で起動し curl で 400/200 が取れる

### M1: エンジン最小(GPT-2, 2〜3日)
- GPT-2 124M を safetensors からロード、CPU で forward pass
- プロンプト1つ → 次トークン logits → greedy で数トークン生成(まずはバイナリ CLI で確認)
- 受け基準: 既知実装(例: HuggingFace transformers の greedy 出力)とトークン列が一致する

### M2: Llama系 + KVキャッシュ + サンプリング(2〜3日)
- `Family` trait 抽象化(重み名マップ、RoPE有無、norm種別)
- Qwen2.5-0.5B-Instruct をロード、KVキャッシュ付き prefill/decode
- temperature/top-k/top-p、seed 固定の再現性テスト
- 受け基準: chat template 適用プロンプトで会話らしい応答、同一 seed で同一出力

### M3: API 完成(非ストリーミング)(2日)
- `POST /v1/messages` 本実装(chat template 適用、usage/stop_reason 計上、エラー形式、認証)
- `GET /v1/models`, `/health`, `pull/models/info` CLI
- 受け基準: Anthropic公式SDK(Python)から `client.messages.create(...)` が通る

### M4: SSE ストリーミング(1〜2日)
- `stream:true` のイベント列実装。非ストリーミングと同一出力になることを exactness テストで確認

### M5: TensorFold 機能の移植(計画的に後続)
- 共通prefix キャッシュ(システムプロンプトのKV再利用)
- `/stats` テレメトリ、メモリプロファイル出力
- 量子化(int8 または Q4)
- 連続バッチング、tool use(content block の tool_use/tool_result)
- コンテキスト圧縮(TensorFold の compaction 相当)

## 最初のモデル候補

| モデル | パラメータ | 選択理由 |
| --- | --- | --- |
| GPT-2 124M(検証用) | 0.12B | 構成が最も単純で forward pass の正しさ検証に最適 |
| Qwen2.5-0.5B-Instruct(本命) | 0.5B | Llama系(RoPE/RMSNorm/SwiGLU/GQA)で現代LLMの標準構成を学べる。chat template あり、bf16で約1GB |

## テスト方針

1. 単体: sampler(分布、top-k/top-p の切詰め)、KVキャッシュのprefill/decode一致性(キャッシュありなしでlogits一致)
2. exactness: 同一seedで greedy=temperature:0 が毎回同一、ストリーミング=非ストリーミング
3. e2e: 公式 SDK での messages.create / stream、エラーケース(max_tokens 未指定 → 400 など)

## リスク・決め事

- CPU のみなので 0.5B でも生成は低速(数 token/s)。それが理解目的では問題ないが、M5 で量子化を早める価値あり
- トークン数の正確な計上は tokenizer で数える(文字数近似はしない)
- `anthropic-version` は受け流す(互換クライアントの多様な値に耐える)
