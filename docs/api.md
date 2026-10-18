# denno API 仕様(対応範囲)

## POST /v1/messages

Anthropic Messages API 互換。対応フィールド:

| フィールド | 対応 |
| --- | --- |
| `model` (必須) | serve 時の `--name` と一致が必要(不一致は 404 not_found_error) |
| `max_tokens` (任意) | 生成上限。未指定時 1024 / 明示時は上限 4096 で clamp (s53)。prompt+max_tokens が context 超過なら 400 |
| `messages` (必須) | role は user/assistant。content は文字列 or `[{type:"text",...}]` |
| `system` | 文字列 or textブロック配列 |
| `temperature` | 0 で greedy |
| `top_p`, `top_k` | 対応 |
| `stop_sequences` | 生成テキスト中の最も早い出現で打ち切り |
| `repetition_penalty` | A2.5: HF 互換の反復ペナルティ (任意。未指定/1.0 = 無効) |
| `frequency_penalty` | s56: OpenAI 互換。文脈中の出現回数 × 係数を減算 (任意。未指定/0 = 無効)。詳細は §反復抑止 |
| `presence_penalty` | s56: OpenAI 互換。文脈に 1 回以上現れたトークンから係数を 1 回減算 (任意。未指定/0 = 無効)。詳細は §反復抑止 |
| `no_repeat_ngram_size` | s56: 既出 n-gram の続きを禁止 (任意。2〜1024。未指定 = 無効)。詳細は §反復抑止 |
| `stream` | true で SSE |
| `tools` | tool use 対応 (下記 §tool use)。空配列は tools 無しと同一 |
| `tool_choice` | `"auto"` / `"none"` / `"any"` / `{"type":"tool","name":"..."}`。既定 auto。tools 無しでの指定は 400 |
| `thinking` | s47/s52/s53c: `"off"` / `"brief"` / `"full"` の文字列、または Anthropic マップ `{"type":"enabled",...}`(= off 扱い・`budget_tokens` は無視)/ `{"type":"disabled"}`(= off 扱い)。既定 brief (= off + 簡潔指示)。未知の type や文字列/マップ以外は 400 (下記 §thinking 制御) |
| `metadata` 他 | 受け流す(パースのみ) |

レスポンス例(非ストリーミング):

```json
{
  "id": "msg_...",
  "type": "message",
  "role": "assistant",
  "model": "gpt2",
  "content": [{"type": "text", "text": "..."}],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {"input_tokens": 12, "output_tokens": 34}
}
```

`stop_reason`: `end_turn` / `max_tokens` / `stop_sequence` / `tool_use`(tools 使用時)

## tool use (tools / tool_choice)

Anthropic 形式のツール定義を chat template の `tools` 変数 (Qwen/HF 形式へ変換) で
モデルに公開し、モデル出力中の `<tool_call>` block を構造化抽出して
Anthropic 形式の `tool_use` block として返す。

```json
{
  "tools": [{
    "name": "get_weather",
    "description": "Get current weather",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}
  }],
  "tool_choice": "auto"
}
```

- `tools[].input_schema` は必須 (JSON Schema・object 以外は 400)。`type` フィールドは受け流す
- `tool_choice`: `auto`(既定)/ `none`(モデルへ非公開・抽出もしない)/ `any` / `{"type":"tool","name":...}`。
  **`any` / `tool` (強制) は拘束デコードが有効**: sample 直前に logits マスクを適用し、モデル生成を
  公式テンプレート形式の `<tool_call>` block へ強制する (`any` = 全定義ツール、`tool` = 指定ツールのみ許容)。
  block 完了 (`</tool_call>` 終端) まで EOS を抑制し、完了後は無マスク。
  `auto` / `none` / tools 無しは従来経路とビット同一
- `tools` + `stream:true` は未対応 (400)。streaming の tool_use 対応は将来課題

### 会話履歴 (assistant tool_use / user tool_result)

```json
{"role": "assistant", "content": [
  {"type": "text", "text": "Let me check."},
  {"type": "tool_use", "id": "toolu_...", "name": "get_weather", "input": {"city": "Tokyo"}}
]}
{"role": "user", "content": [
  {"type": "tool_result", "tool_use_id": "toolu_...", "content": "22C sunny"}
]}
```

- assistant の `tool_use` block は公式テンプレートと同一形式の `<tool_call>` XML テキストへ
  変換して履歴に織り込む (モデル自身の出力と同一形式)。`id` は会話内で一意が必須
- user の `tool_result` block は `<tool_response>` で包んで user テキストへ。
  `tool_use_id` は直前の assistant tool_use block と一致が必要 (未知参照は 400)。
  content は文字列 or text block 配列のみ
- 1 メッセージ内の block 順序は保持 (text と tool_use / tool_result の混在可)

### 応答

モデルが完全な `<tool_call>` block を出力した場合、その block はテキストから除去し
(残テキストは trim)、`tool_use` blocks を content に追加して `stop_reason: "tool_use"` を返す:

```json
{
  "content": [
    {"type": "text", "text": "..."},                       // 非空の場合のみ
    {"type": "tool_use", "id": "toolu_...", "name": "get_weather", "input": {"city": "Tokyo"}}
  ],
  "stop_reason": "tool_use"
}
```

- `id` は `toolu_` + 24 hex (応答ごとに新規採番)
- 抽出対象は公式テンプレートの XML 形式に加え JSON 形式
  (`{"name":..., "arguments":{...}}` — `parameters`/`input` キーも受ける) も許容
- パースに失敗した / 閉じていない `<tool_call>` block は除去せずテキストとして残す
  (モデルが形式例を本文に出した場合への防御)
- tool_call が 1 件も無い場合は従来どおり text block + 元の stop_reason

## thinking 制御 (thinking)

denno 拡張のトップレベル任意フィールド (s47・未指定 = `brief`)。s52 から 2 形式を受理する:

- 文字列形式: `"off"` / `"brief"` / `"full"`
  - `off`: chat template に `enable_thinking=false` を渡す (思考せず即答)
  - `brief` (既定): **`off` と同じく思考を無効化**し、system 冒頭に決定的な
    短答誘導文 (`回答は簡潔に、数文でまとめてください。`) を注入する。既存 system は保持。
    s53c の実測: 思考 ON + 誘導文では 9B が短い system で "Thinking Process:" の
    メタ分析を延々と書き回答に到達しなかったため、思考自体を止める方式に変更した。
  - `full`: 従来経路のまま (注入なし・prompt はビット同一・アンカー不変)
- Anthropic マップ形式 (s52・Hermes 互換):
  - `{"type":"enabled", ...}` は **`off` 扱い** (思考無効 + 簡潔指示)。`budget_tokens` 等の他フィールドは無視 (決定的)
  - `{"type":"disabled"}` は `off` 扱い
  - 未知の `type` (欠落・非文字列含む) は 400 invalid_request_error
- 文字列 / オブジェクト以外 (数値・真偽値・配列等) も 400

不正値の 400 は `tool_choice` と同一のバリデーション慣習 (エラー形式は §エラー形式)。

## 反復抑止 (repetition_penalty / frequency_penalty / presence_penalty / no_repeat_ngram_size)

denno 拡張のトップレベル任意フィールド (s56)。A2.5 の `repetition_penalty` に 3 種を追加した。
**すべて未指定 (または無効値) なら従来経路とビット完全一致** — 数値経路は no-op で、適用自体を省略する。

| フィールド | 範囲 | 無効値 | 意味 |
| --- | --- | --- | --- |
| `repetition_penalty` | 有限値 (1.0 で無効) | 未指定 / `1.0` | HF 互換。正の logit は除算・負は乗算 (A2.5・既存仕様) |
| `frequency_penalty` | 有限値 (推奨 −2.0〜2.0) | 未指定 / `0.0` | 文脈中の**出現回数 × 係数**を logit から減算 |
| `presence_penalty` | 有限値 (推奨 −2.0〜2.0) | 未指定 / `0.0` | 文脈に**1 回以上**現れたトークンから係数を **1 回だけ**減算 |
| `no_repeat_ngram_size` | 整数 `2..=1024` | 未指定 | 文脈に既出の n-gram の**続き**を `-inf` で禁止 (llama.cpp / HF 準拠) |

- `frequency_penalty` / `presence_penalty` は OpenAI 互換の意味で、**`repetition_penalty` とは独立・別物** (前者は単純減算のみ)。負値は加算方向に働く。範囲外も拒否しない (既存 `temperature` と同じ作法)
- `no_repeat_ngram_size` が **1 以下または 1024 超なら 400** (`n == 1` は「直前トークン自身を禁止」という別の意味になり、無効と区別できないため明示的に弾く。無効化はフィールドを省略する)
- NaN / inf (JSON の `1e40` 等) は 400 invalid_request_error。`0.0` / `-0.0` は `None` に正規化する

**適用順** (この順序はビット契約の一部で、変えてはならない):

```text
生 logits
  → ① repetition_penalty     (A2.5 の既存処理・位置/意味は不変)
  → ② frequency_penalty      (出現回数 × 係数の減算)
  → ③ presence_penalty       (出現有無につき 1 回の減算)
  → ④ no_repeat_ngram_size   (既出 n-gram の続きを -inf)
  → temperature → top-k → top-p → sample
```

- 逐次 decode とバッチ decode は同じ関数 (`AntiRepetition::apply`) を通るため、バッチ構成は出力に影響しない (RNG 不使用・決定論契約は不変)
- ①〜④ がすべて無効のとき `logits` には一切触らない (0 の減算すらしない = `-0.0` の符号も変わらない)
- 順序についての正直な注記: ②③ の内部順序は f32 で観測可能だが、④ を ②③ の前後どちらに置いても `-inf` マスクが支配するため f32 では観測できない (変異テストで生存を確認)。「最後に置く」はコード構造とこの仕様で担保する

参考実装: HF transformers `NoRepeatNGramLogitsProcessor` (Apache-2.0) / llama.cpp (MIT) の着想による独自実装 (逐語コピー無し)。

## SSE イベント列 (stream: true)

```
message_start → content_block_start → ping
  → content_block_delta (text_delta) × N
→ content_block_stop → message_delta (stop_reason, usage) → message_stop
```

tools + stream の場合、生成中の `<tool_call>` block は block 単位で確定した時点で tool_use block
として挿入される (content_block_start (type: tool_use, id/name/input) → content_block_delta
(input_json_delta) → content_block_stop を即時完結)。block 境界が delta で分割されても検出され、
パース失敗 / 未完結 block はテキストとして流れる (非ストリーミング抽出 `extract_tool_calls` と
同一規約・delta 結合も同一)。tool_use block を 1 つでも出した場合 message_delta の stop_reason は
`"tool_use"` (非ストリーミングと同一規則)

## エラー形式

```json
{"type": "error", "error": {"type": "invalid_request_error", "message": "..."}}
```

| type | status |
| --- | --- |
| invalid_request_error | 400 |
| authentication_error | 401 |
| not_found_error | 404 |
| api_error | 500 |

認証: `--api-key` 指定時は `x-api-key` ヘッダ(または `Authorization: Bearer`)が必要。
`anthropic-version` ヘッダは検証せず受け流す。

## その他

- `GET /v1/models`: serve 中のモデルID と context_length
- `GET /health`: `{"status": "ok"}`
