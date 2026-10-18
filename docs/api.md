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
| `stream` | true で SSE |
| `tools` | tool use 対応 (下記 §tool use)。空配列は tools 無しと同一 |
| `tool_choice` | `"auto"` / `"none"` / `"any"` / `{"type":"tool","name":"..."}`。既定 auto。tools 無しでの指定は 400 |
| `thinking` | s47/s52: `"off"` / `"brief"` / `"full"` の文字列、または Anthropic マップ `{"type":"enabled",...}`(= brief 扱い・`budget_tokens` は無視)/ `{"type":"disabled"}`(= off 扱い)。既定 brief。未知の type や文字列/マップ以外は 400 (下記 §thinking 制御) |
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
  - `brief` (既定): 思考 ON のまま、system 冒頭に決定的な短答誘導文を注入
    (ソフトな誘導 = ハード制御ではない。既存 system は保持)
  - `full`: 従来経路のまま (注入なし・prompt はビット同一)
- Anthropic マップ形式 (s52・Hermes 互換):
  - `{"type":"enabled", ...}` は `brief` 扱い。`budget_tokens` 等の他フィールドは無視 (決定的)
  - `{"type":"disabled"}` は `off` 扱い
  - 未知の `type` (欠落・非文字列含む) は 400 invalid_request_error
- 文字列 / オブジェクト以外 (数値・真偽値・配列等) も 400

不正値の 400 は `tool_choice` と同一のバリデーション慣習 (エラー形式は §エラー形式)。

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
