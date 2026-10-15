# denno API 仕様(対応範囲)

## POST /v1/messages

Anthropic Messages API 互換。対応フィールド:

| フィールド | 対応 |
| --- | --- |
| `model` (必須) | serve 時の `--name` と一致が必要(不一致は 404 not_found_error) |
| `max_tokens` (必須) | 生成上限。prompt+max_tokens が context 超過なら 400 |
| `messages` (必須) | role は user/assistant。content は文字列 or `[{type:"text",...}]` |
| `system` | 文字列 or textブロック配列 |
| `temperature` | 0 で greedy |
| `top_p`, `top_k` | 対応 |
| `stop_sequences` | 生成テキスト中の最も早い出現で打ち切り |
| `stream` | true で SSE |
| `tools` | tool use 対応 (下記 §tool use)。空配列は tools 無しと同一 |
| `tool_choice` | `"auto"` / `"none"` / `"any"` / `{"type":"tool","name":"..."}`。既定 auto。tools 無しでの指定は 400 |
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
  **`any` / `tool` (強制) は拘束デコードを持たないため auto と同一挙動** (受理のみ・指定ツールの存在検証は実施)
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

## SSE イベント列 (stream: true)

```
message_start → content_block_start → ping
  → content_block_delta (text_delta) × N
→ content_block_stop → message_delta (stop_reason, usage) → message_stop
```

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
