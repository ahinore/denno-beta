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

`stop_reason`: `end_turn` / `max_tokens` / `stop_sequence`

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
