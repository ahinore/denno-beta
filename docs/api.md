# denno API 仕様(対応範囲)

## POST /v1/messages

Anthropic Messages API 互換。対応フィールド:

| フィールド | 対応 |
| --- | --- |
| `model` (必須) | serve 時の `--name` と一致が必要(不一致は 404 not_found_error)。`--name` 未指定時はモデルの repo 名 (例 `Qwen/Qwen2.5-0.5B-Instruct`) が既定値 |
| `max_tokens` (任意) | 生成上限。未指定時 1024 (s53) / 明示時は上限 4096 で clamp (s53)。`0` は「prefill のみ」として 0 のまま尊重。prompt+max_tokens が context 超過なら 400 |
| `max_completion_tokens` | s39: OpenAI chat completions 互換の別名。`max_tokens` 未指定時のみ使う (両方指定時は `max_tokens` 優先)。clamp・既定は同上 |
| `messages` (必須) | role は user/assistant。content は文字列 or `[{type:"text",...}]` |
| `system` | 文字列 or textブロック配列。ブロックの `cache_control` (P805) は受理して無視する |
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
| `thinking` | s47/s52/s53c: `"off"` / `"brief"` / `"full"` の文字列、または Anthropic マップ `{"type":"enabled",...}` / `{"type":"disabled"}`。**どちらのマップも `off` 扱い** (s53c: 思考無効化のみ。`brief` の簡潔指示は注入されない)。`budget_tokens` 等の他フィールドは無視 (決定的)。**未指定 = `brief`**。未知の type や文字列/マップ以外は 400 (下記 §thinking 制御) |
| `metadata` 他 | 受け流す(パースのみ)。未知のフィールドは**拒否しない** (serde 既定 = 未知フィールド無視) |

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
  "usage": {"input_tokens": 12, "output_tokens": 34,
            "cache_read_input_tokens": 0, "cache_creation_input_tokens": 12}
}
```

`stop_reason`: `end_turn` / `max_tokens` / `stop_sequence` / `tool_use`(tools 使用時)

`usage.cache_*` は P805: KV prefix cache の再利用量。`cache_read_input_tokens` は
prefill を省いた先頭トークン数、`cache_creation_input_tokens` は新規に prefill した
トークン数 (`input_tokens = cache_read + cache_creation`)。SSE では
`message_start.message.usage` に同じ 3 値が入る。

## POST /v1/messages/count_tokens

P805: Anthropic 互換のトークン数見積り。生成と同じ正規化 (system / tools /
`thinking` の既定)・同じトークナイザで数え、**prefill も KV 消費もしない**。

リクエストは `/v1/messages` と同じ形 (`model` / `messages` 必須、`system` /
`tools` / `tool_choice` / `thinking` は任意)。`max_tokens` 等は無視される。

```json
{"input_tokens": 128}
```

値は同じ本文の `/v1/messages` が返す `usage.input_tokens` と一致する。
不正な本文は 400、未知モデルは 404。生成中でエンジンロックが 10 秒取れない
場合は 503 overloaded_error。

## POST /v1/chat/completions (OpenAI 互換)

P793: OpenAI Chat Completions API 互換の経路。`/v1/messages` と**同じ** Scheduler・
chat template・KV prefix cache・engine を通る (違うのはワイヤ形式だけ)。既定の
`thinking` も `/v1/messages` と同じ (`brief` = 思考無効 + 短答誘導文) なので、
同じ会話を両方へ投げると本文・usage は一致する。

| フィールド | 対応 |
| --- | --- |
| `model` (任意) | serve 時の `--name`。不一致は 404 (`error.type = invalid_request_error`)。空/未指定は受理 |
| `messages` (必須) | role は `system` / `developer` / `user` / `assistant` / `tool` / `function`。`system`・`developer` は system プロンプトへ、`tool`・`function` は `<tool_response>...</tool_response>` を包んだ user 発話として履歴へ織り込む。assistant の `tool_calls[]` は本文の後ろへ `<tool_call>` XML として連結 (`/v1/messages` の tool_use block と同一テキスト)。未知の role は 400 (`param = messages[i].role`)。空配列は 400 |
| `content` | 文字列 or content parts 配列 (`{"type":"text","text":...}` / `"input_text"`)。text 以外 (image_url 等) は無視。null は空文字 |
| `max_tokens` / `max_completion_tokens` | `max_tokens` 優先。未指定は 1024、明示は上限 4096 で clamp (どちらも `/v1/messages` と同一) |
| `temperature` / `top_p` | 対応 (`--temperature-cap` のクランプも同一) |
| `stop` | 文字列 or 文字列配列 (空文字は無視) |
| `stream` | true で SSE |
| `stream_options.include_usage` | true で最終 usage chunk (choices が空の chunk) を追加 |
| `tools` / `tool_choice` | P797: OpenAI 形式 (`{"type":"function","function":{name,description,parameters}}`) を受理し、chat template の `tools` 変数へ渡す。`tool_choice` は `auto`/`none`/`required`/`{"type":"function","function":{"name":...}}` (詳細は下の「OpenAI tool calling」) |
| `response_format` | `{"type":"text"}` (無変換) / `{"type":"json_object"}` (JSON オブジェクトのみ返す指示を system へ追加) / `{"type":"json_schema", ...}` (P837: スキーマ指示。下の「Structured Outputs」)。それ以外の type は 400 (`param = response_format.type`) |
| その他 | 未知フィールドは無視 (serde 既定) |

応答 (非ストリーミング):

```json
{
  "id": "chatcmpl-1a2b...",
  "object": "chat.completion",
  "created": 1763900000,
  "model": "denno-test",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "..."},
    "logprobs": null,
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 12, "completion_tokens": 34, "total_tokens": 46}
}
```

`finish_reason`: `stop` (`end_turn` / `stop_sequence`) / `length` (`max_tokens`) /
`tool_calls` (tool call を返したとき)。

### OpenAI tool calling (P797)

`tools` を指定すると、モデル出力中の `<tool_call>` block を抽出して OpenAI 形式の
`tool_calls` として返す (`/v1/messages` の tool_use と同じ抽出器)。

非ストリーミング応答の `message`:

```json
{
  "role": "assistant",
  "content": "Checking.",
  "tool_calls": [{
    "id": "call_1a2b...",
    "type": "function",
    "function": {"name": "get_weather", "arguments": "{\"city\":\"Tokyo\"}"}
  }]
}
```

- `arguments` は JSON オブジェクトの **文字列** (`/v1/messages` の `input` と同じ値)
- block を除去した残りテキストは `content` に入る (残らなければ `null`)
- `finish_reason` は `tool_calls`

ストリーミングでは `delta.tool_calls` を OpenAI 仕様どおりに流す:

```
data: {...,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_...","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}
data: {...,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":\"Tokyo\"}"}}]},"finish_reason":null}]}
data: {...,"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}
```

`index` は tool call の通し番号 (0 始まり)。`id`/`type`/`function.name` は
その call の最初の chunk にだけ載り、`function.arguments` は増分だけを載せる
(クライアントは連結して JSON にする)。tool call を含むストリームでは
`finish_reason` が `tool_calls` になる。

- `tool_choice`: `auto` (既定) / `none` (モデルへ非公開・抽出もしない) /
  `required` (= `/v1/messages` の `any`。tool_call block までの拘束デコード) /
  `{"type":"function","function":{"name":"..."}}` (= `{"type":"tool","name":...}`)。
  未定義の名前は 400、`tools` 無しでの `tool_choice` は 400 (`param = tool_choice`)
- 履歴の `assistant.tool_calls[]` は `<tool_call>` XML、`role="tool"` の
  `content` は `<tool_response>` 包みでプロンプトへ入る (モデルから見ると
  `/v1/messages` の tool_use / tool_result と同じ形)。`arguments` が JSON
  オブジェクトの文字列でない履歴は 400 (`param = messages[i].tool_calls[j]`)
- `tool_call_id` は受理するが chat template は参照しない (対応検証はしない)

ストリーミング (`stream: true`)。`event:` 行は付けず `data:` 行だけを流し、
最後に `data: [DONE]` で終わる:

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"denno-test","choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk",...,"choices":[{"index":0,"delta":{"content":"..."},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk",...,"choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"stop"}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk",...,"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":34,"total_tokens":46}}

data: [DONE]
```

(usage chunk は `stream_options.include_usage: true` のときだけ。それ以外の chunk に
`usage` は入らない。) keep-alive は SSE コメント行 (`: ping`) で送るため、
`data:` を JSON として読むクライアントには見えない。生成中のクライアント切断は
scheduler の切断スイープで検知し、KV スロットを解放する (後続リクエストは通る)。

エラーは OpenAI 形式:

```json
{"error": {"message": "...", "type": "invalid_request_error", "param": "messages", "code": null}}
```

400 = リクエスト不正 (本文/role/messages)。404 = 未知モデル。503 = キュー溢れ。
500 = engine 側の失敗。

### Structured Outputs (`response_format: json_schema`, P837)

`response_format` に `json_schema` を渡すと、スキーマを検証したうえで
**スキーマ本文を system プロンプトへ埋め込んで** モデルに JSON を書かせる。

```json
{
  "model": "denno-test",
  "messages": [{"role": "user", "content": "Tokyo の天気は?"}],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "weather",
      "description": "weather answer",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}, "temp_c": {"type": "number"}},
        "required": ["city", "temp_c"],
        "additionalProperties": false
      }
    }
  }
}
```

検証 (失敗は 400、`error.type = invalid_request_error`、
`error.param` は `response_format.json_schema`):

- `json_schema` はオブジェクト。`name` は必須・非空文字列。`description` は文字列。
  `strict` は真偽値。`schema` は必須で、JSON オブジェクトまたは真偽値。
- `schema` は 16 KiB 以内、ネストは 16 段以内。
- `$ref` は **常に 400** (参照解決の機構を持たず、解決せずに渡すと意味が変わるため)。
- `type` は `object` / `array` / `string` / `number` / `integer` / `boolean` / `null` の
  文字列のみ (型の配列形式は非対応)。

`strict: true` の追加制約 (満たさなければ **400**。黙って制約を捨てない):

- `oneOf` / `anyOf` / `allOf` / `not` / `if` / `then` / `else` /
  `patternProperties` / `$defs` / `definitions` / `prefixItems` /
  `unevaluatedProperties` / `unevaluatedItems` / `dependentSchemas` は非対応。
- `type: "object"` かつ `properties` を持つノードは `additionalProperties: false` と、
  全 property を列挙した `required` を要求する (OpenAI の strict と同じ規則)。

`strict` が false / 未指定なら上記キーワードは **受理して無視する**
(スキーマ JSON はそのままプロンプトへ渡るが、このサーバーは検査も保証もしない)。

**保証の範囲 (重要)**: この経路は **トークンレベルの拘束デコードを行わない**。
engine の拘束デコード (`toolmask.rs`) は `<tool_call>` XML スキャフォールド専用で、
任意の JSON Schema から許容トークン集合を組む機構は無い。したがって
「スキーマに適合した JSON が必ず出る」ことは保証しない — プロンプト指示と
リクエスト時のスキーマ検査までを行う。生成結果の再検証 (自動リトライ) もしない。
`json_object` / `text` の挙動は P797 から不変で、`/v1/messages` には影響しない。

## POST /v1/completions (OpenAI 旧 Completions, P837)

chat template を **通さない** raw prompt の続き生成。IDE の補完 / FIM や
評価ハーネスのように、役割ラベルを足したくないクライアント向け。
scheduler・KV prefix cache・サンプラ・usage の数え方は chat 経路と同一で、
違うのは「プロンプトの組み立て (テンプレート無し)」と「ワイヤ形式」だけ。

| フィールド | 対応 |
| --- | --- |
| `model` (任意) | serve 時の `--name`。不一致は 404 |
| `prompt` (必須) | 文字列 or 1 要素の文字列配列。空文字は 400。2 要素以上 (batch) は 400 (`param = prompt`)、トークン ID 配列は 400 (`param = prompt` / `prompt[0]`) |
| `max_tokens` / `max_completion_tokens` | `max_tokens` 優先。未指定は 1024、上限 4096 で clamp (chat と同じ) |
| `temperature` / `top_p` | 対応 (`--temperature-cap` のクランプも同一) |
| `stop` | 文字列 or 文字列配列 (空文字は無視)。打ち切った stop 列は `text` に含めない |
| `stream` | true で SSE |
| `stream_options.include_usage` | true で最終 usage chunk (choices が空) を追加 |
| `echo` | true なら `text` の先頭にプロンプト自身を付ける (SSE では最初の chunk) |
| その他 | 未知フィールドは無視 (serde 既定) |

非ストリーミング応答:

```json
{
  "id": "cmpl-1a2b3c4d5e6f7a8b9c0d1e2f",
  "object": "text_completion",
  "created": 1763900000,
  "model": "denno-test",
  "choices": [{
    "text": " ...",
    "index": 0,
    "logprobs": null,
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 12, "completion_tokens": 34, "total_tokens": 46}
}
```

`finish_reason` は `stop` (`end_turn` / `stop_sequence`) か `length` (`max_tokens`)。

SSE (`stream: true`)。`event:` 行は付けず `data:` 行だけを流し、最後に
`data: [DONE]` で終わる:

```
data: {"id":"cmpl-...","object":"text_completion","created":...,"model":"denno-test","choices":[{"text":" ...","index":0,"logprobs":null,"finish_reason":null}]}

data: {"id":"cmpl-...","object":"text_completion",...,"choices":[{"text":"","index":0,"logprobs":null,"finish_reason":"stop"}]}

data: [DONE]
```

(usage chunk は `stream_options.include_usage: true` のときだけ、最終 chunk の後ろに
`choices: []` で入る。`echo: true` のときはプロンプトが最初の chunk の `text` になる。)

エラーは chat 経路と同じ OpenAI 形式。400 = リクエスト不正 (prompt/model 以外の
型不正を含む)。404 = 未知モデル。503 = キュー溢れ。500 = engine 側の失敗。

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
  - `full`: 従来経路のまま (`enable_thinking` 変数を渡さない = 現行テンプレートでは思考 ON。
    注入なし・prompt はビット同一・アンカー不変)
- Anthropic マップ形式 (s52・Hermes 互換):
  - `{"type":"enabled", ...}` も `{"type":"disabled"}` も **`off` 扱い** (s53c 以降。
    s52 では enabled は `brief` 扱いだった。思考無効化のみで簡潔指示は注入しない)
  - `budget_tokens` 等の他フィールドは無視 (決定的)
  - 未知の `type` (欠落・非文字列含む) は 400 invalid_request_error
- 文字列 / オブジェクト以外 (数値・真偽値・配列等) も 400

**サーバー側の安全弁** (どちらも未指定なら従来とビット完全一致):
`denno serve --repetition-penalty <f32>` (§serve 既定値) と
`denno serve --temperature-cap <f32>` (実効 temperature を `min(要求値, cap)` に頭打ち。
`temperature` 未指定時は sampler 既定の `1.0` をクランプ)。負値・非有限値は起動時に拒否する。
`--temperature-cap 0.7` は「thinking 有効扱いのクライアントが `temperature: 1` を機械的に送る」
ケース (Thinking マップ = off 扱いなのに T=1 だけが残る) への対策として推奨。

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

### serve 既定値 (s56b)

`denno serve --repetition-penalty <f32>` を付けると、**リクエストが `repetition_penalty` を
指定しなかった場合のみ**その値が適用される (リクエスト指定が常に優先)。未指定の起動では
`None` のままなので出力は従来とビット完全一致する。`0` 以下・非有限値は起動時に拒否する。

根拠 (実測): Qwen3.5-9B int8 は「長い system (約 2.6k tok) + tools 9 個 (約 3.6k tok)」の
greedy 経路で出力が崩壊する (`.showError warning` 等の混入・偽 `user` ターン)。
同じ 6k 文脈でも tools 無しでは正常、tools があってもスキーマを縮めると正常。
`--repetition-penalty 1.05` を既定にすると同一プロンプトで正常な短答が得られる。

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

### serve の temperature 上限 (s67b)

`denno serve --temperature-cap <f32>` を付けると、**実効 temperature をこの値で頭打ち**にする
(`min(要求値, cap)`)。リクエストが `temperature` を省略した場合も sampler 既定の `1.0` を
`cap` でクランプする。**未指定 (既定 `None`) なら要求値がそのまま渡るので出力は従来とビット完全一致**。
負値・非有限値は起動時に拒否する。

根拠 (実測): Hermes などのクライアントは thinking を有効にすると Anthropic 互換 API 向けに
`temperature: 1` を機械的に付ける。denno は s52 の仕様で `thinking: {"type":"enabled"}` を
**off 扱い**にするため、思考は無効なのに `T=1` だけが残る。Qwen3.5-9B int8 に
「system 11,949 文字 + tools 26 個 (計 7.6k tok)」を渡すと、`T=1.0` では 2〜3 ターンで
中国語・ハングル等が混入して崩壊する (`…スケジュール bi` / `프로그래밍` / `Web 欢迎大家阅读`) が、
`T=0.7` では同一プロンプトで 4 ターンとも正常。運用では `--temperature-cap 0.7` を推奨。
なお本フラグはサーバー側の安全弁であり、クライアントが送る `temperature` の意味自体は変えない。

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
| overloaded_error | 503 (s 番号なし: P4 の `--max-queue` 上限超過時) |
| api_error | 500 |

- リクエスト本文が JSON として不正 / 必須フィールド欠落 (serde エラー) は 400 invalid_request_error
  (`invalid request body: ...`)。**未知のフィールドはエラーにしない** (無視する)
- 生成開始後に engine 側で判明したリクエスト不正 (prompt + max_tokens が context / KV スロット上限を
  超える等) は、メッセージに `invalid_request_error:` を含む場合のみ 400 に写像し、
  それ以外の実行時失敗は 500 api_error になる (`crates/denno-server/src/lib.rs` の `messages` ハンドラ)
- キュー溢れ (実行中 1 件に加えて `--max-queue` 件が待ち行列にいる) は 503 overloaded_error
  (Anthropic の `overloaded_error` と同じ type 名・`server busy: request queue is full ...`)。
  上表 5 種が s101 時点でコードに存在する全 type

認証: `--api-key` 指定時は `x-api-key` ヘッダ(または `Authorization: Bearer ...`)を検証する。
`anthropic-version` ヘッダは検証せず受け流す。

## その他

- `GET /v1/models`: serve 中のモデルID と context_length。P793 から OpenAI 形式
  (`{"object":"list","data":[{"id","object":"model","created","owned_by":"denno"}]}`) の
  フィールドも併記する (既存の `type` / `display_name` / `context_length` / `has_more` は
  そのまま — Anthropic 側クライアントも同じ応答で動く)
- `GET /health`: `{"status": "ok"}`

## GET /metrics (Prometheus)

serve_flash (`denno flash-serve`、本番 8098) の Prometheus text format 0.0.4。
`Content-Type: text/plain; version=0.0.4; charset=utf-8`。
ゲートは `DENNO_METRICS` (既定 on)。`0` / `false` / `no` で経路ごと消える (404)。

- スクレイプは**読み取りスレッド**が答えるので、生成中でも待たされない
- decode / prefill のホットループはロックも確保もしない (atomic の加算だけ)
- `model_name` ラベルは serve 中のモデル名。`le` のバケットは全ヒストグラム共通で
  `0.05 0.1 0.25 0.5 1 2 4 8 16 32 64 128 +Inf` 秒
- 系列名は vLLM の reader がそのまま読めるように vLLM と同じ (`vllm:` 接頭辞)。
  vLLM に対応する名前が無いものだけ `denno_` を付ける

| 名前 | 型 | 意味 |
| --- | --- | --- |
| `vllm:prompt_tokens_total` | counter | 処理したプロンプトのトークン数 (`usage.input_tokens` の積算) |
| `vllm:generation_tokens_total` | counter | 生成したトークン数 (`usage.output_tokens` の積算) |
| `vllm:request_success_total` | counter | 終端まで走ったリクエスト数 (失敗も含む。失敗は次で分かる) |
| `vllm:request_failure_total` | counter | 生成が途中で失敗したリクエスト数 |
| `vllm:num_requests_running` | gauge | いま生成中のリクエスト数 (0 か 1) |
| `vllm:num_requests_waiting` | gauge | 待ち行列の長さ。denno は 1 本ずつ処理するので常に 0 (待ちはソケットの backlog) |
| `vllm:time_to_first_token_seconds` | histogram | リクエスト受付から最初の 1 トークンまでの秒 |
| `vllm:e2e_request_latency_seconds` | histogram | リクエスト全体の秒 (tokenize + prefill + decode) |
| `vllm:inter_token_latency_seconds` | histogram | トークン間隔の平均 (1 リクエスト 1 サンプル。out < 2 は記録しない) |
| `vllm:request_prefill_time_seconds` | histogram | prefill の秒 (prefix cache で再利用した前置きは含まない) |
| `vllm:prefix_cache_queries_total` | counter | prefix cache を引いたトークン数 (= プロンプトのトークン数) |
| `vllm:prefix_cache_hits_total` | counter | うち再利用できたトークン数 |
| `vllm:prompt_tokens_cached_total` | counter | 再利用したトークン数 (= `prefix_cache_hits_total` と同じ) |
| `vllm:num_preemptions_total` | counter | 先取りの回数。denno は 1 スロットで先取りしないので常に 0 |
| `vllm:kv_cache_usage_perc` | gauge | 文脈の使用率 (いまの文脈長 / `max_ctx`、0..1) |
| `vllm:cache_config_info` | gauge | 常に 1。ラベル `kv_cache_size_tokens` に `max_ctx` |
| `denno_last_decode_ms_per_token` | gauge | 直前のリクエストの decode 速度 (ms/token) |
| `denno_last_prefill_tokens_per_second` | gauge | 直前のリクエストの prefill 速度 (トークン/秒) |
| `denno_expert_vram_hit_rate` | gauge | 直前のリクエストの expert の VRAM ヒット率 (0..1) |
| `denno_expert_disk_read_bytes_total` | counter | プロセスがストレージから読んだバイト (`/proc/self/io` の `read_bytes`。expert の重みが主。ページキャッシュに載ったぶんは増えない) |
| `denno_expert_vram_used_bytes` | gauge | VRAM に常駐している expert のバイト数 |
| `denno_expert_vram_budget_bytes` | gauge | expert の VRAM 予算 (実効上限) |
| `denno_expert_ram_tier_bytes` | gauge | ホスト RAM tier に常駐している expert のバイト数 |
| `denno_expert_ram_tier_experts` | gauge | ホスト RAM tier の expert 本数 |
| `denno_build_info` | gauge | 常に 1。ラベル `build_sha` (短縮 SHA) と `version` |
