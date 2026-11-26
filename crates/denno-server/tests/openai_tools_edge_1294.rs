//! P1294: OpenAI 互換 `POST /v1/chat/completions` の tool 定義エッジ。
//!
//! 1016 (id 一意性 / param 付き 400 / tool_choice 整合) を補完する:
//!
//! - 10 個を超える tool 定義 (11 / 12 個) が全てテンプレートへ届くこと
//!   (`usage.prompt_tokens` が定義数に応じて増える + tool_choice で 11 番目を
//!   指定できる)
//! - `tools[]` の重複名が 400 にならず受理され、その状態でも tool_call id が
//!   応答内で一意になること (ストリーム / 非ストリーム)
//! - `function.name` 欠落・null・非文字列・空白のみの param 付き 400
//! - 並列 tool_call (5 個) の id が全て異なり、index が送出順であること
//!
//! `denno-engine` の tool 用ダミー (ByteLevel デコーダ + `ScriptModel`) を
//! 使い、モデルが実際に `<tool_call>` block を出す状況を HTTP ワイヤで見る。

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{tool_tokenizer, ScriptModel, TOOL_VOCAB};
use denno_engine::tokenizer::DennoTokenizer;
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

/// TOOL_VOCAB の piece → id。
fn tid(piece: &str) -> u32 {
    TOOL_VOCAB
        .iter()
        .find(|(p, _)| *p == piece)
        .map(|(_, id)| *id)
        .unwrap_or_else(|| panic!("vocab missing: {piece}"))
}

/// 完全な tool_call block 1 個 (`city=Tokyo`) を吐くスクリプト。
fn block_script(name: &str) -> Vec<u32> {
    [
        "Checking.",
        "<tool_call>",
        "Ċ",
        "<function=",
        name,
        ">",
        "Ċ",
        "<parameter=",
        "city",
        ">",
        "Ċ",
        "Tokyo",
        "Ċ",
        "</parameter>",
        "Ċ",
        "</function>",
        "Ċ",
        "</tool_call>",
        "eos",
    ]
    .iter()
    .map(|p| tid(p))
    .collect()
}

/// 複数 block を連結したスクリプト (最後の eos 1 個だけ)。
fn multi_block_script(names: &[&str]) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::new();
    for (i, n) in names.iter().enumerate() {
        let mut s = block_script(n);
        if i + 1 != names.len() {
            s.pop(); // 末尾 eos を外して次へ
        }
        out.extend(s);
    }
    out
}

fn script_app(script: Vec<u32>) -> Router {
    let model = ScriptModel::new(script, tid("t1"));
    let engine = Engine::from_parts(Box::new(model), tool_tokenizer().unwrap(), "denno-test", None);
    state_app(engine)
}

/// `ScriptModel` の文脈長だけを 4096 に広げたモデル。
///
/// `ScriptModel::context()` は DUMMY_CONTEXT = 64 で、`max_tokens` が
/// `64 - prompt_tokens` にクランプされるため並列 call を 4 個以上
/// (1 block 約 18 トークン) 出せない。並列 id の検証だけ文脈長を広げる。
struct LongContextScript(ScriptModel);

impl denno_engine::model::Model for LongContextScript {
    fn context(&self) -> usize {
        4096
    }

    fn eos_ids(&self) -> &[u32] {
        self.0.eos_ids()
    }

    fn forward(
        &mut self,
        tokens: &[u32],
        cache: &mut denno_engine::kv_cache::KvCache,
    ) -> anyhow::Result<Vec<f32>> {
        self.0.forward(tokens, cache)
    }

    fn new_cache(&self) -> denno_engine::kv_cache::KvCache {
        denno_engine::kv_cache::KvCache::new(1, 4, 4096)
    }
}

fn long_script_app(script: Vec<u32>) -> Router {
    let model = LongContextScript(ScriptModel::new(script, tid("t1")));
    let engine = Engine::from_parts(Box::new(model), tool_tokenizer().unwrap(), "denno-test", None);
    state_app(engine)
}

/// `tools[]` を描画する chat template 付きトークナイザ。
///
/// `tool_tokenizer()` は template を持たないため `tools` はプロンプトに一切
/// 反映されず、定義数がプロンプト長に出ない。ここだけ tools を 1 件 2 トークン
/// 以上に描く最小テンプレートを被せ、`usage.prompt_tokens` で定義の到達を観測する。
fn tools_aware_tokenizer() -> DennoTokenizer {
    let mut vocab = serde_json::Map::new();
    for (piece, id) in TOOL_VOCAB {
        vocab.insert((*piece).to_string(), json!(id));
    }
    // TOOL_VOCAB の最大 id は 32。使っていない 33 を UNK に割り当てる。
    vocab.insert("[UNK]".to_string(), json!(33));
    let tok_json = json!({
        "version": "1.0",
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {
            "type": "ByteLevel", "add_prefix_space": false,
            "trim_offsets": true, "use_regex": true
        },
        "post_processor": null,
        "decoder": {
            "type": "ByteLevel", "add_prefix_space": false,
            "trim_offsets": true, "use_regex": true
        },
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"}
    });
    let template = "{%- for t in tools %}t0 {{ t.function.name }} {% endfor %}t1";
    DennoTokenizer::from_json_str(&tok_json.to_string(), Some(template.to_string()))
        .expect("tools-aware testutil tokenizer")
}

fn script_app_with(tok: DennoTokenizer, script: Vec<u32>) -> Router {
    let model = ScriptModel::new(script, tid("t1"));
    let engine = Engine::from_parts(Box::new(model), tok, "denno-test", None);
    state_app(engine)
}

fn state_app(engine: Engine) -> Router {
    let info = EngineInfo::from_engine(&engine);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
    let state = Arc::new(AppState {
        engine,
        info,
        api_key: None,
        scheduler,
        model_name: "denno-test".to_string(),
        model_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
        default_repetition_penalty: None,
        temperature_cap: None,
    });
    denno_server::router(state)
}

async fn post(app: &Router, body: Value) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, v)
}

async fn post_sse(app: &Router, body: Value) -> (StatusCode, String) {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

fn sse_chunks(text: &str) -> Vec<Value> {
    text.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).expect("SSE data が JSON ではない"))
        .collect()
}

/// `get_weather` を先頭に置いた n 個の tool 定義 (n >= 2)。
/// 2 番目以降は `tool_<i>` という名前の合成ツール。
fn many_tools(n: usize) -> Value {
    let mut tools = vec![json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    })];
    for i in 1..n {
        tools.push(json!({
            "type": "function",
            "function": {
                "name": format!("tool_{i:02}"),
                "description": format!("Synthetic tool number {i} used by the 1294 edge test"),
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }
        }));
    }
    json!(tools)
}

fn chat_body(stream: bool, tools: Option<Value>) -> Value {
    chat_body_max(stream, tools, 64)
}

fn chat_body_max(stream: bool, tools: Option<Value>, max_tokens: u32) -> Value {
    let mut body = json!({
        "model": "denno-test",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
    });
    if let Some(t) = tools {
        body["tools"] = t;
    }
    body
}

/// `call_` + 24 桁の小文字 hex か。
fn is_call_id(s: &str) -> bool {
    s.strip_prefix("call_")
        .map(|h| h.len() == 24 && h.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()))
        .unwrap_or(false)
}

fn ids_of(calls: &[Value]) -> Vec<String> {
    calls
        .iter()
        .map(|c| c["id"].as_str().expect("id が文字列でない").to_string())
        .collect()
}

fn assert_unique(ids: &[String]) {
    let mut uniq = ids.to_vec();
    uniq.sort();
    uniq.dedup();
    assert_eq!(uniq.len(), ids.len(), "id が重複: {ids:?}");
}

// ---------------------------------------------------------------------------
// 1. 10 個を超える tool 定義が全てプロンプトへ届く
// ---------------------------------------------------------------------------

/// 12 個の定義を受理し、生成は通常どおり tool_calls で終わる。
#[tokio::test]
async fn twelve_tools_are_accepted_and_callable() {
    let app = script_app(block_script("get_weather"));
    let (status, v) = post(&app, chat_body(false, Some(many_tools(12)))).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls", "body = {v}");
    let calls = v["choices"][0]["message"]["tool_calls"]
        .as_array()
        .expect("tool_calls が配列でない");
    assert_eq!(calls.len(), 1, "calls = {calls:?}");
    assert_eq!(calls[0]["function"]["name"], "get_weather");
}

/// 定義数 1 / 11 / 12 で `usage.prompt_tokens` が単調増加する。
///
/// テンプレートは `tools[]` を 1 件ずつ描画するため、12 件目が落とされて
/// いれば 11 件時とプロンプト長が変わらない。増加を観測して「全て届く」ことを
/// 確かめる (実トークナイザで数えた値なので推測ではなく観測)。
#[tokio::test]
async fn each_tool_definition_reaches_the_prompt() {
    async fn prompt_tokens(n: usize) -> u64 {
        let app = script_app_with(tools_aware_tokenizer(), block_script("get_weather"));
        let (status, v) = post(&app, chat_body(false, Some(many_tools(n)))).await;
        assert_eq!(status, StatusCode::OK, "n={n} body = {v}");
        v["usage"]["prompt_tokens"]
            .as_u64()
            .unwrap_or_else(|| panic!("n={n}: usage.prompt_tokens が無い: {v}"))
    }

    let one = prompt_tokens(1).await;
    let eleven = prompt_tokens(11).await;
    let twelve = prompt_tokens(12).await;
    assert!(
        eleven > one,
        "11 件でも 1 件と同じプロンプト長 ({one} -> {eleven}): 定義が落ちている"
    );
    assert!(
        twelve > eleven,
        "12 件目がプロンプトに載っていない ({eleven} -> {twelve})"
    );
}

/// 12 個目の tool を `tool_choice` で名指しできる (定義リストの末尾も検証される)。
#[tokio::test]
async fn tool_choice_can_name_the_last_of_twelve_tools() {
    // モデルは tool_01 を吐く (vocab 外なので実際には get_weather のみ検証可能)。
    // ここでは「12 番目の名前が tools 一覧に存在する」ことだけを見る:
    // 定義が落ちていれば 400 "not defined in tools" になる。
    let app = script_app(block_script("get_weather"));
    let mut body = chat_body(false, Some(many_tools(12)));
    body["tool_choice"] = json!({"type": "function", "function": {"name": "tool_11"}});
    let (status, v) = post(&app, body).await;
    assert_eq!(status, StatusCode::OK, "12 番目の tool が未定義扱い: {v}");
}

// ---------------------------------------------------------------------------
// 2. 重複した tool 名
// ---------------------------------------------------------------------------

/// 同名の tool 定義は 400 にせず受理し、その状態でも id は一意。
#[tokio::test]
async fn duplicate_tool_names_are_accepted_and_ids_stay_unique() {
    let tools = json!([
        {"type": "function", "function": {"name": "get_weather", "description": "first"}},
        {"type": "function", "function": {"name": "get_weather", "description": "second"}},
        // 同じ名前で parameters が違っても受理される (現行仕様を固定する)
        {"type": "function", "function": {
            "name": "run_tool",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
        }}
    ]);
    let app = script_app(multi_block_script(&["get_weather", "get_weather", "run_tool"]));
    let (status, v) = post(&app, chat_body(false, Some(tools))).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
    let calls = v["choices"][0]["message"]["tool_calls"]
        .as_array()
        .expect("tool_calls が配列でない");
    assert_eq!(calls.len(), 3, "calls = {calls:?}");
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    assert_eq!(calls[1]["function"]["name"], "get_weather");
    assert_eq!(calls[2]["function"]["name"], "run_tool");
    let ids = ids_of(calls);
    for id in &ids {
        assert!(is_call_id(id), "id の形が OpenAI 互換でない: {id}");
    }
    assert_unique(&ids);
}

/// ストリームでも重複名の tool_call id は一意で index は 0,1,2。
#[tokio::test]
async fn duplicate_tool_names_stream_ids_stay_unique() {
    let tools = json!([
        {"type": "function", "function": {"name": "get_weather"}},
        {"type": "function", "function": {"name": "get_weather"}},
        {"type": "function", "function": {"name": "run_tool"}}
    ]);
    let app = script_app(multi_block_script(&["get_weather", "get_weather", "run_tool"]));
    let (status, text) = post_sse(&app, chat_body(true, Some(tools))).await;
    assert_eq!(status, StatusCode::OK, "text = {text}");
    assert!(text.trim_end().ends_with("data: [DONE]"), "text = {text}");

    let mut starts: Vec<(u64, String, String)> = Vec::new();
    let mut args: BTreeMap<u64, String> = BTreeMap::new();
    for c in sse_chunks(&text) {
        let Some(tcs) = c["choices"][0]["delta"]["tool_calls"].as_array() else {
            continue;
        };
        for tc in tcs {
            let index = tc["index"].as_u64().expect("index が無い");
            match tc["function"]["name"].as_str() {
                Some(name) => {
                    let id = tc["id"].as_str().expect("start に id が無い").to_string();
                    starts.push((index, id, name.to_string()));
                }
                None => {
                    assert!(tc.get("id").is_none(), "id が開始以外で再送: {tc}");
                    let a = tc["function"]["arguments"].as_str().unwrap_or_default();
                    args.entry(index).or_default().push_str(a);
                }
            }
        }
    }
    assert_eq!(starts.len(), 3, "starts = {starts:?}");
    assert_eq!(
        starts.iter().map(|s| s.0).collect::<Vec<_>>(),
        vec![0, 1, 2],
        "index が送出順でない: {starts:?}"
    );
    assert_eq!(
        starts.iter().map(|s| s.2.as_str()).collect::<Vec<_>>(),
        vec!["get_weather", "get_weather", "run_tool"]
    );
    assert_unique(&starts.iter().map(|s| s.1.clone()).collect::<Vec<_>>());
    assert_eq!(args.len(), 3, "args = {args:?}");
    for (index, raw) in &args {
        let parsed: Value = serde_json::from_str(raw).expect("arguments 増分の連結が JSON でない");
        assert_eq!(parsed, json!({"city": "Tokyo"}), "index = {index}");
    }
}

/// 重複名でも `tool_choice` の名前指定は通る (最初の一致で扱われる)。
#[tokio::test]
async fn duplicate_tool_names_work_with_named_tool_choice() {
    let tools = json!([
        {"type": "function", "function": {"name": "get_weather"}},
        {"type": "function", "function": {"name": "get_weather"}}
    ]);
    let app = script_app(block_script("get_weather"));
    let mut body = chat_body(false, Some(tools));
    body["tool_choice"] = json!({"type": "function", "function": {"name": "get_weather"}});
    let (status, v) = post(&app, body).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
}

// ---------------------------------------------------------------------------
// 3. function.name の欠落 / 不正
// ---------------------------------------------------------------------------

#[tokio::test]
async fn missing_or_invalid_function_name_returns_param_400() {
    let app = script_app(block_script("get_weather"));
    let cases: Vec<(&str, Value, &str)> = vec![
        (
            "function.name 欠落",
            json!([{"type": "function", "function": {"description": "no name"}}]),
            "tools[0].function.name",
        ),
        (
            "function 自体が欠落",
            json!([{"type": "function"}]),
            "tools[0].function.name",
        ),
        (
            "type 省略 + function 空",
            json!([{"function": {}}]),
            "tools[0].function.name",
        ),
        (
            "function.name 空文字",
            json!([{"type": "function", "function": {"name": ""}}]),
            "tools[0].function.name",
        ),
        (
            "function.name 空白のみ",
            json!([{"type": "function", "function": {"name": " \t\n "}}]),
            "tools[0].function.name",
        ),
        (
            "function.name が null",
            json!([{"type": "function", "function": {"name": null}}]),
            "tools[0]",
        ),
        (
            "function.name が数値",
            json!([{"type": "function", "function": {"name": 42}}]),
            "tools[0]",
        ),
        (
            "function が null",
            json!([{"type": "function", "function": null}]),
            "tools[0]",
        ),
        (
            "2 番目の function.name 欠落",
            json!([
                {"type": "function", "function": {"name": "get_weather"}},
                {"type": "function", "function": {"description": "no name"}}
            ]),
            "tools[1].function.name",
        ),
    ];

    for (label, tools, want_param) in cases {
        for stream in [false, true] {
            let (status, v) = post(&app, chat_body(stream, Some(tools.clone()))).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{label} (stream={stream}): {v}");
            assert_eq!(
                v["error"]["type"], "invalid_request_error",
                "{label} (stream={stream}): {v}"
            );
            assert_eq!(
                v["error"]["param"], want_param,
                "{label} (stream={stream}): {v}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 4. 並列 tool_call の id 一意性
// ---------------------------------------------------------------------------

/// 5 個の並列 call: id は 5 個とも異なり、末尾 8 hex (単調カウンタ部) も異なる。
#[tokio::test]
async fn five_parallel_tool_calls_have_distinct_ids() {
    let names = ["get_weather", "run_tool", "get_weather", "run_tool", "get_weather"];
    let app = long_script_app(multi_block_script(&names));
    let (status, v) = post(&app, chat_body_max(false, Some(many_tools(2)), 256)).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
    let calls = v["choices"][0]["message"]["tool_calls"]
        .as_array()
        .expect("tool_calls が配列でない");
    assert_eq!(calls.len(), names.len(), "calls = {calls:?}");
    assert_eq!(
        calls
            .iter()
            .map(|c| c["function"]["name"].as_str().unwrap_or_default())
            .collect::<Vec<_>>(),
        names.to_vec()
    );

    let ids = ids_of(calls);
    for id in &ids {
        assert!(is_call_id(id), "id の形が OpenAI 互換でない: {id}");
    }
    assert_unique(&ids);

    // 同一応答内の一意性は下位 8 hex の単調カウンタで構造的に保証される。
    let mut suffix = ids
        .iter()
        .map(|id| id[id.len() - 8..].to_string())
        .collect::<Vec<_>>();
    suffix.sort();
    suffix.dedup();
    assert_eq!(suffix.len(), ids.len(), "下位 8 hex が重複: {ids:?}");

    // 引数は各 call で独立して同じ内容
    for c in calls {
        let args = c["function"]["arguments"].as_str().expect("arguments が文字列でない");
        let parsed: Value = serde_json::from_str(args).expect("arguments が JSON でない");
        assert_eq!(parsed, json!({"city": "Tokyo"}));
    }
}

/// ストリーム経路でも 5 個の id が全て異なる。
#[tokio::test]
async fn five_parallel_tool_calls_stream_ids_are_distinct() {
    let names = ["get_weather", "run_tool", "get_weather", "run_tool", "get_weather"];
    let app = long_script_app(multi_block_script(&names));
    let (status, text) = post_sse(&app, chat_body_max(true, Some(many_tools(2)), 256)).await;
    assert_eq!(status, StatusCode::OK, "text = {text}");
    let mut starts: Vec<(u64, String)> = Vec::new();
    for c in sse_chunks(&text) {
        let Some(tcs) = c["choices"][0]["delta"]["tool_calls"].as_array() else {
            continue;
        };
        for tc in tcs {
            if tc["function"]["name"].as_str().is_some() {
                let index = tc["index"].as_u64().expect("index が無い");
                let id = tc["id"].as_str().expect("start に id が無い").to_string();
                starts.push((index, id));
            }
        }
    }
    assert_eq!(starts.len(), names.len(), "starts = {starts:?}");
    assert_eq!(
        starts.iter().map(|s| s.0).collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 4],
        "index が送出順でない: {starts:?}"
    );
    let ids = starts.iter().map(|s| s.1.clone()).collect::<Vec<_>>();
    for id in &ids {
        assert!(is_call_id(id), "id の形が OpenAI 互換でない: {id}");
    }
    assert_unique(&ids);
}

/// 重複名 + 並列 + tools 11 件を同時に踏んでも id は一意。
#[tokio::test]
async fn parallel_calls_with_many_duplicate_tools_keep_unique_ids() {
    let mut tools = many_tools(10)
        .as_array()
        .cloned()
        .expect("many_tools は配列");
    tools.push(json!({"type": "function", "function": {"name": "get_weather"}}));
    tools.push(json!({"type": "function", "function": {"name": "get_weather"}}));
    let names = ["get_weather"; 4];
    let app = long_script_app(multi_block_script(&names));
    let (status, v) = post(&app, chat_body_max(false, Some(json!(tools)), 256)).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
    let calls = v["choices"][0]["message"]["tool_calls"]
        .as_array()
        .expect("tool_calls が配列でない");
    assert_eq!(calls.len(), 4, "calls = {calls:?}");
    assert_unique(&ids_of(calls));
}
