//! P1016: OpenAI 互換 `POST /v1/chat/completions` の tool call まわり。
//!
//! - 1 応答内の `tool_calls[].id` が必ず一意 (`call_` + 24hex) であること
//!   (非ストリーム / ストリーム両方)
//! - 壊れた tool 定義 (`tools[]`) が param 付き 400 になること
//! - `tool_choice` の none / required / 名前指定がストリームと非ストリームで
//!   同じ結果 (tool_calls の有無・finish_reason・引数) になること
//!
//! `denno-engine` の tool 用ダミー (ByteLevel デコーダ + `ScriptModel`) を使い、
//! モデルが実際に `<tool_call>` block を出力する状況を HTTP ワイヤで見る。

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{tool_tokenizer, ScriptModel, TOOL_VOCAB};
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

fn tools_json() -> Value {
    json!([{
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
    }, {
        "type": "function",
        "function": {"name": "run_tool", "description": "Run the tool"}
    }])
}

fn chat_body(stream: bool, tools: Option<Value>) -> Value {
    let mut body = json!({
        "model": "denno-test",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 64,
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

// ---------------------------------------------------------------------------
// 1. id の一意性
// ---------------------------------------------------------------------------

#[tokio::test]
async fn non_stream_tool_call_ids_are_unique_per_response() {
    let app = script_app(multi_block_script(&["get_weather", "run_tool", "get_weather"]));
    let (status, v) = post(&app, chat_body(false, Some(tools_json()))).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
    let calls = v["choices"][0]["message"]["tool_calls"]
        .as_array()
        .expect("tool_calls が配列でない");
    assert_eq!(calls.len(), 3, "calls = {calls:?}");

    let ids: Vec<String> = calls
        .iter()
        .map(|c| c["id"].as_str().expect("id が文字列でない").to_string())
        .collect();
    for id in &ids {
        assert!(is_call_id(id), "id の形が OpenAI 互換でない: {id}");
    }
    // 名前が重複していても id は応答内で一意
    let mut uniq = ids.clone();
    uniq.sort();
    uniq.dedup();
    assert_eq!(uniq.len(), ids.len(), "id が重複: {ids:?}");
    // 引数も各 call で独立している
    for c in calls {
        let args = c["function"]["arguments"].as_str().expect("arguments が文字列でない");
        let parsed: Value = serde_json::from_str(args).expect("arguments が JSON でない");
        assert_eq!(parsed, json!({"city": "Tokyo"}));
    }
}

#[tokio::test]
async fn stream_tool_call_ids_are_unique_per_response() {
    let app = script_app(multi_block_script(&["get_weather", "run_tool", "get_weather"]));
    let (status, text) = post_sse(&app, chat_body(true, Some(tools_json()))).await;
    assert_eq!(status, StatusCode::OK, "text = {text}");
    assert!(text.trim_end().ends_with("data: [DONE]"), "text = {text}");
    let chunks = sse_chunks(&text);

    let mut starts: Vec<(u64, String, String)> = Vec::new(); // (index, id, name)
    let mut args: std::collections::BTreeMap<u64, String> = Default::default();
    for c in &chunks {
        let Some(tcs) = c["choices"][0]["delta"]["tool_calls"].as_array() else {
            continue;
        };
        for tc in tcs {
            let index = tc["index"].as_u64().expect("index が無い");
            if let Some(name) = tc["function"]["name"].as_str() {
                let id = tc["id"].as_str().expect("start に id が無い").to_string();
                starts.push((index, id, name.to_string()));
            } else {
                assert!(tc.get("id").is_none(), "id が開始以外で再送されている: {tc}");
                let a = tc["function"]["arguments"].as_str().unwrap_or_default();
                args.entry(index).or_default().push_str(a);
            }
        }
    }
    assert_eq!(starts.len(), 3, "starts = {starts:?}");
    assert_eq!(
        starts.iter().map(|s| s.0).collect::<Vec<_>>(),
        vec![0, 1, 2],
        "index が送出順でない: {starts:?}"
    );
    for (_, id, _) in &starts {
        assert!(is_call_id(id), "id の形が OpenAI 互換でない: {id}");
    }
    let mut ids: Vec<String> = starts.iter().map(|s| s.1.clone()).collect();
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len(), 3, "id が重複: {starts:?}");
    // 引数は index ごとに独立して正しく組み上がる
    assert_eq!(args.len(), 3, "args = {args:?}");
    for (index, raw) in &args {
        let v: Value = serde_json::from_str(raw).expect("arguments 増分の連結が JSON でない");
        assert_eq!(v, json!({"city": "Tokyo"}), "index = {index}");
    }
}

#[tokio::test]
async fn tool_call_ids_differ_across_requests() {
    // ScriptModel は 1 回分の再生なので、リクエストごとに app を作る。
    let app1 = script_app(block_script("get_weather"));
    let (s1, v1) = post(&app1, chat_body(false, Some(tools_json()))).await;
    let app2 = script_app(block_script("get_weather"));
    let (s2, v2) = post(&app2, chat_body(false, Some(tools_json()))).await;
    assert_eq!(s1, StatusCode::OK);
    assert_eq!(s2, StatusCode::OK);
    let id1 = v1["choices"][0]["message"]["tool_calls"][0]["id"]
        .as_str()
        .unwrap()
        .to_string();
    let id2 = v2["choices"][0]["message"]["tool_calls"][0]["id"]
        .as_str()
        .unwrap()
        .to_string();
    assert_ne!(id1, id2, "別リクエストで同じ id が返った: {id1}");
}

// ---------------------------------------------------------------------------
// 2. 壊れた tool 定義は param 付き 400
// ---------------------------------------------------------------------------

#[tokio::test]
async fn malformed_tool_definitions_return_400_with_param_path() {
    let app = script_app(block_script("get_weather"));
    let cases: Vec<(&str, Value, &str)> = vec![
        (
            "name 空",
            json!([{"type": "function", "function": {"name": ""}}]),
            "tools[0].function.name",
        ),
        (
            "name 欠落",
            json!([{"type": "function", "function": {"description": "no name"}}]),
            "tools[0].function.name",
        ),
        (
            "function 欠落",
            json!([{"type": "function"}]),
            "tools[0].function.name",
        ),
        (
            "name が空白のみ",
            json!([{"type": "function", "function": {"name": "   "}}]),
            "tools[0].function.name",
        ),
        (
            "parameters が object でない",
            json!([{"type": "function", "function": {"name": "f", "parameters": [1, 2]}}]),
            "tools[0].function.parameters",
        ),
        (
            "type が function 以外",
            json!([{"type": "retrieval", "function": {"name": "f"}}]),
            "tools[0].type",
        ),
        (
            "2 番目の要素が object でない",
            json!([{"type": "function", "function": {"name": "f"}}, 42]),
            "tools[1]",
        ),
        (
            "2 番目の要素が null",
            json!([{"type": "function", "function": {"name": "f"}}, null]),
            "tools[1]",
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

#[tokio::test]
async fn well_formed_tool_definitions_still_pass() {
    let app = script_app(block_script("get_weather"));
    // parameters 未指定 (= 空 schema) と description のみは従来どおり受理
    let tools = json!([
        {"type": "function", "function": {"name": "get_weather"}},
        {"function": {"name": "run_tool", "description": "d"}}
    ]);
    let (status, v) = post(&app, chat_body(false, Some(tools))).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
}

// ---------------------------------------------------------------------------
// 3. tool_choice の stream / 非 stream 一致
// ---------------------------------------------------------------------------

/// 非ストリーム応答から (finish_reason, content, calls[(name, arguments)]) を取る。
fn non_stream_shape(v: &Value) -> (String, String, Vec<(String, String)>) {
    let choice = &v["choices"][0];
    let finish = choice["finish_reason"].as_str().unwrap_or_default().to_string();
    let msg = &choice["message"];
    let content = msg["content"].as_str().unwrap_or_default().to_string();
    let calls = msg["tool_calls"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|c| {
                    (
                        c["function"]["name"].as_str().unwrap_or_default().to_string(),
                        c["function"]["arguments"].as_str().unwrap_or_default().to_string(),
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    (finish, content, calls)
}

/// SSE 応答から (finish_reason, content, calls[(name, arguments)]) を組み立てる。
fn stream_shape(text: &str) -> (String, String, Vec<(String, String)>) {
    let chunks = sse_chunks(text);
    let mut content = String::new();
    let mut names: std::collections::BTreeMap<u64, String> = Default::default();
    let mut args: std::collections::BTreeMap<u64, String> = Default::default();
    let mut finish = String::new();
    for c in &chunks {
        let choices = c["choices"].as_array().cloned().unwrap_or_default();
        for ch in &choices {
            if let Some(t) = ch["delta"]["content"].as_str() {
                content.push_str(t);
            }
            if let Some(tcs) = ch["delta"]["tool_calls"].as_array() {
                for tc in tcs {
                    let idx = tc["index"].as_u64().expect("index が無い");
                    if let Some(n) = tc["function"]["name"].as_str() {
                        names.insert(idx, n.to_string());
                    }
                    if let Some(a) = tc["function"]["arguments"].as_str() {
                        args.entry(idx).or_default().push_str(a);
                    }
                }
            }
            if let Some(f) = ch["finish_reason"].as_str() {
                finish = f.to_string();
            }
        }
    }
    let calls = names
        .into_iter()
        .map(|(i, n)| (n, args.remove(&i).unwrap_or_default()))
        .collect();
    (finish, content, calls)
}

async fn both_shapes(
    script: Vec<u32>,
    body: Value,
) -> (
    (String, String, Vec<(String, String)>),
    (String, String, Vec<(String, String)>),
) {
    // ScriptModel は 1 回分の再生なので、経路ごとに app を作る。
    let app = script_app(script.clone());
    let (status, v) = post(&app, body.clone()).await;
    assert_eq!(status, StatusCode::OK, "non-stream body = {v}");
    let ns = non_stream_shape(&v);
    let app = script_app(script);
    let mut sse_body = body;
    sse_body["stream"] = json!(true);
    let (status, text) = post_sse(&app, sse_body).await;
    assert_eq!(status, StatusCode::OK, "stream text = {text}");
    let st = stream_shape(&text);
    (ns, st)
}

#[tokio::test]
async fn tool_choice_none_is_identical_stream_and_non_stream() {
    // none: モデルが tool_call を吐いても抽出しない (本文は生のまま)。
    let mut body = chat_body(false, Some(tools_json()));
    body["tool_choice"] = json!("none");
    let (ns, st) = both_shapes(block_script("get_weather"), body).await;

    assert!(ns.2.is_empty(), "none なのに tool_calls が出た: {ns:?}");
    assert!(st.2.is_empty(), "none なのに stream で tool_calls が出た: {st:?}");
    assert_eq!(ns.0, "stop");
    assert_eq!(st.0, "stop");
    assert_eq!(ns.1, st.1, "本文が stream / 非 stream で違う");
    assert!(ns.1.contains("<tool_call>"), "本文に生の block が残っていない: {ns:?}");
}

#[tokio::test]
async fn tool_choice_required_is_identical_stream_and_non_stream() {
    let mut body = chat_body(false, Some(tools_json()));
    body["tool_choice"] = json!("required");
    let (ns, st) = both_shapes(block_script("get_weather"), body).await;

    assert_eq!(ns.0, "tool_calls");
    assert_eq!(st.0, "tool_calls");
    assert_eq!(ns.2.len(), 1);
    assert_eq!(st.2.len(), 1);
    assert_eq!(ns.2[0].0, "get_weather");
    assert_eq!(st.2[0].0, "get_weather");
    assert_eq!(
        serde_json::from_str::<Value>(&ns.2[0].1).unwrap(),
        serde_json::from_str::<Value>(&st.2[0].1).unwrap(),
        "引数が stream / 非 stream で違う"
    );
    assert_eq!(ns.1, st.1);
}

#[tokio::test]
async fn tool_choice_named_is_identical_stream_and_non_stream() {
    let mut body = chat_body(false, Some(tools_json()));
    body["tool_choice"] = json!({"type": "function", "function": {"name": "run_tool"}});
    let (ns, st) = both_shapes(block_script("run_tool"), body).await;

    assert_eq!(ns.0, "tool_calls");
    assert_eq!(st.0, "tool_calls");
    assert_eq!(ns.2.len(), 1);
    assert_eq!(st.2.len(), 1);
    assert_eq!(ns.2[0].0, "run_tool");
    assert_eq!(st.2[0].0, "run_tool");
    assert_eq!(
        serde_json::from_str::<Value>(&ns.2[0].1).unwrap(),
        serde_json::from_str::<Value>(&st.2[0].1).unwrap()
    );
    assert_eq!(ns.1, st.1);
}
