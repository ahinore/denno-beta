//! Task 1069: `docs/openapi.yaml` の拡張パリティ検査。
//!
//! `openapi_contract.rs` (task 893) は「パス/メソッドの存在」と代表ステータスを
//! 突き合わせる。このファイルはその上に 3 つの検査を足す:
//!
//! 1. 仕様が参照する **response コンポーネント** がすべて定義済みで、
//!    空 (description/content 無し) でないこと。
//! 2. 主要 GET (`/health` `/ready` `/metrics` `/v1/models` `/api/tags`
//!    `/api/version`) の実応答が、仕様に書いた **ステータスと Content-Type**
//!    に一致すること (status は仕様に列挙されている必要があり、その status の
//!    content に実 Content-Type が含まれる必要がある)。
//! 3. `/v1/messages` `/v1/chat/completions` `/api/chat` に不正ペイロードを
//!    投げたとき、仕様の 400 応答コンポーネントが指すスキーマ
//!    (`AnthropicError` / `OpenAIError` / `OllamaError`) の必須フィールドを
//!    満たす JSON エラー封筒が返ること。
//!
//! 依存は増やさない: YAML は行ベースで読む (`openapi_contract.rs` と同じ方針)。

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{header, Request};
use axum::Router;
use denno_engine::testutil::{dummy_tokenizer, DummyModel};
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::Value;
use tower::ServiceExt;

/// 仕様書。パスはテストファイルからの相対。
const SPEC: &str = include_str!("../../../docs/openapi.yaml");

fn app() -> Router {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(DummyModel), tok, "denno-test", None);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
    let mut info = EngineInfo::from_engine(&engine.lock().unwrap());
    info.model_dir = "qwen-test-dir".to_string();
    let state = Arc::new(AppState {
        engine,
        info,
        api_key: None,
        scheduler,
        model_name: "denno-test".to_string(),
        model_id: "Qwen/Qwen2.5-3B-Instruct".to_string(),
        default_repetition_penalty: None,
        temperature_cap: None,
    });
    denno_server::router(state)
}

// ---------- 仕様パーサ (行ベース) ----------

/// `components:` 直下の `section:` から「エントリ名 → ブロック本文」を取り出す。
///
/// ブロック本文は元の字下げを保った行の連結。エントリは 4 スペース字下げの
/// `name:` 行で、次の 4 スペース以下の行で閉じる。
fn component_blocks(section: &str) -> Vec<(String, String)> {
    let mut out: Vec<(String, Vec<String>)> = Vec::new();
    let mut in_components = false;
    let mut in_section = false;
    let mut cur: Option<(String, Vec<String>)> = None;

    let flush = |out: &mut Vec<(String, Vec<String>)>,
                 cur: &mut Option<(String, Vec<String>)>| {
        if let Some((name, body)) = cur.take() {
            out.push((name, body));
        }
    };

    for line in SPEC.lines() {
        let t = line.trim_start();
        if t.is_empty() {
            continue;
        }
        let indent = line.len() - t.len();
        if indent == 0 {
            flush(&mut out, &mut cur);
            in_components = t == "components:";
            in_section = false;
            continue;
        }
        if !in_components {
            continue;
        }
        if indent == 2 {
            flush(&mut out, &mut cur);
            in_section = t.trim_end_matches(':') == section;
            continue;
        }
        if !in_section {
            continue;
        }
        if indent == 4 && t.ends_with(':') && !t.starts_with('-') {
            flush(&mut out, &mut cur);
            cur = Some((t.trim_end_matches(':').to_string(), Vec::new()));
            continue;
        }
        if let Some((_, body)) = cur.as_mut() {
            body.push(line.to_string());
        }
    }
    flush(&mut out, &mut cur);
    out
        .into_iter()
        .map(|(name, body)| (name, body.join("\n")))
        .collect()
}

/// 仕様が `$ref: '#/components/responses/X'` で参照している X を列挙する。
fn referenced_response_components() -> Vec<String> {
    let needle = "$ref: '#/components/responses/";
    let mut out = Vec::new();
    for (i, _) in SPEC.match_indices(needle) {
        let rest = &SPEC[i + needle.len()..];
        let end = rest.find('\'').expect("responses $ref が閉じていない");
        out.push(rest[..end].to_string());
    }
    out
}

/// 仕様の 1 ステータス分。
#[derive(Debug)]
struct DocStatus {
    status: u16,
    content_types: Vec<String>,
    /// ステータスが `$ref: '#/components/responses/X'` のときの X。
    response_ref: Option<String>,
}

/// `paths:` から (path, method) の responses を読む。
///
/// 字下げ: path=2, method=4, `responses:`=6, status=8, `content:`=10,
/// media type=12。description のブロック本文 (10 より深い) を media type と
/// 取り違えないよう `content:` の直後だけを拾う。
fn documented_responses(path: &str, method: &str) -> Vec<DocStatus> {
    let mut out: Vec<DocStatus> = Vec::new();
    let mut in_paths = false;
    let mut cur_path = String::new();
    let mut in_op = false;
    let mut in_responses = false;
    let mut in_content = false;

    for line in SPEC.lines() {
        let t = line.trim_start();
        if t.is_empty() {
            continue;
        }
        let indent = line.len() - t.len();
        if indent == 0 {
            in_paths = t == "paths:";
            in_op = false;
            in_responses = false;
            continue;
        }
        if !in_paths {
            continue;
        }
        if indent == 2 && t.starts_with('/') {
            cur_path = t.trim_end_matches(':').to_string();
            in_op = false;
            in_responses = false;
            continue;
        }
        if indent == 4 && cur_path == path {
            in_op = t.trim_end_matches(':').eq_ignore_ascii_case(method);
            in_responses = false;
            in_content = false;
            continue;
        }
        if !in_op {
            continue;
        }
        if indent == 6 {
            in_responses = t.trim_end_matches(':') == "responses";
            in_content = false;
            continue;
        }
        if !in_responses {
            continue;
        }
        match indent {
            8 => {
                let s = t.trim_end_matches(':').trim_matches('\'');
                if let Ok(v) = s.parse::<u16>() {
                    out.push(DocStatus {
                        status: v,
                        content_types: Vec::new(),
                        response_ref: None,
                    });
                }
                in_content = false;
            }
            10 => {
                if t == "content:" {
                    in_content = true;
                } else {
                    in_content = false;
                    if let Some(rest) = t.strip_prefix("$ref:") {
                        if let Some(n) = rest
                            .trim()
                            .strip_prefix("'#/components/responses/")
                        {
                            if let Some(last) = out.last_mut() {
                                last.response_ref = Some(n.trim_end_matches('\'').to_string());
                            }
                        }
                    }
                }
            }
            12 if in_content && t.ends_with(':') => {
                if let Some(last) = out.last_mut() {
                    last.content_types.push(t.trim_end_matches(':').to_string());
                }
            }
            _ => {}
        }
    }
    out
}

/// response コンポーネントのブロックが指す schema 名 (`$ref: '#/components/schemas/X'`)。
fn schema_name_of_response(block: &str) -> Option<String> {
    let needle = "#/components/schemas/";
    let i = block.find(needle)?;
    let rest = &block[i + needle.len()..];
    let end = rest.find('\'').unwrap_or(rest.len());
    Some(rest[..end].to_string())
}

/// `[a, b, c]` 形式のリストを分解する (YAML flow sequence の最小対応)。
fn parse_flow_list(s: &str) -> Vec<String> {
    s.trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split(',')
        .map(|x| x.trim().trim_matches('\'').trim_matches('"').to_string())
        .filter(|x| !x.is_empty())
        .collect()
}

/// スキーマブロックの `required:` を読む。
///
/// - `nested = None` … トップレベルの required (字下げ 6)。
/// - `nested = Some(key)` … `key:` (字下げ 8) の object の required (字下げ 10)。
fn required_keys(block: &str, nested: Option<&str>) -> Vec<String> {
    let mut nested_indent: Option<usize> = None;
    for line in block.lines() {
        let t = line.trim_start();
        if t.is_empty() {
            continue;
        }
        let indent = line.len() - t.len();
        match nested {
            None => {
                if indent == 6 {
                    if let Some(rest) = t.strip_prefix("required:") {
                        return parse_flow_list(rest);
                    }
                }
            }
            Some(key) => {
                if let Some(level) = nested_indent {
                    if indent == level {
                        if let Some(rest) = t.strip_prefix("required:") {
                            return parse_flow_list(rest);
                        }
                    }
                    if indent <= level - 2 {
                        nested_indent = None;
                    }
                } else if indent == 8 && t.ends_with(':') && t.trim_end_matches(':') == key {
                    nested_indent = Some(10);
                }
            }
        }
    }
    Vec::new()
}

// ---------- HTTP ヘルパ ----------

struct Reply {
    status: u16,
    content_type: String,
    body: Vec<u8>,
}

impl Reply {
    fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap_or_else(|e| {
            panic!(
                "本文が JSON でない: {e} (先頭 {:?})",
                String::from_utf8_lossy(&self.body[..self.body.len().min(80)])
            )
        })
    }
}

async fn send(app: &Router, method: &str, uri: &str, body: Option<&str>) -> Reply {
    let mut req = Request::builder().method(method).uri(uri);
    let payload = match body {
        None => Body::empty(),
        Some(b) => {
            req = req.header(header::CONTENT_TYPE, "application/json");
            Body::from(b.to_owned())
        }
    };
    let resp = app
        .clone()
        .oneshot(req.body(payload).unwrap())
        .await
        .unwrap();
    let status = resp.status().as_u16();
    let content_type = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    Reply {
        status,
        content_type,
        body: bytes.to_vec(),
    }
}

// ---------- 1. response コンポーネントの定義と非空性 ----------

#[test]
fn every_referenced_response_component_is_defined_and_non_empty() {
    let defined = component_blocks("responses");
    assert!(
        defined.len() >= 10,
        "components/responses を {} 個しか読めない (パーサ不具合)",
        defined.len()
    );

    let referenced = referenced_response_components();
    assert!(
        referenced.len() >= 10,
        "responses への $ref を {} 個しか見つけられない (パーサ不具合)",
        referenced.len()
    );

    for name in &referenced {
        let Some((_, block)) = defined.iter().find(|(n, _)| n == name) else {
            panic!("仕様が response コンポーネント {name} を参照しているが定義が無い");
        };
        // 非空: description が実文字列で、content と schema を持つこと。
        let desc = block
            .lines()
            .find_map(|l| l.trim_start().strip_prefix("description:"))
            .map(str::trim)
            .unwrap_or("");
        assert!(
            !desc.is_empty(),
            "response コンポーネント {name} の description が空"
        );
        assert!(
            block.contains("content:"),
            "response コンポーネント {name} に content が無い"
        );
        // schema は $ref でもインライン定義でもよいが、どちらかは必須。
        assert!(
            schema_name_of_response(block).is_some() || block.contains("schema:"),
            "response コンポーネント {name} に schema が無い"
        );
    }

    // 定義側も空でないこと (参照されていないものも含めて品質を保つ)。
    for (name, block) in &defined {
        let desc = block
            .lines()
            .find_map(|l| l.trim_start().strip_prefix("description:"))
            .map(str::trim)
            .unwrap_or("");
        assert!(
            !desc.is_empty(),
            "response コンポーネント {name} の description が空"
        );
        assert!(
            block.contains("content:"),
            "response コンポーネント {name} に content が無い"
        );
    }
}

// ---------- 2. GET エンドポイントの status / Content-Type ----------

#[tokio::test]
async fn documented_get_endpoints_match_status_and_content_type() {
    let app = app();
    let cases = [
        "GET /health",
        "GET /ready",
        "GET /metrics",
        "GET /v1/models",
        "GET /api/tags",
        "GET /api/version",
    ];
    for case in cases {
        let (method, path) = case.split_once(' ').unwrap();
        let reply = send(&app, method, path, None).await;
        let docs = documented_responses(path, method);
        assert!(
            !docs.is_empty(),
            "仕様の {method} {path} に responses が無い"
        );
        let statuses: Vec<u16> = docs.iter().map(|d| d.status).collect();
        let doc = docs.iter().find(|d| d.status == reply.status).unwrap_or_else(|| {
            panic!(
                "仕様の {method} {path} に status {} が無い (書かれているのは {:?})",
                reply.status, statuses
            )
        });
        assert!(
            !doc.content_types.is_empty(),
            "仕様の {method} {path} {} に content (media type) が無い",
            reply.status
        );
        assert!(
            doc.content_types
                .iter()
                .any(|t| reply.content_type.contains(t.as_str())),
            "{} {} の Content-Type が {:?} (仕様: {:?})",
            method,
            path,
            reply.content_type,
            doc.content_types
        );
        assert!(
            reply.content_type.contains("application/json")
                || reply.content_type.contains("text/plain"),
            "{} {} の Content-Type が想定外: {:?}",
            method,
            path,
            reply.content_type
        );
    }
}

// ---------- 3. 不正ペイロードのエラー封筒 ----------

/// エンドポイント → (期待する error スキーマ, 不正ペイロードの一覧)。
fn invalid_payload_cases() -> Vec<(&'static str, &'static str, Vec<&'static str>)> {
    vec![
        (
            "POST /v1/messages",
            "AnthropicError",
            vec!["", "not json at all", r#"{"model":1}"#],
        ),
        (
            "POST /v1/chat/completions",
            "OpenAIError",
            vec!["", "not json at all", r#"{"messages":"nope"}"#],
        ),
        (
            "POST /api/chat",
            "OllamaError",
            vec!["", "not json at all", r#"{"messages":"nope"}"#],
        ),
    ]
}

#[tokio::test]
async fn invalid_payloads_return_documented_error_envelopes() {
    let app = app();
    let schemas = component_blocks("schemas");

    for (case, expected_schema, payloads) in invalid_payload_cases() {
        let (method, path) = case.split_once(' ').unwrap();

        // 仕様: この operation は 400 を持ち、その 400 は response コンポーネントを
        // 参照し、そのコンポーネントの schema が期待する error スキーマであること。
        let docs = documented_responses(path, method);
        let bad = docs
            .iter()
            .find(|d| d.status == 400)
            .unwrap_or_else(|| panic!("仕様の {method} {path} に 400 が無い"));
        let resp_ref = bad
            .response_ref
            .as_deref()
            .unwrap_or_else(|| panic!("仕様の {method} {path} 400 が response コンポーネントを参照していない"));
        let (_, resp_block) = component_blocks("responses")
            .into_iter()
            .find(|(n, _)| n == resp_ref)
            .unwrap_or_else(|| panic!("response コンポーネント {resp_ref} が未定義"));
        let schema = schema_name_of_response(&resp_block)
            .unwrap_or_else(|| panic!("{resp_ref} が schema を参照していない"));
        assert_eq!(
            schema, expected_schema,
            "{method} {path} の 400 が指すスキーマが違う"
        );

        let (_, schema_block) = schemas
            .iter()
            .find(|(n, _)| n == expected_schema)
            .unwrap_or_else(|| panic!("schema {expected_schema} が未定義"));
        let top_required = required_keys(schema_block, None);
        let err_required = required_keys(schema_block, Some("error"));
        assert!(
            !top_required.is_empty(),
            "schema {expected_schema} に required が無い"
        );
        // OllamaError.error は文字列なので入れ子の required は無い。
        if expected_schema != "OllamaError" {
            assert!(
                !err_required.is_empty(),
                "schema {expected_schema}.error に required が無い"
            );
        }

        for payload in payloads {
            let reply = send(&app, method, path, Some(payload)).await;
            assert_eq!(
                reply.status, 400,
                "{method} {path} の不正ペイロード {payload:?} が 400 でない"
            );
            assert!(
                reply.content_type.contains("application/json"),
                "{method} {path} のエラー Content-Type が {:?}",
                reply.content_type
            );
            let v = reply.json();
            let obj = v
                .as_object()
                .unwrap_or_else(|| panic!("{method} {path} のエラー本文が object でない"));
            for key in &top_required {
                assert!(
                    obj.contains_key(key),
                    "schema {expected_schema} の必須 {key} が本文に無い: {v}"
                );
            }
            let err = obj
                .get("error")
                .unwrap_or_else(|| panic!("{method} {path} のエラーに error が無い: {v}"));
            match expected_schema {
                // {"error": "..."} — 文字列そのもの。
                "OllamaError" => {
                    let msg = err
                        .as_str()
                        .unwrap_or_else(|| panic!("OllamaError.error が文字列でない: {v}"));
                    assert!(!msg.is_empty(), "OllamaError.error が空: {v}");
                }
                // {"error": {"type","message",...}} — プロトコル別の封筒。
                _ => {
                    let e = err
                        .as_object()
                        .unwrap_or_else(|| panic!("{expected_schema}.error が object でない: {v}"));
                    for key in &err_required {
                        assert!(
                            e.contains_key(key),
                            "schema {expected_schema}.error の必須 {key} が無い: {v}"
                        );
                    }
                    let ty = e
                        .get("type")
                        .and_then(Value::as_str)
                        .unwrap_or_else(|| panic!("{expected_schema}.error.type が文字列でない: {v}"));
                    assert!(!ty.is_empty(), "{expected_schema}.error.type が空: {v}");
                    let msg = e
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or_else(|| panic!("{expected_schema}.error.message が文字列でない: {v}"));
                    assert!(!msg.is_empty(), "{expected_schema}.error.message が空: {v}");
                }
            }
            if expected_schema == "AnthropicError" {
                // schema の enum に invalid_request_error が載っていること。
                assert!(
                    schema_block.contains("invalid_request_error"),
                    "AnthropicError.type の enum に invalid_request_error が無い"
                );
                assert_eq!(
                    err["type"], "invalid_request_error",
                    "Anthropic の 400 は invalid_request_error であるべき: {v}"
                );
                assert_eq!(
                    obj["type"], "error",
                    "Anthropic の封筒は最上位 type=error であるべき: {v}"
                );
            }
        }
    }
}
