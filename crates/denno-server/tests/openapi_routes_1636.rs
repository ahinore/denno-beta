//! task 1636: `docs/openapi.yaml` と denno-server のルータの全経路監査。
//!
//! `tests/openapi_contract.rs` (task 893/1026) は代表ステータスだけを見る。
//! このファイルは「登録されている全ルート」を 1 本ずつ実際に叩き、仕様との
//! 4 方向のズレを落とす:
//!
//! 1. ルータ → 仕様: `router()` / サブルータが登録する (method, path) が
//!    仕様の `x-denno-impl: engine` operation と過不足なく一致する。
//! 2. 仕様 → 実応答: 各 operation に最小リクエストを投げ、返ったステータスが
//!    その operation に書かれていること。ステータスごとの Content-Type も
//!    仕様の記載と一致すること。
//! 3. 認証: `api_key` 有効時に 401 を返すルートは仕様にも 401 が書かれ、
//!    401 を返さないルートには書かれていないこと (双方向)。
//! 4. スキーマ: 200 の JSON 応答と 4xx のエラー封筒が、仕様が `$ref` する
//!    スキーマの `required` を満たすこと。
//!
//! YAML ライブラリは使わない (依存を増やさない)。行ベースのパーサは
//! `openapi_contract.rs` と同じ方針だが、このファイル内で独立に閉じている。

use std::collections::BTreeMap;
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

/// 仕様書 (テストファイルからの相対)。
const SPEC: &str = include_str!("../../../docs/openapi.yaml");
/// メインルータ。
const ROUTER_SRC: &str = include_str!("../src/lib.rs");
/// `.merge` される Ollama サブルータ (`/api/show` / `/api/ps`)。
const OLLAMA_SRC: &str = include_str!("../src/ollama.rs");

/// 監査対象の登録ルート。パーサがサブルータや定数解決を静かに落としたら
/// ここで気づけるよう、期待集合を固定する (task 1636 時点の 18 本)。
const EXPECTED_ROUTES: &[(&str, &str)] = &[
    ("GET", "/health"),
    ("GET", "/metrics"),
    ("GET", "/ready"),
    ("GET", "/v1/models"),
    ("GET", "/v1/models/{id}"),
    ("GET", "/v1/cache/stats"),
    ("DELETE", "/v1/cache"),
    ("POST", "/v1/messages"),
    ("POST", "/v1/messages/count_tokens"),
    ("POST", "/v1/chat/completions"),
    ("POST", "/v1/completions"),
    ("GET", "/api/tags"),
    ("GET", "/api/version"),
    ("POST", "/api/chat"),
    ("POST", "/api/generate"),
    ("POST", "/api/show"),
    ("GET", "/api/ps"),
    ("GET", "/v1/telemetry/ws"),
];

// ---------------------------------------------------------------------------
// 仕様書パーサ (行ベース)
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Clone)]
struct SpecOp {
    /// `x-denno-impl` の値 (engine / flash-cli)。
    imp: Option<String>,
    /// ステータスコード → その応答が宣言する Content-Type の一覧。
    statuses: BTreeMap<u16, Vec<String>>,
    /// ステータスコード → `$ref` するスキーマ名 (あれば)。
    schemas: BTreeMap<u16, String>,
    /// ステータスコード → `$ref` する responses コンポーネント名 (あれば)。
    response_refs: BTreeMap<u16, String>,
}

/// `paths:` 配下の (METHOD, path) → SpecOp。
fn parse_paths() -> BTreeMap<(String, String), SpecOp> {
    let mut out: BTreeMap<(String, String), SpecOp> = BTreeMap::new();
    let mut in_paths = false;
    let mut cur_path = String::new();
    let mut cur_key: Option<(String, String)> = None;
    let mut cur_status: Option<u16> = None;

    for line in SPEC.lines() {
        let indent = line.len() - line.trim_start().len();
        let t = line.trim();
        if indent == 0 {
            in_paths = t == "paths:";
            cur_key = None;
            cur_status = None;
            continue;
        }
        if !in_paths || t.is_empty() {
            continue;
        }
        if indent == 2 && t.starts_with('/') && t.ends_with(':') {
            cur_path = t.trim_end_matches(':').to_string();
            cur_key = None;
            cur_status = None;
        } else if indent == 4 {
            let name = t.trim_end_matches(':');
            let method = match name {
                "get" => Some("GET"),
                "post" => Some("POST"),
                "delete" => Some("DELETE"),
                "put" => Some("PUT"),
                "patch" => Some("PATCH"),
                _ => None,
            };
            if let Some(m) = method {
                let key = (m.to_string(), cur_path.clone());
                out.entry(key.clone()).or_default();
                cur_key = Some(key);
                cur_status = None;
            } else {
                cur_key = None;
                cur_status = None;
            }
        } else if let Some(key) = cur_key.as_ref() {
            if indent == 8 && t.ends_with(':') && !t.starts_with('$') {
                let raw = t.trim_end_matches(':').trim_matches('\'');
                if let Ok(code) = raw.parse::<u16>() {
                    cur_status = Some(code);
                    out.get_mut(key).unwrap().statuses.entry(code).or_default();
                }
            } else if let Some(code) = cur_status {
                if indent == 10 && t.starts_with("$ref:") {
                    let name = ref_name(t);
                    out.get_mut(key).unwrap().response_refs.insert(code, name);
                } else if indent == 12 && t.ends_with(':') && !t.starts_with('$') {
                    let ct = t.trim_end_matches(':').to_string();
                    out.get_mut(key).unwrap().statuses.entry(code).or_default().push(ct);
                } else if indent >= 8 && t.starts_with("$ref: '#/components/schemas/") {
                    let name = ref_name(t);
                    out.get_mut(key).unwrap().schemas.insert(code, name);
                }
            } else if indent == 6 {
                if let Some(rest) = t.strip_prefix("x-denno-impl:") {
                    out.get_mut(key).unwrap().imp = Some(rest.trim().to_string());
                }
            }
        }
    }
    out
}

/// `$ref: '#/components/<section>/<Name>'` から `<Name>` を取る。
fn ref_name(line: &str) -> String {
    let v = line.split_once("#/components/").expect("$ref の形が想定外").1;
    let v = v.trim_end_matches('\'').trim_end_matches('"');
    v.rsplit('/').next().unwrap().to_string()
}

/// `components:` 直下の section → name → ブロック行。
fn parse_components(section: &str) -> BTreeMap<String, Vec<String>> {
    let mut out = BTreeMap::new();
    let mut in_section = false;
    let mut cur: Option<(String, Vec<String>)> = None;

    let flush = |cur: &mut Option<(String, Vec<String>)>,
                 out: &mut BTreeMap<String, Vec<String>>| {
        if let Some((name, body)) = cur.take() {
            out.insert(name, body);
        }
    };

    for line in SPEC.lines() {
        let indent = line.len() - line.trim_start().len();
        let t = line.trim();
        if indent == 0 {
            flush(&mut cur, &mut out);
            in_section = false;
            continue;
        }
        if !in_section && indent == 2 && t.strip_suffix(':') == Some(section) {
            in_section = true;
            continue;
        }
        if !in_section {
            continue;
        }
        if indent <= 2 {
            flush(&mut cur, &mut out);
            in_section = false;
            continue;
        }
        if indent == 4 && t.ends_with(':') {
            flush(&mut cur, &mut out);
            cur = Some((t.trim_end_matches(':').to_string(), Vec::new()));
        } else if let Some((_, body)) = cur.as_mut() {
            if !t.is_empty() {
                body.push(line.to_string());
            }
        }
    }
    flush(&mut cur, &mut out);
    out
}

/// スキーマ (object) の `required` と入れ子の property。
#[derive(Debug, Default)]
struct SchemaNode {
    required: Vec<String>,
    props: BTreeMap<String, SchemaNode>,
}

/// `Name:` 行の字下げを `entry_indent` として、そのブロックを読む。
fn parse_schema(block: &[String], entry_indent: usize) -> SchemaNode {
    let key_indent = entry_indent + 2;
    let mut node = SchemaNode::default();
    let mut in_required_list = false;

    for line in block {
        let indent = line.len() - line.trim_start().len();
        let t = line.trim();
        if indent < key_indent {
            in_required_list = false;
            continue;
        }
        if indent != key_indent {
            if in_required_list && t.starts_with("- ") {
                node.required.push(t[2..].trim().trim_matches('\'').to_string());
            }
            continue;
        }
        in_required_list = false;
        if let Some(rest) = t.strip_prefix("required:") {
            let rest = rest.trim();
            if rest.starts_with('[') {
                for part in rest.trim_matches(|c| c == '[' || c == ']').split(',') {
                    let p = part.trim().trim_matches('\'').trim_matches('"');
                    if !p.is_empty() {
                        node.required.push(p.to_string());
                    }
                }
            } else if rest.is_empty() {
                in_required_list = true;
            }
        } else if t == "properties:" {
            // 直後の key_indent+2 の name: 行が property。
            continue;
        }
    }

    // properties の下位 (key_indent+2 の name:) を拾う。
    let prop_indent = key_indent + 2;
    let mut i = 0;
    while i < block.len() {
        let line = &block[i];
        let indent = line.len() - line.trim_start().len();
        let t = line.trim();
        if indent == prop_indent && t.ends_with(':') && !t.starts_with('-') {
            let name = t.trim_end_matches(':').to_string();
            // この property の本文 = 続く indent > prop_indent の行。
            let mut body = Vec::new();
            let mut j = i + 1;
            while j < block.len() {
                let l = &block[j];
                let li = l.len() - l.trim_start().len();
                if li <= prop_indent {
                    break;
                }
                body.push(l.clone());
                j += 1;
            }
            node.props.insert(name, parse_schema(&body, prop_indent));
            i = j;
        } else {
            i += 1;
        }
    }
    node
}

/// 仕様のスキーマ名 → 解析済みツリー。
fn schemas() -> BTreeMap<String, SchemaNode> {
    parse_components("schemas")
        .into_iter()
        .map(|(name, body)| (name, parse_schema(&body, 4)))
        .collect()
}

/// `components.responses` の Content-Type 一覧。
fn response_content_types() -> BTreeMap<String, Vec<String>> {
    parse_components("responses")
        .into_iter()
        .map(|(name, body)| {
            let cts = body
                .iter()
                .filter_map(|l| {
                    let indent = l.len() - l.trim_start().len();
                    let t = l.trim();
                    (indent == 8 && t.ends_with(':') && !t.starts_with('$'))
                        .then(|| t.trim_end_matches(':').to_string())
                })
                .collect();
            (name, cts)
        })
        .collect()
}

/// `components.responses` のスキーマ名。
fn response_schemas() -> BTreeMap<String, String> {
    parse_components("responses")
        .into_iter()
        .filter_map(|(name, body)| {
            body.iter().find_map(|l| {
                let indent = l.len() - l.trim_start().len();
                (indent > 8 && l.trim().starts_with("$ref: '#/components/schemas/"))
                    .then(|| (name.clone(), ref_name(l.trim())))
            })
        })
        .collect()
}

/// `value` が `node` の `required` を満たすか。property に required があれば再帰。
fn check_required(path: &str, node: &SchemaNode, value: &Value, errs: &mut Vec<String>) {
    let Some(obj) = value.as_object() else {
        errs.push(format!("{path}: オブジェクトでない"));
        return;
    };
    for req in &node.required {
        if !obj.contains_key(req) {
            errs.push(format!("{path}: 必須キー {req} が無い (あるのは {:?})", obj.keys().collect::<Vec<_>>()));
        }
    }
    for (name, sub) in &node.props {
        if let Some(v) = obj.get(name) {
            if !sub.required.is_empty() {
                check_required(&format!("{path}.{name}"), sub, v, errs);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ルータ定義パーサ (ソース行ベース)
// ---------------------------------------------------------------------------

fn str_consts(src: &str) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for line in src.lines() {
        let t = line.trim();
        let Some(rest) = t.strip_prefix("const ") else { continue };
        let Some((name, tail)) = rest.split_once(": &str = \"") else { continue };
        let Some(end) = tail.find('"') else { continue };
        out.insert(name.trim().to_string(), tail[..end].to_string());
    }
    out
}

fn normalize(path: &str) -> String {
    path.split('/')
        .map(|seg| match seg.strip_prefix(':') {
            Some(name) => format!("{{{name}}}"),
            None => seg.to_string(),
        })
        .collect::<Vec<_>>()
        .join("/")
}

fn route_lines(src: &str, consts: &BTreeMap<String, String>) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for line in src.lines() {
        let t = line.trim();
        let Some(rest) = t.strip_prefix(".route(") else { continue };
        let (raw, tail) = if let Some(after) = rest.strip_prefix('"') {
            let end = after.find('"').expect("閉じクォートが無い");
            (after[..end].to_string(), &after[end + 1..])
        } else {
            let comma = rest.find(',').expect("定数 route のカンマが無い");
            let name = rest[..comma].trim();
            let value = consts
                .get(name)
                .unwrap_or_else(|| panic!("パス定数 {name} の定義が見つからない"));
            (value.clone(), &rest[comma..])
        };
        let method = if tail.contains("get(") {
            "GET"
        } else if tail.contains("post(") {
            "POST"
        } else if tail.contains("delete(") {
            "DELETE"
        } else if tail.contains("put(") {
            "PUT"
        } else {
            panic!("未知のメソッド種別: {t}")
        };
        out.push((method.to_string(), normalize(&raw)));
    }
    out
}

fn registered_routes() -> Vec<(String, String)> {
    let consts = str_consts(ROUTER_SRC);
    let mut out = route_lines(ROUTER_SRC, &consts);
    // `.merge` は既知の 2 本だけ。増えたら仕様の追随漏れを落とす。
    let merges: Vec<&str> = ROUTER_SRC
        .lines()
        .filter_map(|l| l.trim().strip_prefix(".merge("))
        .collect();
    assert_eq!(merges.len(), 2, "router() の .merge が 2 本でない: {merges:?}");
    assert!(merges[0].starts_with("cache_admin_router()"), "未知のサブルータ: {}", merges[0]);
    assert!(merges[1].starts_with("ollama::admin_router()"), "未知のサブルータ: {}", merges[1]);
    out.extend(route_lines(OLLAMA_SRC, &BTreeMap::new()));
    out.sort();
    out
}

// ---------------------------------------------------------------------------
// 実リクエスト
// ---------------------------------------------------------------------------

fn app(api_key: Option<&str>) -> Router {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(DummyModel), tok, "denno-test", None);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
    let mut info = EngineInfo::from_engine(&engine.lock().unwrap());
    info.model_dir = "qwen-test-dir".to_string();
    let state = Arc::new(AppState {
        engine,
        info,
        api_key: api_key.map(str::to_string),
        scheduler,
        model_name: "denno-test".to_string(),
        model_id: "Qwen/Qwen2.5-3B-Instruct".to_string(),
        default_repetition_penalty: None,
        temperature_cap: None,
    });
    denno_server::router(state)
}

/// 1 ルート分の最小リクエスト。`uri` だけテンプレートを具体値に置き換える。
struct Probe {
    method: &'static str,
    /// 仕様上のパス (パラメータは `{id}` 表記)。
    path: &'static str,
    uri: &'static str,
    body: Option<&'static str>,
}

impl Probe {
    fn op(&self) -> (String, String) {
        (self.method.to_string(), self.path.to_string())
    }
}

fn probes() -> Vec<Probe> {
    let valid_msg =
        r#"{"model":"denno-test","messages":[{"role":"user","content":"hi"}],"max_tokens":1}"#;
    vec![
        Probe { method: "GET", path: "/health", uri: "/health", body: None },
        Probe { method: "GET", path: "/metrics", uri: "/metrics", body: None },
        Probe { method: "GET", path: "/ready", uri: "/ready", body: None },
        Probe { method: "GET", path: "/v1/models", uri: "/v1/models", body: None },
        Probe { method: "GET", path: "/v1/models/{id}", uri: "/v1/models/denno-test", body: None },
        Probe { method: "GET", path: "/v1/cache/stats", uri: "/v1/cache/stats", body: None },
        Probe { method: "DELETE", path: "/v1/cache", uri: "/v1/cache", body: None },
        // 空ボディは 400 (ハンドラは生きたまま本文検証で弾く)。
        Probe { method: "POST", path: "/v1/messages", uri: "/v1/messages", body: Some("") },
        Probe { method: "POST", path: "/v1/messages", uri: "/v1/messages", body: Some(valid_msg) },
        Probe {
            method: "POST",
            path: "/v1/messages/count_tokens",
            uri: "/v1/messages/count_tokens",
            body: Some(""),
        },
        Probe {
            method: "POST",
            path: "/v1/chat/completions",
            uri: "/v1/chat/completions",
            body: Some(""),
        },
        Probe { method: "POST", path: "/v1/completions", uri: "/v1/completions", body: Some("") },
        Probe { method: "GET", path: "/api/tags", uri: "/api/tags", body: None },
        Probe { method: "GET", path: "/api/version", uri: "/api/version", body: None },
        Probe { method: "POST", path: "/api/chat", uri: "/api/chat", body: Some("") },
        Probe { method: "POST", path: "/api/generate", uri: "/api/generate", body: Some("") },
        Probe {
            method: "POST",
            path: "/api/show",
            uri: "/api/show",
            body: Some(r#"{"model":"denno-test"}"#),
        },
        Probe { method: "GET", path: "/api/ps", uri: "/api/ps", body: None },
        Probe { method: "GET", path: "/v1/telemetry/ws", uri: "/v1/telemetry/ws", body: None },
        // 未知 id の 404 も仕様に書かれていることを確かめる。
        Probe { method: "GET", path: "/v1/models/{id}", uri: "/v1/models/nope", body: None },
    ]
}

struct Answer {
    status: u16,
    content_type: String,
    body: Vec<u8>,
}

async fn call(app: &Router, p: &Probe) -> Answer {
    let mut req = Request::builder().method(p.method).uri(p.uri);
    let body = match p.body {
        None => Body::empty(),
        Some(b) => {
            req = req.header("content-type", "application/json");
            Body::from(b)
        }
    };
    let resp = app
        .clone()
        .oneshot(req.body(body).unwrap())
        .await
        .expect("ルータが応答しない");
    let status = resp.status().as_u16();
    let content_type = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let body = resp.into_body().collect().await.unwrap().to_bytes().to_vec();
    Answer { status, content_type, body }
}

fn documented_content_types(spec: &BTreeMap<(String, String), SpecOp>, op: &(String, String), status: u16) -> Vec<String> {
    let s = spec.get(op).unwrap_or_else(|| panic!("仕様に {op:?} が無い"));
    if let Some(ct) = s.statuses.get(&status) {
        if !ct.is_empty() {
            return ct.clone();
        }
    }
    if let Some(r) = s.response_refs.get(&status) {
        return response_content_types().get(r).cloned().unwrap_or_default();
    }
    Vec::new()
}

fn schema_for(spec: &BTreeMap<(String, String), SpecOp>, op: &(String, String), status: u16) -> Option<String> {
    let s = spec.get(op)?;
    if let Some(name) = s.schemas.get(&status) {
        return Some(name.clone());
    }
    let r = s.response_refs.get(&status)?;
    response_schemas().get(r).cloned()
}

// ---------------------------------------------------------------------------
// テスト
// ---------------------------------------------------------------------------

#[test]
fn registered_routes_match_the_pinned_expectation() {
    let mut got = registered_routes();
    let mut want: Vec<(String, String)> = EXPECTED_ROUTES
        .iter()
        .map(|(m, p)| (m.to_string(), p.to_string()))
        .collect();
    got.sort();
    want.sort();
    assert_eq!(got, want, "router() の登録ルートが固定表とずれている");
}

#[test]
fn every_registered_route_is_documented_and_vice_versa() {
    let spec = parse_paths();
    let mut documented: Vec<(String, String)> = spec
        .iter()
        .filter(|(_, s)| s.imp.as_deref() == Some("engine"))
        .map(|(k, _)| k.clone())
        .collect();
    documented.sort();
    let mut routed = registered_routes();
    routed.sort();
    assert_eq!(routed, documented, "登録ルートと仕様の engine operation が一致しない");

    // 全 operation に x-denno-impl: engine が付いていること。
    let engine_ops = spec.values().filter(|s| s.imp.as_deref() == Some("engine")).count();
    assert_eq!(engine_ops, routed.len(), "x-denno-impl: engine の operation 数がルート数と違う");
}

#[tokio::test]
async fn every_route_answers_a_documented_status_and_content_type() {
    let spec = parse_paths();
    let app = app(None);
    for p in probes() {
        let ans = call(&app, &p).await;
        let op = p.op();
        let s = spec.get(&op).unwrap_or_else(|| panic!("仕様に {op:?} が無い"));
        assert!(
            s.statuses.contains_key(&ans.status),
            "{} {} が {} を返した (仕様の記載: {:?})",
            op.0,
            op.1,
            ans.status,
            s.statuses.keys().collect::<Vec<_>>()
        );
        let documented = documented_content_types(&spec, &op, ans.status);
        assert!(
            !documented.is_empty(),
            "{} {} の {} に Content-Type の記載が無い",
            op.0,
            op.1,
            ans.status
        );
        assert!(
            documented.iter().any(|ct| ans.content_type.starts_with(ct)),
            "{} {} の {} の Content-Type が {:?} (仕様: {:?})",
            op.0,
            op.1,
            ans.status,
            ans.content_type,
            documented
        );
    }
}

#[tokio::test]
async fn auth_required_routes_document_401_and_only_they_do() {
    let spec = parse_paths();
    let app = app(Some("test-key-1636"));
    for p in probes() {
        // 未知 id の 404 確認用 probe は認証有効時も 401 になるので同じ規則で扱える。
        let op = p.op();
        let ans = call(&app, &p).await;
        let documented_401 = spec
            .get(&op)
            .unwrap_or_else(|| panic!("仕様に {op:?} が無い"))
            .statuses
            .contains_key(&401);
        if documented_401 {
            assert_eq!(
                ans.status, 401,
                "{} {} は仕様に 401 があるが、認証無しで {} を返した",
                op.0, op.1, ans.status
            );
        } else {
            assert_ne!(
                ans.status, 401,
                "{} {} は認証無しで 401 を返すのに仕様に 401 が無い",
                op.0, op.1
            );
        }
    }
}

#[tokio::test]
async fn success_and_error_bodies_match_documented_required_fields() {
    let spec = parse_paths();
    let schema_map = schemas();
    let app = app(None);
    let mut checked = 0usize;
    for p in probes() {
        let ans = call(&app, &p).await;
        let op = p.op();
        let Some(name) = schema_for(&spec, &op, ans.status) else { continue };
        let node = schema_map
            .get(&name)
            .unwrap_or_else(|| panic!("スキーマ {name} が見つからない"));
        let value: Value = serde_json::from_slice(&ans.body)
            .unwrap_or_else(|e| panic!("{} {} の本文が JSON でない: {e}", op.0, op.1));
        let mut errs = Vec::new();
        check_required(&format!("{} {} ({name})", op.0, op.1), node, &value, &mut errs);
        assert!(errs.is_empty(), "仕様の required を満たさない:\n{}", errs.join("\n"));
        checked += 1;
    }
    // 監査が空回りしていないこと (200 の成功応答 + 4xx 封筒の両方を見ている)。
    assert!(checked >= 15, "スキーマ検証が {checked} 件しか走っていない");
}

#[tokio::test]
async fn documented_error_envelopes_are_actually_used() {
    let spec = parse_paths();
    let app = app(None);
    // 400 を返す各系統で、仕様のエラー封筒の形 (Anthropic / OpenAI / Ollama) が
    // 実際に出ていることを固定する。
    let cases = [
        ("/v1/messages", "AnthropicError", "Anthropic error.type が無い"),
        ("/v1/chat/completions", "OpenAIError", "OpenAI error.type が無い"),
        ("/api/chat", "OllamaError", "Ollama error が文字列でない"),
    ];
    for (path, schema_name, hint) in cases {
        let p = Probe { method: "POST", path, uri: path, body: Some("") };
        let ans = call(&app, &p).await;
        assert_eq!(ans.status, 400, "{path} が 400 でない");
        let name = schema_for(&spec, &p.op(), 400)
            .unwrap_or_else(|| panic!("{path} の 400 にスキーマが無い"));
        assert_eq!(name, schema_name, "{path} の 400 のスキーマ名");
        let v: Value = serde_json::from_slice(&ans.body).expect("JSON でない");
        match schema_name {
            "AnthropicError" => {
                assert_eq!(v["type"], "error", "{hint}");
                assert!(v["error"]["type"].is_string(), "{hint}");
                assert!(v["error"]["message"].is_string(), "{hint}");
            }
            "OpenAIError" => {
                assert!(v["error"]["message"].is_string(), "{hint}");
                assert!(v["error"]["type"].is_string(), "{hint}");
            }
            "OllamaError" => assert!(v["error"].is_string(), "{hint}"),
            _ => unreachable!(),
        }
    }
}
