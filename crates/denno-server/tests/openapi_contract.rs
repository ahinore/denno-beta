//! Task 893: `docs/openapi.yaml` と実装の契約テスト。
//!
//! 目的は「仕様書と実装のドリフト検出」。仕様書は 2 つの実装を 1 ファイルに
//! 書いているので、各 operation の `x-denno-impl` がどちらで検証するかを決める:
//!
//! - `engine`   … `crates/denno-server` の axum ルータ (`denno serve`)。
//!   このテストが実際にリクエストを投げ、**仕様に書いたステータス** が返ることを見る。
//! - `flash-cli` … `crates/denno-cli/src/serve_flash.rs` の手書き HTTP サーバ
//!   (`denno serve-flash`)。denno-server のテストバイナリからは起動できないので、
//!   ディスパッチのソース（`include_str!`）に (method, path) の対応が
//!   残っていることを見る。**振る舞いの検証ではない**（振る舞いは
//!   `crates/denno-cli` の steer テストが持つ）。
//!
//! 検査は 3 方向:
//!
//! 1. 仕様 → ルータ: `engine` の全 operation が期待ステータスを返す。
//! 2. ルータ → 仕様: `router()` が登録している全ルートが仕様に載っている。
//! 3. 仕様 → flash: `flash-cli` の全 operation がディスパッチに存在する。
//!
//! 仕様書のパーサは YAML ライブラリを使わない（新しい依存を足さないため）。
//! 対象は `paths:` 直下のパス（2 スペース字下げ）と、その直下の
//! メソッド（4 スペース字下げ）、`x-denno-impl`（6 スペース字下げ）だけ。

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{dummy_tokenizer, DummyModel};
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use tower::ServiceExt;

/// 仕様書。パスはテストファイルからの相対。
const SPEC: &str = include_str!("../../../docs/openapi.yaml");
/// flash サーバのディスパッチ（ソースレベルの突き合わせ用）。
const FLASH_SRC: &str = include_str!("../../denno-cli/src/serve_flash.rs");
/// denno-server のルータ定義（逆方向の突き合わせ用）。
const ROUTER_SRC: &str = include_str!("../src/lib.rs");

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Op {
    method: String,
    path: String,
}

impl Op {
    fn new(method: &str, path: &str) -> Self {
        Self { method: method.to_string(), path: path.to_string() }
    }
}

/// 仕様書から (method, path, impl) を取り出す。
fn parse_spec() -> Vec<(Op, String)> {
    let mut out = Vec::new();
    let mut in_paths = false;
    let mut cur_path: Option<String> = None;
    let mut cur_op: Option<Op> = None;
    let mut cur_impl: Option<String> = None;

    let flush = |out: &mut Vec<(Op, String)>,
                 op: &mut Option<Op>,
                 imp: &mut Option<String>| {
        if let Some(o) = op.take() {
            let i = imp.take().unwrap_or_else(|| {
                panic!("仕様の {} {} に x-denno-impl が無い", o.method, o.path)
            });
            out.push((o, i));
        }
    };

    for line in SPEC.lines() {
        let indent = line.len() - line.trim_start().len();
        let t = line.trim_start();
        if indent == 0 {
            if t == "paths:" {
                in_paths = true;
            } else if in_paths {
                flush(&mut out, &mut cur_op, &mut cur_impl);
                in_paths = false;
            }
            continue;
        }
        if !in_paths {
            continue;
        }
        if indent == 2 && t.starts_with('/') && t.ends_with(':') {
            flush(&mut out, &mut cur_op, &mut cur_impl);
            cur_path = Some(t.trim_end_matches(':').to_string());
            continue;
        }
        if indent == 4 {
            let m = match t.trim_end_matches(':') {
                "get" => "GET",
                "post" => "POST",
                "delete" => "DELETE",
                "put" => "PUT",
                "patch" => "PATCH",
                _ => continue,
            };
            flush(&mut out, &mut cur_op, &mut cur_impl);
            let p = cur_path.clone().expect("メソッドがパスの外にある");
            cur_op = Some(Op::new(m, &p));
            continue;
        }
        if indent == 6 {
            if let Some(rest) = t.strip_prefix("x-denno-impl:") {
                cur_impl = Some(rest.trim().to_string());
            }
        }
    }
    flush(&mut out, &mut cur_op, &mut cur_impl);
    assert!(!out.is_empty(), "仕様に paths が 1 つも無い");
    out
}

/// `denno-server::router` が登録しているルートをソースから数える。
fn parse_router_source() -> Vec<Op> {
    let mut out = Vec::new();
    for line in ROUTER_SRC.lines() {
        let t = line.trim();
        let Some(rest) = t.strip_prefix(".route(\"") else { continue };
        let Some(end) = rest.find('"') else { continue };
        let path = &rest[..end];
        let tail = &rest[end + 1..];
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
        out.push(Op::new(method, path));
    }
    assert!(!out.is_empty(), "router() の .route(...) を 1 つも読めない");
    out
}

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

/// `engine` 実装の operation に対する最小リクエストと、仕様に書いた期待値。
///
/// - `body` が `None` のときはボディ無し (GET)、`Some("")` は空ボディ。
/// - 期待ステータスは「この本文を投げたら実際に返る値」。仕様書にも同じ
///   ステータスを書いてあることを別途確認する。
struct Probe {
    op: Op,
    body: Option<&'static str>,
    status: u16,
    /// 応答の Content-Type に含まれるべき文字列。
    content_type: &'static str,
}

fn probes() -> Vec<Probe> {
    let valid_msg = r#"{"model":"denno-test","messages":[{"role":"user","content":"hi"}],"max_tokens":1}"#;
    vec![
        // 運用 (認証不要)
        Probe { op: Op::new("GET", "/health"), body: None, status: 200, content_type: "application/json" },
        Probe { op: Op::new("GET", "/ready"), body: None, status: 200, content_type: "application/json" },
        Probe { op: Op::new("GET", "/metrics"), body: None, status: 200, content_type: "text/plain" },
        // Anthropic / OpenAI / Ollama。空ボディは 400 (ルートが生きたまま
        // 本文検証で弾かれる) を仕様の 400 と突き合わせる。
        Probe { op: Op::new("GET", "/v1/models"), body: None, status: 200, content_type: "application/json" },
        Probe { op: Op::new("POST", "/v1/messages"), body: Some(""), status: 400, content_type: "application/json" },
        Probe { op: Op::new("POST", "/v1/messages"), body: Some(valid_msg), status: 200, content_type: "application/json" },
        Probe { op: Op::new("POST", "/v1/messages/count_tokens"), body: Some(""), status: 400, content_type: "application/json" },
        Probe { op: Op::new("POST", "/v1/chat/completions"), body: Some(""), status: 400, content_type: "application/json" },
        Probe { op: Op::new("POST", "/v1/completions"), body: Some(""), status: 400, content_type: "application/json" },
        Probe { op: Op::new("GET", "/api/tags"), body: None, status: 200, content_type: "application/json" },
        Probe { op: Op::new("GET", "/api/version"), body: None, status: 200, content_type: "application/json" },
        Probe { op: Op::new("POST", "/api/chat"), body: Some(""), status: 400, content_type: "application/json" },
        Probe { op: Op::new("POST", "/api/generate"), body: Some(""), status: 400, content_type: "application/json" },
        // アップグレードヘッダ無しの WS は 400。
        Probe { op: Op::new("GET", "/v1/telemetry/ws"), body: None, status: 400, content_type: "text/plain" },
    ]
}

/// 仕様書の operation 直下（8 スペース字下げ）にある `'NNN':` を集める。
fn documented_statuses(op: &Op) -> Vec<u16> {
    let mut out = Vec::new();
    let mut in_paths = false;
    let mut cur_path = String::new();
    let mut in_op = false;
    for line in SPEC.lines() {
        let indent = line.len() - line.trim_start().len();
        let t = line.trim_start();
        if indent == 0 {
            in_paths = t == "paths:";
            in_op = false;
            continue;
        }
        if !in_paths {
            continue;
        }
        if indent == 2 && t.starts_with('/') {
            cur_path = t.trim_end_matches(':').to_string();
            in_op = false;
        } else if indent == 4 {
            // メソッド行 (get/post/delete/...) だけを op の開始とみなす。
            let name = t.trim_end_matches(':');
            let is_method = matches!(name, "get" | "post" | "delete" | "put" | "patch");
            in_op = is_method
                && cur_path == op.path
                && name.to_uppercase() == op.method;
        } else if indent == 8 && in_op {
            let s = t.trim_end_matches(':');
            let s = s.trim_matches('\'');
            if let Ok(v) = s.parse::<u16>() {
                out.push(v);
            }
        }
    }
    out
}

#[tokio::test]
async fn every_documented_engine_route_answers_as_specified() {
    let app = app();
    let spec = parse_spec();
    let engine_ops: Vec<Op> = spec
        .iter()
        .filter(|(_, i)| i == "engine")
        .map(|(o, _)| o.clone())
        .collect();
    assert!(!engine_ops.is_empty(), "engine 実装の operation が仕様に無い");

    let probes = probes();
    // 仕様とテスト表が 1:1 であること (どちらかの追加漏れを落とす)。
    let mut probe_ops: Vec<Op> = probes.iter().map(|p| p.op.clone()).collect();
    let mut spec_ops: Vec<Op> = engine_ops.clone();
    probe_ops.sort();
    spec_ops.sort();
    probe_ops.dedup();
    spec_ops.dedup();
    assert_eq!(spec_ops, probe_ops, "仕様の engine operation とこのテストの対応表がずれている");

    for p in &probes {
        let mut req = Request::builder().method(p.op.method.as_str()).uri(p.op.path.as_str());
        let body = match p.body {
            None => Body::empty(),
            Some(b) => {
                req = req.header("content-type", "application/json");
                Body::from(b)
            }
        };
        let req = req.body(body).unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        let status = resp.status();
        assert_eq!(
            status.as_u16(),
            p.status,
            "{} {} が {} を返した (仕様は {})",
            p.op.method,
            p.op.path,
            status.as_u16(),
            p.status
        );
        let ct = resp
            .headers()
            .get(axum::http::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        assert!(
            ct.contains(p.content_type),
            "{} {} の Content-Type が {:?} (期待: {})",
            p.op.method,
            p.op.path,
            ct,
            p.content_type
        );
        // 仕様書にも同じステータスが書かれていること。
        let documented = documented_statuses(&p.op);
        assert!(
            documented.contains(&p.status),
            "仕様の {} {} に {} が無い (書かれているのは {:?})",
            p.op.method,
            p.op.path,
            p.status,
            documented
        );
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        if p.content_type == "application/json" {
            serde_json::from_slice::<serde_json::Value>(&bytes).unwrap_or_else(|e| {
                panic!("{} {} の本文が JSON でない: {e}", p.op.method, p.op.path)
            });
        }
    }
}

#[test]
fn every_router_route_is_documented() {
    let mut routed = parse_router_source();
    routed.sort();
    routed.dedup();
    let mut documented: Vec<Op> = parse_spec()
        .into_iter()
        .filter(|(_, i)| i == "engine")
        .map(|(o, _)| o)
        .collect();
    documented.sort();
    documented.dedup();
    assert_eq!(
        routed, documented,
        "router() のルートと docs/openapi.yaml の engine operation が一致しない"
    );
}

#[test]
fn every_documented_flash_route_exists_in_the_dispatch() {
    let spec = parse_spec();
    let flash: Vec<Op> = spec
        .iter()
        .filter(|(_, i)| i == "flash-cli")
        .map(|(o, _)| o.clone())
        .collect();
    assert!(!flash.is_empty(), "flash-cli 実装の operation が仕様に無い");

    for op in &flash {
        let pattern = if let Some(prefix) = op.path.split('{').next() {
            if op.path.contains('{') {
                // パスパラメータ付きは starts_with でディスパッチしている。
                format!("req.method == \"{}\" && path.starts_with(\"{prefix}\")", op.method)
            } else {
                format!("req.method == \"{}\" && path == \"{}\"", op.method, op.path)
            }
        } else {
            unreachable!()
        };
        assert!(
            FLASH_SRC.contains(&pattern),
            "serve_flash.rs に `{pattern}` が無い (仕様: {} {})",
            op.method,
            op.path
        );
    }
}

#[test]
fn spec_has_no_unknown_implementation_or_dangling_refs() {
    let spec = parse_spec();
    for (op, imp) in &spec {
        assert!(
            imp == "engine" || imp == "flash-cli",
            "{} {} の x-denno-impl が未知: {imp}",
            op.method,
            op.path
        );
    }
    // $ref の参照先 (components/schemas と components/responses) が存在するか。
    let defined_schemas: Vec<&str> = SPEC
        .lines()
        .filter_map(|l| l.strip_prefix("    ")?.strip_suffix(':'))
        .collect();
    let mut refs = 0usize;
    let needle = "$ref: '#/components/";
    for (i, _) in SPEC.match_indices(needle) {
        let rest = &SPEC[i + needle.len()..];
        let end = rest.find('\'').unwrap();
        let target = &rest[..end];
        let name = target.rsplit('/').next().unwrap();
        assert!(
            defined_schemas.contains(&name),
            "仕様の $ref が存在しない名前を指している: {name}"
        );
        refs += 1;
    }
    assert!(refs > 20, "$ref を {refs} 個しか見つけられていない (パーサの不具合)");
}

#[test]
fn probe_table_covers_expected_engine_routes() {
    // 主要 4 系統が必ず表に入っていること (表からの削除でテストが無言で
    // 弱くならないようにする)。
    let ops: Vec<Op> = probes().iter().map(|p| p.op.clone()).collect();
    for path in [
        "/v1/messages",
        "/v1/messages/count_tokens",
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/models",
        "/api/tags",
        "/api/chat",
        "/api/generate",
        "/health",
        "/ready",
        "/metrics",
    ] {
        assert!(
            ops.iter().any(|o| o.path == path),
            "{path} が probe 表に無い"
        );
    }
    let flash_ops: Vec<String> = parse_spec()
        .into_iter()
        .filter(|(_, i)| i == "flash-cli")
        .map(|(o, _)| o.path)
        .collect();
    for path in [
        "/v1/steer",
        "/v1/steer/record",
        "/v1/steer/record/stop",
        "/v1/steer/snapshots/{name}",
        "/v1/steer/apply",
        "/v1/steer/analyze",
        "/v1/steer/export",
        "/v1/steer/import",
        "/v1/steer/recipe/export",
        "/v1/steer/recipe/import",
    ] {
        assert!(flash_ops.iter().any(|p| p == path), "仕様に {path} が無い");
    }
    let _ = StatusCode::OK;
}
