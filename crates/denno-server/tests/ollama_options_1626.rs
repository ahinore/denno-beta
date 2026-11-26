//! task 1626: Ollama `/api/generate` / `/api/chat` の `options` 監査。
//!
//! 対象: `crates/denno-server/src/ollama.rs` の `OllamaOptions` /
//! `de_ollama_options` / `max_tokens_from` / `build_params` / `normalize_stop`。
//!
//! 固定する契約:
//!
//! - `options` は省略可。`{}` と `null` は未指定と同じ。配列・文字列等は 400。
//! - `temperature` / `top_p` は数値のみ受理。非有限 (JSON の `1e40` は f32 で inf)
//!   は 400 (OpenAI / Anthropic 経路と同じ)。負値・範囲外は Ollama の意味論
//!   (`temperature <= 0` = greedy、`top_p` の無効値 = 無効) を保つため拒否しない。
//! - `num_predict` は正値のみ採用し、上限 4096 で clamp (`resolve_max_tokens`)。
//!   0 と負値 (-1 / -2 = Ollama の無制限 / context 埋め) は未指定扱い。
//!   非整数・i64 超過は 400。
//! - `stop` は文字列 1 本 or 文字列配列。空文字・空配列・null は停止列なし。
//!   文字列以外の要素 / 文字列でも配列でもない値は 400。
//! - `seed` / `top_k` / `repeat_penalty` / `num_ctx` 等は未知キーとして無視する
//!   (docs/api.md: 「その他 未知フィールドは無視」)。
//!
//! 数値の検証は「有限性」だけで、範囲は変えない (docs/api.md に追記)。
//! dummy モデル + dummy トークナイザ、greedy、sleep 無しで決定的。

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{dummy_tokenizer, DummyModel};
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

const MODEL: &str = "denno-test";

fn app_with_cap(temperature_cap: Option<f32>) -> Router {
    let engine = Engine::from_parts(Box::new(DummyModel), dummy_tokenizer().unwrap(), MODEL, None);
    let info = EngineInfo::from_engine(&engine);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
    let state = Arc::new(AppState {
        engine,
        info,
        api_key: None,
        scheduler,
        model_name: MODEL.to_string(),
        model_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
        default_repetition_penalty: None,
        temperature_cap,
    });
    denno_server::router(state)
}

struct Resp {
    status: StatusCode,
    content_type: String,
    body: String,
}

impl Resp {
    fn json(&self) -> Value {
        serde_json::from_str(&self.body)
            .unwrap_or_else(|e| panic!("本文が JSON ではない ({e}): {:?}", self.body))
    }

    /// Ollama 封筒 `{"error": "<string>"}` を検査してメッセージを返す。
    fn error_message(&self) -> String {
        let v = self.json();
        let obj = v.as_object().unwrap_or_else(|| panic!("error 本文が object でない: {v}"));
        assert_eq!(obj.len(), 1, "error 以外のキーが混ざっている: {v}");
        v["error"]
            .as_str()
            .unwrap_or_else(|| panic!("error が文字列でない: {v}"))
            .to_string()
    }
}

async fn post_raw(app: &Router, uri: &str, body: &str) -> Resp {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let content_type = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    Resp { status, content_type, body: String::from_utf8_lossy(&bytes).to_string() }
}

async fn post(app: &Router, uri: &str, body: &Value) -> Resp {
    post_raw(app, uri, &body.to_string()).await
}

fn chat_body(options: Value) -> Value {
    json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "hello"}],
        "stream": false,
        "options": options,
    })
}

fn generate_body(options: Value) -> Value {
    json!({
        "model": MODEL,
        "prompt": "hello",
        "stream": false,
        "options": options,
    })
}

const URIS: [&str; 2] = ["/api/chat", "/api/generate"];

fn body_for(uri: &str, options: Value) -> Value {
    if uri == "/api/chat" {
        chat_body(options)
    } else {
        generate_body(options)
    }
}

/// テキスト・eval_count・prompt_eval_count・done_reason の組。
/// 生成は greedy なので、オプションが出力へ効いたかはこの 4 つで判定できる。
#[derive(Debug, PartialEq, Eq)]
struct Summary {
    text: String,
    eval_count: u64,
    prompt_eval_count: u64,
    done_reason: String,
}

fn summary(uri: &str, v: &Value) -> Summary {
    let text = if uri == "/api/chat" {
        v["message"]["content"].as_str()
    } else {
        v["response"].as_str()
    }
    .unwrap_or_else(|| panic!("{uri} の本文が文字列でない: {v}"))
    .to_string();
    Summary {
        text,
        eval_count: v["eval_count"].as_u64().unwrap_or_else(|| panic!("eval_count 無し: {v}")),
        prompt_eval_count: v["prompt_eval_count"]
            .as_u64()
            .unwrap_or_else(|| panic!("prompt_eval_count 無し: {v}")),
        done_reason: v["done_reason"]
            .as_str()
            .unwrap_or_else(|| panic!("done_reason 無し: {v}"))
            .to_string(),
    }
}

/// greedy の基準値 (`options: {"temperature": 0}`)。
async fn baseline(app: &Router, uri: &str) -> (Resp, Summary) {
    let r = post(app, uri, &body_for(uri, json!({"temperature": 0}))).await;
    assert_eq!(r.status, StatusCode::OK, "{uri} baseline: {}", r.body);
    let s = summary(uri, &r.json());
    (r, s)
}

fn assert_400(r: &Resp, uri: &str, needles: &[&str]) {
    assert_eq!(r.status, StatusCode::BAD_REQUEST, "{uri}: {}", r.body);
    assert!(
        r.content_type.starts_with("application/json"),
        "{uri} の 400 が JSON でない: {}",
        r.content_type
    );
    let msg = r.error_message();
    for n in needles {
        assert!(msg.contains(n), "{uri} の error に {n:?} が無い: {msg}");
    }
}

// ---------------------------------------------------------------------------
// options 自体の形
// ---------------------------------------------------------------------------

#[tokio::test]
async fn options_missing_empty_and_null_are_unspecified() {
    let app = app_with_cap(None);
    for uri in URIS {
        // 未指定の temperature は sampler 既定 (1.0) で **サンプリング** なので、
        // 本文の一致は検査できない。プロンプト・構造だけを比べる。
        let mut variants: Vec<(&str, Value)> = vec![("empty", json!({})), ("null", Value::Null)];
        let mut body = body_for(uri, json!({}));
        body.as_object_mut().unwrap().remove("options");
        variants.push(("missing", body));

        let mut prompt_counts = Vec::new();
        for (name, body) in variants {
            let r = match name {
                "missing" => post(&app, uri, &body).await,
                _ => post(&app, uri, &body_for(uri, body)).await,
            };
            assert_eq!(r.status, StatusCode::OK, "{uri} options {name}: {}", r.body);
            let v = r.json();
            assert_eq!(v["done"], true, "{uri} options {name}: {v}");
            assert_eq!(v["done_reason"], "stop", "{uri} options {name}: {v}");
            assert!(v["eval_count"].as_u64().unwrap_or(0) >= 1, "{uri} options {name}: {v}");
            let s = summary(uri, &v);
            assert!(!s.text.is_empty(), "{uri} options {name}: {s:?}");
            prompt_counts.push((name, s.prompt_eval_count));
        }
        let (_, first) = prompt_counts[0];
        for (name, n) in &prompt_counts[1..] {
            assert_eq!(*n, first, "{uri} options {name} がプロンプトを変えた: {prompt_counts:?}");
        }
    }
}

#[tokio::test]
async fn options_wrong_shape_is_400() {
    let app = app_with_cap(None);
    for uri in URIS {
        for opts in [json!([0.5]), json!("hot"), json!(3), json!(true)] {
            let r = post(&app, uri, &body_for(uri, opts.clone())).await;
            assert_400(&r, uri, &["invalid request body"]);
        }
    }
}

// ---------------------------------------------------------------------------
// temperature / top_p
// ---------------------------------------------------------------------------

#[tokio::test]
async fn temperature_and_top_p_are_accepted_and_greedy_output_is_unchanged() {
    let app = app_with_cap(None);
    for uri in URIS {
        let (_, base) = baseline(&app, uri).await;
        // temperature <= 0 は greedy。top_p は greedy 経路では読まれない。
        let variants = [
            json!({"temperature": 0}),
            json!({"temperature": 0.0}),
            json!({"temperature": -2}),
            json!({"temperature": 0, "top_p": 0}),
            json!({"temperature": 0, "top_p": 0.5}),
            json!({"temperature": 0, "top_p": 1.0}),
            json!({"temperature": 0, "top_p": 1.5}),
            json!({"temperature": 0, "top_p": -1.0}),
            json!({"temperature": 0, "top_p": 2.0}),
        ];
        for opts in variants {
            let r = post(&app, uri, &body_for(uri, opts.clone())).await;
            assert_eq!(r.status, StatusCode::OK, "{uri} options={opts}: {}", r.body);
            assert_eq!(summary(uri, &r.json()), base, "{uri} options={opts} が出力を変えた");
        }
        // temperature > 0 はサンプリング (非決定的) なので受理だけを検査する。
        for opts in [json!({"temperature": 0.5}), json!({"temperature": 1.0, "top_p": 0.9})] {
            let r = post(&app, uri, &body_for(uri, opts.clone())).await;
            assert_eq!(r.status, StatusCode::OK, "{uri} options={opts}: {}", r.body);
            let v = r.json();
            assert_eq!(v["done"], true, "{uri} options={opts}: {v}");
            assert!(v["eval_count"].as_u64().unwrap_or(0) >= 1, "{uri} {opts}: {v}");
        }
    }
}

#[tokio::test]
async fn non_finite_temperature_and_top_p_are_400() {
    let app = app_with_cap(None);
    for uri in URIS {
        for (key, num) in [
            ("temperature", "1e40"),
            ("temperature", "3.5e38"),
            ("top_p", "1e40"),
            ("top_p", "3.5e38"),
        ] {
            let body = if uri == "/api/chat" {
                format!(
                    r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"hello"}}],"stream":false,"options":{{"{key}":{num}}}}}"#
                )
            } else {
                format!(
                    r#"{{"model":"{MODEL}","prompt":"hello","stream":false,"options":{{"{key}":{num}}}}}"#
                )
            };
            let r = post_raw(&app, uri, &body).await;
            assert_400(&r, uri, &[key, "finite"]);
        }
        // 有限の最大付近は受理される (境界の反対側)。
        let r = post(&app, uri, &body_for(uri, json!({"temperature": 1e38}))).await;
        assert_eq!(r.status, StatusCode::OK, "{uri} 1e38: {}", r.body);
    }
}

#[tokio::test]
async fn temperature_and_top_p_type_errors_are_400() {
    let app = app_with_cap(None);
    for uri in URIS {
        for bad in [json!("0.5"), json!(true), json!([0.5]), json!({"v": 1})] {
            for key in ["temperature", "top_p"] {
                let opts = json!({ key: bad.clone() });
                let r = post(&app, uri, &body_for(uri, opts.clone())).await;
                assert_400(&r, uri, &["invalid request body"]);
            }
            // null は未指定扱い (Option の None)。
            for key in ["temperature", "top_p"] {
                let r = post(&app, uri, &body_for(uri, json!({ key: Value::Null }))).await;
                assert_eq!(r.status, StatusCode::OK, "{uri} {key}=null: {}", r.body);
            }
        }
    }
}

#[tokio::test]
async fn temperature_cap_does_not_reject_requests() {
    let capped = app_with_cap(Some(0.5));
    let plain = app_with_cap(None);
    for uri in URIS {
        // cap 付きでも要求値は受理される (実効値だけが 0.5 に丸められる)。
        for t in [0.0, 0.5, 2.0] {
            let r = post(&capped, uri, &body_for(uri, json!({"temperature": t}))).await;
            assert_eq!(r.status, StatusCode::OK, "{uri} cap t={t}: {}", r.body);
        }
        // greedy は cap の有無で変わらない。
        let (_, base) = baseline(&plain, uri).await;
        let r = post(&capped, uri, &body_for(uri, json!({"temperature": 0}))).await;
        assert_eq!(summary(uri, &r.json()), base, "{uri} cap が greedy 出力を変えた");
        // 非有限の拒否は cap があっても同じ。
        let r = post(&capped, uri, &body_for(uri, json!({"temperature": 1e40}))).await;
        assert_400(&r, uri, &["temperature", "finite"]);
    }
}

// ---------------------------------------------------------------------------
// num_predict
// ---------------------------------------------------------------------------

#[tokio::test]
async fn positive_num_predict_is_applied() {
    let app = app_with_cap(None);
    for uri in URIS {
        let (_, base) = baseline(&app, uri).await;
        for n in [1u64, 2, 5] {
            let r = post(&app, uri, &body_for(uri, json!({"temperature": 0, "num_predict": n}))).await;
            assert_eq!(r.status, StatusCode::OK, "{uri} n={n}: {}", r.body);
            let s = summary(uri, &r.json());
            assert_eq!(s.eval_count, n, "{uri} n={n}: {s:?}");
            assert_eq!(s.done_reason, "length", "{uri} n={n}: {s:?}");
            assert_eq!(s.prompt_eval_count, base.prompt_eval_count, "{uri} n={n} が prompt を変えた");
        }
    }
}

#[tokio::test]
async fn zero_and_negative_num_predict_mean_unspecified() {
    let app = app_with_cap(None);
    for uri in URIS {
        let (_, base) = baseline(&app, uri).await;
        // Ollama: -1 = 無制限, -2 = context 埋め。サーバー既定 (1024) へ戻す。
        for n in [0i64, -1, -2, -100] {
            let r = post(&app, uri, &body_for(uri, json!({"temperature": 0, "num_predict": n}))).await;
            assert_eq!(r.status, StatusCode::OK, "{uri} n={n}: {}", r.body);
            assert_eq!(summary(uri, &r.json()), base, "{uri} n={n} が未指定と違う");
        }
    }
}

#[tokio::test]
async fn num_predict_wrong_type_is_400_and_i64_max_is_accepted() {
    let app = app_with_cap(None);
    for uri in URIS {
        for bad in [json!(2.0), json!(1.5), json!("2"), json!(true), json!([2])] {
            let r = post(&app, uri, &body_for(uri, json!({"num_predict": bad.clone()}))).await;
            assert_400(&r, uri, &["invalid request body"]);
        }
        // i64 を 1 超える値も 400。
        let body = if uri == "/api/chat" {
            format!(
                r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"hello"}}],"stream":false,"options":{{"num_predict":9223372036854775808}}}}"#
            )
        } else {
            format!(r#"{{"model":"{MODEL}","prompt":"hello","stream":false,"options":{{"num_predict":9223372036854775808}}}}"#)
        };
        let r = post_raw(&app, uri, &body).await;
        assert_400(&r, uri, &["invalid request body"]);

        // i64::MAX は clamp (4096) されるだけで拒否しない。dummy は EOS で先に
        // 止まるので eval_count では clamp を観測できない → 関数で固定する。
        let (_, base) = baseline(&app, uri).await;
        let r = post(
            &app,
            uri,
            &body_for(uri, json!({"temperature": 0, "num_predict": i64::MAX})),
        )
        .await;
        assert_eq!(r.status, StatusCode::OK, "{uri} i64::MAX: {}", r.body);
        assert_eq!(summary(uri, &r.json()), base, "{uri} i64::MAX が EOS 前の出力を変えた");
    }
    // 上限 clamp の規則そのもの (max_tokens_from → resolve_max_tokens)。
    assert_eq!(denno_server::resolve_max_tokens(Some(5000), None), denno_server::MAX_TOKENS_CLAMP);
    assert_eq!(denno_server::resolve_max_tokens(Some(10), None), 10);
    assert_eq!(
        denno_server::resolve_max_tokens(Some(usize::MAX), None),
        denno_server::MAX_TOKENS_CLAMP
    );
}

// ---------------------------------------------------------------------------
// stop
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stop_string_and_array_truncate_the_same_way() {
    let app = app_with_cap(None);
    for uri in URIS {
        let (_, base) = baseline(&app, uri).await;
        let words: Vec<&str> = base.text.split_whitespace().collect();
        assert!(words.len() >= 3, "{uri} の基準本文が短い: {:?}", base.text);
        let stop = words[2].to_string(); // 3 語目で切る

        let mut results = Vec::new();
        for stop_opts in [json!(stop.clone()), json!([stop.clone()]), json!(["zzz", stop.clone()])] {
            let r = post(
                &app,
                uri,
                &body_for(uri, json!({"temperature": 0, "stop": stop_opts})),
            )
            .await;
            assert_eq!(r.status, StatusCode::OK, "{uri} stop={stop_opts}: {}", r.body);
            let s = summary(uri, &r.json());
            assert!(!s.text.contains(&stop), "{uri} stop={stop_opts} が本文に残る: {s:?}");
            assert_eq!(s.done_reason, "stop", "{uri} stop={stop_opts}: {s:?}");
            assert!(s.text.len() < base.text.len(), "{uri} stop={stop_opts} が切っていない: {s:?}");
            results.push(s.text);
        }
        assert!(
            results.windows(2).all(|w| w[0] == w[1]),
            "{uri} で文字列 / 配列の stop が同じ結果にならない: {results:?}"
        );
    }
}

#[tokio::test]
async fn empty_stop_forms_mean_no_stop() {
    let app = app_with_cap(None);
    for uri in URIS {
        let (_, base) = baseline(&app, uri).await;
        for stop in [json!(""), json!([]), Value::Null, json!(["", ""])] {
            let r = post(
                &app,
                uri,
                &body_for(uri, json!({"temperature": 0, "stop": stop.clone()})),
            )
            .await;
            assert_eq!(r.status, StatusCode::OK, "{uri} stop={stop}: {}", r.body);
            assert_eq!(summary(uri, &r.json()), base, "{uri} stop={stop} が出力を変えた");
        }
    }
}

#[tokio::test]
async fn stop_type_errors_are_400() {
    let app = app_with_cap(None);
    for uri in URIS {
        for bad in [
            json!(5),
            json!(true),
            json!({"a": 1}),
            json!(["a", 1]),
            json!([null]),
            json!([["nested"]]),
        ] {
            let r = post(&app, uri, &body_for(uri, json!({"stop": bad.clone()}))).await;
            assert_400(&r, uri, &["options.stop"]);
        }
    }
}

// ---------------------------------------------------------------------------
// 未知キー (docs/api.md: 無視する)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn unknown_options_are_ignored() {
    let app = app_with_cap(None);
    for uri in URIS {
        let (_, base) = baseline(&app, uri).await;
        let variants = [
            json!({"temperature": 0, "seed": 42}),
            json!({"temperature": 0, "seed": -7}),
            json!({"temperature": 0, "seed": "abc"}),
            json!({"temperature": 0, "seed": 1.5}),
            json!({"temperature": 0, "seed": {"x": 1}}),
            json!({"temperature": 0, "top_k": 0}),
            json!({"temperature": 0, "top_k": 999}),
            json!({"temperature": 0, "repeat_penalty": 1.1}),
            json!({"temperature": 0, "repeat_penalty": "x"}),
            json!({"temperature": 0, "num_ctx": 4096}),
            json!({"temperature": 0, "num_thread": 8}),
            json!({"temperature": 0, "mirostat": 1, "tfs_z": 1.0, "typical_p": 1.0}),
        ];
        for opts in variants {
            let r = post(&app, uri, &body_for(uri, opts.clone())).await;
            assert_eq!(r.status, StatusCode::OK, "{uri} options={opts}: {}", r.body);
            assert_eq!(summary(uri, &r.json()), base, "{uri} options={opts} が出力を変えた");
        }
    }
}

// ---------------------------------------------------------------------------
// ストリーム: 検証は NDJSON を流す前に走る
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stream_true_options_errors_are_json_400_before_ndjson() {
    let app = app_with_cap(None);
    for uri in URIS {
        let mut body = body_for(uri, json!({"temperature": "hot"}));
        body["stream"] = json!(true);
        let r = post(&app, uri, &body).await;
        assert_400(&r, uri, &["invalid request body"]);

        // 正常な options なら従来どおり NDJSON で最終行だけ done=true。
        let mut body = body_for(uri, json!({"temperature": 0, "num_predict": 3}));
        body["stream"] = json!(true);
        let r = post(&app, uri, &body).await;
        assert_eq!(r.status, StatusCode::OK, "{uri} stream: {}", r.body);
        assert!(
            r.content_type.starts_with("application/x-ndjson"),
            "{uri} content-type = {}",
            r.content_type
        );
        let lines: Vec<Value> = r
            .body
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("NDJSON 行が壊れた ({e}): {l:?}")))
            .collect();
        assert_eq!(lines.last().unwrap()["done"], true, "{uri}: {lines:?}");
        assert_eq!(lines.last().unwrap()["eval_count"], 3, "{uri}: {lines:?}");
        assert!(lines[..lines.len() - 1].iter().all(|l| l["done"] == false), "{uri}: {lines:?}");
    }
}
