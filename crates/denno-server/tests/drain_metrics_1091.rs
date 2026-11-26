//! task 1091: `/metrics` のライフサイクルゲージ
//! (`denno_server_ready` / `denno_server_draining`) の契約テスト。
//!
//! 3 状態を実プロセス内で順に作り、スクレイプごとに次を固定する:
//!
//!   1. loading  — モデル未ロード: ready 0 / draining 0、`/ready` は 503 "loading"
//!   2. ready    — ロード完了:      ready 1 / draining 0、`/ready` は 200 "ready"
//!   3. draining — 停止要求後:      ready 0 / draining 1、`/ready` は 503 "draining"
//!
//! 加えて、各状態の本文全体を text exposition format 0.0.4 として検証する
//! (HELP / TYPE がサンプルより前、系列の重複なし、値は有限)。
//!
//! 計器はプロセス全体で共有されるが、ここで読むのはライフサイクル状態
//! (スケジューラが持つ atomic) だけで、書き込みはこのテストの state に閉じる。

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{dummy_tokenizer, DummyModel};
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::Value;
use tower::ServiceExt;

// ---------------------------------------------------------------------------
// アプリ生成
// ---------------------------------------------------------------------------

/// モデル未ロード状態で起動した state (loading から始める)。
fn state_not_ready() -> Arc<AppState> {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(DummyModel), tok, "denno-test", None);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn_not_ready(engine.clone(), 8);
    let mut info = EngineInfo::from_engine(&engine.lock().unwrap());
    info.model_dir = "qwen-test-dir".to_string();
    Arc::new(AppState {
        engine,
        info,
        api_key: None,
        scheduler,
        model_name: "denno-test".to_string(),
        model_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
        default_repetition_penalty: None,
        temperature_cap: None,
    })
}

async fn get(app: &Router, uri: &str) -> (StatusCode, String) {
    let req = Request::builder()
        .method("GET")
        .uri(uri)
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, String::from_utf8_lossy(&bytes).to_string())
}

async fn scrape(app: &Router) -> String {
    let (status, body) = get(app, "/metrics").await;
    assert_eq!(status, StatusCode::OK, "/metrics が 200 でない");
    body
}

// ---------------------------------------------------------------------------
// exposition format 0.0.4 の検証器 (このタスクの 2 系列 + 全体の文法)
// ---------------------------------------------------------------------------

/// メトリクス名: `[a-zA-Z_:][a-zA-Z0-9_:]*`
fn valid_metric_name(name: &str) -> bool {
    let mut it = name.chars();
    match it.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' || c == ':' => {}
        _ => return false,
    }
    it.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == ':')
}

/// `name{labels} value` / `name value` を 1 行読み `(系列キー, ファミリ名, 値)`。
fn parse_sample(line: &str) -> Result<(String, String, f64), String> {
    let (series, value_text) = line
        .split_once(' ')
        .ok_or_else(|| "名前と値の区切りが無い".to_string())?;
    if value_text.is_empty() || value_text.contains(' ') {
        return Err(format!("値の書式が不正: {value_text:?}"));
    }
    let value: f64 = value_text
        .parse()
        .map_err(|_| format!("値が数値でない: {value_text:?}"))?;
    if !value.is_finite() {
        return Err(format!("値が有限でない: {value_text:?}"));
    }

    let (name, label_text) = match series.find('{') {
        Some(open) => {
            if !series.ends_with('}') {
                return Err(format!("ラベル集合が閉じていない: {series}"));
            }
            let inner = &series[open + 1..series.len() - 1];
            if inner.contains('\n') || inner.contains('\r') {
                return Err(format!("ラベル値に生の改行: {series}"));
            }
            if inner.ends_with(',') {
                return Err(format!("ラベルの末尾にカンマ: {series}"));
            }
            (&series[..open], inner)
        }
        None => {
            if series.contains('}') {
                return Err(format!("対応しない }}: {series}"));
            }
            (series, "")
        }
    };
    if !valid_metric_name(name) {
        return Err(format!("メトリクス名が不正: {name:?}"));
    }
    Ok((
        format!("{name}{{{label_text}}}"),
        name.to_string(),
        value,
    ))
}

/// 本文全体を format 0.0.4 として検証し、系列キー -> 値を返す。
fn validate_exposition(body: &str) -> BTreeMap<String, f64> {
    assert!(body.ends_with('\n'), "本文が改行で終わっていない");
    assert!(!body.ends_with("\n\n"), "本文が二重改行で終わっている");
    assert!(!body.contains('\r'), "CR が混ざっている");

    let mut helps: BTreeSet<String> = BTreeSet::new();
    let mut types: BTreeMap<String, String> = BTreeMap::new();
    let mut samples: BTreeMap<String, f64> = BTreeMap::new();

    for (idx, line) in body.lines().enumerate() {
        assert!(!line.is_empty(), "{idx} 行目が空行");
        assert_eq!(line.trim_end(), line, "{idx} 行目に末尾空白: {line:?}");
        if let Some(rest) = line.strip_prefix("# HELP ") {
            let (name, help) = rest
                .split_once(' ')
                .unwrap_or_else(|| panic!("{idx} 行目: HELP に説明文が無い: {line}"));
            assert!(valid_metric_name(name), "{idx} 行目: HELP の名前が不正");
            assert!(!help.is_empty(), "{idx} 行目: HELP の説明が空");
            assert!(helps.insert(name.to_string()), "{idx} 行目: HELP {name} が重複");
            continue;
        }
        if let Some(rest) = line.strip_prefix("# TYPE ") {
            let (name, ty) = rest
                .split_once(' ')
                .unwrap_or_else(|| panic!("{idx} 行目: TYPE に型が無い: {line}"));
            assert!(valid_metric_name(name), "{idx} 行目: TYPE の名前が不正");
            assert!(
                matches!(ty, "counter" | "gauge" | "histogram" | "summary" | "untyped"),
                "{idx} 行目: 未知の型 {ty}"
            );
            assert!(
                types.insert(name.to_string(), ty.to_string()).is_none(),
                "{idx} 行目: TYPE {name} が重複"
            );
            assert!(
                helps.contains(name),
                "{idx} 行目: TYPE {name} より HELP が後ろ / 欠落"
            );
            continue;
        }
        assert!(!line.starts_with('#'), "{idx} 行目: 未知のコメント行: {line}");

        let (key, name, value) =
            parse_sample(line).unwrap_or_else(|e| panic!("{idx} 行目: サンプルが不正 ({e}): {line}"));
        // histogram の `_bucket` / `_sum` / `_count` はファミリ名へ寄せる。
        let family = ["_bucket", "_sum", "_count"]
            .iter()
            .find_map(|suffix| {
                name.strip_suffix(suffix)
                    .filter(|base| types.get(*base).map(String::as_str) == Some("histogram"))
            })
            .unwrap_or(name.as_str())
            .to_string();
        assert!(
            types.contains_key(&family),
            "{idx} 行目: {family} の TYPE が無い: {line}"
        );
        assert!(
            helps.contains(&family),
            "{idx} 行目: {family} の HELP が無い: {line}"
        );
        assert!(
            samples.insert(key.clone(), value).is_none(),
            "{idx} 行目: 系列が重複: {key}"
        );
    }
    samples
}

fn gauge(samples: &BTreeMap<String, f64>, name: &str) -> f64 {
    *samples
        .get(&format!("{name}{{}}"))
        .unwrap_or_else(|| panic!("{name} の系列が無い"))
}

/// `/ready` の本文から `status` を取り出す。
fn ready_status(body: &str) -> String {
    let v: Value = serde_json::from_str(body).expect("/ready の本文が JSON でない");
    v.get("status")
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("/ready に status が無い: {body}"))
        .to_string()
}

// ---------------------------------------------------------------------------
// 1-3. loading -> ready -> draining の遷移
// ---------------------------------------------------------------------------

#[tokio::test]
async fn gauges_track_loading_ready_and_draining_transitions() {
    let state = state_not_ready();
    let app = denno_server::router(state.clone());
    let lc = state.scheduler.lifecycle();

    // 状態ごとに (ready, draining, /ready の HTTP コード, /ready の status)。
    let mut seen: Vec<(f64, f64)> = Vec::new();

    // --- 1. loading ---
    assert!(!lc.is_ready() && !lc.is_draining());
    let (code, body) = get(&app, "/ready").await;
    assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE, "loading で /ready が 503 でない");
    assert_eq!(ready_status(&body), "loading");

    let text = scrape(&app).await;
    let s = validate_exposition(&text);
    assert_eq!(gauge(&s, "denno_server_ready"), 0.0, "loading: ready が 0 でない");
    assert_eq!(gauge(&s, "denno_server_draining"), 0.0, "loading: draining が 0 でない");
    assert!(text.contains("# TYPE denno_server_ready gauge"));
    assert!(text.contains("# TYPE denno_server_draining gauge"));
    assert!(text.lines().any(|l| l.starts_with("# HELP denno_server_ready ")));
    assert!(text.lines().any(|l| l.starts_with("# HELP denno_server_draining ")));
    seen.push((0.0, 0.0));

    // --- 2. ready ---
    lc.set_ready(true);
    let (code, body) = get(&app, "/ready").await;
    assert_eq!(code, StatusCode::OK, "ready で /ready が 200 でない");
    assert_eq!(ready_status(&body), "ready");

    let s = validate_exposition(&scrape(&app).await);
    assert_eq!(gauge(&s, "denno_server_ready"), 1.0, "ready: ready が 1 でない");
    assert_eq!(gauge(&s, "denno_server_draining"), 0.0, "ready: draining が 0 でない");
    seen.push((1.0, 0.0));

    // --- 3. draining ---
    lc.begin_drain();
    let (code, body) = get(&app, "/ready").await;
    assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE, "draining で /ready が 503 でない");
    assert_eq!(ready_status(&body), "draining");

    let s = validate_exposition(&scrape(&app).await);
    assert_eq!(gauge(&s, "denno_server_ready"), 0.0, "draining: ready が 0 でない");
    assert_eq!(gauge(&s, "denno_server_draining"), 1.0, "draining: draining が 1 でない");
    seen.push((0.0, 1.0));

    // 3 状態すべてを踏んだ証拠 (loading -> ready -> draining)。
    assert_eq!(seen, vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]);
}

// ---------------------------------------------------------------------------
// 4. exposition の妥当性 (既存系列と同居した本文全体)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn lifecycle_gauges_coexist_with_existing_families_validly() {
    let state = state_not_ready();
    let app = denno_server::router(state.clone());

    for phase in ["loading", "ready", "draining"] {
        match phase {
            "ready" => state.scheduler.lifecycle().set_ready(true),
            "draining" => state.scheduler.lifecycle().begin_drain(),
            _ => {}
        }
        let body = scrape(&app).await;
        let samples = validate_exposition(&body);

        // 既存ファミリの骨格が壊れていない。
        for (name, ty) in [
            ("denno_model_info", "gauge"),
            ("denno_requests_in_flight", "gauge"),
            ("denno_requests_queued", "gauge"),
            ("denno_server_ready", "gauge"),
            ("denno_server_draining", "gauge"),
            ("denno_ttft_seconds", "histogram"),
        ] {
            assert!(
                body.contains(&format!("# TYPE {name} {ty}")),
                "{phase}: {name} の TYPE {ty} が無い"
            );
            assert!(
                body.lines().any(|l| l.starts_with(&format!("# HELP {name} "))),
                "{phase}: {name} の HELP が無い"
            );
        }

        // ゲージは 0 / 1 のどちらか (2 値契約)。
        for name in ["denno_server_ready", "denno_server_draining"] {
            let v = gauge(&samples, name);
            assert!(v == 0.0 || v == 1.0, "{phase}: {name} = {v} (0/1 でない)");
        }
    }
}
