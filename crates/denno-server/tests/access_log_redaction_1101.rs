//! task 1101: JSON アクセスログのフィールド揃いと秘匿の監査。
//!
//! - `access_log_line` が `request_id` / `method` / `path` / `status` /
//!   `latency_ms` を必ず含む (1 行 JSON のまま)
//! - クエリ文字列の秘匿対象パラメータ (`api_key` / `key` / `token` /
//!   `secret` / `authorization`) の値がログに出ない
//! - `Authorization` / `x-api-key` ヘッダの値がログに出ない (ミドルウェア経路)
//! - 秘匿対象を含まないパスは加工されない

use std::io::{self, Write};
use std::sync::{Arc, Mutex, MutexGuard};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{dummy_tokenizer, DummyModel};
use denno_engine::Engine;
use denno_server::{
    access_log_line, redact_path_and_query, redact_query, AppState, EngineInfo, Scheduler,
};
use http_body_util::BodyExt;
use serde_json::Value;
use tower::ServiceExt;

// ---------- 共通ヘルパ ----------

/// `set_access_log_json` はプロセス全体の static。並列実行で互いに
/// ゲートを閉じてしまわないよう、切り替えるテストはこのロックを取る。
static JSON_GATE: Mutex<()> = Mutex::new(());

fn json_gate() -> MutexGuard<'static, ()> {
    JSON_GATE.lock().unwrap_or_else(|e| e.into_inner())
}

#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<u8>>>);

impl Capture {
    fn text(&self) -> String {
        String::from_utf8_lossy(&self.0.lock().unwrap()).to_string()
    }
}

impl Write for Capture {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn app() -> Router {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(DummyModel), tok, "denno-test", None);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
    let info = EngineInfo::from_engine(&engine.lock().unwrap());
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

/// キャプチャ済みアクセスログから最初の JSON オブジェクトを取り出す。
fn extract_json(log: &str) -> Value {
    let start = log.find('{').unwrap_or_else(|| panic!("JSON が無い: {log}"));
    let end = log.rfind('}').unwrap_or_else(|| panic!("JSON が閉じていない: {log}"));
    serde_json::from_str(&log[start..=end]).unwrap_or_else(|e| panic!("JSON として読めない: {e}\n{log}"))
}

// ---------- 1. JSON 行のスキーマ ----------

/// 1 リクエスト = 1 行の JSON に、突き合わせに要る 5 フィールドが揃うこと。
#[test]
fn json_access_log_line_has_required_schema_fields() {
    let line = access_log_line("POST", "/v1/messages", 200, 3.25, "req-1101-schema");
    assert!(!line.contains('\n'), "1 行 JSON が複数行になった: {line:?}");
    let v: Value = serde_json::from_str(&line).expect("JSON として読めない");

    assert_eq!(v["request_id"], "req-1101-schema");
    assert_eq!(v["method"], "POST");
    assert_eq!(v["path"], "/v1/messages");
    assert_eq!(v["status"], 200);
    assert!(
        (v["latency_ms"].as_f64().unwrap() - 3.25).abs() < 1e-9,
        "latency_ms が入っていない: {v}"
    );
    // 既存収集系との互換フィールドは同値
    assert!((v["duration_ms"].as_f64().unwrap() - 3.25).abs() < 1e-9);
    assert!(v["ts_ms"].as_u64().unwrap() > 0);
}

/// 秘匿対象パラメータの値を伏せ、名前・順序・非機密パラメータは保つ。
#[test]
fn sensitive_query_values_are_redacted() {
    let path = "/v1/messages?api_key=sk-live-AAAA1111&limit=5&token=tok-BBBB2222\
                &secret=sec-CCCC3333&authorization=Bearer-DDDD4444&key=key-EEEE5555&mode=fast";
    let line = access_log_line("POST", path, 200, 1.0, "req-1101-redact");
    for leaked in [
        "sk-live-AAAA1111",
        "tok-BBBB2222",
        "sec-CCCC3333",
        "Bearer-DDDD4444",
        "key-EEEE5555",
    ] {
        assert!(!line.contains(leaked), "秘匿対象の値 {leaked} がログに漏れた: {line}");
    }
    let v: Value = serde_json::from_str(&line).unwrap();
    let logged = v["path"].as_str().unwrap();
    assert_eq!(
        logged,
        "/v1/messages?api_key=REDACTED&limit=5&token=REDACTED&secret=REDACTED\
         &authorization=REDACTED&key=REDACTED&mode=fast",
        "秘匿の形が変わった: {logged}"
    );
    // 非機密パラメータは残る
    assert!(logged.contains("limit=5"));
    assert!(logged.contains("mode=fast"));
}

/// パラメータ名の大文字小文字や別綴りでも伏せる。値が無い形はそのまま。
#[test]
fn redaction_is_case_insensitive_and_idempotent() {
    assert_eq!(redact_query("API_KEY=abc&TOKEN=def&Key=ghi"), "API_KEY=REDACTED&TOKEN=REDACTED&Key=REDACTED");
    assert_eq!(redact_query("apikey=abc&access_token=def&password=ghi"), "apikey=REDACTED&access_token=REDACTED&password=REDACTED");
    assert_eq!(redact_query("flag&x=1"), "flag&x=1");
    // 2 回かけても結果が変わらない (ミドルウェアと access_log_line の二重適用)
    let once = redact_path_and_query("/v1/x?token=abc&limit=1");
    let twice = redact_path_and_query(&once);
    assert_eq!(once, twice);
    assert_eq!(once, "/v1/x?token=REDACTED&limit=1");
}

// ---------- 2. ミドルウェア経路 (ヘッダとクエリ) ----------

/// 実際のルータ + `log_requests` を通し、`Authorization` / `x-api-key`
/// ヘッダ値とクエリの秘匿値がログ行に現れないこと。
#[test]
fn middleware_access_log_never_carries_secret_headers_or_query_values() {
    let _gate = json_gate();
    const HDR_AUTH: &str = "Bearer sk-live-HDR-9999";
    const HDR_KEY: &str = "sk-live-XKEY-8888";
    const Q_TOKEN: &str = "tok-QUERY-7777";

    let cap = Capture::default();
    let sink = cap.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(move || sink.clone())
        .finish();

    denno_server::set_access_log_json(true);
    let uri = format!("/v1/models?token={Q_TOKEN}&limit=3");
    let outcome = tracing::subscriber::with_default(subscriber, || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let req = Request::builder()
                .method("GET")
                .uri(uri)
                .header("authorization", HDR_AUTH)
                .header("x-api-key", HDR_KEY)
                .body(Body::empty())
                .unwrap();
            let resp = app().oneshot(req).await.unwrap();
            let status = resp.status();
            let body = resp.into_body().collect().await.unwrap().to_bytes();
            (status, body)
        })
    });
    denno_server::set_access_log_json(false);

    let (status, _body) = outcome;
    assert_eq!(status, StatusCode::OK, "ルータ経由のリクエストが失敗した");

    let log = cap.text();
    for leaked in [HDR_AUTH, HDR_KEY, Q_TOKEN, "sk-live-", "tok-QUERY"] {
        assert!(!log.contains(leaked), "秘密 {leaked} がアクセスログに漏れた:\n{log}");
    }
    let v = extract_json(&log);
    assert_eq!(v["method"], "GET");
    assert_eq!(v["status"], 200);
    assert!(v["request_id"].as_str().unwrap().starts_with("req-"));
    assert!(v["latency_ms"].as_f64().unwrap() >= 0.0);
    let path = v["path"].as_str().unwrap();
    assert!(path.starts_with("/v1/models?"), "クエリがログに残っていない: {path}");
    assert!(path.contains("token=REDACTED"), "token が伏せられていない: {path}");
    assert!(path.contains("limit=3"), "非機密パラメータが消えた: {path}");
}

// ---------- 3. 非機密パスは無加工 ----------

/// 秘匿対象を含まないパスは 1 バイトも変わらない。
#[test]
fn benign_paths_are_left_unaltered() {
    for path in [
        "/v1/messages",
        "/v1/models",
        "/v1/models?limit=10&offset=0",
        "/health",
        "/v1/messages?beta=true",
        "/metrics?format=prometheus",
    ] {
        assert_eq!(redact_path_and_query(path), path, "非機密パスが変わった: {path}");
        let line = access_log_line("GET", path, 200, 0.5, "req-1101-benign");
        let v: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["path"].as_str().unwrap(), path, "access_log_line がパスを変えた: {path}");
    }
}

/// 名前が秘匿対象の部分文字列であるだけの非機密パラメータは残す
/// (過剰な秘匿でパスを壊さない)。
#[test]
fn names_merely_containing_sensitive_words_are_kept() {
    for q in ["monkey=1", "keyword=x", "tokens=5", "secrets_count=2"] {
        assert_eq!(redact_query(q), q, "過剰に秘匿した: {q}");
    }
}
