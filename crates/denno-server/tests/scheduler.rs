//! P4: サーバー層スケジューラ (キューイング + 直列実行 + 同時受け付け) の統合テスト。
//!
//! denno-engine の test-utils (ダミーモデル/ダミートークナイザ) を使い、
//! モデルDLなしで HTTP ワイヤレベル (axum Router + tower oneshot) を検証する。
//!
//! カバー項目:
//! - 単発リクエストの exactness (エンジン直接実行と完全一致)
//! - 同時受け付け (複数リクエストが全て受理されて逐次実行される)
//! - 同時実行上限 = 1 (実行中ジョブが 2 以上にならない)
//! - キュー満杯で 503 (overloaded_error)
//! - SSE ストリーミングがキュー経由でも壊れない

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use denno_engine::kv_cache::KvCache;
use denno_engine::model::Model;
use denno_engine::testutil::{
    dummy_tokenizer, kv_sensitive_engine, KvSensitiveModel, DUMMY_CONTEXT, DUMMY_EOS, DUMMY_VOCAB,
};
use denno_engine::{ChatMessage, Engine, GenParams};
use denno_server::{MAX_ACTIVE_SESSIONS, THINKING_BRIEF_INSTRUCTION, AppState, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

/// forward のたに sleep するダミーモデル (DummyModel と同じ決定論的 logits)。
/// 実行時間をコントロールしてキューの満杯/直列実行を観測できるようにする。
struct SleepModel {
    delay: Duration,
}

impl Model for SleepModel {
    fn context(&self) -> usize {
        DUMMY_CONTEXT
    }

    fn eos_ids(&self) -> &[u32] {
        &[DUMMY_EOS]
    }

    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> anyhow::Result<Vec<f32>> {
        std::thread::sleep(self.delay);
        // P17e: 需要ベース確保 (書き込み前に容量確保) のダミー側契約
        let _ = cache.kv_bufs(0, tokens.len());
        cache.advance(tokens.len());
        let last = tokens.last().copied().unwrap_or(0) as usize;
        let mut logits = vec![0.0f32; DUMMY_VOCAB];
        logits[(last + 1) % DUMMY_VOCAB] = 5.0;
        logits[(last + 2) % DUMMY_VOCAB] = 1.0;
        Ok(logits)
    }

    fn new_cache(&self) -> KvCache {
        KvCache::new(1, 4, DUMMY_CONTEXT)
    }
}

/// テスト用 AppState + Router を構築する
fn test_app_with(model: Box<dyn Model>, max_queue: usize) -> (Router, Arc<AppState>) {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(model, tok, "denno-test", None);
    let info = EngineInfo::from_engine(&engine);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), max_queue);
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
    let app = denno_server::router(state.clone());
    (app, state)
}

fn test_app(delay_ms: u64, max_queue: usize) -> (Router, Arc<AppState>) {
    test_app_with(
        Box::new(SleepModel { delay: Duration::from_millis(delay_ms) }),
        max_queue,
    )
}

fn request_body(content: &str, max_tokens: usize, stream: bool) -> String {
    json!({
        "model": "denno-test",
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
        "messages": [{"role": "user", "content": content}]
    })
    .to_string()
}

async fn post(app: &Router, body: String) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, v)
}

fn greedy_params(max_tokens: usize) -> GenParams {
    GenParams { max_tokens, temperature: Some(0.0), ..Default::default() }
}

fn user_msg(text: &str) -> Vec<ChatMessage> {
    vec![ChatMessage { role: "user".into(), content: text.into() }]
}

/// SSE ボディから content_block_delta の text を結合して生成テキストを復元する
fn sse_text(body: &str) -> String {
    let mut out = String::new();
    for line in body.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            if let Ok(v) = serde_json::from_str::<Value>(data) {
                if v["type"] == "content_block_delta" {
                    if let Some(t) = v["delta"]["text"].as_str() {
                        out.push_str(t);
                    }
                }
            }
        }
    }
    out
}

/// ダミーモデルの greedy 生成の期待値 (engine.rs のテストと同一)
const EXPECTED_GREEDY_TEXT: &str = "t5 t6 t7 t8 t9 t10 t11";

// ---------- 単発 exactness ----------

/// 単発リクエストがエンジン直接実行と完全一致すること (P4 で挙動が変わらないこと)
#[tokio::test]
async fn single_request_matches_engine_reference_output() {
    // 基準: スケジューラ無しでエンジンを直接実行
    let tok = dummy_tokenizer().unwrap();
    let mut direct = Engine::from_parts(
        Box::new(SleepModel { delay: Duration::ZERO }),
        tok,
        "denno-test",
        None,
    );
    // s47: 未指定の thinking は brief (既定) → wire 層で短答誘導文が system に注入
    // される。基準も同一の実効 system で直接実行する (等価性検証の趣旨は不変)。
    let reference = direct
        .chat(
            Some(THINKING_BRIEF_INSTRUCTION),
            &user_msg("t0"),
            &greedy_params(20),
            |_| {},
        )
        .unwrap();

    let (app, _state) = test_app(0, 8);
    let (status, body) = post(&app, request_body("t0", 20, false)).await;

    assert_eq!(status, StatusCode::OK, "body = {body}");
    assert_eq!(body["type"], "message");
    assert_eq!(body["role"], "assistant");
    assert_eq!(body["model"], "denno-test");
    assert_eq!(body["content"][0]["type"], "text");
    assert_eq!(body["content"][0]["text"], EXPECTED_GREEDY_TEXT);
    assert_eq!(body["content"][0]["text"], reference.text, "エンジン直接実行と不一致");
    assert_eq!(body["stop_reason"], reference.stop_reason.as_str());
    assert_eq!(body["usage"]["input_tokens"], json!(reference.input_tokens));
    assert_eq!(body["usage"]["output_tokens"], json!(reference.output_tokens));
}

// ---------- 同時受け付け + インターリーブ実行 ----------

/// 複数リクエストが同時に受け付けられ、全て完了する。実行はステップ単位で
/// インターリーブされるため in_flight (進行中) は 2 以上になり得るが、
/// 出力は全リクエストで同一 (exactness)。
#[tokio::test]
async fn concurrent_requests_are_all_accepted_and_interleaved() {
    let (app, state) = test_app(10, 16); // forward 10ms × 9 回 ≒ 90ms/リクエスト

    // 進行中ジョブ数の最大値を監視
    let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let max_in_flight = Arc::new(AtomicUsize::new(0));
    let monitor = {
        let done = done.clone();
        let max_in_flight = max_in_flight.clone();
        let state = state.clone();
        tokio::spawn(async move {
            while !done.load(Ordering::SeqCst) {
                max_in_flight.fetch_max(state.scheduler.in_flight(), Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
    };

    // 4 リクエストを同時送信
    let mut handles = Vec::new();
    for _ in 0..4 {
        let app = app.clone();
        let body = request_body("t0", 20, false);
        handles.push(tokio::spawn(async move { post(&app, body).await }));
    }
    let mut results = Vec::new();
    for h in handles {
        results.push(h.await.unwrap());
    }
    done.store(true, Ordering::SeqCst);
    let _ = monitor.await;

    // 全て 200 で、出力は完全一致
    for (i, (status, body)) in results.iter().enumerate() {
        assert_eq!(*status, StatusCode::OK, "request {i}: {body}");
        assert_eq!(body["content"][0]["text"], EXPECTED_GREEDY_TEXT);
    }

    // インターリーブされていること (decode 中に別リクエストの prefill/decode が
    // 割り込むため進行中が 2 以上になる)。上限は MAX_ACTIVE_SESSIONS。
    let max = max_in_flight.load(Ordering::SeqCst);
    assert!(
        max >= 2,
        "進行中ジョブが 2 以上にならない (インターリーブが効いていない): max={max}"
    );
    assert!(
        max <= MAX_ACTIVE_SESSIONS,
        "進行中ジョブが上限を超えた: max={max}"
    );
    // 全完了後は進行中 0
    assert_eq!(state.scheduler.in_flight(), 0);
}

/// 異なるプロンプトの並列 2 リクエストが、KV 依存モデル (KvSensitiveModel) で
/// 各々の単独実行と完全一致すること (P4 最重要: インターリーブ下の exactness)。
/// DummyModel は KV を参照しないため、KV 切替えの破綻を検出できるのはこのモデル。
#[tokio::test]
async fn concurrent_different_prompts_match_solo_reference() {
    // 単独実行の基準 (各プロンプトを独立エンジンで)
    // 順序注意: リクエスト 1 はトークンを多く生成する "t15"。
    // "t0" はダミーモデルで即 EOS になるため先頭だとオーバーラップ窓が
    // 1 forward 分しかなく、環境によって並行進行が観測できなくなる。
    let refs: Vec<String> = ["t15", "t0"]
        .iter()
        .map(|p| {
            let mut e = kv_sensitive_engine("ref", None).unwrap();
            // トークン数を多めに取り、リクエスト 2 の HTTP セットアップ遅延が
            // あっても必ずリクエスト 1 の decode 中に到着するようにする
            // (s47: 未指定 = brief の実効 system 付き = wire リクエストと同一条件)
            e.chat(
                Some(THINKING_BRIEF_INSTRUCTION),
                &user_msg(p),
                &greedy_params(24),
                |_| {},
            )
            .unwrap()
            .text
        })
        .collect();
    assert_ne!(refs[0], refs[1], "テストの前提: プロンプトが異なれば出力も異なること");

    // forward に僅かな遅延を入れて、リクエスト2の到着中にリクエスト1が
    // decode 中になる (インターリーブが実際に起きる) 状況を作る
    let (app, state) = test_app_with(
        // CI ランナーは HTTP セットアップが遅いことがあるため、リクエスト 2 が
        // 確実にリクエスト 1 の decode 中に到着するよう十分な遅延を入れる
        Box::new(KvSensitiveModel { delay: Duration::from_millis(20) }),
        8,
    );

    let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let max_in_flight = Arc::new(AtomicUsize::new(0));
    let monitor = {
        let done = done.clone();
        let max_in_flight = max_in_flight.clone();
        let state = state.clone();
        tokio::spawn(async move {
            while !done.load(Ordering::SeqCst) {
                max_in_flight.fetch_max(state.scheduler.in_flight(), Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
    };

    let mut handles = Vec::new();
    for (i, p) in ["t15", "t0"].iter().enumerate() {
        if i == 1 {
            // リクエスト 1 が実際に進行中になってから送る (タイミング依存の排除)
            let state2 = state.clone();
            tokio::time::timeout(Duration::from_secs(5), async {
                while state2.scheduler.in_flight() == 0 {
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
            })
            .await
            .expect("リクエスト 1 が in-flight にならない");
        }
        let app = app.clone();
        let body = request_body(p, 24, false);
        handles.push(tokio::spawn(async move { post(&app, body).await }));
    }
    let mut results = Vec::new();
    for h in handles {
        results.push(h.await.unwrap());
    }
    done.store(true, Ordering::SeqCst);
    let _ = monitor.await;

    for (i, ((status, body), r)) in results.iter().zip(refs.iter()).enumerate() {
        assert_eq!(*status, StatusCode::OK, "request {i}: {body}");
        assert_eq!(
            body["content"][0]["text"].as_str().unwrap(),
            r.as_str(),
            "request {i} の出力が単独実行と不一致 (インターリーブで KV 切替えが壊れた)"
        );
    }
    assert!(
        max_in_flight.load(Ordering::SeqCst) >= 2,
        "2 リクエストが並行進行していない (テストがインターリーブを検証できていない)"
    );
    assert_eq!(state.scheduler.in_flight(), 0);
}

/// キューが溢れたら 503 (overloaded_error) を返す。
/// 実行中 1 + キュー (max_queue) が満杯の間に来たリクエストは即座に 503。
#[tokio::test]
async fn queue_overflow_returns_503_overloaded() {
    let (app, state) = test_app(30, 2); // forward 30ms × 8 回 = 240ms/リクエスト
    let mut handles = Vec::new();
    for _ in 0..5 {
        let app = app.clone();
        let body = request_body("t0", 20, false);
        handles.push(tokio::spawn(async move { post(&app, body).await }));
    }
    let mut ok = 0;
    let mut overloaded = 0;
    for h in handles {
        let (status, body) = h.await.unwrap();
        match status {
            StatusCode::OK => {
                ok += 1;
                assert_eq!(body["content"][0]["text"], EXPECTED_GREEDY_TEXT);
            }
            StatusCode::SERVICE_UNAVAILABLE => {
                overloaded += 1;
                assert_eq!(body["type"], "error", "body = {body}");
                assert_eq!(body["error"]["type"], "overloaded_error");
                assert!(body["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains("queue is full"));
            }
            other => panic!("unexpected status {other}: {body}"),
        }
    }
    assert_eq!(ok + overloaded, 5);
    // インターリーブ設計では、ワーカーがステップ間でジョブを pull して
    // アクティブセッション (live + suspend 済み) にするとチャネルスロットが
    // 空くため、受理数はキュー容量ではなく MAX_ACTIVE_SESSIONS まで増え得る。
    // 残りは 503 で即座に拒否される。
    assert!(overloaded >= 1, "503 が 1 件も無い (キュー上限が効いていない)");
    assert!(
        ok >= 2 && ok <= MAX_ACTIVE_SESSIONS,
        "受理数が想定範囲外: ok={ok} overloaded={overloaded}"
    );
    // 満杯で拒否された後もサーバーは継続して応答できる
    let (status, body) = post(&app, request_body("t0", 20, false)).await;
    assert_eq!(status, StatusCode::OK, "body = {body}");
    assert_eq!(body["content"][0]["text"], EXPECTED_GREEDY_TEXT);
    assert_eq!(state.scheduler.in_flight(), 0);
}

// ---------- スケジューラ単体 ----------

/// Scheduler のキュー挙動を直接検証: 実行中に submit した分は 1 件だけ
/// 受理され (キュー上限 1)、それ以降は QueueFull。
#[tokio::test]
async fn scheduler_rejects_beyond_queue_capacity() {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(
        Box::new(SleepModel { delay: Duration::from_millis(50) }),
        tok,
        "denno-test",
        None,
    );
    let engine = Arc::new(Mutex::new(engine));
    let sched = Scheduler::spawn(engine.clone(), 1);

    // 1 件目: 実行開始を待つ
    let rx1 = sched.submit_chat(None, user_msg("t0"), greedy_params(20), None, None, None).unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while sched.in_flight() == 0 {
        assert!(std::time::Instant::now() < deadline, "job never started");
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    assert_eq!(sched.in_flight(), 1);

    // 2 件目: キュー (上限 1) に入る
    let rx2 = sched
        .submit_chat(None, user_msg("t0"), greedy_params(20), None, None, None)
        .expect("2件目はキューに入るべき");
    // 3 件目: 満杯 → QueueFull
    let err = sched
        .submit_chat(None, user_msg("t0"), greedy_params(20), None, None, None)
        .expect_err("3件目は QueueFull であるべき");
    assert!(matches!(err, denno_server::QueueFull));

    let out1 = rx1.await.unwrap().unwrap();
    let out2 = rx2.await.unwrap().unwrap();
    assert_eq!(out1.text, EXPECTED_GREEDY_TEXT);
    assert_eq!(out2.text, EXPECTED_GREEDY_TEXT);
    // 完了後は実行中 0 に戻り、再度受理できる
    assert_eq!(sched.in_flight(), 0);
    let rx3 = sched.submit_chat(None, user_msg("t0"), greedy_params(20), None, None, None).unwrap();
    assert_eq!(rx3.await.unwrap().unwrap().text, EXPECTED_GREEDY_TEXT);
}

// ---------- SSE ストリーミング ----------

/// ストリーミングリクエストがキュー経由でも正常な SSE を返す
/// (event: 行必須・Hermes 互換の根幹。§6-3 の教訓)。
#[tokio::test]
async fn streaming_request_works_through_queue() {
    let (app, _state) = test_app(0, 8);
    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(request_body("t0", 20, true)))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp.headers().get("content-type").unwrap().to_str().unwrap();
    assert!(ct.starts_with("text/event-stream"), "content-type = {ct}");
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8(bytes.to_vec()).unwrap();

    // event: 行が必須 (§6-3)
    for ty in ["message_start", "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop"] {
        assert!(text.contains(&format!("event: {ty}\n")), "missing event: {ty} in:\n{text}");
    }
    // 生成テキストが text_delta として正しく流れる (delta 結合が期待値と一致)
    assert_eq!(sse_text(&text), EXPECTED_GREEDY_TEXT, "delta 結合が期待値と不一致:\n{text}");
    assert!(text.contains("\"stop_reason\":\"end_turn\""));
}

/// 待ち行列に入ったストリーミングリクエストも、順番に実行されて
/// 完全な SSE を返す (待ち時間中は keep-alive が流れる設計)。
#[tokio::test]
async fn queued_streaming_requests_both_complete() {
    let (app, _state) = test_app(10, 8);
    let mut handles = Vec::new();
    for _ in 0..2 {
        let app = app.clone();
        let body = request_body("t0", 20, true);
        handles.push(tokio::spawn(async move {
            let req = Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap();
            app.oneshot(req).await.unwrap()
        }));
    }
    for h in handles {
        let resp = h.await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(text.contains("event: message_start\n"));
        assert_eq!(sse_text(&text), EXPECTED_GREEDY_TEXT);
        assert!(text.contains("event: message_stop\n"));
    }
}
