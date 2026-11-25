//! Task 1004: SSE/ストリームアセンブラのチャンク境界エッジケース。
//!
//! `src/stream_asm.rs` の `StreamAssembler` は crate 内部モジュール
//! (`mod stream_asm;` — public ではない) のため、統合テストから直接構築できない。
//! 本ファイルは公開経路 (`/v1/messages` の SSE) に対して、決定的な
//! `ScriptModel` + バイトレベル `ByteLevel` トークナイザで **任意のチャンク分割**
//! を作り、アセンブラの出力契約を固定する。カバー範囲:
//!
//! 1. マルチバイト UTF-8 がトークン境界 (= SSE の delta 境界) で分断されても
//!    文字が欠落・重複・U+FFFD 化しないこと (全バイト境界の分割 + 1 バイトずつ)。
//! 2. ` think` 風の reasoning タグと `<tool_call>` block を 1 文字 1 トークンで
//!    分割しても、テキストが `extract_tool_calls` と同じ意味論で再構成されること。
//! 3. block の開始/終了マーカー (`<tool_call>` / `</tool_call>`) が境界で分断
//!    されても、マーカーの suffix バイトが text delta に漏れないこと。
//! 4. 空 / 空白のみの delta、EOF (max_tokens) での未完結 block の flush、
//!    ランダム分割に対するゼロ panic とイベント列不変量。
//!
//! 既知の未修正ギャップ: クライアント指定 `stop_sequences` がトークン境界を
//! またぐ場合、engine (`crates/denno-engine/src/engine.rs`) は停止語の prefix を
//! holdback しないため停止語の先頭バイトが delta に漏れる。本タスクの Scope は
//! テストファイル追加のみ (engine は編集禁止) のため、その回帰テストは
//! `#[ignore]` で「修正後に有効化する契約」として置く。

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use denno_engine::kv_cache::KvCache;
use denno_engine::model::Model;
use denno_engine::testutil::ScriptModel;
use denno_engine::tokenizer::DennoTokenizer;
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

const EOS_ID: u32 = 12;

// ---------------------------------------------------------------------------
// バイトレベル語彙 (任意のバイト境界で分割できるトークナイザ)
// ---------------------------------------------------------------------------

/// GPT-2 byte-level 写像: 1 バイト -> トークン文字列を構成する 1 文字。
///
/// トークン文字列はこの写像で組み立て、`ByteLevel` decoder が逆写像して
/// 元のバイト列に戻す。これにより「UTF-8 文字の途中で切れたトークン」を
/// 語彙として表現できる。
fn bl_char(b: u8) -> char {
    match b {
        0..=32 => char::from_u32(256 + b as u32).unwrap(),
        127..=160 => char::from_u32(289 + (b as u32 - 127)).unwrap(),
        173 => char::from_u32(323).unwrap(),
        _ => char::from_u32(b as u32).unwrap(),
    }
}

/// 生バイト列 -> トークン文字列 (語彙キー)。
fn piece(bs: &[u8]) -> String {
    bs.iter().map(|&b| bl_char(b)).collect()
}

/// 語彙 (トークン文字列, id) と script (id 列) を組み立てる。
///
/// - id は出現順に 0 から採番する (12 = EOS は予約)。
/// - 同じバイト列の piece は同一トークンに集約する。
fn build_vocab(pieces: &[Vec<u8>]) -> (Vec<(String, u32)>, Vec<u32>) {
    let mut vocab: Vec<(String, u32)> = Vec::new();
    let mut ids: Vec<u32> = Vec::new();
    let mut next = 0u32;
    for p in pieces {
        let key = piece(p);
        let id = match vocab.iter().find(|(t, _)| *t == key) {
            Some((_, id)) => *id,
            None => {
                if next == EOS_ID {
                    next += 1;
                }
                vocab.push((key, next));
                next += 1;
                next - 1
            }
        };
        ids.push(id);
    }
    vocab.push(("eos".to_string(), EOS_ID));
    (vocab, ids)
}

/// テスト用トークナイザ (`decoder: ByteLevel`)。
fn tokenizer(vocab: &[(String, u32)]) -> DennoTokenizer {
    let mut map = serde_json::Map::new();
    let mut max_id = 0;
    for (t, id) in vocab {
        map.insert(t.clone(), json!(id));
        max_id = max_id.max(*id);
    }
    if !map.contains_key("[UNK]") {
        map.insert("[UNK]".to_string(), json!(max_id + 1));
    }
    let tok_json = json!({
        "version": "1.0",
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": true,
            "use_regex": true
        },
        "model": {"type": "WordLevel", "vocab": map, "unk_token": "[UNK]"}
    })
    .to_string();
    DennoTokenizer::from_json_str(&tok_json, None).unwrap()
}

/// `ScriptModel` を包み、コンテキスト長だけ 512 に広げたモデル。
///
/// `ScriptModel::context()` は `DUMMY_CONTEXT` (64) 固定で、engine は
/// `max_tokens` を `context - prompt_len` に clamp する。長い block を 1 文字
/// ずつ流すテストでは 64 では足りないため、同じ生成規則のまま広げる。
struct BigContextScript(ScriptModel);

impl Model for BigContextScript {
    fn context(&self) -> usize {
        512
    }

    fn eos_ids(&self) -> &[u32] {
        self.0.eos_ids()
    }

    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> anyhow::Result<Vec<f32>> {
        self.0.forward(tokens, cache)
    }

    fn new_cache(&self) -> KvCache {
        KvCache::new(1, 4, 512)
    }
}

fn app(script: Vec<u32>, vocab: &[(String, u32)]) -> Router {
    let mut inner = ScriptModel::new(script, 0);
    inner.vocab = vocab.iter().map(|(_, id)| *id as usize + 1).max().unwrap_or(16);
    let model = BigContextScript(inner);
    let tok = tokenizer(vocab);
    let engine = Engine::from_parts(Box::new(model), tok, "denno-test", None);
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

// ---------------------------------------------------------------------------
// SSE / HTTP ヘルパ
// ---------------------------------------------------------------------------

async fn post_stream(app: &Router, body: Value) -> String {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "stream リクエストが失敗");
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert!(ct.starts_with("text/event-stream"), "content-type = {ct}");
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    String::from_utf8(bytes.to_vec()).expect("SSE 本文が UTF-8 でない")
}

fn request_body(max_tokens: usize, stop: Vec<String>) -> Value {
    let mut body = json!({
        "model": "denno-test",
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": true,
        "messages": [{"role": "user", "content": "hello"}],
    });
    if !stop.is_empty() {
        body["stop_sequences"] = json!(stop);
    }
    body
}

fn sse_events(body: &str) -> Vec<Value> {
    body.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .map(|d| serde_json::from_str::<Value>(d).expect("SSE data 行が JSON でない"))
        .collect()
}

fn text_deltas(events: &[Value]) -> String {
    let mut out = String::new();
    for ev in events {
        if ev["type"] == "content_block_delta" && ev["delta"]["type"] == "text_delta" {
            out.push_str(ev["delta"]["text"].as_str().expect("text_delta に text が無い"));
        }
    }
    out
}

/// tool_use block を出現順に (name, input) で返す。
fn tool_uses(events: &[Value]) -> Vec<(String, Value)> {
    let mut current: Option<String> = None;
    let mut out = Vec::new();
    for ev in events {
        match ev["type"].as_str().unwrap_or_default() {
            "content_block_start" => {
                current = if ev["content_block"]["type"] == "tool_use" {
                    Some(ev["content_block"]["name"].as_str().unwrap().to_string())
                } else {
                    None
                };
            }
            "content_block_delta" if ev["delta"]["type"] == "input_json_delta" => {
                let parsed: Value =
                    serde_json::from_str(ev["delta"]["partial_json"].as_str().unwrap())
                        .expect("partial_json が JSON でない");
                out.push((current.clone().expect("tool_use 外の input_json_delta"), parsed));
            }
            "content_block_stop" => current = None,
            _ => {}
        }
    }
    out
}

fn stop_reason(events: &[Value]) -> String {
    events
        .iter()
        .find(|e| e["type"] == "message_delta")
        .expect("message_delta が無い")["delta"]["stop_reason"]
        .as_str()
        .expect("stop_reason が文字列でない")
        .to_string()
}

fn stop_sequence(events: &[Value]) -> Value {
    events
        .iter()
        .find(|e| e["type"] == "message_delta")
        .expect("message_delta が無い")["delta"]["stop_sequence"]
        .clone()
}

/// 全ケース共通のイベント列不変量。
///
/// - message_start が先頭・message_stop が末尾・各 1 回
/// - content_block_start は index 0 から昇順
/// - delta / stop は open 中の index にのみ対応し、二重 stop が無い
/// - tool_use block を出した場合だけ stop_reason = "tool_use"
/// - text delta に U+FFFD (デコード失敗の痕跡) が無い
fn assert_invariants(events: &[Value], body: &str) {
    assert_eq!(events.first().unwrap()["type"], "message_start");
    assert_eq!(events.last().unwrap()["type"], "message_stop");
    let count = |ty: &str| events.iter().filter(|e| e["type"] == ty).count();
    assert_eq!(count("message_start"), 1);
    assert_eq!(count("message_stop"), 1);
    assert_eq!(count("message_delta"), 1);

    let mut open: Vec<u64> = Vec::new();
    let mut starts: Vec<u64> = Vec::new();
    let mut tool_blocks = 0;
    for ev in events {
        match ev["type"].as_str().unwrap() {
            "content_block_start" => {
                let idx = ev["index"].as_u64().expect("start に index が無い");
                starts.push(idx);
                open.push(idx);
                if ev["content_block"]["type"] == "tool_use" {
                    tool_blocks += 1;
                    assert!(ev["content_block"]["id"]
                        .as_str()
                        .unwrap()
                        .starts_with("toolu_"));
                }
            }
            "content_block_delta" => {
                let idx = ev["index"].as_u64().unwrap();
                assert!(open.contains(&idx), "closed block {idx} への delta: {ev}");
                if ev["delta"]["type"] == "input_json_delta" {
                    assert!(
                        serde_json::from_str::<Value>(ev["delta"]["partial_json"].as_str().unwrap())
                            .is_ok()
                    );
                }
            }
            "content_block_stop" => {
                let idx = ev["index"].as_u64().unwrap();
                let pos = open
                    .iter()
                    .position(|&i| i == idx)
                    .unwrap_or_else(|| panic!("未 open の block {idx} を stop: {ev}"));
                open.remove(pos);
            }
            _ => {}
        }
    }
    assert!(open.is_empty(), "stop されていない block: {open:?}");
    let mut sorted = starts.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted, starts, "start index が昇順でない: {starts:?}");
    assert_eq!(starts.first().copied(), Some(0), "先頭 block は index 0");
    let expected = if tool_blocks > 0 { "tool_use" } else { "" };
    if !expected.is_empty() {
        assert_eq!(stop_reason(events), expected);
    } else {
        assert_ne!(stop_reason(events), "tool_use");
    }
    assert!(!body.contains('\u{FFFD}'), "SSE 本文に U+FFFD が混入");
    for ev in events {
        if ev["delta"]["type"] == "text_delta" {
            assert!(!ev["delta"]["text"].as_str().unwrap().contains('\u{FFFD}'));
        }
    }
}

// ---------------------------------------------------------------------------
// script 構築ヘルパ
// ---------------------------------------------------------------------------

/// 生バイト列の列を 1 トークンずつ流し、(app, 復号後の全文) を返す。
fn stream_of(pieces: &[Vec<u8>]) -> (Router, String) {
    let (vocab, ids) = build_vocab(pieces);
    let mut script = ids;
    script.push(EOS_ID);
    let app = app(script, &vocab);
    let raw: Vec<u8> = pieces.iter().flatten().copied().collect();
    (app, String::from_utf8(raw).expect("テスト script が不正な UTF-8"))
}

/// 文字列を 1 文字 1 トークンに分ける。
fn chars_of(s: &str) -> Vec<Vec<u8>> {
    s.chars().map(|c| c.to_string().into_bytes()).collect()
}

/// 文字列を 1 バイト 1 トークンに分ける (UTF-8 の途中でも切る)。
fn bytes_of(s: &str) -> Vec<Vec<u8>> {
    s.as_bytes().iter().map(|b| vec![*b]).collect()
}

/// `<tool_call>` block (XML 形式)。
fn block(name: &str, key: &str, value: &str) -> String {
    format!(
        "<tool_call>\n<function={name}>\n<parameter={key}>\n{value}\n</parameter>\n</function>\n</tool_call>"
    )
}

// ---------------------------------------------------------------------------
// 1. マルチバイト UTF-8 の分断
// ---------------------------------------------------------------------------

/// 全バイト境界 (文字の途中を含む) で 2 分割しても文字が壊れない。
#[tokio::test]
async fn utf8_split_at_every_byte_boundary_is_lossless() {
    let text = "天山海ok";
    let bytes = text.as_bytes();
    for split in 1..bytes.len() {
        let pieces = vec![bytes[..split].to_vec(), bytes[split..].to_vec()];
        let (app, full) = stream_of(&pieces);
        assert_eq!(full, text);
        let body = post_stream(&app, request_body(16, vec![])).await;
        let events = sse_events(&body);
        assert_invariants(&events, &body);
        assert_eq!(text_deltas(&events), text, "split = {split}");
        assert!(tool_uses(&events).is_empty());
        assert_eq!(stop_reason(&events), "end_turn", "split = {split}");
    }
}

/// 1 バイト 1 トークン (最悪ケース) でも欠落・重複なく 1 回ずつ出る。
#[tokio::test]
async fn utf8_one_byte_per_token_is_exact() {
    let text = "日本語と天気";
    let pieces = bytes_of(text);
    let (app, full) = stream_of(&pieces);
    assert_eq!(full, text);
    let body = post_stream(&app, request_body(64, vec![])).await;
    let events = sse_events(&body);
    assert_invariants(&events, &body);
    assert_eq!(text_deltas(&events), text);
    assert_eq!(text_deltas(&events).chars().count(), text.chars().count());
}

/// tool block の引数値が 1 バイトずつに分断されても入力 JSON が壊れない。
#[tokio::test]
async fn utf8_inside_tool_parameter_single_byte_chunks() {
    let full = format!("前置き {}", block("get_weather", "city", "日本語"));
    let pieces = bytes_of(&full);
    let (app, decoded) = stream_of(&pieces);
    assert_eq!(decoded, full);
    let (expect_text, expect_calls) = denno_server::extract_tool_calls(&full);
    assert_eq!(expect_calls.len(), 1);

    let body = post_stream(&app, request_body(256, vec![])).await;
    let events = sse_events(&body);
    assert_invariants(&events, &body);
    assert_eq!(text_deltas(&events), expect_text);
    let tools = tool_uses(&events);
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].0, "get_weather");
    assert_eq!(tools[0].1, json!({"city": "日本語"}));
    assert_eq!(stop_reason(&events), "tool_use");
}

// ---------------------------------------------------------------------------
// 2. reasoning / think タグ + tool block を 1 文字ずつ
// ---------------------------------------------------------------------------

#[tokio::test]
async fn think_and_tool_block_split_into_single_char_chunks() {
    let full = format!(
        "A 天気を確認します  B {} 終わり",
        block("get_weather", "city", "京都")
    );
    let pieces = chars_of(&full);
    let (app, decoded) = stream_of(&pieces);
    assert_eq!(decoded, full);
    let (expect_text, expect_calls) = denno_server::extract_tool_calls(&full);
    assert_eq!(expect_calls.len(), 1);

    let body = post_stream(&app, request_body(512, vec![])).await;
    let events = sse_events(&body);
    assert_invariants(&events, &body);
    assert_eq!(text_deltas(&events), expect_text);
    // think 風タグはサーバーでは加工されない (テキストとして保存される)
    assert!(text_deltas(&events).contains("天気を確認します"));
    let tools = tool_uses(&events);
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].1, json!({"city": "京都"}));
}

/// 1 文字 1 トークンでもマーカーの suffix が text delta に漏れない。
#[tokio::test]
async fn single_char_chunks_never_leak_marker_bytes() {
    let full = format!("a{}b", block("f_tool", "k", "v"));
    let pieces = chars_of(&full);
    let (app, decoded) = stream_of(&pieces);
    assert_eq!(decoded, full);
    let body = post_stream(&app, request_body(256, vec![])).await;
    let events = sse_events(&body);
    assert_invariants(&events, &body);
    let text = text_deltas(&events);
    assert_eq!(text, "ab");
    for leaked in ["<tool_call>", "</tool_call>", "<function=", "</function>", "parameter="] {
        assert!(!text.contains(leaked), "マーカー {leaked:?} が text に漏れた: {text:?}");
    }
    assert_eq!(tool_uses(&events).len(), 1);
}

// ---------------------------------------------------------------------------
// 3. block 開始/終了マーカーの分断
// ---------------------------------------------------------------------------

/// 終了マーカー `</tool_call>` を全境界で 2 分割しても、block が 1 回だけ
/// tool_use になり、マーカーの部分バイトが text に漏れない。
#[tokio::test]
async fn close_marker_straddles_every_boundary_without_leaking() {
    let full = format!("x{}", block("run_tool", "k", "v"));
    let bytes = full.as_bytes();
    for split in 1..bytes.len() {
        let pieces = vec![bytes[..split].to_vec(), bytes[split..].to_vec()];
        let (app, decoded) = stream_of(&pieces);
        assert_eq!(decoded, full);
        let body = post_stream(&app, request_body(64, vec![])).await;
        let events = sse_events(&body);
        assert_invariants(&events, &body);
        assert_eq!(text_deltas(&events), "x", "split = {split}");
        let tools = tool_uses(&events);
        assert_eq!(tools.len(), 1, "split = {split}");
        assert_eq!(tools[0].0, "run_tool", "split = {split}");
        assert_eq!(tools[0].1, json!({"k": "v"}), "split = {split}");
    }
}

/// 開始マーカー `</tool_call>` を全境界で分断しても同様に漏れない。
#[tokio::test]
async fn open_marker_straddles_every_boundary_without_leaking() {
    let full = format!("ab{}cd", block("run_tool", "k", "v"));
    let bytes = full.as_bytes();
    for split in 1..bytes.len() {
        let pieces = vec![bytes[..split].to_vec(), bytes[split..].to_vec()];
        let (app, decoded) = stream_of(&pieces);
        assert_eq!(decoded, full);
        let body = post_stream(&app, request_body(64, vec![])).await;
        let events = sse_events(&body);
        assert_invariants(&events, &body);
        let text = text_deltas(&events);
        assert!(!text.contains("tool_call"), "split = {split}, text = {text:?}");
        assert!(!text.contains("<function="), "split = {split}, text = {text:?}");
        assert_eq!(text, "abcd", "split = {split}");
        assert_eq!(tool_uses(&events).len(), 1, "split = {split}");
    }
}

// ---------------------------------------------------------------------------
// 4. 空 / 空白のみ / EOF / fuzz
// ---------------------------------------------------------------------------

/// 空白のみ・空トークンだけのストリームでは text delta が 1 つも出ない。
#[tokio::test]
async fn whitespace_only_stream_emits_no_text_delta() {
    let pieces = vec![
        b" ".to_vec(),
        b"\n".to_vec(),
        b"  ".to_vec(),
        Vec::new(),
        b"\n ".to_vec(),
    ];
    let (app, _) = stream_of(&pieces);
    let body = post_stream(&app, request_body(16, vec![])).await;
    let events = sse_events(&body);
    assert_invariants(&events, &body);
    assert_eq!(text_deltas(&events), "");
    assert!(tool_uses(&events).is_empty());
    // index 0 の text block だけが開閉される
    let starts: Vec<u64> = events
        .iter()
        .filter(|e| e["type"] == "content_block_start")
        .map(|e| e["index"].as_u64().unwrap())
        .collect();
    assert_eq!(starts, vec![0]);
    assert_eq!(stop_reason(&events), "end_turn");
}

/// max_tokens (EOF 相当) で未完結 block が来ても panic せず、テキストとして
/// 最後まで flush される (`extract_tool_calls` の「未完結は block でない」と同一)。
#[tokio::test]
async fn abrupt_eof_mid_block_flushes_as_text() {
    let full = "abc<tool_call>\n<function=f>\n<parameter=k>\nv";
    let pieces = bytes_of(full);
    let n = pieces.len();
    let (app, decoded) = stream_of(&pieces);
    assert_eq!(decoded, full);
    // script は [piece..., eos] の n+1 個。eos に届かないよう max_tokens = n。
    let body = post_stream(&app, request_body(n, vec![])).await;
    let events = sse_events(&body);
    assert_invariants(&events, &body);
    assert_eq!(text_deltas(&events), full);
    assert!(tool_uses(&events).is_empty());
    assert_eq!(stop_reason(&events), "max_tokens");
}

/// クライアント stop がトークン境界に揃っている場合は打ち切られ、stop 語は漏れない。
#[tokio::test]
async fn client_stop_sequence_aligned_to_chunks_truncates() {
    let pieces = vec![b"xy".to_vec(), b"STOP".to_vec(), b"z".to_vec()];
    let (app, _) = stream_of(&pieces);
    let body = post_stream(&app, request_body(16, vec!["STOP".to_string()])).await;
    let events = sse_events(&body);
    assert_invariants(&events, &body);
    assert_eq!(text_deltas(&events), "xy");
    assert_eq!(stop_reason(&events), "stop_sequence");
    assert_eq!(stop_sequence(&events), json!("STOP"));
}

/// 既知の engine 側ギャップ (Scope 外): クライアント stop がトークン境界を
/// またぐと、engine が停止語 prefix を holdback しないため先頭バイトが漏れる。
/// engine が修正されたらこの `#[ignore]` を外す (そのとき期待値 "xy" で通る)。
#[tokio::test]
#[ignore = "engine.rs が stop_sequences の prefix holdback を持たない (Scope 外・既知ギャップ)"]
async fn client_stop_sequence_straddling_chunks_does_not_leak() {
    let pieces = vec![
        b"xy".to_vec(),
        b"E".to_vec(),
        b"N".to_vec(),
        b"D".to_vec(),
        b"z".to_vec(),
    ];
    let (app, _) = stream_of(&pieces);
    let body = post_stream(&app, request_body(32, vec!["END".to_string()])).await;
    let events = sse_events(&body);
    assert_invariants(&events, &body);
    assert_eq!(text_deltas(&events), "xy", "stop 語の prefix が漏れた");
    assert_eq!(stop_reason(&events), "stop_sequence");
    assert_eq!(stop_sequence(&events), json!("END"));
}

/// 決定的な擬似ランダム分割・入力に対するゼロ panic + extract との一致。
#[tokio::test]
async fn fuzz_like_chunkings_never_panic_and_match_extract() {
    // 行頭ロールラベル ("user" 等) を偶然作らないアルファベット
    let alphabet: Vec<&[u8]> = vec![
        b"a",
        b"b",
        b" ",
        b"\n",
        b"<",
        b">",
        b"<tool_call>",
        b"</tool_call>",
        b"<function=",
        b"</function>",
        b"<parameter=",
        b"</parameter>",
        b"f",
        b"k",
        b"v",
        "天".as_bytes(),
        "日本語".as_bytes(),
    ];

    // 決定的 LCG (外部 crate に依存しない)
    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    let mut next = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };

    for case in 0..40 {
        let len = 6 + next() % 15;
        let pieces: Vec<Vec<u8>> = (0..len)
            .map(|_| alphabet[next() % alphabet.len()].to_vec())
            .collect();
        let (app, full) = stream_of(&pieces);
        let max_tokens = pieces.len(); // eos に届かせない (EOF 経路も混ぜる)
        let body = post_stream(&app, request_body(max_tokens, vec![])).await;
        let events = sse_events(&body);
        assert_invariants(&events, &body);
        let (expect_text, expect_calls) = denno_server::extract_tool_calls(&full);
        assert_eq!(text_deltas(&events), expect_text, "case = {case}, full = {full:?}");
        let tools = tool_uses(&events);
        assert_eq!(tools.len(), expect_calls.len(), "case = {case}, full = {full:?}");
        for (i, got) in tools.iter().enumerate() {
            assert_eq!(got.0, expect_calls[i].name, "case = {case}");
            assert_eq!(got.1, expect_calls[i].input, "case = {case}");
        }
    }
}
