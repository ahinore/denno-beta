//! task 861: `denno chat` のモックサーバー統合テスト。
//!
//! 本物の `denno` バイナリを子プロセスで起動し、stdin にスクリプトを流して
//! REPL を駆動する。HTTP 側はこのファイル内の小さなモックサーバー
//! (std::net::TcpListener) が受ける。確かめること:
//!
//! 1. anthropic 経路の SSE ストリーミング (text delta が stdout に出る)
//! 2. マルチターン: 2 回目のリクエストに 1 回目の user/assistant が入る
//! 3. openai 経路 + `--system`
//! 4. スラッシュコマンド (/help /steer /stats /save /reset /load /exit)
//! 5. `--model` 未指定時の GET /v1/models による解決

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

#[derive(Clone, Debug)]
struct Req {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: String,
}

impl Req {
    fn json(&self) -> serde_json::Value {
        serde_json::from_str(&self.body).unwrap_or(serde_json::Value::Null)
    }

    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

struct Mock {
    addr: SocketAddr,
    reqs: Arc<Mutex<Vec<Req>>>,
    stop: Arc<AtomicBool>,
}

impl Drop for Mock {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
    }
}

impl Mock {
    fn start() -> Mock {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock server");
        listener.set_nonblocking(true).unwrap();
        let addr = listener.local_addr().unwrap();
        let reqs = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let r = Arc::clone(&reqs);
        let s = Arc::clone(&stop);
        thread::spawn(move || {
            while !s.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((mut conn, _)) => {
                        if let Ok(req) = read_request(&mut conn) {
                            r.lock().unwrap().push(req.clone());
                            let _ = respond(&mut conn, &req);
                        }
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                    }
                    Err(_) => break,
                }
            }
        });
        Mock { addr, reqs, stop }
    }

    fn url(&self) -> String {
        format!("http://{}", self.addr)
    }

    fn reqs(&self) -> Vec<Req> {
        self.reqs.lock().unwrap().clone()
    }
}

fn read_request(conn: &mut TcpStream) -> std::io::Result<Req> {
    conn.set_read_timeout(Some(Duration::from_secs(10)))?;
    let mut buf: Vec<u8> = Vec::new();
    let mut tmp = [0u8; 2048];
    let header_end = loop {
        if let Some(p) = find(&buf, b"\r\n\r\n") {
            break p + 4;
        }
        let n = conn.read(&mut tmp)?;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "eof before headers",
            ));
        }
        buf.extend_from_slice(&tmp[..n]);
    };
    let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let mut lines = head.split("\r\n");
    let mut first = lines.next().unwrap_or("").split_whitespace();
    let method = first.next().unwrap_or("").to_string();
    let path = first.next().unwrap_or("").to_string();
    let mut headers = Vec::new();
    let mut content_length = 0usize;
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            let k = k.trim().to_string();
            let v = v.trim().to_string();
            if k.eq_ignore_ascii_case("content-length") {
                content_length = v.parse().unwrap_or(0);
            }
            headers.push((k, v));
        }
    }
    while buf.len() < header_end + content_length {
        let n = conn.read(&mut tmp)?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
    }
    let body = String::from_utf8_lossy(&buf[header_end..]).to_string();
    Ok(Req {
        method,
        path,
        headers,
        body,
    })
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

fn respond(conn: &mut TcpStream, req: &Req) -> std::io::Result<()> {
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/v1/models") => json_response(
            conn,
            r#"{"object":"list","data":[{"id":"mock-model","object":"model"}],"has_more":false}"#,
        ),
        ("GET", "/v1/steer") => json_response(
            conn,
            r#"{"epoch":3,"capture_mode":"last","recording":null,"snapshots":[{"name":"test-snapshot","tokens":12,"created":0,"layers":[0,1]}],"apply":null}"#,
        ),
        ("POST", "/v1/messages") => sse(conn, &anthropic_sse(&req.body)),
        ("POST", "/v1/chat/completions") => sse(conn, &openai_sse(&req.body)),
        _ => {
            let body = format!(r#"{{"error":"no route {} {}"}}"#, req.method, req.path);
            let head = format!(
                "HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            conn.write_all(head.as_bytes())?;
            conn.write_all(body.as_bytes())?;
            conn.flush()
        }
    }
}

fn json_response(conn: &mut TcpStream, body: &str) -> std::io::Result<()> {
    let head = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    conn.write_all(head.as_bytes())?;
    conn.write_all(body.as_bytes())?;
    conn.flush()
}

fn sse(conn: &mut TcpStream, body: &str) -> std::io::Result<()> {
    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
    conn.write_all(head.as_bytes())?;
    conn.write_all(body.as_bytes())?;
    conn.flush()
}

/// 最後の user 発話と user 数から決定的な応答テキストを作る。
fn expected_reply(body: &str) -> (String, usize) {
    let v: serde_json::Value = serde_json::from_str(body).unwrap_or(serde_json::Value::Null);
    let msgs = v["messages"].as_array().cloned().unwrap_or_default();
    let users: Vec<&serde_json::Value> =
        msgs.iter().filter(|m| m["role"] == "user").collect();
    let last = users
        .last()
        .and_then(|m| m["content"].as_str())
        .unwrap_or("");
    (
        format!("echo:{}|turn:{}", last, users.len()),
        users.len(),
    )
}

fn anthropic_sse(body: &str) -> String {
    let (reply, _) = expected_reply(body);
    let (a, b) = reply.split_at(reply.find('|').unwrap_or(reply.len()));
    let mut out = String::new();
    out.push_str("event: message_start\n");
    out.push_str(
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"m1\",\"model\":\"mock-model\",\"usage\":{\"input_tokens\":7,\"output_tokens\":0}}}\n\n",
    );
    for piece in [a, b] {
        if piece.is_empty() {
            continue;
        }
        out.push_str("event: content_block_delta\n");
        out.push_str(&format!(
            "data: {}\n\n",
            serde_json::json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": piece},
            })
        ));
    }
    out.push_str("event: message_delta\n");
    out.push_str(
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":4}}\n\n",
    );
    out.push_str("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n");
    out
}

fn openai_sse(body: &str) -> String {
    let (reply, _) = expected_reply(body);
    let (a, b) = reply.split_at(reply.find('|').unwrap_or(reply.len()));
    let chunk = |text: &str, finish: serde_json::Value| {
        format!(
            "data: {}\n\n",
            serde_json::json!({
                "id": "c1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "mock-model",
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": finish}],
            })
        )
    };
    let mut out = String::new();
    for piece in [a, b] {
        if !piece.is_empty() {
            out.push_str(&chunk(piece, serde_json::Value::Null));
        }
    }
    out.push_str(&chunk("", serde_json::json!("stop")));
    out.push_str(
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":4}}\n\n",
    );
    out.push_str("data: [DONE]\n\n");
    out
}

fn run_chat(mock: &Mock, args: &[&str], script: &str) -> Output {
    let bin = env!("CARGO_BIN_EXE_denno");
    let mut child = Command::new("timeout")
        .arg("40")
        .arg(bin)
        .arg("chat")
        .arg("--url")
        .arg(mock.url())
        .args(args)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn denno chat");
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(script.as_bytes())
        .expect("write stdin script");
    child.wait_with_output().expect("wait denno chat")
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).to_string()
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).to_string()
}

#[test]
fn anthropic_streams_and_keeps_multi_turn_history() {
    let mock = Mock::start();
    let out = run_chat(
        &mock,
        &["--api", "anthropic", "--max-tokens", "32"],
        "hello\nagain\n/exit\n",
    );
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    let so = stdout(&out);
    assert!(so.contains("echo:hello|turn:1"), "stdout: {so}");
    assert!(so.contains("echo:again|turn:2"), "stdout: {so}");

    let reqs = mock.reqs();
    let chats: Vec<&Req> = reqs.iter().filter(|r| r.path == "/v1/messages").collect();
    assert_eq!(
        chats.len(),
        2,
        "requests: {:?}",
        reqs.iter().map(|r| &r.path).collect::<Vec<_>>()
    );
    // 2 ターン目: user(hello), assistant(echo...), user(again)
    let second = chats[1].json();
    let msgs = second["messages"].as_array().unwrap();
    assert_eq!(msgs.len(), 3, "messages: {msgs:?}");
    assert_eq!(msgs[0]["role"], "user");
    assert_eq!(msgs[0]["content"], "hello");
    assert_eq!(msgs[1]["role"], "assistant");
    assert_eq!(msgs[1]["content"], "echo:hello|turn:1");
    assert_eq!(msgs[2]["content"], "again");
    assert_eq!(second["stream"], true);
    assert_eq!(second["max_tokens"], 32);
}

#[test]
fn resolves_model_from_v1_models() {
    let mock = Mock::start();
    let out = run_chat(&mock, &[], "hi\n/exit\n");
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert!(stdout(&out).contains("echo:hi|turn:1"));
    let reqs = mock.reqs();
    let chat_reqs: Vec<&Req> = reqs.iter().filter(|r| r.path == "/v1/messages").collect();
    assert_eq!(chat_reqs.len(), 1);
    assert_eq!(chat_reqs[0].json()["model"], "mock-model");
    assert!(reqs.iter().any(|r| r.path == "/v1/models"));
}

#[test]
fn openai_streams_and_sends_system_and_api_key() {
    let mock = Mock::start();
    let out = run_chat(
        &mock,
        &[
            "--api",
            "openai",
            "--system",
            "be terse",
            "--api-key",
            "sk-test",
        ],
        "hi\n/exit\n",
    );
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert!(stdout(&out).contains("echo:hi|turn:1"), "stdout: {}", stdout(&out));

    let reqs = mock.reqs();
    let r = reqs
        .iter()
        .find(|r| r.path == "/v1/chat/completions")
        .expect("openai request");
    let body = r.json();
    assert_eq!(body["messages"][0]["role"], "system");
    assert_eq!(body["messages"][0]["content"], "be terse");
    assert_eq!(body["stream_options"]["include_usage"], true);
    assert_eq!(r.header("x-api-key"), Some("sk-test"));
    assert_eq!(r.header("authorization"), Some("Bearer sk-test"));
}

#[test]
fn slash_commands_help_steer_stats_save_load_reset() {
    let mock = Mock::start();
    let dir = std::env::temp_dir().join(format!("denno-chat-it-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let save = dir.join("conv.json");
    let script = format!(
        "hi\n/help\n/stats\n/save {}\n/steer\n/reset\n/stats\n/load {}\n/stats\n/exit\n",
        save.display(),
        save.display()
    );
    let out = run_chat(&mock, &[], &script);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    let so = stdout(&out);
    let se = stderr(&out);
    assert!(so.contains("echo:hi|turn:1"), "stdout: {so}");

    // /help
    assert!(se.contains("/save <file.json>"), "stderr: {se}");
    assert!(se.contains("/steer"), "stderr: {se}");
    // /stats (1 ターン後の実測)
    assert!(se.contains("ttft "), "stderr: {se}");
    assert!(se.contains("out 4"), "stderr: {se}");
    // /save
    assert!(se.contains("saved 2 messages"), "stderr: {se}");
    let saved: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&save).unwrap()).unwrap();
    assert_eq!(saved["version"], 1);
    assert_eq!(saved["messages"].as_array().unwrap().len(), 2);
    assert_eq!(saved["messages"][1]["content"], "echo:hi|turn:1");
    // /steer: モックの GET /v1/steer を表示する
    assert!(se.contains("test-snapshot"), "stderr: {se}");
    assert!(se.contains("\"capture_mode\""), "stderr: {se}");
    // /reset のあとは統計が消える
    let idx = se.find("history cleared").expect("reset notice");
    assert!(se[idx..].contains("no stats yet"), "stderr: {se}");
    // /load
    assert!(se.contains("loaded 2 messages"), "stderr: {se}");

    // 会話 API は 1 回だけ (スラッシュコマンドは送らない)
    assert_eq!(
        mock.reqs().iter().filter(|r| r.path == "/v1/messages").count(),
        1
    );
    std::fs::remove_file(&save).ok();
    std::fs::remove_dir(&dir).ok();
}

#[test]
fn eof_exits_and_errors_do_not_kill_repl() {
    // 存在しない URL 向けの 1 ターンはエラーになるが、REPL は続く。
    let bin = env!("CARGO_BIN_EXE_denno");
    let mut child = Command::new("timeout")
        .arg("40")
        .arg(bin)
        .args(["chat", "--url", "http://127.0.0.1:1", "--timeout", "2"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn denno chat");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"hi\n/help\n")
        .unwrap();
    let out = child.wait_with_output().expect("wait");
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert!(stderr(&out).contains("error:"), "stderr: {}", stderr(&out));
}
