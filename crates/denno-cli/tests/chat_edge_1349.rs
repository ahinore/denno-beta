//! task 1349: `denno chat` のスラッシュコマンド・履歴保存/読込・入力境界の統合テスト。
//!
//! 本物の `denno` バイナリを子プロセスで起動し、stdin にスクリプトを流して REPL を
//! 駆動する。HTTP 側はこのファイル内の小さなモックサーバー (std::net::TcpListener)
//! が受ける。確かめること:
//!
//! 1. `/save` → `/load` の往復で複数ターンの role と本文がそのまま保たれる
//! 2. 存在しない / 壊れた履歴ファイルへの `/load` はエラー表示のみで REPL は続く
//! 3. `/reset` 後のリクエストには新しい user 発話だけが入る
//! 4. 未知のスラッシュコマンドは案内を出して入力を続ける (HTTP は送らない)
//! 5. 空行・空白だけの行は上流へリクエストを送らない

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

#[derive(Clone, Debug)]
struct Req {
    method: String,
    path: String,
    body: String,
}

impl Req {
    fn json(&self) -> serde_json::Value {
        serde_json::from_str(&self.body).unwrap_or(serde_json::Value::Null)
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

    fn chat_bodies(&self) -> Vec<serde_json::Value> {
        self.reqs()
            .iter()
            .filter(|r| r.path == "/v1/messages")
            .map(|r| r.json())
            .collect()
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
    let mut content_length = 0usize;
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                content_length = v.trim().parse().unwrap_or(0);
            }
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
    Ok(Req { method, path, body })
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
        ("POST", "/v1/messages") => sse(conn, &anthropic_sse(&req.body)),
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
fn expected_reply(body: &str) -> String {
    let v: serde_json::Value = serde_json::from_str(body).unwrap_or(serde_json::Value::Null);
    let msgs = v["messages"].as_array().cloned().unwrap_or_default();
    let users: Vec<&serde_json::Value> = msgs.iter().filter(|m| m["role"] == "user").collect();
    let last = users
        .last()
        .and_then(|m| m["content"].as_str())
        .unwrap_or("");
    format!("echo:{}|turn:{}", last, users.len())
}

fn anthropic_sse(body: &str) -> String {
    let reply = expected_reply(body);
    let split = reply.find('|').unwrap_or(reply.len());
    let (a, b) = reply.split_at(split);
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

/// テスト専用の一時ディレクトリ。プロセス id と名前で衝突を避ける。
fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("denno-chat-1349-{}-{}", name, std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn save_load_roundtrip_preserves_multi_turn_roles_and_content() {
    let mock = Mock::start();
    let dir = temp_dir("roundtrip");
    let path = dir.join("conv.json");
    let script = format!(
        "alpha\nbeta\n/save {}\n/reset\n/load {}\ngamma\n/exit\n",
        path.display(),
        path.display()
    );
    let out = run_chat(&mock, &["--api", "anthropic"], &script);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert!(
        stdout(&out).contains("echo:gamma|turn:3"),
        "stdout: {}",
        stdout(&out)
    );

    // /save したファイルは全ターンを role 込みで逐語的に持つ。
    let saved: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(saved["version"], 1);
    assert_eq!(saved["api"], "anthropic");
    let msgs = saved["messages"].as_array().unwrap();
    let pairs: Vec<(&str, &str)> = msgs
        .iter()
        .map(|m| {
            (
                m["role"].as_str().unwrap(),
                m["content"].as_str().unwrap(),
            )
        })
        .collect();
    assert_eq!(
        pairs,
        vec![
            ("user", "alpha"),
            ("assistant", "echo:alpha|turn:1"),
            ("user", "beta"),
            ("assistant", "echo:beta|turn:2"),
        ],
        "saved: {saved}"
    );

    // /reset → /load 後、次のリクエストは復元した 4 発話 + 新規 1 発話になる。
    assert!(stderr(&out).contains("history cleared"));
    assert!(stderr(&out).contains("loaded 4 messages"));
    let bodies = mock.chat_bodies();
    assert_eq!(bodies.len(), 3, "bodies: {bodies:?}");
    let restored = bodies[2]["messages"].as_array().unwrap();
    assert_eq!(restored.len(), 5, "messages: {restored:?}");
    assert_eq!(restored[0]["role"], "user");
    assert_eq!(restored[0]["content"], "alpha");
    assert_eq!(restored[1]["role"], "assistant");
    assert_eq!(restored[1]["content"], "echo:alpha|turn:1");
    assert_eq!(restored[2]["content"], "beta");
    assert_eq!(restored[3]["content"], "echo:beta|turn:2");
    assert_eq!(restored[4]["role"], "user");
    assert_eq!(restored[4]["content"], "gamma");

    std::fs::remove_file(&path).ok();
    std::fs::remove_dir(&dir).ok();
}

#[test]
fn load_missing_or_malformed_file_errors_without_killing_repl() {
    let mock = Mock::start();
    let dir = temp_dir("badload");
    let missing = dir.join("does-not-exist.json");
    let malformed = dir.join("broken.json");
    std::fs::write(&malformed, "{ this is not json").unwrap();
    let script = format!(
        "/load {}\n/load {}\nhi\n/exit\n",
        missing.display(),
        malformed.display()
    );
    let out = run_chat(&mock, &["--api", "anthropic"], &script);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    let se = stderr(&out);
    // 2 回とも REPL 内でエラー表示され、そのあとの発話は普通に通る。
    assert!(
        se.matches("error:").count() >= 2,
        "expected two REPL errors, stderr: {se}"
    );
    assert!(se.contains("does-not-exist.json"), "stderr: {se}");
    assert!(se.contains("broken.json"), "stderr: {se}");
    assert!(se.contains("expected /save format"), "stderr: {se}");
    assert!(
        stdout(&out).contains("echo:hi|turn:1"),
        "stdout: {}",
        stdout(&out)
    );
    // 履歴は空のまま 1 発話だけが送られる。
    let bodies = mock.chat_bodies();
    assert_eq!(bodies.len(), 1);
    assert_eq!(bodies[0]["messages"].as_array().unwrap().len(), 1);

    std::fs::remove_file(&malformed).ok();
    std::fs::remove_dir(&dir).ok();
}

#[test]
fn reset_drops_prior_turns_from_next_request() {
    let mock = Mock::start();
    let out = run_chat(&mock, &["--api", "anthropic"], "first\n/reset\nsecond\n/exit\n");
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert!(stderr(&out).contains("history cleared"));

    let bodies = mock.chat_bodies();
    assert_eq!(bodies.len(), 2, "bodies: {bodies:?}");
    assert_eq!(bodies[0]["messages"].as_array().unwrap().len(), 1);
    let after = bodies[1]["messages"].as_array().unwrap();
    assert_eq!(after.len(), 1, "messages: {after:?}");
    assert_eq!(after[0]["role"], "user");
    assert_eq!(after[0]["content"], "second");
    assert!(!bodies[1].to_string().contains("first"));
}

#[test]
fn unknown_slash_command_prints_hint_and_repl_continues() {
    let mock = Mock::start();
    let out = run_chat(&mock, &["--api", "anthropic"], "/bogus\nhi\n/exit\n");
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    let se = stderr(&out);
    assert!(se.contains("unknown command"), "stderr: {se}");
    assert!(se.contains("/bogus"), "stderr: {se}");
    assert!(se.contains("/help"), "stderr: {se}");
    assert!(
        stdout(&out).contains("echo:hi|turn:1"),
        "stdout: {}",
        stdout(&out)
    );

    // 未知コマンドは上流へ送られない。
    let bodies = mock.chat_bodies();
    assert_eq!(bodies.len(), 1, "bodies: {bodies:?}");
    assert!(!bodies[0].to_string().contains("/bogus"));
}

#[test]
fn empty_and_whitespace_lines_send_no_http_requests() {
    let mock = Mock::start();
    let out = run_chat(&mock, &["--api", "anthropic"], "\n   \n\t\n\n/exit\n");
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert!(mock.chat_bodies().is_empty(), "reqs: {:?}", mock.reqs());
    assert!(stdout(&out).trim().is_empty(), "stdout: {}", stdout(&out));
}
