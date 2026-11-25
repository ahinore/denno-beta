//! task 865: `denno abliterate` のモックサーバー統合テスト。
//!
//! 本物の `denno` バイナリを子プロセスで起動し、HTTP はこのファイル内の
//! 小さなモックサーバー (std::net::TcpListener) が受ける。確かめること:
//!
//! 1. record → プロンプトを 1 本ずつ prefill → stop、を pos / neg の 2 回
//! 2. analyze の応答から層チャートを描き、推奨区間を `*` で示す
//! 3. `.dns` とレシピ JSON の書き出し
//! 4. apply → 差分評価の OFF/ON (DELETE と再 apply) と一致本数の集計
//! 5. サーバーエラー (analyze 400) と不正な --capture-mode で綺麗に止まる

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

#[derive(Clone, Debug)]
struct Req {
    method: String,
    path: String,
    body: Vec<u8>,
}

impl Req {
    fn json(&self) -> serde_json::Value {
        serde_json::from_slice(&self.body).unwrap_or(serde_json::Value::Null)
    }

    fn route(&self) -> &str {
        self.path.split('?').next().unwrap_or(&self.path)
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
        Mock::start_with(0)
    }

    /// mode 0 = 正常、1 = analyze が 400 を返す。
    fn start_with(mode: u8) -> Mock {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock server");
        listener.set_nonblocking(true).unwrap();
        let addr = listener.local_addr().unwrap();
        let reqs = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let applied = Arc::new(AtomicBool::new(false));
        let r = Arc::clone(&reqs);
        let s = Arc::clone(&stop);
        let a = Arc::clone(&applied);
        let mode = Arc::new(AtomicU8::new(mode));
        thread::spawn(move || {
            while !s.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((mut conn, _)) => {
                        if let Ok(req) = read_request(&mut conn) {
                            r.lock().unwrap().push(req.clone());
                            let _ = respond(&mut conn, &req, &a, mode.load(Ordering::SeqCst));
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
    let body = buf[header_end..].to_vec();
    Ok(Req {
        method,
        path,
        body,
    })
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

fn query_param(path: &str, key: &str) -> Option<String> {
    let q = path.split_once('?')?.1;
    for kv in q.split('&') {
        if let Some((k, v)) = kv.split_once('=') {
            if k == key {
                return Some(v.to_string());
            }
        }
    }
    None
}

fn respond(
    conn: &mut TcpStream,
    req: &Req,
    applied: &AtomicBool,
    mode: u8,
) -> std::io::Result<()> {
    match (req.method.as_str(), req.route()) {
        ("GET", "/v1/models") => json_ok(
            conn,
            r#"{"object":"list","data":[{"id":"mock-model","object":"model"}]}"#,
        ),
        ("POST", "/v1/steer/record") => {
            // 記録開始。状態は名前だけ覚えなくてよい (stop は常に成功する)。
            json_ok(conn, r#"{"epoch":1,"recording":{"name":"x","tokens":0}}"#)
        }
        ("POST", "/v1/steer/record/stop") => json_ok(
            conn,
            r#"{"epoch":1,"recording":null,"saved":{"name":"snap","tokens":9,"created":0,"layers":[0,8]}}"#,
        ),
        ("POST", "/v1/steer/analyze") if mode == 1 => {
            let body = r#"{"type":"error","error":{"type":"invalid_request_error","message":"mock analyze failure"}}"#;
            let head = format!(
                "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            conn.write_all(head.as_bytes())?;
            conn.write_all(body.as_bytes())?;
            conn.flush()
        }
        ("POST", "/v1/steer/analyze") => json_ok(conn, &analyze_json()),
        ("GET", "/v1/steer/export") => {
            let name = query_param(&req.path, "name").unwrap_or_else(|| "?".into());
            let body = format!("DNSTEER1mock-{name}");
            raw_ok(conn, body.as_bytes())
        }
        ("POST", "/v1/steer/apply") => {
            applied.store(true, Ordering::SeqCst);
            json_ok(conn, r#"{"epoch":2,"apply":{"layers":[2,5],"alpha":0.6}}"#)
        }
        ("DELETE", "/v1/steer/apply") => {
            applied.store(false, Ordering::SeqCst);
            json_ok(conn, r#"{"epoch":3,"apply":null}"#)
        }
        ("POST", "/v1/steer/recipe/export") => {
            let b = req.json();
            json_ok(
                conn,
                &serde_json::json!({
                    "format": "denno-steer-recipe/1",
                    "name": b["name"],
                    "terms": b["terms"],
                    "layers": b["layers"],
                    "alpha": b["alpha"],
                    "mode": b["mode"],
                    "embedded": [],
                })
                .to_string(),
            )
        }
        ("POST", "/v1/chat/completions") => {
            let b = req.json();
            let prompt = b["messages"][0]["content"].as_str().unwrap_or("");
            let on = applied.load(Ordering::SeqCst);
            let text = if prompt.starts_with("same") {
                "constant text".to_string()
            } else {
                format!("{}:{}", if on { "ON" } else { "OFF" }, prompt)
            };
            let body = serde_json::json!({
                "id": "c1",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3},
            });
            json_ok(conn, &body.to_string())
        }
        _ => {
            let body = format!(
                r#"{{"error":"no route {} {}"}}"#,
                req.method,
                req.route()
            );
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

/// 8 層。sep は 0.1, 0.2, 0.55, 0.9, 0.6, 0.3, 0.15, 0.05。推奨 [2, 5)。
fn analyze_json() -> String {
    let seps = [0.1, 0.2, 0.55, 0.9, 0.6, 0.3, 0.15, 0.05];
    let layers: Vec<serde_json::Value> = seps
        .iter()
        .enumerate()
        .map(|(i, s)| {
            serde_json::json!({
                "layer": i,
                "diff_norm": 0.05 * (i as f64 + 1.0),
                "norm_pos": 1.0,
                "norm_neg": 1.0,
                "cos_sim": 0.95 - 0.05 * i as f64,
                "sep_score": s,
            })
        })
        .collect();
    serde_json::json!({
        "pos": "ablit_pos",
        "neg": "ablit_neg",
        "n_layer": 8,
        "peak_layer": 3,
        "recommended_range": [2, 5],
        "layers": layers,
    })
    .to_string()
}

fn json_ok(conn: &mut TcpStream, body: &str) -> std::io::Result<()> {
    let head = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    conn.write_all(head.as_bytes())?;
    conn.write_all(body.as_bytes())?;
    conn.flush()
}

fn raw_ok(conn: &mut TcpStream, body: &[u8]) -> std::io::Result<()> {
    let head = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    conn.write_all(head.as_bytes())?;
    conn.write_all(body)?;
    conn.flush()
}

struct Fixture {
    dir: PathBuf,
}

impl Fixture {
    fn new(tag: &str) -> Fixture {
        let dir = std::env::temp_dir().join(format!(
            "denno-abliterate-{}-{}-{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        Fixture { dir }
    }

    fn file(&self, name: &str, text: &str) -> PathBuf {
        let p = self.dir.join(name);
        std::fs::write(&p, text).unwrap();
        p
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn run_abliterate(mock: &Mock, args: &[String]) -> Output {
    let bin = env!("CARGO_BIN_EXE_denno");
    let out = Command::new("timeout")
        .arg("60")
        .arg(bin)
        .arg("abliterate")
        .arg("--url")
        .arg(mock.url())
        .args(args)
        .output()
        .expect("run denno abliterate");
    out
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).to_string()
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).to_string()
}

#[test]
fn record_chart_export_apply_and_eval_off_on() {
    let mock = Mock::start();
    let fx = Fixture::new("full");
    let pos = fx.file("pos.txt", "# comment\npos one\n\npos two\n");
    let neg = fx.file("neg.txt", "neg one\nneg two\n");
    let eval = fx.file("eval.txt", "same prompt\nchange me please\n# skip\nanother one\n");
    let out_dir = fx.dir.join("out");
    let recipe = fx.dir.join("r865.json");

    let args: Vec<String> = vec![
        "--pos-file".into(),
        pos.display().to_string(),
        "--neg-file".into(),
        neg.display().to_string(),
        "--eval-file".into(),
        eval.display().to_string(),
        "--capture-mode".into(),
        "last_prompt_token".into(),
        "--alpha".into(),
        "0.6".into(),
        "--out-dir".into(),
        out_dir.display().to_string(),
        "--out-recipe".into(),
        recipe.display().to_string(),
        "--tag".into(),
        "t865".into(),
        "--eval-max-tokens".into(),
        "8".into(),
    ];
    let out = run_abliterate(&mock, &args);
    assert!(out.status.success(), "stderr: {}", stderr(&out));

    let so = stdout(&out);
    let se = stderr(&out);
    // チャート: 8 層ぶんの行 + 見出し + 末尾
    assert!(so.contains("per-layer separation (8 layers"), "stdout: {so}");
    assert!(so.contains("* = recommended [2, 5)"), "stdout: {so}");
    assert!(so.contains("  3 |"), "stdout: {so}");
    assert!(so.contains("peak layer 3"), "stdout: {so}");
    let starred = so
        .lines()
        .filter(|l| l.starts_with("  ") && l.contains('|') && l.ends_with('*'))
        .count();
    assert_eq!(starred, 3, "stdout: {so}");
    // EOF 評価: same は一致、ほか 2 本は変わる
    assert!(so.contains("identical 1 / 3"), "stdout: {so}");
    assert!(so.contains("mean similarity"), "stdout: {so}");
    assert!(so.contains("OFF:change me please => ON:change me please"), "stdout: {so}");

    // .dns
    let pos_dns = std::fs::read(out_dir.join("t865_pos.dns")).expect("pos .dns");
    let neg_dns = std::fs::read(out_dir.join("t865_neg.dns")).expect("neg .dns");
    assert!(pos_dns.starts_with(b"DNSTEER1"), "{pos_dns:?}");
    assert_eq!(
        String::from_utf8_lossy(&pos_dns),
        "DNSTEER1mock-t865_pos"
    );
    assert_eq!(String::from_utf8_lossy(&neg_dns), "DNSTEER1mock-t865_neg");

    // レシピ
    let rv: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&recipe).unwrap()).unwrap();
    assert_eq!(rv["name"], "t865-recipe");
    assert_eq!(rv["layers"], serde_json::json!([2, 5]));
    assert!((rv["alpha"].as_f64().unwrap() - 0.6).abs() < 1e-6);
    assert_eq!(rv["mode"], "subtract");
    assert_eq!(rv["terms"][0]["snapshot"], "t865_pos");
    assert_eq!(rv["terms"][1]["coef"], -1.0);

    // リクエストの並び
    let reqs = mock.reqs();
    let routes: Vec<String> = reqs
        .iter()
        .map(|r| format!("{} {}", r.method, r.route()))
        .collect();
    let pos_idx = routes
        .iter()
        .position(|r| r == "POST /v1/steer/record")
        .expect("record");
    assert_eq!(routes[pos_idx + 3], "POST /v1/steer/record/stop", "{routes:?}");
    assert_eq!(routes[pos_idx + 4], "POST /v1/steer/record", "{routes:?}");
    assert_eq!(routes[pos_idx + 7], "POST /v1/steer/record/stop", "{routes:?}");
    let analyze_idx = routes
        .iter()
        .position(|r| r == "POST /v1/steer/analyze")
        .expect("analyze");
    assert_eq!(analyze_idx, pos_idx + 8, "{routes:?}");
    assert_eq!(reqs[analyze_idx].json(), serde_json::json!({"pos": "t865_pos", "neg": "t865_neg"}));

    // record の body
    assert_eq!(
        reqs[pos_idx].json(),
        serde_json::json!({"name": "t865_pos", "capture_mode": "last_prompt_token"})
    );
    assert_eq!(reqs[pos_idx + 4].json()["name"], "t865_neg");

    // 記録中の生成は 1 トークン、評価は 8 トークン
    let chats: Vec<&Req> = reqs.iter().filter(|r| r.route() == "/v1/chat/completions").collect();
    assert_eq!(chats.len(), 2 + 2 + 3 + 3, "chats: {}", chats.len());
    for c in &chats[..4] {
        assert_eq!(c.json()["max_tokens"], 1);
        assert_eq!(c.json()["stream"], false);
    }
    for c in &chats[4..] {
        assert_eq!(c.json()["max_tokens"], 8);
    }

    // export は pos / neg の 2 回で、どちらも raw (.dns)
    let exports: Vec<&Req> = reqs.iter().filter(|r| r.route() == "/v1/steer/export").collect();
    assert_eq!(exports.len(), 2);
    assert_eq!(exports[0].path, "/v1/steer/export?name=t865_pos");
    assert!(exports.iter().all(|r| !r.path.contains("format=json")));

    // apply (1 回目) → recipe/export → DELETE apply (OFF) → apply (ON)
    let applies: Vec<usize> = routes
        .iter()
        .enumerate()
        .filter(|(_, r)| *r == "POST /v1/steer/apply")
        .map(|(i, _)| i)
        .collect();
    assert_eq!(applies.len(), 2, "{routes:?}");
    let del = routes
        .iter()
        .position(|r| r == "DELETE /v1/steer/apply")
        .expect("delete apply");
    assert!(applies[0] < del && del < applies[1], "{routes:?}");
    let recipe_idx = routes
        .iter()
        .position(|r| r == "POST /v1/steer/recipe/export")
        .expect("recipe export");
    assert!(applies[0] < recipe_idx && recipe_idx < del, "{routes:?}");
    // ON の適用がレシピと同じ (pos - neg, [2,5))
    let on = reqs[applies[1]].json();
    assert_eq!(on["terms"][0]["snapshot"], "t865_pos");
    assert_eq!(on["terms"][1]["coef"], -1.0);
    assert_eq!(on["layers"], serde_json::json!([2, 5]));
    assert!((on["alpha"].as_f64().unwrap() - 0.6).abs() < 1e-6);
    assert_eq!(on["mode"], "subtract");

    // 評価の生成は DELETE の後 (OFF) と 2 回目の apply の後 (ON)
    let first_eval_chat = routes
        .iter()
        .enumerate()
        .filter(|(_, r)| *r == "POST /v1/chat/completions")
        .map(|(i, _)| i)
        .nth(4)
        .expect("eval chat");
    assert!(del < first_eval_chat && first_eval_chat < applies[1], "{routes:?}");
    assert!(se.contains("snapshot t865_pos"), "stderr: {se}");
    assert!(se.contains("apply は ON のまま"), "stderr: {se}");
}

#[test]
fn analyze_server_error_fails_with_message() {
    let mock = Mock::start_with(1);
    let fx = Fixture::new("err");
    let pos = fx.file("pos.txt", "p\n");
    let neg = fx.file("neg.txt", "n\n");
    let args: Vec<String> = vec![
        "--pos-file".into(),
        pos.display().to_string(),
        "--neg-file".into(),
        neg.display().to_string(),
        "--out-dir".into(),
        fx.dir.join("out").display().to_string(),
    ];
    let out = run_abliterate(&mock, &args);
    assert!(!out.status.success(), "stdout: {}", stdout(&out));
    let se = stderr(&out);
    assert!(se.contains("HTTP 400"), "stderr: {se}");
    assert!(se.contains("mock analyze failure"), "stderr: {se}");
    // 分析で落ちるので .dns は書かない
    assert!(!fx.dir.join("out").exists());
}

#[test]
fn bad_capture_mode_stops_before_any_request() {
    let mock = Mock::start();
    let fx = Fixture::new("badmode");
    let pos = fx.file("pos.txt", "p\n");
    let neg = fx.file("neg.txt", "n\n");
    let args: Vec<String> = vec![
        "--pos-file".into(),
        pos.display().to_string(),
        "--neg-file".into(),
        neg.display().to_string(),
        "--capture-mode".into(),
        "last".into(),
    ];
    let out = run_abliterate(&mock, &args);
    assert!(!out.status.success());
    let se = stderr(&out);
    assert!(se.contains("capture_mode"), "stderr: {se}");
    assert!(mock.reqs().is_empty(), "requests: {:?}", mock.reqs());
}

#[test]
fn empty_prompt_file_is_an_error() {
    let mock = Mock::start();
    let fx = Fixture::new("empty");
    let pos = fx.file("pos.txt", "# only a comment\n\n");
    let neg = fx.file("neg.txt", "n\n");
    let args: Vec<String> = vec![
        "--pos-file".into(),
        pos.display().to_string(),
        "--neg-file".into(),
        neg.display().to_string(),
    ];
    let out = run_abliterate(&mock, &args);
    assert!(!out.status.success());
    assert!(stderr(&out).contains("有効なプロンプトが 0 本"), "stderr: {}", stderr(&out));
    assert!(mock.reqs().is_empty());
}
