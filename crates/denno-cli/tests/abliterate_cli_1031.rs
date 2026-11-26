//! task 1031: `denno abliterate` の引数検証・層範囲パース・エラー処理の CLI 試験。
//!
//! 本物の `denno` バイナリを子プロセスで起動し、HTTP はこのファイル内の
//! 小さなモックサーバー (std::net::TcpListener) が受ける。確かめること:
//!
//! 1. `--dry-run` はサーバーへ 1 本も送らず、検証済みの計画だけを stdout に出す
//! 2. 不正な `--tag` / `--alpha` / `--url` / `--timeout` はリクエスト前に落ちる
//! 3. analyze の `recommended_range` が層数を超えたら層数へ丸める
//! 4. 非整数の `recommended_range` は黙って 0 にせずエラーで止まる
//! 5. `recommended_range` が空なら全層へ落ちる (警告は stderr)

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

const LAYERS: usize = 8;

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

/// analyze の応答の作り方。
/// 0 = 正常 `[2, 5)`、1 = 層数超え `[2, 40)`、2 = 非整数 `[1.5, 3]`、3 = 空 `[]`。
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
    fn start(mode: u8) -> Mock {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock server");
        listener.set_nonblocking(true).unwrap();
        let addr = listener.local_addr().unwrap();
        let reqs = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let r = Arc::clone(&reqs);
        let s = Arc::clone(&stop);
        let mode = Arc::new(AtomicU8::new(mode));
        thread::spawn(move || {
            while !s.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((mut conn, _)) => {
                        if let Ok(req) = read_request(&mut conn) {
                            r.lock().unwrap().push(req.clone());
                            let _ = respond(&mut conn, &req, mode.load(Ordering::SeqCst));
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

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
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

fn analyze_json(mode: u8) -> String {
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
    let range = match mode {
        1 => serde_json::json!([2, 40]),
        2 => serde_json::json!([1.5, 3]),
        3 => serde_json::json!([]),
        _ => serde_json::json!([2, 5]),
    };
    serde_json::json!({
        "pos": "ablit_pos",
        "neg": "ablit_neg",
        "n_layer": LAYERS,
        "peak_layer": 3,
        "recommended_range": range,
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

fn respond(conn: &mut TcpStream, req: &Req, mode: u8) -> std::io::Result<()> {
    match (req.method.as_str(), req.route()) {
        ("GET", "/v1/models") => json_ok(
            conn,
            r#"{"object":"list","data":[{"id":"mock-model","object":"model"}]}"#,
        ),
        ("POST", "/v1/steer/record") => {
            json_ok(conn, r#"{"epoch":1,"recording":{"name":"x","tokens":0}}"#)
        }
        ("POST", "/v1/steer/record/stop") => json_ok(
            conn,
            r#"{"epoch":1,"recording":null,"saved":{"name":"snap","tokens":4,"created":0,"layers":[0,8]}}"#,
        ),
        ("POST", "/v1/steer/analyze") => json_ok(conn, &analyze_json(mode)),
        ("GET", "/v1/steer/export") => raw_ok(conn, b"DNSTEER1mock"),
        ("POST", "/v1/steer/apply") => json_ok(conn, r#"{"epoch":2,"apply":null}"#),
        ("DELETE", "/v1/steer/apply") => json_ok(conn, r#"{"epoch":3,"apply":null}"#),
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
        ("POST", "/v1/chat/completions") => json_ok(
            conn,
            r#"{"id":"c1","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}"#,
        ),
        _ => {
            let body = format!(r#"{{"error":"no route {} {}"}}"#, req.method, req.route());
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

struct Fixture {
    dir: PathBuf,
}

impl Fixture {
    fn new(tag: &str) -> Fixture {
        let dir = std::env::temp_dir().join(format!(
            "denno-abliterate-1031-{}-{}-{}",
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

fn run_abliterate(url: &str, args: &[String]) -> Output {
    Command::new("timeout")
        .arg("60")
        .arg(env!("CARGO_BIN_EXE_denno"))
        .arg("abliterate")
        .arg("--url")
        .arg(url)
        .args(args)
        .output()
        .expect("run denno abliterate")
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).to_string()
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).to_string()
}

/// 正常系のフル実行に使う引数。out_dir は fixture 内。
fn full_args(fx: &Fixture) -> Vec<String> {
    let pos = fx.file("pos.txt", "p one\np two\n");
    let neg = fx.file("neg.txt", "n one\n");
    let eval = fx.file("eval.txt", "e one\n");
    vec![
        "--pos-file".into(),
        pos.display().to_string(),
        "--neg-file".into(),
        neg.display().to_string(),
        "--eval-file".into(),
        eval.display().to_string(),
        "--out-dir".into(),
        fx.dir.join("out").display().to_string(),
        "--out-recipe".into(),
        fx.dir.join("r.json").display().to_string(),
        "--tag".into(),
        "t1031".into(),
        "--eval-max-tokens".into(),
        "2".into(),
    ]
}

#[test]
fn dry_run_prints_the_plan_and_sends_nothing() {
    let mock = Mock::start(0);
    let fx = Fixture::new("dry");
    let mut args = full_args(&fx);
    args.push("--dry-run".into());
    let out = run_abliterate(&mock.url(), &args);
    assert!(out.status.success(), "stderr: {}", stderr(&out));

    let so = stdout(&out);
    assert!(so.contains("dry-run: サーバーへは何も送らない"), "{so}");
    assert!(so.contains("prompts         = pos 2 / neg 1 / eval 1"), "{so}");
    assert!(so.contains("snapshots       = t1031_pos, t1031_neg"), "{so}");
    assert!(so.contains("capture_mode    = all"), "{so}");
    assert!(so.contains("alpha           = 1"), "{so}");
    // serde_json の Map は key 順を持つので、部分一致で確かめる。
    assert!(so.contains(r#"record body (pos) = {"#), "{so}");
    assert!(
        so.contains("record body (pos) = {") && so.contains(r#""name":"t1031_pos""#),
        "{so}"
    );
    assert!(so.contains(r#""capture_mode":"all""#), "{so}");
    assert!(so.contains("eval_max_tokens = 2"), "{so}");
    // 何も送っていないし、ファイルも作らない
    assert!(mock.reqs().is_empty(), "requests: {:?}", mock.reqs());
    assert!(!fx.dir.join("out").exists());
    assert!(!fx.dir.join("r.json").exists());
}

#[test]
fn bad_scalars_stop_before_any_request() {
    let mock = Mock::start(0);
    let fx = Fixture::new("scalars");
    let cases: Vec<(&str, Vec<String>, &str)> = vec![
        ("tag", vec!["--tag".into(), "a b".into()], "tag"),
        ("empty-tag", vec!["--tag".into(), "".into()], "tag"),
        ("alpha", vec!["--alpha".into(), "0".into()], "alpha"),
        ("alpha-nan", vec!["--alpha".into(), "NaN".into()], "alpha"),
        ("timeout", vec!["--timeout".into(), "0".into()], "timeout"),
        (
            "capture",
            vec!["--capture-mode".into(), "last".into()],
            "capture_mode",
        ),
    ];
    for (name, extra, needle) in cases {
        let mut args = full_args(&fx);
        args.extend(extra);
        let out = run_abliterate(&mock.url(), &args);
        assert!(!out.status.success(), "{name}: unexpectedly succeeded");
        let se = stderr(&out);
        assert!(se.contains(needle), "{name}: stderr: {se}");
        assert!(mock.reqs().is_empty(), "{name}: requests: {:?}", mock.reqs());
    }
}

#[test]
fn zero_eval_max_tokens_is_rejected_before_any_request() {
    let mock = Mock::start(0);
    let fx = Fixture::new("evmax");
    let pos = fx.file("pos.txt", "p\n");
    let neg = fx.file("neg.txt", "n\n");
    let args: Vec<String> = vec![
        "--pos-file".into(),
        pos.display().to_string(),
        "--neg-file".into(),
        neg.display().to_string(),
        "--eval-max-tokens".into(),
        "0".into(),
    ];
    let out = run_abliterate(&mock.url(), &args);
    assert!(!out.status.success());
    assert!(
        stderr(&out).contains("eval_max_tokens"),
        "stderr: {}",
        stderr(&out)
    );
    assert!(mock.reqs().is_empty(), "requests: {:?}", mock.reqs());
}

#[test]
fn bad_url_is_rejected_without_a_socket() {
    let fx = Fixture::new("url");
    let mut args = full_args(&fx);
    args.push("--dry-run".into());
    for url in ["127.0.0.1:8098", "ftp://127.0.0.1", "http://"] {
        let out = run_abliterate(url, &args);
        assert!(!out.status.success(), "{url}");
        let se = stderr(&out);
        assert!(se.contains("url"), "{url}: stderr: {se}");
        assert!(so_stdout_has_no_plan(&out), "{url}: {}", stdout(&out));
    }
}

fn so_stdout_has_no_plan(o: &Output) -> bool {
    !stdout(o).contains("dry-run:")
}

#[test]
fn recommended_range_beyond_layer_count_is_clamped() {
    let mock = Mock::start(1); // [2, 40) を 8 層へ
    let fx = Fixture::new("clamp");
    let args = full_args(&fx);
    let out = run_abliterate(&mock.url(), &args);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    let so = stdout(&out);
    assert!(so.contains("* = recommended [2, 8)"), "{so}");

    let reqs = mock.reqs();
    let apply = reqs
        .iter()
        .find(|r| r.method == "POST" && r.route() == "/v1/steer/apply")
        .expect("apply");
    assert_eq!(apply.json()["layers"], serde_json::json!([2, 8]));
    let recipe = reqs
        .iter()
        .find(|r| r.route() == "/v1/steer/recipe/export")
        .expect("recipe");
    assert_eq!(recipe.json()["layers"], serde_json::json!([2, 8]));
}

#[test]
fn non_integer_recommended_range_is_an_error() {
    let mock = Mock::start(2); // [1.5, 3]
    let fx = Fixture::new("nonint");
    let args = full_args(&fx);
    let out = run_abliterate(&mock.url(), &args);
    assert!(!out.status.success());
    let se = stderr(&out);
    assert!(se.contains("recommended_range"), "stderr: {se}");
    // 分析で落ちるので export も apply も行かない
    let reqs = mock.reqs();
    assert!(
        !reqs.iter().any(|r| r.route() == "/v1/steer/apply"),
        "{reqs:?}"
    );
    assert!(!fx.dir.join("out").exists());
}

#[test]
fn empty_recommended_range_falls_back_to_all_layers() {
    let mock = Mock::start(3); // []
    let fx = Fixture::new("fallback");
    let args = full_args(&fx);
    let out = run_abliterate(&mock.url(), &args);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    let so = stdout(&out);
    assert!(so.contains("* = recommended [0, 8)"), "{so}");
    assert!(stderr(&out).contains("全層"), "stderr: {}", stderr(&out));
    let reqs = mock.reqs();
    let apply = reqs
        .iter()
        .find(|r| r.method == "POST" && r.route() == "/v1/steer/apply")
        .expect("apply");
    assert_eq!(apply.json()["layers"], serde_json::json!([0, 8]));
}
