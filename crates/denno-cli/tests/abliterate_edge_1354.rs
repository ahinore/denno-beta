//! task 1354: `denno abliterate` の入力検証と出力ファイルのスキーマ境界試験。
//!
//! 本物の `denno` バイナリを子プロセスで起動し、HTTP はこのファイル内の
//! 小さなモックサーバー (std::net::TcpListener) が受ける。確かめること:
//!
//! 1. pos / neg のプロンプトファイルが無い → ファイルパス入りの明確なエラーで
//!    非ゼロ終了し、HTTP を 1 本も送らない
//! 2. 空白だけ / コメントだけのプロンプトファイル → 送信前に明確なエラー
//! 3. プロンプト集の空行・`#` コメント・CRLF が、実際に送る本文で落ちている
//! 4. 書き出した `.dns` (方向) とレシピ JSON (レポート) のキー・有限値・
//!    読み直し (round-trip)
//! 5. 不正な `--capture-mode` と、サーバーが返す不正な層指定
//!    (`recommended_range`) は出力を書く前に拒否する
//!
//! ここで扱う「JSONL」は abliterate の入力には無い。入力は 1 行 1 プロンプトの
//! テキストだけなので、壊れた入力は「非 UTF-8 のプロンプトファイル」で確かめる。

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

/// analyze の応答の作り方。
/// 0 = 正常 `[2, 5)`、1 = 非整数 `[1.5, 3]`、2 = 配列でない、3 = 負値 `[-1, 3]`。
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
        1 => serde_json::json!([1.5, 3]),
        2 => serde_json::json!({"from": 1, "to": 3}),
        3 => serde_json::json!([-1, 3]),
        _ => serde_json::json!([2, 5]),
    };
    serde_json::json!({
        "pos": "ablit_pos",
        "neg": "ablit_neg",
        "n_layer": 8,
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
        ("GET", "/v1/steer/export") => {
            let name = req
                .path
                .split_once("name=")
                .map(|(_, v)| v.split('&').next().unwrap_or(v))
                .unwrap_or("?");
            raw_ok(conn, format!("DNSTEER1mock-{name}").as_bytes())
        }
        ("POST", "/v1/steer/apply") => {
            json_ok(conn, r#"{"epoch":2,"apply":{"layers":[2,5],"alpha":0.7}}"#)
        }
        ("DELETE", "/v1/steer/apply") => json_ok(conn, r#"{"epoch":3,"apply":null}"#),
        ("POST", "/v1/steer/recipe/export") => {
            let b = req.json();
            json_ok(
                conn,
                &serde_json::json!({
                    "format": "denno-steer-recipe/1",
                    "name": b["name"],
                    "description": b["description"],
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

struct Fixture {
    dir: PathBuf,
}

impl Fixture {
    fn new(tag: &str) -> Fixture {
        let dir = std::env::temp_dir().join(format!(
            "denno-abliterate-1354-{}-{}-{}",
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

    fn bytes(&self, name: &str, raw: &[u8]) -> PathBuf {
        let p = self.dir.join(name);
        std::fs::write(&p, raw).unwrap();
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

/// 子プロセスが panic していないこと (「明確なエラー」の確認)。
fn assert_no_panic(o: &Output, ctx: &str) {
    let se = stderr(o);
    assert!(!se.contains("panicked"), "{ctx}: stderr: {se}");
    assert!(!se.contains("RUST_BACKTRACE"), "{ctx}: stderr: {se}");
}

fn base_args(pos: &PathBuf, neg: &PathBuf, out_dir: &PathBuf) -> Vec<String> {
    vec![
        "--pos-file".into(),
        pos.display().to_string(),
        "--neg-file".into(),
        neg.display().to_string(),
        "--out-dir".into(),
        out_dir.display().to_string(),
    ]
}

// ---------------------------------------------------------------------------
// 1. プロンプトファイルが無い
// ---------------------------------------------------------------------------

#[test]
fn missing_prompt_files_fail_with_the_path_in_stderr() {
    let mock = Mock::start(0);
    let fx = Fixture::new("missing");
    let pos = fx.file("pos.txt", "p\n");
    let neg = fx.file("neg.txt", "n\n");
    let out_dir = fx.dir.join("out");

    let absent_pos = fx.dir.join("no-such-pos.txt");
    let absent_neg = fx.dir.join("no-such-neg.txt");

    let cases = [
        (
            "missing-pos",
            base_args(&absent_pos, &neg, &out_dir),
            absent_pos.display().to_string(),
        ),
        (
            "missing-neg",
            base_args(&pos, &absent_neg, &out_dir),
            absent_neg.display().to_string(),
        ),
    ];
    for (name, args, path) in cases {
        let out = run_abliterate(&mock.url(), &args);
        assert!(!out.status.success(), "{name}: unexpectedly succeeded");
        let se = stderr(&out);
        assert!(se.contains(&path), "{name}: stderr: {se}");
        assert!(se.contains("read prompts"), "{name}: stderr: {se}");
        assert_no_panic(&out, name);
        assert!(
            mock.reqs().is_empty(),
            "{name}: requests: {:?}",
            mock.reqs()
        );
    }
    assert!(!out_dir.exists(), "out_dir must not be created");
}

// ---------------------------------------------------------------------------
// 2. 空白 / コメントだけのプロンプトファイル
// ---------------------------------------------------------------------------

#[test]
fn whitespace_only_prompt_files_fail_before_any_request() {
    let mock = Mock::start(0);
    let fx = Fixture::new("blank");
    let good_pos = fx.file("pos.txt", "p\n");
    let good_neg = fx.file("neg.txt", "n\n");
    let blank = fx.file("blank.txt", "   \n\t\n\r\n  \r\n");
    let comment_only = fx.file("comment.txt", "# one\n   # indented-ish\n\n");

    for (name, blank_file) in [("blank", &blank), ("comment-only", &comment_only)] {
        let out_dir = fx.dir.join(format!("out-{name}"));
        let cases = [
            (
                "pos",
                base_args(blank_file, &good_neg, &out_dir),
                blank_file.display().to_string(),
            ),
            (
                "neg",
                base_args(&good_pos, blank_file, &out_dir),
                blank_file.display().to_string(),
            ),
        ];
        for (which, args, path) in cases {
            let ctx = format!("{name}/{which}");
            let out = run_abliterate(&mock.url(), &args);
            assert!(
                !out.status.success(),
                "{ctx}: unexpectedly succeeded: {}",
                stdout(&out)
            );
            let se = stderr(&out);
            assert!(
                se.contains("有効なプロンプトが 0 本"),
                "{ctx}: stderr: {se}"
            );
            assert!(se.contains(&path), "{ctx}: stderr: {se}");
            assert_no_panic(&out, &ctx);
            assert!(
                mock.reqs().is_empty(),
                "{ctx}: requests: {:?}",
                mock.reqs()
            );
        }
        assert!(!out_dir.exists(), "{name}: out_dir must not be created");
    }
}

#[test]
fn whitespace_only_eval_file_fails_before_any_request() {
    let mock = Mock::start(0);
    let fx = Fixture::new("blank-eval");
    let pos = fx.file("pos.txt", "p\n");
    let neg = fx.file("neg.txt", "n\n");
    let eval = fx.file("eval.txt", "\n\t\n");
    let out_dir = fx.dir.join("out");
    let mut args = base_args(&pos, &neg, &out_dir);
    args.extend(["--eval-file".into(), eval.display().to_string()]);

    let out = run_abliterate(&mock.url(), &args);
    assert!(!out.status.success());
    let se = stderr(&out);
    assert!(se.contains("有効なプロンプトが 0 本"), "stderr: {se}");
    assert!(se.contains(&eval.display().to_string()), "stderr: {se}");
    assert_no_panic(&out, "blank-eval");
    assert!(
        mock.reqs().is_empty(),
        "requests: {:?}",
        mock.reqs()
    );
    assert!(!out_dir.exists());
}

// ---------------------------------------------------------------------------
// 3. 壊れた入力 (非 UTF-8 / ディレクトリ)
// ---------------------------------------------------------------------------

#[test]
fn non_utf8_prompt_file_fails_cleanly_with_the_path() {
    let mock = Mock::start(0);
    let fx = Fixture::new("nonutf8");
    // 0xff 0xfe は UTF-8 として不正。行構造だけは持たせる。
    let raw = fx.bytes("pos.bin", b"valid line\n\xff\xfe broken\n");
    let neg = fx.file("neg.txt", "n\n");
    let out_dir = fx.dir.join("out");

    let out = run_abliterate(&mock.url(), &base_args(&raw, &neg, &out_dir));
    assert!(!out.status.success());
    let se = stderr(&out);
    assert!(se.contains(&raw.display().to_string()), "stderr: {se}");
    assert_no_panic(&out, "non-utf8");
    assert!(
        mock.reqs().is_empty(),
        "requests: {:?}",
        mock.reqs()
    );
    assert!(!out_dir.exists());
}

#[test]
fn directory_as_prompt_file_fails_cleanly_with_the_path() {
    let mock = Mock::start(0);
    let fx = Fixture::new("dir-as-file");
    let dir_as_prompt = fx.dir.join("prompts.d");
    std::fs::create_dir_all(&dir_as_prompt).unwrap();
    let neg = fx.file("neg.txt", "n\n");
    let out_dir = fx.dir.join("out");

    let out = run_abliterate(&mock.url(), &base_args(&dir_as_prompt, &neg, &out_dir));
    assert!(!out.status.success());
    let se = stderr(&out);
    assert!(
        se.contains(&dir_as_prompt.display().to_string()),
        "stderr: {se}"
    );
    assert_no_panic(&out, "dir-as-file");
    assert!(
        mock.reqs().is_empty(),
        "requests: {:?}",
        mock.reqs()
    );
}

// ---------------------------------------------------------------------------
// 4. 空行・コメント・CRLF の実配線
// ---------------------------------------------------------------------------

#[test]
fn blank_lines_comments_and_crlf_are_stripped_from_the_sent_prompts() {
    let mock = Mock::start(0);
    let fx = Fixture::new("parse");
    let pos = fx.file(
        "pos.txt",
        "  pos one  \r\n\r\n# comment\r\n\tpos two\t\r\n#tail\r\n",
    );
    let neg = fx.file("neg.txt", "neg one\r\n\r\nneg two\r\n");
    let out_dir = fx.dir.join("out");

    let out = run_abliterate(&mock.url(), &base_args(&pos, &neg, &out_dir));
    assert!(out.status.success(), "stderr: {}", stderr(&out));

    let reqs = mock.reqs();
    let texts: Vec<String> = reqs
        .iter()
        .filter(|r| r.route() == "/v1/chat/completions")
        .map(|r| {
            r.json()["messages"][0]["content"]
                .as_str()
                .unwrap_or("<none>")
                .to_string()
        })
        .collect();
    assert_eq!(
        texts,
        vec!["pos one", "pos two", "neg one", "neg two"],
        "reqs: {:?}",
        reqs
    );
    // CR もコメントも生の本文に残っていない
    for r in &reqs {
        let body = String::from_utf8_lossy(&r.body);
        assert!(!body.contains('\r'), "body kept CR: {body}");
        assert!(!body.contains("# comment"), "body kept a comment: {body}");
    }
}

// ---------------------------------------------------------------------------
// 5. 出力ファイル (.dns / レシピ JSON) のスキーマと round-trip
// ---------------------------------------------------------------------------

#[test]
fn direction_and_recipe_files_have_schema_invariants_and_round_trip() {
    let mock = Mock::start(0);
    let fx = Fixture::new("schema");
    let pos = fx.file("pos.txt", "p one\np two\n");
    let neg = fx.file("neg.txt", "n one\n");
    let out_dir = fx.dir.join("out");
    let recipe = fx.dir.join("r1354.json");
    let mut args = base_args(&pos, &neg, &out_dir);
    args.extend([
        "--out-recipe".into(),
        recipe.display().to_string(),
        "--tag".into(),
        "t1354".into(),
        "--alpha".into(),
        "0.7".into(),
    ]);

    let out = run_abliterate(&mock.url(), &args);
    assert!(out.status.success(), "stderr: {}", stderr(&out));

    // --- 方向 (.dns) ---
    for (name, suffix) in [("t1354_pos.dns", "t1354_pos"), ("t1354_neg.dns", "t1354_neg")] {
        let path = out_dir.join(name);
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(!bytes.is_empty(), "{name} is empty");
        assert!(
            bytes.starts_with(b"DNSTEER1"),
            "{name}: bad magic: {:?}",
            &bytes[..bytes.len().min(9)]
        );
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.ends_with(suffix), "{name}: {text:?}");
        // 書き戻して読み直しても同一 (round-trip load)
        let copy = fx.dir.join(format!("copy-{name}"));
        std::fs::write(&copy, &bytes).unwrap();
        assert_eq!(std::fs::read(&copy).unwrap(), bytes, "{name}");
    }
    let pos_dns = std::fs::read(out_dir.join("t1354_pos.dns")).unwrap();
    let neg_dns = std::fs::read(out_dir.join("t1354_neg.dns")).unwrap();
    assert_ne!(pos_dns, neg_dns, "pos/neg directions must differ");

    // --- レポート (レシピ JSON) ---
    let text = std::fs::read_to_string(&recipe).expect("recipe file");
    assert!(text.ends_with('\n'), "recipe must end with a newline");
    assert!(!text.contains("NaN"), "recipe has NaN: {text}");
    assert!(!text.to_ascii_lowercase().contains("infinity"), "recipe has infinity: {text}");

    let v: serde_json::Value = serde_json::from_str(&text).expect("recipe parses");
    let obj = v.as_object().expect("recipe is an object");
    for key in [
        "format",
        "name",
        "description",
        "terms",
        "layers",
        "alpha",
        "mode",
        "embedded",
    ] {
        assert!(obj.contains_key(key), "recipe missing {key}: {text}");
    }
    assert_eq!(obj["format"], "denno-steer-recipe/1");
    assert_eq!(obj["name"], "t1354-recipe");
    assert_eq!(obj["mode"], "subtract");
    assert_eq!(obj["layers"], serde_json::json!([2, 5]));
    assert!(obj["embedded"].is_array(), "embedded must be an array");

    let alpha = obj["alpha"].as_f64().expect("alpha is a number");
    assert!(alpha.is_finite() && alpha > 0.0, "alpha: {alpha}");
    assert!((alpha - 0.7).abs() < 1e-6, "alpha: {alpha}");

    let terms = obj["terms"].as_array().expect("terms is an array");
    assert_eq!(terms.len(), 2, "terms: {text}");
    for t in terms {
        let coef = t["coef"].as_f64().expect("coef is a number");
        assert!(coef.is_finite(), "coef: {coef}");
        let snap = t["snapshot"].as_str().expect("snapshot is a string");
        assert!(!snap.is_empty(), "empty snapshot name");
    }
    assert_eq!(terms[0]["snapshot"], "t1354_pos");
    assert_eq!(terms[1]["snapshot"], "t1354_neg");
    assert!((terms[0]["coef"].as_f64().unwrap() - 1.0).abs() < 1e-6);
    assert!((terms[1]["coef"].as_f64().unwrap() + 1.0).abs() < 1e-6);

    // 読み直し → 書き戻し (pretty + 改行) が元のテキストと一致する
    let again: serde_json::Value =
        serde_json::from_str(&serde_json::to_string(&v).unwrap()).expect("reparse");
    assert_eq!(again, v);
    assert_eq!(
        format!("{}\n", serde_json::to_string_pretty(&v).unwrap()),
        text,
        "recipe writer must be idempotent"
    );
}

// ---------------------------------------------------------------------------
// 6. 出力前に拒否するもの
// ---------------------------------------------------------------------------

#[test]
fn invalid_capture_mode_is_rejected_upfront() {
    let mock = Mock::start(0);
    let fx = Fixture::new("capture");
    let pos = fx.file("pos.txt", "p\n");
    let neg = fx.file("neg.txt", "n\n");
    let out_dir = fx.dir.join("out");

    for bad in ["last", "ALL", "all ", ""] {
        let mut args = base_args(&pos, &neg, &out_dir);
        args.extend(["--capture-mode".into(), bad.into()]);
        let out = run_abliterate(&mock.url(), &args);
        assert!(!out.status.success(), "{bad:?}: unexpectedly succeeded");
        let se = stderr(&out);
        assert!(se.contains("capture_mode"), "{bad:?}: stderr: {se}");
        assert_no_panic(&out, bad);
    }
    assert!(
        mock.reqs().is_empty(),
        "requests: {:?}",
        mock.reqs()
    );
    assert!(!out_dir.exists(), "out_dir must not be created");
}

#[test]
fn invalid_layer_specification_is_rejected_before_writing_output() {
    // 1 = 非整数 [1.5, 3]、2 = 配列でない、3 = 負値 [-1, 3]
    for mode in [1u8, 2, 3] {
        let mock = Mock::start(mode);
        let fx = Fixture::new(&format!("layer{mode}"));
        let pos = fx.file("pos.txt", "p\n");
        let neg = fx.file("neg.txt", "n\n");
        let out_dir = fx.dir.join("out");
        let recipe = fx.dir.join("r.json");
        let mut args = base_args(&pos, &neg, &out_dir);
        args.extend([
            "--out-recipe".into(),
            recipe.display().to_string(),
            "--tag".into(),
            "t1354".into(),
        ]);

        let out = run_abliterate(&mock.url(), &args);
        assert!(
            !out.status.success(),
            "mode {mode}: unexpectedly succeeded: {}",
            stdout(&out)
        );
        let se = stderr(&out);
        assert!(
            se.contains("recommended_range"),
            "mode {mode}: stderr: {se}"
        );
        assert_no_panic(&out, &format!("layer{mode}"));

        let reqs = mock.reqs();
        assert!(
            !reqs.iter().any(|r| r.route() == "/v1/steer/export"),
            "mode {mode}: export ran: {reqs:?}"
        );
        assert!(
            !reqs.iter().any(|r| r.route() == "/v1/steer/apply"),
            "mode {mode}: apply ran: {reqs:?}"
        );
        assert!(
            !reqs.iter().any(|r| r.route() == "/v1/steer/recipe/export"),
            "mode {mode}: recipe export ran: {reqs:?}"
        );
        assert!(!out_dir.exists(), "mode {mode}: out_dir must not be created");
        assert!(!recipe.exists(), "mode {mode}: recipe must not be written");
    }
}
