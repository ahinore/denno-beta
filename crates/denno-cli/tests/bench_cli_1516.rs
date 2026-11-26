//! task 1516: `denno bench --json` のレポート schema と引数検証の終了コードを
//! 実バイナリ + ローカルのモックサーバーで固定する (CPU のみ、モデル不要)。
//!
//! 見るもの:
//!
//! 1. `--json` が出すレポートの形。キー集合・型・不変条件 (md5 32 hex、
//!    determinism verified、error は null) を固定する。既存の Markdown 表示は
//!    変えない (--json 併用時も stdout に表が出る)。
//! 2. `--json` で書いたレポートが `--compare` でそのまま読めること (往復)。
//! 3. 引数検証の終了コード。clap の使い方エラーは 2、実行時エラーは 1。
//!    無効な組み合わせは黙って無視せず、ファイルを書かずに落ちること。
//!
//! ベンチの数値そのものは検証しない (モックの応答は固定で、時間は測れない)。

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

static SEQ: AtomicUsize = AtomicUsize::new(0);

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(tag: &str) -> Self {
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!(
            "denno1516_{}_{}_{}_{}",
            tag,
            std::process::id(),
            n,
            nanos
        ));
        std::fs::create_dir_all(&path).expect("create temp dir");
        TempDir { path }
    }

    fn join(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn denno(args: &[&str]) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_denno"));
    cmd.args(args);
    cmd.env_remove("NO_COLOR");
    cmd.env("NO_COLOR", "1");
    cmd.output().expect("spawn denno")
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).to_string()
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).to_string()
}

// ---------------------------------------------------------------------------
// モックサーバー (Anthropic Messages API + /v1/models)
// ---------------------------------------------------------------------------

struct MockServer {
    url: String,
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl MockServer {
    fn start(context_length: usize) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock server");
        listener.set_nonblocking(true).expect("nonblocking");
        let addr = listener.local_addr().expect("local addr");
        let stop = Arc::new(AtomicBool::new(false));
        let stop2 = stop.clone();
        let handle = std::thread::spawn(move || {
            while !stop2.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((sock, _)) => {
                        let _ = handle_conn(sock, context_length);
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    Err(_) => break,
                }
            }
        });
        MockServer {
            url: format!("http://{addr}"),
            stop,
            handle: Some(handle),
        }
    }
}

impl Drop for MockServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn read_request(sock: &mut TcpStream) -> (String, serde_json::Value) {
    sock.set_read_timeout(Some(Duration::from_secs(10))).ok();
    let mut buf = Vec::new();
    let mut tmp = [0u8; 1024];
    let header_end;
    loop {
        let n = sock.read(&mut tmp).expect("read request");
        assert!(n > 0, "connection closed before headers");
        buf.extend_from_slice(&tmp[..n]);
        if let Some(pos) = find_subslice(&buf, b"\r\n\r\n") {
            header_end = pos + 4;
            break;
        }
    }
    let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let mut content_length = 0usize;
    let mut path = String::new();
    for (i, line) in head.split("\r\n").enumerate() {
        if i == 0 {
            path = line.split(' ').nth(1).unwrap_or("/").to_string();
        }
        if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
            content_length = v.trim().parse().unwrap_or(0);
        }
    }
    while buf.len() < header_end + content_length {
        let n = sock.read(&mut tmp).expect("read body");
        assert!(n > 0, "connection closed before body");
        buf.extend_from_slice(&tmp[..n]);
    }
    let body = String::from_utf8_lossy(&buf[header_end..header_end + content_length]).to_string();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap_or(serde_json::Value::Null);
    (path, json)
}

fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

fn write_chunk(sock: &mut TcpStream, s: &str) -> std::io::Result<()> {
    write!(sock, "{:x}\r\n{}\r\n", s.len(), s)?;
    sock.flush()
}

/// プロンプトの「トークン数」= 空白区切りの語数。較正の検証には十分。
fn mock_tokens(body: &serde_json::Value) -> usize {
    let text = body["messages"]
        .get(0)
        .and_then(|m| m["content"].as_str())
        .unwrap_or_default();
    text.split_whitespace().count().max(1)
}

fn handle_conn(mut sock: TcpStream, context_length: usize) -> std::io::Result<()> {
    sock.set_nodelay(true).ok();
    let (path, body) = read_request(&mut sock);
    if path.starts_with("/v1/models") {
        let payload = serde_json::json!({
            "object": "list",
            "data": [{"id": "mock-model", "context_length": context_length}],
        })
        .to_string();
        write!(
            sock,
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            payload.len(),
            payload
        )?;
        return Ok(());
    }

    let stream = body["stream"].as_bool().unwrap_or(false);
    let max_tokens = body["max_tokens"].as_u64().unwrap_or(1) as usize;
    let in_tokens = mock_tokens(&body);
    let pieces = max_tokens.clamp(1, 16);

    if !stream {
        let payload = serde_json::json!({
            "type": "message", "role": "assistant", "model": "mock-model",
            "content": [{"type": "text", "text": "mock output"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": in_tokens, "output_tokens": pieces},
        })
        .to_string();
        write!(
            sock,
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            payload.len(),
            payload
        )?;
        return Ok(());
    }

    write!(
        sock,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
    )?;
    sock.flush()?;
    let start = serde_json::json!({
        "type": "message_start",
        "message": {"model": "mock-model", "usage": {"input_tokens": in_tokens, "output_tokens": 0}},
    });
    write_chunk(&mut sock, &format!("event: message_start\ndata: {start}\n\n"))?;
    write_chunk(
        &mut sock,
        &format!(
            "event: content_block_start\ndata: {}\n\n",
            serde_json::json!({"type": "content_block_start", "index": 0,
                "content_block": {"type": "text", "text": ""}})
        ),
    )?;
    for i in 0..pieces {
        let ev = serde_json::json!({
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": format!("mock{i} ")},
        });
        write_chunk(
            &mut sock,
            &format!("event: content_block_delta\ndata: {ev}\n\n"),
        )?;
    }
    write_chunk(
        &mut sock,
        &format!(
            "event: content_block_stop\ndata: {}\n\n",
            serde_json::json!({"type": "content_block_stop", "index": 0})
        ),
    )?;
    write_chunk(
        &mut sock,
        &format!(
            "event: message_delta\ndata: {}\n\n",
            serde_json::json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": pieces}})
        ),
    )?;
    write_chunk(
        &mut sock,
        &format!(
            "event: message_stop\ndata: {}\n\n",
            serde_json::json!({"type": "message_stop"})
        ),
    )?;
    write!(sock, "0\r\n\r\n")?;
    sock.flush()?;
    Ok(())
}

/// モック相手の最小ベンチ。`--seed 11` でプロンプト生成を固定する。
fn bench_args(url: &str, json_out: &Path) -> Vec<String> {
    vec![
        "bench".to_string(),
        "--url".to_string(),
        url.to_string(),
        "--api".to_string(),
        "anthropic".to_string(),
        "--pp".to_string(),
        "16".to_string(),
        "--tg".to_string(),
        "4".to_string(),
        "--depths".to_string(),
        "0".to_string(),
        "--rounds".to_string(),
        "2".to_string(),
        "--no-flush".to_string(),
        "--seed".to_string(),
        "11".to_string(),
        "--det-tokens".to_string(),
        "4".to_string(),
        "--reuse-depth".to_string(),
        "16".to_string(),
        "--timeout".to_string(),
        "20".to_string(),
        "--json".to_string(),
        json_out.to_string_lossy().into_owned(),
    ]
}

fn run_args(args: &[String]) -> Output {
    let refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    denno(&refs)
}

fn keys(v: &serde_json::Value) -> Vec<String> {
    let mut k: Vec<String> = v
        .as_object()
        .unwrap_or_else(|| panic!("not an object: {v}"))
        .keys()
        .cloned()
        .collect();
    k.sort();
    k
}

fn assert_keys(v: &serde_json::Value, expected: &[&str], tag: &str) {
    let mut exp: Vec<String> = expected.iter().map(|s| s.to_string()).collect();
    exp.sort();
    assert_eq!(keys(v), exp, "{tag}: key set changed");
}

fn assert_no_non_finite(raw: &str, tag: &str) {
    assert!(!raw.contains("NaN"), "{tag}: NaN in report");
    assert!(!raw.to_lowercase().contains("inf"), "{tag}: inf in report");
}

// ---------------------------------------------------------------------------
// 1. --json の schema
// ---------------------------------------------------------------------------

#[test]
fn json_report_schema_is_locked() {
    let server = MockServer::start(64);
    let td = TempDir::new("schema");
    let out = td.join("bench.json");
    let o = run_args(&bench_args(&server.url, &out));

    assert_eq!(o.status.code(), Some(0), "stderr:\n{}", stderr(&o));
    // 既定の Markdown 表示は --json 併用時も変わらない。
    let t = stdout(&o);
    assert!(t.contains("## denno bench"), "no markdown heading:\n{t}");
    assert!(t.contains("| pp16 | 0 | 16 |"), "no pp row:\n{t}");
    assert!(t.contains("| tg4 | 0 |"), "no tg row:\n{t}");
    // 書き出し先は stderr に 1 行で出る。
    let e = stderr(&o);
    assert!(e.contains("[bench] done"), "no bench progress line:\n{e}");
    assert!(
        e.contains(&format!("wrote {}", out.display())),
        "no wrote line:\n{e}"
    );

    let raw = std::fs::read_to_string(&out).expect("json report written");
    let v: serde_json::Value = serde_json::from_str(&raw).expect("valid JSON");
    assert_no_non_finite(&raw, "report");
    // 整形は pretty (2 スペース) で、1 行に潰れていない。
    assert!(raw.starts_with("{\n  \""), "not pretty-printed:\n{raw}");
    assert!(raw.contains("\n  \"pp\": [\n"), "pp not pretty-printed:\n{raw}");
    assert!(!raw.ends_with('\n'), "render_json writes no trailing newline");

    assert_keys(
        &v,
        &[
            "url",
            "api",
            "model",
            "context_length",
            "rounds",
            "seed",
            "created_unix",
            "pp",
            "tg",
            "turn2",
            "determinism",
            "errors",
        ],
        "report",
    );
    assert_eq!(v["url"], server.url);
    assert_eq!(v["api"], "anthropic");
    assert_eq!(v["model"], "mock-model");
    assert_eq!(v["context_length"], 64);
    assert_eq!(v["rounds"], 2);
    assert_eq!(v["seed"], 11);
    assert!(v["created_unix"].as_u64().unwrap_or(0) > 0);
    assert_eq!(v["errors"], serde_json::json!([]));

    let pp = &v["pp"][0];
    assert_keys(
        pp,
        &[
            "pp_tokens",
            "depth",
            "total_tokens",
            "ttft_ms",
            "pp_tok_s",
            "ttft_runs_ms",
            "error",
        ],
        "pp row",
    );
    assert_eq!(pp["pp_tokens"], 16);
    assert_eq!(pp["depth"], 0);
    assert_eq!(pp["total_tokens"], 16);
    assert!(pp["error"].is_null(), "pp error: {pp}");
    assert!(pp["ttft_ms"].as_f64().unwrap_or(0.0) >= 0.0);
    assert!(pp["pp_tok_s"].as_f64().unwrap_or(0.0) >= 0.0);
    assert_eq!(pp["ttft_runs_ms"].as_array().map(|a| a.len()), Some(2));

    let tg = &v["tg"][0];
    assert_keys(
        tg,
        &[
            "depth",
            "gen_tokens",
            "prompt_tokens",
            "ttft_ms",
            "tpot_ms",
            "itl_ms",
            "tg_tok_s",
            "runs_tok_s",
            "error",
        ],
        "tg row",
    );
    assert_eq!(tg["depth"], 0);
    assert_eq!(tg["gen_tokens"], 4);
    assert!(tg["error"].is_null(), "tg error: {tg}");
    assert!(tg["tg_tok_s"].as_f64().unwrap_or(0.0) > 0.0);
    assert_eq!(tg["runs_tok_s"].as_array().map(|a| a.len()), Some(2));
    // 3 intervals/round x 2 rounds, 1 TPOT/round x 2 rounds.
    assert_eq!(tg["itl_ms"]["n"], 6);
    assert_eq!(tg["tpot_ms"]["n"], 2);
    for stats in [&tg["tpot_ms"], &tg["itl_ms"]] {
        assert_keys(
            stats,
            &["n", "mean", "median", "min", "max", "stddev", "p90", "p99"],
            "stats",
        );
    }

    let t2 = &v["turn2"];
    assert_keys(
        t2,
        &["depth", "cold_ttft_ms", "reuse_ttft_ms", "speedup", "error"],
        "turn2",
    );
    assert_eq!(t2["depth"], 16);
    assert!(t2["error"].is_null(), "turn2 error: {t2}");
    assert!(t2["cold_ttft_ms"].as_f64().unwrap_or(0.0) > 0.0);
    assert!(t2["reuse_ttft_ms"].as_f64().unwrap_or(0.0) > 0.0);
    assert!(t2["speedup"].as_f64().unwrap_or(0.0) > 0.0);

    let det = &v["determinism"];
    assert_keys(
        det,
        &["tokens", "gen_tokens", "md5", "verified", "round_md5", "error"],
        "determinism",
    );
    assert_eq!(det["tokens"], 256);
    assert_eq!(det["gen_tokens"], 4);
    assert_eq!(det["verified"], true);
    assert!(det["error"].is_null(), "determinism error: {det}");
    let md5 = det["md5"].as_str().expect("md5 string");
    assert_eq!(md5.len(), 32, "md5 not 32 hex: {md5}");
    assert!(
        md5.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
        "md5 not lowercase hex: {md5}"
    );
    let rounds = det["round_md5"].as_array().expect("round_md5 array");
    assert_eq!(rounds.len(), 2);
    assert!(rounds.iter().all(|h| h == &det["md5"]));
}

#[test]
fn json_report_roundtrips_through_compare() {
    let server = MockServer::start(64);
    let td = TempDir::new("compare");
    let a = td.join("a.json");
    let b = td.join("b.json");
    for out in [&a, &b] {
        let o = run_args(&bench_args(&server.url, out));
        assert_eq!(o.status.code(), Some(0), "stderr:\n{}", stderr(&o));
    }

    let o = denno(&[
        "bench",
        "--compare",
        &a.to_string_lossy(),
        &b.to_string_lossy(),
    ]);
    assert_eq!(
        o.status.code(),
        Some(0),
        "compare rejected own --json output\nstdout:\n{}\nstderr:\n{}",
        stdout(&o),
        stderr(&o)
    );
    let t = stdout(&o);
    assert!(t.contains("### denno bench compare"), "no compare table:\n{t}");
    assert!(
        t.contains("determinism md5 MATCH"),
        "determinism not matched:\n{t}"
    );
    assert!(t.contains("regressions: 0"), "unexpected regressions:\n{t}");
}

// ---------------------------------------------------------------------------
// 2. 引数検証と終了コード
// ---------------------------------------------------------------------------

#[test]
fn unknown_api_is_a_runtime_error_without_server() {
    let o = denno(&["bench", "--api", "grpc"]);
    assert_eq!(o.status.code(), Some(1), "stderr:\n{}", stderr(&o));
    assert!(stdout(&o).trim().is_empty(), "no output on validation error");
    let e = stderr(&o);
    assert!(e.contains("Error:"), "no Error line:\n{e}");
    assert!(e.contains("--api: unknown value \"grpc\""), "wrong message:\n{e}");
    assert!(!e.to_lowercase().contains("preflight"), "must fail before I/O:\n{e}");
}

#[test]
fn rounds_zero_fails_before_any_request() {
    let o = denno(&["bench", "--rounds", "0", "--url", "http://127.0.0.1:1"]);
    assert_eq!(o.status.code(), Some(1), "stderr:\n{}", stderr(&o));
    let e = stderr(&o);
    assert!(e.contains("--rounds must be >= 1"), "wrong message:\n{e}");
    assert!(!e.to_lowercase().contains("preflight"), "must fail before I/O:\n{e}");
}

#[test]
fn json_with_compare_is_rejected_and_writes_nothing() {
    let td = TempDir::new("jsoncmp");
    let a = td.join("a.json");
    let b = td.join("b.json");
    std::fs::write(&a, "{}").unwrap();
    std::fs::write(&b, "{}").unwrap();
    let out = td.join("unused.json");
    let o = denno(&[
        "bench",
        "--json",
        &out.to_string_lossy(),
        "--compare",
        &a.to_string_lossy(),
        &b.to_string_lossy(),
    ]);
    assert_eq!(o.status.code(), Some(1), "stderr:\n{}", stderr(&o));
    let e = stderr(&o);
    assert!(
        e.contains("--json is not used with --compare"),
        "wrong message:\n{e}"
    );
    assert!(e.contains("--compare-json"), "no pointer to --compare-json:\n{e}");
    assert!(!out.exists(), "no file may be written on rejection");
}

#[test]
fn fail_on_regress_pct_requires_compare() {
    let o = denno(&["bench", "--fail-on-regress-pct", "5", "--url", "http://127.0.0.1:1"]);
    assert_eq!(o.status.code(), Some(1), "stderr:\n{}", stderr(&o));
    let e = stderr(&o);
    assert!(
        e.contains("--fail-on-regress-pct requires --compare"),
        "wrong message:\n{e}"
    );
    assert!(!e.to_lowercase().contains("preflight"), "must fail before I/O:\n{e}");
}

#[test]
fn unreachable_server_exits_one_and_writes_no_json() {
    let td = TempDir::new("noconn");
    let out = td.join("bench.json");
    let o = denno(&[
        "bench",
        "--url",
        "http://127.0.0.1:1",
        "--pp",
        "16",
        "--depths",
        "0",
        "--rounds",
        "1",
        "--no-flush",
        "--timeout",
        "5",
        "--json",
        &out.to_string_lossy(),
    ]);
    assert_eq!(o.status.code(), Some(1), "stderr:\n{}", stderr(&o));
    let e = stderr(&o);
    assert!(
        e.contains("preflight request to http://127.0.0.1:1 failed"),
        "wrong preflight message:\n{e}"
    );
    assert!(!out.exists(), "no report on a failed preflight");
}

#[test]
fn unwritable_json_path_fails_after_the_run() {
    let server = MockServer::start(64);
    let td = TempDir::new("badpath");
    let out = td.join("missing-dir").join("bench.json");
    let o = run_args(&bench_args(&server.url, &out));
    assert_eq!(o.status.code(), Some(1), "stderr:\n{}", stderr(&o));
    let e = stderr(&o);
    assert!(e.contains("Error:"), "no Error line:\n{e}");
    assert!(
        e.contains(&format!("write {}", out.display())),
        "path not named in write error:\n{e}"
    );
    assert!(!out.exists(), "no file may exist");
}

#[test]
fn clap_usage_errors_exit_two() {
    // --json の値なし
    let o = denno(&["bench", "--json"]);
    assert_eq!(o.status.code(), Some(2), "stderr:\n{}", stderr(&o));
    assert!(stderr(&o).contains("error:"), "no clap error:\n{}", stderr(&o));

    // --compare は 2 パス必須
    let o = denno(&["bench", "--compare", "only.json"]);
    assert_eq!(o.status.code(), Some(2), "stderr:\n{}", stderr(&o));
    assert!(stderr(&o).contains("error:"), "no clap error:\n{}", stderr(&o));

    // 未知のフラグ
    let o = denno(&["bench", "--json-report", "x.json"]);
    assert_eq!(o.status.code(), Some(2), "stderr:\n{}", stderr(&o));
    assert!(stderr(&o).contains("error:"), "no clap error:\n{}", stderr(&o));
}
