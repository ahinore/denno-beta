//! task 1209: denno-view の静的アセット配信に対する HTTP セキュリティ検査。
//!
//! denno-view はバイナリ crate なので Router を統合テストから直接は組めない。
//! `http_contract_949.rs` と同じ方針で、本物のバイナリを空きポートで起動し、
//! 生の HTTP/1.1 を喋って status / ヘッダ / 本文バイト列をそのまま見る。
//! `reqwest` などで URL を正規化すると `/../etc/passwd` が消えてしまうため、
//! リクエスト行には生のターゲットをそのまま載せる。
//!
//! 見ているもの:
//!   1. 未知ルートが 404 で、公開キャッシュ可能 (public / max-age) にならない
//!   2. path traversal (`../`, `%2e%2e`, `%2f`, バックスラッシュ, 四点) が
//!      4xx に落ち、`/etc/passwd` もリポジトリのソースも漏れない
//!   3. NUL バイト (`%00` / 生の 0x00) が 4xx になり、サーバーが落ちない
//!   4. 許可されないメソッドが 405 + `Allow` ヘッダで拒否される
//!   5. 4xx で `Cache-Control` を出す経路 (/vendor/*, /capture.bin) は no-store
//!
//! 上流 (denno サーバー) は存在しない前提。VIEW_UPSTREAM は閉じたポートを
//! 指すので pump は再接続を試みるだけで、静的配信には影響しない。

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// 生の HTTP 応答。ヘッダ名は小文字に落としてある。
struct Resp {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Resp {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }

    fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }

    /// `Allow` をカンマ区切りの大文字トークンへ分解する。
    fn allow(&self) -> Vec<String> {
        self.header("allow")
            .unwrap_or("")
            .split(',')
            .map(|s| s.trim().to_ascii_uppercase())
            .filter(|s| !s.is_empty())
            .collect()
    }
}

/// 生の応答バイト列を分解する。ヘッダ終端が無ければ None。
fn parse(raw: &[u8]) -> Option<Resp> {
    let split = raw.windows(4).position(|w| w == b"\r\n\r\n")?;
    let head = String::from_utf8_lossy(&raw[..split]).into_owned();
    let body = raw[split + 4..].to_vec();
    let mut lines = head.split("\r\n");
    let status: u16 = lines.next()?.split_whitespace().nth(1)?.parse().ok()?;
    let headers = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
        .collect();
    Some(Resp {
        status,
        headers,
        body,
    })
}

/// 応答を 1 本読み切る (`Connection: close` 前提で EOF まで読む)。
fn read_response(mut s: TcpStream, what: &str) -> Resp {
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).expect("read response");
    parse(&raw).unwrap_or_else(|| panic!("no header terminator for {what}"))
}

fn connect(addr: &str, what: &str) -> TcpStream {
    let s = TcpStream::connect(addr).expect("connect to denno-view");
    s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    s.set_write_timeout(Some(Duration::from_secs(10))).unwrap();
    let _ = what;
    s
}

/// 失敗しても panic しない 1 往復 (起動確認用)。
fn try_probe(addr: &str) -> Option<Resp> {
    let mut s = TcpStream::connect(addr).ok()?;
    s.set_read_timeout(Some(Duration::from_secs(5))).ok()?;
    s.set_write_timeout(Some(Duration::from_secs(5))).ok()?;
    let req = format!("GET / HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n");
    s.write_all(req.as_bytes()).ok()?;
    s.flush().ok()?;
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).ok()?;
    parse(&raw)
}

/// `method` と生の `raw_target` をそのままリクエスト行に載せる。
fn request(addr: &str, method: &str, raw_target: &str) -> Resp {
    let mut s = connect(addr, raw_target);
    let req = format!(
        "{method} {raw_target} HTTP/1.1\r\nHost: {addr}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
    );
    s.write_all(req.as_bytes()).expect("write request");
    s.flush().unwrap();
    read_response(s, raw_target)
}

/// 生バイト列をそのまま送る (生の NUL など、書式化できない要求用)。
fn request_raw(addr: &str, raw: &[u8], what: &str) -> Resp {
    let mut s = connect(addr, what);
    s.write_all(raw).expect("write raw request");
    s.flush().unwrap();
    read_response(s, what)
}

fn free_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    l.local_addr().unwrap().port()
}

/// 起動中の denno-view。Drop で子プロセスを落とす。
struct View {
    child: Child,
    port: u16,
}

impl View {
    fn addr(&self) -> String {
        format!("127.0.0.1:{}", self.port)
    }

    fn get(&self, target: &str) -> Resp {
        request(&self.addr(), "GET", target)
    }

    fn method(&self, method: &str, target: &str) -> Resp {
        request(&self.addr(), method, target)
    }
}

impl Drop for View {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// 本物のバイナリを立てて、`GET /` が実際に 200 を返すまで待つ。
///
/// `free_port()` は bind → 解放なので、起動までの間に同じポートが別テストの
/// サーバーに掴まれる競合がある。掴まれた側の子は bind に失敗して即終了するが
/// connect 自体は成功してしまうため、(a) 子が生きていること と (b) `/` が
/// 200 + HTML を返すこと の両方を確認してから採用する。駄目なら別ポートで再試行。
fn start_view() -> View {
    for _ in 0..8 {
        let port = free_port();
        let mut child = Command::new(env!("CARGO_BIN_EXE_denno-view"))
            .env("VIEW_PORT", port.to_string())
            // 上流は意図的に閉じたポート。ページ配信は上流に依存しない。
            .env("VIEW_UPSTREAM", "ws://127.0.0.1:1/v1/telemetry/ws")
            // 配信経路の分岐を固定する (テストランナーの環境変数に依存しない)。
            .env_remove("VIEW_CAPTURE")
            .env_remove("VIEW_REPLAY")
            .env_remove("VIEW_TOKENIZER")
            .env_remove("VIEW_RECORD")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn denno-view binary");

        let addr = format!("127.0.0.1:{port}");
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            // bind に失敗した子は即終了する。生きたまま応答する個体だけを採る。
            if matches!(child.try_wait(), Ok(Some(_))) {
                break;
            }
            if let Some(r) = try_probe(&addr) {
                if r.status == 200 && r.text().starts_with("<!DOCTYPE html>") {
                    return View { child, port };
                }
                break; // 別プロセスが応答している — このポートは捨てる
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = child.kill();
        let _ = child.wait();
    }
    panic!("denno-view did not serve / on any candidate port");
}

/// 4xx はブラウザや中間キャッシュに残してはいけない。denno-view の未知ルートは
/// axum の既定 fallback なので `Cache-Control` を出さない — ヘッダが無いこと
/// 自体は許容しつつ、`public` / `max-age` が付いていたら失敗にする。
/// ヘッダを出すなら `no-store` でなければならない。
fn assert_not_publicly_cacheable(r: &Resp, ctx: &str) {
    let cc = r.header("cache-control").unwrap_or("");
    let lower = cc.to_ascii_lowercase();
    assert!(
        !lower.contains("public"),
        "{ctx}: 4xx を公開キャッシュ可能にしてはいけない: {cc:?}"
    );
    assert!(
        !lower.contains("max-age"),
        "{ctx}: 4xx に max-age を付けてはいけない: {cc:?}"
    );
    if !cc.is_empty() {
        assert!(
            lower.contains("no-store"),
            "{ctx}: 4xx にキャッシュ指示を出すなら no-store のみ: {cc:?}"
        );
    }
}

/// 4xx 応答がファイル内容やリポジトリのソースを漏らしていないこと。
fn assert_no_leak(r: &Resp, ctx: &str) {
    let body = r.text();
    for needle in [
        "root:x:",
        "/bin/bash",
        "[package]",
        "name = \"denno",
        "fn main",
        "RUST_BACKTRACE",
        "panicked at",
    ] {
        assert!(
            !body.contains(needle),
            "{ctx}: 4xx 本文が {needle:?} を含む (情報漏洩/panic): {body:?}"
        );
    }
    assert!(
        body.len() < 4096,
        "{ctx}: 4xx 本文が大きすぎる: {} bytes",
        body.len()
    );
}

#[test]
fn unknown_routes_return_404_and_are_not_publicly_cacheable() {
    let v = start_view();
    for target in [
        "/does-not-exist",
        "/nope/at/all",
        "/index.php",
        "/static/app.js",
        "/frames.mjs.bak",
    ] {
        let r = v.get(target);
        assert_eq!(r.status, 404, "GET {target} は 404 のはず");
        assert_not_publicly_cacheable(&r, target);
        assert_no_leak(&r, target);
        // 未知ルートに「それらしい」本文を返さない (列挙の手掛かりにしない)。
        assert!(
            r.body.is_empty() || r.body.len() < 128,
            "GET {target} の 404 本文が想定より大きい: {} bytes",
            r.body.len()
        );
    }

    // HEAD も同じく 404 (本文なし)。
    let r = v.method("HEAD", "/does-not-exist");
    assert_eq!(r.status, 404, "HEAD 未知ルート");
    assert_not_publicly_cacheable(&r, "HEAD /does-not-exist");

    // 一連の 404 のあとも通常ページは配信され続ける。
    assert_eq!(v.get("/").status, 200, "404 群の後でサーバーが落ちた");
}

#[test]
fn path_traversal_attempts_are_4xx_and_never_leak_files() {
    let v = start_view();
    // 生のリクエスト行で送る。`..` / エンコード / バックスラッシュ / 四点の
    // 変種を混ぜ、いずれも 4xx に落ちて内容を返さないことを確かめる。
    for target in [
        "/../etc/passwd",
        "/../../../../etc/passwd",
        "/%2e%2e/%2e%2e/etc/passwd",
        "/%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "/..%2f..%2fetc%2fpasswd",
        "/vendor/../src/main.rs",
        "/vendor/../../etc/passwd",
        "/vendor/..%2f..%2fetc%2fpasswd",
        "/vendor/%2e%2e/%2e%2e/etc/passwd",
        "/vendor/%2e%2e/Cargo.toml",
        "/vendor/....//etc/passwd",
        "/vendor/..\\..\\etc\\passwd",
        "//etc/passwd",
        "/./../etc/passwd",
        "/vendor/three.module.js/../../../../etc/passwd",
    ] {
        let r = v.get(target);
        assert!(
            (400..500).contains(&r.status),
            "GET {target} は 4xx のはずが {}",
            r.status
        );
        assert_no_leak(&r, target);
    }

    // traversal を受けてもプロセスが生存していること (panic していない)。
    let r = v.get("/");
    assert_eq!(r.status, 200, "traversal 群の後でサーバーが落ちた");
    assert!(r.body.len() > 1000, "/ の本文が空になった");
}

#[test]
fn null_bytes_are_rejected_and_server_survives() {
    let v = start_view();

    // パーセントエンコードされた NUL はルート照合に一致せず 4xx。
    for target in [
        "/%00",
        "/fra%00mes.mjs",
        "/vendor/three.module.js%00",
        "/vendor/..%00/three.module.js",
        "/%2e%2e%00/etc/passwd",
    ] {
        let r = v.get(target);
        assert!(
            (400..500).contains(&r.status),
            "GET {target} は 4xx のはずが {}",
            r.status
        );
        assert_no_leak(&r, target);
        assert_not_publicly_cacheable(&r, target);
    }

    // 生の 0x00 をリクエスト行に載せた場合は HTTP パーサが 400 で弾く。
    let raw = {
        let addr = v.addr();
        format!("GET /fra\u{0}mes.mjs HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n")
    };
    let r = request_raw(&v.addr(), raw.as_bytes(), "raw NUL in request line");
    assert_eq!(r.status, 400, "生の NUL は 400 で拒否する");
    assert_no_leak(&r, "raw NUL");

    // 不正要求のあとも通常ページは配信され続ける。
    assert_eq!(v.get("/").status, 200, "NUL 群の後でサーバーが落ちた");
}

#[test]
fn unsupported_methods_are_rejected_with_405_and_allow_header() {
    let v = start_view();

    // 静的な .mjs は GET/HEAD のみ。POST は 405 + Allow: GET,HEAD。
    let r = v.method("POST", "/frames.mjs");
    assert_eq!(r.status, 405, "POST /frames.mjs");
    let allow = r.allow();
    assert!(allow.contains(&"GET".to_string()), "Allow: {allow:?}");
    assert!(allow.contains(&"HEAD".to_string()), "Allow: {allow:?}");
    assert!(!allow.contains(&"POST".to_string()), "Allow: {allow:?}");
    assert_no_leak(&r, "POST /frames.mjs");

    // ダッシュボードに対する PUT / OPTIONS も 405。
    for method in ["PUT", "OPTIONS", "DELETE"] {
        let r = v.method(method, "/");
        assert_eq!(r.status, 405, "{method} /");
        let allow = r.allow();
        assert!(allow.contains(&"GET".to_string()), "{method} / Allow: {allow:?}");
        assert!(
            !allow.contains(&method.to_string()),
            "{method} / Allow が自分を含む: {allow:?}"
        );
    }

    // /chat は GET と POST の二本立て。DELETE は 405 で、Allow に両方が載る。
    let r = v.method("DELETE", "/chat");
    assert_eq!(r.status, 405, "DELETE /chat");
    let allow = r.allow();
    assert!(allow.contains(&"GET".to_string()), "DELETE /chat Allow: {allow:?}");
    assert!(allow.contains(&"POST".to_string()), "DELETE /chat Allow: {allow:?}");
    assert!(!allow.contains(&"DELETE".to_string()), "DELETE /chat Allow: {allow:?}");

    // WebSocket ルートは GET 以外を受け付けない。
    let r = v.method("POST", "/ws");
    assert_eq!(r.status, 405, "POST /ws");

    // 405 を返したあとも GET / は生きている。
    assert_eq!(v.get("/").status, 200, "405 群の後でサーバーが落ちた");
}

/// 4xx で `Cache-Control` を明示する経路は no-store でなければならない。
/// (未知ルートはヘッダ無し。ここは実際にヘッダを出す 2 経路を固定する。)
#[test]
fn four_xx_paths_that_emit_cache_control_use_no_store() {
    let v = start_view();

    for (target, why) in [
        ("/vendor/nonexistent-addon.js", "vendor 未収録アセット"),
        ("/capture.bin", "capture 未設定"),
    ] {
        let r = v.get(target);
        assert_eq!(r.status, 404, "{why}: {target}");
        let cc = r.header("cache-control").unwrap_or("");
        assert!(
            cc.to_ascii_lowercase().contains("no-store"),
            "{why}: 404 は no-store を明示する: {cc:?}"
        );
        assert_not_publicly_cacheable(&r, target);
    }

    // 200 の vendor 配信は従来どおりキャッシュ可能 (回帰確認)。
    let ok = v.get("/vendor/three.module.js");
    assert_eq!(ok.status, 200, "vendor 200");
    let cc = ok.header("cache-control").unwrap_or("");
    assert!(
        cc.to_ascii_lowercase().contains("max-age="),
        "vendor 200 のキャッシュ指示が消えた: {cc:?}"
    );
}
