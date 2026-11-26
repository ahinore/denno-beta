//! task 949: denno-view の HTTP 契約 (ルート / Content-Type / 静的アセット) の検査。
//!
//! denno-view はバイナリ crate なので Router を統合テストから直接組めない。
//! 代わりに本物のバイナリを空きポートで起動し、生の HTTP/1.1 を喋って応答を
//! そのまま見る (ヘッダ・ステータス・本文のバイト列)。`Connection: close` を
//! 送るので本文は EOF まで読めばよく、chunked の解釈は要らない。
//!
//! 上流 (denno サーバー) は存在しない前提。VIEW_UPSTREAM は閉じたポートを指し、
//! pump は 2 秒間隔で再接続を試みるだけなのでページ配信には影響しない。
//!
//! 見ているもの:
//!   1. /, /2d, /3d が 200 + text/html + 空でない本文
//!   2. 同梱 JS (.mjs と vendor の three.js / addons) が 200 + JavaScript MIME
//!   3. 未知ルートと path traversal が 4xx で返り、/etc/passwd を漏らさず、
//!      サーバーが panic しない (その後の / がまだ 200)
//!   4. 配信アセットに外部 CDN の URL が無い

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

    fn content_type(&self) -> &str {
        self.header("content-type").unwrap_or("")
    }

    fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
}

/// `raw_target` をそのままリクエスト行に載せる (reqwest 等の URL 正規化を避け、
/// `/../etc/passwd` のような生のパスをサーバーまで届ける)。
fn http_get(addr: &str, raw_target: &str) -> Resp {
    let mut s = TcpStream::connect(addr).expect("connect to denno-view");
    s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    s.set_write_timeout(Some(Duration::from_secs(10))).unwrap();
    let req = format!("GET {raw_target} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n");
    s.write_all(req.as_bytes()).expect("write request");
    s.flush().unwrap();
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).expect("read response");

    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .unwrap_or_else(|| panic!("no header terminator for {raw_target}"));
    let head = String::from_utf8_lossy(&raw[..split]).into_owned();
    let body = raw[split + 4..].to_vec();
    let mut lines = head.split("\r\n");
    let status_line = lines.next().unwrap_or_default();
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("bad status line for {raw_target}: {status_line:?}"));
    let headers = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
        .collect();
    Resp {
        status,
        headers,
        body,
    }
}

/// 空きポートを 1 つ確保する (bind → 解放)。競合の窓は小さいが、起動待ちで
/// 失敗したら呼び出し側がメッセージ付きで落ちる。
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
        http_get(&self.addr(), target)
    }
}

impl Drop for View {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// 本物のバイナリを立てて、TCP で繋がるまで待つ。
fn start_view() -> View {
    let port = free_port();
    let child = Command::new(env!("CARGO_BIN_EXE_denno-view"))
        .env("VIEW_PORT", port.to_string())
        // 上流は意図的に閉じたポート。ページ配信は上流に依存しない。
        .env("VIEW_UPSTREAM", "ws://127.0.0.1:1/v1/telemetry/ws")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn denno-view binary");

    let addr = format!("127.0.0.1:{port}");
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        if TcpStream::connect(&addr).is_ok() {
            return View { child, port };
        }
        if Instant::now() >= deadline {
            let mut c = child;
            let _ = c.kill();
            let _ = c.wait();
            panic!("denno-view did not listen on {addr} within 20s");
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// 同梱 JS モジュール (ルートで明示配信されるもの)。
const JS_MODULES: &[&str] = &[
    "/frames.mjs",
    "/view2d.mjs",
    "/arch2d.mjs",
    "/nodes2d.mjs",
    "/hud2d.mjs",
    "/dash.mjs",
    "/keys3d.mjs",
    "/reconnect.mjs",
];

/// /vendor/* 配下 (three.js 本体と addons)。
const VENDOR_FILES: &[&str] = &[
    "/vendor/three.module.js",
    "/vendor/addons/controls/OrbitControls.js",
    "/vendor/addons/postprocessing/EffectComposer.js",
    "/vendor/addons/postprocessing/Pass.js",
    "/vendor/addons/postprocessing/RenderPass.js",
    "/vendor/addons/postprocessing/ShaderPass.js",
    "/vendor/addons/postprocessing/MaskPass.js",
    "/vendor/addons/postprocessing/UnrealBloomPass.js",
    "/vendor/addons/postprocessing/OutputPass.js",
    "/vendor/addons/shaders/CopyShader.js",
    "/vendor/addons/shaders/LuminosityHighPassShader.js",
    "/vendor/addons/shaders/OutputShader.js",
];

#[test]
fn html_pages_return_200_html_and_non_empty_payload() {
    let v = start_view();
    for (path, marker) in [
        ("/", "<!DOCTYPE html>"),
        ("/2d", "<!DOCTYPE html>"),
        ("/3d", "<!DOCTYPE html>"),
    ] {
        let r = v.get(path);
        assert_eq!(r.status, 200, "GET {path} status");
        assert!(
            r.content_type().starts_with("text/html"),
            "GET {path} content-type: {:?}",
            r.content_type()
        );
        assert!(!r.body.is_empty(), "GET {path} body is empty");
        assert!(
            r.body.len() > 1000,
            "GET {path} body suspiciously small: {} bytes",
            r.body.len()
        );
        assert!(r.text().contains(marker), "GET {path} missing {marker:?}");
    }
}

#[test]
fn bundled_js_modules_return_200_with_javascript_mime() {
    let v = start_view();
    for path in JS_MODULES.iter().chain(VENDOR_FILES.iter()) {
        let r = v.get(path);
        assert_eq!(r.status, 200, "GET {path} status");
        let ct = r.content_type();
        assert!(
            ct.starts_with("text/javascript") || ct.starts_with("application/javascript"),
            "GET {path} content-type: {ct:?}"
        );
        assert!(!r.body.is_empty(), "GET {path} body is empty");
        assert!(
            r.text() != "not found",
            "GET {path} served the vendor 404 sentinel as 200"
        );
    }
}

#[test]
fn unknown_routes_and_path_traversal_are_clean_4xx_without_passwd_leak() {
    let v = start_view();
    // 生のリクエスト行で送る (URL 正規化をすり抜けさせない)。
    for target in [
        "/does-not-exist",
        "/nope/at/all",
        "/../etc/passwd",
        "/%2e%2e/%2e%2e/etc/passwd",
        "/vendor/../src/main.rs",
        "/vendor/..%2f..%2fetc%2fpasswd",
        "/vendor/nonexistent-addon.js",
    ] {
        let r = v.get(target);
        assert!(
            (400..500).contains(&r.status),
            "GET {target} expected 4xx, got {}",
            r.status
        );
        let body = r.text();
        assert!(
            !body.contains("root:x:"),
            "GET {target} leaked /etc/passwd content"
        );
        assert!(
            !body.contains("panicked at") && !body.contains("RUST_BACKTRACE"),
            "GET {target} returned a panic body: {body}"
        );
        assert!(
            body.len() < 4096,
            "GET {target} 4xx body unexpectedly large: {} bytes",
            body.len()
        );
    }

    // panic していればプロセスは落ちてここで接続できない。
    let r = v.get("/");
    assert_eq!(r.status, 200, "server did not survive traversal attempts");
}

#[test]
fn vendor_404_is_not_publicly_cacheable_but_200_is() {
    let v = start_view();
    let ok = v.get("/vendor/three.module.js");
    assert_eq!(ok.status, 200);
    assert!(
        ok.header("cache-control")
            .unwrap_or("")
            .contains("max-age="),
        "200 にはキャッシュ指示が要る: {:?}",
        ok.header("cache-control")
    );

    let miss = v.get("/vendor/nonexistent-addon.js");
    assert_eq!(miss.status, 404);
    let cc = miss.header("cache-control").unwrap_or("");
    assert!(
        !cc.contains("public") && !cc.contains("max-age"),
        "404 をブラウザにキャッシュさせない: {cc:?}"
    );
}

#[test]
fn served_assets_have_no_external_cdn_urls() {
    let v = start_view();
    let mut checked = 0usize;
    for path in ["/", "/2d", "/3d"]
        .iter()
        .chain(JS_MODULES.iter())
        .chain(VENDOR_FILES.iter())
    {
        let r = v.get(path);
        assert_eq!(r.status, 200, "GET {path} status");
        let body = r.text().to_ascii_lowercase();
        for needle in [
            "https://cdn",
            "//cdn.",
            "cdnjs.",
            "unpkg",
            "jsdelivr",
            "googleapis",
            "esm.sh",
        ] {
            assert!(
                !body.contains(needle),
                "GET {path} references external CDN token {needle:?}"
            );
        }
        checked += 1;
    }
    assert!(checked >= 15, "only checked {checked} assets");
}
