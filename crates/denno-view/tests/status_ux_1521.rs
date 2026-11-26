//! task 1521: denno-view の health/status エンドポイントと、埋め込み HTML/JS が
//! 「切断」と「stale (WS は開いているがフレームが来ない)」をアクセシブルに
//! 表示するためのフックを固定する。
//!
//! denno-view はバイナリ crate なので Router を統合テストから直接は組めない。
//! `assets_audit_1309.rs` / `http_security_1209.rs` と同じ方針で、本物のバイナリを
//! 空きポートで起動し、生の HTTP/1.1 を喋って応答をそのまま見る。
//!
//! 上流 (denno サーバー) は存在しない前提。VIEW_UPSTREAM は閉じたポートを指すので
//! pump は接続に失敗し続け、/status は upstream.connected=false / stale=true を返す。
//!
//! 見ているもの:
//!   1. GET /status が 200 + JSON で、schema / service / upstream / frames / bytes を
//!      返す。上流未接続なら connected=false, last_frame_age_ms=null, stale=true。
//!   2. GET /healthz が 200 + JSON の最小ヘルス応答。どちらも no-store。
//!   3. `/`, `/2d`, `/3d` の接続帯 (#connbanner) と stale 帯 (#stalestat) が
//!      role="status" / aria-live="polite" を持ち、/status を fetch して
//!      staleLabel() で文言を作る (console に何も出さない)。
//!   4. /reconnect.mjs が staleLabel / isStaleStatus / STALE_AFTER_MS を export する。
//!   5. 配信ページと reconnect.mjs に LAN IP / /home/ パス / デバッグ出力 / 外部 CDN が
//!      無い (SVG 名前空間の http://www.w3.org は対象外)。

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

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

    fn json(&self) -> serde_json::Value {
        serde_json::from_slice(&self.body)
            .unwrap_or_else(|e| panic!("本文が JSON でない: {e}: {}", self.text()))
    }
}

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

fn free_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    l.local_addr().unwrap().port()
}

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

/// 本物のバイナリを立てて `/` が 200 + HTML を返すまで待つ (`assets_audit_1309.rs`
/// と同じ、ポート競合に備えた数回のリトライ付き)。
fn start_view() -> View {
    for _ in 0..8 {
        let port = free_port();
        let mut child = Command::new(env!("CARGO_BIN_EXE_denno-view"))
            .env("VIEW_PORT", port.to_string())
            .env("VIEW_UPSTREAM", "ws://127.0.0.1:1/v1/telemetry/ws")
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
            if matches!(child.try_wait(), Ok(Some(_))) {
                break;
            }
            if TcpStream::connect(&addr).is_ok() {
                let r = http_get(&addr, "/");
                if r.status == 200 && r.text().starts_with("<!DOCTYPE html>") {
                    return View { child, port };
                }
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = child.kill();
        let _ = child.wait();
    }
    panic!("denno-view did not serve / on any candidate port");
}

const PAGES: &[&str] = &["/", "/2d", "/3d"];

/// `id="<id>"` を持つ開始タグ 1 個を切り出す (属性検査用の素朴な走査)。
fn tag_with_id<'a>(html: &'a str, id: &str) -> Option<&'a str> {
    let needle = format!("id=\"{id}\"");
    let at = html.find(&needle)?;
    let start = html[..at].rfind('<')?;
    let end = at + html[at..].find('>')?;
    Some(&html[start..=end])
}

#[test]
fn status_endpoint_reports_health_and_upstream_without_upstream() {
    let v = start_view();
    let r = v.get("/status");
    assert_eq!(r.status, 200, "GET /status");
    let ct = r.content_type().to_ascii_lowercase();
    assert!(
        ct.starts_with("application/json") && ct.contains("charset=utf-8"),
        "GET /status content-type: {:?}",
        r.content_type()
    );
    assert_eq!(
        r.header("cache-control").map(|s| s.to_ascii_lowercase()),
        Some("no-store".to_string()),
        "/status は no-store"
    );

    let j = r.json();
    assert_eq!(j["schema"], 1, "/status schema");
    assert_eq!(j["service"], "denno-view", "/status service");
    assert_eq!(j["status"], "ok", "/status status");
    assert_eq!(j["ok"], true, "/status ok");
    assert!(
        j["frames"].as_u64().is_some(),
        "/status frames が数値: {:?}",
        j["frames"]
    );
    assert!(
        j["bytes"].as_u64().is_some(),
        "/status bytes が数値: {:?}",
        j["bytes"]
    );

    let up = &j["upstream"];
    assert_eq!(
        up["connected"], false,
        "上流ポートは閉じているので connected=false: {up:?}"
    );
    assert_eq!(
        up["stale"], true,
        "フレームが 1 枚も来ていないので stale=true: {up:?}"
    );
    assert!(
        up["last_frame_age_ms"].is_null(),
        "未受信なら last_frame_age_ms=null: {up:?}"
    );
    assert!(
        up["stale_after_ms"].as_u64().is_some_and(|ms| ms > 0),
        "stale_after_ms が正の数: {up:?}"
    );

    let h = v.get("/healthz");
    assert_eq!(h.status, 200, "GET /healthz");
    let hj = h.json();
    assert_eq!(hj["ok"], true, "/healthz ok");
    assert_eq!(hj["service"], "denno-view", "/healthz service");
    assert_eq!(
        h.header("cache-control").map(|s| s.to_ascii_lowercase()),
        Some("no-store".to_string()),
        "/healthz は no-store"
    );
}

#[test]
fn pages_expose_accessible_connection_and_stale_status_hooks() {
    let v = start_view();
    for path in PAGES {
        let r = v.get(path);
        assert_eq!(r.status, 200, "GET {path}");
        let body = std::str::from_utf8(&r.body)
            .unwrap_or_else(|e| panic!("{path}: 本文が valid UTF-8 でない: {e}"));

        for id in ["connbanner", "stalestat"] {
            let tag = tag_with_id(body, id)
                .unwrap_or_else(|| panic!("GET {path}: id={id} の要素が無い"));
            assert!(
                tag.contains("role=\"status\""),
                "GET {path}: #{id} は role=status を持つ: {tag}"
            );
            assert!(
                tag.contains("aria-live=\"polite\""),
                "GET {path}: #{id} は aria-live=polite を持つ: {tag}"
            );
            assert!(
                tag.contains("aria-atomic=\"true\""),
                "GET {path}: #{id} は aria-atomic=true を持つ: {tag}"
            );
        }

        // /status を実際に読み、staleLabel で文言を作る (取得失敗は catch して
        // console に出さない)。
        assert!(
            body.contains("fetch('/status'") || body.contains("fetch(\"/status\""),
            "GET {path}: /status を fetch していない"
        );
        assert!(
            body.contains("staleLabel("),
            "GET {path}: staleLabel() を使っていない"
        );
        assert!(
            !body.contains("console.log(") && !body.contains("console.error(\"status"),
            "GET {path}: status ポーリングがデバッグ出力を含む"
        );
    }

    let js = v.get("/reconnect.mjs");
    assert_eq!(js.status, 200, "GET /reconnect.mjs");
    let src = js.text();
    for needle in [
        "export function staleLabel",
        "export function isStaleStatus",
        "export const STALE_AFTER_MS",
        "telemetry stale",
    ] {
        assert!(src.contains(needle), "/reconnect.mjs に {needle:?} が無い");
    }
}

#[test]
fn served_status_assets_have_no_private_hosts_or_debug_prints() {
    fn has_lan_ip(s: &str) -> bool {
        for prefix in ["192.168.", "172.16.", "172.17.", "172.18.", "10."] {
            if let Some(pos) = s.find(prefix) {
                // "10." は CSS の 10.5px のような数値も拾うので、直後が数字で、
                // さらに次のオクテットの点があるものだけ IP とみなす。
                let before = s[..pos].chars().next_back();
                let ok_before = before.map_or(true, |c| !c.is_ascii_digit());
                let rest = &s[pos + prefix.len()..];
                let next_is_digit = rest.chars().next().is_some_and(|c| c.is_ascii_digit());
                let next_octet = rest.find('.').is_some_and(|d| d <= 3);
                if ok_before && next_is_digit && next_octet {
                    return true;
                }
            }
        }
        false
    }

    let v = start_view();
    let mut checked = 0usize;
    for path in PAGES.iter().copied().chain(["/reconnect.mjs", "/status", "/healthz"]) {
        let r = v.get(path);
        assert_eq!(r.status, 200, "GET {path}");
        let body = r.text();
        assert!(
            !body.contains("/home/"),
            "GET {path}: 絶対パス (/home/) を含む"
        );
        assert!(
            !body.contains("mailto:") && !body.contains("@localhost"),
            "GET {path}: 連絡先/ホスト名らしき文字列を含む"
        );
        assert!(!has_lan_ip(&body), "GET {path}: LAN IP を含む");
        for dbg in ["eprintln!", "dbg!", "debugger", "console.log("] {
            assert!(
                !body.contains(dbg),
                "GET {path}: デバッグ出力 {dbg:?} を含む"
            );
        }
        checked += 1;
    }
    assert!(checked >= 6, "only checked {checked} responses");
}
