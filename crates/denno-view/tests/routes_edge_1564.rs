//! task 1564: denno-view のルート表・Content-Type・404 / メソッド周縁の境界検査。
//!
//! denno-view はバイナリ crate なので Router を統合テストから直接は組めない。
//! `http_contract_949.rs` / `http_security_1209.rs` と同じ方針で、本物のバイナリを
//! 空きポートで起動し、生の HTTP/1.1 を喋って status / ヘッダ / 本文バイト列を
//! そのまま見る (URL 正規化に頼らない)。
//!
//! 見ているもの:
//!   1. 現行ルート (`/`, `/2d`, `/3d`, `/chat`) が 200 + `text/html; charset=utf-8`
//!      で、クエリ文字列を付けても本文が変わらない (同梱 HTML の静的配信)
//!   2. 同梱 .mjs と vendor が正確に `text/javascript; charset=utf-8` を返し、
//!      クエリ文字列で本文・種別が変わらない
//!   3. 近いが未定義のパス (`/dashboard`, `/2d/`, 大文字違い, `index.html` 等) が
//!      空本文の 404 になり、公開キャッシュ指示を付けない
//!   4. `/api/*` は未知パスでも 404 ではなく上流中継 (上流不在時は 502 JSON) —
//!      「未知ルート = 404」の境界が /api 接頭辞で切れていることを固定する
//!   5. メソッド周縁: HEAD は GET と同じ Content-Type / Content-Length で本文なし、
//!      ページへの POST/PUT/DELETE/OPTIONS は 405 + Allow、未知パスは
//!      メソッドに依らず 404 (405 より fallback が優先)
//!   6. HTML / JS 応答の src・href・ES import に外部 URL が 1 つも無い (ゼロ CDN)
//!
//! 上流 (denno サーバー) は存在しない前提。VIEW_UPSTREAM は閉じたポートを指す。

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

/// `method` と生の `raw_target` をそのままリクエスト行に載せる。
fn request(addr: &str, method: &str, raw_target: &str) -> Resp {
    let mut s = TcpStream::connect(addr).expect("connect to denno-view");
    s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    s.set_write_timeout(Some(Duration::from_secs(10))).unwrap();
    let req = format!(
        "{method} {raw_target} HTTP/1.1\r\nHost: {addr}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
    );
    s.write_all(req.as_bytes()).expect("write request");
    s.flush().unwrap();
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).expect("read response");
    parse(&raw).unwrap_or_else(|| panic!("no header terminator for {method} {raw_target}"))
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

/// 本物のバイナリを立て、`GET /` が 200 + HTML を返すまで待つ。
/// `free_port()` は bind → 解放なので競合の窓がある。子が生きていて `/` を
/// 実際に配信している個体だけを採用し、駄目なら別ポートで再試行する。
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
            if matches!(child.try_wait(), Ok(Some(_))) {
                break; // bind 失敗で即終了した個体
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

/// 現行の HTML ページ (ルート表の正)。`/dashboard` というパスは存在せず
/// `/` がダッシュボード (近い名前の 404 は別テストで固定する)。
const HTML_PAGES: &[&str] = &["/", "/2d", "/3d", "/chat"];

/// ルートで明示配信される JS モジュール。
const JS_MODULES: &[&str] = &[
    "/frames.mjs",
    "/view2d.mjs",
    "/arch2d.mjs",
    "/nodes2d.mjs",
    "/hud2d.mjs",
    "/dash.mjs",
    "/keys3d.mjs",
    "/chat.mjs",
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

/// `attr="..."` / `attr='...'` の値を取り出す。直前が空白か `<` のときだけ
/// 属性名とみなす (`data-src=` などを拾わない)。完全なパーサではないが、
/// 参照の取りこぼしより誤検出を避ける設計 (1309 と同じ)。
fn attr_values<'a>(html: &'a str, attr: &str) -> Vec<&'a str> {
    let mut out = Vec::new();
    let mut cursor = 0usize;
    while let Some(pos) = html[cursor..].find(attr) {
        let start = cursor + pos;
        cursor = start + attr.len();
        let boundary = html[..start]
            .chars()
            .next_back()
            .map_or(true, |c| c.is_whitespace() || c == '<');
        if !boundary {
            continue;
        }
        let mut j = cursor;
        while let Some(c) = html[j..].chars().next() {
            if c.is_whitespace() {
                j += c.len_utf8();
            } else {
                break;
            }
        }
        if !html[j..].starts_with('=') {
            continue;
        }
        j += 1;
        while let Some(c) = html[j..].chars().next() {
            if c.is_whitespace() {
                j += c.len_utf8();
            } else {
                break;
            }
        }
        let quote = match html[j..].chars().next() {
            Some(q @ ('"' | '\'')) => q,
            _ => continue,
        };
        j += quote.len_utf8();
        let rest = &html[j..];
        if let Some(end) = rest.find(quote) {
            out.push(&rest[..end]);
            cursor = j + end + quote.len_utf8();
        }
    }
    out
}

/// `from "..."` / `from '...'` / `import("...")` のモジュール指定子を集める。
fn import_specifiers(src: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut push_quoted = |from: usize| {
        if let Some(q @ ('"' | '\'')) = src[from..].chars().next() {
            let rest = &src[from + q.len_utf8()..];
            if let Some(end) = rest.find(q) {
                out.push(rest[..end].to_string());
            }
        }
    };
    for kw in ["from", "import"] {
        let mut cursor = 0usize;
        while let Some(pos) = src[cursor..].find(kw) {
            let start = cursor + pos;
            cursor = start + kw.len();
            let before = src[..start].chars().next_back();
            let ok_before = before.map_or(true, |c| {
                c.is_whitespace() || c == '}' || c == ')' || c == ';' || c == '{'
            });
            if !ok_before {
                continue;
            }
            let mut j = cursor;
            while let Some(c) = src[j..].chars().next() {
                if c.is_whitespace() {
                    j += c.len_utf8();
                } else {
                    break;
                }
            }
            if kw == "import" && !src[j..].starts_with('(') {
                continue; // `import x from` は "from" 側で拾う
            }
            if kw == "import" {
                j += 1;
                while let Some(c) = src[j..].chars().next() {
                    if c.is_whitespace() {
                        j += c.len_utf8();
                    } else {
                        break;
                    }
                }
            }
            push_quoted(j);
        }
    }
    out
}

/// ネットワーク越しの外部参照か (http/https/プロトコル相対)。
fn is_external_network(value: &str) -> bool {
    let v = value.trim();
    v.starts_with("http://") || v.starts_with("https://") || v.starts_with("//")
}

/// 4xx は公開キャッシュ不可 (ヘッダを出すなら no-store のみ)。
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

/// 4xx 本文がファイル内容・リポジトリのソース・panic を漏らしていないこと。
fn assert_no_leak(r: &Resp, ctx: &str) {
    let body = r.text();
    for needle in [
        "root:x:",
        "/bin/bash",
        "[package]",
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
fn html_routes_serve_html_with_charset_and_ignore_query() {
    let v = start_view();

    for path in HTML_PAGES {
        let r = v.get(path);
        assert_eq!(r.status, 200, "GET {path} status");
        let ct = r.content_type().to_ascii_lowercase();
        assert!(
            ct.starts_with("text/html"),
            "GET {path} content-type: {:?}",
            r.content_type()
        );
        assert!(
            ct.contains("charset=utf-8"),
            "GET {path} は charset=utf-8 を宣言する: {:?}",
            r.content_type()
        );
        let body = r.text();
        assert!(
            body.to_ascii_lowercase().contains("<!doctype html>"),
            "GET {path} に DOCTYPE 宣言が無い"
        );
        assert!(
            body.len() > 1000,
            "GET {path} 本文が小さすぎる: {} bytes",
            body.len()
        );
    }

    // クエリ文字列は本文に影響しない (3 画面の ?replay=1 などは JS 側で解釈)。
    for (bare, with_query) in [
        ("/", "/?x=1&y=2"),
        ("/2d", "/2d?focus=layer3"),
        ("/3d", "/3d?replay=1"),
        ("/chat", "/chat?preset=short"),
    ] {
        let a = v.get(bare);
        let b = v.get(with_query);
        assert_eq!(a.status, 200, "GET {bare} status");
        assert_eq!(b.status, 200, "GET {with_query} status");
        assert_eq!(
            a.body, b.body,
            "GET {with_query} の本文が GET {bare} と異なる (クエリで配信内容が変わっている)"
        );
        assert_eq!(
            a.content_type().to_ascii_lowercase(),
            b.content_type().to_ascii_lowercase(),
            "GET {with_query} の Content-Type が GET {bare} と異なる"
        );
    }
}

#[test]
fn javascript_responses_use_exact_javascript_content_type() {
    let v = start_view();

    for path in JS_MODULES.iter().chain(VENDOR_FILES.iter()) {
        let r = v.get(path);
        assert_eq!(r.status, 200, "GET {path} status");
        assert_eq!(
            r.content_type().to_ascii_lowercase(),
            "text/javascript; charset=utf-8",
            "GET {path} の Content-Type は text/javascript; charset=utf-8 に固定する"
        );
        assert!(!r.body.is_empty(), "GET {path} 本文が空");
    }

    // クエリ付きでも種別・本文は同じ (キャッシュバスターを許す)。
    for path in ["/frames.mjs", "/dash.mjs", "/vendor/three.module.js"] {
        let a = v.get(path);
        let b = v.get(&format!("{path}?v=1564"));
        assert_eq!(a.status, 200, "GET {path} status");
        assert_eq!(b.status, 200, "GET {path}?v=1564 status");
        assert_eq!(
            a.content_type().to_ascii_lowercase(),
            b.content_type().to_ascii_lowercase(),
            "GET {path}?v=1564 の Content-Type が変わった"
        );
        assert_eq!(
            a.body, b.body,
            "GET {path}?v=1564 の本文が GET {path} と異なる"
        );
    }
}

#[test]
fn near_miss_routes_are_404_with_empty_body() {
    let v = start_view();

    // `/dashboard` は未定義 (ダッシュボードは `/`)。末尾スラッシュ・大文字違い・
    // 実ファイル名・拡張子付きのどれもルート表に一致しない。
    for target in [
        "/dashboard",
        "/dash",
        "/dashboard/",
        "/2d/",
        "/3d/",
        "/chat/",
        "/2D",
        "/3D",
        "/3D?x=1",
        "/index.html",
        "/2d/index.html",
        "/3d.html",
        "/frames.mjs.bak",
        "/frames.mjs/",
        "/vendor/",
        "/vendor",
    ] {
        let r = v.get(target);
        assert_eq!(r.status, 404, "GET {target} は 404 のはず");
        assert!(
            r.body.is_empty(),
            "GET {target} の 404 本文は空にする (列挙の手掛かりにしない): {} bytes",
            r.body.len()
        );
        assert!(
            !r.content_type().to_ascii_lowercase().starts_with("text/html"),
            "GET {target} の 404 に HTML を返さない: {:?}",
            r.content_type()
        );
        assert_not_publicly_cacheable(&r, target);
        assert_no_leak(&r, target);
    }

    // 404 群のあとも `/` は配信され続ける。
    assert_eq!(v.get("/").status, 200, "404 群の後でサーバーが落ちた");
}

#[test]
fn api_prefix_is_proxied_instead_of_404() {
    let v = start_view();

    // 通常の未知パスは 404。一方 `/api/*` は any() で上流へ中継するので、
    // 上流不在でも 404 ではなく 502 + JSON エラーになる (境界の固定)。
    for target in ["/api/nope", "/api/v1/model/shape", "/api/deep/unknown/path"] {
        let r = v.get(target);
        assert_eq!(
            r.status, 502,
            "GET {target}: 上流不在の /api 中継は 502 のはずが {}",
            r.status
        );
        assert_eq!(
            r.content_type().to_ascii_lowercase(),
            "application/json",
            "GET {target} の Content-Type"
        );
        let body = r.text();
        assert!(
            body.starts_with('{') && body.trim_end().ends_with('}'),
            "GET {target} の本文が JSON でない: {body:?}"
        );
        assert!(
            body.contains("\"message\"") && body.contains("upstream unreachable"),
            "GET {target} がエラー内容を返さない: {body:?}"
        );
        assert!(
            !body.to_ascii_lowercase().contains("no route"),
            "GET {target} を 404 相当として扱っている: {body:?}"
        );
    }

    // メソッドを変えても /api 中継のまま (405 にはならない)。
    for method in ["POST", "DELETE", "PUT"] {
        let r = v.method(method, "/api/nope");
        assert_eq!(r.status, 502, "{method} /api/nope は中継され 502 のはず");
    }

    // 対照: 接頭辞が違えば 404。
    for target in ["/apiX/nope", "/apinope", "/v1/limits/extra"] {
        assert_eq!(v.get(target).status, 404, "GET {target} は 404 のはず");
    }
}

#[test]
fn method_edge_cases_head_and_405_allow() {
    let v = start_view();

    // HEAD は GET と同じ Content-Type / Content-Length を返し、本文は空。
    for path in HTML_PAGES {
        let get = v.get(path);
        let head = v.method("HEAD", path);
        assert_eq!(head.status, 200, "HEAD {path} status");
        assert!(
            head.body.is_empty(),
            "HEAD {path} に本文が付いている: {} bytes",
            head.body.len()
        );
        assert_eq!(
            head.content_type().to_ascii_lowercase(),
            get.content_type().to_ascii_lowercase(),
            "HEAD {path} の Content-Type が GET と異なる"
        );
        let glen = get.header("content-length").and_then(|s| s.parse::<usize>().ok());
        let hlen = head
            .header("content-length")
            .and_then(|s| s.parse::<usize>().ok());
        assert_eq!(
            glen, hlen,
            "HEAD {path} の Content-Length が GET と異なる: get={glen:?} head={hlen:?}"
        );
        assert_eq!(
            glen,
            Some(get.body.len()),
            "GET {path} の Content-Length が本文長と一致しない"
        );
    }

    // ページへの書き込み系メソッドは 405 + Allow: GET,HEAD。
    for path in ["/", "/2d", "/3d"] {
        for method in ["POST", "PUT", "DELETE", "OPTIONS"] {
            let r = v.method(method, path);
            assert_eq!(r.status, 405, "{method} {path}");
            let allow = r.allow();
            assert!(allow.contains(&"GET".to_string()), "{method} {path} Allow: {allow:?}");
            assert!(
                allow.contains(&"HEAD".to_string()),
                "{method} {path} Allow: {allow:?}"
            );
            assert!(
                !allow.contains(&method.to_string()),
                "{method} {path} Allow が自分を含む: {allow:?}"
            );
            assert!(r.body.is_empty(), "{method} {path} の 405 本文は空");
        }
    }

    // 未知パスはメソッドに依らず 404 (fallback が 405 より優先される)。
    for method in ["HEAD", "POST", "PUT", "DELETE"] {
        let r = v.method(method, "/does-not-exist");
        assert_eq!(r.status, 404, "{method} /does-not-exist は 404 のはず");
        assert!(r.body.is_empty(), "{method} /does-not-exist の本文は空");
        assert_no_leak(&r, method);
    }

    // vendor の 200 経路も GET/HEAD のみ。
    let r = v.method("DELETE", "/vendor/three.module.js");
    assert_eq!(r.status, 405, "DELETE /vendor/three.module.js");
    assert!(r.allow().contains(&"GET".to_string()), "Allow: {:?}", r.allow());

    // 一連の拒否のあとも通常配信は生きている。
    assert_eq!(v.get("/").status, 200, "405/404 群の後でサーバーが落ちた");
}

#[test]
fn no_external_cdn_references_in_html_or_js() {
    // 検出器そのものが空振りしていないことを先に固定する。
    assert!(is_external_network("https://cdn.invalid/x.js"));
    assert!(is_external_network("http://unpkg.invalid/x.js"));
    assert!(is_external_network("//cdn.invalid/x.js"));
    assert!(!is_external_network("/vendor/three.module.js"));
    assert_eq!(
        attr_values(r#"<script src="https://cdn.invalid/a.js"></script>"#, "src"),
        vec!["https://cdn.invalid/a.js"]
    );
    assert_eq!(
        import_specifiers("import * as T from 'three';\nimport { a } from '/frames.mjs';"),
        vec!["three".to_string(), "/frames.mjs".to_string()]
    );

    let v = start_view();
    let mut checked = 0usize;
    let mut attrs_seen = 0usize;
    let mut imports_seen = 0usize;

    for path in HTML_PAGES {
        let r = v.get(path);
        assert_eq!(r.status, 200, "GET {path} status");
        let body = r.text();
        for (attr, kind) in [("src", "<script src>"), ("href", "<link href>")] {
            for value in attr_values(&body, attr) {
                attrs_seen += 1;
                assert!(
                    !is_external_network(value),
                    "GET {path}: {kind} が外部を指す: {value:?}"
                );
            }
        }
        // <meta http-equiv=refresh content="0;url=..."> の抜け道も塞ぐ。
        for value in attr_values(&body, "content") {
            let lower = value.to_ascii_lowercase();
            assert!(
                !(lower.contains("url=http") && (lower.contains("http://") || lower.contains("https://"))),
                "GET {path}: meta content が外部 URL を指す: {value:?}"
            );
        }
        checked += 1;
    }

    for path in JS_MODULES.iter().chain(VENDOR_FILES.iter()) {
        let r = v.get(path);
        assert_eq!(r.status, 200, "GET {path} status");
        let body = r.text();
        for spec in import_specifiers(&body) {
            imports_seen += 1;
            assert!(
                !is_external_network(&spec),
                "GET {path}: ES import が外部を指す: {spec:?}"
            );
        }
        checked += 1;
    }

    // 既知の CDN 名も本文全体で禁止する (コメント中の誘導 URL も含む)。
    for path in HTML_PAGES
        .iter()
        .chain(JS_MODULES.iter())
        .chain(VENDOR_FILES.iter())
    {
        let r = v.get(path);
        assert_eq!(r.status, 200, "GET {path} status");
        let body = r.text().to_ascii_lowercase();
        for needle in [
            "cdnjs.",
            "unpkg",
            "jsdelivr",
            "googleapis",
            "esm.sh",
            "https://cdn",
            "//cdn.",
        ] {
            assert!(
                !body.contains(needle),
                "GET {path} が外部 CDN トークン {needle:?} を含む"
            );
        }
    }

    assert!(checked >= 20, "only audited {checked} responses");
    assert!(
        attrs_seen >= 4,
        "src/href を {attrs_seen} 個しか見ていない (抽出漏れの疑い)"
    );
    assert!(
        imports_seen >= 15,
        "import 指定子を {imports_seen} 個しか見ていない (抽出漏れの疑い)"
    );
}
