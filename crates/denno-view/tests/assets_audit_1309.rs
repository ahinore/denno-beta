//! task 1309: denno-view が配信する埋め込みアセット (HTML / MJS / vendor) の
//! 完全性とゼロ CDN 監査。
//!
//! denno-view はバイナリ crate なので Router を統合テストから直接は組めない。
//! `http_contract_949.rs` / `http_security_1209.rs` と同じ方針で、本物のバイナリを
//! 空きポートで起動し、生の HTTP/1.1 を喋って応答ヘッダと本文バイト列をそのまま
//! 見る (reqwest の URL 正規化や文字コード変換を挟まない)。
//!
//! 見ているもの:
//!   1. HTML ページ (`/`, `/2d`, `/3d`, `/chat`) と .mjs / vendor が valid UTF-8 で、
//!      HTML は charset=utf-8 を宣言している
//!   2. `<script src>` / `<link href>` / ES `import` の参照先に外部 http(s):// や
//!      プロトコル相対 // が無い (ゼロ CDN)。SVG 名前空間や ws:// は対象外
//!   3. HTML / importmap が参照するローカルアセットが実際に 200 で解決する
//!      (アセット整合。存在しない参照を残さない)
//!   4. CSP はページ間で一貫している (全ページ無し、または全ページ同一)。
//!      存在するなら default-src 'self' を含み、ワイルドカード '*' を含まない
//!   5. Cache-Control はアセット種別ごとに一貫している:
//!      HTML は公開キャッシュ不可 (public / max-age を付けない)、.mjs は no-cache、
//!      vendor の 200 は max-age、vendor / capture の 404 は no-store
//!
//! 上流 (denno サーバー) は存在しない前提。VIEW_UPSTREAM は閉じたポートを指すので
//! pump は再接続を試みるだけで、静的配信には影響しない。

use std::collections::BTreeSet;
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

    /// 本文が valid UTF-8 であることを確かめて &str を返す。
    fn utf8(&self, what: &str) -> &str {
        std::str::from_utf8(&self.body)
            .unwrap_or_else(|e| panic!("{what}: 本文が valid UTF-8 でない: {e}"))
    }

    fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
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

/// 本物のバイナリを立てて `/` が 200 + HTML を返すまで待つ。ポート競合に備えて
/// 別ポートで数回やり直す (`http_security_1209.rs` と同じ防御)。
fn start_view() -> View {
    for _ in 0..8 {
        let port = free_port();
        let mut child = Command::new(env!("CARGO_BIN_EXE_denno-view"))
            .env("VIEW_PORT", port.to_string())
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
            if TcpStream::connect(&addr).is_ok() {
                // 生きていて / が HTML を返す個体だけ採用する。
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

/// `attr="..."` / `attr='...'` の値を取り出す。HTML 断片の素朴な走査で十分
/// (完全なパーサではないが、src/href の取りこぼしより誤検出を恐れない設計)。
/// 直前が空白か `<` のときだけ属性名とみなす (`data-src=` などを拾わない)。
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
        // 属性名と '=' の間の空白
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
            // キーワード直後の空白を飛ばす
            let mut j = cursor;
            while let Some(c) = src[j..].chars().next() {
                if c.is_whitespace() {
                    j += c.len_utf8();
                } else {
                    break;
                }
            }
            if kw == "import" && !src[j..].starts_with('(') {
                continue; // `import x from` 側は "from" で拾う (静的 import 宣言)
            }
            if kw == "import" {
                j += 1; // '(' を飛ばす
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

/// ローカル参照を配信パスへ解決する。外部 URL・インライン・断片・テンプレートは
/// None (ネットワーク取得の対象外)。
fn resolve_local_asset(value: &str) -> Option<String> {
    let v = value.trim();
    if v.is_empty()
        || v.starts_with('#')
        || v.starts_with("data:")
        || v.starts_with("mailto:")
        || v.starts_with("javascript:")
        || v.starts_with("blob:")
        || is_external_network(v)
        || v.contains("${")
    {
        return None;
    }
    // importmap の bare specifier を同梱 vendor へ解決する。
    if v == "three" {
        return Some("/vendor/three.module.js".to_string());
    }
    if let Some(rest) = v.strip_prefix("three/addons/") {
        return Some(format!("/vendor/addons/{rest}"));
    }
    // query / fragment を落とす。
    let path = v.split('#').next().unwrap_or(v);
    let path = path.split('?').next().unwrap_or(path);
    if path.is_empty() {
        return None;
    }
    if path.starts_with('/') {
        return Some(path.to_string());
    }
    // ./x.mjs や x.mjs はページ直下 (= ルート) 基準で解決する。
    Some(format!("/{}", path.trim_start_matches("./")))
}

#[test]
fn served_html_and_js_are_valid_utf8_and_html_declares_charset() {
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
        let body = r.utf8(path);
        assert!(
            body.to_ascii_lowercase().contains("<!doctype html>"),
            "GET {path} に DOCTYPE 宣言が無い"
        );
        // HTML として妥当な骨格 (head/body) を持つこと。
        let lower = body.to_ascii_lowercase();
        assert!(lower.contains("<html"), "GET {path} に <html> が無い");
        assert!(lower.contains("<head"), "GET {path} に <head> が無い");
        assert!(lower.contains("<body"), "GET {path} に <body> が無い");
        assert!(body.len() > 1000, "GET {path} 本文が小さすぎる: {}", body.len());
    }

    for path in JS_MODULES.iter().chain(VENDOR_FILES.iter()) {
        let r = v.get(path);
        assert_eq!(r.status, 200, "GET {path} status");
        let ct = r.content_type().to_ascii_lowercase();
        assert!(
            ct.starts_with("text/javascript") || ct.starts_with("application/javascript"),
            "GET {path} content-type: {:?}",
            r.content_type()
        );
        let body = r.utf8(path);
        assert!(!body.is_empty(), "GET {path} 本文が空");
        assert!(
            !body.contains('\u{0}'),
            "GET {path} 本文に NUL バイトが混じる"
        );
    }
}

#[test]
fn script_link_and_import_references_are_all_local_zero_cdn() {
    // 検出器そのものが空振りしていないことを先に固定する (負のテストの健全性)。
    assert!(is_external_network("https://cdn.invalid/x.js"));
    assert!(is_external_network("http://unpkg.invalid/x.js"));
    assert!(is_external_network("//cdn.invalid/x.js"));
    assert!(!is_external_network("/vendor/three.module.js"));
    assert!(!is_external_network("ws://127.0.0.1:8098/ws"));
    assert_eq!(
        attr_values(r#"<script src="https://cdn.invalid/a.js"></script>"#, "src"),
        vec!["https://cdn.invalid/a.js"]
    );
    assert_eq!(
        attr_values("<link rel='stylesheet' href='//cdn.invalid/b.css'>", "href"),
        vec!["//cdn.invalid/b.css"]
    );
    assert_eq!(
        import_specifiers("import * as T from 'three';\nimport { a } from '/frames.mjs';"),
        vec!["three".to_string(), "/frames.mjs".to_string()]
    );

    let v = start_view();
    let mut checked = 0usize;
    let mut attrs_seen = 0usize;
    let mut imports_seen = 0usize;

    let mut audit = |path: &str, body: &str, is_html: bool, attrs: &mut usize, imports: &mut usize| {
        if is_html {
            for (attr, kind) in [("src", "<script src>"), ("href", "<link href>")] {
                for value in attr_values(body, attr) {
                    *attrs += 1;
                    if is_external_network(value) {
                        panic!("GET {path}: {kind} が外部を指す: {value:?}");
                    }
                }
            }
            for value in attr_values(body, "content") {
                // <meta http-equiv=refresh content="0;url=..."> などの抜け道も塞ぐ
                if value.to_ascii_lowercase().contains("url=http")
                    && (value.contains("http://") || value.contains("https://"))
                {
                    panic!("GET {path}: meta content が外部 URL を指す: {value:?}");
                }
            }
        }
        for spec in import_specifiers(body) {
            *imports += 1;
            if is_external_network(&spec) {
                panic!("GET {path}: ES import が外部を指す: {spec:?}");
            }
        }
        checked += 1;
    };

    for path in HTML_PAGES {
        let r = v.get(path);
        assert_eq!(r.status, 200, "GET {path} status");
        let body = r.utf8(path).to_string();
        audit(path, &body, true, &mut attrs_seen, &mut imports_seen);
    }
    for path in JS_MODULES.iter().chain(VENDOR_FILES.iter()) {
        let r = v.get(path);
        assert_eq!(r.status, 200, "GET {path} status");
        let body = r.utf8(path).to_string();
        audit(path, &body, false, &mut attrs_seen, &mut imports_seen);
    }

    assert!(checked >= 20, "only audited {checked} assets");
    assert!(attrs_seen >= 4, "src/href を {attrs_seen} 個しか見ていない");
    assert!(
        imports_seen >= 15,
        "import 指定子を {imports_seen} 個しか見ていない"
    );
}

#[test]
fn every_referenced_local_asset_resolves_to_200() {
    let v = start_view();
    // ページとページ内 import から集めた参照先を、実際にサーバーで引く。
    let mut wanted: BTreeSet<String> = BTreeSet::new();

    for path in HTML_PAGES.iter().chain(JS_MODULES.iter()) {
        let r = v.get(path);
        assert_eq!(r.status, 200, "GET {path} status");
        let body = r.utf8(path);
        for attr in ["src", "href"] {
            for value in attr_values(body, attr) {
                if let Some(local) = resolve_local_asset(value) {
                    wanted.insert(local);
                }
            }
        }
        for spec in import_specifiers(body) {
            if let Some(local) = resolve_local_asset(&spec) {
                wanted.insert(local);
            }
        }
    }

    // importmap の bare specifier ('three', 'three/addons/…') は import_specifiers
    // 側で解決済み。ここで最低限の必須アセットが入っていることを確認する。
    for must in [
        "/frames.mjs",
        "/vendor/three.module.js",
        "/vendor/addons/controls/OrbitControls.js",
    ] {
        assert!(
            wanted.contains(must),
            "参照集合に {must} が無い (抽出漏れの疑い): {wanted:?}"
        );
    }

    let mut checked = 0usize;
    for target in &wanted {
        let r = v.get(target);
        assert_eq!(
            r.status, 200,
            "参照先 {target} が {} を返した (dangling asset)",
            r.status
        );
        assert!(!r.body.is_empty(), "参照先 {target} が空");
        checked += 1;
    }
    assert!(checked >= 10, "resolved only {checked} referenced assets");
}

#[test]
fn csp_header_is_consistent_across_pages() {
    let v = start_view();
    let mut seen: Vec<(&str, Option<String>)> = Vec::new();
    for path in HTML_PAGES {
        let r = v.get(path);
        assert_eq!(r.status, 200, "GET {path} status");
        seen.push((path, r.header("content-security-policy").map(str::to_string)));
    }

    // 全ページ無し、または全ページ同一のときだけ一貫とみなす。
    let first = seen[0].1.clone();
    for (path, csp) in &seen {
        assert_eq!(
            *csp, first,
            "CSP がページ間で不一致: / は {first:?}, {path} は {csp:?}"
        );
    }

    if let Some(csp) = &first {
        let lower = csp.to_ascii_lowercase();
        assert!(
            lower.contains("default-src"),
            "CSP は default-src を持つ: {csp:?}"
        );
        assert!(
            lower.contains("'self'"),
            "CSP は 'self' を許可する: {csp:?}"
        );
        assert!(
            !csp.contains('*'),
            "CSP にワイルドカード源を入れない: {csp:?}"
        );
        assert!(
            !lower.contains("http://") && !lower.contains("https://"),
            "CSP に外部ホストを列挙しない: {csp:?}"
        );
    }
}

#[test]
fn cache_control_is_consistent_per_asset_class() {
    let v = start_view();

    // HTML は動的ビュー (同梱アセットのパスを埋め込む) — 公開キャッシュ不可。
    let mut page_cc: Vec<(&str, String)> = Vec::new();
    for path in HTML_PAGES {
        let r = v.get(path);
        assert_eq!(r.status, 200, "GET {path} status");
        let cc = r.header("cache-control").unwrap_or("").to_ascii_lowercase();
        assert!(
            !cc.contains("public"),
            "GET {path} を public キャッシュ不可にする: {cc:?}"
        );
        assert!(
            !cc.contains("max-age"),
            "GET {path} に max-age を付けない: {cc:?}"
        );
        page_cc.push((path, cc));
    }
    // ページ間で同じ方針 (全部ヘッダ無し、または同一指示)。
    let first_cc = page_cc[0].1.clone();
    for (path, cc) in &page_cc {
        assert_eq!(
            *cc, first_cc,
            "HTML の Cache-Control がページ間で不一致: / は {first_cc:?}, {path} は {cc:?}"
        );
    }

    // 同梱 .mjs は常に no-cache (更新を即反映する)。
    for path in JS_MODULES {
        let r = v.get(path);
        assert_eq!(r.status, 200, "GET {path} status");
        let cc = r.header("cache-control").unwrap_or("").to_ascii_lowercase();
        assert_eq!(cc, "no-cache", "GET {path} の Cache-Control: {cc:?}");
    }

    // vendor の 200 はキャッシュ可 (max-age)、404 は no-store。
    for path in VENDOR_FILES {
        let r = v.get(path);
        assert_eq!(r.status, 200, "GET {path} status");
        let cc = r.header("cache-control").unwrap_or("").to_ascii_lowercase();
        assert!(
            cc.contains("public") && cc.contains("max-age="),
            "vendor 200 {path} はキャッシュ可: {cc:?}"
        );
    }
    for (target, why) in [
        ("/vendor/nonexistent-addon.js", "vendor 未収録"),
        ("/capture.bin", "capture 未設定"),
    ] {
        let r = v.get(target);
        assert_eq!(r.status, 404, "{why}: {target}");
        let cc = r.header("cache-control").unwrap_or("").to_ascii_lowercase();
        assert!(
            cc.contains("no-store"),
            "{why}: 404 は no-store: {cc:?}"
        );
        assert!(
            !cc.contains("public") && !cc.contains("max-age"),
            "{why}: 404 を公開キャッシュ不可: {cc:?}"
        );
    }
}
