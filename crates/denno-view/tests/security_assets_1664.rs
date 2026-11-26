//! task 1664: denno-view の埋め込み静的アセット (HTML / インライン CSS / MJS /
//! vendor JS) に対するゼロ外部 URL 監査、Content-Type の正確性、path traversal
//! 拒否の検査。
//!
//! 方針は `assets_audit_1309.rs` / `http_security_1209.rs` と同じ:
//! - 埋め込みアセットそのものは `include_str!` でコンパイル時に読み、本文に
//!   外部 URL (`http://` / `https://`) やプロトコル相対参照が無いことを見る。
//! - 配信時のヘッダと traversal の扱いは、本物のバイナリを空きポートで起動し
//!   生の HTTP/1.1 を喋って確かめる (URL 正規化を挟まない)。
//!
//! 上流 (denno サーバー) は存在しない前提。VIEW_UPSTREAM は閉じたポートを指す。

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// 埋め込み HTML ページ (include_str! の実体)。
const HTML_ASSETS: &[(&str, &str)] = &[
    ("index.html", include_str!("../src/index.html")),
    ("index2d.html", include_str!("../src/index2d.html")),
    ("index3d.html", include_str!("../src/index3d.html")),
    ("chat.html", include_str!("../src/chat.html")),
];

/// 埋め込み JS モジュール (include_str! の実体)。
const JS_ASSETS: &[(&str, &str)] = &[
    ("frames.mjs", include_str!("../src/frames.mjs")),
    ("view2d.mjs", include_str!("../src/view2d.mjs")),
    ("arch2d.mjs", include_str!("../src/arch2d.mjs")),
    ("nodes2d.mjs", include_str!("../src/nodes2d.mjs")),
    ("hud2d.mjs", include_str!("../src/hud2d.mjs")),
    ("dash.mjs", include_str!("../src/dash.mjs")),
    ("keys3d.mjs", include_str!("../src/keys3d.mjs")),
    ("chat.mjs", include_str!("../src/chat.mjs")),
    ("reconnect.mjs", include_str!("../src/reconnect.mjs")),
    (
        "vendor/three.module.js",
        include_str!("../src/vendor/three.module.js"),
    ),
    (
        "vendor/addons/controls/OrbitControls.js",
        include_str!("../src/vendor/addons/controls/OrbitControls.js"),
    ),
    (
        "vendor/addons/postprocessing/EffectComposer.js",
        include_str!("../src/vendor/addons/postprocessing/EffectComposer.js"),
    ),
    (
        "vendor/addons/postprocessing/Pass.js",
        include_str!("../src/vendor/addons/postprocessing/Pass.js"),
    ),
    (
        "vendor/addons/postprocessing/RenderPass.js",
        include_str!("../src/vendor/addons/postprocessing/RenderPass.js"),
    ),
    (
        "vendor/addons/postprocessing/ShaderPass.js",
        include_str!("../src/vendor/addons/postprocessing/ShaderPass.js"),
    ),
    (
        "vendor/addons/postprocessing/MaskPass.js",
        include_str!("../src/vendor/addons/postprocessing/MaskPass.js"),
    ),
    (
        "vendor/addons/postprocessing/UnrealBloomPass.js",
        include_str!("../src/vendor/addons/postprocessing/UnrealBloomPass.js"),
    ),
    (
        "vendor/addons/postprocessing/OutputPass.js",
        include_str!("../src/vendor/addons/postprocessing/OutputPass.js"),
    ),
    (
        "vendor/addons/shaders/CopyShader.js",
        include_str!("../src/vendor/addons/shaders/CopyShader.js"),
    ),
    (
        "vendor/addons/shaders/LuminosityHighPassShader.js",
        include_str!("../src/vendor/addons/shaders/LuminosityHighPassShader.js"),
    ),
    (
        "vendor/addons/shaders/OutputShader.js",
        include_str!("../src/vendor/addons/shaders/OutputShader.js"),
    ),
];

const HTML_PAGES: &[&str] = &["/", "/2d", "/3d", "/chat"];

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

/// 本物のバイナリを立て `/` が 200 + HTML を返すまで待つ。ポート競合に備えて
/// 別ポートで数回やり直す。
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

/// `attr="..."` / `attr='...'` の値を取り出す。直前が空白か `<` のときだけ
/// 属性名とみなす (`data-src=` などを拾わない)。
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
            if kw == "import" {
                if !src[j..].starts_with('(') {
                    continue;
                }
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

/// `<style>...</style>` の中身をすべて取り出す (denno-view の CSS は全ページ
/// インライン)。タグが閉じていない場合は末尾までを CSS とみなす。
fn style_blocks(html: &str) -> Vec<&str> {
    blocks(html, "style")
}

/// `<script>...</script>` の中身をすべて取り出す (インライン JS の走査用)。
/// importmap の JSON ブロックも含むが、JSON には外部 URL 参照が無い前提で
/// 同じリンク走査に掛ける (あれば外部 URL として検出される)。
fn script_blocks(html: &str) -> Vec<&str> {
    blocks(html, "script")
}

fn blocks<'a>(html: &'a str, tag: &str) -> Vec<&'a str> {
    let mut out = Vec::new();
    let lower = html.to_ascii_lowercase();
    let open = format!("<{tag}");
    let close = format!("</{tag}");
    let mut cursor = 0usize;
    while let Some(pos) = lower[cursor..].find(&open) {
        let open_start = cursor + pos;
        let Some(tag_end_rel) = html[open_start..].find('>') else {
            break;
        };
        // `<scripture>` のような別タグを拾わない。
        let after = html[open_start + open.len()..].chars().next();
        if !after.map_or(true, |c| c.is_whitespace() || c == '>') {
            cursor = open_start + open.len();
            continue;
        }
        let body_start = open_start + tag_end_rel + 1;
        match lower[body_start..].find(&close) {
            Some(end) => {
                out.push(&html[body_start..body_start + end]);
                cursor = body_start + end + close.len();
            }
            None => {
                out.push(&html[body_start..]);
                break;
            }
        }
    }
    out
}

fn is_external_url(value: &str) -> bool {
    let v = value.trim().to_ascii_lowercase();
    v.starts_with("http://") || v.starts_with("https://") || v.starts_with("//")
}

/// 本文中の外部 URL を (大小文字を無視して) 探す。
fn external_url_token(body: &str) -> Option<&str> {
    body.split(|c: char| c.is_whitespace() || c == '"' || c == '\'' || c == '(' || c == ')')
        .find(|token| {
            let t = token.to_ascii_lowercase();
            t.starts_with("http://") || t.starts_with("https://")
        })
}

/// JS のコメントだけを空白に置き換える。文字列リテラルの中身は残す
/// (vendor three.js のドキュメント URL コメントを「リンク」と誤検出しないため)。
/// 正規表現リテラル内の `//` は稀なので考慮しない。
fn strip_js_comments(src: &str) -> String {
    #[derive(Clone, Copy, PartialEq)]
    enum St {
        Code,
        Line,
        Block,
        Single,
        Double,
        Template,
    }
    let chars: Vec<char> = src.chars().collect();
    let mut out = String::with_capacity(src.len());
    let mut st = St::Code;
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        match st {
            St::Code => {
                if c == '/' && next == Some('/') {
                    st = St::Line;
                    i += 2;
                } else if c == '/' && next == Some('*') {
                    st = St::Block;
                    i += 2;
                } else {
                    if c == '\'' {
                        st = St::Single;
                    } else if c == '"' {
                        st = St::Double;
                    } else if c == '`' {
                        st = St::Template;
                    }
                    out.push(c);
                    i += 1;
                }
            }
            St::Line => {
                if c == '\n' {
                    out.push('\n');
                    st = St::Code;
                }
                i += 1;
            }
            St::Block => {
                if c == '*' && next == Some('/') {
                    st = St::Code;
                    i += 2;
                } else {
                    if c == '\n' {
                        out.push('\n');
                    }
                    i += 1;
                }
            }
            St::Single | St::Double | St::Template => {
                out.push(c);
                if c == '\\' {
                    if let Some(n) = next {
                        out.push(n);
                        i += 2;
                        continue;
                    }
                }
                let closes = match st {
                    St::Single => c == '\'',
                    St::Double => c == '"',
                    _ => c == '`',
                };
                if closes {
                    st = St::Code;
                }
                i += 1;
            }
        }
    }
    out
}

/// `callee(` の直後の文字列リテラル第 1 引数を取り出す。
fn call_arg_strings(code: &str, callee: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cursor = 0usize;
    while let Some(pos) = code[cursor..].find(callee) {
        let start = cursor + pos;
        cursor = start + callee.len();
        let before = code[..start].chars().next_back();
        if !before.map_or(true, |c| !(c.is_alphanumeric() || c == '_' || c == '$')) {
            continue;
        }
        let mut j = cursor;
        while let Some(c) = code[j..].chars().next() {
            if c.is_whitespace() {
                j += c.len_utf8();
            } else {
                break;
            }
        }
        if !code[j..].starts_with('(') {
            continue;
        }
        j += 1;
        while let Some(c) = code[j..].chars().next() {
            if c.is_whitespace() {
                j += c.len_utf8();
            } else {
                break;
            }
        }
        let quote = match code[j..].chars().next() {
            Some(q @ ('"' | '\'' | '`')) => q,
            _ => continue,
        };
        j += quote.len_utf8();
        let rest = &code[j..];
        if let Some(end) = rest.find(quote) {
            out.push(rest[..end].to_string());
        }
    }
    out
}

/// 実コード中のネットワーク読み込み API と src/href 代入に外部 URL が無いこと。
fn assert_no_external_network_refs(name: &str, code: &str) {
    let code = strip_js_comments(code);
    for callee in [
        "fetch",
        "import",
        "Worker",
        "SharedWorker",
        "EventSource",
        "WebSocket",
        "sendBeacon",
    ] {
        for arg in call_arg_strings(&code, callee) {
            if is_external_url(&arg) {
                panic!("{name}: {callee}() が外部を指す: {arg:?}");
            }
        }
    }
    // `el.src = "https://…"` / `el.href = "//…"` の形。
    let compact: String = code.chars().filter(|c| !c.is_whitespace()).collect();
    let lower = compact.to_ascii_lowercase();
    for prefix in [".src=", ".href=", ".poster=", ".action="] {
        for q in ["\"", "'", "`"] {
            for scheme in ["http://", "https://", "//"] {
                let needle = format!("{prefix}{q}{scheme}");
                if lower.contains(&needle) {
                    panic!("{name}: 代入 {prefix} が外部を指す ({needle:?})");
                }
            }
        }
    }
}


#[test]
fn embedded_html_css_js_assets_have_zero_external_urls() {
    // 検出器の健全性 (空振りしていないこと) を先に固定する。
    assert!(is_external_url("https://cdn.invalid/app.js"));
    assert!(is_external_url("HTTP://cdn.invalid/app.js"));
    assert!(is_external_url("//cdn.invalid/app.js"));
    assert!(!is_external_url("/vendor/three.module.js"));
    assert!(!is_external_url("ws://127.0.0.1:1/v1/telemetry/ws"));
    assert_eq!(
        external_url_token("<script src=\"https://cdn.invalid/a.js\"></script>"),
        Some("https://cdn.invalid/a.js")
    );
    assert_eq!(
        attr_values("<link rel='stylesheet' href='//cdn.invalid/b.css'>", "href"),
        vec!["//cdn.invalid/b.css"]
    );
    assert_eq!(
        import_specifiers("import * as T from 'three';\nimport { a } from '/frames.mjs';"),
        vec!["three".to_string(), "/frames.mjs".to_string()]
    );
    // コメントは落とし、文字列は残す。
    let stripped = strip_js_comments("a(); // fetch('https://cdn.invalid/x')\nb();\n/* c */");
    assert!(!stripped.contains("fetch"));
    assert!(stripped.contains("a();") && stripped.contains("b();"));
    assert_eq!(
        call_arg_strings("await fetch( 'https://cdn.invalid/x' )", "fetch"),
        vec!["https://cdn.invalid/x".to_string()]
    );
    assert_eq!(
        call_arg_strings("new Worker(\"//cdn.invalid/w.js\")", "Worker"),
        vec!["//cdn.invalid/w.js".to_string()]
    );

    let mut scanned_bytes = 0usize;
    let mut attr_refs = 0usize;
    let mut import_refs = 0usize;
    let mut script_bytes = 0usize;

    for (name, body) in HTML_ASSETS {
        scanned_bytes += body.len();

        // <script src> / <link href> / <img src> / <form action> などの
        // 取得先属性はプロトコル相対も外部扱い。
        for attr in ["src", "href", "data", "action", "poster", "srcset"] {
            for value in attr_values(body, attr) {
                for part in value.split(',') {
                    attr_refs += 1;
                    let part = part.trim();
                    if is_external_url(part) {
                        panic!("埋め込み HTML {name} の {attr} が外部を指す: {part:?}");
                    }
                }
            }
        }
        // <meta http-equiv=refresh content="0;url=http://..."> の抜け道。
        for value in attr_values(body, "content") {
            let lower = value.to_ascii_lowercase();
            assert!(
                !lower.contains("http://") && !lower.contains("https://"),
                "埋め込み HTML {name} の meta content が外部 URL を指す: {value:?}"
            );
        }

        // インライン CSS に外部 @import / url() を入れない。
        let styles = style_blocks(body);
        assert!(
            !styles.is_empty(),
            "埋め込み HTML {name} に <style> が無い (CSS 抽出の空振り)"
        );
        for css in styles {
            let lower = css.to_ascii_lowercase();
            assert!(
                !lower.contains("http://") && !lower.contains("https://"),
                "埋め込み CSS ({name}) に外部 URL が混じる"
            );
            assert!(
                !lower.contains("@import"),
                "埋め込み CSS ({name}) に @import がある"
            );
            assert!(
                !lower.contains("url(//"),
                "埋め込み CSS ({name}) にプロトコル相対 url() がある"
            );
        }

        // インライン <script> の静的 import / 動的 import() / 取得 API。
        let scripts = script_blocks(body);
        assert!(
            !scripts.is_empty(),
            "埋め込み HTML {name} に <script> が無い (JS 抽出の空振り)"
        );
        for script in scripts {
            script_bytes += script.len();
            assert_no_external_network_refs(&format!("HTML {name} <script>"), script);
            let stripped = strip_js_comments(script);
            for spec in import_specifiers(&stripped) {
                import_refs += 1;
                if is_external_url(&spec) {
                    panic!("埋め込み HTML {name} の import が外部を指す: {spec:?}");
                }
            }
        }
    }

    for (name, body) in JS_ASSETS {
        scanned_bytes += body.len();
        assert_no_external_network_refs(name, body);
        let stripped = strip_js_comments(body);
        for spec in import_specifiers(&stripped) {
            import_refs += 1;
            if is_external_url(&spec) {
                panic!("埋め込み JS {name} の import が外部を指す: {spec:?}");
            }
        }
    }

    // 監査が空振りしていない下限。CSS は全ページにインラインである前提。
    assert!(HTML_ASSETS.len() >= 4, "HTML アセットが少なすぎる");
    assert!(JS_ASSETS.len() >= 20, "JS アセットが少なすぎる");
    assert!(scanned_bytes > 100_000, "走査した総バイト数が小さすぎる");
    assert!(script_bytes > 0, "インライン JS を走査していない");
    assert!(
        attr_refs >= 2,
        "src/href などの参照を {attr_refs} 個しか見ていない"
    );
    assert!(
        import_refs >= 10,
        "import 指定子を {import_refs} 個しか見ていない"
    );
}

#[test]
fn served_assets_report_accurate_content_types() {
    let v = start_view();

    for path in HTML_PAGES {
        let r = v.get(path);
        assert_eq!(r.status, 200, "GET {path} status");
        let ct = r.content_type().to_ascii_lowercase();
        assert!(
            ct.starts_with("text/html"),
            "GET {path} は text/html: {:?}",
            r.content_type()
        );
        assert!(
            ct.contains("charset=utf-8"),
            "GET {path} は charset=utf-8 を宣言する: {:?}",
            r.content_type()
        );
        assert!(r.body.len() > 1000, "GET {path} 本文が小さすぎる");
    }

    for path in JS_MODULES.iter().chain(VENDOR_FILES.iter()) {
        let r = v.get(path);
        assert_eq!(r.status, 200, "GET {path} status");
        let ct = r.content_type().to_ascii_lowercase();
        assert!(
            ct.starts_with("text/javascript") || ct.starts_with("application/javascript"),
            "GET {path} は JavaScript の MIME で返す: {:?}",
            r.content_type()
        );
        assert!(
            !ct.starts_with("text/html") && !ct.starts_with("application/octet-stream"),
            "GET {path} が JS 以外の MIME で返っている: {:?}",
            r.content_type()
        );
        assert!(
            ct.contains("charset=utf-8"),
            "GET {path} は charset=utf-8 を宣言する: {:?}",
            r.content_type()
        );
        assert!(!r.body.is_empty(), "GET {path} 本文が空");
    }

    // 埋め込みアセットではないが、同じサーバーが返す JSON の型も固定する。
    for path in ["/status", "/healthz", "/code/src"] {
        let r = v.get(path);
        assert_eq!(r.status, 200, "GET {path} status");
        let ct = r.content_type().to_ascii_lowercase();
        assert!(
            ct.starts_with("application/json"),
            "GET {path} は application/json: {:?}",
            r.content_type()
        );
        assert!(
            !ct.starts_with("text/html"),
            "GET {path} が HTML として返っている: {:?}",
            r.content_type()
        );
        assert!(!r.body.is_empty(), "GET {path} 本文が空");
    }

    // 未収録 vendor は 404 でプレーンテキスト。HTML フォールバックに化けない。
    let r = v.get("/vendor/nonexistent-addon-1664.js");
    assert_eq!(r.status, 404, "未収録 vendor は 404");
    let ct = r.content_type().to_ascii_lowercase();
    assert!(
        ct.starts_with("text/plain"),
        "未収録 vendor は text/plain: {:?}",
        r.content_type()
    );
    assert!(
        !r.text().to_ascii_lowercase().contains("<html"),
        "未収録 vendor の 404 が HTML を返している"
    );
}

#[test]
fn path_traversal_targets_are_rejected_without_source_leak() {
    let v = start_view();

    // リポジトリ外のファイルや実装ソースを読ませようとする生のターゲット。
    // 生のリクエスト行に載せるので、`..` が URL 正規化で消えない。
    let targets = [
        "/vendor/../src/main.rs",
        "/vendor/../../Cargo.toml",
        "/vendor/../../../etc/passwd",
        "/vendor/..%2f..%2fCargo.toml",
        "/vendor/%2e%2e/%2e%2e/src/main.rs",
        "/vendor/..\\..\\Cargo.toml",
        "/vendor//etc/passwd",
        "/vendor/./../../Cargo.toml",
        "/../../etc/passwd",
        "/%2e%2e/%2e%2e/etc/passwd",
        "/code/../../Cargo.toml",
        "/code/../src/main.rs",
    ];

    // traversal 応答に混じったら漏洩とみなす文字列。通常の 404 本文には出ない。
    let leak_markers = [
        "fn main(",
        "#[tokio::main]",
        "use axum::",
        "[package]",
        "[workspace]",
        "name = \"denno-view\"",
        "root:x:0:0",
    ];

    for target in targets {
        let r = v.get(target);
        assert!(
            (400..500).contains(&r.status),
            "{target:?} は 4xx で拒否する (実際: {})",
            r.status
        );
        let body = r.text();
        for marker in leak_markers {
            assert!(
                !body.contains(marker),
                "{target:?} の応答に {marker:?} が漏れている: {body:?}"
            );
        }
        // 4xx に公開キャッシュ指示を付けない (`no-store` かヘッダ無し)。
        let cc = r.header("cache-control").unwrap_or("").to_ascii_lowercase();
        if !cc.is_empty() {
            assert!(
                cc.contains("no-store"),
                "{target:?} の 4xx に Cache-Control: {cc:?}"
            );
            assert!(
                !cc.contains("public") && !cc.contains("max-age"),
                "{target:?} の 4xx を公開キャッシュ可能にしない: {cc:?}"
            );
        }
    }

    // traversal を浴びせた後もサーバーは生きており、正規アセットは無傷で返る。
    let ok = v.get("/");
    assert_eq!(ok.status, 200, "traversal 後に GET / が壊れた");
    assert!(ok.text().starts_with("<!DOCTYPE html>"));
    let js = v.get("/vendor/three.module.js");
    assert_eq!(js.status, 200, "traversal 後に正規 vendor が壊れた");
    assert!(
        js.content_type().to_ascii_lowercase().starts_with("text/javascript"),
        "traversal 後の正規 vendor MIME: {:?}",
        js.content_type()
    );
}
