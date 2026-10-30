//! denno-view (P13 骨組み): denno サーバーのテレメトリ WebSocket を受けて、
//! ブラウザの中継 (multiplex) + リプレイ記録 + 簡易ライブダッシュボードを提供する。
//!
//! - 上流: `VIEW_UPSTREAM` (既定 ws://127.0.0.1:8080/v1/telemetry/ws)
//! - 待受: `VIEW_PORT` (既定 8093)。GET / = ダッシュボード、GET /ws = フレーム中継
//! - 記録: `VIEW_RECORD=/path/to/file` でフレームを [u32 len][bytes] で追記
//! - 再生: `VIEW_REPLAY=/path/to/file` で記録ファイルを読み込み `GET /replay`
//!   で生フレーム列を配信 (P16 のリプレイスクラバーがブラウザ側で解析・再生)
//! - キャプチャ: `VIEW_CAPTURE=/path/to/file` の連結フレーム列を `GET /capture.bin`
//!   で配信 (3D ビューワの `?replay=1` が fetch する。WS を開かずに実データを描く)
//!
//! P13 の位置づけ: パイプライン全体 (denno 捕捉 → WS → 中継 → ブラウザ) の
//! 疎通確認が目的。WASM ビューアは P14。

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::response::Html;
use axum::routing::{get, post};
use axum::Router;
use denno_telemetry::{self as telem, Frame};
use futures_util::StreamExt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::broadcast;

/// 上流から受信した生フレーム (エンコード済み) をブラウザへ配信するチャネル
type Bcast = Arc<broadcast::Sender<Vec<u8>>>;

/// 最新状態のスナップショット (ServerInfo / KvUsage の生フレーム)。
/// 新規ブラウザ接続時に即時配信する — ページ読み込みが ServerInfo より後でも
/// 層スタック / KV スロットが構築できるようにする (P14.1)。
type Snap = Arc<std::sync::RwLock<Option<Vec<u8>>>>;

#[derive(Clone)]
struct AppState {
    bcast: Bcast,
    srv: Snap,
    kv: Snap,
    /// チャットプロキシ先 (VIEW_UPSTREAM から導出した http base、例 http://127.0.0.1:8090)
    api: Arc<String>,
    http: reqwest::Client,
}

static FRAMES: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);

/// tokenizer.json (ブラウザ側でのトークン→文字デコード用)。
/// `VIEW_TOKENIZER` でパス指定があった場合のみ読み込む (無ければ 404 → UI は id 表示)。
static TOKENIZER: std::sync::OnceLock<Option<axum::body::Bytes>> = std::sync::OnceLock::new();

/// リプレイ記録ファイル ([u32 len][bytes] の連結)。`VIEW_REPLAY` でパスを指定し、
/// リクエスト毎に読み込む (記録を続けながら再生できる。強制終了で末尾が途切れた
/// 記録はブラウザ側パーサが途中まで読む)。
static REPLAY: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();

/// キャプチャ配信ファイル (`GET /capture.bin`)。8098 の WS メッセージをそのまま
/// 連結したフレーム列 (`DNTL` magic の連なり)。`VIEW_CAPTURE` でパスを指定し、
/// リクエスト毎に読み込む。実データは環境ごとに用意するもので、リポジトリには
/// 同梱しない (未設定・無い場合は 404 を返すだけ)。
static CAPTURE: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();

async fn replay_bin() -> axum::response::Response {
    let not_conf = || {
        axum::response::Response::builder()
            .status(404)
            .body(axum::body::Body::from(
                "replay record not configured (set VIEW_REPLAY)",
            ))
            .unwrap()
    };
    let path = REPLAY.get().and_then(|o| o.as_ref());
    let Some(path) = path else { return not_conf() };
    match std::fs::read(path) {
        Ok(bytes) => axum::response::Response::builder()
            .status(200)
            .header("content-type", "application/octet-stream")
            .body(axum::body::Body::from(bytes))
            .unwrap(),
        Err(e) => axum::response::Response::builder()
            .status(404)
            .body(axum::body::Body::from(format!("replay record read failed: {e}")))
            .unwrap(),
    }
}

/// キャプチャ応答の実体 (グローバルを見ないので単体で叩ける)。
/// 返り値は (status, content-type, body)。未設定・存在しない場合は 404 で短い
/// テキストを返す (ブラウザ側はこれを「キャプチャ無し」としてデモに落とす)。
fn capture_body(path: Option<&str>) -> (u16, &'static str, Vec<u8>) {
    let Some(path) = path else {
        return (
            404,
            "text/plain; charset=utf-8",
            b"capture not configured (set VIEW_CAPTURE)".to_vec(),
        );
    };
    match std::fs::read(path) {
        Ok(bytes) => (200, "application/octet-stream", bytes),
        Err(e) => (
            404,
            "text/plain; charset=utf-8",
            format!("capture read failed: {e}").into_bytes(),
        ),
    }
}

/// GET /capture.bin — `VIEW_CAPTURE` が指す連結フレーム列をそのまま返す。
/// 3D ビューワの `?replay=1` が fetch する (WS を開かずに実データを流し込む)。
async fn capture_bin() -> axum::response::Response {
    let path = CAPTURE.get().and_then(|o| o.as_deref());
    let (status, ctype, body) = capture_body(path);
    axum::response::Response::builder()
        .status(status)
        .header("content-type", ctype)
        .header("cache-control", "no-store")
        .body(axum::body::Body::from(body))
        .unwrap()
}

async fn tokenizer_json() -> axum::response::Response {
    match TOKENIZER.get().and_then(|o| o.as_ref()) {
        Some(bytes) => axum::response::Response::builder()
            .status(200)
            .header("content-type", "application/json")
            .body(axum::body::Body::from(bytes.clone()))
            .unwrap(),
        None => axum::response::Response::builder()
            .status(404)
            .body(axum::body::Body::from(
                "tokenizer not configured (set VIEW_TOKENIZER)",
            ))
            .unwrap(),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();
    let upstream = std::env::var("VIEW_UPSTREAM")
        .unwrap_or_else(|_| "ws://127.0.0.1:8080/v1/telemetry/ws".into());
    let port: u16 = std::env::var("VIEW_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8093);
    let record = std::env::var("VIEW_RECORD").ok();

    // tokenizer.json を起動時に一読 (数 MB 程度。無指定/読込失敗時は配信しない)
    let tok_path = std::env::var("VIEW_TOKENIZER").ok();
    let tok_bytes = tok_path.and_then(|p| match std::fs::read(&p) {
        Ok(b) => {
            tracing::info!("tokenizer.json loaded from {} ({} bytes)", p, b.len());
            Some(axum::body::Bytes::from(b))
        }
        Err(e) => {
            tracing::warn!("tokenizer.json read failed ({}): {e} — token text display disabled", p);
            None
        }
    });
    let _ = TOKENIZER.set(tok_bytes);

    // リプレイ記録のパスを保持 (実体はリクエスト毎に読み込む)
    let replay_path = std::env::var("VIEW_REPLAY").ok();
    if let Some(p) = &replay_path {
        tracing::info!("replay record path: {p}");
    }
    let _ = REPLAY.set(replay_path);

    // キャプチャ配信のパスを保持 (実体はリクエスト毎に読み込む)
    let capture_path = std::env::var("VIEW_CAPTURE").ok();
    if let Some(p) = &capture_path {
        tracing::info!("capture file path: {p}");
    }
    let _ = CAPTURE.set(capture_path);

    let (tx, _) = broadcast::channel::<Vec<u8>>(1024);
    let bcast: Bcast = Arc::new(tx);
    // チャットプロキシ先: ws://host:port/v1/telemetry/ws → http://host:port
    let api = {
        let s = upstream
            .replace("ws://", "http://")
            .replace("wss://", "https://");
        match s.find("/v1/") {
            Some(i) => s[..i].to_string(),
            None => s.trim_end_matches('/').to_string(),
        }
    };
    tracing::info!("chat proxy target: {}/v1/messages", api);
    let app = AppState {
        bcast: bcast.clone(),
        srv: Arc::new(std::sync::RwLock::new(None)),
        kv: Arc::new(std::sync::RwLock::new(None)),
        api: Arc::new(api),
        http: reqwest::Client::new(),
    };

    // 上流接続 (再試行付き) → broadcast へ流す pump
    {
        let bcast = bcast.clone();
        let upstream = upstream.clone();
        let record = record.clone();
        let srv = app.srv.clone();
        let kv = app.kv.clone();
        tokio::spawn(async move {
            pump_upstream(upstream, bcast, record, srv, kv).await;
        });
    }

    // 受信統計の定期ログ
    tokio::spawn(async move {
        let mut last = 0u64;
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            let f = FRAMES.load(Ordering::Relaxed);
            let b = BYTES.load(Ordering::Relaxed);
            if f != last {
                tracing::info!("frames: {} ({} bytes), rate: {}/s", f, b, (f - last) / 5);
                last = f;
            }
        }
    });

    let app = Router::new()
        .route("/", get(dashboard))
        .route("/3d", get(dashboard3d))
        .route("/frames.mjs", get(frames_mjs))
        .route("/vendor/*path", get(vendor))
        .route("/ws", get(view_ws))
        .route("/tokenizer.json", get(tokenizer_json))
        .route("/replay", get(replay_bin))
        .route("/capture.bin", get(capture_bin))
        .route("/code/src", get(code_src))
        .route("/chat", post(chat_proxy))
        .with_state(app);
    let addr = format!("0.0.0.0:{port}");
    tracing::info!("denno-view listening on http://{} (upstream: {})", addr, upstream);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

/// 上流 denno サーバーへ接続してフレームを broadcast に流す。
/// 切断されたら 2 秒間隔で再接続する (denno の再起動に追従)。
async fn pump_upstream(
    url: String,
    bcast: Bcast,
    record: Option<String>,
    srv_snap: Snap,
    kv_snap: Snap,
) {
    let mut record_file = record.and_then(|p| {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(p)
            .ok()
            .map(std::io::BufWriter::new)
    });
    // Gap ログの間引き (洪水時にログが爆発しないよう 1 秒に 1 回にまとめる)
    let mut gap_pending: u32 = 0;
    let mut last_gap_log = std::time::Instant::now();
    loop {
        tracing::info!("connecting upstream: {} ...", url);
        match tokio_tungstenite::connect_async(&url).await {
            Ok((ws, _)) => {
                tracing::info!("upstream connected");
                let (_, mut read) = ws.split();
                while let Some(msg) = read.next().await {
                    match msg {
                        Ok(tokio_tungstenite::tungstenite::Message::Binary(bin)) => {
                            FRAMES.fetch_add(1, Ordering::Relaxed);
                            BYTES.fetch_add(bin.len() as u64, Ordering::Relaxed);
                            // ServerInfo / SessionAdmit / SessionDone / Gap はログにも出す
                            if let Ok(f) = Frame::decode(&bin) {
                                match &f {
                                    Frame::ServerInfo { model, quant, slots, context, layers, .. } => {
                                        *srv_snap.write().unwrap() = Some(bin.clone());
                                        tracing::info!(
                                            "server: {} (quant={}, slots={}, ctx={}, layers={})",
                                            model, quant, slots, context, layers
                                        );
                                    }
                                    Frame::SessionAdmit { session, slot, prompt_tokens } => {
                                        tracing::info!(
                                            "session {} admit (slot {}, prompt {} tok)",
                                            session, slot, prompt_tokens
                                        );
                                    }
                                    Frame::SessionDone { session, reason, output_tokens } => {
                                        tracing::info!(
                                            "session {} done (reason {}, {} tok)",
                                            session, reason, output_tokens
                                        );
                                    }
                                    Frame::KvUsage { .. } => {
                                        *kv_snap.write().unwrap() = Some(bin.clone());
                                    }
                                    Frame::Gap { lost } => {
                                        gap_pending += lost;
                                        if last_gap_log.elapsed() >= std::time::Duration::from_secs(1)
                                        {
                                            tracing::warn!(
                                                "gap: {} frames lost in last {}s",
                                                gap_pending,
                                                last_gap_log.elapsed().as_secs().max(1)
                                            );
                                            gap_pending = 0;
                                            last_gap_log = std::time::Instant::now();
                                        }
                                    }
                                    _ => {}
                                }
                            }
                            if let Some(f) = record_file.as_mut() {
                                use std::io::Write;
                                let len = bin.len() as u32;
                                let _ = f.write_all(&len.to_le_bytes());
                                let _ = f.write_all(&bin);
                            }
                            let _ = bcast.send(bin);
                        }
                        Ok(_) => {} // Ping/Pong はライブラリに任せる
                        Err(e) => {
                            tracing::warn!("upstream error: {e}");
                            break;
                        }
                    }
                }
                tracing::warn!("upstream disconnected, retrying in 2s ...");
            }
            Err(e) => {
                tracing::warn!("connect failed: {e} (retrying in 2s)");
            }
        }
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
    }
}

/// POST /chat — ブラウザのチャットを上流 denno の /v1/messages へ中継する。
/// ボディは Anthropic Messages 形式の JSON をそのまま転送し、応答 (JSON/SSE) を
/// ステータス込みで素通しする。テレメトリは通常経路で流れるのでアニメーションが連動する。
async fn chat_proxy(
    axum::extract::State(app): axum::extract::State<AppState>,
    body: axum::body::Bytes,
) -> axum::response::Response {
    let url = format!("{}/v1/messages", app.api);
    let resp = match app
        .http
        .post(&url)
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            return axum::response::Response::builder()
                .status(502)
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    format!("{{\"error\":{{\"message\":\"upstream unreachable: {e}\",\"type\":\"api_error\"}}}}"),
                ))
                .unwrap();
        }
    };
    let status = axum::http::StatusCode::from_u16(resp.status().as_u16()).unwrap_or_else(|_| {
        axum::http::StatusCode::INTERNAL_SERVER_ERROR
    });
    let ctype = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();
    match resp.bytes().await {
        Ok(bytes) => axum::response::Response::builder()
            .status(status)
            .header("content-type", ctype)
            .body(axum::body::Body::from(bytes))
            .unwrap(),
        Err(e) => axum::response::Response::builder()
            .status(502)
            .body(axum::body::Body::from(format!("upstream read error: {e}")))
            .unwrap(),
    }
}

async fn dashboard() -> Html<&'static str> {
    Html(include_str!("index.html"))
}

/// GET /3d — three.js の 3D ビューア (層の箱 = ニューロン、活性化で発光)。
async fn dashboard3d() -> Html<&'static str> {
    Html(include_str!("index3d.html"))
}

/// GET /frames.mjs — 3D ビューアが import するフレームデコーダ。
/// 形式の正は `denno-telemetry` の v1 (この JS はその写し。DOM 非依存)。
async fn frames_mjs() -> axum::response::Response {
    axum::response::Response::builder()
        .status(200)
        .header("content-type", "text/javascript; charset=utf-8")
        .header("cache-control", "no-cache")
        .body(axum::body::Body::from(
            include_str!("frames.mjs").as_bytes().to_vec(),
        ))
        .unwrap()
}

/// GET /vendor/*path — three.js 本体と addons (ビルド時に埋め込む)。
/// 開発コンテナは外部ネットワークに出られないため、CDN ではなく同梱する。
async fn vendor(axum::extract::Path(path): axum::extract::Path<String>) -> axum::response::Response {
    let (body, mime): (&'static [u8], &'static str) = match path.as_str() {
        "three.module.js" => (include_bytes!("vendor/three.module.js"), "text/javascript; charset=utf-8"),
        "addons/controls/OrbitControls.js" => (include_bytes!("vendor/addons/controls/OrbitControls.js"), "text/javascript; charset=utf-8"),
        "addons/postprocessing/EffectComposer.js" => (include_bytes!("vendor/addons/postprocessing/EffectComposer.js"), "text/javascript; charset=utf-8"),
        "addons/postprocessing/Pass.js" => (include_bytes!("vendor/addons/postprocessing/Pass.js"), "text/javascript; charset=utf-8"),
        "addons/postprocessing/RenderPass.js" => (include_bytes!("vendor/addons/postprocessing/RenderPass.js"), "text/javascript; charset=utf-8"),
        "addons/postprocessing/ShaderPass.js" => (include_bytes!("vendor/addons/postprocessing/ShaderPass.js"), "text/javascript; charset=utf-8"),
        "addons/postprocessing/MaskPass.js" => (include_bytes!("vendor/addons/postprocessing/MaskPass.js"), "text/javascript; charset=utf-8"),
        "addons/postprocessing/UnrealBloomPass.js" => (include_bytes!("vendor/addons/postprocessing/UnrealBloomPass.js"), "text/javascript; charset=utf-8"),
        "addons/postprocessing/OutputPass.js" => (include_bytes!("vendor/addons/postprocessing/OutputPass.js"), "text/javascript; charset=utf-8"),
        "addons/shaders/CopyShader.js" => (include_bytes!("vendor/addons/shaders/CopyShader.js"), "text/javascript; charset=utf-8"),
        "addons/shaders/LuminosityHighPassShader.js" => (include_bytes!("vendor/addons/shaders/LuminosityHighPassShader.js"), "text/javascript; charset=utf-8"),
        "addons/shaders/OutputShader.js" => (include_bytes!("vendor/addons/shaders/OutputShader.js"), "text/javascript; charset=utf-8"),
        _ => (b"not found", "text/plain; charset=utf-8"),
    };
    let status = if body == b"not found" { 404 } else { 200 };
    axum::response::Response::builder()
        .status(status)
        .header("content-type", mime)
        .header("cache-control", "public, max-age=3600")
        .body(axum::body::Body::from(body.to_vec()))
        .unwrap()
}

/// 実装ソース 3 ファイル (実コード表示用)。コンパイル時に埋め込むので
/// ブラウザに見えるコードは常にビルドされたバイナリと同じ世代。
async fn code_src() -> axum::response::Response {
    let body = serde_json::json!({
        "scheduler": include_str!("../../denno-server/src/scheduler.rs"),
        "engine": include_str!("../../denno-engine/src/engine.rs"),
        "llama": include_str!("../../denno-engine/src/model/llama.rs"),
    });
    axum::response::Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .body(axum::body::Body::from(body.to_string()))
        .unwrap()
}

/// GET /ws — ブラウザへのフレーム中継 (複数クライアント可)。
async fn view_ws(
    axum::extract::State(app): axum::extract::State<AppState>,
    upgrade: WebSocketUpgrade,
) -> axum::response::Response {
    upgrade.on_upgrade(move |socket| view_socket(socket, app))
}

async fn view_socket(mut socket: WebSocket, app: AppState) {
    // 新規クライアントにまず最新スナップショット (ServerInfo / KvUsage) を配信 —
    // ページ読み込みがフレーム列の途中でも層スタックと KV スロットが構築される
    let srv_bin = app.srv.read().unwrap().clone();
    if let Some(bin) = srv_bin {
        if socket.send(Message::Binary(bin)).await.is_err() {
            return;
        }
    }
    let kv_bin = app.kv.read().unwrap().clone();
    if let Some(bin) = kv_bin {
        if socket.send(Message::Binary(bin)).await.is_err() {
            return;
        }
    }
    let mut rx = app.bcast.subscribe();
    loop {
        tokio::select! {
            r = rx.recv() => {
                match r {
                    Ok(bin) => {
                        if socket.send(Message::Binary(bin)).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        let gap = Frame::Gap { lost: n as u32 };
                        if socket.send(Message::Binary(gap.encode())).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
            m = socket.recv() => {
                match m {
                    Some(Ok(_)) => {} // クライアント→サーバーのコマンドは P18 (L2 focus) で使用
                    _ => break,
                }
            }
        }
    }
    let _ = telem::dropped(); // 将来の統計表示用 (現状は no-op)
}

#[cfg(test)]
mod tests {
    use super::capture_body;

    /// GET /capture.bin の応答実体。ルート登録そのものはテストしない
    /// (Router を作るには AppState/上流接続が要る) が、404 の 2 経路と
    /// 200 のバイト透過 (0x00 を含む) はここで固定する。
    #[test]
    fn capture_body_404_when_unset_or_missing_and_200_bytes_verbatim() {
        // 1) VIEW_CAPTURE 未設定 → 404 + 設定方法を書いた短いテキスト
        let (status, ctype, body) = capture_body(None);
        assert_eq!(status, 404);
        assert!(ctype.starts_with("text/plain"), "unset content-type: {ctype}");
        let text = String::from_utf8_lossy(&body).into_owned();
        assert!(text.contains("VIEW_CAPTURE"), "unset body: {text}");

        // 2) パスはあるがファイルが無い → 404 (ビューワはデモに落ちる)
        let (status, ctype, body) = capture_body(Some("/nonexistent/denno-capture.bin"));
        assert_eq!(status, 404);
        assert!(ctype.starts_with("text/plain"), "missing content-type: {ctype}");
        let text = String::from_utf8_lossy(&body).into_owned();
        assert!(text.contains("capture read failed"), "missing body: {text}");

        // 3) 実ファイル → 200 + application/octet-stream + バイト列そのまま
        //    (magic "DNTL" + version + kind。0x00 が落ちないこと = 文字列化しない)
        let path = std::env::temp_dir().join(format!(
            "denno-capture-test-{}-{:?}.bin",
            std::process::id(),
            std::thread::current().id()
        ));
        let want: Vec<u8> = b"DNTL\x01\x09\x03\x00\x00\xff".to_vec();
        std::fs::write(&path, &want).expect("write temp capture");
        let (status, ctype, body) = capture_body(Some(path.to_str().unwrap()));
        assert_eq!(status, 200);
        assert_eq!(ctype, "application/octet-stream");
        assert_eq!(body, want);
        let _ = std::fs::remove_file(&path);
    }
}
