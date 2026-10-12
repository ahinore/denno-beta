//! denno-view (P13 骨組み): denno サーバーのテレメトリ WebSocket を受けて、
//! ブラウザの中継 (multiplex) + リプレイ記録 + 簡易ライブダッシュボードを提供する。
//!
//! - 上流: `VIEW_UPSTREAM` (既定 ws://127.0.0.1:8080/v1/telemetry/ws)
//! - 待受: `VIEW_PORT` (既定 8093)。GET / = ダッシュボード、GET /ws = フレーム中継
//! - 記録: `VIEW_RECORD=/path/to/file` でフレームを [u32 len][bytes] で追記
//!   (P14 のリプレイ再生の入力形式)
//!
//! P13 の位置づけ: パイプライン全体 (denno 捕捉 → WS → 中継 → ブラウザ) の
//! 疎通確認が目的。WASM ビューアは P14。

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::response::Html;
use axum::routing::get;
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
}

static FRAMES: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);

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

    let (tx, _) = broadcast::channel::<Vec<u8>>(1024);
    let bcast: Bcast = Arc::new(tx);
    let app = AppState {
        bcast: bcast.clone(),
        srv: Arc::new(std::sync::RwLock::new(None)),
        kv: Arc::new(std::sync::RwLock::new(None)),
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
        .route("/ws", get(view_ws))
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
                                    Frame::ServerInfo { model, quant, slots, context, layers } => {
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

async fn dashboard() -> Html<&'static str> {
    Html(include_str!("index.html"))
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
                    Some(Ok(_)) => {} // クライアント→サーバーのコマンドは P15 (focus) で使用
                    _ => break,
                }
            }
        }
    }
    let _ = telem::dropped(); // 将来の統計表示用 (現状は no-op)
}
