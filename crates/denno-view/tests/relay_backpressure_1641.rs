//! task 1641: denno-view の中継 (/ws) と埋め込み 2D/3D ビューアの
//! 高レート・テレメトリ洪水に対する「有界メモリ」「GAP の明示」「例外を外へ出さない」
//! を固定する。
//!
//! 2 種類の検査を置く:
//!
//!   1. `slow_subscriber_gets_gap_frames_instead_of_unbounded_queueing`
//!      — 本物の denno-view バイナリを、テスト内のモック上流 WS に繋ぎ、
//!      20,000 枚の Prefill フレームを一気に流す。クライアント (テスト) は
//!      洪水の間ずっと読まない。ブロードキャストが有界なら購読が遅れて
//!      `RecvError::Lagged` になり、relay は `Frame::Gap { lost }` を返す。
//!      読んだ Prefill 枚数 + Gap の lost 合計 = 流した枚数 (黙って落とさない)、
//!      かつ読めた枚数は全体より十分小さい (全部はバッファしない) ことを見る。
//!      スリープで待たず、`/status` の frames カウンタと WS の到着順だけで
//!      判定する。
//!
//!   2. 静的監査 — 配信アセットを `include_str!` で読み、次を固定する:
//!      - relay のブロードキャスト容量が小さな定数で、Lagged → Gap の枝がある
//!      - Gap ログは 1 秒に 1 回へ間引かれ、クライアント送信失敗でループを抜ける
//!      - 1D/2D/3D の全 `ws.onmessage` が decode/handler の例外を握る
//!        (`FrameDecodeError` を型で分岐し、外へ throw しない)
//!      - 3 ページとも Gap を表示する (3D は task 1641 で追加)
//!      - 記録・再生・LRU・キューに上限があり、古いものから落ちる
//!      - 2D の壁は rAF + dirty で最大 4 Hz、1D は 33 ms 間隔 + 溢れたら間引き、
//!        3D は rAF ループ (高レートでも描画はディスプレイ周期に丸める)
//!      - 外部 CDN 参照ゼロ / LAN IP・メール・/絶対パス・デバッグ出力なし

use std::sync::Arc;
use std::time::Duration;

use axum::extract::ws::Message as AxumMessage;
use axum::extract::{State, WebSocketUpgrade};
use axum::response::Response;
use axum::routing::get;
use axum::Router;
use denno_telemetry::Frame;
use futures_util::StreamExt;
use tokio::sync::{oneshot, Mutex};

const INDEX_1D: &str = include_str!("../src/index.html");
const INDEX_2D: &str = include_str!("../src/index2d.html");
const INDEX_3D: &str = include_str!("../src/index3d.html");
const MAIN_RS: &str = include_str!("../src/main.rs");
const FRAMES_MJS: &str = include_str!("../src/frames.mjs");
const VIEW2D_MJS: &str = include_str!("../src/view2d.mjs");
const RECONNECT_MJS: &str = include_str!("../src/reconnect.mjs");

// ---------------------------------------------------------------------------
// 1. relay のバックプレッシャ (実プロセス + モック上流)
// ---------------------------------------------------------------------------

/// 洪水の枚数と 1 枚のトークン数。40,000 x 約 530 B ≈ 21 MB で、socket バッファ
/// (このホストは wmem_max 4 MB + 既定 rcvbuf 128 KB、他ホストでも数十 MB には
/// 届かない) を確実に超える。
/// 1 メッセージ = 1 フレームなので、ブロードキャストの ring (1024 本) だけが
/// プロセス内に残り、あとは Lagged → Gap として落ちる。
const FLOOD_FRAMES: usize = 40_000;
const PREFILL_TOKENS: usize = 256;

#[derive(Clone)]
struct MockState {
    trigger: Arc<Mutex<Option<oneshot::Receiver<()>>>>,
    sent: Arc<Mutex<Option<oneshot::Sender<usize>>>>,
}

async fn mock_upstream_ws(ws: WebSocketUpgrade, State(st): State<MockState>) -> Response {
    ws.on_upgrade(move |mut socket| async move {
        // 接続直後に ServerInfo を 1 枚。denno-view はこれをスナップショットとして
        // 保持し、後から来たクライアントへ最初に配る。テストはこの 1 枚を
        // 受け取った時点で「購読が確立した」と判断できる (購読はスナップショット
        // 送信の直後、同じ poll 内で行われる)。
        let probe = Frame::ServerInfo {
            model: "mock".into(),
            model_id: "mock-id".into(),
            quant: "Q4_0".into(),
            slots: 1,
            context: 4096,
            layers: 4,
        }
        .encode();
        if socket.send(AxumMessage::Binary(probe)).await.is_err() {
            return;
        }
        // テストが洪水を許可するまで待つ (スリープ無しの明示同期)。
        let trigger = st.trigger.lock().await.take();
        if let Some(rx) = trigger {
            if rx.await.is_err() {
                return;
            }
        }
        let tokens: Vec<u32> = (0..PREFILL_TOKENS as u32).map(|i| 1000 + i).collect();
        let mut sent = 0usize;
        for i in 0..FLOOD_FRAMES as u32 {
            let bin = Frame::Prefill {
                session: 1,
                prompt_tokens: PREFILL_TOKENS as u32,
                ns: 1_000_000 + i as u64,
                tokens: tokens.clone(),
            }
            .encode();
            if socket.send(AxumMessage::Binary(bin)).await.is_err() {
                break;
            }
            sent += 1;
        }
        if let Some(tx) = st.sent.lock().await.take() {
            let _ = tx.send(sent);
        }
        // テストが読み終えるまで接続を開いたままにする。
        while let Some(Ok(_)) = socket.recv().await {}
    })
}

/// 起動中の denno-view。Drop で子プロセスを落とす。
struct ViewProc {
    child: std::process::Child,
    port: u16,
}

impl Drop for ViewProc {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

async fn wait_ready(port: u16) {
    let url = format!("http://127.0.0.1:{port}/healthz");
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    loop {
        if let Ok(r) = reqwest::get(&url).await {
            if r.status().is_success() {
                return;
            }
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "denno-view が 30 秒以内に /healthz を返さなかった"
        );
        tokio::task::yield_now().await;
    }
}

async fn status_frames(port: u16) -> u64 {
    let url = format!("http://127.0.0.1:{port}/status");
    let text = reqwest::get(&url)
        .await
        .expect("/status に接続できない")
        .text()
        .await
        .expect("/status 本文が読めない");
    let v: serde_json::Value =
        serde_json::from_str(&text).expect("/status が JSON でない");
    v["frames"].as_u64().unwrap_or(0)
}

async fn wait_frames_at_least(port: u16, want: u64) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(60);
    loop {
        if status_frames(port).await >= want {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "upstream の {want} 枚が 60 秒以内に pump へ届かなかった"
        );
        tokio::task::yield_now().await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn slow_subscriber_gets_gap_frames_instead_of_unbounded_queueing() {
    // --- モック上流 (WS) を立てる -----------------------------------------
    let (trigger_tx, trigger_rx) = oneshot::channel::<()>();
    let (sent_tx, sent_rx) = oneshot::channel::<usize>();
    let mock_state = MockState {
        trigger: Arc::new(Mutex::new(Some(trigger_rx))),
        sent: Arc::new(Mutex::new(Some(sent_tx))),
    };
    let router = Router::new()
        .route("/v1/telemetry/ws", get(mock_upstream_ws))
        .with_state(mock_state);
    // 先に bind してポートを確定させる (listen 済みなので spawn 後の取りこぼしが無い)。
    let std_listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind mock upstream");
    std_listener.set_nonblocking(true).expect("nonblocking");
    let mock_port = std_listener.local_addr().unwrap().port();
    let tokio_listener =
        tokio::net::TcpListener::from_std(std_listener).expect("tokio listener");
    tokio::spawn(async move {
        let _ = axum::serve(tokio_listener, router).await;
    });

    // --- 本物の denno-view を空きポートで起動 -----------------------------
    let probe = std::net::TcpListener::bind("127.0.0.1:0").expect("bind view port");
    let view_port = probe.local_addr().unwrap().port();
    drop(probe);
    let child = std::process::Command::new(env!("CARGO_BIN_EXE_denno-view"))
        .env("VIEW_PORT", view_port.to_string())
        .env(
            "VIEW_UPSTREAM",
            format!("ws://127.0.0.1:{mock_port}/v1/telemetry/ws"),
        )
        .env_remove("VIEW_CAPTURE")
        .env_remove("VIEW_REPLAY")
        .env_remove("VIEW_TOKENIZER")
        .env_remove("VIEW_RECORD")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn denno-view");
    let view = ViewProc {
        child,
        port: view_port,
    };
    wait_ready(view.port).await;

    // --- クライアントを繋ぎ、スナップショット 1 枚を読む ------------------
    // この 1 枚を受け取れた = view_socket が subscribe を済ませた、と確定できる
    // (スナップショット送信の直後、同じ poll で subscribe する)。
    let ws_url = format!("ws://127.0.0.1:{}/ws", view.port);
    let (mut client, _resp) = tokio_tungstenite::connect_async(&ws_url)
        .await
        .expect("connect /ws");
    let first = tokio::time::timeout(Duration::from_secs(10), client.next())
        .await
        .expect("初期スナップショットが来ない")
        .expect("WS が閉じた")
        .expect("WS メッセージエラー");
    match first {
        tokio_tungstenite::tungstenite::Message::Binary(bin) => {
            let f = Frame::decode(&bin).expect("初期スナップショットが壊れている");
            assert!(
                matches!(f, Frame::ServerInfo { .. }),
                "最初は ServerInfo スナップショット"
            );
        }
        other => panic!("最初のメッセージがバイナリでない: {other:?}"),
    }

    // --- 洪水を許可して、pump が全部受け取るまで待つ ---------------------
    trigger_tx.send(()).expect("trigger mock upstream");
    let sent = tokio::time::timeout(Duration::from_secs(120), sent_rx)
        .await
        .expect("モックが洪水を送り終えない")
        .expect("sent 通知が落ちた");
    assert_eq!(sent, FLOOD_FRAMES, "モックが送った枚数");
    // probe 1 枚 + 洪水 sent 枚。
    wait_frames_at_least(view.port, sent as u64 + 1).await;

    // --- 遅い購読者の受信を読む ------------------------------------------
    // 各メッセージは「配達されたフレーム」か「Lagged の lost 合計」のどちらかに
    // 必ず入る。配達 + lost = 洪水枚数 になったら終わり (スリープ不要)。
    let mut delivered = 0usize;
    let mut lost: u64 = 0;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(120);
    while delivered as u64 + lost < sent as u64 {
        let msg = tokio::time::timeout_at(deadline, client.next())
            .await
            .unwrap_or_else(|_| {
                panic!("受信が終わらない: delivered={delivered} lost={lost} sent={sent}")
            })
            .expect("WS が閉じた")
            .expect("WS メッセージエラー");
        match msg {
            tokio_tungstenite::tungstenite::Message::Binary(bin) => match Frame::decode(&bin) {
                Ok(Frame::Gap { lost: n }) => lost += n as u64,
                Ok(Frame::Prefill { .. }) => delivered += 1,
                Ok(_) => {}
                Err(e) => panic!("relay が壊れたフレームを送った: {e}"),
            },
            tokio_tungstenite::tungstenite::Message::Close(_) => panic!("洪水の途中で WS が閉じた"),
            _ => {}
        }
    }

    assert_eq!(
        delivered as u64 + lost,
        sent as u64,
        "配達 + Gap の lost が洪水枚数と一致しない (黙って落ちている)"
    );
    assert!(
        lost > 0,
        "40,000 枚 (約 21 MB) を読まずに受けたのに Gap が出ない = 無界にバッファしている"
    );
    assert!(
        delivered < sent / 2,
        "遅い購読者が {delivered}/{sent} 枚も受けた (バッファが有界でない疑い)"
    );

    // 洪水後もプロセスは生きている (panic していない)。
    let h = reqwest::get(format!("http://127.0.0.1:{}/healthz", view.port))
        .await
        .expect("healthz");
    assert!(h.status().is_success(), "洪水後に denno-view が落ちた");
}

// ---------------------------------------------------------------------------
// 2. 静的監査
// ---------------------------------------------------------------------------

/// `ws.onmessage = ...` の本体 (外側の波括弧の中身) を切り出す。
/// 文字列リテラルと行コメントは飛ばして波括弧を数える。
fn ws_onmessage_body<'a>(src: &'a str, name: &str) -> &'a str {
    let at = src
        .find("ws.onmessage")
        .unwrap_or_else(|| panic!("{name}: ws.onmessage が無い"));
    let open = src[at..]
        .find('{')
        .map(|i| at + i)
        .unwrap_or_else(|| panic!("{name}: onmessage の本体が無い"));
    let bytes = src.as_bytes();
    let mut depth = 0usize;
    let mut i = open;
    let mut quote: Option<u8> = None;
    while i < bytes.len() {
        let c = bytes[i];
        if let Some(q) = quote {
            if c == b'\\' {
                i += 2;
                continue;
            }
            if c == q {
                quote = None;
            }
            i += 1;
            continue;
        }
        match c {
            b'\'' | b'"' | b'`' => quote = Some(c),
            b'/' if i + 1 < bytes.len() && bytes[i + 1] == b'/' => {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
                continue;
            }
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &src[open..=i];
                }
            }
            _ => {}
        }
        i += 1;
    }
    panic!("{name}: onmessage の波括弧が閉じない");
}

/// HTML の `attr="..."` / `attr='...'` を集める (assets_audit_1309 と同じ素朴な走査)。
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

/// ES の `from "..."` / `import("...")` の指定子を集める。
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

fn is_external_network(v: &str) -> bool {
    let v = v.trim();
    v.starts_with("http://") || v.starts_with("https://") || v.starts_with("//")
}

#[test]
fn relay_broadcast_is_bounded_and_lagged_turns_into_gap() {
    // main() の本番チャネル (最初の broadcast::channel 呼び出し) の容量を読む。
    let at = MAIN_RS
        .find("broadcast::channel::<Vec<u8>>(")
        .expect("relay のブロードキャストが無い");
    let rest = &MAIN_RS[at + "broadcast::channel::<Vec<u8>>(".len()..];
    let cap: usize = rest
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .expect("ブロードキャスト容量が数値でない");
    assert!(
        (2..=4096).contains(&cap),
        "ブロードキャスト容量 {cap} が有界でない (高レート洪水でメモリが伸びる)"
    );

    // 購読遅れ → Gap フレーム。黙って捨てない。
    assert!(
        MAIN_RS.contains("RecvError::Lagged(n)"),
        "Lagged の枝が無い (購読遅れが Gap にならない)"
    );
    assert!(
        MAIN_RS.contains("Frame::Gap { lost: n as u32 }"),
        "Lagged を Gap フレームに変換していない"
    );
    // Gap ログの洪水を防ぐ間引き (1 秒に 1 回)。
    assert!(
        MAIN_RS.contains("gap_pending") && MAIN_RS.contains("from_secs(1)"),
        "Gap ログの間引きが無い (洪水でログが爆発する)"
    );

    // クライアントへの送信失敗で購読ループを抜ける (詰まったソケットを抱え続けない)。
    let ws = MAIN_RS
        .find("async fn view_socket")
        .expect("view_socket が無い");
    let ws_body = &MAIN_RS[ws..];
    assert!(
        ws_body.contains("socket.send(Message::Binary(bin)).await.is_err()")
            && ws_body.contains("break;"),
        "view_socket が送信失敗で抜けない"
    );

    // /ws は 1 メッセージ = 1 フレームで流す (クライアント側の結合バッファを増やさない)。
    assert!(
        MAIN_RS.contains("let _ = bcast.send(bin);"),
        "pump がフレームを broadcast へ流していない"
    );
}

#[test]
fn every_ws_onmessage_handler_contains_its_exceptions() {
    for (name, src) in [
        ("index.html", INDEX_1D),
        ("index2d.html", INDEX_2D),
        ("index3d.html", INDEX_3D),
    ] {
        let body = ws_onmessage_body(src, name);
        assert!(
            body.contains("try {") && body.contains("catch ("),
            "{name}: onmessage に try/catch が無い (例外が uncaught になる)"
        );
        if name != "index.html" {
            // 1D は decode 全体を 1 つの catch で扱う (型分岐はしない)。
            assert!(
                body.contains("FrameDecodeError"),
                "{name}: decode エラーを型で分岐していない"
            );
        }
        assert!(
            !body.contains("throw "),
            "{name}: onmessage が例外を外へ投げている"
        );
    }

    // 2D はハンドラ呼び出しも握る (decode は通ったが値が壊れている場合)。
    let two_d = ws_onmessage_body(INDEX_2D, "index2d.html");
    assert!(
        two_d.contains("try {\n        handleFrame(r.frame);"),
        "index2d.html: handleFrame が onmessage の try の外にある"
    );
}

#[test]
fn all_three_views_surface_gap_frames() {
    // 1D: 累計して表示
    assert!(
        INDEX_1D.contains("f.kind === \"Gap\"") && INDEX_1D.contains("gapCount += f.lost"),
        "index.html が Gap を数えていない"
    );
    // 2D: 状態表示に出す
    assert!(
        INDEX_2D.contains("f.kind === 'Gap'") && INDEX_2D.contains("frames lost"),
        "index2d.html が Gap を表示していない"
    );
    // 3D: task 1641 で追加。累計を HUD に出す。
    assert!(
        INDEX_3D.contains("case 'Gap':"),
        "index3d.html が Gap を処理していない"
    );
    assert!(
        INDEX_3D.contains("id=\"gap_lost\"") && INDEX_3D.contains("liveState.gapLost"),
        "index3d.html に Gap の累計表示が無い"
    );
    // デコーダとトレースにも Gap がある (記録から消えない)。
    assert!(
        FRAMES_MJS.contains("GAP: 9") && FRAMES_MJS.contains("case KIND.GAP"),
        "frames.mjs が Gap を decode/encode しない"
    );
    assert!(
        VIEW2D_MJS.contains("case 'Gap':") && VIEW2D_MJS.contains("lost: tInt(f.lost, 0)"),
        "view2d.mjs の trace が Gap を記録しない"
    );
}

#[test]
fn viewer_buffers_and_queues_are_capped() {
    // 2D のトレース: 上限枚数で古いものから落とし、落とした数を数える。
    assert!(
        VIEW2D_MJS.contains("export const TRACE_MAX_FRAMES = 2048"),
        "TRACE_MAX_FRAMES が無い"
    );
    assert!(
        VIEW2D_MJS.contains("while (rec.frames.length > rec.maxFrames)"),
        "traceRecord が上限で落とさない"
    );
    assert!(
        VIEW2D_MJS.contains("rec.dropped++"),
        "traceRecord が落とした数を数えない"
    );
    assert!(
        VIEW2D_MJS.contains("clampInt(tInt(opts.maxFrames, TRACE_MAX_FRAMES), 1, 1 << 20)"),
        "maxFrames が clamp されていない"
    );
    // 1 フレームの配列にも上限 (層数に依らず有界)。
    assert!(
        VIEW2D_MJS.contains("TRACE_MAX_ARRAY") && VIEW2D_MJS.contains("TRACE_MAX_PAIRS"),
        "trace の配列上限が無い"
    );

    // 1D の再生キュー: 溢れたら LayerRun を間引いて追いつく。
    assert!(
        INDEX_1D.contains("queue.length > 240"),
        "index.html の再生キューに上限が無い"
    );
    assert!(
        INDEX_1D.contains("if (f.kind === \"LayerRun\") { if (layerKept++ < 48) keep.unshift(f); }"),
        "index.html の間引きが LayerRun に限定されていない"
    );

    // 3D: expert の LRU と転送履歴に上限。
    assert!(
        INDEX_3D.contains("while (lru.length > VRAM_CAP)")
            && INDEX_3D.contains("while (lruRam.length > RAM_CAP)"),
        "index3d.html の expert LRU に上限が無い"
    );
    assert!(
        INDEX_3D.contains("if (transfers.length >= TRANS_N) transfers.shift();"),
        "index3d.html の転送履歴に上限が無い"
    );
}

#[test]
fn renderers_are_coalesced_to_display_rate() {
    // 2D: rAF ループ + dirty。重い再描画は 250 ms に 1 回 (4 Hz) まで。
    assert!(
        INDEX_2D.contains("requestAnimationFrame(tick)"),
        "index2d.html に rAF ループが無い"
    );
    assert!(
        INDEX_2D.contains("now - lastTreeRender > 250"),
        "index2d.html に重い再描画のレート制限が無い"
    );
    assert!(
        INDEX_2D.contains("if (!dirty || !wall) return;"),
        "index2d.html が dirty で描画を間引いていない"
    );
    // 1D: 33 ms タイマで drain (1 メッセージずつ、pacing 付き)。
    assert!(
        INDEX_1D.contains("setInterval(drain, 33)"),
        "index.html の drain がタイマ駆動でない"
    );
    // 3D: rAF ループ。
    assert!(
        INDEX_3D.contains("requestAnimationFrame(frame)"),
        "index3d.html に rAF ループが無い"
    );
}

#[test]
fn decode_has_byte_budget_guards() {
    // count() は残りバイトから要素数の上限を見る (過大 count で巨大配列を確保しない)。
    assert!(
        FRAMES_MJS.contains("n > Math.floor((this.b.length - this.p) / minElem)"),
        "frames.mjs の count() にバイト上限が無い"
    );
    // 端数は partial=true で返し、呼び手が次のメッセージへ繰り越せる。
    assert!(
        FRAMES_MJS.contains("this.start, true)"),
        "frames.mjs が partial を区別しない"
    );
    assert!(
        FRAMES_MJS.contains("this.partial = partial;"),
        "frames.mjs が partial を運ばない"
    );
    // 未知 kind は読み進めずに落とす (後続を壊さない)。
    assert!(
        FRAMES_MJS.contains("unknown kind: ' + kind"),
        "frames.mjs が未知 kind を弾かない"
    );
}

#[test]
fn served_assets_reference_no_external_cdn() {
    let mut refs = 0usize;
    for (name, src, is_html) in [
        ("index.html", INDEX_1D, true),
        ("index2d.html", INDEX_2D, true),
        ("index3d.html", INDEX_3D, true),
        ("frames.mjs", FRAMES_MJS, false),
        ("view2d.mjs", VIEW2D_MJS, false),
        ("reconnect.mjs", RECONNECT_MJS, false),
    ] {
        if is_html {
            for attr in ["src", "href"] {
                for v in attr_values(src, attr) {
                    refs += 1;
                    assert!(
                        !is_external_network(v),
                        "{name}: {attr} が外部を指す: {v:?}"
                    );
                }
            }
        }
        for spec in import_specifiers(src) {
            refs += 1;
            assert!(
                !is_external_network(&spec),
                "{name}: import が外部を指す: {spec:?}"
            );
        }
    }
    assert!(refs >= 10, "参照を {refs} 個しか見ていない (抽出漏れ)");
    // 3D の three.js は同梱 vendor から読む (2D は canvas 描画で three を使わない)。
    assert!(
        INDEX_3D.contains("/vendor/three.module.js"),
        "index3d.html: three がローカル vendor でない"
    );
}

#[test]
fn served_assets_have_no_private_hosts_or_debug_prints() {
    fn has_lan_ip(s: &str) -> bool {
        for prefix in ["192.168.", "172.16.", "172.17.", "172.18.", "10."] {
            if let Some(pos) = s.find(prefix) {
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

    for (name, src) in [
        ("index.html", INDEX_1D),
        ("index2d.html", INDEX_2D),
        ("index3d.html", INDEX_3D),
        ("frames.mjs", FRAMES_MJS),
        ("view2d.mjs", VIEW2D_MJS),
        ("reconnect.mjs", RECONNECT_MJS),
        ("main.rs", MAIN_RS),
    ] {
        assert!(!src.contains("/home/"), "{name}: 絶対パス (/home/) を含む");
        assert!(
            !src.contains("mailto:") && !src.contains("@localhost"),
            "{name}: 連絡先/ホスト名らしき文字列を含む"
        );
        assert!(!has_lan_ip(src), "{name}: LAN IP を含む");
        for dbg in ["console.log(", "debugger", "eprintln!", "dbg!"] {
            assert!(!src.contains(dbg), "{name}: デバッグ出力 {dbg:?} を含む");
        }
    }
}
