//! serve-flash 用の推論テレメトリ (P51)。
//!
//! 27B の `denno serve` は axum + axum の WebSocketUpgrade で
//! `GET /v1/telemetry/ws` を出している (`denno-server/src/lib.rs`) が、
//! serve-flash は axum を使わない自前の TCP ループなので、ここで WebSocket
//! (RFC 6455) のサーバー側を最小実装する。
//!
//! 構造 (推論を絶対に待たせない):
//!
//! - 推論スレッド → `emit()` → bounded mpsc への `try_send`。満杯は捨てて
//!   `dropped` に計上する (denno-telemetry のハブと同じ規律)。
//! - relay スレッド 1 本 → mpsc から 1 フレームずつ取り、登録済みクライアントへ
//!   バイナリフレームで書く。書き込みは 1 クライアント 200ms で打ち切る。
//! - クライアントごとに読み捨てスレッド 1 本 (切断検知 + 受信バイトの廃棄)。
//!
//! 無効化は `DENNO_TELEMETRY=0` (serve-flash の既定は有効)。無効時は
//! relay スレッドもチャネルも作らず、推論側は `on()` の bool 1 個を見るだけ。
//!
//! フレームの形式は `denno-telemetry` の `Frame::encode` をそのまま使う
//! (新規定義は無い)。

use std::io::{BufReader, Read, Write};
use std::net::{Shutdown, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use anyhow::Result;
use denno_telemetry::Frame;

/// フレームチャネルの容量 (フレーム数)。denno-telemetry のハブと同じ 4096。
const CHANNEL_CAP: usize = 4096;
/// 1 クライアントへの書き込みで待つ上限。遅いクライアントで全体を止めない。
const WRITE_TIMEOUT: Duration = Duration::from_millis(200);
/// relay がチャネルを見に行く間隔。
const RELAY_POLL: Duration = Duration::from_millis(50);
/// RFC 6455 のマジック文字列。
const WS_GUID: &str = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

/// 登録済みクライアント (書き込み側のソケット)。
struct Client {
    id: u64,
    stream: TcpStream,
}

/// relay スレッドと読み捨てスレッドが共有する状態。
struct Shared {
    clients: Mutex<Vec<Client>>,
    next_id: AtomicU64,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// テレメトリの送信側 + WS クライアント登録。
pub struct Telemetry {
    on: bool,
    /// WS 接続直後に 1 回だけ送る ServerInfo。
    info: Frame,
    /// 無効時は None (チャネル自体を作らない)。
    tx: Option<SyncSender<Frame>>,
    dropped: Arc<AtomicU64>,
    shared: Arc<Shared>,
}

impl Telemetry {
    pub fn new(on: bool, info: Frame) -> (Self, Option<Receiver<Frame>>) {
        Self::with_capacity(CHANNEL_CAP, on, info)
    }

    fn with_capacity(cap: usize, on: bool, info: Frame) -> (Self, Option<Receiver<Frame>>) {
        let shared = Arc::new(Shared {
            clients: Mutex::new(Vec::new()),
            next_id: AtomicU64::new(1),
        });
        let dropped = Arc::new(AtomicU64::new(0));
        if !on {
            return (
                Telemetry { on: false, info, tx: None, dropped, shared },
                None,
            );
        }
        let (tx, rx) = sync_channel(cap);
        (
            Telemetry { on: true, info, tx: Some(tx), dropped, shared },
            Some(rx),
        )
    }

    pub fn on(&self) -> bool {
        self.on
    }

    /// 推論側から 1 フレーム送る。**待たない**。満杯/切断は捨てて計上する。
    pub fn emit(&self, f: Frame) {
        let Some(tx) = self.tx.as_ref() else {
            return;
        };
        match tx.try_send(f) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    #[cfg(test)]
    fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// relay スレッドを 1 本立てる。無効時は何もしない。
    pub fn start_relay(&self, rx: Receiver<Frame>) -> std::io::Result<()> {
        if !self.on {
            return Ok(());
        }
        let shared = self.shared.clone();
        let dropped = self.dropped.clone();
        std::thread::Builder::new()
            .name("flash-telemetry".into())
            .spawn(move || {
                let mut last_dropped = 0u64;
                loop {
                    // 捨てた分は黙って飛ばさず Gap で明示する
                    let now = dropped.load(Ordering::Relaxed);
                    if now > last_dropped {
                        let lost = (now - last_dropped).min(u32::MAX as u64) as u32;
                        last_dropped = now;
                        broadcast(&shared, &Frame::Gap { lost }.encode());
                    }
                    match rx.recv_timeout(RELAY_POLL) {
                        Ok(f) => broadcast(&shared, &f.encode()),
                        Err(RecvTimeoutError::Timeout) => continue,
                        Err(RecvTimeoutError::Disconnected) => break,
                    }
                }
            })
            .map(|_| ())
    }

    /// `GET /v1/telemetry/ws` の Upgrade を処理する。
    ///
    /// 101 を返し、ServerInfo を 1 フレーム送ってからクライアントを登録する。
    /// `reader` は読み捨てスレッドに渡す (ハンドシェイク時に既に読んだバイトが
    /// BufReader に残っていても捨てずに済む)。
    pub fn accept(
        &self,
        mut stream: TcpStream,
        reader: BufReader<TcpStream>,
        key: &str,
    ) -> Result<()> {
        if !self.on {
            return send_plain(
                &mut stream,
                "503 Service Unavailable",
                "telemetry disabled (set DENNO_TELEMETRY=1 or unset DENNO_TELEMETRY)\n",
            );
        }
        let accept = accept_key(key);
        write!(
            stream,
            "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {accept}\r\n\r\n"
        )?;
        stream.flush()?;
        stream.set_nodelay(true).ok();
        // 遅いクライアントが relay スレッドを止めないように上限を付ける
        stream.set_write_timeout(Some(WRITE_TIMEOUT)).ok();
        let info = self.info.encode();
        let mut msg = Vec::with_capacity(info.len() + 10);
        push_ws_binary(&mut msg, &info);
        stream.write_all(&msg)?;
        stream.flush()?;

        let id = self.shared.next_id.fetch_add(1, Ordering::Relaxed);
        let writer = stream.try_clone()?;
        lock(&self.shared.clients).push(Client { id, stream: writer });
        spawn_reader(id, reader, stream, self.shared.clone());
        Ok(())
    }
}

/// クライアント→サーバーは読むだけ (close も ping も応答しない)。切断を
/// 検知したらソケットを落として relay 側の書き込みを失敗させ、登録を外す。
fn spawn_reader(
    id: u64,
    mut reader: BufReader<TcpStream>,
    stream: TcpStream,
    shared: Arc<Shared>,
) {
    let _ = std::thread::Builder::new()
        .name(format!("flash-telem-rx-{id}"))
        .spawn(move || {
            let mut buf = [0u8; 4096];
            loop {
                match reader.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {}
                }
            }
            let _ = stream.shutdown(Shutdown::Both);
            lock(&shared.clients).retain(|c| c.id != id);
        });
}

/// 1 フレームを全クライアントへ書く。書けなかったクライアントは登録を外す。
/// ロックはソケット複製の間だけ取り、書き込みはロック外で行う。
fn broadcast(shared: &Shared, payload: &[u8]) {
    let mut msg = Vec::with_capacity(payload.len() + 10);
    push_ws_binary(&mut msg, payload);
    let targets: Vec<(u64, TcpStream)> = {
        let clients = lock(&shared.clients);
        clients
            .iter()
            .filter_map(|c| c.stream.try_clone().ok().map(|s| (c.id, s)))
            .collect()
    };
    if targets.is_empty() {
        return;
    }
    let mut dead: Vec<u64> = Vec::new();
    for (id, mut s) in targets {
        if s.write_all(&msg).and_then(|_| s.flush()).is_err() {
            dead.push(id);
        }
    }
    if !dead.is_empty() {
        lock(&shared.clients).retain(|c| !dead.contains(&c.id));
    }
}

fn send_plain(stream: &mut TcpStream, status: &str, body: &str) -> Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )?;
    stream.flush()?;
    Ok(())
}

/// サーバー→クライアントの WS フレーム (FIN + binary、マスク無し)。
fn push_ws_binary(out: &mut Vec<u8>, payload: &[u8]) {
    out.push(0x82);
    let n = payload.len();
    if n < 126 {
        out.push(n as u8);
    } else if n <= u16::MAX as usize {
        out.push(126);
        out.extend_from_slice(&(n as u16).to_be_bytes());
    } else {
        out.push(127);
        out.extend_from_slice(&(n as u64).to_be_bytes());
    }
    out.extend_from_slice(payload);
}

/// `Sec-WebSocket-Key` から `Sec-WebSocket-Accept` を計算する
/// (base64(SHA1(key + GUID)))。
pub fn accept_key(key: &str) -> String {
    let mut s = String::with_capacity(key.len() + WS_GUID.len());
    s.push_str(key);
    s.push_str(WS_GUID);
    base64(&sha1(s.as_bytes()))
}

/// SHA-1 (FIPS 180-1)。依存を増やさないため自前実装。
fn sha1(msg: &[u8]) -> [u8; 20] {
    let mut h: [u32; 5] = [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476, 0xc3d2_e1f0];
    let bit_len = (msg.len() as u64).wrapping_mul(8);
    let mut data = Vec::with_capacity(msg.len() + 72);
    data.extend_from_slice(msg);
    data.push(0x80);
    while data.len() % 64 != 56 {
        data.push(0);
    }
    data.extend_from_slice(&bit_len.to_be_bytes());
    debug_assert_eq!(data.len() % 64, 0);
    for chunk in data.chunks(64) {
        let mut w = [0u32; 80];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[4 * i],
                chunk[4 * i + 1],
                chunk[4 * i + 2],
                chunk[4 * i + 3],
            ]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }
        let (mut a, mut b, mut c, mut d, mut e) = (h[0], h[1], h[2], h[3], h[4]);
        for (i, wi) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | ((!b) & d), 0x5a82_7999u32),
                20..=39 => (b ^ c ^ d, 0x6ed9_eba1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8f1b_bcdc),
                _ => (b ^ c ^ d, 0xca62_c1d6),
            };
            let tmp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(*wi);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = tmp;
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
    }
    let mut out = [0u8; 20];
    for (i, v) in h.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&v.to_be_bytes());
    }
    out
}

/// 標準 base64 (パディングあり)。
fn base64(input: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(input.len().div_ceil(3) * 4);
    for c in input.chunks(3) {
        let n = ((c[0] as u32) << 16)
            | ((*c.get(1).unwrap_or(&0) as u32) << 8)
            | (*c.get(2).unwrap_or(&0) as u32);
        s.push(T[(n >> 18) as usize & 63] as char);
        s.push(T[(n >> 12) as usize & 63] as char);
        if c.len() > 1 {
            s.push(T[(n >> 6) as usize & 63] as char);
        } else {
            s.push('=');
        }
        if c.len() > 2 {
            s.push(T[n as usize & 63] as char);
        } else {
            s.push('=');
        }
    }
    s
}

// ---------- テスト ----------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::BufRead;
    use std::net::TcpListener;

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    fn info_frame() -> Frame {
        Frame::ServerInfo {
            model: "qwen3.8-flash-next".into(),
            model_id: "flash-test".into(),
            quant: "gguf".into(),
            slots: 1,
            context: 4096,
            layers: 48,
        }
    }

    /// 接続済みの (サーバー側, クライアント側) ソケット。
    fn pair() -> (TcpStream, TcpStream) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let cli = TcpStream::connect(addr).unwrap();
        let (srv, _) = l.accept().unwrap();
        cli.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        (srv, cli)
    }

    /// クライアント側で 1 個の WS バイナリフレームを読む (サーバー→クライアントは
    /// マスクされない)。101 のヘッダとフレームを同じ BufReader で読むこと
    /// (別の BufReader を作ると先読みしたフレームを落とす)。
    fn read_ws_frame(r: &mut BufReader<TcpStream>) -> Vec<u8> {
        let mut h = [0u8; 2];
        r.read_exact(&mut h).unwrap();
        assert_eq!(h[0], 0x82, "FIN + binary でない: {:#x}", h[0]);
        assert_eq!(h[1] & 0x80, 0, "サーバー→クライアントはマスク禁止");
        let len = match h[1] & 0x7f {
            126 => {
                let mut b = [0u8; 2];
                r.read_exact(&mut b).unwrap();
                u16::from_be_bytes(b) as usize
            }
            127 => {
                let mut b = [0u8; 8];
                r.read_exact(&mut b).unwrap();
                u64::from_be_bytes(b) as usize
            }
            n => n as usize,
        };
        let mut p = vec![0u8; len];
        r.read_exact(&mut p).unwrap();
        p
    }

    /// 101 応答のヘッダを読み、Accept の値 (小文字化) を返す。
    fn read_101(r: &mut BufReader<TcpStream>) -> (String, Option<String>) {
        let mut status = String::new();
        r.read_line(&mut status).unwrap();
        let mut accept = None;
        loop {
            let mut line = String::new();
            if r.read_line(&mut line).unwrap() == 0 || line == "\r\n" {
                break;
            }
            let low = line.to_ascii_lowercase();
            if let Some(v) = low.strip_prefix("sec-websocket-accept:") {
                accept = Some(v.trim().to_string());
            }
        }
        (status, accept)
    }

    #[test]
    fn sha1_known_vectors() {
        assert_eq!(hex(&sha1(b"")), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
        assert_eq!(hex(&sha1(b"abc")), "a9993e364706816aba3e25717850c26c9cd0d89d");
        assert_eq!(
            hex(&sha1(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")),
            "84983e441c3bd26ebaae4aa1f95129e5e54670f1"
        );
        // 56 バイト境界 (パディングが 2 ブロックに跨る長さ)
        assert_eq!(
            hex(&sha1(&[b'a'; 64])),
            "0098ba824b5c16427bd7a1122a5a442a25ec644d"
        );
    }

    #[test]
    fn base64_vectors() {
        for (inp, want) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(base64(inp.as_bytes()), want, "base64({inp:?})");
        }
    }

    /// RFC 6455 §1.3 の例。
    #[test]
    fn accept_matches_rfc6455_example() {
        assert_eq!(
            accept_key("dGhlIHNhbXBsZSBub25jZQ=="),
            "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
        );
    }

    #[test]
    fn ws_frame_length_encoding() {
        for n in [0usize, 1, 125, 126, 1000, 65535, 65536] {
            let p = vec![7u8; n];
            let mut out = Vec::new();
            push_ws_binary(&mut out, &p);
            assert_eq!(out[0], 0x82);
            let (hdr, want_len) = if n < 126 {
                (2, out[1] as usize)
            } else if n <= 65535 {
                (4, u16::from_be_bytes([out[2], out[3]]) as usize)
            } else {
                let mut b = [0u8; 8];
                b.copy_from_slice(&out[2..10]);
                (10, u64::from_be_bytes(b) as usize)
            };
            assert_eq!(want_len, n, "len={n}");
            assert_eq!(out.len(), hdr + n);
            assert_eq!(&out[hdr..], &p[..]);
        }
    }

    #[test]
    fn handshake_returns_101_and_server_info() {
        let (tele, _rx) = Telemetry::new(true, info_frame());
        let (srv, cli) = pair();
        let reader = BufReader::new(srv.try_clone().unwrap());
        tele.accept(srv, reader, "dGhlIHNhbXBsZSBub25jZQ==").unwrap();
        let mut r = BufReader::new(cli.try_clone().unwrap());
        let (status, accept) = read_101(&mut r);
        assert!(status.starts_with("HTTP/1.1 101 Switching Protocols"), "{status}");
        assert_eq!(accept.as_deref(), Some("s3pplmbitxaq9kygzzhzrbk+xoo="));
        // 続く最初のフレームは ServerInfo
        let p = read_ws_frame(&mut r);
        assert_eq!(&p[..4], b"DNTL");
        assert_eq!(Frame::decode(&p).unwrap(), info_frame());
    }

    #[test]
    fn handshake_503_when_disabled() {
        let (tele, rx) = Telemetry::new(false, info_frame());
        assert!(rx.is_none(), "無効時はチャネルを作らない");
        let (srv, mut cli) = pair();
        let reader = BufReader::new(srv.try_clone().unwrap());
        tele.accept(srv, reader, "dGhlIHNhbXBsZSBub25jZQ==").unwrap();
        let mut buf = [0u8; 256];
        let n = cli.read(&mut buf).unwrap();
        let s = String::from_utf8_lossy(&buf[..n]).to_string();
        assert!(s.starts_with("HTTP/1.1 503 Service Unavailable"), "{s}");
        assert!(!s.to_ascii_lowercase().contains("sec-websocket-accept"));
    }

    #[test]
    fn disabled_telemetry_sends_nothing() {
        let (tele, _rx) = Telemetry::new(false, info_frame());
        tele.emit(Frame::Gap { lost: 1 });
        assert_eq!(tele.dropped(), 0, "無効時はカウンタにも触らない");
    }

    #[test]
    fn emit_drops_and_counts_when_full() {
        let (tele, _rx) = Telemetry::with_capacity(1, true, info_frame());
        tele.emit(Frame::Gap { lost: 1 });
        tele.emit(Frame::Gap { lost: 2 });
        assert_eq!(tele.dropped(), 1);
    }

    #[test]
    fn relay_broadcasts_to_all_clients() {
        let (tele, rx) = Telemetry::new(true, info_frame());
        let mut clis = Vec::new();
        for _ in 0..2 {
            let (srv, cli) = pair();
            let reader = BufReader::new(srv.try_clone().unwrap());
            tele.accept(srv, reader, "dGhlIHNhbXBsZSBub25jZQ==").unwrap();
            let mut r = BufReader::new(cli.try_clone().unwrap());
            let (status, _) = read_101(&mut r);
            assert!(status.starts_with("HTTP/1.1 101"), "{status}");
            let info = read_ws_frame(&mut r);
            assert_eq!(Frame::decode(&info).unwrap(), info_frame());
            clis.push(r);
        }
        tele.start_relay(rx.unwrap()).unwrap();
        let f = Frame::Token { session: 3, token: 42, pos: 7 };
        tele.emit(f.clone());
        for r in clis.iter_mut() {
            let p = read_ws_frame(r);
            assert_eq!(Frame::decode(&p).unwrap(), f);
        }
    }

    /// 切断したクライアントを落としても、残りには配信が続く。
    #[test]
    fn dead_client_is_dropped_and_others_still_get_frames() {
        let (tele, rx) = Telemetry::new(true, info_frame());
        let (srv_a, cli_a) = pair();
        let reader = BufReader::new(srv_a.try_clone().unwrap());
        tele.accept(srv_a, reader, "dGhlIHNhbXBsZSBub25jZQ==").unwrap();
        let mut r_a = BufReader::new(cli_a.try_clone().unwrap());
        let (status, _) = read_101(&mut r_a);
        assert!(status.starts_with("HTTP/1.1 101"), "{status}");
        let _ = read_ws_frame(&mut r_a);

        let (srv_b, cli_b) = pair();
        let reader = BufReader::new(srv_b.try_clone().unwrap());
        tele.accept(srv_b, reader, "dGhlIHNhbXBsZSBub25jZQ==").unwrap();
        let mut r_b = BufReader::new(cli_b.try_clone().unwrap());
        let (status, _) = read_101(&mut r_b);
        assert!(status.starts_with("HTTP/1.1 101"), "{status}");
        let _ = read_ws_frame(&mut r_b);

        drop(r_a); // A を切る (BufReader の複製も落とさないと EOF にならない)
        drop(cli_a);
        tele.start_relay(rx.unwrap()).unwrap();
        // A の読み捨てスレッドが EOF を拾うのを待つ
        std::thread::sleep(Duration::from_millis(100));
        for i in 0..3 {
            tele.emit(Frame::Token { session: 1, token: i, pos: i });
        }
        for i in 0..3 {
            let p = read_ws_frame(&mut r_b);
            assert_eq!(Frame::decode(&p).unwrap(), Frame::Token { session: 1, token: i, pos: i });
        }
    }

    /// リクエストの `Sec-WebSocket-Key` を読む (実クライアントは鍵を自分で作る)。
    /// 値は base64 なので小文字化してはいけない (Accept が変わる)。
    fn read_handshake_key(r: &mut BufReader<TcpStream>) -> String {
        let mut key = String::new();
        loop {
            let mut line = String::new();
            if r.read_line(&mut line).unwrap() == 0 || line == "\r\n" {
                break;
            }
            if let Some((k, v)) = line.split_once(':') {
                if k.trim().eq_ignore_ascii_case("sec-websocket-key") {
                    key = v.trim().to_string();
                }
            }
        }
        key
    }

    async fn next_msg<S>(ws: &mut S) -> tokio_tungstenite::tungstenite::Message
    where
        S: futures_util::Stream<
                Item = Result<
                    tokio_tungstenite::tungstenite::Message,
                    tokio_tungstenite::tungstenite::Error,
                >,
            > + Unpin,
    {
        use futures_util::StreamExt;
        tokio::time::timeout(Duration::from_secs(5), ws.next())
            .await
            .expect("フレームが来ない")
            .expect("ストリームが終了した")
            .expect("WS エラー")
    }

    /// 実クライアント (denno-view が使うのと同じ tokio-tungstenite) で 101 が通り、
    /// ServerInfo → Token の順にバイナリフレームが届くこと。
    #[tokio::test]
    async fn tungstenite_client_interop() {
        use tokio_tungstenite::tungstenite::Message;

        let (tele, rx) = Telemetry::new(true, info_frame());
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let srv_thread = std::thread::spawn(move || {
            let (srv, _) = l.accept().unwrap();
            let mut r = BufReader::new(srv.try_clone().unwrap());
            let key = read_handshake_key(&mut r);
            tele.accept(srv, r, &key).unwrap();
            tele.start_relay(rx.unwrap()).unwrap();
            tele.emit(Frame::Token { session: 2, token: 9, pos: 4 });
            tele
        });

        let url = format!("ws://{addr}/v1/telemetry/ws");
        let (mut ws, resp) = tokio_tungstenite::connect_async(&url).await.unwrap();
        assert_eq!(resp.status().as_u16(), 101);
        let want = [info_frame(), Frame::Token { session: 2, token: 9, pos: 4 }];
        for w in want {
            match next_msg(&mut ws).await {
                Message::Binary(b) => assert_eq!(Frame::decode(&b).unwrap(), w),
                other => panic!("バイナリフレームでない: {other:?}"),
            }
        }
        drop(ws);
        let _tele = srv_thread.join().unwrap();
    }
}
