//! s44: serve 経路の非同期 (ノンブロッキング) ロギング。
//!
//! ## 背景 (wedge 事例の根因・REPORT_NEXT17 §4)
//!
//! 旧実装 (`tracing_subscriber::fmt().init()`) はログ行を呼び出し元スレッドから
//! 同期的に stdout へ書く。stdout がパイプ (例: `denno serve ... | tee log`) の場合:
//!
//! 1. 消費者 (tee / 端末) が凍結するとパイプバッファ (64KB) が満杯になる
//! 2. 最初のログ書き込みスレッドは `std::io::stdout()` のグローバルロックを保持した
//!    まま `write()` で永久ブロックする (実測: wchan `anon_pipe_write`)
//! 3. 以後の全ログ書き込みはそのグローバルロック待ち (futex) で停止する
//!    (`log_requests` ミドルウェアは全リクエストで 1 行書くため、接続タスクごとに
//!    tokio ワーカーが停止する)
//! 4. ワーカー枯渇 → ランタイム全体が進行不能 (CPU 0%・プロセス生存・`/health`
//!    無応答 = 実機 wedge 事例と一致。パイプを排出すると即座に復旧することを実証)
//!
//! ## 対策
//!
//! ログ行を有界キュー (`try_send`・満杯時は破棄してカウント) に入れ、stdout への
//! 書き込みは専用スレッド 1 本だけが行う。サーバスレッド (tokio ワーカー・
//! スケジューラ) は stdout ロックやパイプ書き込みで決してブロックしない。
//!
//! - キュー満杯: 行を破棄し、カウント。復旧後に専用スレッドが
//!   `denno-logger: N log line(s) dropped ...` の 1 行で報告する (ログはベストエフォート)
//! - EPIPE (消費者死亡): 書き込みエラーを無視 (行喪失・クラッシュ無し)
//! - 専用スレッドが凍結したパイプで止まり続ける場合もあるが、それはログ 1 本の
//!   デーモンスレッドのみで、サーバの応答性には影響しない
//!
//! 数値経路・決定論への影響は構造的にゼロ (ロギングのみ・推論/サンプリングに触れない)。

use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::Arc;
use std::time::Duration;

use tracing_subscriber::fmt::MakeWriter;

/// キュー容量 (行数)。凍結中に破棄を始めるまでのバッファ。
/// 典型的な行 ~数百 byte を仮定した上限 ~数 MB 相当。
const QUEUE_CAPACITY: usize = 4096;

/// 専用スレッドがドロップ数を確認する間隔 (キューが空いている時のポーリング)。
const DROPS_POLL_INTERVAL: Duration = Duration::from_millis(500);

/// ログ出力先 (本番: stdout・テスト: 注入可能なシンク)。
type Sink = Box<dyn Write + Send>;

/// ワーカーハンドル間で共有する状態。
struct Shared {
    tx: SyncSender<Vec<u8>>,
    /// 専用スレッドと共有するドロップカウンタ
    dropped: Arc<AtomicUsize>,
}

/// 非同期ログライタ (`tracing_subscriber::fmt` の MakeWriter)。
/// `make_writer()` はイベントごとにハンドルを返し、その `write()` は
/// キューへの `try_send` のみを行う (ブロックしない)。
#[derive(Clone)]
pub struct NonBlockingStdout {
    shared: Arc<Shared>,
}

impl NonBlockingStdout {
    /// stdout を出力先として専用スレッドを起動する。
    pub fn spawn() -> Self {
        Self::spawn_with(Box::new(io::stdout()), QUEUE_CAPACITY)
    }

    /// 出力先と容量を指定して起動する (テスト用)。
    pub fn spawn_with(sink: Sink, capacity: usize) -> Self {
        let (tx, rx) = sync_channel::<Vec<u8>>(capacity.max(1));
        let dropped = Arc::new(AtomicUsize::new(0));
        let thread_dropped = dropped.clone();
        std::thread::Builder::new()
            .name("denno-logger".into())
            .spawn(move || logger_loop(rx, sink, thread_dropped))
            .expect("denno-logger スレッドの起動に失敗");
        Self { shared: Arc::new(Shared { tx, dropped }) }
    }

    /// 起動後に破棄されたログ行の累積 (ドロップ通知で消費される前の値)。
    /// 観測用。
    pub fn dropped(&self) -> usize {
        self.shared.dropped.load(Ordering::Relaxed)
    }
}

impl<'a> MakeWriter<'a> for NonBlockingStdout {
    type Writer = LogWriter;

    fn make_writer(&'a self) -> Self::Writer {
        LogWriter { shared: self.shared.clone() }
    }
}

/// イベント 1 件分の書き込みハンドル。`write()` はブロックしない。
pub struct LogWriter {
    shared: Arc<Shared>,
}

impl Write for LogWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self.shared.tx.try_send(buf.to_vec()) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                // キュー満杯: 行を破棄してカウント (後で専用スレッドが報告する)。
                // 決して呼び出し元をブロックしない — これが wedge 対策の本体。
                self.shared.dropped.fetch_add(1, Ordering::Relaxed);
            }
            Err(TrySendError::Disconnected(_)) => {
                // 専用スレッド死亡: 黙って捨てる (ロギングはベストエフォート)
            }
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// 専用スレッドのメインループ。キューから取り出してシンクへ書く (単一書き手。
/// 行のインターリーブは起きない)。
fn logger_loop(rx: Receiver<Vec<u8>>, mut sink: Sink, dropped: Arc<AtomicUsize>) {
    loop {
        match rx.recv_timeout(DROPS_POLL_INTERVAL) {
            Ok(line) => {
                // EPIPE 等の書き込みエラーは無視 (行喪失。決してパニックしない)
                let _ = sink.write_all(&line);
                let _ = sink.flush();
                report_drops(&mut sink, &dropped);
            }
            Err(RecvTimeoutError::Timeout) => {
                report_drops(&mut sink, &dropped);
            }
            Err(RecvTimeoutError::Disconnected) => {
                // 全ハンドルが drop された: 残りを吐き出して終了
                while let Ok(line) = rx.try_recv() {
                    let _ = sink.write_all(&line);
                }
                report_drops(&mut sink, &dropped);
                return;
            }
        }
    }
}

/// 溜まったドロップ数を 1 行で報告してカウンタをリセットする。
fn report_drops(sink: &mut Sink, dropped: &AtomicUsize) {
    let n = dropped.swap(0, Ordering::Relaxed);
    if n > 0 {
        let msg = format!(
            "denno-logger: {n} log line(s) dropped (stdout consumer stalled or queue full)\n"
        );
        let _ = sink.write_all(msg.as_bytes());
        let _ = sink.flush();
    }
}

/// serve のロギングを非同期ライタで初期化する (CLI から serve 起動時に呼ぶ)。
/// フィルタは従来経路と同一の規則 (RUST_LOG・未設定時は info)。
pub fn init_serve_logging() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let writer = NonBlockingStdout::spawn();
    tracing_subscriber::fmt().with_env_filter(filter).with_writer(writer).init();
}

// ---------- P877: アクセスログの形式 ----------

/// `--log-format json` のとき true。`log_requests` ミドルウェアが読む。
static ACCESS_JSON: AtomicBool = AtomicBool::new(false);

/// アクセスログを JSON 形式にする (CLI の `--log-format json`)。
/// 他のログ行 (起動・エラー) は従来のテキストのまま — アクセスログだけを
/// 機械可読にする (ログ収集側で 1 行 1 JSON として扱える)。
pub fn set_access_log_json(on: bool) {
    ACCESS_JSON.store(on, Ordering::Relaxed);
}

/// アクセスログが JSON 形式か。
pub fn access_log_json() -> bool {
    ACCESS_JSON.load(Ordering::Relaxed)
}

// ---------- task 1101: クエリ文字列の秘匿 ----------

/// 値の秘匿対象になるクエリパラメータ名 (小文字で比較)。
/// task 1101 の指定: `api_key` / `key` / `token` / `secret` / `authorization`。
/// 同じ意味の綴り (`apikey` / `api-key` / `access_token` など) も値ごと伏せる。
const SENSITIVE_QUERY_NAMES: &[&str] = &[
    "api_key",
    "apikey",
    "api-key",
    "key",
    "token",
    "access_token",
    "refresh_token",
    "secret",
    "client_secret",
    "authorization",
    "auth",
    "password",
];

/// 秘匿した値の置換文字列。
const REDACTED: &str = "REDACTED";

/// クエリパラメータ名が秘匿対象か (ASCII 大文字小文字を無視)。
pub fn is_sensitive_query_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    SENSITIVE_QUERY_NAMES.contains(&lower.as_str())
}

/// クエリ文字列 (`?` の後ろ) のうち、秘匿対象パラメータの値だけを
/// [`REDACTED`] に置き換える。名前・順序・区切り (`&`) はそのまま保つ。
/// 秘匿対象が無ければ入力と同一の文字列を返す。
pub fn redact_query(query: &str) -> String {
    query
        .split('&')
        .map(|pair| match pair.split_once('=') {
            Some((name, _)) if is_sensitive_query_name(name) => format!("{name}={REDACTED}"),
            _ => pair.to_string(),
        })
        .collect::<Vec<_>>()
        .join("&")
}

/// パス (+ クエリ) からアクセスログ用の文字列を作る。
/// クエリが無ければそのまま、あれば秘匿対象の値だけを伏せる。
/// 冪等 (伏せた後の `REDACTED` は再適用しても変わらない)。
pub fn redact_path_and_query(path: &str) -> String {
    match path.split_once('?') {
        Some((p, q)) => format!("{p}?{}", redact_query(q)),
        None => path.to_string(),
    }
}

/// 1 リクエストぶんの JSON アクセスログ行 (`--log-format json`)。
/// `log_requests` ミドルウェアが応答後に 1 行だけ出す。
///
/// task 911: `request_id` は応答ヘッダ `x-request-id` と同じ値
/// (クライアントが送った ID の無害化版、または生成した ID)。ログ行と
/// クライアント側の記録を突き合わせるための鍵になる。
///
/// task 1101: `path` はクエリ込み。秘匿対象パラメータの値は
/// [`redact_path_and_query`] で伏せる (ヘッダ値は決して載せない)。
/// 経過時間は `latency_ms` (task 1101 の新名) と `duration_ms`
/// (既存の収集系との互換のため同値で残す) の両方に入れる。
pub fn access_log_line(
    method: &str,
    path: &str,
    status: u16,
    duration_ms: f64,
    request_id: &str,
) -> String {
    let ts_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let path = redact_path_and_query(path);
    serde_json::json!({
        "ts_ms": ts_ms,
        "method": method,
        "path": path,
        "status": status,
        "latency_ms": duration_ms,
        "duration_ms": duration_ms,
        "request_id": request_id,
    })
    .to_string()
}

// ---------- テスト ----------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    /// テスト用シンク: ゲートが開くまで write_all をブロックし、
    /// 受け取った内容をバッファに蓄える (専用スレッド = ロガーが止まる)。
    struct GatedSink {
        state: Arc<Mutex<GatedState>>,
    }

    struct GatedState {
        open: bool,
        /// ロガースレッドが write に突入した (ブロック開始) 後に true
        entered: bool,
        buf: Vec<u8>,
    }

    impl GatedSink {
        fn new() -> (Self, Arc<Mutex<GatedState>>) {
            let state = Arc::new(Mutex::new(GatedState { open: false, entered: false, buf: Vec::new() }));
            (Self { state: state.clone() }, state)
        }
    }

    impl Write for GatedSink {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            {
                let mut st = self.state.lock().unwrap();
                st.entered = true;
            }
            // ゲートが開くまでポーリングで待つ (ロガースレッドを止める)
            loop {
                {
                    let st = self.state.lock().unwrap();
                    if st.open {
                        break;
                    }
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            let mut st = self.state.lock().unwrap();
            st.buf.extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    /// `pred` が true になるまで待つ (ロガースレッドの進行待ち)。
    fn wait_until(state: &Arc<Mutex<GatedState>>, pred: impl Fn(&GatedState) -> bool, secs: u64, what: &str) {
        let deadline = Instant::now() + Duration::from_secs(secs);
        loop {
            {
                let st = state.lock().unwrap();
                if pred(&st) {
                    return;
                }
            }
            assert!(
                Instant::now() < deadline,
                "ロガーが待ち条件に到達しない ({what})"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// バッファが `needle` を含むまで待つ (ロガースレッドの進行待ち)。
    fn wait_contains(state: &Arc<Mutex<GatedState>>, needle: &str, secs: u64) -> String {
        wait_until(state, |st| String::from_utf8_lossy(&st.buf).contains(needle), secs, needle);
        let st = state.lock().unwrap();
        String::from_utf8_lossy(&st.buf).to_string()
    }

    /// シンクが停止してキューが満杯でも write() はブロックせず、行を破棄して
    /// カウントする。シンク復旧後に破棄数が 1 行で報告される。
    /// (実機 wedge の再現条件「stdout 消費者の凍結」に対する契約)
    #[test]
    fn write_never_blocks_on_stalled_consumer() {
        let (sink, state) = GatedSink::new();
        let writer = NonBlockingStdout::spawn_with(Box::new(sink), 2);

        // 1 行目: ロガーが拾ってシンク内で停止するのを待つ (レース排除)
        writer
            .make_writer()
            .write_all(b"line-1 blocked in logger\n")
            .unwrap();
        wait_until(&state, |st| st.entered, 10, "line-1 の write 突入");

        // シンク停止中にさらに 3 行: 2-3 行目はキュー (容量 2 で満杯)、
        // 4 行目は破棄。全て即座に返る (ブロックしたらこのテストは timeout で落ちる)
        let t0 = Instant::now();
        writer.make_writer().write_all(b"line-2 queued\n").unwrap();
        writer.make_writer().write_all(b"line-3 queued\n").unwrap();
        writer.make_writer().write_all(b"line-4 dropped\n").unwrap();
        let elapsed = t0.elapsed();
        assert!(
            elapsed < Duration::from_secs(2),
            "write() がブロックした: {elapsed:?}"
        );
        assert_eq!(writer.dropped(), 1, "満杯分 1 行がカウントされる");

        // シンクを開放 → ロガーが 3 行を順に書き、ドロップを報告する
        state.lock().unwrap().open = true;
        let buf = wait_contains(&state, "line-3", 10);
        assert!(buf.contains("line-1 blocked in logger"));
        assert!(buf.contains("line-2 queued"));
        assert!(buf.contains("line-3 queued"));
        // 破棄された行は出ない
        assert!(!buf.contains("line-4 dropped\n"));
        let buf = wait_contains(&state, "dropped", 10);
        assert!(buf.contains("1 log line(s) dropped"));
        // 報告後はカウンタがリセットされる
        assert_eq!(writer.dropped(), 0);

        // 復旧後の書き込みは普通に流れる
        writer.make_writer().write_all(b"line-5 after recovery\n").unwrap();
        let buf = wait_contains(&state, "line-5", 10);
        assert!(buf.contains("line-5 after recovery"));
    }

    /// シンクの書き込みエラー (EPIPE 相当) は無視される: パニックもブロックも無く、
    /// 後続の書き込みも継続する。
    #[test]
    fn sink_write_errors_are_swallowed() {
        struct ErrorSink;
        impl Write for ErrorSink {
            fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
                Err(io::Error::from_raw_os_error(32)) // EPIPE (Linux)
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        let writer = NonBlockingStdout::spawn_with(Box::new(ErrorSink), 8);
        for i in 0..16 {
            writer
                .make_writer()
                .write_all(format!("line-{i}\n").as_bytes())
                .unwrap();
        }
        // ロガーが 16 行を消化するのを十分待って、さらに書いてもパニックしないことで
        // スレッドが健常なことを確認
        std::thread::sleep(Duration::from_millis(300));
        writer.make_writer().write_all(b"after errors\n").unwrap();
        assert_eq!(writer.dropped(), 0);
    }

    /// P877: `--log-format json` のゲートと、1 行 JSON の中身。
    #[test]
    fn access_log_json_toggle_and_line_shape() {
        set_access_log_json(true);
        assert!(access_log_json());
        let line = access_log_line("POST", "/v1/messages", 200, 12.5, "req-000000000001");
        let v: serde_json::Value = serde_json::from_str(&line).expect("JSON として読めない");
        assert_eq!(v["method"], "POST");
        assert_eq!(v["path"], "/v1/messages");
        assert_eq!(v["status"], 200);
        assert_eq!(v["request_id"], "req-000000000001");
        assert!((v["latency_ms"].as_f64().unwrap() - 12.5).abs() < 1e-9);
        assert!((v["duration_ms"].as_f64().unwrap() - 12.5).abs() < 1e-9);
        assert!(v["ts_ms"].as_u64().unwrap() > 0);
        // 既定 (text) に戻す — 同じプロセスの他テストに漏らさない。
        set_access_log_json(false);
        assert!(!access_log_json());
    }
}
