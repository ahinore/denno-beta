//! Task 1019: `logging.rs` のエッジケース (壊れ入力・巨大行・並行書き込み)。
//!
//! 対象 API (crates/denno-server/src/logging.rs を全読して確認した公開面):
//! - `NonBlockingStdout::spawn_with(sink, capacity)` / `spawn()` / `dropped()`
//! - `MakeWriter for NonBlockingStdout` → `LogWriter::write` (非ブロッキング)
//! - `access_log_line(method, path, status, duration_ms)` / `set_access_log_json` /
//!   `access_log_json`
//!
//! `logging.rs` はテキスト整形を自前で行わない (整形は tracing-subscriber の fmt
//! レイヤ、本モジュールはバイト列のパススルーと有界キューだけを担う)。したがって
//! ここでの「テキスト整形」の契約は
//!   (a) 渡された行のバイト列が無改変・無分割でシンクに届くこと
//!   (b) 1 イベント = 1 write = 1 行 (インターリーブしないこと)
//! である。JSON 側は `access_log_line` が serde_json で 1 行に直列化する。
//!
//! Request-ID とヘッダ (Authorization 等) の redaction は **本モジュールには
//! 実装が無い** (grep で確認: request_id / redact / Authorization の参照ゼロ)。
//! アクセスログは `log_requests` ミドルウェアが method / path / status /
//! duration だけを渡すため、ヘッダ値がログに混入する経路自体が存在しない。
//! この「未対応 = 固定フィールド集合」をテストで固定する。

use std::io::{self, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use denno_server::{access_log_line, NonBlockingStdout};
use tracing_subscriber::fmt::MakeWriter;

/// 待ち時間の上限 (ロガースレッドの進行待ち)。
const WAIT: Duration = Duration::from_secs(10);

// ---------- テスト用シンク ----------

/// 受け取った write を「呼び出し単位」でそのまま蓄える。
/// write の境界 = ログ 1 行の境界 (本モジュールは 1 イベント 1 write_all)。
#[derive(Clone, Default)]
struct Capture {
    segments: Arc<Mutex<Vec<Vec<u8>>>>,
}

impl Capture {
    fn new() -> Self {
        Self::default()
    }

    fn snapshot(&self) -> Vec<u8> {
        self.segments.lock().unwrap().iter().flatten().copied().collect()
    }

    fn segments(&self) -> Vec<Vec<u8>> {
        self.segments.lock().unwrap().clone()
    }
}

impl Write for Capture {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.segments.lock().unwrap().push(buf.to_vec());
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// 開くまで write をブロックするゲート (凍結した stdout 消費者に相当)。
struct Gate {
    open: Mutex<bool>,
    cv: Condvar,
}

impl Gate {
    fn new() -> Self {
        Self { open: Mutex::new(false), cv: Condvar::new() }
    }

    fn wait(&self) {
        let mut open = self.open.lock().unwrap();
        while !*open {
            open = self.cv.wait(open).unwrap();
        }
    }

    fn release(&self) {
        *self.open.lock().unwrap() = true;
        self.cv.notify_all();
    }
}

/// write 突入時に入場回数を数え、ゲートが開くまでブロックするシンク。
struct GatedSink {
    buf: Arc<Mutex<Vec<Vec<u8>>>>,
    gate: Arc<Gate>,
    entered: Arc<AtomicUsize>,
}

impl Write for GatedSink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.entered.fetch_add(1, Ordering::SeqCst);
        self.gate.wait();
        self.buf.lock().unwrap().push(buf.to_vec());
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn spawn_capture(capacity: usize) -> (NonBlockingStdout, Capture) {
    let cap = Capture::new();
    let writer = NonBlockingStdout::spawn_with(Box::new(cap.clone()), capacity);
    (writer, cap)
}

/// `pred` が満たされるまで待ち、その時点のバッファを返す。
fn wait_until(cap: &Capture, what: &str, pred: impl Fn(&[u8]) -> bool) -> Vec<u8> {
    let deadline = Instant::now() + WAIT;
    loop {
        let buf = cap.snapshot();
        if pred(&buf) {
            return buf;
        }
        assert!(Instant::now() < deadline, "ロガーが待ち条件に到達しない: {what}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn lossy(buf: &[u8]) -> String {
    String::from_utf8_lossy(buf).to_string()
}

// ---------- JSON アクセスログ ----------

/// 制御文字・引用符・改行・非 ASCII を含んでも 1 行の妥当な JSON になり、
/// 各フィールドが元の値へ完全に復元されること。
#[test]
fn json_access_log_escapes_control_quotes_newlines_and_non_ascii() {
    let method = "GE\"T";
    let path = "/v1/messages?q=\"a\"\\b\nc\td\re\u{1b}[31m\u{7}日本語/é";
    let line = access_log_line(method, path, 418, 1234.5);

    // 1 行 JSON: 生の改行・制御文字を含まない (すべてエスケープされる)
    assert!(!line.contains('\n'), "生の改行が残っている: {line:?}");
    assert!(!line.contains('\r'), "生の CR が残っている: {line:?}");
    for b in line.bytes() {
        assert!(
            b >= 0x20 || b == b'\t',
            "生の制御バイト 0x{b:02x} が出力に残っている"
        );
    }
    assert_eq!(line.matches('\n').count(), 0);

    let v: serde_json::Value = serde_json::from_str(&line).expect("JSON として読めない");
    assert_eq!(v["method"], method);
    assert_eq!(v["path"], path, "パスが往復で一致しない");
    assert_eq!(v["status"], 418);
    assert!((v["duration_ms"].as_f64().unwrap() - 1234.5).abs() < 1e-9);
    assert!(v["ts_ms"].as_u64().unwrap() > 0);
}

/// Request-ID / ヘッダ redaction は未実装なので、アクセスログは固定の 5 フィールド
/// だけを持つ。Authorization らしき文字列を path に混ぜても「値」に閉じ込められ、
/// 追加フィールド (request_id / authorization / headers) にはならないこと。
#[test]
fn json_access_log_has_fixed_fields_and_no_request_id_or_header_leak() {
    let secret = "Authorization: Bearer sk-test-0123456789";
    let line = access_log_line(
        "POST",
        &format!("/v1/messages?x={secret}"),
        200,
        0.0,
    );
    let v: serde_json::Value = serde_json::from_str(&line).unwrap();
    let obj = v.as_object().expect("JSON オブジェクトでない");

    let mut keys: Vec<&str> = obj.keys().map(|k| k.as_str()).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        vec!["duration_ms", "method", "path", "status", "ts_ms"],
        "フィールド集合が変わった (request_id/ヘッダ漏えいの余地): {keys:?}"
    );
    for forbidden in ["request_id", "authorization", "headers", "api_key", "cookie"] {
        assert!(!obj.contains_key(forbidden), "禁止フィールド {forbidden} が存在する");
    }
    // 秘密らしき文字列は path の値の中だけに現れ、独立した値としては現れない
    assert_eq!(v["path"], format!("/v1/messages?x={secret}"));
    assert_eq!(line.matches("sk-test-0123456789").count(), 1);
}

// ---------- テキスト (パススルー) ----------

/// テキスト行は無改変でシンクへ渡る: 制御文字・引用符・改行・非 ASCII の
/// バイト列がそのまま、1 write で、順序を保って届く。
#[test]
fn text_lines_pass_through_verbatim_as_single_writes() {
    let (writer, cap) = spawn_capture(8);

    let lines: [&[u8]; 3] = [
        b"plain line\n",
        "quote:\" backslash:\\ tab:\t non-ascii:日本語 é\n".as_bytes(),
        b"ctrl:\x01\x07\x1b[31m cr:\r nul-free\n",
    ];
    for line in lines {
        writer.make_writer().write_all(line).unwrap();
    }

    let total: usize = lines.iter().map(|l| l.len()).sum();
    wait_until(&cap, "3 行の到着", |buf| buf.len() >= total);

    let segs = cap.segments();
    assert_eq!(segs.len(), 3, "write が分割/結合された: {segs:?}");
    for (got, want) in segs.iter().zip(lines.iter()) {
        assert_eq!(got.as_slice(), *want, "バイト列が改変された");
        assert_eq!(*got.last().unwrap(), b'\n', "行が改行で終わっていない");
    }
    assert_eq!(writer.dropped(), 0);
}

/// 巨大な 1 行 (512KiB 超) は切り詰められず、丸ごと 1 write で届く
/// (本モジュールは行長で truncate しない。上限はキュー行数だけ)。
#[test]
fn oversized_single_line_is_not_truncated() {
    let (writer, cap) = spawn_capture(4);

    let mut line = String::with_capacity(600_000);
    line.push_str(&"x".repeat(500_000));
    line.push_str("日本語の末尾");
    line.push_str("\"\t\r\n");
    let bytes = line.into_bytes();

    writer.make_writer().write_all(&bytes).unwrap();
    let buf = wait_until(&cap, "巨大行の全体到着", |b| b.len() >= bytes.len());

    assert_eq!(buf.len(), bytes.len(), "行が切り詰められた");
    assert_eq!(buf, bytes, "巨大行の内容が改変された");
    let segs = cap.segments();
    assert_eq!(segs.len(), 1, "巨大行が複数 write に分割された");
    assert_eq!(writer.dropped(), 0);
}

// ---------- キュー満杯 (破棄) ----------

/// キュー満杯時は行単位で破棄される: 部分行は決して届かず、破棄数は 1 行で
/// 報告され、復旧後はそのまま流れる。write はブロックしない。
#[test]
fn queue_pressure_drops_whole_lines_never_partial() {
    let buf = Arc::new(Mutex::new(Vec::<Vec<u8>>::new()));
    let gate = Arc::new(Gate::new());
    let entered = Arc::new(AtomicUsize::new(0));
    let sink = GatedSink { buf: buf.clone(), gate: gate.clone(), entered: entered.clone() };
    let writer = NonBlockingStdout::spawn_with(Box::new(sink), 1);

    // 1 行目をロガーに掴ませ、シンク内でブロックさせる
    writer.make_writer().write_all(b"line-1 first\n").unwrap();
    let deadline = Instant::now() + WAIT;
    while entered.load(Ordering::SeqCst) == 0 {
        assert!(Instant::now() < deadline, "ロガーが write に突入しない");
        std::thread::sleep(Duration::from_millis(5));
    }

    // 容量 1: line-2 がキュー、line-3..5 は破棄。すべて即座に返る。
    let t0 = Instant::now();
    writer.make_writer().write_all(b"line-2 queued\n").unwrap();
    writer.make_writer().write_all(b"line-3 dropped\n").unwrap();
    writer.make_writer().write_all(b"line-4 dropped\n").unwrap();
    writer.make_writer().write_all(b"line-5 dropped\n").unwrap();
    assert!(t0.elapsed() < Duration::from_secs(2), "write がブロックした");

    assert_eq!(writer.dropped(), 3, "破棄数は行単位 (3 行)");
    gate.release();

    let deadline = Instant::now() + WAIT;
    loop {
        if buf.lock().unwrap().iter().any(|s| s == b"line-2 queued\n") {
            break;
        }
        assert!(Instant::now() < deadline, "復旧後の行が流れない");
        std::thread::sleep(Duration::from_millis(5));
    }

    let segs = buf.lock().unwrap().clone();
    let text = lossy(&segs.concat());
    assert!(text.contains("line-1 first\n"), "1 行目が欠落: {text}");
    assert!(text.contains("line-2 queued\n"), "2 行目が欠落: {text}");
    assert!(text.contains("3 log line(s) dropped"), "破棄報告が出ない: {text}");
    for dropped in ["line-3", "line-4", "line-5"] {
        assert!(!text.contains(dropped), "破棄した行が出ている: {dropped}");
    }
    // 部分行は存在しない: どの write も 'line-' で始まる完全な行 + 改行
    for seg in &segs {
        let s = lossy(seg);
        if s.starts_with("denno-logger:") {
            assert!(s.ends_with('\n'), "破棄報告が 1 行でない: {s:?}");
        } else {
            assert!(s.ends_with('\n'), "部分行が届いた: {s:?}");
        }
    }
    assert_eq!(writer.dropped(), 0, "報告後はカウンタがリセットされる");
}

// ---------- 並行書き込み ----------

/// 複数スレッドからの同時書き込みでパニックせず、全行が 1 write = 1 行のまま
/// 欠落無く (容量に余裕がある前提) 届くこと。
#[test]
fn concurrent_writers_emit_whole_lines_without_panics() {
    const THREADS: usize = 8;
    const LINES: usize = 64;
    let (writer, cap) = spawn_capture(4096);

    let mut expected: Vec<Vec<u8>> = Vec::with_capacity(THREADS * LINES);
    let mut handles = Vec::with_capacity(THREADS);
    for t in 0..THREADS {
        for i in 0..LINES {
            expected.push(format!("t{t:02}-{i:03} \"q\" 日本語\t{0}\n", "y".repeat(i % 7)).into_bytes());
        }
        let w = writer.clone();
        handles.push(std::thread::spawn(move || {
            for i in 0..LINES {
                let line = format!("t{t:02}-{i:03} \"q\" 日本語\t{0}\n", "y".repeat(i % 7));
                w.make_writer().write_all(line.as_bytes()).expect("write_all が失敗");
            }
        }));
    }
    for h in handles {
        h.join().expect("書き込みスレッドがパニックした");
    }

    let total: usize = expected.iter().map(|l| l.len()).sum();
    wait_until(&cap, "並行書き込みの全行到着", |b| b.len() >= total);

    let segs = cap.segments();
    assert_eq!(writer.dropped(), 0, "容量に余裕があるのに破棄された");
    assert_eq!(segs.len(), THREADS * LINES, "write 数が行数と一致しない");

    let mut got: Vec<Vec<u8>> = segs;
    let mut want = expected;
    // スレッド間の順序は非決定。write 単位で整列して集合一致を確認する。
    got.sort_unstable();
    want.sort_unstable();
    assert_eq!(got, want, "行の欠落・混入・分割がある");
    for seg in &got {
        assert_eq!(*seg.last().unwrap(), b'\n');
        assert!(seg.iter().all(|&b| b != 0), "NUL が混入した");
    }
}

/// `access_log_line` と JSON ゲートの並行アクセスがパニックしないこと
/// (グローバルは AtomicBool。読み書きの競合で壊れない)。
#[test]
fn concurrent_access_log_calls_do_not_panic() {
    let handles: Vec<_> = (0..8)
        .map(|t| {
            std::thread::spawn(move || {
                for i in 0..200 {
                    denno_server::set_access_log_json(i % 2 == 0 || t % 2 == 0);
                    let _ = denno_server::access_log_json();
                    let line = access_log_line("GET", &format!("/p/{t}/{i}\n\t\"日本語\""), 200, 0.5);
                    let v: serde_json::Value = serde_json::from_str(&line).expect("JSON 不正");
                    assert_eq!(v["method"], "GET");
                }
            })
        })
        .collect();
    for h in handles {
        h.join().expect("スレッドがパニックした");
    }
    // 後続テストの前提 (text モード) に戻す
    denno_server::set_access_log_json(false);
}

// ---------- テキスト整形 (tracing-subscriber 経由・プロセスで 1 回) ----------

/// 実際の fmt レイヤ + 非同期ライタで 1 イベントを流し、特殊文字を含む
/// メッセージが壊れず UTF-8 のまま 1 行 (末尾改行 1 個) で届くことを確認する。
/// グローバルサブスクライバはプロセスで 1 度しか設定できないため、この
/// テストファイル内で唯一の init 呼び出しとする。
#[test]
fn text_fmt_end_to_end_preserves_special_characters() {
    let cap = Capture::new();
    let writer = NonBlockingStdout::spawn_with(Box::new(cap.clone()), 64);
    tracing_subscriber::fmt()
        .with_ansi(false)
        .with_env_filter(tracing_subscriber::EnvFilter::new("info"))
        .with_writer(writer)
        .init();

    let msg = "q=\"v\" tab:\t nl-next 日本語 é 0x1b-esc";
    tracing::info!("{}", msg);

    let buf = wait_until(&cap, "fmt イベントの到着", |b| b.ends_with(b"\n"));
    let text = String::from_utf8(buf.clone()).expect("出力が UTF-8 でない");
    assert!(text.contains(msg), "メッセージが改変された: {text:?}");
    assert!(text.contains("日本語"), "非 ASCII が失われた: {text:?}");
    assert_eq!(buf.iter().filter(|&&b| b == b'\n').count(), 1, "改行数が想定外: {text:?}");
}
