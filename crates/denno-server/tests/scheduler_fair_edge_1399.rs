//! Task 1399: denno-server スケジューラの公平性・飽和・スロット回復の edge テスト。
//!
//! P1041 (`tests/scheduler_fair_1041.rs`) が監査した不変条件の上に、次を固定する:
//!
//! 1. **マルチプロトコル FIFO**: 受付キューは `mpsc::sync_channel` で、ワーカーは
//!    「空きスロットがある限り先頭から pull し、pull 順に prefill する」。
//!    `Job::Stream` (Anthropic SSE) / `Job::OpenaiStream` (OpenAI SSE) /
//!    `Job::Chat` (非ストリーム = Ollama 経路を含む) は同じ 1 本のキューに載るので、
//!    種別が混在しても prefill 開始順 = 受付順 になる。
//! 2. **zero-leak スロット回復**: 生成エラー (`aborted_error`) とクライアント切断
//!    (`aborted_disconnected`) のどちらで Run が終わっても、KV スロットは
//!    解放され (`slot_busy_count() == 0`)、後続リクエストが全スロットを使える。
//! 3. **stats 会計**: `completed` / `aborted_timeout` / `aborted_disconnected` /
//!    `aborted_error` は 1 リクエストにつき必ず 1 つだけ増え、合計が受付数に一致する。
//!
//! **決定論**: 壁時計の sleep で順序を作らない。同期は次の 2 種類だけで取る。
//! - エンジン Mutex をテストが保持してワーカーを admit で止める (順序の固定)
//! - `ProbeModel` が prefill ごとに std mpsc へ印を送り、必要なら gate チャネルで
//!   ブロックする (prefill 順の観測と、生成途中の停止点)
//! 待ちは `recv_timeout` / `tokio::time::timeout` の上限付きで、上限に達したら
//! パニックする (無限待ちにしない)。4 コア CI でも順序の前提が崩れない。
//!
//! CPU のみ (ダミーモデル / ダミートークナイザ)。実モデルは使わない。

use std::convert::Infallible;
use std::sync::mpsc as std_mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::response::sse::Event;
use denno_engine::kv_cache::KvCache;
use denno_engine::model::Model;
use denno_engine::testutil::{dummy_tokenizer, DUMMY_CONTEXT, DUMMY_EOS, DUMMY_VOCAB};
use denno_engine::{ChatMessage, ChatOutput, Engine, GenParams};
use denno_server::{Limits, Scheduler};
use tokio::sync::mpsc;

// ---------------------------------------------------------------------------
// ProbeModel: prefill 順の観測 + 失敗注入 + gate (生成途中の停止点)
// ---------------------------------------------------------------------------

/// prefill (入力長 > 1 の forward) のたびに、content の印トークンを
/// `prefill_tx` へ送る。`fail_on` に一致する印なら forward を失敗させる。
/// `gate` が Some のときは prefill の先頭で 1 受信ぶんブロックする
/// (テストが明示的に解放するまでワーカーはエンジン Mutex を握ったまま止まる)。
struct ProbeModel {
    prefill_tx: std_mpsc::Sender<u32>,
    fail_on: Option<u32>,
    gate: Option<std_mpsc::Receiver<()>>,
}

impl ProbeModel {
    /// tokens から「content の印」を 1 つ選ぶ。ダミーの chat template は語彙外の
    /// 語を `[UNK]`(= 20) にするので、語彙内 (0..DUMMY_VOCAB) かつ EOS 以外の
    /// 最初のトークンがそのまま content の先頭語になる。
    fn marker(tokens: &[u32]) -> Option<u32> {
        tokens
            .iter()
            .copied()
            .find(|&t| (t as usize) < DUMMY_VOCAB && t != DUMMY_EOS)
    }
}

impl Model for ProbeModel {
    fn context(&self) -> usize {
        DUMMY_CONTEXT
    }

    fn eos_ids(&self) -> &[u32] {
        // EOS を返さない = max_tokens まで decode する (生成途中の停止点を作れる)
        &[]
    }

    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> anyhow::Result<Vec<f32>> {
        if tokens.len() > 1 {
            if let Some(m) = Self::marker(tokens) {
                let _ = self.prefill_tx.send(m);
                if self.fail_on == Some(m) {
                    return Err(anyhow::anyhow!("synthetic prefill failure (marker {m})"));
                }
            }
            if let Some(gate) = self.gate.as_mut() {
                let _ = gate.recv();
            }
        }
        // P17e: 需要ベース確保 (書き込み前に容量確保) のダミー側契約
        let _ = cache.kv_bufs(0, tokens.len());
        cache.advance(tokens.len());
        let last = tokens.last().copied().unwrap_or(0) as usize;
        let mut logits = vec![0.0f32; DUMMY_VOCAB];
        logits[(last + 1) % DUMMY_VOCAB] = 5.0;
        logits[(last + 2) % DUMMY_VOCAB] = 1.0;
        Ok(logits)
    }

    fn new_cache(&self) -> KvCache {
        KvCache::new(1, 4, DUMMY_CONTEXT)
    }
}

// ---------------------------------------------------------------------------
// ヘルパ
// ---------------------------------------------------------------------------

fn params(max_tokens: usize) -> GenParams {
    GenParams { max_tokens, temperature: Some(0.0), ..Default::default() }
}

fn user_msg(text: &str) -> Vec<ChatMessage> {
    vec![ChatMessage { role: "user".into(), content: text.into() }]
}

fn engine_with(model: Box<dyn Model>) -> Arc<Mutex<Engine>> {
    let tok = dummy_tokenizer().unwrap();
    Arc::new(Mutex::new(Engine::from_parts(model, tok, "denno-test", None)))
}

/// テスト用スケジューラ。`prefill_rx` で prefill の印を観測できる。
fn probe_scheduler(
    fail_on: Option<u32>,
    gate: Option<std_mpsc::Receiver<()>>,
    max_queue: usize,
) -> (Scheduler, Arc<Mutex<Engine>>, std_mpsc::Receiver<u32>) {
    let (prefill_tx, prefill_rx) = std_mpsc::channel();
    let model = ProbeModel { prefill_tx, fail_on, gate };
    let engine = engine_with(Box::new(model));
    let sched = Scheduler::spawn_with_limits(engine.clone(), max_queue, Limits::default());
    (sched, engine, prefill_rx)
}

fn slot_busy(engine: &Arc<Mutex<Engine>>) -> usize {
    engine.lock().unwrap().slot_busy_count()
}

fn slot_count(engine: &Arc<Mutex<Engine>>) -> usize {
    engine.lock().unwrap().slot_count()
}

/// prefill の印を `n` 個、順序を保って受け取る (上限付き)。
fn take_markers(rx: &std_mpsc::Receiver<u32>, n: usize) -> Vec<u32> {
    (0..n)
        .map(|i| {
            rx.recv_timeout(Duration::from_secs(10))
                .unwrap_or_else(|_| panic!("{i} 個目の prefill が 10s 以内に来ない"))
        })
        .collect()
}

/// Chat を 1 件受付して結果を待つ。
async fn run_chat(
    sched: &Scheduler,
    content: &str,
    max_tokens: usize,
) -> anyhow::Result<ChatOutput> {
    let rx = sched
        .submit_chat(None, user_msg(content), params(max_tokens), None, None, None)
        .expect("Chat が QueueFull で拒否された (テストの max_queue を見直す)");
    tokio::time::timeout(Duration::from_secs(20), rx)
        .await
        .expect("Chat が 20s 以内に返らない (ワーカーが塞がっている)")
        .expect("Chat の応答チャネルが落ちた")
}

/// SSE (Stream) を最後まで読み切る。返り値はイベント数。
async fn drain_stream(mut rx: mpsc::Receiver<Result<Event, Infallible>>) -> usize {
    let mut n = 0;
    while tokio::time::timeout(Duration::from_secs(20), rx.recv())
        .await
        .expect("SSE が 20s 以内に終わらない")
        .is_some()
    {
        n += 1;
    }
    n
}

// ---------------------------------------------------------------------------
// 1. マルチプロトコル FIFO: 3 種を 1 本のキューへ受付順に積む
// ---------------------------------------------------------------------------

/// `Stream` (Anthropic SSE) → `OpenaiStream` (OpenAI SSE) → `Chat` (非ストリーム)
/// の順に受付し、prefill の開始順が受付順と一致することを検証する。
///
/// 順序の固定: エンジン Mutex をテストが保持したまま 3 件を受付する。ワーカーは
/// 1 件目の admit で Mutex 待ちに止まるので、3 件は受付順そのままキューへ入り、
/// 解放後は FIFO で pull される (待ち合わせに sleep を使わない)。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fifo_prefill_order_matches_admission_across_three_job_kinds() {
    let (sched, engine, prefill_rx) = probe_scheduler(None, None, 8);
    let guard = engine.lock().unwrap();

    // 1. Anthropic SSE (Job::Stream)
    let (ev_tx, ev_rx) = mpsc::channel::<Result<Event, Infallible>>(64);
    sched
        .submit_stream(None, user_msg("t13"), params(2), None, None, None, ev_tx)
        .expect("Stream が拒否された");

    // 2. OpenAI SSE (Job::OpenaiStream) — 生イベントの型名は非公開なので推論に任せる
    let (raw_tx, mut raw_rx) = mpsc::channel(64);
    sched
        .submit_openai_stream(None, user_msg("t14"), params(2), None, None, None, raw_tx)
        .expect("OpenaiStream が拒否された");

    // 3. 非ストリーム (Job::Chat)
    let chat_rx = sched
        .submit_chat(None, user_msg("t15"), params(2), None, None, None)
        .expect("Chat が拒否された");

    drop(guard);

    let order = take_markers(&prefill_rx, 3);
    assert_eq!(
        order,
        vec![13, 14, 15],
        "prefill 開始順が受付順 (Stream → OpenaiStream → Chat) と違う (FIFO 公平性の破れ)"
    );

    let out = tokio::time::timeout(Duration::from_secs(20), chat_rx)
        .await
        .expect("Chat が 20s 以内に返らない")
        .expect("Chat の応答チャネルが落ちた")
        .expect("Chat がエラー");
    assert_eq!(out.output_tokens, 2, "Chat の生成数が max_tokens と違う");

    let n_ev = drain_stream(ev_rx).await;
    assert!(n_ev >= 4, "Anthropic SSE のイベントが少なすぎる: {n_ev}");

    // OpenAI SSE も終端まで読む (channel の閉鎖 = Run の終端)
    let mut n_raw = 0;
    while tokio::time::timeout(Duration::from_secs(20), raw_rx.recv())
        .await
        .expect("OpenAI SSE が 20s 以内に終わらない")
        .is_some()
    {
        n_raw += 1;
    }
    assert!(n_raw >= 2, "OpenAI SSE のイベントが少なすぎる: {n_raw}");

    let st = sched.stats();
    assert_eq!(st.completed, 3, "3 件すべてが完了計数に載っていない: {st:?}");
    assert_eq!(st.aborted_timeout, 0, "{st:?}");
    assert_eq!(st.aborted_disconnected, 0, "{st:?}");
    assert_eq!(st.aborted_error, 0, "{st:?}");
    assert_eq!(sched.in_flight(), 0, "完了後も in_flight が残っている");
    assert_eq!(sched.pending(), 0, "完了後も pending が残っている");
    assert_eq!(slot_busy(&engine), 0, "完了後に KV スロットが漏れている");
}

// ---------------------------------------------------------------------------
// 2. 同時実行上限を超える件数でも受付順が保たれる (種別混在)
// ---------------------------------------------------------------------------

/// `MAX_ACTIVE_SESSIONS` (4) を超える 12 件を、3 種を回しながら 1 本のスレッドで
/// 受付する。prefill の印は受付順に 0..12 で現れ、全件が完了する。
/// (キュー待ちが発生しても、種別ごとの優先度が無いことを固定する)
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fifo_order_holds_beyond_slot_capacity_for_mixed_job_kinds() {
    const N: usize = 12;
    let (sched, engine, prefill_rx) = probe_scheduler(None, None, 16);

    // 各ジョブの content 先頭語 = 0..N-1 (prompt を一意にして prefix cache の
    // 全一致を起こさない)。2 語目で prefill forward の入力長 > 1 を保証する。
    let (done_tx, mut done_rx) = mpsc::unbounded_channel::<usize>();
    for i in 0..N {
        let content = format!("t{i} t13");
        match i % 3 {
            0 => {
                let (ev_tx, mut ev_rx) = mpsc::channel::<Result<Event, Infallible>>(64);
                sched
                    .submit_stream(None, user_msg(&content), params(1), None, None, None, ev_tx)
                    .expect("Stream が拒否された");
                let done_tx = done_tx.clone();
                tokio::spawn(async move {
                    while ev_rx.recv().await.is_some() {}
                    let _ = done_tx.send(i);
                });
            }
            1 => {
                let (raw_tx, mut raw_rx) = mpsc::channel(64);
                sched
                    .submit_openai_stream(None, user_msg(&content), params(1), None, None, None, raw_tx)
                    .expect("OpenaiStream が拒否された");
                let done_tx = done_tx.clone();
                tokio::spawn(async move {
                    while raw_rx.recv().await.is_some() {}
                    let _ = done_tx.send(i);
                });
            }
            _ => {
                let rx = sched
                    .submit_chat(None, user_msg(&content), params(1), None, None, None)
                    .expect("Chat が拒否された");
                let done_tx = done_tx.clone();
                tokio::spawn(async move {
                    let _ = rx.await;
                    let _ = done_tx.send(i);
                });
            }
        }
    }

    let mut finished = Vec::new();
    for _ in 0..N {
        finished.push(
            tokio::time::timeout(Duration::from_secs(30), done_rx.recv())
                .await
                .expect("30s 以内に全件が完了しない (飢餓)")
                .expect("完了通知チャネルが落ちた"),
        );
    }
    finished.sort_unstable();
    assert_eq!(finished, (0..N).collect::<Vec<_>>(), "完了しなかったジョブがある");

    // prefill は 1 回ずつ、受付順に記録されている
    let mut order = Vec::new();
    while let Ok(m) = prefill_rx.try_recv() {
        order.push(m);
    }
    assert_eq!(
        order,
        (0..N as u32).collect::<Vec<_>>(),
        "12 件の prefill 開始順が受付順と違う (スロット超過時の FIFO 公平性の破れ)"
    );

    let st = sched.stats();
    assert_eq!(st.completed, N, "全件が完了計数に載っていない: {st:?}");
    assert_eq!(st.aborted_timeout, 0, "{st:?}");
    assert_eq!(st.aborted_disconnected, 0, "{st:?}");
    assert_eq!(st.aborted_error, 0, "{st:?}");
    assert_eq!(slot_busy(&engine), 0, "完了後に KV スロットが漏れている");
}

// ---------------------------------------------------------------------------
// 3. 生成エラー: スロットが漏れず、後続が全スロットを使える
// ---------------------------------------------------------------------------

/// prefill で失敗するジョブを 1 件流す。応答はエラー、`aborted_error` は 1、
/// スロットは 0 busy に戻り、その後の健常ジョブが全件通る。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn prefill_error_releases_slot_and_is_counted_once() {
    let (sched, engine, _prefill_rx) = probe_scheduler(Some(9), None, 64);

    let err = run_chat(&sched, "t9", 2)
        .await
        .expect_err("prefill 失敗のジョブが成功してしまった");
    assert!(
        format!("{err:#}").contains("synthetic prefill failure"),
        "エラー理由が注入した失敗と違う: {err:#}"
    );

    let st = sched.stats();
    assert_eq!(st.aborted_error, 1, "エラー計数が 1 でない: {st:?}");
    assert_eq!(st.completed, 0, "{st:?}");
    assert_eq!(st.aborted_disconnected, 0, "{st:?}");
    assert_eq!(st.aborted_timeout, 0, "{st:?}");
    assert_eq!(sched.in_flight(), 0, "エラー後も in_flight が残っている");
    assert_eq!(sched.pending(), 0, "エラー後も pending が残っている");
    assert_eq!(slot_busy(&engine), 0, "エラーで KV スロットが漏れた");

    // スロット数ぶんの健常ジョブが同時に通る (漏れていれば install で失敗する)
    let slots = slot_count(&engine);
    assert!(slots >= 4, "スロット数が想定より少ない: {slots}");
    let mut rxs = Vec::new();
    // 注入した失敗トークン (t9) を含めない content で埋める
    let words = [0usize, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15];
    for i in 0..slots {
        let content = format!("t{} t13", words[i % words.len()]);
        let rx = sched
            .submit_chat(None, user_msg(&content), params(2), None, None, None)
            .expect("健常ジョブが QueueFull で拒否された");
        rxs.push(rx);
    }
    for (i, rx) in rxs.into_iter().enumerate() {
        let out = tokio::time::timeout(Duration::from_secs(20), rx)
            .await
            .unwrap_or_else(|_| panic!("{i} 件目の健常ジョブが 20s 以内に完了しない"))
            .unwrap_or_else(|_| panic!("{i} 件目の応答チャネルが落ちた"))
            .unwrap_or_else(|e| panic!("{i} 件目の健常ジョブがエラー: {e:#} (スロット漏れの疑い)"));
        assert_eq!(out.output_tokens, 2, "{i} 件目の生成数が違う");
    }

    let st = sched.stats();
    assert_eq!(st.completed, slots, "健常ジョブの完了数が違う: {st:?}");
    assert_eq!(st.aborted_error, 1, "健常ジョブがエラー計数に載った: {st:?}");
    assert_eq!(slot_busy(&engine), 0, "健常ジョブ完了後に KV スロットが漏れている");
}

// ---------------------------------------------------------------------------
// 4. キュー内で切断されたジョブ: 容量を食い続けず、スロットも取らない
// ---------------------------------------------------------------------------

/// ワーカーを 1 件目で止め、2 件目をキューに入れてから受信側を drop する。
/// 解放後、1 件目は完走し、2 件目は `aborted_disconnected` として 1 回だけ
/// 数えられ、スロットは占有されない。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn queued_job_disconnected_before_pull_is_counted_and_leaves_no_slot() {
    let (sched, engine, _prefill_rx) = probe_scheduler(None, None, 8);
    let guard = engine.lock().unwrap();

    let first = sched
        .submit_chat(None, user_msg("t13"), params(2), None, None, None)
        .expect("1 件目が拒否された");
    // 2 件目はキューに載り、ワーカーが pull する前に受信側が消える
    let second = sched
        .submit_chat(None, user_msg("t14"), params(2), None, None, None)
        .expect("2 件目が拒否された");
    drop(second);

    drop(guard);

    let out = tokio::time::timeout(Duration::from_secs(20), first)
        .await
        .expect("1 件目が 20s 以内に返らない")
        .expect("1 件目の応答チャネルが落ちた")
        .expect("1 件目がエラー");
    assert_eq!(out.output_tokens, 2);

    let st = sched.stats();
    assert_eq!(st.completed, 1, "1 件目が完了計数に載っていない: {st:?}");
    assert_eq!(
        st.aborted_disconnected, 1,
        "キュー内切断が打ち切り計数に 1 回だけ載っていない: {st:?}"
    );
    assert_eq!(st.aborted_error, 0, "{st:?}");
    assert_eq!(st.aborted_timeout, 0, "{st:?}");
    assert_eq!(slot_busy(&engine), 0, "切断ジョブが KV スロットを漏らした");
    assert_eq!(sched.in_flight(), 0, "in_flight が 0 に戻らない");
    assert_eq!(sched.pending(), 0, "pending が 0 に戻らない");
}

// ---------------------------------------------------------------------------
// 5. 生成途中 (prefill 後) の切断: スロットを解放して後続が通る
// ---------------------------------------------------------------------------

/// gate で prefill の途中にワーカーを止め、その状態で Chat の受信側を drop する。
/// gate 解放後、当該 Run は `aborted_disconnected` として打ち切られ、KV スロットは
/// 解放される (直後に投入したジョブが同じスロットで完走する)。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn active_job_disconnect_releases_slot_for_the_next_request() {
    let (gate_tx, gate_rx) = std_mpsc::channel::<()>();
    let (sched, engine, prefill_rx) = probe_scheduler(None, Some(gate_rx), 8);

    let rx = sched
        .submit_chat(None, user_msg("t13"), params(5), None, None, None)
        .expect("1 件目が拒否された");
    // prefill の先頭でワーカーが gate 待ちに入る = Run は KV スロットを保持中
    let m = prefill_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("1 件目の prefill が始まらない");
    assert_eq!(m, 13, "1 件目の印が違う");
    // 注: この時点でワーカーは forward の中でエンジン Mutex を保持しているため、
    // `slot_busy()` を読むとデッドロックする。スロットの解放は完了後に見る。

    // 生成途中でクライアントが消える
    drop(rx);
    // gate を解放すると prefill が完了し、次のステップ境界で切断が検出される
    gate_tx.send(()).expect("gate の送信に失敗");
    // 後続用の permit も先に積んでおく (2 回目の prefill で消費される)
    gate_tx.send(()).expect("gate の送信に失敗");

    // 後続ジョブの完走が「切断 Run の後始末が終わった」ことの同期点になる
    let out = run_chat(&sched, "t14", 2)
        .await
        .expect("後続ジョブがエラー (スロットが解放されていない疑い)");
    assert_eq!(out.output_tokens, 2);

    let st = sched.stats();
    assert_eq!(
        st.aborted_disconnected, 1,
        "生成途中の切断が打ち切り計数に載っていない: {st:?}"
    );
    assert_eq!(st.completed, 1, "後続ジョブの完了計数が違う: {st:?}");
    assert_eq!(st.aborted_error, 0, "{st:?}");
    assert_eq!(st.aborted_timeout, 0, "{st:?}");
    assert_eq!(slot_busy(&engine), 0, "切断後に KV スロットが漏れている");
    assert_eq!(sched.in_flight(), 0, "in_flight が 0 に戻らない");
    assert_eq!(sched.pending(), 0, "pending が 0 に戻らない");
}

// ---------------------------------------------------------------------------
// 6. stats 会計: 4 つの計数は排他で、合計 = 受付数
// ---------------------------------------------------------------------------

/// 完了 3 / エラー 1 / 切断 1 を混在させ、各計数が 1 リクエスト 1 回だけ増え、
/// 合計が受付数と一致することを検証する (`aborted_timeout` は 0 のまま)。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stats_counters_partition_all_submitted_requests() {
    let (sched, engine, _prefill_rx) = probe_scheduler(Some(9), None, 8);

    // 1 件目をエンジン Mutex で止めたまま、完了 2 件 + 切断 1 件を受付する
    let guard = engine.lock().unwrap();
    let a = sched
        .submit_chat(None, user_msg("t13"), params(2), None, None, None)
        .expect("A が拒否された");
    let b = sched
        .submit_chat(None, user_msg("t14"), params(2), None, None, None)
        .expect("B が拒否された");
    let (ev_tx, ev_rx) = mpsc::channel::<Result<Event, Infallible>>(64);
    sched
        .submit_stream(None, user_msg("t15"), params(2), None, None, None, ev_tx)
        .expect("C が拒否された");
    // C の受信側は即座に drop → 受理後に切断として打ち切られる
    drop(ev_rx);
    drop(guard);

    let out_a = tokio::time::timeout(Duration::from_secs(20), a)
        .await
        .expect("A が返らない")
        .expect("A の応答チャネルが落ちた")
        .expect("A がエラー");
    assert_eq!(out_a.output_tokens, 2);
    let out_b = tokio::time::timeout(Duration::from_secs(20), b)
        .await
        .expect("B が返らない")
        .expect("B の応答チャネルが落ちた")
        .expect("B がエラー");
    assert_eq!(out_b.output_tokens, 2);

    // 同期点: このジョブの完走後に、切断 C の後始末も済んでいる
    let out_f = run_chat(&sched, "t1", 2).await.expect("F がエラー");
    assert_eq!(out_f.output_tokens, 2);

    // 失敗ジョブ (prefill で注入エラー)
    run_chat(&sched, "t9", 2)
        .await
        .expect_err("失敗ジョブが成功してしまった");

    let st = sched.stats();
    assert_eq!(st.completed, 3, "完了計数が違う: {st:?}");
    assert_eq!(st.aborted_disconnected, 1, "切断計数が違う: {st:?}");
    assert_eq!(st.aborted_error, 1, "エラー計数が違う: {st:?}");
    assert_eq!(st.aborted_timeout, 0, "期限計数が動いた: {st:?}");
    let total = st.completed + st.aborted_timeout + st.aborted_disconnected + st.aborted_error;
    assert_eq!(total, 5, "計数の合計が受付数 (5) と一致しない: {st:?}");
    assert_eq!(slot_busy(&engine), 0, "会計テスト後に KV スロットが漏れている");
    assert_eq!(sched.in_flight(), 0, "in_flight が 0 に戻らない");
    assert_eq!(sched.pending(), 0, "pending が 0 に戻らない");
}
