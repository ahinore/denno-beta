//! task 1304: バイナリ <-> NDJSON 変換の並行ストレステスト。
//!
//! 8 本のスレッドを同時に走らせ、各スレッドが **混在 17 kind** のフレーム列を
//!
//! ```text
//! Frame -> バイナリ列 -> Frame -> NDJSON 列 -> Frame -> バイナリ列
//! ```
//!
//! の順で往復変換する。検証する不変条件:
//!
//! - **ゼロデータ損失**: 送ったフレーム数と受けたフレーム数が一致し、内容も
//!   完全一致する (並び順・フィールド値・u64 の精度を含む)。
//! - **重複・混線なし**: スレッド間でフレームが混ざらず、各スレッドの結果は
//!   自分が書いた列とだけ一致する。
//! - **決定性**: 同じ入力から 2 回変換してもバイト列が同一。
//! - **ストリーム復帰**: NDJSON は小さな BufReader バッファで読み、行境界を
//!   跨いでも取りこぼさない。
//!
//! 公開 API のみを使う。panic しないことは素で呼んでテストが落ちないことで
//! 検証する (catch_unwind は使わない)。

use std::io::{BufReader, Cursor};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

use denno_telemetry::{
    frame_to_ndjson, parse_frame_line, read_frames_ndjson, write_frames_ndjson, Frame,
};

/// 同時に走らせるスレッド数 (タスク指定)。
const THREADS: usize = 8;

/// 1 スレッドが扱うラウンド数。1 ラウンドで 17 kind を 1 枚ずつ出すので、
/// 1 スレッド = 17 * ROUNDS 枚、全体 = 8 * 17 * ROUNDS 枚。
const ROUNDS: u32 = 240;

/// 全 kind 数 (1..=17)。
const KINDS: u32 = 17;

/// `t` (スレッド番号) と `r` (ラウンド) から決定的にフレームを作る。
///
/// 並列配列 (`busy`/`t_lens`, `idx`/`val`, `experts`/`weights`) の長さは必ず
/// 一致させる (encode の切り揃え仕様に入らない正規化済み入力)。
fn batch_frame(t: usize, r: u32) -> Frame {
    let layer = (r % 48) as u16;
    let t32 = t as u32;
    let uniq = t32.wrapping_mul(1_000_003).wrapping_add(r);
    match r % KINDS {
        0 => Frame::ServerInfo {
            model: format!("model-{t}-{r}-日本語"),
            model_id: format!("id/{t}/{r}"),
            quant: if r % 2 == 0 { "fp32".into() } else { "q4_k_m".into() },
            slots: 1 + (r % 8),
            context: 4096 + r,
            layers: 48,
        },
        1 => Frame::SessionAdmit {
            session: uniq,
            slot: r % 4,
            prompt_tokens: r * 3 + 1,
        },
        2 => Frame::SessionDone {
            session: uniq,
            reason: (r % 4) as u8,
            output_tokens: r + t32,
        },
        3 => Frame::Prefill {
            session: uniq,
            prompt_tokens: 16 + (r % 512),
            ns: if r % 41 == 0 { u64::MAX } else { (uniq as u64) * 7_919 },
            tokens: vec![r, r.wrapping_add(1), t32],
        },
        4 => Frame::Token {
            session: uniq,
            token: r.wrapping_mul(31).wrapping_add(t32),
            pos: r,
        },
        5 => Frame::Step {
            width: 1 + (r % 8),
            ns: (uniq as u64).wrapping_mul(1_000_003),
        },
        6 => Frame::OpSummary {
            ops: vec![(0u8, r as u64), (31, u64::MAX - r as u64)],
        },
        7 => Frame::KvUsage {
            busy: vec![r % 2 == 0, true, false],
            t_lens: vec![r, 0, u32::MAX - r],
        },
        8 => Frame::Gap { lost: r % 97 },
        9 => Frame::TopK {
            session: uniq,
            chosen: r % 64,
            entropy_f16: (0x3c00 + (r % 0x400)) as u16,
            entries: vec![
                (r % 1024, 0x3c00u16),
                (r.wrapping_add(1) % 1024, 0x3800),
                (u32::MAX - r, 0xffff),
            ],
        },
        10 => Frame::LayerRun {
            layer,
            ops: vec![(4u8, (r as u64) * 1200), (5, 900), (31, 0)],
        },
        11 => Frame::LayerAct {
            layer,
            idx: vec![r, r.wrapping_add(1), r.wrapping_add(2)],
            val: vec![0x3c00, (r % 0x7c00) as u16, 0x0000],
        },
        12 => Frame::RoleAct {
            layer,
            attn: 0x3c00,
            gdn: (r % 0x3c00) as u16,
            router: 0x4000,
            shex: 0x7bff,
            ple: if r % 2 == 0 { 0 } else { u16::MAX },
        },
        13 => Frame::ExpertResidency {
            layer,
            in_vram: vec![(r % 512) as u16, ((r + 7) % 512) as u16],
        },
        14 => Frame::TierStats {
            vram_used_mb: 512 + r,
            vram_total_mb: 24_576,
            ram_experts: r % 1024,
            ssd_reads: uniq.wrapping_mul(17),
        },
        15 => Frame::RouterTopK {
            layer,
            experts: vec![(r % 512) as u16, ((r + 1) % 512) as u16, ((r + 2) % 512) as u16],
            weights: vec![0x3c00, 0x3000, 0x2800],
        },
        _ => Frame::ExpertSkip {
            layer,
            experts: vec![(r % 512) as u16, ((r + 11) % 512) as u16],
        },
    }
}

/// 1 スレッド分の入力フレーム列。17 kind を ROUNDS 回ぶん。
fn batch_frames(t: usize) -> Vec<Frame> {
    let mut frames = Vec::with_capacity((ROUNDS * KINDS) as usize);
    for r in 0..ROUNDS {
        for k in 0..KINDS {
            frames.push(batch_frame(t, r * KINDS + k));
        }
    }
    frames
}

/// バイナリ列に書き出す (1 フレームずつ append)。
fn encode_wire(frames: &[Frame]) -> Vec<u8> {
    let mut wire = Vec::new();
    for f in frames {
        wire.extend_from_slice(&f.encode());
    }
    wire
}

/// フレームを 1 枚ずつ [`Frame::decode`] して戻す。
fn decode_wire(wire: &[u8], frames: &[Frame]) -> Vec<Frame> {
    let mut out = Vec::with_capacity(frames.len());
    let mut off = 0usize;
    for (i, expected) in frames.iter().enumerate() {
        let mut size = expected.encode().len();
        // 期待サイズで 1 枚読む。一致しなければ 1 バイトずつ広げて探す
        // (encode が決定的なので通常は 1 回で当たる)。
        loop {
            let end = off + size;
            assert!(end <= wire.len(), "binary 列が途中で切れた (frame {i})");
            match Frame::decode(&wire[off..end]) {
                Ok(f) => {
                    out.push(f);
                    off = end;
                    break;
                }
                Err(e) if size < wire.len() - off => {
                    let _ = e;
                    size += 1;
                }
                Err(e) => panic!("binary decode 失敗 (frame {i}): {e}"),
            }
        }
    }
    assert_eq!(off, wire.len(), "binary 列の末尾に余りがある: {off} / {}", wire.len());
    out
}

/// バイナリ -> Frame -> NDJSON -> Frame -> バイナリ を往復し、Frame 列を返す。
fn convert_roundtrip(frames: &[Frame]) -> Vec<Frame> {
    // 1) バイナリ化して戻す。
    let wire = encode_wire(frames);
    let from_binary = decode_wire(&wire, frames);
    assert_eq!(from_binary, frames, "バイナリ往復でデータが変わった");

    // 2) NDJSON 化してストリームで読み戻す (小さめバッファで行境界跨ぎを踏む)。
    let mut ndjson = Vec::new();
    let written = write_frames_ndjson(&mut ndjson, &from_binary).expect("ndjson write");
    assert_eq!(written, from_binary.len(), "write_frames_ndjson の戻り値が違う");
    assert_eq!(ndjson.iter().filter(|&&b| b == b'\n').count(), frames.len(), "行数が違う");

    let reader = BufReader::with_capacity(37, Cursor::new(&ndjson));
    let (from_ndjson, errors) = read_frames_ndjson(reader);
    assert!(errors.is_empty(), "有効なストリームをエラーにした: {errors:?}");
    assert_eq!(from_ndjson, frames, "NDJSON 往復でデータが変わった");

    // 3) 1 行ずつ parse しても同じ結果になる。
    for (i, f) in from_ndjson.iter().enumerate() {
        let line = frame_to_ndjson(f);
        let back = parse_frame_line(&line).unwrap_or_else(|e| panic!("line {i} parse: {e}"));
        assert_eq!(&back, f, "行単位の往復で変わった (line {i})");
    }

    // 4) NDJSON -> Frame -> バイナリが元のバイト列と一致する (決定的)。
    let wire2 = encode_wire(&from_ndjson);
    assert_eq!(wire2, wire, "NDJSON を挟むとバイナリ列が変わった");

    // 5) もう一度書いても同じ NDJSON になる (非決定性の検出)。
    let mut ndjson2 = Vec::new();
    write_frames_ndjson(&mut ndjson2, &from_ndjson).expect("ndjson rewrite");
    assert_eq!(ndjson2, ndjson, "NDJSON 出力が非決定的");

    from_ndjson
}

#[test]
fn eight_threads_roundtrip_mixed_17_kinds_with_zero_loss() {
    // 全スレッドを同時に解放するためのバリア。
    let barrier = Arc::new(Barrier::new(THREADS));
    // 各スレッドが回収したフレーム数 (全体の損失検出用)。
    let total_frames = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..THREADS)
        .map(|t| {
            let barrier = Arc::clone(&barrier);
            let total_frames = Arc::clone(&total_frames);
            thread::spawn(move || {
                let expected = batch_frames(t);
                assert_eq!(expected.len(), (ROUNDS * KINDS) as usize);

                // 全 kind が入っていること (取りこぼし検出)。
                let mut kinds: Vec<u8> = expected.iter().map(Frame::kind).collect();
                kinds.sort_unstable();
                kinds.dedup();
                assert_eq!(
                    kinds,
                    (1..=17).collect::<Vec<u8>>(),
                    "thread {t} の入力が 17 kind を網羅していない"
                );

                barrier.wait();

                // 往復変換 (writer と reader の両方の役割)。
                let got = convert_roundtrip(&expected);

                // ゼロデータ損失: 枚数と内容が完全一致。
                assert_eq!(got.len(), expected.len(), "thread {t}: フレーム数が減った");
                assert_eq!(got, expected, "thread {t}: フレーム内容が変わった");

                total_frames.fetch_add(got.len(), Ordering::SeqCst);
                got
            })
        })
        .collect();

    let mut all: Vec<Vec<Frame>> = Vec::with_capacity(THREADS);
    for h in handles {
        all.push(h.join().expect("worker thread panicked"));
    }

    // 全スレッドの合計枚数が期待どおり (重複加算・欠落の検出)。
    let expected_total = THREADS * (ROUNDS * KINDS) as usize;
    assert_eq!(
        total_frames.load(Ordering::SeqCst),
        expected_total,
        "合計フレーム数が期待と違う"
    );

    // スレッド間でフレームが混ざっていないこと: 各スレッドの結果は
    // 自分が書いた列とだけ一致する。
    for (t, got) in all.iter().enumerate() {
        let expected = batch_frames(t);
        assert_eq!(got, &expected, "thread {t} の結果が他のスレッドの入力と混ざった");
    }

    // 全スレッドのフレームを連結しても、入力の連結と完全一致する。
    let mut flat_expected: Vec<Frame> = Vec::with_capacity(expected_total);
    let mut flat_got: Vec<Frame> = Vec::with_capacity(expected_total);
    for t in 0..THREADS {
        flat_expected.extend(batch_frames(t));
        flat_got.extend(all[t].iter().cloned());
    }
    assert_eq!(flat_got.len(), expected_total);
    assert_eq!(flat_got, flat_expected, "スレッド横断でデータが変わった");
}

#[test]
fn concurrent_readers_of_shared_stream_see_full_content() {
    // 8 本のスレッドが **同じ** NDJSON バイト列を同時に読む。各スレッドが
    // 全フレームを回収できれば、リーダー側に共有可変状態の取りこぼしが無い。
    let frames = batch_frames(0);
    let mut ndjson = Vec::new();
    write_frames_ndjson(&mut ndjson, &frames).expect("ndjson write");
    let ndjson = Arc::new(ndjson);

    let barrier = Arc::new(Barrier::new(THREADS));
    let handles: Vec<_> = (0..THREADS)
        .map(|_| {
            let ndjson = Arc::clone(&ndjson);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                let reader = BufReader::with_capacity(41, Cursor::new(ndjson.as_slice()));
                let (got, errors) = read_frames_ndjson(reader);
                assert!(errors.is_empty(), "共有ストリームでエラー: {errors:?}");
                got
            })
        })
        .collect();

    for h in handles {
        let got = h.join().expect("reader thread panicked");
        assert_eq!(got.len(), frames.len(), "共有ストリームの読み取り枚数が違う");
        assert_eq!(got, frames, "共有ストリームの内容が違う");
    }
}
