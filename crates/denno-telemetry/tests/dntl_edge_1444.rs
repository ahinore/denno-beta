//! task 1444: DNTL フレーム codec の境界と送出ハブの溢れ / 無購読者時の安全性。
//!
//! 検証すること:
//! - 全 17 kind の encode → decode 往復 (再 encode も同一バイト列)
//! - varint / f16 / 文字列長の境界値でも往復が崩れない
//! - magic / version / kind / 長さヘッダの破損・切り詰めを panic せず Err で弾く
//! - 容量ちょうどの溢れで `dropped()` が差分計上され、relay 規則で作った
//!   `Gap { lost }` がそのまま往復・受信できる
//! - 購読者 0 人でも emit は panic せず、溢れはドロップ計上だけで済む
//!
//! ハブはプロセス全体で 1 本の static チャネルなので、ハブを触るテストは
//! [`serial`] で直列化する。時刻依存の待ちは使わず、全て non-blocking で行う。

use denno_telemetry as telem;
use denno_telemetry::{put_varint_u32, ClientsGuard, Frame, MAGIC, VERSION};
use std::sync::{Mutex, MutexGuard};

/// グローバルハブを共有するテストを直列化する。
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// ハブに溜まっているフレームを全部取り出す (非ブロッキング)。
fn drain_all() -> Vec<Frame> {
    let mut out = Vec::new();
    while let Some(f) = telem::try_recv_nowait() {
        out.push(f);
    }
    out
}

/// テスト用にハブを有効化し、前のテストの残りを捨てる。
fn setup() {
    telem::set_enabled_for_tests(true);
    assert!(telem::enabled(), "テレメトリを有効化できていない");
    let _ = drain_all();
}

/// 全 kind を 1 枚ずつ (往復・切り詰め・破損テストの入力)。
fn sample_frames() -> Vec<Frame> {
    vec![
        Frame::ServerInfo {
            model: "denno-model".into(),
            model_id: "m-1".into(),
            quant: "q4".into(),
            slots: 4,
            context: 32_768,
            layers: 24,
        },
        Frame::SessionAdmit { session: 7, slot: 2, prompt_tokens: 34 },
        Frame::SessionDone { session: 7, reason: telem::done_reason::MAX_TOKENS, output_tokens: 26 },
        Frame::Prefill {
            session: 7,
            prompt_tokens: 34,
            ns: 123_456_789,
            tokens: vec![972, 198, 3_585, 370, 151_644],
        },
        Frame::Token { session: 7, token: 99_999, pos: 12 },
        Frame::Step { width: 4, ns: 8_765_432 },
        Frame::OpSummary { ops: vec![(4, 1_000), (11, 2_500)] },
        Frame::KvUsage { busy: vec![true, false, true, true], t_lens: vec![10, 0, 300, 777] },
        Frame::Gap { lost: 42 },
        Frame::TopK {
            session: 3,
            chosen: 831,
            entropy_f16: telem::f32_to_f16_bits(1.25),
            entries: vec![
                (831, telem::f32_to_f16_bits(0.5)),
                (12, telem::f32_to_f16_bits(0.25)),
            ],
        },
        Frame::LayerRun { layer: 5, ops: vec![(1, 2_100), (3, 154_000), (2, 8_800)] },
        Frame::LayerAct {
            layer: 47,
            idx: vec![0, 1_531, 2_560],
            val: vec![0x3c00, 0x3800, 0x0001],
        },
        Frame::RoleAct {
            layer: 47,
            attn: 0,
            gdn: 0x3c00,
            router: 0x3800,
            shex: 0,
            ple: 1,
        },
        Frame::ExpertResidency { layer: 5, in_vram: (0..telem::RESIDENCY_MAX as u16).collect() },
        Frame::TierStats {
            vram_used_mb: 4_096,
            vram_total_mb: 8_192,
            ram_experts: 64,
            ssd_reads: 1_234_567,
        },
        Frame::RouterTopK {
            layer: 3,
            experts: (0..10u16).map(|i| 502 + i).collect(),
            weights: (0..10u16).map(|i| 0x3c00 - i * 13).collect(),
        },
        Frame::ExpertSkip { layer: 47, experts: (0..10u16).map(|i| 300 + i).collect() },
    ]
}

/// 全 kind が往復し、再 encode が同一バイト列になること。
#[test]
fn roundtrip_every_kind_is_canonical() {
    let frames = sample_frames();
    let mut kinds = std::collections::BTreeSet::new();
    for f in &frames {
        kinds.insert(f.kind());
    }
    assert_eq!(kinds.len(), 17, "全 17 kind を網羅していない: {kinds:?}");

    for f in frames {
        let enc = f.encode();
        assert_eq!(&enc[..4], &MAGIC, "magic が違う kind {}", f.kind());
        assert_eq!(enc[4], VERSION, "version が違う kind {}", f.kind());
        assert_eq!(enc[5], f.kind(), "kind バイトが違う kind {}", f.kind());

        let dec = Frame::decode(&enc)
            .unwrap_or_else(|e| panic!("decode 失敗 kind {}: {e}", f.kind()));
        assert_eq!(dec, f, "往復で値が変わる kind {}", f.kind());
        assert_eq!(dec.encode(), enc, "再 encode が非正規 kind {}", f.kind());
        assert_eq!(dec.kind(), f.kind(), "復号後の kind が違う");

        // NDJSON の kind 名も全 kind で双方向に引ける。
        let name = telem::ndjson_kind_name(f.kind()).expect("kind 名が無い");
        assert_eq!(telem::ndjson_kind_from_name(name), Some(f.kind()));
    }
}

/// varint / f16 / 文字列長の境界値でも往復が保たれること。
#[test]
fn roundtrip_boundary_values() {
    let cases = vec![
        Frame::Gap { lost: 0 },
        Frame::Gap { lost: 127 },
        Frame::Gap { lost: 128 },
        Frame::Gap { lost: 16_383 },
        Frame::Gap { lost: 16_384 },
        Frame::Gap { lost: u32::MAX },
        Frame::Token { session: u32::MAX, token: 0, pos: u32::MAX },
        Frame::Step { width: u32::MAX, ns: u64::MAX },
        Frame::ServerInfo {
            model: String::new(),
            model_id: String::new(),
            quant: String::new(),
            slots: 0,
            context: u32::MAX,
            layers: 0,
        },
        Frame::Prefill { session: 0, prompt_tokens: 0, ns: 0, tokens: Vec::new() },
        Frame::KvUsage { busy: Vec::new(), t_lens: Vec::new() },
        Frame::LayerAct { layer: 0, idx: Vec::new(), val: Vec::new() },
        Frame::ExpertResidency { layer: u16::MAX, in_vram: Vec::new() },
        Frame::RouterTopK { layer: u16::MAX, experts: Vec::new(), weights: Vec::new() },
        Frame::ExpertSkip { layer: 0, experts: vec![u16::MAX] },
        Frame::RoleAct {
            layer: u16::MAX,
            attn: u16::MAX,
            gdn: u16::MAX,
            router: 0,
            shex: 0x7c00,
            ple: 0xfc00,
        },
    ];
    for f in cases {
        let enc = f.encode();
        let dec = Frame::decode(&enc)
            .unwrap_or_else(|e| panic!("境界値の decode 失敗 kind {}: {e}", f.kind()));
        assert_eq!(dec, f, "境界値の往復失敗 kind {}", f.kind());
        assert_eq!(dec.encode(), enc, "境界値の再 encode が違う kind {}", f.kind());
    }
}

/// 先頭からの全切り詰めが Err になること (panic しないこと自体も検証)。
#[test]
fn every_truncated_prefix_is_rejected() {
    let frames = vec![
        Frame::ServerInfo {
            model: "abc".into(),
            model_id: "m".into(),
            quant: "q4".into(),
            slots: 1,
            context: 2,
            layers: 3,
        },
        Frame::Prefill { session: 1, prompt_tokens: 2, ns: 3, tokens: vec![4, 5] },
        Frame::TopK { session: 1, chosen: 2, entropy_f16: 0x3c00, entries: vec![(9, 0x3800)] },
        Frame::LayerRun { layer: 1, ops: vec![(2, 3)] },
        Frame::LayerAct { layer: 1, idx: vec![2], val: vec![0x3c00] },
        Frame::ExpertResidency { layer: 1, in_vram: vec![2, 3] },
        Frame::RouterTopK { layer: 1, experts: vec![2], weights: vec![0x3c00] },
        Frame::Gap { lost: 128 },
    ];
    for f in frames {
        let enc = f.encode();
        for cut in 0..enc.len() {
            match Frame::decode(&enc[..cut]) {
                Err(_) => {}
                Ok(got) => panic!(
                    "kind {}: 先頭 {cut}/{} バイトの切り詰めが通った: {got:?}",
                    f.kind(),
                    enc.len()
                ),
            }
        }
    }
}

/// magic の 1 バイト破損 / version 破損 / 未知 kind をすべて弾くこと。
#[test]
fn corrupted_magic_version_and_kind_are_rejected() {
    let good = Frame::Token { session: 1, token: 2, pos: 3 }.encode();

    // magic の各バイトを 1 つずつ壊す。
    for i in 0..4 {
        for bad in [0x00u8, 0x44, 0xff] {
            if bad == good[i] {
                continue;
            }
            let mut buf = good.clone();
            buf[i] = bad;
            let err = Frame::decode(&buf).expect_err(&format!("magic byte {i} 破損が通った"));
            assert!(err.contains("magic"), "magic 破損のエラー文言: {err}");
        }
    }

    // version を変える。
    for v in [0u8, 2, 255] {
        let mut buf = good.clone();
        buf[4] = v;
        let err = Frame::decode(&buf).expect_err("version 破損が通った");
        assert!(err.contains("version"), "version 破損のエラー文言: {err}");
    }

    // 未知 kind (0 と 17 より上) は unknown kind。
    for k in [0u8, 18, 99, 255] {
        let mut buf = good.clone();
        buf[5] = k;
        let err = Frame::decode(&buf).expect_err("未知 kind が通った");
        assert!(err.contains("unknown kind"), "kind {k} のエラー文言: {err}");
    }

    // 6 バイト未満は magic 判定の前に弾く。
    assert!(Frame::decode(&[]).is_err());
    assert!(Frame::decode(b"DNT").is_err());
    assert!(Frame::decode(&MAGIC).is_err());
}

/// 長さヘッダ (文字列長 / 要素数) の破損・過大値を確保せず弾くこと。
#[test]
fn corrupted_length_headers_are_rejected() {
    // ServerInfo の model 長 = u32::MAX、本体なし。
    let mut v = Vec::new();
    v.extend_from_slice(&MAGIC);
    v.push(VERSION);
    v.push(telem::kind::SERVER_INFO);
    put_varint_u32(&mut v, u32::MAX);
    assert!(Frame::decode(&v).is_err(), "過大な文字列長が通った");
    // 長さ 5 と言いながら 2 バイトしか無い。
    v.push(b'a');
    v.push(b'b');
    assert!(Frame::decode(&v).is_err(), "実データ不足の文字列長が通った");

    // Prefill の要素数 = u32::MAX、残り 0 バイト。
    let mut v = Vec::new();
    v.extend_from_slice(&MAGIC);
    v.push(VERSION);
    v.push(telem::kind::PREFILL);
    put_varint_u32(&mut v, 1); // session
    put_varint_u32(&mut v, 1); // prompt_tokens
    put_varint_u32(&mut v, 0); // ns
    put_varint_u32(&mut v, u32::MAX); // count
    let err = Frame::decode(&v).expect_err("過大 count が通った");
    assert!(err.contains("count") || err.contains("truncated"), "{err}");

    // count の varint 自体が途中で切れている (0x80 で終端しない)。
    let mut v = Vec::new();
    v.extend_from_slice(&MAGIC);
    v.push(VERSION);
    v.push(telem::kind::OP_SUMMARY);
    v.push(0x80);
    let err = Frame::decode(&v).expect_err("varint 切り詰めが通った");
    assert!(err.contains("truncated"), "{err}");

    // 文字列長が残りバイト数を超える (長さ 3 / 実データ 2)。
    let mut v = Vec::new();
    v.extend_from_slice(&MAGIC);
    v.push(VERSION);
    v.push(telem::kind::SERVER_INFO);
    put_varint_u32(&mut v, 3);
    v.extend_from_slice(b"ab");
    assert!(Frame::decode(&v).is_err(), "長さ超過の ServerInfo が通った");

    // 余剰バイトは trailing として弾く。
    let mut enc = Frame::Gap { lost: 1 }.encode();
    enc.push(0);
    let err = Frame::decode(&enc).expect_err("余剰バイトが通った");
    assert!(err.contains("trailing"), "{err}");
}

/// 容量ちょうどまで溜まり、溢れ分がドロップ計上されること。
#[test]
fn bounded_hub_overflow_is_counted_without_blocking() {
    let _s = serial();
    setup();
    let cap = telem::channel_cap();
    assert!(cap >= 64, "capacity が小さすぎる: {cap}");

    let base = telem::dropped();
    for i in 0..cap {
        telem::emit(Frame::Token { session: 1, token: i as u32, pos: i as u32 });
    }
    assert_eq!(telem::dropped(), base, "容量ちょうどまでは落ちない");

    // 溢れさせる (relay が観測するのはこの差分)。
    for _ in 0..7 {
        telem::emit(Frame::Gap { lost: 1 });
    }
    assert_eq!(telem::dropped() - base, 7, "溢れ 7 枚がドロップ計上されない");

    let got = drain_all();
    assert_eq!(got.len(), cap, "受信は容量ぶんだけ");
    assert!(
        got.iter().all(|f| !matches!(f, Frame::Gap { .. })),
        "ハブが Gap を合成している (購読側の責務)"
    );
}

/// relay 規則 (ドロップ差分 → Gap 1 枚) で作った GAP が送出・往復できること。
#[test]
fn overflow_gap_frame_emission_roundtrips() {
    let _s = serial();
    setup();
    let cap = telem::channel_cap();

    for lost_expected in [1u64, 2, 127, 128, 300] {
        let _ = drain_all();
        let before = telem::dropped();
        let send = cap as u64 + lost_expected;
        for i in 0..send {
            telem::emit(Frame::Token { session: 2, token: i as u32, pos: i as u32 });
        }
        let lost = telem::dropped() - before;
        assert_eq!(lost, lost_expected, "overflow {lost_expected} の計上が違う");

        // 購読側の責務として 1 枚の Gap を作る (u32 に飽和)。
        let gap = Frame::Gap { lost: lost.min(u32::MAX as u64) as u32 };
        assert_eq!(gap.kind(), telem::kind::GAP);
        let enc = gap.encode();
        assert_eq!(&enc[..4], &MAGIC);
        assert_eq!(enc[4], VERSION);
        assert_eq!(enc[5], telem::kind::GAP);
        assert_eq!(Frame::decode(&enc).expect("Gap の復号"), gap);

        // 溢れたフレームを読み切ってから Gap を流すと、そのまま受信できる。
        let got = drain_all();
        assert_eq!(got.len(), cap, "受信は容量ぶん");
        telem::emit(gap.clone());
        let mut tail = drain_all();
        assert_eq!(tail.len(), 1, "Gap が受信できない");
        let recv = tail.pop().expect("Gap が無い");
        assert_eq!(recv, gap, "受信した Gap の lost が違う");
        assert_eq!(Frame::decode(&recv.encode()).expect("受信 Gap の復号"), gap);

        // 会計: 送出 = 受信 + ドロップ。
        assert_eq!(got.len() as u64 + lost, send, "会計が合わない");
    }
}

/// 購読者 0 人でも溢れはドロップ計上だけで、panic も合成 Gap もしないこと。
#[test]
fn zero_subscriber_overflow_is_drop_only() {
    let _s = serial();
    setup();
    let clients = ClientsGuard::new(0);
    assert_eq!(telem::clients(), 0, "購読者数が 0 になっていない");
    // lazy が既定 on なら「読む相手なし」と判定される。
    assert_eq!(telem::has_clients(), !telem::lazy_enabled(), "0 人時の判定");

    let cap = telem::channel_cap();
    let base = telem::dropped();
    for i in 0..cap + 3 {
        telem::emit(Frame::Token { session: 3, token: i as u32, pos: i as u32 });
    }
    assert_eq!(telem::dropped() - base, 3, "購読者 0 人時の溢れ計上");
    let got = drain_all();
    assert_eq!(got.len(), cap, "受信数が容量と違う");
    assert!(
        !got.iter().any(|f| matches!(f, Frame::Gap { .. })),
        "購読者 0 人でハブが Gap を合成した"
    );

    drop(clients);
    assert_eq!(telem::clients(), 0, "ClientsGuard が戻していない");
}

/// 購読者 0 人でも GAP フレーム自体は欠落なく届くこと (ドロップと混同しない)。
#[test]
fn zero_subscriber_gap_frame_is_not_dropped() {
    let _s = serial();
    setup();
    let clients = ClientsGuard::new(0);
    assert_eq!(telem::clients(), 0);

    let base = telem::dropped();
    let gap = Frame::Gap { lost: 7 };
    telem::emit(gap.clone());
    assert_eq!(telem::dropped(), base, "容量内の Gap がドロップ計上された");

    let got = drain_all();
    assert_eq!(got, vec![gap], "購読者 0 人で Gap が消えた");

    // 購読者 0 人のまま、溢れさせても GAP の値そのものは壊れない。
    let cap = telem::channel_cap();
    for i in 0..cap + 1 {
        telem::emit(Frame::Token { session: 4, token: i as u32, pos: i as u32 });
    }
    let lost = telem::dropped() - base;
    assert_eq!(lost, 1, "溢れ 1 枚の計上");
    let tail_gap = Frame::Gap { lost: lost as u32 };
    assert_eq!(Frame::decode(&tail_gap.encode()).expect("Gap 往復"), tail_gap);
    assert_eq!(drain_all().len(), cap, "受信は容量ぶん");

    drop(clients);
    assert_eq!(telem::clients(), 0);
}
