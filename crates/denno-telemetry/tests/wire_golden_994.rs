//! task 994: ワイヤ形式のバイトレベル golden テスト。
//!
//! ここでは `Frame` の全 variant について **期待バイト列をリテラルで固定** する。
//! `encode` の実装を書き換えて 1 バイトでも並びが変われば、このテストが落ちる。
//! 仕様は `docs/TELEMETRY_PROTOCOL.md` (同じ task で追加) を参照。
//!
//! 固定している不変条件:
//!
//! - 封筒: magic `DNTL` / version 1 / kind 1 バイト
//! - payload の整数は LEB128 varint、f16 は 2 バイト little-endian
//! - 文字列は長さ varint + UTF-8 生バイト
//! - `bool` は 1 バイト (encode は 0 / 1)
//! - 長さの食い違う並列配列は短い方に切り揃える
//! - 切り詰め・未知 kind・未知 version・末尾の余分バイトは Err (panic しない)

use denno_telemetry::{
    f32_to_f16_bits, kind, ndjson_kind_from_name, ndjson_kind_name, put_varint_u32,
    put_varint_u64, Frame, MAGIC, VERSION,
};

/// 封筒 (magic + version + kind) + payload。
fn frame_bytes(k: u8, payload: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(6 + payload.len());
    v.extend_from_slice(&MAGIC);
    v.push(VERSION);
    v.push(k);
    v.extend_from_slice(payload);
    v
}

/// kind バイト → 1 枚のフレーム (golden 表の入力)。
fn golden_frames() -> Vec<(Frame, u8, Vec<u8>)> {
    vec![
        (
            Frame::ServerInfo {
                model: "m".into(),
                model_id: "".into(),
                quant: "fp32".into(),
                slots: 1,
                context: 2,
                layers: 3,
            },
            kind::SERVER_INFO,
            // len 1 "m" / len 0 / len 4 "fp32" / 1 / 2 / 3
            b"\x01m\x00\x04fp32\x01\x02\x03".to_vec(),
        ),
        (
            Frame::SessionAdmit { session: 1, slot: 2, prompt_tokens: 300 },
            kind::SESSION_ADMIT,
            // 1 / 2 / 300 = 0xAC 0x02
            b"\x01\x02\xac\x02".to_vec(),
        ),
        (
            Frame::SessionDone { session: 7, reason: 1, output_tokens: 26 },
            kind::SESSION_DONE,
            // 7 / reason 1 (raw u8) / 26
            b"\x07\x01\x1a".to_vec(),
        ),
        (
            Frame::Prefill {
                session: 1,
                prompt_tokens: 2,
                ns: 128,
                tokens: vec![1, 2],
            },
            kind::PREFILL,
            // 1 / 2 / 128 = 0x80 0x01 / count 2 / tokens 1,2
            b"\x01\x02\x80\x01\x02\x01\x02".to_vec(),
        ),
        (
            Frame::Token { session: 1, token: 42, pos: 7 },
            kind::TOKEN,
            b"\x01\x2a\x07".to_vec(),
        ),
        (
            Frame::Step { width: 4, ns: 8_765_432 },
            kind::STEP,
            // 4 / 8765432 = 0xF8 0xFF 0x96 0x04
            b"\x04\xf8\xff\x96\x04".to_vec(),
        ),
        (
            Frame::OpSummary { ops: vec![(4, 1_000), (11, 2_500)] },
            kind::OP_SUMMARY,
            // count 2 / op 4 + ns 1000 (0xE8 0x07) / op 11 + ns 2500 (0xC4 0x13)
            b"\x02\x04\xe8\x07\x0b\xc4\x13".to_vec(),
        ),
        (
            Frame::KvUsage { busy: vec![true, false], t_lens: vec![10, 0] },
            kind::KV_USAGE,
            // count 2 / busy 1,0 / t_lens 10,0
            b"\x02\x01\x00\x0a\x00".to_vec(),
        ),
        (
            Frame::Gap { lost: 42 },
            kind::GAP,
            b"\x2a".to_vec(),
        ),
        (
            Frame::TopK {
                session: 3,
                chosen: 831,
                entropy_f16: 0x3c00,
                entries: vec![(5, 0x3800)],
            },
            kind::TOP_K,
            // 3 / 831 = 0xBF 0x06 / entropy LE 0x00 0x3C / count 1 / id 5 / p LE 0x00 0x38
            b"\x03\xbf\x06\x00\x3c\x01\x05\x00\x38".to_vec(),
        ),
        (
            Frame::LayerRun { layer: 5, ops: vec![(1, 1_000)] },
            kind::LAYER_RUN,
            // layer 5 / count 1 / op 1 + ns 1000
            b"\x05\x01\x01\xe8\x07".to_vec(),
        ),
        (
            Frame::LayerAct {
                layer: 12,
                idx: vec![0, 1],
                val: vec![0x3c00, 0x3800],
            },
            kind::LAYER_ACT,
            // layer 12 / count 2 / idx 0,1 / val LE 00 3C, 00 38
            b"\x0c\x02\x00\x01\x00\x3c\x00\x38".to_vec(),
        ),
        (
            Frame::RoleAct { layer: 47, attn: 0x3c00, gdn: 0x3800, router: 0, shex: 1, ple: 0 },
            kind::ROLE_ACT,
            // layer 47 / 5 × f16 LE 固定順 (attn, gdn, router, shex, ple)
            b"\x2f\x00\x3c\x00\x38\x00\x00\x01\x00\x00\x00".to_vec(),
        ),
        (
            Frame::ExpertResidency { layer: 4, in_vram: vec![7] },
            kind::EXPERT_RESIDENCY,
            b"\x04\x01\x07".to_vec(),
        ),
        (
            Frame::TierStats {
                vram_used_mb: 4_096,
                vram_total_mb: 8_192,
                ram_experts: 64,
                ssd_reads: 1_234_567,
            },
            kind::TIER_STATS,
            // 4096 = 80 20 / 8192 = 80 40 / 64 = 40 / 1234567 = 87 AD 4B
            b"\x80\x20\x80\x40\x40\x87\xad\x4b".to_vec(),
        ),
        (
            Frame::RouterTopK { layer: 3, experts: vec![511], weights: vec![0x3c00] },
            kind::ROUTER_TOPK,
            // layer 3 / count 1 / expert 511 = 0xFF 0x03 / weight LE 00 3C
            b"\x03\x01\xff\x03\x00\x3c".to_vec(),
        ),
        (
            Frame::ExpertSkip { layer: 5, experts: vec![0, 1] },
            kind::EXPERT_SKIP,
            b"\x05\x02\x00\x01".to_vec(),
        ),
    ]
}

#[test]
fn golden_bytes_match_literals() {
    for (frame, k, payload) in golden_frames() {
        let want = frame_bytes(k, &payload);
        let got = frame.encode();
        assert_eq!(
            got,
            want,
            "kind {} ({:?}) のワイヤ表現が変わった: got={:02x?} want={:02x?}",
            k,
            frame,
            got,
            want
        );
        // 封筒
        assert_eq!(&got[..4], b"DNTL");
        assert_eq!(got[4], VERSION);
        assert_eq!(got[5], frame.kind(), "kind() と封筒の kind が違う");
    }
}

#[test]
fn golden_bytes_decode_back_to_same_frame() {
    for (frame, k, payload) in golden_frames() {
        let bytes = frame_bytes(k, &payload);
        assert_eq!(
            Frame::decode(&bytes),
            Ok(frame.clone()),
            "golden バイト列が decode できない (kind {k})"
        );
    }
}

/// kind 番号はブランチを跨いで固定。詰め直し・再利用を禁止する。
#[test]
fn kind_byte_table_is_frozen() {
    let table: [(u8, &str); 17] = [
        (1, "server_info"),
        (2, "session_admit"),
        (3, "session_done"),
        (4, "prefill"),
        (5, "token"),
        (6, "step"),
        (7, "op_summary"),
        (8, "kv_usage"),
        (9, "gap"),
        (10, "top_k"),
        (11, "layer_run"),
        (12, "layer_act"),
        (13, "role_act"),
        (14, "expert_residency"),
        (15, "tier_stats"),
        (16, "router_topk"),
        (17, "expert_skip"),
    ];
    for (i, (b, name)) in table.iter().enumerate() {
        assert_eq!(*b as usize, i + 1, "kind 番号に穴が空いている");
        assert_eq!(ndjson_kind_name(*b), Some(*name));
        assert_eq!(ndjson_kind_from_name(name), Some(*b));
    }
    // 全 variant が表の kind を返す (取りこぼし検出)。
    let kinds: Vec<u8> = golden_frames().iter().map(|(f, _, _)| f.kind()).collect();
    let mut want: Vec<u8> = table.iter().map(|(b, _)| *b).collect();
    want.sort_unstable();
    let mut got = kinds;
    got.sort_unstable();
    assert_eq!(got, want, "golden 表に無い / 重複した kind がある");
}

#[test]
fn varint_encoding_is_leb128() {
    let cases: [(u32, &[u8]); 8] = [
        (0, b"\x00"),
        (1, b"\x01"),
        (127, b"\x7f"),
        (128, b"\x80\x01"),
        (300, b"\xac\x02"),
        (16_383, b"\xff\x7f"),
        (16_384, b"\x80\x80\x01"),
        (u32::MAX, b"\xff\xff\xff\xff\x0f"),
    ];
    for (v, want) in cases {
        let mut out = Vec::new();
        put_varint_u32(&mut out, v);
        assert_eq!(out, want, "u32 {v} の varint が違う");
        let mut out64 = Vec::new();
        put_varint_u64(&mut out64, v as u64);
        assert_eq!(out64, want, "u64 {v} の varint が違う");
    }
    let mut out = Vec::new();
    put_varint_u64(&mut out, u64::MAX);
    assert_eq!(out, b"\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01");
}

#[test]
fn f16_bits_are_little_endian_in_frames() {
    // 0x3c00 = 1.0, 0x3800 = 0.5, 0xc000 = -2.0
    assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
    assert_eq!(f32_to_f16_bits(0.5), 0x3800);
    assert_eq!(f32_to_f16_bits(-2.0), 0xc000);
    let f = Frame::RoleAct { layer: 0, attn: 0x3c00, gdn: 0x3800, router: 0xc000, shex: 0, ple: 0 };
    let enc = f.encode();
    // 封筒 6B + layer 1B の後ろに attn/gdn/router が LE で並ぶ。
    assert_eq!(&enc[7..13], &[0x00, 0x3c, 0x00, 0x38, 0x00, 0xc0]);
}

/// 長さの食い違う並列配列は短い方に切り揃える (decode はその本数だけ読む)。
/// この規則を変えると既存の壊れたフレームが decode できなくなるので固定する。
#[test]
fn mismatched_parallel_arrays_are_truncated_to_shortest() {
    let kv = Frame::KvUsage {
        busy: vec![true, false],
        t_lens: vec![1, 2, 3],
    };
    let want_kv = frame_bytes(kind::KV_USAGE, b"\x02\x01\x00\x01\x02");
    assert_eq!(kv.encode(), want_kv);
    assert_eq!(
        Frame::decode(&want_kv),
        Ok(Frame::KvUsage { busy: vec![true, false], t_lens: vec![1, 2] })
    );

    let la = Frame::LayerAct { layer: 1, idx: vec![4, 5], val: vec![0x3c00] };
    let want_la = frame_bytes(kind::LAYER_ACT, b"\x01\x01\x04\x00\x3c");
    assert_eq!(la.encode(), want_la);
    assert_eq!(
        Frame::decode(&want_la),
        Ok(Frame::LayerAct { layer: 1, idx: vec![4], val: vec![0x3c00] })
    );

    let rt = Frame::RouterTopK { layer: 2, experts: vec![7, 8], weights: vec![0x3c00] };
    let want_rt = frame_bytes(kind::ROUTER_TOPK, b"\x02\x01\x07\x00\x3c");
    assert_eq!(rt.encode(), want_rt);
    assert_eq!(
        Frame::decode(&want_rt),
        Ok(Frame::RouterTopK { layer: 2, experts: vec![7], weights: vec![0x3c00] })
    );
}

/// 空の配列は count 0 だけを書き、封筒 6B + 最小フィールドで終わる。
#[test]
fn empty_payload_frames_are_minimal() {
    let cases: [(Frame, u8, &[u8]); 7] = [
        (Frame::OpSummary { ops: Vec::new() }, kind::OP_SUMMARY, b"\x00"),
        (Frame::KvUsage { busy: Vec::new(), t_lens: Vec::new() }, kind::KV_USAGE, b"\x00"),
        (Frame::TopK { session: 0, chosen: 0, entropy_f16: 0, entries: Vec::new() }, kind::TOP_K, b"\x00\x00\x00\x00\x00"),
        (Frame::LayerRun { layer: 0, ops: Vec::new() }, kind::LAYER_RUN, b"\x00\x00"),
        (Frame::LayerAct { layer: 0, idx: Vec::new(), val: Vec::new() }, kind::LAYER_ACT, b"\x00\x00"),
        (Frame::ExpertResidency { layer: 0, in_vram: Vec::new() }, kind::EXPERT_RESIDENCY, b"\x00\x00"),
        (Frame::ExpertSkip { layer: 0, experts: Vec::new() }, kind::EXPERT_SKIP, b"\x00\x00"),
    ];
    for (f, k, payload) in cases {
        let want = frame_bytes(k, payload);
        assert_eq!(f.encode(), want, "kind {k} の空フレーム");
        assert_eq!(Frame::decode(&want), Ok(f));
    }
}

/// golden の全バイト列について、1 バイトでも削ると decode は Err (panic しない)。
#[test]
fn every_truncation_of_golden_is_rejected() {
    for (_, k, payload) in golden_frames() {
        let bytes = frame_bytes(k, &payload);
        for cut in 0..bytes.len() {
            let r = Frame::decode(&bytes[..cut]);
            assert!(r.is_err(), "kind {k} の {cut} バイト切り詰めが通った: {r:?}");
        }
    }
}

/// 末尾に 1 バイト足すと Err (v1 は厳密長。前方互換は version で切る)。
#[test]
fn trailing_bytes_are_rejected() {
    for (_, k, payload) in golden_frames() {
        let mut bytes = frame_bytes(k, &payload);
        bytes.push(0);
        assert!(Frame::decode(&bytes).is_err(), "kind {k} の末尾 1B が無視された");
    }
}

#[test]
fn unknown_version_and_kind_are_rejected() {
    let bytes = frame_bytes(kind::TOKEN, b"\x01\x02\x03");
    for bad_version in [0u8, 2, 255] {
        let mut b = bytes.clone();
        b[4] = bad_version;
        assert!(Frame::decode(&b).is_err(), "version {bad_version} が通った");
    }
    for bad_kind in [0u8, 18, 255] {
        let mut b = bytes.clone();
        b[5] = bad_kind;
        assert!(Frame::decode(&b).is_err(), "kind {bad_kind} が通った");
    }
    // magic 不一致
    let mut b = bytes.clone();
    b[0] = b'X';
    assert!(Frame::decode(&b).is_err());
    // 6 バイト未満
    assert!(Frame::decode(&bytes[..5]).is_err());
}

/// 実形 router_topk (層 47 / expert 502..511 / top-10) は 48 バイト。
/// 6 + 1 (layer) + 1 (count) + 10×2 (index は 2B) + 10×2 (f16) = 48。
#[test]
fn router_topk_realistic_payload_is_48_bytes() {
    let f = Frame::RouterTopK {
        layer: 47,
        experts: (0..10u16).map(|i| 502 + i).collect(),
        weights: (0..10u16).map(|i| 0x3c00 - i * 13).collect(),
    };
    let enc = f.encode();
    assert_eq!(enc.len(), 48);
    assert_eq!(&enc[..8], b"DNTL\x01\x10\x2f\x0a");
    assert_eq!(Frame::decode(&enc), Ok(f));
}
