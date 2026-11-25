//! task 914: フレーム codec の堅牢性 (切り詰め / 過大 count / 非有限値) と
//! [`RollingLayerStats`] の境界テスト。
//!
//! ここでは公開 API だけを使う (crate 内部の private 実装に依存しない)。
//! 「panic しない」ことは、decode を素で呼んで panic すればテストが落ちることで
//! 検証する (catch_unwind は使わない)。

use denno_telemetry::{
    f32_to_f16_bits, f16_bits_to_f32, kind, layer_act_frame, normalize_roles, put_varint_u32,
    topk_softmax, Frame, LayerWindowStats, RollingLayerStats, MAGIC, VERSION,
};

/// 全 kind を 1 枚ずつ (往復・切り詰め・変異テストの入力)。
fn sample_frames() -> Vec<Frame> {
    vec![
        Frame::ServerInfo {
            model: "denno-dev".into(),
            model_id: "Qwen/Qwen2.5-0.5B-Instruct".into(),
            quant: "fp32".into(),
            slots: 4,
            context: 32768,
            layers: 24,
        },
        Frame::SessionAdmit { session: 7, slot: 2, prompt_tokens: 34 },
        Frame::SessionDone { session: 7, reason: 2, output_tokens: 26 },
        Frame::Prefill {
            session: 7,
            prompt_tokens: 34,
            ns: 123_456_789,
            tokens: vec![972, 198, 3585, 370, 151_644],
        },
        Frame::Token { session: 7, token: 99_999, pos: 12 },
        Frame::Step { width: 4, ns: 8_765_432 },
        Frame::OpSummary { ops: vec![(4, 1_000), (11, 2_500)] },
        Frame::KvUsage { busy: vec![true, false, true, true], t_lens: vec![10, 0, 300, 777] },
        Frame::Gap { lost: 42 },
        Frame::TopK {
            session: 3,
            chosen: 831,
            entropy_f16: f32_to_f16_bits(1.25),
            entries: vec![(831, f32_to_f16_bits(0.5)), (12, f32_to_f16_bits(0.25))],
        },
        Frame::LayerRun { layer: 5, ops: vec![(1, 2_100), (3, 154_000), (2, 8_800)] },
        Frame::LayerAct {
            layer: 47,
            idx: vec![0, 1531, 2560],
            val: vec![0x3c00, 0x3800, 0x0001],
        },
        Frame::LayerAct { layer: 12, idx: Vec::new(), val: Vec::new() },
        Frame::RoleAct { layer: 5, attn: 0, gdn: 0, router: 0, shex: 0, ple: 0 },
        Frame::RoleAct { layer: 47, attn: 0, gdn: 0x3c00, router: 0x3800, shex: 0, ple: 1 },
        Frame::ExpertResidency { layer: 3, in_vram: Vec::new() },
        Frame::ExpertResidency { layer: 5, in_vram: (0..512u16).collect() },
        Frame::TierStats { vram_used_mb: 4_096, vram_total_mb: 8_192, ram_experts: 64, ssd_reads: 1_234_567 },
        Frame::RouterTopK { layer: 47, experts: Vec::new(), weights: Vec::new() },
        Frame::RouterTopK {
            layer: 3,
            experts: (0..10u16).map(|i| 502 + i).collect(),
            weights: (0..10u16).map(|i| 0x3c00 - i * 13).collect(),
        },
        Frame::ExpertSkip { layer: 5, experts: vec![0] },
        Frame::ExpertSkip { layer: 47, experts: (0..10u16).map(|i| 300 + i).collect() },
    ]
}

/// 決定的な擬似乱数 (外部 crate を使わない)。
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u32(&mut self) -> u32 {
        // Numerical Recipes LCG (64bit)
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }
}

// ---------- 往復 / 決定性 ----------

#[test]
fn roundtrip_all_kinds_is_deterministic() {
    for f in sample_frames() {
        let a = f.encode();
        let b = f.encode();
        assert_eq!(a, b, "encode が非決定的 (kind {})", f.kind());
        assert_eq!(&a[..4], &MAGIC);
        assert_eq!(a[4], VERSION);
        assert_eq!(a[5], f.kind());
        let dec = Frame::decode(&a).unwrap_or_else(|e| panic!("decode failed kind {}: {e}", f.kind()));
        assert_eq!(dec, f, "roundtrip failed kind {}", f.kind());
        // 再 encode は同じバイト列 (正規形)
        assert_eq!(dec.encode(), a, "re-encode differs kind {}", f.kind());
    }
}

#[test]
fn roundtrip_is_stable_through_two_generations() {
    for f in sample_frames() {
        let once = Frame::decode(&f.encode()).unwrap();
        let twice = Frame::decode(&once.encode()).unwrap();
        assert_eq!(once, twice, "kind {}", f.kind());
    }
}

// ---------- 切り詰め ----------

#[test]
fn every_truncated_prefix_is_rejected() {
    for f in sample_frames() {
        let enc = f.encode();
        for cut in 0..enc.len() {
            let part = &enc[..cut];
            match Frame::decode(part) {
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

#[test]
fn truncated_two_byte_fields_do_not_panic() {
    // 2 バイト読みの直前で切る (旧実装は p[i+1] で panic した経路)。
    for f in sample_frames() {
        let enc = f.encode();
        // 末尾 1 バイトを削る = f16 / 重みの 2 バイト目が欠ける形を必ず含む
        let part = &enc[..enc.len() - 1];
        assert!(Frame::decode(part).is_err(), "kind {}", f.kind());
    }
}

#[test]
fn trailing_bytes_are_rejected() {
    for f in sample_frames() {
        let mut enc = f.encode();
        enc.push(0);
        let err = Frame::decode(&enc).expect_err("余剰バイトが通った");
        assert!(err.contains("trailing"), "kind {}: {err}", f.kind());
    }
}

// ---------- ヘッダ検証 ----------

#[test]
fn header_validation() {
    assert!(Frame::decode(&[]).is_err());
    assert!(Frame::decode(b"DNT").is_err(), "6 バイト未満は Err");
    assert!(Frame::decode(b"XXXX\x01\x01").is_err(), "magic 違い");
    assert!(Frame::decode(&[b'D', b'N', b'T', b'L', 9, 1]).is_err(), "未知 version");
    assert!(Frame::decode(&[b'D', b'N', b'T', b'L', 1, 99]).is_err(), "未知 kind");
    let err = Frame::decode(&[b'D', b'N', b'T', b'L', 1, 99]).unwrap_err();
    assert!(err.contains("unknown kind"), "{err}");
}

// ---------- 過大 count ----------

/// `MAGIC | VERSION | kind | 前置き | count=u32::MAX` を作る。
fn oversized(k: u8, prefix: &[u8]) -> Vec<u8> {
    let mut v = Vec::new();
    v.extend_from_slice(&MAGIC);
    v.push(VERSION);
    v.push(k);
    v.extend_from_slice(prefix);
    put_varint_u32(&mut v, u32::MAX);
    v
}

#[test]
fn oversized_counts_are_rejected_without_allocation() {
    let cases: Vec<(u8, Vec<u8>)> = vec![
        // Prefill: session, prompt_tokens, ns(u64=0) の後に count
        (kind::PREFILL, vec![1, 1, 0]),
        (kind::OP_SUMMARY, vec![]),
        (kind::KV_USAGE, vec![]),
        // TopK: session, chosen, entropy(2B)
        (kind::TOP_K, vec![1, 1, 0, 0]),
        (kind::LAYER_RUN, vec![0]),
        (kind::LAYER_ACT, vec![0]),
        (kind::EXPERT_RESIDENCY, vec![0]),
        (kind::ROUTER_TOPK, vec![0]),
        (kind::EXPERT_SKIP, vec![0]),
    ];
    for (k, prefix) in cases {
        let buf = oversized(k, &prefix);
        let err = Frame::decode(&buf).unwrap_err();
        assert!(
            err.contains("count") || err.contains("truncated"),
            "kind {k}: 期待したエラーでない: {err}"
        );
    }
}

#[test]
fn oversized_string_lengths_are_rejected() {
    // ServerInfo: model の長さだけ u32::MAX で中身が無い
    let mut v = Vec::new();
    v.extend_from_slice(&MAGIC);
    v.push(VERSION);
    v.push(kind::SERVER_INFO);
    put_varint_u32(&mut v, u32::MAX);
    assert!(Frame::decode(&v).is_err());
    // 長さが u32::MAX で実データが 1 バイトだけ
    v.push(0x41);
    assert!(Frame::decode(&v).is_err());
}

#[test]
fn out_of_range_expert_indices_are_rejected() {
    // 各 kind: 1 要素で index が u16 を超える
    for k in [kind::ROUTER_TOPK, kind::EXPERT_SKIP] {
        let mut v = Vec::new();
        v.extend_from_slice(&MAGIC);
        v.push(VERSION);
        v.push(k);
        put_varint_u32(&mut v, 0); // layer
        put_varint_u32(&mut v, 1); // n
        put_varint_u32(&mut v, 70_000); // expert index
        if k == kind::ROUTER_TOPK {
            v.extend_from_slice(&0x3c00u16.to_le_bytes()); // weight
        }
        assert!(Frame::decode(&v).is_err(), "kind {k}");
    }
    // ExpertResidency も同じ (旧実装は黙って u16 に切り詰めていた)
    let mut v = Vec::new();
    v.extend_from_slice(&MAGIC);
    v.push(VERSION);
    v.push(kind::EXPERT_RESIDENCY);
    put_varint_u32(&mut v, 0);
    put_varint_u32(&mut v, 1);
    put_varint_u32(&mut v, 70_000);
    assert!(Frame::decode(&v).is_err());
}

// ---------- 変異 (panic しない + 再 encode 安定) ----------

#[test]
fn byte_mutations_never_panic_and_decode_reencode_is_stable() {
    let mut rng = Lcg::new(0x00c0_ffee_1234_5678);
    for f in sample_frames() {
        let base = f.encode();
        for _ in 0..300 {
            let mut buf = base.clone();
            let flips = 1 + (rng.next_u32() % 3) as usize;
            for _ in 0..flips {
                let pos = (rng.next_u32() as usize) % buf.len();
                buf[pos] = (rng.next_u32() & 0xff) as u8;
            }
            // panic すればテストが落ちる (これ自体が「panic しない」の検証)
            if let Ok(got) = Frame::decode(&buf) {
                let re = got.encode();
                let again = Frame::decode(&re)
                    .unwrap_or_else(|e| panic!("再 encode が decode できない kind {}: {e}", got.kind()));
                assert_eq!(again, got, "再 encode で値が変わる kind {}", got.kind());
            }
        }
    }
}

#[test]
fn random_payloads_never_panic() {
    let mut rng = Lcg::new(0xdead_beef_0bad_f00d);
    for _ in 0..2000 {
        let len = (rng.next_u32() % 64) as usize;
        let mut buf = Vec::with_capacity(len + 6);
        buf.extend_from_slice(&MAGIC);
        buf.push(VERSION);
        buf.push((rng.next_u32() % 20) as u8);
        for _ in 0..len {
            buf.push((rng.next_u32() & 0xff) as u8);
        }
        let _ = Frame::decode(&buf); // panic したらテスト失敗
    }
}

// ---------- 長さの食い違い (encode 側の正規化) ----------

#[test]
fn mismatched_vector_lengths_roundtrip_with_min_length() {
    // KvUsage: busy 4 / t_lens 2 → 2 本に切り揃えて往復
    let f = Frame::KvUsage { busy: vec![true, false, true, true], t_lens: vec![10, 20] };
    assert_eq!(
        Frame::decode(&f.encode()).unwrap(),
        Frame::KvUsage { busy: vec![true, false], t_lens: vec![10, 20] }
    );
    // LayerAct: idx 2 / val 3 → 2 本
    let f = Frame::LayerAct { layer: 1, idx: vec![5, 6], val: vec![1, 2, 3] };
    assert_eq!(
        Frame::decode(&f.encode()).unwrap(),
        Frame::LayerAct { layer: 1, idx: vec![5, 6], val: vec![1, 2] }
    );
    // RouterTopK: experts 3 / weights 1 → 1 本
    let f = Frame::RouterTopK { layer: 2, experts: vec![1, 2, 3], weights: vec![0x3c00] };
    assert_eq!(
        Frame::decode(&f.encode()).unwrap(),
        Frame::RouterTopK { layer: 2, experts: vec![1], weights: vec![0x3c00] }
    );
}

// ---------- 非有限値 ----------

#[test]
fn normalize_roles_rejects_non_finite() {
    // NaN / Inf は 0 に落ち、有限値だけで正規化する
    assert_eq!(normalize_roles([f32::NAN, f32::INFINITY, 1.0, 0.0, 0.0]), [0.0, 0.0, 1.0, 0.0, 0.0]);
    assert_eq!(normalize_roles([f32::NAN; 5]), [0.0; 5]);
    assert_eq!(normalize_roles([f32::INFINITY; 5]), [0.0; 5]);
    // 出力は常に有限で 0..1
    for r in [
        [f32::NAN, 1.0, 2.0, -3.0, f32::NEG_INFINITY],
        [1e30, 1.0, 0.0, 0.0, 0.0],
        [-1.0, -2.0, 0.0, 0.0, 0.0],
    ] {
        let n = normalize_roles(r);
        for v in n {
            assert!(v.is_finite(), "{n:?}");
            assert!((0.0..=1.0).contains(&v), "{n:?}");
        }
    }
}

#[test]
fn layer_act_frame_skips_non_finite_values() {
    let mut scratch = Vec::new();
    let x = [f32::NAN, 1.0, f32::INFINITY, -2.0, 0.0, f32::NEG_INFINITY];
    match layer_act_frame(9, &x, 10, &mut scratch) {
        Frame::LayerAct { layer: 9, idx, val } => {
            // 有限値だけ: index 1 (|1.0|) と 3 (|-2.0|)。昇順。
            assert_eq!(idx, vec![1, 3]);
            // 最大 2.0 で正規化: 1.0 → 0.5 (0x3800), 2.0 → 1.0 (0x3c00)
            assert_eq!(val, vec![0x3800, 0x3c00]);
            for &v in &val {
                assert!(!f16_bits_to_f32(v).is_nan());
            }
        }
        other => panic!("LayerAct でない: {other:?}"),
    }
    // 全部非有限 → 空フレーム (0/0 を作らない)
    let mut scratch = Vec::new();
    let bad = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
    assert_eq!(
        layer_act_frame(1, &bad, 4, &mut scratch),
        Frame::LayerAct { layer: 1, idx: Vec::new(), val: Vec::new() }
    );
}

#[test]
fn topk_softmax_sanitizes_non_finite_logits() {
    for logits in [
        vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0],
        vec![f32::NAN; 8],
        vec![f32::INFINITY; 4],
        vec![f32::NEG_INFINITY; 4],
    ] {
        let tk = topk_softmax(&logits, 3);
        assert!(tk.entropy.is_finite(), "entropy={} logits={logits:?}", tk.entropy);
        for (id, p) in &tk.entries {
            assert!(p.is_finite(), "id={id} p={p} logits={logits:?}");
            assert!(*p >= 0.0, "id={id} p={p}");
        }
    }
    // f16 の非有限は往復でビットが保たれる (NaN は NaN のまま)
    let n = f32_to_f16_bits(f32::NAN);
    assert!(f16_bits_to_f32(n).is_nan());
    assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7c00);
    assert_eq!(f32_to_f16_bits(f32::NEG_INFINITY), 0xfc00);
}

// ---------- RollingLayerStats ----------

#[test]
fn rolling_stats_tracks_count_min_max_mean() {
    let mut s = RollingLayerStats::new(4, 8);
    assert_eq!(s.n_layers(), 4);
    assert_eq!(s.window(), 8);
    // 空の窓は 0 (NaN ではない)
    let e = s.stats(0).unwrap();
    assert_eq!(e, LayerWindowStats { count: 0, min: 0.0, max: 0.0, mean: 0.0 });
    for v in [1.0f32, 2.0, 3.0, 4.0] {
        assert!(s.push(0, v));
    }
    let st = s.stats(0).unwrap();
    assert_eq!(st.count, 4);
    assert_eq!(st.min, 1.0);
    assert_eq!(st.max, 4.0);
    assert_eq!(st.mean, 2.5);
    // 層ごとに独立
    assert!(s.push(1, -7.0));
    assert_eq!(s.stats(1).unwrap().min, -7.0);
    assert_eq!(s.stats(1).unwrap().count, 1);
    assert_eq!(s.stats(0).unwrap().count, 4);
}

#[test]
fn rolling_stats_window_is_bounded() {
    let mut s = RollingLayerStats::new(3, 4);
    for i in 0..50 {
        s.push(1, i as f32);
    }
    let st = s.stats(1).unwrap();
    assert_eq!(st.count, 4, "窓長を超えて保持している");
    assert_eq!(st.min, 46.0);
    assert_eq!(st.max, 49.0);
    assert_eq!(st.mean, 47.5);
    // 保持サンプルは n_layers * window 以下 (unbounded でない)
    assert!(s.total_samples() <= 3 * 4);
}

#[test]
fn rolling_stats_rejects_non_finite_values() {
    let mut s = RollingLayerStats::new(2, 4);
    assert!(!s.push(0, f32::NAN));
    assert!(!s.push(0, f32::INFINITY));
    assert!(!s.push(0, f32::NEG_INFINITY));
    assert_eq!(s.rejected(0), Some(3));
    assert_eq!(s.rejected_total(), 3);
    // 拒否された値は統計に入らない
    assert_eq!(s.stats(0).unwrap().count, 0);
    assert!(s.push(0, 5.0));
    let st = s.stats(0).unwrap();
    assert_eq!(st.count, 1);
    assert_eq!(st.min, 5.0);
    assert_eq!(st.max, 5.0);
    assert_eq!(st.mean, 5.0);
    // 範囲外の層は記録しない
    assert!(!s.push(2, 1.0));
    assert!(s.stats(2).is_none());
    assert!(s.rejected(2).is_none());
}

#[test]
fn rolling_stats_push_slice_and_reset() {
    let mut s = RollingLayerStats::new(2, 8);
    let accepted = s.push_slice(0, &[1.0, f32::NAN, 3.0, f32::INFINITY, 5.0]);
    assert_eq!(accepted, 3, "非有限は数えない");
    assert_eq!(s.stats(0).unwrap().count, 3);
    assert_eq!(s.rejected(0), Some(2));
    assert!(s.reset_layer(0));
    assert_eq!(s.stats(0).unwrap().count, 0);
    assert_eq!(s.rejected(0), Some(0));
    assert!(!s.reset_layer(9));
    // 全体 reset
    s.push(1, 2.0);
    s.push(1, f32::NAN);
    s.reset();
    assert_eq!(s.total_samples(), 0);
    assert_eq!(s.rejected_total(), 0);
}

#[test]
fn rolling_stats_zero_window_and_zero_layers() {
    // window 0 は 1 に切り上げ (除算 0 を作らない)
    let mut s = RollingLayerStats::new(2, 0);
    assert_eq!(s.window(), 1);
    s.push(0, 1.0);
    s.push(0, 2.0);
    let st = s.stats(0).unwrap();
    assert_eq!(st.count, 1);
    assert_eq!(st.min, 2.0);
    assert_eq!(st.max, 2.0);
    // 層 0 個
    let mut empty = RollingLayerStats::new(0, 4);
    assert!(!empty.push(0, 1.0));
    assert!(empty.stats(0).is_none());
    assert_eq!(empty.total_samples(), 0);
}
