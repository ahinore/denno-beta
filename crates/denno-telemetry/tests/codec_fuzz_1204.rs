//! task 1204: codec の極値まわりの総当たり / ファズ回帰テスト。
//!
//! 対象 (公開 API のみ。crate 内部の private 実装には触れない):
//!
//! 1. f16 変換 (`f32_to_f16_bits` / `f16_bits_to_f32`) の
//!    NaN / Inf / 非正規 (subnormal) / -0.0 の往復と境界値。
//!    非 NaN の f16 パターン 63,488 個を全て往復させて bit 一致を確認する。
//! 2. LEB128 varint の 10 バイト上限 (u64 の最大長) と桁あふれ、
//!    11 バイト = 長すぎ / 途中終端 = 切り詰めの扱い。
//! 3. 決定的な擬似乱数で作った 1,000 本の壊れたバッファを `Frame::decode` に
//!    通し、panic しないこと・通ったものは再 encode が安定であることを確認する。
//!
//! 「panic しない」は catch_unwind ではなく、素で呼んで落ちたらテストが落ちる
//! ことで検証する (既存 `codec_robustness.rs` と同じ方針)。

use denno_telemetry::{
    f16_bits_to_f32, f32_to_f16_bits, get_varint_u32, get_varint_u64, put_varint_u32,
    put_varint_u64, Frame, MAGIC, VERSION,
};

// ---------- f16 ----------

#[test]
fn f16_inf_and_signed_zero_roundtrip() {
    // ±Inf: f32 → f16 は 0x7c00 / 0xfc00、戻すと ±Inf。
    assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7c00);
    assert_eq!(f32_to_f16_bits(f32::NEG_INFINITY), 0xfc00);
    assert_eq!(f16_bits_to_f32(0x7c00), f32::INFINITY);
    assert_eq!(f16_bits_to_f32(0xfc00), f32::NEG_INFINITY);
    // f16 の ±Inf は f32 の ±Inf と同じ符号ビット。
    assert_eq!(f16_bits_to_f32(0x7c00).to_bits(), f32::INFINITY.to_bits());
    assert_eq!(f16_bits_to_f32(0xfc00).to_bits(), f32::NEG_INFINITY.to_bits());

    // -0.0: 符号ビットが落ちないこと (== 0.0 の比較では区別できないので bits で見る)。
    let nz = f32_to_f16_bits(-0.0f32);
    assert_eq!(nz, 0x8000, "-0.0 の f16 ビット");
    assert_eq!(f32_to_f16_bits(0.0f32), 0x0000);
    let back = f16_bits_to_f32(nz);
    assert_eq!(back.to_bits(), (-0.0f32).to_bits(), "-0.0 → f16 → f32 で符号が消えた");
    assert_eq!(f16_bits_to_f32(0x0000).to_bits(), 0.0f32.to_bits());
    // 0 に丸まる極小の負値も符号付き 0 になる。
    assert_eq!(f32_to_f16_bits(-f32::MIN_POSITIVE), 0x8000);
}

#[test]
fn f16_nan_payloads_are_preserved_as_nan() {
    // f32 の NaN は quiet bit を立てた 0x7e00 になり、符号は保たれる。
    assert_eq!(f32_to_f16_bits(f32::NAN), 0x7e00);
    assert_eq!(f32_to_f16_bits(-f32::NAN), 0xfe00);
    // f32 の signaling NaN (mant の MSB 以外) も quiet 化される。
    let snan = f32::from_bits(0x7f80_0001);
    assert_eq!(f32_to_f16_bits(snan), 0x7e00);

    // f16 の NaN パターン (指数 0x1f かつ mant != 0) は全て f32 の NaN になり、
    // 再 encode で quiet NaN (0x7e00 / 0xfe00) に正規化される。
    for h in 0u16..=0xffff {
        let exp = (h >> 10) & 0x1f;
        let mant = h & 0x03ff;
        if exp != 0x1f || mant == 0 {
            continue; // Inf と通常値は別テスト
        }
        let f = f16_bits_to_f32(h);
        assert!(f.is_nan(), "f16 NaN パターン {h:#06x} が f32 で NaN でない");
        let want = if h & 0x8000 != 0 { 0xfe00 } else { 0x7e00 };
        assert_eq!(f32_to_f16_bits(f), want, "f16 {h:#06x} の再 encode");
    }
}

#[test]
fn f16_subnormals_roundtrip_bit_exactly() {
    // 最小非正規 = 2^-24、最大非正規 = 1023 * 2^-24。
    assert_eq!(f16_bits_to_f32(0x0001), 2.0f32.powi(-24));
    assert_eq!(f16_bits_to_f32(0x03ff), 1023.0 * 2.0f32.powi(-24));
    assert_eq!(f16_bits_to_f32(0x8001), -2.0f32.powi(-24));

    // 非正規 1022 個 × 符号 2 = 2044 パターンの往復が bit 一致。
    for mant in 1u16..=0x03ff {
        for sign in [0u16, 0x8000] {
            let h = sign | mant;
            let f = f16_bits_to_f32(h);
            assert!(f.is_finite() && f != 0.0, "{h:#06x} が有限の非零でない");
            assert_eq!(f32_to_f16_bits(f), h, "非正規 {h:#06x} の往復でビットが変わる");
        }
    }
    // 非正規は全て 2^-14 未満 (最小正規より小さい)。
    assert!(f16_bits_to_f32(0x03ff) < f16_bits_to_f32(0x0400));
}

#[test]
fn f16_all_non_nan_patterns_roundtrip_exhaustively() {
    // 0x0000..=0xffff のうち NaN 以外 (63,488 個) は f16 → f32 → f16 で
    // ビットが完全に戻る (f16 は f32 の部分集合なので丸め誤差が出ない)。
    let mut checked = 0usize;
    for h in 0u16..=0xffff {
        let exp = (h >> 10) & 0x1f;
        let mant = h & 0x03ff;
        if exp == 0x1f && mant != 0 {
            continue; // NaN
        }
        let back = f32_to_f16_bits(f16_bits_to_f32(h));
        assert_eq!(back, h, "f16 {h:#06x} の往復でビットが変わる ({back:#06x})");
        checked += 1;
    }
    assert_eq!(checked, 0x1_0000 - 2 * 1023, "非 NaN のパターン数");
}

#[test]
fn f16_overflow_and_underflow_boundaries() {
    // f16 最大 = 65504。65520 以上は RNE で Inf。
    assert_eq!(f16_bits_to_f32(0x7bff), 65504.0);
    assert_eq!(f32_to_f16_bits(65504.0), 0x7bff);
    assert_eq!(f32_to_f16_bits(65520.0), 0x7c00, "65520 は Inf に丸まる");
    assert_eq!(f32_to_f16_bits(1.0e30), 0x7c00);
    assert_eq!(f32_to_f16_bits(-1.0e30), 0xfc00);

    // 2^-24 の半分 (2^-25) は RNE で 0、それより上は最小非正規 1 に丸まる。
    assert_eq!(f32_to_f16_bits(2.0f32.powi(-24)), 0x0001);
    assert_eq!(f32_to_f16_bits(2.0f32.powi(-25)), 0x0000, "2^-25 ちょうどは 0");
    assert_eq!(f32_to_f16_bits(2.0f32.powi(-25) * 1.5), 0x0001);
    assert!(f32_to_f16_bits(1.0e-9) == 0x0000, "桁落ちは 0 になる");

    // 正規の下限 2^-14 (= 1024 * 2^-24) と、直下の非正規グリッド。
    // f16 の非正規は 2^-24 刻みなので、1023.5 * 2^-24 は RNE の同値点
    // (1023 と 1024 の中間) で、偶数側 1024 = 最小正規に丸まる。
    assert_eq!(f32_to_f16_bits(2.0f32.powi(-14)), 0x0400);
    assert_eq!(f32_to_f16_bits(1023.0 * 2.0f32.powi(-24)), 0x03ff);
    assert_eq!(f32_to_f16_bits(1023.25 * 2.0f32.powi(-24)), 0x03ff);
    assert_eq!(f32_to_f16_bits(1023.5 * 2.0f32.powi(-24)), 0x0400, "RNE の同値点");
}

// ---------- varint ----------

#[test]
fn varint_u64_ten_byte_extremes_roundtrip() {
    // u64::MAX は 10 バイト (LEB128 の最大長)。
    let mut b = Vec::new();
    put_varint_u64(&mut b, u64::MAX);
    assert_eq!(b.len(), 10, "u64::MAX は 10 バイト");
    assert_eq!(get_varint_u64(&b), Ok((u64::MAX, 10)));

    // 2^63 は 10 バイト目だけが 0x01 になる最大の「きれいな」10 バイト値。
    let mut b = Vec::new();
    put_varint_u64(&mut b, 1u64 << 63);
    assert_eq!(b, vec![0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x01]);
    assert_eq!(get_varint_u64(&b), Ok((1u64 << 63, 10)));

    // 境界値の往復 (9 / 10 バイトの境目)。
    for v in [
        0u64,
        1,
        127,
        128,
        (1u64 << 7) - 1,
        1u64 << 7,
        (1u64 << 56) - 1,
        1u64 << 56,
        (1u64 << 63) - 1,
        1u64 << 63,
        u64::MAX - 1,
        u64::MAX,
    ] {
        let mut b = Vec::new();
        put_varint_u64(&mut b, v);
        assert_eq!(get_varint_u64(&b), Ok((v, b.len())), "値 {v}");
        assert!(b.len() <= 10);
    }
}

#[test]
fn varint_ten_byte_overflow_is_truncated_not_panicked() {
    // 9 バイトの継続 + 10 バイト目に bit63 より上のビット (0x02)。
    // LEB128 としては u64 に収まらない (= 桁あふれ) が、実装はシフトで
    // 溢れたビットを黙って捨て、末尾バイトで終端するので Ok を返す。
    // ここで固定したいのは「panic しない」「上位ビットが値に混ざらない」の 2 点。
    let mut buf = vec![0x80u8; 9];
    buf.push(0x02);
    let (v, n) = get_varint_u64(&buf).expect("10 バイト目が終端なら Ok");
    assert_eq!(n, 10);
    assert_eq!(v, 0, "bit63 より上のビットが値に混ざった: {v:#x}");

    // 10 バイト目が 0x7f でも値は下位 64 bit に切り詰められる (panic しない)。
    // 0b111_1111 << 63 は bit63 だけが残り、他は捨てられる。
    let mut buf = vec![0x80u8; 9];
    buf.push(0x7f);
    let (v, n) = get_varint_u64(&buf).expect("終端しているので Ok");
    assert_eq!(n, 10);
    assert_eq!(v, 1u64 << 63, "bit63 より上のビットが値に混ざった: {v:#x}");

    // u32 側は 10 バイト値でも u32 を超えたら Err。
    let mut b = Vec::new();
    put_varint_u64(&mut b, u32::MAX as u64 + 1);
    assert!(get_varint_u32(&b).is_err(), "u32 を超える値が通った");
    let err = get_varint_u32(&b).unwrap_err();
    assert!(err.contains("overflows u32"), "{err}");
    // u32::MAX ちょうどは通る。
    let mut b = Vec::new();
    put_varint_u32(&mut b, u32::MAX);
    assert_eq!(b.len(), 5);
    assert_eq!(get_varint_u32(&b), Ok((u32::MAX, 5)));
}

#[test]
fn varint_eleven_bytes_and_truncation_are_errors() {
    // 11 バイト (全て継続ビット) は長すぎ。
    let long = vec![0x80u8; 11];
    assert_eq!(get_varint_u64(&long), Err("varint too long".into()));
    // 10 バイト全て継続ビットは、終端が無いので切り詰め。
    let no_end = vec![0x80u8; 10];
    assert_eq!(get_varint_u64(&no_end), Err("varint truncated".into()));
    // 空 / 途中終端なし。
    assert_eq!(get_varint_u64(&[]), Err("varint truncated".into()));
    assert_eq!(get_varint_u64(&[0x81]), Err("varint truncated".into()));
    // get_varint_u32 も同じ境界 (8 バイト継続でも Err)。
    assert!(get_varint_u32(&vec![0x80u8; 10]).is_err());
    // 消費バイト数は終端バイトまで (余りは読まない)。
    assert_eq!(get_varint_u64(&[0x05, 0xff, 0xff]), Ok((5, 1)));
}

// ---------- 壊れたバッファ (1,000 本) ----------

/// 決定的な擬似乱数 (外部 crate を使わない)。既存テストと同じ LCG。
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }
    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u32() as usize) % n
        }
    }
}

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
            ns: u64::MAX,
            tokens: vec![972, 198, 3585, 370, 151_644],
        },
        Frame::Token { session: 7, token: 99_999, pos: 12 },
        Frame::Step { width: 4, ns: 8_765_432 },
        Frame::OpSummary { ops: vec![(4, 1_000), (11, u64::MAX)] },
        Frame::KvUsage { busy: vec![true, false, true, true], t_lens: vec![10, 0, 300, 777] },
        Frame::Gap { lost: 42 },
        Frame::TopK {
            session: 3,
            chosen: 831,
            entropy_f16: f32_to_f16_bits(1.25),
            entries: vec![(831, f32_to_f16_bits(0.5)), (12, 0x0001)],
        },
        Frame::LayerRun { layer: 5, ops: vec![(1, 2_100), (3, 154_000), (2, 8_800)] },
        Frame::LayerAct { layer: 47, idx: vec![0, 1531, 2560], val: vec![0x3c00, 0x3800, 0x0001] },
        Frame::RoleAct { layer: 47, attn: 0, gdn: 0x3c00, router: 0x3800, shex: 0, ple: 1 },
        Frame::ExpertResidency { layer: 5, in_vram: (0..512u16).collect() },
        Frame::TierStats { vram_used_mb: 4_096, vram_total_mb: 8_192, ram_experts: 64, ssd_reads: 1_234_567 },
        Frame::RouterTopK {
            layer: 3,
            experts: (0..10u16).map(|i| 502 + i).collect(),
            weights: (0..10u16).map(|i| 0x3c00 - i * 13).collect(),
        },
        Frame::ExpertSkip { layer: 47, experts: (0..10u16).map(|i| 300 + i).collect() },
    ]
}

#[test]
fn thousand_corrupt_buffers_never_panic_and_reencode_stably() {
    let mut rng = Lcg::new(0x1204_c0de_f00d_5eed);
    let frames = sample_frames();
    let mut ok = 0usize;
    let mut err = 0usize;

    for i in 0..1_000usize {
        // 3 種類の壊し方を回す: バイト変異 / 切り詰め / でたらめなペイロード。
        let buf: Vec<u8> = match i % 3 {
            0 => {
                let base = frames[i % frames.len()].encode();
                let mut b = base;
                let flips = 1 + rng.below(4);
                for _ in 0..flips {
                    if b.is_empty() {
                        break;
                    }
                    let pos = rng.below(b.len());
                    b[pos] ^= (rng.next_u32() & 0xff) as u8;
                }
                // たまにヘッダ (magic / version / kind) も壊す。
                if i % 7 == 0 && b.len() >= 6 {
                    let pos = rng.below(6);
                    b[pos] = (rng.next_u32() & 0xff) as u8;
                }
                b
            }
            1 => {
                let base = frames[i % frames.len()].encode();
                let cut = rng.below(base.len() + 1);
                base[..cut].to_vec()
            }
            _ => {
                // magic / version は正しく、kind と中身はでたらめ。
                let len = rng.below(64);
                let mut b = Vec::with_capacity(len + 6);
                b.extend_from_slice(&MAGIC);
                b.push(VERSION);
                b.push((rng.next_u32() % 18) as u8);
                for _ in 0..len {
                    b.push((rng.next_u32() & 0xff) as u8);
                }
                b
            }
        };

        // panic すればここでテストが落ちる (これ自体が「panic しない」の検証)。
        match Frame::decode(&buf) {
            Err(_) => err += 1,
            Ok(got) => {
                ok += 1;
                let re = got.encode();
                let again = Frame::decode(&re).unwrap_or_else(|e| {
                    panic!("#{i}: 再 encode した kind {} が decode できない: {e}", got.kind())
                });
                assert_eq!(again, got, "#{i}: 再 encode で値が変わる kind {}", got.kind());
                assert_eq!(again.encode(), re, "#{i}: 再 encode が非決定的");
            }
        }
    }

    // コーパスが両方の結果を含むこと (片方だけなら壊し方が偏っている)。
    assert!(err > 500, "Err が少なすぎる: {err} / 1000");
    assert!(ok > 0, "Ok が 1 本も無い: 変異が強すぎる");
    assert_eq!(ok + err, 1_000);
}
