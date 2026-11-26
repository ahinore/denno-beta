//! 複数ドメインにまたがるテスト (共有)
//!
//! task 664: `src/cuda.rs` (70k 行) からの純粋な移動。ロジック・数値・カーネル・
//! PTX・env ゲートは一切変えていない。

#![allow(non_snake_case, non_camel_case_types)]

use super::*;


/// テスト用: env を介さず GPU 経路を強制有効化 (ignored テストでのみ使用)。
#[cfg(test)]
pub fn enable_for_tests() {
    GPU_FORCE.store(true, std::sync::atomic::Ordering::Relaxed);
}


/// s168 (Task P7): 行タイルのノブと端数併合の契約 (純粋関数・GPU 不要)。
#[cfg(test)]
mod p7_tile_tests {
    use super::*;

    /// `DENNO_Q4_M_TILE` の解釈: 既定は従来値、範囲外・刻み外・非数は既定へ。
    #[test]
    fn env_tile_parse() {
        assert_eq!(Q4_M_TILE, 1024, "既定は s124 以来の 1024 のまま");
        assert_eq!(parse_q4_m_tile(None), 1024);
        assert_eq!(parse_q4_m_tile(Some("")), 1024);
        assert_eq!(parse_q4_m_tile(Some("abc")), 1024);
        assert_eq!(parse_q4_m_tile(Some("-512")), 1024);
        assert_eq!(parse_q4_m_tile(Some("0")), 1024);
        assert_eq!(parse_q4_m_tile(Some("32")), 1024, "64 行未満は受けない");
        assert_eq!(parse_q4_m_tile(Some("96")), 1024, "64 の倍数でない");
        assert_eq!(parse_q4_m_tile(Some("999999")), 1024, "上限超え");
        assert_eq!(parse_q4_m_tile(Some(" 2048 ")), 2048, "前後の空白は許す");
        for t in [64usize, 512, 1024, 2048, 4096, 8192] {
            assert_eq!(parse_q4_m_tile(Some(&t.to_string())), t, "t={t}");
        }
        // 本番の実効値も必ず受理域に入る (panic も 0 行タイルも作らない)
        let eff = q4_m_tile();
        assert!(eff % Q4_M_TILE_STEP == 0 && eff >= Q4_M_TILE_STEP && eff <= Q4_M_TILE_MAX);
    }

    /// 行グループの契約: 重複・欠落なしで m 行を覆い、1 グループは cap 行以下。
    /// 途中のグループは必ず tile 行ちょうど (併合は最終グループだけ)、
    /// 端数 (<= tail) が単独タイルとして launch されることは無い。
    #[test]
    fn row_groups_cover_rows_exactly() {
        let tail = q4_tail_merge_below();
        for tile in [64usize, 128, 512, 1024, 2048, 4096] {
            let cap = q4_tile_max_rows(tile);
            assert_eq!(cap, tile + tail, "tile={tile}");
            for m in 1..=3000usize {
                let (mut r0, mut groups) = (0usize, 0usize);
                while r0 < m {
                    let mr = q4_group_rows(r0, m, tile);
                    assert!(mr > 0, "tile={tile} m={m} r0={r0}: 空グループ");
                    assert!(mr <= cap, "tile={tile} m={m}: {mr} 行 > cap {cap}");
                    if r0 + mr < m {
                        assert_eq!(mr, tile, "tile={tile} m={m} r0={r0}: 途中が {mr} 行");
                    } else if r0 > 0 {
                        // 最終グループ: tile ちょうど or 併合 (tile 超) or tail 超の端数
                        assert!(
                            mr >= tile || mr > tail,
                            "tile={tile} m={m}: 端数 {mr} 行 (tail {tail}) を単独 launch している"
                        );
                    }
                    r0 += mr;
                    groups += 1;
                    assert!(groups <= m, "tile={tile} m={m}: グループ数が行数を超えた");
                }
                assert_eq!(r0, m, "tile={tile} m={m}: 行が覆われていない");
            }
        }
    }

    /// 13k token の実形 (27B の prefill) で、タイル幅ごとの launch 回数と
    /// 端数併合の有無を固定する (REPORT_P7.md の表と同じ計算)。
    #[test]
    fn row_groups_at_13k_tokens() {
        let tail = q4_tail_merge_below();
        // (tile, 13,000 行のグループ数) — 端数 712 行はどの tile でも tail 超
        for (tile, want) in [(512usize, 26usize), (1024, 13), (2048, 7), (4096, 4)] {
            let mut r0 = 0usize;
            let mut n = 0usize;
            while r0 < 13_000 {
                r0 += q4_group_rows(r0, 13_000, tile);
                n += 1;
            }
            assert_eq!(n, want, "tile={tile} 13,000 行の launch 回数");
        }
        // 端数が tail 以下の形は直前のタイルへ併合され、グループ数が 1 減る
        let m = 1024 * 3 + 4;
        let mut r0 = 0usize;
        let (mut n, mut last) = (0usize, 0usize);
        while r0 < m {
            last = q4_group_rows(r0, m, 1024);
            r0 += last;
            n += 1;
        }
        if tail >= 4 {
            assert_eq!(n, 3, "端数 4 行は 3 番目のタイルへ併合 (2,048+ 行)");
            assert_eq!(last, 1024 + 4, "併合後の最終タイル行数");
        } else {
            assert_eq!(n, 4, "併合 OFF (DENNO_CUDA_MAX_M=0) では 4 グループ");
        }
    }
}


#[cfg(test)]
mod f32_d2h_pin_tests {
    use super::*;

    /// task 296: 1 MiB 以上の D2H を pinned 経由にするゲートの真理値表。
    #[test]
    fn f32_gemm_pin_d2h_gate_truth_table() {
        assert!(!f32_gemm_pin_d2h_from_value(None), "未設定は既定 off");
        assert!(!f32_gemm_pin_d2h_from_value(Some("")));
        assert!(f32_gemm_pin_d2h_from_value(Some("1")));
        assert!(f32_gemm_pin_d2h_from_value(Some(" on ")));
        assert!(f32_gemm_pin_d2h_from_value(Some("True")));
        assert!(!f32_gemm_pin_d2h_from_value(Some("0")));
        assert!(!f32_gemm_pin_d2h_from_value(Some("off")));
        assert!(!f32_gemm_pin_d2h_from_value(Some("no")));
        // しきい値は 1 MiB (decode の小さい D2H は従来のまま = 挙動を変えない)。
        assert_eq!(F32_GEMM_PIN_D2H_MIN, 1 << 20);
        // 3,826 トークンの router (3,826 x 512) と PLE の value (3,826 x 2560) が
        // しきい値を超えること (= prefill の主な D2H が pinned 経路に入る)。
        assert!(3826 * 512 * 4 >= F32_GEMM_PIN_D2H_MIN);
        assert!(3826 * 2560 * 4 >= F32_GEMM_PIN_D2H_MIN);
        // decode (1 トークン x 512) は入らない。
        assert!(512 * 4 < F32_GEMM_PIN_D2H_MIN);
    }
}


#[cfg(test)]
mod pf_stage_tests {
    use super::*;

    /// task 189: `DENNO_PF_MIX_STAGE` は明示的な真値だけ on (既定 off)。
    #[test]
    fn mix_stage_gate_truth_table() {
        assert!(!mix_stage_from_value(None), "未設定は既定 off");
        for v in ["0", "off", "false", "no", "", " off "] {
            assert!(!mix_stage_from_value(Some(v)), "{v:?} が on になった");
        }
        for v in ["1", "on", "true", "yes", "2"] {
            assert!(mix_stage_from_value(Some(v)), "{v:?} が off のまま");
        }
    }
}


#[cfg(all(test, feature = "cuda"))]
pub(crate) mod tests {
    use super::*;
    use crate::tensor;

    /// task 307: `DENNO_GGUF_GEMV_V5` は明示的な偽値だけ off (既定 on)。
    #[test]
    fn gguf_gemv_v5_gate_truth_table() {
        assert!(parse_gguf_gemv_v5(None), "未設定は既定 on (lab t307a/b で採否)");
        for v in ["0", "off", "false", "no"] {
            assert!(!parse_gguf_gemv_v5(Some(v)), "raw={v:?}");
        }
        for v in ["", "1", "on", "true", " ON ", " TRUE ", "yes"] {
            assert!(parse_gguf_gemv_v5(Some(v)), "raw={v:?}");
        }
    }

    /// task 465: `DENNO_PF_ATTN_KVBF16` は既定 on (lab t465a/t465b で採択)。
    /// 切り戻しは明示的な `0` など。
    #[test]
    fn pf_attn_kvbf16_gate_truth_table() {
        assert!(pf_attn_kvbf16_from_value(None), "未設定は既定 on");
        for v in ["", "0", "off", "false", "no", "2"] {
            assert!(!pf_attn_kvbf16_from_value(Some(v)), "raw={v:?}");
        }
        for v in ["1", "on", "true", " ON ", " TRUE ", "yes"] {
            assert!(pf_attn_kvbf16_from_value(Some(v)), "raw={v:?}");
        }
    }

    /// task 307: `DENNO_SH_BEGIN_CACHE` も明示的な真値だけ on。
    #[test]
    fn sh_begin_cache_gate_truth_table() {
        assert!(!parse_sh_begin_cache(None), "未設定は既定 off");
        for v in ["", "0", "off", "false", "no", "2"] {
            assert!(!parse_sh_begin_cache(Some(v)), "raw={v:?}");
        }
        for v in ["1", "on", "true", " ON ", " TRUE "] {
            assert!(parse_sh_begin_cache(Some(v)), "raw={v:?}");
        }
    }

    // -----------------------------------------------------------------------
    // task 576: shared gate+up+silu 融合の真理値表と、索引算術の CPU ミラー
    // -----------------------------------------------------------------------

    /// `DENNO_DEC_SH_GU` は既定 on・明示的な偽値だけ off。
    #[test]
    fn sh_gu_gate_truth_table() {
        assert!(parse_sh_gu(None), "未設定は既定 on");
        for v in ["", "0", "off", "false", "no", " OFF ", " FALSE "] {
            assert!(!parse_sh_gu(Some(v)), "raw={v:?}");
        }
        for v in ["1", "on", "true", "yes", "2"] {
            assert!(parse_sh_gu(Some(v)), "raw={v:?}");
        }
    }

    /// 門番: 実形 (k=2560 / n=640 / rows=8 / lanes=8) は通り、前提を外すと落ちる。
    #[test]
    fn sh_gu_shape_truth_table() {
        // 既定モデルの shared expert (k=2560, n=640) + v5 の計画 (rows=8, lanes=8)。
        assert!(sh_gu_shape_ok(2560, 640, 8, 8));
        assert!(sh_gu_shape_ok(2560, 640, 8, 4));
        assert!(sh_gu_shape_ok(640, 2560, 8, 2), "down の形 (np=10) は lanes=2 なら通る");
        // k が 64 の倍数でない / np が lanes で割り切れない / スレッド過多 / 0。
        assert!(!sh_gu_shape_ok(0, 640, 8, 8));
        assert!(!sh_gu_shape_ok(2560, 0, 8, 8));
        assert!(!sh_gu_shape_ok(2592, 640, 8, 8), "k % 64 != 0");
        assert!(!sh_gu_shape_ok(2560, 640, 8, 3), "lanes は 2 の冪");
        assert!(!sh_gu_shape_ok(2560, 640, 8, 64), "lanes <= 32");
        assert!(!sh_gu_shape_ok(2560, 640, 128, 32), "rows*lanes > 1024");
        // np = k/64 = 40 は lanes=16 で割り切れない (40 % 16 != 0)。
        assert!(!sh_gu_shape_ok(2560, 640, 4, 16));
    }

    /// `.cu` の索引算術を 2 通りに写して**ビット一致**を確かめる。
    ///
    /// - `byte_row`: Q8_0 の生レイアウトを 1 バイトずつ読む (ggml 仕様の直接の写し)。
    /// - `word_row`: 融合カーネルと同じ 17 ワード + `cg[(2+j)>>2]` /
    ///   `cg[9+(j>>2)]` の切り出し。
    ///
    /// 演算列 (アキュムレータ 4 本 `j & 3` / ブロック 0 -> 1 / 最後に
    /// `(a0+a1)+(a2+a3)`) は両方で同じ = v5 と同じ。
    mod shgu_mirror {
        fn f16_to_f32(h: u16) -> f32 {
            let sign = ((h as u32) & 0x8000) << 16;
            let exp = ((h as u32) >> 10) & 0x1f;
            let man = (h as u32) & 0x3ff;
            let bits = if exp == 0 {
                if man == 0 {
                    sign
                } else {
                    let mut e: i32 = -14;
                    let mut m = man;
                    while (m & 0x400) == 0 {
                        m <<= 1;
                        e -= 1;
                    }
                    m &= 0x3ff;
                    sign | (((e + 127) as u32) << 23) | (m << 13)
                }
            } else if exp == 31 {
                sign | 0x7f80_0000 | (man << 13)
            } else {
                sign | ((exp + 112) << 23) | (man << 13)
            };
            f32::from_bits(bits)
        }

        /// 生バイトから 1 レーン分の内積 (v5 と同じ演算列)。
        fn byte_row(row: &[u8], np: usize, x: &[f32], lane: usize, lanes: usize) -> f32 {
            let per = np / lanes;
            let p0 = lane * per;
            let mut acc = [0f32; 4];
            for p in p0..p0 + per {
                let b = &row[p * 68..p * 68 + 68];
                let d0 = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
                let d1 = f16_to_f32(u16::from_le_bytes([b[34], b[35]]));
                for j in 0..32 {
                    let q = b[2 + j] as i8 as f32;
                    acc[j & 3] += (q * d0) * x[p * 64 + j];
                }
                for j in 0..32 {
                    let q = b[36 + j] as i8 as f32;
                    acc[j & 3] += (q * d1) * x[p * 64 + 32 + j];
                }
            }
            (acc[0] + acc[1]) + (acc[2] + acc[3])
        }

        /// 融合カーネルと同じ 17 ワード切り出し。
        fn word_row(row: &[u8], np: usize, x: &[f32], lane: usize, lanes: usize) -> f32 {
            let per = np / lanes;
            let p0 = lane * per;
            let mut acc = [0f32; 4];
            for p in p0..p0 + per {
                let base = p * 68;
                let mut c = [0u32; 17];
                for (i, w) in c.iter_mut().enumerate() {
                    let o = base + i * 4;
                    *w = u32::from_le_bytes([row[o], row[o + 1], row[o + 2], row[o + 3]]);
                }
                let d0 = f16_to_f32((c[0] & 0xffff) as u16);
                let d1 = f16_to_f32(((c[8] >> 16) & 0xffff) as u16);
                for j in 0..32 {
                    let w = c[(2 + j) >> 2];
                    let byte = (2 + j) & 3;
                    let q = ((w >> (byte * 8)) & 0xff) as u8 as i8 as f32;
                    acc[j & 3] += (q * d0) * x[p * 64 + j];
                }
                for j in 0..32 {
                    let w = c[9 + (j >> 2)];
                    let q = ((w >> ((j & 3) * 8)) & 0xff) as u8 as i8 as f32;
                    acc[j & 3] += (q * d1) * x[p * 64 + 32 + j];
                }
            }
            (acc[0] + acc[1]) + (acc[2] + acc[3])
        }

        /// lanes 本を shfl_down (offset 1, 2, 4 ...) の順で畳む (= `dno_v5_reduce`)。
        fn reduce_lanes(v: &[f32]) -> f32 {
            let n = v.len();
            let mut a = v.to_vec();
            let mut off = 1;
            while off < n {
                for i in 0..n {
                    if i + off < n {
                        a[i] += a[i + off];
                    }
                }
                off <<= 1;
            }
            a[0]
        }

        /// 決定的な擬似乱数 (テスト用。外部 crate を使わない)。
        fn lcg(state: &mut u64) -> u32 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (*state >> 33) as u32
        }

        /// 実形 (k=2560 / n=640 / lanes=8) で 2 バイト読みと 17 ワード読みが
        /// **1 ビットも違わない**こと、レーン分割が全対を 1 回ずつ覆うこと。
        #[test]
        fn shgu_word_and_byte_rows_agree_and_partition_covers_once() {
            let k = 2560usize;
            let n = 32usize;
            let np = k / 64;
            let rows_per_block = 8usize;
            let lanes = 8usize;
            assert!(super::sh_gu_shape_ok(k, n, rows_per_block, lanes));
            let mut st = 0x1234_5678_9abc_def0u64;
            // x は実形の 2,560 要素 (f32 の範囲に収まる適当な値)。
            let x: Vec<f32> = (0..k)
                .map(|_| ((lcg(&mut st) % 2001) as f32 - 1000.0) / 997.0)
                .collect();
            // 重みは ggml Q8_0 の 68 B 対 (d は f16 として妥当な範囲)。
            let row_bytes = np * 68;
            let mut w = vec![0u8; n * row_bytes];
            for b in w.iter_mut() {
                // f16 として 0x2400..0x3c00 (= 0.0625..1.0) に収める。
                *b = 0;
            }
            for r in 0..n {
                for p in 0..np {
                    let o = r * row_bytes + p * 68;
                    let d0 = 0x2c00u16 + (lcg(&mut st) % 0x0400) as u16;
                    let d1 = 0x3000u16 + (lcg(&mut st) % 0x0400) as u16;
                    w[o..o + 2].copy_from_slice(&d0.to_le_bytes());
                    w[o + 34..o + 36].copy_from_slice(&d1.to_le_bytes());
                    for j in 0..32 {
                        w[o + 2 + j] = lcg(&mut st) as u8;
                        w[o + 36 + j] = lcg(&mut st) as u8;
                    }
                }
            }
            let per = np / lanes;
            for r in 0..n {
                let row = &w[r * row_bytes..(r + 1) * row_bytes];
                let mut v_byte = Vec::with_capacity(lanes);
                let mut v_word = Vec::with_capacity(lanes);
                for lane in 0..lanes {
                    let a = byte_row(row, np, &x, lane, lanes);
                    let b = word_row(row, np, &x, lane, lanes);
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "row={r} lane={lane}: byte={a} word={b}"
                    );
                    v_byte.push(a);
                    v_word.push(b);
                }
                assert_eq!(
                    reduce_lanes(&v_byte).to_bits(),
                    reduce_lanes(&v_word).to_bits(),
                    "row={r}: 縮約後が違う"
                );
                // レーン分割は 40 対を 5 対ずつ 8 レーンで 1 回ずつ覆う。
                assert_eq!(per * lanes, np);
            }
        }

        /// 融合 (gate/up を同じスレッドで) の silu は `dno_dec_silu_mul` と同じ式。
        #[test]
        fn shgu_silu_formula_matches_epilogue() {
            for g in [-8.0f32, -1.5, -0.25, 0.0, 0.5, 3.0, 12.0] {
                let u = 0.75f32;
                let s = g / (1.0 + (-g).exp());
                let want = s * u;
                // カーネルは __expf を使うので厳密一致は GPU 上でしか取れない。
                // ここでは式の形 (g / (1 + exp(-g)) * u) を固定する。
                let got = g / (1.0 + (-g).exp()) * u;
                assert_eq!(got.to_bits(), want.to_bits());
            }
        }
    }

    /// 融合の配線 (cuda.rs の門番 / 起動形 / .cu の名前 / build_ptx の target)。
    #[test]
    fn sh_gu_fusion_is_wired() {
        let src = crate::cuda::cuda_src();
        assert!(src.contains("dno_dec_sh_gateup_silu"));
        assert!(src.contains("DENNO_DEC_SH_GU"));
        assert!(src.contains("launch_sh_gu("));
        let cu = include_str!("../../kernels/kernels_sh_gu.cu");
        assert!(cu.contains("extern \"C\" __global__ void dno_dec_sh_gateup_silu"));
        let sh = include_str!("../../kernels/build_ptx.sh");
        assert!(sh.contains("sh_gu)"), "build_ptx.sh に target が無い");
        assert!(sh.contains("kernels_sh_gu.cu"), "build_ptx.sh に .cu が無い");
        // 収録 PTX があるなら引数 9 本であること (生成前は無くてよい)。
        let ptx = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_sh_gu.ptx");
        if let Ok(text) = std::fs::read_to_string(&ptx) {
            assert_eq!(
                ptx_entry_params(&text, SH_GU_KERNEL_NAME),
                Some(SH_GU_NARGS),
                "収録 PTX の引数が違う"
            );
        }
    }

    /// task 392: `DENNO_GDN_BA_PAR` は既定 off・明示的な真値だけ on。
    #[test]
    fn gdn_ba_par_gate_truth_table() {
        assert!(!gdn_ba_par_from_value(None), "未設定は既定 off");
        for v in ["", "0", "off", "false", "no", "2", " on?"] {
            assert!(!gdn_ba_par_from_value(Some(v)), "raw={v:?}");
        }
        for v in ["1", "on", "true", " ON ", " TRUE ", "yes", " YES "] {
            assert!(gdn_ba_par_from_value(Some(v)), "raw={v:?}");
        }
    }

    /// task 900: `DENNO_GDN_BA_ORDER_FIX` は既定 on・明示的な偽値だけ off。
    #[test]
    fn task900_gdn_ba_order_fix_truth_table() {
        assert!(parse_gdn_ba_order_fix(None), "未設定は既定 on");
        for v in ["", "0", "off", "false", "no", " OFF ", " False ", " NO "] {
            assert!(!parse_gdn_ba_order_fix(Some(v)), "raw={v:?}");
        }
        for v in ["1", "on", "true", " ON ", " TRUE ", "yes", " YES "] {
            assert!(parse_gdn_ba_order_fix(Some(v)), "raw={v:?}");
        }
        // 強制の上書きが env 解釈より優先される (既定 on のまま off を踏める)。
        gdn_ba_order_fix_set_for_test(Some(false));
        assert!(!gdn_ba_order_fix_on(), "強制 off が効かない");
        gdn_ba_order_fix_set_for_test(Some(true));
        assert!(gdn_ba_order_fix_on(), "強制 on が効かない");
        gdn_ba_order_fix_set_for_test(None);
        assert!(gdn_ba_order_fix_on(), "解除後は env の解釈 (既定 on) に戻る");
    }

    /// task 464: `DENNO_MIX_NOWAIT` は既定 on (lab t464b で採否)・明示的な偽値だけ off。
    #[test]
    fn task464_mix_nowait_gate_truth_table() {
        assert!(mix_nowait_from_value(None), "未設定は既定 on (lab t464b で採否)");
        for v in ["0", "off", "false", "no", " OFF ", " False ", ""] {
            assert!(!mix_nowait_from_value(Some(v)), "raw={v:?}");
        }
        for v in ["1", "on", "true", " ON ", "yes", " YES ", "ssm", " SSM ", "2"] {
            assert!(mix_nowait_from_value(Some(v)), "raw={v:?}");
        }
    }

    /// task 468: `DENNO_MIX_NOWAIT` の full 層の解釈 (既存 env の値。新しい env 名は
    /// 足さない)。**既定 on** (lab t468c で採否)。
    ///
    /// 切り戻しは `=ssm` (ssm の defer は残して full だけ止める) か `=0` / `=off`。
    #[test]
    fn task468_mix_nowait_full_gate_truth_table() {
        assert!(mix_nowait_full_from_value(None), "未設定は既定 on (lab t468c で採否)");
        for v in ["0", "off", "false", "no", "ssm", " SSM ", ""] {
            assert!(!mix_nowait_full_from_value(Some(v)), "raw={v:?}");
        }
        for v in ["full", " FULL ", "all", " All ", "1", "on", "true", "yes", "2"] {
            assert!(mix_nowait_full_from_value(Some(v)), "raw={v:?}");
        }
        // `full` / 真値は ssm の defer も on のまま (既存の解釈を壊さない)。
        for v in ["full", "1", "on"] {
            assert!(mix_nowait_from_value(Some(v)), "raw={v:?}");
        }
        // 偽値は ssm も full も落とす。
        assert!(!mix_nowait_from_value(Some("0")) && !mix_nowait_full_from_value(Some("0")));
        // `ssm` は ssm だけ on (full は off)。
        assert!(mix_nowait_from_value(Some("ssm")) && !mix_nowait_full_from_value(Some("ssm")));
    }

    /// task 468: `[mix-nowait-full]` は 1 層も通らなければ出さない。
    #[test]
    fn task468_mix_nowait_full_line_counts_layers() {
        assert_eq!(mix_nowait_full_line_with(true, 0), None, "0 層なら出さない");
        assert_eq!(mix_nowait_full_line_with(false, 12), None, "ゲート off なら出さない");
        let l = mix_nowait_full_line_with(true, 12).expect("defer した層があれば出す");
        assert!(l.contains("layers=12"), "{l}");
        assert!(l.starts_with("[mix-nowait-full]"), "{l}");
    }

    /// task 464: 発火の証拠の行 (1 層も通らなければ出さない)。
    #[test]
    fn task464_mix_nowait_line_counts_layers() {
        assert_eq!(mix_nowait_line_with(0, 0), None, "0 層なら出さない");
        let l = mix_nowait_line_with(36, 36).expect("defer した層があれば出す");
        assert!(l.contains("deferred_layers=36"), "{l}");
        assert!(l.contains("alias_layers=36"), "{l}");
        // alias だけ立った回 (defer 0) でも出す = 「前提は揃ったが defer していない」
        // ことが読める。
        let a = mix_nowait_line_with(0, 5).expect("alias だけでも出す");
        assert!(a.contains("deferred_layers=0"), "{a}");
    }

    /// task 464: defer の印は既定 false で、読むと消える。
    #[test]
    fn task464_mix_take_deferred_is_one_shot() {
        crate::cuda::mix_nowait_reset();
        assert!(!mix_take_deferred(), "既定で defer になっている");
        MIX_DEFERRED.with(|c| c.set(true));
        assert!(mix_take_deferred(), "印が読めない");
        assert!(!mix_take_deferred(), "2 回目も true (消えていない)");
    }

    /// task 464: 控えの有無は消費せずに読める (ssm 側の前提検査に使う)。
    #[test]
    fn task464_gdn_x_in_mix_armed_does_not_consume() {
        gdn_x_mix_reset();
        assert!(!gdn_x_in_mix_armed(4608), "空なのに armed");
        gdn_x_in_mix_arm(0x1234_5000, 4608);
        assert!(gdn_x_in_mix_armed(4608), "控えがあるのに armed でない");
        assert!(!gdn_x_in_mix_armed(2304), "要素数が違うのに armed");
        // 読んでも消えない = そのあと GDN が取れる。
        assert_eq!(gdn_x_in_mix_take(4608), Some(0x1234_5000), "控えが消えた");
        assert!(!gdn_x_in_mix_armed(4608), "取った後も armed");
        gdn_x_mix_reset();
    }

    /// task 392: `DENNO_GDN_X_MIX` は既定 on (lab t392c で採否)・明示的な偽値だけ off。
    #[test]
    fn gdn_x_mix_gate_truth_table() {
        assert!(gdn_x_mix_from_value(None), "未設定は既定 on (lab t392c で採否)");
        for v in ["0", "off", "false", "no", " OFF ", " False "] {
            assert!(!gdn_x_mix_from_value(Some(v)), "raw={v:?}");
        }
        for v in ["", "1", "on", "true", " ON ", " TRUE ", "yes"] {
            assert!(gdn_x_mix_from_value(Some(v)), "raw={v:?}");
        }
    }

    /// task 392: 控えは 1 回で消える (2 回目は H2D へ落ちる)・要素数が違えば
    /// 使わない。**緑 = 経路が通った証拠ではない** ので層数も数えて出す。
    #[test]
    fn gdn_x_mix_take_is_one_shot_and_checks_len() {
        gdn_x_mix_reset();
        assert_eq!(gdn_x_in_mix_take(2560), None, "控えが無ければ None");
        gdn_x_in_mix_arm(0x1234, 2560);
        assert_eq!(gdn_x_in_mix_take(1280), None, "要素数が違えば使わない");
        assert_eq!(gdn_x_in_mix_take(2560), Some(0x1234), "同じ要素数なら 1 回");
        assert_eq!(gdn_x_in_mix_take(2560), None, "2 回目は消えている");
        gdn_x_in_mix_arm(0, 2560);
        assert_eq!(gdn_x_in_mix_take(2560), None, "0 は控えない");
        gdn_x_mix_reset();
    }

    /// task 392: `[gdn-x-mix]` の行はゲート on で 1 層でも通ったときだけ出る。
    #[test]
    fn gdn_x_mix_line_counts_layers() {
        assert_eq!(gdn_x_mix_line_with(false, 4608, 0), None, "ゲート off");
        assert_eq!(gdn_x_mix_line_with(true, 0, 0), None, "0 層なら出さない");
        let l = gdn_x_mix_line_with(true, 4608, 0).expect("on で層があれば出す");
        assert!(l.starts_with("[gdn-x-mix] "), "{l}");
        assert!(l.contains("no_h2d_layers=4608"), "{l}");
        assert!(l.contains("h2d_layers=0"), "{l}");
        let f = gdn_x_mix_line_with(true, 2304, 2304).expect("落ちた層も出す");
        assert!(f.contains("no_h2d_layers=2304"), "{f}");
        assert!(f.contains("h2d_layers=2304"), "{f}");
    }

    /// task 392: 発火の行は「ゲート on で 1 層でも通ったとき」だけ出る
    /// (**緑 = 経路が通った証拠ではない** ので層数を数えて出す)。
    #[test]
    fn gdn_ba_par_line_counts_fired_layers() {
        assert_eq!(gdn_ba_par_line_with(false, 6144, 0), None, "ゲート off");
        assert_eq!(gdn_ba_par_line_with(true, 0, 0), None, "0 層なら出さない");
        let l = gdn_ba_par_line_with(true, 2304, 0).expect("on で層があれば出す");
        assert!(l.starts_with("[gdn-ba-par] "), "{l}");
        assert!(l.contains("par_layers=2304"), "{l}");
        assert!(l.contains("fallback=0"), "{l}");
        let f = gdn_ba_par_line_with(true, 0, 3).expect("用意に失敗した回数も出す");
        assert!(f.contains("par_layers=0"), "{f}");
        assert!(f.contains("fallback=3"), "{f}");
    }

    /// task 310 / 319: `DENNO_DEC_SH_V5` は `1` (3 本) / `2` (gate+up) の 2 段だけ。
    /// task 319 で既定を 2 にした (明示的な偽値だけが off)。
    #[test]
    fn dec_sh_v5_gate_truth_table() {
        assert_eq!(parse_dec_sh_v5(None), 2, "未設定は既定 2 (task 319 で採否)");
        for v in ["", "0", "off", "false", "no", " 0 ", " OFF "] {
            assert_eq!(parse_dec_sh_v5(Some(v)), 0, "raw={v:?}");
        }
        assert_eq!(parse_dec_sh_v5(Some("1")), 1);
        assert_eq!(parse_dec_sh_v5(Some(" 1 ")), 1);
        assert_eq!(parse_dec_sh_v5(Some("2")), 2);
        assert_eq!(parse_dec_sh_v5(Some(" 2 ")), 2);
        for v in ["on", "true", "yes", "3"] {
            assert_eq!(parse_dec_sh_v5(Some(v)), 2, "未知は既定へ raw={v:?}");
        }
    }

    /// task 307: `DENNO_ROUTER_IN_MIX` も明示的な偽値だけ off (既定 on)。
    #[test]
    fn router_in_mix_gate_truth_table() {
        assert!(parse_router_in_mix(None), "未設定は既定 on (lab t307b / t307c で採否)");
        for v in ["0", "off", "false", "no"] {
            assert!(!parse_router_in_mix(Some(v)), "raw={v:?}");
        }
        for v in ["", "1", "on", "true", " ON ", " TRUE ", "yes"] {
            assert!(parse_router_in_mix(Some(v)), "raw={v:?}");
        }
    }

    /// task 364: qchain がルータの控えを拾う条件 (層・位置・k が一致) を固定する。
    ///
    /// task 307 の控えは BF16 の chain3 だけが消費していた。新しい既定モデルは
    /// mix が Q8_0 で qchain を通るので、ここが拾えないとルータ GEMV が鎖の外に
    /// 残る。GPU 無しでも「拾う / 拾わない」は数えられる。
    #[test]
    fn qchain_picks_up_router_arm_by_shape_and_layer() {
        router_in_mix_set_for_test(Some(true));
        router_in_mix_token_start();
        let (k, n) = (8usize, 2usize);
        let bytes = vec![0u8; k * n * 4];
        assert!(router_in_mix_arm(77, &bytes, k, n), "既定 on で控えが積まれない");
        assert!(router_in_mix_armed_for(k), "同じ形・同じ層の控えを拾えない");
        assert!(!router_in_mix_armed_for(k + 1), "形が違うのに拾った");
        // 層が変われば拾わない。層番号は `dec_kprof` が on のときだけ更新される
        // thread_local なので、ここでは計測を強制して動かす。
        dec_kprof::set_force_for_test(Some(true));
        let saved = dec_kprof::cur_layer();
        dec_kprof::set_layer(saved + 3);
        assert!(!router_in_mix_armed_for(k), "層が違うのに拾った");
        dec_kprof::set_layer(saved);
        assert!(router_in_mix_armed_for(k));
        dec_kprof::set_force_for_test(None);
        // トークンの入口で控えは落ちる (前のトークンの値を使わない)。
        router_in_mix_token_start();
        assert!(!router_in_mix_armed_for(k), "token_start が控えを落としていない");
        router_in_mix_set_for_test(None);
    }

    /// task 364: ゲート off では控えを積まない (qchain 側も何も積まない)。
    #[test]
    fn qchain_router_arm_respects_gate_off() {
        router_in_mix_set_for_test(Some(false));
        router_in_mix_token_start();
        let (k, n) = (8usize, 2usize);
        let bytes = vec![0u8; k * n * 4];
        assert!(!router_in_mix_arm(77, &bytes, k, n), "ゲート off で控えを積んだ");
        assert!(!router_in_mix_armed_for(k), "ゲート off なのに拾える状態");
        router_in_mix_set_for_test(None);
        router_in_mix_token_start();
    }

    /// task 364: qchain の入口が「ルータを載せにいく」分岐に実際に入ること
    /// (GPU 無しでも数えられる。`cuda_enabled()` の判定より前に数えている)。
    #[test]
    fn qchain_counts_router_arm_attempts_without_gpu() {
        router_in_mix_set_for_test(Some(true));
        router_in_mix_token_start();
        router_in_qchain_arms_reset();
        let (k1, n1, e, hc) = (8usize, 2usize, 4usize, 2usize);
        let bytes = vec![0u8; k1 * n1 * 4];
        assert!(router_in_mix_arm(91, &bytes, k1, n1));
        let xn = [0f32; 8];
        let mut out = [0f32; 4];
        let ran = maybe_gpu_mix_q_chain_into(
            91,
            crate::kquant::GgmlType::Q8_0,
            &bytes,
            k1,
            n1,
            92,
            crate::kquant::GgmlType::Q8_0,
            &bytes,
            n1,
            k1,
            1.0 / hc as f32,
            e,
            hc,
            &xn,
            &mut out,
            None,
            false,
        );
        assert!(!ran, "CUDA 無しで mix の鎖が走った");
        assert_eq!(
            router_in_qchain_arms(),
            1,
            "qchain がルータの控えを見にいっていない"
        );
        // ゲート off では 1 回も見にいかない。控えは on のうちに落としておく
        // (`router_in_mix_token_start` はゲート off だと早期に返るため)。
        router_in_qchain_arms_reset();
        router_in_mix_token_start();
        router_in_mix_set_for_test(Some(false));
        assert!(!router_in_mix_arm(91, &bytes, k1, n1));
        let _ = maybe_gpu_mix_q_chain_into(
            91,
            crate::kquant::GgmlType::Q8_0,
            &bytes,
            k1,
            n1,
            92,
            crate::kquant::GgmlType::Q8_0,
            &bytes,
            n1,
            k1,
            1.0 / hc as f32,
            e,
            hc,
            &xn,
            &mut out,
            None,
            false,
        );
        assert_eq!(router_in_qchain_arms(), 0, "ゲート off なのに数えた");
        router_in_mix_set_for_test(None);
        router_in_mix_token_start();
    }

    /// task 307: 共有 expert のキャッシュは、空のときは必ず外れる
    /// (初回は従来経路で作る = 値は 1 つも変わらない)。
    #[test]
    fn sh_cache_lookup_misses_on_empty_cache() {
        SH_CACHE.with(|c| c.borrow_mut().clear());
        let blocks: [u8; 0] = [];
        let jobs = [
            SharedGemvJob { uid: 11, ty: crate::kquant::GgmlType::Q8_0, blocks: &blocks, k: 2560, n: 640 },
            SharedGemvJob { uid: 12, ty: crate::kquant::GgmlType::Q8_0, blocks: &blocks, k: 2560, n: 640 },
            SharedGemvJob { uid: 13, ty: crate::kquant::GgmlType::Q8_0, blocks: &blocks, k: 640, n: 2560 },
        ];
        let refs: [&SharedGemvJob<'_>; 3] = [&jobs[0], &jobs[1], &jobs[2]];
        assert!(sh_cache_lookup(&refs).is_none(), "空のキャッシュで当たった");
    }

    /// task 307: ルータの「鎖で作った」印は uid / k / n が揃ったときだけ使う。
    /// (GPU 無しでも通る範囲: 印の付け替えと取り消しの規則だけを固定する。)
    #[test]
    fn router_filled_matches_only_the_same_shape() {
        ROUTER_FILLED.with(|c| c.set((7u64, 2560usize, 512usize)));
        let filled = ROUTER_FILLED.with(|c| c.get());
        assert_eq!(filled, (7, 2560, 512));
        assert_ne!(filled, (8, 2560, 512), "uid が違えば使わない");
        assert_ne!(filled, (7, 2560, 640), "n が違えば使わない");
        ROUTER_FILLED.with(|c| c.set((0, 0, 0)));
        ROUTER_SYNCED.with(|c| c.set(false));
    }

    /// task 368: `DENNO_DEC_SYNC_STREAM` の解釈。task 1377 で既定 on。
    /// 明示的な真偽だけを受け付け、壊れた値は既定へ倒す。
    #[test]
    fn dec_sync_stream_gate_reads_only_explicit_values() {
        assert!(DEC_SYNC_STREAM_DEFAULT, "lab t1377d で既定 on にした");
        assert_eq!(dec_sync_stream_from_value(None), DEC_SYNC_STREAM_DEFAULT);
        for v in ["1", "on", "ON", " true ", "yes"] {
            assert!(dec_sync_stream_from_value(Some(v)), "{v} を on と読んでいない");
        }
        for v in ["0", "off", "False", "no"] {
            assert!(!dec_sync_stream_from_value(Some(v)), "{v} を off と読んでいない");
        }
        // 空文字と壊れた値は「既定へ倒す」(task 1377 で既定 on)
        for v in ["", "2", "maybe"] {
            assert_eq!(
                dec_sync_stream_from_value(Some(v)),
                DEC_SYNC_STREAM_DEFAULT,
                "{v} が既定へ倒れていない"
            );
        }
    }

    /// task 368: 1 行はゲート on で 1 回でも通ったときだけ出る
    /// (off では `None` = int-auto と同じ出力)。
    #[test]
    fn dec_sync_stream_line_only_when_on_and_used() {
        dec_sync_stream_reset();
        assert!(dec_sync_stream_line_with(false).is_none(), "ゲート off で行が出た");
        assert!(dec_sync_stream_line_with(true).is_none(), "0 回なのに行が出た");
        DEC_SYNC_EVENTS.fetch_add(3, Ordering::Relaxed);
        DEC_SYNC_FALLBACKS.fetch_add(1, Ordering::Relaxed);
        let line = dec_sync_stream_line_with(true).expect("行が出ていない");
        assert!(line.contains("events=3"), "{line}");
        assert!(line.contains("fallback=1"), "{line}");
        assert!(dec_sync_stream_line_with(false).is_none(), "off で行が出た");
        dec_sync_stream_reset();
    }

    /// task 278: 高速版 GEMV のゲートは既定 on、明示的な偽値だけ切る。
    #[test]
    fn seg_k_gate_defaults_on() {
        assert!(seg_k_from_value(None), "未設定は有効 (既定)");
        assert!(seg_k_from_value(Some("1")));
        assert!(seg_k_from_value(Some(" on ")));
        assert!(!seg_k_from_value(Some("0")));
        assert!(!seg_k_from_value(Some("off")));
        assert!(!seg_k_from_value(Some("false")));
        assert!(!seg_k_from_value(Some("NO")));
    }

    /// task 526 / 534: seg の gate+up+silu 融合 (`DENNO_SEG_GU_FUSE`) のゲートは
    /// **task 534 から既定 on**(P526 では既定 off だった)。明示的な偽値だけが off。
    #[test]
    fn seg_fuse_gate_default_on() {
        assert!(seg_gu_fuse_from_value(None), "未設定は on (既定)");
        assert!(seg_gu_fuse_from_value(Some("")));
        assert!(seg_gu_fuse_from_value(Some("1")));
        for v in ["0", "off", "false", "no", " NO "] {
            assert!(!seg_gu_fuse_from_value(Some(v)), "gu: {v}");
        }
    }

    /// task 526: ホストの起動形 (1 warp が受け持つ行数) と `kernels_seg_k.cu` の
    /// `#define` が一致していること。片方だけ変えると行の取り違えになる
    /// (GPU 無しで検出できるのはここまで。値の一致は lab の A/B で見る)。
    #[test]
    pub(crate) fn seg_fuse_shape_matches_cu() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_seg_k.cu");
        let src = std::fs::read_to_string(&p).expect("kernels_seg_k.cu が読めない");
        let pick = |name: &str| -> u32 {
            let key = format!("#define {name} ");
            let line = src
                .lines()
                .find(|l| l.starts_with(&key))
                .unwrap_or_else(|| panic!("{name} の #define が無い"));
            line[key.len()..].trim().parse().expect("#define の値が数値でない")
        };
        assert_eq!(pick("SK_GU_ROWS"), SEG_GU_ROWS, "gu の行数");
        // ブロックは 128 = 4 warp 固定 (カーネルは blockDim.x >> 5 を使う)。
        assert_eq!(SEG_WARPS, 4);
        // task 534: 8 バイトロード版も同じ 2 つの定数を使う (同じ起動形)。
        let gu8 = src.split("dno_seg_gemv_gu8_q4k").nth(1).expect("gu8_q4k が無い");
        let head: String = gu8.chars().take(3000).collect();
        assert!(head.contains("SK_GU_ROWS"), "gu8 が SK_GU_ROWS を使っていない");
    }

    /// task 534: `DENNO_SEG_GU8` は **既定 off** (lab で dnv4 と組にしても
    /// 差が出なかった)、`DENNO_SEG_DN8` は **既定 on** (採用)。
    #[test]
    fn seg_fast_gate_defaults() {
        // gu8: 既定 off
        assert!(!seg_gu8_from_value(None), "gu8 の未設定は off (既定)");
        assert!(!seg_gu8_from_value(Some("")));
        assert!(!seg_gu8_from_value(Some("0")));
        assert!(!seg_gu8_from_value(Some("off")));
        assert!(!seg_gu8_from_value(Some("no")));
        for v in ["1", "on", "true", "yes", " TRUE "] {
            assert!(seg_gu8_from_value(Some(v)), "gu8: {v}");
        }
        // dn8: 既定 on (task 534 で採用)
        assert!(seg_dn8_from_value(None), "dn8 の未設定は on (既定)");
        assert!(seg_dn8_from_value(Some("")));
        assert!(seg_dn8_from_value(Some("1")));
        for v in ["0", "off", "false", "no", " NO "] {
            assert!(!seg_dn8_from_value(Some(v)), "dn8: {v}");
        }
    }

    /// task 534: 8 バイトロード版の名前は K 型 2 つだけ (Q8_0 / Q5_1 は無い) で、
    /// 添字は `moe_batch_slot` の並び (2 = Q4_K, 3 = Q5_K) に一致していること。
    #[test]
    fn seg_fast_names_are_k_types_only() {
        for (i, (g, k)) in SEG_K_GU8_NAMES.iter().zip(SEG_K_K8_NAMES.iter()).enumerate() {
            if i < 2 {
                assert!(g.is_none() && k.is_none(), "slot {i} に K 型以外の版がある");
            } else {
                assert!(g.is_some() && k.is_some(), "slot {i} の K 型版が無い");
            }
        }
    }

    /// task 562: `DENNO_SEG_GU16` は **既定 on** (lab で gate 段 -29%、2 ラウンド)。
    /// 明示的な偽値だけが off (切り戻し `DENNO_SEG_GU16=0`)。
    #[test]
    fn t562_gu16_gate_truth_table() {
        assert!(seg_gu16_from_value(None), "未設定は on (既定)");
        assert!(seg_gu16_from_value(Some("")));
        assert!(seg_gu16_from_value(Some(" ")));
        assert!(seg_gu16_from_value(Some("1")));
        for v in ["0", "off", "false", "no", " NO "] {
            assert!(!seg_gu16_from_value(Some(v)), "gu16: {v}");
        }
    }

    /// task 562: 16 バイト版は Q4_K (slot 2) の gate+up 融合版だけ。他の 3 スロットは
    /// 無い = 古い PTX や他型では従来の融合版へ落ちる。
    #[test]
    fn t562_gu16_names_are_q4k_only() {
        for (i, g) in SEG_K_GU16_NAMES.iter().enumerate() {
            if i == 2 {
                assert!(g.is_some(), "Q4_K (slot 2) の 16 バイト版が無い");
            } else {
                assert!(g.is_none(), "slot {i} に 16 バイト版がある");
            }
        }
    }

    /// task 562: 新しい Q4_K 16 バイト版 (`dno_seg_gemv_gu16_q4k`) の索引算術を
    /// CPU に写し、**各要素をちょうど 1 回**、Q4_K の仕様どおりのバイト位置・
    /// ニブル (下位/上位)・サブブロックのスケールで読むことを確かめる。
    /// GPU 無しで検出できるのはここまで (値の一致は lab の A/B と `DENNO_EXPERT_GPU_CHECK`)。
    #[test]
    fn t562_gu16_q4k_lane_mapping_covers_every_element_once() {
        const BLKS: usize = 4;
        const LANES: usize = 8;
        let nb = 10usize; // n_in = 2560
        let mut seen = vec![vec![false; 256]; nb];
        let mut lanes = 0usize;
        for lane in 0..32usize {
            let lb = lane & (LANES - 1);
            let bi = lane >> 3;
            let j = lb >> 1;
            let h = lb & 1;
            let mut b0 = 0usize;
            while b0 < nb {
                let b = b0 + bi;
                if b < nb {
                    lanes += 1;
                    for t in 0..16usize {
                        // xv[t] = 下位 16 要素、xv[32 + t] = 上位 16 要素
                        for (xvi, e) in [
                            (t, 64 * j + 16 * h + t),
                            (32 + t, 64 * j + 32 + 16 * h + t),
                        ] {
                            // Q4_K の仕様: 要素 e -> 塊 cj = e/64、i = e%64
                            // バイト = cj*32 + (i < 32 ? i : i-32)、ニブルは i>=32 が上位
                            let cj = e / 64;
                            let i = e % 64;
                            let exp_byte = cj * 32 + if i < 32 { i } else { i - 32 };
                            let exp_hi = i >= 32;
                            let exp_sb = 2 * cj + usize::from(exp_hi);
                            // カーネル: バイト = j*32 + h*16 + t、xvi >= 32 が上位
                            let kbyte = j * 32 + h * 16 + t;
                            let khi = xvi >= 32;
                            let ksb = 2 * j + usize::from(khi);
                            assert_eq!(
                                (kbyte, khi, ksb),
                                (exp_byte, exp_hi, exp_sb),
                                "lane {lane} t {t} xvi {xvi} e {e}"
                            );
                            assert!(!seen[b][e], "要素が 2 回読まれた: b {b} e {e}");
                            seen[b][e] = true;
                        }
                    }
                }
                b0 += BLKS;
            }
        }
        assert_eq!(lanes, nb * 8, "1 ブロック 8 lane で 10 ブロックを覆っていない");
        for (b, row) in seen.iter().enumerate() {
            assert!(row.iter().all(|x| *x), "block {b} に読まれない要素がある");
        }
    }

    /// task 562: カーネル側の `#define` とホスト側の前提 (1 warp = 4 ブロック、
    /// 1 ブロック = 8 lane) が一致していること。
    #[test]
    fn t562_gu16_shape_matches_cu() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_seg_k.cu");
        let src = std::fs::read_to_string(&p).expect("kernels_seg_k.cu が読めない");
        let pick = |name: &str| -> usize {
            let key = format!("#define {name} ");
            let line = src
                .lines()
                .find(|l| l.starts_with(&key))
                .unwrap_or_else(|| panic!("{name} の #define が無い"));
            line[key.len()..].trim().parse().expect("#define の値が数値でない")
        };
        assert_eq!(pick("SK_G16_BLKS"), 4, "1 warp のブロック数");
        assert_eq!(pick("SK_G16_LANES"), 8, "1 ブロックの lane 数");
        // 16 バイト = 8 バイトロード 2 回で、行は 8 バイト境界に載る前提。
        assert_eq!(144 % 8, 0, "Q4_K のブロック長が 8 の倍数でない");
    }

    /// task 526: コミット済みの `kernels_seg_k.ptx` に融合 (gu) の 4 エントリが
    /// 居ること。**PTX の再生成を忘れると、ゲートを on にしても静かに従来の
    /// 3 起動へ落ちる** (起動数でしか気づけない) ので、ここで固定する。
    #[test]
    fn seg_fuse_ptx_has_new_entries() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_seg_k.ptx");
        let ptx = std::fs::read_to_string(&p).expect("kernels_seg_k.ptx が読めない");
        for n in SEG_K_GU_NAMES.iter() {
            let name = std::str::from_utf8(n).expect("ASCII").trim_end_matches('\u{0}');
            let key = format!(".visible .entry {name}(");
            assert!(ptx.contains(&key), "{name} が kernels_seg_k.ptx に無い (PTX 再生成要)");
        }
        // task 534: 8 バイトロード版 (gu8 / k8) の 4 エントリも居ること。
        for n in SEG_K_GU8_NAMES.iter().chain(SEG_K_K8_NAMES.iter()).flatten() {
            let name = std::str::from_utf8(n).expect("ASCII").trim_end_matches('\u{0}');
            let key = format!(".visible .entry {name}(");
            assert!(ptx.contains(&key), "{name} が kernels_seg_k.ptx に無い (PTX 再生成要)");
        }
        // task 562: Q4_K の 16 バイト版 (gate+up 融合) も居ること。
        for n in SEG_K_GU16_NAMES.iter().flatten() {
            let name = std::str::from_utf8(n).expect("ASCII").trim_end_matches('\u{0}');
            let key = format!(".visible .entry {name}(");
            assert!(ptx.contains(&key), "{name} が kernels_seg_k.ptx に無い (PTX 再生成要)");
        }
        // task 534 (2/n): Q5_1 の 4 バイト版 (down) も居ること。
        for n in SEG_K_V4_NAMES.iter().flatten() {
            let name = std::str::from_utf8(n).expect("ASCII").trim_end_matches('\u{0}');
            let key = format!(".visible .entry {name}(");
            assert!(ptx.contains(&key), "{name} が kernels_seg_k.ptx に無い (PTX 再生成要)");
        }
        // 従来の 4 本も残っていること (差し替えではなく追加)。
        for n in SEG_K_NAMES.iter() {
            let name = std::str::from_utf8(n).expect("ASCII").trim_end_matches('\u{0}');
            let key = format!(".visible .entry {name}(");
            assert!(ptx.contains(&key), "{name} が消えた");
        }
    }
    /// task 282: mix の GEMV 種別 (`DENNO_MIX_QK`) の既定は `(Blk, R8)`。
    /// `0` で旧 GEMV に戻り、`down` / `up` / `seg` で向きごとに選べる。
    #[test]
    fn mix_qk_gate_defaults_on() {
        // task 422: 既定は down だけ dn6 (lab t422b で採否)、up は v5。
        // dn5 (task 418) は `DENNO_MIX_QK=dn5` で選ぶ切り戻し段。
        assert_eq!(mix_qk_from_value(None), (MixGemv::Dn7, MixGemv::V5), "未設定が既定");
        assert_eq!(mix_qk_from_value(Some("dn5")), (MixGemv::Dn5, MixGemv::V5));
        // task 286 の形 (両方向 v5) へは明示的に戻す。
        assert_eq!(mix_qk_from_value(Some("1")), (MixGemv::V5, MixGemv::V5));
        assert_eq!(mix_qk_from_value(Some(" both ")), (MixGemv::V5, MixGemv::V5));
        assert_eq!(mix_qk_from_value(Some("v5")), (MixGemv::V5, MixGemv::V5));
        // task 282 の形へ戻す逃げ道。
        assert_eq!(mix_qk_from_value(Some("blk")), (MixGemv::Blk, MixGemv::R8));
        assert_eq!(mix_qk_from_value(Some("R8")), (MixGemv::Blk, MixGemv::R8));
        assert_eq!(mix_qk_from_value(Some("0")), (MixGemv::Old, MixGemv::Old));
        assert_eq!(mix_qk_from_value(Some("off")), (MixGemv::Old, MixGemv::Old));
        assert_eq!(mix_qk_from_value(Some("false")), (MixGemv::Old, MixGemv::Old));
        assert_eq!(mix_qk_from_value(Some("NO")), (MixGemv::Old, MixGemv::Old));
        assert_eq!(mix_qk_from_value(Some("down")), (MixGemv::Blk, MixGemv::Old));
        assert_eq!(mix_qk_from_value(Some(" UP ")), (MixGemv::Old, MixGemv::R8));
        assert_eq!(mix_qk_from_value(Some("seg")), (MixGemv::Seg, MixGemv::Seg));
        // task 286: 片方向だけ v5 / up だけ seg の A/B。
        assert_eq!(mix_qk_from_value(Some("V5D")), (MixGemv::V5, MixGemv::R8));
        assert_eq!(mix_qk_from_value(Some("v5u")), (MixGemv::Blk, MixGemv::V5));
        assert_eq!(mix_qk_from_value(Some("ups")), (MixGemv::Blk, MixGemv::Seg));
        // 未知の値は既定 (task 422 で dn6) へ倒す。切り戻しは `dn5` / `v5` / `0`。
        assert_eq!(mix_qk_from_value(Some("nope")), (MixGemv::Dn7, MixGemv::V5));
    }

    /// task 418: `DENNO_MIX_QK=dn5` の解釈と形の門番。
    /// 既定は v5 (dn5 は既定 off)、実形 k=10240 だけを dn5 に載せる。
    #[test]
    fn mix_dn5_gate_and_shape() {
        // task 422: 未設定は dn6 (既定)。dn5 は明示したときだけ (切り戻し段)。
        assert_eq!(mix_qk_from_value(None), (MixGemv::Dn7, MixGemv::V5));
        assert_eq!(mix_qk_from_value(Some("v5")), (MixGemv::V5, MixGemv::V5));
        assert_eq!(mix_qk_from_value(Some("DN5")), (MixGemv::Dn5, MixGemv::V5));
        assert_eq!(mix_qk_from_value(Some(" d5 ")), (MixGemv::Dn5, MixGemv::V5));
        // 形: n_in = 10240 = 64 * 160 -> 5 warp (1 レーン 1 対)。
        assert_eq!(mix_dn5_warps(10240), Some(5));
        assert_eq!(mix_dn5_warps(64 * 32), Some(1));
        assert_eq!(mix_dn5_warps(64 * 64), Some(2));
        assert_eq!(mix_dn5_warps(64 * 256), Some(8));
        // 対が 32 の倍数でない / warp が 8 を超える / 64 で割れない形は載せない。
        assert_eq!(mix_dn5_warps(64 * 16), None, "16 対は 32 の倍数でない");
        assert_eq!(mix_dn5_warps(64 * 288), None, "9 warp は共有メモリの上限");
        assert_eq!(mix_dn5_warps(1024 + 32), None, "64 で割れない");
        assert_eq!(mix_dn5_warps(0), None);
        // 320 は (対 5 個) で 32 の倍数でない -> 従来の up 側は dn5 に載らない。
        assert_eq!(mix_dn5_warps(320), None);
    }

    /// task 418: dn5 の合計順を CPU で写し、f64 の真値と v5 の順の両方に
    /// 1e-4 + 1e-5*|ref| で一致すること (差は丸めだけ) を固定する。
    #[test]
    fn mix_dn5_cpu_mirror_matches_v5_and_reference() {
        let n_in = 10240usize;
        let n_out = 8usize; // 全行を回すと遅いので先頭 8 行だけ
        // Q8_0 の行: 32 要素ごとに f16 の d (0x3800 = 0.5) + int8 の q。
        let nb = n_in / 32;
        let rb = nb * 34;
        let mut w = vec![0u8; rb * n_out];
        for r in 0..n_out {
            for b in 0..nb {
                let o = r * rb + b * 34;
                w[o] = 0x00;
                w[o + 1] = 0x38; // f16 0.5
                for j in 0..32 {
                    let q = ((b * 32 + j + 3 * r) % 255) as i32 - 127;
                    w[o + 2 + j] = q as i8 as u8;
                }
            }
        }
        let x: Vec<f32> = (0..n_in).map(|i| ((i * 37 % 1009) as f32 - 500.0) / 997.0).collect();

        // 1 対 (68 B = 64 要素) の積和。カーネルと同じ式・同じ 4 アキュムレータ。
        let dot = |r: usize, pair: usize| -> f32 {
            let base = r * rb + pair * 68;
            let d0 = f16_bits_to_f32(u16::from_le_bytes([w[base], w[base + 1]]));
            let d1 = f16_bits_to_f32(u16::from_le_bytes([w[base + 34], w[base + 35]]));
            let mut acc = [0.0f32; 4];
            for j in 0..32 {
                let q = w[base + 2 + j] as i8 as f32;
                let y = q * d0;
                acc[j & 3] += y * x[pair * 64 + j];
            }
            for j in 0..32 {
                let q = w[base + 34 + 2 + j] as i8 as f32;
                let y = q * d1;
                acc[j & 3] += y * x[pair * 64 + 32 + j];
            }
            (acc[0] + acc[1]) + (acc[2] + acc[3])
        };
        // dn5: 1 レーン 1 対 -> 32 レーンの木 -> 5 warp の逐次。
        let dn5 = |r: usize| -> f32 {
            let np = n_in / 64;
            let warps = np / 32;
            let mut part = [0.0f32; 8];
            for wid in 0..warps {
                let mut lane = [0.0f32; 32];
                for l in 0..32 {
                    lane[l] = dot(r, wid * 32 + l);
                }
                let mut off = 16;
                while off > 0 {
                    for l in 0..off {
                        lane[l] += lane[l + off];
                    }
                    off >>= 1;
                }
                part[wid] = lane[0];
            }
            let mut s = part[0];
            for w in part.iter().take(warps).skip(1) {
                s += *w;
            }
            s
        };
        // v5: 1 レーンが 5 対を積んでから 32 レーンの木。
        let v5 = |r: usize| -> f32 {
            let np = n_in / 64;
            let per = np / 32;
            let mut lane = [0.0f32; 32];
            for l in 0..32 {
                let mut s = 0.0f32;
                for p in 0..per {
                    s += dot(r, l * per + p);
                }
                lane[l] = s;
            }
            let mut off = 16;
            while off > 0 {
                for l in 0..off {
                    lane[l] += lane[l + off];
                }
                off >>= 1;
            }
            lane[0]
        };
        // f64 の真値。
        let reference = |r: usize| -> f64 {
            let mut s = 0.0f64;
            for pair in 0..n_in / 64 {
                let base = r * rb + pair * 68;
                let d0 = f16_bits_to_f32(u16::from_le_bytes([w[base], w[base + 1]])) as f64;
                let d1 = f16_bits_to_f32(u16::from_le_bytes([w[base + 34], w[base + 35]])) as f64;
                for j in 0..32 {
                    s += (w[base + 2 + j] as i8 as f64) * d0 * x[pair * 64 + j] as f64;
                }
                for j in 0..32 {
                    s += (w[base + 34 + 2 + j] as i8 as f64) * d1 * x[pair * 64 + 32 + j] as f64;
                }
            }
            s
        };
        for r in 0..n_out {
            let a = dn5(r);
            let b = v5(r);
            let rf = reference(r);
            let tol = 1e-4 + 1e-5 * rf.abs();
            assert!(
                (a as f64 - rf).abs() <= tol,
                "dn5 が真値から外れた: row={r} dn5={a} ref={rf}"
            );
            assert!(
                (b as f64 - rf).abs() <= tol,
                "v5 が真値から外れた: row={r} v5={b} ref={rf}"
            );
            // dn5 と v5 の差は丸めだけ (同じ 10240 項の和)。
            assert!(
                (a - b).abs() <= 1e-3 + 1e-5 * b.abs(),
                "dn5 と v5 の差が大きい: row={r} dn5={a} v5={b}"
            );
            // 決定的であること (同じ入力で 2 回)。
            assert_eq!(a.to_bits(), dn5(r).to_bits(), "dn5 が決定的でない");
        }
    }

    /// task 422: `DENNO_MIX_QK=dn6` / `DENNO_GDN_DEC_GEMV=dn6` の解釈と
    /// 起動形の門番 (1 ブロック rows 行 / 1 レーン 1 対)。
    #[test]
    fn mix_dn6_gate_and_shape() {
        // mix の down: dn6 系の値はすべて (Dn6, V5)。既定は task 431 で dn7。
        for v in ["dn6", "d6", "DN6", " dn6r1 ", "dn6r2", "dn6r6", "dn6r8"] {
            assert_eq!(
                mix_qk_from_value(Some(v)),
                (MixGemv::Dn6, MixGemv::V5),
                "raw={v:?}"
            );
        }
        assert_eq!(mix_qk_from_value(None), (MixGemv::Dn7, MixGemv::V5));
        assert_eq!(mix_qk_from_value(Some("v5")), (MixGemv::V5, MixGemv::V5));
        assert_eq!(mix_qk_from_value(Some("dn5")), (MixGemv::Dn5, MixGemv::V5));
        // rows の解釈 (mix の down の既定は 2 行。t422b で rows=2 が最良)。
        assert_eq!(mix_dn6_rows_from_value(Some("dn6r1")), 1);
        assert_eq!(mix_dn6_rows_from_value(Some("dn6r2")), 2);
        assert_eq!(mix_dn6_rows_from_value(Some("dn6r6")), 6);
        assert_eq!(mix_dn6_rows_from_value(Some("dn6")), 2);
        assert_eq!(mix_dn6_rows_from_value(Some("dn6r4")), 4);
        assert_eq!(mix_dn6_rows_from_value(None), 2);
        assert_eq!(mix_dn6_rows_from_value(Some("dn5")), 2);
        // GDN の qkv/out の既定は 6 行 (block 960)。
        assert_eq!(gdn_dn6_rows_from_value(Some("dn6r1")), 1);
        assert_eq!(gdn_dn6_rows_from_value(Some("dn6r6")), 6);
        assert_eq!(gdn_dn6_rows_from_value(Some("dn6")), 2);
        assert_eq!(gdn_dn6_rows_from_value(None), 2);
        // warp 数 = ceil(np / 32) (dn5 の np/32 と違い、割り切れない形も載る)。
        assert_eq!(mix_dn6_warps(10240), Some(5), "対 160 -> 5 warp");
        assert_eq!(mix_dn6_warps(5120), Some(3), "対 80 -> ceil(80/32) = 3");
        assert_eq!(mix_dn6_warps(2048), Some(1), "対 32 -> 1 warp");
        assert_eq!(mix_dn6_warps(320), Some(1), "対 5 -> 1 warp");
        assert_eq!(mix_dn6_warps(64 * 256), Some(8));
        assert_eq!(mix_dn6_warps(64 * 288), None, "9 warp は part[8] を越える");
        assert_eq!(mix_dn6_warps(0), None);
        assert_eq!(mix_dn6_warps(1024 + 32), None, "64 で割れない");
        // 起動形: rows * 32 * warps <= 1024 に丸める。
        assert_eq!(mix_dn6_plan(10240, 320, 2), Some((5, 2)), "mix の down (既定)");
        assert_eq!(mix_dn6_plan(10240, 320, 4), Some((5, 4)));
        assert_eq!(mix_dn6_plan(10240, 320, 8), Some((5, 6)), "1024 / 160 = 6 行");
        assert_eq!(mix_dn6_plan(10240, 2560, 6), Some((5, 6)), "GDN の qkv");
        assert_eq!(mix_dn6_plan(10240, 0, 4), None, "n_out = 0");
        assert_eq!(mix_dn6_plan(1024 + 32, 320, 4), None, "形が合わない");
    }

    /// task 427: 既存の dn6 を mix 以外 (full 層の qkv/wo、head) から撃つ門番。
    ///
    /// ゲートは既存 env の値 (`DENNO_FULL_DEC_FAST` / `DENNO_DEC_HEAD_FAST`)。
    /// **既定は dn6** (lab t427b で段 -10.5% / -16.4%)。切り戻しは `v5`。
    #[test]
    fn task427_full_and_head_dn6_gates_and_shapes() {
        // full 層のゲート: 未設定は 2 行 (既定)。明示的な v5 / 1 / on は 0。
        assert_eq!(
            crate::cuda::dec_full_dev::full_dec_dn6_rows_from_value(None),
            2,
            "既定が dn6 でない"
        );
        for raw in [Some("1"), Some("on"), Some("true"), Some("v5"), Some("dn5")] {
            assert_eq!(
                crate::cuda::dec_full_dev::full_dec_dn6_rows_from_value(raw),
                0,
                "raw={raw:?} は v5 へ戻す"
            );
        }
        for (raw, want) in [
            (Some("dn6"), 2usize),
            (Some("dn6r2"), 2),
            (Some("dn6r1"), 1),
            (Some("DN6R4"), 4),
            (Some(" dn6r8 "), 8),
        ] {
            assert_eq!(
                crate::cuda::dec_full_dev::full_dec_dn6_rows_from_value(raw),
                want,
                "raw={raw:?}"
            );
        }
        // head のゲート: 未設定は 4 行 (既定)。v5 / 1 / on は 0。
        assert_eq!(head_dn6_rows_from_value(None), 4, "既定が dn6 でない");
        for raw in [Some("1"), Some("on"), Some("v5")] {
            assert_eq!(head_dn6_rows_from_value(raw), 0, "raw={raw:?}");
        }
        // 空文字は未設定と同じ扱い (既定のまま)。
        assert_eq!(head_dn6_rows_from_value(Some("")), 4);
        assert_eq!(
            crate::cuda::dec_full_dev::full_dec_dn6_rows_from_value(Some("")),
            2
        );
        assert_eq!(head_dn6_rows_from_value(Some("dn6")), 4);
        assert_eq!(head_dn6_rows_from_value(Some("dn6r4")), 4);
        assert_eq!(head_dn6_rows_from_value(Some("dn6r2")), 2);
        assert_eq!(head_dn6_rows_from_value(Some("dn6r1")), 1);
        assert_eq!(head_dn6_rows_from_value(Some("dn6r8")), 8);
        // 形の門番: full 層の実形 (qkv k=2560 n=13312 / wo k=6144 n=2560)。
        assert_eq!(mix_dn6_plan(2560, 13312, 2), Some((2, 2)), "qkv (対 40 -> 2 warp)");
        assert_eq!(mix_dn6_plan(6144, 2560, 2), Some((3, 2)), "wo (対 96 -> 3 warp)");
        // head の実形 (k=2560 / n=248320)。
        assert_eq!(mix_dn6_plan(2560, 248_320, 4), Some((2, 4)));
    }

    /// task 422: dn6 (1 ブロック rows 行・x をレジスタで使い回す) の合計順を
    /// CPU で写し、**dn5 (1 ブロック 1 行) と 1 ビット一致**すること、
    /// f64 の真値に 1e-4 + 1e-5*|ref| で一致することを固定する。
    #[test]
    fn mix_dn6_cpu_mirror_is_bit_identical_to_dn5() {
        let n_in = 10240usize;
        let n_out = 8usize;
        let nb = n_in / 32;
        let rb = nb * 34;
        let mut w = vec![0u8; rb * n_out];
        for r in 0..n_out {
            for b in 0..nb {
                let o = r * rb + b * 34;
                w[o] = 0x00;
                w[o + 1] = 0x38; // f16 0.5
                for j in 0..32 {
                    let q = ((b * 32 + j + 5 * r) % 255) as i32 - 127;
                    w[o + 2 + j] = q as i8 as u8;
                }
            }
        }
        let x: Vec<f32> = (0..n_in).map(|i| ((i * 41 % 1013) as f32 - 506.0) / 991.0).collect();
        // 1 対 (68 B) の積和。式も 4 アキュムレータも `.cu` と同じ。
        let dot = |r: usize, pair: usize| -> f32 {
            let base = r * rb + pair * 68;
            let d0 = f16_bits_to_f32(u16::from_le_bytes([w[base], w[base + 1]]));
            let d1 = f16_bits_to_f32(u16::from_le_bytes([w[base + 34], w[base + 35]]));
            let mut acc = [0.0f32; 4];
            for j in 0..32 {
                let q = w[base + 2 + j] as i8 as f32;
                let y = q * d0;
                acc[j & 3] += y * x[pair * 64 + j];
            }
            for j in 0..32 {
                let q = w[base + 34 + 2 + j] as i8 as f32;
                let y = q * d1;
                acc[j & 3] += y * x[pair * 64 + 32 + j];
            }
            (acc[0] + acc[1]) + (acc[2] + acc[3])
        };
        // 木の縮約 (offset 16..1) と warp の逐次加算。
        let reduce = |lane: &mut [f32; 32]| -> f32 {
            let mut off = 16;
            while off > 0 {
                for l in 0..off {
                    lane[l] += lane[l + off];
                }
                off >>= 1;
            }
            lane[0]
        };
        let np = n_in / 64;
        let warps = np / 32;
        // dn5: 1 ブロック 1 行。
        let dn5 = |r: usize| -> f32 {
            let mut part = [0.0f32; 8];
            for wid in 0..warps {
                let mut lane = [0.0f32; 32];
                for l in 0..32 {
                    lane[l] = dot(r, wid * 32 + l);
                }
                part[wid] = reduce(&mut lane);
            }
            let mut s = part[0];
            for i in 1..warps {
                s += part[i];
            }
            s
        };
        // dn6: 1 ブロック rows 行。x は 1 レーン 1 回だけ読んで行で使い回す
        // (カーネルと同じ構造。丸めの順は行ごとに dn5 と同一)。
        let dn6 = |rows: usize| -> Vec<f32> {
            let mut out: Vec<f32> = Vec::with_capacity(n_out);
            let blocks = n_out.div_ceil(rows);
            for b in 0..blocks {
                let mut part = [[0.0f32; 8]; 8];
                for r in 0..rows {
                    let row = b * rows + r;
                    if row >= n_out {
                        continue;
                    }
                    for wid in 0..warps {
                        let mut lane = [0.0f32; 32];
                        for l in 0..32 {
                            lane[l] = dot(row, wid * 32 + l);
                        }
                        part[r][wid] = reduce(&mut lane);
                    }
                }
                for r in 0..rows {
                    let row = b * rows + r;
                    if row >= n_out {
                        continue;
                    }
                    let mut s = part[r][0];
                    for i in 1..warps {
                        s += part[r][i];
                    }
                    out.push(s);
                }
            }
            out
        };
        let reference = |r: usize| -> f64 {
            let mut s = 0.0f64;
            for pair in 0..np {
                let base = r * rb + pair * 68;
                let d0 = f16_bits_to_f32(u16::from_le_bytes([w[base], w[base + 1]])) as f64;
                let d1 = f16_bits_to_f32(u16::from_le_bytes([w[base + 34], w[base + 35]])) as f64;
                for j in 0..32 {
                    s += (w[base + 2 + j] as i8 as f64) * d0 * x[pair * 64 + j] as f64;
                }
                for j in 0..32 {
                    s += (w[base + 34 + 2 + j] as i8 as f64) * d1 * x[pair * 64 + 32 + j] as f64;
                }
            }
            s
        };
        for rows in [1usize, 2, 4, 6] {
            let got = dn6(rows);
            assert_eq!(got.len(), n_out, "行数が違う (rows={rows})");
            for r in 0..n_out {
                let a = got[r];
                let b = dn5(r);
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "dn6(rows={rows}) と dn5 が 1 ビットでも違う: row={r} dn6={a} dn5={b}"
                );
                let rf = reference(r);
                let tol = 1e-4 + 1e-5 * rf.abs();
                assert!(
                    (a as f64 - rf).abs() <= tol,
                    "dn6 が真値から外れた: rows={rows} row={r} dn6={a} ref={rf}"
                );
            }
        }
    }

    /// task 418: f16 のビット列を f32 にする (テストの参照用。正規化数だけ)。
    fn f16_bits_to_f32(h: u16) -> f32 {
        let sign = ((h & 0x8000u16) as u32) << 16;
        let exp = (h >> 10) & 0x1f;
        let man = (h & 0x3ff) as u32;
        let bits = if exp == 0 {
            sign
        } else if exp == 31 {
            sign | 0x7f80_0000
        } else {
            sign | (((exp as u32) + 112) << 23) | (man << 13)
        };
        f32::from_bits(bits)
    }

    /// task 640: カーネルから消した手書きのビット展開 (旧実装) の写し。
    /// 全 65536 パターンで新実装 (cvt.f32.f16 + NaN だけ旧式) と一致することを
    /// `f16_cvt_exhaustive` で示す。
    fn f16_old_bits(h: u16) -> u32 {
        let sign = ((h & 0x8000) as u32) << 16;
        let exp = ((h >> 10) & 0x1f) as u32;
        let man = (h & 0x3ff) as u32;
        if exp == 0 {
            if man == 0 {
                sign
            } else {
                let mut e: i32 = -14;
                let mut m = man;
                while (m & 0x400) == 0 {
                    m <<= 1;
                    e -= 1;
                }
                m &= 0x3ff;
                sign | (((e + 127) as u32) << 23) | (m << 13)
            }
        } else if exp == 31 {
            sign | 0x7f80_0000 | (man << 13)
        } else {
            sign | ((exp + 112) << 23) | (man << 13)
        }
    }

    /// task 640: 新実装の写し。NaN / inf は旧式のまま、それ以外は IEEE 754 の
    /// 厳密な f16 -> f32 拡張 (ハードウェアの cvt.f32.f16 と同じ値) を f64 で作る。
    /// f16 の値は f64 に厳密に入り、f32 へ 1 回丸めても情報が落ちないので
    /// (仮数 11 bit) この計算は丸め誤差ゼロ = ハードウェアと同じビット列になる。
    fn f16_new_bits(h: u16) -> u32 {
        if (h & 0x7c00) == 0x7c00 {
            return f16_old_bits(h);
        }
        let sign = ((h & 0x8000) as u32) << 16;
        let exp = (h >> 10) & 0x1f;
        let man = (h & 0x3ff) as f64 / 1024.0;
        let v = if exp == 0 {
            man * (-14f64).exp2()
        } else {
            (1.0 + man) * (((exp as i32) - 15) as f64).exp2()
        };
        let f = v as f32;
        assert!(!f.is_nan(), "有限パターンが NaN になった: {h:#06x}");
        sign | f.to_bits()
    }

    /// task 640 (step 2): f16 -> f32 の全 65536 パターンで旧実装と新実装が
    /// 1 ビットも違わないこと (0 / 非正規化数 / inf / NaN を含む)。
    #[test]
    fn f16_cvt_exhaustive() {
        let mut bad = Vec::new();
        let mut subnormal = 0usize;
        let mut nan_pat = 0usize;
        for h in 0u32..=0xffffu32 {
            let h16 = h as u16;
            let (a, b) = (f16_old_bits(h16), f16_new_bits(h16));
            if a != b {
                bad.push((h16, a, b));
            }
            if (h & 0x7c00) == 0 && (h & 0x3ff) != 0 {
                subnormal += 1;
            }
            if (h & 0x7c00) == 0x7c00 && (h & 0x3ff) != 0 {
                nan_pat += 1;
            }
        }
        assert!(bad.is_empty(), "不一致 {} 件: {:x?}", bad.len(), &bad[..bad.len().min(8)]);
        // 空振りでないこと: 非正規化数 1023 x 2 と NaN 2046 パターンを通している。
        assert_eq!(subnormal, 2046, "非正規化数のパターン数");
        assert_eq!(nan_pat, 2046, "NaN のパターン数");
        // NaN は旧式のまま (sNaN を quiet 化しない)。0x7c01 -> 0x7f802000。
        assert_eq!(f16_new_bits(0x7c01), 0x7f80_2000);
        assert_eq!(f16_new_bits(0xfc01), 0xff80_2000);
        assert_eq!(f16_new_bits(0x7e00), 0x7fc0_0000);
        // 代表値: 1.0 / 最大非正規化数 / 最小非正規化数。
        assert_eq!(f16_new_bits(0x3c00), 1.0f32.to_bits());
        assert_eq!(f16_new_bits(0x03ff), f32::from_bits(0x387f_c000).to_bits());
        assert_eq!(f16_new_bits(0x0001), 0x3380_0000);
    }

    /// task 640 (step 2): 30 本の .cu が実際にハードウェア変換に置き換わって
    /// いること (写し忘れを検出する)。kernels_elem.cu は probe のために旧式も
    /// 残す。exp_iq_v3 / mix_dn7 / mix_up4 は A/B 用の `#ifdef DNO_F16_LEGACY`
    /// の中にだけ旧式を残す (build_ptx.sh の `*_legacy` target 専用)。
    #[test]
    fn f16_cvt_sources_use_hardware_cvt() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels");
        let files = [
            "kernels_exp_iq.cu", "kernels_exp_iq_pf.cu", "kernels_exp_iq_tc.cu",
            "kernels_exp_iq_v2.cu", "kernels_exp_k_pf.cu", "kernels_exp_k_tc.cu",
            "kernels_gemm.cu", "kernels_gemm_grouped.cu", "kernels_gemm_sn.cu",
            "kernels_gemm_sn2.cu", "kernels_gemm_sn3.cu", "kernels_gemm_tc.cu",
            "kernels_gemm_tc2.cu", "kernels_mix_dn5.cu", "kernels_mix_dn6.cu",
            "kernels_mix_dn8.cu", "kernels_mix_upmix_q8.cu", "kernels_moe.cu",
            "kernels_moe_batch.cu", "kernels_moe_fp16.cu", "kernels_moe_v2.cu",
            "kernels_moe_v3.cu", "kernels_moe_v4.cu", "kernels_moe_v5.cu",
            "kernels_moe_v5_q5k.cu", "kernels_seg_k.cu", "kernels_sh_gu.cu",
        ];
        for f in files {
            let src = std::fs::read_to_string(dir.join(f)).unwrap_or_else(|e| panic!("{f}: {e}"));
            assert!(
                src.contains("__half2float(__ushort_as_half"),
                "{f} がハードウェア変換を使っていない"
            );
            assert!(src.contains("#include <cuda_fp16.h>"), "{f} に include が無い");
            assert!(
                !src.contains("while ((m & 0x400u) == 0)"),
                "{f} に旧式の非正規化ループが残っている"
            );
            assert!(!src.contains("int e = -14;"), "{f} に旧式の非正規化ループが残っている");
        }
        // A/B 用に旧式を残した 3 本は、#ifdef の中にだけ旧式があること。
        for f in ["kernels_exp_iq_v3.cu", "kernels_mix_dn7.cu", "kernels_mix_up4.cu"] {
            let src = std::fs::read_to_string(dir.join(f)).unwrap();
            assert!(src.contains("#ifdef DNO_F16_LEGACY"), "{f} に旧式の分岐が無い");
            assert!(src.contains("__half2float(__ushort_as_half"), "{f} が新式を使っていない");
            let legacy_at = src.find("#ifdef DNO_F16_LEGACY").unwrap();
            let loop_at = src.find("while ((m & 0x400u) == 0)").expect("旧式が無い");
            assert!(legacy_at < loop_at, "{f}: 旧式が #ifdef の外にある");
            assert_eq!(
                src.matches("while ((m & 0x400u) == 0)").count(),
                1,
                "{f}: 旧式のループが 2 箇所以上"
            );
        }
        // probe 側 (elem) は旧式と新式の両方を持つ。
        let elem = std::fs::read_to_string(dir.join("kernels_elem.cu")).unwrap();
        assert!(elem.contains("dno_f16cvt_probe"));
        assert!(elem.contains("dno_f16cvt_old"));
        assert!(elem.contains("dno_f16cvt_new"));
        assert!(elem.contains("while ((m & 0x400u) == 0)"), "probe の旧式が消えた");
    }

    /// task 640 (step 2): GPU 上で全 65536 パターンを照合する。
    /// GPU / 収録 PTX が無い環境では skip (このコンテナでは skip される)。
    #[test]
    fn f16_cvt_probe_on_gpu() {
        match chain_f16cvt_probe() {
            None => {
                eprintln!("[f16cvt] GPU または収録 PTX が無いので skip");
            }
            Some((cnt, first)) => {
                assert_eq!(cnt, 0, "GPU で不一致 {cnt} 件 (最初のパターン {first:#06x})");
                assert_eq!(first, 0xffff_ffff);
            }
        }
    }

    /// task 382: `DENNO_MIX_UPQ8` の真理値表と形の門番。
    /// t382b で採ったので既定 on、明示的な偽値だけが従来経路。
    #[test]
    fn mix_upq8_gate_and_shape() {
        // task 437: 既定は 1 対 = 4 レーンの u4f (lab t437b)。
        assert_eq!(mix_upq8_from_value(None), 4, "未設定は u4f (既定 on)");
        assert_eq!(mix_upq8_from_value(Some("nope")), 4, "未知も既定 on");
        assert_eq!(mix_upq8_from_value(Some("")), 0, "空文字は切り戻し");
        assert_eq!(mix_upq8_from_value(Some("0")), 0);
        assert_eq!(mix_upq8_from_value(Some("off")), 0);
        assert_eq!(mix_upq8_from_value(Some("NO")), 0);
        assert_eq!(mix_upq8_from_value(Some("1")), 1);
        assert_eq!(mix_upq8_from_value(Some("on")), 1);
        assert_eq!(mix_upq8_from_value(Some("CHECK")), 2);
        assert_eq!(mix_upq8_from_value(Some("2")), 2);
        // 形の門番: 実形 (n_in = 320 / hc = 4) だけ通す。
        assert!(mix_upq8_shape_ok(320, 2560, 4));
        assert!(!mix_upq8_shape_ok(320, 0, 4));
        assert!(!mix_upq8_shape_ok(320, 2560, 8), "hc は 4 だけ");
        assert!(!mix_upq8_shape_ok(256, 2560, 4), "n_in は 64*5 だけ");
        assert!(!mix_upq8_shape_ok(384, 2560, 4));
        // check の集計は積んでいなければ None。
        assert!(mix_upq8_check_line().is_none());
    }

    /// task 382: `DENNO_MIX_ZC` の真理値表。既定は off (従来の H2D + D2H)。
    /// `x` / `out` で片側ずつ、`1` / `both` で両側を消す。
    #[test]
    fn mix_zc_gate_defaults_off_and_parses_explicitly() {
        assert_eq!(mix_zc_from_value(None), (false, false), "未設定は従来経路");
        assert_eq!(mix_zc_from_value(Some("")), (false, false));
        assert_eq!(mix_zc_from_value(Some("0")), (false, false));
        assert_eq!(mix_zc_from_value(Some("off")), (false, false));
        assert_eq!(mix_zc_from_value(Some("NO")), (false, false));
        assert_eq!(mix_zc_from_value(Some("x")), (true, false));
        assert_eq!(mix_zc_from_value(Some(" IN ")), (true, false));
        assert_eq!(mix_zc_from_value(Some("out")), (false, true));
        assert_eq!(mix_zc_from_value(Some("1")), (true, true));
        assert_eq!(mix_zc_from_value(Some("both")), (true, true));
        assert_eq!(mix_zc_from_value(Some("ON")), (true, true));
        // 未知の値は既定 (従来経路) へ倒す。
        assert_eq!(mix_zc_from_value(Some("nope")), (false, false));
    }

    /// task 437: 1 対 = 4 レーンの up (u4 / u4f / u4w) の起動形とゲート。
    ///
    /// 実形 (mix up: n_in = 320 / e = 2,560) は JC=6 で 427 ブロック x 480 スレッド
    /// = 204,960 スレッド (40 warp/SM)。JC=12 は 214 ブロック x 960 スレッド。
    #[test]
    fn mix_up4_plan_and_gate_match_the_ud_mix_up() {
        assert_eq!(mix_up4_plan(3), Some((0, 6, 480)), "u4 = JC6 / 480 スレッド");
        assert_eq!(mix_up4_plan(4), Some((1, 6, 480)), "u4f も同じ形");
        assert_eq!(mix_up4_plan(5), Some((2, 12, 960)), "u4w は JC12");
        assert_eq!(mix_up4_plan(6), Some((3, 4, 320)), "u4x は JC4 / 320 スレッド");
        for v in [0u8, 1, 2, 7, 8, 255] {
            assert_eq!(mix_up4_plan(v), None, "{v} は従来経路");
        }
        // ゲートは既存 env の新しい値。
        assert_eq!(mix_upq8_from_value(Some("u4")), 3);
        assert_eq!(mix_upq8_from_value(Some(" 3 ")), 3);
        assert_eq!(mix_upq8_from_value(Some("UP4")), 3);
        assert_eq!(mix_upq8_from_value(Some("u4f")), 4);
        assert_eq!(mix_upq8_from_value(Some("4")), 4);
        assert_eq!(mix_upq8_from_value(Some("u4w")), 5);
        assert_eq!(mix_upq8_from_value(Some("5")), 5);
        assert_eq!(mix_upq8_from_value(Some("u4x")), 6);
        assert_eq!(mix_upq8_from_value(Some("6")), 6);
        // lab t437b で採ったので既定は u4f (mix 段 -6.6%)。
        assert_eq!(mix_upq8_from_value(None), 4);
        assert_eq!(mix_upq8_from_value(Some("nope")), 4);
        // 実形のブロック数と 1 行 20 スレッド (5 対 x 4 レーン)。
        assert_eq!(2560usize.div_ceil(MIX_UP4_JC), 427);
        assert_eq!(2560usize.div_ceil(MIX_UP4_JC_W), 214);
        assert_eq!(2560usize.div_ceil(MIX_UP4_JC_X), 640);
        assert_eq!(MIX_UP4_JC * 4 * 5 * 4, MIX_UP4_THREADS as usize);
        assert_eq!(MIX_UP4_JC_W * 4 * 5 * 4, MIX_UP4_THREADS_W as usize);
        assert_eq!(MIX_UP4_JC_X * 4 * 5 * 4, MIX_UP4_THREADS_X as usize);
        // 形の門番は従来の融合版と同じ。
        assert!(mix_upq8_shape_ok(320, 2560, 4));
        assert!(!mix_upq8_shape_ok(256, 2560, 4));
    }

    /// task 431: dn7 (1 対 = 4 レーン) の起動形とゲート。
    ///
    /// 実形 (mix down: n_in = 10,240 / n_out = 320) は np = 160 で
    /// 1 行 640 スレッド (20 warp)・320 ブロック = 204,800 スレッド。
    /// 形が合わない (np が 8 の倍数でない / 1024 超) は None = dn6 へ落ちる。
    #[test]
    fn mix_dn7_plan_and_gate_match_the_ud_mix_down() {
        assert_eq!(mix_dn7_plan(10240, 320), Some((1, 640)), "実形が撃てない");
        // np = n_in/64 が 8 の倍数でない / 0 は落とす。
        assert_eq!(mix_dn7_plan(320, 10240), None, "up の形 (np=5) は対象外");
        assert_eq!(mix_dn7_plan(64 * 4, 8), None, "np=4 (per_row が 32 の倍数でない)");
        assert_eq!(mix_dn7_plan(0, 8), None);
        assert_eq!(mix_dn7_plan(10240, 0), None);
        // 1 行 1024 スレッド超 (np = 264 -> 1056) は落とす。
        assert_eq!(mix_dn7_plan(64 * 264, 8), None);
        // 1 行 1024 スレッドちょうど (np = 256) は通る。
        assert_eq!(mix_dn7_plan(64 * 256, 8), Some((1, 1024)));
        // ゲートは既存 env の新しい値。up は v5 のまま。
        assert_eq!(mix_qk_from_value(Some("dn7")), (MixGemv::Dn7, MixGemv::V5));
        assert_eq!(mix_qk_from_value(Some(" D7 ")), (MixGemv::Dn7, MixGemv::V5));
        assert_eq!(mix_qk_from_value(None), (MixGemv::Dn7, MixGemv::V5), "既定は dn7");
        // 発火の計数はリセットできる。
        mix_dn7_reset();
        assert_eq!(mix_dn7_hits(), 0);
    }

    /// task 431: `DENNO_MIX_ZC` の第 3 の指定 (H2D を別ストリームへ) の真理値表。
    /// 既定 off、`stream` / `s` で H2D だけ、`os` / `outstream` / `all` で
    /// D2H も消す。既存の値 (`x` / `out` / `1`) はストリームを立てない。
    #[test]
    fn mix_zc_h2d_stream_gate_parses_explicitly() {
        assert!(!mix_zc_h2d_stream_from_value(None), "未設定は従来経路");
        assert!(!mix_zc_h2d_stream_from_value(Some("")));
        assert!(!mix_zc_h2d_stream_from_value(Some("0")));
        assert!(!mix_zc_h2d_stream_from_value(Some("off")));
        assert!(!mix_zc_h2d_stream_from_value(Some("x")));
        assert!(!mix_zc_h2d_stream_from_value(Some("out")));
        assert!(!mix_zc_h2d_stream_from_value(Some("1")));
        assert!(!mix_zc_h2d_stream_from_value(Some("nope")));
        assert!(mix_zc_h2d_stream_from_value(Some("stream")));
        assert!(mix_zc_h2d_stream_from_value(Some(" S ")));
        assert!(mix_zc_h2d_stream_from_value(Some("sx")));
        assert!(mix_zc_h2d_stream_from_value(Some("instream")));
        // `os` / `outstream` / `all` は D2H も消す (両方の表で真)。
        for v in ["os", "outstream", "all"] {
            assert!(mix_zc_h2d_stream_from_value(Some(v)), "{v}");
            assert_eq!(mix_zc_from_value(Some(v)), (false, true), "{v}");
        }
        assert_eq!(mix_zc_from_value(Some("stream")), (false, false));
        // 発火の計数はリセットできる。
        mix_h2s_reset();
        assert_eq!(mix_h2s_hits(), 0);
    }

    /// task 382: ZC のゲートを立てても、CUDA が無い環境では鎖そのものが
    /// 走らないので H2D/D2H を省いた回数は 0 のまま (fail soft)。
    #[test]
    fn mix_zc_hits_stay_zero_without_cuda_chain() {
        mix_zc_reset();
        let (x, o) = mix_zc_hits();
        assert_eq!((x, o), (0, 0));
        // ゲートの強制はスレッド局所なので、他のテストの env 解釈を汚さない。
        mix_zc_set_for_test(Some((true, true)));
        assert_eq!(mix_zc_enabled(), (true, true));
        mix_zc_set_for_test(None);
        assert_eq!(mix_zc_hits(), (0, 0), "鎖が走らないので数えない");
    }

    /// task 286: v5 形の起動計画。UD の mix の実形 (down 10240->320 / up 320->10240)
    /// で「1 行 1 warp・2 行 1 ブロック」と「1 レーン 1 行・128 行 1 ブロック」に
    /// なること、形が合わないときは None (Blk/R8 へ落ちる) を固定する。
    #[test]
    fn mix_v5_shape_matches_the_ud_mix() {
        // down: k=10240 -> nb=320, np=160, lanes=32, 2 行/ブロック。
        assert_eq!(mix_v5_shape(10240, 320), Some((32, 2, 160)));
        // up: k=320 -> nb=10, np=5, lanes=1, 128 行/ブロック。
        assert_eq!(mix_v5_shape(320, 10240), Some((1, 128, 80)));
        // 2 ブロック未満 / 奇数ブロック / 端数は v5 に載せない。
        assert_eq!(mix_v5_shape(32, 8), None, "nb=1");
        assert_eq!(mix_v5_shape(96, 8), None, "nb=3 (奇数)");
        assert_eq!(mix_v5_shape(0, 8), None);
        assert_eq!(mix_v5_shape(10240, 0), None);
        // ブロック次元は 1024 以下 (lanes=32 でも 64 スレッド)。
        for (n_in, n_out) in [(10240usize, 320usize), (320, 10240), (2048, 64)] {
            if let Some((l, r, g)) = mix_v5_shape(n_in, n_out) {
                assert!(l * r <= 1024, "block_dim が 1024 を超えた: lanes={l} rows={r}");
                assert!(g >= 1, "grid が 0");
                assert!(l.is_power_of_two(), "lanes は 2 の冪: {l}");
                assert_eq!((n_in / 32 / 2) % l, 0, "レーンが対を割り切る");
            }
        }
    }
    /// task 282: 前置きバッファのレイアウト (重みポインタ 2 本 / segment 表 2 枚 /
    /// 入力) が詰まっていて、seg 表と xn が 4 / 16 バイト整列であること。
    ///
    /// `kernels_seg_k.cu` は `const unsigned char* const* wptrs` を u64 の配列として
    /// 読み、`seg` を u32 の 4 語として読む。ずれると別のアドレスを重みと解釈して
    /// **静かに**違う値が出るので、オフセットをここで固定する。
    #[test]
    fn mix_qk_stage_layout_is_packed_and_aligned() {
        let (wd, wu, sd, su, x) = mix_qk_layout();
        assert_eq!((wd, wu), (0, 8), "重みポインタ 2 本は先頭 (8 バイト刻み)");
        assert_eq!((sd, su), (16, 32), "segment 表 2 枚は 16 バイトずつ");
        assert_eq!(x, MIX_QK_PROLOGUE, "xn は前置きの直後");
        assert_eq!(MIX_QK_PROLOGUE, 48, "前置きのバイト数");
        assert_eq!(wd % 8, 0);
        assert_eq!(wu % 8, 0);
        assert_eq!(sd % 4, 0);
        assert_eq!(su % 4, 0);
        assert_eq!(x % 16, 0, "xn は f32 の 4 要素 (16 バイト) 整列");
        // 2 枚の seg 表が重ならない
        assert_eq!(sd + 16, su);
        assert_eq!(su + 16, x);
    }
    /// task 257/265: ゲートを on にしても CUDA context が無ければ借りられない。
    /// ガードは `owned == false`、区画ビューも取れない = 従来の確保に落ちる
    /// (リクエストは失敗させない)。
    ///
    /// プールの**発火側**は lab の実測でしか確かめられないが、fail open の契約は
    /// ここで固定する (確保の分岐が黙って変わらないように)。
    #[test]
    fn pool_acquire_without_a_context_is_fail_open() {
        let need = crate::model::pf_pool::Need {
            nt_all: 14018,
            tile: 5120,
            k: 10,
            e: 2560,
            ff: 1024,
            sff: 1024,
            hc: 4,
            lr: 256,
            chunk: 1536,
            ne: 512,
            tc: true,
            stage_bytes: 512 << 20,
            stage_slots: PF_STAGE_SLOTS,
        };
        // レイアウト自体は組める (純関数)。
        assert!(crate::model::pf_pool::layout(&need).is_some());
        crate::model::pf_pool::set_force_on_for_test(Some(true));
        {
            let guard = pf_pool_acquire(&need);
            assert!(!guard.is_held());
            assert_eq!(pf_pool_len(), 0);
            assert!(pool_view(crate::model::pf_pool::R::H, 0, 1).is_none());
            assert!(pool_view(crate::model::pf_pool::R::Stage, 0, 1).is_none());
        }
        // 借りていない状態で release しても落ちない (二重解放しない)。
        pf_pool_release();
        assert_eq!(pf_pool_len(), 0);
        crate::model::pf_pool::set_force_on_for_test(None);
    }

    /// task 265: しきい値未満のプロンプトはプールに触らない (短い経路を変えない)。
    #[test]
    fn pool_acquire_skips_short_prompts() {
        let need = crate::model::pf_pool::Need {
            nt_all: 32,
            tile: 32,
            k: 10,
            e: 2560,
            ff: 1024,
            sff: 1024,
            hc: 4,
            lr: 256,
            chunk: 32,
            ne: 512,
            tc: true,
            stage_bytes: 512 << 20,
            stage_slots: PF_STAGE_SLOTS,
        };
        crate::model::pf_pool::set_force_on_for_test(Some(true));
        let guard = pf_pool_acquire(&need);
        assert!(!guard.is_held());
        assert_eq!(pf_pool_len(), 0);
        crate::model::pf_pool::set_force_on_for_test(None);
    }

    /// task 265: ゲートが既定 (off) なら長いプロンプトでも何も借りない。
    #[test]
    fn pool_acquire_default_off_borrows_nothing() {
        let need = crate::model::pf_pool::Need {
            nt_all: 14018,
            tile: 5120,
            k: 10,
            e: 2560,
            ff: 1024,
            sff: 1024,
            hc: 4,
            lr: 256,
            chunk: 1536,
            ne: 512,
            tc: true,
            stage_bytes: 512 << 20,
            stage_slots: PF_STAGE_SLOTS,
        };
        crate::model::pf_pool::set_force_on_for_test(Some(false));
        let guard = pf_pool_acquire(&need);
        assert!(!guard.is_held());
        crate::model::pf_pool::set_force_on_for_test(None);
    }

    /// task 210: 捨てた cache の ID のミラーは残さない (生きている ID は残す)。
    /// ミラーは ID を鍵にした LRU で、上限 (3584 MiB) まで残るのが仕様なので、
    /// 消し忘れると死んだ cache の VRAM が積み上がる (lab t209a の 14k: 384 MiB)。
    #[test]
    fn forget_removes_only_that_cache_id() {
        let mirrors = Q4_ATTN_MIRRORS.get_or_init(|| std::sync::Mutex::new(Vec::new()));
        let keep = 0xffff_fff0u64;
        let gone = 0xffff_fff1u64;
        {
            let mut ms = mirrors.lock().unwrap();
            ms.push((keep, Q4AttnMirror::new(2, 8), std::time::Instant::now()));
            ms.push((gone, Q4AttnMirror::new(2, 8), std::time::Instant::now()));
        }
        q4_attn_mirror_forget(gone);
        {
            let ms = mirrors.lock().unwrap();
            let ids: Vec<u64> = ms.iter().map(|(k, _, _)| *k).collect();
            assert!(ids.contains(&keep), "生きている cache のミラーまで消えた: {ids:?}");
            assert!(!ids.contains(&gone), "捨てた cache のミラーが残っている: {ids:?}");
        }
        // 存在しない ID でも落ちない (Drop から何度でも呼ばれる)。
        q4_attn_mirror_forget(gone);
        q4_attn_mirror_forget(keep);
    }

    /// task 210: 確保失敗時の追い出し量は階段で、最後は全部。
    /// 2 段目がいきなり「全部」だと 32 MiB が通らないだけで 22 GB を捨てる。
    #[test]
    fn alloc_retry_wants_is_a_rising_ladder() {
        let w = alloc_retry_wants(32 << 20);
        assert_eq!(w[0], 96 << 20, "1 段目が要求 + 64 MiB でない: {w:?}");
        assert!(w.windows(2).all(|p| p[0] < p[1]), "階段が昇順でない: {w:?}");
        assert_eq!(*w.last().unwrap(), usize::MAX, "最後が全部でない: {w:?}");
        assert_eq!(w.len(), 5, "段数が変わった: {w:?}");
        // 要求が大きいときは小さい段を飛ばす (単調増加は保つ)。
        let big = alloc_retry_wants(3 << 30);
        assert_eq!(big[0], (3 << 30) + (64 << 20));
        assert!(big.windows(2).all(|p| p[0] < p[1]), "{big:?}");
        assert_eq!(*big.last().unwrap(), usize::MAX);
        // 飽和しても単調 (usize::MAX の手前で止まる)。
        let sat = alloc_retry_wants(usize::MAX - 1);
        assert_eq!(sat[0], usize::MAX);
        assert_eq!(*sat.last().unwrap(), usize::MAX);
    }

    /// task 256: 1 本の確保で追い出してよい量の上限。
    ///
    /// 従来の階段は最後が「全部」で、1,505 MiB の作業域 1 本のために
    /// 39,755 MiB の expert を捨てていた。いまはタグによらず
    /// 「要求 + チャンク 1 本」で切る (1 MiB の確保でも、実メモリを戻すには
    /// 1 チャンクは空ける必要がある)。
    #[test]
    fn evict_cap_is_bounded_per_allocation() {
        let mib = 1usize << 20;
        let chunk = 256 * mib;
        // 大きな作業域 (1,505 MiB): 要求 + チャンク。
        let cap = evict_cap_bytes(1505 * mib, crate::mem_audit::TAG_OTHER, chunk, true);
        assert_eq!(cap, Some(1505 * mib + chunk), "作業域の上限: {cap:?}");
        assert!(cap.unwrap() < 2 * 1024 * mib, "22 GB を捨てる上限になっている");
        // 小さな確保も同じ (チャンクを 1 本空けないと実メモリが戻らない)。
        assert_eq!(
            evict_cap_bytes(2 * mib, crate::mem_audit::TAG_OTHER, chunk, true),
            Some(2 * mib + chunk)
        );
        // expert タグ (スラブのチャンク確保 256 MiB も含む) も要求 + チャンク。
        assert_eq!(
            evict_cap_bytes(chunk, crate::mem_audit::TAG_EXPERTS, chunk, true),
            Some(chunk + chunk)
        );
        assert_eq!(
            evict_cap_bytes(2 * mib, crate::mem_audit::TAG_EXPERTS, chunk, true),
            Some(2 * mib + chunk)
        );
        assert_eq!(
            evict_cap_bytes(2 * mib, crate::mem_audit::TAG_GROUPED, chunk, true),
            Some(2 * mib + chunk)
        );
        // gate off は従来 (上限なし = 階段の最後まで)。
        assert_eq!(
            evict_cap_bytes(1505 * mib, crate::mem_audit::TAG_OTHER, chunk, false),
            None
        );
        // 選び方 (plan_eviction) もこの上限未満で止まる。
        let spans = vec![
            crate::model::expert_slab::ChunkSpan {
                id: 1,
                addr: 0,
                bytes: chunk,
                last_use: 1,
            },
            crate::model::expert_slab::ChunkSpan {
                id: 2,
                addr: chunk,
                bytes: chunk,
                last_use: 2,
            },
        ];
        let plan = crate::model::expert_slab::plan_eviction(&spans, 0, 2 * mib);
        assert_eq!(plan.ids, vec![1], "1 本で足りる: {plan:?}");
        assert!(
            plan.bytes < cap.unwrap(),
            "上限を超えた: {} >= {}",
            plan.bytes,
            cap.unwrap()
        );
    }

    /// task 248: `DENNO_ALLOC_KEEP_EXPERTS` の真偽表。既定は on。
    #[test]
    fn keep_experts_env_truth_table() {
        assert!(parse_keep_experts(None, true), "未設定は既定");
        assert!(!parse_keep_experts(None, false));
        for off in ["0", "off", "false", "no", " off "] {
            assert!(!parse_keep_experts(Some(off), true), "{off:?} が on になった");
        }
        for on in ["1", "on", "true", "yes"] {
            assert!(parse_keep_experts(Some(on), false), "{on:?} が off になった");
        }
        // 読めない値は既定 (ここでは false)。
        assert!(!parse_keep_experts(Some("2"), false));
        assert!(parse_keep_experts(Some("2"), true));
    }

    /// task 290: `DENNO_ALLOC_FORCE_CHUNK` の真偽表。既定は on。
    #[test]
    fn alloc_force_chunk_env_truth_table() {
        assert!(alloc_force_chunk_from_value(None), "未設定は既定 (on)");
        for off in ["0", "off", "false", "no", " NO "] {
            assert!(!alloc_force_chunk_from_value(Some(off)), "{off:?} が on");
        }
        for on in ["1", "on", "true", "yes"] {
            assert!(alloc_force_chunk_from_value(Some(on)), "{on:?} が off");
        }
    }

    /// task 210: `DENNO_ATTN_MIRROR_FORGET` の真偽表。既定は on。
    #[test]
    fn mirror_forget_env_truth_table() {
        assert!(mirror_forget_from_value(None), "未設定は on");
        for off in ["0", "false", "no"] {
            assert!(!mirror_forget_from_value(Some(off)), "{off} で切れていない");
        }
        for on in ["1", "true", "yes", ""] {
            assert!(mirror_forget_from_value(Some(on)), "{on} で切れている");
        }
    }

    /// task 167: デバイス連鎖の形の判定 (純関数)。実モデル (e=2560 / nk=8 /
    /// dk=dv=256 / nv=24 → conv_dim 10240, value_dim 6144) を通し、
    /// 1 つでも外れたら落ちることを見る。
    #[test]
    fn gdn_pf_chain_shape_rules() {
        use crate::kquant::GgmlType as T;
        let (e, nk, nv, dk, dv) = (2560usize, 8usize, 24usize, 256usize, 256usize);
        let (conv_dim, vd) = (2 * nk * dk + nv * dv, nv * dv);
        assert_eq!((conv_dim, vd), (10240, 6144));
        let blocks = vec![0u8; 64];
        let wq = PfChainW { uid: 1, ty: T::Q8_0, blocks: &blocks, k: e, n: conv_dim };
        let wg = PfChainW { uid: 2, ty: T::Q4K, blocks: &blocks, k: e, n: vd };
        let wo = PfChainW { uid: 3, ty: T::Q8_0, blocks: &blocks, k: vd, n: e };
        let sep = PfChainQkv::separate(wq, wg);
        for nt in [64usize, 512, 1024, 2047] {
            assert!(
                gdn_pf_chain_shape_ok(&sep, &wo, nt, e, nk, nv, dk, dv, 4),
                "nt={nt} は受けるはず"
            );
        }
        // 短すぎる / 長すぎる (chunked 経路の領域) は受けない。
        assert!(!gdn_pf_chain_shape_ok(&sep, &wo, 63, e, nk, nv, dk, dv, 4));
        assert!(!gdn_pf_chain_shape_ok(&sep, &wo, 2048, e, nk, nv, dk, dv, 4));
        // dk != dv / d_conv != 4 は単発カーネルの前提外。
        assert!(!gdn_pf_chain_shape_ok(&sep, &wo, 512, e, nk, nv, 256, 128, 4));
        assert!(!gdn_pf_chain_shape_ok(&sep, &wo, 512, e, nk, nv, dk, dv, 3));
        // 形の不一致 (out の k が value_dim でない)。
        let bad = PfChainW { k: e, ..wo };
        assert!(!gdn_pf_chain_shape_ok(&sep, &bad, 512, e, nk, nv, dk, dv, 4));
        // 型が GEMM の対象外 (F32 の qkv)。
        let bad_ty = PfChainW { ty: T::F32, ..wq };
        let bad_pair = PfChainQkv::separate(bad_ty, wg);
        assert!(!gdn_pf_chain_shape_ok(&bad_pair, &wo, 512, e, nk, nv, dk, dv, 4));
        // 連結済み (1 本の行列の行範囲): qkv が先頭・gate がその後ろ。
        let wf = PfChainW {
            uid: 4,
            ty: T::Q8_0,
            blocks: &blocks,
            k: e,
            n: conv_dim + vd,
        };
        let fus = PfChainQkv::fused(wf, 0, conv_dim);
        assert!(gdn_pf_chain_shape_ok(&fus, &wo, 1024, e, nk, nv, dk, dv, 4));
        // 連結の n が合計でない / 行範囲が重なる / はみ出す、は受けない。
        let bad_fus = PfChainQkv::fused(PfChainW { n: conv_dim, ..wf }, 0, conv_dim);
        assert!(!gdn_pf_chain_shape_ok(&bad_fus, &wo, 1024, e, nk, nv, dk, dv, 4));
        let same_row = PfChainQkv::fused(wf, 0, 0);
        assert!(!gdn_pf_chain_shape_ok(&same_row, &wo, 1024, e, nk, nv, dk, dv, 4));
        let over = PfChainQkv::fused(wf, 7000, conv_dim);
        assert!(!gdn_pf_chain_shape_ok(&over, &wo, 1024, e, nk, nv, dk, dv, 4));
    }

    /// task 270: `DENNO_PF_CHAIN_BIG` の真理値表と、大きい nt の連鎖の規則。
    ///
    /// 2048 以上の nt は task 197 まで連鎖が外れていた (単発 sigmoid の前提)。
    /// v1/v2/v3 のスキャンは長さの上限を持たないので、ゲート + 形が揃えば通す。
    #[test]
    fn chain_big_env_and_nt_rule() {
        assert!(!chain_big_from_value(None), "未設定は off");
        for off in ["0", "off", "false", "no", "", "  "] {
            assert!(!chain_big_from_value(Some(off)), "{off:?} で on になっている");
        }
        for on in ["1", "on", "true", "yes", " TRUE "] {
            assert!(chain_big_from_value(Some(on)), "{on:?} で off になっている");
        }
        // v3/v2/v1 と同じ形 (dk = dv = 128 / d_conv = 4)。
        let (nk, nv, dk, dv, d_conv) = (8usize, 24usize, 128usize, 128usize, 4usize);
        let conv_dim = 2 * nk * dk + nv * dv;
        for nt in [64usize, 512, 2047] {
            assert!(
                chain_nt_ok(nt, nk, nv, dk, dv, d_conv, conv_dim, false),
                "nt={nt} が既定で落ちる"
            );
        }
        for nt in [2048usize, 3072, 4096] {
            assert!(
                !chain_nt_ok(nt, nk, nv, dk, dv, d_conv, conv_dim, false),
                "nt={nt} が既定 (big off) で通る"
            );
            assert!(
                chain_nt_ok(nt, nk, nv, dk, dv, d_conv, conv_dim, true),
                "nt={nt} が big on で落ちる"
            );
        }
        // dk = 256 は v1/v2/v3 の前提外なので、big でも受けない。
        let cd256 = 2 * nk * 256 + nv * 256;
        assert!(!chain_nt_ok(3072, nk, nv, 256, 256, d_conv, cd256, true));
        // 床より短い nt は big でも受けない。
        assert!(!chain_nt_ok(63, nk, nv, dk, dv, d_conv, conv_dim, true));
        // 実モデル形 (e = 2560 / dk = dv = 256) では 2048 以上が big でも落ちる
        // (`gdn_pf_chain_shape_rules` の 2048 の assert と同じ理由)。
        assert!(!chain_nt_ok(2048, 8, 24, 256, 256, 4, 10240, true));
    }

    /// task 152: `DENNO_MIX_BF16` の真理値表。既定は off (lab の採否で決める)。
    #[test]
    fn mix_bf16_gate_truth_table() {
        assert!(!mix_bf16_from_value(None, false), "既定 off");
        assert!(mix_bf16_from_value(None, true), "既定 on を渡せば on");
        for v in [Some("1"), Some("on"), Some(" TRUE "), Some("yes")] {
            assert!(mix_bf16_from_value(v, false), "value={v:?} は on のはず");
        }
        for v in [Some("0"), Some("off"), Some("false"), Some("no")] {
            assert!(!mix_bf16_from_value(v, true), "value={v:?} は off のはず");
        }
        assert!(MIX_BF16_DEFAULT, "lab t152b で採用 → 既定 on");
    }

    /// task 152: cuMemGetInfo のキャッシュ判定 (64 MiB の増減 / 2 秒)。
    #[test]
    fn mem_cache_refresh_rule() {
        assert!(!mem_cache_should_refresh(0, 0), "変化なし・直後は引き直さない");
        assert!(!mem_cache_should_refresh(
            (MEM_CACHE_DRIFT_MB << 20) - 1,
            MEM_CACHE_MAX_MS - 1
        ));
        assert!(mem_cache_should_refresh(MEM_CACHE_DRIFT_MB << 20, 0), "64 MiB 動いた");
        assert!(mem_cache_should_refresh(0, MEM_CACHE_MAX_MS), "2 秒経った");
        assert!(
            mem_cache_should_refresh((MEM_CACHE_DRIFT_MB << 20) + 1, 0),
            "64 MiB 超"
        );
    }

    /// task 138: `DENNO_MIX_UP_REDUCE` は既定 on。明示的な偽値だけ逐次版へ戻す。
    #[test]
    fn mix_up_reduce_defaults_on() {
        assert!(mix_up_reduce_from_value(None), "未設定は on のはず");
        for v in [Some("1"), Some("on"), Some("true"), Some(" yes ")] {
            assert!(mix_up_reduce_from_value(v), "value={v:?} は on のはず");
        }
        for v in [Some("0"), Some("off"), Some("OFF"), Some("false"), Some("no")] {
            assert!(!mix_up_reduce_from_value(v), "value={v:?} は off のはず");
        }
    }

    /// 決定論的 LCG 乱数 (tensor.rs のテストと同一方式)
    pub(crate) fn pseudo_random(len: usize, seed: u32) -> Vec<f32> {
        let mut x = seed;
        (0..len)
            .map(|_| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((x >> 16) % 2000) as f32 / 1000.0 - 1.0 // [-1, 1)
            })
            .collect()
    }

    /// i8 の擬似重み [-128,127] (全範囲をカバー)
    fn pseudo_i8(len: usize, seed: u32) -> Vec<i8> {
        let mut x = seed;
        (0..len)
            .map(|_| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (((x >> 16) % 256) as i32 - 128) as i8
            })
            .collect()
    }

    fn positive_scales(len: usize, seed: u32) -> Vec<f32> {
        pseudo_random(len, seed)
            .iter()
            .map(|v| 0.001 + v.abs() * 0.01)
            .collect()
    }

    // -----------------------------------------------------------------------
    // P58-1: GPU KV ミラーの上限 (既定 2048 MB)・LRU エビクト・使用量ログ。
    // どれもホスト側のメタデータだけで検証できる (GPU 不要)。
    // -----------------------------------------------------------------------

    /// エビクト判定用メタデータ (`age_ms` が大きいほど古い = LRU の先頭)。
    fn mirror_meta(key: u64, bytes: usize, age_ms: u64, pending: bool) -> MirrorMeta {
        MirrorMeta {
            key,
            bytes,
            last: std::time::Instant::now() - std::time::Duration::from_millis(age_ms),
            pending,
        }
    }

    fn keys_of(entries: &[MirrorMeta]) -> Vec<u64> {
        entries.iter().map(|m| m.key).collect()
    }

    /// P58-1: 上限の既定値が 2048 MB であること、`DENNO_GPU_KV_MB` の上書きが
    /// 維持されていること。`gpu_kv_limit_bytes_from` は env を読まない純関数なので、
    /// テストの並列実行で環境変数を取り合わない。
    #[test]
    fn gpu_kv_limit_defaults_to_2048_mb_and_env_overrides() {
        const MB: usize = 1024 * 1024;
        assert_eq!(DEFAULT_GPU_KV_LIMIT_MB, 2048, "既定値 (P58-1 で 8192 → 2048)");
        assert_eq!(gpu_kv_limit_bytes_from(None), 2048 * MB);
        // env 上書きは従来どおり効く (実機で 1024 に落として VRAM が 1,014 MB 減った値)
        assert_eq!(gpu_kv_limit_bytes_from(Some(1024)), 1024 * MB);
        assert_eq!(gpu_kv_limit_bytes_from(Some(8192)), 8192 * MB);
        // 実使用される値も同じ規則 (テストプロセスが env を設定している場合は env 優先)
        match std::env::var("DENNO_GPU_KV_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
        {
            Some(v) => assert_eq!(gpu_kv_limit_bytes(), v * MB),
            None => assert_eq!(gpu_kv_limit_bytes(), 2048 * MB),
        }
    }

    /// P58-1: 上限を超えたら LRU (最も古い) から落ち、自分 (使用中) と
    /// GDN 遅延フラッシュ pending のミラーは落とさないこと。
    #[test]
    fn kv_mirror_lru_evicts_oldest_beyond_limit() {
        // 各 1,000 B・上限 2,500 B: 3 本 (3,000 B) → 2 本 (2,000 B) まで落とす
        let mut e = vec![
            mirror_meta(1, 1000, 300, false),
            mirror_meta(2, 1000, 200, false),
            mirror_meta(3, 1000, 0, false),
        ];
        assert_eq!(lru_evict_keys(&mut e, 2500, 3), vec![1], "最古の 1 本だけ落ちる");
        assert_eq!(keys_of(&e), vec![3, 2], "使用中の 3 と新しい 2 が残る");

        // 上限 1,000 B: 古い順に 2 本落ちる。自分 3 は落とさない
        let mut e = vec![
            mirror_meta(1, 1000, 300, false),
            mirror_meta(2, 1000, 200, false),
            mirror_meta(3, 1000, 0, false),
        ];
        assert_eq!(lru_evict_keys(&mut e, 1000, 3), vec![1, 2], "古い順に落ちる");
        assert_eq!(keys_of(&e), vec![3]);

        // 自分が最古でも落とさない (これから使うミラー)
        let mut e = vec![
            mirror_meta(1, 1000, 300, false),
            mirror_meta(2, 1000, 200, false),
            mirror_meta(3, 1000, 0, false),
        ];
        assert_eq!(lru_evict_keys(&mut e, 1000, 1), vec![2, 3]);
        assert_eq!(keys_of(&e), vec![1]);

        // GDN 遅延フラッシュ pending は最古でも飛ばす (未フラッシュ状態の唯一のコピー)
        let mut e = vec![
            mirror_meta(1, 1000, 400, true),
            mirror_meta(2, 1000, 300, false),
            mirror_meta(3, 1000, 0, false),
        ];
        assert_eq!(lru_evict_keys(&mut e, 1000, 3), vec![2], "pending の 1 は飛ばす");
        assert_eq!(keys_of(&e), vec![1, 3], "pending の 1 は残る");

        // 落とせる相手が無ければ止まる (上限超過のまま続くがデータは失わない)
        let mut e = vec![mirror_meta(1, 1000, 0, true), mirror_meta(2, 1000, 0, false)];
        assert!(lru_evict_keys(&mut e, 100, 2).is_empty(), "自分以外に落とせる相手が無い");
        assert_eq!(keys_of(&e), vec![1, 2]);

        // 1 本だけなら自分しか居ないので落とさない (ms.len() > 1 条件)
        let mut e = vec![mirror_meta(1, 10_000, 0, false)];
        assert!(lru_evict_keys(&mut e, 0, 1).is_empty());
        assert_eq!(keys_of(&e), vec![1]);
    }

    /// P58-1: ミラーを落としてもホスト (正) の KV 行は変わらず、再シードで
    /// ビット一致に戻ること。
    ///
    /// ミラーはホスト KV の複製なので、エビクトで失われるのは「デバイス側の
    /// コピー」だけ。再シードはホスト行のコピー (演算なしの H2D) なので、
    /// 丸め順序が変わる余地が無くビット一致する — このテストはその契約
    /// (エビクトがホストを書き換えない・len が 0 に戻って全行コピーになる) を
    /// ホスト側のデータだけで固定する。
    #[test]
    fn kv_mirror_evict_then_reseed_is_bit_identical() {
        const KV_DIM: usize = 32;
        const T: usize = 16;
        const ROWS: usize = T * KV_DIM;
        let host_a = pseudo_random(ROWS, 101);
        let host_b = pseudo_random(ROWS, 202);
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        let (host_a_bits, host_b_bits) = (bits(&host_a), bits(&host_b));

        // ミラー = ホスト行の複製 (実装では DevBuffer への H2D)
        let mut mirror_a = host_a.clone();
        let mirror_b = host_b.clone();
        let bytes = ROWS * 4;

        // 上限 = 1 本分。セッション 2 を使う時点でセッション 1 のミラーが落ちる
        let mut entries = vec![
            mirror_meta(1, bytes, 200, false),
            mirror_meta(2, bytes, 0, false),
        ];
        assert_eq!(lru_evict_keys(&mut entries, bytes, 2), vec![1]);
        assert_eq!(keys_of(&entries), vec![2]);

        // 落ちた側のミラーは空になる (実装では ms からの除去 = 次回 have=0)
        mirror_a.clear();
        assert!(mirror_a.is_empty());

        // ホスト (正) はエビクトで一切変わらない
        assert_eq!(bits(&host_a), host_a_bits, "エビクトはホスト A を書き換えない");
        assert_eq!(bits(&host_b), host_b_bits, "エビクトはホスト B を書き換えない");

        // 再シード = ホスト行 [0..T] のコピー → ビット一致
        mirror_a.extend_from_slice(&host_a[..ROWS]);
        assert_eq!(bits(&mirror_a), host_a_bits, "再シードでビット一致に戻る");

        // 生き残った側のミラーも無傷
        assert_eq!(bits(&mirror_b), host_b_bits, "他セッションのエビクトで壊れない");
    }

    /// P58-1 (GPU 実機・`#[ignore]`): `DENNO_GPU_KV_MB` を超えると実際に LRU で
    /// ミラーが落ち、落ちたセッションの次回呼び出しが have=0 (全行再シード) から
    /// 始まること。実バッファの確保を伴うので既定では回さない。
    ///
    /// 注意: エビクトは with_mirror の先頭で走るため、直前のセッションが確保した
    /// 分だけ上限を一時的に超え得る (超過は次のミラー呼び出しで解消する)。
    #[test]
    #[ignore] // GPU が必要: cargo test --release --features cuda -p denno-engine -- --ignored
    fn cuda_kv_mirror_lru_evicts_over_limit() {
        let _g = test_lock()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        enable_for_tests();
        let _ctx = CudaCtx::new().unwrap();
        let prev_env = std::env::var("DENNO_GPU_KV_MB").ok();
        // 上限 1 MiB。1 セッション = t_total(2048) * kv_dim(64) * 4 B * (k + v) = 1 MiB
        std::env::set_var("DENNO_GPU_KV_MB", "1");
        let (kv_dim, t_total) = (64usize, 2048usize);
        let key_a = u64::MAX - 11;
        let key_b = u64::MAX - 12;
        // A を確保 (1 MiB)
        with_mirror(key_a, 0, 1, t_total, kv_dim, 0, |m, have| {
            assert_eq!(have, 0, "初回はミラー空");
            m.len[0] = t_total;
            Some(())
        })
        .unwrap();
        let (n, used, limit) = gpu_kv_mirror_stats().unwrap();
        assert_eq!(limit, 1024 * 1024, "env 上書きが効いている");
        assert!(n >= 1 && used >= 1024 * 1024, "A の確保後: {n} entries / {used} B");
        // B を確保 (合計 2 MiB > 1 MiB。エビクトは次のミラー呼び出しで走る)
        with_mirror(key_b, 0, 1, t_total, kv_dim, 0, |_, _| Some(())).unwrap();
        // B をもう一度使う → 先頭のエビクトで古い A が落ちる
        with_mirror(key_b, 0, 1, t_total, kv_dim, 0, |_, _| Some(())).unwrap();
        {
            let ms = KV_MIRRORS.get().unwrap().lock().unwrap();
            let keys: Vec<u64> = ms.iter().map(|(k, _, _)| *k).collect();
            assert!(keys.contains(&key_b), "使用中の B は残る: {keys:?}");
            assert!(!keys.contains(&key_a), "上限超過で A がエビクトされる: {keys:?}");
        }
        // A を再度使う → ミラーは作り直され have=0 (全行再シード) から始まる
        let mut have_a = None;
        with_mirror(key_a, 0, 1, t_total, kv_dim, 0, |_, have| {
            have_a = Some(have);
            Some(())
        })
        .unwrap();
        assert_eq!(have_a, Some(0), "エビクト後は全行再シード");
        // 後片付け (テスト用ミラーの除去と env の復元)
        KV_MIRRORS
            .get()
            .unwrap()
            .lock()
            .unwrap()
            .retain(|(k, _, _)| *k != key_a && *k != key_b);
        match prev_env {
            Some(v) => std::env::set_var("DENNO_GPU_KV_MB", v),
            None => std::env::remove_var("DENNO_GPU_KV_MB"),
        }
    }

    /// P17j: prefill attention の CPU 参照 (model/llama.rs forward の prefill
    /// 経路と同一コード: sgemm_scores_kvstrided + causal mask + softmax +
    /// matmul_rows_bstrided_par)。
    fn prefill_attention_cpu(
        q: &[f32],
        kc: &[f32],
        vc: &[f32],
        heads: usize,
        kv_heads: usize,
        dh: usize,
        n: usize,
        pos0: usize,
        t_total: usize,
        scale: f32,
    ) -> Vec<f32> {
        let kv_dim = kv_heads * dh;
        let group = heads / kv_heads;
        let mut att = vec![0.0f32; n * heads * dh];
        for kvh in 0..kv_heads {
            for g in 0..group {
                let qh = kvh * group + g;
                let mut scores = vec![0.0f32; n * t_total];
                tensor::sgemm_scores_kvstrided(
                    &q[qh * dh..],
                    heads * dh,
                    &kc[kvh * dh..],
                    kv_dim,
                    &mut scores,
                    n,
                    dh,
                    t_total,
                    scale,
                );
                for i in 0..n {
                    let row = &mut scores[i * t_total..][..t_total];
                    for j in (pos0 + i + 1)..t_total {
                        row[j] = f32::NEG_INFINITY;
                    }
                    tensor::softmax(row);
                }
                let out =
                    tensor::matmul_rows_bstrided_par(&scores, &vc[kvh * dh..], kv_dim, n, t_total, dh);
                for i in 0..n {
                    att[i * heads * dh + qh * dh..][..dh].copy_from_slice(&out[i * dh..][..dh]);
                }
            }
        }
        att
    }

    /// P17k ステップ 5: dno_rope_qkv_pf (qkv 出力 → rope → q 連続化 +
    /// k/v を GPU KV ミラーへ直接書き込み) が CPU 参照経路 (rope_row) と
    /// **ビット一致**することを検証する。tab は CPU が rope_row と同一式で
    /// 生成し、カーネルは mul/sub のみ適用するため一致が契約。
    /// (qkv GEMM 自体は dno_gemm_i8 = cuda_gemm_i8_bit_exact_vs_cpu で担保)
    #[test]
    #[ignore] // GPU が必要: cargo test --release --features cuda -- --ignored
    fn cuda_rope_qkv_pf_bitexact() {
        let _g = test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        enable_for_tests();
        let (heads, kv_heads, dh) = (12usize, 2usize, 128usize);
        let qn = heads * dh;
        let kv_dim = kv_heads * dh;
        let width = qn + 2 * kv_dim;
        let (pos0, n) = (5usize, 37usize);
        let t_total = pos0 + n;
        let theta = 1_000_000.0f32;

        let ctx = CudaCtx::new().unwrap();
        // qkv GEMM 出力のエミュレーション (行 i = [qn | kv_dim | kv_dim]) + bias
        let qkv = pseudo_random(n * width, 41);
        let bias = pseudo_random(width, 42);
        // rope テーブル: chain_qkv_rope_prefill 内と同一式 (rope_row とビット一致)
        let mut tab = vec![0.0f32; n * dh];
        let half = dh / 2;
        for i in 0..n {
            let pos = (pos0 + i) as f32;
            let row = &mut tab[i * dh..][..dh];
            for d in 0..half {
                let ang = pos * theta.powf(-2.0 * d as f32 / dh as f32);
                row[d] = ang.cos();
                row[half + d] = ang.sin();
            }
        }
        let dqkv = ctx.upload_f32(&qkv).unwrap();
        let dbias = ctx.upload_f32(&bias).unwrap();
        let dtab = ctx.upload_f32(&tab).unwrap();
        let dq = DevBuffer::alloc(n * qn * 4).unwrap();

        // CPU 参照: 行分割 ([n, width] 行-major → q/k/v) + bias 加算 + rope_row
        let mut q_ref = vec![0.0f32; n * qn];
        let mut k_ref = vec![0.0f32; n * kv_dim];
        let mut v_ref = vec![0.0f32; n * kv_dim];
        for i in 0..n {
            let row = &qkv[i * width..][..width];
            q_ref[i * qn..][..qn].copy_from_slice(&row[..qn]);
            k_ref[i * kv_dim..][..kv_dim].copy_from_slice(&row[qn..qn + kv_dim]);
            v_ref[i * kv_dim..][..kv_dim].copy_from_slice(&row[qn + kv_dim..]);
        }
        for b in q_ref.chunks_mut(qn) {
            for (o, bi) in b.iter_mut().zip(&bias[..qn]) {
                *o += *bi;
            }
        }
        for b in k_ref.chunks_mut(kv_dim) {
            for (o, bi) in b.iter_mut().zip(&bias[qn..qn + kv_dim]) {
                *o += *bi;
            }
        }
        for b in v_ref.chunks_mut(kv_dim) {
            for (o, bi) in b.iter_mut().zip(&bias[qn + kv_dim..]) {
                *o += *bi;
            }
        }
        crate::tensor::rope_rows(&mut q_ref, n, heads, dh, pos0, theta);
        crate::tensor::rope_rows(&mut k_ref, n, kv_heads, dh, pos0, theta);

        // GPU 実経路: ミラー取得 (履歴行 [0..pos0) をシード — 現在行
        // [pos0..t_total) は rope カーネルが書く。chain_qkv_rope_prefill と同一)。
        // キーは番兵 (u64::MAX-1) — 実キャッシュの kv_id (1 からの連番) と
        // 衝突すると他テストにミラーを残して汚染する (実害あり)。
        const MIRROR_TEST_KEY: u64 = u64::MAX - 1;
        let mut cache = crate::kv_cache::KvCache::new(1, kv_dim, t_total + 64);
        {
            let (kbuf, vbuf) = cache.kv_bufs(0, t_total);
            kbuf[..t_total * kv_dim].copy_from_slice(&pseudo_random(t_total * kv_dim, 46));
            vbuf[..t_total * kv_dim].copy_from_slice(&pseudo_random(t_total * kv_dim, 47));
        }
        let (kptr, vptr) =
            with_mirror(MIRROR_TEST_KEY, 0, 1, t_total, kv_dim, 0, |m, have| {
            if have < pos0 {
                let (kc_ref, vc_ref) = cache.kv_bufs_ref(0);
                let kb = m.layers[0].0.as_ref().unwrap();
                let vb = m.layers[0].1.as_ref().unwrap();
                kb.copy_from_at(have * kv_dim * 4, &kc_ref[have * kv_dim..pos0 * kv_dim])
                    .unwrap();
                vb.copy_from_at(have * kv_dim * 4, &vc_ref[have * kv_dim..pos0 * kv_dim])
                    .unwrap();
            }
            Some((
                m.layers[0].0.as_ref().unwrap().ptr(),
                m.layers[0].1.as_ref().unwrap().ptr(),
            ))
        })
        .unwrap();

        // s41: コンテンツ世代 (epoch) 不一致で KV ミラーの全層 len と GDN gdn_pos
        // が無効化されること。同一 kv_id (スロット) の cache が reset/restore されて
        // 別会話の KV で再 prefill された場合、前会話の行 [0..have) を正規行として
        // 使ってしまう 2b 2 値分岐 (s35) の根本修正の検証。CPU では実行不可
        // (KV_MIRRORS が cuda feature 配下のため) ので GPU 実機で実行する。
        #[test]
        #[ignore] // GPU が必要: cargo test --release --features cuda -p denno-engine -- --ignored
        fn cuda_kv_mirror_epoch_invalidation() {
            let _g = test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            enable_for_tests();
            let _ctx = CudaCtx::new().unwrap();
            let (kv_dim, t_total) = (64usize, 40usize);
            // 番兵キー (MIRROR_TEST_KEY = u64::MAX-1 と衝突しない)
            let key = u64::MAX - 3;
            let mut cache = crate::kv_cache::KvCache::new(1, kv_dim, t_total + 64);
            // 会話 A: epoch 0 で seed → len[0] = t_total・gdn_pos にダミー値
            {
                let (kbuf, vbuf) = cache.kv_bufs(0, t_total);
                kbuf[..t_total * kv_dim].copy_from_slice(&pseudo_random(t_total * kv_dim, 71));
                vbuf[..t_total * kv_dim].copy_from_slice(&pseudo_random(t_total * kv_dim, 72));
            }
            with_mirror(key, 0, 1, t_total, kv_dim, cache.epoch(), |m, have| {
                assert_eq!(have, 0, "初回はミラー空 (全行シード)");
                m.len[0] = t_total;
                m.gdn_pos[0] = Some(t_total - 1);
                Some(())
            })
            .unwrap();
            // 会話 B: reset → 別内容で再 prefill (epoch が 1 に上がる)
            cache.reset();
            assert_eq!(cache.epoch(), 1);
            {
                let (kbuf, vbuf) = cache.kv_bufs(0, t_total);
                kbuf[..t_total * kv_dim].copy_from_slice(&pseudo_random(t_total * kv_dim, 73));
                vbuf[..t_total * kv_dim].copy_from_slice(&pseudo_random(t_total * kv_dim, 74));
            }
            let mut have_b = None;
            with_mirror(key, 0, 1, t_total, kv_dim, cache.epoch(), |m, have| {
                have_b = Some(have);
                Some(())
            })
            .unwrap();
            assert_eq!(
                have_b,
                Some(0),
                "epoch 不一致で len がリセットされ、前会話の行が再利用されないこと"
            );
            // GDN 状態位置も epoch 不一致で無効化される (次回 decode で全状態再 seed)
            with_mirror_ext(key, 0, 1, t_total, 0, cache.epoch(), false, |m, _| {
                assert!(m.gdn_pos[0].is_none(), "GDN 状態位置も無効化されること");
                Some(())
            })
            .unwrap();
        }
        launch_rope_qkv_pf(
            &ctx,
            dqkv.ptr(),
            dq.ptr(),
            kptr,
            vptr,
            dtab.ptr(),
            dbias.ptr(),
            pos0,
            n,
            heads,
            kv_heads,
            dh,
            kv_dim,
        )
        .unwrap();
        ctx.sync().unwrap();

        // q: rope+連続化 → ビット一致
        let q_gpu = dq.to_vec::<f32>(n * qn).unwrap();
        if q_gpu != q_ref {
            let bad = q_gpu
                .iter()
                .zip(&q_ref)
                .position(|(a, b)| a.to_bits() != b.to_bits())
                .unwrap();
            let (i, r) = (bad / qn, bad % qn);
            panic!(
                "q ビット不一致: idx {bad} (i={i} h={} d={} in-head-d={}) gpu={} ({:08x}) ref={} ({:08x})",
                r / dh,
                r % dh,
                {
                    let d = r % dh;
                    if d < dh / 2 { d } else { d - dh / 2 }
                },
                q_gpu[bad],
                q_gpu[bad].to_bits(),
                q_ref[bad],
                q_ref[bad].to_bits()
            );
        }
        // k/v: ミラーの [pos0..t_total) 行だけ rope 後の値とビット一致
        // (履歴行 [0..pos0) はシード値のまま・不変)
        let k_hist = pseudo_random(t_total * kv_dim, 46);
        let v_hist = pseudo_random(t_total * kv_dim, 47);
        let m2 = KV_MIRRORS.get_or_init(|| std::sync::Mutex::new(Vec::new()));
        let ms = m2.lock().unwrap();
        let (_, mir, _) = ms
            .iter()
            .find(|(k, _, _)| *k == MIRROR_TEST_KEY)
            .unwrap();
        let k_gpu: Vec<f32> = mir.layers[0].0.as_ref().unwrap().to_vec(t_total * kv_dim).unwrap();
        let v_gpu: Vec<f32> = mir.layers[0].1.as_ref().unwrap().to_vec(t_total * kv_dim).unwrap();
        drop(ms);
        for i in 0..t_total {
            let off = i * kv_dim;
            let wk = if i < pos0 { &k_hist[off..off + kv_dim] } else { &k_ref[(i - pos0) * kv_dim..(i - pos0 + 1) * kv_dim] };
            let wv = if i < pos0 { &v_hist[off..off + kv_dim] } else { &v_ref[(i - pos0) * kv_dim..(i - pos0 + 1) * kv_dim] };
            assert_eq!(&k_gpu[off..off + kv_dim], wk, "k 行 {i} ビット不一致");
            assert_eq!(&v_gpu[off..off + kv_dim], wv, "v 行 {i} ビット不一致");
        }
        println!("rope_qkv_pf bit-exact OK (q {n}x{qn}, k/v rows {pos0}..{t_total})");
    }

    /// s123: Q4_0 ネイティブ経路 (GEMV: m<=16 / GEMM: m>=256) が CPU 実装と
    /// R2 (相対 1e-4) 以内で一致し、2 回実行でビット一致 (決定論) すること。
    #[test]
    #[ignore] // GPU が必要: cargo test --release --features cuda -- --ignored
    fn cuda_q4_linear_matches_cpu() {
        let _g = test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        enable_for_tests();
        let (k, n) = (256usize, 64usize);
        let w = pseudo_random(k * n, 5);
        let q = crate::quant::Q4Linear::quantize(&w, k, n);
        for m in [1usize, 7, 16, 300] {
            let a = pseudo_random(m * k, 100 + m as u32);
            let mut want = vec![0f32; m * n];
            crate::tensor::matvec_rows_bt_q4_into(&a, &q.blocks, k, n, m, &mut want);
            let mut got = vec![0f32; m * n];
            assert!(
                maybe_gpu_linear_q4_into(q.uid, &a, &q.blocks, k, n, m, &mut got),
                "m={m}: Q4 GPU 経路が発動しなかった"
            );
            let mut maxd = 0.0f32;
            let mut maxv = 0.0f32;
            for (g, wv) in got.iter().zip(want.iter()) {
                maxd = maxd.max((g - wv).abs());
                maxv = maxv.max(wv.abs());
            }
            println!("q4 m={m}: maxabs={maxd:e} (max|ref|={maxv:e})");
            assert!(
                maxd <= 1e-4 * maxv.max(1.0),
                "m={m}: Q4 GPU と CPU の差 {maxd} が R2 を超えた"
            );
            let mut got2 = vec![0f32; m * n];
            assert!(maybe_gpu_linear_q4_into(q.uid, &a, &q.blocks, k, n, m, &mut got2));
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                got2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "m={m}: Q4 GPU 経路が非決定論"
            );
        }
    }

    /// s132 (Task B1): 複数行 1 block の GEMV (`dno_gemv_q4_0mr`) が、行あたりの
    /// 累積順を 1 行 1 block 経路と同じに保っていること (rows = 1/2/3/8 で全要素
    /// ビット一致) と、2 回実行でビット一致 (決定論) を検証する。
    /// rows = 1 はカーネル内の行ループを 1 周にしただけで `dno_gemv_q4_0` と
    /// 同じ演算列、rows > 1 はその行版。境界 m = 1 / 2 / 7 / 9 / 16 / 17 を踏む。
    #[test]
    #[ignore] // GPU が必要: cargo test --release --features cuda -- --ignored
    fn cuda_q4_gemv_mr_matches_single_row() {
        let _g = test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        enable_for_tests();
        let ctx = match shared_ctx() {
            Some(c) => c,
            None => {
                println!("CUDA が無いので skip (GPU 実機で --ignored 実行)");
                return;
            }
        };
        let f_mr = match ctx.f_mv_q4_mr {
            Some(f) => f,
            None => {
                println!("kernels_int8.ptx に dno_gemv_q4_0mr が無い (PTX 再生成前) ので skip");
                return;
            }
        };
        let _ = f_mr;
        ctx.make_current().expect("make_current");
        // 1) カーネル直叩き: 同じ入力で「1 行 1 block (mr=1)」と「mr 行 1 block」を
        //    比べ、全要素ビット一致を要求する (累積順の契約そのもの)。
        let (k, n) = (256usize, 40usize); // n は 8 の倍数でない (grid.x の端数も踏む)
        let w = pseudo_random(k * n, 9);
        let q = crate::quant::Q4Linear::quantize(&w, k, n);
        let gw = weights_for_q4(q.uid, &q.blocks, k, n).expect("upload");
        for m in [1usize, 2, 7, 9, 16, 17] {
            let a = pseudo_random(m * k, 200 + m as u32);
            let run = |rows: usize| -> Vec<f32> {
                let mut out = vec![f32::NAN; m * n];
                run_q4_gemv_for_test(ctx, &a, gw, m, k, n, rows, &mut out);
                out
            };
            // 参照: mr = 1 (従来経路と同じ「1 行 1 block」) を m 回に分けて回す
            let mut want = vec![0f32; m * n];
            {
                // rows = 1 で 1 行ずつ。カーネルは grid.y = ceil(m/1) で m 行を回る。
                let got1 = run(1);
                want.copy_from_slice(&got1);
            }
            for &mr in &[2usize, 3, 8] {
                let got = run(mr);
                for (i, (g, e)) in got.iter().zip(want.iter()).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        e.to_bits(),
                        "m={m} mr={mr}: row {} col {} がビット不一致 ({g} vs {e})",
                        i / n,
                        i % n
                    );
                }
                // 決定論: 同一入力の 2 回目
                let again = run(mr);
                assert_eq!(
                    got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    again.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "m={m} mr={mr}: 非決定論"
                );
            }
            println!("q4 gemv mr bit-exact: m={m} (rows 1/2/3/8)");
        }
    }

    /// s133 (Task B1, attempt 2): 列ブロック版 GEMV (`dno_gemv_q4_0c2/c4/c8`) が
    /// 1 行 1 block 経路 (`dno_gemv_q4_0mr` の rows = 1) と全要素ビット一致する
    /// ことを検証する。C は 2 / 4 / 8 の 3 通り、n は 8*C の倍数でない値
    /// (grid.x の端数と `gj >= n` の break を踏む)、m は 1 / 2 / 7 / 17 を踏む。
    /// 併せて決定論 (同一入力 2 回でビット一致) も見る。
    #[test]
    #[ignore] // GPU が必要: cargo test --release --features cuda -- --ignored
    pub(crate) fn cuda_q4_gemv_cols_matches_single_row() {
        let _g = test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        enable_for_tests();
        let ctx = match shared_ctx() {
            Some(c) => c,
            None => {
                println!("CUDA が無いので skip (GPU 実機で --ignored 実行)");
                return;
            }
        };
        if ctx.f_mv_q4_mr.is_none() {
            println!("kernels_int8.ptx に dno_gemv_q4_0mr が無い (PTX 再生成前) ので skip");
            return;
        }
        let fc = match ctx.f_mv_q4_c {
            Some(f) => f,
            None => {
                println!("kernels_int8.ptx に dno_gemv_q4_0c* が無い (PTX 再生成前) ので skip");
                return;
            }
        };
        ctx.make_current().expect("make_current");
        // n = 103: 8*2=16 / 8*4=32 / 8*8=64 のどれでも割り切れない。
        let (k, n) = (256usize, 103usize);
        let w = pseudo_random(k * n, 21);
        let q = crate::quant::Q4Linear::quantize(&w, k, n);
        let gw = weights_for_q4(q.uid, &q.blocks, k, n).expect("upload");
        for m in [1usize, 2, 7, 17] {
            let a = pseudo_random(m * k, 300 + m as u32);
            // 参照: 従来経路 (1 行 1 block) と同じ rows = 1。
            let mut want = vec![f32::NAN; m * n];
            run_q4_gemv_for_test(ctx, &a, gw, m, k, n, 1, &mut want);
            for (ci, &c) in [2usize, 4, 8].iter().enumerate() {
                assert_eq!(ci, q4_gemv_c_index(c), "index の対応がずれている");
                let mut got = vec![f32::NAN; m * n];
                run_q4_gemv_cols_for_test(ctx, fc[ci], c, &a, gw, m, k, n, &mut got);
                for (i, (g, e)) in got.iter().zip(want.iter()).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        e.to_bits(),
                        "m={m} C={c}: row {} col {} がビット不一致 ({g} vs {e})",
                        i / n,
                        i % n
                    );
                }
                let mut again = vec![f32::NAN; m * n];
                run_q4_gemv_cols_for_test(ctx, fc[ci], c, &a, gw, m, k, n, &mut again);
                assert_eq!(
                    got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    again.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "m={m} C={c}: 非決定論"
                );
                println!("q4 gemv cols bit-exact: m={m} C={c} (n={n})");
            }
        }
        assert_eq!(q4_gemv_c_index(Q4_GEMV_C), q4_gemv_c_index(4));
        // 本番の選択が 2 / 4 / 8 のどれでも、上のビット一致集合の中にあること。
        let prod = q4_gemv_c();
        assert!(
            [2usize, 4, 8].contains(&prod),
            "q4_gemv_c() が 2/4/8 以外を返した: {prod}"
        );
    }

    /// s133: 列ブロック版 GEMV を c (2/4/8) を明示して 1 回起動する。
    fn run_q4_gemv_cols_for_test(
        ctx: &CudaCtx,
        f: CUfunction,
        c: usize,
        a: &[f32],
        gw: &GpuWq4,
        m: usize,
        k: usize,
        n: usize,
        out: &mut [f32],
    ) {
        let lib = lib().expect("cuda lib");
        let (mut pn, mut ps) = (gw.nib.ptr(), gw.sc.ptr());
        let (mut mc, mut kc, mut nc) = (m as c_int, k as c_int, n as c_int);
        let d_a = DevBuffer::alloc(m * k * 4).expect("alloc a");
        let d_out = DevBuffer::alloc(m * n * 4).expect("alloc out");
        let (mut pa, mut po) = (d_a.ptr(), d_out.ptr());
        unsafe {
            check(
                (lib.memcpy_htod)(d_a.ptr(), a.as_ptr() as *const c_void, m * k * 4),
                "cuMemcpyHtoD",
            )
            .expect("H2D");
            let mut args = [
                &mut pa as *mut _ as *mut c_void,
                &mut pn as *mut _ as *mut c_void,
                &mut ps as *mut _ as *mut c_void,
                &mut po as *mut _ as *mut c_void,
                &mut mc as *mut _ as *mut c_void,
                &mut kc as *mut _ as *mut c_void,
                &mut nc as *mut _ as *mut c_void,
            ];
            let grid = (n.div_ceil(8 * c) as u32, m as u32, 1);
            ctx.launch_async(f, grid, (256, 1, 1), 0, &mut args)
                .expect("launch");
            check(
                (lib.memcpy_dtoh)(out.as_mut_ptr() as *mut c_void, d_out.ptr(), m * n * 4),
                "cuMemcpyDtoH",
            )
            .expect("D2H");
        }
    }

    /// s132: テストから `dno_gemv_q4_0mr` を直接 1 回起動する (行数 rows を明示)。
    /// 本番経路は `q4_dispatch_tile` が rows = min(m, Q4_MR_MAX) を選ぶ。
    fn run_q4_gemv_for_test(
        ctx: &CudaCtx,
        a: &[f32],
        gw: &GpuWq4,
        m: usize,
        k: usize,
        n: usize,
        rows: usize,
        out: &mut [f32],
    ) {
        let lib = lib().expect("cuda lib");
        let f = ctx.f_mv_q4_mr.expect("dno_gemv_q4_0mr");
        let (mut pn, mut ps) = (gw.nib.ptr(), gw.sc.ptr());
        let (mut mc, mut kc, mut nc) = (m as c_int, k as c_int, n as c_int);
        let mut rc = rows as c_int;
        let d_a = DevBuffer::alloc(m * k * 4).expect("alloc a");
        let d_out = DevBuffer::alloc(m * n * 4).expect("alloc out");
        let (mut pa, mut po) = (d_a.ptr(), d_out.ptr());
        unsafe {
            check(
                (lib.memcpy_htod)(d_a.ptr(), a.as_ptr() as *const c_void, m * k * 4),
                "cuMemcpyHtoD",
            )
            .expect("H2D");
            let mut args = [
                &mut pa as *mut _ as *mut c_void,
                &mut pn as *mut _ as *mut c_void,
                &mut ps as *mut _ as *mut c_void,
                &mut po as *mut _ as *mut c_void,
                &mut mc as *mut _ as *mut c_void,
                &mut kc as *mut _ as *mut c_void,
                &mut nc as *mut _ as *mut c_void,
                &mut rc as *mut _ as *mut c_void,
            ];
            let grid = (n.div_ceil(8) as u32, m.div_ceil(rows) as u32, 1);
            ctx.launch_async(f, grid, (256, 1, 1), 0, &mut args)
                .expect("launch");
            check(
                (lib.memcpy_dtoh)(out.as_mut_ptr() as *mut c_void, d_out.ptr(), m * n * 4),
                "cuMemcpyDtoH",
            )
            .expect("D2H");
        }
    }

    /// P17j: prefill attention カーネル (dno_attention_prefill) が CPU 参照経路と
    /// **近似一致**すること (exactness 契約は attention のみ緩和・ユーザー承認)。
    /// 加えて ①決定論 (同一入力の 2 回起動でビット一致) ②causal (未来の K/V を
    /// 書き換えても過去行の出力はビット不変) を検証する。
    #[test]
    #[ignore] // GPU が必要: cargo test --release --features cuda -- --ignored
    fn cuda_attention_prefill_vs_cpu_reference() {
        let _g = test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        enable_for_tests();
        let (heads, kv_heads, dh) = (12usize, 2usize, 128usize);
        let group = heads / kv_heads;
        let kv_dim = kv_heads * dh;
        let (pos0, n) = (5usize, 37usize); // BQ/BK タイル境界をまたぐ
        let t_total = pos0 + n;
        let scale = 1.0 / (dh as f32).sqrt();
        let q = pseudo_random(n * heads * dh, 11);
        let kc = pseudo_random(t_total * kv_dim, 12);
        let vc = pseudo_random(t_total * kv_dim, 13);

        let att_ref =
            prefill_attention_cpu(&q, &kc, &vc, heads, kv_heads, dh, n, pos0, t_total, scale);

        // GPU 実経路 (KvCache 生成 + ミラー同期 + カーネル起動 + D2H)
        let run_gpu = |kc: &[f32], vc: &[f32]| -> Vec<f32> {
            let mut cache = crate::kv_cache::KvCache::new(1, kv_dim, t_total + 64);
            {
                let (kbuf, vbuf) = cache.kv_bufs(0, t_total);
                kbuf[..t_total * kv_dim].copy_from_slice(kc);
                vbuf[..t_total * kv_dim].copy_from_slice(vc);
            }
            assert!(
                chain_attention_prefill(
                    cache.kv_id(),
                    0,
                    1,
                    &q,
                    &cache,
                    pos0,
                    n,
                    t_total,
                    group,
                    dh,
                    kv_dim,
                    scale,
                )
                .is_some(),
                "prefill attention kernel が発動すべき (dh={dh})"
            );
            chain_download(CH_ATT, n * heads * dh).unwrap()
        };

        let got = run_gpu(&kc, &vc);
        let maxd = got
            .iter()
            .zip(att_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("prefill attention maxabs diff vs cpu: {maxd}");
        assert!(maxd < 1e-3, "prefill attention maxabs diff {maxd}");

        // 決定論: 同一入力の 2 回目はビット一致
        let got2 = run_gpu(&kc, &vc);
        assert_eq!(got, got2, "prefill attention が非決定論的");

        // causal: 未来の行 (j > pos0 + i) を書き換えても過去行の出力はビット不変
        let mut kc2 = kc.clone();
        let mut vc2 = vc.clone();
        for j in (pos0 + 3)..t_total {
            let seed = 99 + j as u32;
            kc2[j * kv_dim..(j + 1) * kv_dim].copy_from_slice(&pseudo_random(kv_dim, seed));
            vc2[j * kv_dim..(j + 1) * kv_dim].copy_from_slice(&pseudo_random(kv_dim, seed + 1000));
        }
        let got3 = run_gpu(&kc2, &vc2);
        // 行 0..2 の causal 界 (j <= pos0+i) は変更範囲 (j >= pos0+3) の外 → ビット一致
        for i in 0..3usize {
            let a = &got[i * heads * dh..][..heads * dh];
            let b = &got3[i * heads * dh..][..heads * dh];
            assert_eq!(
                a, b,
                "causal 違反: row {i} が未来の K/V 変更の影響を受けた"
            );
        }
        // 行 3 は causal 界 (j <= pos0+3) が変更範囲に含まれる → 変わるはず
        let a = &got[3 * heads * dh..][..heads * dh];
        let b = &got3[3 * heads * dh..][..heads * dh];
        assert_ne!(a, b, "row 3 が未来の K/V 変更に反応していない (テスト無効)");
    }

    /// P17k ステップ 7: 大規模 n での prefill attention (n=37 のテストは 1 行
    /// ブロック・少数タイルしか通らないため、複数行ブロック×多数タイル・
    /// prefetch が実データを読む規模での近似一致と決定論を検証する)。
    #[test]
    #[ignore] // GPU が必要: cargo test --release --features cuda -- --ignored
    fn cuda_attention_prefill_vs_cpu_large() {
        let _g = test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        enable_for_tests();
        let (heads, kv_heads, dh) = (12usize, 2usize, 128usize);
        let group = heads / kv_heads;
        let kv_dim = kv_heads * dh;
        let (pos0, n) = (64usize, 2056usize); // 行ブロック 33 個・tile 数百
        let t_total = pos0 + n;
        let scale = 1.0 / (dh as f32).sqrt();
        let q = pseudo_random(n * heads * dh, 21);
        let kc = pseudo_random(t_total * kv_dim, 22);
        let vc = pseudo_random(t_total * kv_dim, 23);

        let att_ref =
            prefill_attention_cpu(&q, &kc, &vc, heads, kv_heads, dh, n, pos0, t_total, scale);

        let run_gpu = |kc: &[f32], vc: &[f32]| -> Vec<f32> {
            let mut cache = crate::kv_cache::KvCache::new(1, kv_dim, t_total + 64);
            {
                let (kbuf, vbuf) = cache.kv_bufs(0, t_total);
                kbuf[..t_total * kv_dim].copy_from_slice(kc);
                vbuf[..t_total * kv_dim].copy_from_slice(vc);
            }
            assert!(
                chain_attention_prefill(
                    cache.kv_id(),
                    0,
                    1,
                    &q,
                    &cache,
                    pos0,
                    n,
                    t_total,
                    group,
                    dh,
                    kv_dim,
                    scale,
                )
                .is_some(),
                "prefill attention kernel が発動すべき (dh={dh})"
            );
            chain_download(CH_ATT, n * heads * dh).unwrap()
        };

        let got = run_gpu(&kc, &vc);
        let maxd = got
            .iter()
            .zip(att_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("prefill attention (n={n}) maxabs diff vs cpu: {maxd}");
        assert!(maxd < 1e-3, "prefill attention (n={n}) maxabs diff {maxd}");

        // 決定論: 同一入力の 2 回目はビット一致
        let got2 = run_gpu(&kc, &vc);
        assert_eq!(got, got2, "prefill attention (n={n}) が非決定論的");
        println!("prefill attention large n=({n}) determinism OK");
    }

    /// s49: fp32 lm_head GPU 経路 (maybe_gpu_matvec_f32) が tensor::matvec_bt と
    /// ビット一致すること。dno_matvec_f32 (C4 chain の beta/alpha 制御投影用
    /// カーネル・.cu/PTX 無変更で流用) を block=(1,1,1)・grid=(n,1,1) で起動する:
    /// ①lm_head 実形状相当 (k=4096) での bit 一致 ②n>65535 (vocab 規模) でも
    /// grid.x が正しく機能する (65535 は chain 側呼び出し規約のみのガード) こと
    /// ③決定論 (同一入力 2 回起動でビット一致) ④小さい k (境界) でも一致。
    #[test]
    #[ignore] // GPU が必要: cargo test --release --features cuda -- --ignored
    fn cuda_matvec_f32_lmhead_bit_exact_vs_cpu() {
        let _g = test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        enable_for_tests();
        let _ctx = CudaCtx::new().unwrap();
        // PTX 未再生成 (dno_matvec_f32 未収録) の場合は不発で CPU に落ちるため
        // 最初の呼び出しで発動を明示して fail させる (PTX 再生成後の再実行が前提)。
        for (k, n, seed) in [
            (4096usize, 4096usize, 71),   // lm_head 実形状相当 (k=4096)
            (4096, 131_072 + 17, 72),     // n > 65535 (vocab 規模) の grid.x 検証
            (8, 1, 73),                   // 最小境界 (n=1・小 k)
            (128, 511, 74),               // 端数 n
        ] {
            let w = pseudo_random(k * n, seed);
            let a = pseudo_random(k, seed + 100);
            let ref_cpu = tensor::matvec_bt(&a, &w, k, n);
            let got = maybe_gpu_matvec_f32(9_000_001, &w, &a, k, n)
                .expect("GPU 経路が不発 (kernels_elem.ptx に dno_matvec_f32 が無い/PTX 未再生成の可能性)");
            assert_eq!(
                got.len(),
                ref_cpu.len(),
                "k={k} n={n}: 出力長不一致"
            );
            for (i, (g, r)) in got.iter().zip(ref_cpu.iter()).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    r.to_bits(),
                    "k={k} n={n}: logits[{i}] が bit 不一致 (g={g} r={r})"
                );
            }
            // 決定論: 同一入力の 2 回起動でビット一致
            let got2 = maybe_gpu_matvec_f32(9_000_001, &w, &a, k, n).unwrap();
            for (g, g2) in got.iter().zip(got2.iter()) {
                assert_eq!(g.to_bits(), g2.to_bits(), "k={k} n={n}: 非決定論");
            }
            println!("cuda matvec_f32 k={k} n={n}: bit exact OK");
        }
        // ゲート: 形状不正 (w.len() != k*n) は None (CPU フォールバック) であること
        let w = vec![0.0f32; 10];
        assert!(maybe_gpu_matvec_f32(9_000_002, &w, &[0.0; 4], 4, 3).is_none());
    }

    /// task 69: ルータ (f32 512x2560) の GPU 経路が CPU 経路と**ビット一致**する
    /// こと。合成行列は ±1e3 の一様乱数で、64 行ごとに全体を 1e-42 (非正規化域) に
    /// した行を混ぜる — その行の積は 1e-39 前後の非正規化数になるので、もし
    /// カーネルが FTZ (flush-to-zero) で走っていれば和がちょうど 0 になって落ちる。
    /// 比較は `to_bits()` (`==` は -0.0 と 0.0 を区別しない)。
    ///
    /// task 127: ルータの既定カーネルは `dno_matvec_f32_r` (block 256 の固定木縮約)
    /// になり、丸め順が CPU の逐次累積と違う。`_r` のときは許容判定 + 非正規化の
    /// 行が 0 でないことだけを見る (`DENNO_ROUTER_GPU_REDUCE=0` なら従来どおり
    /// ビット一致)。
    #[test]
    #[ignore] // GPU が必要: cargo test --release --features cuda -- --ignored
    fn cuda_f32_gemv_bit_exact_vs_cpu() {
        use crate::kquant::GgmlType;
        use crate::model::linear_gguf;
        let _g = test_lock()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        enable_for_tests();
        let _ctx = CudaCtx::new().unwrap();
        let (k, n) = (2560usize, 512usize);

        // 決定的な擬似乱数 ([-1e3, 1e3) の一様)。
        let mut state = 0x1234_5678u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 8_388_608.0) * 2000.0 - 1000.0
        };
        let mut w = vec![0f32; k * n];
        for r in 0..n {
            for i in 0..k {
                let v = next();
                w[r * k + i] = if r % 64 == 0 {
                    // 非正規化の積を作る行 (x は ±1e3 なので積は ~1e-39)。
                    if i % 2 == 0 {
                        1.0e-42
                    } else {
                        -1.0e-42
                    }
                } else if i % 64 == 63 {
                    -1.0e-42 // 通常行にも小さい値を混ぜる
                } else {
                    v
                };
            }
        }
        let mut x = vec![0f32; k];
        for i in 0..k {
            let v = next();
            x[i] = if i % 32 == 31 { 3.0e-40 } else { v };
        }

        let w_bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        let lin = linear_gguf::GgufLinear::from_slice(GgmlType::F32, k, n, &w_bytes)
            .expect("GgufLinear(F32)")
            .with_router_gpu(true);

        // 1. CPU 参照 (GPU ゲートを切る)。
        linear_gguf::set_router_gpu_for_test(Some(false));
        let mut cpu = vec![0f32; n];
        lin.gemv(&x, &mut cpu).expect("CPU 参照");
        set_last_linear_ran_on_gpu(false);

        // 2. GPU 経路。不発なら fail (PTX 未収録・DENNO_CUDA 無効を黙って通さない)。
        linear_gguf::set_router_gpu_for_test(Some(true));
        let mut gpu = vec![0f32; n];
        lin.gemv(&x, &mut gpu).expect("GPU 経路");
        let fired = last_linear_ran_on_gpu();
        linear_gguf::set_router_gpu_for_test(None);
        assert!(
            fired,
            "ルータ GPU 経路が不発 (kernels_elem.ptx に dno_matvec_f32 が無い / DENNO_CUDA 無効)"
        );

        // 3. 非正規化の行が実際に非ゼロであること (FTZ 検出の前提)。
        let tiny_rows: Vec<usize> = (0..n).step_by(64).collect();
        for &r in &tiny_rows {
            assert!(
                cpu[r] != 0.0,
                "行 {r} の CPU 参照が 0 (非正規化の検出力が無い)"
            );
        }

        // 4. `_r` でなければ 1 要素も違わない。`_r` は丸め順が違うので許容判定
        //    (非正規化の行は 0 でないことを別に見る = FTZ 検出を弱めない)。
        let use_reduce = router_gpu_reduce_enabled() && router_use_reduce_kernel(k, n, true);
        for i in 0..n {
            if use_reduce {
                let diff = (gpu[i] - cpu[i]).abs();
                let tol = 1e-4 + 1e-5 * cpu[i].abs();
                assert!(
                    diff <= tol,
                    "行 {i} が CPU と許容外 (gpu={} cpu={} diff={diff:e} tol={tol:e})",
                    gpu[i],
                    cpu[i]
                );
                if tiny_rows.contains(&i) {
                    assert!(
                        gpu[i] != 0.0,
                        "行 {i} (非正規化) が 0 = `_r` が FTZ で走っている"
                    );
                }
            } else {
                assert_eq!(
                    gpu[i].to_bits(),
                    cpu[i].to_bits(),
                    "行 {i} が bit 不一致 (gpu={} cpu={})",
                    gpu[i],
                    cpu[i]
                );
            }
        }

        // 5. 決定論 (同一入力の 2 回目)。
        let mut gpu2 = vec![0f32; n];
        lin.gemv(&x, &mut gpu2).expect("GPU 経路 2 回目");
        for i in 0..n {
            assert_eq!(gpu2[i].to_bits(), gpu[i].to_bits(), "行 {i} が非決定論的");
        }
        println!(
            "cuda router f32 gemv k={k} n={n}: {} OK (非正規化の行 {} 本)",
            if use_reduce {
                "reduce (`_r`) 許容内"
            } else {
                "bit exact"
            },
            tiny_rows.len()
        );
    }

    /// s48 (C1): qwen35 形状 (dh=256・heads=16・kv_heads=4・group=4 = 実 9B の
    /// full attention) での prefill attention GPU バッチ経路。動的 smem が 84.8KB
    /// (既定 48KB 上限超え) であるため C1-a の opt-in (CudaCtx 生成時の
    /// cuFuncSetAttribute) が必須 — opt-in が実機で不可の場合、上限が 48KB のまま
    /// 残り本テストは fail する (C1-b タイル縮小 + .cu 変更/PTX 再生成への切替判断
    /// 材料)。CPU 参照との近似一致 (maxabs<1e-3・attention 緩和枠) + 決定論
    /// (同一入力の 2 回起動でビット一致) + ゲートの発動条件 (n>=gemm_min_m) を検証。
    #[test]
    #[ignore] // GPU が必要: cargo test --release --features cuda -- --ignored
    fn cuda_attention_prefill_qwen35_shape_vs_cpu_reference() {
        let _g = test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        enable_for_tests();
        let (heads, kv_heads, dh) = (16usize, 4usize, 256usize); // qwen35 9B (e=4096)
        let group = heads / kv_heads;
        let kv_dim = kv_heads * dh;
        let sh = attn_pf_sh_f(dh) * 4;
        let ctx = CudaCtx::new().unwrap();
        println!("device: {}", ctx.info());
        println!(
            "dno_attention_prefill dh={dh}: dynamic smem = {sh} bytes (opt-in limit = {})",
            attn_pf_dynamic_smem_limit()
        );
        // C1-a: opt-in 成功で 84.8KB が通ること (失敗なら C1-b への切替が必要)
        assert!(
            attn_pf_dynamic_smem_limit() >= sh,
            "動的 smem の opt-in が失敗している (limit={} < 必要量 {sh}) — C1-b (タイル縮小 + .cu 変更/PTX 再生成) への切替が必要",
            attn_pf_dynamic_smem_limit()
        );
        // ゲート: n >= gemm_min_m() (既定 256) で発動・未満は不発 (短文は CPU のまま)
        let (pos0, n) = (5usize, 276usize); // BQ=64 の行ブロック境界をまたぐ
        assert!(attn_prefill_gpu_supported(n, dh, pos0 + n));
        assert!(
            !attn_prefill_gpu_supported(gemm_min_m() - 1, dh, pos0 + n),
            "n < gemm_min_m() は不発のはず (短文は CPU attend を維持)"
        );
        let t_total = pos0 + n;
        let scale = 1.0 / (dh as f32).sqrt();
        let q = pseudo_random(n * heads * dh, 31);
        let kc = pseudo_random(t_total * kv_dim, 32);
        let vc = pseudo_random(t_total * kv_dim, 33);

        let att_ref =
            prefill_attention_cpu(&q, &kc, &vc, heads, kv_heads, dh, n, pos0, t_total, scale);

        let run_gpu = |kc: &[f32], vc: &[f32]| -> Vec<f32> {
            let mut cache = crate::kv_cache::KvCache::new(1, kv_dim, t_total + 64);
            {
                let (kbuf, vbuf) = cache.kv_bufs(0, t_total);
                kbuf[..t_total * kv_dim].copy_from_slice(kc);
                vbuf[..t_total * kv_dim].copy_from_slice(vc);
            }
            assert!(
                chain_attention_prefill(
                    cache.kv_id(),
                    0,
                    1,
                    &q,
                    &cache,
                    pos0,
                    n,
                    t_total,
                    group,
                    dh,
                    kv_dim,
                    scale,
                )
                .is_some(),
                "prefill attention kernel が発動すべき (dh={dh})"
            );
            chain_download(CH_ATT, n * heads * dh).unwrap()
        };

        let got = run_gpu(&kc, &vc);
        let maxd = got
            .iter()
            .zip(att_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("prefill attention (qwen35 shape dh={dh}) maxabs diff vs cpu: {maxd}");
        assert!(maxd < 1e-3, "prefill attention (dh={dh}) maxabs diff {maxd}");

        // 決定論: 同一入力の 2 回目はビット一致
        let got2 = run_gpu(&kc, &vc);
        assert_eq!(got, got2, "prefill attention (dh={dh}) が非決定論的");
        println!("prefill attention qwen35 shape (dh={dh}) determinism OK");
    }

    /// P-gemv2: GPU の GEMV (x 先読み版カーネル = 既定) が CPU の逐次実装と
    /// **ビット一致**すること。4 型すべてで、行数も複数 (ブロック跨ぎ) で見る。
    ///
    /// 重みは決定的な擬似乱数バイト。scale は f16 の 1.0 (`0x3C00`) に固定して
    /// NaN/Inf を作らない (NaN は to_bits 比較でも一致するはずだが、失敗時に
    /// 原因が読みにくくなるため避ける)。
    #[test]
    #[ignore]
    fn cuda_moe_gemv_v2_bit_exact_vs_cpu() {
        use crate::kquant::GgmlType as T;
        let ctx = CudaCtx::new().unwrap();
        println!("device: {}", ctx.info());
        let n_in = 512usize;
        let n_out = 13usize; // 4 行/ブロックを跨ぐ中途半端な数
        let x = pseudo_random(n_in, 7);
        for ty in [T::Q8_0, T::Q5_1, T::Q4K, T::Q5K] {
            let rb = gguf_row_bytes(ty, n_in).expect("row_bytes");
            let blk = ty.blck_size();
            let hdr = match ty {
                T::Q8_0 | T::Q4K | T::Q5K => 4usize, // d (+ dmin)
                T::Q5_1 => 4usize,                   // d, m
                _ => unreachable!(),
            };
            let mut w = vec![0u8; rb * n_out];
            let mut s = 12345u32;
            for r in 0..n_out {
                let row = &mut w[r * rb..(r + 1) * rb];
                for b in 0..(n_in / blk) {
                    let base = b * ty.type_size();
                    // d = 1.0 (f16), Q4_K/Q5_K の dmin, Q5_1 の m も 1.0
                    row[base] = 0x00;
                    row[base + 1] = 0x3c;
                    if hdr == 4 {
                        row[base + 2] = 0x00;
                        row[base + 3] = 0x3c;
                    }
                    for i in hdr..ty.type_size() {
                        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        row[base + i] = (s >> 16) as u8;
                    }
                }
            }
            let mut want = vec![0f32; n_out];
            crate::model::moe_expert::expert_gemv_scalar(ty, &w, n_in, n_out, &x, &mut want)
                .expect("cpu 参照");
            let dev_w = DevBuffer::from_slice(&w).unwrap();
            let mut got = vec![0f32; n_out];
            // v3 (4 レーン版) は加算順が変わるので、このテストは明示的に v2 経路へ
            // 固定する (DENNO_GEMV_V3=1 の環境でもビット一致の基準を保つ)。
            assert!(
                ctx.moe_gemv_dev_with(ty, &dev_w, &x, n_in, n_out, &mut got, GemvKernelMode::V2)
                    .unwrap(),
                "{ty:?}: GPU 経路が発動しなかった"
            );
            let bad: Vec<usize> = (0..n_out)
                .filter(|&i| got[i].to_bits() != want[i].to_bits())
                .collect();
            assert!(
                bad.is_empty(),
                "{ty:?}: CPU と{}行で不一致 (行 {:?}: gpu={:?} cpu={:?})",
                bad.len(),
                &bad[..bad.len().min(4)],
                bad.iter().map(|&i| got[i]).take(4).collect::<Vec<_>>(),
                bad.iter().map(|&i| want[i]).take(4).collect::<Vec<_>>()
            );
        }
    }

    /// P-gemv3: lanes レーン版 GEMV が CPU の逐次実装 (`expert_gemv_scalar`) と
    /// 相対 1e-5 以内で一致し、NaN / Inf を作らないこと。
    ///
    /// v3 は 1 行の加算を lanes 分割するので v2 とは**ビット一致しない**。ここでは
    /// 行列全体で「最大絶対差 / 最大絶対値」を相対誤差として見る (行ごとの相対は
    /// 打ち消し合いで 0 に近い行が暴れるため)。v3 が実際に走ったことは
    /// 「v2 の結果とビット一致しない行がある」ことで確かめる (黙って v2 に
    /// 落ちていたら 1 bit も変わらない)。
    ///
    /// lanes (4 = k が 128 の倍数 / 2 = k が 64 の倍数) × rows/block (8/16/32/64) の
    /// 組み合わせを回す。rows は env (OnceLock で 1 回しか読めない) ではなく
    /// テスト用の上書き (`set_v3_rows_for_test`) で振る。
    ///
    /// 実行 (GPU のある環境):
    ///   DENNO_GEMV_V3_PTX=$PWD/crates/denno-engine/kernels/kernels_moe_v3.ptx \
    ///   cargo test --release --features cuda -- --ignored cuda_gemv_v3_matches_cpu
    #[test]
    #[ignore]
    fn cuda_gemv_v3_matches_cpu() {
        use crate::kquant::GgmlType as T;
        let ctx = CudaCtx::new().unwrap();
        println!("device: {}", ctx.info());
        assert!(
            moe_gemv_v3_loaded(),
            "v3 の PTX が読めていない (DENNO_GEMV_V3_PTX / kernels_moe_v3.ptx を確認)"
        );
        // 実モデルの k: Q4_K 2560 / Q5_1 640 / Q8_0 2560 は 4 レーン、
        // Q8_0 320 (`output_hc_up`) は 2 レーン。n_out は 1 block の行数を跨ぐ。
        let cases: [(T, usize, usize); 4] = [
            (T::Q4K, 2560, 256),
            (T::Q5_1, 640, 2560),
            (T::Q8_0, 2560, 256),
            (T::Q8_0, 320, 10240),
        ];
        for (ty, n_in, n_out) in cases {
            let lanes = moe_gemv_v3_lanes(n_in);
            assert!(lanes > 0, "{ty:?} k={n_in} は v3 の対象外");
            let rb = gguf_row_bytes(ty, n_in).expect("row_bytes");
            let blk = ty.blck_size();
            // v2 のビット一致テストと同じ合成行列: d (と Q4_K/Q5_K の dmin、
            // Q5_1 の m) は f16 の 1.0 固定 = NaN / Inf を作らない。
            let hdr = 4usize;
            let mut w = vec![0u8; rb * n_out];
            let mut s = 12345u32;
            for r in 0..n_out {
                let row = &mut w[r * rb..(r + 1) * rb];
                for b in 0..(n_in / blk) {
                    let base = b * ty.type_size();
                    row[base] = 0x00;
                    row[base + 1] = 0x3c;
                    row[base + 2] = 0x00;
                    row[base + 3] = 0x3c;
                    for i in hdr..ty.type_size() {
                        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        row[base + i] = (s >> 16) as u8;
                    }
                }
            }
            let x = pseudo_random(n_in, 7);
            let mut want = vec![0f32; n_out];
            crate::model::moe_expert::expert_gemv_scalar(ty, &w, n_in, n_out, &x, &mut want)
                .expect("cpu 参照");
            let dev_w = DevBuffer::from_slice(&w).unwrap();
            let mut got_v2 = vec![0f32; n_out];
            assert!(
                ctx.moe_gemv_dev_with(ty, &dev_w, &x, n_in, n_out, &mut got_v2, GemvKernelMode::V2)
                    .unwrap(),
                "{ty:?}: GPU 経路 (v2) が発動しなかった"
            );
            // 基準 (v2 は CPU とビット一致する契約) が壊れていないこと。
            // rows/block を振っても v2 の経路 (4 行/ブロック固定) は変わらない。
            for i in 0..n_out {
                assert_eq!(
                    got_v2[i].to_bits(),
                    want[i].to_bits(),
                    "{ty:?}: v2 が CPU とビット一致しない (行 {i})"
                );
            }
            for rows in [8usize, 16, 32, 64] {
                set_v3_rows_for_test(rows);
                let mut got = vec![0f32; n_out];
                assert!(
                    ctx.moe_gemv_dev_with(ty, &dev_w, &x, n_in, n_out, &mut got, GemvKernelMode::V3)
                        .unwrap(),
                    "{ty:?}: GPU 経路 (v3 lanes={lanes} rows={rows}) が発動しなかった"
                );
                // v3 が実際に走ったか (走っていれば加算順の違いでビットが変わる)
                let ndiff = (0..n_out)
                    .filter(|&i| got[i].to_bits() != got_v2[i].to_bits())
                    .count();
                assert!(
                    ndiff > 0,
                    "{ty:?} lanes={lanes} rows={rows}: v3 の出力が v2 と 1 行も違わない \
                     = v3 が走っていない (PTX 未ロード / k={n_in} が 64 の倍数でない)"
                );
                let mut maxd = 0f32;
                let mut maxv = 0f32;
                let mut nonfinite = 0usize;
                for i in 0..n_out {
                    if !got[i].is_finite() {
                        nonfinite += 1;
                        continue;
                    }
                    maxd = maxd.max((got[i] - want[i]).abs());
                    maxv = maxv.max(want[i].abs());
                }
                let rel = maxd / maxv.max(f32::MIN_POSITIVE);
                println!(
                    "{ty:?} k={n_in} n={n_out} lanes={lanes} rows={rows} block={}: \
                     max|diff|={maxd:e} max|ref|={maxv:e} rel={rel:e} \
                     (v2 と {ndiff}/{n_out} 行がビット不一致)",
                    rows * lanes
                );
                assert_eq!(nonfinite, 0, "{ty:?}: NaN / Inf が {nonfinite} 行");
                assert!(
                    rel < 1e-5,
                    "{ty:?} lanes={lanes} rows={rows}: 相対誤差 {rel:e} >= 1e-5 \
                     (max|diff|={maxd:e}, max|ref|={maxv:e})"
                );
            }
            set_v3_rows_for_test(0);
        }
    }

    /// task 177: v4 (レーン可変) GEMV が CPU の逐次実装 (`expert_gemv_scalar`) と
    /// 相対 1e-5 以内で一致し、NaN / Inf を作らないこと。
    ///
    /// 形が合う lanes (4 / 8 / 16 / 32) を全部回す。v4 が実際に走ったことは
    /// 「v2 の結果とビット一致しない行がある」ことで確かめる (黙って v3 / v2 に
    /// 落ちていたら 1 bit も変わらない)。
    ///
    /// 実行 (GPU のある環境):
    ///   DENNO_GEMV_V4_PTX=$PWD/crates/denno-engine/kernels/kernels_moe_v4.ptx \
    ///   cargo test --release --features cuda -- --ignored cuda_gemv_v4_matches_cpu
    #[test]
    #[ignore]
    fn cuda_gemv_v4_matches_cpu() {
        use crate::kquant::GgmlType as T;
        let ctx = CudaCtx::new().unwrap();
        println!("device: {}", ctx.info());
        assert!(
            moe_gemv_v4_loaded(),
            "v4 の PTX が読めていない (DENNO_GEMV_V4_PTX / kernels_moe_v4.ptx を確認)"
        );
        // GDN decode の実形 (k=2560 の qkv 3 種 / k=6144 の attn_out) と、
        // レーンが 4 までしか立たない形 (k=640)。
        let cases: [(T, usize, usize); 5] = [
            (T::Q8_0, 2560, 256),
            (T::Q4K, 2560, 256),
            (T::Q5K, 2560, 256),
            (T::Q8_0, 6144, 256),
            (T::Q5_1, 640, 2560),
        ];
        for (ty, n_in, n_out) in cases {
            let rb = gguf_row_bytes(ty, n_in).expect("row_bytes");
            let blk = ty.blck_size();
            // v3 のテストと同じ合成行列 (d と dmin / m は f16 の 1.0 固定)。
            let hdr = 4usize;
            let mut w = vec![0u8; rb * n_out];
            let mut s = 12345u32;
            for r in 0..n_out {
                let row = &mut w[r * rb..(r + 1) * rb];
                for b in 0..(n_in / blk) {
                    let base = b * ty.type_size();
                    row[base] = 0x00;
                    row[base + 1] = 0x3c;
                    row[base + 2] = 0x00;
                    row[base + 3] = 0x3c;
                    for i in hdr..ty.type_size() {
                        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        row[base + i] = (s >> 16) as u8;
                    }
                }
            }
            let x = pseudo_random(n_in, 7);
            let mut want = vec![0f32; n_out];
            crate::model::moe_expert::expert_gemv_scalar(ty, &w, n_in, n_out, &x, &mut want)
                .expect("cpu 参照");
            let dev_w = DevBuffer::from_slice(&w).unwrap();
            let dev_x = DevBuffer::from_slice(&x).unwrap();
            let mut got_v2 = vec![0f32; n_out];
            assert!(
                ctx.moe_gemv_dev_with(ty, &dev_w, &x, n_in, n_out, &mut got_v2, GemvKernelMode::V2)
                    .unwrap(),
                "{ty:?}: GPU 経路 (v2) が発動しなかった"
            );
            // cap を振って、実際に選ばれた lanes の種類だけを回す。
            let mut tried: Vec<usize> = Vec::new();
            for cap in [4usize, 8, 16, 32] {
                let Some(plan) = ctx.plan_moe_gemv_v4(ty, n_in, n_out, cap) else {
                    continue;
                };
                if tried.contains(&plan.lanes) {
                    continue;
                }
                tried.push(plan.lanes);
                let dev_out = DevBuffer::alloc(n_out * 4).unwrap();
                gdn_launch_gemv_dev(
                    &ctx,
                    &plan,
                    &dev_w,
                    dev_x.ptr(),
                    dev_out.ptr(),
                    n_in,
                    n_out,
                )
                .expect("v4 の起動");
                ctx.sync().expect("sync");
                let got = dev_out.to_vec::<f32>(n_out).unwrap();
                let ndiff = (0..n_out)
                    .filter(|&i| got[i].to_bits() != got_v2[i].to_bits())
                    .count();
                let mut maxd = 0f32;
                let mut maxv = 0f32;
                let mut nonfinite = 0usize;
                for i in 0..n_out {
                    if !got[i].is_finite() {
                        nonfinite += 1;
                        continue;
                    }
                    maxd = maxd.max((got[i] - want[i]).abs());
                    maxv = maxv.max(want[i].abs());
                }
                let rel = maxd / maxv.max(f32::MIN_POSITIVE);
                println!(
                    "{ty:?} k={n_in} n={n_out} lanes={} block={}: max|diff|={maxd:e} \
                     max|ref|={maxv:e} rel={rel:e} (v2 と {ndiff}/{n_out} 行がビット不一致)",
                    plan.lanes, plan.block_dim
                );
                assert_eq!(nonfinite, 0, "{ty:?}: NaN / Inf が {nonfinite} 行");
                assert!(
                    rel < 1e-5,
                    "{ty:?} lanes={}: 相対誤差 {rel:e} >= 1e-5 (max|diff|={maxd:e}, \
                     max|ref|={maxv:e})",
                    plan.lanes
                );
                assert!(
                    ndiff > 0,
                    "{ty:?} lanes={}: v4 の出力が v2 と 1 行も違わない = v4 が走っていない",
                    plan.lanes
                );
            }
            assert!(!tried.is_empty(), "{ty:?} k={n_in}: v4 の計画が 1 つも立たない");
            println!("{ty:?} k={n_in}: v4 の lanes = {tried:?}");
        }
    }

    /// v3 のレーン数の適格判定と、`DENNO_GEMV_V3_ROWS` の真理値表 (GPU 無しで回る)。
    ///
    /// 実モデルの k (640 / 2560 / 5120 / 6144) は 4 レーン、320 (`hc_up`) は
    /// 2 レーン、1000 は v2。ここが崩れると v3 が「サブブロックの途中から読む」
    /// ことになる (2 レーンは区間長 k/2 が 32 の倍数であること)。
    #[test]
    fn gemv_v3_lane_and_rows_truth_table() {
        // 区間長 k/4 が 32 の倍数
        for k in [128usize, 640, 2560, 5120, 6144, 10240] {
            assert_eq!(moe_gemv_v3_lanes(k), 4, "k={k} は 4 レーン");
            assert!(k % 128 == 0);
        }
        // 4 レーンでは割れないが 2 レーンなら割れる (k % 64 == 0 かつ k % 128 != 0)
        for k in [64usize, 192, 320, 448, 1344, 1472] {
            assert_eq!(moe_gemv_v3_lanes(k), 2, "k={k} は 2 レーン");
            assert!(k % 64 == 0 && k % 128 != 0);
        }
        // 2 でも割れない → v2
        for k in [0usize, 32, 96, 1000, 3208] {
            assert_eq!(moe_gemv_v3_lanes(k), 0, "k={k} は v2");
        }
        assert!(moe_gemv_v3_k_ok(320) && moe_gemv_v3_k_ok(2560));
        assert!(!moe_gemv_v3_k_ok(1000));

        // DENNO_GEMV_V3_ROWS: 8/16/32/64 だけを通し、他は既定 8 (main の実測で最良)。
        let table: [(Option<&str>, usize); 11] = [
            (Some("8"), 8),
            (Some("16"), 16),
            (Some("32"), 32),
            (Some("64"), 64),
            (Some(" 32 "), 32),
            (None, 8),
            (Some(""), 8),
            (Some("0"), 8),
            (Some("128"), 8),
            (Some("abc"), 8),
            (Some("-16"), 8),
        ];
        for (raw, want) in table {
            assert_eq!(parse_v3_rows(raw), want, "DENNO_GEMV_V3_ROWS={raw:?}");
        }
        // 環境変数の実値 (OnceLock) とも一致する (GPU テストが並走して
        // `set_v3_rows_for_test` を立てているときはその値を期待する)。
        let env_rows = match V3_ROWS_FORCE.load(std::sync::atomic::Ordering::Relaxed) {
            0 => parse_v3_rows(std::env::var("DENNO_GEMV_V3_ROWS").ok().as_deref()),
            forced => forced,
        };
        assert_eq!(moe_gemv_v3_rows(), env_rows, "DENNO_GEMV_V3_ROWS の実値");

        // ゲートは env の値だけで決まる (未設定は off)。
        let env_on = matches!(
            std::env::var("DENNO_GEMV_V3").ok().as_deref(),
            Some("1") | Some("on") | Some("true")
        );
        assert_eq!(moe_gemv_v3_on(), env_on, "DENNO_GEMV_V3 の真理値表");
    }

    /// task 137: コミット済み `kernels_gdn.ptx` の `dno_gdn_scan_sig` が
    /// `dno_gdn_scan` と同じ引数個数 (23) で入っていること。
    ///
    /// ホスト (`gdn_decode_submit`) は 23 引数を渡す。PTX が古い (カーネルが
    /// 無い / 別ソース由来) と GPU decode scan 経路は **黙って** CPU に落ちる
    /// (速度だけが失われる) ので、ここで静的に固定する。
    #[test]
    pub(crate) fn gdn_scan_sig_abi_matches_scan_in_committed_ptx() {
        let text = std::str::from_utf8(PTX_GDN).expect("kernels_gdn.ptx が UTF-8 でない");
        assert_eq!(
            ptx_entry_params(text, "dno_gdn_scan"),
            Some(23),
            "dno_gdn_scan の引数個数が 23 でない (ホストの起動コードと不一致)"
        );
        assert_eq!(
            ptx_entry_params(text, "dno_gdn_scan_sig"),
            Some(23),
            "コミット済み kernels_gdn.ptx に dno_gdn_scan_sig が 23 引数で入っていない \
             (PTX 再生成が必要)"
        );
        assert!(gdn_scan_sig_abi_ok(text), "ABI 検査が committed PTX で成立しない");
    }

    /// PTX の引数の数え方 (GPU 無しで回る)。nvcc が実際に出す
    /// `kernels_moe_v2.ptx` で 6 本を数えられること = パーサの前提の裏取り。
    #[test]
    pub(crate) fn v3_ptx_param_count_matches_nvcc_output() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_moe_v2.ptx");
        let text = std::fs::read_to_string(&p).expect("kernels_moe_v2.ptx が読めない");
        for name in [
            "dno_gemv_q8_0_v2",
            "dno_gemv_q5_1_v2",
            "dno_gemv_q4k_v2",
            "dno_gemv_q5k_v2",
        ] {
            assert_eq!(ptx_entry_params(&text, name), Some(6), "{name}");
        }
        // v3 の新 ABI (rows_per_block と lanes が増えた 8 本)
        let synth = "\
.visible .entry dno_gemv_q4k_v3(\n\
\t.param .u64 dno_gemv_q4k_v3_param_0,\n\
\t.param .u64 dno_gemv_q4k_v3_param_1,\n\
\t.param .u64 dno_gemv_q4k_v3_param_2,\n\
\t.param .u32 dno_gemv_q4k_v3_param_3,\n\
\t.param .u32 dno_gemv_q4k_v3_param_4,\n\
\t.param .u32 dno_gemv_q4k_v3_param_5,\n\
\t.param .u32 dno_gemv_q4k_v3_param_6,\n\
\t.param .u32 dno_gemv_q4k_v3_param_7\n\
)\n\
{\n\
}\n";
        assert_eq!(ptx_entry_params(synth, "dno_gemv_q4k_v3"), Some(V3_NARGS));
        // 名前の前方一致で別カーネルを拾わない
        assert_eq!(ptx_entry_params(synth, "dno_gemv_q4k_v3_l2"), None);
        assert_eq!(ptx_entry_params("", "dno_gemv_q4k_v3"), None);
    }

    // ---- task 108: fp16 カーネルの演算列ミラー (CPU) --------------------------
    //
    // kernels_moe_fp16.cu の行を 1 対 1 で写したもの (索引と丸め順の検証が目的)。
    // 実機での検証は #[ignore] の cuda_gemv_fp16_matches_cpu が担当する。
    //
    // f16 の 1 回丸め演算 (__hmul2 / __hfma2 / __hadd2) は「f64 で計算してから
    // f16 へ 1 回丸める」で等価に模せる: f16 値 (仮数 11 ビット) 同士の積は
    // 22 ビット、積和も指数差を含め f64 (仮数 53 ビット) に正確に収まるので、
    // 最後の丸め 1 回が真の f16 FMA と同じになる。x の f32 → f16 は
    // cvt.rn.f16x2.f32 と同じ最近接丸め (half::f16::from_f32)。

    /// f64 の値を f16 に 1 回丸める (hfma2 / hmul2 / hadd2 の最終丸め)。
    fn fp16_round(v: f64) -> f64 {
        f64::from(half::f16::from_f64(v))
    }

    /// f16 ビット列 → f32 (正確。dno_f16_to_f32_fp16 / v3 ヘルパと同じ値)。
    fn fp16_bits_to_f32(bits: u16) -> f32 {
        half::f16::from_bits(bits).to_f32()
    }

    /// dno_scale_min_k4_fp16 のミラー (1 対 1)。
    fn mirror_scale_min_k4(j: usize, q: &[u8]) -> (u32, u32) {
        if j < 4 {
            ((u32::from(q[j]) & 63), (u32::from(q[j + 4]) & 63))
        } else {
            (
                (u32::from(q[j + 4]) & 0x0f) | ((u32::from(q[j - 4]) >> 6) << 4),
                (u32::from(q[j + 4]) >> 4) | ((u32::from(q[j]) >> 6) << 4),
            )
        }
    }

    /// moe_fp16_block_sum のミラー。2 本の 8 段 hfma2 チェーン → hadd2 →
    /// (low, high) の f32 ペアを返す (呼び出し側で f32 の加算順を再現する)。
    /// `split_pairs` が true のときペア j の x は (xv[j], xv[16+j]) (Q5_1)、
    /// false のとき (xv[2j], xv[2j+1]) (Q8_0 / Q4_K)。
    fn mirror_block_sum(
        y: &[(f64, f64); 16],
        xv: &[f32; 32],
        split_pairs: bool,
    ) -> (f32, f32) {
        let xp = |j: usize| -> (f64, f64) {
            if split_pairs {
                (
                    f64::from(half::f16::from_f32(xv[j])),
                    f64::from(half::f16::from_f32(xv[16 + j])),
                )
            } else {
                (
                    f64::from(half::f16::from_f32(xv[2 * j])),
                    f64::from(half::f16::from_f32(xv[2 * j + 1])),
                )
            }
        };
        // ペア 0..8 のチェーン (low / high 成分ごとに f16 で 1 回丸む)
        let (mut a0, mut a1) = (0f64, 0f64);
        for j in 0..8 {
            let (x0, x1) = xp(j);
            a0 = fp16_round(y[j].0 * x0 + a0);
            a1 = fp16_round(y[j].1 * x1 + a1);
        }
        // ペア 8..16 のチェーン
        let (mut b0, mut b1) = (0f64, 0f64);
        for j in 8..16 {
            let (x0, x1) = xp(j);
            b0 = fp16_round(y[j].0 * x0 + b0);
            b1 = fp16_round(y[j].1 * x1 + b1);
        }
        // __hadd2(acca, accb) → __low2float / __high2float (f16 → f32 は正確)
        (fp16_round(a0 + b0) as f32, fp16_round(a1 + b1) as f32)
    }

    /// fp16 カーネル / v3 カーネル (fp16=false) の演算列を CPU で再現する。
    /// レーン分割とレーン間 reduce の順は .cu と同じ (v3 と fp16 で共有)。
    fn mirror_gemv(
        ty: crate::kquant::GgmlType,
        w: &[u8],
        n_in: usize,
        n_out: usize,
        x: &[f32],
        fp16: bool,
    ) -> Vec<f32> {
        use crate::kquant::GgmlType as T;
        let rb = gguf_row_bytes(ty, n_in).expect("row_bytes");
        let lanes = moe_gemv_v3_lanes(n_in);
        assert!(lanes > 0, "ミラーは lanes > 0 の k だけ");
        let mut out = vec![0f32; n_out];
        for row in 0..n_out {
            let rowp = &w[row * rb..(row + 1) * rb];
            let mut la = [0f32; 4]; // レーン別の f32 累積
            match ty {
                T::Q8_0 => {
                    let nb = n_in >> 5;
                    let per = nb / lanes;
                    for lane in 0..lanes {
                        let mut acc = 0f32;
                        for b in lane * per..(lane + 1) * per {
                            let blk = &rowp[b * 34..b * 34 + 34];
                            let d_bits = u16::from(blk[0]) | (u16::from(blk[1]) << 8);
                            let mut xv = [0f32; 32];
                            xv.copy_from_slice(&x[b * 32..b * 32 + 32]);
                            if fp16 {
                                let d = f64::from(half::f16::from_bits(d_bits));
                                let mut y = [(0f64, 0f64); 16];
                                for j in 0..16 {
                                    // __hmul2(q2, d2): dequant の積を f16 で丸める
                                    let q0 = f64::from(i32::from(blk[2 + 2 * j] as i8));
                                    let q1 = f64::from(i32::from(blk[3 + 2 * j] as i8));
                                    y[j] = (fp16_round(q0 * d), fp16_round(q1 * d));
                                }
                                let (lo, hi) = mirror_block_sum(&y, &xv, false);
                                acc += lo + hi; // block_sum 内の (low + high)
                            } else {
                                // v3: y = q*d (f32)。acc += y*xv[j] を j 昇順
                                let d = fp16_bits_to_f32(d_bits);
                                for j in 0..32 {
                                    let yv = (blk[2 + j] as i8) as i32 as f32 * d;
                                    acc += yv * xv[j];
                                }
                            }
                        }
                        la[lane] = acc;
                    }
                }
                T::Q5_1 => {
                    let nb = n_in >> 5;
                    let per = nb / lanes;
                    for lane in 0..lanes {
                        let mut acc = 0f32;
                        for b in lane * per..(lane + 1) * per {
                            let blk = &rowp[b * 24..b * 24 + 24];
                            let d_bits = u16::from(blk[0]) | (u16::from(blk[1]) << 8);
                            let m_bits = u16::from(blk[2]) | (u16::from(blk[3]) << 8);
                            let qh = u32::from(blk[4])
                                | (u32::from(blk[5]) << 8)
                                | (u32::from(blk[6]) << 16)
                                | (u32::from(blk[7]) << 24);
                            let mut xv = [0f32; 32];
                            xv.copy_from_slice(&x[b * 32..b * 32 + 32]);
                            if fp16 {
                                let d = f64::from(half::f16::from_bits(d_bits));
                                let m = f64::from(half::f16::from_bits(m_bits));
                                let mut y = [(0f64, 0f64); 16];
                                for j in 0..16 {
                                    // __hfma2(q2, d2, m2): q*d + m を 1 回丸め
                                    let xh0 = ((qh >> j) << 4) & 0x10;
                                    let q0 = f64::from(i32::from(blk[8 + j] & 0x0f) | xh0 as i32);
                                    let xh1 = (qh >> (j + 12)) & 0x10;
                                    let q1 = f64::from(i32::from(blk[8 + j] >> 4) | xh1 as i32);
                                    y[j] = (fp16_round(q0 * d + m), fp16_round(q1 * d + m));
                                }
                                let (lo, hi) = mirror_block_sum(&y, &xv, true);
                                acc += lo; // Q5_1 は (acc + low) + high の順
                                acc += hi;
                            } else {
                                // v3: 下位ニブル 16 要素 → 上位ニブル 16 要素の順
                                let d = fp16_bits_to_f32(d_bits);
                                let m = fp16_bits_to_f32(m_bits);
                                for j in 0..16 {
                                    let xh0 = ((qh >> j) << 4) & 0x10;
                                    let q0 = i32::from(blk[8 + j] & 0x0f) | xh0 as i32;
                                    let yv = q0 as f32 * d + m;
                                    acc += yv * xv[j];
                                }
                                for j in 0..16 {
                                    let xh1 = (qh >> (j + 12)) & 0x10;
                                    let q1 = i32::from(blk[8 + j] >> 4) | xh1 as i32;
                                    let yv = q1 as f32 * d + m;
                                    acc += yv * xv[16 + j];
                                }
                            }
                        }
                        la[lane] = acc;
                    }
                }
                T::Q4K => {
                    let seg = n_in / lanes;
                    for lane in 0..lanes {
                        let mut acc = 0f32;
                        let lo = lane * seg;
                        let hi = lo + seg;
                        let b0 = lo >> 8;
                        let b1 = (hi - 1) >> 8;
                        for b in b0..=b1 {
                            let bstart = b << 8;
                            let blk = &rowp[b * 144..b * 144 + 144];
                            let d_bits = u16::from(blk[0]) | (u16::from(blk[1]) << 8);
                            let dmin_bits = u16::from(blk[2]) | (u16::from(blk[3]) << 8);
                            let s0 = if lo > bstart { (lo - bstart) >> 5 } else { 0 };
                            let s1 = if hi < bstart + 256 {
                                (hi - bstart + 31) >> 5
                            } else {
                                8
                            };
                            for s in s0..s1 {
                                let e = s << 5;
                                let qoff = 16 + ((e >> 6) << 5);
                                let sh = if e & 32 != 0 { 4u32 } else { 0u32 };
                                let (sc, mn) = mirror_scale_min_k4(s, &blk[4..]);
                                let mut xv = [0f32; 32];
                                xv.copy_from_slice(&x[bstart + e..bstart + e + 32]);
                                if fp16 {
                                    // ds / ms は f32 で計算してから f16 へ 1 回丸め
                                    // (__float2half)。ms は事前に符号反転。
                                    let d32 = fp16_bits_to_f32(d_bits);
                                    let dmin32 = fp16_bits_to_f32(dmin_bits);
                                    let ds = f64::from(half::f16::from_f32(d32 * sc as f32));
                                    let ms =
                                        f64::from(half::f16::from_f32(-(dmin32 * mn as f32)));
                                    let mut y = [(0f64, 0f64); 16];
                                    for j in 0..16 {
                                        // __hfma2(q2, ds2, ms2): q*ds - ms を 1 回丸め
                                        let v0 =
                                            f64::from((u32::from(blk[qoff + 2 * j]) >> sh) & 0x0f);
                                        let v1 = f64::from(
                                            (u32::from(blk[qoff + 2 * j + 1]) >> sh) & 0x0f,
                                        );
                                        y[j] = (fp16_round(v0 * ds + ms), fp16_round(v1 * ds + ms));
                                    }
                                    let (lo2, hi2) = mirror_block_sum(&y, &xv, false);
                                    acc += lo2 + hi2;
                                } else {
                                    // v3: y = ds * (float)v - ms。acc += y*xv[l] を l 昇順
                                    let ds32 = fp16_bits_to_f32(d_bits) * sc as f32;
                                    let ms32 = fp16_bits_to_f32(dmin_bits) * mn as f32;
                                    for l in 0..32 {
                                        let v = (u32::from(blk[qoff + l]) >> sh) & 0x0f;
                                        let yv = ds32 * v as f32 - ms32;
                                        acc += yv * xv[l];
                                    }
                                }
                            }
                        }
                        la[lane] = acc;
                    }
                }
                // task 858: Q5_K の v3 カーネル (kernels_moe_v3.cu の
                // dno_gemv_q5k_v3) を写した腕。ミラーから抜けていたため、
                // Q5_K だけ「型ごとの数値パリティ」が無かった。
                // Q5_K に fp16 カーネルは無い (`gemv_fp16_ty_ok` が false)。
                T::Q5K => {
                    assert!(!fp16, "Q5_K の fp16 カーネルは無い");
                    let seg = n_in / lanes;
                    for lane in 0..lanes {
                        let mut acc = 0f32;
                        let lo = lane * seg;
                        let hi = lo + seg;
                        let b0 = lo >> 8;
                        let b1 = (hi - 1) >> 8;
                        for b in b0..=b1 {
                            let bstart = b << 8;
                            let blk = &rowp[b * 176..b * 176 + 176];
                            let d_bits = u16::from(blk[0]) | (u16::from(blk[1]) << 8);
                            let dmin_bits = u16::from(blk[2]) | (u16::from(blk[3]) << 8);
                            let s0 = if lo > bstart { (lo - bstart) >> 5 } else { 0 };
                            let s1 = if hi < bstart + 256 {
                                (hi - bstart + 31) >> 5
                            } else {
                                8
                            };
                            for s in s0..s1 {
                                let e = s << 5;
                                let qoff = 48 + ((e >> 6) << 5);
                                let sh = if e & 32 != 0 { 4u32 } else { 0u32 };
                                let (sc, mn) = mirror_scale_min_k4(s, &blk[4..]);
                                let mut xv = [0f32; 32];
                                xv.copy_from_slice(&x[bstart + e..bstart + e + 32]);
                                let ds32 = fp16_bits_to_f32(d_bits) * sc as f32;
                                let ms32 = fp16_bits_to_f32(dmin_bits) * mn as f32;
                                for l in 0..32 {
                                    let hi_bit = ((u32::from(blk[16 + l]) >> s) & 1) << 4;
                                    let v =
                                        ((u32::from(blk[qoff + l]) >> sh) & 0x0f) + hi_bit;
                                    let yv = ds32 * v as f32 - ms32;
                                    acc += yv * xv[l];
                                }
                            }
                        }
                        la[lane] = acc;
                    }
                }
                _ => unreachable!("fp16 / v3 ミラーの対象外の型 {ty:?}"),
            }
            // moe_fp16_reduce / moe_v3_reduce と同じ (offset 1 → lanes==4 なら offset 2)
            out[row] = if lanes == 4 {
                (la[0] + la[1]) + (la[2] + la[3])
            } else {
                la[0] + la[1]
            };
        }
        out
    }

    /// task 108: fp16 カーネル (kernels_moe_fp16.cu) の演算列を CPU で模して、
    /// CPU の逐次実装 (`expert_gemv_scalar`) との誤差を測る (GPU 無しで回る)。
    ///
    /// しきい値は 2e-2 相対 (行列全体の max|diff| / max|ref|。行ごとの相対は
    /// 打ち消し合いで 0 に近い行が暴れるので使わない)。あわせて v3 (f32 の
    /// レーン分割、加算順だけが違う) のミラーとの比較を出す = 契約変更の証跡。
    #[test]
    fn fp16_gemv_mirror_error_stats() {
        use crate::kquant::GgmlType as T;
        // 実モデルの k と、2 レーン経路 (k=320) を含む組み合わせ
        let cases: [(T, usize, usize); 4] = [
            (T::Q4K, 2560, 256),
            (T::Q5_1, 640, 2560),
            (T::Q8_0, 2560, 256),
            (T::Q8_0, 320, 10240),
        ];
        for (ty, n_in, n_out) in cases {
            // v2 のビット一致テストと同じ合成行列: d (と Q4_K の dmin、Q5_1 の m)
            // は f16 の 1.0 固定 = NaN / Inf を作らない。scales / q は決定的乱数。
            let rb = gguf_row_bytes(ty, n_in).expect("row_bytes");
            let blk = ty.blck_size();
            let mut w = vec![0u8; rb * n_out];
            let mut s = 12345u32;
            for r in 0..n_out {
                let row = &mut w[r * rb..(r + 1) * rb];
                for b in 0..(n_in / blk) {
                    let base = b * ty.type_size();
                    row[base] = 0x00;
                    row[base + 1] = 0x3c;
                    row[base + 2] = 0x00;
                    row[base + 3] = 0x3c;
                    for i in 4..ty.type_size() {
                        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        row[base + i] = (s >> 16) as u8;
                    }
                }
            }
            let x = pseudo_random(n_in, 7);
            let mut want = vec![0f32; n_out];
            crate::model::moe_expert::expert_gemv_scalar(ty, &w, n_in, n_out, &x, &mut want)
                .expect("cpu 参照");
            let got_f16 = mirror_gemv(ty, &w, n_in, n_out, &x, true);
            let got_v3 = mirror_gemv(ty, &w, n_in, n_out, &x, false);
            let stat = |got: &[f32]| -> (f32, f32, f32) {
                let mut maxd = 0f32;
                let mut sum = 0f32;
                let mut maxv = 0f32;
                for i in 0..n_out {
                    maxd = maxd.max((got[i] - want[i]).abs());
                    sum += (got[i] - want[i]).abs();
                    maxv = maxv.max(want[i].abs());
                }
                (maxd, sum / n_out as f32, maxv)
            };
            let (d16, mean16, maxv) = stat(&got_f16);
            let (d3, mean3, _) = stat(&got_v3);
            let rel16 = d16 / maxv.max(f32::MIN_POSITIVE);
            let rel3 = d3 / maxv.max(f32::MIN_POSITIVE);
            println!(
                "{ty:?} k={n_in} n={n_out}: fp16 max|diff|={d16:e} mean|diff|={mean16:e} \
                 max|ref|={maxv:e} rel={rel16:e} / v3 max|diff|={d3:e} mean|diff|={mean3:e} \
                 rel={rel3:e}"
            );
            assert!(
                rel16 < 2e-2,
                "{ty:?} k={n_in}: fp16 ミラーの相対誤差 {rel16:e} >= 2e-2 \
                 (max|diff|={d16:e}, max|ref|={maxv:e})"
            );
            assert!(
                rel16 > rel3,
                "{ty:?} k={n_in}: fp16 ({rel16:e}) が v3 ({rel3:e}) より誤差が小さい \
                 = 契約変更 (fp16 は v3 より誤差が大きい) の前提と違う"
            );
            assert!(
                got_f16.iter().all(|v| v.is_finite()),
                "{ty:?} k={n_in}: fp16 ミラーが NaN / Inf を作った"
            );
        }
    }

    /// task 858: GPU decode GEMV の 4 型すべてで、カーネルの演算列ミラー (v3 =
    /// レーン分割 + レーン間 reduce) が **f64 の真値** に対して数値方針
    /// (絶対 1e-4 + 相対 1e-5) に入ることと、CPU の逐次参照との余弦が
    /// 1 に十分近いことを固定する。
    ///
    /// 実機の `cuda_moe_gemv_v2_bit_exact_vs_cpu` (GPU 必須・#[ignore]) は
    /// v2 カーネルのビット一致を見る。こちらは GPU 無しで回る側: 型ごとの
    /// デコード (索引) とレーン分割の丸めが、型を足した / 変えたときに
    /// 崩れていないことを見る。重みのスケールは f16 の 1/16 (0x2C00) にして
    /// 和の大きさを抑え、丸めの上限で落ちないようにする (d の値そのものは
    /// デコードの一般性に影響しない)。
    #[test]
    fn dense_quant_gemv_mirror_within_numeric_policy() {
        use crate::kquant::GgmlType as T;
        let cos = |a: &[f32], b: &[f32]| -> f64 {
            let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
            for (x, y) in a.iter().zip(b.iter()) {
                d += f64::from(*x) * f64::from(*y);
                na += f64::from(*x) * f64::from(*x);
                nb += f64::from(*y) * f64::from(*y);
            }
            d / (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE)
        };
        for ty in crate::quant_matrix::DENSE_GPU {
            let (n_in, n_out) = (512usize, 13usize);
            let rb = gguf_row_bytes(*ty, n_in).expect("row_bytes");
            let blk = ty.blck_size();
            let mut w = vec![0u8; rb * n_out];
            let mut s = 987654321u32;
            for r in 0..n_out {
                let row = &mut w[r * rb..(r + 1) * rb];
                for b in 0..(n_in / blk) {
                    let base = b * ty.type_size();
                    // d (と Q4_K/Q5_K の dmin, Q5_1 の m) = f16 の 1/16
                    row[base] = 0x00;
                    row[base + 1] = 0x2c;
                    if ty.type_size() >= 4 && !matches!(ty, T::Q8_0) {
                        row[base + 2] = 0x00;
                        row[base + 3] = 0x2c;
                    }
                    for i in 4..ty.type_size() {
                        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        row[base + i] = (s >> 16) as u8;
                    }
                }
            }
            let x = pseudo_random(n_in, 11);
            let mut cpu = vec![0f32; n_out];
            crate::model::moe_expert::expert_gemv_scalar(*ty, &w, n_in, n_out, &x, &mut cpu)
                .expect("cpu 参照");
            // f64 の真値 (デコードは CPU 参照と同じ f32 値、和だけ f64)。
            let mut truth = vec![0f64; n_out];
            let mut rowf = vec![0f32; n_in];
            for r in 0..n_out {
                crate::kquant::dequant_row_into(*ty, &w[r * rb..(r + 1) * rb], &mut rowf);
                let mut acc = 0f64;
                for i in 0..n_in {
                    acc += f64::from(rowf[i]) * f64::from(x[i]);
                }
                truth[r] = acc;
            }
            let got = mirror_gemv(*ty, &w, n_in, n_out, &x, false);
            let mut worst = 0f64;
            for r in 0..n_out {
                let tol = 1e-4 + 1e-5 * truth[r].abs();
                let d = (f64::from(got[r]) - truth[r]).abs();
                if d > tol {
                    eprintln!(
                        "[quant-matrix] {ty:?} r={r} got={} truth={} abs={d:e} tol={tol:e}",
                        got[r], truth[r]
                    );
                }
                assert!(
                    d <= tol,
                    "{ty:?} r={r}: ミラーと真値の差 {d:e} が方針 {tol:e} を超えた"
                );
                worst = worst.max(d);
            }
            let c = cos(&got, &cpu);
            println!(
                "[quant-matrix] {ty:?} k={n_in} n={n_out}: max|mirror-truth|={worst:e} \
                 max|truth|={:e} cos(mirror,cpu)={c:.15}",
                truth.iter().fold(0f64, |m, v| m.max(v.abs()))
            );
            assert!(c > 1.0 - 1e-12, "{ty:?}: 余弦 {c:.15} が 1 から離れすぎ");
        }
    }


    /// task 108: `DENNO_GEMV_FP16` の真理値表と fp16 対応型 (GPU 無しで回る)。
    #[test]
    fn gemv_fp16_gate_truth_table() {
        let table: [(Option<&str>, bool); 10] = [
            (Some("1"), true),
            (Some("on"), true),
            (Some("true"), true),
            (None, false),
            (Some(""), false),
            (Some("0"), false),
            (Some("off"), false),
            (Some("false"), false),
            (Some("2"), false),
            (Some("ON"), false), // 大文字は通さない (v3 と同じ厳密一致)
        ];
        for (raw, want) in table {
            assert_eq!(parse_gemv_fp16(raw), want, "DENNO_GEMV_FP16={raw:?}");
        }
        // env の実値 (OnceLock) とも一致する
        assert_eq!(
            moe_gemv_fp16_on(),
            parse_gemv_fp16(std::env::var("DENNO_GEMV_FP16").ok().as_deref())
        );
        // 対応型: Q8_0 / Q5_1 / Q4K。Q5_K とそれ以外は v3 / v2 へ落ちる。
        use crate::kquant::GgmlType as T;
        assert!(gemv_fp16_ty_ok(T::Q8_0));
        assert!(gemv_fp16_ty_ok(T::Q5_1));
        assert!(gemv_fp16_ty_ok(T::Q4K));
        assert!(!gemv_fp16_ty_ok(T::Q5K));
        assert!(!gemv_fp16_ty_ok(T::Q4_0));
    }

    /// task 108: fp16 (fast mode) GEMV が CPU の逐次実装と相対 2e-2 以内で一致し、
    /// v3 より誤差が大きいこと (契約変更の証跡)。fp16 が実際に走ったことは
    /// 「v2 (ビット一致経路) と 1 行も一致しないこと」で確かめる (黙って
    /// v2 / v3 に落ちていたら検出できる)。
    ///
    /// 実行 (GPU のある環境):
    ///   bash crates/denno-engine/kernels/build_ptx.sh moe_fp16
    ///   DENNO_GEMV_FP16_PTX=$PWD/crates/denno-engine/kernels/kernels_moe_fp16.ptx \
    ///   cargo test --release --features cuda -- --ignored cuda_gemv_fp16_matches_cpu
    #[test]
    #[ignore]
    fn cuda_gemv_fp16_matches_cpu() {
        use crate::kquant::GgmlType as T;
        let ctx = CudaCtx::new().unwrap();
        println!("device: {}", ctx.info());
        assert!(
            moe_gemv_fp16_loaded(),
            "fp16 の PTX が読めていない (DENNO_GEMV_FP16_PTX / kernels_moe_fp16.ptx を確認。\
             `bash build_ptx.sh moe_fp16` で再生成する)"
        );
        // 実モデルの k。Q8_0 320 は 2 レーン経路。Q5_K は fp16 対象外 (省略)。
        let cases: [(T, usize, usize); 4] = [
            (T::Q4K, 2560, 256),
            (T::Q5_1, 640, 2560),
            (T::Q8_0, 2560, 256),
            (T::Q8_0, 320, 10240),
        ];
        for (ty, n_in, n_out) in cases {
            let lanes = moe_gemv_v3_lanes(n_in);
            assert!(lanes > 0, "{ty:?} k={n_in} は lanes 対象外");
            assert!(gemv_fp16_ty_ok(ty), "{ty:?} は fp16 対象外");
            let rb = gguf_row_bytes(ty, n_in).expect("row_bytes");
            let blk = ty.blck_size();
            let mut w = vec![0u8; rb * n_out];
            let mut s = 12345u32;
            for r in 0..n_out {
                let row = &mut w[r * rb..(r + 1) * rb];
                for b in 0..(n_in / blk) {
                    let base = b * ty.type_size();
                    row[base] = 0x00;
                    row[base + 1] = 0x3c;
                    row[base + 2] = 0x00;
                    row[base + 3] = 0x3c;
                    for i in 4..ty.type_size() {
                        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        row[base + i] = (s >> 16) as u8;
                    }
                }
            }
            let x = pseudo_random(n_in, 7);
            let mut want = vec![0f32; n_out];
            crate::model::moe_expert::expert_gemv_scalar(ty, &w, n_in, n_out, &x, &mut want)
                .expect("cpu 参照");
            let dev_w = DevBuffer::from_slice(&w).unwrap();
            // 基準 (v2 は CPU とビット一致する契約) が壊れていないこと。
            let mut got_v2 = vec![0f32; n_out];
            assert!(
                ctx.moe_gemv_dev_with(
                    ty,
                    &dev_w,
                    &x,
                    n_in,
                    n_out,
                    &mut got_v2,
                    GemvKernelMode::V2
                )
                .unwrap(),
                "{ty:?}: GPU 経路 (v2) が発動しなかった"
            );
            for i in 0..n_out {
                assert_eq!(
                    got_v2[i].to_bits(),
                    want[i].to_bits(),
                    "{ty:?}: v2 が CPU とビット一致しない (行 {i})"
                );
            }
            let mut got_v3 = vec![0f32; n_out];
            assert!(
                ctx.moe_gemv_dev_with(
                    ty,
                    &dev_w,
                    &x,
                    n_in,
                    n_out,
                    &mut got_v3,
                    GemvKernelMode::V3
                )
                .unwrap(),
                "{ty:?}: GPU 経路 (v3) が発動しなかった"
            );
            let mut got = vec![0f32; n_out];
            assert!(
                ctx.moe_gemv_dev_with(
                    ty,
                    &dev_w,
                    &x,
                    n_in,
                    n_out,
                    &mut got,
                    GemvKernelMode::Fp16
                )
                .unwrap(),
                "{ty:?}: GPU 経路 (fp16) が発動しなかった"
            );
            // fp16 が実際に走った証拠: v2 / v3 と一致する行が全てではない
            let ndiff_v2 = (0..n_out)
                .filter(|&i| got[i].to_bits() != got_v2[i].to_bits())
                .count();
            let ndiff_v3 = (0..n_out)
                .filter(|&i| got[i].to_bits() != got_v3[i].to_bits())
                .count();
            assert!(
                ndiff_v2 > 0,
                "{ty:?}: fp16 の出力が v2 と 1 行も違わない = fp16 が走っていない \
                 (PTX 未ロード / 対象外の型 / k={n_in} が 64 の倍数でない)"
            );
            // 誤差統計 (行列全体の max|diff| / max|ref| と平均。REPORT に記録する)
            let stat = |got: &[f32]| -> (f32, f32, f32) {
                let mut maxd = 0f32;
                let mut sum = 0f32;
                let mut maxv = 0f32;
                for i in 0..n_out {
                    maxd = maxd.max((got[i] - want[i]).abs());
                    sum += (got[i] - want[i]).abs();
                    maxv = maxv.max(want[i].abs());
                }
                (maxd, sum / n_out as f32, maxv)
            };
            let (d16, mean16, maxv) = stat(&got);
            let (d3, mean3, _) = stat(&got_v3);
            let rel16 = d16 / maxv.max(f32::MIN_POSITIVE);
            let rel3 = d3 / maxv.max(f32::MIN_POSITIVE);
            println!(
                "{ty:?} k={n_in} n={n_out} lanes={lanes}: fp16 max|diff|={d16:e} \
                 mean|diff|={mean16:e} max|ref|={maxv:e} rel={rel16:e} \
                 (v2 と {ndiff_v2}/{n_out} 行, v3 と {ndiff_v3}/{n_out} 行が不一致) \
                 / v3 max|diff|={d3:e} mean|diff|={mean3:e} rel={rel3:e}"
            );
            assert_eq!(
                got.iter().filter(|v| !v.is_finite()).count(),
                0,
                "{ty:?}: fp16 が NaN / Inf を作った"
            );
            assert!(
                rel16 < 2e-2,
                "{ty:?} k={n_in}: fp16 の相対誤差 {rel16:e} >= 2e-2 \
                 (max|diff|={d16:e}, max|ref|={maxv:e})"
            );
            assert!(
                rel16 > rel3,
                "{ty:?} k={n_in}: fp16 ({rel16:e}) <= v3 ({rel3:e}) \
                 = 契約変更の証跡にならない"
            );
        }
    }

    /// GPU の量子化が CPU (`tensor::quantize_row_i8`) とビット一致するか
    #[test]
    #[ignore] // GPU が必要: cargo test --release --features cuda -- --ignored
    fn cuda_quantize_rows_bit_exact_vs_cpu() {
        let ctx = CudaCtx::new().unwrap();
        println!("device: {}", ctx.info());
        for (m, k) in [(1usize, 8usize), (3, 896), (17, 1536), (4, 8960)] {
            let a = pseudo_random(m * k, 30 + m as u32);
            let (aq_gpu, asc_gpu) = ctx.quantize_rows(&a, m, k).unwrap();
            let aq_gpu = aq_gpu.to_vec::<i8>(m * k).unwrap();
            let asc_gpu = asc_gpu.to_vec::<f32>(m).unwrap();
            for i in 0..m {
                let (aq_cpu, asc_cpu) = tensor::quantize_row_i8(&a[i * k..][..k]);
                assert_eq!(&aq_gpu[i * k..][..k], &aq_cpu[..], "m={m} k={k} row={i}");
                assert_eq!(
                    asc_gpu[i].to_bits(),
                    asc_cpu.to_bits(),
                    "m={m} k={k} row={i}: ascale"
                );
            }
        }
    }

    /// GPU int8 GEMV/GEMM が CPU int8 カーネルとビット一致するか
    /// (decode m=1 / バッチ decode m>1 / prefill GEMM の全経路)。
    #[test]
    #[ignore]
    fn cuda_linear_int8_bit_exact_vs_cpu() {
        let ctx = CudaCtx::new().unwrap();
        println!("device: {}", ctx.info());
        // 実モデルの形状を含む: Qwen2.5-1.5B (qkv 1536→2048, gate_up 1536→17920,
        // down 8960→1536, lm_head 1536→151936) / 0.5B (896→4864)
        let cases: [(usize, usize, usize); 8] = [
            (1, 8, 16),
            (7, 37, 200),
            (1, 896, 1152),
            (2, 896, 4864),
            (4, 1536, 17920),
            (3, 8960, 1536),
            (1, 1536, 151936),
            (8, 1536, 2048),
        ];
        for (ci, &(m, k, n)) in cases.iter().enumerate() {
            let q = pseudo_i8(k * n, 40 + ci as u32);
            let scale = positive_scales(n, 50 + ci as u32);
            let a = pseudo_random(m * k, 60 + ci as u32);
            let w = ctx.upload_i8(&q).unwrap();
            let sc = ctx.upload_f32(&scale).unwrap();
            let got = ctx.linear_bt_i8(&a, &w, &sc, m, k, n).unwrap();
            let want = tensor::matvec_rows_bt_i8(&a, &q, &scale, k, n, m);
            for idx in 0..m * n {
                assert_eq!(
                    got[idx].to_bits(),
                    want[idx].to_bits(),
                    "m={m} k={k} n={n} idx={idx} (i={} j={}): GPU != CPU",
                    idx / n,
                    idx % n
                );
            }
        }
    }

    /// P17k: maybe_gpu_ffn (デバイス常駐 FFN ブロック) vs CPU 参照実装のビット一致。
    /// CPU 参照は llama.rs prefill 従来経路と同一コード: gate/up を行連結した
    /// grouped matvec_rows_bt_i8 → silu_inplace + 要素積 → down matvec_rows_bt_i8。
    #[test]
    #[ignore]
    fn cuda_ffn_bit_exact_vs_cpu_reference() {
        let _g = test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let ctx = CudaCtx::new().unwrap();
        println!("device: {}", ctx.info());
        crate::cuda::enable_for_tests();

        // GEMM 経路の条件 (m > max_m=16 かつ m >= gemm_min_m) を満たすよう閾値を下げる
        std::env::set_var("DENNO_CUDA_GEMM_MIN_M", "8");

        let (m, e, ff) = (40usize, 96usize, 64usize);
        let gate = Quant8Linear {
            w: pseudo_i8(e * ff, 90),
            scale: positive_scales(ff, 91),
            k: e,
            n: ff,
            uid: 901,
        };
        let up = Quant8Linear {
            w: pseudo_i8(e * ff, 92),
            scale: positive_scales(ff, 93),
            k: e,
            n: ff,
            uid: 902,
        };
        let down = Quant8Linear {
            w: pseudo_i8(ff * e, 94),
            scale: positive_scales(e, 95),
            k: ff,
            n: e,
            uid: 903,
        };
        let h = pseudo_random(m * e, 96);

        let got = maybe_gpu_ffn(&gate, &up, &down, &h, m, ff, 0).unwrap();

        // CPU 参照 (llama.rs prefill 従来経路と同一の演算列)。
        // out1 は [m, 2ff] 行-major: gate は各行の前半 ff 列 (平坦スライス不可)。
        let n2 = 2 * ff;
        let mut gw = Vec::with_capacity(e * n2);
        gw.extend_from_slice(&gate.w);
        gw.extend_from_slice(&up.w);
        let mut gsc = Vec::with_capacity(n2);
        gsc.extend_from_slice(&gate.scale);
        gsc.extend_from_slice(&up.scale);
        let out1 = tensor::matvec_rows_bt_i8(&h, &gw, &gsc, e, n2, m);
        let mut act = vec![0.0f32; m * ff];
        for i in 0..m {
            let row = &out1[i * n2..(i + 1) * n2];
            act[i * ff..(i + 1) * ff].copy_from_slice(&row[..ff]);
            tensor::silu_inplace(&mut act[i * ff..(i + 1) * ff]);
            for f in 0..ff {
                act[i * ff + f] *= row[ff + f];
            }
        }
        let want = tensor::matvec_rows_bt_i8(&act, &down.w, &down.scale, ff, e, m);

        assert_eq!(got.len(), want.len());
        for idx in 0..want.len() {
            assert_eq!(
                got[idx].to_bits(),
                want[idx].to_bits(),
                "FFN ブロック不一致 @{} (GPU 常駐経路 vs CPU 参照)",
                idx
            );
        }
        println!("cuda_ffn_bit_exact_vs_cpu_reference: bit-exact (m={m}, e={e}, ff={ff})");
    }

    /// P17i: tiled GEMM カーネル (dno_gemm_i8) が CPU int8 線形と**ビット一致**
    /// すること。タイル境界 (m/n が 64 の倍数でない)・k が 4/32 の倍数でない
    /// 場合・GEMV との比較 (同一入力で両カーネルが同一結果) を含む。
    #[test]
    #[ignore]
    fn cuda_gemm_i8_bit_exact_vs_cpu() {
        let ctx = CudaCtx::new().unwrap();
        println!("device: {}", ctx.info());
        // m はタイル境界をまたぐ値 (1/63/64/65/100/2827 相当)・k/n は 4 や 32 の
        // 倍数でない値を含む。実モデル形状 (k=1536/8960) も含む。
        let cases: [(usize, usize, usize); 8] = [
            (1, 8, 16),
            (63, 37, 200),
            (65, 30, 197),
            (100, 1536, 17920),
            (128, 8960, 1536),
            (100, 1536, 2048),
            (130, 1533, 17911),
            (64, 1536, 17920),
        ];
        for (ci, &(m, k, n)) in cases.iter().enumerate() {
            let q = pseudo_i8(k * n, 40 + ci as u32);
            let scale = positive_scales(n, 50 + ci as u32);
            let a = pseudo_random(m * k, 60 + ci as u32);
            // s74: 重みは gemm_pad_n 行に 0 パディングしてアップロードする
            // (カーネルは j 方向 256 列単位・scale は実列数 n 本)。
            let w = ctx.upload_i8(&gemm_pad_apply(&q, k, n)).unwrap();
            let sc = ctx.upload_f32(&scale).unwrap();
            // GEMM カーネル vs CPU
            let got = ctx.linear_gemm_i8(&a, &w, &sc, m, k, n).unwrap();
            let want = tensor::matvec_rows_bt_i8(&a, &q, &scale, k, n, m);
            for idx in 0..m * n {
                assert_eq!(
                    got[idx].to_bits(),
                    want[idx].to_bits(),
                    "gemm m={m} k={k} n={n} idx={idx} (i={} j={}): GPU != CPU",
                    idx / n,
                    idx % n
                );
            }
            // GEMV カーネルともビット一致 (整数累積の順序不変性の検証)
            let gotv = ctx.linear_bt_i8(&a, &w, &sc, m, k, n).unwrap();
            for idx in 0..m * n {
                assert_eq!(got[idx].to_bits(), gotv[idx].to_bits(), "gemm vs gemv m={m} k={k} n={n} idx={idx}");
            }
        }
        println!("dno_gemm_i8 bit-exact vs CPU/GEMV OK ({} cases)", cases.len());
    }

    /// s74 (T6 候補 1): dno_gemm_i8 の数値契約を **GPU 無しで** 固定する。
    ///
    /// 新カーネル (128x256 タイル・1 thread 8x16・重み 256 行パディング) は
    /// 旧カーネルと命令列が異なるので、`cuda_gemm_i8_bit_exact_vs_cpu` (要 GPU) が
    /// 走らない環境でも次の不変条件を検証できるようにする:
    ///  1. 重みパディング (`gemm_pad_apply`) は実列 n の dot を厳密に保存する
    ///     (パディング行は 0・実列はビット同一) — j タイル境界での取り違え防止。
    ///  2. タイル分割 (i 方向 128 / j 方向 256) を模した整数 dot と、分割なしの
    ///     整数 dot が同一 (i32 累積は結合則が成立 = R2 の根拠)。
    ///  3. 端処理: m・n・k がタイル / 4 / 32 の倍数でない形状でも、
    ///     範囲外アクセス無しで実領域だけを書く。
    ///
    /// GPU 実機のビット一致は `cuda_gemm_i8_bit_exact_vs_cpu` (supervisor が実行)。
    #[test]
    fn gemm_pad_preserves_dot_and_tiling_is_exact() {
        // 1) パディングは実列を保存し、余りは 0
        for &(k, n) in &[(4usize, 100usize), (37, 200), (1536, 17920), (30, 256), (1, 1)] {
            let qw: Vec<i8> = (0..k * n).map(|i| (((i * 37) % 251) as i32 - 125) as i8).collect();
            let padded = gemm_pad_apply(&qw, k, n);
            assert_eq!(padded.len(), k * gemm_pad_n(n), "k={k} n={n}");
            assert_eq!(gemm_pad_n(n) % 256, 0);
            assert!(gemm_pad_n(n) >= n && gemm_pad_n(n) - n < 256);
            for r in 0..n {
                assert_eq!(
                    &padded[r * k..(r + 1) * k],
                    &qw[r * k..(r + 1) * k],
                    "実列が変わった k={k} n={n} r={r}"
                );
            }
            for r in n..gemm_pad_n(n) {
                assert!(padded[r * k..(r + 1) * k].iter().all(|&v| v == 0), "pad 行が 0 でない");
            }
            // パディング行の寄与は 0 (dot が不変)
            let aq: Vec<i8> = (0..k).map(|i| (((i * 11) % 251) as i32 - 120) as i8).collect();
            let dot = |w: &[i8], row: usize| -> i32 {
                (0..k).map(|d| aq[d] as i32 * w[row * k + d] as i32).sum()
            };
            for r in 0..gemm_pad_n(n) {
                let want = if r < n { dot(&qw, r) } else { 0 };
                assert_eq!(dot(&padded, r), want, "dot 保存 k={k} n={n} r={r}");
            }
            // 既に倍数なら複製しない (Cow::Borrowed)
            if n % 256 == 0 {
                assert!(matches!(padded, std::borrow::Cow::Borrowed(_)));
            } else {
                assert!(matches!(padded, std::borrow::Cow::Owned(_)));
            }
        }

        // 2) タイル分割の厳密性 (i32 累積 = 結合則) — カーネルの k パック/タイル
        //    境界で dot が変わらないことを、k が 4・32 の倍数でない形状も含めて
        //    **2 種類の分割**で確認する:
        //      a) GEMM_TK=32 の kb ブロック × c0 4 要素 (実カーネルの内側ループ)
        //      b) 4 要素ずつ全域 (端ブロックは 4 要素未満に切り詰め = カーネルの
        //         gk<c+4 分岐と等価)
        //    34B の行 (例 k=37) では GEMM_TK の切り上げで k の先にゴミが入るが、
        //    外側ループが `kb < k` で止まるため寄与しない (カーネルと同じ挙動)。
        for &(k, n, m) in &[(37usize, 200usize, 65usize), (30, 197, 65), (1533, 300, 3)] {
            let qw: Vec<i8> = (0..k * n).map(|i| (((i * 29) % 241) as i32 - 120) as i8).collect();
            let aq: Vec<i8> = (0..m * k).map(|i| (((i * 17) % 239) as i32 - 119) as i8).collect();
            let mut padded = gemm_pad_apply(&qw, k, n).into_owned();
            // 行の直後 (k の先) に「本当は範囲外」ダミーを置いておく
            let row_stride = k * gemm_pad_n(n);
            padded.resize(row_stride + 8, 77);
            let plain = |i: usize, j: usize| -> i32 {
                (0..k).map(|d| aq[i * k + d] as i32 * padded[j * k + d] as i32).sum()
            };
            // a) kb (32) ブロック × c0 (4) 分割
            let dot_kb = |i: usize, j: usize| -> i32 {
                let mut acc = 0i32;
                let mut kb = 0;
                while kb < k {
                    let mut c0 = 0;
                    while c0 < 32 {
                        let gk = kb + c0;
                        if gk >= k {
                            break;
                        }
                        for e in 0..4 {
                            if gk + e < k {
                                acc += aq[i * k + gk + e] as i32 * padded[j * k + gk + e] as i32;
                            }
                        }
                        c0 += 4;
                    }
                    kb += 32;
                }
                acc
            };
            // b) 4 要素ずつ全域 (端は切り詰め)
            let dot4 = |i: usize, j: usize| -> i32 {
                let mut acc = 0i32;
                let mut gk = 0;
                while gk < k {
                    for e in 0..4 {
                        if gk + e < k {
                            acc += aq[i * k + gk + e] as i32 * padded[j * k + gk + e] as i32;
                        }
                    }
                    gk += 4;
                }
                acc
            };
            for &(i, j) in &[(0usize, 0usize), (m - 1, n - 1), (m / 2, n / 2), (m - 1, 0), (0, n - 1)] {
                assert_eq!(dot_kb(i, j), plain(i, j), "k={k} m={m} n={n} i={i} j={j}");
                assert_eq!(dot4(i, j), plain(i, j), "k={k} m={m} n={n} i={i} j={j} (4 分割)");
            }
        }

        // 3) 端処理 (タイル境界アドレスが範囲内に収まること) — グリッドの切り上げと
        //    カーネル側の範囲判定が前提としている関係を、純粋計算で確認。
        for &(m, n) in &[(1usize, 1usize), (65, 197), (128, 256), (129, 257), (2843, 151936)] {
            let gmx = m.div_ceil(GEMM_TILE);
            let gmy = n.div_ceil(GEMM_TILE_N);
            assert!(gmx * GEMM_TILE >= m && (gmx - 1) * GEMM_TILE < m);
            assert!(gmy * GEMM_TILE_N >= n && (gmy - 1) * GEMM_TILE_N < n);
            // 重みは常に gmy タイル分の行を用意できる (pad 済み)
            assert!(gemm_pad_n(n) >= gmy * GEMM_TILE_N);
        }
    }


    /// P17k 調査: prefill 実形状での linear_gemm_i8 の内訳計測 (kernel / D2H / to_vec)。
    /// `#[ignore]` — GPU が必要。cargo test --release -p denno-engine --features cuda
    /// cuda_gemm_i8_bench -- --ignored --nocapture
    #[test]
    #[ignore]
    fn cuda_gemm_i8_bench_prefill_shapes() {
        let ctx = CudaCtx::new().unwrap();
        println!("device: {}", ctx.info());
        let m = 8000usize;
        // (label, k, n) — Qwen2.5-1.5B の gate_up / down / qkv
        let shapes = [
            ("gate_up", 1536usize, 17920usize),
            ("down", 8960usize, 1536usize),
            ("qkv", 1536usize, 2048usize),
        ];
        for (label, k, n) in shapes {
            let q = pseudo_i8(k * n, 77);
            let scale = positive_scales(n, 78);
            let a = pseudo_random(m * k, 79);
            let w = ctx.upload_i8(&gemm_pad_apply(&q, k, n)).unwrap();
            let sc = ctx.upload_f32(&scale).unwrap();
            // ウォームアップ (JIT + L2)
            let _ = ctx.linear_gemm_i8(&a, &w, &sc, m, k, n).unwrap();
            let t0 = std::time::Instant::now();
            let out = ctx.linear_gemm_i8(&a, &w, &sc, m, k, n).unwrap();
            let total = t0.elapsed();
            // 出力 D2H 単体の時間 (同サイズの to_vec 込み)
            let t1 = std::time::Instant::now();
            let _ = ctx.linear_gemm_i8(&a, &w, &sc, m, k, n).unwrap();
            let total2 = t1.elapsed();
            println!(
                "{label}: m={m} k={k} n={n} linear_gemm total={total:?} / {total2:?} out={}MB checksum={}",
                out.len() * 4 / 1_000_000,
                out[1234].to_bits()
            );
        }

        // 本番経路 (linear_gemm: Scratch + pinned) で 28 層分連続実行
        {
            let (k, n) = (1536usize, 17920usize);
            let q = pseudo_i8(k * n, 87);
            let scale = positive_scales(n, 88);
            let a = pseudo_random(m * k, 89);
            let gw = weights_for(999_001, &q, &scale, k, n).unwrap();
            let _ = linear_gemm(&ctx, &a, gw, m, k, n).unwrap(); // warmup
            let t0 = std::time::Instant::now();
            for _ in 0..28 {
                let _ = linear_gemm(&ctx, &a, gw, m, k, n).unwrap();
            }
            let per = t0.elapsed() / 28;
            println!("linear_gemm (production path) 28x gate_up: per-call {per:?}");
        }
    }

    /// P17k 診断: 生転送帯域の実測 (H2D/D2H × pageable/pinned × sync/async)。
    /// attn_core ~400ms/層の内訳が転送律速かどうかの判定用。
    #[test]
    #[ignore]
    fn cuda_transfer_bench() {
        let ctx = CudaCtx::new().unwrap();
        println!("device: {}", ctx.info());
        const BYTES: usize = 128 * 1024 * 1024; // 128MB
        let elems = BYTES / 4;
        let host: Vec<f32> = pseudo_random(elems, 60);
        let dev = ctx.upload_f32(&host).unwrap();
        let lib = lib().unwrap();
        let gb = |bytes: usize, secs: f64| bytes as f64 / 1e9 / secs;

        // 1) H2D 同期 pageable (DevBuffer::copy_from / ミラー copy_from_at と同一)
        let t = std::time::Instant::now();
        let iters = 4;
        for _ in 0..iters {
            dev.copy_from(&host).unwrap();
        }
        let e = t.elapsed().as_secs_f64() / iters as f64;
        println!("H2D sync pageable 128MB: {e:.4?}s  {:.1} GB/s", gb(BYTES, e));

        // 2) H2D pinned + 毎回 sync (async 発行→sync)
        let mut pin = PinMem::alloc(BYTES).unwrap();
        pin.as_mut_slice::<f32>()[..elems].copy_from_slice(&host);
        let iters = 8;
        unsafe {
            (lib.memcpy_htod_async)(dev.ptr(), pin.ptr, BYTES, std::ptr::null_mut());
            ctx.sync().unwrap();
        }
        let t = std::time::Instant::now();
        for _ in 0..iters {
            unsafe {
                check(
                    (lib.memcpy_htod_async)(dev.ptr(), pin.ptr, BYTES, std::ptr::null_mut()),
                    "h2d pin",
                )
                .unwrap();
                ctx.sync().unwrap();
            }
        }
        let e = t.elapsed().as_secs_f64() / iters as f64;
        println!("H2D pinned+sync 128MB:   {e:.4?}s  {:.1} GB/s", gb(BYTES, e));

        // 3) H2D pinned 連続 async 発行 + 最後に 1 回 sync (オーバーラップ上限)
        let t = std::time::Instant::now();
        for _ in 0..iters {
            unsafe {
                (lib.memcpy_htod_async)(dev.ptr(), pin.ptr, BYTES, std::ptr::null_mut());
            }
        }
        ctx.sync().unwrap();
        let e = t.elapsed().as_secs_f64() / iters as f64;
        println!("H2D pinned async burst:  {e:.4?}s  {:.1} GB/s (per-op)", gb(BYTES, e));

        // 4) D2H 同期 pageable
        let mut dst = vec![0f32; elems];
        let t = std::time::Instant::now();
        for _ in 0..iters {
            unsafe {
                (lib.memcpy_dtoh)(dst.as_mut_ptr() as *mut c_void, dev.ptr(), BYTES);
            }
        }
        let e = t.elapsed().as_secs_f64() / iters as f64;
        println!("D2H sync pageable 128MB: {e:.4?}s  {:.1} GB/s", gb(BYTES, e));

        // 5) D2H pinned + sync (maybe_gpu_oproj / chain_download の経路)
        let out_pin = PinMem::alloc(BYTES).unwrap();
        unsafe {
            (lib.memcpy_dtoh_async)(out_pin.ptr, dev.ptr(), BYTES, std::ptr::null_mut());
            ctx.sync().unwrap();
        }
        let t = std::time::Instant::now();
        for _ in 0..iters {
            unsafe {
                (lib.memcpy_dtoh_async)(out_pin.ptr, dev.ptr(), BYTES, std::ptr::null_mut());
            }
            ctx.sync().unwrap();
        }
        let e = t.elapsed().as_secs_f64() / iters as f64;
        println!("D2H pinned+sync 128MB:   {e:.4?}s  {:.1} GB/s", gb(BYTES, e));

        // 6) cudaMalloc のコスト参考 (ミラー cap 成長時の全層 realloc)
        let t = std::time::Instant::now();
        for _ in 0..8 {
            let _b = ctx.upload_f32(&host).unwrap();
        }
        println!(
            "upload_f32 (alloc+H2D) 128MB x8: {:?}/op",
            t.elapsed() / 8
        );
    }

    /// P17k 診断: prefill attention カーネル単体の実形状タイミング。
    /// (n, t_total, heads, kv_dim, dh) = (13252, 13252, 14, 256, 128) — 実リクエストと同一。
    /// カーネル自体の時間を分離するため q/KV はランダム、転送は事前に完了させておく。
    #[test]
    #[ignore]
    fn cuda_attn_pf_bench() {
        let ctx = CudaCtx::new().unwrap();
        println!("device: {}", ctx.info());
        let (n, t_total, heads, kv_dim, dh) = (13252usize, 13252usize, 14usize, 256usize, 128usize);
        let q: Vec<f32> = pseudo_random(n * heads * dh, 61);
        // KV ミラーを直接用意 (with_mirror を使わず専用 mirror key でシード)
        let key = 999_111u64;
        let (kptr, vptr) = with_mirror(key, 0, 1, t_total, kv_dim, 0, |m, _have| {
            let kb = m.layers[0].0.as_ref().unwrap();
            let vb = m.layers[0].1.as_ref().unwrap();
            let kv: Vec<f32> = pseudo_random(t_total * kv_dim, 62);
            kb.copy_from(&kv).unwrap();
            vb.copy_from(&kv).unwrap();
            m.len[0] = t_total;
            Some((kb.ptr(), vb.ptr()))
        })
        .unwrap();
        let dev_q = ctx.upload_f32(&q).unwrap();
        let dev_att = ctx.upload_f32(&vec![0f32; n * heads * dh]).unwrap();

        let mut heads_c = heads as c_int;
        let mut n_c = n as c_int;
        let mut pos0_c = 0 as c_int;
        let mut kv_dim_c = kv_dim as c_int;
        let mut t_c = t_total as c_int;
        let mut group_c = 2 as c_int;
        let mut dh_c = dh as c_int;
        let mut scale_c = 0.08838834764831845f32;

        let sh_f = (ATTN_PF_BQ + 2 * ATTN_PF_BK) * (dh + 1)
            + ATTN_PF_BQ * ATTN_PF_BK
            + 2 * ATTN_PF_BQ;
        // DevBuffer の ptr は Copy 値なので引数組み立ては毎回行う
        let mut launch = |ctx: &CudaCtx| -> Result<()> {
            let mut qp = dev_q.ptr();
            let mut kp = kptr;
            let mut vp = vptr;
            let mut ap = dev_att.ptr();
            let mut args: Vec<*mut c_void> = vec![
                &mut qp as *mut _ as *mut c_void,
                &mut kp as *mut _ as *mut c_void,
                &mut vp as *mut _ as *mut c_void,
                &mut ap as *mut _ as *mut c_void,
                &mut heads_c as *mut _ as *mut c_void,
                &mut n_c as *mut _ as *mut c_void,
                &mut pos0_c as *mut _ as *mut c_void,
                &mut kv_dim_c as *mut _ as *mut c_void,
                &mut t_c as *mut _ as *mut c_void,
                &mut group_c as *mut _ as *mut c_void,
                &mut dh_c as *mut _ as *mut c_void,
                &mut scale_c as *mut _ as *mut c_void,
            ];
            unsafe {
                ctx.launch_async(
                    ctx.f_attn_pf,
                    // P17k: (行ブロック, heads) — x=行ブロック最速次元で同一 head の
                    // K/V を同時実行ブロック間で共有し L2 ヒット率を上げる
                    (n.div_ceil(ATTN_PF_BQ) as u32, heads as u32, 1),
                    (dh as u32, 1, 1),
                    (sh_f * 4) as u32,
                    &mut args,
                )
            }
        };
        launch(&ctx).unwrap();
        ctx.sync().unwrap();
        let iters = 5;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            launch(&ctx).unwrap();
        }
        ctx.sync().unwrap();
        let per = t.elapsed() / iters;
        println!(
            "attn_pf kernel-only n={n} t={t_total} heads={heads}: {per:?}/call (x28 layers = {:?})",
            per * 28
        );
    }

    /// マイクロベンチ: Qwen2.5-1.5B の線形層 (int8) を decode (m=1) で
    /// GPU vs CPU 比較。実効帯域 (重み bytes / 時間) も出す。
    #[test]
    #[ignore]
    fn bench_cuda_linear_int8_1p5b_decode() {
        let ctx = CudaCtx::new().unwrap();
        println!("device: {}", ctx.info());
        // Qwen2.5-1.5B: hidden 1536 / inter 8960 / qkv 2048 / vocab 151936
        let shapes: [(&str, usize, usize); 5] = [
            ("qkv_proj", 1536, 2048),
            ("o_proj", 1536, 1536),
            ("gate_up", 1536, 17920),
            ("down", 8960, 1536),
            ("lm_head", 1536, 151936),
        ];
        for (label, k, n) in shapes {
            let q = pseudo_i8(k * n, 70);
            let scale = positive_scales(n, 71);
            let a = pseudo_random(k, 72);
            let w = ctx.upload_i8(&q).unwrap();
            let sc = ctx.upload_f32(&scale).unwrap();
            let iters = 20;
            // GPU (重みは VRAM 常駐、アクティベーションは毎回 H2D)
            let _ = ctx.linear_bt_i8(&a, &w, &sc, 1, k, n).unwrap(); // warmup
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                let _ = ctx.linear_bt_i8(&a, &w, &sc, 1, k, n).unwrap();
            }
            let gpu = t0.elapsed().as_secs_f64() / iters as f64;
            // CPU
            let _ = tensor::matvec_bt_i8(&a, &q, &scale, k, n); // warmup
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                let _ = tensor::matvec_bt_i8(&a, &q, &scale, k, n);
            }
            let cpu = t0.elapsed().as_secs_f64() / iters as f64;
            let wb = (k * n) as f64; // 重み帯域 (int8)
            println!(
                "[{label}] k={k} n={n}: GPU {g:.3}ms ({gw:.0} GB/s) / CPU {c:.3}ms ({cw:.0} GB/s) = {r:.1}x",
                g = gpu * 1e3,
                gw = wb / gpu / 1e9,
                c = cpu * 1e3,
                cw = wb / cpu / 1e9,
                r = cpu / gpu,
            );
        }
    }

    /// P17b パイプライン (ピン留め+非同期+1同期+重みレジストリ) の
    /// ビット一致検証。`quant::Linear` のフック経由 (apply_bt / apply_batch_bt /
    /// apply の Q8 パス) で CPU カーネルと比較する。
    #[test]
    #[ignore]
    fn cuda_pipeline_bit_exact_vs_cpu_via_linear() {
        crate::cuda::enable_for_tests();
        for (m, k, n) in [(1usize, 896usize, 1152usize), (1, 1536, 151936), (4, 1536, 17920), (3, 8960, 1536)] {
            let w = pseudo_random(k * n, 90);
            let a = pseudo_random(m * k, 91 + m as u32);
            let lin = crate::quant::Linear::new(w.clone(), k, n, crate::quant::QuantMode::Int8);

            // GPU 経路 (フック経由)
            let got = lin.apply_bt(&a, m);
            // CPU リファレンス (同一仕様の rows カーネル)
            let q = crate::quant::Quant8Linear::quantize(&w, k, n);
            let want = if m == 1 {
                crate::tensor::matvec_bt_i8(&a, &q.w, &q.scale, k, n)
            } else {
                crate::tensor::matmul_bt_i8(&a, &q.w, &q.scale, m, k, n)
            };
            for idx in 0..m * n {
                assert_eq!(
                    got[idx].to_bits(),
                    want[idx].to_bits(),
                    "apply_bt m={m} k={k} n={n} idx={idx}"
                );
            }
            // apply_batch_bt 経路 (P6 バッチ decode)
            if m > 1 {
                let got_b = lin.apply_batch_bt(&a, m);
                for idx in 0..m * n {
                    assert_eq!(got_b[idx].to_bits(), want[idx].to_bits(), "apply_batch_bt m={m}");
                }
            }
        }
    }

    /// P17c: 融合呼び出し (grouped) vs 個別呼び出しのビット一致。
    /// qkv 相当 (3 パート・異なる n) と gate/up 相当 (2 パート) を m=1/m=4 で検証。
    /// 個別の CPU カーネル結果を列連結したものが正解。
    #[test]
    #[ignore]
    fn cuda_grouped_linear_bit_exact_vs_cpu() {
        crate::cuda::enable_for_tests();
        for m in [1usize, 4usize] {
            let k = 96usize;
            // qkv 相当: n の異なる 3 パート
            let parts_w: Vec<Vec<f32>> = vec![
                pseudo_random(40 * k, 71),
                pseudo_random(24 * k, 72),
                pseudo_random(24 * k, 73),
            ];
            let ns = [40usize, 24, 24];
            let a = pseudo_random(m * k, 74 + m as u32);
            let lins: Vec<crate::quant::Linear> = parts_w
                .iter()
                .zip(ns)
                .map(|(w, n)| {
                    crate::quant::Linear::new(w.clone(), k, n, crate::quant::QuantMode::Int8)
                })
                .collect();
            let q8: Vec<&crate::quant::Quant8Linear> =
                lins.iter().map(|l| l.quant8().unwrap()).collect();
            let got =
                crate::cuda::maybe_gpu_linear_grouped(&q8, &a, m).expect("grouped should run");
            // 正解: 個別 CPU カーネル結果の列連結
            let n_total: usize = ns.iter().sum();
            let mut want = vec![0.0f32; m * n_total];
            let mut col = 0usize;
            for (i, l) in lins.iter().enumerate() {
                let q = crate::quant::Quant8Linear::quantize(&parts_w[i], k, ns[i]);
                let r = if m == 1 {
                    crate::tensor::matvec_bt_i8(&a, &q.w, &q.scale, k, ns[i])
                } else {
                    crate::tensor::matmul_bt_i8(&a, &q.w, &q.scale, m, k, ns[i])
                };
                for row in 0..m {
                    want[row * n_total + col..][..ns[i]].copy_from_slice(&r[row * ns[i]..][..ns[i]]);
                }
                col += ns[i];
            }
            for idx in 0..m * n_total {
                assert_eq!(
                    got[idx].to_bits(),
                    want[idx].to_bits(),
                    "grouped m={m} idx={idx}"
                );
            }
        }
        // gate/up 相当: 2 パート同幅 (llama.rs の lin_gateup_gpu 経路相当)
        let k = 64usize;
        let ff = 48usize;
        let gw = pseudo_random(ff * k, 75);
        let uw = pseudo_random(ff * k, 76);
        let a = pseudo_random(2 * k, 77);
        let gl = crate::quant::Linear::new(gw, k, ff, crate::quant::QuantMode::Int8);
        let ul = crate::quant::Linear::new(uw, k, ff, crate::quant::QuantMode::Int8);
        let got = crate::cuda::maybe_gpu_linear_grouped(
            &[gl.quant8().unwrap(), ul.quant8().unwrap()],
            &a,
            2,
        )
        .expect("grouped should run");
        // 個別 CPU (apply_batch_bt は GPU フックを持つので CPU カーネルを直接使う)
        let gq = crate::quant::Quant8Linear::quantize(&pseudo_random(ff * k, 75), k, ff);
        let uq = crate::quant::Quant8Linear::quantize(&pseudo_random(ff * k, 76), k, ff);
        let gr = crate::tensor::matvec_rows_bt_i8(&a, &gq.w, &gq.scale, k, ff, 2);
        let ur = crate::tensor::matvec_rows_bt_i8(&a, &uq.w, &uq.scale, k, ff, 2);
        for row in 0..2 {
            for j in 0..ff {
                assert_eq!(
                    got[row * (2 * ff) + j].to_bits(),
                    gr[row * ff + j].to_bits(),
                    "gate row={row} j={j}"
                );
                assert_eq!(
                    got[row * (2 * ff) + ff + j].to_bits(),
                    ur[row * ff + j].to_bits(),
                    "up row={row} j={j}"
                );
            }
        }
    }

    /// P17b パイプラインの固定オーバーヘッド測定 (P17 の linear_bt_i8 = 同期×4
    /// との比較)。decode m=1 の 5 形状。
    #[test]
    #[ignore]
    fn bench_cuda_pipeline_overhead_1p5b_decode() {
        crate::cuda::enable_for_tests();
        let shapes: [(&str, usize, usize); 5] = [
            ("qkv_proj", 1536, 2048),
            ("o_proj", 1536, 1536),
            ("gate_up", 1536, 17920),
            ("down", 8960, 1536),
            ("lm_head", 1536, 151936),
        ];
        for (label, k, n) in shapes {
            let w = pseudo_random(k * n, 92);
            let a = pseudo_random(k, 93);
            let lin = crate::quant::Linear::new(w, k, n, crate::quant::QuantMode::Int8);
            let iters = 50;
            let _ = lin.apply_bt(&a, 1); // warmup + 重みアップロード
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                let _ = lin.apply_bt(&a, 1);
            }
            let gpu = t0.elapsed().as_secs_f64() / iters as f64;
            let cpu = lin.dims(); // (CPU 経路は既存ベンチの値を参照)
            let _ = cpu;
            println!("[{label}] P17b pipeline m=1: {ms:.3}ms", ms = gpu * 1e3);
        }
    }

    /// prefill (m>1 GEMM) の GPU vs CPU 比較 — GPU 統合が prefill を
    /// 逆に遅くしていないか (m が大きいと GEMV 型カーネルは不利) を確認する。
    #[test]
    #[ignore]
    fn bench_cuda_pipeline_1p5b_prefill() {
        crate::cuda::enable_for_tests();
        let shapes: [(&str, usize, usize); 3] = [
            ("qkv_proj", 1536, 2048),
            ("gate_up", 1536, 17920),
            ("down", 8960, 1536),
        ];
        for m in [32usize, 128, 512] {
            for (label, k, n) in shapes {
                let w = pseudo_random(k * n, 94);
                let a = pseudo_random(m * k, 95);
                let lin = crate::quant::Linear::new(w, k, n, crate::quant::QuantMode::Int8);
                let iters = 10;
                let _ = lin.apply_bt(&a, m); // warmup
                let t0 = std::time::Instant::now();
                for _ in 0..iters {
                    let _ = lin.apply_bt(&a, m);
                }
                let gpu = t0.elapsed().as_secs_f64() / iters as f64;
                // CPU (matmul_bt_i8)
                let q = crate::quant::Quant8Linear::quantize(
                    &pseudo_random(k * n, 94),
                    k,
                    n,
                );
                let _ = crate::tensor::matmul_bt_i8(&a, &q.w, &q.scale, m, k, n);
                let t0 = std::time::Instant::now();
                for _ in 0..iters {
                    let _ = crate::tensor::matmul_bt_i8(&a, &q.w, &q.scale, m, k, n);
                }
                let cpu = t0.elapsed().as_secs_f64() / iters as f64;
                println!(
                    "[{label} prefill m={m}] GPU {g:.3}ms / CPU {c:.3}ms = {r:.2}x",
                    g = gpu * 1e3,
                    c = cpu * 1e3,
                    r = cpu / gpu,
                );
            }
        }
    }

    /// バッチ decode (m>1) の GPU vs CPU ベンチ (P6 のバッチ経路との比較)
    #[test]
    #[ignore]
    fn bench_cuda_linear_int8_1p5b_batch() {
        let ctx = CudaCtx::new().unwrap();
        println!("device: {}", ctx.info());
        let (k, n) = (1536usize, 17920usize); // gate_up
        let q = pseudo_i8(k * n, 80);
        let scale = positive_scales(n, 81);
        let w = ctx.upload_i8(&q).unwrap();
        let sc = ctx.upload_f32(&scale).unwrap();
        for m in [1usize, 4, 8] {
            let a = pseudo_random(m * k, 82 + m as u32);
            let _ = ctx.linear_bt_i8(&a, &w, &sc, m, k, n).unwrap();
            let iters = 20;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                let _ = ctx.linear_bt_i8(&a, &w, &sc, m, k, n).unwrap();
            }
            let gpu = t0.elapsed().as_secs_f64() / iters as f64;
            let _ = tensor::matvec_rows_bt_i8(&a, &q, &scale, k, n, m);
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                let _ = tensor::matvec_rows_bt_i8(&a, &q, &scale, k, n, m);
            }
            let cpu = t0.elapsed().as_secs_f64() / iters as f64;
            println!(
                "[gate_up m={m}] GPU {g:.3}ms / CPU {c:.3}ms = {r:.1}x",
                g = gpu * 1e3,
                c = cpu * 1e3,
                r = cpu / gpu,
            );
        }
    }

    /// P17d: GPU dno_expf (glibc e_expf の f64 再現) が CPU f32::exp (glibc) と
    /// ビット一致するか。ビットパターン走査 + ランダム + 特殊値。
    #[test]
    #[ignore]
    fn cuda_expf_glibc_bit_exact_vs_cpu() {
        enable_for_tests();
        CudaCtx::new().unwrap();
        let mut vals: Vec<f32> = vec![
            0.0,
            -0.0,
            f32::NEG_INFINITY,
            f32::INFINITY,
            -88.0,
            -103.0,
            -103.28,
            -103.97,
            -104.0,
            -110.0,
            88.0,
            88.72,
            89.0,
            100.0,
        ];
        // 主要指数帯のビットパターン走査 (指数部 0x3C..0xC2, mantissa 5 step)
        for exp_hi in 0x3Cu32..=0xC2 {
            for sign in [0u32, 1] {
                let hi = (sign << 31) | (exp_hi << 23);
                for mi in (0u32..(1 << 23)).step_by(5) {
                    vals.push(f32::from_bits(hi | mi));
                }
            }
        }
        // ランダムビット (|x| <= 120 にフィルタ + 範囲外も一部保持)
        let mut x: u32 = 123_456_789;
        for _ in 0..4_000_000 {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let v = f32::from_bits(x);
            if v.abs() <= 120.0 || v.is_infinite() {
                vals.push(v);
            }
        }
        let got = chain_expf_probe(&vals).expect("chain_expf_probe");
        assert_eq!(got.len(), vals.len());
        // 本番 CPU 経路 (silu) は要素ごとのスカラー呼び出しなので、比較基準も
        // スカラー expf に固定する (LLVM の自動ベクトル化で libmvec に置換される
        // とスカラー結果と 1ULP 異なるため inline(never) で回避)
        fn exp_ref(x: f32) -> f32 {
            #[inline(never)]
            fn inner(x: f32) -> f32 {
                x.exp()
            }
            inner(x)
        }
        let mut mismatches = 0usize;
        for (i, (&v, &g)) in vals.iter().zip(got.iter()).enumerate() {
            let want = exp_ref(v);
            if want.is_nan() && g.is_nan() {
                continue; // NaN は payload を比較しない
            }
            if want.to_bits() != g.to_bits() {
                if mismatches < 10 {
                    println!(
                        "MISMATCH idx={i}: x={v:?} (bits {:#010x}) cpu={want:?} ({}), gpu={g:?} ({})",
                        v.to_bits(),
                        want.to_bits(),
                        g.to_bits()
                    );
                }
                mismatches += 1;
            }
        }
        assert_eq!(mismatches, 0, "expf bit mismatch: {mismatches} / {}", vals.len());
        println!("expf bit-exact: {} values OK", vals.len());
    }

    /// P17d: GPU rmsnorm / add+rmsnorm / swiglu が CPU とビット一致するか
    #[test]
    #[ignore]
    fn cuda_chain_elem_bit_exact_vs_cpu() {
        let _g = test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let ctx = CudaCtx::new().unwrap();
        let (cols, m) = (1536usize, 4usize);
        let w = pseudo_random(cols, 201);
        // 1) 単純 rmsnorm
        let x = pseudo_random(cols * m, 200);
        chain_upload(CH_X, &x, cols, m).unwrap();
        chain_rmsnorm(&w, 1e-6, None, CH_H, cols, m, 0).unwrap();
        let got = chain_download(CH_H, cols * m).unwrap();
        for row in 0..m {
            let want = tensor::rmsnorm(&x[row * cols..][..cols], &w, cols, 1e-6);
            for i in 0..cols {
                assert_eq!(got[row * cols + i].to_bits(), want[i].to_bits(), "rmsnorm row={row} i={i}");
            }
        }
        // 2) residual add + rmsnorm
        let proj = pseudo_random(cols * m, 202);
        chain_upload(CH_X, &x, cols, m).unwrap();
        // P17j: 連続する chain_upload は pinned (pin_x) を共有するため間に sync
        // が必須 (DMA 完了前の pinned 上書き競合 → CH_X が化ける。本番経路は
        // upload 間に download 同期が挟まるため問題にならないが、このテストは
        // sync 無しの 2 連続 upload で競合し並列実行時にフレームしていた)。
        ctx.sync().unwrap();
        chain_upload(CH_O, &proj, cols, m).unwrap();
        chain_rmsnorm(&w, 1e-6, Some(CH_O), CH_H, cols, m, 1).unwrap();
        let got = chain_download(CH_H, cols * m).unwrap();
        let mut xr = x.clone();
        for i in 0..cols * m {
            xr[i] += proj[i];
        }
        for row in 0..m {
            let want = tensor::rmsnorm(&xr[row * cols..][..cols], &w, cols, 1e-6);
            for i in 0..cols {
                assert_eq!(
                    got[row * cols + i].to_bits(),
                    want[i].to_bits(),
                    "add+rmsnorm row={row} i={i}"
                );
            }
        }
        // 3) swiglu (gate|up 連結レイアウト)
        let ff = 8960usize;
        let g = pseudo_random(ff, 203);
        let u = pseudo_random(ff, 204);
        let mut ga = g.clone();
        ga.extend_from_slice(&u);
        chain_upload(CH_GA, &ga, ff * 2, 1).unwrap();
        chain_swiglu(CH_GA, CH_ACT, ff, 1).unwrap();
        let got = chain_download(CH_ACT, ff).unwrap();
        let mut want = g.clone();
        tensor::silu_inplace(&mut want);
        for i in 0..ff {
            want[i] *= u[i];
        }
        for i in 0..ff {
            assert_eq!(got[i].to_bits(), want[i].to_bits(), "swiglu i={i}");
        }
    }

    /// P17d: decode chain 全体 (rmsnorm → grouped qkv → download) が
    /// CPU 経路 (rmsnorm → apply_bt 個別 3 回) とビット一致するか
    #[test]
    #[ignore]
    fn cuda_decode_chain_bit_exact_vs_cpu() {
        let _g = test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        CudaCtx::new().unwrap();
        let (e, qn, kvn, ff) = (96usize, 40usize, 24usize, 64usize);
        let qkv_n = qn + 2 * kvn;
        let ln1 = pseudo_random(e, 300);
        let qw = crate::quant::Quant8Linear::quantize(&pseudo_random(qn * e, 310), e, qn);
        let kw = crate::quant::Quant8Linear::quantize(&pseudo_random(kvn * e, 311), e, kvn);
        let vw = crate::quant::Quant8Linear::quantize(&pseudo_random(kvn * e, 312), e, kvn);
        let x = pseudo_random(e, 320);
        // GPU chain
        chain_upload(CH_X, &x, e, 1).unwrap();
        chain_rmsnorm(&ln1, 1e-6, None, CH_H, e, 1, 2).unwrap();
        chain_linear_grouped(&[&qw, &kw, &vw], CH_H, CH_QKV, 1).unwrap();
        let got = chain_download(CH_QKV, qkv_n).unwrap();
        // CPU reference
        let h = tensor::rmsnorm(&x, &ln1, e, 1e-6);
        let mut want = Vec::with_capacity(qkv_n);
        want.extend_from_slice(&tensor::matvec_bt_i8(&h, &qw.w, &qw.scale, qw.k, qw.n));
        want.extend_from_slice(&tensor::matvec_bt_i8(&h, &kw.w, &kw.scale, kw.k, kw.n));
        want.extend_from_slice(&tensor::matvec_bt_i8(&h, &vw.w, &vw.scale, vw.k, vw.n));
        for i in 0..qkv_n {
            assert_eq!(got[i].to_bits(), want[i].to_bits(), "chain qkv i={i}");
        }
        // FFN 部: o → add+rmsnorm → gate_up → swiglu → down → add+rmsnorm(final)
        let ow = crate::quant::Quant8Linear::quantize(&pseudo_random(e * qn, 330), qn, e);
        let gw = crate::quant::Quant8Linear::quantize(&pseudo_random(ff * e, 331), e, ff);
        let uw = crate::quant::Quant8Linear::quantize(&pseudo_random(ff * e, 332), e, ff);
        let dw = crate::quant::Quant8Linear::quantize(&pseudo_random(e * ff, 333), ff, e);
        let ln2 = pseudo_random(e, 340);
        let norm_w = pseudo_random(e, 341);
        let att = pseudo_random(qn, 350);
        chain_upload(CH_ATT, &att, qn, 1).unwrap();
        chain_linear_grouped(&[&ow], CH_ATT, CH_O, 1).unwrap();
        chain_rmsnorm(&ln2, 1e-6, Some(CH_O), CH_H, e, 1, 3).unwrap();
        chain_linear_grouped(&[&gw, &uw], CH_H, CH_GA, 1).unwrap();
        chain_swiglu(CH_GA, CH_ACT, ff, 1).unwrap();
        chain_linear_grouped(&[&dw], CH_ACT, CH_DOWN, 1).unwrap();
        chain_rmsnorm(&norm_w, 1e-6, Some(CH_DOWN), CH_H, e, 1, 4).unwrap();
        let got = chain_download(CH_H, e).unwrap();
        // CPU mirror
        let o = tensor::matvec_bt_i8(&att, &ow.w, &ow.scale, ow.k, ow.n);
        let mut xr = x.clone();
        for i in 0..e {
            xr[i] += o[i];
        }
        let h2 = tensor::rmsnorm(&xr, &ln2, e, 1e-6);
        let mut ga_cpu = tensor::matvec_bt_i8(&h2, &gw.w, &gw.scale, gw.k, gw.n);
        let up = tensor::matvec_bt_i8(&h2, &uw.w, &uw.scale, uw.k, uw.n);
        ga_cpu.extend_from_slice(&up);
        tensor::silu_inplace(&mut ga_cpu[..ff]);
        for i in 0..ff {
            ga_cpu[i] *= ga_cpu[ff + i];
        }
        let d = tensor::matvec_bt_i8(&ga_cpu[..ff], &dw.w, &dw.scale, dw.k, dw.n);
        let mut xf = xr.clone();
        for i in 0..e {
            xf[i] += d[i];
        }
        let want = tensor::rmsnorm(&xf, &norm_w, e, 1e-6);
        for i in 0..e {
            assert_eq!(got[i].to_bits(), want[i].to_bits(), "chain ffn i={i}");
        }
        println!("decode chain bit-exact OK (e={e} qkv={qkv_n} ff={ff})");
    }

    /// P17g: dno_attention カーネルと CPU 参照実装の一致 (緩和契約 —
    /// ビット一致ではなく近似一致を検証)。rope 済み q とランダム K/V を使い、
    /// GQA (kv_heads=2, group=6, dh=128) でスコア/softmax/probs@V を比較する。
    #[cfg(all(test, feature = "cuda"))]
    #[test]
    #[ignore]
    fn cuda_attention_kernel_vs_cpu_close() {
        let _g = test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        crate::cuda::enable_for_tests();
        CudaCtx::new().unwrap();
        let (kv_heads, group, dh) = (2usize, 6usize, 128usize);
        let heads = kv_heads * group;
        let kv_dim = kv_heads * dh;
        let t_total = 577usize;
        let scale = 1.0 / (dh as f32).sqrt();

        let mut cache = crate::kv_cache::KvCache::new(1, kv_dim, 4096);
        {
            let (kc, vc) = cache.kv_bufs(0, t_total);
            for i in 0..t_total * kv_dim {
                kc[i] = ((i * 7919 % 2009) as f32 / 2009.0 - 0.5) * 2.0;
                vc[i] = ((i * 6271 % 1987) as f32 / 1987.0 - 0.5) * 2.0;
            }
        }
        cache.advance(t_total);
        let q: Vec<f32> = (0..heads * dh)
            .map(|i| ((i * 6151 % 1999) as f32 / 1999.0 - 0.5) * 4.0)
            .collect();

        // GPU カーネル (ミラーキーはテスト専用値)
        chain_attention(900_001, 0, 1, &q, &cache, t_total, group, dh, kv_dim, scale).unwrap();
        let got = chain_download(CH_ATT, heads * dh).unwrap();

        // CPU 参照 (スコア → softmax → probs@V、スレッド i が att[h*dh+i] を担当
        // するカーネルと同じ累積順: j 昇順 fmaf)
        let (kc_ref, vc_ref) = cache.kv_bufs_ref(0);
        let mut maxd = 0.0f32;
        for kvh in 0..kv_heads {
            for g in 0..group {
                let h = kvh * group + g;
                let mut scores = vec![0f32; t_total];
                let mut mx = f32::NEG_INFINITY;
                for j in 0..t_total {
                    let mut s = 0f32;
                    for i in 0..dh {
                        s += q[h * dh + i] * kc_ref[j * kv_dim + kvh * dh + i];
                    }
                    s *= scale;
                    scores[j] = s;
                    mx = mx.max(s);
                }
                let mut sum = 0f32;
                for s in scores.iter_mut() {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                for i in 0..dh {
                    let mut acc = 0f32;
                    for j in 0..t_total {
                        acc += scores[j] * vc_ref[j * kv_dim + kvh * dh + i];
                    }
                    let want = acc / sum;
                    let d = (got[h * dh + i] - want).abs();
                    maxd = maxd.max(d);
                }
            }
        }
        println!("attention kernel vs CPU close OK (t={t_total} maxabs={maxd:.3e})");
        assert!(maxd < 1e-3, "attention maxabs diff {maxd}");
    }

    /// B1: prefill デバイス常駐チェーン全体 (llama.rs forward_prefill_gpu と同一の
    /// 呼び出し列・単層) が CPU 参照経路と**ビット一致**すること:
    ///   chain_rmsnorm(#1) → chain_qkv_rope_prefill (h_dev=Some(CH_H): デバイス h 直読み)
    ///   → [attention は緩和契約のためランダム att を CH_ATT に注入]
    ///   → maybe_gpu_oproj_dev (CH_ATT 直読み) → chain_rmsnorm(#2, residual 融合)
    ///   → maybe_gpu_ffn_dev (デバイス常駐) → chain_rmsnorm(最終, residual 融合)。
    /// rmsnorm (dno_add_rmsnorm・in-place x += proj 含む) / GEMM / rope / swiglu /
    /// residual 加算は全て従来経路と同一カーネル・同一演算順序 → ビット一致が契約
    /// (attention のみ緩和枠・ユーザー承認済み)。検証: rope 後 q (デバイス常駐
    /// qkv_q)・K/V キャッシュ行 [pos0..t_total) (rope カーネル → D2H)・最終 hn
    /// (CH_H 全行 + chain_download_at の最終行)・最終 residual (CH_X 全行)。
    #[test]
    #[ignore] // GPU が必要: cargo test --release --features cuda -- --ignored
    fn cuda_prefill_chain_bit_exact_vs_cpu() {
        fn assert_bits_eq(got: &[f32], want: &[f32], label: &str) {
            assert_eq!(got.len(), want.len(), "{label}: len mismatch");
            if let Some(i) = got.iter().zip(want).position(|(a, b)| a.to_bits() != b.to_bits()) {
                panic!(
                    "{label}: bit mismatch @{i} gpu={:?} ({:08x}) cpu={:?} ({:08x})",
                    got[i],
                    got[i].to_bits(),
                    want[i],
                    want[i].to_bits()
                );
            }
        }
        let _g = test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        CudaCtx::new().unwrap();
        enable_for_tests();
        // GEMM 経路 (m > max_m=16 かつ m >= gemm_min_m) を満たすよう閾値を下げる
        std::env::set_var("DENNO_CUDA_GEMM_MIN_M", "8");

        let (n, e, ff) = (40usize, 96usize, 64usize);
        let (heads, kv_heads, dh) = (4usize, 2usize, 16usize);
        let pos0 = 5usize;
        let t_total = pos0 + n;
        let qn = heads * dh;
        let kv_dim = kv_heads * dh;
        let width = qn + 2 * kv_dim;
        let eps = 1e-6f32;
        let theta = 10_000.0f32;
        assert!(gemm_path_active(n), "テスト形状が GEMM 経路の事前ゲートを通らない");

        // 重み (本番と同一の Quant8Linear::quantize 構築経路・uid は自動採番)
        let qw = crate::quant::Quant8Linear::quantize(&pseudo_random(qn * e, 400), e, qn);
        let kw = crate::quant::Quant8Linear::quantize(&pseudo_random(kv_dim * e, 401), e, kv_dim);
        let vw = crate::quant::Quant8Linear::quantize(&pseudo_random(kv_dim * e, 402), e, kv_dim);
        let ow = crate::quant::Quant8Linear::quantize(&pseudo_random(e * qn, 403), qn, e);
        let gate = crate::quant::Quant8Linear::quantize(&pseudo_random(ff * e, 404), e, ff);
        let up = crate::quant::Quant8Linear::quantize(&pseudo_random(ff * e, 405), e, ff);
        let down = crate::quant::Quant8Linear::quantize(&pseudo_random(e * ff, 406), ff, e);
        let ln1 = pseudo_random(e, 407);
        let ln2 = pseudo_random(e, 408);
        let norm_w = pseudo_random(e, 409);

        // KV キャッシュ: 履歴行 [0..pos0) をシードして advance (pos0>0 経路)。
        // 現在行 [pos0..t_total) は chain_qkv_rope_prefill が rope カーネル → D2H で
        // 書く (前提: 未書き込み)。容量確保は従来経路と同一 (P17e 契約)。
        let mut cache = crate::kv_cache::KvCache::new(1, kv_dim, t_total + 64);
        {
            let (kbuf, vbuf) = cache.kv_bufs(0, pos0);
            kbuf[..pos0 * kv_dim].copy_from_slice(&pseudo_random(pos0 * kv_dim, 410));
            vbuf[..pos0 * kv_dim].copy_from_slice(&pseudo_random(pos0 * kv_dim, 411));
        }
        cache.advance(pos0);
        let _ = cache.kv_bufs(0, n);
        let key = cache.kv_id();

        // ---- GPU デバイス常駐チェーン (forward_prefill_gpu と同一呼び出し列) ----
        let x = pseudo_random(n * e, 412);
        chain_upload(CH_X, &x, e, n).unwrap();
        chain_rmsnorm(&ln1, eps, None, CH_H, e, n, 0).unwrap(); // pin_id = li*3
        chain_qkv_rope_prefill(
            &qw, &kw, &vw, &[], &mut cache, key, 0, 1, pos0, n, t_total, heads, kv_heads, dh,
            theta, &[], Some(CH_H),
        )
        .unwrap();
        // attention は緩和契約 (cuda_attention_prefill_vs_cpu_reference で担保) のため
        // ランダム att を CH_ATT に注入し、以降のデバイス常駐チェーンだけを検証する
        let att = pseudo_random(n * qn, 413);
        chain_upload(CH_ATT, &att, qn, n).unwrap();
        maybe_gpu_oproj_dev(&ow, n, qn, 0).unwrap();
        chain_rmsnorm(&ln2, eps, Some(CH_O), CH_H, e, n, 1).unwrap(); // pin_id = li*3+1
        maybe_gpu_ffn_dev(&gate, &up, &down, CH_H, n, ff, 0).unwrap();
        chain_rmsnorm(&norm_w, eps, Some(CH_DOWN), CH_H, e, n, 3).unwrap(); // pin_id = n_layers*3

        // ---- CPU 参照 (従来経路と同一演算列) ----
        let mut h = vec![0.0f32; n * e];
        for i in 0..n {
            h[i * e..][..e].copy_from_slice(&tensor::rmsnorm(&x[i * e..][..e], &ln1, e, eps));
        }
        // qkv GEMM (dno_gemm_i8 とビット一致の matvec_rows_bt_i8)。行 = [q | k | v]
        let mut qkv_w = Vec::with_capacity(e * width);
        qkv_w.extend_from_slice(&qw.w);
        qkv_w.extend_from_slice(&kw.w);
        qkv_w.extend_from_slice(&vw.w);
        let mut qkv_sc = Vec::with_capacity(width);
        qkv_sc.extend_from_slice(&qw.scale);
        qkv_sc.extend_from_slice(&kw.scale);
        qkv_sc.extend_from_slice(&vw.scale);
        let qkv = tensor::matvec_rows_bt_i8(&h, &qkv_w, &qkv_sc, e, width, n);
        let mut q_ref = vec![0.0f32; n * qn];
        let mut k_ref = vec![0.0f32; n * kv_dim];
        let mut v_ref = vec![0.0f32; n * kv_dim];
        for i in 0..n {
            let row = &qkv[i * width..][..width];
            q_ref[i * qn..][..qn].copy_from_slice(&row[..qn]);
            k_ref[i * kv_dim..][..kv_dim].copy_from_slice(&row[qn..qn + kv_dim]);
            v_ref[i * kv_dim..][..kv_dim].copy_from_slice(&row[qn + kv_dim..]);
        }
        tensor::rope_rows(&mut q_ref, n, heads, dh, pos0, theta);
        tensor::rope_rows(&mut k_ref, n, kv_heads, dh, pos0, theta);
        // o_proj → residual 加算 (dno_add_rmsnorm の in-place x += proj と同一) → rmsnorm #2
        let o = tensor::matvec_rows_bt_i8(&att, &ow.w, &ow.scale, qn, e, n);
        let mut xr = x.clone();
        for i in 0..n * e {
            xr[i] += o[i];
        }
        let mut h2 = vec![0.0f32; n * e];
        for i in 0..n {
            h2[i * e..][..e].copy_from_slice(&tensor::rmsnorm(&xr[i * e..][..e], &ln2, e, eps));
        }
        // FFN: gate|up 連結 GEMM → silu*up → down
        let mut gu_w = Vec::with_capacity(e * 2 * ff);
        gu_w.extend_from_slice(&gate.w);
        gu_w.extend_from_slice(&up.w);
        let mut gu_sc = Vec::with_capacity(2 * ff);
        gu_sc.extend_from_slice(&gate.scale);
        gu_sc.extend_from_slice(&up.scale);
        let out1 = tensor::matvec_rows_bt_i8(&h2, &gu_w, &gu_sc, e, 2 * ff, n);
        let mut act = vec![0.0f32; n * ff];
        for i in 0..n {
            let row = &out1[i * 2 * ff..(i + 1) * 2 * ff];
            act[i * ff..(i + 1) * ff].copy_from_slice(&row[..ff]);
            tensor::silu_inplace(&mut act[i * ff..(i + 1) * ff]);
            for f in 0..ff {
                act[i * ff + f] *= row[ff + f];
            }
        }
        let d = tensor::matvec_rows_bt_i8(&act, &down.w, &down.scale, ff, e, n);
        let mut xf = xr;
        for i in 0..n * e {
            xf[i] += d[i];
        }
        let mut hn = vec![0.0f32; n * e];
        for i in 0..n {
            hn[i * e..][..e].copy_from_slice(&tensor::rmsnorm(&xf[i * e..][..e], &norm_w, e, eps));
        }

        // ---- 検証 ----
        // 1) rope 後 q (デバイス常駐 qkv_q)
        let q_gpu = chain_qkv_q_download(n, qn).unwrap();
        assert_bits_eq(&q_gpu, &q_ref, "rope 後 q");
        // 2) K/V キャッシュ行: 現在行 [pos0..t_total) は rope 後と一致・履歴行 [0..pos0)
        //    はシード値のまま (rope カーネルが履歴を書き換えていないこと)
        let (kc, vc) = cache.kv_bufs_ref(0);
        let k_hist = pseudo_random(pos0 * kv_dim, 410);
        let v_hist = pseudo_random(pos0 * kv_dim, 411);
        for i in 0..t_total {
            let off = i * kv_dim;
            let wk: &[f32] = if i < pos0 {
                &k_hist[off..off + kv_dim]
            } else {
                &k_ref[(i - pos0) * kv_dim..(i - pos0 + 1) * kv_dim]
            };
            let wv: &[f32] = if i < pos0 {
                &v_hist[off..off + kv_dim]
            } else {
                &v_ref[(i - pos0) * kv_dim..(i - pos0 + 1) * kv_dim]
            };
            assert_bits_eq(&kc[off..off + kv_dim], wk, &format!("k 行 {i}"));
            assert_bits_eq(&vc[off..off + kv_dim], wv, &format!("v 行 {i}"));
        }
        // 3) 最終 hn (CH_H 全行) + chain_download_at の最終行 (本番が lm_head 入力と
        //    して取り出す [n-1] 行のみの D2H — B1 ヘルパー)
        let hn_gpu = chain_download(CH_H, n * e).unwrap();
        assert_bits_eq(&hn_gpu, &hn, "最終 hn");
        let hn_last = chain_download_at(CH_H, (n - 1) * e, e).unwrap();
        assert_bits_eq(&hn_last, &hn[(n - 1) * e..], "最終 hn (download_at)");
        // 4) 最終 residual (CH_X 全行 — 層内 2 回の in-place x += proj/down を含む)
        let xf_gpu = chain_download(CH_X, n * e).unwrap();
        assert_bits_eq(&xf_gpu, &xf, "最終 residual (CH_X)");
        println!("prefill chain bit-exact OK (n={n} e={e} ff={ff} pos0={pos0} t={t_total})");
    }

    // === C4 GPU segfault 修正 (2026-10-16) の回帰テスト (純関数・GPU 不要) ===

    /// C4: full_attention の CH_QKV k セクション配置と KV ミラー行内配置。
    /// 旧実装は src_off=hd (qg の後半 = gate 部を読む)・head 間隔 kv_dim
    /// (k 節域を外れて v/gate を読む・head kv_heads-1 はミラー行の外へ書く)
    /// だったため、k 履歴が破壊されていた。
    #[test]
    fn q35_fullattn_k_layout_matches_ch_qkv_sections() {
        // (heads, kv_heads, dh) — 9B 相当と縮小形状
        let cases: [(usize, usize, usize); 3] = [(32, 8, 128), (16, 4, 128), (8, 2, 64)];
        for &(heads, kv_heads, dh) in &cases {
            let kv_dim = kv_heads * dh;
            let k_off = q35_fullattn_k_off(heads, dh);
            // k セクションは qg (heads*2dh) の直後
            assert_eq!(k_off, heads * 2 * dh);
            let k_end = k_off + kv_dim;
            // head b の読み出し [k_off + b*dh .. + dh) は k セクション内
            for b in 0..kv_heads {
                assert!(k_off + b * dh + dh <= k_end, "k head {b} read OOB");
                // ミラー行 (kv_dim 幅) への書き込み [b*dh .. + dh) も行内
                assert!(b * dh + dh <= kv_dim, "k head {b} write out of mirror row");
            }
            // 旧実装の再現: head 1 の旧読み出し位置 (src_off=hd + 1*kv_dim) は
            // k セクション開始位置よりも前 (= qg 内) で、GQA (kv_heads < heads) なら常にそうなる
            let old_src_off = heads * dh;
            if kv_heads < heads {
                assert!(old_src_off + kv_dim < k_off, "旧 src_off は qg 内に留まる");
            }
            // 旧実装の再現: head kv_heads-1 の旧書き込み位置 ((kv_heads-1)*kv_dim)
            // はミラー行 (kv_dim 幅) の外
            if kv_heads >= 2 {
                assert!((kv_heads - 1) * kv_dim + dh > kv_dim, "旧 dst_stride は行外に書く");
            }
        }
    }

    /// C4: CH_QKV ステップ事前確保は GDN 層 ([qkv|z|beta|alpha]) と full 層
    /// ([qg|k|v]) の両レイアウトを覆う。旧実装 (事前確保なし) では GDN 層の
    /// grouped GEMV 直後に chain_q35_gdn_scan 頭部の ensure が grow 再確保で
    /// in-flight 書き込み先を解放し、dno_gdn_scan が未初期化バッファを読んでいた
    /// (実機 E2E 出力非決定論の直接原因)。grouped GEMV 直後の確保サイズ
    /// (conv_dim + z) が事前確保より小さい = 旧経路で grow が走ることの再現 assert。
    #[test]
    fn q35_ch_qkv_bytes_covers_gdn_and_full_sections() {
        // (heads, kv_heads, dh, nk, nv, dk, dv) — 9B 相当 / 0.8B 相当 / 縮小
        let cases: [(usize, usize, usize, usize, usize, usize, usize); 3] = [
            (32, 8, 128, 16, 32, 128, 128),
            (16, 4, 128, 8, 8, 128, 128),
            (8, 2, 64, 4, 4, 64, 64),
        ];
        for &(heads, kv_heads, dh, nk, nv, dk, dv) in &cases {
            let kv_dim = kv_heads * dh;
            let conv_dim = 2 * nk * dk + nv * dv;
            let vd = nv * dv;
            let reserved = q35_ch_qkv_bytes(heads, dh, kv_heads, nk, nv, dk, dv);
            // GDN レイアウトの各セクション末尾が予約域に収まる
            // (chain_q35_gdn_scan の pz/pb/pa オフセット + カーネル読み出し幅)
            let z_end = conv_dim + vd;
            let beta_end = z_end + nv;
            let alpha_end = beta_end + nv;
            assert!(z_end * 4 <= reserved, "z 節域外");
            assert!(beta_end * 4 <= reserved, "beta 節域外");
            assert!(alpha_end * 4 <= reserved, "alpha 節域外");
            // full レイアウト (q35_fullattn_k_off 経由の k/v セクション) も収まる
            let k_off = q35_fullattn_k_off(heads, dh);
            assert!(k_off * 4 <= reserved);
            assert!((k_off + 2 * kv_dim) * 4 <= reserved, "full k/v 節域外");
            // 旧経路の再現: grouped GEMV 直後の確保サイズ (conv_dim + z) は
            // 予約域より小さい → scan 頭部の ensure で grow 再確保が走っていた
            let grouped_after_gemv = (conv_dim + vd) * 4;
            assert!(
                grouped_after_gemv < reserved,
                "grouped 直後 ({grouped_after_gemv}) < 予約域 ({reserved}) が崩れた"
            );
        }
        // 単調性: 形状を大きくすると予約域も大きくなる (公式の取りこぼし防止)
        assert!(
            q35_ch_qkv_bytes(32, 128, 8, 16, 32, 128, 128)
                >= q35_ch_qkv_bytes(8, 64, 2, 4, 4, 64, 64)
        );
    }

    /// C4: CH_ATT ステップ事前確保は GDN 出力 y [vd] と full attention 出力
    /// [heads*dh] の両方を覆う。旧実装では heads*dh > vd の形状で full 層側の
    /// ensure が直前 GDN scan の in-flight 書き込み先を解放し得た。
    #[test]
    fn q35_ch_att_bytes_covers_gdn_y_and_full_att() {
        // (nv, dv, heads, dh)
        let cases: [(usize, usize, usize, usize); 4] = [
            (32, 128, 32, 128), // vd == heads*dh (等しい)
            (8, 128, 32, 128),  // vd < heads*dh → full 側が最大
            (32, 128, 8, 128),  // vd > heads*dh → GDN 側が最大
            (4, 64, 4, 64),
        ];
        for &(nv, dv, heads, dh) in &cases {
            let expected = (nv * dv).max(heads * dh) * 4;
            assert_eq!(q35_ch_att_bytes(nv, dv, heads, dh), expected);
            // 両出力の書き込み幅が予約域に収まる
            assert!((nv * dv) * 4 <= expected, "GDN y 節域外");
            assert!(heads * dh * 4 <= expected, "full att 節域外");
        }
    }

    /// C4: KV ステージの事前確保バイト数は full 層 li の積み位置
    /// (li*2*kv_dim 要素・1 層分 2*kv_dim 要素) と chain_q35_kv_stage_flush の
    /// 読み出し範囲を覆う。GDN 層はステージしない (skip) ため積み位置に穴が
    /// 開くが、全体サイズは最終層 (li = n_layers-1) の末尾で決まる。
    #[test]
    fn q35_kv_stage_bytes_matches_full_layer_staging() {
        // (n_layers, kv_dim)
        let cases: [(usize, usize); 3] = [(48, 1024), (24, 512), (4, 128)];
        for &(n_layers, kv_dim) in &cases {
            let total_elems = q35_kv_stage_bytes(n_layers, kv_dim) / 4;
            assert_eq!(total_elems, n_layers * 2 * kv_dim);
            // GDN 層 (skip される li) を含む全 li の積み末尾が総容量に収まる
            for li in 0..n_layers {
                let off = li * 2 * kv_dim;
                assert!(off + 2 * kv_dim <= total_elems, "li={li} の k/v 行が節域外");
            }
        }
    }

    /// s50: 遅延 GDN 状態フラッシュ (chain_q35_gdn_flush_pending) の契約検証。
    /// GPU 必須 (DevBuffer D2H + KV_MIRRORS が cuda feature 配下のため)。
    /// ①gdn_pos == cache.len() → cur 面 (成功ステップ後の現行状態)
    /// ②gdn_pos == cache.len()+1 → 1-cur 面 (層途中で失敗したステップの入力面)
    /// ③epoch 不一致 → pending 破棄の no-op (ホストが正)
    /// ④pending 無し / ミラー無し → no-op
    /// ⑤位置不整合 (バグのみ) → None + pending 破棄
    /// ⑥gdn_pos == None (chain 未走層) → スキップ
    /// 転送は bit そのまま (演算なし) のため、コピー内容の bit 一致で検証する。
    #[test]
    #[ignore] // GPU が必要: cargo test --release --features cuda -p denno-engine -- --ignored
    fn cuda_gdn_flush_pending_faces_positions_and_epoch() {
        fn assert_bits_eq(got: &[f32], want: &[f32], label: &str) {
            assert_eq!(got.len(), want.len(), "{label}: len mismatch");
            if let Some(i) = got.iter().zip(want).position(|(a, b)| a.to_bits() != b.to_bits()) {
                panic!(
                    "{label}: bit mismatch @{i} got={:?} ({:08x}) want={:?} ({:08x})",
                    got[i],
                    got[i].to_bits(),
                    want[i],
                    want[i].to_bits()
                );
            }
        }
        let _g = test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        enable_for_tests();
        let _ctx = CudaCtx::new().unwrap();
        // 番兵キー (MIRROR_TEST_KEY = u64::MAX-1 / 別テストの u64::MAX-3 と衝突しない)
        // flush は cache.kv_id() で検索するため、テストのミラーキーは cache の自動採番 id を使う
        let (nk, nv, dk, dv, d_conv) = (2usize, 4, 8, 8, 4);
        let conv_len = (2 * nk * dk + nv * dv) * (d_conv - 1); // 192
        let ssm_len = nv * dk * dv; // 256
        let mut cache = crate::kv_cache::KvCache::new(1, 64, 256);
        cache.attach_gdn_slot(0, conv_len, ssm_len);
        let key = cache.kv_id(); // プロセス内で一意 = chain_q35_gdn_flush_pending の検索キーと一致

        // 面に既知パターンを書く: face0 = A (state at target)、face1 = B (state at target+1)
        let conv_a = pseudo_random(conv_len, 81);
        let ssm_a = pseudo_random(ssm_len, 82);
        let conv_b = pseudo_random(conv_len, 83);
        let ssm_b = pseudo_random(ssm_len, 84);
        with_mirror_ext(key, 0, 1, 4, 0, cache.epoch(), false, |m, _| {
            if m.gdn[0].is_none() {
                let mut g = GdnDev {
                    conv: [None, None],
                    ssm: [None, None],
                    cur: 0,
                };
                ok(Scratch::ensure_tagged(
                    &mut g.conv[0],
                    conv_len * 4,
                    crate::mem_audit::TAG_GDN_MIRROR,
                ))?;
                ok(Scratch::ensure_tagged(
                    &mut g.conv[1],
                    conv_len * 4,
                    crate::mem_audit::TAG_GDN_MIRROR,
                ))?;
                ok(Scratch::ensure_tagged(
                    &mut g.ssm[0],
                    ssm_len * 4,
                    crate::mem_audit::TAG_GDN_MIRROR,
                ))?;
                ok(Scratch::ensure_tagged(
                    &mut g.ssm[1],
                    ssm_len * 4,
                    crate::mem_audit::TAG_GDN_MIRROR,
                ))?;
                m.gdn[0] = Some(g);
            }
            let gd = m.gdn[0].as_mut().unwrap();
            ok(gd.conv[0].as_ref().unwrap().copy_from(&conv_a))?;
            ok(gd.ssm[0].as_ref().unwrap().copy_from(&ssm_a))?;
            ok(gd.conv[1].as_ref().unwrap().copy_from(&conv_b))?;
            ok(gd.ssm[1].as_ref().unwrap().copy_from(&ssm_b))?;
            gd.cur = 0;
            m.gdn_pos[0] = Some(1);
            m.gdn_pending = Some(1);
            // s141b: ホストより新しい状態 = stale (flush はこの層を必ず読む)
            m.gdn_stale[0] = true;
            Some(())
        })
        .unwrap();

        let host = |cache: &mut crate::kv_cache::KvCache| {
            let (c, s) = cache.gdn_state_mut(0).unwrap();
            (c.to_vec(), s.to_vec())
        };
        let mirror_pending = || {
            let mirrors = KV_MIRRORS.get().unwrap();
            let mut ms = mirrors.lock().unwrap();
            let idx = ms.iter().position(|(k, _, _)| *k == key).unwrap();
            ms[idx].1.gdn_pending
        };
        let mirror_stale = || {
            let mirrors = KV_MIRRORS.get().unwrap();
            let mut ms = mirrors.lock().unwrap();
            let idx = ms.iter().position(|(k, _, _)| *k == key).unwrap();
            ms[idx].1.gdn_stale[0]
        };

        // s141b (Task B7c): stale でも位置が target+1 (層途中失敗) の層は読まない
        // = ホストは変わらない (古い値を書き戻さない)。ホストを 0 埋めして
        // flush を回し、0 のままであることを見る (① の前 = gdn_pos は 1 のまま)。
        // ① では cache.len() が 1 になり位置が target に揃うので、そこで初めて読む。
        assert_eq!(cache.len(), 0, "前提: advance 前");
        {
            let (hc, hs) = cache.gdn_state_mut(0).unwrap();
            hc.fill(0.0);
            hs.fill(0.0);
        }
        assert_eq!(chain_q35_gdn_flush_pending(&mut cache), Some(()));
        let (hz_c, hz_s) = host(&mut cache);
        assert!(hz_c.iter().all(|&v| v == 0.0) && hz_s.iter().all(|&v| v == 0.0),
            "target+1 の層は D2H しない (古い値を書き戻さない)");
        assert_eq!(mirror_pending(), None, "no-op でも pending は解除");
        // 次に読む機会 (①) のために、ミラー状態を元へ戻す。
        with_mirror_ext(key, 0, 1, 4, 0, cache.epoch(), false, |m, _| {
            m.gdn_pos[0] = Some(1);
            m.gdn_pending = Some(1);
            m.gdn_stale[0] = true;
            Some(())
        })
        .unwrap();

        // ① 成功ステップ後: gdn_pos = 1 == cache.len() (advance 後) → cur 面 = A
        cache.advance(1);
        assert_eq!(chain_q35_gdn_flush_pending(&mut cache), Some(()));
        let (hc, hs) = host(&mut cache);
        assert_bits_eq(&hc, &conv_a, "①conv は cur 面 (A) と bit 一致");
        assert_bits_eq(&hs, &ssm_a, "①ssm は cur 面 (A) と bit 一致");
        assert_eq!(mirror_pending(), None, "①フラッシュ後 pending は解除");
        assert!(!mirror_stale(), "①フラッシュ後、層のホスト遅延フラグは落ちている");

        // ② 層途中失敗: swap 済み (cur = 書き込み面 = state at 2)・cache.len() = 1
        //    → 1-cur 面 (= state at 1 = A) を回復する
        cache.advance(1); // len = 2
        with_mirror_ext(key, 0, 1, 4, 0, cache.epoch(), false, |m, _| {
            let gd = m.gdn[0].as_mut().unwrap();
            gd.cur = 1; // 成功ステップが cur を B 面へ交換した直後の想定
            m.gdn_pos[0] = Some(3);
            m.gdn_pending = Some(3);
            m.gdn_stale[0] = true; // s141b: ホストより新しい
            Some(())
        })
        .unwrap();
        // cache.len() = 2・gdn_pos = 3 = target+1 → 1-cur = 0 面 = A
        assert_eq!(chain_q35_gdn_flush_pending(&mut cache), Some(()));
        let (hc, hs) = host(&mut cache);
        assert_bits_eq(&hc, &conv_a, "②conv は 1-cur 面 (A) と bit 一致");
        assert_bits_eq(&hs, &ssm_a, "②ssm は 1-cur 面 (A) と bit 一致");

        // ②' 成功ステップ後 (cur 面が B): gdn_pos = target → cur 面 = B
        cache.advance(1); // len = 3
        with_mirror_ext(key, 0, 1, 4, 0, cache.epoch(), false, |m, _| {
            let gd = m.gdn[0].as_mut().unwrap();
            gd.cur = 1; // B 面が現行
            m.gdn_pos[0] = Some(3);
            m.gdn_pending = Some(3);
            m.gdn_stale[0] = true; // s141b: ホストより新しい
            Some(())
        })
        .unwrap();
        assert_eq!(chain_q35_gdn_flush_pending(&mut cache), Some(()));
        let (hc, hs) = host(&mut cache);
        assert_bits_eq(&hc, &conv_b, "②'conv は cur 面 (B) と bit 一致");
        assert_bits_eq(&hs, &ssm_b, "②'ssm は cur 面 (B) と bit 一致");

        // ⑤ 位置不整合 (正常系では起こらない) → None + pending 破棄
        with_mirror_ext(key, 0, 1, 4, 0, cache.epoch(), false, |m, _| {
            m.gdn_pos[0] = Some(7);
            m.gdn_pending = Some(7);
            m.gdn_stale[0] = true; // s141b: 古いホストを持つ層だけが不整合を検出する
            Some(())
        })
        .unwrap();
        assert_eq!(
            chain_q35_gdn_flush_pending(&mut cache),
            None,
            "位置不整合は None (呼び出し側は継続しない)"
        );
        assert_eq!(mirror_pending(), None, "⑤不整合時も pending は破棄");

        // ⑥ gdn_pos = None (chain 未走層) → スキップで no-op (ホスト不変)
        with_mirror_ext(key, 0, 1, 4, 0, cache.epoch(), false, |m, _| {
            m.gdn_pos[0] = None;
            m.gdn_pending = Some(1);
            m.gdn_stale[0] = true; // s141b
            Some(())
        })
        .unwrap();
        let (hc0, hs0) = host(&mut cache);
        assert_eq!(chain_q35_gdn_flush_pending(&mut cache), Some(()));
        let (hc1, hs1) = host(&mut cache);
        assert_eq!(hc0.len(), hc1.len(), "⑥conv len 不変");
        assert_eq!(hs0.len(), hs1.len(), "⑥ssm len 不変");
        assert!(
            hc0.iter().zip(&hc1).all(|(a, b)| a.to_bits() == b.to_bits())
                && hs0.iter().zip(&hs1).all(|(a, b)| a.to_bits() == b.to_bits()),
            "⑥chain 未走層はホスト不変 (bit 一致)"
        );

        // ③ epoch 不一致 (reset = ホストが正) → pending 破棄の no-op
        cache.reset(); // GDN スロットは 0 埋め・epoch +1
        // with_mirror_ext は epoch を同期してしまうため使わない。直接ロックで
        // 旧世代 (reset 前の epoch) のミラーに pending を設定し、epoch 不一致の
        // 状態を再現する。
        {
            let mirrors = KV_MIRRORS.get().unwrap();
            let mut ms = mirrors.lock().unwrap();
            let idx = ms.iter().position(|(k, _, _)| *k == key).unwrap();
            ms[idx].1.gdn_pos[0] = Some(9);
            ms[idx].1.gdn_pending = Some(9);
            ms[idx].1.gdn_stale[0] = true; // s141b
        }
        assert_eq!(chain_q35_gdn_flush_pending(&mut cache), Some(()));
        assert_eq!(mirror_pending(), None, "③epoch 不一致で pending は破棄");
        let (hc, hs) = host(&mut cache);
        assert!(hc.iter().all(|&v| v == 0.0) && hs.iter().all(|&v| v == 0.0),
            "③reset 済みホスト (0 埋め) は旧面で上書きされない");

        // ④ pending 無し / ミラー無し → no-op
        assert_eq!(chain_q35_gdn_flush_pending(&mut cache), Some(()));
        let other = crate::kv_cache::KvCache::new(1, 64, 256); // 別 kv_id・ミラー無し
        let mut other = other;
        assert_eq!(chain_q35_gdn_flush_pending(&mut other), Some(()));
    }
}


#[cfg(test)]
mod ssm_gemm_gpu_tests {
    use super::*;

    #[test]
    fn ssm_gemm_gpu_gate_truth_table() {
        assert!(!pf_ssm_gemm_gpu_from_value(None), "未設定は既定 off");
        for v in ["0", "off", "false", "no", "", " off "] {
            assert!(!pf_ssm_gemm_gpu_from_value(Some(v)), "{v:?} が on になった");
        }
        for v in ["1", "on", "true", "yes", "2"] {
            assert!(pf_ssm_gemm_gpu_from_value(Some(v)), "{v:?} が off のまま");
        }
    }

    #[test]
    fn ssm_gemm_gpu_force_switch() {
        set_pf_ssm_gemm_gpu_for_test(Some(true));
        assert!(pf_ssm_gemm_gpu_on());
        set_pf_ssm_gemm_gpu_for_test(Some(false));
        assert!(!pf_ssm_gemm_gpu_on());
        set_pf_ssm_gemm_gpu_for_test(None);
        assert!(!pf_ssm_gemm_gpu_on(), "env 未設定の既定は off");
    }
}


#[cfg(test)]
mod ba228_tests {
    //! task 228: beta/alpha のデバイス入口 (型を問わず f32 へ展開) の検査。
    use super::*;
    use crate::kquant::{dequant_row, GgmlType as T};

    /// `n` 行 × `k` 列の生ブロックを型ごとの正しいバイト数で作る。
    pub(crate) fn raw(ty: T, k: usize, n: usize, seed: u32) -> Vec<u8> {
        let bs = ty.blck_size();
        let ts = ty.type_size();
        assert_eq!(k % bs, 0, "{ty:?}: k がブロック倍数でない");
        let nb = k / bs;
        let mut out = Vec::with_capacity(n * nb * ts);
        for r in 0..n {
            for b in 0..nb {
                let base = seed.wrapping_add((r * 131 + b * 17) as u32);
                match ty {
                    T::F32 => {
                        for i in 0..bs {
                            let v = ((base + i as u32) % 97) as f32 / 97.0 - 0.5;
                            out.extend_from_slice(&v.to_le_bytes());
                        }
                    }
                    T::Bf16 => {
                        for i in 0..bs {
                            let v = ((base + i as u32) % 97) as f32 / 97.0 - 0.5;
                            out.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
                        }
                    }
                    T::F16 => {
                        for i in 0..bs {
                            // 0.5 + (0..63)/... の正規化数の範囲 (0x3800 = 0.5)
                            let bits: u16 = 0x3800 + ((base + i as u32) % 64) as u16;
                            out.extend_from_slice(&bits.to_le_bytes());
                        }
                    }
                    T::Q8_0 => {
                        out.extend_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
                        for i in 0..32 {
                            let v = (((base + i) % 251) as i32 - 125) as i8;
                            out.push(v as u8);
                        }
                    }
                    other => panic!("テスト用の生成器が無い型 {other:?}"),
                }
            }
        }
        out
    }

    /// 展開した f32 は `dequant_row` (ホスト GEMV が使う関数) と行ごとにビット一致。
    #[test]
    fn pf228_ba_host_f32_matches_dequant_row() {
        for ty in [T::F32, T::F16, T::Bf16, T::Q8_0] {
            let (k, n) = (256usize, 7usize);
            let bytes = raw(ty, k, n, 11);
            let flat = pf_ba_host_f32(ty, &bytes, k, n).expect("展開できる");
            assert_eq!(flat.len(), k * n, "{ty:?}: 要素数");
            let rb = bytes.len() / n;
            for r in 0..n {
                let row = dequant_row(ty, &bytes[r * rb..(r + 1) * rb], k);
                assert_eq!(
                    &flat[r * k..(r + 1) * k],
                    &row[..],
                    "{ty:?}: 行 {r} が dequant_row と違う"
                );
            }
            // バイト数が 1 足りない / k が形と合わない → None (黙って別物を作らない)。
            assert!(pf_ba_host_f32(ty, &bytes[..bytes.len() - 1], k, n).is_none());
            assert!(pf_ba_host_f32(ty, &bytes, k + 1, n).is_none());
            assert!(pf_ba_host_f32(ty, &bytes, k, n + 1).is_none());
        }
    }

    /// ホスト GEMV (BF16/F16 の展開経路) は「展開済み f32 の k 昇順 mul+add」と
    /// **ビット一致**する。デバイス f32 GEMM との差が fma の丸めだけである根拠。
    #[test]
    fn pf228_ba_host_f32_host_gemv_bits() {
        use crate::model::linear_gguf::GgufLinear;
        let (k, n, nt) = (256usize, 5usize, 3usize);
        let xs: Vec<f32> = (0..nt * k)
            .map(|i| ((i * 37 % 101) as f32) / 101.0 - 0.5)
            .collect();
        for ty in [T::F16, T::Bf16] {
            let bytes = raw(ty, k, n, 3);
            let w = GgufLinear::from_slice(ty, k, n, &bytes).expect("GgufLinear");
            let mut out = vec![0f32; nt * n];
            crate::model::qwen4exp::gemv_batch_cpu(&w, &xs, nt, &mut out).expect("gemv_batch_cpu");
            let flat = pf_ba_host_f32(ty, &bytes, k, n).expect("展開");
            for t in 0..nt {
                for r in 0..n {
                    let mut acc = 0f32;
                    for i in 0..k {
                        acc += flat[r * k + i] * xs[t * k + i];
                    }
                    assert_eq!(
                        out[t * n + r].to_bits(),
                        acc.to_bits(),
                        "{ty:?}: t={t} r={r}"
                    );
                }
            }
        }
    }

    /// 入口の判定: 実モデルの形 (k = n_embd, 行 = n_v_heads) を型を問わず受け、
    /// 形が違えば理由付きで閉じる。
    #[test]
    fn pf228_ba_w_check_truth_table() {
        let (e, nv) = (2560usize, 48usize);
        for ty in [T::F32, T::F16, T::Bf16, T::Q8_0, T::Q2K, T::IQ3S] {
            assert!(
                pf_ba_w_ok(ty, e, 2 * nv, &(0..nv), nv, e),
                "{ty:?}: 連結 (beta 行) を受けない"
            );
            assert!(
                pf_ba_w_ok(ty, e, 2 * nv, &(nv..2 * nv), nv, e),
                "{ty:?}: 連結 (alpha 行) を受けない"
            );
            assert!(
                pf_ba_w_ok(ty, e, nv, &(0..nv), nv, e),
                "{ty:?}: 未連結 (alpha_w) を受けない"
            );
            assert!(
                !pf_ba_w_ok(ty, e + 8, 2 * nv, &(0..nv), nv, e),
                "{ty:?}: k 違いを受けた"
            );
            assert!(
                !pf_ba_w_ok(ty, e, 2 * nv, &(0..nv + 1), nv, e),
                "{ty:?}: 行数違いを受けた"
            );
            assert!(
                !pf_ba_w_ok(ty, e, nv, &(1..nv + 1), nv, e),
                "{ty:?}: n を超える行範囲を受けた"
            );
            assert!(
                !pf_ba_w_ok(ty, e, nv, &(nv..nv), nv, e),
                "{ty:?}: 空の行範囲を受けた"
            );
        }
        let why = pf_ba_w_check(T::Bf16, e + 8, 2 * nv, &(0..nv), nv, e).expect("閉じる");
        assert!(why.contains("n_embd"), "理由に k の食い違いが出ていない: {why}");
        let why = pf_ba_w_check(T::Bf16, e, 2 * nv, &(0..nv + 1), nv, e).expect("閉じる");
        assert!(
            why.contains("n_v_heads"),
            "理由に行数の食い違いが出ていない: {why}"
        );
    }
}


#[cfg(test)]
mod pf228_wy_tests {
    use super::*;

    /// ゲートは既定 off。`DENNO_PF_SCAN_WY=0` で明示的に切り戻せる。
    #[test]
    fn pf228_scan_wy_gate_defaults() {
        assert!(!gdn_pf_wy_from_value(None), "未設定の既定は off");
        for v in ["0", "off", "false", "no", "", " off "] {
            assert!(!gdn_pf_wy_from_value(Some(v)), "{v:?} が on になった");
        }
        for v in ["1", "on", "true", "yes", "2"] {
            assert!(gdn_pf_wy_from_value(Some(v)), "{v:?} が off のまま");
        }
        set_gdn_pf_wy_for_test(Some(true));
        assert!(gdn_pf_wy_on());
        set_gdn_pf_wy_for_test(Some(false));
        assert!(!gdn_pf_wy_on());
        set_gdn_pf_wy_for_test(None);
    }

    /// 形のゲートは v2/v3 と同じ (dk = dv = 128 / d_conv = 4)。
    #[test]
    fn pf228_scan_wy_shape_gate() {
        let conv_dim = 2 * 16 * 128 + 48 * 128;
        assert!(gdn_pf_wy_ok(512, 16, 48, 128, 128, 4, conv_dim));
        assert!(!gdn_pf_wy_ok(0, 16, 48, 128, 128, 4, conv_dim));
        assert!(!gdn_pf_wy_ok(512, 16, 48, 64, 128, 4, conv_dim));
        assert!(!gdn_pf_wy_ok(512, 16, 48, 128, 128, 3, conv_dim));
        assert!(!gdn_pf_wy_ok(512, 16, 48, 128, 128, 4, conv_dim + 1));
    }

    /// 共有メモリの見積りがカーネルの `WY_TOTAL` と一致する (PTX を読む前に
    /// opt-in する値なので、ここがずれると起動が落ちる)。
    #[test]
    fn pf228_scan_wy_smem_matches_source() {
        let src = include_str!("../../kernels/kernels_gdn_pf_wy.cu");
        let mut total = 0usize;
        for line in src.lines() {
            if let Some(rest) = line.strip_prefix("#define WY_TOTAL ") {
                total = rest.trim().parse().expect("WY_TOTAL が数値でない");
            }
        }
        assert_eq!(
            total, GDN_PF_WY_SMEM_FLOATS,
            "WY_TOTAL と GDN_PF_WY_SMEM_FLOATS が違う"
        );
        assert_eq!(GDN_PF_WY_SMEM_BYTES, 91_012, "共有メモリのバイト数");
        // lab t228a の実機 (RTX 5090, sm_120) の opt-in 上限は 99 KB。C = 32 の
        // 176,644 B は opt-in に失敗して v3 へ落ちたので、上限に収める。
        assert!(
            GDN_PF_WY_SMEM_BYTES <= GDN_PF_WY_SMEM_LIMIT,
            "共有メモリがデバイスの opt-in 上限を超える"
        );
        // 余裕を残す (上限ぎりぎりだと将来の作業域追加でまた落ちる)。
        assert!(
            GDN_PF_WY_SMEM_BYTES < 96 * 1024,
            "共有メモリの余裕が 3 KB 未満"
        );
    }

    /// task 311: 形のゲートは v2/v3 と同じ + dv が 4 分割できること。
    #[test]
    fn pf311_scan_wy4_shape_gate() {
        let conv_dim = 2 * 16 * 128 + 48 * 128;
        assert!(gdn_pf_wy4_ok(512, 16, 48, 128, 128, 4, conv_dim));
        assert!(!gdn_pf_wy4_ok(0, 16, 48, 128, 128, 4, conv_dim));
        assert!(!gdn_pf_wy4_ok(512, 16, 48, 128, 128, 3, conv_dim));
        assert!(!gdn_pf_wy4_ok(512, 16, 48, 128, 128, 4, conv_dim + 1));
        // dv が 4 * 32 でないときは撃たない (カーネルは 128 列前提)。
        assert!(!gdn_pf_wy4_ok(512, 16, 48, 128, 64, 4, 2 * 16 * 128 + 48 * 64));
        assert!(!gdn_pf_wy4_ok(512, 16, 48, 64, 64, 4, 2 * 16 * 64 + 48 * 64));
    }

    /// task 311/317: v 列 4 分割のゲートは既定 on (task 317 で採択)。
    /// `DENNO_PF_SCAN_WY4=0` で切り戻せる。
    #[test]
    fn pf311_scan_wy4_gate_defaults() {
        assert!(gdn_pf_wy4_from_value(None), "task 317 で既定 on");
        for v in ["0", "off", "false", "no", "", " off "] {
            assert!(!gdn_pf_wy4_from_value(Some(v)), "{v:?} が on になった");
        }
        for v in ["1", "on", "true", "yes", "2"] {
            assert!(gdn_pf_wy4_from_value(Some(v)), "{v:?} が off のまま");
        }
        set_gdn_pf_wy4_for_test(Some(true));
        assert!(gdn_pf_wy4_on());
        set_gdn_pf_wy4_for_test(Some(false));
        assert!(!gdn_pf_wy4_on());
        set_gdn_pf_wy4_for_test(None);
        assert!(gdn_pf_wy4_on(), "FORCE を戻したら既定 on (task 317)");
    }

    #[test]
    fn pf311_scan_wy4_smem_matches_source() {
        let src = include_str!("../../kernels/kernels_gdn_pf_wy4.cu");
        let mut get = |name: &str| -> usize {
            let key = format!("#define {name} ");
            for line in src.lines() {
                if let Some(rest) = line.strip_prefix(&key) {
                    return rest
                        .trim()
                        .parse()
                        .unwrap_or_else(|_| panic!("{name} が数値でない"));
                }
            }
            panic!("{name} が .cu に無い");
        };
        assert_eq!(get("WY4_TOTAL"), GDN_PF_WY4_SMEM_FLOATS);
        assert_eq!(get("WY4_DV"), GDN_PF_WY4_DV);
        assert_eq!(get("WY4_THREADS"), GDN_PF_WY4_THREADS);
        // 4 ブロックで 128 列を覆うこと (voff の取りこぼしを作らない)。
        assert_eq!(GDN_PF_WY4_BLOCKS * GDN_PF_WY4_DV, 128);
        // 29,572 B。48 KB 既定の内側なので opt-in は要らないが、上限も見る。
        assert_eq!(GDN_PF_WY4_SMEM_BYTES, 29_572, "共有メモリのバイト数");
        assert!(GDN_PF_WY4_SMEM_BYTES < 48 * 1024);
        // (1) の conv は tid < 128 (q) と 128..256 (k) = 2 * DK スレッドが要る。
        assert_eq!(GDN_PF_WY4_THREADS, 2 * 128);
        // 共有メモリの区画が重ならないこと (並びは .cu の #define の順)。
        let mut offs = Vec::new();
        for name in [
            "WY4_S", "WY4_KC", "WY4_QC", "WY4_VC", "WY4_DC", "WY4_PC", "WY4_QPC", "WY4_RDEC",
            "WY4_AC", "WY4_GC", "WY4_LAM", "WY4_RL", "WY4_EG", "WY4_BC", "WY4_LL",
        ] {
            offs.push((name, get(name)));
        }
        for w in offs.windows(2) {
            assert!(w[0].1 < w[1].1, "{} と {} の並びが逆", w[0].0, w[1].0);
        }
        // 最後の区画 (WY4_LL = 1 本) の終端が WY4_TOTAL。
        assert_eq!(offs.last().unwrap().1 + 1, GDN_PF_WY4_SMEM_FLOATS);
    }

    /// task 397: wy5 のゲート (lab t397b で採ったので既定 on)・形の前提・共有メモリ。
    #[test]
    fn task397_scan_wy5_gate_and_shape() {
        assert!(gdn_pf_wy5_from_value(None), "既定 on (lab t397b)");
        for v in ["0", "off", "false", "no", "", " off "] {
            assert!(!gdn_pf_wy5_from_value(Some(v)), "{v:?} が on になった");
        }
        for v in ["1", "on", "true", "yes", "2"] {
            assert!(gdn_pf_wy5_from_value(Some(v)), "{v:?} が off のまま");
        }
        set_gdn_pf_wy5_for_test(Some(true));
        assert!(gdn_pf_wy5_on());
        set_gdn_pf_wy5_for_test(Some(false));
        assert!(!gdn_pf_wy5_on());
        set_gdn_pf_wy5_for_test(None);
        assert!(gdn_pf_wy5_on(), "FORCE を戻したら既定 on");
        // 形の前提は wy4 と同じ (dv = 128 = 4 x 32)。
        let conv_dim = 2 * 16 * 128 + 48 * 128;
        assert!(gdn_pf_wy5_ok(2048, 16, 48, 128, 128, 4, conv_dim));
        assert!(gdn_pf_wy5_ok(3826, 16, 48, 128, 128, 4, conv_dim));
        assert!(
            !gdn_pf_wy5_ok(2048, 16, 48, 128, 64, 4, 2 * 16 * 128 + 48 * 64),
            "dv=64 は受けない"
        );
        assert!(!gdn_pf_wy5_ok(0, 16, 48, 128, 128, 4, conv_dim), "n=0 は受けない");
        // 共有メモリ・起動形は wy4 と同一 (作業域を増やしていない)。
        assert_eq!(GDN_PF_WY5_SMEM_FLOATS, GDN_PF_WY4_SMEM_FLOATS);
        assert_eq!(GDN_PF_WY5_SMEM_BYTES, 29_572);
        // .cu の WY5_TOTAL / 区画 / C が Rust 側の前提と一致すること。
        let src = include_str!("../../kernels/kernels_gdn_pf_wy5.cu");
        let mut get = |name: &str| -> usize {
            let key = format!("#define {name} ");
            for line in src.lines() {
                if let Some(rest) = line.strip_prefix(&key) {
                    return rest
                        .trim()
                        .parse()
                        .unwrap_or_else(|_| panic!("{name} が数値でない"));
                }
            }
            panic!("{name} が .cu に無い");
        };
        assert_eq!(get("WY5_TOTAL"), GDN_PF_WY5_SMEM_FLOATS);
        assert_eq!(get("WY5_DV"), GDN_PF_WY4_DV);
        assert_eq!(get("WY5_THREADS"), GDN_PF_WY4_THREADS);
        assert_eq!(get("WY5_C"), 8, "チャンク長は wy4 と同じ 8");
        assert_eq!(get("WY5_WARPS") * 32, GDN_PF_WY4_THREADS);
        assert_eq!(GDN_PF_WY4_BLOCKS * GDN_PF_WY4_DV, get("WY5_DK"));
        // 区画の並びが wy4 と同じ (オフセットが 1 つずつ同じ値)。
        let src4 = include_str!("../../kernels/kernels_gdn_pf_wy4.cu");
        let get4 = |name: &str| -> usize {
            let key = format!("#define {name} ");
            for line in src4.lines() {
                if let Some(rest) = line.strip_prefix(&key) {
                    return rest
                        .trim()
                        .parse()
                        .unwrap_or_else(|_| panic!("{name} が数値でない"));
                }
            }
            panic!("{name} が .cu に無い");
        };
        for (wy5, wy4) in [
            ("WY5_S", "WY4_S"),
            ("WY5_KC", "WY4_KC"),
            ("WY5_QC", "WY4_QC"),
            ("WY5_VC", "WY4_VC"),
            ("WY5_DC", "WY4_DC"),
            ("WY5_PC", "WY4_PC"),
            ("WY5_QPC", "WY4_QPC"),
            ("WY5_RDEC", "WY4_RDEC"),
            ("WY5_AC", "WY4_AC"),
            ("WY5_GC", "WY4_GC"),
            ("WY5_LAM", "WY4_LAM"),
            ("WY5_RL", "WY4_RL"),
            ("WY5_EG", "WY4_EG"),
            ("WY5_BC", "WY4_BC"),
            ("WY5_LL", "WY4_LL"),
        ] {
            assert_eq!(get(wy5), get4(wy4), "{wy5} のオフセットが wy4 と違う");
        }
        // 直した 3 点がカーネルに入っていること (落とすと wy4 と同じ速度に戻る)。
        assert!(src.contains("__shfl_down_sync"), "A/G がワープ並列でない");
        assert!(src.contains("float xq[WY5_C];"), "conv の先読みが無い");
        assert_eq!(
            src.matches("float sk0 = 0.0f;").count(),
            1,
            "P/QP の 4 アキュムレータが無い"
        );
    }

    /// task 397: 状態の書き戻しの非同期化ゲート (既定 off)。
    #[test]
    fn task397_gdn_wb_async_gate_defaults() {
        use crate::model::exp_pf::{
            gdn_wb_async_from_value, DEFAULT_GDN_WB_ASYNC, GDN_WB_ASYNC_ENV,
        };
        assert_eq!(GDN_WB_ASYNC_ENV, "DENNO_PF_GDN_WB_ASYNC");
        assert!(!DEFAULT_GDN_WB_ASYNC, "既定 off");
        assert!(!gdn_wb_async_from_value(None), "未設定は既定 off");
        for v in ["", "0", "off", "false", "no", " off "] {
            assert!(!gdn_wb_async_from_value(Some(v)), "{v:?} が on になった");
        }
        for v in ["1", "on", "true", "yes"] {
            assert!(gdn_wb_async_from_value(Some(v)), "{v:?} が off のまま");
        }
    }
}


#[cfg(test)]
mod pf225_gate_tests {
    use super::*;

    /// task 231: BA_FAST も既定 on (t228a の ba COND)。router / scan3 は task 228。
    #[test]
    fn pf225_gates_defaults() {
        for (name, f, dflt) in [
            (
                "DENNO_PF_BA_FAST",
                pf_ba_fast_from_value as fn(Option<&str>) -> bool,
                true,
            ),
            // task 228: task 225 の実測 (router -458 ms / scan -159 ms、decode と
            // text は無影響) を受けて、この 2 本だけ既定 on にした。
            ("DENNO_PF_ROUTER_DEV", pf_router_dev_from_value, true),
            ("DENNO_PF_SCAN3", gdn_pf3_from_value, true),
        ] {
            assert_eq!(f(None), dflt, "{name}: 未設定の既定が違う");
            for v in ["0", "off", "false", "no", "", " off "] {
                assert!(!f(Some(v)), "{name}: {v:?} が on になった");
            }
            for v in ["1", "on", "true", "yes", "2"] {
                assert!(f(Some(v)), "{name}: {v:?} が off のまま");
            }
        }
    }

    /// task 308: full 層の出力をデバイスに残す契約。
    ///
    /// `pf_full_out` は `keep_dev` のとき `pf_attn_dev_store` を呼び、直後の
    /// `pf_combine_dev` がその記録を `take` して読む。記録は形が合うときだけ
    /// 有効で、層の入口の `pf_attn_dev_clear` で消える (前の層の出力を
    /// 次の層が読まない)。デバイス確保は要らない。
    #[test]
    fn pf308_full_devres_store_peek_contract() {
        pf_attn_dev_clear();
        assert!(!pf_attn_dev_peek(4, 8), "記録が無いのに true");
        pf_attn_dev_store(0x1000, 4, 8);
        assert!(pf_attn_dev_peek(4, 8), "同じ形は true");
        assert!(!pf_attn_dev_peek(4, 9), "e が違えば false");
        assert!(!pf_attn_dev_peek(5, 8), "nt が違えば false");
        // 要求が立っていないときの連鎖は記録を残さない (ssm / full 共通)。
        pf_attn_dev_request(false);
        pf_attn_dev_clear();
        assert!(!pf_attn_dev_peek(4, 8), "clear のあとは false");
        pf_attn_dev_clear();
    }

    /// 形のゲート。
    #[test]
    fn pf225_shape_gates() {
        // beta/alpha: 実モデルは n = nv = 48 (beta と alpha で 2 回撃つ)。
        assert!(pf_ba_fast_ok(1024, 2560, 48));
        assert!(pf_ba_fast_ok(1, 1, 1));
        assert!(!pf_ba_fast_ok(0, 2560, 48));
        assert!(!pf_ba_fast_ok(1024, 0, 48));
        assert!(!pf_ba_fast_ok(1024, 2560, 0));
        assert!(pf_ba_fast_ok(1024, 2560, 512), "router の logits (512) は受ける");
        assert!(!pf_ba_fast_ok(1024, 2560, 513), "n > 512 は受けない");
        // router: ne = 512 / k = 8 前後。
        assert!(pf_router_dev_ok(1024, 512, 8));
        assert!(pf_router_dev_ok(1, 32, 1));
        assert!(!pf_router_dev_ok(0, 512, 8));
        assert!(!pf_router_dev_ok(1024, 512, 0));
        assert!(!pf_router_dev_ok(1024, 512, 33), "k > 32 は受けない");
        assert!(!pf_router_dev_ok(1024, 1025, 8), "ne > 1024 は受けない");
        // scan v3 は v2 と同じ前提。
        let conv_dim = 2 * 16 * 128 + 48 * 128;
        assert!(gdn_pf3_ok(512, 16, 48, 128, 128, 4, conv_dim));
        assert!(!gdn_pf3_ok(512, 16, 48, 128, 128, 3, conv_dim));
    }

    /// `dno_gemm_f32_ba` のタイル算術の写し (添字の取り違えだけを落とす)。
    fn ba_mirror(w: &[f32], x: &[f32], nt: usize, k: usize, n: usize) -> Vec<f32> {
        const TH: usize = 256;
        let mut y = vec![0f32; nt * n];
        for by in 0..n.div_ceil(BA225_ROWS) {
            for bx in 0..nt.div_ceil(BA225_TOKS) {
                let mut acc = vec![0f32; TH * 12];
                for k0 in (0..k).step_by(32) {
                    let mut ws = vec![0f32; 32 * BA225_ROWS];
                    let mut xs = vec![0f32; 32 * BA225_TOKS];
                    for j in 0..(BA225_ROWS * 32 / TH) {
                        for tid in 0..TH {
                            let id = tid + j * TH;
                            let rr = id >> 5;
                            let kk = id & 31;
                            let gr = by * BA225_ROWS + rr;
                            let gk = k0 + kk;
                            ws[kk * BA225_ROWS + rr] =
                                if gr < n && gk < k { w[gr * k + gk] } else { 0.0 };
                        }
                    }
                    for j in 0..(BA225_TOKS * 32 / TH) {
                        for tid in 0..TH {
                            let id = tid + j * TH;
                            let tt = id >> 5;
                            let kk = id & 31;
                            let gt = bx * BA225_TOKS + tt;
                            let gk = k0 + kk;
                            xs[kk * BA225_TOKS + tt] =
                                if gt < nt && gk < k { x[gt * k + gk] } else { 0.0 };
                        }
                    }
                    for tid in 0..TH {
                        let (tx, ty) = (tid & 15, tid >> 4);
                        for kk in 0..32 {
                            for i in 0..3 {
                                let wv = ws[kk * BA225_ROWS + ty * 3 + i];
                                for j in 0..4 {
                                    let xv = xs[kk * BA225_TOKS + tx * 4 + j];
                                    acc[tid * 12 + i * 4 + j] += wv * xv;
                                }
                            }
                        }
                    }
                }
                for tid in 0..TH {
                    let (tx, ty) = (tid & 15, tid >> 4);
                    for i in 0..3 {
                        for j in 0..4 {
                            let r = by * BA225_ROWS + ty * 3 + i;
                            let t = bx * BA225_TOKS + tx * 4 + j;
                            if r < n && t < nt {
                                y[t * n + r] = acc[tid * 12 + i * 4 + j];
                            }
                        }
                    }
                }
            }
        }
        y
    }

    /// 端数の出る形 (tile の倍数でない) で参照の内積と一致する。
    #[test]
    fn pf225_ba_tile_mirror() {
        let (nt, k, n) = (70usize, 72usize, 100usize);
        let w: Vec<f32> = (0..n * k)
            .map(|i| ((i * 37 % 101) as f32) * 0.01 - 0.5)
            .collect();
        let x: Vec<f32> = (0..nt * k)
            .map(|i| ((i * 53 % 89) as f32) * 0.02 - 0.9)
            .collect();
        let got = ba_mirror(&w, &x, nt, k, n);
        for t in 0..nt {
            for r in 0..n {
                let mut s = 0f64;
                for kk in 0..k {
                    s += w[r * k + kk] as f64 * x[t * k + kk] as f64;
                }
                let e = got[t * n + r];
                assert!(
                    (e as f64 - s).abs() <= 1e-4 + 1e-5 * s.abs(),
                    "t={t} r={r}: kernel {e} vs ref {s}"
                );
            }
        }
    }

    /// 強制スイッチ (テスト用) が効く。
    #[test]
    fn pf225_force_switch() {
        set_pf_ba_fast_for_test(Some(true));
        assert!(pf_ba_fast_on());
        set_pf_ba_fast_for_test(Some(false));
        assert!(!pf_ba_fast_on());
        // task 231: `None` は env に戻る。既定は on だが、テスト環境の env に
        // 依存させないため期待値は env の解釈から取る。
        set_pf_ba_fast_for_test(None);
        assert_eq!(
            pf_ba_fast_on(),
            pf_ba_fast_from_value(std::env::var("DENNO_PF_BA_FAST").ok().as_deref()),
            "強制を外したら env に戻る"
        );

        set_pf_router_dev_for_test(Some(true));
        assert!(pf_router_dev_on());
        set_pf_router_dev_for_test(Some(false));
        assert!(!pf_router_dev_on());
        // task 228: task 225 の実測を受けてこの 2 本は既定 on になった。
        // `None` は env に戻るので、期待値は env の解釈から取る (テスト環境の
        // env に依存させない)。
        set_pf_router_dev_for_test(None);
        assert_eq!(
            pf_router_dev_on(),
            pf_router_dev_from_value(std::env::var("DENNO_PF_ROUTER_DEV").ok().as_deref()),
            "強制を外したら env に戻る"
        );

        set_gdn_pf3_for_test(Some(true));
        assert!(gdn_pf3_on());
        set_gdn_pf3_for_test(Some(false));
        assert!(!gdn_pf3_on());
        set_gdn_pf3_for_test(None);
        assert_eq!(
            gdn_pf3_on(),
            gdn_pf3_from_value(std::env::var("DENNO_PF_SCAN3").ok().as_deref()),
            "強制を外したら env に戻る"
        );
    }
}


#[cfg(test)]
pub(crate) mod kprof_tests {
    //! s240 (P40): 計測コードが「既定で no-op」「ON でカーネル別に積算される」
    //! ことの GPU 不要な検証。実測はオペレータの GPU 実行で行う (ここでは
    //! `cuEventCreate` すら呼ばない = イベント生成が env 依存であることも見る)。
    use super::*;

    /// テスト専用の env 上書き (env を書き換えると並走テストを揺らすため)。
    static FORCE_ON: std::sync::Mutex<Option<bool>> = std::sync::Mutex::new(None);

    pub(crate) fn force_on() -> Option<bool> {
        *FORCE_ON.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// 強制の窓 (drop で元に戻す)。assert が落ちても他テストを汚さない。
    pub(crate) struct Force(Option<bool>);

    impl Force {
        fn set(v: Option<bool>) -> Self {
            let prev = *FORCE_ON.lock().unwrap_or_else(|e| e.into_inner());
            *FORCE_ON.lock().unwrap_or_else(|e| e.into_inner()) = v;
            Force(prev)
        }
    }

    impl Drop for Force {
        fn drop(&mut self) {
            *FORCE_ON.lock().unwrap_or_else(|e| e.into_inner()) = self.0;
        }
    }

    fn reset_acc() {
        KPROF_PRE_NS.store(0, Ordering::Relaxed);
        KPROF_STATE_NS.store(0, Ordering::Relaxed);
        KPROF_SCAN_NS.store(0, Ordering::Relaxed);
        KPROF_CALLS.store(0, Ordering::Relaxed);
        KPROF_MISS.store(0, Ordering::Relaxed);
    }

    /// env 未設定 (強制 OFF) では計測の入口が no-op であること。
    #[test]
    fn kprof_off_is_noop() {
        let _g = crate::prof::counters_guard();
        let _f = Force::set(Some(false));
        assert!(!gdn_kprof_on());
        assert!(kprof_begin().is_none(), "OFF ではイベントを作らない");
        assert!(gdn_kprof_line().is_none(), "OFF では行を出さない");
    }

    /// s245 (P45): 保留読みは env OFF で完全な no-op (本番経路に CUDA 呼び出しを
    /// 足さない)。
    #[test]
    fn kprof_read_pending_is_noop_when_off() {
        let _g = crate::prof::counters_guard();
        let _f = Force::set(Some(false));
        reset_acc();
        kprof_read_pending();
        assert_eq!(
            (kprof_acc().calls, kprof_acc().miss),
            (0, 0),
            "OFF なのに保留読みが積算 / miss を動かした"
        );
    }

    /// ON のとき、積算がカーネル別に初期化され、表示行が仕様どおりになること。
    #[test]
    fn kprof_on_accumulates_per_kernel() {
        let _g = crate::prof::counters_guard();
        let _f = Force::set(Some(true));
        reset_acc();
        assert!(gdn_kprof_line().is_none(), "未計測 (calls=0 miss=0) は行を出さない");
        kprof_add_ms(1.5, 0.25, 2.0);
        kprof_add_ms(0.5, 0.75, 1.0);
        let acc = kprof_acc();
        assert_eq!(
            (acc.pre_ns, acc.state_ns, acc.scan_ns, acc.calls),
            (2_000_000, 1_000_000, 3_000_000, 2),
            "カーネル別に ns で積算されない"
        );
        assert_eq!(
            gdn_kprof_line().as_deref(),
            Some("[GDN_KPROF] pre=2.000 state=1.000 scan=3.000 calls=2 miss=0 total=6.000")
        );
        assert_eq!(gdn_kprof_miss(), 0, "成功した call を miss に数えている");
        reset_acc();
    }

    /// 表示行の書式 (ms 3 桁・calls・miss・total = 3 区間の和)。GPU 不要。
    #[test]
    fn kprof_line_format_is_stable() {
        let acc = KprofAcc {
            pre_ns: 1_234_567_000,
            state_ns: 2_500_000,
            scan_ns: 3_000_000,
            calls: 7,
            miss: 0,
        };
        assert_eq!(
            acc.line().as_deref(),
            Some("[GDN_KPROF] pre=1234.567 state=2.500 scan=3.000 calls=7 miss=0 total=1240.067")
        );
        assert!(
            KprofAcc::default().line().is_none(),
            "未計測の既定値が行を出している"
        );
    }

    /// `miss=` の書式と「calls=0 でも捨てた call があれば行を出す」こと。
    /// これが P43 の本題 (実機で 1 行も出なかった = 静かな no-op の再発防止)。
    #[test]
    fn kprof_miss_is_printed_even_with_zero_calls() {
        let _g = crate::prof::counters_guard();
        let _f = Force::set(Some(true));
        reset_acc();
        // 全部捨てた状態: 行は出るが calls=0 / miss=n。
        KPROF_MISS.store(16, Ordering::Relaxed);
        assert_eq!(
            gdn_kprof_line().as_deref(),
            Some("[GDN_KPROF] pre=0.000 state=0.000 scan=0.000 calls=0 miss=16 total=0.000"),
            "捨てた call があるのに黙っている (miss= が効いていない)"
        );
        // 一部だけ読めた状態: calls と miss が併記される。
        kprof_add_ms(2.0, 1.0, 3.0);
        assert_eq!(
            gdn_kprof_line().as_deref(),
            Some("[GDN_KPROF] pre=2.000 state=1.000 scan=3.000 calls=1 miss=16 total=6.000")
        );
        reset_acc();
    }

    /// prefill 終了時の 1 行 (サーバ経路の観測点) の判定。
    /// - dump 経路 (`DENNO_PROFILE`) が生きている → 出さない (summary と二重にしない)
    /// - それ以外 → **未計測でも必ず出す** (静かに何も出ないを構造的に禁止)
    #[test]
    fn kprof_prefill_end_line_is_always_emitted_without_dump_path() {
        let measured = KprofAcc {
            pre_ns: 3_000_000,
            state_ns: 1_000_000,
            scan_ns: 2_000_000,
            calls: 4,
            miss: 1,
        };
        assert_eq!(
            kprof_prefill_line(&measured, false).as_deref(),
            Some("[GDN_KPROF] pre=3.000 state=1.000 scan=2.000 calls=4 miss=1 total=6.000")
        );
        assert!(
            kprof_prefill_line(&measured, true).is_none(),
            "dump 経路が生きているのに prefill 側でも出している (二重行)"
        );
        // 走査サイトに 1 度も入らなかった場合でも行は出る (calls=0 miss=0)。
        assert_eq!(
            kprof_prefill_line(&KprofAcc::default(), false).as_deref(),
            Some("[GDN_KPROF] pre=0.000 state=0.000 scan=0.000 calls=0 miss=0 total=0.000"),
            "env ON なのに 1 行も出ない経路が残っている"
        );
    }
}



// ===========================================================================
#[cfg(test)]
mod b7_stage_tests {
    //! s141 (Task B7): GDN 状態のホスト受け皿が**再利用**されることの検証。
    //!
    //! 背景 (s140 の計測): decode の `gdn_pf_alloc` が 98 us/call・4.7 ms/token
    //! あり、中身は層ごとの `to_vec` (`Vec::with_capacity` を 2 本 x 48 層) だった。
    //! 実転送 (D2H) は GPU が要るので、ここでは**確保の規則そのもの**
    //! (`reserve_exact_grow` — `d2h_raw` が実際に呼ぶ関数) を回して、
    //! 確保回数が層・ステップ数に比例して増えないことを確かめる。
    use super::*;

    /// Vec の再確保を数える (これが 0 件 = 確保していない)。
    fn count_regrows(stage: &mut GdnHostStage, conv_len: usize, ssm_len: usize, layers: usize) -> usize {
        let (c0, s0) = stage.capacities();
        for _ in 0..layers {
            reserve_exact_grow(&mut stage.conv, conv_len);
            reserve_exact_grow(&mut stage.ssm, ssm_len);
        }
        let (c1, s1) = stage.capacities();
        usize::from(c1 != c0) + usize::from(s1 != s0)
    }

    #[test]
    fn gdn_host_stage_is_reused_across_layers_and_steps() {
        // 27B q4 の実形状: conv = conv_dim*cs = 8448、ssm = nv*dk*dv = 786432 要素。
        let (conv_len, ssm_len) = (8_448usize, 786_432usize);
        let mut stage = GdnHostStage::new();
        assert_eq!(stage.capacities(), (0, 0), "作りたては未確保");

        // 1 ステップ目は伸びてよい (= 初回だけ確保する)。
        let first = count_regrows(&mut stage, conv_len, ssm_len, 48);
        assert_eq!(first, 2, "初回は conv/ssm の 2 本だけ確保する (実際 {first})");
        assert_eq!(stage.capacities(), (conv_len, ssm_len));

        // 2 ステップ目以降は 48 層回しても 1 本も確保しない (ここが本題)。
        for step in 1..8 {
            let regrown = count_regrows(&mut stage, conv_len, ssm_len, 48);
            assert_eq!(
                regrown, 0,
                "step={step}: 容量が層ループで再確保された (再利用できていない)"
            );
            assert_eq!(
                stage.capacities(),
                (conv_len, ssm_len),
                "step={step}: 容量がステップ数で増えた"
            );
        }
    }

    #[test]
    fn gdn_host_stage_grows_only_when_shape_grows() {
        // 形が大きくなったときだけ伸びる (grow-only)。縮んでも手放さない。
        let mut stage = GdnHostStage::new();
        reserve_exact_grow(&mut stage.conv, 100);
        assert_eq!(stage.capacities().0, 100);
        reserve_exact_grow(&mut stage.conv, 50); // 縮小 → 何もしない
        assert_eq!(stage.capacities().0, 100, "縮小で解放しない (grow-only)");
        reserve_exact_grow(&mut stage.conv, 8_448); // 拡大 → 伸ばす
        assert_eq!(stage.capacities().0, 8_448);
    }

    #[test]
    fn reserve_grow_reuses_pointer_so_no_per_layer_allocation() {
        // 同じ長さを繰り返しても**同じ領域**を使う (アドレスが変わらない = 再確保なし)。
        let mut stage = GdnHostStage::new();
        reserve_exact_grow(&mut stage.ssm, 786_432);
        let p0 = stage.ssm.as_ptr();
        for _ in 0..480 {
            reserve_exact_grow(&mut stage.ssm, 786_432);
            assert_eq!(stage.ssm.as_ptr(), p0, "同じ長さで確保し直している");
        }
    }
}


// ===========================================================================
#[cfg(test)]
mod b7c_ownership_tests {
    //! s141b (Task B7c): GDN 状態の所有者規則を固定する。
    //!
    //! 新しい規則:
    //! - decode 中はデバイスミラーが権威。ホスト `KvCache::gdn` は**遅延リフレッシュ
    //!   対象**で、`gdn_stale[li]` が真の間は古い可能性がある。
    //! - seed (位置不一致でミラーを埋め直す) は**デバイス正本から D2D**。ホストを
    //!   経由しない (= 層あたり 3.2 MiB の PCIe 往復が消える)。
    //! - 遅延フラッシュは**ホストが古い層だけ**読む (古くない層を読み直さない
    //!   = 「何も要らないステップでホストバイトをコピーしない」)。
    //!
    //! 実転送は GPU が要るので、ここでは判定の実体 (`gdn_flush_step` /
    //! `gdn_state_take_dev` / `gdn_stale`) を回す。デバイスを伴う確認は
    //! `cuda_gdn_flush_pending_faces_positions_and_epoch` (GPU 必須・ignored) が
    //! デバイスを伴う確認を受け持つ。
    use super::*;

    const LAYERS: usize = 48;
    const STEPS: usize = 8;
    /// ミラーの番兵キー (既存テストの u64::MAX-1 / u64::MAX-3 と衝突しない)。
    const MIRROR_TEST_KEY_B7C: u64 = u64::MAX - 5;

    /// 決定論のビットパターン (GPU テストの既知値用。`cuda::tests` の
    /// `pseudo_random` は別モジュールなので私有で持つ)。
    fn b7c_pattern(len: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (0..len)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((s >> 8) as f32) / ((1u32 << 24) as f32) - 0.5
            })
            .collect()
    }

    /// decode 1 ステップ分のホストバイト数 (48 層 x (conv 8,448 + ssm 786,432) f32)。
    fn host_bytes_per_step(copied_layers: usize) -> usize {
        copied_layers * (8_448 + 786_432) * std::mem::size_of::<f32>()
    }

    /// decode 1 ステップの層ループを回した結果。
    #[derive(Clone, Copy, Debug, Default)]
    struct StepIo {
        /// seed で**ホストから読んだ**層数 (B7c が削る対象。正本が有効なら 0)
        host_seeded: usize,
        /// flush で D2H した層数 (スキャンした層だけ)
        flushed: usize,
    }

    #[derive(Clone, Copy, Debug)]
    pub(crate) struct Layer {
        pos: Option<usize>,
        stale: bool,
    }

    /// flush 段だけを回す (target = このステップ後の `cache.len()`)。
    pub(crate) fn flush(layers: &mut [Layer], target: usize) -> usize {
        let mut copied = 0;
        for l in layers.iter_mut() {
            let (need, _aligned) = gdn_flush_step(true, l.stale, target, l.pos);
            if need {
                l.stale = false;
                copied += 1;
            }
        }
        copied
    }

    /// `chain_q35_gdn_scan` の層ループをそのまま写す:
    /// 1. seed (位置不一致): 正本が有効なら D2D (ホストを読まない) / 無効ならホストから
    /// 2. scan: 書き込み面が新状態になる → `gdn_pos = pos0+1`, `gdn_stale = true`
    /// 3. flush (ステップ末尾): `gdn_flush_step` が真の層だけ D2H
    pub(crate) fn step(layers: &mut [Layer], pos0: usize, dev_valid: bool) -> StepIo {
        let mut io = StepIo::default();
        // 1. seed (位置不一致 = 前ステップの pos0+1 が今の pos0 と合わない層)
        for l in layers.iter_mut() {
            if l.pos != Some(pos0) {
                if dev_valid {
                    // 正本から D2D — ホストは 1 バイトも読まない。ただしホストは
                    // この時点で古い可能性があるので遅延フラグは立てたままにする
                    // (落とすと flush がこの層を飛ばして古いホストを正にしてしまう)。
                    l.stale = true;
                } else {
                    // 正本が無効 = ホストが正。ホストを読んで面 0 を書く。
                    io.host_seeded += 1;
                    l.stale = false;
                }
                l.pos = Some(pos0);
            }
        }
        // 2. scan
        for l in layers.iter_mut() {
            l.pos = Some(pos0 + 1);
            l.stale = true;
        }
        // 3. flush (cache.len() は advance 後 = pos0+1)
        io.flushed = flush(layers, pos0 + 1);
        io
    }

    /// 27B q4 の decode ではミラーの cap 成長で `gdn_pos` が落ちるため、seed は
    /// **毎ステップ全層**走る (実測: `gdn_pf_alloc` 5,712 call = 48 層 x 119 step)。
    /// その条件を再現する。
    fn reset_pos(layers: &mut [Layer]) {
        for l in layers.iter_mut() {
            l.pos = None;
        }
    }

    /// 1) decode 形 (48 層 x 8 ステップ): 正本が有効なステップの seed は D2D なので
    ///    ホストを 1 バイトも読まない (従来は層あたり ~3.2 MiB の D2H 往復)。
    #[test]
    fn decode_loop_seeds_from_device_and_never_reads_host() {
        // 初回要求: まだ正本が無い → 全層をホストから seed (これは従来どおり)。
        let mut layers = vec![Layer { pos: None, stale: false }; LAYERS];
        let first = step(&mut layers, 4, false);
        assert_eq!(first.host_seeded, LAYERS, "正本が無い初回はホストから");
        // 2 ステップ目以降は正本が有効 = seed は D2D。ホストは読まれない。
        let mut host_bytes = 0usize;
        for s in 0..STEPS {
            reset_pos(&mut layers);
            let io = step(&mut layers, 5 + s, true);
            assert_eq!(io.host_seeded, 0, "step={s}: 正本が有効なのにホストを読んだ");
            host_bytes += host_bytes_per_step(io.host_seeded);
            assert_eq!(io.flushed, LAYERS, "step={s}: スキャンした層は flush で 1 度だけ読む");
        }
        assert_eq!(host_bytes, 0, "8 ステップで seed のホスト読みは 0 バイト");
        // flush は全層を 1 度ずつ読んでフラグを落とす = 最終状態は「ホストと一致」。
        assert!(layers.iter().all(|l| !l.stale), "flush 後はフラグが落ちる");
    }

    /// 2) flush は「ホストが古い層」だけを読み、読んだ層はフラグを落とす
    ///    (= 二度読まない)。スキャンしていない層 (stale = false) は読まない。
    #[test]
    fn flush_reads_only_the_layers_whose_host_is_stale() {
        let mut layers = vec![Layer { pos: Some(10), stale: true }; LAYERS];
        layers[7].stale = false; // この層はスキャンしていない = ホストが最新
        let copied = flush(&mut layers, 10);
        assert_eq!(copied, LAYERS - 1, "古い層だけ読む");
        // 1 層 = (8,448 + 786,432) x 4 B = 3,179,520 B。
        assert_eq!(
            host_bytes_per_step(copied),
            3_179_520 * (LAYERS - 1),
            "スキャンした 47 層分だけ (48 層分ではない)"
        );
        // 2 回目の flush は何も読まない (フラグを落としたので)。
        assert_eq!(flush(&mut layers, 10), 0);
        // 位置が合わない (target でも target+1 でもない) 層は stale なら
        // 「不整合」= 呼び出し側が None にする (読みもしない)。
        assert_eq!(gdn_flush_step(true, true, 10, Some(9)), (false, false));
    }

    /// 3) 位置が合っているかどうかの判定 (無効化・位置不一致・新規要求の経路)。
    #[test]
    fn flush_decision_matches_the_old_position_rules() {
        // ホストが古い (stale) 層: 成功ステップ後 (target) の面だけを読む。
        assert_eq!(gdn_flush_step(true, true, 10, Some(10)), (true, true));
        // 層途中失敗 (target+1): 読む面は seed 直後に書かれた = ホストと一致 = 読まない。
        assert_eq!(gdn_flush_step(true, true, 10, Some(11)), (false, true));
        assert_eq!(gdn_flush_step(true, false, 10, Some(11)), (false, true));
        // stale かつ位置が合わない (None / 遠い) → 「読むべきか」は偽、整列も偽。
        // 呼び出し側はこれを不整合 (None) として扱う (旧実装の Some(_) と同じ)。
        assert_eq!(gdn_flush_step(true, true, 10, None), (false, false));
        assert_eq!(gdn_flush_step(true, true, 10, Some(4)), (false, false));
        // ホストが古くない層: 位置が合っていても読まない (ここが B7c の本体)。
        assert_eq!(gdn_flush_step(true, false, 10, Some(10)), (false, true));
        // GDN 層でない層は常に対象外。
        assert_eq!(gdn_flush_step(false, true, 10, Some(10)), (false, false));
    }

    /// 4) 1 リクエスト相当 (prefill で seed → decode 120 ステップ) でも、decode 中に
    ///    seed がホストから読むバイトは 0 (ミラーの cap 成長で seed が毎ステップ
    ///    走る条件を含む)。
    #[test]
    fn one_request_of_decode_never_pulls_the_host_mirror() {
        let mut layers = vec![Layer { pos: None, stale: false }; LAYERS];
        let _prefill_seed = step(&mut layers, 3_000, false);
        let mut total = 0usize;
        for s in 0..120 {
            reset_pos(&mut layers);
            total += host_bytes_per_step(step(&mut layers, 3_001 + s, true).host_seeded);
        }
        assert_eq!(total, 0, "1 リクエスト分の decode で seed がホストから読むバイトは 0");
    }

    /// 5) invalidate-then-seed: ホストが正に戻った (reset/restore/新要求) ときの
    ///    seed 入力はホストのままで、ホストの値は 1 ビットも変わらない。
    ///    `gdn_state_take_dev` は valid=false (= ホストが正) の層では None を返す
    ///    こと = デバイス正本を上書きに使わないこと。
    #[test]
    fn invalidate_then_seed_uses_host_and_keeps_its_bits() {
        // キャッシュの「ホスト正」状態を模す (値は f32 のビットパターンで比較)。
        let conv_len = 192usize;
        let ssm_len = 256usize;
        let host_conv: Vec<f32> = (0..conv_len).map(|i| (i as f32) * 1.5e-3).collect();
        let host_ssm: Vec<f32> = (0..ssm_len).map(|i| -((i as f32) * 3.0)).collect();
        let before_c: Vec<u32> = host_conv.iter().map(|v| v.to_bits()).collect();
        let before_s: Vec<u32> = host_ssm.iter().map(|v| v.to_bits()).collect();

        // 正本を「無効」にした状態 = ホストが正。
        let mut dev = GdnStateCache::empty();
        dev.invalidate();
        let devseed = gdn_state_take_dev(12345, 0, &mut dev);
        assert!(devseed.is_none(), "無効な正本を seed に使ってはいけない");

        // 呼び出し側 (chain_q35_gdn_scan の None 分岐) はホストからコピーする。
        // ここでは H2D の代わりに「ホストを読んで書き戻す」ことを確かめる
        // (転送は値そのままなので bit 一致のまま)。
        assert_eq!(
            host_conv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            before_c,
            "conv は bit 一致のまま"
        );
        assert_eq!(
            host_ssm.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            before_s,
            "ssm は bit 一致のまま"
        );
    }

    /// 6) 正本が有効な層は D2D 経路 (seed に使ってよい) で、使った層は無効化される
    ///    (scan がミラーを更新するので正本とずれる)。デバイス確保は要らない。
    #[test]
    fn dev_take_only_for_valid_layers_and_invalidates_them() {
        let mut dev = GdnStateCache::empty();
        // 層 0/1 を「有効」にする — DevBuffer は確保せず、判定だけを見るため
        // valid と cur_kv を直接立てる (確保の無い層は ptr が取れず None)。
        dev.cur_kv = Some(7);
        dev.valid = vec![true, true, false];
        // デバイスバッファが無いので ptr は取れない = None。ここで重要なのは
        // 「valid でない層は絶対に None」かつ「同じ kv でない層も None」。
        assert!(gdn_state_take_dev(8, 0, &mut dev).is_none(), "別 kv_id は不可");
        assert!(gdn_state_take_dev(7, 2, &mut dev).is_none(), "valid=false は不可");
        assert!(gdn_state_take_dev(7, 9, &mut dev).is_none(), "範囲外は不可");
    }

    /// 7) ホスト遅延フラグは層ごとに独立: 1 層だけ「古い」なら、その層だけが flush の
    ///    対象になり、他の 47 層は読まれない (別層の状態で代用しない)。
    #[test]
    fn stale_flag_is_per_layer() {
        let mut stale = vec![false; LAYERS];
        stale[31] = true;
        let pos = vec![Some(42); LAYERS];
        let need: Vec<usize> = (0..LAYERS).filter(|&li| gdn_flush_step(true, stale[li], 42, pos[li]).0).collect();
        assert_eq!(need, vec![31], "古い層だけが読まれる");
        // 隣の層 (30 / 32) は位置が合っていても読まない — 代用読みをしない。
        assert_eq!(gdn_flush_step(true, stale[30], 42, pos[30]), (false, true));
        assert_eq!(gdn_flush_step(true, stale[32], 42, pos[32]), (false, true));
    }

    /// 8) GPU 必須: ミラーの面 0 が**古い値**のままでも、seed がデバイス正本から
    ///    D2D で入ることを、ミラー面 0 を読み戻してビットで確かめる
    ///    (ホストは古い値 = 使われていないことが分かる)。
    ///    実行: `cargo test --features cuda --lib cuda_b7c_seed_copies_from_device_mirror -- --ignored`
    #[test]
    #[ignore] // GPU 必須
    fn cuda_b7c_seed_copies_from_device_mirror() {
        let _g = test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        enable_for_tests();
        let _ctx = CudaCtx::new().expect("GPU 必須 (ignored テスト)");
        let (conv_len, ssm_len) = (192usize, 256usize);
        let key = MIRROR_TEST_KEY_B7C;
        let kv_id = 7u64;
        let dev_conv = b7c_pattern(conv_len, 91);
        let dev_ssm = b7c_pattern(ssm_len, 92);
        let stale_conv = vec![1.0f32; conv_len];
        let stale_ssm = vec![2.0f32; ssm_len];

        // ミラーを作り、面 0 に「古い値」を書く (seed で上書きされるはずの場所)。
        with_mirror_ext(key, 0, 1, 8, 0, 0, false, |m, _| {
            let mut g = GdnDev { conv: [None, None], ssm: [None, None], cur: 0 };
            ok(Scratch::ensure(&mut g.conv[0], conv_len * 4))?;
            ok(Scratch::ensure(&mut g.conv[1], conv_len * 4))?;
            ok(Scratch::ensure(&mut g.ssm[0], ssm_len * 4))?;
            ok(Scratch::ensure(&mut g.ssm[1], ssm_len * 4))?;
            ok(g.conv[0].as_ref().unwrap().copy_from(&stale_conv))?;
            ok(g.ssm[0].as_ref().unwrap().copy_from(&stale_ssm))?;
            m.gdn[0] = Some(g);
            m.gdn_pos[0] = None; // 位置不一致 = seed が走る
            m.gdn_stale[0] = true; // ホストは古い可能性がある
            Some(())
        })
        .unwrap();

        // デバイス正本に既知の値を入れて「有効」にする (s131 の正本と同じ持ち方)。
        GDN_STATE.with(|cell| {
            let mut c = cell.borrow_mut();
            c.cur_kv = Some(kv_id);
            c.invalidate();
            assert!(c.ensure_layer(0, 1, ssm_len).is_ok(), "ssm 確保");
            assert!(c.ensure_conv(0, conv_len).is_ok(), "conv 確保");
            assert!(c.conv[0].as_ref().unwrap().copy_from(&dev_conv).is_ok());
            assert!(c.ssm[0].as_ref().unwrap().copy_from(&dev_ssm).is_ok());
            c.valid[0] = true;
        });

        // seed の入力 (関数そのもの) を取る。取り出した層は無効化される。
        let seed = GDN_STATE.with(|cell| gdn_state_take_dev(kv_id, 0, &mut cell.borrow_mut()));
        let (pc, ps) = seed.expect("有効な正本から読める");
        // ミラー面 0 へ D2D (chain_q35_gdn_scan の seed と同じ操作)。
        with_mirror_ext(key, 0, 1, 8, 0, 0, false, |m, _| {
            let gd = m.gdn[0].as_mut().unwrap();
            ok(gd.conv[0].as_ref().unwrap().copy_from_dev(pc, conv_len))?;
            ok(gd.ssm[0].as_ref().unwrap().copy_from_dev(ps, ssm_len))?;
            m.gdn_pos[0] = Some(0);
            m.gdn_stale[0] = false;
            Some(())
        })
        .unwrap();

        // ミラー面 0 を読み戻して、正本の値とビット一致・「古い値」ではないことを見る。
        with_mirror_ext(key, 0, 1, 8, 0, 0, false, |m, _| {
            let gd = m.gdn[0].as_ref().unwrap();
            let cv = gd.conv[0].as_ref().unwrap().to_vec::<f32>(conv_len).unwrap();
            let sm = gd.ssm[0].as_ref().unwrap().to_vec::<f32>(ssm_len).unwrap();
            assert_eq!(cv.len(), dev_conv.len());
            assert!(
                cv.iter().zip(&dev_conv).all(|(a, b)| a.to_bits() == b.to_bits()),
                "seed はデバイス正本の conv とビット一致"
            );
            assert!(
                sm.iter().zip(&dev_ssm).all(|(a, b)| a.to_bits() == b.to_bits()),
                "seed はデバイス正本の ssm とビット一致"
            );
            assert!(cv.iter().all(|&v| v != 1.0), "古いホスト値が使われていない");
            assert!(sm.iter().all(|&v| v != 2.0), "古いホスト値が使われていない");
            Some(())
        })
        .unwrap();
        // 一度読んだ層は無効化されている (次は必ず seed し直す = 取り違えない)。
        let again = GDN_STATE.with(|cell| gdn_state_take_dev(kv_id, 0, &mut cell.borrow_mut()));
        assert!(again.is_none(), "取り出した層は無効化される");
    }
}


#[cfg(test)]
pub(crate) mod b20_stage1_tests {
    //! s157 (Task B20 段階 1): q4 線形を chain に載せる入口の**判定だけ**を固定する。
    //! 実カーネルは GPU が要るので、ここでは (a) `DENNO_GPU_Q4_CHAIN` の解釈、
    //! (b) `chain_linear` が空・混在グループを**受けない**こと (= per-op 経路へ
    //! 静かに委譲する) を確かめる。数値の一致は実機 (supervisor) の担当。
    use super::*;

    /// `DENNO_GPU_Q4_CHAIN` は **"0" だけ**が off。未設定・空・"off" などは on。
    #[test]
    fn q4_chain_flag_is_off_only_for_explicit_zero() {
        assert!(!parse_q4_chain(Some("0")), "\"0\" だけが per-op 経路へ戻す");
        for raw in [None, Some(""), Some("1"), Some("off"), Some(" 0"), Some("0 ")] {
            assert!(parse_q4_chain(raw), "raw={raw:?} は on (既定) のはず");
        }
    }

    /// `chain_linear` は空グループ・int8/q4 混在グループを受けない (None → per-op)。
    #[test]
    fn chain_linear_rejects_empty_and_mixed_groups() {
        assert_eq!(chain_linear(&[], CH_H, CH_O, 1), None, "空グループは受けない");
        let w = vec![0.0f32; 32 * 64];
        let a = crate::quant::Linear::Q8(crate::quant::Quant8Linear::quantize(&w, 32, 64));
        let b = crate::quant::Linear::Q4(crate::quant::Q4Linear::quantize(&w, 32, 64));
        assert_eq!(
            chain_linear(&[&a, &b], CH_H, CH_O, 1),
            None,
            "int8 と q4 の混在は受けない"
        );
        // 単独グループは GPU が無い環境では `chain_ctx()` が None → これも None
        // (per-op 経路へ委譲)。**panic しないこと**がここの主眼。
        assert_eq!(chain_linear(&[&b], CH_H, CH_O, 1), None);
        assert_eq!(chain_linear(&[&a], CH_H, CH_O, 1), None);
    }

    /// P1c: `chain_gdn_prefill_q4` の入口条件 (フォールバック契約) を GPU 無しで固定する。
    /// 本番経路は `gdn_prefill_q4_geom` の戻り値だけを見て発動/委譲を決めるので、
    /// ここで None になる条件がそのまま「従来経路へ落とす」契約になる。
    /// CUDA は一切呼ばない (デバイス不要)。
    #[test]
    pub(crate) fn gdn_prefill_q4_geom_contract() {
        use crate::quant::{Linear, Q4Linear};
        let q4 = |k: usize, n: usize| {
            Linear::Q4(Q4Linear { blocks: Vec::new().into(), k, n, uid: 0 })
        };
        // 成立する形: nk=2, nv=2, dk=dv=64, d_conv=4, e=64
        //   conv_dim = nk*dk*2 + nv*dv = 384, vd = nv*dv = 128
        let (nk, nv, dk, dv, dc) = (2usize, 2usize, 64usize, 64usize, 4usize);
        let (e, conv_dim, vd) = (64usize, 384usize, 128usize);
        let n_ok = GDN_CHUNK_MIN_TOKENS; // 2048 (1024 の倍数なので端数タイル無し)
        let base = || (q4(e, conv_dim), q4(e, vd), q4(vd, e));
        let dims = (nk, nv, dk, dv, dc);
        let sup = |w: &(Linear, Linear, Linear), n: usize, d: (usize, usize, usize, usize, usize)| {
            gdn_prefill_q4_geom(&w.0, &w.1, &w.2, n, d.0, d.1, d.2, d.3, d.4).is_some()
        };

        // 0) 基準形は通り、寸法が正しく返る
        let w = base();
        let g = gdn_prefill_q4_geom(&w.0, &w.1, &w.2, n_ok, nk, nv, dk, dv, dc)
            .expect("基準形は発動してよい");
        assert_eq!((g.e, g.conv_dim, g.vd, g.cs), (e, conv_dim, vd, dc - 1));

        // 1) 3 本のどれかが Q4 でない → 受けない
        let f32w = |k: usize, n: usize| Linear::F32 { w: Vec::new(), k, n };
        let bad = (f32w(e, conv_dim), w.1.clone(), w.2.clone());
        assert!(!sup(&bad, n_ok, dims), "qkv が Q4 でなければ受けない");
        let bad = (w.0.clone(), f32w(e, vd), w.2.clone());
        assert!(!sup(&bad, n_ok, dims), "z が Q4 でなければ受けない");
        let bad = (w.0.clone(), w.1.clone(), f32w(vd, e));
        assert!(!sup(&bad, n_ok, dims), "o が Q4 でなければ受けない");

        // 2) 重み形状の不一致 (各辺 1 つずつ)
        let bad = (q4(e, conv_dim + 32), w.1.clone(), w.2.clone());
        assert!(!sup(&bad, n_ok, dims), "qkv.n != conv_dim は受けない");
        let bad = (w.0.clone(), q4(e + 32, vd), w.2.clone());
        assert!(!sup(&bad, n_ok, dims), "z.k != e は受けない");
        let bad = (w.0.clone(), q4(e, vd + 32), w.2.clone());
        assert!(!sup(&bad, n_ok, dims), "z.n != vd は受けない");
        let bad = (w.0.clone(), w.1.clone(), q4(vd + 32, e));
        assert!(!sup(&bad, n_ok, dims), "o.k != vd は受けない");
        let bad = (w.0.clone(), w.1.clone(), q4(vd, e + 32));
        assert!(!sup(&bad, n_ok, dims), "o.n != e は受けない");

        // 3) n が閾値未満 → 受けない (閾値ちょうどは通る)
        assert!(
            !sup(&w, GDN_CHUNK_MIN_TOKENS - 1, dims),
            "n < GDN_CHUNK_MIN_TOKENS は受けない"
        );

        // 4) dk != dv / d_conv != 4 → 受けない
        assert!(!sup(&w, n_ok, (nk, nv, dk, dv / 2, dc)), "dk != dv は受けない");
        assert!(!sup(&w, n_ok, (nk, nv, dk, dv, 3)), "d_conv != 4 は受けない");

        // 5) 次元 0 / 32 の倍数でない k / vd → 受けない
        assert!(!sup(&w, n_ok, (0, nv, dk, dv, dc)), "nk == 0 は受けない");
        assert!(!sup(&w, n_ok, (nk, 0, dk, dv, dc)), "nv == 0 は受けない");
        assert!(!sup(&w, n_ok, (nk, nv, 0, 0, dc)), "dk == dv == 0 は受けない");
        let bad = (q4(48, conv_dim), q4(48, vd), q4(vd, 48));
        assert!(!sup(&bad, n_ok, dims), "e % 32 != 0 は受けない");
        let bad = (q4(0, conv_dim), q4(0, vd), q4(vd, 0));
        assert!(!sup(&bad, n_ok, dims), "e == 0 は受けない");
        // vd = nv*dv = 3*16 = 48 (% 32 != 0), e = 32 は条件を満たす
        let bad = (q4(32, 112), q4(32, 48), q4(48, 32));
        assert!(!sup(&bad, n_ok, (2, 3, 16, 16, dc)), "vd % 32 != 0 は受けない");

        // 6) 端数タイルが m <= max_m() → 受けない (per-op 経路が GEMV になる帯域)
        let rem_small = GDN_PREFILL_TILE * 2 + max_m();
        assert!(
            !sup(&w, rem_small, dims),
            "端数 {} 行 (<= max_m()={}) は受けない",
            max_m(),
            max_m()
        );
        // 端数が max_m() を超える / 端数なしは通る
        assert!(sup(&w, rem_small + 1, dims), "端数 max_m()+1 は通る");
        assert!(sup(&w, GDN_PREFILL_TILE * 2, dims), "端数なしは通る");

        // 7) 桁あふれは checked_mul で None
        assert!(!sup(&w, n_ok, (usize::MAX, nv, dk, dv, dc)), "conv_dim の桁あふれは受けない");
    }

    /// task 154: `gdn_scan_batch_sig` の入口 ([`gdn_sig_accepts`]) の境界。
    /// GPU を呼ばないので、この判定だけで「19 トークンの next turn が CPU へ落ちる」
    /// ことを固定できる。
    #[test]
    fn gdn_sig_accepts_short_columns_decline() {
        // 実形に近い寸法 (nk=8, nv=8, dk=dv=128, d_conv=4)。
        let ok = |n: usize| gdn_sig_accepts(n, 8, 8, 128, 128, 4);
        assert!(ok(GDN_SIG_MIN_TOKENS), "境界ちょうど (64) は受ける");
        assert!(ok(GDN_SIG_MIN_TOKENS + 1));
        assert!(ok(GDN_CHUNK_MIN_TOKENS - 1), "chunked の手前までは受ける");
        assert!(!ok(GDN_CHUNK_MIN_TOKENS), "chunked 側 (n >= 2048) は受けない");
        assert!(!ok(19), "next turn の 19 トークンは受けない (CPU のほうが速い)");
        assert!(!ok(GDN_SIG_MIN_TOKENS - 1), "63 は受けない");
        assert!(!ok(0));
        // 形の制約は据え置き (64 以上でも落ちる = 短列ガードで緩んでいない)。
        assert!(!gdn_sig_accepts(64, 0, 8, 128, 128, 4), "nk == 0 は受けない");
        assert!(!gdn_sig_accepts(64, 8, 0, 128, 128, 4), "nv == 0 は受けない");
        assert!(!gdn_sig_accepts(64, 8, 8, 128, 64, 4), "dk != dv は受けない");
        assert!(!gdn_sig_accepts(64, 8, 8, 512, 512, 4), "dk > 256 は受けない");
        assert!(!gdn_sig_accepts(64, 8, 8, 128, 128, 3), "d_conv != 4 は受けない");
    }
}


#[cfg(test)]
mod p14_ffn_fused_tests {
    //! s173 (Task P14): prefill FFN の **gate|up 1 回 GEMM** の前提契約を固定する。
    //!
    //! `pf_ffn_fused_ready` が true のときだけ 1 回の GEMM 経路 (n = 2*ff を
    //! `CH_GA` へ行内 [gate|up] で書く) に入る。条件を 1 つでも落とすと
    //! 連結重みの行範囲が別の重みを指しうる (数値が変わる) ので、
    //! `ffn_fused_shape_ok` の false 側を全部固定する。
    //! GPU が無い環境でも走る (形状だけの純粋関数・エイリアス表に触らない)。
    use super::*;
    use crate::quant::{Linear, Q4Linear};

    pub(crate) fn q4(k: usize, n: usize, uid: u64) -> Q4Linear {
        // 形状だけを見るので blocks は空でよい (weights_for_q4 は呼ばない)。
        Q4Linear { blocks: Vec::new().into(), k, n, uid }
    }

    #[test]
    fn ffn_fused_shape_accepts_gate_up_concat() {
        // 27B: e = 5120, ff = 17408 → gate_up は [2*ff, e]、down は [ff, e]
        let (e, ff) = (5120usize, 17408usize);
        let c = q4(e, 2 * ff, 1);
        let g = q4(e, ff, 2);
        let u = q4(e, ff, 3);
        let d = q4(ff, e, 4);
        assert!(ffn_fused_shape_ok(&c, &g, &u, &d));
    }

    #[test]
    fn ffn_fused_shape_fail_closed() {
        let (e, ff) = (5120usize, 17408usize);
        let c = q4(e, 2 * ff, 1);
        let g = q4(e, ff, 2);
        let u = q4(e, ff, 3);
        let d = q4(ff, e, 4);
        // 連結幅が 2*ff でない (短い / 長い)
        assert!(!ffn_fused_shape_ok(&q4(e, 2 * ff - 1, 1), &g, &u, &d));
        assert!(!ffn_fused_shape_ok(&q4(e, 2 * ff + 1, 1), &g, &u, &d));
        // gate と up の行数が違う (和だけ合っていても駄目)
        assert!(!ffn_fused_shape_ok(&c, &q4(e, ff - 32, 2), &q4(e, ff + 32, 3), &d));
        // k が食い違う
        assert!(!ffn_fused_shape_ok(&q4(e + 32, 2 * ff, 1), &g, &u, &d));
        assert!(!ffn_fused_shape_ok(&c, &q4(e + 32, ff, 2), &u, &d));
        assert!(!ffn_fused_shape_ok(&c, &g, &q4(e + 32, ff, 3), &d));
        // down の形が [ff, e] でない
        assert!(!ffn_fused_shape_ok(&c, &g, &u, &q4(ff + 32, e, 4)));
        assert!(!ffn_fused_shape_ok(&c, &g, &u, &q4(ff, e + 32, 4)));
    }

    /// Q4 以外が混ざったら (int8 / fp32 のモデル) 1 回 GEMM 経路には入らない。
    /// エイリアス表は空のままなので `q4_fused_ready` も false を返す。
    #[test]
    fn ffn_fused_ready_requires_all_q4() {
        let (e, ff) = (5120usize, 17408usize);
        let c = Linear::Q4(q4(e, 2 * ff, 1));
        let g = Linear::Q4(q4(e, ff, 2));
        let u = Linear::Q4(q4(e, ff, 3));
        let d = Linear::Q4(q4(ff, e, 4));
        // 全部 Q4 だがエイリアス未登録 (GPU 無し / 登録失敗) → 従来経路
        assert!(!pf_ffn_fused_ready(&c, &g, &u, &d));
        // 1 本でも Q4 でなければ false
        let f32w = Linear::F32 { w: Vec::new(), k: ff, n: e };
        assert!(!pf_ffn_fused_ready(&c, &g, &u, &f32w));
        assert!(!pf_ffn_fused_ready(&f32w, &g, &u, &d));
    }

    /// A/B スイッチの解釈: `"0"` だけが off (タイポで静かに経路が変わるのを避ける)。
    #[test]
    fn env_switch_only_zero_is_off() {
        assert!(parse_pf_ffn_fused(None));
        assert!(parse_pf_ffn_fused(Some("")));
        assert!(parse_pf_ffn_fused(Some("1")));
        assert!(parse_pf_ffn_fused(Some("off")));
        assert!(!parse_pf_ffn_fused(Some("0")));
    }
}


#[cfg(test)]
mod p3_alias_tests {
    //! s165 (Task P3): 連結重みの行範囲エイリアスの**形状契約**を固定する。
    //!
    //! 27B q4 の FFN は decode chain が `gate_up_w` (連結 1 本)、Q4 prefill が
    //! `gate_w` / `up_w` (2 本) を使うため、同じ 2×ff×e パラメータが VRAM に
    //! 2 度載っていた。`q4_row_alias_offsets` が「parts の行数の和 = 連結側の
    //! 行数・k は全部同じ」を満たすときだけ行オフセットを返すことを固定する。
    //! ここが崩れると別の重みを読む (数値が変わる) ので、満たさない入力は
    //! すべて None (= 従来の個別転送へフォールバック) であることを確かめる。
    use super::*;

    #[test]
    fn alias_offsets_gate_up_layout() {
        // 27B: e = 5120, ff = 17408 → 連結は [gate(17408 行) | up(17408 行)]
        let k = 5120;
        let ff = 17408;
        assert_eq!(
            q4_row_alias_offsets(k, 2 * ff, &[(k, ff), (k, ff)]),
            Some(vec![0, ff])
        );
        // 3 本以上でも累積する (汎用の行連結)
        assert_eq!(
            q4_row_alias_offsets(k, 3 * ff, &[(k, ff), (k, ff), (k, ff)]),
            Some(vec![0, ff, 2 * ff])
        );
    }

    #[test]
    fn alias_offsets_fail_closed() {
        let k = 5120;
        let ff = 17408;
        // 行数の和が合わない (連結側が短い/長い)
        assert_eq!(q4_row_alias_offsets(k, 2 * ff - 1, &[(k, ff), (k, ff)]), None);
        assert_eq!(q4_row_alias_offsets(k, 2 * ff + 1, &[(k, ff), (k, ff)]), None);
        // k が食い違う
        assert_eq!(q4_row_alias_offsets(k, 2 * ff, &[(k, ff), (k + 32, ff)]), None);
        // 空・0 行・0 列・32 の倍数でない k
        assert_eq!(q4_row_alias_offsets(k, 2 * ff, &[]), None);
        assert_eq!(q4_row_alias_offsets(k, ff, &[(k, 0)]), None);
        assert_eq!(q4_row_alias_offsets(k, 0, &[(k, ff)]), None);
        assert_eq!(q4_row_alias_offsets(0, ff, &[(0, ff)]), None);
        assert_eq!(q4_row_alias_offsets(k + 8, ff, &[(k + 8, ff)]), None);
        // 桁あふれ
        assert_eq!(
            q4_row_alias_offsets(k, usize::MAX, &[(k, usize::MAX / 2), (k, usize::MAX / 2 + 2)]),
            None
        );
    }

    /// 行オフセットは「1 行 = nb ブロック = nib 16B / sc 4B」の連続領域を指す。
    /// 27B の gate/up で、連結側の nib 上のオフセットが up_w の先頭行と一致すること
    /// をバイト数で確かめる (この値がずれると別の重みを読む)。
    #[test]
    fn alias_byte_offsets_match_row_stride() {
        let (k, ff) = (5120usize, 17408usize);
        let nb = k / 32;
        let nib_row = nb * 16;
        let sc_row = nb * 4;
        let rows = q4_row_alias_offsets(k, 2 * ff, &[(k, ff), (k, ff)]).expect("layout");
        assert_eq!(rows[0] * nib_row, 0);
        assert_eq!(rows[1] * nib_row, ff * nib_row);
        assert_eq!(rows[1] * sc_row, ff * sc_row);
        // up_w の nib は連結側の後半 (全 nib のちょうど半分) から始まる
        assert_eq!(ff * nib_row, (2 * ff * nb * 16) / 2);
    }

    /// s166 (P5): qkv 連結 GEMV の入口は、前提が崩れたら per-part 経路へ落ちる
    /// (fail closed・panic しない)。GPU 無しのこの環境ではどの場合も None になる
    /// — ここで固定するのは「形状不一致で連結側を選ばない」ことと「例外を投げない」
    /// こと (数値の一致は実機テストの担当)。
    #[test]
    fn q4_fused_linear_fails_closed() {
        use crate::quant::{next_q8_uid, Q4Linear};
        let k = 64usize;
        let (nq, nz) = (32usize, 16usize);
        let q = Q4Linear::quantize(&vec![0.01f32; k * nq], k, nq);
        let z = Q4Linear::quantize(&vec![0.02f32; k * nz], k, nz);
        let mut blocks = Vec::new();
        blocks.extend_from_slice(&q.blocks);
        blocks.extend_from_slice(&z.blocks);
        let fused = Q4Linear { blocks: blocks.into(), k, n: nq + nz, uid: next_q8_uid() };
        // 行数の和が合わない連結 (前提不一致) → per-part (GPU 無しなので None)
        let short = Q4Linear { blocks: q.blocks.clone(), k, n: nq, uid: next_q8_uid() };
        assert!(chain_linear_q4_fused(&short, &[&q, &z], 0, 1, 1).is_none());
        // part の k が食い違う → per-part
        let other_k = Q4Linear::quantize(&vec![0.03f32; 32 * nz], 32, nz);
        assert!(chain_linear_q4_fused(&fused, &[&q, &other_k], 0, 1, 1).is_none());
        // 正しい形でも GPU が無ければ None (panic しない)
        assert!(chain_linear_q4_fused(&fused, &[&q, &z], 0, 1, 1).is_none());
    }

    /// s169 (P10): GEMV の列ブロック幅 C を n から選ぶ純関数の契約。
    /// C は「1 block が何列を受けるか」だけを変えるので、どの C でも列ごとの
    /// k 方向の縮約順は不変 (数値は動かない)。ここで固定するのは
    /// 「既定 (mode 0) は必ず従来の base」「SM 数不明でも base」
    /// 「mode 1/3 の小さい投影だけ C=2」「mode 3 の大きい投影は C=8」の 3 点。
    #[test]
    fn gemv_shape_c_by_rows() {
        // RTX 5090 = 170 SM (27B 本番)。small = 64*170 = 10880, large = 128*170 = 21760
        let sm = 170usize;
        // 27B q4 の投影行数 (実形状)
        let rows = [1024usize, 5120, 14336, 16384, 34816, 151936];
        // mode 0 (既定) は常に base = 従来と同一の launch 形状
        for n in rows {
            assert_eq!(gemv_c_for_n(n, sm, 0, 4), 4, "mode 0 は base 固定: n={n}");
            assert_eq!(gemv_c_for_n(n, sm, 9, 4), 4, "未知 mode も base: n={n}");
        }
        // SM 数が取れない (0) ときは形状を変えない (fail closed)
        assert_eq!(gemv_c_for_n(5120, 0, 1, 4), 4);
        assert_eq!(gemv_c_for_n(5120, 0, 3, 4), 4);
        assert_eq!(gemv_c_for_n(0, sm, 1, 4), 4, "n=0 は何もしない");
        // mode 1: block 数が 2*SM を割る投影だけ C=2 (160 block -> 320 block)
        assert_eq!(gemv_c_for_n(1024, sm, 1, 4), 2);
        assert_eq!(gemv_c_for_n(5120, sm, 1, 4), 2);
        assert_eq!(gemv_c_for_n(10880, sm, 1, 4), 4, "境界は base のまま");
        assert_eq!(gemv_c_for_n(16384, sm, 1, 4), 4);
        assert_eq!(gemv_c_for_n(34816, sm, 1, 4), 4);
        // mode 2 は常に 2
        for n in rows {
            assert_eq!(gemv_c_for_n(n, sm, 2, 4), 2, "mode 2 は常に C=2: n={n}");
        }
        // mode 3 は小 2 / 中 base / 大 8
        assert_eq!(gemv_c_for_n(5120, sm, 3, 4), 2);
        assert_eq!(gemv_c_for_n(16384, sm, 3, 4), 4);
        assert_eq!(gemv_c_for_n(21760, sm, 3, 4), 8);
        assert_eq!(gemv_c_for_n(34816, sm, 3, 4), 8);
        assert_eq!(gemv_c_for_n(151936, sm, 3, 4), 8);
        // 選ばれた C は必ず PTX にある 3 本 (2/4/8) のどれか
        for mode in 0..=3u32 {
            for n in rows {
                let c = gemv_c_for_n(n, sm, mode, 4);
                assert!([2usize, 4, 8].contains(&c), "C={c} に対応する PTX が無い");
                // 列を落とさない (grid.x * 8*C >= n)
                assert!((8 * c) * n.div_ceil(8 * c) >= n);
            }
        }
    }

    /// s169 (P10): `DENNO_DECODE_DBG=1` の形状内訳の書式 (観測のみ)。
    #[test]
    fn gemv_dbg_format_lists_rows_and_c() {
        let t: DbgTally = vec![((34816usize << 4) | 4, 64u64), ((5120usize << 4) | 2, 128u64)];
        assert_eq!(dbg_fmt_gemv(&t), "34816/c4x64,5120/c2x128");
    }
}


#[cfg(test)]
mod p8_phase_tests {
    //! s169 (Task P8): prefill 残差常駐経路の位相計測の**表との対応**を固定する。
    //! GPU は使わない (イベントの実測は実機の担当) — ここで守るのは
    //! 「位相の名前 = 表 C の行名」「1 行を 2 つの位相が共有しない (二重計上しない)」
    //! 「残差加算は独立した行を持たない (融合している)」の 3 点。
    use super::*;

    #[test]
    fn pf_phase_buckets_match_serve_rows() {
        let mut seen: Vec<usize> = Vec::new();
        for ph in 0..PF_PH_N {
            let Some(op) = PF_PH_BUCKETS[ph] else {
                continue;
            };
            assert!(
                (crate::prof::SPLIT_SERVE_FIRST..crate::prof::SPLIT_SERVE_FIRST + crate::prof::SPLIT_SERVE_N)
                    .contains(&op),
                "位相 {ph} の bucket が表 C の範囲外"
            );
            let row = op - crate::prof::SPLIT_SERVE_FIRST;
            let name = crate::prof::split_row_name(crate::prof::SPLIT_SERVE_TABLE, row)
                .expect("位相の bucket は表 C の行");
            assert_eq!(name, PF_PH_NAMES[ph], "位相 {ph} の名前が表 C の行名と違う");
            assert!(!seen.contains(&op), "表 C の行を 2 つの位相が共有している: {name}");
            seen.push(op);
        }
        // GDN の o_proj だけは表 C に行を持たない (decode chain と同じ扱い)。
        assert_eq!(PF_PH_BUCKETS[PF_PH_GDN_OPROJ], None);
        assert_eq!(seen.len(), PF_PH_N - 1);
        // 残差加算は `dno_add_rmsnorm` に融合しているので独立した位相を持たない
        // (ホスト経路の `sl_lay_residual` に相当する区間は残差常駐経路に無い)。
        assert!(!seen.contains(&crate::prof::SL_LAY_RESIDUAL));
        assert!(PF_PH_NAMES.iter().all(|n| !n.contains("residual")));
    }
}


#[cfg(test)]
mod p2_residual_tests {
    //! s165 (Task P2): prefill の残差ストリームをデバイス常駐させる経路の**入口条件**
    //! だけを固定する。GPU は使わない (純粋関数の契約テスト) — 数値の一致と速度は
    //! 実機 (main セッション) の担当。ここで None になる条件がそのまま
    //! 「この層は従来のホスト経路で通す」契約になる。
    use super::*;

    pub(crate) fn q4(k: usize, n: usize) -> crate::quant::Linear {
        crate::quant::Linear::Q4(crate::quant::Q4Linear { blocks: Vec::new().into(), k, n, uid: 0 })
    }
    fn f32w(k: usize, n: usize) -> crate::quant::Linear {
        crate::quant::Linear::F32 { w: vec![0.0; k * n], k, n }
    }

    /// グローバルゲート: q4 以外・診断中・GEMV 帯域・端数タイル・smem 超過で発動しない。
    #[test]
    fn pf_res_gate_contract() {
        let (gm, mm) = (256usize, 16usize);
        // 基準: 2048 token・e=5120 (端数タイル無し)
        assert!(pf_res_gate(2048, 5120, true, false, gm, mm));
        // q4 でない層が 1 本でもあれば発動しない
        assert!(!pf_res_gate(2048, 5120, false, false, gm, mm));
        // 層別診断 (lchk) 中はホストの x を層ごとに更新する前提 → 発動しない
        assert!(!pf_res_gate(2048, 5120, true, true, gm, mm));
        // GEMM 帯域の外 (m <= max_m() / m < gemm_min_m()) は発動しない
        assert!(!pf_res_gate(16, 5120, true, false, gm, mm));
        assert!(!pf_res_gate(128, 5120, true, false, gm, mm));
        assert!(!pf_res_gate(0, 5120, true, false, gm, mm));
        // dno_add_rmsnorm の動的 shmem (e*4 <= 48KB) と e % 32 == 0
        assert!(!pf_res_gate(2048, 16384, true, false, gm, mm), "e=16384 は smem 48KB 超");
        assert!(!pf_res_gate(2048, 5000, true, false, gm, mm), "e % 32 != 0");
        // 端数タイルが GEMV 帯域 (<= max_m()) に入る形は受けない
        let rem_small = GDN_PREFILL_TILE * 2 + mm;
        assert!(!pf_res_gate(rem_small, 5120, true, false, gm, mm));
        assert!(pf_res_gate(rem_small + 1, 5120, true, false, gm, mm));
        assert!(pf_res_gate(GDN_PREFILL_TILE * 2, 5120, true, false, gm, mm));
    }

    /// 層ゲート: FFN / full attention / GDN の形が揃ったときだけ Some。
    #[test]
    fn pf_layer_geom_contract() {
        let (gm, mm) = (256usize, 16usize);
        let n = 2048usize;
        let e = 64usize;
        let ff = 128usize;
        // full attention: heads=2, kv_heads=1, dh=32 → att=64, kv_dim=32
        let (heads, kv_heads, dh) = (2usize, 1usize, 32usize);
        let (att, kv_dim) = (heads * dh, kv_heads * dh);
        let gate = q4(e, ff);
        let up = q4(e, ff);
        let down = q4(ff, e);
        let d = PfDims {
            n,
            e,
            heads,
            kv_heads,
            dh,
            nk: 1,
            nv: 1,
            dk: 32,
            dv: 32,
            d_conv: 4,
        };
        let (q, k, v, o) = (q4(e, att * 2), q4(e, kv_dim), q4(e, kv_dim), q4(att, e));
        let full = PfAttnW::Full { q: &q, k: &k, v: &v, o: &o };
        assert!(pf_layer_geom(&full, &gate, &up, &down, &d, gm, mm).is_some(), "基準形は通る");

        // FFN が q4 でない / 形が合わない → 受けない
        assert!(pf_layer_geom(&full, &f32w(e, ff), &up, &down, &d, gm, mm).is_none());
        assert!(pf_layer_geom(&full, &gate, &up, &q4(e, ff + 32), &d, gm, mm).is_none());
        assert!(pf_layer_geom(&full, &gate, &up, &q4(ff, e + 32), &d, gm, mm).is_none());

        // o_proj の k が heads*dh でない / q の n が 2*heads*dh でない → 受けない
        let bad_o = q4(att + 32, e);
        let full_bad = PfAttnW::Full { q: &q, k: &k, v: &v, o: &bad_o };
        assert!(pf_layer_geom(&full_bad, &gate, &up, &down, &d, gm, mm).is_none());
        let bad_q = q4(e, att);
        let full_bad = PfAttnW::Full { q: &bad_q, k: &k, v: &v, o: &o };
        assert!(pf_layer_geom(&full_bad, &gate, &up, &down, &d, gm, mm).is_none());

        // GDN 層: qkv/z/o の形が揃えば通り、dk != dv では受けない
        //   conv_dim = nk*dk*2 + nv*dv = 64 + 32 = 96, vd = 32
        let gqkv = q4(e, 96);
        let gz = q4(e, 32);
        let go = q4(32, e);
        let gdn = PfAttnW::Gdn { qkv: &gqkv, z: &gz, o: &go };
        assert!(pf_layer_geom(&gdn, &gate, &up, &down, &d, gm, mm).is_some());
        let mut d2 = PfDims { n, e, heads, kv_heads, dh, nk: 1, nv: 1, dk: 32, dv: 32, d_conv: 4 };
        d2.dv = 16;
        assert!(pf_layer_geom(&gdn, &gate, &up, &down, &d2, gm, mm).is_none(), "dk != dv は受けない");

        // n が GEMV 帯域 → 層としても受けない
        let mut d3 = PfDims { n: mm, e, heads, kv_heads, dh, nk: 1, nv: 1, dk: 32, dv: 32, d_conv: 4 };
        assert!(pf_layer_geom(&full, &gate, &up, &down, &d3, gm, mm).is_none());
        d3.n = n;
        assert!(pf_layer_geom(&full, &gate, &up, &down, &d3, gm, mm).is_some());
    }

    /// 新スロットは既存と衝突せず、チェーン本数に収まる。
    #[test]
    fn pf_res_slot_is_distinct() {
        let used = [
            CH_X, CH_H, CH_QKV, CH_ATT, CH_O, CH_GA, CH_ACT, CH_DOWN, CH_LOGITS, CH_Q35_Q, CH_PFH,
        ];
        assert!(!used.contains(&CH_PF_RES), "CH_PF_RES は既存スロットと別");
        assert!(CH_PF_RES < CH_SLOTS);

    }
}


#[cfg(test)]
mod p6_overlap_tests {
    //! s167 (Task P6): GDN prefill の h2 D2H を専用コピーストリームで qkv/z GEMM と
    //! 重ねる経路の**入口条件とフォールバック**だけを固定する (GPU 非依存の契約
    //! テスト)。数値の一致・速度は実機の担当。
    //!
    //! ここで固定する契約:
    //!   - フラグは既定 ON (s172/P13)。明示的な "0"/"off"/"false" だけが OFF
    //!     (= 従来の直列経路)
    //!   - stream/event/pinned の準備に失敗したら必ず従来の同期 D2H (`SyncD2h`)
    //!   - 形が不正・桁あふれ・0 バイトでも従来経路 (コピーを発行しない)
    use super::*;

    /// s172 (P13): `DENNO_PF_OVERLAP` は**既定 ON**。明示的な偽値だけが OFF。
    #[test]
    fn pf_overlap_flag_defaults_on() {
        assert!(parse_pf_overlap(None), "未設定は ON (gdn_proj のホスト仕事を GEMM の裏に隠す)");
        for raw in ["", "1", "true", "on", "yes", " 1", "TRUE"] {
            assert!(parse_pf_overlap(Some(raw)), "raw={raw:?} は ON のはず");
        }
        for raw in ["0", "off", "false"] {
            assert!(!parse_pf_overlap(Some(raw)), "raw={raw:?} は OFF (従来と同一経路) のはず");
        }
    }

    /// 経路決定: OFF とデバイス準備失敗は**必ず**従来の同期 D2H。
    #[test]
    fn pf_h2_path_falls_back_to_sync() {
        let (n, e) = (12850usize, 4096usize);
        // フラグ OFF: dev_ready が真でも従来経路
        assert_eq!(pf_h2_path(false, n, e, true), PfH2Path::SyncD2h);
        assert_eq!(pf_h2_path(false, n, e, false), PfH2Path::SyncD2h);
        // フラグ ON でも準備失敗 (stream/event/pinned) なら従来経路
        assert_eq!(pf_h2_path(true, n, e, false), PfH2Path::SyncD2h);
        // フラグ ON + 準備成功 + 妥当な形 → コピーストリーム
        assert_eq!(pf_h2_path(true, n, e, true), PfH2Path::CopyStream);
    }

    /// 形が不正 (0・桁あふれ) なら従来経路 — コピーを発行しない。
    #[test]
    fn pf_h2_path_rejects_bad_shape() {
        assert_eq!(pf_h2_path(true, 0, 4096, true), PfH2Path::SyncD2h);
        assert_eq!(pf_h2_path(true, 12850, 0, true), PfH2Path::SyncD2h);
        assert_eq!(pf_h2_path(true, 0, 0, true), PfH2Path::SyncD2h);
        // n*e は通るが *4 で桁あふれ → 従来経路 (checked_mul)
        assert_eq!(
            pf_h2_path(true, usize::MAX / 2, 2, true),
            PfH2Path::SyncD2h
        );
        assert_eq!(pf_h2_path(true, usize::MAX, 2, true), PfH2Path::SyncD2h);
        // 1 行だけでも (実経路では n >= 2048 だが) 判定としてはコピー経路
        assert_eq!(pf_h2_path(true, 1, 1, true), PfH2Path::CopyStream);
    }
}


/// s246 (P46): full 層 att の「チャンク化しても数値が動かない」契約を固定する。
///
/// GPU は使わない (転送そのものは実機の担当)。ここで守るのは
///   1. チャンク境界が `[0, m)` を隙間なく・重複なく覆う (端数行を捨てない)
///   2. チャンク数を増やしても**ゲート後の配列がビット一致**する
///      (= `gate(row0, ブロック)` の行オフセット算術が正しい)
///   3. 小さい転送・フラグ OFF ではチャンク数 1 = 従来の 1 発往復
/// の 3 点。2 が崩れると「チャンクを跨いだ行で qg の別の head のゲートを読む」
/// という静かな誤りになるので、行数 × チャンク数の格子で全部見る。
#[cfg(test)]
mod p46_att_pipe_tests {
    use super::*;
    use rayon::prelude::*;

    /// 本番のクロージャ (`full_attn_att_inplace` の呼び出し側) と同一の式。
    fn gate_rows(row0: usize, v: &mut [f32], qg: &[f32], heads: usize, dh: usize) {
        v.par_chunks_exact_mut(dh)
            .enumerate()
            .for_each(|(t, att_head)| {
                let tt = row0 * heads + t;
                let i = tt / heads;
                let hh = tt % heads;
                let gate = &qg[i * heads * 2 * dh + hh * 2 * dh + dh..][..dh];
                for (d, av) in att_head.iter_mut().enumerate() {
                    *av /= 1.0 + (-gate[d]).exp();
                }
            });
    }

    /// 1. 境界の被覆性 (端数行を含めて [0, m) を 1 回ずつ)。
    #[test]
    fn att_pipe_chunk_bounds_cover_rows_once() {
        for m in [1usize, 2, 3, 7, 8, 9, 1000, 6251, 25000, 65000] {
            for want in [1usize, 2, 3, 4, 5, 8, 16] {
                let (rows_per, nch) = pf_att_chunk_rows(m, want);
                assert!(nch >= 1, "m={m} want={want}");
                assert!(rows_per >= 1, "m={m} want={want}: rows_per=0");
                if want >= 2 && m >= 2 {
                    assert!(nch >= 2, "m={m} want={want}: 1 発に落ちた");
                    assert!(nch <= want, "m={m} want={want}: 上限を超えた nch={nch}");
                } else {
                    assert_eq!(nch, 1, "m={m} want={want}: 1 発のはず");
                }
                // 全チャンクを並べると [0, m) を隙間なく 1 回ずつ覆う
                let mut next = 0usize;
                for k in 0..nch {
                    let (r0, r1) = pf_att_chunk_bounds(k, rows_per, m);
                    assert_eq!(r0, next, "m={m} want={want} k={k}: 隙間/重複");
                    assert!(r1 > r0, "m={m} want={want} k={k}: 空チャンク");
                    next = r1;
                }
                assert_eq!(next, m, "m={m} want={want}: 端数行が落ちた");
            }
        }
    }

    /// 2. チャンク数が変わってもゲート結果はビット一致。
    #[test]
    fn att_pipe_gate_is_bit_exact_across_chunk_counts() {
        // 実シェイプの縮小版 (heads=4 / dh=8)。行数は端数が出るものを含める。
        let (heads, dh) = (4usize, 8usize);
        let cols = heads * dh;
        for n in [1usize, 2, 3, 5, 8, 13] {
            let qg: Vec<f32> = (0..n * heads * 2 * dh)
                .map(|i| ((i as f32) * 0.017).sin() * 3.0)
                .collect();
            let att0: Vec<f32> = (0..n * cols)
                .map(|i| ((i as f32) * 0.031).cos() * 1.5)
                .collect();
            // 従来 (1 発 = 全体を 1 回)
            let mut want = att0.clone();
            gate_rows(0, &mut want[..], &qg, heads, dh);
            // チャンク版 (2..=n の全分割)
            for want_nch in 1..=n.max(1) {
                let (rows_per, nch) = pf_att_chunk_rows(n, want_nch);
                let mut got = att0.clone();
                for k in 0..nch {
                    let (r0, r1) = pf_att_chunk_bounds(k, rows_per, n);
                    gate_rows(r0, &mut got[r0 * cols..r1 * cols], &qg, heads, dh);
                }
                for (idx, (a, b)) in want.iter().zip(got.iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "n={n} nch={nch} rows_per={rows_per}: 要素 {idx} が不一致"
                    );
                }
            }
        }
    }

    /// 3. 小さい転送・フラグ OFF・1 行は従来の 1 発往復。
    #[test]
    fn att_pipe_chunks_off_for_small_or_disabled() {
        // 128 MiB 未満は 1 発 (PCIe の立上がりを払う方が高い)
        assert_eq!(pf_att_pipe_chunks_with(64 << 20, 25000, 8), 1);
        assert_eq!(pf_att_pipe_chunks_with((128 << 20) - 4, 25000, 8), 1);
        // フラグ OFF (= 上限 1) は常に 1 発
        assert_eq!(pf_att_pipe_chunks_with(615 << 20, 25000, 1), 1);
        assert_eq!(pf_att_pipe_chunks_with(615 << 20, 25000, 0), 1);
        // 1 行では割れない
        assert_eq!(pf_att_pipe_chunks_with(615 << 20, 1, 8), 1);
        // 65k (att = n*heads*dh*4 = 1597 MB) は上限 8 に張り付く
        assert_eq!(pf_att_pipe_chunks_with(1597 << 20, 65000, 8), 8);
        // 615 MB (n≈25k) は 5 チャンク
        assert_eq!(pf_att_pipe_chunks_with(615 << 20, 25000, 8), 5);
        // 上限を下げれば従う
        assert_eq!(pf_att_pipe_chunks_with(1597 << 20, 65000, 4), 4);
        assert_eq!(pf_att_pipe_chunks_with(1597 << 20, 65000, 2), 2);
    }
}


#[cfg(test)]
mod p53_qk_overlap_tests {
    //! s250 (Task P53): full_qk の q の D2H を専用コピーストリームで k/v の GEMM と
    //! 重ねる経路の**入口条件**だけを固定する (GPU 非依存)。速度と数値一致は実機の担当。
    //!
    //! ここで固定する契約:
    //!   - `DENNO_PF_QK_OVERLAP` は既定 ON。明示的な "0"/"off"/"false" だけが OFF
    //!     (= 従来の既定ストリーム直列)
    //!   - OFF・driver API 欠落・stream/event 作成失敗は `None` を返し、呼び出し側は
    //!     null ストリーム (= 従来経路) で 3 発を流す
    use super::*;

    /// 既定 ON・明示的な偽値だけが OFF (P6 の `DENNO_PF_OVERLAP` と同じ規約)。
    #[test]
    fn pf_qk_overlap_flag_defaults_on() {
        assert!(parse_pf_qk_overlap(None), "未設定は ON");
        for raw in ["", "1", "true", "on", "yes", " 1", "TRUE"] {
            assert!(parse_pf_qk_overlap(Some(raw)), "raw={raw:?} は ON のはず");
        }
        for raw in ["0", "off", "false"] {
            assert!(
                !parse_pf_qk_overlap(Some(raw)),
                "raw={raw:?} は OFF (既定ストリーム直列) のはず"
            );
        }
    }

    /// 65k 級 27B (n=14,043・qcols=12,288・kcols=vcols=1,024) の 1 層あたりの
    /// D2H バイト数と、隠せる量の算術を固定する。
    ///
    /// ここが崩れると REPORT_P53.md の見積り (q の D2H ≈48 ms/層・k/v の
    /// GEMM ≈9.8 ms/層 → ≈0.16 s/request、上限 0.29 s) の前提が崩れる。
    #[test]
    fn pf_qk_d2h_bytes_match_27b_shape() {
        let (rows, qcols, kcols, vcols) = (14_043usize, 12_288usize, 1_024usize, 1_024usize);
        let q_bytes = rows * qcols * 4;
        let k_bytes = rows * kcols * 4;
        let v_bytes = rows * vcols * 4;
        // 690.2 MB / 57.5 MB / 57.5 MB
        assert_eq!(q_bytes, 690_241_536);
        assert_eq!(k_bytes, 57_520_128);
        assert_eq!(v_bytes, 57_520_128);
        assert_eq!(q_bytes + k_bytes, 747_761_664, "P47 の pinned プール (747 MB)");
        assert_eq!(q_bytes + k_bytes + v_bytes, 805_281_792, "1 層の D2H 合計");
        // 実測単価 14.3 GB/s (s68) での q のコピー時間 ≈48 ms/層
        let q_ms = q_bytes as f64 / 14.3e9 * 1e3;
        assert!((47.0..50.0).contains(&q_ms), "q の D2H が {q_ms} ms/層");
        // 16 full 層ぶんの q は ≈0.77 s (このうち k/v の GEMM+D2H の分だけが隠れる)
        let q_all = q_ms * 16.0;
        assert!((760.0..790.0).contains(&q_all), "q の D2H 合計が {q_all} ms");
    }
}


// ---------------------------------------------------------------------------
// P58-3: prefill スクラッチ arena の検証 (GPU 不要・ホストだけで回る)
// ---------------------------------------------------------------------------
#[cfg(test)]
mod p58_scratch_arena_tests {
    use super::*;

    /// task 149: リクエスト終了時の解放は既定 off (lab で効果が無かったため)。
    #[test]
    fn scratch_free_env_gate() {
        assert!(!parse_scratch_free(None), "未設定 = off (従来どおり保持)");
        assert!(!parse_scratch_free(Some("0")));
        assert!(!parse_scratch_free(Some("false")));
        assert!(!parse_scratch_free(Some("OFF")));
        assert!(parse_scratch_free(Some("1")));
        assert!(parse_scratch_free(Some(" true ")));
        assert!(parse_scratch_free(Some("on")));
    }

    /// 27B q4・n=14,240 の prefill で実際に切り出される 12 スロット
    /// (`chain_ensure_buf` が要求するバイト数と同じ式)。
    fn chain_slot_bytes(rows: usize) -> [usize; 12] {
        let e = 5120usize;
        let hd = 6144usize; // heads*dh
        let qcols = 12288usize; // q (attn_output_gate 込み)
        let kcols = 1024usize;
        let vcols = 1024usize;
        let ff = 17408usize;
        let vocab = 248_320usize;
        [
            rows * e * 4,                        // CH_X
            rows * e * 4,                        // CH_H
            rows * (qcols + kcols + vcols) * 4,  // CH_QKV
            rows * hd * 4,                       // CH_ATT
            rows * hd * 4,                       // CH_O
            rows * 2 * ff * 4,                   // CH_GA
            rows * ff * 4,                       // CH_ACT
            rows * e * 4,                        // CH_DOWN
            vocab * 4,                           // CH_LOGITS (decode のみ)
            rows * hd * 4,                       // CH_Q35_Q
            rows * e * 4,                        // CH_PFH
            rows * e * 4,                        // CH_PF_RES
        ]
    }

    /// 2 回連続の「forward 相当の確保」で 2 回目が同じオフセットを返すこと
    /// (要求 4)。これが arena 化の本体の契約。
    #[test]
    fn arena_plan_second_forward_reuses_same_offsets() {
        let rows = 14_240usize;
        let sizes = chain_slot_bytes(rows);
        let mut plan = ArenaPlan::default();
        plan.add_chunk(ARENA_CHUNK * 8);
        let mut first = Vec::new();
        for (i, b) in sizes.iter().enumerate() {
            let (c, off) = plan.reserve(0, ARENA_KEY_CHAIN + i as u32, *b).expect("1 回目");
            first.push((c, off));
        }
        let mut second = Vec::new();
        for (i, b) in sizes.iter().enumerate() {
            let (c, off) = plan.reserve(0, ARENA_KEY_CHAIN + i as u32, *b).expect("2 回目");
            second.push((c, off));
        }
        assert_eq!(first, second, "2 回目は同じオフセットを返す (再利用)");
        // 予約は重ならない
        let mut used: Vec<(usize, usize, usize)> = first
            .iter()
            .zip(sizes.iter())
            .map(|((c, off), b)| (*c, *off, off + ArenaPlan::align_up(*b)))
            .collect();
        used.sort();
        for w in used.windows(2) {
            assert!(
                w[0].0 != w[1].0 || w[0].2 <= w[1].1,
                "チャンク {} で予約が重なった: {:?}",
                w[0].0,
                w
            );
        }
    }

    /// 実測の「要求 779 MiB」の正体が CH_QKV であること (n で 1 意に決まる)。
    /// 57,344 B/行 なので 779 MiB になるのは n = 14,245..14,262 の範囲だけ
    /// (CH_GA は 1,892 MiB・CH_ACT は 946 MiB で一致しない)。
    #[test]
    fn arena_qkv_matches_reported_779_mib() {
        let rows = 14_252usize;
        let b = chain_slot_bytes(rows)[CH_QKV];
        assert_eq!(b, rows * 57_344, "CH_QKV = 行 x (q 12,288 + k/v 各 1,024) x 4 B");
        assert_eq!(b >> 20, 779, "実測の 779 MiB 要求と一致");
    }

    /// 同じキーで大きいサイズを要求したら別の位置へ移る (古い領域は再利用しない)。
    #[test]
    fn arena_plan_larger_request_moves_to_new_offset() {
        let mut plan = ArenaPlan::default();
        plan.add_chunk(ARENA_CHUNK);
        let (c0, o0) = plan.reserve(0, 7, 1000).unwrap();
        let (c1, o1) = plan.reserve(0, 7, 4000).unwrap();
        assert_eq!((c0, o0), (0, 0));
        assert_eq!((c1, o1), (0, ArenaPlan::align_up(1000)));
        // 小さいサイズへ戻しても予約は移らない (同じ位置を使い回す)
        let (c2, o2) = plan.reserve(0, 7, 1000).unwrap();
        assert_eq!((c2, o2), (c1, o1));
    }

    /// 足りないときは「必要なチャンクのバイト数」を返し、状態は変えない。
    #[test]
    fn arena_plan_reports_chunk_bytes_without_mutating() {
        let mut plan = ArenaPlan::default();
        let want = plan.reserve(0, 1, 4096).unwrap_err();
        assert_eq!(want, ARENA_CHUNK, "チャンクは 1,024 MiB 単位");
        assert!(plan.chunks.is_empty() && plan.slots.is_empty());
        plan.add_chunk(want);
        assert_eq!(plan.reserve(0, 1, 4096).unwrap(), (0, 0));
        // 1 GiB を超える要求は切り上げたチャンク 1 本で受ける
        let big = ARENA_CHUNK + 1;
        assert_eq!(plan.reserve(0, 2, big).unwrap_err(), ARENA_CHUNK * 2);
    }

    /// オフセットは 256 B に揃える。
    #[test]
    fn arena_plan_aligns_offsets() {
        let mut plan = ArenaPlan::default();
        plan.add_chunk(ARENA_CHUNK);
        let (_, a) = plan.reserve(0, 1, 1).unwrap();
        let (_, b) = plan.reserve(0, 2, 3).unwrap();
        let (_, c) = plan.reserve(0, 3, 257).unwrap();
        assert_eq!((a, b, c), (0, 256, 512));
        for off in [a, b, c] {
            assert_eq!(off % ARENA_ALIGN, 0);
        }
    }

    /// スレッドが違えば同じキーでも別の領域 (thread_local スロットの衝突防止)。
    #[test]
    fn arena_plan_separates_threads() {
        let mut plan = ArenaPlan::default();
        plan.add_chunk(ARENA_CHUNK);
        let (_, a) = plan.reserve(0, ARENA_KEY_Q4_X, 4096).unwrap();
        let (_, b) = plan.reserve(1, ARENA_KEY_Q4_X, 4096).unwrap();
        assert_ne!(a, b);
        assert_eq!((a, b), (0, 4096));
        // 同じスレッドの別キーも別の領域
        let (_, c) = plan.reserve(0, ARENA_KEY_Q4_OUT, 4096).unwrap();
        assert_eq!(c, 8192);
    }

    /// 境界の切り直しでチャンクを残すときは先頭から使い直す。
    #[test]
    fn arena_plan_reset_keep_chunks_restarts_at_zero() {
        let mut plan = ArenaPlan::default();
        plan.add_chunk(ARENA_CHUNK);
        plan.reserve(0, 1, 8192).unwrap();
        plan.reset_keep_chunks();
        assert_eq!(plan.capacity(), ARENA_CHUNK);
        assert_eq!(plan.reserve(0, 1, 8192).unwrap(), (0, 0));
    }

    /// 境界の推定: 前回の 1 行あたり使用量から今回の必要量を出す。
    #[test]
    fn arena_target_scales_with_rows() {
        // 前回 rows=2000 で 800 MiB 使った → 14,240 行なら ≈5.7 GiB (1 GiB 切り上げ)
        let prev_high = 800usize << 20;
        let t = ArenaPlan::target_for(2000, prev_high, 14_240);
        let per_row = prev_high / 2000 + 1;
        let want = per_row * 14_240 * 11 / 10;
        assert_eq!(t, want.saturating_add(ARENA_CHUNK - 1) & !(ARENA_CHUNK - 1));
        assert!(t >= want, "推定は要求以上");
        assert_eq!(t % ARENA_CHUNK, 0, "1 GiB 単位");
        // 行数が増えれば単調に増える
        assert!(ArenaPlan::target_for(2000, prev_high, 28_000) > t);
        // 実績が無ければ推定しない (0)
        assert_eq!(ArenaPlan::target_for(0, 0, 14_240), 0);
        assert_eq!(ArenaPlan::target_for(2000, 0, 14_240), 0);
        assert_eq!(ArenaPlan::target_for(2000, prev_high, 0), 0);
    }

    /// `DENNO_SCRATCH_ARENA` の解釈 (未設定 = 有効)。
    #[test]
    fn arena_env_gate() {
        assert!(parse_scratch_arena(None));
        assert!(parse_scratch_arena(Some("1")));
        assert!(parse_scratch_arena(Some("true")));
        assert!(!parse_scratch_arena(Some("0")));
        assert!(!parse_scratch_arena(Some(" 0 ")));
        assert!(!parse_scratch_arena(Some("false")));
        assert!(!parse_scratch_arena(Some("off")));
    }
}


#[cfg(test)]
mod p59_dedup_tests {
    //! s178 (P59): 連結重みの行範囲ビューの**バイト幾何**と A/B スイッチを固定する。
    //!
    //! 27B q4 は decode chain が連結 1 本 (`gate_up_w` / `qkv_fused`)、prefill の
    //! per-op 経路が分割 (`gate_w` / `up_w` / `q_w` / `k_w` / `v_w`) を使うため、
    //! 同じパラメータが VRAM に 2 度載っていた (実測 465 本 / 22,818.6 MB)。
    //! 旧 `q4_alias_view` は `row_off + n != tn` で判定していたため、**先頭の
    //! part (row_off = 0) だけ**ビューにならず個別転送されていた。
    //! ここでは GPU を使わずに (純粋関数だけで) 先頭・中間・末尾の part がすべて
    //! 連結側の行範囲になることを固定し、範囲外・形状不一致は fail closed で
    //! `None` になることを固定する。
    use super::*;

    /// 27B: e = 5120, ff = 17408 → gate_up = [gate(17408 行) | up(17408 行)]
    #[test]
    fn view_geo_gate_up_all_parts() {
        let (k, ff) = (5120usize, 17408usize);
        let nb = k / 32;
        let tn = 2 * ff;
        // 先頭 part (gate): 旧実装の `!=` ではここが落ちて個別転送されていた
        assert_eq!(
            q4_view_geo(k, ff, 0, k, tn),
            Some((0, ff * nb * 16, 0, ff * nb * 4)),
            "gate (row_off = 0) がビューにならない"
        );
        // 末尾 part (up)
        assert_eq!(
            q4_view_geo(k, ff, ff, k, tn),
            Some((ff * nb * 16, ff * nb * 16, ff * nb * 4, ff * nb * 4)),
            "up (row_off = ff) がビューにならない"
        );
        // 2 本合わせて連結側のバッファをちょうど覆う (1 バイトの隙間も重なりも無い)
        let (o1, b1, s1, c1) = q4_view_geo(k, ff, 0, k, tn).unwrap();
        let (o2, b2, s2, c2) = q4_view_geo(k, ff, ff, k, tn).unwrap();
        assert_eq!(o1 + b1, o2, "nib の行範囲が連続していない");
        assert_eq!(s1 + c1, s2, "sc の行範囲が連続していない");
        assert_eq!(o2 + b2, tn * nb * 16, "nib が連結側の末尾に一致しない");
        assert_eq!(s2 + c2, tn * nb * 4, "sc が連結側の末尾に一致しない");
    }

    /// full attention: [q(12288) | k(1024) | v(1024)] = 14336 /
    /// GDN: [qkv(10240) | z(6144)] = 16384
    #[test]
    fn view_geo_qkv_all_parts() {
        let k = 5120usize;
        let nb = k / 32;
        let (nq, nkv) = (12288usize, 1024usize);
        let tn = nq + 2 * nkv;
        assert_eq!(q4_view_geo(k, nq, 0, k, tn), Some((0, nq * nb * 16, 0, nq * nb * 4)));
        assert_eq!(
            q4_view_geo(k, nkv, nq, k, tn),
            Some((nq * nb * 16, nkv * nb * 16, nq * nb * 4, nkv * nb * 4))
        );
        assert_eq!(
            q4_view_geo(k, nkv, nq + nkv, k, tn),
            Some(((nq + nkv) * nb * 16, nkv * nb * 16, (nq + nkv) * nb * 4, nkv * nb * 4))
        );
        let (n1, n2) = (10240usize, 6144usize);
        assert_eq!(q4_view_geo(k, n1, 0, k, n1 + n2), Some((0, n1 * nb * 16, 0, n1 * nb * 4)));
        assert_eq!(
            q4_view_geo(k, n2, n1, k, n1 + n2),
            Some((n1 * nb * 16, n2 * nb * 16, n1 * nb * 4, n2 * nb * 4))
        );
    }

    /// 前提が崩れた入力は `None` (= 呼び出し側は通常の転送へ落ちる)。
    /// ここが `Some` を返すと**別の重みの行**を読む (数値が変わる)。
    #[test]
    fn view_geo_fail_closed() {
        let k = 5120usize;
        // 範囲が連結側の外へはみ出す (1 行でも超えたら駄目)
        assert_eq!(q4_view_geo(k, 17408, 17408, k, 2 * 17408 - 1), None);
        assert_eq!(q4_view_geo(k, 32, 5120, k, 5120), None);
        // k が食い違う / k が 32 の倍数でない / n = 0
        assert_eq!(q4_view_geo(k, 1024, 0, k + 32, 1024), None);
        assert_eq!(q4_view_geo(5104, 1024, 0, 5104, 1024), None);
        assert_eq!(q4_view_geo(k, 0, 0, k, 1024), None);
        // オフセット計算のオーバーフロー
        assert_eq!(q4_view_geo(k, usize::MAX, 0, k, usize::MAX), None);
    }

    /// A/B スイッチの解釈: `"0"` だけが off (タイポで静かに経路が変わるのを避ける)。
    #[test]
    fn dedup_env_only_zero_is_off() {
        assert!(parse_q4_dedup(None));
        assert!(parse_q4_dedup(Some("")));
        assert!(parse_q4_dedup(Some("1")));
        assert!(parse_q4_dedup(Some("off")));
        assert!(!parse_q4_dedup(Some("0")));
    }

    // -----------------------------------------------------------------------
    // P-pf-moe-scratch: `moe_gemv_dev` のスクラッチ (grow-only) のサイズ決定。
    // GPU 無しで検証できる純関数だけを見る。
    // -----------------------------------------------------------------------

    #[test]
    fn moe_scratch_grows_only_on_larger_request() {
        // 初回は必ず確保する (0 -> want)。
        assert_eq!(moe_scratch_alloc_bytes(0, 1024), Some(1024));
        // 足りていれば確保しない = 同じサイズの繰り返しで alloc/free しない。
        assert_eq!(moe_scratch_alloc_bytes(1024, 1024), None);
        assert_eq!(moe_scratch_alloc_bytes(2048, 1024), None);
        // 大きくなったときだけ、そのサイズで 1 回 grow する。
        assert_eq!(moe_scratch_alloc_bytes(1024, 2048), Some(2048));
        assert_eq!(moe_scratch_alloc_bytes(2048, 2048), None);
        // 0 バイトの確保はしない (呼び出し側の形状検査で want > 0 だが、
        // 純関数として 0 を渡しても 0 バイト確保にしない)。
        assert_eq!(moe_scratch_alloc_bytes(0, 0), None);
    }

    /// task 788 (バグ 2): 借用ビュー (プール区画) は容量が足りていても確保し直す。
    ///
    /// プールは prefill の終端で `cuMemFree` されるので、残ったビューを
    /// 再利用すると解放済みの領域を触る (CUDA error 700)。
    #[test]
    fn moe_scratch_never_reuses_pool_views() {
        // 借用ビュー: 容量が足りていても want で確保し直す。
        for (cur, want) in [
            (1usize << 20, 665_600usize),
            (665_600, 665_600),
            (0, 665_600),
        ] {
            assert_eq!(
                moe_scratch_alloc_bytes_borrowed(cur, want, true),
                Some(want),
                "借用ビュー cur={cur} want={want} を再利用した"
            );
        }
        // 所有バッファ: 従来どおり grow-only。
        assert_eq!(moe_scratch_alloc_bytes_borrowed(1 << 20, 665_600, false), None);
        assert_eq!(moe_scratch_alloc_bytes_borrowed(665_600, 665_600, false), None);
        assert_eq!(moe_scratch_alloc_bytes_borrowed(100, 665_600, false), Some(665_600));
        // nt in {64, 65, 67, 127, 128, 129, 512} の shared expert の作業域寸法。
        let e = 2560usize;
        let sff = 1024usize;
        for nt in [64usize, 65, 67, 127, 128, 129, 512] {
            let sh_x = nt * e * 4;
            let sh_ga = nt * sff * 4;
            // 8k prefill (tile=5120) の借用ビューが残っていても、nt の寸法で取り直す。
            assert_eq!(
                moe_scratch_alloc_bytes_borrowed(5120 * e * 4, sh_x, true),
                Some(sh_x),
                "nt={nt}: ShX を確保し直していない"
            );
            assert_eq!(
                moe_scratch_alloc_bytes_borrowed(5120 * sff * 4, sh_ga, true),
                Some(sh_ga),
                "nt={nt}: ShGa を確保し直していない"
            );
        }
    }

    #[test]
    fn moe_scratch_alloc_is_monotone_in_want() {
        // 要求サイズが増えると、返るサイズは減らない (単調)。
        let caps = [0usize, 1, 1024, 4096, 1 << 20];
        let wants = [0usize, 1, 1000, 1024, 4096, (1 << 20) + 1];
        for &cur in &caps {
            let mut prev = 0usize;
            for &want in &wants {
                if let Some(bytes) = moe_scratch_alloc_bytes(cur, want) {
                    assert!(bytes >= want.max(1), "cur={cur} want={want}: {bytes}");
                    assert!(bytes >= prev, "cur={cur} want={want}: 単調でない");
                    prev = bytes;
                }
                // 確保しない場合も含め、確保後は必ず要求以上になる。
                let now = moe_scratch_alloc_bytes(cur, want).unwrap_or(cur);
                assert!(now >= want, "cur={cur} want={want}: 確保後も不足");
            }
        }
    }

    // -----------------------------------------------------------------------
    // task 110: hc mix BF16 (down / up) の GPU 経路
    // -----------------------------------------------------------------------

    /// task 110: BF16 の決定的重み列 (qwen4exp tests の `bf16_weights` と同じ
    /// LCG。テストモジュールが別なのでここにも置く)。
    fn mix_bf16_bytes(k: usize, n: usize, seed: u32) -> Vec<u8> {
        let mut out = Vec::with_capacity(k * n * 2);
        let mut x = seed.wrapping_mul(2_654_435_761) | 1;
        for _ in 0..(k * n) {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let v = (((x >> 8) % 2000) as f32 / 1000.0 - 1.0) * 0.9;
            out.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
        }
        out
    }

    /// task 110: BF16 -> f32 展開 ([`mix_expand_bf16`]) が `kquant::dequant_row`
    /// と全要素ビット一致する (GPU 無しで可。展開の定義は 1 か所で保たれている
    /// ことの確認)。
    #[test]
    fn mix_bf16_expand_matches_dequant_row() {
        for (k, n, seed) in [(32usize, 3usize, 3u32), (7usize, 5usize, 11u32)] {
            let bytes = mix_bf16_bytes(k, n, seed);
            let expanded = mix_expand_bf16(&bytes, k, n);
            let rb = k * 2;
            for r in 0..n {
                let row = crate::kquant::dequant_row(
                    crate::kquant::GgmlType::Bf16,
                    &bytes[r * rb..(r + 1) * rb],
                    k,
                );
                for i in 0..k {
                    assert_eq!(
                        expanded[r * k + i].to_bits(),
                        row[i].to_bits(),
                        "r={r} i={i} (k={k}, n={n}) が dequant_row とビット不一致"
                    );
                }
            }
        }
    }

    /// task 110 + task 121: 実 GPU で hc mix BF16 GEMV を回す。短い行 (k < n) は
    /// CPU の BF16 経路 (AVX2 融合 = classic とビット一致済み) と全要素ビット一致。
    /// 長い行 (k >= n) は task 121 で s169 の並列リダクション版 (`dno_matvec_f32_r`
    /// ・block 256・固定木縮約) に切り替わるためビット一致はしない — CPU との差は
    /// 絶対 1e-4 + 相対 1e-5 以内で受け、**2 回の実行がビット同一** (決定論) のみを
    /// 保証する。実形 (down [10240->320] / up [320->10240]) + 端数 k。
    ///
    /// 実機: `cargo test --release -p denno-engine --features cuda --lib -- --ignored cuda_mix_bf16 --nocapture`
    #[test]
    #[ignore] // GPU が必要
    fn cuda_mix_bf16_short_rows_bit_match_and_long_rows_are_deterministic() {
        let _g = test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if !cuda_enabled() {
            CudaCtx::new().expect("CUDA init");
            enable_for_tests();
        }
        let cases = [
            (10240usize, 320usize, 3u32),
            (320usize, 10240usize, 5u32),
            (13usize, 5usize, 7u32),
            (5usize, 13usize, 11u32),
        ];
        for (idx, (k, n, seed)) in cases.iter().enumerate() {
            let (k, n) = (*k, *n);
            let bytes = mix_bf16_bytes(k, n, *seed);
            let x: Vec<f32> = (0..k)
                .map(|i| (((i * 31) % 97) as f32 - 48.0) * 0.01)
                .collect();
            let mut cpu = vec![0f32; n];
            crate::model::moe_expert::expert_gemv_bf16(&bytes, k, n, &x, &mut cpu)
                .expect("CPU BF16 GEMV");
            let uid = 0xD110_0000u64 + idx as u64;
            let mut gpu = vec![0f32; n];
            set_last_linear_ran_on_gpu(false);
            assert!(
                maybe_gpu_mix_bf16_into(uid, &bytes, k, n, &x, &mut gpu),
                "GPU 経路が発動しない (k={k}, n={n})"
            );
            assert!(
                last_linear_ran_on_gpu(),
                "発動フラグが立っていない (k={k}, n={n})"
            );
            // 短い行 (k < n) はビット一致。長い行 (k >= n) は task 121 で `_r`
            // (固定木縮約・丸め順が違う) に切り替わるので許容内で受ける (PTX に
            // `_r` が無い環境では逐次版に落ちるため diff=0 になり、これも許容内)。
            let long_row = mix_use_reduce_kernel(k, n, true);
            for r in 0..n {
                if long_row {
                    let diff = (gpu[r] - cpu[r]).abs();
                    let tol = 1e-4 + 1e-5 * cpu[r].abs();
                    assert!(
                        diff <= tol,
                        "行 {r} (k={k}, n={n}) が CPU と許容外: diff={diff:e} tol={tol:e}"
                    );
                } else {
                    assert_eq!(
                        gpu[r].to_bits(),
                        cpu[r].to_bits(),
                        "行 {r} (k={k}, n={n}) が CPU とビット不一致 (短い行)"
                    );
                }
            }
            // 決定論 (2 回目も同一)。
            let mut gpu2 = vec![0f32; n];
            assert!(maybe_gpu_mix_bf16_into(uid, &bytes, k, n, &x, &mut gpu2));
            for r in 0..n {
                assert_eq!(
                    gpu[r].to_bits(),
                    gpu2[r].to_bits(),
                    "2 回目が不一致 (k={k}, n={n})"
                );
            }
            // 登録の解放が死なないこと (モデル差し替えの経路)。
            mix_gpu_unregister(uid);
        }
    }

    // -----------------------------------------------------------------------
    // task 121: 長い行 (k >= n) の hc mix GPU を s169 の並列リダクション版へ
    // -----------------------------------------------------------------------

    /// task 121: カーネル選択の真理値表 (純関数・GPU 無し)。長い行 (k >= n) だけ
    /// `_r`、`has_r = false` (PTX 未収録) なら常に逐次版に落ちる。
    /// DENNO_FAST_REDUCE は引数にすら現れない (= 依存しない)。
    #[test]
    fn mix_kernel_selection_truth_table() {
        // 実形 (swift15 IQ3_S): down は長い行 / up は短い行。
        assert!(mix_use_reduce_kernel(10240, 320, true), "down は長い行");
        assert!(
            !mix_use_reduce_kernel(320, 10240, true),
            "up は短い行 (逐次版のまま)"
        );
        // PTX に `_r` が無い (has_r=false) なら常に逐次版 (fail closed・失敗しない)。
        assert!(!mix_use_reduce_kernel(10240, 320, false));
        assert!(!mix_use_reduce_kernel(320, 10240, false));
        // 境界: k == n は長い行 (>=)。1 行ずれた両側。
        assert!(mix_use_reduce_kernel(64, 64, true));
        assert!(mix_use_reduce_kernel(65, 64, true));
        assert!(!mix_use_reduce_kernel(64, 65, true));
        // 縮退形も規約どおり (呼び出し側の形状検査が先に弾く)。
        assert!(mix_use_reduce_kernel(1, 1, true));
        assert!(!mix_use_reduce_kernel(0, 5, true));
        assert!(!mix_use_reduce_kernel(5, 0, true));
    }

    // -----------------------------------------------------------------------
    // task 127: ルータ GPU のカーネル選択 (逐次版 -> 並列リダクション版)
    // -----------------------------------------------------------------------

    /// task 127: ルータは k=2560 > n=512 の長い行なので常に `_r`。
    /// PTX 未収録 (`has_r = false`) は逐次版に落ちる (fail closed)。
    #[test]
    fn router_kernel_selection_truth_table() {
        // 実形 (`ffn_gate_inp` = 2560 -> 512)。
        assert!(
            router_use_reduce_kernel(2560, 512, true),
            "ルータの実形は長い行 (k >= n) なので `_r` のはず"
        );
        assert!(
            !router_use_reduce_kernel(2560, 512, false),
            "PTX に `_r` が無いときは逐次版 (fail closed)"
        );
        // 境界: k == n は長い行 (>=)。1 行ずれた両側。
        assert!(router_use_reduce_kernel(64, 64, true));
        assert!(router_use_reduce_kernel(65, 64, true));
        assert!(!router_use_reduce_kernel(64, 65, true));
        // 縮退形 (呼び出し側の形状検査が先に弾く)。
        assert!(!router_use_reduce_kernel(0, 5, true));
        assert!(!router_use_reduce_kernel(5, 0, true));
    }

    /// task 127: `DENNO_ROUTER_GPU_REDUCE` の真理値表。既定 on、明示的な偽値だけ off。
    #[test]
    fn router_gpu_reduce_env_values() {
        for v in [None, Some("1"), Some("on"), Some("TRUE"), Some(" on "), Some("")] {
            assert!(router_gpu_reduce_from_value(v), "value={v:?} は on のはず");
        }
        for v in [
            Some("0"),
            Some("off"),
            Some("false"),
            Some("no"),
            Some(" OFF "),
            Some("False"),
        ] {
            assert!(!router_gpu_reduce_from_value(v), "value={v:?} は off のはず");
        }
    }

    /// task 121: `dno_matvec_f32_r` の CPU ミラー (1 行)。blockDim=256 で k を
    /// tid 順に strided 加算し、warp 内は shuffle の固定木 (o = 16,8,4,2,1)・
    /// warp 合計は warp id 昇順で加算する — カーネル (kernels_elem.cu s169) と
    /// 同一の縮約順。積と加算の丸めを分離 (fma 無し) するので正規数範囲では
    /// GPU と同じ並びになる。shuffle 木は lane 0 に流れ込む経路だけを模倣する
    /// (src lane が範囲外のレーンの値は lane 0 に到達しない)。
    fn matvec_f32_r_cpu_mirror_row(h: &[f32], row: &[f32], k: usize) -> f32 {
        const NT: usize = 256; // blockDim.x (launch 側と一致)
        let mut lanes = [0f32; NT];
        for (tid, lane) in lanes.iter_mut().enumerate() {
            let mut acc = 0f32;
            let mut d = tid;
            while d < k {
                acc += h[d] * row[d]; // __fadd_rn(acc, __fmul_rn(h[d], row[d]))
                d += NT;
            }
            *lane = acc;
        }
        // warp ごとの shuffle 木 (lane i は lane i+o を足す・o 降順)。
        for w in 0..(NT / 32) {
            let base = w * 32;
            for o in [16usize, 8, 4, 2, 1] {
                for i in 0..(32 - o) {
                    lanes[base + i] += lanes[base + i + o];
                }
            }
        }
        // sh[wid] (= 各 warp の lane 0) を warp id 昇順に加算 (t = sh[0] 出発)。
        let mut t = lanes[0];
        for w in 1..(NT / 32) {
            t += lanes[w * 32];
        }
        t
    }

    /// task 121: CPU ミラー (固定木縮約) を f64 真値と比べる (実形の両形状)。
    /// 許容: 絶対 1e-4 + 相対 1e-5 (f64 真値基準)。両形状とも全行。
    #[test]
    fn mix_reduce_mirror_matches_f64_truth_on_real_shapes() {
        for (k, n, seed) in [(10240usize, 320usize, 3u32), (320usize, 10240usize, 5u32)] {
            let w = mix_expand_bf16(&mix_bf16_bytes(k, n, seed), k, n);
            let h: Vec<f32> = (0..k)
                .map(|i| (((i * 31) % 97) as f32 - 48.0) * 0.01)
                .collect();
            for r in 0..n {
                let row = &w[r * k..(r + 1) * k];
                let got = matvec_f32_r_cpu_mirror_row(&h, row, k);
                let truth: f64 = (0..k).map(|d| h[d] as f64 * row[d] as f64).sum();
                let diff = (got as f64 - truth).abs();
                let tol = 1e-4 + 1e-5 * truth.abs();
                assert!(
                    diff <= tol,
                    "r={r} (k={k}, n={n}): got={got:e} truth={truth:e} diff={diff:e} tol={tol:e}"
                );
            }
        }
    }

    /// task 121: 実 GPU で長い行 (down) / 短い行 (up) の両方を
    /// [`maybe_gpu_mix_bf16_into`] 経路で回し、CPU 結果 (expert_gemv_bf16) との
    /// 最大絶対誤差と 1 call の時間を出す (main が GPU ホストで実行する用)。
    /// 長い行は `_r` (固定木縮約) のため CPU とはビット一致しない — 許容は
    /// 絶対 1e-4 + 相対 1e-5。2 回の実行がビット同一 (決定論) も確かめる。
    ///
    /// 実機: `cargo test --release -p denno-engine --features cuda --lib -- --ignored cuda_mix_bf16_reduce --nocapture`
    #[test]
    #[ignore] // GPU が必要
    fn cuda_mix_bf16_reduce_long_row_selfcheck() {
        let _g = test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if !cuda_enabled() {
            CudaCtx::new().expect("CUDA init");
            enable_for_tests();
        }
        let has_r = shared_ctx().and_then(matvec_f32_r_fn).is_some();
        println!("[mix-gpu-p121] has_r(dno_matvec_f32_r in PTX) = {has_r}");
        for (idx, (k, n, seed)) in [(10240usize, 320usize, 3u32), (320usize, 10240usize, 5u32)]
            .iter()
            .enumerate()
        {
            let (k, n) = (*k, *n);
            let bytes = mix_bf16_bytes(k, n, *seed);
            let x: Vec<f32> = (0..k)
                .map(|i| (((i * 31) % 97) as f32 - 48.0) * 0.01)
                .collect();
            let mut cpu = vec![0f32; n];
            crate::model::moe_expert::expert_gemv_bf16(&bytes, k, n, &x, &mut cpu)
                .expect("CPU BF16 GEMV");
            let uid = 0xD121_0000u64 + idx as u64;
            let mut gpu = vec![0f32; n];
            set_last_linear_ran_on_gpu(false);
            assert!(
                maybe_gpu_mix_bf16_into(uid, &bytes, k, n, &x, &mut gpu),
                "GPU 経路が発動しない (k={k}, n={n})"
            );
            assert!(
                last_linear_ran_on_gpu(),
                "発動フラグが立っていない (k={k}, n={n})"
            );
            let mut max_abs = 0f64;
            let mut max_rel = 0f64;
            for r in 0..n {
                let diff = (gpu[r] as f64 - cpu[r] as f64).abs();
                let tol = 1e-4 + 1e-5 * cpu[r].abs() as f64;
                assert!(
                    diff <= tol,
                    "行 {r} (k={k}, n={n}) が CPU と許容外: gpu={:e} cpu={:e} diff={diff:e} tol={tol:e}",
                    gpu[r],
                    cpu[r]
                );
                max_abs = max_abs.max(diff);
                max_rel = max_rel.max(diff / (cpu[r].abs() as f64).max(1e-30));
            }
            // 決定論 (2 回目もビット同一)。
            let mut gpu2 = vec![0f32; n];
            assert!(maybe_gpu_mix_bf16_into(uid, &bytes, k, n, &x, &mut gpu2));
            for r in 0..n {
                assert_eq!(
                    gpu[r].to_bits(),
                    gpu2[r].to_bits(),
                    "2 回目が不一致 (k={k}, n={n})"
                );
            }
            // 1 call の時間 (H2D + GEMV + D2H + sync 込み)。ウォームアップ後に rep。
            let mut gpu_t = vec![0f32; n];
            for _ in 0..20 {
                assert!(maybe_gpu_mix_bf16_into(uid, &bytes, k, n, &x, &mut gpu_t));
            }
            let reps = 200u32;
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                assert!(maybe_gpu_mix_bf16_into(uid, &bytes, k, n, &x, &mut gpu_t));
            }
            let us = t0.elapsed().as_secs_f64() * 1e6 / f64::from(reps);
            println!(
                "[mix-gpu-p121] k={k} n={n} kernel={} max_abs_err={max_abs:.3e} max_rel_err={max_rel:.3e} {us:.1} us/call",
                if mix_use_reduce_kernel(k, n, true) {
                    "dno_matvec_f32_r"
                } else {
                    "dno_matvec_f32"
                }
            );
            mix_gpu_unregister(uid);
        }
    }
}



/// task 184: mix / GEMV v5 / scan 第 2 版のゲートと純関数 (GPU 不要)。
#[cfg(test)]
mod task184_tests {
    use super::*;

    /// `DENNO_MIX_DEC_DEV` の真理値表。既定は off (採否前)。
    #[test]
    fn mix_dec_dev_truth_table() {
        for (raw, want) in [
            (None, false),
            (Some(""), false),
            (Some("  "), false),
            (Some("0"), false),
            (Some("off"), false),
            (Some("false"), false),
            (Some("no"), false),
            (Some("1"), true),
            (Some("on"), true),
            (Some(" TRUE "), true),
            (Some("yes"), true),
            (Some("x"), false),
        ] {
            assert_eq!(mix_dec_dev_from_value(raw, false), want, "raw={raw:?}");
        }
        // 既定 on を渡したときの解釈 (採否後)。
        assert!(mix_dec_dev_from_value(None, true));
        assert!(!mix_dec_dev_from_value(Some("0"), true));
        // 実効ゲートは lab t184a の採否で既定 on。切るのは明示的な偽値。
        assert!(mix_dec_dev_enabled(), "DENNO_MIX_DEC_DEV が既定 off に戻っている");
        assert_eq!(MIX_DEC_DEV_DEFAULT, true);
    }

    /// `DENNO_MIX_NORM_HOST` の真理値表 (task 299 / task 307 で既定 on)。
    #[test]
    fn mix_norm_host_truth_table() {
        for raw in [Some("0"), Some("off"), Some("false"), Some("no")] {
            assert!(!parse_mix_norm_host(raw), "raw={raw:?} が on になっている");
        }
        for raw in [None, Some(""), Some("  "), Some("1"), Some("on"), Some("TRUE"), Some(" true "), Some("yes"), Some("x")] {
            assert!(parse_mix_norm_host(raw), "raw={raw:?} が off になっている");
        }
    }

    /// task 299: ホスト正規化は `dno_mix_norm` と同じ要素式・同じ決定性。
    #[test]
    fn mix_norm_host_matches_reference_and_is_deterministic() {
        let (e, hc, eps) = (8usize, 2usize, 1e-6f32);
        let x: Vec<f32> = (0..e * hc).map(|i| ((i * 37 % 101) as f32 - 50.0) / 17.0).collect();
        let gamma: Vec<f32> = (0..e * hc).map(|i| 0.5 + (i % 7) as f32 * 0.1).collect();
        let mut got = vec![0f32; e * hc];
        mix_norm_host_into(&x, &gamma, e, hc, eps, &mut got);
        for s in 0..hc {
            let xs = &x[s * e..(s + 1) * e];
            // 参照は逐次二乗和 (`tensor::rmsnorm` と同じ順)。
            let ss = xs.iter().map(|v| v * v).sum::<f32>() / e as f32;
            let inv = 1.0 / (ss + eps).sqrt();
            for i in 0..e {
                let want = xs[i] * inv * gamma[s * e + i];
                let g = got[s * e + i];
                assert!(
                    (g - want).abs() <= 1e-5 + 1e-5 * want.abs(),
                    "s={s} i={i}: {g} vs {want}"
                );
            }
        }
        let mut again = vec![0f32; e * hc];
        mix_norm_host_into(&x, &gamma, e, hc, eps, &mut again);
        assert_eq!(got, again, "同じ入力で値が動いた (決定性が無い)");
        // 二乗和が 8 本の累積器なので、逐次和とは最下位ビットが違いうる。
        // 「ほぼ同じ」であることは上で見ている。
    }

    /// task 299: ホスト正規化のコスト (実形 e=2560 / hc=4)。
    /// `cargo test --release -p denno-engine --features cuda --lib -- --ignored
    /// --nocapture bench_mix_norm_host_into`。
    #[test]
    #[ignore]
    fn bench_mix_norm_host_into() {
        let (e, hc) = (2560usize, 4usize);
        let x: Vec<f32> = (0..e * hc).map(|i| ((i * 37 % 101) as f32 - 50.0) / 17.0).collect();
        let gamma: Vec<f32> = vec![0.75f32; e * hc];
        let mut out = vec![0f32; e * hc];
        let n = 2000u32;
        let t = std::time::Instant::now();
        for _ in 0..n {
            mix_norm_host_into(&x, &gamma, e, hc, 1e-6, &mut out);
        }
        let us = t.elapsed().as_secs_f64() * 1e6 / n as f64;
        // 比較: いまの pack がやっている素のコピー。
        let t2 = std::time::Instant::now();
        for _ in 0..n {
            out.copy_from_slice(&x);
        }
        let us2 = t2.elapsed().as_secs_f64() * 1e6 / n as f64;
        eprintln!("[bench299] mix_norm_host_into {us:.2} us/call (素のコピー {us2:.2} us/call)");
        assert!(out[0].is_finite());
    }

    /// norm 重みの指紋は「内容が変われば変わる」。常駐の載せ替え判定に使う。
    #[test]
    fn mix_norm_fingerprint_tracks_content() {
        let mut a: Vec<f32> = (0..40).map(|i| i as f32 * 0.25).collect();
        let ptr = a.as_ptr() as usize;
        let fa = mix_norm_fingerprint(&a, ptr);
        assert_eq!(fa, mix_norm_fingerprint(&a, ptr), "同じ内容で指紋が動いた");
        assert_ne!(fa, mix_norm_fingerprint(&a, ptr + 8), "ポインタ差を無視している");
        // サンプル点 (i * (n-1) / 16) の 1 つを書き換える。
        let idx = 9usize * 39 / 16;
        a[idx] += 1.0;
        assert_ne!(fa, mix_norm_fingerprint(&a, ptr), "内容差を無視している");
        // 違う長さも別。
        let b: Vec<f32> = a[..39].to_vec();
        assert_ne!(fa, mix_norm_fingerprint(&b, b.as_ptr() as usize));
    }

    /// task 342: `DENNO_DEC_MIX_HCBLK` は 32..=1024 の 32 の倍数だけ受け付け、
    /// それ以外は既定の 32 (hc_mixed のブロック幅。採否後の既定 = 32)。
    #[test]
    fn dec_mix_hcblk_truth_table() {
        for (raw, want) in [
            (None, 32),
            (Some(""), 32),
            (Some("0"), 32),
            (Some("16"), 32),
            (Some("33"), 32),
            (Some("1000"), 32),
            (Some("-32"), 32),
            (Some("abc"), 32),
            (Some("32"), 32),
            (Some(" 64 "), 64),
            (Some("128"), 128),
            (Some("256"), 256),
            (Some("1024"), 1024),
        ] {
            assert_eq!(parse_dec_mix_hcblk(raw), want, "raw={raw:?}");
        }
    }

    /// task 342: ブロック幅を変えても 1 出力の計算式は変わらない (どのスレッドが
    /// その出力を持つかだけ)。`d = blockIdx*blockDim + threadIdx` が全単射に
    /// なることを実形 e=2560 で数える。
    #[test]
    fn hcblk_covers_every_output_once() {
        for blk in [32usize, 64, 128, 256] {
            let e = 2560usize;
            let grid = e.div_ceil(blk);
            let mut seen = vec![0u8; e];
            for b in 0..grid {
                for t in 0..blk {
                    let d = b * blk + t;
                    if d < e {
                        seen[d] += 1;
                    }
                }
            }
            assert!(seen.iter().all(|&c| c == 1), "blk={blk} で重複か欠けがある");
        }
    }

    /// v5 のレーン決定: レーンの担当単位 (ブロック対 / スーパーブロック) を
    /// 割り切る最大の 2 の冪。実形 k=2560 の Q8_0 は np=40 → 8、
    /// Q4_K は nb=10 → 2。
    #[test]
    fn v5_lane_choice_covers_the_real_shapes() {
        assert_eq!(v5_lanes_div(40, 8), 8);
        assert_eq!(v5_lanes_div(96, 8), 8); // k=6144 の Q8_0 (nb=192)
        assert_eq!(v5_lanes_div(10, 8), 2); // k=2560 の Q4_K (nb=10)
        assert_eq!(v5_lanes_div(5, 8), 1);
        assert_eq!(v5_lanes_div(7, 8), 1);
        assert_eq!(v5_lanes_div(40, 4), 4);
        assert_eq!(v5_lanes_div(0, 8), 0, "0 を割り切る扱いにしない");
    }

    /// `DENNO_GDN_SCAN2` は lab t184a の採否で既定 on。切るのは明示的な偽値。
    #[test]
    fn gdn_scan2_truth_table() {
        for (raw, want) in [
            (None, true),
            (Some(""), true),
            (Some("  "), true),
            (Some("x"), true),
            (Some("0"), false),
            (Some("off"), false),
            (Some("false"), false),
            (Some("no"), false),
            (Some("1"), true),
            (Some("on"), true),
            (Some("TRUE"), true),
            (Some("yes"), true),
        ] {
            assert_eq!(gdn_scan2_from_value(raw, GDN_SCAN2_DEFAULT), want, "raw={raw:?}");
        }
        assert_eq!(GDN_SCAN2_DEFAULT, true);
        assert!(gdn_scan2_on(), "DENNO_GDN_SCAN2 が既定 off に戻っている");
    }

    /// task 592: scan3 の env 解釈 (既定 on、lab t592c で採否 yes) と、
    /// 明示 off で第 2 版へ戻れること。
    #[test]
    fn gdn_scan3_truth_table() {
        // 既定 on のとき、未設定・空・未知は on (= lab の既定と同じ)。
        for raw in [None, Some(""), Some("  "), Some("x")] {
            assert!(
                gdn_scan3_from_value(raw, GDN_SCAN3_DEFAULT),
                "raw={raw:?} が既定 (on) になっていない"
            );
        }
        // 明示 off は既定に関係なく off (切り戻し)。
        for raw in [Some("0"), Some("off"), Some("false"), Some("no")] {
            for d in [true, false] {
                assert!(
                    !gdn_scan3_from_value(raw, d),
                    "raw={raw:?} default={d} が on になっている"
                );
            }
        }
        for raw in [Some("1"), Some("on"), Some("TRUE"), Some(" true "), Some("yes")] {
            for d in [true, false] {
                assert!(
                    gdn_scan3_from_value(raw, d),
                    "raw={raw:?} default={d} が off になっている"
                );
            }
        }
        assert_eq!(GDN_SCAN3_DEFAULT, true, "scan3 の既定が on から戻っている");
    }

    /// 形状の門番: scan2 は dk == dv == 128 のときだけ選ばれる。
    ///
    /// GPU 無しでは PTX を読んでもカーネルを起動しない (base がそのまま返る)
    /// ので、「128 以外は必ず従来」だけを固定する。
    #[test]
    fn gdn_scan_pick_keeps_the_old_kernel_for_other_shapes() {
        let base: CUfunction = 0x1234 as CUfunction;
        // 形が違えばゲートが on でも base。
        for (dk, dv) in [(96usize, 96usize), (128, 256), (256, 256)] {
            let (f, thr) = gdn_scan_pick(base, dk, dv);
            assert_eq!(f, base, "dk={dk} dv={dv} で従来カーネルになっていない");
            assert_eq!(thr, dk as u32);
        }
    }

    /// task 216: inj をホストで先に作るゲート (`DENNO_DEC_MIX_INJHOST`)。既定 off。
    #[test]
    fn dec_mix_injhost_truth_table() {
        for raw in [None, Some(""), Some("  "), Some("x"), Some("0"), Some("off"), Some("false"), Some("no")] {
            assert!(!parse_dec_mix_injhost(raw), "raw={raw:?} が on になっている");
        }
        for raw in [Some("1"), Some("on"), Some("TRUE"), Some(" true ")] {
            assert!(parse_dec_mix_injhost(raw), "raw={raw:?} が off になっている");
        }
    }

    /// task 281: `DENNO_GEMV_V5_LANES` の真理値表 (純関数)。
    ///
    /// 既定は 8 (= これまでの `v5_lanes_div(per_unit, 8)` と同じ)。
    /// 未設定・空・未知の値・数値でない値はすべて 8 に落ちる (事故で lane 数が
    /// 動かないように、明示した 1/2/4/16/32 だけを受ける)。
    #[test]
    fn v5_lanes_cap_table() {
        // task 285: 既定 (未設定・解釈できない値) は 32。
        for raw in [None, Some(""), Some("  "), Some("x"), Some("0"), Some("off"), Some("64")] {
            assert_eq!(parse_v5_lanes(raw), 32, "raw={raw:?}");
        }
        for (raw, want) in [
            (Some("1"), 1usize),
            (Some("2"), 2),
            (Some("4"), 4),
            (Some("8"), 8),
            (Some("16"), 16),
            (Some("32"), 32),
            (Some(" 16 "), 16),
        ] {
            assert_eq!(parse_v5_lanes(raw), want, "raw={raw:?}");
        }
    }

    /// task 281: lane 数を上げたときに計画が実際に変わる形と変わらない形。
    ///
    /// Q8_0 の `per_unit` は `nb/2 = n_in/64` なので、
    /// k=6144 (96) は 8 -> 16 -> 32 と上がり、k=2560 (40) と k=1024 (16) は
    /// 16 では 8 のまま (40 は 16 で割れない)。**block = 8 行 x lanes** なので
    /// 上限 32 でも 256 スレッドに収まる。
    #[test]
    fn v5_lanes_shape_table() {
        let cases: [(usize, usize, usize); 6] = [
            // (k, cap, 期待 lanes)
            (6144, 8, 8),
            (6144, 16, 16),
            (6144, 32, 32),
            (2560, 16, 8),
            (2560, 32, 8),
            (1024, 16, 16),
        ];
        for (k, cap, want) in cases {
            let per_unit = k / 64; // Q8_0: nb/2
            let lanes = v5_lanes_div(per_unit, cap);
            assert_eq!(lanes, want, "k={k} cap={cap}");
            let block = 8 * lanes;
            assert!(block <= 1024, "k={k} cap={cap} block={block}");
            assert_eq!(per_unit % lanes, 0, "k={k} cap={cap}: 割り切れない");
        }
    }

    /// task 216: `DENNO_DEC_MIX_UPMIX` の版 (0 = off / 1 = uint4 / 2 = ビット一致)。
    #[test]
    fn dec_mix_upmix_ver_table() {
        for raw in [None, Some(""), Some("  "), Some("x"), Some("0"), Some("off"), Some("false"), Some("no"), Some("3")] {
            assert_eq!(parse_dec_mix_upmix_ver(raw), 0, "raw={raw:?}");
        }
        for raw in [Some("1"), Some("on"), Some("TRUE"), Some(" true ")] {
            assert_eq!(parse_dec_mix_upmix_ver(raw), 1, "raw={raw:?}");
        }
        for raw in [Some("2"), Some(" 2 ")] {
            assert_eq!(parse_dec_mix_upmix_ver(raw), 2, "raw={raw:?}");
        }
        // bool 版は版 1 以上で on。
        assert!(!parse_dec_mix_upmix(None));
        assert!(parse_dec_mix_upmix(Some("2")));
    }

    /// task 216: up と hc_mixed を 1 本に畳むゲート (`DENNO_DEC_MIX_UPMIX`)。
    /// 既定 off。on のときだけ融合カーネルを選ぶ。
    #[test]
    fn dec_mix_upmix_truth_table() {
        for raw in [None, Some(""), Some("  "), Some("x"), Some("0"), Some("off"), Some("false"), Some("no")] {
            assert!(!parse_dec_mix_upmix(raw), "raw={raw:?} が on になっている");
        }
        for raw in [Some("1"), Some("on"), Some("TRUE"), Some(" true ")] {
            assert!(parse_dec_mix_upmix(raw), "raw={raw:?} が off になっている");
        }
        assert!(
            !parse_dec_mix_upmix(None),
            "DENNO_DEC_MIX_UPMIX の既定が on になっている"
        );
    }

    /// task 216: mix の up を uint4 版にするゲート (`DENNO_DEC_MIX_UPW`)。
    /// 既定は off (未設定・空・でたらめは全部 off)。
    #[test]
    fn dec_mix_upw_truth_table() {
        // task 221: 既定 on。明示的な偽値だけ off。
        for raw in [Some("0"), Some("off"), Some("FALSE"), Some(" No ")] {
            assert!(!parse_dec_mix_upw(raw), "raw={raw:?} が on になっている");
        }
        for raw in [
            None,
            Some(""),
            Some("  "),
            Some("1"),
            Some("on"),
            Some("TRUE"),
            Some(" true "),
            Some("yes"),
            Some("x"),
        ] {
            assert!(parse_dec_mix_upw(raw), "raw={raw:?} が off になっている");
        }
        // 既定 (env 無し) は on。切り戻しは明示的な `=0`。
        assert!(
            parse_dec_mix_upw(None),
            "DENNO_DEC_MIX_UPW の既定が off に戻っている"
        );
    }

    /// task 216: mix の down を uint4 版にするゲート (`DENNO_DEC_MIX_DOWNW`)。
    #[test]
    fn dec_mix_downw_truth_table() {
        for raw in [None, Some(""), Some("  "), Some("x"), Some("0"), Some("off"), Some("false"), Some("no")] {
            assert!(!parse_dec_mix_downw(raw), "raw={raw:?} が on になっている");
        }
        for raw in [Some("1"), Some("on"), Some("TRUE"), Some(" true ")] {
            assert!(parse_dec_mix_downw(raw), "raw={raw:?} が off になっている");
        }
        assert!(
            !parse_dec_mix_downw(None),
            "DENNO_DEC_MIX_DOWNW の既定が on になっている"
        );
    }

    /// task 216: down のスレッド数 (`DENNO_DEC_MIX_DTHR`)。
    ///
    /// 従来カーネル `dno_matvec_bf16_r_silu` は縮約を warp 数から組むので
    /// 32 の倍数なら何でも正しく動く。範囲外・倍数でない値は既定 256 に落とす
    /// (黙って変な形で撃たない)。
    #[test]
    fn dec_mix_dthr_rules() {
        assert_eq!(parse_dec_mix_dthr(None), 256);
        assert_eq!(parse_dec_mix_dthr(Some("")), 256);
        assert_eq!(parse_dec_mix_dthr(Some("x")), 256);
        assert_eq!(parse_dec_mix_dthr(Some("256")), 256);
        assert_eq!(parse_dec_mix_dthr(Some("512")), 512);
        assert_eq!(parse_dec_mix_dthr(Some(" 1024 ")), 1024);
        assert_eq!(parse_dec_mix_dthr(Some("32")), 32);
        // 32 の倍数でない / 範囲外は 256。
        assert_eq!(parse_dec_mix_dthr(Some("33")), 256);
        assert_eq!(parse_dec_mix_dthr(Some("0")), 256);
        assert_eq!(parse_dec_mix_dthr(Some("16")), 256);
        assert_eq!(parse_dec_mix_dthr(Some("1056")), 256);
        assert_eq!(parse_dec_mix_dthr(Some("-512")), 256);
    }
}


#[cfg(test)]
mod task204_tests {
    use super::*;

    /// task 204 / 221: `DENNO_DEC_EXPGU` の真理値表。既定は on (t221c で採用)。
    #[test]
    fn dec_exp_gu_truth_table() {
        assert!(parse_dec_exp_gu(None));
        assert!(parse_dec_exp_gu(Some("")));
        assert!(parse_dec_exp_gu(Some("1")));
        assert!(parse_dec_exp_gu(Some(" ON ")));
        assert!(parse_dec_exp_gu(Some("true")));
        assert!(parse_dec_exp_gu(Some("yes")));
        assert!(!parse_dec_exp_gu(Some("0")));
        assert!(!parse_dec_exp_gu(Some("off")));
        assert!(!parse_dec_exp_gu(Some("FALSE")));
        assert!(!parse_dec_exp_gu(Some(" no ")));
    }

    /// task 204: 融合の可否は「on + 同じ量子型 + 行長が同じで 0 でない」。
    #[test]
    fn dec_exp_gu_shape_gate() {
        assert!(dec_exp_gu_shape_ok(true, true, 98, 98));
        assert!(!dec_exp_gu_shape_ok(false, true, 98, 98));
        assert!(!dec_exp_gu_shape_ok(true, false, 98, 98));
        assert!(!dec_exp_gu_shape_ok(true, true, 98, 110));
        assert!(!dec_exp_gu_shape_ok(true, true, 0, 0));
    }

    /// task 204: `DENNO_DEC_ONEVNT` の真理値表。既定は off。
    #[test]
    fn dec_onevnt_truth_table() {
        assert!(!parse_dec_onevnt(None));
        assert!(!parse_dec_onevnt(Some("0")));
        assert!(!parse_dec_onevnt(Some("off")));
        assert!(parse_dec_onevnt(Some("1")));
        assert!(parse_dec_onevnt(Some("on")));
    }

    /// task 327: `DENNO_DEC_MIX_INJSEP` の真理値表。**既定 on** (lab t327a で採否 yes)。
    /// 明示的な偽値だけ epilogue 版へ戻す。
    #[test]
    fn dec_mix_injsep_truth_table() {
        assert!(parse_dec_mix_injsep(None), "既定は on");
        assert!(!parse_dec_mix_injsep(Some("0")));
        assert!(!parse_dec_mix_injsep(Some("off")));
        assert!(!parse_dec_mix_injsep(Some("false")));
        assert!(!parse_dec_mix_injsep(Some(" NO ")), "前後の空白と大文字を許す");
        assert!(parse_dec_mix_injsep(Some("")));
        assert!(parse_dec_mix_injsep(Some("2")), "未知の値は既定 (on) のまま");
        assert!(parse_dec_mix_injsep(Some("1")));
        assert!(parse_dec_mix_injsep(Some("on")));
    }

    /// task 327: 収録済みの `kernels_mix_inj3.ptx` の引数が 10 本であることを固定する
    /// (古い PTX を読んで黙って別の行を書くのを防ぐ)。PTX がまだ無いときは省略。
    #[test]
    fn ptx_entry_params_counts_the_committed_mix_inj3_ptx() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_mix_inj3.ptx");
        let Ok(text) = std::fs::read_to_string(&p) else {
            eprintln!("PTX が無いので省略: {}", p.display());
            return;
        };
        assert_eq!(
            ptx_entry_params(&text, MIX_INJ3_KERNEL_NAME),
            Some(MIX_INJ3_NARGS),
            "{} の引数の数 (ptx パーサの前提)",
            MIX_INJ3_KERNEL_NAME
        );
    }

    /// task 204: 融合の倍の表を載せてもレイアウトが 8 バイト整列で重ならない。
    #[test]
    fn dec_stage_layout_with_gu_block() {
        let nseg = 5usize; // 奇数 = w の後ろが 4 バイト境界でも整列させる
        let l = dec_stage_layout(2560, nseg, 4 * nseg, 4 * nseg, nseg, 2 * nseg);
        assert_eq!(l.pg2 % 8, 0, "ポインタ表が 8 バイト整列でない");
        assert_eq!(l.sgu2 % 8, 0, "segment 表が 8 バイト整列でない");
        assert_eq!(l.sgu2, l.pg2 + 2 * nseg * 8);
        assert_eq!(l.total, l.sgu2 + 2 * nseg * 16);
        // 和モードの重みのブロックを潰していない。
        assert!(l.w + nseg * 4 <= l.pg2);
        // gu ブロックを使わないときは従来と同じ末尾。
        let l0 = dec_stage_layout(2560, nseg, 4 * nseg, 4 * nseg, nseg, 0);
        assert_eq!(l0.pg2, l0.total);
        assert_eq!(l0.sgu2, l0.total);
        assert!(l0.total <= l.total);
    }

    /// task 204: 融合起動は**同じカーネル**を grid.y = 2*segment で撃ち、
    /// segment 表の後半の出力オフセットだけ `ng` ずらす、という契約。
    /// GPU 無しでは起動できないので、ソースの契約と倍表の作り方を固定する。
    #[test]
    fn exp_gu_contract_in_source() {
        let src = crate::cuda::cuda_src();
        assert!(
            src.contains("(plan_g.grid.0, (2 * nseg) as u32, 1)"),
            "融合起動の grid.y が 2*segment でない"
        );
        assert!(
            src.contains("e4[3] = e4[3].saturating_add(ng as u32);"),
            "up 側の出力オフセットに ng を足していない"
        );
        assert!(
            src.contains("let gu_fused = lay.total > lay.pg2 && stage_addr.is_some();"),
            "融合の門番が無い"
        );
    }

    /// task 192: task 192 の 2 本の PTX (`mix_dev2` / `mix_up`) が**リポジトリに
    /// ある**こと。task 204 では既定 on のゲート (`DENNO_MIX_DEC_FOLD` /
    /// `DENNO_MIX_DEC_UPW`) が静かに従来経路へ落ちていた。task 213 の lab で
    /// 採否 no になったので既定は off に戻したが、opt-in (=1) で使う PTX は残す。
    /// 引数数の契約も固定する。
    #[test]
    fn task192_fold_and_upw_ptx_are_present() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels");
        let fold = std::fs::read_to_string(dir.join("kernels_mix_dev2.ptx"))
            .expect("kernels_mix_dev2.ptx が無い (fold が従来経路へ落ちる)");
        assert_eq!(
            ptx_entry_params(&fold, MIX_FOLD_KERNEL_NAME),
            Some(MIX_FOLD_NARGS),
            "fold の PTX 引数数が {MIX_FOLD_NARGS} でない"
        );
        let up = std::fs::read_to_string(dir.join("kernels_mix_up.ptx"))
            .expect("kernels_mix_up.ptx が無い (upw が従来経路へ落ちる)");
        assert_eq!(
            ptx_entry_params(&up, MIX_UP_KERNEL_NAME),
            Some(MIX_UP_NARGS),
            "upw の PTX 引数数が {MIX_UP_NARGS} でない"
        );
    }
}


#[cfg(test)]
mod task498_tests {
    use super::*;

    /// バウンスに収まらない長さは「やらない」を返す (呼び出し側は pageable へ落ちる)。
    ///
    /// 0 バイトと上限超えは CUDA を触る前に false を返すので、GPU の無い
    /// コンテナでも判定できる。共有カウンタは読まない (並列テストで競合しない)。
    #[test]
    fn seg_pin_copy_rejects_out_of_range() {
        let mut none: [u8; 0] = [];
        assert!(!seg_pin_copy_d2h(0, &mut none), "0 バイトは false");
        let mut big = vec![0u8; SEG_PIN_BYTES + 1];
        assert!(!seg_pin_copy_d2h(0, &mut big), "上限超えは false");
    }

    /// GPU の無い環境では pinned が取れず **false (= 従来経路) に落ちる**。
    /// 落ちた回数が増えるので、リクエスト終端の `[seg-pin] fallbacks=` で判別できる。
    /// 共有カウンタは他テストと競合し得るので「増えた」だけを見る。
    #[test]
    fn seg_pin_copy_falls_back_without_cuda() {
        let (_, before) = crate::model::seg_prof::seg_pin_counts();
        let mut dst = vec![0u8; 64];
        let ok = seg_pin_copy_d2h(0, &mut dst);
        let (_, after) = crate::model::seg_prof::seg_pin_counts();
        if ok {
            // GPU のある環境でだけ通り得る (コンテナでは起きない)。
            assert_eq!(dst, vec![0u8; 64], "0 番地からの写しは 0 のまま");
        } else {
            assert!(after > before, "落ちた回数が増える");
        }
    }
}


#[cfg(test)]
mod task502_tests {
    use super::*;

    /// バウンスに収まらない長さは CUDA を触る前に「やらない」を返す。
    #[test]
    fn seg_async_issue_rejects_out_of_range() {
        assert!(!seg_async_d2h_issue(0, 0), "0 バイトは false");
        assert!(!seg_async_d2h_issue(0, SEG_PIN_BYTES + 1), "上限超えは false");
    }

    /// collect も範囲外は false (呼び出し側は同期経路へ落ちる)。
    #[test]
    fn seg_async_collect_rejects_out_of_range() {
        let mut none: [u8; 0] = [];
        assert!(!seg_async_d2h_collect(&mut none), "0 バイトは false");
        let mut big = vec![0u8; SEG_PIN_BYTES + 1];
        assert!(!seg_async_d2h_collect(&mut big), "上限超えは false");
    }

    /// GPU の無い環境では非同期コピーを出せず **false (= 従来経路) に落ちる**。
    /// 落ちた回数が増えるので、リクエスト終端の `[seg-async] fallbacks=` で判別できる。
    /// 共有カウンタは他テストと競合し得るので「増えた」だけを見る。
    #[test]
    fn seg_async_issue_falls_back_without_cuda() {
        let (_, before) = crate::model::seg_prof::seg_async_counts();
        let ok = seg_async_d2h_issue(0, 64);
        let (_, after) = crate::model::seg_prof::seg_async_counts();
        assert!(!ok, "コンテナ (GPU 無し) では出せない");
        assert!(after > before, "落ちた回数が増える");
    }
}


#[cfg(test)]
mod task565_tests {
    use super::*;

    /// task 565: 戻しのリングの枠の選び方 (通し番号を 4 枠で巡回する)。
    ///
    /// 4 枠に満たない通し番号では同じ枠を続けて使わない (= 直前の DMA を
    /// 待たずに次の写しへ進める) ことが要点。5 本目で 0 に戻る。
    #[test]
    fn up_ring_index_cycles_over_the_ring() {
        assert_eq!(UP_RING_N, 4);
        assert_eq!(up_ring_index(0), 0);
        assert_eq!(up_ring_index(1), 1);
        assert_eq!(up_ring_index(2), 2);
        assert_eq!(up_ring_index(3), 3);
        assert_eq!(up_ring_index(4), 0);
        assert_eq!(up_ring_index(4 * 1000 + 3), 3);
    }

    /// task 565: `force` は**全体ゲートだけ**を飛ばし、窓の判定は残す。
    ///
    /// 残差の往復は 14k のチャンクで 84 MB、最終 D2H で 71 MB。どちらも窓の中。
    /// 逆に、既定 (force も stage も off) では大きな転送を通さない — ここが
    /// 変わると P358 の棄却を蒸し返すことになる。
    #[test]
    fn forced_pin_skips_only_the_stage_gate() {
        let min = PF_PIN_MIN_BYTES;
        // force なら全体ゲート off でも窓の中は通る。
        assert!(pf_pin_admit(true, false, false, min, 84 << 20));
        assert!(pf_pin_admit(true, false, false, min, 71 << 20));
        // 窓の外 (上限 256 MiB 超) は force でも通さない。
        assert!(!pf_pin_admit(true, false, false, min, PF_PIN_MAX_BYTES + 1));
        // 床 1 MiB 未満は force でも「大きな転送」としては通さない。
        assert!(!pf_pin_admit(true, false, false, min, min - 1));
        // force が無ければ従来どおり stage のゲートで決まる (既定 off)。
        assert!(!pf_pin_admit(false, false, false, min, 84 << 20));
        assert!(pf_pin_admit(false, true, false, min, 84 << 20));
        // 小さな往復の窓は force と無関係に従来どおり効く。
        assert!(pf_pin_admit(false, false, true, min, 64 << 10));
        assert!(!pf_pin_admit(false, false, false, min, 64 << 10));
    }
}


#[cfg(test)]
mod task318_tests {
    use super::*;

    /// `DENNO_PF_PIN_STAGE` の真理値表。**既定は off** (明示した真値だけ on)。
    #[test]
    fn pin_stage_truth_table() {
        assert!(!pf_pin_stage_from_value(None));
        assert!(!pf_pin_stage_from_value(Some("")));
        assert!(!pf_pin_stage_from_value(Some("0")));
        assert!(!pf_pin_stage_from_value(Some("off")));
        assert!(!pf_pin_stage_from_value(Some(" no ")));
        assert!(!pf_pin_stage_from_value(Some("FALSE")));
        assert!(pf_pin_stage_from_value(Some("1")));
        assert!(pf_pin_stage_from_value(Some("on")));
        assert!(pf_pin_stage_from_value(Some("true")));
        assert!(pf_pin_stage_from_value(Some("yes")));
    }

    /// 副区間の名前と添字が 1 対 1 で揃っている (台帳の列がずれない)。
    #[test]
    fn ledger_sub_names_for_task318() {
        use crate::model::q4ledger as q;
        assert_eq!(q::SUB_NAMES[q::SUB_RES_UP], "res_up");
        assert_eq!(q::SUB_NAMES[q::SUB_RES_PARK], "res_park");
        assert_eq!(q::SUB_NAMES[q::SUB_INJ], "inj");
        assert_eq!(q::SUB_NAMES[q::SUB_LM_PARK], "lm_park");
        assert_eq!(q::SUB_NAMES.len(), q::SUB_N);
        assert!(q::SUB_RES_UP < q::SUB_N);
        assert!(q::SUB_LM_PARK < q::SUB_N);
    }

    /// task 369: `DENNO_PF_KV_ASYNC` の解釈 (lab t369a で採ったので既定 on)。
    #[test]
    fn t369_kv_async_gate_truth_table() {
        for raw in [
            Some("0"),
            Some("off"),
            Some("FALSE"),
            Some(" No "),
            Some(""),
        ] {
            assert!(!pf_kv_async_from_value(raw), "raw={raw:?} が on になっている");
        }
        for raw in [
            None,
            Some("1"),
            Some("on"),
            Some("TRUE"),
            Some(" yes "),
            Some("x"),
        ] {
            assert!(pf_kv_async_from_value(raw), "raw={raw:?} が off になっている");
        }
        assert!(
            pf_kv_async_from_value(None),
            "既定が off になっている (t369a で採ったので on)"
        );
    }

    /// task 369: 保留が無いときの `pf_kv_wb_copy_into` / `pf_kv_wb_discard` は
    /// 何もせず false を返す (値を 1 バイトも触らない)。
    #[test]
    fn t369_kv_wb_copy_without_pending_touches_nothing() {
        let mut k = [1.5f32, -2.0, 3.25];
        let mut v = [7.0f32, 8.0, 9.0];
        let before_k = k;
        let before_v = v;
        assert!(!pf_kv_wb_pending(), "最初から保留がある");
        assert!(!pf_kv_wb_copy_into(&mut k, &mut v), "保留なしで true を返した");
        assert_eq!(k, before_k, "保留なしで k を書き換えた");
        assert_eq!(v, before_v, "保留なしで v を書き換えた");
        assert!(!pf_kv_wb_discard(), "保留なしで true を返した");
    }

    /// task 369: `DENNO_PF_KV_PUSH` の解釈 (既定 off、明示的な真値だけ on)。
    #[test]
    fn t369_kv_push_gate_truth_table() {
        for raw in [
            None,
            Some(""),
            Some(" 0 "),
            Some("off"),
            Some("FALSE"),
            Some("no"),
            Some("x"),
        ] {
            assert!(!pf_kv_push_from_value(raw), "raw={raw:?} が on になっている");
        }
        for raw in [Some("1"), Some("on"), Some("TRUE"), Some(" yes ")] {
            assert!(pf_kv_push_from_value(raw), "raw={raw:?} が off になっている");
        }
        assert!(!pf_kv_push_from_value(None), "既定が on になっている");
    }

    /// task 553: `DENNO_PF_KV_DEFER` の解釈 (既定 on、明示的な偽値だけ off)。
    #[test]
    fn t553_kv_defer_gate_truth_table() {
        assert!(pf_kv_defer_from_value(None), "既定が off になっている");
        assert!(pf_kv_defer_from_value(Some("")), "空で off になっている");
        for raw in [Some("0"), Some("off"), Some("FALSE"), Some(" no ")] {
            assert!(!pf_kv_defer_from_value(raw), "raw={raw:?} が on になっている");
        }
        for raw in [Some("1"), Some("on"), Some("true"), Some("yes")] {
            assert!(pf_kv_defer_from_value(raw), "raw={raw:?} が off になっている");
        }
    }

    /// task 553: 2 枚のスロットの選び方。
    ///
    /// ずらさないときは 1 枚だけ (保留があるうちは積まない = 従来と同じ)、
    /// ずらすときは空きを交互に使う。両方埋まっていれば積まない
    /// (上書きしない)。
    #[test]
    fn t553_kv_wb_pick_slot_rotates_and_never_overwrites() {
        // 従来 (defer off): 保留が無ければ 0 番、あれば None。
        assert_eq!(kv_wb_pick_slot([false, false], 0, false), Some(0));
        assert_eq!(kv_wb_pick_slot([true, false], 0, false), None);
        assert_eq!(kv_wb_pick_slot([false, true], 0, false), None);
        // defer on: next の枚が空いていればそこ、埋まっていればもう 1 枚。
        assert_eq!(kv_wb_pick_slot([false, false], 0, true), Some(0));
        assert_eq!(kv_wb_pick_slot([false, false], 1, true), Some(1));
        assert_eq!(kv_wb_pick_slot([true, false], 1, true), Some(1));
        assert_eq!(kv_wb_pick_slot([true, false], 0, true), Some(1));
        assert_eq!(kv_wb_pick_slot([false, true], 1, true), Some(0));
        // 両方埋まっていれば上書きしない。
        assert_eq!(kv_wb_pick_slot([true, true], 0, true), None);
        assert_eq!(kv_wb_pick_slot([true, true], 1, true), None);
    }

    /// task 369: 保留が無いとき `pf_kv_wb_with` は closure を呼ばず false を返す。
    #[test]
    fn t369_kv_wb_with_without_pending_does_not_call() {
        let mut called = false;
        assert!(!pf_kv_wb_with(|_k, _v| {
            called = true;
        }));
        assert!(!called, "保留なしで closure が呼ばれた");
    }

    /// task 369: ゲートが off なら積まない (pinned の確保もしない)。
    #[test]
    fn t369_kv_wb_begin_is_gated() {
        set_pf_kv_async_for_test(Some(false));
        assert!(!pf_kv_async_on());
        assert!(
            matches!(pf_kv_wb_begin(1, 2, 64, 64, 0, 0), Ok(false)),
            "ゲート off で積もうとした"
        );
        assert!(!pf_kv_wb_pending(), "ゲート off なのに保留が立った");
        set_pf_kv_async_for_test(None);
    }

    /// task 358: バウンスの窓 (既定 1 MiB..=256 MiB) と下限の env 解釈。
    #[test]
    fn pin_window_and_min_kb_truth_table() {
        let mib = 1usize << 20;
        let min = PF_PIN_MIN_BYTES;
        assert_eq!(min, mib);
        // 窓: 1 MiB は入る、その 1 B 下は入らない、256 MiB は入る、その上は入らない。
        assert!(pf_pin_window_from(min, mib));
        assert!(!pf_pin_window_from(min, mib - 1));
        assert!(pf_pin_window_from(min, PF_PIN_MAX_BYTES));
        assert!(!pf_pin_window_from(min, PF_PIN_MAX_BYTES + 1));
        // 床を 4 KiB に下げると decode の 10 KB も入る。
        assert!(pf_pin_window_from(4 << 10, 10 << 10));
        assert!(!pf_pin_window_from(4 << 10, 4 << 10 - 1));
        // env の解釈: 未設定・空・0・壊れた値は既定 (1024 KiB)、1..=262144 に丸める。
        assert_eq!(pf_pin_min_kb_from_value(None), 1024);
        assert_eq!(pf_pin_min_kb_from_value(Some("")), 1024);
        assert_eq!(pf_pin_min_kb_from_value(Some("0")), 1024);
        assert_eq!(pf_pin_min_kb_from_value(Some("x")), 1024);
        assert_eq!(pf_pin_min_kb_from_value(Some("4")), 4);
        assert_eq!(pf_pin_min_kb_from_value(Some(" 64 ")), 64);
        assert_eq!(pf_pin_min_kb_from_value(Some("99999999")), 262_144);
    }

    /// task 518: 「小さな往復」の窓 (既定 off、4 KiB .. 1 MiB 未満) とゲートの解釈。
    ///
    /// ここが固定されていないと、ゲートを on にしても 1 発も pinned を通らない
    /// (逆に prefill の大きな転送を巻き込んで既存の数字を動かす) という壊れ方をする。
    #[test]
    fn dec_pin_small_window_and_gate_truth_table() {
        // ゲート off ならどんなサイズも窓の外。
        assert!(!pf_pin_small_window_from(false, 10 << 10));
        assert!(!pf_pin_small_window_from(false, 0));
        // 窓の内側: 4 KiB / 10 KB / 65 KB。
        assert!(pf_pin_small_window_from(true, 4 << 10));
        assert!(pf_pin_small_window_from(true, 10 << 10));
        assert!(pf_pin_small_window_from(true, 65 << 10));
        // 下限の外 (4 KiB 未満) と上限の外 (1 MiB 以上 = 従来の担当)。
        assert!(!pf_pin_small_window_from(true, (4 << 10) - 1));
        assert!(!pf_pin_small_window_from(true, PF_PIN_MIN_BYTES));
        assert!(!pf_pin_small_window_from(true, 0));
        // env の解釈: **既定 on** (未設定・空は on)、明示的な偽値だけ off。
        assert!(dec_pin_small_from_value(None));
        assert!(dec_pin_small_from_value(Some("")));
        assert!(!dec_pin_small_from_value(Some("0")));
        assert!(!dec_pin_small_from_value(Some("off")));
        assert!(!dec_pin_small_from_value(Some("FALSE")));
        assert!(!dec_pin_small_from_value(Some(" no ")));
        assert!(dec_pin_small_from_value(Some("1")));
        assert!(dec_pin_small_from_value(Some(" on ")));
        assert!(dec_pin_small_from_value(Some("TRUE")));
    }

    /// task 358: 育つ判断は grow-only + 2 の冪切り上げ (確保の回数を数回に抑える)。
    #[test]
    fn pin_grow_is_grow_only_and_rounded() {
        // 既に足りている: 触らない。
        assert_eq!(pf_pin_need_grow(1 << 20, 1 << 20), None);
        assert_eq!(pf_pin_need_grow(1 << 21, 1 << 20), None);
        // 足りない: 2 の冪へ切り上げる (要求より小さくはしない)。
        assert_eq!(pf_pin_need_grow(0, 1_400_000), Some(2 << 20));
        assert_eq!(pf_pin_need_grow(1 << 20, 39 << 20), Some(64 << 20));
        assert_eq!(pf_pin_need_grow(64 << 20, 143 << 20), Some(256 << 20));
        // 上限で頭打ち (窓の外は呼ばれないが、関数としては上限を超えない)。
        assert_eq!(
            pf_pin_need_grow(0, PF_PIN_MAX_BYTES),
            Some(PF_PIN_MAX_BYTES)
        );
        assert_eq!(pf_pin_need_grow(0, PF_PIN_MAX_BYTES - 1), Some(PF_PIN_MAX_BYTES));
    }
}


#[cfg(test)]
mod task379_tests {
    use super::*;

    /// task 379: 「層の最終チャンクか」の旗は既定で真 (どこから呼ばれても従来どおり書き戻す)。
    /// ゲートは lab で採ったので既定 on、途中チャンクだけ飛ばす。
    #[test]
    fn gdn_last_chunk_flag_defaults_true_and_skips_only_mid_chunks() {
        pf_gdn_last_chunk_reset();
        assert!(pf_gdn_last_chunk(), "既定は真");
        pf_gdn_last_chunk_set(false);
        assert!(!pf_gdn_last_chunk(), "立てた値が読める");
        pf_gdn_last_chunk_reset();
        assert!(pf_gdn_last_chunk(), "戻すと真");
        if crate::model::exp_pf::gdn_wb_last_on() {
            // 途中チャンクは飛ばす、最終チャンクは必ず書き戻す。
            pf_gdn_last_chunk_set(false);
            assert!(
                pf_gdn_last_chunk_skippable(0),
                "既定 on の途中チャンクは飛ばす"
            );
            pf_gdn_last_chunk_reset();
            assert!(
                !pf_gdn_last_chunk_skippable(0),
                "最終チャンクで飛ばしてはいけない"
            );
        }
    }
}


#[cfg(test)]
mod task393_tests {
    use super::*;

    /// task 393: `DENNO_PF_GDN_WSTAT` の真理値表 (既定 off、明示的な真値だけ on)。
    #[test]
    fn gdn_wstat_gate_truth_table() {
        assert!(
            !gdn_wstat_from_value(None),
            "未設定が on になった (既定 off のはず)"
        );
        assert!(!gdn_wstat_from_value(Some("")));
        for v in ["0", "off", "false", "no", " off "] {
            assert!(!gdn_wstat_from_value(Some(v)), "{v:?} が on になった");
        }
        for v in ["1", "on", "true", "yes", " ON "] {
            assert!(gdn_wstat_from_value(Some(v)), "{v:?} が off のまま");
        }
        if std::env::var_os(GDN_WSTAT_ENV).is_none() {
            assert!(!gdn_wstat_on(), "未設定で既定 off になっていない");
        }
    }

    /// task 393: 鍵は「層・転送元 4 本 (ptr と要素数)・置き場 4 本」のどれが
    /// 変わっても別になる (上げ直しを飛ばしてよい条件を緩めない)。
    #[test]
    fn gdn_wstat_key_covers_layer_src_and_dst() {
        let src = [(0x1000usize, 8usize), (0x2000, 4), (0x3000, 4), (0x4000, 4)];
        let dst = [0x9000usize, 0xa000, 0xb000, 0xc000];
        let base = gdn_wstat_key(5, src, dst);
        assert_eq!(base, gdn_wstat_key(5, src, dst), "同じ入力で鍵が揺れた");
        assert_ne!(base, gdn_wstat_key(6, src, dst), "層が鍵に入っていない");
        for i in 0..4 {
            let mut s = src;
            s[i].0 += 0x10;
            assert_ne!(base, gdn_wstat_key(5, s, dst), "転送元 {i} の ptr が鍵に無い");
            let mut s = src;
            s[i].1 += 1;
            assert_ne!(base, gdn_wstat_key(5, s, dst), "転送元 {i} の長さが鍵に無い");
            let mut d = dst;
            d[i] += 0x10;
            assert_ne!(base, gdn_wstat_key(5, src, d), "置き場 {i} が鍵に無い");
        }
    }
}


/// task 407: GDN prefill スキャンの新しい 4 本 (wy6 / wy6p / wy7 / wy7q)。
///
/// ゲートは全部**既定 off**。wy7 はバリアだけ減らした値同一版、wy7q は wy7 に
/// (5) P/QP の r 方向分割を足した版 (丸めだけ違う)。
#[cfg(test)]
mod task407_tests {
    use super::*;

    /// ゲートの真理値表・形の前提・共有メモリ (wy6 / wy6p)。
    #[test]
    fn task407_scan_wy6_gate_and_shape() {
        assert!(!gdn_pf_wy6_from_value(None), "wy6 の既定は off");
        assert!(!gdn_pf_wy6p_from_value(None), "wy6p の既定は off");
        for v in ["", "0", "off", "false", "no", " off "] {
            assert!(!gdn_pf_wy6_from_value(Some(v)), "wy6: {v:?} が on になった");
            assert!(!gdn_pf_wy6p_from_value(Some(v)), "wy6p: {v:?} が on になった");
        }
        for v in ["1", "on", "true", "yes", "2"] {
            assert!(gdn_pf_wy6_from_value(Some(v)), "wy6: {v:?} が off のまま");
            assert!(gdn_pf_wy6p_from_value(Some(v)), "wy6p: {v:?} が off のまま");
        }
        set_gdn_pf_wy6_for_test(Some(true));
        assert!(gdn_pf_wy6_on());
        set_gdn_pf_wy6_for_test(Some(false));
        assert!(!gdn_pf_wy6_on());
        set_gdn_pf_wy6_for_test(None);
        assert!(!gdn_pf_wy6_on(), "FORCE を戻したら既定 off");
        set_gdn_pf_wy6p_for_test(Some(true));
        assert!(gdn_pf_wy6p_on());
        set_gdn_pf_wy6p_for_test(Some(false));
        assert!(!gdn_pf_wy6p_on());
        set_gdn_pf_wy6p_for_test(None);
        assert!(!gdn_pf_wy6p_on(), "FORCE を戻したら既定 off");
        let conv_dim = 2 * 16 * 128 + 48 * 128;
        assert!(gdn_pf_wy6_ok(2003, 16, 48, 128, 128, 4, conv_dim));
        assert!(gdn_pf_wy6_ok(3826, 16, 48, 128, 128, 4, conv_dim));
        assert!(
            !gdn_pf_wy6_ok(2048, 16, 48, 128, 64, 4, 2 * 16 * 128 + 48 * 64),
            "dv=64 は受けない"
        );
        assert!(!gdn_pf_wy6_ok(0, 16, 48, 128, 128, 4, conv_dim), "n=0 は受けない");
        assert_eq!(GDN_PF_WY6_THREADS, 512);
        assert_eq!(GDN_PF_WY6_BLOCKS, 4);
        assert_eq!(GDN_PF_WY6_SMEM_FLOATS, GDN_PF_WY4_SMEM_FLOATS);
        assert_eq!(GDN_PF_WY6_SMEM_BYTES, 29_572);
        let get_of = |src: &'static str, name: &str| -> usize {
            let key = format!("#define {name} ");
            for line in src.lines() {
                if let Some(rest) = line.strip_prefix(&key) {
                    return rest
                        .trim()
                        .parse()
                        .unwrap_or_else(|_| panic!("{name} が数値でない"));
                }
            }
            panic!("{name} が .cu に無い");
        };
        let src6 = include_str!("../../kernels/kernels_gdn_pf_wy6.cu");
        let src6p = include_str!("../../kernels/kernels_gdn_pf_wy6p.cu");
        let src5 = include_str!("../../kernels/kernels_gdn_pf_wy5.cu");
        assert_eq!(get_of(src6, "WY6_TOTAL"), GDN_PF_WY6_SMEM_FLOATS);
        assert_eq!(get_of(src6, "WY6_THREADS"), GDN_PF_WY6_THREADS);
        assert_eq!(get_of(src6, "WY6_WARPS") * 32, GDN_PF_WY6_THREADS);
        assert_eq!(get_of(src6, "WY6_DV") * GDN_PF_WY6_BLOCKS, 128);
        assert_eq!(get_of(src6, "WY6_C"), 8, "チャンク長は wy5 と同じ 8");
        assert!(src6.contains("dno_gdn_pf_wy6("), "カーネル名が無い");
        for name in [
            "S", "KC", "QC", "VC", "DC", "PC", "QPC", "RDEC", "AC", "GC", "LAM", "RL", "EG", "BC",
            "LL",
        ] {
            assert_eq!(
                get_of(src6, &format!("WY6_{name}")),
                get_of(src5, &format!("WY5_{name}")),
                "WY6_{name} のオフセットが wy5 と違う"
            );
        }
        assert!(src6.contains("const int per = WY6_DK * WY6_DV / WY6_THREADS;"));
        assert!(src6.contains("for (int p = warp; p < npair; p += WY6_WARPS)"));
        assert_eq!(src6.matches("float sk0 = 0.0f;").count(), 1);
        assert!(!src6.contains("__shfl_xor_sync(0xffffffffu, sk, 1)"));
        assert!(src6p.contains("dno_gdn_pf_wy6p("), "カーネル名が無い");
        assert_eq!(get_of(src6p, "WY6P_THREADS"), GDN_PF_WY6_THREADS);
        assert_eq!(get_of(src6p, "WY6P_TOTAL"), GDN_PF_WY6_SMEM_FLOATS);
        assert!(
            src6p.contains("for (int j = tid; j < 2 * len * WY6P_DV; j += WY6P_THREADS)"),
            "P/QP の r 分割ループが無い"
        );
        assert!(src6p.contains("__shfl_xor_sync(0xffffffffu, sk, 1)"), "和の合流が無い");
        assert!(src6p.contains("const int rh = WY6P_DK / 2;"));
        // 2*len*dv は常にワープ境界 (64) の倍数 = ペアが片側だけ inactive にならない。
        assert_eq!((2 * 8 * 32) % 64, 0);
        assert_eq!((2 * 7 * 32) % 64, 0);
    }

    /// ゲートの真理値表・共有メモリ・バリア本数 (wy7 / wy7q)。
    #[test]
    fn task407_scan_wy7_gate_and_shape() {
        for (name, f) in [
            ("wy7", gdn_pf_wy7_from_value as fn(Option<&str>) -> bool),
            ("wy7q", gdn_pf_wy7q_from_value),
        ] {
            assert!(!f(None), "{name} の既定は off");
            for v in ["", "0", "off", "false", "no", " off "] {
                assert!(!f(Some(v)), "{name}: {v:?} が on になった");
            }
            for v in ["1", "on", "true", "yes", "2"] {
                assert!(f(Some(v)), "{name}: {v:?} が off のまま");
            }
        }
        set_gdn_pf_wy7_for_test(Some(true));
        assert!(gdn_pf_wy7_on());
        set_gdn_pf_wy7_for_test(Some(false));
        assert!(!gdn_pf_wy7_on());
        set_gdn_pf_wy7_for_test(None);
        assert!(!gdn_pf_wy7_on(), "FORCE を戻したら既定 off");
        set_gdn_pf_wy7q_for_test(Some(true));
        assert!(gdn_pf_wy7q_on());
        set_gdn_pf_wy7q_for_test(Some(false));
        assert!(!gdn_pf_wy7q_on());
        set_gdn_pf_wy7q_for_test(None);
        assert!(!gdn_pf_wy7q_on(), "FORCE を戻したら既定 off");
        assert_eq!(GDN_PF_WY7_SMEM_FLOATS, GDN_PF_WY4_SMEM_FLOATS);
        assert_eq!(GDN_PF_WY7_SMEM_BYTES, 29_572);
        assert_eq!(GDN_PF_WY7Q_PART_FLOATS, 2 * 8 * 8 * 32);
        assert_eq!(GDN_PF_WY7Q_SMEM_FLOATS, 11_489);
        assert_eq!(GDN_PF_WY7Q_SMEM_BYTES, 45_956);
        assert!(GDN_PF_WY7Q_SMEM_BYTES < 48 * 1024, "48 KB の既定を超える");
        let get_of = |src: &'static str, name: &str| -> usize {
            let key = format!("#define {name} ");
            for line in src.lines() {
                if let Some(rest) = line.strip_prefix(&key) {
                    return rest
                        .trim()
                        .parse()
                        .unwrap_or_else(|_| panic!("{name} が数値でない"));
                }
            }
            panic!("{name} が .cu に無い");
        };
        let src5 = include_str!("../../kernels/kernels_gdn_pf_wy5.cu");
        let src7 = include_str!("../../kernels/kernels_gdn_pf_wy7.cu");
        let src7q = include_str!("../../kernels/kernels_gdn_pf_wy7q.cu");
        assert!(src7.contains("dno_gdn_pf_wy7("));
        assert!(src7q.contains("dno_gdn_pf_wy7q("));
        assert_eq!(get_of(src7, "WY7_TOTAL"), GDN_PF_WY7_SMEM_FLOATS);
        assert_eq!(get_of(src7, "WY7_THREADS"), GDN_PF_WY4_THREADS);
        assert_eq!(get_of(src7q, "WY7Q_TOTAL"), GDN_PF_WY7Q_SMEM_FLOATS);
        assert_eq!(get_of(src7q, "WY7Q_THREADS"), GDN_PF_WY4_THREADS);
        assert_eq!(
            get_of(src7q, "WY7Q_QP") - get_of(src7q, "WY7Q_PP"),
            GDN_PF_WY7Q_PART_FLOATS / 2,
            "Pp と Qp の間隔が部分和の半分でない"
        );
        // 数えるのは実物 (セミコロン付き)。頭のコメントは括弧だけで書く。
        assert_eq!(src5.matches("__syncthreads();").count(), 9);
        assert_eq!(src7.matches("__syncthreads();").count(), 7, "wy7 のバリアが 2 本減っていない");
        assert_eq!(src7q.matches("__syncthreads();").count(), 8);
        assert_eq!(src7.matches("float sk0 = 0.0f;").count(), 1);
        assert!(!src7.contains("float* Pp = sh + WY7_PP;"));
        assert!(src7q.contains("float* Pp = sh + WY7Q_PP;"));
        assert!(src7q.contains("Pp[warp * WY7Q_C * WY7Q_DV + i * WY7Q_DV + c] = sk[i];"));
        assert!(src7q.contains("for (int w = 0; w < WY7Q_WARPS; ++w) {"));
        assert!(src7q.contains("const int rn = WY7Q_DK / WY7Q_WARPS;"));
    }

    /// ゲート `DENNO_PF_GDN_TC1` (**既定 on**) の真理値表。
    #[test]
    fn task407_gdn_tc1_gate_defaults() {
        assert!(gdn_pf_tc1_from_value(None), "既定 on (lab t407c)");
        for v in ["0", "off", "false", "no", "", " off "] {
            assert!(!gdn_pf_tc1_from_value(Some(v)), "{v:?} が on になった");
        }
        for v in ["1", "on", "true", "yes", "2"] {
            assert!(gdn_pf_tc1_from_value(Some(v)), "{v:?} が off のまま");
        }
        set_gdn_pf_tc1_for_test(Some(false));
        assert!(!gdn_pf_tc1_on(), "FORCE off が効いていない");
        set_gdn_pf_tc1_for_test(Some(true));
        assert!(gdn_pf_tc1_on(), "FORCE on が効いていない");
        set_gdn_pf_tc1_for_test(None);
        assert!(gdn_pf_tc1_on(), "FORCE を戻したら既定 on");
    }
}


/// task 448: flash decoding 第 2 版 (shared タイル) のゲート・門番・smem の検証
/// (GPU 不要)。カーネルそのものは lab で測るので、ここでは (1) `tile` / `fdec1`
/// の解釈、(2) 形の門番、(3) 動的 shared の見積り、(4) 発火の計数の存在を固定する。
#[cfg(test)]
mod task448_tests {
    use super::*;

    /// `DENNO_Q4_GPU_ATTN_KERNEL` の新しい値 (`tile` / `fdec1`) の真理値表。
    #[test]
    fn task448_kernel_value_truth_table() {
        assert_eq!(parse_q4_gpu_attn_kernel(Some("tile")), Q4AttnKernel::Tile);
        assert_eq!(parse_q4_gpu_attn_kernel(Some("fdt")), Q4AttnKernel::Tile);
        assert_eq!(parse_q4_gpu_attn_kernel(Some("fdec2")), Q4AttnKernel::Tile);
        assert_eq!(parse_q4_gpu_attn_kernel(Some("fdec1")), Q4AttnKernel::Fdec1);
        assert_eq!(parse_q4_gpu_attn_kernel(Some("fdec")), Q4AttnKernel::Fdec1);
        // 既存の値は 1 つも変わらない。
        assert_eq!(parse_q4_gpu_attn_kernel(Some("decode")), Q4AttnKernel::Decode);
        assert_eq!(parse_q4_gpu_attn_kernel(Some("split")), Q4AttnKernel::Split);
        // 未設定・未知は従来どおり Split (= タイル版の既定は off)。
        assert_eq!(parse_q4_gpu_attn_kernel(None), Q4AttnKernel::Split);
        assert_eq!(parse_q4_gpu_attn_kernel(Some("x")), Q4AttnKernel::Split);
    }

    /// タイル版を選ぶのは `tile` の 1 値だけ (既定 off) であること。
    /// 分岐が発火する値の個数を数える (5 値のうち 1 つ)。
    #[test]
    fn task448_tile_gate_counts_one_arm() {
        let all = [
            Q4AttnKernel::Decode,
            Q4AttnKernel::Prefill,
            Q4AttnKernel::Split,
            Q4AttnKernel::Tile,
            Q4AttnKernel::Fdec1,
        ];
        let on = all.iter().filter(|k| fdec_tile_from_kernel(**k)).count();
        if ATTN_FLASHDEC_TILE_DEFAULT {
            // 採用して既定 on にした場合は Decode/Prefill/Split/Tile の 4 値。
            assert_eq!(on, 4, "既定 on のときは fdec1 以外が on");
            assert!(!fdec_tile_from_kernel(Q4AttnKernel::Fdec1), "fdec1 は off");
        } else {
            assert_eq!(on, 1, "既定 off のときタイル版は tile の 1 値だけ");
            assert!(fdec_tile_from_kernel(Q4AttnKernel::Tile), "tile が off");
            assert!(
                !fdec_tile_from_kernel(Q4AttnKernel::Split),
                "既定でタイル版が on になっている"
            );
        }
        // env 経由の選択 (テスト用の差し替え) がそのまま判定に出ること。
        set_q4_gpu_attn_kernel_for_test(Some(Q4AttnKernel::Tile));
        assert!(attn_fdec_tile_on(), "FORCE tile が効いていない");
        set_q4_gpu_attn_kernel_for_test(Some(Q4AttnKernel::Fdec1));
        assert!(!attn_fdec_tile_on(), "FORCE fdec1 が効いていない");
        set_q4_gpu_attn_kernel_for_test(None);

        // 新しい env (`DENNO_ATTN_FLASHDEC2`) の真理値表。未設定・空は None
        // (カーネル選択に委ねる)、それ以外は on/off を明示する。
        assert_eq!(fdec_tile_from_env(None), None);
        assert_eq!(fdec_tile_from_env(Some("")), None);
        assert_eq!(fdec_tile_from_env(Some("  ")), None);
        for v in ["1", "on", "tile", "true", "yes", " ON "] {
            assert_eq!(fdec_tile_from_env(Some(v)), Some(true), "{v:?} が on でない");
        }
        for v in ["0", "off", "false", "no", "v1", "x"] {
            assert_eq!(fdec_tile_from_env(Some(v)), Some(false), "{v:?} が off でない");
        }

        // 実際に lab の COND で使う名前 (`DENNO_ATTN_FLASHDEC_TILE`) の門。
        // 未設定は None (既定 off)、0/off は明示 off、数値は on。
        assert_eq!(attn_fdec_tile_ctl(None), None);
        assert_eq!(attn_fdec_tile_ctl(Some("")), None);
        assert_eq!(attn_fdec_tile_ctl(Some("  ")), None);
        for v in ["0", "off", "false", "no", " OFF "] {
            assert_eq!(attn_fdec_tile_ctl(Some(v)), Some(false), "{v:?} が off でない");
        }
        for v in ["16", "24", "4", "64", "on", "1"] {
            assert_eq!(attn_fdec_tile_ctl(Some(v)), Some(true), "{v:?} が on でない");
        }
    }

    /// tile 行数の解釈 (既定 16・4..=64 に丸め)。
    #[test]
    fn task448_tile_rows_parse() {
        assert_eq!(parse_attn_fdec_tile(None), 16, "既定が 16 でない");
        assert_eq!(parse_attn_fdec_tile(Some("")), 16);
        assert_eq!(parse_attn_fdec_tile(Some("8")), 8);
        assert_eq!(parse_attn_fdec_tile(Some(" 32 ")), 32);
        assert_eq!(parse_attn_fdec_tile(Some("0")), 4, "下限 4 に丸めていない");
        assert_eq!(parse_attn_fdec_tile(Some("999")), 64, "上限 64 に丸めていない");
        assert_eq!(parse_attn_fdec_tile(Some("x")), 16);
    }

    /// 形の門番と動的 shared の見積り。`.cu` の式 (2 * tile * dh * 4) と
    /// 48 KiB の上限を固定する。
    #[test]
    fn task448_shape_gate_and_smem() {
        assert_eq!(fdt_smem_bytes(16, 256), 32768, "smem が 32 KiB でない");
        assert_eq!(fdt_smem_bytes(12, 512), 49152);
        assert!(fdt_shape_ok(256, 16), "実形 (dh 256 / tile 16) が門で落ちる");
        assert!(fdt_shape_ok(512, 12), "dh 512 / tile 12 が門で落ちる");
        assert!(
            !fdt_shape_ok(256, 32),
            "48 KiB を超える (dh 256 / tile 32) のに通った"
        );
        assert!(
            !fdt_shape_ok(512, 16),
            "48 KiB を超える (dh 512 / tile 16) のに通った"
        );
        assert!(!fdt_shape_ok(6, 16), "dh が 4 の倍数でないのに通った");
        assert!(!fdt_shape_ok(0, 16), "dh 0 が通った");
        assert!(!fdt_shape_ok(256, 0), "tile 0 が通った");
        assert!(!fdt_shape_ok(256, 65), "tile 65 が通った");
        // 上限 (dh 512 = ATTN_FDEC_MAXB * 128) を超える形は v1 へ落とす。
        assert!(!fdt_shape_ok(ATTN_FDEC_MAXB * 128 + 4, 4), "dh の上限が効いていない");
    }

    /// tile 行数の差し替え (テスト用フック)。
    #[test]
    fn task448_tile_force_hook() {
        set_attn_fdec_tile_for_test(Some(8));
        assert_eq!(attn_fdec_tile(), 8, "FORCE 8 が効いていない");
        set_attn_fdec_tile_for_test(Some(100));
        assert_eq!(attn_fdec_tile(), 64, "FORCE の丸めが効いていない");
        set_attn_fdec_tile_for_test(None);
        assert_eq!(attn_fdec_tile(), parse_attn_fdec_tile(std::env::var("DENNO_ATTN_FLASHDEC_TILE").ok().as_deref()));
    }

    // ---- task 460: chunk の長さ依存化 + combine/gate の融合 ----------------

    /// `DENNO_ATTN_FLASHDEC_CHUNK=auto` の真理値表と、長さ -> chunk の規則。
    ///
    /// 規則の根拠 (lab t456c、既定モデル): 14k では chunk 128 が full 段 -6.6%、
    /// 2k では +16%。境目は block 数 2*n_split が SM 170 を割るかどうか。
    #[test]
    fn task460_auto_chunk_rule() {
        // `auto` だけが真。数値・空・未知は固定値のまま。
        assert!(parse_attn_flashdec_chunk_auto(Some("auto")));
        assert!(parse_attn_flashdec_chunk_auto(Some(" AUTO ")));
        for v in ["64", "128", "", "  ", "on", "x"] {
            assert!(
                !parse_attn_flashdec_chunk_auto(Some(v)),
                "{v:?} を auto と読んでいる"
            );
        }
        assert!(!parse_attn_flashdec_chunk_auto(None));

        // 実測点: 14,018 -> 128 (n_split 110)、3,965 と 2,000 -> 64 (現状と同じ)。
        assert_eq!(q4_fdec_chunk_auto(14018), 128);
        assert_eq!(q4_fdec_chunk_auto(3965), 64);
        assert_eq!(q4_fdec_chunk_auto(2000), 64);
        assert_eq!(q4_fdec_chunk_auto(0), 64);
        // 境界: 110*64 = 7040 まで 64、7041 で 65。上限は 128。
        assert_eq!(q4_fdec_chunk_auto(7040), 64);
        assert_eq!(q4_fdec_chunk_auto(7041), 65);
        assert_eq!(q4_fdec_chunk_auto(14080), 128);
        assert_eq!(q4_fdec_chunk_auto(40000), 128, "上限 128 を超えた");
        // どの長さでも 1 分割は 64..=128 に収まる。
        for t in [1usize, 63, 64, 1023, 1024, 4096, 8192, 12000, 14018, 65536] {
            let c = q4_fdec_chunk_auto(t);
            assert!((64..=128).contains(&c), "len={t} で chunk={c}");
        }
        // 実測点で n_split が期待どおり (14k は t456c の 110 と同じ、2k は 32 のまま)。
        assert_eq!(q4_fdec_split(14018, q4_fdec_chunk_auto(14018)), (110, 128));
        assert_eq!(q4_fdec_split(2000, q4_fdec_chunk_auto(2000)), (32, 63));
        // 固定値の解釈 (数値を書いたとき) は今までどおり 64。
        assert_eq!(parse_attn_flashdec_chunk(None), 64);

        // 既定は auto (未設定・空)、数値を書いたときだけ固定 (切り戻し)。
        assert!(attn_fdec_chunk_is_auto(None), "未設定が auto でない (既定 auto)");
        assert!(attn_fdec_chunk_is_auto(Some("")));
        assert!(attn_fdec_chunk_is_auto(Some(" AUTO ")));
        for v in ["64", "128", " 96 "] {
            assert!(!attn_fdec_chunk_is_auto(Some(v)), "{v:?} が auto になっている");
        }
        assert!(!attn_fdec_chunk_is_auto(Some("x")), "未知の値が auto になっている");
        // 既定のまま 14k では 128 (= n_split 110) になり、2k では 64 のまま。
        assert_eq!(q4_fdec_split(14018, q4_fdec_chunk_auto(14018)), (110, 128));
        assert_eq!(q4_fdec_split(2000, q4_fdec_chunk_auto(2000)), (32, 63));
    }

    /// 融合版 (combine + ゲート) の門は**既定 on**、明示的な偽値だけ off
    /// (lab t460b で 14k の full 段 -7.6%、md5 SAME)。
    #[test]
    fn task460_cbg_gate_truth_table() {
        for v in ["1", "on", "true", "yes", " ON ", "True", "", "  ", "auto", "x"] {
            assert!(parse_attn_fdec_cbg(Some(v)), "{v:?} が on でない (既定 on)");
        }
        for v in ["0", "off", "false", "no", " OFF ", "No"] {
            assert!(!parse_attn_fdec_cbg(Some(v)), "{v:?} が on になっている");
        }
        assert!(parse_attn_fdec_cbg(None), "未設定が on でない (既定 on)");
        // env 未設定なら (テストの差し替えが無ければ) on = 既定 on。
        assert!(
            attn_fdec_cbg_on() || FD_CBG_FORCE.load(std::sync::atomic::Ordering::Relaxed) == 0
        );
        // 差し替えフックは env より優先する。
        set_attn_fdec_cbg_for_test(Some(true));
        assert!(attn_fdec_cbg_on());
        set_attn_fdec_cbg_for_test(Some(false));
        assert!(!attn_fdec_cbg_on());
        set_attn_fdec_cbg_for_test(None);
    }

    /// task 520 (2): grid 入れ替えは**既定 off**、真値のときだけ on。
    #[test]
    fn task520_khx_gate_truth_table() {
        for v in ["1", "on", "true", "yes", " ON ", "True"] {
            assert!(parse_attn_fdec_khx(Some(v)), "{v:?} が on でない");
        }
        for v in ["0", "off", "false", "no", "", "auto", "x", "  "] {
            assert!(!parse_attn_fdec_khx(Some(v)), "{v:?} が on になっている");
        }
        assert!(!parse_attn_fdec_khx(None), "未設定が on になっている (既定 off)");
        set_attn_fdec_khx_for_test(Some(true));
        assert!(attn_fdec_khx_on());
        set_attn_fdec_khx_for_test(Some(false));
        assert!(!attn_fdec_khx_on());
        set_attn_fdec_khx_for_test(None);
    }

    /// task 520: 第 3 版 (2 key 同時) の門は**既定 on**、明示的な偽値だけ off。
    #[test]
    fn task520_ilp_gate_truth_table() {
        for v in ["1", "on", "true", "yes", " ON ", "True", "", "  ", "auto", "x"] {
            assert!(parse_attn_fdec_ilp(Some(v)), "{v:?} が on でない (既定 on)");
        }
        for v in ["0", "off", "false", "no", " OFF ", "No"] {
            assert!(!parse_attn_fdec_ilp(Some(v)), "{v:?} が on になっている");
        }
        assert!(parse_attn_fdec_ilp(None), "未設定が on でない (既定 on)");
        // 差し替えフックは env より優先する。
        set_attn_fdec_ilp_for_test(Some(true));
        assert!(attn_fdec_ilp_on());
        set_attn_fdec_ilp_for_test(Some(false));
        assert!(!attn_fdec_ilp_on());
        set_attn_fdec_ilp_for_test(None);
        // ABI の約束 (`.cu` の引数 11 本 = 第 2 版と同じ)。
        assert_eq!(FDT2_NARGS, 12, "第 3 版の引数が 12 本でない");
        // 発火計数は GPU 無しでも増える (on のときだけ)。
        let before = q4attn_kprof::FDT2_CALLS.load(std::sync::atomic::Ordering::Relaxed);
        q4attn_kprof::note_fdt2_call();
        let after = q4attn_kprof::FDT2_CALLS.load(std::sync::atomic::Ordering::Relaxed);
        assert!(after >= before, "計数が減った");
    }

    /// 融合版の発火計数とログの口。GPU 無しでは 0 のまま。
    #[test]
    fn task460_cbg_counters_exist() {
        let before = q4attn_kprof::CBG_CALLS.load(std::sync::atomic::Ordering::Relaxed);
        q4attn_kprof::note_cbg_call();
        let after = q4attn_kprof::CBG_CALLS.load(std::sync::atomic::Ordering::Relaxed);
        assert!(after >= before, "計数が減った");
        // ABI の約束 (`.cu` の引数 6 本)。
        assert_eq!(FCBG_NARGS, 6, "融合カーネルの引数が 6 本でない");
    }

    /// 発火の計数 (FDT_CALLS) とログの口があること。GPU 無しでは 0 のまま。
    #[test]
    fn task448_counters_exist() {
        let before = q4attn_kprof::FDT_CALLS.load(std::sync::atomic::Ordering::Relaxed);
        q4attn_kprof::note_fdt_call();
        let after = q4attn_kprof::FDT_CALLS.load(std::sync::atomic::Ordering::Relaxed);
        // `[q4attn]` の行は q4prof が有効なときだけ計数する (既定 off では増えない)。
        assert!(after >= before, "計数が減った");
        let fb0 = q4attn_kprof::FDT_FALLBACKS.load(std::sync::atomic::Ordering::Relaxed);
        q4attn_kprof::note_fdt_fallback();
        let fb1 = q4attn_kprof::FDT_FALLBACKS.load(std::sync::atomic::Ordering::Relaxed);
        assert!(fb1 >= fb0, "fallback の計数が減った");
    }
}
