// task 668: `mod tests` の後半 (`mod h2`)。`tests.rs` から `include!` で取り込む
// (include! はモジュール本体に差し込むので、内側ドキュメント `//!` は使えない)。

    mod h2 {
        use super::*;
        use std::sync::atomic::Ordering;

        // 次元 (tests/qwen4exp_fwd.rs の合成モデルと同じ値。目録の下限を満たす最小)
        const E: usize = 256;
        const VOCAB: usize = 8;
        const NHEAD: usize = 2;
        const HEAD: usize = 256;
        const NKV: usize = 1;
        const NEXP: usize = 2;
        const FF: usize = 256;
        const HC: usize = 2;
        const HC_LR: usize = 32;
        const SSM_CONV: usize = 4;
        const SSM_STATE: usize = 32;
        const SSM_GROUP: usize = 2;
        const SSM_DT: usize = 2;
        const SSM_INNER: usize = 64;
        const INTERVAL: usize = 4;
        const IDX_HEAD: usize = 2;
        const IDX_DIM: usize = 32;
        const IDX_TOP_K: usize = 8;
        const RATIO: u32 = 2;
        const PLE_HEADS_PER_NGRAM: usize = 2;
        const EPL: usize = 64;
        const PLE_LAYER: usize = 1;
        const PLE_NGRAM: usize = 3;
        const PLE_CONV: usize = 4;
        const PLE_ROWS: u64 = 40;
        const LAYERS: usize = 4;

        const GT_F32: u32 = 6;
        const GT_ARRAY: u32 = 9;
        const GT_U64: u32 = 10;

        enum V {
            U32(u32),
            F32(f32),
            S(&'static str),
            A32(Vec<u32>),
            A64(Vec<u64>),
            AStr(Vec<String>),
        }

        fn push_str(b: &mut Vec<u8>, s: &str) {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
        }

        /// 決定的な擬似乱数バイト (0..=255)。
        fn pat(i: usize, seed: u32) -> u8 {
            let mut x = (i as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(seed.wrapping_mul(40_503));
            x ^= x >> 13;
            x = x.wrapping_mul(0x9E37_79B1);
            x ^= x >> 16;
            x as u8
        }

        fn write_meta(b: &mut Vec<u8>, k: &str, v: &V) {
            push_str(b, k);
            match v {
                V::U32(x) => {
                    b.extend_from_slice(&GT_U32.to_le_bytes());
                    b.extend_from_slice(&x.to_le_bytes());
                }
                V::F32(x) => {
                    b.extend_from_slice(&GT_F32.to_le_bytes());
                    b.extend_from_slice(&x.to_le_bytes());
                }
                V::S(s) => {
                    b.extend_from_slice(&GT_STRING.to_le_bytes());
                    push_str(b, s);
                }
                V::A32(v) => {
                    b.extend_from_slice(&GT_ARRAY.to_le_bytes());
                    b.extend_from_slice(&GT_U32.to_le_bytes());
                    b.extend_from_slice(&(v.len() as u64).to_le_bytes());
                    for x in v {
                        b.extend_from_slice(&x.to_le_bytes());
                    }
                }
                V::A64(v) => {
                    b.extend_from_slice(&GT_ARRAY.to_le_bytes());
                    b.extend_from_slice(&GT_U64.to_le_bytes());
                    b.extend_from_slice(&(v.len() as u64).to_le_bytes());
                    for x in v {
                        b.extend_from_slice(&x.to_le_bytes());
                    }
                }
                V::AStr(v) => {
                    b.extend_from_slice(&GT_ARRAY.to_le_bytes());
                    b.extend_from_slice(&GT_STRING.to_le_bytes());
                    b.extend_from_slice(&(v.len() as u64).to_le_bytes());
                    for s in v {
                        push_str(b, s);
                    }
                }
            }
        }

        fn kvs() -> Vec<(&'static str, V)> {
            let mut ratios = vec![0u32; LAYERS];
            for (il, r) in ratios.iter_mut().enumerate() {
                if (il + 1) % INTERVAL == 0 {
                    *r = RATIO;
                }
            }
            vec![
                ("general.architecture", V::S("qwen4exp")),
                ("general.alignment", V::U32(ALIGN as u32)),
                ("qwen4exp.block_count", V::U32(LAYERS as u32)),
                ("qwen4exp.context_length", V::U32(1024)),
                ("qwen4exp.embedding_length", V::U32(E as u32)),
                ("qwen4exp.attention.head_count", V::U32(NHEAD as u32)),
                ("qwen4exp.attention.head_count_kv", V::U32(NKV as u32)),
                ("qwen4exp.attention.key_length", V::U32(HEAD as u32)),
                ("qwen4exp.attention.value_length", V::U32(HEAD as u32)),
                ("qwen4exp.attention.layer_norm_rms_epsilon", V::F32(1e-6)),
                ("qwen4exp.rope.dimension_count", V::U32(4)),
                ("qwen4exp.rope.dimension_sections", V::A32(vec![2, 2, 2, 0])),
                ("qwen4exp.rope.freq_base", V::F32(1e7)),
                ("qwen4exp.expert_count", V::U32(NEXP as u32)),
                // 2 expert を両方選ばせる: MoE のバッチ経路 (同じ expert を選んだ
                // トークンをまとめる) を実際に通すため。1 だと 1 トークン 1 expert で
                // 逐次版と同じ入口に落ち、まとめる経路が一度も走らない。
                ("qwen4exp.expert_used_count", V::U32(NEXP as u32)),
                ("qwen4exp.expert_feed_forward_length", V::U32(FF as u32)),
                (
                    "qwen4exp.expert_shared_feed_forward_length",
                    V::U32(FF as u32),
                ),
                ("qwen4exp.ssm.conv_kernel", V::U32(SSM_CONV as u32)),
                ("qwen4exp.ssm.state_size", V::U32(SSM_STATE as u32)),
                ("qwen4exp.ssm.group_count", V::U32(SSM_GROUP as u32)),
                ("qwen4exp.ssm.time_step_rank", V::U32(SSM_DT as u32)),
                ("qwen4exp.ssm.inner_size", V::U32(SSM_INNER as u32)),
                ("qwen4exp.full_attention_interval", V::U32(INTERVAL as u32)),
                ("qwen4exp.hyper_connection.count", V::U32(HC as u32)),
                ("qwen4exp.hyper_connection.low_rank", V::U32(HC_LR as u32)),
                ("qwen4exp.attention.indexer.head_count", V::U32(IDX_HEAD as u32)),
                ("qwen4exp.attention.indexer.key_length", V::U32(IDX_DIM as u32)),
                ("qwen4exp.attention.indexer.top_k", V::U32(IDX_TOP_K as u32)),
                ("qwen4exp.attention.compress_ratios", V::A32(ratios)),
                ("qwen4exp.embedding_length_per_layer_input", V::U32(EPL as u32)),
                ("qwen4exp.ple.layers", V::A32(vec![PLE_LAYER as u32])),
                ("qwen4exp.ple.ngram_size", V::U32(PLE_NGRAM as u32)),
                (
                    "qwen4exp.ple.heads_per_ngram",
                    V::U32(PLE_HEADS_PER_NGRAM as u32),
                ),
                ("qwen4exp.ple.conv_kernel", V::U32(PLE_CONV as u32)),
                ("qwen4exp.ple.eos_token_id", V::U32(7)),
                ("qwen4exp.ple.image_token_id", V::U32(9)),
                ("qwen4exp.ple.layer_multipliers", V::A64(vec![1, 2, 3])),
                ("qwen4exp.ple.head_offsets", V::A64(vec![0, 10, 20, 30])),
                ("qwen4exp.ple.head_vocab_sizes", V::A64(vec![10, 10, 10, 10])),
                ("tokenizer.ggml.tokens", V::AStr(vec![String::new(); VOCAB])),
            ]
        }

        /// 最小 GGUF (v3) を書く。データ部は 32 B 整列。
        fn write_model(path: &Path, tensors: &[(String, Vec<u64>, u32, Vec<u8>)]) {
            let ks = kvs();
            let mut head = Vec::new();
            head.extend_from_slice(b"GGUF");
            head.extend_from_slice(&3u32.to_le_bytes());
            head.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
            head.extend_from_slice(&(ks.len() as u64).to_le_bytes());
            for (k, v) in &ks {
                write_meta(&mut head, k, v);
            }

            let mut offs = Vec::with_capacity(tensors.len());
            let mut cur = 0usize;
            for (_, _, _, data) in tensors {
                offs.push(cur as u64);
                cur = align_up(cur + data.len(), ALIGN);
            }
            for ((name, dims, ty, _), off) in tensors.iter().zip(&offs) {
                push_str(&mut head, name);
                head.extend_from_slice(&(dims.len() as u32).to_le_bytes());
                for d in dims {
                    head.extend_from_slice(&d.to_le_bytes());
                }
                head.extend_from_slice(&ty.to_le_bytes());
                head.extend_from_slice(&off.to_le_bytes());
            }

            let mut out = head;
            out.resize(align_up(out.len(), ALIGN), 0);
            for (_, _, _, data) in tensors {
                out.extend_from_slice(data);
                out.resize(align_up(out.len(), ALIGN), 0);
            }
            std::fs::write(path, out).expect("合成 GGUF の書き出し");
        }

        /// 目録の量子型 → ggml_type id。
        fn ggml_id(ty: GgmlType) -> u32 {
            match ty {
                GgmlType::F32 => 0,
                GgmlType::F16 => 1,
                GgmlType::Q5_1 => 7,
                GgmlType::Q8_0 => 8,
                GgmlType::Q4K => 12,
                GgmlType::Q5K => 13,
                GgmlType::IQ4Nl => 20,
                GgmlType::Bf16 => 30,
                other => panic!("テストヘルパ未対応の型 {other:?}"),
            }
        }

        /// テンソルの中身。**有限値だけ**を作る (スケール欄を乱数にすると NaN になる)。
        fn tensor_bytes(name: &str, dims: &[u64], ty: GgmlType, seed: u32) -> Vec<u8> {
            let ne: u64 = dims.iter().product();
            let bs = ty.blck_size() as u64;
            let ts = ty.type_size() as u64;
            let nb = (ne / bs * ts + if ne % bs == 0 { 0 } else { ts }) as usize;
            let mut b: Vec<u8> = (0..nb).map(|i| pat(i, 7 + seed)).collect();
            match ty {
                GgmlType::F32 => {
                    // norm の gamma は 1.0 (変換器が 1 + w に畳んだ前提の値)
                    let gamma = name.ends_with("norm.weight");
                    for i in 0..nb / 4 {
                        let v = if gamma {
                            1.0f32
                        } else {
                            (((i as u32).wrapping_add(seed * 5) % 17) as f32 - 8.0) * 0.01
                        };
                        b[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
                    }
                }
                GgmlType::Bf16 => {
                    for i in 0..nb / 2 {
                        let v = (((i as u32).wrapping_add(seed * 3) % 13) as f32 - 6.0) * 0.01;
                        let bits = (v.to_bits() >> 16) as u16;
                        b[i * 2..i * 2 + 2].copy_from_slice(&bits.to_le_bytes());
                    }
                }
                GgmlType::Q8_0 => {
                    for blk in 0..nb / 34 {
                        let o = blk * 34;
                        b[o..o + 2].copy_from_slice(&f16_bytes(0.02));
                        for j in 0..32u32 {
                            // ブロックごとに量子値をずらす (行ごとに違う値になる)
                            let q = ((j as i32 + blk as i32 * 3 + seed as i32 * 7) % 11) - 5;
                            b[o + 2 + j as usize] = q as i8 as u8;
                        }
                    }
                }
                GgmlType::Q5_1 => {
                    for blk in 0..nb / 24 {
                        let o = blk * 24;
                        b[o..o + 2].copy_from_slice(&f16_bytes(0.02));
                        b[o + 2..o + 4].copy_from_slice(&f16_bytes(0.001));
                    }
                }
                GgmlType::Q4K => {
                    for blk in 0..nb / 144 {
                        let o = blk * 144;
                        b[o..o + 2].copy_from_slice(&f16_bytes(0.02));
                        b[o + 2..o + 4].copy_from_slice(&f16_bytes(0.001));
                    }
                }
                GgmlType::Q5K => {
                    // 176 B = d(2) + dmin(2) + scales(12) + qh(32) + qs(128)
                    for blk in 0..nb / 176 {
                        let o = blk * 176;
                        b[o..o + 2].copy_from_slice(&f16_bytes(0.02));
                        b[o + 2..o + 4].copy_from_slice(&f16_bytes(0.001));
                    }
                }
                GgmlType::IQ4Nl => {
                    for blk in 0..nb / 18 {
                        let o = blk * 18;
                        b[o..o + 2].copy_from_slice(&f16_bytes(0.02));
                    }
                }
                other => panic!("テストヘルパ未対応の型 {other:?}"),
            }
            b
        }

        /// メタデータだけの GGUF から params を取って、目録の全テンソルを
        /// 決定的な中身で埋めた GGUF を書く。
        fn build(dir: &TempDir) -> PathBuf {
            let meta = dir.path("h2_meta.gguf");
            write_model(&meta, &[]);
            let f = GgufFile::open(&meta).expect("メタデータ GGUF を開けない");
            let p = Qwen4ExpParams::from_meta(&f).expect("from_meta が通らない");
            let specs = expected_tensors(&p).expect("目録を組めない");
            let mut tensors = Vec::with_capacity(specs.len());
            for (i, e) in specs.iter().enumerate() {
                let data = tensor_bytes(&e.name, &e.dims, e.ty, i as u32 + 1);
                tensors.push((e.name.clone(), e.dims.clone(), ggml_id(e.ty), data));
            }
            let path = dir.path("h2_model.gguf");
            write_model(&path, &tensors);
            path
        }

        /// 1 回ぶんの prefill を新しいモデルで回し、logits と
        /// 「バッチ分岐 / バッチ GEMV が何回走ったか」を返す。
        fn run(f: &GgufFile, ctx: &[u32], batch: bool, chunk: usize) -> (Vec<f32>, usize, usize) {
            let mut m = Qwen4Exp::new(f).expect("Qwen4Exp::new");
            let b0 = PREFILL_BATCH_CALLS.load(Ordering::Relaxed);
            let g0 = GEMV_BATCH_CALLS.load(Ordering::Relaxed);
            let out = if batch {
                m.forward_prefill_batch(ctx, chunk).expect("バッチ prefill")
            } else {
                m.forward_prefill_with(ctx, false).expect("逐次 prefill")
            };
            (
                out,
                PREFILL_BATCH_CALLS.load(Ordering::Relaxed) - b0,
                GEMV_BATCH_CALLS.load(Ordering::Relaxed) - g0,
            )
        }


        /// task 862: page hint の追跡が **prefill の実経路** から呼ばれているか。
        ///
        /// ゲートを足しただけ (緑だが分岐が無い) を防ぐため、prefill を 1 回回して
        /// 追跡したトークン数と選択の本数を数える。ここは配線の確認だけで、
        /// 落とす / 戻すの効果は `page_hint_drops_cold_expert_pages` が見る。
        #[test]
        fn page_hint_hook_is_wired_into_prefill() {
            use crate::model::exp_page_hint as ph;
            let _g = moe_test_guard();
            let dir = TempDir::new("h2_page_hint");
            let path = build(&dir);
            let f = GgufFile::open(&path).expect("open");
            let mut m = Qwen4Exp::new(&f).expect("Qwen4Exp::new");
            let ctx: Vec<u32> = (0..8u32).map(|i| i % VOCAB as u32).collect();
            ph::reset_for_tests();
            ph::set_enabled_for_tests(Some(true));
            ph::begin_prefill(4096);
            m.forward_prefill_batch(&ctx, 4).expect("バッチ prefill");
            let tok = ph::tracked_tokens();
            let _ = ph::take_plan();
            ph::set_enabled_for_tests(None);
            ph::reset_for_tests();
            assert_eq!(tok, ctx.len(), "prefill の全トークンが層 0 で数えられる");
        }

        /// task 874: dequant 再利用キャッシュの配線。実 prefill 経路
        /// (`forward_prefill_with` → `forward_prefill_inner`) から
        /// `begin_prefill` / `finish_prefill` が呼ばれ、展開が走ることを数える。
        /// 緑 = 発動の証拠ではないので、`fills` が 0 でないことまで見る。
        #[test]
        fn pf_dq_cache_hook_fires_in_prefill() {
            use crate::model::qwen4exp::pf_dq_cache as dqc;
            let _g = moe_test_guard();
            let dir = TempDir::new("h2_dq_cache_wired");
            let path = build(&dir);
            let f = GgufFile::open(&path).expect("open");
            let ctx: Vec<u32> = (0..8u32).map(|i| i % VOCAB as u32).collect();
            // VNNI (int8 活性) を切って比べる。切らないと基準側が活性を int8 へ
            // 量子化するぶん (既存テストの許容 5e-2/層) の差が乗り、
            // 累積順の差と区別できない。
            moe_expert::set_vnni_for_test(Some(false));
            // off (既定)
            let mut m0 = Qwen4Exp::new(&f).expect("Qwen4Exp::new");
            let off = m0.forward_prefill_with(&ctx, true).expect("prefill off");
            // on
            dqc::set_stats_for_tests(true);
            dqc::set_cap_for_tests(16 << 20);
            let mut m1 = Qwen4Exp::new(&f).expect("Qwen4Exp::new");
            let on = m1.forward_prefill_with(&ctx, true).expect("prefill on");
            let line = dqc::take_line_for_tests().expect("stats 行が出る");
            dqc::clear_cap_for_tests();
            dqc::clear_stats_for_tests();
            moe_expert::set_vnni_for_test(None);
            assert!(
                line.contains("fills=") && !line.contains("fills=0 "),
                "キャッシュ経路が 1 度も走っていない: {line}"
            );
            // 値は近い (融合カーネルとの累積順の違いだけ)。
            assert_eq!(off.len(), on.len());
            let scale = off.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6);
            let max_d = off
                .iter()
                .zip(&on)
                .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
            assert!(
                max_d / scale < 1e-4,
                "max|Δ|/max|ref| = {} (off/on の差が大きすぎる)",
                max_d / scale
            );
        }

        /// task 874: チャンクを跨いだヒット。同じ expert が次のチャンクでも
        /// 選ばれるモデル (NEXP=2, k=2) で chunk=2 を回し、2 チャンク目以降が
        /// キャッシュに当たることを数える。
        #[test]
        fn pf_dq_cache_hits_across_chunks() {
            use crate::model::qwen4exp::pf_dq_cache as dqc;
            let _g = moe_test_guard();
            let dir = TempDir::new("h2_dq_cache_hits");
            let path = build(&dir);
            let f = GgufFile::open(&path).expect("open");
            let ctx: Vec<u32> = (0..8u32).map(|i| i % VOCAB as u32).collect();
            dqc::set_stats_for_tests(true);
            dqc::set_cap_for_tests(16 << 20);
            let pf = dqc::begin_prefill(ctx.len(), 4);
            let mut m = Qwen4Exp::new(&f).expect("Qwen4Exp::new");
            let _ = m.forward_prefill_batch(&ctx, 2).expect("バッチ prefill");
            drop(pf); // 終端処理 (統計行)
            let line = dqc::take_line_for_tests().expect("stats 行が出る");
            dqc::clear_cap_for_tests();
            dqc::clear_stats_for_tests();
            // 4 チャンク × (2 expert × 4 層) = 32 回の選択、distinct は 8。
            assert!(line.contains("sel=32"), "{line}");
            assert!(line.contains("distinct=8"), "{line}");
            assert!(!line.contains("hits=0 "), "跨ぎヒットが 0: {line}");
            assert!(line.contains("evict=0"), "予算内なので落ちない: {line}");
        }

        /// task 172 の 3 本は task 181 (lab t181b) で**既定 on**にした
        /// (3 本 + 小さい N + ミラー + D2H の組で cold prefill -8.9% / -9.4%)。
        #[test]
        fn pf2_dev_gates_default_on() {
            assert!(pf_full_dev_from_value(None), "未設定は on (lab t181b)");
            assert!(pf_shared_dev_from_value(None), "未設定は on (lab t181b)");
            assert!(pf_ple_gpu_from_value(None), "未設定は on (lab t181b)");
            for v in ["0", "off", "false", "no", "", " 0 ", "OFF"] {
                assert!(!pf_full_dev_from_value(Some(v)), "full: {v:?} が on になった");
                assert!(!pf_shared_dev_from_value(Some(v)), "shared: {v:?} が on になった");
                assert!(!pf_ple_gpu_from_value(Some(v)), "ple: {v:?} が on になった");
            }
            for v in ["1", "on", "true", "yes", " ON "] {
                assert!(pf_full_dev_from_value(Some(v)), "full: {v:?} が off になった");
                assert!(pf_shared_dev_from_value(Some(v)), "shared: {v:?} が off になった");
                assert!(pf_ple_gpu_from_value(Some(v)), "ple: {v:?} が off になった");
            }
        }

        /// task 200: `DENNO_PF_ATTN_FLAT` / `DENNO_PF_ATTN_DEVRES` は lab t200b で
        /// **既定 on** にした。明示 off の表が残っていることも固定する。
        #[test]
        fn pf_attn_flat_gate_values() {
            assert!(pf_attn_flat_from_value(None), "未設定は on (lab t200b)");
            assert!(pf_attn_devres_from_value(None), "未設定は on (lab t200b)");
            for v in ["0", "off", "false", "no", "", " 0 ", "OFF"] {
                assert!(!pf_attn_flat_from_value(Some(v)), "{v:?} が on になった");
                assert!(!pf_attn_devres_from_value(Some(v)), "{v:?} が on になった");
            }
            for v in ["1", "on", "true", "yes", " ON "] {
                assert!(pf_attn_flat_from_value(Some(v)), "{v:?} が off のまま");
                assert!(pf_attn_devres_from_value(Some(v)), "{v:?} が off のまま");
            }
        }

        /// task 308: `DENNO_PF_FULL_DEVRES` は full 層の出力をデバイスに残す
        /// (D2H と combine の H2D を消す)。lab t308b で採用して**既定 on**。
        #[test]
        fn pf_full_devres_gate_values() {
            assert!(pf_full_devres_from_value(None), "未設定は on");
            for v in ["0", "off", "false", "no", "", " 0 ", "OFF"] {
                assert!(!pf_full_devres_from_value(Some(v)), "{v:?} が on になった");
            }
            for v in ["1", "on", "true", "yes", " ON "] {
                assert!(pf_full_devres_from_value(Some(v)), "{v:?} が off のまま");
            }
            if std::env::var_os("DENNO_PF_FULL_DEVRES").is_none() {
                assert!(pf_full_devres_on());
            }
        }

        /// task 308: `DENNO_PF_LM_ROUTER_DEV` は層優先 MoE の router を
        /// デバイスで 1 回にまとめる。lab t308b で採用して**既定 on**。
        #[test]
        fn pf_lm_router_dev_gate_values() {
            assert!(pf_lm_router_dev_from_value(None), "未設定は on");
            for v in ["0", "off", "false", "no", "", " 0 ", "OFF"] {
                assert!(!pf_lm_router_dev_from_value(Some(v)), "{v:?} が on になった");
            }
            for v in ["1", "on", "true", "yes", " ON "] {
                assert!(pf_lm_router_dev_from_value(Some(v)), "{v:?} が off のまま");
            }
            if std::env::var_os("DENNO_PF_LM_ROUTER_DEV").is_none() {
                assert!(pf_lm_router_dev_on());
            }
        }

        /// task 183: `DENNO_PF_FULL_FUSED` は lab t183a で**既定 on**にした。
        /// 融合 qkv を 1 回の GEMM で撃つ経路の切り戻しゲート (`0` で切る)。
        #[test]
        fn pf_full_fused_gate_default_on() {
            assert!(pf_full_fused_from_value(None), "未設定は on (lab t183a で採用)");
            for v in ["0", "off", "false", "no", "", " 0 ", "OFF"] {
                assert!(!pf_full_fused_from_value(Some(v)), "{v:?} が on になった");
            }
            for v in ["1", "on", "true", "yes", " ON "] {
                assert!(pf_full_fused_from_value(Some(v)), "{v:?} が off になった");
            }
            // 上書き (テスト用) が効き、戻すと env に戻ること。
            let before = pf_full_fused_on();
            set_pf_full_fused_for_test(Some(true));
            assert!(pf_full_fused_on());
            set_pf_full_fused_for_test(Some(false));
            assert!(!pf_full_fused_on());
            set_pf_full_fused_for_test(None);
            assert_eq!(pf_full_fused_on(), before);
        }

        /// task 172: デバイス連鎖のゲートを on にしても、GPU の無いこの環境では
        /// fail closed で従来経路と**ビット一致**する。断念が数えられたことも見る
        /// (ゲートが実際に読まれた証拠)。
        ///
        /// 実モデルと同じ構成 (`SsmAttn` が `FullAttn` へ委譲する) を組む。
        #[cfg(feature = "cuda")]
        #[test]
        fn pf2_dev_chain_falls_back_without_gpu() {
            use crate::model::qwen4exp_attn::FullAttn;
            use crate::model::qwen4exp_ssm::SsmAttn;
            let _g = moe_test_guard();
            moe_expert::set_vnni_for_test(Some(false));
            let dir = TempDir::new("pf2_dev_chain");
            let path = build(&dir);
            let f = GgufFile::open(&path).expect("GGUF を開けない");
            let p = Qwen4ExpParams::from_meta(&f).expect("from_meta");
            // 1 回ぶんのバッチ prefill (毎回モデルを組み直す)。
            let run_full = |ctx: &[u32], chunk: usize| -> Vec<f32> {
                let full = FullAttn::from_gguf(&f, &p, 2048).expect("FullAttn");
                let attn = SsmAttn::with_full(&f, &p, Box::new(full)).expect("SsmAttn");
                let mut m = Qwen4Exp::with_attn(&f, Box::new(attn)).expect("モデルを組めない");
                m.forward_prefill_batch(ctx, chunk).expect("バッチ prefill")
            };
            // M = 17 (連鎖の下限未満) / 512・1,024 (実チャンク)
            for &(n, chunk) in &[(17usize, 17usize), (512, 512), (1024, 512)] {
                let ctx: Vec<u32> = (0..n).map(|i| (i % VOCAB) as u32).collect();
                set_pf2_for_test(PF2_GATE_FULL, Some(false));
                set_pf2_for_test(PF2_GATE_SHARED, Some(false));
                set_pf2_for_test(PF2_GATE_PLE, Some(false));
                let off = run_full(&ctx, chunk);
                set_pf2_for_test(PF2_GATE_FULL, Some(true));
                set_pf2_for_test(PF2_GATE_SHARED, Some(true));
                set_pf2_for_test(PF2_GATE_PLE, Some(true));
                crate::cuda::pf2_stats_reset();
                let on = run_full(&ctx, chunk);
                set_pf2_for_test(PF2_GATE_FULL, None);
                set_pf2_for_test(PF2_GATE_SHARED, None);
                set_pf2_for_test(PF2_GATE_PLE, None);
                assert_eq!(
                    off, on,
                    "M={n} chunk={chunk}: ゲート on で logits が変わった (fail closed していない)"
                );
                let st = crate::cuda::pf2_stats();
                assert!(
                    st[1] >= 1,
                    "M={n}: full 連鎖の断念が数えられていない (ゲートが読まれていない): {st:?}"
                );
                assert!(
                    st[3] >= 1,
                    "M={n}: shared 連鎖の断念が数えられていない (ゲートが読まれていない): {st:?}"
                );
            }
        }

        /// task 217: 4 つの block を全部切った層は**恒等**になる
        /// (残差に何も足さず、attn / MoE / shared / PLE を計算しない)。
        /// PLE を持つ層でも注入だけが飛ぶので、残差は入力のまま。
        #[test]
        fn intervene_all_blocks_off_makes_layer_identity() {
            use crate::model::qwen4exp_attn::FullAttn;
            use crate::model::qwen4exp_ssm::SsmAttn;
            let _g = moe_test_guard();
            let dir = TempDir::new("iv_layer_identity");
            let path = build(&dir);
            let f = GgufFile::open(&path).expect("GGUF");
            let p = Qwen4ExpParams::from_meta(&f).expect("from_meta");
            let full = FullAttn::from_gguf(&f, &p, 2048).expect("FullAttn");
            let attn = SsmAttn::with_full(&f, &p, Box::new(full)).expect("SsmAttn");
            let mut m = Qwen4Exp::with_attn(&f, Box::new(attn)).expect("model");
            let hp = m.hp();
            let ctx = [1u32, 2, 3, 1];
            let shape = intervene::Shape {
                n_layer: p.n_layer,
                n_expert: p.n_expert,
                n_expert_used: p.n_expert_used,
                n_hc: p.n_hc,
            };
            let ple = m.gather_ple(&ctx).expect("PLE");
            for il in 0..p.n_layer {
                let blocks = vec![
                    (il, IvBlock::Attn),
                    (il, IvBlock::Moe),
                    (il, IvBlock::Shared),
                    (il, IvBlock::Ple),
                ];
                let cfg = intervene::config_for_test(&shape, vec![], blocks, vec![]).expect("cfg");
                let mut res = m.forward_embed(&ctx, &hp).expect("embed");
                let before = res.data.clone();
                m.run_layer(il, &mut res, ple.as_ref(), &hp, None, None, false, IvView::of(&cfg))
                    .expect("run_layer");
                for i in 0..before.len() {
                    assert_eq!(
                        res.data[i].to_bits(),
                        before[i].to_bits(),
                        "層 {il}: 全 block を切ったのに残差の要素 {i} が変わった"
                    );
                }
            }
        }

        #[test]
        fn prefill_batch_gate_defaults_off() {
            // DENNO_PREFILL_BATCH の解釈 (既定 off)。env を触らずに純粋関数だけを見る。
            assert!(!prefill_batch_from_value(None), "未設定は既定で無効");
            assert!(prefill_batch_from_value(Some("1")));
            assert!(!prefill_batch_from_value(Some("0")));
            assert!(!prefill_batch_from_value(Some("")));
            assert!(
                !prefill_batch_from_value(Some("true")),
                "\"1\" 以外は無効 (既定 off を崩さない)"
            );
        }

        /// hc_mix のバッチ版が逐次版とビット一致する (要素ごとの演算は同じ関数)。
        #[test]
        fn hc_mix_batch_matches_seq() {
            let hp = HyperParams {
                hc: 2,
                n_embd: 8,
                low_rank: 4,
                eps: 1e-6,
            };
            let hc_dim = hp.hc * hp.n_embd;
            let norm = vec![1.0f32; hc_dim];
            let down = f32_linear(hc_dim, hp.low_rank, 0.1);
            let up = f32_linear(hp.low_rank, hc_dim, 0.2);
            let inject = f32_linear(hc_dim, hp.hc, 0.3);
            let ress: Vec<HyperStreams> = (0..3)
                .map(|t| HyperStreams {
                    data: (0..hc_dim)
                        .map(|i| (((i * 7 + t * 13) % 23) as f32 - 11.0) * 0.03)
                        .collect(),
                    hc: hp.hc,
                    n_embd: hp.n_embd,
                })
                .collect();

            let (mixed, injs) = hc_mix_quant_batch(
                &ress,
                &norm,
                &down,
                &up,
                Some(&inject),
                &hp,
                &mut MixScratch::default(),
            )
            .expect("hc_mix_quant_batch");
            for (t, r) in ress.iter().enumerate() {
                let (m, i) = hc_mix_quant(r, &norm, &down, &up, Some(&inject), &hp)
                    .expect("hc_mix_quant");
                assert_bitwise_eq(&m, &mixed[t], &format!("hc_mix mixed (token {t})"));
                assert_bitwise_eq(
                    &i,
                    &injs[t * hp.hc..(t + 1) * hp.hc],
                    &format!("hc_mix inject (token {t})"),
                );
            }
            assert!(
                mixed[0].iter().any(|v| *v != 0.0),
                "mixed が全 0 (経路の生死が分からない)"
            );
        }

        /// P86c: batch の nt == 1 は逐次 fast 経路 ([`hc_mix_quant_fast`]) に
        /// 委譲する (2 行カーネル + `HC_DOWN_ROW_PAR_MIN` の行並列 +
        /// `MIX_SCRATCH` 使い回しを持つ P84/P86 の入口そのもの)。
        ///
        /// 委譲が実際に発火したことは `MIX_FAST_CALLS` の増分で示す。batch 本体は
        /// fast を呼ばないので nt = 2 では増えないのが対照になる。出力は
        /// 逐次 [`hc_mix_quant`] と `to_bits()` 一致で、nt = 2 の batch 本体の
        /// token 0 とも一致する (gemv_batch は行ごとの独立な行列積なので、
        /// 委譲で batch 経路の値が動かないことの固定)。
        #[test]
        fn hc_mix_batch_nt1_delegates_to_seq_fast() {
            // task 126 の 1 トークン経路ゲート (既定 on) を切って、batch 本体の
            // 挙動だけを見る。
            let _g = moe_test_guard();
            set_prefill_mix_seq_for_test(Some(false));
            let (hp, norm, down, up, inject, res) = mix_q8_case(2, 32, 32);
            let make = |d: &[f32]| HyperStreams {
                data: d.to_vec(),
                hc: hp.hc,
                n_embd: hp.n_embd,
            };
            let calls = || MIX_FAST_CALLS.load(std::sync::atomic::Ordering::Relaxed);

            // nt = 1: 委譲が発火して fast がちょうど 1 回呼ばれる。
            let before = calls();
            let (mixed, injs) = hc_mix_quant_batch(
                &[make(&res.data)],
                &norm,
                &down,
                &up,
                Some(&inject),
                &hp,
                &mut MixScratch::default(),
            )
            .expect("hc_mix_quant_batch (nt=1)");
            assert_eq!(
                calls() - before,
                1,
                "nt=1 で逐次 fast 経路に委譲していない (2 行カーネルに届かない)"
            );
            assert_eq!(injs.len(), hp.hc, "nt=1 の inject は [nt][hc] = hc 本");

            // 出力は逐次版とビット一致 (委譲先は逐次 fast そのもの)。
            let (m, i) =
                hc_mix_quant(&res, &norm, &down, &up, Some(&inject), &hp).expect("hc_mix_quant");
            assert_bitwise_eq(&m, &mixed[0], "hc_mix mixed (nt=1 委譲)");
            assert_bitwise_eq(&i, &injs, "hc_mix inject (nt=1 委譲)");

            // nt >= 2 は batch 本体のまま (委譲しない)。
            let before = calls();
            let (mixed2, _) = hc_mix_quant_batch(
                &[make(&res.data), res],
                &norm,
                &down,
                &up,
                Some(&inject),
                &hp,
                &mut MixScratch::default(),
            )
            .expect("hc_mix_quant_batch (nt=2)");
            assert_eq!(calls() - before, 0, "nt=2 で委譲が発火した");
            assert_bitwise_eq(&mixed2[0], &mixed[0], "hc_mix mixed (nt=2 token 0)");
            set_prefill_mix_seq_for_test(None);
        }

        /// task 126: `DENNO_PREFILL_MIX_SEQ` / `DENNO_PREFILL_MIX_SEQ_MAX` の解釈
        /// (既定 on / 既定 64)。
        #[test]
        fn prefill_mix_seq_gate_values() {
            assert!(prefill_mix_seq_from_value(None), "未設定は既定で有効");
            assert!(prefill_mix_seq_from_value(Some("1")));
            assert!(!prefill_mix_seq_from_value(Some("0")), "0 で従来経路に戻る");
            assert!(!prefill_mix_seq_from_value(Some("off")));
            assert!(!prefill_mix_seq_from_value(Some("false")));
            assert_eq!(prefill_mix_seq_max_from_value(None), PREFILL_MIX_SEQ_MAX);
            assert_eq!(prefill_mix_seq_max_from_value(Some("32")), 32);
            assert_eq!(prefill_mix_seq_max_from_value(Some(" 16 ")), 16);
            assert_eq!(
                prefill_mix_seq_max_from_value(Some("abc")),
                PREFILL_MIX_SEQ_MAX,
                "数値でない値は既定に戻す"
            );
            assert_eq!(prefill_mix_seq_max_from_value(Some("0")), 0);
        }

        /// task 126: しきい値以下では 1 トークン経路に振れ、値は batch 経路と
        /// ビット一致する (CPU)。しきい値を下回る nt では batch 本体のまま。
        #[test]
        fn prefill_mix_seq_matches_batch() {
            // ゲートの上書きはプロセス全体で 1 個なので直列化する。
            let _g = moe_test_guard();
            let (hp, norm, down, up, inject, res) = mix_q8_case(2, 32, 32);
            let nt = 3usize;
            let ress: Vec<HyperStreams> = (0..nt)
                .map(|t| HyperStreams {
                    data: res.data.iter().map(|v| v + t as f32 * 0.01).collect(),
                    hc: hp.hc,
                    n_embd: hp.n_embd,
                })
                .collect();
            let calls = || MIX_FAST_CALLS.load(std::sync::atomic::Ordering::Relaxed);
            let run = || {
                hc_mix_quant_batch(
                    &ress,
                    &norm,
                    &down,
                    &up,
                    Some(&inject),
                    &hp,
                    &mut MixScratch::default(),
                )
                .expect("hc_mix_quant_batch")
            };

            // 基準: ゲート off = batch 本体。
            set_prefill_mix_seq_for_test(Some(false));
            let off = run();
            assert!(
                off.0.iter().flat_map(|r| r.iter()).any(|v| *v != 0.0),
                "mixed が全 0 (経路の生死が分からない)"
            );

            // ゲート on: 1 トークン経路へ振れる (fast の呼び出しが nt 回増える)。
            set_prefill_mix_seq_for_test(Some(true));
            let before = calls();
            let on = run();
            assert_eq!(
                calls() - before,
                nt,
                "1 トークン経路へ振れていない (MIX_FAST_CALLS が nt 回増えない)"
            );
            for t in 0..nt {
                assert_bitwise_eq(&off.0[t], &on.0[t], &format!("mixed token {t}"));
                assert_bitwise_eq(
                    &off.1[t * hp.hc..(t + 1) * hp.hc],
                    &on.1[t * hp.hc..(t + 1) * hp.hc],
                    &format!("inject token {t}"),
                );
            }

            // しきい値を nt 未満にすると batch 本体へ戻る。
            set_prefill_mix_seq_max_for_test(nt - 1);
            let before = calls();
            let capped = run();
            assert_eq!(
                calls() - before,
                0,
                "しきい値を超えているのに 1 トークン経路へ振れた"
            );
            for t in 0..nt {
                assert_bitwise_eq(&off.0[t], &capped.0[t], &format!("mixed (capped) token {t}"));
            }
            set_prefill_mix_seq_max_for_test(0);
            set_prefill_mix_seq_for_test(None);
        }

        /// Q8_0 のダミー重み (34 B ブロック: f16 スケール + i8 x 32)。
        /// 実ファイルの `hc_*` と同じ量子型で、`linear()` と同じく `fast_q8` を立てる。
        fn q8_0_weights(k: usize, n: usize, seed: u32) -> Vec<u8> {
            assert_eq!(k % 32, 0, "Q8_0 のブロックは 32 要素");
            let mut b = vec![0u8; k * n / 32 * 34];
            let scale = half::f16::from_f32(0.02).to_le_bytes();
            for blk in 0..b.len() / 34 {
                let o = blk * 34;
                b[o..o + 2].copy_from_slice(&scale);
                for j in 0..32usize {
                    let q = ((j as i32 + blk as i32 * 3 + seed as i32 * 7) % 11) - 5;
                    b[o + 2 + j] = q as i8 as u8;
                }
            }
            b
        }

        /// Q8_0 の `hc_down` / `hc_up` / `hc_inject` と入力 1 本。
        fn mix_q8_case(
            hc: usize,
            e: usize,
            lr: usize,
        ) -> (HyperParams, Vec<f32>, GgufLinear, GgufLinear, GgufLinear, HyperStreams) {
            let hp = HyperParams {
                hc,
                n_embd: e,
                low_rank: lr,
                eps: 1e-6,
            };
            let hc_dim = hc * e;
            let norm: Vec<f32> = (0..hc_dim).map(|i| 0.8 + (i % 7) as f32 * 0.05).collect();
            let mk = |k: usize, n: usize, seed: u32| {
                GgufLinear::from_slice(GgmlType::Q8_0, k, n, &q8_0_weights(k, n, seed))
                    .expect("GgufLinear::from_slice (Q8_0)")
                    .with_fast_q8(true)
            };
            let down = mk(hc_dim, lr, 3);
            let up = mk(lr, hc_dim, 5);
            let inject = mk(hc_dim, hc, 7);
            let res = HyperStreams {
                data: (0..hc_dim)
                    .map(|i| (((i * 11) % 17) as f32 - 8.0) * 0.05)
                    .collect(),
                hc,
                n_embd: e,
            };
            (hp, norm, down, up, inject, res)
        }

        /// task 110: BF16 版の hc mix フィクスチャ (`mix_q8_case` と同じ形・
        /// 実 decode の down/up 実型)。重みは既存の [`bf16_weights`] (pat4 ベース
        /// の決定的バイト列)。
        fn mix_bf16_case(
            hc: usize,
            e: usize,
            lr: usize,
        ) -> (HyperParams, Vec<f32>, GgufLinear, GgufLinear, GgufLinear, HyperStreams) {
            let hp = HyperParams {
                hc,
                n_embd: e,
                low_rank: lr,
                eps: 1e-6,
            };
            let hc_dim = hc * e;
            let norm: Vec<f32> = (0..hc_dim).map(|i| 0.8 + (i % 7) as f32 * 0.05).collect();
            let mk = |k: usize, n: usize, seed: u32| {
                GgufLinear::from_slice(GgmlType::Bf16, k, n, &bf16_weights(k, n, seed))
                    .expect("GgufLinear::from_slice (Bf16)")
            };
            let down = mk(hc_dim, lr, 3);
            let up = mk(lr, hc_dim, 5);
            let inject = mk(hc_dim, hc, 7);
            let res = HyperStreams {
                data: (0..hc_dim)
                    .map(|i| (((i * 11) % 17) as f32 - 8.0) * 0.05)
                    .collect(),
                hc,
                n_embd: e,
            };
            (hp, norm, down, up, inject, res)
        }

        /// IQ4_XS の 1 ブロック (136 B = d f16 + scales_h u16 + scales_l u8x4 + qs u8x128)。
        ///
        /// dequant は任意のバイト列で決定的 (KVALUES_IQ4NL の 4bit テーブル参照)。
        /// moe_expert のテストと同じ形 (スケール類だけ有限値に固定)。
        fn iq4_xs_block(seed: u32) -> Vec<u8> {
            let ts = GgmlType::IQ4Xs.type_size();
            let mut b = vec![0u8; ts];
            b[0..2].copy_from_slice(&half::f16::from_f32(0.01).to_le_bytes());
            let sh = (pat4(0, seed) as u16) | (((pat4(1, seed) as u16) & 0x3) << 8);
            b[2..4].copy_from_slice(&sh.to_le_bytes());
            for i in 4..ts {
                b[i] = pat4(i, seed);
            }
            b
        }

        /// 決定的バイト列 (0..=255)。moe_expert の `pat` と同じ式。
        fn pat4(i: usize, seed: u32) -> u8 {
            let mut x = (i as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(seed.wrapping_mul(40_503));
            x ^= x >> 13;
            x = x.wrapping_mul(0x9E37_79B1);
            x ^= x >> 16;
            x as u8
        }

        /// IQ4_XS の重み行列 (行優先の生バイト)。`k` は 256 の倍数。
        fn iq4_xs_weights(k: usize, n: usize, seed: u32) -> Vec<u8> {
            assert_eq!(k % 256, 0, "IQ4_XS のブロックは 256 要素");
            let nblk = k / 256;
            let mut v = Vec::new();
            for r in 0..n {
                for blk in 0..nblk {
                    v.extend_from_slice(&iq4_xs_block((seed as usize * 10_000 + r * 100 + blk) as u32));
                }
            }
            v
        }

        /// BF16 のダミー重み (1 要素 2 B = f32 の上位 16bit)。
        ///
        /// linear_gguf の `make_block` (Bf16 アーム) と同じ式 (`pat4` から f32 を作り
        /// 上位 16bit に落とす)。dequant は bit 複製 (`f32::from_bits(u16 << 16)`) で
        /// 丸めが入らないので、書いた値がそのまま f32 として読み出る。
        fn bf16_weights(k: usize, n: usize, seed: u32) -> Vec<u8> {
            let mut b = vec![0u8; k * n * 2];
            for (i, o) in b.chunks_exact_mut(2).enumerate() {
                let v = ((pat4(i, seed) as f32) / 128.0) - 1.0;
                let bits = (v.to_bits() >> 16) as u16;
                o.copy_from_slice(&bits.to_le_bytes());
            }
            b
        }

        /// BF16 の逐次参照 (`GgufLinear::gemv*` のどの経路も通らない独立実装)。
        ///
        /// 行ごとに `dequant_row_into` して `acc += row[i] * x[i]` を i の昇順に
        /// 加算する (`gemv_scalar_body` と同じ演算列)。行並列の有無で動いては
        /// いけない基準値。
        fn bf16_gemv_seq(lin: &GgufLinear, x: &[f32]) -> Vec<f32> {
            let (k, n) = (lin.k(), lin.n());
            let rb = k * 2; // BF16 は 1 要素 2 B
            let b = lin.blocks().as_slice();
            let mut row = vec![0f32; k];
            let mut out = vec![0f32; n];
            for r in 0..n {
                crate::kquant::dequant_row_into(GgmlType::Bf16, &b[r * rb..(r + 1) * rb], &mut row);
                let mut acc = 0f32;
                for i in 0..k {
                    acc += row[i] * x[i];
                }
                out[r] = acc;
            }
            out
        }

        /// `DENNO_MIX_FAST` の解釈 (既定 on、`0` / `false` / `off` だけ旧経路)。
        #[test]
        fn mix_fast_gate_defaults_on() {
            assert!(mix_fast_from_value(None), "未設定は既定で on");
            assert!(mix_fast_from_value(Some("1")));
            assert!(mix_fast_from_value(Some("")));
            assert!(!mix_fast_from_value(Some("0")), "\"0\" で旧経路");
            assert!(!mix_fast_from_value(Some("false")));
            assert!(!mix_fast_from_value(Some("OFF")));
        }

        /// 高速経路 (既定 on) が旧経路とビット一致する。実モデルと同じ
        /// Q8_0 + `fast_q8` の融合カーネルを通る形で比べる。
        #[test]
        fn hc_mix_fast_matches_ref_bitexact() {
            let (hp, norm, down, up, inject, res) = mix_q8_case(2, 32, 32);
            let before = MIX_FAST_CALLS.load(std::sync::atomic::Ordering::Relaxed);
            let (m_fast, i_fast) =
                hc_mix_quant(&res, &norm, &down, &up, Some(&inject), &hp).expect("fast");
            assert!(
                MIX_FAST_CALLS.load(std::sync::atomic::Ordering::Relaxed) > before,
                "既定 on の高速経路が通っていない"
            );
            let (m_ref, i_ref) =
                hc_mix_quant_ref(&res, &norm, &down, &up, Some(&inject), &hp).expect("ref");
            assert_bitwise_eq(&m_fast, &m_ref, "hc_mix_quant fast vs ref (mixed)");
            assert_bitwise_eq(&i_fast, &i_ref, "hc_mix_quant fast vs ref (inject)");
            assert!(
                m_fast.iter().any(|v| *v != 0.0),
                "mixed が全 0 (経路の生死が分からない)"
            );
        }

        /// 高速経路の `hc_mixed` が rayon 経路に入る形 (n_embd = しきい値) でも
        /// 旧経路とビット一致する。
        #[test]
        fn hc_mix_fast_matches_ref_at_par_threshold() {
            let (hp, norm, down, up, inject, res) = mix_q8_case(4, 1024, 32);
            assert_eq!(hp.n_embd, crate::model::hyper::MIXED_PAR_MIN_E);
            let (m_fast, _) =
                hc_mix_quant(&res, &norm, &down, &up, Some(&inject), &hp).expect("fast");
            let (m_ref, _) =
                hc_mix_quant_ref(&res, &norm, &down, &up, Some(&inject), &hp).expect("ref");
            assert_bitwise_eq(&m_fast, &m_ref, "hc_mix_quant fast vs ref (e = 1024)");
        }

        /// `hc_*_down` (320 行) が高速経路で行並列に入り、値は旧経路とビット一致する。
        ///
        /// 既定の門番は 512 行なので、旧経路では 1 スレッドだった分岐。
        #[test]
        fn mix_fast_down_gemv_takes_row_parallel() {
            use crate::model::moe_expert::ROW_PAR_REGIONS;
            let (hp, norm, down, up, inject, res) = mix_q8_case(2, 160, 320);
            assert_eq!(down.n(), 320, "テストの前提 (down は 320 行)");
            let before = ROW_PAR_REGIONS.load(std::sync::atomic::Ordering::Relaxed);
            let (m_fast, i_fast) =
                hc_mix_quant(&res, &norm, &down, &up, Some(&inject), &hp).expect("fast");
            assert!(
                ROW_PAR_REGIONS.load(std::sync::atomic::Ordering::Relaxed) > before,
                "320 行の down が行並列に入っていない"
            );
            let (m_ref, i_ref) =
                hc_mix_quant_ref(&res, &norm, &down, &up, Some(&inject), &hp).expect("ref");
            assert_bitwise_eq(&m_fast, &m_ref, "down 行並列の有無 (mixed)");
            assert_bitwise_eq(&i_fast, &i_ref, "down 行並列の有無 (inject)");
        }

        // -------------------------------------------------------------------
        // task 110: hc mix BF16 down/up の GPU 経路 (`DENNO_MIX_GPU`、既定 off)
        // -------------------------------------------------------------------

        /// task 110: 上書きを必ず env 読みに戻す (panic しても戻る)。
        /// `MIX_GPU_OVERRIDE` はプロセス全体で 1 個なので、触るテストは
        /// [`mix_gpu_test_lock`] で直列化してからこのガードを立てる。
        struct MixGpuGuard;
        static MIX_GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        impl MixGpuGuard {
            fn new() -> Self {
                set_mix_gpu_for_test(None);
                Self
            }
        }
        impl Drop for MixGpuGuard {
            fn drop(&mut self) {
                set_mix_gpu_for_test(None);
            }
        }

        /// task 110: 上書きを触るテストの直列化 (linear_gguf の `par_test_lock` と
        /// 同じ流儀。ロックはテスト側の変数で保持する)。
        fn mix_gpu_test_lock() -> std::sync::MutexGuard<'static, ()> {
            MIX_GPU_TEST_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        }

        /// task 138: `DENNO_MIX_CHAIN` は既定 on。明示的な偽値だけ切る。
        #[test]
        fn mix_chain_gate_defaults_on() {
            assert!(mix_chain_from_value(None), "未設定は on のはず");
            for v in [Some("1"), Some("on"), Some("ON"), Some("true"), Some(" yes ")] {
                assert!(mix_chain_from_value(v), "value={v:?} は on のはず");
            }
            for v in [Some("0"), Some("off"), Some("OFF"), Some("false"), Some("no")] {
                assert!(!mix_chain_from_value(v), "value={v:?} は off のはず");
            }
        }

        /// task 150: `DENNO_DECODE_CHAIN2` は lab t150b で採用して既定 on、
        /// 部分ゲート `DENNO_DECODE_CHAIN2_EXP` / `_MIX` も既定 on。部分ゲートは
        /// 束ねゲートが on のときだけ効く (純関数で固定する)。
        #[test]
        fn decode_chain2_gates_truth_table() {
            // lab t150b で採用 → 既定 on (明示的な偽値だけ切る)。
            assert!(decode_chain2_from_value(None), "未設定は on のはず");
            for v in [
                Some("1"),
                Some("on"),
                Some("ON"),
                Some(" true "),
                Some("yes"),
                Some(""),
            ] {
                assert!(decode_chain2_from_value(v), "value={v:?} は on のはず");
            }
            for v in [Some("0"), Some("off"), Some("OFF"), Some("false"), Some("no")] {
                assert!(!decode_chain2_from_value(v), "value={v:?} は off のはず");
            }
            assert!(decode_chain2_exp_from_value(None), "未設定は on のはず");
            for v in [Some("1"), Some("on"), Some("ON"), Some(" yes "), Some("")] {
                assert!(decode_chain2_exp_from_value(v), "value={v:?} は on のはず");
            }
            for v in [Some("0"), Some("off"), Some("false"), Some("no")] {
                assert!(!decode_chain2_exp_from_value(v), "value={v:?} は off のはず");
            }
            // 合成: 束ねゲートが off なら部分ゲートの値に依らず off。
            for exp in [false, true] {
                assert!(!chain2_exp_active(false, exp), "束ねゲート off で on になった");
                assert_eq!(chain2_exp_active(true, exp), exp);
            }
            // MIX の部分ゲートも EXP と同じ規則 (既定 on)。
            assert!(decode_chain2_mix_from_value(None));
            assert!(!decode_chain2_mix_from_value(Some("0")));
            assert!(decode_chain2_mix_from_value(Some("on")));
        }

        /// task 150: ヒットとミスの寄与を `sel` の並びへ戻す (位置の取り違え・
        /// 長さの不一致を静かに通さない)。
        #[test]
        #[cfg(feature = "cuda")]
        fn scatter_contribs_places_hits_and_misses() {
            let e = 2usize;
            let pos = [0usize, 3];
            let h = [10.0f32, 11.0, 40.0, 41.0];
            let miss_pos = [1usize, 2, 4];
            let miss_h = [20.0f32, 21.0, 30.0, 31.0, 50.0, 51.0];
            let mut hs: Vec<Option<Vec<f32>>> = vec![None; 5];
            scatter_contribs(&mut hs, &pos, &h, &miss_pos, &miss_h, e).expect("scatter");
            assert_eq!(hs[0].as_deref(), Some(&[10.0f32, 11.0][..]));
            assert_eq!(hs[1].as_deref(), Some(&[20.0f32, 21.0][..]));
            assert_eq!(hs[2].as_deref(), Some(&[30.0f32, 31.0][..]));
            assert_eq!(hs[3].as_deref(), Some(&[40.0f32, 41.0][..]));
            assert_eq!(hs[4].as_deref(), Some(&[50.0f32, 51.0][..]));
            // ミスを渡さなければヒットだけが埋まる (従来経路と同じ形)。
            let mut hs2: Vec<Option<Vec<f32>>> = vec![None; 5];
            scatter_contribs(&mut hs2, &pos, &h, &[], &[], e).expect("scatter hits only");
            assert!(hs2[0].is_some() && hs2[3].is_some());
            assert!(hs2[1].is_none() && hs2[2].is_none() && hs2[4].is_none());
            // 長さが合わない / 範囲外は Err (別の expert の値を足さない)。
            let mut hs3: Vec<Option<Vec<f32>>> = vec![None; 5];
            assert!(scatter_contribs(&mut hs3, &pos, &h[..2], &[], &[], e).is_err());
            assert!(scatter_contribs(&mut hs3, &[5], &h[..2], &[], &[], e).is_err());
            // 表が空なのにバッファだけ非空 → Err (lab t150a で CHAIN2=0 が踏んだ形)。
            assert!(scatter_contribs(&mut hs3, &[0], &h[..2], &[], &h[..4], e).is_err());
            // miss_slots が確保長の唯一の入口: overlap off は 0 本。
            assert_eq!(miss_slots(false, 4), 0);
            assert_eq!(miss_slots(true, 4), 4);
        }

        /// task 150: `dno_hc_mixed_f32` の 1 要素を .cu の行順で写したミラー。
        /// CPU の `hc_mixed_into` と **ビット一致** することを固定する
        /// (sigmoid は 1/(1+exp(-v))・加算は s 昇順・1/hc は和の後)。
        #[test]
        #[cfg(feature = "cuda")]
        fn hc_mixed_kernel_mirror_matches_cpu() {
            fn sig(v: f32) -> f32 {
                1.0 / (1.0 + (-v).exp())
            }
            for (hc, e) in [(1usize, 1usize), (4, 7), (4, 2560), (2, 96)] {
                let hp = HyperParams {
                    hc,
                    n_embd: e,
                    low_rank: 320,
                    eps: 1e-6,
                };
                let n = hc * e;
                let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.017).sin() * 3.0).collect();
                let g: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.011).cos() * 5.0).collect();
                let mut want = vec![0.0f32; e];
                hyper::hc_mixed_into(&x, &g, &mut want, &hp);
                // dno_hc_mixed_f32 の 1 スレッド分 (s = 0 で代入、s >= 1 で加算)。
                let inv_hc = 1.0f32 / hc as f32;
                let mut got = vec![0.0f32; e];
                for (d, slot) in got.iter_mut().enumerate() {
                    let mut acc = x[d] * sig(g[d]);
                    for s in 1..hc {
                        let i = s * e + d;
                        acc += x[i] * sig(g[i]);
                    }
                    *slot = acc * inv_hc;
                }
                assert_eq!(want, got, "hc={hc} e={e} で CPU と一致しない");
            }
        }

        /// task 154: `DENNO_PF_RESIDENT` (master。既定 on) と副ゲートの真理値表。
        #[test]
        fn pf_gate_truth_table() {
            // master: 未設定と空文字は既定 (on)、明示的な偽値だけ off。
            assert!(pf_resident_from_value(None), "未設定は on (lab t154e で採用)");
            assert!(pf_resident_from_value(Some("")), "空文字は既定 (on)");
            for v in ["0", "off", "false", "no", " OFF ", " 0 "] {
                assert!(!pf_resident_from_value(Some(v)), "{v:?} が on になった");
            }
            for v in ["1", "on", "true", "yes", " ON "] {
                assert!(pf_resident_from_value(Some(v)), "{v:?} が off になった");
            }
            // 副ゲート: task 181 (lab t181b) で組として目標に届いたので **既定 on**。
            assert!(pf_on_from_value(None), "未設定は on (lab t181b で採用)");
            for v in ["0", "off", "false", "no", " OFF ", " 0 "] {
                assert!(!pf_on_from_value(Some(v)), "{v:?} が on になった");
            }
            for v in ["1", "on", "true", "yes", " ON "] {
                assert!(pf_on_from_value(Some(v)), "{v:?} が off になった");
            }
        }

        /// task 167: `DENNO_GDN_PF_CHAIN` の真理値表。lab t167c で採用して**既定 on**、
        /// 明示的な偽値だけ off。
        #[test]
        fn gdn_pf_chain_gate_truth_table() {
            assert!(pf_gdn_chain_from_value(None), "未設定は on (採用済み)");
            assert!(pf_gdn_chain_from_value(Some("")), "空文字は既定 (on)");
            for v in ["0", "off", "false", "no", " OFF "] {
                assert!(!pf_gdn_chain_from_value(Some(v)), "{v:?} が on になった");
            }
            for v in ["1", "on", "true", "yes"] {
                assert!(pf_gdn_chain_from_value(Some(v)), "{v:?} が off になった");
            }
        }

        /// task 145: `DENNO_MIX_BATCH_GPU` の真理値表 (既定 on) と、バッチ mix の
        /// 門番 (BF16 / nt >= 64 / 積 >= 4M) の境界。GPU 無しでも検証できる。
        #[test]
        fn mix_batch_gpu_gate_and_thresholds() {
            for v in [None, Some(""), Some("1"), Some("on"), Some("ON"), Some("true"), Some(" yes ")] {
                assert!(mix_batch_gpu_from_value(v), "value={v:?} は on のはず");
            }
            for v in [Some("0"), Some("off"), Some("false"), Some("no")] {
                assert!(!mix_batch_gpu_from_value(v), "value={v:?} は off のはず");
            }
            let _l = mix_gpu_test_lock();
            set_mix_batch_gpu_for_test(Some(false));
            assert!(!mix_batch_gpu_enabled(), "off の上書きが効いていない");
            set_mix_batch_gpu_for_test(Some(true));
            assert!(mix_batch_gpu_enabled(), "on の上書きが効いていない");
            set_mix_batch_gpu_for_test(None);

            // 門番: 既定 off は何も通さない。
            use crate::kquant::GgmlType as T;
            assert!(!mix_batch_gpu_ok(T::Bf16, 10240, 320, 512, false));
            // 実形 (down k=10240 n=320 nt=512) は通る。
            assert!(mix_batch_gpu_ok(T::Bf16, 10240, 320, 512, true));
            // up (k=320 n=10240 nt=512) も通る。
            assert!(mix_batch_gpu_ok(T::Bf16, 320, 10240, 512, true));
            // BF16 以外は通さない。
            assert!(!mix_batch_gpu_ok(T::Q8_0, 10240, 320, 512, true));
            // nt が小さい (decode 級) は通さない。
            assert!(!mix_batch_gpu_ok(T::Bf16, 10240, 320, 63, true));
            assert!(mix_batch_gpu_ok(T::Bf16, 10240, 320, 64, true));
            // inject 級でも積が足りない形 (k=2560 n=4 nt=64 = 655k) は通さない。
            assert!(!mix_batch_gpu_ok(T::Bf16, 2560, 4, 64, true));
        }

        /// task 145: ゲートを on にしても GPU が無い / 未対応なら CPU バッチと
        /// 同じ値になる (戻り経路の健全性)。合成モデルの prefill を 2 回組み直して
        /// 比べる (PLE の conv 履歴が進むので同じ個体で 2 回は回さない)。
        #[test]
        fn mix_batch_gpu_falls_back_to_cpu_identically() {
            let _g = moe_test_guard();
            let _l = mix_gpu_test_lock();
            let dir = TempDir::new("mix_batch_gpu_fallback");
            let path = build(&dir);
            let ctx: Vec<u32> = vec![1, 2, 3, 4, 5, 6];

            set_mix_batch_gpu_for_test(Some(false));
            let mut m1 = Qwen4Exp::new(&GgufFile::open(&path).expect("open")).expect("new");
            let off = m1.forward_prefill_with(&ctx, true).expect("prefill (off)");

            set_mix_batch_gpu_for_test(Some(true));
            let mut m2 = Qwen4Exp::new(&GgufFile::open(&path).expect("open")).expect("new");
            let on = m2.forward_prefill_with(&ctx, true).expect("prefill (on)");

            set_mix_batch_gpu_for_test(None);
            assert_eq!(off.len(), on.len(), "logits の長さが違う");
            for (i, (a, b)) in off.iter().zip(on.iter()).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "logits[{i}] が違う ({a} != {b})");
            }
        }

        /// task 154: `pf_lin_ok` は常駐連鎖が扱える型 (BF16 / F32 / 量子 4 種) を
        /// 受け、形が違えば落とす。lab t154b/t154c では inject (Q8_0) を弾いて
        /// 常駐経路が丸ごと不発になったので、型の受容をここで固定する。
        #[test]
        fn pf_lin_ok_accepts_quant_and_rejects_shape() {
            use crate::kquant::GgmlType as T;
            use crate::model::linear_gguf::GgufLinear;
            let k = 64usize;
            let n = 2usize;
            let ty_bytes = |ty: T| -> Vec<u8> {
                match ty {
                    T::F32 => vec![0u8; k * n * 4],
                    T::Bf16 => vec![0u8; k * n * 2],
                    T::Q8_0 => vec![0u8; (k / 32) * 34 * n],
                    T::Q5_1 => vec![0u8; (k / 32) * 24 * n],
                    T::Q4K | T::Q5K => vec![0u8; (k / 256).max(1) * 144 * n + 4096],
                    _ => Vec::new(),
                }
            };
            for ty in [T::F32, T::Bf16, T::Q8_0, T::Q5_1] {
                let l = GgufLinear::from_slice(ty, k, n, &ty_bytes(ty)).expect("linear");
                assert!(Qwen4Exp::pf_lin_ok(&l, k, n), "{ty:?} が弾かれた");
                assert!(!Qwen4Exp::pf_lin_ok(&l, k, n + 1), "{ty:?}: 形違いを通した");
                assert!(!Qwen4Exp::pf_lin_ok(&l, k + 32, n), "{ty:?}: k 違いを通した");
            }
        }

        /// task 154: PLE の射影をチャンクでまとめても値はビット一致する
        /// (`gemv_batch_cpu` は逐次版と同じカーネル・同じ加算順)。
        #[test]
        fn ple_inject_batch_matches_per_token_bits() {
            let _g = moe_test_guard();
            let dir = TempDir::new("ple_inject_batch");
            let path = build(&dir);
            let f = GgufFile::open(&path).expect("open");
            let m = Qwen4Exp::new(&f).expect("new");
            let il = PLE_LAYER;
            let pw = m.layers[il].ple.as_ref().expect("PLE 層");
            let hp = m.hp();
            let e = m.p.n_embd;
            let hc = hp.hc;
            let hc_dim = hc * e;
            let nt = 5usize;
            let mut rnd = 0x1234_5678u64;
            let mut next = move || {
                rnd = rnd
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((rnd >> 33) as f32) / (u32::MAX as f32) - 0.5
            };
            let base: Vec<f32> = (0..nt * hc_dim).map(|_| 0.2 * next()).collect();
            let emb_len = pw.key.k();
            let emb: Vec<f32> = (0..nt * emb_len).map(|_| 0.2 * next()).collect();
            let mk = |data: &[f32]| -> Vec<HyperStreams> {
                (0..nt)
                    .map(|t| HyperStreams {
                        data: data[t * hc_dim..(t + 1) * hc_dim].to_vec(),
                        hc,
                        n_embd: e,
                    })
                    .collect()
            };
            let hist_len = (pw.kern - 1) * m.p.ple_ngram_size * hc_dim;
            let hist0: Vec<f32> = (0..hist_len)
                .map(|i| 0.05 * (((i * 7) % 13) as f32 - 6.0))
                .collect();
            let rows: Vec<&[f32]> = (0..nt).map(|t| &emb[t * emb_len..(t + 1) * emb_len]).collect();
            // 基準: 1 トークンずつ (task 189 の作業域を使わない = 毎回確保する経路)。
            crate::ple::set_pf_ple_par_for_test(Some(false));
            let mut a = mk(&base);
            let mut hist_a = hist0.clone();
            for t in 0..nt {
                let pe = &emb[t * emb_len..(t + 1) * emb_len];
                a[t] = ple_inject(&m.p, pw, pe, &a[t], &hp, &mut hist_a, None).expect("逐次");
            }
            // task 189: バッチ経路を作業域の確保あり / 使い回し (冷・温) で回し、
            // どれも逐次版とビット一致することを固定する。
            for gate in [Some(false), Some(true), Some(true)] {
                let mut b = mk(&base);
                let mut hist_b = hist0.clone();
                crate::ple::set_pf_ple_par_for_test(gate);
                ple_inject_batch(&m.p, pw, &rows, &mut b, &hp, &mut hist_b, None)
                    .expect("バッチ");
                for t in 0..nt {
                    let want: Vec<u32> = a[t].data.iter().map(|v| v.to_bits()).collect();
                    let got: Vec<u32> = b[t].data.iter().map(|v| v.to_bits()).collect();
                    assert_eq!(want, got, "gate={gate:?} トークン {t} の出力がビット不一致");
                }
                let want: Vec<u32> = hist_a.iter().map(|v| v.to_bits()).collect();
                let got: Vec<u32> = hist_b.iter().map(|v| v.to_bits()).collect();
                assert_eq!(want, got, "gate={gate:?} の conv 履歴がビット不一致");
            }
            crate::ple::set_pf_ple_par_for_test(None);
        }

        /// task 195: `DENNO_PF_PLE_FAST` の平坦経路 (リング履歴 + その場更新) が
        /// 逐次版とビット一致すること。履歴はチャンクをまたぐので 2 チャンク回す。
        #[test]
        fn ple_inject_batch_flat_matches_per_token_bits() {
            let _g = moe_test_guard();
            let dir = TempDir::new("ple_inject_flat");
            let path = build(&dir);
            let f = GgufFile::open(&path).expect("open");
            let m = Qwen4Exp::new(&f).expect("new");
            let il = PLE_LAYER;
            let pw = m.layers[il].ple.as_ref().expect("PLE 層");
            let hp = m.hp();
            let e = m.p.n_embd;
            let hc = hp.hc;
            let hc_dim = hc * e;
            let nt = 5usize;
            let chunks = 2usize;
            let mut rnd = 0x0bad_f00d_dead_beefu64;
            let mut next = move || {
                rnd = rnd
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((rnd >> 33) as f32) / (u32::MAX as f32) - 0.5
            };
            let emb_len = pw.key.k();
            let hist_len = (pw.kern - 1) * m.p.ple_ngram_size * hc_dim;
            let hist0: Vec<f32> = (0..hist_len)
                .map(|i| 0.05 * (((i * 7) % 13) as f32 - 6.0))
                .collect();
            // 基準 (1 トークンずつ) と平坦経路で、同じ入力を 2 チャンク流す。
            let mut hist_a = hist0.clone();
            let mut hist_b = hist0.clone();
            for c in 0..chunks {
                let base: Vec<f32> = (0..nt * hc_dim).map(|_| 0.2 * next()).collect();
                let emb: Vec<f32> = (0..nt * emb_len).map(|_| 0.2 * next()).collect();
                let rows: Vec<&[f32]> =
                    (0..nt).map(|t| &emb[t * emb_len..(t + 1) * emb_len]).collect();
                let mut a: Vec<HyperStreams> = (0..nt)
                    .map(|t| HyperStreams {
                        data: base[t * hc_dim..(t + 1) * hc_dim].to_vec(),
                        hc,
                        n_embd: e,
                    })
                    .collect();
                crate::ple::set_pf_ple_par_for_test(Some(false));
                for t in 0..nt {
                    let pe = &emb[t * emb_len..(t + 1) * emb_len];
                    a[t] = ple_inject(&m.p, pw, pe, &a[t], &hp, &mut hist_a, None)
                        .expect("逐次");
                }
                let mut b = base.clone();
                ple_inject_batch_flat(&m.p, pw, &rows, &mut b, &hp, &mut hist_b, None)
                    .expect("平坦");
                for t in 0..nt {
                    let want: Vec<u32> = a[t].data.iter().map(|v| v.to_bits()).collect();
                    let got: Vec<u32> = b[t * hc_dim..(t + 1) * hc_dim]
                        .iter()
                        .map(|v| v.to_bits())
                        .collect();
                    assert_eq!(want, got, "チャンク {c} トークン {t} の出力がビット不一致");
                }
                let want: Vec<u32> = hist_a.iter().map(|v| v.to_bits()).collect();
                let got: Vec<u32> = hist_b.iter().map(|v| v.to_bits()).collect();
                assert_eq!(want, got, "チャンク {c} の conv 履歴がビット不一致");
            }
            crate::ple::set_pf_ple_par_for_test(None);
        }

        /// task 300: PLE の作業域使い回しの契約。
        ///
        /// - ゲート off (`reuse=false`) は毎回 `len` ちょうどの新しい Vec を返す。
        /// - on は grow-only の使い回しで、**返る長さは常に `len` ちょうど**
        ///   (大きいチャンクの後で小さいチャンクが来ても切り詰める)。
        /// - 返却した Vec は次の take でそのまま返ってくる (確保し直さない)。
        #[test]
        fn ple_reuse_scratch_contract() {
            // ゲートの真理値表 (未設定・空は既定 on。明示的な偽値だけ off)。
            assert!(pf_ple_reuse_from_value(None));
            assert!(pf_ple_reuse_from_value(Some("")));
            assert!(!pf_ple_reuse_from_value(Some("0")));
            assert!(!pf_ple_reuse_from_value(Some("off")));
            assert!(pf_ple_reuse_from_value(Some("1")));
            assert!(pf_ple_reuse_from_value(Some(" on ")));
            assert!(pf_ple_reuse_from_value(Some("TRUE")));

            // task 505: `DENNO_PF_PLE_FLAT_REUSE` は測って採るまで既定 off。
            for v in [None, Some(""), Some("0"), Some("off"), Some("no")] {
                assert!(
                    !pf_ple_flat_reuse_from_value(v),
                    "flat_reuse が on になった: {v:?}"
                );
            }
            for v in [Some("1"), Some("on"), Some(" on "), Some("TRUE")] {
                assert!(
                    pf_ple_flat_reuse_from_value(v),
                    "flat_reuse が off のまま: {v:?}"
                );
            }

            // task 393: `DENNO_PF_PLE_PIN` は lab で採ったので既定 on
            // (明示的な偽値だけ off)。
            assert!(pf_ple_pin_from_value(None), "未設定が off になった");
            assert!(pf_ple_pin_from_value(Some("")));
            for v in ["0", "off", "false", "no"] {
                assert!(!pf_ple_pin_from_value(Some(v)), "{v:?} が on になった");
            }
            for v in ["1", "on", "true", "yes", " ON "] {
                assert!(pf_ple_pin_from_value(Some(v)), "{v:?} が off のまま");
            }

            // off: 毎回新しい Vec (長さはちょうど)。
            let (a, r) = pf_ple_back_take_with(8, false);
            assert!(!r);
            assert_eq!(a.len(), 8);
            let (b, r) = pf_ple_back_take_with(4, false);
            assert!(!r);
            assert_eq!(b.len(), 4);

            // on: 返却 → 次の take で同じ容量を再利用。
            let (mut v, r) = pf_ple_back_take_with(8, true);
            assert!(r);
            assert_eq!(v.len(), 8);
            v.fill(7.0);
            let ptr = v.as_ptr();
            pf_ple_back_return(v, true);
            let (v2, r2) = pf_ple_back_take_with(4, true);
            assert!(r2);
            assert_eq!(v2.len(), 4, "長さは要求どおりに切り詰める");
            assert_eq!(v2.as_ptr(), ptr, "同じ領域を使い回す");
            // 使い回しでも中身は古い値が残っている (呼び出し側が全要素を書く契約)。
            assert_eq!(v2[0].to_bits(), 7.0f32.to_bits());
            pf_ple_back_return(v2, true);

            // key / value も同じ契約 (長さはそれぞれ独立)。
            let ((k, vl), r3) = pf_ple_kv_take_with(6, 3, true);
            assert!(r3);
            assert_eq!((k.len(), vl.len()), (6, 3));
            let kp = k.as_ptr();
            pf_ple_kv_return((k, vl), true);
            let ((k2, vl2), r4) = pf_ple_kv_take_with(5, 2, true);
            assert!(r4);
            assert_eq!((k2.len(), vl2.len()), (5, 2));
            assert_eq!(k2.as_ptr(), kp, "key も同じ領域を使い回す");
            pf_ple_kv_return((k2, vl2), true);
            // 0 長は使い回さない (呼び出し側の空スライスを壊さない)。
            let ((k3, v3), r5) = pf_ple_kv_take_with(0, 0, true);
            assert!(!r5);
            assert!(k3.is_empty() && v3.is_empty());
        }

        /// task 393: pinned の枝 (`DENNO_PF_PLE_PIN`) の契約。
        ///
        /// - 返る長さは要求ちょうど (大きいチャンクの後で小さいチャンクが来ても切る)
        /// - 返却した 1 本を次の take がそのまま使う (同じポインタ)
        /// - 大きい要求は育つ
        ///
        /// このコンテナには CUDA ドライバが無いので `PfHostBuf` は `Vec` に落ちるが、
        /// 契約 (長さ・使い回し・書き込み) はどちらの実体でも同じ。
        #[test]
        fn ple_pin_back_contract() {
            let (mut a, r) = pf_ple_back_take_pin(8, true, true);
            assert!(r, "使い回しの印が立っていない");
            assert_eq!(a.len(), 8);
            a.fill(3.5);
            let ptr = a.as_ptr();
            assert!(a.iter().all(|v| *v == 3.5), "全要素を書けない");
            pf_ple_back_return(a, true);

            let (mut b, r2) = pf_ple_back_take_pin(4, true, true);
            assert!(r2);
            assert_eq!(b.len(), 4, "長さは要求どおりに切り詰める");
            assert_eq!(b.as_ptr(), ptr, "同じ領域を使い回す");
            b.fill(0.0);
            pf_ple_back_return(b, true);

            // 大きい要求は育つ。
            let (c, r3) = pf_ple_back_take_pin(16, true, true);
            assert!(r3);
            assert_eq!(c.len(), 16);
            pf_ple_back_return(c, true);
            // reuse=false は使い回さない (毎回新しいバッファ)。
            let (d, r4) = pf_ple_back_take_pin(4, false, true);
            assert!(!r4);
            assert_eq!(d.len(), 4);
        }

        /// task 195: 並列版 (`ple_inject_batch_par`) も逐次版とビット一致すること。
        #[test]
        fn ple_inject_batch_par_matches_per_token_bits() {
            let _g = moe_test_guard();
            let dir = TempDir::new("ple_inject_par");
            let path = build(&dir);
            let f = GgufFile::open(&path).expect("open");
            let m = Qwen4Exp::new(&f).expect("new");
            let il = PLE_LAYER;
            let pw = m.layers[il].ple.as_ref().expect("PLE 層");
            let hp = m.hp();
            let e = m.p.n_embd;
            let hc = hp.hc;
            let hc_dim = hc * e;
            let nt = 37usize; // 端数 (rayon のチャンク割りが割り切れない)
            let chunks = 2usize;
            let mut rnd = 0x1950_0bad_c0ffee01u64;
            let mut next = move || {
                rnd = rnd
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((rnd >> 33) as f32) / (u32::MAX as f32) - 0.5
            };
            let emb_len = pw.key.k();
            let hist_len = (pw.kern - 1) * m.p.ple_ngram_size * hc_dim;
            let hist0: Vec<f32> = (0..hist_len)
                .map(|i| 0.05 * (((i * 7) % 13) as f32 - 6.0))
                .collect();
            let mut hist_a = hist0.clone();
            let mut hist_b = hist0.clone();
            let mut rms_ref = 0.0f32;
            for c in 0..chunks {
                let base: Vec<f32> = (0..nt * hc_dim).map(|_| 0.2 * next()).collect();
                let emb: Vec<f32> = (0..nt * emb_len).map(|_| 0.2 * next()).collect();
                let rows: Vec<&[f32]> =
                    (0..nt).map(|t| &emb[t * emb_len..(t + 1) * emb_len]).collect();
                let mut a: Vec<HyperStreams> = (0..nt)
                    .map(|t| HyperStreams {
                        data: base[t * hc_dim..(t + 1) * hc_dim].to_vec(),
                        hc,
                        n_embd: e,
                    })
                    .collect();
                crate::ple::set_pf_ple_par_for_test(Some(false));
                for t in 0..nt {
                    let pe = &emb[t * emb_len..(t + 1) * emb_len];
                    let want_rms = t + 1 == nt;
                    a[t] = ple_inject(
                        &m.p,
                        pw,
                        pe,
                        &a[t],
                        &hp,
                        &mut hist_a,
                        want_rms.then_some(&mut rms_ref),
                    )
                    .expect("逐次");
                }
                let mut b = base.clone();
                let mut last = 0.0f32;
                ple_inject_batch_par(&m.p, pw, &rows, &mut b, &hp, &mut hist_b, Some(&mut last))
                    .expect("並列");
                for t in 0..nt {
                    let want: Vec<u32> = a[t].data.iter().map(|v| v.to_bits()).collect();
                    let got: Vec<u32> = b[t * hc_dim..(t + 1) * hc_dim]
                        .iter()
                        .map(|v| v.to_bits())
                        .collect();
                    assert_eq!(want, got, "チャンク {c} トークン {t} の出力がビット不一致");
                }
                let want: Vec<u32> = hist_a.iter().map(|v| v.to_bits()).collect();
                let got: Vec<u32> = hist_b.iter().map(|v| v.to_bits()).collect();
                assert_eq!(want, got, "チャンク {c} の conv 履歴がビット不一致");
                // 寄与 RMS も逐次版と一致すること (最後のトークンで取る)。
                assert_eq!(
                    last.to_bits(),
                    rms_ref.to_bits(),
                    "チャンク {c} の寄与 RMS がビット不一致"
                );
            }
            crate::ple::set_pf_ple_par_for_test(None);
        }

        /// task 195: conv の履歴はリングでも線形でも同じ値になること
        /// (head をずらしながら 1 トークンずつ進め、線形版と突き合わせる)。
        #[test]
        fn ple_conv_ring_matches_linear() {
            let mut rnd = 0x5eed_1950_0000_0001u64;
            let mut next = move || {
                rnd = rnd
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((rnd >> 33) as f32) / (u32::MAX as f32) - 0.5
            };
            for &(kern, dil) in &[(4usize, 3usize), (4, 1), (2, 1), (3, 2)] {
                let hc_dim = 13usize;
                let hist = (kern - 1) * dil;
                let w: Vec<f32> = (0..hc_dim * kern).map(|_| next()).collect();
                let hist0: Vec<f32> = (0..hist * hc_dim).map(|_| next()).collect();
                let mut lin = hist0.clone();
                let mut ring = hist0.clone();
                let mut head = 0usize;
                for _ in 0..(hist + 3) {
                    let x: Vec<f32> = (0..hc_dim).map(|_| next()).collect();
                    let mut a = vec![0.0f32; hc_dim];
                    ple_conv_silu(&mut a, &x, &lin, 0, &w, kern, dil, hc_dim);
                    let mut b = vec![0.0f32; hc_dim];
                    ple_conv_silu(&mut b, &x, &ring, head, &w, kern, dil, hc_dim);
                    for c in 0..hc_dim {
                        assert_eq!(a[c].to_bits(), b[c].to_bits(), "conv がビット不一致");
                    }
                    // task 195: kern 優先の重み版 (`conv1d_t`) も同じ値になること。
                    let mut wt = vec![0.0f32; w.len()];
                    for c in 0..hc_dim {
                        for k in 0..kern {
                            wt[k * hc_dim + c] = w[c * kern + k];
                        }
                    }
                    let mut t = vec![0.0f32; hc_dim];
                    ple_conv_silu_t(&mut t, &x, &ring, head, &wt, kern, dil, hc_dim);
                    for c in 0..hc_dim {
                        assert_eq!(
                            b[c].to_bits(),
                            t[c].to_bits(),
                            "kern 優先の重みで conv がビット不一致"
                        );
                    }
                    if hist > 0 {
                        ple_hist_push_shift(&mut lin, &x, hc_dim);
                        ple_hist_push_ring(&mut ring, &mut head, &x, hc_dim);
                        // リングを線形に戻すと線形版と一致する。
                        let mut tmp = Vec::new();
                        let mut linz = ring.clone();
                        ple_hist_to_linear(&mut linz, head, hc_dim, &mut tmp);
                        for i in 0..hist * hc_dim {
                            assert_eq!(
                                lin[i].to_bits(),
                                linz[i].to_bits(),
                                "履歴の線形化がビット不一致"
                            );
                        }
                    }
                }
            }
        }

        /// task 195: 新しいゲートの真理値表 (task 198 で未設定は既定 on)。
        #[test]
        fn pf195_gate_truth_tables() {
            assert!(pf_ple_fast_from_value(None), "未設定は既定 on");
            assert!(pf_router_fast_from_value(None), "未設定は既定 on");
            for v in ["0", "off", "false", "no", "", " off "] {
                assert!(!pf_ple_fast_from_value(Some(v)), "{v:?} が on");
                assert!(!pf_router_fast_from_value(Some(v)), "{v:?} が on");
            }
            for v in ["1", "on", "true", "yes", "2"] {
                assert!(pf_ple_fast_from_value(Some(v)), "{v:?} が off");
                assert!(pf_router_fast_from_value(Some(v)), "{v:?} が off");
            }
        }

        /// task 195: 挿入法の top-k が `select_topk_into` と同一の選択になること
        /// (同値を大量に含む入力・k = n・k = 1 も含める)。
        #[test]
        fn router_topk_fast_matches_select_nth() {
            let mut rnd = 0x1950_1950_1950_1950u64;
            let mut next = move || {
                rnd = rnd
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((rnd >> 33) as f32) / (u32::MAX as f32) - 0.5
            };
            for &(ne, k) in &[(512usize, 8usize), (512, 1), (128, 32), (64, 64), (37, 5)] {
                for variant in 0..3 {
                    let mut probs: Vec<f32> = (0..ne).map(|_| next()).collect();
                    if variant == 1 {
                        // 同値を大量に (量子化された確率を模す)。
                        for v in probs.iter_mut() {
                            *v = (*v * 4.0).round() / 4.0;
                        }
                    } else if variant == 2 {
                        // 全部同じ値。
                        probs.iter_mut().for_each(|v| *v = 0.5);
                    }
                    let mut buf: Vec<u32> = Vec::new();
                    let mut a = vec![0usize; k];
                    select_topk_into(&probs, k, &mut buf, &mut a);
                    let mut b = vec![0usize; k];
                    select_topk_into_fast(&probs, k, &mut b);
                    assert_eq!(a, b, "ne={ne} k={k} variant={variant} で選択が違う");
                }
            }
        }

            /// task 189: conv のループ入れ替え (k 外側) が旧ループとビット一致すること。
        #[test]
        fn ple_conv_loop_interchange_is_bit_exact() {
            let mut rnd = 0x9e37_79b9_7f4a_7c15u64;
            let mut next = move || {
                rnd = rnd
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((rnd >> 33) as f32) / (u32::MAX as f32) - 0.5
            };
            for &(kern, dil) in &[(4usize, 3usize), (4, 1), (2, 1), (3, 2), (5, 4), (1, 3)] {
                let hc_dim = 37usize; // 端数 (8 の倍数でない) も通す
                let hist = (kern - 1) * dil;
                let x: Vec<f32> = (0..hc_dim).map(|_| next()).collect();
                let hb: Vec<f32> = (0..hist * hc_dim).map(|_| next()).collect();
                let w: Vec<f32> = (0..hc_dim * kern).map(|_| next()).collect();
                let want = ple_conv_silu_ref(&x, &hb, &w, kern, dil, hc_dim);
                let mut got = vec![0.0f32; hc_dim];
                ple_conv_silu(&mut got, &x, &hb, 0, &w, kern, dil, hc_dim);
                for c in 0..hc_dim {
                    assert_eq!(
                        want[c].to_bits(),
                        got[c].to_bits(),
                        "kern={kern} dil={dil} c={c}: {} != {}",
                        want[c],
                        got[c]
                    );
                }
            }
        }

    /// task 154: 常駐ゲートを on にしても、GPU / PTX / 形の前提が無ければ
        /// 常駐経路は **層を 1 つも進めずに** 従来経路へ戻る (logits はビット一致・
        /// attention の状態も 1 回ぶんだけ進む)。
        #[test]
        fn pf_resident_declines_identically_without_gpu() {
            let _g = moe_test_guard();
            let dir = TempDir::new("pf_resident_fallback");
            let path = build(&dir);
            let ctx: Vec<u32> = vec![1, 2, 3, 4, 5, 6];

            for w in [PF_SUB_GDN, PF_SUB_MIX] {
                set_pf_sub_for_test(w, Some(false));
            }
            let mut m1 = Qwen4Exp::new(&GgufFile::open(&path).expect("open")).expect("new");
            let off = m1.forward_prefill_with(&ctx, true).expect("prefill (off)");

            for w in [PF_SUB_GDN, PF_SUB_MIX] {
                set_pf_sub_for_test(w, Some(true));
            }
            // master は既定 on (真理値表は `pf_gate_truth_table`)。GPU / PTX が無い
            // この環境では入口の `pf_begin` が false を返し、層を進めずに従来経路へ
            // 落ちる。ここで見るのは「落ちたときの logits が off と同一」であること。
            let mut m2 = Qwen4Exp::new(&GgufFile::open(&path).expect("open")).expect("new");
            let on = m2.forward_prefill_with(&ctx, true).expect("prefill (on)");
            for w in [PF_SUB_GDN, PF_SUB_MIX] {
                set_pf_sub_for_test(w, None);
            }
            assert_eq!(off.len(), on.len(), "logits の長さが違う");
            for (i, (a, b)) in off.iter().zip(on.iter()).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "logits[{i}] が違う ({a} != {b})");
            }
        }

        /// task 110: `DENNO_MIX_GPU` の真理値表。既定 off、明示的な真値だけ on
        /// (`router_gpu_from_value` と同一規約)。
        #[test]
        fn mix_gpu_gate_defaults_off() {
            for v in [
                Some("1"),
                Some("on"),
                Some("ON"),
                Some("true"),
                Some(" yes "),
            ] {
                assert!(mix_gpu_from_value(v), "value={v:?} は on のはず");
            }
            for v in [
                None,
                Some(""),
                Some("0"),
                Some("off"),
                Some("false"),
                Some("no"),
            ] {
                assert!(!mix_gpu_from_value(v), "value={v:?} は off のはず");
            }
            let _l = mix_gpu_test_lock();
            let _g = MixGpuGuard::new();
            set_mix_gpu_for_test(Some(false));
            assert!(!mix_gpu_enabled(), "off の上書きが効いていない");
            set_mix_gpu_for_test(Some(true));
            assert!(mix_gpu_enabled(), "on の上書きが効いていない");
        }

        /// task 110: 配線確認。テスト上書きが on のときだけ hc_mix_quant から
        /// cuda 入口 (`MIX_GPU_ATTEMPTS`) に届くこと。入口は `cuda_enabled()` の
        /// 判定より**前**で数えるので、GPU 無し環境でも配線だけ検証できる
        /// (BF16 実型の down/up で 1 呼び出し 2 回届く)。
        #[cfg(feature = "cuda")]
        #[test]
        fn mix_gpu_gate_wiring_reaches_cuda_entry() {
            let _l = mix_gpu_test_lock();
            let _g = MixGpuGuard::new();
            // hc_mix_quant は mix_kprof::rowpar のラベルを立てる (= 計上を積む) ので、
            // rowpar::stats を before/after 差分で数えるテスト群と同じロックで
            // 直列化する (実 decode 形の GEMV を呼ぶテストの共通の規約)。
            let _m = moe_test_guard();
            let (hp, norm, down, up, inject, res) = mix_bf16_case(2, 32, 8);
            let attempts = || crate::cuda::mix_gpu_attempts();
            set_mix_gpu_for_test(Some(false));
            let before = attempts();
            hc_mix_quant(&res, &norm, &down, &up, Some(&inject), &hp).expect("mix (off)");
            assert_eq!(attempts(), before, "off なのに cuda 入口に届いた");
            set_mix_gpu_for_test(Some(true));
            let before = attempts();
            hc_mix_quant(&res, &norm, &down, &up, Some(&inject), &hp).expect("mix (on)");
            assert!(
                attempts() >= before + 2,
                "on なのに cuda 入口に届いていない (down + up = 2 回 / 呼び出し)"
            );
        }

        /// task 110: GPU 経路が実行する演算列 (= BF16 -> f32 展開 + 1 行の k 昇順
        /// 逐次内積 `acc += row[i] * x[i]`、積と加算の丸め分離) が、hc mix が CPU
        /// で通る BF16 経路 (2 行カーネル指定 = 実 decode の呼び出し形) と
        /// **全要素ビット一致**すること。
        ///
        /// 参照は既存の独立実装 [`bf16_gemv_seq`] (どの GEMV 経路も通らない)。
        /// 実形 (down [10240->320] / up [320->10240]) + 4 で割り切れない端数 k
        /// (GPU カーネル `dno_matvec_f32` の 4 倍展開ループと端数ループの両方)。
        /// カーネルは `mul.rn.f32` + `add.rn.f32` のみ・fma.rn 0 個・.ftz 0 個・
        /// k 昇順直列 (PTX を静的確認済み) なので、この逐次列とビット一致する。
        /// 実 GPU での最終確認は cuda.rs の cuda_mix_bf16_bit_matches_classic。
        #[test]
        fn mix_gpu_bf16_real_shapes_match_sequential_ref_bitwise() {
            // 実 decode の呼び出し形 (2row 指定) は rowpar region を立てるので、
            // プロセス共通の ROW_PAR_REGIONS を before/after 差分で数えるテスト
            // (expert_loop_one_region_per_layer 等、moe_test_guard を取る) と
            // 同じロックで直列化する。
            use crate::model::moe_expert::set_bf16_exact_for_test;
            let _g = moe_test_guard();
            // task 119: 既定が FMA (逐次参照と非ビット一致) になったので、このテストは
            // exact を強制して従来カーネル (GPU カーネル dno_matvec_f32 と同じ逐次列)
            // との一致を見る (期待値は変えない)。
            set_bf16_exact_for_test(Some(true));
            let cases: [(&str, usize, usize, u32, usize); 4] = [
                ("hc_down 実形", 10240, 320, 3, HC_DOWN_ROW_PAR_MIN),
                ("hc_up 実形", 320, 10240, 5, 0),
                ("端数 k=13", 13, 5, 7, HC_DOWN_ROW_PAR_MIN),
                ("端数 k=5", 5, 13, 11, 0),
            ];
            for (label, k, n, seed, row_par_min) in cases {
                let bytes = bf16_weights(k, n, seed);
                let lin =
                    GgufLinear::from_slice(GgmlType::Bf16, k, n, &bytes).expect("GgufLinear (Bf16)");
                let x: Vec<f32> = (0..k)
                    .map(|i| (((i * 31) % 97) as f32 - 48.0) * 0.01)
                    .collect();
                let want = bf16_gemv_seq(&lin, &x);
                let mut out = vec![0f32; n];
                lin.gemv_label_par_2row(label, &x, &mut out, row_par_min)
                    .expect("BF16 GEMV (2row 実経路)");
                assert_bitwise_eq(&out, &want, &format!("{label} (k={k}, n={n})"));
            }
            set_bf16_exact_for_test(None);
        }

        /// P84: 2 行同時カーネル ([`hc_mix_quant_fast`] の既定経路) が classic
        /// カーネルと**ビット一致**する。
        ///
        /// 1 行 / 実形の down (320 行) と up (10240 行) / 端数 (奇数行) / 門番未満
        /// (行並列が立たない逐次) を比べ、さらに `gemv_batch` の 1 トークン
        /// (prefill と同じ経路) とも比べる。行ごとの累積順序は 2 行化で変わらない
        /// ので、全部 `to_bits()` で一致するはず。
        #[test]
        fn hc_mix_2row_kernel_matches_classic_bitexact() {
            // (k, n): down 実形 / up 実形 / 1 行 / 端数 (偶数側が切り捨てられる奇数) /
            // 門番未満の逐次。k は 32 の倍数 (Q8_0 のブロック)。
            let shapes = [
                ("down 実形", 10240usize, 320usize),
                ("up 実形", 320, 10240),
                ("1 行", 10240, 1),
                ("端数 (down)", 10240, 321),
                ("端数 (up)", 320, 10241),
                ("門番未満", 10240, 63),
            ];
            for (name, k, n) in shapes {
                let lin = GgufLinear::from_slice(GgmlType::Q8_0, k, n, &q8_0_weights(k, n, 11))
                    .expect("GgufLinear::from_slice (Q8_0)")
                    .with_fast_q8(true);
                let x: Vec<f32> =
                    (0..k).map(|i| (((i * 31) % 97) as f32 - 48.0) * 0.011).collect();
                // classic (P84 までの入口。`gemv_label_par` は門番 64 を明示、
                // `gemv_label` は既定門番 512) と 2 行同時版。
                let mut a = vec![0.0f32; n];
                lin.gemv_label_par("hc_2row_test_classic", &x, &mut a, 64)
                    .expect("gemv_label_par (classic)");
                let mut b = vec![0.0f32; n];
                lin.gemv_label_par_2row("hc_2row_test_2row", &x, &mut b, 64)
                    .expect("gemv_label_par_2row");
                assert_bitwise_eq(&b, &a, &format!("2row vs classic ({name})"));
                let mut c = vec![0.0f32; n];
                lin.gemv_label("hc_2row_test_label", &x, &mut c)
                    .expect("gemv_label (classic 既定門番)");
                assert_bitwise_eq(&b, &c, &format!("2row vs gemv_label ({name})"));
                assert!(
                    a.iter().any(|v| *v != 0.0),
                    "{name}: 出力が全 0 (経路の生死が分からない)"
                );
                // batch (`gemv_batch` = prefill の経路) の 1 トークンとも一致。
                let mut bat = vec![0.0f32; n];
                gemv_batch(&lin, &x, 1, &mut bat).expect("gemv_batch (nt=1)");
                assert_bitwise_eq(&bat, &a, &format!("batch vs classic ({name})"));
            }
        }

        /// P84: mix 全体でも 2 行同時カーネル (既定 on の高速経路) が旧経路と
        /// ビット一致することを**実形** (hc=4 / e=2560 / lr=320: down は
        /// 10240 x 320、up は 320 x 10240) で確かめる。
        /// `hc_mix_fast_matches_ref_bitexact` が小さい形、このテストが実形を通る。
        #[test]
        fn hc_mix_2row_fast_matches_ref_bitexact_realish() {
            let (hp, norm, down, up, inject, res) = mix_q8_case(4, 2560, 320);
            assert_eq!(down.k(), 4 * 2560, "down の入力長 (hc_dim)");
            assert_eq!(down.n(), 320, "down の行数 (hc_lr)");
            assert_eq!(up.n(), 4 * 2560, "up の出力長 (hc_dim)");
            let (m_fast, i_fast) =
                hc_mix_quant(&res, &norm, &down, &up, Some(&inject), &hp).expect("fast");
            let (m_ref, i_ref) =
                hc_mix_quant_ref(&res, &norm, &down, &up, Some(&inject), &hp).expect("ref");
            assert_bitwise_eq(&m_fast, &m_ref, "2row 高速経路 vs 旧経路 (mixed)");
            assert_bitwise_eq(&i_fast, &i_ref, "2row 高速経路 vs 旧経路 (inject)");
        }

        /// `DENNO_MIX_KPROF` の区間計測が全区間で回る (強制 on で確認)。
        #[test]
        fn mix_kprof_counts_every_region() {
            let (hp, norm, down, up, inject, res) = mix_q8_case(2, 32, 32);
            mix_kprof::set_for_test(true);
            mix_kprof::reset();
            for _ in 0..64 {
                let _ = hc_mix_quant_ref(&res, &norm, &down, &up, Some(&inject), &hp)
                    .expect("ref");
            }
            let line = mix_kprof::line().expect("強制 on なのに line() が None");
            assert!(line.contains("[mix-kprof] calls="), "行の形が違う: {line}");
            for (name, slot) in [
                ("gn", mix_kprof::GN),
                ("down", mix_kprof::DOWN),
                ("lo", mix_kprof::LO),
                ("up", mix_kprof::UP),
                ("mixed", mix_kprof::MIXED),
                ("inj", mix_kprof::INJ),
                ("inj_apply", mix_kprof::INJ_APPLY),
                ("total", mix_kprof::TOTAL),
            ] {
                assert!(
                    mix_kprof::ns_of(slot) > 0,
                    "区間 {name} が 0 ns (計測が通っていない): {line}"
                );
            }
            mix_kprof::reset();
            mix_kprof::set_for_test(false);
        }

        /// task 107: decode (nt=1) の mix 実形状 (BF16, hc=4 / e=2560 / lr=320:
        /// swift15 の実形) で `[mix-kprof]` の 1 行を出す。
        ///
        /// 実機の `DENNO_MIX_KPROF=1` (serve の `prof_flash_q4` が 1 トークン末尾で
        /// 出す行) と同じ形式をコンテナで出す。重みは BF16 の決定列で、
        /// - `mats=32`: 重み 32 組を轮回して L3 から追い出した DRAM 条件
        /// - `mats=1`:  同じ組の再読み (L3 常駐条件)
        /// の 2 通り。呼び出し数は 1 トークンぶん (`2 * n_layer + 1` = 129 の 3 倍)。
        ///
        /// 値の検査ではないので assert は行の形だけ。数値は実行ごとに動く
        /// (共有ホスト)。`hc_mix_quant_fast` が `mix_kprof` を積むのは FORCE のとき
        /// だけなので、env は汚さない。
        ///
        /// `cargo test --release -p denno-engine --lib --features cuda -- \
        ///  --ignored --nocapture mix_kprof_decode_real_shape_dump`
        #[test]
        #[ignore = "実形状の CPU マイクロベンチ (手動)"]
        fn mix_kprof_decode_real_shape_dump() {
            use crate::model::linear_gguf::par_test_lock;
            let _g = par_test_lock();
            let hp = HyperParams { hc: 4, n_embd: 2560, low_rank: 320, eps: 1e-6 };
            let (e, hc, lr) = (hp.n_embd, hp.hc, hp.low_rank);
            let hc_dim = hc * e;

            // BF16 重み (決定的。f32 -> bf16 は 1 回の丸めなのでバイト列は環境に依らない)。
            let bf16_bytes = |k: usize, n: usize, seed: u32| -> Vec<u8> {
                let mut b = vec![0u8; k * n * 2];
                for (i, w) in b.chunks_exact_mut(2).enumerate() {
                    let v = (((i.wrapping_mul(2654435761) ^ seed as usize) % 19) as f32 - 9.0) * 0.12;
                    let bits = (v.to_bits() >> 16) as u16;
                    w.copy_from_slice(&bits.to_le_bytes());
                }
                b
            };
            let mk = |k: usize, n: usize, seed: u32| {
                GgufLinear::from_slice(GgmlType::Bf16, k, n, &bf16_bytes(k, n, seed))
                    .expect("GgufLinear::from_slice (Bf16)")
            };
            let slots: Vec<(GgufLinear, GgufLinear, GgufLinear)> = (0..32usize)
                .map(|s| (mk(hc_dim, lr, s as u32 * 3 + 1), mk(lr, hc_dim, s as u32 * 3 + 2), mk(hc_dim, hc, s as u32 * 3 + 3)))
                .collect();
            let res = HyperStreams {
                data: (0..hc_dim).map(|i| (((i * 11) % 17) as f32 - 8.0) * 0.05).collect(),
                hc,
                n_embd: e,
            };
            let norm: Vec<f32> = (0..hc_dim).map(|i| 0.8 + (i % 7) as f32 * 0.05).collect();

            let calls = 129usize * 3; // 1 トークン = 2 * n_layer + 1 呼び出しの 3 トークンぶん
            mix_kprof::set_for_test(true);
            // ウォームアップ (プール起動とページ立ち上げ)。積んだ数値は捨てる。
            for s in 0..8 {
                let (down, up, inject) = &slots[s];
                let _ = hc_mix_quant_fast(&res, &norm, down, up, Some(inject), &hp, false).expect("warm");
            }
            for (mats, slot_pick) in [("mats=32 (DRAM)", None), ("mats=1  (L3)", Some(0usize))] {
                mix_kprof::reset();
                for c in 0..calls {
                    let idx = slot_pick.unwrap_or(c % slots.len());
                    let (down, up, inject) = &slots[idx];
                    let _ = hc_mix_quant_fast(&res, &norm, down, up, Some(inject), &hp, false)
                        .expect("fast");
                }
                let line = mix_kprof::line().expect("強制 on なのに line() が None");
                println!("\n=== decode mix 実形 {mats} calls={calls} ===\n{line}");
                mix_kprof::reset();
            }
            mix_kprof::set_for_test(false);
        }

        /// task 107: decode (nt=1) の mix 実形状でカーネル幾何 (チャンク数 /
        /// 行グループ幅 / プリフェッチ距離) の交互 A/B を取る。
        ///
        /// 全 variant を 1 ラウンドに交互に回して最小値を取る (共有ホストの
        /// ラン間ドリフトを消す)。呼び出しは `hc_mix_quant_fast` 1 本道
        /// (gn -> down -> lo -> up -> mixed -> inj) で、重み 32 組を轮回した
        /// DRAM 条件。数値は実行ごとに動く ( assert 無しの計測テスト)。
        ///
        /// `cargo test --release -p denno-engine --lib --features cuda -- \
        ///  --ignored --nocapture bench_mix_decode_call_ab`
        #[test]
        #[ignore = "実形状の CPU マイクロベンチ (手動)"]
        fn bench_mix_decode_call_ab() {
            use crate::model::linear_gguf::par_test_lock;
            use crate::model::moe_expert::{
                set_bf16_chunk_for_test, set_bf16_group_for_test, set_bf16_pf_for_test,
            };
            let _g = par_test_lock();
            let hp = HyperParams { hc: 4, n_embd: 2560, low_rank: 320, eps: 1e-6 };
            let (e, hc, lr) = (hp.n_embd, hp.hc, hp.low_rank);
            let hc_dim = hc * e;
            let bf16_bytes = |k: usize, n: usize, seed: u32| -> Vec<u8> {
                let mut b = vec![0u8; k * n * 2];
                for (i, w) in b.chunks_exact_mut(2).enumerate() {
                    let v = (((i.wrapping_mul(2654435761) ^ seed as usize) % 19) as f32 - 9.0) * 0.12;
                    let bits = (v.to_bits() >> 16) as u16;
                    w.copy_from_slice(&bits.to_le_bytes());
                }
                b
            };
            let mk = |k: usize, n: usize, seed: u32| {
                GgufLinear::from_slice(GgmlType::Bf16, k, n, &bf16_bytes(k, n, seed))
                    .expect("GgufLinear::from_slice (Bf16)")
            };
            let slots: Vec<(GgufLinear, GgufLinear, GgufLinear)> = (0..32usize)
                .map(|s| {
                    (
                        mk(hc_dim, lr, s as u32 * 3 + 1),
                        mk(lr, hc_dim, s as u32 * 3 + 2),
                        mk(hc_dim, hc, s as u32 * 3 + 3),
                    )
                })
                .collect();
            let res = HyperStreams {
                data: (0..hc_dim).map(|i| (((i * 11) % 17) as f32 - 8.0) * 0.05).collect(),
                hc,
                n_embd: e,
            };
            let norm: Vec<f32> = (0..hc_dim).map(|i| 0.8 + (i % 7) as f32 * 0.05).collect();
            let bytes_per_call = ((hc_dim * lr + lr * hc_dim + hc_dim * hc) * 2) as f64;

            // (名前, chunk, group, pf)。Some は明示値 (従来幾何は G=8/pf=0)。
            let variants: [(&str, Option<usize>, Option<usize>, Option<usize>); 8] = [
                ("base (G=8 pf=0)", None, Some(8), Some(0)),
                ("default (G=4 pf=8)", None, None, None),
                ("G=4 pf=2", None, Some(4), Some(2)),
                ("G=4 pf=4", None, Some(4), Some(4)),
                ("G=4 pf=12", None, Some(4), Some(12)),
                ("G=4 pf=24", None, Some(4), Some(24)),
                ("G=4 pf=48", None, Some(4), Some(48)),
                ("G=4 pf=0", None, Some(4), Some(0)),
            ];
            let reps = 16usize;
            let per = 43usize; // 1 variant あたりの呼び出し数 (129 の 1/3)
            let mut best = vec![f64::INFINITY; variants.len()];
            let mut seq = 0usize;
            // ウォームアップ (プール / ページ)。
            for s in 0..8 {
                let (down, up, inject) = &slots[s];
                let _ = hc_mix_quant_fast(&res, &norm, down, up, Some(inject), &hp, false).expect("warm");
            }
            for rep in 0..reps {
                // 順序を rep ごとに回転する (固定順の系統バイアスを消す)。
                for k in 0..variants.len() {
                    let vi = (k + rep) % variants.len();
                    let (_, chunk, group, pf) = variants[vi];
                    set_bf16_chunk_for_test(chunk);
                    set_bf16_group_for_test(group);
                    set_bf16_pf_for_test(pf);
                    let t = std::time::Instant::now();
                    for _ in 0..per {
                        let (down, up, inject) = &slots[seq % slots.len()];
                        seq += 1;
                        let _ = hc_mix_quant_fast(&res, &norm, down, up, Some(inject), &hp, false)
                            .expect("fast");
                    }
                    let ms = t.elapsed().as_secs_f64() * 1e3 / per as f64;
                    best[vi] = best[vi].min(ms);
                }
            }
            set_bf16_chunk_for_test(None);
            set_bf16_group_for_test(None);
            set_bf16_pf_for_test(None);
            println!(
                "\n=== decode mix 実形 A/B (reps={reps} x {per} calls, DRAM 回転, bytes/call={:.2} MB) ===",
                bytes_per_call / 1e6
            );
            for (i, (name, ..)) in variants.iter().enumerate() {
                println!(
                    "{:<26} best={:7.4} ms/call ({:5.1} GB/s)  129 呼び出し = {:6.1} ms/tok",
                    name,
                    best[i],
                    bytes_per_call / (best[i] * 1e-3) / 1e9,
                    best[i] * 129.0
                );
            }
        }

        /// task 87: `attn_kprof` の 1 行組み立て (calls / ms / gpu 発数)。
        ///
        /// 強制 on で note を積み、serve (`prof_flash_q4`) が読む形の行が
        /// 出ることを確かめる。`gpu` は cuda ビルドでは「直前の線形呼び出し」の
        /// 実行先を数えるだけ (テストでは GPU を回さない) ので、上限 (calls)
        /// だけを見る。終わりに必ず FORCE を戻す (並行する hc_mix 系テストが
        /// env の解釈に従うことを保つ)。
        #[test]
        fn attn_kprof_line_counts_calls_and_gpu_hits() {
            // ssm_decode_fills_attn_kprof_slots と同じカウンタを使うので順番に走らせる。
            let _g = crate::model::moe_expert::moe_test_lock();
            crate::model::qwen4exp::attn_kprof::set_for_test(true);
            crate::model::qwen4exp::attn_kprof::reset();
            let t = std::time::Instant::now();
            std::thread::sleep(std::time::Duration::from_micros(50));
            crate::model::qwen4exp::attn_kprof::note_qkv(t);
            crate::model::qwen4exp::attn_kprof::note_gate(t);
            attn_kprof::note_out(t);
            let line = crate::model::qwen4exp::attn_kprof::line().expect("強制 on なのに line() が None");
            assert!(
                line.starts_with("[attn-kprof] | qkv calls=1"),
                "行の頭 (qkv calls) が違う: {line}"
            );
            assert!(line.contains("gate calls=1"), "gate の calls が無い: {line}");
            assert!(line.contains("out calls=1"), "out の calls が無い: {line}");
            // gpu は 0..=calls の範囲 (cuda ビルドでは直前の線形呼び出し次第)。
            for name in ["qkv", "gate", "out"] {
                let part = line
                    .split(&format!("{name} calls=1 "))
                    .nth(1)
                    .unwrap_or_default();
                let gpu: u64 = part
                    .split(" gpu=")
                    .nth(1)
                    .and_then(|s| s.split(' ').next())
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                assert!(gpu <= 1, "{name}: gpu 発数 {gpu} が calls を超えた: {line}");
            }
            attn_kprof::reset();
            let line2 = attn_kprof::line().expect("reset 後も on なら行は出る");
            assert!(
                line2.contains("qkv calls=0 ms=0.0"),
                "reset が効いていない: {line2}"
            );
            crate::model::qwen4exp::attn_kprof::set_for_test(false);
            assert!(crate::model::qwen4exp::attn_kprof::line().is_none(), "off なのに行が出る");
        }

        /// P84: 行並列 region のカウンタが down / up で積まれる
        /// (実機で「発火の有無・チャンク数・実効並列度」が読めることの証拠)。
        ///
        /// down は 320 行 (門番 [`HC_DOWN_ROW_PAR_MIN`] = 64 以上)、up は 640 行
        /// (既定門番 = moe_expert の `PAR_MIN_ROWS` = 512 以上) の形で
        /// 1 呼び出し回す (実形 27B では down=320 / up=10240 行。up は既定門番で発火)。
        /// inject にはラベルを立てないので、region 数が down=1 / up=1 のままで
        /// あることも同時に確認する (expert GEMV 等が混ざらないことの証拠)。
        #[test]
        fn mix_kprof_rowpar_counts_down_and_up() {
            use mix_kprof::rowpar;
            let (hp, norm, down, up, inject, res) = mix_q8_case(2, 320, 320);
            mix_kprof::set_for_test(true);
            mix_kprof::reset();
            let _ = hc_mix_quant(&res, &norm, &down, &up, Some(&inject), &hp).expect("fast");
            let (rd, fd, cd, _, _) = rowpar::stats(rowpar::L_DOWN as usize - 1);
            let (ru, fu, cu, _, _) = rowpar::stats(rowpar::L_UP as usize - 1);
            assert_eq!(rd, 1, "down region が 1 回計上されていない");
            assert_eq!(fd, 1, "down の行並列が発火していない (320 行 >= 門番 64)");
            assert!(cd >= 2, "down が 1 チャンクしか割れていない (cd={cd})");
            assert_eq!(ru, 1, "up region が 1 回計上されていない");
            assert_eq!(fu, 1, "up の行並列が発火していない (640 行 >= 既定門番 512)");
            assert!(cu >= 2, "up が 1 チャンクしか割れていない (cu={cu})");
            // 1 行の内訳に par 部が載る (serve が読む形)。
            let line = mix_kprof::line().expect("強制 on なのに line() が None");
            assert!(
                line.contains("down regions=1 fired=1"),
                "行の par 部が違う: {line}"
            );
            assert!(
                line.contains("up regions=1 fired=1"),
                "行の par 部が違う: {line}"
            );
            mix_kprof::reset();
            mix_kprof::set_for_test(false);
        }

        /// task 96: 発火 region の skew 3 項目 (`t_first=` / `spread=` / `tstart=`) が
        /// `[mix-kprof]` の 1 行に載る。
        ///
        /// down (320 行 >= 門番 64) と up (640 行 >= 既定門番 512) が 1 region ずつ
        /// 発火する形で mix を 1 回回し、(a) 項目が出ること、(b) `skew()` の生の値が
        /// 構造上の制約 (最初の完了 <= 発火 wall、開始オフセットの和 <= チャンク数 ×
        /// 発火 wall) を満たすことを確かめる。値の大きさ自体はタイミング依存なので
        /// 見ない (ホストで変わる)。
        #[test]
        fn mix_kprof_rowpar_skew_fields_on_fired_region() {
            use mix_kprof::rowpar;
            let (hp, norm, down, up, inject, res) = mix_q8_case(2, 320, 320);
            mix_kprof::set_for_test(true);
            mix_kprof::reset();
            let _ = hc_mix_quant(&res, &norm, &down, &up, Some(&inject), &hp).expect("fast");
            let line = mix_kprof::line().expect("強制 on なのに line() が None");
            // 生の値は reset の前に読む (reset は rowpar も 0 に戻す)。
            let (fd, _, cd, _, _) = rowpar::stats(rowpar::L_DOWN as usize - 1);
            let (fe_d, ss_d, fw_d) = rowpar::skew(rowpar::L_DOWN as usize - 1);
            mix_kprof::reset();
            mix_kprof::set_for_test(false);

            for name in ["down", "up"] {
                // rowpar 部だけを切り出す (region 計時の "{name} {ns} ns" と
                // gemv の "{name} calls=" も "{name} " で始まるので
                // "regions=" を含む位置で切る)。
                let seg = line
                    .split(&format!(" | {name} regions="))
                    .nth(1)
                    .unwrap_or_default();
                assert!(
                    seg.starts_with("1 fired=1") && seg.contains(" t_first="),
                    "{name}: skew 項目が載っていない: {line}"
                );
                assert!(seg.contains(" spread="), "{name}: spread が無い: {line}");
                assert!(seg.contains(" tstart="), "{name}: tstart が無い: {line}");
            }
            // 生の値の構造制約: 最初のチャンク完了は発火 wall 以内、開始オフセットの
            // 和はチャンク数 x 発火 wall 以内 (各チャンクの開始 + 実時間 <= wall)。
            assert_eq!(fd, 1, "down の発火 region が 1 でない");
            assert!(fe_d > 0, "down の t_first が 0 (skew が積まれていない)");
            assert!(
                fe_d <= fw_d,
                "down の最初の完了 {fe_d} ns が発火 wall {fw_d} ns を超えた"
            );
            assert!(
                ss_d <= cd.saturating_mul(fw_d),
                "down の開始オフセット和 {ss_d} が chunks={cd} x wall={fw_d} を超えた"
            );
        }

        /// task 96: skew の表示書式 (ns -> us、チャンク平均)。`record` を直接積んで
        /// `[mix-kprof]` 行の部分文字列を確かめる (タイミング非依存の固定値)。
        /// fired=1 / chunks=4 / wall=250us / comp=700us / t_first=150us /
        /// start_sum=120us -> tstart = 120/4 = 30us、spread = 250-150 = 100us。
        #[test]
        fn mix_kprof_rowpar_skew_line_format_is_deterministic() {
            use mix_kprof::rowpar;
            mix_kprof::set_for_test(true);
            mix_kprof::reset();
            rowpar::record(
                rowpar::L_DOWN,
                true,
                4,
                250_000,
                700_000,
                8,
                150_000,
                120_000,
            );
            let line = mix_kprof::line().expect("強制 on なのに line() が None");
            mix_kprof::reset();
            mix_kprof::set_for_test(false);
            let seg = line
                .split(" | down regions=")
                .nth(1)
                .unwrap_or_default();
            assert!(
                seg.contains("wall=250us comp=700us par_eff=2.8"),
                "既存項目の書式が違う: {seg}"
            );
            assert!(
                seg.contains("t_first=150us spread=100us tstart=30us"),
                "skew 項目の書式が違う: {seg}"
            );
        }

        /// task 88: down / up GEMV の実行先内訳 (calls / ms / gpu / ty) が
        /// `[mix-kprof]` の 1 行に載る ([`attn_kprof`] と同じ形式)。
        ///
        /// 強制 on で note を積み、serve (`prof_flash_q4`) が読む形を確かめる。
        /// `gpu` は cuda ビルドで「直前の線形呼び出し」の実行先を数えるだけ
        /// (テストでは GPU を回さない) ので、上限 (calls) だけを見る。
        #[test]
        fn mix_kprof_gemv_line_counts_calls_type_and_gpu() {
            use crate::kquant::GgmlType;
            mix_kprof::set_for_test(true);
            mix_kprof::reset();
            let t = std::time::Instant::now();
            std::thread::sleep(std::time::Duration::from_micros(50));
            mix_kprof::gemv::note_down(Some(t), GgmlType::IQ4Xs);
            mix_kprof::gemv::note_up(Some(t), GgmlType::IQ4Xs);
            mix_kprof::gemv::note_inj(Some(t), GgmlType::Bf16);
            let line = mix_kprof::line().expect("強制 on なのに line() が None");
            assert!(
                line.contains("gemv down calls=1 ms="),
                "down の形が違う: {line}"
            );
            assert!(line.contains("ty=IQ4_XS"), "dtype が無い: {line}");
            assert!(line.contains("gemv up calls=1 ms="), "up の形が違う: {line}");
            // task 96: inj も同じ形式で載る (型は down/up と独立に表示)。
            assert!(line.contains("gemv inj calls=1 ms="), "inj の形が違う: {line}");
            assert!(line.contains("ty=BF16"), "inj の dtype が無い: {line}");
            // gpu は 0..=calls の範囲 (cuda ビルドでは直前の線形呼び出し次第)。
            let part = line
                .split("gemv down calls=1 ")
                .nth(1)
                .unwrap_or_default();
            let gpu: u64 = part
                .split(" gpu=")
                .nth(1)
                .and_then(|s| s.split(' ').next())
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            assert!(gpu <= 1, "down: gpu 発数 {gpu} が calls を超えた: {line}");
            // reset で戻る (ty も未積算の "-" に戻る)。
            mix_kprof::reset();
            let line2 = mix_kprof::line().expect("reset 後も on なら行は出る");
            assert!(
                line2.contains("gemv down calls=0 ms=0.0"),
                "reset が効いていない: {line2}"
            );
            assert!(line2.contains("ty=-"), "reset で ty が残っている: {line2}");
            // off のときは note も積まない。
            mix_kprof::set_for_test(false);
            mix_kprof::gemv::note_down(Some(t), GgmlType::IQ4Xs);
            mix_kprof::set_for_test(true);
            mix_kprof::reset();
            let line3 = mix_kprof::line().expect("強制 on なのに line() が None");
            assert!(
                line3.contains("gemv down calls=0 ms=0.0"),
                "off で note が積まれた: {line3}"
            );
            mix_kprof::reset();
            mix_kprof::set_for_test(false);
        }

        /// task 88: hc down が Q8_0 でない型 (IQ4_XS) のときも、fast 経路の
        /// two_row バイパスが行並列 (門番 [`HC_DOWN_ROW_PAR_MIN`] = 64) に入り、
        /// 旧経路 (classic カーネル + 既定門番) とビット一致する。
        ///
        /// 実形の down (k=10240 / n=320) を IQ4_XS で作る。up は k=320
        /// (low_rank) で IQ4_XS が表現できない (256 要素/ブロック、
        /// `row_bytes` がエラー) ので Q8_0 のまま混在で比べる。
        /// 行並列の発火は `mix_kprof::rowpar` のラベル方式で見る (ラベルを立てて
        /// いない呼び出しは 1 件も積まない。プロセス共通の `ROW_PAR_REGIONS` を
        /// before/after で見ると、同じカウンタを使う他のテストと並列実行で干渉する)。
        #[test]
        fn hc_mix_iq4xs_down_takes_row_parallel_bitexact() {
            // 行並列の計数と BF16 の上書きはプロセス全体で 1 個なので、それを触る
            // テスト (moe_expert の bf16_tests など) と直列にする。
            let _g = crate::model::moe_expert::moe_test_lock();
            use mix_kprof::rowpar;
            let hc = 4usize;
            let e = 2560usize;
            let lr = 320usize;
            let hp = HyperParams { hc, n_embd: e, low_rank: lr, eps: 1e-6 };
            let hc_dim = hc * e;
            assert_eq!(hc_dim, 10240, "実形の down 入力 (hc_dim)");
            let norm: Vec<f32> = (0..hc_dim).map(|i| 0.8 + (i % 7) as f32 * 0.05).collect();
            let down = GgufLinear::from_slice(GgmlType::IQ4Xs, hc_dim, lr, &iq4_xs_weights(hc_dim, lr, 3))
                .expect("GgufLinear::from_slice (IQ4_XS down)");
            let up = GgufLinear::from_slice(GgmlType::Q8_0, lr, hc_dim, &q8_0_weights(lr, hc_dim, 5))
                .expect("GgufLinear::from_slice (Q8_0 up)")
                .with_fast_q8(true);
            let inject = GgufLinear::from_slice(GgmlType::Q8_0, hc_dim, hc, &q8_0_weights(hc_dim, hc, 7))
                .expect("GgufLinear::from_slice (Q8_0 inject)")
                .with_fast_q8(true);
            let res = HyperStreams {
                data: (0..hc_dim)
                    .map(|i| (((i * 11) % 17) as f32 - 8.0) * 0.05)
                    .collect(),
                hc,
                n_embd: e,
            };
            mix_kprof::set_for_test(true);
            mix_kprof::reset();
            let (m_fast, i_fast) =
                hc_mix_quant(&res, &norm, &down, &up, Some(&inject), &hp).expect("fast");
            // down の行並列が 1 region・1 発火 (320 行 >= 門番 64)。
            let (rd, fd, cd, _, _) = rowpar::stats(rowpar::L_DOWN as usize - 1);
            mix_kprof::reset();
            mix_kprof::set_for_test(false);
            assert_eq!(rd, 1, "IQ4_XS の down region が 1 回計上されていない");
            assert_eq!(fd, 1, "IQ4_XS の down が行並列に入っていない");
            assert!(cd >= 2, "IQ4_XS の down が 1 チャンクしか割れていない (cd={cd})");
            let (m_ref, i_ref) =
                hc_mix_quant_ref(&res, &norm, &down, &up, Some(&inject), &hp).expect("ref");
            assert_bitwise_eq(&m_fast, &m_ref, "IQ4_XS down (mixed)");
            assert_bitwise_eq(&i_fast, &i_ref, "IQ4_XS down (inject)");
            assert!(
                m_fast.iter().any(|v| *v != 0.0),
                "mixed が全 0 (経路の生死が分からない)"
            );
        }

        /// task 91: hc down/up の実型は BF16 (実機 P90 の mix-kprof `ty=BF16 gpu=0`)。
        /// 無量子型アームに落ちていた two_row 呼び出しが、行並列の行チャンク
        /// ([`for_each_row_chunk`]) に乗り、`mix_kprof::rowpar` の region / fired が
        /// 計上される。旧経路 ([`hc_mix_quant_ref`]) とビット一致する。
        ///
        /// 実形の down (k=10240 / n=320、門番 [`HC_DOWN_ROW_PAR_MIN`] = 64) と
        /// up (k=320 / n=10240、`row_par_min=0` は既定の門番 512 に委譲) を BF16 で
        /// 作る。up は k=320 が BF16 (1 要素 2 B) で表現できるので、down/up とも
        /// 実型に揃える (IQ4_XS のテストと違い混在にしない)。
        /// region 計上は `mix_kprof::rowpar` のラベル方式で見る ([`rowpar::stats`])。
        #[test]
        fn hc_mix_bf16_down_up_take_row_parallel_bitexact() {
            // 行並列の計数と BF16 の上書きはプロセス全体で 1 個なので、それを触る
            // テスト (moe_expert の bf16_tests など) と直列にする。
            let _g = crate::model::moe_expert::moe_test_lock();
            use mix_kprof::rowpar;
            let hc = 4usize;
            let e = 2560usize;
            let lr = 320usize;
            let hp = HyperParams { hc, n_embd: e, low_rank: lr, eps: 1e-6 };
            let hc_dim = hc * e;
            assert_eq!(hc_dim, 10240, "実形の down 入力 (hc_dim)");
            let norm: Vec<f32> = (0..hc_dim).map(|i| 0.8 + (i % 7) as f32 * 0.05).collect();
            let down =
                GgufLinear::from_slice(GgmlType::Bf16, hc_dim, lr, &bf16_weights(hc_dim, lr, 3))
                    .expect("GgufLinear::from_slice (BF16 down)");
            let up =
                GgufLinear::from_slice(GgmlType::Bf16, lr, hc_dim, &bf16_weights(lr, hc_dim, 5))
                    .expect("GgufLinear::from_slice (BF16 up)");
            let inject =
                GgufLinear::from_slice(GgmlType::Bf16, hc_dim, hc, &bf16_weights(hc_dim, hc, 7))
                    .expect("GgufLinear::from_slice (BF16 inject)");
            let res = HyperStreams {
                data: (0..hc_dim)
                    .map(|i| (((i * 11) % 17) as f32 - 8.0) * 0.05)
                    .collect(),
                hc,
                n_embd: e,
            };
            mix_kprof::set_for_test(true);
            mix_kprof::reset();
            let (m_fast, i_fast) =
                hc_mix_quant(&res, &norm, &down, &up, Some(&inject), &hp).expect("fast");
            let line = mix_kprof::line().expect("強制 on なのに line() が None");
            // down の行並列が 1 region・1 発火 (320 行 >= 門番 64)。
            let (rd, fd, cd, _, _) = rowpar::stats(rowpar::L_DOWN as usize - 1);
            // up の行並列が 1 region・1 発火 (10240 行 >= 既定門番 512)。
            let (ru, fu, cu, _, _) = rowpar::stats(rowpar::L_UP as usize - 1);
            mix_kprof::reset();
            mix_kprof::set_for_test(false);
            assert_eq!(rd, 1, "BF16 の down region が 1 回計上されていない");
            assert_eq!(fd, 1, "BF16 の down が行並列に入っていない");
            assert!(cd >= 2, "BF16 の down が 1 チャンクしか割れていない (cd={cd})");
            assert_eq!(ru, 1, "BF16 の up region が 1 回計上されていない");
            assert_eq!(fu, 1, "BF16 の up が行並列に入っていない");
            assert!(cu >= 2, "BF16 の up が 1 チャンクしか割れていない (cu={cu})");
            assert!(
                line.contains("ty=BF16"),
                "mix-kprof の型表示が BF16 でない: {line}"
            );
            let (m_ref, i_ref) =
                hc_mix_quant_ref(&res, &norm, &down, &up, Some(&inject), &hp).expect("ref");
            assert_bitwise_eq(&m_fast, &m_ref, "BF16 down/up (mixed)");
            assert_bitwise_eq(&i_fast, &i_ref, "BF16 down/up (inject)");
            assert!(
                m_fast.iter().any(|v| *v != 0.0),
                "mixed が全 0 (経路の生死が分からない)"
            );
        }

        /// task 184: `DENNO_MIX_DEC_DEV` が立つと fast 経路が **デバイス経路を試す**
        /// (GPU 無しでは入口で false になり、従来のホスト経路へ落ちる)。
        ///
        /// 「緑 = 分岐が発火した証拠」にしないため、入口のカウンタ ([`crate::cuda::
        /// mix_dev_attempts`]) が増えたことを数える。GPU が無い環境では従来経路の
        /// 値が ref とビット一致する (縮約はツリーに変わらない = デバイスは走らない)。
        #[cfg(feature = "cuda")]
        #[test]
        fn mix_dec_dev_gate_attempts_then_falls_back_without_gpu() {
            let _lk = mix_gpu_test_lock();
            let (hp, norm, down, up, inject, res) = mix_bf16_case(4, 320, 64);
            crate::cuda::mix_dev_attempts_reset();
            crate::cuda::mix_dec_dev_set_for_test(Some(true));
            let got = hc_mix_quant(&res, &norm, &down, &up, Some(&inject), &hp);
            let attempts = crate::cuda::mix_dev_attempts();
            crate::cuda::mix_dec_dev_set_for_test(None);
            let (m_fast, i_fast) = got.expect("fast");
            assert!(
                attempts >= 1,
                "DENNO_MIX_DEC_DEV を立てたのにデバイス経路の入口に来ていない"
            );
            // GPU が無い (= 入口で false) ときだけビット一致を主張できる。
            if crate::cuda::cuda_enabled() {
                return;
            }
            let (m_ref, i_ref) =
                hc_mix_quant_ref(&res, &norm, &down, &up, Some(&inject), &hp).expect("ref");
            assert_bitwise_eq(&m_fast, &m_ref, "device 不発の fallback (mixed)");
            assert_bitwise_eq(&i_fast, &i_ref, "device 不発の fallback (inject)");
            assert!(
                m_fast.iter().any(|v| *v != 0.0),
                "mixed が全 0 (経路の生死が分からない)"
            );
        }

        /// task 96: hc mix の inject (`hc_inject`, k=hc_dim / n=hc の小行列) が
        /// BF16 のとき、task 92 の AVX2 融合カーネルに**今日の経路のまま**乗ることの
        /// 契約 (mix-kprof の `gemv inj ty=` で実型が BF16 だったときに当てはまる)。
        ///
        /// down/up を Q8_0 (`BF16_FUSED_CALLS` に載らない型) にして inject だけ
        /// BF16 にすると、`hc_mix_quant` 1 回で融合カウンタがちょうど 1 増える
        /// (inject の GEMV 1 発だけが融合経路を通る。n=hc は行並列の門番未満なので
        /// 1 チャンク = 単独発火)。`set_bf16_fast_for_test(Some(false))` の classic
        /// (dequant + 逐次内積) と to_bits で一致する (BF16 -> f32 変換は正確で、
        /// 積和の順序も同じ。[`hc_mix_bf16_down_up_take_row_parallel_bitexact`] と
        /// 同じ根拠)。inject が F32 など BF16 以外の実型なら融合カーネルは
        /// 適用できない (REPORT の判定を参照)。
        #[test]
        fn hc_mix_bf16_inject_takes_fused_kernel_bitexact() {
            // 行並列の計数と BF16 の上書きはプロセス全体で 1 個なので、それを触る
            // テスト (moe_expert の bf16_tests など) と直列にする。
            let _g = crate::model::moe_expert::moe_test_lock();
            use crate::model::moe_expert::{
                set_bf16_exact_for_test, set_bf16_fast_for_test, BF16_FUSED_CALLS,
            };
            let hc = 4usize;
            let e = 2560usize;
            let lr = 320usize;
            let hp = HyperParams { hc, n_embd: e, low_rank: lr, eps: 1e-6 };
            let hc_dim = hc * e;
            let norm: Vec<f32> = (0..hc_dim).map(|i| 0.8 + (i % 7) as f32 * 0.05).collect();
            let down = GgufLinear::from_slice(GgmlType::Q8_0, hc_dim, lr, &q8_0_weights(hc_dim, lr, 3))
                .expect("GgufLinear::from_slice (Q8_0 down)")
                .with_fast_q8(true);
            let up = GgufLinear::from_slice(GgmlType::Q8_0, lr, hc_dim, &q8_0_weights(lr, hc_dim, 5))
                .expect("GgufLinear::from_slice (Q8_0 up)")
                .with_fast_q8(true);
            let inject =
                GgufLinear::from_slice(GgmlType::Bf16, hc_dim, hc, &bf16_weights(hc_dim, hc, 7))
                    .expect("GgufLinear::from_slice (BF16 inject)");
            let res = HyperStreams {
                data: (0..hc_dim)
                    .map(|i| (((i * 11) % 17) as f32 - 8.0) * 0.05)
                    .collect(),
                hc,
                n_embd: e,
            };
            let fused = || BF16_FUSED_CALLS.load(std::sync::atomic::Ordering::Relaxed);
            // task 119: 既定が FMA (classic と非ビット一致) になったので、融合 vs
            // classic のビット一致の契約は exact を強制して見る (期待値は変えない)。
            set_bf16_exact_for_test(Some(true));

            // 融合 on (既定): inject の GEMV 1 発だけ融合カウンタが増える。
            set_bf16_fast_for_test(Some(true));
            let before = fused();
            let (m_fast, i_fast) =
                hc_mix_quant(&res, &norm, &down, &up, Some(&inject), &hp).expect("fast");
            assert_eq!(
                fused() - before,
                1,
                "BF16 inject が融合カーネルを通っていない (down/up 分以外の増分が 1 でない)"
            );

            // 融合 off (classic): inject は dequant + 逐次内積に落ちる。値はビット一致。
            set_bf16_fast_for_test(Some(false));
            let (m_ref, i_ref) =
                hc_mix_quant(&res, &norm, &down, &up, Some(&inject), &hp).expect("classic");
            set_bf16_fast_for_test(None);
            assert_bitwise_eq(&i_fast, &i_ref, "BF16 inject (融合 vs classic)");
            assert_bitwise_eq(&m_fast, &m_ref, "BF16 inject case (mixed)");
            assert!(
                i_fast.iter().any(|v| *v != 0.0),
                "inject が全 0 (経路の生死が分からない)"
            );
            set_bf16_exact_for_test(None);
        }

        /// task 119: 既定 (FMA) の decode mix 経路が BF16 融合カーネルの FMA 版を
        /// 通ること、と exact (DENNO_BF16_EXACT=1 相当) で通らなくなること。
        ///
        /// 実形の BF16 down/up/inject で `hc_mix_quant` を 1 回回し、`BF16_FMA_CALLS`
        /// の差分で見る (緑 = 通った、ではないので counter を見る。down + up +
        /// inject の融合入口 3 呼び出し以上)。exact を強すると FMA カウンタは増えず、
        /// 融合入口のカウンタ (`BF16_FUSED_CALLS`) は増える = 逐次加算カーネルのまま。
        /// mixed / inject の値差は「絶対 1e-4 + 相対 1e-5」以内 (FMA は classic と
        /// ビット一致しない。moe_expert.rs の `bf16_tests` と同じ契約)。
        #[test]
        fn hc_mix_bf16_fma_fires_and_exact_gate_restores_sequential() {
            // BF16 の上書き (exact / fast) はプロセス全体で 1 個。触るテストは直列にする。
            let _bf16_lock = crate::model::moe_expert::moe_test_lock();
            use crate::model::moe_expert::{
                set_bf16_exact_for_test, BF16_FMA_CALLS, BF16_FUSED_CALLS,
            };
            use std::sync::atomic::Ordering::Relaxed;
            let hc = 4usize;
            let e = 2560usize;
            let lr = 320usize;
            let hp = HyperParams { hc, n_embd: e, low_rank: lr, eps: 1e-6 };
            let hc_dim = hc * e;
            let norm: Vec<f32> = (0..hc_dim).map(|i| 0.8 + (i % 7) as f32 * 0.05).collect();
            let down =
                GgufLinear::from_slice(GgmlType::Bf16, hc_dim, lr, &bf16_weights(hc_dim, lr, 3))
                    .expect("GgufLinear::from_slice (BF16 down)");
            let up =
                GgufLinear::from_slice(GgmlType::Bf16, lr, hc_dim, &bf16_weights(lr, hc_dim, 5))
                    .expect("GgufLinear::from_slice (BF16 up)");
            let inject =
                GgufLinear::from_slice(GgmlType::Bf16, hc_dim, hc, &bf16_weights(hc_dim, hc, 7))
                    .expect("GgufLinear::from_slice (BF16 inject)");
            let res = HyperStreams {
                data: (0..hc_dim)
                    .map(|i| (((i * 11) % 17) as f32 - 8.0) * 0.05)
                    .collect(),
                hc,
                n_embd: e,
            };

            // FMA (既定相当を明示)。
            set_bf16_exact_for_test(Some(false));
            let fma0 = BF16_FMA_CALLS.load(Relaxed);
            let (m_fma, i_fma) =
                hc_mix_quant(&res, &norm, &down, &up, Some(&inject), &hp).expect("fast");
            let fma_calls = BF16_FMA_CALLS.load(Relaxed) - fma0;
            assert!(
                fma_calls >= 3,
                "FMA カーネルが mix 経路で発火していない (BF16_FMA_CALLS 増分 {fma_calls})"
            );

            // exact (DENNO_BF16_EXACT=1 相当): FMA カウンタは増えない。
            set_bf16_exact_for_test(Some(true));
            let fma1 = BF16_FMA_CALLS.load(Relaxed);
            let fused0 = BF16_FUSED_CALLS.load(Relaxed);
            let (m_ex, i_ex) =
                hc_mix_quant(&res, &norm, &down, &up, Some(&inject), &hp).expect("exact");
            assert_eq!(
                BF16_FMA_CALLS.load(Relaxed),
                fma1,
                "exact で FMA カーネルが発火した"
            );
            assert!(
                BF16_FUSED_CALLS.load(Relaxed) > fused0,
                "exact で融合入口を通っていない"
            );
            set_bf16_exact_for_test(None);

            // 許容差 (mixed / inject の両方)。絶対 1e-4 + 相対 1e-5。
            for (tag, a, b) in [("mixed", &m_fma, &m_ex), ("inject", &i_fma, &i_ex)] {
                for (r, (fa, ex)) in a.iter().zip(b.iter()).enumerate() {
                    let tol = 1e-4 + 1e-5 * ex.abs();
                    assert!(
                        (fa - ex).abs() <= tol,
                        "{tag} 行 {r}: FMA {fa} と exact {ex} の差が許容 {tol} を超える"
                    );
                }
                assert!(a.iter().any(|v| *v != 0.0), "{tag} が全 0");
            }
        }

        /// task 91: BF16 実形の GEMV が、入口 (`gemv` / `gemv_label_par_2row` (two_row)
        /// / `gemv_label_par` (row_par_min 明示)) とスレッド数 (1 / 8) に依らず
        /// **to_bits で一致**する。
        ///
        /// down 実形 (k=10240 / n=320) と up 実形 (k=320 / n=10240)。参照は
        /// [`bf16_gemv_seq`] (エンジンの経路を通らない逐次実装)。`row_par_min`
        /// 明示版は down と同じ門番 64 を使う (up は two_row で既定門番 512 を通る)。
        #[test]
        fn hc_bf16_gemv_real_shape_bitexact_across_paths() {
            // BF16 の上書き (exact / fast) はプロセス全体で 1 個。触るテストは直列にする。
            let _bf16_lock = crate::model::moe_expert::moe_test_lock();
            use crate::model::moe_expert::set_bf16_exact_for_test;
            // task 119: 既定が FMA (逐次参照 bf16_gemv_seq と非ビット一致) になったので、
            // このテストは exact を強制して従来カーネルで見る (期待値は変えない)。
            set_bf16_exact_for_test(Some(true));
            for (name, k, n, min) in [
                ("down 実形", 10240usize, 320usize, HC_DOWN_ROW_PAR_MIN),
                ("up 実形", 320usize, 10240usize, 0usize),
            ] {
                let lin =
                    GgufLinear::from_slice(GgmlType::Bf16, k, n, &bf16_weights(k, n, 11))
                        .expect("GgufLinear::from_slice (BF16)");
                let x: Vec<f32> =
                    (0..k).map(|i| (((i * 31) % 97) as f32 - 48.0) * 0.011).collect();
                let want = bf16_gemv_seq(&lin, &x);
                assert!(want.iter().any(|v| *v != 0.0), "{name}: 参照が全 0");

                // 旧経路 (`gemv` = cpu_gemv_par の行チャンク。two_row / row_par_min 無し)。
                let mut a = vec![0f32; n];
                lin.gemv(&x, &mut a).expect("gemv (旧経路)");
                assert_bitwise_eq(&a, &want, &format!("BF16 {name}: 旧経路 vs 逐次参照"));

                // 新経路 (two_row 明示。row_par_min=0 は既定門番 512 に委譲)。
                let mut b = vec![0f32; n];
                lin.gemv_label_par_2row("hc_bf16_2row", &x, &mut b, min)
                    .expect("gemv_label_par_2row (two_row)");
                assert_bitwise_eq(&b, &want, &format!("BF16 {name}: two_row vs 逐次参照"));

                // 新経路 (row_par_min 明示のみ。down と同じ門番 64。up は行数が
                // 10240 なので 64 以上で発火する)。
                let mut c = vec![0f32; n];
                lin.gemv_label_par("hc_bf16_par", &x, &mut c, HC_DOWN_ROW_PAR_MIN)
                    .expect("gemv_label_par (row_par_min)");
                assert_bitwise_eq(&c, &want, &format!("BF16 {name}: row_par_min vs 逐次参照"));

                // スレッド数で値が動かない (1 と 8)。
                let run = |threads: usize| -> Vec<f32> {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build()
                        .expect("thread pool");
                    let mut out = vec![0f32; n];
                    pool.install(|| {
                        lin.gemv_label_par_2row("hc_bf16_2row", &x, &mut out, min)
                            .expect("gemv_label_par_2row")
                    });
                    out
                };
                assert_bitwise_eq(&run(1), &want, &format!("BF16 {name}: 1 スレッド"));
                assert_bitwise_eq(&run(8), &want, &format!("BF16 {name}: 8 スレッド"));
            }
            set_bf16_exact_for_test(None);
        }

        /// P86: 実機で起きていたこと (GPU GEMV が hc mix の down / up を奪い、
        /// CPU の行並列が一度も走らない = `regions=0`) を GPU 無しで再現し、
        /// two_row のバイパスで CPU に落ちることを確かめる (cuda ビルドのみ)。
        ///
        /// intercept は `maybe_gpu_gguf_gemv_into` を「GPU が取った」ことにする
        /// (out を書かない)。旧コードでは fast 経路の down / up がここで消えて
        /// mixed が作業域の残骸になる。修正後は 2 行カーネルで計算されるので
        /// 旧経路 (CPU) とビット一致し、rowpar region が down / up で 1 回ずつ
        /// 積まれる。inject は plain 入口のまま intercept に奪われるので比較しない
        /// (intercept は値を作らない。実機の GPU なら実値が入る)。
        #[cfg(feature = "cuda")]
        #[test]
        fn hc_mix_quant_reaches_cpu_2row_even_when_gpu_intercepts() {
            use crate::cuda::GemvInterceptGuard;
            use mix_kprof::rowpar;

            let (hp, norm, down, up, inject, res) = mix_q8_case(2, 320, 320);
            let (m_ref, _i_ref) =
                hc_mix_quant_ref(&res, &norm, &down, &up, Some(&inject), &hp).expect("ref");
            let before = crate::model::linear_gguf::TWO_ROW_CPU_CALLS
                .load(std::sync::atomic::Ordering::Relaxed);

            let _g = GemvInterceptGuard::new();
            mix_kprof::set_for_test(true);
            mix_kprof::reset();
            let (m, _i) = hc_mix_quant(&res, &norm, &down, &up, Some(&inject), &hp).expect("fast");
            assert_bitwise_eq(&m, &m_ref, "intercept 下の高速経路 (mixed)");
            let (rd, fd, ..) = rowpar::stats(rowpar::L_DOWN as usize - 1);
            let (ru, fu, ..) = rowpar::stats(rowpar::L_UP as usize - 1);
            assert_eq!((rd, fd), (1, 1), "down が CPU の行並列 region に届いていない");
            assert_eq!((ru, fu), (1, 1), "up が CPU の行並列 region に届いていない");
            assert_eq!(
                crate::model::linear_gguf::TWO_ROW_CPU_CALLS
                    .load(std::sync::atomic::Ordering::Relaxed),
                before + 2,
                "down / up の two_row 呼び出しが CPU に届いていない"
            );
            mix_kprof::reset();
            mix_kprof::set_for_test(false);
        }

        /// 実形状 (hc=4 / n_embd=2560 / low_rank=320 / Q8_0 + fast_q8) の
        /// `hc_mix_quant` 1 呼び出しと区間内訳。手動実行:
        /// `cargo test --release -p denno-engine --lib --features cuda -- --ignored --nocapture bench_hc_mix_quant`
        #[test]
        #[ignore = "実形状の CPU マイクロベンチ (手動)"]
        fn bench_hc_mix_quant_real_shape() {
            use crate::model::linear_gguf::par_test_lock;
            let _g = par_test_lock();
            let (hp, _norm, down, up, inject, res) = mix_q8_case(4, 2560, 320);
            let norm: Vec<f32> = (0..hp.hc * hp.n_embd)
                .map(|i| 0.9 + ((i % 13) as f32) * 0.01)
                .collect();
            let reps = 20usize;
            let best = |mut f: Box<dyn FnMut()>| -> f64 {
                let mut best = f64::INFINITY;
                for _ in 0..reps {
                    let t = std::time::Instant::now();
                    f();
                    best = best.min(t.elapsed().as_secs_f64() * 1e3);
                }
                best
            };
            // warm-up (スレッドプールと作業域を作る)
            for _ in 0..3 {
                let _ = hc_mix_quant(&res, &norm, &down, &up, Some(&inject), &hp).expect("warm");
            }
            println!(
                "\n=== hc_mix_quant 1 呼び出し (n_embd={} hc={} low_rank={}) ===",
                hp.n_embd, hp.hc, hp.low_rank
            );
            println!(
                "rayon threads = {}",
                std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "(default)".into())
            );

            let old = best(Box::new(|| {
                let _ = hc_mix_quant_ref(&res, &norm, &down, &up, Some(&inject), &hp).expect("ref");
            }));
            let new = best(Box::new(|| {
                let _ = hc_mix_quant(&res, &norm, &down, &up, Some(&inject), &hp).expect("fast");
            }));
            println!("旧経路 (MIX_FAST=0) {:8.3} ms/call", old);
            println!("高速経路 (既定)     {:8.3} ms/call  ({:.2}x)", new, old / new);

            // 区間内訳は 1 回ぶんの ns に直して出す (reps で割る)。
            let breakdown = |fast: bool| {
                mix_kprof::set_for_test(true);
                mix_kprof::reset();
                for _ in 0..reps {
                    let r = if fast {
                        hc_mix_quant(&res, &norm, &down, &up, Some(&inject), &hp)
                    } else {
                        hc_mix_quant_ref(&res, &norm, &down, &up, Some(&inject), &hp)
                    };
                    let _ = r.expect("breakdown");
                }
                println!(
                    "{}  [{} / {reps} 回]",
                    mix_kprof::line().expect("kprof が None"),
                    if fast { "高速経路" } else { "旧経路" }
                );
                for (label, slot) in [
                    ("gn", mix_kprof::GN),
                    ("down", mix_kprof::DOWN),
                    ("lo", mix_kprof::LO),
                    ("up", mix_kprof::UP),
                    ("mixed", mix_kprof::MIXED),
                    ("inj", mix_kprof::INJ),
                    ("inj_apply", mix_kprof::INJ_APPLY),
                    ("total", mix_kprof::TOTAL),
                ] {
                    println!(
                        "  {label:>9}: {:8.1} us/call",
                        mix_kprof::ns_of(slot) as f64 / reps as f64 / 1e3
                    );
                }
                mix_kprof::reset();
                mix_kprof::set_for_test(false);
            };
            breakdown(false);
            breakdown(true);
        }

        /// hyper のバッチ版が逐次版とビット一致する。
        ///
        /// nt = 1 / 3 は逐次ループ、nt = 64 は rayon 経路
        /// ([`hyper::HC_BATCH_PAR_MIN_TOKENS`]) に入る。どちらもトークン内の
        /// 演算順序は変えていないので `to_bits()` で一致する。
        #[test]
        fn hyper_batch_matches_seq_bitwise() {
            let hp = HyperParams {
                hc: 4,
                n_embd: 16,
                low_rank: 5,
                eps: 1e-6,
            };
            let e = hp.n_embd;
            let hc_dim = hp.hc * hp.n_embd;
            assert_eq!(hc_dim, 64, "テストの前提 (hc_dim = 64)");
            let norm: Vec<f32> = (0..hc_dim).map(|i| 0.5 + ((i * 7) % 11) as f32 * 0.1).collect();

            let make = |nt: usize| -> Vec<HyperStreams> {
                (0..nt)
                    .map(|t| HyperStreams {
                        data: (0..hc_dim)
                            .map(|i| (((i * 7 + t * 13) % 23) as f32 - 11.0) * 0.03)
                            .collect(),
                        hc: hp.hc,
                        n_embd: e,
                    })
                    .collect()
            };

            let par_before = par_calls();
            // task 368: decode (`nt == 1`) の mix norm は `DENNO_HC_NORM_QUAD` (既定 on)
            // で **4 本アキュムレータの二乗和**になる。意図した丸め変更なので、
            // 「逐次版とバッチ版がビット一致する」この契約は**ゲート off の経路**で
            // 検査する (on のときの差は数 ulp。decode の出力は lab で md5 SAME)。
            hyper::hc_norm_quad_set_for_test(Some(false));
            for &nt in &[1usize, 3, 64] {
                let ress = make(nt);

                // 1. grouped RMS 正規化
                let seq_xn: Vec<f32> = ress
                    .iter()
                    .flat_map(|r| hyper::grouped_rms_norm(r, &norm, &hp))
                    .collect();
                let bat_xn = hyper::grouped_rms_norm_batch(&ress, &norm, &hp);
                assert_bitwise_eq(&seq_xn, &bat_xn, &format!("grouped_rms_norm nt={nt}"));

                // 2. lo の後処理 (silu(v/hc))
                let lo_in: Vec<f32> = (0..nt * hp.low_rank)
                    .map(|i| (((i * 11) % 19) as f32 - 9.0) * 0.02)
                    .collect();
                let mut lo_seq = lo_in.clone();
                for t in 0..nt {
                    hyper::hc_lo_apply(&mut lo_seq[t * hp.low_rank..(t + 1) * hp.low_rank], &hp);
                }
                let mut lo_bat = lo_in.clone();
                hyper::hc_lo_apply_batch(&mut lo_bat, nt, &hp);
                assert_bitwise_eq(&lo_seq, &lo_bat, &format!("hc_lo_apply nt={nt}"));

                // 3. mixed (s の昇順に加算してから 1/hc)
                let gate: Vec<f32> = (0..nt * hc_dim)
                    .map(|i| (((i * 17) % 29) as f32 - 14.0) * 0.05)
                    .collect();
                let seq_mixed: Vec<f32> = (0..nt)
                    .flat_map(|t| {
                        hyper::hc_mixed(
                            &seq_xn[t * hc_dim..(t + 1) * hc_dim],
                            &gate[t * hc_dim..(t + 1) * hc_dim],
                            &hp,
                        )
                    })
                    .collect();
                let bat_mixed = hyper::hc_mixed_batch(&seq_xn, &gate, nt, &hp);
                assert_bitwise_eq(&seq_mixed, &bat_mixed, &format!("hc_mixed nt={nt}"));

                // 4. inject の後処理 (silu 無し)
                let inj_in: Vec<f32> = (0..nt * hp.hc)
                    .map(|i| (((i * 5) % 7) as f32 - 3.0) * 0.1)
                    .collect();
                let mut inj_seq = inj_in.clone();
                for t in 0..nt {
                    hyper::hc_inject_apply(&mut inj_seq[t * hp.hc..(t + 1) * hp.hc], &hp);
                }
                let mut inj_bat = inj_in.clone();
                hyper::hc_inject_apply_batch(&mut inj_bat, nt, &hp);
                assert_bitwise_eq(&inj_seq, &inj_bat, &format!("hc_inject_apply nt={nt}"));

                // 5. combine (逐次版は新しい HyperStreams を作る / バッチ版は in-place)
                let outs: Vec<f32> = (0..nt * e)
                    .map(|i| (((i * 23) % 31) as f32 - 15.0) * 0.01)
                    .collect();
                let mut seq_res = make(nt);
                for t in 0..nt {
                    let out = hyper::hc_combine(
                        &seq_res[t],
                        &outs[t * e..(t + 1) * e],
                        &inj_seq[t * hp.hc..(t + 1) * hp.hc],
                        &hp,
                    );
                    seq_res[t] = out;
                }
                let mut bat_res = make(nt);
                hyper::hc_combine_batch(&mut bat_res, &outs, &inj_bat, &hp);
                for t in 0..nt {
                    assert_bitwise_eq(
                        &seq_res[t].data,
                        &bat_res[t].data,
                        &format!("hc_combine nt={nt} token={t}"),
                    );
                }
                assert!(
                    bat_res.iter().flat_map(|r| r.data.iter()).any(|v| *v != 0.0),
                    "combine の出力が全 0 (経路の生死が分からない)"
                );
            }

            // nt >= 64 で rayon 経路に実際に入ったことを数える
            // (feature par のときだけ。off なら逐次で回るのが正しい)。
            #[cfg(feature = "par")]
            assert!(
                par_calls() > par_before,
                "nt=64 でバッチ版の rayon 経路に入っていない (HC_BATCH_PAR_CALLS が増えない)"
            );
            #[cfg(not(feature = "par"))]
            assert_eq!(par_calls(), par_before, "par off なのに並列カウンタが増えた");
            // task 368: 強制を戻す (--test-threads=1 ではスレッドを跨ぐ)。
            hyper::hc_norm_quad_set_for_test(None);
        }

        /// バッチ版が rayon 経路に入った回数 (テスト用カウンタ)。
        #[cfg(feature = "par")]
        fn par_calls() -> usize {
            hyper::HC_BATCH_PAR_CALLS.load(std::sync::atomic::Ordering::Relaxed)
        }

        #[cfg(not(feature = "par"))]
        fn par_calls() -> usize {
            0
        }

        /// マイクロベンチ (手動)。mix の**要素演算だけ**を旧ループ / バッチで比べる。
        ///
        /// 行列積 (gemv) は GPU 側の仕事なので含めない。実形 (hc=4, n_embd=2560,
        /// hc_dim=10240) で nt 別に 200 回まわし、ms/トークンを出す。
        ///
        /// `cargo test --release -p denno-engine --lib hc_mix_elementwise_microbench -- --ignored --nocapture --test-threads=1`
        #[test]
        #[ignore]
        fn hc_mix_elementwise_microbench() {
            let hp = HyperParams {
                hc: 4,
                n_embd: 2560,
                low_rank: 320,
                eps: 1e-6,
            };
            let e = hp.n_embd;
            let hc_dim = hp.hc * hp.n_embd; // 10240 (実形)
            let reps = 200usize;
            let norm: Vec<f32> = (0..hc_dim).map(|i| 0.9 + ((i % 13) as f32) * 0.01).collect();

            println!("reps={reps} hc_dim={hc_dim} low_rank={}", hp.low_rank);
            println!("nt\told_ms_per_tok\tnew_ms_per_tok\tratio(old/new)");
            for &nt in &[1usize, 8, 64, 512] {
                let base: Vec<Vec<f32>> = (0..nt)
                    .map(|t| {
                        (0..hc_dim)
                            .map(|i| (((i * 7 + t * 13) % 23) as f32 - 11.0) * 0.03)
                            .collect()
                    })
                    .collect();
                let gate: Vec<f32> = (0..nt * hc_dim)
                    .map(|i| (((i * 17) % 29) as f32 - 14.0) * 0.05)
                    .collect();
                let lo_in: Vec<f32> = (0..nt * hp.low_rank)
                    .map(|i| (((i * 11) % 19) as f32 - 9.0) * 0.02)
                    .collect();
                let inj_in: Vec<f32> = (0..nt * hp.hc)
                    .map(|i| (((i * 5) % 7) as f32 - 3.0) * 0.1)
                    .collect();
                let outs: Vec<f32> = (0..nt * e)
                    .map(|i| (((i * 23) % 31) as f32 - 15.0) * 0.01)
                    .collect();

                // 旧: トークンごとに関数を呼ぶ (hc_mix_quant_batch の元の形)
                let mut ress: Vec<HyperStreams> = base
                    .iter()
                    .map(|d| HyperStreams {
                        data: d.clone(),
                        hc: hp.hc,
                        n_embd: e,
                    })
                    .collect();
                let mut lo_buf = lo_in.clone();
                let t0 = std::time::Instant::now();
                for _ in 0..reps {
                    for t in 0..nt {
                        let xn = hyper::grouped_rms_norm(&ress[t], &norm, &hp);
                        let lo = &mut lo_buf[t * hp.low_rank..(t + 1) * hp.low_rank];
                        lo.copy_from_slice(&lo_in[t * hp.low_rank..(t + 1) * hp.low_rank]);
                        hyper::hc_lo_apply(lo, &hp);
                        let mixed =
                            hyper::hc_mixed(&xn, &gate[t * hc_dim..(t + 1) * hc_dim], &hp);
                        let mut inj = inj_in[t * hp.hc..(t + 1) * hp.hc].to_vec();
                        hyper::hc_inject_apply(&mut inj, &hp);
                        ress[t] = hyper::hc_combine(
                            &ress[t],
                            &outs[t * e..(t + 1) * e],
                            &inj,
                            &hp,
                        );
                        std::hint::black_box(&mixed);
                        std::hint::black_box(&lo);
                    }
                }
                let old_ms = t0.elapsed().as_secs_f64() * 1e3 / (reps * nt) as f64;

                // 新: バッチ版 (トークン方向に 1 回で回す。nt >= 64 は rayon)
                let mut ress2: Vec<HyperStreams> = base
                    .iter()
                    .map(|d| HyperStreams {
                        data: d.clone(),
                        hc: hp.hc,
                        n_embd: e,
                    })
                    .collect();
                let mut lo_buf2 = lo_in.clone();
                let mut inj_buf2 = inj_in.clone();
                let t1 = std::time::Instant::now();
                for _ in 0..reps {
                    let xn_flat = hyper::grouped_rms_norm_batch(&ress2, &norm, &hp);
                    lo_buf2.copy_from_slice(&lo_in);
                    hyper::hc_lo_apply_batch(&mut lo_buf2, nt, &hp);
                    let mixed = hyper::hc_mixed_batch(&xn_flat, &gate, nt, &hp);
                    inj_buf2.copy_from_slice(&inj_in);
                    hyper::hc_inject_apply_batch(&mut inj_buf2, nt, &hp);
                    hyper::hc_combine_batch(&mut ress2, &outs, &inj_buf2, &hp);
                    std::hint::black_box(&mixed);
                    std::hint::black_box(&lo_buf2);
                }
                let new_ms = t1.elapsed().as_secs_f64() * 1e3 / (reps * nt) as f64;
                println!(
                    "{nt}\t{old_ms:.4}\t{new_ms:.4}\t{:.2}",
                    old_ms / new_ms
                );
            }
        }

        /// MoE FFN のバッチ版が逐次版とビット一致する (同じ expert を選んだ
        /// トークンをまとめる経路 = `m > 1` を実際に通す)。
        #[test]
        fn moe_ffn_batch_matches_seq() {
            // P-PF-GB: CPU 参照とのビット比較なので GPU GEMM を切る。
            gpu_gemm_off();
            let _g = moe_test_guard();
            let dir = TempDir::new("h2moe");
            let path = write_expert_gguf(&dir);
            let f = GgufFile::open(&path).expect("open");
            let p = cache_test_params();
            assert!(
                p.n_expert_used > 1,
                "まとめる経路を通すには n_expert_used > 1 が必要"
            );
            let lw = cache_test_layer(&f, &p);
            let nt = 3;
            let xs: Vec<f32> = (0..nt * p.n_embd)
                .map(|i| (((i * 37 % 101) as f32) - 50.0) * 0.013)
                .collect();

            // ビット一致は f32 経路どうしで見る (VNNI は活性を int8 に量子化するので別枠)。
            let bat = moe_ffn_quant_batch_opts(&p, &lw, &xs, nt, None, moe_schedule(), false)
                .expect("batch");
            for t in 0..nt {
                let seq = moe_ffn_quant(&p, &lw, &xs[t * p.n_embd..(t + 1) * p.n_embd], None)
                    .expect("seq");
                assert_bitwise_eq(&seq, &bat[t * p.n_embd..(t + 1) * p.n_embd], &format!("moe token {t}"));
            }
            assert!(
                bat.iter().any(|v| *v != 0.0),
                "MoE の出力が全 0 (経路の生死が分からない)"
            );
        }

        /// P33: ルータ F32 行列積のトークンブロック経路が、1 トークンずつの
        /// `gemv` (従来経路) とビット一致する。
        ///
        /// 実形 (k=2560 / n=512 / nt=512) と、端数が出る形 (k=2570 / n=6 / nt=71:
        /// トークン 7 本・行 2 行・列 2 列があまる) の両方で確かめる。
        #[test]
        fn gemv_batch_f32_blocked_matches_gemv() {
            use crate::model::moe_expert::set_f32_exact_for_test;
            gpu_gemm_off();
            let _g = moe_test_guard();
            // task 119: 参照の `gemv` (decode) は既定で FMA カーネルに変わるので、
            // トークンブロック経路 (prefill、スカラー) とのビット一致の契約は
            // exact を強制して見る (期待値は変えない)。
            set_f32_exact_for_test(Some(true));
            for (k, n, nt) in [(2560usize, 512usize, 512usize), (2570, 6, 71)] {
                let mut bytes = vec![0u8; n * k * 4];
                for i in 0..n * k {
                    let r = (i as u64)
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let v = ((r >> 33) % 2001) as f32;
                    bytes[i * 4..i * 4 + 4].copy_from_slice(&((v - 1000.0) * 1e-3).to_le_bytes());
                }
                let w = GgufLinear::from_slice(GgmlType::F32, k, n, &bytes).expect("F32 層");
                let xs: Vec<f32> = (0..nt * k)
                    .map(|i| (((i * 37 + 11) % 401) as f32 - 200.0) * 1e-3)
                    .collect();
                // 参照: 1 トークンずつ `gemv` (従来経路) で計算する。
                let mut want = vec![0.0f32; nt * n];
                for t in 0..nt {
                    w.gemv(&xs[t * k..(t + 1) * k], &mut want[t * n..(t + 1) * n])
                        .expect("gemv");
                }
                let before = GEMV_BATCH_BLOCKED_CALLS.load(std::sync::atomic::Ordering::Relaxed);
                let mut got = vec![0.0f32; nt * n];
                gemv_batch(&w, &xs, nt, &mut got).expect("gemv_batch");
                let after = GEMV_BATCH_BLOCKED_CALLS.load(std::sync::atomic::Ordering::Relaxed);
                if nt / ROUTER_TB >= 2 && n >= ROUTER_RB {
                    assert!(
                        after > before,
                        "k={k} n={n} nt={nt}: トークンブロック経路が発動していない"
                    );
                }
                assert_bitwise_eq(&want, &got, &format!("k={k} n={n} nt={nt}"));
            }
        }

        /// P33: 長いバッチ (nt=64) でも MoE のバッチ経路が逐次版とビット一致する。
        /// ルータ (`ffn_gate_inp`) がトークンブロック経路に入ることをカウンタで見る。
        #[test]
        fn moe_ffn_batch_long_matches_seq() {
            gpu_gemm_off();
            let _g = moe_test_guard();
            let dir = TempDir::new("h2moe64");
            let path = write_expert_gguf(&dir);
            let f = GgufFile::open(&path).expect("open");
            let p = cache_test_params();
            let lw = cache_test_layer(&f, &p);
            let nt = 64;
            let xs: Vec<f32> = (0..nt * p.n_embd)
                .map(|i| (((i * 37 % 101) as f32) - 50.0) * 0.013)
                .collect();

            let before = GEMV_BATCH_BLOCKED_CALLS.load(std::sync::atomic::Ordering::Relaxed);
            // VNNI は活性を int8 に落とすので逐次 (f32) とビット一致しない。
            // このテストは「バッチ経路の値が逐次と 1 ビットも変わらない」ことの検査なので
            // f32 経路 (vnni=false) で回す。VNNI 経路の一致は
            // `moe_flat_vnni_fires_and_matches_within_tolerance` が見る。
            let bat = moe_ffn_quant_batch_opts(&p, &lw, &xs, nt, None, moe_schedule(), false)
                .expect("batch");
            assert!(
                GEMV_BATCH_BLOCKED_CALLS.load(std::sync::atomic::Ordering::Relaxed) > before,
                "ルータのトークンブロック経路が発動していない"
            );
            for t in 0..nt {
                let seq = moe_ffn_quant(&p, &lw, &xs[t * p.n_embd..(t + 1) * p.n_embd], None)
                    .expect("seq");
                assert_bitwise_eq(
                    &seq,
                    &bat[t * p.n_embd..(t + 1) * p.n_embd],
                    &format!("moe long token {t}"),
                );
            }
            assert!(
                bat.iter().any(|v| *v != 0.0),
                "MoE の出力が全 0 (経路の生死が分からない)"
            );
        }

        /// flat スケジューラ (層全体の (expert, 行列, 行チャンク) を 1 本のタスク列で回す) が
        /// legacy (expert ごと × 行列ごとに 1 回 rayon) とビット一致する。
        ///
        /// 同じ入力に対して両方の経路を**明示的に**通し、出力を突き合わせる。小さい形なので
        /// `m` は 1〜4 に散らばり、`m == 1` の expert と `m > 1` の expert の両方ができる。
        #[test]
        fn moe_flat_matches_legacy_bitwise() {
            gpu_gemm_off();
            let _g = moe_test_guard();
            let dir = TempDir::new("h2flat");
            let path = write_expert_gguf(&dir);
            let f = GgufFile::open(&path).expect("open");
            let p = cache_test_params();
            let lw = cache_test_layer(&f, &p);
            let nt = 8;
            let xs: Vec<f32> = (0..nt * p.n_embd)
                .map(|i| (((i * 37 % 101) as f32) - 50.0) * 0.013)
                .collect();

            let flat0 = MOE_FLAT_CALLS.load(std::sync::atomic::Ordering::Relaxed);
            let leg0 = MOE_LEGACY_CALLS.load(std::sync::atomic::Ordering::Relaxed);
            // ビット一致は f32 経路どうしで見る (VNNI は活性を int8 に量子化するので別枠)。
            let flat = moe_ffn_quant_batch_opts(&p, &lw, &xs, nt, None, MoeSchedule::Flat, false)
                .expect("flat");
            let legacy = moe_ffn_quant_batch_with(&p, &lw, &xs, nt, None, MoeSchedule::Legacy)
                .expect("legacy");
            assert!(
                MOE_FLAT_CALLS.load(std::sync::atomic::Ordering::Relaxed) > flat0,
                "flat 経路が発動していない"
            );
            assert!(
                MOE_LEGACY_CALLS.load(std::sync::atomic::Ordering::Relaxed) > leg0,
                "legacy 経路が発動していない"
            );
            assert!(
                flat.iter().any(|v| *v != 0.0),
                "flat の出力が全 0 (経路の生死が分からない)"
            );
            assert_bitwise_eq(&legacy, &flat, "flat vs legacy");

            // 逐次版 (トークンごとの `moe_ffn_quant`) とも一致する
            for t in 0..nt {
                let seq = moe_ffn_quant(&p, &lw, &xs[t * p.n_embd..(t + 1) * p.n_embd], None)
                    .expect("seq");
                assert_bitwise_eq(
                    &seq,
                    &flat[t * p.n_embd..(t + 1) * p.n_embd],
                    &format!("flat token {t}"),
                );
            }
        }

        /// flat スケジューラの VNNI 経路 (`vpdpbusd`) が実際に発火し、f32 経路と
        /// 許容誤差で一致する。ビット一致は求めない (prefill は 2026-10-28 の
        /// オペレーター指示でビット一致契約の外)。
        #[test]
        fn moe_flat_vnni_fires_and_matches_within_tolerance() {
            gpu_gemm_off();
            let _g = moe_test_guard();
            if !moe_expert::vnni_supported() {
                eprintln!("skip moe_flat_vnni_fires_and_matches_within_tolerance: avx512vnni が無い");
                return;
            }
            let dir = TempDir::new("h2vnni");
            let path = write_expert_gguf(&dir);
            let f = GgufFile::open(&path).expect("open");
            let p = cache_test_params();
            let lw = cache_test_layer(&f, &p);
            let nt = 8;
            let xs: Vec<f32> = (0..nt * p.n_embd)
                .map(|i| (((i * 37 % 101) as f32) - 50.0) * 0.013)
                .collect();

            let before = moe_expert::vnni_calls();
            let got = moe_ffn_quant_batch_opts(&p, &lw, &xs, nt, None, MoeSchedule::Flat, true)
                .expect("flat vnni");
            assert!(
                moe_expert::vnni_calls() > before,
                "VNNI 経路が発火していない (flat から vpdpbusd が呼ばれていない)"
            );
            let want = moe_ffn_quant_batch_opts(&p, &lw, &xs, nt, None, MoeSchedule::Flat, false)
                .expect("flat f32");
            assert!(got.iter().any(|v| *v != 0.0), "VNNI の出力が全 0");
            let scale = want.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6);
            let max_rel = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs() / scale)
                .fold(0f32, f32::max);
            let mean_rel = got
                .iter()
                .zip(&want)
                .map(|(a, b)| ((a - b).abs() / scale) as f64)
                .sum::<f64>()
                / got.len() as f64;
            eprintln!("flat VNNI vs f32: max rel {max_rel:.3e} / mean rel {mean_rel:.3e}");
            assert!(max_rel < 5e-2, "flat VNNI の相対誤差 {max_rel} が大きすぎる");
        }

        /// `DENNO_MOE_FLAT` の解釈 (未設定・不明な値は既定の Auto に落ちる)。
        #[test]
        fn moe_schedule_env_values() {
            assert_eq!(moe_schedule_from_value(None), MoeSchedule::Auto);
            assert_eq!(moe_schedule_from_value(Some("1")), MoeSchedule::Flat);
            assert_eq!(moe_schedule_from_value(Some("0")), MoeSchedule::Legacy);
            assert_eq!(moe_schedule_from_value(Some("")), MoeSchedule::Auto);
            assert_eq!(moe_schedule_from_value(Some("true")), MoeSchedule::Auto);
        }

        /// タスク分割が `[0, n_out)` を 1 度ずつ覆う (端数・0 行も落とさない)。
        #[test]
        fn moe_row_chunks_cover_all_rows_once() {
            for n_out in [0usize, 1, 31, 32, 33, 640, 2560] {
                let mut v = Vec::new();
                moe_row_chunks(0, 0, 1, n_out, MOE_FLAT_ROWS, &mut v).expect("chunks");
                let mut covers = vec![0u8; n_out];
                for t in &v {
                    assert!(t.a < t.b, "空のチャンク (n_out={n_out})");
                    assert!(t.b as usize <= n_out, "行が範囲外 (n_out={n_out})");
                    for r in t.a as usize..t.b as usize {
                        covers[r] += 1;
                    }
                }
                assert!(
                    covers.iter().all(|&c| c == 1),
                    "n_out={n_out}: 行の重複か欠落がある"
                );
                assert_eq!(v.len(), n_out.div_ceil(MOE_FLAT_ROWS), "n_out={n_out}: 本数");
            }
        }

        /// `moe_split_n` の真理値表 (task 104)。
        #[test]
        fn moe_split_n_table() {
            // 無効 (max_c <= 1) と m が小さいときは割らない。
            for max_c in [0usize, 1] {
                for m in [0usize, 1, 63, 64, 492] {
                    assert_eq!(moe_split_n(m, max_c, 64), 1, "max_c={max_c} m={m}");
                }
            }
            // しきい値未満。
            assert_eq!(moe_split_n(63, 4, 64), 1);
            assert_eq!(moe_split_n(1, 4, 64), 1);
            // しきい値以上: min(4, ceil(m/64))。
            assert_eq!(moe_split_n(64, 4, 64), 1, "m == min_m は 1 (割る余地が無い)");
            assert_eq!(moe_split_n(65, 4, 64), 2);
            assert_eq!(moe_split_n(128, 4, 64), 2);
            assert_eq!(moe_split_n(192, 4, 64), 3);
            assert_eq!(moe_split_n(256, 4, 64), 4);
            assert_eq!(moe_split_n(492, 4, 64), 4);
            assert_eq!(moe_split_n(492, 2, 64), 2);
            assert_eq!(moe_split_n(492, 8, 64), 8);
            // min_m == 0 はしきい値なし。
            assert_eq!(moe_split_n(5, 4, 0), 4);
        }

        /// `moe_push_tasks` が token チャンク × 行チャンクの全領域を 1 度ずつ覆う。
        #[test]
        fn moe_push_tasks_cover_all_tokens_and_rows() {
            let (max_c, min_m) = (4usize, 64usize);
            for m in [1usize, 63, 64, 65, 129, 256, 492] {
                for n_out in [1usize, 33, 640] {
                    let mut v = Vec::new();
                    moe_push_tasks(0, m, n_out, MOE_FLAT_ROWS, max_c, min_m, &mut v)
                        .expect("tasks");
                    let c = moe_split_n(m, max_c, min_m);
                    assert_eq!(
                        v.len(),
                        c * n_out.div_ceil(MOE_FLAT_ROWS),
                        "m={m} n_out={n_out}: タスク数"
                    );
                    let mut covers = vec![0u32; m * n_out];
                    for t in &v {
                        let (t0, t1) = (t.t0 as usize, t.t1 as usize);
                        let (a, b) = (t.a as usize, t.b as usize);
                        assert!(t0 < t1 && a < b, "空チャンク m={m}");
                        for tok in t0..t1 {
                            for r in a..b {
                                covers[tok * n_out + r] += 1;
                            }
                        }
                    }
                    assert!(
                        covers.iter().all(|&c| c == 1),
                        "m={m} n_out={n_out}: 領域の重複か欠落"
                    );
                }
            }
        }

        /// `DENNO_MOE_TOKSPLIT` / `DENNO_MOE_TOKSPLIT_MIN` の解釈。
        #[test]
        fn moe_tok_split_env_values() {
            assert_eq!(moe_tok_split_max_from_value(None), 4);
            assert_eq!(moe_tok_split_max_from_value(Some("")), 4);
            assert_eq!(moe_tok_split_max_from_value(Some("nope")), 4);
            assert_eq!(moe_tok_split_max_from_value(Some("0")), 0);
            assert_eq!(moe_tok_split_max_from_value(Some("2")), 2);
            assert_eq!(moe_tok_split_max_from_value(Some("999")), 64, "上限 64");
            assert_eq!(moe_tok_split_min_from_value(None), 64);
            assert_eq!(moe_tok_split_min_from_value(Some("0")), 0);
            assert_eq!(moe_tok_split_min_from_value(Some("128")), 128);
        }

        /// task 104: token 分割 on/off で出力が 1 ビットも変わらないこと。
        ///
        /// m に偏り (max 492 / mean ≈ 59) を作り、off / 2 分割 / 4 分割 / しきい値 32 を
        /// 同じ入力で回して `to_bits()` 比較する。VNNI 経路 (活性の int8 量子化が入る)
        /// も「off と on の比較」がビット一致することを見る (f32 との比較は別枠)。
        #[test]
        fn moe_toksplit_matches_nosplit_bitwise() {
            gpu_gemm_off();
            let _g = moe_test_guard();
            let (n_in, n_ff, ne, k) = (2560usize, 640usize, 86usize, 10usize);
            let dir = TempDir::new("h2toksplit");
            let path = write_expert_gguf_big(&dir, n_in, n_ff, ne);
            let f = GgufFile::open(&path).expect("open");
            let p = bench_params(n_in, n_ff, ne, k);
            let lw = cache_test_layer(&f, &p);

            // m に強い偏りを作る (実機プロファイルの形: 大きい expert が少数)。
            // nt = Σm / k が max(m) 以上になるようにする (m <= nt の実モデルの形)。
            let mut ms: Vec<usize> = vec![492, 300, 200, 150, 120, 100, 80, 70, 65, 64];
            while ms.len() < ne {
                ms.push(45);
            }
            let sum: usize = ms.iter().sum();
            let nt = sum.div_ceil(k);
            assert!(nt >= *ms.iter().max().unwrap(), "nt={nt} が max m 未満");

            let total = nt * k;
            let mut slots: Vec<usize> = (0..total).collect();
            let mut seed = 777u64;
            for i in (1..total).rev() {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let j = ((seed >> 33) as usize) % (i + 1);
                slots.swap(i, j);
            }
            let mut it = slots.into_iter();
            let mut by_expert: Vec<Vec<(usize, usize)>> = vec![Vec::new(); ne];
            for (ei, &m) in ms.iter().enumerate() {
                let mut v: Vec<(usize, usize)> = (0..m)
                    .map(|_| {
                        let s = it.next().expect("slot");
                        (s / k, s % k)
                    })
                    .collect();
                v.sort_unstable();
                by_expert[ei] = v;
            }
            let weights = vec![0.37f32; nt * k];
            let xs: Vec<f32> = (0..nt * n_in)
                .map(|i| (((i * 37) % 101) as f32 - 50.0) * 0.003)
                .collect();

            let run = |max_c: usize, min_m: usize, vnni: bool| -> Vec<f32> {
                set_moe_tok_split_for_test(Some(max_c), Some(min_m));
                let mut out = vec![0.0f32; nt * n_in];
                moe_experts_flat(
                    &lw, &by_expert, &weights, k, &xs, n_in, n_ff, MOE_FLAT_ROWS, &mut out, vnni,
                    None,
                )
                .expect("flat");
                out
            };

            // f32 経路: off が基準。
            let split0 = MOE_TOKSPLIT_EXPERTS.load(std::sync::atomic::Ordering::Relaxed);
            let ref_off = run(0, 64, false);
            let out2 = run(2, 64, false);
            let out4 = run(4, 64, false);
            let out4m32 = run(4, 32, false);
            assert_bitwise_eq(&ref_off, &out2, "toksplit 2 vs off (f32)");
            assert_bitwise_eq(&ref_off, &out4, "toksplit 4 vs off (f32)");
            assert_bitwise_eq(&ref_off, &out4m32, "toksplit 4/min32 vs off (f32)");
            let split_cnt = MOE_TOKSPLIT_EXPERTS.load(std::sync::atomic::Ordering::Relaxed)
                - split0;
            assert!(
                split_cnt > 0,
                "token 分割が一度も発動していない (m に偏りが無い?)"
            );
            assert!(
                ref_off.iter().any(|v| *v != 0.0),
                "出力が全 0 (経路の生死が分からない)"
            );

            // VNNI 経路: off と on の比較がビット一致 (量子化の token 分割は独立)。
            let ref_v = run(0, 64, true);
            let out_v4 = run(4, 64, true);
            assert_bitwise_eq(&ref_v, &out_v4, "toksplit 4 vs off (VNNI)");
            set_moe_tok_split_for_test(None, None);
        }

        /// task 104 のマイクロベンチ (手動)。実機プロファイルの m 分布 (偏り:
        /// mean ≈ 21 / max 492) を再現した 1 層ぶんで、token 分割 off / 2 / 4 を
        /// 同一バイナリ・同一ループ内で交互に測る (rep の min を取る)。
        /// あわせて活性の int8 量子化 ([`moe_quant_all`]) だけの A/B も出す
        /// (expert 単位 1 タスクだった位相 = 分割の効果が最も大きい場所)。
        ///
        /// 実行 (16 スレッド):
        /// `RAYON_NUM_THREADS=16 cargo test --release -p denno-engine --features cuda --lib \
        ///   moe_tok_split_microbench -- --ignored --nocapture --test-threads=1`
        #[test]
        #[ignore = "CPU のスケジュール A/B (実機の ms/token は main が測る)"]
        fn moe_tok_split_microbench() {
            gpu_gemm_off();
            let _g = moe_test_guard();
            let (n_in, n_ff, ne, k) = (2560usize, 640usize, 256usize, 10usize);
            let dir = TempDir::new("h2tokbench");
            let path = write_expert_gguf_big(&dir, n_in, n_ff, ne);
            let f = GgufFile::open(&path).expect("open");
            let p = bench_params(n_in, n_ff, ne, k);
            let lw = cache_test_layer(&f, &p);

            // 実機プロファイルの m 分布を比例サンプルする (task 104 の背景の値):
            // m=1: 22070 / 2-3: 12110 / 4-7: 16990 / 8-15: 16570 / 16-31: 15220 /
            //      32-63: 13540 / 64: 8920 / 65+: 6330 (max 492)
            let buckets: [(u64, usize, usize); 8] = [
                (22070, 1, 1),
                (12110, 2, 3),
                (16990, 4, 7),
                (16570, 8, 15),
                (15220, 16, 31),
                (13540, 32, 63),
                (8920, 64, 64),
                (6330, 65, 492),
            ];
            let total_w: u64 = buckets.iter().map(|b| b.0).sum();
            let mut seed = 0x9E37_79B9_7F4A_7C15u64;
            let mut rnd = move || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed
            };
            let mut ms: Vec<usize> = Vec::with_capacity(ne);
            for _ in 0..ne {
                let r = rnd() % total_w;
                let mut acc = 0u64;
                let mut m = 1usize;
                for &(w, lo, hi) in &buckets {
                    acc += w;
                    if r < acc {
                        m = lo + (rnd() as usize) % (hi - lo + 1);
                        break;
                    }
                }
                ms.push(m);
            }
            ms.sort_unstable_by(|a, b| b.cmp(a));
            let sum: usize = ms.iter().sum();
            let nt = sum.div_ceil(k);
            println!(
                "threads={} ne={ne} n_in={n_in} n_ff={n_ff} k={k} tot_m={sum} nt={nt} max_m={} mean={:.1}",
                rayon::current_num_threads(),
                ms[0],
                sum as f64 / ne as f64,
            );
            assert!(nt >= ms[0], "nt={nt} が max m ({}) 未満", ms[0]);

            let total = nt * k;
            let mut slots: Vec<usize> = (0..total).collect();
            let mut s2 = 424242u64;
            for i in (1..total).rev() {
                s2 = s2
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let j = ((s2 >> 33) as usize) % (i + 1);
                slots.swap(i, j);
            }
            let mut it = slots.into_iter();
            let mut by_expert: Vec<Vec<(usize, usize)>> = vec![Vec::new(); ne];
            for (ei, &m) in ms.iter().enumerate() {
                let mut v: Vec<(usize, usize)> = (0..m)
                    .map(|_| {
                        let s = it.next().expect("slot");
                        (s / k, s % k)
                    })
                    .collect();
                v.sort_unstable();
                by_expert[ei] = v;
            }
            let weights = vec![0.1f32; nt * k];
            let xs: Vec<f32> = (0..nt * n_in)
                .map(|i| (((i * 37) % 101) as f32 - 50.0) * 0.003)
                .collect();

            let variants: [(usize, usize); 3] = [(0, 64), (2, 64), (4, 64)];
            let outs: Vec<Vec<f32>> = (0..variants.len()).map(|_| vec![0.0f32; nt * n_in]).collect();
            let mut outs = outs;
            // 温め + 値の突き合わせ
            for (vi, out) in outs.iter_mut().enumerate() {
                let (max_c, min_m) = variants[vi];
                set_moe_tok_split_for_test(Some(max_c), Some(min_m));
                for v in out.iter_mut() {
                    *v = 0.0;
                }
                moe_experts_flat(
                    &lw, &by_expert, &weights, k, &xs, n_in, n_ff, MOE_FLAT_ROWS, out, true,
                    None,
                )
                .expect("flat");
            }
            assert_bitwise_eq(&outs[0], &outs[1], "bench: 2 分割 vs off");
            assert_bitwise_eq(&outs[0], &outs[2], "bench: 4 分割 vs off");

            let reps = 7usize;
            let mut best = vec![f64::INFINITY; outs.len()];
            for _ in 0..reps {
                for (vi, out) in outs.iter_mut().enumerate() {
                    let (max_c, min_m) = variants[vi];
                    set_moe_tok_split_for_test(Some(max_c), Some(min_m));
                    for v in out.iter_mut() {
                        *v = 0.0;
                    }
                    let t0 = std::time::Instant::now();
                    moe_experts_flat(
                        &lw, &by_expert, &weights, k, &xs, n_in, n_ff, MOE_FLAT_ROWS, out, true,
                        None,
                    )
                    .expect("flat");
                    let us = t0.elapsed().as_secs_f64() * 1e6;
                    best[vi] = best[vi].min(us);
                }
            }
            println!(
                "layer: off={:8.0}us | split2={:8.0} ({:.3}x) | split4={:8.0} ({:.3}x)",
                best[0],
                best[1],
                best[0] / best[1],
                best[2],
                best[0] / best[2],
            );

            // 量子化だけの A/B (gather 出力と同じ expert 並びに組み直す。全 expert が
            // nonempty なので eid[pos] == pos)。
            let mut xin_q = vec![0.0f32; sum * n_in];
            let mut off_x = Vec::with_capacity(ne);
            let mut o = 0usize;
            for &m in &ms {
                off_x.push(o);
                o += m * n_in;
            }
            for (pos, picks) in by_expert.iter().enumerate() {
                for (idx, &(t, _)) in picks.iter().enumerate() {
                    xin_q[off_x[pos] + idx * n_in..off_x[pos] + (idx + 1) * n_in]
                        .copy_from_slice(&xs[t * n_in..(t + 1) * n_in]);
                }
            }
            let blk = moe_expert::vnni_act_block(lw.gate_exps.ty)
                .expect("bench: gate の型が VNNI 非対応");
            let mut bufs: Vec<_> = (0..variants.len())
                .map(|_| (Vec::<i8>::new(), Vec::<f32>::new(), Vec::<f32>::new()))
                .collect();
            for (vi, (q8, qsc, qgs)) in bufs.iter_mut().enumerate() {
                let (max_c, min_m) = variants[vi];
                set_moe_tok_split_for_test(Some(max_c), Some(min_m));
                let mut b = f64::INFINITY;
                for _ in 0..reps {
                    let t0 = std::time::Instant::now();
                    moe_quant_all(
                        &xin_q, &ms, &off_x, n_in, blk, tok_split_eff(max_c), min_m, q8, qsc, qgs,
                    );
                    b = b.min(t0.elapsed().as_secs_f64() * 1e6);
                }
                println!("quant[{vi}] max_c={max_c}: {b:8.0}us");
            }
            assert_eq!(
                bufs[0].0.len(),
                bufs[2].0.len(),
                "bench quant: q8 の長さが違う"
            );
            for (i, (a, b)) in bufs[0].0.iter().zip(bufs[2].0.iter()).enumerate() {
                assert_eq!(a, b, "bench quant: q8 の {i} 番目が違う (4 分割 vs off)");
            }
            assert_bitwise_eq(&bufs[0].1, &bufs[2].1, "bench quant: qsc 4 分割 vs off");
            assert_bitwise_eq(&bufs[0].2, &bufs[2].2, "bench quant: qgs 4 分割 vs off");
            set_moe_tok_split_for_test(None, None);
        }

        /// マイクロベンチの variant 値 (0 = off) を `moe_quant_all` の実引数へ。
        /// (off を 0 のまま渡すと `moe_split_n` は max_c<=1 で 1 を返すので同じ意味だが、
        /// 意図を明示するために 1 を渡す。)
        fn tok_split_eff(max_c: usize) -> usize {
            max_c.max(1)
        }

        /// 実形状 (n_in=2560, n_ff=640) の expert 3 本だけを持つ合成 GGUF。
        ///
        /// 実形は 1 expert あたり gate/up が 921 KB・down が 1.23 MB なので、
        /// 86 expert で約 264 MB になる (テスト用の一時ファイル)。
        fn write_expert_gguf_big(dir: &TempDir, n_in: usize, n_ff: usize, ne: usize) -> PathBuf {
            let path = dir.path("experts_big.gguf");
            let mut gate = Vec::new();
            let mut up = Vec::new();
            let mut down = Vec::new();
            for e in 0..ne {
                gate.extend_from_slice(&q4k_expert_bytes(n_in, n_ff, e));
                up.extend_from_slice(&q4k_expert_bytes(n_in, n_ff, e + 7));
                down.extend_from_slice(&q5_1_expert_bytes(n_ff, n_in, e + 3));
            }
            let tensors = vec![
                (
                    "blk.0.ffn_gate_exps.weight".to_string(),
                    vec![n_in as u64, n_ff as u64, ne as u64],
                    TY_Q4_K,
                    gate,
                ),
                (
                    "blk.0.ffn_up_exps.weight".to_string(),
                    vec![n_in as u64, n_ff as u64, ne as u64],
                    TY_Q4_K,
                    up,
                ),
                (
                    "blk.0.ffn_down_exps.weight".to_string(),
                    vec![n_ff as u64, n_in as u64, ne as u64],
                    TY_Q5_1,
                    down,
                ),
            ];
            let kvs = [
                ("general.architecture", Kv::Str("qwen4exp")),
                ("general.alignment", Kv::U32(ALIGN as u32)),
            ];
            write_gguf(&path, &kvs, &tensors);
            path
        }

        /// マイクロベンチ用の諸元 (実形)。
        fn bench_params(n_in: usize, n_ff: usize, ne: usize, k: usize) -> Qwen4ExpParams {
            let mut p = cache_test_params();
            p.n_embd = n_in;
            p.n_ff_exp = n_ff;
            p.n_ff_shexp = n_ff;
            p.n_expert = ne;
            p.n_expert_used = k;
            p
        }

        /// マイクロベンチ (手動)。選択 expert の FFN **1 層ぶん**を
        /// flat (層全体のタスク表) と legacy (expert ごと) で A/B する。
        ///
        /// 実形: 86 expert / n_embd 2560 / n_ff_exp 640 / gate・up は Q4_K、down は Q5_1
        /// (Flash-Next の実型)。m は 5〜120 に散らし (平均 ≈ 60 = 512 トークン × top-10 / 86)、
        /// 一様な m で閾値の当たりも取る。
        ///
        /// このコンテナはホスト共有で load average が動くので、**同一バイナリ・同一ループ内**で
        /// legacy / flat (行幅 4 種) を交互に測り、rep の min を取る (別バイナリ比較はしない)。
        ///
        /// 16 スレッドで走らせる:
        /// `RAYON_NUM_THREADS=16 cargo test --release -p denno-engine --features cuda --lib \
        ///   moe_flat_schedule_microbench -- --ignored --nocapture --test-threads=1`
        #[test]
        #[ignore = "CPU のスケジュール A/B (実機の ms/token は main が測る)"]
        fn moe_flat_schedule_microbench() {
            gpu_gemm_off();
            let _g = moe_test_guard();
            let (n_in, n_ff, ne, k) = (2560usize, 640usize, 86usize, 10usize);
            let dir = TempDir::new("h2bench");
            let path = write_expert_gguf_big(&dir, n_in, n_ff, ne);
            let f = GgufFile::open(&path).expect("open");
            let p = bench_params(n_in, n_ff, ne, k);
            let lw = cache_test_layer(&f, &p);
            let widths = [64usize, 128, 256, 512];

            // expert ごとの件数 `ms` から、(t, j) の割り当て・重み・入力を組み立てる。
            // スロット (t, j) を決定的にシャッフルして各 expert に配るので、
            // 1 expert のトークンは層全体に散らばる (実モデルと同じ形)。
            let build = |ms: &[usize]| {
                let sum: usize = ms.iter().sum();
                let nt = sum.div_ceil(k);
                let total = nt * k;
                let mut slots: Vec<usize> = (0..total).collect();
                let mut seed = 12345u64;
                for i in (1..total).rev() {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let j = ((seed >> 33) as usize) % (i + 1);
                    slots.swap(i, j);
                }
                let mut it = slots.into_iter();
                let mut by_expert: Vec<Vec<(usize, usize)>> = vec![Vec::new(); ms.len()];
                for (ei, &m) in ms.iter().enumerate() {
                    let mut v: Vec<(usize, usize)> = (0..m)
                        .map(|_| {
                            let s = it.next().expect("slot");
                            (s / k, s % k)
                        })
                        .collect();
                    // legacy の `position()` 順 (トークン昇順, 同トークン内は j 昇順)
                    v.sort_unstable();
                    by_expert[ei] = v;
                }
                let weights = vec![0.1f32; nt * k];
                let xs: Vec<f32> = (0..nt * n_in)
                    .map(|i| (((i * 37) % 101) as f32 - 50.0) * 0.003)
                    .collect();
                (by_expert, weights, xs, nt)
            };

            println!(
                "threads={} ne={ne} n_in={n_in} n_ff={n_ff} k={k}",
                rayon::current_num_threads()
            );

            // 1 つの分布について legacy + flat (行幅 4 種) を交互に測り、min を出す。
            let run = |ms: &[usize], label: &str| {
                let (by_expert, weights, xs, nt) = build(ms);
                let mut outs: Vec<Vec<f32>> = (0..1 + widths.len())
                    .map(|_| vec![0.0f32; nt * n_in])
                    .collect();
                // 温め (ページフォルト・分岐・レイアウト) + 値の突き合わせ
                for (vi, out) in outs.iter_mut().enumerate() {
                    if vi == 0 {
                        moe_experts_legacy(&lw, &by_expert, &weights, k, &xs, n_in, n_ff, out)
                            .expect("legacy");
                    } else {
                        moe_experts_flat(
                            &lw, &by_expert, &weights, k, &xs, n_in, n_ff, widths[vi - 1], out,
                            false,
                            None,
                        )
                        .expect("flat");
                    }
                }
                for vi in 1..outs.len() {
                    assert_bitwise_eq(
                        &outs[0],
                        &outs[vi],
                        &format!("{label}: flat(rows={}) vs legacy", widths[vi - 1]),
                    );
                }

                let reps = 3usize;
                let mut best = vec![f64::INFINITY; outs.len()];
                for _ in 0..reps {
                    for (vi, out) in outs.iter_mut().enumerate() {
                        for v in out.iter_mut() {
                            *v = 0.0;
                        }
                        let t0 = std::time::Instant::now();
                        if vi == 0 {
                            moe_experts_legacy(&lw, &by_expert, &weights, k, &xs, n_in, n_ff, out)
                                .expect("legacy");
                        } else {
                            moe_experts_flat(
                                &lw, &by_expert, &weights, k, &xs, n_in, n_ff, widths[vi - 1], out,
                                false,
                                None,
                            )
                            .expect("flat");
                        }
                        let us = t0.elapsed().as_secs_f64() * 1e6;
                        best[vi] = best[vi].min(us);
                    }
                }
                let tot: usize = ms.iter().sum();
                println!(
                    "{label:>10} (tot_m={tot:5}) legacy={:8.0}us | flat 32={:8.0} ({:.2}x) \
                     64={:8.0} ({:.2}x) 128={:8.0} ({:.2}x) 256={:8.0} ({:.2}x)",
                    best[0],
                    best[1],
                    best[0] / best[1],
                    best[2],
                    best[0] / best[2],
                    best[3],
                    best[0] / best[3],
                    best[4],
                    best[0] / best[4],
                );
            };

            // 参考: 作業バッファを毎層 `vec![0.0; n]` で作るときの「確保 + 初回タッチ」。
            // flat は層をまたいでプールを使い回すので、この分は毎層は払わない。
            let alloc_probe = |tot: usize| -> f64 {
                let t0 = std::time::Instant::now();
                {
                    let mut v = vec![0.0f32; tot * n_in];
                    let mut a = vec![0.0f32; tot * n_ff];
                    let mut b = vec![0.0f32; tot * n_ff];
                    let mut hh = vec![0.0f32; tot * n_in];
                    for x in v.iter_mut() {
                        *x = 1.0;
                    }
                    for x in a.iter_mut() {
                        *x = 1.0;
                    }
                    for x in b.iter_mut() {
                        *x = 1.0;
                    }
                    for x in hh.iter_mut() {
                        *x = 1.0;
                    }
                    std::hint::black_box((&v, &a, &b, &hh));
                }
                t0.elapsed().as_secs_f64() * 1e6
            };

            // 実測されている分布: m は 5〜120 (平均 ≈ 60)
            let spread: Vec<usize> = (0..ne).map(|ei| 5 + (ei * 37) % 116).collect();
            let tot: usize = spread.iter().sum();
            println!(
                "m=5..120: mean={:.1}  alloc+touch(毎層)={:.0}us",
                tot as f32 / ne as f32,
                alloc_probe(tot)
            );
            run(&spread, "m=5..120");

            // 閾値 (`MOE_FLAT_MIN_M`) の当たり: 一様な m
            for &mu in &[1usize, 2, 4, 8, 16, 32, 64, 120] {
                run(&vec![mu; ne], &format!("m={mu}"));
            }
        }

        /// task 291: UD-Q4_K_XL の prefill 形 (ne=512 / k=8 / nt=3826) で
        /// 選択 expert のバッチ経路を測る。既定の `moe_flat_schedule_microbench`
        /// は VNNI を切って legacy と比べるためのものなので、こちらは
        /// **本番と同じ VNNI の有無**を引数で切り替え、型別 (kprof) と
        /// 呼び出し側 (router / shared 込み) の合計も見る。
        ///
        /// `RAYON_NUM_THREADS=16 cargo test --release -p denno-engine --features cuda \
        ///   --lib moe_ud_prefill_microbench -- --ignored --nocapture --test-threads=1`
        #[test]
        #[ignore = "CPU の実測 (lab の ms/token は別)"]
        fn moe_ud_prefill_microbench() {
            gpu_gemm_off();
            let _g = moe_test_guard();
            // ne / nt / reps は env で縮められる (既定は実形)。
            let envu = |k: &str, d: usize| -> usize {
                std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
            };
            let (n_in, n_ff) = (2560usize, 640usize);
            let ne = envu("T291_NE", 512);
            let k = envu("T291_K", 8);
            let nt = envu("T291_NT", 3826);
            let reps = envu("T291_REPS", 3);
            let dir = TempDir::new("t291bench");
            let path = write_expert_gguf_big(&dir, n_in, n_ff, ne);
            let f = GgufFile::open(&path).expect("open");
            let p = bench_params(n_in, n_ff, ne, k);
            let lw = cache_test_layer(&f, &p);

            // 実モデルと同じ散らばり: 1 トークン = k 個の expert を一様に選ぶ。
            let mut by_expert: Vec<Vec<(usize, usize)>> = vec![Vec::new(); ne];
            let mut weights = vec![0.0f32; nt * k];
            for t in 0..nt {
                for j in 0..k {
                    let h = (t as u64).wrapping_mul(0x9e3779b97f4a7c15) ^ (j as u64 + 1) * 0xbf58_476d_1ce4_e5b9;
                    let ei = ((h >> 33) as usize) % ne;
                    by_expert[ei].push((t, j));
                    weights[t * k + j] = 0.125;
                }
            }
            let xs: Vec<f32> = (0..nt * n_in)
                .map(|i| (((i * 37) % 101) as f32 - 50.0) * 0.003)
                .collect();
            let tot_m: usize = by_expert.iter().map(|v| v.len()).sum();
            let m_max = by_expert.iter().map(|v| v.len()).max().unwrap_or(0);
            println!(
                "threads={} ne={ne} k={k} nt={nt} tot_m={tot_m} mean_m={:.1} max_m={m_max} vnni_supported={} vnni_enabled={}",
                rayon::current_num_threads(),
                tot_m as f32 / ne as f32,
                moe_expert::vnni_supported(),
                moe_expert::vnni_enabled()
            );

            let mut out = vec![0.0f32; nt * n_in];
            let mut best = [f64::INFINITY; 2];
            for rep in 0..reps {
                for (vi, &vnni) in [false, true].iter().enumerate() {
                    for v in out.iter_mut() {
                        *v = 0.0;
                    }
                    let t0 = std::time::Instant::now();
                    crate::model::exp_kprof::reset();
                    let got = moe_ffn_quant_batch_opts(
                        &p,
                        &lw,
                        &xs,
                        nt,
                        None,
                        MoeSchedule::Auto,
                        vnni,
                    )
                    .expect("batch");
                    let ms = t0.elapsed().as_secs_f64() * 1e3;
                    std::hint::black_box(&got);
                    out.copy_from_slice(&got);
                    if vnni {
                        if let Some(l) = crate::model::exp_kprof::line() {
                            println!("  kprof: {l}");
                        }
                    }
                    best[vi] = best[vi].min(ms);
                    if rep == 0 {
                        println!(
                            "  vnni={vnni}: {ms:.1} ms/layer ({:.2} ms/token, 48 層 {:.1} s)",
                            ms / nt as f64,
                            ms * 48.0 / 1e3
                        );
                    }
                }
            }
            println!(
                "min: vnni=0 {:.1} ms/layer ({:.2} ms/tok) | vnni=1 {:.1} ms/layer ({:.2} ms/tok)",
                best[0],
                best[0] / nt as f64,
                best[1],
                best[1] / nt as f64
            );
        }

        /// バッチ GEMV は逐次版と同じ**入口**を通るので、融合カーネルを
        /// 含めてビット一致する。2 つの入口 (`GgufLinear::gemv` = expert_gemv、
        /// `ExpertW::gemv` = expert_gemv_any) を別々に確かめる
        /// (Q5_1 / Q8_0 は入口でカーネルが変わるため)。
        #[test]
        fn gemv_batch_matches_gemv_bitwise() {
            // P-PF-GB: CPU 参照とのビット比較なので GPU GEMM を切る (契約は prefill のみ解除)。
            gpu_gemm_off();
            let dir = TempDir::new("h2gemv");
            let path = write_expert_gguf(&dir);
            let f = GgufFile::open(&path).expect("open");
            let p = cache_test_params();
            let lw = cache_test_layer(&f, &p);
            let nt = 3;
            for (w, name) in [
                (&lw.gate_exps, "gate (Q4_K = 融合カーネル)"),
                (&lw.down_exps, "down (Q5_1 = 入口でカーネルが変わる)"),
            ] {
                let xs: Vec<f32> = (0..nt * w.n_in)
                    .map(|i| (((i * 29) % 101) as f32 - 50.0) * 0.007)
                    .collect();

                // 1. GgufLinear 経路 (`linear()` と同じく fast_q8 = true で組む)
                let lin = GgufLinear::from_slice(
                    w.ty,
                    w.n_in,
                    w.n_out,
                    w.slice(0).expect("expert 0"),
                )
                .expect("GgufLinear::from_slice")
                .with_fast_q8(true);
                let mut bat = vec![0.0f32; nt * w.n_out];
                gemv_batch(&lin, &xs, nt, &mut bat).expect("gemv_batch");
                let mut seq = vec![0.0f32; nt * w.n_out];
                for t in 0..nt {
                    let mut one = vec![0.0f32; w.n_out];
                    lin.gemv(&xs[t * w.n_in..(t + 1) * w.n_in], &mut one)
                        .expect("gemv");
                    seq[t * w.n_out..(t + 1) * w.n_out].copy_from_slice(&one);
                }
                assert!(
                    bat.iter().any(|v| *v != 0.0),
                    "{name}: GgufLinear 経路の出力が全 0 (経路の生死が分からない)"
                );
                assert_bitwise_eq(&seq, &bat, &format!("{name} / GgufLinear"));

                // 2. ExpertW 経路 (expert_gemv_any = Q5_1/Q8_0 も融合)
                let mut bat2 = vec![0.0f32; nt * w.n_out];
                expert_gemv_batch(w, 0, &xs, nt, &mut bat2).expect("expert_gemv_batch");
                let mut seq2 = vec![0.0f32; nt * w.n_out];
                for t in 0..nt {
                    let mut one = vec![0.0f32; w.n_out];
                    w.gemv(0, &xs[t * w.n_in..(t + 1) * w.n_in], &mut one)
                        .expect("ExpertW::gemv");
                    seq2[t * w.n_out..(t + 1) * w.n_out].copy_from_slice(&one);
                }
                assert_bitwise_eq(&seq2, &bat2, &format!("{name} / ExpertW"));
            }

            // Q8_0 は fast_q8 = true のとき融合カーネルになる (モデルの linear() と
            // 同じ設定)。ここも行優先バッチでビット一致すること。
            let (k8, n8) = (64usize, 5usize);
            let bytes8 = tensor_bytes("w", &[k8 as u64, n8 as u64], GgmlType::Q8_0, 11);
            let lin8 = GgufLinear::from_slice(GgmlType::Q8_0, k8, n8, &bytes8)
                .expect("Q8_0 の GgufLinear")
                .with_fast_q8(true);
            let xs8: Vec<f32> = (0..nt * k8)
                .map(|i| (((i * 17) % 97) as f32 - 48.0) * 0.011)
                .collect();
            let mut b8 = vec![0.0f32; nt * n8];
            gemv_batch(&lin8, &xs8, nt, &mut b8).expect("gemv_batch (Q8_0)");
            let mut s8 = vec![0.0f32; nt * n8];
            for t in 0..nt {
                let mut one = vec![0.0f32; n8];
                lin8.gemv(&xs8[t * k8..(t + 1) * k8], &mut one)
                    .expect("gemv (Q8_0)");
                s8[t * n8..(t + 1) * n8].copy_from_slice(&one);
            }
            assert!(
                b8.iter().any(|v| *v != 0.0),
                "Q8_0: 出力が全 0 (経路の生死が分からない)"
            );
            assert_bitwise_eq(&s8, &b8, "Q8_0 (fast_q8 = 融合)");
        }

        /// 合成 GGUF に書いた 4 種の量子型 (Q8_0 / Q5_1 / Q4_K / Q5_K) で、
        ///
        /// (a) `gemv_batch(nt=1)` が逐次 `gemv` とビット一致する、
        /// (b) `gemv_batch(nt=64)` の各トークンが `gemv` (1 トークン) とビット一致する、
        /// (c) 行数の多い形が「ブロック 1 回デコード」経路を通る (カウンタで確認)、
        ///
        /// を確かめる。Q8_0 は `fast_q8` の両方 (融合カーネル / スカラー) を通す。
        /// 行数の少ない形 (n=5) は旧経路 (行ごとに nt 回デコード) に落ちるが、
        /// 同じ `gemv_batch` から同じ値が出ることをここで見る。
        #[test]
        fn gemv_batch_all_quant_types_bitwise() {
            // P-PF-GB: CPU 参照とのビット比較なので GPU GEMM を切る (契約は prefill のみ解除)。
            gpu_gemm_off();
            let dir = TempDir::new("h2q4gt");
            let types = [
                GgmlType::Q8_0,
                GgmlType::Q5_1,
                GgmlType::Q4K,
                GgmlType::Q5K,
            ];
            // 大きい形: ブロック 1 回デコード経路 + トークンのタイル分割
            // (T = GEMV_BATCH_TILE_BYTES / (4k) = 32 < nt=64 なので 2 枚に割れる)。
            let (bk, bn, bnt) = (2048usize, GEMV_BATCH_MIN_ROWS, 64usize);
            // 小さい形: 旧経路。
            let (sk, sn, snt) = (512usize, 5usize, 3usize);

            // 境目は純粋関数で固定する (行数 / トークン数)
            assert!(!gemv_batch_blockdecode(GEMV_BATCH_MIN_ROWS - 1, 64));
            assert!(gemv_batch_blockdecode(GEMV_BATCH_MIN_ROWS, 64));
            assert!(gemv_batch_blockdecode(bn, bnt), "大きい形がバッチ経路に載らない");
            assert!(!gemv_batch_blockdecode(sn, snt), "小さい形がバッチ経路に載った");
            assert!(!gemv_batch_blockdecode(usize::MAX, 0), "nt=0 では回さない");

            let mut tensors = Vec::new();
            for (i, ty) in types.iter().enumerate() {
                for (tag, k, n) in [("b", bk, bn), ("s", sk, sn)] {
                    let name = format!("{tag}{i}");
                    let data = tensor_bytes(&name, &[k as u64, n as u64], *ty, i as u32 + 1);
                    tensors.push((name, vec![k as u64, n as u64], ggml_id(*ty), data));
                }
            }
            let path = dir.path("gemv_types.gguf");
            write_model(&path, &tensors);
            let f = GgufFile::open(&path).expect("合成 GGUF を開けない");

            let c0 = GEMV_BATCH_BLOCKDECODE_CALLS.load(Ordering::Relaxed);
            for (i, ty) in types.iter().enumerate() {
                for (tag, k, n, nt) in [("b", bk, bn, bnt), ("s", sk, sn, snt)] {
                    let name = format!("{tag}{i}");
                    let (file, off, len) = f.tensor_mapping_parts(&name).expect("データ範囲");
                    let map = f.mmap_handle_at(file).expect("part の mmap");
                    let bytes = &map[off..off + len];
                    for fast_q8 in [false, true] {
                        let lin = GgufLinear::from_slice(*ty, k, n, bytes)
                            .expect("GgufLinear")
                            .with_fast_q8(fast_q8);
                        let want = |i: usize, w: usize| -> Vec<f32> {
                            (0..w).map(|j| (((i + j * 37) % 101) as f32 - 50.0) * 0.011).collect()
                        };

                        // (a) nt=1 が逐次版とビット一致
                        let x1 = want(0, k);
                        let mut b1 = vec![0.0f32; n];
                        gemv_batch(&lin, &x1, 1, &mut b1).expect("gemv_batch (nt=1)");
                        let mut s1 = vec![0.0f32; n];
                        lin.gemv(&x1, &mut s1).expect("gemv");
                        assert!(
                            s1.iter().any(|v| *v != 0.0),
                            "{name}: 出力が全 0 (経路の生死が分からない)"
                        );
                        assert_bitwise_eq(
                            &s1,
                            &b1,
                            &format!("{name} fast_q8={fast_q8} nt=1 / gemv"),
                        );

                        // (b) nt トークンの各行が 1 トークンの計算とビット一致
                        let mut xs = Vec::with_capacity(nt * k);
                        for t in 0..nt {
                            xs.extend_from_slice(&want(t + 1, k));
                        }
                        let mut bat = vec![0.0f32; nt * n];
                        gemv_batch(&lin, &xs, nt, &mut bat).expect("gemv_batch (nt)");
                        for t in 0..nt {
                            let mut seq = vec![0.0f32; n];
                            lin.gemv(&xs[t * k..(t + 1) * k], &mut seq).expect("gemv");
                            assert_bitwise_eq(
                                &seq,
                                &bat[t * n..(t + 1) * n],
                                &format!("{name} fast_q8={fast_q8} token {t} / gemv"),
                            );
                        }
                    }
                }
            }
            assert!(
                GEMV_BATCH_BLOCKDECODE_CALLS.load(Ordering::Relaxed) > c0,
                "ブロック 1 回デコード経路が 1 回も走っていない"
            );
        }

        /// 実形状 (Flash-Next / n_embd 2560) の Q8_0 行列で、バッチ GEMV の
        /// 1 呼び出しを CPU で測る。**GPU 実測の代わりではない**。
        /// カーネル構造の A/B (per-token 呼び出し vs ブロック 1 回デコード) 用。
        #[test]
        #[ignore = "CPU のカーネル A/B (実機の ms/token は main が測る)"]
        fn gemv_batch_shape_ab() {
            // P-PF-GB: CPU 参照とのビット比較なので GPU GEMM を切る (契約は prefill のみ解除)。
            gpu_gemm_off();
            use std::time::Instant;
            let nt = 64usize;
            // (名前, k, n) — qwen4exp の実形状 (n_embd 2560 / conv_dim 10240 /
            // inner 6144 / kv 1024 / q 12288 / shexp 640 / hc_lr 320)
            let shapes = [
                ("ssm_qkv", 2560usize, 10240usize),
                ("ssm_gate", 2560, 6144),
                ("ssm_out", 6144, 2560),
                ("hc_up", 320, 10240),
                ("hc_down", 10240, 320),
                ("shexp_gate", 2560, 640),
                ("shexp_down", 640, 2560),
                ("full_q", 2560, 12288),
                ("full_kv", 2560, 1024),
                ("full_o", 6144, 2560),
            ];
            let mut tot_new = 0.0f64;
            let mut tot_old = 0.0f64;
            for (name, k, n) in shapes {
                let bytes = tensor_bytes("w", &[k as u64, n as u64], GgmlType::Q8_0, 3);
                let lin = GgufLinear::from_slice(GgmlType::Q8_0, k, n, &bytes)
                    .expect("GgufLinear")
                    .with_fast_q8(true);
                let xs: Vec<f32> = (0..nt * k)
                    .map(|i| (((i * 31) % 97) as f32 - 48.0) * 0.01)
                    .collect();
                let mut outs = vec![0.0f32; nt * n];
                let mut old = vec![0.0f32; nt * n];
                gemv_batch(&lin, &xs, nt, &mut outs).expect("warm-up");
                assert!(outs.iter().any(|v| *v != 0.0), "{name}: 出力が全 0");
                // 旧経路 (行ごとに 1 行 GEMV を nt 回) を同じ入力で測る
                gemv_rows_pertoken(
                    GgmlType::Q8_0,
                    true,
                    &bytes,
                    k,
                    n,
                    &xs,
                    nt,
                    &mut old,
                )
                .expect("pertoken");
                assert_bitwise_eq(&old, &outs, &format!("{name}: 旧経路と一致"));
                let reps = 7;
                let mut best_new = f64::MAX;
                let mut best_old = f64::MAX;
                for _ in 0..reps {
                    let t0 = Instant::now();
                    gemv_batch(&lin, &xs, nt, &mut outs).expect("bench");
                    best_new = best_new.min(t0.elapsed().as_secs_f64());
                    let t0 = Instant::now();
                    gemv_rows_pertoken(
                        GgmlType::Q8_0,
                        true,
                        &bytes,
                        k,
                        n,
                        &xs,
                        nt,
                        &mut old,
                    )
                    .expect("bench old");
                    best_old = best_old.min(t0.elapsed().as_secs_f64());
                }
                tot_new += best_new;
                tot_old += best_old;
                println!(
                    "{name:<11} k={k:<6} n={n:<6} 新 {:7.3} / 旧 {:7.3} ms/chunk  x{:.2}",
                    best_new * 1e3,
                    best_old * 1e3,
                    best_old / best_new
                );
            }
            println!(
                "合計 新 {:.3} / 旧 {:.3} ms/chunk (= {:.3} / {:.3} ms/token)  x{:.2}",
                tot_new * 1e3,
                tot_old * 1e3,
                tot_new * 1e3 / nt as f64,
                tot_old * 1e3 / nt as f64,
                tot_old / tot_new
            );
        }

        /// 層優先バッチ prefill の logits が逐次経路とビット一致する。
        ///
        /// 併せて「バッチ分岐とバッチ GEMV が実際に走った」ことをカウンタで見る
        /// (緑 = 経路が発動した証拠ではない)。チャンク 1 / 3 / 8 で切れ目を変えても
        /// 一致することを確かめる (チャンク跨ぎの状態持ち越しが無いことの確認)。
        #[test]
        fn prefill_batch_matches_sequential() {
            let _g = moe_test_guard();
            // このテストは「バッチ経路が逐次経路とビット一致する」ことを固定する。
            // VNNI は活性を int8 に量子化するので、ここでは明示的に切る
            // (既定は VNNI 有効。VNNI の誤差は moe_flat_vnni_fires_and_matches_within_tolerance)。
            moe_expert::set_vnni_for_test(Some(false));
            let dir = TempDir::new("h2model");
            let path = build(&dir);
            let f = GgufFile::open(&path).expect("合成モデルを開けない");
            // 語彙内のトークン列 (PLE の n-gram 窓は末尾 3 個)
            let ctx: Vec<u32> = vec![1, 2, 3, 4, 5, 6];

            let (seq, bseq, _) = run(&f, &ctx, false, 0);
            assert_eq!(bseq, 0, "逐次経路でバッチ分岐が走った");
            assert_eq!(seq.len(), VOCAB, "logits の長さが n_vocab と違う");
            assert!(seq.iter().all(|v| v.is_finite()), "逐次 prefill に NaN/inf");
            assert!(
                seq.iter().any(|&v| v != seq[0]),
                "logits が全要素同値 (経路の生死が分からない)"
            );

            // 決定性 (同じモデルを組み直して同じ値)
            let (seq2, _, _) = run(&f, &ctx, false, 0);
            assert_bitwise_eq(&seq, &seq2, "逐次 prefill 2 回目");

            for chunk in [1usize, 3, 8] {
                let p0 = PLE_BATCH_CALLS.load(Ordering::Relaxed);
                let (bat, nb, ng) = run(&f, &ctx, true, chunk);
                assert!(nb > 0, "chunk {chunk}: 層優先バッチ経路が発動していない");
                assert!(ng > 0, "chunk {chunk}: バッチ GEMV が 1 回も走っていない");
                // P78: PLE のまとめ引きが実際に走ったか (値の一致だけでは分からない)。
                assert!(
                    PLE_BATCH_CALLS.load(Ordering::Relaxed) > p0,
                    "chunk {chunk}: PLE のまとめ引きが 1 回も走っていない"
                );
                assert_bitwise_eq(&seq, &bat, &format!("chunk {chunk}"));
            }
            moe_expert::set_vnni_for_test(None);
        }

        /// task 200: 平坦な attn 入口 (`DENNO_PF_ATTN_FLAT` の分岐) が
        /// 行入口と同じ値を返し、**両方の分岐が実際に走る**ことを固定する。
        ///
        /// 層ループ本体 (常駐経路) は GPU が要るので CPU では通らない。ここでは
        /// 層ループが呼ぶのと同じ関数 [`Qwen4Exp::prefill_attn_chunk`] を 2 つの
        /// モデルで 1 回ずつ呼び、値と通過カウンタを見る。
        #[test]
        fn prefill_attn_flat_matches_rows_and_fires() {
            let _g = moe_test_guard();
            let dir = TempDir::new("h2attnflat");
            let path = build(&dir);
            let f = GgufFile::open(&path).expect("合成モデルを開けない");
            let p = Qwen4ExpParams::from_meta(&f).expect("from_meta");
            let e = p.n_embd;
            let n_layer = p.n_layer;
            let nt = 3usize;
            // 行ごとに違う 0 でない入力。
            let flat: Vec<f32> = (0..nt * e)
                .map(|i| 0.25 * (((i * 7) % 11) as f32) - 1.0)
                .collect();
            let curs: Vec<Vec<f32>> = flat.chunks_exact(e).map(|c| c.to_vec()).collect();

            let mut m_rows = Qwen4Exp::new(&f).expect("Qwen4Exp::new (rows)");
            let mut m_flat = Qwen4Exp::new(&f).expect("Qwen4Exp::new (flat)");
            let r0 = PF_ATTN_ROWS_CALLS.load(Ordering::Relaxed);
            let f0 = PF_ATTN_FLAT_CALLS.load(Ordering::Relaxed);
            for il in 0..n_layer {
                let a = Qwen4Exp::prefill_attn_chunk(
                    m_rows.attn.as_mut(),
                    il,
                    &flat,
                    &curs,
                    nt,
                    e,
                    false,
                    false,
                )
                .expect("行入口");
                let b = Qwen4Exp::prefill_attn_chunk(
                    m_flat.attn.as_mut(),
                    il,
                    &flat,
                    &curs,
                    nt,
                    e,
                    true,
                    false,
                )
                .expect("平坦入口");
                assert_eq!(a.len(), nt * e, "層 {il}: 出力長");
                assert_bitwise_eq(&a, &b, &format!("層 {il} 行入口 vs 平坦入口"));
            }
            assert_eq!(
                PF_ATTN_ROWS_CALLS.load(Ordering::Relaxed) - r0,
                n_layer,
                "行入口の通過数"
            );
            assert_eq!(
                PF_ATTN_FLAT_CALLS.load(Ordering::Relaxed) - f0,
                n_layer,
                "平坦入口の通過数"
            );
        }

        /// task 134: prefill の full attention が GPU の分岐に入り、**短い
        /// チャンクは構造ゲートで CPU に落ちる**ことを固定する。
        ///
        /// 実モデルと同じ構成 (`SsmAttn` が `FullAttn` へ委譲する) を組む。
        /// dev コンテナ (GPU 無し) では `q4_attn_prefill` が `None` を返すので、
        /// 分岐に入ったことは `attend_batch_gpu_tries` の増分で見る (緑 = 発動の
        /// 証拠ではない)。GPU が発動していないので値は逐次経路とビット一致する。
        #[cfg(feature = "cuda")]
        #[test]
        fn prefill_gpu_attn_tries_then_falls_back_on_short_chunk() {
            use crate::model::qwen4exp_attn::{
                attend_batch_gpu_calls, attend_batch_gpu_tries, FullAttn,
            };
            use crate::model::qwen4exp_ssm::SsmAttn;
            let _g = moe_test_guard();
            moe_expert::set_vnni_for_test(Some(false));
            let dir = TempDir::new("h2gpuattnpf");
            let path = build(&dir);
            let f = GgufFile::open(&path).expect("合成モデルを開けない");
            let p = Qwen4ExpParams::from_meta(&f).expect("from_meta");
            let ctx: Vec<u32> = vec![1, 2, 3, 4, 5, 6];

            // 1 回ぶんのバッチ prefill (毎回モデルを組み直す)。
            // 戻り = (logits, GPU 分岐に入った回数, GPU 出力を採用した回数)。
            let run_full = |chunk: usize| -> (Vec<f32>, usize, usize) {
                let full = FullAttn::from_gguf(&f, &p, 128).expect("FullAttn");
                let attn = SsmAttn::with_full(&f, &p, Box::new(full)).expect("SsmAttn");
                let mut m = Qwen4Exp::with_attn(&f, Box::new(attn)).expect("モデルを組めない");
                let t0 = attend_batch_gpu_tries();
                let g0 = attend_batch_gpu_calls();
                let out = m
                    .forward_prefill_batch(&ctx, chunk)
                    .expect("バッチ prefill");
                (
                    out,
                    attend_batch_gpu_tries() - t0,
                    attend_batch_gpu_calls() - g0,
                )
            };

            // ゲート off (既定) では分岐に入らない。
            crate::cuda::set_q4_gpu_attn_pf_for_test(Some(false));
            let (off, off_tries, _) = run_full(6);
            assert_eq!(off_tries, 0, "ゲート off なのに GPU の分岐に入った");
            assert!(
                off.iter().any(|&v| v != off[0]),
                "logits が全要素同値 (経路の生死が分からない)"
            );

            // しきい値を 0 にして分岐を開ける。6 行のチャンクは構造ゲート
            // (`attn_prefill_shape_ok` の行数下限 ATTN_Q4_PF_MIN_ROWS = 8、
            // task 135 以前は gemm_min_m() = 256) で CPU。
            crate::cuda::set_q4_gpu_attn_pf_for_test(Some(true));
            crate::cuda::set_q4_gpu_attn_pf_min_for_test(Some(0), Some(0));
            let (on, on_tries, on_calls) = run_full(6);
            assert!(
                on_tries > 0,
                "ゲートを開けたのに GPU の分岐に入っていない (配線が届いていない)"
            );
            assert_eq!(
                on_calls, 0,
                "6 行のチャンクで GPU が発動した (短いチャンクの下限が効いていない)"
            );
            assert_bitwise_eq(&off, &on, "GPU 分岐のフォールバック");

            crate::cuda::set_q4_gpu_attn_pf_for_test(None);
            crate::cuda::set_q4_gpu_attn_pf_min_for_test(None, None);
            moe_expert::set_vnni_for_test(None);
        }

        /// `DENNO_PLE_BATCH` の解釈: **既定 on**、`0` / `false` / `off` だけ無効。
        #[test]
        fn ple_batch_gate_defaults_on() {
            assert!(ple_batch_from_value(None), "未設定は既定で有効");
            assert!(ple_batch_from_value(Some("1")));
            assert!(ple_batch_from_value(Some("")), "空文字は既定 (有効) のまま");
            assert!(!ple_batch_from_value(Some("0")));
            assert!(!ple_batch_from_value(Some("false")));
            assert!(!ple_batch_from_value(Some("OFF")), "大文字も無効");
        }

        /// P78: まとめ引き (既定) と 1 トークンずつの旧経路がビット一致し、
        /// 分岐が実際に切り替わること (カウンタで通過を確認する)。
        #[test]
        fn prefill_ple_batch_off_matches_on() {
            let _g = moe_test_guard();
            moe_expert::set_vnni_for_test(Some(false));
            let dir = TempDir::new("h2plebatch");
            let path = build(&dir);
            let f = GgufFile::open(&path).expect("合成モデルを開けない");
            let ctx: Vec<u32> = vec![1, 2, 3, 4, 5, 6];

            set_ple_batch_for_test(Some(true));
            let p0 = PLE_BATCH_CALLS.load(Ordering::Relaxed);
            let (on, _, _) = run(&f, &ctx, true, 8);
            assert!(
                PLE_BATCH_CALLS.load(Ordering::Relaxed) > p0,
                "まとめ引きが 1 回も走っていない"
            );

            set_ple_batch_for_test(Some(false));
            let p1 = PLE_BATCH_CALLS.load(Ordering::Relaxed);
            let (off, _, _) = run(&f, &ctx, true, 8);
            assert_eq!(
                PLE_BATCH_CALLS.load(Ordering::Relaxed),
                p1,
                "off なのにまとめ引きが走った"
            );

            set_ple_batch_for_test(None);
            assert_bitwise_eq(&on, &off, "まとめ引き on/off");
            assert!(
                on.iter().any(|&v| v != on[0]),
                "logits が全要素同値 (経路の生死が分からない)"
            );
            moe_expert::set_vnni_for_test(None);
        }

        /// PLE の行を `Option` 込みでビット比較する (task 254 の先読み用)。
        fn assert_ple_bitwise(a: &[Option<Vec<f32>>], b: &[Option<Vec<f32>>], what: &str) {
            assert_eq!(a.len(), b.len(), "{what}: 本数が違う");
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                match (x, y) {
                    (Some(x), Some(y)) => {
                        assert_eq!(x.len(), y.len(), "{what}: 行 {i} の長さが違う");
                        for k in 0..x.len() {
                            assert_eq!(
                                x[k].to_bits(),
                                y[k].to_bits(),
                                "{what}: 行 {i} 要素 {k} がビット一致しない"
                            );
                        }
                    }
                    (None, None) => {}
                    _ => panic!("{what}: 行 {i} の Some/None が違う"),
                }
            }
        }

        /// task 254: 次チャンクの PLE 先読み ([`PlePrefetch`]) が、直接引いた値と
        /// ビット一致すること + 分岐が起動すること (カウンタで確認)。
        ///
        /// 常駐経路 (`forward_prefill_batch_resident`) は cuda が要るので、ここでは
        /// スレッドの起動・join と値の同一性だけを確かめる。
        #[test]
        fn ple_prefetch_matches_direct_gather() {
            let _g = moe_test_guard();
            moe_expert::set_vnni_for_test(Some(false));
            let dir = TempDir::new("h2plepre");
            let path = build(&dir);
            let f = GgufFile::open(&path).expect("合成モデルを開けない");
            let ctx: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
            let m = Qwen4Exp::new(&f).expect("Qwen4Exp::new");
            assert!(m.ple.is_some(), "合成モデルに PLE が無い");

            // 直接引く (2 チャンクに割る)。
            let d0 = m.gather_ple_many(&ctx[..4], 0).expect("direct 0");
            let d1 = m.gather_ple_many(&ctx[..8], 4).expect("direct 1");
            assert!(
                d0.iter().flatten().any(|r| r.iter().any(|&v| v != 0.0)),
                "PLE の行が全部 0 (経路が死んでいる)"
            );

            // 先読みスレッドで同じ範囲を引く。
            let (fires0, _, _) = prekprof::ple_pre_counts();
            let p0 = PlePrefetch::spawn(m.ple.as_ref().map(std::sync::Arc::clone), &ctx, 0, 4)
                .join()
                .expect("prefetch 0");
            let p1 = PlePrefetch::spawn(m.ple.as_ref().map(std::sync::Arc::clone), &ctx, 4, 8)
                .join()
                .expect("prefetch 1");
            let (fires1, toks1, _) = prekprof::ple_pre_counts();
            assert_eq!(fires1 - fires0, 2, "先読みの起動回数が合わない");
            assert!(toks1 > 0, "先読みのトークン数が 0");

            assert_ple_bitwise(&d0, &p0, "先読み 0");
            assert_ple_bitwise(&d1, &p1, "先読み 1");

            // PLE を持たない・空の範囲・範囲外では起動しない (fail closed)。
            assert!(!PlePrefetch::spawn(None, &ctx, 0, 4).is_some(), "PLE 無しで起動");
            assert!(
                !PlePrefetch::spawn(m.ple.as_ref().map(std::sync::Arc::clone), &ctx, 8, 8).is_some(),
                "空の範囲で起動"
            );
            assert!(
                !PlePrefetch::spawn(m.ple.as_ref().map(std::sync::Arc::clone), &ctx, 0, 9).is_some(),
                "範囲外で起動"
            );
            moe_expert::set_vnni_for_test(None);
        }

        /// P78: `DENNO_PRE_KPROF` の区間計測が prefill で実際に積まれ、
        /// decode (同じ `forward_token_at` を通る) では積まれないこと。
        #[test]
        fn prekprof_records_sections_in_prefill_only() {
            let _g = moe_test_guard();
            moe_expert::set_vnni_for_test(Some(false));
            prekprof::set_for_test(true);
            prekprof::reset();
            let dir = TempDir::new("prekprof");
            let path = build(&dir);
            let f = GgufFile::open(&path).expect("合成モデルを開けない");
            let ctx: Vec<u32> = vec![1, 2, 3, 4];
            let mut m = Qwen4Exp::new(&f).expect("Qwen4Exp::new");

            // 逐次 prefill (ガード経路) で embed / ple / layer が積まれる。
            let _ = m.forward_prefill_with(&ctx, false).expect("逐次 prefill");
            assert!(prekprof::ns(prekprof::EMBED) > 0, "embed 区間が積まれていない");
            assert!(prekprof::ns(prekprof::PLE) > 0, "ple 区間が積まれていない");
            assert!(prekprof::ns(prekprof::LAYER) > 0, "layer 区間が積まれていない");

            // 同じ関数を通る decode では積まない (prefill のガードの外)。
            let before = prekprof::ns(prekprof::PLE);
            let _ = m.forward_token(&ctx).expect("decode");
            assert_eq!(
                prekprof::ns(prekprof::PLE),
                before,
                "decode で PLE 区間が積まれた"
            );

            // レポートは 2 行で、PLE の pread 統計と累積のリセットを行う。
            let lines = prekprof::report("unit", ctx.len(), 1_000_000);
            assert_eq!(lines.len(), 2, "{lines:?}");
            assert!(lines[0].starts_with("[pre-kprof] unit tok=4"), "{:?}", lines[0]);
            assert!(!lines[1].contains(" rows=0 "), "PLE の行数が 0: {:?}", lines[1]);
            assert_eq!(prekprof::ns(prekprof::PLE), 0, "report が累積を消していない");

            // バッチ経路でも同じ区間が立つ (こちらはチャンク単位)。
            prekprof::reset();
            let _ = m.forward_prefill_batch(&ctx, 2).expect("バッチ prefill");
            assert!(prekprof::ns(prekprof::EMBED) > 0, "バッチ: embed が積まれていない");
            assert!(prekprof::ns(prekprof::PLE) > 0, "バッチ: ple が積まれていない");
            assert!(prekprof::ns(prekprof::LAYER) > 0, "バッチ: layer が積まれていない");

            prekprof::reset();
            prekprof::set_for_test(false);
            moe_expert::set_vnni_for_test(None);
        }

        /// task 138: decode の台帳が**実際に** mix / ssm / full / router / shared に
        /// 積まれること (配線の生死)。GPU 無しでも壁時計は積まれる。
        #[test]
        fn q4ledger_records_decode_items() {
            let _g = moe_test_guard();
            let dir = TempDir::new("q4ledger_items");
            let path = build(&dir);
            let f = GgufFile::open(&path).expect("合成モデルを開けない");
            let ctx: Vec<u32> = vec![1, 2, 3, 4];
            let mut m = Qwen4Exp::new(&f).expect("Qwen4Exp::new");

            q4ledger::set_for_test(true);
            q4ledger::reset();
            let _ = m.forward_token(&ctx).expect("decode");

            for (i, name) in [q4ledger::MIX, q4ledger::SSM, q4ledger::FULL]
                .iter()
                .map(|&i| (i, q4ledger::NAMES[i]))
            {
                assert!(
                    q4ledger::row(i).wall_ns > 0,
                    "{name} の区間が積まれていない (配線が無い)"
                );
            }
            for (i, name) in [q4ledger::ROUTER, q4ledger::SHARED]
                .iter()
                .map(|&i| (i, q4ledger::NAMES[i]))
            {
                assert!(
                    q4ledger::row(i).wall_ns > 0,
                    "{name} の区間が積まれていない (moe の中の配線が無い)"
                );
            }
            // task 155: 層ループの端 (`loop`) は decode でも必ず積まれる。
            assert!(
                q4ledger::row(q4ledger::LOOP).wall_ns > 0,
                "loop の区間が積まれていない (層ループの配線が無い)"
            );
            // task 155: decode の通常経路はフックを登録しないので `hooks` は 0。
            assert_eq!(
                q4ledger::row(q4ledger::HOOKS).wall_ns,
                0,
                "購読者の居ないフックが呼ばれている"
            );
            // 行は 1 行のヘッダ + 列見出し + 積まれた item の数だけ出る。
            let lines = q4ledger::line(1, 0);
            assert!(lines.len() >= 3, "{lines:?}");
            assert!(
                lines.iter().any(|l| l.contains(" router ")),
                "router の行が無い: {lines:?}"
            );

            q4ledger::reset();
            q4ledger::set_for_test(false);
        }

        /// task 155: フックを登録した経路 (`debug_layer`) では `hooks` 行が積まれる。
        ///
        /// 購読者が居ないときに呼び出しを飛ばす分岐の**生きている側**が配線されて
        /// いることの確認 (片側だけテストすると「常に飛ばす」実装でも通る)。
        #[test]
        fn q4ledger_hooks_row_fires_when_subscribed() {
            let _g = moe_test_guard();
            let dir = TempDir::new("q4ledger_hooks");
            let path = build(&dir);
            let f = GgufFile::open(&path).expect("合成モデルを開けない");
            let ctx: Vec<u32> = vec![1, 2, 3];
            let mut m = Qwen4Exp::new(&f).expect("Qwen4Exp::new");

            q4ledger::set_for_test(true);
            q4ledger::reset();
            let mut n = 0usize;
            m.debug_layer(&ctx, 0, &mut |_, _| n += 1)
                .expect("debug_layer");
            assert!(n > 0, "フックの sink が 1 回も呼ばれていない");
            assert!(
                q4ledger::row(q4ledger::HOOKS).wall_ns > 0,
                "hooks の区間が積まれていない (購読側の配線が無い)"
            );
            q4ledger::reset();
            q4ledger::set_for_test(false);
        }

        /// task 145: prefill の台帳が**実際に** experts を含む各区間に積まれ、
        /// 全体 (`other`) が残りを吸うこと (配線の生死)。GPU 無しでも壁時計は積まれる。
        #[test]
        fn q4ledger_records_prefill_items() {
            let _g = moe_test_guard();
            let dir = TempDir::new("q4ledger_prefill");
            let path = build(&dir);
            let f = GgufFile::open(&path).expect("合成モデルを開けない");
            let ctx: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
            let mut m = Qwen4Exp::new(&f).expect("Qwen4Exp::new");

            q4ledger::set_for_test(true);
            q4ledger::reset();
            let _ = m.forward_prefill_with(&ctx, true).expect("batch prefill");

            for i in [
                q4ledger::MIX,
                q4ledger::SSM,
                q4ledger::FULL,
                q4ledger::ROUTER,
                q4ledger::SHARED,
                q4ledger::EXPERTS,
            ] {
                assert!(
                    q4ledger::row(i).wall_ns > 0,
                    "{} の区間が積まれていない (prefill の配線が無い)",
                    q4ledger::NAMES[i]
                );
            }
            // 全体の壁時計は別 1 本で測る (`other` は残差なので 0 でもよい)。
            assert!(
                q4ledger::row(q4ledger::OTHER).launches == 0,
                "prefill は other の区間を使わない (残差で出す)"
            );
            let lines = q4ledger::line_prefill(6);
            assert!(lines.len() >= 4, "{lines:?}");
            assert!(lines[0].starts_with("[q4ledger-pf] tok=6"), "{}", lines[0]);
            assert!(
                lines.iter().any(|l| l.contains(" experts ")),
                "experts の行が無い: {lines:?}"
            );
            // 名前付き区間の和は全体を超えない (二重計上しない) かつ other が残差。
            let total = q4ledger::prefill_wall_ns();
            assert!(total > 0, "全体の壁時計が積まれていない");
            let sum: u64 = [
                q4ledger::MIX,
                q4ledger::SSM,
                q4ledger::FULL,
                q4ledger::ROUTER,
                q4ledger::SHARED,
                q4ledger::EXPERTS,
            ]
            .iter()
            .map(|&i| q4ledger::row(i).wall_ns)
            .sum();
            assert!(
                sum <= total,
                "区間の和 {sum} が全体 {total} を超えた (二重計上)"
            );
            // `other` の行は残差として必ず出る (0 でも出る)。
            assert!(
                lines.last().unwrap().contains(" other "),
                "other (残差) の行が無い: {lines:?}"
            );

            q4ledger::reset();
            q4ledger::set_for_test(false);
        }

        /// task 65: 合成モデルの 1 回の prefill で `DENNO_CPU_GEMV_PAR` の
        /// 行並列が実際に何度も立つこと (= decode/prefill の各入口
        /// (router / ssm_beta / ssm_alpha / hc_inject) が `GgufLinear::gemv` を
        /// 通っている) と、立っても logits がビット一致すること。
        ///
        /// 合成形は小さいのでしきい値だけテスト用に 1 に下げる。しきい値そのものは
        /// `linear_gguf` の実形状テストで固定している。
        #[test]
        fn cpu_gemv_par_gate_is_live_end_to_end() {
            use crate::model::linear_gguf as lg;
            let _g = moe_test_guard();
            let _l = lg::par_test_lock();
            moe_expert::set_vnni_for_test(Some(false));
            // task 119b: 既定の F32 経路は FMA カーネル (classic と非ビット一致) に
            // なったので、off/on のビット一致の契約は exact を強制して見る (期待値は
            // 変えない。FMA 経路の契約は linear_gguf の
            // f32_router_fma_matches_classic_within_tolerance が見る)。
            moe_expert::set_f32_exact_for_test(Some(true));
            let dir = TempDir::new("cpu_gemv_par_e2e");
            let path = build(&dir);
            let f = GgufFile::open(&path).expect("open");
            let ctx: Vec<u32> = vec![1, 2, 3, 4, 5, 6];

            // 1. 旧 (off): 新経路は 1 回も立たない。
            lg::set_cpu_gemv_par_for_test(Some(false));
            lg::set_par_min_rows_for_test(Some(1));
            lg::par_reset();
            let (off, _, _) = run(&f, &ctx, false, 0);
            let launches_off = lg::par_launches();

            // 2. 新 (on): 入口ごとに立つ (6 トークン × 4 層 × 複数本)。
            lg::set_cpu_gemv_par_for_test(Some(true));
            lg::par_reset();
            let (on, _, _) = run(&f, &ctx, false, 0);
            let launches_on = lg::par_launches();
            let tasks_on = lg::par_tasks();
            lg::set_cpu_gemv_par_for_test(None);
            lg::set_par_min_rows_for_test(None);
            moe_expert::set_vnni_for_test(None);
            moe_expert::set_f32_exact_for_test(None);

            assert_eq!(launches_off, 0, "off で行並列が立った");
            assert!(
                launches_on >= 10,
                "行並列が入口からほとんど立っていない ({launches_on} 回)"
            );
            assert!(tasks_on >= launches_on, "タスク数 {tasks_on} < 領域数 {launches_on}");
            assert_bitwise_eq(&off, &on, "cpu gemv par e2e (off vs on)");
        }

        /// P56: 逐次 prefill (`forward_prefill_with(ctx, false)`) の RouterTopK は
        /// **チャンク末尾の 1 トークンだけ**。ctx が 6 トークンでも層数ぶんしか
        /// 出ない (トークンごとに送ると 6 × LAYERS 枚になってしまう)。
        #[test]
        fn router_topk_prefill_seq_sends_last_token_only() {
            let _g = moe_test_guard();
            let dir = TempDir::new("router_topk_seq");
            let path = build(&dir);
            let f = GgufFile::open(&path).expect("合成モデルを開けない");
            let ctx: Vec<u32> = vec![1, 2, 3, 4, 5, 6];

            let prev_on = denno_telemetry::enabled();
            let prev_arch = denno_telemetry::arch_enabled();
            // task 155: arch フレームは購読者 (WS) が居るときだけ組み立てる。
            // このテストは組み立て側を見るので、ビューワ 1 人ぶんを立てる。
            let _cg = denno_telemetry::ClientsGuard::new(1);
            denno_telemetry::set_enabled_for_tests(true);
            denno_telemetry::set_arch_for_tests(true);
            while denno_telemetry::try_recv_nowait().is_some() {}

            let mut m = Qwen4Exp::new(&f).expect("Qwen4Exp::new");
            let logits = m.forward_prefill_with(&ctx, false).expect("逐次 prefill");
            assert_eq!(logits.len(), VOCAB);

            let mut layers: Vec<u16> = Vec::new();
            let mut sizes: Vec<usize> = Vec::new();
            while let Some(fr) = denno_telemetry::try_recv_nowait() {
                if let denno_telemetry::Frame::RouterTopK { layer, experts, .. } = fr {
                    layers.push(layer);
                    sizes.push(experts.len());
                }
            }
            assert_eq!(
                layers,
                (0..LAYERS as u16).collect::<Vec<_>>(),
                "逐次 prefill の RouterTopK が層数ぶん (末尾トークン 1 回) でない"
            );
            assert!(sizes.iter().all(|&n| n == NEXP), "top-k の本数が違う: {sizes:?}");

            denno_telemetry::set_arch_for_tests(prev_arch);
            denno_telemetry::set_enabled_for_tests(prev_on);
        }

        /// P58: 構造テレメトリ (RoleAct / ExpertResidency / TierStats) の送出。
        ///
        /// - arch off では 3 種とも **0 枚** (ゲートが効いている)
        /// - arch on にしても logits は arch off と**ビット一致** (読むだけ)
        /// - prefill (逐次) は末尾トークン 1 回 → RoleAct が層数ぶん、
        ///   TierStats が 1 枚以上。各 RoleAct の 5 値は有限で 0..1
        /// - decode の 1 トークンでも RoleAct が層数ぶん
        /// - 使わない役割 (ssm 層の attn / full attention 層の gdn) は 0.0
        ///
        /// ExpertResidency は VRAM 経路 (GPU) が無いと 1 枚も出ないので、
        /// 出たときだけ「n ≤ 512・重複なし」を検査する (差分規則そのものは
        /// `residency_delta_emits_only_changed_layers` が固定する)。
        #[test]
        fn arch_emit_frames() {
            let _g = moe_test_guard();
            let dir = TempDir::new("arch_emit");
            let path = build(&dir);
            let f = GgufFile::open(&path).expect("合成モデルを開けない");
            let ctx: Vec<u32> = vec![1, 2, 3];

            fn drain() -> Vec<denno_telemetry::Frame> {
                let mut v = Vec::new();
                while let Some(fr) = denno_telemetry::try_recv_nowait() {
                    v.push(fr);
                }
                v
            }

            let prev_on = denno_telemetry::enabled();
            let prev_arch = denno_telemetry::arch_enabled();
            // task 155: arch フレームは購読者 (WS) が居るときだけ組み立てる。
            // このテストは組み立て側を見るので、ビューワ 1 人ぶんを立てる。
            let _cg = denno_telemetry::ClientsGuard::new(1);

            // (0) テレメトリ自体が off → arch が on でも 0 枚
            denno_telemetry::set_enabled_for_tests(false);
            denno_telemetry::set_arch_for_tests(true);
            drain();
            let mut m_none = Qwen4Exp::new(&f).expect("Qwen4Exp::new");
            m_none
                .forward_prefill_with(&ctx, false)
                .expect("prefill (telemetry off)");
            let got = drain();
            assert!(
                got.is_empty(),
                "テレメトリ off でフレームが出た: {} 枚 ({got:?})",
                got.len()
            );

            denno_telemetry::set_enabled_for_tests(true);

            // (1) arch off → フレーム 0 枚
            denno_telemetry::set_arch_for_tests(false);
            drain();
            let mut m_off = Qwen4Exp::new(&f).expect("Qwen4Exp::new");
            let l_off = m_off
                .forward_prefill_with(&ctx, false)
                .expect("prefill (arch off)");
            let got = drain();
            assert!(
                got.is_empty(),
                "arch off でフレームが出た: {} 枚 ({got:?})",
                got.len()
            );

            // (2) arch on → RoleAct は層数ぶん・TierStats は 1 枚以上・logits は不変
            denno_telemetry::set_arch_for_tests(true);
            drain();
            let mut m_on = Qwen4Exp::new(&f).expect("Qwen4Exp::new");
            let l_on = m_on
                .forward_prefill_with(&ctx, false)
                .expect("prefill (arch on)");
            assert_bitwise_eq(&l_off, &l_on, "arch on/off の prefill logits");

            let pre = drain();
            let mut role_layers: Vec<u16> = Vec::new();
            let mut kind_zero: Vec<(u16, u16, u16)> = Vec::new(); // (層, attn, gdn)
            let mut tier = 0usize;
            let mut residency = 0usize;
            let mut res_layers: Vec<u16> = Vec::new();
            let mut res_nonempty = 0usize;
            for fr in &pre {
                match fr {
                    denno_telemetry::Frame::RoleAct {
                        layer,
                        attn,
                        gdn,
                        router,
                        shex,
                        ple,
                    } => {
                        for (name, bits) in
                            [("attn", attn), ("gdn", gdn), ("router", router), ("shex", shex), ("ple", ple)]
                        {
                            let v = denno_telemetry::f16_bits_to_f32(*bits);
                            assert!(v.is_finite(), "層 {layer} の {name} が有限でない: {v}");
                            assert!(
                                (0.0..=1.0).contains(&v),
                                "層 {layer} の {name} が 0..1 の外: {v}"
                            );
                        }
                        role_layers.push(*layer);
                        kind_zero.push((*layer, *attn, *gdn));
                    }
                    denno_telemetry::Frame::ExpertResidency { layer, in_vram } => {
                        residency += 1;
                        res_layers.push(*layer);
                        if !in_vram.is_empty() {
                            res_nonempty += 1;
                        }
                        assert!(
                            in_vram.len() <= denno_telemetry::RESIDENCY_MAX,
                            "層 {layer}: 本数 {} が上限 {} を超える",
                            in_vram.len(),
                            denno_telemetry::RESIDENCY_MAX
                        );
                        let mut s = in_vram.clone();
                        s.sort_unstable();
                        s.dedup();
                        assert_eq!(s.len(), in_vram.len(), "層 {layer}: expert index が重複");
                    }
                    denno_telemetry::Frame::TierStats { .. } => tier += 1,
                    _ => {}
                }
            }
            assert_eq!(
                role_layers,
                (0..LAYERS as u16).collect::<Vec<_>>(),
                "prefill の RoleAct が層数ぶん (末尾トークン 1 回) でない"
            );
            assert!(tier >= 1, "TierStats が 1 枚も出ていない");
            // P60: 初回は**全層**送る (在庫の無い層は空)。GPU 無しのビルドでは
            // staging も LRU も空なので、48 枚すべて空になる。
            assert_eq!(
                res_layers,
                (0..LAYERS as u16).collect::<Vec<_>>(),
                "初回の ExpertResidency が全層でない (枚数 {residency})"
            );
            assert_eq!(
                res_nonempty, 0,
                "GPU 無しのビルドで在庫が非空になっている (staging の残り?)"
            );
            // 層の種別で使わない役割は 0.0 で埋める (full attention は (il+1) % INTERVAL == 0)。
            for (layer, attn, gdn) in kind_zero {
                if (layer as usize + 1) % INTERVAL == 0 {
                    assert_eq!(gdn, 0, "full attention 層 {layer} の gdn が 0 でない");
                } else {
                    assert_eq!(attn, 0, "ssm 層 {layer} の attn が 0 でない");
                }
            }

            // (3) decode の 1 トークン (t+1 == nt) でも RoleAct が層数ぶん出る
            drain();
            m_on.forward_token_tapped(&[1, 2, 3, 4], |_| {})
                .expect("decode (arch on)");
            let mut dec: Vec<u16> = Vec::new();
            let mut dec_res = 0usize;
            while let Some(fr) = denno_telemetry::try_recv_nowait() {
                match fr {
                    denno_telemetry::Frame::RoleAct { layer, .. } => dec.push(layer),
                    denno_telemetry::Frame::ExpertResidency { .. } => dec_res += 1,
                    _ => {}
                }
            }
            assert_eq!(
                dec,
                (0..LAYERS as u16).collect::<Vec<_>>(),
                "decode の RoleAct が層数ぶん出ていない"
            );
            // P60: 初回の全層送出を済ませた後は、在庫が変わらなければ 1 枚も出さない
            // (「初回」を毎トークン繰り返していないことの確認)。
            assert_eq!(
                dec_res, 0,
                "在庫が変わっていないのに ExpertResidency が {dec_res} 枚出た"
            );

            denno_telemetry::set_arch_for_tests(prev_arch);
            denno_telemetry::set_enabled_for_tests(prev_on);
        }

        /// P58: TierStats の間引き。decode は 100 トークンごと、prefill 終了時は
        /// その場で 1 枚 (トークンごとには出さない)。
        #[test]
        fn tier_stats_emitted_every_100_decode_tokens() {
            let _g = moe_test_guard();
            let dir = TempDir::new("tier_stats");
            let path = build(&dir);
            let f = GgufFile::open(&path).expect("合成モデルを開けない");

            fn tier_count() -> usize {
                let mut n = 0;
                while let Some(fr) = denno_telemetry::try_recv_nowait() {
                    if matches!(fr, denno_telemetry::Frame::TierStats { .. }) {
                        n += 1;
                    }
                }
                n
            }

            let prev_on = denno_telemetry::enabled();
            let prev_arch = denno_telemetry::arch_enabled();
            // task 155: arch フレームは購読者 (WS) が居るときだけ組み立てる。
            // このテストは組み立て側を見るので、ビューワ 1 人ぶんを立てる。
            let _cg = denno_telemetry::ClientsGuard::new(1);
            denno_telemetry::set_enabled_for_tests(true);
            denno_telemetry::set_arch_for_tests(true);
            while denno_telemetry::try_recv_nowait().is_some() {}

            let mut m = Qwen4Exp::new(&f).expect("Qwen4Exp::new");
            for _ in 0..TIER_STATS_TOKENS - 1 {
                m.arch_emit_token();
            }
            assert_eq!(tier_count(), 0, "100 トークン未満で TierStats が出た");
            m.arch_emit_token();
            assert_eq!(tier_count(), 1, "100 トークン目で TierStats が出ていない");

            m.arch_emit_prefill_done();
            assert_eq!(tier_count(), 1, "prefill 終了時に TierStats が出ていない");

            denno_telemetry::set_arch_for_tests(prev_arch);
            denno_telemetry::set_enabled_for_tests(prev_on);
        }

        // -----------------------------------------------------------------
        // P33: MoE ルータ (gate 行列積 + softmax + top-k + m 集計) の実形状ベンチ
        // -----------------------------------------------------------------

        /// 実形状のルータ 1 層ぶんの入力を組み立てる (モデル不要)。
        ///
        /// `blk.N.ffn_gate_inp.weight` は実行 GGUF で `[2560, 512] ty=F32`
        /// (5,242,880 B)。値を [-1, 1] の決定的な擬似乱数にするので、
        /// logits は N(0, 3.4) 程度に散り、top-k はほぼ一様に選ばれる。
        fn router_bench_inputs(nt: usize, ne: usize, n_in: usize) -> (GgufLinear, Vec<f32>) {
            let mut bytes = vec![0u8; ne * n_in * 4];
            for i in 0..ne * n_in {
                let r = (i as u64)
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let v = ((r >> 33) % 2001) as f32;
                bytes[i * 4..i * 4 + 4].copy_from_slice(&((v - 1000.0) * 1e-3).to_le_bytes());
            }
            let w = GgufLinear::from_slice(GgmlType::F32, n_in, ne, &bytes).expect("gate_inp F32");
            let xs: Vec<f32> = (0..nt * n_in)
                .map(|i| (((i * 37 + 11) % 401) as f32 - 200.0) * 1e-3)
                .collect();
            (w, xs)
        }

        /// `reps` 回まわして最小値 (ms) を返す。ホストは共有なので**同一実行内**で比べる。
        fn best_of_ms(reps: usize, mut f: impl FnMut()) -> f64 {
            let mut best = f64::INFINITY;
            for _ in 0..reps {
                let t = std::time::Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64() * 1e3);
            }
            best
        }

        /// ルータ 1 層ぶん (nt=512) の (a) gate 行列積 / (b) softmax /
        /// (c) top-k + m 集計 を、**旧実装と新実装の両方**で同一ループ内で測る。
        /// 1 層合計と ms/token/層 も出す。
        ///
        /// `cargo test --release -p denno-engine --lib --features cuda \
        ///   moe_router_microbench -- --ignored --nocapture --test-threads=1`
        /// を `RAYON_NUM_THREADS=8` と `16` で回す。
        #[test]
        #[ignore = "CPU ルータのマイクロベンチ (実機の ms/token は main が測る)"]
        fn moe_router_microbench() {
            gpu_gemm_off();
            let _g = moe_test_guard();
            let (nt, ne, n_in, k) = (512usize, 512usize, 2560usize, 10usize);
            let (gate, xs) = router_bench_inputs(nt, ne, n_in);
            let reps = 20usize;
            println!(
                "threads={} nt={nt} ne={ne} n_in={n_in} k={k} reps={reps}",
                rayon::current_num_threads()
            );

            // ---- (a) gate 行列積: out[t*ne + r] = dot(row_r, xs[t]) ----
            let bytes = gate.blocks().as_slice();
            let rb = bytes.len() / ne;
            let ty = gate.ty();
            let mut old = vec![0.0f32; nt * ne];
            let a_old = best_of_ms(reps, || {
                gemv_f32_pertoken(ty, bytes, rb, n_in, ne, &xs, nt, &mut old)
            });
            let mut new = vec![0.0f32; nt * ne];
            let a_new = best_of_ms(reps, || {
                let wf = dequant_rows_f32(ty, bytes, rb, n_in, ne);
                gemv_f32_blocked(&wf, n_in, ne, &xs, nt, &mut new);
            });
            assert_bitwise_eq(&old, &new, "(a) 旧 vs 新 (gate 行列積)");
            let a_new_dequant = best_of_ms(reps, || {
                let wf = dequant_rows_f32(ty, bytes, rb, n_in, ne);
                std::hint::black_box(wf.as_ptr());
            });

            // ---- (b) softmax (トークンごと) ----
            let mut probs = old.clone();
            let probs0 = probs.clone();
            let b_old = best_of_ms(reps, || {
                for t in 0..nt {
                    softmax(&mut probs[t * ne..(t + 1) * ne]);
                }
            });
            probs.copy_from_slice(&probs0);
            let b_new = best_of_ms(reps, || {
                #[cfg(feature = "par")]
                {
                    use rayon::prelude::*;
                    probs.par_chunks_mut(ne).for_each(softmax);
                }
                #[cfg(not(feature = "par"))]
                for t in 0..nt {
                    softmax(&mut probs[t * ne..(t + 1) * ne]);
                }
            });
            // (c) 用の入力は 1 回だけ作る (どちらの実装もこの probs を読む)
            probs.copy_from_slice(&probs0);
            for t in 0..nt {
                softmax(&mut probs[t * ne..(t + 1) * ne]);
            }

            // ---- (c) top-k 選択 + 再正規化 + expert ごとのトークン集計 ----
            // 旧: トークンごとの Vec + 「最大値を k 回走査」の選択
            let mut m_old = (0usize, 0usize, 0usize);
            let c_old = best_of_ms(reps, || {
                let mut sels: Vec<Vec<usize>> = Vec::with_capacity(nt);
                let mut weights: Vec<Vec<f32>> = Vec::with_capacity(nt);
                for t in 0..nt {
                    let pr = &probs[t * ne..(t + 1) * ne];
                    let mut sel = vec![0usize; k];
                    crate::model::moe::select_topk_scan(pr, k, &mut sel);
                    let mut sum = 0.0f32;
                    for &i in &sel {
                        sum += pr[i];
                    }
                    let mut w = vec![0.0f32; k];
                    for (j, &i) in sel.iter().enumerate() {
                        w[j] = pr[i] / sum;
                    }
                    sels.push(sel);
                    weights.push(w);
                }
                let mut by_expert: Vec<Vec<(usize, usize)>> = vec![Vec::new(); ne];
                for (t, sel) in sels.iter().enumerate() {
                    for (j, &ei) in sel.iter().enumerate() {
                        by_expert[ei].push((t, j));
                    }
                }
                let nonempty = by_expert.iter().filter(|p| !p.is_empty()).count();
                let m_max = by_expert.iter().map(|p| p.len()).max().unwrap_or(0);
                m_old = (nonempty, nt * k / nonempty.max(1), m_max);
                std::hint::black_box((sels, weights, by_expert));
            });

            // 新: 平らな配列 + select_nth_unstable_by の選択 + 件数を先に数える集計
            let mut m_new = (0usize, 0usize, 0usize);
            let c_new = best_of_ms(reps, || {
                let mut sels = vec![0usize; nt * k];
                let mut weights = vec![0.0f32; nt * k];
                let renorm = |pr: &[f32], sel: &[usize], w: &mut [f32]| {
                    let mut sum = 0.0f32;
                    for &i in sel {
                        sum += pr[i];
                    }
                    for (j, &i) in sel.iter().enumerate() {
                        w[j] = pr[i] / sum;
                    }
                };
                #[cfg(feature = "par")]
                {
                    use rayon::prelude::*;
                    sels.par_chunks_mut(k)
                        .zip(weights.par_chunks_mut(k))
                        .enumerate()
                        .for_each_init(Vec::<u32>::new, |buf, (t, (s, w))| {
                            let pr = &probs[t * ne..(t + 1) * ne];
                            select_topk_into(pr, k, buf, s);
                            renorm(pr, s, w);
                        });
                }
                #[cfg(not(feature = "par"))]
                {
                    let mut buf: Vec<u32> = Vec::new();
                    for t in 0..nt {
                        let pr = &probs[t * ne..(t + 1) * ne];
                        select_topk_into(pr, k, &mut buf, &mut sels[t * k..(t + 1) * k]);
                        renorm(pr, &sels[t * k..(t + 1) * k], &mut weights[t * k..(t + 1) * k]);
                    }
                }
                let mut counts = vec![0usize; ne];
                for &ei in &sels {
                    counts[ei] += 1;
                }
                let mut by_expert: Vec<Vec<(usize, usize)>> = counts
                    .iter()
                    .map(|&c| Vec::with_capacity(c))
                    .collect();
                for t in 0..nt {
                    for j in 0..k {
                        by_expert[sels[t * k + j]].push((t, j));
                    }
                }
                let nonempty = by_expert.iter().filter(|p| !p.is_empty()).count();
                let m_max = by_expert.iter().map(|p| p.len()).max().unwrap_or(0);
                m_new = (nonempty, nt * k / nonempty.max(1), m_max);
                std::hint::black_box((sels, weights, by_expert));
            });
            assert_eq!(m_old, m_new, "(c) 旧 vs 新で m の分布が違う");

            let tot_old = a_old + b_old + c_old;
            let tot_new = a_new + b_new + c_new;
            println!("                       旧 (ms/層)   新 (ms/層)   倍率");
            println!("(a) gate 行列積       {a_old:9.3}   {a_new:9.3}   x{:.2}", a_old / a_new);
            println!("(b) softmax           {b_old:9.3}   {b_new:9.3}   x{:.2}", b_old / b_new);
            println!("(c) top-k + m 集計    {c_old:9.3}   {c_new:9.3}   x{:.2}", c_old / c_new);
            println!("合計                  {tot_old:9.3}   {tot_new:9.3}   x{:.2}", tot_old / tot_new);
            println!(
                "ms/token/層            {:.4}      {:.4}",
                tot_old / nt as f64,
                tot_new / nt as f64
            );
            println!(
                "うち (a) の重み展開    {a_new_dequant:9.3} ms/層 (新経路の内訳)"
            );
            println!(
                "m 分布: nonempty={} mean={} max={}",
                m_new.0, m_new.1, m_new.2
            );
        }

        /// task 126: prefill の hc mix を実形 (BF16 down/up/inject) で A/B する手動
        /// マイクロベンチ。
        ///
        /// `hc_mix_quant_batch` (現行の prefill 経路) と、同じ入力をトークンごとに
        /// `hc_mix_quant` へ渡すループ (decode と同じ 1 トークン経路) を nt 別に同じ
        /// 回数だけ回して ms/層 を出す。このコンテナは GPU が無いので両方 CPU。
        ///
        /// cargo test --release -p denno-engine --features cuda --lib \
        ///   -- --ignored --nocapture --test-threads=1 bench_mix_batch_real_shape
        #[test]
        #[ignore = "実形状の CPU マイクロベンチ (手動)"]
        fn bench_mix_batch_real_shape() {
            use crate::model::linear_gguf::par_test_lock;
            let _g = par_test_lock();
            let hp = HyperParams {
                hc: 4,
                n_embd: 2560,
                low_rank: 320,
                eps: 1e-6,
            };
            let (e, hc, lr) = (hp.n_embd, hp.hc, hp.low_rank);
            let hc_dim = hc * e;
            // BF16 の決定的なダミー重み ([n][k] の行優先、実ファイルと同じ並び)。
            let bf16 = |k: usize, n: usize, seed: u32| -> Vec<u8> {
                let mut v = Vec::with_capacity(k * n * 2);
                for i in 0..k * n {
                    let h = (i as u32)
                        .wrapping_mul(2654435761)
                        .wrapping_add(seed.wrapping_mul(97));
                    let x = (h % 2001) as f32 / 1000.0 - 1.0;
                    v.extend_from_slice(&((x.to_bits() >> 16) as u16).to_le_bytes());
                }
                v
            };
            let down = GgufLinear::from_slice(GgmlType::Bf16, hc_dim, lr, &bf16(hc_dim, lr, 3))
                .expect("hc_down");
            let up = GgufLinear::from_slice(GgmlType::Bf16, lr, hc_dim, &bf16(lr, hc_dim, 5))
                .expect("hc_up");
            let inject = GgufLinear::from_slice(GgmlType::Bf16, hc_dim, hc, &bf16(hc_dim, hc, 7))
                .expect("hc_inj");
            let norm: Vec<f32> = (0..hc_dim).map(|i| 0.9 + ((i % 13) as f32) * 0.01).collect();
            // batch 経路そのものを測る (task 126 のゲートは既定 on なので明示的に切る)。
            set_prefill_mix_seq_for_test(Some(false));
            println!("hc_dim={hc_dim} low_rank={lr} (BF16 down/up/inject)");
            println!("nt\tbatch_ms\tper_tok_ms\tratio(batch/per)");
            for &nt in &[1usize, 4, 8, 16, 28, 32, 33, 64, 128, 256, 512] {
                let ress: Vec<HyperStreams> = (0..nt)
                    .map(|t| HyperStreams {
                        data: (0..hc_dim)
                            .map(|i| (((i * 7 + t * 13) % 23) as f32 - 11.0) * 0.03)
                            .collect(),
                        hc,
                        n_embd: e,
                    })
                    .collect();
                let reps = (256 / nt).max(4);
                let mut scratch = MixScratch::default();
                let _ = hc_mix_quant_batch(
                    &ress,
                    &norm,
                    &down,
                    &up,
                    Some(&inject),
                    &hp,
                    &mut scratch,
                )
                .expect("warmup batch");
                let t0 = std::time::Instant::now();
                for _ in 0..reps {
                    let (m, i) = hc_mix_quant_batch(
                        &ress,
                        &norm,
                        &down,
                        &up,
                        Some(&inject),
                        &hp,
                        &mut scratch,
                    )
                    .expect("batch");
                    std::hint::black_box((&m, &i));
                }
                let batch_ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
                // 1 トークン経路 (decode と同じ) を nt 回。MIX_SCRATCH はスレッドローカル。
                let t1 = std::time::Instant::now();
                for _ in 0..reps {
                    let mut mixed = Vec::with_capacity(nt * e);
                    let mut injs = Vec::with_capacity(nt * hc);
                    for t in 0..nt {
                        let (m, i) = hc_mix_quant(
                            &ress[t],
                            &norm,
                            &down,
                            &up,
                            Some(&inject),
                            &hp,
                        )
                        .expect("pertoken");
                        mixed.extend_from_slice(&m);
                        injs.extend_from_slice(&i);
                    }
                    std::hint::black_box((&mixed, &injs));
                }
                let per_ms = t1.elapsed().as_secs_f64() * 1e3 / reps as f64;
                println!(
                    "{nt}\t{batch_ms:.3}\t{per_ms:.3}\t{:.2}",
                    batch_ms / per_ms
                );
            }

            // nt=28 の区間内訳 (task 126 で batch 経路にも足した [mix-kprof])。
            let nt = 28usize;
            let ress: Vec<HyperStreams> = (0..nt)
                .map(|t| HyperStreams {
                    data: (0..hc_dim)
                        .map(|i| (((i * 7 + t * 13) % 23) as f32 - 11.0) * 0.03)
                        .collect(),
                    hc,
                    n_embd: e,
                })
                .collect();
            mix_kprof::set_for_test(true);
            let mut scratch = MixScratch::default();
            // warmup (thread-local の確保を先に済ませる)
            let _ = hc_mix_quant_batch(
                &ress,
                &norm,
                &down,
                &up,
                Some(&inject),
                &hp,
                &mut scratch,
            )
            .expect("warmup");
            mix_kprof::reset();
            let _ = hc_mix_quant_batch(
                &ress,
                &norm,
                &down,
                &up,
                Some(&inject),
                &hp,
                &mut scratch,
            )
            .expect("batch");
            if let Some(l) = mix_kprof::line() {
                println!("batch(nt=28) {l}");
            }
            mix_kprof::reset();
            for r in ress.iter() {
                let _ = hc_mix_quant(r, &norm, &down, &up, Some(&inject), &hp).expect("seq");
            }
            if let Some(l) = mix_kprof::line() {
                println!("seq(nt=28)   {l}");
            }
            mix_kprof::reset();
            mix_kprof::set_for_test(false);
            set_prefill_mix_seq_for_test(None);
        }
    }
