//! task 668: `mod tests` の前半 (合成 GGUF のヘルパーと大半の単体テスト)。
//! 後半の `mod h2` は分量が多いので `tests_h2.rs` に置き、`include!` で同じモジュールへ
//! 連結する (スコープを変えないため。`use super::*` の解決先を保つ)。

    use super::*;

    /// task 609: `DENNO_PF_ER_PAGE_HIT` が**本経路**に配線されているか。
    ///
    /// ゲートを足したのに呼び出し側へ繋ぎ忘れる (緑だが分岐が無い) のを防ぐ。
    /// `pf_early_route_prefetch` の本体が 3 つ (ゲート / 判定 / 台帳) を呼び、
    /// 止めたら `None` を返していることだけをソースで固定する。
    #[test]
    fn pf609_er_page_hit_is_wired_into_the_prefetch() {
        let src = crate::model::qwen4exp::src_text();
        let i = src
            .find("fn pf_early_route_prefetch(")
            .expect("pf_early_route_prefetch が無い");
        let body = &src[i..i + src[i..].find("\n}\n").map(|e| e + 3).unwrap_or(0)];
        assert!(body.contains("er_page_hit_on()"), "ゲートを読んでいない");
        assert!(body.contains("er_stream_page_hit("), "レートの判定をしていない");
        assert!(body.contains("note_er_page_hit()"), "発火を数えていない");
        // 止める判定は router の同期より前 (窓の判定の直後) にある。
        let gate = body.find("er_page_hit_on()").unwrap();
        let router = body.find("pf_router_dev_topk(").expect("router が無い");
        assert!(gate < router, "ゲートが router の同期より後ろにある");
        // 先読みの本体 (計画を組むループ) はゲートより後ろ。
        let plan = body.find("for (i, key) in keys.iter().enumerate()").expect("計画が無い");
        assert!(gate < plan && router < plan);
    }

    /// task 317: keep の判定は層種別ごとのゲートに従う (full = `DENNO_PF_MIXKEEP`、
    /// ssm = `DENNO_PF_MIXKEEP_SSM`)。dev-in が off・平坦でない・短いチャンクは
    /// 常に D2H (ホストが正) に戻す。
    #[cfg(feature = "cuda")]
    #[test]
    fn pf317_mix_keep_layer_per_kind() {
        use crate::cuda;
        cuda::pf_dev_in_set_for_test(2);
        cuda::pf_mix_keep_set_for_test(1);
        cuda::pf_mix_keep_ssm_set_for_test(1);
        // 両方 off: どちらも D2H する。
        assert!(!Qwen4Exp::pf_mix_keep_layer(true, 128, Some(LayerKind::FullAttn)));
        assert!(!Qwen4Exp::pf_mix_keep_layer(true, 128, Some(LayerKind::Ssm)));
        // full だけ on。
        cuda::pf_mix_keep_set_for_test(2);
        assert!(Qwen4Exp::pf_mix_keep_layer(true, 128, Some(LayerKind::FullAttn)));
        assert!(!Qwen4Exp::pf_mix_keep_layer(true, 128, Some(LayerKind::Ssm)));
        // ssm だけ on (task 317 の追加ぶん)。
        cuda::pf_mix_keep_set_for_test(1);
        cuda::pf_mix_keep_ssm_set_for_test(2);
        assert!(!Qwen4Exp::pf_mix_keep_layer(true, 128, Some(LayerKind::FullAttn)));
        assert!(Qwen4Exp::pf_mix_keep_layer(true, 128, Some(LayerKind::Ssm)));
        // 前提 (平坦・nt >= 64・層種別が分かる) が欠けたら D2H。
        assert!(
            !Qwen4Exp::pf_mix_keep_layer(false, 128, Some(LayerKind::Ssm)),
            "平坦でないのに keep"
        );
        assert!(
            !Qwen4Exp::pf_mix_keep_layer(true, 63, Some(LayerKind::Ssm)),
            "短い nt で keep"
        );
        assert!(
            !Qwen4Exp::pf_mix_keep_layer(true, 128, None),
            "層種別なしで keep"
        );
        // 連鎖が off (`DENNO_GDN_PF_CHAIN=0`) なら D2H を飛ばさない
        // (飛ばすと復元の D2H が 1 本増えるだけ)。
        set_pf_gdn_chain_for_test(Some(false));
        assert!(!Qwen4Exp::pf_mix_keep_layer(true, 128, Some(LayerKind::Ssm)));
        set_pf_gdn_chain_for_test(Some(true));
        assert!(Qwen4Exp::pf_mix_keep_layer(true, 128, Some(LayerKind::Ssm)));
        set_pf_gdn_chain_for_test(None);
        // dev-in off (`DENNO_PF_MIXDEVIN=0`) は常に D2H。
        cuda::pf_dev_in_set_for_test(1);
        assert!(!Qwen4Exp::pf_mix_keep_layer(true, 128, Some(LayerKind::Ssm)));
        cuda::pf_dev_in_set_for_test(0);
        cuda::pf_mix_keep_ssm_set_for_test(0);
    }

    /// `cuda` feature の有無で GPU GEMM を切る (無いビルドでは何もしない)。
    ///
    /// 既定ビルド (`--features cuda` なし) のテストが `crate::cuda` を直接
    /// 参照していて、cd23ec8fc 以降 `--lib` のテストがコンパイルできなかった。
    fn gpu_gemm_off() {
        #[cfg(feature = "cuda")]
        crate::cuda::set_gpu_gemm_enabled(false);
    }

    /// task 214: ヒープ実体の重みでは mmap のハンドルが無いので `slice_mapped` は
    /// `None` を返し、呼び出し側 (`submit_deferred_upload`) は従来の同期 H2D へ
    /// 落ちる。範囲の計算が `slice` と同じことも確かめる。
    #[test]
    fn task214_slice_mapped_is_none_for_heap_blocks() {
        let (n_in, n_out, n_expert) = (8usize, 4usize, 2usize);
        let per_expert = n_in * n_out * 4; // F32
        let w = ExpertW {
            name: "task214".to_string(),
            blocks: WeightBlocks::from_vec(vec![0u8; per_expert * n_expert]),
            ty: GgmlType::F32,
            n_in,
            n_out,
            n_expert,
            file: 0,
            abs: 0,
        };
        assert!(w.slice_mapped(0).expect("範囲は正しい").is_none());
        assert_eq!(w.slice(0).expect("スライス").len(), per_expert);
        assert_eq!(w.slice(1).expect("スライス").len(), per_expert);
        // 範囲外の index でも「ヒープだから None」を先に返す (呼び出し側は
        // 従来の同期経路へ落ち、そちらが index を弾く)。
        assert!(w.slice_mapped(n_expert).expect("ヒープは None").is_none());
    }

    /// task 297: prefill の終端で作業域を OS へ返す (`moe_scratch_release`)。
    /// 保持されたままにすると decode のあいだ実形で 782 MB を抱える。
    #[test]
    fn moe_scratch_release_drops_retained_workspace() {
        let _ = moe_scratch_take(); // 他のテストの残りを退避
        moe_scratch_put(MoeScratch::default());
        assert!(
            MOE_SCRATCH.lock().unwrap().is_some(),
            "put で保持されるはず (既定 2048 MiB)"
        );
        moe_scratch_release();
        assert!(
            MOE_SCRATCH.lock().unwrap().is_none(),
            "release で None に戻るはず"
        );
        // 2 回呼んでも壊れない (prefill の終端が二重に走っても安全)
        moe_scratch_release();
        assert!(MOE_SCRATCH.lock().unwrap().is_none());
    }

    /// task 297: 作業域を使い回す上限の既定は 2048 MiB (`DENNO_MOE_SCRATCH_KEEP_MB` で
    /// 動かせる)。実形の作業域 782 MB は 2048 に収まるので 48 層で使い回せる。
    /// 512 を明示すると従来 (毎層捨てる) に戻る。
    #[test]
    fn moe_scratch_keep_mb_defaults_to_2048() {
        assert_eq!(moe_scratch_keep_mb_from_value(None), 2048);
        assert_eq!(MOE_SCRATCH_KEEP_MB_DEFAULT, 2048);
        assert_eq!(moe_scratch_keep_mb_from_value(Some("512")), 512);
        assert_eq!(moe_scratch_keep_mb_from_value(Some(" 0 ")), 0);
        // 数値でない値は既定へ (壊れた env で作業域を毎層捨てない)
        assert_eq!(moe_scratch_keep_mb_from_value(Some("x")), 2048);
    }

    #[test]
    fn ty_name_covers_manifest_types() {
        // 目録に出る型の表記がそのまま出ること (エラーメッセージの読み手は目録)
        assert_eq!(ty_name(GgmlType::Q4K), "Q4_K");
        assert_eq!(ty_name(GgmlType::Q5_1), "Q5_1");
        assert_eq!(ty_name(GgmlType::Q8_0), "Q8_0");
        assert_eq!(ty_name(GgmlType::IQ4Nl), "IQ4_NL");
        assert_eq!(ty_name(GgmlType::Bf16), "BF16");
        assert_eq!(ty_name(GgmlType::F32), "F32");
    }

    #[test]
    fn error_display_is_one_line_per_item() {
        let e = Qwen4ExpError(vec!["a: 1".to_string(), "b: 2".to_string()]);
        assert_eq!(e.to_string(), "a: 1\nb: 2");
        assert_eq!(e.to_string().lines().count(), 2);
    }

    #[test]
    fn expected_dims_renders_min_last() {
        let e = ExpectedTensor {
            name: "t".to_string(),
            dims: vec![160, 40],
            ty: GgmlType::IQ4Nl,
            ty_alt: &[],
            min_last_dim: true,
        };
        assert_eq!(e.fmt_dims(), "[160, 40 以上]");
    }

    // -----------------------------------------------------------------
    // P-MOE-G1: expert キャッシュ経路 (合成 GGUF)
    // -----------------------------------------------------------------

    use std::path::{Path, PathBuf};

    use crate::model::expert_cache::CacheStats;

    /// `moe_ffn_quant` (mmap 経路 = 並列分岐に入りうる) を回すテストを直列化する。
    /// P-MOE-G5 のカウンタ (PAR_CONTRIB_CALLS / SEQ_CALLS) はプロセス全体の
    /// 静的値なので、並行に走る別テストの寄与を混ぜないため。
    static MOE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn moe_test_guard() -> std::sync::MutexGuard<'static, ()> {
        // VNNI のゲートを開くテスト (moe_expert 側) と同じミューテックス。
        moe_expert::moe_test_lock()
    }

    /// 合成 GGUF のデータ部アラインメント (`gguf.rs` の既定と同じ)。
    const ALIGN: usize = 32;
    const GT_U32: u32 = 4;
    const GT_STRING: u32 = 8;
    /// ggml_type id
    const TY_Q4_K: u32 = 12;
    const TY_Q5_1: u32 = 7;

    /// 合成モデルの次元 (Q4_K は ne[0] が 256 の倍数である必要がある)。
    const N_EMBD: usize = 256;
    const N_FF: usize = 640;
    const N_EXPERT: usize = 4;

    enum Kv {
        U32(u32),
        Str(&'static str),
    }

    impl Kv {
        fn write(&self, b: &mut Vec<u8>) {
            match self {
                Kv::U32(v) => {
                    b.extend_from_slice(&GT_U32.to_le_bytes());
                    b.extend_from_slice(&v.to_le_bytes());
                }
                Kv::Str(s) => {
                    b.extend_from_slice(&GT_STRING.to_le_bytes());
                    push_str(b, s);
                }
            }
        }
    }

    fn push_str(b: &mut Vec<u8>, s: &str) {
        b.extend_from_slice(&(s.len() as u64).to_le_bytes());
        b.extend_from_slice(s.as_bytes());
    }

    fn align_up(v: usize, a: usize) -> usize {
        (v + a - 1) & !(a - 1)
    }

    /// 最小 GGUF (v3) を書き出す。データ部は 32 B 整列。
    fn write_gguf(
        path: &Path,
        kvs: &[(&str, Kv)],
        tensors: &[(String, Vec<u64>, u32, Vec<u8>)],
    ) {
        let mut head = Vec::new();
        head.extend_from_slice(b"GGUF");
        head.extend_from_slice(&3u32.to_le_bytes());
        head.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        head.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        for (k, v) in kvs {
            push_str(&mut head, k);
            v.write(&mut head);
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

    fn f16_bytes(v: f32) -> [u8; 2] {
        half::f16::from_f32(v).to_le_bytes()
    }

    /// Q4_K (144 B = 256 要素) の 1 expert ぶん。expert ごとに値が違うこと。
    fn q4k_expert_bytes(n_in: usize, n_out: usize, e: usize) -> Vec<u8> {
        let nb = n_in / 256;
        let mut v = Vec::with_capacity(n_out * nb * 144);
        for r in 0..n_out {
            for b in 0..nb {
                v.extend_from_slice(&f16_bytes(0.02 + 0.001 * ((r + b + e) % 7) as f32));
                v.extend_from_slice(&f16_bytes(0.0005));
                for i in 0..12 {
                    v.push((0x11 + ((r * 3 + i + e) % 6)) as u8);
                }
                for i in 0..128 {
                    v.push(((i * 7 + r * 13 + b * 3 + e * 5) % 251) as u8);
                }
            }
        }
        v
    }

    /// Q5_1 (24 B = 32 要素) の 1 expert ぶん。
    fn q5_1_expert_bytes(n_in: usize, n_out: usize, e: usize) -> Vec<u8> {
        let nb = n_in / 32;
        let mut v = Vec::with_capacity(n_out * nb * 24);
        for r in 0..n_out {
            for b in 0..nb {
                v.extend_from_slice(&f16_bytes(0.02 + 0.001 * ((r + b + e) % 5) as f32));
                v.extend_from_slice(&f16_bytes(0.0004));
                let qh = ((r * 37 + b * 11 + e * 7) as u32) & 0xffff
                    | (((r as u32) & 0xffff) << 16);
                v.extend_from_slice(&qh.to_le_bytes());
                for i in 0..16 {
                    v.push(((i * 13 + r * 5 + b * 3 + e * 11) % 251) as u8);
                }
            }
        }
        v
    }

    /// テスト用の一時ディレクトリ (drop で消す)。
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            use std::sync::atomic::{AtomicU32, Ordering};
            static N: AtomicU32 = AtomicU32::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            let p = std::env::temp_dir().join(format!(
                "denno_q4exp_cache_{tag}_{}_{n}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).expect("temp dir");
            TempDir(p)
        }

        fn path(&self, name: &str) -> PathBuf {
            self.0.join(name)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// 実モデルと同じ型の expert 3 本 (Q4_K gate/up・Q5_1 down・4 expert) を持つ
    /// 合成 GGUF を書く。
    fn write_expert_gguf(dir: &TempDir) -> PathBuf {
        let path = dir.path("experts.gguf");
        let (n_in, n_ff, ne) = (N_EMBD, N_FF, N_EXPERT);
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

    /// P64: 層数と次元を指定して expert 3 行列 (`blk.{il}.ffn_{gate,up,down}_exps`)
    /// を `nl` 層ぶん書く。1 層版の [`write_expert_gguf`] と同じ型 (Q4_K gate/up・
    /// Q5_1 down) で、層ごとに expert のバイト値を変えてある。
    fn write_expert_gguf_n(
        dir: &TempDir,
        name: &str,
        n_embd: usize,
        n_ff: usize,
        ne: usize,
        nl: usize,
    ) -> PathBuf {
        let path = dir.path(name);
        let mut tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)> = Vec::new();
        for il in 0..nl {
            let (mut gate, mut up, mut down) = (Vec::new(), Vec::new(), Vec::new());
            for e in 0..ne {
                let s = e + 1000 * il;
                gate.extend_from_slice(&q4k_expert_bytes(n_embd, n_ff, s));
                up.extend_from_slice(&q4k_expert_bytes(n_embd, n_ff, s + 7));
                down.extend_from_slice(&q5_1_expert_bytes(n_ff, n_embd, s + 3));
            }
            tensors.push((
                format!("blk.{il}.ffn_gate_exps.weight"),
                vec![n_embd as u64, n_ff as u64, ne as u64],
                TY_Q4_K,
                gate,
            ));
            tensors.push((
                format!("blk.{il}.ffn_up_exps.weight"),
                vec![n_embd as u64, n_ff as u64, ne as u64],
                TY_Q4_K,
                up,
            ));
            tensors.push((
                format!("blk.{il}.ffn_down_exps.weight"),
                vec![n_ff as u64, n_embd as u64, ne as u64],
                TY_Q5_1,
                down,
            ));
        }
        let kvs = [
            ("general.architecture", Kv::Str("qwen4exp")),
            ("general.alignment", Kv::U32(ALIGN as u32)),
        ];
        write_gguf(&path, &kvs, &tensors);
        path
    }

    /// P64 ベンチ用: `real` に挙げた expert だけ実データを書き、他は 0 (疎) のままに
    /// する巨大な合成 GGUF。512 expert × 3 行列を 1 本にすると 1.5 GB になるので、
    /// [`write_gguf`] のように全体を `Vec` に組まず、ヘッダだけ書いてから
    /// `set_len` + pwrite で必要な expert を置く。
    #[cfg(feature = "par")]
    fn write_expert_gguf_sparse(
        path: &Path,
        n_embd: usize,
        n_ff: usize,
        ne: usize,
        real: &[usize],
    ) {
        use std::io::{Seek, SeekFrom, Write};

        // 1 expert ぶんのバイト数 (gate/up は Q4_K、down は Q5_1)。
        let gate_e = n_ff * (n_embd / 256) * 144;
        let down_e = n_embd * (n_ff / 32) * 24;
        let lens = [ne * gate_e, ne * gate_e, ne * down_e];
        let dims: [Vec<u64>; 3] = [
            vec![n_embd as u64, n_ff as u64, ne as u64],
            vec![n_embd as u64, n_ff as u64, ne as u64],
            vec![n_ff as u64, n_embd as u64, ne as u64],
        ];
        let names = [
            "blk.0.ffn_gate_exps.weight".to_string(),
            "blk.0.ffn_up_exps.weight".to_string(),
            "blk.0.ffn_down_exps.weight".to_string(),
        ];
        let tys = [TY_Q4_K, TY_Q4_K, TY_Q5_1];

        // ヘッダは [`write_gguf`] と同じ並び (データは 32 B 整列で順に置く)。
        let mut head = Vec::new();
        head.extend_from_slice(b"GGUF");
        head.extend_from_slice(&3u32.to_le_bytes());
        head.extend_from_slice(&(3u64).to_le_bytes());
        head.extend_from_slice(&(2u64).to_le_bytes());
        push_str(&mut head, "general.architecture");
        Kv::Str("qwen4exp").write(&mut head);
        push_str(&mut head, "general.alignment");
        Kv::U32(ALIGN as u32).write(&mut head);
        let mut offs = Vec::new();
        let mut cur = 0usize;
        for len in &lens {
            offs.push(cur);
            cur = align_up(cur + len, ALIGN);
        }
        for i in 0..3 {
            push_str(&mut head, &names[i]);
            head.extend_from_slice(&(dims[i].len() as u32).to_le_bytes());
            for d in &dims[i] {
                head.extend_from_slice(&d.to_le_bytes());
            }
            head.extend_from_slice(&tys[i].to_le_bytes());
            head.extend_from_slice(&(offs[i] as u64).to_le_bytes());
        }

        let head_len = align_up(head.len(), ALIGN);
        let mut file = std::fs::File::create(path).expect("疎 GGUF の作成");
        file.write_all(&head).expect("ヘッダ");
        file.set_len((head_len + cur) as u64).expect("set_len");
        let per = [gate_e, gate_e, down_e];
        for &e in real {
            for (i, bytes) in [
                q4k_expert_bytes(n_embd, n_ff, e),
                q4k_expert_bytes(n_embd, n_ff, e + 7),
                q5_1_expert_bytes(n_ff, n_embd, e + 3),
            ]
            .iter()
            .enumerate()
            {
                file.seek(SeekFrom::Start((head_len + offs[i] + e * per[i]) as u64))
                    .expect("seek");
                file.write_all(bytes).expect("expert のバイト列");
            }
        }
    }

    /// `moe_ffn_quant` が読むぶんだけ埋めた諸元 (validate は通さない)。
    fn cache_test_params() -> Qwen4ExpParams {
        Qwen4ExpParams {
            n_layer: 1,
            n_ctx: 8,
            n_embd: N_EMBD,
            n_head: 1,
            n_head_kv: 1,
            n_embd_head_k: 32,
            n_embd_head_v: 32,
            rms_eps: 1e-6,
            n_rot: 32,
            rope_sections: [0, 0, 0, 0],
            rope_freq_base: 1e7,
            n_expert: N_EXPERT,
            n_expert_used: 2,
            n_ff_exp: N_FF,
            n_ff_shexp: 64,
            ssm_d_conv: 4,
            ssm_d_state: 32,
            ssm_n_group: 1,
            ssm_dt_rank: 1,
            ssm_d_inner: 32,
            full_attn_interval: 4,
            n_hc: 1,
            hc_low_rank: 32,
            indexer_n_head: 1,
            indexer_head_size: 32,
            indexer_top_k: 1,
            compress_ratios: vec![0],
            n_embd_per_layer: 0,
            ple_layers: Vec::new(),
            ple_ngram_size: 0,
            ple_heads_per_ngram: 0,
            ple_conv_kernel: 0,
            ple_eos_token_id: 0,
            ple_image_token_id: 0,
            ple_layer_multipliers: Vec::new(),
            ple_head_offsets: Vec::new(),
            ple_head_vocab_sizes: Vec::new(),
            n_vocab: 8,
        }
    }

    /// 決定的な BF16 の重み (実モデルの ple_key / ple_value の候補型)。
    fn bf16_linear(k: usize, n: usize, seed: f32) -> GgufLinear {
        let mut v = Vec::with_capacity(k * n * 2);
        for i in 0..k * n {
            let x = seed + (i % 17) as f32 * 0.01;
            v.extend_from_slice(&((x.to_bits() >> 16) as u16).to_le_bytes());
        }
        GgufLinear::new(GgmlType::Bf16, k, n, WeightBlocks::from_vec(v))
            .expect("GgufLinear::new (BF16)")
    }

    /// 決定的な f32 の重み (F32 の GgufLinear)。
    fn f32_linear(k: usize, n: usize, seed: f32) -> GgufLinear {
        let mut v = Vec::with_capacity(k * n * 4);
        for i in 0..k * n {
            let x = seed + (i % 17) as f32 * 0.01;
            v.extend_from_slice(&x.to_le_bytes());
        }
        GgufLinear::new(GgmlType::F32, k, n, WeightBlocks::from_vec(v))
            .expect("GgufLinear::new (F32)")
    }

    /// `moe_ffn_quant` が触るフィールドだけ実体を入れた層。
    /// hc_* は MoE では使わないので 1x1 のダミー。
    fn cache_test_layer(f: &GgufFile, p: &Qwen4ExpParams) -> LayerW {
        layer_at(f, p, 0)
    }

    /// P64: `blk.{il}` のテンソルで組んだ層 (層番号だけ差し替える)。
    fn layer_at(f: &GgufFile, p: &Qwen4ExpParams, il: usize) -> LayerW {
        let name = |s: &str| format!("blk.{il}.{s}");
        LayerW {
            il,
            hc_attn_norm: vec![1.0; p.n_embd],
            hc_attn_down: f32_linear(1, 1, 0.5),
            hc_attn_up: f32_linear(1, 1, 0.5),
            hc_attn_inject: f32_linear(1, 1, 0.5),
            hc_ffn_norm: vec![1.0; p.n_embd],
            hc_ffn_down: f32_linear(1, 1, 0.5),
            hc_ffn_up: f32_linear(1, 1, 0.5),
            hc_ffn_inject: f32_linear(1, 1, 0.5),
            gate_inp: f32_linear(p.n_embd, p.n_expert, 0.1).with_router_gpu(true),
            gate_inp_shexp: f32_linear(p.n_embd, 1, 0.2),
            gate_exps: ExpertW::load(f, &name("ffn_gate_exps.weight"), p).expect("gate"),
            up_exps: ExpertW::load(f, &name("ffn_up_exps.weight"), p).expect("up"),
            down_exps: ExpertW::load(f, &name("ffn_down_exps.weight"), p).expect("down"),
            gate_shexp: f32_linear(p.n_embd, p.n_ff_shexp, 0.3),
            up_shexp: f32_linear(p.n_embd, p.n_ff_shexp, 0.4),
            down_shexp: f32_linear(p.n_ff_shexp, p.n_embd, 0.5),
            ple: None,
        }
    }

    /// 行列ごとの 1 expert のバイト数 (合成 GGUF の実値)。
    fn expert_lens(lw: &LayerW) -> [usize; 3] {
        [
            lw.gate_exps.slice_abs(0).expect("gate len").1,
            lw.up_exps.slice_abs(0).expect("up len").1,
            lw.down_exps.slice_abs(0).expect("down len").1,
        ]
    }

    fn cache_test_x(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.013)
            .collect()
    }

    /// P64: 層ごとに違う入力 (seed で位相をずらすだけ。決定的)。
    fn cache_test_x_seed(n: usize, seed: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 37 + seed * 101) % 101) as f32 - 50.0) * 0.013)
            .collect()
    }

    /// P64: 次元だけ差し替えた諸元 (合成 GGUF に合わせる)。
    fn params_dims(n_embd: usize, n_ff: usize, ne: usize, k: usize) -> Qwen4ExpParams {
        let mut p = cache_test_params();
        p.n_embd = n_embd;
        p.n_ff_exp = n_ff;
        p.n_expert = ne;
        p.n_expert_used = k;
        p
    }

    /// P64 ベンチ用: `x` に対して logit が expert index の降順になるルータ。
    ///
    /// 行 `e` を `x * ((ne - e) * 0.001)` にすると `logit_e = (ne - e) * 0.001 * |x|^2`
    /// で、top-k は必ず index 昇順の `0..k` になる。疎 GGUF で「実データを置いた
    /// expert」と選択を一致させるために使う。
    #[cfg(feature = "par")]
    fn gate_inp_for(x: &[f32], ne: usize) -> GgufLinear {
        let n = x.len();
        let mut v = Vec::with_capacity(n * ne);
        for e in 0..ne {
            let s = (ne - e) as f32 * 0.001;
            for &xv in x {
                v.extend_from_slice(&(xv * s).to_le_bytes());
            }
        }
        GgufLinear::new(GgmlType::F32, n, ne, WeightBlocks::from_vec(v))
            .expect("GgufLinear::new (gate_inp)")
    }

    fn assert_bitwise_eq(a: &[f32], b: &[f32], what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: 長さが違う");
        for i in 0..a.len() {
            assert_eq!(
                a[i].to_bits(),
                b[i].to_bits(),
                "{what}: 要素 {i} がビット一致しない (mmap {} / cache {})",
                a[i],
                b[i]
            );
        }
    }

    /// head (`output`) 用の Q8_0 ダミー重み (34 B ブロック = f16 スケール + i8 x 32)。
    /// 実ファイルと同じ量子型で、`linear()` と同じく `fast_q8` を立てて使う。
    fn head_q8_weights(k: usize, n: usize, seed: u32) -> Vec<u8> {
        assert_eq!(k % 32, 0, "Q8_0 のブロックは 32 要素");
        let mut b = vec![0u8; k * n / 32 * 34];
        let scale = half::f16::from_f32(0.02).to_le_bytes();
        for blk in 0..b.len() / 34 {
            let o = blk * 34;
            b[o..o + 2].copy_from_slice(&scale);
            for j in 0..32usize {
                let q = ((j as i32 * 5 + blk as i32 + seed as i32 * 11) % 13) - 6;
                b[o + 2 + j] = q as i8 as u8;
            }
        }
        b
    }

    /// head の形 (k x n、Q8_0 + fast_q8) と入力 1 本。
    fn head_gemv_case(k: usize, n: usize) -> (GgufLinear, Vec<f32>) {
        let lin = GgufLinear::from_slice(GgmlType::Q8_0, k, n, &head_q8_weights(k, n, 3))
            .expect("GgufLinear::from_slice (Q8_0)")
            .with_fast_q8(true);
        let x: Vec<f32> = (0..k).map(|i| ((i % 23) as f32 - 11.0) * 0.05).collect();
        (lin, x)
    }

    /// `DENNO_HEAD_PAR` の解釈 (既定 on、`0` / `false` / `off` だけ従来経路)。
    #[test]
    fn head_par_gate_defaults_on() {
        assert_eq!(head_row_par_min_from(None), HEAD_ROW_PAR_MIN);
        assert_eq!(head_row_par_min_from(Some("1")), HEAD_ROW_PAR_MIN);
        assert_eq!(head_row_par_min_from(Some("")), HEAD_ROW_PAR_MIN);
        assert_eq!(head_row_par_min_from(Some("0")), 0, "\"0\" で従来経路");
        assert_eq!(head_row_par_min_from(Some("false")), 0);
        assert_eq!(head_row_par_min_from(Some("OFF")), 0);
    }

    /// 明示の行並列 (`gemv_label_par`) が汎用の門番 (`gemv_label`) と**ビット一致**する。
    ///
    /// 実形の head は 151936 行なのでテストでは縮めるが、通るカーネル (Q8_0 の AVX2
    /// 融合) と行内の `b` 昇順は実形と同じ。行は独立なので、行の切り方だけが違う。
    #[test]
    fn head_par_matches_default_path_bits() {
        let (lin, x) = head_gemv_case(256, 1024);
        let mut a = vec![0f32; lin.n()];
        let mut b = vec![0f32; lin.n()];
        lin.gemv_label("output", &x, &mut a).expect("default");
        lin.gemv_label_par("output", &x, &mut b, HEAD_ROW_PAR_MIN)
            .expect("par");
        assert!(
            a.iter().any(|v| *v != 0.0),
            "全 0 (経路の生死が分からない)"
        );
        assert_bitwise_eq(&a, &b, "head: 汎用の門番 vs 明示の行並列");
    }

    /// head の値が**スレッド数に依らない** (1 スレッドと 8 スレッドでビット一致)。
    #[test]
    fn head_par_bits_do_not_depend_on_threads() {
        let (lin, x) = head_gemv_case(256, 1024);
        let run = |threads: usize| -> Vec<f32> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            let mut out = vec![0f32; lin.n()];
            pool.install(|| {
                lin.gemv_label_par("output", &x, &mut out, HEAD_ROW_PAR_MIN)
                    .expect("gemv_label_par")
            });
            out
        };
        let one = run(1);
        let eight = run(8);
        assert_bitwise_eq(&one, &eight, "head: 1 スレッド vs 8 スレッド");
    }

    /// 行並列の分岐が**実際に通る** (門番 8 行 <= 1024 行なので行並列領域が立つ)。
    ///
    /// カウンタはプロセス全体なので並列に走る他のテストの分も混ざる。ここでは
    /// 「増えた」ことだけを見る (増えない側の断定は他テストと競合して flaky になる)。
    #[test]
    fn head_row_par_branch_is_live() {
        use crate::model::moe_expert::ROW_PAR_REGIONS;
        let ld = || ROW_PAR_REGIONS.load(std::sync::atomic::Ordering::Relaxed);
        let (lin, x) = head_gemv_case(256, 1024);
        assert!(lin.n() >= HEAD_ROW_PAR_MIN, "テストの前提 (行数 >= 門番)");
        let before = ld();
        let mut out = vec![0f32; lin.n()];
        lin.gemv_label_par("output", &x, &mut out, HEAD_ROW_PAR_MIN)
            .expect("gemv_label_par");
        assert!(
            ld() > before,
            "head の行並列が立っていない (門番 {HEAD_ROW_PAR_MIN} <= 行数 {})",
            lin.n()
        );
    }

    /// `DENNO_HEAD_PROF` の行が `[head] calls=.. total=.. per_call=..us` の形で出る。
    #[test]
    fn headprof_line_format() {
        headprof::set_for_test(true);
        headprof::reset();
        let t = headprof::stamp().expect("強制 on で stamp が Some");
        headprof::mark(headprof::GEMV, Some(t), 34 * 8 * 2);
        headprof::mark(headprof::MIX, Some(t), 0);
        let g = headprof::line(headprof::GEMV).expect("gemv の行が出ない");
        assert!(g.starts_with("[head] calls="), "{g}");
        assert!(g.contains(" total="), "{g}");
        assert!(g.contains("per_call="), "{g}");
        assert!(g.contains("us"), "{g}");
        assert!(g.contains("gbps="), "gbps が無い: {g}");
        let m = headprof::line(headprof::MIX).expect("mix の行が出ない");
        assert!(m.starts_with("[head-mix] calls="), "{m}");
        assert!(!m.contains("gbps="), "mixer に gbps が付いている: {m}");
        headprof::reset();
        assert!(
            headprof::line(headprof::GEMV).is_none(),
            "reset 後に行が出る"
        );
        // off では stamp を取らない (計測ゼロ)。
        headprof::set_for_test(false);
        assert!(headprof::stamp().is_none(), "off なのに stamp が Some");
    }

    /// task 146: decode 経路 (`moe_ffn_quant_impl`) が選択 expert を数えている。
    ///
    /// ゲートを強制 on にして 1 層 1 トークン回し、積まれたセル数が top-k と
    /// 一致することを見る (「緑 = 分岐が発火した証拠ではない」の裏取り)。
    #[test]
    fn exp_stats_counts_selected_experts_on_decode_path() {
        let _g = moe_test_guard();
        let dir = TempDir::new("exp_stats_count");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let lw = cache_test_layer(&f, &p);
        let x = cache_test_x(p.n_embd);

        let _on = crate::model::exp_stats::enable_for_test();
        let before: u64 = crate::model::exp_stats::session()
            .map(|c| c.snapshot().iter().map(|v| *v as u64).sum())
            .unwrap_or(0);
        let _ = moe_ffn_quant_with(&p, &lw, &x, None, false, Route::Cpu).expect("Cpu 経路");
        let cells = crate::model::exp_stats::session().expect("カウンタが作られていない");
        let snap = cells.snapshot();
        let after: u64 = snap.iter().map(|v| *v as u64).sum();
        assert_eq!(
            after - before,
            p.n_expert_used as u64,
            "1 層 1 トークンで top-k 本ぶん積まれていない"
        );
        let (nl, ne) = cells.dims();
        assert_eq!((nl, ne), (p.n_layer, p.n_expert));
        // 積まれたのは層 0 のセルだけ (合成モデルは層 0 だけ回す)
        let layers: Vec<usize> = snap
            .iter()
            .enumerate()
            .filter(|(_, v)| **v > 0)
            .map(|(i, _)| i / ne)
            .collect();
        assert!(
            layers.iter().all(|l| *l == 0),
            "層 0 以外に積まれた: {layers:?}"
        );
    }

    /// task 263: prefill のバッチ経路が 8 トークンに 1 回だけ数えている。
    ///
    /// nt=24, k=n_expert_used なら標本は t=0,8,16 の 3 本 → 3 * k 回。
    #[test]
    fn exp_stats_counts_prefill_batch_path() {
        let _g = moe_test_guard();
        let dir = TempDir::new("exp_stats_batch");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let lw = cache_test_layer(&f, &p);
        let x = cache_test_x(p.n_embd);
        let nt = 24usize;
        let xs: Vec<f32> = (0..nt)
            .flat_map(|i| x.iter().map(move |v| v * (i as f32 + 1.0)).collect::<Vec<_>>())
            .collect();

        let _on = crate::model::exp_stats::enable_for_test();
        let before: u64 = crate::model::exp_stats::session()
            .map(|c| c.snapshot().iter().map(|v| *v as u64).sum())
            .unwrap_or(0);
        let out = moe_ffn_quant_batch(&p, &lw, &xs, nt, None).expect("batch");
        assert_eq!(out.len(), nt * p.n_embd);
        let cells = crate::model::exp_stats::session().expect("カウンタが作られていない");
        let snap = cells.snapshot();
        let after: u64 = snap.iter().map(|v| *v as u64).sum();
        let samples = nt.div_ceil(crate::model::exp_stats::PREFILL_STRIDE) as u64;
        assert_eq!(
            after - before,
            samples * p.n_expert_used as u64,
            "prefill の計数が標本数 x k でない (samples={samples}, k={})",
            p.n_expert_used
        );
        // 積まれたのは層 0 のセルだけ (合成モデルは層 0 だけ回す)
        let ne = p.n_expert;
        assert!(snap
            .iter()
            .enumerate()
            .filter(|(_, v)| **v > 0)
            .all(|(i, _)| i / ne == 0));
    }

    /// task 155: `DENNO_SHARED_TAIL` の解釈と、GPU が無いときのフォールバック。
    ///
    /// サイドストリームに積めない環境では `finish` が false を返し、呼び出し側は
    /// 従来の同期経路に落ちる (出力には 1 要素も触らない)。
    #[test]
    fn shared_tail_env_and_cpu_fallback() {
        for v in [None, Some("1"), Some("on"), Some("true")] {
            assert!(shared_tail_from_value(v), "{v:?} は on のはず");
        }
        for v in [Some("0"), Some("off"), Some("false")] {
            assert!(!shared_tail_from_value(v), "{v:?} は off のはず");
        }
        let _g = moe_test_guard();
        let dir = TempDir::new("shared_tail");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let lw = cache_test_layer(&f, &p);
        let act = cache_test_x(lw.down_shexp.k());
        let mut out = vec![7.0f32; p.n_embd];

        // 強制 on でも GPU が無ければ積めない → false (フォールバック)。
        set_shared_tail_for_test(Some(true));
        let mut ov = SharedDownOverlap::begin(&lw, &act, p.n_embd);
        assert!(
            !ov.finish(&mut out),
            "GPU 無しでサイドストリームに積めてしまった"
        );
        assert!(
            out.iter().all(|&v| v == 7.0),
            "フォールバック時に出力を触っている"
        );
        // ゲート off も同じ (積まない)。
        set_shared_tail_for_test(Some(false));
        let mut ov = SharedDownOverlap::begin(&lw, &act, p.n_embd);
        assert!(!ov.finish(&mut out));
        set_shared_tail_for_test(None);
    }

    /// P39: `DENNO_EXPERT_GPU` が未設定 (= 経路 Cpu) のときは従来の
    /// `moe_expert::expert_gemv_any` が expert 1 本あたり 3 回 (gate / up / down)
    /// 呼ばれる。GPU 経路が既定で発動しないことを**呼び出し回数**で確かめる
    /// (緑 = 経路が発動した証拠ではない、の裏取り)。
    #[test]
    fn moe_ffn_quant_cpu_route_uses_expert_gemv_any() {
        let _g = moe_test_guard();
        let dir = TempDir::new("route_cpu");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let lw = cache_test_layer(&f, &p);
        let x = cache_test_x(p.n_embd);

        // 既定 (env 未設定) の経路と、明示的に Cpu を渡した経路がビット一致すること。
        let base = moe_ffn_quant(&p, &lw, &x, None).expect("既定経路");
        let before = moe_expert::any_calls();
        let got = moe_ffn_quant_with(&p, &lw, &x, None, false, Route::Cpu).expect("Cpu 経路");
        let calls = moe_expert::any_calls() - before;

        assert_eq!(
            calls,
            3 * p.n_expert_used,
            "CPU 経路で expert_gemv_any が 3 本 × top-k 回呼ばれていない (既定で GPU 経路に入った?)"
        );
        assert_bitwise_eq(&base, &got, "route=Cpu");
    }

    // ------------------------------------------------------------------
    // task 124: shared expert の重ね打ち (gate/up を router/expert と重ねる)
    // ------------------------------------------------------------------

    /// ゲートの解釈 (純関数)。既定 on、`0` / `off` / `false` だけ off。
    #[test]
    fn shared_overlap_gate_from_value() {
        assert!(shared_overlap_from_value(None), "未設定は on");
        assert!(shared_overlap_from_value(Some("1")));
        assert!(shared_overlap_from_value(Some("on")));
        assert!(shared_overlap_from_value(Some("yes")));
        assert!(shared_overlap_from_value(Some("")));
        for v in ["0", "off", "false"] {
            assert!(!shared_overlap_from_value(Some(v)), "{v} は off");
        }
    }

    /// Q8_0 のダミー重み (34 B ブロック: f16 スケール + i8 x 32)。
    #[cfg(feature = "cuda")]
    fn overlap_q8_weights(k: usize, n: usize, seed: u32) -> Vec<u8> {
        assert_eq!(k % 32, 0, "Q8_0 のブロックは 32 要素");
        let mut b = vec![0u8; k * n / 32 * 34];
        let scale = half::f16::from_f32(0.02).to_le_bytes();
        for blk in 0..b.len() / 34 {
            let o = blk * 34;
            b[o..o + 2].copy_from_slice(&scale);
            for j in 0..32usize {
                let q = ((j as i32 + blk as i32 * 5 + seed as i32 * 7) % 13) - 6;
                b[o + 2 + j] = q as i8 as u8;
            }
        }
        b
    }

    /// task 124: shared expert だけ Q8_0 にした層。
    ///
    /// 実機の shared expert と同じ型。fixture 既定の F32 のままだと GPU 経路の
    /// 対象外 (`plan_moe_gemv` が `None`) で、重ね打ちの検査にならない。
    #[cfg(feature = "cuda")]
    fn overlap_test_layer(f: &GgufFile, p: &Qwen4ExpParams) -> LayerW {
        let mut lw = layer_at(f, p, 0);
        let q8 = |k: usize, n: usize, seed: u32| {
            GgufLinear::from_slice(GgmlType::Q8_0, k, n, &overlap_q8_weights(k, n, seed))
                .expect("GgufLinear::from_slice (Q8_0)")
        };
        lw.gate_shexp = q8(p.n_embd, p.n_ff_shexp, 11);
        lw.up_shexp = q8(p.n_embd, p.n_ff_shexp, 13);
        lw.down_shexp = q8(p.n_ff_shexp, p.n_embd, 17);
        lw
    }

    /// task 187: 4 発経路 (`DENNO_DEC_SHARED_DEV`) の偽 GPU 実行。
    /// gate → up → silu*up → down を 1 本のサイドストリームに積む経路を、
    /// ホストの `expert_gemv` で計算して join する。
    #[cfg(feature = "cuda")]
    #[test]
    fn shared_dev_fires_and_matches_two_step_path() {
        let _g = moe_test_guard();
        let dir = TempDir::new("shared_dev");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let lw = overlap_test_layer(&f, &p);
        let x = cache_test_x(p.n_embd);

        // 1) 従来 (2 段) 経路。
        set_dec_shared_dev_for_test(Some(false));
        set_shared_overlap_for_test(None);
        let want = moe_ffn_quant_with(&p, &lw, &x, None, false, Route::Cpu)
            .expect("従来経路");
        let s0 = crate::cuda::shared_overlap_stats();

        // 2) 4 発経路 (偽 GPU)。
        set_dec_shared_dev_for_test(Some(true));
        let _fake = crate::cuda::SharedOverlapFakeGuard::new();
        set_shared_overlap_for_test(Some(true));
        let got = moe_ffn_quant_with(&p, &lw, &x, None, false, Route::Cpu)
            .expect("4 発経路");
        let s1 = crate::cuda::shared_overlap_stats();
        set_shared_overlap_for_test(None);
        set_dec_shared_dev_for_test(None);

        assert!(
            s1.taken > s0.taken,
            "4 発経路が GPU へ積まれていない (taken {} -> {})",
            s0.taken,
            s1.taken
        );
        assert!(
            s1.joined > s0.joined,
            "4 発経路の join が起きていない (joined {} -> {})",
            s0.joined,
            s1.joined
        );
        assert_bitwise_eq(&want, &got, "4 発経路 vs 2 段経路");
    }

    /// task 124: 偽 GPU を立てたままゲート on で MoE を 1 回回す。
    #[cfg(feature = "cuda")]
    fn shared_overlap_run(p: &Qwen4ExpParams, lw: &LayerW, x: &[f32]) -> Vec<f32> {
        let _fake = crate::cuda::SharedOverlapFakeGuard::new();
        set_shared_overlap_for_test(Some(true));
        let out = moe_ffn_quant_with(p, lw, x, None, false, Route::Cpu).expect("重ね打ち経路");
        set_shared_overlap_for_test(None);
        out
    }

    /// task 124: ゲート on で重ね打ちが発火し (begin / taken / join が増える)、
    /// join の写し先と順序が正しい (ゲート off の従来経路とビット一致)。
    #[cfg(feature = "cuda")]
    #[test]
    fn shared_overlap_fires_and_matches_off_path() {
        let _g = moe_test_guard();
        let dir = TempDir::new("shared_overlap_on");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let lw = overlap_test_layer(&f, &p);
        let x = cache_test_x(p.n_embd);

        // 1) ゲート off = 従来の順序。begin は 1 回も呼ばれない。
        set_shared_overlap_for_test(Some(false));
        let s0 = crate::cuda::shared_overlap_stats();
        let off = moe_ffn_quant_with(&p, &lw, &x, None, false, Route::Cpu).expect("ゲート off");
        let s1 = crate::cuda::shared_overlap_stats();
        set_shared_overlap_for_test(None);
        assert_eq!(s1.begin, s0.begin, "ゲート off なのに begin が呼ばれた");

        // 2) ゲート on + 偽 GPU = 重ね打ちの join 経路。
        let on = shared_overlap_run(&p, &lw, &x);
        let s2 = crate::cuda::shared_overlap_stats();
        assert!(s2.begin > s1.begin, "重ね打ちの begin が発火していない");
        assert!(
            s2.taken > s1.taken,
            "GPU へ積めていない (偽 GPU が効いていない)"
        );
        assert!(s2.joined > s1.joined, "join が発火していない");
        assert_eq!(s2.aborted, s1.aborted, "abort が増えた");
        assert_eq!(s2.failed, s1.failed, "join 失敗が増えた");
        assert_bitwise_eq(&on, &off, "shared overlap (on vs off)");

        // 3) ゲート on でも GPU が無ければ従来経路 (このコンテナがその状態)。
        set_shared_overlap_for_test(Some(true));
        let s3 = crate::cuda::shared_overlap_stats();
        let cpu = moe_ffn_quant_with(&p, &lw, &x, None, false, Route::Cpu).expect("GPU 無し");
        let s4 = crate::cuda::shared_overlap_stats();
        set_shared_overlap_for_test(None);
        assert!(s4.begin > s3.begin, "begin が呼ばれていない");
        assert_eq!(s4.taken, s3.taken, "GPU が無いのに積めたことになった");
        assert_bitwise_eq(&cpu, &off, "shared overlap (GPU 無し)");
    }

    /// task 124: join が失敗したら従来の同期経路へ落ちる。
    ///
    /// 出力は同じ・abort が二重に走らないこと。GPU 無しでも到達できる
    /// (偽 GPU + 失敗フック)。
    #[cfg(feature = "cuda")]
    #[test]
    fn shared_overlap_join_failure_falls_back() {
        let _g = moe_test_guard();
        let dir = TempDir::new("shared_overlap_fail");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let lw = overlap_test_layer(&f, &p);
        let x = cache_test_x(p.n_embd);

        set_shared_overlap_for_test(Some(false));
        let off = moe_ffn_quant_with(&p, &lw, &x, None, false, Route::Cpu).expect("ゲート off");
        set_shared_overlap_for_test(None);

        let s0 = crate::cuda::shared_overlap_stats();
        let got = {
            let _fake = crate::cuda::SharedOverlapFakeGuard::new();
            let _fail = crate::cuda::SharedOverlapFailGuard::new();
            set_shared_overlap_for_test(Some(true));
            let r = moe_ffn_quant_with(&p, &lw, &x, None, false, Route::Cpu);
            set_shared_overlap_for_test(None);
            r.expect("join 失敗でも計算できる")
        };
        let s1 = crate::cuda::shared_overlap_stats();
        assert!(s1.taken > s0.taken, "偽 GPU で積めていない");
        assert!(s1.failed > s0.failed, "join 失敗の経路が発火していない");
        assert_eq!(
            s1.aborted, s0.aborted,
            "join 失敗のあとに Drop が二重に drain した"
        );
        assert_bitwise_eq(&got, &off, "join 失敗 → 従来経路");
    }

    /// task 124: join せずに捨てても in-flight が残らない (Drop が drain する)。
    /// そのあとの層が普通に計算できることまで見る (壊れた状態を持ち越さない)。
    #[cfg(feature = "cuda")]
    #[test]
    fn shared_overlap_drop_without_join_drains() {
        let _g = moe_test_guard();
        let dir = TempDir::new("shared_overlap_abort");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let lw = overlap_test_layer(&f, &p);
        let x = cache_test_x(p.n_embd);

        set_shared_overlap_for_test(Some(false));
        let off = moe_ffn_quant_with(&p, &lw, &x, None, false, Route::Cpu).expect("ゲート off");
        set_shared_overlap_for_test(None);

        let s0 = crate::cuda::shared_overlap_stats();
        {
            let _fake = crate::cuda::SharedOverlapFakeGuard::new();
            set_shared_overlap_for_test(Some(true));
            // ここは 2 段経路 (gate/up の 2 本) の契約を見るテスト。task 187 の
            // 4 発経路は既定 on なので、明示的に切って 2 本を固定する。
            set_dec_shared_dev_for_test(Some(false));
            let ov = SharedExpertOverlap::begin(&lw, &x);
            set_dec_shared_dev_for_test(None);
            set_shared_overlap_for_test(None);
            assert!(!ov.pending.is_empty(), "偽 GPU で積めていない");
            assert_eq!(ov.pending.len(), 2, "積んだ本数が 2 でない");
            // join せずに捨てる (エラー経路の想定)
        }
        let s1 = crate::cuda::shared_overlap_stats();
        assert_eq!(s1.taken, s0.taken + 1, "積めていない");
        assert_eq!(s1.aborted, s0.aborted + 1, "Drop の drain が走っていない");
        assert_eq!(s1.joined, s0.joined, "join していないのに joined が増えた");

        // 壊れたまま残っていない: もう一度積んで join でき、値も同じ。
        let again = shared_overlap_run(&p, &lw, &x);
        assert_bitwise_eq(&again, &off, "abort のあとの重ね打ち");
    }

    /// M=1 VNNI のゲートを開けたとき、decode の CPU 経路が `expert_gemv_any` ではなく
    /// VNNI カーネルを通ること (配線の確認。緑 = 経路が発動した証拠ではないので数を数える)。
    #[test]
    fn moe_ffn_quant_cpu_route_uses_vnni_when_enabled() {
        let _g = moe_test_guard();
        if !moe_expert::vnni_supported() {
            eprintln!("skip moe_ffn_quant_cpu_route_uses_vnni_when_enabled: avx512vnni が無い");
            return;
        }
        let dir = TempDir::new("route_vnni");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let lw = cache_test_layer(&f, &p);
        let x = cache_test_x(p.n_embd);

        let before_any = moe_expert::any_calls();
        let before_vnni = moe_expert::gemv_vnni_calls();
        let got = {
            let _gate = moe_expert::VnniGateGuard::on();
            moe_ffn_quant_with(&p, &lw, &x, None, false, Route::Cpu).expect("Cpu 経路")
        };
        let vnni = moe_expert::gemv_vnni_calls() - before_vnni;
        let any = moe_expert::any_calls() - before_any;
        // `any_calls` はスレッドローカルなので厳密に見る。`gemv_vnni_calls` は
        // プロセス全体の数え上げで、moe のガードを取らない他テストの forward が
        // 混ざりうるので下限で見る (0 なら配線が死んでいる)。
        assert_eq!(any, 0, "VNNI 経路なのに expert_gemv_any も呼ばれている");
        assert!(
            vnni >= 3 * p.n_expert_used,
            "VNNI カーネルが expert 1 本あたり 3 回 (gate / up / down) 呼ばれていない ({vnni})"
        );
        assert!(
            got.iter().any(|v| *v != 0.0),
            "出力が全 0 (経路の生死が分からない)"
        );
        assert!(got.iter().all(|v| v.is_finite()), "非有限値が出た");
    }

    // ===========================================================================
    // task 120: decode (m=1) の expert 剪定ゲート (DENNO_EXPERT_TOPK / MIN_W) と
    // 重み統計 (DENNO_EXPERT_W_STATS)。既定は全部 off で、off のときの経路と値は
    // 親経路と 1 ビットも違わない。
    // ===========================================================================

    /// `DENNO_EXPERT_TOPK` の真理値表。
    #[test]
    fn expert_topk_gate_truth_table() {
        let k10 = expert_topk_from_value(None, 10);
        assert_eq!(k10, None, "未設定は off");
        assert_eq!(expert_topk_from_value(Some(""), 10), None, "空は off");
        assert_eq!(expert_topk_from_value(Some("   "), 10), None, "空白のみは off");
        assert_eq!(expert_topk_from_value(Some("abc"), 10), None, "非数は off");
        assert_eq!(expert_topk_from_value(Some("0"), 10), None, "0 は off");
        assert_eq!(expert_topk_from_value(Some("-1"), 10), None, "負は off");
        assert_eq!(expert_topk_from_value(Some("1.5"), 10), None, "小数は off");
        assert_eq!(
            expert_topk_from_value(Some("11"), 10),
            None,
            "k 超過は無効 = off"
        );
        assert_eq!(expert_topk_from_value(Some("1"), 10), Some(1));
        assert_eq!(expert_topk_from_value(Some(" 9 "), 10), Some(9), "前後の空白は許す");
        assert_eq!(
            expert_topk_from_value(Some("10"), 10),
            Some(10),
            "n == k は有効 (切る相手が無い = 値は親経路と同じ)"
        );
        // k が小さいモデルでは範囲判定も働く。
        assert_eq!(expert_topk_from_value(Some("3"), 2), None);
        assert_eq!(expert_topk_from_value(Some("2"), 2), Some(2));
        assert_eq!(expert_topk_from_value(Some("1"), 2), Some(1));
    }

    /// `DENNO_EXPERT_MIN_W` の真理値表。
    #[test]
    fn expert_min_w_gate_truth_table() {
        assert_eq!(expert_min_w_from_value(None), None, "未設定は off");
        assert_eq!(expert_min_w_from_value(Some("")), None, "空は off");
        assert_eq!(expert_min_w_from_value(Some("  ")), None, "空白のみは off");
        assert_eq!(expert_min_w_from_value(Some("0")), None, "0 は off");
        assert_eq!(expert_min_w_from_value(Some("0.0")), None, "0.0 は off");
        assert_eq!(expert_min_w_from_value(Some("-0.1")), None, "負は off");
        assert_eq!(expert_min_w_from_value(Some("NaN")), None, "NaN は off");
        assert_eq!(expert_min_w_from_value(Some("inf")), None, "inf は off");
        assert_eq!(expert_min_w_from_value(Some("abc")), None, "非数は off");
        assert_eq!(expert_min_w_from_value(Some("0.05")), Some(0.05));
        assert_eq!(expert_min_w_from_value(Some(" 0.1 ")), Some(0.1), "前後の空白は許す");
        assert_eq!(
            expert_min_w_from_value(Some("1")),
            Some(1.0),
            "1.0 は「最高重みの 1 個を常に残す」の意味で有効"
        );
    }

    /// `DENNO_EXPERT_W_STATS` の真理値表。
    #[test]
    fn expert_w_stats_gate_truth_table() {
        assert!(!expert_w_stats_from_value(None), "未設定は off");
        assert!(!expert_w_stats_from_value(Some("")), "空は off");
        assert!(!expert_w_stats_from_value(Some("0")), "0 は off");
        assert!(!expert_w_stats_from_value(Some("2")), "2 は off (曖昧値は拒む)");
        assert!(!expert_w_stats_from_value(Some("yes")), "yes は off");
        assert!(expert_w_stats_from_value(Some("1")));
        assert!(expert_w_stats_from_value(Some("on")));
        assert!(expert_w_stats_from_value(Some("ON")));
        assert!(expert_w_stats_from_value(Some("true")));
    }

    /// `DENNO_EXPERT_TOPK=1` で expert が本当に 1 本だけ走ること (本数の裏取り) と、
    /// 出力が「`n_expert_used=1` のモデル」の親経路とビット一致すること。
    ///
    /// top-k の確率上位 n 個 = `select_topk(probs, n)` と同じ集合なので、
    /// ゲートで切った結果は最初から k=n のモデルと同じ選択・同じ再正規化になる
    /// (丸め順序も同じ: index 昇順の逐次)。さらに上書きが有効でも両ゲートが
    /// off なら親経路とビット一致する (ゲート機構そのものが値を触らない証拠)。
    #[test]
    fn expert_topk_gate_runs_fewer_experts_and_matches_topn_params() {
        let _g = moe_test_guard();
        set_expert_prune_measurement(None);
        let dir = TempDir::new("p120_topk");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let lw = cache_test_layer(&f, &p);
        let x = cache_test_x(p.n_embd);

        // 親経路 (ゲート off): expert 2 本ぶん (gate/up/down = 6 GEMV)。
        let off = moe_ffn_quant_with(&p, &lw, &x, None, false, Route::Cpu).expect("off");
        let before = moe_expert::any_calls();
        let _ = moe_ffn_quant_with(&p, &lw, &x, None, false, Route::Cpu).expect("off2");
        assert_eq!(
            moe_expert::any_calls() - before,
            3 * p.n_expert_used,
            "ゲート off で expert 本数が変わった"
        );

        // 上書きが有効だが両ゲート off → 親経路とビット一致。
        set_expert_prune_measurement(Some(ExpertPruneMeasurement {
            topk: None,
            min_w: None,
            w_stats: false,
        }));
        let noop = moe_ffn_quant_with(&p, &lw, &x, None, false, Route::Cpu).expect("no-op");
        set_expert_prune_measurement(None);
        assert_bitwise_eq(&off, &noop, "task120: 上書き有効・両ゲート off");

        // TOPK=1: expert 1 本ぶん (3 GEMV) だけ走り、k=1 のモデルとビット一致。
        let mut p1 = cache_test_params();
        p1.n_expert_used = 1;
        let ref1 = moe_ffn_quant_with(&p1, &lw, &x, None, false, Route::Cpu).expect("k=1");
        set_expert_prune_measurement(Some(ExpertPruneMeasurement {
            topk: Some(1),
            min_w: None,
            w_stats: false,
        }));
        let before = moe_expert::any_calls();
        let got = moe_ffn_quant_with(&p, &lw, &x, None, false, Route::Cpu).expect("TOPK=1");
        let calls = moe_expert::any_calls() - before;
        set_expert_prune_measurement(None);
        assert_eq!(
            calls,
            3,
            "TOPK=1 なのに expert 1 本ぶん (3 GEMV) を走っていない ({calls})"
        );
        assert_bitwise_eq(&got, &ref1, "task120: TOPK=1 vs n_expert_used=1 の親経路");
    }

    /// 合成ルータ: expert 0 と 1 を同値 (全行 1.0)、2 と 3 を 0.0 にする。
    /// `x` を全正にすると top-2 は必ず `[0, 1]` で、再正規化重みはちょうど
    /// `(0.5, 0.5)` になる (MIN_W の閾値の両側を確定して作れる)。
    fn p120_router_layer(f: &GgufFile, p: &Qwen4ExpParams, eq_rows: bool) -> LayerW {
        let mut lw = cache_test_layer(f, p);
        let mut v = Vec::with_capacity(p.n_embd * p.n_expert * 4);
        for e in 0..p.n_expert {
            let val: f32 = if !eq_rows {
                // 独占ルータ: expert 0 だけ巨大、残りは 0。
                if e == 0 {
                    10.0
                } else {
                    0.0
                }
            } else if e < 2 {
                1.0
            } else {
                0.0
            };
            for _ in 0..p.n_embd {
                v.extend_from_slice(&val.to_le_bytes());
            }
        }
        lw.gate_inp = GgufLinear::new(
            GgmlType::F32,
            p.n_embd,
            p.n_expert,
            WeightBlocks::from_vec(v),
        )
        .expect("gate_inp (F32)");
        lw
    }

    fn p120_flat_x(n: usize) -> Vec<f32> {
        vec![0.05; n]
    }

    /// task 217: expert 無効化がルータに効く。
    ///
    /// - 無効 expert は選ばれず、残りの上位 k 本が index 昇順で選ばれる
    /// - 重みは残りだけで再正規化される
    /// - 出力はマスク無しと変わる (無効 expert が寄与しない)
    /// - 残りの確率が全部 0 でも NaN にせず一様に分ける
    /// - 空の設定は従来とビット一致
    #[test]
    fn intervene_expert_off_changes_router_and_renormalizes() {
        let _g = moe_test_guard();
        set_expert_prune_measurement(None);
        let dir = TempDir::new("iv_expert_off");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let x = p120_flat_x(p.n_embd);
        let shape = intervene::Shape {
            n_layer: p.n_layer,
            n_expert: p.n_expert,
            n_expert_used: p.n_expert_used,
            n_hc: p.n_hc,
        };
        let cfg = intervene::config_for_test(&shape, vec![(0, 0)], vec![], vec![]).expect("cfg");
        let iv = IvView::of(&cfg);

        // (1) 2 本が同点のルータ: expert 0 を切ると 1 と 2 が残る。
        let lw = p120_router_layer(&f, &p, true);
        let mut probs = vec![0.0f32; p.n_expert];
        lw.gate_inp.gemv_label("gate_inp", &x, &mut probs).expect("gate_inp");
        softmax(&mut probs);
        assert_eq!(select_topk(&probs, p.n_expert_used), vec![0, 1]);
        let mask = iv.expert_mask(0).expect("層 0 の mask");
        let sel = select_topk_mask(&probs, p.n_expert_used, mask);
        assert_eq!(sel, vec![1, 2], "無効 expert を除いた上位 k が違う: {sel:?}");

        let base = moe_ffn_quant_full_opts(
            &p, &lw, &x, None, false, Route::Cpu, None, true, false, IvView::none(),
        )
        .expect("no intervention");
        let off = moe_ffn_quant_full_opts(
            &p, &lw, &x, None, false, Route::Cpu, None, true, false, iv,
        )
        .expect("expert 0 off");
        assert!(
            base.iter().zip(off.iter()).any(|(a, b)| a.to_bits() != b.to_bits()),
            "expert を切ったのに出力が同じ"
        );

        // 重みの記録: 残り 2 本で再正規化 (probs[1] >> probs[2])。
        set_expert_prune_measurement(Some(ExpertPruneMeasurement {
            topk: None,
            min_w: None,
            w_stats: true,
        }));
        let _ = moe_ffn_quant_full_opts(
            &p, &lw, &x, None, false, Route::Cpu, None, true, false, iv,
        )
        .expect("expert 0 off (stats)");
        set_expert_prune_measurement(None);
        let st = take_expert_w_stats().expect("stats");
        assert_eq!(st.calls(), 1);
        assert_eq!(st.mean_experts_per_call(), 2.0, "k 本より少ない");
        let w1 = st.mean_rank_weight(1).expect("rank1");
        let w2 = st.mean_rank_weight(2).expect("rank2");
        assert!((w1 - 1.0).abs() < 1e-4, "rank1 = {w1} (残りの最大に寄るはず)");
        assert!(w2 > 0.0 && w2 < 1e-3, "rank2 = {w2} (残りの 2 番目はほぼ 0)");
        assert!(w1.is_finite() && w2.is_finite(), "重みが有限でない");

        // (2) 独占ルータ: 残りの確率は softmax のアンダーフローで 0。
        //     0/0 にせず一様 0.5 / 0.5 にする。
        let lw2 = p120_router_layer(&f, &p, false);
        set_expert_prune_measurement(Some(ExpertPruneMeasurement {
            topk: None,
            min_w: None,
            w_stats: true,
        }));
        let _ = moe_ffn_quant_full_opts(
            &p, &lw2, &x, None, false, Route::Cpu, None, true, false, iv,
        )
        .expect("exclusive router, expert 0 off");
        set_expert_prune_measurement(None);
        let st = take_expert_w_stats().expect("stats");
        let a = st.mean_rank_weight(1).expect("rank1");
        let b = st.mean_rank_weight(2).expect("rank2");
        assert!(a.is_finite() && b.is_finite(), "NaN が出た: rank1={a} rank2={b}");
        assert!((a - 0.5).abs() < 1e-6 && (b - 0.5).abs() < 1e-6, "一様でない: {a} {b}");

        // 空の設定は従来と 1 ビットも変わらない。
        let empty = intervene::config_for_test(&shape, vec![], vec![], vec![]).expect("empty");
        let iv_empty = IvView::of(&empty);
        assert!(!iv_empty.is_active());
        let same = moe_ffn_quant_full_opts(
            &p, &lw, &x, None, false, Route::Cpu, None, true, false, iv_empty,
        )
        .expect("empty cfg");
        for i in 0..base.len() {
            assert_eq!(same[i].to_bits(), base[i].to_bits(), "空設定でビットが動いた (i={i})");
        }
    }

    /// task 217: block skip の意味。
    ///
    /// `moe` を切った出力 + `shared` を切った出力 = 切らない出力 (どれも f32 の
    /// 同じ足し込み順なのでビット一致する)。片方だけ切ると片方の寄与だけが残る。
    #[test]
    fn intervene_block_skip_splits_moe_and_shared() {
        let _g = moe_test_guard();
        set_expert_prune_measurement(None);
        let dir = TempDir::new("iv_block_skip");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let lw = cache_test_layer(&f, &p);
        let x = cache_test_x(p.n_embd);
        let shape = intervene::Shape {
            n_layer: p.n_layer,
            n_expert: p.n_expert,
            n_expert_used: p.n_expert_used,
            n_hc: p.n_hc,
        };
        let run = |iv: IvView<'_>| {
            moe_ffn_quant_full_opts(&p, &lw, &x, None, false, Route::Cpu, None, true, false, iv)
                .expect("moe")
        };
        let full = run(IvView::none());
        let moe_off = run(IvView::of(
            &intervene::config_for_test(&shape, vec![], vec![(0, IvBlock::Moe)], vec![])
                .expect("moe off"),
        ));
        let shared_off = run(IvView::of(
            &intervene::config_for_test(&shape, vec![], vec![(0, IvBlock::Shared)], vec![])
                .expect("shared off"),
        ));
        let both_off = run(IvView::of(
            &intervene::config_for_test(
                &shape,
                vec![],
                vec![(0, IvBlock::Moe), (0, IvBlock::Shared)],
                vec![],
            )
            .expect("both off"),
        ));
        assert!(both_off.iter().all(|v| *v == 0.0), "両方切ったのに 0 でない");
        for j in 0..full.len() {
            assert_eq!(
                full[j].to_bits(),
                (moe_off[j] + shared_off[j]).to_bits(),
                "moe_off + shared_off が full と違う (j={j})"
            );
        }
        assert!(
            moe_off.iter().zip(&shared_off).any(|(a, b)| a.to_bits() != b.to_bits()),
            "片方だけ切ったのに寄与が同じ"
        );
    }

    /// `DENNO_EXPERT_MIN_W` の発火: 閾値の両側で expert 本数が変わり、
    /// 「落とす相手が無い」設定では親経路とビット一致、「1 本に絞る」設定では
    /// `n_expert_used=1` の親経路とビット一致する。
    #[test]
    fn expert_min_w_gate_drops_low_weight_experts() {
        let _g = moe_test_guard();
        set_expert_prune_measurement(None);
        let dir = TempDir::new("p120_minw");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let lw = p120_router_layer(&f, &p, true);
        let x = p120_flat_x(p.n_embd);

        // 親経路: 重み (0.5, 0.5)、expert 2 本 (6 GEMV)。
        let off = moe_ffn_quant_with(&p, &lw, &x, None, false, Route::Cpu).expect("off");

        // MIN_W=0.5: 重みは x 未満ではない (>= 0.5 を残す) → 1 本も落ちない。
        // 2 回目の再正規化は丸ごと飛ぶので親経路とビット一致。
        set_expert_prune_measurement(Some(ExpertPruneMeasurement {
            topk: None,
            min_w: Some(0.5),
            w_stats: false,
        }));
        let before = moe_expert::any_calls();
        let keep_all = moe_ffn_quant_with(&p, &lw, &x, None, false, Route::Cpu).expect("MIN_W=0.5");
        assert_eq!(
            moe_expert::any_calls() - before,
            3 * p.n_expert_used,
            "MIN_W=0.5 で重み 0.5 の expert が落ちた (境界は「残す」側)"
        );
        assert_bitwise_eq(&off, &keep_all, "task120: MIN_W=0.5 (落ちる expert 無し)");

        // MIN_W=0.51: 重み 0.5 は x 未満 → expert 1 が落ちて 1 本だけ残る。
        // k=1 の親経路 (重みちょうど 1.0) とビット一致する。
        let mut p1 = cache_test_params();
        p1.n_expert_used = 1;
        let ref1 = moe_ffn_quant_with(&p1, &lw, &x, None, false, Route::Cpu).expect("k=1");
        set_expert_prune_measurement(Some(ExpertPruneMeasurement {
            topk: None,
            min_w: Some(0.51),
            w_stats: false,
        }));
        let before = moe_expert::any_calls();
        let got = moe_ffn_quant_with(&p, &lw, &x, None, false, Route::Cpu).expect("MIN_W=0.51");
        let calls = moe_expert::any_calls() - before;
        set_expert_prune_measurement(None);
        assert_eq!(
            calls,
            3,
            "MIN_W=0.51 で expert 1 本ぶん (3 GEMV) を走っていない ({calls})"
        );
        assert_bitwise_eq(&got, &ref1, "task120: MIN_W=0.51 vs n_expert_used=1 の親経路");
    }

    /// 役割テレメ (P54 の router 活性) は剪定後の expert 集合を示す。
    ///
    /// 独占ルータ (expert 0 がほぼ 1.0) で TOPK=1 を掛けると、router の活性は
    /// 「残った expert 1 本の確率」と一致する (ゲート off なら top-k 全員の和)。
    #[test]
    fn expert_router_rms_shows_kept_experts() {
        let _g = moe_test_guard();
        set_expert_prune_measurement(None);
        let dir = TempDir::new("p120_router");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let lw = p120_router_layer(&f, &p, false);
        let x = p120_flat_x(p.n_embd);

        // 期待値を手で計算 (moe_ffn_quant_impl と同じ手順: gemv → softmax → top-k)。
        let mut probs = vec![0.0f32; p.n_expert];
        lw.gate_inp
            .gemv_label("gate_inp", &x, &mut probs)
            .expect("gate_inp");
        softmax(&mut probs);
        let sel_full = select_topk(&probs, p.n_expert_used);
        let mut sum_full = 0.0f32;
        for &i in &sel_full {
            sum_full += probs[i];
        }
        let sel1 = select_topk(&probs, 1);
        assert_eq!(sel1, vec![0], "独占ルータの top-1 は expert 0");

        // ゲート off: 活性 = top-k 全員の確率和。
        let mut roles_off = RoleRms::default();
        moe_ffn_quant_full(
            &p, &lw, &x, None, false, Route::Cpu, Some(&mut roles_off), false,
        IvView::none())
        .expect("off");
        assert_eq!(roles_off.router.to_bits(), sum_full.to_bits(), "off: 活性 = top-k 確率和");

        // TOPK=1: 活性 = 残った expert 0 の確率だけ。
        set_expert_prune_measurement(Some(ExpertPruneMeasurement {
            topk: Some(1),
            min_w: None,
            w_stats: false,
        }));
        let mut roles_on = RoleRms::default();
        moe_ffn_quant_full(
            &p, &lw, &x, None, false, Route::Cpu, Some(&mut roles_on), false,
        IvView::none())
        .expect("TOPK=1");
        set_expert_prune_measurement(None);
        assert_eq!(
            roles_on.router.to_bits(),
            probs[0].to_bits(),
            "TOPK=1: 活性 = 残った expert の確率和"
        );
        assert!(
            roles_on.router > roles_off.router * 0.99,
            "独占ルータなら top-1 の和は top-2 の和とほぼ同じはず"
        );
    }

    /// W_STATS の蓄積と 1 行形式。off では何も溜まらず、on では層呼び出しごとに
    /// rank 別の重みと実行 expert 数が積まれる。
    #[test]
    fn expert_w_stats_accum_line_and_off_cost() {
        let _g = moe_test_guard();

        // off (上書き無し・env 未設定): 記録しても溜まらない。
        set_expert_prune_measurement(None);
        assert!(take_expert_w_stats().is_none(), "off なのに統計がある");

        set_expert_prune_measurement(Some(ExpertPruneMeasurement {
            topk: None,
            min_w: None,
            w_stats: false,
        }));
        expert_w_stats_record(&[0.5, 0.5], 2);
        assert!(
            take_expert_w_stats().is_none(),
            "w_stats=false なのに記録されている"
        );

        // on: 2 呼び出しぶん積む。rank は呼び出しが持つ expert 数まで。
        // 値は 2 の冪 (f32 で正確) にして f64 の期待値を確定させる。
        set_expert_prune_measurement(Some(ExpertPruneMeasurement {
            topk: None,
            min_w: None,
            w_stats: true,
        }));
        expert_w_stats_record(&[0.5, 0.25, 0.125], 3);
        expert_w_stats_record(&[0.625, 0.375], 2);
        let s = take_expert_w_stats().expect("on なのに統計が無い");
        assert_eq!(s.calls(), 2);
        assert_eq!(s.mean_experts_per_call(), 2.5);
        assert_eq!(s.mean_rank_weight(1), Some(0.5625), "(0.5 + 0.625) / 2");
        assert_eq!(s.mean_rank_weight(2), Some(0.3125), "(0.25 + 0.375) / 2");
        assert_eq!(s.mean_rank_weight(3), Some(0.125), "rank 3 は 1 呼び出しだけ");
        assert_eq!(s.mean_rank_weight(4), None, "rank 4 を持つ呼び出しは無い");
        assert_eq!(s.mean_rank_weight(0), None, "rank は 1 始まり");
        assert_eq!(s.mean_rank_weight(11), None, "rank は 10 まで");
        let line = s.line();
        assert!(line.starts_with("calls=2 "), "{line}");
        assert!(line.contains("mean_experts=2.50"), "{line}");
        assert!(line.contains("w1=0.5625"), "{line}");
        assert!(line.contains("w3=0.1250"), "{line}");
        assert!(line.contains("w4=-"), "{line}");
        assert!(line.contains("w10=-"), "{line}");
        assert!(
            take_expert_w_stats().is_none(),
            "取り出したら蓄積が空に戻る"
        );

        // 経路発火: moe 1 回で calls=1 / experts=top-k。ゲートで切ればその本数。
        let dir = TempDir::new("p120_wstats");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let lw = cache_test_layer(&f, &p);
        let x = cache_test_x(p.n_embd);
        let _ = moe_ffn_quant_with(&p, &lw, &x, None, false, Route::Cpu).expect("moe");
        let s = take_expert_w_stats().expect("moe 1 回ぶんの統計");
        assert_eq!(s.calls(), 1);
        assert_eq!(s.mean_experts_per_call(), 2.0, "ゲート off の実行 expert 数");
        set_expert_prune_measurement(Some(ExpertPruneMeasurement {
            topk: Some(1),
            min_w: None,
            w_stats: true,
        }));
        let _ = moe_ffn_quant_with(&p, &lw, &x, None, false, Route::Cpu).expect("moe TOPK=1");
        let s = take_expert_w_stats().expect("TOPK=1 の統計");
        set_expert_prune_measurement(None);
        assert_eq!(s.mean_experts_per_call(), 1.0, "TOPK=1 で 1 本に減る");
        assert_eq!(
            s.mean_rank_weight(2),
            None,
            "1 本に切られた呼び出しに rank 2 は無い"
        );
    }

    /// P58: ExpertResidency は「前回の送出から変わった層だけ」送る。
    ///
    /// - 集合が同じ層は出さない (変化が無い層は沈黙する)
    /// - 集合が変わった層は新しい集合で出す
    /// - 前回は載っていたが今回 0 本になった層は**空**で出す
    ///   (出さないとビューアが古い在庫を描き続ける)
    #[test]
    fn residency_delta_emits_only_changed_layers() {
        let prev = vec![(0u16, vec![1u16, 2]), (3, vec![7]), (5, vec![9])];
        assert!(
            residency_delta(&prev, &prev).is_empty(),
            "変化が無いのに層を送っている"
        );

        // 層 3 は集合が変わり、層 9 が新しく載り、層 5 は全部追い出された。
        let now = vec![(0, vec![1, 2]), (3, vec![7, 8]), (9, vec![4])];
        assert_eq!(
            residency_delta(&prev, &now),
            vec![(3, vec![7, 8]), (9, vec![4]), (5, Vec::new())],
            "差分の層・順序・空フレームが仕様と違う"
        );

        // 全滅 → 前回載っていた層をすべて空で送る (在庫を 0 本に更新する)。
        assert_eq!(
            residency_delta(&prev, &[]),
            vec![(0, Vec::new()), (3, Vec::new()), (5, Vec::new())]
        );
    }

    /// P60: 初回送出は全層 (在庫の無い層は空)。差分では初期状態を描けないため。
    #[test]
    fn initial_residency_covers_every_layer() {
        // 層 3 にだけ在庫がある状態。
        let now = vec![(3u16, vec![7u16, 9])];
        let full = full_residency(&now, 5);
        assert_eq!(
            full,
            vec![
                (0, Vec::new()),
                (1, Vec::new()),
                (2, Vec::new()),
                (3, vec![7, 9]),
                (4, Vec::new()),
            ],
            "初回の全層送出が層数ぶん並んでいない"
        );
        // 在庫が 1 本も無くても全層を空で返す (ビューワが「空」を描ける)。
        let full = full_residency(&[], 3);
        assert_eq!(full, vec![(0, Vec::new()), (1, Vec::new()), (2, Vec::new())]);

        // 初回を済ませた後は、在庫が同じなら差分は空 (毎トークン全層を送らない)。
        assert!(residency_delta(&full, &[]).is_empty(), "空の層を送り直している");
        // 初回の写しに在庫が入った層は、同じ在庫なら沈黙する。
        assert!(residency_delta(&full_residency(&now, 5), &now).is_empty());
    }

    /// P56: ルーティングの期待値 (テレメトリに依らない参照)。同じ式を手で回す。
    fn expected_router_topk(p: &Qwen4ExpParams, lw: &LayerW, x: &[f32]) -> (Vec<usize>, Vec<f32>) {
        let mut probs = vec![0.0f32; p.n_expert];
        lw.gate_inp.gemv(x, &mut probs).expect("gate_inp");
        softmax(&mut probs);
        let sel = select_topk(&probs, p.n_expert_used);
        let mut sum = 0.0f32;
        for &i in &sel {
            sum += probs[i];
        }
        let w: Vec<f32> = sel.iter().map(|&i| probs[i] / sum).collect();
        (sel, w)
    }

    /// P56: RouterTopK の送出。
    ///
    /// - `DENNO_TELEMETRY_ARCH` が off なら 1 枚も出ない (ゲートが効いている)
    /// - on なら decode の 1 トークン (moe_ffn_quant 1 回) で 1 枚、中身は
    ///   select_topk の結果と一致する (重みは f16)
    /// - prefill のバッチ経路は**チャンク末尾の 1 トークンだけ**送る
    #[test]
    fn router_topk_frames_are_arch_gated_and_last_token_only() {
        let _g = moe_test_guard();
        let dir = TempDir::new("router_topk");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let lw = cache_test_layer(&f, &p);
        let x = cache_test_x(p.n_embd);

        fn drain() -> Vec<denno_telemetry::Frame> {
            let mut v = Vec::new();
            while let Some(fr) = denno_telemetry::try_recv_nowait() {
                v.push(fr);
            }
            v
        }
        fn is_router(fr: &denno_telemetry::Frame) -> bool {
            matches!(fr, denno_telemetry::Frame::RouterTopK { .. })
        }

        let (sel, w) = expected_router_topk(&p, &lw, &x);
        let want = denno_telemetry::router_topk_frame(lw.il as u16, &sel, &w);
        // 期待値そのものの健全性: top-k 本で、重みは合計 1・降順。
        assert_eq!(sel.len(), p.n_expert_used);
        let total: f32 = w.iter().sum();
        assert!((total - 1.0).abs() < 1e-6, "重みの合計が 1 でない: {total}");

        let prev_on = denno_telemetry::enabled();
        let prev_arch = denno_telemetry::arch_enabled();
        denno_telemetry::set_enabled_for_tests(true);
        denno_telemetry::set_arch_for_tests(false);

        // (1) ARCH off → RouterTopK は 0 枚
        drain();
        moe_ffn_quant(&p, &lw, &x, None).expect("arch off");
        let got = drain();
        assert!(
            !got.iter().any(is_router),
            "DENNO_TELEMETRY_ARCH=0 で RouterTopK が出た: {got:?}"
        );

        // (2) ARCH on → 1 呼び出し (1 トークン) で 1 枚、中身は select_topk と一致
        denno_telemetry::set_arch_for_tests(true);
        drain();
        moe_ffn_quant(&p, &lw, &x, None).expect("arch on");
        let got = drain();
        assert_eq!(
            got.iter().filter(|fr| **fr == want).count(),
            1,
            "decode 1 トークンで期待した RouterTopK が 1 枚でない: {got:?}"
        );

        // (3) prefill (バッチ, nt=3) → チャンク末尾のトークンだけ 1 枚
        let xs: Vec<f32> = [1.0f32, 0.5, -1.0]
            .iter()
            .flat_map(|&a| cache_test_x(p.n_embd).iter().map(move |v| v * a).collect::<Vec<_>>())
            .collect();
        let (sel0, w0) = expected_router_topk(&p, &lw, &xs[..p.n_embd]);
        let first = denno_telemetry::router_topk_frame(lw.il as u16, &sel0, &w0);
        let (sel2, w2) = expected_router_topk(&p, &lw, &xs[2 * p.n_embd..]);
        let last = denno_telemetry::router_topk_frame(lw.il as u16, &sel2, &w2);
        drain();
        let out = moe_ffn_quant_batch(&p, &lw, &xs, 3, None).expect("batch");
        assert_eq!(out.len(), 3 * p.n_embd);
        let got = drain();
        assert_eq!(
            got.iter().filter(|fr| **fr == last).count(),
            1,
            "prefill でチャンク末尾の RouterTopK が 1 枚でない: {got:?}"
        );
        assert_eq!(
            got.iter().filter(|fr| **fr == first).count(),
            0,
            "prefill で非末尾トークンの RouterTopK が送られた: {got:?}"
        );

        // env 由来の状態に戻す (他のテストに漏らさない)
        denno_telemetry::set_arch_for_tests(prev_arch);
        denno_telemetry::set_enabled_for_tests(prev_on);
    }

    /// task 220: ExpertSkip の送出 (既定 off / 介入のある層だけ / バッチは末尾)。
    ///
    /// ルータは expert 0 と 1 が同点 (上位 2 本)。expert 0 を無効化すると実際に
    /// 使うのは [1, 2] になり、「無効化が無ければ選ばれていた 0」が skip になる。
    #[test]
    fn expert_skip_frames_are_gated_and_list_masked_topk() {
        let _g = moe_test_guard();
        set_expert_prune_measurement(None);
        let dir = TempDir::new("expert_skip");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let lw = p120_router_layer(&f, &p, true);
        let x = p120_flat_x(p.n_embd);
        let shape = intervene::Shape {
            n_layer: p.n_layer,
            n_expert: p.n_expert,
            n_expert_used: p.n_expert_used,
            n_hc: p.n_hc,
        };
        let cfg = intervene::config_for_test(&shape, vec![(0, 0)], vec![], vec![]).expect("cfg");
        let iv = IvView::of(&cfg);

        fn drain() -> Vec<denno_telemetry::Frame> {
            let mut v = Vec::new();
            while let Some(fr) = denno_telemetry::try_recv_nowait() {
                v.push(fr);
            }
            v
        }
        fn skips(v: &[denno_telemetry::Frame]) -> Vec<(u16, Vec<u16>)> {
            v.iter()
                .filter_map(|fr| match fr {
                    denno_telemetry::Frame::ExpertSkip { layer, experts } => {
                        Some((*layer, experts.clone()))
                    }
                    _ => None,
                })
                .collect()
        }

        let prev_on = denno_telemetry::enabled();
        let prev_arch = denno_telemetry::arch_enabled();
        denno_telemetry::set_enabled_for_tests(true);
        denno_telemetry::set_arch_for_tests(true);
        let run = |iv: IvView<'_>| {
            moe_ffn_quant_full_opts(&p, &lw, &x, None, false, Route::Cpu, None, true, false, iv)
                .expect("moe")
        };

        // (1) 既定 off → 介入があっても 1 枚も出ない (分岐で終わる)。
        set_expert_skip_for_tests(Some(false));
        drain();
        let _ = run(iv);
        assert!(
            skips(&drain()).is_empty(),
            "DENNO_TELEM_EXPERT_SKIP off で ExpertSkip が出た"
        );

        // (2) ゲート on でも介入が無ければ出ない (描くものが無い)。
        set_expert_skip_for_tests(Some(true));
        drain();
        let _ = run(IvView::none());
        assert!(
            skips(&drain()).is_empty(),
            "介入なしで ExpertSkip が出た (mask が無い層で送っている)"
        );

        // (3) ゲート on + 介入 → 層 0 に「無効化が無ければ top-k だった expert 0」。
        drain();
        let _ = run(iv);
        assert_eq!(
            skips(&drain()),
            vec![(0u16, vec![0u16])],
            "skip の内容が「マスクされた top-k」でない"
        );

        // (4) バッチ経路: 末尾の 1 トークンだけ 1 枚。
        //     どのトークンも同じ入力にして、末尾でも expert 0 が top-k に入るようにする。
        let xs: Vec<f32> = [1.0f32, 1.0, 1.0]
            .iter()
            .flat_map(|&a| p120_flat_x(p.n_embd).iter().map(move |v| v * a).collect::<Vec<_>>())
            .collect();
        drain();
        let out = moe_ffn_quant_batch_tapped(&p, &lw, &xs, 3, None, None, true, iv)
            .expect("batch");
        assert_eq!(out.len(), 3 * p.n_embd);
        assert_eq!(
            skips(&drain()),
            vec![(0u16, vec![0u16])],
            "バッチで skip が 1 枚 (末尾) でない"
        );

        // 状態を戻す (他のテストに漏らさない)。
        set_expert_skip_for_tests(None);
        denno_telemetry::set_arch_for_tests(prev_arch);
        denno_telemetry::set_enabled_for_tests(prev_on);
    }

    /// task 187: `DENNO_DEC_EXP_SUMDEV` / `DENNO_DEC_SHARED_DEV` の真理値表。
    /// どちらも lab t187c の採否で**既定 on** (明示的な偽値だけ切る)。
    #[test]
    fn dec_exp_sumdev_gate_truth_table() {
        for v in [None, Some("1"), Some("on"), Some("true"), Some(" ON "), Some("2")] {
            assert!(dec_exp_sumdev_from_value(v), "{v:?} が on にならない");
            assert!(dec_shared_dev_from_value(v), "{v:?} が on にならない");
        }
        for v in [Some("0"), Some("off"), Some("false"), Some("no"), Some(" OFF ")] {
            assert!(!dec_exp_sumdev_from_value(v), "{v:?} が on になる");
            assert!(!dec_shared_dev_from_value(v), "{v:?} が on になる");
        }
    }

    /// task 281: `DENNO_EXP_SCRATCH` の真理値表。
    ///
    /// **既定 off** (未設定・空・未知の値は off)。明示的な真値だけ on。
    /// on のときの数値は従来と 1 bit も変わらない (h のゼロ埋めをやめるだけ)。
    #[test]
    fn exp_scratch_gate_truth_table() {
        // task 285: 既定 on (未設定・空・"  " は on)。明示的な偽値だけ off。
        for v in [Some("0"), Some("off"), Some("OFF"), Some("false"), Some("no"), Some(" 0 ")] {
            assert!(!exp_scratch_from_value(v), "{v:?} が on になる");
        }
        for v in [None, Some(""), Some("  "), Some("1"), Some("on"), Some("ON"), Some(" true "), Some("yes")] {
            assert!(exp_scratch_from_value(v), "{v:?} が on にならない");
        }
    }

    /// task 285: `DENNO_DEC_ROUTER_TOPK_FAST` の真理値表 (既定 on・明示の偽値だけ off)。
    #[test]
    fn dec_router_topk_fast_truth_table() {
        for v in [Some("0"), Some("off"), Some("OFF"), Some("false"), Some("no"), Some(" 0 ")] {
            assert!(
                !dec_router_topk_fast_from_value(v),
                "{v:?} が on になる"
            );
        }
        for v in [None, Some(""), Some("1"), Some("on"), Some("true"), Some("yes")] {
            assert!(
                dec_router_topk_fast_from_value(v),
                "{v:?} が on にならない"
            );
        }
    }

    /// task 187: デバイス和のミラー (`dno_dec_exp_sum` と同じ順序) が、**ミスの無い
    /// 層では従来のホスト足し込みとビット一致**すること。
    ///
    /// 実形 (e = 2,560 / top-k = 10) で 3 トークンぶん、重みと寄与と shared の係数を
    /// トークンごとに変えて回す (状態を持ち越す)。ここが崩れると md5 が動く。
    #[test]
    fn dec_exp_sum_mirror_matches_host_order_without_misses() {
        let e = 2560usize;
        let nseg = 10usize;
        let mut st = 0x1234_5678_9abc_def0u64;
        let mut rnd = move || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((st >> 33) as f32 / 2147483648.0) - 0.5
        };
        for tok in 0..3usize {
            let mut h = vec![0.0f32; nseg * e];
            for v in h.iter_mut() {
                *v = rnd();
            }
            let w: Vec<f32> = (0..nseg)
                .map(|t| (t as f32 + 1.0 + tok as f32) / 55.0)
                .collect();
            let sh: Vec<f32> = (0..e).map(|_| rnd()).collect();
            let sgw = 0.31f32 + tok as f32 * 0.17;
            // 従来: ホストの逐次足し込み (index 昇順 → shared)。
            let mut host = vec![0.0f32; e];
            for (t, &pe) in w.iter().enumerate() {
                for j in 0..e {
                    host[j] += pe * h[t * e + j];
                }
            }
            for j in 0..e {
                host[j] += sh[j] * sgw;
            }
            // カーネルと同じ順序。
            let mut dev = vec![0.0f32; e];
            dec_exp_sum_mirror(&h, &w, e, Some((&sh, sgw)), &mut dev);
            assert_eq!(host, dev, "tok {tok}: ミスの無い層でホストと一致しない");
        }
    }

    /// task 187: ミスのある層では「ヒットを先に全部足してからミス」になる
    /// (丸め順が変わる)。差は f32 の丸めだけ (相対 1e-6 以内) であることを固定する。
    #[test]
    fn dec_exp_sum_mirror_with_miss_stays_within_rounding() {
        let e = 64usize;
        let nseg = 4usize;
        let mut h = vec![0.0f32; nseg * e];
        for (i, v) in h.iter_mut().enumerate() {
            *v = ((i % 37) as f32 - 18.0) * 0.013;
        }
        let w = [0.4f32, 0.3, 0.2, 0.1];
        let sh = vec![0.5f32; e];
        let sgw = 0.7f32;
        // 従来: 3 本目 (index 2) がミスで、その寄与をホストで足す。
        let mut miss = vec![0.0f32; e];
        for (j, m) in miss.iter_mut().enumerate() {
            *m = 0.02 * (j as f32 + 1.0);
        }
        let mut host = vec![0.0f32; e];
        for (t, &pe) in w.iter().enumerate() {
            let src: &[f32] = if t == 2 { &miss } else { &h[t * e..(t + 1) * e] };
            for j in 0..e {
                host[j] += pe * src[j];
            }
        }
        for j in 0..e {
            host[j] += sh[j] * sgw;
        }
        // SUMDEV: ヒット (0,1,3) を先に和にしてからミスを足す。
        let hits = [0usize, 1, 3];
        let hh: Vec<f32> = hits
            .iter()
            .flat_map(|&t| h[t * e..(t + 1) * e].to_vec())
            .collect();
        let hw: Vec<f32> = hits.iter().map(|&t| w[t]).collect();
        let mut dev = vec![0.0f32; e];
        dec_exp_sum_mirror(&hh, &hw, e, None, &mut dev);
        for j in 0..e {
            dev[j] += w[2] * miss[j];
        }
        // shared の足し込みは呼び出し側の共通コード (従来と同じ 1 演算)。
        for j in 0..e {
            dev[j] += sh[j] * sgw;
        }
        for (j, (a, b)) in host.iter().zip(dev.iter()).enumerate() {
            let rel = (a - b).abs() / a.abs().max(1e-30);
            assert!(rel <= 1e-6, "j={j}: host={a} dev={b} rel={rel}");
        }
    }

    /// P43: GPU 経路 (`Route::Vram`) を要求しても、CUDA が無い / expert が VRAM に
    /// 載らない / カーネル (PTX) が無いときは per-expert の CPU 経路へ落ちる。
    /// 値は既定経路とビット一致し、呼び出し回数も 3 × top-k のまま (fail closed)。
    #[test]
    fn moe_ffn_quant_vram_route_falls_back_to_cpu() {
        let _g = moe_test_guard();
        let dir = TempDir::new("route_vram");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let lw = cache_test_layer(&f, &p);
        let x = cache_test_x(p.n_embd);

        let base = moe_ffn_quant_with(&p, &lw, &x, None, false, Route::Cpu).expect("Cpu 経路");
        let before = moe_expert::any_calls();
        let got = moe_ffn_quant_with(&p, &lw, &x, None, false, Route::Vram)
            .expect("Vram 経路 (落ちる)");
        let calls = moe_expert::any_calls() - before;

        assert_eq!(
            calls,
            3 * p.n_expert_used,
            "層一括/VRAM 経路が載らないのに CPU へ落ちていない (expert_gemv_any の回数)"
        );
        assert_bitwise_eq(&base, &got, "route=Vram → CPU 経路");
    }

    /// P45: ヒット率 0 / 0.3 / 0.9 で経路が期待どおりに選ばれ、**どの経路でも出力が
    /// CPU 経路と一致する**こと (argmax と max_abs)。
    ///
    /// GPU の無いビルドでは `Route::Vram` も CPU に落ちるので、ここで確かまるのは
    /// 「判定」と「CPU 経路との一致」。`--features cuda` かつ GPU のある環境では
    /// 0.9 のケースが実際に GPU 経路を通る (同じテストが argmax / max_abs を比べる)。
    #[test]
    fn expert_gate_picks_route_by_hit_rate_and_output_matches_cpu() {
        let _g = moe_test_guard();
        let dir = TempDir::new("gate_route");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let lw = cache_test_layer(&f, &p);
        let x = cache_test_x(p.n_embd);

        // 参照: 従来の CPU 経路 (mmap のスライスを直接 GEMV)。
        let want = moe_ffn_quant_with(&p, &lw, &x, None, false, Route::Cpu).expect("CPU 参照");
        let argmax = |v: &[f32]| {
            let mut best = 0usize;
            for (i, &x) in v.iter().enumerate() {
                if x > v[best] {
                    best = i;
                }
            }
            best
        };
        // ミスしたときに VRAM へ積む量 (合成 GGUF の実バイト数から)。
        let per_expert: u64 = expert_lens(&lw).iter().map(|&b| b as u64).sum();
        let h2d = per_expert * p.n_expert_used as u64;

        for (rate, want_gpu) in [(0.0f64, false), (0.3, false), (0.9, true)] {
            let mut gate = expert_vram::ExpertGate::new(expert_vram::GateConfig::default());
            let hits = (rate * 10.0).round() as u32;
            for _ in 0..expert_vram::GATE_WINDOW_TOKENS {
                gate.record(hits, 10);
            }
            let d = gate.decide(0, Some(h2d));
            assert_eq!(
                d.uses_gpu(),
                want_gpu,
                "ヒット率 {rate} の経路選択が違う ({d:?})"
            );
            let route = if d.uses_gpu() {
                Route::Vram
            } else {
                Route::Cpu
            };
            let got = moe_ffn_quant_with(&p, &lw, &x, None, false, route).expect("経路");
            assert_eq!(
                argmax(&got),
                argmax(&want),
                "ヒット率 {rate}: argmax が CPU 経路と違う"
            );
            let max_abs = got
                .iter()
                .zip(want.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert_eq!(
                max_abs, 0.0,
                "ヒット率 {rate}: max_abs {max_abs} で CPU 経路と一致しない"
            );
        }
    }

    /// P45: `DENNO_EXPERT_GPU_MIN_HIT=0` は「常に GPU」= P39 までと同じ判定。
    /// 既定 (0.5) では冷えた状態が CPU に落ちる。
    #[test]
    fn expert_gate_default_threshold_falls_back_when_cold() {
        let cold = |min_hit: f32| {
            let mut g = expert_vram::ExpertGate::new(expert_vram::GateConfig {
                min_hit,
                ..Default::default()
            });
            g.decide(0, Some(1024))
        };
        assert!(
            !cold(expert_vram::DEFAULT_EXPERT_GPU_MIN_HIT).uses_gpu(),
            "既定 0.5 は起動直後 CPU"
        );
        assert!(cold(0.0).uses_gpu(), "0 は常に GPU (P39 までと同じ)");
    }

    /// P45: ゲートが見積る「ミス時に積む量」が、合成 GGUF の実バイト数と一致する。
    ///
    /// ここが `None` になるとゲートは fail-safe で常に CPU になるので、
    /// 「見積れる」こと自体をテストで固定する。
    #[test]
    fn expert_h2d_bytes_one_matches_expert_lens() {
        let _g = moe_test_guard();
        let dir = TempDir::new("h2d_bytes");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let lw = cache_test_layer(&f, &p);
        let want: u64 = expert_lens(&lw).iter().map(|&b| b as u64).sum();
        assert_eq!(
            expert_h2d_bytes_one(&lw, p.n_embd, p.n_ff_exp),
            Some(want),
            "見積りが実バイト数と違う (ゲートが fail-safe で CPU に固定される)"
        );
    }

    /// P45: `Route::Vram` の層入口 ([`gate_decide`]) が、冷えたゲートで CPU・
    /// 温まったゲートで GPU を返し、H2D 量が読めない型では CPU に倒れること。
    ///
    /// env を使わずに、`moe_ffn_quant_with` が毎層呼ぶ関数をそのまま通す
    /// (グローバルのゲートは既定の CPU 経路では作られないので、ここでは実体を渡す)。
    #[test]
    fn gate_decide_routes_layers_without_env() {
        let gate = std::sync::Mutex::new(expert_vram::ExpertGate::new(
            expert_vram::GateConfig::default(),
        ));
        // 冷えている (サンプル 0): 1 層目から CPU。
        assert_eq!(
            gate_decide(Some(&gate), 0, 2, Some(1024)),
            expert_vram::GateDecision::Cpu,
            "冷えたゲートで GPU を選んだ"
        );
        // 層 1 以降はトークン先頭の判定をそのまま使う (トークン内で混ざらない)。
        assert_eq!(
            gate_decide(Some(&gate), 1, 2, Some(1024)),
            expert_vram::GateDecision::Cpu
        );
        // ミス時の H2D 量が読めない (None) は fail-safe で CPU。
        assert_eq!(
            gate_decide(Some(&gate), 0, 2, None),
            expert_vram::GateDecision::Cpu,
            "見積り不能で GPU に倒れた"
        );
        // 全ヒットのトークンで温めると GPU。
        {
            let mut g = gate.lock().expect("gate");
            for _ in 0..expert_vram::GATE_WINDOW_TOKENS {
                g.record(10, 10);
            }
        }
        assert_eq!(
            gate_decide(Some(&gate), 0, 2, Some(1024)),
            expert_vram::GateDecision::Gpu,
            "ヒット率 1.0 でも GPU に戻らない"
        );
        // ゲート無し = 既定の CPU 経路 / `DENNO_EXPERT_GPU_MIN_HIT=0`。
        assert_eq!(
            gate_decide(None, 0, 2, None),
            expert_vram::GateDecision::Gpu,
            "ゲート無しで CPU に倒れた (P39 までの経路と変わる)"
        );
    }

    /// task 830: TinyLFU の下では P45 のヒット率降格を使わない (迂回)。
    ///
    /// 同じ「ヒット率 0.3」でも、迂回ありは GPU・なしは CPU。迂回が実際に
    /// 効いたことは判定の計数 (`bypass_gpu` が増え `cpu` が 0) で確かめる。
    #[test]
    fn gate_decide_bypass_survives_tinylfu_low_hit_rate() {
        let mk = |bypass: bool| {
            let mut g = expert_vram::ExpertGate::new(expert_vram::GateConfig::default())
                .with_bypass(bypass);
            for _ in 0..expert_vram::GATE_WINDOW_TOKENS {
                g.record(3, 10); // 1 トークン = 全 10 本中 3 本ヒット
            }
            std::sync::Mutex::new(g)
        };
        let legacy = mk(false);
        let tinylfu = mk(true);
        assert_eq!(
            gate_decide(Some(&legacy), 0, 2, Some(1024)),
            expert_vram::GateDecision::Cpu,
            "迂回なし (従来) はヒット率 0.3 で CPU に倒れる"
        );
        assert_eq!(
            gate_decide(Some(&tinylfu), 0, 2, Some(1024)),
            expert_vram::GateDecision::Gpu,
            "tinylfu の迂回が効いていない (当たっている expert も CPU に落ちる)"
        );
        // 発火の証拠: 迂回の計数だけが増える
        let (gpu, cpu, probe, bypass) = tinylfu.lock().expect("gate").decision_counts();
        assert_eq!((gpu, cpu, probe, bypass), (0, 0, 0, 1));
        // 見積り不能はゲートと同じく fail-safe で CPU のまま
        assert_eq!(
            gate_decide(Some(&tinylfu), 0, 2, None),
            expert_vram::GateDecision::Cpu,
            "迂回で fail-safe まで外れた"
        );
    }

    #[test]
    fn moe_par_gate_value_parsing() {
        // DENNO_MOE_PAR の解釈 (既定 1)。env を触らずに純粋関数だけを見る。
        assert!(moe_par_from_value(None), "未設定は既定で有効");
        assert!(moe_par_from_value(Some("1")));
        assert!(moe_par_from_value(Some("8")));
        assert!(!moe_par_from_value(Some("0")), "\"0\" で無効");
        assert!(
            moe_par_from_value(Some("")),
            "空文字は「未設定」と同じ扱い (有効)"
        );
    }

    /// P-MOE-G5: expert ループの並列化は数値を 1 ビットも変えない。
    ///
    /// 同じ合成 expert で `par=0` (従来の逐次) と `par=1` (expert 並列) を回し、
    /// `to_bits()` が全要素一致することを見る。併せて「並列分岐が実際に通った」
    /// 「内側が行並列なしの入口を通った」をカウンタで確かめる (緑 = 発動の証拠ではない)。
    #[test]
    fn moe_expert_loop_par_is_bit_identical_to_seq() {
        let _g = moe_test_guard();
        // P74: このテストが見るのは**旧 (expert 並列) 経路**のカウンタ本数。
        // 既定 on の平坦化経路は `PAR_CONTRIB_CALLS` を積まず `FLAT_DECODE_REGIONS`
        // を積むので、ここでは明示的に off へ固定する (期待値は緩めない)。
        set_expert_flat_decode_for_test(Some(false));
        let dir = TempDir::new("par");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let lw = cache_test_layer(&f, &p);
        let x = cache_test_x(p.n_embd);
        assert!(
            p.n_expert_used > 1,
            "並列経路を通すには選択 expert が 2 個以上必要 (n_expert_used = {})",
            p.n_expert_used
        );

        // 1. 従来経路 (DENNO_MOE_PAR=0 相当)
        let seq = moe_ffn_quant_with_par(&p, &lw, &x, None, false).expect("逐次 (par=0)");
        assert!(seq.iter().all(|v| v.is_finite()), "逐次経路に NaN/inf");
        assert!(seq.iter().any(|v| *v != 0.0), "逐次経路が全 0");

        // 2. 並列経路 (DENNO_MOE_PAR=1 相当)
        let par_before = PAR_CONTRIB_CALLS.load(std::sync::atomic::Ordering::Relaxed);
        let seq_before = moe_expert::SEQ_CALLS.load(std::sync::atomic::Ordering::Relaxed);
        let par = moe_ffn_quant_with_par(&p, &lw, &x, None, true).expect("並列 (par=1)");
        let par_calls = PAR_CONTRIB_CALLS.load(std::sync::atomic::Ordering::Relaxed) - par_before;
        let seq_calls = moe_expert::SEQ_CALLS.load(std::sync::atomic::Ordering::Relaxed) - seq_before;

        // 3. 並列分岐が発動したこと (top-k の数だけ寄与計算が走る)
        //
        // カウンタはプロセス全体の AtomicUsize なので、`--test-threads` を 1 にしない
        // 実行では**同じバイナリで並行に走る他のテスト** (h2 / engine 側の moe 呼び出し) の
        // 分がこの窓に入る (実測: 2 のはずが 3)。厳密な本数は直列実行
        // (`--test-threads=1`、deploy8098.sh の既定) のときにだけ意味を持つ。
        // ここでは「並列分岐に実際に入った」ことが言えればよいので下限で見る。
        #[cfg(feature = "par")]
        {
            assert!(
                par_calls >= p.n_expert_used,
                "並列の寄与計算が選択 expert 数ぶん走っていない (par=1 で並列分岐に入っていない): par_calls={par_calls} used={}",
                p.n_expert_used
            );
            // 内側は gemv_seq = 1 expert あたり gate/up/down の 3 本
            assert!(
                seq_calls >= 3 * p.n_expert_used,
                "並列時は内側が行並列なしの入口 (gemv_seq) を通るはず: seq_calls={seq_calls} want>={}",
                3 * p.n_expert_used
            );
        }
        #[cfg(not(feature = "par"))]
        {
            assert_eq!(par_calls, 0, "par 機能なしで並列分岐に入った");
            assert_eq!(seq_calls, 0, "par 機能なしで逐次入口を通った");
        }

        // 4. ビット一致
        assert_bitwise_eq(&seq, &par, "par=0 vs par=1");
        set_expert_flat_decode_for_test(None);
    }

    /// task 151: prefill の GPU 経路は「ヒット (GPU) とミス (CPU) の行を
    /// **expert index 昇順で 1 本ずつ**足す」ので、ヒット率が変わっても出力は
    /// CPU だけの経路とビット一致する。
    ///
    /// GPU の無いここでは、ヒット側も同じ flat で計算して行を作る
    /// (`exp_pf_add_all` の足し込み順・行表が flat の scatter と同じことを見る)。
    #[test]
    fn exp_pf_add_all_is_bit_identical_for_any_split() {
        let _g = moe_test_guard();
        let dir = TempDir::new("pf_add");
        let (ne, k) = (8usize, 3usize);
        let (e, ff) = (512usize, 640usize);
        let path = write_expert_gguf_n(&dir, "pf_add.gguf", e, ff, ne, 1);
        let f = GgufFile::open(&path).expect("open");
        let p = params_dims(e, ff, ne, k);
        let lw = layer_at(&f, &p, 0);
        let nt = 6usize;
        let xs = cache_test_x_seed(nt * e, 11);
        // トークンごとの選択 (expert index 昇順 = 重みの添字 j の順)。
        let mut weights = vec![0.0f32; nt * k];
        let mut by_expert: Vec<Vec<(usize, usize)>> = vec![Vec::new(); ne];
        for t in 0..nt {
            let mut sel: Vec<usize> = (0..k).map(|j| (t * 2 + j) % ne).collect();
            sel.sort_unstable();
            for (j, &ei) in sel.iter().enumerate() {
                weights[t * k + j] = 0.3 - 0.01 * j as f32;
                by_expert[ei].push((t, j));
            }
        }
        // 参照: CPU だけの経路 (flat が h を out へ足す)。
        let mut out_ref = vec![0.0f32; nt * e];
        moe_experts_flat(
            &lw,
            &by_expert,
            &weights,
            k,
            &xs,
            e,
            ff,
            MOE_FLAT_ROWS,
            &mut out_ref,
            false,
            None,
        )
        .expect("参照 flat");
        assert!(
            out_ref.iter().any(|v| *v != 0.0),
            "参照が全 0 (経路の生死が分からない)"
        );
        // ヒットの内訳を変えて 3 通り (全部ミス / 半分 / 全部ヒット)。
        for cut in [0usize, 3, ne] {
            let pick: Vec<bool> = (0..ne).map(|ei| ei < cut).collect();
            let mut miss_be: Vec<Vec<(usize, usize)>> = Vec::with_capacity(ne);
            let mut hit_be: Vec<Vec<(usize, usize)>> = Vec::with_capacity(ne);
            for (ei, picks) in by_expert.iter().enumerate() {
                let hit = pick[ei];
                miss_be.push(if hit { Vec::new() } else { picks.clone() });
                hit_be.push(if hit { picks.clone() } else { Vec::new() });
            }
            let mut scratch = vec![0.0f32; nt * e];
            let mut miss_h: Vec<f32> = Vec::new();
            moe_experts_flat(
                &lw,
                &miss_be,
                &weights,
                k,
                &xs,
                e,
                ff,
                MOE_FLAT_ROWS,
                &mut scratch,
                false,
                Some(&mut miss_h),
            )
            .expect("ミス flat");
            assert!(
                scratch.iter().all(|v| *v == 0.0),
                "capture が out を触った (cut={cut})"
            );
            let mut hit_h: Vec<f32> = Vec::new();
            moe_experts_flat(
                &lw,
                &hit_be,
                &weights,
                k,
                &xs,
                e,
                ff,
                MOE_FLAT_ROWS,
                &mut scratch,
                false,
                Some(&mut hit_h),
            )
            .expect("ヒット flat");
            // 行表と足し込み順 (`moe_experts_iq_pf_batch` と同じ規則)。
            let plan = crate::model::exp_pf::build_plan(&by_expert, &pick);
            let mut hoff = vec![usize::MAX; ne];
            for (s, &ei) in plan.experts.iter().enumerate() {
                hoff[ei] = plan.seg[s * 4 + 1] as usize;
            }
            let mut moff = vec![usize::MAX; ne];
            let mut off = 0usize;
            for (ei, picks) in miss_be.iter().enumerate() {
                if picks.is_empty() {
                    continue;
                }
                moff[ei] = off;
                off += picks.len();
            }
            assert_eq!(off * e, miss_h.len(), "ミスの行数と h の長さが違う");
            assert_eq!(
                plan.total_rows * e,
                hit_h.len(),
                "ヒットの行数と h の長さが違う"
            );
            let mut order: Vec<Vec<(u32, u32, bool)>> = vec![Vec::new(); nt];
            for (ei, picks) in by_expert.iter().enumerate() {
                let (ho, mo) = (hoff[ei], moff[ei]);
                for (idx, &(t, j)) in picks.iter().enumerate() {
                    let wi = (t * k + j) as u32;
                    if ho != usize::MAX {
                        order[t].push((wi, (ho + idx) as u32, true));
                    } else {
                        assert_ne!(mo, usize::MAX, "ミスの行先が無い (expert {ei})");
                        order[t].push((wi, (mo + idx) as u32, false));
                    }
                }
            }
            let mut out: Vec<f32> = vec![0.0f32; nt * e];
            exp_pf_add_all(&mut out, &order, &hit_h, &miss_h, &weights, e);
            assert_bitwise_eq(&out_ref, &out, &format!("hit cut={cut}"));
        }
    }

    /// P64: `DENNO_EXPERT_LOOP` の値解釈。**既定は off**。
    #[test]
    fn expert_loop_gate_values() {
        assert!(!expert_loop_from_value(None), "未設定は off (既定 off)");
        assert!(!expert_loop_from_value(Some("")), "空文字は off");
        assert!(!expert_loop_from_value(Some("0")), "0 は off");
        assert!(!expert_loop_from_value(Some("2")), "1 以外の数は off");
        assert!(expert_loop_from_value(Some("1")), "1 は on");
        assert!(expert_loop_from_value(Some(" on ")), "on は on (前後の空白は無視)");
        assert!(expert_loop_from_value(Some("true")), "true は on");
    }

    /// P64: decode 1 層あたりの rayon 並列領域の回数。
    ///
    /// - 新経路 (`DENNO_EXPERT_LOOP=1` 相当): **1 回**。
    /// - 従来の既定 (par=1): 1 回 (expert 並列)。
    /// - 従来の par=0: 行列ごとに立つので 選択 expert 数 x 3 回。
    ///
    /// 行並列は `n_out >= PAR_MIN_ROWS` (512) でしか立たないので、この合成層は
    /// `n_embd = 512` (down の `n_out` = 512) にしてある。
    #[test]
    #[cfg(feature = "par")]
    fn expert_loop_one_region_per_layer() {
        let _g = moe_test_guard();
        // P74: 2. の「旧 par=1」は `PAR_CONTRIB_REGIONS` が 1 回立つことを見る。
        // 既定 on の平坦化経路はこの領域を立てない (`FLAT_DECODE_REGIONS` を立てる)
        // ので、旧経路の本数を数えるここでは明示的に off へ固定する。
        set_expert_flat_decode_for_test(Some(false));
        let dir = TempDir::new("loop_num");
        let (nl, ne, k) = (3usize, 8usize, 3usize);
        let path = write_expert_gguf_n(&dir, "3l.gguf", 512, 640, ne, nl);
        let f = GgufFile::open(&path).expect("open");
        let p = params_dims(512, 640, ne, k);
        let xs: Vec<Vec<f32>> = (0..nl).map(|t| cache_test_x_seed(p.n_embd, t)).collect();
        let load = |c: &std::sync::atomic::AtomicUsize| c.load(std::sync::atomic::Ordering::Relaxed);

        for il in 0..nl {
            let lw = layer_at(&f, &p, il);
            let x = &xs[il];

            // 1. 新経路: 層あたり 1 領域・内側の行並列なし。
            let (e0, c0, r0) = (load(&EXPERT_LOOP_REGIONS), load(&PAR_CONTRIB_REGIONS), load(&moe_expert::ROW_PAR_REGIONS));
            let _ = moe_ffn_quant_full_opts(&p, &lw, x, None, true, Route::Cpu, None, true, true, IvView::none())
                .expect("新経路");
            assert_eq!(load(&EXPERT_LOOP_REGIONS) - e0, 1, "層 {il}: 新経路の並列領域が 1 回でない");
            assert_eq!(load(&PAR_CONTRIB_REGIONS) - c0, 0, "層 {il}: 新経路で旧の並列領域が立った");
            assert_eq!(load(&moe_expert::ROW_PAR_REGIONS) - r0, 0, "層 {il}: 新経路の内側で行並列が立った");

            // 2. 旧経路 (既定 par=1): expert 並列の 1 領域・内側は行並列なし。
            let (e0, c0, r0) = (load(&EXPERT_LOOP_REGIONS), load(&PAR_CONTRIB_REGIONS), load(&moe_expert::ROW_PAR_REGIONS));
            let _ = moe_ffn_quant_full_opts(&p, &lw, x, None, true, Route::Cpu, None, true, false, IvView::none())
                .expect("旧 par=1");
            assert_eq!(load(&PAR_CONTRIB_REGIONS) - c0, 1, "層 {il}: 旧 par=1 の並列領域が 1 回でない");
            assert_eq!(load(&moe_expert::ROW_PAR_REGIONS) - r0, 0, "層 {il}: 旧 par=1 の内側で行並列が立った");
            assert_eq!(load(&EXPERT_LOOP_REGIONS) - e0, 0, "層 {il}: loop_on=false なのに新経路が走った");

            // 3. 旧経路 (par=0): 行列ごとに立つ (選択 expert 数 x 3)。
            let r0 = load(&moe_expert::ROW_PAR_REGIONS);
            let _ = moe_ffn_quant_full_opts(&p, &lw, x, None, false, Route::Cpu, None, true, false, IvView::none())
                .expect("旧 par=0");
            assert_eq!(
                load(&moe_expert::ROW_PAR_REGIONS) - r0,
                k * 3,
                "層 {il}: 旧 par=0 の行並列領域が k*3 でない"
            );
        }
        set_expert_flat_decode_for_test(None);
    }

    /// P64: 新経路 (1 層 1 並列領域) が旧経路とビット一致する。
    ///
    /// 合成重みで 3 層ぶん回し、旧 par=1 / 旧 par=0 / 新 / 新+常駐キャッシュ の 4 本を
    /// `to_bits()` で比べる。ビット一致の根拠は「1 要素の計算順を変えていない」こと:
    /// 1 expert は同じ `gemv_seq` (同じカーネル・同じ行順)、足し込みは index 昇順の
    /// 逐次で、並列化は expert 単位だけ。
    #[test]
    fn expert_loop_is_bit_identical_to_old_paths() {
        let _g = moe_test_guard();
        let dir = TempDir::new("loop_bits");
        let (nl, ne, k) = (3usize, 8usize, 3usize);
        let path = write_expert_gguf_n(&dir, "3l.gguf", 512, 640, ne, nl);
        let f = GgufFile::open(&path).expect("open");
        let p = params_dims(512, 640, ne, k);
        let lw0 = layer_at(&f, &p, 0);
        let lens = expert_lens(&lw0);
        let set = ExpertCacheSet::new_with_slots([4, 4, 4], lens, &[path.clone()], 4).expect("cache set");

        let run = |lw: &LayerW, x: &[f32], cache: Option<&ExpertCacheSet>, par: bool, lp: bool| {
            moe_ffn_quant_full_opts(&p, lw, x, cache, par, Route::Cpu, None, true, lp, IvView::none())
                .expect("moe_ffn_quant_full_opts")
        };

        for il in 0..nl {
            let lw = layer_at(&f, &p, il);
            let x = cache_test_x_seed(p.n_embd, il);
            let old_par = run(&lw, &x, None, true, false);
            let old_seq = run(&lw, &x, None, false, false);
            assert!(old_par.iter().all(|v| v.is_finite()), "層 {il}: 旧経路に NaN/inf");
            assert!(old_par.iter().any(|v| *v != 0.0), "層 {il}: 旧経路が全 0");
            assert_bitwise_eq(&old_par, &old_seq, &format!("層 {il}: 旧 par=1 vs 旧 par=0"));

            let new = run(&lw, &x, None, true, true);
            assert_bitwise_eq(&old_par, &new, &format!("層 {il}: 旧 vs 新 (mmap)"));
            let new_c = run(&lw, &x, Some(&set), true, true);
            assert_bitwise_eq(&old_par, &new_c, &format!("層 {il}: 旧 vs 新 (常駐キャッシュ)"));
        }
    }

    /// P70/P74: `DENNO_EXPERT_FLAT_DECODE` の値解釈 (既定 on・明示的な偽値だけ off)。
    #[test]
    fn expert_flat_decode_gate_values() {
        assert!(expert_flat_decode_from_value(None), "未設定は既定で on");
        assert!(expert_flat_decode_from_value(Some("")), "空文字も on");
        assert!(expert_flat_decode_from_value(Some("1")));
        assert!(expert_flat_decode_from_value(Some("on")));
        assert!(expert_flat_decode_from_value(Some("ON")));
        assert!(expert_flat_decode_from_value(Some("true")));
        assert!(expert_flat_decode_from_value(Some("yes")), "未知の値は on のまま");
        assert!(!expert_flat_decode_from_value(Some("0")), "0 は off");
        assert!(!expert_flat_decode_from_value(Some("off")), "off は off");
        assert!(!expert_flat_decode_from_value(Some("OFF")), "大文字 OFF も off");
        assert!(!expert_flat_decode_from_value(Some("false")), "false は off");
        assert!(!expert_flat_decode_from_value(Some("no")), "no は off");
        assert!(!expert_flat_decode_from_value(Some(" off ")), "前後の空白は無視");
        assert!(expert_flat_decode_from_value(Some(" 1 ")));
    }

    /// P74: テスト用上書きが、env 由来の `OnceLock` より優先すること。
    ///
    /// `OnceLock` はプロセスで 1 回しか決まらないので、既定を on にした後で
    /// 「旧経路の本数を見るテスト」を書くには上書き口が要る。ここでは
    /// OnceLock を既定 (on) で初期化させたうえで、off の上書きが勝つことを見る。
    #[test]
    fn expert_flat_decode_override_beats_env() {
        let _g = moe_test_guard();
        set_expert_flat_decode_for_test(None);
        assert!(expert_flat_decode_enabled(), "既定 (env 未設定) は on");
        set_expert_flat_decode_for_test(Some(false));
        assert!(
            !expert_flat_decode_enabled(),
            "off の上書きが OnceLock に負けている"
        );
        set_expert_flat_decode_for_test(Some(true));
        assert!(expert_flat_decode_enabled(), "on の上書きが効いていない");
        set_expert_flat_decode_for_test(None);
        assert!(expert_flat_decode_enabled(), "上書きを外したら既定 (on) に戻る");
    }

    /// task 153: ミス expert の寄与を行まで平坦化しても、expert 並列と 1 ビットも
    /// 変わらない (ミスは 1 層 0〜3 本なので、ここが実際に通る形で固定する)。
    #[test]
    #[cfg(all(feature = "par", feature = "cuda"))]
    fn miss_flat_matches_expert_parallel() {
        gpu_gemm_off();
        let _g = moe_test_guard();
        let dir = TempDir::new("p153miss");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let lw = cache_test_layer(&f, &p);
        let x = cache_test_x(p.n_embd);
        // 1 本 (ヒット率が高いときの典型) と 2 本 (まとめてミスしたとき)。
        for sel in [vec![2usize], vec![0usize, 3]] {
            let mut a = vec![0.0f32; sel.len() * p.n_embd];
            let mut b = vec![0.0f32; sel.len() * p.n_embd];
            set_expert_miss_flat_for_test(Some(false));
            expert_miss_contribs_into(&lw, &sel, &x, p.n_embd, p.n_ff_exp, &mut a)
                .expect("expert 並列");
            set_expert_miss_flat_for_test(Some(true));
            expert_miss_contribs_into(&lw, &sel, &x, p.n_embd, p.n_ff_exp, &mut b)
                .expect("行平坦化");
            let ab: Vec<u32> = a.iter().map(|v| v.to_bits()).collect();
            let bb: Vec<u32> = b.iter().map(|v| v.to_bits()).collect();
            assert_eq!(ab, bb, "sel={sel:?} で値が違う");
            assert!(ab.iter().any(|&v| v != 0), "sel={sel:?}: 0 のまま (計算していない)");
        }
        set_expert_miss_flat_for_test(None);
        assert!(expert_miss_flat_enabled(), "既定 (env 未設定) は on");
    }

    /// task 153: `DENNO_EXP_MISS_FLAT` の値解釈 (既定 on、明示的な偽値だけ off)。
    #[test]
    fn expert_miss_flat_env_values() {
        for v in [None, Some("1"), Some("on"), Some("true"), Some(""), Some("yes")] {
            assert!(expert_miss_flat_from_value(v), "{v:?} は on のはず");
        }
        for v in [Some("0"), Some("off"), Some("OFF"), Some("false"), Some("no"), Some(" 0 ")] {
            assert!(!expert_miss_flat_from_value(v), "{v:?} は off のはず");
        }
    }

    /// P70: expert 横断の平坦化経路は、従来の expert 並列と**ビット一致**する
    /// (層 1 回ぶんを丸ごと比較 = ルータ・shared expert も同じ経路)。
    #[test]
    #[cfg(feature = "par")]
    fn decode_flat_matches_expert_parallel() {
        gpu_gemm_off();
        let _g = moe_test_guard();
        let dir = TempDir::new("p70flat");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        assert!(
            p.n_expert_used > 1,
            "平坦化の意味が出るのは 2 expert 以上を選ぶ形"
        );
        let lw = cache_test_layer(&f, &p);
        let x = cache_test_x(p.n_embd);

        let before = FLAT_DECODE_REGIONS.load(std::sync::atomic::Ordering::Relaxed);
        let old = moe_ffn_quant_impl(&p, &lw, &x, None, true, Route::Cpu, None, true, false, false, IvView::none())
            .expect("従来 (expert 並列)");
        let new = moe_ffn_quant_impl(&p, &lw, &x, None, true, Route::Cpu, None, true, false, true, IvView::none())
            .expect("新 (expert 横断平坦化)");
        assert!(
            old.iter().any(|v| *v != 0.0),
            "MoE の出力が全 0 (経路の生死が分からない)"
        );
        assert_bitwise_eq(&old, &new, "平坦化 vs expert 並列");
        assert!(
            FLAT_DECODE_REGIONS.load(std::sync::atomic::Ordering::Relaxed) > before,
            "平坦化の分岐が通っていない (カウンタが増えていない)"
        );
    }

    /// task 285: decode の router top-k は挿入法 (`select_topk_into_fast`) と
    /// `select_nth_unstable_by` 版で**選ばれる集合も MoE の出力もビット一致**する
    /// こと。あわせて挿入法の分岐が実際に通ったことをカウンタで確かめる。
    #[test]
    #[cfg(feature = "par")]
    fn dec_router_topk_fast_matches_select_nth() {
        gpu_gemm_off();
        let _g = moe_test_guard();
        let dir = TempDir::new("p285topk");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let lw = cache_test_layer(&f, &p);
        let x = cache_test_x(p.n_embd);

        let before = DEC_ROUTER_TOPK_FAST_CALLS.load(std::sync::atomic::Ordering::Relaxed);
        set_dec_router_topk_fast_for_test(Some(false));
        let slow = moe_ffn_quant_impl(
            &p,
            &lw,
            &x,
            None,
            true,
            Route::Cpu,
            None,
            true,
            false,
            false,
            IvView::none(),
        )
        .expect("select_nth 版");
        let mid = DEC_ROUTER_TOPK_FAST_CALLS.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(mid, before, "off なのに挿入法の分岐が通った");

        set_dec_router_topk_fast_for_test(Some(true));
        let fast = moe_ffn_quant_impl(
            &p,
            &lw,
            &x,
            None,
            true,
            Route::Cpu,
            None,
            true,
            false,
            false,
            IvView::none(),
        )
        .expect("挿入法");
        assert!(
            DEC_ROUTER_TOPK_FAST_CALLS.load(std::sync::atomic::Ordering::Relaxed) > mid,
            "挿入法の分岐が通っていない (カウンタが増えていない)"
        );
        assert!(
            slow.iter().any(|v| *v != 0.0),
            "MoE の出力が全 0 (経路の生死が分からない)"
        );
        assert_bitwise_eq(&slow, &fast, "router top-k: 挿入法 vs select_nth");
        set_dec_router_topk_fast_for_test(None);
        assert!(
            dec_router_topk_fast_on(),
            "既定 (env 未設定・上書き解除) は on"
        );
    }

    /// P70: 行チャンク幅を変えても expert 寄与はビット一致する
    /// (割り切れない幅 7 も含める = 末尾の端数チャンクを通す)。
    #[test]
    #[cfg(feature = "par")]
    fn decode_flat_row_chunk_is_independent() {
        gpu_gemm_off();
        let _g = moe_test_guard();
        let dir = TempDir::new("p70chunk");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let lw = cache_test_layer(&f, &p);
        let x = cache_test_x(p.n_embd);
        let sel: Vec<usize> = (0..p.n_expert_used).collect();
        let (e, ff) = (p.n_embd, p.n_ff_exp);

        let mut want = vec![0.0f32; sel.len() * e];
        expert_contribs_par_into(&lw, &sel, &x, e, ff, None, &mut want).expect("従来経路");
        assert!(want.iter().any(|v| *v != 0.0), "寄与が全 0");

        for rows in [7usize, 32, 64, 96, 1000] {
            let mut got = vec![0.0f32; want.len()];
            let t0 = FLAT_DECODE_TASKS.load(std::sync::atomic::Ordering::Relaxed);
            expert_contribs_flat_decode_into(&lw, &sel, &x, e, ff, &mut got, rows)
                .expect("平坦化");
            // タスク本数 = expert × (gate + up のチャンク + down のチャンク)。
            let want_tasks = sel.len()
                * (2 * ff.div_ceil(rows) + e.div_ceil(rows));
            assert_eq!(
                FLAT_DECODE_TASKS.load(std::sync::atomic::Ordering::Relaxed) - t0,
                want_tasks,
                "rows={rows}: 行タスクの本数が違う"
            );
            assert_bitwise_eq(&want, &got, &format!("rows={rows}"));
        }
        // 0 行は落ちる (空タスクを回さない)。
        let mut got = vec![0.0f32; want.len()];
        assert!(expert_contribs_flat_decode_into(&lw, &sel, &x, e, ff, &mut got, 0).is_err());
    }

    /// task 361: `DENNO_EXP_MISS_ROWS` の解釈 (純関数) と既定値。
    ///
    /// - 未設定は 16 (ミス 1 本の層で gate/up が 80 タスク = 14 スレッドで
    ///   端数が出ない幅。lab t361b でミス CPU 時間の中央値 -2.4〜-5.8%)。
    /// - 数字だけを受け付ける。0 と壊れた値は既定へ倒す (0 行はタスクを作れない)。
    #[test]
    fn exp_miss_rows_value() {
        assert_eq!(miss_flat_rows_from_value(None), MISS_FLAT_ROWS);
        assert_eq!(MISS_FLAT_ROWS, 16);
        assert_eq!(miss_flat_rows_from_value(Some(" 32 ")), 32);
        assert_eq!(miss_flat_rows_from_value(Some("64")), 64);
        assert_eq!(miss_flat_rows_from_value(Some("0")), MISS_FLAT_ROWS);
        assert_eq!(miss_flat_rows_from_value(Some("junk")), MISS_FLAT_ROWS);
        // 起動行 (テストプロセスでは env 未設定 = 既定と出る)。
        assert_eq!(exp_miss_rows_report(), "16 (既定)");
    }

    /// P70 マイクロベンチ: decode の expert 寄与 (10 expert) と層 1 回ぶんを
    /// 従来 (1 expert = 1 タスク) / 平坦化 (expert 横断の行タスク) で A/B する。
    ///
    /// 実形: n_embd 2560 / n_ff_exp 640 / gate・up は Q4_K、down は Q5_1 /
    /// top-k 10 (実モデルと同じ型・同じ行長)。512 expert の疎 GGUF に
    /// **選ばれる 10 expert だけ実データ**を置く (残りは穴)。
    ///
    /// このコンテナはホスト共有で load average が動くので、**同一バイナリ・
    /// 同一ループ内**で従来 / 平坦化 (行幅 3 種) を交互に測り、rep の min を取る。
    /// 実機の ms/token は main が測る (ここは CPU のスケジュール差だけ)。
    ///
    /// `cargo test --release -p denno-engine --lib --features cuda bench_decode_flat \
    ///   -- --ignored --nocapture --test-threads=1`
    #[test]
    #[cfg(feature = "par")]
    #[ignore = "P70 マイクロベンチ (CPU のスケジュール A/B。実機は main が測る)"]
    fn bench_decode_flat() {
        gpu_gemm_off();
        let _g = moe_test_guard();
        let (n_embd, n_ff, ne, k) = (2560usize, 640usize, 512usize, 10usize);
        let x = cache_test_x_seed(n_embd, 0);
        let sel: Vec<usize> = (0..k).collect();
        let dir = TempDir::new("p70bench");
        let path = dir.path("bench.gguf");
        write_expert_gguf_sparse(&path, n_embd, n_ff, ne, &sel);
        let f = GgufFile::open(&path).expect("open");
        let p = params_dims(n_embd, n_ff, ne, k);
        let mut lw = layer_at(&f, &p, 0);
        lw.gate_inp = gate_inp_for(&x, ne);

        // 選ばれる expert が「実データを置いた expert」と一致すること
        // (穴を読むと 0 が返り、比較が無意味になる)。
        let mut probs = vec![0.0f32; ne];
        lw.gate_inp.gemv(&x, &mut probs).expect("router");
        softmax(&mut probs);
        assert_eq!(select_topk(&probs, k), sel, "ルータの top-k が 0..k でない");

        let rows_opts = [32usize, 64, 128, 256];
        let reps = 20usize;
        let inner = 5usize;
        for threads in [8usize, 16] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            let mut hs_old = vec![0.0f32; k * n_embd];
            let mut hs_new: Vec<Vec<f32>> = (0..rows_opts.len())
                .map(|_| vec![0.0f32; k * n_embd])
                .collect();
            // 温め + 値の突き合わせ (経路が本当に同じ値を作ることを先に固定する)。
            pool.install(|| {
                expert_contribs_par_into(&lw, &sel, &x, n_embd, n_ff, None, &mut hs_old)
                    .expect("従来");
                for (i, &rows) in rows_opts.iter().enumerate() {
                    expert_contribs_flat_decode_into(
                        &lw, &sel, &x, n_embd, n_ff, &mut hs_new[i], rows,
                    )
                    .expect("平坦化");
                    assert_bitwise_eq(
                        &hs_old,
                        &hs_new[i],
                        &format!("expert 寄与 threads={threads} rows={rows}"),
                    );
                }
                assert!(hs_old.iter().any(|v| *v != 0.0), "寄与が全 0");
            });

            let mut best = vec![f64::INFINITY; 1 + rows_opts.len()];
            for _ in 0..reps {
                for vi in 0..1 + rows_opts.len() {
                    let t0 = std::time::Instant::now();
                    pool.install(|| {
                        for _ in 0..inner {
                            if vi == 0 {
                                expert_contribs_par_into(
                                    &lw, &sel, &x, n_embd, n_ff, None, &mut hs_old,
                                )
                                .expect("従来");
                            } else {
                                expert_contribs_flat_decode_into(
                                    &lw, &sel, &x, n_embd, n_ff, &mut hs_new[vi - 1],
                                    rows_opts[vi - 1],
                                )
                                .expect("平坦化");
                            }
                        }
                    });
                    let ms = t0.elapsed().as_secs_f64() * 1e3 / inner as f64;
                    best[vi] = best[vi].min(ms);
                }
            }

            // 層 1 回ぶん (ルータ + expert + shared expert)。平坦化は既定行幅。
            let mut out_old = Vec::new();
            let mut out_new = Vec::new();
            pool.install(|| {
                out_old = moe_ffn_quant_impl(
                    &p, &lw, &x, None, true, Route::Cpu, None, true, false, false,
                    IvView::none(),
                )
                .expect("層 従来");
                out_new = moe_ffn_quant_impl(
                    &p, &lw, &x, None, true, Route::Cpu, None, true, false, true,
                    IvView::none(),
                )
                .expect("層 平坦化");
            });
            assert_bitwise_eq(&out_old, &out_new, "層 1 回 (ルータ+expert+shared)");

            let mut layer_best = [f64::INFINITY; 2];
            for _ in 0..reps {
                for vi in 0..2 {
                    let t0 = std::time::Instant::now();
                    pool.install(|| {
                        for _ in 0..inner {
                            if vi == 0 {
                                out_old = moe_ffn_quant_impl(
                                    &p, &lw, &x, None, true, Route::Cpu, None, true, false, false,
                                    IvView::none(),
                                )
                                .expect("層 従来");
                            } else {
                                out_new = moe_ffn_quant_impl(
                                    &p, &lw, &x, None, true, Route::Cpu, None, true, false, true,
                                    IvView::none(),
                                )
                                .expect("層 平坦化");
                            }
                        }
                    });
                    let ms = t0.elapsed().as_secs_f64() * 1e3 / inner as f64;
                    layer_best[vi] = layer_best[vi].min(ms);
                }
            }

            eprintln!(
                "[bench70] threads={threads} ne={ne} k={k} e={n_embd} ff={n_ff} reps={reps}x{inner}"
            );
            eprintln!("[bench70] expert  old={:7.3} ms", best[0]);
            for (i, &rows) in rows_opts.iter().enumerate() {
                eprintln!(
                    "[bench70]           rows={rows:<3} {:7.3} ms ({:.2}x)",
                    best[i + 1],
                    best[0] / best[i + 1]
                );
            }
            eprintln!(
                "[bench70] layer   old={:7.3} ms | flat={:7.3} ms ({:.2}x)  \
                 48 層換算 old={:.1} / flat={:.1} ms/token (推定)",
                layer_best[0],
                layer_best[1],
                layer_best[0] / layer_best[1],
                layer_best[0] * 48.0,
                layer_best[1] * 48.0,
            );
        }
    }

    /// task 348/361 マイクロベンチ: ミス経路 (1 層のミス 1〜3 本) の行幅。
    ///
    /// 実形 (e=2560 / ff=640 / 512 expert) の疎 GGUF で
    /// [`expert_contribs_flat_decode_into`] をミス本数 × 行幅で交互に測る。
    /// 行幅は値に影響しない (測る前に全組み合わせのビット一致を確かめる)。
    ///
    /// この環境 (dev コンテナ) の値。実機の decode は lab が測る。
    ///
    /// `cargo test --release -p denno-engine --lib --features cuda -- \
    ///   --ignored --nocapture --test-threads=1 bench_miss_flat_rows`
    #[test]
    #[cfg(feature = "par")]
    #[ignore]
    fn bench_miss_flat_rows() {
        gpu_gemm_off();
        let _g = moe_test_guard();
        let (n_embd, n_ff, ne) = (2560usize, 640usize, 512usize);
        let x = cache_test_x_seed(n_embd, 0);
        let real: Vec<usize> = (0..10).collect();
        let dir = TempDir::new("t361bench");
        let path = dir.path("bench.gguf");
        write_expert_gguf_sparse(&path, n_embd, n_ff, ne, &real);
        let f = GgufFile::open(&path).expect("open");
        let p = params_dims(n_embd, n_ff, ne, 10);
        let lw = layer_at(&f, &p, 0);
        let rows_opts = [8usize, 16, 32, 46, 64, 92, 128, 256];
        let reps = 20usize;
        for threads in [14usize, 16] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            for k in [1usize, 2, 3, 10] {
                let sel: Vec<usize> = (0..k).collect();
                let mut want = vec![0.0f32; k * n_embd];
                let mut got = vec![0.0f32; k * n_embd];
                pool.install(|| {
                    expert_contribs_flat_decode_into(
                        &lw, &sel, &x, n_embd, n_ff, &mut want, DECODE_FLAT_ROWS,
                    )
                    .expect("従来幅");
                    assert!(want.iter().any(|v| *v != 0.0), "寄与が全 0");
                    for &rows in rows_opts.iter() {
                        expert_contribs_flat_decode_into(
                            &lw, &sel, &x, n_embd, n_ff, &mut got, rows,
                        )
                        .expect("平坦化");
                        assert_bitwise_eq(&want, &got, &format!("k={k} rows={rows}"));
                    }
                });
                let mut best = vec![f64::INFINITY; rows_opts.len()];
                for _ in 0..reps {
                    for (i, &rows) in rows_opts.iter().enumerate() {
                        let t0 = std::time::Instant::now();
                        pool.install(|| {
                            expert_contribs_flat_decode_into(
                                &lw, &sel, &x, n_embd, n_ff, &mut got, rows,
                            )
                            .expect("平坦化");
                        });
                        best[i] = best[i].min(t0.elapsed().as_secs_f64());
                    }
                }
                let mut line = format!("[bench361miss] threads={threads} k={k}:");
                for (i, &rows) in rows_opts.iter().enumerate() {
                    let name = if rows == 0 {
                        "auto".to_string()
                    } else {
                        rows.to_string()
                    };
                    line.push_str(&format!(" {name}={:.1}us", best[i] * 1e6));
                }
                eprintln!("{line}");
            }
        }
    }

    /// P77 マイクロベンチ: 実形 (e=2560 / ff=640 / 512 expert / top-10) の 1 層を
    /// 平坦化経路で 1 回回し、`DENNO_EXP_KPROF` の区間内訳を出す。
    ///
    /// 実行: `DENNO_EXP_KPROF=1 cargo test --release -p denno-engine --lib --features par \
    ///        -- --ignored --nocapture --test-threads=1 bench_exp_kprof_layer`
    ///
    /// **この環境 (dev コンテナ) の値**。実機の decode は main が測る。
    #[test]
    #[cfg(feature = "par")]
    #[ignore = "P77 マイクロベンチ (実測。通常のテストでは走らせない)"]
    fn bench_exp_kprof_layer() {
        use crate::model::exp_kprof as kp;
        gpu_gemm_off();
        let _g = moe_test_guard();
        let (n_embd, n_ff, ne, k) = (2560usize, 640usize, 512usize, 10usize);
        let x = cache_test_x_seed(n_embd, 0);
        let sel: Vec<usize> = (0..k).collect();
        let dir = TempDir::new("p77bench");
        let path = dir.path("kprof_bench.gguf");
        write_expert_gguf_sparse(&path, n_embd, n_ff, ne, &sel);
        let f = GgufFile::open(&path).expect("open");
        let p = params_dims(n_embd, n_ff, ne, k);
        let mut lw = layer_at(&f, &p, 0);
        lw.gate_inp = gate_inp_for(&x, ne);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .expect("pool");
        let run = |on: bool| {
            kp::set_for_test(on);
            kp::reset();
            let out = pool
                .install(|| {
                    moe_ffn_quant_impl(
                        &p, &lw, &x, None, true, Route::Cpu, None, false, false, true,
                        IvView::none(),
                    )
                })
                .expect("層 1 回");
            (out, if on { kp::line() } else { None })
        };
        // 1 回目はページフォルト込みなので捨てる (2 回目を測る)。
        let _ = run(false);
        let _ = run(true);
        let (want, _) = run(false);
        let (got, line) = run(true);
        kp::set_for_test(false);
        assert_bitwise_eq(&want, &got, "prof on/off の層 1 回");
        // 1 トークン = 48 層なので、この 1 層の 48 倍が decode の expert 時間になる。
        for l in line.expect("レポート").lines() {
            eprintln!("{l}");
        }
        let (calls, bytes, rows, ns) = kp::mat_stats(kp::GATE);
        eprintln!(
            "[bench77] gate {calls} 呼び出し / {rows} 行 / {:.1} MB / {:.3} ms \
             (48 層換算 {:.1} ms/token)",
            bytes as f64 / 1e6,
            ns as f64 / 1e6,
            ns as f64 / 1e6 * 48.0
        );
        kp::reset();
    }

    /// P77: 平坦化経路 (decode の既定) の区間内訳が実際に積まれること。
    ///
    /// `DENNO_EXP_KPROF` を強制した状態で「平坦化 off / on」の層 1 回を回し、
    /// (1) 出力がビット一致 (prof 経路は数値に触れない)、(2) gather / silu / scatter の
    /// ns が 0 でない、(3) `line()` が行列ごとの行を出す、を確かめる。
    #[test]
    #[cfg(feature = "par")]
    fn exp_kprof_flat_path_fills_intervals() {
        use crate::model::exp_kprof as kp;
        gpu_gemm_off();
        let _g = moe_test_guard();
        let (n_embd, n_ff, ne, k) = (256usize, 64usize, 8usize, 2usize);
        let x = cache_test_x_seed(n_embd, 0);
        let sel: Vec<usize> = (0..k).collect();
        let dir = TempDir::new("p77kprof");
        let path = dir.path("kprof.gguf");
        write_expert_gguf_sparse(&path, n_embd, n_ff, ne, &sel);
        let f = GgufFile::open(&path).expect("open");
        let p = params_dims(n_embd, n_ff, ne, k);
        let mut lw = layer_at(&f, &p, 0);
        lw.gate_inp = gate_inp_for(&x, ne);

        let run = |flat: bool| {
            moe_ffn_quant_impl(&p, &lw, &x, None, true, Route::Cpu, None, false, false, flat, IvView::none())
                .expect("層 1 回")
        };
        // 既定 (prof off) の基準
        kp::set_for_test(false);
        let want = run(true);
        // prof on: 区間が積まれ、値は変わらない
        kp::reset();
        kp::set_for_test(true);
        let got_flat = run(true);
        let got_old = run(false);
        let g = kp::layer_ns(kp::GATHER);
        let si = kp::layer_ns(kp::SILU);
        let sc = kp::layer_ns(kp::SCAT);
        let line = kp::line().expect("prof on なのにレポートが無い");
        let (gate_calls, gate_bytes, _, _) = kp::mat_stats(kp::GATE);
        let (down_calls, down_bytes, _, _) = kp::mat_stats(kp::DOWN);
        kp::set_for_test(false);
        kp::reset();

        assert_bitwise_eq(&want, &got_flat, "prof on/off の平坦化経路");
        assert_bitwise_eq(&got_flat, &got_old, "prof on の平坦化 vs 従来");
        // 平坦化 on の 1 層 + 従来 1 層ぶんが積まれている
        assert!(g > 0, "gather が 0 ns");
        assert!(si > 0, "silu が 0 ns");
        assert!(sc > 0, "scatter が 0 ns");
        assert!(gate_calls >= 2, "gate の呼び出しが {gate_calls} 回");
        assert!(down_calls >= 2, "down の呼び出しが {down_calls} 回");
        assert!(gate_bytes > 0 && down_bytes > 0, "バイトが 0");
        assert!(line.contains("gate calls="), "{line}");
        assert!(line.contains("down calls="), "{line}");
        assert!(line.contains("GB/s"), "{line}");
    }

    /// P64 マイクロベンチ: 1 層ぶん (512 expert 中 top-10、3 行列、`n_in = 2560` /
    /// `n_ff = 640`) を 8 スレッドで 100 回流し、旧経路と新経路の時間を出す。
    ///
    /// 実行: `cargo test --release -p denno-engine --lib -- --ignored --nocapture bench_expert_loop_layer`
    ///
    /// 重みは疎ファイル (選択される expert 0..9 だけ実データ)。ループだけの比較と、
    /// ルータ・shared expert を含む層 1 回ぶんの比較の両方を出す。数値は**実測**で、
    /// この環境 (コンテナ・共有ホスト) の値であることに注意。
    #[test]
    #[cfg(feature = "par")]
    #[ignore = "P64 マイクロベンチ (実測。通常のテストでは走らせない)"]
    fn bench_expert_loop_layer() {
        let _g = moe_test_guard();
        let dir = TempDir::new("bench_loop");
        let (n_embd, n_ff, ne, k) = (2560usize, 640usize, 512usize, 10usize);
        let x = cache_test_x_seed(n_embd, 0);
        let sel: Vec<usize> = (0..k).collect();
        let path = dir.path("bench.gguf");
        write_expert_gguf_sparse(&path, n_embd, n_ff, ne, &sel);
        let f = GgufFile::open(&path).expect("open");
        let p = params_dims(n_embd, n_ff, ne, k);
        let mut lw = layer_at(&f, &p, 0);
        lw.gate_inp = gate_inp_for(&x, ne);

        // 選択が「実データを置いた expert」と一致すること (穴を読んだら比較が無意味)。
        let mut probs = vec![0.0f32; ne];
        lw.gate_inp.gemv(&x, &mut probs).expect("router");
        softmax(&mut probs);
        assert_eq!(select_topk(&probs, k), sel, "ルータの top-k が 0..k でない");

        fn time_min<R, F: FnMut() -> R>(reps: usize, mut f: F) -> (f64, f64, R) {
            let mut best = f64::INFINITY;
            let mut sum = 0.0f64;
            let mut out: Option<R> = None;
            for _ in 0..reps {
                let t = std::time::Instant::now();
                let r = f();
                let ms = t.elapsed().as_secs_f64() * 1e3;
                best = best.min(ms);
                sum += ms;
                out = Some(r);
            }
            (best, sum / reps as f64, out.expect("reps > 0"))
        }

        let reps = 100usize;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .expect("8 スレッドのプール");

        // 1. expert ループだけ (ルータ・shared expert を含まない)。
        let (par_ms, par_avg, par_hs) = pool.install(|| {
            time_min(reps, || {
                expert_contribs_par(&lw, &sel, &x, n_embd, n_ff, None).expect("旧 par")
            })
        });
        let (loop_ms, loop_avg, loop_hs) = pool.install(|| {
            time_min(reps, || {
                expert_contribs_loop(&lw, &sel, &x, n_embd, n_ff, None).expect("新")
            })
        });
        let par_flat: Vec<f32> = par_hs.iter().flatten().copied().collect();
        assert_bitwise_eq(&par_flat, &loop_hs, "ループ: 旧 vs 新");

        // 2. 層 1 回ぶん (ルータ + expert + shared expert)。
        let (layer_old_ms, layer_old_avg, old_out) = pool.install(|| {
            time_min(reps, || {
                moe_ffn_quant_full_opts(&p, &lw, &x, None, true, Route::Cpu, None, true, false, IvView::none())
                    .expect("旧 層 par=1")
            })
        });
        let (layer_seq_ms, _, seq_out) = pool.install(|| {
            time_min(reps, || {
                moe_ffn_quant_full_opts(&p, &lw, &x, None, false, Route::Cpu, None, true, false, IvView::none())
                    .expect("旧 層 par=0")
            })
        });
        let (layer_new_ms, layer_new_avg, new_out) = pool.install(|| {
            time_min(reps, || {
                moe_ffn_quant_full_opts(&p, &lw, &x, None, true, Route::Cpu, None, true, true, IvView::none())
                    .expect("新 層")
            })
        });
        assert_bitwise_eq(&old_out, &new_out, "層: 旧 par=1 vs 新");
        assert_bitwise_eq(&old_out, &seq_out, "層: 旧 par=1 vs 旧 par=0");

        eprintln!(
            "[bench64] threads=8 reps={reps} experts={ne} topk={k} n_in={n_embd} n_ff={n_ff}"
        );
        eprintln!(
            "[bench64] loop  旧(expert 並列) min/avg {par_ms:.3} / {par_avg:.3} ms  \
             新(1 領域) min/avg {loop_ms:.3} / {loop_avg:.3} ms  比 {:.3}",
            loop_ms / par_ms
        );
        eprintln!(
            "[bench64] layer 旧 par=1 {layer_old_ms:.3} / {layer_old_avg:.3} ms  \
             旧 par=0 {layer_seq_ms:.3} ms  新 {layer_new_ms:.3} / {layer_new_avg:.3} ms  \
             比 新/旧(par=1) {:.3}  新/旧(par=0) {:.3}",
            layer_new_ms / layer_old_ms,
            layer_new_ms / layer_seq_ms
        );
    }

    /// task 67 ベンチ用: `nl` 層ぶんの expert 3 行列を、`real` の expert だけ実データで
    /// 置いた疎 GGUF。全 expert・全層のバイト列は同じ (ページの位置だけを分けるのが
    /// 目的なので、変種間の出力はビット一致するはず)。
    ///
    /// 見かけの大きさは `nl` × 3 行列 × `ne` expert (512 expert・48 層で 75 GB)。
    /// 実データは `nl` × `real.len()` × 1 expert ぶん (10 expert・48 層で 1.5 GB)。
    ///
    /// 戻り値は実データを書いた `(ファイル内オフセット, 長さ)` の一覧
    /// (ベンチの prewarm が穴を読まずに page cache へ載せるために使う)。
    fn write_expert_gguf_fault_bench(
        path: &Path,
        n_embd: usize,
        n_ff: usize,
        ne: usize,
        nl: usize,
        real: &[usize],
    ) -> Vec<(u64, u64)> {
        use std::io::{Seek, SeekFrom, Write};

        let gate_e = n_ff * (n_embd / 256) * 144;
        let down_e = n_embd * (n_ff / 32) * 24;
        let lens = [ne * gate_e, ne * gate_e, ne * down_e];
        let per = [gate_e, gate_e, down_e];

        // データ部のレイアウト (層ごとに gate / up / down を 32 B 整列で並べる)。
        let mut cur = 0usize;
        let mut offs_all: Vec<[usize; 3]> = Vec::with_capacity(nl);
        for _ in 0..nl {
            let mut offs = [0usize; 3];
            for i in 0..3 {
                offs[i] = cur;
                cur = align_up(cur + lens[i], ALIGN);
            }
            offs_all.push(offs);
        }

        let mut head = Vec::new();
        head.extend_from_slice(b"GGUF");
        head.extend_from_slice(&3u32.to_le_bytes());
        head.extend_from_slice(&((3 * nl) as u64).to_le_bytes());
        head.extend_from_slice(&(2u64).to_le_bytes());
        push_str(&mut head, "general.architecture");
        Kv::Str("qwen4exp").write(&mut head);
        push_str(&mut head, "general.alignment");
        Kv::U32(ALIGN as u32).write(&mut head);
        for il in 0..nl {
            for i in 0..3 {
                let name = match i {
                    0 => format!("blk.{il}.ffn_gate_exps.weight"),
                    1 => format!("blk.{il}.ffn_up_exps.weight"),
                    _ => format!("blk.{il}.ffn_down_exps.weight"),
                };
                let dims: [u64; 3] = if i < 2 {
                    [n_embd as u64, n_ff as u64, ne as u64]
                } else {
                    [n_ff as u64, n_embd as u64, ne as u64]
                };
                let ty = if i < 2 { TY_Q4_K } else { TY_Q5_1 };
                push_str(&mut head, &name);
                head.extend_from_slice(&(3u32).to_le_bytes());
                for d in dims {
                    head.extend_from_slice(&d.to_le_bytes());
                }
                head.extend_from_slice(&ty.to_le_bytes());
                head.extend_from_slice(&(offs_all[il][i] as u64).to_le_bytes());
            }
        }

        let head_len = align_up(head.len(), ALIGN);
        let mut file = std::fs::File::create(path).expect("疎 GGUF の作成");
        file.write_all(&head).expect("ヘッダ");
        file.set_len((head_len + cur) as u64).expect("set_len");

        // 3 行列ぶんのバイト列を 1 回だけ作り、全 expert の位置へ書く。
        let bytes = [
            q4k_expert_bytes(n_embd, n_ff, 3),
            q4k_expert_bytes(n_embd, n_ff, 11),
            q5_1_expert_bytes(n_ff, n_embd, 5),
        ];
        let mut ranges: Vec<(u64, u64)> = Vec::with_capacity(nl * real.len() * 3);
        for il in 0..nl {
            for &e in real {
                for (i, b) in bytes.iter().enumerate() {
                    let off = (head_len + offs_all[il][i] + e * per[i]) as u64;
                    file.seek(SeekFrom::Start(off)).expect("seek");
                    file.write_all(b).expect("expert のバイト列");
                    ranges.push((off, b.len() as u64));
                }
            }
        }
        // 冷 A/B で fadvise(DONTNEED) を効かせるには dirty を落としておく。
        file.sync_all().expect("sync_all");
        ranges
    }

    /// task 67: 3 ゲートの値解釈 (純粋関数)。
    #[test]
    fn expert_fault_gates_from_value() {
        // prefetch: 既定 off。1 / on / true のときだけ on。
        assert!(!expert_prefetch_from_value(None));
        assert!(!expert_prefetch_from_value(Some("0")));
        assert!(!expert_prefetch_from_value(Some("off")));
        assert!(!expert_prefetch_from_value(Some("")));
        assert!(expert_prefetch_from_value(Some("1")));
        assert!(expert_prefetch_from_value(Some("on")));
        assert!(expert_prefetch_from_value(Some("true")));
        // hugepage: 既定 off。
        assert!(!expert_hugepage_from_value(None));
        assert!(!expert_hugepage_from_value(Some("0")));
        assert!(expert_hugepage_from_value(Some("1")));
        assert!(expert_hugepage_from_value(Some("ON")));
        // scratch: 既定 off。
        assert!(!expert_scratch_from_value(None));
        assert!(!expert_scratch_from_value(Some("off")));
        assert!(expert_scratch_from_value(Some("1")));
        assert!(expert_scratch_from_value(Some("true")));
    }

    /// task 67: 3 ゲートが実際に分岐を変えること (緑 = 通った証拠ではないので、
    /// カウンタで通過を数える) と、どの組み合わせでも出力がビット一致すること。
    ///
    /// - prefetch=1: `ExpertW::prefetch` が選択 expert × 3 行列ぶん呼ばれる
    /// - hugepage=1: `MADV_HUGEPAGE` が expert テンソル 3 本ぶん呼ばれる
    /// - scratch=1: スレッドローカルの作業域が expert ごとに 1 回使われる
    #[test]
    #[cfg(feature = "par")]
    fn expert_fault_gates_take_effect_and_stay_bit_identical() {
        use std::sync::atomic::Ordering;
        let _g = moe_test_guard();
        // P74: prefetch / scratch のカウンタは**旧 (expert 単位) 経路**の
        // `ExpertW::prefetch` とスレッドローカル作業域が積む。既定 on の平坦化
        // 経路はどちらも通らないので、ゲートの発動回数を見るここでは明示的に
        // off へ固定する (ゲート自体の意味は変えない)。
        set_expert_flat_decode_for_test(Some(false));
        let dir = TempDir::new("gates67");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let x = cache_test_x(p.n_embd);

        let run = |lw: &LayerW| {
            // emit_router=false: テレメトリを出す他のテストにフレームを混ぜない。
            moe_ffn_quant_full_opts(&p, lw, &x, None, true, Route::Cpu, None, false, false, IvView::none())
                .expect("moe_ffn_quant")
        };

        // 1. 既定 (3 つとも off): 先読みも hugepage も scratch も 0 回。
        set_expert_fault_gates_for_test(None, None, None);
        let lw = cache_test_layer(&f, &p);
        let p0 = EXPERT_PREFETCH_CALLS.load(Ordering::Relaxed);
        let s0 = EXPERT_SCRATCH_REUSES.load(Ordering::Relaxed);
        let want = run(&lw);
        assert_eq!(
            EXPERT_PREFETCH_CALLS.load(Ordering::Relaxed) - p0,
            0,
            "既定で先読みが走った"
        );
        assert_eq!(
            EXPERT_SCRATCH_REUSES.load(Ordering::Relaxed) - s0,
            0,
            "既定で scratch を使った"
        );
        assert!(want.iter().all(|v| v.is_finite()), "既定経路に NaN/inf");

        // 2. prefetch=1: 選択 expert (k) × 3 行列ぶん。
        set_expert_fault_gates_for_test(Some(true), Some(false), Some(false));
        let p1 = EXPERT_PREFETCH_CALLS.load(Ordering::Relaxed);
        let got = run(&lw);
        let calls = EXPERT_PREFETCH_CALLS.load(Ordering::Relaxed) - p1;
        assert_eq!(
            calls,
            p.n_expert_used * 3,
            "先読みの回数が expert × 3 でない: {calls}"
        );
        assert_bitwise_eq(&want, &got, "prefetch=1");

        // 3. hugepage=1: 層を読み直すと expert テンソル 3 本ぶん madvise する。
        set_expert_fault_gates_for_test(Some(false), Some(true), Some(false));
        let h1 = EXPERT_HUGEPAGE_ADVISES.load(Ordering::Relaxed);
        let lw_h = cache_test_layer(&f, &p);
        let advises = EXPERT_HUGEPAGE_ADVISES.load(Ordering::Relaxed) - h1;
        assert_eq!(advises, 3, "MADV_HUGEPAGE が 3 テンソルでない: {advises}");
        let got = run(&lw_h);
        assert_bitwise_eq(&want, &got, "hugepage=1");

        // 4. scratch=1: 作業域を expert ごとに 1 回借りる。
        set_expert_fault_gates_for_test(Some(false), Some(false), Some(true));
        let s1 = EXPERT_SCRATCH_REUSES.load(Ordering::Relaxed);
        let got = run(&lw);
        let reuses = EXPERT_SCRATCH_REUSES.load(Ordering::Relaxed) - s1;
        assert_eq!(
            reuses, p.n_expert_used,
            "scratch の回数が expert 数でない: {reuses}"
        );
        assert_bitwise_eq(&want, &got, "scratch=1");

        set_expert_fault_gates_for_test(None, None, None);
        set_expert_flat_decode_for_test(None);
    }

    /// task 67 マイクロベンチ: expert GEMV の minor fault / 時間 / 実効 GB/s を
    /// 変種ごとに測る。
    ///
    /// 実形状 (e = 2560 / ff = 640 / 512 expert / top-10) の疎 GGUF を作り、48 層を
    /// 6 変種 × 8 層に分ける (変種ごとに触るページが違う = 毎回ほぼ cold)。1 変種
    /// 8 層を測って 6 倍し「1 トークン (48 層) 相当」に換算する (**換算値**)。
    /// 同じ 8 層をもう 1 回回した「温」も出す (フォールトを含まない時間)。
    ///
    /// 実行: `cargo test --release -p denno-engine --lib -- --ignored --nocapture --test-threads=1 bench_expert_faults`
    #[test]
    #[cfg(feature = "par")]
    #[ignore = "task 67 マイクロベンチ (実測。通常のテストでは走らせない)"]
    fn bench_expert_faults() {
        let _g = moe_test_guard();
        let (n_embd, n_ff, ne, k) = (2560usize, 640usize, 512usize, 10usize);
        let nl = 48usize;
        let sel: Vec<usize> = (0..k).collect();
        let x = cache_test_x_seed(n_embd, 0);
        let p = params_dims(n_embd, n_ff, ne, k);

        // 1. 疎 GGUF (見かけ 75 GB / 実データ 1.5 GB)。/tmp は overlay なので
        //    workspace の target/ に置く (ext4 の sparse + mmap)。
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("target")
            .join("bench67");
        std::fs::create_dir_all(&dir).expect("bench67 dir");
        let path = dir.join("expert_faults.gguf");
        let t_write = std::time::Instant::now();
        let ranges = write_expert_gguf_fault_bench(&path, n_embd, n_ff, ne, nl, &sel);
        let apparent = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        eprintln!(
            "[bench67] 疎 GGUF: {} (見かけ {:.1} GB / 実データ約 {:.2} GB / 書き {:?})",
            path.display(),
            apparent as f64 / 1e9,
            ranges.iter().map(|r| r.1).sum::<u64>() as f64 / 1e9,
            t_write.elapsed()
        );

        // /proc/self/stat の 10 番目 (minflt) と 12 番目 (majflt)。comm は括弧を
        // 含みうるので、最後の ')' 以降をフィールド 3 として数える。
        fn stat_faults() -> (u64, u64) {
            let s = std::fs::read_to_string("/proc/self/stat").expect("/proc/self/stat");
            let rest = &s[s.rfind(')').expect("stat: )") + 1..];
            let f: Vec<&str> = rest.split_whitespace().collect();
            (
                f[7].parse().expect("minflt"),
                f[9].parse().expect("majflt"),
            )
        }

        /// 実データの範囲だけを read して page cache に載せる (穴は読まない)。
        fn prewarm_ranges(path: &Path, ranges: &[(u64, u64)]) {
            use std::io::{Read, Seek, SeekFrom};
            let mut f = std::fs::File::open(path).expect("prewarm open");
            let mut buf = vec![0u8; 1 << 20];
            let mut total = 0u64;
            for &(off, len) in ranges {
                f.seek(SeekFrom::Start(off)).expect("prewarm seek");
                let mut left = len as usize;
                while left > 0 {
                    let want = left.min(buf.len());
                    let n = f.read(&mut buf[..want]).expect("prewarm read");
                    if n == 0 {
                        break;
                    }
                    left -= n;
                    total += n as u64;
                }
            }
            eprintln!(
                "[bench67] prewarm: {:.2} GB を page cache に載せた",
                total as f64 / 1e9
            );
        }

        /// ファイルの page cache を落とす (冷 A/B 用)。mapped のページは落ちないので、
        /// 先に mmap を全部 drop しておくこと。
        fn fadvise_dontneed(path: &Path) {
            #[cfg(target_os = "linux")]
            {
                use std::os::unix::io::AsRawFd;
                extern "C" {
                    fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
                }
                const POSIX_FADV_DONTNEED: i32 = 4;
                let f = std::fs::File::open(path).expect("fadvise open");
                let rc = unsafe { posix_fadvise(f.as_raw_fd(), 0, 0, POSIX_FADV_DONTNEED) };
                eprintln!("[bench67] fadvise(DONTNEED) rc={rc}");
            }
        }

        /// 1 パス = `layers` ぶんの expert ループ (prefetch → gate/up/down)。
        fn run_pass(layers: &[LayerW], sel: &[usize], x: &[f32], e: usize, ff: usize) -> Vec<f32> {
            let mut out = vec![0.0f32; e];
            for lw in layers {
                prefetch_selected(lw, sel);
                if expert_scratch_enabled() {
                    let mut hs = vec![0.0f32; sel.len() * e];
                    expert_contribs_par_into(lw, sel, x, e, ff, None, &mut hs).expect("par_into");
                    for (t, _) in sel.iter().enumerate() {
                        let hh = &hs[t * e..(t + 1) * e];
                        for j in 0..e {
                            out[j] += hh[j];
                        }
                    }
                } else {
                    let hs = expert_contribs_par(lw, sel, x, e, ff, None).expect("par");
                    for (t, _) in sel.iter().enumerate() {
                        let hh = &hs[t];
                        for j in 0..e {
                            out[j] += hh[j];
                        }
                    }
                }
            }
            out
        }

        fn median_f(mut v: Vec<f64>) -> f64 {
            v.sort_by(|a, b| a.partial_cmp(b).expect("NaN"));
            v[v.len() / 2]
        }
        fn median_u(mut v: Vec<u64>) -> u64 {
            v.sort_unstable();
            v[v.len() / 2]
        }

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .expect("8 スレッドのプール");

        prewarm_ranges(&path, &ranges);

        // ---- A. 温 page cache (minor fault だけ) ----
        // 6 変種 × 8 ラウンド × 1 層 = 48 層。ラウンドごとに層を変えるので、どの
        // 測定も「まだ 1 ページも触っていない」層から始まる (cold)。2×2 (prefetch ×
        // scratch) に hugepage と base の繰り返しを足して、同じ実行で比べる。
        let variants: [(&str, Option<bool>, Option<bool>, Option<bool>); 6] = [
            ("pf0,sc0", Some(false), Some(false), Some(false)),
            ("pf0,sc1", Some(false), Some(false), Some(true)),
            ("pf1,sc0", Some(true), Some(false), Some(false)),
            ("pf1,sc1", Some(true), Some(false), Some(true)),
            ("huge=1", Some(false), Some(true), Some(false)),
            ("pf0,sc0(再)", Some(false), Some(false), Some(false)),
        ];
        let (per_round, rounds) = (1usize, 8usize);
        let mut stats: Vec<(Vec<f64>, Vec<u64>, Vec<f64>, Vec<u64>)> = (0..variants.len())
            .map(|_| (Vec::new(), Vec::new(), Vec::new(), Vec::new()))
            .collect();
        let mut reference: Option<Vec<f32>> = None;
        let bytes_per_layer;
        eprintln!(
            "[bench67] 形 e={n_embd} ff={n_ff} ne={ne} topk={k} layers={nl} / 温: {rounds} ラウンド × {per_round} 層（換算 ×{}）",
            nl / per_round
        );
        {
            let f = GgufFile::open(&path).expect("open");
            {
                let mut lw = layer_at(&f, &p, 0);
                lw.gate_inp = f32_linear(1, 1, 0.1);
                let lens = expert_lens(&lw);
                bytes_per_layer = (lens[0] + lens[1] + lens[2]) * k;
            }
            for round in 0..rounds {
                for slot in 0..variants.len() {
                    // 変種の順番をラウンドごとに回す (「そのラウンドの最初の測定」が
                    // いつも同じ変種にならないように)。
                    let vi = (slot + round) % variants.len();
                    let (name, pf, hp, sc) = &variants[vi];
                    // ゲートは層を読む前に立てる (hugepage は ExpertW::load で投げる)。
                    set_expert_fault_gates_for_test(*pf, *hp, *sc);
                    let il0 = round * variants.len() * per_round + slot * per_round;
                    let mut layers: Vec<LayerW> = Vec::with_capacity(per_round);
                    for il in il0..il0 + per_round {
                        let mut lw = layer_at(&f, &p, il);
                        // ルータはこのベンチでは使わない (層ごとの 5 MB の F32 を捨てる)。
                        lw.gate_inp = f32_linear(1, 1, 0.1);
                        layers.push(lw);
                    }

                    let (f0, m0) = stat_faults();
                    let t0 = std::time::Instant::now();
                    let out = pool.install(|| run_pass(&layers, &sel, &x, n_embd, n_ff));
                    let cold_ms = t0.elapsed().as_secs_f64() * 1e3;
                    let (f1, m1) = stat_faults();

                    let t1 = std::time::Instant::now();
                    let out2 = pool.install(|| run_pass(&layers, &sel, &x, n_embd, n_ff));
                    let warm_ms = t1.elapsed().as_secs_f64() * 1e3;
                    let (f2, _) = stat_faults();

                    assert_bitwise_eq(&out, &out2, &format!("{name} r{round}: cold vs warm"));
                    match &reference {
                        Some(r) => assert_bitwise_eq(r, &out, &format!("{name} r{round}: 変種間")),
                        None => reference = Some(out.clone()),
                    }
                    eprintln!(
                        "[bench67] A r{round} {name:<12} cold {cold_ms:6.2} ms  minor {:5}  major {}  \
                         warm {warm_ms:6.2} ms (warm minor {})",
                        f1 - f0,
                        m1 - m0,
                        f2 - f1
                    );
                    let s = &mut stats[vi];
                    s.0.push(cold_ms);
                    s.1.push(f1 - f0);
                    s.2.push(warm_ms);
                    s.3.push(m1 - m0);
                }
            }
        }

        let per_round_bytes = bytes_per_layer as f64 * per_round as f64;
        let scale = nl as f64 / per_round as f64;
        eprintln!(
            "[bench67] === 温 page cache: {rounds} ラウンドの中央値（換算 ×{scale} = 48 層）==="
        );
        for (vi, (name, ..)) in variants.iter().enumerate() {
            let s = &stats[vi];
            let cold = median_f(s.0.clone());
            let warm = median_f(s.2.clone());
            let cold_min = s.0.iter().copied().fold(f64::INFINITY, f64::min);
            let warm_min = s.2.iter().copied().fold(f64::INFINITY, f64::min);
            let faults = median_u(s.1.clone());
            let major = median_u(s.3.clone());
            eprintln!(
                "[bench67] {name:<12} cold min {cold_min:6.2} / 中央 {cold:6.2} ms/{per_round}層 \
                 (換算 {:6.1} ms/token)  minor {faults:5} (換算 {:6.0}/token, {:5.2}/GEMV)  \
                 major {major}  warm min {warm_min:6.2} / 中央 {warm:6.2} (換算 {:6.1})  \
                 {:5.2} GB/s cold(min) / {:5.2} GB/s warm(min)",
                cold * scale,
                faults as f64 * scale,
                faults as f64 / (per_round * k * 3) as f64,
                warm * scale,
                per_round_bytes / (cold_min / 1e3) / 1e9,
                per_round_bytes / (warm_min / 1e3) / 1e9
            );
        }

        // ---- B. 冷 page cache (major fault / I/O がある状態) ----
        // 30 GB のモデルは 60 GB のマシンでもページが追い出されうる。prefetch
        // (`MADV_WILLNEED`) が効くのはこの条件だけなので、fadvise(DONTNEED) で
        // 落としてから prefetch の有無を比べる。
        eprintln!("[bench67] === 冷 page cache: fadvise(DONTNEED) の後、新しい写像で {per_round} 層 ===");
        for (name, pf) in [("prefetch=1", true), ("prefetch=0", false)] {
            for round in 0..3 {
                fadvise_dontneed(&path);
                set_expert_fault_gates_for_test(Some(pf), Some(false), Some(false));
                let (f0, m0) = stat_faults();
                let t0 = std::time::Instant::now();
                let out = {
                    let f = GgufFile::open(&path).expect("open (cold)");
                    let mut layers: Vec<LayerW> = Vec::with_capacity(per_round);
                    for il in 0..per_round {
                        let mut lw = layer_at(&f, &p, il);
                        lw.gate_inp = f32_linear(1, 1, 0.1);
                        layers.push(lw);
                    }
                    pool.install(|| run_pass(&layers, &sel, &x, n_embd, n_ff))
                    // f と layers はここで drop (次の fadvise を効かせる)
                };
                let cold_ms = t0.elapsed().as_secs_f64() * 1e3;
                let (f1, m1) = stat_faults();
                if let Some(r) = &reference {
                    assert_bitwise_eq(r, &out, &format!("冷 {name} r{round}"));
                }
                eprintln!(
                    "[bench67] B r{round} {name:<10} cold {cold_ms:6.2} ms/{per_round}層 (換算 {:6.1} ms/token)  \
                     minor {:5} (換算 {:6.0}/token)  major {}  {:5.2} GB/s",
                    cold_ms * scale,
                    f1 - f0,
                    (f1 - f0) as f64 * scale,
                    m1 - m0,
                    per_round_bytes / (cold_ms / 1e3) / 1e9
                );
            }
        }
        set_expert_fault_gates_for_test(None, None, None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// task 65: decode の入口 (`moe_ffn_quant` = router の GEMV) が
    /// `DENNO_CPU_GEMV_PAR` の分岐に実際に入ること。
    ///
    /// 合成モデルは 256→4 と小さく既定のしきい値 (8 行 / 32768 MAC) には届かない
    /// ので、しきい値だけテスト用に 1 に下げて「入口 → `GgufLinear::gemv` →
    /// 行並列」の配線を見る。しきい値そのものの根拠は `linear_gguf` 側の
    /// 実形状テスト (`cpu_gemv_par_*`) で固定している。
    #[test]
    fn router_gemv_takes_cpu_gemv_par_gate() {
        use crate::model::linear_gguf as lg;
        // 入口 (`moe_ffn_quant`) は RouterTopK のフレームを出すので、
        // テレメトリを数えるテストと直列化する (moe_test_guard を先に取る)。
        let _m = moe_test_guard();
        let _g = lg::par_test_lock();
        let dir = TempDir::new("cpu_gemv_par");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let lw = cache_test_layer(&f, &p);
        let x = cache_test_x(p.n_embd);

        // 1. 既定 (off): 新経路は 1 回も立たない。
        lg::set_cpu_gemv_par_for_test(Some(false));
        lg::set_par_min_rows_for_test(None);
        lg::par_reset();
        let off = moe_ffn_quant(&p, &lw, &x, None).expect("off");
        assert_eq!(lg::par_launches(), 0, "off で行並列が立った");

        // 2. on: 入口 1 回で F32 の GEMV が新経路に入る (router を含む)。
        lg::set_par_min_rows_for_test(Some(1));
        lg::set_cpu_gemv_par_for_test(Some(true));
        lg::par_reset();
        let on = moe_ffn_quant(&p, &lw, &x, None).expect("on");
        let launches = lg::par_launches();
        let tasks = lg::par_tasks();
        lg::set_cpu_gemv_par_for_test(None);
        lg::set_par_min_rows_for_test(None);
        assert!(
            launches >= 1,
            "moe_ffn_quant の入口から行並列が 1 回も立たない (配線が切れている)"
        );
        assert!(tasks >= launches, "タスク数 {tasks} < 領域数 {launches}");

        // 3. 出力はビット一致 (行の分割は丸め順序を変えない)。
        assert_bitwise_eq(&off, &on, "router gate off vs on");
    }

    /// task 69: `moe_ffn_quant` の入口から GPU 経路の**分岐に実際に入る**こと。
    ///
    /// このコンテナに GPU は無いので計算は CPU に落ちるが、`cuda_enabled()` より
    /// 前のカウンタで「配線が生きている」ことを数える。ゲート off では 0 回。
    #[cfg(feature = "cuda")]
    #[test]
    fn router_gpu_gate_is_live_end_to_end() {
        use crate::model::linear_gguf as lg;
        let _m = moe_test_guard();
        let _g = lg::par_test_lock();
        let dir = TempDir::new("router_gpu_e2e");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let lw = cache_test_layer(&f, &p);
        let x = cache_test_x(p.n_embd);

        // 1. ゲート off: 入口に 1 回も入らない。
        lg::set_router_gpu_for_test(Some(false));
        crate::cuda::router_gpu_attempts_reset();
        let off = moe_ffn_quant(&p, &lw, &x, None).expect("off");
        let n_off = crate::cuda::router_gpu_attempts();

        // 2. ゲート on: 層 1 回 = 1 回入る (GPU が無いので中で CPU に落ちる)。
        lg::set_router_gpu_for_test(Some(true));
        crate::cuda::router_gpu_attempts_reset();
        let on = moe_ffn_quant(&p, &lw, &x, None).expect("on");
        let n_on = crate::cuda::router_gpu_attempts();
        lg::set_router_gpu_for_test(None);

        assert_eq!(n_off, 0, "ゲート off なのにルータ GPU 入口に入った");
        assert_eq!(n_on, 1, "ルータ GPU 入口の回数が層数と違う (配線が切れている)");
        // 3. 数値はどちらの経路でも同一 (CPU に落ちた場合)。
        assert_bitwise_eq(&off, &on, "router gpu gate off vs on");
    }

    /// task 69: ルータ GPU 経路の対象は `ffn_gate_inp` **だけ**であること。
    ///
    /// `with_router_gpu(true)` を付けた 1 本だけが対象で、shared expert の
    /// ルータ (`ffn_gate_inp_shexp`) や他の F32 線形は CPU のまま (VRAM を
    /// 余計に使わない)。実際に GPU で回ることはこのコンテナでは確かめられない
    /// (`cuda_f32_gemv_bit_exact_vs_cpu` が実機用の #[ignore])。
    #[test]
    fn router_gpu_flag_is_on_gate_inp_only() {
        let dir = TempDir::new("router_gpu_flag");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let lw = cache_test_layer(&f, &p);
        assert!(
            lw.gate_inp.router_gpu(),
            "ffn_gate_inp がルータ GPU 経路の対象になっていない (配線が切れている)"
        );
        assert!(
            !lw.gate_inp_shexp.router_gpu(),
            "shared expert のルータまで対象になっている"
        );
        for (name, on) in [
            ("gate_shexp", lw.gate_shexp.router_gpu()),
            ("up_shexp", lw.up_shexp.router_gpu()),
            ("down_shexp", lw.down_shexp.router_gpu()),
            ("hc_attn_down", lw.hc_attn_down.router_gpu()),
            ("hc_ffn_inject", lw.hc_ffn_inject.router_gpu()),
        ] {
            assert!(!on, "{name} までルータ扱いになっている");
        }
    }

    /// task 182: `DENNO_PF_ROUTER_GPU` / `DENNO_PF_PLE_F32_GPU` の解釈。
    /// lab t182b で combo が目標を満たしたので **未設定は on**、明示的な偽値だけ off。
    #[test]
    fn pf182_gates_default_on() {
        assert!(pf182_from_value(None), "未設定が off になった (lab t182b で既定 on)");
        for v in ["0", "off", "false", "no", "", " 0 ", "OFF"] {
            assert!(!pf182_from_value(Some(v)), "{v:?} が on になった");
        }
        for v in ["1", "on", "true", "yes", " ON ", "Yes"] {
            assert!(pf182_from_value(Some(v)), "{v:?} が off になった");
        }
    }

    /// task 182: 通常順 prefill の router GPU 入口が層 1 回ぶんだけ呼ばれ、
    /// GPU の無いこの環境では CPU とビット一致すること。
    ///
    /// 最小トークン数未満 (19 トークンの next turn 相当) では入口に入らない
    /// ことも数える。GPU 入口のカウンタは `cuda_enabled()` より前で増えるので、
    /// 配線が生きている証拠になる。
    #[cfg(feature = "cuda")]
    #[test]
    fn pf182_router_batch_gate_wiring() {
        let _m = moe_test_guard();
        let dir = TempDir::new("pf182_router");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let lw = cache_test_layer(&f, &p);
        let nt = PF_ROUTER_GPU_MIN_TOK;
        let xs = cache_test_x(nt * p.n_embd);

        // 1. ゲート off: 入口に 1 回も入らない。
        set_pf182_for_test(PF182_GATE_ROUTER, Some(false));
        crate::cuda::f32_gemm_batch_attempts_reset();
        let off = moe_ffn_quant_batch_opts(&p, &lw, &xs, nt, None, moe_schedule(), false)
            .expect("batch (off)");
        assert_eq!(
            crate::cuda::f32_gemm_batch_attempts(),
            0,
            "ゲート off なのに router の GPU 入口に入った"
        );

        // 2. ゲート on: 層 1 回 = 1 回。GPU が無いので中で CPU に落ちる。
        set_pf182_for_test(PF182_GATE_ROUTER, Some(true));
        crate::cuda::f32_gemm_batch_attempts_reset();
        let on = moe_ffn_quant_batch_opts(&p, &lw, &xs, nt, None, moe_schedule(), false)
            .expect("batch (on)");
        assert_eq!(
            crate::cuda::f32_gemm_batch_attempts(),
            1,
            "router GPU 入口の回数が層数と違う (配線が切れている)"
        );
        assert_eq!(off.len(), on.len());
        for i in 0..off.len() {
            assert_eq!(
                off[i].to_bits(),
                on[i].to_bits(),
                "要素 {i}: off={} on={}",
                off[i],
                on[i]
            );
        }

        // 3. 最小トークン数未満は入口に入らない (19 トークンの next turn を
        //    従来どおり CPU のままにする境界)。
        let nt2 = PF_ROUTER_GPU_MIN_TOK - 1;
        let xs2 = cache_test_x(nt2 * p.n_embd);
        crate::cuda::f32_gemm_batch_attempts_reset();
        let _ = moe_ffn_quant_batch_opts(&p, &lw, &xs2, nt2, None, moe_schedule(), false)
            .expect("batch (small)");
        assert_eq!(
            crate::cuda::f32_gemm_batch_attempts(),
            0,
            "nt={nt2} で router の GPU 入口に入った"
        );
        set_pf182_for_test(PF182_GATE_ROUTER, None);
    }

    /// task 182: ルータの丸めが top-k の**選択**を変えるのは、k 番目と k+1 番目の
    /// 確率が丸めの幅に収まる同値のときだけであること。
    ///
    /// 実形 (2,560 -> 512、n_expert_used = 5) で、CPU のバッチ経路と GPU の
    /// `dno_gemm_f32_mt` と同じ演算列 (FMA・k 昇順) の確率を出して top-k を比べる。
    /// 変わった行は必ず「境界の差が 8 ulp 以内」であることを要求する。
    #[test]
    fn pf182_router_topk_rounding_only_flips_ties() {
        let (k, n, nt, kk) = (2560usize, 512usize, 16usize, 5usize);
        let w = f32_linear(k, n, 0.001);
        let xs = cache_test_x(nt * k);
        // CPU のバッチ経路 (実装が prefill で使う経路そのもの)
        let mut cpu = vec![0.0f32; nt * n];
        gemv_batch(&w, &xs, nt, &mut cpu).expect("gemv_batch");
        // GPU カーネルと同じ演算列: 1 出力につき k 昇順の `fma(w, x, acc)`。
        // (`kernels_elem.cu` の `dno_gemm_f32_mt` は `__fmaf_rn` を kk 昇順に
        //  32 要素ずつ、k0 タイル昇順で回す = 全体で k 昇順の 1 本の鎖。)
        let bytes = w.blocks().as_slice();
        let mut gpu = vec![0.0f32; nt * n];
        for t in 0..nt {
            for r in 0..n {
                let row = &bytes[r * k * 4..(r + 1) * k * 4];
                let xr = &xs[t * k..(t + 1) * k];
                let mut acc = 0.0f32;
                for i in 0..k {
                    let wv = f32::from_le_bytes([
                        row[i * 4],
                        row[i * 4 + 1],
                        row[i * 4 + 2],
                        row[i * 4 + 3],
                    ]);
                    acc = wv.mul_add(xr[i], acc);
                }
                gpu[t * n + r] = acc;
            }
        }
        let mut diff_rows = 0usize;
        let mut buf = Vec::new();
        for t in 0..nt {
            let mut pc = cpu[t * n..(t + 1) * n].to_vec();
            let mut pg = gpu[t * n..(t + 1) * n].to_vec();
            softmax(&mut pc);
            softmax(&mut pg);
            let mut sc = vec![0usize; kk];
            let mut sg = vec![0usize; kk];
            select_topk_into(&pc, kk, &mut buf, &mut sc);
            select_topk_into(&pg, kk, &mut buf, &mut sg);
            if sc != sg {
                diff_rows += 1;
                let mut sorted = pc.clone();
                sorted.sort_by(|a, b| b.partial_cmp(a).expect("確率に NaN"));
                let gap = sorted[kk - 1] - sorted[kk];
                let scale = sorted[kk - 1].abs().max(1e-30);
                assert!(
                    gap <= 8.0 * f32::EPSILON * scale,
                    "行 {t}: 同値でないのに top-k が変わった (gap={gap:e}, p={scale:e})"
                );
            }
        }
        eprintln!("[pf182] router top-k が丸めで変わった行: {diff_rows}/{nt}");
        assert!(
            diff_rows <= nt / 2,
            "丸めだけで半数以上の行の top-k が変わった ({diff_rows}/{nt})"
        );
    }

    /// task 182: PLE の key / value 射影 (F32) の GPU 入口がチャンク 1 回で
    /// 2 回 (key + value) 呼ばれ、GPU の無いこの環境では CPU とビット一致すること。
    #[cfg(feature = "cuda")]
    #[test]
    fn pf182_ple_f32_gate_wiring() {
        let _m = moe_test_guard();
        let mut p = cache_test_params();
        // conv 履歴の長さは `(kern-1) * ngram_size * hc_dim`。
        p.ple_conv_kernel = 2;
        p.ple_ngram_size = 1;
        let hp = HyperParams {
            hc: p.n_hc,
            n_embd: p.n_embd,
            low_rank: p.hc_low_rank,
            eps: p.rms_eps,
        };
        let e = p.n_embd;
        let hc_dim = p.hc_dim();
        let pw = PleW {
            key: f32_linear(e, hc_dim, 0.1),
            value: f32_linear(e, e, 0.2),
            norm_key: vec![1.0; hc_dim],
            norm_query: vec![1.0; hc_dim],
            norm_conv: vec![1.0; hc_dim],
            conv1d: vec![0.25; 2 * hc_dim],
            conv1d_t: vec![0.25; 2 * hc_dim],
            kern: 2,
        };
        let nt = PF_PLE_F32_GPU_MIN_TOK;
        let flat: Vec<f32> = cache_test_x(nt * e);
        let rows: Vec<&[f32]> = (0..nt).map(|t| &flat[t * e..(t + 1) * e]).collect();
        let mk_res = |p: &Qwen4ExpParams| -> Vec<HyperStreams> {
            (0..nt)
                .map(|t| HyperStreams {
                    data: cache_test_x_seed(p.hc_dim(), t),
                    hc: p.n_hc,
                    n_embd: p.n_embd,
                })
                .collect()
        };
        let mut hist = vec![0.0f32; hc_dim];

        // 1. ゲート off: 入口に入らない。
        set_pf182_for_test(PF182_GATE_PLE_F32, Some(false));
        let mut r_off = mk_res(&p);
        crate::cuda::f32_gemm_batch_attempts_reset();
        ple_inject_batch(&p, &pw, &rows, &mut r_off, &hp, &mut hist, None).expect("ple off");
        assert_eq!(
            crate::cuda::f32_gemm_batch_attempts(),
            0,
            "ゲート off なのに PLE の GPU 入口に入った"
        );

        // 2. ゲート on: key と value で 2 回。GPU が無いので CPU に落ちる。
        set_pf182_for_test(PF182_GATE_PLE_F32, Some(true));
        let mut r_on = mk_res(&p);
        let mut hist2 = vec![0.0f32; hc_dim];
        crate::cuda::f32_gemm_batch_attempts_reset();
        ple_inject_batch(&p, &pw, &rows, &mut r_on, &hp, &mut hist2, None).expect("ple on");
        assert_eq!(
            crate::cuda::f32_gemm_batch_attempts(),
            2,
            "PLE の GPU 入口が key/value の 2 回でない (配線が切れている)"
        );
        for (t, (a, b)) in r_off.iter().zip(r_on.iter()).enumerate() {
            assert_eq!(a.data.len(), b.data.len());
            for i in 0..a.data.len() {
                assert_eq!(
                    a.data[i].to_bits(),
                    b.data[i].to_bits(),
                    "トークン {t} 要素 {i}: off={} on={}",
                    a.data[i],
                    b.data[i]
                );
            }
        }

        // 3. 実ファイルの型が BF16 でも配線が同じように生きること。
        let pw16 = PleW {
            key: bf16_linear(e, hc_dim, 0.1),
            value: bf16_linear(e, e, 0.2),
            norm_key: vec![1.0; hc_dim],
            norm_query: vec![1.0; hc_dim],
            norm_conv: vec![1.0; hc_dim],
            conv1d: vec![0.25; 2 * hc_dim],
            conv1d_t: vec![0.25; 2 * hc_dim],
            kern: 2,
        };
        set_pf182_for_test(PF182_GATE_PLE_F32, Some(false));
        let mut r16_off = mk_res(&p);
        let mut hist2b = vec![0.0f32; hc_dim];
        crate::cuda::f32_gemm_batch_attempts_reset();
        ple_inject_batch(&p, &pw16, &rows, &mut r16_off, &hp, &mut hist2b, None)
            .expect("ple bf16 off");
        assert_eq!(crate::cuda::f32_gemm_batch_attempts(), 0);
        set_pf182_for_test(PF182_GATE_PLE_F32, Some(true));
        let mut r16_on = mk_res(&p);
        let mut hist3 = vec![0.0f32; hc_dim];
        crate::cuda::f32_gemm_batch_attempts_reset();
        ple_inject_batch(&p, &pw16, &rows, &mut r16_on, &hp, &mut hist3, None)
            .expect("ple bf16 on");
        assert_eq!(
            crate::cuda::f32_gemm_batch_attempts(),
            2,
            "BF16 の PLE で GPU 入口が key/value の 2 回でない"
        );
        for (t, (a, b)) in r16_off.iter().zip(r16_on.iter()).enumerate() {
            for i in 0..a.data.len() {
                assert_eq!(
                    a.data[i].to_bits(),
                    b.data[i].to_bits(),
                    "BF16 トークン {t} 要素 {i}: off={} on={}",
                    a.data[i],
                    b.data[i]
                );
            }
        }
        set_pf182_for_test(PF182_GATE_PLE_F32, None);
    }

    #[test]
    fn expert_cache_path_is_bit_identical_to_mmap() {
        let _g = moe_test_guard();
        let dir = TempDir::new("bit");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let lw = cache_test_layer(&f, &p);
        let x = cache_test_x(p.n_embd);

        // 1. 従来経路 (mmap + prefetch)
        let want = moe_ffn_quant(&p, &lw, &x, None).expect("mmap 経路");
        assert!(want.iter().all(|v| v.is_finite()), "mmap 経路に NaN/inf");
        assert!(want.iter().any(|v| *v != 0.0), "mmap 経路が全 0");

        // 2. キャッシュ経路 (2 スロット)
        let lens = expert_lens(&lw);
        let set =
            ExpertCacheSet::new_with_slots([2, 2, 2], lens, &[path.clone()], 2).expect("cache set");
        assert_eq!(set.slots(ExpertMatrix::Gate), 2, "スロット数が 2 でない");
        let par0 = PAR_CONTRIB_CALLS.load(std::sync::atomic::Ordering::Relaxed);
        let loop0 = EXPERT_LOOP_REGIONS.load(std::sync::atomic::Ordering::Relaxed);
        let got = moe_ffn_quant(&p, &lw, &x, Some(&set)).expect("cache 経路");
        assert_bitwise_eq(&want, &got, "mmap vs cache");
        // P38: キャッシュ経路でも expert 並列の分岐が発動している。旧実装は
        // `parallel = par && cache.is_none()` で、キャッシュを有効にすると
        // 10 expert が逐次に落ちていた (並列化の喪失が cache 大敗の主因)。
        // P64: `DENNO_EXPERT_LOOP=1` のときは新経路 (1 層 1 領域) が動くので、
        // どちらかが発動していればよい。
        #[cfg(feature = "par")]
        {
            let jobs = PAR_CONTRIB_CALLS.load(std::sync::atomic::Ordering::Relaxed) - par0;
            let regions = EXPERT_LOOP_REGIONS.load(std::sync::atomic::Ordering::Relaxed) - loop0;
            assert!(
                jobs == p.n_expert_used || regions == 1,
                "キャッシュ経路が並列分岐にも新経路にも入っていない (jobs={jobs}, regions={regions})"
            );
        }

        // 3. 初回は全ミス (top-2 なので行列ごとに 2 回読む)
        for (k, &m) in ExpertMatrix::ALL.iter().enumerate() {
            let s = set.stats(m);
            assert_eq!(s.hits, 0, "{m:?}: 初回でヒットした");
            assert_eq!(s.misses, 2, "{m:?}: 初回のミス数");
            assert_eq!(s.bytes_read, 2 * lens[k] as u64, "{m:?}: 読んだバイト数");
            assert_eq!(s.evictions, 0, "{m:?}: 2 スロットに top-2 で追い出しは起きない");
        }

        // 4. 同じ x をもう一度 → 同じ expert が選ばれるのでヒット (ミスは増えない)
        let before: Vec<u64> = ExpertMatrix::ALL.iter().map(|&m| set.stats(m).misses).collect();
        let again = moe_ffn_quant(&p, &lw, &x, Some(&set)).expect("cache 経路 2 回目");
        assert_bitwise_eq(&want, &again, "mmap vs cache (2 回目)");
        for (k, &m) in ExpertMatrix::ALL.iter().enumerate() {
            let s = set.stats(m);
            assert_eq!(s.misses, before[k], "{m:?}: 同じ expert なのにミスが増えた");
            assert_eq!(s.hits, 2, "{m:?}: 2 回目でヒットしていない");
        }
        assert_eq!(set.hit_rate(ExpertMatrix::Gate), 0.5, "ヒット率");
    }

    /// task 260: ホスト RAM tier のミス計算が mmap とビット一致し、1 回目で
    /// 埋まり 2 回目はヒットすること (分岐が発火した回数を数える)。
    #[cfg(feature = "cuda")]
    #[test]
    fn task260_ram_tier_miss_path_is_bit_identical_and_fills_on_use() {
        let _g = moe_test_guard();
        struct Clear;
        impl Drop for Clear {
            fn drop(&mut self) {
                crate::model::expert_ram::install(None);
            }
        }
        let _c = Clear;
        crate::model::expert_ram::install(None);

        let dir = TempDir::new("ram260");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let lw = cache_test_layer(&f, &p);
        let x = cache_test_x(p.n_embd);
        let (e, ff) = (p.n_embd, p.n_ff_exp);
        let sel: Vec<usize> = (0..p.n_expert_used).collect();

        // 1. tier 無し (mmap) の寄与
        let mut want = vec![0.0f32; sel.len() * e];
        expert_miss_contribs_into(&lw, &sel, &x, e, ff, &mut want).expect("mmap 経路");
        assert!(want.iter().all(|v| v.is_finite()), "mmap 経路に NaN/inf");

        // 2. tier を top-k がちょうど入る予算で作る
        let lens = expert_lens(&lw);
        let n_ex = p.n_expert;
        let specs = [
            RamSpec {
                slot_bytes: lens[0],
                total_bytes: lens[0] as u64 * n_ex as u64,
                n_expert: n_ex,
            },
            RamSpec {
                slot_bytes: lens[1],
                total_bytes: lens[1] as u64 * n_ex as u64,
                n_expert: n_ex,
            },
            RamSpec {
                slot_bytes: lens[2],
                total_bytes: lens[2] as u64 * n_ex as u64,
                n_expert: n_ex,
            },
        ];
        let per: u64 = lens.iter().map(|&b| b as u64).sum();
        let set = std::sync::Arc::new(
            ExpertCacheSet::new_ram(per * sel.len() as u64, specs, &[path.clone()], 0, false)
                .expect("ram set"),
        );
        assert_eq!(
            set.slots(ExpertMatrix::Gate),
            sel.len(),
            "予算からスロット数が決まっていない"
        );
        crate::model::expert_ram::install(Some(std::sync::Arc::clone(&set)));

        // 2. tier が空ならミスは mmap へ落ちる (pread して埋めない)。
        //    これが lab t260a/b の教訓: decode のミスで埋めると 3 倍遅くなる。
        let mut got = vec![0.0f32; sel.len() * e];
        expert_miss_contribs_into(&lw, &sel, &x, e, ff, &mut got).expect("ram 1 回目");
        assert_bitwise_eq(&want, &got, "mmap vs ram tier (空の tier)");
        assert!(set.peek_only(), "RAM tier はヒットだけを引くモードのはず");
        assert_eq!(set.used_experts(), 0, "decode のミスで tier を埋めた");
        for &m in ExpertMatrix::ALL.iter() {
            let s = set.stats(m);
            assert_eq!(s.misses, 0, "{m:?}: peek なのにミスを数えた");
            assert_eq!(s.bytes_read, 0, "{m:?}: peek なのに pread した");
        }

        // 3. 明示的に埋める (起動時の充填と同じ経路) → 2 回目はヒット。
        for &ei in sel.iter() {
            assert!(
                ram_regions(&lw, ei, true).is_some(),
                "tier へ充填できない (expert {ei})"
            );
        }
        let reads: Vec<u64> = ExpertMatrix::ALL
            .iter()
            .map(|&m| set.stats(m).bytes_read)
            .collect();
        let mut again = vec![0.0f32; sel.len() * e];
        expert_miss_contribs_into(&lw, &sel, &x, e, ff, &mut again).expect("ram 2 回目");
        assert_bitwise_eq(&want, &again, "mmap vs ram tier (充填後)");
        for (k, &m) in ExpertMatrix::ALL.iter().enumerate() {
            let s = set.stats(m);
            assert_eq!(s.hits, sel.len() as u64, "{m:?}: 充填後にヒットしていない");
            assert_eq!(s.bytes_read, reads[k], "{m:?}: ヒットなのに pread した");
        }
        // GPU の無いテストでは pinned は取れない = ヒープで動く
        assert_eq!(set.pinned_bytes(), 0, "GPU 無しで pinned が取れた");
        crate::model::expert_ram::install(None);
    }

    #[test]
    fn one_slot_cache_keeps_evicting() {
        // P74: `moe_ffn_quant` を通るので、他の MoE テストと同じロックを取る。
        // 取らないと、テレメトリを ON にしている別テストの窓で RouterTopK の
        // フレームが共有チャネルに混ざる (実測: -j4 で
        // `router_topk_frames_are_arch_gated_and_last_token_only` が落ちた)。
        let _g = moe_test_guard();
        let dir = TempDir::new("one");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let lw = cache_test_layer(&f, &p);
        let x = cache_test_x(p.n_embd);
        let lens = expert_lens(&lw);

        let mut set =
            ExpertCacheSet::new_with_slots([1, 1, 1], lens, &[path.clone()], 1).expect("cache set");
        for m in ExpertMatrix::ALL {
            assert_eq!(set.slots(m), 1, "{m:?}: スロット数が 1 でない");
        }

        // 1 スロットで top-2 → 1 回の forward で必ず追い出しが起きる
        moe_ffn_quant(&p, &lw, &x, Some(&set)).expect("1 回目");
        let first: Vec<(u64, u64)> = ExpertMatrix::ALL
            .iter()
            .map(|&m| (set.stats(m).misses, set.stats(m).evictions))
            .collect();
        for (k, &m) in ExpertMatrix::ALL.iter().enumerate() {
            let (misses, evictions) = first[k];
            assert_eq!(misses, 2, "{m:?}: 1 回目のミス数");
            assert_eq!(evictions, 1, "{m:?}: 1 回目の追い出し数");
        }

        // 2 回目も同じ expert を引くが、1 スロットなので前回の 2 個目しか残っていない
        // → 少なくとも 1 回は必ずミスし、追い出しが増える
        moe_ffn_quant(&p, &lw, &x, Some(&set)).expect("2 回目");
        for (k, &m) in ExpertMatrix::ALL.iter().enumerate() {
            let s = set.stats(m);
            assert!(s.misses > first[k].0, "{m:?}: 2 回目でミスが増えていない");
            assert!(s.evictions > first[k].1, "{m:?}: 2 回目で追い出しが増えていない");
        }
    }

    #[test]
    fn zero_slots_still_returns_mmap_bytes() {
        let _g = moe_test_guard();
        let dir = TempDir::new("zero");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let lw = cache_test_layer(&f, &p);
        let x = cache_test_x(p.n_embd);
        let lens = expert_lens(&lw);

        let want = moe_ffn_quant(&p, &lw, &x, None).expect("mmap 経路");
        let mut set =
            ExpertCacheSet::new_with_slots([0, 0, 0], lens, &[path.clone()], 0).expect("cache set");
        for m in ExpertMatrix::ALL {
            assert_eq!(set.slots(m), 0, "{m:?}: スロット数が 0 でない");
        }
        let got = moe_ffn_quant(&p, &lw, &x, Some(&set)).expect("slots=0 経路");
        assert_bitwise_eq(&want, &got, "mmap vs slots=0");
        for (k, &m) in ExpertMatrix::ALL.iter().enumerate() {
            let s = set.stats(m);
            assert_eq!((s.hits, s.misses), (0, 2), "{m:?}: slots=0 の統計");
            assert_eq!(s.bytes_read, 2 * lens[k] as u64, "{m:?}: 読んだバイト数");
        }
    }

    #[test]
    fn build_expert_caches_takes_mb_and_per_matrix_max() {
        let dir = TempDir::new("build");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let layers = vec![cache_test_layer(&f, &p)];
        let lens = expert_lens(&layers[0]);

        // MB = 0 は「無効」= 構築しない (mmap 経路のまま)
        assert!(
            build_expert_caches_with_mb(0, &f, &layers)
                .expect("mb=0")
                .is_none(),
            "mb=0 でキャッシュを組んだ"
        );

        // MB = 4: 行列ごとに 4 MiB / slot_bytes (切り捨て)
        let set = build_expert_caches_with_mb(4, &f, &layers)
            .expect("mb=4")
            .expect("キャッシュが組まれていない");
        assert_eq!(set.mb(), 4);
        for (k, &m) in ExpertMatrix::ALL.iter().enumerate() {
            assert_eq!(set.slot_bytes(m), lens[k], "{m:?}: slot_bytes");
            assert_eq!(
                set.slots(m),
                4 * 1024 * 1024 / lens[k],
                "{m:?}: スロット数が MB の式と違う"
            );
            assert_eq!(set.stats(m), CacheStats::default(), "{m:?}: 初期統計");
        }
    }

    #[test]
    fn slot_bytes_below_the_largest_expert_is_an_error() {
        // P74: `moe_ffn_quant` を通るので他の MoE テストと同じロックを取る。
        let _g = moe_test_guard();
        // 静かに切り詰めないこと (Q5_1 down の 1 expert より小さいスロット)
        let dir = TempDir::new("small");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let lw = cache_test_layer(&f, &p);
        let x = cache_test_x(p.n_embd);
        let mut lens = expert_lens(&lw);
        lens[2] -= 1; // down だけ 1 バイト足りない

        let mut set =
            ExpertCacheSet::new_with_slots([2, 2, 2], lens, &[path.clone()], 2).expect("cache set");
        let err = moe_ffn_quant(&p, &lw, &x, Some(&set)).expect_err("切り詰めて通った");
        assert!(
            err.to_string().contains("1 スロット"),
            "エラーがスロット超過でない: {err}"
        );
    }

    // -----------------------------------------------------------------
    // H2: prefill のバッチ化 (層優先) — 逐次経路とのビット一致
    //
    // 合成モデルは**目録 (`expected_tensors`) から組む**: テンソル名・形状・
    // 量子型をここに写さないので、目録が変わればテストも自動で追随する。
    // -----------------------------------------------------------------

include!("tests_h2.rs");
