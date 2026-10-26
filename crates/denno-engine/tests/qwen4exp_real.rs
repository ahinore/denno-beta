//! Flash-Next の**実データ**で GPU GEMV のビット一致を見る。
//!
//! 合成した重みでの一致は `moe_gpu` / `gguf_linear_gpu` が見ている。ここは
//! 実ファイルの量子化ブロック (Q4_K / Q5_1 / Q8_0) をそのまま使って、
//! 「量子化の端ケース (符号・最大値・非正規化)」で CPU と GPU が同じ値を返すかを見る。
//!
//! 実行:
//!   DENNO_Q4_DIR=<分割 GGUF のあるディレクトリ> DENNO_CUDA=1 \
//!     cargo test --release --features cuda --test qwen4exp_real -- --ignored --test-threads=1 --nocapture
//!
//! ディレクトリが無ければ静かに skip する (CI で落とさない)。

#[cfg(feature = "cuda")]
mod real {
    use denno_engine::gguf::GgufFile;
    use denno_engine::kquant::GgmlType;
    use denno_engine::model::linear_gguf::GgufLinear;
    use denno_engine::model::moe_expert::{expert_gemv, ExpertSlice};
    use std::path::PathBuf;

    fn parts() -> Option<Vec<PathBuf>> {
        let dir = std::env::var("DENNO_Q4_DIR").ok()?;
        let mut v: Vec<PathBuf> = std::fs::read_dir(&dir)
            .ok()?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map(|x| x == "gguf").unwrap_or(false))
            .collect();
        if v.is_empty() {
            return None;
        }
        v.sort();
        Some(v)
    }

    fn xvec(k: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..k)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32) / 16_777_216.0 - 0.5
            })
            .collect()
    }

    /// 名前と expert 番号から安定な uid を作る (FNV-1a)。
    fn uid_of(name: &str, e: usize) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in name.as_bytes().iter().chain(&[0u8]).chain(&(e as u32).to_le_bytes()) {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        h
    }

    fn bits_eq(a: &[f32], b: &[f32]) -> Option<usize> {
        if a.len() != b.len() {
            return Some(usize::MAX);
        }
        (0..a.len()).find(|&i| a[i].to_bits() != b[i].to_bits())
    }

    fn check_expert(f: &GgufFile, name: &str, n_in: usize, n_out: usize, n_expert: usize, e: usize) {
        let info = f.tensor(name).expect("テンソルが無い").clone();
        let ty = info.ty();
        let all = f.tensor_bytes(name).expect("tensor_bytes");
        let sl = ExpertSlice::new(ty, n_in, n_out, n_expert, e).expect("ExpertSlice");
        let bytes = &all[sl.off..sl.end()];

        let x = xvec(n_in, 0x5d_e1_2b_9c_11_33_77_55);
        let mut want = vec![0f32; n_out];
        expert_gemv(ty, bytes, n_in, n_out, &x, &mut want).expect("CPU 参照");

        let uid = uid_of(name, e);
        let mut got = vec![0f32; n_out];
        let used = denno_engine::cuda::maybe_gpu_gguf_gemv_into(uid, ty, bytes, n_in, n_out, &x, &mut got);
        assert!(used, "{name} expert {e}: GPU 経路が使われなかった");
        match bits_eq(&got, &want) {
            None => eprintln!("{name} expert {e} ({ty:?}): {n_out} 行ビット一致"),
            Some(i) => panic!(
                "{name} expert {e} ({ty:?}): 行 {i} で不一致 gpu={} cpu={}",
                got[i], want[i]
            ),
        }
        // 2 回目は重みが常駐している (転送なし) 経路
        let mut got2 = vec![0f32; n_out];
        assert!(denno_engine::cuda::maybe_gpu_gguf_gemv_into(
            uid, ty, bytes, n_in, n_out, &x, &mut got2
        ));
        assert_eq!(bits_eq(&got2, &want), None, "{name} expert {e}: 2 回目で不一致");
        eprintln!("{name} expert {e}: 常駐パスも一致");
    }

    fn check_linear(f: &GgufFile, name: &str, k: usize, n: usize) {
        let info = f.tensor(name).expect("テンソルが無い").clone();
        let ty = info.ty();
        let bytes = f.tensor_bytes(name).expect("tensor_bytes");
        let lin = GgufLinear::from_slice(ty, k, n, bytes).expect("GgufLinear");
        let x = xvec(k, 0x1f_2e_3d_4c_5b_6a_79_88);
        let mut want = vec![0f32; n];
        lin.gemv(&x, &mut want).expect("CPU 参照");
        let mut got = vec![0f32; n];
        let used = denno_engine::cuda::maybe_gpu_gguf_gemv_into(
            lin.uid(),
            ty,
            lin.blocks().as_slice(),
            k,
            n,
            &x,
            &mut got,
        );
        assert!(used, "{name}: GPU 経路が使われなかった");
        match bits_eq(&got, &want) {
            None => eprintln!("{name} ({ty:?}): {n} 行ビット一致"),
            Some(i) => panic!("{name}: 行 {i} で不一致 gpu={} cpu={}", got[i], want[i]),
        }
    }

    #[test]
    #[ignore]
    fn real_expert_q4k_gate_blk0() {
        let Some(p) = parts() else {
            eprintln!("skip: DENNO_Q4_DIR が無い");
            return;
        };
        let f = GgufFile::open_split(&p).expect("open_split");
        // blk.0.ffn_gate_exps: [2560, 640, 512]
        check_expert(&f, "blk.0.ffn_gate_exps.weight", 2560, 640, 512, 0);
        check_expert(&f, "blk.0.ffn_gate_exps.weight", 2560, 640, 512, 511);
    }

    #[test]
    #[ignore]
    fn real_expert_q5_1_down_blk0() {
        let Some(p) = parts() else {
            eprintln!("skip: DENNO_Q4_DIR が無い");
            return;
        };
        let f = GgufFile::open_split(&p).expect("open_split");
        // blk.0.ffn_down_exps: [640, 2560, 512]
        check_expert(&f, "blk.0.ffn_down_exps.weight", 640, 2560, 512, 3);
    }

    #[test]
    #[ignore]
    fn real_linear_q8_0_shexp_blk0() {
        let Some(p) = parts() else {
            eprintln!("skip: DENNO_Q4_DIR が無い");
            return;
        };
        let f = GgufFile::open_split(&p).expect("open_split");
        // blk.0.ffn_up_shexp: [2560, 640] Q8_0
        check_linear(&f, "blk.0.ffn_up_shexp.weight", 2560, 640);
    }
}
