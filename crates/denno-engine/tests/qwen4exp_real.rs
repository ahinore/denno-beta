//! Flash-Next の**実データ**で GPU GEMV のビット一致を見る。
//!
//! 合成した重みでの一致は `moe_gpu` / `gguf_linear_gpu` が見ている。ここは実ファイルの
//! 量子化ブロック (Q4_K / Q5_K / Q5_1 / Q8_0) をそのまま使い、量子化の端 (符号・最大値・
//! 非正規化) でも CPU と GPU が同じ値を返すかを見る。
//!
//! モデル (111 GB) は GPU コンテナに mount されていないため、必要な 1 expert ぶんの
//! 生バイトを事前に切り出しておく (`q4_probe.py`)。probe ディレクトリの `manifest.tsv` を
//! 読み、各行 (file, ty, k, n, e) について CPU 参照と GPU のビット一致を確かめる。
//!
//! 実行:
//!   DENNO_Q4_PROBE=<probe ディレクトリ> DENNO_CUDA=1 \
//!     cargo test --release --features cuda --test qwen4exp_real -- --ignored --test-threads=1 --nocapture
//!
//! probe が無ければ静かに skip する (CI で落とさない)。

#[cfg(feature = "cuda")]
mod real {
    use denno_engine::kquant::GgmlType;
    use denno_engine::model::moe_expert::expert_gemv;
    use std::path::PathBuf;

    fn ty_of(name: &str) -> Option<GgmlType> {
        Some(match name {
            "Q8_0" => GgmlType::Q8_0,
            "Q5_1" => GgmlType::Q5_1,
            "Q4_K" => GgmlType::Q4K,
            "Q5_K" => GgmlType::Q5K,
            _ => return None,
        })
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

    /// ファイル名と expert 番号から安定な uid を作る (FNV-1a)。
    fn uid_of(name: &str, e: usize) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in name
            .as_bytes()
            .iter()
            .copied()
            .chain([0u8])
            .chain((e as u32).to_le_bytes())
        {
            h ^= b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        h
    }

    fn first_diff(a: &[f32], b: &[f32]) -> Option<usize> {
        if a.len() != b.len() {
            return Some(usize::MAX);
        }
        (0..a.len()).find(|&i| a[i].to_bits() != b[i].to_bits())
    }

    #[test]
    #[ignore]
    fn real_gguf_gemv_matches_cpu() {
        let Ok(dir) = std::env::var("DENNO_Q4_PROBE") else {
            eprintln!("skip: DENNO_Q4_PROBE 未設定");
            return;
        };
        let dir = PathBuf::from(dir);
        let man = dir.join("manifest.tsv");
        let Ok(text) = std::fs::read_to_string(&man) else {
            eprintln!("skip: manifest が無い ({})", man.display());
            return;
        };
        let mut n_case = 0usize;
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let f: Vec<&str> = line.split('\t').collect();
            assert_eq!(f.len(), 5, "manifest の列数: {line}");
            let file = dir.join(f[0]);
            let Some(ty) = ty_of(f[1]) else {
                eprintln!("skip: 未対応の型 {}", f[1]);
                continue;
            };
            let k: usize = f[2].parse().expect("k");
            let n: usize = f[3].parse().expect("n");
            let e: usize = f[4].parse().expect("e");
            let bytes = std::fs::read(&file).expect("probe の読み込み");
            assert_eq!(bytes.len() % n, 0, "{file:?}: 行数で割り切れない");

            let x = xvec(k, 0x5d_e1_2b_9c_11_33_77_55);
            let mut want = vec![0f32; n];
            expert_gemv(ty, &bytes, k, n, &x, &mut want).expect("CPU 参照");

            let uid = uid_of(f[0], e);
            for pass in 0..2 {
                let mut got = vec![0f32; n];
                let used = denno_engine::cuda::maybe_gpu_gguf_gemv_into(
                    uid, ty, &bytes, k, n, &x, &mut got,
                );
                assert!(used, "{}: GPU 経路が使われなかった", f[0]);
                match first_diff(&got, &want) {
                    None => {}
                    Some(i) => panic!(
                        "{} pass {pass}: 行 {i} で不一致 gpu={} cpu={}",
                        f[0], got[i], want[i]
                    ),
                }
            }
            eprintln!("{} ({ty:?}, {k}x{n}): 2 パスとも {n} 行ビット一致", f[0]);
            n_case += 1;
        }
        assert!(n_case > 0, "probe のケースが 0 件");
        eprintln!("{n_case} 件すべて一致");
    }
}
