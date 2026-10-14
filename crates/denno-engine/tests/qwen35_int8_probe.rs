//! per-tensor int8 量子化誤差の調査 (実 GGUF・ignored)。
//! Linear::quantize (per-row scale = max|w|/127) を各テンソルに適用し
//! 相対 Frobenius 誤差を報告する。GDN 経路のどのテンソルが悪いかを特定。

use denno_engine::gguf::GgufFile;
use denno_engine::quant::Linear;

#[test]
#[ignore]
fn qwen35_per_tensor_int8_error() {
    let path = std::env::var("GGUF_PATH").expect("GGUF_PATH を設定してください");
    let f = GgufFile::open(std::path::Path::new(&path)).expect("GGUF 開失敗");

    // Linear 化されるテンソル名のパターンを列挙
    let mut names: Vec<&String> = f
        .tensors
        .keys()
        .filter(|n| {
            let s = n.as_str();
            let base = s.split('.').nth(2).unwrap_or("");
            matches!(
                base,
                "attn_qkv"
                    | "attn_gate"
                    | "ssm_beta"
                    | "ssm_alpha"
                    | "ssm_out"
                    | "attn_q"
                    | "attn_k"
                    | "attn_v"
                    | "attn_output"
                    | "ffn_gate"
                    | "ffn_up"
                    | "ffn_down"
            ) && s.ends_with(".weight")
        })
        .collect();
    names.sort();

    // 形状はテンソル情報から取得 (GGUF ne = [ne0, ne1] = [in, out])
    let mut worst: Vec<(f64, String)> = Vec::new();
    for name in &names {
        let info = &f.tensors[*name];
        let ne = &info.dims;
        let (k, n) = (ne[0] as usize, ne[1] as usize);
        let w = f.dequant_tensor(name).expect("dequant 失敗");
        assert_eq!(w.len(), k * n, "{name}: {} != {}x{}", w.len(), k, n);
        let q = denno_engine::quant::Quant8Linear::quantize(&w, k, n);
        // 相対 Frobenius 誤差
        let mut num = 0f64;
        let mut den = 0f64;
        for j in 0..n {
            let s = q.scale[j];
            for i in 0..k {
                let wv = w[j * k + i] as f64;
                let qv = q.w[j * k + i] as f64 * s as f64;
                num += (wv - qv) * (wv - qv);
                den += wv * wv;
            }
        }
        let rel = if den > 0.0 { (num / den).sqrt() } else { 0.0 };
        worst.push((rel, name.to_string()));
    }
    worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    eprintln!("=== int8 per-row 相対誤差 (降順・上位 15) ===");
    for (rel, name) in worst.iter().take(15) {
        eprintln!("{rel:.6}  {name}");
    }
    eprintln!("=== 最小 5 ===");
    for (rel, name) in worst.iter().rev().take(5) {
        eprintln!("{rel:.6}  {name}");
    }
}
