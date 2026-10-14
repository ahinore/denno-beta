//! 実 GGUF (Qwen3.5-0.8B) の GDN テンソルを dequant して gdn.rs forward に通す統合テスト。
//! 環境変数: GGUF_PATH (必須・ignored)。
//!
//! 検証内容:
//! 1. blk.0 が GDN 層 (ssm_* テンソル存在)・全必要テンソルが揃っている
//! 2. dequant → GDN forward が有限値で・出力スケールが妥当
//! 3. 決定論 (2 回実行で bit 一致)
//! 4. prefill と逐次の bit 一致 (実重み)
//! 5. 0 入力継続で状態が減衰 (decay の符号確認)

use denno_engine::gguf::GgufFile;
use denno_engine::gdn::{GdnParams, GdnState, GdnWeights};

fn lcg_f32(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1103515245).wrapping_add(12345);
    (((*state >> 16) & 0xff) as f32) / 255.0 - 0.5
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
#[ignore]
fn gdn_layer0_real_gguf_forward() {
    // CI (-- --ignored) ではモデルが無いのでスキップ(パニックにしない)
    let Ok(path) = std::env::var("GGUF_PATH") else {
        eprintln!("skip: GGUF_PATH 未設定");
        return;
    };
    let f = GgufFile::open(&path).unwrap();

    // 1. 必要テンソルの存在確認
    let names = [
        "blk.0.attn_qkv.weight",
        "blk.0.attn_gate.weight",
        "blk.0.ssm_alpha.weight",
        "blk.0.ssm_beta.weight",
        "blk.0.ssm_a",
        "blk.0.ssm_dt.bias",
        "blk.0.ssm_conv1d.weight",
        "blk.0.ssm_norm.weight",
        "blk.0.ssm_out.weight",
    ];
    for n in names {
        assert!(f.tensor(n).is_some(), "{} が無い (blk.0 は GDN 層か?)", n);
    }

    // 2. GDN 重みを構築 (GGUF ne=[in,out] 行優先 → w[out*in+j] でそのまま使える)
    let p = GdnParams::qwen35_08b();
    let w_qkv = f.dequant_tensor("blk.0.attn_qkv.weight").unwrap();
    assert_eq!(w_qkv.len(), p.qkv_dim() * p.n_embd, "attn_qkv 要素数");
    let w_gate = f.dequant_tensor("blk.0.attn_gate.weight").unwrap();
    assert_eq!(w_gate.len(), p.value_dim() * p.n_embd);
    let w_alpha = f.dequant_tensor("blk.0.ssm_alpha.weight").unwrap();
    assert_eq!(w_alpha.len(), p.n_v_heads * p.n_embd);
    let w_beta = f.dequant_tensor("blk.0.ssm_beta.weight").unwrap();
    assert_eq!(w_beta.len(), p.n_v_heads * p.n_embd);
    let conv1d = f.dequant_tensor("blk.0.ssm_conv1d.weight").unwrap();
    assert_eq!(conv1d.len(), p.conv_dim() * p.d_conv);
    let w_out = f.dequant_tensor("blk.0.ssm_out.weight").unwrap();
    assert_eq!(w_out.len(), p.n_embd * p.value_dim());

    let ssm_a = f.dequant_tensor("blk.0.ssm_a").unwrap();
    assert_eq!(ssm_a.len(), p.n_v_heads, "ssm_a は f32 [n_v_heads]");
    assert!(ssm_a.iter().all(|v| *v < 0.0), "ssm_a は負値 (-exp(A_log))");
    let ssm_dt = f.dequant_tensor("blk.0.ssm_dt.bias").unwrap();
    assert_eq!(ssm_dt.len(), p.n_v_heads);
    let ssm_norm = f.dequant_tensor("blk.0.ssm_norm.weight").unwrap();
    assert_eq!(ssm_norm.len(), p.head_v_dim);

    let w = GdnWeights {
        w_qkv,
        w_gate,
        w_beta,
        w_alpha,
        conv1d,
        ssm_a,
        ssm_dt,
        ssm_norm,
        w_out,
    };

    // 3. 決定論的入力 (LCG) で 8 トークン forward
    let t_len = 8;
    let mut st1 = GdnState::new(&p);
    let mut rng: u32 = 0x33557799;
    let xs: Vec<Vec<f32>> = (0..t_len)
        .map(|_| (0..p.n_embd).map(|_| lcg_f32(&mut rng)).collect())
        .collect();
    let out1 = p.forward(&w, &xs, &mut st1);

    // 有限値・スケール妥当性
    for (t, o) in out1.iter().enumerate() {
        assert!(o.iter().all(|v| v.is_finite()), "t={t} 非有限");
        let mx = o.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        assert!(mx > 1e-6 && mx < 1e3, "t={t}: 出力スケール異常 mx={mx}");
        eprintln!("t={t}: maxabs={mx:.6} out[0..3]={:.3?}", &o[0..3]);
    }

    // 4. 決定論: 同一入力を独立 state で再実行 → bit 一致
    let mut st2 = GdnState::new(&p);
    let out2 = p.forward(&w, &xs, &mut st2);
    for (a, b) in out1.iter().zip(out2.iter()) {
        assert_eq!(bits(a), bits(b), "決定論違反");
    }
    assert_eq!(bits(&st1.ssm), bits(&st2.ssm), "SSM 状態が非決定論");

    // 5. prefill vs 逐次 (実重み)
    let mut st3 = GdnState::new(&p);
    for (t, x) in xs.iter().enumerate() {
        let y = p.forward_token(&w, x, &mut st3);
        assert_eq!(bits(&y), bits(&out1[t]), "t={t}: prefill と逐次不一致 (実重み)");
    }

    // 6. decay の符号: g = ssm_a * softplus(dt_bias) は全 head で負 (exp(g) < 1)
    for h in 0..p.n_v_heads {
        let g = w.ssm_a[h] * (1.0 + w.ssm_dt[h].exp()).ln();
        assert!(g < 0.0, "h={h}: decay g={g} が非負 (ssm_a/softplus の解釈ミス)");
        eprintln!("h={h}: ssm_a={:.4} dt={:.4} g={:.6} exp(g)={:.6}", w.ssm_a[h], w.ssm_dt[h], g, g.exp());
    }
}
