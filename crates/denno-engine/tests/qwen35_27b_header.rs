//! Qwen3.8-27B 実 GGUF のヘッダのみの構造検証 (ignored・CPU・dequant 無し)。
//! 環境変数: QWEN35_27B_DIR (config.json + *.gguf を含むディレクトリ・必須)。
//! 未設定なら graceful skip (CI ではモデルが無いため常時 skip)。
//!
//! 27B は CPU ではフル推論が非現実的なため、実 GGUF を開いてヘッダ (テンソル名・
//! 形状・KV) だけを読み、ローダが想定する構造と一致することを検証する。実機 E2E
//! の前に静的に構造不一致 (out-gate 構造の違い・MTP 層の扱い) を検出するための
//! テスト。2026-10-16 の 27B out-gate 構造調査 (transformers modeling_qwen3_5 +
//! llama.cpp src/models/qwen35.cpp との照合) で確定した期待値を符号化したもの。
//!
//! 検証内容:
//! 1. 層数の決定元: GGUF block_count == config num_hidden_layers + nextn_predict_layers
//!    (27B: 65 = 64 + 1・nextn は MTP 層としてトランクから除外)
//! 2. trunk 層の構成: 48 GDN + 16 full_attention (i%4==3)・config layer_types と一致
//! 3. GDN 層の形状 (blk.0): attn_qkv [e, 2kd+vd]・attn_gate [e, vd]・ssm_* ・
//!    ssm_out [vd, e]
//! 4. full_attention 層の形状 (blk.3): attn_q [e, 2*heads*dh] (head ごと
//!    [q|gate] 融合)・attn_output [heads*dh, e] (非正方形 — 9B は偶然正方形)・
//!    q/k_norm [dh]
//! 5. MTP 層 (blk.64): nextn.eh_proj [2e, e]・enorm/hnorm [e]・ssm_a 無し
//! 6. embedding/untied lm_head: token_embd/output.weight [e, vocab]

use denno_engine::gguf::GgufFile;
use serde_json::Value;

fn kv_u64(f: &GgufFile, key: &str) -> u64 {
    f.get(key)
        .and_then(|v| v.as_u64())
        .unwrap_or_else(|| panic!("KV {key} が無い"))
}

fn ne(t: &denno_engine::gguf::GgufTensorInfo) -> Vec<u64> {
    let last = t.dims.iter().rposition(|d| *d != 1).map_or(0, |p| p + 1);
    t.dims[..last].to_vec()
}

fn expect(f: &GgufFile, name: &str, want: &[u64]) {
    let t = f
        .tensor(name)
        .unwrap_or_else(|| panic!("テンソル {name} が無い"));
    assert_eq!(
        ne(t),
        want,
        "{name}: 形状不一致 (got {:?}, want {want:?})",
        ne(t)
    );
}

#[test]
#[ignore]
fn qwen38_27b_gguf_header_structure() {
    // CI (-- --ignored) ではモデルが無いのでスキップ(パニックにしない)
    let Ok(dir) = std::env::var("QWEN35_27B_DIR") else {
        eprintln!("skip: QWEN35_27B_DIR 未設定");
        return;
    };
    let dir = std::path::PathBuf::from(dir);

    // config (VLM ラッパ text_config を優先 — ローダと同一規約)
    let cfg: Value =
        serde_json::from_reader(std::fs::File::open(dir.join("config.json")).expect("config.json"))
            .expect("config.json パース");
    let tc = cfg.get("text_config").unwrap_or(&cfg);

    let e = tc["hidden_size"].as_u64().expect("hidden_size");
    let heads = tc["num_attention_heads"].as_u64().expect("heads");
    let kvh = tc["num_key_value_heads"].as_u64().expect("kv_heads");
    let dh = tc["head_dim"].as_u64().expect("head_dim");
    let n_layer = tc["num_hidden_layers"].as_u64().expect("n_layer");
    let vocab = tc["vocab_size"].as_u64().expect("vocab_size");
    let nk = tc["linear_num_key_heads"].as_u64().expect("nk");
    let nv = tc["linear_num_value_heads"].as_u64().expect("nv");
    let dk = tc["linear_key_head_dim"].as_u64().expect("dk");
    let dv = tc["linear_value_head_dim"].as_u64().expect("dv");
    let d_conv = tc["linear_conv_kernel_dim"].as_u64().expect("d_conv");

    // GGUF 選択 (ローダ pick_gguf と同一規約: Q8_0 > UD-IQ2_XXS > 辞書順・
    // DENNO_GGUF_FILE で明示指定可)
    let gguf = if let Ok(name) = std::env::var("DENNO_GGUF_FILE") {
        dir.join(name)
    } else {
        let mut cands: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
            .expect("dir")
            .filter_map(|en| {
                let p = en.ok()?.path();
                (p.extension().and_then(|s| s.to_str()) == Some("gguf")).then_some(p)
            })
            .collect();
        assert!(!cands.is_empty(), "GGUF が見つからない: {dir:?}");
        cands.sort();
        cands
            .iter()
            .find(|p| p.to_string_lossy().contains("Q8_0"))
            .or_else(|| cands.iter().find(|p| p.to_string_lossy().contains("UD-IQ2_XXS")))
            .unwrap_or(&cands[0])
            .clone()
    };
    let f = GgufFile::open(&gguf).expect("GGUF を開けない");
    eprintln!(
        "27B GGUF: {:?} ({} tensors) — ヘッダのみ検証 (dequant 無し)",
        gguf,
        f.tensors.len()
    );

    // 1. 層数の決定元: trunk = config num_hidden_layers・超過分 = nextn (MTP)。
    //    ローダはこの整合を検証し nextn 層を読み飛ばす (実装: qwen35.rs load)。
    let arch = f.get("general.architecture").and_then(|v| v.as_str()).expect("arch");
    let bc = kv_u64(&f, &format!("{arch}.block_count"));
    let nextn = kv_u64(&f, &format!("{arch}.nextn_predict_layers"));
    assert_eq!(bc, n_layer + nextn, "block_count != trunk + nextn");
    assert_eq!(nextn, 1, "27B は nextn_predict_layers=1");
    assert_eq!(n_layer, 64, "27B は trunk 64 層");

    // 2. trunk 層の構成 (ssm_a 有無で GDN / full を判別 — ローダと同一規約)
    let mut n_gdn = 0u64;
    for i in 0..n_layer {
        let p = format!("blk.{i}.");
        let is_gdn = f.tensor(&format!("{p}ssm_a")).is_some();
        let want_gdn = tc["layer_types"][i as usize].as_str() == Some("linear_attention");
        assert_eq!(is_gdn, want_gdn, "blk.{i}: 層種別が config layer_types と不一致");
        assert!(
            f.tensor(&format!("{p}attn_norm.weight")).is_some()
                && f.tensor(&format!("{p}post_attention_norm.weight")).is_some()
                && f.tensor(&format!("{p}ffn_gate.weight")).is_some()
                && f.tensor(&format!("{p}ffn_up.weight")).is_some()
                && f.tensor(&format!("{p}ffn_down.weight")).is_some(),
            "blk.{i}: 共通テンソル欠落"
        );
        if is_gdn {
            n_gdn += 1;
        } else {
            // 4. full_attention: head ごと [q|gate] 融合 + 非正方形 attn_output。
            //    out-gate は attention 後 o_proj 前 (sigmoid) — 9B/27B 共通。
            expect(&f, &format!("{p}attn_q.weight"), &[e, 2 * heads * dh]);
            expect(&f, &format!("{p}attn_k.weight"), &[e, kvh * dh]);
            expect(&f, &format!("{p}attn_v.weight"), &[e, kvh * dh]);
            expect(&f, &format!("{p}attn_q_norm.weight"), &[dh]);
            expect(&f, &format!("{p}attn_k_norm.weight"), &[dh]);
            expect(&f, &format!("{p}attn_output.weight"), &[heads * dh, e]);
            assert_ne!(heads * dh, e, "想定では 27B は heads*dh != e (非正方形)");
        }
    }
    assert_eq!(n_gdn, 48, "27B は 48 GDN");
    assert_eq!(n_layer - n_gdn, 16, "27B は 16 full_attention");

    // 3. GDN 層 (blk.0) の形状
    let (kd, vd) = (nk * dk, nv * dv);
    let qkv_dim = 2 * kd + vd;
    expect(&f, "blk.0.attn_qkv.weight", &[e, qkv_dim]);
    expect(&f, "blk.0.attn_gate.weight", &[e, vd]);
    expect(&f, "blk.0.ssm_beta.weight", &[e, nv]);
    expect(&f, "blk.0.ssm_alpha.weight", &[e, nv]);
    expect(&f, "blk.0.ssm_conv1d.weight", &[d_conv, qkv_dim]);
    expect(&f, "blk.0.ssm_a", &[nv]);
    expect(&f, "blk.0.ssm_dt.bias", &[nv]);
    expect(&f, "blk.0.ssm_norm.weight", &[dv]);
    expect(&f, "blk.0.ssm_out.weight", &[vd, e]);

    // 5. MTP (nextn) 層 blk.64: trunk ではない (nextn.eh_proj を持ち ssm_a 無し)。
    //    ローダは block_count 超過分をこの構成でのみ nextn と認めて読み飛ばす。
    let m = format!("blk.{n_layer}.");
    expect(&f, &format!("{m}nextn.eh_proj.weight"), &[2 * e, e]);
    expect(&f, &format!("{m}nextn.enorm.weight"), &[e]);
    expect(&f, &format!("{m}nextn.hnorm.weight"), &[e]);
    assert!(f.tensor(&format!("{m}ssm_a")).is_none(), "MTP 層に ssm_a がある (nextn 判定規約崩れ)");
    assert!(f.tensor(&format!("{m}attn_q.weight")).is_some(), "MTP 層の attention テンソル");

    // 6. embedding / untied lm_head / 最終 norm
    expect(&f, "token_embd.weight", &[e, vocab]);
    expect(&f, "output.weight", &[e, vocab]);
    assert!(
        f.tensor("output.weight").is_some(),
        "27B は untied (output.weight が別テンソル)"
    );
    let norm_name = if f.tensor("output_norm.weight").is_some() {
        "output_norm.weight"
    } else {
        "model.norm.weight"
    };
    expect(&f, norm_name, &[e]);

    eprintln!("27B ヘッダ構造検証 OK: trunk 64 (=48 GDN + 16 full) + MTP 1・vocab={vocab}・e={e}・heads={heads}×dh={dh} (attn_output [{}, {}])", heads*dh, e);
}
