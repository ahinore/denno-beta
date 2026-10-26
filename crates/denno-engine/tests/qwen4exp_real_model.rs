//! 実モデル（Qwen3.8-Flash-Next UD-Q4_K_XL）の目録検証。
//!
//! `DENNO_Q4_DIR` に 4 分割 GGUF のディレクトリを渡すと、`qwen4exp::validate` を
//! 実際の 1,224 テンソルに対して走らせ、諸元と本数を固定値と突き合わせる。
//! 環境変数が無ければ静かに skip する（GPU も重みのロードもしない。ヘッダだけ読む）。

use std::path::PathBuf;

use denno_engine::gguf::GgufFile;
use denno_engine::model::qwen4exp;

fn parts() -> Option<Vec<PathBuf>> {
    let dir = std::env::var("DENNO_Q4_DIR").ok()?;
    let dir = PathBuf::from(dir);
    let mut v: Vec<PathBuf> = std::fs::read_dir(&dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().map(|x| x == "gguf").unwrap_or(false)
                && p.file_name()
                    .and_then(|s| s.to_str())
                    .map(|s| s.contains("of-00004"))
                    .unwrap_or(false)
        })
        .collect();
    v.sort();
    if v.len() != 4 {
        return None;
    }
    Some(v)
}

#[test]
fn real_model_manifest_matches() {
    let parts = match parts() {
        Some(p) => p,
        None => {
            eprintln!("DENNO_Q4_DIR が無いので skip");
            return;
        }
    };

    let f = GgufFile::open_split(&parts).expect("分割 GGUF を開けない");
    let p = qwen4exp::validate(&f).expect("目録検証に失敗した");

    assert_eq!(p.n_layer, 48, "block_count");
    assert_eq!(p.n_embd, 2560, "embedding_length");
    assert_eq!(p.n_head, 24, "head_count");
    assert_eq!(p.n_head_kv, 2, "head_count_kv");
    assert_eq!(p.n_embd_head_k, 256, "key_length");
    assert_eq!(p.n_embd_head_v, 256, "value_length");
    assert_eq!(p.n_expert, 512, "expert_count");
    assert_eq!(p.n_expert_used, 10, "expert_used_count");
    assert_eq!(p.n_ff_exp, 640, "expert_ffn_length");
    assert_eq!(p.ssm_d_state, 128, "ssm.state_size");
    assert_eq!(p.ssm_n_group, 16, "ssm.group_count");
    assert_eq!(p.n_hc, 4, "hyper_connection.count");
    assert_eq!(p.hc_low_rank, 320, "hyper_connection.low_rank");
    assert_eq!(p.indexer_n_head, 4, "indexer.head_count");
    assert_eq!(p.indexer_top_k, 2048, "indexer.top_k");
    assert_eq!(p.n_full_attn_layers(), 12, "full attention 層数");
    assert_eq!(p.n_ssm_layers(), 36, "ssm 層数");
    assert_eq!(p.layer_kind(3).unwrap(), qwen4exp::LayerKind::FullAttn);
    assert_eq!(p.layer_kind(2).unwrap(), qwen4exp::LayerKind::Ssm);

    let n = f.tensor_order.len();
    assert_eq!(n, 1224, "テンソル本数");

    let bytes: u64 = f
        .tensor_order
        .iter()
        .filter_map(|t| f.tensors.get(t))
        .map(|t| t.nbytes)
        .sum();
    // 111.32 GB (10^9 基準)。ヘッダの宣言値の合計。
    assert!(
        bytes > 111_000_000_000 && bytes < 112_000_000_000,
        "合計バイト数が想定外: {bytes}"
    );
    eprintln!("tensors={n} bytes={bytes}");
}
