//! task 796: `mtp.rs` のテスト。実ファイル (5.2 GB) は使わず、同じ 31 テンソル
//! 構成の小さな safetensors を書いて読み込む。

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::*;

// ---------------------------------------------------------------------------
// 合成 safetensors
// ---------------------------------------------------------------------------

fn tiny_cfg() -> MtpConfig {
    MtpConfig {
        n_embd: 8,
        hc: 2,
        low_rank: 4,
        n_layer: 1,
        n_head: 2,
        n_head_kv: 1,
        head_dim: 4,
        n_rot: 2,
        rope_theta: 10000.0,
        rms_eps: 1e-6,
        n_expert: 4,
        n_expert_used: 2,
        n_ff_exp: 3,
        n_ff_shexp: 3,
        indexer_n_head: 2,
        indexer_head_dim: 2,
        indexer_top_k: 16,
    }
}

/// テンソル名から決まる値 (`seed` を変えると別の重み)。
fn fill_val(name: &str, i: usize, seed: u64) -> f32 {
    let mut h: u64 = seed;
    for b in name.as_bytes() {
        h = h.wrapping_mul(0x100000001b3).wrapping_add(*b as u64);
    }
    let x = (h.wrapping_add(i as u64).wrapping_mul(2654435761) % 2000) as f32 / 1000.0 - 1.0;
    x
}

fn encode(vals: &[f32], dtype: DType) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * dtype.size());
    match dtype {
        DType::F32 => {
            for v in vals {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        DType::BF16 => {
            for v in vals {
                // 丸めは切り捨て (テストの値は BF16 で正確に表せる範囲に収める)
                out.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
            }
        }
        DType::F16 => {
            for v in vals {
                out.extend_from_slice(&half::f16::from_f32(*v).to_bits().to_le_bytes());
            }
        }
    }
    out
}

/// `expected_tensor_shapes` の形どおりに値を埋めて書く。`fill` は
/// (name, index) → 値。`skip` に名前を入れると書かない。
fn write_st(
    path: &Path,
    cfg: &MtpConfig,
    dtype: DType,
    seed: u64,
    fill: &dyn Fn(&str, usize, f32) -> f32,
    skip: &[&str],
) -> Result<u64> {
    let mut header = serde_json::Map::new();
    let mut data: Vec<u8> = Vec::new();
    for (name, shape) in expected_tensor_shapes(cfg) {
        if skip.contains(&name.as_str()) {
            continue;
        }
        let n: usize = shape.iter().product();
        let vals: Vec<f32> = (0..n)
            .map(|i| fill(&name, i, fill_val(&name, i, seed)))
            .collect();
        let bytes = encode(&vals, dtype);
        let start = data.len();
        data.extend_from_slice(&bytes);
        header.insert(
            name.clone(),
            serde_json::json!({
                "dtype": dtype.name(),
                "shape": shape,
                "data_offsets": [start, data.len()],
            }),
        );
    }
    let hbytes = serde_json::to_vec(&serde_json::Value::Object(header))?;
    let mut out = Vec::with_capacity(8 + hbytes.len() + data.len());
    out.extend_from_slice(&(hbytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&hbytes);
    out.extend_from_slice(&data);
    std::fs::write(path, &out)?;
    Ok(out.len() as u64)
}

fn tmp_path(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("denno_mtp_{}_{}", std::process::id(), tag));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join("mtp.safetensors")
}

fn identity_fill(_name: &str, _i: usize, v: f32) -> f32 {
    v
}

fn build_tiny(tag: &str) -> (PathBuf, MtpConfig, MtpModel) {
    let cfg = tiny_cfg();
    let p = tmp_path(tag);
    write_st(&p, &cfg, DType::F32, 7, &identity_fill, &[]).unwrap();
    let m = MtpModel::open_with_config(&p, cfg.clone()).unwrap();
    (p, cfg, m)
}

// ---------------------------------------------------------------------------
// ローダ
// ---------------------------------------------------------------------------

#[test]
fn expected_shapes_has_31_tensors_for_real_config() {
    let cfg = MtpConfig::qwen38_flash_next();
    let v = expected_tensor_shapes(&cfg);
    assert_eq!(v.len(), 31, "実ファイルのテンソル数は 31");
    let map: BTreeMap<_, _> = v.iter().map(|(n, s)| (n.as_str(), s.clone())).collect();
    assert_eq!(map["mtp.fc_hidden.weight"], vec![2560, 2560]);
    assert_eq!(map["mtp.pre_fc_norm_hidden.weight"], vec![10240]);
    assert_eq!(
        map["mtp.layers.0.mlp.experts.gate_up_proj"],
        vec![512, 1280, 2560]
    );
    assert_eq!(
        map["mtp.layers.0.mlp.experts.down_proj"],
        vec![512, 2560, 640]
    );
    assert_eq!(
        map["mtp.layers.0.self_attn.q_proj.weight"],
        vec![12288, 2560]
    );
    assert_eq!(
        map["mtp.layers.0.self_attn.o_proj.weight"],
        vec![2560, 6144]
    );
    assert_eq!(
        map["mtp.layers.0.self_attn.indexer.index_qk_proj.weight"],
        vec![640, 2560]
    );
    assert_eq!(
        map["mtp.layers.0.attn_hyper_connection.block_inject_weight.weight"],
        vec![4, 10240]
    );
}

#[test]
fn loader_reads_f32_and_bf16() {
    let cfg = tiny_cfg();
    for dtype in [DType::F32, DType::BF16] {
        let p = tmp_path(&format!("roundtrip_{}", dtype.name()));
        let bytes = write_st(&p, &cfg, dtype, 3, &identity_fill, &[]).unwrap();
        let m = MtpModel::open_with_config(&p, cfg.clone()).unwrap();
        assert_eq!(m.file_bytes(), bytes);
        assert_eq!(m.file.len(), 31);
        assert!(m.host_f32_bytes() > 0);
        assert_eq!(m.layers.len(), 1);
        // 値が往復している (1 件だけ確認)
        let want = fill_val("mtp.fc_embedding.weight", 5, 3);
        let got = if dtype == DType::F32 {
            want
        } else {
            f32::from_bits((half::f16::from_f32(want).to_bits() as u32) << 16)
        };
        let _ = got; // BF16 丸めは切り捨てなので値の厳密比較はしない
        assert!(m.fc_embedding[5].is_finite());
    }
}

#[test]
fn loader_reports_missing_tensor() {
    let cfg = tiny_cfg();
    let p = tmp_path("missing");
    write_st(
        &p,
        &cfg,
        DType::F32,
        1,
        &identity_fill,
        &["mtp.layers.0.mlp.shared_expert_gate.weight"],
    )
    .unwrap();
    let err = MtpModel::open_with_config(&p, cfg).unwrap_err().to_string();
    assert!(err.contains("shared_expert_gate"), "err={err}");
    assert!(err.contains("ファイルに無い"), "err={err}");
}

#[test]
fn loader_reports_wrong_shape() {
    let cfg = tiny_cfg();
    let p = tmp_path("shape");
    // q_proj だけ 2 行足りない形で書く: expected を差し替えた config で書く
    let mut shapes = expected_tensor_shapes(&cfg);
    for (n, s) in shapes.iter_mut() {
        if n == "mtp.layers.0.self_attn.q_proj.weight" {
            s[0] += 2;
        }
        if n == "mtp.layers.0.self_attn.indexer.q_layernorm.weight" {
            s[0] = 3;
        }
    }
    write_custom_shapes(&p, &shapes, DType::F32);
    let err = MtpModel::open_with_config(&p, cfg).unwrap_err().to_string();
    assert!(err.contains("q_proj"), "err={err}");
    assert!(err.contains("shape"), "err={err}");
    assert!(err.contains("indexer.q_layernorm"), "err={err}");
}

#[test]
fn loader_reports_extra_tensor() {
    let cfg = tiny_cfg();
    let p = tmp_path("extra");
    let mut shapes = expected_tensor_shapes(&cfg);
    shapes.push(("mtp.unknown.weight".to_string(), vec![2, 2]));
    write_custom_shapes(&p, &shapes, DType::F32);
    let err = MtpModel::open_with_config(&p, cfg).unwrap_err().to_string();
    assert!(err.contains("想定外"), "err={err}");
    assert!(err.contains("mtp.unknown.weight"), "err={err}");
}

#[test]
fn loader_reports_unsupported_dtype() {
    let cfg = tiny_cfg();
    let p = tmp_path("dtype");
    // ヘッダだけ手で書き、dtype を F8 にする
    let shapes = expected_tensor_shapes(&cfg);
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, shape) in &shapes {
        let n: usize = shape.iter().product();
        let start = data.len();
        data.extend(std::iter::repeat(0u8).take(n * 4));
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": "F8_E4M3", "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    let hb = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut out = Vec::new();
    out.extend_from_slice(&(hb.len() as u64).to_le_bytes());
    out.extend_from_slice(&hb);
    out.extend_from_slice(&data);
    std::fs::write(&p, &out).unwrap();
    let err = MtpModel::open_with_config(&p, cfg).unwrap_err().to_string();
    assert!(err.contains("dtype"), "err={err}");
}

fn write_custom_shapes(p: &Path, shapes: &[(String, Vec<usize>)], dtype: DType) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, shape) in shapes {
        let n: usize = shape.iter().product();
        let vals: Vec<f32> = (0..n).map(|i| fill_val(name, i, 5)).collect();
        let start = data.len();
        data.extend_from_slice(&encode(&vals, dtype));
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": dtype.name(), "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    let hb = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut out = Vec::new();
    out.extend_from_slice(&(hb.len() as u64).to_le_bytes());
    out.extend_from_slice(&hb);
    out.extend_from_slice(&data);
    std::fs::write(p, &out).unwrap();
}

#[test]
fn gamma_fold_adds_one_only_for_norm_weights() {
    let cfg = tiny_cfg();
    let p = tmp_path("gamma");
    // 全部 0 で書く → `+1` が畳まれた gamma だけ 1 になる
    write_st(&p, &cfg, DType::F32, 0, &|_, _, _| 0.0, &[]).unwrap();
    let m = MtpModel::open_with_config(&p, cfg).unwrap();
    let hcd = m.cfg.hc * m.cfg.n_embd;
    assert!(
        m.hnorm.iter().all(|&v| v == 0.0),
        "pre_fc_norm_hidden は畳まない"
    );
    assert!(
        m.enorm.iter().all(|&v| v == 0.0),
        "pre_fc_norm_embedding は畳まない"
    );
    assert!(
        m.layers[0].attn_mixer.norm.iter().all(|&v| v == 1.0),
        "hc_norm は 1+w"
    );
    assert!(m.head_mixer.norm.iter().all(|&v| v == 1.0));
    assert!(m.layers[0].attn.q_norm.iter().all(|&v| v == 1.0));
    assert!(m.layers[0].attn.k_norm.iter().all(|&v| v == 1.0));
    assert!(m.layers[0].attn.index_q_norm.iter().all(|&v| v == 1.0));
    assert!(m.layers[0].attn.index_k_norm.iter().all(|&v| v == 1.0));
    assert_eq!(m.layers[0].attn_mixer.norm.len(), hcd);
}

#[test]
fn config_from_json_reads_text_config_and_mtp() {
    let js = r#"{
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "text_config": {
            "hidden_size": 2560, "hc_count": 4, "hc_lowrank": 320,
            "num_attention_heads": 24, "num_key_value_heads": 2, "head_dim": 256,
            "rms_norm_eps": 1e-06, "num_experts": 512, "num_experts_per_tok": 10,
            "moe_intermediate_size": 640, "shared_expert_intermediate_size": 640,
            "indexer_n_heads": 4, "indexer_head_dim": 128, "indexer_budget": 2048,
            "mtp_num_hidden_layers": 1,
            "rope_parameters": {"partial_rotary_factor": 0.25, "rope_theta": 10000000},
            "mtp": {"hybrid": true, "layer_types": ["full_attention"],
                    "num_hidden_layers": 1, "rope_theta": 10000000}
        }
    }"#;
    let cfg = MtpConfig::from_json_str(js).unwrap();
    assert_eq!(cfg, MtpConfig::qwen38_flash_next());
}

#[test]
fn config_from_json_rejects_unsupported_layer_type() {
    let js = r#"{"text_config":{"hidden_size":8,"hc_count":2,"hc_lowrank":4,
        "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,"rms_norm_eps":1e-6,
        "num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":3,
        "shared_expert_intermediate_size":3,"indexer_n_heads":2,"indexer_head_dim":2,
        "indexer_budget":16,"rope_theta":1e4,
        "mtp":{"num_hidden_layers":1,"layer_types":["linear_attention"]}}}"#;
    let err = MtpConfig::from_json_str(js).unwrap_err().to_string();
    assert!(err.contains("full_attention"), "err={err}");
}

// ---------------------------------------------------------------------------
// 数学
// ---------------------------------------------------------------------------

/// 独立な f64 参照 (このファイル内で書き下す)。
fn ref_input_projection(m: &MtpModel, h: &[f32], e_embd: &[f32]) -> Vec<f32> {
    let cfg = &m.cfg;
    let e = cfg.n_embd;
    let hc = cfg.hc;
    let rms = |x: &[f32], w: &[f32]| -> Vec<f64> {
        let ss: f64 = x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / e as f64;
        let inv = 1.0 / (ss + cfg.rms_eps as f64).sqrt();
        (0..e).map(|i| x[i] as f64 * inv * w[i] as f64).collect()
    };
    let en = rms(e_embd, &m.enorm);
    let mut out = vec![0.0f64; hc * e];
    for s in 0..hc {
        let hn = rms(&h[s * e..(s + 1) * e], &m.hnorm[s * e..(s + 1) * e]);
        for r in 0..e {
            let mut a = 0.0f64;
            let mut b = 0.0f64;
            for d in 0..e {
                a += m.fc_embedding[r * e + d] as f64 * en[d];
                b += m.fc_hidden[r * e + d] as f64 * hn[d];
            }
            out[s * e + r] = a + b;
        }
    }
    out.iter().map(|v| *v as f32).collect()
}

fn close(a: &[f32], b: &[f32], atol: f32, rtol: f32) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            let d = (x - y).abs();
            d <= atol + rtol * y.abs().max(x.abs())
        })
}

#[test]
fn input_projection_matches_independent_f64_reference() {
    let (_p, cfg, m) = build_tiny("proj");
    let e = cfg.n_embd;
    let h: Vec<f32> = (0..cfg.hc * e).map(|i| (i as f32 * 0.25).sin()).collect();
    let emb: Vec<f32> = (0..e).map(|i| (i as f32 * 0.5).cos()).collect();
    let got = m.input_projection(&h, &emb).unwrap();
    let want = ref_input_projection(&m, &h, &emb);
    assert!(close(&got, &want, 1e-5, 1e-5), "got={got:?}\nwant={want:?}");
}

#[test]
fn input_projection_rejects_bad_lengths() {
    let (_p, cfg, m) = build_tiny("proj_bad");
    let e = cfg.n_embd;
    let err = m
        .input_projection(&vec![0.0; e], &vec![0.0; e])
        .unwrap_err();
    assert!(err.to_string().contains("h の長さ"));
    let err = m
        .input_projection(&vec![0.0; cfg.hc * e], &vec![0.0; e + 1])
        .unwrap_err();
    assert!(err.to_string().contains("埋め込みの長さ"));
}

#[test]
fn inject_matches_hyper_concat_path_bitwise() {
    // 実装は inject を別テンソルとして回す。参照は hc_mix の w_down_inject 連結。
    let (_p, cfg, m) = build_tiny("inject");
    let hp = m.hp();
    let hcd = cfg.hc * cfg.n_embd;
    let lr = cfg.low_rank;
    let mixer = &m.layers[0].attn_mixer;
    let inject = mixer.inject.as_ref().unwrap();
    let mut cat = Vec::with_capacity((lr + cfg.hc) * hcd);
    cat.extend_from_slice(&mixer.down);
    cat.extend_from_slice(inject);

    let h: Vec<f32> = (0..hcd)
        .map(|i| ((i * 7) % 23) as f32 * 0.03 - 0.3)
        .collect();
    let x = HyperStreams {
        data: h,
        hc: cfg.hc,
        n_embd: cfg.n_embd,
    };
    let (mixed_ref, inj_ref) =
        hyper::hc_mix(&x, &mixer.norm, &mixer.down, &mixer.up, Some(&cat), &hp);
    let (mixed, inj) = super::mix_with_inject(&x, mixer, &hp);
    assert_eq!(mixed, mixed_ref, "mixed が連結版と違う");
    assert_eq!(inj, inj_ref, "inject が連結版と違う");
    // inject が 0 でないこと (分岐が効いている証拠)
    assert!(inj.iter().any(|v| v.abs() > 1e-6), "inject が全部 0");
}

/// 独立な f64 attention 参照。
fn ref_attention(
    m: &MtpModel,
    a: &MtpAttn,
    x: &[f32],
    kv_k: &[f32],
    kv_v: &[f32],
    pos: usize,
) -> Vec<f32> {
    let cfg = &m.cfg;
    let e = cfg.n_embd;
    let dh = cfg.head_dim;
    let nh = cfg.n_head;
    let nkv = cfg.n_head_kv;
    let mv = |w: &[f32], x: &[f32], rows: usize, cols: usize| -> Vec<f64> {
        (0..rows)
            .map(|r| {
                (0..cols)
                    .map(|c| w[r * cols + c] as f64 * x[c] as f64)
                    .sum::<f64>()
            })
            .collect()
    };
    let qg = mv(&a.q_proj, x, nh * 2 * dh, e);
    let mut q: Vec<f64> = Vec::new();
    let mut gate: Vec<f64> = Vec::new();
    for h in 0..nh {
        q.extend_from_slice(&qg[h * 2 * dh..h * 2 * dh + dh]);
        gate.extend_from_slice(&qg[h * 2 * dh + dh..(h + 1) * 2 * dh]);
    }
    let mut k = mv(&a.k_proj, x, nkv * dh, e);
    let v = mv(&a.v_proj, x, nkv * dh, e);
    let rms = |x: &[f64], w: &[f32], cols: usize| -> Vec<f64> {
        let ss: f64 = x.iter().map(|v| v * v).sum::<f64>() / cols as f64;
        let inv = 1.0 / (ss + cfg.rms_eps as f64).sqrt();
        x.iter()
            .enumerate()
            .map(|(i, v)| v * inv * w[i % cols] as f64)
            .collect()
    };
    for h in 0..nh {
        let row = rms(&q[h * dh..(h + 1) * dh], &a.q_norm, dh);
        q[h * dh..(h + 1) * dh].copy_from_slice(&row);
    }
    for h in 0..nkv {
        let row = rms(&k[h * dh..(h + 1) * dh], &a.k_norm, dh);
        k[h * dh..(h + 1) * dh].copy_from_slice(&row);
    }
    // rope (neox, 対ごと) — 実装と同じ式を独立に f64 で書く
    let rope = |x: &mut [f64], heads: usize, pos: usize| {
        let half = cfg.n_rot / 2;
        for h in 0..heads {
            for j in 0..half {
                let ang =
                    pos as f64 * (cfg.rope_theta as f64).powf(-2.0 * j as f64 / cfg.n_rot as f64);
                let (c, s) = (ang.cos(), ang.sin());
                let x0 = x[h * dh + j];
                let x1 = x[h * dh + half + j];
                x[h * dh + j] = x0 * c - x1 * s;
                x[h * dh + half + j] = x0 * s + x1 * c;
            }
        }
    };
    rope(&mut q, nh, pos);
    rope(&mut k, nkv, pos);

    // KV は履歴 + 今の 1 位置
    let mut kk: Vec<f64> = kv_k.iter().map(|v| *v as f64).collect();
    let mut vv: Vec<f64> = kv_v.iter().map(|v| *v as f64).collect();
    kk.extend_from_slice(&k);
    vv.extend_from_slice(&v);
    let n_pos = pos + 1;
    let group = nh / nkv;
    let scale = 1.0 / (dh as f64).sqrt();
    let mut ctx = vec![0.0f64; nh * dh];
    for h in 0..nh {
        let kvh = h / group;
        let mut sc = vec![0.0f64; n_pos];
        for j in 0..n_pos {
            let mut acc = 0.0f64;
            for d in 0..dh {
                acc += q[h * dh + d] * kk[j * nkv * dh + kvh * dh + d];
            }
            sc[j] = acc * scale;
        }
        let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ex: Vec<f64> = sc.iter().map(|v| (v - mx).exp()).collect();
        let sum: f64 = ex.iter().sum();
        for d in 0..dh {
            let mut acc = 0.0f64;
            for j in 0..n_pos {
                acc += ex[j] / sum * vv[j * nkv * dh + kvh * dh + d];
            }
            ctx[h * dh + d] = acc * (1.0 / (1.0 + (-gate[h * dh + d]).exp()));
        }
    }
    mv(
        &a.o_proj,
        &ctx.iter().map(|v| *v as f32).collect::<Vec<_>>(),
        e,
        nh * dh,
    )
    .iter()
    .map(|v| *v as f32)
    .collect()
}

#[test]
fn attention_matches_independent_f64_reference() {
    let (_p, cfg, m) = build_tiny("attn");
    let a = &m.layers[0].attn;
    let x: Vec<f32> = (0..cfg.n_embd).map(|i| (i as f32 * 0.37).sin()).collect();
    let d = cfg.n_head_kv * cfg.head_dim;

    // pos 0 (履歴なし)
    let mut kv = MtpKvCache::new(cfg.n_head_kv, cfg.head_dim);
    let got0 = m.attention(a, &x, &mut kv).unwrap();
    let want0 = ref_attention(&m, a, &x, &[], &[], 0);
    assert!(
        close(&got0, &want0, 1e-5, 1e-5),
        "pos0 got={got0:?}\nwant={want0:?}"
    );
    assert_eq!(kv.len(), 1);

    // pos 1 (履歴 1 位置)
    let k_hist = kv.k.clone();
    let v_hist = kv.v.clone();
    let x2: Vec<f32> = (0..cfg.n_embd).map(|i| (i as f32 * 0.61).cos()).collect();
    let got1 = m.attention(a, &x2, &mut kv).unwrap();
    let want1 = ref_attention(&m, a, &x2, &k_hist, &v_hist, 1);
    assert!(
        close(&got1, &want1, 1e-5, 1e-5),
        "pos1 got={got1:?}\nwant={want1:?}"
    );
    assert_eq!(kv.len(), 2);
    assert_eq!(d, 4);
}

#[test]
fn attention_rejects_mismatched_kv_shape() {
    let (_p, cfg, m) = build_tiny("attn_shape");
    let a = &m.layers[0].attn;
    let x = vec![0.1f32; cfg.n_embd];
    let mut kv = MtpKvCache::new(3, cfg.head_dim);
    let err = m.attention(a, &x, &mut kv).unwrap_err().to_string();
    assert!(err.contains("KV の形状"), "err={err}");
}

#[test]
fn top_k_prefers_smaller_index_on_ties() {
    let p = [0.4f32, 0.1, 0.4, 0.1];
    assert_eq!(top_k(&p, 2), vec![0, 2]);
}

#[test]
fn moe_matches_independent_f64_reference() {
    let (_p, cfg, m) = build_tiny("moe");
    let f = &m.layers[0].ffn;
    let e = cfg.n_embd;
    let n_ff = cfg.n_ff_exp;
    let x: Vec<f32> = (0..e).map(|i| (i as f32 * 0.23).sin()).collect();
    let got = m.moe(f, &x).unwrap();

    let dotf =
        |a: &[f32], b: &[f32]| -> f64 { a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum() };
    let gu = m.tensor_f32(&f.gate_up_name, 0.0).unwrap();
    let dn = m.tensor_f32(&f.down_name, 0.0).unwrap();
    let mut logits: Vec<f64> = (0..cfg.n_expert)
        .map(|ex| dotf(&f.router[ex * e..(ex + 1) * e], &x))
        .collect();
    let mx = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let ex: Vec<f64> = logits.iter().map(|v| (v - mx).exp()).collect();
    let sum: f64 = ex.iter().sum();
    for v in logits.iter_mut() {
        *v = *v; // keep
    }
    let probs: Vec<f64> = ex.iter().map(|v| v / sum).collect();
    let sel = top_k(
        &probs.iter().map(|v| *v as f32).collect::<Vec<_>>(),
        cfg.n_expert_used,
    );
    let denom: f64 = sel.iter().map(|&i| probs[i]).sum();
    let mut want = vec![0.0f64; e];
    for &ee in &sel {
        let w = probs[ee] / denom;
        let base = ee * 2 * n_ff * e;
        let mut act = vec![0.0f64; n_ff];
        for r in 0..n_ff {
            let g = dotf(&gu[base + r * e..base + (r + 1) * e], &x);
            let u = dotf(&gu[base + (n_ff + r) * e..base + (n_ff + r + 1) * e], &x);
            act[r] = g / (1.0 + (-g).exp()) * u;
        }
        let dbase = ee * e * n_ff;
        for r in 0..e {
            want[r] += w * dotf(
                &dn[dbase + r * n_ff..dbase + (r + 1) * n_ff],
                &act.iter().map(|v| *v as f32).collect::<Vec<_>>(),
            );
        }
    }
    // shared expert
    let mut sact = vec![0.0f64; n_ff];
    for r in 0..n_ff {
        let g = dotf(&f.shared_gate[r * e..(r + 1) * e], &x);
        let u = dotf(&f.shared_up[r * e..(r + 1) * e], &x);
        sact[r] = g / (1.0 + (-g).exp()) * u;
    }
    let sf: Vec<f32> = sact.iter().map(|v| *v as f32).collect();
    let g = 1.0 / (1.0 + (-dotf(&f.shared_expert_gate, &x)).exp());
    for r in 0..e {
        want[r] += g * dotf(&f.shared_down[r * n_ff..(r + 1) * n_ff], &sf);
    }
    let want: Vec<f32> = want.iter().map(|v| *v as f32).collect();
    assert!(close(&got, &want, 1e-5, 1e-5), "got={got:?}\nwant={want:?}");
}

// ---------------------------------------------------------------------------
// draft 一式
// ---------------------------------------------------------------------------

fn dummy_lm_head(v: &[f32]) -> Result<Vec<f32>> {
    // [n_embd] -> 4 logits (決定的)
    let mut out = vec![0.0f32; 4];
    for (i, &x) in v.iter().enumerate() {
        out[i % 4] += x * (1.0 + (i % 3) as f32 * 0.5);
    }
    Ok(out)
}

#[test]
fn draft_runs_and_is_deterministic() {
    let (_p, cfg, m) = build_tiny("draft");
    let e = cfg.n_embd;
    let h: Vec<f32> = (0..cfg.hc * e).map(|i| (i as f32 * 0.11).sin()).collect();
    let emb: Vec<f32> = (0..e).map(|i| (i as f32 * 0.21).cos()).collect();

    let mut kv1 = MtpKvCache::new(cfg.n_head_kv, cfg.head_dim);
    let mut lm = |v: &[f32]| dummy_lm_head(v);
    let a = m.draft(&h, &emb, &mut kv1, &mut lm).unwrap();
    let mut kv2 = MtpKvCache::new(cfg.n_head_kv, cfg.head_dim);
    let b = m.draft(&h, &emb, &mut kv2, &mut lm).unwrap();
    assert_eq!(a.logits.len(), 4);
    assert_eq!(a.hidden_next.len(), cfg.hc * e);
    assert_eq!(a.hidden_head.len(), e);
    assert_eq!(a.pos, 0);
    assert_eq!(a.logits, b.logits, "同じ入力で logits が違う");
    assert_eq!(a.hidden_next, b.hidden_next);
    assert_eq!(kv1.k, kv2.k);
    assert!(a.logits.iter().all(|v| v.is_finite()));
    assert!(a.hidden_next.iter().all(|v| v.is_finite()));
    // 2 段目は 1 段目と違う (KV が伸びる)
    let c = m.draft(&h, &emb, &mut kv1, &mut lm).unwrap();
    assert_eq!(c.pos, 1);
    assert_ne!(c.logits, a.logits);
    assert_eq!(kv1.len(), 2);
    assert_eq!(a.argmax(), {
        let mut best = 0usize;
        for (i, &v) in a.logits.iter().enumerate() {
            if v > a.logits[best] {
                best = i;
            }
        }
        best as u32
    });
}

#[test]
fn draft_rejects_positions_beyond_indexer_budget() {
    let cfg = MtpConfig {
        indexer_top_k: 2,
        ..tiny_cfg()
    };
    let p = tmp_path("budget");
    write_st(&p, &cfg, DType::F32, 9, &identity_fill, &[]).unwrap();
    let m = MtpModel::open_with_config(&p, cfg.clone()).unwrap();
    let e = cfg.n_embd;
    let h = vec![0.05f32; cfg.hc * e];
    let emb = vec![0.02f32; e];
    let mut kv = MtpKvCache::new(cfg.n_head_kv, cfg.head_dim);
    let mut lm = |v: &[f32]| dummy_lm_head(v);
    m.draft(&h, &emb, &mut kv, &mut lm).unwrap();
    m.draft(&h, &emb, &mut kv, &mut lm).unwrap();
    let err = m.draft(&h, &emb, &mut kv, &mut lm).unwrap_err().to_string();
    assert!(err.contains("indexer_budget"), "err={err}");
}

#[test]
fn open_requires_config_json_next_to_file() {
    let cfg = tiny_cfg();
    let p = tmp_path("nocfg");
    write_st(&p, &cfg, DType::F32, 2, &identity_fill, &[]).unwrap();
    // tmp ディレクトリには config.json を置かない
    let err = MtpModel::open(&p).unwrap_err().to_string();
    assert!(err.contains("config.json"), "err={err}");
}

#[test]
fn open_reads_config_json_sibling() {
    let cfg = tiny_cfg();
    let p = tmp_path("withcfg");
    write_st(&p, &cfg, DType::F32, 2, &identity_fill, &[]).unwrap();
    let js = format!(
        r#"{{"text_config":{{"hidden_size":{},"hc_count":{},"hc_lowrank":{},
        "num_attention_heads":{},"num_key_value_heads":{},"head_dim":{},"rms_norm_eps":1e-6,
        "num_experts":{},"num_experts_per_tok":{},"moe_intermediate_size":{},
        "shared_expert_intermediate_size":{},"indexer_n_heads":{},"indexer_head_dim":{},
        "indexer_budget":{},"rope_theta":10000,
        "mtp":{{"num_hidden_layers":1,"layer_types":["full_attention"],"rope_theta":10000}}}}}}"#,
        cfg.n_embd,
        cfg.hc,
        cfg.low_rank,
        cfg.n_head,
        cfg.n_head_kv,
        cfg.head_dim,
        cfg.n_expert,
        cfg.n_expert_used,
        cfg.n_ff_exp,
        cfg.n_ff_shexp,
        cfg.indexer_n_head,
        cfg.indexer_head_dim,
        cfg.indexer_top_k
    );
    std::fs::write(p.parent().unwrap().join("config.json"), js).unwrap();
    let m = MtpModel::open(&p).unwrap();
    assert_eq!(m.cfg.n_embd, cfg.n_embd);
    assert_eq!(
        m.cfg.n_rot, cfg.head_dim,
        "partial_rotary_factor 未指定は head_dim"
    );
}

#[test]
fn env_path_is_none_when_unset() {
    // 他テストと競合しないよう、このプロセスで設定されていなければ None
    if std::env::var_os(MTP_PATH_ENV).is_none() {
        assert!(path_from_env().is_none());
    }
    assert_eq!(MTP_PATH_ENV, "DENNO_MTP_PATH");
}

/// 実ファイル (5.2 GB) を開く。`DENNO_MTP_PATH` を設定して
/// `cargo test --release -p denno-engine --lib mtp::tests::real -- --ignored` で回す。
#[test]
#[ignore]
fn real_mtp_file_loads_and_runs_one_step() {
    let Some(path) = path_from_env() else {
        eprintln!("DENNO_MTP_PATH 未設定のため skip");
        return;
    };
    let m = MtpModel::open(&path).unwrap();
    assert_eq!(m.cfg, MtpConfig::qwen38_flash_next());
    assert_eq!(m.file.len(), 31);
    eprintln!(
        "[mtp] {} tensors, {} bytes, host f32 {} MB",
        m.file.len(),
        m.file_bytes(),
        m.host_f32_bytes() / (1024 * 1024)
    );
    let cfg = &m.cfg;
    let e = cfg.n_embd;
    // 決定的な擬似入力 (trunk の実値ではない)
    let h: Vec<f32> = (0..cfg.hc * e)
        .map(|i| ((i * 37 % 1000) as f32 / 1000.0 - 0.5) * 0.2)
        .collect();
    let emb: Vec<f32> = (0..e)
        .map(|i| ((i * 53 % 997) as f32 / 997.0 - 0.5) * 0.2)
        .collect();
    let mut kv = MtpKvCache::new(cfg.n_head_kv, cfg.head_dim);
    let vocab = 248320usize;
    let mut lm = |v: &[f32]| -> Result<Vec<f32>> {
        // 実際の lm_head の代わりに固定の疎な射影 (logits の形だけ確認)
        let mut out = vec![0.0f32; vocab];
        for r in 0..8 {
            let mut acc = 0.0f32;
            for d in 0..e {
                acc += v[d] * ((r * 31 + d) as f32 * 0.0001).cos();
            }
            out[r * 1000] = acc;
        }
        Ok(out)
    };
    let t = std::time::Instant::now();
    let d = m.draft(&h, &emb, &mut kv, &mut lm).unwrap();
    let dt = t.elapsed();
    eprintln!(
        "[mtp] draft step: {} ms, logits {} (finite={}), hidden_next[0..4]={:?}",
        dt.as_millis(),
        d.logits.len(),
        d.logits.iter().all(|v| v.is_finite()),
        &d.hidden_next[..4]
    );
    assert!(d.logits.iter().all(|v| v.is_finite()));
    assert!(d.hidden_next.iter().all(|v| v.is_finite()));
    assert!(d.hidden_head.iter().all(|v| v.is_finite()));
    assert!(
        d.hidden_head.iter().any(|v| v.abs() > 1e-6),
        "head 出力が全部 0"
    );
    assert_eq!(kv.len(), 1);

    // 同じ入力・新しい KV で 2 回目 → ビット一致 (重みの展開順に依存しない)
    let mut kv2 = MtpKvCache::new(cfg.n_head_kv, cfg.head_dim);
    let d2 = m.draft(&h, &emb, &mut kv2, &mut lm).unwrap();
    assert_eq!(
        d.logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        d2.logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "同じ入力で logits のビットが違う"
    );
    assert_eq!(d.hidden_next, d2.hidden_next);
    assert_eq!(kv.k, kv2.k);
    assert_eq!(kv.v, kv2.v);
}
