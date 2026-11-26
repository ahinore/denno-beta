//! task 1409: `auto_config` / `gpu_budget` の境界と決定性テスト。
//!
//! 対象は公開 API の純関数だけ (env は触らない)。
//! - `plan_gpu_config` を 8 / 12 / 16 / 24 / 32 / 48 / 80 GiB で回し、
//!   予算が 0 に潰れない・使える VRAM を超えない・VRAM 増加に対して単調で
//!   あることを縛る。
//! - 0 / 1 MiB / u64::MAX のモデル長、2 GiB / 1 TiB のカードで panic せず
//!   上限に収まることを縛る。
//! - `plan_env_pairs` のキー順と値が決定的であること (同じ入力で同じ列) を縛る。
//! - `gpu_budget` の見積り式 (`q8_0_row_bytes` / `classify_weights` /
//!   `flash_budget` / `n_full_layers` / `WeightLru`) の境界と飽和を縛る。
//!
//! CPU-only・高速。`cargo test --release -p denno-engine --test auto_config_edge_1409`。

use denno_engine::auto_config::{
    plan_env_pairs, plan_gpu_config, GpuConfig, ModelQuant, DENSE_WORKING_SET_MB, EXPERT_CAP_MB,
    MIN_EXPERT_MB, MIN_HIT_ZERO_FROM_MB, PLAN_RESERVE_MB, RAM_TIER_MAX_VRAM_MB, SMALL_VRAM_MB,
    WEIGHT_CAP_MB,
};
use denno_engine::gpu_budget::{
    classify_weights, flash_budget, n_full_layers, q8_0_row_bytes, WeightLru, WeightTotals,
    GPU_W_RESERVE_MB, SCRATCH_ARENA_CHUNK,
};
use denno_engine::kquant::GgmlType;

/// MiB per GiB。
const GIB: u64 = 1024;
/// 実カードを想定したホスト RAM (64 GiB)。
const RAM_64G: u64 = 64 * GIB;

/// 検証対象の VRAM 段 (MiB)。8 / 12 / 16 / 24 / 32 / 48 / 80 GiB。
fn tiers() -> [u64; 7] {
    [8, 12, 16, 24, 32, 48, 80].map(|g| g * GIB)
}

/// 実モデル相当の重み合計バイト数。
fn model_bytes(q: ModelQuant) -> u64 {
    match q {
        ModelQuant::Q3 => 84_000_000_000,
        ModelQuant::Q4 => 111_000_000_000,
    }
}

/// 全段の計画を組む (同じ引数なら同じ値になることの確認にも使う)。
fn plan(vram_mb: u64, bytes: u64, q: ModelQuant) -> GpuConfig {
    plan_gpu_config(vram_mb, RAM_64G, bytes, q)
}

/// 予算の共通不変条件。`usable` は計画の分母。
fn assert_invariants(c: &GpuConfig, usable: u64) {
    // 単一項は必ず「使える VRAM」以下。0 に潰さない。
    assert!(
        c.gpu_weight_mb <= usable,
        "weight={} > usable={usable}",
        c.gpu_weight_mb
    );
    assert!(
        c.expert_vram_mb <= usable,
        "expert={} > usable={usable}",
        c.expert_vram_mb
    );
    assert!(c.gpu_weight_mb <= WEIGHT_CAP_MB);
    assert!(c.expert_vram_mb <= EXPERT_CAP_MB);
    // 和は「上限の和」なので 40% + 72% = 112% まで。120% を超えたら丸めの暴走。
    assert!(
        c.gpu_weight_mb + c.expert_vram_mb <= usable * 120 / 100,
        "sum={} > usable*1.2={}",
        c.gpu_weight_mb + c.expert_vram_mb,
        usable * 120 / 100
    );
    // 負値・NaN は u64 / f32 の型で防がれるが、範囲も縛る。
    if let Some(h) = c.expert_gpu_min_hit {
        assert!(h.is_finite(), "min_hit が有限でない: {h}");
        assert!((0.0..=1.0).contains(&h), "min_hit が範囲外: {h}");
    }
    assert_eq!(c.reserve_mb, PLAN_RESERVE_MB);
    assert_eq!(c.dense_est_mb, c.model_mb.saturating_mul(c.quant.dense_pct()) / 100);
    // expert_gpu は expert_vram_mb > 0 と同値。
    assert_eq!(c.expert_gpu, c.expert_vram_mb > 0);
    assert_eq!(c.experts_on_gpu(), c.expert_vram_mb > 0 && c.expert_gpu);
    if c.expert_vram_mb == 0 {
        assert_eq!(c.expert_gpu_min_hit, None, "expert 無しで min_hit を出した");
    }
}

#[test]
fn plan_budget_invariants_all_tiers() {
    for vram in tiers() {
        let usable = vram - PLAN_RESERVE_MB;
        for q in [ModelQuant::Q3, ModelQuant::Q4] {
            let bytes = model_bytes(q);
            let c = plan(vram, bytes, q);
            assert_eq!(c.vram_mb, vram);
            assert_eq!(c.ram_mb, RAM_64G);
            assert_eq!(c.quant, q);
            assert_eq!(c.model_mb, bytes >> 20);
            assert_invariants(&c, usable);
            // 8 GiB 以上は dense も expert も 0 に潰さない。
            assert!(
                c.gpu_weight_mb >= 256,
                "{vram} MiB {q:?}: weight={} が床未満",
                c.gpu_weight_mb
            );
            assert!(
                c.expert_vram_mb >= MIN_EXPERT_MB,
                "{vram} MiB {q:?}: expert={} が下限未満",
                c.expert_vram_mb
            );
        }
    }
}

#[test]
fn plan_tier_table_is_stable() {
    // (公称 GiB, weight MiB, expert MiB)。task 886 の段の表を 48 / 80 まで延長。
    // 8 GiB は task 932 の監査で dense のワーキングセット床に載せた (以前 2,867)。
    let table: [(u64, u64, u64); 7] = [
        (8, 5_632, 1_536),
        (12, 5_632, 5_632),
        (16, 6_144, 9_216),
        (24, 9_420, 16_957),
        (32, 12_000, 22_000),
        (48, 12_000, 22_000),
        (80, 12_000, 22_000),
    ];
    for (gb, weight, expert) in table {
        for q in [ModelQuant::Q3, ModelQuant::Q4] {
            let c = plan(gb * GIB, model_bytes(q), q);
            assert_eq!(c.gpu_weight_mb, weight, "{gb} GiB {q:?} weight");
            assert_eq!(c.expert_vram_mb, expert, "{gb} GiB {q:?} expert");
        }
        // weight は段ごとに単調 (非減少)。
        let c_q4 = plan(gb * GIB, model_bytes(ModelQuant::Q4), ModelQuant::Q4);
        assert!(c_q4.gpu_weight_mb <= WEIGHT_CAP_MB);
    }
}

#[test]
fn plan_budgets_monotone_with_vram() {
    for q in [ModelQuant::Q3, ModelQuant::Q4] {
        let mut prev: Option<GpuConfig> = None;
        for vram in tiers() {
            let c = plan(vram, model_bytes(q), q);
            if let Some(p) = prev.as_ref() {
                assert!(
                    c.gpu_weight_mb >= p.gpu_weight_mb,
                    "{q:?}: weight が VRAM 増加で減った ({} -> {})",
                    p.gpu_weight_mb,
                    c.gpu_weight_mb
                );
                assert!(
                    c.expert_vram_mb >= p.expert_vram_mb,
                    "{q:?}: expert が VRAM 増加で減った ({} -> {})",
                    p.expert_vram_mb,
                    c.expert_vram_mb
                );
                assert!(c.dense_est_mb == p.dense_est_mb, "{q:?}: dense 推定が揺れた");
            }
            prev = Some(c);
        }
    }
    // 8 GiB から 32 GiB までは両方とも厳密に増える (段の取りこぼし検出)。
    let small = plan(8 * GIB, model_bytes(ModelQuant::Q4), ModelQuant::Q4);
    let big = plan(32 * GIB, model_bytes(ModelQuant::Q4), ModelQuant::Q4);
    assert!(big.gpu_weight_mb > small.gpu_weight_mb);
    assert!(big.expert_vram_mb > small.expert_vram_mb);
}

#[test]
fn plan_tier_flags_follow_nominal_boundaries() {
    for vram in tiers() {
        let c = plan(vram, model_bytes(ModelQuant::Q4), ModelQuant::Q4);
        let usable = vram - PLAN_RESERVE_MB;
        // 12 GiB 以上で P45 ゲートを 0 にする。11 GiB 以下は既定のまま。
        if vram >= MIN_HIT_ZERO_FROM_MB {
            assert_eq!(c.expert_gpu_min_hit, Some(0.0), "{vram} MiB");
        } else {
            assert_eq!(c.expert_gpu_min_hit, None, "{vram} MiB");
        }
        // prefill 資源は 10 GiB / 16 GiB の上限境界で切り替わる。
        if vram <= SMALL_VRAM_MB {
            assert_eq!((c.pread_threads, c.pf_chunk), (Some(8), Some(1024)));
        } else if vram <= 16 * GIB {
            assert_eq!((c.pread_threads, c.pf_chunk), (Some(12), Some(1536)));
        } else {
            assert_eq!((c.pread_threads, c.pf_chunk), (None, None));
        }
        // RAM tier は 16 GiB 以下のカードだけ。
        assert_eq!(
            c.expert_ram.is_some(),
            vram <= RAM_TIER_MAX_VRAM_MB + denno_engine::auto_config::VRAM_TIER_SLACK_MB
        );
        // 小さいカード (10-16 GiB) は dense のワーキングセット床を割らない。
        if (10 * GIB..=16 * GIB).contains(&vram) {
            assert!(
                c.gpu_weight_mb >= DENSE_WORKING_SET_MB,
                "{vram} MiB: weight={} < ワーキングセット",
                c.gpu_weight_mb
            );
            // 小さいカードは dense + expert を使える VRAM に収める。
            assert!(
                c.gpu_weight_mb + c.expert_vram_mb <= usable,
                "{vram} MiB: {} + {} > {usable}",
                c.gpu_weight_mb,
                c.expert_vram_mb
            );
        }
        // 式の係数と表の整合 (48 / 80 GiB は絶対上限で頭打ち)。
        if vram >= 32 * GIB {
            assert_eq!(c.gpu_weight_mb, WEIGHT_CAP_MB, "{vram} MiB");
            assert_eq!(c.expert_vram_mb, EXPERT_CAP_MB, "{vram} MiB");
        }
    }
}

#[test]
fn plan_zero_and_tiny_cards_do_not_panic() {
    for vram in [0u64, 1, 512, PLAN_RESERVE_MB - 1, PLAN_RESERVE_MB] {
        let usable = vram.saturating_sub(PLAN_RESERVE_MB);
        for bytes in [0u64, 1, 111_000_000_000, u64::MAX] {
            let c = plan(vram, bytes, ModelQuant::Q4);
            assert_eq!(c.vram_mb, vram);
            assert_eq!(c.model_mb, bytes >> 20);
            // usable が 0 なら予算も 0。それ以外でも single 項は usable 以下。
            assert!(c.gpu_weight_mb <= usable);
            assert!(c.expert_vram_mb <= usable);
            assert!(c.gpu_weight_mb <= WEIGHT_CAP_MB);
            assert!(c.expert_vram_mb <= EXPERT_CAP_MB);
            assert!(c.expert_gpu_min_hit.is_none());
        }
    }
    // 2 GiB のカード: expert を切る (1 本も入らない)。
    let tiny = plan(2 * GIB, model_bytes(ModelQuant::Q4), ModelQuant::Q4);
    assert_eq!(tiny.expert_vram_mb, 0);
    assert!(!tiny.experts_on_gpu());
    assert_eq!(tiny.expert_gpu_min_hit, None);
}

#[test]
fn plan_zero_model_never_exceeds_real_model_budget() {
    for vram in tiers() {
        let usable = vram - PLAN_RESERVE_MB;
        for q in [ModelQuant::Q3, ModelQuant::Q4] {
            let zero = plan(vram, 0, q);
            let one = plan(vram, 1, q);
            let real = plan(vram, model_bytes(q), q);
            for c in [&zero, &one] {
                assert_invariants(c, usable);
                assert_eq!(c.model_mb, 0);
                assert_eq!(c.dense_est_mb, 0);
                // モデル 0 バイトでも実モデルの予算を超えない。
                assert!(c.gpu_weight_mb <= real.gpu_weight_mb);
                assert!(c.expert_vram_mb <= real.expert_vram_mb);
            }
        }
    }
}

#[test]
fn plan_extreme_model_size_and_huge_card() {
    // u64::MAX バイトのモデル: 飽和して panic しない。予算は上限で頭打ち。
    let c = plan(32 * GIB, u64::MAX, ModelQuant::Q3);
    assert_eq!(c.model_mb, u64::MAX >> 20);
    assert_eq!(
        c.dense_est_mb,
        c.model_mb.saturating_mul(ModelQuant::Q3.dense_pct()) / 100
    );
    assert_eq!(c.gpu_weight_mb, WEIGHT_CAP_MB);
    assert_eq!(c.expert_vram_mb, EXPERT_CAP_MB);
    assert_invariants(&c, 32 * GIB - PLAN_RESERVE_MB);

    // 1 TiB のカード: 割合の項は絶対上限を超えるので頭打ち。
    // ただし expert の下限判定は「使える VRAM の 12%」を絶対上限と比べるため、
    // 使える VRAM が約 183 GiB を超えると expert は 0 になる
    // (22,000 / 0.12 = 183,333 MiB が境界)。ここは公開挙動として縛る。
    let huge = plan(1024 * GIB, model_bytes(ModelQuant::Q4), ModelQuant::Q4);
    assert_eq!(huge.gpu_weight_mb, WEIGHT_CAP_MB);
    assert_eq!(huge.expert_vram_mb, 0, "12% 下限が絶対上限を超える段");
    assert_eq!(huge.expert_gpu_min_hit, None);
    assert_eq!((huge.pread_threads, huge.pf_chunk), (None, None));
    assert_eq!(huge.expert_ram, None);
    assert_invariants(&huge, 1024 * GIB - PLAN_RESERVE_MB);

    // 12% 下限の切り替わりを 1 MiB で確かめる (usable 183,341 / 183,342)。
    let on = plan(184_365, model_bytes(ModelQuant::Q4), ModelQuant::Q4);
    let off = plan(184_366, model_bytes(ModelQuant::Q4), ModelQuant::Q4);
    assert_eq!(on.expert_vram_mb, EXPERT_CAP_MB);
    assert_eq!(off.expert_vram_mb, 0);
}

#[test]
fn plan_is_deterministic() {
    for vram in tiers() {
        for q in [ModelQuant::Q3, ModelQuant::Q4] {
            let a = plan(vram, model_bytes(q), q);
            let b = plan(vram, model_bytes(q), q);
            assert_eq!(a, b, "{vram} MiB {q:?}: 同じ入力で値が変わった");
        }
    }
}

/// `plan_env_pairs` の期待するキー列 (実装の push 順)。
fn expected_keys(c: &GpuConfig) -> Vec<&'static str> {
    let mut v = vec![
        "DENNO_VRAM_RESERVE_MB",
        "DENNO_GPU_WEIGHT_MB",
        "DENNO_EXPERT_VRAM_MB",
    ];
    if c.experts_on_gpu() {
        v.push("DENNO_EXPERT_GPU");
        v.push("DENNO_EXPERT_IQ_GPU");
    }
    if c.expert_gpu_min_hit.is_some() {
        v.push("DENNO_EXPERT_GPU_MIN_HIT");
    }
    if c.expert_ram.is_some() {
        v.push("DENNO_EXPERT_RAM_MB");
    }
    if c.pread_threads.is_some() {
        v.push("DENNO_PF_PREAD_THREADS");
    }
    if c.pf_chunk.is_some() {
        v.push("DENNO_PF_CHUNK");
    }
    v
}

#[test]
fn env_pairs_key_order_is_stable_across_tiers() {
    for vram in tiers() {
        let c = plan(vram, model_bytes(ModelQuant::Q4), ModelQuant::Q4);
        let pairs = plan_env_pairs(&c);
        let keys: Vec<&str> = pairs.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, expected_keys(&c), "{vram} MiB: キー順が変わった");
        // 重複キーは出さない。
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), keys.len(), "{vram} MiB: キーが重複");
        // 2 回呼んでも同じ列 (決定性)。
        assert_eq!(pairs, plan_env_pairs(&c), "{vram} MiB: 生成が非決定的");
    }
}

#[test]
fn env_pairs_values_match_config() {
    // 32 GiB: 手動ピークと同じ値。min_hit は "0"。
    let c = plan(32 * GIB, model_bytes(ModelQuant::Q4), ModelQuant::Q4);
    let p = plan_env_pairs(&c);
    let get = |k: &str| {
        p.iter()
            .find(|(n, _)| *n == k)
            .map(|(_, v)| v.as_str())
    };
    assert_eq!(p[0], ("DENNO_VRAM_RESERVE_MB", "1024".to_string()));
    assert_eq!(get("DENNO_GPU_WEIGHT_MB"), Some("12000"));
    assert_eq!(get("DENNO_EXPERT_VRAM_MB"), Some("22000"));
    assert_eq!(get("DENNO_EXPERT_GPU"), Some("1"));
    assert_eq!(get("DENNO_EXPERT_IQ_GPU"), Some("1"));
    assert_eq!(get("DENNO_EXPERT_GPU_MIN_HIT"), Some("0"));
    assert_eq!(get("DENNO_EXPERT_RAM_MB"), None);
    assert_eq!(get("DENNO_PF_PREAD_THREADS"), None);
    assert_eq!(get("DENNO_PF_CHUNK"), None);

    // 8 GiB: expert は載せるが min_hit は出さない。RAM tier と prefill は明示。
    let c8 = plan(8 * GIB, model_bytes(ModelQuant::Q4), ModelQuant::Q4);
    let p8 = plan_env_pairs(&c8);
    let get8 = |k: &str| {
        p8.iter()
            .find(|(n, _)| *n == k)
            .map(|(_, v)| v.as_str())
    };
    assert_eq!(get8("DENNO_EXPERT_GPU_MIN_HIT"), None);
    assert_eq!(get8("DENNO_EXPERT_RAM_MB"), Some("auto"));
    assert_eq!(get8("DENNO_PF_PREAD_THREADS"), Some("8"));
    assert_eq!(get8("DENNO_PF_CHUNK"), Some("1024"));
    // 値は必ず config のフィールドと一致する。
    let w = c8.gpu_weight_mb.to_string();
    let e = c8.expert_vram_mb.to_string();
    assert_eq!(get8("DENNO_GPU_WEIGHT_MB"), Some(w.as_str()));
    assert_eq!(get8("DENNO_EXPERT_VRAM_MB"), Some(e.as_str()));

    // expert を切るカードでは GPU フラグを出さない (`0` だけ書く)。
    let tiny = plan(2 * GIB, model_bytes(ModelQuant::Q4), ModelQuant::Q4);
    let pt = plan_env_pairs(&tiny);
    assert!(pt.iter().all(|(n, _)| *n != "DENNO_EXPERT_GPU"));
    assert!(pt
        .iter()
        .any(|(n, v)| *n == "DENNO_EXPERT_VRAM_MB" && v == "0"));
}

#[test]
fn gpu_budget_q8_row_bytes_boundaries() {
    assert_eq!(q8_0_row_bytes(0), None);
    assert_eq!(q8_0_row_bytes(1), None);
    assert_eq!(q8_0_row_bytes(31), None);
    assert_eq!(q8_0_row_bytes(32), Some(34));
    assert_eq!(q8_0_row_bytes(33), None);
    assert_eq!(q8_0_row_bytes(64), Some(68));
    // 4096 は 128 ブロック x 34 B。
    assert_eq!(q8_0_row_bytes(4096), Some(128 * 34));
    assert_eq!(q8_0_row_bytes(u64::MAX), None);
}

#[test]
fn gpu_budget_classify_weights_boundaries() {
    // (名前, 型, [k, n, 1, 1], nbytes)
    let t = |name: &'static str, ty: GgmlType, k: u64, n: u64, bytes: u64| {
        (name, ty, [k, n, 1, 1], bytes)
    };
    let totals = classify_weights([
        // デバイス常駐 (Q8_0) はそのまま加算。
        t("attn_q.weight", GgmlType::Q8_0, 4096, 8, 1000),
        // expert は型に関係なく expert 側 (VRAM を食わない)。
        t("ffn_gate_exps.weight", GgmlType::Q4K, 4096, 256, 9999),
        // IQ3_S はロード時に Q8_0 へ作り直される: 行長 x n で見積る。
        t("attn_k.weight", GgmlType::IQ3S, 4096, 2, 7),
        // F32 は mmap のまま (host)。
        t("hc_head.weight", GgmlType::F32, 512, 3, 500),
        // k が 32 の倍数でない量子型は host 側へ落ちる。
        t("odd.weight", GgmlType::IQ3S, 33, 2, 11),
    ]);
    assert_eq!(totals.expert_bytes, 9999);
    assert_eq!(totals.expert_tensors, 1);
    assert_eq!(totals.dev_bytes, 1000 + 128 * 34 * 2);
    assert_eq!(totals.dev_tensors, 2, "Q8_0 1 本 + IQ3S の Q8 見積り 1 本");
    assert_eq!(totals.host_bytes, 500 + 11);
    assert_eq!(totals.host_tensors, 2);
    assert_eq!(totals.max_k, 4096);
    assert_eq!(totals.max_n, 8, "Q8_0 と IQ3S の n の最大");

    // 空入力は全ゼロ。
    assert_eq!(
        classify_weights(Vec::<(&str, GgmlType, [u64; 4], u64)>::new()),
        WeightTotals::default()
    );

    // 行長 0 (k = 0) の量子型は host 側。
    let zero_k = classify_weights([t("z.weight", GgmlType::IQ4Nl, 0, 4, 64)]);
    assert_eq!(zero_k.dev_bytes, 0);
    assert_eq!(zero_k.host_bytes, 64);
}

#[test]
fn gpu_budget_flash_budget_boundaries() {
    // 空 totals: すべて 0、arena だけフラグで決まる。
    let none = flash_budget(WeightTotals::default(), 0, 0, 0, false);
    assert_eq!(none.weights_dev, 0);
    assert_eq!(none.staging_peak, 0);
    assert_eq!(none.kv_host_max, 0);
    assert_eq!(none.arena_dev, 0);
    let with_q4 = flash_budget(WeightTotals::default(), 0, 0, 0, true);
    assert_eq!(with_q4.arena_dev, SCRATCH_ARENA_CHUNK);

    // KV は full 層数 x ctx x kv_dim x 2 x 4。ctx を倍にすると倍。
    let totals = WeightTotals {
        dev_bytes: 1 << 20,
        dev_tensors: 1,
        max_k: 100,
        max_n: 20,
        ..WeightTotals::default()
    };
    let b1 = flash_budget(totals, 8, 128, 1024, false);
    assert_eq!(b1.kv_host_max, 8 * 1024 * 128 * 2 * 4);
    let b2 = flash_budget(totals, 8, 128, 2048, false);
    assert_eq!(b2.kv_host_max, b1.kv_host_max * 2);
    assert_eq!(b1.staging_peak, (100 + 20) * 4);
    assert_eq!(b1.weights_dev, 1 << 20);

    // 極端値は飽和して panic しない (負値・巻き戻りなし)。
    let sat = WeightTotals {
        dev_bytes: u64::MAX,
        dev_tensors: usize::MAX,
        max_k: u64::MAX,
        max_n: u64::MAX,
        ..WeightTotals::default()
    };
    let bs = flash_budget(sat, usize::MAX, usize::MAX, usize::MAX, true);
    assert_eq!(bs.staging_peak, u64::MAX);
    assert_eq!(bs.kv_host_max, u64::MAX);
    assert_eq!(bs.weights_dev, u64::MAX);
    assert_eq!(bs.arena_dev, SCRATCH_ARENA_CHUNK);
}

#[test]
fn gpu_budget_n_full_layers_boundaries() {
    assert_eq!(n_full_layers(0, 4), 0);
    assert_eq!(n_full_layers(64, 0), 0, "interval 0 は 0 除算にしない");
    assert_eq!(n_full_layers(1, 4), 1);
    assert_eq!(n_full_layers(4, 4), 1);
    assert_eq!(n_full_layers(5, 4), 2);
    assert_eq!(n_full_layers(61, 4), 16);
    // interval 1 は全層。
    assert_eq!(n_full_layers(61, 1), 61);
}

#[test]
fn gpu_budget_weight_lru_cap_is_deterministic() {
    // 上限 1000 B: 400 + 400 は収まり、3 本目で最古 (1) を落とす。
    let mut lru = WeightLru::new(1000);
    assert!(lru.note_insert(1, 400).is_empty());
    assert!(lru.note_insert(2, 400).is_empty());
    let v = lru.note_insert(3, 400);
    assert_eq!(v, vec![1], "最古の未使用を 1 本だけ落とす");
    assert!(!lru.contains(1));
    assert_eq!(lru.used_bytes(), 800);
    assert_eq!(lru.len(), 2);
    assert_eq!(lru.evicted_tensors(), 1);
    assert_eq!(lru.evicted_bytes(), 400);

    // 1 本だけで上限を超える登録は落とさない (上限は一時超過)。
    let mut big = WeightLru::new(100);
    assert!(big.note_insert(7, 250).is_empty());
    assert!(big.contains(7));
    assert_eq!(big.used_bytes(), 250);

    // cap 0 は無制限 = 1 本も落とさない。
    let mut free = WeightLru::new(0);
    for uid in 0..10 {
        assert!(free.note_insert(uid, 1 << 20).is_empty());
    }
    assert_eq!(free.len(), 10);
    assert_eq!(free.evicted_tensors(), 0);

    // pin 中の uid は退避しない。同じ入力なら同じ順で落ちる (決定性)。
    let mut pinned = WeightLru::new(1000);
    pinned.note_insert(1, 400);
    pinned.note_insert(2, 400);
    assert!(pinned.pin(1));
    assert_eq!(pinned.note_insert(3, 400), vec![2]);
    assert!(pinned.contains(1));
    pinned.unpin(1);
    // 予約 (`GPU_W_RESERVE_MB`) は 256 MiB。値そのものは公開定数。
    assert_eq!(GPU_W_RESERVE_MB, 256);
}
