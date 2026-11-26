//! task 1559: prefill の資源配分 (`pf_budget`) と作業域プール (`pf_pool`) の境界。
//!
//! 対象は公開 API の純関数だけで、`/proc/meminfo` も `DENNO_PF_*` も読まない
//! (env を触らないのでテスト間で干渉しない)。RAM は [`MemFacts`] を直接組み、
//! 16 / 64 / 128 GiB のホストを仮定する。GPU も実 RAM も見ないので、16 GiB の
//! CI ランナーでも同じ結果になる。
//!
//! 縛るもの:
//! - 段 (Healthy / Tight / Critical) の閾値と、ヒステリシスの復帰線。
//! - 配分が `max(匿名上限, 床)` を超えないこと、128 GiB 級の上限。
//! - 1K / 4K / 16K / 32K の作業域見積り (task 915) が実区画を覆うこと。
//! - `layout` / `borrow_plan` / `choose_big` / `pool_budget` の境界
//!   (上限ちょうどは収まる、1 バイト足りないと fail open)。
//!
//! CPU-only・高速。`cargo test --release -p denno-engine --test pf_budget_edge_1559`。

use denno_engine::model::pf_budget::{
    base_tuning, plan_for, plan_for_hyst, pressure_of, pressure_of_hyst, MemFacts, Pressure,
    MIN_POOL_MB_STRICT, MIN_RC_MB_STRICT,
};
use denno_engine::model::pf_pool::{
    big_candidates, borrow_plan, choose_big, fits, layout, lm_work_need, pool_budget, Layout, Need,
    R, DEFAULT_BIG_CHUNK, DEFAULT_POOL_BIG_MB, DEFAULT_POOL_MB, POOL_ALIGN,
};

/// 1 MiB (バイト)。
const MIB: usize = 1 << 20;
/// 1 GiB (MiB)。
const GIB_MIB: u64 = 1024;
/// 1 GiB (バイト)。
const GIB: usize = 1 << 30;

/// 検証する文脈長 (1K / 4K / 16K / 32K)。
const CONTEXTS: [usize; 4] = [1_024, 4_096, 16_384, 32_768];

/// 本番モデル相当の形 (e=2560 / k=10 / hc=4 / ff=1024 / chunk=1536)。
fn need_nt(nt: usize) -> Need {
    Need {
        nt_all: nt,
        tile: nt.min(5_120),
        k: 10,
        e: 2_560,
        ff: 1_024,
        sff: 1_024,
        hc: 4,
        lr: 256,
        chunk: 1_536,
        ne: 512,
        tc: true,
        stage_bytes: 512 * MIB,
        stage_slots: 3,
    }
}

/// swap に余裕があるホスト (swap 由来の Critical を起こさない)。
fn facts(total_gib: u64, avail_gib: u64) -> MemFacts {
    MemFacts {
        total_kb: total_gib * GIB_MIB * 1024,
        avail_kb: avail_gib * GIB_MIB * 1024,
        swap_total_kb: 8 * GIB_MIB * 1024,
        swap_free_kb: 8 * GIB_MIB * 1024,
    }
}

/// `lm_work_need` が覆う区画 (task 915 の対象。shared expert / staging / 表は含まない)。
const COVERED: [R; 14] = [
    R::LmMixed,
    R::LmFfn,
    R::LmInj,
    R::DevRes,
    R::DevXn,
    R::DevGate,
    R::DevLo,
    R::DevMixed,
    R::DevInj,
    R::DevBo,
    R::DevGamma,
    R::DevPark,
    R::H,
    R::Act,
];

/// 覆う区画の合計バイト数。
fn covered_bytes(l: &Layout) -> usize {
    COVERED.iter().map(|&r| l.region(r).bytes).sum()
}

/// 空きが十分な 64 GiB ホストでは `base_tuning` と 1 バイトも変わらない。
/// 16 GiB ホストの Healthy は本数・タイル・置き場が同じで、プールだけが
/// 匿名上限 (`avail * 60% / 4`) に切られる。
#[test]
fn edge_healthy_is_identity_on_16_and_64_gib() {
    let b = base_tuning();
    // 16 GiB: Healthy の床は 12 GiB (18% の 2.9 GiB より大きい)。
    let small = plan_for(facts(16, 13), b);
    assert_eq!(small.pressure, Pressure::Healthy);
    assert_eq!(small.tuning.pread_threads, b.pread_threads);
    assert_eq!(small.tuning.ra_threads, b.ra_threads);
    assert_eq!(small.tuning.rc_cache_mb, b.rc_cache_mb);
    assert_eq!(small.tuning.lm_tile, b.lm_tile);
    assert!(small.tuning.pool_mb < b.pool_mb, "16 GiB は匿名上限が効く");
    assert!(small.tuning.pool_mb >= 512);
    // 64 GiB の平常時。
    for avail in [30u64, 45, 60] {
        let p = plan_for(facts(64, avail), b);
        assert_eq!(p.pressure, Pressure::Healthy, "avail={avail} GiB");
        assert_eq!(p.tuning, b, "avail={avail} GiB で恒等");
    }
    // 1K-32K のどの文脈長でも、配分は文脈長を見ない (同じ入力なら同じ値)。
    for nt in CONTEXTS {
        let p = plan_for(facts(64, 60), b);
        assert_eq!(p.tuning, b, "nt={nt}");
    }
}

/// 16 / 64 / 128 GiB の段の閾値と単調性。
#[test]
fn edge_pressure_thresholds_across_host_sizes() {
    let cases: [(u64, u64, &str); 8] = [
        // 16 GiB: Healthy >= 12 GiB、Tight >= 6 GiB。
        (16, 12, "healthy"),
        (16, 11, "tight"),
        (16, 6, "tight"),
        (16, 5, "critical"),
        // 64 GiB: Healthy >= 12 GiB (18% = 11.5 GiB より床が大きい)。
        (64, 12, "healthy"),
        (64, 11, "tight"),
        (64, 5, "critical"),
        // 128 GiB: Healthy >= 23.04 GiB (18%) なので 23 GiB は Tight。
        (128, 23, "tight"),
    ];
    for (total, avail, want) in cases {
        assert_eq!(
            pressure_of(facts(total, avail)).as_str(),
            want,
            "total={total} avail={avail}"
        );
    }
    assert_eq!(pressure_of(facts(128, 24)).as_str(), "healthy");
}

/// 配分は「匿名上限 / 4 (プール)・/8 (置き場) を床で切り上げた値」を超えない。
///
/// 非厳格モードの床は 512 / 64 MiB、厳格モードは 128 / 32 MiB。
#[test]
fn edge_plan_respects_anonymous_cap_and_floors() {
    let b = base_tuning();
    for (total, avail) in [(16u64, 8u64), (16, 4), (64, 8), (64, 5), (128, 20)] {
        let m = facts(total, avail);
        let anon = (avail * 1024) * 60 / 100; // MiB
        let plan = plan_for(m, b);
        let pool_bound = (anon / 4).max(512) as usize;
        let rc_bound = (anon / 8).max(64) as usize;
        assert!(
            plan.tuning.pool_mb <= pool_bound,
            "total={total} avail={avail}: pool={} > bound={pool_bound}",
            plan.tuning.pool_mb
        );
        assert!(
            plan.tuning.rc_cache_mb <= rc_bound,
            "total={total} avail={avail}: rc={} > bound={rc_bound}",
            plan.tuning.rc_cache_mb
        );
        assert!(plan.tuning.pool_mb >= 512);
        assert!(plan.tuning.rc_cache_mb >= 64);

        // 厳格モードは cap を勝たせ、床だけ下げる (cap 自体は同じ式)。
        let strict = plan_for_hyst(m, b, None, true);
        let strict_pool_bound = (anon / 4).max(MIN_POOL_MB_STRICT as u64) as usize;
        let strict_rc_bound = (anon / 8).max(MIN_RC_MB_STRICT as u64) as usize;
        assert!(
            strict.tuning.pool_mb <= strict_pool_bound,
            "strict: pool={} > bound={strict_pool_bound}",
            strict.tuning.pool_mb
        );
        assert!(
            strict.tuning.rc_cache_mb <= strict_rc_bound,
            "strict: rc={} > bound={strict_rc_bound}",
            strict.tuning.rc_cache_mb
        );
        assert!(strict.tuning.pool_mb >= MIN_POOL_MB_STRICT);
        assert!(strict.tuning.rc_cache_mb >= MIN_RC_MB_STRICT);
        // 床が下がったぶん、厳格モードは非厳格より小さくなれる。
        assert!(strict.tuning.pool_mb <= plan.tuning.pool_mb);
        assert!(strict.tuning.rc_cache_mb <= plan.tuning.rc_cache_mb);
    }
}

/// 128 GiB 級で余った RAM を使う段の上限 (rc <= 2048 / tile <= 16384 / pool <= 4096)。
#[test]
fn edge_large_host_caps() {
    let b = base_tuning();
    let p = plan_for(facts(128, 100), b);
    assert_eq!(p.pressure, Pressure::Healthy);
    assert!(p.tuning.rc_cache_mb > b.rc_cache_mb);
    assert!(p.tuning.rc_cache_mb <= 2_048);
    assert!(p.tuning.lm_tile > b.lm_tile);
    assert!(p.tuning.lm_tile <= 16_384);
    assert!(p.tuning.pool_mb > b.pool_mb);
    assert!(p.tuning.pool_mb <= 4_096);
    assert_eq!(
        p.tuning.pread_threads, b.pread_threads,
        "pread の本数は lab t606b で 32 が負けたので上げない"
    );
}

/// ヒステリシスの復帰線: 降格は即時、復帰は閾値の 10% 外側。
#[test]
fn edge_hysteresis_recovery_lines() {
    let b = base_tuning();
    // 64 GiB: Healthy = 12 GiB、Tight = 6 GiB。復帰線は 13,516 / 6,758 MiB。
    let hyst = |avail_mib: u64, prev: Pressure| {
        let mut m = facts(64, 0);
        m.avail_kb = avail_mib * 1024;
        pressure_of_hyst(m, Some(prev))
    };
    // Healthy -> Tight は即時 (余裕を要求しない)。
    assert_eq!(hyst(12_287, Pressure::Healthy), Pressure::Tight);
    // Tight -> Healthy は 13,516 MiB 以上。
    assert_eq!(hyst(13_515, Pressure::Tight), Pressure::Tight);
    assert_eq!(hyst(13_516, Pressure::Tight), Pressure::Healthy);
    // Tight -> Critical は即時。
    assert_eq!(hyst(6_143, Pressure::Tight), Pressure::Critical);
    // Critical -> 復帰も同じ復帰線 (Tight は 6,758 MiB、Healthy は 13,516 MiB)。
    assert_eq!(hyst(6_757, Pressure::Critical), Pressure::Critical);
    assert_eq!(hyst(6_758, Pressure::Critical), Pressure::Tight);
    assert_eq!(hyst(13_516, Pressure::Critical), Pressure::Healthy);
    // prev が無いときは素の判定と同じ。
    assert_eq!(hyst(6_000, Pressure::Healthy), Pressure::Critical);
    let _ = plan_for(facts(64, 60), b);
}

/// 1K / 4K / 16K / 32K の作業域見積り (task 915) が実区画を覆い、過大でもない。
#[test]
fn edge_workspace_estimate_covers_all_contexts() {
    for nt in CONTEXTS {
        let n = need_nt(nt);
        let l = layout(&n).expect("レイアウトが組める");
        let regions = covered_bytes(&l);
        let need_b = lm_work_need(&n, 1);
        assert!(
            need_b >= regions,
            "見積りが実区画より小さい: nt={nt} need={} regions={}",
            need_b >> 20,
            regions >> 20
        );
        // H を nt_all 行で取るぶんの余裕 + 64 MiB の余白までは許す。
        let h_slack = (nt - n.tile) * n.e * 4;
        assert!(
            need_b <= regions + 64 * MIB + h_slack,
            "見積りが過大: nt={nt} need={} regions={}",
            need_b >> 20,
            regions >> 20
        );
        // 文脈長に対して単調に増える。
        if nt > CONTEXTS[0] {
            let prev = need_nt(nt / 2);
            assert!(
                lm_work_need(&n, 1) > lm_work_need(&prev, 1),
                "見積りが増えない: nt={nt}"
            );
        }
    }
}

/// 1K / 4K は既定のプール (2,688 MiB) に収まり、16K / 32K は収まらない。
/// 16K / 32K は大きい計画の上限 (4,864 MiB) に収まる。
#[test]
fn edge_pool_layout_boundaries_by_context() {
    let default_cap = DEFAULT_POOL_MB * MIB;
    let big_cap = DEFAULT_POOL_BIG_MB * MIB;
    let totals: Vec<usize> = CONTEXTS
        .iter()
        .map(|&nt| layout(&need_nt(nt)).expect("レイアウト").total)
        .collect();
    for (i, &nt) in CONTEXTS.iter().enumerate() {
        assert_eq!(
            fits(&layout(&need_nt(nt)).unwrap(), default_cap),
            nt <= 4_096,
            "既定のプールの境界: nt={nt} total={} MiB",
            totals[i] >> 20
        );
        assert!(
            fits(&layout(&need_nt(nt)).unwrap(), big_cap),
            "大きい上限に収まる: nt={nt} total={} MiB",
            totals[i] >> 20
        );
        if i > 0 {
            assert!(totals[i] > totals[i - 1], "文脈長で単調: nt={nt}");
        }
    }
    // 16K の作業域は既定のプールをわずかに超える (これが大きい計画の根拠)。
    let l16 = layout(&need_nt(16_384)).expect("レイアウト");
    assert!(l16.total > default_cap);
    assert!(l16.total <= big_cap);
}

/// 区画の先頭は 1 KiB に揃い、区画は重ならない。
#[test]
fn edge_layout_regions_are_aligned_and_disjoint() {
    for nt in CONTEXTS {
        let l = layout(&need_nt(nt)).expect("レイアウト");
        assert_eq!(l.total % POOL_ALIGN, 0, "合計が整列していない: nt={nt}");
        let mut prev_end = 0usize;
        for r in R::ALL {
            let reg = l.region(r);
            assert_eq!(reg.off % POOL_ALIGN, 0, "区画 {:?} の先頭: nt={nt}", r);
            assert!(
                reg.off >= prev_end,
                "区画 {:?} が前と重なる: nt={nt} off={} prev_end={prev_end}",
                r,
                reg.off
            );
            prev_end = reg.off + reg.bytes;
        }
        assert_eq!(prev_end, l.total, "最後の区画が合計に届く: nt={nt}");
    }
}

/// `borrow_plan` は上限ちょうどで借り、1 バイト足りないと fail open。
/// 最小トークン数も境界ちょうどで借りる。
#[test]
fn edge_borrow_plan_exact_cap_and_min_tokens() {
    for nt in CONTEXTS {
        let n = need_nt(nt);
        let total = layout(&n).expect("レイアウト").total;
        // 上限ちょうどは収まる。
        let ok = borrow_plan(&n, true, 1, total).expect("上限ちょうどは借りる");
        assert_eq!(ok.total, total);
        // 1 バイト足りないと借りない (従来の確保へ落ちる)。
        assert_eq!(borrow_plan(&n, true, 1, total - 1), None, "nt={nt}");
        // ゲート off は常に借りない。
        assert_eq!(borrow_plan(&n, false, 1, usize::MAX), None, "nt={nt}");
        // 最小トークン数の境界。
        assert!(borrow_plan(&n, true, nt, total).is_some(), "nt={nt}");
        assert_eq!(borrow_plan(&n, true, nt + 1, total), None, "nt={nt}");
    }
}

/// `choose_big` は 1K-32K すべてで大きい上限に収まる組を返し、0 バイトなら None。
#[test]
fn edge_choose_big_all_contexts() {
    let big_cap = DEFAULT_POOL_BIG_MB * MIB;
    for nt in CONTEXTS {
        let n = need_nt(nt);
        let (chunk, tile, l) =
            choose_big(&n, big_cap, DEFAULT_BIG_CHUNK).unwrap_or_else(|| panic!("nt={nt}"));
        assert!(l.total <= big_cap, "nt={nt} total={}", l.total >> 20);
        assert!(chunk >= 1 && chunk <= DEFAULT_BIG_CHUNK, "nt={nt} chunk={chunk}");
        assert!(tile >= 1 && tile <= nt, "nt={nt} tile={tile}");
        // 候補の並びに含まれ、それより前の候補はすべて上限を超える
        // (「最初に収まった組」を返す契約)。
        let cands = big_candidates(nt, n.chunk, n.tile, DEFAULT_BIG_CHUNK);
        let idx = cands
            .iter()
            .position(|&c| c == (chunk, tile))
            .unwrap_or_else(|| panic!("nt={nt} の候補に無い"));
        for &(c, t) in &cands[..idx] {
            let cand = Need { chunk: c, tile: t, ..n };
            if let Some(cl) = layout(&cand) {
                assert!(cl.total > big_cap, "nt={nt} 手前の候補が収まっている");
            }
        }
    }
    // 上限 0 では収まる組が無い (fail open)。
    assert_eq!(choose_big(&need_nt(1_024), 0, DEFAULT_BIG_CHUNK), None);
}

/// プール予算は `min(cap, 実空き + 追い出せる量 - 余白)` で飽和する。
#[test]
fn edge_pool_budget_boundaries() {
    let cap = DEFAULT_POOL_MB * MIB;
    let reserve = 1_024 * MIB;
    assert_eq!(pool_budget(cap, None, 0, reserve), cap, "実空き不明は cap");
    assert_eq!(pool_budget(cap, Some(8 * GIB), 0, reserve), cap);
    // 追い出せる量を足しても cap を超えない。
    assert_eq!(pool_budget(cap, Some(4 * GIB), 4 * GIB, reserve), cap);
    // 足りないときは 0 で飽和し、panic しない。
    assert_eq!(pool_budget(cap, Some(0), 0, reserve), 0);
    assert_eq!(pool_budget(cap, Some(1), 1, reserve), 0);
    assert_eq!(pool_budget(usize::MAX, Some(usize::MAX), 0, 1), usize::MAX - 1);
}

/// 異常値 (総 RAM 1 MiB / 空き 0 / 巨大値) でも panic せず範囲に収まる。
#[test]
fn edge_extreme_inputs_are_clamped() {
    let b = base_tuning();
    for m in [
        MemFacts {
            total_kb: 1,
            avail_kb: 0,
            swap_total_kb: 0,
            swap_free_kb: 0,
        },
        MemFacts {
            total_kb: u64::MAX,
            avail_kb: u64::MAX,
            swap_total_kb: 0,
            swap_free_kb: 0,
        },
    ] {
        let p = plan_for(m, b);
        assert!(p.tuning.pread_threads >= 4 && p.tuning.pread_threads <= 64);
        assert!(p.tuning.ra_threads >= 8 && p.tuning.ra_threads <= 32);
        assert!(p.tuning.rc_cache_mb >= 64 && p.tuning.rc_cache_mb <= 2_048);
        assert!(p.tuning.lm_tile >= 1_024);
        assert!(p.tuning.pool_mb >= 512 && p.tuning.pool_mb <= 8_192);
    }
    // 形が 0 のレイアウトは None (panic しない)。
    let base = need_nt(1_024);
    for bad in [
        Need { nt_all: 0, ..base },
        Need { tile: 0, ..base },
        Need { k: 0, ..base },
        Need { e: 0, ..base },
        Need { ff: 0, ..base },
    ] {
        assert_eq!(layout(&bad), None);
    }
}

/// 同じ入力なら同じ値 (決定性)。段・配分・レイアウト・見積りを 3 回ずつ比べる。
#[test]
fn edge_determinism() {
    let b = base_tuning();
    for (total, avail) in [(16u64, 8u64), (64, 60), (128, 100)] {
        let first = plan_for(facts(total, avail), b);
        for _ in 0..2 {
            assert_eq!(plan_for(facts(total, avail), b), first);
        }
    }
    for nt in CONTEXTS {
        let n = need_nt(nt);
        let first = layout(&n).expect("レイアウト");
        for _ in 0..2 {
            assert_eq!(layout(&n), Some(first));
            assert_eq!(lm_work_need(&n, 1), lm_work_need(&n, 1));
        }
    }
}
