//! task 1414: denno-engine のステアリング / abliteration の公開フックを
//! CPU だけで叩く境界テスト。
//!
//! 対象は `denno_engine::model::steer` の公開 API:
//! - [`steer::set_apply`] (単一規則) と [`steer::set_rules`] (重ね掛け)
//! - [`steer::Apply::apply_one`] / [`steer::Rule::apply_one`] の層範囲と形の扱い
//! - [`steer::import_snapshot`] の形・非有限値の検証
//!
//! 見る性質:
//! - `alpha = 0` は恒等 (ビット一致)。
//! - NaN / Inf の alpha・coef・平均は弾く。
//! - 空の terms / rules と、モデルと違う形のスナップショットは弾く。
//! - 重ね掛けは規則の順に 1 個ずつ掛けた結果とビット一致する。
//! - project は `v_hat` 方向の成分を落とす (直交成分は保存、2 回目は不変)。
//! - 層範囲は `[from, to)` (`from` は含み `to` は含まない)。
//! - 同じ入力・同じ設定なら出力はビット一致 (決定的)。
//!
//! グローバル (スナップショット / 適用状態 / env) を触るので、各テストは
//! 1 本の Mutex で直列化し、毎回 [`steer::reset_for_test`] から始める。

use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

use denno_engine::model::steer::{self, Apply, CaptureMode, Mode, RuleSpec, Shape, Snapshot, Term};

/// グローバルを触るテストの直列化。
static LOCK: Mutex<()> = Mutex::new(());

fn shape() -> Shape {
    Shape {
        n_layer: 4,
        n_hc: 2,
        n_embd: 4,
    }
}

fn test_dir(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("denno-steer-1414-{}-{tag}", std::process::id()))
}

/// ロックを取り、スナップショットの保存先を空にして、形を初期化する。
fn setup(tag: &str) -> MutexGuard<'static, ()> {
    let g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = test_dir(tag);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("一時ディレクトリ");
    std::env::set_var("DENNO_STEER_DIR", &dir);
    steer::reset_for_test();
    steer::init(shape(), "test-model");
    g
}

/// `[layer][stream][n_embd]` の並びで平均を作る。
fn mean_of(f: impl Fn(usize, usize, usize) -> f32) -> Vec<f32> {
    let sh = shape();
    let mut v = Vec::with_capacity(sh.len());
    for il in 0..sh.n_layer {
        for s in 0..sh.n_hc {
            for d in 0..sh.n_embd {
                v.push(f(il, s, d));
            }
        }
    }
    v
}

fn snapshot(name: &str, mean: Vec<f32>) -> Snapshot {
    let sh = shape();
    Snapshot {
        name: name.to_string(),
        tokens: 1,
        created: 1,
        model: String::new(),
        layers: sh.n_layer,
        n_hc: sh.n_hc,
        n_embd: sh.n_embd,
        mean,
    }
}

/// 形が合うスナップショットを入れて返す。
fn install(name: &str, f: impl Fn(usize, usize, usize) -> f32) -> Snapshot {
    steer::import_snapshot(snapshot(name, mean_of(f)), None).expect("スナップショットを入れられる")
}

fn term(name: &str, coef: f32) -> Term {
    Term {
        snapshot: name.to_string(),
        coef,
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn assert_approx(a: f32, b: f32, eps: f32, what: &str) {
    assert!(
        (a - b).abs() <= eps,
        "{what}: {a} と {b} の差が {eps} を超える"
    );
}

/// 1 トークンぶんの残差。
fn data() -> Vec<f32> {
    let st = shape().stride();
    (0..st).map(|i| 0.25 + 0.5 * i as f32).collect()
}

fn single(apply: &Apply, layer: usize, input: &[f32]) -> Vec<f32> {
    let mut d = input.to_vec();
    apply.apply_one(layer, &mut d);
    d
}

/// `v` / `v_hat` から (layer, stream) の成分を取り出す。
fn vhat_slice(apply: &Apply, layer: usize, stream: usize) -> Vec<f32> {
    let sh = shape();
    let base = layer * sh.stride() + stream * sh.n_embd;
    apply.vhat[base..base + sh.n_embd].to_vec()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[test]
fn alpha_zero_is_bit_identical() {
    let _g = setup("alpha0");
    install("a", |il, s, d| 0.1 * (il as f32 + 1.0) + 0.7 * s as f32 + 0.3 * d as f32);
    let h = data();

    for mode in [Mode::Subtract, Mode::Project] {
        steer::set_apply(vec![term("a", 1.0)], 0, shape().n_layer, 0.0, mode)
            .expect("alpha 0 は通る");
        let apply = steer::snapshot().apply.clone().expect("適用中");
        for il in 0..shape().n_layer {
            let out = single(&apply, il, &h);
            assert_eq!(bits(&out), bits(&h), "{mode:?} の層 {il} が alpha 0 で動いた");
        }
    }
}

#[test]
fn alpha_zero_with_rules_is_bit_identical() {
    let _g = setup("alpha0_rules");
    install("a", |_, s, d| 0.2 + 0.4 * s as f32 + 0.1 * d as f32);
    install("b", |il, _, d| 0.9 - 0.2 * il as f32 + 0.05 * d as f32);
    let h = data();

    steer::set_rules(vec![
        RuleSpec {
            terms: vec![term("a", 1.5)],
            from: 0,
            to: 2,
            alpha: 0.0,
            mode: Mode::Subtract,
        },
        RuleSpec {
            terms: vec![term("b", -0.75)],
            from: 1,
            to: 4,
            alpha: 0.0,
            mode: Mode::Project,
        },
    ])
    .expect("alpha 0 の重ね掛けは通る");
    let apply = steer::snapshot().apply.clone().expect("適用中");
    for il in 0..shape().n_layer {
        let out = single(&apply, il, &h);
        assert_eq!(bits(&out), bits(&h), "層 {il} が alpha 0 で動いた");
    }
}

#[test]
fn non_finite_alpha_and_coef_are_rejected() {
    let _g = setup("nonfinite");
    install("a", |il, _, d| 0.1 * il as f32 + 0.2 * d as f32);

    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let e = steer::set_apply(vec![term("a", 1.0)], 0, 4, bad, Mode::Subtract)
            .expect_err("非有限の alpha は弾く");
        assert!(!e.is_empty());

        let e = steer::set_rules(vec![RuleSpec {
            terms: vec![term("a", 1.0)],
            from: 0,
            to: 4,
            alpha: bad,
            mode: Mode::Project,
        }])
        .expect_err("非有限の規則 alpha は弾く");
        assert!(!e.is_empty());
    }

    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(
            steer::set_apply(vec![term("a", bad)], 0, 4, 1.0, Mode::Subtract).is_err(),
            "非有限の coef は弾く"
        );
        assert!(
            steer::set_rules(vec![RuleSpec {
                terms: vec![term("a", bad)],
                from: 0,
                to: 4,
                alpha: 1.0,
                mode: Mode::Subtract,
            }])
            .is_err(),
            "非有限の規則 coef は弾く"
        );
    }
}

#[test]
fn non_finite_mean_is_rejected_on_import() {
    let _g = setup("nonfinite_mean");
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut m = mean_of(|_, _, d| 1.0 + d as f32);
        m[shape().stride() + 2] = bad;
        assert!(
            steer::import_snapshot(snapshot("bad", m), None).is_err(),
            "非有限の平均 {bad} は弾く"
        );
    }
    assert!(steer::snapshots().iter().all(|s| s.name != "bad"));
}

#[test]
fn empty_terms_and_empty_rules_are_rejected() {
    let _g = setup("empty");
    install("a", |_, _, d| 0.3 + d as f32);

    let e = steer::set_apply(Vec::new(), 0, 4, 1.0, Mode::Subtract)
        .expect_err("空の terms は弾く");
    assert!(!e.is_empty());

    let e = steer::set_rules(Vec::new()).expect_err("空の rules は弾く");
    assert!(!e.is_empty());

    // 規則の中の terms が空でも弾く。
    assert!(
        steer::set_rules(vec![RuleSpec {
            terms: Vec::new(),
            from: 0,
            to: 4,
            alpha: 1.0,
            mode: Mode::Subtract,
        }])
        .is_err()
    );

    // 存在しないスナップショット名も弾く。
    assert!(steer::set_apply(vec![term("missing", 1.0)], 0, 4, 1.0, Mode::Subtract).is_err());
}

#[test]
fn mismatched_snapshot_dimensions_are_rejected() {
    let _g = setup("dims");
    let sh = shape();

    // 層数が違う。
    let mut s = snapshot("wrong_layers", vec![0.0; sh.len()]);
    s.layers = sh.n_layer - 1;
    assert!(steer::import_snapshot(s, None).is_err());

    // ストリーム数が違う。
    let mut s = snapshot("wrong_hc", vec![0.0; sh.len()]);
    s.n_hc = sh.n_hc + 1;
    assert!(steer::import_snapshot(s, None).is_err());

    // 埋め込み次元が違う。
    let mut s = snapshot("wrong_embd", vec![0.0; sh.len()]);
    s.n_embd = sh.n_embd + 1;
    assert!(steer::import_snapshot(s, None).is_err());

    // メタは合うが本体の長さが足りない。
    let mut s = snapshot("short_body", vec![0.0; sh.len()]);
    s.mean.pop();
    assert!(steer::import_snapshot(s, None).is_err());

    // どれも一覧に入っていない。
    let names: Vec<String> = steer::snapshots().iter().map(|s| s.name.clone()).collect();
    for n in ["wrong_layers", "wrong_hc", "wrong_embd", "short_body"] {
        assert!(!names.iter().any(|x| x == n), "{n} が入ってしまった");
    }
}

#[test]
fn multi_rule_composition_matches_sequential_application() {
    let _g = setup("compose");
    install("a", |il, s, d| 0.1 * (il as f32 + 1.0) + 0.4 * s as f32 + 0.2 * d as f32);
    install("b", |il, s, d| 0.8 - 0.15 * il as f32 + 0.3 * s as f32 - 0.1 * d as f32);

    steer::set_rules(vec![
        RuleSpec {
            terms: vec![term("a", 0.25)],
            from: 1,
            to: 3,
            alpha: 0.25,
            mode: Mode::Subtract,
        },
        RuleSpec {
            terms: vec![term("b", 0.5)],
            from: 2,
            to: 4,
            alpha: 0.5,
            mode: Mode::Project,
        },
    ])
    .expect("重ね掛けを設定できる");
    let apply = steer::snapshot().apply.clone().expect("適用中");

    assert!(!apply.is_single_rule());
    assert_eq!(apply.rules.len(), 2);
    assert_eq!((apply.from, apply.to), (1, 4), "包みは全規則の外側");
    assert_eq!(apply.terms.len(), 2, "terms は全規則の連結");
    assert_eq!((apply.alpha, apply.mode), (0.25, Mode::Subtract), "先頭規則の値");

    let h = data();
    for il in 0..shape().n_layer {
        let got = single(&apply, il, &h);
        let mut want = h.clone();
        for r in &apply.rules {
            r.apply_one(il, &mut want, shape().stride(), shape().n_embd, shape().n_hc);
        }
        assert_eq!(bits(&got), bits(&want), "層 {il} の合成が逐次適用と違う");
    }
}

#[test]
fn multi_rule_layer_targeting_is_per_rule() {
    let _g = setup("rules_layers");
    install("a", |il, s, d| 0.5 + 0.2 * il as f32 + 0.3 * s as f32 + 0.1 * d as f32);
    install("b", |il, s, d| 0.4 - 0.1 * il as f32 + 0.6 * s as f32 + 0.2 * d as f32);

    steer::set_rules(vec![
        RuleSpec {
            terms: vec![term("a", 1.0)],
            from: 1,
            to: 3,
            alpha: 0.5,
            mode: Mode::Subtract,
        },
        RuleSpec {
            terms: vec![term("b", 1.0)],
            from: 2,
            to: 4,
            alpha: 0.5,
            mode: Mode::Project,
        },
    ])
    .expect("重ね掛けを設定できる");
    let apply = steer::snapshot().apply.clone().expect("適用中");
    let h = data();

    // covers は規則の和。
    for (il, want) in [(0, false), (1, true), (2, true), (3, true), (4, false)] {
        assert_eq!(apply.covers(il), want, "covers({il})");
    }
    // 規則の端は `[from, to)`。
    let r0 = &apply.rules[0];
    let r1 = &apply.rules[1];
    for (il, want) in [(0, false), (1, true), (2, true), (3, false)] {
        assert_eq!(r0.covers(il), want, "規則 0 の covers({il})");
    }
    for (il, want) in [(1, false), (2, true), (3, true), (4, false)] {
        assert_eq!(r1.covers(il), want, "規則 1 の covers({il})");
    }

    // 層 0 は範囲外で不変。
    assert_eq!(bits(&single(&apply, 0, &h)), bits(&h));
    // 層 1 は規則 0 だけ。
    let mut only0 = h.clone();
    r0.apply_one(1, &mut only0, shape().stride(), shape().n_embd, shape().n_hc);
    assert_eq!(bits(&single(&apply, 1, &h)), bits(&only0));
    // 層 3 は規則 1 だけ。
    let mut only1 = h.clone();
    r1.apply_one(3, &mut only1, shape().stride(), shape().n_embd, shape().n_hc);
    assert_eq!(bits(&single(&apply, 3, &h)), bits(&only1));
    // 層 2 は両方 (順序どおり)。
    let mut both = h.clone();
    r0.apply_one(2, &mut both, shape().stride(), shape().n_embd, shape().n_hc);
    r1.apply_one(2, &mut both, shape().stride(), shape().n_embd, shape().n_hc);
    assert_eq!(bits(&single(&apply, 2, &h)), bits(&both));
}

#[test]
fn single_rule_layer_boundaries() {
    let _g = setup("bounds");
    install("a", |il, s, d| 0.2 * (il as f32 + 1.0) + 0.5 * s as f32 + 0.1 * d as f32);
    let h = data();
    let st = shape().stride();

    steer::set_apply(vec![term("a", 1.0)], 1, 3, 1.0, Mode::Subtract).expect("境界は通る");
    let apply = steer::snapshot().apply.clone().expect("適用中");
    let v = apply.v.clone();

    for il in [0usize, 3] {
        assert_eq!(bits(&single(&apply, il, &h)), bits(&h), "層 {il} は範囲外");
        assert!(!apply.covers(il));
    }
    for il in [1usize, 2] {
        assert!(apply.covers(il));
        let base = il * st;
        let mut want = h.clone();
        for d in 0..st {
            want[d] -= v[base + d];
        }
        assert_eq!(bits(&single(&apply, il, &h)), bits(&want), "層 {il} の subtract");
    }
    // 範囲外の `v` は 0。
    assert!(v[..st].iter().all(|&x| x == 0.0));
    assert!(v[3 * st..].iter().all(|&x| x == 0.0));

    // `to == n_layer` は許す。
    steer::set_apply(vec![term("a", 1.0)], 2, shape().n_layer, 1.0, Mode::Subtract)
        .expect("to == n_layer は通る");
    // `to > n_layer` は弾く。
    assert!(steer::set_apply(vec![term("a", 1.0)], 0, shape().n_layer + 1, 1.0, Mode::Subtract).is_err());
    // `from > to` は弾く。
    assert!(steer::set_apply(vec![term("a", 1.0)], 3, 2, 1.0, Mode::Subtract).is_err());

    // `from == to` は空範囲: 全層で不変。
    steer::set_apply(vec![term("a", 1.0)], 2, 2, 1.0, Mode::Project).expect("空範囲は通る");
    let apply = steer::snapshot().apply.clone().expect("適用中");
    for il in 0..shape().n_layer {
        assert!(!apply.covers(il), "空範囲の covers({il})");
        assert_eq!(bits(&single(&apply, il, &h)), bits(&h), "空範囲の層 {il}");
    }
}

#[test]
fn projection_removes_component_and_is_idempotent() {
    let _g = setup("project");
    install("a", |il, s, d| {
        0.3 + 0.7 * (il as f32) + 1.1 * s as f32 + 0.4 * (d as f32 + 1.0)
    });
    steer::set_apply(vec![term("a", 1.0)], 0, shape().n_layer, 1.0, Mode::Project)
        .expect("project を設定できる");
    let apply = steer::snapshot().apply.clone().expect("適用中");
    let sh = shape();
    let h = data();

    for il in 0..sh.n_layer {
        for s in 0..sh.n_hc {
            let vh = vhat_slice(&apply, il, s);
            let norm: f32 = dot(&vh, &vh).sqrt();
            assert_approx(norm, 1.0, 1e-5, "v_hat のノルム");

            let once = single(&apply, il, &h);
            let hs = &once[s * sh.n_embd..(s + 1) * sh.n_embd];
            assert_approx(dot(hs, &vh), 0.0, 1e-5, "射影後に v_hat 方向が残る");

            // 直交成分 (e0 - vh[0] * vh) は保存される。
            let mut w = vec![0.0f32; sh.n_embd];
            w[0] = 1.0;
            for d in 0..sh.n_embd {
                w[d] -= vh[0] * vh[d];
            }
            assert_approx(dot(&w, &vh), 0.0, 1e-5, "w は v_hat と直交");
            let hin = &h[s * sh.n_embd..(s + 1) * sh.n_embd];
            assert_approx(dot(hs, &w), dot(hin, &w), 1e-4, "直交成分が保存されない");

            // 2 回目は不変 (冪等)。
            let twice = single(&apply, il, &once);
            let hs2 = &twice[s * sh.n_embd..(s + 1) * sh.n_embd];
            for d in 0..sh.n_embd {
                assert_approx(hs2[d], hs[d], 1e-5, "射影が冪等でない");
            }
        }
    }
}

#[test]
fn normalization_is_per_stream_and_zero_stays_zero() {
    let _g = setup("normalize");
    // 層 2 のストリーム 1 だけ 0 にする。
    install("a", |il, s, d| {
        if il == 2 && s == 1 {
            0.0
        } else {
            0.5 + 0.25 * il as f32 + 0.75 * s as f32 + 0.1 * d as f32
        }
    });
    let sh = shape();
    let st = sh.stride();

    steer::set_apply(vec![term("a", 1.0)], 1, 3, 1.0, Mode::Project).expect("設定できる");
    let apply = steer::snapshot().apply.clone().expect("適用中");

    // 範囲外は 0。
    assert!(apply.vhat[..st].iter().all(|&x| x == 0.0));
    assert!(apply.vhat[3 * st..].iter().all(|&x| x == 0.0));
    // 0 ベクトルの stream も 0 のまま (正規化しない)。
    let zero_base = 2 * st + sh.n_embd;
    assert!(apply.vhat[zero_base..zero_base + sh.n_embd]
        .iter()
        .all(|&x| x == 0.0));

    // 0 ベクトルの stream は project しても動かない。
    let h = data();
    let out = single(&apply, 2, &h);
    assert_eq!(
        bits(&out[sh.n_embd..2 * sh.n_embd]),
        bits(&h[sh.n_embd..2 * sh.n_embd]),
        "0 ベクトルの stream が動いた"
    );
}

#[test]
fn outputs_are_deterministic_across_builds() {
    let _g = setup("determinism");
    install("a", |il, s, d| 0.13 * (il as f32 + 1.0) + 0.29 * s as f32 + 0.07 * d as f32);
    install("b", |il, s, d| 0.71 - 0.11 * il as f32 + 0.37 * s as f32 - 0.03 * d as f32);
    let h = data();

    let build = || {
        steer::set_rules(vec![
            RuleSpec {
                terms: vec![term("a", 0.5), term("b", -0.25)],
                from: 0,
                to: 3,
                alpha: 0.35,
                mode: Mode::Subtract,
            },
            RuleSpec {
                terms: vec![term("b", 2.0)],
                from: 1,
                to: 4,
                alpha: 0.65,
                mode: Mode::Project,
            },
        ])
        .expect("設定できる");
        steer::snapshot().apply.clone().expect("適用中")
    };

    let a1 = build();
    let a2 = build();
    assert_eq!(bits(&a1.v), bits(&a2.v), "v がビルド間で違う");
    assert_eq!(bits(&a1.vhat), bits(&a2.vhat), "v_hat がビルド間で違う");

    for il in 0..shape().n_layer {
        let o1 = single(&a1, il, &h);
        let o2 = single(&a2, il, &h);
        assert_eq!(bits(&o1), bits(&o2), "層 {il} の出力がビルド間で違う");
        // 同じ適用を繰り返しても同じ。
        let o3 = single(&a1, il, &h);
        assert_eq!(bits(&o1), bits(&o3), "層 {il} の出力が呼び出し間で違う");
    }
}

#[test]
fn short_or_out_of_range_data_is_untouched() {
    let _g = setup("short");
    install("a", |il, s, d| 0.4 + 0.2 * il as f32 + 0.6 * s as f32 + 0.15 * d as f32);
    steer::set_apply(vec![term("a", 1.0)], 1, 3, 1.0, Mode::Project).expect("設定できる");
    let apply = steer::snapshot().apply.clone().expect("適用中");

    // 形 (stride) より短いバッファには何もしない。
    let mut short = vec![0.5f32; shape().stride() - 1];
    let before = short.clone();
    apply.apply_one(1, &mut short);
    assert_eq!(bits(&short), bits(&before));

    // 範囲外の層には何もしない。
    let mut exact = data();
    let before = exact.clone();
    apply.apply_one(shape().n_layer, &mut exact);
    assert_eq!(bits(&exact), bits(&before));

    // Rule::apply_one も同じ (st = 0 は何もしない)。
    install("b", |_, _, d| 0.3 + 0.2 * d as f32);
    steer::set_apply(vec![term("b", 1.0)], 0, 4, 1.0, Mode::Subtract).expect("設定できる");
    let apply = steer::snapshot().apply.clone().expect("適用中");
    let rule = &apply.rules;
    assert!(rule.is_empty(), "単一規則は rules が空");
    let mut d = data();
    let before = d.clone();
    apply.apply_one(0, &mut d);
    assert_ne!(bits(&d), bits(&before), "有効な層は動くはず");
}

#[test]
fn shape_and_mode_helpers() {
    let s = shape();
    assert!(s.is_valid());
    assert_eq!(s.stride(), s.n_hc * s.n_embd);
    assert_eq!(s.len(), s.n_layer * s.stride());
    assert!(!s.is_empty());

    let bad = [
        Shape { n_layer: 0, ..s },
        Shape { n_layer: 1025, ..s },
        Shape { n_hc: 0, ..s },
        Shape { n_hc: 33, ..s },
        Shape { n_embd: 0, ..s },
        Shape { n_embd: 65537, ..s },
    ];
    for b in bad {
        assert!(!b.is_valid(), "{b:?} は不正のはず");
    }
    assert!(Shape { n_layer: 1024, n_hc: 32, n_embd: 65536 }.is_valid());
    assert!(Shape::default().is_empty());

    for m in [Mode::Subtract, Mode::Project] {
        assert_eq!(Mode::parse(m.name()), Some(m));
    }
    assert_eq!(Mode::parse("bogus"), None);

    for c in [CaptureMode::All, CaptureMode::LastPromptToken] {
        assert_eq!(CaptureMode::parse(c.name()), Some(c));
    }
    assert_eq!(CaptureMode::parse("bogus"), None);
}
