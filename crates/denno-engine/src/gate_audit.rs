//! task 213: 既定 on のゲートが必要とする PTX の台帳と、起動時の監査。
//!
//! 背景 (REPORT_P204): `DENNO_MIX_DEC_FOLD` と `DENNO_MIX_DEC_UPW` は既定 on なのに
//! その PTX (`kernels_mix_dev2.ptx` / `kernels_mix_up.ptx`) が int-auto に入っておらず、
//! decode は**静かに**従来経路 (mix 5 起動/call) を走っていた。台帳の起動数でしか
//! 気づけなかった。同じ形の事故 (既定 on のゲート + 欠けた PTX + 無言のフォールバック)
//! を起動時に 1 行で見えるようにする。
//!
//! - [`DEFAULT_ON_GATES`] が「既定 on のゲート → 必要な PTX とエントリ」の唯一の台帳。
//! - [`audit_lines`] を serve の起動時に呼ぶ。PTX が無い / エントリが無いゲートは
//!   1 行の WARN になる (起動は止めない。フォールバック自体は従来どおり動く)。
//! - `tests/kernel_invariants.rs` が同じ台帳をテキストとして読み、PTX とエントリの
//!   実在を検査する (cuda feature 無しでも走る)。
//!
//! 台帳の各行は**既定で on のゲート**だけを持つ。既定 off のゲート (例:
//! `DENNO_MIX_DEC_UPW` (legacy。task 213 の lab で採否 no)、`DENNO_PF_MIX_INJ`) は、
//! 明示的に on にしたときだけ PTX が要るのでここには入れない。`on` は実際の
//! ゲート関数を指すので、既定を変えたのに台帳を直し忘れると `--lib` の監査テストが落ちる。
//!
//! task 221 で `DENNO_MIX_DEC_FOLD` と `DENNO_DEC_MIX_UPW` が既定 on になったので
//! 2 行足した。`DENNO_DEC_EXPGU` も既定 on だが、使うのは `DENNO_EXPERT_BATCH` と
//! 同じ `kernels_moe_batch.ptx` (同じ 4 エントリ) なので行を分けない — ファイルが
//! 欠ければ EXPERT_BATCH の行が WARN を出す。

use std::path::{Path, PathBuf};

/// 1 行 = ゲート 1 本が必要とする PTX。
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct GatePtx {
    /// ゲートの env 名 (人が読むためと、WARN の識別子)。
    pub env: &'static str,
    /// `kernels/` 配下の PTX ファイル名。
    pub ptx: &'static str,
    /// PTX に必ず居るエントリ名 (ローダが `module_get_function` で引く名前)。
    pub entries: &'static [&'static str],
    /// `kernels/` の代わりに PTX を差し替える env (`DENNO_*_PTX`)。空なら無し。
    pub path_env: &'static str,
    /// 実際のゲート関数 (既定 on のもの)。
    pub on: fn() -> bool,
}

/// 既定 on のゲートと、そのゲートが使う PTX の台帳。
///
/// 並びは decode → prefill → GDN → その他の順。同じ PTX を複数のゲートが使う
/// ことがある (`kernels_dec_epi.ptx` は和と shared の 2 本)。
///
/// cuda feature 無しではゲート関数が存在しないので、台帳ごと `cfg` で外す
/// (`audit_lines` は空を返す)。台帳のテキストは kernel_invariants が読むので、
/// feature の有無によらずリポジトリには残る。
#[cfg(feature = "cuda")]
pub const DEFAULT_ON_GATES: &[GatePtx] = &[
    GatePtx {
        env: "DENNO_FULL_DEC_DEV",
        ptx: "kernels_dec_dev.ptx",
        entries: &["dno_dec_qknorm_rope", "dno_dec_gate_mul"],
        path_env: "DENNO_DEC_DEV_PTX",
        on: || crate::cuda::dec_full_dev::on(),
    },
    GatePtx {
        env: "DENNO_MIX_DEC_DEV",
        ptx: "kernels_mix_dev.ptx",
        entries: &["dno_mix_norm", "dno_mix_inj"],
        path_env: "DENNO_MIX_DEV_PTX",
        on: crate::cuda::mix_dec_dev_enabled,
    },
    GatePtx {
        env: "DENNO_MIX_DEC_FOLD",
        ptx: "kernels_mix_dev2.ptx",
        entries: &["dno_mix_norm_inj"],
        path_env: "DENNO_MIX_DEV2_PTX",
        on: crate::cuda::mix_dec_fold_on,
    },
    GatePtx {
        env: "DENNO_DEC_MIX_UPW",
        ptx: "kernels_mix_up.ptx",
        entries: &["dno_mix_up_bf16"],
        path_env: "DENNO_MIX_UP_PTX",
        on: crate::cuda::dec_mix_upw_on,
    },
    GatePtx {
        env: "DENNO_ATTN_FLASHDEC",
        ptx: "kernels_attn_dec.ptx",
        entries: &["dno_attn_flashdec", "dno_attn_flashdec_combine"],
        path_env: "DENNO_ATTN_FLASHDEC_PTX",
        on: crate::cuda::attn_flashdec_enabled,
    },
    GatePtx {
        env: "DENNO_GDN_DEC_GEMV",
        ptx: "kernels_moe_v4.ptx",
        entries: &["dno_gemv_q8_0_v4", "dno_gemv_q4k_v4"],
        path_env: "DENNO_GEMV_V4_PTX",
        on: crate::cuda::gdn_dec_gemv_v4_on,
    },
    GatePtx {
        env: "DENNO_GDN_DEC_GEMV",
        ptx: "kernels_moe_v5.ptx",
        entries: &["dno_gemv_q8_0_v5", "dno_gemv_q4k_v5"],
        path_env: "DENNO_GEMV_V5_PTX",
        on: crate::cuda::gdn_dec_gemv_v5_on,
    },
    GatePtx {
        env: "DENNO_GDN_DEC_GEMV_Q5K",
        ptx: "kernels_moe_v5_q5k.ptx",
        entries: &["dno_gemv_q5k_v5"],
        path_env: "DENNO_GEMV_V5_Q5K_PTX",
        on: crate::cuda::gdn_dec_gemv_q5k_on,
    },
    GatePtx {
        env: "DENNO_EXPERT_BATCH",
        ptx: "kernels_moe_batch.ptx",
        entries: &[
            "dno_moe_gemv_seg_q8_0",
            "dno_moe_gemv_seg_q5_1",
            "dno_moe_gemv_seg_q4k",
            "dno_moe_gemv_seg_q5k",
        ],
        path_env: "DENNO_MOE_BATCH_PTX",
        on: crate::model::expert_batch::batch_enabled,
    },
    GatePtx {
        env: "DENNO_DEC_EXP_SUMDEV",
        ptx: "kernels_dec_epi.ptx",
        entries: &["dno_dec_exp_sum"],
        path_env: "DENNO_DEC_EPI_PTX",
        on: crate::model::qwen4exp::dec_exp_sumdev_enabled,
    },
    GatePtx {
        env: "DENNO_DEC_SHARED_DEV",
        ptx: "kernels_dec_epi.ptx",
        entries: &["dno_dec_silu_mul"],
        path_env: "DENNO_DEC_EPI_PTX",
        on: crate::model::qwen4exp::dec_shared_dev_enabled,
    },
    GatePtx {
        env: "DENNO_GDN_SCAN2",
        ptx: "kernels_gdn_scan2.ptx",
        entries: &["dno_gdn_scan_sig2"],
        path_env: "DENNO_GDN_SCAN2_PTX",
        on: crate::cuda::gdn_scan2_on,
    },
    GatePtx {
        env: "DENNO_GDN_PF_FAST",
        ptx: "kernels_gdn_pf.ptx",
        entries: &["dno_gdn_pf_scan128"],
        path_env: "DENNO_GDN_PF_PTX",
        on: crate::cuda::gdn_pf_fast_on,
    },
    GatePtx {
        env: "DENNO_PF_SCAN_FAST",
        ptx: "kernels_gdn_pf2.ptx",
        entries: &["dno_gdn_pf_scan64", "dno_gdn_pf_norm128"],
        path_env: "DENNO_GDN_PF2_PTX",
        on: crate::cuda::gdn_pf2_on,
    },
    GatePtx {
        env: "DENNO_PF_FULL_DEV",
        ptx: "kernels_pf2.ptx",
        entries: &["dno_pf2_qknorm_rope", "dno_pf2_gate_mul"],
        path_env: "DENNO_PF2_PTX",
        on: crate::model::qwen4exp::pf_full_dev_on,
    },
    GatePtx {
        env: "DENNO_PF_FULL_FUSED",
        ptx: "kernels_pf3.ptx",
        entries: &["dno_pf3_gate_mul", "dno_pf3_gather"],
        path_env: "DENNO_PF3_PTX",
        on: crate::model::qwen4exp::pf_full_fused_on,
    },
    GatePtx {
        env: "DENNO_PF_TC_GEMM",
        ptx: "kernels_gemm_tc.ptx",
        entries: &["dno_gemm_tc_q8_0_w"],
        path_env: "DENNO_PF_TC_PTX",
        on: crate::cuda::pf_tc_gemm_on,
    },
    GatePtx {
        env: "DENNO_PF_TC_GEMM2",
        ptx: "kernels_gemm_tc2.ptx",
        entries: &["dno_gemm_tc2_q8_0_w", "dno_gemm_tc2_q4k_w", "dno_gemm_tc2_q5k_w"],
        path_env: "DENNO_PF_TC2_PTX",
        on: crate::cuda::pf_tc2_gemm_on,
    },
    GatePtx {
        env: "DENNO_PF_TC_SMALLN",
        ptx: "kernels_gemm_sn.ptx",
        entries: &["dno_gemm_sn_q8_0_w", "dno_gemm_sn_q4k_w"],
        path_env: "DENNO_PF_SN_PTX",
        on: crate::cuda::pf_tc_smalln_on,
    },
    GatePtx {
        env: "DENNO_PF_TC_SHARED2",
        ptx: "kernels_gemm_sn2.ptx",
        entries: &["dno_gemm_sn2_q8_0_w", "dno_gemm_sn2_q4k_w"],
        path_env: "DENNO_PF_SN2_PTX",
        on: crate::cuda::pf_tc_shared2_on,
    },
    GatePtx {
        env: "DENNO_PF_MIX_SPLIT",
        ptx: "kernels_mix_split.ptx",
        entries: &["dno_mix_split_gemm"],
        path_env: "DENNO_MIX_SPLIT_PTX",
        on: crate::cuda::mix_split_on,
    },
    GatePtx {
        env: "DENNO_PF_MIX_INJ2",
        ptx: "kernels_mix_inj2.ptx",
        entries: &["dno_mix_inj2_bf16"],
        path_env: "DENNO_MIX_INJ2_PTX",
        on: crate::cuda::mix_inj2_on,
    },
];

/// `kernels/` の場所 (ビルド時の manifest から)。`DENNO_*_PTX` の差し替えも見る。
#[cfg(feature = "cuda")]
fn ptx_path(g: &GatePtx) -> PathBuf {
    if !g.path_env.is_empty() {
        if let Ok(p) = std::env::var(g.path_env) {
            if !p.is_empty() {
                return PathBuf::from(p);
            }
        }
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels").join(g.ptx)
}

/// PTX のテキストにエントリが居るか。PTX は `.visible .entry <name>(` の形。
pub fn ptx_has_entry(text: &str, name: &str) -> bool {
    text.contains(&format!(".entry {name}"))
}

/// 1 本のゲートを検査する。`Some(1 行の WARN)` = 問題あり、`None` = ok。
#[cfg(feature = "cuda")]
fn check_gate(g: &GatePtx) -> Option<String> {
    let path = ptx_path(g);
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            return Some(format!(
                "[gates] WARN {} は既定 on だが {} が読めない ({e}) → その機能は静かに従来経路へ落ちる",
                g.env,
                path.display()
            ))
        }
    };
    let text = String::from_utf8_lossy(&bytes);
    for name in g.entries {
        if !ptx_has_entry(&text, name) {
            return Some(format!(
                "[gates] WARN {} は既定 on だが {} に {} が無い → その機能は静かに従来経路へ落ちる",
                g.env,
                path.display(),
                name
            ));
        }
    }
    None
}

/// 起動時に出す行。1 行目の summary + 問題のあるゲートごとに 1 行。
///
/// 起動は止めない (PTX が無ければ従来経路で動く。黙って落ちるのをやめるのが目的)。
#[cfg(feature = "cuda")]
pub fn audit_lines() -> Vec<String> {
    let mut on = 0usize;
    let mut warn = Vec::new();
    for g in DEFAULT_ON_GATES {
        if !(g.on)() {
            continue;
        }
        on += 1;
        if let Some(line) = check_gate(g) {
            warn.push(line);
        }
    }
    let mut lines = Vec::with_capacity(1 + warn.len());
    lines.push(format!(
        "[gates] 既定 on の PTX 監査: {on} 本のゲートを確認 / 欠落 {} 本",
        warn.len()
    ));
    lines.extend(warn);
    lines
}

/// cuda feature 無しでは GPUkernel が無いので監査もしない。
#[cfg(not(feature = "cuda"))]
pub fn audit_lines() -> Vec<String> {
    Vec::new()
}

#[cfg(feature = "cuda")]
#[cfg(test)]
mod tests {
    use super::*;

    /// 台帳が空でないこと (空にしたら監査が無意味になる)。
    #[test]
    fn table_is_not_empty() {
        assert!(DEFAULT_ON_GATES.len() >= 20, "台帳が {}", DEFAULT_ON_GATES.len());
    }

    /// すべてのゲート関数が「未設定 = 既定 on」を返すこと。
    ///
    /// 既定を off に変えたのに台帳から消し忘れる / 台帳に既定 off のゲートを
    /// 入れる、の両方を落とす。テストは `DENNO_*` を触らないので OnceLock は
    /// 未設定の env を読む。
    #[test]
    fn every_listed_gate_is_on_by_default() {
        for g in DEFAULT_ON_GATES {
            assert!((g.on)(), "{} が既定 on でない (台帳が古い)", g.env);
        }
    }

    /// 同じ env は同じ PTX の組を指す (v4/v5 のように複数 PTX は許す)。
    #[test]
    fn entries_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for g in DEFAULT_ON_GATES {
            assert!(
                seen.insert((g.env, g.ptx, g.entries.join(","))),
                "{} / {} が重複",
                g.env,
                g.ptx
            );
        }
    }

    /// 台帳の PTX はツリーに実在し、必要なエントリを持つ (cuda 無しでも読める)。
    #[test]
    fn listed_ptx_exist_with_entries() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels");
        for g in DEFAULT_ON_GATES {
            // 差し替え env はテストでは使わない (既定のファイルを見る)。
            let path = dir.join(g.ptx);
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("{} が読めない: {e}", path.display()));
            for name in g.entries {
                assert!(
                    ptx_has_entry(&text, name),
                    "{} に {} が無い (台帳と PTX が食い違う)",
                    g.ptx,
                    name
                );
            }
        }
    }

    /// このツリーでは既定 on のゲート全部に PTX が揃っている = WARN 0 本。
    /// summary 行の書式もここで固定する (serve の起動ログがこの 1 行を出す)。
    #[test]
    fn audit_reports_zero_missing_in_this_tree() {
        let lines = audit_lines();
        assert_eq!(lines.len(), 1, "想定外の WARN: {lines:?}");
        assert!(
            lines[0].starts_with("[gates] 既定 on の PTX 監査: "),
            "summary の接頭辞が変わった: {}",
            lines[0]
        );
        assert!(
            lines[0].ends_with(" / 欠落 0 本"),
            "summary の末尾が変わった: {}",
            lines[0]
        );
    }

    /// PTX が無いゲートは 1 行の WARN になる (発火の確認。既定 on の監査と同じ道)。
    #[test]
    fn missing_ptx_becomes_one_warn_line() {
        let g = GatePtx {
            env: "DENNO_TEST_MISSING_PTX",
            ptx: "no_such_kernels_file.ptx",
            entries: &["dno_test"],
            path_env: "",
            on: || true,
        };
        let line = check_gate(&g).expect("WARN にならない (静かに落ちる)");
        assert!(line.starts_with("[gates] WARN DENNO_TEST_MISSING_PTX"), "{line}");
        assert!(line.contains("no_such_kernels_file.ptx"), "{line}");
        assert!(line.contains("従来経路へ落ちる"), "{line}");
    }
}
