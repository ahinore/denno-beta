//! P43: decode の expert FFN を「1 層 = 3 起動 / 2 同期」にまとめる (ホスト側)。
//!
//! # 何をするか
//!
//! P39 の VRAM 常駐経路は expert 1 本ごとに `moe_gemv_dev` を 3 回呼んでいたので、
//! 1 トークン = 480 expert x 3 行列 = **1,440 起動 + 1,440 同期**になり、
//! 実測 382 ms/token (CPU は 189 ms/token) の主因になっていた
//! (1 回 0.2 ms なら 288 ms で、観測した差 382 - 189 とほぼ一致する)。
//!
//! ここでは 1 層の top-k (実形 10 expert) を segment 表にまとめ、
//! カーネルには**デバイスポインタ表**で expert ごとの重み先頭を渡す:
//!
//! ```text
//! 1. acquire x k        … 選択 expert を VRAM に載せる (LRU。ミス時だけ H2D)
//! 2. seg_table_shared_x … gate/up 用 (全 expert が同じ入力 x を読む)
//! 3. moe_gemv_seg_pair  … gate 1 起動 + up 1 起動 → **1 同期**
//! 4. silu_mul           … act = silu(g) * u (per-expert 版と同じ式・同じ順)
//! 5. seg_table_per_act  … down 用 (expert ごとの act を読む)
//! 6. moe_gemv_seg_one   … down 1 起動 → **1 同期**
//! ```
//!
//! 起動は 3 回・同期は 2 回。H2D は入力 1 本と act 1 本、D2H は g/u/h の 3 本で、
//! 行列ごとのコピーが層あたり 30 回から 5 回に減る。
//!
//! # 数値 (ビット一致)
//!
//! `kernels_moe_batch.cu` の 1 行の累積は `kernels_moe.cu` と同一
//! (1 スレッドが i の昇順に f32 で加算、同じ dequant 式)。違いは
//! 「重みの先頭を segment ごとに引く」ことだけで、**同じ expert・同じ入力なら
//! 10 回起動版と同じ値**になる。`silu_mul` も `tensor::silu_inplace` + 積と同じ
//! 演算列。したがって寄与 h は 1 ビットも変わらず、重み付き足し込みの順序
//! (expert index 昇順の逐次) も呼び出し側でそのまま維持できる。
//!
//! # ゲート
//!
//! - 呼び出し側 (`qwen4exp::moe_ffn_quant_with`) は `Route::Vram` のときだけ
//!   この経路を試す。既定 (env 未設定) は従来の CPU 経路。
//! - `DENNO_EXPERT_BATCH=0` で per-expert 経路に固定できる (A/B 用。既定は有効)。
//! - カーネル (PTX) が無い / 型が未対応 / expert が 1 本も載らない、のいずれでも
//!   `Ok(None)` を返し、呼び出し側は per-expert 経路へ落ちる (fail closed)。

/// `DENNO_EXPERT_BATCH`: 層一括経路を使うか。`"0"` のときだけ無効 (既定は有効)。
pub const EXPERT_BATCH_ENV: &str = "DENNO_EXPERT_BATCH";

/// [`EXPERT_BATCH_ENV`] の解釈 (純関数)。`DENNO_MOE_PAR` と同じ集合。
pub fn batch_from_value(v: Option<&str>) -> bool {
    !matches!(v.map(str::trim), Some("0") | Some("off") | Some("false"))
}

/// 層一括経路が有効か (プロセスで 1 回だけ env を読む)。
pub fn batch_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| batch_from_value(std::env::var(EXPERT_BATCH_ENV).ok().as_deref()))
}

/// gate/up 用 segment 表 (`[eid, m=1, x_off=0, out_off=i*n_out]`)。
///
/// gate と up は**同じ表**を使える: どちらも全 expert が同じ入力ベクトル x を
/// 読み、出力は expert ごとに `i * n_out` へ並べる (segment の順 = sel の順)。
pub fn seg_table_shared_x(ids: &[u16], n_out: usize) -> Vec<u32> {
    let mut t = Vec::with_capacity(ids.len() * 4);
    for (i, &eid) in ids.iter().enumerate() {
        t.push(eid as u32);
        t.push(1); // m = 1 (decode)
        t.push(0); // x_off = 0 (全 expert 共通の入力)
        t.push((i * n_out) as u32);
    }
    t
}

/// down 用 segment 表 (`[eid, m=1, x_off=i*n_in, out_off=i*n_out]`)。
///
/// down は expert ごとに違う act を読むので、入力も segment ごとにずらす
/// (`act` は expert の並びに詰まっている)。
pub fn seg_table_per_act(ids: &[u16], n_in: usize, n_out: usize) -> Vec<u32> {
    let mut t = Vec::with_capacity(ids.len() * 4);
    for (i, &eid) in ids.iter().enumerate() {
        t.push(eid as u32);
        t.push(1);
        t.push((i * n_in) as u32);
        t.push((i * n_out) as u32);
    }
    t
}

/// `act = silu(g) * u` (要素ごと)。
///
/// `tensor::silu_inplace` の式 `v / (1 + (-v).exp())` と積の順をそのまま使う
/// (per-expert 経路と 1 ビットも変えないための実装。テストが突き合わせる)。
pub fn silu_mul(g: &[f32], u: &[f32], act: &mut [f32]) {
    for i in 0..act.len() {
        let v = g[i];
        act[i] = (v / (1.0 + (-v).exp())) * u[i];
    }
}

// ---------------------------------------------------------------------------
// 起動計画 (cuda.rs の moe_gemv_seg_pair / moe_gemv_seg_one が使う)
// ---------------------------------------------------------------------------
//
// ここに置いてあるのは **GPU 無しで検証できる** ようにするため (cuda.rs は
// feature = "cuda" のときしかコンパイルされない)。

/// 1 起動ぶんの形 ([`seg_launch_plan`] の出力)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegLaunch {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub smem: u32,
}

/// 層一括 GEMV の起動形を決める (純関数)。`None` = この形では起動しない。
///
/// - grid = (行タイル ceil(n_out/4), segment, トークン)
/// - block = (256, 1, 1) = 4 行 x 64 レーン (`kernels_moe_batch.cu` の
///   MB_ROWS = 4 / MB_LANES = 64)
/// - 動的共有メモリ = 4 * row_bytes (行の生バイトを 4 本ぶん)
pub fn seg_launch_plan(
    nseg: usize,
    m_max: usize,
    n_out: usize,
    row_bytes: usize,
) -> Option<SegLaunch> {
    const ROWS: usize = 4; // kernels_moe_batch.cu の MB_ROWS
    if nseg == 0 || m_max == 0 || n_out == 0 || row_bytes == 0 {
        return None;
    }
    let tiles = n_out.div_ceil(ROWS);
    // grid の上限 (y/z は 65535)。実形 (n_out 2560 / nseg 10) では届かない。
    if nseg > u32::MAX as usize || tiles > 65535 || m_max > 65535 {
        return None;
    }
    let smem = ROWS.checked_mul(row_bytes)?;
    // 48 KiB の既定を超えると起動できない (P40 と同じ 46,000 B ゲート)。
    if smem > 46_000 {
        return None;
    }
    Some(SegLaunch {
        grid: (tiles as u32, nseg as u32, m_max as u32),
        block: (256, 1, 1),
        smem: smem as u32,
    })
}

/// segment 表 (`[eid, m, x_off, out_off]` x nseg) から
/// (最大の x 範囲, 最大の out 範囲) を求める (要素数)。**純関数**。
///
/// 呼び出し側は `x.len() >= 最大 x 範囲` / `out.len() >= 最大 out 範囲` を
/// 満たすこと (カーネルは segment ごとのオフセットだけを読み書きする)。
pub fn seg_extents(seg: &[u32], n_in: usize, n_out: usize) -> anyhow::Result<(usize, usize)> {
    use anyhow::bail;
    if !seg.len().is_multiple_of(4) {
        bail!("seg_extents: 表の長さ {} が 4 の倍数でない", seg.len());
    }
    let (mut mx, mut mo) = (0usize, 0usize);
    for s in seg.chunks(4) {
        let m = s[1] as usize;
        if m == 0 {
            bail!("seg_extents: m が 0 の segment がある");
        }
        let mn_in = m
            .checked_mul(n_in)
            .ok_or_else(|| anyhow::anyhow!("n_in*m が溢れる"))?;
        let mn_out = m
            .checked_mul(n_out)
            .ok_or_else(|| anyhow::anyhow!("n_out*m が溢れる"))?;
        let xe = (s[2] as usize)
            .checked_add(mn_in)
            .ok_or_else(|| anyhow::anyhow!("x_off + m*n_in が溢れる"))?;
        let oe = (s[3] as usize)
            .checked_add(mn_out)
            .ok_or_else(|| anyhow::anyhow!("out_off + m*n_out が溢れる"))?;
        mx = mx.max(xe);
        mo = mo.max(oe);
    }
    Ok((mx, mo))
}

// ---------------------------------------------------------------------------
// 参照実装 (GPU 無しで「1 回起動版 == 10 回起動版」を示すための模型)
// ---------------------------------------------------------------------------

/// カーネルの 1 行累積と同じ演算列 (i の昇順に f32 で `acc += y * x[i]`)。
///
/// dequant 済みの行を受けるので、量子型に依存しない。`#[inline(never)]` は
/// 最適化で演算列が畳まれないようにするための目印 (`moe_expert` の `dot` と同じ)。
#[inline(never)]
pub fn row_dot(w_row: &[f32], x: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..x.len() {
        acc += w_row[i] * x[i];
    }
    acc
}

/// 層一括版の模型: segment 表の `x_off` / `out_off` に従い、**segment ごとに
/// 違う重み** (`segs[s]`) を掛ける。テスト専用。
///
/// `segs[s]` は segment s の expert の重みで、`n_out` 本の「n_in 要素の行」。
/// 1 起動版 (1 つの表で全 segment を回す) の意味論をそのまま書いてある。
pub fn ref_seg_gemv(
    segs: &[Vec<Vec<f32>>],
    seg: &[u32],
    x: &[f32],
    n_in: usize,
    n_out: usize,
) -> Vec<f32> {
    assert_eq!(seg.len() % 4, 0, "segment 表は 4 語/本");
    assert_eq!(segs.len(), seg.len() / 4, "重みの本数が segment 数と違う");
    let mut out = vec![0.0f32; 0];
    for (s, words) in seg.chunks(4).enumerate() {
        let x0 = words[2] as usize;
        let o0 = words[3] as usize;
        let xe = x0 + n_in;
        let oe = o0 + n_out;
        if out.len() < oe {
            out.resize(oe, 0.0);
        }
        for r in 0..n_out {
            out[o0 + r] = row_dot(&segs[s][r], &x[x0..xe]);
        }
    }
    out
}

/// per-expert 版の模型: expert ごとに 1 本ずつ、同じ `x` で `rows` を掛ける。
/// テスト専用。
pub fn ref_single_gemv(rows: &[Vec<f32>], x: &[f32], n_out: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_out];
    for r in 0..n_out {
        out[r] = row_dot(&rows[r], x);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(n: usize) -> Vec<u16> {
        (0..n).map(|i| (i * 3 + 1) as u16).collect()
    }

    /// gate/up の表: 全 segment の x_off が 0、out_off が i*n_out。
    #[test]
    fn shared_x_table_puts_inputs_at_zero() {
        let n = 10usize;
        let ff = 640usize;
        let t = seg_table_shared_x(&ids(n), ff);
        assert_eq!(t.len(), n * 4);
        for i in 0..n {
            assert_eq!(t[i * 4], (i as u16 * 3 + 1) as u32, "eid は sel の順");
            assert_eq!(t[i * 4 + 1], 1, "decode は m=1");
            assert_eq!(t[i * 4 + 2], 0, "入力は全 expert 共通");
            assert_eq!(t[i * 4 + 3] as usize, i * ff);
        }
    }

    /// down の表: 入力も segment ごとにずれる。
    #[test]
    fn per_act_table_shifts_inputs() {
        let n = 10usize;
        let (ff, e) = (640usize, 2560usize);
        let t = seg_table_per_act(&ids(n), ff, e);
        for i in 0..n {
            assert_eq!(t[i * 4 + 2] as usize, i * ff, "act は expert ごと");
            assert_eq!(t[i * 4 + 3] as usize, i * e, "出力も expert ごと");
        }
    }

    /// 1 回起動版 (segment 表) と 10 回起動版 (per-expert) がビット一致する。
    ///
    /// GPU 無しではカーネルを走らせられないので、**同じ演算列の参照実装**で
    /// 「表のオフセットが正しく、累積順が同じなら一致する」ことを示す。
    /// 実機の突き合わせは `DENNO_EXPERT_GPU_CHECK=1` が層ごとに出す。
    #[test]
    fn batched_segments_match_per_expert_bitwise() {
        let n = 10usize;
        let (n_in, n_out) = (37usize, 11usize);
        let x: Vec<f32> = (0..n_in)
            .map(|i| ((i * 29 % 71) as f32 - 35.0) * 0.011)
            .collect();
        let seg = seg_table_shared_x(&ids(n), n_out);
        // expert ごとに違う重み (決定的・乱数なし)。
        let all_rows: Vec<Vec<Vec<f32>>> = (0..n)
            .map(|k| {
                (0..n_out)
                    .map(|r| {
                        (0..n_in)
                            .map(|i| (((k * 31 + r * 17 + i * 7) % 23) as f32 - 11.0) * 0.37)
                            .collect()
                    })
                    .collect()
            })
            .collect();

        // 1 起動版: 1 つの segment 表で全 expert を回す。
        let got = ref_seg_gemv(&all_rows, &seg, &x, n_in, n_out);
        // 10 回起動版: expert ごとに 1 本ずつ同じ x で回して並べる。
        let mut want = vec![0.0f32; n * n_out];
        for k in 0..n {
            let one = ref_single_gemv(&all_rows[k], &x, n_out);
            want[k * n_out..(k + 1) * n_out].copy_from_slice(&one);
        }
        assert_eq!(got.len(), want.len());
        assert!(got.iter().any(|v| *v != 0.0), "出力が全部 0 では比較にならない");
        for i in 0..got.len() {
            assert_eq!(
                got[i].to_bits(),
                want[i].to_bits(),
                "要素 {i} で 1 回起動版と per-expert 版がずれた"
            );
        }
    }

    /// down の表でも「expert ごとの act を読む」ことが一致する。
    #[test]
    fn per_act_table_matches_per_expert_bitwise() {
        let n = 10usize;
        let (ff, e) = (19usize, 23usize);
        let acts: Vec<Vec<f32>> = (0..n)
            .map(|k| {
                (0..ff)
                    .map(|i| (((k * 13 + i * 5) % 17) as f32 - 8.0) * 0.23)
                    .collect()
            })
            .collect();
        let mut flat = Vec::with_capacity(n * ff);
        for a in &acts {
            flat.extend_from_slice(a);
        }
        let seg = seg_table_per_act(&ids(n), ff, e);
        let all_rows: Vec<Vec<Vec<f32>>> = (0..n)
            .map(|k| {
                (0..e)
                    .map(|r| {
                        (0..ff)
                            .map(|i| (((k * 7 + r * 3 + i * 11) % 29) as f32 - 14.0) * 0.19)
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let got = ref_seg_gemv(&all_rows, &seg, &flat, ff, e);
        let mut want = vec![0.0f32; n * e];
        for k in 0..n {
            let one = ref_single_gemv(&all_rows[k], &acts[k], e);
            want[k * e..(k + 1) * e].copy_from_slice(&one);
        }
        assert_eq!(got.len(), want.len());
        assert!(got.iter().any(|v| *v != 0.0));
        for i in 0..got.len() {
            assert_eq!(got[i].to_bits(), want[i].to_bits(), "要素 {i} でずれた");
        }
    }

    /// `silu_mul` は `tensor::silu_inplace` + 積とビット一致。
    #[test]
    fn silu_mul_matches_silu_then_multiply() {
        let g: Vec<f32> = (0..97).map(|i| (i as f32 - 48.0) * 0.13).collect();
        let u: Vec<f32> = (0..97).map(|i| ((i * 7 % 41) as f32 - 20.0) * 0.017).collect();
        let mut want = g.clone();
        crate::tensor::silu_inplace(&mut want);
        for i in 0..want.len() {
            want[i] *= u[i];
        }
        let mut got = vec![0.0f32; g.len()];
        silu_mul(&g, &u, &mut got);
        for i in 0..got.len() {
            assert_eq!(
                got[i].to_bits(),
                want[i].to_bits(),
                "要素 {i}: per-expert 経路の silu と一致しない"
            );
        }
    }

    /// 既定は有効。`"0"` / `"off"` / `"false"` で無効 (A/B 用)。
    #[test]
    fn batch_gate_parsing() {
        assert!(batch_from_value(None), "未設定は有効 (既定)");
        assert!(batch_from_value(Some("")));
        assert!(batch_from_value(Some("1")));
        assert!(!batch_from_value(Some("0")));
        assert!(!batch_from_value(Some("off")));
        assert!(!batch_from_value(Some("false")));
        assert!(!batch_from_value(Some(" 0 ")));
    }

    // -----------------------------------------------------------------------
    // 起動計画 (cuda.rs の moe_gemv_seg_* が使う純関数)
    // -----------------------------------------------------------------------

    /// 実形の起動形: 10 segment x 4 行タイル = grid.y 10、block 256。
    #[test]
    fn launch_plan_sizes_and_rejects() {
        // gate (n_out 640 / Q4_K k=2560 の row_bytes 1440)
        let p = seg_launch_plan(10, 1, 640, 1440).expect("plan");
        assert_eq!(p.grid, (160, 10, 1), "行タイル 160 x segment 10 x トークン 1");
        assert_eq!(p.block, (256, 1, 1));
        assert_eq!(p.smem, 4 * 1440);
        assert!(p.smem < 46_000, "48 KiB 既定の内側");
        // down (n_out 2560 / Q5_1 k=640 の row_bytes 480)
        let q = seg_launch_plan(10, 1, 2560, 480).expect("plan");
        assert_eq!(q.grid, (640, 10, 1));
        assert_eq!(q.smem, 4 * 480);
        // 空・巨大は断る (呼び出し側は per-expert / CPU へ)
        assert!(seg_launch_plan(0, 1, 640, 1440).is_none());
        assert!(seg_launch_plan(10, 0, 640, 1440).is_none());
        assert!(seg_launch_plan(10, 1, 0, 1440).is_none());
        assert!(seg_launch_plan(10, 1, 640, 0).is_none());
        // 4 * 12,000 = 48,000 B は 48 KiB 既定に近いので断る
        assert!(seg_launch_plan(10, 1, 640, 12_000).is_none());
    }

    /// gate/up と down の表が、それぞれ「入力 1 本」「expert ごと 1 本」を要求する。
    #[test]
    fn seg_extents_match_the_two_passes() {
        let (n_in, n_out, nseg) = (2560usize, 640usize, 10usize);
        let gu = seg_table_shared_x(&ids(nseg), n_out);
        let (mx, mo) = seg_extents(&gu, n_in, n_out).unwrap();
        assert_eq!(mx, n_in, "gate/up の入力は 1 本 (全 expert 共通)");
        assert_eq!(mo, nseg * n_out);

        let d = seg_table_per_act(&ids(nseg), n_in, n_out);
        let (mx, mo) = seg_extents(&d, n_in, n_out).unwrap();
        assert_eq!(mx, nseg * n_in, "down の入力は expert ごとに 1 本");
        assert_eq!(mo, nseg * n_out);
    }

    /// 壊れた表は積まない (m = 0 / 4 語でない)。
    #[test]
    fn seg_extents_rejects_broken_tables() {
        assert!(seg_extents(&[0, 1, 0], 2560, 640).is_err(), "4 の倍数でない");
        assert!(
            seg_extents(&[0, 0, 0, 0], 2560, 640).is_err(),
            "m = 0 は積まない契約"
        );
    }
}
