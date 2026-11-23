//! attention (decode / prefill / flashdec / KV ミラー / rope)
//!
//! task 664: `src/cuda.rs` (70k 行) からの純粋な移動。ロジック・数値・カーネル・
//! PTX・env ゲートは一切変えていない。

#![allow(non_snake_case, non_camel_case_types)]

use super::*;



// ---- task 179: decode の full attention 1 層をデバイスで閉じる -----------------
//
// 背景 (lab t179a の平均表): full 層 1 回の wall ~264 us の内訳は
//   GPU カーネル 176 / コピー 43 / gap 28 / ホスト仕事 31 (うち gate_mul 10.4,
//   rope_q 3.8, kv_split 1.1, rope_k 0.6, mirror_book 0.5, cache_push 0.4)
// だった。qkv GEMV の出力をホストへ戻す → 正規化 + rope → ミラーへ H2D →
// attention → D2H → 出力ゲート → 出力投影 H2D → D2H という往復が、
// コピーとホスト仕事と GPU の gap を層ごとに作っている。
//
// ここでは 1 層を**デバイスで閉じる**: qkv GEMV (デバイス出力) → q/k 正規化 +
// rope + gate 取り出し (kernels_dec_dev.cu) → KV ミラー 1 行を D2D →
// attention (デバイス入力・出力) → 出力ゲート → 出力投影 GEMV (デバイス入力)
// → 最後に 2 本の D2H (層出力と KV 行) と 1 回の同期だけ。
//
// 数値: 既存経路と**ビット一致**する。
// - rope の cos/sin はホストで CPU と同じ式 (`theta.powf(-2j/n_rot)` の f32) で
//   作り、カーネル引数として**値渡し**する (デバイスの powf/cosf は使わない)。
// - 正規化の二乗和は thread 0 の逐次 (CPU と同じ順)、`(x*inv)*w` の左結合。
// - 出力ゲートは dno_expf (glibc expf の再現) + `1/(1+e)` + 乗算。
// - GEMV は既存と同じカーネル・同じ起動形状 (`plan_moe_gemv`) を使う。
//
// フォールバック (None): ゲート off / 形が合わない / PTX が無い / ミラーが
// 全行シードを要する (cap 成長・新しい層・世代違い) / 起動に失敗。呼び出し側は
// 従来のホスト経路をそのまま走らせる (数値は従来どおり)。
// ---------------------------------------------------------------------------
pub(crate) mod dec_full_dev {
    use super::*;
    use std::cell::{Cell, RefCell};

    /// 既定値。lab t179c の VERDICT (decode -5.5%、深い 2 点は -6.6/-6.4%、
    /// md5 SAME、errors 0、prefill -0.8%) で採否を yes にしたので既定 on。
    /// `DENNO_FULL_DEC_DEV=0` で従来経路へ戻る。
    const DEFAULT_ON: bool = true;

    /// `DENNO_FULL_DEC_DEV` (既定 `DEFAULT_ON`)。
    pub(crate) fn on() -> bool {
        if let Some(v) = force_for_test() {
            return v;
        }
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| match std::env::var("DENNO_FULL_DEC_DEV").ok() {
            None => DEFAULT_ON,
            Some(v) => {
                let v = v.trim().to_ascii_lowercase();
                !matches!(v.as_str(), "" | "0" | "off" | "false" | "no")
            }
        })
    }

    pub(crate) fn force_for_test() -> Option<bool> {
        #[cfg(test)]
        {
            return FORCE.with(|c| c.get());
        }
        #[cfg(not(test))]
        {
            None
        }
    }

    /// テストが env を立てずに on/off を強制する。
    #[cfg(test)]
    pub(crate) fn set_force_for_test(v: Option<bool>) {
        FORCE.with(|c| c.set(v));
    }

    /// `DENNO_FULL_DEC_MIRROR_FILL` (task 572, **既定 on**)。
    ///
    /// この経路は KV ミラーの 1 行追記 (`mirror_append`) が「ミラーがちょうど
    /// 1 行遅れている」ときだけ通り、1 行でも足りないと `None` を返して**その層は
    /// 以後ずっとホスト経路へ落ちる** (ホスト経路は自分の push 後の長さまでしか
    /// ミラーを同期しないので、次もまた 1 行足りない)。この門を立てると、足りない
    /// 行だけをホストの K/V から H2D で埋めてから追記をやり直す (埋めるのは 1 回
    /// だけで、以降は普通の D2D 追記に戻る)。
    ///
    /// task 572 の lab t572b / t572c (既定モデル、depth8k、2 ラウンド) で
    /// **full 段 1.25-1.26 -> 0.96 ms/token (-23%)、短い decode 10.4 -> 10.0
    /// ms/tok (-3.4%)、8k 後と prefill は不変、md5 全 COND SAME、errors 0**。
    /// 切り戻しは `DENNO_FULL_DEC_MIRROR_FILL=0`。
    fn mirror_fill_on() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            mirror_fill_from_value(std::env::var("DENNO_FULL_DEC_MIRROR_FILL").ok().as_deref())
        })
    }

    /// [`mirror_fill_on`] の解釈 (純関数)。未設定は既定 (on)、明示的な偽値だけ off。
    fn mirror_fill_from_value(v: Option<&str>) -> bool {
        match v {
            None => true,
            Some(s) => {
                let s = s.trim().to_ascii_lowercase();
                !matches!(s.as_str(), "" | "0" | "off" | "false" | "no")
            }
        }
    }

    /// 一度に埋めてよい行数 (これを超える遅れは従来どおり断る = 巨大な H2D を
    /// 層ごとに撃たない)。既定 64 行。
    fn mirror_fill_max_rows() -> usize {
        static N: OnceLock<usize> = OnceLock::new();
        *N.get_or_init(|| match std::env::var("DENNO_FULL_DEC_MIRROR_FILL_MAX").ok() {
            Some(v) => v.trim().parse::<usize>().unwrap_or(64),
            None => 64,
        })
    }

    /// 埋めた回数 / 行数 (lab の `[full-dec-dbg]` に出す)。
    pub(crate) static MIRROR_FILLS: std::sync::atomic::AtomicU64 =
        std::sync::atomic::AtomicU64::new(0);
    pub(crate) static MIRROR_FILL_ROWS: std::sync::atomic::AtomicU64 =
        std::sync::atomic::AtomicU64::new(0);

    /// 埋めてよい遅れか (純関数)。`have` 行を持っている状態で `host_len` 行が
    /// 必要。0 (既に足りている) と上限超えは埋めない。
    fn mirror_fill_gap_ok(host_len: usize, have: usize, max_rows: usize) -> Option<usize> {
        let gap = host_len.saturating_sub(have);
        if gap == 0 || gap > max_rows {
            None
        } else {
            Some(gap)
        }
    }

    /// rope の cos/sin 表の上限 (対の本数)。`n_rot <= 2 * CS_MAX`。
    const CS_MAX: usize = 64;

    thread_local! {
        static FORCE: Cell<Option<bool>> = const { Cell::new(None) };
        static ST: RefCell<St> = const { RefCell::new(St::new()) };
    }

    /// qkv の融合 GEMV 1 発ぶんの記述 (連結できない層は 3 本に分かれる)。
    #[derive(Clone, Copy)]
    pub(crate) struct Unit<'a> {
        pub uid: u64,
        pub ty: crate::kquant::GgmlType,
        pub blocks: &'a [u8],
        pub k: usize,
        pub n: usize,
        /// 出力 (`qkv` バッファ) の何行目から書くか。
        pub out_row: usize,
    }

    /// 呼び出し 1 回ぶんの入力 (すべてホスト側の値)。
    pub(crate) struct Call<'a> {
        pub cache_id: u64,
        pub n_layer: usize,
        pub layer: usize,
        pub layer_gen: u64,
        /// push 後の行数 (= pos + 1)
        pub len: usize,
        pub kv_dim: usize,
        pub heads: usize,
        pub group: usize,
        pub dh: usize,
        pub scale: f32,
        pub n_rot: usize,
        pub pos: usize,
        pub theta: f32,
        pub eps: f32,
        pub q_norm: &'a [f32],
        pub k_norm: &'a [f32],
        /// qkv の GEMV (q|gate|k|v)。連結できた層は 1 本、できない層は 3 本。
        pub qkv_units: &'a [Unit<'a>],
        /// 入力長 (= 全 unit 共通) と出力の合計行数。
        pub qkv_k: usize,
        pub qkv_n: usize,
        /// q / k / v の部分が `qkv` バッファの何行目から始まるか (float 単位)。
        /// カーネルは [q|gate][k][v] の並びを前提にするので、連結時に並びが
        /// 変わった層 (fuse_parts の zero-copy reorder) はここで弾く。
        pub q_off: usize,
        pub k_off: usize,
        pub v_off: usize,
        /// 出力投影
        pub wo_uid: u64,
        pub wo_ty: crate::kquant::GgmlType,
        pub wo_blocks: &'a [u8],
        pub wo_k: usize,
        pub wo_n: usize,
        pub n_embd: usize,
        /// 層の入力 (ホスト)
        pub cur: &'a [f32],
        /// task 572: この層のホスト KV キャッシュの行 `[0, len-1)` (f32)。ミラーが
        /// 遅れている回に、足りない行だけを埋めるのに使う (`DENNO_FULL_DEC_MIRROR_FILL`)。
        pub k_host: &'a [f32],
        pub v_host: &'a [f32],
        /// task 468: 層の入力を**デバイスから**読む (直前の mix の鎖が残した
        /// `mixed` の f32 `qkv_k` 本)。`Some` のとき `cur` は空でよく、
        /// pinned への写しも H2D も撃たない (値は同じバイト列)。
        pub x_dev: Option<u64>,
    }

    /// 戻り (層の出力と、ホスト KV キャッシュへ push する行)。
    pub(crate) struct Out {
        pub out: Vec<f32>,
        pub k: Vec<f32>,
        pub v: Vec<f32>,
    }

    /// 層ごとの norm 重み (ホスト側ポインタが変わったら上げ直す)。
    struct Norms {
        key: (usize, usize),
        q: DevBuffer,
        k: DevBuffer,
    }

    pub(crate) struct St {
        qkv_in: Option<DevBuffer>,
        qkv: Option<DevBuffer>,
        qout: Option<DevBuffer>,
        gate: Option<DevBuffer>,
        kout: Option<DevBuffer>,
        kv: Option<DevBuffer>,
        out: Option<DevBuffer>,
        pin_x: Option<PinMem>,
        pin_out: Option<PinMem>,
        pin_kv: Option<PinMem>,
        norms: Vec<Option<Norms>>,
    }

    impl St {
        const fn new() -> Self {
            Self {
                qkv_in: None,
                qkv: None,
                qout: None,
                gate: None,
                kout: None,
                kv: None,
                out: None,
                pin_x: None,
                pin_out: None,
                pin_kv: None,
                norms: Vec::new(),
            }
        }
    }

    impl<'a> Call<'a> {
        /// 形が dev 経路で扱えるか (ホスト側だけで判定できる)。
        fn ok_shape(&self) -> bool {
            let nkv = if self.group > 0 {
                self.heads / self.group
            } else {
                0
            };
            self.len > 0
                && self.len <= 65535
                && self.heads > 0
                && self.group > 0
                && self.group <= 32
                && self.heads % self.group == 0
                && self.dh >= 32
                && self.dh <= 1024
                && self.dh % 4 == 0
                && self.kv_dim == nkv * self.dh
                && self.n_rot % 2 == 0
                && self.n_rot <= self.dh
                && self.n_rot / 2 <= CS_MAX
                && self.q_norm.len() == self.dh
                && self.k_norm.len() == self.dh
                && self.qkv_n == 2 * self.heads * self.dh + 2 * self.kv_dim
                && self.qkv_k > 0
                && (self.x_dev.is_some() || self.cur.len() == self.qkv_k)
                && self.units_ok()
                // カーネルの前提: [q(2dh/head)][k][v] の並び。
                && self.q_off == 0
                && self.k_off == 2 * self.heads * self.dh
                && self.v_off == 2 * self.heads * self.dh + self.kv_dim
                && self.wo_k == self.heads * self.dh
                && self.wo_n == self.n_embd
                && self.n_embd > 0
        }

        /// qkv の unit 列が「同じ入力長・隙間なく qkv_n 行」を満たすか。
        fn units_ok(&self) -> bool {
            if self.qkv_units.is_empty() || self.qkv_units.len() > 4 {
                return false;
            }
            let mut at = 0usize;
            for u in self.qkv_units {
                if u.k != self.qkv_k || u.n == 0 || u.out_row != at {
                    return false;
                }
                at += u.n;
            }
            at == self.qkv_n
        }
    }

    /// この経路の attention の下限行数 (既定 1 = 常に GPU)。
    fn full_dec_attn_min() -> usize {
        static N: OnceLock<usize> = OnceLock::new();
        *N.get_or_init(|| match std::env::var("DENNO_FULL_DEC_ATTN_MIN").ok() {
            Some(v) => v.trim().parse::<usize>().unwrap_or(1),
            None => 1,
        })
    }

    /// 1 行だけログる (フォールバックの理由。既定は出さない)。
    fn note_fallback(why: &str) {
        static ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
            eprintln!("[full-dec-dev] 従来経路へ: {why}");
        }
    }

    /// task 192: full 層の GEMV を v5 (無ければ v4) で回すゲート
    /// (`DENNO_FULL_DEC_FAST`・既定 on)。
    ///
    /// lab t179c の `[dec-kprof-avg]` full 表では 1 層の GPU 191 us のうち
    /// **qkv GEMV 43 us + wo GEMV 24 us** だった。この経路の qkv は
    /// `gemv_kernel_mode()` (既定 v2 = 4 行/ブロック・shared 経由) のままなので、
    /// GDN と同じ v5 (ベクタロード + 4 アキュムレータ) に置き換える。
    /// 数値は v2 / v4 / v5 で丸め順が違う (決定論は保つ)。
    const FULL_DEC_FAST_DEFAULT: bool = true;

    /// `DENNO_FULL_DEC_FAST` の解釈 (純関数。明示的な偽値だけが off)。
    pub(crate) fn parse_full_dec_fast(raw: Option<&str>) -> bool {
        !matches!(
            raw.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
            Some("0") | Some("off") | Some("false") | Some("no")
        )
    }

    fn fast_on() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            match std::env::var("DENNO_FULL_DEC_FAST").ok() {
                None => FULL_DEC_FAST_DEFAULT,
                Some(v) => parse_full_dec_fast(Some(v.as_str())),
            }
        })
    }

    /// task 427: qkv / 出力投影の GEMV を dn6 (1 ブロック rows 行 / 1 レーン 1 対) で
    /// 回す行数 (純関数)。`DENNO_FULL_DEC_FAST` の**既存 env に値を足す**形にした
    /// (lab の COND は新しい env 名を落とすことがある)。
    ///
    /// **既定 2 行** (lab t427b: full 段 1.52 -> 1.36 ms/token = -10.5%、gpu 0.99 ->
    /// 0.84。短い decode 11.95 -> 11.7 ms/tok)。切り戻しは `DENNO_FULL_DEC_FAST=v5`
    /// (または `1` / `on` / `0`)。
    /// `dn6` / `dn6r2` = 2 行、`dn6r1` = 1、`dn6r4` = 4、`dn6r8` = 8。
    /// それ以外 (未設定・`1`・`on` など) は 0 = これまでどおり v5。
    pub(crate) fn full_dec_dn6_rows_from_value(raw: Option<&str>) -> usize {
        match raw.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
            Some("dn6") | Some("dn6r2") => 2,
            Some("dn6r1") => 1,
            Some("dn6r4") => 4,
            Some("dn6r8") => 8,
            Some("v5") | Some("dn5") | Some("1") | Some("on") | Some("true") | Some("0")
            | Some("off") | Some("false") | Some("no") => 0,
            _ => 2,
        }
    }

    /// [`full_dec_dn6_rows_from_value`] をプロセスで 1 回だけ読む。
    fn dn6_rows() -> usize {
        static ROWS: OnceLock<usize> = OnceLock::new();
        *ROWS.get_or_init(|| {
            full_dec_dn6_rows_from_value(std::env::var("DENNO_FULL_DEC_FAST").ok().as_deref())
        })
    }

    /// qkv / 出力投影の GEMV 計画。fast では v5 → v4 の順に試す。
    pub(crate) fn dec_plan(
        ctx: &CudaCtx,
        ty: crate::kquant::GgmlType,
        k: usize,
        n: usize,
    ) -> Result<Option<MoeGemvPlan>> {
        if fast_on() {
            if let Some(p) = ctx.plan_moe_gemv_v5(ty, k, n) {
                return Ok(Some(p));
            }
            if let Some(p) = ctx.plan_moe_gemv_v4(ty, k, n, gdn_dec_v4_lanes_cap()) {
                return Ok(Some(p));
            }
        }
        ctx.plan_moe_gemv(ty, k, n, gemv_kernel_mode())
    }

    /// 1 層をデバイスで回す。`None` = この回は従来経路。
    /// task 307: `DENNO_FULL_DEC_DBG` の解釈 (純関数)。**既定 off**。
    ///
    /// on のときだけ、この経路の呼び出し回数と「従来経路へ落ちた理由」を
    /// 256 回ごとに 1 行で出す (数値・経路は 1 つも変えない)。
    fn dbg_on() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            !matches!(
                std::env::var("DENNO_FULL_DEC_DBG")
                    .ok()
                    .map(|s| s.trim().to_ascii_lowercase())
                    .as_deref(),
                None | Some("") | Some("0") | Some("off") | Some("false") | Some("no")
            )
        })
    }

    /// 落ちた理由の数え上げ (0=gate, 1=shape, 2=ctx, 3=mirror, 4=plan, 5=その他)。
    static FULL_DEC_FAIL: [std::sync::atomic::AtomicU64; 6] =
        [const { std::sync::atomic::AtomicU64::new(0) }; 6];
    static FULL_DEC_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    /// task 468: 入力の H2D を省いた層数 (mix の鎖がデバイスに残した `mixed` を
    /// そのまま読んだ回。発火の証拠 = `[full-x-dev]`)。
    static FULL_X_DEV_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    /// task 468: 1 層ぶん数える。
    fn note_full_x_dev_hit() {
        FULL_X_DEV_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// task 468: 積算を 0 に戻す (リクエスト終端)。
    pub(crate) fn x_dev_reset() {
        FULL_X_DEV_HITS.store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// task 468: 入力の H2D を省いた層数。
    pub(crate) fn x_dev_hits() -> u64 {
        FULL_X_DEV_HITS.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// task 468: ミラーが「この層の 1 行追記」を受けられる状態か (書かずに見るだけ)。
    ///
    /// [`mirror_append`] の前提検査と同じ条件を、書き込み無しで評価する。
    /// defer の可否 (入力の H2D を省いてよいか) を mix の前に決めるために使う
    /// (種の無いミラーで `forward` が `None` を返すと要求が落ちるため)。
    pub(crate) fn mirror_ready(
        cache_id: u64,
        n_layer: usize,
        kv_dim: usize,
        layer: usize,
        gen: u64,
        len: usize,
    ) -> bool {
        if len == 0 || layer >= n_layer {
            return false;
        }
        let Some(mirrors) = Q4_ATTN_MIRRORS.get() else {
            return false;
        };
        let ms = match mirrors.lock() {
            Ok(g) => g,
            Err(e) => e.into_inner(),
        };
        let Some((_, m, _)) = ms.iter().find(|(k, _, _)| *k == cache_id) else {
            return false;
        };
        if m.kv_dim != kv_dim || m.state.len.len() != n_layer {
            return false;
        }
        if m.k[layer].is_none() || m.v[layer].is_none() {
            return false;
        }
        if len > m.state.cap || q4_mirror_cap(len, m.state.cap) > m.state.cap {
            return false;
        }
        q4_mirror_have(&m.state, layer, len, gen) + 1 == len
    }

    /// task 468: 「この呼び出しはデバイス入力を読める形か」を**ホスト側だけで**
    /// 判定する (mix の鎖の D2H と待ちを省いてよいかの事前検査)。
    ///
    /// `len` は push 後の KV 行数 (= pos + 1)。形の静的検査 (`ok_shape`) は
    /// モデル固定なのでここでは見ない (崩れていたら `forward` が `None` を返し、
    /// 呼び出し側が要求を落とす = 黙って別経路で計算しない)。
    pub(crate) fn dev_input_ok(len: usize) -> bool {
        on() && cuda_enabled() && gpu_attention_enabled() && q4_gpu_attn_enabled()
            && len >= full_dec_attn_min()
            && len <= 65535
            && chain_ctx().is_some()
    }

    /// 理由コードを 1 つ数え、256 回ごとに集計を出す。
    fn dbg_note(idx: usize) {
        if !dbg_on() {
            return;
        }
        use std::sync::atomic::Ordering as AO;
        let calls = FULL_DEC_CALLS.fetch_add(1, AO::Relaxed) + 1;
        FULL_DEC_FAIL[idx].fetch_add(1, AO::Relaxed);
        if calls % 256 == 1 {
            let f: Vec<u64> = FULL_DEC_FAIL.iter().map(|c| c.load(AO::Relaxed)).collect();
            eprintln!(
                "[full-dec-dbg] calls={calls} gate={} shape={} ctx={} mirror={} plan={} other={} fill={} fill_rows={}",
                f[0], f[1], f[2], f[3], f[4], f[5],
                MIRROR_FILLS.load(AO::Relaxed),
                MIRROR_FILL_ROWS.load(AO::Relaxed)
            );
        }
    }

    /// 内部エラーの文言から理由コードを選ぶ。
    fn dbg_reason(msg: &str) -> usize {
        if msg.contains("ミラー") {
            3
        } else if msg.contains("plan") || msg.contains("GEMV") {
            4
        } else {
            5
        }
    }

    pub(crate) fn forward(a: &Call<'_>) -> Option<Out> {
        if !on() {
            dbg_note(0);
            return None;
        }
        // GPU attention のゲートを従来経路と揃える (無効なら CPU へ落ちるので
        // ここで何もせず None を返す)。
        if !cuda_enabled() || !gpu_attention_enabled() || !q4_gpu_attn_enabled() {
            dbg_note(0);
            return None;
        }
        // task 179: 従来経路は `len < q4_gpu_attn_min()` (既定 64) で CPU へ
        // 落ちるが、この経路は attention が同じ submission に載るので起動の
        // 追加コストが無い。短い文脈でも GPU の split カーネルを使う
        // (丸め順は CPU と違う。attention は P17g から緩和済みの契約)。
        // `DENNO_FULL_DEC_ATTN_MIN` で下限を戻せる。
        if a.len < full_dec_attn_min() {
            dbg_note(0);
            return None;
        }
        if !a.ok_shape() {
            dbg_note(1);
            note_fallback("形が合わない");
            return None;
        }
        let Some(ctx) = chain_ctx() else {
            dbg_note(2);
            note_fallback("chain_ctx が無い");
            return None;
        };
        if let Err(e) = ctx.make_current() {
            dbg_note(2);
            note_fallback(&format!("cuCtxSetCurrent: {e:#}"));
            return None;
        }
        match ST.with(|cell| forward_inner(&ctx, a, &mut cell.borrow_mut())) {
            Ok(v) => Some(v),
            Err(e) => {
                let msg = format!("{e:#}");
                dbg_note(dbg_reason(&msg));
                note_fallback(&msg);
                None
            }
        }
    }

    fn forward_inner(ctx: &CudaCtx, a: &Call<'_>, st: &mut St) -> Result<Out> {
        let qd = a.heads * a.dh;
        let kv_dim = a.kv_dim;
        let nkv = a.heads / a.group;
        // task 319: この経路のホスト仕事の内訳 (`DENNO_GDN_DEC_KPROF` の
        // `host step` 表に `fd_*` として出る)。off は `on()` の読み 1 回だけで
        // Instant を取らない (数値経路は 1 つも変えない)。
        let mut kp_t = std::time::Instant::now();
        let kp_on = crate::cuda::dec_kprof::on();
        macro_rules! kp_mark {
            ($name:literal) => {
                if kp_on {
                    crate::cuda::dec_kprof::note_since($name, kp_t);
                    kp_t = std::time::Instant::now();
                }
            };
        }

        // 1) qkv GEMV: ホスト cur -> デバイス (D2H しない)
        // task 468: 入力がデバイスにある回 (直前の mix の鎖が残した `mixed`) は
        // pinned へ写さず H2D も撃たない (同じバイト列なので値は変わらない)。
        Scratch::ensure(&mut st.qkv, a.qkv_n * 4)?;
        let x_ptr = match a.x_dev {
            Some(p) => {
                note_full_x_dev_hit();
                p
            }
            None => {
                Scratch::ensure(&mut st.qkv_in, a.qkv_k * 4)?;
                Scratch::ensure_pin(&mut st.pin_x, a.qkv_k * 4)?;
                {
                    let px = st.pin_x.as_mut().expect("ensure 済み");
                    px.as_mut_slice::<f32>()[..a.qkv_k].copy_from_slice(a.cur);
                }
                let xp = st.qkv_in.as_ref().expect("ensure 済み").ptr();
                let lib = lib()?;
                let pin = st.pin_x.as_ref().expect("ensure 済み");
                unsafe {
                    check(
                        (lib.memcpy_htod_async)(
                            xp,
                            pin.ptr as *const c_void,
                            a.qkv_k * 4,
                            std::ptr::null_mut(),
                        ),
                        "cuMemcpyHtoDAsync (full dec dev)",
                    )?;
                }
                xp
            }
        };
        let qkv_ptr = st.qkv.as_ref().expect("ensure 済み").ptr();
        for u in a.qkv_units {
            let w = weights_for_gguf(u.uid, u.ty, u.blocks, u.k, u.n)?;
            // task 427: dn6 (`DENNO_FULL_DEC_FAST=dn6*`) は 1 ブロック rows 行 /
            // 1 レーン 1 対 (`dno_mix_gemv_q8_0_dn6`)。形 (Q8_0 / n_in % 64 == 0 /
            // warps <= 8) と PTX が揃えば v5 の代わりに撃つ (丸め順だけが変わる)。
            // task 504: `DENNO_FULL_DEC_FAST=dn7` なら先に dn7 (1 対 = 4 レーン)。
            if full_dec_dn7_on() {
                if let (Some((rows7, thr7)), Some(rb)) =
                    (dn7_dec_plan(u.ty, u.k, u.n), gguf_row_bytes(u.ty, u.k))
                {
                    if w.buf.bytes() == u.n * rb {
                        launch_mix_dn7(
                            ctx,
                            w.buf.ptr(),
                            x_ptr,
                            qkv_ptr + (u.out_row * 4) as u64,
                            u.k,
                            u.n,
                            rb,
                            rows7,
                            thr7,
                        )?;
                        note_dn7_full_hit();
                        continue;
                    }
                }
            }
            let dn6 = dn6_gemv_plan(u.ty, u.k, u.n, dn6_rows());
            if let (Some((warps, rows)), Some(rb)) = (dn6, gguf_row_bytes(u.ty, u.k)) {
                if w.buf.bytes() != u.n * rb {
                    bail!("qkv 重みのバイト数が合わない ({:?})", u.ty);
                }
                launch_dn6_gemv(
                    ctx,
                    w.buf.ptr(),
                    x_ptr,
                    qkv_ptr + (u.out_row * 4) as u64,
                    u.k,
                    u.n,
                    rb,
                    warps,
                    rows,
                )?;
                note_dn6_full_hit();
                continue;
            }
            let Some(plan) = dec_plan(ctx, u.ty, u.k, u.n)? else {
                bail!("qkv GEMV の plan が無い ({:?})", u.ty);
            };
            if w.buf.bytes() != u.n * plan.rb {
                bail!("qkv 重みのバイト数が合わない ({:?})", u.ty);
            }
            launch_plan(
                ctx,
                &plan,
                w.buf.ptr(),
                x_ptr,
                qkv_ptr + (u.out_row * 4) as u64,
                u.k,
                u.n,
            )?;
        }
        kp_mark!("fd_qkv");

        // 2) q/k 正規化 + rope + gate 取り出し
        let kernels = dev_kernels()?;
        Scratch::ensure(&mut st.qout, qd * 4)?;
        Scratch::ensure(&mut st.gate, qd * 4)?;
        Scratch::ensure(&mut st.kout, kv_dim * 4)?;
        Scratch::ensure(&mut st.kv, 2 * kv_dim * 4)?;
        upload_norms(st, a)?;
        let half = a.n_rot / 2;
        let mut cs = [0f32; 2 * CS_MAX];
        {
            let posf = a.pos as f32;
            for j in 0..half {
                let ang = posf * a.theta.powf(-2.0 * j as f32 / a.n_rot as f32);
                cs[j * 2] = ang.cos();
                cs[j * 2 + 1] = ang.sin();
            }
        }
        let (qout, gout, kout, kv) = (
            st.qout.as_ref().expect("ensure 済み").ptr(),
            st.gate.as_ref().expect("ensure 済み").ptr(),
            st.kout.as_ref().expect("ensure 済み").ptr(),
            st.kv.as_ref().expect("ensure 済み").ptr(),
        );
        {
            let norms = st.norms[a.layer].as_ref().expect("直前に用意した");
            let (mut p_qkv, mut p_q, mut p_g, mut p_k, mut p_kv) =
                (qkv_ptr, qout, gout, kout, kv);
            let (mut p_wq, mut p_wk) = (norms.q.ptr(), norms.k.ptr());
            let mut p_cs = cs;
            let (mut heads_c, mut nkv_c, mut dh_c) =
                (a.heads as c_int, nkv as c_int, a.dh as c_int);
            let (mut nrot_c, mut qg_c, mut kvd_c) =
                (a.n_rot as c_int, (2 * qd) as c_int, kv_dim as c_int);
            let mut eps_c = a.eps;
            let mut args: Vec<*mut c_void> = vec![
                &mut p_qkv as *mut _ as *mut c_void,
                &mut p_q as *mut _ as *mut c_void,
                &mut p_g as *mut _ as *mut c_void,
                &mut p_k as *mut _ as *mut c_void,
                &mut p_kv as *mut _ as *mut c_void,
                &mut p_wq as *mut _ as *mut c_void,
                &mut p_wk as *mut _ as *mut c_void,
                &mut p_cs as *mut _ as *mut c_void,
                &mut heads_c as *mut _ as *mut c_void,
                &mut nkv_c as *mut _ as *mut c_void,
                &mut dh_c as *mut _ as *mut c_void,
                &mut nrot_c as *mut _ as *mut c_void,
                &mut qg_c as *mut _ as *mut c_void,
                &mut kvd_c as *mut _ as *mut c_void,
                &mut eps_c as *mut _ as *mut c_void,
            ];
            unsafe {
                ctx.launch_async(
                    kernels.f_norm,
                    ((a.heads + nkv) as u32, 1, 1),
                    (a.dh as u32, 1, 1),
                    (2 * a.dh * 4) as u32,
                    &mut args,
                )?;
            }
        }
        kp_mark!("fd_prep");

        // 3) KV ミラー 1 行を D2D で追記
        let (mkptr, mvptr) = match mirror_append(
            a.cache_id,
            a.n_layer,
            kv_dim,
            a.layer,
            a.layer_gen,
            a.len,
            kout,
            kv + (kv_dim * 4) as u64,
        ) {
            Some(p) => p,
            None => {
                // task 572: ミラーが遅れている回は、足りない行だけをホストの K/V から
                // 埋めてから追記をやり直す (`DENNO_FULL_DEC_MIRROR_FILL`, 既定 off)。
                // 埋めるのは 1 回だけ。以降は普通の 1 行 D2D 追記に戻る。
                let mut filled = false;
                if mirror_fill_on() {
                    let host_len = a.len.saturating_sub(1);
                    let (kn, vn) = (
                        a.k_host.len() / kv_dim.max(1),
                        a.v_host.len() / kv_dim.max(1),
                    );
                    if host_len > 0 && kn >= host_len && vn >= host_len {
                        let have =
                            q4_attn_mirror_have_rows(a.cache_id, a.n_layer, kv_dim, a.layer, a.layer_gen)
                                .unwrap_or(0);
                        if let Some(gap) =
                            mirror_fill_gap_ok(host_len, have, mirror_fill_max_rows())
                        {
                            if q4_attn_mirror_prepare(
                                a.cache_id,
                                a.n_layer,
                                kv_dim,
                                a.layer,
                                a.layer_gen,
                                host_len,
                                a.k_host,
                                a.v_host,
                                false,
                            )
                            .is_ok()
                            {
                                filled = true;
                                MIRROR_FILLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                MIRROR_FILL_ROWS
                                    .fetch_add(gap as u64, std::sync::atomic::Ordering::Relaxed);
                            }
                        }
                    }
                }
                if !filled {
                    bail!("ミラーの 1 行追記ができない (全行シードが必要)");
                }
                mirror_append(
                    a.cache_id,
                    a.n_layer,
                    kv_dim,
                    a.layer,
                    a.layer_gen,
                    a.len,
                    kout,
                    kv + (kv_dim * 4) as u64,
                )
                .ok_or_else(|| anyhow!("ミラーを埋めた後も 1 行追記ができない"))?
            }
        };
        kp_mark!("fd_mirror");

        // 4) attention (デバイス入力・出力)。task 460: 融合版 (combine + gate) を
        //    選んだときは a_gated = true で返る (下の出力ゲートを飛ばす)。
        let (aptr, a_gated) = attn_launch(
            ctx, qout, mkptr, mvptr, a.len, kv_dim, a.heads, a.group, a.dh, a.scale, gout,
        )?;
        kp_mark!("fd_attn");

        // 5) 出力ゲート (デバイス)。融合版が掛け済みなら飛ばす。
        if !a_gated {
            let mut p_out = aptr;
            let mut p_gate = gout;
            let mut n_c = qd as c_int;
            let mut args: Vec<*mut c_void> = vec![
                &mut p_out as *mut _ as *mut c_void,
                &mut p_gate as *mut _ as *mut c_void,
                &mut n_c as *mut _ as *mut c_void,
            ];
            unsafe {
                ctx.launch_async(
                    kernels.f_gate,
                    ((qd as u32).div_ceil(256), 1, 1),
                    (256, 1, 1),
                    0,
                    &mut args,
                )?;
            }
        }
        kp_mark!("fd_gate");

        // 6) 出力投影 (デバイス入力)。既定は v4 (レーン数可変) — lab t177c で
        //    同じ形 (k=6144 Q8_0 / n=2560) が 51.2 -> 25.7 us。形が合わない・
        //    PTX が無いときは従来 (v3) の plan へ。
        let w_wo = weights_for_gguf(a.wo_uid, a.wo_ty, a.wo_blocks, a.wo_k, a.wo_n)?;
        Scratch::ensure(&mut st.out, a.wo_n * 4)?;
        let optr = st.out.as_ref().expect("ensure 済み").ptr();
        // task 427: dn6 経路 (既定は v5)。形が合わなければ下の v5 / v4 へ落ちる。
        // task 504: `DENNO_FULL_DEC_FAST=dn7` なら先に dn7 (1 対 = 4 レーン)。
        let dn6_wo = if full_dec_dn7_on() {
            let dn7_wo = dn7_dec_plan(a.wo_ty, a.wo_k, a.wo_n);
            if let (Some((rows7, thr7)), Some(rb)) = (dn7_wo, gguf_row_bytes(a.wo_ty, a.wo_k)) {
                if w_wo.buf.bytes() != a.wo_n * rb {
                    bail!("出力投影の重みのバイト数が合わない");
                }
                launch_mix_dn7(
                    ctx,
                    w_wo.buf.ptr(),
                    aptr,
                    optr,
                    a.wo_k,
                    a.wo_n,
                    rb,
                    rows7,
                    thr7,
                )?;
                note_dn7_full_hit();
                None
            } else {
                dn6_gemv_plan(a.wo_ty, a.wo_k, a.wo_n, dn6_rows())
            }
        } else {
            dn6_gemv_plan(a.wo_ty, a.wo_k, a.wo_n, dn6_rows())
        };
        if let (Some((warps, rows)), Some(rb)) = (dn6_wo, gguf_row_bytes(a.wo_ty, a.wo_k)) {
            if w_wo.buf.bytes() != a.wo_n * rb {
                bail!("出力投影の重みのバイト数が合わない");
            }
            launch_dn6_gemv(
                ctx,
                w_wo.buf.ptr(),
                aptr,
                optr,
                a.wo_k,
                a.wo_n,
                rb,
                warps,
                rows,
            )?;
            note_dn6_full_hit();
        } else {
            let plan_wo = wo_plan(ctx, a)?;
            let Some(plan_wo) = plan_wo else {
                bail!("出力投影の plan が無い ({:?})", a.wo_ty);
            };
            if w_wo.buf.bytes() != a.wo_n * plan_wo.rb {
                bail!("出力投影の重みのバイト数が合わない");
            }
            launch_plan(ctx, &plan_wo, w_wo.buf.ptr(), aptr, optr, a.wo_k, a.wo_n)?;
        }
        kp_mark!("fd_wo");

        // 7) D2H 2 本 + 1 回の同期
        Scratch::ensure_pin(&mut st.pin_out, a.wo_n * 4)?;
        Scratch::ensure_pin(&mut st.pin_kv, 2 * kv_dim * 4)?;
        {
            let lib = lib()?;
            let po = st.pin_out.as_ref().expect("ensure 済み");
            let pk = st.pin_kv.as_ref().expect("ensure 済み");
            unsafe {
                check(
                    (lib.memcpy_dtoh_async)(po.ptr, optr, a.wo_n * 4, std::ptr::null_mut()),
                    "cuMemcpyDtoHAsync (full dec dev out)",
                )?;
                check(
                    (lib.memcpy_dtoh_async)(pk.ptr, kv, 2 * kv_dim * 4, std::ptr::null_mut()),
                    "cuMemcpyDtoHAsync (full dec dev kv)",
                )?;
            }
        }
        kp_mark!("fd_d2h");
        // task 368: 読むのは null ストリームの非同期コピーが書いた pinned だけなので、
        // 待つ範囲を null ストリームに狭めてよい (`DENNO_DEC_SYNC_STREAM`、既定 off)。
        ctx.sync_decode()?;
        kp_mark!("fd_sync");
        let out: Vec<f32> = st
            .pin_out
            .as_ref()
            .expect("ensure 済み")
            .as_slice::<f32>()[..a.wo_n]
            .to_vec();
        let kvh: Vec<f32> = st
            .pin_kv
            .as_ref()
            .expect("ensure 済み")
            .as_slice::<f32>()[..2 * kv_dim]
            .to_vec();
        let k = kvh[..kv_dim].to_vec();
        let v = kvh[kv_dim..].to_vec();
        kp_mark!("fd_tail");
        Ok(Out { out, k, v })
    }

    /// norm 重みを層ごとに 1 回だけ上げる (ポインタが変われば上げ直す)。
    fn upload_norms(st: &mut St, a: &Call<'_>) -> Result<()> {
        let key = (a.q_norm.as_ptr() as usize, a.k_norm.as_ptr() as usize);
        if st.norms.len() <= a.layer {
            st.norms.resize_with(a.layer + 1, || None);
        }
        let need = match &st.norms[a.layer] {
            Some(n) => n.key != key,
            None => true,
        };
        if need {
            let q = DevBuffer::alloc_tagged(a.dh * 4, crate::mem_audit::TAG_OTHER)?;
            q.copy_from(a.q_norm)?;
            let k = DevBuffer::alloc_tagged(a.dh * 4, crate::mem_audit::TAG_OTHER)?;
            k.copy_from(a.k_norm)?;
            st.norms[a.layer] = Some(Norms { key, q, k });
        }
        Ok(())
    }

    /// 出力投影の plan。既定は v4 (`DENNO_FULL_DEC_GEMV=v3` で従来へ)。
    /// task 192: `DENNO_FULL_DEC_FAST` では v5 を先に試す。
    fn wo_plan(ctx: &CudaCtx, a: &Call<'_>) -> Result<Option<MoeGemvPlan>> {
        static V4: OnceLock<bool> = OnceLock::new();
        let v4_on = *V4.get_or_init(|| {
            !matches!(
                std::env::var("DENNO_FULL_DEC_GEMV").ok().as_deref(),
                Some("v3") | Some("0") | Some("off") | Some("false")
            )
        });
        if fast_on() {
            if let Some(p) = ctx.plan_moe_gemv_v5(a.wo_ty, a.wo_k, a.wo_n) {
                return Ok(Some(p));
            }
        }
        if v4_on {
            if let Some(p) = ctx.plan_moe_gemv_v4(a.wo_ty, a.wo_k, a.wo_n, gdn_dec_v4_lanes_cap()) {
                return Ok(Some(p));
            }
        }
        ctx.plan_moe_gemv(a.wo_ty, a.wo_k, a.wo_n, gemv_kernel_mode())
    }

    /// GEMV の起動 (引数は `moe_gemv_dev_with` と同一)。
    pub(crate) fn launch_plan(
        ctx: &CudaCtx,
        plan: &MoeGemvPlan,
        wptr: CUdeviceptr,
        xptr: CUdeviceptr,
        optr: CUdeviceptr,
        n_in: usize,
        n_out: usize,
    ) -> Result<()> {
        let (mut pw, mut px, mut po) = (wptr, xptr, optr);
        let (mut n_in_c, mut n_out_c, mut rb_c) =
            (n_in as c_int, n_out as c_int, plan.rb as c_int);
        let (mut rows_c, mut lanes_c) = (plan.rows_per_block as c_int, plan.lanes as c_int);
        let mut args = [
            &mut pw as *mut _ as *mut c_void,
            &mut px as *mut _ as *mut c_void,
            &mut po as *mut _ as *mut c_void,
            &mut n_in_c as *mut _ as *mut c_void,
            &mut n_out_c as *mut _ as *mut c_void,
            &mut rb_c as *mut _ as *mut c_void,
            &mut rows_c as *mut _ as *mut c_void,
            &mut lanes_c as *mut _ as *mut c_void,
        ];
        let nargs = if plan.lanes > 1 { 8 } else { 6 };
        unsafe {
            ctx.launch_async(
                plan.f,
                (plan.grid, 1, 1),
                (plan.block_dim, 1, 1),
                plan.shmem,
                &mut args[..nargs],
            )
        }
    }

    /// task 572: ミラーがこの層で何行持っているか (世代が違えば 0)。
    /// 読み取りだけで、台帳もバッファも触らない。エントリが無ければ `None`。
    fn q4_attn_mirror_have_rows(
        cache_id: u64,
        n_layer: usize,
        kv_dim: usize,
        layer: usize,
        gen: u64,
    ) -> Option<usize> {
        let mirrors = Q4_ATTN_MIRRORS.get()?;
        let ms = match mirrors.lock() {
            Ok(g) => g,
            Err(e) => e.into_inner(),
        };
        let (_, m, _) = ms.iter().find(|(k, _, _)| *k == cache_id)?;
        if m.kv_dim != kv_dim || m.state.len.len() != n_layer {
            return None;
        }
        Some(q4_mirror_have(&m.state, layer, usize::MAX, gen))
    }

    /// ミラーへ 1 行だけ D2D で追記する。全行シードが要るときは None。
    #[allow(clippy::too_many_arguments)]
    fn mirror_append(
        cache_id: u64,
        n_layer: usize,
        kv_dim: usize,
        layer: usize,
        gen: u64,
        len: usize,
        src_k: CUdeviceptr,
        src_v: CUdeviceptr,
    ) -> Option<(CUdeviceptr, CUdeviceptr)> {
        if len == 0 || layer >= n_layer {
            return None;
        }
        let mirrors = Q4_ATTN_MIRRORS.get_or_init(|| std::sync::Mutex::new(Vec::new()));
        let mut ms = match mirrors.lock() {
            Ok(g) => g,
            Err(e) => e.into_inner(),
        };
        let now = std::time::Instant::now();
        let mut meta: Vec<MirrorMeta> = ms
            .iter()
            .map(|(k, m, t)| MirrorMeta {
                key: *k,
                bytes: m.bytes(),
                last: *t,
                pending: false,
            })
            .collect();
        let evicted = lru_evict_keys(&mut meta, q4_attn_mirror_limit_bytes(), cache_id);
        if !evicted.is_empty() {
            ms.retain(|(k, _, _)| !evicted.contains(k));
        }
        let idx = match ms.iter().position(|(k, _, _)| *k == cache_id) {
            Some(i) => {
                ms[i].2 = now;
                i
            }
            None => {
                ms.push((cache_id, Q4AttnMirror::new(n_layer, kv_dim), now));
                ms.len() - 1
            }
        };
        if ms[idx].1.kv_dim != kv_dim || ms[idx].1.state.len.len() != n_layer {
            ms[idx].1 = Q4AttnMirror::new(n_layer, kv_dim);
        }
        let m = &mut ms[idx].1;
        if m.k[layer].is_none() || m.v[layer].is_none() {
            return None;
        }
        if len > m.state.cap || q4_mirror_cap(len, m.state.cap) > m.state.cap {
            return None;
        }
        let have = q4_mirror_have(&m.state, layer, len, gen);
        if have + 1 != len {
            return None;
        }
        let row_bytes = kv_dim * 4;
        let need = len * row_bytes;
        let (kb, vb) = (m.k[layer].as_ref()?, m.v[layer].as_ref()?);
        if kb.bytes() < need || vb.bytes() < need {
            return None;
        }
        let off = ((len - 1) * row_bytes) as u64;
        let lib = lib().ok()?;
        unsafe {
            if (lib.memcpy_dtod_async)(kb.ptr() + off, src_k, row_bytes, std::ptr::null_mut())
                != CUDA_SUCCESS
            {
                return None;
            }
            if (lib.memcpy_dtod_async)(vb.ptr() + off, src_v, row_bytes, std::ptr::null_mut())
                != CUDA_SUCCESS
            {
                return None;
            }
        }
        q4_mirror_note(&mut m.state, layer, len, gen);
        Some((kb.ptr(), vb.ptr()))
    }

    /// attention をデバイス入出力で 2 発打ち、出力のデバイスポインタを返す。
    #[allow(clippy::too_many_arguments)]
    fn attn_launch(
        ctx: &CudaCtx,
        qptr: CUdeviceptr,
        kptr: CUdeviceptr,
        vptr: CUdeviceptr,
        len: usize,
        kv_dim: usize,
        heads: usize,
        group: usize,
        dh: usize,
        scale: f32,
        // task 460: combine + ゲートの融合を使うときのゲート (0 = 融合しない)。
        gate: CUdeviceptr,
    ) -> Result<(CUdeviceptr, bool)> {
        let n_kv = heads / group;
        let qd = heads * dh;
        let mut kv_dim_c = kv_dim as c_int;
        let mut t_c = len as c_int;
        let mut group_c = group as c_int;
        let mut dh_c = dh as c_int;
        let mut scale_c = scale;
        let mut heads_c = heads as c_int;
        let mut nsplit_c;
        // 既定 (t179a の log と同じ) は「1024 未満は split・以上は flashdec」。
        if attn_flashdec_enabled() && len >= attn_flashdec_min_ctx() {
            if let Some(k) = fdec_kernels() {
                let (n_split, chunk) = q4_fdec_split(len, attn_flashdec_chunk_for(len));
                // task 448: タイル版 (K/V を shared へ協調ロード) を先に試す。
                // decode の実サーバ経路はここ (デバイス連鎖) なので、門もここで見る。
                // 形か PTX が欠けたら v1 (`dno_attn_flashdec`) で回す。
                let tile = attn_fdec_tile();
                let tile_on = attn_fdec_tile_on();
                fdt_gate_log(tile_on);
                fdt2_gate_log();
                // task 520: 第 3 版 (2 key 同時) -> 第 2 版 (tile) -> v1 の順に試す。
                let shape_ok = tile_on && fdt_shape_ok(dh, tile);
                let t2_k = if shape_ok && attn_fdec_ilp_on() {
                    fdt2_kernels()
                } else {
                    None
                };
                let tile_k = if t2_k.is_none() && shape_ok {
                    fdt_kernels()
                } else {
                    None
                };
                let (f_dec, shmem) = if let Some(k2) = t2_k {
                    fdt2_log(len, n_split, tile);
                    (k2.f_t2, fdt_smem_bytes(tile, dh) as u32)
                } else if let Some(kt) = tile_k {
                    fdt_log(len, n_split, chunk, dh, tile);
                    (kt.f_t, fdt_smem_bytes(tile, dh) as u32)
                } else {
                    fdec_log(len, n_split, chunk, kv_dim, dh, group);
                    (k.f_dec, 0u32)
                };
                let tile_arg = t2_k.is_some() || tile_k.is_some();
                let khx = t2_k.is_some() && attn_fdec_khx_on();
                // task 460: combine と出力ゲートの融合 (既定 on)。PTX か ABI が
                // 欠ければ None になり、下の従来 2 起動へ落ちる。
                let cbg_k = if gate != 0 && attn_fdec_cbg_on() {
                    fcbg_gate_log(true);
                    fcbg_kernels()
                } else {
                    fcbg_gate_log(gate != 0);
                    None
                };
                nsplit_c = n_split as c_int;
                let nkv_c = n_kv as c_int;
                let part_bytes = heads * n_split * (dh + ATTN_FDEC_PART_EXTRA) * 4;
                let gated = cbg_k.is_some();
                return Q4ATTN_SCRATCH.with(|cell| -> Result<(CUdeviceptr, bool)> {
                    let mut s = cell.borrow_mut();
                    Scratch::ensure(&mut s.out, qd * 4)?;
                    Scratch::ensure(&mut s.part, part_bytes)?;
                    let mut qptr_c = qptr;
                    let aptr = s.out.as_ref().expect("ensure 済み").ptr();
                    let mut aptr_c = aptr;
                    let mut partptr = s.part.as_ref().expect("ensure 済み").ptr();
                    let mut kptr_c = kptr;
                    let mut vptr_c = vptr;
                    let mut args: Vec<*mut c_void> = vec![
                        &mut qptr_c as *mut _ as *mut c_void,
                        &mut kptr_c as *mut _ as *mut c_void,
                        &mut vptr_c as *mut _ as *mut c_void,
                        &mut partptr as *mut _ as *mut c_void,
                        &mut kv_dim_c as *mut _ as *mut c_void,
                        &mut t_c as *mut _ as *mut c_void,
                        &mut group_c as *mut _ as *mut c_void,
                        &mut dh_c as *mut _ as *mut c_void,
                        &mut scale_c as *mut _ as *mut c_void,
                        &mut nsplit_c as *mut _ as *mut c_void,
                    ];
                    // task 448/520: タイル版と第 3 版だけ 11 本目 (tile 行数) を渡す。
                    let mut tile_c = tile as c_int;
                    if tile_arg {
                        args.push(&mut tile_c as *mut _ as *mut c_void);
                    }
                    // task 520 (2): 第 3 版だけ 12 本目 (grid 入れ替え) を渡す。
                    let mut khx_c = khx as c_int;
                    if t2_k.is_some() {
                        args.push(&mut khx_c as *mut _ as *mut c_void);
                    }
                    let tl = q4attn_kprof::stamp();
                    // task 520 (2): khx のときは (kh, sp) の順に並べる。
                    let grid = if khx {
                        (n_kv as u32, n_split as u32, 1)
                    } else {
                        (n_split as u32, n_kv as u32, 1)
                    };
                    unsafe {
                        ctx.launch_async(
                            f_dec,
                            grid,
                            ((32 * group) as u32, 1, 1),
                            shmem,
                            &mut args,
                        )?;
                    }
                    // task 460: 融合版 (combine + ゲート) があれば 1 起動で済ませる。
                    if let Some(kc) = cbg_k {
                        let mut gptr_c = gate;
                        let mut args3: Vec<*mut c_void> = vec![
                            &mut partptr as *mut _ as *mut c_void,
                            &mut aptr_c as *mut _ as *mut c_void,
                            &mut gptr_c as *mut _ as *mut c_void,
                            &mut group_c as *mut _ as *mut c_void,
                            &mut dh_c as *mut _ as *mut c_void,
                            &mut nsplit_c as *mut _ as *mut c_void,
                        ];
                        unsafe {
                            ctx.launch_async(
                                kc.f_cbg,
                                (group as u32, nkv_c as u32, 1),
                                (dh as u32, 1, 1),
                                0,
                                &mut args3,
                            )?;
                        }
                        q4attn_kprof::note_cbg_call();
                    } else {
                        let mut args2: Vec<*mut c_void> = vec![
                            &mut partptr as *mut _ as *mut c_void,
                            &mut aptr_c as *mut _ as *mut c_void,
                            &mut group_c as *mut _ as *mut c_void,
                            &mut dh_c as *mut _ as *mut c_void,
                            &mut nsplit_c as *mut _ as *mut c_void,
                        ];
                        unsafe {
                            ctx.launch_async(
                                k.f_cb,
                                (group as u32, nkv_c as u32, 1),
                                (dh as u32, 1, 1),
                                0,
                                &mut args2,
                            )?;
                        }
                    }
                    q4attn_kprof::note_launch(tl);
                    q4attn_kprof::note_call();
                    if t2_k.is_some() {
                        q4attn_kprof::note_fdt2_call();
                        q4attn_kprof::note_fdt_call();
                    } else if tile_k.is_some() {
                        q4attn_kprof::note_fdt_call();
                    } else {
                        q4attn_kprof::note_fdec_call();
                    }
                    Ok((aptr, gated))
                });
            }
        }
        // 従来の split 経路 (短い文脈はこちらが既定)。
        let (f_dec, f_cb) = match (ctx.f_attn_dec, ctx.f_attn_dec_cb) {
            (Some(a), Some(b)) => (a, b),
            _ => bail!("attention の split カーネルが無い"),
        };
        let (n_split, _chunk) = q4_dec_split(len, q4_gpu_attn_split());
        nsplit_c = n_split as c_int;
        let shmem = (q4_dec_sh_f(dh) * 4) as u32;
        Q4ATTN_SCRATCH.with(|cell| -> Result<(CUdeviceptr, bool)> {
            let mut s = cell.borrow_mut();
            let part_bytes = heads * n_split * (dh + 2) * 4;
            Scratch::ensure(&mut s.out, qd * 4)?;
            Scratch::ensure(&mut s.part, part_bytes)?;
            let mut qptr_c = qptr;
            let aptr = s.out.as_ref().expect("ensure 済み").ptr();
            let mut aptr_c = aptr;
            let mut partptr = s.part.as_ref().expect("ensure 済み").ptr();
            let mut kptr_c = kptr;
            let mut vptr_c = vptr;
            let mut args: Vec<*mut c_void> = vec![
                &mut qptr_c as *mut _ as *mut c_void,
                &mut kptr_c as *mut _ as *mut c_void,
                &mut vptr_c as *mut _ as *mut c_void,
                &mut partptr as *mut _ as *mut c_void,
                &mut kv_dim_c as *mut _ as *mut c_void,
                &mut t_c as *mut _ as *mut c_void,
                &mut group_c as *mut _ as *mut c_void,
                &mut dh_c as *mut _ as *mut c_void,
                &mut scale_c as *mut _ as *mut c_void,
                &mut nsplit_c as *mut _ as *mut c_void,
            ];
            let tl = q4attn_kprof::stamp();
            unsafe {
                ctx.launch_async(
                    f_dec,
                    (n_split as u32, heads_c as u32, 1),
                    (dh as u32, 1, 1),
                    shmem,
                    &mut args,
                )?;
            }
            let mut args2: Vec<*mut c_void> = vec![
                &mut partptr as *mut _ as *mut c_void,
                &mut aptr_c as *mut _ as *mut c_void,
                &mut nsplit_c as *mut _ as *mut c_void,
                &mut dh_c as *mut _ as *mut c_void,
            ];
            unsafe {
                ctx.launch_async(f_cb, (heads as u32, 1, 1), (dh as u32, 1, 1), 0, &mut args2)?;
            }
            q4attn_kprof::note_launch(tl);
            q4attn_kprof::note_call();
            q4attn_kprof::note_split_call();
            Ok((aptr, false))
        })
    }

    /// 新しい .cu (`kernels_dec_dev.ptx`) のカーネル。
    struct DevKernels {
        _module: CUmodule,
        f_norm: CUfunction,
        f_gate: CUfunction,
    }

    unsafe impl Send for DevKernels {}
    unsafe impl Sync for DevKernels {}

    fn dev_kernels() -> Result<&'static DevKernels> {
        static K: OnceLock<Option<DevKernels>> = OnceLock::new();
        let got = K.get_or_init(|| {
            let path = dec_dev_ptx_path();
            let mut image = match std::fs::read(&path) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("[full-dec-dev] PTX が読めない ({}) : {e}", path.display());
                    return None;
                }
            };
            image.push(0);
            let lib = lib().ok()?;
            let mut module: CUmodule = std::ptr::null_mut();
            unsafe {
                if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void)
                    != CUDA_SUCCESS
                {
                    eprintln!("[full-dec-dev] cuModuleLoadData 失敗");
                    return None;
                }
            }
            let mut f_norm: CUfunction = std::ptr::null_mut();
            let mut f_gate: CUfunction = std::ptr::null_mut();
            unsafe {
                if (lib.module_get_function)(
                    &mut f_norm,
                    module,
                    b"dno_dec_qknorm_rope\0".as_ptr() as *const c_char,
                ) != CUDA_SUCCESS
                    || (lib.module_get_function)(
                        &mut f_gate,
                        module,
                        b"dno_dec_gate_mul\0".as_ptr() as *const c_char,
                    ) != CUDA_SUCCESS
                {
                    eprintln!("[full-dec-dev] カーネルが PTX に無い");
                    return None;
                }
            }
            Some(DevKernels {
                _module: module,
                f_norm,
                f_gate,
            })
        });
        got.as_ref()
            .ok_or_else(|| anyhow!("kernels_dec_dev.ptx が無い"))
    }

    /// `DENNO_DEC_DEV_PTX` (デバッグ用) → 既定は manifest からの相対。
    fn dec_dev_ptx_path() -> std::path::PathBuf {
        if let Ok(p) = std::env::var("DENNO_DEC_DEV_PTX") {
            return std::path::PathBuf::from(p);
        }
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_dec_dev.ptx")
    }

    #[cfg(test)]
    pub(crate) mod tests {
        use super::*;

        /// env を見ずに on/off を固定する (assert が落ちても戻す)。
        struct Force(Option<bool>);
        impl Force {
            fn set(v: Option<bool>) -> Self {
                let prev = FORCE.with(|c| c.get());
                FORCE.with(|c| c.set(v));
                Force(prev)
            }
        }
        impl Drop for Force {
            fn drop(&mut self) {
                FORCE.with(|c| c.set(self.0));
            }
        }

        /// 実形 (n_embd 2560 / 24 q head / 2 kv head / dh 256 / n_rot 64) の入力。
        fn call_with<'a>(
            q_norm: &'a [f32],
            k_norm: &'a [f32],
            cur: &'a [f32],
            blocks: &'a [u8],
            units: &'a [Unit<'a>],
        ) -> Call<'a> {
            Call {
                cache_id: 1,
                n_layer: 48,
                layer: 3,
                layer_gen: 0,
                len: 3_965,
                kv_dim: 512,
                heads: 24,
                group: 12,
                dh: 256,
                scale: 0.0625,
                n_rot: 64,
                pos: 3_964,
                theta: 10_000.0,
                eps: 1e-6,
                q_norm,
                k_norm,
                qkv_units: units,
                qkv_k: 2_560,
                qkv_n: 13_312,
                q_off: 0,
                k_off: 12_288,
                v_off: 12_800,
                wo_uid: 2,
                wo_ty: crate::kquant::GgmlType::Q8_0,
                wo_blocks: blocks,
                wo_k: 6_144,
                wo_n: 2_560,
                n_embd: 2_560,
                cur,
                k_host: &[],
                v_host: &[],
                x_dev: None,
            }
        }

        /// task 572: 埋めてよい遅れかの真理値表。0 (既に足りている) と上限超えは
        /// 埋めない、1..=max だけ埋める。
        #[test]
        fn mirror_fill_gap_truth_table() {
            // 1 行遅れ (いちばん普通の回) は埋める。
            assert_eq!(mirror_fill_gap_ok(10, 9, 64), Some(1));
            // 数十行遅れも上限までなら埋める。
            assert_eq!(mirror_fill_gap_ok(100, 36, 64), Some(64));
            // 既に足りている (0) は埋めない。
            assert_eq!(mirror_fill_gap_ok(10, 10, 64), None);
            // ミラーが先行しているのも埋めない (負の差)。
            assert_eq!(mirror_fill_gap_ok(10, 12, 64), None);
            // 上限を 1 行でも超えたら断る (巨大な H2D を撃たない)。
            assert_eq!(mirror_fill_gap_ok(100, 35, 64), None);
            // 先頭 (host_len 0) も埋めない。
            assert_eq!(mirror_fill_gap_ok(0, 0, 64), None);
        }

        /// task 572: 門の解釈。未設定は既定 on、明示的な偽値だけ off。
        #[test]
        fn mirror_fill_gate_truth_table() {
            assert!(mirror_fill_from_value(None), "未設定で off になった");
            for v in ["", "0", "off", "false", "no", " OFF "] {
                assert!(
                    !mirror_fill_from_value(Some(v)),
                    "{v:?} で on になった"
                );
            }
            for v in ["1", "on", "true", "yes", " ON "] {
                assert!(mirror_fill_from_value(Some(v)), "{v:?} で off のまま");
            }
        }

        /// 実形の unit 1 本 (連結できた層と同じ形)。
        fn one_unit<'a>(blocks: &'a [u8]) -> Vec<Unit<'a>> {
            vec![Unit {
                uid: 1,
                ty: crate::kquant::GgmlType::Q8_0,
                blocks,
                k: 2_560,
                n: 13_312,
                out_row: 0,
            }]
        }

        /// 連結できない層 (3 本) も unit 列として通ること、隙間・行数違いは落ちること。
        #[test]
        fn units_gate_truth_table() {
            let qn = vec![1f32; 256];
            let kn = vec![1f32; 256];
            let cur = vec![0f32; 2_560];
            let blocks = vec![0u8; 1024];
            // 3 本 (q|gate / k / v) に分かれた形: 12,288 + 512 + 512 = 13,312。
            let three = vec![
                Unit {
                    uid: 1,
                    ty: crate::kquant::GgmlType::Q8_0,
                    blocks: &blocks,
                    k: 2_560,
                    n: 12_288,
                    out_row: 0,
                },
                Unit {
                    uid: 2,
                    ty: crate::kquant::GgmlType::Q5K,
                    blocks: &blocks,
                    k: 2_560,
                    n: 512,
                    out_row: 12_288,
                },
                Unit {
                    uid: 3,
                    ty: crate::kquant::GgmlType::Q5K,
                    blocks: &blocks,
                    k: 2_560,
                    n: 512,
                    out_row: 12_800,
                },
            ];
            let c = call_with(&qn, &kn, &cur, &blocks, &three);
            assert!(c.ok_shape(), "3 本に分かれた形が通らない");
            // 行の隙間。
            let mut gap = three.clone();
            gap[1].out_row = 12_289;
            let c = call_with(&qn, &kn, &cur, &blocks, &gap);
            assert!(!c.ok_shape());
            // 合計行数が足りない。
            let mut short = three.clone();
            short.pop();
            let c = call_with(&qn, &kn, &cur, &blocks, &short);
            assert!(!c.ok_shape());
            // 入力長が違う。
            let mut badk = three.clone();
            badk[2].k = 2_304;
            let c = call_with(&qn, &kn, &cur, &blocks, &badk);
            assert!(!c.ok_shape());
            // 空。
            let empty: Vec<Unit<'_>> = Vec::new();
            let c = call_with(&qn, &kn, &cur, &blocks, &empty);
            assert!(!c.ok_shape());
            // 連結時に並びが変わった層 (カーネルの前提と違う) は落ちる。
            let one = one_unit(&blocks);
            let mut c = call_with(&qn, &kn, &cur, &blocks, &one);
            assert!(c.ok_shape());
            c.k_off = 0;
            c.q_off = 512;
            assert!(!c.ok_shape());
        }

        /// ゲートを off にすると GPU に一切触らない (従来経路へ)。
        /// 既定は lab t179c の採否で on (下の `DEFAULT_ON`)。
        #[test]
        fn gate_off_makes_forward_inert() {
            let _f = Force::set(Some(false));
            assert!(!on());
            let qn = vec![1f32; 256];
            let kn = vec![1f32; 256];
            let cur = vec![0f32; 2_560];
            let blocks = vec![0u8; 1024];
            let units = one_unit(&blocks);
            let c = call_with(&qn, &kn, &cur, &blocks, &units);
            assert!(c.ok_shape(), "実形がゲートを通らない");
            // off のときは GPU に一切触らず None (従来経路へ)。
            assert!(forward(&c).is_none());
        }

        #[test]
        fn shape_gate_truth_table() {
            let qn = vec![1f32; 256];
            let kn = vec![1f32; 256];
            let cur = vec![0f32; 2_560];
            let blocks = vec![0u8; 1024];
            let units = one_unit(&blocks);
            let base = call_with(&qn, &kn, &cur, &blocks, &units);
            assert!(base.ok_shape());
            // n_rot が表の上限 (128) を超える。
            let mut c = call_with(&qn, &kn, &cur, &blocks, &units);
            c.n_rot = 130;
            assert!(!c.ok_shape());
            // kv_dim が n_kv * dh と違う。
            let mut c = call_with(&qn, &kn, &cur, &blocks, &units);
            c.kv_dim = 256;
            assert!(!c.ok_shape());
            // qkv の行数が [q|gate|k|v] と違う。
            let mut c = call_with(&qn, &kn, &cur, &blocks, &units);
            c.qkv_n = 12_288;
            assert!(!c.ok_shape());
            // 入力長が k と違う。
            let mut c = call_with(&qn, &kn, &cur, &blocks, &units);
            c.qkv_k = 2_304;
            assert!(!c.ok_shape());
            // 出力投影の入力長が q_dim と違う。
            let mut c = call_with(&qn, &kn, &cur, &blocks, &units);
            c.wo_k = 6_400;
            assert!(!c.ok_shape());
            // head 数が group で割り切れない。
            let mut c = call_with(&qn, &kn, &cur, &blocks, &units);
            c.group = 5;
            assert!(!c.ok_shape());
            // len が 0 / 65536。
            let mut c = call_with(&qn, &kn, &cur, &blocks, &units);
            c.len = 0;
            assert!(!c.ok_shape());
            let mut c = call_with(&qn, &kn, &cur, &blocks, &units);
            c.len = 65_536;
            assert!(!c.ok_shape());
        }

        /// ミラーが「全行シードが要る」状況では None を返す (呼び出し側が従来経路へ)。
        /// このコンテナには GPU が無いので、ミラーが空のまま = None が正しい。
        #[test]
        fn mirror_append_refuses_without_a_seeded_mirror() {
            let r = mirror_append(0xdead_beef, 48, 512, 3, 0, 2, 0, 0);
            assert!(r.is_none(), "種の無いミラーへ 1 行だけ書こうとした");
            // len = 0 / 層が範囲外も None。
            assert!(mirror_append(0xdead_beef, 48, 512, 3, 0, 0, 0, 0).is_none());
            assert!(mirror_append(0xdead_beef, 48, 512, 48, 0, 1, 0, 0).is_none());
        }
    }
}


// ---------------------------------------------------------------------------
// s138 (Task B6・計測のみ): decode chain のステップ分割 (表 A) と、線形ラッパの
// 分割 (表 B) のための小道具。**表示専用** (prof.rs の SPLIT_* にしか積まない)。
//
// 前提: chain (CUDA 部分) は 1 ステップで使う GPU 資源が 1 本の stream に直列で、
// 呼び出しスレッドも 1 本。したがって「段」は thread-local の 1 個の状態で足りる
// (BF/多セッションの chain はスレッドごとに別 state = 混ざらない)。
// ---------------------------------------------------------------------------

thread_local! {
    /// 現在走っている chain の段 (prof.rs の CHAIN_*)。0 = 計測していない。
    pub(crate) static CHAIN_SEG: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// 1 ステップ分の積算 (秒)。`take` で取り出す。
    pub(crate) static CHAIN_ACC: std::cell::RefCell<[f64; crate::prof::SPLIT_ROWS]> =
        const { std::cell::RefCell::new([0.0; crate::prof::SPLIT_ROWS]) };
    /// 段の切り替え時刻。
    pub(crate) static CHAIN_TICK: std::cell::Cell<Option<Instant>> = const { std::cell::Cell::new(None) };
    /// どの段からも取れなかった sync の残余 (秒)。ステップ終端で待ち行へ入れる。
    pub(crate) static CHAIN_RESID: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
}


/// 段の id を (表, 行) へ畳む。chain 用の id かどうかだけ見ればよい。
#[inline]
fn chain_row(op: usize) -> Option<usize> {
    match crate::prof::split_slot(op) {
        Some((t, i)) if t == crate::prof::SPLIT_CHAIN_TABLE => Some(i),
        _ => None,
    }
}


/// 現在の段に `secs` を足す (段が 0 = 計測していないなら何もしない)。
#[inline]
fn chain_charge(secs: f64) {
    let cur = CHAIN_SEG.with(|c| c.get());
    if cur == 0 {
        return;
    }
    if let Some(i) = chain_row(cur) {
        CHAIN_ACC.with(|a| a.borrow_mut()[i] += secs);
    }
}


/// 段を `op` へ切り替える。計測していない (段 0) ときは何もしない —
/// 呼び出し側は `chain_step_begin` を打ったステップの中でだけ切り替える。
#[inline]
pub fn chain_seg(op: usize) {
    if !crate::prof::split_on() {
        return;
    }
    let cur = CHAIN_SEG.with(|c| c.get());
    if cur == 0 || chain_row(op).is_none() {
        return;
    }
    let now = Instant::now();
    let prev = CHAIN_TICK.with(|c| c.replace(Some(now)));
    if let Some(t0) = prev {
        chain_charge(now.duration_since(t0).as_secs_f64());
    }
}


/// 直前の段 (`op`) を締めて、**同じ段**の tick を打ち直す。
/// ステップ内で同じ段が続く区間 (発行の合間) の区切りに使う。計測 OFF は分岐 1 回。
#[inline]
pub fn chain_seg_mark(op: usize) {
    if !crate::prof::split_on() {
        return;
    }
    let cur = CHAIN_SEG.with(|c| c.get());
    if cur == 0 || cur != op || chain_row(op).is_none() {
        return;
    }
    let now = Instant::now();
    let prev = CHAIN_TICK.with(|c| c.replace(Some(now)));
    if let Some(t0) = prev {
        chain_charge(now.duration_since(t0).as_secs_f64());
    }
}


/// s140: 「このステップは chain 本体まで到達した」ことを表 A のサンプル数に数える。
/// 1 ステップ 1 回 (`forward_decode_chain` が段の切り替えを始める直前)。
#[inline]
pub fn chain_step_entered() {
    if !crate::prof::split_on() {
        return;
    }
    if CHAIN_SEG.with(|c| c.get()) == 0 {
        return;
    }
    crate::prof::split_tick_call(crate::prof::SPLIT_CHAIN_TABLE);
}


/// `op` の区間だけを測る (段を切り替えずに、その間の時間を `op` へ引く)。
///
/// 使いどころは「sync の中に入った 1 本のカーネル実行」— decode では
/// `cuCtxSynchronize` が直前に発行したカーネルの完了を待つ時間が支配的なので、
/// 発行直後と sync 直後に同じ tick を読み、その差を `kernel_wait` として
/// 待ち行から分離する (待ちの合計は動かない)。計測 OFF はクロック読みゼロ。
#[inline]
fn chain_mark_span(op: usize, t0: Instant) {
    if !crate::prof::split_on() {
        return;
    }
    if CHAIN_SEG.with(|c| c.get()) == 0 {
        return;
    }
    if let Some(i) = chain_row(op) {
        CHAIN_ACC.with(|a| a.borrow_mut()[i] += t0.elapsed().as_secs_f64());
    }
}


/// 計測するなら今の時刻 (呼び出し側はこれが None なら何もしない)。
/// 公開しているのは「発行したカーネルの完了待ち」を `chain_kernel` へ移すため —
/// 呼び出し側 (decode chain) が発行の直後と sync の直後に 1 回ずつ読む。
#[inline]
pub fn chain_tick() -> Option<Instant> {
    if crate::prof::split_on() {
        Some(Instant::now())
    } else {
        None
    }
}


thread_local! {
    /// 直近の `chain_sync_observed` が観測した (sync の所要秒, 直前に発行した
    /// カーネルの実行秒)。計測 OFF では None のまま = 呼び出し側は何もしない。
    pub(crate) static CHAIN_SIG: std::cell::Cell<Option<(f64, f64)>> = const { std::cell::Cell::new(None) };
}


/// chain の sync を分割表の観点で観測する (`chain_sync_pending` の直後に 1 回)。
///
/// `secs` = sync の所要、`kernel_secs` = 直前に発行したカーネルの tick からの経過。
/// 両方が揃っているときだけ「カーネル実行 = sync 所要のうち tick からの経過」を
/// `chain_kernel` へ移し、待ち行から同じ秒数を引く (待ちの合計は動かない)。
/// `chain_step_begin` していない (段 0) や計測 OFF では何もしない。
///
/// ここで観測した値は `chain_sig` が 1 回だけ返す (表へは呼び出し側が流す)。
pub fn chain_sync_observed(secs: f64, kernel_secs: Option<f64>) {
    if !crate::prof::split_on() {
        return;
    }
    if CHAIN_SEG.with(|c| c.get()) == 0 {
        return;
    }
    // カーネル実行は待ちを超えない (tick は sync の直前に打つので、超えるのは
    // 時計の粒度か別スレッドの割り込み) — 超えた分は待ち側に残す。
    let sig = match kernel_secs {
        Some(k) if k > 0.0 && secs > 0.0 => (secs, k.min(secs)),
        _ => (secs, 0.0),
    };
    CHAIN_SIG.with(|c| c.set(Some(sig)));
}


/// `chain_sync_observed` が観測した (sync 秒, カーネル実行秒) を 1 回だけ返す。
/// 計測 OFF / 未観測は None (呼び出し側は何もしない)。
#[inline]
pub fn chain_sig() -> Option<(f64, f64)> {
    CHAIN_SIG.with(|c| c.replace(None))
}


/// ステップ先頭: chain の計測を開き、最初の段を `CHAIN_OUTER` (embedding 用意・
/// reserve など chain の外側) にする。呼び出しは decode chain の入口 1 か所だけで、
/// 対応する `chain_step_end` がステップ末尾に 1 か所ある (失敗して途中で抜ける
/// ステップはそこまでが積まれ、以後のステップで上書きされる)。
///
/// 計測 OFF は分岐 1 回 (クロックも読まない)。
pub fn chain_step_begin() {
    // s166 (P5): 観測カウンタのリセット (env OFF は分岐 1 回) — prefill で積んだ
    // 分を decode の数字に混ぜない。
    decode_dbg_reset();
    if !crate::prof::split_on() {
        return;
    }
    CHAIN_ACC.with(|a| *a.borrow_mut() = [0.0; crate::prof::SPLIT_ROWS]);
    CHAIN_RESID.with(|c| c.set(0.0));
    CHAIN_TICK.with(|c| c.set(Some(Instant::now())));
    CHAIN_SEG.with(|c| c.set(crate::prof::CHAIN_OUTER));
}


/// ステップ末尾: 最後の段を締め、1 ステップ分を表 A へ流し込む。呼び出しは
/// `chain_step_begin` と対で 1 か所。計測 OFF は分岐 1 回。
pub fn chain_step_end() {
    // s166 (P5): 1 ステップ分の launch / copy 件数を stderr へ (env OFF は分岐 1 回)。
    decode_dbg_flush();
    if !crate::prof::split_on() {
        return;
    }
    if CHAIN_SEG.with(|c| c.get()) == 0 {
        return; // begin していない (別の経路) — 何も積まない
    }
    // 最後の段をここで締める (tick はこの 1 回だけ)
    if let Some(t0) = CHAIN_TICK.with(|c| c.replace(None)) {
        chain_charge(Instant::now().duration_since(t0).as_secs_f64());
    }
    CHAIN_SEG.with(|c| c.set(0));
    chain_step_flush();
}


/// ステップ末尾: 最後の段の区間を締めて、1 ステップ分を表 A へ流し込む。
/// 締めの tick は `chain_step_end` (呼び出し側の 1 行) の側で読む。
///
/// 「どの sync からも取れなかった待ち」(`CHAIN_RESID`) は**待ち行**へ入れる —
/// sync 区間の理解できない残余は発行ではなく待ちだから (排他の和は動かない)。
fn chain_step_flush() {
    let rows = CHAIN_ACC.with(|a| {
        let acc = &mut *a.borrow_mut();
        let rows = *acc;
        *acc = [0.0; crate::prof::SPLIT_ROWS];
        rows
    });
    let mut acc = rows;
    let resid = CHAIN_RESID.with(|c| c.replace(0.0));
    if resid > 0.0 {
        // どの sync からも取れなかった待ちは残余行 (= `CHAIN_SYNC_RESIDUAL`) へ。
        // 待ちの合計は動かない (残余も待ちとして数える)。
        acc[crate::prof::CHAIN_SYNC_RESIDUAL - crate::prof::SPLIT_CHAIN_FIRST] += resid;
    }
    crate::prof::split_flush_step(crate::prof::SPLIT_CHAIN_TABLE, &acc, 1, None);
}


/// chain の sync (`cuCtxSynchronize`) を「段」の観点で観測する。
///
/// sync の所要 `secs` と、sync の直前まで走っていた段 `prev` が分かるとき、
/// その待ちの帰属は 1 つに決まる:
///   - prev が launch 系 (`chain_launch`) … 直前に発行したカーネルの完了待ち。
///     `chain_kernel` として積み、待ち行 (`prev`) からは **同じ秒数を引く**
///     (待ちの合計は動かない = 排他の和が保たれる)。
///   - prev が既に待ち (`chain_sync_wait` / `chain_sync_resid`) … 前の sync で
///     取り切れなかった分。`chain_sync_wait` に積む。
///   - それ以外 … 待ち行へ (残余はステップ末尾でまとめて待ち行に寄せる)。
///
/// `kernel` は直前に発行した 1 本のカーネルの実行時間 (発行直後と sync 直後の
/// tick の差)。計測 OFF なら `secs` / `kernel` は読まれない (呼び出し側が None)。
#[inline]
fn chain_observe_sync(secs: f64, kernel: Option<f64>) {
    if !crate::prof::split_on() {
        return;
    }
    let prev = CHAIN_SEG.with(|c| c.get());
    if prev == 0 {
        return;
    }
    let wait_row = crate::prof::CHAIN_SYNC_WAIT - crate::prof::SPLIT_CHAIN_FIRST;
    let launch_row = crate::prof::CHAIN_LAUNCH - crate::prof::SPLIT_CHAIN_FIRST;
    let kern_row = crate::prof::CHAIN_KERNEL_WAIT - crate::prof::SPLIT_CHAIN_FIRST;
    match (prev == crate::prof::CHAIN_LAUNCH, kernel) {
        // launch 直後の sync: カーネル実行 = 待ちの本体
        (true, Some(k)) if k > 0.0 => {
            CHAIN_ACC.with(|a| {
                let mut a = a.borrow_mut();
                a[kern_row] += k;
                // 待ちの合計を二重にしないため、帰属元の launch から引く
                a[launch_row] -= k;
                if secs > k {
                    a[wait_row] += secs - k;
                }
            });
        }
        // launch 以外からの sync は待ち行へ (残余は末尾で待ち行に寄せる)
        _ => {
            if secs > 0.0 {
                CHAIN_ACC.with(|a| a.borrow_mut()[wait_row] += secs);
            }
        }
    }
}


pub(crate) fn q4_alias_table() -> &'static std::sync::Mutex<std::collections::HashMap<u64, (u64, usize)>> {
    Q4_ALIAS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}


/// s165: `concat` (行連結済みの 1 本) の行範囲を `parts` の各 uid に割り当てる。
/// 呼び出し側は「parts の k がすべて concat.k と一致し、行数の和が concat.n」を
/// 満たすときだけ呼ぶ。**前提を満たさない・Q4 でない・転送に失敗した場合は
/// 何も登録せず false を返す** → 呼び出し側は従来どおり個別転送にフォールバック
/// する (fail closed)。登録済み (既に個別に VRAM へ載っている) uid は、in-flight
/// のカーネルが読みうるため置き換えない (その層は節約なし・数値は不変)。
pub fn q4_register_row_alias(concat: &crate::quant::Linear, parts: &[&crate::quant::Linear]) -> bool {
    let Some(c) = q4_of(concat) else { return false };
    if parts.is_empty() {
        return false;
    }
    let mut qs: Vec<&crate::quant::Q4Linear> = Vec::with_capacity(parts.len());
    for p in parts {
        let Some(p) = q4_of(p) else { return false };
        qs.push(p);
    }
    q4_register_row_alias_q4(c, &qs)
}


/// s166 (P5): [`q4_register_row_alias`] の Q4 版 (decode chain の qkv 連結用)。
pub(crate) fn q4_register_row_alias_q4(
    c: &crate::quant::Q4Linear,
    parts: &[&crate::quant::Q4Linear],
) -> bool {
    if parts.is_empty() || !q4_dedup_enabled() {
        return false;
    }
    // 形状契約: parts はすべて Q4・k は concat と一致・行数の和が concat.n。
    let mut geom = Vec::with_capacity(parts.len());
    let mut uids = Vec::with_capacity(parts.len());
    for p in parts {
        geom.push((p.k, p.n));
        uids.push(p.uid);
    }
    let Some(rows) = q4_row_alias_offsets(c.k, c.n, &geom) else {
        return false;
    };
    let offs: Vec<(u64, usize)> = uids.into_iter().zip(rows).collect();
    // s178 (P59): 既に「連結側が載っていて parts がその行範囲」なら何もしない。
    if q4_fused_ready(c.uid, parts) {
        return true;
    }
    // s178 (P59): parts が**自分の実体として**載っているなら連結側を上げない
    // (上げると同じパラメータが VRAM に 2 重に載る = P59 で削った分が戻る)。
    // エイリアスビューとして載っている場合は実体ではないので、上の
    // `q4_fused_ready` が先に true を返してここへは来ない。
    if parts.iter().any(|p| q4_resident(p.uid)) {
        return false;
    }
    // 連結側を先にレジストリへ載せる (エイリアスの参照先が存在する状態にする)。
    if weights_for_q4(c.uid, &c.blocks, c.k, c.n).is_err() {
        return false;
    }
    let table = q4_alias_table();
    let mut t = match table.lock() {
        Ok(t) => t,
        Err(_) => return false,
    };
    for (uid, row_off) in offs {
        if uid == c.uid {
            return false;
        }
        t.entry(uid).or_insert((c.uid, row_off));
    }
    true
}


/// s165: `uid` にエイリアスが登録されていれば、連結側のデバイス領域を指す
/// ビューを `map` に登録して返す。形状・サイズが 1 つでも食い違えば `None`
/// (呼び出し側は通常の転送へ落ちる = fail closed)。
///
/// s178 (P59): 範囲判定を [`q4_view_geo`] に切り出し、先頭の part (row_off = 0)
/// もビューになるようにした。`DENNO_Q4_DEDUP=0` では常に `None`
/// (エイリアスを使わない = 各 uid を個別に転送する従来動作)。
pub(crate) fn q4_alias_view(
    uid: u64,
    k: usize,
    n: usize,
    map: &mut std::collections::HashMap<u64, Box<GpuWq4>>,
) -> Option<&'static GpuWq4> {
    if !q4_dedup_enabled() || map.contains_key(&uid) {
        return None;
    }
    let (tgt, row_off) = {
        let t = q4_alias_table().lock().ok()?;
        *t.get(&uid)?
    };
    // 参照先 (連結側) がまだ転送されていなければ何もしない。
    let (nib_ptr, sc_ptr, tn, tk) = {
        let e = map.get(&tgt)?;
        (e.nib.ptr(), e.sc.ptr(), e.n, e.k)
    };
    if nib_ptr == 0 || sc_ptr == 0 {
        return None;
    }
    let (nib_off, nib_bytes, sc_off, sc_bytes) = q4_view_geo(k, n, row_off, tk, tn)?;
    // 範囲が連結側バッファ内に収まることを確認 (収まらなければ fail closed)。
    {
        let e = map.get(&tgt)?;
        if nib_off + nib_bytes > e.nib.bytes() || sc_off + sc_bytes > e.sc.bytes() {
            return None;
        }
    }
    let v = GpuWq4 {
        nib: DevBuffer::view(nib_ptr + nib_off as u64, nib_bytes),
        sc: DevBuffer::view(sc_ptr + sc_off as u64, sc_bytes),
        k,
        n,
    };
    map.insert(uid, Box::new(v));
    let e = map.get(&uid).expect("just inserted");
    Some(unsafe { std::mem::transmute::<&GpuWq4, &'static GpuWq4>(e.as_ref()) })
}


/// s123: Q4_0 の 1 層ぶん。小さい m は GEMV (1 warp = 1 行)、大きい m は
/// タイル GEMM (1 block = 16 行 x 256 列)。H2D → kernel → D2H → sync ×1。
pub(crate) fn q4_dispatch(
    ctx: &CudaCtx,
    a: &[f32],
    gw: &GpuWq4,
    m: usize,
    k: usize,
    n: usize,
    use_gemm: bool,
    out: &mut [f32],
) -> Result<()> {
    // s124 (P4-6): m を固定タイルに割る。`SCRATCH` は thread_local かつ grow-only
    // なので、プロンプト長そのままの m を渡すと (m*k + m*n)*4 B が GPU 経路を
    // 通ったスレッド数ぶん積み上がる (27B・10.8k token で約 11 GB になり、続く
    // リクエストが cuMemAlloc error 2 で CPU へ落ちた)。
    // 行は互いに独立で、1 行の演算順はタイル分割で変わらないため出力は bit 一致する。
    // s168 (P7): タイル幅は `DENNO_Q4_M_TILE` の実効値、端数は直前のタイルへ併合する。
    let tile = q4_m_tile();
    let lib = lib()?;
    ctx.make_current()?;
    let mut r0 = 0;
    while r0 < m {
        let mr = q4_group_rows(r0, m, tile);
        let a_t = &a[r0 * k..r0 * k + mr * k];
        let out_t = &mut out[r0 * n..r0 * n + mr * n];
        q4_dispatch_tile(ctx, lib, a_t, gw, mr, k, n, use_gemm, out_t)?;
        r0 += mr;
    }
    Ok(())
}


/// s124: [`q4_dispatch`] の 1 タイル (m <= 256) を実行する。scratch はここで
/// m <= 256 に固定されるため、プロンプトが何万 token でも増えない。
pub(crate) fn q4_dispatch_tile(
    ctx: &CudaCtx,
    lib: &'static CudaLib,
    a: &[f32],
    gw: &GpuWq4,
    m: usize,
    k: usize,
    n: usize,
    use_gemm: bool,
    out: &mut [f32],
) -> Result<()> {
    ctx.make_current()?;
    // s148 (Task B10・計測のみ): serve (q4) 経路の線形呼び出しの中身を段へ割る。
    // s138 の表 B は wrapper (`lin_stage` / `linear_pinned`) の内側だけを数えて
    // いたため、本番の GEMM 経路 (`q4_dispatch_tile`) の時間に届いていなかった
    // (実測: 608 call / 7,846 ms に対し 802 サンプル / 445.8 ms)。ここは本番が
    // 実際に通るタイル実装の内側で、1 タイル = 1 サンプル。計測 OFF は tick が
    // `None` で、`split_mark` / `span_end` とも分岐 1 回で戻る (クロック読みなし)。
    let mut t = split_stage();
    // s148: 表 C の段。1 タイルの中で互いに排他 (確保 / H2D / launch / D2H / sync)。
    //
    // `st` は「いま開いている段」で、閉じるたびに**開き直さない** — `span_begin` は
    // 計測 OFF で `None` を返すので、閉じた後に開き直すと (計測 ON でも) ここで
    // 落とした区間が二度と閉じられない。各段の開始は必ず次の段を閉じた直後で取る。
    // こうすると段の和が「タイル入口 → 戻り」の壁時計に一致し、閉じ漏れが
    // `sl_lin_other` (残余) ではなく**未帰属の穴**として現れない。
    let mut st = crate::prof::span_begin();
    SCRATCH.with(|cell| -> Result<()> {
        let mut s = cell.borrow_mut();
        let Scratch { x, out: dout, hx, ho, .. } = &mut *s;
        let bytes_x = m * k * 4;
        let bytes_out = m * n * 4;
        Scratch::ensure_arena(x, ARENA_KEY_Q4_X, bytes_x, crate::mem_audit::TAG_GEMM)?;
        Scratch::ensure_arena(dout, ARENA_KEY_Q4_OUT, bytes_out, crate::mem_audit::TAG_GEMM)?;
        split_mark(&mut t, crate::prof::Q4P_OUT_ALLOC);
        // s153 (Task B14): クロック読みは 1 回 (`lin_stage_taken_ns`)。読んだ ns を
        // そのまま表 C へ積むので、子の和が親 (`sl_lin`) を超えず、
        // `sl_lin_other = 全体 - 段の和` が成り立つ。
        let out_ns = crate::prof::lin_stage_taken_ns(st.take());
        crate::prof::span_end_ns(out_ns, crate::prof::SL_LIN_OUTALLOC);
        // 表 C: staging のピン留め領域の確保 (段の開始はここ)
        st = crate::prof::span_begin();
        Scratch::ensure_pin(hx, bytes_x)?;
        Scratch::ensure_pin(ho, bytes_out)?;
        split_mark(&mut t, crate::prof::Q4P_PIN_ALLOC);
        // 表 C: ピン留め確保は転送準備の一部なので `sl_lin_other` に寄せる
        // (`sl_lin_quant` は Q4 では行わない = 0 サンプルのまま)。
        // (割り当て済みの 2 本目以降は即時なので、ここは実質初回だけの時間。)
        crate::prof::lin_other_close(st.take());
        // 表 C: 活性化の staging コピー (hx ← a) から H2D 発行まで
        st = crate::prof::span_begin();
        let (dx, dout) = (x.as_ref().unwrap(), dout.as_ref().unwrap());
        let (hx, ho) = (hx.as_mut().unwrap(), ho.as_mut().unwrap());
        hx.as_mut_slice::<f32>()[..m * k].copy_from_slice(a);
        split_mark(&mut t, crate::prof::Q4P_H2D_STAGE);
        unsafe {
            check(
                (lib.memcpy_htod_async)(
                    dx.ptr(),
                    hx.ptr as *const c_void,
                    bytes_x,
                    std::ptr::null_mut(),
                ),
                "cuMemcpyHtoDAsync",
            )?;
            split_mark(&mut t, crate::prof::Q4P_H2D_ISSUE);
            let h2d_ns = crate::prof::lin_stage_taken_ns(st.take());
            crate::prof::span_end_ns(h2d_ns, crate::prof::SL_LIN_H2D);
            // 表 C: 引数の組み立て + カーネル発行 (段の開始 = H2D を閉じた直後)
            st = crate::prof::span_begin();
            let (mut pa, mut pn, mut ps, mut po) =
                (dx.ptr(), gw.nib.ptr(), gw.sc.ptr(), dout.ptr());
            let (mut mc, mut kc, mut nc) = (m as c_int, k as c_int, n as c_int);
            // s132: 1 block が受ける行数 (mr)。m = 1 は従来の 1 行 1 block 経路
            // (dno_gemv_q4_0) をそのまま通すため、mr は 2 以上でだけ使う。
            let mut mrc = (m.min(Q4_MR_MAX)) as c_int;
            let mut args_mr = vec![
                &mut pa as *mut _ as *mut c_void,
                &mut pn as *mut _ as *mut c_void,
                &mut ps as *mut _ as *mut c_void,
                &mut po as *mut _ as *mut c_void,
                &mut mc as *mut _ as *mut c_void,
                &mut kc as *mut _ as *mut c_void,
                &mut nc as *mut _ as *mut c_void,
                &mut mrc as *mut _ as *mut c_void,
            ];
            let mut args = [
                &mut pa as *mut _ as *mut c_void,
                &mut pn as *mut _ as *mut c_void,
                &mut ps as *mut _ as *mut c_void,
                &mut po as *mut _ as *mut c_void,
                &mut mc as *mut _ as *mut c_void,
                &mut kc as *mut _ as *mut c_void,
                &mut nc as *mut _ as *mut c_void,
            ];
            if use_gemm {
                // Q4_GM=64 (m 方向) / Q4_GN=128 (n 方向)・block (16,8)
                let grid = (m.div_ceil(64) as u32, n.div_ceil(128) as u32, 1);
                ctx.launch_async(ctx.f_gemm_q4, grid, (16, 8, 1), 0, &mut args)?;
            } else if let Some(fc) = ctx.f_mv_q4_c {
                // s133 (Task B1, attempt 2): 1 block = 8 warp x C 列 = 8*C 列、
                // 1 行。1 warp が A の 32 値 (ブロック b) をレジスタに 1 回だけ
                // 読み、C 列ぶんの重みに使い回す。行ごとの累積順は 1 行 1 block
                // 経路と同一なので m = 1 もビット一致。
                // grid.x は 8*C 列を 1 ブロックで受ける形。
                // s169 (P10): C は n から選ぶ (`DENNO_GEMV_SHAPE`。既定 0 = 従来)。
                let c = gemv_c_for_n(n, gemv_sm_count(), gemv_shape_mode(), q4_gemv_c());
                let f_c = fc[q4_gemv_c_index(c)];
                let cols = 8 * c;
                let grid = (n.div_ceil(cols) as u32, m as u32, 1);
                ctx.launch_async(f_c, grid, (256, 1, 1), 0, &mut args)?;
            } else if let (Some(f_mr), 2..) = (ctx.f_mv_q4_mr, m) {
                // s132 (Task B1): 1 block = 8 列 (8 warp) x mr 行。重みは 1 回だけ
                // 読んで mr 行に使い回す (従来は grid.y = m で行ごとに再読み)。
                // mr = min(m, 8)・端数は最終 block の rows = m - i0 で落とす。
                // 列ブロック版が PTX に無いときだけここを通る (フォールバック)。
                let grid = (n.div_ceil(8) as u32, m.div_ceil(mrc as usize) as u32, 1);
                ctx.launch_async(f_mr, grid, (256, 1, 1), 0, &mut args_mr)?;
            } else {
                // 1 block = 8 行 (8 warp)・grid.x = n/8, grid.y = m
                let grid = (n.div_ceil(8) as u32, m as u32, 1);
                ctx.launch_async(ctx.f_mv_q4, grid, (256, 1, 1), 0, &mut args)?;
            }
            split_mark(&mut t, crate::prof::Q4P_KERNEL_LAUNCH);
            // 表 C: 発行段の終わり (ここまでが launch、以降は D2H と待ち)
            let launch_ns = crate::prof::lin_stage_taken_ns(st.take());
            crate::prof::span_end_ns(launch_ns, crate::prof::SL_LIN_LAUNCH);
            // 表 C: 出力 D2H の段は「発行の直前」から開く (sync 完了 + ホストコピーまで)
            st = if st.is_none() { crate::prof::span_begin() } else { st };
            check(
                (lib.memcpy_dtoh_async)(ho.ptr, dout.ptr(), bytes_out, std::ptr::null_mut()),
                "cuMemcpyDtoHAsync",
            )?;
            split_mark(&mut t, crate::prof::Q4P_D2H_ISSUE);
            crate::prof::mark_d2h_src(crate::prof::D2H_SRC_LINEAR);
            ctx.sync()?;
            split_mark(&mut t, crate::prof::Q4P_SYNC_WAIT);
        }
        out.copy_from_slice(&ho.as_slice::<f32>()[..m * n]);
        split_flush(&mut t);
        // 表 C: 出力 D2H = 発行 + sync の完了待ち + ピン留め領域 → out のコピー。
        // 待ち (`SL_LIN_SYNC`) は別に取らない — D2H の中に吸収される (排他の和を保つ)。
        // `SL_LIN_SYNC` は表 C の行として予約されているが、q4 タイルでは常に 0 サンプル
        // で、その分はこの行 (D2H) に含まれる (COVERAGE には 0 として出る)。
        let d2h_ns = crate::prof::lin_stage_taken_ns(st);
        crate::prof::span_end_ns(d2h_ns, crate::prof::SL_LIN_D2H);
        Ok(())
    })
}


thread_local! {
    pub(crate) static PF_DEV: std::cell::RefCell<PfDev> = const { std::cell::RefCell::new(PfDev::new()) };
    /// task 200: ssm 連鎖がデバイスに残した層出力 (直後の combine が転送なしで読む)。
    pub(crate) static PF_ATTN_DEV_OUT: std::cell::RefCell<Option<(CUdeviceptr, usize, usize)>> =
        const { std::cell::RefCell::new(None) };
    /// task 200: この attn 呼び出しで層出力をデバイスに残してよいか (呼び出し側が設定)。
    /// 常駐経路の平坦入口だけが true にする (従来経路は必ずホスト出力を受け取る)。
    pub(crate) static PF_ATTN_DEV_REQ: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}


/// task 200: 層出力をデバイスに残してよいかを設定する (attn 呼び出しの前後で切る)。
pub fn pf_attn_dev_request(on: bool) {
    PF_ATTN_DEV_REQ.with(|c| c.set(on));
}


/// task 200: デバイス出力の要求が出ているか (連鎖が読む)。
pub(crate) fn pf_attn_dev_requested() -> bool {
    PF_ATTN_DEV_REQ.with(|c| c.get())
}


/// task 200: 直前の層出力がデバイスに残っているかを捨てる (層の入口で呼ぶ)。
pub fn pf_attn_dev_clear() {
    PF_ATTN_DEV_OUT.with(|c| *c.borrow_mut() = None);
}


/// task 200: ssm 連鎖が層出力をデバイスに残したことを記録する。
pub(crate) fn pf_attn_dev_store(ptr: CUdeviceptr, nt: usize, e: usize) {
    PF_ATTN_DEV_OUT.with(|c| *c.borrow_mut() = Some((ptr, nt, e)));
}


/// task 308: 層出力がデバイスに残っているか (形が合うときだけ true)。
/// 記録は消さない (直後の [`pf_combine_dev`] が `take` する)。
pub(crate) fn pf_attn_dev_peek(nt: usize, e: usize) -> bool {
    PF_ATTN_DEV_OUT.with(|c| matches!(&*c.borrow(), Some((_, dnt, de)) if *dnt == nt && *de == e))
}


/// task 222: `DENNO_PF_ATTN_TC` の値の解釈 (純関数)。明示的な偽値だけ off。
///
/// **既定 on** (lab t222b で採用): 14,018 の full の gpu が 2,184 → 1,636 ms、
/// 3,826 で 236 → 193 ms、cold prefill 14,018 が -1.9%、decode と next turn は
/// 変わらず、errors 0・text sane。attention は P17g の緩和枠なので md5 は
/// スコアの BF16 丸めで変わる。
pub(crate) fn pf_attn_tc_from_value(v: Option<&str>) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        Some(s) => matches!(s.as_str(), "1" | "on" | "true" | "yes"),
        None => true,
    }
}


/// prefill の attention を v2 カーネル (bf16 mma + 4x16 の PV) で回すか
/// (**既定 on**。`0` で切る)。
pub(crate) fn pf_attn_tc_on() -> bool {
    #[cfg(test)]
    {
        match PF_ATTN_TC_FORCE.load(Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| pf_attn_tc_from_value(std::env::var("DENNO_PF_ATTN_TC").ok().as_deref()))
}


#[cfg(test)]
static PF_ATTN_TC_FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);


#[cfg(test)]
pub(crate) fn set_pf_attn_tc_for_test(v: Option<bool>) {
    PF_ATTN_TC_FORCE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        Ordering::Relaxed,
    );
}


/// v2 カーネルの静的共有メモリ (`kernels_attn_pf2.cu` の A2_SMEM_BYTES と一致)。
pub(crate) const ATTN_PF2_SMEM_BYTES: usize = 92_928;

/// v2 カーネルが前提にする head 次元 (`kernels_attn_pf2.cu` のスレッド配置)。
pub(crate) const ATTN_PF2_DH: usize = 256;

/// v2 カーネルが 1 ブロックで扱うクエリ行数 (ホストの grid x の分母)。
pub(crate) const ATTN_PF2_BQ: usize = 64;


/// task 230: `DENNO_PF_ATTN_V3` の値の解釈 (純関数)。明示的な偽値だけ off。
///
/// **既定 on** (task 244 の lab t244a で採択)。`0` / `off` / `false` / `no` で
/// 従来 (v2 / v1) の経路に戻せる。`1` で v3 カーネル (`kernels_attn_pf3.cu`、
/// PV も bf16 mma・head 優先グリッド・行最大は最初のタイルで 1 回)。
///
/// lab t230b (d9f3d10、depth8k、PROFILE=1、PTX=attn_pf3、14,018 cold prefill) は
/// **不採用**だった: full 段は 2,036.8 → 1,196.1 ms (-41%、gpu 1,638.5 → 850.2) と
/// 速いのに全体は 27.40 → 27.93 s (+1.9%)。ただしその run は中央の 3 COND
/// (attn3 / mixkeep / both) だけホスト側の `other` 残差が 1.4〜1.7 s → 2.95〜3.01 s
/// に伸びており、v3 を使わない mixkeep でも同じだった (共有ホストの負荷揺れ)。
///
/// task 244 の lab t244a (58356ab、同 scenario、PTX=attn_pf3) は 6 COND を
/// t0/base/attn3/base2/attn3b/base3 と交互に置いて測り直した (14,018 cold prefill、
/// 実測 s): t0 26.02 / base 26.45 / attn3 25.59 / base2 25.95 / attn3b 25.15 /
/// base3 25.95。捨て COND を除いた base の平均 26.117 に対して attn3 の平均
/// 25.37 で **-2.86%**。3,826 は base 7.127 → attn3 7.10 (-0.37%)、decode は
/// 20.23 → 20.25 ms/tok (+0.08%)。台帳では full 段 1,989 → 1,200 ms
/// (-789 ms) だけで、他の名前付き段 (mix / router / shared / ssm / experts) は
/// すべて ±0.3% 以内。md5 は c2000_turn2 と c8000 で DIFF (p と V の bf16 丸め、
/// 相対 5e-3 は CPU 模擬 `pf230_attn_pf3_tile_matches_host` で固定)。
///
/// 2 回目 (lab t244b、4a997d4、生成 PTX を commit して `PTX=` 無し・既定 on、
/// COND=t0(on)/off/on/off2/on2/off3、load_before 7.23 の混んだ状態) でも
/// 14,018 cold prefill は off の平均 25.907 s に対し on 25.13 s で **-3.00%**、
/// 3,826 は 7.11 → 7.06 s (-0.70%)、decode は 20.27 → 20.25 ms/tok。
/// 台帳の差は full 段 -790 ms だけで、`other` 残差は全 COND で 1.39 s に揃った
/// (round 1 の揺れはホスト負荷)。md5 は round 1 の attn3 と一致した
/// (commit した PTX が lab の生成物と同じ挙動)。
pub(crate) fn pf_attn_v3_from_value(v: Option<&str>) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        Some(s) => matches!(s.as_str(), "1" | "on" | "true" | "yes"),
        None => true,
    }
}


/// prefill の attention を v3 カーネルで回すか (**既定 on**、task 244 で採択)。
pub(crate) fn pf_attn_v3_on() -> bool {
    #[cfg(test)]
    {
        match PF_ATTN_V3_FORCE.load(Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| pf_attn_v3_from_value(std::env::var("DENNO_PF_ATTN_V3").ok().as_deref()))
}


#[cfg(test)]
static PF_ATTN_V3_FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);


#[cfg(test)]
pub(crate) fn set_pf_attn_v3_for_test(v: Option<bool>) {
    PF_ATTN_V3_FORCE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        Ordering::Relaxed,
    );
}


/// v3 カーネルの静的共有メモリ (`kernels_attn_pf3.cu` の A3_SMEM_BYTES と一致)。
pub(crate) const ATTN_PF3_SMEM_BYTES: usize = 89_600;

/// v3 カーネルが前提にする head 次元 (`kernels_attn_pf3.cu` のスレッド配置)。
pub(crate) const ATTN_PF3_DH: usize = 256;

/// v3 カーネルが 1 ブロックで扱うクエリ行数 (ホストの grid y の分母)。
pub(crate) const ATTN_PF3_BQ: usize = 64;


// ---------------------------------------------------------------------------
// task 465: prefill attention v4 (K/V を bf16 で読む, `DENNO_PF_ATTN_KVBF16`)
// ---------------------------------------------------------------------------
//
// `full` 段の GPU は 14k で 637.7 ms (lab t461d)、うち約 100 ms が qkv / out の
// GEMM なので attention 本体は約 540 ms。v3 の 1 ブロックは 64 クエリ行で K/V の
// タイルを 32 キーずつ読み、1 クエリ行あたりプレフィックス全体を読む: 14k の
// 1 層で 219 ブロック x 24 head x 平均 7,010 キー x 512 次元 x 4 B x 2 本 = 75 GB、
// 12 層で約 900 GB。540 ms は **1.67 TB/s で HBM の帯域 (1.79 TB/s) に張り付いて**
// いる。クエリ行を増やす手は無い (sm_120 の動的 smem 上限 99 KB に対し v3 が
// 89.6 KB、q_sm だけで 33.8 KB)。
//
// 一方、v3 は K/V を f32 で読み、smem へ入れるときに `__floats2bfloat162_rn` で
// bf16 に丸めている = **カーネルが使う値は元から bf16**。そこで bf16 のミラーを
// 1 本持ち、f32 のミラーから新チャンクの行だけを落として v4 に渡す。読みは
// ちょうど半分になり、**値は 1 ビットも変わらない** (md5 SAME が期待値)。
//
// ゲート: `DENNO_PF_ATTN_KVBF16`。**既定 on** (lab t465a/t465b で採択)。
// 切り戻しは `DENNO_PF_ATTN_KVBF16=0` (値は v3 と同じ)。

/// v4 カーネルの静的共有メモリ (`kernels_attn_pf4.cu` の A4_SMEM_BYTES と一致)。
pub(crate) const ATTN_PF4_SMEM_BYTES: usize = 89_600;

/// v4 カーネルが前提にする head 次元 (`kernels_attn_pf4.cu` のスレッド配置)。
pub(crate) const ATTN_PF4_DH: usize = 256;

/// v4 カーネルが 1 ブロックで扱うクエリ行数 (ホストの grid y の分母)。
pub(crate) const ATTN_PF4_BQ: usize = 64;

/// v4 カーネルのスレッド数 (`A4_THREADS`)。
pub(crate) const ATTN_PF4_THREADS: usize = 256;

/// bf16 へ落とすカーネルのスレッド数 (`A4_CONV_THREADS`)。
pub(crate) const ATTN_PF4_CONV_THREADS: usize = 256;


/// `DENNO_PF_ATTN_KVBF16` の解釈 (純関数)。**既定 on** (lab t465a/t465b で採択)。
///
/// 実測 (depth8k、14k cold、2 ラウンド): `full` 段の GPU が 638.3 / 639.5 ms に
/// 対して **552.7 / 552.7 ms (-13.4% / -13.6%)**、同段の wall が 826.7 / 833.8 に
/// 対して 766.4 / 770.3 ms (-7.3% / -7.6%)。3,826 は同段が小さく差なし
/// (round 2 で 108.0 対 107.9 ms)、decode も全 COND 11.3 ms/tok で不変、
/// md5 は全 COND・全コンテキスト SAME、errors 0、text sane。
/// 切り戻しは `DENNO_PF_ATTN_KVBF16=0`。
pub(crate) fn pf_attn_kvbf16_from_value(v: Option<&str>) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        // 未設定は既定 on。明示的な真値だけ on、`0` などで v3 へ戻す。
        Some(s) => matches!(s.as_str(), "1" | "on" | "true" | "yes"),
        None => true,
    }
}


#[cfg(test)]
static PF_ATTN_KVBF16_FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);


/// prefill の attention を bf16 の K/V で回すか (**既定 off**)。
pub(crate) fn pf_attn_kvbf16_on() -> bool {
    #[cfg(test)]
    {
        match PF_ATTN_KVBF16_FORCE.load(Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        pf_attn_kvbf16_from_value(std::env::var("DENNO_PF_ATTN_KVBF16").ok().as_deref())
    })
}


#[cfg(test)]
pub(crate) fn set_pf_attn_kvbf16_for_test(v: Option<bool>) {
    PF_ATTN_KVBF16_FORCE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        Ordering::Relaxed,
    );
}


fn pf_attn_pf4_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_PF_ATTN4_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_attn_pf4.ptx"),
    }
}


/// v4 の 2 つの入口 (attention 本体 + f32 -> bf16 の変換)。
pub(crate) struct PfAttnV4 {
    pub(crate) attn: CUfunction,
    conv: CUfunction,
}


unsafe impl Send for PfAttnV4 {}

unsafe impl Sync for PfAttnV4 {}


/// v4 カーネルのハンドル (ゲート off / PTX 無し / opt-in 失敗は `None`)。
///
/// `None` のとき呼び出し側は v3 へ落ちる (fail closed)。
pub(crate) fn pf_attn_pf4_kernels() -> Option<&'static PfAttnV4> {
    static K: OnceLock<Option<PfAttnV4>> = OnceLock::new();
    K.get_or_init(|| {
        if !pf_attn_kvbf16_on() {
            return None;
        }
        let path = pf_attn_pf4_ptx_path();
        let mut image = match std::fs::read(&path) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[pf-attn4] PTX が無い ({}) : {e} → v3 へ", path.display());
                return None;
            }
        };
        image.push(0);
        let ctx = shared_ctx()?;
        let _ = ctx;
        let l = lib().ok()?;
        unsafe {
            let mut module: CUmodule = std::ptr::null_mut();
            if (l.module_load_data)(&mut module, image.as_ptr() as *const c_void) != CUDA_SUCCESS {
                eprintln!("[pf-attn4] PTX を読めない → v3 へ");
                return None;
            }
            let mut attn: CUfunction = std::ptr::null_mut();
            let mut conv: CUfunction = std::ptr::null_mut();
            if (l.module_get_function)(
                &mut attn,
                module,
                b"dno_attention_prefill4\0".as_ptr() as *const c_char,
            ) != CUDA_SUCCESS
            {
                eprintln!("[pf-attn4] 入口 dno_attention_prefill4 が無い → v3 へ");
                return None;
            }
            if (l.module_get_function)(
                &mut conv,
                module,
                b"dno_kv_f32_to_bf16\0".as_ptr() as *const c_char,
            ) != CUDA_SUCCESS
            {
                eprintln!("[pf-attn4] 入口 dno_kv_f32_to_bf16 が無い → v3 へ");
                return None;
            }
            let ok = match l.func_set_attribute {
                Some(fn_attr) => {
                    (fn_attr)(
                        attn,
                        CU_FUNC_ATTR_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        ATTN_PF4_SMEM_BYTES as c_int,
                    ) == 0
                }
                None => false,
            };
            if !ok {
                eprintln!("[pf-attn4] 動的 smem の opt-in に失敗 → v3 へ");
                return None;
            }
            eprintln!(
                "[pf-attn4] K/V を bf16 で読む prefill attention を使う (smem {} B)",
                ATTN_PF4_SMEM_BYTES
            );
            Some(PfAttnV4 { attn, conv })
        }
    })
    .as_ref()
}


/// v4 用の bf16 K/V ミラー (1 本を層ごとに使い回す)。
pub(crate) struct PfAttnV4Kv {
    k: Option<DevBuffer>,
    v: Option<DevBuffer>,
    /// 次に変換する行 (0 から層の中で積み上がる。`pos0 == 0` でリセット)。
    next_row: usize,
    kv_dim: usize,
}


thread_local! {
    pub(crate) static PF_ATTN_V4_KV: std::cell::RefCell<PfAttnV4Kv> = const {
        std::cell::RefCell::new(PfAttnV4Kv { k: None, v: None, next_row: 0, kv_dim: 0 })
    };
}


/// f32 のミラーの行 `[pos0, pos0 + nt)` を bf16 へ落とし、bf16 のポインタを返す。
///
/// 層優先 prefill は 1 層のチャンクを続けて回すので、`pos0` が前回の続き
/// (`next_row`) と一致する限り差分だけ落とせばよい。`pos0 == 0` は層の先頭。
/// 食い違ったら `None` を返して v3 へ落とす (誤った行を読ませない)。
pub(crate) fn pf_attn_v4_kv_prepare(
    kptr: CUdeviceptr,
    vptr: CUdeviceptr,
    pos0: usize,
    nt: usize,
    t_total: usize,
    kv_dim: usize,
) -> Option<(CUdeviceptr, CUdeviceptr)> {
    let ctx = shared_ctx()?;
    let fns = pf_attn_pf4_kernels()?;
    PF_ATTN_V4_KV.with(|cell| {
        let mut g = cell.borrow_mut();
        if g.kv_dim != kv_dim {
            g.k = None;
            g.v = None;
            g.kv_dim = kv_dim;
            g.next_row = 0;
        }
        let need = t_total
            .checked_add(t_total / 4 + 8192)?
            .checked_mul(kv_dim)?
            .checked_mul(2)?;
        if need == 0 {
            return None;
        }
        if g.k.as_ref().map(|b| b.bytes()).unwrap_or(0) < need {
            let k = DevBuffer::alloc_tagged(need, crate::mem_audit::TAG_GEMV).ok()?;
            let v = DevBuffer::alloc_tagged(need, crate::mem_audit::TAG_GEMV).ok()?;
            g.k = Some(k);
            g.v = Some(v);
            g.next_row = 0;
        }
        if nt == 0 {
            return None;
        }
        // 行 [from, pos0 + nt) を落とす。層の先頭 (pos0 == 0) と、前回の続き
        // (next_row == pos0) は差分だけ。想定外の順 (逐次 prefill・確保し直し) は
        // 0 からやり直す (読み手は常に [0, pos0+nt) しか見ない)。
        let from = if pos0 == 0 || g.next_row > pos0 {
            0
        } else if g.next_row == pos0 {
            pos0
        } else {
            // next_row < pos0: 途中の行が落ちていないので 0 から。
            0
        };
        let rows = pos0 + nt - from;
        let kb = g.k.as_ref()?.ptr();
        let vb = g.v.as_ref()?.ptr();
        // 行 [from, pos0+nt) を落とし、行 `from` から `rows` 行ぶん変換する。
        let elems4 = (rows * kv_dim / 4).max(1);
        let blocks = elems4
            .div_ceil(ATTN_PF4_CONV_THREADS)
            .min(65535) as u32;
        let (mut a, mut b) = (kptr, vptr);
        let mut c = kb;
        let mut d = vb;
        let (mut r0, mut n_c, mut kvd) =
            (from as c_int, rows as c_int, kv_dim as c_int);
        let mut args: Vec<*mut c_void> = vec![
            &mut a as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
            &mut c as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
            &mut r0 as *mut _ as *mut c_void,
            &mut n_c as *mut _ as *mut c_void,
            &mut kvd as *mut _ as *mut c_void,
        ];
        unsafe {
            ctx.launch_async(
                fns.conv,
                (blocks, 1, 1),
                (ATTN_PF4_CONV_THREADS as u32, 1, 1),
                0,
                &mut args,
            )
            .ok()?;
        }
        g.next_row = pos0 + nt;
        Some((kb, vb))
    })
}


/// 非同期の k/v 書き戻しの置き場 (スレッド局所・grow-only)。
///
/// task 553: 待ちを後ろへずらすときは 2 枚を交互に使う (1 枚だと次の
/// `begin` がまだ読んでいない転送を上書きしてしまう)。`seq` は古い順に
/// 流すための通し番号。
pub(crate) struct KvWbSlot {
    pub(crate) pin: Option<PinMem>,
    pub(crate) ev: Option<CUevent>,
    /// 保留中のバイト数 (k, v)。(0, 0) で保留なし。
    pub(crate) bytes: (usize, usize),
    /// 保留の持ち主 (層, 行数)。
    pub(crate) owner: (usize, usize),
    /// 積んだ順 (0 = 空)。
    pub(crate) seq: u64,
}


impl KvWbSlot {
    pub(crate) const fn new() -> Self {
        Self {
            pin: None,
            ev: None,
            bytes: (0, 0),
            owner: (0, 0),
            seq: 0,
        }
    }

    pub(crate) fn pending(&self) -> bool {
        self.bytes.0 > 0
    }
}


pub(crate) struct KvWbSc {
    pub(crate) slots: [KvWbSlot; 2],
    /// 次に `begin` が書くスロット (`defer` のときだけ交互に進める)。
    pub(crate) next: usize,
    /// `begin` の通し番号。
    pub(crate) seq: u64,
}


thread_local! {
    pub(crate) static KV_WB_SC: std::cell::RefCell<KvWbSc> = const {
        std::cell::RefCell::new(KvWbSc {
            slots: [KvWbSlot::new(), KvWbSlot::new()],
            next: 0,
            seq: 0,
        })
    };
}


/// 次に `begin` が使うスロットを選ぶ (純関数。テストで固定する)。
///
/// `defer` のときは空いている枚を選ぶ (2 枚交互)。そうでなければ 1 枚しか
/// 使わない (従来と同じく、保留があるうちは積まない)。
pub(crate) fn kv_wb_pick_slot(pending: [bool; 2], next: usize, defer: bool) -> Option<usize> {
    if !defer {
        return if pending[0] || pending[1] { None } else { Some(0) };
    }
    let a = next % 2;
    if !pending[a] {
        return Some(a);
    }
    let b = (next + 1) % 2;
    if !pending[b] {
        return Some(b);
    }
    None
}


/// 一番古い保留スロット。
pub(crate) fn kv_wb_oldest(s: &KvWbSc) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (i, sl) in s.slots.iter().enumerate() {
        if !sl.pending() {
            continue;
        }
        match best {
            None => best = Some(i),
            Some(j) if sl.seq < s.slots[j].seq => best = Some(i),
            _ => {}
        }
    }
    best
}


// P17j: prefill attention カーネルのタイル定数 (kernels_elem.cu の PF_BQ/PF_BK
// と同期すること)。BQ=行タイル・BK=key タイル。
// P17k ステップ 7 修正: ステップ 4 でカーネルを 48/8→(実験 64/8) に変えた際に
// ここが 16/32 のまま残り、①grid が約 4 倍過剰 (i0 >= n の無効ブロックがフルの
// time ループを回す) ②sh_f も 16/32 基準で、カーネルの必要量を下回ると
// smem 領域外書きになり error 700 (不正アドレス) になっていた —
// 「~43KB 以上で error 700」の正体はこの不一致。64/8 (43.8KB) に統一。
pub(crate) const ATTN_PF_BQ: usize = 64;

pub(crate) const ATTN_PF_BK: usize = 8;


// s48 (C1-a): dno_attention_prefill の動的 smem。カーネル側は extern __shared__
// の静的宣言が無く launch 時の sharedMemBytes をそのまま使うため、48KB 既定上限は
// ホスト側ゲートだけの制約だった。qwen35 (dh=256) は 1 ブロック 84.8KB を要求し
// 48KB を超えるため、モジュールロード時に cuFuncSetAttribute
// (CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES) で opt-in して引き上げる。
// 実機で opt-in 不可 (デバイスの opt-in 上限 < 必要量・古い driver でシンボル無し等)
// の場合も致命的ではない: 上限は 48KB のまま = qwen35 prefill attention は
// attn_prefill_gpu_supported ゲートで静かに CPU へフォールバックし、llama (dh=128
// = 43.8KB) は従来どおり発動する。.cu / PTX は無変更 (C1-b のタイル縮小が必要に
// なった場合のみ .cu 変更 + PTX 再生成が発生する)。
pub(crate) const ATTN_PF_SMEM_DEFAULT: usize = 48 * 1024;

pub(crate) static ATTN_PF_DYN_SMEM: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(ATTN_PF_SMEM_DEFAULT);


/// dno_attention_prefill が許容する動的 smem の上限 (bytes)。既定 48KB・
/// CudaCtx 生成時の opt-in (cuFuncSetAttribute) 成功時はデバイスの opt-in 上限。
pub fn attn_pf_dynamic_smem_limit() -> usize {
    ATTN_PF_DYN_SMEM.load(std::sync::atomic::Ordering::Relaxed)
}


/// dno_attention_prefill の動的 smem 必要量 (float 数)。
/// q_sm [BQ][dh+1] + k_sm [BK][dh+1] + v_sm [BK][dh+1] + s_sm [BQ*BK]
/// + corr_sm [BQ] + mnew_sm [BQ] (kernels_elem.cu と同期すること)。
pub(crate) fn attn_pf_sh_f(dh: usize) -> usize {
    (ATTN_PF_BQ + 2 * ATTN_PF_BK) * (dh + 1) + ATTN_PF_BQ * ATTN_PF_BK + 2 * ATTN_PF_BQ
}


pub(crate) struct ChainScratch {
    pub(crate) bufs: Vec<Option<DevBuffer>>,
    pub(crate) pin_x: Option<PinMem>,  // x / att アップロード用 (使用間に必ず sync が挟まる)
    pub(crate) pin_dn: Option<PinMem>, // ダウンロード用
    pub(crate) aq: Option<DevBuffer>,  // int8 カーネル用スクラッチ (chain 専用)
    pub(crate) attn_q: Option<DevBuffer>,  // P17g: attention q アップロード用
    pub(crate) attn_sc: Option<DevBuffer>, // P17g: attention scores スクラッチ [heads*t_total]
    pub(crate) attn_pf_q: Option<DevBuffer>, // P17j: prefill attention q アップロード用 [n*heads*dh]
    pub(crate) rope_tab: Option<DevBuffer>,  // P17h: rope cos/sin テーブル [cos(half)|sin(half)]
    pub(crate) rope_bias: Option<DevBuffer>, // P17h: qkv bias パック [qb|kb|vb]
    pub(crate) pin_rope: Option<PinMem>,     // P17h: rope テーブル/... ステップ 1 回のアップロード用
    pub(crate) pin_pf: Option<PinMem>,       // P17k: prefill attention q の pinned ステージング用
    pub(crate) kv_stage: Option<PinMem>,     // P17h: k/v の非同期 D2H ステージング [n_layers][k|v]
    // P17k ステップ 5: qkv→rope→attention デバイス連続化用 (prefill)
    pub(crate) qkv_x: Option<DevBuffer>,     // GEMM 入力 h [n, e] fp32
    pub(crate) qkv_aq: Option<DevBuffer>,    // GEMM 入力量子化 [n, e] int8
    pub(crate) qkv_asc: Option<DevBuffer>,   // GEMM 入力スケール [n] f32
    pub(crate) qkv_pf: Option<DevBuffer>,    // GEMM 出力 [n, qn+kn+vn] (q/k/v 連結・行-major)
    pub(crate) qkv_q: Option<DevBuffer>,     // rope 後 q [n, heads*dh] (attention 入力)
    pub(crate) rope_tab_pf: Option<DevBuffer>, // rope テーブル [n, dh] (行 i = pos0+i の cos|sin)
    pub(crate) pin_qkv: Option<PinMem>,      // h / rope テーブル / bias アップロード用 pinned
    // B3: b>1 バッチ decode の rope (batch_rope_qk)
    pub(crate) br_q: Option<DevBuffer>,      // [b, heads*dh] f32 (in-place rope)
    pub(crate) br_k: Option<DevBuffer>,      // [b, kv_dim] f32 (in-place rope)
    pub(crate) br_tab: Option<DevBuffer>,    // [b, dh] f32 テーブル (行 i = poss[i] の [cos|sin])
    pub(crate) br_pin_qk: Option<PinMem>,    // q/k の H2D/D2H ステージング (sync を挟むので再利用安全)
    pub(crate) br_pin_tab: Option<PinMem>,   // テーブル H2D ステージング
    pub(crate) br_tab_key: Vec<usize>,       // テーブルキャッシュキー (poss・完全一致)
    pub(crate) br_tab_dh: usize,             // テーブルキャッシュキー (dh)
    pub(crate) br_tab_theta: f32,            // テーブルキャッシュキー (theta・bits 比較)
    pub(crate) br_tab_valid: bool,           // dev 上のテーブルがキャッシュキーと一致するか
    // P17h: ステップ内で sync が無くなるため pinned は (層×用途) ごとに専用化する
    // (async H2D の DMA 完了前にホストが上書きする競合を避ける)。id は呼び出し側が
    // 層番号と用途から一意に採番する。
    pub(crate) pins: Vec<Option<PinMem>>,
    // C4: qwen35 chain
    // rope テーブル [cos(n_rot/2) | sin(n_rot/2)] + 転送元 pinned
    pub(crate) q35_tab: Option<DevBuffer>,
    pub(crate) q35_pin_tab: Option<PinMem>,
    /// fp32 重みのデバイスキャッシュ (beta/alpha 制御投影・QK-Norm q/k norm)。
    /// Linear::F32 / Vec<f32> には uid が無いため (model_id, layer, tag) でキー化
    /// する。model_id はロード毎に採番されるため、モデル再ロードで誤キャッシュに
    /// なることはない。
    pub(crate) q35_w_cache: std::collections::HashMap<(u64, usize, u8), DevBuffer>,

    /// s165: rmsnorm 重みのデバイス常駐キャッシュ (`chain_rmsnorm_cached`)。
    /// キーは (model_id, pin_id = 層×用途)。decode chain が 1 ステップに 129 回
    /// 同じ重みを上げ直していた分を初回 1 回にする。
    pub(crate) rms_w: std::collections::HashMap<(u64, usize), DevBuffer>,
    // s167 (P6): GDN prefill の h2 D2H を qkv/z GEMM と重ねるための一式。
    // `DENNO_PF_OVERLAP=1` のときだけ作られる (フラグ OFF では None のまま =
    // 従来の同期 D2H と同一経路・未使用ストリームを残さない)。
    pub(crate) h2_stream: Option<CopyStream>,  // 非ブロッキングの専用コピーストリーム
    pub(crate) h2_ev_ready: Option<CopyEvent>, // main stream の rmsnorm 完了 (発行前の順序付け)
    pub(crate) h2_ev_done: Option<CopyEvent>,  // コピーストリームの D2H 完了 (join)
    pub(crate) pin_h2: Option<PinMem>,         // D2H の宛先 (pinned・grow-only)
    pub(crate) h2_pending: bool,               // 発行済みで未 join の D2H があるか
    // s246 (P46): full 層 att の D2H/H2D と sigmoid ゲートを重ねるための
    // チャンク完了イベント (既定ストリーム・grow-only・層をまたいで再利用)。
    pub(crate) pipe_ev: Vec<CopyEvent>,
    // s247 (P47): full_qk (q/k 射影) の D2H 宛先 = pinned staging (grow-only)。
    // 1 本のプールを q/k の 2 スライスに割り、層をまたいで使い回す (確保は
    // リクエスト最初の full 層で 1 回だけ)。寿命の契約は `PfQkView` を参照。
    pub(crate) pin_pf_qkv: Option<PinMem>,
    // s250 (P53): full_qk の q の D2H を k/v の GEMM と重ねるための一式。
    // h2 の一式 (s167/P6) とは別に持つ — h2 の join 契約 (`h2_pending`) と
    // 混ぜると、同じストリーム上で h2 のコピーと q のコピーが直列化する。
    pub(crate) qk_stream: Option<CopyStream>,  // 非ブロッキングの専用コピーストリーム
    pub(crate) qk_ev_q: Option<CopyEvent>,     // q の GEMM 完了 (発行前の順序付け)
}


pub(crate) fn chain_err(e: anyhow::Error) -> anyhow::Error {
    tracing::error!("cuda chain failed (falling back to CPU): {e:#}");
    eprintln!("cuda chain failed (falling back to CPU): {e:#}");
    GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
    e
}


fn chain_fail(e: anyhow::Error) -> Option<()> {
    chain_err(e);
    None
}


/// s48 (C1): qwen35 prefill の full attention を GPU バッチ (dno_attention_prefill
/// 1 発) にしてよいかの事前ゲート。構造的に確実に失敗する形状を開始前に弾き、
/// GPU_BROKEN ラッチの無駄な発火を防ぐ (llama B1 の prefill_chain_gpu_supported
/// と同一の趣旨)。qwen35 forward_prefill の Full 分岐から呼ばれる。
/// - 発動ゲート: `n >= gemm_min_m()` (既定 256) — 短文 (1b 含む m≤255) は CPU
///   attend のまま bit 不変 (アンカー温存・`maybe_gpu_linear` の GEMM 発動帯域と
///   一致させる設計)
/// - `DENNO_GPU_ATTN=0` で全経路現行に戻る (escape hatch 流用・新 env は作らない)
/// - 動的 smem: dh=256 は 84.8KB で 48KB 既定上限を超える → CudaCtx 生成時の
///   cuFuncSetAttribute opt-in が成功している場合だけ true (失敗なら false =
///   CPU attend フォールバック・llama は影響を受けない)
pub fn attn_prefill_gpu_supported(n: usize, dh: usize, t_total: usize) -> bool {
    attn_prefill_gpu_supported_min(n, dh, t_total, gemm_min_m())
}


/// task 135: 行数の下限だけを呼び出し側が決められる版。
///
/// qwen4exp の prefill attention は attention 専用の下限
/// [`ATTN_Q4_PF_MIN_ROWS`] を使う (`gemm_min_m()` は GEMM の帯域しきい値で、
/// attention のカーネル制約ではない)。qwen35 は従来どおり
/// [`attn_prefill_gpu_supported`] (= この関数を `gemm_min_m()` で呼ぶ) を通るので
/// 挙動は変わらない。
pub fn attn_prefill_gpu_supported_min(
    n: usize,
    dh: usize,
    t_total: usize,
    min_rows: usize,
) -> bool {
    if !cuda_enabled() || !gpu_attention_enabled() {
        return false;
    }
    if !attn_prefill_shape_ok(n, dh, t_total, min_rows) {
        return false;
    }
    // opt-in の結果 (ATTN_PF_DYN_SMEM) は CudaCtx 生成時に確定する。ここまで到達
    // したということは cuda 有効なので、未初期化ならここで生成する (直後の
    // chain_attention_prefill 内部でも生成されるため前倒しで生成するだけ = 新規
    // の副作用は無い)。生成失敗なら CPU フォールバック (GPU_BROKEN にはしない)。
    if shared_ctx().is_none() {
        return false;
    }
    true
}


/// [`attn_prefill_gpu_supported_min`] の純関数部分 (env もデバイスも読まない)。
///
/// `min_rows` だけが差し替え可能で、それ以外 (n > 0・t_total >= n・dh の範囲・
/// 動的 smem の上限) は従来のゲートと同一。
pub(crate) fn attn_prefill_shape_ok(
    n: usize,
    dh: usize,
    t_total: usize,
    min_rows: usize,
) -> bool {
    if n == 0 || t_total == 0 || n < min_rows || t_total < n {
        return false;
    }
    if dh < ATTN_PF_BQ || dh > 1024 {
        return false;
    }
    if attn_pf_sh_f(dh) * 4 > attn_pf_dynamic_smem_limit() {
        return false;
    }
    true
}


pub(crate) fn chain_ctx() -> Option<&'static CudaCtx> {
    if !cuda_enabled() {
        return None;
    }
    let ctx = shared_ctx()?;
    if let Err(e) = ctx.make_current() {
        chain_err(anyhow!("cuCtxSetCurrent: {e:#}"));
        return None;
    }
    Some(ctx)
}


pub(crate) fn chain_ensure_buf(slot: usize, bytes: usize) -> Result<(), anyhow::Error> {
    CHAIN.with(|cell| {
        let mut s = cell.borrow_mut();
        // P58-3: prefill の一時バッファは arena から切り出す。同じキー・同じ
        // サイズ以下なら同じオフセットが返るので、リクエストごとの
        // cuMemAlloc / cuMemFree が消える。arena 無効・失敗時は従来の個別確保
        // (fail-open・数値不変)。
        Scratch::ensure_arena(
            &mut s.bufs[slot],
            ARENA_KEY_CHAIN + slot as u32,
            bytes,
            crate::mem_audit::TAG_CHAIN,
        )
            .map_err(chain_err)
    })
}


/// ホストデータをスロットへアップロード (async、同期はしない)。
/// 呼び出し間に sync が挟まる (qkv download) ので pinned バッファの再利用は安全。
pub fn chain_upload(slot: usize, data: &[f32], cols: usize, m: usize) -> Option<()> {
    let _ = chain_ctx()?;
    let bytes = m.checked_mul(cols)?.checked_mul(4)?;
    if data.len() != m * cols {
        return chain_fail(anyhow!("chain_upload: size mismatch"));
    }
    ok(chain_ensure_buf(slot, bytes))?;
    CHAIN.with(|cell| -> Option<()> {
        let mut s = cell.borrow_mut();
        ok(Scratch::ensure_pin(&mut s.pin_x, bytes))?;
        let buf = s.bufs[slot].as_ref().unwrap().ptr();
        let pin = s.pin_x.as_mut().unwrap();
        pin.as_mut_slice::<f32>()[..m * cols].copy_from_slice(data);
        let lib = ok(lib())?;
        unsafe {
            ok(check(
                (lib.memcpy_htod_async)(buf, pin.ptr as *const c_void, bytes, std::ptr::null_mut()),
                "cuMemcpyHtoDAsync (chain)",
            ))
        }?;
        // s166 (P5): 観測のみ
        dbg_htod(bytes);
        Some(())
    })
}


/// s55 (T1 step2・計測専用): デフォルトストリームの同期を 1 回打つだけの関数。
/// `chain_download` が内部で行う `ctx.sync()` と**同一の API 呼び出し**
/// (`cuCtxSynchronize`) を、本番 D2H を発行する前に別計測点として観測するための
/// 入口。同期以外の副作用は無く (バッファ確保・転送の発行をしない)、呼んでも
/// 数値経路には触れない。呼び出し側 (qwen35 decode) は `prof::enabled()` が真の
/// ときだけこれを呼ぶ — 本番経路 (非 prof) からは決して到達しない。
pub fn chain_sync_pending() -> Option<()> {
    let ctx = chain_ctx()?;
    ok(ctx.sync())
}


/// 最終 logits の戻しは decode 1 ステップ 1 回の最大 D2H。発行元の件数分類は
/// `prof::chain_download_site` (純関数) がスロットから決める (従来はすべて
/// `D2H_SRC_OTHER` に落ちていた — s108)。
/// s88 (Step 1): 発行点ごとの時間 (表示専用)。計測 OFF では時刻を読まない。
/// ここで件数も一緒に積むので、内側では `mark_d2h_src` を呼ばない
/// (呼ぶと calls が 2 倍になる)。
pub fn chain_download(slot: usize, len: usize) -> Option<Vec<f32>> {
    let site = crate::prof::chain_download_site(slot);
    let t_site = if crate::prof::d2h_src_on() {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let r = chain_download_inner(slot, len);
    match t_site {
        Some(t) => crate::prof::mark_d2h_src_ns(site, t.elapsed().as_nanos() as u64),
        // 計測 OFF: 従来どおり件数だけ (実際は `mark_d2h_src` 側も no-op)。
        None => crate::prof::mark_d2h_src(site),
    }
    r
}


fn chain_download_inner(slot: usize, len: usize) -> Option<Vec<f32>> {
    let ctx = chain_ctx()?;
    let bytes = len.checked_mul(4)?;
    ok(chain_ensure_buf(slot, bytes))?;
    CHAIN.with(|cell| -> Option<Vec<f32>> {
        let buf = cell.borrow().bufs[slot].as_ref()?.ptr();
        // s69: pinned `pin_dn` に落としてから `to_vec` すると同じ領域を 2 度
        // 書くことになる (93MB で約 20ms)。出力 Vec の未初期化領域へ直接
        // D2H し、全要素をコピーが埋めてから set_len する。
        let mut out: Vec<f32> = Vec::with_capacity(len);
        {
            let lib = ok(lib())?;
            unsafe {
                ok(check(
                    (lib.memcpy_dtoh_async)(
                        out.as_mut_ptr() as *mut c_void,
                        buf,
                        bytes,
                        std::ptr::null_mut(),
                    ),
                    "cuMemcpyDtoHAsync (chain)",
                ))?;
            }
        }
        ok(ctx.sync())?;
        unsafe { out.set_len(len) }
        // s166 (P5): 観測のみ
        dbg_d2h(bytes);
        Some(out)
    })
}


/// スロット内の要素オフセット `elem_off` から `len` 個の f32 をダウンロード
/// (D2H + 同期)。B1: prefill チェーンの最終 norm 出力 (CH_H の最終行のみ)
/// 取得用 — [n, e] 全行の D2H を避ける。呼び出し時点でスロットには
/// (elem_off + len) 個分のデータが存在することを前提とする (不足時は None)。
/// 発行元の件数分類は `prof::chain_download_site` (logits 以外は `D2H_SRC_LINEAR`
/// — 活性化の一部を戻す点で linear 系の部分コピーと同類)。
pub fn chain_download_at(slot: usize, elem_off: usize, len: usize) -> Option<Vec<f32>> {
    // s88 (Step 1): 発行点ごとの時間 (表示専用・計測 OFF はクロック読みなし)。
    let t_site = if crate::prof::d2h_src_on() {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let r = chain_download_at_inner(slot, elem_off, len);
    let site = crate::prof::chain_download_site(slot);
    match t_site {
        Some(t) => crate::prof::mark_d2h_src_ns(site, t.elapsed().as_nanos() as u64),
        None => crate::prof::mark_d2h_src(site),
    }
    r
}


fn chain_download_at_inner(slot: usize, elem_off: usize, len: usize) -> Option<Vec<f32>> {
    let ctx = chain_ctx()?;
    let bytes = len.checked_mul(4)?;
    let need = elem_off.checked_mul(4)?.checked_add(bytes)?;
    if !CHAIN.with(|cell| {
        cell.borrow()
            .bufs
            .get(slot)
            .map_or(false, |b| b.as_ref().map_or(false, |b| b.bytes() >= need))
    }) {
        return None;
    }
    CHAIN.with(|cell| -> Option<Vec<f32>> {
        let buf = cell.borrow().bufs[slot].as_ref()?.ptr() + (elem_off * 4) as CUdeviceptr;
        // s69: chain_download と同じ理由で出力 Vec へ直接 D2H する。
        let mut out: Vec<f32> = Vec::with_capacity(len);
        {
            let lib = ok(lib())?;
            unsafe {
                ok(check(
                    (lib.memcpy_dtoh_async)(
                        out.as_mut_ptr() as *mut c_void,
                        buf,
                        bytes,
                        std::ptr::null_mut(),
                    ),
                    "cuMemcpyDtoHAsync (chain download at)",
                ))?;
            }
        }
        ok(ctx.sync())?;
        unsafe { out.set_len(len) }
        // s166 (P5): 観測のみ
        dbg_d2h(bytes);
        Some(out)
    })
}


/// out = rmsnorm(x [+ residual]) — residual スロットを指定すると x += p を先に
/// 行う (o_proj / down 出力の残差加算と融合)。CPU tensor::rmsnorm とビット一致。
/// grid = m ブロック、平方和は thread 0 が CPU と同一の逐次順で計算。
/// `pin_id` は w アップロード用 pinned の専用スロット (P17h: ステップ内に sync が
/// 無いため層×用途で一意に採番する — async H2D の DMA 完了前の上書き競合回避)。
pub fn chain_rmsnorm(
    w: &[f32],
    eps: f32,
    add_slot: Option<usize>,
    out_slot: usize,
    cols: usize,
    m: usize,
    pin_id: usize,
) -> Option<()> {
    chain_rmsnorm_slot(CH_X, w, eps, add_slot, out_slot, cols, m, pin_id)
}


/// s165 (P2): 入力スロットを指定できる rmsnorm。カーネル (`dno_add_rmsnorm`)・
/// grid・block・動的 shmem・引数の並びは [`chain_rmsnorm`] と 1 つも変わらず、
/// 違うのは x のポインタだけ — したがって出力は `chain_rmsnorm` とビット一致する
/// (prefill の残差常駐経路が `CH_PF_RES` を読むために使う)。
pub fn chain_rmsnorm_slot(
    x_slot: usize,
    w: &[f32],
    eps: f32,
    add_slot: Option<usize>,
    out_slot: usize,
    cols: usize,
    m: usize,
    pin_id: usize,
) -> Option<()> {
    let ctx = chain_ctx()?;
    if w.len() != cols {
        return chain_fail(anyhow!("chain_rmsnorm: w size mismatch"));
    }
    let wb = cols * 4;
    ok(chain_ensure_buf(out_slot, m * cols * 4 + wb))?; // 末尾 wb バイトは w の作業領域
    CHAIN.with(|cell| -> Option<()> {
        let mut s = cell.borrow_mut();
        // w は呼び出しごとの専用 pinned から非同期アップロード (6KB 程度)。
        if s.pins.len() <= pin_id {
            while s.pins.len() <= pin_id {
                s.pins.push(None);
            }
        }
        let wsrc = {
            ok(Scratch::ensure_pin(&mut s.pins[pin_id], wb))?;
            let pin = s.pins[pin_id].as_mut().unwrap();
            pin.as_mut_slice::<f32>()[..cols].copy_from_slice(w);
            pin.ptr as *const c_void
        };
        let out_buf = s.bufs[out_slot].as_ref().unwrap();
        let wdev = out_buf.ptr() + (m * cols * 4) as u64; // バッファ末尾に置く
        let xptr = s.bufs[x_slot].as_ref()?.ptr();
        let pptr = match add_slot {
            Some(sl) => s.bufs[sl].as_ref().unwrap().ptr(),
            None => 0, // NULL → 単純 rmsnorm
        };
        let optr = out_buf.ptr();
        let lib = ok(lib())?;
        unsafe {
            ok(check(
                (lib.memcpy_htod_async)(wdev, wsrc, wb, std::ptr::null_mut()),
                "cuMemcpyHtoDAsync (chain rmsnorm w)",
            ))?;
        }
        // s166 (P5): 観測のみ (この経路は decode chain では使わない — 0 回が期待値)
        dbg_htod(wb);
        let (mut px, mut pp, mut pw, mut po) = (xptr, pptr, wdev, optr);
        let (mut mc, mut ec) = (cols as c_int, eps);
        let mut args = [
            &mut px as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut pw as *mut _ as *mut c_void,
            &mut po as *mut _ as *mut c_void,
            &mut mc as *mut _ as *mut c_void,
            &mut ec as *mut _ as *mut c_void,
        ];
        ok(unsafe {
            ctx.launch_async(ctx.f_addnorm, (m as u32, 1, 1), (256, 1, 1), cols as u32 * 4, &mut args)
        })
    })
}


/// s165: [`chain_rmsnorm`] の**重み常駐**版 (decode chain 専用)。
///
/// `chain_rmsnorm` は呼び出しごとに `w` (e×4 = 20,480 B) を pinned から
/// 非同期 H2D する。decode chain は 1 ステップに 2×64+1 = 129 回呼ぶので、
/// 同じ重みを毎ステップ 129 回 (2.6 MB) 上げ直していた — nsys の
/// 「20,480 B × 25,872」がこれで、重みが変化しない以上まったくの無駄。
///
/// ここでは `(key, pin_id)` でデバイス常駐させ、**初回だけ**転送する
/// (`q35_w_dev` の (model_id, layer, tag) と同じ方針)。`key` はモデル
/// ロード毎に一意な `mid`、`pin_id` は呼び出し側が層×用途で一意に採番する
/// (P17h 規約) ので、ヒット = 同じ重み。サイズが変わったら作り直す。
///
/// 数値は `chain_rmsnorm` と同一: 同じカーネル (`dno_add_rmsnorm`)、同じ引数、
/// 同じ重みバイト列で、変わるのは重みの置き場所だけ。重みの転送は
/// `DevBuffer::from_slice` (同期 H2D) で行うが、これは (key, pin_id) ごとに
/// 1 回だけで、以降のステップは 0 回。
pub fn chain_rmsnorm_cached(
    w: &[f32],
    eps: f32,
    add_slot: Option<usize>,
    out_slot: usize,
    cols: usize,
    m: usize,
    pin_id: usize,
    key: u64,
) -> Option<()> {
    let ctx = chain_ctx()?;
    if w.len() != cols {
        return chain_fail(anyhow!("chain_rmsnorm_cached: w size mismatch"));
    }
    let wb = cols * 4;
    ok(chain_ensure_buf(out_slot, m.checked_mul(cols)?.checked_mul(4)?))?;
    CHAIN.with(|cell| -> Option<()> {
        let mut s = cell.borrow_mut();
        let stale = match s.rms_w.get(&(key, pin_id)) {
            Some(b) => b.bytes() != wb,
            None => true,
        };
        if stale {
            let buf = ok(DevBuffer::from_slice_tagged(w, crate::mem_audit::TAG_CHAIN))?;
            s.rms_w.insert((key, pin_id), buf);
        }
        let wdev = s.rms_w.get(&(key, pin_id))?.ptr();
        let xptr = s.bufs[CH_X].as_ref()?.ptr();
        let pptr = match add_slot {
            Some(sl) => s.bufs.get(sl)?.as_ref()?.ptr(),
            None => 0, // NULL → 単純 rmsnorm
        };
        let optr = s.bufs.get(out_slot)?.as_ref()?.ptr();
        let (mut px, mut pp, mut pw, mut po) = (xptr, pptr, wdev, optr);
        let (mut mc, mut ec) = (cols as c_int, eps);
        let mut args = [
            &mut px as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut pw as *mut _ as *mut c_void,
            &mut po as *mut _ as *mut c_void,
            &mut mc as *mut _ as *mut c_void,
            &mut ec as *mut _ as *mut c_void,
        ];
        ok(unsafe {
            ctx.launch_async(ctx.f_addnorm, (m as u32, 1, 1), (256, 1, 1), cols as u32 * 4, &mut args)
        })
    })
}


/// swiglu: ga スロットの [gate|up] から act = (g/(1+exp(-g)))*u を計算
/// (デバイス内完結)。CPU silu_inplace + gate[i] *= up[i] とビット一致。
pub fn chain_swiglu(ga_slot: usize, act_slot: usize, ff: usize, m: usize) -> Option<()> {
    let ctx = chain_ctx()?;
    ok(chain_ensure_buf(act_slot, m * ff * 4))?;
    CHAIN.with(|cell| -> Option<()> {
        let s = cell.borrow();
        let gaptr = s.bufs[ga_slot].as_ref().unwrap().ptr();
        let optr = s.bufs[act_slot].as_ref().unwrap().ptr();
        let (mut pg, mut pu, mut po) = (gaptr, gaptr + (ff as u64) * 4, optr);
        let mut tc = (m * ff) as c_int;
        let mut args = [
            &mut pg as *mut _ as *mut c_void,
            &mut pu as *mut _ as *mut c_void,
            &mut po as *mut _ as *mut c_void,
            &mut tc as *mut _ as *mut c_void,
        ];
        let grid = ((m * ff) as u32).div_ceil(256);
        ok(unsafe { ctx.launch_async(ctx.f_swiglu, (grid, 1, 1), (256, 1, 1), 0, &mut args) })
    })
}


/// s70: 行内 [gate|up] 連結レイアウト (width = 2*ff) の swiglu。m > 1 でも
/// 使えるのは `dno_swiglu_int` 側 (`dno_swiglu` は gate/up がブロック分離の
/// 前提なので m=1 専用)。演算列は CPU の `gv / (1+exp(-gv)) * uv` と同一。
pub fn chain_swiglu_rows(ga_slot: usize, act_slot: usize, ff: usize, m: usize) -> Option<()> {
    let ctx = chain_ctx()?;
    if ff == 0 || m == 0 {
        return None;
    }
    ok(chain_ensure_buf(act_slot, m.checked_mul(ff)?.checked_mul(4)?))?;
    CHAIN.with(|cell| -> Option<()> {
        let s = cell.borrow();
        let gaptr = s.bufs[ga_slot].as_ref()?.ptr();
        let optr = s.bufs[act_slot].as_ref()?.ptr();
        let (mut pg, mut po) = (gaptr, optr);
        let mut ffc = ff as c_int;
        let mut width = (2 * ff) as c_int;
        let mut tc = (m * ff) as i64;
        let mut args = [
            &mut pg as *mut _ as *mut c_void,
            &mut po as *mut _ as *mut c_void,
            &mut ffc as *mut _ as *mut c_void,
            &mut width as *mut _ as *mut c_void,
            &mut tc as *mut _ as *mut c_void,
        ];
        let grid = ((m * ff) as u64).div_ceil(256) as u32;
        ok(unsafe { ctx.launch_async(ctx.f_swiglu_int, (grid, 1, 1), (256, 1, 1), 0, &mut args) })
    })
}


/// [`chain_mlp_prefill_q4`] の 1 タイル分 GEMM (per-op 経路と同一カーネル・
/// 同一グリッド・同一引数)。a/out はデバイスポインタ。
pub(crate) fn chain_gemm_q4_pf(
    gw: &GpuWq4,
    a: u64,
    out: u64,
    m: usize,
    k: usize,
    n: usize,
) -> Option<()> {
    let ctx = chain_ctx()?;
    unsafe {
        let (mut pa, mut pn, mut ps, mut po) = (a, gw.nib.ptr(), gw.sc.ptr(), out);
        let (mut mc, mut kc, mut nc) = (m as c_int, k as c_int, n as c_int);
        let mut args = [
            &mut pa as *mut _ as *mut c_void,
            &mut pn as *mut _ as *mut c_void,
            &mut ps as *mut _ as *mut c_void,
            &mut po as *mut _ as *mut c_void,
            &mut mc as *mut _ as *mut c_void,
            &mut kc as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
        ];
        let grid = (m.div_ceil(64) as u32, n.div_ceil(128) as u32, 1);
        ok(ctx.launch_async(ctx.f_gemm_q4, grid, (16, 8, 1), 0, &mut args))
    }
}


/// gate / up が別バッファにある swiglu (`dno_swiglu`)。
pub(crate) fn chain_swiglu_split(gate: u64, up: u64, out: u64, total: usize) -> Option<()> {
    let ctx = chain_ctx()?;
    unsafe {
        let (mut pg, mut pu, mut po) = (gate, up, out);
        let mut tc = total as c_int;
        let mut args = [
            &mut pg as *mut _ as *mut c_void,
            &mut pu as *mut _ as *mut c_void,
            &mut po as *mut _ as *mut c_void,
            &mut tc as *mut _ as *mut c_void,
        ];
        let grid = (total as u64).div_ceil(256) as u32;
        ok(ctx.launch_async(ctx.f_swiglu, (grid, 1, 1), (256, 1, 1), 0, &mut args))
    }
}


/// s173 (P14): 行内 [gate|up] レイアウト (`width = 2*ff`) の swiglu を生ポインタで
/// 起動する (`dno_swiglu_int`)。[`chain_swiglu_rows`] のスロット版と同じカーネル・
/// 同じ引数で、CHAIN を借りている最中 (タイルループ内) から呼べるようにしたもの。
/// 演算列は `dno_swiglu` と同一 (`dno_expf` + `__fdiv_rn` + `__fadd_rn` +
/// `__fmul_rn`) なので、CPU の `silu(g)*u` とビット一致。
pub(crate) fn chain_swiglu_int_at(ga: u64, out: u64, ff: usize, total: usize) -> Option<()> {
    let ctx = chain_ctx()?;
    if ff == 0 || total == 0 {
        return None;
    }
    unsafe {
        let (mut pg, mut po) = (ga, out);
        let mut ffc = ff as c_int;
        let mut width = (2 * ff) as c_int;
        let mut tc = total as i64;
        let mut args = [
            &mut pg as *mut _ as *mut c_void,
            &mut po as *mut _ as *mut c_void,
            &mut ffc as *mut _ as *mut c_void,
            &mut width as *mut _ as *mut c_void,
            &mut tc as *mut _ as *mut c_void,
        ];
        let grid = (total as u64).div_ceil(256) as u32;
        ok(ctx.launch_async(ctx.f_swiglu_int, (grid, 1, 1), (256, 1, 1), 0, &mut args))
    }
}


// ---------------------------------------------------------------------------
// P17g: GPU KV ミラー + decode attention (exactness 緩和・ユーザー承認)。
//
// ホスト KvCache は CPU 経路・snapshot/prefix cache の正として残し、decode
// chain で使う K/V のコピーをデバイス側に保持する (per-session ミラー)。
// ミラーはホストキャッシュの行 [0..len) と常に同一内容に保たれる
// (不足行は次回の chain_attention で一括 H2D)。ミラー破損/ロールバック/
// エビクト時も len 比較で自動再シードされるため、ホストが正・ミラーは従。
// メモリ上限 `DENNO_GPU_KV_MB` (既定 2048MB — P58-1 で 8192 から引き下げ。
// 27B 本番で既定のままだとミラーが約 1 GB 常駐し VRAM 枯渇を起こした) を超えると
// LRU でエビクト (必要時に再シード)。実使用量は起動時と各リクエスト完了時に
// `gpu kv mirror: N entries, X MB / limit Y MB` として info ログに出る。
//
// 注意: attention のみ exactness 契約が緩和されている (CPU と GPU で累積順が
// 異なり出力ビットは一致しない)。線形層 (int8) / expf / swiglu / rmsnorm は
// 従来どおりビット一致。`DENNO_GPU_ATTN=0` で無効化 (CPU attention・
// ビット一致経路に戻る)。

/// GPU attention の有効フラグ (既定 ON・`DENNO_GPU_ATTN=0` で無効)。
/// テストから差し替え可能にするため AtomicBool。
pub(crate) static GPU_ATTN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

pub(crate) static GPU_ATTN_INIT: std::sync::OnceLock<()> = std::sync::OnceLock::new();


pub(crate) struct KvMirror {
    /// (k, v) デバイスバッファ per layer。各 [cap, kv_dim] f32
    pub(crate) layers: Vec<(Option<DevBuffer>, Option<DevBuffer>)>,
    pub(crate) cap: usize,
    /// layer ごとにミラー済みの行数 (ホスト cache.len() との差分を H2D する)
    pub(crate) len: Vec<usize>,
    /// ミラー行が同期しているホスト KV コンテンツの世代 (s41)。
    /// KvCache::epoch() と一致するときだけ行 [0..len) / GDN 状態を再利用してよい。
    /// 不一致 (同一スロットで reset/restore → 別会話で再 prefill) なら with_mirror_ext
    /// が全層の len と gdn_pos を無効化する (s35 の 2b 2 値分岐の修正)。
    pub(crate) epoch: u64,
    /// C4 (qwen35): GDN 状態ミラー。GDN 層のみ Some (full attention 層は None)。
    /// 状態サイズはコンテキスト長に依存しないため cap 成長の影響を受けない。
    pub(crate) gdn: Vec<Option<GdnDev>>,
    /// GDN ミラーが同期している位置 (層ごと・cache.len() 相当)。Some(pos0) なら
    /// seed 不要。層ごとに持つ (層ごとの seed 判定を独立させる — 1 層でも
    /// gdn_pos を共有すると残り層が全て再 seed になってしまう)。
    pub(crate) gdn_pos: Vec<Option<usize>>,
    /// s50: 遅延 GDN 状態フラッシュの pending。decode chain は GDN の新状態
    /// (conv/ssm) をステップ毎にホストへ戻さず GPU ミラーを権威のままにする
    /// (ステップ毎 D2H が decode の支配項だったため)。Some(pos) = 最後に遅延した
    /// decode ステップの pos0+1 (正常完了後は cache.len() と一致)。Some の間は
    /// ホスト GDN 状態が古い可能性があり、chain_q35_gdn_flush_pending が
    /// 「ホスト状態が実際に必要になる時点」(prefill 開始前・CPU フォールバック時)
    /// でミラーから回復する。pending 保持中のミラーは LRU エビクト対象外
    /// (フラッシュ前にデータが失われると回復不能になるため)。
    pub(crate) gdn_pending: Option<usize>,
    /// s141b (Task B7c): 層ごとの「ホスト `KvCache::gdn` がデバイスより古い可能性が
    /// ある」フラグ。decode chain はステップ毎にホストへ戻さない (s50 から) ので、
    /// 一度でも scan を走らせた層はここが真になる。真の層はホストを seed に使っては
    /// ならない = `chain_q35_gdn_scan` の seed はデバイス正本から D2D で取る。
    /// ホストを読む側は `chain_q35_gdn_flush_pending` (= 本フラグのある層だけ D2H)
    /// か `KvCache::flush_gdn_device_state` を通ってから読むこと。
    /// ミラーに面 0 を書き直した層 (seed 直後) は偽に戻る。
    pub(crate) gdn_stale: Vec<bool>,
}


impl KvMirror {
    pub(crate) fn bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|(k, v)| {
                k.as_ref().map(|b| b.bytes()).unwrap_or(0)
                    + v.as_ref().map(|b| b.bytes()).unwrap_or(0)
            })
            .sum::<usize>()
            + self.gdn.iter().flatten().map(|g| g.bytes()).sum::<usize>()
    }
}


pub(crate) static KV_MIRRORS: std::sync::OnceLock<std::sync::Mutex<Vec<(u64, KvMirror, std::time::Instant)>>> =
    std::sync::OnceLock::new();


pub(crate) fn kv_mirror_cap(t_total: usize, old_cap: usize) -> usize {
    // s128 (P4-7): 1,024 行刻みで切り上げる。以前は next_power_of_two だったため
    // 10,880 token で 16,384 行 (= 必要量の 1.51 倍) を確保していた。
    // cap 成長は全層の再シードを伴うので、刻みは 1,024 行 (27B で 134 MB) に固定し、
    // 必要量を超えるときは一気に必要量まで上げる (リクエスト内で再成長しない)。
    let need = t_total.div_ceil(1024) * 1024;
    need.max(old_cap + 1024).max(1024)
}


/// decode chain 用 GPU attention。q (CPU 側で rope 済み [heads*dh]) と
/// ホスト KvCache (現在レイヤの行は呼び出し側が書き込み済み) を受け取り、
/// ミラーを同期して attention カーネルを起動し、結果を CH_ATT に書く。
/// 戻り None = CPU attention にフォールバック (forward 全体を従来経路で再実行)。
#[allow(clippy::too_many_arguments)]
pub fn chain_attention(
    key: u64,
    li: usize,
    n_layers: usize,
    q: &[f32],
    cache: &crate::kv_cache::KvCache,
    t_total: usize,
    group: usize,
    dh: usize,
    kv_dim: usize,
    scale: f32,
) -> Option<()> {
    if !gpu_attention_enabled() {
        return None;
    }
    let ctx = chain_ctx()?;
    let heads = q.len() / dh;
    if heads == 0 || group == 0 || dh == 0 || dh > 1024 || t_total == 0 || t_total > cache.max_len()
    {
        return None;
    }
    let mut kv_dim_c = kv_dim as c_int;
    let mut t_c = t_total as c_int;
    let mut group_c = group as c_int;
    let mut dh_c = dh as c_int;

    // CH_ATT は後続の o_proj 入力そのもの (chain_upload を省略)
    ok(chain_ensure_buf(CH_ATT, heads * dh * 4))?;

    // ミラーの取得/生成 + レイヤ li の行同期。ポインタはロック解放前に抽出する。
    let (mut kptr, mut vptr) =
        with_mirror(key, li, n_layers, t_total, kv_dim, cache.epoch(), |m, have| {
        // 行同期: ミラーが持っている [0..have) とホストの [0..t_total) を突き合わせ。
        // ホストの行 pos0 は呼び出し側が既に書き込み済み。
        // ロールバック (restore/reset) で have > t_total になったら全再シード (have=0)。
        let (kc_ref, vc_ref) = cache.kv_bufs_ref(li);
        if have < t_total {
            let kb = m.layers[li].0.as_ref().unwrap();
            let vb = m.layers[li].1.as_ref().unwrap();
            ok(kb.copy_from_at(have * kv_dim * 4, &kc_ref[have * kv_dim..t_total * kv_dim]))?;
            ok(vb.copy_from_at(have * kv_dim * 4, &vc_ref[have * kv_dim..t_total * kv_dim]))?;
        }
        m.len[li] = t_total;
        Some((
            m.layers[li].0.as_ref().unwrap().ptr(),
            m.layers[li].1.as_ref().unwrap().ptr(),
        ))
    })?;

    // q / scores スクラッチ (スレッドローカル — chain は 1 スレッド 1 セッション)
    CHAIN.with(|cell| -> Option<()> {
        let mut s = cell.borrow_mut();
        ok(Scratch::ensure_tagged(
            &mut s.attn_q,
            q.len() * 4,
            crate::mem_audit::TAG_CHAIN,
        ))?;
        ok(Scratch::ensure_tagged(
            &mut s.attn_sc,
            heads * t_total * 4,
            crate::mem_audit::TAG_CHAIN,
        ))?;
        ok(s.attn_q.as_ref().unwrap().copy_from(q))?;
        let mut qptr = s.attn_q.as_ref().unwrap().ptr();
        let mut scptr = s.attn_sc.as_ref().unwrap().ptr();
        let mut aptr = s.bufs[CH_ATT].as_ref().unwrap().ptr();
        let mut scale_c = scale;
        let mut args: Vec<*mut c_void> = vec![
            &mut qptr as *mut _ as *mut c_void,
            &mut kptr as *mut _ as *mut c_void,
            &mut vptr as *mut _ as *mut c_void,
            &mut aptr as *mut _ as *mut c_void,
            &mut scptr as *mut _ as *mut c_void,
            &mut kv_dim_c as *mut _ as *mut c_void,
            &mut t_c as *mut _ as *mut c_void,
            &mut group_c as *mut _ as *mut c_void,
            &mut dh_c as *mut _ as *mut c_void,
            &mut scale_c as *mut _ as *mut c_void,
        ];
        ok(unsafe {
            ctx.launch_async(
                ctx.f_attn,
                (heads as u32, 1, 1),
                (dh as u32, 1, 1),
                (dh * 4) as u32,
                &mut args,
            )
        })
    })
}

/// task 135: attention 自身の構造上の下限 (1 チャンクの行数)。
///
/// 従来は `attn_prefill_gpu_supported` が `gemm_min_m()` (既定 256) を借りて
/// いたため、next turn の 19 行チャンクや長い prefill の尾チャンク (194 / 242 行)
/// が CPU に落ちていた。GEMM の 256 は「CPU sgemm の方が速い帯域」の話で、
/// attention のカーネル制約ではない (制約は dh >= ATTN_PF_BQ と smem だけ)。
///
/// lab t135a / t135b (depth, depth8k) の実測: 8 行で next turn prefill が
/// c2000 1.41 -> 1.03 s、c8000 2.53 -> 1.10 s (19 行・14k 文脈で 1 層 127 ms ->
/// 10 ms)、cold prefill も尾チャンクが乗って -3.3% / -3.9%。decode と短い
/// プロンプトはノイズ内 (±2%)、md5 は 6 本とも同一。7 行以下は CPU に残す。
pub(crate) const ATTN_Q4_PF_MIN_ROWS: usize = 8;


/// `DENNO_Q4_GPU_ATTN` の解釈 (純関数・env を読まない)。
pub(crate) fn parse_q4_gpu_attn(v: Option<&str>) -> bool {
    match v.map(str::trim) {
        Some("0") | Some("off") | Some("false") => false,
        Some("1") | Some("on") | Some("true") => true,
        _ => Q4_GPU_ATTN_DEFAULT,
    }
}


/// `DENNO_Q4_GPU_ATTN_MIN` の解釈 (純関数)。未設定・不正値は既定。
pub(crate) fn parse_q4_gpu_attn_min(v: Option<&str>) -> usize {
    v.and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(Q4_GPU_ATTN_MIN_DEFAULT)
}


/// `DENNO_Q4_ATTN_MIRROR_MB` の解釈 (純関数)。未設定・不正値は既定。
fn q4_attn_mirror_limit_from(v: Option<usize>) -> usize {
    v.unwrap_or(Q4_GPU_ATTN_MIRROR_MB_DEFAULT) * 1024 * 1024
}


fn q4_attn_mirror_limit_bytes() -> usize {
    q4_attn_mirror_limit_from(
        std::env::var("DENNO_Q4_ATTN_MIRROR_MB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok()),
    )
}


/// テスト用: env を介さず q4 GPU attention の ON/OFF を差し替える
/// (`None` で env の解釈に戻す)。
#[cfg(test)]
#[doc(hidden)]
pub fn set_q4_gpu_attn_for_test(on: Option<bool>) {
    Q4_ATTN_FORCE.store(match on {
        Some(true) => 1,
        Some(false) => 0,
        None => -1,
    }, std::sync::atomic::Ordering::Relaxed);
}


/// テスト用: `DENNO_Q4_GPU_ATTN_MIN` を差し替える (`None` で env の解釈に戻す)。
#[cfg(test)]
#[doc(hidden)]
pub fn set_q4_gpu_attn_min_for_test(min: Option<usize>) {
    Q4_ATTN_MIN_FORCE.store(min.unwrap_or(usize::MAX), std::sync::atomic::Ordering::Relaxed);
}


// ---- task 134: prefill の full attention を GPU へ -------------------------

/// `DENNO_Q4_GPU_ATTN_PF` の解釈 (純関数・env を読まない)。未設定は既定。
pub(crate) fn parse_q4_gpu_attn_pf(v: Option<&str>) -> bool {
    match v.map(str::trim) {
        Some("0") | Some("off") | Some("false") => false,
        Some("1") | Some("on") | Some("true") => true,
        _ => Q4_GPU_ATTN_PF_DEFAULT,
    }
}


/// テスト用: prefill ゲートを env を介さず差し替える (`None` で env の解釈に戻す)。
#[cfg(test)]
#[doc(hidden)]
pub fn set_q4_gpu_attn_pf_for_test(on: Option<bool>) {
    Q4_ATTN_PF_FORCE.store(
        match on {
            Some(true) => 1,
            Some(false) => 0,
            None => -1,
        },
        std::sync::atomic::Ordering::Relaxed,
    );
}


/// テスト用: prefill のしきい値 (行数・文脈長) を差し替える
/// (`None` で env の解釈に戻す)。
#[cfg(test)]
#[doc(hidden)]
pub fn set_q4_gpu_attn_pf_min_for_test(rows: Option<usize>, kv: Option<usize>) {
    Q4_ATTN_PF_MIN_FORCE.store(rows.unwrap_or(usize::MAX), std::sync::atomic::Ordering::Relaxed);
    Q4_ATTN_PF_MIN_KV_FORCE.store(kv.unwrap_or(usize::MAX), std::sync::atomic::Ordering::Relaxed);
}


// ---- task 135: decode のカーネル選択と t 分割 -----------------------------

/// decode 経路で使うカーネル (`DENNO_Q4_GPU_ATTN_KERNEL`)。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Q4AttnKernel {
    /// 既存 `dno_attention` (1 q head = 1 block・dno_expf)。qwen35 もこれ。
    Decode,
    /// 既存 `dno_attention_prefill` を n=1・pos0=t_total-1 で使う (exp2f)。
    Prefill,
    /// task 135 で追加した `dno_attention_decode` (t を分割・exp2f)。
    Split,
    /// task 448: flash decoding 第 2 版 (`kernels_attn_dec2.cu` の
    /// `dno_attn_flashdec_t`、K/V を shared のタイルへ協調ロード)。
    /// flash decoding の門 (`DENNO_ATTN_FLASHDEC` と文脈長) は共通で、
    /// この値は「どちらのカーネルで回すか」だけを選ぶ。
    Tile,
    /// task 448: flash decoding 第 1 版 (`dno_attn_flashdec`) を強制する。
    /// タイル版の A/B と切り戻し用 (数値はどちらも固定分割・固定順)。
    Fdec1,
}


/// task 448: flash decoding を **タイル版** (`dno_attn_flashdec_t`) で回すか。
///
/// `DENNO_ATTN_FLASHDEC_TILE` (0/off で v1、4..=64 で on) が最優先、
/// 次に `DENNO_ATTN_FLASHDEC2`、未設定なら `DENNO_Q4_GPU_ATTN_KERNEL`
/// (`tile` で on、`fdec1` で v1) を見る。どれも未設定ならこの既定。
/// task 448 の lab t448d の A/B (これ = タイル版 vs `DENNO_ATTN_FLASHDEC_TILE=0`
/// = v1) で採否を決めた。
pub(crate) const ATTN_FLASHDEC_TILE_DEFAULT: bool = true;


/// `dno_attention_decode` の 1 タイルの key 数。`.cu` の `DNO_DEC_BK` と一致
/// させること (動的 smem の見積りに使う)。
pub(crate) const ATTN_DEC_BK: usize = 8;


/// `DENNO_Q4_GPU_ATTN_KERNEL` の解釈 (純関数・env を読まない)。
pub(crate) fn parse_q4_gpu_attn_kernel(v: Option<&str>) -> Q4AttnKernel {
    match v.map(str::trim) {
        Some("decode") | Some("dno") => Q4AttnKernel::Decode,
        Some("pf") | Some("prefill") => Q4AttnKernel::Prefill,
        Some("split") | Some("dec") => Q4AttnKernel::Split,
        // task 448: flash decoding の実装選択 (v1 / タイル版)。
        Some("tile") | Some("fdt") | Some("fdec2") => Q4AttnKernel::Tile,
        Some("fdec1") | Some("fdec") => Q4AttnKernel::Fdec1,
        _ => Q4_GPU_ATTN_KERNEL_DEFAULT,
    }
}


/// `DENNO_Q4_GPU_ATTN_SPLIT` の解釈 (純関数)。1..=64 に丸める。
pub(crate) fn parse_q4_gpu_attn_split(v: Option<&str>) -> usize {
    match v.and_then(|s| s.trim().parse::<usize>().ok()) {
        Some(n) => n.clamp(1, 64),
        None => Q4_GPU_ATTN_SPLIT_DEFAULT,
    }
}


// ---- task 160: flash decoding (split-K・GQA まとめ読み) ---------------------
//
// 既存 `dno_attention_decode` は 1 block = (split, q head) で、GQA の group 本の
// q head が同じ KV 行を group 回読む (L2 で拾えるが block 数も 24x8 しか無い)。
// 新しい `kernels_attn_dec.cu` は 1 block = (split, kv head)、block 内の warp が
// q head を 1 本ずつ持ち、行は L1 で共有する。詳細は .cu の先頭コメント。
//
// 既定は lab の結果で決める (`ATTN_FLASHDEC_DEFAULT` / `_MIN_CTX_DEFAULT`)。

/// `.cu` の `DNO_FD_MAXB`。1 lane が 4 要素ずつ持つので dh <= MAXB * 128。
pub(crate) const ATTN_FDEC_MAXB: usize = 4;

/// part 1 エントリの float 数 = dh + 2 (先頭 2 つが m, l)。`.cu` と一致させる。
const ATTN_FDEC_PART_EXTRA: usize = 2;

/// 分割数の上限 (`.cu` の grid.x と combine のループ上限)。
/// lab t160a: n_kv = 2 のモデルでは 56 分割 = 112 block でしかなく SM 170 を
/// 埋められなかった (14k で 258 us = 212 GB/s)。分割数は block 数そのものなので
/// 256 まで許す (combine は split 昇順の逐次ループなので上限だけが制約)。
pub(crate) const ATTN_FDEC_SPLIT_MAX: usize = 256;

/// 1 分割あたりの key 数の既定 (`DENNO_ATTN_FLASHDEC_CHUNK`)。
pub(crate) const ATTN_FLASHDEC_CHUNK_DEFAULT: usize = 64;

/// 1 分割の key 数の下限 (これ未満に割ると block だけ増えて無駄)。
pub(crate) const ATTN_FLASHDEC_CHUNK_MIN: usize = 32;


/// `DENNO_ATTN_FLASHDEC` の既定。lab t163a (新ホスト、depth8k) で採用:
/// 14,157 next turn 31 -> 27 ms/tok (-12.9%)、3,965 27 -> 26 (-3.7%)、
/// 短い decode と prefill は 2% 以内、md5 SAME。既定 on。
const ATTN_FLASHDEC_DEFAULT: bool = true;

/// この KV 本数未満は従来カーネル (`DENNO_ATTN_FLASHDEC_MIN_CTX`)。
const ATTN_FLASHDEC_MIN_CTX_DEFAULT: usize = 1024;


#[cfg(test)]
static ATTN_FDEC_FORCE: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);

#[cfg(test)]
static ATTN_FDEC_MIN_CTX_FORCE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(usize::MAX);

#[cfg(test)]
static ATTN_FDEC_CHUNK_FORCE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(usize::MAX);


/// `DENNO_ATTN_FLASHDEC` の解釈 (純関数)。未設定・不正値は既定。
fn parse_attn_flashdec(v: Option<&str>) -> bool {
    match v.map(str::trim) {
        Some("0") => false,
        Some("1") => true,
        _ => ATTN_FLASHDEC_DEFAULT,
    }
}


/// `DENNO_ATTN_FLASHDEC_MIN_CTX` の解釈 (純関数)。
fn parse_attn_flashdec_min_ctx(v: Option<&str>) -> usize {
    match v.and_then(|s| s.trim().parse::<usize>().ok()) {
        Some(n) => n,
        None => ATTN_FLASHDEC_MIN_CTX_DEFAULT,
    }
}


/// `DENNO_ATTN_FLASHDEC_CHUNK` の解釈 (純関数)。32..=4096 に丸める。
pub(crate) fn parse_attn_flashdec_chunk(v: Option<&str>) -> usize {
    match v.and_then(|s| s.trim().parse::<usize>().ok()) {
        Some(n) => n.clamp(ATTN_FLASHDEC_CHUNK_MIN, 4096),
        None => ATTN_FLASHDEC_CHUNK_DEFAULT,
    }
}


// ---------------------------------------------------------------------------
// task 460: 1 分割の key 数を文脈長で変える (`DENNO_ATTN_FLASHDEC_CHUNK=auto`)
// ---------------------------------------------------------------------------
//
// 動機 (task 456 の lab t456c、既定モデル、depth8k):
//   chunk 128 にすると 14k の full 段が 1.77 -> 1.65 ms/token (-6.6%、n_split が
//   220 -> 110 になり combine の読みが半減) だが、2k では 1.19 -> 1.38 (+16%)
//   になる (block 数 2*n_split = 62 が SM 170 を割る)。よって固定値ではなく
//   **block 数 2*n_split >= 220 を保つ範囲で 1 分割を大きくする**。
//
// 規則 (純関数): chunk = ceil(len / 110) を 64..=128 に clamp。
//   - 14,018 -> 128 (t456c で測った最良と同じ n_split 110)
//   - 3,965 / 2,000 -> 64 (現状と同じ。n_split 62 を 32 にしない)
//   - 8,000 -> 73 (n_split 110)
/// `auto` のときの目標分割数 (block 数 2*n_split が SM 170 を余裕で超える)。
pub(crate) const ATTN_FLASHDEC_AUTO_SPLITS: usize = 110;

/// `auto` のときの 1 分割の上限 key 数。
pub(crate) const ATTN_FLASHDEC_AUTO_CHUNK_MAX: usize = 128;


/// `DENNO_ATTN_FLASHDEC_CHUNK` が `auto` か (純関数。前後の空白と大小は無視)。
pub(crate) fn parse_attn_flashdec_chunk_auto(v: Option<&str>) -> bool {
    matches!(
        v.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("auto")
    )
}


/// `DENNO_ATTN_FLASHDEC_CHUNK` の値を「auto かどうか」に落とす (純関数)。
///
/// 未設定・空・`auto` は auto (**既定 = auto**)。数値は固定値のまま (切り戻し)、
/// 未知の値も固定 64。lab t460a/t460b で 14k の full 段が -6 to -8% だったので
/// 既定を auto にした。切り戻しは `DENNO_ATTN_FLASHDEC_CHUNK=64`。
pub(crate) fn attn_fdec_chunk_is_auto(v: Option<&str>) -> bool {
    match v {
        None => true,
        Some(s) => {
            let t = s.trim();
            if t.is_empty() {
                true
            } else if t.parse::<usize>().is_ok() {
                false
            } else {
                parse_attn_flashdec_chunk_auto(Some(t))
            }
        }
    }
}


/// テスト用: flash decoding のゲートを差し替える (`None` で env の解釈に戻す)。
#[cfg(test)]
#[doc(hidden)]
pub fn set_attn_flashdec_for_test(on: Option<bool>) {
    ATTN_FDEC_FORCE.store(
        match on {
            Some(true) => 1,
            Some(false) => 0,
            None => -1,
        },
        std::sync::atomic::Ordering::Relaxed,
    );
}


/// テスト用: 短い文脈のしきい値を差し替える (`None` で env の解釈に戻す)。
#[cfg(test)]
#[doc(hidden)]
pub fn set_attn_flashdec_min_ctx_for_test(n: Option<usize>) {
    ATTN_FDEC_MIN_CTX_FORCE.store(n.unwrap_or(usize::MAX), std::sync::atomic::Ordering::Relaxed);
}


/// テスト用: 1 分割の key 数を差し替える (`None` で env の解釈に戻す)。
#[cfg(test)]
#[doc(hidden)]
pub fn set_attn_flashdec_chunk_for_test(n: Option<usize>) {
    ATTN_FDEC_CHUNK_FORCE.store(n.unwrap_or(usize::MAX), std::sync::atomic::Ordering::Relaxed);
}


/// flash decoding を使うか (`DENNO_ATTN_FLASHDEC`)。
pub(crate) fn attn_flashdec_enabled() -> bool {
    #[cfg(test)]
    {
        let f = ATTN_FDEC_FORCE.load(std::sync::atomic::Ordering::Relaxed);
        if f >= 0 {
            return f == 1;
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse_attn_flashdec(std::env::var("DENNO_ATTN_FLASHDEC").ok().as_deref()))
}


/// この KV 本数未満では flash decoding を使わない (`DENNO_ATTN_FLASHDEC_MIN_CTX`)。
pub(crate) fn attn_flashdec_min_ctx() -> usize {
    #[cfg(test)]
    {
        let f = ATTN_FDEC_MIN_CTX_FORCE.load(std::sync::atomic::Ordering::Relaxed);
        if f != usize::MAX {
            return f;
        }
    }
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        parse_attn_flashdec_min_ctx(std::env::var("DENNO_ATTN_FLASHDEC_MIN_CTX").ok().as_deref())
    })
}


/// 1 分割あたりの key 数 (`DENNO_ATTN_FLASHDEC_CHUNK`)。
fn attn_flashdec_chunk() -> usize {
    #[cfg(test)]
    {
        let f = ATTN_FDEC_CHUNK_FORCE.load(std::sync::atomic::Ordering::Relaxed);
        if f != usize::MAX {
            return f;
        }
    }
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        parse_attn_flashdec_chunk(std::env::var("DENNO_ATTN_FLASHDEC_CHUNK").ok().as_deref())
    })
}


/// この文脈長で使う 1 分割の key 数 (task 460)。
///
//   既定 (未設定・空) と `auto` のときは文脈長で変える ([`q4_fdec_chunk_auto`])。
/// 数値を書いたときだけ [`attn_flashdec_chunk`] の固定値 (切り戻し)。
fn attn_flashdec_chunk_for(t_total: usize) -> usize {
    #[cfg(test)]
    {
        let f = ATTN_FDEC_CHUNK_FORCE.load(std::sync::atomic::Ordering::Relaxed);
        if f != usize::MAX {
            return f;
        }
    }
    static AUTO: OnceLock<bool> = OnceLock::new();
    let auto = *AUTO.get_or_init(|| {
        attn_fdec_chunk_is_auto(std::env::var("DENNO_ATTN_FLASHDEC_CHUNK").ok().as_deref())
    });
    if auto {
        q4_fdec_chunk_auto(t_total)
    } else {
        attn_flashdec_chunk()
    }
}


/// タイル版の tile 行数の既定 (`DENNO_ATTN_FLASHDEC_TILE`、4..=64)。
const ATTN_FD_TILE_DEFAULT: usize = 16;


/// `DENNO_ATTN_FLASHDEC_TILE` の解釈 (純関数)。4..=64 に丸める。
pub(crate) fn parse_attn_fdec_tile(v: Option<&str>) -> usize {
    match v.and_then(|s| s.trim().parse::<usize>().ok()) {
        Some(n) => n.clamp(4, 64),
        None => ATTN_FD_TILE_DEFAULT,
    }
}


/// タイル版の tile 行数 (`DENNO_ATTN_FLASHDEC_TILE`)。
pub(crate) fn attn_fdec_tile() -> usize {
    #[cfg(test)]
    {
        let f = FD_TILE_FORCE.load(std::sync::atomic::Ordering::Relaxed);
        if f != usize::MAX {
            return f.clamp(4, 64);
        }
    }
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| parse_attn_fdec_tile(std::env::var("DENNO_ATTN_FLASHDEC_TILE").ok().as_deref()))
}


/// テスト用: tile 行数を差し替える (`None` で env の解釈に戻す)。
#[cfg(test)]
#[doc(hidden)]
pub fn set_attn_fdec_tile_for_test(n: Option<usize>) {
    FD_TILE_FORCE.store(n.unwrap_or(usize::MAX), std::sync::atomic::Ordering::Relaxed);
}


/// task 448: `DENNO_ATTN_FLASHDEC_TILE` を on/off の門としても使う (純関数)。
///
/// - 未設定・空: `None` (既定 [`ATTN_FLASHDEC_TILE_DEFAULT`] に従う)
/// - `0` / `off` / `false` / `no`: `Some(false)` = タイル版を使わない (v1)
/// - それ以外 (数値・`on` など): `Some(true)` = タイル版。行数は
///   [`parse_attn_fdec_tile`] が 4..=64 に丸める
///
/// lab の COND は **名前に数字を含む env を落とす** 回がある (t448a の
/// `DENNO_Q4_GPU_ATTN_KERNEL=tile` と t448b の `DENNO_ATTN_FLASHDEC2=1` は
/// どちらも届かず v1 が走った。同じ COND の `DENNO_ATTN_FLASHDEC_TILE=24` は
/// 届いた) ので、A/B は数字を含まないこの名前で取る。
pub(crate) fn attn_fdec_tile_ctl(v: Option<&str>) -> Option<bool> {
    let t = v?.trim().to_ascii_lowercase();
    if t.is_empty() {
        return None;
    }
    match t.as_str() {
        "0" | "off" | "false" | "no" => Some(false),
        _ => Some(true),
    }
}


/// `DENNO_ATTN_FLASHDEC_TILE` / `DENNO_ATTN_FLASHDEC2` の実値 (1 回だけ読む)。
fn attn_fdec_tile_env() -> Option<bool> {
    static V: OnceLock<Option<bool>> = OnceLock::new();
    *V.get_or_init(|| {
        attn_fdec_tile_ctl(std::env::var("DENNO_ATTN_FLASHDEC_TILE").ok().as_deref())
            .or_else(|| {
                fdec_tile_from_env(std::env::var("DENNO_ATTN_FLASHDEC2").ok().as_deref())
            })
    })
}


/// task 448: タイル版で回すか (`DENNO_ATTN_FLASHDEC_TILE` が優先、
/// 次に `DENNO_ATTN_FLASHDEC2`、未設定なら `DENNO_Q4_GPU_ATTN_KERNEL`)。
pub(crate) fn attn_fdec_tile_on() -> bool {
    match attn_fdec_tile_env() {
        Some(b) => b,
        None => fdec_tile_from_kernel(q4_gpu_attn_kernel()),
    }
}


/// task 448: 起動時の 1 行 (診断)。lab の COND で env が届いたか、測った
/// バイナリが新しいかをログだけで判別できるようにする。数値経路には触らない。
pub fn attn_fdec_boot_line() -> String {
    let tile_on = attn_fdec_tile_on();
    let k = match q4_gpu_attn_kernel() {
        Q4AttnKernel::Decode => "decode",
        Q4AttnKernel::Prefill => "pf",
        Q4AttnKernel::Split => "split",
        Q4AttnKernel::Tile => "tile",
        Q4AttnKernel::Fdec1 => "fdec1",
    };
    format!(
        "[attn-fdec] kernel={k} fdt_on={tile_on} tile={} ilp={} khx={} fdtv=5 (task 520)",
        attn_fdec_tile(),
        attn_fdec_ilp_on(),
        attn_fdec_khx_on()
    )
}


/// `DENNO_ATTN_FLASHDEC_ILP` の解釈 (純関数)。**既定 on**。
/// 明示的な偽値 (`0` / `off` / `false` / `no`) のときだけ第 2 版へ戻す。
pub(crate) fn parse_attn_fdec_ilp(v: Option<&str>) -> bool {
    match v {
        Some(s) => !matches!(
            s.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        ),
        None => true,
    }
}


/// テスト用: 第 3 版のゲートを差し替える (`None` で env の解釈に戻す)。
#[cfg(test)]
#[doc(hidden)]
pub fn set_attn_fdec_ilp_for_test(on: Option<bool>) {
    FD_ILP_FORCE.store(
        match on {
            Some(true) => 1,
            Some(false) => 0,
            None => -1,
        },
        std::sync::atomic::Ordering::Relaxed,
    );
}


/// 第 3 版を使うか (`DENNO_ATTN_FLASHDEC_ILP`、既定 on)。
pub(crate) fn attn_fdec_ilp_on() -> bool {
    #[cfg(test)]
    {
        let f = FD_ILP_FORCE.load(std::sync::atomic::Ordering::Relaxed);
        if f >= 0 {
            return f == 1;
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        parse_attn_fdec_ilp(std::env::var("DENNO_ATTN_FLASHDEC_ILP").ok().as_deref())
    })
}


/// `DENNO_ATTN_FLASHDEC_KHX` の解釈 (純関数)。**既定 off**。
///
/// grid を (n_kv, n_split) に入れ替えて、同じ row の kv head 0/1 の半分
/// (各 1 KB) を隣り合う block が同時に読むようにする。計算は 1 ビットも
/// 変わらない (block の (kh, sp) の割り当てだけ)。
pub(crate) fn parse_attn_fdec_khx(v: Option<&str>) -> bool {
    match v {
        Some(s) => matches!(
            s.trim().to_ascii_lowercase().as_str(),
            "1" | "on" | "true" | "yes"
        ),
        None => false,
    }
}


/// テスト用: grid 入れ替えのゲートを差し替える (`None` で env の解釈に戻す)。
#[cfg(test)]
#[doc(hidden)]
pub fn set_attn_fdec_khx_for_test(on: Option<bool>) {
    FD_KHX_FORCE.store(
        match on {
            Some(true) => 1,
            Some(false) => 0,
            None => -1,
        },
        std::sync::atomic::Ordering::Relaxed,
    );
}


/// grid を入れ替えるか (`DENNO_ATTN_FLASHDEC_KHX`、既定 off)。
pub(crate) fn attn_fdec_khx_on() -> bool {
    #[cfg(test)]
    {
        let f = FD_KHX_FORCE.load(std::sync::atomic::Ordering::Relaxed);
        if f >= 0 {
            return f == 1;
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        parse_attn_fdec_khx(std::env::var("DENNO_ATTN_FLASHDEC_KHX").ok().as_deref())
    })
}


/// `DENNO_ATTN_FLASHDEC_CBG` の解釈 (純関数)。**既定 on**。
///
/// lab t460b (既定モデル、depth8k) で full 段が 14k 後 1.76 -> 1.63 ms/token
/// (-7.6%、`dno_attn_flashdec_combine` 16.2 + `dno_dec_gate_mul` 3.8 us/層が
/// 融合版 10.8 us/層になった)。明示的な偽値 (`0` / `off` / `false` / `no`) だけ切る。
/// PTX が無い・ABI が違うときは combine 単体 + gate の従来 2 起動へ落ちる。
pub(crate) fn parse_attn_fdec_cbg(v: Option<&str>) -> bool {
    match v {
        Some(s) => !matches!(
            s.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        ),
        None => true,
    }
}


/// テスト用: 融合ゲートを差し替える (`None` で env の解釈に戻す)。
#[cfg(test)]
#[doc(hidden)]
pub fn set_attn_fdec_cbg_for_test(on: Option<bool>) {
    FD_CBG_FORCE.store(
        match on {
            Some(true) => 1,
            Some(false) => 0,
            None => -1,
        },
        std::sync::atomic::Ordering::Relaxed,
    );
}


/// combine + ゲートの融合を使うか (`DENNO_ATTN_FLASHDEC_CBG`、既定 on)。
pub(crate) fn attn_fdec_cbg_on() -> bool {
    #[cfg(test)]
    {
        let f = FD_CBG_FORCE.load(std::sync::atomic::Ordering::Relaxed);
        if f >= 0 {
            return f == 1;
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        parse_attn_fdec_cbg(std::env::var("DENNO_ATTN_FLASHDEC_CBG").ok().as_deref())
    })
}


/// テスト用: decode のカーネル選択を差し替える (`None` で env の解釈に戻す)。
#[cfg(test)]
#[doc(hidden)]
pub fn set_q4_gpu_attn_kernel_for_test(k: Option<Q4AttnKernel>) {
    Q4_ATTN_KERNEL_FORCE.store(
        match k {
            Some(Q4AttnKernel::Decode) => 0,
            Some(Q4AttnKernel::Prefill) => 1,
            Some(Q4AttnKernel::Split) => 2,
            Some(Q4AttnKernel::Tile) => 3,
            Some(Q4AttnKernel::Fdec1) => 4,
            None => -1,
        },
        std::sync::atomic::Ordering::Relaxed,
    );
}


/// テスト用: 分割数を差し替える (`None` で env の解釈に戻す)。
#[cfg(test)]
#[doc(hidden)]
pub fn set_q4_gpu_attn_split_for_test(n: Option<usize>) {
    Q4_ATTN_SPLIT_FORCE.store(n.unwrap_or(usize::MAX), std::sync::atomic::Ordering::Relaxed);
}


/// prefill の attention を GPU で回してよいか (env としきい値だけ。
/// デバイス・形状の制約は [`attn_prefill_gpu_supported`] 側で見る)。
///
/// 短いプロンプトを CPU に残すための 2 段のしきい値:
/// `n` (チャンクのトークン数) と `t_total` (チャンク終端の位置数)。
pub(crate) fn q4_attn_pf_eligible(n: usize, t_total: usize) -> bool {
    q4_attn_pf_gate_values(
        q4_gpu_attn_pf_enabled(),
        n,
        t_total,
        q4_gpu_attn_pf_min_rows(),
        q4_gpu_attn_pf_min_kv(),
    )
}


/// [`q4_attn_pf_eligible`] の判定そのもの (純関数。env もデバイスも読まない)。
pub(crate) fn q4_attn_pf_gate_values(
    on: bool,
    n: usize,
    t_total: usize,
    min_rows: usize,
    min_kv: usize,
) -> bool {
    if !on || n == 0 || t_total < n {
        return false;
    }
    n >= min_rows && t_total >= min_kv
}


/// 失敗は静かに CPU へ戻す (GPU_BROKEN は立てない)。
pub(crate) fn q4_attn_fail(e: anyhow::Error) -> Option<()> {
    static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        tracing::warn!("q4 gpu attention failed, using CPU: {e:#}");
    }
    None
}


/// task 174: 投機読み 1 行ぶんの余白。
///
/// flashdec (`kernels_attn_dec.cu` の `dno_attn_flashdec`) は 2 段パイプラインの
/// 先読みで**分割の終端の 1 行先** `jhi` を読む。最終分割ではそれが `t_total`
/// 行目 = テンソルの外で、読んだ値は使わない (ループはその手前で終わる) が、
/// `t_total` がミラーの cap とちょうど等しいとき (len = 4,096 / cap = 4,096 など)
/// にその 1 行が**確保の外**を踏み、`cuCtxSynchronize` が error 700 になる。
/// lab t174a の失敗はまさに len=4,096 (= cap) の 1 回目で、1 行先を指す
/// `H2D@ ... +8,386,560 B` (= 4,095 行目) の直後だった。
///
/// ミラーの確保だけ cap + 1 行に増やして、この投機読みを確保の内側に収める
/// (台帳の cap は変えないので、成長の段も分割数も今までどおり)。
pub(crate) const ATTN_MIRROR_GUARD_ROWS: usize = 1;


pub(crate) struct Q4AttnMirror {
    kv_dim: usize,
    state: Q4MirrorState,
    k: Vec<Option<DevBuffer>>,
    v: Vec<Option<DevBuffer>>,
}


impl Q4AttnMirror {
    pub(crate) fn new(n_layer: usize, kv_dim: usize) -> Self {
        Self {
            kv_dim,
            state: Q4MirrorState {
                cap: 0,
                len: vec![0; n_layer],
                gen: vec![0; n_layer],
            },
            k: (0..n_layer).map(|_| None).collect(),
            v: (0..n_layer).map(|_| None).collect(),
        }
    }

    pub(crate) fn bytes(&self) -> usize {
        self.k
            .iter()
            .chain(self.v.iter())
            .filter_map(|b| b.as_ref())
            .map(|b| b.bytes())
            .sum()
    }
}


/// ミラーの実使用量 `(エントリ数, 合計バイト数, 上限バイト数)` (診断用)。
pub fn q4_attn_mirror_stats() -> Option<(usize, usize, usize)> {
    let mirrors = Q4_ATTN_MIRRORS.get()?;
    let ms = match mirrors.lock() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    };
    let used = ms.iter().map(|(_, m, _)| m.bytes()).sum();
    Some((ms.len(), used, q4_attn_mirror_limit_bytes()))
}


/// task 210: [`q4_attn_mirror_forget`] の呼び出し回数 (プロセス通算)。
pub fn q4_attn_mirror_forgets() -> usize {
    Q4_ATTN_MIRROR_FORGETS.load(std::sync::atomic::Ordering::Relaxed)
}


/// task 210: `DENNO_ATTN_MIRROR_FORGET` (既定 on)。`0` / `false` / `no` で
/// 「cache を捨てても GPU ミラーを残す」= task 209 以前の挙動 (A/B と切り戻し用)。
static ATTN_MIRROR_FORGET: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

static ATTN_MIRROR_FORGET_INIT: std::sync::OnceLock<()> = std::sync::OnceLock::new();


/// task 210: 捨てた `AttnCache` のミラーを解放するか。
pub fn attn_mirror_forget_on() -> bool {
    ATTN_MIRROR_FORGET_INIT.get_or_init(|| {
        let on = mirror_forget_from_value(
            std::env::var("DENNO_ATTN_MIRROR_FORGET").ok().as_deref(),
        );
        ATTN_MIRROR_FORGET.store(on, std::sync::atomic::Ordering::Relaxed);
    });
    ATTN_MIRROR_FORGET.load(std::sync::atomic::Ordering::Relaxed)
}


/// テスト用: ゲートを直接差し替える (戻り値は設定前の値)。
#[doc(hidden)]
pub fn set_attn_mirror_forget_for_test(on: bool) -> bool {
    ATTN_MIRROR_FORGET_INIT.get_or_init(|| {});
    ATTN_MIRROR_FORGET.swap(on, std::sync::atomic::Ordering::Relaxed)
}


/// task 210: 系列状態 ([`crate::model::qwen4exp_attn::AttnCache`]) が消えたら、
/// その ID のミラーを VRAM から落とす。
///
/// ミラーは `AttnCache::id()` を鍵にした LRU で、**捨てた cache のぶんも上限
/// (既定 3584 MiB) まで残る**。task 206 の文脈の差し替えは
/// `take_seq_state` が空の cache を新しい ID で作り、置き換えのリクエストが
/// それを使うので、リクエストごとに 1 本ずつ増える。lab t209a の 14k では
/// 死んだ 5 本が 384 MiB を占め、prefill の頭で VRAM が 25 MiB まで詰まって
/// expert の LRU を全部追い出させた (decode 21 → 35 ms/tok)。
///
/// 存在しない ID でも何もしない (呼び出し側は `Drop` から呼ぶ)。
pub fn q4_attn_mirror_forget(cache_id: u64) {
    // 発火テスト用のカウントは「呼ばれたか」なので、ミラーが 1 本も無い
    // (= static 未初期化) でも積む。
    Q4_ATTN_MIRROR_FORGETS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let Some(mirrors) = Q4_ATTN_MIRRORS.get() else {
        return;
    };
    let mut ms = match mirrors.lock() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    };
    Q4_ATTN_MIRROR_FORGETS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    ms.retain(|(k, _, _)| *k != cache_id);
}


/// cap 成長: **既に使った層だけ**バッファを作り直し、旧バッファの有効行を
/// D2D で温存する (再シードしない)。まだ使っていない層 (ssm 層) は確保しない
/// — AttnCache は全 48 層ぶんの枠を持つが、full attention は 12 層だけなので、
/// ここで絞らないと VRAM が 4 倍になる (65,536 行で 12 GiB)。
/// 失敗したら呼び出し側がエントリごと捨てる。
fn q4_attn_grow(m: &mut Q4AttnMirror, new_cap: usize, kv_dim: usize) -> Result<()> {
    let bytes = q4_mirror_alloc_rows(new_cap)
        .checked_mul(kv_dim)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| anyhow!("q4 attn mirror: cap {new_cap} が大きすぎる"))?;
    for l in 0..m.state.len.len() {
        if m.k[l].is_none() && m.v[l].is_none() {
            continue; // まだ使っていない層 (full attention でない層)
        }
        let keep = m.state.len[l].min(new_cap);
        for slot in [&mut m.k[l], &mut m.v[l]] {
            let nb = DevBuffer::alloc_tagged(bytes, crate::mem_audit::TAG_ATTN_MIRROR)?;
            if keep > 0 {
                if let Some(old) = slot.as_ref() {
                    nb.copy_from_dev(old.ptr(), keep * kv_dim)?;
                }
            }
            *slot = Some(nb);
        }
    }
    m.state.cap = new_cap;
    Ok(())
}


/// 層 `layer` の行 `[0..len)` をミラーへ同期し、K/V のデバイスポインタを返す。
///
/// task 134: decode (1 トークン) と prefill (1 チャンク) で同じ関数を使う。
/// `len` が decode なら積んだ本数、prefill ならチャンク終端 (`pos0 + n`)。
#[allow(clippy::too_many_arguments)]
fn q4_attn_mirror_sync(
    m: &mut Q4AttnMirror,
    layer: usize,
    len: usize,
    gen: u64,
    kv_dim: usize,
    k: &[f32],
    v: &[f32],
    pf: bool,
) -> Result<(CUdeviceptr, CUdeviceptr)> {
    let new_cap = q4_mirror_cap(len, m.state.cap);
    if new_cap > m.state.cap {
        q4_attn_grow(m, new_cap, kv_dim)?;
    }
    if m.k[layer].is_none() || m.v[layer].is_none() {
        let bytes = q4_mirror_alloc_rows(m.state.cap) * kv_dim * 4;
        m.k[layer] = Some(DevBuffer::alloc_tagged(
            bytes,
            crate::mem_audit::TAG_ATTN_MIRROR,
        )?);
        m.v[layer] = Some(DevBuffer::alloc_tagged(
            bytes,
            crate::mem_audit::TAG_ATTN_MIRROR,
        )?);
        // task 174: 片方でも作り直したら、その層は中身を失っている。台帳を
        // 0 にして次の `have` で全行 H2D させる (古い行数を信じない)。
        m.state.len[layer] = 0;
    }
    // task 174: カーネルへ渡す行数が実バッファに収まっていることを毎回確かめる
    // (OOB 読みを「illegal address」ではなく明示エラーにする)。確保は
    // grow-only なので普段は 1 回の比較で終わる。
    if len > m.state.cap {
        bail!(
            "q4 attn mirror: len {len} が cap {} を超えた (層 {layer}, kv_dim {kv_dim})",
            m.state.cap
        );
    }
    let need_bytes = q4_mirror_alloc_rows(len)
        .checked_mul(kv_dim)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| anyhow!("q4 attn mirror: len*kv_dim が溢れた ({len} x {kv_dim})"))?;
    for (slot_name, slot) in [("k", &m.k[layer]), ("v", &m.v[layer])] {
        match slot.as_ref() {
            Some(b) if b.bytes() >= need_bytes => {}
            Some(b) => bail!(
                "q4 attn mirror: {slot_name} のバッファ {} B < 必要 {need_bytes} B \
                 (層 {layer}, len {len}, cap {})",
                b.bytes(),
                m.state.cap
            ),
            None => bail!("q4 attn mirror: {slot_name} のバッファが無い (層 {layer})"),
        }
    }
    let have = q4_mirror_have(&m.state, layer, len, gen);
    if have < len {
        let off = have * kv_dim;
        let end = len * kv_dim;
        let t = q4attn_kprof::stamp();
        let kb = m.k[layer].as_ref().expect("直前に確保した");
        let vb = m.v[layer].as_ref().expect("直前に確保した");
        kb.copy_from_at(off * 4, &k[off..end])?;
        vb.copy_from_at(off * 4, &v[off..end])?;
        if pf {
            q4attn_kprof::note_pf_h2d(t, (end - off) * 4 * 2);
        } else {
            q4attn_kprof::note_h2d(t, have == 0, (end - off) * 4 * 2);
        }
    }
    q4_mirror_note(&mut m.state, layer, len, gen);
    Ok((
        m.k[layer].as_ref().expect("確保済み").ptr(),
        m.v[layer].as_ref().expect("確保済み").ptr(),
    ))
}


/// ミラーのエントリ (鍵 = `AttnCache::id()`) を引き、層の行 `[0..len)` を
/// 同期して K/V のデバイスポインタを返す。予算超過分は LRU で落とす。
///
/// 失敗したらそのエントリごと捨てる (中途半端なミラーを持たない = 次回は
/// 全行シードから)。task 133 の decode と task 134 の prefill の共通経路。
/// `pf` は計測の振り分けだけ (数値・転送内容は同じ)。
#[allow(clippy::too_many_arguments)]
pub(crate) fn q4_attn_mirror_prepare(
    cache_id: u64,
    n_layer: usize,
    kv_dim: usize,
    layer: usize,
    gen: u64,
    len: usize,
    k: &[f32],
    v: &[f32],
    pf: bool,
) -> Result<(CUdeviceptr, CUdeviceptr)> {
    // task 179: ミラーの帳簿 (mutex / LRU / 環境変数) を名前付きで測る。
    let t_book = dec_kprof::on().then(std::time::Instant::now);
    let mirrors = Q4_ATTN_MIRRORS.get_or_init(|| std::sync::Mutex::new(Vec::new()));
    let mut ms = match mirrors.lock() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    };
    let idx = q4_attn_mirror_entry(&mut ms, cache_id, n_layer, kv_dim);
    if let Some(t) = t_book {
        dec_kprof::note_since("mirror_book", t);
    }
    match q4_attn_mirror_sync(&mut ms[idx].1, layer, len, gen, kv_dim, k, v, pf) {
        Ok(p) => Ok(p),
        Err(e) => {
            ms.retain(|(k, _, _)| *k != cache_id);
            Err(e)
        }
    }
}


/// ミラーのエントリ (鍵 = `AttnCache::id()`) を引き、無ければ作り、
/// 予算超過分を LRU で落とす。`idx` を返す。
///
/// task 133 の decode と task 134 / 181 の prefill の共通部分。
fn q4_attn_mirror_entry(
    ms: &mut Vec<(u64, Q4AttnMirror, std::time::Instant)>,
    cache_id: u64,
    n_layer: usize,
    kv_dim: usize,
) -> usize {
    let now = std::time::Instant::now();
    // 予算超過分を LRU で落とす (自分は落とさない)。
    let mut meta: Vec<MirrorMeta> = ms
        .iter()
        .map(|(k, m, t)| MirrorMeta {
            key: *k,
            bytes: m.bytes(),
            last: *t,
            pending: false,
        })
        .collect();
    let evicted = lru_evict_keys(&mut meta, q4_attn_mirror_limit_bytes(), cache_id);
    if !evicted.is_empty() {
        ms.retain(|(k, _, _)| !evicted.contains(k));
    }
    let idx = match ms.iter().position(|(k, _, _)| *k == cache_id) {
        Some(i) => {
            ms[i].2 = now;
            i
        }
        None => {
            ms.push((cache_id, Q4AttnMirror::new(n_layer, kv_dim), now));
            ms.len() - 1
        }
    };
    if ms[idx].1.kv_dim != kv_dim || ms[idx].1.state.len.len() != n_layer {
        ms[idx].1 = Q4AttnMirror::new(n_layer, kv_dim);
    }
    idx
}


/// task 181: チャンクの k / v (**デバイスに既にある**) をミラーの行
/// `[pos0, t_total)` へ直接書く (D2D)。
///
/// 従来は `pf_full_qkv` が k / v を D2H し、ホストの `AttnCache` へ積み、
/// `pf_full_out` が `q4_attn_mirror_prepare` で **同じ行を H2D し直して**いた。
/// 行 `[0, pos0)` がミラーに正しく載っていれば、残りの行はデバイス間コピーで
/// 足りる (H2D 0 B)。載っていなければ `Err` を返し、呼び出し側は従来経路
/// (= 差分 H2D) へ戻る。
///
/// 台帳 (`Q4MirrorState`) の扱いは [`q4_attn_mirror_sync`] と同じ:
/// 世代が変わった層・縮んだ層は「持っていない」と見なす (task 133)。
#[allow(clippy::too_many_arguments)]
pub(crate) fn q4_attn_mirror_stage_dev(
    cache_id: u64,
    n_layer: usize,
    kv_dim: usize,
    layer: usize,
    gen: u64,
    pos0: usize,
    t_total: usize,
    kdev: CUdeviceptr,
    vdev: CUdeviceptr,
) -> Result<(CUdeviceptr, CUdeviceptr)> {
    if t_total < pos0 {
        bail!("q4 attn mirror stage: pos0 {pos0} が t_total {t_total} を超える");
    }
    let mirrors = Q4_ATTN_MIRRORS.get_or_init(|| std::sync::Mutex::new(Vec::new()));
    let mut ms = match mirrors.lock() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    };
    let idx = q4_attn_mirror_entry(&mut ms, cache_id, n_layer, kv_dim);
    let r = (|| -> Result<()> {
        let m = &mut ms[idx].1;
        let new_cap = q4_mirror_cap(t_total, m.state.cap);
        if new_cap > m.state.cap {
            // 成長は旧バッファの有効行を D2D で温存する (task 174 の余白込み)。
            q4_attn_grow(m, new_cap, kv_dim)?;
        }
        if m.k[layer].is_none() || m.v[layer].is_none() {
            let bytes = q4_mirror_alloc_rows(m.state.cap) * kv_dim * 4;
            m.k[layer] = Some(DevBuffer::alloc_tagged(
                bytes,
                crate::mem_audit::TAG_ATTN_MIRROR,
            )?);
            m.v[layer] = Some(DevBuffer::alloc_tagged(
                bytes,
                crate::mem_audit::TAG_ATTN_MIRROR,
            )?);
            m.state.len[layer] = 0;
        }
        if t_total > m.state.cap {
            bail!(
                "q4 attn mirror stage: len {t_total} が cap {} を超えた (層 {layer})",
                m.state.cap
            );
        }
        // 行 [0, pos0) がこの世代の内容で載っているか (載っていなければ H2D 経路へ)。
        let have = q4_mirror_have(&m.state, layer, pos0, gen);
        if have != pos0 {
            bail!("q4 attn mirror stage: 行 [0,{pos0}) が無い (have={have}, gen={gen})");
        }
        let nb = t_total - pos0;
        if nb > 0 {
            let need_bytes = t_total * kv_dim * 4;
            for (slot_name, slot) in [("k", &m.k[layer]), ("v", &m.v[layer])] {
                match slot.as_ref() {
                    Some(b) if b.bytes() >= need_bytes => {}
                    _ => bail!("q4 attn mirror stage: {slot_name} のバッファが足りない"),
                }
            }
            let kb = m.k[layer].as_ref().expect("直前に確認した");
            let vb = m.v[layer].as_ref().expect("直前に確認した");
            kb.copy_from_dev_at(pos0 * kv_dim * 4, kdev, nb * kv_dim)?;
            vb.copy_from_dev_at(pos0 * kv_dim * 4, vdev, nb * kv_dim)?;
        }
        q4_mirror_note(&mut m.state, layer, t_total, gen);
        Ok(())
    })();
    match r {
        Ok(()) => Ok((
            ms[idx].1.k[layer]
                .as_ref()
                .ok_or_else(|| anyhow!("q4 attn mirror stage: k が無い"))?
                .ptr(),
            ms[idx].1.v[layer]
                .as_ref()
                .ok_or_else(|| anyhow!("q4 attn mirror stage: v が無い"))?
                .ptr(),
        )),
        Err(e) => {
            // 中途半端なミラーを持たない (次回は全行シードから)。
            ms.retain(|(k, _, _)| *k != cache_id);
            Err(e)
        }
    }
}


/// task 181: `q4_attn_mirror_stage_dev` が既に置いたミラーのポインタを引く
/// (**コピーもしない**)。台帳が `(layer, gen, t_total)` と一致するときだけ
/// `Some`。呼び出し側はこれが `None` のとき従来の `prepare` を使う。
pub(crate) fn q4_attn_mirror_ready(
    cache_id: u64,
    kv_dim: usize,
    layer: usize,
    gen: u64,
    t_total: usize,
) -> Option<(CUdeviceptr, CUdeviceptr)> {
    let mirrors = Q4_ATTN_MIRRORS.get()?;
    let mut ms = match mirrors.lock() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    };
    let now = std::time::Instant::now();
    let idx = ms.iter().position(|(k, _, _)| *k == cache_id)?;
    ms[idx].2 = now;
    let m = &ms[idx].1;
    if m.kv_dim != kv_dim || layer >= m.state.len.len() {
        return None;
    }
    if m.state.len[layer] != t_total || m.state.gen[layer] != gen {
        return None;
    }
    let kb = m.k[layer].as_ref()?;
    let vb = m.v[layer].as_ref()?;
    if kb.bytes() < t_total * kv_dim * 4 || vb.bytes() < t_total * kv_dim * 4 {
        return None;
    }
    Some((kb.ptr(), vb.ptr()))
}


pub(crate) struct Q4AttnScratch {
    q: Option<DevBuffer>,
    sc: Option<DevBuffer>,
    out: Option<DevBuffer>,
    /// task 135: t 分割 decode の部分和 `[heads, n_split, dh + 2]`
    part: Option<DevBuffer>,
}


thread_local! {
    pub(crate) static Q4ATTN_SCRATCH: std::cell::RefCell<Q4AttnScratch> = std::cell::RefCell::new(Q4AttnScratch {
        q: None,
        sc: None,
        out: None,
        part: None,
    });
}


/// qwen4exp full attention の decode 1 トークンぶんの入力 (CPU 側で rope 済み)。
pub struct Q4AttnCall<'a> {
    /// `AttnCache::id()` (ミラーの鍵)
    pub cache_id: u64,
    /// キャッシュの層数
    pub n_layer: usize,
    /// この層
    pub layer: usize,
    /// `AttnCache::gen(layer)` (内容世代)
    pub layer_gen: u64,
    /// 積まれている位置数 (= K/V の行数)
    pub len: usize,
    /// 1 行の長さ = n_head_kv * head_dim
    pub kv_dim: usize,
    /// q head 数
    pub heads: usize,
    /// q head / kv head
    pub group: usize,
    /// head_dim
    pub dh: usize,
    /// qk scale (softmax 前の乗数)
    pub scale: f32,
    /// rope 済み q `[heads * dh]`
    pub q: &'a [f32],
    /// 層の K 行 `[len * kv_dim]`
    pub k: &'a [f32],
    /// 層の V 行 `[len * kv_dim]`
    pub v: &'a [f32],
}


/// qwen4exp full attention (decode) を GPU で回す。
///
/// 戻り `Some(out)` = `[heads * dh]` の attention 出力 (CPU とは丸め順が違う)。
/// `None` = この呼び出しでは GPU を使わない (呼び出し側は CPU 経路へ)。
///
/// task 135: どのカーネルを使うかは `DENNO_Q4_GPU_ATTN_KERNEL` で選ぶ
/// (`decode` = 既存 `dno_attention` / `pf` = `dno_attention_prefill` を n=1 で /
/// `split` = t 分割の `dno_attention_decode`)。既定は
/// [`Q4_GPU_ATTN_KERNEL_DEFAULT`]。
pub fn q4_attn_decode(a: &Q4AttnCall<'_>) -> Option<Vec<f32>> {
    if !cuda_enabled() || !gpu_attention_enabled() || !q4_gpu_attn_enabled() {
        return None;
    }
    if a.len < q4_gpu_attn_min() {
        return None;
    }
    // task 160: flash decoding (split-K・GQA まとめ読み)。既定は
    // `ATTN_FLASHDEC_DEFAULT` (lab の結果で決める)。短い文脈は従来カーネルの方が
    // 速いので `DENNO_ATTN_FLASHDEC_MIN_CTX` で切る。PTX 不在・形状不一致は
    // 静かに従来経路へ (数値の契約は同じ緩和枠)。
    if attn_flashdec_enabled() && a.len >= attn_flashdec_min_ctx() {
        // task 448: タイル版 (K/V を shared へ協調ロード) を先に試す。門・形・
        // PTX のどれかが欠けたら v1 へ落ちる (v1 と 1 ビットも変わらない)。
        let tile_on = attn_fdec_tile_on();
        fdt_gate_log(tile_on);
        if tile_on {
            if let Some(v) = q4_attn_decode_fdec(a, FdecImpl::Tile) {
                return Some(v);
            }
            q4attn_kprof::note_fdt_fallback();
        }
        match q4_attn_decode_flashdec(a) {
            Some(v) => return Some(v),
            None => q4attn_kprof::note_fdec_fallback(),
        }
    }
    match q4_gpu_attn_kernel() {
        Q4AttnKernel::Decode => q4_attn_decode_single(a),
        Q4AttnKernel::Prefill => q4_attn_decode_via_prefill(a),
        // 新カーネルが PTX に無い (未再生成)・形状が合わないときは従来の
        // `dno_attention` で回す。数値の契約は同じ緩和枠なので、経路が変わっても
        // 「静かに遅いだけ」で済む。落ちた回数は [q4attn] の split fallback で見る。
        // task 448: `tile` / `fdec1` は flashdec の実装選択なので、flashdec の門を
        // 外れたとき (短い文脈) は split と同じ扱いにする。
        Q4AttnKernel::Split | Q4AttnKernel::Tile | Q4AttnKernel::Fdec1 => {
            match q4_attn_decode_split(a) {
                Some(v) => Some(v),
                None => {
                    q4attn_kprof::note_split_fallback();
                    q4_attn_decode_single(a)
                }
            }
        }
    }
}


/// 既存 `dno_attention` (1 q head = 1 block・dno_expf) の decode 経路。
fn q4_attn_decode_single(a: &Q4AttnCall<'_>) -> Option<Vec<f32>> {
    // dno_attention の制約 (kernels_elem.cu)。外れたら CPU へ静かに落ちる。
    // blockDim = dh の木縮約は 2 の冪でしか正しくない。
    if a.heads == 0
        || a.group == 0
        || a.heads % a.group != 0
        || a.dh == 0
        || a.dh > 1024
        || !a.dh.is_power_of_two()
        || a.len == 0
        || a.layer >= a.n_layer
    {
        return None;
    }
    let n_kv = a.heads / a.group;
    if a.kv_dim != n_kv * a.dh || a.q.len() != a.heads * a.dh {
        return None;
    }
    let rows = a.len.checked_mul(a.kv_dim)?;
    if a.k.len() < rows || a.v.len() < rows {
        return None;
    }
    let ctx = chain_ctx()?;

    // ミラー同期 (この層の K/V を len 行まで)。失敗したらエントリごと捨てて CPU。
    let (kptr, vptr) = q4ok(q4_attn_mirror_prepare(
        a.cache_id,
        a.n_layer,
        a.kv_dim,
        a.layer,
        a.layer_gen,
        a.len,
        a.k,
        a.v,
        false,
    ))?;

    // カーネル起動 (スクラッチはスレッドローカル grow-only)。
    let qd = a.heads * a.dh;
    let mut kv_dim_c = a.kv_dim as c_int;
    let mut t_c = a.len as c_int;
    let mut group_c = a.group as c_int;
    let mut dh_c = a.dh as c_int;
    let mut scale_c = a.scale;
    let out = Q4ATTN_SCRATCH.with(|cell| -> Option<Vec<f32>> {
        let mut s = cell.borrow_mut();
        let qbytes = qd.checked_mul(4)?;
        let scbytes = a.heads.checked_mul(a.len)?.checked_mul(4)?;
        q4ok(Scratch::ensure(&mut s.q, qbytes))?;
        q4ok(Scratch::ensure(&mut s.sc, scbytes))?;
        q4ok(Scratch::ensure(&mut s.out, qbytes))?;
        let th = q4attn_kprof::stamp();
        q4ok(s.q.as_ref().expect("ensure 済み").copy_from(a.q))?;
        q4attn_kprof::note_q(th, qbytes);
        let mut qptr = s.q.as_ref().expect("ensure 済み").ptr();
        let mut scptr = s.sc.as_ref().expect("ensure 済み").ptr();
        let mut aptr = s.out.as_ref().expect("ensure 済み").ptr();
        let mut kptr_c = kptr;
        let mut vptr_c = vptr;
        let mut args: Vec<*mut c_void> = vec![
            &mut qptr as *mut _ as *mut c_void,
            &mut kptr_c as *mut _ as *mut c_void,
            &mut vptr_c as *mut _ as *mut c_void,
            &mut aptr as *mut _ as *mut c_void,
            &mut scptr as *mut _ as *mut c_void,
            &mut kv_dim_c as *mut _ as *mut c_void,
            &mut t_c as *mut _ as *mut c_void,
            &mut group_c as *mut _ as *mut c_void,
            &mut dh_c as *mut _ as *mut c_void,
            &mut scale_c as *mut _ as *mut c_void,
        ];
        let tl = q4attn_kprof::stamp();
        q4ok(unsafe {
            ctx.launch_async(
                ctx.f_attn,
                (a.heads as u32, 1, 1),
                (a.dh as u32, 1, 1),
                (a.dh * 4) as u32,
                &mut args,
            )
        })?;
        q4attn_kprof::note_launch(tl);
        let tw = q4attn_kprof::stamp();
        // task 368: 直後に読むのは null ストリームのカーネルが書いた `out` だけなので、
        // 待つ範囲を null ストリームに狭めてよい (`DENNO_DEC_SYNC_STREAM`、既定 off)。
        q4ok(ctx.sync_decode())?;
        q4attn_kprof::note_wait(tw);
        let td = q4attn_kprof::stamp();
        let outv = q4ok(s.out.as_ref().expect("ensure 済み").to_vec::<f32>(qd))?;
        q4attn_kprof::note_d2h(td, qbytes);
        q4attn_kprof::note_call();
        Some(outv)
    })?;
    Some(out)
}


/// task 135 step 3: decode 1 トークンを t 分割カーネル (`dno_attention_decode`
/// + `dno_attention_decode_combine`) で回す。
///
/// 既存 `dno_attention` は 1 q head = 1 block なので 24 block しか立たず、
/// t=14k では 1 SM あたりの帯域が上限になる。ここでは t を
/// `DENNO_Q4_GPU_ATTN_SPLIT` (既定 8) 個に固定分割し、heads x 分割数の block で
/// 全 SM を使う。分割は `t_total` と分割数だけで決まる固定分割・縮約順は昇順・
/// 原子操作なしなので**決定論** (CPU とのビット一致はしない: attention の緩和枠)。
///
/// 2 発目 (combine) は部分和 (m, l, acc) を split 昇順に畳むだけの小さな
/// カーネル。部分和バッファは VRAM 上に置いたまま (D2H しない)。
fn q4_attn_decode_split(a: &Q4AttnCall<'_>) -> Option<Vec<f32>> {
    if a.heads == 0
        || a.group == 0
        || a.heads % a.group != 0
        || a.dh == 0
        // blockDim = dh で 1 warp 以上が要る (warp が key を分担するため
        // dh < 32 だと nwarp = 0 になり分担ループが回らない)。実運用は dh=256。
        || a.dh < 32
        || a.dh > 1024
        || a.len == 0
        || a.len > 65535
        || a.layer >= a.n_layer
    {
        return None;
    }
    let n_kv = a.heads / a.group;
    if a.kv_dim != n_kv * a.dh || a.q.len() != a.heads * a.dh {
        return None;
    }
    let rows = a.len.checked_mul(a.kv_dim)?;
    if a.k.len() < rows || a.v.len() < rows {
        return None;
    }
    let ctx = chain_ctx()?;
    let (f_dec, f_cb) = (ctx.f_attn_dec?, ctx.f_attn_dec_cb?);
    let (n_split, _chunk) = q4_dec_split(a.len, q4_gpu_attn_split());

    // ミラー同期は decode / prefill と同一の台帳 (1 行追記)。
    let (kptr, vptr) = q4ok(q4_attn_mirror_prepare(
        a.cache_id,
        a.n_layer,
        a.kv_dim,
        a.layer,
        a.layer_gen,
        a.len,
        a.k,
        a.v,
        false,
    ))?;

    let qd = a.heads * a.dh;
    let mut kv_dim_c = a.kv_dim as c_int;
    let mut t_c = a.len as c_int;
    let mut group_c = a.group as c_int;
    let mut dh_c = a.dh as c_int;
    let mut scale_c = a.scale;
    let mut nsplit_c = n_split as c_int;
    let shmem = (q4_dec_sh_f(a.dh) * 4) as u32;
    let out = Q4ATTN_SCRATCH.with(|cell| -> Option<Vec<f32>> {
        let mut s = cell.borrow_mut();
        let qbytes = qd.checked_mul(4)?;
        let part_bytes = a
            .heads
            .checked_mul(n_split)?
            .checked_mul(a.dh.checked_add(2)?)?
            .checked_mul(4)?;
        q4ok(Scratch::ensure(&mut s.q, qbytes))?;
        q4ok(Scratch::ensure(&mut s.out, qbytes))?;
        q4ok(Scratch::ensure(&mut s.part, part_bytes))?;
        let th = q4attn_kprof::stamp();
        q4ok(s.q.as_ref().expect("ensure 済み").copy_from(a.q))?;
        q4attn_kprof::note_q(th, qbytes);
        let mut qptr = s.q.as_ref().expect("ensure 済み").ptr();
        let mut aptr = s.out.as_ref().expect("ensure 済み").ptr();
        let mut partptr = s.part.as_ref().expect("ensure 済み").ptr();
        let mut kptr_c = kptr;
        let mut vptr_c = vptr;
        let mut args: Vec<*mut c_void> = vec![
            &mut qptr as *mut _ as *mut c_void,
            &mut kptr_c as *mut _ as *mut c_void,
            &mut vptr_c as *mut _ as *mut c_void,
            &mut partptr as *mut _ as *mut c_void,
            &mut kv_dim_c as *mut _ as *mut c_void,
            &mut t_c as *mut _ as *mut c_void,
            &mut group_c as *mut _ as *mut c_void,
            &mut dh_c as *mut _ as *mut c_void,
            &mut scale_c as *mut _ as *mut c_void,
            &mut nsplit_c as *mut _ as *mut c_void,
        ];
        let tl = q4attn_kprof::stamp();
        q4ok(unsafe {
            ctx.launch_async(
                f_dec,
                // split が最速次元 (同じ split の head 群が同時に走り K/V を L2 共有)
                (n_split as u32, a.heads as u32, 1),
                (a.dh as u32, 1, 1),
                shmem,
                &mut args,
            )
        })?;
        let mut args2: Vec<*mut c_void> = vec![
            &mut partptr as *mut _ as *mut c_void,
            &mut aptr as *mut _ as *mut c_void,
            &mut nsplit_c as *mut _ as *mut c_void,
            &mut dh_c as *mut _ as *mut c_void,
        ];
        q4ok(unsafe {
            ctx.launch_async(
                f_cb,
                (a.heads as u32, 1, 1),
                (a.dh as u32, 1, 1),
                0,
                &mut args2,
            )
        })?;
        q4attn_kprof::note_launch(tl);
        let tw = q4attn_kprof::stamp();
        // task 368: 直後に読むのは null ストリームのカーネルが書いた `out` だけなので、
        // 待つ範囲を null ストリームに狭めてよい (`DENNO_DEC_SYNC_STREAM`、既定 off)。
        q4ok(ctx.sync_decode())?;
        q4attn_kprof::note_wait(tw);
        let td = q4attn_kprof::stamp();
        let outv = q4ok(s.out.as_ref().expect("ensure 済み").to_vec::<f32>(qd))?;
        q4attn_kprof::note_d2h(td, qbytes);
        q4attn_kprof::note_call();
        q4attn_kprof::note_split_call();
        Some(outv)
    })?;
    Some(out)
}


/// task 160: decode 1 トークンを flash decoding (`dno_attn_flashdec` +
/// `dno_attn_flashdec_combine`) で回す。
///
/// 既存 `dno_attention_decode` との違いは「誰が同じ KV 行を読むか」だけ:
/// 既存は 1 block = (split, q head) で GQA の group 本の q head が同じ行を
/// group 回読み、block も 4 warp しか無い。ここは 1 block = (split, kv head)・
/// warp = q head 1 本で、行は block 内の L1 に 1 回だけ DRAM から来る。
/// 分割は (t_total, n_split) だけで決まる固定分割・縮約は分割昇順なので
/// **決定論** (CPU とはビット一致しない: attention の緩和枠)。
///
/// 同期は既存 split 経路と同じで層あたり 1 回 (`ctx.sync()`。出力をホストが
/// 読むため)。カーネル 2 発は同じストリームに積むだけ。
fn q4_attn_decode_flashdec(a: &Q4AttnCall<'_>) -> Option<Vec<f32>> {
    q4_attn_decode_fdec(a, FdecImpl::V1)
}


/// [`FdecImpl`] を選んで回す本体。分割 (`q4_fdec_split`)・部分和の置き場
/// (`part`)・combine はどちらの実装でも共通で、**K/V をどう読むかだけ**が違う。
fn q4_attn_decode_fdec(a: &Q4AttnCall<'_>, imp: FdecImpl) -> Option<Vec<f32>> {
    if a.heads == 0 || a.group == 0 || a.heads % a.group != 0 || a.group > 32 {
        return None;
    }
    // 1 lane が float4 (4 要素) ずつ持ち、1 巡で 128 要素。dh は 4 の倍数だけ。
    if a.dh < 4 || a.dh % 4 != 0 || a.dh > ATTN_FDEC_MAXB * 128 {
        return None;
    }
    if a.len == 0 || a.len > 65535 || a.layer >= a.n_layer {
        return None;
    }
    let n_kv = a.heads / a.group;
    if a.kv_dim != n_kv * a.dh || a.q.len() != a.heads * a.dh {
        return None;
    }
    let rows = a.len.checked_mul(a.kv_dim)?;
    if a.k.len() < rows || a.v.len() < rows {
        return None;
    }
    let ctx = chain_ctx()?;
    // task 448: タイル版は tile 行と smem の門を通ったときだけ。落ちたら None を
    // 返し、呼び出し側 ([`q4_attn_decode`]) が v1 で回す。
    let tile = attn_fdec_tile();
    // task 520: タイル版の経路では第 3 版 (2 key 同時) を先に試す。
    let mut used_t2 = false;
    let (f_dec, f_cb, shmem, tile_c) = match imp {
        FdecImpl::V1 => {
            let k = fdec_kernels()?;
            (k.f_dec, k.f_cb, 0u32, 0 as c_int)
        }
        FdecImpl::Tile => {
            if !fdt_shape_ok(a.dh, tile) {
                return None;
            }
            // combine は v1 のモジュールのものを使う (同じ分割・同じ part 形式)。
            let kv1 = fdec_kernels()?;
            match if attn_fdec_ilp_on() { fdt2_kernels() } else { None } {
                Some(k2) => {
                    used_t2 = true;
                    (
                        k2.f_t2,
                        kv1.f_cb,
                        fdt_smem_bytes(tile, a.dh) as u32,
                        tile as c_int,
                    )
                }
                None => {
                    let kt = fdt_kernels()?;
                    (
                        kt.f_t,
                        kv1.f_cb,
                        fdt_smem_bytes(tile, a.dh) as u32,
                        tile as c_int,
                    )
                }
            }
        }
    };
    let (n_split, chunk) = q4_fdec_split(a.len, attn_flashdec_chunk_for(a.len));
    match imp {
        FdecImpl::V1 => fdec_log(a.len, n_split, chunk, a.kv_dim, a.dh, a.group),
        FdecImpl::Tile => {
            if used_t2 {
                fdt2_log(a.len, n_split, tile);
            } else {
                fdt_log(a.len, n_split, chunk, a.dh, tile)
            }
        }
    }

    // task 520 (2): 第 3 版だけ 12 本目 (grid 入れ替え) を持つ。
    let khx = used_t2 && attn_fdec_khx_on();
    // ミラー同期は decode / prefill / split と同一の台帳 (1 行追記)。
    let (kptr, vptr) = q4ok(q4_attn_mirror_prepare(
        a.cache_id,
        a.n_layer,
        a.kv_dim,
        a.layer,
        a.layer_gen,
        a.len,
        a.k,
        a.v,
        false,
    ))?;

    let qd = a.heads * a.dh;
    let mut kv_dim_c = a.kv_dim as c_int;
    let mut t_c = a.len as c_int;
    let mut group_c = a.group as c_int;
    let mut dh_c = a.dh as c_int;
    let mut scale_c = a.scale;
    let mut nsplit_c = n_split as c_int;
    let nkv_c = n_kv as c_int;
    let out = Q4ATTN_SCRATCH.with(|cell| -> Option<Vec<f32>> {
        let mut s = cell.borrow_mut();
        let qbytes = qd.checked_mul(4)?;
        let part_bytes = a
            .heads
            .checked_mul(n_split)?
            .checked_mul(a.dh.checked_add(ATTN_FDEC_PART_EXTRA)?)?
            .checked_mul(4)?;
        q4ok(Scratch::ensure(&mut s.q, qbytes))?;
        q4ok(Scratch::ensure(&mut s.out, qbytes))?;
        q4ok(Scratch::ensure(&mut s.part, part_bytes))?;
        let th = q4attn_kprof::stamp();
        q4ok(s.q.as_ref().expect("ensure 済み").copy_from(a.q))?;
        q4attn_kprof::note_q(th, qbytes);
        let mut qptr = s.q.as_ref().expect("ensure 済み").ptr();
        let mut aptr = s.out.as_ref().expect("ensure 済み").ptr();
        let mut partptr = s.part.as_ref().expect("ensure 済み").ptr();
        let mut kptr_c = kptr;
        let mut vptr_c = vptr;
        let mut args: Vec<*mut c_void> = vec![
            &mut qptr as *mut _ as *mut c_void,
            &mut kptr_c as *mut _ as *mut c_void,
            &mut vptr_c as *mut _ as *mut c_void,
            &mut partptr as *mut _ as *mut c_void,
            &mut kv_dim_c as *mut _ as *mut c_void,
            &mut t_c as *mut _ as *mut c_void,
            &mut group_c as *mut _ as *mut c_void,
            &mut dh_c as *mut _ as *mut c_void,
            &mut scale_c as *mut _ as *mut c_void,
            &mut nsplit_c as *mut _ as *mut c_void,
        ];
        // task 448: タイル版だけ 11 本目 (tile 行数) を渡す。
        let mut tile_c = tile_c;
        let mut khx_c = khx as c_int;
        if imp == FdecImpl::Tile {
            args.push(&mut tile_c as *mut _ as *mut c_void);
        }
        // task 520 (2): 第 3 版だけ 12 本目 (grid 入れ替え) を渡す。
        if used_t2 {
            args.push(&mut khx_c as *mut _ as *mut c_void);
        }
        let tl = q4attn_kprof::stamp();
        // task 520 (2): khx のときは (kh, sp) の順に並べる。
        let grid = if khx {
            (n_kv as u32, n_split as u32, 1)
        } else {
            (n_split as u32, n_kv as u32, 1)
        };
        q4ok(unsafe {
            ctx.launch_async(
                f_dec,
                grid,
                // 1 warp = 1 q head
                ((32 * a.group) as u32, 1, 1),
                shmem,
                &mut args,
            )
        })?;
        let mut args2: Vec<*mut c_void> = vec![
            &mut partptr as *mut _ as *mut c_void,
            &mut aptr as *mut _ as *mut c_void,
            &mut group_c as *mut _ as *mut c_void,
            &mut dh_c as *mut _ as *mut c_void,
            &mut nsplit_c as *mut _ as *mut c_void,
        ];
        q4ok(unsafe {
            ctx.launch_async(
                f_cb,
                (a.group as u32, nkv_c as u32, 1),
                (a.dh as u32, 1, 1),
                0,
                &mut args2,
            )
        })?;
        q4attn_kprof::note_launch(tl);
        let tw = q4attn_kprof::stamp();
        // task 368: 直後に読むのは null ストリームのカーネルが書いた `out` だけなので、
        // 待つ範囲を null ストリームに狭めてよい (`DENNO_DEC_SYNC_STREAM`、既定 off)。
        q4ok(ctx.sync_decode())?;
        q4attn_kprof::note_wait(tw);
        let td = q4attn_kprof::stamp();
        let outv = q4ok(s.out.as_ref().expect("ensure 済み").to_vec::<f32>(qd))?;
        q4attn_kprof::note_d2h(td, qbytes);
        q4attn_kprof::note_call();
        match imp {
            FdecImpl::V1 => q4attn_kprof::note_fdec_call(),
            FdecImpl::Tile => {
                if used_t2 {
                    q4attn_kprof::note_fdt2_call();
                }
                q4attn_kprof::note_fdt_call();
            }
        }
        Some(outv)
    })?;
    Some(out)
}


/// qwen4exp full attention の prefill 1 チャンクぶんの入力。
///
/// q は CPU 側で QK-norm + rope 済み、K/V はチャンクを push 済みのホスト
/// キャッシュ (層 `layer` の `[0, t_total)`)。
pub struct Q4AttnPfCall<'a> {
    /// `AttnCache::id()` (ミラーの鍵。decode と同じ台帳を共有する)
    pub cache_id: u64,
    /// キャッシュの層数
    pub n_layer: usize,
    /// この層
    pub layer: usize,
    /// `AttnCache::gen(layer)` (内容世代)
    pub layer_gen: u64,
    /// チャンク終端の位置数 (= pos0 + n)。ミラーはここまで同期する
    pub t_total: usize,
    /// チャンク先頭の位置 (最初のクエリ行の絶対位置)
    pub pos0: usize,
    /// チャンクのトークン数 (クエリ行数)
    pub n: usize,
    /// 1 行の長さ = n_head_kv * head_dim
    pub kv_dim: usize,
    /// q head 数
    pub heads: usize,
    /// q head / kv head
    pub group: usize,
    /// head_dim
    pub dh: usize,
    /// qk scale (softmax 前の乗数)
    pub scale: f32,
    /// rope 済み q `[n * heads * dh]` (トークン順・head 順)
    pub q: &'a [f32],
    /// 層の K 行 `[t_total * kv_dim]`
    pub k: &'a [f32],
    /// 層の V 行 `[t_total * kv_dim]`
    pub v: &'a [f32],
}


/// qwen4exp full attention (prefill の 1 チャンク) を GPU で回す。
///
/// `dno_attention_prefill` (flash 風 online softmax・固定タイル) を 1 発打ち、
/// `[n, heads*dh]` を返す。因果性はカーネル内で `pos0` を使って閉じる
/// (クエリ行 `i` は `0..=pos0+i`)。マスクされたキーはスコアが -inf で
/// exp2 が厳密に 0 になるため、**チャンクの切り方に依存しない** (行チャンク不変)。
///
/// 戻り `Some(out)` = GPU の出力 (CPU とは丸め順が違う。決定論は維持)。
/// `None` = この呼び出しでは GPU を使わない (呼び出し側は CPU 経路へ)。
pub fn q4_attn_prefill(a: &Q4AttnPfCall<'_>) -> Option<Vec<f32>> {
    if !cuda_enabled() || !gpu_attention_enabled() {
        return None;
    }
    // 発動ゲート (env + 2 段のしきい値) と shape ゲート。行数の下限は attention
    // 専用の ATTN_Q4_PF_MIN_ROWS を使う (GEMM の gemm_min_m() は借りない)。
    if !q4_attn_pf_eligible(a.n, a.t_total) {
        return None;
    }
    if !attn_prefill_gpu_supported_min(a.n, a.dh, a.t_total, ATTN_Q4_PF_MIN_ROWS) {
        return None;
    }
    q4_attn_prefill_launch(a, false, ATTN_Q4_PF_MIN_ROWS)
}


/// decode 1 トークンを `dno_attention_prefill` で回す (task 135 step 1)。
///
/// n=1・pos0=t_total-1 で同じカーネルに入れる。`dno_attention` との違いは exp が
/// exp2f であることだけで、行方向は PF_BQ=64 行のうち 63 行が無効なので
/// block 数は heads のまま = t 分割の効果は無い (分割は step 3 の
/// `dno_attention_decode`)。ゲートは decode のもの (`DENNO_Q4_GPU_ATTN*`) を使う。
///
/// 実測 (lab t135a): 3,954 文脈で 1 呼び出し 2,269 us = 従来 `dno_attention` の
/// 586 us の 3.9 倍。**この経路は遅い** (64 行の無効行を計算する)。既定にはしない。
fn q4_attn_decode_via_prefill(a: &Q4AttnCall<'_>) -> Option<Vec<f32>> {
    if a.dh < ATTN_PF_BQ || a.dh > 1024 || a.len == 0 || a.len > 65535 {
        return None;
    }
    if a.heads == 0 || a.group == 0 || a.heads % a.group != 0 || a.layer >= a.n_layer {
        return None;
    }
    let n_kv = a.heads / a.group;
    if a.kv_dim != n_kv * a.dh || a.q.len() != a.heads * a.dh {
        return None;
    }
    let rows = a.len.checked_mul(a.kv_dim)?;
    if a.k.len() < rows || a.v.len() < rows {
        return None;
    }
    let call = Q4AttnPfCall {
        cache_id: a.cache_id,
        n_layer: a.n_layer,
        layer: a.layer,
        layer_gen: a.layer_gen,
        t_total: a.len,
        pos0: a.len - 1,
        n: 1,
        kv_dim: a.kv_dim,
        heads: a.heads,
        group: a.group,
        dh: a.dh,
        scale: a.scale,
        q: a.q,
        k: a.k,
        v: a.v,
    };
    q4_attn_prefill_launch(&call, true, 1)
}


/// [`q4_attn_prefill`] / [`q4_attn_decode_via_prefill`] の実体 (ゲート済み)。
///
/// `dec` は計測の振り分けだけ (数値・転送内容・起動形状は同じ)。`min_rows` は
/// 形状ゲートの行数の下限で、decode は 1 行から通す。
fn q4_attn_prefill_launch(
    a: &Q4AttnPfCall<'_>,
    dec: bool,
    min_rows: usize,
) -> Option<Vec<f32>> {
    if a.pos0.checked_add(a.n)? != a.t_total || a.layer >= a.n_layer {
        return None;
    }
    if !attn_prefill_shape_ok(a.n, a.dh, a.t_total, min_rows) {
        return None;
    }
    // カーネルの int 引数 (t_total は 65,535 までに抑える: qwen35 と同じ上限)。
    if a.t_total > 65535 || a.heads == 0 || a.group == 0 || a.heads % a.group != 0 {
        return None;
    }
    let n_kv = a.heads / a.group;
    if a.kv_dim != n_kv * a.dh {
        return None;
    }
    let qd = a.heads.checked_mul(a.dh)?;
    if a.q.len() != a.n.checked_mul(qd)? {
        return None;
    }
    let rows = a.t_total.checked_mul(a.kv_dim)?;
    if a.k.len() < rows || a.v.len() < rows {
        return None;
    }
    let ctx = chain_ctx()?;

    // ミラー同期 (decode と同一の台帳・再同期規則)。チャンクの追記だけを送る。
    let (kptr, vptr) = q4ok(q4_attn_mirror_prepare(
        a.cache_id,
        a.n_layer,
        a.kv_dim,
        a.layer,
        a.layer_gen,
        a.t_total,
        a.k,
        a.v,
        !dec,
    ))?;

    let mut heads_c = a.heads as c_int;
    let mut n_c = a.n as c_int;
    let mut pos0_c = a.pos0 as c_int;
    let mut kv_dim_c = a.kv_dim as c_int;
    let mut t_c = a.t_total as c_int;
    let mut group_c = a.group as c_int;
    let mut dh_c = a.dh as c_int;
    let mut scale_c = a.scale;
    let shmem = (attn_pf_sh_f(a.dh) * 4) as u32;
    let out = Q4ATTN_SCRATCH.with(|cell| -> Option<Vec<f32>> {
        let mut s = cell.borrow_mut();
        let qbytes = a.n.checked_mul(qd)?.checked_mul(4)?;
        q4ok(Scratch::ensure(&mut s.q, qbytes))?;
        q4ok(Scratch::ensure(&mut s.out, qbytes))?;
        let th = q4attn_kprof::stamp();
        q4ok(s.q.as_ref().expect("ensure 済み").copy_from(a.q))?;
        if dec {
            q4attn_kprof::note_q(th, qbytes);
        } else {
            q4attn_kprof::note_pf_q(th, qbytes);
        }
        let mut qptr = s.q.as_ref().expect("ensure 済み").ptr();
        let mut aptr = s.out.as_ref().expect("ensure 済み").ptr();
        let mut kptr_c = kptr;
        let mut vptr_c = vptr;
        let mut args: Vec<*mut c_void> = vec![
            &mut qptr as *mut _ as *mut c_void,
            &mut kptr_c as *mut _ as *mut c_void,
            &mut vptr_c as *mut _ as *mut c_void,
            &mut aptr as *mut _ as *mut c_void,
            &mut heads_c as *mut _ as *mut c_void,
            &mut n_c as *mut _ as *mut c_void,
            &mut pos0_c as *mut _ as *mut c_void,
            &mut kv_dim_c as *mut _ as *mut c_void,
            &mut t_c as *mut _ as *mut c_void,
            &mut group_c as *mut _ as *mut c_void,
            &mut dh_c as *mut _ as *mut c_void,
            &mut scale_c as *mut _ as *mut c_void,
        ];
        let tl = q4attn_kprof::stamp();
        q4ok(unsafe {
            ctx.launch_async(
                ctx.f_attn_pf,
                // qwen35 prefill と同一の形状: (行ブロック, head)。
                (a.n.div_ceil(ATTN_PF_BQ) as u32, a.heads as u32, 1),
                (a.dh as u32, 1, 1),
                shmem,
                &mut args,
            )
        })?;
        if dec {
            q4attn_kprof::note_launch(tl);
        } else {
            q4attn_kprof::note_pf_launch(tl);
        }
        let tw = q4attn_kprof::stamp();
        q4ok(ctx.sync())?;
        if dec {
            q4attn_kprof::note_wait(tw);
        } else {
            q4attn_kprof::note_pf_wait(tw);
        }
        let td = q4attn_kprof::stamp();
        let outv = q4ok(s.out.as_ref().expect("ensure 済み").to_vec::<f32>(a.n * qd))?;
        if dec {
            q4attn_kprof::note_d2h(td, qbytes);
            q4attn_kprof::note_call();
        } else {
            q4attn_kprof::note_pf_d2h(td, qbytes);
            q4attn_kprof::note_pf_call();
        }
        Some(outv)
    })?;
    Some(out)
}


/// task 133: GPU full attention の内訳 (DENNO_Q4_PROFILE / DENNO_PF_ATT_DBG のときだけ
/// 積む。1 リクエストぶんを `line()` で 1 行に出し、`reset()` で戻す)。
pub mod q4attn_kprof {
    use std::sync::atomic::{AtomicU64, Ordering};

    /// カーネルを回した回数 (= layer × decode token)
    pub static CALLS: AtomicU64 = AtomicU64::new(0);
    /// 遅延シードで送ったバイト数 (prefill 履歴の初回 H2D)
    pub static SEED_BYTES: AtomicU64 = AtomicU64::new(0);
    /// 1 トークンぶんの追記で送ったバイト数
    pub static ROW_BYTES: AtomicU64 = AtomicU64::new(0);
    /// q の H2D バイト数
    pub static Q_BYTES: AtomicU64 = AtomicU64::new(0);
    /// 出力の D2H バイト数
    pub static OUT_BYTES: AtomicU64 = AtomicU64::new(0);
    /// K/V 追記 H2D の累計時間
    pub static H2D_NS: AtomicU64 = AtomicU64::new(0);
    /// q の H2D 累計時間
    pub static Q_NS: AtomicU64 = AtomicU64::new(0);
    /// launch 発行の累計時間
    pub static LAUNCH_NS: AtomicU64 = AtomicU64::new(0);
    /// launch 後の同期 (カーネル待ち) 累計時間
    pub static WAIT_NS: AtomicU64 = AtomicU64::new(0);
    /// 出力 D2H の累計時間
    pub static D2H_NS: AtomicU64 = AtomicU64::new(0);

    #[inline]
    pub(crate) fn on() -> bool {
        crate::model::qwen4exp::q4prof::timed()
    }

    /// 計測が要るときだけ `Instant` を取る (既定 off では `None`)。
    #[inline]
    pub fn stamp() -> Option<std::time::Instant> {
        if on() {
            Some(std::time::Instant::now())
        } else {
            None
        }
    }

    #[inline]
    pub(crate) fn add_ns(t: Option<std::time::Instant>, c: &AtomicU64) {
        if let Some(t) = t {
            c.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    }

    pub(crate) fn note_h2d(t: Option<std::time::Instant>, seed: bool, bytes: usize) {
        if !on() {
            return;
        }
        let b = bytes as u64;
        if seed {
            SEED_BYTES.fetch_add(b, Ordering::Relaxed);
        } else {
            ROW_BYTES.fetch_add(b, Ordering::Relaxed);
        }
        add_ns(t, &H2D_NS);
    }

    pub(crate) fn note_q(t: Option<std::time::Instant>, bytes: usize) {
        if !on() {
            return;
        }
        Q_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
        add_ns(t, &Q_NS);
    }

    pub(crate) fn note_launch(t: Option<std::time::Instant>) {
        if on() {
            add_ns(t, &LAUNCH_NS);
        }
    }

    pub(crate) fn note_wait(t: Option<std::time::Instant>) {
        if on() {
            add_ns(t, &WAIT_NS);
        }
    }

    pub(crate) fn note_d2h(t: Option<std::time::Instant>, bytes: usize) {
        if !on() {
            return;
        }
        OUT_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
        add_ns(t, &D2H_NS);
    }

    pub(crate) fn note_call() {
        if on() {
            CALLS.fetch_add(1, Ordering::Relaxed);
        }
    }

    // ---- task 134: prefill (1 チャンク = 1 層 1 呼び出し) -------------------

    /// prefill でカーネルを回した回数 (= layer x chunk)
    pub static PF_CALLS: AtomicU64 = AtomicU64::new(0);
    /// prefill で K/V ミラーへ送ったバイト数 (チャンクの追記)
    pub static PF_H2D_BYTES: AtomicU64 = AtomicU64::new(0);
    /// prefill の q の H2D バイト数
    pub static PF_Q_BYTES: AtomicU64 = AtomicU64::new(0);
    /// prefill の出力 D2H バイト数
    pub static PF_OUT_BYTES: AtomicU64 = AtomicU64::new(0);
    /// prefill の K/V 追記 H2D の累計時間
    pub static PF_H2D_NS: AtomicU64 = AtomicU64::new(0);
    /// prefill の q の H2D 累計時間
    pub static PF_Q_NS: AtomicU64 = AtomicU64::new(0);
    /// prefill の launch 発行の累計時間
    pub static PF_LAUNCH_NS: AtomicU64 = AtomicU64::new(0);
    /// prefill のカーネル待ちの累計時間
    pub static PF_WAIT_NS: AtomicU64 = AtomicU64::new(0);
    /// prefill の出力 D2H の累計時間
    pub static PF_D2H_NS: AtomicU64 = AtomicU64::new(0);

    // ---- task 135: t 分割 decode のカウンタ --------------------------------
    /// `dno_attention_decode` を回した回数 (= layer x decode token)
    pub static SPLIT_CALLS: AtomicU64 = AtomicU64::new(0);
    /// split を選んだが新カーネルが使えず従来 `dno_attention` へ落ちた回数。
    /// PTX 未再生成・形状不一致の検出用 (この値が 0 でない run の decode は
    /// split の効果を含まない)。
    pub static SPLIT_FALLBACKS: AtomicU64 = AtomicU64::new(0);

    pub(crate) fn note_split_call() {
        if on() {
            SPLIT_CALLS.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(crate) fn note_split_fallback() {
        if on() {
            SPLIT_FALLBACKS.fetch_add(1, Ordering::Relaxed);
        }
    }

    // ---- task 160: flash decoding のカウンタ --------------------------------
    /// `dno_attn_flashdec` を回した回数 (= layer x decode token)
    pub static FDEC_CALLS: AtomicU64 = AtomicU64::new(0);
    /// flash decoding を選んだが PTX / 形状の都合で従来カーネルへ落ちた回数
    /// (この値が 0 でない run の decode は flash decoding の効果を含まない)。
    pub static FDEC_FALLBACKS: AtomicU64 = AtomicU64::new(0);

    pub(crate) fn note_fdec_call() {
        if on() {
            FDEC_CALLS.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(crate) fn note_fdec_fallback() {
        if on() {
            FDEC_FALLBACKS.fetch_add(1, Ordering::Relaxed);
        }
    }

    // ---- task 448: flash decoding 第 2 版 (shared タイル) のカウンタ ----------
    /// `dno_attn_flashdec_t` を回した回数 (= layer x decode token)
    pub static FDT_CALLS: AtomicU64 = AtomicU64::new(0);
    /// タイル版を選んだが門 (形 / PTX / ABI) で落ちて v1 へ落ちた回数。
    /// この値が 0 でない run の decode はタイル版の効果を含まない。
    pub static FDT_FALLBACKS: AtomicU64 = AtomicU64::new(0);

    pub(crate) fn note_fdt_call() {
        if on() {
            FDT_CALLS.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(crate) fn note_fdt_fallback() {
        if on() {
            FDT_FALLBACKS.fetch_add(1, Ordering::Relaxed);
        }
    }

    // ---- task 520: flash decoding 第 3 版 (2 key 同時) のカウンタ --------------
    /// `dno_attn_flashdec_t2` を回した回数 (= layer x decode token)。
    /// 0 のままなら第 3 版は選ばれていない (ゲートが off か PTX/形で落ちた)。
    pub static FDT2_CALLS: AtomicU64 = AtomicU64::new(0);

    pub(crate) fn note_fdt2_call() {
        if on() {
            FDT2_CALLS.fetch_add(1, Ordering::Relaxed);
        }
    }

    // ---- task 460: combine + ゲートの融合版を回した回数 ----------------------
    /// `dno_attn_flashdec_cbg` を回した回数 (= layer x decode token)。
    /// 0 のままなら融合版は選ばれていない (既定 off)。
    pub static CBG_CALLS: AtomicU64 = AtomicU64::new(0);

    pub(crate) fn note_cbg_call() {
        if on() {
            CBG_CALLS.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(crate) fn note_pf_h2d(t: Option<std::time::Instant>, bytes: usize) {
        if !on() {
            return;
        }
        PF_H2D_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
        add_ns(t, &PF_H2D_NS);
    }

    pub(crate) fn note_pf_q(t: Option<std::time::Instant>, bytes: usize) {
        if !on() {
            return;
        }
        PF_Q_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
        add_ns(t, &PF_Q_NS);
    }

    pub(crate) fn note_pf_launch(t: Option<std::time::Instant>) {
        if on() {
            add_ns(t, &PF_LAUNCH_NS);
        }
    }

    pub(crate) fn note_pf_wait(t: Option<std::time::Instant>) {
        if on() {
            add_ns(t, &PF_WAIT_NS);
        }
    }

    pub(crate) fn note_pf_d2h(t: Option<std::time::Instant>, bytes: usize) {
        if !on() {
            return;
        }
        PF_OUT_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
        add_ns(t, &PF_D2H_NS);
    }

    pub(crate) fn note_pf_call() {
        if on() {
            PF_CALLS.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// prefill ぶんの累計を 0 に戻す。
    pub fn reset_pf() {
        for c in [
            &PF_CALLS,
            &PF_H2D_BYTES,
            &PF_Q_BYTES,
            &PF_OUT_BYTES,
            &PF_H2D_NS,
            &PF_LAUNCH_NS,
            &PF_WAIT_NS,
            &PF_D2H_NS,
        ] {
            c.store(0, Ordering::Relaxed);
        }
    }

    /// 1 リクエストぶんの 1 行 (呼び出しが 0 なら `None`)。
    pub fn line() -> Option<String> {
        let calls = CALLS.load(Ordering::Relaxed);
        let pf_calls = PF_CALLS.load(Ordering::Relaxed);
        if calls == 0 && pf_calls == 0 {
            return None;
        }
        let mb = |b: u64| b as f64 / (1024.0 * 1024.0);
        let ms = |ns: u64| ns as f64 / 1e6;
        let us = |ns: u64| ns as f64 / 1e3 / calls.max(1) as f64;
        let (entries, used, limit) = super::q4_attn_mirror_stats().unwrap_or((0, 0, 0));
        Some(format!(
            "[q4attn] calls={calls} h2d seed={:.1}MB row={:.1}MB q={:.1}MB | out={:.1}MB | \
             h2d={:.1}ms q={:.1}ms launch={:.1}ms wait={:.1}ms d2h={:.1}ms | \
             per-call h2d={:.1}us launch={:.1}us wait={:.1}us d2h={:.1}us | \
             mirror={entries} entries {:.0}MB / {}MB | \
             pf calls={pf_calls} h2d={:.1}MB q={:.1}MB out={:.1}MB | \
             pf h2d={:.1}ms launch={:.1}ms wait={:.1}ms d2h={:.1}ms | \
             split calls={} fallback={} | fdec calls={} fallback={} | \
             fdt calls={} fallback={} tile={} | cbg calls={} | t2 calls={}",
            mb(SEED_BYTES.load(Ordering::Relaxed)),
            mb(ROW_BYTES.load(Ordering::Relaxed)),
            mb(Q_BYTES.load(Ordering::Relaxed)),
            mb(OUT_BYTES.load(Ordering::Relaxed)),
            ms(H2D_NS.load(Ordering::Relaxed)),
            ms(Q_NS.load(Ordering::Relaxed)),
            ms(LAUNCH_NS.load(Ordering::Relaxed)),
            ms(WAIT_NS.load(Ordering::Relaxed)),
            ms(D2H_NS.load(Ordering::Relaxed)),
            us(H2D_NS.load(Ordering::Relaxed)),
            us(LAUNCH_NS.load(Ordering::Relaxed)),
            us(WAIT_NS.load(Ordering::Relaxed)),
            us(D2H_NS.load(Ordering::Relaxed)),
            mb(used as u64),
            limit / (1024 * 1024),
            mb(PF_H2D_BYTES.load(Ordering::Relaxed)),
            mb(PF_Q_BYTES.load(Ordering::Relaxed)),
            mb(PF_OUT_BYTES.load(Ordering::Relaxed)),
            ms(PF_H2D_NS.load(Ordering::Relaxed)),
            ms(PF_LAUNCH_NS.load(Ordering::Relaxed)),
            ms(PF_WAIT_NS.load(Ordering::Relaxed)),
            ms(PF_D2H_NS.load(Ordering::Relaxed)),
            SPLIT_CALLS.load(Ordering::Relaxed),
            SPLIT_FALLBACKS.load(Ordering::Relaxed),
            FDEC_CALLS.load(Ordering::Relaxed),
            FDEC_FALLBACKS.load(Ordering::Relaxed),
            FDT_CALLS.load(Ordering::Relaxed),
            FDT_FALLBACKS.load(Ordering::Relaxed),
            super::attn_fdec_tile(),
            CBG_CALLS.load(Ordering::Relaxed),
            FDT2_CALLS.load(Ordering::Relaxed),
        ))
    }

    /// 累計を 0 に戻す。
    pub fn reset() {
        for c in [
            &CALLS,
            &SEED_BYTES,
            &ROW_BYTES,
            &Q_BYTES,
            &OUT_BYTES,
            &H2D_NS,
            &Q_NS,
            &LAUNCH_NS,
            &WAIT_NS,
            &D2H_NS,
            &SPLIT_CALLS,
            &SPLIT_FALLBACKS,
            &FDEC_CALLS,
            &FDEC_FALLBACKS,
            &FDT_CALLS,
            &FDT_FALLBACKS,
            &CBG_CALLS,
            &FDT2_CALLS,
        ] {
            c.store(0, Ordering::Relaxed);
        }
        // task 134: prefill の累計も同じリクエスト境界で戻す。
        reset_pf();
    }
}


/// s176 (P20): full 層 attention の q ステージング (pinned) を **prepare の出力先**
/// として貸し出す。`f` に渡すスライスに書いた内容を、直後の
/// [`chain_attention_prefill_pinned`] がそのまま H2D する。
///
/// 従来は prepare がホスト `Vec` (`buf_q`) に書き、`chain_attention_prefill` が
/// それを pinned へ `copy_from_slice` してから H2D していた (27B n=14,043 で
/// 345 MB/層・このコンテナの実測 21.1 ms/層 = 0.34 s/request)。ここでは書き込み先を
/// pinned そのものにするので、**転送バイト数・カーネル・発行順は不変**で
/// ホスト memcpy 1 回だけが消える (値は 1 ビットも変わらない)。
///
/// 安全条件 (呼び出し側が守る): 次に同じ pin を上書きするのは次の層の prepare で、
/// その前に次層の qkv chain の `sync` が入る (この H2D は既に完了している)。
pub fn chain_pf_q_pin_fill<F>(n_elems: usize, f: F) -> Option<()>
where
    F: FnOnce(&mut [f32]),
{
    let _ = chain_ctx()?;
    let bytes = n_elems.checked_mul(4)?;
    if n_elems == 0 {
        return None;
    }
    CHAIN.with(|cell| -> Option<()> {
        let mut s = cell.borrow_mut();
        ok(Scratch::ensure_pin(&mut s.pin_pf, bytes))?;
        let pin = s.pin_pf.as_mut()?;
        f(&mut pin.as_mut_slice::<f32>()[..n_elems]);
        Some(())
    })
}


/// P17j: prefill (n > 1) 用 GPU attention (dno_attention_prefill・flash 風
/// online softmax)。q ([n, heads*dh]、rope/QK-Norm 加算済み) とホスト KvCache
/// (現在レイヤの行 [pos0..t_total) は呼び出し側が書き込み済み) を受け取り、
/// ミラーを同期して prefill attention カーネルを起動し、結果 [n, heads*dh] を
/// CH_ATT に書く (呼び出し側が chain_download する)。
/// 戻り None = CPU attention にフォールバック。prefill は q/KV が全てホスト側に
/// あるため decode chain と異なり **attention ブロック内で完結**する
/// (forward 全体の再実行は不要・ホスト状態は不変)。
/// 数値は CPU (sgemm 経路) とビット不一致 — exactness 契約の attention 緩和
/// (ユーザー承認・P17g) と同一枠。**決定論は維持** (カーネル内の固定タイル・
/// 固定累積順・原子操作なし)。
#[allow(clippy::too_many_arguments)]
pub fn chain_attention_prefill(
    key: u64,
    li: usize,
    n_layers: usize,
    q: &[f32],
    cache: &crate::kv_cache::KvCache,
    pos0: usize,
    n: usize,
    t_total: usize,
    group: usize,
    dh: usize,
    kv_dim: usize,
    scale: f32,
) -> Option<()> {
    if n == 0 || dh == 0 {
        return None;
    }
    let heads = q.len() / (n * dh);
    if heads == 0 || n.checked_mul(dh).map(|x| x * heads) != Some(q.len()) {
        return None;
    }
    chain_attention_prefill_impl(
        key,
        li,
        n_layers,
        Some(q),
        heads,
        cache,
        pos0,
        n,
        t_total,
        group,
        dh,
        kv_dim,
        scale,
    )
}


/// s176 (P20): q が **pinned ステージングに既に書かれている**前提の起動
/// ([`chain_pf_q_pin_fill`] の直後に呼ぶ)。カーネル・grid・block・引数・発行順は
/// [`chain_attention_prefill`] と同一で、消えるのは「ホスト q → pinned」の
/// memcpy 1 回だけ。
#[allow(clippy::too_many_arguments)]
pub fn chain_attention_prefill_pinned(
    key: u64,
    li: usize,
    n_layers: usize,
    heads: usize,
    cache: &crate::kv_cache::KvCache,
    pos0: usize,
    n: usize,
    t_total: usize,
    group: usize,
    dh: usize,
    kv_dim: usize,
    scale: f32,
) -> Option<()> {
    chain_attention_prefill_impl(
        key, li, n_layers, None, heads, cache, pos0, n, t_total, group, dh, kv_dim, scale,
    )
}


#[allow(clippy::too_many_arguments)]
fn chain_attention_prefill_impl(
    key: u64,
    li: usize,
    n_layers: usize,
    q: Option<&[f32]>,
    heads: usize,
    cache: &crate::kv_cache::KvCache,
    pos0: usize,
    n: usize,
    t_total: usize,
    group: usize,
    dh: usize,
    kv_dim: usize,
    scale: f32,
) -> Option<()> {
    if !gpu_attention_enabled() {
        return None;
    }
    if n == 0 || t_total == 0 || group == 0 || t_total > cache.max_len() {
        return None;
    }
    // カーネル制約: block = dh スレッドが PF_BQ 行 owner + 出力次元 d=tid を
    // 担当するため dh >= PF_BQ。動的 smem は 48KB 既定上限を opt-in 済み上限
    // (s48 C1-a: qwen35 dh=256 = 84.8KB は cuFuncSetAttribute で通す) に引き上げ。
    if dh < ATTN_PF_BQ || dh > 1024 {
        return None;
    }
    if attn_pf_sh_f(dh) * 4 > attn_pf_dynamic_smem_limit() {
        return None;
    }
    // q の要素数 (Some/None どちらでも同じ — ステージングの大きさを決める)
    let q_elems = n.checked_mul(heads)?.checked_mul(dh)?;
    if let Some(q) = q {
        if q.len() != q_elems {
            return None;
        }
    }
    let ctx = chain_ctx()?;

    let mut heads_c = heads as c_int;
    let mut n_c = n as c_int;
    let mut pos0_c = pos0 as c_int;
    let mut kv_dim_c = kv_dim as c_int;
    let mut t_c = t_total as c_int;
    let mut group_c = group as c_int;
    let mut dh_c = dh as c_int;
    let mut scale_c = scale;

    ok(chain_ensure_buf(CH_ATT, n * heads * dh * 4))?;

    // ミラーの取得/生成 + レイヤ li の行同期 (decode と同一経路)。
    let (mut kptr, mut vptr) =
        with_mirror(key, li, n_layers, t_total, kv_dim, cache.epoch(), |m, have| {
        let (kc_ref, vc_ref) = cache.kv_bufs_ref(li);
        if have < t_total {
            let kb = m.layers[li].0.as_ref().unwrap();
            let vb = m.layers[li].1.as_ref().unwrap();
            ok(kb.copy_from_at(have * kv_dim * 4, &kc_ref[have * kv_dim..t_total * kv_dim]))?;
            ok(vb.copy_from_at(have * kv_dim * 4, &vc_ref[have * kv_dim..t_total * kv_dim]))?;
        }
        m.len[li] = t_total;
        Some((
            m.layers[li].0.as_ref().unwrap().ptr(),
            m.layers[li].1.as_ref().unwrap().ptr(),
        ))
    })?;

    CHAIN.with(|cell| -> Option<()> {
        let mut s = cell.borrow_mut();
        ok(Scratch::ensure(&mut s.attn_pf_q, q_elems * 4))?;
        // P17k: pinned ステージング + async H2D (同期 pageable copy_from より高速)。
        // DMA 完了は同一ストリーム順序でカーネルが待つため sync 不要。pin_pf の
        // 次回上書きは次層の attention で、その前に oproj/download の sync が挟まる。
        ok(Scratch::ensure_pin(&mut s.pin_pf, q_elems * 4))?;
        let buf = s.attn_pf_q.as_ref().unwrap().ptr();
        // s176 (P20): q が None = 呼び出し側 (`chain_pf_q_pin_fill`) が既にこの
        // pinned へ書いている。その場合はコピーしない (345 MB/層の memcpy が消える)。
        if let Some(q) = q {
            s.pin_pf.as_mut().unwrap().as_mut_slice::<f32>()[..q_elems].copy_from_slice(q);
        }
        let lib = ok(lib())?;
        unsafe {
            ok(check(
                (lib.memcpy_htod_async)(buf, s.pin_pf.as_ref().unwrap().ptr as *const c_void, q_elems * 4, std::ptr::null_mut()),
                "cuMemcpyHtoDAsync (attn_pf_q)",
            ))?;
        }
        // s180 (P41): attention カーネルの発行直前 = full_core の区間 [prep] の
        // 終端 (ホスト prepare + KV ミラー / q の H2D までが prep)。イベントのみ。
        fc_kprof_mark(1);
        let mut qptr = s.attn_pf_q.as_ref().unwrap().ptr();
        let mut aptr = s.bufs[CH_ATT].as_ref().unwrap().ptr();
        let mut args: Vec<*mut c_void> = vec![
            &mut qptr as *mut _ as *mut c_void,
            &mut kptr as *mut _ as *mut c_void,
            &mut vptr as *mut _ as *mut c_void,
            &mut aptr as *mut _ as *mut c_void,
            &mut heads_c as *mut _ as *mut c_void,
            &mut n_c as *mut _ as *mut c_void,
            &mut pos0_c as *mut _ as *mut c_void,
            &mut kv_dim_c as *mut _ as *mut c_void,
            &mut t_c as *mut _ as *mut c_void,
            &mut group_c as *mut _ as *mut c_void,
            &mut dh_c as *mut _ as *mut c_void,
            &mut scale_c as *mut _ as *mut c_void,
        ];
        ok(unsafe {
            ctx.launch_async(
                ctx.f_attn_pf,
                // P17k: (行ブロック, heads) — x=行ブロック最速次元で同一 head の
                // K/V を同時実行ブロック間で共有し L2 ヒット率を上げる
                (n.div_ceil(ATTN_PF_BQ) as u32, heads as u32, 1),
                (dh as u32, 1, 1),
                (attn_pf_sh_f(dh) * 4) as u32,
                &mut args,
            )
        })?;
        // s180 (P41): 発行直後 = full_core の区間 [attn] の終端 (カーネルの完了は
        // 次の既存 D2H の sync まで待たない — イベントはストリーム順に乗るだけ)。
        fc_kprof_mark(2);
        Some(())
    })
}


/// P17g: バッチ decode 用 — セッションごとに `chain_attention` を起動し、
/// 結果をホスト `att` [b, heads*dh] に集める。solo chain と同一カーネル・
/// 同一入力 → 単独/バッチでビット一致を維持 (exactness 契約 P6)。
/// 全セッション成功時 true。1 つでも失敗すれば false を返し、呼び出し側は
/// 全セッションを CPU attention でやり直す (GPU/CPU 混合の数値混在を防止)。
#[allow(clippy::too_many_arguments)]
pub fn chain_attention_batch(
    li: usize,
    n_layers: usize,
    q: &[f32],
    caches: &[&crate::kv_cache::KvCache],
    heads: usize,
    group: usize,
    dh: usize,
    kv_dim: usize,
    scale: f32,
    att: &mut [f32],
) -> bool {
    let hd = heads * dh;
    if !gpu_attention_enabled()
        || caches.is_empty()
        || q.len() != caches.len() * hd
        || att.len() != q.len()
    {
        return false;
    }
    for (i, cache) in caches.iter().enumerate() {
        let t_total = cache.len() + 1;
        let ok1 = chain_attention(
            cache.kv_id(),
            li,
            n_layers,
            &q[i * hd..][..hd],
            cache,
            t_total,
            group,
            dh,
            kv_dim,
            scale,
        );
        if ok1.is_none() {
            return false;
        }
        match chain_download(CH_ATT, hd) {
            Some(row) => att[i * hd..][..hd].copy_from_slice(&row),
            None => return false,
        }
    }
    true
}


// ===========================================================================
// P17h: decode chain の同期削減 (rope GPU 化 + qkv デバイス常駐)
// ===========================================================================

/// P17h: rope テーブル [cos(half)|sin(half)] をデバイスへ準備 (1 ステップ 1 回)。
/// CPU tensor::rope_row と**同一の式**を同一プロセスの glibc sinf/cosf で計算する
/// ためテーブル値はビット一致。デバイスバッファはステップ中ずっと保持される。
pub fn chain_rope_prepare(pos0: usize, dh: usize, theta: f32) -> Option<()> {
    let _ = chain_ctx()?;
    let half = dh / 2;
    let bytes = half * 2 * 4;
    CHAIN.with(|cell| -> Option<()> {
        let mut s = cell.borrow_mut();
        ok(Scratch::ensure(&mut s.rope_tab, bytes))?;
        ok(Scratch::ensure_pin(&mut s.pin_rope, bytes))?;
        {
            let tab = s.pin_rope.as_mut().unwrap().as_mut_slice::<f32>();
            let pos = pos0 as f32;
            for d in 0..half {
                let ang = pos * theta.powf(-2.0 * d as f32 / dh as f32);
                tab[d] = ang.cos();
                tab[half + d] = ang.sin();
            }
        }
        let dev = s.rope_tab.as_ref().unwrap().ptr();
        let pin = s.pin_rope.as_ref().unwrap();
        let lib = ok(lib())?;
        unsafe {
            ok(check(
                (lib.memcpy_htod_async)(dev, pin.ptr as *const c_void, bytes, std::ptr::null_mut()),
                "cuMemcpyHtoDAsync (rope tab)",
            ))
        }
    })
}


/// P17h: CH_QKV の q/k/v に bias 加算 + rope を適用 (デバイス内・in-place)。
/// CPU 経路 (add_bias → rope_row) とビット一致 (bias は単一 add・rope は
/// mul/sub のみ・-fmad=false)。bias 無しの要素は NULL として渡す。
/// `pin_id` は bias アップロード用 pinned スロット (層ごとに一意)。
pub fn chain_rope_qkv(
    q_b: Option<&[f32]>,
    k_b: Option<&[f32]>,
    v_b: Option<&[f32]>,
    heads: usize,
    kv_heads: usize,
    dh: usize,
    pin_id: usize,
) -> Option<()> {
    let ctx = chain_ctx()?;
    let hd = heads.checked_mul(dh)?;
    let kv_dim = kv_heads.checked_mul(dh)?;
    let bias_len = hd + 2 * kv_dim;
    CHAIN.with(|cell| -> Option<()> {
        let mut s = cell.borrow_mut();
        ok(Scratch::ensure(&mut s.rope_bias, bias_len * 4))?;
        let bptr = if q_b.is_some() || k_b.is_some() || v_b.is_some() {
            if s.pins.len() <= pin_id {
                while s.pins.len() <= pin_id {
                    s.pins.push(None);
                }
            }
            ok(Scratch::ensure_pin(&mut s.pins[pin_id], bias_len * 4))?;
            {
                let pin = s.pins[pin_id].as_mut().unwrap();
                let p = pin.as_mut_slice::<f32>();
                p[..bias_len].fill(0.0);
                if let Some(b) = q_b {
                    p[..b.len()].copy_from_slice(b);
                }
                if let Some(b) = k_b {
                    p[hd..hd + b.len()].copy_from_slice(b);
                }
                if let Some(b) = v_b {
                    p[hd + kv_dim..hd + kv_dim + b.len()].copy_from_slice(b);
                }
            }
            let bp = s.rope_bias.as_ref().unwrap().ptr();
            let pp = s.pins[pin_id].as_ref().unwrap();
            let lib = ok(lib())?;
            unsafe {
                ok(check(
                    (lib.memcpy_htod_async)(
                        bp,
                        pp.ptr as *const c_void,
                        bias_len * 4,
                        std::ptr::null_mut(),
                    ),
                    "cuMemcpyHtoDAsync (rope bias)",
                ))?;
            }
            bp
        } else {
            0 as CUdeviceptr // NULL — bias 無し
        };
        let qkv_ptr = s.bufs[CH_QKV].as_ref().unwrap().ptr();
        let tab_ptr = s.rope_tab.as_ref().unwrap().ptr();
        let (mut pq, mut pt, mut pb) = (qkv_ptr, tab_ptr, bptr);
        let (mut hc, mut kc, mut dc) =
            (heads as c_int, kv_heads as c_int, dh as c_int);
        let mut args: Vec<*mut c_void> = vec![
            &mut pq as *mut _ as *mut c_void,
            &mut pt as *mut _ as *mut c_void,
            &mut pb as *mut _ as *mut c_void,
            &mut hc as *mut _ as *mut c_void,
            &mut kc as *mut _ as *mut c_void,
            &mut dc as *mut _ as *mut c_void,
        ];
        ok(unsafe {
            ctx.launch_async(
                ctx.f_rope,
                ((heads + 2 * kv_heads) as u32, 1, 1),
                (dh as u32, 1, 1),
                0,
                &mut args,
            )
        })
    })
}


// ===========================================================================
// B3: b>1 バッチ decode の rope GPU 化 (P17 系 rope カーネルの再利用)
// ===========================================================================

/// B3: b>1 バッチ decode の rope を GPU で実行する (q/k in-place・セッションごと
/// に独立した位置)。CPU 経路 (forward_batch の session ごと `tensor::rope_row`
/// 適用) と**ビット一致** (exactness 契約: rope は緩和対象外):
/// - テーブルは CPU が rope_row と同一式 (同一プロセスの glibc powf/cosf/sinf)
///   で生成 → テーブル値はビット一致
/// - カーネル `dno_rope_qk_batch` は mul/sub のみ (-fmad=false → mul.rn/sub.rn/
///   add.rn = Rust スカラー演算と同一の丸め)
/// - bias/QK-Norm は呼び出し側が既に CPU で適用済みの q/k を渡す (rope は
///   bias/QK-Norm と無関係・CPU 経路と同一の演算列)
/// 失敗時は false を返しホスト q/k は不変 (呼び出し側が CPU rope でやり直せる —
/// D2H 完了後に初めてホストへ書き戻す)。テーブルは (poss・dh・theta) 完全一致
/// でキャッシュ (1 ステップの層ループで同一 poss が層数回来るため H2D もスキップ)。
#[allow(clippy::too_many_arguments)]
pub fn batch_rope_qk(
    q: &mut [f32],
    k: &mut [f32],
    poss: &[usize],
    heads: usize,
    kv_heads: usize,
    dh: usize,
    theta: f32,
) -> bool {
    let Some(ctx) = chain_ctx() else { return false };
    let b = poss.len();
    // カーネル/転送の形状ガード (grid/block 上限・size_t オーバーフロー防止)。
    // 非該当 shapes は静かに CPU フォールバック (GPU_BROKEN ラッチも不要)。
    let Some(hd) = heads.checked_mul(dh) else { return false };
    let Some(kv_dim) = kv_heads.checked_mul(dh) else { return false };
    let Some(q_floats) = b.checked_mul(hd) else { return false };
    let Some(k_floats) = b.checked_mul(kv_dim) else { return false };
    if b == 0 || b > 65535 || dh == 0 || dh > 1024 || heads + kv_heads > 65535 {
        return false;
    }
    let q_bytes = q_floats * 4;
    let k_bytes = k_floats * 4;
    let tab_floats = b * dh;
    let tab_bytes = tab_floats * 4;
    CHAIN.with(|cell| -> bool {
        let mut s = cell.borrow_mut();
        // --- テーブル (キャッシュヒットなら生成 + H2D をスキップ) ---
        let cache_hit = s.br_tab_valid
            && s.br_tab_dh == dh
            && s.br_tab_theta.to_bits() == theta.to_bits()
            && s.br_tab_key.len() == b
            && s.br_tab_key.iter().zip(poss.iter()).all(|(a, p)| a == p);
        if !cache_hit {
            if ok(Scratch::ensure(&mut s.br_tab, tab_bytes)).is_none()
                || ok(Scratch::ensure_pin(&mut s.br_pin_tab, tab_bytes)).is_none()
            {
                return false;
            }
            {
                // tensor::rope_row と同一式で生成 (同一プロセスの glibc
                // powf/cosf/sinf → 値はビット一致・P17h chain_rope_prepare と同じ)
                let tab = s.br_pin_tab.as_mut().unwrap().as_mut_slice::<f32>();
                let half = dh / 2;
                for (i, &pos) in poss.iter().enumerate() {
                    let posf = pos as f32;
                    let row = &mut tab[i * dh..][..dh];
                    for d in 0..half {
                        let ang = posf * theta.powf(-2.0 * d as f32 / dh as f32);
                        row[d] = ang.cos();
                        row[half + d] = ang.sin();
                    }
                }
            }
            let lib = match ok(lib()) {
                Some(l) => l,
                None => return false,
            };
            let dev = s.br_tab.as_ref().unwrap().ptr();
            let pin = s.br_pin_tab.as_ref().unwrap();
            unsafe {
                if ok(check(
                    (lib.memcpy_htod_async)(dev, pin.ptr as *const c_void, tab_bytes, std::ptr::null_mut()),
                    "cuMemcpyHtoDAsync (batch rope tab)",
                ))
                .is_none()
                {
                    return false;
                }
            }
            s.br_tab_key.clear();
            s.br_tab_key.extend_from_slice(poss);
            s.br_tab_dh = dh;
            s.br_tab_theta = theta;
            s.br_tab_valid = true;
        }
        // --- バッファ確保 + q/k アップロード (pin は [q|k] 連結で 1 本) ---
        if ok(Scratch::ensure(&mut s.br_q, q_bytes)).is_none()
            || ok(Scratch::ensure(&mut s.br_k, k_bytes)).is_none()
            || ok(Scratch::ensure_pin(&mut s.br_pin_qk, q_bytes + k_bytes)).is_none()
        {
            return false;
        }
        let lib = match ok(lib()) {
            Some(l) => l,
            None => return false,
        };
        let (dq, dk) = (
            s.br_q.as_ref().unwrap().ptr(),
            s.br_k.as_ref().unwrap().ptr(),
        );
        unsafe {
            {
                let pin = s.br_pin_qk.as_mut().unwrap();
                pin.as_mut_slice::<f32>()[..q_floats].copy_from_slice(q);
                if ok(check(
                    (lib.memcpy_htod_async)(dq, pin.ptr as *const c_void, q_bytes, std::ptr::null_mut()),
                    "cuMemcpyHtoDAsync (batch rope q)",
                ))
                .is_none()
                {
                    return false;
                }
            }
            {
                let pin = s.br_pin_qk.as_mut().unwrap();
                pin.as_mut_slice::<f32>()[q_floats..q_floats + k_floats].copy_from_slice(k);
                let ksrc = (pin.ptr as *const u8).add(q_bytes) as *const c_void;
                if ok(check(
                    (lib.memcpy_htod_async)(dk, ksrc, k_bytes, std::ptr::null_mut()),
                    "cuMemcpyHtoDAsync (batch rope k)",
                ))
                .is_none()
                {
                    return false;
                }
            }
        }
        // --- カーネル起動 (grid = (heads + kv_heads, b)・block = (dh)) ---
        let (mut pq, mut pk, mut pt) = (dq, dk, s.br_tab.as_ref().unwrap().ptr());
        let (mut hc, mut khc, mut dc) = (heads as c_int, kv_heads as c_int, dh as c_int);
        let mut args: Vec<*mut c_void> = vec![
            &mut pq as *mut _ as *mut c_void,
            &mut pk as *mut _ as *mut c_void,
            &mut pt as *mut _ as *mut c_void,
            &mut hc as *mut _ as *mut c_void,
            &mut khc as *mut _ as *mut c_void,
            &mut dc as *mut _ as *mut c_void,
        ];
        if ok(unsafe {
            ctx.launch_async(
                ctx.f_rope_b,
                ((heads + kv_heads) as u32, b as u32, 1),
                (dh as u32, 1, 1),
                0,
                &mut args,
            )
        })
        .is_none()
        {
            return false;
        }
        if ok(ctx.sync()).is_none() {
            return false;
        }
        // --- 結果 D2H ([q|k] を pin の非重複領域へ・両方成功するまでホスト q/k は不変) ---
        unsafe {
            {
                let pin = s.br_pin_qk.as_mut().unwrap();
                if ok(check(
                    (lib.memcpy_dtoh_async)(pin.ptr, dq, q_bytes, std::ptr::null_mut()),
                    "cuMemcpyDtoHAsync (batch rope q)",
                ))
                .is_none()
                {
                    return false;
                }
            }
            {
                let pin = s.br_pin_qk.as_mut().unwrap();
                let kdst = (pin.ptr as *mut u8).add(q_bytes) as *mut c_void;
                if ok(check(
                    (lib.memcpy_dtoh_async)(kdst, dk, k_bytes, std::ptr::null_mut()),
                    "cuMemcpyDtoHAsync (batch rope k)",
                ))
                .is_none()
                {
                    return false;
                }
            }
        }
        if ok(ctx.sync()).is_none() {
            return false;
        }
        {
            let pin = s.br_pin_qk.as_ref().unwrap();
            let sl = pin.as_slice::<f32>();
            q.copy_from_slice(&sl[..q_floats]);
            k.copy_from_slice(&sl[q_floats..q_floats + k_floats]);
        }
        true
    })
}


/// P17k ステップ 5: dno_rope_qkv_pf の起動 (chain_qkv_rope_prefill とテスト共用)。
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_rope_qkv_pf(
    ctx: &CudaCtx,
    qkv: CUdeviceptr,
    qout: CUdeviceptr,
    km: CUdeviceptr,
    vm: CUdeviceptr,
    tab: CUdeviceptr,
    bias: CUdeviceptr,
    pos0: usize,
    n: usize,
    heads: usize,
    kv_heads: usize,
    dh: usize,
    kv_dim: usize,
) -> Result<()> {
    let (mut pq, mut pqo, mut pkm, mut pvm, mut pt, mut pb) = (qkv, qout, km, vm, tab, bias);
    let (mut p0c, mut hc, mut khc, mut dc, mut kdc) = (
        pos0 as c_int,
        heads as c_int,
        kv_heads as c_int,
        dh as c_int,
        kv_dim as c_int,
    );
    let mut args: Vec<*mut c_void> = vec![
        &mut pq as *mut _ as *mut c_void,
        &mut pqo as *mut _ as *mut c_void,
        &mut pkm as *mut _ as *mut c_void,
        &mut pvm as *mut _ as *mut c_void,
        &mut pt as *mut _ as *mut c_void,
        &mut pb as *mut _ as *mut c_void,
        &mut p0c as *mut _ as *mut c_void,
        &mut hc as *mut _ as *mut c_void,
        &mut khc as *mut _ as *mut c_void,
        &mut dc as *mut _ as *mut c_void,
        &mut kdc as *mut _ as *mut c_void,
    ];
    unsafe {
        ctx.launch_async(
            ctx.f_rope_pf,
            ((heads + 2 * kv_heads) as u32, n as u32, 1),
            (dh as u32, 1, 1),
            0,
            &mut args,
        )
    }
}


/// P17k ステップ 5: prefill の qkv GEMM → bias/rope → K/V キャッシュ書き込みまでを
/// デバイスで完結させる (qkv→rope→attention 連続化の前半)。従来経路との差分:
/// - qkv GEMM 出力 [n, width] (122MB @13k tok) の D2H + q/k/v 分割コピーが消える
/// - rope が CPU (f32 powf/cos/sin × n×(heads+kv_heads)×half) から GPU へ
/// - rope 後 q の H2D (81MB) が消える (attention は chain_attention_prefill_dev
///   でデバイス q を直接読む)
/// - K/V は rope カーネルが GPU ミラーに直接書き、[pos0..t_total) 行だけ D2H
///   (27MB) でホストキャッシュへ (旧: CPU キャッシュ書き込み + ミラー H2D 27MB)
///
/// ビット一致: GEMM は dno_gemm_i8、bias は単一 add、rope は mul/sub (tab は
/// CPU が rope_row と同一式で生成) — CPU 経路 (lin_qkv_gpu → add_bias →
/// rope_rows → cache 書き込み) と q/K/V は完全一致。緩和不要。
///
/// 前提: 呼び出し側が bias/QK-Norm 無しを確認済み・cache は t_total 行分の容量
/// 確保済み・現在レイヤのホスト行 [pos0..t_total) は未書き込み (本関数が D2H で
/// 書く)。失敗時 None (ホスト状態不変 → 呼び出し側が従来経路でやり直し)。
/// ロープテーブル/rope カーネル成功後にミラー len を t_total に確定する。
#[allow(clippy::too_many_arguments)]
pub fn chain_qkv_rope_prefill(
    q_w: &Quant8Linear,
    k_w: &Quant8Linear,
    v_w: &Quant8Linear,
    h: &[f32],
    cache: &mut crate::kv_cache::KvCache,
    key: u64,
    li: usize,
    n_layers: usize,
    pos0: usize,
    n: usize,
    t_total: usize,
    heads: usize,
    kv_heads: usize,
    dh: usize,
    theta: f32,
    bias: &[f32],
    // B1: Some(slot) なら h は chain_rmsnorm が既にそのスロットに書き込んだ
    // デバイス常駐データとして直接読む (ホスト h の H2D 81MB/層を廃止)。
    // None なら従来どおりホスト `h` をアップロードする。Some 時は `h` を空でよい。
    h_dev: Option<usize>,
) -> Option<()> {
    if !gpu_attention_enabled() || !gemm_enabled() {
        return None;
    }
    if !(n > max_m() && n >= gemm_min_m()) {
        return None;
    }
    if n == 0 || t_total > cache.max_len() || n > 65535 {
        return None; // rope カーネルの grid.y 上限
    }
    let qn = heads.checked_mul(dh)?;
    let kv_dim = kv_heads.checked_mul(dh)?;
    let width = qn.checked_add(2 * kv_dim)?;
    let e = q_w.k;
    if k_w.k != e || v_w.k != e || k_w.n != kv_dim || v_w.n != kv_dim || q_w.n != qn {
        return None;
    }
    if h_dev.is_none() && h.len() != n * e {
        return None;
    }
    let ctx = chain_ctx()?;
    let lib = ok(lib())?;

    // ミラー取得 + 履歴行 [have..pos0) の同期 (現在行 [pos0..t_total) は rope
    // カーネルが書く)。have > pos0 はミラーがホストに先行 → フォールバック
    // (ホストキャッシュが完全でないため)。
    let (kptr, vptr) =
        with_mirror(key, li, n_layers, t_total, kv_dim, cache.epoch(), |m, have| {
        if have > pos0 {
            return None;
        }
        if have < pos0 {
            let (kc_ref, vc_ref) = cache.kv_bufs_ref(li);
            let kb = m.layers[li].0.as_ref().unwrap();
            let vb = m.layers[li].1.as_ref().unwrap();
            ok(kb.copy_from_at(have * kv_dim * 4, &kc_ref[have * kv_dim..pos0 * kv_dim]))?;
            ok(vb.copy_from_at(have * kv_dim * 4, &vc_ref[have * kv_dim..pos0 * kv_dim]))?;
        }
        Some((
            m.layers[li].0.as_ref().unwrap().ptr(),
            m.layers[li].1.as_ref().unwrap().ptr(),
        ))
    })?;

    let uids = [q_w.uid, k_w.uid, v_w.uid];
    let gw = match weights_for_group(&uids, &[q_w, k_w, v_w], e, width) {
        Ok(g) => g,
        Err(err) => {
            tracing::error!("CUDA qkv weight upload failed, using CPU: {err:#}");
            GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
            return None;
        }
    };

    // bias: [qb | kb | vb] 連結 1 バッファ (空なら無し)。dno_rope_qkv_pf が
    // rope 前に加算する (CPU 経路の add_bias → rope_rows と同一順序・ビット一致)。
    let bias_len = bias.len();
    if !bias.is_empty() && bias_len != width {
        return None;
    }

    CHAIN.with(|cell| -> Option<()> {
        let mut s = cell.borrow_mut();
        ok(Scratch::ensure(&mut s.qkv_aq, n * e))?;
        ok(Scratch::ensure(&mut s.qkv_asc, n * 4))?;
        ok(Scratch::ensure(&mut s.qkv_pf, n * width * 4))?;
        ok(Scratch::ensure(&mut s.qkv_q, n * qn * 4))?;
        ok(Scratch::ensure(&mut s.rope_tab_pf, (n * dh + bias_len) * 4))?;
        ok(Scratch::ensure_pin(&mut s.pin_qkv, (n * e * 4).max(n * dh * 4 + bias_len * 4)))?;

        // GEMM 入力 h: B1 では chain_rmsnorm が書いたデバイス常駐スロット (CH_H)
        // を直接読む (H2D 不要)。None の場合は従来どおり pinned からアップロード。
        let dx = match h_dev {
            Some(slot) => {
                let b = match s.bufs.get(slot) {
                    Some(Some(b)) => b,
                    _ => return chain_fail(anyhow!("chain_qkv_rope_prefill: h_dev slot missing")),
                };
                if b.bytes() < n * e * 4 {
                    return chain_fail(anyhow!("chain_qkv_rope_prefill: h_dev slot too small"));
                }
                b.ptr()
            }
            None => {
                ok(Scratch::ensure(&mut s.qkv_x, n * e * 4))?;
                // h アップロード (pinned + async)。pin は次層で上書きするが、その前に
                // 本関数内の sync が挟まるため安全。
                {
                    let pin = s.pin_qkv.as_mut().unwrap();
                    pin.as_mut_slice::<f32>()[..n * e].copy_from_slice(h);
                }
                let pin_ptr = s.pin_qkv.as_ref().unwrap().ptr;
                unsafe {
                    ok(check(
                        (lib.memcpy_htod_async)(
                            s.qkv_x.as_ref().unwrap().ptr(),
                            pin_ptr as *const c_void,
                            n * e * 4,
                            std::ptr::null_mut(),
                        ),
                        "cuMemcpyHtoDAsync (qkv x)",
                    ))?;
                }
                s.qkv_x.as_ref().unwrap().ptr()
            }
        };
        let (daq, dasc, dqkv, dq, dtab) = (
            s.qkv_aq.as_ref().unwrap().ptr(),
            s.qkv_asc.as_ref().unwrap().ptr(),
            s.qkv_pf.as_ref().unwrap().ptr(),
            s.qkv_q.as_ref().unwrap().ptr(),
            s.rope_tab_pf.as_ref().unwrap().ptr(),
        );
        let pin_ptr = s.pin_qkv.as_ref().unwrap().ptr;
        // 1) qkv GEMM: h [n,e] → [n, width] (dno_gemm_i8・CPU lin とビット一致)
        crate::prof::time_layer(crate::prof::QKV, li, || -> Option<()> {
            let (mut px, mut paq, mut pasc) = (dx, daq, dasc);
            let (mut mc, mut kc_) = (n as c_int, e as c_int);
            let mut args = [
                &mut px as *mut _ as *mut c_void,
                &mut paq as *mut _ as *mut c_void,
                &mut pasc as *mut _ as *mut c_void,
                &mut mc as *mut _ as *mut c_void,
                &mut kc_ as *mut _ as *mut c_void,
            ];
            ok(unsafe { ctx.launch_async(ctx.f_quant, (n as u32, 1, 1), (256, 1, 1), 256 * 4, &mut args) })?;
            let (mut paqi, mut pasci, mut pw, mut psc, mut pout) =
                (daq, dasc, gw.w.ptr(), gw.sc.ptr(), dqkv);
            let (mut mc, mut kc_, mut nc) = (n as c_int, e as c_int, width as c_int);
            let mut args = [
                &mut paqi as *mut _ as *mut c_void,
                &mut pasci as *mut _ as *mut c_void,
                &mut pw as *mut _ as *mut c_void,
                &mut psc as *mut _ as *mut c_void,
                &mut pout as *mut _ as *mut c_void,
                &mut mc as *mut _ as *mut c_void,
                &mut kc_ as *mut _ as *mut c_void,
                &mut nc as *mut _ as *mut c_void,
            ];
            let gmx = n.div_ceil(GEMM_TILE) as u32;
            let gmy = width.div_ceil(GEMM_TILE_N) as u32;
            ok(unsafe { ctx.launch_async(ctx.f_gemm, (gmx, gmy, 1), (GEMM_BX, GEMM_BY, 1), 0, &mut args) })
        })?;

        // 2) rope テーブル (行ごと pos0+i・rope_row と同一式) と bias
        //    ([qb|kb|vb] 連結) を pinned に入れてアップロード。
        {
            let pin = s.pin_qkv.as_mut().unwrap();
            let tab = &mut pin.as_mut_slice::<f32>()[..n * dh];
            let half = dh / 2;
            for i in 0..n {
                let pos = (pos0 + i) as f32;
                let row = &mut tab[i * dh..][..dh];
                for d in 0..half {
                    let ang = pos * theta.powf(-2.0 * d as f32 / dh as f32);
                    row[d] = ang.cos();
                    row[half + d] = ang.sin();
                }
            }
            if bias_len > 0 {
                let dst = &mut pin.as_mut_slice::<f32>()[n * dh..n * dh + bias_len];
                dst.copy_from_slice(bias);
            }
        }
        unsafe {
            ok(check(
                (lib.memcpy_htod_async)(dtab, pin_ptr as *const c_void, n * dh * 4, std::ptr::null_mut()),
                "cuMemcpyHtoDAsync (rope tab pf)",
            ))?;
            if bias_len > 0 {
                ok(check(
                    (lib.memcpy_htod_async)(
                        dtab + (n * dh * 4) as CUdeviceptr,
                        (pin_ptr as *const u8).add(n * dh * 4) as *const c_void,
                        bias_len * 4,
                        std::ptr::null_mut(),
                    ),
                    "cuMemcpyHtoDAsync (rope bias pf)",
                ))?;
            }
        }
        // 3) rope: q → qkv_q (連続)、k/v → ミラーの (pos0+i) 行へ
        let dbias = if bias_len > 0 {
            dtab + (n * dh * 4) as CUdeviceptr
        } else {
            0
        };
        crate::prof::time_layer(crate::prof::ROPE, li, || {
            ok(launch_rope_qkv_pf(
                ctx, dqkv, dq, kptr, vptr, dtab, dbias, pos0, n, heads, kv_heads, dh, kv_dim,
            ))
        })?;
        // 4) sync して K/V 現在行をホストキャッシュへ D2H (27MB・~3ms)。
        //    以降ホスト側に q/k/v Vec は存在しない。
        ok(ctx.sync())?;
        // s65: 発行元の件数分類 (K/V 行 2 本・表示専用)
        crate::prof::mark_d2h_src(crate::prof::D2H_SRC_KV_ROW);
        crate::prof::mark_d2h_src(crate::prof::D2H_SRC_KV_ROW);
        {
            let kb = cache.k_buf(li, 0);
            ok(unsafe {
                check(
                    (lib.memcpy_dtoh)(
                        kb[pos0 * kv_dim..t_total * kv_dim].as_mut_ptr() as *mut c_void,
                        kptr + (pos0 * kv_dim * 4) as CUdeviceptr,
                        n * kv_dim * 4,
                    ),
                    "cuMemcpyDtoH (qkv pf k)",
                )
            })?;
        }
        {
            let vb = cache.v_buf(li, 0);
            ok(unsafe {
                check(
                    (lib.memcpy_dtoh)(
                        vb[pos0 * kv_dim..t_total * kv_dim].as_mut_ptr() as *mut c_void,
                        vptr + (pos0 * kv_dim * 4) as CUdeviceptr,
                        n * kv_dim * 4,
                    ),
                    "cuMemcpyDtoH (qkv pf v)",
                )
            })?;
        }
        Some(())
    })?;

    // 5) ミラー len 確定 (rope カーネルのミラー書き込みが成功した後でのみ)。
    with_mirror(key, li, n_layers, t_total, kv_dim, cache.epoch(), |m, _| {
        m.len[li] = t_total;
        Some(())
    })
}


/// P17k ステップ 5: prefill GPU attention — q が **デバイス常駐** (chain_
/// qkv_rope_prefill が書いた qkv_q) で K/V ミラーも同期済みの場合の variant。
/// q の H2D (81MB/層) とミラー行同期が不要。数値は chain_attention_prefill と
/// 完全同一 (同一カーネル・同一引数・q の内容も同一)。
/// 戻り None = CPU attention へ (chain_qkv_q_download で q を取得してから)。
#[allow(clippy::too_many_arguments)]
pub fn chain_attention_prefill_dev(
    key: u64,
    li: usize,
    n_layers: usize,
    pos0: usize,
    n: usize,
    t_total: usize,
    heads: usize,
    group: usize,
    dh: usize,
    kv_dim: usize,
    scale: f32,
    epoch: u64,
) -> Option<()> {
    if !gpu_attention_enabled() {
        return None;
    }
    if n == 0 || t_total == 0 || group == 0 || t_total > 65535 {
        return None;
    }
    // カーネル制約: chain_attention_prefill と同一 (s48 C1-a: smem 上限は opt-in 済み)
    if dh < ATTN_PF_BQ || dh > 1024 {
        return None;
    }
    if attn_pf_sh_f(dh) * 4 > attn_pf_dynamic_smem_limit() {
        return None;
    }
    let qn = heads.checked_mul(dh)?;
    let ctx = chain_ctx()?;
    ok(chain_ensure_buf(CH_ATT, n * qn * 4))?;
    let (mut kptr, mut vptr) =
        with_mirror(key, li, n_layers, t_total, kv_dim, epoch, |m, _| {
        // 現在行は chain_qkv_rope_prefill が書き込み済み (len が t_total に確定済み)。
        if m.len[li] < t_total {
            return None;
        }
        Some((
            m.layers[li].0.as_ref().unwrap().ptr(),
            m.layers[li].1.as_ref().unwrap().ptr(),
        ))
    })?;
    CHAIN.with(|cell| -> Option<()> {
        let mut s = cell.borrow_mut();
        let mut qptr = s.qkv_q.as_ref()?.ptr();
        let mut aptr = s.bufs[CH_ATT].as_ref().unwrap().ptr();
        let (mut heads_c, mut n_c, mut pos0_c) =
            (heads as c_int, n as c_int, pos0 as c_int);
        let (mut kv_dim_c, mut t_c, mut group_c) =
            (kv_dim as c_int, t_total as c_int, group as c_int);
        let (mut dh_c, mut scale_c) = (dh as c_int, scale);
        let mut args: Vec<*mut c_void> = vec![
            &mut qptr as *mut _ as *mut c_void,
            &mut kptr as *mut _ as *mut c_void,
            &mut vptr as *mut _ as *mut c_void,
            &mut aptr as *mut _ as *mut c_void,
            &mut heads_c as *mut _ as *mut c_void,
            &mut n_c as *mut _ as *mut c_void,
            &mut pos0_c as *mut _ as *mut c_void,
            &mut kv_dim_c as *mut _ as *mut c_void,
            &mut t_c as *mut _ as *mut c_void,
            &mut group_c as *mut _ as *mut c_void,
            &mut dh_c as *mut _ as *mut c_void,
            &mut scale_c as *mut _ as *mut c_void,
        ];
        ok(unsafe {
            ctx.launch_async(
                ctx.f_attn_pf,
                // P17k: (行ブロック, heads) — x=行ブロック最速次元で同一 head の
                // K/V を同時実行ブロック間で共有し L2 ヒット率を上げる
                (n.div_ceil(ATTN_PF_BQ) as u32, heads as u32, 1),
                (dh as u32, 1, 1),
                (attn_pf_sh_f(dh) * 4) as u32,
                &mut args,
            )
        })
    })
}


/// P17k ステップ 5: デバイス常駐 q (qkv_q) をホストにダウンロードする
/// (attention フォールバック用)。同期 D2H。
pub fn chain_qkv_q_download(n: usize, qn: usize) -> Option<Vec<f32>> {
    CHAIN.with(|cell| -> Option<Vec<f32>> {
        let s = cell.borrow_mut();
        let buf = s.qkv_q.as_ref()?;
        ok(buf.to_vec(n * qn))
    })
}


/// P17h: decode chain 用 GPU attention — q/k/v が **デバイス常駐** (CH_QKV) の
/// 場合の variant。rope 済み k/v 現在行を CH_QKV からミラーへ D2D し、履歴行を
/// ホストキャッシュから H2D 同期してカーネルを起動する。rope 済み k/v は
/// pinned ステージへ非同期 D2H し、ホストキャッシュへの反映はステップ最後の
/// sync 後に `chain_kv_stage_flush` で行う (層あたりの sync を廃止するのが目的)。
/// 戻り None = forward 全体を従来経路で再実行。
#[allow(clippy::too_many_arguments)]
pub fn chain_attention_dev(
    key: u64,
    li: usize,
    n_layers: usize,
    cache: &crate::kv_cache::KvCache,
    pos0: usize,
    t_total: usize,
    heads: usize,
    group: usize,
    dh: usize,
    kv_dim: usize,
    scale: f32,
) -> Option<()> {
    if !gpu_attention_enabled() {
        return None;
    }
    let ctx = chain_ctx()?;
    if heads == 0 || group == 0 || dh == 0 || dh > 1024 || t_total == 0 || t_total > cache.max_len()
    {
        return None;
    }
    ok(chain_ensure_buf(CH_ATT, heads * dh * 4))?;

    let (mut kptr, mut vptr) =
        with_mirror(key, li, n_layers, t_total, kv_dim, cache.epoch(), |m, have| {
        // 履歴行 [have..pos0) をホストキャッシュから同期 (現在行 pos0 は CH_QKV から D2D)。
        if have < pos0 {
            let (kc_ref, vc_ref) = cache.kv_bufs_ref(li);
            let kb = m.layers[li].0.as_ref().unwrap();
            let vb = m.layers[li].1.as_ref().unwrap();
            ok(kb.copy_from_at(have * kv_dim * 4, &kc_ref[have * kv_dim..pos0 * kv_dim]))?;
            ok(vb.copy_from_at(have * kv_dim * 4, &vc_ref[have * kv_dim..pos0 * kv_dim]))?;
        }
        // 現在行 (pos0) を CH_QKV からデバイス内コピー (rope 済み・ビット一致)。
        let qkv_off = (heads * dh * 4) as u64;
        let (kp_src, vp_src) = CHAIN.with(|cell| -> Option<(CUdeviceptr, CUdeviceptr)> {
            let s = cell.borrow();
            let base = s.bufs[CH_QKV].as_ref()?.ptr();
            Some((base + qkv_off, base + qkv_off + (kv_dim * 4) as u64))
        })?;
        let kb = m.layers[li].0.as_ref().unwrap();
        let vb = m.layers[li].1.as_ref().unwrap();
        let lib = ok(lib())?;
        let rowb = kv_dim * 4;
        unsafe {
            ok(check(
                (lib.memcpy_dtod_async)(
                    kb.ptr() + (pos0 * rowb) as u64,
                    kp_src,
                    rowb,
                    std::ptr::null_mut(),
                ),
                "cuMemcpyDtoDAsync (k row)",
            ))?;
            ok(check(
                (lib.memcpy_dtod_async)(
                    vb.ptr() + (pos0 * rowb) as u64,
                    vp_src,
                    rowb,
                    std::ptr::null_mut(),
                ),
                "cuMemcpyDtoDAsync (v row)",
            ))?;
        }
        m.len[li] = t_total;
        // roped k/v を pinned ステージへ非同期 D2H (ホスト反映は sync 後)。
        CHAIN.with(|cell| -> Option<()> {
            let mut s = cell.borrow_mut();
            let stage_bytes = n_layers * 2 * rowb;
            ok(Scratch::ensure_pin(&mut s.kv_stage, stage_bytes))?;
            let off = li * 2 * rowb;
            let pin = s.kv_stage.as_ref().unwrap();
            unsafe {
                ok(check(
                    (lib.memcpy_dtoh_async)(
                        (pin.ptr as *mut c_void).add(off),
                        kp_src,
                        rowb,
                        std::ptr::null_mut(),
                    ),
                    "cuMemcpyDtoHAsync (k stage)",
                ))?;
                ok(check(
                    (lib.memcpy_dtoh_async)(
                        (pin.ptr as *mut c_void).add(off + rowb),
                        vp_src,
                        rowb,
                        std::ptr::null_mut(),
                    ),
                    "cuMemcpyDtoHAsync (v stage)",
                ))?;
            }
            Some(())
        })?;
        Some((kb.ptr(), vb.ptr()))
    })?;

    // scores スクラッチ + カーネル起動 (q は CH_QKV 先頭 — アップロード不要)
    CHAIN.with(|cell| -> Option<()> {
        let mut s = cell.borrow_mut();
        ok(Scratch::ensure(&mut s.attn_sc, heads * t_total * 4))?;
        let mut qptr = s.bufs[CH_QKV].as_ref().unwrap().ptr();
        let mut scptr = s.attn_sc.as_ref().unwrap().ptr();
        let mut aptr = s.bufs[CH_ATT].as_ref().unwrap().ptr();
        let mut scale_c = scale;
        let mut kv_dim_c = kv_dim as c_int;
        let mut t_c = t_total as c_int;
        let mut group_c = group as c_int;
        let mut dh_c = dh as c_int;
        let mut args: Vec<*mut c_void> = vec![
            &mut qptr as *mut _ as *mut c_void,
            &mut kptr as *mut _ as *mut c_void,
            &mut vptr as *mut _ as *mut c_void,
            &mut aptr as *mut _ as *mut c_void,
            &mut scptr as *mut _ as *mut c_void,
            &mut kv_dim_c as *mut _ as *mut c_void,
            &mut t_c as *mut _ as *mut c_void,
            &mut group_c as *mut _ as *mut c_void,
            &mut dh_c as *mut _ as *mut c_void,
            &mut scale_c as *mut _ as *mut c_void,
        ];
        ok(unsafe {
            ctx.launch_async(
                ctx.f_attn,
                (heads as u32, 1, 1),
                (dh as u32, 1, 1),
                (dh * 4) as u32,
                &mut args,
            )
        })
    })
}


/// P17h: ステージングした rope 済み k/v をホスト KvCache へ反映。
/// ステップ最後の sync (logits download) 後に呼ぶ — 全 D2H 完了済み。
/// ホストキャッシュは従来どおり正として機能し続ける (CPU 経路/snapshot 互換)。
pub fn chain_kv_stage_flush(
    cache: &mut crate::kv_cache::KvCache,
    n_layers: usize,
    kv_dim: usize,
    pos0: usize,
) -> Option<()> {
    CHAIN.with(|cell| -> Option<()> {
        let s = cell.borrow();
        let stage = s.kv_stage.as_ref()?;
        let slice = stage.as_slice::<f32>();
        for li in 0..n_layers {
            let off = li * 2 * kv_dim;
            let (kc, vc) = cache.kv_bufs(li, 1);
            kc[pos0 * kv_dim..][..kv_dim].copy_from_slice(&slice[off..off + kv_dim]);
            vc[pos0 * kv_dim..][..kv_dim].copy_from_slice(&slice[off + kv_dim..off + 2 * kv_dim]);
        }
        Some(())
    })
}


/// テスト用: dno_expf (glibc expf の bit-exact 再現) を要素ごとに適用する
pub fn chain_expf_probe(inp: &[f32]) -> Option<Vec<f32>> {
    let ctx = chain_ctx()?;
    let n = inp.len();
    chain_upload(CH_GA, inp, n, 1)?;
    ok(chain_ensure_buf(CH_ACT, n * 4))?;
    CHAIN.with(|cell| -> Option<()> {
        let s = cell.borrow();
        let ip = s.bufs[CH_GA].as_ref().unwrap().ptr();
        let op = s.bufs[CH_ACT].as_ref().unwrap().ptr();
        let (mut pi, mut po) = (ip, op);
        let mut nc = n as c_int;
        let mut args = [
            &mut pi as *mut _ as *mut c_void,
            &mut po as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
        ];
        let grid = (n as u32).div_ceil(256);
        ok(unsafe { ctx.launch_async(ctx.f_expf, (grid, 1, 1), (256, 1, 1), 0, &mut args) })
    })?;
    chain_download(CH_ACT, n)
}


/// task 640: f16 -> f32 の全 65536 パターンで「旧 (手書きのビット展開)」と
/// 「新 (cvt.f32.f16)」を GPU 上で突き合わせる。戻り値は
/// (不一致の件数, 最初の不一致パターン)。不一致が無ければ (0, 0xffff_ffff)。
/// GPU / 収録 PTX が無ければ None (テストは skip する)。
pub fn chain_f16cvt_probe() -> Option<(u32, u32)> {
    let ctx = chain_ctx()?;
    let fp = ctx.f_f16probe?;
    ok(chain_ensure_buf(CH_ACT, 8))?;
    chain_upload(CH_ACT, &[0.0f32, f32::from_bits(0xffff_ffff)], 2, 1)?;
    CHAIN.with(|cell| -> Option<()> {
        let s = cell.borrow();
        let p = s.bufs[CH_ACT].as_ref()?.ptr();
        let mut pc = p;
        let mut pf = p + 4;
        let mut args = [&mut pc as *mut _ as *mut c_void, &mut pf as *mut _ as *mut c_void];
        // 256 x 16 ブロックで 65536 パターンを 16 個ずつ舐める。
        ok(unsafe { ctx.launch_async(fp, (16, 1, 1), (256, 1, 1), 0, &mut args) })
    })?;
    let out = chain_download(CH_ACT, 2)?;
    Some((out[0].to_bits(), out[1].to_bits()))
}


#[derive(Clone, Copy)]
pub(crate) struct Q35Fns {
    pub(crate) f_matvec_f32: CUfunction,
    // s169: 並列リダクション版 (DENNO_FAST_REDUCE=1 のときだけ使う。PTX 未収録なら None)
    f_matvec_f32_r: Option<CUfunction>,
    pub(crate) f_qknorm_rope: CUfunction,
    pub(crate) f_gate: CUfunction,
}


// 中身は生ハンドル (関数ハンドル)。所有権の移動は伴わないため Send/Sync を
// 与える (CudaCtx と同一の規約・driver がスレッド安全に管理する)。
unsafe impl Send for Q35Fns {}

unsafe impl Sync for Q35Fns {}


static Q35_FNS: OnceLock<Option<Q35Fns>> = OnceLock::new();


pub(crate) fn q35_fns() -> Option<&'static Q35Fns> {
    Q35_FNS
        .get_or_init(|| {
            let ctx = chain_ctx()?;
            let lib = ok(lib())?;
            let get = |name: &[u8]| -> Option<CUfunction> {
                let mut f: CUfunction = std::ptr::null_mut();
                let r = unsafe {
                    (lib.module_get_function)(
                        &mut f,
                        ctx.module_elem,
                        name.as_ptr() as *const c_char,
                    )
                };
                if r != 0 {
                    None
                } else {
                    Some(f)
                }
            };
            Some(Q35Fns {
                f_matvec_f32: get(b"dno_matvec_f32\0")?,
                f_matvec_f32_r: get(b"dno_matvec_f32_r\0"),
                f_qknorm_rope: get(b"dno_qknorm_rope_pi\0")?,
                f_gate: get(b"dno_attn_gate_mul\0")?,
            })
        })
        .as_ref()
}


/// C4: qwen35 chain 用新カーネルが利用可能か (発動前ゲート用)。
pub fn q35_fns_available() -> bool {
    q35_fns().is_some()
}


/// C4: qwen35 chain 用 fp32 重みのキャッシュタグ。
/// (model_id, layer, tag) をキーにデバイスバッファをキャッシュする
/// (Linear::F32 / Vec<f32> には uid が無いため)。
pub const Q35_TAG_BETA: u8 = 0; // ssm_beta.weight [e, nv]

pub const Q35_TAG_ALPHA: u8 = 1; // ssm_alpha.weight [e, nv]

pub const Q35_TAG_QNORM: u8 = 2; // q_norm.weight [dh]

pub const Q35_TAG_KNORM: u8 = 3; // k_norm.weight [dh]

pub const Q35_TAG_CONV: u8 = 4; // conv1d.weight [conv_dim, d_conv]

pub const Q35_TAG_WA: u8 = 5; // ssm_a [nv]

pub const Q35_TAG_WDT: u8 = 6; // ssm_dt.bias [nv]

pub const Q35_TAG_WNORM: u8 = 7; // ssm_norm.weight [dv]

/// s174 (P16): ssm_beta|ssm_alpha の**行連結** [2*nv, e] (行順 [beta | alpha])。
/// decode chain が beta/alpha の fp32 matvec を 1 発で回すためだけのタグ。
pub const Q35_TAG_BA: u8 = 8;


/// C4: full_attention 層の CH_QKV 内 k セクション先頭 (要素オフセット)。
/// レイアウトは [qg (heads*2dh) | k (kv_heads*dh) | v (kv_heads*dh)] —
/// chain_linear_grouped が parts 順に連結する。q は head ごと [q|gate] 融合で
/// head 間隔 2*dh だが、k/v は head 間隔 dh (chain_q35_fullattn の k 経路が
/// この関数を使い、誤ったオフセット/ストライドを防ぐ)。
pub(crate) fn q35_fullattn_k_off(heads: usize, dh: usize) -> usize {
    heads * 2 * dh
}


/// C4: decode chain 1 ステップの CH_QKV 要請バイト数 (純関数・単一ソース)。
/// - GDN 層: [qkv (2nk*dk+nv*dv) | z (nv*dv) | beta (nv) | alpha (nv)]
///   (= chain_q35_gdn_scan の `width`)
/// - full 層: [qg (heads*2dh) | k (kv_heads*dh) | v (kv_heads*dh)]
/// の大きい方。ステップ開始時にこのサイズで一括確保する。
///
/// 2994feb 時点では事前確保が無く、GDN 層の grouped GEMV (qkv+z) 直後に
/// chain_q35_gdn_scan 頭部の `chain_ensure_buf(CH_QKV, width*4)` が grow 再確保
/// (旧バッファ解放 + 新バッファ確保) を走らせていた。直前 launch の in-flight
/// 書き込み先が解放され、新バッファの [0..qkv+z) は誰にも書かれないまま
/// dno_gdn_scan が未初期化内容を読む (初期化されていない VRAM 内容は起動ごとに
/// 異なるため、実機 E2E の run 間出力差 + 冒頭乱れの直接原因)。
pub fn q35_ch_qkv_bytes(
    heads: usize,
    dh: usize,
    kv_heads: usize,
    nk: usize,
    nv: usize,
    dk: usize,
    dv: usize,
) -> usize {
    let kv_dim = kv_heads * dh;
    let gdn = (2 * nk * dk + nv * dv) + nv * dv + 2 * nv;
    let full = heads * 2 * dh + 2 * kv_dim;
    gdn.max(full) * 4
}


/// C4: decode chain 1 ステップの CH_ATT 要請バイト数 (純関数)。
/// GDN 層の出力 y は [vd = nv*dv]・full 層は [heads*dh]。
/// CH_QKV と同一の競合クラス (full 層側の ensure が直前 GDN scan の in-flight
/// 書き込み先を解放し得る) をステップ開始時の一括確保で排除する。
pub fn q35_ch_att_bytes(nv: usize, dv: usize, heads: usize, dh: usize) -> usize {
    (nv * dv).max(heads * dh) * 4
}


/// C4: KV ステージ (full 層の k/v 行 staged D2H 転送先 pinned) のステップ全体
/// バイト数 (純関数)。full 層 li の積み位置は li*2*kv_dim 要素・1 層分は
/// 2*kv_dim 要素 (chain_q35_kv_stage_flush の読み出しと同一式)。
pub fn q35_kv_stage_bytes(n_layers: usize, kv_dim: usize) -> usize {
    n_layers * 2 * kv_dim * 4
}


/// C4: fp32 重みのデバイスキャッシュ取得 (無ければ同期 H2D でアップロード)。
/// 初回 1 回のみ同期コピーが走る (chain 発動セッションの最初のステップ)。
pub(crate) fn q35_w_dev(mid: u64, li: usize, tag: u8, w: &[f32]) -> Option<CUdeviceptr> {
    let _ = chain_ctx()?;
    CHAIN.with(|cell| -> Option<CUdeviceptr> {
        let mut s = cell.borrow_mut();
        if let Some(b) = s.q35_w_cache.get(&(mid, li, tag)) {
            if b.bytes() == w.len() * 4 {
                return Some(b.ptr());
            }
        }
        let buf = ok(DevBuffer::from_slice_tagged(w, crate::mem_audit::TAG_CHAIN))?;
        let p = buf.ptr();
        s.q35_w_cache.insert((mid, li, tag), buf);
        Some(p)
    })
}


/// C4: qwen35 chain の発動条件 (カーネル形状ゲート)。数値契約は
/// rmsnorm/int8 GEMV/swiglu/expf/rope/QK-Norm/gate/fp32 matvec がビット一致、
/// attention スコア/softmax のみ緩和 (P17g・ユーザー承認済み)、GDN スキャンは
/// bit 一致不要契約 (決定論のみ)。形状はカーネルの前提 (gdn.cu / kernels_elem.cu
/// の C4 節コメント) と一致させる。
#[allow(clippy::too_many_arguments)]
pub fn q35_chain_shapes_ok(
    e: usize,
    ff: usize,
    dh: usize,
    heads: usize,
    kv_heads: usize,
    n_rot: usize,
    dk: usize,
    dv: usize,
    d_conv: usize,
) -> bool {
    // dno_add_rmsnorm の動的共有メモリ上限 (48KB / 4B)
    if e == 0 || e > 12_288 || ff == 0 {
        return false;
    }
    if dh == 0 || dh > 1024 || heads == 0 || kv_heads == 0 || heads % kv_heads != 0 {
        return false;
    }
    // partial interleaved rope: ペア数 n_rot/2 は dh 内・ペア単位で回転
    if n_rot == 0 || n_rot > dh || n_rot % 2 != 0 {
        return false;
    }
    // dno_gdn_scan の前提 (gdn.cu 先頭コメント): dk==dv・d_conv==4・dk<=256
    dk == dv && d_conv == 4 && (1..=256).contains(&dk)
}


/// C4: ホスト生成の partial interleaved rope テーブル [cos(n_rot/2) | sin(n_rot/2)]
/// をデバイスへアップロード (1 ステップ 1 回)。テーブルの生成は model::qwen35 側
/// (rope_partial_interleaved と同一式) が担当し、cuda 側は転送のみを担う
/// (テーブル値のビット一致を生成箇所で保証する)。専用 pinned スロットを使う
/// (P17h 規約: ステップ内 sync 無しのため pinned は用途ごとに専用化)。
pub fn chain_q35_rope_pi_upload(tab: &[f32]) -> Option<()> {
    let _ = chain_ctx()?;
    let bytes = tab.len().checked_mul(4)?;
    if bytes == 0 {
        return None;
    }
    CHAIN.with(|cell| -> Option<()> {
        let mut s = cell.borrow_mut();
        ok(Scratch::ensure(&mut s.q35_tab, bytes))?;
        ok(Scratch::ensure_pin(&mut s.q35_pin_tab, bytes))?;
        {
            let pin = s.q35_pin_tab.as_mut().unwrap();
            pin.as_mut_slice::<f32>()[..tab.len()].copy_from_slice(tab);
        }
        let dev = s.q35_tab.as_ref().unwrap().ptr();
        let pin = s.q35_pin_tab.as_ref().unwrap();
        let lib = ok(lib())?;
        unsafe {
            ok(check(
                (lib.memcpy_htod_async)(dev, pin.ptr as *const c_void, bytes, std::ptr::null_mut()),
                "cuMemcpyHtoDAsync (q35 rope tab)",
            ))
        }?;
        // s166 (P5): 観測のみ (1 ステップ 1 回・pos 依存なので削れない)
        dbg_htod(bytes);
        Some(())
    })
}


/// C4: GDN 制御投影 (beta/alpha) を dno_matvec_f32 で CH_H → CH_QKV テールへ
/// 出力する (tensor::matvec_bt とビット一致)。out_off は CH_QKV 内の要素
/// オフセット (grouped GEMV 出力 [qkv|z] の直後)。CH_QKV はテール分もここで
/// 確保する (chain_linear_grouped の ensure は grow-only なので上書きされない)。
pub(crate) fn chain_q35_ctl_f32(
    mid: u64,
    li: usize,
    tag: u8,
    w: &[f32],
    k: usize,
    n: usize,
    out_off: usize,
) -> Option<()> {
    let ctx = chain_ctx()?;
    let fns = q35_fns()?;
    if w.len() != k * n || n == 0 || n > 65535 {
        return None;
    }
    ok(chain_ensure_buf(CH_QKV, (out_off + n) * 4))?;
    let wptr = q35_w_dev(mid, li, tag, w)?;
    CHAIN.with(|cell| -> Option<()> {
        let s = cell.borrow();
        let hptr = s.bufs[CH_H].as_ref()?.ptr();
        let optr = s.bufs[CH_QKV].as_ref()?.ptr() + (out_off * 4) as u64;
        let (mut ph, mut pw, mut po) = (hptr, wptr, optr);
        let (mut kc, mut nc) = (k as c_int, n as c_int);
        let mut args = [
            &mut ph as *mut _ as *mut c_void,
            &mut pw as *mut _ as *mut c_void,
            &mut po as *mut _ as *mut c_void,
            &mut kc as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
        ];
        // s169: DENNO_FAST_REDUCE=1 なら固定の木で縮約する並列版 (block 256)。
        // 丸め順が変わるため既定は逐次版 (bit 一致を守る)。
        let (func, thr) = match (fns.f_matvec_f32_r, fast_reduce_enabled()) {
            (Some(fr), true) => (fr, 256u32),
            _ => (fns.f_matvec_f32, 128u32),
        };
        ok(unsafe { ctx.launch_async(func, (n as u32, 1, 1), (thr, 1, 1), 0, &mut args) })
    })
}


/// s174 (P16): beta|alpha 制御投影を **1 発**の fp32 matvec で計算する
/// (decode chain 専用・[`chain_q35_ctl_f32`] の 2 発を 1 発に畳んだ版)。
///
/// 背景: decode 1 ステップの `dno_matvec_f32` は GDN 層あたり 2 発 (beta・alpha)。
/// 両者は**同じ入力 (CH_H)・同じ形状 (k=e, n=nv)・隣接した発行**で、出力も
/// CH_QKV テールの連続領域 ([beta | alpha]) なので、行連結した重み 1 本
/// ([2*nv, e] row-major) と grid=(2*nv) の 1 発で置き換えられる
/// (48 層 x 2 発 = 96 -> 48 発/token)。
///
/// 数値は不変: `dno_matvec_f32` は 1 行 = 1 block で `j = blockIdx.x`・
/// `row = w + (size_t)j*k`・`out[j] = s`、行 j の逐次縮約は他の行にも n の値にも
/// 依存しない。行順を [beta 全行 | alpha 全行] にしても、各行の入力ベクトル・
/// 重み行・累積順は 2 発のときと同一 = ビット一致。カーネル・引数の並び・
/// block 数・縮約順は 1 つも変えない (grid.x の行数だけが 2 倍)。
///
/// `None` は「この経路を使えない」(形状不一致) — 呼び出し側が従来の 2 発に落ちる。
/// 発行そのものの失敗も `None` で、chain 全体が失敗する (数値経路は従来と同じ
/// fail closed)。
pub(crate) fn chain_q35_ctl_f32_ba(
    mid: u64,
    li: usize,
    w: &[f32],
    k: usize,
    n: usize,
    out_off: usize,
) -> Option<()> {
    let ctx = chain_ctx()?;
    let fns = q35_fns()?;
    let rows = n.checked_mul(2)?;
    if w.len() != k * rows || n == 0 || rows > 65535 {
        return None;
    }
    ok(chain_ensure_buf(CH_QKV, (out_off + rows) * 4))?;
    let wptr = q35_w_dev(mid, li, Q35_TAG_BA, w)?;
    CHAIN.with(|cell| -> Option<()> {
        let s = cell.borrow();
        let hptr = s.bufs[CH_H].as_ref()?.ptr();
        let optr = s.bufs[CH_QKV].as_ref()?.ptr() + (out_off * 4) as u64;
        let (mut ph, mut pw, mut po) = (hptr, wptr, optr);
        let (mut kc, mut nc) = (k as c_int, rows as c_int);
        let mut args = [
            &mut ph as *mut _ as *mut c_void,
            &mut pw as *mut _ as *mut c_void,
            &mut po as *mut _ as *mut c_void,
            &mut kc as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
        ];
        // 逐次版 / 並列版の選択は chain_q35_ctl_f32 と同一 (既定は逐次版 = bit 一致)。
        let (func, thr) = match (fns.f_matvec_f32_r, fast_reduce_enabled()) {
            (Some(fr), true) => (fr, 256u32),
            _ => (fns.f_matvec_f32, 128u32),
        };
        ok(unsafe { ctx.launch_async(func, (rows as u32, 1, 1), (thr, 1, 1), 0, &mut args) })
    })
}


/// C4: QK-Norm + partial interleaved rope (dno_qknorm_rope_pi)。
/// src スロットの src_off (要素) を先頭に stride x_stride で並ぶ heads 行
/// (block = heads) を処理し、dst_ptr + i*dst_stride へ書く。
/// - q: src = CH_QKV (head ごと [q|gate] 融合・x_stride = 2*dh) → CH_Q35_Q
/// - k: src = CH_QKV の k 部 (x_stride = kv_dim) → KV ミラーの pos0 行
/// rope テーブルは chain_q35_rope_pi_upload 済み (q35_tab) を使う。
#[allow(clippy::too_many_arguments)]
pub(crate) fn chain_q35_qknorm_rope(
    mid: u64,
    li: usize,
    tag: u8,
    w: &[f32],
    src_slot: usize,
    src_off: usize,
    x_stride: usize,
    heads: usize,
    cols: usize,
    n_rot: usize,
    eps: f32,
    dst_ptr: CUdeviceptr,
    dst_stride: usize,
) -> Option<()> {
    let ctx = chain_ctx()?;
    let fns = q35_fns()?;
    if w.len() != cols
        || heads == 0
        || cols == 0
        || cols > 1024
        || n_rot == 0
        || n_rot > cols
        || n_rot % 2 != 0
    {
        return None;
    }
    ok(chain_ensure_buf(src_slot, (src_off + heads * x_stride) * 4))?;
    let wptr = q35_w_dev(mid, li, tag, w)?;
    let tabp = CHAIN.with(|cell| -> Option<CUdeviceptr> {
        let s = cell.borrow();
        Some(s.q35_tab.as_ref()?.ptr())
    })?;
    CHAIN.with(|cell| -> Option<()> {
        let s = cell.borrow();
        let xptr = s.bufs[src_slot].as_ref()?.ptr() + (src_off * 4) as u64;
        let (mut px, mut pxs, mut pw, mut pt, mut po, mut pst) = (
            xptr,
            x_stride as c_int,
            wptr,
            tabp,
            dst_ptr,
            dst_stride as c_int,
        );
        let (mut cc, mut nc, mut ec) = (cols as c_int, n_rot as c_int, eps);
        let mut args = [
            &mut px as *mut _ as *mut c_void,
            &mut pxs as *mut _ as *mut c_void,
            &mut pw as *mut _ as *mut c_void,
            &mut pt as *mut _ as *mut c_void,
            &mut po as *mut _ as *mut c_void,
            &mut pst as *mut _ as *mut c_void,
            &mut cc as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
            &mut ec as *mut _ as *mut c_void,
        ];
        ok(unsafe {
            ctx.launch_async(
                fns.f_qknorm_rope,
                (heads as u32, 1, 1),
                (cols as u32, 1, 1),
                ((cols + 1) * 4) as u32,
                &mut args,
            )
        })
    })
}


/// C4: qwen35 full-attention 層の decode attention (デバイス常駐)。
/// 前提: CH_QKV に [qg (heads*2dh) | k (kv_dim) | v (kv_dim)] がある
/// (chain_linear_grouped 直後)。処理:
/// 1. KV ミラー取得 + 履歴行 [have..pos0) をホストから同期
/// 2. k: dno_qknorm_rope_pi が KV ミラーの pos0 行へ直接書く (QK-Norm + rope)
///    q: 同カーネルが CH_Q35_Q へ書く (attention 入力)
/// 3. v: CH_QKV からミラー pos0 行へ D2D + staged async D2H (k/v 行)
/// 4. dno_attention (GQA・緩和枠) → dno_attn_gate_mul (sigmoid ゲート・ビット一致)
/// ホストキャッシュへの反映は sync 後の chain_q35_kv_stage_flush。
/// 失敗時 None (forward 全体を従来経路でやり直し — ホスト状態不変のため安全)。
#[allow(clippy::too_many_arguments)]
pub fn chain_q35_fullattn(
    mid: u64,
    key: u64,
    li: usize,
    n_layers: usize,
    cache: &crate::kv_cache::KvCache,
    pos0: usize,
    heads: usize,
    kv_heads: usize,
    dh: usize,
    n_rot: usize,
    eps: f32,
    q_norm_w: &[f32],
    k_norm_w: &[f32],
) -> Option<()> {
    if !gpu_attention_enabled() {
        return None;
    }
    let ctx = chain_ctx()?;
    let fns = q35_fns()?;
    let kv_dim = kv_heads * dh;
    let t_total = pos0 + 1;
    if heads == 0 || dh == 0 || dh > 1024 || t_total == 0 || t_total > cache.max_len() {
        return None;
    }
    let group = heads / kv_heads;
    let hd = heads * dh;
    ok(chain_ensure_buf(CH_ATT, hd * 4))?;
    ok(chain_ensure_buf(CH_Q35_Q, hd * 4))?;

    // 1) ミラー取得 + 履歴行同期 (ポインタはロック解放前に抽出)
    let (mut kptr, mut vptr) =
        with_mirror(key, li, n_layers, t_total, kv_dim, cache.epoch(), |m, have| {
        if have < pos0 {
            let (kc_ref, vc_ref) = cache.kv_bufs_ref(li);
            let kb = m.layers[li].0.as_ref().unwrap();
            let vb = m.layers[li].1.as_ref().unwrap();
            ok(kb.copy_from_at(have * kv_dim * 4, &kc_ref[have * kv_dim..pos0 * kv_dim]))?;
            ok(vb.copy_from_at(have * kv_dim * 4, &vc_ref[have * kv_dim..pos0 * kv_dim]))?;
        }
        m.len[li] = t_total;
        Some((
            m.layers[li].0.as_ref().unwrap().ptr(),
            m.layers[li].1.as_ref().unwrap().ptr(),
        ))
    })?;

    // 2) QK-Norm + rope: k はミラー pos0 行へ直接書く・q は CH_Q35_Q へ
    //    (CH_QKV レイアウト: [qg (heads*2dh) | k (kv_dim) | v (kv_dim)] —
    //    k セクション先頭は q35_fullattn_k_off (heads*2*dh)・head b の行内
    //    位置は b*dh (head 間隔 dh。融合 [q|gate] の q とは間隔が異なる)。
    //    旧実装は src_off=hd (qg 後半を読む)・head 間隔 kv_dim (k 節域を外れて
    //    読む + ミラー行の外へ書く) で、k 履歴を破壊していた。
    let k_off = q35_fullattn_k_off(heads, dh);
    let km_row = kptr + (pos0 * kv_dim * 4) as u64;
    let q_ptr = CHAIN.with(|cell| -> Option<CUdeviceptr> {
        let s = cell.borrow();
        Some(s.bufs[CH_Q35_Q].as_ref()?.ptr())
    })?;
    chain_q35_qknorm_rope(
        mid, li, Q35_TAG_QNORM, q_norm_w,
        CH_QKV, 0, 2 * dh, heads, dh, n_rot, eps,
        q_ptr,
        dh,
    )?;
    chain_q35_qknorm_rope(
        mid, li, Q35_TAG_KNORM, k_norm_w,
        CH_QKV, k_off, dh, kv_heads, dh, n_rot, eps, km_row, dh,
    )?;

    // 3) v 行 D2D + k/v 行の staged async D2H (ホスト反映は sync 後)
    let (vp_src, kv_dim64) = CHAIN.with(|cell| -> Option<(CUdeviceptr, u64)> {
        let s = cell.borrow();
        let base = s.bufs[CH_QKV].as_ref()?.ptr();
        Some((base + ((k_off + kv_dim) * 4) as u64, (kv_dim * 4) as u64))
    })?;
    let lib = ok(lib())?;
    unsafe {
        ok(check(
            (lib.memcpy_dtod_async)(
                vptr + (pos0 as u64 * kv_dim64),
                vp_src,
                kv_dim64 as usize,
                std::ptr::null_mut(),
            ),
            "cuMemcpyDtoDAsync (q35 v row)",
        ))?;
    }
    CHAIN.with(|cell| -> Option<()> {
        let mut s = cell.borrow_mut();
        let stage_bytes = q35_kv_stage_bytes(n_layers, kv_dim);
        ok(Scratch::ensure_pin(&mut s.kv_stage, stage_bytes))?;
        let off = li * 2 * kv_dim * 4;
        let pin = s.kv_stage.as_ref().unwrap();
        unsafe {
            ok(check(
                (lib.memcpy_dtoh_async)(
                    (pin.ptr as *mut c_void).add(off),
                    km_row,
                    kv_dim * 4,
                    std::ptr::null_mut(),
                ),
                "cuMemcpyDtoHAsync (q35 k stage)",
            ))?;
            ok(check(
                (lib.memcpy_dtoh_async)(
                    (pin.ptr as *mut c_void).add(off + kv_dim * 4),
                    vp_src,
                    kv_dim * 4,
                    std::ptr::null_mut(),
                ),
                "cuMemcpyDtoHAsync (q35 v stage)",
            ))?;
            // s65: 発行元の件数分類 (full 層の K/V 行 2 本。staged async 発行だが
            // 件数は d2h_state の calls に含まれる — 表示専用)
            crate::prof::mark_d2h_src(crate::prof::D2H_SRC_KV_ROW);
            crate::prof::mark_d2h_src(crate::prof::D2H_SRC_KV_ROW);
            // s166 (P5): 観測のみ (full 層 1 つあたり k/v 2 本・ホスト KV キャッシュ契約)
            dbg_d2h(kv_dim * 4);
            dbg_d2h(kv_dim * 4);
        }
        Some(())
    })?;

    // 4) attention (q は CH_Q35_Q) + sigmoid 出力ゲート
    let scale = 1.0 / (dh as f32).sqrt();
    CHAIN.with(|cell| -> Option<()> {
        let mut s = cell.borrow_mut();
        ok(Scratch::ensure(&mut s.attn_sc, heads * t_total * 4))?;
        let mut qptr = s.bufs[CH_Q35_Q].as_ref()?.ptr();
        let mut scptr = s.attn_sc.as_ref().unwrap().ptr();
        let mut aptr = s.bufs[CH_ATT].as_ref().unwrap().ptr();
        let mut scale_c = scale;
        let mut kv_dim_c = kv_dim as c_int;
        let mut t_c = t_total as c_int;
        let mut group_c = group as c_int;
        let mut dh_c = dh as c_int;
        let mut args: Vec<*mut c_void> = vec![
            &mut qptr as *mut _ as *mut c_void,
            &mut kptr as *mut _ as *mut c_void,
            &mut vptr as *mut _ as *mut c_void,
            &mut aptr as *mut _ as *mut c_void,
            &mut scptr as *mut _ as *mut c_void,
            &mut kv_dim_c as *mut _ as *mut c_void,
            &mut t_c as *mut _ as *mut c_void,
            &mut group_c as *mut _ as *mut c_void,
            &mut dh_c as *mut _ as *mut c_void,
            &mut scale_c as *mut _ as *mut c_void,
        ];
        ok(unsafe {
            ctx.launch_async(
                ctx.f_attn,
                (heads as u32, 1, 1),
                (dh as u32, 1, 1),
                (dh * 4) as u32,
                &mut args,
            )
        })?;
        // sigmoid 出力ゲート: att[h,i] /= (1 + exp(-qg[h*2dh + dh + i]))
        let mut qgptr = s.bufs[CH_QKV].as_ref()?.ptr();
        let (mut hc, mut dhc) = (heads as c_int, dh as c_int);
        let mut gargs = [
            &mut aptr as *mut _ as *mut c_void,
            &mut qgptr as *mut _ as *mut c_void,
            &mut hc as *mut _ as *mut c_void,
            &mut dhc as *mut _ as *mut c_void,
        ];
        ok(unsafe {
            ctx.launch_async(fns.f_gate, (heads as u32, 1, 1), (dh as u32, 1, 1), 0, &mut gargs)
        })
    })
}


/// C4: KV ステージ pinned をステップ開始時に全体分事前確保する (s50 から GDN は
/// ステージしないため kv_stage のみ)。chain_q35_fullattn 内の ensure は事前確保後
/// no-op の二重防御。
pub fn chain_q35_kv_stage_begin(stage_bytes: usize) -> Option<()> {
    CHAIN.with(|cell| -> Option<()> {
        let mut s = cell.borrow_mut();
        ok(Scratch::ensure_pin(&mut s.kv_stage, stage_bytes))
    })
}


/// C4: チャネルスロットをステップ開始時に事前確保する (同期・カーネル発行なし)。
/// kv_stage と同一の「最初に全体サイズを確保」規約の CH_QKV /
/// CH_ATT 版。事前確保後はステップ内の ensure が全て no-op になり、
/// in-flight カーネルの書き込み先 / 読み出し元の grow 再確保による解放が
/// 構造的に起こらない (q35_ch_qkv_bytes の doc を参照)。
pub fn chain_q35_step_reserve(slot: usize, bytes: usize) -> Option<()> {
    let _ = chain_ctx()?;
    ok(chain_ensure_buf(slot, bytes))
}


/// C4: int8 量子化スクラッチ (aq) をステップ開始時に最大 k 分事前確保する。
/// chain_linear_grouped はグループごとの k で grow するため、2994feb 時点では
/// 初回 down GEMV (k=ff) で再確保が走り、直前 GEMV (gate/up) の in-flight
/// 読み出し元を解放していた (解放済みブロックの内容は多くの場合残るが UB)。
pub fn chain_q35_step_reserve_aq(max_k: usize) -> Option<()> {
    let _ = chain_ctx()?;
    CHAIN.with(|cell| -> Option<()> {
        let mut s = cell.borrow_mut();
        ok(Scratch::ensure(&mut s.aq, 1 * max_k))
    })
}


/// C4: qwen35 用 KV ステージフラッシュ (GDN 層は KV をステージしないため
/// 該当層をスキップする。llama 用 chain_kv_stage_flush は全層をフラッシュする
/// ため流用できない)。
pub fn chain_q35_kv_stage_flush(
    cache: &mut crate::kv_cache::KvCache,
    n_layers: usize,
    kv_dim: usize,
    pos0: usize,
    skip: &[usize],
) -> Option<()> {
    CHAIN.with(|cell| -> Option<()> {
        let s = cell.borrow();
        let stage = s.kv_stage.as_ref()?;
        let slice = stage.as_slice::<f32>();
        let mut si = 0usize;
        for li in 0..n_layers {
            if si < skip.len() && skip[si] == li {
                si += 1;
                continue;
            }
            let off = li * 2 * kv_dim;
            let (kc, vc) = cache.kv_bufs(li, 1);
            kc[pos0 * kv_dim..][..kv_dim].copy_from_slice(&slice[off..off + kv_dim]);
            vc[pos0 * kv_dim..][..kv_dim].copy_from_slice(&slice[off + kv_dim..off + 2 * kv_dim]);
        }
        Some(())
    })
}


// ===========================================================================
// ---------------------------------------------------------------------------
// task 133: qwen4exp full attention GPU 経路のミラー台帳 (GPU 不要) と
// 実機でのみ回す数値一致テスト
// ---------------------------------------------------------------------------
#[cfg(test)]
mod q4_attn_tests {
    use super::*;
    use crate::model::qwen4exp_attn::AttnCache;

    /// 決定論的 LCG 乱数 (`tests::pseudo_random` と同じ式)。このモジュールは
    /// GPU 無しでも回るので自前で持つ。
    pub(crate) fn pseudo_random(len: usize, seed: u32) -> Vec<f32> {
        let mut x = seed;
        (0..len)
            .map(|_| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((x >> 16) % 2000) as f32 / 1000.0 - 1.0
            })
            .collect()
    }

    /// テスト用の「デバイスミラー」。本番の DevBuffer の代わりに Vec<f32> を持ち、
    /// cap 成長は D2D と同じく先頭 `keep` 行を温存する。
    struct SimMirror {
        state: Q4MirrorState,
        k: Vec<Vec<f32>>,
        v: Vec<Vec<f32>>,
        row: usize,
    }

    impl SimMirror {
        fn new(n_layer: usize, row: usize) -> Self {
            Self {
                state: Q4MirrorState {
                    cap: 0,
                    len: vec![0; n_layer],
                    gen: vec![0; n_layer],
                },
                k: vec![Vec::new(); n_layer],
                v: vec![Vec::new(); n_layer],
                row,
            }
        }

        /// cap 成長 (D2D で先頭 keep 行を温存)。行数は `q4_mirror_cap` と同じ規則。
        /// task 174: 確保は実物と同じ cap + 投機読みの余白 (1 行)。
        fn grow(&mut self, new_cap: usize) {
            let rows = q4_mirror_alloc_rows(new_cap);
            for l in 0..self.k.len() {
                let keep = self.state.len[l].min(new_cap);
                let mut nk = vec![f32::NAN; rows * self.row];
                let mut nv = vec![f32::NAN; rows * self.row];
                nk[..keep * self.row].copy_from_slice(&self.k[l][..keep * self.row]);
                nv[..keep * self.row].copy_from_slice(&self.v[l][..keep * self.row]);
                self.k[l] = nk;
                self.v[l] = nv;
            }
            self.state.cap = new_cap;
        }

        /// 本番の q4_attn_mirror_sync と同じ手順 (cap 成長 → have → 差分コピー → note)。
        /// 戻り値は今回ホストから送った行数 (have)。
        fn sync(&mut self, cache: &AttnCache, li: usize) -> usize {
            let len = cache.len(li).unwrap();
            let gen = cache.gen(li).unwrap();
            let new_cap = q4_mirror_cap(len, self.state.cap);
            if new_cap > self.state.cap {
                self.grow(new_cap);
            }
            let have = q4_mirror_have(&self.state, li, len, gen);
            let row = self.row;
            self.k[li][have * row..len * row]
                .copy_from_slice(&cache.k(li).unwrap()[have * row..len * row]);
            self.v[li][have * row..len * row]
                .copy_from_slice(&cache.v(li).unwrap()[have * row..len * row]);
            q4_mirror_note(&mut self.state, li, len, gen);
            // task 174: flashdec が踏む「分割の終端 + 1 行」(= len 行目) が
            // 確保の内側にあること。実物は cap + 1 行なので常に成り立つ。
            assert!(
                (len + ATTN_MIRROR_GUARD_ROWS) * self.row <= self.k[li].len(),
                "投機読みの {len} 行目が確保の外 (層 {li}, cap {})",
                self.state.cap
            );
            assert!(
                (len + ATTN_MIRROR_GUARD_ROWS) * self.row <= self.v[li].len(),
                "投機読みの {len} 行目が確保の外 (層 {li}, cap {})",
                self.state.cap
            );
            have
        }

        /// task 181: 本番の `q4_attn_mirror_stage_dev` と同じ手順
        /// (cap 成長 → 行 [0, pos0) の確認 → チャンクの行だけ書く → note)。
        ///
        /// `Err` は本番と同じ意味で、「ミラーが行 [0, pos0) を持っていないので
        /// 呼び出し側は従来の `sync` (差分 H2D) へ戻る」。
        fn stage(&mut self, cache: &AttnCache, li: usize, pos0: usize) -> Result<usize, String> {
            let len = cache.len(li).unwrap();
            let gen = cache.gen(li).unwrap();
            if len < pos0 {
                return Err(format!("len {len} < pos0 {pos0}"));
            }
            let new_cap = q4_mirror_cap(len, self.state.cap);
            if new_cap > self.state.cap {
                self.grow(new_cap);
            }
            let have = q4_mirror_have(&self.state, li, pos0, gen);
            if have != pos0 {
                return Err(format!("行 [0,{pos0}) が無い (have={have}, gen={gen})"));
            }
            let row = self.row;
            let need = (len) * row;
            assert!(
                self.k[li].len() >= need,
                "ミラーのバッファが足りない (len={len}, cap={})",
                self.state.cap
            );
            self.k[li][pos0 * row..len * row]
                .copy_from_slice(&cache.k(li).unwrap()[pos0 * row..len * row]);
            self.v[li][pos0 * row..len * row]
                .copy_from_slice(&cache.v(li).unwrap()[pos0 * row..len * row]);
            q4_mirror_note(&mut self.state, li, len, gen);
            Ok(len - pos0)
        }

        fn assert_matches(&self, cache: &AttnCache, li: usize, what: &str) {
            let len = cache.len(li).unwrap();
            let row = self.row;
            assert_eq!(
                &self.k[li][..len * row],
                &cache.k(li).unwrap()[..len * row],
                "K ミラーがホストと違う ({what}, 層 {li})"
            );
            assert_eq!(
                &self.v[li][..len * row],
                &cache.v(li).unwrap()[..len * row],
                "V ミラーがホストと違う ({what}, 層 {li})"
            );
        }
    }

    fn row_values(base: f32, row: usize) -> Vec<f32> {
        (0..row).map(|i| base + i as f32 * 0.5).collect()
    }

    fn push_rows(cache: &mut AttnCache, n_layer: usize, row: usize, base: f32) {
        for l in 0..n_layer {
            let k = row_values(base + l as f32 * 100.0, row);
            let v = row_values(base + l as f32 * 100.0 + 50.0, row);
            cache.push(l, &k, &v).unwrap();
        }
    }

    /// cap の伸び方 (倍々・下限 1,024 行・必要量は一気に取る)。
    #[test]
    fn q4_mirror_cap_doubles_and_has_floor() {
        assert_eq!(q4_mirror_cap(1, 0), 1024);
        assert_eq!(q4_mirror_cap(1024, 0), 1024);
        assert_eq!(q4_mirror_cap(1025, 1024), 2048);
        assert_eq!(q4_mirror_cap(2000, 1024), 2048);
        assert_eq!(q4_mirror_cap(5000, 0), 8192);
        assert_eq!(q4_mirror_cap(65536, 32768), 65536);
        // 縮む必要は無い (grow-only)
        assert_eq!(q4_mirror_cap(10, 1024), 1024);
    }

    /// task 174: lab の失敗と同じ形 (len = cap = 4,096) を CPU ミラーで通す。
    /// flashdec は分割の終端の 1 行先 (= t_total 行目) を投機読みするので、
    /// 確保が cap + 1 行でないとこの長さで確保の外を踏む。SimMirror::sync が
    /// その行の存在を毎回 assert する。
    #[test]
    fn q4_mirror_sync_at_full_capacity_leaves_the_speculative_row_in_bounds() {
        let (n_layer, row) = (1usize, 4usize);
        let mut cache = AttnCache::new(n_layer, 2, 2, 8192).unwrap();
        let mut mir = SimMirror::new(n_layer, row);
        for i in 0..4096 {
            push_rows(&mut cache, n_layer, row, i as f32);
        }
        assert_eq!(mir.sync(&cache, 0), 0, "初回は全行シード");
        assert_eq!(mir.state.cap, 4096);
        assert_eq!(cache.len(0).unwrap(), mir.state.cap, "テスト前提: len = cap ちょうど");
        mir.assert_matches(&cache, 0, "cap ちょうど");
        // 4,097 行目で 8,192 段へ伸びる。既存 4,096 行は D2D 温存で送らない。
        push_rows(&mut cache, n_layer, row, 9999.0);
        assert_eq!(mir.sync(&cache, 0), 4096, "段の直後は 1 行だけ送る");
        assert_eq!(mir.state.cap, 8192);
        mir.assert_matches(&cache, 0, "段の直後");
    }

    /// task 181: ミラーの差分経路 (`q4_attn_mirror_stage_dev`) が**全行コピーと
    /// 同じ内容**になること。チャンク 1,000 行で 4,096 行まで積み、途中で
    /// cap が 1,024 -> 2,048 -> 4,096 と伸びる (プロンプトの真ん中で grow)、
    /// 最後は len = cap ちょうどで終わる。
    ///
    /// 最初のチャンクだけは台帳が空なので直接書きが `Err` になり、従来の差分
    /// H2D (= 全行) に戻る。2 チャンク目以降はチャンクの行だけを書き、毎回
    /// ホストの KV と行単位で一致することを確かめる。
    #[test]
    fn q4_mirror_delta_matches_full_copy_across_growth() {
        let (n_layer, row) = (2usize, 4usize);
        let chunk = 1000usize;
        let total = 4096usize;
        let mut cache = AttnCache::new(n_layer, 2, 2, 8192).unwrap();
        let mut mir = SimMirror::new(n_layer, row);
        let mut pos = 0usize;
        let mut fell_back = 0usize;
        let mut staged_rows = 0usize;
        while pos < total {
            let n = chunk.min(total - pos);
            for i in 0..n {
                push_rows(&mut cache, n_layer, row, (pos + i) as f32);
            }
            for l in 0..n_layer {
                match mir.stage(&cache, l, pos) {
                    Ok(nb) => {
                        assert_eq!(nb, n, "直接書きの行数がチャンクと違う");
                        staged_rows += nb;
                    }
                    Err(_) => {
                        // 従来経路: 台帳の差分だけ H2D する (初回は全行)。
                        mir.sync(&cache, l);
                        fell_back += 1;
                    }
                }
                mir.assert_matches(&cache, l, "差分経路");
            }
            pos += n;
        }
        assert_eq!(pos, total);
        assert_eq!(mir.state.cap, 4096, "len = cap ちょうどで終わること");
        // 台帳の have は pos0 で一致するので、**最初のチャンクから** チャンクの行
        // だけを直接書ける (最初は have(0) == 0 == pos0)。
        assert_eq!(fell_back, 0, "直接書きに戻った回数");
        assert_eq!(staged_rows, total * n_layer, "直接書きした行数");
        // 世代が変わった (clear / truncate 相当) あとは、pos0 > 0 の直接書きを
        // 断る (呼び出し側は従来の全行 H2D へ戻る)。
        cache.clear_all();
        for i in 0..600 {
            push_rows(&mut cache, n_layer, row, 100.0 + i as f32);
        }
        assert!(mir.stage(&cache, 0, 512).is_err(), "世代が違うのに直接書いた");
        // pos0 = 0 は「全行の書き直し」になるので通る (こちらも内容は一致する)。
        assert_eq!(mir.stage(&cache, 0, 0).unwrap(), 600);
        mir.assert_matches(&cache, 0, "clear 後");
    }

    /// task 181: 直接書きは「行 [0, pos0) をミラーが既に持っている」ときだけ。
    /// 空のミラーに pos0 > 0 を要求したら断る (呼び出し側は H2D 経路へ)。
    #[test]
    fn q4_mirror_delta_declines_when_prefix_is_missing() {
        let (n_layer, row) = (1usize, 4usize);
        let mut cache = AttnCache::new(n_layer, 2, 2, 4096).unwrap();
        let mut mir = SimMirror::new(n_layer, row);
        for i in 0..512 {
            push_rows(&mut cache, n_layer, row, i as f32);
        }
        // 空のミラーに pos0 = 512 は無理。
        assert!(mir.stage(&cache, 0, 512).is_err());
        // pos0 = 0 なら書ける (全行)。
        assert_eq!(mir.stage(&cache, 0, 0).unwrap(), 512);
        mir.assert_matches(&cache, 0, "全行");
        // 次のチャンクは先頭 512 行が載っているので、512 行だけ書けばよい。
        for i in 512..1024 {
            push_rows(&mut cache, n_layer, row, i as f32);
        }
        assert_eq!(mir.stage(&cache, 0, 512).unwrap(), 512);
        mir.assert_matches(&cache, 0, "差分");
        // pos0 = 0 を同じ世代で要求すると「全行を書き直す」になる (台帳の have は
        // 0 を返すため)。本番の `pf_full_qkv` は必ず pos0 = t_total - nt で呼び、
        // 書く行数はチャンクの nt 行なので、これは無害な書き直しになる。
        assert_eq!(mir.stage(&cache, 0, 0).unwrap(), 1024);
        mir.assert_matches(&cache, 0, "全行の書き直し");
    }

    /// task 133 の本丸: clear / clear_all / truncate_layer のあとで
    /// ミラーが古い行を使わないこと。実装と同じ q4_mirror_* を使う。
    #[test]
    fn q4_mirror_resyncs_after_truncate_clear_and_reuse() {
        let n_layer = 2usize;
        let row = 4usize;
        let mut cache = AttnCache::new(n_layer, 2, 2, 4096).unwrap();
        let mut mir = SimMirror::new(n_layer, row);

        // 1. 初回: 5 行を push → 全行シード
        for i in 0..5 {
            push_rows(&mut cache, n_layer, row, i as f32);
        }
        for l in 0..n_layer {
            assert_eq!(mir.sync(&cache, l), 0, "初回は全行シード (層 {l})");
            mir.assert_matches(&cache, l, "初回");
        }
        // 2 回目は差分ゼロ
        assert_eq!(mir.sync(&cache, 0), 5, "追記なしなら送らない");

        // 2. truncate → 同じ位置に別の行が入る (spec rollback)
        cache.truncate_layer(0, 3).unwrap();
        push_rows(&mut cache, n_layer, row, 900.0);
        let have = mir.sync(&cache, 0);
        assert_eq!(have, 0, "truncate 後は世代が変わるので全行シード");
        mir.assert_matches(&cache, 0, "truncate 後");
        // 触っていない層は差分のまま (ミラーが持っている 5 行は送らない)
        assert_eq!(mir.sync(&cache, 1), 5, "他層は 1 行追記だけ");

        // 3. clear_all → 同じ長さで別内容 (prefix reuse / model reuse の reset)。
        //    世代を見ない実装だと「len が同じだから」で古い行を残す。
        let before = cache.k(0).unwrap().to_vec();
        cache.clear_all();
        for i in 0..6 {
            push_rows(&mut cache, n_layer, row, 3000.0 + i as f32);
        }
        for l in 0..n_layer {
            let have = mir.sync(&cache, l);
            assert_eq!(have, 0, "clear_all 後は全行シード (層 {l})");
            mir.assert_matches(&cache, l, "clear_all 後");
        }
        assert_ne!(
            cache.k(0).unwrap()[..row],
            before[..row],
            "テスト前提: 内容が変わっている"
        );

        // 4. 単層 clear → 2 行だけ積み直す
        cache.clear(1).unwrap();
        push_rows(&mut cache, n_layer, row, 5000.0);
        assert_eq!(mir.sync(&cache, 1), 0, "clear(層) 後は全行シード");
        mir.assert_matches(&cache, 1, "clear(層) 後");

        // 5. cap 成長: 1,024 行を超えると倍に伸びるが、既存行は D2D 温存 (再シードしない)
        let need = 1100usize;
        while cache.len(0).unwrap() < need {
            let base = 7000.0 + cache.len(0).unwrap() as f32;
            push_rows(&mut cache, n_layer, row, base);
        }
        let have = mir.sync(&cache, 0);
        assert!(have > 0, "cap 成長で全行再シードしてはいけない (have={have})");
        assert_eq!(mir.state.cap, 2048, "cap は倍々で伸びる");
        mir.assert_matches(&cache, 0, "cap 成長後");
        // 他層も D2D で温存されている (1 行のまま。全行再シードなら 0 が返る)
        assert_eq!(mir.sync(&cache, 1), 1, "他層も温存");
        mir.assert_matches(&cache, 1, "cap 成長後の他層");
    }

    /// ゲートの解釈 (未設定は既定・明示の偽値だけ OFF)。
    #[test]
    fn q4_gpu_attn_gate_parsing() {
        assert_eq!(parse_q4_gpu_attn(None), Q4_GPU_ATTN_DEFAULT);
        assert!(!parse_q4_gpu_attn(Some("0")));
        assert!(!parse_q4_gpu_attn(Some("off")));
        assert!(!parse_q4_gpu_attn(Some("false")));
        assert!(parse_q4_gpu_attn(Some("1")));
        assert!(parse_q4_gpu_attn(Some("on")));
        assert!(parse_q4_gpu_attn(Some("true")));
        assert_eq!(parse_q4_gpu_attn_min(None), Q4_GPU_ATTN_MIN_DEFAULT);
        assert_eq!(parse_q4_gpu_attn_min(Some("1234")), 1234);
        assert_eq!(parse_q4_gpu_attn_min(Some("x")), Q4_GPU_ATTN_MIN_DEFAULT);
        assert_eq!(q4_attn_mirror_limit_from(None), 3584 * 1024 * 1024);
        assert_eq!(q4_attn_mirror_limit_from(Some(16)), 16 * 1024 * 1024);
    }

    /// task 134: prefill ゲートの解釈としきい値 (純関数)。
    /// 短いプロンプト・小さいチャンクを CPU に残す 2 段の下限を見る。
    #[test]
    fn q4_pf_gate_values_and_thresholds() {
        // 既定は on (lab t134a で目標達成)。明示の偽値だけ off。
        assert_eq!(parse_q4_gpu_attn_pf(None), Q4_GPU_ATTN_PF_DEFAULT);
        assert!(Q4_GPU_ATTN_PF_DEFAULT, "lab の判定で既定 on");
        assert!(parse_q4_gpu_attn_pf(None), "未設定は既定 (on)");
        assert!(!parse_q4_gpu_attn_pf(Some("0")));
        assert!(!parse_q4_gpu_attn_pf(Some("off")));
        assert!(!parse_q4_gpu_attn_pf(Some("false")));
        assert!(parse_q4_gpu_attn_pf(Some("1")));
        assert!(parse_q4_gpu_attn_pf(Some("on")));
        assert!(parse_q4_gpu_attn_pf(Some("true")));
        assert_eq!(
            parse_usize_default(None, Q4_GPU_ATTN_PF_MIN_DEFAULT),
            Q4_GPU_ATTN_PF_MIN_DEFAULT
        );
        assert_eq!(parse_usize_default(Some("512"), 256), 512);
        assert_eq!(parse_usize_default(Some("x"), 256), 256);
        assert_eq!(q4_attn_mirror_limit_from(None), 3584 * 1024 * 1024);

        // (on, n, t_total, min_rows, min_kv)
        let g = |on: bool, n: usize, t: usize, mr: usize, mk: usize| {
            q4_attn_pf_gate_values(on, n, t, mr, mk)
        };
        // 短いプロンプト (26 token) は既定のしきい値で CPU。env ゲート自身は
        // しきい値だけを見るので、0 を渡せば通る (26 token を弾くのは構造ゲート
        // `attn_prefill_gpu_supported` の n >= gemm_min_m() 側)。
        assert!(!g(false, 512, 4096, 256, 512), "off で通った");
        assert!(g(true, 26, 26, 0, 0), "しきい値 0 なら env ゲートは通す");
        assert!(!g(true, 26, 26, 256, 512), "短いプロンプトが既定のしきい値で通った");
        assert!(!g(true, 512, 256, 256, 512), "文脈長が下限未満で通った");
        assert!(!g(true, 128, 4096, 256, 512), "チャンク行数が下限未満で通った");
        assert!(g(true, 512, 1024, 256, 512), "通常のチャンクが弾かれた");
        assert!(g(true, 256, 512, 256, 512), "両方の下限ちょうどが弾かれた");
        assert!(!g(true, 0, 512, 0, 0), "n = 0 を通した");
        assert!(!g(true, 512, 100, 0, 0), "t_total < n を通した (位置の整合)");
        // 既定値の関係: task 135 で行数の下限は attention 専用の値になった
        // (GEMM の gemm_min_m() = 256 は借りない)。
        assert_eq!(Q4_GPU_ATTN_PF_MIN_DEFAULT, ATTN_Q4_PF_MIN_ROWS);
        assert_eq!(Q4_GPU_ATTN_PF_MIN_DEFAULT, 8);
        assert!(ATTN_Q4_PF_MIN_ROWS < gemm_min_m());
        assert!(Q4_GPU_ATTN_PF_MIN_KV_DEFAULT > Q4_GPU_ATTN_MIN_DEFAULT);
    }

    /// task 181: `DENNO_PF_D2H_DIRECT` の解釈 (既定 on。明示的な偽値だけ off)。
    #[test]
    fn pf_d2h_direct_gate_parsing() {
        assert!(pf_d2h_direct_from_value(None), "未設定は既定 on (lab t181b で採用)");
        for v in ["0", "off", "false", "no", "", " OFF "] {
            assert!(!pf_d2h_direct_from_value(Some(v)), "{v}");
        }
        for v in ["1", "on", "true", "yes", " ON "] {
            assert!(pf_d2h_direct_from_value(Some(v)), "{v}");
        }
        let before = pf_d2h_direct_on();
        set_pf_d2h_direct_for_test(Some(true));
        assert!(pf_d2h_direct_on());
        set_pf_d2h_direct_for_test(Some(false));
        assert!(!pf_d2h_direct_on());
        set_pf_d2h_direct_for_test(None);
        assert_eq!(pf_d2h_direct_on(), before);
    }

    /// task 181: `DENNO_PF_MIRROR_DELTA` の解釈 (既定 on。明示的な偽値だけ off)。
    #[test]
    fn pf_mirror_delta_gate_parsing() {
        assert!(pf_mirror_delta_from_value(None), "未設定は既定 on (lab t181b で採用)");
        for v in ["0", "off", "false", "no", "", " OFF "]
        {
            assert!(!pf_mirror_delta_from_value(Some(v)), "{v}");
        }
        for v in ["1", "on", "true", "yes", " ON "] {
            assert!(pf_mirror_delta_from_value(Some(v)), "{v}");
        }
        // 強制上書き (テストは 1 プロセス並列なので set/戻しを対で行う)。
        let before = pf_mirror_delta_on();
        set_pf_mirror_delta_for_test(Some(true));
        assert!(pf_mirror_delta_on());
        set_pf_mirror_delta_for_test(Some(false));
        assert!(!pf_mirror_delta_on());
        set_pf_mirror_delta_for_test(None);
        assert_eq!(pf_mirror_delta_on(), before);
    }

    /// task 134: prefill がチャンクを積んだ後、**最初の decode トークンは
    /// 1 行だけ**送る (全行シードに戻らない)。prefill と decode が同じ台帳
    /// (`q4_mirror_*`) を共有していることの固定。
    #[test]
    fn q4_mirror_prefill_then_decode_syncs_one_row() {
        let mut st = Q4MirrorState {
            cap: 0,
            len: vec![0; 2],
            gen: vec![0; 2],
        };
        // チャンク 1 (512 行) / チャンク 2 (1,024 行) を prefill の規則で積む。
        assert_eq!(q4_mirror_have(&st, 0, 512, 0), 0, "初回は全行シード");
        q4_mirror_note(&mut st, 0, 512, 0);
        assert_eq!(q4_mirror_have(&st, 0, 1024, 0), 512, "2 チャンク目は追記 512 行");
        q4_mirror_note(&mut st, 0, 1024, 0);
        // prefill 直後: 全行そろっている。
        assert_eq!(q4_mirror_have(&st, 0, 1024, 0), 1024);
        // 最初の decode トークン (len = 1,025) は 1 行だけ。
        assert_eq!(q4_mirror_have(&st, 0, 1025, 0), 1024);
        // 使っていない層 (ssm) は 0 のまま = 確保もしない。
        assert_eq!(q4_mirror_have(&st, 1, 1025, 0), 0);
        // clear_all / truncate 相当 (世代 +1) は全行シードに戻る。
        assert_eq!(q4_mirror_have(&st, 0, 1025, 1), 0);
        // cap は t_total で倍々に育つ (prefill でも同じ規則)。
        assert_eq!(q4_mirror_cap(1024, 0), 1024);
        assert_eq!(q4_mirror_cap(1025, 1024), 2048);
    }

    /// task 174: 容量の段 (4,096 -> 8,192 -> 16,384) をまたぐ長さで、
    /// (a) cap が必ず len 以上、(b) 成長しても**既に送った行数を捨てない**
    /// (D2D で温存する行数 = keep)、(c) 追記ぶん [have, len) が連続している、
    /// (d) flash decoding の分割と部分和バッファが len に収まる、を固定する。
    ///
    /// lab t173a / t172a の decode は 4,096 を超えた直後 (len=4,097) ではなく
    /// **n_split が 63 -> 64 になる len=4,033** で error 700 を出したので、
    /// 分割の側も同じ長さで見る。
    #[test]
    fn q4_mirror_growth_across_capacity_steps_keeps_rows() {
        let (heads, n_kv, dh) = (24usize, 2usize, 256usize);
        let kv_dim = n_kv * dh;
        // 4,096 段の手前 / ちょうど / 直後、次の段、最終段。
        let lens = [4095usize, 4096, 4097, 8192, 16384];
        let mut st = Q4MirrorState {
            cap: 0,
            len: vec![0; 2],
            gen: vec![0; 2],
        };
        let mut prev_len = 0usize;
        for &len in &lens {
            let old_cap = st.cap;
            let new_cap = q4_mirror_cap(len, old_cap);
            assert!(new_cap >= len, "cap {new_cap} < len {len}");
            assert!(new_cap.is_power_of_two(), "cap {new_cap} が 2 の冪でない");
            assert!(new_cap >= 1024, "cap {new_cap} が下限 1,024 未満");
            // task 174: flashdec は分割の終端の 1 行先を投機読みする。確保は
            // cap + 余白なので、len が cap とちょうど等しくても (4,096) 読める。
            assert!(
                q4_mirror_alloc_rows(new_cap) > len,
                "len={len}, cap={new_cap}: 投機読み 1 行ぶんの余白が無い"
            );
            assert_eq!(q4_mirror_alloc_rows(new_cap), new_cap + 1);
            // 成長時の D2D で温存する行数 (q4_attn_grow の keep と同じ式)。
            let keep = st.len[0].min(new_cap);
            assert_eq!(
                keep, st.len[0],
                "成長で送信済みの行が落ちる (len={len}, cap {old_cap} -> {new_cap})"
            );
            st.cap = new_cap;
            // この長さまでは全行がデバイスにある、という台帳にする。
            let have = q4_mirror_have(&st, 0, len, 0);
            assert_eq!(have, prev_len.min(len), "have が前回の行数と違う (len={len})");
            assert!(have <= len);
            // 実際に H2D する範囲は [have, len) で、必ずバッファ内に収まる。
            let end = len;
            assert!(end * kv_dim * 4 <= st.cap * kv_dim * 4, "len={len} がバッファ外");
            q4_mirror_note(&mut st, 0, len, 0);
            assert_eq!(q4_mirror_have(&st, 0, len, 0), len);
            assert_eq!(q4_mirror_have(&st, 0, len + 1, 0), len, "次トークンは追記 1 行");
            prev_len = len;
        }
        // 段をまたいだ直後 (4,096 -> 4,097) でも全行を送り直さない。
        let mut st2 = Q4MirrorState {
            cap: 4096,
            len: vec![4096; 1],
            gen: vec![0; 1],
        };
        let cap2 = q4_mirror_cap(4097, st2.cap);
        assert_eq!(cap2, 8192, "4,097 で 8,192 段へ上がる");
        assert_eq!(q4_mirror_have(&st2, 0, 4097, 0), 4096, "4,096 行を温存する");
        st2.cap = cap2;
        // 使っていない層 (ssm) は 0 のまま = 確保もしない。
        assert_eq!(q4_mirror_have(&st2, 1, 4097, 0), 0);

        // flash decoding の分割と部分和バッファ: どの長さでも
        // 最後の part 添字が heads * n_split * (dh + 2) - 1 に収まる。
        for &len in &[1usize, 4032, 4033, 4095, 4096, 4097, 8192, 16384] {
            let (n_split, chunk) = q4_fdec_split(len, ATTN_FLASHDEC_CHUNK_DEFAULT);
            assert!(n_split >= 1 && n_split <= ATTN_FDEC_SPLIT_MAX);
            assert_eq!(chunk, len.div_ceil(n_split), "chunk がカーネル式と違う");
            assert!(n_split * chunk >= len, "区間が足りない (len={len})");
            // カーネルの書き込み位置 (kh, sp, hh) の最大値。
            let entries = heads * n_split;
            let last = (entries - 1) * (dh + ATTN_FDEC_PART_EXTRA) + dh + ATTN_FDEC_PART_EXTRA - 1;
            let part_floats = heads * n_split * (dh + ATTN_FDEC_PART_EXTRA);
            assert_eq!(last, part_floats - 1, "part の末尾が 1 要素ずれる (len={len})");
        }
        // task 174: 失敗の直接原因。flashdec の 2 段パイプラインは `++j` のあと
        // `if (j < jhi)` で `j + 1` を先読みするので、**分割の終端 jhi の 1 行先**
        // まで読む。最終分割では jhi = t_total なので、確保が t_total 行ちょうど
        // (len = cap = 4,096) だと確保の外を踏む。ここではその最大行を数える。
        for &len in &[4033usize, 4096, 8192] {
            let (n_split, _) = q4_fdec_split(len, ATTN_FLASHDEC_CHUNK_DEFAULT);
            let chunk = len.div_ceil(n_split);
            let mut max_read = 0usize;
            for sp in 0..n_split {
                let jlo = sp * chunk;
                let jhi = (jlo + chunk).min(len);
                let mut j = jlo;
                if j < jhi {
                    max_read = max_read.max(j);
                }
                if j + 1 < jhi {
                    max_read = max_read.max(j + 1);
                }
                while j < jhi {
                    j += 1;
                    if j < jhi {
                        max_read = max_read.max(j + 1); // A の先読み
                    }
                    if j >= jhi {
                        break;
                    }
                    j += 1;
                    if j < jhi {
                        max_read = max_read.max(j + 1); // B の先読み
                    }
                }
            }
            assert!(max_read <= len, "先読みが len を超えた (len={len}, max={max_read})");
            // 最終分割に 2 行以上あれば、その先読みは t_total 行目 (= len) を踏む。
            if len - (n_split - 1) * chunk >= 2 {
                assert_eq!(max_read, len, "先読みの最大行が len でない (len={len})");
            }
            assert!(
                q4_mirror_alloc_rows(len) > max_read,
                "len={len}: 先読みの {max_read} 行目が確保の外"
            );
        }

        // n_split が 64 に上がる境目 (lab の crash 点) の分割。
        assert_eq!(q4_fdec_split(4032, 64), (63, 64));
        assert_eq!(q4_fdec_split(4033, 64), (64, 64));
        assert_eq!(q4_fdec_split(4097, 64), (65, 64));
    }

    /// GPU が要る実機テスト (task 134)。prefill の 1 チャンクを CPU 参照と
    /// 突き合わせ、その直後の decode 1 トークンも同じミラーで合うことを見る。
    /// `cargo test --release --features cuda -p denno-engine --lib
    /// q4_gpu_attn_prefill -- --ignored --test-threads=1` でだけ回す。
    #[test]
    #[ignore = "GPU が必要"]
    fn q4_gpu_attn_prefill_matches_cpu_and_decode() {
        let _g = test_lock()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        enable_for_tests();
        let _ = CudaCtx::new().expect("GPU が無い");
        set_q4_gpu_attn_pf_for_test(Some(true));
        set_q4_gpu_attn_pf_min_for_test(Some(0), Some(0));
        set_q4_gpu_attn_for_test(Some(true));
        set_q4_gpu_attn_min_for_test(Some(0));

        // dh >= ATTN_PF_BQ (= 64) がカーネルの下限。smem は 23.4 KB で opt-in 不要。
        let (heads, n_kv, dh, n_layer, max_ctx) = (4usize, 2usize, 64usize, 1usize, 1024usize);
        let group = heads / n_kv;
        let kv_dim = n_kv * dh;
        let mut cache = AttnCache::new(n_layer, n_kv, dh, max_ctx).unwrap();
        let mut lcg = {
            let mut s = 999u64;
            move || {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 30) as f32) - 1.0
            }
        };
        let push = |cache: &mut AttnCache, lcg: &mut dyn FnMut() -> f32| {
            let k: Vec<f32> = (0..kv_dim).map(|_| lcg()).collect();
            let v: Vec<f32> = (0..kv_dim).map(|_| lcg()).collect();
            cache.push(0, &k, &v).unwrap();
        };
        let scale = 1.0 / (dh as f32).sqrt();

        // 2 チャンク (300 + 300 行)。n >= gemm_min_m() = 256 が必要。
        let mut base = 0usize;
        for chunk in [300usize, 300] {
            for _ in 0..chunk {
                push(&mut cache, &mut lcg);
            }
            let t_total = base + chunk;
            let qs: Vec<f32> = (0..chunk * heads * dh).map(|_| lcg()).collect();
            let call = Q4AttnPfCall {
                cache_id: cache.id(),
                n_layer,
                layer: 0,
                layer_gen: cache.gen(0).unwrap(),
                t_total,
                pos0: base,
                n: chunk,
                kv_dim,
                heads,
                group,
                dh,
                scale,
                q: &qs,
                k: cache.k(0).unwrap(),
                v: cache.v(0).unwrap(),
            };
            let got = q4_attn_prefill(&call).expect("prefill の GPU 経路が発動していない");
            assert_eq!(got.len(), chunk * heads * dh);
            for t in 0..chunk {
                let qrow = &qs[t * heads * dh..(t + 1) * heads * dh];
                let want = cpu_reference(
                    qrow,
                    cache.k(0).unwrap(),
                    cache.v(0).unwrap(),
                    base + t + 1,
                    kv_dim,
                    heads,
                    group,
                    dh,
                    scale,
                );
                for i in 0..heads * dh {
                    let tol = 1e-4 + 1e-5 * want[i].abs();
                    assert!(
                        (got[t * heads * dh + i] - want[i]).abs() <= tol,
                        "chunk {chunk} 行 {t} 要素 {i}: GPU {} vs CPU {} (許容 {tol})",
                        got[t * heads * dh + i],
                        want[i]
                    );
                }
            }
            base = t_total;
        }

        // prefill の直後に decode 1 トークン (全行シードし直さずに合うか)。
        push(&mut cache, &mut lcg);
        let len = base + 1;
        let q: Vec<f32> = (0..heads * dh).map(|_| lcg()).collect();
        let dcall = Q4AttnCall {
            cache_id: cache.id(),
            n_layer,
            layer: 0,
            layer_gen: cache.gen(0).unwrap(),
            len,
            kv_dim,
            heads,
            group,
            dh,
            scale,
            q: &q,
            k: cache.k(0).unwrap(),
            v: cache.v(0).unwrap(),
        };
        let got = q4_attn_decode(&dcall).expect("decode の GPU 経路が発動していない");
        let want = cpu_reference(
            &q,
            cache.k(0).unwrap(),
            cache.v(0).unwrap(),
            len,
            kv_dim,
            heads,
            group,
            dh,
            scale,
        );
        for i in 0..heads * dh {
            let tol = 1e-4 + 1e-5 * want[i].abs();
            assert!(
                (got[i] - want[i]).abs() <= tol,
                "prefill 直後の decode 要素 {i}: GPU {} vs CPU {} (許容 {tol})",
                got[i],
                want[i]
            );
        }

        set_q4_gpu_attn_pf_for_test(None);
        set_q4_gpu_attn_pf_min_for_test(None, None);
        set_q4_gpu_attn_for_test(None);
        set_q4_gpu_attn_min_for_test(None);
    }

    /// ミラーの VRAM 上限: 65,536 行でも 3.3 GB 未満 (使用層 12 本のとき)。
    /// 48 層ぶん確保すると 4 倍になり上限を割る — q4_attn_grow が未使用層を
    /// 飛ばす理由をここで固定する。
    #[test]
    fn q4_mirror_bytes_bounds_vram_at_max_ctx() {
        let (cap, kv_dim, full_layers, n_layer) = (65_536usize, 512usize, 12usize, 48usize);
        let used = q4_mirror_bytes(cap, kv_dim, full_layers);
        // task 174: 実確保は投機読みの余白 1 行を含む (65,537 行)。
        assert_eq!(
            used, 3_221_274_624,
            "65,536 行 (+ 余白 1 行) x 512 x 4 B x K/V x 12 層"
        );
        assert!(used < 3_300_000_000, "3.3 GB 未満でない: {used}");
        assert!(
            used <= q4_attn_mirror_limit_from(None),
            "既定のミラー上限 ({}) を超える: {used}",
            q4_attn_mirror_limit_from(None)
        );
        // 未使用層まで確保すると 4 倍 (= 12 GiB) で上限を割る — これを避けるのが
        // q4_attn_grow のスキップ
        let all = q4_mirror_bytes(cap, kv_dim, n_layer);
        assert_eq!(all, 4 * used);
        assert!(all > q4_attn_mirror_limit_from(None));
    }

    /// GPU が要る実機テスト。`cargo test --release --features cuda -p denno-engine
    /// --lib q4_attn -- --ignored --test-threads=1` でだけ回す。
    ///
    /// dno_attention の出力を CPU 参照と突き合わせ (絶対 1e-4 + 相対 1e-5)、
    /// truncate / clear のあとも一致することを見る (ミラーが古い行を混ぜない)。
    #[test]
    #[ignore = "GPU が必要"]
    fn q4_gpu_attn_matches_cpu_after_truncate_and_clear() {
        let _g = test_lock()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        enable_for_tests();
        let _ = CudaCtx::new().expect("GPU が無い");
        set_q4_gpu_attn_for_test(Some(true));
        set_q4_gpu_attn_min_for_test(Some(0));

        let (heads, n_kv, dh, n_layer, max_ctx) = (4usize, 2usize, 8usize, 1usize, 64usize);
        let group = heads / n_kv;
        let kv_dim = n_kv * dh;
        let mut cache = AttnCache::new(n_layer, n_kv, dh, max_ctx).unwrap();
        let mut lcg = {
            let mut s = 12345u64;
            move || {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 30) as f32) - 1.0
            }
        };
        let push = |cache: &mut AttnCache, lcg: &mut dyn FnMut() -> f32| {
            let k: Vec<f32> = (0..kv_dim).map(|_| lcg()).collect();
            let v: Vec<f32> = (0..kv_dim).map(|_| lcg()).collect();
            cache.push(0, &k, &v).unwrap();
        };
        for _ in 0..37 {
            push(&mut cache, &mut lcg);
        }
        let q: Vec<f32> = (0..heads * dh).map(|_| lcg()).collect();
        let scale = 0.0625f32;

        let check = |cache: &AttnCache, what: &str| {
            let len = cache.len(0).unwrap();
            let call = Q4AttnCall {
                cache_id: cache.id(),
                n_layer,
                layer: 0,
                layer_gen: cache.gen(0).unwrap(),
                len,
                kv_dim,
                heads,
                group,
                dh,
                scale,
                q: &q,
                k: cache.k(0).unwrap(),
                v: cache.v(0).unwrap(),
            };
            let got = q4_attn_decode(&call).expect("GPU 経路が発動していない");
            let want = cpu_reference(&q, cache.k(0).unwrap(), cache.v(0).unwrap(), len, kv_dim, heads, group, dh, scale);
            for i in 0..got.len() {
                let tol = 1e-4 + 1e-5 * want[i].abs();
                assert!(
                    (got[i] - want[i]).abs() <= tol,
                    "{what}: 要素 {i} が GPU {} と CPU {} で許容差 {tol} を超えた",
                    got[i],
                    want[i]
                );
            }
        };

        check(&cache, "初回 (37 行)");
        // truncate して別の行を push (spec rollback 相当)
        cache.truncate_layer(0, 20).unwrap();
        push(&mut cache, &mut lcg);
        check(&cache, "truncate 後 (21 行)");
        // clear_all して同じ長さで別内容 (reset / prefix reuse 相当)
        cache.clear_all();
        for _ in 0..21 {
            push(&mut cache, &mut lcg);
        }
        check(&cache, "clear_all 後 (21 行)");

        set_q4_gpu_attn_for_test(None);
        set_q4_gpu_attn_min_for_test(None);
    }

    /// task 135: decode カーネルの選択と t 分割数の解釈 (純関数)。
    #[test]
    fn q4_attn_kernel_and_split_parsing() {
        assert_eq!(parse_q4_gpu_attn_kernel(None), Q4_GPU_ATTN_KERNEL_DEFAULT);
        assert_eq!(
            parse_q4_gpu_attn_kernel(Some("decode")),
            Q4AttnKernel::Decode
        );
        assert_eq!(parse_q4_gpu_attn_kernel(Some("dno")), Q4AttnKernel::Decode);
        assert_eq!(parse_q4_gpu_attn_kernel(Some("pf")), Q4AttnKernel::Prefill);
        assert_eq!(
            parse_q4_gpu_attn_kernel(Some("prefill")),
            Q4AttnKernel::Prefill
        );
        assert_eq!(parse_q4_gpu_attn_kernel(Some("split")), Q4AttnKernel::Split);
        assert_eq!(parse_q4_gpu_attn_kernel(Some("dec")), Q4AttnKernel::Split);
        assert_eq!(
            parse_q4_gpu_attn_kernel(Some("x")),
            Q4_GPU_ATTN_KERNEL_DEFAULT
        );
        assert_eq!(parse_q4_gpu_attn_split(None), Q4_GPU_ATTN_SPLIT_DEFAULT);
        assert_eq!(parse_q4_gpu_attn_split(Some("16")), 16);
        assert_eq!(parse_q4_gpu_attn_split(Some("1")), 1);
        assert_eq!(parse_q4_gpu_attn_split(Some("0")), 1, "0 は 1 に丸める");
        assert_eq!(parse_q4_gpu_attn_split(Some("9999")), 64, "上限は 64");
        assert_eq!(
            parse_q4_gpu_attn_split(Some("x")),
            Q4_GPU_ATTN_SPLIT_DEFAULT
        );
    }

    /// task 135: t の固定分割 (ホストとカーネルが同じ式を使う)。
    #[test]
    fn q4_dec_split_is_fixed_and_covers_all_keys() {
        assert_eq!(q4_dec_split(14018, 8), (8, 1753));
        assert_eq!(q4_dec_split(3954, 8), (8, 495));
        assert_eq!(q4_dec_split(28, 8), (7, 4));
        assert_eq!(q4_dec_split(1, 8), (1, 1));
        assert_eq!(q4_dec_split(0, 8), (1, 1));
        // 昇順・重複なしで [0, t_total) を覆い、空の分割を作らない。
        for t in [1usize, 2, 7, 8, 9, 17, 64, 65, 1000, 14018] {
            for n in [1usize, 2, 3, 8, 16, 64] {
                let (ns, chunk) = q4_dec_split(t, n);
                assert!(ns >= 1 && chunk >= 1, "t={t} n={n}");
                assert!((ns - 1) * chunk < t, "最後の分割が空 (t={t} n={n})");
                assert!(ns * chunk >= t, "区間が足りない (t={t} n={n})");
                assert!(ns <= n, "分割数が要求を超えた (t={t} n={n})");
            }
        }
    }

    /// task 135: attention 専用の構造下限 (行数) と、qwen35 が従来の
    /// `gemm_min_m()` のままであること。
    #[test]
    fn attn_prefill_shape_min_rows_for_qwen4exp_only() {
        // dh=64 は 23.4KB で 48KB 既定上限に収まる (dh=256 は実機の opt-in が要る)。
        let dh = 64usize;
        // attention 専用の下限 8 行: next turn の 19 行・尾チャンクの 194/242 行が通る。
        assert!(attn_prefill_shape_ok(19, dh, 3954, ATTN_Q4_PF_MIN_ROWS));
        assert!(attn_prefill_shape_ok(194, dh, 14018, ATTN_Q4_PF_MIN_ROWS));
        assert!(attn_prefill_shape_ok(242, dh, 3826, ATTN_Q4_PF_MIN_ROWS));
        assert!(attn_prefill_shape_ok(8, dh, 512, ATTN_Q4_PF_MIN_ROWS));
        assert!(!attn_prefill_shape_ok(7, dh, 3954, ATTN_Q4_PF_MIN_ROWS));
        // 従来の GEMM 下限 (gemm_min_m() = 256) では同じ行数が落ちる = 変更点。
        assert!(!attn_prefill_shape_ok(19, dh, 3954, gemm_min_m()));
        assert!(!attn_prefill_shape_ok(242, dh, 3826, gemm_min_m()));
        // 形状の制約は従来どおり (dh >= ATTN_PF_BQ・t_total >= n・n > 0)。
        assert!(!attn_prefill_shape_ok(19, 32, 3954, ATTN_Q4_PF_MIN_ROWS));
        assert!(!attn_prefill_shape_ok(19, dh, 18, ATTN_Q4_PF_MIN_ROWS));
        assert!(!attn_prefill_shape_ok(0, dh, 3954, ATTN_Q4_PF_MIN_ROWS));
        // 本番 qwen4exp の dh=256 は 84.8KB (実機では CudaCtx 生成時に opt-in)。
        assert_eq!(attn_pf_sh_f(256) * 4, 84_800);
    }

    /// task 135: 分割 attention のアルゴリズム (部分 online softmax + combine) を
    /// CPU で書き下し、通常の softmax 参照と一致することを見る。カーネル自体は
    /// GPU が要る (`q4_attn_decode_split_matches_cpu` は --ignored)。
    fn split_attention_cpu_ref(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        len: usize,
        kv_dim: usize,
        heads: usize,
        group: usize,
        dh: usize,
        scale: f32,
        n_split: usize,
    ) -> Vec<f32> {
        let (ns, chunk) = q4_dec_split(len, n_split);
        let mut out = vec![0f32; heads * dh];
        for h in 0..heads {
            let off = (h / group) * dh;
            let mut part = vec![0f32; ns * (dh + 2)];
            for sp in 0..ns {
                let jlo = sp * chunk;
                let jhi = ((sp + 1) * chunk).min(len);
                let mut acc = vec![0f32; dh];
                let mut m_run = f32::NEG_INFINITY;
                let mut l_run = 0f32;
                let mut j0 = jlo;
                while j0 < jhi {
                    let nk = (jhi - j0).min(ATTN_DEC_BK);
                    let mut ss = [0f32; ATTN_DEC_BK];
                    for kk in 0..nk {
                        let krow = &k[(j0 + kk) * kv_dim + off..][..dh];
                        let mut s = 0f32;
                        for d in 0..dh {
                            s += q[h * dh + d] * krow[d];
                        }
                        ss[kk] = s * scale;
                    }
                    let mut mx = m_run;
                    for kk in 0..nk {
                        mx = mx.max(ss[kk]);
                    }
                    let corr = ((m_run - mx) * std::f32::consts::LOG2_E).exp2();
                    let mut l = l_run * corr;
                    let mut pp = [0f32; ATTN_DEC_BK];
                    for kk in 0..nk {
                        pp[kk] = ((ss[kk] - mx) * std::f32::consts::LOG2_E).exp2();
                        l += pp[kk];
                    }
                    for a in acc.iter_mut() {
                        *a *= corr;
                    }
                    for kk in 0..nk {
                        let vrow = &v[(j0 + kk) * kv_dim + off..][..dh];
                        for d in 0..dh {
                            acc[d] = pp[kk].mul_add(vrow[d], acc[d]);
                        }
                    }
                    l_run = l;
                    m_run = mx;
                    j0 += ATTN_DEC_BK;
                }
                part[sp * (dh + 2)] = m_run;
                part[sp * (dh + 2) + 1] = l_run;
                part[sp * (dh + 2) + 2..][..dh].copy_from_slice(&acc);
            }
            // combine (split 昇順)
            let mut mg = f32::NEG_INFINITY;
            for sp in 0..ns {
                mg = mg.max(part[sp * (dh + 2)]);
            }
            for d in 0..dh {
                let mut denom = 0f32;
                let mut num = 0f32;
                for sp in 0..ns {
                    let p = &part[sp * (dh + 2)..][..dh + 2];
                    let e = ((p[0] - mg) * std::f32::consts::LOG2_E).exp2();
                    denom = p[1].mul_add(e, denom);
                    num = p[2 + d].mul_add(e, num);
                }
                out[h * dh + d] = if denom > 0.0 { num / denom } else { 0.0 };
            }
        }
        out
    }

    /// task 135: 分割数・分割位置を変えても dense な softmax と同じ値になる
    /// (アルゴリズムの検証。カーネルの丸め順とは無関係)。
    #[test]
    fn q4_attn_split_algorithm_matches_dense_softmax() {
        let (heads, n_kv, dh) = (4usize, 2usize, 16usize);
        let group = heads / n_kv;
        let kv_dim = n_kv * dh;
        let scale = 1.0 / (dh as f32).sqrt();
        for len in [1usize, 7, 8, 9, 33, 257] {
            let q = pseudo_random(heads * dh, 7 + len as u32);
            let k = pseudo_random(len * kv_dim, 11 + len as u32);
            let v = pseudo_random(len * kv_dim, 13 + len as u32);
            let want = cpu_reference(&q, &k, &v, len, kv_dim, heads, group, dh, scale);
            for n_split in [1usize, 2, 3, 8, 16] {
                let got = split_attention_cpu_ref(
                    &q, &k, &v, len, kv_dim, heads, group, dh, scale, n_split,
                );
                let maxd = got
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    maxd < 1e-5,
                    "len={len} n_split={n_split} の分割和が dense softmax と違う: maxabs={maxd}"
                );
            }
        }
    }

    /// task 160: flash decoding のアルゴリズム (kv head ごとに warp が q head 1 本を
    /// 持ち、key を 1 本ずつ online softmax で畳んで split 昇順に combine する) を
    /// CPU で書き下したもの。`kernels_attn_dec.cu` と同じ式・同じ分割
    /// ([`q4_fdec_split`])・同じ part の添字を使う。
    #[allow(clippy::too_many_arguments)]
    fn flashdec_attention_cpu_ref(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        len: usize,
        kv_dim: usize,
        heads: usize,
        group: usize,
        dh: usize,
        scale: f32,
        chunk_keys: usize,
    ) -> Vec<f32> {
        let (ns, chunk) = q4_fdec_split(len, chunk_keys);
        let n_kv = heads / group;
        let stride = dh + ATTN_FDEC_PART_EXTRA;
        // カーネルと同じ添字: part[(kh * ns + sp) * group + hh]
        let mut part = vec![0f32; n_kv * ns * group * stride];
        for kh in 0..n_kv {
            let off = kh * dh;
            for hh in 0..group {
                let h = kh * group + hh;
                for sp in 0..ns {
                    let a = sp * chunk;
                    let b = ((sp + 1) * chunk).min(len);
                    let base = ((kh * ns + sp) * group + hh) * stride;
                    let mut accs = vec![0f32; dh];
                    let mut m = -3.402823466e+38f32;
                    let mut l = 0f32;
                    for j in a..b {
                        let krow = &k[j * kv_dim + off..][..dh];
                        let mut s = 0f32;
                        for d in 0..dh {
                            s += q[h * dh + d] * krow[d];
                        }
                        let sc = s * scale;
                        let corr;
                        let p;
                        if sc > m {
                            corr = ((m - sc) * std::f32::consts::LOG2_E).exp2();
                            m = sc;
                            p = 1.0f32; // exp2f(0) と厳密に同じ
                        } else {
                            corr = 1.0f32;
                            p = ((sc - m) * std::f32::consts::LOG2_E).exp2();
                        }
                        l = l * corr + p;
                        let vrow = &v[j * kv_dim + off..][..dh];
                        for d in 0..dh {
                            accs[d] = accs[d] * corr + p * vrow[d];
                        }
                    }
                    part[base] = m;
                    part[base + 1] = l;
                    part[base + 2..][..dh].copy_from_slice(&accs);
                }
            }
        }
        // combine (split 昇順)
        let mut out = vec![0f32; heads * dh];
        for kh in 0..n_kv {
            for hh in 0..group {
                let h = kh * group + hh;
                let base = ((kh * ns) * group + hh) * stride;
                let step = group * stride;
                let mut mg = -3.402823466e+38f32;
                for sp in 0..ns {
                    mg = mg.max(part[base + sp * step]);
                }
                for d in 0..dh {
                    let mut denom = 0f32;
                    let mut num = 0f32;
                    for sp in 0..ns {
                        let p = &part[base + sp * step..][..stride];
                        let e = ((p[0] - mg) * std::f32::consts::LOG2_E).exp2();
                        denom = p[1].mul_add(e, denom);
                        num = p[2 + d].mul_add(e, num);
                    }
                    out[h * dh + d] = if denom > 0.0 { num / denom } else { 0.0 };
                }
            }
        }
        out
    }

    /// task 160: 文脈長からの分割数の選択。空の split を作らず、分割数が上限に
    /// 収まり、返す chunk がカーネル側の `ceil(t_total / n_split)` と一致すること。
    #[test]
    fn q4_fdec_split_is_bounded_and_matches_kernel_formula() {
        for t in [1usize, 2, 17, 511, 1024, 4096, 14139, 65535] {
            for keys in [1usize, 32, 128, 256, 1024, 4096, 16384] {
                let (ns, chunk) = q4_fdec_split(t, keys);
                assert!(ns >= 1 && ns <= ATTN_FDEC_SPLIT_MAX, "t={t} keys={keys}");
                // カーネルは自分で chunk = ceil(t_total / n_split) を計算する。
                assert_eq!(chunk, t.div_ceil(ns), "kernel の chunk と不一致 (t={t} keys={keys})");
                assert!(ns * chunk >= t, "区間が足りない (t={t} keys={keys})");
                assert!(ns == 1 || (ns - 1) * chunk < t, "最後の split が空 (t={t})");
            }
        }
        // 14,139 は 256 key 分割で 56 分割 = 224 block (n_kv = 4)
        assert_eq!(q4_fdec_split(14139, 256), (56, 253));
        assert_eq!(q4_fdec_split(3954, 256), (16, 248));
        assert_eq!(q4_fdec_split(1, 256), (1, 1));
        assert_eq!(q4_fdec_split(0, 256), (1, 1));
        // 既定 chunk では 14k で SM 170 を埋めるだけの block が出ること
        // (n_kv = 2 のモデルで block 数 = n_split x 2)。
        let (ns, _) = q4_fdec_split(14139, ATTN_FLASHDEC_CHUNK_DEFAULT);
        assert!(ns * 2 >= 340, "14k の block が足りない: {ns} x 2");
    }

    /// task 160: flash decoding のゲートの解釈 (純関数)。
    #[test]
    fn attn_flashdec_gate_parses() {
        // task 163: lab t163a の結果で既定 on にした (うっかり戻したらここで落ちる)。
        assert!(
            ATTN_FLASHDEC_DEFAULT,
            "flash decoding は t163a で採用済み (既定 on が契約)"
        );
        assert!(parse_attn_flashdec(Some("1")));
        assert!(!parse_attn_flashdec(Some("0")));
        assert_eq!(parse_attn_flashdec(Some("x")), ATTN_FLASHDEC_DEFAULT);
        assert_eq!(parse_attn_flashdec(None), ATTN_FLASHDEC_DEFAULT);
        assert_eq!(parse_attn_flashdec_min_ctx(Some("2048")), 2048);
        assert_eq!(parse_attn_flashdec_min_ctx(Some("")), ATTN_FLASHDEC_MIN_CTX_DEFAULT);
        assert_eq!(parse_attn_flashdec_min_ctx(None), ATTN_FLASHDEC_MIN_CTX_DEFAULT);
        assert_eq!(parse_attn_flashdec_chunk(Some("8")), ATTN_FLASHDEC_CHUNK_MIN);
        assert_eq!(parse_attn_flashdec_chunk(Some("99999")), 4096);
        assert_eq!(parse_attn_flashdec_chunk(Some("128")), 128);
        assert_eq!(parse_attn_flashdec_chunk(None), ATTN_FLASHDEC_CHUNK_DEFAULT);
    }

    /// task 160: flash decoding のアルゴリズムが dense softmax と一致すること
    /// (許容差は task 144 の abs 1e-4 + rel 1e-5)。長さ 1 / 17 / 511 / 4,096 /
    /// 14,000 — 分割の倍数でない長さ (17, 511, 14,000) を含む。
    /// GQA 比 3 / 4 / 6 と、lab t163a/t163b で測った実形状 (24 heads / 2 kv /
    /// dh 256 = 比 12) を通す。
    #[test]
    fn q4_attn_flashdec_algorithm_matches_dense_softmax() {
        for &(heads, n_kv, dh) in &[
            (6usize, 2usize, 64usize),
            (8, 2, 32),
            (24, 4, 32),
            (24, 2, 256),
        ] {
            let group = heads / n_kv;
            let kv_dim = n_kv * dh;
            let scale = 1.0 / (dh as f32).sqrt();
            for &len in &[1usize, 17, 511, 4096, 14000] {
                let q = pseudo_random(heads * dh, 7 + len as u32);
                let k = pseudo_random(len * kv_dim, 11 + len as u32);
                let v = pseudo_random(len * kv_dim, 13 + len as u32);
                let want = cpu_reference(&q, &k, &v, len, kv_dim, heads, group, dh, scale);
                for &chunk in &[32usize, 256, 1024] {
                    let got = flashdec_attention_cpu_ref(
                        &q, &k, &v, len, kv_dim, heads, group, dh, scale, chunk,
                    );
                    let (ns, _) = q4_fdec_split(len, chunk);
                    for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                        let tol = 1e-4 + 1e-5 * b.abs();
                        assert!(
                            (a - b).abs() <= tol,
                            "len={len} heads={heads} n_kv={n_kv} chunk={chunk} n_split={ns} \
                             i={i}: flashdec={a} dense={b} (diff {})",
                            (a - b).abs()
                        );
                    }
                }
            }
        }
    }

    /// task 160: flash decoding を既存 split カーネルと CPU 参照に突き合わせる
    /// (GPU が要る)。長さ 1 / 17 / 511 / 4,096 / 14,000、GQA 比 2 / 3 / 6。
    /// 決定論 (同一入力でビット一致) と、分割数を変えても近いことも見る。
    /// `cargo test --release --features cuda -p denno-engine --lib
    /// q4_attn_flashdec -- --ignored --test-threads=1` でだけ回す。
    #[test]
    #[ignore = "GPU が必要"]
    fn q4_attn_flashdec_matches_split_kernel() {
        let _g = test_lock()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        enable_for_tests();
        let _ = CudaCtx::new().expect("GPU が無い");
        set_q4_gpu_attn_for_test(Some(true));
        set_q4_gpu_attn_min_for_test(Some(0));
        set_q4_gpu_attn_kernel_for_test(Some(Q4AttnKernel::Split));
        set_q4_gpu_attn_split_for_test(Some(8));
        set_attn_flashdec_min_ctx_for_test(Some(0));
        set_attn_flashdec_chunk_for_test(Some(256));

        for &(heads, n_kv, dh) in &[(4usize, 2usize, 64usize), (6, 2, 64), (24, 4, 32)] {
            let group = heads / n_kv;
            let kv_dim = n_kv * dh;
            let scale = 1.0 / (dh as f32).sqrt();
            for &len in &[1usize, 17, 511, 4096, 14000] {
                let mut cache = AttnCache::new(1, n_kv, dh, len + 1).unwrap();
                let mut lcg = {
                    let mut s = 4242u64 ^ (len as u64);
                    move || {
                        s = s
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        ((s >> 33) as f32 / (1u64 << 30) as f32) - 1.0
                    }
                };
                for _ in 0..len {
                    let k: Vec<f32> = (0..kv_dim).map(|_| lcg()).collect();
                    let v: Vec<f32> = (0..kv_dim).map(|_| lcg()).collect();
                    cache.push(0, &k, &v).unwrap();
                }
                let q: Vec<f32> = (0..heads * dh).map(|_| lcg()).collect();
                let call = Q4AttnCall {
                    cache_id: cache.id(),
                    n_layer: 1,
                    layer: 0,
                    layer_gen: cache.gen(0).unwrap(),
                    len,
                    kv_dim,
                    heads,
                    group,
                    dh,
                    scale,
                    q: &q,
                    k: cache.k(0).unwrap(),
                    v: cache.v(0).unwrap(),
                };
                let want = cpu_reference(
                    &q,
                    cache.k(0).unwrap(),
                    cache.v(0).unwrap(),
                    len,
                    kv_dim,
                    heads,
                    group,
                    dh,
                    scale,
                );
                // 1) flash decoding
                set_attn_flashdec_for_test(Some(true));
                let got = q4_attn_decode(&call).expect("flash decoding が発動していない");
                // 2) 既存 split カーネル
                set_attn_flashdec_for_test(Some(false));
                let got_split = q4_attn_decode(&call).expect("split 経路が発動していない");
                for (name, got) in [("fdec", &got), ("split", &got_split)] {
                    for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                        let tol = 1e-4 + 1e-5 * b.abs();
                        assert!(
                            (a - b).abs() <= tol,
                            "{name}: len={len} heads={heads} i={i}: {a} vs {b} (diff {})",
                            (a - b).abs()
                        );
                    }
                }
                // 3) 2 つのカーネルは同じ答え (丸め順だけが違う)
                for (i, (a, b)) in got.iter().zip(&got_split).enumerate() {
                    let tol = 1e-4 + 1e-5 * b.abs();
                    assert!((a - b).abs() <= tol, "fdec と split が違う: len={len} i={i}");
                }
                // 4) 決定論
                set_attn_flashdec_for_test(Some(true));
                let got2 = q4_attn_decode(&call).expect("flash decoding が発動していない");
                assert_eq!(got, got2, "flash decoding が非決定論的 (len={len})");
                // 5) 分割数を変えても近い
                set_attn_flashdec_chunk_for_test(Some(64));
                let got3 = q4_attn_decode(&call).expect("flash decoding が発動していない");
                for (i, (a, b)) in got3.iter().zip(&want).enumerate() {
                    let tol = 1e-4 + 1e-5 * b.abs();
                    assert!((a - b).abs() <= tol, "chunk=64 で値が違う: len={len} i={i}");
                }
                set_attn_flashdec_chunk_for_test(Some(256));
            }
        }

        set_attn_flashdec_for_test(None);
        set_attn_flashdec_min_ctx_for_test(None);
        set_attn_flashdec_chunk_for_test(None);
        set_q4_gpu_attn_kernel_for_test(None);
        set_q4_gpu_attn_split_for_test(None);
        set_q4_gpu_attn_min_for_test(None);
        set_q4_gpu_attn_for_test(None);
    }

    /// task 135: t 分割 decode attention を CPU 参照と突き合わせる (GPU が要る)。
    /// 決定論 (同一入力でビット一致) と分割数を変えても近いことも見る。
    /// `cargo test --release --features cuda -p denno-engine --lib
    /// q4_attn_decode_split -- --ignored --test-threads=1` でだけ回す。
    #[test]
    #[ignore = "GPU が必要"]
    fn q4_attn_decode_split_matches_cpu() {
        let _g = test_lock()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        enable_for_tests();
        let _ = CudaCtx::new().expect("GPU が無い");
        set_q4_gpu_attn_for_test(Some(true));
        set_q4_gpu_attn_min_for_test(Some(0));
        set_q4_gpu_attn_kernel_for_test(Some(Q4AttnKernel::Split));
        set_q4_gpu_attn_split_for_test(Some(8));

        let (heads, n_kv, dh) = (4usize, 2usize, 64usize);
        let group = heads / n_kv;
        let kv_dim = n_kv * dh;
        let mut cache = AttnCache::new(1, n_kv, dh, 4096).unwrap();
        let mut lcg = {
            let mut s = 4242u64;
            move || {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 30) as f32) - 1.0
            }
        };
        for _ in 0..1000 {
            let k: Vec<f32> = (0..kv_dim).map(|_| lcg()).collect();
            let v: Vec<f32> = (0..kv_dim).map(|_| lcg()).collect();
            cache.push(0, &k, &v).unwrap();
        }
        let len = cache.len(0).unwrap();
        let scale = 1.0 / (dh as f32).sqrt();
        let q: Vec<f32> = (0..heads * dh).map(|_| lcg()).collect();
        let call = Q4AttnCall {
            cache_id: cache.id(),
            n_layer: 1,
            layer: 0,
            layer_gen: cache.gen(0).unwrap(),
            len,
            kv_dim,
            heads,
            group,
            dh,
            scale,
            q: &q,
            k: cache.k(0).unwrap(),
            v: cache.v(0).unwrap(),
        };
        let want = cpu_reference(&q, cache.k(0).unwrap(), cache.v(0).unwrap(), len, kv_dim, heads, group, dh, scale);
        let got = q4_attn_decode(&call).expect("split 経路が発動していない");
        let maxd = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(maxd < 1e-4, "split decode が CPU 参照と違う: maxabs={maxd}");
        // 決定論 (同一入力の 2 回目はビット一致)
        let got2 = q4_attn_decode(&call).expect("split 経路が発動していない");
        assert_eq!(got, got2, "split decode が非決定論的");
        // 分割数を変えても値は近い (丸め順だけが変わる)
        set_q4_gpu_attn_split_for_test(Some(3));
        let got3 = q4_attn_decode(&call).expect("split 経路が発動していない");
        let maxd3 = got3
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(maxd3 < 1e-4, "分割数 3 で値が違う: maxabs={maxd3}");

        set_q4_gpu_attn_kernel_for_test(None);
        set_q4_gpu_attn_split_for_test(None);
        set_q4_gpu_attn_min_for_test(None);
        set_q4_gpu_attn_for_test(None);
    }

    /// テスト用の CPU 参照 (dno_attention と同じ式・同じ順序)。
    #[allow(clippy::too_many_arguments)]
    fn cpu_reference(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        len: usize,
        kv_dim: usize,
        heads: usize,
        group: usize,
        dh: usize,
        scale: f32,
    ) -> Vec<f32> {
        let mut out = vec![0f32; heads * dh];
        for h in 0..heads {
            let off = (h / group) * dh;
            let mut scores = vec![0f32; len];
            let mut mx = f32::NEG_INFINITY;
            for j in 0..len {
                let krow = &k[j * kv_dim + off..][..dh];
                let mut acc = 0f32;
                for d in 0..dh {
                    acc += q[h * dh + d] * krow[d];
                }
                scores[j] = acc * scale;
                mx = mx.max(scores[j]);
            }
            let mut sum = 0f32;
            for s in scores.iter_mut() {
                *s = (*s - mx).exp();
                sum += *s;
            }
            let inv = 1.0f32 / sum;
            for j in 0..len {
                let vrow = &v[j * kv_dim + off..][..dh];
                let p = scores[j];
                for d in 0..dh {
                    out[h * dh + d] += p * vrow[d];
                }
            }
            for d in 0..dh {
                out[h * dh + d] *= inv;
            }
        }
        out
    }
}


/// task 270: `DENNO_PF_CHAIN_BIG` — GDN prefill の連鎖を `nt >= 2048` でも使う。
///
/// 既定 **off**。`DENNO_PF_CHUNK` を 2048 以上にしたときだけ意味がある。
/// **明示的な真値だけ on** (未設定・`0`・空は off)。
///
/// なぜ要るか: 連鎖の判定は「単発 sigmoid スキャン (`dno_gdn_scan_sig`) が受ける
/// `nt < GDN_CHUNK_MIN_TOKENS (2048)`」で切られていた (task 167)。task 197 で
/// チャンク 2048 が +39.6% になったのは、この境界で連鎖が外れ、ssm が
/// 3.96 → 24.7 s に落ちたため。その後 task 186/225/228 で v2/v3/WY の
/// スキャンが入り、どれもトークン長の上限を持たない (状態はデバイス正本、
/// トークンはカーネル内で順に進む)。よって形とカーネルが揃えば
/// 2048 以上でも連鎖を使ってよい。
pub const CHAIN_BIG_ENV: &str = "DENNO_PF_CHAIN_BIG";


/// [`CHAIN_BIG_ENV`] の解釈 (純関数)。明示的な真値だけ on。
pub fn chain_big_from_value(v: Option<&str>) -> bool {
    matches!(
        v.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("1") | Some("on") | Some("true") | Some("yes")
    )
}


#[cfg(test)]
static CHAIN_BIG_FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);


/// `DENNO_PF_CHAIN_BIG` が on か (プロセスで 1 回だけ env を読む)。
pub fn chain_big_on() -> bool {
    #[cfg(test)]
    {
        match CHAIN_BIG_FORCE.load(std::sync::atomic::Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| chain_big_from_value(std::env::var(CHAIN_BIG_ENV).ok().as_deref()))
}


/// task 270: 連鎖が受ける `nt` の範囲 (純関数)。
///
/// - `nt < GDN_SIG_MIN_TOKENS`: 短すぎる (decode と同じ床)。
/// - `nt < GDN_CHUNK_MIN_TOKENS (2048)`: 単発 sigmoid スキャンが受ける (従来)。
/// - `nt >= 2048`: `big` (= `DENNO_PF_CHAIN_BIG`) が on で、上限の無いスキャンの
///   形 (`gdn_pf2_ok` = dk = dv = 128 / d_conv = 4 / conv_dim 一致) が揃うときだけ。
///
/// `big` を引数にして、ゲートの env を触らずに規則をテストできるようにする。
pub fn chain_nt_ok(
    nt: usize,
    nk: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    d_conv: usize,
    conv_dim: usize,
    big: bool,
) -> bool {
    if nt < GDN_SIG_MIN_TOKENS {
        return false;
    }
    if nt < GDN_CHUNK_MIN_TOKENS {
        return true;
    }
    big && gdn_pf2_ok(nt, nk, nv, dk, dv, d_conv, conv_dim)
}


/// s167 (P6): h2 の D2H を専用コピーストリームへ**発行するだけ** (同期しない)。
/// `CH_PFH` の `elem_off` 要素目から `len` 個を pinned の `pin_h2` へ落とす。
/// 呼び出し時点で `CH_PFH` の当該範囲が既定ストリーム上で確定している前提
/// (rmsnorm の直後に呼ぶ) で、イベントで既定ストリーム → コピーストリームの
/// 順序を作る。完了は [`chain_pf_h2_join`] で待つ。
///
/// 失敗はすべて `None` で、**GPU_BROKEN は立てない**: フラグ OFF・driver API
/// 欠落・stream/event/pinned の作成失敗は「従来の同期 D2H でよい」種類の失敗
/// なので、呼び出し側は静かに従来経路へ落ちる。
pub(crate) fn chain_pf_h2_begin(slot: usize, elem_off: usize, len: usize) -> Option<()> {
    let _ctx = chain_ctx()?;
    let bytes = len.checked_mul(4)?;
    if bytes == 0 {
        return None;
    }
    let need = elem_off.checked_mul(4)?.checked_add(bytes)?;
    let lib = lib().ok()?;
    let (stream_create, event_create, event_record, stream_wait_event) = (
        lib.stream_create?,
        lib.event_create?,
        lib.event_record?,
        lib.stream_wait_event?,
    );
    CHAIN.with(|cell| -> Option<()> {
        let mut s = cell.borrow_mut();
        // 未 join のコピーが残っている状態で発行し直すと宛先 (pin_h2) を取り合う。
        // 呼び出し規約違反なので受けない (join が必ず h2_pending を落とす)。
        if s.h2_pending {
            return None;
        }
        // 1) コピーストリーム一式を初回だけ作る。
        if s.h2_stream.is_none() {
            // 非ブロッキング必須: legacy (既定) ストリームは blocking ストリームと
            // 暗黙に同期するため、blocking で作ると GEMM がコピー完了を待ってしまい
            // 1 バイトも重ならない。
            const CU_STREAM_NON_BLOCKING: u32 = 0x01;
            const CU_EVENT_DISABLE_TIMING: u32 = 0x02;
            let mut st: CUstream = std::ptr::null_mut();
            if unsafe { stream_create(&mut st, CU_STREAM_NON_BLOCKING) } != CUDA_SUCCESS {
                return None;
            }
            let st = CopyStream(st);
            let mut ev_ready: CUevent = std::ptr::null_mut();
            if unsafe { event_create(&mut ev_ready, CU_EVENT_DISABLE_TIMING) } != CUDA_SUCCESS {
                return None;
            }
            let ev_ready = CopyEvent(ev_ready);
            let mut ev_done: CUevent = std::ptr::null_mut();
            if unsafe { event_create(&mut ev_done, CU_EVENT_DISABLE_TIMING) } != CUDA_SUCCESS {
                return None;
            }
            let ev_done = CopyEvent(ev_done);
            s.h2_stream = Some(st);
            s.h2_ev_ready = Some(ev_ready);
            s.h2_ev_done = Some(ev_done);
        }
        // 2) 宛先 (pinned) と転送元を用意する。pinned の確保失敗も従来経路へ
        //    (ページロック上限などは致命的でないので GPU_BROKEN は立てない)。
        if Scratch::ensure_pin(&mut s.pin_h2, bytes).is_err() {
            return None;
        }
        let dst = match s.pin_h2.as_ref() {
            Some(p) if p.bytes >= bytes => p.ptr,
            _ => return None,
        };
        let src = match s.bufs.get(slot).and_then(|b| b.as_ref()) {
            Some(b) if b.bytes() >= need => b.ptr(),
            _ => return None,
        };
        let (st, ev_ready, ev_done) = (
            s.h2_stream.as_ref()?.0,
            s.h2_ev_ready.as_ref()?.0,
            s.h2_ev_done.as_ref()?.0,
        );
        // 3) 既定ストリーム (null) の rmsnorm 完了をイベントで受けてから発行する。
        if unsafe { event_record(ev_ready, std::ptr::null_mut()) } != CUDA_SUCCESS {
            return None;
        }
        if unsafe { stream_wait_event(st, ev_ready, 0) } != CUDA_SUCCESS {
            return None;
        }
        if unsafe { (lib.memcpy_dtoh_async)(dst, src + (elem_off * 4) as CUdeviceptr, bytes, st) }
            != CUDA_SUCCESS
        {
            return None;
        }
        // 4) 完了イベントをコピーストリームに打つ (join はこれだけを待つ)。
        if unsafe { event_record(ev_done, st) } != CUDA_SUCCESS {
            return None;
        }
        s.h2_pending = true;
        Some(())
    })
}


/// s167 (P6): 発行済みの h2 D2H を join する (完了イベントを待つ)。ここから先で
/// ホストが `pin_h2` を読める。`h2_pending` が無い (発行していない) なら None。
///
/// 待ちに失敗しても `h2_pending` は落とす — 以後の層は従来の同期 D2H に戻る
/// (同じストリームに残った古いコピーは、次のコピーの完了イベントより前に
/// 必ず終わるので、宛先の取り合いにはならない)。
pub(crate) fn chain_pf_h2_join() -> Option<()> {
    let _ctx = chain_ctx()?;
    let event_sync = lib().ok()?.event_sync?;
    CHAIN.with(|cell| -> Option<()> {
        let mut s = cell.borrow_mut();
        if !s.h2_pending {
            return None;
        }
        let ev = s.h2_ev_done.as_ref()?.0;
        s.h2_pending = false;
        if unsafe { event_sync(ev) } != CUDA_SUCCESS {
            return None;
        }
        Some(())
    })
}


// ---------------------------------------------------------------------------
// s165 (P2): prefill の残差ストリーム (h) をデバイス常駐させる経路の小道具。
//
// 従来の prefill は層ごとに「rmsnorm(h) をホストで計算 → h2 を H2D → 投影 →
// 出力を D2H → 残差加算をホスト」を繰り返していた (1 層 2 往復 + ホスト 2 パス)。
// P2 では h を `CH_PF_RES` に置いたまま、層の中で
//   rmsnorm (デバイス) → 投影 (デバイス) → 残差加算 (次の rmsnorm の proj 引数)
// を通す。ホストが本当に必要とするものだけが往復する:
//   - GDN 層: beta/alpha を計算するための h2 の D2H 1 回 (状態契約は従来どおり)
//   - full attention 層: q/k/v はホスト計算なので h2 の D2H 1 回 + att の H2D 1 回
//   - FFN: 往復なし
//
// 残差加算に専用カーネルは要らない (PTX に単独の add は無い) — `dno_add_rmsnorm`
// の `proj` 引数が「x += proj」と「rmsnorm(x)」を 1 回で行うので、層の最後の
// 投影出力を**次の rmsnorm の add_slot** として渡す。演算順はホスト経路と同一
// (ホストも `x += proj` してから `rmsnorm(x)` を計算する)。
// ---------------------------------------------------------------------------

/// 層 1 枚ぶんの attention 投影の重み (事前検査・重み登録で使う)。
pub enum PfAttnW<'a> {
    Gdn {
        qkv: &'a crate::quant::Linear,
        z: &'a crate::quant::Linear,
        o: &'a crate::quant::Linear,
    },
    Full {
        q: &'a crate::quant::Linear,
        k: &'a crate::quant::Linear,
        v: &'a crate::quant::Linear,
        o: &'a crate::quant::Linear,
    },
}


/// s165 (P2): 残差常駐経路が使うスロットを事前に確保する (rmsnorm の前に呼ぶ)。
/// ここで失敗すれば残差はまだ触っていないので、呼び出し側は安全にフォールバックできる。
pub fn chain_pf_reserve(n: usize, e: usize) -> Option<()> {
    let _ = chain_ctx()?;
    let rows = n.checked_mul(e)?.checked_mul(4)?;
    ok(chain_ensure_buf(CH_PF_RES, rows))?;
    ok(chain_ensure_buf(CH_PFH, rows.checked_add(e.checked_mul(4)?)?))?;
    ok(chain_ensure_buf(CH_O, rows))?;
    ok(chain_ensure_buf(CH_DOWN, rows))?;
    Some(())
}


/// s165 (P2): 残差ストリーム (h) を `CH_PF_RES` へ 1 回だけアップロードする
/// (非同期・同期なし)。prefill の開始時に 1 回だけ呼ぶ。
pub fn chain_pf_res_upload(h: &[f32], m: usize, e: usize) -> Option<()> {
    let _ = chain_ctx()?;
    let n_elems = m.checked_mul(e)?;
    let bytes = n_elems.checked_mul(4)?;
    if n_elems == 0 || h.len() < n_elems {
        return None;
    }
    ok(chain_ensure_buf(CH_PF_RES, bytes))?;
    CHAIN.with(|cell| -> Option<()> {
        let buf = cell.borrow().bufs[CH_PF_RES].as_ref()?.ptr();
        let lib = lib().ok()?;
        unsafe {
            ok(check(
                (lib.memcpy_htod_async)(buf, h.as_ptr() as *const c_void, bytes, std::ptr::null_mut()),
                "cuMemcpyHtoDAsync (pf residual)",
            ))
        }
    })
}


/// s165 (P2): `CH_PF_RES` の内容をホストへ戻す (フォールバック時の復元・同期あり)。
pub fn chain_pf_res_download(m: usize, e: usize, out: &mut [f32]) -> Option<()> {
    let n_elems = m.checked_mul(e)?;
    if n_elems == 0 || out.len() < n_elems {
        return None;
    }
    let v = chain_download_at(CH_PF_RES, 0, n_elems)?;
    out[..n_elems].copy_from_slice(&v);
    Some(())
}


/// s165 (P2): ホストで計算した attention 出力 (sigmoid ゲート適用済み) を
/// `CH_ATT` へ 1 回だけアップロードする。pinned は層ごとに専用 (li*3+2) —
/// 非同期 H2D の完了前に次の層が同じ領域を上書きしないため
/// (`chain_rmsnorm` の pin と同じ規約)。
pub fn chain_pf_upload_att(att: &[f32], cols: usize, m: usize, li: usize) -> Option<()> {
    let _ = chain_ctx()?;
    let n_elems = m.checked_mul(cols)?;
    let bytes = n_elems.checked_mul(4)?;
    if n_elems == 0 || att.len() < n_elems {
        return None;
    }
    ok(chain_ensure_buf(CH_ATT, bytes))?;
    let pin_id = li.checked_mul(3)?.checked_add(2)?;
    CHAIN.with(|cell| -> Option<()> {
        let mut s = cell.borrow_mut();
        while s.pins.len() <= pin_id {
            s.pins.push(None);
        }
        ok(Scratch::ensure_pin(&mut s.pins[pin_id], bytes))?;
        let pin = s.pins[pin_id].as_mut().unwrap();
        pin.as_mut_slice::<f32>()[..n_elems].copy_from_slice(&att[..n_elems]);
        let src = pin.ptr as *const c_void;
        let buf = s.bufs[CH_ATT].as_ref()?.ptr();
        let lib = lib().ok()?;
        unsafe {
            ok(check(
                (lib.memcpy_htod_async)(buf, src, bytes, std::ptr::null_mut()),
                "cuMemcpyHtoDAsync (pf att)",
            ))
        }
    })
}


/// s172 (P12): prefill attention の出力を **pinned ステージングへ直接** D2H し、
/// 呼び出し側のクロージャでその場を書き換えてから同じバッファで H2D する
/// (`DENNO_PF_ATT_INPLACE=0` で従来の
/// `chain_download` → ホストで変換 → `chain_pf_upload_att` に戻る)。
///
/// 従来経路の無駄 (27B n=14,043 で 1 層あたり):
///   1. `chain_download` が **pageable な新しい `Vec`** (345 MB) を確保して D2H
///      → 毎層 345 MB の page fault + driver の内部ステージング経由。
///   2. 呼び出し側がゲートを掛けたあと `att.copy_from_slice(&ga)` で
///      ホスト・ホストの 345 MB コピー (読み + 書き = 690 MB)。
///   3. `chain_pf_upload_att` が pinned へもう一度 345 MB コピー (690 MB)。
/// ここでは D2H の宛先を 3 の pinned バッファそのものにして、変換もそこで行い、
/// そのまま H2D する — 転送量は同じ (345 MB 下り + 345 MB 上り) で、
/// 余計な 1.4 GB/層 のホスト memcpy と 345 MB/層 の確保が消える。
///
/// 安全性: D2H は `ctx.sync()` で完了を待ってから書き換える。pinned バッファは
/// 層ごと (`chain_pf_upload_att` と同一の `li*3+2`) なので、次の層が同じ領域を
/// 使うのは次の層の qkv chain の同期より後 = この H2D は既に完了している。
/// カーネル・引数・転送バイト数は従来と同一で、値も 1 ビットも変わらない。
///
/// s246 (P46): `gate` を `FnMut(行オフセット, 行ブロック)` に変え、転送を
/// **行ブロック単位のチャンク**に割って「D2H(k) 完了 → gate(k) → H2D(k)」を
/// 重ねる (`DENNO_PF_ATT_PIPE`・既定 ON)。従来は
/// `D2H 全体 → ctx.sync() → gate 全体 → H2D 全体` で、ホストは D2H の完了
/// (= PCIe 1 往復ぶん) を待ち終えてからゲートを始め、ゲートが終わるまで H2D を
/// 発行できなかった (FC_KPROF 65k・16 層: d2h=394 ms / tail=1014 ms)。
/// チャンク化しても転送バイト数・宛先オフセット・ゲートの式・`CH_ATT` の最終内容は
/// 1 ビットも変わらない — 各要素は同じ式で 1 回だけゲートされ、同じオフセットへ
/// 同じバイト列が戻る。変わるのは「いつ発行するか」だけ。
pub fn chain_pf_att_inplace<F>(li: usize, cols: usize, m: usize, mut gate: F) -> Option<()>
where
    F: FnMut(usize, &mut [f32]),
{
    if cols == 0 || m == 0 {
        return None;
    }
    let ctx = chain_ctx()?;
    let n_elems = m.checked_mul(cols)?;
    let bytes = n_elems.checked_mul(4)?;
    ok(chain_ensure_buf(CH_ATT, bytes))?;
    let pin_id = li.checked_mul(3)?.checked_add(2)?;
    CHAIN.with(|cell| -> Option<()> {
        let mut s = cell.borrow_mut();
        while s.pins.len() <= pin_id {
            s.pins.push(None);
        }
        ok(Scratch::ensure_pin(&mut s.pins[pin_id], bytes))?;
        let lib = lib().ok()?;
        let (buf, pin_ptr) = {
            let buf = s.bufs[CH_ATT].as_ref()?.ptr();
            let pin = s.pins[pin_id].as_ref()?;
            if pin.bytes < bytes {
                return None;
            }
            (buf, pin.ptr)
        };
        // s246 (P46): チャンク数 (1 = 従来の 1 発往復)。イベントを作れない環境・
        // 小さい転送・フラグ OFF では 1 = 従来経路そのものに落ちる。
        let (rows_per, nch) = pf_att_chunk_rows(m, pf_att_pipe_chunks(bytes, m));
        if nch >= 2 && pf_att_ensure_events(&mut s, nch) {
            return pf_att_inplace_piped(
                &mut s, lib, ctx, buf, pin_ptr, pin_id, cols, m, bytes, rows_per, nch, &mut gate,
            );
        }
        // 1) CH_ATT → pinned (宛先が pinned なので pageable 経路の内部ステージングが無い)
        unsafe {
            ok(check(
                (lib.memcpy_dtoh_async)(pin_ptr, buf, bytes, std::ptr::null_mut()),
                "cuMemcpyDtoHAsync (pf att pin)",
            ))?;
        }
        // s180 (P41): att の D2H 発行点 = full_core の区間 [d2h] の終端
        // (イベントはストリーム順。sync はこの直後の既存の 1 回だけ)。
        fc_kprof_mark(3);
        ok(ctx.sync())?;
        // 2) pinned の上で変換 (数値は呼び出し側のクロージャ = 従来と同じ式)
        {
            let pin = s.pins[pin_id].as_mut().unwrap();
            let v = pin.as_mut_slice::<f32>();
            gate(0, &mut v[..n_elems]);
        }
        // 3) pinned → CH_ATT (async H2D。同一ストリームの後続 GEMM が順序で待つ)
        unsafe {
            ok(check(
                (lib.memcpy_htod_async)(buf, pin_ptr as *const c_void, bytes, std::ptr::null_mut()),
                "cuMemcpyHtoDAsync (pf att gate)",
            ))?;
        }
        Some(())
    })
}


/// s246 (P46): att の D2H/H2D とゲートを重ねる経路のチャンク数。
///
/// 1 = 従来の 1 発往復。`DENNO_PF_ATT_PIPE=0` で常に 1、`=N` で上限 N
/// (未設定は既定の上限)。転送が小さいときは重ねる意味が無いので 1。
fn pf_att_pipe_chunks(bytes: usize, m: usize) -> usize {
    pf_att_pipe_chunks_with(bytes, m, pf_att_pipe_max())
}


/// s246 (P46): 既定の上限 (env `DENNO_PF_ATT_PIPE`・起動時に 1 回だけ読む)。
fn pf_att_pipe_max() -> usize {
    /// 既定の上限。チャンクあたりの API 呼び出しは 4 回 (D2H / record / sync /
    /// H2D) なので、増やすと 1 層あたり数十 µs の固定費が乗る。
    const MAX_CHUNKS: usize = 8;
    static MAX: OnceLock<usize> = OnceLock::new();
    *MAX.get_or_init(|| match std::env::var("DENNO_PF_ATT_PIPE") {
        Ok(v) => match v.trim().parse::<usize>() {
            Ok(n) => n,
            // 数値でない値 (空文字・"on" 等) は既定の上限として扱う。
            Err(_) => MAX_CHUNKS,
        },
        Err(_) => MAX_CHUNKS,
    })
}


/// s246 (P46): `pf_att_pipe_chunks` の純関数部分 (env を読まない = テスト可能)。
pub(crate) fn pf_att_pipe_chunks_with(bytes: usize, m: usize, max: usize) -> usize {
    /// 1 チャンクの目標バイト数。これ以下なら 1 発 (PCIe の立上がりを払う方が高い)。
    const CHUNK_BYTES: usize = 128 << 20;
    if max < 2 || bytes < CHUNK_BYTES || m < 2 {
        return 1;
    }
    bytes.div_ceil(CHUNK_BYTES).clamp(2, max).min(m)
}


/// s246 (P46): `m` 行を高々 `nch` 個の行ブロックに割る。戻り値は
/// `(rows_per, 実際のチャンク数)`。行ブロックは `[0, m)` を重複なく覆い、
/// 最終チャンクだけ短くなる (端数行を捨てない)。`nch < 2` は 1 を返す
/// (= 従来の 1 発往復)。
pub(crate) fn pf_att_chunk_rows(m: usize, nch: usize) -> (usize, usize) {
    if m == 0 || nch < 2 {
        return (m, 1);
    }
    let rows_per = m.div_ceil(nch.min(m));
    (rows_per, m.div_ceil(rows_per))
}


/// s246 (P46): チャンク `k` の行範囲 `[r0, r1)`。`pf_att_chunk_rows` の
/// `(rows_per, nch)` に対して全チャンクが `[0, m)` を隙間なく覆う。
pub(crate) fn pf_att_chunk_bounds(k: usize, rows_per: usize, m: usize) -> (usize, usize) {
    let r0 = k.saturating_mul(rows_per).min(m);
    let r1 = r0.saturating_add(rows_per).min(m);
    (r0, r1)
}


/// s246 (P46): パイプライン用の完了イベントを `n` 個そろえる (grow-only)。
/// 作れなければ false (呼び出し側は従来の 1 発往復に落ちる)。
fn pf_att_ensure_events(s: &mut ChainScratch, n: usize) -> bool {
    if s.pipe_ev.len() >= n {
        return true;
    }
    let Some(create) = lib().ok().and_then(|l| l.event_create) else {
        return false;
    };
    const CU_EVENT_DISABLE_TIMING: u32 = 0x02;
    while s.pipe_ev.len() < n {
        let mut ev: CUevent = std::ptr::null_mut();
        if unsafe { create(&mut ev, CU_EVENT_DISABLE_TIMING) } != CUDA_SUCCESS || ev.is_null() {
            return false;
        }
        s.pipe_ev.push(CopyEvent(ev));
    }
    true
}


/// s246 (P46): `chain_pf_att_inplace` の重ねる版。呼び出し規約と数値の契約は
/// あちらと同じで、違いは「D2H をチャンクに割り、チャンクが着いた順に
/// ゲートして H2D する」ことだけ。
///
/// 失敗時の安全性 (呼び出し側は `None` で `chain_download` 経路へ落ちる):
///   - D2H のどれかが失敗した時点では `CH_ATT` を 1 バイトも書いていないので、
///     そのまま `None` を返してよい (従来の 1 発 D2H 失敗と同一)。
///   - H2D が途中で失敗した場合は `CH_ATT` が**部分更新**になる。そのまま
///     抜けると呼び出し側のフォールバックがゲート済みの行をもう一度ゲートして
///     しまうので、残りのチャンクをゲートし終えてから**全体を 1 発の同期 H2D**
///     で戻す (pinned はその時点で完成形 = `CH_ATT` は従来と同じ内容になる)。
#[allow(clippy::too_many_arguments)]
fn pf_att_inplace_piped<F>(
    s: &mut ChainScratch,
    lib: &'static CudaLib,
    ctx: &'static CudaCtx,
    buf: CUdeviceptr,
    pin_ptr: *mut c_void,
    pin_id: usize,
    cols: usize,
    m: usize,
    bytes: usize,
    rows_per: usize,
    nch: usize,
    gate: &mut F,
) -> Option<()>
where
    F: FnMut(usize, &mut [f32]),
{
    let (rec, ev_sync) = (lib.event_record?, lib.event_sync?);
    let pin_base = pin_ptr as usize;
    // 1) 全チャンクの D2H を発行し、チャンクごとに完了イベントを打つ
    //    (既定 = null ストリーム。イベントはストリーム順に乗るだけ)。
    for k in 0..nch {
        let (r0, r1) = pf_att_chunk_bounds(k, rows_per, m);
        let off = r0.checked_mul(cols)?.checked_mul(4)?;
        let len = r1.checked_sub(r0)?.checked_mul(cols)?.checked_mul(4)?;
        let ev = s.pipe_ev.get(k)?.0;
        unsafe {
            ok(check(
                (lib.memcpy_dtoh_async)(
                    (pin_base + off) as *mut c_void,
                    buf + off as u64,
                    len,
                    std::ptr::null_mut(),
                ),
                "cuMemcpyDtoHAsync (pf att pin chunk)",
            ))?;
            if rec(ev, std::ptr::null_mut()) != CUDA_SUCCESS {
                // イベントを打てない = 完了を待てない。CH_ATT は未変更なので
                // 従来の 1 発経路へ戻す (呼び出し側のフォールバックも安全)。
                return None;
            }
        }
    }
    // s180 (P41): att の D2H 発行点 = full_core の区間 [d2h] の終端
    // (最後のチャンクの完了がこの位置に乗る。従来の 1 発と同じ意味)。
    fc_kprof_mark(3);
    // 2) チャンクごとに「完了待ち → ゲート → H2D」。ゲートはチャンクの完了前に
    //    触ってはいけない (未転送の領域を 0 で割る) ので、必ずイベントを待つ。
    let mut h2d_failed = false;
    for k in 0..nch {
        let (r0, r1) = pf_att_chunk_bounds(k, rows_per, m);
        let ev = s.pipe_ev.get(k)?.0;
        if unsafe { ev_sync(ev) } != CUDA_SUCCESS {
            // イベント待ちが使えない: 全ストリーム同期で代替する (安全側)。
            ok(ctx.sync())?;
        }
        {
            let pin = s.pins.get_mut(pin_id)?.as_mut()?;
            let v = pin.as_mut_slice::<f32>();
            gate(r0, &mut v[r0 * cols..r1 * cols]);
        }
        if h2d_failed {
            continue;
        }
        let off = r0 * cols * 4;
        let len = (r1 - r0) * cols * 4;
        let rc = unsafe {
            (lib.memcpy_htod_async)(
                buf + off as u64,
                (pin_base + off) as *const c_void,
                len,
                std::ptr::null_mut(),
            )
        };
        if rc != CUDA_SUCCESS {
            h2d_failed = true;
        }
    }
    if h2d_failed {
        // 全チャンクがゲート済み = pinned は完成形。部分更新のまま抜けると
        // フォールバックが二重ゲートするので、全体を 1 発で戻す。
        unsafe {
            ok(check(
                (lib.memcpy_htod)(buf, pin_ptr as *const c_void, bytes),
                "cuMemcpyHtoD (pf att gate fallback)",
            ))?;
        }
        ok(ctx.sync())?;
    }
    Some(())
}


/// s247 (P47): full_qk (q/k 射影) の D2H 宛先を返すビュー。
///
/// [`chain_qkv_prefill_q4_dev`] が D2H した先 — pinned ステージング
/// (`pin_pf_qkv` の 2 スライス)、またはフォールバック時の呼び出し側 `Vec` —
/// を q/k の 2 スライスとして見せる。**ホスト memcpy は挟まない** (pinned →
/// heap の写しはこのホストで 4.7 GB/s しか出ず、12.88 GB/request なら +2.7 s
/// の新規費用になる)。値は D2H で戻したバイト列そのもの。
///
/// **寿命の契約**: 同じスレッドで次の [`chain_qkv_prefill_q4_dev`] が走るまで
/// 有効 (`pin_h2` / `pins[]` のステージングと同じ規約)。層ループは 1 層ずつ
/// 「qkv チェーン → ホスト prepare/attend」を直列に回すので、1 層の中で使い切る
/// 限り競合しない。破ると**読む前に上書きされる**ので、デバッグビルドでは
/// 世代 (`PFQK_GEN`) の不一致で panic する。
pub struct PfQkView {
    q: *const f32,
    k: *const f32,
    q_len: usize,
    k_len: usize,
    pinned: bool,
    gen: u64,
}


impl PfQkView {
    /// q 投影出力 `[rows, qcols]` (pinned ステージング or フォールバックの Vec)。
    pub fn q(&self) -> &[f32] {
        debug_assert_eq!(
            self.gen,
            pf_qkv_gen(),
            "PfQkView が次の qkv チェーンで無効化された (寿命の契約違反)"
        );
        unsafe { std::slice::from_raw_parts(self.q, self.q_len) }
    }

    /// k 投影出力 `[rows, kcols]`。
    pub fn k(&self) -> &[f32] {
        debug_assert_eq!(
            self.gen,
            pf_qkv_gen(),
            "PfQkView が次の qkv チェーンで無効化された (寿命の契約違反)"
        );
        unsafe { std::slice::from_raw_parts(self.k, self.k_len) }
    }

    /// D2H の宛先が pinned ステージングだったか (計測・切り分け用)。
    pub fn pinned(&self) -> bool {
        self.pinned
    }
}


thread_local! {
    /// s247 (P47): `PfQkView` の世代。チェーンが走るたびに進む
    /// (古いビューの読み出しをデバッグビルドで検出するためだけのもの)。
    pub(crate) static PFQK_GEN: std::cell::Cell<u64> = std::cell::Cell::new(0);
}


/// ページロック失敗の報告は 1 回だけ (層ごとに 16 回出さない)。
static PFQK_PIN_FAILED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);


thread_local! {
    /// s247 (P47): pinned 経路が初めて効いたことを 1 回だけ報告するためのフラグ。
    pub(crate) static PFQK_PIN_USED: std::cell::Cell<bool> = std::cell::Cell::new(false);
}


/// s165 (P2): full attention 層の q/k/v 投影を**デバイス常駐の入力** (`CH_PFH`,
/// rmsnorm 出力・先頭から `rows` 行) から回し、**出力だけ**をホストへ戻す。
/// KV キャッシュはホストが正で q/k/v の値はホストの prepare が要るので、
/// 値は必ず戻す (戻す量は q/k/v の幅で、h2 の H2D 3 回より小さい)。
/// 3 本の GEMM は `CH_QKV` の連続領域 (q|k|v) に行オフセットで書き、
/// D2H 3 発 + sync 1 回で落とす。
/// カーネル・grid・引数は per-op 経路と同一 (行ごとに独立 → bit 一致)。
///
/// s247 (P47): D2H の宛先 (q/k) を **pinned ステージング** (`pin_pf_qkv`) にし、
/// 戻り値の [`PfQkView`] でそのスライスをそのまま呼び出し側へ渡す。
/// 発行回数 (3 発)・発行順 (q → k → v)・転送バイト数・`ctx.sync()` 1 回は従来と
/// 同一で、変わるのは「同じバイト列をどこへ置くか」だけ = 数値は不変。
/// v の宛先は従来どおり呼び出し側のバッファ (s240/P44: ホスト KV キャッシュの行
/// そのもの = prepare が v を 1 バイトも触らない)。pinned を確保できないときは
/// 呼び出し側の `Vec` へ直接 D2H する従来経路に落ちる (fail-closed・数値不変)。
pub fn chain_qkv_prefill_q4_dev(
    q_w: &crate::quant::Linear,
    k_w: &crate::quant::Linear,
    v_w: &crate::quant::Linear,
    rows: usize,
    li: usize,
    // s247 (P47): q/k の D2H 宛先。pinned ステージングを確保できた層では
    // **読まれない** (pageable フォールバックの宛先としてだけ使い、そのときだけ
    // 長さを合わせる) — pinned が効く層で 747 MB の mmap をしないため。
    out_qg: &mut Vec<f32>,
    out_k: &mut Vec<f32>,
    out_v: &mut [f32],
) -> Option<PfQkView> {
    if !cuda_enabled() || !gemm_enabled() || GPU_BROKEN.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    let (qn, qcols) = q4_dims(q_w)?;
    let (kn, kcols) = q4_dims(k_w)?;
    let (vn, vcols) = q4_dims(v_w)?;
    if qn != kn || qn != vn || qn == 0 || qn % 32 != 0 {
        return None;
    }
    if qcols % 32 != 0 || kcols % 32 != 0 || vcols % 32 != 0 {
        return None;
    }
    if rows < gemm_min_m() || rows <= max_m() {
        return None;
    }
    if out_v.len() < rows * vcols {
        return None;
    }
    let ctx = chain_ctx()?;
    let (q_uid, q_blk, _, _) = q4_parts(q_w)?;
    let (k_uid, k_blk, _, _) = q4_parts(k_w)?;
    let (v_uid, v_blk, _, _) = q4_parts(v_w)?;
    let gwq = weights_for_q4(q_uid, q_blk, qn, qcols).ok()?;
    let gwk = weights_for_q4(k_uid, k_blk, kn, kcols).ok()?;
    let gwv = weights_for_q4(v_uid, v_blk, vn, vcols).ok()?;
    let total = rows.checked_mul(qcols.checked_add(kcols)?.checked_add(vcols)?)?;
    ok(chain_ensure_buf(CH_QKV, total.checked_mul(4)?))?;
    // 入力は CH_PFH の先頭 rows 行 (rmsnorm 出力) — 行数ぶんの領域を要求
    ok(chain_ensure_buf(CH_PFH, rows.checked_mul(qn)?.checked_mul(4)?))?;
    let (pin, pqkv) = CHAIN.with(|cell| -> Option<(u64, u64)> {
        let s = cell.borrow();
        Some((s.bufs[CH_PFH].as_ref()?.ptr(), s.bufs[CH_QKV].as_ref()?.ptr()))
    })?;
    let q_off = (rows * qcols * 4) as u64;
    let k_off = (rows * (qcols + kcols) * 4) as u64;
    // s169 (P8): qkv GEMM + 3 発の D2H + 既存の sync までが full_qk の区間。
    pf_phase(PF_PH_FULL_QK);
    // s250 (P53): q の D2H を発行するストリーム。作れなければ null のまま =
    // 従来どおり既定ストリームで 3 発を直列に流す (数値は同一)。
    let mut q_stream: CUstream = std::ptr::null_mut();
    crate::prof::time_layer(crate::prof::QKV, li, || -> Option<()> {
        chain_gemm_q4_pf(gwq, pin, pqkv, rows, qn, qcols)?;
        // s250 (P53): q の GEMM 完了をここで打ち、コピーストリームを参加させる。
        // k/v の GEMM は既定ストリームでこの後ろに続くので、q の D2H
        // (rows*qcols*4 B) がその後ろに隠れる。記録点を k/v の起動より後にすると
        // イベントが 3 発全部の完了を待ち、1 バイトも重ならない。
        q_stream = pf_qk_copy_begin().unwrap_or(std::ptr::null_mut());
        chain_gemm_q4_pf(gwk, pin, pqkv + q_off, rows, kn, kcols)?;
        chain_gemm_q4_pf(gwv, pin, pqkv + k_off, rows, vn, vcols)
    })?;
    // ---- s247 (P47): D2H の宛先 (q/k) を pinned ステージングにする ----
    // プールは 1 本 (`pin_pf_qkv`) を q/k の 2 スライスに割り、grow-only で層を
    // またいで使い回す (ページロックは最初の full 層の 1 回だけ)。上限超過・
    // ページロック失敗は従来の pageable 経路へ落ちる (fail-closed)。
    let q_elems = rows.checked_mul(qcols)?;
    let k_elems = rows.checked_mul(kcols)?;
    let q_bytes = q_elems.checked_mul(4)?;
    let k_bytes = k_elems.checked_mul(4)?;
    let stage_bytes = q_bytes.checked_add(k_bytes)?;
    let mut dst_q: *mut c_void = std::ptr::null_mut();
    let mut dst_k: *mut c_void = std::ptr::null_mut();
    let mut pinned = false;
    if stage_bytes <= pf_qkv_pin_max() {
        CHAIN.with(|cell| {
            let mut s = cell.borrow_mut();
            if Scratch::ensure_pin(&mut s.pin_pf_qkv, stage_bytes).is_err() {
                // pinned の失敗は致命的でない (`pin_h2` と同じ扱い = GPU_BROKEN を
                // 立てない)。理由だけ 1 回出す。
                if !PFQK_PIN_FAILED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!(
                        "full_qk pinned staging: cuMemHostAlloc({stage_bytes} B) に失敗 → pageable 経路"
                    );
                }
                return;
            }
            if let Some(p) = s.pin_pf_qkv.as_ref() {
                if p.bytes >= stage_bytes {
                    dst_q = p.ptr;
                    // q と k は同じプールの連続した 2 スライス (q_off=0, k_off=q_bytes)
                    dst_k = unsafe { (p.ptr as *mut u8).add(q_bytes) as *mut c_void };
                    pinned = true;
                }
            }
        });
        if pinned && pf_qkv_pin_dbg() && !PFQK_PIN_USED.with(|c| c.replace(true)) {
            eprintln!(
                "full_qk pinned staging: {stage_bytes} B (q {q_bytes} + k {k_bytes}, n={rows}) — D2H 宛先 = pinned"
            );
        }
    }
    if !pinned {
        // フォールバック = 現行経路そのもの (呼び出し側の Vec へ直接 D2H)。
        // 長さはここで合わせる: pinned が効く層では 747 MB の確保をしない。
        if out_qg.len() < q_elems {
            out_qg.clear();
            out_qg.resize(q_elems, 0.0);
        }
        if out_k.len() < k_elems {
            out_k.clear();
            out_k.resize(k_elems, 0.0);
        }
        dst_q = out_qg.as_mut_ptr() as *mut c_void;
        dst_k = out_k.as_mut_ptr() as *mut c_void;
    }
    // D2H 3 発 (async) → sync 1 回 — 発行回数・発行順は従来と同一 (連結しない)
    CHAIN.with(|cell| -> Option<()> {
        let s = cell.borrow();
        let base = s.bufs[CH_QKV].as_ref()?.ptr();
        let lib = lib().ok()?;
        unsafe {
            ok(check(
                (lib.memcpy_dtoh_async)(dst_q, base, q_bytes, q_stream),
                "cuMemcpyDtoHAsync (pf qkv q)",
            ))?;
            ok(check(
                (lib.memcpy_dtoh_async)(
                    dst_k,
                    base + q_off,
                    k_bytes,
                    std::ptr::null_mut(),
                ),
                "cuMemcpyDtoHAsync (pf qkv k)",
            ))?;
            ok(check(
                (lib.memcpy_dtoh_async)(
                    out_v.as_mut_ptr() as *mut c_void,
                    base + k_off,
                    rows * vcols * 4,
                    std::ptr::null_mut(),
                ),
                "cuMemcpyDtoHAsync (pf qkv v)",
            ))
        }
    })?;
    crate::prof::mark_d2h_src(crate::prof::D2H_SRC_LINEAR);
    ok(ctx.sync())?;
    // ここまでで D2H は完了している (sync 1 回)。ビューの世代を進めて
    // 前のビューを無効化する。
    let gen = PFQK_GEN.with(|c| {
        let g = c.get().wrapping_add(1);
        c.set(g);
        g
    });
    Some(PfQkView {
        q: dst_q as *const f32,
        k: dst_k as *const f32,
        q_len: q_elems,
        k_len: k_elems,
        pinned,
        gen,
    })
}


/// s165 (P2): attention 出力 (`CH_ATT`) → o_proj (q4) → `CH_O` (デバイス常駐・
/// 同期なし)。per-op 経路 (`maybe_gpu_linear_q4_into`) と同一カーネル・同一引数。
/// m を 1 回で渡すのは、行ごとに独立で 1 行の i32 累積順が変わらないため
/// (`chain_gdn_prefill_q4` の o_proj と同じ根拠)。
pub fn chain_oproj_prefill_q4(o_w: &crate::quant::Linear, rows: usize, li: usize) -> Option<()> {
    let ow = match o_w {
        crate::quant::Linear::Q4(q) => q,
        _ => return None,
    };
    if !cuda_enabled() || !gemm_enabled() || GPU_BROKEN.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    if rows < gemm_min_m() || rows <= max_m() || ow.k == 0 || ow.n == 0 || ow.k % 32 != 0 {
        return None;
    }
    let ctx = chain_ctx()?;
    let gw = weights_for_q4(ow.uid, &ow.blocks, ow.k, ow.n).ok()?;
    let out_bytes = rows.checked_mul(ow.n)?.checked_mul(4)?;
    let att_bytes = rows.checked_mul(ow.k)?.checked_mul(4)?;
    ok(chain_ensure_buf(CH_O, out_bytes))?;
    let (pa, po) = CHAIN.with(|cell| -> Option<(u64, u64)> {
        let s = cell.borrow();
        let a = s.bufs[CH_ATT].as_ref()?;
        if a.bytes() < att_bytes {
            return None;
        }
        Some((a.ptr(), s.bufs[CH_O].as_ref()?.ptr()))
    })?;
    let _ = ctx;
    // s169 (P8): att の H2D は呼び出し側、ここは o_proj GEMM だけの区間。
    pf_phase(PF_PH_FULL_OPROJ);
    crate::prof::time_layer(crate::prof::OPROJ, li, || chain_gemm_q4_pf(gw, pa, po, rows, ow.k, ow.n))
}
