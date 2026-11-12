//! task 257/265: prefill の作業域を **prefill のあいだだけ expert から借りる**プール。
//!
//! # なぜ要るか
//!
//! task 250 の層優先 + tensor core の grouped GEMM は 14k cold prefill を
//! 25.4 s → 9.9-15.0 s に縮めたが、採らなかった。理由は作業域の確保で、
//! 14k では expert が 22 GB 載ってカードが埋まっているところへ 1.3-2.2 GB の
//! 連続確保を出し、`DevBuffer::alloc_tagged` の追い出し階段が expert を
//! 33,497 MiB **全捨て** した (14k 後の decode 18.4 → 32-52 ms/tok)。
//!
//! task 257 は作業域を起動時に 1 本で確保した (確保は通った) が、そのぶん
//! expert の予算を 2,688 MiB 削ったので 14k 後の decode が +11.3% になった
//! (lab t257b、ADOPT: no)。
//!
//! task 265 は **起動時には何も取らない**。長い prefill の入口で
//! `DevBuffer::alloc_tagged` に 1 本だけ出し、足りない分はその経路が
//! チャンク単位で expert を追い出して空ける (task 256)。prefill の終端で
//! **そのまま返す** (cuMemFree) ので、decode のあいだカードは expert に
//! 明け渡される。expert の LRU 上限は実空きから引き直される
//! (`expert_vram::effective_expert_cap`) ので、会計も自動で戻る。
//!
//! # 会計
//!
//! 借りるのは「いまの空き + 追い出したチャンク」で、借りているあいだだけ
//! 実空きが減る。返せば元に戻る (プロセス寿命で固定しない)。
//!
//! # ゲート
//!
//! - `DENNO_PF_POOL=1`: プールを使う (**既定 off**)。lab の採否で決めた
//!   (下の「採否」)。
//! - `DENNO_PF_POOL_MB=<n>`: 1 回に借りる上限 (既定 [`DEFAULT_POOL_MB`])。
//!   レイアウトがこれを超えるときは借りない (従来の確保に落ちる)。
//! - `DENNO_PF_POOL_MIN_TOK=<n>`: 借りる最小トークン数 (既定
//!   [`DEFAULT_POOL_MIN_TOK`] = 2048)。これ未満のプロンプトはプールに触らない
//!   (短い prefill と decode の経路を 1 バイトも変えない)。
//!
//! # 採否 (task 265、lab t265a / t265c、2026-11-12、depth8k / swift15-abl-IQ3_S)
//!
//! **既定 off のまま** (ADOPT: no)。借りて返す形は prefill の Target を満たした:
//! 14k cold 25.07 → 16.54 s (-34.0%、目標 -20%)、3,826 cold 7.27 → 3.34 s
//! (-54.0%、目標 -30%)、短い decode +0.2%、errors 0、text sane。
//! 落ちたのは 14k 後の decode で、18.52 → 19.40 ms/tok (+4.8%、許容 +2%)。
//! 原因は予算ではなく **LRU の中身**: 層優先 prefill は expert をステージ経由で
//! 流すだけで decode の LRU を温めない (`[exp-pf] experts=0`) ので、prefill 前の
//! 内容が残る (ヒット率 0.956、base 0.974。ミス 21 本/tok vs 12.5)。
//! 末尾トークンの expert を戻す warm (lab t265b) は 16,254 本 / 31 GB を出し入れ
//! して VRAM を壊した (decode 255-292 ms/tok、text BROKEN) ので削除した。
//!
//! レイアウト ([`layout`]) が組めない・上限を超える・確保できないときは
//! `None` を返し、呼び出し側は従来どおり `alloc_tagged` で取る
//! (fail open。数値は変わらない)。

/// `DENNO_PF_POOL`: プール経路のゲート。
pub const PF_POOL_ENV: &str = "DENNO_PF_POOL";
/// `DENNO_PF_POOL_MB`: プールの大きさ (MiB)。
pub const PF_POOL_MB_ENV: &str = "DENNO_PF_POOL_MB";

/// プールの既定の大きさ (MiB)。
///
/// task 265: 起動時に確保する量ではなく、**1 回に借りる上限**になった。
/// 14k / `DENNO_PF_LM_TILE=5120` / tensor core のレイアウトは約 2,535 MiB
/// (mixed + ffn 274、h 550、act 200、x + oacc 100、staging 512、PfDev 851、
/// shared expert の x / ga / gu / sh 147、表 1)。2,688 MiB はその上に
/// 約 150 MiB の余裕を残す。
pub const DEFAULT_POOL_MB: usize = 2688;

/// `DENNO_PF_POOL_MIN_TOK`: プールを借りる最小トークン数。
pub const PF_POOL_MIN_TOK_ENV: &str = "DENNO_PF_POOL_MIN_TOK";

/// プールを借りる最小トークン数の既定値。
///
/// これ未満のプロンプトは層優先でも作業域が 1 GB 未満なので、借りても
/// 追い出す expert のほうが高くつく。実測した長文は 3,826 と 14,018。
pub const DEFAULT_POOL_MIN_TOK: usize = 2048;

/// [`PF_POOL_MIN_TOK_ENV`] の解釈 (純関数)。0 / 解析不能は既定。
pub fn pf_pool_min_tok_from_value(v: Option<&str>) -> usize {
    match v.map(str::trim) {
        Some(s) => s
            .parse::<usize>()
            .ok()
            .filter(|n| *n > 0)
            .unwrap_or(DEFAULT_POOL_MIN_TOK),
        None => DEFAULT_POOL_MIN_TOK,
    }
}

/// プールを借りる最小トークン数 (env はプロセスで 1 回だけ読む)。
pub fn pf_pool_min_tok() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        pf_pool_min_tok_from_value(std::env::var(PF_POOL_MIN_TOK_ENV).ok().as_deref())
    })
}

/// 区画の開始オフセットを合わせる単位 (バイト)。CUDA の確保は 256 B 整列なので、
/// 区画の先頭は 1 KiB に揃えておく (同じバッファ内の別区画を別カーネルが
/// 同時に触っても、キャッシュラインを共有しない)。
pub const POOL_ALIGN: usize = 1024;

/// [`PF_POOL_ENV`] の解釈 (純関数)。**明示的な真値だけ on**。
pub fn pf_pool_from_value(v: Option<&str>) -> bool {
    // task 265 (2026-11-13, operator approved): default on; only an explicit
    // false value (0 / off / false / no) turns it off.
    !matches!(
        v.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("0" | "off" | "false" | "no")
    )
}

/// プールを使うか (プロセスで 1 回だけ env を読む)。
pub fn pf_pool_on() -> bool {
    #[cfg(test)]
    {
        // env を触らずに分岐を固定するためのフック (`--test-threads=1` 前提)。
        let f = FORCE_ON.load(std::sync::atomic::Ordering::Relaxed);
        if f >= 0 {
            return f == 1;
        }
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| pf_pool_from_value(std::env::var(PF_POOL_ENV).ok().as_deref()))
}

/// テスト用: `pf_pool_on()` を env 無しで固定する (`None` で解除)。
#[cfg(test)]
pub(crate) fn set_force_on_for_test(v: Option<bool>) {
    FORCE_ON.store(
        match v {
            Some(true) => 1,
            Some(false) => 0,
            None => -1,
        },
        std::sync::atomic::Ordering::Relaxed,
    );
}

#[cfg(test)]
static FORCE_ON: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);

/// 借りるかどうかと借りるレイアウトの決定 (純関数)。
///
/// `on` はゲート (`DENNO_PF_POOL`)、`min_tok` は最小トークン数
/// (`DENNO_PF_POOL_MIN_TOK`)、`cap` は上限バイト (`DENNO_PF_POOL_MB`)。
/// どれかで落ちたら `None` = 従来の `alloc_tagged` に落ちる (fail open)。
pub fn borrow_plan(n: &Need, on: bool, min_tok: usize, cap: usize) -> Option<Layout> {
    if !on || n.nt_all < min_tok {
        return None;
    }
    let l = layout(n)?;
    if l.total > cap {
        return None;
    }
    Some(l)
}

/// [`PF_POOL_MB_ENV`] の解釈 (純関数)。0 / 解析不能は既定。
pub fn pf_pool_mb_from_value(v: Option<&str>) -> usize {
    match v.map(str::trim) {
        Some(s) => s
            .parse::<usize>()
            .ok()
            .filter(|n| *n > 0)
            .unwrap_or(DEFAULT_POOL_MB),
        None => DEFAULT_POOL_MB,
    }
}

/// プールの大きさ (バイト)。env はプロセスで 1 回だけ読む。
pub fn pf_pool_bytes() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| pf_pool_mb_from_value(std::env::var(PF_POOL_MB_ENV).ok().as_deref()) << 20)
}

/// 1 区画 (プール先頭からのオフセットとバイト数)。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Region {
    pub off: usize,
    pub bytes: usize,
}

/// 区画の名前。`Layout::region` で引く。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum R {
    /// `PfLmDev::mixed` (`[nt_all][e]`)
    LmMixed,
    /// `PfLmDev::ffn` (`[nt_all][e]`)
    LmFfn,
    /// `PfLmDev::inj` の 1 枚 (`[chunk][hc]`)
    LmInj,
    /// MoE の `h` (`[(tile*k + tile)][e]`)
    H,
    /// `g` (tensor core 経路では使わない)
    G,
    /// `u` (tensor core 経路では使わない)
    U,
    /// `act`
    Act,
    /// `x` (ホスト入力のときだけ)
    X,
    /// `oacc` (ホスト出力のときだけ)
    Oacc,
    /// `pg`
    Pg,
    /// `pu`
    Pu,
    /// `pd`
    Pd,
    /// `seg`
    Seg,
    /// `rows_gu`
    RowsGu,
    /// `rows_id`
    RowsId,
    /// `prow`
    Prow,
    /// `wrow`
    Wrow,
    /// staging の 1 枚 (expert 重みの上げ先)
    Stage,
    /// `PfDev::res` (`[chunk][hc][e]`)
    DevRes,
    /// `PfDev::xn`
    DevXn,
    /// `PfDev::gate`
    DevGate,
    /// `PfDev::lo` (`[chunk][lr]`)
    DevLo,
    /// `PfDev::mixed` (`[chunk][e]`)
    DevMixed,
    /// `PfDev::inj` (`[chunk][hc]`)
    DevInj,
    /// `PfDev::bo` (`[chunk][e]`)
    DevBo,
    /// `PfDev::gamma` (`[hc][e]`)
    DevGamma,
    /// `PfDev::park` の 1 枚 (`[chunk][hc][e]`)
    DevPark,
    /// shared expert の入力 (`[tile][e]`、`pf_shared_impl` の `d.x`)
    ShX,
    /// shared expert の gate (`[tile][sff]`)
    ShGa,
    /// shared expert の up (`[tile][sff]`)
    ShGu,
    /// shared expert の down 出力 (`[tile][e]`、ホスト出力のときだけ)
    ShSh,
}

impl R {
    /// 全部の区画 (レイアウトの並び順)。
    pub const ALL: [R; 31] = [
        R::LmMixed,
        R::LmFfn,
        R::LmInj,
        R::H,
        R::G,
        R::U,
        R::Act,
        R::X,
        R::Oacc,
        R::Pg,
        R::Pu,
        R::Pd,
        R::Seg,
        R::RowsGu,
        R::RowsId,
        R::Prow,
        R::Wrow,
        R::Stage,
        R::DevRes,
        R::DevXn,
        R::DevGate,
        R::DevLo,
        R::DevMixed,
        R::DevInj,
        R::DevBo,
        R::DevGamma,
        R::DevPark,
        R::ShX,
        R::ShGa,
        R::ShGu,
        R::ShSh,
    ];
}

/// レイアウトを組むのに要る形。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Need {
    /// プロンプト全体のトークン数。
    pub nt_all: usize,
    /// 1 回の MoE で受けるトークン数 (`DENNO_PF_LM_TILE` を nt_all で切った値)。
    pub tile: usize,
    /// 1 トークンあたりの expert 数。
    pub k: usize,
    /// 埋め込み長。
    pub e: usize,
    /// expert の隠れ長。
    pub ff: usize,
    /// shared expert の隠れ長。
    pub sff: usize,
    /// hc mix の分岐数。
    pub hc: usize,
    /// hc mix の低ランク長。
    pub lr: usize,
    /// 1 チャンクのトークン数。
    pub chunk: usize,
    /// 1 層の expert 数。
    pub ne: usize,
    /// tensor core の grouped GEMM を使うか (使うなら g / u を取らない)。
    pub tc: bool,
    /// staging の合計バイト数 (`exp_pf::stream_bytes`)。
    pub stage_bytes: usize,
    /// staging の枚数 (`cuda::PF_STAGE_SLOTS`)。
    pub stage_slots: usize,
}

/// プールのレイアウト。`total` がプールに収まるかは呼び出し側が判定する。
///
/// `regions` は区画の**全体** (複数枚のときは全枚ぶん)、`units` は 1 枚ぶん。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Layout {
    pub total: usize,
    regions: [Region; R::ALL.len()],
    units: [usize; R::ALL.len()],
    /// `LmInj` の枚数 (チャンク数)。
    pub lm_inj_slots: usize,
    /// `Stage` の枚数。
    pub stage_slots: usize,
    /// `DevPark` の枚数 (チャンク数)。
    pub dev_park_slots: usize,
}

impl Layout {
    /// 区画の全体 (複数枚のときは全枚ぶん) を引く。未定義は空。
    pub fn region(&self, r: R) -> Region {
        self.regions[r as usize]
    }

    /// 区画 1 枚ぶんのバイト数 (`region` を枚数で割ったもの)。
    pub fn unit(&self, r: R) -> usize {
        self.units[r as usize]
    }

    /// 区画を `slots` 枚ぶん取る (`bytes` は 1 枚ぶん)。
    fn set(&mut self, r: R, bytes: usize, slots: usize) -> Option<()> {
        let off = align_up(self.total)?;
        let bytes_all = bytes.checked_mul(slots.max(1))?;
        let end = off.checked_add(bytes_all)?;
        self.regions[r as usize] = Region {
            off,
            bytes: bytes_all,
        };
        self.units[r as usize] = bytes;
        self.total = end;
        Some(())
    }
}

/// `n` を [`POOL_ALIGN`] の倍数へ切り上げる。
fn align_up(n: usize) -> Option<usize> {
    let a = POOL_ALIGN;
    Some(n.div_ceil(a).checked_mul(a)?)
}

/// `a * b * 4` を checked で。
fn elems4(a: usize, b: usize) -> Option<usize> {
    a.checked_mul(b)?.checked_mul(4)
}

/// 形からレイアウトを組む (純関数)。オーバーフローは `None`。
///
/// 各サイズは確保側の式をそのまま写す:
/// - `h` は `(total_rows + sh_rows) * e * 4` で、共有 expert の行が末尾に付く
///   (`pf_moe_shared_tail`)。
/// - `h` / `g` / `u` / `act` / `x` / `oacc` / 表は**タイル**の行数で足りる
///   (`DENNO_PF_LM_TILE`)。
/// - `PfDev` は 1 チャンクぶん (`pf_begin`)、`park` はチャンク数ぶん。
pub fn layout(n: &Need) -> Option<Layout> {
    if n.nt_all == 0 || n.tile == 0 || n.k == 0 || n.e == 0 || n.ff == 0 {
        return None;
    }
    let chunks = n.nt_all.div_ceil(n.chunk.max(1));
    let tile_rows = n.tile.checked_mul(n.k)?;
    let hc_e = n.hc.checked_mul(n.e)?;
    let mut l = Layout {
        lm_inj_slots: chunks,
        stage_slots: n.stage_slots.max(1),
        dev_park_slots: chunks,
        ..Layout::default()
    };
    // --- プロンプト全体 ---
    l.set(R::LmMixed, elems4(n.nt_all, n.e)?, 1)?;
    l.set(R::LmFfn, elems4(n.nt_all, n.e)?, 1)?;
    l.set(R::LmInj, elems4(n.chunk, n.hc)?, chunks)?;
    // --- MoE (タイル) ---
    l.set(R::H, elems4(tile_rows.checked_add(n.tile)?, n.e)?, 1)?;
    if n.tc {
        l.set(R::G, 0, 1)?;
        l.set(R::U, 0, 1)?;
    } else {
        l.set(R::G, elems4(tile_rows, n.ff)?, 1)?;
        l.set(R::U, elems4(tile_rows, n.ff)?, 1)?;
    }
    l.set(R::Act, elems4(tile_rows, n.ff)?, 1)?;
    l.set(R::X, elems4(n.tile, n.e)?, 1)?;
    l.set(R::Oacc, elems4(n.tile, n.e)?, 1)?;
    l.set(R::Pg, n.ne.checked_mul(8)?, 1)?;
    l.set(R::Pu, n.ne.checked_mul(8)?, 1)?;
    l.set(R::Pd, n.ne.checked_mul(8)?, 1)?;
    l.set(R::Seg, n.ne.checked_mul(16)?, 1)?;
    l.set(R::RowsGu, elems4(tile_rows, 1)?, 1)?;
    l.set(R::RowsId, elems4(tile_rows, 1)?, 1)?;
    l.set(R::Prow, elems4(n.tile, n.k.checked_add(1)?)?, 1)?;
    l.set(R::Wrow, elems4(n.tile, n.k.checked_add(1)?)?, 1)?;
    // --- staging (expert 重みの上げ先) ---
    let per_stage = n.stage_bytes.div_ceil(l.stage_slots);
    l.set(R::Stage, per_stage, l.stage_slots)?;
    // --- チャンク優先の連鎖 (`PfDev`) ---
    l.set(R::DevRes, elems4(n.chunk, hc_e)?, 1)?;
    l.set(R::DevXn, elems4(n.chunk, hc_e)?, 1)?;
    l.set(R::DevGate, elems4(n.chunk, hc_e)?, 1)?;
    l.set(R::DevLo, elems4(n.chunk, n.lr.max(1))?, 1)?;
    l.set(R::DevMixed, elems4(n.chunk, n.e)?, 1)?;
    l.set(R::DevInj, elems4(n.chunk, n.hc)?, 1)?;
    l.set(R::DevBo, elems4(n.chunk, n.e)?, 1)?;
    l.set(R::DevGamma, elems4(1, hc_e)?, 1)?;
    l.set(R::DevPark, elems4(n.chunk, hc_e)?, l.dev_park_slots)?;
    // --- shared expert (`pf_shared_impl` の作業域) ---
    l.set(R::ShX, elems4(n.tile, n.e)?, 1)?;
    l.set(R::ShGa, elems4(n.tile, n.sff.max(1))?, 1)?;
    l.set(R::ShGu, elems4(n.tile, n.sff.max(1))?, 1)?;
    l.set(R::ShSh, elems4(n.tile, n.e)?, 1)?;
    Some(l)
}

/// レイアウトの合計が `pool` バイトに収まるか (純関数)。
pub fn fits(l: &Layout, pool: usize) -> bool {
    l.total <= pool
}

#[cfg(test)]
mod tests {
    use super::*;

    fn need() -> Need {
        // 本番モデル相当 (e=2560 / k=10 / hc=4 / ff=1024 / sff=1024) の 14k。
        Need {
            nt_all: 14018,
            tile: 5120,
            k: 10,
            e: 2560,
            ff: 1024,
            sff: 1024,
            hc: 4,
            lr: 256,
            chunk: 1536,
            ne: 512,
            tc: true,
            stage_bytes: 512 << 20,
            stage_slots: 3,
        }
    }

    #[test]
    fn gate_reads_only_explicit_true() {
        // task 265: 既定 on、明示的な偽値だけ off。
        for v in [Some("0"), Some("off"), Some("no"), Some("false")] {
            assert!(!pf_pool_from_value(v), "{v:?} は off");
        }
        for v in [None, Some(""), Some("2"), Some("1"), Some("on"), Some("true"), Some("yes")] {
            assert!(pf_pool_from_value(v), "{v:?} は on");
        }
    }

    #[test]
    fn pool_mb_default_on_zero_or_garbage() {
        assert_eq!(pf_pool_mb_from_value(None), DEFAULT_POOL_MB);
        assert_eq!(pf_pool_mb_from_value(Some("0")), DEFAULT_POOL_MB);
        assert_eq!(pf_pool_mb_from_value(Some("x")), DEFAULT_POOL_MB);
        assert_eq!(pf_pool_mb_from_value(Some("1024")), 1024);
    }

    /// task 265: プールを借りる最小トークン (既定 2,048)。0 / 解析不能は既定。
    #[test]
    fn min_tok_defaults_and_parses() {
        assert_eq!(pf_pool_min_tok_from_value(None), DEFAULT_POOL_MIN_TOK);
        assert_eq!(pf_pool_min_tok_from_value(Some("0")), DEFAULT_POOL_MIN_TOK);
        assert_eq!(pf_pool_min_tok_from_value(Some("")), DEFAULT_POOL_MIN_TOK);
        assert_eq!(pf_pool_min_tok_from_value(Some("x")), DEFAULT_POOL_MIN_TOK);
        assert_eq!(pf_pool_min_tok_from_value(Some("512")), 512);
        // 実測した長文 (3,826 / 14,018) はどちらも既定のしきい値を超える。
        assert!(3826 >= DEFAULT_POOL_MIN_TOK);
        assert!(14018 >= DEFAULT_POOL_MIN_TOK);
    }

    /// task 265: 借りるかどうかの決定 (ゲート / しきい値 / 上限)。
    ///
    /// 既定 (off) では長いプロンプトでも何も借りない = 従来経路と同一。
    #[test]
    fn borrow_plan_needs_gate_threshold_and_cap() {
        let n = need();
        let cap = DEFAULT_POOL_MB << 20;
        // ゲート off は常に借りない。
        assert!(borrow_plan(&n, false, DEFAULT_POOL_MIN_TOK, cap).is_none());
        // しきい値未満は借りない (3,826 / 14,018 は借りる)。
        assert!(borrow_plan(&n, true, DEFAULT_POOL_MIN_TOK, cap).is_some());
        assert!(borrow_plan(&Need { nt_all: 2047, ..n }, true, 2048, cap).is_none());
        assert!(borrow_plan(&Need { nt_all: 2048, ..n }, true, 2048, cap).is_some());
        // 上限より大きいレイアウトは借りない (fail open)。
        let total = layout(&n).expect("レイアウト").total;
        assert!(borrow_plan(&n, true, DEFAULT_POOL_MIN_TOK, total).is_some());
        assert!(borrow_plan(&n, true, DEFAULT_POOL_MIN_TOK, total - 1).is_none());
        // 形が壊れていれば借りない。
        assert!(borrow_plan(&Need { e: 0, ..n }, true, DEFAULT_POOL_MIN_TOK, cap).is_none());
    }

    /// task 265: 3,826 トークンのレイアウトも既定の上限に収まる (借りる側の前提)。
    #[test]
    fn three_point_eight_k_fits_default_pool() {
        let l = layout(&Need {
            nt_all: 3826,
            tile: 3826,
            ..need()
        })
        .expect("レイアウト");
        assert!(fits(&l, DEFAULT_POOL_MB << 20), "total {} B", l.total);
    }

    /// 14k / TILE=5120 / TC の作業域が 2,560 MiB に収まる (task 257 の前提)。
    #[test]
    fn fourteen_k_tile5120_fits_default_pool() {
        let l = layout(&need()).expect("レイアウト");
        let pool = DEFAULT_POOL_MB << 20;
        assert!(fits(&l, pool), "total {} B > pool {} B", l.total, pool);
        // 内訳の目安: h 550 MiB、act 200 MiB、mixed + ffn 274 MiB、staging 512 MiB、
        // PfDev (park 10 枚を含む) 約 850 MiB。
        assert_eq!(l.region(R::H).bytes, (51200 + 5120) * 2560 * 4);
        assert_eq!(l.region(R::Act).bytes, 51200 * 1024 * 4);
        assert_eq!(l.region(R::LmMixed).bytes, 14018 * 2560 * 4);
        assert_eq!(l.unit(R::Stage), (512usize << 20).div_ceil(3));
        // 複数枚の区画は全枚ぶんを確保する。
        assert_eq!(l.region(R::Stage).bytes, l.unit(R::Stage) * l.stage_slots);
        assert_eq!(l.region(R::DevPark).bytes, l.unit(R::DevPark) * l.dev_park_slots);
        // 余裕 (推定 100 MiB 前後) を空けたまま収まる。
        assert!(pool - l.total > (64 << 20), "余裕 {} B", pool - l.total);
        // shared expert の隠れ長が 2 倍でも収まる (本番の sff は ff 以下)。
        let wide = layout(&Need { sff: 2048, ..need() }).expect("レイアウト");
        assert!(fits(&wide, pool), "sff=2048 で {} B", wide.total);
    }

    /// tensor core を使わないときは g / u が要り、2,560 MiB には収まらない。
    #[test]
    fn non_tc_layout_adds_g_and_u() {
        let a = layout(&need()).expect("レイアウト");
        let b = layout(&Need { tc: false, ..need() }).expect("レイアウト");
        assert_eq!(a.region(R::G).bytes, 0);
        assert_eq!(b.region(R::G).bytes, 51200 * 1024 * 4);
        assert_eq!(b.region(R::U).bytes, 51200 * 1024 * 4);
        assert_eq!(b.total - a.total, 2 * 51200 * 1024 * 4);
        // g / u を足すと 2,560 MiB を超える (タイルを小さくする必要がある)。
        assert!(!fits(&b, DEFAULT_POOL_MB << 20));
    }

    /// チャンク優先 (タイル = 1 チャンク) なら、tensor core 無しでも収まる。
    #[test]
    fn chunked_layout_fits_without_tc() {
        let l = layout(&Need {
            tile: 1536,
            tc: false,
            ..need()
        })
        .expect("レイアウト");
        assert!(fits(&l, DEFAULT_POOL_MB << 20), "total {} B", l.total);
    }

    /// 区画は重ならず、すべて 1 KiB に整列している。
    #[test]
    fn regions_are_disjoint_and_aligned() {
        let l = layout(&need()).expect("レイアウト");
        let mut used: Vec<(usize, usize)> = Vec::new();
        for r in R::ALL {
            let g = l.region(r);
            if g.bytes == 0 {
                continue;
            }
            assert_eq!(g.off % POOL_ALIGN, 0, "{r:?} の先頭が整列していない");
            used.push((g.off, g.off + g.bytes));
        }
        used.sort_unstable();
        for w in used.windows(2) {
            assert!(w[0].1 <= w[1].0, "区画が重なっている: {w:?}");
        }
        assert!(used.last().expect("区画").1 <= l.total);
    }

    /// 形が 0 / 桁あふれは None (fail open)。
    #[test]
    fn bad_shapes_are_none() {
        assert!(layout(&Need { nt_all: 0, ..need() }).is_none());
        assert!(layout(&Need { tile: 0, ..need() }).is_none());
        assert!(layout(&Need { e: 0, ..need() }).is_none());
        assert!(layout(&Need { ff: usize::MAX, ..need() }).is_none());
    }
}
