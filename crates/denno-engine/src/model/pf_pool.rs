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

/// プールの大きさ (バイト)。
///
/// task 814: `DENNO_PF_POOL_MB` が未設定ならメモリ空きに応じた配分
/// ([`crate::model::pf_budget`]) を使う (空きが十分なら [`DEFAULT_POOL_MB`])。
/// 借りるたびに引くので、長い prefill の途中で空きが減っても次の借用から
/// 小さくなる (`borrow_plan` は上限を超えると従来の確保へ落ちる)。
pub fn pf_pool_bytes() -> usize {
    static V: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    match *V.get_or_init(|| {
        std::env::var(PF_POOL_MB_ENV)
            .ok()
            .filter(|s| !s.trim().is_empty())
            .map(|s| pf_pool_mb_from_value(Some(&s)))
    }) {
        Some(mb) => mb << 20,
        None => crate::model::pf_budget::tuning().pool_mb << 20,
    }
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

// ---------------------------------------------------------------------------
// task 279: 長いプロンプトで「1 パスあたりのトークン数」とチャンクを上げる
// ---------------------------------------------------------------------------
//
// # なぜ要るか
//
// 層優先 prefill の MoE は **層ごとに 1 回** で、トークンは `DENNO_PF_LM_TILE`
// (既定 5,120) のタイルに割って回す。ミス expert の重みはタイルごとに VRAM へ
// 上げ直すので、14,018 トークンは 3 パスになり、上げる量はほぼパス数に比例する
// (lab t270b の台帳: 9 パスのとき 156.5 GB、stage_copy 6.06 s / ptrtab 8.48 s)。
// パス数を減らすにはタイルを大きくするしかなく、その作業域 (`h` / `act`) は
// expert の LRU から借りるしかない。task 265 のプール (長い prefill のあいだ
// だけ借りて返す) がその形を用意したので、ここでは
// **借りられる量からタイルとチャンクを選ぶ**。
//
// # 選び方 ([`choose_big`])
//
// 1. タイルはパス数が少ない順 (`nt_all`, `nt_all/2`, ... と `DENNO_PF_LM_TILE`
//    まで)。パス数が expert の H2D 量を決めるので、まずこれを最大化する。
// 2. そのタイルでチャンクを大きい順 (`DENNO_PF_POOL_BIG_CHUNK` から
//    `DENNO_PF_CHUNK` まで)。チャンクは mix / attn / ssm の反復回数を減らす。
// 3. レイアウトが上限 ([`PF_POOL_BIG_MB_ENV`]) に収まる最初の組を採る。
//
// 収まる組が無い / 確保に失敗したときは呼び出し側が**従来のチャンクとタイル**へ
// 戻す (リクエストは失敗させない)。

/// `DENNO_PF_POOL_BIG`: 長いプロンプトで大きいタイル / チャンクを使う
/// (task 279、**既定 on**。`0` / `off` / `false` / `no` で切る)。
///
/// # 実測 (lab t279c、sha 3796e6b25、depth8k、swift15-abl-IQ3_S)
///
/// base = 現在の int-auto 既定 (プール 2,688 MiB / タイル 5,120 / チャンク 1,536)。
/// on = `DENNO_PF_POOL_BIG=1` (タイル = 全トークン = 1 パス / チャンク 2,047)。
/// 各 COND は 2 回ずつ、先頭に捨て COND (`x0`)、末尾に base の繰り返し。
///
/// | 指標 | base | on | 差 |
/// | --- | ---: | ---: | ---: |
/// | 14,018 cold prefill | 16.76 / 16.80 s | 8.43 / 10.23 s | **-39 / -50%** |
/// | 3,826 cold prefill | 3.33 / 3.65 s | 3.35 / 3.62 s | 差なし |
/// | 短い decode | 19.4 ms/tok | 19.4 ms/tok | 0% |
/// | 14k cold 後の decode | 20.8 / 20.4 | 19.5 / 20.4 | 悪化なし |
/// | 14k turn2 の decode | 20.5 / 20.3 | 18.8 / 19.5 | 悪化なし |
/// | 上げた expert 重み | 90.9 GB | 28.2 GB | -69% |
///
/// 14k の 1 パス版で借りるのは 4,094 MiB (base は 2,498 MiB)。その確保が
/// 追い出した 783 本 (1,622 MiB) は prefill の終端で LRU へ戻すので、14k 後の
/// ヒット率は 0.956 (base) 対 0.959 (on) で悪化しない。
pub const PF_POOL_BIG_ENV: &str = "DENNO_PF_POOL_BIG";

/// [`PF_POOL_BIG_ENV`] の既定 (task 279 の lab t279c で採用)。
pub const DEFAULT_PF_POOL_BIG: bool = true;

/// `DENNO_PF_POOL_BIG_MB`: 大きい計画のときに借りてよい上限 (MiB、既定 4,864)。
///
/// task 265 の既定 (2,688 MiB) はタイル 5,120 のレイアウト (約 2,498 MiB) に
/// 合わせた値。パス数を 1 にすると `h` が 550 → 1,505 MiB、`act` が 200 → 547 MiB
/// 増えるので、そのぶんを上乗せした値。**実測で決める** (lab t279a/b)。
pub const PF_POOL_BIG_MB_ENV: &str = "DENNO_PF_POOL_BIG_MB";

/// [`PF_POOL_BIG_MB_ENV`] の既定 (MiB)。
///
/// 14k / タイル = 全トークン / チャンク 2,047 / TC のレイアウトが 4,341 MiB
/// (lab t279a の `[pf-pool-big]` は 4,094 MiB) なので、task 265 が見つけた
/// 「prefill 中のプール外の確保 (attention のミラー) が空き不足で 255 MiB の
/// チャンクを追い出す」ぶんの余裕を足す。
pub const DEFAULT_POOL_BIG_MB: usize = 4864;

/// `DENNO_PF_POOL_BIG_CHUNK`: 大きい計画で使うチャンクの上限 (トークン、既定 2,047)。
///
/// 既定を GDN prefill のデバイス連鎖の境界 (`GDN_CHUNK_MIN_TOKENS` = 2,048) の
/// 手前に置いてある。境界を越えると `DENNO_PF_CHAIN_BIG` (task 270) が要り、
/// lab t279a ではチャンク 2,047 (8.31 s) と 3,072 (8.46 s) で差が無かったので、
/// 未確認の形を既定にしない。上限 4,096 は t270a/b で連鎖が生きることを
/// 確認した最大 (明示指定 + `DENNO_PF_CHAIN_BIG=1` のときだけ到達する)。
pub const PF_POOL_BIG_CHUNK_ENV: &str = "DENNO_PF_POOL_BIG_CHUNK";

/// [`PF_POOL_BIG_CHUNK_ENV`] の既定 (トークン)。
pub const DEFAULT_BIG_CHUNK: usize = 2047;

/// `DENNO_PF_POOL_BIG_FREE=1`: 大きい計画の上限を**実空きだけ**から決める
/// (task 878、既定 off)。
///
/// # なぜ要るか
///
/// `DEFAULT_POOL_BIG_MB` (4,864 MiB) は 14k の 1 パスに合わせた値で、16k を
/// 超えると必ず 2 パス以上になる。2 パスは expert 重みの H2D / pread が
/// パス数倍になるので、cold prefill の壁時計がそのまま伸びる。
///
/// lab t878a (MODEL=flash-next-UD-Q4_K_XL、gen、実測) では:
///
/// | 文脈 | 固定上限 4,864 | 実空き - 余白 |
/// | --- | ---: | ---: |
/// | 10,816 tok (1 パス / 作業域 3,388 MiB) | 4,864 | 22,739 |
/// | 22,758 tok (2 パス / 作業域 4,200 MiB) | 4,864 | 22,739 |
///
/// 空きは 22.7 GiB も余っているのに固定上限が 2 パスを強制していた。
///
/// # 何をするか
///
/// on では上限を `min(PF_POOL_BIG_FREE_MB, 実空き - 余白)` にする。**追い出せる
/// expert を足さない**のが要点で、expert の LRU を削ってまで作業域を広げない
/// (確保できたとしても prefill 後の decode ヒット率を落とす。task 420 / 286c)。
/// 空いているカードでは 1 パス、混んでいるカードでは従来どおり複数パスになる。
///
/// # 実測 (lab t878b、sha 8f90119、MODEL=flash-next-UD-Q4_K_XL、gen)
///
/// **利得は確認できず、採らない** (`ADOPT: no`)。
///
/// | 文脈 | base (4,864 MiB 上限) | free-only (16,384 MiB 上限) |
/// | --- | --- | --- |
/// | 4,921 cold | 39.8 s (1 パス / 1,996 MiB) | 40.4 s (同じ plan) |
/// | 10,816 cold | 140.5 s (1 パス / 3,388 MiB) | 172.6 s (**同じ plan**) |
/// | 22,758 cold | 311.3 s (2 パス / 4,200 MiB) | 384.4 s (1 パス / 6,202 MiB) |
/// | 22,758 turn2 (再利用) | 1.53 s | 1.45 s (差なし) |
/// | md5 (6 本) | — | base と全一致 |
///
/// 10,816 の 2 本は**同じ plan** なのに 140.5 対 172.6 s (+23%) 動いた。
/// ラボのディスクは同一バイナリでも 2 倍動く (既知) ので、22,758 の
/// 311.3 対 384.4 s (+23%) も plan の差とは言えない。つまりこの A/B は
/// 判定不能で、1 パス化に利得がある証拠は無い。
///
/// 参考: 1 パスにすると prefill のピークは dev used 12,265 → 14,403 MiB、
/// 未計上の residual は 5,234 → 7,238 MiB に増えた (expert の追い出しは 0 本)。
/// 作業域は増えるが速度は上がらない、が読める範囲の事実。既定は off のまま残す。
pub const PF_POOL_BIG_FREE_ENV: &str = "DENNO_PF_POOL_BIG_FREE";

/// [`PF_POOL_BIG_FREE_ENV`] の既定 (off)。
pub const DEFAULT_PF_POOL_BIG_FREE: bool = false;

/// `DENNO_PF_POOL_BIG_FREE_MB`: free-only 計画の固定上限 (MiB、既定 16,384)。
///
/// 32,768 トークンの 1 パス (タイル = 全トークン) が収まる大きさ。
/// 実空きがこれを超えても、ここで止める (断片化の余裕を残す)。
pub const PF_POOL_BIG_FREE_MB_ENV: &str = "DENNO_PF_POOL_BIG_FREE_MB";

/// [`PF_POOL_BIG_FREE_MB_ENV`] の既定 (MiB)。
pub const DEFAULT_POOL_BIG_FREE_MB: usize = 16_384;

/// チャンクの上限 (これ以上は連鎖の形が未確認)。
pub const MAX_BIG_CHUNK: usize = 4096;

/// [`PF_POOL_BIG_ENV`] の解釈 (純関数)。既定 on、**明示的な偽値だけ off**。
///
/// task 265 の 3 本 (`DENNO_PF_POOL` / `DENNO_PF_LM_LAYER` / `DENNO_PF_TC_EXP`) と
/// 同じ規約にした (未設定・空・真値は on)。
pub fn pf_pool_big_from_value(v: Option<&str>) -> bool {
    !matches!(
        v.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("0" | "off" | "false" | "no")
    )
}

/// 大きい計画を使うか (プロセスで 1 回だけ env を読む)。
pub fn pf_pool_big_on() -> bool {
    #[cfg(test)]
    {
        let f = BIG_FORCE_ON.load(std::sync::atomic::Ordering::Relaxed);
        if f >= 0 {
            return f == 1;
        }
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| pf_pool_big_from_value(std::env::var(PF_POOL_BIG_ENV).ok().as_deref()))
}

/// テスト用: `pf_pool_big_on()` を env 無しで固定する (`None` で解除)。
#[cfg(test)]
pub(crate) fn set_big_force_on_for_test(v: Option<bool>) {
    BIG_FORCE_ON.store(
        match v {
            Some(true) => 1,
            Some(false) => 0,
            None => -1,
        },
        std::sync::atomic::Ordering::Relaxed,
    );
}

#[cfg(test)]
static BIG_FORCE_ON: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);

/// [`PF_POOL_BIG_MB_ENV`] の解釈 (純関数)。0 / 解析不能は既定。
pub fn pf_pool_big_mb_from_value(v: Option<&str>) -> usize {
    match v.map(str::trim) {
        Some(s) => s
            .parse::<usize>()
            .ok()
            .filter(|n| *n > 0)
            .unwrap_or(DEFAULT_POOL_BIG_MB),
        None => DEFAULT_POOL_BIG_MB,
    }
}

/// 大きい計画の借りる上限 (バイト)。env はプロセスで 1 回だけ読む。
pub fn pf_pool_big_bytes() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        pf_pool_big_mb_from_value(std::env::var(PF_POOL_BIG_MB_ENV).ok().as_deref()) << 20
    })
}

/// [`PF_POOL_BIG_FREE_ENV`] の解釈 (純関数)。**明示的な真値だけ on**。
pub fn pf_pool_big_free_from_value(v: Option<&str>) -> bool {
    matches!(
        v.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("1" | "on" | "true" | "yes")
    )
}

/// free-only 計画を使うか (プロセスで 1 回だけ env を読む)。
pub fn pf_pool_big_free_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        pf_pool_big_free_from_value(std::env::var(PF_POOL_BIG_FREE_ENV).ok().as_deref())
    })
}

/// [`PF_POOL_BIG_FREE_MB_ENV`] の解釈 (純関数)。0 / 解析不能は既定。
pub fn pf_pool_big_free_mb_from_value(v: Option<&str>) -> usize {
    match v.map(str::trim) {
        Some(s) => s
            .parse::<usize>()
            .ok()
            .filter(|n| *n > 0)
            .unwrap_or(DEFAULT_POOL_BIG_FREE_MB),
        None => DEFAULT_POOL_BIG_FREE_MB,
    }
}

/// free-only 計画の固定上限 (バイト)。env はプロセスで 1 回だけ読む。
pub fn pf_pool_big_free_bytes() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        pf_pool_big_free_mb_from_value(std::env::var(PF_POOL_BIG_FREE_MB_ENV).ok().as_deref()) << 20
    })
}

/// [`PF_POOL_BIG_CHUNK_ENV`] の解釈 (純関数)。0 / 解析不能は既定。
pub fn big_chunk_from_value(v: Option<&str>) -> usize {
    match v.map(str::trim) {
        Some(s) => s
            .parse::<usize>()
            .ok()
            .filter(|n| *n > 0)
            .map(|n| n.clamp(1, MAX_BIG_CHUNK))
            .unwrap_or(DEFAULT_BIG_CHUNK),
        None => DEFAULT_BIG_CHUNK,
    }
}

/// 大きい計画のチャンク上限 (トークン)。env はプロセスで 1 回だけ読む。
pub fn big_chunk_max() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| big_chunk_from_value(std::env::var(PF_POOL_BIG_CHUNK_ENV).ok().as_deref()))
}

/// `DENNO_PF_POOL_RESERVE_MB`: プールを借りたあとに残す空き (MiB、既定 1,024)。
///
/// task 290: 借りる上限を「実空き + 追い出せる expert - 余白」から決める。余白を
/// 残さないと、借りた直後の空きが 100-200 MiB になり、prefill の後半で出る
/// 数 MiB の確保 (attention のミラー・ssm のスクラッチ) が断片化で落ちる
/// (lab t286c の UD-Q4_K_XL 14k は空き 3 MiB で 3 MiB の確保に失敗した)。
pub const PF_POOL_RESERVE_MB_ENV: &str = "DENNO_PF_POOL_RESERVE_MB";

/// [`PF_POOL_RESERVE_MB_ENV`] の既定 (MiB)。
pub const DEFAULT_POOL_RESERVE_MB: usize = 1024;

/// `DENNO_PF_POOL_RESERVE_CPU_MB`: 層優先の **dev MoE が使えない**モデルで
/// プールのあとに残す空き (MiB、既定 3,072)。
///
/// expert の型が IQ カーネル非対応 (UD-Q4_K_XL など) だと、層優先の MoE は
/// ホスト経路へ落ち、ステージング / 作業域を**プールとは別に**確保する。
/// lab t286c の 14k では、この経路が借りた 4,779 MiB の上にさらに約 2.3 GB を
/// 使ってカードを埋めた。余白を多めに取って、1 パスに固執せずパス数を増やす。
pub const PF_POOL_RESERVE_CPU_MB_ENV: &str = "DENNO_PF_POOL_RESERVE_CPU_MB";

/// [`PF_POOL_RESERVE_CPU_MB_ENV`] の既定 (MiB)。
pub const DEFAULT_POOL_RESERVE_CPU_MB: usize = 3072;

/// [`PF_POOL_RESERVE_MB_ENV`] の解釈 (純関数)。0 / 解析不能は既定。
pub fn pf_pool_reserve_mb_from_value(v: Option<&str>) -> usize {
    match v.map(str::trim) {
        Some(s) => s
            .parse::<usize>()
            .ok()
            .filter(|n| *n > 0)
            .unwrap_or(DEFAULT_POOL_RESERVE_MB),
        None => DEFAULT_POOL_RESERVE_MB,
    }
}

/// [`PF_POOL_RESERVE_CPU_MB_ENV`] の解釈 (純関数)。0 / 解析不能は既定。
pub fn pf_pool_reserve_cpu_mb_from_value(v: Option<&str>) -> usize {
    match v.map(str::trim) {
        Some(s) => s
            .parse::<usize>()
            .ok()
            .filter(|n| *n > 0)
            .unwrap_or(DEFAULT_POOL_RESERVE_CPU_MB),
        None => DEFAULT_POOL_RESERVE_CPU_MB,
    }
}

/// dev MoE が使えるモデルで残す余白 (バイト)。env はプロセスで 1 回だけ読む。
pub fn pf_pool_reserve_bytes() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        pf_pool_reserve_mb_from_value(std::env::var(PF_POOL_RESERVE_MB_ENV).ok().as_deref()) << 20
    })
}

/// dev MoE が使えないモデルで残す余白 (バイト)。env はプロセスで 1 回だけ読む。
pub fn pf_pool_reserve_cpu_bytes() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        pf_pool_reserve_cpu_mb_from_value(std::env::var(PF_POOL_RESERVE_CPU_MB_ENV).ok().as_deref())
            << 20
    })
}

/// task 290: 借りてよい上限 = `min(cap, 実空き + 追い出せる expert - 余白)`。
///
/// `free == None` (実空きが読めない) のときは `cap` をそのまま返す (従来どおり)。
/// 追い出した expert は prefill の終端で LRU へ戻すので、`evictable` は
/// 「LRU に載っている量」まででよい (呼び出し側が `cap` で切る)。
pub fn pool_budget(cap: usize, free: Option<usize>, evictable: usize, reserve: usize) -> usize {
    match free {
        None => cap,
        Some(f) => cap.min(f.saturating_add(evictable).saturating_sub(reserve)),
    }
}

/// 大きい計画の候補 (純関数)。戻り値は `(chunk, tile)` の並び。
///
/// 並びは「MoE のパス数が少ない順 → チャンクが大きい順」。パス数は
/// `ceil(nt_all / tile)` で、expert 重みの H2D 量はほぼパス数に比例する。
/// チャンクは `nt_all / 2` 以下に抑える (1 枚になると層優先の意味が消え、
/// 呼び出し側がチャンク優先へ落ちる)。
pub fn big_candidates(
    nt_all: usize,
    base_chunk: usize,
    base_tile: usize,
    max_chunk: usize,
) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    if nt_all == 0 {
        return out;
    }
    let base_tile = base_tile.clamp(1, nt_all);
    let base_chunk = base_chunk.max(1);
    let hi_chunk = max_chunk.clamp(1, MAX_BIG_CHUNK).max(base_chunk);
    // タイル: パス数 1 から、`DENNO_PF_LM_TILE` (既定 5,120) まで。
    let mut tiles: Vec<usize> = Vec::new();
    let max_passes = nt_all.div_ceil(base_tile);
    for p in 1..=max_passes {
        let t = nt_all.div_ceil(p);
        if t < base_tile {
            break;
        }
        if tiles.last() != Some(&t) {
            tiles.push(t);
        }
    }
    if !tiles.contains(&base_tile) {
        tiles.push(base_tile);
    }
    // チャンク: 上限 → 2 倍 → 従来。どれも `hi_chunk` と nt_all/2 を超えない。
    let mut chunks: Vec<usize> = Vec::new();
    for c in [hi_chunk, base_chunk.saturating_mul(2).min(hi_chunk), base_chunk] {
        let c = c.min(nt_all / 2).max(base_chunk);
        if !chunks.contains(&c) {
            chunks.push(c);
        }
    }
    for t in tiles {
        for &c in &chunks {
            out.push((c, t));
        }
    }
    out
}

/// 上限 `cap` に収まる最良の `(chunk, tile, layout)` を選ぶ (純関数)。
///
/// 候補の順に試して**最初に収まった組**を返す (fail open = `None`)。
pub fn choose_big(n: &Need, cap: usize, max_chunk: usize) -> Option<(usize, usize, Layout)> {
    for (chunk, tile) in big_candidates(n.nt_all, n.chunk, n.tile, max_chunk) {
        let cand = Need { chunk, tile, ..*n };
        let Some(l) = layout(&cand) else {
            continue;
        };
        if l.total <= cap {
            return Some((chunk, tile, l));
        }
    }
    None
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
    /// task 788 (バグ 2 の前提): plen=64/65/67/127/128/129/512 の shared expert
    /// の作業域 (`ShX` / `ShGa` / `ShGu` / `ShSh`) が nt ぶん取れ、区画が
    /// 重ならず total に収まること。
    ///
    /// 65 / 67 / 127 / 129 は 16 / 32 / 64 の倍数でない (端数行)。寸法が
    /// nt に比例していること (切り捨て・切り上げが無いこと) を固定する。
    #[test]
    fn small_prefill_shared_regions_cover_nt() {
        for nt in [64usize, 65, 67, 127, 128, 129, 512] {
            let mut n = need();
            n.nt_all = nt;
            n.tile = nt;
            n.chunk = nt;
            let l = layout(&n).unwrap_or_else(|| panic!("nt={nt}: レイアウトが組めない"));
            assert_eq!(l.unit(R::ShX), nt * n.e * 4, "nt={nt}: ShX");
            assert_eq!(l.unit(R::ShGa), nt * n.sff * 4, "nt={nt}: ShGa");
            assert_eq!(l.unit(R::ShGu), nt * n.sff * 4, "nt={nt}: ShGu");
            assert_eq!(l.unit(R::ShSh), nt * n.e * 4, "nt={nt}: ShSh");
            let mut spans: Vec<(usize, usize)> = [R::ShX, R::ShGa, R::ShGu, R::ShSh]
                .iter()
                .map(|&r| {
                    let g = l.region(r);
                    (g.off, g.off.checked_add(g.bytes).expect("区画が溢れる"))
                })
                .collect();
            spans.sort();
            for w in spans.windows(2) {
                assert!(w[0].1 <= w[1].0, "nt={nt}: 区画が重なる {w:?}");
            }
            assert!(
                spans.last().expect("4 区画").1 <= l.total,
                "nt={nt}: 区画が total を超える"
            );
        }
    }

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

    // ----------------------------------------------------------------------
    // task 279: 大きいタイル / チャンクの選択
    // ----------------------------------------------------------------------

    /// ゲートは既定 on、明示的な偽値だけ off (task 265 の 3 本と同じ規約)。
    #[test]
    fn big_gate_is_on_unless_explicitly_off() {
        for v in [Some("0"), Some("off"), Some("false"), Some("no"), Some(" NO ")] {
            assert!(!pf_pool_big_from_value(v), "big({v:?}) が on");
        }
        for v in [None, Some(""), Some("1"), Some("on"), Some("true"), Some(" yes ")] {
            assert!(pf_pool_big_from_value(v), "big({v:?}) が off");
        }
        assert!(DEFAULT_PF_POOL_BIG, "既定は on (lab t279c で採用)");
        // 上限とチャンクの既定 (0 / 解析不能は既定)。
        assert_eq!(pf_pool_big_mb_from_value(None), DEFAULT_POOL_BIG_MB);
        assert_eq!(pf_pool_big_mb_from_value(Some("0")), DEFAULT_POOL_BIG_MB);
        assert_eq!(pf_pool_big_mb_from_value(Some("x")), DEFAULT_POOL_BIG_MB);
        assert_eq!(pf_pool_big_mb_from_value(Some("5200")), 5200);
        assert_eq!(big_chunk_from_value(None), DEFAULT_BIG_CHUNK);
        assert_eq!(big_chunk_from_value(Some("0")), DEFAULT_BIG_CHUNK);
        assert_eq!(big_chunk_from_value(Some("4096")), 4096);
        assert_eq!(big_chunk_from_value(None), 2047);
        // 上限 4,096 を超える指定は丸める (未確認の形を作らない)。
        assert_eq!(big_chunk_from_value(Some("8192")), MAX_BIG_CHUNK);
    }

    /// テスト用の強制スイッチ (env を触らずに分岐を固定する)。
    #[test]
    fn big_force_switch_for_tests() {
        set_big_force_on_for_test(Some(false));
        assert!(!pf_pool_big_on());
        set_big_force_on_for_test(Some(true));
        assert!(pf_pool_big_on());
        set_big_force_on_for_test(None);
        assert!(pf_pool_big_on(), "env 未設定の既定は on");
    }

    /// 候補は「パス数が少ない順 → チャンクが大きい順」で、従来のタイルも残る。
    #[test]
    fn big_candidates_reduce_passes_first() {
        let c = big_candidates(14018, 1536, 5120, 2047);
        // パス数を 1 にした組が先頭 (チャンクは大きい順)。
        assert_eq!(c[0], (2047, 14018), "先頭が 1 パスでない: {c:?}");
        assert_eq!(c[1], (1536, 14018));
        // そのあと 2 パス、最後に従来のタイル (3 パス)。
        let passes: Vec<usize> = c.iter().map(|&(_, t)| 14018usize.div_ceil(t)).collect();
        assert!(passes.windows(2).all(|w| w[0] <= w[1]), "パス数が単調でない: {passes:?}");
        assert!(c.contains(&(2047, 5120)), "従来タイルが候補に無い: {c:?}");
        assert_eq!(passes.last().copied(), Some(3));
        // 1 枚 (= 層優先の意味が消える) は作らない。
        assert!(c.iter().all(|&(ch, _)| ch * 2 <= 14018), "{c:?}");
        // タイルの下限は `DENNO_PF_LM_TILE`。
        assert!(c.iter().all(|&(_, t)| t >= 5120), "{c:?}");
        // 3,826 (タイルが既に全トークン) はタイルを上げない。
        let c2 = big_candidates(3826, 1536, 3826, 3072);
        assert!(c2.iter().all(|&(_, t)| t == 3826), "{c2:?}");
        assert_eq!(c2[0], (1913, 3826), "チャンクは nt/2 まで: {c2:?}");
    }

    /// 上限が増えないと 1 パスには上がらない (= 既定では従来と同じ形)。
    #[test]
    fn choose_big_only_when_the_cap_allows_it() {
        let n = need();
        let base_total = layout(&n).expect("レイアウト").total;
        // task 265 の既定 (2,688 MiB) ではタイルが上がらない = パス数は 3 のまま。
        let same = choose_big(&n, DEFAULT_POOL_MB << 20, DEFAULT_BIG_CHUNK).expect("収まる");
        assert_eq!(same.1, n.tile, "タイルが上がっている: {same:?}");
        assert_eq!(14018usize.div_ceil(same.1), 3);
        assert!(same.2.total <= (DEFAULT_POOL_MB << 20));
        // レイアウトの合計ちょうどでも従来の形が選べる (fail open の下限)。
        let tight = choose_big(&n, base_total, DEFAULT_BIG_CHUNK).expect("収まる");
        assert_eq!((tight.0, tight.1), (n.chunk, n.tile));
        assert_eq!(tight.2.total, base_total);
        // 4,864 MiB なら 1 パス + チャンク 2,047 (連鎖の境界の手前)。
        let big = choose_big(&n, DEFAULT_POOL_BIG_MB << 20, DEFAULT_BIG_CHUNK).expect("1 パス");
        assert_eq!((big.0, big.1), (2047, 14018));
        assert!(big.2.total <= (DEFAULT_POOL_BIG_MB << 20));
        // 余裕 (task 265 の実測で 255 MiB の追い出しが起きた) を残す。
        assert!(
            (DEFAULT_POOL_BIG_MB << 20) - big.2.total > (128 << 20),
            "余裕 {} B",
            (DEFAULT_POOL_BIG_MB << 20) - big.2.total
        );
        // 収まる組が無ければ None (呼び出し側は従来の形へ)。
        assert!(choose_big(&n, 1 << 20, DEFAULT_BIG_CHUNK).is_none());
        // 上限を 1,536 に落とすと、1 パスのままチャンクは従来のまま。
        let c1536 = choose_big(&n, DEFAULT_POOL_BIG_MB << 20, 1536).expect("1 パス");
        assert_eq!((c1536.0, c1536.1), (1536, 14018));
    }

    /// task 878: free-only のゲートは**明示的な真値だけ** on (既定 off)。
    #[test]
    fn big_free_gate_reads_only_explicit_true() {
        for v in [None, Some(""), Some("0"), Some("off"), Some("false"), Some("no")] {
            assert!(!pf_pool_big_free_from_value(v), "{v:?} は off");
        }
        for v in [Some("1"), Some("on"), Some("true"), Some("yes"), Some(" ON ")] {
            assert!(pf_pool_big_free_from_value(v), "{v:?} は on");
        }
        assert!(!DEFAULT_PF_POOL_BIG_FREE, "既定は off");
    }

    /// task 878: free-only の上限 env は 0 / 解析不能で既定。
    #[test]
    fn big_free_mb_default_on_zero_or_garbage() {
        assert_eq!(pf_pool_big_free_mb_from_value(None), DEFAULT_POOL_BIG_FREE_MB);
        assert_eq!(pf_pool_big_free_mb_from_value(Some("0")), DEFAULT_POOL_BIG_FREE_MB);
        assert_eq!(pf_pool_big_free_mb_from_value(Some("x")), DEFAULT_POOL_BIG_FREE_MB);
        assert_eq!(pf_pool_big_free_mb_from_value(Some("8192")), 8192);
        assert_eq!(DEFAULT_POOL_BIG_FREE_MB, 16_384);
    }

    /// task 878: evictable を 0 にすると「実空き - 余白」までしか借りない
    /// (= expert の LRU を削ってまで作業域を広げない)。
    #[test]
    fn free_only_budget_never_counts_experts() {
        let free = Some(22_739usize << 20);
        let evictable = 8_000usize << 20;
        let reserve = 3_072usize << 20;
        let with_experts = pool_budget(DEFAULT_POOL_BIG_MB << 20, free, evictable, reserve);
        let free_only = pool_budget(DEFAULT_POOL_BIG_FREE_MB << 20, free, 0, reserve);
        // 従来は固定上限 4,864 MiB で止まる。
        assert_eq!(with_experts, DEFAULT_POOL_BIG_MB << 20);
        // free-only は実空き - 余白 (19,667 MiB) と固定上限 16,384 MiB の小さい方。
        assert_eq!(free_only, DEFAULT_POOL_BIG_FREE_MB << 20);
        assert!(free_only > with_experts, "free-only のほうが広い: {free_only} vs {with_experts}");
        // 実空きが固定上限に満たなければ、実空き - 余白までしか借りない。
        assert_eq!(
            pool_budget(DEFAULT_POOL_BIG_FREE_MB << 20, Some(10_000 << 20), 0, reserve),
            (10_000 - 3_072) << 20,
            "実空き 10,000 - 余白 3,072 = 6,928 MiB"
        );
        // 実空きが読めないときは固定上限 (従来と同じ fail-open)。
        assert_eq!(
            pool_budget(DEFAULT_POOL_BIG_FREE_MB << 20, None, 0, reserve),
            DEFAULT_POOL_BIG_FREE_MB << 20
        );
        // 混んでいるカードでは free-only でも増えない (余白を割らない)。
        let tight = pool_budget(DEFAULT_POOL_BIG_FREE_MB << 20, Some(2_000 << 20), 0, reserve);
        assert_eq!(tight, 0, "空きが余白未満なら借りない");
    }

    fn need_nt(nt: usize) -> Need {
        Need {
            nt_all: nt,
            tile: 5_120,
            ..need()
        }
    }

    /// task 878: 16k / 32k のタイルと作業域の不変条件。
    ///
    /// - 既定の固定上限 (4,864 MiB) では 16k で 1 パスに届かない (= 2 パス以上)。
    ///   lab t878a の 22,758 tok が 2 パス / 作業域 4,200 MiB だったのと同じ形。
    /// - free-only の 16,384 MiB なら 16k も 32k も 1 パスに収まる。
    /// - 作業域は上限を超えない (超えるなら `choose_big` は別の組を返す)。
    #[test]
    fn big_plan_at_16k_and_32k() {
        let cap_def = DEFAULT_POOL_BIG_MB << 20;
        let cap_free = DEFAULT_POOL_BIG_FREE_MB << 20;
        // 16k: 既定ではタイルが全トークンに上がらない。
        let tight = choose_big(&need_nt(16_384), cap_def, DEFAULT_BIG_CHUNK).expect("収まる");
        assert!(tight.1 < 16_384, "既定で 1 パスに上がってはいけない: {}", tight.1);
        assert!(16_384usize.div_ceil(tight.1) >= 2);
        // 16k: free-only なら 1 パス。
        let k16 = choose_big(&need_nt(16_384), cap_free, DEFAULT_BIG_CHUNK).expect("1 パス");
        assert_eq!(k16.1, 16_384, "16k は 1 パス");
        assert!(k16.2.total <= cap_free);
        assert!(k16.2.total > cap_def, "既定の上限では収まらない大きさ: {}", k16.2.total);
        // 32k: free-only でも 1 パスで、上限 16 GiB に収まる。
        let k32 = choose_big(&need_nt(32_768), cap_free, DEFAULT_BIG_CHUNK).expect("1 パス");
        assert_eq!(k32.1, 32_768, "32k も 1 パス");
        assert!(k32.2.total <= cap_free, "32k の作業域 {} B", k32.2.total);
        // 作業域は文脈長に対して単調 (32k > 16k)、かつ 2 倍未満。
        assert!(k32.2.total > k16.2.total);
        assert!(k32.2.total < k16.2.total * 2);
        // 32,768 は max_ctx の上限側 (65,536 の半分)。既定の上限では 2 パス以上。
        let k32_def = choose_big(&need_nt(32_768), cap_def, DEFAULT_BIG_CHUNK).expect("収まる");
        assert!(
            32_768usize.div_ceil(k32_def.1) >= 2,
            "既定は 32k で複数パス: タイル {}",
            k32_def.1
        );
        assert!(k32_def.2.total <= cap_def);
    }

    /// 3,826 の大きい形 (チャンク 1,913) も既定のプールに収まる。
    #[test]
    fn big_plan_for_three_point_eight_k_fits() {
        let n = Need {
            nt_all: 3826,
            tile: 3826,
            ..need()
        };
        let (c, t, l) = choose_big(&n, DEFAULT_POOL_BIG_MB << 20, DEFAULT_BIG_CHUNK).expect("収まる");
        assert_eq!(t, 3826);
        assert_eq!(c, 1913);
        assert!(fits(&l, DEFAULT_POOL_MB << 20), "total {} B", l.total);
    }

    // ----------------------------------------------------------------------
    // task 290: 実空きと余白から借りる上限を決める
    // ----------------------------------------------------------------------

    /// 余白は 0 / 解析不能で既定、明示値はそのまま (両方の env)。
    #[test]
    fn reserve_defaults_and_parsing() {
        assert_eq!(pf_pool_reserve_mb_from_value(None), DEFAULT_POOL_RESERVE_MB);
        assert_eq!(
            pf_pool_reserve_mb_from_value(Some("0")),
            DEFAULT_POOL_RESERVE_MB
        );
        assert_eq!(
            pf_pool_reserve_mb_from_value(Some("x")),
            DEFAULT_POOL_RESERVE_MB
        );
        assert_eq!(pf_pool_reserve_mb_from_value(Some("2048")), 2048);
        assert_eq!(
            pf_pool_reserve_cpu_mb_from_value(None),
            DEFAULT_POOL_RESERVE_CPU_MB
        );
        assert_eq!(
            pf_pool_reserve_cpu_mb_from_value(Some("1")),
            1,
            "明示値は既定より小さくても使う (A/B 用)"
        );
        // CPU 経路の余白は dev MoE の余白より広い (実測の必要量が違う)。
        assert!(DEFAULT_POOL_RESERVE_CPU_MB > DEFAULT_POOL_RESERVE_MB);
    }

    /// 上限は `実空き + 追い出せる expert - 余白` を `cap` で切ったもの。
    #[test]
    fn pool_budget_leaves_the_reserve() {
        let cap = DEFAULT_POOL_BIG_MB << 20;
        let free = Some(2919usize << 20);
        let evictable = 4864usize << 20;
        // UD-Q4_K_XL の 14k (lab t286c) 相当: 余白 3,072 MiB なら 1 パスは入らない。
        let cpu = pool_budget(
            cap,
            free,
            evictable,
            DEFAULT_POOL_RESERVE_CPU_MB << 20,
        );
        assert_eq!(cpu, (2919 + 4864 - 3072) << 20);
        assert!(cpu < (4779usize << 20), "1 パスが入ってしまう: {cpu}");
        // dev MoE が使えるモデル (余白 1,024 MiB) なら上限は cap のまま。
        let dev = pool_budget(cap, free, evictable, DEFAULT_POOL_RESERVE_MB << 20);
        assert_eq!(dev, cap);
        // 実空きが読めないときは従来どおり cap。
        assert_eq!(pool_budget(cap, None, evictable, 0), cap);
        // 追い出せる expert が無ければ空きから余白を引いた分だけ。
        assert_eq!(pool_budget(cap, Some(4096 << 20), 0, 1024 << 20), 3072 << 20);
        // 空きが余白未満なら 0 (= 借りない)。
        assert_eq!(pool_budget(cap, Some(512 << 20), 0, 1024 << 20), 0);
    }

    /// 計画の判定: 1 パスが入らなければパス数を増やし、入らなければ借りない。
    #[test]
    fn plan_falls_back_to_more_passes_when_the_budget_is_small() {
        let n = need();
        // 候補はチャンクも下げるので、1 パスの**最小** (従来チャンク) で判定する。
        let one_min = layout(
            &Need {
                chunk: n.chunk,
                tile: n.nt_all,
                ..n
            },
        )
        .expect("1 パス")
        .total;
        let two_tile = n.nt_all.div_ceil(2);
        let two_min = layout(
            &Need {
                chunk: n.chunk,
                tile: two_tile,
                ..n
            },
        )
        .expect("2 パス")
        .total;
        assert!(two_min < one_min, "2 パスのほうが大きい: {two_min} vs {one_min}");
        // 1 パスが入る上限なら 1 パス (チャンクは収まる中で大きいほう)。
        let (_, t, l) = choose_big(&n, one_min, DEFAULT_BIG_CHUNK).expect("1 パスが入る");
        assert_eq!(t, n.nt_all);
        assert!(l.total <= one_min);
        // 1 パスが入らない上限なら 2 パスへ落ちる (借りないのではなくパスを増やす)。
        let (_, t2, l2) = choose_big(&n, one_min - 1, DEFAULT_BIG_CHUNK).expect("2 パスが入る");
        assert_eq!(t2, two_tile);
        assert_eq!(n.nt_all.div_ceil(t2), 2);
        assert!(l2.total <= one_min - 1);
        // 2 パスも入らない上限なら従来タイル (3 パス)。
        let (_, t3, _) = choose_big(&n, two_min - 1, DEFAULT_BIG_CHUNK).expect("従来タイル");
        assert_eq!(t3, n.tile);
        // 従来タイルすら入らなければ借りない (fail open)。
        let base_total = layout(&n).expect("従来").total;
        assert!(choose_big(&n, base_total - 1, DEFAULT_BIG_CHUNK).is_none());
    }

    /// task 420: UD-Q4_K_XL (e = 2560 / k = 10 / ff = 640 / ne = 512) の 14k。
    ///
    /// プール計画が g / u を取り (IQ 版の判定) 余白が CPU 経路の 3,072 MiB だと、
    /// 実空き 2,913 MiB では **1 パスに届かない** (lab t420a の base と同じ 7,009)。
    /// g / u を取らず (K 版の TC に乗る) 余白が dev 経路の 1,024 MiB なら 1 パス。
    #[test]
    fn ud_k_model_one_pass_needs_the_k_tc_plan() {
        let base = Need {
            nt_all: 14018,
            tile: 5120,
            k: 10,
            e: 2560,
            ff: 640,
            sff: 1024,
            hc: 4,
            lr: 256,
            chunk: 1536,
            ne: 512,
            tc: true,
            stage_bytes: 512 << 20,
            stage_slots: 3,
        };
        let (free, cap) = (2913usize << 20, DEFAULT_POOL_BIG_MB << 20);
        // 旧: g / u を取り、余白 3,072 MiB (CPU 経路と誤判定)。
        let old_budget = pool_budget(
            cap,
            Some(free),
            cap,
            DEFAULT_POOL_RESERVE_CPU_MB << 20,
        );
        let (_, t_old, _) =
            choose_big(&Need { tc: false, ..base }, old_budget, DEFAULT_BIG_CHUNK)
                .expect("2 パスは収まる");
        assert_eq!(t_old, 7009, "旧計画は 2 パス (lab t420a の base と同じ)");
        // 新: g / u を取らず、余白 1,024 MiB (dev 経路)。
        let new_budget = pool_budget(cap, Some(free), cap, DEFAULT_POOL_RESERVE_MB << 20);
        let (_, t_new, l_new) = choose_big(&base, new_budget, DEFAULT_BIG_CHUNK)
            .expect("1 パスが収まる");
        assert_eq!(t_new, base.nt_all, "新計画は 1 パス");
        assert!(l_new.total < (4779usize << 20), "作業域も旧の 4,779 MiB より小さい");
    }
}
