//! task 260: expert 重みの**ホスト RAM tier** (VRAM tier と mmap の間)。
//!
//! # なぜ要るか
//!
//! production の 2026-11-12 の観測: swift15-abl IQ3_S の expert 52.8 GB は
//! read-only mmap で、`DENNO_START_PREFETCH=1` が起動時に page cache へ読む
//! (52,815 MB / 1.3-1.8 s)。ところが page cache は**誰の物でもない**ので、
//! 別モデルの大きな読みで追い出される。実際、別モデルを回して戻すと起動の
//! prefetch が disk から読み直し (load 19.9 s、平常 2.7 s)、最初のリクエストが
//! 149 ms/tok (平常 26) になった。UD-Q4_K_XL は 111 GB で RAM に収まらず、
//! 初回 797 ms/tok / 以降 57-229 ms/tok が disk 由来だった。
//!
//! # 何をするか
//!
//! `DENNO_EXPERT_RAM_MB` (MiB) の予算で、expert の生バイトを**所有する**
//! ホストバッファへ常駐させる。常駐先は CUDA の `cuMemHostAlloc` (pinned。
//! 取れなければヒープ)。page cache と違って追い出されず、H2D の転送元にも
//! そのまま使える (pinned 52.6 GB/s / pageable 19.6 GB/s)。
//!
//! 引く順番は **VRAM tier → RAM tier → mmap**。RAM tier は
//! [`crate::model::expert_cache::ExpertCacheSet`] の固定長ストア版なので、
//! LRU・鍵 ((層, expert))・返すバイト列は従来の pread キャッシュと同じ。
//!
//! # ゲート
//!
//! - `DENNO_EXPERT_RAM_MB`: **未設定 / `0` / `off` は無効** (従来と 1 バイトも
//!   変わらない)。`auto` は自動 = `MemAvailable` − **モデルの非 expert バイト** −
//!   余裕 (`DENNO_EXPERT_RAM_MARGIN_MB`, 既定 8192 MiB) を、モデルの expert 総
//!   バイト数で頭打ちにする。数値は MiB。
//!
//!   **非 expert バイトを引く理由** (task 268): expert 以外 (dense 重み / PLE の
//!   表 / KV) は page cache か mmap から読むので、tier がその場所を奪うと
//!   decode が壊れる。t260c の実測では 59 GiB の機械で IQ3_S の expert 47 GB を
//!   所有すると MemAvailable が 8-10 GB まで落ち、warm decode が 19.4 →
//!   46.4 ms/tok になった (tier が page cache を追い出した)。UD-Q4_K_XL は
//!   104 GB 中 77 GB が expert、27 GB が非 expert なので、auto は
//!   `MemAvailable − 27 GB − 8 GiB` ≒ 22 GB になる。
//!
//!   **既定 off にしてある理由**: 上と同じ (大きさを絞る規則を入れるまでは
//!   off。task 268 で規則を入れたが、既定 off は変えない =
//!   `DENNO_EXPERT_RAM_MB` を明示したときだけ動く)。
//! - `DENNO_EXPERT_RAM_PIN`: `0` で pinned を諦めてヒープ (A/B 用。既定 on)。
//! - `DENNO_EXPERT_RAM_FILL`: `0` で起動時の充填と追従 (再調整) をしない
//!   (使用時に埋まる。既定 on)。task 263 から充填は**起動プラン**が行う
//!   (VRAM tier の残りを回数の多い順に)。tier の作成もプランの中。
//! - `DENNO_EXPERT_RAM_REBALANCE_SECS`: 走行中に RAM tier を回数の上位へ
//!   寄せ直す間隔 (秒、既定 30、`0` で止める)。低優先度 (decode 中は譲る)。
//! - `DENNO_EXPERT_RAM_DBG`: `1` で 1 リクエストごとに 1 行 stderr。

use std::sync::{Arc, RwLock};

use crate::model::expert_cache::ExpertCacheSet;

/// 予算 (MiB) の env。`0` で無効、未設定 / `auto` で自動。
pub const RAM_MB_ENV: &str = "DENNO_EXPERT_RAM_MB";
/// pinned を試すか (`0` でヒープ)。既定 on。
pub const PIN_ENV: &str = "DENNO_EXPERT_RAM_PIN";
/// 起動時に予算まで充填するか (`0` で off)。既定 on。
pub const FILL_ENV: &str = "DENNO_EXPERT_RAM_FILL";
/// 自動予算で `MemAvailable` に残す余裕 (MiB)。既定 8192。
pub const MARGIN_MB_ENV: &str = "DENNO_EXPERT_RAM_MARGIN_MB";
/// 充填のスレッド数。既定 12 (`start-warm` と同じ)。
pub const THREADS_ENV: &str = "DENNO_EXPERT_RAM_THREADS";
/// 1 リクエストごとに 1 行 stderr を出す。
pub const DBG_ENV: &str = "DENNO_EXPERT_RAM_DBG";

/// 自動予算で `MemAvailable` に残す余裕の既定 (8 GiB)。
///
/// システム / Hermes の dev コンテナ / dense 重みの page cache のぶん。
/// task 文の指定値。
pub const DEFAULT_MARGIN_MB: u64 = 8192;

/// `DENNO_EXPERT_RAM_MB` の解釈。
///
/// - `None` (未設定) / 空 / `0` / `off` / `false` → `Some(0)` = 無効 (既定)
/// - `auto` → `None` = 自動
/// - 数値 → `Some(MiB)`
/// - その他 → `Some(0)` (無効。誤植で勝手にメモリを確保しない)
pub fn mb_from_value(v: Option<&str>) -> Option<u64> {
    let s = match v {
        Some(s) => s.trim(),
        None => return Some(0),
    };
    if s.eq_ignore_ascii_case("auto") {
        return None;
    }
    if s.is_empty()
        || matches!(
            s.to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        )
    {
        return Some(0);
    }
    s.parse::<u64>().ok().map(Some).unwrap_or(Some(0))
}

/// ゲートが有効か (0 でも自動でもない)。
pub fn enabled_from_value(v: Option<&str>) -> bool {
    mb_from_value(v) != Some(0)
}

/// 予算 (バイト)。`avail` は `MemAvailable`、`expert_bytes` はモデルの
/// expert 総バイト数 (上限)。
///
/// 自動は `avail - margin` を expert 総量で頭打ちにする。数値指定でも
/// `avail - margin` を超えない (指定した予算で機械を押し潰さない)。
///
/// task 268: モデルの**非 expert バイト**を渡す版が [`budget_bytes_model`]。
/// こちらは `non_expert = 0` の後方互換 (テストと、モデル総量が分からない
/// 呼び出し用)。
pub fn budget_bytes(v: Option<&str>, avail: Option<u64>, expert_bytes: u64, margin: u64) -> u64 {
    budget_bytes_model(v, avail, expert_bytes, 0, margin)
}

/// task 268: `non_expert_bytes` (モデル総バイト − expert 総バイト) を
/// `MemAvailable` から引く予算。
///
/// - 自動: `avail - non_expert - margin` (負なら 0) を expert 総量で頭打ち。
/// - 数値: `min(指定, avail - non_expert - margin, expert_bytes)`。
///   `MemAvailable` が読めないときは指定値をそのまま使う (従来どおり)。
///
/// 規則の根拠は module doc の t260c (47 GB の tier が page cache を追い出して
/// warm decode を 2.4 倍にした)。expert 以外が page cache に残る大きさに
/// 抑える、が規則。
pub fn budget_bytes_model(
    v: Option<&str>,
    avail: Option<u64>,
    expert_bytes: u64,
    non_expert_bytes: u64,
    margin: u64,
) -> u64 {
    let keep = non_expert_bytes.saturating_add(margin);
    let room = match avail {
        Some(a) => a.saturating_sub(keep),
        None => 0,
    };
    match mb_from_value(v) {
        Some(0) => 0,
        Some(mb) => {
            let want = mb << 20;
            match avail {
                Some(_) => want.min(expert_bytes).min(room),
                // MemAvailable が読めないときは明示予算をそのまま使う
                None => want.min(expert_bytes),
            }
        }
        None => room.min(expert_bytes),
    }
}

/// `DENNO_EXPERT_RAM_MARGIN_MB` の解釈 (未設定・非数は既定)。
pub fn margin_mb_from_value(v: Option<&str>) -> u64 {
    match v.map(str::trim).and_then(|s| s.parse::<u64>().ok()) {
        Some(mb) if mb > 0 => mb,
        _ => DEFAULT_MARGIN_MB,
    }
}

/// `DENNO_EXPERT_RAM_PIN` の解釈 (**既定 on**。明示的な偽値だけ off)。
pub fn pin_from_value(v: Option<&str>) -> bool {
    !matches!(
        v.map(str::trim).map(|s| s.to_ascii_lowercase()),
        Some(ref s) if matches!(s.as_str(), "0" | "off" | "false" | "no")
    )
}

/// `DENNO_EXPERT_RAM_FILL` の解釈 (**既定 on**)。
pub fn fill_from_value(v: Option<&str>) -> bool {
    pin_from_value(v)
}

fn env(k: &str) -> Option<String> {
    std::env::var(k).ok()
}

/// ゲートが有効か (プロセスで 1 回だけ env を読む)。
///
/// **テストでは既定 off** (env 未設定のとき)。合成モデルを組むテストが毎回
/// 数十 GB の予算を計算したり、グローバルな tier がテスト間で混ざったりしない
/// ようにするため。env を明示したときはテストでもその値に従う。
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        if cfg!(test) && env(RAM_MB_ENV).is_none() {
            return false;
        }
        enabled_from_value(env(RAM_MB_ENV).as_deref())
    })
}

pub fn want_pinned() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| pin_from_value(env(PIN_ENV).as_deref()))
}

pub fn fill_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| fill_from_value(env(FILL_ENV).as_deref()))
}

pub fn margin_bytes() -> u64 {
    static V: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *V.get_or_init(|| margin_mb_from_value(env(MARGIN_MB_ENV).as_deref()).saturating_mul(1 << 20))
}

pub fn threads() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        env(THREADS_ENV)
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(12)
            .min(32)
    })
}

pub fn dbg_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        matches!(
            env(DBG_ENV).as_deref().map(str::trim),
            Some("1") | Some("true") | Some("on")
        )
    })
}

/// 明示予算 (診断用。自動のときは `None`)。
pub fn budget_mb_env() -> Option<u64> {
    mb_from_value(env(RAM_MB_ENV).as_deref())
}

/// task 268: env と `MemAvailable` から予算 (バイト) を決める。
///
/// モデルの非 expert バイトを渡す版だけを残す (task 263 までの
/// `budget_bytes_env(expert_bytes)` は、非 expert を引かない古い規則だった)。
pub fn budget_bytes_env_model(expert_bytes: u64, non_expert_bytes: u64) -> u64 {
    budget_bytes_model(
        env(RAM_MB_ENV).as_deref(),
        mem_available(),
        expert_bytes,
        non_expert_bytes,
        margin_bytes(),
    )
}

// ---------------------------------------------------------------------------
// プロセスで 1 本の tier
// ---------------------------------------------------------------------------
//
// VRAM tier ([`crate::model::expert_vram`]) と同じく、モデルではなくプロセスに
// 1 本置く。forward の深い所 (expert のミス計算 / H2D の転送元 / prefill の
// ストリーム) から引くので、引数で配ると経路が増えすぎる。
//
// モデルを差し替えるときは [`install`] が古い Arc を落とす (pinned も解放される)。

static TIER: RwLock<Option<Arc<ExpertCacheSet>>> = RwLock::new(None);

/// tier を差し替える (`None` で外す)。古い tier はここで落ちる。
pub fn install(set: Option<Arc<ExpertCacheSet>>) {
    let mut g = TIER.write().unwrap_or_else(|e| e.into_inner());
    *g = set;
}

/// いまの tier。参照カウントを 1 つ進めて返す (ロックはすぐ離す)。
pub fn get() -> Option<Arc<ExpertCacheSet>> {
    TIER.read()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map(Arc::clone)
}

/// tier が有効か (ロックを取らない軽い判定は `enabled()` のほう)。
pub fn active() -> bool {
    TIER.read()
        .unwrap_or_else(|e| e.into_inner())
        .is_some()
}

// ---------------------------------------------------------------------------
// 統計 (1 リクエストぶん)
// ---------------------------------------------------------------------------
//
// 数字は tier 自身の LRU 統計から読む (別に数えると、経路ごとに数え漏れが出る)。

/// 1 リクエストぶんの 1 行。tier が無効なら `None`。
pub fn stats_line() -> Option<String> {
    let set = get()?;
    let (mut hits, mut misses, mut read) = (0u64, 0u64, 0u64);
    for m in crate::model::expert_cache::ExpertMatrix::ALL {
        let s = set.stats(m);
        hits += s.hits;
        misses += s.misses;
        read += s.bytes_read;
    }
    let n = hits + misses;
    let used = set.used_experts();
    let cap = set.slots(crate::model::expert_cache::ExpertMatrix::Gate);
    let pinned = set.pinned_bytes() >> 20;
    let reserved = set.reserved_bytes() >> 20;
    // task 268: VRAM ミスの転送元の内訳 (RAM tier から / mmap から)。
    let (mram, mmmap) = miss_stats();
    Some(format!(
        "[exp-ram] lookups={n} hits={hits} miss={misses} hit_rate={:.3} resident={used}/{cap} pinned={pinned}MiB reserved={reserved}MiB reads_mib={} miss_ram={mram} miss_mmap={mmmap}",
        if n > 0 { hits as f64 / n as f64 } else { 0.0 },
        read >> 20
    ))
}

/// 積算を 0 に戻す (リクエスト終端)。
pub fn stats_reset() {
    if let Some(set) = get() {
        for m in crate::model::expert_cache::ExpertMatrix::ALL {
            set.reset_stats_for(m);
        }
    }
    miss_reset();
}

// ---------------------------------------------------------------------------
// task 268: VRAM ミスの転送元 (RAM tier / mmap) の内訳
// ---------------------------------------------------------------------------
//
// VRAM に載っていない expert を上げるとき、`acquire_expert` は
// **RAM tier → mmap** の順に転送元を選ぶ。どちらを使ったかを数える。
// 「tier が効いている」を緑のテストではなく数字で示すための計器。

static MISS_FROM_RAM: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static MISS_FROM_MMAP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// VRAM ミス 1 本の転送元を数える (`from_ram = true` は RAM tier のヒット)。
pub fn note_miss_source(from_ram: bool) {
    use std::sync::atomic::Ordering;
    if from_ram {
        MISS_FROM_RAM.fetch_add(1, Ordering::Relaxed);
    } else {
        MISS_FROM_MMAP.fetch_add(1, Ordering::Relaxed);
    }
}

/// `(RAM tier から上げた本数, mmap から上げた本数)`。
pub fn miss_stats() -> (u64, u64) {
    use std::sync::atomic::Ordering;
    (
        MISS_FROM_RAM.load(Ordering::Relaxed),
        MISS_FROM_MMAP.load(Ordering::Relaxed),
    )
}

/// 内訳を 0 に戻す。
pub fn miss_reset() {
    use std::sync::atomic::Ordering;
    MISS_FROM_RAM.store(0, Ordering::Relaxed);
    MISS_FROM_MMAP.store(0, Ordering::Relaxed);
}

/// task 263: 常駐している expert の鍵 (エンドポイントと再調整用)。
pub fn keys() -> Vec<crate::model::expert_cache::ExpertKey> {
    get().map(|s| s.keys()).unwrap_or_default()
}

/// task 263: 常駐している実バイト数。
pub fn used_bytes() -> u64 {
    get().map(|s| s.used_bytes()).unwrap_or(0)
}

/// task 263: `(常駐本数, スロット数, pinned バイト, 使用バイト)`。tier が無ければ `None`。
pub fn status() -> Option<(usize, usize, u64, u64)> {
    let set = get()?;
    Some((
        set.used_experts(),
        set.slots(crate::model::expert_cache::ExpertMatrix::Gate),
        set.pinned_bytes(),
        set.used_bytes(),
    ))
}

/// task 263: 再調整の間隔 (秒)。
///
/// `DENNO_EXPERT_RAM_REBALANCE_SECS` は `0` で再調整を止める (既定 30 秒)。
/// 起動プランのあと、この間隔で「いまの回数の上位」に RAM tier を寄せる。
pub const REBALANCE_SECS_ENV: &str = "DENNO_EXPERT_RAM_REBALANCE_SECS";

/// [`REBALANCE_SECS_ENV`] の既定 (秒)。
pub const DEFAULT_REBALANCE_SECS: u64 = 30;

/// 再調整の間隔 (秒)。`0` = しない。
pub fn rebalance_secs() -> u64 {
    static V: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *V.get_or_init(|| match env(REBALANCE_SECS_ENV) {
        Some(s) => s.trim().parse::<u64>().unwrap_or(DEFAULT_REBALANCE_SECS),
        None => DEFAULT_REBALANCE_SECS,
    })
}

/// task 907: 再調整で「落とす鍵」と「載せる鍵」を決める (純関数)。
///
/// - `plan_asc` は**出現回数の昇順** (少ない順。`exp_stats::preload_order` の順)。
/// - `cap` は RAM tier に載せられる本数。
/// - `vram` は VRAM に常駐している鍵 (同じ expert を 2 か所に持たない)。
/// - `have` は RAM tier に常駐している鍵。
///
/// 戻り値は `(落とす鍵, 載せる鍵)`。どちらも決定的で、載せる順は `plan_asc` の
/// 昇順 (= 一番熱いのが最後に触られて、次に落とされにくい)。
///
/// # thrash を避ける形
///
/// 落とす集合は「いま常駐していて、欲しい集合に入らないもの」だけ。欲しい集合は
/// 回数の上位 `cap` 本から VRAM の同居を除いたもの。drop を先に全部済ませてから
/// 昇順に埋めるので、埋めている途中で欲しい鍵が LRU に追い出されない
/// (埋める本数 ≤ 落とした本数 + 空きスロット)。
///
/// 適用後にもう一度呼ぶと `(空, 空)` を返す (固定点 = 30 秒ごとに同じ鍵を
/// 落として載せ直す thrash が起きない)。
pub fn rebalance_plan(
    plan_asc: &[(u16, u16)],
    cap: usize,
    vram: &std::collections::HashSet<(u16, u16)>,
    have: &[(u16, u16)],
) -> (Vec<(u16, u16)>, Vec<(u16, u16)>) {
    if cap == 0 || plan_asc.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let want: std::collections::HashSet<(u16, u16)> = plan_asc
        .iter()
        .rev()
        .filter(|k| !vram.contains(k))
        .take(cap)
        .copied()
        .collect();
    let have_set: std::collections::HashSet<(u16, u16)> = have.iter().copied().collect();
    let drop: Vec<(u16, u16)> = have
        .iter()
        .copied()
        .filter(|k| !want.contains(k))
        .collect();
    let fill: Vec<(u16, u16)> = plan_asc
        .iter()
        .copied()
        .filter(|k| want.contains(k) && !have_set.contains(k))
        .collect();
    (drop, fill)
}

/// `MemAvailable` (バイト)。
pub fn mem_available() -> Option<u64> {
    crate::model::expert_cache::mem_available()
}

// ---------------------------------------------------------------------------
// 診断 (task 260 step 1: ホスト側のバイトがどこから来るか)
// ---------------------------------------------------------------------------
//
// 測るもの:
// - `/proc/self/stat` の minflt / majflt の**リクエスト前後の差**。
//   major fault = disk から読んだ回数 (page cache に無かった)。
// - expert テンソルの mmap の常駐量 (mincore)。「page cache に何割載っているか」。
// - MemAvailable。
//
// ゲート: `DENNO_EXPERT_RAM_PROBE=1`、または `DENNO_Q4_PROFILE` が立っている
// (`mincore` は高いので `DENNO_EXPERT_RAM_PROBE=1` のときだけ走る)。
// `DENNO_EXPERT_RAM_DROP=1` は**リクエストの先頭で expert の page cache を落とす**
// (冷えた機械の再現。dense 重みは落とさない)。

/// `DENNO_EXPERT_RAM_PROBE`: 1 で mincore まで測る。
pub const PROBE_ENV: &str = "DENNO_EXPERT_RAM_PROBE";
/// `DENNO_EXPERT_RAM_DROP`: 1 でリクエストごとに expert の page cache を落とす。
pub const DROP_ENV: &str = "DENNO_EXPERT_RAM_DROP";
/// task 268: `1` で**起動プランの直前**に expert の page cache を落とす
/// (本番の「モデルを切り替えた直後」の再現。テスト用のゲート)。
///
/// `DENNO_EXPERT_RAM_DROP` はリクエストの先頭で落とすので初回に間に合わない
/// (lab t263b: 1 本目は温かいまま 19.3 ms/tok、2 本目が 60 ms/tok になった)。
/// こちらは起動プラン (= 事前ロード) が 1 バイトも読む前に落とすので、
/// 「冷えた機械で最初のリクエスト」をそのまま測れる。
pub const DROP_ON_START_ENV: &str = "DENNO_EXPERT_DROP_CACHE_ON_START";
/// `DENNO_EXPERT_RAM_ADAPT`: 1 で使用時に埋める (LRU)。**既定 off**。
///
/// off のときの tier は「起動時に充填した固定の集合」で、decode のミスでは
/// ファイルを読み直さない。lab t260a の実測: decode のミスで pread して
/// 埋める経路は、ヒット率が上がる前に 29% 遅くなった (page cache と同じ
/// バイト列をもう一度読むだけで、page cache を追い出す)。
pub const ADAPT_ENV: &str = "DENNO_EXPERT_RAM_ADAPT";

/// expert テンソルのファイル範囲 (part, off, len)。落とすときに使う。
static RANGES: RwLock<Vec<(usize, u64, u64)>> = RwLock::new(Vec::new());
/// GGUF のパートのパス。
static PARTS: RwLock<Vec<std::path::PathBuf>> = RwLock::new(Vec::new());
/// expert テンソルの mmap 範囲 (ptr, len)。常駐量を測るのに使う。
static MMAP: RwLock<Vec<(usize, usize)>> = RwLock::new(Vec::new());
/// 前回の (minflt, majflt)。
static LAST_FAULTS: std::sync::Mutex<(u64, u64)> = std::sync::Mutex::new((0, 0));
/// 前回の `/proc/self/io` の (rchar, read_bytes)。task 514。
static LAST_IO: std::sync::Mutex<(u64, u64)> = std::sync::Mutex::new((0, 0));
/// モデル全パートの mmap (ptr, len)。「モデル全体で page cache に何 MiB」を測る。task 514。
static PARTMAP: RwLock<Vec<(usize, usize)>> = RwLock::new(Vec::new());

/// task 514: `DENNO_PF_FADV_END=1` で**リクエスト終端**に expert 範囲の page cache を
/// `posix_fadvise(DONTNEED)` で落とす。**既定 off**。
///
/// なぜ終端か: このコンテナ (kernel 7.0.0) で実測した挙動 —
/// (a) mmap してフォールトしたページは `fadvise(DONTNEED)` では落ちない
/// (mapcount > 0 のページは invalidate されない。`/tmp/pc_test2.py` の実測:
/// 64 MiB をフォールト → drop 後も 65,536 KiB のまま)、(b) pread で page cache に
/// 入っただけのページは落ちる (同 0 KiB)。
/// つまりこの drop は **pread の読み残しだけ**を落とし、mmap で使ったページは
/// 残す。目的は「expert の pread (リクエストあたり 43-53 GB) が dense / PLE の
/// page cache を追い出し続ける」のを止めること。
pub const FADV_END_ENV: &str = "DENNO_PF_FADV_END";

/// [`FADV_END_ENV`] が on か。
pub fn fadv_end_enabled() -> bool {
    flag(FADV_END_ENV)
}

fn flag(k: &str) -> bool {
    matches!(
        env(k).as_deref().map(str::trim),
        Some("1") | Some("true") | Some("on")
    )
}

pub fn probe_enabled() -> bool {
    flag(PROBE_ENV) || env("DENNO_Q4_PROFILE").is_some()
}

pub fn drop_enabled() -> bool {
    flag(DROP_ENV)
}

/// task 268: 起動プランの直前の page cache drop が有効か。
pub fn drop_on_start_enabled() -> bool {
    flag(DROP_ON_START_ENV)
}

/// 使用時に埋めるか (`DENNO_EXPERT_RAM_ADAPT=1`。既定 off)。
pub fn adapt_enabled() -> bool {
    flag(ADAPT_ENV)
}

/// モデルの expert 範囲を覚える (リクエストをまたいで使う)。
pub fn install_ranges(
    parts: Vec<std::path::PathBuf>,
    ranges: Vec<(usize, u64, u64)>,
    mmap: Vec<(usize, usize)>,
) {
    *PARTS.write().unwrap_or_else(|e| e.into_inner()) = parts;
    *RANGES.write().unwrap_or_else(|e| e.into_inner()) = ranges;
    *MMAP.write().unwrap_or_else(|e| e.into_inner()) = mmap;
}

/// task 514: `/proc/self/io` の (rchar, read_bytes)。
///
/// `read_bytes` は**ディスクから実際に読んだ**バイト (page cache で済んだぶんは
/// 入らない)。`rchar` は read 系 (pread / readahead を含む) の要求バイト。
/// この 2 つの差が「page cache が効いたぶん」になる。
fn proc_io() -> Option<(u64, u64)> {
    let s = std::fs::read_to_string("/proc/self/io").ok()?;
    let mut rchar = None;
    let mut read_bytes = None;
    for line in s.lines() {
        if let Some(v) = line.strip_prefix("rchar: ") {
            rchar = v.trim().parse::<u64>().ok();
        } else if let Some(v) = line.strip_prefix("read_bytes: ") {
            read_bytes = v.trim().parse::<u64>().ok();
        }
    }
    Some((rchar?, read_bytes?))
}

/// task 514: モデル全パートの mmap 範囲を覚える (page cache の常駐量を測る)。
pub fn install_part_map(v: Vec<(usize, usize)>) {
    *PARTMAP.write().unwrap_or_else(|e| e.into_inner()) = v;
}

/// expert テンソルの mmap の常駐バイト (mincore の合計)。
fn expert_resident() -> u64 {
    let mm = MMAP.read().unwrap_or_else(|e| e.into_inner());
    mm.iter().map(|&(p, l)| resident_bytes(p, l)).sum()
}

/// モデル全体の mmap の常駐バイト (mincore の合計)。dense / PLE も含む。
fn file_resident() -> u64 {
    let pm = PARTMAP.read().unwrap_or_else(|e| e.into_inner());
    pm.iter().map(|&(p, l)| resident_bytes(p, l)).sum()
}

/// task 514: リクエスト終端で expert 範囲の page cache を落とす。
///
/// ゲート ([`FADV_END_ENV`]) off なら `None`。落とせるのは **mapcount 0 の
/// ページだけ** なので、mmap で使ったページは残り、pread の読み残しだけが
/// 消える (モジュール冒頭の実測を参照)。
pub fn drop_expert_pages_end(tag: &str) -> Option<String> {
    if !fadv_end_enabled() {
        return None;
    }
    let parts = PARTS.read().unwrap_or_else(|e| e.into_inner()).clone();
    let ranges = RANGES.read().unwrap_or_else(|e| e.into_inner()).clone();
    if parts.is_empty() || ranges.is_empty() {
        return Some(format!(
            "[exp-pc-end] {tag} 範囲が未登録なので落とせない ({FADV_END_ENV})"
        ));
    }
    let probe = flag(PROBE_ENV);
    let (before, before_file) = if probe {
        (expert_resident(), file_resident())
    } else {
        (0, 0)
    };
    let t0 = std::time::Instant::now();
    let line = drop_ranges(&parts, &ranges);
    let s = t0.elapsed().as_secs_f64();
    let mut out = format!("[exp-pc-end] {tag} s={s:.2} {line}");
    if probe {
        out.push_str(&format!(
            " expert_resident={}MiB->{}MiB file_resident={}MiB->{}MiB",
            before >> 20,
            expert_resident() >> 20,
            before_file >> 20,
            file_resident() >> 20
        ));
    }
    Some(out)
}

/// `/proc/self/stat` の (minflt, majflt)。
fn faults() -> Option<(u64, u64)> {
    let s = std::fs::read_to_string("/proc/self/stat").ok()?;
    // 2 番目 (comm) は括弧でくくられ、空白を含みうる。最後の ')' の後ろから数える。
    let rest = &s[s.rfind(')')? + 1..];
    let f: Vec<&str> = rest.split_whitespace().collect();
    // rest[0] = state (stat の 3 番目)。minflt は 10 番目、majflt は 12 番目。
    Some((f.get(7)?.parse().ok()?, f.get(9)?.parse().ok()?))
}

/// [ptr, ptr+len) のうち常駐しているバイト数 (mincore)。取れなければ 0。
#[cfg(unix)]
fn resident_bytes(ptr: usize, len: usize) -> u64 {
    if len == 0 || ptr == 0 {
        return 0;
    }
    const PAGE: usize = 4096;
    let start = ptr & !(PAGE - 1);
    let pages = (ptr + len - start).div_ceil(PAGE);
    if pages == 0 {
        return 0;
    }
    let mut vec = vec![0u8; pages];
    // glibc の `mincore(2)` (syscall の薄い包み)。
    unsafe {
        #[link(name = "c")]
        extern "C" {
            fn mincore(addr: *mut std::ffi::c_void, length: usize, vec: *mut u8) -> i32;
        }
        let rc = mincore(start as *mut std::ffi::c_void, pages * PAGE, vec.as_mut_ptr());
        if rc != 0 {
            return 0;
        }
    }
    vec.iter().filter(|&&b| b & 1 != 0).count() as u64 * PAGE as u64
}

#[cfg(not(unix))]
fn resident_bytes(_ptr: usize, _len: usize) -> u64 {
    0
}

/// リクエスト終端の 1 行。ゲート off なら `None`。
pub fn probe_line(tag: &str) -> Option<String> {
    if !probe_enabled() {
        return None;
    }
    let (minflt, majflt) = faults()?;
    let (dm, dM) = {
        let mut g = LAST_FAULTS.lock().unwrap_or_else(|e| e.into_inner());
        let d = (minflt.saturating_sub(g.0), majflt.saturating_sub(g.1));
        *g = (minflt, majflt);
        d
    };
    let mut line = format!(
        "[exp-ram-probe] {tag} minflt={dm} majflt={dM} memavail={}MiB",
        mem_available().map(|b| b >> 20).unwrap_or(0)
    );
    // task 514: ディスクから実際に読んだバイト (`/proc/self/io`)。
    // `io_read_MB` がタスク文の「disk bytes」、`io_rchar_MB` は read 系の要求量。
    if let Some((rc, rb)) = proc_io() {
        let mut g = LAST_IO.lock().unwrap_or_else(|e| e.into_inner());
        let d = (
            rc.saturating_sub(g.0) as f64 / 1e6,
            rb.saturating_sub(g.1) as f64 / 1e6,
        );
        *g = (rc, rb);
        line.push_str(&format!(
            " io_rchar_MB={:.1} io_read_MB={:.1}",
            d.0, d.1
        ));
    }
    if flag(PROBE_ENV) {
        let ranges = MMAP.read().unwrap_or_else(|e| e.into_inner());
        let mut res = 0u64;
        let mut tot = 0u64;
        for &(p, l) in ranges.iter() {
            res += resident_bytes(p, l);
            tot += l as u64;
        }
        line.push_str(&format!(
            " expert_resident={}MiB/{}MiB ({:.1}%)",
            res >> 20,
            tot >> 20,
            if tot > 0 {
                res as f64 * 100.0 / tot as f64
            } else {
                0.0
            }
        ));
        // task 514: モデル全体 (dense / PLE を含む) の常駐。差分が「expert 以外」。
        let fr = file_resident();
        let ftot: u64 = PARTMAP
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .map(|&(_, l)| l as u64)
            .sum();
        if ftot > 0 {
            line.push_str(&format!(
                " file_resident={}MiB/{}MiB ({:.1}%)",
                fr >> 20,
                ftot >> 20,
                fr as f64 * 100.0 / ftot as f64
            ));
        }
    }
    Some(line)
}

/// リクエストの先頭で expert の page cache を落とす (診断。dense は落とさない)。
/// ゲート off なら空 (呼び出し側は何も出力しない)。
pub fn drop_pages_if_enabled() -> Vec<String> {
    if !drop_enabled() {
        return Vec::new();
    }
    let parts = PARTS.read().unwrap_or_else(|e| e.into_inner()).clone();
    let ranges = RANGES.read().unwrap_or_else(|e| e.into_inner()).clone();
    vec![drop_ranges(&parts, &ranges)]
}

/// task 268: 指定した範囲の page cache を落とす (起動プラン用)。
///
/// `ranges` は `(パート番号, ファイル内オフセット, 長さ)`。戻り値は 1 行のログ。
/// **ゲートは呼び出し側** (`drop_on_start_enabled`)。
pub fn drop_ranges(
    parts: &[std::path::PathBuf],
    ranges: &[(usize, u64, u64)],
) -> String {
    let mut files: Vec<std::fs::File> = Vec::with_capacity(parts.len());
    for p in parts {
        match std::fs::File::open(p) {
            Ok(f) => files.push(f),
            Err(_) => return "[exp-cold] drop: ファイルを開けない".to_string(),
        }
    }
    let mut ok = 0usize;
    let mut err = 0usize;
    let mut bytes = 0u64;
    for &(part, off, len) in ranges {
        let Some(f) = files.get(part) else {
            err += 1;
            continue;
        };
        if fadvise_dontneed(f, off, len).is_ok() {
            ok += 1;
            bytes = bytes.saturating_add(len);
        } else {
            err += 1;
        }
    }
    format!(
        "[exp-cold] expert page cache dropped: ok={ok} err={err} {}MiB ({DROP_ON_START_ENV})",
        bytes >> 20
    )
}

/// task 268: 範囲の常駐バイト (mincore)。起動時の drop の前後を測るのに使う。
pub fn resident_bytes_of(ptr: usize, len: usize) -> u64 {
    resident_bytes(ptr, len)
}

#[cfg(unix)]
fn fadvise_dontneed(f: &std::fs::File, off: u64, len: u64) -> std::io::Result<()> {
    use std::os::unix::io::AsRawFd;
    extern "C" {
        fn posix_fadvise(
            fd: std::os::raw::c_int,
            off: i64,
            len: i64,
            advice: std::os::raw::c_int,
        ) -> std::os::raw::c_int;
    }
    const POSIX_FADV_DONTNEED: std::os::raw::c_int = 4;
    let rc = unsafe {
        posix_fadvise(
            f.as_raw_fd(),
            off as i64,
            len as i64,
            POSIX_FADV_DONTNEED,
        )
    };
    if rc == 0 {
        Ok(())
    } else {
        Err(std::io::Error::from_raw_os_error(rc))
    }
}

#[cfg(not(unix))]
fn fadvise_dontneed(_f: &std::fs::File, _off: u64, _len: u64) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "posix_fadvise は unix のみ",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mb_values_are_explicit() {
        // 既定 (未設定) は無効
        assert_eq!(mb_from_value(None), Some(0));
        assert_eq!(mb_from_value(Some("")), Some(0));
        // 自動は明示したときだけ
        assert_eq!(mb_from_value(Some("auto")), None);
        assert_eq!(mb_from_value(Some("AUTO")), None);
        assert_eq!(mb_from_value(Some("0")), Some(0));
        assert_eq!(mb_from_value(Some(" off ")), Some(0));
        assert_eq!(mb_from_value(Some("false")), Some(0));
        assert_eq!(mb_from_value(Some("4096")), Some(4096));
        // 誤植は無効 (勝手にメモリを確保しない)
        assert_eq!(mb_from_value(Some("4G")), Some(0));
        assert!(!enabled_from_value(Some("0")));
        assert!(!enabled_from_value(None));
        assert!(enabled_from_value(Some("auto")));
        assert!(enabled_from_value(Some("1024")));
    }

    #[test]
    fn budget_is_margin_bounded_and_capped_by_experts() {
        let gb = 1u64 << 30;
        // 未設定は 0 (既定 off)
        assert_eq!(budget_bytes(None, Some(50 * gb), 100 * gb, 8 * gb), 0);
        // 自動 (明示): MemAvailable - margin と expert 総量の小さいほう
        assert_eq!(
            budget_bytes(Some("auto"), Some(50 * gb), 100 * gb, 8 * gb),
            42 * gb
        );
        assert_eq!(
            budget_bytes(Some("auto"), Some(50 * gb), 10 * gb, 8 * gb),
            10 * gb
        );
        // 明示でも MemAvailable - margin を超えない
        assert_eq!(
            budget_bytes(Some("40960"), Some(50 * gb), 100 * gb, 8 * gb),
            40 * gb
        );
        assert_eq!(
            budget_bytes(Some("1024"), Some(50 * gb), 100 * gb, 8 * gb),
            gb
        );
        // 無効
        assert_eq!(budget_bytes(Some("0"), Some(50 * gb), 100 * gb, 8 * gb), 0);
        // MemAvailable が読めないときは明示予算だけが残る (自動は 0)
        assert_eq!(budget_bytes(Some("auto"), None, 100 * gb, 8 * gb), 0);
        assert_eq!(budget_bytes(Some("2048"), None, 100 * gb, 8 * gb), 2 * gb);
        // 余裕より空きが少ない
        assert_eq!(
            budget_bytes(Some("auto"), Some(4 * gb), 100 * gb, 8 * gb),
            0
        );
    }

    /// task 268: auto は「非 expert バイト + 余裕」を MemAvailable から引く。
    ///
    /// UD-Q4_K_XL の実形 (expert 77.0 GB / 非 expert 27.0 GB / MemAvailable
    /// 57 GiB / 余裕 8 GiB) で、tier が page cache を食い潰さない大きさになる。
    #[test]
    fn auto_budget_leaves_room_for_non_expert_bytes() {
        let gib = 1u64 << 30;
        let expert = 77 * gib;
        let non_expert = 27 * gib;
        // 旧規則 (非 expert を引かない) は 49 GiB を返した = t260c の失敗。
        assert_eq!(budget_bytes(Some("auto"), Some(57 * gib), expert, 8 * gib), 49 * gib);
        // 新規則: 57 - 27 - 8 = 22 GiB
        assert_eq!(
            budget_bytes_model(Some("auto"), Some(57 * gib), expert, non_expert, 8 * gib),
            22 * gib
        );
        // 明示値も同じ上限で頭打ち (22 GiB を超えて page cache を奪わない)
        assert_eq!(
            budget_bytes_model(Some("40960"), Some(57 * gib), expert, non_expert, 8 * gib),
            22 * gib
        );
        // 非 expert が小さければ従来どおり expert 総量で頭打ち
        assert_eq!(
            budget_bytes_model(Some("auto"), Some(57 * gib), 10 * gib, 2 * gib, 8 * gib),
            10 * gib
        );
        // 空きが非 expert + 余裕に満たない → 0 (tier を作らない)
        assert_eq!(
            budget_bytes_model(Some("auto"), Some(30 * gib), expert, non_expert, 8 * gib),
            0
        );
        // MemAvailable が読めない: 明示値だけが残る
        assert_eq!(
            budget_bytes_model(Some("4096"), None, expert, non_expert, 8 * gib),
            4 * gib
        );
        assert_eq!(
            budget_bytes_model(Some("auto"), None, expert, non_expert, 8 * gib),
            0
        );
    }

    /// task 268: ミス転送元の内訳が数えられ、reset で 0 に戻ること。
    #[test]
    fn miss_source_counters_are_reset() {
        miss_reset();
        note_miss_source(true);
        note_miss_source(true);
        note_miss_source(false);
        assert_eq!(miss_stats(), (2, 1));
        miss_reset();
        assert_eq!(miss_stats(), (0, 0));
    }

    /// task 268: 起動時 drop のゲートは未設定なら不活性。
    #[test]
    fn drop_on_start_is_inert_without_the_gate() {
        if std::env::var(DROP_ON_START_ENV).is_ok() {
            return; // ゲートを立てて走らせている環境では検査しない
        }
        assert!(!drop_on_start_enabled());
    }

    #[test]
    fn margin_and_flags_default() {
        assert_eq!(margin_mb_from_value(None), DEFAULT_MARGIN_MB);
        assert_eq!(margin_mb_from_value(Some("0")), DEFAULT_MARGIN_MB);
        assert_eq!(margin_mb_from_value(Some("2048")), 2048);
        assert!(pin_from_value(None));
        assert!(!pin_from_value(Some("0")));
        assert!(!pin_from_value(Some("off")));
        assert!(fill_from_value(None));
        assert!(!fill_from_value(Some("false")));
    }

    #[test]
    fn stats_line_is_none_without_tier() {
        install(None);
        assert!(stats_line().is_none());
        assert!(!active());
    }

    /// `/proc/self/stat` の minflt / majflt が読めること (単調非減少)。
    ///
    /// 「触ったら必ず増える」はアロケータの再利用で落ちるので検査しない
    /// (この binary は 1 プロセスで全テストを回す)。
    #[test]
    fn faults_are_readable_and_monotonic() {
        let a = faults().expect("/proc/self/stat が読めない");
        let mut v = vec![0u8; 8 << 20];
        for i in (0..v.len()).step_by(1024) {
            v[i] = 1;
        }
        let b = faults().expect("/proc/self/stat が読めない");
        assert!(b.0 >= a.0 && b.1 >= a.1, "fault が減った: {a:?} -> {b:?}");
        assert!(a.0 > 0 || b.0 > 0, "minflt が 0 のまま (解析が怪しい)");
    }

    /// mincore の包みが、触ったページを常駐として数えること。
    #[test]
    fn resident_bytes_counts_touched_pages() {
        let n = 8 << 20;
        let mut v = vec![0u8; n];
        for i in (0..n).step_by(4096) {
            v[i] = 1;
        }
        let r = resident_bytes(v.as_ptr() as usize, n);
        assert!(
            r >= (n as u64) / 2,
            "mincore が常駐を数えていない: {r} / {n}"
        );
        assert_eq!(resident_bytes(0, n), 0);
        assert_eq!(resident_bytes(v.as_ptr() as usize, 0), 0);
    }

    /// drop のゲートが off なら何もしない (1 行も出さない)。
    #[test]
    fn drop_is_inert_without_the_gate() {
        if std::env::var(DROP_ENV).is_ok() {
            return; // ゲートを立てて走らせている環境では検査しない
        }
        assert!(drop_pages_if_enabled().is_empty());
    }

    /// task 514: `DENNO_PF_FADV_END` の終端 drop もゲート off なら `None`。
    #[test]
    fn end_drop_is_inert_without_the_gate() {
        if std::env::var(FADV_END_ENV).is_ok() {
            return;
        }
        assert!(drop_expert_pages_end("test").is_none());
    }

    /// task 514: この環境での `posix_fadvise(DONTNEED)` の挙動を固定する。
    ///
    /// 実測 (kernel 7.0.0): **mapcount 0 のページは落ちる**が、
    /// **mmap してフォールトしたページは落ちない**。`DENNO_PF_FADV_END` を
    /// 「リクエスト終端に expert 範囲を落とす」設計にしたのはこの性質のため
    /// (pread の読み残しだけが消え、mmap で使ったページは残る)。
    #[test]
    #[cfg(unix)]
    fn fadvise_drops_unmapped_but_not_mapped_pages() {
        use std::io::Write;
        use std::os::unix::fs::FileExt;
        let dir = std::env::temp_dir().join("denno_t514_fadv");
        let _ = std::fs::create_dir_all(&dir);
        let p = dir.join("f.bin");
        let n = 8usize << 20;
        {
            let mut f = std::fs::File::create(&p).expect("create");
            f.write_all(&vec![9u8; n]).expect("write");
            f.sync_all().expect("sync");
        }
        let file = std::fs::File::open(&p).expect("open");
        let map = unsafe { memmap2::Mmap::map(&file).expect("mmap") };
        let ptr = map.as_ptr() as usize;
        let parts = vec![p.clone()];
        let ranges = vec![(0usize, 0u64, n as u64)];

        // (1) pread で page cache に入れる (まだ 1 ページもフォールトしていない)。
        let mut buf = vec![0u8; n];
        let mut off = 0usize;
        while off < n {
            let got = file.read_at(&mut buf[off..], off as u64).expect("pread");
            off += got;
        }
        assert!(buf[n - 1] == 9);
        let before = resident_bytes(ptr, n);
        assert_eq!(before, n as u64, "pread のページが常駐していない: {before}");
        let _ = drop_ranges(&parts, &ranges);
        assert_eq!(
            resident_bytes(ptr, n),
            0,
            "unmapped のページが落ちていない"
        );

        // (2) mmap でフォールトしてから落とす → 残る。
        let mut s = 0u64;
        for i in (0..n).step_by(4096) {
            s += map[i] as u64;
        }
        assert!(s > 0);
        let faulted = resident_bytes(ptr, n);
        assert!(faulted >= (n as u64) / 2, "フォールトしていない: {faulted}");
        let _ = drop_ranges(&parts, &ranges);
        assert!(
            resident_bytes(ptr, n) >= faulted / 2,
            "mapped のページまで落ちている (設計と違う)"
        );
    }

    /// task 514: `/proc/self/io` の差分が読める (read した量が rchar に出る)。
    #[test]
    #[cfg(unix)]
    fn proc_io_see_pread_bytes() {
        use std::io::Write;
        use std::os::unix::fs::FileExt;
        let dir = std::env::temp_dir().join("denno_t514_io");
        let _ = std::fs::create_dir_all(&dir);
        let p = dir.join("i.bin");
        let n = 4usize << 20;
        {
            let mut f = std::fs::File::create(&p).expect("create");
            f.write_all(&vec![1u8; n]).expect("write");
        }
        let (r0, b0) = proc_io().expect("/proc/self/io");
        let file = std::fs::File::open(&p).expect("open");
        let mut buf = vec![0u8; n];
        let mut off = 0usize;
        while off < n {
            let got = file.read_at(&mut buf[off..], off as u64).expect("pread");
            off += got;
        }
        let (r1, b1) = proc_io().expect("/proc/self/io");
        assert!(
            r1.saturating_sub(r0) >= n as u64,
            "rchar の差分が足りない: {} < {n}",
            r1 - r0
        );
        // read_bytes は「ディスクから読んだ」ぶんなので 0 でもよい (page cache)。
        assert!(b1 >= b0);
    }

    /// task 514: `install_part_map` が覚えた範囲で `file_resident` が数えられる。
    #[test]
    fn part_map_is_used_by_file_resident() {
        let n = 4 << 20;
        let mut v = vec![0u8; n];
        for i in (0..n).step_by(4096) {
            v[i] = 7;
        }
        install_part_map(vec![(v.as_ptr() as usize, n)]);
        let r = file_resident();
        assert!(r >= (n as u64) / 2, "file_resident が数えていない: {r}");
        install_part_map(Vec::new());
        assert_eq!(file_resident(), 0);
    }

    // -----------------------------------------------------------------
    // task 907: 再調整プラン (thrash しない形) と 32-64 GiB RAM の予算
    // -----------------------------------------------------------------

    /// 「回数の昇順」の合成プラン。(0, 0) が一番熱く、(47, 255) が一番冷たい。
    fn synthetic_plan(n_layer: u16, n_expert: u16) -> Vec<(u16, u16)> {
        let mut v: Vec<(u64, (u16, u16))> = (0..n_layer)
            .flat_map(|l| (0..n_expert).map(move |e| (l, e)))
            .map(|(l, e)| {
                let count = (n_layer as u64 - l as u64) * 1000 + (n_expert as u64 - e as u64);
                (count, (l, e))
            })
            .collect();
        v.sort_by_key(|(c, k)| (*c, *k));
        v.into_iter().map(|(_, k)| k).collect()
    }

    fn index_of(plan: &[(u16, u16)]) -> std::collections::HashMap<(u16, u16), usize> {
        plan.iter().enumerate().map(|(i, k)| (*k, i)).collect()
    }

    /// 欲しい集合は回数の上位 `cap` 本。載せる順は昇順 (冷たい順) で決定的。
    #[test]
    fn task907_rebalance_plan_takes_the_hottest_cap_and_fills_ascending() {
        let plan = synthetic_plan(48, 256);
        let cap = 10_922; // 32 GiB / 3 MiB 相当
        // 冷たい方から 1,000 本だけ常駐している (起動充填が途中で止まった形)。
        let have: Vec<(u16, u16)> = plan.iter().take(1_000).copied().collect();
        let vram = std::collections::HashSet::new();
        let (drop, fill) = rebalance_plan(&plan, cap, &vram, &have);
        assert_eq!(drop.len(), 1_000, "欲しくない常駐を落としていない");
        let want: std::collections::HashSet<(u16, u16)> =
            plan.iter().rev().take(cap).copied().collect();
        assert!(fill.iter().all(|k| want.contains(k)), "上位 cap 本以外を埋めた");
        assert_eq!(fill.len(), cap, "上位 cap 本を埋めていない");
        // 昇順 = 一番熱いのが最後に触られる。
        let idx = index_of(&plan);
        let pos: Vec<usize> = fill.iter().map(|k| idx[k]).collect();
        let mut sorted = pos.clone();
        sorted.sort_unstable();
        assert_eq!(pos, sorted, "載せる順が昇順でない (熱いのが先に落ちる)");
        // 落とす集合と載せる集合は交わらない。
        let fset: std::collections::HashSet<(u16, u16)> = fill.iter().copied().collect();
        assert!(drop.iter().all(|k| !fset.contains(k)));
    }

    /// 定常状態では `(空, 空)`。30 秒ごとに同じ鍵を落として載せ直さない。
    #[test]
    fn task907_rebalance_plan_is_a_fixpoint() {
        let plan = synthetic_plan(48, 256);
        let cap = 6_000;
        let have: Vec<(u16, u16)> = plan.iter().rev().take(cap).copied().collect();
        let vram = std::collections::HashSet::new();
        let (drop, fill) = rebalance_plan(&plan, cap, &vram, &have);
        assert!(drop.is_empty() && fill.is_empty(), "定常状態で thrash した");
        // cap が 0 / プランが空なら何もしない。
        assert_eq!(
            rebalance_plan(&plan, 0, &vram, &have),
            (Vec::new(), Vec::new())
        );
        assert_eq!(
            rebalance_plan(&[], cap, &vram, &have),
            (Vec::new(), Vec::new())
        );
    }

    /// 回数が 1 本ぶん入れ替わったときの入れ替えは 1 本だけ (雪崩を起こさない)。
    #[test]
    fn task907_rebalance_plan_churns_one_key_for_a_one_key_drift() {
        let plan = synthetic_plan(48, 256);
        let cap = 6_000;
        let have: Vec<(u16, u16)> = plan.iter().rev().take(cap).copied().collect();
        // 一番冷たかった鍵が一番熱くなる (新しい話題が 1 本だけ入った形)。
        let mut drifted = plan.clone();
        let hot = drifted.remove(0);
        drifted.push(hot);
        let vram = std::collections::HashSet::new();
        let (drop, fill) = rebalance_plan(&drifted, cap, &vram, &have);
        assert_eq!(
            (drop.len(), fill.len()),
            (1, 1),
            "1 本の変化で {} 本落として {} 本載せた (thrash)",
            drop.len(),
            fill.len()
        );
        // 落ちるのは cap の境界で押し出された鍵、載るのは新しく熱くなった鍵。
        assert_eq!(drop[0], plan[plan.len() - cap], "押し出しの鍵が違う");
        assert_eq!(fill[0], hot);
        assert_ne!(drop[0], fill[0]);
        // もう一度呼ぶと固定点。
        let mut have2 = have.clone();
        have2.retain(|k| *k != drop[0]);
        have2.push(fill[0]);
        assert_eq!(
            rebalance_plan(&drifted, cap, &vram, &have2),
            (Vec::new(), Vec::new())
        );
    }

    /// VRAM に居る鍵は RAM tier に重複して置かない (二重常駐しない)。
    #[test]
    fn task907_rebalance_plan_never_keeps_a_vram_key_in_ram() {
        let plan = synthetic_plan(48, 256);
        let cap = 6_000;
        let vram: std::collections::HashSet<(u16, u16)> =
            plan.iter().rev().take(500).copied().collect();
        let have: Vec<(u16, u16)> = plan.iter().rev().take(cap).copied().collect();
        let (drop, fill) = rebalance_plan(&plan, cap, &vram, &have);
        assert!(fill.iter().all(|k| !vram.contains(k)), "VRAM の鍵を載せた");
        assert_eq!(drop.len(), 500, "VRAM へ移った鍵を RAM に残した");
        assert!(drop.iter().all(|k| vram.contains(k)));
        // 欲しい集合 (cap 本から VRAM を除く) のうち、いま載っていない 500 本を埋める。
        assert_eq!(fill.len(), 500, "欲しい集合の残りを埋めていない");
    }

    /// 32-64 GiB の RAM で、auto 予算が page cache を食い潰さないこと。
    #[test]
    fn task907_ram_budget_fits_32_to_64_gib_of_ram() {
        let gib = 1u64 << 30;
        let expert = 77 * gib;
        let non_expert = 27 * gib;
        let margin = 8 * gib;
        // 64 GiB: 64 - 27 - 8 = 29 GiB
        assert_eq!(
            budget_bytes_model(Some("auto"), Some(64 * gib), expert, non_expert, margin),
            29 * gib
        );
        // 32 GiB: 32 - 27 - 8 < 0 → tier を作らない
        assert_eq!(
            budget_bytes_model(Some("auto"), Some(32 * gib), expert, non_expert, margin),
            0
        );
        // 指定 16 GiB でも空きの範囲 (48 - 27 - 8 = 13 GiB) を超えない
        assert_eq!(
            budget_bytes_model(Some("16384"), Some(48 * gib), expert, non_expert, margin),
            13 * gib
        );
        // どこでも「空き - (非 expert + 余裕)」と expert 総量を超えない。
        for avail_gib in 32..=64u64 {
            let b = budget_bytes_model(Some("auto"), Some(avail_gib * gib), expert, non_expert, margin);
            let room = (avail_gib * gib).saturating_sub(non_expert.saturating_add(margin));
            assert!(b <= expert, "{avail_gib} GiB: expert 総量を超えた");
            assert!(b <= room, "{avail_gib} GiB: {b} > room {room}");
        }
    }
}
