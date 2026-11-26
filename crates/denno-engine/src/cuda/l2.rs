//! task 1372 (移植は task 1577): RTX 5090 の 96 MB L2 を「使い回す小さな作業領域」に
//! 固定する (`CU_LIMIT_PERSISTING_L2_CACHE_SIZE` + `CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW`)。
//!
//! 既定 **off**。`DENNO_CUDA_L2_PERSIST=1` (1/on/true/yes) のときだけ動く。
//! 量子化・カーネル・数値には一切触らない (md5 不変)。効くのは L2 の置き方だけ。
//!
//! task 1577 の差分: ストリームの窓を張るのは **GDN 状態の連結プール**
//! (`DENNO_CUDA_GDN_POOL`) の 1 本だけにした。窓は persisting 予約の大きさへ
//! clamp する ([`persist_range_budgeted`])。br1372 が持っていた arena チャンク
//! (1 GiB) の窓は移植していない — 大きい窓が先に勝つと GDN プールが固定されず、
//! 「1 本の窓で GDN 全層を固定する」というこのタスクの検証ができなくなるため。
//!
//! 設計:
//! - 属性 (`L2_CACHE_SIZE` / `MAX_PERSISTING_L2_CACHE_SIZE` /
//!   `MAX_ACCESS_POLICY_WINDOW_SIZE`) は `cuDeviceGetAttribute` で 1 回だけ引く。
//! - `cuCtxSetLimit(CU_LIMIT_PERSISTING_L2_CACHE_SIZE, bytes)` は CudaCtx 生成時に
//!   1 回だけ。上限は `DENNO_CUDA_L2_PERSIST_MB` (既定 48 MiB) を
//!   max_persisting / L2 サイズで clamp した値。
//! - 窓 (`accessPolicyWindow`) はストリームごとに 1 本しか持てないので、
//!   task 1577 では GDN 状態プールの連続範囲を 1 本だけ固定する。
//!   窓が既により大きい範囲を指しているときは張り替えない
//!   (`window_action`。窓の取り合いで driver を毎トークン呼ばないため)。
//!   hitProp=PERSISTING / missProp=STREAMING。
//! - 属性が 0 (未対応 GPU / driver) やシンボル欠落では静かに no-op
//!   (失敗も panic もしない)。

#![allow(non_camel_case_types)]

use super::*;

/// CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE
pub(crate) const CU_DEV_ATTR_L2_CACHE_SIZE: c_int = 38;
/// CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE
pub(crate) const CU_DEV_ATTR_MAX_PERSISTING_L2_CACHE_SIZE: c_int = 108;
/// CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE
pub(crate) const CU_DEV_ATTR_MAX_ACCESS_POLICY_WINDOW_SIZE: c_int = 109;

/// CU_LIMIT_PERSISTING_L2_CACHE_SIZE
pub(crate) const CU_LIMIT_PERSISTING_L2_CACHE_SIZE: c_int = 0x06;
/// CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW
pub(crate) const CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW: c_int = 1;

/// CU_ACCESS_PROPERTY_NORMAL (窓の miss 側の既定値。テストで値だけ固定する)
#[allow(dead_code)]
pub(crate) const CU_ACCESS_PROPERTY_NORMAL: c_int = 0;
pub(crate) const CU_ACCESS_PROPERTY_STREAMING: c_int = 1;
pub(crate) const CU_ACCESS_PROPERTY_PERSISTING: c_int = 2;

/// `CUaccessPolicyWindow` (CUDA driver ヘッダと同じ並び・同じサイズ)。
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CUaccessPolicyWindow {
    pub base_ptr: CUdeviceptr,
    pub num_bytes: usize,
    pub hit_ratio: f32,
    pub hit_prop: c_int,
    pub miss_prop: c_int,
}

/// `CUstreamAttrValue`。実体は union だが、driver は先頭の
/// `accessPolicyWindow` だけを読む。ヘッダの union と同じ 32 バイトに固定する
/// (テストで `size_of` と各フィールドのオフセットを固定している)。
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) union CUstreamAttrValue {
    pub access_policy_window: CUaccessPolicyWindow,
    pub sync_policy: c_int,
    /// CUDA ヘッダの `CUstreamAttrValue` と同じ 32 バイトにするための詰め物。
    pub _pad: [u64; 4],
}


pub(crate) const MIB: usize = 1024 * 1024;
/// `DENNO_CUDA_L2_PERSIST_MB` 未指定時の上限 (MiB)。
pub(crate) const DEFAULT_LIMIT_MIB: usize = 48;
/// 未指定時にこれを超えない (48-64 MiB の枠。48 が既定なので実質 48)。
pub(crate) const HARD_CAP_MIB: usize = 64;


/// `DENNO_CUDA_L2_PERSIST` の真偽表 (純関数)。1/on/true/yes だけが有効。
/// 未設定・空・0/off/false/no・その他の値はすべて off。
pub(crate) fn parse_enabled(raw: Option<&str>) -> bool {
    matches!(
        raw.map(|v| v.trim().to_ascii_lowercase()).as_deref(),
        Some("1") | Some("on") | Some("true") | Some("yes")
    )
}


/// `DENNO_CUDA_L2_PERSIST_MB` の解釈 (純関数)。正の整数だけを採用する。
pub(crate) fn parse_mb(raw: Option<&str>) -> Option<usize> {
    raw.map(str::trim)
        .filter(|s| !s.is_empty())
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|v| *v > 0)
}


/// persisting L2 に要求するバイト数を決める (純関数)。
///
/// - `max_persisting == 0` (未対応) なら 0。
/// - 要求 (`req_mb`) が無ければ既定 48 MiB。
/// - `max_persisting` と `l2_bytes` の小さい方で clamp する
///   (driver は上限超過の `cuCtxSetLimit` を `CUDA_ERROR_INVALID_VALUE` で
///   弾くので、ここで丸めてから渡す)。
pub(crate) fn clamp_limit_bytes(
    l2_bytes: usize,
    max_persisting: usize,
    req_mb: Option<usize>,
) -> usize {
    if max_persisting == 0 || l2_bytes == 0 {
        return 0;
    }
    let cap = max_persisting.min(l2_bytes);
    let want = match req_mb {
        Some(mb) => mb.saturating_mul(MIB),
        None => DEFAULT_LIMIT_MIB.min(HARD_CAP_MIB) * MIB,
    };
    want.min(cap)
}


/// 窓の `num_bytes` を決める (純関数)。0 (未対応) なら 0。
/// `MAX_ACCESS_POLICY_WINDOW_SIZE` を超えると driver が弾くので clamp する。
pub(crate) fn window_bytes(requested: usize, max_window: usize) -> usize {
    if max_window == 0 {
        0
    } else {
        requested.min(max_window)
    }
}


/// 実際に効いている設定 (診断ログ・テスト用)。
///
/// フィールドは `[l2-persist]` 1 行とテストでしか読まない (ホットパスでは
/// `enabled` と `limit_applied` だけを見る)。
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct L2Config {
    pub enabled: bool,
    pub l2_bytes: usize,
    pub max_persisting: usize,
    pub max_window: usize,
    pub limit_bytes: usize,
    /// 有効化できなかった理由 (有効なら空文字)。
    pub reason: &'static str,
}


pub(crate) struct L2State {
    pub cfg: L2Config,
    /// `cuCtxSetLimit` が成功したか (窓を張る前提条件)。
    pub limit_applied: bool,
    /// いま張っている窓 (base, num_bytes)。ストリームごとに 1 本しか持てない。
    pub window: Mutex<Option<(CUdeviceptr, usize)>>,
}


static STATE: OnceLock<L2State> = OnceLock::new();


/// `dev` の属性を 1 個引く。引けなければ None。
fn attr(dev: CUdevice, id: c_int) -> Option<usize> {
    let l = lib().ok()?;
    let f = l.device_get_attribute?;
    let mut v: c_int = 0;
    if unsafe { f(&mut v, id, dev) } != CUDA_SUCCESS || v <= 0 {
        return None;
    }
    Some(v as usize)
}


/// CudaCtx 生成時に 1 回だけ呼ぶ。env off / 未対応 GPU では no-op。
///
/// 戻り値は常に `&'static L2State` (無効でも `cfg.enabled == false` で返る)。
pub(crate) fn init(dev: CUdevice) -> &'static L2State {
    STATE.get_or_init(|| {
        let raw = std::env::var("DENNO_CUDA_L2_PERSIST").ok();
        let want = parse_enabled(raw.as_deref());
        if !want {
            return disabled(0, 0, 0, "env off");
        }
        let l2 = attr(dev, CU_DEV_ATTR_L2_CACHE_SIZE).unwrap_or(0);
        let maxp = attr(dev, CU_DEV_ATTR_MAX_PERSISTING_L2_CACHE_SIZE).unwrap_or(0);
        let maxw = attr(dev, CU_DEV_ATTR_MAX_ACCESS_POLICY_WINDOW_SIZE).unwrap_or(0);
        if maxp == 0 {
            return disabled(l2, 0, maxw, "persisting L2 unsupported");
        }
        let req_mb = parse_mb(std::env::var("DENNO_CUDA_L2_PERSIST_MB").ok().as_deref());
        let limit = clamp_limit_bytes(l2, maxp, req_mb);
        if limit == 0 {
            return disabled(l2, maxp, maxw, "clamped to 0");
        }
        let mut limit_applied = false;
        if let Ok(l) = lib() {
            if let Some(f) = l.ctx_set_limit {
                limit_applied =
                    unsafe { f(CU_LIMIT_PERSISTING_L2_CACHE_SIZE, limit) } == CUDA_SUCCESS;
            }
        }
        let reason = if limit_applied { "" } else { "cuCtxSetLimit failed" };
        let st = L2State {
            cfg: L2Config {
                enabled: limit_applied,
                l2_bytes: l2,
                max_persisting: maxp,
                max_window: maxw,
                limit_bytes: limit,
                reason,
            },
            limit_applied,
            window: Mutex::new(None),
        };
        tracing::info!(
            "[l2-persist] {} l2={} MiB max_persist={} MiB max_window={} MiB limit={} MiB{}",
            if st.cfg.enabled { "on" } else { "off" },
            l2 / MIB,
            maxp / MIB,
            maxw / MIB,
            limit / MIB,
            if st.cfg.reason.is_empty() {
                String::new()
            } else {
                format!(" ({})", st.cfg.reason)
            }
        );
        st
    })
}


fn disabled(l2: usize, maxp: usize, maxw: usize, reason: &'static str) -> L2State {
    L2State {
        cfg: L2Config {
            enabled: false,
            l2_bytes: l2,
            max_persisting: maxp,
            max_window: maxw,
            limit_bytes: 0,
            reason,
        },
        limit_applied: false,
        window: Mutex::new(None),
    }
}


/// 初期化済みなら設定を返す (未初期化 = None)。
pub(crate) fn state() -> Option<&'static L2State> {
    STATE.get()
}


/// 既存の窓に対する操作の決定 (純関数)。ストリームに窓は 1 本しか持てないので、
/// 「大きい範囲 > 小さい範囲」「同じ窓はそのまま」の規則をここに閉じ込める。
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum WindowAction {
    /// 既に同じ窓が効いている (driver を呼ばない)。
    Reuse,
    /// 既存の窓の方が大きい (張り替えない)。
    Skip,
    /// この範囲で張り替える。
    Replace,
}

pub(crate) fn window_action(
    existing: Option<(CUdeviceptr, usize)>,
    base: CUdeviceptr,
    nb: usize,
) -> WindowAction {
    match existing {
        None => WindowAction::Replace,
        Some((b, n)) if b == base && n == nb => WindowAction::Reuse,
        Some((_, n)) if n >= nb => WindowAction::Skip,
        _ => WindowAction::Replace,
    }
}


/// 再利用する device 範囲を persisting にする (default stream 上の窓)。
///
/// ストリームに窓は 1 本しか持てないので、**既により大きい (または同サイズの)
/// 窓があれば張り替えない**。これで「1 GiB の arena チャンクが、あとから来る
/// 層ごとの GDN 状態 (数百 KiB) に上書きされる」のを防ぐ。
/// 戻り値: この範囲が persisting になっている (既にそうだった場合も含む) なら true。
pub(crate) fn persist_range(base: CUdeviceptr, bytes: usize) -> bool {
    let Some(st) = STATE.get() else {
        return false;
    };
    if !st.cfg.enabled || !st.limit_applied || base == 0 {
        return false;
    }
    let nb = window_bytes(bytes, st.cfg.max_window);
    if nb == 0 {
        return false;
    }
    let mut w = match st.window.lock() {
        Ok(g) => g,
        Err(_) => return false,
    };
    if let Some((b, n)) = *w {
        match window_action(Some((b, n)), base, nb) {
            WindowAction::Reuse => return true, // 毎トークンの再設定を避ける
            WindowAction::Skip => return false, // より大きい範囲を優先したまま
            WindowAction::Replace => {}
        }
    }
    let Ok(l) = lib() else { return false };
    let Some(f) = l.stream_set_attribute else {
        return false;
    };
    let v = CUstreamAttrValue {
        access_policy_window: CUaccessPolicyWindow {
            base_ptr: base,
            num_bytes: nb,
            hit_ratio: 1.0,
            hit_prop: CU_ACCESS_PROPERTY_PERSISTING,
            miss_prop: CU_ACCESS_PROPERTY_STREAMING,
        },
    };
    let rc = unsafe { f(std::ptr::null_mut(), CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW, &v) };
    if rc != CUDA_SUCCESS {
        tracing::info!("[l2-persist] window 設定に失敗 (rc={rc}) base=0x{base:x} bytes={nb}");
        return false;
    }
    // 1 行だけ出す (ゲートが発火した証拠。失敗時は上の行が出る)。
    tracing::info!(
        "[l2-persist] window base=0x{base:x} bytes={} MiB hit=persisting miss=streaming",
        nb / MIB
    );
    *w = Some((base, nb));
    true
}


/// task 1577: `bytes` が persisting 予約 (`DENNO_CUDA_L2_PERSIST_MB`) より大きい
/// 連続領域を対象にするとき用。窓を予約サイズまでに切り詰めてから張る。
///
/// 窓の `num_bytes` が予約を超えると driver は「予約に入りきらない分」を
/// 追い出すので、hit_ratio=1.0 のままだと全層を舐めるプールでは当たり率が
/// 落ちる。予約ぶんだけを persisting にし、残りは STREAMING のままにする
/// (数値・発行順は不変。効くのは L2 の置き方だけ)。
pub(crate) fn persist_range_budgeted(base: CUdeviceptr, bytes: usize) -> bool {
    let Some(st) = STATE.get() else {
        return false;
    };
    let budget = st.cfg.limit_bytes;
    if budget == 0 || bytes == 0 {
        return false;
    }
    persist_range(base, bytes.min(budget))
}


/// 窓と persisting の予約を捨てる (teardown / モデル入れ替え時)。
pub(crate) fn reset() {
    let Some(st) = STATE.get() else { return };
    if !st.cfg.enabled {
        return;
    }
    if let Ok(mut w) = st.window.lock() {
        if w.is_none() {
            return;
        }
        *w = None;
    }
    if let Ok(l) = lib() {
        if let Some(f) = l.ctx_reset_persisting_l2 {
            let _ = unsafe { f() };
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    /// env の真偽表: 1/on/true/yes (大小・前後空白を許す) だけ有効。
    #[test]
    fn l2_persist_env_truth_table() {
        for on in ["1", "on", "ON", " true ", "yes", "Yes", "TRUE"] {
            assert!(parse_enabled(Some(on)), "{on:?} は有効のはず");
        }
        for off in ["", "0", "off", "OFF", "false", "no", "2", "enable", "1x"] {
            assert!(!parse_enabled(Some(off)), "{off:?} は無効のはず");
        }
        assert!(!parse_enabled(None));
    }

    /// MB は正の整数だけ。負・小数・空・非数は None (既定へ落ちる)。
    #[test]
    fn l2_persist_mb_parse() {
        assert_eq!(parse_mb(Some("32")), Some(32));
        assert_eq!(parse_mb(Some(" 64 ")), Some(64));
        assert_eq!(parse_mb(Some("0")), None);
        assert_eq!(parse_mb(Some("-1")), None);
        assert_eq!(parse_mb(Some("1.5")), None);
        assert_eq!(parse_mb(Some("")), None);
        assert_eq!(parse_mb(None), None);
    }

    /// 上限の clamp: 未対応 (0) / 小さい max / 要求過大 / 既定。
    #[test]
    fn l2_persist_limit_clamping() {
        // 未対応 GPU: 何を要求しても 0
        assert_eq!(clamp_limit_bytes(96 * MIB, 0, Some(48)), 0);
        // 既定は 48 MiB
        assert_eq!(clamp_limit_bytes(96 * MIB, 72 * MIB, None), DEFAULT_LIMIT_MIB * MIB);
        // 要求は max_persisting で丸める
        assert_eq!(clamp_limit_bytes(96 * MIB, 72 * MIB, Some(10_000)), 72 * MIB);
        // max_persisting が L2 より大きい (あり得ない値) でも L2 で丸める
        assert_eq!(clamp_limit_bytes(32 * MIB, 72 * MIB, Some(64)), 32 * MIB);
        // 明示要求はそのまま (上限内)
        assert_eq!(clamp_limit_bytes(96 * MIB, 72 * MIB, Some(16)), 16 * MIB);
        // 48 MiB 未満の max では max がそのまま既定になる
        assert_eq!(clamp_limit_bytes(96 * MIB, 20 * MIB, None), 20 * MIB);
    }

    /// 窓のバイト数: max_window=0 (未対応) は 0、超える要求は clamp。
    #[test]
    fn l2_persist_window_clamping() {
        assert_eq!(window_bytes(1 << 30, 0), 0);
        assert_eq!(window_bytes(1 << 30, 128 * MIB), 128 * MIB);
        assert_eq!(window_bytes(4 * MIB, 128 * MIB), 4 * MIB);
        assert_eq!(window_bytes(0, 128 * MIB), 0);
    }

    /// ABI: `CUstreamAttrValue` は CUDA ヘッダの union と同じ 32 バイト、
    /// `CUaccessPolicyWindow` は 32 バイトで、各フィールドのオフセットが固定。
    /// ここがずれると driver が別の値を窓として読む (エラーも出ない)。
    #[test]
    fn l2_persist_struct_abi() {
        assert_eq!(std::mem::size_of::<CUaccessPolicyWindow>(), 32);
        assert_eq!(std::mem::align_of::<CUaccessPolicyWindow>(), 8);
        assert_eq!(std::mem::size_of::<CUstreamAttrValue>(), 32);
        assert_eq!(std::mem::align_of::<CUstreamAttrValue>(), 8);
        assert_eq!(std::mem::offset_of!(CUaccessPolicyWindow, base_ptr), 0);
        assert_eq!(std::mem::offset_of!(CUaccessPolicyWindow, num_bytes), 8);
        assert_eq!(std::mem::offset_of!(CUaccessPolicyWindow, hit_ratio), 16);
        assert_eq!(std::mem::offset_of!(CUaccessPolicyWindow, hit_prop), 20);
        assert_eq!(std::mem::offset_of!(CUaccessPolicyWindow, miss_prop), 24);
        // driver ヘッダの enum 値 (CU_ACCESS_PROPERTY_*)。ここがずれると
        // 「persisting のつもりが normal」になり、窓が黙って効かなくなる。
        assert_eq!(CU_ACCESS_PROPERTY_NORMAL, 0);
        assert_eq!(CU_ACCESS_PROPERTY_STREAMING, 1);
        assert_eq!(CU_ACCESS_PROPERTY_PERSISTING, 2);
        assert_eq!(CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW, 1);
        assert_eq!(CU_LIMIT_PERSISTING_L2_CACHE_SIZE, 0x06);
        assert_eq!(CU_DEV_ATTR_L2_CACHE_SIZE, 38);
        assert_eq!(CU_DEV_ATTR_MAX_PERSISTING_L2_CACHE_SIZE, 108);
        assert_eq!(CU_DEV_ATTR_MAX_ACCESS_POLICY_WINDOW_SIZE, 109);
    }

    /// 窓の張り替え規則: 大きい範囲を優先し、同じ窓は再利用する。
    /// これが壊れると、毎トークン driver を呼ぶか、arena の窓が GDN の
    /// 小さな窓に奪われる。
    #[test]
    fn l2_persist_window_action() {
        assert_eq!(window_action(None, 0x1000, 4096), WindowAction::Replace);
        // 同じ窓 → driver を呼ばない
        assert_eq!(
            window_action(Some((0x1000, 4096)), 0x1000, 4096),
            WindowAction::Reuse
        );
        // 既存が大きい → 奪わない
        assert_eq!(
            window_action(Some((0x1000, 128 * MIB)), 0x2000, 4 * MIB),
            WindowAction::Skip
        );
        // 同サイズの別 base → 奪わない (先着優先)
        assert_eq!(
            window_action(Some((0x1000, 4096)), 0x2000, 4096),
            WindowAction::Skip
        );
        // 新しく来た方が大きい → 張り替える
        assert_eq!(
            window_action(Some((0x1000, 4096)), 0x2000, 8 * MIB),
            WindowAction::Replace
        );
    }

    /// 未初期化 (driver 無し / GPU 無し) では窓の設定が静かに false を返し、
    /// `reset()` も panic しない。CPU-only CI で走る経路。
    #[test]
    fn l2_persist_noop_without_init() {
        // このプロセスで他のテストが init している可能性は低いが、
        // init 済みでも「無効」なら false のはず (env off が既定)。
        assert!(!persist_range(0, 0));
        // task 1577: 予算で clamp する版も同じく no-op。
        assert!(!persist_range_budgeted(0, 0));
        assert!(!persist_range_budgeted(0x1000, 64 * MIB));
        reset(); // panic しないこと
        if let Some(st) = state() {
            if !st.cfg.enabled {
                assert!(!persist_range(0x1000, 4096));
                assert!(!persist_range_budgeted(0x1000, 4096));
            }
        }
    }
}
