//! task 1377: Blackwell (SM 90+) の Programmatic Dependent Launch (PDL)。
//!
//! `cuLaunchKernelEx` に `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`
//! を付けると、同じストリームの**次の**カーネルを、前のカーネルの最後の
//! ストアが終わる前から起動準備できる (grid の立ち上げと重みの先読みを隠す)。
//!
//! 既定 **off**。`DENNO_CUDA_PDL=1` (1/on/true/yes) のときだけ動く。
//! 数値・カーネル・発行順は変えない (md5 不変が前提)。効くのは発行の
//! 属性だけなので、前のカーネルが完走してから次が走る順序は同じ。
//!
//! 設計:
//! - `cuLaunchKernelEx` は `dlsym` で **任意** に取る。古い driver /
//!   シンボル欠落では `None` = 従来の `cuLaunchKernel` に静かに落ちる。
//! - SM は `CudaCtx::new` が 1 回だけ [`set_device_sm`] で置く。90 未満
//!   (Ampere 以前) では属性を付けない。
//! - 付けるのは **decode 1 トークンの鎖** (`forward_token_at` の `nt == 1` 窓) の
//!   **null ストリーム** の発行だけ。[`decode_window`] のガードが窓を作る。
//! - 属性付き起動が失敗したら (driver が属性を知らない等) その 1 回は
//!   `cuLaunchKernel` でやり直す。失敗は数えて `[cuda-pdl]` の 1 行に出す。
//!
//! 前のカーネルが `cudaTriggerProgrammaticLaunchCompletion` を呼ばない場合、
//! トリガは「前のカーネルの完走時」に暗黙に発火するので、後続が前の書き込みを
//! 読む順序は保たれる (PTX 側の変更は不要。task 1377 は .ptx を触らない)。

use super::*;
use std::cell::Cell;
use std::sync::atomic::{AtomicI32, AtomicU64};

thread_local! {
    /// いま decode 1 トークンの鎖の中にいる深さ (ガードの入れ子を数える)。
    /// 発行は decode を回しているスレッドのものなので、スレッドローカルで数える。
    static DECODE_DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// `DENNO_CUDA_PDL`: このモジュールを有効にする env。
pub(crate) const PDL_ENV: &str = "DENNO_CUDA_PDL";

/// `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION` (CUDA 12.x)。
pub(crate) const CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION: c_int = 6;

/// PDL を付けてよい最小の SM 番号。Hopper = 90、Blackwell の RTX 5090 = 120。
///
/// driver の `cuDeviceComputeCapability` は (major, minor) を返すので、
/// **必ず [`sm_number`] で 1 つの番号に直してから比べる** (5090 は (12, 0) で
/// major だけ見ると 12 < 90 になり、属性が一度も付かない。lab t1377b で踏んだ)。
pub(crate) const MIN_PDL_SM: i32 = 90;

/// (major, minor) を SM 番号にする。`(12, 0)` -> 120、`(9, 0)` -> 90。
pub(crate) fn sm_number(major: i32, minor: i32) -> i32 {
    major.saturating_mul(10).saturating_add(minor)
}

/// `CUlaunchAttributeValue` の実体は 64 バイトの union。driver は
/// `programmaticStreamSerializationAllowed` (先頭 4 バイト) だけを読む。
/// ヘッダと同じ 64 バイトに固定する (テストで `size_of` を縛る)。
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) union CUlaunchAttributeValue {
    pub programmatic_stream_serialization_allowed: c_int,
    /// ヘッダの union と同じ 64 バイトにするための詰め物。
    pub _pad: [u64; 8],
}

/// `CUlaunchAttribute` (CUDA driver ヘッダと同じ並び・同じサイズ)。
///
/// ヘッダでは `id` の後ろに `64 - sizeof(id)` の詰め物があり、`value` は
/// オフセット 64 から始まる (全体 128 バイト)。
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CUlaunchAttribute {
    pub id: c_int,
    pub pad: [u8; 60],
    pub value: CUlaunchAttributeValue,
}

/// `CUlaunchConfig` (CUDA driver ヘッダと同じ並び)。
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CUlaunchConfig {
    pub grid_dim_x: u32,
    pub grid_dim_y: u32,
    pub grid_dim_z: u32,
    pub block_dim_x: u32,
    pub block_dim_y: u32,
    pub block_dim_z: u32,
    pub shared_mem_bytes: u32,
    pub h_stream: CUstream,
    pub attrs: *mut CUlaunchAttribute,
    pub num_attrs: u32,
}

/// `cuLaunchKernelEx(const CUlaunchConfig*, CUfunction, void**, void**)` の型は
/// `cuda::FnCuLaunchKernelEx` (mod.rs) に置く。

static ENABLED: OnceLock<bool> = OnceLock::new();
/// デバイスの compute capability major (未設定 = 0)。
static DEVICE_SM: AtomicI32 = AtomicI32::new(0);
/// 属性付きで発行できた回数。
static PDL_LAUNCHES: AtomicU64 = AtomicU64::new(0);
/// 属性付き起動が失敗して従来経路でやり直した回数。
static PDL_FALLBACKS: AtomicU64 = AtomicU64::new(0);
/// 直近の失敗コード (fallback の理由。成功したら 0 に戻さない)。
static LAST_ERR: AtomicI32 = AtomicI32::new(0);
/// SM が足りない / decode の鎖の外だったので属性を付けなかった回数。
static PDL_SKIPPED: AtomicU64 = AtomicU64::new(0);

/// `DENNO_CUDA_PDL` の真偽表 (純関数)。1/on/true/yes だけが有効。
/// 未設定・空・0/off/false/no・その他の値はすべて off。
pub(crate) fn parse_enabled(raw: Option<&str>) -> bool {
    matches!(
        raw.map(|v| v.trim().to_ascii_lowercase()).as_deref(),
        Some("1") | Some("on") | Some("true") | Some("yes")
    )
}

/// ゲートが on か (プロセスで 1 回だけ env を読む)。
pub(crate) fn enabled() -> bool {
    *ENABLED.get_or_init(|| parse_enabled(std::env::var(PDL_ENV).ok().as_deref()))
}

/// `CudaCtx::new` がデバイスの compute capability を 1 回だけ置く
/// (引数は [`sm_number`] で直した SM 番号)。
pub(crate) fn set_device_sm(sm: i32) {
    DEVICE_SM.store(sm, Ordering::Relaxed);
}

/// このデバイスで PDL 属性を付けてよいか (純関数)。
pub(crate) fn sm_supports(major: i32) -> bool {
    major >= MIN_PDL_SM
}

/// この発行に PDL 属性を付けるか。`decode` は [`decode_window`] の深さ。
///
/// - ゲート off / SM 不足は付けない。
/// - decode の鎖の外 (prefill など) は付けない。
/// - null ストリーム以外は付けない (decode の鎖は null ストリームに積まれる。
///   背景の上げ直しや side stream は前後関係が違うので触らない)。
pub(crate) fn eligible_with(gate: bool, major: i32, decode: bool, stream_is_null: bool) -> bool {
    gate && sm_supports(major) && decode && stream_is_null
}

/// [`eligible_with`] の実環境版。ホットパスでは分岐 3 回だけ。
pub(crate) fn eligible(stream: CUstream) -> bool {
    if !enabled() {
        return false;
    }
    let ok = eligible_with(
        true,
        DEVICE_SM.load(Ordering::Relaxed),
        in_decode_window(),
        stream.is_null(),
    );
    if !ok {
        PDL_SKIPPED.fetch_add(1, Ordering::Relaxed);
    }
    ok
}

/// decode 1 トークンの鎖のガード。`on == false` では何もしない。
///
/// drop で深さを 1 戻すので、早期 return や `?` でも窓が漏れない。
pub(crate) struct DecodeWindow(bool);

impl Drop for DecodeWindow {
    fn drop(&mut self) {
        if self.0 {
            DECODE_DEPTH.with(|c| c.set(c.get().saturating_sub(1)));
        }
    }
}

/// decode の鎖に入る (`nt == 1` のときだけ `on`)。ガードを保持している間、
/// null ストリームの発行に PDL 属性が付く。
pub(crate) fn decode_window(on: bool) -> DecodeWindow {
    if on {
        DECODE_DEPTH.with(|c| c.set(c.get() + 1));
    }
    DecodeWindow(on)
}

/// いま decode の鎖の中か (テスト・診断用)。
pub(crate) fn in_decode_window() -> bool {
    DECODE_DEPTH.with(|c| c.get() > 0)
}

/// いまの深さ (テスト用)。
#[cfg(test)]
pub(crate) fn decode_depth() -> usize {
    DECODE_DEPTH.with(|c| c.get())
}

/// 属性 1 個 (PDL のみ) を作る。
pub(crate) fn programmatic_attr() -> CUlaunchAttribute {
    CUlaunchAttribute {
        id: CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
        pad: [0; 60],
        value: CUlaunchAttributeValue {
            programmatic_stream_serialization_allowed: 1,
        },
    }
}

/// 属性 1 個を持つ起動 config を組む。`attr` は呼び出し側のローカルで、
/// config より長生きする必要がある (`attrs` がそのアドレスを指す)。
pub(crate) fn launch_config(
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shmem: u32,
    stream: CUstream,
    attr: &mut CUlaunchAttribute,
) -> CUlaunchConfig {
    *attr = programmatic_attr();
    CUlaunchConfig {
        grid_dim_x: grid.0,
        grid_dim_y: grid.1,
        grid_dim_z: grid.2,
        block_dim_x: block.0,
        block_dim_y: block.1,
        block_dim_z: block.2,
        shared_mem_bytes: shmem,
        h_stream: stream,
        attrs: attr as *mut CUlaunchAttribute,
        num_attrs: 1,
    }
}

/// 属性付きで発行できた。
pub(crate) fn note_launch() {
    PDL_LAUNCHES.fetch_add(1, Ordering::Relaxed);
}

/// 属性付き起動が `err` で失敗した (従来経路でやり直す)。
pub(crate) fn note_fallback(err: CUresult) {
    PDL_FALLBACKS.fetch_add(1, Ordering::Relaxed);
    LAST_ERR.store(err, Ordering::Relaxed);
}

/// 積算の行 (`DENNO_CUDA_PDL` が on で、1 回でも判定したときだけ)。
/// 発火の証拠: `launches` が 0 なら属性は 1 度も付いていない。
pub fn stats_line() -> Option<String> {
    stats_line_with(enabled())
}

/// [`stats_line`] の本体 (ゲートの真偽を引数で受ける = テストで両側を踏める)。
pub(crate) fn stats_line_with(gate: bool) -> Option<String> {
    if !gate {
        return None;
    }
    let l = PDL_LAUNCHES.load(Ordering::Relaxed);
    let f = PDL_FALLBACKS.load(Ordering::Relaxed);
    let s = PDL_SKIPPED.load(Ordering::Relaxed);
    if l == 0 && f == 0 && s == 0 {
        return None;
    }
    Some(format!(
        "[cuda-pdl] launches={l} fallback={f} skipped={s} last_err={} sm={}",
        LAST_ERR.load(Ordering::Relaxed),
        DEVICE_SM.load(Ordering::Relaxed)
    ))
}

/// 積算を 0 に戻す (リクエスト単位で読む)。
pub fn reset() {
    PDL_LAUNCHES.store(0, Ordering::Relaxed);
    PDL_FALLBACKS.store(0, Ordering::Relaxed);
    PDL_SKIPPED.store(0, Ordering::Relaxed);
    LAST_ERR.store(0, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_truth_table() {
        for v in ["1", "on", "ON", " true ", "yes", "Yes"] {
            assert!(parse_enabled(Some(v)), "{v}");
        }
        for v in ["", "0", "off", "false", "no", "2", "auto"] {
            assert!(!parse_enabled(Some(v)), "{v}");
        }
        assert!(!parse_enabled(None));
    }

    #[test]
    fn attr_abi_layout() {
        // CUDA driver ヘッダの ABI: value は 64 バイトでオフセット 64。
        assert_eq!(std::mem::size_of::<CUlaunchAttributeValue>(), 64);
        assert_eq!(std::mem::size_of::<CUlaunchAttribute>(), 128);
        assert_eq!(
            std::mem::offset_of!(CUlaunchAttribute, value),
            64,
            "value は id の 64 バイト後ろ"
        );
        // CUlaunchConfig: hStream の前に 4 バイトの詰め物、全体 56 バイト。
        assert_eq!(std::mem::size_of::<CUlaunchConfig>(), 56);
        assert_eq!(std::mem::offset_of!(CUlaunchConfig, h_stream), 32);
        assert_eq!(std::mem::offset_of!(CUlaunchConfig, attrs), 40);
        assert_eq!(std::mem::offset_of!(CUlaunchConfig, num_attrs), 48);
    }

    #[test]
    fn attr_contents() {
        let a = programmatic_attr();
        assert_eq!(a.id, CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION);
        assert_eq!(a.id, 6);
        assert_eq!(a.pad, [0u8; 60]);
        unsafe {
            assert_eq!(a.value.programmatic_stream_serialization_allowed, 1);
        }
    }

    #[test]
    fn config_carries_one_attr_and_geometry() {
        let mut a = programmatic_attr();
        let stream = 0x1234usize as CUstream;
        let c = launch_config((7, 1, 1), (128, 1, 1), 4096, stream, &mut a);
        assert_eq!((c.grid_dim_x, c.grid_dim_y, c.grid_dim_z), (7, 1, 1));
        assert_eq!((c.block_dim_x, c.block_dim_y, c.block_dim_z), (128, 1, 1));
        assert_eq!(c.shared_mem_bytes, 4096);
        assert_eq!(c.h_stream, stream);
        assert_eq!(c.num_attrs, 1);
        assert_eq!(c.attrs, &mut a as *mut CUlaunchAttribute);
    }

    #[test]
    fn eligible_gate_truth_table() {
        // gate off / sm 80 / decode 外 / side stream はどれも付けない
        assert!(eligible_with(true, 120, true, true));
        assert!(eligible_with(true, 90, true, true));
        assert!(!eligible_with(false, 120, true, true));
        assert!(!eligible_with(true, 89, true, true));
        assert!(!eligible_with(true, 80, true, true));
        assert!(!eligible_with(true, 120, false, true));
        assert!(!eligible_with(true, 120, true, false));
        assert!(sm_supports(90));
        assert!(sm_supports(120));
        assert!(!sm_supports(89));
        // (major, minor) の生値ではなく SM 番号で比べること (lab t1377b の罠)
        assert_eq!(sm_number(12, 0), 120);
        assert_eq!(sm_number(9, 0), 90);
        assert!(sm_supports(sm_number(12, 0)));
        assert!(!sm_supports(12), "major だけを見ると 5090 を取りこぼす");
    }

    #[test]
    fn decode_window_nests_and_closes_on_drop() {
        let before = decode_depth();
        {
            let _w = decode_window(true);
            assert!(in_decode_window());
            {
                let _w2 = decode_window(true);
                assert_eq!(decode_depth(), before + 2);
            }
            assert_eq!(decode_depth(), before + 1);
        }
        assert_eq!(decode_depth(), before);
        // off のガードは深さを動かさない
        let _w = decode_window(false);
        assert_eq!(decode_depth(), before);
    }

    #[test]
    fn stats_line_only_when_gate_and_activity() {
        // ゲート off は常に None
        assert!(stats_line_with(false).is_none());
        // ゲート on でも 1 回も判定していなければ None
        reset();
        assert!(stats_line_with(true).is_none());
        note_launch();
        let line = stats_line_with(true).expect("1 回発行した");
        assert!(line.starts_with("[cuda-pdl] launches=1 fallback=0"));
        assert!(line.contains("skipped=0"));
        assert!(line.contains("last_err=0"));
        // 失敗は fallback と last_err に残る
        note_fallback(1);
        let line = stats_line_with(true).expect("失敗も数える");
        assert!(line.contains("fallback=1"));
        assert!(line.contains("last_err=1"));
        assert!(!line.contains('\n'), "1 行であること");
        reset();
        assert!(stats_line_with(true).is_none());
    }
}
