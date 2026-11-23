//! Hyper-Connections mix (decode / prefill)
//!
//! task 664: `src/cuda.rs` (70k 行) からの純粋な移動。ロジック・数値・カーネル・
//! PTX・env ゲートは一切変えていない。

#![allow(non_snake_case, non_camel_case_types)]

use super::*;


/// task 230: `DENNO_PF_MIXKEEP` の解釈 (純関数)。**task 317 で既定 on**。
///
/// on にすると、full attention 層の attn mix は `mixed` を D2H せず、直後の
/// qkv がデバイスの `mixed` を直接読む (H2D 1 本と D2H 1 本を消す)。
///
/// task 230 の t230b (14,018 cold prefill) では mix 段 1,103.9 → 1,069.8 ms
/// (-34 ms、D2H 13,780 → 12,058 MB、full の H2D 480 → 360 本 = -1.72 GB) と
/// 小さく、全体はノイズ (b0 27.40 s → 28.54 s) だったので既定 off だった。
/// task 317 (t317a) では ssm 層のぶん (`DENNO_PF_MIXKEEP_SSM`) と wy4 を
/// 重ねた条件で 3,826 cold が 2.945 → 2.78 s (-5.4%)、full の H2D は
/// 481.94 → 11.80 MB。値は変えない (md5 SAME)。
pub(crate) fn pf_mix_keep_from_value(v: Option<&str>) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        Some(s) => matches!(s.as_str(), "1" | "on" | "true" | "yes"),
        None => true,
    }
}


#[cfg(test)]
static PF_MIX_KEEP_FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);


/// テスト用: ゲートを強制する (0 = env に戻す)。
#[cfg(test)]
pub(crate) fn pf_mix_keep_set_for_test(v: u8) {
    PF_MIX_KEEP_FORCE.store(v, Ordering::Relaxed);
}


/// task 230: full 層の attn mix を D2H しないか (`DENNO_PF_MIXKEEP`、**task 317 で既定 on**)。
pub(crate) fn pf_mix_keep_on() -> bool {
    #[cfg(test)]
    {
        match PF_MIX_KEEP_FORCE.load(Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| pf_mix_keep_from_value(std::env::var("DENNO_PF_MIXKEEP").ok().as_deref()))
}


/// task 317: `DENNO_PF_MIXKEEP_SSM` の解釈 (純関数)。**既定 on**。
///
/// on にすると、ssm 層の attn mix も `mixed` を D2H しない。直後の GDN 連鎖
/// (`gdn_pf_chain`) がデバイスの `mixed` をそのまま GEMM の入力に読み、
/// `b.x` への H2D も 1 本消える (値は同じバイト列)。
///
/// 連鎖が前提外でホスト経路へ落ちる場合 (型・形・ゲート) は、
/// `SsmLayer::forward_batch_flat` が [`pf_mixed_restore_host`] でホストへ
/// 埋め直してから計算する (古いバイト列を読まない)。
///
/// lab t317a (d59f6df54、depth8k): keep (MIXKEEP + これ) で mix 段の D2H が
/// 1,880.56 → 0.00 MB、ssm の H2D が 1,575.34 → 164.92 MB、full の H2D が
/// 481.94 → 11.80 MB。3,826 cold は base 2.94 s のまま (keep 単独 2.91 s)。
/// md5 は全条件 SAME (値は変えない)。
pub(crate) fn pf_mix_keep_ssm_from_value(v: Option<&str>) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        Some(s) => matches!(s.as_str(), "1" | "on" | "true" | "yes"),
        None => true,
    }
}


#[cfg(test)]
static PF_MIX_KEEP_SSM_FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);


/// テスト用: ssm 側のゲートを強制する (0 = env に戻す)。
#[cfg(test)]
pub(crate) fn pf_mix_keep_ssm_set_for_test(v: u8) {
    PF_MIX_KEEP_SSM_FORCE.store(v, Ordering::Relaxed);
}


/// task 317: ssm 層の attn mix を D2H しないか (`DENNO_PF_MIXKEEP_SSM`、**既定 on**)。
pub(crate) fn pf_mix_keep_ssm_on() -> bool {
    #[cfg(test)]
    {
        match PF_MIX_KEEP_SSM_FORCE.load(Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        pf_mix_keep_ssm_from_value(std::env::var("DENNO_PF_MIXKEEP_SSM").ok().as_deref())
    })
}


thread_local! {
    /// task 209: 「このホスト側ポインタの `mixed` はデバイスの `mixed` と同一」の記録。
    pub(crate) static PF_MIXED_CUR: std::cell::Cell<(*const f32, usize, usize)> =
        const { std::cell::Cell::new((std::ptr::null(), 0, 0)) };
    /// task 230: 直前の mix を D2H していない (ホスト側のバイト列は古い)。
    /// デバイスの `mixed` だけが正で、D2H する前にホスト経路へ落ちたら
    /// [`pf_mixed_restore_host`] で埋め直す。
    pub(crate) static PF_MIXED_DEV_ONLY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}


/// ホスト側の `mixed` がデバイスの `mixed` と同じ内容であることを記録する
/// (`pf_take_mixed` の直後に呼ぶ)。
pub fn pf_mixed_mark(host: *const f32, nt: usize, e: usize) {
    PF_MIXED_CUR.with(|c| c.set((host, nt, e)));
    PF_MIXED_DEV_ONLY.with(|c| c.set(false));
}


/// task 230: ホストへ降ろさずに「デバイスの `mixed` だけが正」と記録する。
///
/// 次の GEMM がデバイスから直接読める層 (full attention の qkv) だけで使う。
/// ホストのバイト列は古いままなので、ホスト経路に落ちるときは
/// [`pf_mixed_restore_host`] を必ず通すこと。
pub fn pf_mixed_keep_dev(host: *const f32, nt: usize, e: usize) {
    PF_MIXED_CUR.with(|c| c.set((host, nt, e)));
    PF_MIXED_DEV_ONLY.with(|c| c.set(true));
}


/// 記録を消す (非常駐経路へ落ちたとき・リクエストの終わり)。
pub fn pf_mixed_mark_clear() {
    PF_MIXED_CUR.with(|c| c.set((std::ptr::null(), 0, 0)));
    PF_MIXED_DEV_ONLY.with(|c| c.set(false));
}


/// このホスト配列が「D2H していない (デバイスだけが正)」か。
pub(crate) fn pf_mixed_dev_only(host: *const f32, nt: usize, e: usize) -> bool {
    let m = PF_MIXED_CUR.with(|c| c.get());
    pf_mixed_mark_matches(m, host, nt, e) && PF_MIXED_DEV_ONLY.with(|c| c.get())
}


/// D2H していない `mixed` をホストへ降ろし、記録を「同一」に戻す。
///
/// ホスト経路 (非常駐の qkv / ssM) へ落ちるときだけ呼ぶ。降ろせなければ false
/// (そのときは呼び出し側が失敗する = 古いバイト列で計算しない)。
pub(crate) fn pf_mixed_restore_host(host: *mut f32, nt: usize, e: usize) -> bool {
    if host.is_null() || nt == 0 || e == 0 {
        return false;
    }
    if !pf_mixed_dev_only(host as *const f32, nt, e) {
        return false;
    }
    let ok = PF_DEV.with(|c| {
        let d = c.borrow();
        match d.mixed.as_ref() {
            Some(b) if b.bytes() >= nt * e * 4 => unsafe {
                b.to_slice(std::slice::from_raw_parts_mut(host, nt * e)).is_ok()
            },
            _ => false,
        }
    });
    if ok {
        PF_MIXED_DEV_ONLY.with(|c| c.set(false));
    }
    ok
}


/// 記録が (host, nt, e) と一致するか (純関数・テスト用)。
pub(crate) fn pf_mixed_mark_matches(
    mark: (*const f32, usize, usize),
    host: *const f32,
    nt: usize,
    e: usize,
) -> bool {
    !mark.0.is_null() && std::ptr::eq(mark.0, host) && mark.1 == nt && mark.2 == e
}


/// この `xs` をデバイスの `mixed` から読んでよいなら、そのデバイスポインタ。
///
/// ゲート off・記録なし・ポインタ不一致・作業域の形違いはすべて `None`
/// (呼び出し側は従来どおりホストから上げる)。
pub(crate) fn pf_mixed_dev_in_ptr(host: *const f32, nt: usize, e: usize) -> Option<CUdeviceptr> {
    if !pf_dev_in_on() || nt == 0 || e == 0 {
        return None;
    }
    let m = PF_MIXED_CUR.with(|c| c.get());
    if !pf_mixed_mark_matches(m, host, nt, e) {
        return None;
    }
    PF_DEV.with(|c| {
        let d = c.borrow();
        let b = d.mixed.as_ref()?;
        (d.e == e && nt <= d.nt && b.bytes() >= nt * e * 4).then(|| b.ptr())
    })
}


/// `DENNO_PF_MIX_TC` の値の解釈 (純関数)。明示的な偽値だけ off。
///
/// **既定 on** (lab t222b で採用): 14,018 の mix の gpu が 1,061 → 792 ms、
/// 3,826 で 285 → 214 ms、cold prefill 14,018 が -0.9%、decode と next turn は
/// 変わらず、errors 0・text sane。md5 は BF16 丸めで変わる。
pub(crate) fn pf_mix_tc_from_value(v: Option<&str>) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        Some(s) => matches!(s.as_str(), "1" | "on" | "true" | "yes"),
        None => true,
    }
}


/// prefill の hc mix (BF16 の down / up) を sn3 のテンソルコア GEMM で回すか
/// (**既定 on**。`0` で切る)。
pub(crate) fn pf_mix_tc_on() -> bool {
    #[cfg(test)]
    {
        match PF_MIX_TC_FORCE.load(Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| pf_mix_tc_from_value(std::env::var("DENNO_PF_MIX_TC").ok().as_deref()))
}


/// `DENNO_PF_MIX_Q8TC` の値の解釈 (純関数)。未設定は既定 on、明示的な真値だけ on。
///
/// **既定 on** (lab t301a で採用)。UD-Q4_K_XL の hc mix は **Q8_0** の
/// down / up で、既定では従来の量子 GEMM (1 起動 16 行 x 128 トークン、
/// 16 スレッド) を通る。同じ sn3 の Q8_0 入口 (bf16 mma、k 昇順・FP32
/// アキュムレート) に載せ替えると、14,018 の mix が 31.06 -> 1.11 s
/// (-96.4%)、cold prefill が 143.45 -> 115.26 s (-19.7%) になった。
pub(crate) fn pf_mix_q8_tc_from_value(v: Option<&str>) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        Some(s) => matches!(s.as_str(), "1" | "on" | "true" | "yes"),
        None => true,
    }
}


/// prefill の hc mix のうち **Q8_0 の重み** (UD-Q4_K_XL) を sn3 で回すか
/// (**既定 on**。`DENNO_PF_MIX_Q8TC=0` で切る)。
pub(crate) fn pf_mix_q8_tc_on() -> bool {
    #[cfg(test)]
    {
        match PF_MIX_Q8TC_FORCE.load(Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| pf_mix_q8_tc_from_value(std::env::var("DENNO_PF_MIX_Q8TC").ok().as_deref()))
}

#[cfg(test)]
static PF_MIX_TC_FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

#[cfg(test)]
static PF_MIX_Q8TC_FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);


#[cfg(test)]
pub(crate) fn set_pf_mix_q8_tc_for_test(v: Option<bool>) {
    PF_MIX_Q8TC_FORCE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        Ordering::Relaxed,
    );
}


#[cfg(test)]
pub(crate) fn set_pf_mix_tc_for_test(v: Option<bool>) {
    PF_MIX_TC_FORCE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        Ordering::Relaxed,
    );
}


/// task 307: `DENNO_ROUTER_IN_MIX` の解釈 (純関数)。**既定 on** (task 307 で採否。
/// lab t307b で router 行 1.15 → 0.18 ms/token、decode -2.0%、md5 SAME)。
///
/// FFN mix の鎖 (down → up → hc_mixed) の末尾にルータ GEMV を積み、D2H と同期を
/// 鎖の 1 回にまとめる。明示的な偽値だけ off。
pub(crate) fn parse_router_in_mix(raw: Option<&str>) -> bool {
    !matches!(
        raw.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("0") | Some("off") | Some("false") | Some("no")
    )
}


/// task 307: ルータ GEMV を mix の鎖に載せるか (env から 1 回だけ読む)。
///
/// 動機 (lab t299b の `[q4ledger]`): router 行は 1.08 ms/token で、内訳は
/// gpu 0.64 / sync 0.63 / cpu 0.39。mix の鎖は同じ層で既に 1 回同期している
/// (mix 行の sync) ので、ルータ GEMV をその鎖の末尾に積めば、起動の往復と
/// 同期 1 回 (層あたり 5〜6 us) が消える。入力は鎖の中間バッファ `dm`
/// (= `mixed` のデバイスコピー) をそのまま読むので H2D も消える。
pub fn router_in_mix_on() -> bool {
    #[cfg(test)]
    if let Some(v) = ROUTER_IN_MIX_FORCE.with(|c| c.get()) {
        return v;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse_router_in_mix(std::env::var("DENNO_ROUTER_IN_MIX").ok().as_deref()))
}


/// task 364: テスト用のゲート強制 (None = env の解釈)。
#[cfg(test)]
thread_local! {
    pub(crate) static ROUTER_IN_MIX_FORCE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}


/// task 364: `DENNO_ROUTER_IN_MIX` をテストから固定する。
#[cfg(test)]
pub(crate) fn router_in_mix_set_for_test(v: Option<bool>) {
    ROUTER_IN_MIX_FORCE.with(|c| c.set(v));
}


/// task 364: [`ROUTER_IN_QCHAIN_HITS`] を読む。
pub(crate) fn router_in_mix_qchain_hits() -> u64 {
    ROUTER_IN_QCHAIN_HITS.load(std::sync::atomic::Ordering::Relaxed)
}


/// task 364: 計数を 0 に戻す (リクエスト終端)。
pub(crate) fn router_in_mix_qchain_reset() {
    ROUTER_IN_QCHAIN_HITS.store(0, std::sync::atomic::Ordering::Relaxed);
}


/// task 307: この層の FFN mix の鎖にルータ GEMV を載せる (decode の層入口で呼ぶ)。
///
/// `false` = 載せない (ゲート off・形が合わない)。載らなかった場合は従来どおり
/// `maybe_gpu_router_gemv_begin` が H2D + 起動 + D2H を行う (数値は同じ).
pub fn router_in_mix_arm(uid: u64, bytes: &[u8], k: usize, n: usize) -> bool {
    if !router_in_mix_on() {
        return false;
    }
    if k == 0 || n == 0 || bytes.len() != k.saturating_mul(n).saturating_mul(4) {
        return false;
    }
    ROUTER_ARM.with(|c| {
        *c.borrow_mut() = Some(RouterArm {
            uid,
            ptr: bytes.as_ptr(),
            len: bytes.len(),
            k,
            n,
            layer: dec_kprof::cur_layer(),
            pos: dec_kprof::cur_pos(),
        })
    });
    true
}


/// task 307: トークンの入口でルータの控えを落とす (層 0 の入口で呼ぶ)。
///
/// 前のトークンの `ROUTER_FILLED` / 控えが残っていても、次のトークンでは
/// 必ず「mix の鎖が作った値」だけを使う。
pub fn router_in_mix_token_start() {
    if !router_in_mix_on() {
        return;
    }
    ROUTER_FILLED.with(|c| c.set((0, 0, 0)));
    ROUTER_SYNCED.with(|c| c.set(false));
    ROUTER_ARM.with(|c| {
        let _ = c.borrow_mut().take();
    });
}


/// task 364: 控え (`ROUTER_ARM`) がこの形・この層で消費できる状態か (**取らない**)。
///
/// [`router_in_mix_enqueue`] と同じ条件を読むだけの判定。qchain の入口で
/// 「載せにいく」回数を数えるのと、診断に使う。
#[cfg(test)]
pub(crate) fn router_in_mix_armed_for(k: usize) -> bool {
    ROUTER_ARM.with(|c| {
        c.borrow().as_ref().is_some_and(|job| {
            job.k == k && job.layer == dec_kprof::cur_layer() && job.pos == dec_kprof::cur_pos()
        })
    })
}


/// task 307: mix の鎖の末尾にルータ GEMV を 1 発積む (鎖の中から呼ぶ)。
///
/// 戻り値は積んだ出力行数 (0 = 積んでいない)。`x` は鎖の中間バッファ
/// (= `mixed` のデバイスコピー) で、`k` と一致するときだけ積む。
fn router_in_mix_enqueue(
    ctx: &CudaCtx,
    x_dev: CUdeviceptr,
    k: usize,
    w_keep: &mut Option<GgufWRef>,
) -> usize {
    let Some(job) = ROUTER_ARM.with(|c| c.borrow_mut().take()) else {
        return 0;
    };
    if job.k != k || job.layer != dec_kprof::cur_layer() || job.pos != dec_kprof::cur_pos() {
        return 0;
    }
    let bytes = unsafe { std::slice::from_raw_parts(job.ptr, job.len) };
    let w = match weights_for_f32(job.uid, bytes, job.k, job.n) {
        Ok(w) => w,
        Err(_) => return 0,
    };
    let f_seq = matvec_f32_fn(ctx);
    let f_red = matvec_f32_r_fn(ctx);
    let use_reduce = router_gpu_reduce_enabled() && router_use_reduce_kernel(job.k, job.n, f_red.is_some());
    let Some(f) = (if use_reduce { f_red } else { f_seq }) else {
        return 0;
    };
    let thr: u32 = if use_reduce { 256 } else { 1 };
    let Ok(l) = lib() else {
        return 0;
    };
    let r = ROUTER_SC.with(|cell| -> Result<()> {
        let mut s = cell.borrow_mut();
        let RouterScratch { out: dout, ho, .. } = &mut *s;
        Scratch::ensure(dout, job.n * 4)?;
        Scratch::ensure_pin(ho, job.n * 4)?;
        let dout = dout.as_ref().expect("ensure 済み");
        let mut po = dout.ptr();
        let mut pw = w.buf.ptr;
        let mut ph = x_dev;
        let (mut kc, mut nc) = (job.k as c_int, job.n as c_int);
        let mut args = [
            &mut ph as *mut _ as *mut c_void,
            &mut pw as *mut _ as *mut c_void,
            &mut po as *mut _ as *mut c_void,
            &mut kc as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
        ];
        unsafe {
            ctx.launch_async(f, (job.n as u32, 1, 1), (thr, 1, 1), 0, &mut args)?;
            let ho = ho.as_ref().expect("ensure 済み");
            check(
                (l.memcpy_dtoh_async)(
                    ho.ptr,
                    dout.ptr(),
                    job.n * 4,
                    std::ptr::null_mut(),
                ),
                "cuMemcpyDtoHAsync (router in mix)",
            )?;
        }
        crate::prof::mark_d2h_src(crate::prof::D2H_SRC_LINEAR);
        Ok(())
    });
    match r {
        Ok(()) => {
            *w_keep = Some(w);
            ROUTER_FILLED.with(|c| c.set((job.uid, job.k, job.n)));
            job.n
        }
        Err(_) => 0,
    }
}


// ===========================================================================
// task 110: hc mix (hc_down / hc_up) の BF16 GEMV を GPU へ (decode 96 回/token)
// ===========================================================================
// 実測 (メインセッション 2026-11-02、swift15 IQ3_S、DENNO_MIX_KPROF=1): decode
// mix 47.1 ms/tok の 91% が hc_down / hc_up の BF16 GEMV。down = [10240->320] /
// up = [320->10240]、48 層 x 2 = 96 回/token、重み 630 MB/token を CPU が読む
// 帯域律速 (316 MMAC/token で 42.9 ms = 7.4 GMAC/s)。
//
// カーネルは新規に作らず、s49 / task 69 と同じ `dno_matvec_f32` を流用する
// (.cu / PTX は 1 バイトも変えない。BF16 を直接読むカーネルは現行 PTX に無い):
//   - 1 行 = 1 ブロック (grid=(n,1,1)・block=(1,1,1))・k 昇順の直列加算。
//     PTX は `mul.rn.f32` + `add.rn.f32` のみ (fma.rn 0 個・.ftz 0 個)。
//   - CPU 参照は `kquant::dequant_row_into` (BF16 -> f32 は上位 16bit 複製 =
//     情報落ちなし) + 1 行の `acc += row[i] * x[i]` (i 昇順・積と加算の丸め分離)。
//     同じ演算列なので**ビット一致**する (task 92 の AVX2 融合カーネルも
//     「積を i 昇順で逐次加算・FMA 無し」で classic とビット一致済み)。
//   - BF16 のまま VRAM に載せられない (カーネルは f32 を読む) ため f32 に展開
//     して常駐する (630 MB BF16 -> 1,260 MB f32。BF16 -> f32 は情報落ちなしで
//     値は不変)。展開は初回の GPU 呼び出しで 1 回だけ。
//   - 常駐は s49 (LMV32_W) と同じ**専用レジストリ** (eviction 無し・uid key)。
//     gguf_registry (DENNO_GPU_WEIGHT_MB) にも expert LRU (DENNO_EXPERT_VRAM_MB)
//     にも乗らないので、ルータ / lm_head / expert と枠を奪い合わない。
//   - 失敗時 (VRAM 足りない等) は CPU へフォールバックして warning を 1 行だけ
//     出す (MIX_GPU_BROKEN ラッチ。2 回目以降は warn 無しで静かに CPU に落ちる)。
//     cuda_enabled() が false (GPU 無し・DENNO_CUDA 未設定) は正常フォール
//     バックなので warn を出さない。

/// task 110: hc mix BF16 経路のスコープド恒久ラッチ (s49 / task 69 とは独立。
/// 本経路の失敗で他の GPU 経路を落とさない)。
static MIX_GPU_BROKEN: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);


/// task 110: hc mix BF16 経路の入口に入った回数 (テスト専用)。
/// 「ゲートを足したが分岐が通っていない」を GPU 無しで検出するために、
/// `cuda_enabled()` の判定より**前**で数える (task 69 の ROUTER_GPU_ATTEMPTS
/// と同じ流儀)。
#[cfg(test)]
static MIX_GPU_ATTEMPTS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);


/// task 110: [`MIX_GPU_ATTEMPTS`] を読む (テスト専用)。
#[cfg(test)]
pub(crate) fn mix_gpu_attempts() -> usize {
    MIX_GPU_ATTEMPTS.load(std::sync::atomic::Ordering::Relaxed)
}


/// task 110: hc mix BF16 経路の失敗 warning (1 行だけ。ラッチは swap で 1 回保証)。
fn mix_gpu_broken_once(msg: &str) {
    let prev = MIX_GPU_BROKEN.swap(true, std::sync::atomic::Ordering::Relaxed);
    if !prev {
        tracing::warn!("CUDA hc mix BF16 GPU disabled, using CPU: {msg}");
    }
}


/// task 110: BF16 重みを f32 に展開する (行ごとに [`crate::kquant::dequant_row_into`]
/// を通す = BF16 -> f32 変換の定義を 1 か所に保つ。BF16 は f32 の上位 16bit なので
/// 情報落ちは無い)。
pub(crate) fn mix_expand_bf16(bytes: &[u8], k: usize, n: usize) -> Vec<f32> {
    let rb = k * 2;
    let mut out = vec![0f32; k * n];
    let mut row = vec![0f32; k];
    for r in 0..n {
        crate::kquant::dequant_row_into(
            crate::kquant::GgmlType::Bf16,
            &bytes[r * rb..(r + 1) * rb],
            &mut row,
        );
        out[r * k..(r + 1) * k].copy_from_slice(&row);
    }
    out
}


/// task 110: 展開済み f32 重みの常駐エントリ (s49 の LMV32_W と同じ方針:
/// eviction 無し・uid key。解放は [`mix_gpu_unregister`] が `GgufLinear::drop` から)。
///
/// task 152: `bf16 == true` のときは `buf` が **BF16 の生バイト** (k*n*2 B) で、
/// カーネルは `dno_matvec_bf16*` / `dno_gemm_bf16_mt` を使う。f32 展開しないので
/// 常駐が半分になる (lab t149c の w:mix_f32 2,425 MiB のうち約 1,200 MiB が戻る)。
pub(crate) struct MixGpuW {
    buf: DevBuffer,
    k: usize,
    n: usize,
    bf16: bool,
}


/// task 152: 常駐重みのポインタ (モードは呼び出し側が先に決めている)。
#[derive(Clone, Copy)]
pub(crate) struct MixW {
    pub(crate) ptr: CUdeviceptr,
}


/// task 152: `DENNO_MIX_BF16` の解釈 (純関数)。明示的な真値だけ on。
/// 既定は [`MIX_BF16_DEFAULT`] (lab の採否で決める)。
pub(crate) fn mix_bf16_from_value(v: Option<&str>, default_on: bool) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        Some(s) if matches!(s.as_str(), "1" | "true" | "on" | "yes") => true,
        Some(s) if matches!(s.as_str(), "0" | "false" | "off" | "no") => false,
        _ => default_on,
    }
}


/// task 152: BF16 のまま読むカーネルを使うか (`DENNO_MIX_BF16`)。
fn mix_bf16_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        mix_bf16_from_value(std::env::var("DENNO_MIX_BF16").ok().as_deref(), MIX_BF16_DEFAULT)
    })
}


static MIX_GPU_W: OnceLock<std::sync::Mutex<std::collections::HashMap<u64, MixGpuW>>> =
    OnceLock::new();


/// task 110: hc mix 用スクラッチ。ルータの RouterScratch と分ける (mix down は
/// x=41KB/out=1.3KB、up は x=1.3KB/out=41KB と交互に来るが grow-only なので
/// 41KB x 2 で固定)。decode は 1 スレッド直列だが thread_local にする理由は同じ。
pub(crate) struct MixGpuScratch {
    x: Option<DevBuffer>,
    /// task 138: チェーン経路の down 出力 (lo_apply 済み) を device に置く場所。
    lo: Option<DevBuffer>,
    out: Option<DevBuffer>,
    /// task 150: chain2 の hc_mixed 出力 ([e]) を device に置く場所。
    mixed: Option<DevBuffer>,
    hx: Option<PinMem>,
    ho: Option<PinMem>,
    /// task 282: 量子 chain の高速 GEMV 用の前置き
    /// (`[wptr_d][wptr_u][seg_d][seg_u][xn]` を 1 本の H2D で送る)。
    stage: Option<DevBuffer>,
    hstage: Option<PinMem>,
    /// task 382: `DENNO_MIX_ZC` 用。device から直接見える pinned。
    /// `xmap` = down の入力 (`xn`、H2D を消す)、`omap` = `mixed` の行き先
    /// (hc_mixed に直接書かせて D2H を消す)。`None` の間は従来経路。
    xmap: Option<MapMem>,
    omap: Option<MapMem>,
    /// task 382: `DENNO_MIX_UPQ8=check` の比較用 (融合版の出力 / その受け皿)。
    chk: Option<DevBuffer>,
    hchk: Option<PinMem>,
    /// task 431: `DENNO_MIX_ZC=stream` の H2D 専用ストリーム + 完了イベント。
    /// 鎖のカーネルは null ストリームのまま (発行順も同期点も変えない)。
    cstream: Option<CopyStream>,
    cev: Option<CopyEvent>,
}


thread_local! {
    pub(crate) static MIX_GPU_SC: std::cell::RefCell<MixGpuScratch> =
        std::cell::RefCell::new(MixGpuScratch {
            x: None,
            lo: None,
            out: None,
            mixed: None,
            hx: None,
            ho: None,
            stage: None,
            hstage: None,
            xmap: None,
            omap: None,
            chk: None,
            hchk: None,
            cstream: None,
            cev: None,
        });
}


/// task 431: H2D 専用の非ブロッキングストリーム + 完了イベントを用意する
/// (`DENNO_MIX_ZC=stream` のときだけ呼ばれる)。作れなければ `None` = 従来経路。
fn mix_h2s_ensure(s: &mut MixGpuScratch) -> Option<(CUstream, CUevent)> {
    if let (Some(st), Some(ev)) = (s.cstream.as_ref(), s.cev.as_ref()) {
        return Some((st.0, ev.0));
    }
    const CU_STREAM_NON_BLOCKING: u32 = 0x01;
    const CU_EVENT_DISABLE_TIMING: u32 = 0x02;
    let l = lib().ok()?;
    let (Some(stream_create), Some(event_create)) = (l.stream_create, l.event_create) else {
        return None;
    };
    let mut st: CUstream = std::ptr::null_mut();
    if unsafe { stream_create(&mut st, CU_STREAM_NON_BLOCKING) } != CUDA_SUCCESS {
        return None;
    }
    let mut ev: CUevent = std::ptr::null_mut();
    if unsafe { event_create(&mut ev, CU_EVENT_DISABLE_TIMING) } != CUDA_SUCCESS {
        unsafe {
            if let Some(f) = l.stream_destroy {
                let _ = f(st);
            }
        }
        return None;
    }
    s.cstream = Some(CopyStream(st));
    s.cev = Some(CopyEvent(ev));
    Some((st, ev))
}


/// task 121: mix GPU で並列リダクション版 (`dno_matvec_f32_r`) に切り替える
/// 「長い行」の閾値 (k >= n × この比)。実測 (swift15-abl IQ3_S・128 tok・main) では
/// down (k=10240, n=320) が 281 us/行列とほぼ直列 (1 ブロック = 1 行で k を 1
/// スレッドが舐めるため並列度が n しかない) で、up (k=320, n=10240) は 55
/// us/行列と n 側に並列度が足りている。比 1.0 = k が n 以上なら `_r` を使う。
const MIX_REDUCE_MIN_RATIO: f64 = 1.0;


/// task 121: mix GPU matvec のカーネル選択 (純関数・GPU 無しで検証可)。
/// true = `dno_matvec_f32_r` (block 256・k をスレッド分割して warp shuffle の
/// 固定木で縮約・決定論) / false = `dno_matvec_f32` (block 1・k 逐次)。
/// 長い行 (k >= n × [`MIX_REDUCE_MIN_RATIO`]) だけ `_r` にする。短い行は並列度が
/// 行数 n 側にあり逐次版で十分 (これを `_r` に変えると up が遅くなる)。
/// 縮退形 (n == 0) は安全側で逐次版 (実際は呼び出し側の形状検査が先に弾く)。
/// `has_r` は PTX に `_r` が無いとき false となり逐次版に落ちる (fail closed・
/// エラーにはしない)。[`fast_reduce_enabled`] (DENNO_FAST_REDUCE) には依存しない
/// (mix の長い行はオペレーター方針で bit 一致契約の外 = 速度優先・決定論のみ)。
pub(crate) fn mix_use_reduce_kernel(k: usize, n: usize, has_r: bool) -> bool {
    has_r && n > 0 && (k as f64) >= (n as f64) * MIX_REDUCE_MIN_RATIO
}


/// task 110 (task 121 で長い行を並列リダクション版に): hc mix BF16 (m==1) の
/// GEMV を GPU で実行する。
///
/// 戻り値 false = CPU パスを使うこと (GPU 無効・PTX 未収録・形状不一致・確保失敗・
/// 実行エラー)。数値は短い行 (k < n) が CPU 経路とビット一致するので、そちらは
/// どちらに落ちても出力は変わらない。長い行 (k >= n、[`mix_use_reduce_kernel`])
/// は s169 の並列リダクション版 `dno_matvec_f32_r` (block 256・warp shuffle の
/// 固定木) を使い、丸め順が CPU の逐次累積と違うためビット一致は**しない**
/// (実行ごとの決定論は保つ。オペレーター方針: 速度優先)。uid は `GgufLinear` の
/// もの (モデルロード毎に一意) で、そのまま常駐レジストリの key にする。bytes は
/// BF16 の生バイト (k * n * 2)。
///
/// ゲート (`DENNO_MIX_GPU`、既定 off) の判定は呼び出し側 (qwen4exp の
/// hc_mix_quant_fast) が持つ。
pub fn maybe_gpu_mix_bf16_into(
    uid: u64,
    bytes: &[u8],
    k: usize,
    n: usize,
    x: &[f32],
    out: &mut [f32],
) -> bool {
    use std::sync::atomic::Ordering as AO;
    #[cfg(test)]
    MIX_GPU_ATTEMPTS.fetch_add(1, AO::Relaxed);
    if !cuda_enabled() {
        return false;
    }
    if MIX_GPU_BROKEN.load(AO::Relaxed) {
        return false;
    }
    if k == 0
        || n == 0
        || x.len() != k
        || out.len() != n
        || bytes.len() != k.saturating_mul(n).saturating_mul(2)
        || k > i32::MAX as usize
        || n > i32::MAX as usize
    {
        mix_gpu_broken_once(&format!(
            "shape mismatch (k={k}, n={n}, bytes={} / want {})",
            bytes.len(),
            k.saturating_mul(n).saturating_mul(2)
        ));
        return false;
    }
    let Some(ctx) = shared_ctx() else {
        mix_gpu_broken_once("shared_ctx() is None");
        return false;
    };
    if let Err(e) = ctx.make_current() {
        mix_gpu_broken_once(&format!("cuCtxSetCurrent: {e:#}"));
        return false;
    }
    // カーネルハンドル (lazy fetch)。未収録 PTX は 1 行 warn して CPU に落とす。
    // task 152: `DENNO_MIX_BF16=1` かつ BF16 カーネルが PTX にあれば、重みを f32 に
    // 展開せず BF16 のまま読む (常駐が半分)。無ければ従来の f32 展開経路。
    let want_bf16 = mix_bf16_enabled() && matvec_bf16_fn(ctx).is_some();
    let (f_seq, f_red) = if want_bf16 {
        (matvec_bf16_fn(ctx), matvec_bf16_r_fn(ctx))
    } else {
        (matvec_f32_fn(ctx), matvec_f32_r_fn(ctx))
    };
    let Some(f) = f_seq else {
        mix_gpu_broken_once("kernels_elem.ptx に dno_matvec_f32 が無い (PTX 再生成が必要)");
        return false;
    };
    // task 121: 長い行 (k >= n) は s169 の並列リダクション版 (`dno_matvec_f32_r`)
    // を使う。逐次版は 1 ブロック = 1 行で k を 1 スレッドが舐めるため、down
    // (k=10240, n=320) のような形で並列度が足りない (実測 281 us/行列・ほぼ DRAM
    // 律速に達しない)。`_r` は DENNO_FAST_REDUCE と無関係にこの経路だけで選ぶ
    // (mix の長い行はオペレーター方針で bit 一致契約の外)。PTX に `_r` が無ければ
    // (Option が None) 逐次版に落ちる = 失敗にはしない (fail closed)。
    let fr = f_red;
    let use_reduce = mix_use_reduce_kernel(k, n, fr.is_some());
    // 重みの展開 + 常駐 (初回 1 回・同期 H2D)。ロック内で二重アップロード防止。
    let wp = match mix_gpu_weight(uid, bytes, k, n, want_bf16) {
        Ok(w) => w.ptr,
        Err(e) => {
            mix_gpu_broken_once(&format!("hc mix 重みの常駐に失敗: {e:#}"));
            return false;
        }
    };
    let lib = match lib() {
        Ok(l) => l,
        Err(e) => {
            mix_gpu_broken_once(&format!("CUDA lib load failed: {e:#}"));
            return false;
        }
    };
    let r = MIX_GPU_SC.with(|cell| -> Result<()> {
        let mut s = cell.borrow_mut();
        let MixGpuScratch {
            x: dx,
            out: dout,
            hx,
            ho,
            ..
        } = &mut *s;
        let bytes_x = k * 4;
        let bytes_out = n * 4;
        Scratch::ensure(dx, bytes_x)?;
        Scratch::ensure(dout, bytes_out)?;
        Scratch::ensure_pin(hx, bytes_x)?;
        Scratch::ensure_pin(ho, bytes_out)?;
        let (dx, dout) = (dx.as_ref().unwrap(), dout.as_ref().unwrap());
        let (hx, ho) = (hx.as_mut().unwrap(), ho.as_mut().unwrap());
        // task 177: mix 1 回の操作別内訳 (`DENNO_GDN_DEC_KPROF`・既定 off)。
        let _kp_win = dec_kprof::window(
            dec_kprof::MIX,
            dec_kprof::cur_layer(),
            dec_kprof::cur_pos(),
            0,
        );
        hx.as_mut_slice::<f32>()[..k].copy_from_slice(x);
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
            let (mut ph, mut pw, mut po) = (dx.ptr(), wp, dout.ptr());
            let (mut kc, mut nc) = (k as c_int, n as c_int);
            let mut args = [
                &mut ph as *mut _ as *mut c_void,
                &mut pw as *mut _ as *mut c_void,
                &mut po as *mut _ as *mut c_void,
                &mut kc as *mut _ as *mut c_void,
                &mut nc as *mut _ as *mut c_void,
            ];
            // task 121: 長い行は `_r` (1 ブロック = 1 行・block 256 で k を分割し
            // warp shuffle の固定木で縮約・決定論)、短い行は従来どおり逐次版
            // (カーネルが block 内全スレッドで同一計算するため block=(1,1,1)。
            // grid.x は 2^31-1 まで合法)。
            let (mv_fn, mv_thr): (CUfunction, u32) = if use_reduce {
                (fr.expect("use_reduce は has_r=true を要求する"), 256)
            } else {
                (f, 1)
            };
            ctx.launch_async(mv_fn, (n as u32, 1, 1), (mv_thr, 1, 1), 0, &mut args)?;
            check(
                (lib.memcpy_dtoh_async)(ho.ptr, dout.ptr(), bytes_out, std::ptr::null_mut()),
                "cuMemcpyDtoHAsync",
            )?;
            crate::prof::mark_d2h_src(crate::prof::D2H_SRC_LINEAR);
        }
        ctx.sync()?;
        out.copy_from_slice(&ho.as_slice::<f32>()[..n]);
        Ok(())
    });
    match r {
        Ok(()) => {
            set_last_linear_ran_on_gpu(true);
            true
        }
        Err(e) => {
            mix_gpu_broken_once(&format!("hc mix BF16 matvec failed: {e:#}"));
            false
        }
    }
}


/// task 145: prefill の hc mix (BF16 重み) を **m トークンまとめて** GPU で回す。
///
/// `out[t * n + r] = dot(bf16 を f32 展開した行 r, xs[t])`。
/// 重みは decode の mix GPU 経路と**同じ常駐バッファ** ([`mix_gpu_weight`]) を
/// 使うので、既に載っていれば転送は起きない。1 呼び出し = H2D 1 回 + 起動 1 回 +
/// 同期 1 回 + D2H 1 回。数値は FMA + タイル順なので CPU の mul+add 逐次とは
/// ビット一致しない (prefill のバッチ行列積は速度優先の契約)。
///
/// `false` = GPU でやらなかった (呼び出し側は CPU バッチへ)。
pub fn maybe_gpu_mix_bf16_batch_into(
    uid: u64,
    bytes: &[u8],
    k: usize,
    n: usize,
    xs: &[f32],
    nt: usize,
    out: &mut [f32],
) -> bool {
    if !cuda_enabled() || MIX_GPU_BROKEN.load(std::sync::atomic::Ordering::Relaxed) {
        return false;
    }
    if k == 0
        || n == 0
        || nt < 2
        || xs.len() != nt * k
        || out.len() != nt * n
        || bytes.len() != k.saturating_mul(n).saturating_mul(2)
        || k > i32::MAX as usize
        || n > i32::MAX as usize
        || nt > i32::MAX as usize
    {
        return false;
    }
    let Some(ctx) = shared_ctx() else {
        return false;
    };
    if let Err(e) = ctx.make_current() {
        mix_gpu_broken_once(&format!("cuCtxSetCurrent (batch): {e:#}"));
        return false;
    }
    // task 152: BF16 カーネルがあれば重みを BF16 のまま置く (常駐が半分)。
    let want_bf16 = mix_bf16_enabled() && gemm_bf16_mt_fn(ctx).is_some();
    let f = if want_bf16 {
        gemm_bf16_mt_fn(ctx)
    } else {
        gemm_f32_mt_fn(ctx)
    };
    let Some(f) = f else {
        // PTX 未収録 (PTX=elem の再生成待ち)。ラッチはしない。
        return false;
    };
    let wp = match mix_gpu_weight(uid, bytes, k, n, want_bf16) {
        Ok(w) => w.ptr,
        Err(e) => {
            mix_gpu_broken_once(&format!("batch の重み常駐に失敗: {e:#}"));
            return false;
        }
    };
    let lib = match lib() {
        Ok(l) => l,
        Err(e) => {
            mix_gpu_broken_once(&format!("CUDA lib load failed (batch): {e:#}"));
            return false;
        }
    };
    let r = MIX_GPU_SC.with(|cell| -> Result<()> {
        let mut s = cell.borrow_mut();
        let MixGpuScratch {
            x: dx, out: dout, ..
        } = &mut *s;
        let bytes_x = nt * k * 4;
        let bytes_out = nt * n * 4;
        Scratch::ensure(dx, bytes_x)?;
        Scratch::ensure(dout, bytes_out)?;
        let (dx, dout) = (dx.as_ref().unwrap(), dout.as_ref().unwrap());
        // s68 の実測 (pinned 経由 + memcpy は pageable 直の 4 倍遅い) に従い、
        // ホスト側のコピーを挟まず呼び出し元のスライス / 出力先へ直接発行する。
        dx.copy_from(xs)?;
        unsafe {
            let (mut pw, mut px, mut po) = (wp, dx.ptr(), dout.ptr());
            let (mut nt_c, mut k_c, mut n_c) = (nt as c_int, k as c_int, n as c_int);
            let mut args = [
                &mut pw as *mut _ as *mut c_void,
                &mut px as *mut _ as *mut c_void,
                &mut po as *mut _ as *mut c_void,
                &mut nt_c as *mut _ as *mut c_void,
                &mut k_c as *mut _ as *mut c_void,
                &mut n_c as *mut _ as *mut c_void,
            ];
            let grid = (
                ((n + GEMM_MT_ROWS - 1) / GEMM_MT_ROWS) as u32,
                ((nt + GEMM_MT_TOKS - 1) / GEMM_MT_TOKS) as u32,
                1,
            );
            ctx.launch_async(f, grid, (GEMM_MT_ROWS as u32, 16, 1), 0, &mut args)?;
            ctx.sync()?;
            // D2H は同期コピーで `out` へ直接 (中間 Vec / pinned を作らない)。
            check(
                (lib.memcpy_dtoh)(
                    out.as_mut_ptr() as *mut c_void,
                    dout.ptr(),
                    bytes_out,
                ),
                "cuMemcpyDtoH (mix batch)",
            )?;
            crate::prof::mark_d2h_src(crate::prof::D2H_SRC_LINEAR);
        }
        Ok(())
    });
    match r {
        Ok(()) => {
            set_last_linear_ran_on_gpu(true);
            true
        }
        Err(e) => {
            mix_gpu_broken_once(&format!("hc mix BF16 batch GEMM failed: {e:#}"));
            false
        }
    }
}


// ---------------------------------------------------------------------------
// task 186: prefill hc mix の down / up 用 GEMM (`kernels_mix_pf.cu`、既定 off)
// ---------------------------------------------------------------------------
//
// 既存 `dno_gemm_bf16_mt` は 1 ブロック = 16 行 x 32 トークンで、k タイル 32 ごとに
// `__syncthreads` を 2 回打つ (barrier あたりの FMA が 64)。lab t183a の 14k
// prefill で mix の down + up は 4.2 GFLOP を 1.27 ms = 3.1 TFLOP/s (fp32 山の 3%)
// で回っており、同じホストの隣の GEMM は 70-75 TFLOP/s。mix の 146 MB / 1.34 ms は
// 109 GB/s なので帯域律速でもない ([REPORT_P186.md] 0.1 の表)。
//
// 新しい `dno_mix_pf_gemm` は 32 トークン x 64 行 x k タイル 16 で、barrier あたりの
// FMA が 256。**値は既存カーネルとビット一致** (1 出力 1 アキュムレータ・k 昇順の
// `__fmaf_rn`、BF16 -> f32 は上位 16 bit の複製で CPU の dequant と同一) なので、
// 採否は速度だけで決まる。
//
// PTX はクレートの `kernels/kernels_mix_pf.ptx` (lab が `bash build_ptx.sh mix_pf`
// で生成) を実行時に読む。無い・読めない・entry が無いときは従来カーネルへ
// fail closed (ゲート on でも既定 off と同じ経路・同じ値)。

/// 1 ブロックの出力行数 (カーネルの MIXP_BN)。
pub(crate) const MIX_PF_BN: usize = 64;

/// 1 ブロックのトークン数 (カーネルの MIXP_BM)。
pub(crate) const MIX_PF_BM: usize = 32;

/// スレッド数 (カーネルの MIXP_THREADS)。
pub(crate) const MIX_PF_THREADS: usize = 128;

/// 新しいカーネルを使う最小トークン数 (それ未満は既存カーネルのほうが速い)。
pub const MIX_PF_MIN_TOK: usize = 64;


/// task 186: `kernels_mix_pf.cu` の関数一式。
pub(crate) struct MixPfKernels {
    #[allow(dead_code)]
    module: CUmodule,
    gemm: CUfunction,
}


// 生ハンドル (PfKernels と同じ規約)。
unsafe impl Send for MixPfKernels {}

unsafe impl Sync for MixPfKernels {}


/// PTX の置き場所。`DENNO_MIX_PF_PTX` があればそれ、無ければクレートの
/// `kernels/kernels_mix_pf.ptx`。
fn mix_pf_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_MIX_PF_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_mix_pf.ptx"),
    }
}


/// カーネルを取得する (初回だけ PTX を読む)。失敗は None を焼き付ける。
/// **current context を張った後に呼ぶこと**。
fn mix_pf_kernels() -> Option<&'static MixPfKernels> {
    static K: OnceLock<Option<MixPfKernels>> = OnceLock::new();
    K.get_or_init(|| {
        let path = mix_pf_ptx_path();
        let mut image = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                tracing::info!(
                    "[mix-pf] PTX が読めない ({}) : {e} → 従来の GEMM へ",
                    path.display()
                );
                return None;
            }
        };
        image.push(0);
        let lib = lib().ok()?;
        let mut module: CUmodule = std::ptr::null_mut();
        unsafe {
            if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void) != CUDA_SUCCESS {
                tracing::info!("[mix-pf] cuModuleLoadData 失敗 → 従来の GEMM へ");
                return None;
            }
            let mut gemm: CUfunction = std::ptr::null_mut();
            if (lib.module_get_function)(
                &mut gemm,
                module,
                b"dno_mix_pf_gemm\0".as_ptr() as *const c_char,
            ) != CUDA_SUCCESS
            {
                tracing::info!("[mix-pf] PTX に dno_mix_pf_gemm が無い → 従来の GEMM へ");
                return None;
            }
            log_kernel_attr(lib, gemm, "dno_mix_pf_gemm");
            Some(MixPfKernels { module, gemm })
        }
    })
    .as_ref()
}


/// task 186: `DENNO_PF_MIX_FAST` の解釈 (純関数)。`0`/`off`/`false`/`no`/空だけが off。
pub(crate) fn mix_pf_fast_from_value(v: Option<&str>) -> bool {
    match v {
        None => false,
        Some(s) => !matches!(s.trim(), "0" | "off" | "false" | "no" | ""),
    }
}


/// ゲートが on か (プロセスで 1 回だけ読む)。
pub(crate) fn mix_pf_fast_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| mix_pf_fast_from_value(std::env::var("DENNO_PF_MIX_FAST").ok().as_deref()))
}


/// task 186: その形を新しいカーネルで回してよいか (純関数)。
///
/// BF16 重み・n が 1 タイル以上・nt が [`MIX_PF_MIN_TOK`] 以上・k が 1 以上。
/// inject (n=4) のような小さい N は従来カーネル (grid を 16 行で割る) のほうが速い。
pub(crate) fn mix_pf_ok(k: usize, n: usize, nt: usize, bf16: bool) -> bool {
    bf16 && k > 0 && n >= MIX_PF_BN && nt >= MIX_PF_MIN_TOK
}


// ---------------------------------------------------------------------------
// task 189: mix の inject (n = hc <= 8) を 1 スレッド 1 出力の GEMV で回す
// ---------------------------------------------------------------------------
//
// `dno_gemm_bf16_mt` は 1 ブロック 16 行 x 32 トークンなので、n=4 の inject では
// grid.x = 1 (16 行のうち 4 行だけ有効)・ブロック数 16 本になり、21 M MAC の
// 仕事に 329 us/呼び出し (2,688 呼び出しで 884 ms) かかっていた (lab t189a の
// `[pf-mix-kprof] inj`)。詳細は `kernels/kernels_mix_inj.cu` の先頭。

/// 1 ブロックのスレッド数 (カーネルの `MIXINJ_THREADS` と一致させる)。
pub(crate) const MIX_INJ_THREADS: usize = 128;

/// この n 以下を新しいカーネルで回す (inject の実形は hc=4)。
pub(crate) const MIX_INJ_MAX_N: usize = 8;

/// これ未満のトークン数は従来カーネル (起動のほうが高い)。
pub const MIX_INJ_MIN_TOK: usize = 32;


fn mix_inj_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_MIX_INJ_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_mix_inj.ptx"),
    }
}


pub(crate) struct MixInjKernels {
    #[allow(dead_code)]
    module: CUmodule,
    gemv: CUfunction,
}


// 生ハンドル (MixPfKernels / PfKernels と同じ規約)。
unsafe impl Send for MixInjKernels {}

unsafe impl Sync for MixInjKernels {}


/// カーネルを取得する (初回だけ PTX を読む)。失敗は None を焼き付ける。
/// **current context を張った後に呼ぶこと**。
fn mix_inj_kernels() -> Option<&'static MixInjKernels> {
    static K: OnceLock<Option<MixInjKernels>> = OnceLock::new();
    K.get_or_init(|| {
        let path = mix_inj_ptx_path();
        let mut image = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                tracing::info!(
                    "[mix-inj] PTX が読めない ({}) : {e} → 従来の GEMM へ",
                    path.display()
                );
                return None;
            }
        };
        image.push(0);
        let lib = lib().ok()?;
        let mut module: CUmodule = std::ptr::null_mut();
        unsafe {
            if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void) != CUDA_SUCCESS {
                tracing::info!("[mix-inj] cuModuleLoadData 失敗 → 従来の GEMM へ");
                return None;
            }
            let mut gemv: CUfunction = std::ptr::null_mut();
            if (lib.module_get_function)(
                &mut gemv,
                module,
                b"dno_mix_inj_bf16\0".as_ptr() as *const c_char,
            ) != CUDA_SUCCESS
            {
                tracing::info!("[mix-inj] PTX に dno_mix_inj_bf16 が無い → 従来の GEMM へ");
                return None;
            }
            log_kernel_attr(lib, gemv, "dno_mix_inj_bf16");
            Some(MixInjKernels { module, gemv })
        }
    })
    .as_ref()
}


/// task 189: `DENNO_PF_MIX_INJ` の解釈 (純関数)。`0`/`off`/`false`/`no`/空だけが off。
pub(crate) fn mix_inj_from_value(v: Option<&str>) -> bool {
    match v {
        None => false,
        Some(s) => !matches!(s.trim(), "0" | "off" | "false" | "no" | ""),
    }
}


/// 新しい inject GEMV を使うか (`DENNO_PF_MIX_INJ`、既定 off。lab で決める)。
pub(crate) fn mix_inj_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| mix_inj_from_value(std::env::var("DENNO_PF_MIX_INJ").ok().as_deref()))
}


/// task 189: その形を新しいカーネルで回してよいか (純関数)。
///
/// BF16 重み・1 <= n <= [`MIX_INJ_MAX_N`]・nt >= [`MIX_INJ_MIN_TOK`]・k > 0。
pub(crate) fn mix_inj_ok(k: usize, n: usize, nt: usize, bf16: bool) -> bool {
    bf16 && k > 0 && n >= 1 && n <= MIX_INJ_MAX_N && nt >= MIX_INJ_MIN_TOK
}


#[cfg(test)]
mod mix_inj_tests {
    use super::*;

    #[test]
    fn mix_inj_gate_truth_table() {
        assert!(!mix_inj_from_value(None), "未設定は既定 off");
        for v in ["0", "off", "false", "no", "", " off "] {
            assert!(!mix_inj_from_value(Some(v)), "{v:?} が on になった");
        }
        for v in ["1", "on", "true", "yes", "2"] {
            assert!(mix_inj_from_value(Some(v)), "{v:?} が off のまま");
        }
    }

    #[test]
    fn mix_inj_ok_checks_shape() {
        // 実形の inject: k = hc*e = 10240, n = hc = 4, nt = 512
        assert!(mix_inj_ok(10240, 4, 512, true));
        assert!(mix_inj_ok(10240, 8, 512, true));
        // down (n=320) / up (n=10240) は対象外 (従来の GEMM のほうが速い)
        assert!(!mix_inj_ok(2560, 320, 512, true));
        assert!(!mix_inj_ok(320, 10240, 512, true));
        // f32 展開した重みは対象外
        assert!(!mix_inj_ok(10240, 4, 512, false));
        // 短い prefill は従来カーネル
        assert!(!mix_inj_ok(10240, 4, 19, true));
        assert!(!mix_inj_ok(0, 4, 512, true));
        assert!(!mix_inj_ok(10240, 0, 512, true));
        assert!(!mix_inj_ok(10240, 9, 512, true));
    }
}


// ---------------------------------------------------------------------------
// task 195: mix の down / up を **n 方向に細かく割る** 新しい GEMM
// ---------------------------------------------------------------------------
//
// lab t189a / t189c の実測: mix の down は [nt=512, k=10,240] x [10,240, n=320] の
// 1.68 G MAC を 490 us (3.4 TMAC/s)。既存 `dno_gemm_bf16_mt` は 1 ブロック
// 16 行 x 32 トークン・1 スレッド 2 アキュムレータなので、共有メモリ読みが
// 1.5 回/FMA で ILP も足りない。mix_pf (task 186) は 16 アキュムレータだが
// 1 ブロック 64 行なので down では 5 x 16 = 80 ブロックしか出ない。
// 新しいカーネルは 1 ブロック 32 行 x 32 トークン・8 アキュムレータで、
// down は 10 x 16 = 160 ブロック、up は 320 x 16 = 5,120 ブロックになる。
// **値は既存カーネルとビット一致** (1 出力 1 アキュムレータ・k 昇順)。
//
// PTX はクレートの `kernels/kernels_mix_split.ptx` (lab で
// `bash build_ptx.sh mix_split` を実行して生成) を実行時に読む。無い・読めない・
// entry が無いときは従来カーネルへ fail closed。

/// 1 ブロックの出力行数 (カーネルの MXS_BN)。
pub(crate) const MIX_SPLIT_BN: usize = 32;

/// 1 ブロックのトークン数 (カーネルの MXS_BM)。
pub(crate) const MIX_SPLIT_BM: usize = 32;

/// スレッド数 (カーネルの MXS_THREADS)。
pub(crate) const MIX_SPLIT_THREADS: usize = 128;


fn mix_split_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_MIX_SPLIT_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_mix_split.ptx"),
    }
}


/// task 195: `kernels_mix_split.cu` の関数一式。
///
/// フィールド名 `gemm_split` は kernel_invariants の起動サイト解析が
/// `module_get_function` のローカル名と突き合わせるために一意にしてある
/// (mix_pf の `gemm` と同名だと表が上書きされる)。
pub(crate) struct MixSplitKernels {
    #[allow(dead_code)]
    module: CUmodule,
    gemm_split: CUfunction,
}


// 生ハンドル (MixPfKernels / PfKernels と同じ規約)。
unsafe impl Send for MixSplitKernels {}

unsafe impl Sync for MixSplitKernels {}


/// カーネルを取得する (初回だけ PTX を読む)。失敗は None を焼き付ける。
/// **current context を張った後に呼ぶこと**。
fn mix_split_kernels() -> Option<&'static MixSplitKernels> {
    static K: OnceLock<Option<MixSplitKernels>> = OnceLock::new();
    K.get_or_init(|| {
        let path = mix_split_ptx_path();
        let mut image = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                tracing::info!(
                    "[mix-split] PTX が読めない ({}) : {e} → 従来の GEMM へ",
                    path.display()
                );
                return None;
            }
        };
        image.push(0);
        let lib = lib().ok()?;
        let mut module: CUmodule = std::ptr::null_mut();
        unsafe {
            if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void) != CUDA_SUCCESS {
                tracing::info!("[mix-split] cuModuleLoadData 失敗 → 従来の GEMM へ");
                return None;
            }
            let mut gemm_split: CUfunction = std::ptr::null_mut();
            if (lib.module_get_function)(
                &mut gemm_split,
                module,
                b"dno_mix_split_gemm\0".as_ptr() as *const c_char,
            ) != CUDA_SUCCESS
            {
                tracing::info!("[mix-split] PTX に dno_mix_split_gemm が無い → 従来の GEMM へ");
                return None;
            }
            log_kernel_attr(lib, gemm_split, "dno_mix_split_gemm");
            Some(MixSplitKernels {
                module,
                gemm_split,
            })
        }
    })
    .as_ref()
}


/// task 195: `DENNO_PF_MIX_SPLIT` の解釈 (純関数)。`0`/`off`/`false`/`no`/空だけが off。
///
/// **既定 on** (lab t195a/t195b: 単独で -2.0%/-2.3%、4 ゲート同時で -6%)。
pub(crate) fn mix_split_from_value(v: Option<&str>) -> bool {
    match v {
        None => true,
        Some(s) => !matches!(s.trim(), "0" | "off" | "false" | "no" | ""),
    }
}


/// 新しい n 分割 GEMM を使うか (`DENNO_PF_MIX_SPLIT`。task 198 で既定 on)。
pub(crate) fn mix_split_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| mix_split_from_value(std::env::var("DENNO_PF_MIX_SPLIT").ok().as_deref()))
}


/// task 195: その形を n 分割カーネルで回してよいか (純関数)。
///
/// BF16 重み・n >= 1 タイル・nt >= 1 トークンタイル・k > 0。inject (n=hc) は
/// [`mix_inj2_ok`] の担当なので除く。
pub(crate) fn mix_split_ok(k: usize, n: usize, nt: usize, bf16: bool) -> bool {
    bf16 && k > 0 && n >= MIX_SPLIT_BN && nt >= MIX_SPLIT_BM
}


// ---------------------------------------------------------------------------
// task 195: mix の inject を **k 方向にレーン分割**する GEMV
// ---------------------------------------------------------------------------
//
// task 189 の `dno_mix_inj_bf16` はビット一致を保つため k 昇順 1 アキュムレータで
// 444 us/呼び出し (既存 GEMM は 329 us) だった。こちらは k を 256 スレッドに
// インターリーブ分割し、固定木で縮約する。**丸め順が変わる (ビット一致ではない)**
// が、決定論的。詳細は `kernels/kernels_mix_inj2.cu` の先頭。

/// 1 ブロックのスレッド数 (カーネルの MXI2_THREADS)。
pub(crate) const MIX_INJ2_THREADS: usize = 256;

/// この n 以下を新しいカーネルで回す (inject の実形は hc=4)。
pub(crate) const MIX_INJ2_MAX_N: usize = 8;


fn mix_inj2_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_MIX_INJ2_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_mix_inj2.ptx"),
    }
}


pub(crate) struct MixInj2Kernels {
    #[allow(dead_code)]
    module: CUmodule,
    gemv2: CUfunction,
}


// 生ハンドル (MixPfKernels / PfKernels と同じ規約)。
unsafe impl Send for MixInj2Kernels {}

unsafe impl Sync for MixInj2Kernels {}


/// カーネルを取得する (初回だけ PTX を読む)。失敗は None を焼き付ける。
/// **current context を張った後に呼ぶこと**。
fn mix_inj2_kernels() -> Option<&'static MixInj2Kernels> {
    static K: OnceLock<Option<MixInj2Kernels>> = OnceLock::new();
    K.get_or_init(|| {
        let path = mix_inj2_ptx_path();
        let mut image = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                tracing::info!(
                    "[mix-inj2] PTX が読めない ({}) : {e} → 従来の GEMM へ",
                    path.display()
                );
                return None;
            }
        };
        image.push(0);
        let lib = lib().ok()?;
        let mut module: CUmodule = std::ptr::null_mut();
        unsafe {
            if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void) != CUDA_SUCCESS {
                tracing::info!("[mix-inj2] cuModuleLoadData 失敗 → 従来の GEMM へ");
                return None;
            }
            let mut gemv2: CUfunction = std::ptr::null_mut();
            if (lib.module_get_function)(
                &mut gemv2,
                module,
                b"dno_mix_inj2_bf16\0".as_ptr() as *const c_char,
            ) != CUDA_SUCCESS
            {
                tracing::info!("[mix-inj2] PTX に dno_mix_inj2_bf16 が無い → 従来の GEMM へ");
                return None;
            }
            log_kernel_attr(lib, gemv2, "dno_mix_inj2_bf16");
            Some(MixInj2Kernels { module, gemv2 })
        }
    })
    .as_ref()
}


/// task 195: `DENNO_PF_MIX_INJ2` の解釈 (純関数)。`0`/`off`/`false`/`no`/空だけが off。
///
/// **既定 on** (lab t195a: -2.4%/-1.8%。丸め順が変わる = md5 DIFF はこのゲート由来)。
pub(crate) fn mix_inj2_from_value(v: Option<&str>) -> bool {
    match v {
        None => true,
        Some(s) => !matches!(s.trim(), "0" | "off" | "false" | "no" | ""),
    }
}


/// 新しい inject GEMV (k 分割) を使うか (`DENNO_PF_MIX_INJ2`。task 198 で既定 on)。
pub(crate) fn mix_inj2_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| mix_inj2_from_value(std::env::var("DENNO_PF_MIX_INJ2").ok().as_deref()))
}


/// task 195: その形を k 分割の inject で回してよいか (純関数)。
///
/// BF16 重み・1 <= n <= [`MIX_INJ2_MAX_N`]・k > 0・nt >= 1。1 ブロック 1 トークン
/// なのでトークン数の下限は無い (起動数 = nt)。
pub(crate) fn mix_inj2_ok(k: usize, n: usize, nt: usize, bf16: bool) -> bool {
    bf16 && k > 0 && n >= 1 && n <= MIX_INJ2_MAX_N && nt >= 1
}


#[cfg(test)]
mod mix_split_inj2_tests {
    use super::*;

    #[test]
    fn mix_split_gate_truth_table() {
        assert!(
            mix_split_from_value(None),
            "未設定は既定 on (task 198 で反転。明示 off は下の表)"
        );
        for v in ["0", "off", "false", "no", "", " off "] {
            assert!(!mix_split_from_value(Some(v)), "{v:?} が on になった");
        }
        for v in ["1", "on", "true", "yes", "2"] {
            assert!(mix_split_from_value(Some(v)), "{v:?} が off のまま");
        }
    }

    #[test]
    fn mix_split_ok_checks_shape() {
        // 実形: down (k=10240, n=320) / up (k=320, n=10240)、nt は 512
        assert!(mix_split_ok(10240, 320, 512, true));
        assert!(mix_split_ok(320, 10240, 512, true));
        // inject (n=4) は対象外
        assert!(!mix_split_ok(10240, 4, 512, true));
        // f32 展開した重みは対象外
        assert!(!mix_split_ok(10240, 320, 512, false));
        // 1 タイル未満の n / nt は従来カーネル
        assert!(!mix_split_ok(10240, 31, 512, true));
        assert!(!mix_split_ok(10240, 320, 31, true));
        assert!(!mix_split_ok(0, 320, 512, true));
    }

    #[test]
    fn mix_inj2_gate_truth_table() {
        assert!(
            mix_inj2_from_value(None),
            "未設定は既定 on (task 198 で反転。明示 off は下の表)"
        );
        for v in ["0", "off", "false", "no", "", " off "] {
            assert!(!mix_inj2_from_value(Some(v)), "{v:?} が on になった");
        }
        for v in ["1", "on", "true", "yes", "2"] {
            assert!(mix_inj2_from_value(Some(v)), "{v:?} が off のまま");
        }
    }

    #[test]
    fn mix_inj2_ok_checks_shape() {
        // 実形の inject: k = hc*e = 10240, n = hc = 4, nt = 512
        assert!(mix_inj2_ok(10240, 4, 512, true));
        assert!(mix_inj2_ok(10240, 8, 1, true));
        // down / up は対象外 (n > 8)
        assert!(!mix_inj2_ok(10240, 320, 512, true));
        assert!(!mix_inj2_ok(320, 10240, 512, true));
        // f32 展開した重みは対象外
        assert!(!mix_inj2_ok(10240, 4, 512, false));
        assert!(!mix_inj2_ok(0, 4, 512, true));
        assert!(!mix_inj2_ok(10240, 0, 512, true));
        assert!(!mix_inj2_ok(10240, 9, 512, true));
    }

    /// task 617: `DENNO_PF_INJ_F32_BF16` の真理値表 (**既定 on**、明示的な偽値だけ off)。
    #[test]
    fn pf_inj_f32_bf16_gate_truth_table() {
        assert!(
            pf_inj_f32_bf16_from_value(None),
            "未設定は既定 on (lab t617a/t617b で採否 yes)"
        );
        for v in ["0", "off", "false", "no", "", " off "] {
            assert!(!pf_inj_f32_bf16_from_value(Some(v)), "{v:?} が on になった");
        }
        for v in ["1", "on", "true", "yes", " 1 ", "2"] {
            assert!(pf_inj_f32_bf16_from_value(Some(v)), "{v:?} が off のまま");
        }
    }

    /// task 617: BF16 化してよい形は inject の実形だけ (k = 10240, n = hc <= 8)。
    #[test]
    fn pf_inj_bf16_shape_checks_inject_form() {
        assert!(pf_inj_bf16_shape(10240, 4));
        assert!(pf_inj_bf16_shape(10240, 8));
        assert!(pf_inj_bf16_shape(2560, 1));
        // down / up の形は対象外。
        assert!(!pf_inj_bf16_shape(10240, 320));
        assert!(!pf_inj_bf16_shape(320, 10240));
        assert!(!pf_inj_bf16_shape(10240, 9));
        assert!(!pf_inj_bf16_shape(0, 4));
        assert!(!pf_inj_bf16_shape(10240, 0));
        // `mix_inj2_ok` と両立する (BF16 化したあとに k 分割で撃てる)。
        assert!(mix_inj2_ok(10240, 4, 512, pf_inj_bf16_shape(10240, 4)));
    }

    /// task 617: f32 -> bf16 の丸め (RNE) と、NaN / 端数の扱い。
    #[test]
    fn f32_bytes_to_bf16_bytes_rounds_to_nearest_even() {
        let f = |v: f32| v.to_bits().to_le_bytes();
        let mut src = Vec::new();
        for v in [1.0f32, -2.5, 0.0, 1e-30] {
            src.extend_from_slice(&f(v));
        }
        let out = f32_bytes_to_bf16_bytes(&src).expect("4 の倍数");
        assert_eq!(out.len(), src.len() / 2, "バイト数が半分でない");
        let bf = |i: usize| u16::from_le_bytes([out[i * 2], out[i * 2 + 1]]);
        assert_eq!(bf(0), 0x3f80, "1.0 は 0x3f80");
        assert_eq!(bf(1), 0xc020, "-2.5 は 0xc020");
        assert_eq!(bf(2), 0x0000, "0.0 は 0x0000");
        // bf16 の指数は f32 と同じ 8 bit なので 1e-30 も表せる (丸め誤差 1% 以内)。
        let back = f32::from_bits((bf(3) as u32) << 16);
        assert!(
            (back - 1e-30f32).abs() <= 1e-30f32 * 0.01,
            "1e-30 の丸めが大きすぎる: {back:e}"
        );

        // 中点 (仮数の下位 16 bit がちょうど 0x8000) は偶数側へ。
        let mid_even = f32::from_bits(0x3f80_8000); // 1.0 + 1 ulp(bf16 の中点)
        let mid_odd = f32::from_bits(0x3f81_8000);
        let src = [mid_even, mid_odd]
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect::<Vec<u8>>();
        let out = f32_bytes_to_bf16_bytes(&src).unwrap();
        let bf = |i: usize| u16::from_le_bytes([out[i * 2], out[i * 2 + 1]]);
        assert_eq!(bf(0), 0x3f80, "中点は偶数 (0x3f80) へ");
        assert_eq!(bf(1), 0x3f82, "中点は偶数 (0x3f82) へ");

        // NaN は inf に化けない。
        let nan = f32::from_bits(0x7fc0_1234);
        let src = nan.to_bits().to_le_bytes().to_vec();
        let out = f32_bytes_to_bf16_bytes(&src).unwrap();
        let hi = u16::from_le_bytes([out[0], out[1]]);
        assert_eq!(hi & 0x7f80, 0x7f80, "指数は全部 1 のまま");
        assert_ne!(hi & 0x007f, 0, "仮数が 0 でない = NaN のまま");

        // 端数 (4 の倍数でない) は None。
        assert!(f32_bytes_to_bf16_bytes(&[0u8; 6]).is_none());
        assert_eq!(f32_bytes_to_bf16_bytes(&[]).unwrap().len(), 0);
    }

    /// task 617: 発火の計数 (撃った回数が数えられ、reset で戻る)。
    #[test]
    fn pf_inj2_calls_counts_and_resets() {
        pf_inj2_calls_reset();
        assert_eq!(pf_inj2_calls(), 0);
        PF_INJ2_CALLS.fetch_add(2, Ordering::Relaxed);
        assert_eq!(pf_inj2_calls(), 2);
        pf_inj2_calls_reset();
        assert_eq!(pf_inj2_calls(), 0);
    }
}


#[cfg(test)]
mod mix_pf_tests {
    use super::*;

    #[test]
    fn mix_pf_fast_gate_truth_table() {
        assert!(!mix_pf_fast_from_value(None), "未設定は既定 off");
        for v in ["0", "off", "false", "no", "", " off "] {
            assert!(!mix_pf_fast_from_value(Some(v)), "{v:?} が on になった");
        }
        for v in ["1", "on", "true", "yes", "2"] {
            assert!(mix_pf_fast_from_value(Some(v)), "{v:?} が off のまま");
        }
    }

    #[test]
    fn mix_pf_ok_checks_shape() {
        // 実形: down (k=2560, n=320) / up (k=320, n=10240)
        assert!(mix_pf_ok(2560, 320, 512, true));
        assert!(mix_pf_ok(320, 10240, 512, true));
        // inject (n=4) は従来カーネル
        assert!(!mix_pf_ok(10240, 4, 512, true));
        // f32 展開した重みは対象外 (PfWKind::F32 は従来カーネル)
        assert!(!mix_pf_ok(2560, 320, 512, false));
        // 短い prefill (次のターンの 19 トークン) は従来カーネル
        assert!(!mix_pf_ok(2560, 320, 19, true));
        assert!(!mix_pf_ok(0, 320, 512, true));
        assert!(!mix_pf_ok(2560, 0, 512, true));
    }
}


// ---------------------------------------------------------------------------
// task 189: mix 1 呼び出しの段ごとの内訳 (`DENNO_PF_MIX_KPROF=1`・既定 off)
// ---------------------------------------------------------------------------
//
// 台帳の `mix` 行は「発行 + 同期の待ち」の壁時計で、`gpu` は
// `cuCtxSynchronize` の待ちから空 sync の床を引いただけなので、**別ストリーム
// (prefill の expert upload) の完了待ち**も同じ欄に入る。REPORT_P186 はこれを
// mix のカーネル時間と読み違えた。ここでは段の前後に CUDA イベントを打って
// 段の GPU 時間だけを積み、同期は
//
// - `sync_own`: null ストリーム (mix 自身の 6 起動) の完了待ち
// - `sync_other`: そのあと `cuCtxSynchronize` が**別ストリーム**を待った分
//
// に分けて積む。`sync_other` が大きければ待ちの正体は expert upload。
pub mod pf_mix_kprof {
    use super::*;

    /// 段の番号 (この順で 1 行に出す)。
    pub const NORM: usize = 0;
    pub const DOWN: usize = 1;
    pub const LO: usize = 2;
    pub const UP: usize = 3;
    pub const MIXED: usize = 4;
    pub const INJ: usize = 5;
    pub const SYNC_OWN: usize = 6;
    pub const SYNC_OTHER: usize = 7;
    pub const D2H: usize = 8;
    pub const N: usize = 9;
    pub const NAMES: [&str; N] = [
        "norm", "down", "lo", "up", "mixed", "inj", "sync_own", "sync_other", "d2h",
    ];

    pub(crate) static NS: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
    pub(crate) static CNT: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
    /// mix 呼び出しの回数 (1 回あたりに割る分母)。
    pub(crate) static CALLS: AtomicU64 = AtomicU64::new(0);

    /// 計測するか (`DENNO_PF_MIX_KPROF` があるとき)。
    pub fn on() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("DENNO_PF_MIX_KPROF").is_some())
    }

    /// 積算を 0 に戻す。
    pub fn reset() {
        for i in 0..N {
            NS[i].store(0, Ordering::Relaxed);
            CNT[i].store(0, Ordering::Relaxed);
        }
        CALLS.store(0, Ordering::Relaxed);
        // task 617: inject を k 分割で撃った回数も同じ区切りで戻す。
        pf_inj2_calls_reset();
        open_reset();
    }

    /// mix 1 呼び出しを数える。
    pub fn note_call() {
        CALLS.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn add_ns(i: usize, ns: u64) {
        if i >= N {
            return;
        }
        NS[i].fetch_add(ns, Ordering::Relaxed);
        CNT[i].fetch_add(1, Ordering::Relaxed);
    }

    /// ホストで測った区間 (同期の待ち・D2H) を積む。
    pub fn add_host(i: usize, t: Option<std::time::Instant>) {
        if !on() {
            return;
        }
        if let Some(t) = t {
            add_ns(i, t.elapsed().as_nanos() as u64);
        }
    }

    /// 1 行にまとめる (1 本も打っていなければ `None`)。
    pub fn lines() -> Vec<String> {
        if !on() {
            return Vec::new();
        }
        let calls = CALLS.load(Ordering::Relaxed);
        if calls == 0 {
            return Vec::new();
        }
        let mut s = format!(
            "[pf-mix-kprof] calls={calls} (1 呼び出し平均 us / 合計 ms, イベント + ホスト計測)"
        );
        for i in 0..N {
            let n = CNT[i].load(Ordering::Relaxed);
            if n == 0 {
                continue;
            }
            let ns = NS[i].load(Ordering::Relaxed);
            s.push_str(&format!(
                " | {} {:.1}/{:.1}",
                NAMES[i],
                ns as f64 / 1e3 / n as f64,
                ns as f64 / 1e6
            ));
        }
        // task 617: inject を k 分割カーネル (BF16 重み) で撃った回数。
        let inj2 = pf_inj2_calls();
        if inj2 > 0 {
            s.push_str(&format!(" | inj2_calls={inj2}"));
        }
        vec![s]
    }

    /// イベントのプール (スレッドローカル)。収集のたびに先頭から使い直す。
    pub(crate) struct Pool {
        evs: Vec<CUevent>,
        used: usize,
        dead: bool,
    }

    thread_local! {
        static P: std::cell::RefCell<Pool> = const {
            std::cell::RefCell::new(Pool { evs: Vec::new(), used: 0, dead: false })
        };
        static OPEN: std::cell::RefCell<[Option<u32>; N]> = const {
            std::cell::RefCell::new([None; N])
        };
        static PEND: std::cell::RefCell<Vec<(u8, u32, u32)>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }

    pub(crate) fn mark() -> Option<u32> {
        P.with(|c| {
            let mut p = c.borrow_mut();
            if p.dead {
                return None;
            }
            if p.used >= p.evs.len() {
                let Some(create) = lib().ok().and_then(|l| l.event_create) else {
                    p.dead = true;
                    return None;
                };
                let mut ev: CUevent = std::ptr::null_mut();
                // CU_EVENT_DEFAULT (0) = 計時有効。
                if unsafe { create(&mut ev, 0) } != CUDA_SUCCESS || ev.is_null() {
                    p.dead = true;
                    return None;
                }
                p.evs.push(ev);
            }
            let Some(rec) = lib().ok().and_then(|l| l.event_record) else {
                p.dead = true;
                return None;
            };
            let ev = p.evs[p.used];
            if unsafe { rec(ev, std::ptr::null_mut()) } != CUDA_SUCCESS {
                p.dead = true;
                return None;
            }
            let i = p.used as u32;
            p.used += 1;
            Some(i)
        })
    }

    pub(crate) fn open_reset() {
        OPEN.with(|c| {
            let mut o = c.borrow_mut();
            for v in o.iter_mut() {
                *v = None;
            }
        });
        PEND.with(|c| c.borrow_mut().clear());
        P.with(|c| c.borrow_mut().used = 0);
    }

    /// 段の入口 (on でなければ 1 分岐)。
    pub fn begin(item: usize) {
        if !on() || item >= N {
            return;
        }
        let a = mark();
        OPEN.with(|c| c.borrow_mut()[item] = a);
    }

    /// 段の出口。
    pub fn end(item: usize) {
        if !on() || item >= N {
            return;
        }
        let a = OPEN.with(|c| c.borrow_mut()[item].take());
        let Some(a) = a else { return };
        if let Some(b) = mark() {
            PEND.with(|c| c.borrow_mut().push((item as u8, a, b)));
        }
    }

    /// 積算する (**同期の後**に呼ぶ。イベントが完了している必要がある)。
    pub fn collect() {
        if !on() {
            return;
        }
        let Some(f) = lib().ok().and_then(|l| l.event_elapsed_time) else {
            return;
        };
        let pend = PEND.with(|c| std::mem::take(&mut *c.borrow_mut()));
        if pend.is_empty() {
            return;
        }
        P.with(|c| {
            let p = c.borrow();
            for (item, a, b) in pend {
                let (Some(&ea), Some(&eb)) = (p.evs.get(a as usize), p.evs.get(b as usize)) else {
                    continue;
                };
                let mut ms = 0.0f32;
                if unsafe { f(&mut ms, ea, eb) } == CUDA_SUCCESS && ms.is_finite() && ms >= 0.0 {
                    add_ns(item as usize, (ms as f64 * 1e6) as u64);
                }
            }
        });
        // イベントは先頭から使い直す (プールは 20 本で足りる)。
        P.with(|c| c.borrow_mut().used = 0);
    }
}


/// task 189: `DENNO_PF_MIX_STAGE` の解釈 (純関数)。`0`/`off`/`false`/`no`/空だけが off。
pub(crate) fn mix_stage_from_value(v: Option<&str>) -> bool {
    match v {
        None => false,
        Some(s) => !matches!(s.trim(), "0" | "off" | "false" | "no" | ""),
    }
}


/// task 189: mix / combine の同期を**自ストリームだけ**にするか
/// (`DENNO_PF_MIX_STAGE`、既定 off。lab の結果で決める)。
pub(crate) fn mix_stage_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| mix_stage_from_value(std::env::var("DENNO_PF_MIX_STAGE").ok().as_deref()))
}


/// task 233: 直前の `pf_mix` の出力 (`d.mixed`) を置き場の `start` 行目へ D2D で写す。
#[cfg(feature = "cuda")]
pub fn pf_lm_park_mixed(start: usize, nt: usize, e: usize) -> bool {
    // task 318: `other` の中の D2D を分けて数える。
    let _sub = crate::model::q4ledger::sub(crate::model::q4ledger::SUB_LM_PARK);
    let bytes = nt.saturating_mul(e).saturating_mul(4);
    PF_DEV.with(|cell| -> bool {
        let d = cell.borrow();
        let Some(src) = d.mixed.as_ref() else {
            return false;
        };
        let ok = PF_LM.with(|lc| -> bool {
            let l = lc.borrow();
            let Some(dst) = l.mixed.as_ref() else {
                return false;
            };
            let off = start.saturating_mul(e).saturating_mul(4);
            if bytes == 0 || src.bytes() < bytes || dst.bytes() < off + bytes {
                return false;
            }
            match lib() {
                Ok(lib) => unsafe {
                    check(
                        (lib.memcpy_dtod_async)(
                            dst.ptr() + off as u64,
                            src.ptr(),
                            bytes,
                            std::ptr::null_mut(),
                        ),
                        "cuMemcpyDtoDAsync (lm mixed park)",
                    )
                    .is_ok()
                },
                Err(_) => false,
            }
        });
        ok
    })
}


/// task 233: 置き場の `start` 行目から `nt` 行ぶんの入力 (`[nt][e]` f32) の先頭。
#[cfg(feature = "cuda")]
pub fn pf_lm_mixed_ptr(start: usize, nt: usize, e: usize) -> Option<CUdeviceptr> {
    PF_LM.with(|cell| -> Option<CUdeviceptr> {
        let l = cell.borrow();
        let b = l.mixed.as_ref()?;
        let off = start.saturating_mul(e).saturating_mul(4);
        let bytes = nt.saturating_mul(e).saturating_mul(4);
        if bytes == 0 || b.bytes() < off + bytes {
            return None;
        }
        Some(b.ptr() + off as u64)
    })
}


/// task 233: 置き場の mixed を丸ごとホストへ戻す (デバイス経路を諦めたとき用)。
#[cfg(feature = "cuda")]
pub fn pf_lm_take_mixed(out: &mut [f32]) -> bool {
    PF_LM.with(|cell| -> bool {
        let l = cell.borrow();
        let Some(b) = l.mixed.as_ref() else {
            return false;
        };
        if b.bytes() < out.len() * 4 {
            return false;
        }
        b.to_slice(out).is_ok()
    })
}


/// task 154: デバイスの `mixed` (`[nt][n_embd]`) をホストへ戻す。
pub fn pf_take_mixed(out: &mut [f32]) -> bool {
    // task 254: `other` の中身 (`[q4ledger-pf] sub other`) に積む。
    let _sub = crate::model::q4ledger::sub(crate::model::q4ledger::SUB_MIXD2H);
    // task 189: mix 1 呼び出しの D2H (mixed の 5.1 MB) も段として積む。
    let t_d2h = pf_mix_kprof::on().then(std::time::Instant::now);
    let r = PF_DEV.with(|cell| -> bool {
        let d = cell.borrow();
        match d.mixed.as_ref() {
            Some(b) if b.bytes() >= out.len() * 4 => {
                // task 318: `DENNO_PF_PIN_STAGE` では pinned 経由 (速い)。失敗したら従来へ。
                match b.to_slice_pinned(out) {
                    Ok(true) => return true,
                    _ => {}
                }
                if pf_d2h_direct_on() {
                    // task 181: 中間 Vec を作らず直接 D2H (値は同じ)。
                    return b.to_slice(out).is_ok();
                }
                match b.to_vec::<f32>(out.len()) {
                    Ok(v) => {
                        out.copy_from_slice(&v);
                        true
                    }
                    Err(_) => false,
                }
            }
            _ => false,
        }
    });
    pf_mix_kprof::add_host(pf_mix_kprof::D2H, t_d2h);
    r
}

/// mixed 版のスレッド数 (カーネルの `PFE_MIXED_THREADS`、既存と同じ 256)。
pub(crate) const PF_ELEM_MIXED_THREADS: usize = 256;

/// これ未満のトークン数は従来カーネル (起動のほうが高い)。
pub(crate) const PF_MIX_FASTSIG_MIN_TOK: usize = 32;

/// fast sigmoid 版を使ってよい hc の上限 (inject と同じ 8)。
pub(crate) const PF_MIX_FASTSIG_MAX_HC: usize = 8;


/// `DENNO_PF_MIX_NORMP` の解釈 (純関数)。**明示的な偽値だけ off** = 既定 on。
///
/// lab t457b で採択 (mix 段の GPU busy 242.6/243.4 -> 211.4/211.7 ms = -13%)。
pub(crate) fn pf_mix_normp_from_value(v: Option<&str>) -> bool {
    match v {
        None => true,
        Some(s) => !matches!(
            s.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no" | ""
        ),
    }
}


/// norm の二乗和分担版を使うか (既定 on。切り戻しは `DENNO_PF_MIX_NORMP=0`)。
pub(crate) fn pf_mix_normp_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| pf_mix_normp_from_value(std::env::var("DENNO_PF_MIX_NORMP").ok().as_deref()))
}


/// その e を分担版で回してよいか (純関数)。
///
/// 1 スレッドのレジスタ上限 (`PF_ELEM_NORM_THREADS` x [`PF_ELEM_MAX_PER`]) に収まる
/// ことだけが条件。実形は e=2,560 (10 要素/スレッド)。
pub(crate) fn pf_mix_normp_ok(e: usize) -> bool {
    e > 0 && e <= PF_ELEM_NORM_THREADS * PF_ELEM_MAX_PER
}


/// `DENNO_PF_MIX_FASTSIG` の解釈 (純関数)。**明示的な偽値だけ off** = 既定 on。
///
/// lab t457b で採択 (mix 段の GPU busy 242.6/243.4 -> 214.8/214.6 ms = -12%)。
pub(crate) fn pf_mix_fastsig_from_value(v: Option<&str>) -> bool {
    match v {
        None => true,
        Some(s) => !matches!(
            s.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no" | ""
        ),
    }
}


/// mixed の fast sigmoid 版を使うか (既定 on。切り戻しは `DENNO_PF_MIX_FASTSIG=0`)。
pub(crate) fn pf_mix_fastsig_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        pf_mix_fastsig_from_value(std::env::var("DENNO_PF_MIX_FASTSIG").ok().as_deref())
    })
}


/// その形を fast sigmoid 版で回してよいか (純関数)。
pub(crate) fn pf_mix_fastsig_ok(e: usize, hc: usize, nt: usize) -> bool {
    e > 0 && hc >= 1 && hc <= PF_MIX_FASTSIG_MAX_HC && nt >= PF_MIX_FASTSIG_MIN_TOK
}


/// task 154: hc mix の 1 段 (norm → down → lo → up → mixed → inject) をデバイスで回し、
/// `mixed` (と inject 列) をデバイスに残す。
///
/// 成功したら `pf_take_mixed` で `[nt][n_embd]` を取れる。`false` は CPU 経路へ。
/// `res` は呼び出し側が `pf_upload_res` で入れたものを使う。
#[allow(clippy::too_many_arguments)]
pub fn pf_mix(
    norm: &[f32],
    down: PfLin<'_>,
    up: PfLin<'_>,
    inject: PfLin<'_>,
    nt: usize,
    hc: usize,
    e: usize,
    lr: usize,
    eps: f32,
) -> bool {
    if PF_BROKEN.load(std::sync::atomic::Ordering::Relaxed) {
        return false;
    }
    let hc_dim = hc * e;
    if norm.len() != hc_dim
        || down.k != hc_dim
        || down.n != lr
        || up.k != lr
        || up.n != hc_dim
        || inject.k != hc_dim
        || inject.n != hc
    {
        return pf_fail("mix の形が違う");
    }
    let Some(k) = pf_kernels() else {
        return false;
    };
    let Some(ctx) = shared_ctx() else {
        return false;
    };
    if ctx.make_current().is_err() {
        return pf_fail("cuCtxSetCurrent 失敗");
    }
    let f_bf16 = gemm_bf16_mt_fn(ctx);
    let f_f32 = gemm_f32_mt_fn(ctx);
    let want_bf16 = mix_bf16_enabled() && f_bf16.is_some();
    if f_f32.is_none() {
        return pf_fail("f32 GEMM カーネルが無い");
    }
    let Some((pd, pd_k)) = pf_weight(down, want_bf16) else {
        return false;
    };
    let Some((pu, pu_k)) = pf_weight(up, want_bf16) else {
        return false;
    };
    let Some((pi, pi_k)) = pf_weight(inject, want_bf16) else {
        return false;
    };
    if pf_mix_kprof::on() {
        pf_mix_kprof::note_call();
    }
    let r = PF_DEV.with(|cell| -> Result<(), String> {
        let d = cell.borrow();
        if nt > d.nt || hc != d.hc || e != d.e || lr != d.lr {
            return Err(format!(
                "作業域の形が違う (要求 nt={nt} hc={hc} e={e} lr={lr} / 確保 nt={} hc={} e={} lr={})",
                d.nt, d.hc, d.e, d.lr
            ));
        }
        let (Some(res), Some(xn), Some(lo), Some(gate), Some(mixed), Some(inj), Some(gamma)) = (
            d.res.as_ref(),
            d.xn.as_ref(),
            d.lo.as_ref(),
            d.gate.as_ref(),
            d.mixed.as_ref(),
            d.inj.as_ref(),
            d.gamma.as_ref(),
        ) else {
            return Err("作業域が無い".into());
        };
        // task 334: norm (gamma) は層の定数なので、デバイス常駐表から取れたら
        // H2D しない (`DENNO_PF_MIX_GAMMA_CACHE`)。取れないときは従来どおり
        // `d.gamma` へ同期コピー (同じバイト列)。
        let pnorm = match pf_norm_dev(norm) {
            Some(p) => p,
            None => {
                gamma.copy_from(norm).map_err(|e| format!("norm H2D: {e:#}"))?;
                gamma.ptr()
            }
        };
        let inv_hc = 1.0f32 / hc as f32;
        let mut rows = (nt * hc) as i32;
        let (mut pr, mut px, mut pl, mut pg, mut pm, mut pj, mut pw) = (
            res.ptr(),
            xn.ptr(),
            lo.ptr(),
            gate.ptr(),
            mixed.ptr(),
            inj.ptr(),
            pnorm,
        );
        // 1. grouped RMS norm (res → xn)
        pf_mix_kprof::begin(pf_mix_kprof::NORM);
        {
            let (mut e_c, mut hc_c) = (e as i32, hc as i32);
            let mut eps_c = eps;
            let mut args = [
                &mut pr as *mut _ as *mut c_void,
                &mut pw as *mut _ as *mut c_void,
                &mut px as *mut _ as *mut c_void,
                &mut rows as *mut _ as *mut c_void,
                &mut e_c as *mut _ as *mut c_void,
                &mut hc_c as *mut _ as *mut c_void,
                &mut eps_c as *mut _ as *mut c_void,
            ];
            // task 457: 二乗和を block で分担する版 (`DENNO_PF_MIX_NORMP`、既定 on)。
            // PTX が無い / e がレジスタ上限を超えるときは従来カーネルへ (同じ値)。
            let normp = if pf_mix_normp_on() && pf_mix_normp_ok(e) {
                pf_elem_kernels()
            } else {
                None
            };
            match normp {
                Some(fns) => {
                    let threads = PF_ELEM_NORM_THREADS as u32;
                    let grid = (rows as u32, 1, 1);
                    pf_elem_note(
                        "normp",
                        grid,
                        threads,
                        &format!("e={e} rows={rows} hc={hc} inv=tree"),
                    );
                    unsafe {
                        ctx.launch_async(fns.rmsgroup_p, grid, (threads, 1, 1), 0, &mut args)
                            .map_err(|e| format!("rmsgroup_p launch: {e:#}"))?;
                    }
                    pf_log_grid("normp", grid, threads);
                }
                None => {
                    unsafe {
                        ctx.launch_async(k.rmsgroup, (rows as u32, 1, 1), (128, 1, 1), 0, &mut args)
                            .map_err(|e| format!("rmsgroup launch: {e:#}"))?;
                    }
                    pf_log_grid("norm", (rows as u32, 1, 1), 128);
                }
            }
        }
        pf_mix_kprof::end(pf_mix_kprof::NORM);
        // 2. down GEMM (xn → lo)
        let gemm = |w: CUdeviceptr,
                    x: CUdeviceptr,
                    y: CUdeviceptr,
                    kk: usize,
                    nn: usize,
                    kind: PfWKind,
                    tag: &'static str|
         -> Result<(), String> {
            // task 195: inject (n <= 8) を k 方向のレーン分割で回す
            // (`DENNO_PF_MIX_INJ2`。丸め順は変わるが決定論的)。
            if mix_inj2_on() && mix_inj2_ok(kk, nn, nt, kind == PfWKind::Bf16) {
                if let Some(fns) = mix_inj2_kernels() {
                    // task 617: inject を k 分割で撃った回数 (発火の証拠)。
                    if tag == "inject" {
                        PF_INJ2_CALLS.fetch_add(1, Ordering::Relaxed);
                    }
                    let (mut pw, mut px, mut py) = (w, x, y);
                    let (mut nt_c, mut k_c, mut n_c) = (nt as c_int, kk as c_int, nn as c_int);
                    let mut args = [
                        &mut pw as *mut _ as *mut c_void,
                        &mut px as *mut _ as *mut c_void,
                        &mut py as *mut _ as *mut c_void,
                        &mut nt_c as *mut _ as *mut c_void,
                        &mut k_c as *mut _ as *mut c_void,
                        &mut n_c as *mut _ as *mut c_void,
                    ];
                    let grid = (nt as u32, 1, 1);
                    pf_log_grid(tag, grid, MIX_INJ2_THREADS as u32);
                    unsafe {
                        ctx.launch_async(
                            fns.gemv2,
                            grid,
                            (MIX_INJ2_THREADS as u32, 1, 1),
                            0,
                            &mut args,
                        )
                        .map_err(|e| format!("mix_inj2 gemv launch: {e:#}"))?;
                    }
                    return Ok(());
                }
            }
            // task 222: BF16 の down / up をテンソルコアの GEMM で回す
            // (`DENNO_PF_MIX_TC`。**既定 on**、lab t222b で採用)。形は実モデルで
            // down: k=10240 n=320 / up: k=320 n=10240。
            // task 301: UD-Q4_K_XL の mix は **Q8_0** なので、sn3 の Q8_0 入口へも
            // 回せるようにする (`DENNO_PF_MIX_Q8TC`、既定 off)。カーネルは同じ
            // `kernels_gemm_sn3.ptx` で、丸め (bf16) と加算順 (k 昇順) も同じ。
            let mix_sn3 = match kind {
                PfWKind::Bf16 if pf_mix_tc_on() => Some((crate::kquant::GgmlType::Bf16, kk * 2)),
                PfWKind::Quant(crate::kquant::GgmlType::Q8_0, _, rb, _) if pf_mix_q8_tc_on() => {
                    Some((crate::kquant::GgmlType::Q8_0, rb))
                }
                _ => None,
            };
            if let Some((ty, rb)) = mix_sn3 {
                if pf_tc_sn3_eligible(ty, nt, kk, nn, 1) {
                    if let Some(fns) = tc_sn3_kernels() {
                        return sn3_gemm_launch(ctx, fns, ty, w, x, y, nt, kk, nn, rb)
                            .map_err(|e| format!("mix_tc gemm launch: {e:#}"));
                    }
                }
            }
            // task 195: down / up を n 方向に細かく割るカーネル
            // (`DENNO_PF_MIX_SPLIT`。値は従来カーネルとビット一致)。
            if mix_split_on() && mix_split_ok(kk, nn, nt, kind == PfWKind::Bf16) {
                if let Some(fns) = mix_split_kernels() {
                    let (mut pw, mut px, mut py) = (w, x, y);
                    let (mut nt_c, mut k_c, mut n_c) = (nt as c_int, kk as c_int, nn as c_int);
                    let mut args = [
                        &mut pw as *mut _ as *mut c_void,
                        &mut px as *mut _ as *mut c_void,
                        &mut py as *mut _ as *mut c_void,
                        &mut nt_c as *mut _ as *mut c_void,
                        &mut k_c as *mut _ as *mut c_void,
                        &mut n_c as *mut _ as *mut c_void,
                    ];
                    let grid = (
                        nn.div_ceil(MIX_SPLIT_BN) as u32,
                        nt.div_ceil(MIX_SPLIT_BM) as u32,
                        1,
                    );
                    pf_log_grid(tag, grid, MIX_SPLIT_THREADS as u32);
                    unsafe {
                        ctx.launch_async(
                            fns.gemm_split,
                            grid,
                            (MIX_SPLIT_THREADS as u32, 1, 1),
                            0,
                            &mut args,
                        )
                        .map_err(|e| format!("mix_split gemm launch: {e:#}"))?;
                    }
                    return Ok(());
                }
            }
            // task 186: BF16 の down / up を広いタイルのカーネルで回す
            // (`DENNO_PF_MIX_FAST`・既定 off。値は従来カーネルとビット一致)。
            // task 189: inject (n <= 8) は 1 スレッド 1 出力の GEMV で回す。
            if mix_inj_on() && mix_inj_ok(kk, nn, nt, kind == PfWKind::Bf16) {
                if let Some(fns) = mix_inj_kernels() {
                    let (mut pw, mut px, mut py) = (w, x, y);
                    let (mut nt_c, mut k_c, mut n_c) = (nt as c_int, kk as c_int, nn as c_int);
                    let mut args = [
                        &mut pw as *mut _ as *mut c_void,
                        &mut px as *mut _ as *mut c_void,
                        &mut py as *mut _ as *mut c_void,
                        &mut nt_c as *mut _ as *mut c_void,
                        &mut k_c as *mut _ as *mut c_void,
                        &mut n_c as *mut _ as *mut c_void,
                    ];
                    let grid = (((nt * nn).div_ceil(MIX_INJ_THREADS)) as u32, 1, 1);
                    pf_log_grid(tag, grid, MIX_INJ_THREADS as u32);
                    unsafe {
                        ctx.launch_async(
                            fns.gemv,
                            grid,
                            (MIX_INJ_THREADS as u32, 1, 1),
                            0,
                            &mut args,
                        )
                        .map_err(|e| format!("mix_inj gemv launch: {e:#}"))?;
                    }
                    return Ok(());
                }
            }
            let fast_bf16 = kind == PfWKind::Bf16 && mix_pf_fast_on() && mix_pf_ok(kk, nn, nt, true);
            let fast_fn = if fast_bf16 { mix_pf_kernels() } else { None };
            if let Some(fns) = fast_fn {
                let (mut pw, mut px, mut py) = (w, x, y);
                let (mut nt_c, mut k_c, mut n_c) = (nt as c_int, kk as c_int, nn as c_int);
                let mut args = [
                    &mut pw as *mut _ as *mut c_void,
                    &mut px as *mut _ as *mut c_void,
                    &mut py as *mut _ as *mut c_void,
                    &mut nt_c as *mut _ as *mut c_void,
                    &mut k_c as *mut _ as *mut c_void,
                    &mut n_c as *mut _ as *mut c_void,
                ];
                let grid = (
                    ((nn + MIX_PF_BN - 1) / MIX_PF_BN) as u32,
                    ((nt + MIX_PF_BM - 1) / MIX_PF_BM) as u32,
                    1,
                );
                pf_log_grid(tag, grid, MIX_PF_THREADS as u32);
                unsafe {
                    ctx.launch_async(fns.gemm, grid, (MIX_PF_THREADS as u32, 1, 1), 0, &mut args)
                        .map_err(|e| format!("mix_pf gemm launch: {e:#}"))?;
                }
                return Ok(());
            }
            match kind {
                PfWKind::Bf16 | PfWKind::F32 => {
                    let f = if kind == PfWKind::Bf16 {
                        f_bf16.ok_or_else(|| "bf16 GEMM カーネルが無い".to_string())?
                    } else {
                        f_f32.ok_or_else(|| "f32 GEMM カーネルが無い".to_string())?
                    };
                    let (mut pw, mut px, mut py) = (w, x, y);
                    let (mut nt_c, mut k_c, mut n_c) = (nt as c_int, kk as c_int, nn as c_int);
                    let mut args = [
                        &mut pw as *mut _ as *mut c_void,
                        &mut px as *mut _ as *mut c_void,
                        &mut py as *mut _ as *mut c_void,
                        &mut nt_c as *mut _ as *mut c_void,
                        &mut k_c as *mut _ as *mut c_void,
                        &mut n_c as *mut _ as *mut c_void,
                    ];
                    let grid = (
                        ((nn + GEMM_MT_ROWS - 1) / GEMM_MT_ROWS) as u32,
                        ((nt + GEMM_MT_TOKS - 1) / GEMM_MT_TOKS) as u32,
                        1,
                    );
                    pf_log_grid(tag, grid, GEMM_MT_ROWS as u32);
                    unsafe {
                        ctx.launch_async(f, grid, (GEMM_MT_ROWS as u32, 16, 1), 0, &mut args)
                            .map_err(|e| format!("gemm launch: {e:#}"))?;
                    }
                }
                PfWKind::Quant(_ty, slot, rb, blk) => {
                    // 量子 GEMM は 1 起動 128 トークンまで (`moe_gemm_dev` と同じ)。
                    let f = ctx
                        .f_gemm_q
                        .ok_or_else(|| "量子 GEMM カーネルが無い".to_string())?[slot];
                    const ROWS: usize = 16;
                    const NTG_MAX: usize = 8;
                    if kk % blk != 0 {
                        return Err("量子 GEMM のブロック形が合わない".into());
                    }
                    let shmem = (ROWS * blk * 4) as u32;
                    let mut t0 = 0usize;
                    while t0 < nt {
                        let m = (nt - t0).min(ROWS * NTG_MAX);
                        let xo = (t0 * kk * 4) as u64;
                        let yo = (t0 * nn * 4) as u64;
                        let (mut pw, mut px, mut py) = (w, x + xo, y + yo);
                        let (mut m_c, mut ni_c, mut no_c, mut rb_c) =
                            (m as c_int, kk as c_int, nn as c_int, rb as c_int);
                        let mut args = [
                            &mut pw as *mut _ as *mut c_void,
                            &mut px as *mut _ as *mut c_void,
                            &mut py as *mut _ as *mut c_void,
                            &mut m_c as *mut _ as *mut c_void,
                            &mut ni_c as *mut _ as *mut c_void,
                            &mut no_c as *mut _ as *mut c_void,
                            &mut rb_c as *mut _ as *mut c_void,
                        ];
                        let grid = (((nn + ROWS - 1) / ROWS) as u32, 1, 1);
                        unsafe {
                            ctx.launch_async(f, grid, (ROWS as u32, 16, 1), shmem, &mut args)
                                .map_err(|e| format!("q gemm launch: {e:#}"))?;
                        }
                        t0 += m;
                    }
                }
            }
            Ok(())
        };
        pf_mix_kprof::begin(pf_mix_kprof::DOWN);
        gemm(pd, px, pl, hc_dim, lr, pd_k, "down")?;
        pf_mix_kprof::end(pf_mix_kprof::DOWN);
        // 3. lo apply (in-place, nt*low_rank 要素)
        pf_mix_kprof::begin(pf_mix_kprof::LO);
        {
            let mut total = (nt * lr) as i64;
            let mut inv_c = inv_hc;
            let grid = ((nt * lr).div_ceil(256)) as u32;
            let mut args = [
                &mut pl as *mut _ as *mut c_void,
                &mut total as *mut _ as *mut c_void,
                &mut inv_c as *mut _ as *mut c_void,
            ];
            unsafe {
                ctx.launch_async(k.lo, (grid, 1, 1), (256, 1, 1), 0, &mut args)
                    .map_err(|e| format!("lo launch: {e:#}"))?;
            }
            pf_log_grid("lo", (grid, 1, 1), 256);
        }
        pf_mix_kprof::end(pf_mix_kprof::LO);
        // 4. up GEMM (lo → gate)
        pf_mix_kprof::begin(pf_mix_kprof::UP);
        gemm(pu, pl, pg, lr, hc_dim, pu_k, "up")?;
        pf_mix_kprof::end(pf_mix_kprof::UP);
        // 5. mixed (xn, gate → mixed)
        pf_mix_kprof::begin(pf_mix_kprof::MIXED);
        {
            let (mut e_c, mut hc_c) = (e as i32, hc as i32);
            let mut inv_c = inv_hc;
            let grid = ((e.div_ceil(256)) as u32, nt as u32, 1);
            let mut args = [
                &mut px as *mut _ as *mut c_void,
                &mut pg as *mut _ as *mut c_void,
                &mut pm as *mut _ as *mut c_void,
                &mut e_c as *mut _ as *mut c_void,
                &mut hc_c as *mut _ as *mut c_void,
                &mut inv_c as *mut _ as *mut c_void,
            ];
            // task 457: sigmoid を f32 の速い経路にした版 (`DENNO_PF_MIX_FASTSIG`、既定 on)。
            // 値は 2 ULP 程度変わる (丸めの変更)。形が外れたら従来カーネルへ。
            let fastsig = if pf_mix_fastsig_on() && pf_mix_fastsig_ok(e, hc, nt) {
                pf_elem_kernels()
            } else {
                None
            };
            match fastsig {
                Some(fns) => {
                    let threads = PF_ELEM_MIXED_THREADS as u32;
                    pf_elem_note(
                        "mixed_fs",
                        grid,
                        threads,
                        &format!("e={e} hc={hc} nt={nt} sigmoid=__expf"),
                    );
                    unsafe {
                        ctx.launch_async(fns.mixed_fs, grid, (threads, 1, 1), 0, &mut args)
                            .map_err(|e| format!("mixed_fs launch: {e:#}"))?;
                    }
                    pf_log_grid("mixed_fs", grid, threads);
                }
                None => {
                    unsafe {
                        ctx.launch_async(k.mixed, grid, (256, 1, 1), 0, &mut args)
                            .map_err(|e| format!("mixed launch: {e:#}"))?;
                    }
                    pf_log_grid("mixed", grid, 256);
                }
            }
        }
        pf_mix_kprof::end(pf_mix_kprof::MIXED);
        // 6. inject GEMM (xn → inj) + 1/hc
        pf_mix_kprof::begin(pf_mix_kprof::INJ);
        gemm(pi, px, pj, hc_dim, hc, pi_k, "inject")?;
        {
            let mut n_c = (nt * hc) as i32;
            let mut inv_c = inv_hc;
            let mut args = [
                &mut pj as *mut _ as *mut c_void,
                &mut n_c as *mut _ as *mut c_void,
                &mut inv_c as *mut _ as *mut c_void,
            ];
            let grid = ((nt * hc).div_ceil(256)) as u32;
            unsafe {
                ctx.launch_async(k.inj, (grid, 1, 1), (256, 1, 1), 0, &mut args)
                    .map_err(|e| format!("inj launch: {e:#}"))?;
            }
            pf_log_grid("inj_scale", (grid, 1, 1), 256);
        }
        pf_mix_kprof::end(pf_mix_kprof::INJ);
        // task 189: 同期は `DENNO_PF_MIX_STAGE` で自ストリームだけにできる。
        // task 334: `DENNO_PF_STAGE_NOSYNC` なら待たない (次の段は同じストリーム)。
        let t_other = pf_stage_sync_maybe(ctx).map_err(|e| format!("sync: {e:#}"))?;
        pf_mix_kprof::add_host(pf_mix_kprof::SYNC_OTHER, t_other);
        pf_mix_kprof::collect();
        Ok(())
    });
    match r {
        Ok(()) => true,
        Err(e) => pf_fail(&e),
    }
}


// ---------------------------------------------------------------------------
// task 218: 残差の平均の記録 / ステアリング (`kernels_steer.cu`)
// ---------------------------------------------------------------------------
//
// 常駐 prefill (`DENNO_PF_RESIDENT`) では残差がデバイスに載ったまま層を回る。
// 記録 (層の出口の足し込み) と適用 (層の出口の引き算) もデバイスで閉じる。
// 層ごとに残差を D2H しない。
//
// PTX はクレートの `kernels/kernels_steer.ptx` (lab が `bash build_ptx.sh steer`
// で生成) を**実行時に読む**。無い・読めない・entry が無いときは
// [`pf_steer_acc_ensure`] / [`pf_steer_apply_begin`] が false を返し、
// 呼び出し側は常駐連鎖を諦めてホスト経路 (host 側の記録 / 適用) へ落ちる。
// 値は変わらない (加算順は同じ)。

/// `kernels_steer.cu` の関数一式。
pub(crate) struct SteerKernels {
    #[allow(dead_code)]
    module: CUmodule,
    pub(crate) acc: CUfunction,
    pub(crate) sub: CUfunction,
    pub(crate) proj: CUfunction,
}


// 生ハンドル (PfKernels と同じ規約)。
unsafe impl Send for SteerKernels {}

unsafe impl Sync for SteerKernels {}


const STEER_NAMES: [&[u8]; 3] = [
    b"dno_steer_acc\0",
    b"dno_steer_sub\0",
    b"dno_steer_proj\0",
];


/// PTX の置き場所。`DENNO_STEER_PTX` があればそれ、無ければクレートの
/// `kernels/kernels_steer.ptx`。
///
/// task 218: 収録した PTX は lab t218b (sha bfb4b2fe1) が `PTX=steer` で
/// `kernels_steer.cu` から生成したもの (sha256 eae1eca4…、entry 3 本、
/// `.target sm_75`)。再生成は `bash kernels/build_ptx.sh steer`。
/// 引数の数は `kernel_invariants::steer_ptx_matches_the_host_launch_abi` が
/// 固定している (acc 6 / sub 7 / proj 7)。
fn steer_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_STEER_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_steer.ptx"),
    }
}


/// カーネルを取得する (初回だけ PTX を読む)。失敗は None を焼き付ける。
/// **current context を張った後に呼ぶこと**。
pub(crate) fn steer_kernels() -> Option<&'static SteerKernels> {
    static K: OnceLock<Option<SteerKernels>> = OnceLock::new();
    K.get_or_init(|| {
        let path = steer_ptx_path();
        let mut image = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                tracing::info!(
                    "[steer] PTX が読めない ({}) : {e} → ホスト側で記録/適用",
                    path.display()
                );
                return None;
            }
        };
        image.push(0);
        let lib = lib().ok()?;
        let mut module: CUmodule = std::ptr::null_mut();
        unsafe {
            if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void) != CUDA_SUCCESS {
                tracing::info!("[steer] cuModuleLoadData 失敗 → ホスト側で記録/適用");
                return None;
            }
        }
        let mut fns: [CUfunction; 3] = [std::ptr::null_mut(); 3];
        for (i, name) in STEER_NAMES.iter().enumerate() {
            unsafe {
                if (lib.module_get_function)(&mut fns[i], module, name.as_ptr() as *const c_char)
                    != CUDA_SUCCESS
                {
                    tracing::info!(
                        "[steer] PTX に {} が無い → ホスト側で記録/適用",
                        String::from_utf8_lossy(&name[..name.len() - 1])
                    );
                    return None;
                }
            }
        }
        Some(SteerKernels {
            module,
            acc: fns[0],
            sub: fns[1],
            proj: fns[2],
        })
    })
    .as_ref()
}


/// デバイス側の作業域 (加算域は複数リクエストに跨って残す)。
pub(crate) struct SteerDev {
    pub(crate) acc: Option<DevBuffer>, // [n_layer][hc][e]
    pub(crate) v: Option<DevBuffer>,   // [n_layer][hc][e]
    n_layer: usize,
    hc: usize,
    e: usize,
}


impl SteerDev {
    pub(crate) const fn new() -> Self {
        Self {
            acc: None,
            v: None,
            n_layer: 0,
            hc: 0,
            e: 0,
        }
    }

    /// 形が変わったら両方捨てる (古い領域を流用しない)。
    pub(crate) fn fit(&mut self, n_layer: usize, hc: usize, e: usize) -> Result<()> {
        if self.n_layer != n_layer || self.hc != hc || self.e != e {
            self.acc = None;
            self.v = None;
            self.n_layer = n_layer;
            self.hc = hc;
            self.e = e;
        }
        Ok(())
    }
}


thread_local! {
    pub(crate) static STEER_DEV: std::cell::RefCell<SteerDev> =
        const { std::cell::RefCell::new(SteerDev::new()) };
}


/// task 152: BF16 のまま読む mix カーネルの既定値 (**on**)。
///
/// lab t152b の g22 条件で採否 OK (`REPORT_P152.md`)。`w:mix_f32` 2,425 MiB ->
/// `w:mix_bf16` 1,213 MiB、md5 は全条件 SAME (f32 版と同じ演算列)。
/// `DENNO_MIX_BF16=0` で従来の f32 展開経路に戻せる。
pub(crate) const MIX_BF16_DEFAULT: bool = true;


/// `MIX_GPU_W` の登録・形状検査・初回の同期 H2D を 1 箇所にまとめたもの
/// ([`maybe_gpu_mix_bf16_into`] と同じ規約)。uid は `GgufLinear` のもの。
///
/// task 152: `want_bf16` のときは生バイト (BF16) をそのまま上げる。既に逆の
/// モードで載っている uid を要求されたら **エラー**にする (黙って別解釈の
/// バイト列で計算しない。呼び出し側は CPU へ落ちる)。
pub(crate) fn mix_gpu_weight(uid: u64, bytes: &[u8], k: usize, n: usize, want_bf16: bool) -> Result<MixW> {
    let reg = MIX_GPU_W.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let mut map = match reg.lock() {
        Ok(g) => g,
        Err(_) => bail!("hc mix weight registry poisoned"),
    };
    match map.get(&uid) {
        Some(w) if w.k == k && w.n == n && w.bf16 == want_bf16 => Ok(MixW {
            ptr: w.buf.ptr(),
        }),
        Some(_) => bail!("uid 重複: uid={uid} に別の形状/モード (k={k}, n={n}, bf16={want_bf16})"),
        None => {
            let (buf, tag) = if want_bf16 {
                match DevBuffer::from_slice_tagged(bytes, crate::mem_audit::TAG_MIX_BF16) {
                    Ok(b) => (b, crate::mem_audit::TAG_MIX_BF16),
                    Err(e) => bail!("BF16 のままの mix 重みを VRAM に置けない: {e:#}"),
                }
            } else {
                let host = mix_expand_bf16(bytes, k, n);
                match DevBuffer::from_slice_tagged(&host, crate::mem_audit::TAG_MIX_F32) {
                    Ok(b) => (b, crate::mem_audit::TAG_MIX_F32),
                    Err(e) => bail!("f32 展開重みの VRAM 確保に失敗: {e:#}"),
                }
            };
            tracing::info!(
                "[mix-gpu] uid={uid} bf16 mix k={k} n={n} {:.1} MB ({}・常駐)",
                buf.bytes() as f64 / 1e6,
                if want_bf16 { "BF16 のまま" } else { "f32 展開済み" }
            );
            let p = buf.ptr();
            debug_assert_eq!(
                tag,
                if want_bf16 {
                    crate::mem_audit::TAG_MIX_BF16
                } else {
                    crate::mem_audit::TAG_MIX_F32
                }
            );
            map.insert(uid, MixGpuW { buf, k, n, bf16: want_bf16 });
            Ok(MixW { ptr: p })
        }
    }
}


/// task 138: `DENNO_MIX_UP_REDUCE` の値を解釈する (純関数・テスト用)。
/// **既定 on** で、明示的な偽値だけ切る (切ると逐次版 = 従来の丸め順)。
pub(crate) fn mix_up_reduce_from_value(v: Option<&str>) -> bool {
    match v {
        Some(s) => !matches!(
            s.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        ),
        None => true,
    }
}


/// task 138: チェーン経路の up GEMV に並列リダクション版 (`dno_matvec_f32_r`)
/// を使う (`DENNO_MIX_UP_REDUCE`、**既定 on**。`=0` で逐次版に戻る)。
/// up の形は k < n (320 x 5120) で、逐次版は 1 ブロック 1 スレッドのため
/// 並列度が n 側にしか無い。lab t138b の実測で mix の同期待ちが
/// 6.42 → 4.65 ms/tok になった (丸め順が変わるが決定論は保つ)。
fn mix_up_reduce_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        mix_up_reduce_from_value(std::env::var("DENNO_MIX_UP_REDUCE").ok().as_deref())
    })
}


/// task 138: hc mix (BF16) の **down → lo_apply → up を 1 回の同期に畳む**。
///
/// 従来は down GEMV (H2D + kernel + D2H + sync)、ホストで `hc_lo_apply`、
/// up GEMV (H2D + kernel + D2H + sync) の 2 往復だった。down の出力
/// (`low_rank` 本) はそのまま lo_apply の入力なので、silu を畳んだ
/// [`matvec_f32_r_silu`] (`dno_matvec_f32_r_silu`) で down を書き、その出力を
/// デバイス上で up に渡す。丸めは GPU 経路の現行と同じ (dno_expf は glibc expf
/// のビット一致実装) なので logits は従来経路とビット一致する。
///
/// 戻り値 `false` = 従来の 2 往復経路を使うこと (GPU 無効・PTX 未収録・形状不一致・
/// 確保失敗・実行エラー)。`x` は `[k1]`、`out` は `[n2]`。
#[allow(clippy::too_many_arguments)]
pub fn maybe_gpu_mix_bf16_chain_into(
    down_uid: u64,
    down_bytes: &[u8],
    k1: usize,
    n1: usize,
    up_uid: u64,
    up_bytes: &[u8],
    _k2: usize,
    n2: usize,
    inv_hc: f32,
    x: &[f32],
    out: &mut [f32],
) -> bool {
    use std::sync::atomic::Ordering as AO;
    if !cuda_enabled() || MIX_GPU_BROKEN.load(AO::Relaxed) {
        return false;
    }
    if k1 == 0
        || n1 == 0
        || n2 == 0
        || n1 != _k2
        || x.len() != k1
        || out.len() != n2
        || k1 < n1
        || down_bytes.len() != k1.saturating_mul(n1).saturating_mul(2)
        || up_bytes.len() != n1.saturating_mul(n2).saturating_mul(2)
        || k1 > i32::MAX as usize
        || n1 > i32::MAX as usize
        || n2 > i32::MAX as usize
    {
        return false;
    }
    let Some(ctx) = shared_ctx() else {
        return false;
    };
    if let Err(e) = ctx.make_current() {
        mix_gpu_broken_once(&format!("cuCtxSetCurrent: {e:#}"));
        return false;
    }
    // down は silu 込みの並列リダクション版。未収録 PTX は従来経路へ (ラッチしない)。
    // task 152: BF16 カーネルがあれば重みは BF16 のまま (常駐が半分)。
    let want_bf16 = mix_bf16_enabled() && matvec_bf16_r_silu_fn(ctx).is_some();
    let f_down = if want_bf16 {
        matvec_bf16_r_silu_fn(ctx)
    } else {
        matvec_f32_r_silu_fn(ctx)
    };
    let Some(f_down) = f_down else {
        return false;
    };
    let up_reduce = mix_up_reduce_enabled();
    let f_up = if up_reduce {
        if want_bf16 {
            matvec_bf16_r_fn(ctx)
        } else {
            matvec_f32_r_fn(ctx)
        }
    } else if want_bf16 {
        matvec_bf16_fn(ctx)
    } else {
        matvec_f32_fn(ctx)
    };
    let Some(f_up) = f_up else {
        return false;
    };
    let up_thr: u32 = if up_reduce { 256 } else { 1 };
    let wp1 = match mix_gpu_weight(down_uid, down_bytes, k1, n1, want_bf16) {
        Ok(w) => w.ptr,
        Err(e) => {
            mix_gpu_broken_once(&format!("down 重みの常駐に失敗: {e:#}"));
            return false;
        }
    };
    let wp2 = match mix_gpu_weight(up_uid, up_bytes, n1, n2, want_bf16) {
        Ok(w) => w.ptr,
        Err(e) => {
            mix_gpu_broken_once(&format!("up 重みの常駐に失敗: {e:#}"));
            return false;
        }
    };
    let lib = match lib() {
        Ok(l) => l,
        Err(e) => {
            mix_gpu_broken_once(&format!("CUDA lib load failed: {e:#}"));
            return false;
        }
    };
    let r = MIX_GPU_SC.with(|cell| -> Result<()> {
        let mut s = cell.borrow_mut();
        let MixGpuScratch {
            x: dx,
            lo: dlo,
            out: dout,
            mixed: _,
            hx,
            ho,
            ..
        } = &mut *s;
        let bytes_x = k1 * 4;
        let bytes_lo = n1 * 4;
        let bytes_out = n2 * 4;
        Scratch::ensure(dx, bytes_x)?;
        Scratch::ensure(dlo, bytes_lo)?;
        Scratch::ensure(dout, bytes_out)?;
        Scratch::ensure_pin(hx, bytes_x)?;
        Scratch::ensure_pin(ho, bytes_out)?;
        let (dx, dlo, dout) = (
            dx.as_ref().unwrap(),
            dlo.as_ref().unwrap(),
            dout.as_ref().unwrap(),
        );
        let (hx, ho) = (hx.as_mut().unwrap(), ho.as_mut().unwrap());
        hx.as_mut_slice::<f32>()[..k1].copy_from_slice(x);
        unsafe {
            check(
                (lib.memcpy_htod_async)(
                    dx.ptr(),
                    hx.ptr as *const c_void,
                    bytes_x,
                    std::ptr::null_mut(),
                ),
                "cuMemcpyHtoDAsync (mix chain)",
            )?;
            // down + lo_apply (silu 込み) → lo はデバイスに残る。
            let (mut ph, mut pw, mut po) = (dx.ptr(), wp1, dlo.ptr());
            let (mut kc, mut nc) = (k1 as c_int, n1 as c_int);
            let mut invc = inv_hc;
            let mut args = [
                &mut ph as *mut _ as *mut c_void,
                &mut pw as *mut _ as *mut c_void,
                &mut po as *mut _ as *mut c_void,
                &mut kc as *mut _ as *mut c_void,
                &mut nc as *mut _ as *mut c_void,
                &mut invc as *mut _ as *mut c_void,
            ];
            ctx.launch_async(f_down, (n1 as u32, 1, 1), (256, 1, 1), 0, &mut args)?;
            // up: 入力は lo_apply 済みの lo (同じ既定ストリームなので順序保証)。
            let (mut ph2, mut pw2, mut po2) = (dlo.ptr(), wp2, dout.ptr());
            let (mut kc2, mut nc2) = (n1 as c_int, n2 as c_int);
            let mut args2 = [
                &mut ph2 as *mut _ as *mut c_void,
                &mut pw2 as *mut _ as *mut c_void,
                &mut po2 as *mut _ as *mut c_void,
                &mut kc2 as *mut _ as *mut c_void,
                &mut nc2 as *mut _ as *mut c_void,
            ];
            ctx.launch_async(
                f_up,
                (n2 as u32, 1, 1),
                (up_thr, 1, 1),
                0,
                &mut args2,
            )?;
            check(
                (lib.memcpy_dtoh_async)(ho.ptr, dout.ptr(), bytes_out, std::ptr::null_mut()),
                "cuMemcpyDtoHAsync (mix chain)",
            )?;
            crate::prof::mark_d2h_src(crate::prof::D2H_SRC_LINEAR);
        }
        ctx.sync()?;
        out.copy_from_slice(&ho.as_slice::<f32>()[..n2]);
        Ok(())
    });
    match r {
        Ok(()) => {
            set_last_linear_ran_on_gpu(true);
            true
        }
        Err(e) => {
            mix_gpu_broken_once(&format!("hc mix chain failed: {e:#}"));
            false
        }
    }
}


/// task 150: `dno_hc_mixed_f32` (hc_mixed の device 版) を PTX から取る。
/// 未収録 (旧 PTX) は `None` を返し、呼び出し側は従来経路へ落ちる (ラッチしない)。
fn hc_mixed_f32_fn(ctx: &CudaCtx) -> Option<CUfunction> {
    struct HcMixedFn(CUfunction);
    unsafe impl Send for HcMixedFn {}
    unsafe impl Sync for HcMixedFn {}
    static FN: OnceLock<Option<HcMixedFn>> = OnceLock::new();
    FN.get_or_init(|| {
        let lib = lib().ok()?;
        let mut f: CUfunction = std::ptr::null_mut();
        let r = unsafe {
            (lib.module_get_function)(
                &mut f,
                ctx.module_elem,
                b"dno_hc_mixed_f32\0".as_ptr() as *const c_char,
            )
        };
        if r != 0 {
            None
        } else {
            Some(HcMixedFn(f))
        }
    })
    .as_ref()
    .map(|h| h.0)
}


/// task 150: hc mix の **down → lo_apply → up → hc_mixed** を 1 回の同期に畳む。
///
/// task 138 の [`maybe_gpu_mix_bf16_chain_into`] は up までを GPU で閉じ、`hc_mixed`
/// (ストリームごとの sigmoid と加算) はホストで計算していた。ここでは `hc_mixed` も
/// device の `dno_hc_mixed_f32` で閉じ、ホストへ返すのは `mixed[e]` だけにする
/// (従来は `gate[hc*e]` を返してホストで計算 = D2H も 4 倍)。
///
/// 丸めは CPU の `hyper::hc_mixed_into_par` と同一 (dno_expf + `__f*_rn`・要素独立)
/// なので値はビット一致する。戻り値 `false` = 従来経路を使うこと (GPU 無効・PTX
/// 未収録・形状不一致・確保失敗・実行エラー)。
///
/// `x` は `[k1]` (= `[hc][e]`)、`out` は `[e]`。
#[allow(clippy::too_many_arguments)]
pub fn maybe_gpu_mix_bf16_chain2_into(
    down_uid: u64,
    down_bytes: &[u8],
    k1: usize,
    n1: usize,
    up_uid: u64,
    up_bytes: &[u8],
    k2: usize,
    n2: usize,
    inv_hc: f32,
    e: usize,
    x: &[f32],
    out: &mut [f32],
) -> bool {
    use std::sync::atomic::Ordering as AO;
    if !cuda_enabled() || MIX_GPU_BROKEN.load(AO::Relaxed) {
        return false;
    }
    if k1 == 0
        || n1 == 0
        || n2 == 0
        || e == 0
        || k2 != n1
        || n2 % e != 0
        || n2 / e > 64
        || x.len() != k1
        || out.len() != e
        || n2 != k1
        || k1 < n1
        || down_bytes.len() != k1.saturating_mul(n1).saturating_mul(2)
        || up_bytes.len() != n1.saturating_mul(n2).saturating_mul(2)
        || k1 > i32::MAX as usize
        || n1 > i32::MAX as usize
        || n2 > i32::MAX as usize
        || e > i32::MAX as usize
    {
        return false;
    }
    let hc = n2 / e;
    let Some(ctx) = shared_ctx() else {
        return false;
    };
    if let Err(e) = ctx.make_current() {
        mix_gpu_broken_once(&format!("cuCtxSetCurrent: {e:#}"));
        return false;
    }
    let Some(f_mixed) = hc_mixed_f32_fn(ctx) else {
        return false;
    };
    // task 152: BF16 カーネルがあれば重みは BF16 のまま (常駐が半分)。
    let want_bf16 = mix_bf16_enabled() && matvec_bf16_r_silu_fn(ctx).is_some();
    let f_down = if want_bf16 {
        matvec_bf16_r_silu_fn(ctx)
    } else {
        matvec_f32_r_silu_fn(ctx)
    };
    let Some(f_down) = f_down else {
        return false;
    };
    let up_reduce = mix_up_reduce_enabled();
    let f_up = if up_reduce {
        if want_bf16 {
            matvec_bf16_r_fn(ctx)
        } else {
            matvec_f32_r_fn(ctx)
        }
    } else if want_bf16 {
        matvec_bf16_fn(ctx)
    } else {
        matvec_f32_fn(ctx)
    };
    let Some(f_up) = f_up else {
        return false;
    };
    let up_thr: u32 = if up_reduce { 256 } else { 1 };
    let wp1 = match mix_gpu_weight(down_uid, down_bytes, k1, n1, want_bf16) {
        Ok(w) => w.ptr,
        Err(e) => {
            mix_gpu_broken_once(&format!("down 重みの常駐に失敗: {e:#}"));
            return false;
        }
    };
    let wp2 = match mix_gpu_weight(up_uid, up_bytes, n1, n2, want_bf16) {
        Ok(w) => w.ptr,
        Err(e) => {
            mix_gpu_broken_once(&format!("up 重みの常駐に失敗: {e:#}"));
            return false;
        }
    };
    let lib = match lib() {
        Ok(l) => l,
        Err(e) => {
            mix_gpu_broken_once(&format!("CUDA lib load failed: {e:#}"));
            return false;
        }
    };
    const MIXED_THREADS: usize = 256;
    let r = MIX_GPU_SC.with(|cell| -> Result<()> {
        // task 177: mix 1 回の操作別内訳 (`DENNO_GDN_DEC_KPROF`・既定 off)。
        // decode の mix 本線はこの chain2 経路 (down + lo_apply → up → hc_mixed)。
        let _kp_win = dec_kprof::window(
            dec_kprof::MIX,
            dec_kprof::cur_layer(),
            dec_kprof::cur_pos(),
            1,
        );
        let mut s = cell.borrow_mut();
        // task 307: 鎖の末尾に積むルータ GEMV の重み (同期の後まで生かす)。
        let mut router_w: Option<GgufWRef> = None;
        let MixGpuScratch {
            x: dx,
            lo: dlo,
            out: dout,
            mixed: dm,
            hx,
            ho,
            ..
        } = &mut *s;
        Scratch::ensure(dx, k1 * 4)?;
        Scratch::ensure(dlo, n1 * 4)?;
        Scratch::ensure(dout, n2 * 4)?;
        Scratch::ensure(dm, e * 4)?;
        Scratch::ensure_pin(hx, k1 * 4)?;
        Scratch::ensure_pin(ho, e * 4)?;
        let (dx, dlo, dout, dm) = (
            dx.as_ref().unwrap(),
            dlo.as_ref().unwrap(),
            dout.as_ref().unwrap(),
            dm.as_ref().unwrap(),
        );
        let (hx, ho) = (hx.as_mut().unwrap(), ho.as_mut().unwrap());
        dec_kprof::step("mix_h2d_pack", || {
            hx.as_mut_slice::<f32>()[..k1].copy_from_slice(x)
        });
        let kp_cp = dec_kprof::op_begin(
            || "H2D@a".to_string(),
            (0, 0, 0),
            (0, 0, 0),
            0,
            k1 * 4,
        );
        unsafe {
            check(
                (lib.memcpy_htod_async)(
                    dx.ptr(),
                    hx.ptr as *const c_void,
                    k1 * 4,
                    std::ptr::null_mut(),
                ),
                "cuMemcpyHtoDAsync (mix chain2)",
            )?;
            // down + lo_apply (silu 込み) → lo はデバイスに残る。
            let (mut ph, mut pw, mut po) = (dx.ptr(), wp1, dlo.ptr());
            let (mut kc, mut nc) = (k1 as c_int, n1 as c_int);
            let mut invc = inv_hc;
            let mut args = [
                &mut ph as *mut _ as *mut c_void,
                &mut pw as *mut _ as *mut c_void,
                &mut po as *mut _ as *mut c_void,
                &mut kc as *mut _ as *mut c_void,
                &mut nc as *mut _ as *mut c_void,
                &mut invc as *mut _ as *mut c_void,
            ];
            ctx.launch_async(f_down, (n1 as u32, 1, 1), (256, 1, 1), 0, &mut args)?;
            // up: 入力は lo_apply 済みの lo (同じ既定ストリームなので順序保証)。
            let (mut ph2, mut pw2, mut po2) = (dlo.ptr(), wp2, dout.ptr());
            let (mut kc2, mut nc2) = (n1 as c_int, n2 as c_int);
            let mut args2 = [
                &mut ph2 as *mut _ as *mut c_void,
                &mut pw2 as *mut _ as *mut c_void,
                &mut po2 as *mut _ as *mut c_void,
                &mut kc2 as *mut _ as *mut c_void,
                &mut nc2 as *mut _ as *mut c_void,
            ];
            ctx.launch_async(f_up, (n2 as u32, 1, 1), (up_thr, 1, 1), 0, &mut args2)?;
            // hc_mixed: xn (= dx) と gate (= dout) から mixed を作る。
            let (mut px, mut pg, mut pm) = (dx.ptr(), dout.ptr(), dm.ptr());
            let (mut ec, mut hcc) = (e as c_int, hc as c_int);
            let mut invc2 = inv_hc;
            let mut args3 = [
                &mut px as *mut _ as *mut c_void,
                &mut pg as *mut _ as *mut c_void,
                &mut pm as *mut _ as *mut c_void,
                &mut ec as *mut _ as *mut c_void,
                &mut hcc as *mut _ as *mut c_void,
                &mut invc2 as *mut _ as *mut c_void,
            ];
            let grid = (e.div_ceil(MIXED_THREADS).max(1) as u32, 1, 1);
            ctx.launch_async(
                f_mixed,
                grid,
                (MIXED_THREADS as u32, 1, 1),
                0,
                &mut args3,
            )?;
            // task 307: ルータ GEMV を鎖の末尾に 1 発積む (D2H は下の mixed と
            // 同じ 1 回の同期でまとめる)。載らなければ 0 = 従来経路。
            let _ = router_in_mix_enqueue(ctx, dm.ptr(), e, &mut router_w);
            let kp_cp = dec_kprof::op_begin(
                || "D2H@a".to_string(),
                (0, 0, 0),
                (0, 0, 0),
                0,
                e * 4,
            );
            check(
                (lib.memcpy_dtoh_async)(ho.ptr, dm.ptr(), e * 4, std::ptr::null_mut()),
                "cuMemcpyDtoHAsync (mix chain2)",
            )?;
            dec_kprof::op_end(kp_cp);
            crate::prof::mark_d2h_src(crate::prof::D2H_SRC_LINEAR);
        }
        ctx.sync()?;
        drop(router_w);
        dec_kprof::step("mix_pack", || {
            out.copy_from_slice(&ho.as_slice::<f32>()[..e])
        });
        Ok(())
    });
    match r {
        Ok(()) => {
            set_last_linear_ran_on_gpu(true);
            true
        }
        Err(e) => {
            mix_gpu_broken_once(&format!("hc mix chain2 failed: {e:#}"));
            false
        }
    }
}


/// task 282: `DENNO_MIX_QK` — 量子型 mix の fused chain (task 266) の中で
/// **1 行 = 1 warp の高速 GEMV** (`kernels_seg_k.ptx`) を使う。
///
/// なぜ要るか: chain の GEMV は `plan_moe_gemv` の v2 形 (1 行 = 1 スレッドの
/// 直列累積) で、UD-Q4_K_XL の mix (down 10240→320 / up 320→10240) は
/// **1 呼び出し 0.14 ms** かかっていた (lab t266r1b の `[q4prof]` mix=0.279 ms/層、
/// 1 トークン 96 呼び出しで 13.4 ms)。
///
/// 値は `both` (既定) / `0` / `down` / `up`。**向きごとに切り替えられる**のは、
/// lab t282a で「両方 seg_k」が旧 GEMV より遅かったため (mix の up は
/// 1 行 320 要素 = warp が 10 反復しかせず、行数 10240 の割に 1 warp の仕事が
/// 小さすぎる)。down (行数 320 / k 10240) と up を別々に A/B できるようにする。
/// 丸め順が変わる (warp の木) ので切り戻しは `DENNO_MIX_QK=0`。
pub const MIX_QK_ENV: &str = "DENNO_MIX_QK";


/// task 282: mix の GEMV 種別。`Old` = 従来の `plan_moe_gemv` (v3 形)、
/// `Seg` = `kernels_seg_k.cu` の 1 行 1 warp、`Blk` = 同ファイルの 1 行 1 ブロック
/// (行数が少ない down 用)、`R8` = 同ファイルの 1 warp 8 行 (行数が多い up 用)。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MixGemv {
    /// 従来 (`plan_moe_gemv` の v3 形。lab t282b の実測で 1 行列 54 us)。
    Old,
    /// 1 行 = 1 warp (`dno_seg_gemv_k_q8_0`)。
    Seg,
    /// 1 行 = 1 ブロック 256 スレッド (`dno_mix_gemv_q8_0_rowblk`)。
    Blk,
    /// 1 warp = 8 行 (`dno_mix_gemv_q8_0_r8`)。
    R8,
    /// task 286: 既存の `dno_gemv_q8_0_v5` (`kernels_moe_v5.ptx`、GDN の
    /// decode GEMV と同じ形)。**2 ブロック = 68 B を 17 ワードで読み**、
    /// 次の対を先読みし、アキュムレータを 4 本に分ける。
    ///
    /// lab t286a の `[dec-kprof-avg]` (採否前の既定) で mix の GEMV は
    /// down `rowblk` 31.8 us / up `r8` 39.8 us だった (3.48 MB を 109 / 87 GB/s)。
    /// 同じ 3.48 MB を v5 形 (GDN の qkv) は 374 GB/s で読んでいる。
    /// PTX は既にコミット済みなので、この経路は**ホスト側の切り替えだけ**。
    V5,
    /// task 418: 1 ブロック = 1 行・1 レーン = 1 対 (`kernels_mix_dn5.cu`)。
    /// down 専用 (n_in = 10,240 / n_out = 320 の形)。v5 の 160 ブロック
    /// x 64 スレッドに対し 320 ブロック x 160 スレッド = 5 倍のスレッドで、
    /// ループも先読みも無い。合計順が v5 と違う (丸めだけ)。
    Dn5,
    /// task 422: 1 ブロック = rows 行・1 レーン = 1 対 (`kernels_mix_dn6.cu`)。
    /// dn5 を「複数行」に一般化した形で、x の 64 要素をレジスタに 1 回だけ
    /// 載せて rows 行で使い回す。行ごとの積和と縮約順は dn5 と同一なので、
    /// **値は dn5 と 1 ビットも変わらない**。
    Dn6,
    /// task 431: 1 対 = 4 レーン (`kernels_mix_dn7.cu`)。1 レーン = 1 対では
    /// スレッド数が n_out * n_pairs で頭打ちになり 9.4 warp/SM しか出ないため、
    /// 16 要素ずつ 4 レーンに割って 4 倍のスレッドを出す (丸め順が変わる)。
    Dn7,
}


/// [`MIX_QK_ENV`] の解釈 (純関数)。返り値は `(down, up)` の種別。
///
/// - 未設定 / 未知: `(Dn5, V5)` = 既定 (task 418。down だけ dn5 の新カーネル)
/// - `0` / `off` / `false` : `(Old, Old)` = 切り戻し (task 266 より前)
/// - `blk`                : `(Blk, R8)` = task 282 の形 (採否前の既定)
/// - `down`               : `(Blk, Old)`
/// - `up`                 : `(Old, R8)`
/// - `seg`                : `(Seg, Seg)` = task 278 の seg_k をそのまま
/// - `v5d` / `v5u`        : 片方向だけ v5 (A/B 用)
/// - `ups`                : `(Blk, Seg)` = down は task 282 のまま、up だけ
///   1 行 1 warp の A/B 用
/// - `dn5` / `d5`         : `(Dn5, V5)` = task 418 の down (1 ブロック 1 行)
/// - `v5` / `1` / `both`  : `(V5, V5)` = task 286 の形 (task 418 の切り戻し)
pub fn mix_qk_from_value(v: Option<&str>) -> (MixGemv, MixGemv) {
    match v.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("0") | Some("off") | Some("false") | Some("no") => (MixGemv::Old, MixGemv::Old),
        Some("blk") | Some("r8") => (MixGemv::Blk, MixGemv::R8),
        Some("down") => (MixGemv::Blk, MixGemv::Old),
        Some("up") => (MixGemv::Old, MixGemv::R8),
        Some("seg") => (MixGemv::Seg, MixGemv::Seg),
        Some("v5d") => (MixGemv::V5, MixGemv::R8),
        Some("v5u") => (MixGemv::Blk, MixGemv::V5),
        Some("ups") => (MixGemv::Blk, MixGemv::Seg),
        // task 418: down だけ 1 ブロック 1 行・1 レーン 1 対の新カーネル。
        Some("dn5") | Some("d5") => (MixGemv::Dn5, MixGemv::V5),
        // task 431: 1 対 = 4 レーンの dn7 (丸め順が変わる)。
        Some("dn7") | Some("d7") => (MixGemv::Dn7, MixGemv::V5),
        // task 422: down を「1 ブロック rows 行」に一般化した dn6
        // (`mix_dn6_rows_from_value` が rows を決める。値は dn5 と同一)。
        Some("dn6") | Some("d6") | Some("dn6r1") | Some("dn6r2") | Some("dn6r4")
        | Some("dn6r6") | Some("dn6r8") | Some("d6r1") | Some("d6r2") | Some("d6r4")
        | Some("d6r6") | Some("d6r8") => {
            (MixGemv::Dn6, MixGemv::V5)
        }
        // task 286 の形へ明示的に戻す逃げ道 (task 418 の切り戻し)。
        Some("1") | Some("both") | Some("v5") => (MixGemv::V5, MixGemv::V5),
        // task 422: lab t422b で短い decode 12.9 -> 12.5 (-3.1%)、14k 後 -4.1%、
        // mix 段 -6.7%、md5 全 SAME (rows=2 が rows=4 より全点で良かった) なので
        // **down の既定を dn6 (rows=2) にした**。
        // task 431: lab t431c で 1 対 4 レーンの dn7 が mix 段 -5〜-7%
        // (8 リクエスト中 7 本。最初の 1 本だけは新カーネルの初回ロードで +13%)、
        // 短い decode -1.0%、14k 後 -2.3%、prefill +0.9%、md5 全 SAME、
        // errors 0 だったので **既定を dn7 にした**。形が合わない型・k
        // (np が 8 の倍数でない / 1 行 1024 超) や PTX 欠けは `mix_dn7_plan` の
        // 門番で dn6 へ落ちる。
        // 切り戻しは `DENNO_MIX_QK=dn6` / `dn5` (task 418) / `v5` (task 286)。
        _ => (MixGemv::Dn7, MixGemv::V5),
    }
}


/// [`MIX_QK_ENV`] をプロセスで 1 回だけ読む。`(down, up)` の種別。
pub fn mix_qk_enabled() -> (MixGemv, MixGemv) {
    static ON: OnceLock<(MixGemv, MixGemv)> = OnceLock::new();
    *ON.get_or_init(|| mix_qk_from_value(std::env::var(MIX_QK_ENV).ok().as_deref()))
}


/// gate_audit の台帳用 (既定 on のゲート)。どちらかが seg_k 系なら true。
pub fn mix_qk_gate_on() -> bool {
    mix_qk_enabled() != (MixGemv::Old, MixGemv::Old)
}


/// task 422: dn6 が選ばれているか (gate_audit の台帳用)。
/// mix の down (`DENNO_MIX_QK=dn6`) か GDN (`DENNO_GDN_DEC_GEMV=dn6`) の
/// どちらかで選ばれたら true。
pub fn mix_dn6_gate_on() -> bool {
    matches!(mix_qk_enabled().0, MixGemv::Dn6 | MixGemv::Dn7) || gdn_dec_gemv_dn6_on()
}


/// task 282: 高速 GEMV が実際に撃たれた回数 (診断用。プロセス通算)。
static MIX_QK_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);


/// task 282: 高速 GEMV 用の前置きバッファのレイアウト (純関数)。
///
/// 1 本の H2D で「重みポインタ 2 本 + segment 表 2 枚 + xn」を送る。
/// 返り値は `(wptr_down, wptr_up, seg_down, seg_up, x)` のバイトオフセット。
/// segment 表は 4 語 (`kernels_seg_k.cu` の規約: id / m / x_off / out_off) で、
/// どちらも `m=1, x_off=0, out_off=0`。`x` は 16 バイト整列。
pub fn mix_qk_layout() -> (usize, usize, usize, usize, usize) {
    (0, 8, 16, 32, 48)
}


/// task 282: 高速 GEMV の前置き (`[wptr_d][wptr_u][seg_d][seg_u]`) のバイト数。
pub(crate) const MIX_QK_PROLOGUE: usize = 48;


/// task 286: [`mix_v5_plan`] の形の部分だけを取り出した純関数。
/// 返り値は `(lanes, rows_per_block, grid)`。形が合わなければ `None`。
pub fn mix_v5_shape(n_in: usize, n_out: usize) -> Option<(usize, usize, u32)> {
    if n_in == 0 || n_out == 0 || n_in % 32 != 0 {
        return None;
    }
    let nb = n_in / 32;
    if nb < 2 || nb % 2 != 0 {
        return None;
    }
    let np = nb / 2;
    let lanes = v5_lanes_div(np, 32);
    if lanes == 0 {
        return None;
    }
    let rows = if lanes >= 16 {
        2
    } else if lanes >= 8 {
        4
    } else {
        (128 / lanes).max(1)
    };
    let block_dim = rows * lanes;
    if block_dim == 0 || block_dim > 1024 {
        return None;
    }
    Some((lanes, rows, n_out.div_ceil(rows) as u32))
}


/// task 286: mix の Q8_0 GEMV を v5 形 (`dno_gemv_q8_0_v5`) で回すときの
/// 起動計画。形が合わない / PTX が無いときは `None` (呼び出し側は Blk/R8 へ)。
///
/// v5 の制約:
/// - Q8_0 のみ。`nb = n_in / 32` が 2 以上かつ偶数 (2 ブロック = 68 B の対で読む)。
/// - `lanes` は 2 の冪で `np = nb / 2` を割り切ること (レーンの担当区間が
///   等分できること)。`lanes` は 1 も許す (`dno_v5_reduce` が空ループになる)。
///
/// 形の選び方 (lab の実測で決める): レーンが多いほど 1 行を 1 warp で読むので
/// 行数が少ない down (n_out=320) は `lanes=32` / 2 行 1 ブロック (grid=n_out/2)。
/// 行数が多い up (n_out=10,240) は `lanes=1` / 128 行 1 ブロック (grid=n_out/128)。
pub fn mix_v5_plan(
    ty: crate::kquant::GgmlType,
    n_in: usize,
    n_out: usize,
) -> Option<MoeGemvPlan> {
    use crate::kquant::GgmlType as T;
    if ty != T::Q8_0 {
        return None;
    }
    let (lanes, rows, grid) = mix_v5_shape(n_in, n_out)?;
    let rb = gguf_row_bytes(ty, n_in)?;
    Some(MoeGemvPlan {
        f: gemv_v5_kernels()?.f[0],
        rows_per_block: rows,
        block_dim: (rows * lanes) as u32,
        shmem: 0,
        lanes,
        rb,
        grid,
    })
}


/// task 286: [`mix_v5_plan`] の計画を 1 本積む (v5 は引数 8 本。`lanes=1` でも
/// `rows_per_block` / `lanes` を渡す = 6 本で撃つと未定義値になる)。
fn launch_mix_v5(
    ctx: &CudaCtx,
    plan: &MoeGemvPlan,
    w: CUdeviceptr,
    x: CUdeviceptr,
    o: CUdeviceptr,
    n_in: usize,
    n_out: usize,
) -> Result<()> {
    let (mut pw, mut px, mut po) = (w, x, o);
    let (mut ni, mut no, mut rb) = (n_in as c_int, n_out as c_int, plan.rb as c_int);
    let (mut rows, mut lanes) = (plan.rows_per_block as c_int, plan.lanes as c_int);
    let mut args = [
        &mut pw as *mut _ as *mut c_void,
        &mut px as *mut _ as *mut c_void,
        &mut po as *mut _ as *mut c_void,
        &mut ni as *mut _ as *mut c_void,
        &mut no as *mut _ as *mut c_void,
        &mut rb as *mut _ as *mut c_void,
        &mut rows as *mut _ as *mut c_void,
        &mut lanes as *mut _ as *mut c_void,
    ];
    unsafe {
        ctx.launch_async(
            plan.f,
            (plan.grid, 1, 1),
            (plan.block_dim, 1, 1),
            plan.shmem,
            &mut args,
        )
    }
}


// ===========================================================================
// task 418: mix の down (Q8_0) を 1 ブロック 1 行・1 レーン 1 対で読む
// (`DENNO_MIX_QK=dn5`)
// ===========================================================================
//
// 実形 (n_in = 10,240 = 160 対、n_out = 320、3.48 MB) を v5 は
// `lanes=32 / rows=2` = **160 ブロック x 64 スレッド**で撃つ (1 レーンが 5 対を
// 逐次ループ)。lab t418a の `[dec-kprof-avg]` (band 0) で down は
// **19.4 us = 180 GB/s** で、同じ band の inject (`dno_matvec_f32_r`、
// 512 ブロック x 256 スレッド、13 MB) は 7.1 us だった。差は並列度。
//
// 新カーネルは 1 ブロック = 1 行、block = 32 * (n_in / 64 / 32) = 160 スレッド
// (1 レーン = 1 対 = 68 B を 1 回だけ読む)。ループも先読みも無い。
// 縮約は warp 内の木 -> warp ごとの部分和を共有メモリ -> スレッド 0 が固定順。
// 合計順が v5 と違う (丸めだけ) ので既定 off。値はゲートで切り戻せる。

/// 新カーネルの名前。
const MIX_DN5_KERNEL_NAME: &str = "dno_mix_gemv_q8_0_dn5";


/// 新カーネルの引数の数 (w, x, out, n_in, n_out, row_bytes)。
const MIX_DN5_NARGS: usize = 6;


/// `.cu` の `part[8]` (warp の部分和の上限)。
const MIX_DN5_MAX_WARPS: usize = 8;


pub(crate) struct MixDn5Kernels {
    #[allow(dead_code)]
    module: CUmodule,
    f: CUfunction,
}


unsafe impl Send for MixDn5Kernels {}

unsafe impl Sync for MixDn5Kernels {}


static MIX_DN5_KERNELS: OnceLock<Option<MixDn5Kernels>> = OnceLock::new();


fn mix_dn5_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_MIX_DN5_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_mix_dn5.ptx"),
    }
}


/// 新カーネルを取得する (初回だけ PTX を読む。失敗は `None` を焼き付ける)。
fn mix_dn5_kernels() -> Option<&'static MixDn5Kernels> {
    MIX_DN5_KERNELS
        .get_or_init(|| {
            let path = mix_dn5_ptx_path();
            let mut image = match std::fs::read(&path) {
                Ok(b) => b,
                Err(_) => return None,
            };
            let text = String::from_utf8_lossy(&image).to_string();
            if ptx_entry_params(&text, MIX_DN5_KERNEL_NAME) != Some(MIX_DN5_NARGS) {
                eprintln!(
                    "[mix-dn5] PTX の引数が {MIX_DN5_NARGS} 本でない → v5 の down へ。\
                     `bash build_ptx.sh mix_dn5` で再生成する"
                );
                return None;
            }
            image.push(0);
            let lib = lib().ok()?;
            unsafe {
                let mut module: CUmodule = std::ptr::null_mut();
                if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void)
                    != CUDA_SUCCESS
                {
                    return None;
                }
                let mut cname = MIX_DN5_KERNEL_NAME.as_bytes().to_vec();
                cname.push(0);
                let mut f: CUfunction = std::ptr::null_mut();
                if (lib.module_get_function)(&mut f, module, cname.as_ptr() as *const c_char)
                    != CUDA_SUCCESS
                {
                    eprintln!("[mix-dn5] PTX に {MIX_DN5_KERNEL_NAME} が無い → v5 の down へ");
                    return None;
                }
                Some(MixDn5Kernels { module, f })
            }
        })
        .as_ref()
}


/// 新カーネルが載っているか (テスト / 診断用)。
pub fn mix_dn5_loaded() -> bool {
    mix_dn5_kernels().is_some()
}


/// 形の門番 (純関数)。`.cu` の前提は
/// 「Q8_0 / n_in % 64 == 0 / 対の数 np = n_in / 64 が 32 の倍数 / np/32 <= 8」。
/// 返り値は 1 ブロックの warp 数 (= np / 32)。
pub fn mix_dn5_warps(n_in: usize) -> Option<usize> {
    if n_in == 0 || n_in % 64 != 0 {
        return None;
    }
    let np = n_in / 64;
    if np % 32 != 0 {
        return None;
    }
    let warps = np / 32;
    if warps == 0 || warps > MIX_DN5_MAX_WARPS {
        return None;
    }
    Some(warps)
}


/// 新カーネルを 1 本積む (grid = n_out、block = 32 * warps)。
fn launch_mix_dn5(
    ctx: &CudaCtx,
    w: CUdeviceptr,
    x: CUdeviceptr,
    o: CUdeviceptr,
    n_in: usize,
    n_out: usize,
    row_bytes: usize,
    warps: usize,
) -> Result<()> {
    let k = mix_dn5_kernels().ok_or_else(|| anyhow!("dn5 のカーネルが無い"))?;
    let (mut pw, mut px, mut po) = (w, x, o);
    let (mut ni, mut no, mut rb) = (n_in as c_int, n_out as c_int, row_bytes as c_int);
    let mut args = [
        &mut pw as *mut _ as *mut c_void,
        &mut px as *mut _ as *mut c_void,
        &mut po as *mut _ as *mut c_void,
        &mut ni as *mut _ as *mut c_void,
        &mut no as *mut _ as *mut c_void,
        &mut rb as *mut _ as *mut c_void,
    ];
    unsafe {
        ctx.launch_async(
            k.f,
            (n_out.max(1) as u32, 1, 1),
            ((32 * warps) as u32, 1, 1),
            0,
            &mut args,
        )
    }
}


/// task 418: dn5 の down を撃った回数 (診断。`[mix-qchain]` の `dn5=` に出る)。
static MIX_DN5_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);


/// task 418: dn5 の down を撃った回数。
pub fn mix_dn5_hits() -> u64 {
    MIX_DN5_HITS.load(std::sync::atomic::Ordering::Relaxed)
}


/// task 418: 集計を 0 に戻す (リクエスト終端)。
pub fn mix_dn5_reset() {
    MIX_DN5_HITS.store(0, std::sync::atomic::Ordering::Relaxed);
}


// ===========================================================================
// task 422: `dno_mix_gemv_q8_0_dn6` — dn5 を「1 ブロック = rows 行」に一般化
// (`DENNO_MIX_QK=dn6` / `DENNO_GDN_DEC_GEMV=dn6`)
// ===========================================================================
//
// lab t422a の `[dec-kprof-avg]` (band 0、既定モデル) で、Q8_0 の GEMV は 2 か所が
// 残っていた:
//   - mix の down (`dn5`): 320 x 160 = 15.4 us。重み 3.48 MB + **x 13.1 MB**
//     (320 ブロックが同じ 41 KB を読む) = 16.6 MB を 1.08 TB/s。
//   - GDN の qkv/out (`dno_gemv_q8_0_v5`): 320 x 256 が 2 本/層で 35.4 us ずつ。
//     v5 は 8 行 1 ブロック・1 レーンが 5 対を逐次ループ (重み 27.9 MB を 787 GB/s)。
// dn6 は 1 レーン = 1 対のまま、**1 ブロックが rows 行を持つ**ので
//   (a) 1 レーンの逐次ループが無くなる (v5 の 5 対 -> 1 対)
//   (b) x の 64 要素をレジスタに 1 回だけ載せて rows 行で使い回す (x が 1/rows)
// を行う。行ごとの積和と縮約の順は dn5 と同一なので、**値は dn5 と 1 ビットも
// 変わらない** (v5 から切り替えたときだけ合計順が変わる = 丸めだけ)。

/// 新カーネルの名前。
const MIX_DN6_KERNEL_NAME: &str = "dno_mix_gemv_q8_0_dn6";


/// 新カーネルの引数の数 (w, x, out, n_in, n_out, row_bytes, rows)。
const MIX_DN6_NARGS: usize = 7;


/// `.cu` の `part[8][8]` (warp の部分和 / 行の上限)。
const MIX_DN6_MAX_WARPS: usize = 8;

const MIX_DN6_MAX_ROWS: usize = 8;


/// 1 ブロックのスレッド上限 (rows * 32 * warps)。
const MIX_DN6_MAX_THREADS: usize = 1024;


pub(crate) struct MixDn6Kernels {
    #[allow(dead_code)]
    module: CUmodule,
    f: CUfunction,
}


unsafe impl Send for MixDn6Kernels {}

unsafe impl Sync for MixDn6Kernels {}


static MIX_DN6_KERNELS: OnceLock<Option<MixDn6Kernels>> = OnceLock::new();


fn mix_dn6_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_MIX_DN6_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_mix_dn6.ptx"),
    }
}


/// 新カーネルを取得する (初回だけ PTX を読む。失敗は `None` を焼き付ける)。
pub(crate) fn mix_dn6_kernels() -> Option<&'static MixDn6Kernels> {
    MIX_DN6_KERNELS
        .get_or_init(|| {
            let path = mix_dn6_ptx_path();
            let mut image = match std::fs::read(&path) {
                Ok(b) => b,
                Err(_) => return None,
            };
            let text = String::from_utf8_lossy(&image).to_string();
            if ptx_entry_params(&text, MIX_DN6_KERNEL_NAME) != Some(MIX_DN6_NARGS) {
                eprintln!(
                    "[mix-dn6] PTX の引数が {MIX_DN6_NARGS} 本でない → dn5 / v5 へ。\
                     `bash build_ptx.sh mix_dn6` で再生成する"
                );
                return None;
            }
            image.push(0);
            let lib = lib().ok()?;
            unsafe {
                let mut module: CUmodule = std::ptr::null_mut();
                if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void)
                    != CUDA_SUCCESS
                {
                    return None;
                }
                let mut cname = MIX_DN6_KERNEL_NAME.as_bytes().to_vec();
                cname.push(0);
                let mut f: CUfunction = std::ptr::null_mut();
                if (lib.module_get_function)(&mut f, module, cname.as_ptr() as *const c_char)
                    != CUDA_SUCCESS
                {
                    eprintln!("[mix-dn6] PTX に {MIX_DN6_KERNEL_NAME} が無い → dn5 / v5 へ");
                    return None;
                }
                Some(MixDn6Kernels { module, f })
            }
        })
        .as_ref()
}


/// 新カーネルが載っているか (テスト / 診断用)。
pub fn mix_dn6_loaded() -> bool {
    mix_dn6_kernels().is_some()
}


/// 形の門番 (純関数)。`.cu` の前提は
/// 「Q8_0 / n_in % 64 == 0 / 対の数 np = n_in / 64 が 1 以上 /
///   warps = ceil(np / 32) が 8 以下」。返り値は 1 行あたりの warp 数。
pub fn mix_dn6_warps(n_in: usize) -> Option<usize> {
    if n_in == 0 || n_in % 64 != 0 {
        return None;
    }
    let np = n_in / 64;
    if np == 0 {
        return None;
    }
    let warps = np.div_ceil(32);
    if warps == 0 || warps > MIX_DN6_MAX_WARPS {
        return None;
    }
    Some(warps)
}


/// 起動形 (純関数)。`Some((warps, rows))` なら撃てる。
///
/// `want_rows` は 1 ブロックが持つ行数の希望。`rows * 32 * warps <= 1024` と
/// `rows <= 8` に丸める (丸めた結果が 1 未満なら撃たない)。
pub fn mix_dn6_plan(n_in: usize, n_out: usize, want_rows: usize) -> Option<(usize, usize)> {
    if n_out == 0 {
        return None;
    }
    let warps = mix_dn6_warps(n_in)?;
    let fit = MIX_DN6_MAX_THREADS / (32 * warps);
    let rows = want_rows.clamp(1, fit.min(MIX_DN6_MAX_ROWS));
    if rows == 0 {
        return None;
    }
    let grid = n_out.div_ceil(rows);
    if grid == 0 || grid > u32::MAX as usize {
        return None;
    }
    Some((warps, rows))
}


/// 新カーネルを 1 本積む (grid = ceil(n_out / rows)、block = 32 * warps * rows)。
pub(crate) fn launch_mix_dn6(
    ctx: &CudaCtx,
    w: CUdeviceptr,
    x: CUdeviceptr,
    o: CUdeviceptr,
    n_in: usize,
    n_out: usize,
    row_bytes: usize,
    warps: usize,
    rows: usize,
) -> Result<()> {
    let k = mix_dn6_kernels().ok_or_else(|| anyhow!("dn6 のカーネルが無い"))?;
    let (mut pw, mut px, mut po) = (w, x, o);
    let (mut ni, mut no, mut rb) = (n_in as c_int, n_out as c_int, row_bytes as c_int);
    let mut rw = rows as c_int;
    let mut args = [
        &mut pw as *mut _ as *mut c_void,
        &mut px as *mut _ as *mut c_void,
        &mut po as *mut _ as *mut c_void,
        &mut ni as *mut _ as *mut c_void,
        &mut no as *mut _ as *mut c_void,
        &mut rb as *mut _ as *mut c_void,
        &mut rw as *mut _ as *mut c_void,
    ];
    let grid = n_out.div_ceil(rows).max(1) as u32;
    let block = (32 * warps * rows) as u32;
    unsafe { ctx.launch_async(k.f, (grid, 1, 1), (block, 1, 1), 0, &mut args) }
}


/// task 422: dn6 を撃った回数 (診断。`[mix-qchain]` の `dn6=` に出る)。
pub(crate) static MIX_DN6_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);


/// task 422: dn6 を撃った回数。
pub fn mix_dn6_hits() -> u64 {
    MIX_DN6_HITS.load(std::sync::atomic::Ordering::Relaxed)
}


/// task 422: 集計を 0 に戻す。
pub fn mix_dn6_reset() {
    MIX_DN6_HITS.store(0, std::sync::atomic::Ordering::Relaxed);
}


/// `DENNO_MIX_QK` / `DENNO_GDN_DEC_GEMV` の値から dn6 の rows を決める (純関数)。
///
/// `dn6` / `d6` = 4 行 (実測で選ぶ)、`dn6r2` = 2 行、`dn6r6` = 6 行
/// (GDN の qkv/out は 6 行 = block 960)。それ以外は既定 4。
pub fn mix_dn6_rows_from_value(v: Option<&str>) -> usize {
    match v.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("dn6r1") | Some("d6r1") => 1,
        Some("dn6r4") | Some("d6r4") => 4,
        Some("dn6r6") | Some("d6r6") => 6,
        Some("dn6r8") | Some("d6r8") => 8,
        // task 422: 既定 (未設定 / dn6 / d6 / それ以外) は 2 行。
        // lab t422b で rows=2 が rows=4 より短い decode・14k 後とも良かった。
        _ => 2,
    }
}


/// dn6 の rows (mix の down 用。env から 1 回だけ読む)。
fn mix_dn6_rows() -> usize {
    static R: OnceLock<usize> = OnceLock::new();
    *R.get_or_init(|| {
        mix_dn6_rows_from_value(std::env::var(MIX_QK_ENV).ok().as_deref())
    })
}


// ===========================================================================
// task 496: dn6 の**レーン詰め**版 (`dno_mix_gemv_q8_0_dn8`、`DENNO_GDN_DEC_GEMV=dn8`)
// ===========================================================================
//
// dn6 は 1 レーン = 1 対 (64 要素) なので、レーン数 = 32 * ceil(np / 32) に固定される
// (np = n_in / 64)。既定モデルの GDN qkv は n_in = 2,560 / n_out = 16,384 で
// **np = 40** = 64 レーン中 24 レーンが遊ぶ (埋まり 62.5%)。t492c の ssm 段
// (3.03 ms/token) のうち最大の 1 本がこの qkv で、実測は約 1.27 TB/s
// (ピーク 1.79 TB/s の 71%)。
//
// dn8 は行の割り当てを「1 ブロック rows 行」から「1 ブロック = rows * np 個の
// (行, 対)」へ変え、レーンを (行, 対) で埋める (items = rows * np)。x は shared に
// 1 回だけ置いて rows 本で使い回す。**1 対の積和は dn6 と同一**で、変わるのは
// 対の部分和の足し方だけ (dn6 = warp 木 + warp の逐次、dn8 = 対の昇順 1 本)。
// 行ごとに独立で決定的。np が 32 の倍数の形 (GDN の out = np 96 など) では
// dn6 が既に 100% なので dn8 は使わない (dn6 のまま)。

/// 新カーネルの名前。
const MIX_DN8_KERNEL_NAME: &str = "dno_mix_gemv_q8_0_dn8";


/// 新カーネルの引数の数 (w, x, out, n_in, n_out, row_bytes, rows)。
const MIX_DN8_NARGS: usize = 7;


/// 1 ブロックのスレッド上限 (items = rows * np)。
const MIX_DN8_MAX_THREADS: usize = 1024;


/// 1 ブロックが持つ行数の上限 (`.cu` は rows を引数で受けるだけなので自由。
/// t496a で rows 2 -> 4 -> 8 が単調に良かったので 16 まで試せるようにする)。
const MIX_DN8_MAX_ROWS: usize = 16;


/// 1 ブロックのスレッド数上限 (レジスタの都合)。
///
/// `.cu` は 1 レーンが x の 64 要素をレジスタに持つので ~100 レジスタ/スレッド
/// 使う。t496b の `d8a8` (out を block 768 で撃った回) は
/// `cuLaunchKernel が失敗: error 701` = LAUNCH_OUT_OF_RESOURCES で落ちた
/// (768 x 100 > 65,536)。640 は t496b の `d8a16` (qkv を block 640 で撃った回)
/// で通っているので、それを上限にする。
const MIX_DN8_MAX_BLOCK: usize = 640;


/// shared の上限 (x の詰め物込み。48 KiB を超える形は dn6 へ)。
const MIX_DN8_MAX_SHMEM: usize = 48 * 1024;


/// dn8 の起動形 (純関数 [`mix_dn8_plan`] の結果)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MixDn8Plan {
    /// 1 行あたりの対の数 (n_in / 64)。
    pub np: usize,
    /// 1 ブロックが持つ行数。
    pub rows: usize,
    /// 1 ブロックのスレッド数 (32 の倍数)。
    pub block: usize,
    /// 動的 shared のバイト数。
    pub shmem: usize,
    /// grid (= ceil(n_out / rows))。
    pub grid: usize,
}


pub(crate) struct MixDn8Kernels {
    #[allow(dead_code)]
    module: CUmodule,
    f: CUfunction,
}


unsafe impl Send for MixDn8Kernels {}

unsafe impl Sync for MixDn8Kernels {}


static MIX_DN8_KERNELS: OnceLock<Option<MixDn8Kernels>> = OnceLock::new();


fn mix_dn8_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_MIX_DN8_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_mix_dn8.ptx"),
    }
}


/// 新カーネルを取得する (初回だけ PTX を読む。失敗は `None` を焼き付ける)。
pub(crate) fn mix_dn8_kernels() -> Option<&'static MixDn8Kernels> {
    MIX_DN8_KERNELS
        .get_or_init(|| {
            let path = mix_dn8_ptx_path();
            let mut image = match std::fs::read(&path) {
                Ok(b) => b,
                Err(_) => return None,
            };
            let text = String::from_utf8_lossy(&image).to_string();
            if ptx_entry_params(&text, MIX_DN8_KERNEL_NAME) != Some(MIX_DN8_NARGS) {
                eprintln!(
                    "[mix-dn8] PTX の引数が {MIX_DN8_NARGS} 本でない → dn6 / v5 へ。\
                     `bash build_ptx.sh mix_dn8` で再生成する"
                );
                return None;
            }
            image.push(0);
            let lib = lib().ok()?;
            unsafe {
                let mut module: CUmodule = std::ptr::null_mut();
                if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void)
                    != CUDA_SUCCESS
                {
                    return None;
                }
                let mut cname = MIX_DN8_KERNEL_NAME.as_bytes().to_vec();
                cname.push(0);
                let mut f: CUfunction = std::ptr::null_mut();
                if (lib.module_get_function)(&mut f, module, cname.as_ptr() as *const c_char)
                    != CUDA_SUCCESS
                {
                    eprintln!("[mix-dn8] PTX に {MIX_DN8_KERNEL_NAME} が無い → dn6 / v5 へ");
                    return None;
                }
                Some(MixDn8Kernels { module, f })
            }
        })
        .as_ref()
}


/// 新カーネルが載っているか (テスト / 診断用)。
pub fn mix_dn8_loaded() -> bool {
    mix_dn8_kernels().is_some()
}


/// 形の門番 (純関数)。dn6 よりレーンが埋まる形だけを受ける。
///
/// 前提: `np = n_in / 64` が 1 以上 / `np % 32 != 0` (32 の倍数なら dn6 で十分) /
/// `items = rows * np <= 1024` / `shared = (np * 65 + items) * 4 <= 48 KiB`。
pub fn mix_dn8_plan(n_in: usize, n_out: usize, want_rows: usize) -> Option<MixDn8Plan> {
    mix_dn8_plan_with(n_in, n_out, want_rows, false)
}


/// [`mix_dn8_plan`] の本体。`allow_full` は「np が 32 の倍数の形 (dn6 で既に
/// レーンが 100% 埋まる形) でも dn8 を使う」か。
///
/// その形でも dn8 は **x を shared に 1 回だけ置いて rows 本で使い回す**ので、
/// x のグローバル読み (1 ブロックあたり n_in 本) が 1/rows になる
/// (GDN の out は n_in = 6,144 で、rows=2 の dn6 は 1,280 ブロック x 24 KB を
/// 読む)。レーンの詰まりは dn6 と同じなので、埋まりの比較は課さない。
pub fn mix_dn8_plan_with(
    n_in: usize,
    n_out: usize,
    want_rows: usize,
    allow_full: bool,
) -> Option<MixDn8Plan> {
    if n_in == 0 || n_in % 64 != 0 || n_out == 0 {
        return None;
    }
    let np = n_in / 64;
    if np == 0 {
        return None;
    }
    let lane_gain = np % 32 != 0;
    if !lane_gain && !allow_full {
        // dn6 が既に 100% 埋まる (qkv の out 側や mix down はこちら)。
        return None;
    }
    let rows = want_rows.clamp(1, MIX_DN8_MAX_ROWS);
    let items = np.checked_mul(rows)?;
    if items == 0 || items > MIX_DN8_MAX_THREADS {
        return None;
    }
    let block = 32 * items.div_ceil(32);
    if block > MIX_DN8_MAX_BLOCK {
        return None;
    }
    if lane_gain {
        // dn6 の埋まり (32 * ceil(np / 32) レーン) より 5 ポイント以上良いときだけ。
        let dn6_lanes = 32 * np.div_ceil(32);
        if dn8_fill(items, block) <= dn8_fill(np, dn6_lanes) + 0.05 {
            return None;
        }
    }
    let shmem = (np * 65 + items).checked_mul(4)?;
    if shmem > MIX_DN8_MAX_SHMEM {
        return None;
    }
    let grid = n_out.div_ceil(rows);
    if grid == 0 || grid > u32::MAX as usize {
        return None;
    }
    Some(MixDn8Plan {
        np,
        rows,
        block,
        shmem,
        grid,
    })
}


/// dn8 を 1 本積む (grid = plan.grid、block = plan.block、shmem = plan.shmem)。
pub(crate) fn launch_mix_dn8(
    ctx: &CudaCtx,
    plan: &MixDn8Plan,
    w: CUdeviceptr,
    x: CUdeviceptr,
    o: CUdeviceptr,
    n_in: usize,
    n_out: usize,
    row_bytes: usize,
) -> Result<()> {
    let k = mix_dn8_kernels().ok_or_else(|| anyhow!("dn8 のカーネルが無い"))?;
    let (mut pw, mut px, mut po) = (w, x, o);
    let (mut ni, mut no, mut rb) = (n_in as c_int, n_out as c_int, row_bytes as c_int);
    let mut rw = plan.rows as c_int;
    let mut args = [
        &mut pw as *mut _ as *mut c_void,
        &mut px as *mut _ as *mut c_void,
        &mut po as *mut _ as *mut c_void,
        &mut ni as *mut _ as *mut c_void,
        &mut no as *mut _ as *mut c_void,
        &mut rb as *mut _ as *mut c_void,
        &mut rw as *mut _ as *mut c_void,
    ];
    unsafe {
        ctx.launch_async(
            k.f,
            (plan.grid as u32, 1, 1),
            (plan.block as u32, 1, 1),
            plan.shmem as u32,
            &mut args,
        )
    }
}


/// task 496: dn8 を撃った回数 (診断。`[mix-qchain]` の `dn8=` に出る)。
pub(crate) static MIX_DN8_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);


/// task 496: dn8 を撃った回数。
pub fn mix_dn8_hits() -> u64 {
    MIX_DN8_HITS.load(std::sync::atomic::Ordering::Relaxed)
}


/// task 496: 集計を 0 に戻す。
pub fn mix_dn8_reset() {
    MIX_DN8_HITS.store(0, std::sync::atomic::Ordering::Relaxed);
}


// ===========================================================================
// task 431: mix の down を 1 対 = 4 レーンで回す (`DENNO_MIX_QK=dn7`)
// ===========================================================================
//
// lab t431a の mix band (`[dec-kprof-avg]`、1 呼び出しあたり) は
//   H2D@a 10.3 / down 14.0 / up 10.8 / router 6.9 / D2H@a 5.4 us
// で、down は 3.48 MB を 245 GB/s。**同じ dn6 の PTX を GDN の qkv
// (grid 1,280 x 192) で撃つと 962 GB/s** 出るので、差はカーネルの中身ではなく
// 同時に走る warp 数 (mix down は 9.4 warp/SM)。1 レーン = 1 対のままだと
// スレッド数が n_out * n_pairs で固定されるため、dn7 は 1 対を 4 レーンに割る
// (`kernels_mix_dn7.cu`)。実形で 204,800 スレッド = 37.6 warp/SM。
//
// 値は丸め順だけが変わる (1 行の和を 16 要素ずつ 4 レーンに分けて足す)。
// ゲートは既存 env の新しい値 `DENNO_MIX_QK=dn7`。形が合わなければ dn6 へ落ちる。

/// 新カーネル名 (`kernels_mix_dn7.cu`)。
const MIX_DN7_KERNEL_NAME: &str = "dno_mix_gemv_q8_0_dn7";

/// 引数の数 (w, x, out, n_in, n_out, row_bytes, rows)。
const MIX_DN7_NARGS: usize = 7;

/// 1 ブロックのスレッド上限。
const MIX_DN7_MAX_THREADS: usize = 1024;

/// 1 レーンが担当する要素数 (`.cu` の `16 * sub`)。
const MIX_DN7_LANES: usize = 4;


pub(crate) struct MixDn7Kernels {
    #[allow(dead_code)]
    module: CUmodule,
    f: CUfunction,
}


unsafe impl Send for MixDn7Kernels {}

unsafe impl Sync for MixDn7Kernels {}


static MIX_DN7_KERNELS: OnceLock<Option<MixDn7Kernels>> = OnceLock::new();


fn mix_dn7_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_MIX_DN7_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_mix_dn7.ptx"),
    }
}


/// 新カーネルを取得する (初回だけ PTX を読む。失敗は `None` を焼き付ける)。
pub(crate) fn mix_dn7_kernels() -> Option<&'static MixDn7Kernels> {
    MIX_DN7_KERNELS
        .get_or_init(|| {
            let path = mix_dn7_ptx_path();
            let mut image = match std::fs::read(&path) {
                Ok(b) => b,
                Err(_) => return None,
            };
            let text = String::from_utf8_lossy(&image).to_string();
            if ptx_entry_params(&text, MIX_DN7_KERNEL_NAME) != Some(MIX_DN7_NARGS) {
                eprintln!(
                    "[mix-dn7] PTX の引数が {MIX_DN7_NARGS} 本でない → dn6 / dn5 / v5 へ。\
                     `bash build_ptx.sh mix_dn7` で再生成する"
                );
                return None;
            }
            image.push(0);
            let lib = lib().ok()?;
            unsafe {
                let mut module: CUmodule = std::ptr::null_mut();
                if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void)
                    != CUDA_SUCCESS
                {
                    return None;
                }
                let mut cname = MIX_DN7_KERNEL_NAME.as_bytes().to_vec();
                cname.push(0);
                let mut f: CUfunction = std::ptr::null_mut();
                if (lib.module_get_function)(&mut f, module, cname.as_ptr() as *const c_char)
                    != CUDA_SUCCESS
                {
                    eprintln!("[mix-dn7] PTX に {MIX_DN7_KERNEL_NAME} が無い → dn6 / dn5 / v5 へ");
                    return None;
                }
                Some(MixDn7Kernels { module, f })
            }
        })
        .as_ref()
}


/// 新カーネルが載っているか (テスト / 診断用)。
pub fn mix_dn7_loaded() -> bool {
    mix_dn7_kernels().is_some()
}


/// 起動形 (純関数)。`Some((rows, threads))` なら撃てる。
///
/// `.cu` の前提は「Q8_0 / n_in % 64 == 0 / 1 行あたり np*4 スレッド
/// (np = n_in/64) が 32 の倍数 = np % 8 == 0 / np*4 <= 1024 / warp 数 <= 32」。
/// 実形 (n_in = 10,240) は np = 160 で 1 行 640 スレッド・20 warp。
pub fn mix_dn7_plan(n_in: usize, n_out: usize) -> Option<(usize, usize)> {
    if n_in == 0 || n_out == 0 || n_in % 64 != 0 {
        return None;
    }
    let np = n_in / 64;
    if np == 0 || np % 8 != 0 {
        return None;
    }
    let per_row = np * MIX_DN7_LANES;
    if per_row > MIX_DN7_MAX_THREADS || per_row / 32 > 32 {
        return None;
    }
    // 1 ブロックは 1 行 (rows=1)。2 行にすると per_row*2 が上限を超える実形がある。
    let rows = 1;
    if n_out.div_ceil(rows) > u32::MAX as usize {
        return None;
    }
    Some((rows, per_row))
}


/// 新カーネルを 1 本積む (grid = ceil(n_out / rows)、block = rows * np * 4)。
pub(crate) fn launch_mix_dn7(
    ctx: &CudaCtx,
    w: CUdeviceptr,
    x: CUdeviceptr,
    o: CUdeviceptr,
    n_in: usize,
    n_out: usize,
    row_bytes: usize,
    rows: usize,
    threads: usize,
) -> Result<()> {
    let k = mix_dn7_kernels().ok_or_else(|| anyhow!("dn7 のカーネルが無い"))?;
    let (mut pw, mut px, mut po) = (w, x, o);
    let (mut ni, mut no, mut rb) = (n_in as c_int, n_out as c_int, row_bytes as c_int);
    let mut rw = rows as c_int;
    let mut args = [
        &mut pw as *mut _ as *mut c_void,
        &mut px as *mut _ as *mut c_void,
        &mut po as *mut _ as *mut c_void,
        &mut ni as *mut _ as *mut c_void,
        &mut no as *mut _ as *mut c_void,
        &mut rb as *mut _ as *mut c_void,
        &mut rw as *mut _ as *mut c_void,
    ];
    // block = rows (1) * threads。grid は行数。
    unsafe {
        ctx.launch_async(
            k.f,
            (n_out.div_ceil(rows) as u32, 1, 1),
            (threads as u32, 1, 1),
            0,
            &mut args,
        )
    }
}


/// dn7 を撃った回数 (診断。`[mix-qchain]` の `dn7=` に出る)。
static MIX_DN7_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);


/// dn7 を撃った回数。
pub fn mix_dn7_hits() -> u64 {
    MIX_DN7_HITS.load(std::sync::atomic::Ordering::Relaxed)
}


/// 集計を 0 に戻す。
pub fn mix_dn7_reset() {
    MIX_DN7_HITS.store(0, std::sync::atomic::Ordering::Relaxed);
}


// ===========================================================================
// task 437: mix の up を 1 対 = 4 レーンで回す (`DENNO_MIX_UPQ8=u4` / `u4f` / `u4w`)
// ===========================================================================
//
// lab t437a (int-auto af07b1008) の mix band (`[dec-kprof-avg]`、1 呼び出し):
// down (dn7) 12.1 us / up (`dno_mix_upmix_q8_0`) 11.3 us / router 7.1 us。
// up は grid 427 x block 128 = 54,656 スレッド = 10 warp/SM で 3.48 MB を
// 322 GB/s。down を dn7 (1 対 = 4 レーン) にしたときと同じ理屈で、up も
// 1 対を 4 レーンに割って同時 warp 数を 4 倍にする (`kernels_mix_up4.cu`)。
// 門番・引数・出力の意味は従来の融合版 (`dno_mix_upmix_q8_0`) と同じ。
//
// ゲートは既存 env の新しい値: `DENNO_MIX_UPQ8=u4` (= 3) / `=u4f` (= 4,
// exp だけ SFU) / `=u4w` (= 5, JC=12)。**既定は従来の融合版のまま**。

/// 1 対 = 4 レーンの up (JC=6、exp は glibc ビット一致)。
const MIX_UP4_KERNEL: &str = "dno_mix_upmix_q8_up4";


/// 同上で exp だけ `__expf` (比較用)。
const MIX_UP4_KERNEL_F: &str = "dno_mix_upmix_q8_up4f";


/// 同上で JC=12 (1 ブロック 960 スレッド)。
const MIX_UP4_KERNEL_W: &str = "dno_mix_upmix_q8_up4w";


/// 同上で JC=4 (1 ブロック 320 スレッド、grid 1.5 倍) + exp は SFU。
const MIX_UP4_KERNEL_X: &str = "dno_mix_upmix_q8_up4x";


/// 引数の数 (lo, w, xn, out, n_in, e, inv_hc, row_bytes)。
const MIX_UP4_NARGS: usize = 8;


/// `.cu` の JC=6 (1 ブロック 24 行 = 480 スレッド)。
pub(crate) const MIX_UP4_JC: usize = 6;


/// `.cu` の JC=6 のスレッド数。
pub(crate) const MIX_UP4_THREADS: u32 = 480;


/// `.cu` の JC=12 (1 ブロック 48 行 = 960 スレッド)。
pub(crate) const MIX_UP4_JC_W: usize = 12;


/// `.cu` の JC=12 のスレッド数。
pub(crate) const MIX_UP4_THREADS_W: u32 = 960;


/// `.cu` の JC=4 (1 ブロック 16 行 = 320 スレッド)。
pub(crate) const MIX_UP4_JC_X: usize = 4;


/// `.cu` の JC=4 のスレッド数。
pub(crate) const MIX_UP4_THREADS_X: u32 = 320;


pub(crate) struct MixUp4Kernels {
    #[allow(dead_code)]
    module: CUmodule,
    up4: CUfunction,
    up4f: CUfunction,
    up4w: CUfunction,
    up4x: CUfunction,
}


unsafe impl Send for MixUp4Kernels {}

unsafe impl Sync for MixUp4Kernels {}


static MIX_UP4_KERNELS: OnceLock<Option<MixUp4Kernels>> = OnceLock::new();


fn mix_up4_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_MIX_UP4_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_mix_up4.ptx"),
    }
}


/// 3 本とも載ったときだけ `Some` (1 本でも欠けたら従来の融合版へ落とす)。
fn mix_up4_kernels() -> Option<&'static MixUp4Kernels> {
    MIX_UP4_KERNELS
        .get_or_init(|| {
            let path = mix_up4_ptx_path();
            let mut image = match std::fs::read(&path) {
                Ok(b) => b,
                Err(_) => return None,
            };
            let text = String::from_utf8_lossy(&image).to_string();
            for name in [
                MIX_UP4_KERNEL,
                MIX_UP4_KERNEL_F,
                MIX_UP4_KERNEL_W,
                MIX_UP4_KERNEL_X,
            ] {
                if ptx_entry_params(&text, name) != Some(MIX_UP4_NARGS) {
                    eprintln!(
                        "[mix-up4] PTX の {name} の引数が {MIX_UP4_NARGS} 本でない → 従来の融合版へ。\
                         `bash build_ptx.sh mix_up4` で再生成する"
                    );
                    return None;
                }
            }
            image.push(0);
            let lib = lib().ok()?;
            unsafe {
                let mut module: CUmodule = std::ptr::null_mut();
                if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void)
                    != CUDA_SUCCESS
                {
                    return None;
                }
                let mut get = |name: &str| -> Option<CUfunction> {
                    let mut cname = name.as_bytes().to_vec();
                    cname.push(0);
                    let mut f: CUfunction = std::ptr::null_mut();
                    if (lib.module_get_function)(&mut f, module, cname.as_ptr() as *const c_char)
                        != CUDA_SUCCESS
                    {
                        eprintln!("[mix-up4] PTX に {name} が無い → 従来の融合版へ");
                        return None;
                    }
                    Some(f)
                };
                let up4 = get(MIX_UP4_KERNEL)?;
                let up4f = get(MIX_UP4_KERNEL_F)?;
                let up4w = get(MIX_UP4_KERNEL_W)?;
                let up4x = get(MIX_UP4_KERNEL_X)?;
                Some(MixUp4Kernels { module, up4, up4f, up4w, up4x })
            }
        })
        .as_ref()
}


/// 1 対 = 4 レーンの up が載っているか (テスト / 診断用)。
pub fn mix_up4_loaded() -> bool {
    mix_up4_kernels().is_some()
}


/// 台帳 (`gate_audit::DEFAULT_ON_GATES`) 用の bool。
/// 1 対 = 4 レーンの up を選ぶ値 (u4 / u4f / u4w / u4x) のときだけ true。
pub fn mix_up4_gate_on() -> bool {
    mix_up4_plan(mix_upq8_on()).is_some()
}


/// `mix_upq8_on()` の 3/4/5/6 を (種類 0=u4 1=u4f 2=u4w 3=u4x, JC, threads) へ。
/// 0/1/2 とその他は `None` (従来経路)。
pub fn mix_up4_plan(v: u8) -> Option<(u8, usize, u32)> {
    match v {
        3 => Some((0, MIX_UP4_JC, MIX_UP4_THREADS)),
        4 => Some((1, MIX_UP4_JC, MIX_UP4_THREADS)),
        5 => Some((2, MIX_UP4_JC_W, MIX_UP4_THREADS_W)),
        6 => Some((3, MIX_UP4_JC_X, MIX_UP4_THREADS_X)),
        _ => None,
    }
}


static MIX_UP4_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);


/// 1 対 = 4 レーンの up を撃った回数 (診断。`[mix-upq8] up4=` に出る)。
pub fn mix_up4_hits() -> u64 {
    MIX_UP4_HITS.load(std::sync::atomic::Ordering::Relaxed)
}


pub fn mix_up4_reset() {
    MIX_UP4_HITS.store(0, std::sync::atomic::Ordering::Relaxed);
}


// ===========================================================================
// task 382: mix の silu + up + hc_mixed を 1 本に畳む (`DENNO_MIX_UPQ8`)
// ===========================================================================
//
// 実形 (n_in = 320 = 5 対、n_out = 10240、e = 2560、hc = 4) で、従来の 3 本は
//   lo_silu 3.3 us (2 ブロック、320 要素) + up 13.1 us (3.48 MB を 266 GB/s)
//   + hc_mixed 5.8 us (gate 40 KB の往復 + 起動)
// を払っていた (t372a の `[dec-kprof-avg]`)。up が遅いのは v5 の `lanes=1`
// (1 スレッド 1 行) ではスレッド数が n_out で頭打ちになり 4 warp/SM しか
// 出ないため。融合版はレーン 5 (1 レーン = 1 対) にして 5 倍のスレッドを出し、
// silu と hc_mixed も同じカーネルの中で済ませる (`kernels_mix_upmix_q8.cu`)。
//
// 丸め順は v5 + `dno_hc_mixed_f32` と揃えていない (レーン 0..4 の逐次加算)。
// ゲート `DENNO_MIX_UPQ8` は既定 on (lab t382b の 2 ラウンドで採否 yes)、
// `=0` で従来の 3 本経路、`=check` で従来経路の出力と max|diff| を比べる。

/// 融合版のカーネル名。
const MIX_UPQ8_KERNEL_NAME: &str = "dno_mix_upmix_q8_0";


/// 融合版の引数の数 (lo, w, xn, out, n_in, e, inv_hc, row_bytes)。
const MIX_UPQ8_NARGS: usize = 8;


/// `.cu` の UPQ8_JC (1 ブロックが受け持つ e の数)。
const MIX_UPQ8_JC: usize = 6;


/// `.cu` の UPQ8_LANES (1 レーン = 1 対 = 64 要素)。
const MIX_UPQ8_LANES: usize = 5;


/// `.cu` の UPQ8_THREADS。
const MIX_UPQ8_THREADS: u32 = 128;


pub(crate) struct MixUpq8Kernels {
    #[allow(dead_code)]
    module: CUmodule,
    f: CUfunction,
}


unsafe impl Send for MixUpq8Kernels {}

unsafe impl Sync for MixUpq8Kernels {}


static MIX_UPQ8_KERNELS: OnceLock<Option<MixUpq8Kernels>> = OnceLock::new();


fn mix_upq8_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_MIX_UPQ8_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_mix_upmix_q8.ptx"),
    }
}


/// 融合版のカーネルを取得する (初回だけ PTX を読む。失敗は `None` を焼き付ける)。
fn mix_upq8_kernels() -> Option<&'static MixUpq8Kernels> {
    MIX_UPQ8_KERNELS
        .get_or_init(|| {
            let path = mix_upq8_ptx_path();
            let mut image = match std::fs::read(&path) {
                Ok(b) => b,
                Err(_) => return None,
            };
            let text = String::from_utf8_lossy(&image).to_string();
            if ptx_entry_params(&text, MIX_UPQ8_KERNEL_NAME) != Some(MIX_UPQ8_NARGS) {
                eprintln!(
                    "[mix-upq8] PTX の引数が {MIX_UPQ8_NARGS} 本でない → 従来の 3 本経路へ。\
                     `bash build_ptx.sh mix_upmix_q8` で再生成する"
                );
                return None;
            }
            image.push(0);
            let lib = lib().ok()?;
            unsafe {
                let mut module: CUmodule = std::ptr::null_mut();
                if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void)
                    != CUDA_SUCCESS
                {
                    return None;
                }
                let mut cname = MIX_UPQ8_KERNEL_NAME.as_bytes().to_vec();
                cname.push(0);
                let mut f: CUfunction = std::ptr::null_mut();
                if (lib.module_get_function)(&mut f, module, cname.as_ptr() as *const c_char)
                    != CUDA_SUCCESS
                {
                    eprintln!("[mix-upq8] PTX に {MIX_UPQ8_KERNEL_NAME} が無い → 従来の 3 本経路へ");
                    return None;
                }
                Some(MixUpq8Kernels { module, f })
            }
        })
        .as_ref()
}


/// 融合版が載っているか (テスト / 診断用)。
pub fn mix_upq8_loaded() -> bool {
    mix_upq8_kernels().is_some()
}


/// 形の門番 (純関数)。`.cu` の前提 (n_in = 64 * 5、hc = 4) と同じ。
pub fn mix_upq8_shape_ok(n_in: usize, e: usize, hc: usize) -> bool {
    n_in == 64 * MIX_UPQ8_LANES && e > 0 && hc == 4
}


/// `DENNO_MIX_UPQ8` の解釈 (純関数)。0 = 従来経路 / 1 = 融合 / 2 = 比較 /
/// 3 = 1 対 4 レーン (u4) / 4 = 同 (exp だけ SFU、**既定**) / 5 = 同 (JC=12) /
/// 6 = 同 (JC=4 + SFU)。
///
/// lab t382b (2 ラウンド、base 16.2 / fused 15.8 = -2.5%、mix 段 4.38 -> 3.93 ms/tok
/// = -10.3%、prefill を変えず、errors 0、md5 6 コンテキスト SAME) で融合版を採り、
/// task 437 の lab t437b (2 ラウンド x 6 COND、mix 段 3.02/3.09 -> **2.82/2.82
/// ms/tok** = -6.6%、短い decode 11.6 -> 11.3-11.4、14k 後 13.4 -> 13.0、errors 0、
/// md5 6 コンテキスト SAME) で **1 対 = 4 レーンの u4f を既定にした**。
/// u4f は exp だけ `__expf` (SFU、f32 で 2 ULP 程度) なので、glibc 一致の exp で
/// 回したいときは `DENNO_MIX_UPQ8=u4`、従来の融合版は `=1`、3 本経路は `=0`。
const MIX_UPQ8_DEFAULT: u8 = 4;


pub fn mix_upq8_from_value(v: Option<&str>) -> u8 {
    match v.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("1") | Some("on") | Some("true") | Some("yes") => 1,
        Some("check") | Some("2") => 2,
        // task 437: 1 対 = 4 レーンの up (`kernels_mix_up4.cu`)。
        Some("3") | Some("u4") | Some("up4") => 3,
        Some("4") | Some("u4f") | Some("up4f") => 4,
        Some("5") | Some("u4w") | Some("up4w") => 5,
        Some("6") | Some("u4x") | Some("up4x") => 6,
        Some("0") | Some("off") | Some("false") | Some("no") | Some("") => 0,
        _ => MIX_UPQ8_DEFAULT,
    }
}


pub const MIX_UPQ8_ENV: &str = "DENNO_MIX_UPQ8";


/// ゲートの実値 (プロセスで 1 回だけ env を読む)。
pub fn mix_upq8_on() -> u8 {
    #[cfg(test)]
    if let Some(v) = MIX_UPQ8_FORCE.with(|c| c.get()) {
        return v;
    }
    static ON: OnceLock<u8> = OnceLock::new();
    *ON.get_or_init(|| mix_upq8_from_value(std::env::var(MIX_UPQ8_ENV).ok().as_deref()))
}


/// 台帳 (`gate_audit::DEFAULT_ON_GATES`) 用の bool。
pub fn mix_upq8_gate_on() -> bool {
    mix_upq8_on() != 0
}


#[cfg(test)]
thread_local! {
    pub(crate) static MIX_UPQ8_FORCE: std::cell::Cell<Option<u8>> = const { std::cell::Cell::new(None) };
}


/// テスト用: ゲートの強制 (`None` = env の解釈)。
#[cfg(test)]
pub(crate) fn mix_upq8_set_for_test(v: Option<u8>) {
    MIX_UPQ8_FORCE.with(|c| c.set(v));
}


static MIX_UPQ8_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);


/// 融合カーネルを撃った回数 (診断。`PROFILE` の `[mix-upq8]` 行)。
pub fn mix_upq8_hits() -> u64 {
    MIX_UPQ8_HITS.load(std::sync::atomic::Ordering::Relaxed)
}


pub fn mix_upq8_reset() {
    MIX_UPQ8_HITS.store(0, std::sync::atomic::Ordering::Relaxed);
}


/// `=check` の比較結果を積む (鎖を回したスレッドがそのまま呼ぶ)。
fn mix_upq8_note_check(worst: f64, worst_ref: f64, bad: u64, n: u64) {
    UPQ8_CHECK.with(|c| {
        let mut s = c.borrow_mut();
        if worst > s.0 {
            s.0 = worst;
            s.1 = worst_ref;
        }
        s.2 += bad;
        s.3 += n;
    });
}


/// `=check` の集計を 1 行で取り出す (積んでいなければ None)。取り出すと 0 に戻る。
pub fn mix_upq8_check_line() -> Option<String> {
    let v = UPQ8_CHECK.with(|c| {
        let mut s = c.borrow_mut();
        let v = *s;
        *s = (0.0, 0.0, 0, 0);
        v
    });
    if v.3 == 0 {
        return None;
    }
    Some(format!(
        "[mix-upq8] check max|diff|={:.3e} (|ref|={:.3e}) bad={}/{} (閾値 1e-4+1e-5*|ref|)",
        v.0, v.1, v.2, v.3
    ))
}


// ===========================================================================
// task 382: mix の鎖から H2D / D2H を消す (`DENNO_MIX_ZC`)
// ===========================================================================
//
// 鎖 1 回の実測 (t372a、`[dec-kprof-avg]` の op 表) は
//   H2D@a 5.7 us / down 6.4 / lo_silu 3.3 / up 13.1 / hc_mixed 5.8 / D2H@a 2.6
// で、**両端の往復 8.3 us は「xn 40 KB を送る」「mixed 10 KB を戻す」だけ**。
// xn は device から直接読める pinned (`CU_MEMHOSTALLOC_DEVICEMAP`) に置けば
// H2D が丸ごと消え、mixed は hc_mixed にその pinned へ書かせれば D2H が消える。
// カーネルは 1 本も変えない (ポインタが変わるだけ) ので値はビット一致。

/// `DENNO_MIX_ZC` の解釈 (純関数)。返り値は `(H2D を消す, D2H を消す)`。
///
/// - 未設定 / `0` / `off` (既定): 従来 (H2D + D2H)。
/// - `x` / `in`: H2D だけ消す (`down` がホストの `xn` を PCIe 越しに読む)。
/// - `out`: D2H だけ消す (`hc_mixed` がホストメモリへ直接書く)。
/// - `1` / `on` / `both`: 両方。
/// - task 431: `os` / `outstream` / `all` は「D2H を消す + H2D を別ストリームへ」
///   (H2D は消さない。下の [`mix_zc_h2d_stream_from_value`] を見る)。
pub fn mix_zc_from_value(v: Option<&str>) -> (bool, bool) {
    match v.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("x") | Some("in") => (true, false),
        Some("out") => (false, true),
        Some("1") | Some("on") | Some("true") | Some("yes") | Some("both") => (true, true),
        // task 431: 第 3 の指定 (`H2D を別ストリーム`) と組む値。
        Some("os") | Some("outstream") | Some("all") => (false, true),
        _ => (false, false),
    }
}


/// task 431: `DENNO_MIX_ZC` の第 3 の指定 — **xn の H2D を専用の
/// 非ブロッキングストリームへ出す** (純関数)。
///
/// なぜ要るか (lab t431a の `[dec-kprof-avg]` mix band、1 呼び出しあたり):
///   H2D@a 10.3 us / down 14.0 / up 10.8 / router 6.9 / D2H@a 5.4
/// で、**41 KB の H2D が 10.3 us** かかっている。中身は転送 (0.8 us) ではなく
/// 「null ストリームの前の仕事 (attention) が終わるまでの待ち」なので、
/// コピーだけ別ストリームへ出せば前の仕事と重なって消える。鎖のカーネルは
/// null ストリームのままで、コピーの完了イベントを 1 本待たせるだけ。
/// 値は 1 ビットも変わらない (同じバイトが同じ場所へ行く)。
///
/// 値: `stream` / `s` (H2D だけ別ストリーム) と、上の `os` / `outstream` / `all`
/// (D2H も消す)。既定 off。
pub fn mix_zc_h2d_stream_from_value(v: Option<&str>) -> bool {
    matches!(
        v.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("stream")
            | Some("s")
            | Some("os")
            | Some("outstream")
            | Some("all")
            | Some("sx")
            | Some("instream")
    )
}


pub const MIX_ZC_ENV: &str = "DENNO_MIX_ZC";


/// ゲートの実値 (プロセスで 1 回だけ env を読む)。
pub fn mix_zc_enabled() -> (bool, bool) {
    #[cfg(test)]
    if let Some(v) = MIX_ZC_FORCE.with(|c| c.get()) {
        return v;
    }
    static ON: OnceLock<(bool, bool)> = OnceLock::new();
    *ON.get_or_init(|| mix_zc_from_value(std::env::var(MIX_ZC_ENV).ok().as_deref()))
}


#[cfg(test)]
thread_local! {
    pub(crate) static MIX_ZC_FORCE: std::cell::Cell<Option<(bool, bool)>> = const { std::cell::Cell::new(None) };
}


/// テスト用: ゲートの強制 (`None` = env の解釈)。
#[cfg(test)]
pub(crate) fn mix_zc_set_for_test(v: Option<(bool, bool)>) {
    MIX_ZC_FORCE.with(|c| c.set(v));
}


/// task 431: H2D を別ストリームへ出すかの実値 (プロセスで 1 回だけ env を読む)。
pub fn mix_zc_h2d_stream() -> bool {
    #[cfg(test)]
    if let Some(v) = MIX_ZC_H2S_FORCE.with(|c| c.get()) {
        return v;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| mix_zc_h2d_stream_from_value(std::env::var(MIX_ZC_ENV).ok().as_deref()))
}


#[cfg(test)]
thread_local! {
    pub(crate) static MIX_ZC_H2S_FORCE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}


/// テスト用: [`mix_zc_h2d_stream`] の強制 (`None` = env の解釈)。
#[cfg(test)]
pub(crate) fn mix_zc_h2d_stream_set_for_test(v: Option<bool>) {
    MIX_ZC_H2S_FORCE.with(|c| c.set(v));
}


static MIX_ZC_X_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

static MIX_ZC_OUT_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// task 431: xn の H2D を別ストリームへ出した回数 (発火の証拠)。
static MIX_H2S_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);


/// task 431: 別ストリームへ出した H2D の回数 (診断)。
pub fn mix_h2s_hits() -> u64 {
    MIX_H2S_HITS.load(std::sync::atomic::Ordering::Relaxed)
}


/// task 431: 計数を 0 に戻す (リクエスト終端)。
pub fn mix_h2s_reset() {
    MIX_H2S_HITS.store(0, std::sync::atomic::Ordering::Relaxed);
}


/// 実際に H2D を省いた回数 / D2H を省いた回数 (診断)。
pub fn mix_zc_hits() -> (u64, u64) {
    use std::sync::atomic::Ordering as AO;
    (
        MIX_ZC_X_HITS.load(AO::Relaxed),
        MIX_ZC_OUT_HITS.load(AO::Relaxed),
    )
}


pub fn mix_zc_reset() {
    use std::sync::atomic::Ordering as AO;
    MIX_ZC_X_HITS.store(0, AO::Relaxed);
    MIX_ZC_OUT_HITS.store(0, AO::Relaxed);
}


/// task 282: `[mix-qchain]` 行に出す高速 GEMV の発火回数。
pub fn mix_qk_hits() -> u64 {
    MIX_QK_HITS.load(std::sync::atomic::Ordering::Relaxed)
}


/// task 282: 集計を 0 に戻す (リクエスト終端)。
pub fn mix_qk_reset() {
    MIX_QK_HITS.store(0, std::sync::atomic::Ordering::Relaxed);
}


/// task 266: hc mix が **量子型** (Q8_0 / Q5_1 / Q4_K / Q5_K) のときの
/// **down → lo_apply(silu) → up → hc_mixed を 1 回の同期に畳む**経路。
///
/// `xn` は呼び出し側が計算済みの正規化入力 (`[hc][e]`)。UD-Q4_K_XL の mix は
/// Q8_0 で、従来は 1 呼び出し 2 往復 (down の GEMV と up の GEMV がそれぞれ
/// H2D + 同期 + D2H) = 1 トークン 194 起動 / 194 同期だった (REPORT_P261.md)。
/// ここでは中間 (`lo`, `gate`) をデバイスに残したまま 4 本積み、同期は末尾の 1 回
/// だけにする。値は CPU とビット一致 (silu は dno_expf = glibc expf の逐語再現)。
///
/// `false` = 経路が使えない (型・形・PTX・重みの常駐)。呼び出し側は従来経路へ。
#[allow(clippy::too_many_arguments)]
pub fn maybe_gpu_mix_q_chain_into(
    down_uid: u64,
    down_ty: crate::kquant::GgmlType,
    down_bytes: &[u8],
    k1: usize,
    n1: usize,
    up_uid: u64,
    up_ty: crate::kquant::GgmlType,
    up_bytes: &[u8],
    k2: usize,
    n2: usize,
    inv_hc: f32,
    e: usize,
    hc: usize,
    xn: &[f32],
    out: &mut [f32],
    // task 282: GPU の鎖を積んだあと・待つ前にホストで走らせる仕事
    // (hc_inject の GEMV。GPU が遊んでいる間に出せるので待ちに隠れる)。
    pre_wait: Option<&mut dyn FnMut() -> Result<()>>,
    // task 464: 呼び出し側が「`mixed` をホストで読まない」と言っているか
    // (`DENNO_MIX_NOWAIT`)。true なら D2H を撃たず、イベントも待たない
    // (次の鎖 or 後続の GDN submission の同期にまとめる)。
    defer: bool,
) -> bool {
    use std::sync::atomic::Ordering as AO;
    // task 364: ルータ GEMV をこの鎖に載せにいく回数 (テスト専用。GPU 無しでも
    // 数える)。`DENNO_ROUTER_IN_MIX=0` のときは控えが無いので 0 のまま。
    #[cfg(test)]
    if router_in_mix_armed_for(k1) {
        ROUTER_IN_QCHAIN_ARMS.fetch_add(1, AO::Relaxed);
    }
    if !cuda_enabled() || MIX_GPU_BROKEN.load(AO::Relaxed) {
        return false;
    }
    // 形: down は k1 -> n1、up は n1 -> n2 (= hc * e)。
    if k1 == 0
        || n1 == 0
        || n2 == 0
        || e == 0
        || hc == 0
        || k2 != n1
        || n2 != k1
        || n2 / e != hc
        || xn.len() != k1
        || out.len() != e
        || k1 < n1
        || k1 > i32::MAX as usize
        || n1 > i32::MAX as usize
        || n2 > i32::MAX as usize
        || e > i32::MAX as usize
    {
        return false;
    }
    // 4 型だけ (Q8_0 / Q5_1 / Q4_K / Q5_K)。
    let ty_ok = |ty: crate::kquant::GgmlType| {
        use crate::kquant::GgmlType as T;
        matches!(ty, T::Q8_0 | T::Q5_1 | T::Q4K | T::Q5K)
    };
    if !ty_ok(down_ty) || !ty_ok(up_ty) {
        return false;
    }
    if gguf_row_bytes(down_ty, k1).is_none() || gguf_row_bytes(up_ty, n1).is_none() {
        return false;
    }
    let Some(ctx) = shared_ctx() else {
        return false;
    };
    if let Err(err) = ctx.make_current() {
        mix_gpu_broken_once(&format!("cuCtxSetCurrent (mix q chain): {err:#}"));
        return false;
    }
    let Some(f_mixed) = hc_mixed_f32_fn(ctx) else {
        return false;
    };
    let Some(mdk) = mix_dev_kernels() else {
        return false;
    };
    // lo_apply のカーネルが無い PTX (task 266 より前) では通さない。
    let Some(f_silu) = mdk.lo_silu else {
        return false;
    };
    let wd = match weights_for_gguf(down_uid, down_ty, down_bytes, k1, n1) {
        Ok(w) => w.buf.ptr(),
        Err(err) => {
            mix_gpu_broken_once(&format!("mix q chain down 重みの常駐に失敗: {err:#}"));
            return false;
        }
    };
    let wu = match weights_for_gguf(up_uid, up_ty, up_bytes, n1, n2) {
        Ok(w) => w.buf.ptr(),
        Err(err) => {
            mix_gpu_broken_once(&format!("mix q chain up 重みの常駐に失敗: {err:#}"));
            return false;
        }
    };
    let Some(lib) = lib().ok() else {
        return false;
    };
    const MIXED_THREADS: usize = 256;
    // task 282: 高速 GEMV の前置きレイアウト (重みポインタ 2 + segment 2 + xn)。
    let (off_wd, off_wu, off_sd, off_su, off_x) = mix_qk_layout();
    let mut pre_wait = pre_wait;
    // task 364: 鎖の末尾に積むルータ GEMV の重み (同期の後まで生かす)。
    let mut router_w: Option<GgufWRef> = None;
    // task 464: 前回 defer した鎖がまだ流れていれば、scratch (`stage` / `hx`) を
    // 上書きする前に待つ。実際には直後の GDN submission が同期しているので
    // ここで待つことはほぼ無い (イベントは完了済み)。
    if let Err(err) = mix_pending_drain() {
        mix_gpu_broken_once(&format!("mix pending wait: {err:#}"));
        return false;
    }
    let r = MIX_GPU_SC.with(|cell| -> Result<()> {
        let _kp_win = dec_kprof::window(
            dec_kprof::MIX,
            dec_kprof::cur_layer(),
            dec_kprof::cur_pos(),
            3,
        );
        let mut s = cell.borrow_mut();
        // task 431: H2D を別ストリームへ出すか (既定 off)。ストリームと
        // イベントは `&mut *s` を分解する前に取る。
        let h2s = if mix_zc_h2d_stream() {
            mix_h2s_ensure(&mut s)
        } else {
            None
        };
        let MixGpuScratch {
            x: dx,
            lo: dlo,
            out: dout,
            mixed: dm,
            hx,
            ho,
            stage,
            hstage,
            xmap,
            omap,
            chk: dchk,
            hchk,
            ..
        } = &mut *s;
        // この H2D を積むストリーム (`h2s` が無ければ null = 従来)。
        let h2d_stream: CUstream = h2s.map(|(st, _)| st).unwrap_or(std::ptr::null_mut());
        // task 382: 鎖の両端の往復 (H2D xn / D2H mixed) を消すかの実値。
        // H2D を消せるのは down/up が両方 v5 のときだけ (stage の前置きを
        // 読むのは Blk / Seg のカーネルで、v5 は重みポインタを引数で受ける)。
        let zc = mix_zc_enabled();
        // task 282: mix の GEMV を向きごとに選ぶ (`DENNO_MIX_QK`)。
        // seg_k 系は Q8_0 のときだけ (mix の実型)。形が合わなければ旧 GEMV。
        let (want_down, want_up) = mix_qk_enabled();
        let q8 = down_ty == crate::kquant::GgmlType::Q8_0
            && up_ty == crate::kquant::GgmlType::Q8_0;
        let rd = gguf_row_bytes(down_ty, k1);
        let ru = gguf_row_bytes(up_ty, n1);
        let kern = if q8 { seg_k_kernels() } else { None };
        // task 286: v5 形 (既存 `dno_gemv_q8_0_v5`)。形が合わなければ下で落とす。
        let v5_down = if want_down == MixGemv::V5 {
            mix_v5_plan(down_ty, k1, n1)
        } else {
            None
        };
        let v5_up = if want_up == MixGemv::V5 {
            mix_v5_plan(up_ty, n1, n2)
        } else {
            None
        };
        // task 418: dn5 (1 ブロック 1 行) は Q8_0 の実形だけ。門番は
        // `mix_dn5_warps` (np = n_in/64 が 32 の倍数、warp 数 <= 8) と PTX。
        let dn5_warps = if want_down == MixGemv::Dn5
            && down_ty == crate::kquant::GgmlType::Q8_0
            && mix_dn5_kernels().is_some()
        {
            mix_dn5_warps(k1)
        } else {
            None
        };
        // task 422: dn6 (1 ブロック rows 行) は dn5 の一般化。rows は
        // `DENNO_MIX_QK` の値 (`dn6` = 4 行) から。
        let dn6_plan = if want_down == MixGemv::Dn6
            && down_ty == crate::kquant::GgmlType::Q8_0
            && mix_dn6_kernels().is_some()
        {
            mix_dn6_plan(k1, n1, mix_dn6_rows())
        } else {
            None
        };
        // task 431: dn7 (1 対 4 レーン) の起動形。門番は `mix_dn7_plan`。
        let dn7_plan = if want_down == MixGemv::Dn7
            && down_ty == crate::kquant::GgmlType::Q8_0
            && mix_dn7_kernels().is_some()
        {
            mix_dn7_plan(k1, n1)
        } else {
            None
        };
        let down_kind = match (want_down, kern, rd) {
            (MixGemv::Dn7, _, _) if dn7_plan.is_some() => MixGemv::Dn7,
            (MixGemv::Dn6, _, _) if dn6_plan.is_some() => MixGemv::Dn6,
            (MixGemv::Dn5, _, _) if dn5_warps.is_some() => MixGemv::Dn5,
            (MixGemv::V5, _, _) if v5_down.is_some() => MixGemv::V5,
            (MixGemv::Blk, Some(k), Some(_)) if k.mix_rowblk.is_some() => MixGemv::Blk,
            (MixGemv::Seg, Some(k), Some(_)) if moe_batch_slot(down_ty).is_some() => MixGemv::Seg,
            _ => MixGemv::Old,
        };
        let up_kind = match (want_up, kern, ru) {
            (MixGemv::V5, _, _) if v5_up.is_some() => MixGemv::V5,
            (MixGemv::R8, Some(k), Some(_)) if k.mix_r8.is_some() => MixGemv::R8,
            (MixGemv::Seg, Some(k), Some(_)) if moe_batch_slot(up_ty).is_some() => MixGemv::Seg,
            _ => MixGemv::Old,
        };
        let fast_down = down_kind != MixGemv::Old;
        let fast_up = up_kind != MixGemv::Old;
        let need_stage = fast_down || fast_up;
        Scratch::ensure(dx, k1 * 4)?;
        Scratch::ensure(dlo, n1 * 4)?;
        Scratch::ensure(dout, n2 * 4)?;
        Scratch::ensure(dm, e * 4)?;
        Scratch::ensure_pin(hx, k1 * 4)?;
        Scratch::ensure_pin(ho, e * 4)?;
        if need_stage {
            Scratch::ensure(stage, MIX_QK_PROLOGUE + k1 * 4)?;
            Scratch::ensure_pin(hstage, MIX_QK_PROLOGUE + k1 * 4)?;
        }
        // task 382: device から見える pinned を用意する。確保に失敗したら
        // その方向だけ従来経路へ落ちる (鎖そのものは止めない)。
        let mut zc_x = zc.0 && down_kind == MixGemv::V5 && up_kind == MixGemv::V5;
        let mut zc_out = zc.1;
        if zc_x && MapMem::ensure(xmap, k1 * 4).is_err() {
            zc_x = false;
        }
        if zc_out && MapMem::ensure(omap, e * 4).is_err() {
            zc_out = false;
        }
        // task 382: silu + up + hc_mixed の融合版を撃つか (`DENNO_MIX_UPQ8`)。
        // 形 (up が v5 / Q8_0 / n1 = 320 / hc = 4) が揃わなければ従来の 3 本。
        let fu_pre: Option<u8> = {
            let g = mix_upq8_on();
            if g != 0
                && fast_up
                && up_kind == MixGemv::V5
                && up_ty == crate::kquant::GgmlType::Q8_0
                && mix_upq8_shape_ok(n1, e, hc)
                && !zc_x
            {
                Some(g)
            } else {
                None
            }
        };
        let (dx, dlo, dout, dm) = (
            dx.as_ref().unwrap(),
            dlo.as_ref().unwrap(),
            dout.as_ref().unwrap(),
            dm.as_ref().unwrap(),
        );
        let (hx, ho) = (hx.as_mut().unwrap(), ho.as_mut().unwrap());
        // task 431: 実際に積んだ H2D の本数 (0 ならイベントを待たせない)。
        let mut h2d_copies = 0usize;
        if need_stage && zc_x {
            // task 382: xn を device から見える pinned へ置くだけ (転送命令なし)。
            dec_kprof::step("mix_h2d_pack", || {
                let m = xmap.as_mut().expect("確保済み");
                m.pin.as_mut_slice::<f32>()[..k1].copy_from_slice(xn);
            });
        } else if need_stage {
            // 前置き (重みポインタ 2 本 + segment 表 2 枚 + xn) を 1 本で送る。
            let st = stage.as_ref().expect("確保済み");
            let hs = hstage.as_mut().expect("確保済み");
            dec_kprof::step("mix_h2d_pack", || {
                let head = hs.as_mut_slice::<u8>();
                head[off_wd..off_wd + 8].copy_from_slice(&wd.to_le_bytes());
                head[off_wu..off_wu + 8].copy_from_slice(&wu.to_le_bytes());
                for off in [off_sd, off_su] {
                    head[off..off + 16].fill(0);
                    // seg[1] = m (decode は 1)、seg[2] / seg[3] = x_off / out_off = 0。
                    head[off + 4..off + 8].copy_from_slice(&1u32.to_le_bytes());
                }
                hs.as_mut_slice::<f32>()[off_x / 4..off_x / 4 + k1].copy_from_slice(xn);
            });
            let kp_cp = dec_kprof::op_begin(
                || "H2D@a".to_string(),
                (0, 0, 0),
                (0, 0, 0),
                0,
                MIX_QK_PROLOGUE + k1 * 4,
            );
            unsafe {
                check(
                    (lib.memcpy_htod_async)(
                        st.ptr(),
                        hs.ptr as *const c_void,
                        MIX_QK_PROLOGUE + k1 * 4,
                        h2d_stream,
                    ),
                    "cuMemcpyHtoDAsync (mix qk)",
                )?;
            }
            dec_kprof::op_end(kp_cp);
            h2d_copies += 1;
        } else {
            dec_kprof::step("mix_h2d_pack", || {
                hx.as_mut_slice::<f32>()[..k1].copy_from_slice(xn)
            });
            let kp_cp = dec_kprof::op_begin(
                || "H2D@a".to_string(),
                (0, 0, 0),
                (0, 0, 0),
                0,
                k1 * 4,
            );
            unsafe {
                check(
                    (lib.memcpy_htod_async)(
                        dx.ptr(),
                        hx.ptr as *const c_void,
                        k1 * 4,
                        h2d_stream,
                    ),
                    "cuMemcpyHtoDAsync (mix q chain)",
                )?;
            }
            dec_kprof::op_end(kp_cp);
            h2d_copies += 1;
        }
        // down が旧 GEMV のときは xn を従来どおり dx にも送る (前置き側の xn は
        // 使わない。up だけ高速にする組合せのための 40 KB の二重送り)。
        if !fast_down {
            dec_kprof::step("mix_h2d_pack", || {
                hx.as_mut_slice::<f32>()[..k1].copy_from_slice(xn)
            });
            let kp_cp = dec_kprof::op_begin(
                || "H2D@a".to_string(),
                (0, 0, 0),
                (0, 0, 0),
                0,
                k1 * 4,
            );
            unsafe {
                check(
                    (lib.memcpy_htod_async)(
                        dx.ptr(),
                        hx.ptr as *const c_void,
                        k1 * 4,
                        h2d_stream,
                    ),
                    "cuMemcpyHtoDAsync (mix q chain)",
                )?;
            }
            dec_kprof::op_end(kp_cp);
            h2d_copies += 1;
        }
        // task 431: 別ストリームへ出した H2D を、鎖のカーネル (null ストリーム)
        // の前に 1 本のイベントで待たせる。ここまではホスト側の仕事だけで、
        // コピーは前の層の GPU 仕事と重なって走る。
        if let Some((st_side, ev_side)) = h2s {
            if h2d_copies > 0 {
                let ok = match (lib.event_record, lib.stream_wait_event) {
                    (Some(rec), Some(wev)) => unsafe {
                        rec(ev_side, st_side) == CUDA_SUCCESS
                            && wev(std::ptr::null_mut(), ev_side, 0) == CUDA_SUCCESS
                    },
                    _ => false,
                };
                if ok {
                    MIX_H2S_HITS.fetch_add(1, AO::Relaxed);
                } else {
                    // 順序が取れない環境では安全側に倒す (ホストで待つ)。
                    ctx.sync()?;
                }
            }
        }
        // デバイス側の xn の置き場所 (down が高速なら前置きバッファの末尾)。
        let xdev = if zc_x {
            xmap.as_ref().expect("確保済み").dev
        } else if fast_down {
            stage.as_ref().expect("確保済み").ptr() + off_x as u64
        } else {
            dx.ptr()
        };
        // 1. down (量子 GEMV, デバイス内)
        if fast_down {
            let st = stage.as_ref().expect("確保済み");
            let rb_d = rd.expect("形は検査済み");
            unsafe {
                match down_kind {
                    MixGemv::Dn5 => {
                        // task 418: 1 ブロック 1 行・1 レーン 1 対。重みは生ポインタ。
                        let warps = dn5_warps.expect("検査済み");
                        launch_mix_dn5(ctx, wd, xdev, dlo.ptr(), k1, n1, rb_d, warps)?;
                        MIX_DN5_HITS.fetch_add(1, AO::Relaxed);
                    }
                    MixGemv::Dn6 => {
                        // task 422: 1 ブロック rows 行・1 レーン 1 対 (dn5 と同じ値)。
                        let (warps, rows) = dn6_plan.expect("検査済み");
                        launch_mix_dn6(ctx, wd, xdev, dlo.ptr(), k1, n1, rb_d, warps, rows)?;
                        MIX_DN6_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    MixGemv::Dn7 => {
                        // task 431: 1 対 4 レーン (スレッド数 4 倍、丸め順が変わる)。
                        let (rows, threads) = dn7_plan.expect("検査済み");
                        launch_mix_dn7(ctx, wd, xdev, dlo.ptr(), k1, n1, rb_d, rows, threads)?;
                        MIX_DN7_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    MixGemv::V5 => {
                        // task 286: v5 形 (68 B の対読み + 先読み)。重みは生ポインタ。
                        let plan = v5_down.as_ref().expect("検査済み");
                        launch_mix_v5(ctx, plan, wd, xdev, dlo.ptr(), k1, n1)?;
                    }
                    MixGemv::Blk => {
                        // 1 行 = 1 ブロック (256 スレッド)。k を 8 warp に分ける。
                        let f = kern.and_then(|k| k.mix_rowblk).expect("検査済み");
                        let (mut pw, mut px, mut po) = (wd, xdev, dlo.ptr());
                        let (mut ni, mut no, mut rb) =
                            (k1 as c_int, n1 as c_int, rb_d as c_int);
                        let mut args = [
                            &mut pw as *mut _ as *mut c_void,
                            &mut px as *mut _ as *mut c_void,
                            &mut po as *mut _ as *mut c_void,
                            &mut ni as *mut _ as *mut c_void,
                            &mut no as *mut _ as *mut c_void,
                            &mut rb as *mut _ as *mut c_void,
                        ];
                        ctx.launch_async(f, (n1 as u32, 1, 1), (256, 1, 1), 32, &mut args)?;
                    }
                    _ => {
                        let fk = kern.expect("検査済み");
                        let slot_d = moe_batch_slot(down_ty).expect("検査済み");
                        let mut pw = st.ptr() + off_wd as u64;
                        let mut px = xdev;
                        let mut po = dlo.ptr();
                        let mut ps = st.ptr() + off_sd as u64;
                        let (mut ns, mut ni, mut no, mut rb) =
                            (1 as c_int, k1 as c_int, n1 as c_int, rb_d as c_int);
                        let mut args = [
                            &mut pw as *mut _ as *mut c_void,
                            &mut px as *mut _ as *mut c_void,
                            &mut po as *mut _ as *mut c_void,
                            &mut ps as *mut _ as *mut c_void,
                            &mut ns as *mut _ as *mut c_void,
                            &mut ni as *mut _ as *mut c_void,
                            &mut no as *mut _ as *mut c_void,
                            &mut rb as *mut _ as *mut c_void,
                        ];
                        ctx.launch_async(
                            fk.fns[slot_d],
                            (((n1 + 3) / 4) as u32, 1, 1),
                            (128, 1, 1),
                            0,
                            &mut args,
                        )?;
                    }
                }
            }
        } else {
            let ok = ctx.moe_gemv_dev_ptrs(down_ty, wd, dx.ptr(), dlo.ptr(), k1, n1)?;
            if !ok {
                return Err(anyhow!("down の GEMV 計画が無い"));
            }
        }
        // task 382: silu + up + hc_mixed を 1 本に畳む (`DENNO_MIX_UPQ8`)。
        // 形 (n1 = 320 / hc = 4 / Q8_0 / v5) と PTX が揃ったときだけ。
        let upq8 = fu_pre.unwrap_or(0);
        let loaded = fu_pre.is_some() && mix_upq8_kernels().is_some();
        // task 437: 1 対 = 4 レーンの up (u4 / u4f / u4w) が載っていればそちら。
        let up4_plan = if fu_pre.is_some() && mix_up4_kernels().is_some() {
            mix_up4_plan(upq8)
        } else {
            None
        };
        // fu_use: 融合版 (従来 or 1 対 4 レーン) を出力に使う / fu_chk: 従来経路の出力と比べるだけ。
        let fu_use = (loaded && upq8 == 1) || up4_plan.is_some();
        let fu_chk = loaded && upq8 == 2;
        let fu = fu_use || fu_chk;
        if fu_chk {
            Scratch::ensure(dchk, e * 4)?;
            Scratch::ensure_pin(hchk, e * 4)?;
        }
        // 4. hc_mixed (xn と gate から mixed) の行き先。
        // task 382: zc_out のときはホストの pinned へ直接書く (D2H を消す)。
        // task 431: 融合版 (`DENNO_MIX_UPQ8`) は **この mdev をそのまま受け取る**
        // ようにした (それまでは融合版だけ `dm` へ書いていて、zc_out と併用すると
        // D2H を省いた先が空のままだった = 併用できなかった)。
        let mdev = if zc_out {
            omap.as_ref().expect("確保済み").dev
        } else {
            dm.ptr()
        };
        if fu {
            // task 437: u4 系はカーネルと起動形 (JC x 80 スレッド) が違う。
            let (f, grid, threads) = if let Some((kind, jc, th)) = up4_plan {
                let k = mix_up4_kernels().expect("直前に確認");
                let f = match kind {
                    1 => k.up4f,
                    2 => k.up4w,
                    3 => k.up4x,
                    _ => k.up4,
                };
                (f, (e.div_ceil(jc)).max(1) as u32, th)
            } else {
                let k = mix_upq8_kernels().expect("直前に確認");
                (
                    k.f,
                    (e.div_ceil(MIX_UPQ8_JC)).max(1) as u32,
                    MIX_UPQ8_THREADS,
                )
            };
            let pout = if fu_use {
                mdev
            } else {
                dchk.as_ref().expect("確保済み").ptr()
            };
            let (mut plo, mut pw2, mut pxn, mut pm2) = (dlo.ptr(), wu, xdev, pout);
            let (mut ni2, mut ec2) = (n1 as c_int, e as c_int);
            let mut inv2 = inv_hc;
            let mut rb2 = ru.expect("形は検査済み") as c_int;
            let mut args = [
                &mut plo as *mut _ as *mut c_void,
                &mut pw2 as *mut _ as *mut c_void,
                &mut pxn as *mut _ as *mut c_void,
                &mut pm2 as *mut _ as *mut c_void,
                &mut ni2 as *mut _ as *mut c_void,
                &mut ec2 as *mut _ as *mut c_void,
                &mut inv2 as *mut _ as *mut c_void,
                &mut rb2 as *mut _ as *mut c_void,
            ];
            unsafe {
                ctx.launch_async(f, (grid, 1, 1), (threads, 1, 1), 0, &mut args)?;
            }
            if up4_plan.is_some() {
                MIX_UP4_HITS.fetch_add(1, AO::Relaxed);
            } else if fu_use {
                MIX_UPQ8_HITS.fetch_add(1, AO::Relaxed);
            }
        }
        // 2. lo_apply (silu(v * inv_hc)) — in-place (融合版では fused が中で行う)
        if !fu_use {
        unsafe {
            let (mut pv, mut nc) = (dlo.ptr(), n1 as c_int);
            let mut invc = inv_hc;
            let mut args = [
                &mut pv as *mut _ as *mut c_void,
                &mut nc as *mut _ as *mut c_void,
                &mut invc as *mut _ as *mut c_void,
            ];
            let grid = n1.div_ceil(THREADS_Q).max(1) as u32;
            ctx.launch_async(f_silu, (grid, 1, 1), (THREADS_Q as u32, 1, 1), 0, &mut args)?;
        }
        }
        // 3. up (量子 GEMV)
        if !fu_use && fast_up {
            let st = stage.as_ref().expect("確保済み");
            let rb_u = ru.expect("形は検査済み");
            let fk = kern.expect("検査済み");
            unsafe {
                if up_kind == MixGemv::V5 {
                    // task 286: v5 形。n_in=320 (10 ブロック) は lanes=1 / 128 行。
                    let plan = v5_up.as_ref().expect("検査済み");
                    launch_mix_v5(ctx, plan, wu, dlo.ptr(), dout.ptr(), n1, n2)?;
                } else if up_kind == MixGemv::R8 {
                    // 1 warp = 8 行。1 ブロック = 4 warp = 32 行。
                    let f = fk.mix_r8.expect("検査済み");
                    let mut pw = st.ptr() + off_wu as u64;
                    let mut px = dlo.ptr();
                    let mut po = dout.ptr();
                    let mut ps = st.ptr() + off_su as u64;
                    let (mut ns, mut ni, mut no, mut rb) =
                        (1 as c_int, n1 as c_int, n2 as c_int, rb_u as c_int);
                    let mut args = [
                        &mut pw as *mut _ as *mut c_void,
                        &mut px as *mut _ as *mut c_void,
                        &mut po as *mut _ as *mut c_void,
                        &mut ps as *mut _ as *mut c_void,
                        &mut ns as *mut _ as *mut c_void,
                        &mut ni as *mut _ as *mut c_void,
                        &mut no as *mut _ as *mut c_void,
                        &mut rb as *mut _ as *mut c_void,
                    ];
                    ctx.launch_async(
                        f,
                        (((n2 + 31) / 32) as u32, 1, 1),
                        (128, 1, 1),
                        0,
                        &mut args,
                    )?;
                } else {
                    let slot_u = moe_batch_slot(up_ty).expect("検査済み");
                    let mut pw = st.ptr() + off_wu as u64;
                    let mut px = dlo.ptr();
                    let mut po = dout.ptr();
                    let mut ps = st.ptr() + off_su as u64;
                    let (mut ns, mut ni, mut no, mut rb) =
                        (1 as c_int, n1 as c_int, n2 as c_int, rb_u as c_int);
                    let mut args = [
                        &mut pw as *mut _ as *mut c_void,
                        &mut px as *mut _ as *mut c_void,
                        &mut po as *mut _ as *mut c_void,
                        &mut ps as *mut _ as *mut c_void,
                        &mut ns as *mut _ as *mut c_void,
                        &mut ni as *mut _ as *mut c_void,
                        &mut no as *mut _ as *mut c_void,
                        &mut rb as *mut _ as *mut c_void,
                    ];
                    ctx.launch_async(
                        fk.fns[slot_u],
                        (((n2 + 3) / 4) as u32, 1, 1),
                        (128, 1, 1),
                        0,
                        &mut args,
                    )?;
                }
            }
        } else if !fu_use {
            let ok = ctx.moe_gemv_dev_ptrs(up_ty, wu, dlo.ptr(), dout.ptr(), n1, n2)?;
            if !ok {
                return Err(anyhow!("up の GEMV 計画が無い"));
            }
        }
        // 4. hc_mixed (xn と gate から mixed) — 行き先は上で決めた `mdev`。
        if !fu_use {
            unsafe {
                let (mut px, mut pg, mut pm) = (xdev, dout.ptr(), mdev);
                let (mut ec, mut hcc) = (e as c_int, hc as c_int);
                let mut invc2 = inv_hc;
                let mut args = [
                    &mut px as *mut _ as *mut c_void,
                    &mut pg as *mut _ as *mut c_void,
                    &mut pm as *mut _ as *mut c_void,
                    &mut ec as *mut _ as *mut c_void,
                    &mut hcc as *mut _ as *mut c_void,
                    &mut invc2 as *mut _ as *mut c_void,
                ];
                let grid = (e.div_ceil(MIXED_THREADS).max(1)) as u32;
                ctx.launch_async(
                    f_mixed,
                    (grid, 1, 1),
                    (MIXED_THREADS as u32, 1, 1),
                    0,
                    &mut args,
                )?;
            }
        }
        // task 364: ルータ GEMV を鎖の末尾に 1 発積む。
        // task 307 の `DENNO_ROUTER_IN_MIX` は BF16 の chain2 / chain3 にしか
        // 入っておらず、新しい既定モデル (unsloth 11-13、mix は Q8_0) はこちらを
        // 通るため、ルータだけが鎖の外に残って層ごとに 1 往復 (起動 + D2H + 同期)
        // していた。入力は `dm` (= `mixed` のデバイスコピー) をそのまま読み、
        // D2H は下の鎖の同期 (イベント 1 本) でまとめて待つ。載らなければ従来経路が
        // 別途走る (同じカーネル・同じ入力なので値は変わらない)。
        let enq = router_in_mix_enqueue(ctx, mdev, e, &mut router_w);
        if enq != 0 {
            ROUTER_IN_QCHAIN_HITS.fetch_add(1, AO::Relaxed);
        }
        // task 464: defer の回は `mixed` をホストへ降ろさない (GDN が `dm` を
        // そのまま読む)。降ろさないので下のイベント待ちも要らない。
        let mut d2h_done = false;
        if !zc_out && !defer {
            let kp_cp = dec_kprof::op_begin(
                || "D2H@a".to_string(),
                (0, 0, 0),
                (0, 0, 0),
                0,
                e * 4,
            );
            unsafe {
                check(
                    (lib.memcpy_dtoh_async)(ho.ptr, dm.ptr(), e * 4, std::ptr::null_mut()),
                    "cuMemcpyDtoHAsync (mix q chain)",
                )?;
            }
            d2h_done = true;
            dec_kprof::op_end(kp_cp);
            crate::prof::mark_d2h_src(crate::prof::D2H_SRC_LINEAR);
        }
        // task 382: 比較モードでは融合版の出力も戻す (同じ待ちで 1 本増やすだけ)。
        if fu_chk {
            let kp_cp = dec_kprof::op_begin(
                || "D2H@a".to_string(),
                (0, 0, 0),
                (0, 0, 0),
                0,
                e * 4,
            );
            unsafe {
                check(
                    (lib.memcpy_dtoh_async)(
                        hchk.as_ref().expect("確保済み").ptr,
                        dchk.as_ref().expect("確保済み").ptr(),
                        e * 4,
                        std::ptr::null_mut(),
                    ),
                    "cuMemcpyDtoHAsync (mix upq8 check)",
                )?;
            }
            dec_kprof::op_end(kp_cp);
            crate::prof::mark_d2h_src(crate::prof::D2H_SRC_LINEAR);
        }
        if fast_down || fast_up {
            MIX_QK_HITS.fetch_add(1, AO::Relaxed);
        }
        // task 282: GPU の鎖が走っている間にホストの inject GEMV を済ませる
        // (待ちに隠れる。値は従来と 1 ビットも変わらない)。
        if let Some(f) = pre_wait.as_deref_mut() {
            f()?;
        }
        // task 282: 待ちは null ストリームのイベント 1 本。文脈同期は背景の
        // 上げ直し (非ブロッキングストリーム) まで待ってしまうので使わない
        // (task 278 の層一括経路と同じ理由)。イベントが張れない環境だけ
        // 従来どおり文脈同期へ落とす。
        // task 464: defer の回はイベントを打つだけで待たない (後続の GDN
        // submission が同じ null ストリームの最後で待つ)。イベントが張れない
        // 環境だけは降ろし直してから文脈同期する (defer しない)。
        let mut deferred = false;
        if seg_ev::record_raw(8) {
            if defer {
                deferred = true;
                MIX_PENDING_EV.with(|c| c.set(true));
            } else {
                seg_ev::wait(8)?;
            }
        } else {
            if defer && !zc_out && !d2h_done {
                unsafe {
                    check(
                        (lib.memcpy_dtoh_async)(ho.ptr, dm.ptr(), e * 4, std::ptr::null_mut()),
                        "cuMemcpyDtoHAsync (mix q chain defer 落ち)",
                    )?;
                }
            }
            ctx.sync()?;
        }
        // task 392: この鎖が作った `mixed` (= 次の段の入力 cur) のデバイス
        // コピーを控える。ssm 層の GDN submission が同じバイト列を H2D で
        // 上げ直さずに読めるようにするため (`DENNO_GDN_X_MIX`、既定 on)。
        if crate::cuda::gdn_x_mix_on() {
            crate::cuda::gdn_x_in_mix_arm(mdev, e);
        }
        // task 364: 鎖に積んだルータ GEMV の重み参照を手放す (同期の後)。
        drop(router_w);
        if deferred {
            // task 464: ホスト値は作らない (呼び出し側は空の Vec を受け取る)。
            MIX_DEFERRED.with(|c| c.set(true));
            MIX_NOWAIT_HITS.fetch_add(1, AO::Relaxed);
        } else {
            MIX_DEFERRED.with(|c| c.set(false));
            dec_kprof::step("mix_pack", || {
                if zc_out {
                    // task 382: hc_mixed がホストの pinned へ直接書いている。
                    let m = omap.as_ref().expect("確保済み");
                    out.copy_from_slice(&m.pin.as_slice::<f32>()[..e]);
                } else {
                    out.copy_from_slice(&ho.as_slice::<f32>()[..e]);
                }
            });
        }
        // task 382: 比較モード (`DENNO_MIX_UPQ8=check`) — 従来の 3 本の出力
        // (`ho`) と融合版の出力 (`hchk`) を突き合わせる。値を変えないので
        // テキストと md5 は従来経路のまま。
        if fu_chk {
            let a = ho.as_slice::<f32>();
            let b = hchk.as_ref().expect("確保済み").as_slice::<f32>();
            let mut worst = 0.0f64;
            let mut worst_ref = 0.0f64;
            let mut bad = 0u64;
            for i in 0..e {
                let (x, y) = (a[i] as f64, b[i] as f64);
                let d = (x - y).abs();
                if d > worst {
                    worst = d;
                    worst_ref = x.abs();
                }
                if d > 1e-4 + 1e-5 * x.abs() {
                    bad += 1;
                }
            }
            mix_upq8_note_check(worst, worst_ref, bad, e as u64);
        }
        // task 382: 発火の証拠 (PROFILE の `[mix-zc]` 行のもと)。
        if zc_x {
            MIX_ZC_X_HITS.fetch_add(1, AO::Relaxed);
        }
        if zc_out {
            MIX_ZC_OUT_HITS.fetch_add(1, AO::Relaxed);
        }
        Ok(())
    });
    match r {
        Ok(()) => {
            set_last_linear_ran_on_gpu(true);
            true
        }
        Err(e) => {
            mix_gpu_broken_once(&format!("mix q chain failed: {e:#}"));
            false
        }
    }
}


/// task 110: hc mix BF16 展開重みの常駐を解放する (`GgufLinear::drop` から)。
/// 載っていない uid は触らない。MIX_GPU_W が未初期化 (一度も GPU に載せていない)
/// なら何もしない。
pub fn mix_gpu_unregister(uid: u64) {
    let Some(reg) = MIX_GPU_W.get() else {
        return;
    };
    if let Ok(mut map) = reg.lock() {
        map.remove(&uid);
    }
}


// ===========================================================================
// task 184: mix の norm / inject をデバイスへ (`DENNO_MIX_DEC_DEV`)
// ===========================================================================
//
// 背景 (lab t179a の平均表 `DENNO_GDN_DEC_KPROF`): mix 1 回のホスト仕事は
// mix_norm 5.1 us + mix_inj 3.8 us = 8.9 us で、decode は 1 トークンに
// 2*n_layer+1 = 97 回 mix を呼ぶ (0.86 ms/token)。どちらも「1 スレッドの逐次和」
// なので、デバイスのツリー縮約に置き換えて mix の submission に畳む。
//
// ホストが回していた順序 (norm → inject と down/up の直列) を、1 回の同期で
// 閉じる 1 本の submission にする:
//   H2D x → dno_mix_norm → (dno_mix_inj | down+silu → up → hc_mixed)
//   → D2H mixed[e] + inj[hc] → 同期 1 回
//
// 値: 縮約がツリーになるので CPU の逐次和とは丸め順が変わる (md5 は DIFF しうる)。
// 要素ごとの演算列は CPU と同じ。fail closed (PTX 無し / 形違い / 確保失敗 →
// `false` を返し、呼び出し側は従来のホスト経路をそのまま走らせる)。

/// `DENNO_MIX_DEC_DEV` の解釈 (純関数)。既定は [`MIX_DEC_DEV_DEFAULT`]。
pub(crate) fn mix_dec_dev_from_value(v: Option<&str>, default_on: bool) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        Some(s) if matches!(s.as_str(), "1" | "true" | "on" | "yes") => true,
        Some(s) if matches!(s.as_str(), "0" | "false" | "off" | "no") => false,
        _ => default_on,
    }
}


/// task 184: mix の norm / inject をデバイスで回すか (`DENNO_MIX_DEC_DEV`)。
///
/// lab t184a (depth8k) で採否 yes: 128 トークンの生 decode 秒が
/// 3.39/2.93 -> 3.33/2.88 (短)、3.05 -> 2.99 (3,965)、3.07 -> 3.03 (14,157)
/// = -1.3..-2.0%。単独でも効き、3 ゲート同時でも加算的だった (mix 単独 -1.8%)。
/// 戻すのは `DENNO_MIX_DEC_DEV=0`。
pub(crate) const MIX_DEC_DEV_DEFAULT: bool = true;


pub(crate) fn mix_dec_dev_enabled() -> bool {
    #[cfg(test)]
    if let Some(v) = MIX_DEC_DEV_FORCE.with(|c| c.get()) {
        return v;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        mix_dec_dev_from_value(
            std::env::var("DENNO_MIX_DEC_DEV").ok().as_deref(),
            MIX_DEC_DEV_DEFAULT,
        )
    })
}


/// テスト用: ゲートの強制 (None = env の解釈)。
#[cfg(test)]
thread_local! {
    pub(crate) static MIX_DEC_DEV_FORCE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}


#[cfg(test)]
pub(crate) fn mix_dec_dev_set_for_test(v: Option<bool>) {
    MIX_DEC_DEV_FORCE.with(|c| c.set(v));
}


/// task 184: chain3 の入口に入った回数 (テスト専用)。ゲートを足したが分岐が
/// 通っていない、を GPU 無しで検出するために `cuda_enabled()` の判定より前に数える。
#[cfg(test)]
static MIX_DEV_ATTEMPTS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);


/// task 184: [`MIX_DEV_ATTEMPTS`] を読む (テスト専用)。
#[cfg(test)]
pub(crate) fn mix_dev_attempts() -> usize {
    MIX_DEV_ATTEMPTS.load(std::sync::atomic::Ordering::Relaxed)
}


/// task 184: 入口のカウンタを 0 に戻す (テスト専用)。
#[cfg(test)]
pub(crate) fn mix_dev_attempts_reset() {
    MIX_DEV_ATTEMPTS.store(0, std::sync::atomic::Ordering::Relaxed);
}


pub(crate) struct MixDevKernels {
    #[allow(dead_code)]
    module: CUmodule,
    norm: CUfunction,
    inj: CUfunction,
    /// task 266: lo_apply (silu(v * inv_hc))。**古い PTX には無い**ので任意。
    /// 無いときは Q8 の fused chain が CPU 経路へ落ちる (従来の経路は不変)。
    lo_silu: Option<CUfunction>,
}


unsafe impl Send for MixDevKernels {}

unsafe impl Sync for MixDevKernels {}


static MIX_DEV_KERNELS: OnceLock<Option<MixDevKernels>> = OnceLock::new();


fn mix_dev_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_MIX_DEV_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_mix_dev.ptx"),
    }
}


/// mix_dev のカーネルを取得する (初回だけ PTX を読む。失敗は `None` を焼き付ける)。
fn mix_dev_kernels() -> Option<&'static MixDevKernels> {
    MIX_DEV_KERNELS
        .get_or_init(|| {
            let path = mix_dev_ptx_path();
            let mut image = match std::fs::read(&path) {
                Ok(b) => b,
                Err(_) => return None,
            };
            let text = String::from_utf8_lossy(&image).to_string();
            // ABI: dno_mix_norm は (x, gamma, out, e, eps) の 5 本、
            // dno_mix_inj は (w, x, out, k, n) の 5 本。
            if ptx_entry_params(&text, "dno_mix_norm") != Some(5)
                || ptx_entry_params(&text, "dno_mix_inj") != Some(5)
            {
                return None;
            }
            image.push(0);
            let lib = lib().ok()?;
            unsafe {
                let mut module: CUmodule = std::ptr::null_mut();
                if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void)
                    != CUDA_SUCCESS
                {
                    return None;
                }
                let get = |name: &[u8]| -> Option<CUfunction> {
                    let mut f: CUfunction = std::ptr::null_mut();
                    let r = (lib.module_get_function)(
                        &mut f,
                        module,
                        name.as_ptr() as *const c_char,
                    );
                    if r == CUDA_SUCCESS {
                        Some(f)
                    } else {
                        None
                    }
                };
                let norm = get(b"dno_mix_norm\0")?;
                let inj = get(b"dno_mix_inj\0")?;
                // task 266: 任意 (古い PTX には無い)。
                let lo_silu = get(b"dno_mix_lo_silu\0");
                Some(MixDevKernels {
                    module,
                    norm,
                    inj,
                    lo_silu,
                })
            }
        })
        .as_ref()
}


// ---------------------------------------------------------------------------
// task 192: mix の norm と inject を 1 本に畳んだ `kernels_mix_dev2.cu`。
//
// lab t184a の `[dec-kprof-avg]` mix 表で、`dno_mix_norm` 6.2 us と
// `dno_mix_inj` 5.6 us は hc=4 の小さな仕事で**起動の固定費が支配的**だった。
// 1 本にすると起動が 1 回減る (値は 2 本版とビット一致)。
// ---------------------------------------------------------------------------

/// mix の fold 版のカーネル名。
pub(crate) const MIX_FOLD_KERNEL_NAME: &str = "dno_mix_norm_inj";


/// fold 版の引数の数 (x, gamma, out, w, inj, e, hc, k, eps)。
pub(crate) const MIX_FOLD_NARGS: usize = 9;


/// `DENNO_MIX_DEC_FOLD` の解釈 (純関数)。
///
/// **既定 on** (task 221 の lab t221c で `all` の一部として採用)。明示的な偽値だけ off。
/// 値は 2 本版とビット一致 (norm と inject を 1 本に畳むだけ) なので、lab でも
/// 単独 cond は md5 SAME だった。t221b の単独は短い点 +1.1% (1 サンプル・
/// 起動が 1 本減る側なのでノイズと見る)、t221c の `all` は -3.5%。
/// 切り戻しは `DENNO_MIX_DEC_FOLD=0`。
pub(crate) fn parse_mix_dec_fold(raw: Option<&str>) -> bool {
    !matches!(
        raw.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("0") | Some("off") | Some("false") | Some("no")
    )
}


/// `DENNO_MIX_DEC_UPW` の解釈 (純関数)。
///
/// mix の up 投影を 1 warp = 1 行の版で回すか。**既定 off** (fold と同じ lab)。
/// fold (`DENNO_MIX_DEC_FOLD`) とは独立に切れる (別の cond で測るため)。
/// task 213 で fold と束ねて測って採否 no になり、`bbb1a603f` で既定 off に戻した
/// 規則と同じにする (int-auto 389419b2e では PTX が無く、既定 on のままでも
/// 静かに従来経路へ落ちていた = 実挙動は off。PTX を戻すときに既定が動かない
/// ようにするため、明示 on だけを on とする)。
pub(crate) fn parse_mix_dec_upw(raw: Option<&str>) -> bool {
    matches!(
        raw.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("1") | Some("on") | Some("true") | Some("yes")
    )
}


/// up を 1 warp = 1 行の版で回すか (`DENNO_MIX_DEC_UPW`)。
pub(crate) fn mix_dec_upw_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse_mix_dec_upw(std::env::var("DENNO_MIX_DEC_UPW").ok().as_deref()))
}


/// task 216: `DENNO_DEC_MIX_UPW` の解釈 (純関数)。
///
/// decode の mix の up 投影を 1 warp = 1 行 + uint4 読みの版
/// (`kernels_mix_up.cu`) で回すか。**既定 on** (task 221 の lab t221c で採用)。
/// task 192 の `DENNO_MIX_DEC_UPW` とは別名にして、fold と切り離して
/// 単独で A/B できるようにした (task 213 は fold と束ねて測っていた)。
/// t221b の単独: 短い点 -2.4% (3 点とも -1.9〜-2.5%)。切り戻しは
/// `DENNO_DEC_MIX_UPW=0` だけでよい (legacy の `DENNO_MIX_DEC_UPW` は既定 off で、
/// `=1` を明示しているときだけ OR で効く)。
pub(crate) fn parse_dec_mix_upw(raw: Option<&str>) -> bool {
    !matches!(
        raw.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("0") | Some("off") | Some("false") | Some("no")
    )
}


/// task 216: up を uint4 版で回すか (`DENNO_DEC_MIX_UPW`)。
/// task 221 で既定 on。`gate_audit` の台帳から参照するので pub(crate)。
pub(crate) fn dec_mix_upw_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse_dec_mix_upw(std::env::var("DENNO_DEC_MIX_UPW").ok().as_deref()))
}


/// task 216: `DENNO_DEC_MIX_DOWNW` の解釈 (純関数)。
///
/// decode の mix の down 投影 (BF16 + silu) を uint4 読みの版
/// (`kernels_mix_down.cu`) で回すか。**既定 off**。
pub(crate) fn parse_dec_mix_downw(raw: Option<&str>) -> bool {
    matches!(
        raw.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("1") | Some("on") | Some("true")
    )
}


/// task 216: down を uint4 版で回すか (`DENNO_DEC_MIX_DOWNW`)。
fn dec_mix_downw_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        parse_dec_mix_downw(std::env::var("DENNO_DEC_MIX_DOWNW").ok().as_deref())
    })
}


/// task 216: `DENNO_DEC_MIX_DTHR` の解釈 (純関数)。
///
/// 従来カーネルで down を撃つときの 1 ブロックのスレッド数 (既定 256 =
/// これまでの挙動)。32 の倍数で 32..=1024 のときだけ採用し、それ以外は 256。
/// `dno_matvec_bf16_r_silu` は縮約をブロック内の warp 数で回すので
/// 任意の 32 の倍数で正しく動く (固定木の段数だけが変わる)。
pub(crate) fn parse_dec_mix_dthr(raw: Option<&str>) -> usize {
    match raw.map(|s| s.trim().parse::<usize>()) {
        Some(Ok(v)) if (32..=1024).contains(&v) && v % 32 == 0 => v,
        _ => 256,
    }
}


/// task 216: down のスレッド数 (`DENNO_DEC_MIX_DTHR`、既定 256)。
fn dec_mix_dthr() -> usize {
    static THR: OnceLock<usize> = OnceLock::new();
    *THR.get_or_init(|| parse_dec_mix_dthr(std::env::var("DENNO_DEC_MIX_DTHR").ok().as_deref()))
}


/// task 216: `DENNO_DEC_MIX_UPMIX` の解釈 (純関数)。
///
/// decode の mix の up と hc_mixed を 1 本に畳む版 (`kernels_mix_upmix.cu`) で
/// 回すか。**既定 off** (明示的な真値だけ on)。on のときは up も uint4 版になる
/// (融合カーネルが up を含むため)。
pub(crate) fn parse_dec_mix_upmix(raw: Option<&str>) -> bool {
    parse_dec_mix_upmix_ver(raw) != 0
}


/// task 216: `DENNO_DEC_MIX_UPMIX` の版。0 = off、1 = uint4 版 (丸めが変わる)、
/// 2 = **ビット一致**版 (`dno_mix_upmix2_bf16`。従来の up + hc_mixed と同値)。
pub(crate) fn parse_dec_mix_upmix_ver(raw: Option<&str>) -> u8 {
    match raw.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        Some("2") => 2,
        Some("1") | Some("on") | Some("true") => 1,
        _ => 0,
    }
}


/// task 216: `DENNO_DEC_MIX_INJHOST` の解釈 (純関数)。
///
/// mix の inject (hc 要素) を **ホストで先に**作るか。GPU の鎖
/// (norm → down → up → mixed) が走っている間に CPU で済ませるので、
/// GPU の inj カーネル 1 本とその分の待ちが消える。既定 off。
pub(crate) fn parse_dec_mix_injhost(raw: Option<&str>) -> bool {
    matches!(
        raw.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("1") | Some("on") | Some("true")
    )
}


/// task 216: inj をホストで先に作るか (`DENNO_DEC_MIX_INJHOST`)。
pub(crate) fn dec_mix_injhost_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse_dec_mix_injhost(std::env::var("DENNO_DEC_MIX_INJHOST").ok().as_deref()))
}


/// task 216: up + hc_mixed の融合版の版 (`DENNO_DEC_MIX_UPMIX`、既定 0)。
fn dec_mix_upmix_ver() -> u8 {
    static VER: OnceLock<u8> = OnceLock::new();
    *VER.get_or_init(|| {
        parse_dec_mix_upmix_ver(std::env::var("DENNO_DEC_MIX_UPMIX").ok().as_deref())
    })
}


/// task 216: up + hc_mixed の融合版で回すか (`DENNO_DEC_MIX_UPMIX`)。
fn dec_mix_upmix_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse_dec_mix_upmix(std::env::var("DENNO_DEC_MIX_UPMIX").ok().as_deref()))
}


/// task 342: decode の mix 本線 (chain3) の `hc_mixed` のブロック幅
/// (`DENNO_DEC_MIX_HCBLK`、既定 32)。切り戻しは `DENNO_DEC_MIX_HCBLK=256`。
///
/// なぜ効くか: `dno_hc_mixed_f32` は **1 出力 1 スレッド**で、出力ごとに
/// `dno_expf` を hc 回 (= 4 回) 呼ぶ。`dno_expf` は glibc expf のビット一致
/// 実装で **double** (fp64) の FMA/乗算を使う。RTX 5090 の fp64 は fp32 の
/// 1/64 (SM あたり 2 レーン) なので、fp64 の warp 命令 1 本は約 16 サイクル。
/// 既定だった grid = ceil(e/256) = 10 ブロック (e = 2560) では **10 SM にしか
/// 載らず**、1 SM あたり 8 warp × 36 fp64 命令 = 288 命令 = 約 4,600 サイクル
/// (3 us 超) が fp64 の床になっていた。
/// ブロック幅を 32 にすると 80 ブロック = 80 SM に散り、床は 1/8 になる。
///
/// **値はビット一致**: 1 出力を計算するスレッドの式も s 昇順の加算順も
/// カーネル側のままで、どのブロックがそのスレッドを持つかだけが変わる。
/// 実測 (lab t342a/b/c、各 2 ラウンド): `dno_hc_mixed_f32` 5.8 -> 3.9 us/窓、
/// mix 段の gpu 2.71-2.75 -> 2.61-2.64 ms/token (-3〜-4%)、decode 中央値 -0.5%。
pub const DEC_MIX_HCBLK_ENV: &str = "DENNO_DEC_MIX_HCBLK";


/// [`DEC_MIX_HCBLK_ENV`] の解釈 (純関数)。32..=1024 の 32 の倍数だけ受け付け、
/// それ以外 (未設定・数でない・範囲外) は 32 (task 342 の採否後の既定)。
pub fn parse_dec_mix_hcblk(raw: Option<&str>) -> usize {
    match raw.map(str::trim).and_then(|s| s.parse::<usize>().ok()) {
        Some(v) if v >= 32 && v <= 1024 && v % 32 == 0 => v,
        _ => 32,
    }
}


/// task 342: `hc_mixed` のブロック幅 (`DENNO_DEC_MIX_HCBLK`)。
fn dec_mix_hcblk() -> usize {
    static ON: OnceLock<usize> = OnceLock::new();
    *ON.get_or_init(|| parse_dec_mix_hcblk(std::env::var(DEC_MIX_HCBLK_ENV).ok().as_deref()))
}


/// hc 方向を畳んだ norm+inject を使うか (`DENNO_MIX_DEC_FOLD`)。
pub(crate) fn mix_dec_fold_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse_mix_dec_fold(std::env::var("DENNO_MIX_DEC_FOLD").ok().as_deref()))
}


/// task 275: `DENNO_DEC_MIX_NORM_PAR` の解釈 (純関数)。`"1"` / `"true"` / `"on"` だけが真。
fn parse_mix_norm_par(raw: Option<&str>) -> bool {
    // tasks 269+275 adopted by the operator (2026-11-13): default on; only an
    // explicit false value (0 / off / false / no) turns it off.
    !matches!(
        raw.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("0" | "off" | "false" | "no")
    )
}


/// task 275: norm を hc ブロックへ割り、inject を down の epilogue へ移すか
/// (`DENNO_DEC_MIX_NORM_PAR`)。**既定 off** (採否は lab で決める)。
///
/// いまは `DENNO_MIX_DEC_FOLD` が既定 on で、`dno_mix_norm_inj` が **grid = 1**
/// (1 SM) で正規化と inject を回している。inject は正規化後の全ストリームを
/// 読むので 1 ブロックに閉じていたが、正規化した x は以降**読むだけ**なので
/// inject は down の epilogue へ移せる。起動回数は増えない。
fn mix_norm_par_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        parse_mix_norm_par(std::env::var("DENNO_DEC_MIX_NORM_PAR").ok().as_deref())
    })
}


/// task 299: mix の grouped RMS 正規化をホストで済ませるか
/// (`DENNO_MIX_NORM_HOST`)。**task 307 で既定 on** (lab t299b で mix 行
/// -0.22 ms/token、t307b で decode -1.3%・md5 SAME 2 回)。明示的な偽値だけ off。
///
/// 既定経路 (`DENNO_DEC_MIX_NORM_PAR` on) は正規化を `dno_mix_norm`
/// (grid = hc = 4、256 スレッド) で回す。台帳の実測は 1 呼び出し 6.5 us で、
/// 動かすバイト数は 40 KB しかない = **起動と占有**で決まっている。
/// x は H2D する直前にホストにあり、gamma も同じ長さでホストにあるので、
/// H2D する中身を「正規化済みの x」に差し替えれば同じ計算をホストで済ませられ、
/// 起動 1 本と GPU の 6.5 us が消える (ホスト側は 1 パス増えるだけ)。
///
/// 丸め順が変わる (二乗和を 8 本の累積器で足す) ので値はビット一致しない。
fn mix_norm_host_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse_mix_norm_host(std::env::var("DENNO_MIX_NORM_HOST").ok().as_deref()))
}


/// [`mix_norm_host_on`] の解釈 (純関数。**明示的な偽値だけ off** = 既定 on)。
pub(crate) fn parse_mix_norm_host(raw: Option<&str>) -> bool {
    !matches!(
        raw.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("0") | Some("off") | Some("false") | Some("no")
    )
}


/// task 299: ホスト正規化の経路に入れる形か。
///
/// 正規化を消すと inject の入力 (正規化済み x) も消えるので、inject を down の
/// epilogue で回す版 (`dno_mix_down_bf16_inj`) が選べるときだけ入る。
fn mix_norm_host_ready(hc: usize, down_thr: u32) -> bool {
    mix_norm_host_on()
        && mix_norm_par_on()
        && hc >= 1
        && hc <= 4
        && down_thr >= 256
        && mix_dev3_kernels().is_some()
}


/// task 299: `x[hc][e]` の grouped RMS 正規化 (`out = (x * inv) * gamma`) をホストで。
///
/// 要素ごとの式は `dno_mix_norm` と同じ (`__fmul_rn` / `__fadd_rn` 相当の
/// 丸め)。二乗和だけは 8 本の累積器で足す (順序は固定 = 決定性は保つ)。
/// `out` は `x` と別でも同一でもよい (縮約が終わるまで書かない)。
pub(crate) fn mix_norm_host_into(x: &[f32], gamma: &[f32], e: usize, hc: usize, eps: f32, out: &mut [f32]) {
    debug_assert_eq!(x.len(), e.saturating_mul(hc), "x は [hc][e]");
    debug_assert_eq!(gamma.len(), x.len(), "gamma は [hc][e]");
    debug_assert!(out.len() >= x.len(), "out が [hc][e] より短い");
    for s in 0..hc {
        let xs = &x[s * e..(s + 1) * e];
        let mut acc = [0f32; 8];
        let mut i = 0usize;
        while i + 8 <= e {
            for (j, a) in acc.iter_mut().enumerate() {
                let v = xs[i + j];
                *a += v * v;
            }
            i += 8;
        }
        let mut tail = 0f32;
        while i < e {
            tail += xs[i] * xs[i];
            i += 1;
        }
        let ss = ((acc[0] + acc[1]) + (acc[2] + acc[3]))
            + ((acc[4] + acc[5]) + (acc[6] + acc[7]))
            + tail;
        let inv = 1.0f32 / (ss / e as f32 + eps).sqrt();
        let g = &gamma[s * e..(s + 1) * e];
        let o = &mut out[s * e..(s + 1) * e];
        for ((d, &v), &w) in o.iter_mut().zip(xs.iter()).zip(g.iter()) {
            *d = (v * inv) * w;
        }
    }
}


/// task 204: mix の D2H 待ちを `cuCtxSynchronize` から null ストリームの
/// イベント 1 本へ (`DENNO_DEC_ONEVNT`・既定 off)。
fn mix_onevnt_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse_dec_onevnt(std::env::var("DENNO_DEC_ONEVNT").ok().as_deref()))
}


pub(crate) struct MixFoldKernels {
    #[allow(dead_code)]
    module: CUmodule,
    f: CUfunction,
}


unsafe impl Send for MixFoldKernels {}

unsafe impl Sync for MixFoldKernels {}


static MIX_FOLD_KERNELS: OnceLock<Option<MixFoldKernels>> = OnceLock::new();


fn mix_fold_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_MIX_FOLD_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_mix_dev2.ptx"),
    }
}


/// fold 版のカーネルを取得する (初回だけ PTX を読む。失敗は `None` を焼き付ける)。
fn mix_fold_kernels() -> Option<&'static MixFoldKernels> {
    MIX_FOLD_KERNELS
        .get_or_init(|| {
            let path = mix_fold_ptx_path();
            let mut image = match std::fs::read(&path) {
                Ok(b) => b,
                Err(_) => return None,
            };
            let text = String::from_utf8_lossy(&image).to_string();
            if ptx_entry_params(&text, MIX_FOLD_KERNEL_NAME) != Some(MIX_FOLD_NARGS) {
                eprintln!(
                    "[mix-fold] PTX の引数が {MIX_FOLD_NARGS} 本でない → 2 本版へ。\
                     `bash build_ptx.sh mix_dev2` で再生成する"
                );
                return None;
            }
            image.push(0);
            let lib = lib().ok()?;
            unsafe {
                let mut module: CUmodule = std::ptr::null_mut();
                if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void)
                    != CUDA_SUCCESS
                {
                    return None;
                }
                let mut cname = MIX_FOLD_KERNEL_NAME.as_bytes().to_vec();
                cname.push(0);
                let mut f: CUfunction = std::ptr::null_mut();
                if (lib.module_get_function)(&mut f, module, cname.as_ptr() as *const c_char)
                    != CUDA_SUCCESS
                {
                    eprintln!("[mix-fold] PTX に {MIX_FOLD_KERNEL_NAME} が無い → 2 本版へ");
                    return None;
                }
                Some(MixFoldKernels { module, f })
            }
        })
        .as_ref()
}


/// fold 版が載っているか (テスト専用)。
#[cfg(test)]
pub(crate) fn mix_fold_loaded() -> bool {
    mix_fold_kernels().is_some()
}


// ---------------------------------------------------------------------------
// task 275: down の epilogue に inject の内積を足した版 (`kernels_mix_dev3.cu`)。
//
// `DENNO_DEC_MIX_NORM_PAR` のとき、norm+inject の 1 ブロック版の代わりに
//   `dno_mix_norm` (grid = hc、kernels_mix_dev.ptx の既存カーネル)
//   → `dno_mix_down_bf16_inj` (down と inject を 1 起動で)
// で回す。起動回数は今と同じ。値はビット一致 (v2 と同じ木・同じ刻み)。
// ---------------------------------------------------------------------------

/// down+inject 版のカーネル名。
const MIX_DEV3_KERNEL_NAME: &str = "dno_mix_down_bf16_inj";


/// down+inject 版の引数の数 (x, w, out, k, n, inv_hc, wi, inj, ki, hc)。
const MIX_DEV3_NARGS: usize = 10;


pub(crate) struct MixDev3Kernels {
    #[allow(dead_code)]
    module: CUmodule,
    f: CUfunction,
}


unsafe impl Send for MixDev3Kernels {}

unsafe impl Sync for MixDev3Kernels {}


static MIX_DEV3_KERNELS: OnceLock<Option<MixDev3Kernels>> = OnceLock::new();


fn mix_dev3_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_MIX_DEV3_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_mix_dev3.ptx"),
    }
}


/// down+inject 版のカーネルを取得する (初回だけ PTX を読む。失敗は `None` を焼き付ける)。
fn mix_dev3_kernels() -> Option<&'static MixDev3Kernels> {
    MIX_DEV3_KERNELS
        .get_or_init(|| {
            let path = mix_dev3_ptx_path();
            let mut image = match std::fs::read(&path) {
                Ok(b) => b,
                Err(_) => return None,
            };
            let text = String::from_utf8_lossy(&image).to_string();
            if ptx_entry_params(&text, MIX_DEV3_KERNEL_NAME) != Some(MIX_DEV3_NARGS) {
                eprintln!(
                    "[mix-dev3] PTX の引数が {MIX_DEV3_NARGS} 本でない → fold 版へ。\
                     `bash build_ptx.sh mix_dev3` で再生成する"
                );
                return None;
            }
            image.push(0);
            let lib = lib().ok()?;
            unsafe {
                let mut module: CUmodule = std::ptr::null_mut();
                if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void)
                    != CUDA_SUCCESS
                {
                    return None;
                }
                let mut cname = MIX_DEV3_KERNEL_NAME.as_bytes().to_vec();
                cname.push(0);
                let mut f: CUfunction = std::ptr::null_mut();
                if (lib.module_get_function)(&mut f, module, cname.as_ptr() as *const c_char)
                    != CUDA_SUCCESS
                {
                    eprintln!("[mix-dev3] PTX に {MIX_DEV3_KERNEL_NAME} が無い → fold 版へ");
                    return None;
                }
                Some(MixDev3Kernels { module, f })
            }
        })
        .as_ref()
}


/// down+inject 版が載っているか (テスト専用)。
#[cfg(test)]
pub(crate) fn mix_dev3_loaded() -> bool {
    mix_dev3_kernels().is_some()
}


// ---------------------------------------------------------------------------
// task 327: inject を down の**追加ブロック**で回す版 (`kernels_mix_inj3.cu`)。
//
// task 275 の `dno_mix_down_bf16_inj` は inject を `blockIdx.x < hc` の 4 ブロックの
// **epilogue** で回すので、その 4 ブロックだけ「down の行 (7.6 us) + inject (4.7 us)」
// を直列に払う (実測 12.1-12.3 us。P323 の t323d)。inject は down の出力を読まない
// ので、grid = n + hc にして inject を後ろのブロックへ移せば down と同じ時刻に始まり、
// 7.6 us の裏へ隠れる。**値はビット一致** (同じ木・同じ刻み。行の割り当てだけが違う)。
//
// ゲート `DENNO_DEC_MIX_INJSEP` (既定 off)。
// ---------------------------------------------------------------------------

/// inject を追加ブロックで回す版のカーネル名。
pub(crate) const MIX_INJ3_KERNEL_NAME: &str = "dno_mix_down_bf16_inj3";


/// その引数の数 (x, w, out, k, n, inv_hc, wi, inj, ki, hc)。
pub(crate) const MIX_INJ3_NARGS: usize = 10;


pub(crate) struct MixInj3Kernels {
    #[allow(dead_code)]
    module: CUmodule,
    f: CUfunction,
}


unsafe impl Send for MixInj3Kernels {}

unsafe impl Sync for MixInj3Kernels {}


static MIX_INJ3_KERNELS: OnceLock<Option<MixInj3Kernels>> = OnceLock::new();


fn mix_inj3_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_MIX_INJ3_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_mix_inj3.ptx"),
    }
}


/// [`MIX_INJ3_KERNEL_NAME`] の解釈 (純関数)。**明示的な偽値だけ off** = 既定 on。
///
/// lab t327a で採否 yes (14k 後の decode -2.6%、短い decode -0.9%、prefill ±0、
/// md5 は全 COND SAME)。`0` / `off` / `false` / `no` で epilogue 版へ戻る。
pub(crate) fn parse_dec_mix_injsep(raw: Option<&str>) -> bool {
    !matches!(
        raw.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("0") | Some("off") | Some("false") | Some("no")
    )
}


/// task 327: inject を追加ブロックで回すか (`DENNO_DEC_MIX_INJSEP`・既定 on)。
fn mix_injsep_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse_dec_mix_injsep(std::env::var("DENNO_DEC_MIX_INJSEP").ok().as_deref()))
}


/// inject を追加ブロックで回す版のカーネルを取得する
/// (初回だけ PTX を読む。失敗は `None` を焼き付ける → epilogue 版へ)。
fn mix_inj3_kernels() -> Option<&'static MixInj3Kernels> {
    MIX_INJ3_KERNELS
        .get_or_init(|| {
            let path = mix_inj3_ptx_path();
            let mut image = match std::fs::read(&path) {
                Ok(b) => b,
                Err(_) => return None,
            };
            let text = String::from_utf8_lossy(&image).to_string();
            if ptx_entry_params(&text, MIX_INJ3_KERNEL_NAME) != Some(MIX_INJ3_NARGS) {
                eprintln!(
                    "[mix-inj3] PTX の引数が {MIX_INJ3_NARGS} 本でない → epilogue 版へ。\
                     `bash build_ptx.sh mix_inj3` で再生成する"
                );
                return None;
            }
            image.push(0);
            let lib = lib().ok()?;
            unsafe {
                let mut module: CUmodule = std::ptr::null_mut();
                if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void)
                    != CUDA_SUCCESS
                {
                    return None;
                }
                let mut cname = MIX_INJ3_KERNEL_NAME.as_bytes().to_vec();
                cname.push(0);
                let mut f: CUfunction = std::ptr::null_mut();
                if (lib.module_get_function)(&mut f, module, cname.as_ptr() as *const c_char)
                    != CUDA_SUCCESS
                {
                    eprintln!("[mix-inj3] PTX に {MIX_INJ3_KERNEL_NAME} が無い → epilogue 版へ");
                    return None;
                }
                Some(MixInj3Kernels { module, f })
            }
        })
        .as_ref()
}


// ---------------------------------------------------------------------------
// task 192: mix の up 投影 (BF16) を 1 warp = 1 行で回す `kernels_mix_up.cu`。
//
// lab t192a の mix 表: `dno_matvec_bf16_r` (up、10,240 行 x k=320) が 14.0 us
// (実効 468 GB/s)。1 ブロック 256 スレッドで 1 行 (320 要素) なので 1 スレッド
// 1.25 要素で、起動と縮約の固定費が支配的だった。1 warp = 1 行にする。
// ---------------------------------------------------------------------------

/// up 版のカーネル名。
pub(crate) const MIX_UP_KERNEL_NAME: &str = "dno_mix_up_bf16";


/// up 版の引数の数 (w, x, out, k, n)。
pub(crate) const MIX_UP_NARGS: usize = 5;


/// up 版の 1 ブロックの行数 (`kernels_mix_up.cu` の DNO_MIXUP_WARPS)。
const MIX_UP_ROWS: usize = 8;


pub(crate) struct MixUpKernels {
    #[allow(dead_code)]
    module: CUmodule,
    f: CUfunction,
}


unsafe impl Send for MixUpKernels {}

unsafe impl Sync for MixUpKernels {}


static MIX_UP_KERNELS: OnceLock<Option<MixUpKernels>> = OnceLock::new();


fn mix_up_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_MIX_UP_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_mix_up.ptx"),
    }
}


/// up 版のカーネルを取得する (初回だけ PTX を読む。失敗は `None` を焼き付ける)。
fn mix_up_kernels() -> Option<&'static MixUpKernels> {
    MIX_UP_KERNELS
        .get_or_init(|| {
            let path = mix_up_ptx_path();
            let mut image = match std::fs::read(&path) {
                Ok(b) => b,
                Err(_) => return None,
            };
            let text = String::from_utf8_lossy(&image).to_string();
            if ptx_entry_params(&text, MIX_UP_KERNEL_NAME) != Some(MIX_UP_NARGS) {
                eprintln!(
                    "[mix-up] PTX の引数が {MIX_UP_NARGS} 本でない → 従来カーネルへ。\
                     `bash build_ptx.sh mix_up` で再生成する"
                );
                return None;
            }
            image.push(0);
            let lib = lib().ok()?;
            unsafe {
                let mut module: CUmodule = std::ptr::null_mut();
                if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void)
                    != CUDA_SUCCESS
                {
                    return None;
                }
                let mut cname = MIX_UP_KERNEL_NAME.as_bytes().to_vec();
                cname.push(0);
                let mut f: CUfunction = std::ptr::null_mut();
                if (lib.module_get_function)(&mut f, module, cname.as_ptr() as *const c_char)
                    != CUDA_SUCCESS
                {
                    eprintln!("[mix-up] PTX に {MIX_UP_KERNEL_NAME} が無い → 従来カーネルへ");
                    return None;
                }
                Some(MixUpKernels { module, f })
            }
        })
        .as_ref()
}


/// up 版が載っているか (テスト専用)。
#[cfg(test)]
pub(crate) fn mix_up_loaded() -> bool {
    mix_up_kernels().is_some()
}


// ---------------------------------------------------------------------------
// task 216: mix の down 投影 (BF16 + silu) を uint4 読みで回す
// `kernels_mix_down.cu`。
//
// lab t192a の mix 表 (実測): 1 ブロック 256 スレッドで 1 行を読む従来形の up は
// 14.0 us = 実効 468 GB/s だった。1 レーンが 2 B (bf16 1 要素) ずつ読むので
// 1 warp 1 命令で 64 B しか運べない。up は uint4 (1 命令 512 B) に変えて
// 速くなった (task 192)。down は k = 10,240 = 20 KB/行 と行が長いのに
// 同じ 2 B 読みのままなので、同じ手当てをする。
//
// task 216 で分かったこと: mix の同期 1 回 (32 us) の実体は **この 4-5 本の
// カーネルの GPU 時間**で、ホストが値を読む往復は 4 us 程度しかない
// (REPORT_P216 の台帳)。したがって待ちを縮めるには、待っている仕事を短くする。
// ---------------------------------------------------------------------------

/// down 版のカーネル名。
const MIX_DOWN_KERNEL_NAME: &str = "dno_mix_down_bf16";


/// down 版の引数の数 (x, w, out, k, n, inv_hc)。
const MIX_DOWN_NARGS: usize = 6;


/// down 版の 1 ブロックのスレッド数 (`kernels_mix_down.cu` の既定)。
const MIX_DOWN_THREADS: usize = 512;


pub(crate) struct MixDownKernels {
    #[allow(dead_code)]
    module: CUmodule,
    f: CUfunction,
}


unsafe impl Send for MixDownKernels {}

unsafe impl Sync for MixDownKernels {}


static MIX_DOWN_KERNELS: OnceLock<Option<MixDownKernels>> = OnceLock::new();


fn mix_down_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_MIX_DOWN_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_mix_down.ptx"),
    }
}


/// down 版のカーネルを取得する (初回だけ PTX を読む。失敗は `None` を焼き付ける)。
fn mix_down_kernels() -> Option<&'static MixDownKernels> {
    MIX_DOWN_KERNELS
        .get_or_init(|| {
            let path = mix_down_ptx_path();
            let mut image = match std::fs::read(&path) {
                Ok(b) => b,
                Err(_) => return None,
            };
            let text = String::from_utf8_lossy(&image).to_string();
            if ptx_entry_params(&text, MIX_DOWN_KERNEL_NAME) != Some(MIX_DOWN_NARGS) {
                eprintln!(
                    "[mix-down] PTX の引数が {MIX_DOWN_NARGS} 本でない → 従来カーネルへ。\
                     `bash build_ptx.sh mix_down` で再生成する"
                );
                return None;
            }
            image.push(0);
            let lib = lib().ok()?;
            unsafe {
                let mut module: CUmodule = std::ptr::null_mut();
                if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void)
                    != CUDA_SUCCESS
                {
                    return None;
                }
                let mut cname = MIX_DOWN_KERNEL_NAME.as_bytes().to_vec();
                cname.push(0);
                let mut f: CUfunction = std::ptr::null_mut();
                if (lib.module_get_function)(&mut f, module, cname.as_ptr() as *const c_char)
                    != CUDA_SUCCESS
                {
                    eprintln!("[mix-down] PTX に {MIX_DOWN_KERNEL_NAME} が無い → 従来カーネルへ");
                    return None;
                }
                Some(MixDownKernels { module, f })
            }
        })
        .as_ref()
}


/// down 版が載っているか (テスト専用)。
#[cfg(test)]
pub(crate) fn mix_down_loaded() -> bool {
    mix_down_kernels().is_some()
}


// ---------------------------------------------------------------------------
// task 216: mix の up と hc_mixed を 1 本に畳む `kernels_mix_upmix.cu`。
//
// 従来は up が gate[hc*e] を書いて、その直後の `dno_hc_mixed_f32`
// (grid = ceil(e/256) = 10 ブロック) が silu して hc 方向に足していた。
// mixed しか要らないので gate の 40 KB 往復と起動 1 本が無駄。丸め順は 2 本版と
// 同一なので値はビット一致 (up の縮約は `dno_mix_up_bf16` と同じ、mixed は
// `dno_hc_mixed_f32` と同じ s 昇順 + inv_hc)。
// ---------------------------------------------------------------------------

/// upmix 版のカーネル名。
const MIX_UPMIX_KERNEL_NAME: &str = "dno_mix_upmix_bf16";


/// upmix 版の引数の数 (lo, w, xn, out, k, n, e, hc, inv_hc)。
const MIX_UPMIX_NARGS: usize = 9;


/// upmix 版が 1 ブロックで受け持つ e の数 (`.cu` の DNO_UPMIX_EPB)。
const MIX_UPMIX_EPB: usize = 2;


/// upmix 版が前提にする hc (`.cu` の DNO_UPMIX_HC)。
const MIX_UPMIX_HC: usize = 4;


pub(crate) struct MixUpmixKernels {
    #[allow(dead_code)]
    module: CUmodule,
    f: CUfunction,
}


unsafe impl Send for MixUpmixKernels {}

unsafe impl Sync for MixUpmixKernels {}


static MIX_UPMIX_KERNELS: OnceLock<Option<MixUpmixKernels>> = OnceLock::new();


fn mix_upmix_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_MIX_UPMIX_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_mix_upmix.ptx"),
    }
}


/// upmix 版のカーネルを取得する (初回だけ PTX を読む。失敗は `None` を焼き付ける)。
fn mix_upmix_kernels() -> Option<&'static MixUpmixKernels> {
    MIX_UPMIX_KERNELS
        .get_or_init(|| {
            let path = mix_upmix_ptx_path();
            let mut image = match std::fs::read(&path) {
                Ok(b) => b,
                Err(_) => return None,
            };
            let text = String::from_utf8_lossy(&image).to_string();
            if ptx_entry_params(&text, MIX_UPMIX_KERNEL_NAME) != Some(MIX_UPMIX_NARGS) {
                eprintln!(
                    "[mix-upmix] PTX の引数が {MIX_UPMIX_NARGS} 本でない → 2 本経路へ。\
                     `bash build_ptx.sh mix_upmix` で再生成する"
                );
                return None;
            }
            image.push(0);
            let lib = lib().ok()?;
            unsafe {
                let mut module: CUmodule = std::ptr::null_mut();
                if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void)
                    != CUDA_SUCCESS
                {
                    return None;
                }
                let mut cname = MIX_UPMIX_KERNEL_NAME.as_bytes().to_vec();
                cname.push(0);
                let mut f: CUfunction = std::ptr::null_mut();
                if (lib.module_get_function)(&mut f, module, cname.as_ptr() as *const c_char)
                    != CUDA_SUCCESS
                {
                    eprintln!("[mix-upmix] PTX に {MIX_UPMIX_KERNEL_NAME} が無い → 2 本経路へ");
                    return None;
                }
                Some(MixUpmixKernels { module, f })
            }
        })
        .as_ref()
}


/// upmix 版が載っているか (テスト専用)。
#[cfg(test)]
pub(crate) fn mix_upmix_loaded() -> bool {
    mix_upmix_kernels().is_some()
}


/// task 216: ビット一致版のカーネル名 (`DENNO_DEC_MIX_UPMIX=2`)。
const MIX_UPMIX2_KERNEL_NAME: &str = "dno_mix_upmix2_bf16";


/// ビット一致版の引数の数 (upmix と同じ 9 本)。
const MIX_UPMIX2_NARGS: usize = 9;


pub(crate) struct MixUpmix2Kernels {
    #[allow(dead_code)]
    module: CUmodule,
    f: CUfunction,
}


unsafe impl Send for MixUpmix2Kernels {}

unsafe impl Sync for MixUpmix2Kernels {}


static MIX_UPMIX2_KERNELS: OnceLock<Option<MixUpmix2Kernels>> = OnceLock::new();


/// ビット一致版のカーネルを取得する (同じ `kernels_mix_upmix.ptx` から別エントリを
/// 読む。初回だけ。失敗は `None` を焼き付ける)。
fn mix_upmix2_kernels() -> Option<&'static MixUpmix2Kernels> {
    MIX_UPMIX2_KERNELS
        .get_or_init(|| {
            let path = mix_upmix_ptx_path();
            let mut image = match std::fs::read(&path) {
                Ok(b) => b,
                Err(_) => return None,
            };
            let text = String::from_utf8_lossy(&image).to_string();
            if ptx_entry_params(&text, MIX_UPMIX2_KERNEL_NAME) != Some(MIX_UPMIX2_NARGS) {
                eprintln!(
                    "[mix-upmix2] PTX の引数が {MIX_UPMIX2_NARGS} 本でない → 2 本経路へ。\
                     `bash build_ptx.sh mix_upmix` で再生成する"
                );
                return None;
            }
            image.push(0);
            let lib = lib().ok()?;
            unsafe {
                let mut module: CUmodule = std::ptr::null_mut();
                if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void)
                    != CUDA_SUCCESS
                {
                    return None;
                }
                let mut cname = MIX_UPMIX2_KERNEL_NAME.as_bytes().to_vec();
                cname.push(0);
                let mut f: CUfunction = std::ptr::null_mut();
                if (lib.module_get_function)(&mut f, module, cname.as_ptr() as *const c_char)
                    != CUDA_SUCCESS
                {
                    eprintln!("[mix-upmix2] PTX に {MIX_UPMIX2_KERNEL_NAME} が無い → 2 本経路へ");
                    return None;
                }
                Some(MixUpmix2Kernels { module, f })
            }
        })
        .as_ref()
}


/// ビット一致版が載っているか (テスト専用)。
#[cfg(test)]
pub(crate) fn mix_upmix2_loaded() -> bool {
    mix_upmix2_kernels().is_some()
}


/// norm 重み (host の `[hc][e]` f32) の常駐。キーは (ポインタ, 長さ) で、
/// 内容の指紋 (16 点サンプル) が変われば載せ替える。
///
/// `norm` は `GgufLinear` ではなく素のスライスなので uid が無い。プロセス内で
/// モデルを建て直すとアドレスが再利用されうるため、ポインタだけを鍵にしない
/// (指紋が違えば再アップロードする)。
pub(crate) struct MixNormW {
    buf: DevBuffer,
    fp: u64,
}


static MIX_NORM_W: OnceLock<std::sync::Mutex<std::collections::HashMap<(usize, usize), MixNormW>>> =
    OnceLock::new();


/// 16 点サンプル + 長さ + ポインタの指紋 (内容が違えば必ず変わる実用上の鍵)。
pub(crate) fn mix_norm_fingerprint(norm: &[f32], ptr: usize) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ (norm.len() as u64);
    let n = norm.len();
    for i in 0..16u64 {
        let idx = if n == 0 {
            0
        } else {
            ((i * (n as u64 - 1)) / 16) as usize
        };
        let v = norm.get(idx).copied().unwrap_or(0.0);
        h ^= v.to_bits() as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h ^= ptr as u64;
    h = h.wrapping_mul(0x0000_0100_0000_01b3);
    h
}


/// norm 重みを VRAM に載せて (初回だけ) ポインタを返す。
fn mix_norm_weight(norm: &[f32]) -> Result<CUdeviceptr> {
    let reg = MIX_NORM_W.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let mut map = match reg.lock() {
        Ok(g) => g,
        Err(_) => bail!("mix norm 重みの登録簿が poisoned"),
    };
    let key = (norm.as_ptr() as usize, norm.len());
    let fp = mix_norm_fingerprint(norm, key.0);
    if let Some(w) = map.get(&key) {
        if w.fp == fp {
            return Ok(w.buf.ptr());
        }
        map.remove(&key);
    }
    let buf = DevBuffer::from_slice_tagged(norm, crate::mem_audit::TAG_MIX_F32)?;
    let ptr = buf.ptr();
    map.insert(key, MixNormW { buf, fp });
    Ok(ptr)
}


/// task 184: decode の hc mix を norm から inject まで 1 回の submission で閉じる。
///
/// `x` は mix の入力 (`[hc][e]` = 残差ストリーム)。`mixed` は `[e]`、
/// `inj_out` は `[hc]` (1/hc は掛けない。呼び出し側が `hc_inject_apply` する)。
/// `inject` が None のとき (head の mix) は inject を撃たず `inj_out` を 0 で埋める。
///
/// `false` = 従来経路 (ホストの norm / inject + chain2) を使うこと。
#[allow(clippy::too_many_arguments)]
pub fn maybe_gpu_mix_bf16_chain3_into(
    down_uid: u64,
    down_bytes: &[u8],
    k1: usize,
    n1: usize,
    up_uid: u64,
    up_bytes: &[u8],
    k2: usize,
    n2: usize,
    inject: Option<(u64, usize, usize, &[u8])>,
    norm: &[f32],
    eps: f32,
    inv_hc: f32,
    e: usize,
    hc: usize,
    x: &[f32],
    mixed: &mut [f32],
    inj_out: &mut [f32],
) -> bool {
    use std::sync::atomic::Ordering as AO;
    #[cfg(test)]
    MIX_DEV_ATTEMPTS.fetch_add(1, AO::Relaxed);
    if !cuda_enabled() || MIX_GPU_BROKEN.load(AO::Relaxed) {
        return false;
    }
    if k1 == 0
        || n1 == 0
        || n2 == 0
        || e == 0
        || hc == 0
        || k2 != n1
        || n2 != k1
        || n2 / e != hc
        || norm.len() != k1
        || x.len() != k1
        || mixed.len() != e
        || inj_out.len() != hc
        || k1 < n1
        || down_bytes.len() != k1.saturating_mul(n1).saturating_mul(2)
        || up_bytes.len() != n1.saturating_mul(n2).saturating_mul(2)
    {
        return false;
    }
    if let Some((_, ik, inn, ib)) = inject {
        if ik != k1 || inn != hc || ib.len() != ik.saturating_mul(inn).saturating_mul(2) {
            return false;
        }
    }
    let Some(ctx) = shared_ctx() else {
        return false;
    };
    if let Err(err) = ctx.make_current() {
        mix_gpu_broken_once(&format!("cuCtxSetCurrent (chain3): {err:#}"));
        return false;
    }
    // デバイスの norm / inject と BF16 の mix 重みは同じ前提 (BF16 常駐)。
    if !mix_bf16_enabled() {
        return false;
    }
    let Some(f_dev) = mix_dev_kernels() else {
        return false;
    };
    let Some(f_mixed) = hc_mixed_f32_fn(ctx) else {
        return false;
    };
    // task 216: down は uint4 版 (`DENNO_DEC_MIX_DOWNW`、k が 8 の倍数のとき) か、
    // 従来カーネルを `DENNO_DEC_MIX_DTHR` スレッドで回す (既定 256 = これまでと同じ)。
    let down_fast = if dec_mix_downw_on() && k1 % 8 == 0 {
        mix_down_kernels().map(|k| k.f)
    } else {
        None
    };
    let (f_down, down_thr): (CUfunction, u32) = match down_fast {
        Some(f) => (f, MIX_DOWN_THREADS as u32),
        None => match matvec_bf16_r_silu_fn(ctx) {
            Some(f) => (f, dec_mix_dthr() as u32),
            None => return false,
        },
    };
    let up_reduce = mix_up_reduce_enabled();
    // task 216: up と hc_mixed を 1 本に畳む版 (`DENNO_DEC_MIX_UPMIX`)。
    // =1 は uint4 版 (行内の合計順が変わる)、=2 は **ビット一致**版。
    // どちらも hc == 4 が前提。=1 は更に k が 8 の倍数 (uint4 読み) が前提。
    let upmix_ver = dec_mix_upmix_ver();
    let upmix = if up_reduce && hc == MIX_UPMIX_HC {
        match upmix_ver {
            2 => mix_upmix2_kernels().map(|k| k.f),
            1 if k2 % 8 == 0 => mix_upmix_kernels().map(|k| k.f),
            _ => None,
        }
    } else {
        None
    };
    let upmix_v2 = upmix.is_some() && upmix_ver == 2;
    // task 192: `DENNO_MIX_DEC_FOLD` では up を 1 warp = 1 行の版で回す
    // (k が 8 の倍数のときだけ。形が違えば従来カーネルへ)。
    // task 216: 同じ版を `DENNO_DEC_MIX_UPW` 単独でも選べるようにした
    // (task 213 は fold と束ねて測っていたので、up だけの効果が分かれていなかった)。
    let up_fast = if upmix.is_none()
        && (mix_dec_upw_on() || dec_mix_upw_on())
        && up_reduce
        && k2 % 8 == 0
    {
        mix_up_kernels().map(|k| k.f)
    } else {
        None
    };
    let f_up = if up_fast.is_some() {
        up_fast
    } else if up_reduce {
        matvec_bf16_r_fn(ctx)
    } else {
        matvec_bf16_fn(ctx)
    };
    let Some(f_up) = f_up else {
        return false;
    };
    let up_thr: u32 = if up_fast.is_some() {
        256
    } else if up_reduce {
        256
    } else {
        1
    };
    let up_grid: u32 = if up_fast.is_some() {
        n2.div_ceil(MIX_UP_ROWS) as u32
    } else {
        n2 as u32
    };
    let wp1 = match mix_gpu_weight(down_uid, down_bytes, k1, n1, true) {
        Ok(w) => w.ptr,
        Err(err) => {
            mix_gpu_broken_once(&format!("chain3 down 重みの常駐に失敗: {err:#}"));
            return false;
        }
    };
    let wp2 = match mix_gpu_weight(up_uid, up_bytes, n1, n2, true) {
        Ok(w) => w.ptr,
        Err(err) => {
            mix_gpu_broken_once(&format!("chain3 up 重みの常駐に失敗: {err:#}"));
            return false;
        }
    };
    let wpj = match inject {
        Some((uid, ik, inn, ib)) => match mix_gpu_weight(uid, ib, ik, inn, true) {
            Ok(w) => Some(w.ptr),
            Err(err) => {
                mix_gpu_broken_once(&format!("chain3 inject 重みの常駐に失敗: {err:#}"));
                return false;
            }
        },
        None => None,
    };
    let wpg = if mix_norm_host_ready(hc, down_thr) {
        // task 299: 正規化をホストで済ませる回は norm を撃たないので、重みの
        // 常駐 (指紋 16 点 + mutex) も要らない。
        0
    } else {
        match mix_norm_weight(norm) {
            Ok(p) => p,
            Err(err) => {
                mix_gpu_broken_once(&format!("chain3 norm 重みの常駐に失敗: {err:#}"));
                return false;
            }
        }
    };
    let Some(lib) = lib().ok() else {
        return false;
    };
    const THREADS: usize = 256;
    let r = MIX_GPU_SC.with(|cell| -> Result<()> {
        // task 177: mix 1 回の操作別内訳 (`DENNO_GDN_DEC_KPROF`・既定 off)。
        // site 2 = norm/inject までデバイスで閉じる経路。
        let _kp_win = dec_kprof::window(
            dec_kprof::MIX,
            dec_kprof::cur_layer(),
            dec_kprof::cur_pos(),
            2,
        );
        let mut s = cell.borrow_mut();
        let MixGpuScratch {
            x: dx,
            lo: dlo,
            out: dout,
            mixed: dm,
            hx,
            ho,
            ..
        } = &mut *s;
        Scratch::ensure(dx, k1 * 4)?;
        Scratch::ensure(dlo, n1 * 4)?;
        Scratch::ensure(dout, n2 * 4)?;
        // mixed[e] の後ろに inj[hc] を続けて置き、D2H を 1 本で済ませる。
        Scratch::ensure(dm, e * 4 + hc * 4)?;
        Scratch::ensure_pin(hx, k1 * 4)?;
        Scratch::ensure_pin(ho, e * 4 + hc * 4)?;
        let (dx, dlo, dout, dm) = (
            dx.as_ref().unwrap(),
            dlo.as_ref().unwrap(),
            dout.as_ref().unwrap(),
            dm.as_ref().unwrap(),
        );
        let (hx, ho) = (hx.as_mut().unwrap(), ho.as_mut().unwrap());
        // task 307: 鎖の末尾に積むルータ GEMV の重み (同期の後まで生かす)。
        let mut router_w: Option<GgufWRef> = None;
        // inject の出力は mixed の直後 (down の出力 dlo とは別領域)。
        let pinj = dm.ptr() + (e * 4) as CUdeviceptr;
        // task 299: この回は正規化をホストで済ませる (`dno_mix_norm` を撃たない)。
        let norm_host = mix_norm_host_ready(hc, down_thr);
        dec_kprof::step("mix_h2d_pack", || {
            if norm_host {
                mix_norm_host_into(x, norm, e, hc, eps, &mut hx.as_mut_slice::<f32>()[..k1]);
            } else {
                hx.as_mut_slice::<f32>()[..k1].copy_from_slice(x);
            }
        });
        let kp_cp = dec_kprof::op_begin(
            || "H2D@a".to_string(),
            (0, 0, 0),
            (0, 0, 0),
            0,
            k1 * 4,
        );
        unsafe {
            check(
                (lib.memcpy_htod_async)(
                    dx.ptr(),
                    hx.ptr as *const c_void,
                    k1 * 4,
                    std::ptr::null_mut(),
                ),
                "cuMemcpyHtoDAsync (mix chain3)",
            )?;
            // 1. grouped RMS 正規化 (x → xn、in-place)。block = 1 ストリーム。
            // task 192: `DENNO_MIX_DEC_FOLD` では norm と inject を 1 本に畳む
            // (hc 方向を 1 ブロックへ。値は 2 本版とビット一致)。
            // hc は block = 256 * hc に畳む (グループ = ストリーム)。
            // task 275: norm を hc ブロックへ割り、inject を down の epilogue へ。
            // (down_thr >= 256 が前提: inject の木は 256 スレッド固定)
            let folded = if mix_dec_fold_on() && hc <= 4 {
                mix_fold_kernels().map(|k| k.f)
            } else {
                None
            };
            let down_inj = if mix_norm_par_on() && hc >= 1 && hc <= 4 && down_thr >= 256 {
                mix_dev3_kernels().map(|k| k.f)
            } else {
                None
            };
            // task 299: `norm_host` のときは正規化済みの x を上げてあるので撃たない。
            debug_assert!(!norm_host || down_inj.is_some(), "norm_host は down_inj 経路が前提");
            if down_inj.is_some() {
                // 1. grouped RMS 正規化だけ (grid = hc = ブロックごとに 1 ストリーム)。
                //    inject は 3. の down の epilogue で回す (x は読み取りのみ)。
                if !norm_host {
                    let (mut px, mut pg, mut po) = (dx.ptr(), wpg, dx.ptr());
                    let (mut ec, mut epsc) = (e as c_int, eps);
                    let mut args = [
                        &mut px as *mut _ as *mut c_void,
                        &mut pg as *mut _ as *mut c_void,
                        &mut po as *mut _ as *mut c_void,
                        &mut ec as *mut _ as *mut c_void,
                        &mut epsc as *mut _ as *mut c_void,
                    ];
                    ctx.launch_async(
                        f_dev.norm,
                        (hc as u32, 1, 1),
                        (THREADS as u32, 1, 1),
                        0,
                        &mut args,
                    )?;
                }
            } else if let Some(f_fold) = folded {
                let (mut px, mut pg, mut po) = (dx.ptr(), wpg, dx.ptr());
                let (mut pw, mut pi) = (
                    wpj.unwrap_or(0),
                    if wpj.is_some() { pinj } else { 0 },
                );
                let (mut ec, mut hcc, mut kc) = (e as c_int, hc as c_int, k1 as c_int);
                let mut epsc = eps;
                let mut args = [
                    &mut px as *mut _ as *mut c_void,
                    &mut pg as *mut _ as *mut c_void,
                    &mut po as *mut _ as *mut c_void,
                    &mut pw as *mut _ as *mut c_void,
                    &mut pi as *mut _ as *mut c_void,
                    &mut ec as *mut _ as *mut c_void,
                    &mut hcc as *mut _ as *mut c_void,
                    &mut kc as *mut _ as *mut c_void,
                    &mut epsc as *mut _ as *mut c_void,
                ];
                ctx.launch_async(
                    f_fold,
                    (1, 1, 1),
                    ((THREADS * hc) as u32, 1, 1),
                    0,
                    &mut args,
                )?;
            } else {
                {
                    let (mut px, mut pg, mut po) = (dx.ptr(), wpg, dx.ptr());
                    let (mut ec, mut epsc) = (e as c_int, eps);
                    let mut args = [
                        &mut px as *mut _ as *mut c_void,
                        &mut pg as *mut _ as *mut c_void,
                        &mut po as *mut _ as *mut c_void,
                        &mut ec as *mut _ as *mut c_void,
                        &mut epsc as *mut _ as *mut c_void,
                    ];
                    ctx.launch_async(
                        f_dev.norm,
                        (hc as u32, 1, 1),
                        (THREADS as u32, 1, 1),
                        0,
                        &mut args,
                    )?;
                }
                // 2. inject の内積 (xn → inj)。inject が無い層は撃たない。
                if let (Some(pwj), Some(_)) = (wpj, inject) {
                    let (mut pw, mut px, mut po) = (pwj, dx.ptr(), pinj);
                    let (mut kc, mut nc) = (k1 as c_int, hc as c_int);
                    let mut args = [
                        &mut pw as *mut _ as *mut c_void,
                        &mut px as *mut _ as *mut c_void,
                        &mut po as *mut _ as *mut c_void,
                        &mut kc as *mut _ as *mut c_void,
                        &mut nc as *mut _ as *mut c_void,
                    ];
                    ctx.launch_async(
                        f_dev.inj,
                        (hc as u32, 1, 1),
                        (THREADS as u32, 1, 1),
                        0,
                        &mut args,
                    )?;
                }
            }
            // 3. down + lo_apply (silu 込み) → lo はデバイスに残る。
            // task 327: inject を「追加ブロック」(grid = n + hc) で回す版が使えるか。
            // inject の無い層 (wpj = None) は grid = n のままで、epilogue 版と同じ
            // (inject のブロックを撃たない)。
            let inj3 = if down_inj.is_some() && wpj.is_some() && mix_injsep_on() {
                mix_inj3_kernels().map(|k| k.f)
            } else {
                None
            };
            if let Some(f_di) = down_inj {
                // task 275: inject を同じ起動の epilogue で回す (blockIdx.x < hc)。
                let (mut ph, mut pw, mut po) = (dx.ptr(), wp1, dlo.ptr());
                let (mut kc, mut nc) = (k1 as c_int, n1 as c_int);
                let mut invc = inv_hc;
                let (mut pwi, mut pinj_a) = (wpj.unwrap_or(0), pinj);
                let (mut kic, mut hcc2) = (k1 as c_int, hc as c_int);
                if wpj.is_none() {
                    pinj_a = 0;
                }
                let mut args = [
                    &mut ph as *mut _ as *mut c_void,
                    &mut pw as *mut _ as *mut c_void,
                    &mut po as *mut _ as *mut c_void,
                    &mut kc as *mut _ as *mut c_void,
                    &mut nc as *mut _ as *mut c_void,
                    &mut invc as *mut _ as *mut c_void,
                    &mut pwi as *mut _ as *mut c_void,
                    &mut pinj_a as *mut _ as *mut c_void,
                    &mut kic as *mut _ as *mut c_void,
                    &mut hcc2 as *mut _ as *mut c_void,
                ];
                // task 327: `inj3` があるときは inject の行を後ろの hc ブロックへ出す。
                let (f_di, gx) = match inj3 {
                    Some(f3) => (f3, n1 as u32 + hc as u32),
                    None => (f_di, n1 as u32),
                };
                ctx.launch_async(f_di, (gx, 1, 1), (down_thr, 1, 1), 0, &mut args)?;
            } else {
                let (mut ph, mut pw, mut po) = (dx.ptr(), wp1, dlo.ptr());
                let (mut kc, mut nc) = (k1 as c_int, n1 as c_int);
                let mut invc = inv_hc;
                let mut args = [
                    &mut ph as *mut _ as *mut c_void,
                    &mut pw as *mut _ as *mut c_void,
                    &mut po as *mut _ as *mut c_void,
                    &mut kc as *mut _ as *mut c_void,
                    &mut nc as *mut _ as *mut c_void,
                    &mut invc as *mut _ as *mut c_void,
                ];
                // task 216: スレッド数は `down_thr` (uint4 版は 512、従来版は
                // `DENNO_DEC_MIX_DTHR` の値。両カーネルとも blockDim から縮約を組む)。
                ctx.launch_async(f_down, (n1 as u32, 1, 1), (down_thr, 1, 1), 0, &mut args)?;
            }
            // 4. up: 入力は lo_apply 済みの lo。
            // task 216: `DENNO_DEC_MIX_UPMIX` のときは 4 と 5 を 1 本に畳む
            // (gate[hc*e] の 40 KB 往復と起動 1 本が消える。値はビット一致)。
            if let Some(f_upmix) = upmix {
                let (mut pl, mut pwu, mut pxn, mut pm) = (dlo.ptr(), wp2, dx.ptr(), dm.ptr());
                let (mut kc, mut nc, mut ec, mut hcc) =
                    (k2 as c_int, n2 as c_int, e as c_int, hc as c_int);
                let mut invc = inv_hc;
                let mut args4 = [
                    &mut pl as *mut _ as *mut c_void,
                    &mut pwu as *mut _ as *mut c_void,
                    &mut pxn as *mut _ as *mut c_void,
                    &mut pm as *mut _ as *mut c_void,
                    &mut kc as *mut _ as *mut c_void,
                    &mut nc as *mut _ as *mut c_void,
                    &mut ec as *mut _ as *mut c_void,
                    &mut hcc as *mut _ as *mut c_void,
                    &mut invc as *mut _ as *mut c_void,
                ];
                // ビット一致版は 1 ブロック = 1 出力 e、uint4 版は 1 ブロック = 2 個。
                let grid = if upmix_v2 {
                    (e.max(1) as u32, 1, 1)
                } else {
                    (e.div_ceil(MIX_UPMIX_EPB).max(1) as u32, 1, 1)
                };
                ctx.launch_async(f_upmix, grid, (256, 1, 1), 0, &mut args4)?;
            } else {
                let (mut ph2, mut pw2, mut po2) = (dlo.ptr(), wp2, dout.ptr());
                let (mut kc2, mut nc2) = (n1 as c_int, n2 as c_int);
                let mut args2 = [
                    &mut ph2 as *mut _ as *mut c_void,
                    &mut pw2 as *mut _ as *mut c_void,
                    &mut po2 as *mut _ as *mut c_void,
                    &mut kc2 as *mut _ as *mut c_void,
                    &mut nc2 as *mut _ as *mut c_void,
                ];
                ctx.launch_async(f_up, (up_grid, 1, 1), (up_thr, 1, 1), 0, &mut args2)?;
                // 5. hc_mixed: xn と gate から mixed を作る。
                let (mut px, mut pg, mut pm) = (dx.ptr(), dout.ptr(), dm.ptr());
                let (mut ec, mut hcc) = (e as c_int, hc as c_int);
                let mut invc2 = inv_hc;
                let mut args3 = [
                    &mut px as *mut _ as *mut c_void,
                    &mut pg as *mut _ as *mut c_void,
                    &mut pm as *mut _ as *mut c_void,
                    &mut ec as *mut _ as *mut c_void,
                    &mut hcc as *mut _ as *mut c_void,
                    &mut invc2 as *mut _ as *mut c_void,
                ];
                let hcblk = dec_mix_hcblk();
                let grid = (e.div_ceil(hcblk).max(1) as u32, 1, 1);
                ctx.launch_async(f_mixed, grid, (hcblk as u32, 1, 1), 0, &mut args3)?;
            }
            // task 307: ルータ GEMV を鎖の末尾に 1 発積む (D2H は下の 1 本と
            // 同じ同期でまとめる)。載らなければ 0 = 従来経路。
            let _ = router_in_mix_enqueue(ctx, dm.ptr(), e, &mut router_w);
            // 6. D2H 1 本 (mixed[e] + inj[hc] は同じバッファに連続で置いてある)。
            let kp_cp = dec_kprof::op_begin(
                || "D2H@a".to_string(),
                (0, 0, 0),
                (0, 0, 0),
                0,
                (e + hc) * 4,
            );
            check(
                (lib.memcpy_dtoh_async)(
                    ho.ptr,
                    dm.ptr(),
                    (e + hc) * 4,
                    std::ptr::null_mut(),
                ),
                "cuMemcpyDtoHAsync (mix chain3)",
            )?;
            dec_kprof::op_end(kp_cp);
            crate::prof::mark_d2h_src(crate::prof::D2H_SRC_LINEAR);
        }
        // task 204: `DENNO_DEC_ONEVNT=1` では `cuCtxSynchronize` (コンテキスト
        // 全体 = サイドストリームも待つ) の代わりに null ストリームのイベント
        // 1 本で待つ。待つ範囲だけが変わり、発行済みの仕事は 1 つも変わらない。
        if mix_onevnt_on() && sync_default_stream() {
            // イベントで待てた (台帳には sync 1 回として積まれる)。
        } else {
            ctx.sync()?;
        }
        drop(router_w);
        dec_kprof::step("mix_pack", || {
            let got = ho.as_slice::<f32>();
            mixed.copy_from_slice(&got[..e]);
            if wpj.is_some() {
                inj_out.copy_from_slice(&got[e..e + hc]);
            } else {
                for v in inj_out.iter_mut() {
                    *v = 0.0;
                }
            }
        });
        Ok(())
    });
    match r {
        Ok(()) => {
            set_last_linear_ran_on_gpu(true);
            true
        }
        Err(e) => {
            mix_gpu_broken_once(&format!("hc mix chain3 failed: {e:#}"));
            false
        }
    }
}


// ---------------------------------------------------------------------------
// task 392 (2/n): mix 鎖が作った `mixed` のデバイスコピーを GDN の入力に使う
// (`DENNO_GDN_X_MIX`)
// ---------------------------------------------------------------------------
//
// ssm 層の入力 `cur` は、直前の attn mix が作った `mixed` そのもの。mix の
// Q8_0 鎖 (`maybe_gpu_mix_q_chain_into`) は `hc_mixed` の出力を**デバイスにも
// 残している** (`dm`) ので、GDN の submission は同じバイト列を H2D で
// 上げ直さなくてよい。lab t392a の `[dec-kprof-avg]` (gdn) では 1 層あたり
// H2D が 5.6 us (36 KB、pageable の同期コピー) なので、36 層で
// 0.2 ms/token。値は同じバイト列なのでビット一致。
//
// 鎖が不発 / 形が違う / ba をホストで計算するモデル (ba_units が空) では
// 従来どおり H2D する (fail closed)。
pub const GDN_X_MIX_ENV: &str = "DENNO_GDN_X_MIX";


/// [`GDN_X_MIX_ENV`] の解釈 (純関数)。**既定 on** (lab t392c で採否 yes)。
///
/// 明示的な偽値 (`0` / `off` / `false` / `no`) だけ切る。
pub(crate) fn gdn_x_mix_from_value(v: Option<&str>) -> bool {
    !matches!(
        v.map(str::trim).map(|s| s.to_ascii_lowercase()).as_deref(),
        Some("0") | Some("off") | Some("false") | Some("no")
    )
}


/// `DENNO_GDN_X_MIX` が有効か (プロセスで 1 回だけ env を読む)。
pub(crate) fn gdn_x_mix_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| gdn_x_mix_from_value(std::env::var(GDN_X_MIX_ENV).ok().as_deref()))
}

/// 控えを使って H2D を省いた層数 / 控えが無くて H2D した層数。
static GDN_X_MIX_HITS: AtomicU64 = AtomicU64::new(0);

static GDN_X_MIX_MISS: AtomicU64 = AtomicU64::new(0);


/// 直前の mix 鎖が作った `mixed` (デバイス) を控える。
pub(crate) fn gdn_x_in_mix_arm(ptr: u64, n: usize) {
    if ptr == 0 || n == 0 {
        return;
    }
    GDN_X_LEN.store(n as u64, Ordering::Relaxed);
    GDN_X_PTR.store(ptr, Ordering::Relaxed);
}


/// 控えを取り出す (1 回で消える)。要素数が合わなければ `None`。
pub(crate) fn gdn_x_in_mix_take(n: usize) -> Option<u64> {
    if GDN_X_LEN.load(Ordering::Relaxed) != n as u64 {
        return None;
    }
    let p = GDN_X_PTR.swap(0, Ordering::Relaxed);
    GDN_X_LEN.store(0, Ordering::Relaxed);
    (p != 0).then_some(p)
}


/// H2D を省いた層を 1 つ数える。
pub(crate) fn note_gdn_x_mix_hit() {
    GDN_X_MIX_HITS.fetch_add(1, Ordering::Relaxed);
}


/// 控えが無くて H2D した層を 1 つ数える。
pub(crate) fn note_gdn_x_mix_miss() {
    GDN_X_MIX_MISS.fetch_add(1, Ordering::Relaxed);
}


/// 1 リクエストぶんの 1 行 (ゲート on で 1 層も通らなければ `None`)。
pub fn gdn_x_mix_line() -> Option<String> {
    gdn_x_mix_line_with(
        gdn_x_mix_on(),
        GDN_X_MIX_HITS.load(Ordering::Relaxed),
        GDN_X_MIX_MISS.load(Ordering::Relaxed),
    )
}


/// [`gdn_x_mix_line`] の本体 (テストで両側を踏めるように引数で受ける)。
pub(crate) fn gdn_x_mix_line_with(on: bool, hits: u64, miss: u64) -> Option<String> {
    if !on || (hits == 0 && miss == 0) {
        return None;
    }
    Some(format!("[gdn-x-mix] no_h2d_layers={hits} h2d_layers={miss}"))
}


/// 積算を 0 に戻す (リクエスト終端)。
pub fn gdn_x_mix_reset() {
    GDN_X_MIX_HITS.store(0, Ordering::Relaxed);
    GDN_X_MIX_MISS.store(0, Ordering::Relaxed);
    GDN_X_PTR.store(0, Ordering::Relaxed);
    GDN_X_LEN.store(0, Ordering::Relaxed);
}


// ---------------------------------------------------------------------------
// task 464: ssm 層の attn mix で鎖の待ちを GDN の同期へ後ろ倒しにする
// (`DENNO_MIX_NOWAIT`)
// ---------------------------------------------------------------------------
//
// decode 1 トークンは mix の Q8_0 鎖 (`maybe_gpu_mix_q_chain_into`) を 97 回
// 呼び、そのたびに `cuEventSynchronize` 1 本で待つ。ssm 層 (36 層) の **attn の
// mix** に限れば、鎖の出力 `mixed` をホストで読む相手が居ない:
//
// - 直後の GDN submission は `DENNO_GDN_X_MIX` でデバイスの `dm` をそのまま読む
//   (ホストの `cur` は使わない)。
// - inject (`ij`) は鎖の外でホストが計算する (`pre_wait`)。
// - router GEMV が鎖に載るのは ffn の mix だけで、その回は defer しない。
//
// よって (a) `mixed` の D2H を撃たず、(b) 鎖のイベントを待たずに GDN の
// submission (同じ null ストリーム) の最後の D2H 待ちへまとめられる。値は同じ
// バイト列 (ホストへ降ろさないだけ) なのでビット一致。
//
// 次に鎖を呼ぶときは scratch (`stage` / `hx`) を上書きする前に前回のイベントを
// 待つ ([`mix_pending_drain`])。実際にはその前に GDN が同期しているので待ちは
// ほぼ 0。イベントが張れない環境では従来どおり文脈同期して defer しない。
//
// **既定 on** (lab t464b で採否: mix 段 2.81/2.85/2.82 -> 2.68/2.62/2.62 ms/token、
// step 11.37 -> 11.23、md5 全 SAME)。切り戻しは `DENNO_MIX_NOWAIT=0`。
pub const MIX_NOWAIT_ENV: &str = "DENNO_MIX_NOWAIT";


/// [`MIX_NOWAIT_ENV`] の解釈 (純関数)。**既定 on**、明示的な偽値だけ off。
pub(crate) fn mix_nowait_from_value(v: Option<&str>) -> bool {
    !matches!(
        v.map(str::trim).map(|s| s.to_ascii_lowercase()).as_deref(),
        Some("0") | Some("off") | Some("false") | Some("no") | Some("")
    )
}


/// `DENNO_MIX_NOWAIT` が有効か (プロセスで 1 回だけ env を読む)。
pub(crate) fn mix_nowait_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| mix_nowait_from_value(std::env::var(MIX_NOWAIT_ENV).ok().as_deref()))
}


// ---------------------------------------------------------------------------
// task 468: full 層の attn mix も defer する (`DENNO_MIX_NOWAIT=full`)
// ---------------------------------------------------------------------------
//
// full 層 (12 層/token) の attn の mix も、鎖の出力 `mixed` を**ホストで読まない**
// 相手が居る: 直後の full attention の device 経路 (`dec_full_dev`) が qkv GEMV の
// 入力に同じデバイスバッファ (`dm`) を使えるようにした (task 468)。よって ssm 層と
// 同じく (a) D2H を撃たず (b) イベントも待たない。待ちは次の鎖 (同じ層の ffn mix)
// の入口の drain へ移るが、その前に full attention の 5 起動と同期が入るので
// 待ちはほぼ 0 になる。
//
// **既定 on** (lab t468c で採否: 14k 後の decode が 12.3 -> 12.0 ms/token、
// 短い decode と prefill は不変、md5 全 SAME)。切り戻しは `DENNO_MIX_NOWAIT=ssm`
// (ssm の defer は残して full だけ止める) か `=0` (両方止める)。
// 既存の env に値を足す形にした (lab の COND は新しい env 名を落とすことがある)。
pub(crate) fn mix_nowait_full_from_value(v: Option<&str>) -> bool {
    match v.map(str::trim).map(|s| s.to_ascii_lowercase()).as_deref() {
        None => true,
        Some("0") | Some("off") | Some("false") | Some("no") | Some("") | Some("ssm") => false,
        _ => true,
    }
}


/// `DENNO_MIX_NOWAIT=full` (full 層も defer) か。プロセスで 1 回だけ読む。
pub(crate) fn mix_nowait_full_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        mix_nowait_full_from_value(std::env::var(MIX_NOWAIT_ENV).ok().as_deref())
    })
}


/// 直前の鎖が残した `mixed` (デバイス) を 1 回だけ取り出す。
///
/// full attention の device 経路が qkv GEMV の入力に使う (ssm の
/// [`gdn_x_in_mix_take`] と同じ控え)。要素数が合わなければ `None`。
pub(crate) fn full_x_in_mix_take(n: usize) -> Option<u64> {
    gdn_x_in_mix_take(n)
}


/// full 層で H2D を省いた層数の行 (1 層も通らなければ `None`)。
pub fn mix_nowait_full_line() -> Option<String> {
    mix_nowait_full_line_with(
        mix_nowait_full_on(),
        crate::cuda::dec_full_dev::x_dev_hits(),
    )
}


/// [`mix_nowait_full_line`] の本体 (テストで両側を踏めるように引数で受ける)。
pub(crate) fn mix_nowait_full_line_with(on: bool, layers: u64) -> Option<String> {
    if !on || layers == 0 {
        return None;
    }
    Some(format!("[mix-nowait-full] layers={layers}"))
}


/// 積算を 0 に戻す (リクエスト終端)。
pub fn mix_nowait_full_reset() {
    crate::cuda::dec_full_dev::x_dev_reset();
}


thread_local! {
    /// 前回 defer した鎖のイベントをまだ待っていない (= 次の鎖の入口で待つ)。
    pub(crate) static MIX_PENDING_EV: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// 直前の鎖が defer したか (呼び出し側が読んで消す)。
    pub(crate) static MIX_DEFERRED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}


/// defer した層数 (発火の証拠 `[mix-nowait]`)。
static MIX_NOWAIT_HITS: AtomicU64 = AtomicU64::new(0);


/// 前回 defer した鎖の完了を待つ (scratch を上書きする前に呼ぶ)。
fn mix_pending_drain() -> Result<()> {
    if MIX_PENDING_EV.with(|c| c.replace(false)) {
        seg_ev::wait(8)?;
    }
    Ok(())
}


/// 直前の鎖が defer したか (読むと消える)。
///
/// `true` のとき `out` (mixed) はホストへ降ろされていない = ホスト値は不定。
pub(crate) fn mix_take_deferred() -> bool {
    MIX_DEFERRED.with(|c| c.replace(false))
}


/// 控え (`mixed` のデバイスコピー) が要素数 `n` で置かれているか (消費しない)。
///
/// ssm 層が「入力はデバイス側にある」と判定するために使う。
pub(crate) fn gdn_x_in_mix_armed(n: usize) -> bool {
    GDN_X_LEN.load(Ordering::Relaxed) == n as u64 && GDN_X_PTR.load(Ordering::Relaxed) != 0
}


/// defer した層数の行 (ゲート on で 1 層も通らなければ `None`)。
pub fn mix_nowait_line() -> Option<String> {
    mix_nowait_line_with(
        MIX_NOWAIT_HITS.load(Ordering::Relaxed),
        GDN_X_MIX_HITS.load(Ordering::Relaxed),
    )
}


/// [`mix_nowait_line`] の本体 (テストで両側を踏めるように引数で受ける)。
pub(crate) fn mix_nowait_line_with(deferred: u64, alias_hits: u64) -> Option<String> {
    if deferred == 0 && alias_hits == 0 {
        return None;
    }
    Some(format!(
        "[mix-nowait] deferred_layers={deferred} alias_layers={alias_hits}"
    ))
}


/// 積算を 0 に戻す (リクエスト終端)。
///
/// `MIX_PENDING_EV` は残す (defer した鎖がまだ流れているかもしれないので、次の
/// 鎖の入口で待たせる)。
pub fn mix_nowait_reset() {
    MIX_NOWAIT_HITS.store(0, Ordering::Relaxed);
    MIX_DEFERRED.with(|c| c.set(false));
}
