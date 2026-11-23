//! prefill 常駐チェーン (pf_begin / pf_pool / pf_ple_dev / pf_full_* / res_xfer)
//!
//! task 664: `src/cuda.rs` (70k 行) からの純粋な移動。ロジック・数値・カーネル・
//! PTX・env ゲートは一切変えていない。

#![allow(non_snake_case, non_camel_case_types)]

use super::*;


/// 1 層ぶんの作業域 (grow-only。スレッドローカル)。
pub(crate) struct PfScratch {
    pub(crate) x: Option<DevBuffer>,
    pub(crate) g: Option<DevBuffer>,
    pub(crate) u: Option<DevBuffer>,
    pub(crate) act: Option<DevBuffer>,
    pub(crate) h: Option<DevBuffer>,
    pub(crate) pg: Option<DevBuffer>,
    pub(crate) pu: Option<DevBuffer>,
    pub(crate) pd: Option<DevBuffer>,
    pub(crate) seg: Option<DevBuffer>,
    pub(crate) rows_gu: Option<DevBuffer>,
    pub(crate) rows_id: Option<DevBuffer>,
    /// task 162: 行の加重和 (ストリーミング経路) 用。
    pub(crate) prow: Option<DevBuffer>,
    pub(crate) wrow: Option<DevBuffer>,
    pub(crate) oacc: Option<DevBuffer>,
}


impl PfScratch {
    pub(crate) fn new() -> Self {
        PfScratch {
            x: None,
            g: None,
            u: None,
            act: None,
            h: None,
            pg: None,
            pu: None,
            pd: None,
            seg: None,
            rows_gu: None,
            rows_id: None,
            prow: None,
            wrow: None,
            oacc: None,
        }
    }
}


thread_local! {
    pub(crate) static PF_SCRATCH: std::cell::RefCell<PfScratch> =
        std::cell::RefCell::new(PfScratch::new());
}


/// task 233: prefill の終端で MoE の作業域 (14k で約 2.5 GB) を返す。
///
/// 抱えたままだと decode の expert LRU がその分だけ狭くなる (task 227 の
/// 14k 後 decode 36 ms/tok の一因)。次の prefill は同じ手順で確保し直す。
#[cfg(feature = "cuda")]
pub fn pf_scratch_drop() {
    PF_SCRATCH.with(|cell| {
        *cell.borrow_mut() = PfScratch::new();
    });
}


/// ステージング 1 枚。
pub(crate) struct PfSlot {
    pub(crate) buf: Option<DevBuffer>,
    pub(crate) cap: usize,
    /// 一度でも H2D を発行したか (`ev_use` の有効性)。
    pub(crate) used: bool,
    /// この枚への H2D の完了。
    pub(crate) ev_up: Option<CopyEvent>,
    /// この枚を読むカーネルの完了 (再利用の前に待つ)。
    pub(crate) ev_use: Option<CopyEvent>,
}


impl PfSlot {
    pub(crate) fn new() -> Self {
        PfSlot {
            buf: None,
            cap: 0,
            used: false,
            ev_up: None,
            ev_use: None,
        }
    }
}


/// スレッドごとのステージング一式。
pub(crate) struct PfStream {
    pub(crate) slot: [PfSlot; PF_STAGE_SLOTS],
    /// スロットごとの pinned バウンスバッファ (grow-only)。
    pub(crate) pin: [Option<PinMem>; PF_STAGE_SLOTS],
    /// H2D 専用の非ブロッキングストリーム。
    pub(crate) up: Option<CopyStream>,
    /// 実 H2D を 1 回だけ測る。
    pub(crate) probed: bool,
    /// task 165: グループの通し番号 (層をまたいで増える。スロットの持ち主を
    /// 一意にするため。0 に戻すと前の層の完了イベントを誤認する)。
    pub(crate) next_g: u64,
    /// task 309: この層の expert 型が K 系か。pinned バウンスの作り方
    /// (`DENNO_PF_STREAM_WC` の既定) を型で変える。
    pub(crate) k_mode: bool,
    /// task 325: スロットごとの pinned が write-combined か (確保したときの
    /// 実際の値)。WC のメモリへ `read_at` で書くと極端に遅いので、
    /// pread 経路はこのフラグが偽のスロットだけを使う。
    pin_wc: [bool; PF_STAGE_SLOTS],
}


impl PfStream {
    pub(crate) fn new() -> Self {
        PfStream {
            slot: std::array::from_fn(|_| PfSlot::new()),
            pin: std::array::from_fn(|_| None),
            up: None,
            probed: false,
            next_g: 0,
            k_mode: false,
            pin_wc: [false; PF_STAGE_SLOTS],
        }
    }
}


/// 1 グループの上げの結果 (従来経路 = 計算スレッドが直列に写す)。
pub(crate) struct PfPump {
    /// グループ内の expert の (gate, up, down) デバイスポインタ (items と同じ順)。
    pub(crate) ptrs: Vec<[u64; 3]>,
    /// 上げたバイト数。
    pub(crate) bytes: usize,
    /// ホスト側の写しの時間。
    pub(crate) copy_ns: u64,
    /// 積み込み (非同期コピー + イベント記録) の時間。
    pub(crate) enq_ns: u64,
    /// 最初の 1 回だけ測った H2D (バイト, 時間)。
    pub(crate) probe: Option<(usize, u64)>,
}


/// ホスト側の転送元 (mmap の 1 区間)。
#[derive(Clone, Copy)]
pub(crate) struct PfHostRange {
    pub(crate) ptr: usize,
    pub(crate) len: usize,
    /// task 325: mmap のときの (GGUF パート番号, ファイル内絶対オフセット)。
    /// RAM tier の区間は `None` (ファイルに無い)。
    pub(crate) src: Option<(u32, u64)>,
}


impl PfHostRange {
    pub(crate) fn of(b: &[u8]) -> Self {
        PfHostRange {
            ptr: b.as_ptr() as usize,
            len: b.len(),
            src: None,
        }
    }

    /// task 325: ファイル範囲も持つ版 (`read_at` 経路が使う)。
    pub(crate) fn of_src(b: &[u8], src: (u32, u64)) -> Self {
        PfHostRange {
            ptr: b.as_ptr() as usize,
            len: b.len(),
            src: Some(src),
        }
    }
}


/// (pinned 内のオフセット, 転送元) を pinned へ並列に写す。
///
/// 計算スレッド (従来経路) と上げ専用スレッド (task 165) が同じ関数を使う
/// (どちらもデバイスへ渡るバイト列は同じ)。
///
/// task 325: `pread` が真なら、ファイル範囲を持つ区間は `read_at` で直接
/// pinned へ読む (`DENNO_PF_PREAD`)。読めなかった区間だけ memcpy へ落ちる。
pub(crate) fn pf_copy_into_pin(pin_base: usize, parts: &[(usize, PfHostRange)], pread: bool) {
    // task 530: このグループ (= 1 回の呼び出し) の時間を、ディスク読みを
    // 含むかで分けて積む (barrier のせいでミスが直列化するかの判定材料)。
    let t_grp = std::time::Instant::now();
    let mut group_miss = 0usize;
    // task 325: pread のときは「塊」を単位に仕事を作る (1 つの区間を
    // `DENNO_PF_PREAD_KB` ずつに切り、並列度を上げる)。
    // task 474: 区間**丸ごと**が先読みの置き場 (`DENNO_PF_EARLY_ROUTE` の RAM) に
    // 載っていれば、その区間は 1 本の memcpy にする (ディスクを読まない)。
    // task 546: 6 番目は「page cache に載っている (または判定不能)」= buffered で
    // 読んでよい。`DENNO_PF_PREAD_ODIRECT_BOUNCE` のときだけ false が入る。
    let mut tasks: Vec<(usize, Option<(u32, u64)>, usize, usize, bool, bool)> = Vec::new();
    if pread {
        let blk = exp_pf::pread_bytes().max(1);
        // task 522: 読みの「形」を台帳へ積む (診断専用、既定 on)。
        let mut ledger: Vec<(u32, u64, usize)> = Vec::new();
        // task 522: まず「区間の列」を作ってから併合し、最後に塊へ切る
        // (先に切ると併合の単位が崩れる)。
        let mut runs: Vec<(usize, u32, u64, usize, usize)> = Vec::new();
        for &(off, r) in parts {
            if r.len == 0 {
                continue;
            }
            match r.src {
                Some(s) => {
                    if exp_pf::rc_has(s.0, s.1, r.len) {
                        tasks.push((off, Some(s), r.ptr, r.len, true, true));
                    } else {
                        if exp_pf::range_ledger_on() {
                            ledger.push((s.0, s.1, r.len));
                        }
                        runs.push((off, s.0, s.1, r.len, r.ptr));
                    }
                }
                None => tasks.push((off, None, r.ptr, r.len, false, true)),
            }
        }
        if !runs.is_empty() {
            if exp_pf::pread_merge_on() {
                exp_pf::merge_adjacent_runs(&mut runs);
            }
            for &(off, part, at, len, ptr) in runs.iter() {
                // task 325: 1 区間を `DENNO_PF_PREAD_KB` ずつに切る。
                for (k, n) in exp_pf::pread_blocks(len, blk) {
                    tasks.push((off + k, Some((part, at + k as u64)), ptr + k, n, false, true));
                }
            }
        }
        exp_pf::note_read_ranges(&ledger);
    } else {
        for &(off, r) in parts {
            tasks.push((off, None, r.ptr, r.len, false, true));
        }
    }
    // task 530: page cache の常駐を `mincore(2)` で予測する (計器、既定 on) と、
    // 載っていない区間を先に並べるゲート (`DENNO_PF_PREAD_NC_FIRST`)。
    // グループは barrier なので、最後に残った 1 本がディスク待ちだと
    // 残りのスレッドが遊ぶ。ミスを先に投げると、その待ちが写しの後ろに隠れる。
    // task 546: バウンス経路 (`DENNO_PF_PREAD_ODIRECT_BOUNCE`) も同じ予測を使う
    // (載っている区間を O_DIRECT で読むと page cache のヒットを捨ててしまう)。
    if pread && (exp_pf::nc_on() || exp_pf::nc_first_on() || exp_pf::odb_on()) {
        let t_nc = std::time::Instant::now();
        let mut flags: Vec<bool> = Vec::with_capacity(tasks.len());
        let (mut judged, mut miss, mut unknown) = (0usize, 0usize, 0usize);
        for &(_off, src, ptr, len, cached, _res) in tasks.iter() {
            if src.is_some() && !cached {
                // SAFETY: `ptr` は mmap の区間の先頭 (この層の処理中は有効)。
                match unsafe { exp_pf::range_resident(ptr, len) } {
                    Some(true) => {
                        flags.push(true);
                        judged += 1;
                    }
                    Some(false) => {
                        flags.push(false);
                        judged += 1;
                        miss += 1;
                        group_miss += len;
                    }
                    None => {
                        flags.push(true);
                        unknown += 1;
                    }
                }
            } else {
                flags.push(true);
            }
        }
        exp_pf::note_range_residency(
            judged,
            miss,
            group_miss,
            unknown,
            t_nc.elapsed().as_nanos() as u64,
        );
        // task 546: 判定を各タスクへ配る (true = buffered でよい)。
        for (t, &f) in tasks.iter_mut().zip(flags.iter()) {
            t.5 = f;
        }
        if exp_pf::nc_first_on() {
            let ord = exp_pf::nc_first_order(&flags);
            let moved = ord.iter().enumerate().filter(|&(i, &j)| i != j).count();
            let old = std::mem::take(&mut tasks);
            tasks = ord.into_iter().map(|i| old[i]).collect();
            exp_pf::note_nc_order(moved);
        }
    }
    type PfTask = (usize, Option<(u32, u64)>, usize, usize, bool, bool);
    let work = |&(off, src, ptr, len, cached, resident): &PfTask| {
        let dst = (pin_base + off) as *mut u8;
        if let Some((part, at)) = src {
            // SAFETY: 区間は互いに重ならず、`pin_base` は `need` バイト以上ある。
            if cached && unsafe { exp_pf::rc_copy_into(part, at, len, dst) } {
                return;
            }
            // task 546: page cache に載っていない区間だけ、O_DIRECT + バウンスで読む
            // (page cache を経由しない = 常駐を追い出さない)。載っている区間と
            // 失敗した区間は従来の buffered pread。
            if !resident && exp_pf::odb_on() {
                exp_pf::note_rd_enter(len);
                let ok = unsafe { exp_pf::read_part_ptr_od_bounce(part, at, dst, len) };
                exp_pf::note_rd_leave();
                if ok {
                    return;
                }
            } else if !resident {
                exp_pf::note_odb(0, len, 0, 0);
            }
            // task 538: ディスク (または page cache) からの pread。同時本数を数える。
            exp_pf::note_rd_enter(len);
            let ok = unsafe { exp_pf::read_part_ptr(part, at, dst, len) };
            exp_pf::note_rd_leave();
            if ok {
                return;
            }
        }
        // SAFETY: 同上。転送元は mmap (この層の処理が終わるまで有効)。
        unsafe {
            std::ptr::copy_nonoverlapping(ptr as *const u8, dst, len);
        }
    };
    #[cfg(feature = "par")]
    {
        use rayon::prelude::*;
        // task 325: pread 専用のスレッド数を指定されたら、その本数のプールで回す
        // (`read_at` は同期なので、同時に飛ぶ I/O の本数 = スレッド数)。
        // task 554: 上げスレッドが 2 本でも pool は 1 本を共有する (2 本に分けた版は
        // 既定モデルの 3,826 cold が返らなくなったので捨てた。exp_pf の
        // `UP_WORKERS_ENV` の doc を参照)。
        let nt = exp_pf::pread_threads();
        if pread && nt > 0 {
            // task 558: `DENNO_PF_PREAD_POOL_PER_UP` のときは、上げスレッドの
            // 番号ごとに 32 本のプールを持たせる (2 本で同時 64 本)。t554a/b が
            // 測った形。計算スレッド経路 (番号無し) と、ゲート off は 1 本共有のまま。
            let pool = match exp_pf::pread_pool_per_up_on() {
                true => pf_up_idx()
                    .and_then(|i| pf_pread_pool_up(nt, i))
                    .or_else(|| pf_pread_pool(nt)),
                false => pf_pread_pool(nt),
            };
            if let Some(pool) = pool {
                pool.install(|| tasks.par_iter().for_each(work));
                // task 546: バウンス経路も同じ台帳 (ディスク待ちの切り分け) を残す。
                if exp_pf::nc_on() || exp_pf::odb_on() {
                    exp_pf::note_staging_group(group_miss, t_grp.elapsed().as_nanos() as u64);
                }
                return;
            }
        }
        tasks.par_iter().for_each(work);
    }
    #[cfg(not(feature = "par"))]
    {
        tasks.iter().for_each(work);
    }
    if exp_pf::nc_on() || exp_pf::odb_on() {
        exp_pf::note_staging_group(group_miss, t_grp.elapsed().as_nanos() as u64);
    }
}


/// task 239: pinned の `[start, start + len)` だけを写す (分割 DMA 用)。
///
/// 分割しても**デバイスへ渡るバイト列は 1 分割と同一** (区間の切り方が違うだけ)。
/// `parts` の各区間と交わる部分だけを切り出して、既存の並列写しへ渡す。
pub(crate) fn pf_copy_into_pin_range(
    pin_base: usize,
    parts: &[(usize, PfHostRange)],
    start: usize,
    len: usize,
    pread: bool,
) {
    let end = start.saturating_add(len);
    let mut sub: Vec<(usize, PfHostRange)> = Vec::with_capacity(parts.len());
    for &(off, r) in parts {
        let r_end = off.saturating_add(r.len);
        let s = off.max(start);
        let e = r_end.min(end);
        if e <= s {
            continue;
        }
        sub.push((
            s,
            PfHostRange {
                ptr: r.ptr + (s - off),
                len: e - s,
                // task 325: 切り出した先頭のファイル位置へずらす。
                src: r.src.map(|(part, at)| (part, at + (s - off) as u64)),
            },
        ));
    }
    if !sub.is_empty() {
        pf_copy_into_pin(pin_base, &sub, pread);
    }
}


/// 1 グループぶんのスロット準備 (確保と配置)。
pub(crate) struct PfSlotPlan {
    pub(crate) slot: usize,
    /// グループ内の expert の (gate, up, down) デバイスポインタ (items と同じ順)。
    pub(crate) ptrs: Vec<[u64; 3]>,
    /// 上げるバイト数 (ステージングの先頭から連続)。
    pub(crate) need: usize,
    /// 行列ごとのバイト数 (pageable 経路が使う)。
    pub(crate) lens: Vec<[usize; 3]>,
    /// pinned 内の (オフセット, 転送元) をステージングと同じ並びで並べたもの。
    pub(crate) parts: Vec<(usize, PfHostRange)>,
    /// task 325: このグループの写しを `read_at` で行うか。
    pub(crate) pread: bool,
}


/// グループ `g` のステージングと pinned を用意し、置き場所を決める。
///
/// **再利用待ちは呼び出し側の責任** (上げ専用スレッドは自分のスレッドで待つ)。
/// ただしバッファを**広げる**ときだけ、ここで前のグループの計算完了を待つ
/// (使用中のデバイスメモリを解放しないため)。広げるのは最初の数層だけで、
/// 定常状態では 1 回も起きない。
pub(crate) fn pf_slot_prepare(
    st: &mut PfStream,
    g: u64,
    items: &[PfItem<'_>],
    pinned: bool,
) -> Result<PfSlotPlan> {
    let s = (g as usize) % PF_STAGE_SLOTS;
    let mut lens: Vec<[usize; 3]> = Vec::with_capacity(items.len());
    let mut need = 0usize;
    for it in items {
        let b = &it.1;
        let l = [b[0].len(), b[1].len(), b[2].len()];
        need += l
            .iter()
            .map(|&x| pf_stage_align_up(x))
            .sum::<usize>();
        lens.push(l);
    }
    // 1. 確保 (足りなければこの 1 回だけ広げる)。
    if st.slot[s].cap < need {
        // 使用中のバッファは解放しない: このスロットを読んだ計算の完了を待つ。
        if st.slot[s].used {
            if let Some(ev) = st.slot[s].ev_use.as_ref() {
                let Some(es) = lib()?.event_sync else {
                    bail!("cuEventSynchronize が無い");
                };
                unsafe {
                    check(es(ev.0), "cuEventSynchronize (stage grow)")?;
                }
            }
        }
        let want = need.max(PF_STAGE_MIN_MB << 20);
        match pf_stage_alloc(want, s) {
            Some(b) => {
                st.slot[s].cap = b.bytes();
                st.slot[s].buf = Some(b);
            }
            None => bail!(
                "ステージング {} MiB を確保できない (空き {} MiB)",
                want >> 20,
                dev_mem_mb().map(|(f, _)| f).unwrap_or(0)
            ),
        }
    }
    // 確保できたサイズが必要量に足りないなら、このグループは上げない
    // (呼び出し側が従来経路へ落ちる)。
    if st.slot[s].cap < need {
        bail!(
            "ステージングが足りない (必要 {} MiB / 確保 {} MiB)",
            need >> 20,
            st.slot[s].cap >> 20
        );
    }
    // 2. デバイス側の置き場所を決める。
    let base = st.slot[s].buf.as_ref().expect("確保済み").ptr();
    let mut ptrs: Vec<[u64; 3]> = vec![[0u64; 3]; items.len()];
    // 3. バウンスの置き場所。task 227: 転送元が pin のときだけ確保する
    //    (page / reg は mmap をそのまま DMA するので写しが要らない)。
    //    置き場所 (parts) は 3 経路とも要るので常に作る。
    // task 522: `DENNO_PF_PIN_FILE_ORDER` では (i, m) をファイル順に並べて詰める
    // (デバイスへ渡るバイト列は同じで、置き場所の順序だけが変わる)。
    let ord: Vec<(usize, usize)> = if exp_pf::pin_file_order_on() {
        let src: Vec<Option<[(u32, u64); 3]>> = items
            .iter()
            .map(|it| it.2.map(|s| [(s[0].0, s[0].1), (s[1].0, s[1].1), (s[2].0, s[2].1)]))
            .collect();
        exp_pf::pin_order(&src)
    } else {
        (0..items.len())
            .flat_map(|i| (0..3).map(move |m| (i, m)))
            .collect()
    };
    let mut parts: Vec<(usize, PfHostRange)> = Vec::with_capacity(items.len() * 3);
    let mut cursor = 0usize;
    for &(i, m) in ord.iter() {
        let b = &items[i].1[m];
        if !b.is_empty() {
            // task 325: mmap の区間はファイル位置も持たせる (pread 用)。
            let r = match items[i].2 {
                Some(s) => PfHostRange::of_src(b, s[m]),
                None => PfHostRange::of(b),
            };
            parts.push((cursor, r));
        }
        ptrs[i][m] = base + cursor as u64;
        cursor += pf_stage_align_up(lens[i][m]);
    }
    if pinned && exp_pf::stream_src() == 0 {
        let cap = st.pin[s].as_ref().map(|p| p.bytes).unwrap_or(0);
        if cap < need {
            // task 296: `DENNO_PF_PIN_GROW=1` では余裕を足して確保し、グループの
            // 必要量が数 MB 揺れるたびの `cuMemHostAlloc` を避ける。
            let want = if exp_pf::pin_grow() {
                exp_pf::pin_grow_target(need, st.slot[s].cap)
            } else {
                need
            }
            .max(1);
            // task 296: 確保し直しの回数と時間を診断で読めるようにする
            // (`DENNO_PF_LM_DBG=1` の `[pf-lm-dbg] stream` 行)。
            let t_pin = std::time::Instant::now();
            // task 349: `DENNO_PF_STREAM_PREAD` のときだけ非 WC にする
            // (WC のメモリへ `read(2)` すると極端に遅いので pread が使えない)。
            let wc = exp_pf::stream_wc_for(st.k_mode) && !exp_pf::stream_pread();
            st.pin[s] = Some(PinMem::alloc_flags(want, wc)?);
            st.pin_wc[s] = wc;
            pf_stream_t_add(6, t_pin.elapsed().as_nanos() as u64);
            pf_stream_t_add(7, 1);
        }
    }
    // task 325: pread は「pinned バウンスがあって、write-combined でない」
    // スロットだけ使う (WC への read(2) は極端に遅い)。パートが開けていない
    // ときも memcpy へ落ちる (read_part_ptr が false を返す)。
    let pread = pinned
        && exp_pf::stream_src() == 0
        && exp_pf::pread_on()
        && !st.pin_wc[s]
        && st.pin[s].is_some();
    Ok(PfSlotPlan {
        slot: s,
        ptrs,
        need,
        lens,
        parts,
        pread,
    })
}


// ---------------------------------------------------------------------------
// task 227: 転送経路の素の速度を 1 回だけ測る (`DENNO_PF_DMA_PROBE=1`)
// ---------------------------------------------------------------------------
//
// 台帳は「写し 5.90 s + H2D のブロック 8.85 s」までしか言わない。ここでは
// **同じホストで、同じ driver で** 素の速度を測り、どこが詰まっているかを
// 切り分ける。数値経路は変えない (測るだけで、結果は `[pf-probe]` の行)。
//
// 測るもの:
// 1. WC pinned → device の DMA (単独 / GPU が memset で埋まっているとき /
//    ホスト 8 スレッドが別の pinned へ写しているとき)
// 2. `cuMemcpyHtoDAsync` の**呼び出し**がホストをブロックするか (呼び出しの ns)
// 3. mmap を pageable のまま積んだ速度 (写しを消したときの上限)
// 4. mmap を `cuMemHostRegister` で pin した速度と、その登録の時間
// 5. 2 本のストリームに分けた速度 (1 本との差)
//
// # 実測 (lab t227b、256 MB。GB/s は **呼び出し + 待ち** で割った値)
//
// WC pinned 単体 56.8 / 56 MB でも 56.1 (呼び出しは 1-27 µs で**非ブロック**) /
// GPU を memset で埋めても 56.6 (**GPU と取り合わない**) / ホスト 8 スレッドが
// 写していると 27.0 / mmap を pageable で 21.1 (呼び出しが 13 ms ブロック) /
// 2 ストリーム 24.3 / **実経路の再現 (56 MB を写して DMA、6 回) 11.1** /
// 4 MB ずつ交互 9.5。実経路の実測 10.7 GB/s と一致する = DDR 3 往復の帯域律速。
pub fn pf_dma_probe(host: &[u8]) -> bool {
    static DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if DONE.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return false;
    }
    let Ok(l) = lib() else {
        return false;
    };
    let Some(ctx) = shared_ctx() else {
        return false;
    };
    if ctx.make_current().is_err() {
        return false;
    }
    let mb: usize = std::env::var("DENNO_PF_PROBE_MB")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(256);
    let want = mb.clamp(16, 1024) << 20;
    let n = want.min(host.len());
    if n < (8 << 20) {
        eprintln!("[pf-probe] 転送元が小さい ({} B) — 測らない", host.len());
        return false;
    }
    let hp = host.as_ptr() as usize;
    // 探針はプールを使わない (`usize::MAX` の枚数は無いので必ず通常の確保)。
    let Some(dev) = pf_stage_alloc(n, usize::MAX) else {
        eprintln!("[pf-probe] デバイス {} MiB を確保できない — 測らない", n >> 20);
        return false;
    };
    let Some(busy_dev) = pf_stage_alloc(64 << 20, usize::MAX) else {
        return false;
    };
    let Ok(wc) = PinMem::alloc_flags(n, true) else {
        return false;
    };
    let Ok(cc) = PinMem::alloc_flags(n, false) else {
        return false;
    };
    let (Some(create), Some(ec), Some(record), Some(esync)) = (
        l.stream_create,
        l.event_create,
        l.event_record,
        l.event_sync,
    ) else {
        return false;
    };
    let mut up: CUstream = std::ptr::null_mut();
    let mut up2: CUstream = std::ptr::null_mut();
    let mut ev: CUevent = std::ptr::null_mut();
    let mut ev2: CUevent = std::ptr::null_mut();
    unsafe {
        const NONBLOCK: u32 = 0x01;
        if create(&mut up, NONBLOCK) != CUDA_SUCCESS {
            return false;
        }
        if create(&mut up2, NONBLOCK) != CUDA_SUCCESS {
            return false;
        }
        if ec(&mut ev, 0) != CUDA_SUCCESS || ec(&mut ev2, 0) != CUDA_SUCCESS {
            return false;
        }
    }
    let devp = dev.ptr();
    let busy = busy_dev.ptr();
    let wcp = wc.ptr;
    // 中身を作る (WC は CPU が書いて DMA が読む)。
    unsafe {
        std::ptr::copy_nonoverlapping(hp as *const u8, wcp as *mut u8, n);
    }
    // 1 回の DMA を 3 回測り、(合計 GB/s, 呼び出しの最大 ns, 完了待ちの最大 ns) を返す。
    // **合計 = 呼び出し + 待ち** で割る (pageable は呼び出しがブロックするので、
    // 待ちだけで割ると桁が狂う。lab t227a で実際に踏んだ)。
    let one = |bytes: usize, from: *const c_void, stream: CUstream, ev: CUevent| -> (f64, u64, u64) {
        let mut best = 0.0f64;
        let mut enq_max = 0u64;
        let mut wait_max = 0u64;
        for _ in 0..3 {
            let t = std::time::Instant::now();
            let rc = unsafe { (l.memcpy_htod_async)(devp, from, bytes, stream) };
            let enq = t.elapsed().as_nanos() as u64;
            if rc != CUDA_SUCCESS {
                return (0.0, enq, 0);
            }
            unsafe {
                let _ = (record)(ev, stream);
            }
            let t2 = std::time::Instant::now();
            if unsafe { (esync)(ev) } != CUDA_SUCCESS {
                return (0.0, enq, 0);
            }
            let wait = t2.elapsed().as_nanos() as u64;
            let gbs = bytes as f64 / (enq + wait).max(1) as f64;
            if gbs > best {
                best = gbs;
            }
            enq_max = enq_max.max(enq);
            wait_max = wait_max.max(wait);
        }
        (best, enq_max, wait_max)
    };
    // 8 スレッドで `blk` バイトを mmap から wc の `at` へ写してから、その範囲を DMA する。
    let wcp_u = wcp as usize;
    let copy_then_dma_at = |at: usize, blk: usize| -> (f64, u64, u64) {
        std::thread::scope(|s| {
            let mut hs = Vec::new();
            for i in 0..8usize {
                let part = (blk / 8) & !4095;
                if part == 0 {
                    continue;
                }
                let off = at + i * part;
                let len = if i == 7 { at + blk - off } else { part };
                hs.push(s.spawn(move || unsafe {
                    std::ptr::copy_nonoverlapping(
                        (hp + off) as *const u8,
                        (wcp_u + off) as *mut u8,
                        len,
                    );
                }));
            }
            for h in hs {
                let _ = h.join();
            }
        });
        one(blk, (wcp as usize + at) as *const c_void, up, ev)
    };
    let copy_then_dma = |blk: usize| copy_then_dma_at(0, blk);
    // 写し (8 スレッド) + DMA を交互に流す = 実経路の再現。
    // `blk` バイトを 8 スレッドで wc へ写してから DMA するのを `reps` 回。
    let pipeline = |blk: usize, reps: usize, chunk: usize| -> (f64, u64) {
        let mut enq_max = 0u64;
        let mut wait_ns = 0u64;
        let t_all = std::time::Instant::now();
        for _ in 0..reps {
            if chunk == 0 {
                let (_, e1, w1) = copy_then_dma(blk);
                enq_max = enq_max.max(e1);
                wait_ns += w1;
            } else {
                let mut at = 0usize;
                while at < blk {
                    let b = chunk.min(blk - at);
                    let (_, e1, w1) = copy_then_dma_at(at, b);
                    enq_max = enq_max.max(e1);
                    wait_ns += w1;
                    at += b;
                }
            }
        }
        let total = (blk * reps) as f64 / t_all.elapsed().as_nanos().max(1) as f64;
        (total, enq_max)
    };
    // (a) WC pinned 単独 (256 MB)
    let (wc_alone, wc_enq, wc_wait) = one(n, wcp as *const c_void, up, ev);
    let small = (56usize << 20).min(n);
    let (wc_small, wc_small_enq, wc_small_wait) = one(small, wcp as *const c_void, up, ev);
    // (b) GPU を memset で埋めながら
    let mut memset_rounds = 0usize;
    {
        let ms = l.memset_d32_async;
        for _ in 0..2000 {
            if unsafe { ms(busy, 0, (64 << 20) / 4, std::ptr::null_mut()) } != CUDA_SUCCESS {
                break;
            }
            memset_rounds += 1;
        }
        let (wc_busy_memset, wm_enq, wm_wait) = one(n, wcp as *const c_void, up, ev);
        let _ = ctx.sync();
        // (c) ホスト 8 スレッドが別の pinned へ写しながら
        let stop = std::sync::atomic::AtomicBool::new(false);
        let cc_ptr = cc.ptr as usize;
        let (copied, (wc_busy_copy8, c8_enq, c8_wait)) = std::thread::scope(|s| {
            let mut hs = Vec::new();
            for i in 0..8usize {
                let stop = &stop;
                hs.push(s.spawn(move || {
                    let chunk = (n / 8) & !4095;
                    let off = i * chunk;
                    let mut total = 0usize;
                    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                (hp + off) as *const u8,
                                (cc_ptr + off) as *mut u8,
                                chunk,
                            );
                        }
                        total += chunk;
                    }
                    total
                }));
            }
            let r = one(n, wcp as *const c_void, up, ev);
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
            let mut sum = 0usize;
            for h in hs {
                sum += h.join().unwrap_or(0);
            }
            (sum, r)
        });
        // (d) mmap を pageable のまま (256 MB / 56 MB)
        let (page_gbs, page_enq, page_wait) = one(n, hp as *const c_void, up, ev);
        // (e) mmap を pin して直接
        let a = hp & !4095usize;
        let b = (hp + n + 4095) & !4095usize;
        let mut reg_ok = false;
        let mut reg_flags = 0u32;
        let mut reg_ns = 0u64;
        if let Some(f) = l.mem_host_register {
            for flags in PF_REG_FLAG_TRIES {
                let t = std::time::Instant::now();
                let rc = unsafe { f(a as *mut c_void, b - a, flags) };
                let ns = t.elapsed().as_nanos() as u64;
                eprintln!(
                    "[pf-probe] register flags={flags:#x} rc={rc} ms={:.1}",
                    ns as f64 / 1e6
                );
                if rc == CUDA_SUCCESS {
                    reg_ok = true;
                    reg_flags = flags;
                    reg_ns = ns;
                    break;
                }
            }
        }
        let (reg_gbs, map_gbs) = if reg_ok {
            let (g, _, _) = one(n, hp as *const c_void, up, ev);
            let mut mg = 0.0f64;
            if let Some(gd) = l.mem_host_get_device_ptr {
                let dtod = l.memcpy_dtod_async;
                let mut dptr: CUdeviceptr = 0;
                if unsafe { gd(&mut dptr, a as *mut c_void, 0) } == CUDA_SUCCESS && dptr != 0 {
                    for _ in 0..3 {
                        let t = std::time::Instant::now();
                        let rc = unsafe { (dtod)(devp, dptr, n, up) };
                        if rc != CUDA_SUCCESS {
                            break;
                        }
                        unsafe {
                            let _ = (record)(ev, up);
                        }
                        let t2 = std::time::Instant::now();
                        if unsafe { (esync)(ev) } != CUDA_SUCCESS {
                            break;
                        }
                        let g2 = n as f64
                            / (t.elapsed().as_nanos() + t2.elapsed().as_nanos()).max(1) as f64;
                        mg = mg.max(g2);
                    }
                }
            }
            (g, mg)
        } else {
            (0.0, 0.0)
        };
        // (f) 2 本のストリーム (前半 / 後半)
        let half = (n / 2) & !4095;
        let mut two = 0.0f64;
        if half > 0 {
            for _ in 0..3 {
                let t = std::time::Instant::now();
                let rc1 = unsafe { (l.memcpy_htod_async)(devp, hp as *const c_void, half, up) };
                let rc2 = unsafe {
                    (l.memcpy_htod_async)(devp + half as u64, wcp as *const c_void, half, up2)
                };
                if rc1 != CUDA_SUCCESS || rc2 != CUDA_SUCCESS {
                    break;
                }
                unsafe {
                    let _ = (record)(ev, up);
                    let _ = (record)(ev2, up2);
                }
                let t2 = std::time::Instant::now();
                if unsafe { (esync)(ev) } != CUDA_SUCCESS || unsafe { (esync)(ev2) } != CUDA_SUCCESS
                {
                    break;
                }
                let g2 = (half * 2) as f64
                    / (t.elapsed().as_nanos() + t2.elapsed().as_nanos()).max(1) as f64;
                two = two.max(g2);
            }
        }
        // (g) 実経路の再現: 56 MB を写して DMA を 6 回 / 4 MB ずつ交互に
        let (pipe_big, pipe_big_enq) = pipeline(small, 6, 0);
        let (pipe_chunk, pipe_chunk_enq) = pipeline(small, 6, 4 << 20);
        // (h) CPU の写し (mmap → cacheable pinned、1 スレッド)
        let t = std::time::Instant::now();
        unsafe {
            std::ptr::copy_nonoverlapping(hp as *const u8, cc.ptr as *mut u8, n);
        }
        let cpu_gbs = n as f64 / t.elapsed().as_nanos().max(1) as f64;

        eprintln!(
            "[pf-probe] dma n_MB={} wc_GBs={wc_alone:.1} wc_enq_us={:.1} wc_wait_us={:.1} \
             small_MB={} wc_small_GBs={wc_small:.1} wc_small_enq_us={:.1} wc_small_wait_us={:.1} \
             page_GBs={page_gbs:.1} page_enq_us={:.1} page_wait_us={:.1} \
             two_GBs={two:.1} reg_GBs={reg_gbs:.1} map_GBs={map_gbs:.1} cpu_copy_GBs={cpu_gbs:.1}",
            n >> 20,
            wc_enq as f64 / 1e3,
            wc_wait as f64 / 1e3,
            small >> 20,
            wc_small_enq as f64 / 1e3,
            wc_small_wait as f64 / 1e3,
            page_enq as f64 / 1e3,
            page_wait as f64 / 1e3
        );
        eprintln!(
            "[pf-probe] busy wc_memset_GBs={wc_busy_memset:.1} wc_memset_enq_us={:.1} \
             wc_memset_wait_us={:.1} wc_copy8_GBs={wc_busy_copy8:.1} wc_copy8_enq_us={:.1} \
             wc_copy8_wait_us={:.1} copy8_GB={:.1} memset_rounds={memset_rounds}",
            wm_enq as f64 / 1e3,
            wm_wait as f64 / 1e3,
            c8_enq as f64 / 1e3,
            c8_wait as f64 / 1e3,
            copied as f64 / 1e9
        );
        eprintln!(
            "[pf-probe] pipeline blk_MB={} big_GBs={pipe_big:.1} big_enq_us={:.1} \
             chunk4_GBs={pipe_chunk:.1} chunk4_enq_us={:.1}",
            small >> 20,
            pipe_big_enq as f64 / 1e3,
            pipe_chunk_enq as f64 / 1e3
        );
        eprintln!(
            "[pf-probe] reg ok={} flags={reg_flags:#x} reg_ms={:.1} reg_GBs_per_s={:.2}",
            u8::from(reg_ok),
            reg_ns as f64 / 1e6,
            if reg_ok && reg_ns > 0 {
                (b - a) as f64 / reg_ns as f64
            } else {
                0.0
            }
        );
        if reg_ok {
            if let Some(u) = l.mem_host_unregister {
                unsafe {
                    let _ = u(a as *mut c_void);
                }
            }
        }
    }
    true
}


/// task 481: `parts` の `[start, start+len)` だけを `dst` の先頭へ写す。
///
/// [`pf_copy_into_pin_range`] は**絶対**オフセットへ書く (大きなバウンス用)。
/// こちらは小さなバウンスの先頭へ詰める (オフセットを `start` だけ引く)。
pub(crate) fn pf_copy_into_pin_at(
    dst: usize,
    parts: &[(usize, PfHostRange)],
    start: usize,
    len: usize,
    pread: bool,
) {
    let end = start.saturating_add(len);
    let mut sub: Vec<(usize, PfHostRange)> = Vec::with_capacity(parts.len());
    for &(off, r) in parts {
        let r_end = off.saturating_add(r.len);
        let s = off.max(start);
        let e = r_end.min(end);
        if e <= s {
            continue;
        }
        sub.push((
            s - start,
            PfHostRange {
                ptr: r.ptr + (s - off),
                len: e - s,
                src: r.src.map(|(part, at)| (part, at + (s - off) as u64)),
            },
        ));
    }
    if !sub.is_empty() {
        pf_copy_into_pin(dst, &sub, pread);
    }
}


/// 1 層ぶんのストリーミング入力 (plan 順)。
///
/// `dev` は VRAM 常駐 (ヒット) のポインタ、`host` はミスのホストバイト列。
/// どちらか一方だけが `Some`。
///
/// `src` は task 325: `host` が mmap のときの (GGUF パート番号, ファイル内
/// 絶対オフセット)。RAM tier の区間は `None`。
pub struct PfStreamEntry<'a> {
    pub expert: u16,
    pub dev: Option<[u64; 3]>,
    pub host: Option<[&'a [u8]; 3]>,
    pub src: Option<[(u32, u64); 3]>,
}


/// task 325: 写しの依頼 1 本 (expert 番号, ホストのバイト列, ファイル位置)。
pub type PfItem<'a> = (u16, [&'a [u8]; 3], Option<[(u32, u64); 3]>);


/// task 233: 小さなホスト配列を**非同期で**デバイスへ積む。
///
/// 戻り値 `false` = bounce が足りない (呼び出し側は同期コピーへ落ちる)。
/// 同じ層の中で使う位置は毎回ずらすので、前のコピーが読んでいる領域を
/// 上書きしない。
pub(crate) fn pf_tiny_htod(dst: CUdeviceptr, data: &[u8]) -> bool {
    if data.is_empty() {
        return true;
    }
    const CAP: usize = 1usize << 20;
    if data.len() > CAP {
        return false;
    }
    let Ok(l) = lib() else {
        return false;
    };
    PF_TINY.with(|c| {
        let mut t = c.borrow_mut();
        if t.0.as_ref().map(|p| p.bytes).unwrap_or(0) < CAP {
            match PinMem::alloc(CAP) {
                Ok(m) => t.0 = Some(m),
                Err(_) => return false,
            }
        }
        // 使い切ったら 1 回だけ GPU を止めて先頭へ戻す。ここで止めないと、
        // まだ読まれている bounce を上書きしうる (層をまたいだ再入の保険)。
        if t.1 + data.len() > CAP {
            if let Some(ctx) = shared_ctx() {
                let _ = ctx.sync();
            }
            t.1 = 0;
        }
        let off = t.1;
        if off + data.len() > CAP {
            return false;
        }
        let pin = t.0.as_ref().expect("確保済み");
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                (pin.ptr as *mut u8).add(off),
                data.len(),
            );
            let r = (l.memcpy_htod_async)(
                dst,
                (pin.ptr as *mut u8).add(off) as *mut std::ffi::c_void,
                data.len(),
                std::ptr::null_mut(),
            );
            if r != CUDA_SUCCESS {
                return false;
            }
        }
        t.1 = off + data.len();
        true
    })
}


/// task 233: ストリーミング経路の入出力の置き場所。
///
/// 層優先 prefill は活性をデバイスへ置いたままにするので、入力も出力も
/// ホストへ戻さない版が要る (`Dev`)。
pub enum PfStreamIo<'a> {
    /// 従来 (ホストの活性)。
    Host {
        xs: &'a [f32],
        out: &'a mut [f32],
    },
    /// デバイスの活性。`sh` は共有 expert の出力 (`[ntok][e]`、MoE の作業域の
    /// 末尾の行)。`0` なら共有 expert 無し。`sh_w` はトークンごとの係数
    /// (`sigmoid(gate_inp_shexp @ x)`、長さ `ntok`)。
    Dev {
        xs: CUdeviceptr,
        out: CUdeviceptr,
        sh: CUdeviceptr,
        sh_w: &'a [f32],
    },
}


/// task 318: `DENNO_PF_PIN_STAGE` の値の解釈 (純関数)。
///
/// `1` / `true` / `yes` / `on` (と未知の値) を on、`0` / `false` / `no` / `off` /
/// 空文字を off にする。既定は off (env が無いとき)。
pub fn pf_pin_stage_from_value(v: Option<&str>) -> bool {
    match v {
        None => false,
        Some(s) => {
            let t = s.trim().to_ascii_lowercase();
            !matches!(t.as_str(), "0" | "false" | "no" | "off" | "")
        }
    }
}


/// task 518: `bytes` が「小さな往復」の窓に入るか (純関数)。
pub(crate) fn pf_pin_small_window_from(small_on: bool, bytes: usize) -> bool {
    small_on && (PF_PIN_SMALL_MIN_BYTES..PF_PIN_MIN_BYTES).contains(&bytes)
}


/// [`PF_PIN_MIN_KB_ENV`] の解釈 (純関数)。1..=262144 KiB に丸める。未設定は既定。
pub(crate) fn pf_pin_min_kb_from_value(v: Option<&str>) -> usize {
    let Some(s) = v.map(str::trim).filter(|s| !s.is_empty()) else {
        return PF_PIN_MIN_BYTES >> 10;
    };
    match s.parse::<usize>() {
        Ok(kb) if kb > 0 => kb.min(262_144),
        _ => PF_PIN_MIN_BYTES >> 10,
    }
}


/// 下限 (バイト)。プロセスで 1 回だけ env を読む。
fn pf_pin_min_bytes() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        pf_pin_min_kb_from_value(std::env::var(PF_PIN_MIN_KB_ENV).ok().as_deref()) << 10
    })
}


/// task 358: `bytes` が pinned バウンスの窓に入るか (純関数)。
///
/// 小さい転送は写しの固定費が勝ち、大きすぎる転送は pinned を食い潰す。
pub(crate) fn pf_pin_window_from(min: usize, bytes: usize) -> bool {
    (min..=PF_PIN_MAX_BYTES).contains(&bytes)
}


/// task 565: どの転送を pinned バウンスに通すかの判定 (純関数)。
///
/// `force` は**全体ゲートだけ**を飛ばす (窓の判定は同じく効く)。残差の往復
/// (`DENNO_PF_RES_PIN`) のように「この転送だけ」を通したいときに使う。
pub(crate) fn pf_pin_admit(
    force: bool,
    stage_on: bool,
    small_on: bool,
    min: usize,
    bytes: usize,
) -> bool {
    let big = (force || stage_on) && pf_pin_window_from(min, bytes);
    big || pf_pin_small_window_from(small_on, bytes)
}


/// task 358: 共有バウンスを `want` へ育てる必要があるか (純関数)。
///
/// 育つときは **2 の冪へ切り上げる**。1.4 MB → 39 MB → 143 MB と昇順に来るので、
/// 毎回ちょうどの大きさで取り直すと `cuMemHostAlloc` (device を同期させる) が
/// 何度も走る。切り上げれば 1 プロセスで数回で済む。
pub(crate) fn pf_pin_need_grow(cur: usize, want: usize) -> Option<usize> {
    if cur >= want {
        return None;
    }
    let pow = want.next_power_of_two().min(PF_PIN_MAX_BYTES);
    Some(pow.max(want))
}


/// 層ループの同期 H2D / D2H を pinned バウンス経由にするか。
fn pf_pin_stage_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        pf_pin_stage_from_value(std::env::var("DENNO_PF_PIN_STAGE").ok().as_deref())
    })
}


/// task 358: pinned バウンスの実体 (**プロセスで 1 本**、grow-only)。
///
/// task 318 は `thread_local!` に持たせたため、rayon のワーカー 14 本が
/// それぞれ 143 MiB の pinned を `cuMemHostAlloc` で取りに行った
/// (lab t318a: 14k cold 34.8 -> 55.3 s = +59%、3,826 も +10%)。pinned の
/// 確保は device 全体を同期させるので、**回数を 1 回に固定する**のが要点。
/// ここでは共有 1 本 + ロックにし、サイズは「必要になったときだけ」育てる
/// (3,826 の 39 MiB -> 14k の 143 MiB で 2 回)。
pub(crate) struct PfPinShared {
    /// 転送先/転送元の pinned バッファ。
    buf: Option<PinMem>,
    /// 育ち直した回数 (診断: 1 プロセスで数回であること)。
    grows: u64,
    /// 他のスレッドがバウンスを使っていて待った回数と合計 ns (診断)。
    waits: u64,
    wait_ns: u64,
    /// バウンス経由で流したバイト数と回数 (診断: 発火の証拠)。
    bytes: u64,
    calls: u64,
}


/// 共有バウンス (プロセスで 1 本)。
fn pf_pin_shared() -> &'static std::sync::Mutex<PfPinShared> {
    static S: OnceLock<std::sync::Mutex<PfPinShared>> = OnceLock::new();
    S.get_or_init(|| {
        std::sync::Mutex::new(PfPinShared {
            buf: None,
            grows: 0,
            waits: 0,
            wait_ns: 0,
            bytes: 0,
            calls: 0,
        })
    })
}


/// task 358: 共有バウンスの台帳 (grows, waits, wait_ns, bytes, calls)。
pub fn pf_pin_stats() -> (u64, u64, u64, u64, u64) {
    match pf_pin_shared().lock() {
        Ok(g) => (g.grows, g.waits, g.wait_ns, g.bytes, g.calls),
        Err(p) => {
            let g = p.into_inner();
            (g.grows, g.waits, g.wait_ns, g.bytes, g.calls)
        }
    }
}


/// task 358: 台帳を 0 に戻す (リクエスト終端)。
pub fn pf_pin_reset() {
    let m = pf_pin_shared();
    let mut g = match m.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    g.grows = 0;
    g.waits = 0;
    g.wait_ns = 0;
    g.bytes = 0;
    g.calls = 0;
}


/// task 358: 1 行のレポート (発火が 0 なら `None` = 出力しない)。
pub fn pf_pin_line() -> Option<String> {
    let (grows, waits, wait_ns, bytes, calls) = pf_pin_stats();
    if calls == 0 {
        return None;
    }
    Some(format!(
        "[pf-pin] calls={calls} MB={:.1} grows={grows} waits={waits} wait_ms={:.1}",
        bytes as f64 / 1e6,
        wait_ns as f64 / 1e6
    ))
}


pub(crate) fn with_pf_pin_gate<T>(
    bytes: usize,
    force: bool,
    f: impl FnOnce(*mut c_void) -> Result<T>,
) -> Result<Option<T>> {
    if bytes == 0 {
        return Ok(None);
    }
    // task 518: decode の小さな往復は専用の窓 (`DENNO_DEC_PIN_SMALL`)。
    // prefill の大きな転送は従来どおり `DENNO_PF_PIN_STAGE` + 床 1 MiB の判定。
    if !pf_pin_admit(
        force,
        pf_pin_stage_on(),
        dec_pin_small_on(),
        pf_pin_min_bytes(),
        bytes,
    ) {
        return Ok(None);
    }
    let m = pf_pin_shared();
    let t0 = std::time::Instant::now();
    let mut g = match m.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    // ロック待ち (他のスレッドがバウンスを使っていた時間) を数える。
    // 200 us を超えたものだけ「待ち」として数える (通常は 0)。
    let lock_ns = t0.elapsed().as_nanos() as u64;
    if lock_ns > 200_000 {
        g.waits += 1;
        g.wait_ns += lock_ns;
    }
    let cur = g.buf.as_ref().map(|b| b.bytes).unwrap_or(0);
    if let Some(want) = pf_pin_need_grow(cur, bytes) {
        // 新しい 1 本を取る前に古い 1 本を落とす (pinned を二重に持たない)。
        g.buf = None;
        match PinMem::alloc(want) {
            Ok(p) => {
                g.buf = Some(p);
                g.grows += 1;
            }
            Err(e) => {
                static WARNED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    tracing::info!("[pf-pin] pinned バウンスを確保できない ({want} B): {e:#}");
                }
                return Ok(None);
            }
        }
    }
    let pin = g.buf.as_ref().expect("確保済み").ptr;
    g.calls += 1;
    g.bytes += bytes as u64;
    f(pin).map(Some)
}


// ---------------------------------------------------------------------------
// task 209: prefill の活性バッファを pinned ホストメモリにする (`DENNO_PF_PIN`)
// ---------------------------------------------------------------------------
//
// 動機 (lab t203c の 14,018 トークン cold prefill の台帳): mix の D2H 13.78 GB が
// 785 ms (= 17.6 GB/s) かかっている。宛先は `vec![0.0f32; nt*e]` のページャブルな
// メモリで、driver が内部 staging を挟むためこの速さになる。このホストの実測は
// pinned 52.6 GB/s / pageable 19.6 GB/s なので、**宛先/元を pinned にするだけ**で
// 同じバイト列が 2.8 倍速く動く (値は 1 bit も変わらない)。
//
// `PfHostBuf` は「pinned が取れたら pinned、取れなければ Vec」の 1 本ぶんの
// ホストバッファ。API はスライスだけなので、呼び出し側は確保の仕方だけを変える。
// pinned は確保・解放が高いので、スレッド局所の空きリストで使い回す。

/// `DENNO_PF_PIN` の解釈 (純関数)。**既定 on** (lab t209a の pin で採用)、
/// 明示的な偽値だけ off。
pub(crate) fn pf_pin_from_value(v: Option<&str>) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        None => true,
        Some(s) => !matches!(s.as_str(), "0" | "off" | "false" | "no" | ""),
    }
}


/// テスト用: ゲートを強制する (0 = env に戻す)。
#[cfg(test)]
pub(crate) fn pf_pin_set_for_test(v: u8) {
    PF_PIN_FORCE.store(v, Ordering::Relaxed);
}


/// task 209: prefill の活性バッファを pinned ホストメモリにするか (`DENNO_PF_PIN`)。
///
/// **既定 on** (lab t209a の pin が 3,826 -3.9% / 14,018 -3.4%、both で
/// -4.8% / -4.0%。`=0` で切り戻し)。
pub(crate) fn pf_pin_on() -> bool {
    #[cfg(test)]
    {
        match PF_PIN_FORCE.load(Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| pf_pin_from_value(std::env::var("DENNO_PF_PIN").ok().as_deref()))
}


thread_local! {
    /// task 209: 使い終わった pinned ホスト領域 (スレッドごと。prefill は 1 本)。
    pub(crate) static PF_HOST_FREE: std::cell::RefCell<Vec<PinMem>> =
        const { std::cell::RefCell::new(Vec::new()) };
}


/// task 209: prefill の活性 1 本ぶんのホストバッファ。
///
/// pinned を確保できたらそちらを使い、できなければ `Vec` に落ちる。**中身は
/// 保証しない** — 読む前に全要素を書く経路 (D2H の宛先・GEMM の出力先) だけが
/// 使う。`Vec` に落ちたときは従来どおり 0 埋め。
pub struct PfHostBuf {
    mem: Option<PinMem>,
    v: Vec<f32>,
    len: usize,
}


impl PfHostBuf {
    /// 長さ `len` のバッファ (中身は不定)。pinned が取れなければ 0 埋めの `Vec`。
    pub fn new(len: usize) -> Self {
        if len > 0 && pf_pin_on() {
            // 空きリストから 1 本借りる (足りなければ新規に確保して差し替える)。
            let reused = PF_HOST_FREE.with(|c| {
                let mut q = c.borrow_mut();
                match q.pop() {
                    Some(m) if m.bytes >= len * 4 => Some(m),
                    Some(m) => {
                        q.push(m);
                        None
                    }
                    None => None,
                }
            });
            let mem = match reused {
                Some(m) => Some(m),
                None => match PinMem::alloc(len * 4) {
                    Ok(m) => Some(m),
                    Err(_) => {
                        tracing::info!("[pf-pin] pinned を確保できない → Vec に落ちる (len={len})");
                        None
                    }
                },
            };
            if let Some(mem) = mem {
                return Self {
                    mem: Some(mem),
                    v: Vec::new(),
                    len,
                };
            }
        }
        Self {
            mem: None,
            v: vec![0.0f32; len],
            len,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// pinned で確保できたか (台帳・テスト用)。
    pub fn is_pinned(&self) -> bool {
        self.mem.is_some()
    }

    pub fn as_slice(&self) -> &[f32] {
        match &self.mem {
            Some(m) => unsafe { std::slice::from_raw_parts(m.ptr as *const f32, self.len) },
            None => &self.v,
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        match &mut self.mem {
            Some(m) => unsafe { std::slice::from_raw_parts_mut(m.ptr as *mut f32, self.len) },
            None => &mut self.v,
        }
    }
}


impl Drop for PfHostBuf {
    fn drop(&mut self) {
        if let Some(m) = self.mem.take() {
            // 使い回す (スレッド局所)。コンテキスト破棄より先に drop される想定。
            PF_HOST_FREE.with(|c| c.borrow_mut().push(m));
        }
    }
}


#[cfg(test)]
mod pf_pin_tests {
    use super::*;

    /// task 209: `DENNO_PF_PIN` は **既定 on** (lab t209a の pin で採用)。
    #[test]
    fn pf_pin_gate_truth_table() {
        assert!(pf_pin_from_value(None), "未設定は既定 on");
        for v in ["0", "off", "false", "no", "", " off ", "FALSE"] {
            assert!(!pf_pin_from_value(Some(v)), "{v:?} が on になった");
        }
        for v in ["1", "on", "true", "yes", " ON "] {
            assert!(pf_pin_from_value(Some(v)), "{v:?} が off のまま");
        }
    }

    /// task 209: `DENNO_PF_MIXDEVIN` も **既定 on** (lab t209a の devin で採用)。
    #[test]
    fn pf_dev_in_gate_truth_table() {
        assert!(pf_dev_in_from_value(None), "未設定は既定 on");
        for v in ["0", "off", "false", "no", "", "FALSE"] {
            assert!(!pf_dev_in_from_value(Some(v)), "{v:?} が on になった");
        }
        for v in ["1", "on", "true", "yes"] {
            assert!(pf_dev_in_from_value(Some(v)), "{v:?} が off のまま");
        }
    }

    /// task 209: `PfHostBuf` はスライスとして読み書きでき、長さが要求どおり。
    /// pinned が取れる環境では pinned、取れない環境では `Vec` に落ちる (どちらも
    /// 値の経路は同じ)。
    #[test]
    fn pf_host_buf_round_trips_and_falls_back() {
        pf_pin_set_for_test(1); // 強制 off
        let mut b = PfHostBuf::new(37);
        assert_eq!(b.len(), 37);
        assert!(!b.is_pinned(), "強制 off で pinned になった");
        b.as_mut_slice()[0] = 1.5;
        b.as_mut_slice()[36] = -2.5;
        assert_eq!(b.as_slice()[0], 1.5);
        assert_eq!(b.as_slice()[36], -2.5);
        assert_eq!(b.as_slice().len(), 37);
        drop(b);

        pf_pin_set_for_test(2); // 強制 on (確保できなければ Vec に落ちる)
        let mut b = PfHostBuf::new(8);
        assert_eq!(b.len(), 8);
        b.as_mut_slice()[3] = 7.0;
        assert_eq!(b.as_slice()[3], 7.0);
        assert_eq!(b.as_slice().len(), 8);
        pf_pin_set_for_test(0);

        // 空でも安全。
        let b = PfHostBuf::new(0);
        assert_eq!(b.len(), 0);
        assert!(b.is_empty());
    }

    /// task 209: 記録は (ホストポインタ, nt, e) が全部一致したときだけ有効。
    #[test]
    fn pf_mixed_mark_requires_exact_match() {
        let a = [0.0f32; 4];
        let b = [0.0f32; 4];
        let mark = (a.as_ptr(), 2usize, 2usize);
        assert!(pf_mixed_mark_matches(mark, a.as_ptr(), 2, 2));
        assert!(!pf_mixed_mark_matches(mark, b.as_ptr(), 2, 2), "別ポインタ");
        assert!(!pf_mixed_mark_matches(mark, a.as_ptr(), 1, 2), "nt 違い");
        assert!(!pf_mixed_mark_matches(mark, a.as_ptr(), 2, 4), "e 違い");
        assert!(
            !pf_mixed_mark_matches((std::ptr::null(), 2, 2), a.as_ptr(), 2, 2),
            "未記録 (null) が一致扱いになった"
        );
    }

    /// task 230: `pf_mixed_keep_dev` は「デバイスだけが正」を立て、
    /// `pf_mixed_mark` / `pf_mixed_mark_clear` がそれを落とすこと。
    #[test]
    fn pf230_keep_dev_sets_dev_only() {
        let a = [0.0f32; 4];
        // 記録が無ければ dev-only ではない。
        pf_mixed_mark_clear();
        assert!(!pf_mixed_dev_only(a.as_ptr(), 2, 2));
        // keep_dev → dev-only。形が違えば一致しない。
        pf_mixed_keep_dev(a.as_ptr(), 2, 2);
        assert!(pf_mixed_dev_only(a.as_ptr(), 2, 2));
        assert!(!pf_mixed_dev_only(a.as_ptr(), 1, 2), "nt 違いが一致した");
        assert!(!pf_mixed_dev_only(a.as_ptr(), 2, 4), "e 違いが一致した");
        // 同一マークを打つと dev-only が落ちる (ホストが正に戻る)。
        pf_mixed_mark(a.as_ptr(), 2, 2);
        assert!(!pf_mixed_dev_only(a.as_ptr(), 2, 2));
        // clear でも落ちる。
        pf_mixed_keep_dev(a.as_ptr(), 2, 2);
        pf_mixed_mark_clear();
        assert!(!pf_mixed_dev_only(a.as_ptr(), 2, 2));
    }

    /// task 230/317: ゲートの解釈。v3 は task 244、MIXKEEP は task 317 で既定 on。
    #[test]
    fn pf230_gate_values() {
        // v3: 未指定は on (task 244 t244a で採択)、明示的な偽値で off。
        assert!(pf_attn_v3_from_value(None), "v3 の既定が on でない");
        for v in [Some("0"), Some("off"), Some("false"), Some("no"), Some("")] {
            assert!(!pf_attn_v3_from_value(v), "v3 が off でない: {v:?}");
        }
        for v in [Some("1"), Some("on"), Some("TRUE"), Some(" yes ")] {
            assert!(pf_attn_v3_from_value(v), "v3 が on でない: {v:?}");
        }
        // mixkeep: task 317 で既定 on (task 230/232 では off だった)。
        for v in [Some("0"), Some("off"), Some("false"), Some("no"), Some("")] {
            assert!(!pf_mix_keep_from_value(v), "mixkeep が off でない: {v:?}");
        }
        assert!(pf_mix_keep_from_value(None), "mixkeep の既定が on でない");
        for v in [Some("1"), Some("on"), Some("TRUE"), Some(" yes ")] {
            assert!(pf_mix_keep_from_value(v), "mixkeep が on でない: {v:?}");
        }
    }

    /// task 317: `DENNO_PF_MIXKEEP_SSM` は既定 on (明示的な偽値だけ off)。
    #[test]
    fn pf317_mixkeep_ssm_gate_values() {
        for v in [Some("0"), Some("off"), Some(""), Some("no")] {
            assert!(!pf_mix_keep_ssm_from_value(v), "ssm mixkeep が off でない: {v:?}");
        }
        assert!(pf_mix_keep_ssm_from_value(None), "ssm mixkeep の既定が on でない");
        for v in [Some("1"), Some("on"), Some("TRUE"), Some(" yes ")] {
            assert!(pf_mix_keep_ssm_from_value(v), "ssm mixkeep が on でない: {v:?}");
        }
        // 強制フラグは env より優先する (既定 = env)。
        pf_mix_keep_ssm_set_for_test(2);
        assert!(pf_mix_keep_ssm_on(), "強制 on が効いていない");
        pf_mix_keep_ssm_set_for_test(1);
        assert!(!pf_mix_keep_ssm_on(), "強制 off が効いていない");
        pf_mix_keep_ssm_set_for_test(0);
        crate::cuda::pf_reset_for_test();
    }

    /// task 317: 記録が合えばデバイスの `mixed` を指し、合わなければ `None`。
    /// `DENNO_PF_MIXDEVIN` (既定 on) が off なら常に `None`。
    #[test]
    fn pf317_dev_in_ptr_needs_matching_mark() {
        let a = [0.0f32; 8];
        let b = [0.0f32; 8];
        pf_dev_in_set_for_test(2);
        pf_mixed_keep_dev(a.as_ptr(), 2, 4);
        // 作業域が無いのでポインタは返らないが、記録の一致判定はここで効く。
        assert!(pf_mixed_dev_only(a.as_ptr(), 2, 4));
        assert!(!pf_mixed_dev_only(b.as_ptr(), 2, 4), "別ポインタが一致した");
        assert!(!pf_mixed_dev_only(a.as_ptr(), 1, 4), "nt 違いが一致した");
        pf_dev_in_set_for_test(1);
        assert!(pf_mixed_dev_in_ptr(a.as_ptr(), 2, 4).is_none(), "devin off で返った");
        pf_dev_in_set_for_test(0);
        pf_mixed_mark_clear();
        crate::cuda::pf_reset_for_test();
    }
}


/// P58-3: prefill のリクエスト境界で arena を切り直す。
///
/// 境界では一時領域の中身が全て死んでいる (残差は直後に H2D で入れ直す) ので、
/// **古いチャンクを解放してから**新しいサイズを確保する。個別確保のままだと
/// 「旧領域を保持したまま新領域を要求」してピークが 2 倍になり、779 MiB が
/// 315 MiB の空きに載らない。容量が今回の推定と同程度なら解放せずに
/// オフセットを 0 に戻すだけ (チャーンを作らない)。
///
/// 他スレッドの予約が残っているときは何もしない (並行リクエストのビューを
/// 壊さない)。数値には触らない。
pub fn pf_scratch_begin(rows: usize) {
    if !arena_enabled() {
        return;
    }
    let tid = arena_tid();
    let arena = match ARENA.get() {
        Some(a) => a,
        None => return,
    };
    let mut g = match arena.lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    if g.plan.slots.keys().any(|(t, _)| *t != tid) {
        return;
    }
    // 先にビューを落としてから解放する (dangling を作らない)。
    drop_arena_views();
    let prev_rows = g.last_rows;
    let prev_high = g.plan.high_water;
    let target = ArenaPlan::target_for(prev_rows, prev_high, rows);
    let cap = g.plan.capacity();
    let keep = cap > 0 && target <= cap && target.saturating_mul(2) >= cap;
    let (freed, alloc) = if keep {
        g.plan.reset_keep_chunks();
        (0usize, 0usize)
    } else {
        // 解放の前に GPU を止める (前のリクエストの非同期コピーが残っていても
        // 解放済み領域を読まない)。境界の 1 回だけ・所要は µs 級。
        if let Some(ctx) = shared_ctx() {
            if ctx.make_current().is_ok() {
                let _ = ctx.sync();
            }
        }
        let freed = g.bufs.len();
        g.bufs.clear(); // 古い方を先に解放する (cuMemFree)
        g.plan.reset();
        let alloc = if target > 0 {
            match DevBuffer::alloc_tagged(target, crate::mem_audit::TAG_ARENA) {
                Ok(b) => {
                    g.bufs.push(b);
                    g.plan.add_chunk(target);
                    g.grows += 1;
                    target
                }
                Err(e) => {
                    if arena_dbg() {
                        eprintln!("scratch arena: 事前確保 {target} B に失敗: {e}");
                    }
                    0
                }
            }
        } else {
            0
        };
        (freed, alloc)
    };
    g.last_rows = rows;
    if arena_dbg() {
        eprintln!(
            "scratch arena: 境界 rows={rows} 解放 {freed} 本 / 事前確保 {alloc} B (推定 {target} B)"
        );
    }
}


/// task 149: リクエスト終了時に prefill の arena チャンクを**解放する**。
///
/// prefill の一時領域はリクエストを跨いで内容を保持しない (`pf_scratch_begin` の
/// 契約と同じ) ので、decode の間に何 GiB も抱えたままにしない。次に prefill が
/// 走るときは `pf_scratch_begin` / `arena_alloc` が同じ手順で確保し直すため、
/// **ピークは変わらない** (従来も境界で「解放してから確保」していた)。
///
/// 他スレッドの予約が残っているときは何もしない (並行リクエストのビューを壊さない)。
/// 数値には触らない。
pub fn pf_scratch_release() -> usize {
    if !arena_enabled() || !scratch_free_enabled() {
        return 0;
    }
    let tid = arena_tid();
    let Some(arena) = ARENA.get() else {
        return 0;
    };
    let Ok(mut g) = arena.lock() else {
        return 0;
    };
    if g.plan.slots.keys().any(|(t, _)| *t != tid) {
        return 0;
    }
    if g.bufs.is_empty() {
        return 0;
    }
    // ビューを先に落としてから解放する (dangling を作らない)。非同期コピーが
    // 残っていても解放済み領域を読まないよう、境界の 1 回だけ GPU を止める。
    drop_arena_views();
    if let Some(ctx) = shared_ctx() {
        if ctx.make_current().is_ok() {
            let _ = ctx.sync();
        }
    }
    let n = g.bufs.len();
    let bytes = g.plan.capacity();
    g.bufs.clear(); // ここで cuMemFree
    g.plan.reset();
    if arena_dbg() {
        eprintln!("scratch arena: リクエスト終了で {n} 本 / {bytes} B を解放");
    }
    if crate::mem_audit::audit_on() {
        let free = dev_mem_mb().map(|(f, _)| f).unwrap_or(0);
        tracing::info!(
            "[mem-audit] prefill arena を {n} 本 {} MiB 解放 (dev free {} MiB)",
            bytes >> 20,
            free
        );
    }
    bytes
}




// ---------------------------------------------------------------------------
// task 154: prefill のデバイス常駐 mix 連鎖 (`DENNO_PF_RESIDENT`)
// ---------------------------------------------------------------------------
//
// 狙い: hc mix 1 回ごとの往復 (res H2D → down GEMM → lo D2H → lo H2D → up GEMM →
// gate D2H → ホストで norm/silu/gating) を、デバイス上の 1 連鎖に置き換える。
// 入力 `res` はチャンク開始時に 1 回だけ上げ、以後は層をまたいでデバイスに置く
// (`pf_upload_res` / `pf_download_res` は呼び出し側が層の境界でだけ使う)。
//
// 使うカーネル:
//   dno_pf_rmsgroup / dno_pf_lo / dno_pf_mixed / dno_pf_inj / dno_pf_combine
//   (kernels_pf.cu、PTX は `bash build_ptx.sh pf`) と、既存の
//   dno_gemm_bf16_mt (k=hc_dim→low_rank と low_rank→hc_dim と hc_dim→hc)。
//
// 数値: 要素演算は CPU 参照と 1 演算 1 丸めで一致するように書いてある
// (二乗和は i 昇順の逐次)。GEMM は task 145 のバッチ mix と同じ
// `dno_gemm_bf16_mt` なので、`DENNO_MIX_BATCH_GPU=1` のときの値と一致する。

pub(crate) struct PfKernels {
    #[allow(dead_code)]
    module: CUmodule,
    pub(crate) rmsgroup: CUfunction,
    pub(crate) lo: CUfunction,
    pub(crate) mixed: CUfunction,
    pub(crate) inj: CUfunction,
    combine: CUfunction,
}


// 生ハンドル (CudaCtx / MoeBatchKernels と同じ規約)。
unsafe impl Send for PfKernels {}

unsafe impl Sync for PfKernels {}


/// カーネル一式を読む (初回だけ。失敗は None を焼き付ける = 二度と試さない)。
pub(crate) fn pf_kernels() -> Option<&'static PfKernels> {
    static K: OnceLock<Option<PfKernels>> = OnceLock::new();
    K.get_or_init(|| {
        let mut image = pf_ptx_image()?;
        image.push(0);
        let lib = lib().ok()?;
        let mut module: CUmodule = std::ptr::null_mut();
        unsafe {
            if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void) != CUDA_SUCCESS {
                tracing::info!("[pf] cuModuleLoadData 失敗 → CPU 経路");
                return None;
            }
        }
        let mut fns: [CUfunction; 5] = [std::ptr::null_mut(); 5];
        for (i, name) in PF_NAMES.iter().enumerate() {
            unsafe {
                if (lib.module_get_function)(&mut fns[i], module, name.as_ptr() as *const c_char)
                    != CUDA_SUCCESS
                {
                    tracing::info!(
                        "[pf] PTX に {} が無い → CPU 経路",
                        String::from_utf8_lossy(&name[..name.len() - 1])
                    );
                    return None;
                }
            }
        }
        Some(PfKernels {
            module,
            rmsgroup: fns[0],
            lo: fns[1],
            mixed: fns[2],
            inj: fns[3],
            combine: fns[4],
        })
    })
    .as_ref()
}


/// task 189: null ストリーム (mix / combine の起動先) の完了だけを待つ。
///
/// `cuCtxSynchronize` は**コンテキスト内の全ストリーム**を待つので、prefill の
/// expert upload 用の非ブロッキングストリーム (最大 14 s ぶん enqueue される) の
/// 完了も待ってしまう。イベントを null ストリームに打って `cuEventSynchronize`
/// すれば、待つのは自ストリームの仕事だけになる (legacy default stream の暗黙
/// 同期は**ブロッキング**ストリームだけが対象で、upload は
/// `CU_STREAM_NON_BLOCKING` なので係らない)。
pub(crate) fn pf_sync_own_stream() -> Result<()> {
    let l = lib()?;
    let (Some(create), Some(rec), Some(es)) = (l.event_create, l.event_record, l.event_sync) else {
        bail!("cuEventCreate / cuEventRecord / cuEventSynchronize が無い");
    };
    let ev = PF_SYNC_EV.with(|c| {
        let mut ev = c.get();
        if ev.is_null() {
            let mut e: CUevent = std::ptr::null_mut();
            // CU_EVENT_DISABLE_TIMING = 0x02
            if unsafe { create(&mut e, 0x02) } != CUDA_SUCCESS || e.is_null() {
                return std::ptr::null_mut();
            }
            c.set(e);
            ev = e;
        }
        ev
    });
    if ev.is_null() {
        bail!("同期用イベントを作れない");
    }
    unsafe {
        check(rec(ev, std::ptr::null_mut()), "cuEventRecord (pf stage)")?;
        check(es(ev), "cuEventSynchronize (pf stage)")?;
    }
    Ok(())
}


/// 形状と作業域。層をまたいで再利用する (grow-only)。
pub(crate) struct PfDev {
    pub(crate) res: Option<DevBuffer>,   // [nt][hc][e]
    pub(crate) xn: Option<DevBuffer>,    // [nt][hc][e]
    pub(crate) lo: Option<DevBuffer>,    // [nt][low_rank]
    pub(crate) gate: Option<DevBuffer>,  // [nt][hc][e]
    pub(crate) mixed: Option<DevBuffer>, // [nt][e]
    pub(crate) inj: Option<DevBuffer>,   // [nt][hc]
    bo: Option<DevBuffer>,    // [nt][e]
    pub(crate) gamma: Option<DevBuffer>, // [hc][e]
    /// task 227: 層優先のとき、チャンクの残差を層のあいだ退避しておく置き場
    /// (D2D で出し入れする。ホスト経由より 1 桁速い)。
    park: Vec<DevBuffer>,
    pub(crate) nt: usize,
    pub(crate) hc: usize,
    pub(crate) e: usize,
    pub(crate) lr: usize,
}


impl PfDev {
    pub(crate) const fn new() -> Self {
        Self {
            res: None,
            xn: None,
            lo: None,
            gate: None,
            mixed: None,
            inj: None,
            bo: None,
            gamma: None,
            park: Vec::new(),
            nt: 0,
            hc: 0,
            e: 0,
            lr: 0,
        }
    }
}


/// task 154: チャンクぶんの作業域を用意する。`false` で CPU 経路へ。
///
/// 直前の `res` の中身は保証しない (呼び出し側が `pf_upload_res` で入れる)。
pub fn pf_begin(nt: usize, hc: usize, e: usize, lr: usize) -> bool {
    // task 209: 前のリクエストの「mixed はデバイスと同一」の記録を捨てる。
    pf_mixed_mark_clear();
    if PF_BROKEN.load(std::sync::atomic::Ordering::Relaxed) {
        return false;
    }
    if nt == 0 || hc == 0 || e == 0 || lr == 0 || hc * e > (1 << 20) {
        return false;
    }
    if pf_kernels().is_none() {
        return false;
    }
    let Some(ctx) = shared_ctx() else {
        return pf_fail("CUDA context が無い");
    };
    if ctx.make_current().is_err() {
        return pf_fail("cuCtxSetCurrent 失敗");
    }
    if gemm_bf16_mt_fn(ctx).is_none() && gemm_f32_mt_fn(ctx).is_none() {
        return pf_fail("GEMM カーネルが PTX に無い");
    }
    let r = PF_DEV.with(|cell| -> anyhow::Result<()> {
        let mut d = cell.borrow_mut();
        if d.nt != nt || d.hc != hc || d.e != e || d.lr != lr {
            // 形が変わったら全部作り直す (古い領域を流用しない)。
            d.res = None;
            d.bo = None;
            d.mixed = None;
            d.inj = None;
            d.nt = nt;
            d.hc = hc;
            d.e = e;
            d.lr = lr;
        }
        let hc_dim = hc * e;
        // task 257: プールがあれば区画 (prefill 中に cuMemAlloc を出さない)。
        ensure_region(&mut d.res, crate::model::pf_pool::R::DevRes, 0, nt * hc_dim * 4)?;
        ensure_region(&mut d.xn, crate::model::pf_pool::R::DevXn, 0, nt * hc_dim * 4)?;
        ensure_region(&mut d.lo, crate::model::pf_pool::R::DevLo, 0, nt * lr * 4)?;
        ensure_region(&mut d.gate, crate::model::pf_pool::R::DevGate, 0, nt * hc_dim * 4)?;
        ensure_region(&mut d.mixed, crate::model::pf_pool::R::DevMixed, 0, nt * e * 4)?;
        ensure_region(&mut d.inj, crate::model::pf_pool::R::DevInj, 0, nt * hc * 4)?;
        ensure_region(&mut d.bo, crate::model::pf_pool::R::DevBo, 0, nt * e * 4)?;
        ensure_region(&mut d.gamma, crate::model::pf_pool::R::DevGamma, 0, hc_dim * 4)?;
        Ok(())
    });
    match r {
        Ok(()) => true,
        Err(e) => pf_fail(&format!("作業域の確保に失敗: {e:#}")),
    }
}


/// task 227: 層優先 prefill 用の残差の退避域を用意する (`rows` 行 × hc × e)。
///
/// 層優先では層ループの外側にチャンクを回すので、層 L のあいだ全チャンクの
/// 残差を同時に持つ必要がある。`PF_DEV.res` は 1 チャンクぶんしかないため、
/// チャンクごとにここへ **D2D で** 退避する (ホスト経由は 1 桁遅い)。
pub fn pf_res_park_ensure(slots: usize, rows: usize, hc: usize, e: usize) -> bool {
    let want = rows.saturating_mul(hc).saturating_mul(e).saturating_mul(4);
    if slots == 0 || want == 0 {
        return false;
    }
    PF_DEV.with(|cell| -> bool {
        let mut d = cell.borrow_mut();
        while d.park.len() < slots {
            // task 257: プールがあればその区画 (D2D の退避域も prefill 中に取らない)。
            let i = d.park.len();
            let mut slot: Option<DevBuffer> = None;
            match ensure_region(&mut slot, crate::model::pf_pool::R::DevPark, i, want) {
                Ok(()) => d.park.push(slot.expect("ensure_region が入れる")),
                Err(e) => {
                    tracing::warn!("pf_res_park_ensure: 確保に失敗: {e:#}");
                    return false;
                }
            }
            // 既にある枚が小さいときは作り直す (grow-only)。
            let n = d.park.len();
            if d.park[n - 1].bytes() < want {
                d.park.pop();
                return false;
            }
        }
        for b in d.park.iter() {
            if b.bytes() < want {
                return false;
            }
        }
        true
    })
}


/// 残差を退避域 `i` へ D2D で写す (`rows` 行ぶん)。
pub fn pf_res_park_save(i: usize, rows: usize, hc: usize, e: usize) -> bool {
    // task 318: `other` の中の D2D を分けて数える。
    let _sub = crate::model::q4ledger::sub(crate::model::q4ledger::SUB_RES_PARK);
    let bytes = rows.saturating_mul(hc).saturating_mul(e).saturating_mul(4);
    PF_DEV.with(|cell| -> bool {
        let d = cell.borrow();
        let (Some(res), Some(dst)) = (d.res.as_ref(), d.park.get(i)) else {
            return false;
        };
        if bytes == 0 || res.bytes() < bytes || dst.bytes() < bytes {
            return false;
        }
        match lib() {
            Ok(l) => unsafe {
                check(
                    (l.memcpy_dtod_async)(dst.ptr(), res.ptr(), bytes, std::ptr::null_mut()),
                    "cuMemcpyDtoDAsync (res park save)",
                )
                .is_ok()
            },
            Err(_) => false,
        }
    })
}


/// 退避域 `i` から残差へ D2D で戻す (`rows` 行ぶん)。
pub fn pf_res_park_load(i: usize, rows: usize, hc: usize, e: usize) -> bool {
    // task 318: `other` の中の D2D を分けて数える。
    let _sub = crate::model::q4ledger::sub(crate::model::q4ledger::SUB_RES_PARK);
    let bytes = rows.saturating_mul(hc).saturating_mul(e).saturating_mul(4);
    PF_DEV.with(|cell| -> bool {
        let d = cell.borrow();
        let (Some(res), Some(src)) = (d.res.as_ref(), d.park.get(i)) else {
            return false;
        };
        if bytes == 0 || res.bytes() < bytes || src.bytes() < bytes {
            return false;
        }
        match lib() {
            Ok(l) => unsafe {
                check(
                    (l.memcpy_dtod_async)(res.ptr(), src.ptr(), bytes, std::ptr::null_mut()),
                    "cuMemcpyDtoDAsync (res park load)",
                )
                .is_ok()
            },
            Err(_) => false,
        }
    })
}


/// task 279: プールが追い出した expert の鍵を取り出す (取り出したら空)。
#[cfg(feature = "cuda")]
pub fn pf_pool_take_evicted() -> Vec<crate::model::expert_cache::ExpertKey> {
    match PF_POOL_EVICTED.lock() {
        Ok(mut g) => std::mem::take(&mut *g),
        Err(_) => Vec::new(),
    }
}


/// task 265: 借りたプールの返却責任 (drop で expert に返す)。
///
/// prefill の入口で受け取り、関数の終端 (早期撤退も含む) で落とす。借りられな
/// かったときは `owned == false` で、drop は何もしない。
#[cfg(feature = "cuda")]
pub struct PfPoolGuard {
    owned: bool,
}


#[cfg(feature = "cuda")]
impl PfPoolGuard {
    /// この prefill が実際にプールを借りたか。
    pub fn is_held(&self) -> bool {
        self.owned
    }
}


#[cfg(feature = "cuda")]
impl Drop for PfPoolGuard {
    fn drop(&mut self) {
        if self.owned {
            pf_pool_release();
        }
        // task 297: prefill の終端で MoE の作業域も OS へ返す。使い回しが効くのは
        // この prefill の中 (48 層) だけなので、decode のあいだ抱えない。
        crate::model::qwen4exp::moe_scratch_release();
    }
}


/// task 265: いま借りているプールのバイト数 (0 = 借りていない)。
#[cfg(feature = "cuda")]
pub fn pf_pool_len() -> usize {
    PF_POOL
        .lock()
        .ok()
        .and_then(|g| g.as_ref().map(|b| b.bytes()))
        .unwrap_or(0)
}


/// task 265: 長い prefill の入口で作業域を expert から借りる。
///
/// - `n.nt_all` が `DENNO_PF_POOL_MIN_TOK` 未満 → 借りない (短い経路を変えない)。
/// - レイアウトが組めない / `DENNO_PF_POOL_MB` を超える → 借りない。
/// - 確保に失敗 → 借りない (呼び出し側は従来の `alloc_tagged`)。
///
/// 成功したら区画を組んで `owned == true` のガードを返す。同じスレッドで
/// 二重に呼ばれたときは既存の区画を組み直すだけ (返すのは最初のガード)。
#[cfg(feature = "cuda")]
pub fn pf_pool_acquire(n: &crate::model::pf_pool::Need) -> PfPoolGuard {
    pf_pool_acquire_capped(
        n,
        crate::model::pf_pool::pf_pool_bytes(),
        crate::model::pf_pool::pf_pool_reserve_bytes(),
    )
}


/// task 279: 上限を指定して借りる (大きい計画は `DENNO_PF_POOL_BIG_MB` を使う)。
///
/// 上限は `cap` だけで決まり、レイアウトは `n` (チャンク / タイル) から組む。
/// 呼び出し側が選んだ大きい形がここで上限を超える / 確保できないときは
/// `owned == false` を返し、呼び出し側は従来のチャンク・タイルへ戻す。
///
/// task 290: 借りたあとに `reserve` バイトを残す。足りない分は**先に**チャンク
/// 単位で expert を追い出しておく (確保の失敗経路に任せると、借りた直後の空きが
/// 100-200 MiB になり、prefill の後半の数 MiB の確保が断片化で落ちる)。追い出した
/// 鍵は [`PF_POOL_EVICTED`] に積み、呼び出し側が prefill の終端で LRU へ戻す。
#[cfg(feature = "cuda")]
pub fn pf_pool_acquire_capped(
    n: &crate::model::pf_pool::Need,
    cap: usize,
    reserve: usize,
) -> PfPoolGuard {
    let not_held = PfPoolGuard { owned: false };
    // ゲート / しきい値 / 上限は純関数で決める (既定では何も借りない =
    // 従来の `alloc_tagged` と 1 バイトも変わらない)。
    let Some(l) = crate::model::pf_pool::borrow_plan(
        n,
        crate::model::pf_pool::pf_pool_on(),
        crate::model::pf_pool::pf_pool_min_tok(),
        cap,
    ) else {
        return not_held;
    };
    if PF_POOL_OWNED.with(|c| c.get()) {
        // 同じリクエストの中で 2 回目 (層優先 → チャンク優先の落ちなど)。
        PF_POOL_LAYOUT.with(|c| *c.borrow_mut() = Some(l));
        return not_held;
    }
    // task 279: 前のリクエストの鍵を持ち越さない (借り直すときに必ず空から)。
    if let Ok(mut g) = PF_POOL_EVICTED.lock() {
        g.clear();
    }
    let Some(ctx) = shared_ctx() else {
        return not_held;
    };
    if ctx.make_current().is_err() {
        return not_held;
    }
    // 借りる判断と確保を 1 つのロックで行う (別スレッドが借りている最中に
    // 上書きすると、そのビューが dangling になる)。
    let mut slot = match PF_POOL.lock() {
        Ok(g) => g,
        Err(_) => return not_held,
    };
    if slot.is_some() {
        return not_held;
    }
    let free_before = dev_mem_mb().map(|(f, _)| f).map(|f| f << 20);
    // task 279: この確保が追い出した expert の鍵を控える (借りる前に掃除する)。
    let _ = take_alloc_evicted();
    // task 290: 借りたあとに `reserve` を残すため、足りない分を**先に**空ける。
    // ここで空けた鍵も `take_alloc_evicted` が拾うので、終端で LRU へ戻る。
    if let Some(free) = free_before {
        let keep_free = l.total.saturating_add(reserve);
        if keep_free > free {
            let (rep, keys) = crate::model::expert_vram::emergency_free_chunks_keys(keep_free);
            if !rep.is_empty() {
                let n = keys.len();
                note_alloc_evicted(keys);
                eprintln!(
                    "[pf-pool] 余白 {} MiB を残すため先に空けた (空き {} MiB → expert {} 本 / {})",
                    reserve >> 20,
                    free >> 20,
                    n,
                    rep.line()
                );
            }
        }
    }
    match DevBuffer::alloc_tagged(l.total, crate::mem_audit::TAG_OTHER) {
        Ok(b) => {
            let bytes = b.bytes();
            *slot = Some(b);
            drop(slot);
            let evicted = take_alloc_evicted();
            let evicted_n = evicted.len();
            if let Ok(mut g) = PF_POOL_EVICTED.lock() {
                *g = evicted;
            }
            let free_after = dev_mem_mb().map(|(f, _)| f).map(|f| f << 20);
            PF_POOL_OWNED.with(|c| c.set(true));
            PF_POOL_LAYOUT.with(|c| *c.borrow_mut() = Some(l));
            eprintln!(
                "[pf-pool] expert から作業域 {} MiB を借りた (タイル {} / {} トークン / 空き {} → {} MiB / 追い出し {} 本 / 余白 {} MiB)",
                bytes >> 20,
                n.tile,
                n.nt_all,
                free_before.map(|b| b >> 20).unwrap_or(0),
                free_after.map(|b| b >> 20).unwrap_or(0),
                evicted_n,
                reserve >> 20
            );
            PfPoolGuard { owned: true }
        }
        Err(e) => {
            drop(slot);
            // task 290: 先に空けた expert は、借りられなかったので呼び出し側が
            // 終端で LRU へ戻せるように残す (PF_POOL_EVICTED は借用の前に空にしてある)。
            let keys = take_alloc_evicted();
            if !keys.is_empty() {
                if let Ok(mut g) = PF_POOL_EVICTED.lock() {
                    g.extend(keys);
                }
            }
            tracing::warn!("pf_pool: 作業域 {} MiB を借りられない: {e:#}", l.total >> 20);
            eprintln!(
                "[pf-pool] 作業域 {} MiB を借りられない → 従来の確保",
                l.total >> 20
            );
            not_held
        }
    }
}


/// task 265: 借りたプールを expert に返す (cuMemFree)。
///
/// 先に 3 つの置き場 (活性 / スクラッチ / PfDev) を落とす。残ったビューは
/// 解放後に dangling になるため、順序はここで固定する。
#[cfg(feature = "cuda")]
pub fn pf_pool_release() {
    pf_lm_release();
    pf_scratch_drop();
    pf_release();
    PF_POOL_LAYOUT.with(|c| *c.borrow_mut() = None);
    let buf = match PF_POOL.lock() {
        Ok(mut g) => g.take(),
        Err(_) => None,
    };
    PF_POOL_OWNED.with(|c| c.set(false));
    if let Some(b) = buf {
        let bytes = b.bytes();
        drop(b);
        let free = dev_mem_mb().map(|(f, _)| f).unwrap_or(0);
        eprintln!(
            "[pf-pool] 作業域 {} MiB を expert に返した (空き {} MiB)",
            bytes >> 20,
            free
        );
    }
}


// ---------------------------------------------------------------------------
// task 233: 層優先 prefill の「全トークンの活性」をデバイスに置いたままにする
// ---------------------------------------------------------------------------
//
// task 227 の層優先は、層ごとに全トークンの活性 (14k で 143 MB) を router /
// shared / expert の 3 段で上げ直し、shared と expert の出力 (同じく 143 MB) を
// 毎層ホストへ戻していた (14k で 34 GB)。ここはその往復を消すための置き場。
//
// - `mixed`: `pf_mix` (ffn 入口) の出力 `[nt_all][e]`。層のあいだ保持する。
// - `ffn`: MoE の出力 `[nt_all][e]`。combine がここから直接読む。
// - `inj`: チャンクごとの inject `[rows][hc]`。combine が読む
//   (task 227 は最後のチャンクの inject を全チャンクに使っていた)。
//
// どれも 1 リクエストで 1 回だけ確保し、prefill の終端で [`pf_lm_release`] が返す。

/// 層優先 prefill の活性の置き場 (スレッドローカル)。
#[cfg(feature = "cuda")]
#[derive(Default)]
pub(crate) struct PfLmDev {
    pub(crate) mixed: Option<DevBuffer>,
    ffn: Option<DevBuffer>,
    inj: Vec<DevBuffer>,
    nt: usize,
    e: usize,
    hc: usize,
    chunk: usize,
}


/// task 233: 層優先 prefill の置き場を用意する。
///
/// `nt` はプロンプト全体のトークン数、`chunk` は 1 回の mix のトークン数。
/// 形が変わったら作り直す。確保できなければ false (呼び出し側はチャンク優先へ)。
#[cfg(feature = "cuda")]
pub fn pf_lm_ensure(nt: usize, e: usize, hc: usize, chunk: usize) -> bool {
    if nt == 0 || e == 0 || hc == 0 || chunk == 0 {
        return false;
    }
    let slots = nt.div_ceil(chunk);
    PF_LM.with(|cell| -> bool {
        let mut d = cell.borrow_mut();
        if d.nt != nt || d.e != e || d.hc != hc || d.chunk != chunk {
            *d = PfLmDev {
                nt,
                e,
                hc,
                chunk,
                ..PfLmDev::default()
            };
        }
        let r = (|| -> anyhow::Result<()> {
            // task 257: プールがあればその区画 (prefill 中に cuMemAlloc を出さない)。
            ensure_region(&mut d.mixed, crate::model::pf_pool::R::LmMixed, 0, nt * e * 4)?;
            ensure_region(&mut d.ffn, crate::model::pf_pool::R::LmFfn, 0, nt * e * 4)?;
            while d.inj.len() < slots {
                let i = d.inj.len();
                let mut slot: Option<DevBuffer> = None;
                ensure_region(&mut slot, crate::model::pf_pool::R::LmInj, i, chunk * hc * 4)?;
                d.inj.push(slot.expect("ensure_region が入れる"));
            }
            Ok(())
        })();
        match r {
            Ok(()) => true,
            Err(err) => {
                tracing::warn!("pf_lm_ensure: 確保に失敗: {err:#}");
                false
            }
        }
    })
}


/// task 233: MoE の出力 (`[nt][e]` f32) を置く先頭 (先頭行から)。
#[cfg(feature = "cuda")]
pub fn pf_lm_ffn_ptr(nt: usize, e: usize) -> Option<CUdeviceptr> {
    PF_LM.with(|cell| -> Option<CUdeviceptr> {
        let l = cell.borrow();
        let b = l.ffn.as_ref()?;
        if b.bytes() < nt.saturating_mul(e).saturating_mul(4) {
            return None;
        }
        Some(b.ptr())
    })
}


/// task 233: チャンク `i` の inject (`d.inj`) を置き場へ D2D で退避する。
#[cfg(feature = "cuda")]
pub fn pf_lm_inj_save(i: usize, rows: usize, hc: usize) -> bool {
    // task 318: `other` の中の D2D を分けて数える。
    let _sub = crate::model::q4ledger::sub(crate::model::q4ledger::SUB_INJ);
    let bytes = rows.saturating_mul(hc).saturating_mul(4);
    PF_DEV.with(|cell| -> bool {
        let d = cell.borrow();
        let Some(src) = d.inj.as_ref() else {
            return false;
        };
        PF_LM.with(|lc| -> bool {
            let l = lc.borrow();
            let Some(dst) = l.inj.get(i) else {
                return false;
            };
            if bytes == 0 || src.bytes() < bytes || dst.bytes() < bytes {
                return false;
            }
            match lib() {
                Ok(lib) => unsafe {
                    check(
                        (lib.memcpy_dtod_async)(dst.ptr(), src.ptr(), bytes, std::ptr::null_mut()),
                        "cuMemcpyDtoDAsync (lm inj park)",
                    )
                    .is_ok()
                },
                Err(_) => false,
            }
        })
    })
}


/// task 233: 置き場の inject を `d.inj` へ D2D で戻す (combine の直前)。
#[cfg(feature = "cuda")]
pub fn pf_lm_inj_load(i: usize, rows: usize, hc: usize) -> bool {
    // task 318: `other` の中の D2D を分けて数える。
    let _sub = crate::model::q4ledger::sub(crate::model::q4ledger::SUB_INJ);
    let bytes = rows.saturating_mul(hc).saturating_mul(4);
    PF_LM.with(|lc| -> bool {
        let l = lc.borrow();
        let Some(src) = l.inj.get(i) else {
            return false;
        };
        PF_DEV.with(|cell| -> bool {
            let d = cell.borrow();
            let Some(dst) = d.inj.as_ref() else {
                return false;
            };
            if bytes == 0 || src.bytes() < bytes || dst.bytes() < bytes {
                return false;
            }
            match lib() {
                Ok(lib) => unsafe {
                    check(
                        (lib.memcpy_dtod_async)(dst.ptr(), src.ptr(), bytes, std::ptr::null_mut()),
                        "cuMemcpyDtoDAsync (lm inj load)",
                    )
                    .is_ok()
                },
                Err(_) => false,
            }
        })
    })
}


/// task 233: ホストで計算した MoE の出力を置き場へ上げる (デバイス経路を諦めたとき用)。
#[cfg(feature = "cuda")]
pub fn pf_lm_upload_ffn(src: &[f32]) -> bool {
    PF_LM.with(|cell| -> bool {
        let l = cell.borrow();
        match l.ffn.as_ref() {
            Some(b) if b.bytes() >= src.len() * 4 => b.copy_from(src).is_ok(),
            _ => false,
        }
    })
}


/// task 233: 層優先の置き場を返す (prefill の終端で VRAM を decode に戻す)。
#[cfg(feature = "cuda")]
pub fn pf_lm_release() {
    PF_LM.with(|cell| {
        let mut l = cell.borrow_mut();
        *l = PfLmDev::default();
    });
}


/// task 581: 残差の往復のホスト側 — 写しの並列化と内訳。
///
/// # 何を直すか (P565 / P577 の台帳、既定モデル 14k cold)
///
/// `sub other res_up=89.3 ms [h=574.2M/7 d=71.1M/1]` = 645 MB。`DENNO_PF_RES_PIN`
/// (task 565) で pinned バウンスへは通っているが、それでも 7.2 GB/s しか出ない。
/// 1 本は「pageable の memcpy (1 スレッド) → `cuMemcpyHtoD`」の直列で、84 MB を
/// 1 コアで写している。**バイト列は変わらない**ので決定性も md5 も動かない。
///
/// 内訳 (`DENNO_PF_RES_XPROF=1`) は台帳の `[q4ledger-pf] resx` 行に出す。
#[cfg(feature = "cuda")]
pub mod res_xfer {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// 専用の pinned バウンス (残差だけが使う。共有バウンスのロックを取らない)。
    ///
    /// `PinnedHost::alloc` は device 全体を同期させるので、**育つときだけ**呼ぶ
    /// (共有バウンスと同じ規律。3,826 の 39 MiB -> 14k の 143 MiB で 2 回)。
    struct ResPin {
        buf: Option<PinnedHost>,
    }

    fn res_pin() -> &'static std::sync::Mutex<ResPin> {
        static S: std::sync::OnceLock<std::sync::Mutex<ResPin>> = std::sync::OnceLock::new();
        S.get_or_init(|| std::sync::Mutex::new(ResPin { buf: None }))
    }

    /// 写し専用のプール (層ループの rayon を奪わない)。本数は控えめに 8。
    fn copy_pool() -> Option<&'static rayon::ThreadPool> {
        static P: std::sync::OnceLock<Option<rayon::ThreadPool>> = std::sync::OnceLock::new();
        P.get_or_init(|| {
            let n = rayon::current_num_threads().clamp(2, 8);
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .thread_name(|i| format!("pf-res-{i}"))
                .build()
                .ok()
        })
        .as_ref()
    }

    /// 1 タスクの塊 (これ未満は直列のほうが速い)。
    const BLK: usize = 1 << 20;
    /// 並列に分ける下限。
    const MT_MIN: usize = 4 << 20;

    /// `len` バイトを `src` から `dst` へ写す (中身は同じバイト列)。
    ///
    /// `par` が真で 4 MiB 以上なら 1 MiB の塊に分けて並列に写す。行の区切りと
    /// 無関係な**バイト列の写し**なので、値も決定性も変わらない。
    pub(crate) fn copy_bytes(dst: *mut u8, src: *const u8, len: usize, par: bool) {
        if len == 0 {
            return;
        }
        if !par || len < MT_MIN {
            unsafe { std::ptr::copy_nonoverlapping(src, dst, len) };
            return;
        }
        // 生ポインタは usize で閉包へ渡す (塊は互いに重ならない)。
        let (dst_a, src_a) = (dst as usize, src as usize);
        let nblk = (len + BLK - 1) / BLK;
        let run = || {
            use rayon::prelude::*;
            (0..nblk).into_par_iter().for_each(|i| {
                let off = i * BLK;
                let n = BLK.min(len - off);
                let d = (dst_a + off) as *mut u8;
                let s = (src_a + off) as *const u8;
                unsafe { std::ptr::copy_nonoverlapping(s, d, n) };
            });
        };
        match copy_pool() {
            Some(p) => p.install(run),
            None => run(),
        }
    }

    /// 残差の H2D (pinned 経由)。成功したら `Ok(())`。
    ///
    /// 専用バウンスが取れない (pinned の確保に失敗) ときは `Err` を返し、
    /// 呼び出し側は従来の共有バウンス経路へ落ちる (値は同じ)。
    pub(crate) fn to_dev(dst: CUdeviceptr, data: &[f32], par: bool, xp: bool) -> Result<()> {
        let bytes = std::mem::size_of_val(data);
        if bytes == 0 {
            return Ok(());
        }
        let mut g = res_pin().lock().unwrap_or_else(|e| e.into_inner());
        let cur = g.buf.as_ref().map(|b| b.bytes()).unwrap_or(0);
        if cur < bytes {
            // 新しい 1 本を取る前に古い 1 本を落とす (pinned を二重に持たない)。
            g.buf = None;
            match PinnedHost::alloc(bytes.next_power_of_two().max(bytes)) {
                Some(p) => g.buf = Some(p),
                None => bail!("残差用の pinned バウンスを確保できない ({bytes} B)"),
            }
        }
        let pin = g.buf.as_ref().expect("確保済み").ptr();
        let t = std::time::Instant::now();
        copy_bytes(pin, data.as_ptr() as *const u8, bytes, par);
        let gather_ns = t.elapsed().as_nanos() as u64;
        let t = std::time::Instant::now();
        let lib = lib()?;
        unsafe {
            check(
                (lib.memcpy_htod)(dst, pin as *const c_void, bytes),
                "cuMemcpyHtoD (res pin, mt)",
            )?;
        }
        if xp {
            note_up(gather_ns, t.elapsed().as_nanos() as u64, bytes);
        }
        Ok(())
    }

    /// 残差の D2H (pinned 経由)。成功したら `Ok(())`。
    pub(crate) fn to_host(src: CUdeviceptr, out: &mut [f32], par: bool, xp: bool) -> Result<()> {
        let bytes = std::mem::size_of_val(out);
        if bytes == 0 {
            return Ok(());
        }
        let mut g = res_pin().lock().unwrap_or_else(|e| e.into_inner());
        let cur = g.buf.as_ref().map(|b| b.bytes()).unwrap_or(0);
        if cur < bytes {
            g.buf = None;
            match PinnedHost::alloc(bytes.next_power_of_two().max(bytes)) {
                Some(p) => g.buf = Some(p),
                None => bail!("残差用の pinned バウンスを確保できない ({bytes} B)"),
            }
        }
        let pin = g.buf.as_ref().expect("確保済み").ptr();
        let t = std::time::Instant::now();
        let lib = lib()?;
        unsafe {
            check(
                (lib.memcpy_dtoh)(pin as *mut c_void, src, bytes),
                "cuMemcpyDtoH (res pin, mt)",
            )?;
        }
        let dma_ns = t.elapsed().as_nanos() as u64;
        let t = std::time::Instant::now();
        copy_bytes(
            out.as_mut_ptr() as *mut u8,
            pin as *const u8,
            bytes,
            par,
        );
        if xp {
            note_down(t.elapsed().as_nanos() as u64, dma_ns, bytes);
        }
        Ok(())
    }

    // ---- 内訳 (診断。既定 off では 1 度も触らない) ----
    static UP_GATHER_NS: AtomicU64 = AtomicU64::new(0);
    static UP_DMA_NS: AtomicU64 = AtomicU64::new(0);
    static UP_BYTES: AtomicU64 = AtomicU64::new(0);
    static UP_CALLS: AtomicU64 = AtomicU64::new(0);
    static DOWN_GATHER_NS: AtomicU64 = AtomicU64::new(0);
    static DOWN_DMA_NS: AtomicU64 = AtomicU64::new(0);
    static DOWN_BYTES: AtomicU64 = AtomicU64::new(0);
    static DOWN_CALLS: AtomicU64 = AtomicU64::new(0);

    fn note_up(gather_ns: u64, dma_ns: u64, bytes: usize) {
        UP_GATHER_NS.fetch_add(gather_ns, Ordering::Relaxed);
        UP_DMA_NS.fetch_add(dma_ns, Ordering::Relaxed);
        UP_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
        UP_CALLS.fetch_add(1, Ordering::Relaxed);
    }

    fn note_down(gather_ns: u64, dma_ns: u64, bytes: usize) {
        DOWN_GATHER_NS.fetch_add(gather_ns, Ordering::Relaxed);
        DOWN_DMA_NS.fetch_add(dma_ns, Ordering::Relaxed);
        DOWN_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
        DOWN_CALLS.fetch_add(1, Ordering::Relaxed);
    }

    /// 積んだ内訳を 1 行で返し、0 に戻す (prefill 1 回につき 1 行)。
    pub fn take_line() -> Option<String> {
        let c = UP_CALLS.swap(0, Ordering::Relaxed) + DOWN_CALLS.swap(0, Ordering::Relaxed);
        if c == 0 {
            return None;
        }
        let up_g = UP_GATHER_NS.swap(0, Ordering::Relaxed) as f64 / 1e6;
        let up_d = UP_DMA_NS.swap(0, Ordering::Relaxed) as f64 / 1e6;
        let up_b = UP_BYTES.swap(0, Ordering::Relaxed) as f64 / 1e6;
        let dn_g = DOWN_GATHER_NS.swap(0, Ordering::Relaxed) as f64 / 1e6;
        let dn_d = DOWN_DMA_NS.swap(0, Ordering::Relaxed) as f64 / 1e6;
        let dn_b = DOWN_BYTES.swap(0, Ordering::Relaxed) as f64 / 1e6;
        Some(format!(
            "[q4ledger-pf] resx up gather={up_g:.1} dma={up_d:.1}ms ({up_b:.1}MB) \
             down dma={dn_d:.1} gather={dn_g:.1}ms ({dn_b:.1}MB)"
        ))
    }

    #[cfg(test)]
    pub(crate) mod tests {
        use super::*;

        /// 並列の写しは直列と同じバイト列 (決定性の証拠)。塊の境界も跨ぐ。
        #[test]
        fn task581_copy_bytes_is_identical_to_serial() {
            for len in [0usize, 1, 4095, 1 << 20, (4 << 20) + 12345] {
                let src: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
                let mut a = vec![0u8; len];
                let mut b = vec![0u8; len];
                copy_bytes(a.as_mut_ptr(), src.as_ptr(), len, false);
                copy_bytes(b.as_mut_ptr(), src.as_ptr(), len, true);
                assert_eq!(a, src, "直列の写しが違う (len={len})");
                assert_eq!(b, src, "並列の写しが違う (len={len})");
            }
        }

        /// 内訳の行は「1 回も転送していない」ときは出ない (余計な行を足さない)。
        #[test]
        fn task581_xprof_line_only_when_used() {
            // 他のテストと共有の静的カウンタなので、ここでは take してから見る。
            let _ = take_line();
            assert!(take_line().is_none(), "転送 0 回で行が出た");
        }
    }
}


/// task 154: デバイスの残差をホストへ戻す (`res` は [nt][hc][e])。
///
/// task 300: `DENNO_PF_PLE_REUSE` のときは一時 Vec を作らず `out` へ直接
/// D2H する (`out` は 14k のチャンクで 82 MB あり、`to_vec` + `copy_from_slice`
/// は同じ大きさの確保と写しをもう 1 回ぶん払っていた)。値は同じ。
pub fn pf_download_res(out: &mut [f32]) -> bool {
    // task 318: `other` の中の残差の往復を分けて数える。
    let _sub = crate::model::q4ledger::sub(crate::model::q4ledger::SUB_RES_UP);
    let direct = crate::model::qwen4exp::pf_ple_reuse_on();
    // task 565: `DENNO_PF_RES_PIN` なら全体ゲートと無関係に pinned へ通す
    // (このホストの pageable D2H は 1 GB/s しか出ない)。
    let force = crate::model::exp_pf::pf_res_pin();
    // task 581: 専用バウンス + 並列写し (pinned からの取り出し側)。
    let xp = crate::model::exp_pf::pf_res_xprof();
    let mt = crate::model::exp_pf::pf_res_mt();
    let use_xfer = mt || xp;
    PF_DEV.with(|cell| -> bool {
        let d = cell.borrow();
        match d.res.as_ref() {
            Some(b) if b.bytes() >= out.len() * 4 => {
                if use_xfer {
                    if res_xfer::to_host(b.ptr(), out, mt, xp).is_ok() {
                        return true;
                    }
                }
                if force {
                    if let Ok(true) = b.to_slice_pinned_forced(out) {
                        return true;
                    }
                }
                // task 318: `DENNO_PF_PIN_STAGE` では pinned 経由 (速い)。失敗したら従来へ。
                if let Ok(true) = b.to_slice_pinned(out) {
                    return true;
                }
                if direct {
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
    })
}


/// task 154: 残差の **1 行だけ** をホストへ戻す (層の出口フック用)。
///
/// `t` はチャンク内のトークン番号、`out` は `[hc*e]`。
pub fn pf_download_res_row(t: usize, hc_dim: usize, out: &mut [f32]) -> bool {
    if out.len() != hc_dim {
        return false;
    }
    PF_DEV.with(|cell| -> bool {
        let d = cell.borrow();
        match d.res.as_ref() {
            Some(b) if b.bytes() >= (t + 1) * hc_dim * 4 => {
                let bytes = hc_dim * 4;
                let src = b.ptr() + (t * hc_dim * 4) as u64;
                match lib() {
                    Ok(l) => unsafe {
                        check(
                            (l.memcpy_dtoh)(
                                out.as_mut_ptr() as *mut c_void,
                                src,
                                bytes,
                            ),
                            "cuMemcpyDtoH (res row)",
                        )
                        .is_ok()
                    },
                    Err(_) => false,
                }
            }
            _ => false,
        }
    })
}


/// task 334: mix の norm (`gamma`) のデバイス常駐表。
///
/// 従来は `pf_mix` のたびに `d.gamma` へホストから上げ直していた。14k の層優先
/// prefill は 48 層 × 7 チャンク × 2 mix = 672 回で、1 回 40 KB。**バイト数は
/// 小さいが、`cuMemcpyHtoD` は同じストリームの前のカーネルを待つ同期コピー**
/// なので、待ちを外した木ではこれが新しい直列点になる (推定)。
///
/// 鍵は「ホスト配列の先頭ポインタ + 長さ」。層の重みはロード時に 1 回確保されて
/// プロセスの寿命まで動かないので、同じアドレスに別の norm が来ることはない。
/// 念のため先頭 4 要素も鍵に混ぜる。表が上限に達したら**何もせず従来経路**
/// (d.gamma へ同期コピー) に落ちる — 使用中の DevBuffer を解放しない。
pub(crate) fn pf_norm_dev(norm: &[f32]) -> Option<CUdeviceptr> {
    /// 表の上限 (48 層 × 2 本 = 96 本 + 予備)。1 本 40 KB なので 20 MiB 未満。
    const MAX: usize = 512;
    if !crate::model::exp_pf::mix_gamma_cache_on() || norm.len() < 4 {
        return None;
    }
    static REG: OnceLock<Mutex<Vec<(u64, usize, u32, DevBuffer)>>> = OnceLock::new();
    let head = norm[0].to_bits() ^ norm[1].to_bits().rotate_left(7)
        ^ norm[2].to_bits().rotate_left(13)
        ^ norm[3].to_bits().rotate_left(19);
    let key = (norm.as_ptr() as u64, norm.len(), head);
    let reg = REG.get_or_init(|| Mutex::new(Vec::new()));
    let mut v = reg.lock().ok()?;
    if let Some((_, _, _, b)) = v
        .iter()
        .find(|(p, n, h, _)| *p == key.0 && *n == key.1 && *h == key.2)
    {
        return Some(b.ptr());
    }
    if v.len() >= MAX {
        return None;
    }
    let b = DevBuffer::from_slice(norm).ok()?;
    let p = b.ptr();
    v.push((key.0, key.1, key.2, b));
    Some(p)
}


/// 常駐した重み 1 本の種類 (どの GEMM カーネルで叩くか)。
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PfWKind {
    /// `dno_gemm_bf16_mt` (BF16 のまま)。
    Bf16,
    /// `dno_gemm_f32_mt` (f32 に展開済み / 元から F32)。
    F32,
    /// 量子型 (`f_gemm_q` の添字, 1 行の生バイト数, 量子ブロック長) と **型そのもの**。
    ///
    /// 型を持つのはテンソルコア GEMM の選択に要るため (task 175)。v1 カーネル
    /// (`dno_gemm_tc_q8_0_w`) は Q8_0 専用で、Q4_K / Q5_1 の重みを渡すと別の型
    /// として読んで出力が壊れる (lab t175a で実際に踏んだ: shared expert の
    /// gate は Q4_K)。
    Quant(crate::kquant::GgmlType, usize, usize, usize),
}


/// 重み 1 本を常駐させる。
///
/// - BF16: [`mix_gpu_weight`] (task 152。`want_bf16` が偽なら f32 に展開)
/// - F32:  そのままのバイト列 (40 KB 級の inject 用)
/// - Q8_0 / Q5_1 / Q4K / Q5K: [`weights_for_gguf`] (量子のまま)
/// - それ以外: `None` (呼び出し側は層を進める前に従来経路へ)
pub(crate) fn pf_weight(l: PfLin<'_>, want_bf16: bool) -> Option<(CUdeviceptr, PfWKind)> {
    use crate::kquant::GgmlType as T;
    match l.ty {
        T::Bf16 => {
            if l.blocks.len() != l.k.saturating_mul(l.n).saturating_mul(2) {
                return None;
            }
            match mix_gpu_weight(l.uid, l.blocks, l.k, l.n, want_bf16) {
                Ok(w) => Some((
                    w.ptr,
                    if want_bf16 { PfWKind::Bf16 } else { PfWKind::F32 },
                )),
                Err(e) => {
                    tracing::info!("[pf] 重みの常駐に失敗 (uid={}): {e:#}", l.uid);
                    None
                }
            }
        }
        T::F32 => {
            let want = l.k.checked_mul(l.n)?.checked_mul(4)?;
            if l.blocks.len() != want {
                return None;
            }
            // task 617: inject の実形 (n = hc <= 8) は **重みだけ BF16 に丸めて**
            // `dno_mix_inj2_bf16` (k 分割・1 ブロック 1 トークン) へ回す。
            // f32 のまま汎用 GEMM に渡すと 16 行タイルに 4 行しか乗らない。
            if want_bf16
                && pf_inj_f32_bf16_on()
                && mix_inj2_on()
                && pf_inj_bf16_shape(l.k, l.n)
            {
                if let Some(bits) = f32_bytes_to_bf16_bytes(l.blocks) {
                    let reg = PF_INJ_BF_W.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
                    let mut m = match reg.lock() {
                        Ok(g) => g,
                        Err(_) => return None,
                    };
                    if let Some(b) = m.get(&l.uid) {
                        return Some((b.ptr(), PfWKind::Bf16));
                    }
                    if let Ok(b) = DevBuffer::from_slice_tagged(&bits, crate::mem_audit::TAG_OTHER) {
                        let p = b.ptr();
                        m.insert(l.uid, b);
                        return Some((p, PfWKind::Bf16));
                    }
                }
            }
            let reg = PF_F32_W.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
            let mut m = match reg.lock() {
                Ok(g) => g,
                Err(_) => return None,
            };
            if let Some(b) = m.get(&l.uid) {
                return Some((b.ptr(), PfWKind::F32));
            }
            let b = DevBuffer::from_slice_tagged(l.blocks, crate::mem_audit::TAG_OTHER).ok()?;
            let p = b.ptr();
            m.insert(l.uid, b);
            Some((p, PfWKind::F32))
        }
        ty => {
            let (slot, blk) = match ty {
                T::Q8_0 => (0, 32usize),
                T::Q5_1 => (1, 32usize),
                T::Q4K => (2, 256usize),
                T::Q5K => (3, 256usize),
                _ => return None,
            };
            if l.k % blk != 0 {
                return None;
            }
            let rb = gguf_row_bytes(ty, l.k)?;
            if l.blocks.len() != l.n.checked_mul(rb)? {
                return None;
            }
            match weights_for_gguf(l.uid, ty, l.blocks, l.k, l.n) {
                Ok(w) => Some((w.buf.ptr(), PfWKind::Quant(ty, slot, rb, blk))),
                Err(e) => {
                    tracing::info!("[pf] 量子重みの常駐に失敗 (uid={}): {e:#}", l.uid);
                    None
                }
            }
        }
    }
}


/// task 154: mix の 1 行列 (`GgufLinear` の素性)。
#[derive(Clone, Copy)]
pub struct PfLin<'a> {
    /// `GgufLinear::uid` (常駐の鍵)。
    pub uid: u64,
    /// 重みの量子型 (BF16 だけ受ける)。
    pub ty: crate::kquant::GgmlType,
    /// 生の重みバイト列。
    pub blocks: &'a [u8],
    /// 入力次元。
    pub k: usize,
    /// 出力次元。
    pub n: usize,
}


/// task 195: 診断 — mix の各カーネルの起動形と SM 数を **1 形につき 1 回**だけ出す。
///
/// `DENNO_PF_MIX_KPROF` が立っているときだけ動く。`blocks` は grid の積、`sms` は
/// `gemv_sm_count()` (コンテキスト初期化時に取った実機の SM 数。0 = 不明)、
/// `waves` = blocks / sms。1 ブロックが 1 SM を占有するとは限らないので waves は
/// 「何波ぶんのブロックがあるか」の目安 (推定)。
pub(crate) fn pf_log_grid(tag: &str, grid: (u32, u32, u32), threads: u32) {
    if !pf_mix_kprof::on() {
        return;
    }
    thread_local! {
        static SEEN: std::cell::RefCell<Vec<(String, u32, u32, u32, u32)>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }
    let key = (tag.to_string(), grid.0, grid.1, grid.2, threads);
    let fresh = SEEN.with(|c| {
        let mut v = c.borrow_mut();
        if v.iter().any(|e| {
            e.0 == key.0 && e.1 == key.1 && e.2 == key.2 && e.3 == key.3 && e.4 == key.4
        }) {
            false
        } else {
            v.push(key.clone());
            true
        }
    });
    if !fresh {
        return;
    }
    let blocks = grid.0 as u64 * grid.1 as u64 * grid.2 as u64;
    let sms = gemv_sm_count();
    let waves = if sms > 0 {
        blocks as f64 / sms as f64
    } else {
        f64::NAN
    };
    eprintln!(
        "[pf-mix-grid] {tag} grid={}x{}x{} blocks={} threads={} sms={} waves={:.2}",
        grid.0, grid.1, grid.2, blocks, threads, sms, waves
    );
}


pub(crate) struct PfElemKernels {
    #[allow(dead_code)]
    module: CUmodule,
    pub(crate) rmsgroup_p: CUfunction,
    pub(crate) mixed_fs: CUfunction,
}


// 生ハンドル (PfKernels / MixInjKernels と同じ規約)。
unsafe impl Send for PfElemKernels {}

unsafe impl Sync for PfElemKernels {}


/// PTX の置き場所。`DENNO_PF_ELEM_PTX` があればそれ、無ければクレートの
/// `kernels/kernels_pf_elem.ptx`。
fn pf_elem_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_PF_ELEM_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_pf_elem.ptx"),
    }
}


/// カーネルを取得する (初回だけ PTX を読む)。失敗は None を焼き付ける。
pub(crate) fn pf_elem_kernels() -> Option<&'static PfElemKernels> {
    static K: OnceLock<Option<PfElemKernels>> = OnceLock::new();
    K.get_or_init(|| {
        let path = pf_elem_ptx_path();
        let mut image = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                tracing::info!(
                    "[pf-elem] PTX が読めない ({}) : {e} → 従来の要素カーネルへ",
                    path.display()
                );
                return None;
            }
        };
        // 引数の数はモジュールを載せる前に見る (ABI の取り違えを先に落とす)。
        {
            let text = String::from_utf8_lossy(&image).to_string();
            for (name, want) in [("dno_pf_rmsgroup_p", 7usize), ("dno_pf_mixed_fs", 6)] {
                match ptx_entry_params(&text, name) {
                    Some(n) if n == want => {}
                    other => {
                        tracing::info!(
                            "[pf-elem] {name} の引数が {other:?} (期待 {want}) → 従来の要素カーネルへ"
                        );
                        return None;
                    }
                }
            }
        }
        image.push(0);
        let lib = lib().ok()?;
        let mut module: CUmodule = std::ptr::null_mut();
        unsafe {
            if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void) != CUDA_SUCCESS {
                tracing::info!("[pf-elem] cuModuleLoadData 失敗 → 従来の要素カーネルへ");
                return None;
            }
            let mut fns: [CUfunction; 2] = [std::ptr::null_mut(); 2];
            let names: [&[u8]; 2] = [b"dno_pf_rmsgroup_p\0", b"dno_pf_mixed_fs\0"];
            for (i, name) in names.iter().enumerate() {
                if (lib.module_get_function)(&mut fns[i], module, name.as_ptr() as *const c_char)
                    != CUDA_SUCCESS
                {
                    tracing::info!(
                        "[pf-elem] PTX に {} が無い → 従来の要素カーネルへ",
                        String::from_utf8_lossy(&name[..name.len() - 1])
                    );
                    return None;
                }
            }
            log_kernel_attr(lib, fns[0], "dno_pf_rmsgroup_p");
            log_kernel_attr(lib, fns[1], "dno_pf_mixed_fs");
            Some(PfElemKernels {
                module,
                rmsgroup_p: fns[0],
                mixed_fs: fns[1],
            })
        }
    })
    .as_ref()
}


/// 発火の証拠を 1 形につき 1 回だけ出す (`PROFILE` の有無に依らない)。
///
/// lab の COND でゲートが届いたか・新カーネルが走ったかを 1 見で分けるため。
/// `[pf-mix-grid]` と違い `DENNO_PF_MIX_KPROF` を要求しない (本番の既定 on でも
/// プロセスあたり数行しか出ない)。
pub(crate) fn pf_elem_note(tag: &str, grid: (u32, u32, u32), threads: u32, detail: &str) {
    thread_local! {
        static SEEN: std::cell::RefCell<Vec<(String, u32, u32, u32, u32)>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }
    let key = (tag.to_string(), grid.0, grid.1, grid.2, threads);
    let fresh = SEEN.with(|c| {
        let mut v = c.borrow_mut();
        if v.contains(&key) {
            false
        } else {
            v.push(key.clone());
            true
        }
    });
    if !fresh {
        return;
    }
    eprintln!(
        "[pf-elem] {tag} grid={}x{}x{} threads={threads} {detail}",
        grid.0, grid.1, grid.2
    );
}


#[cfg(test)]
mod pf_elem_tests {
    use super::*;

    /// ゲートの真理値表 (**明示的な偽値だけ off** = 既定 on。切り戻し用)。
    #[test]
    fn pf_elem_gate_truth_table() {
        assert!(
            pf_mix_normp_from_value(None) && pf_mix_fastsig_from_value(None),
            "未設定が既定 on でない"
        );
        for v in [Some("0"), Some("off"), Some("false"), Some("no"), Some(""), Some(" OFF "), Some("0 ")] {
            assert!(!pf_mix_normp_from_value(v), "normp {v:?} が on になった");
            assert!(!pf_mix_fastsig_from_value(v), "fastsig {v:?} が on になった");
        }
        for v in [Some("1"), Some("on"), Some("true"), Some("yes"), Some(" ON "), Some("2")] {
            assert!(pf_mix_normp_from_value(v), "normp {v:?} が on でない");
            assert!(pf_mix_fastsig_from_value(v), "fastsig {v:?} が on でない");
        }
    }

    /// norm 分担版の形の門番: レジスタ上限まで、それ以上は従来へ。
    #[test]
    fn pf_elem_normp_ok_bounds() {
        assert!(pf_mix_normp_ok(2560), "実形 e=2,560 が落ちている");
        assert!(pf_mix_normp_ok(1));
        assert!(pf_mix_normp_ok(PF_ELEM_NORM_THREADS * PF_ELEM_MAX_PER));
        assert!(!pf_mix_normp_ok(0));
        assert!(!pf_mix_normp_ok(PF_ELEM_NORM_THREADS * PF_ELEM_MAX_PER + 1));
    }

    /// fast sigmoid 版の形の門番。
    #[test]
    fn pf_elem_fastsig_ok_bounds() {
        assert!(pf_mix_fastsig_ok(2560, 4, 1913), "実形が落ちている");
        assert!(pf_mix_fastsig_ok(2560, 1, 32));
        assert!(!pf_mix_fastsig_ok(2560, 4, 19), "短い prefill は従来");
        assert!(!pf_mix_fastsig_ok(2560, 0, 512));
        assert!(!pf_mix_fastsig_ok(2560, 9, 512), "hc>8 は従来");
        assert!(!pf_mix_fastsig_ok(0, 4, 512));
    }

    /// 実形の grid が 1 ブロック 1 行 / e を 256 で割った形になること。
    #[test]
    fn pf_elem_real_shape_grid() {
        let (nt, hc, e) = (1913usize, 4usize, 2560usize);
        assert_eq!((nt * hc) as u32, 7652, "norm の grid が実測と違う");
        assert_eq!(e.div_ceil(PF_ELEM_MIXED_THREADS) as u32, 10, "mixed の grid.x が違う");
        // 1 スレッドあたりの要素数 = ceil(e/256) = 10 <= PF_ELEM_MAX_PER
        assert_eq!(e.div_ceil(PF_ELEM_NORM_THREADS), 10);
        assert!(e.div_ceil(PF_ELEM_NORM_THREADS) <= PF_ELEM_MAX_PER);
    }

    /// 分担版の二乗和 (スレッド内 10 要素 + 32 本の木 + 8 warp) をホストで再現し、
    /// 従来の逐次和と f64 真値の両方に 1e-6 相対で合うこと (= 木の順序の取り違え
    /// が無いこと) を確かめる。
    #[test]
    fn pf_elem_norm_tree_matches_reference() {
        let e = 2560usize;
        let nt = PF_ELEM_NORM_THREADS;
        let x: Vec<f32> = (0..e)
            .map(|i| ((i as f32) * 0.001).sin() + 0.5)
            .collect();
        // (a) 従来 (1 スレッドが i 昇順)
        let mut serial = 0.0f32;
        for v in x.iter() {
            serial = fadd(serial, fmul(*v, *v));
        }
        // (b) 分担版
        let nper = e.div_ceil(nt);
        let mut part = vec![0.0f32; nt];
        for (t, p) in part.iter_mut().enumerate() {
            let mut ss = 0.0f32;
            for k in 0..nper {
                let i = t + k * nt;
                if i < e {
                    ss = fadd(ss, fmul(x[i], x[i]));
                }
            }
            *p = ss;
        }
        // warp 内の木 (__shfl_down_sync の off=16..1)。
        // 注意: shfl は**その時点のレーンの値**を読むので、ホストの鏡も
        // step ごとに別バッファへ書く (in-place で足すと木が壊れる)。
        for w in 0..(nt / 32) {
            let base = w * 32;
            let mut cur: Vec<f32> = part[base..base + 32].to_vec();
            for off in [16usize, 8, 4, 2, 1] {
                let mut next = cur.clone();
                for l in 0..32 {
                    if l + off < 32 {
                        next[l] = fadd(cur[l], cur[l + off]);
                    }
                }
                cur = next;
            }
            part[base] = cur[0];
        }
        let mut tree = part[0];
        for w in 1..(nt / 32) {
            tree = fadd(tree, part[w * 32]);
        }
        // f64 真値
        let truth: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let rel = |a: f32, b: f64| ((a as f64) - b).abs() / b.abs();
        assert!(rel(tree, truth) < 1e-6, "tree {tree} vs truth {truth}");
        assert!(rel(serial, truth) < 1e-6, "serial {serial} vs truth {truth}");
        // 分担版は従来と厳密一致はしない (丸め順が違う) が、差は 1e-7 相対以内。
        let d = ((tree as f64) - (serial as f64)).abs() / (serial as f64).abs();
        assert!(d < 1e-6, "tree {tree} / serial {serial} の相対差 {d}");
    }

    /// fast sigmoid の値が従来 (f64 expf 相当の f32 参照) と 3e-7 相対以内で、
    /// 端点 (0 / 1) で壊れないこと。
    #[test]
    fn pf_elem_fastsig_values() {
        for g in [-40.0f32, -8.0, -1.0, 0.0, 1.0, 8.0, 40.0] {
            let fast = fdiv(1.0, fadd(1.0, expf_f32(-g)));
            let slow = 1.0f32 / (1.0f32 + (-g as f64).exp() as f32);
            let d = (fast - slow).abs() / slow.max(1e-30);
            assert!(d < 3e-7, "g={g}: fast {fast} slow {slow} 相対差 {d}");
        }
        // 飽和: 大きな負の gate で sigmoid -> 0、正で 1。
        assert_eq!(fdiv(1.0, fadd(1.0, expf_f32(1000.0))), 0.0);
        let s = fdiv(1.0, fadd(1.0, expf_f32(-1000.0)));
        assert!((s - 1.0).abs() < 1e-7, "{s}");
    }

    // --- ホスト側の鏡 (カーネルの 1 演算 1 丸めと同じ) ---
    fn fadd(a: f32, b: f32) -> f32 {
        a + b
    }
    fn fmul(a: f32, b: f32) -> f32 {
        a * b
    }
    fn fdiv(a: f32, b: f32) -> f32 {
        a / b
    }
    /// `__expf` の近似 (2^-x を f32 で)。値の比較用の粗い鏡。
    fn expf_f32(x: f32) -> f32 {
        if x > 88.0 {
            return f32::INFINITY;
        }
        if x < -103.0 {
            return 0.0;
        }
        // 2^(x/ln2) を exp2 の分解で (f32 の丸めのみ)。
        let t = (x as f64) * std::f64::consts::LOG2_E;
        (2.0f64).powf(t) as f32
    }
}



// ---------------------------------------------------------------------------
// task 461: prefill の PLE 注入をデバイスで閉じる (`DENNO_PF_PLE_DEV`)
// ---------------------------------------------------------------------------
//
// 動機は `kernels_pf_ple.cu` の先頭。lab t457c の `[q4ledger-pf]` (既定モデル、
// depth8k、PROFILE=1) では PLE 層 (48 層中 2 層) が 1 チャンク層あたり
// res 82 MB を D2H → ホストのトークンループ → 82 MB を H2D し、14k cold で
// `sub other` の `ple` 212.9 ms + `res_up` 245.8 ms = prefill の 8% だった。
// res / key / value はどれも元から VRAM にあるので、往復をやめて 2 本の
// 要素カーネルで閉じる (gate: 1 ブロック = 1 トークン / conv: 1 ブロック = 列タイル)。
//
// 既定 **off**。PTX が読めない / entry が無い / 引数の数が違う / 形が門番の外、
// のどれでも `Ok(None)` を返し、呼び出し側は従来のホスト経路をそのまま回す
// (res は 1 バイトも触らない)。**起動を始めた後**の失敗は `Err` にする
// (res を書き換えたかもしれないので黙って落とさない)。

/// `DENNO_PF_PLE_DEV` の env 名。
pub const PF_PLE_DEV_ENV: &str = "DENNO_PF_PLE_DEV";

/// `DENNO_PF_PLE_PTX` (PTX の置き場所。未設定はクレートの `kernels/`)。
pub const PF_PLE_PTX_ENV: &str = "DENNO_PF_PLE_PTX";


/// gate カーネルの引数の数 (カーネルと一致しない PTX は読まない)。
pub(crate) const PF_PLE_GATE_NARGS: usize = 11;

/// conv カーネルの引数の数。
pub(crate) const PF_PLE_CONV_NARGS: usize = 18;

/// 1 ブロックのスレッド数 (カーネルの `PLE_THREADS`)。
pub(crate) const PF_PLE_THREADS: usize = 256;

/// hc の上限 (カーネルの `PLE_MAX_HC`)。
pub(crate) const PF_PLE_MAX_HC: usize = 8;

/// e の上限 (カーネルの `PLE_MAX_E`)。
pub(crate) const PF_PLE_MAX_E: usize = 8192;

/// conv のカーネル幅の上限 (カーネルの `PLE_MAX_KERN`)。
pub(crate) const PF_PLE_MAX_KERN: usize = 8;

/// 履歴の長さの上限 (カーネルの `PLE_MAX_HIST`)。
pub(crate) const PF_PLE_MAX_HIST: usize = 32;

/// これ未満のトークン数は従来経路 (起動と往復の固定費が勝つ)。
pub(crate) const PF_PLE_DEV_MIN_TOK: usize = 64;


/// `DENNO_PF_PLE_DEV` の解釈。**明示的な偽値だけ off** = 既定 on。
///
/// lab t461d で採った: 3,826 cold prefill が base2 2,211 / 2,216 ms に対し
/// **2,143 / 2,106 ms (-3.2% / -4.7%、2 ラウンドとも)**、`other` 段は
/// 190.5 -> 85.3 ms。14k は ±1% (other -83 / experts +87 で相殺)、decode は不変。
pub(crate) fn pf_ple_dev_from_value(v: Option<&str>) -> bool {
    match v {
        None => true,
        Some(s) => !matches!(
            s.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no" | ""
        ),
    }
}


/// デバイス経路を使うか (プロセスで 1 回だけ env を読む。既定 on。
/// 切り戻しは `DENNO_PF_PLE_DEV=0`)。
pub(crate) fn pf_ple_dev_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| pf_ple_dev_from_value(std::env::var(PF_PLE_DEV_ENV).ok().as_deref()))
}


/// 形の門番 (純関数)。**res を触る前に**全部見る。
///
/// 実形は nt=1,913/2,002、hc=4、e=2,560、kern=2..4、hist=2..8。
pub(crate) fn pf_ple_dev_ok(
    nt: usize,
    hc: usize,
    e: usize,
    kern: usize,
    dil: usize,
    hist: usize,
) -> bool {
    nt >= PF_PLE_DEV_MIN_TOK
        && hc >= 1
        && hc <= PF_PLE_MAX_HC
        && e > 0
        && e <= PF_PLE_MAX_E
        && kern >= 1
        && kern <= PF_PLE_MAX_KERN
        && dil >= 1
        && hist == (kern - 1) * dil
        && hist <= PF_PLE_MAX_HIST
}


pub(crate) struct PfPleKernels {
    #[allow(dead_code)]
    module: CUmodule,
    gate: CUfunction,
    conv: CUfunction,
}


// 生ハンドル (PfElemKernels と同じ規約)。
unsafe impl Send for PfPleKernels {}

unsafe impl Sync for PfPleKernels {}


/// PTX の置き場所。`DENNO_PF_PLE_PTX` があればそれ、無ければクレートの
/// `kernels/kernels_pf_ple.ptx`。
fn pf_ple_ptx_path() -> std::path::PathBuf {
    match std::env::var(PF_PLE_PTX_ENV) {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_pf_ple.ptx"),
    }
}


/// カーネルを取得する (初回だけ PTX を読む)。失敗は None を焼き付ける。
fn pf_ple_kernels() -> Option<&'static PfPleKernels> {
    static K: OnceLock<Option<PfPleKernels>> = OnceLock::new();
    K.get_or_init(|| {
        let path = pf_ple_ptx_path();
        let mut image = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                tracing::info!(
                    "[pf-ple] PTX が読めない ({}) : {e} → 従来のホスト経路へ",
                    path.display()
                );
                return None;
            }
        };
        // 引数の数はモジュールを載せる前に見る (ABI の取り違えを先に落とす)。
        {
            let text = String::from_utf8_lossy(&image).to_string();
            for (name, want) in [
                ("dno_pf_ple_gate", PF_PLE_GATE_NARGS),
                ("dno_pf_ple_conv", PF_PLE_CONV_NARGS),
            ] {
                match ptx_entry_params(&text, name) {
                    Some(n) if n == want => {}
                    other => {
                        tracing::info!(
                            "[pf-ple] {name} の引数が {other:?} (期待 {want}) → 従来のホスト経路へ"
                        );
                        return None;
                    }
                }
            }
        }
        image.push(0);
        let lib = lib().ok()?;
        let mut module: CUmodule = std::ptr::null_mut();
        unsafe {
            if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void) != CUDA_SUCCESS {
                tracing::info!("[pf-ple] cuModuleLoadData 失敗 → 従来のホスト経路へ");
                return None;
            }
            let mut fns: [CUfunction; 2] = [std::ptr::null_mut(); 2];
            let names: [&[u8]; 2] = [b"dno_pf_ple_gate\0", b"dno_pf_ple_conv\0"];
            for (i, name) in names.iter().enumerate() {
                if (lib.module_get_function)(&mut fns[i], module, name.as_ptr() as *const c_char)
                    != CUDA_SUCCESS
                {
                    tracing::info!(
                        "[pf-ple] PTX に {} が無い → 従来のホスト経路へ",
                        String::from_utf8_lossy(&name[..name.len() - 1])
                    );
                    return None;
                }
            }
            log_kernel_attr(lib, fns[0], "dno_pf_ple_gate");
            log_kernel_attr(lib, fns[1], "dno_pf_ple_conv");
            Some(PfPleKernels {
                module,
                gate: fns[0],
                conv: fns[1],
            })
        }
    })
    .as_ref()
}


/// デバイス経路の作業域 (grow-only)。スレッド局所。
pub(crate) struct PfPleDev {
    pub(crate) flat: Option<DevBuffer>,  // [nt][e]
    pub(crate) key: Option<DevBuffer>,   // [nt][hc*e]
    pub(crate) value: Option<DevBuffer>, // [nt][e]
    pub(crate) gate: Option<DevBuffer>,  // [nt][hc]
    pub(crate) invc: Option<DevBuffer>,  // [nt][hc]
    pub(crate) part: Option<DevBuffer>,  // [hc][ntiles]
    pub(crate) wgk: Option<DevBuffer>,   // [hc][e]
    pub(crate) wgq: Option<DevBuffer>,   // [hc][e]
    pub(crate) wgc: Option<DevBuffer>,   // [hc][e]
    pub(crate) wwt: Option<DevBuffer>,   // [kern][hc][e]
    pub(crate) hist: Vec<Option<DevBuffer>>, // 層ごとの [hist][hc][e]
    /// 発火の証拠を 1 形につき 1 回だけ出すための記録。
    pub(crate) noted: Vec<(usize, usize, usize)>,
}


/// 発火の証拠 (`[pf-ple] dev ...` を 1 形につき 1 回)。
fn pf_ple_note(nt: usize, hc: usize, e: usize, kern: usize, hist: usize) {
    let fresh = PLE_DEV.with(|c| {
        let mut d = c.borrow_mut();
        let key = (nt, hc, e);
        if d.noted.contains(&key) {
            false
        } else {
            d.noted.push(key);
            true
        }
    });
    if fresh {
        eprintln!(
            "[pf-ple] dev nt={nt} hc={hc} e={e} kern={kern} hist={hist} \
             gate=dno_pf_ple_gate conv=dno_pf_ple_conv"
        );
    }
}


/// PLE の射影 (key / value) を**デバイスの出力域へ**回す。
///
/// 量子型 (Q8_0 / Q5_1 / Q4_K / Q5_K) は従来のホスト経路 (`pf_ple_proj`) と
/// **同じ** `pf2_gemm` を、出力だけデバイスにして撃つ (D2H をしない)。
/// F32 / BF16 / F16 は [`f32_gemm_dev_out`]。
fn pf_ple_proj_dev_out(
    ctx: &CudaCtx,
    key: PfLin<'_>,
    value: PfLin<'_>,
    dev_x: CUdeviceptr,
    nt: usize,
    e: usize,
    dev_k: CUdeviceptr,
    dev_v: CUdeviceptr,
) -> bool {
    use crate::kquant::GgmlType as T;
    let quant = |t: T| matches!(t, T::Q8_0 | T::Q5_1 | T::Q4K | T::Q5K);
    if quant(key.ty) && quant(value.ty) {
        let (Some((pk, pk_k)), Some((pv, pv_k))) =
            (pf_weight(key, false), pf_weight(value, false))
        else {
            return false;
        };
        return pf2_gemm(ctx, pk, pk_k, dev_x, dev_k, nt, e, key.n).is_ok()
            && pf2_gemm(ctx, pv, pv_k, dev_x, dev_v, nt, e, value.n).is_ok();
    }
    f32_gemm_dev_out(
        key.uid,
        key.blocks,
        key.k,
        key.n,
        dev_x,
        nt,
        dev_k,
        key.ty,
    ) && f32_gemm_dev_out(
        value.uid,
        value.blocks,
        value.k,
        value.n,
        dev_x,
        nt,
        dev_v,
        value.ty,
    )
}


/// 失敗の理由を 1 回だけ出す (lab のログで原因を 1 行で分けるため)。
fn pf_ple_warn(why: &str) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| eprintln!("[pf-ple] {why} → 従来のホスト経路へ"));
}


/// デバイス経路 1 回ぶんの入力。
pub struct PfPleDevArgs<'a> {
    /// PLE の行を詰めた `[nt][e]` (gather 済み)。
    pub flat: &'a [f32],
    pub nt: usize,
    pub hc: usize,
    pub e: usize,
    /// key / value の射影の重み。
    pub key: PfLin<'a>,
    pub value: PfLin<'a>,
    /// grouped RMS の重み (`[hc][e]`)。
    pub gk: &'a [f32],
    pub gq: &'a [f32],
    pub gc: &'a [f32],
    /// conv の重み (`[kern][hc][e]`、`conv1d_t` の並び)。
    pub wt: &'a [f32],
    /// 履歴 (`[hist][hc][e]`)。入力でもあり、更新後の値も返す。
    pub hist_host: &'a mut [f32],
    pub hist_len: usize,
    /// ホストの履歴をデバイスへ上げるか。層の最初のチャンク (または直前の
    /// チャンクがホスト経路だったとき) だけ true。上げ直さないと、デバイス側の
    /// 履歴が正本のまま層を進める (同期と D2H を 1 層 1 回に減らすため)。
    pub upload_hist: bool,
    /// 履歴をホストへ戻すか。層の**最後のチャンク**だけ true。
    /// D2H は同期を伴うので、途中のチャンクでは払わない (lab t461c で
    /// 14k の `experts` が +89 ms 悪化した原因がこの同期だった)。
    pub flush_hist: bool,
    pub kern: usize,
    pub dil: usize,
    /// 層番号 (履歴の置き場の鍵)。
    pub li: usize,
    pub eps: f32,
    /// 最終トークンの寄与 RMS が要るか (役割テレメトリ)。
    pub want: bool,
}


/// デバイス経路の結果。
pub struct PfPleDevOut {
    /// `want` のときの (gated + conv) の RMS。
    pub contrib_rms: f32,
}


/// PLE の注入 1 回 (gate → conv+silu → res 加算 → 履歴の更新) をデバイスで回す。
///
/// `Ok(None)` = 何もせず従来経路へ / `Ok(Some(_))` = 完了 /
/// `Err` = 起動を始めた後に失敗 (res を書き換えたかもしれない)。
pub fn pf_ple_dev_run(a: PfPleDevArgs<'_>) -> Result<Option<PfPleDevOut>, String> {
    let (nt, hc, e) = (a.nt, a.hc, a.e);
    if !pf_ple_dev_ok(nt, hc, e, a.kern, a.dil, a.hist_len) {
        return Ok(None);
    }
    let hc_dim = hc * e;
    if a.flat.len() != nt * e
        || a.gk.len() != hc_dim
        || a.gq.len() != hc_dim
        || a.gc.len() != hc_dim
        || a.wt.len() != a.kern * hc_dim
        || a.hist_host.len() != a.hist_len * hc_dim
        || a.key.k != e
        || a.key.n != hc_dim
        || a.value.k != e
        || a.value.n != e
    {
        return Ok(None);
    }
    let Some(k) = pf_ple_kernels() else {
        return Ok(None);
    };
    let Some(ctx) = shared_ctx() else {
        return Ok(None);
    };
    if ctx.make_current().is_err() {
        return Ok(None);
    }
    // 残差のデバイス域 (pf_mix / pf_upload_res と同じもの) が無ければ従来経路。
    let res_ptr = pf_res_dev_ptr();
    if res_ptr == 0 {
        return Ok(None);
    }
    // ここから先は「始めたら最後まで」。失敗は Err。
    let ntiles = e.div_ceil(PF_PLE_THREADS);
    let mut parts = vec![0.0f32; hc * ntiles];
    let r = PLE_DEV.with(|cell| -> Result<Option<PfPleDevOut>, String> {
        let mut d = cell.borrow_mut();
        let mut ensure = |cap: &mut Option<DevBuffer>, bytes: usize| -> Result<(), String> {
            Scratch::ensure(cap, bytes).map_err(|e| format!("PLE 作業域の確保: {e:#}"))
        };
        ensure(&mut d.flat, nt * e * 4)?;
        ensure(&mut d.key, nt * hc_dim * 4)?;
        ensure(&mut d.value, nt * e * 4)?;
        ensure(&mut d.gate, nt * hc * 4)?;
        ensure(&mut d.invc, nt * hc * 4)?;
        ensure(&mut d.part, hc * ntiles * 4)?;
        ensure(&mut d.wgk, hc_dim * 4)?;
        ensure(&mut d.wgq, hc_dim * 4)?;
        ensure(&mut d.wgc, hc_dim * 4)?;
        ensure(&mut d.wwt, a.kern * hc_dim * 4)?;
        if d.hist.len() <= a.li {
            d.hist.resize_with(a.li + 1, || None);
        }
        ensure(&mut d.hist[a.li], a.hist_len.max(1) * hc_dim * 4)?;
        let (flat, keyv, valv) = (
            d.flat.as_ref().expect("ensure 済み"),
            d.key.as_ref().expect("ensure 済み"),
            d.value.as_ref().expect("ensure 済み"),
        );
        let (gate, invc, part) = (
            d.gate.as_ref().expect("ensure 済み"),
            d.invc.as_ref().expect("ensure 済み"),
            d.part.as_ref().expect("ensure 済み"),
        );
        let (wgk, wgq, wgc, wwt) = (
            d.wgk.as_ref().expect("ensure 済み"),
            d.wgq.as_ref().expect("ensure 済み"),
            d.wgc.as_ref().expect("ensure 済み"),
            d.wwt.as_ref().expect("ensure 済み"),
        );
        let hist = d.hist[a.li].as_ref().expect("ensure 済み");
        // 入力の H2D (flat / 重み / 履歴)。ここまで落ちたら何も走っていない。
        flat.copy_from(a.flat)
            .map_err(|e| format!("flat H2D: {e:#}"))?;
        wgk.copy_from(a.gk).map_err(|e| format!("gk H2D: {e:#}"))?;
        wgq.copy_from(a.gq).map_err(|e| format!("gq H2D: {e:#}"))?;
        wgc.copy_from(a.gc).map_err(|e| format!("gc H2D: {e:#}"))?;
        wwt.copy_from(a.wt).map_err(|e| format!("wt H2D: {e:#}"))?;
        if a.hist_len > 0 && a.upload_hist {
            hist.copy_from(&a.hist_host[..])
                .map_err(|e| format!("hist H2D: {e:#}"))?;
        }
        // 射影 2 本 (デバイス出力。D2H をしない)。
        //
        // 既定モデルの ple_key / ple_value は **Q8_0** (目録も Q8_0) で、
        // 従来のホスト経路は `pf_ple_proj` の量子 GEMM (`pf2_gemm`) で回す。
        // ここも**同じカーネル**を出力だけデバイスにして撃つ (値は同一)。
        // F32 / BF16 / F16 の目録のときは f32 GEMM のデバイス出力版。
        if !pf_ple_proj_dev_out(
            ctx, a.key, a.value, flat.ptr(), nt, e, keyv.ptr(), valv.ptr(),
        ) {
            pf_ple_warn("射影 (デバイス出力) に失敗");
            return Err("PLE の射影 (デバイス出力) に失敗".to_string());
        }
        // gate / invc
        let mut p_key = keyv.ptr();
        let mut p_res = res_ptr;
        let mut p_val = valv.ptr();
        let mut p_gk = wgk.ptr();
        let mut p_gq = wgq.ptr();
        let mut p_gate = gate.ptr();
        let mut p_invc = invc.ptr();
        let (mut e_c, mut hc_c) = (e as c_int, hc as c_int);
        let mut eps_c = a.eps;
        let mut invs_c = 1.0f32 / (e as f32).sqrt();
        let mut args = [
            &mut p_key as *mut _ as *mut c_void,
            &mut p_res as *mut _ as *mut c_void,
            &mut p_val as *mut _ as *mut c_void,
            &mut p_gk as *mut _ as *mut c_void,
            &mut p_gq as *mut _ as *mut c_void,
            &mut p_gate as *mut _ as *mut c_void,
            &mut p_invc as *mut _ as *mut c_void,
            &mut e_c as *mut _ as *mut c_void,
            &mut hc_c as *mut _ as *mut c_void,
            &mut eps_c as *mut _ as *mut c_void,
            &mut invs_c as *mut _ as *mut c_void,
        ];
        unsafe {
            ctx.launch_async(
                k.gate,
                (nt as u32, 1, 1),
                (PF_PLE_THREADS as u32, 1, 1),
                0,
                &mut args,
            )
            .map_err(|e| format!("dno_pf_ple_gate の起動: {e:#}"))?;
        }
        // conv + silu + res 加算
        let mut p_gc = wgc.ptr();
        let mut p_wwt = wwt.ptr();
        let mut p_hist_in = hist.ptr();
        let mut p_hist_out = hist.ptr();
        let mut p_part = part.ptr();
        let (mut nt_c, mut kern_c, mut dil_c) = (nt as c_int, a.kern as c_int, a.dil as c_int);
        let (mut hist_c, mut head_c) = (a.hist_len as c_int, 0 as c_int);
        let (mut want_c, mut ntiles_c) = (a.want as c_int, ntiles as c_int);
        let mut args2 = [
            &mut p_val as *mut _ as *mut c_void,
            &mut p_gate as *mut _ as *mut c_void,
            &mut p_invc as *mut _ as *mut c_void,
            &mut p_gc as *mut _ as *mut c_void,
            &mut p_wwt as *mut _ as *mut c_void,
            &mut p_hist_in as *mut _ as *mut c_void,
            &mut p_hist_out as *mut _ as *mut c_void,
            &mut p_res as *mut _ as *mut c_void,
            &mut p_part as *mut _ as *mut c_void,
            &mut nt_c as *mut _ as *mut c_void,
            &mut e_c as *mut _ as *mut c_void,
            &mut hc_c as *mut _ as *mut c_void,
            &mut kern_c as *mut _ as *mut c_void,
            &mut dil_c as *mut _ as *mut c_void,
            &mut hist_c as *mut _ as *mut c_void,
            &mut head_c as *mut _ as *mut c_void,
            &mut want_c as *mut _ as *mut c_void,
            &mut ntiles_c as *mut _ as *mut c_void,
        ];
        unsafe {
            ctx.launch_async(
                k.conv,
                (ntiles as u32, hc as u32, 1),
                (PF_PLE_THREADS as u32, 1, 1),
                0,
                &mut args2,
            )
            .map_err(|e| format!("dno_pf_ple_conv の起動: {e:#}"))?;
        }
        // 履歴と寄与 (少量) を戻す。**どちらも要らないチャンクでは同期しない**
        // (同期は expert の上げストリームまで待つので、層ごとに払うと高い)。
        let need_hist = a.hist_len > 0 && a.flush_hist;
        if need_hist || a.want {
            ctx.sync().map_err(|e| format!("PLE の同期: {e:#}"))?;
        }
        if need_hist {
            hist.to_slice(&mut a.hist_host[..])
                .map_err(|e| format!("hist D2H: {e:#}"))?;
        }
        let mut contrib = 0.0f32;
        if a.want {
            part.to_slice(&mut parts)
                .map_err(|e| format!("part D2H: {e:#}"))?;
            // CPU 参照と同じ順 (s 昇順 → 列タイル昇順) に足す。
            let mut sq = 0.0f32;
            for i in 0..parts.len() {
                sq += parts[i];
            }
            contrib = if hc_dim == 0 {
                0.0
            } else {
                (sq / hc_dim as f32).sqrt()
            };
        }
        Ok(Some(PfPleDevOut {
            contrib_rms: contrib,
        }))
    });
    // `PLE_DEV` の borrow を解いてから発火の証拠を出す (内側で呼ぶと
    // 「RefCell already borrowed」で落ちる。lab t461b で踏んだ)。
    if matches!(r, Ok(Some(_))) {
        pf_ple_note(nt, hc, e, a.kern, a.hist_len);
    }
    r
}


/// task 154: hc combine (`res[i] += bo * (2*sigmoid(inj))`) をデバイスで回す。
///
/// `bo` は `[nt][n_embd]` (attn / ffn の出力)。`inj` は直前の [`pf_mix`] が残した
/// デバイス上の値をそのまま使う。
pub fn pf_combine(bo: &[f32], nt: usize, hc: usize, e: usize) -> bool {
    if bo.len() != nt * e {
        return pf_fail("combine の bo 長が違う");
    }
    pf_combine_core(Some(bo), None, nt, hc, e)
}


/// task 200: 直前に ssm 連鎖が**デバイスへ書いた**層出力を、H2D せずに combine する
/// (`DENNO_PF_ATTN_DEVRES`)。記録が無い/形が違うときは false を返す
/// (呼び出し側は要求を落とす — attention の状態は既に進んでいる)。
pub fn pf_combine_dev(nt: usize, hc: usize, e: usize) -> bool {
    let Some((ptr, dnt, de)) = PF_ATTN_DEV_OUT.with(|c| c.borrow_mut().take()) else {
        return pf_fail("combine (デバイス出力): 記録が無い");
    };
    if dnt != nt || de != e {
        return pf_fail("combine (デバイス出力): 形が違う");
    }
    pf_combine_core(None, Some(ptr), nt, hc, e)
}


/// task 233: 層優先 prefill で、MoE の出力 (デバイスの `[nt][e]`) をそのまま
/// combine する。`inj` は呼び出し側が [`pf_lm_inj_load`] で戻しておく。
pub fn pf_combine_at(dev: CUdeviceptr, nt: usize, hc: usize, e: usize) -> bool {
    if dev == 0 {
        return pf_fail("combine (デバイス指定): ポインタが 0");
    }
    pf_combine_core(None, Some(dev), nt, hc, e)
}


fn pf_combine_core(
    bo: Option<&[f32]>,
    dev: Option<CUdeviceptr>,
    nt: usize,
    hc: usize,
    e: usize,
) -> bool {
    // task 254: `other` の中身 (`[q4ledger-pf] sub other`) に積む。
    let _sub = crate::model::q4ledger::sub(crate::model::q4ledger::SUB_COMBINE);
    if PF_BROKEN.load(std::sync::atomic::Ordering::Relaxed) {
        return false;
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
    let r = PF_DEV.with(|cell| -> Result<(), String> {
        let d = cell.borrow();
        if nt > d.nt || hc != d.hc || e != d.e {
            return Err(format!("combine の形が違う (nt={nt} hc={hc} e={e})"));
        }
        let (Some(res), Some(inj)) = (d.res.as_ref(), d.inj.as_ref()) else {
            return Err("作業域が無い".into());
        };
        // task 200: `dev` が来たら転送しない (attn が既にその位置へ書いている)。
        let pb = match (bo, dev) {
            (Some(b), _) => {
                let Some(dbo) = d.bo.as_ref() else {
                    return Err("作業域が無い (bo)".into());
                };
                // task 318: `DENNO_PF_PIN_STAGE` では pinned 経由 (速い)。
                match dbo.copy_from_pinned(b) {
                    Ok(true) => {}
                    _ => {
                        dbo.copy_from(b).map_err(|e| format!("bo H2D: {e:#}"))?;
                    }
                }
                dbo.ptr()
            }
            (None, Some(p)) => p,
            (None, None) => return Err("combine の入力が無い".into()),
        };
        let (mut pr, mut pbv, mut pj) = (res.ptr(), pb, inj.ptr());
        let (mut e_c, mut hc_c) = (e as i32, hc as i32);
        let grid = ((e.div_ceil(256)) as u32, (nt * hc) as u32, 1);
        let mut args = [
            &mut pr as *mut _ as *mut c_void,
            &mut pbv as *mut _ as *mut c_void,
            &mut pj as *mut _ as *mut c_void,
            &mut e_c as *mut _ as *mut c_void,
            &mut hc_c as *mut _ as *mut c_void,
        ];
        unsafe {
            ctx.launch_async(k.combine, grid, (256, 1, 1), 0, &mut args)
                .map_err(|e| format!("combine launch: {e:#}"))?;
        }
        // task 189: mix と同じく、expert upload の別ストリームを待たない。
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


fn pf2_warn(msg: &str) {
    if !PF2_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        tracing::info!("[pf2] {msg} → 従来経路");
        eprintln!("[pf2] {msg} → 従来経路");
    }
}


/// [`PF2_STATS`] の読み出し。
pub fn pf2_stats() -> [u64; 6] {
    let mut out = [0u64; 6];
    for (i, c) in PF2_STATS.iter().enumerate() {
        out[i] = c.load(Ordering::Relaxed);
    }
    out
}


/// [`PF2_STATS`] を 0 に戻す。
pub fn pf2_stats_reset() {
    for c in PF2_STATS.iter() {
        c.store(0, Ordering::Relaxed);
    }
}


#[inline]
fn pf2_note(idx: usize) {
    let n = PF2_STATS[idx].fetch_add(1, Ordering::Relaxed);
    // 成立した stage は最初の 1 回だけ 1 行出す (lab のログで発火を確認するため)。
    if idx % 2 == 0 && n == 0 {
        eprintln!(
            "[pf2] {} 連鎖 発火 (nt={})",
            ["full", "full", "shared", "shared", "ple", "ple"][idx.min(5)],
            pf2_last_nt()
        );
    }
}


fn pf2_last_nt() -> usize {
    PF2_LAST_NT.with(|c| c.get())
}


/// ログ用に行数を控える (`pf_full_qkv` / `pf_shared` / `pf_ple_proj` の入口)。
fn pf2_set_nt(nt: usize) {
    PF2_LAST_NT.with(|c| c.set(nt));
}


/// 呼び出し側 (モデル層) の入口で断念したことを数える。
///
/// `0` = full の成立 / `1` = full の断念 / `2`/`3` = shared / `4`/`5` = ple。
/// 形状や重みの融合など、ここでしか分からない理由で落ちたとき用。
pub fn pf2_decline(idx: usize) {
    pf2_note(idx.min(5));
}


pub(crate) struct Pf2Kernels {
    /// モジュールは生かしておく必要がある (関数ハンドルの持ち主)。
    #[allow(dead_code)]
    module: CUmodule,
    qknorm_rope: CUfunction,
    gate_mul: CUfunction,
}


unsafe impl Send for Pf2Kernels {}

unsafe impl Sync for Pf2Kernels {}


/// PTX の置き場所。`DENNO_PF2_PTX` があればそれ、無ければクレートの
/// `kernels/kernels_pf2.ptx`。
fn pf2_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_PF2_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_pf2.ptx"),
    }
}


/// カーネルを取得する (初回だけ PTX を読む)。失敗は None を焼き付ける。
/// **current context を張った後に呼ぶこと**。
fn pf2_kernels() -> Option<&'static Pf2Kernels> {
    static K: OnceLock<Option<Pf2Kernels>> = OnceLock::new();
    K.get_or_init(|| {
        let path = pf2_ptx_path();
        let mut image = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                pf2_warn(&format!("PTX が読めない ({}) : {e}", path.display()));
                return None;
            }
        };
        image.push(0);
        let lib = match lib() {
            Ok(l) => l,
            Err(_) => return None,
        };
        let mut module: CUmodule = std::ptr::null_mut();
        unsafe {
            if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void) != CUDA_SUCCESS {
                pf2_warn("cuModuleLoadData 失敗");
                return None;
            }
            let mut fns: [CUfunction; 2] = [std::ptr::null_mut(); 2];
            for (i, name) in PF2_NAMES.iter().enumerate() {
                if (lib.module_get_function)(&mut fns[i], module, name.as_ptr() as *const c_char)
                    != CUDA_SUCCESS
                {
                    pf2_warn(&format!(
                        "PTX に {} が無い",
                        String::from_utf8_lossy(&name[..name.len() - 1])
                    ));
                    return None;
                }
            }
            Some(Pf2Kernels {
                module,
                qknorm_rope: fns[0],
                gate_mul: fns[1],
            })
        }
    })
    .as_ref()
}


// ---------------------------------------------------------------------------
// 作業域 (thread_local, grow-only)
// ---------------------------------------------------------------------------

pub(crate) struct Pf2Dev {
    x: Option<DevBuffer>,   // [nt][e]
    qg: Option<DevBuffer>,  // [nt][2*heads*dh]  (attn_q の生の射影)
    /// task 183: 融合 qkv 1 本の生の射影 [nt][n_all] (n_all = 2*qd + 2*kv)。
    /// `pf_full_qkv_fused` が撃ち、`pf_full_out` が gate をここから読む。
    qkv: Option<DevBuffer>,
    q: Option<DevBuffer>,   // [nt][heads*dh]    (norm+rope 済み)
    kd: Option<DevBuffer>,  // [nt][kv_dim]
    vd: Option<DevBuffer>,  // [nt][kv_dim]
    att: Option<DevBuffer>, // [nt][heads*dh]
    o: Option<DevBuffer>,   // [nt][e]
    cs: Option<DevBuffer>,  // [nt][n_rot]
    qn: Option<DevBuffer>,  // [dh]
    kn: Option<DevBuffer>,  // [dh]
    ga: Option<DevBuffer>,  // [nt][sff] (shared の gate)
    gu: Option<DevBuffer>,  // [nt][sff] (shared の up)
    sh: Option<DevBuffer>,  // [nt][e]   (shared の down 出力)
    /// task 181: 直前の `pf_full_qkv` が KV ミラーへ直接書いた範囲
    /// `(cache_id, layer, gen, t_total)`。`pf_full_out` がこれを見て
    /// ミラーの H2D を飛ばす (`DENNO_PF_MIRROR_DELTA`)。
    mir: Option<(u64, usize, u64, usize)>,
}


impl Pf2Dev {
    pub(crate) const fn new() -> Self {
        Self {
            x: None,
            qg: None,
            qkv: None,
            q: None,
            kd: None,
            vd: None,
            att: None,
            o: None,
            cs: None,
            qn: None,
            kn: None,
            ga: None,
            gu: None,
            sh: None,
            mir: None,
        }
    }
}


/// 必要バイト数まで育てる (grow-only。中身は保証しない)。
fn pf2_ensure(slot: &mut Option<DevBuffer>, bytes: usize) -> Result<()> {
    let need = bytes.max(4);
    if slot.as_ref().map(|b| b.bytes()).unwrap_or(0) < need {
        *slot = Some(DevBuffer::alloc_tagged(need, crate::mem_audit::TAG_GEMV)?);
    }
    Ok(())
}


// ---------------------------------------------------------------------------
// 同期しないデバイス GEMM
// ---------------------------------------------------------------------------

/// task 175: `pf2_gemm` が選ぶテンソルコア経路 (純関数)。0 = 従来、1 = v1、2 = v2、
/// 3 = 小さい N 用 (task 181。BN=64)。
///
/// **型を必ず見る**のが要点。v1 カーネル (`dno_gemm_tc_q8_0_w`) は Q8_0 専用で、
/// Q4_K / Q5_1 の重みを渡すとバイト列を別の型として読む (= 出力が壊れる)。
/// task 172 の `pf2_gemm` は形だけで v1 を選んでいたため、shared expert の
/// gate (このモデルでは Q4_K) が壊れ、prefill の出力が全部 `!` になった
/// (lab t175a)。v2 は Q8_0 / Q4_K / Q5_K を型で切り替える。
///
/// 環境ゲート (`DENNO_PF_TC_GEMM` / `DENNO_PF_TC_GEMM2` / `DENNO_PF_TC_SMALLN`)
/// は呼び出し側で見る。ゲートが off のとき kind 3 は v1 → 従来カーネルへ落ちる。
pub(crate) fn pf2_tc_kind(
    ty: crate::kquant::GgmlType,
    nt: usize,
    n_in: usize,
    n_out: usize,
) -> u8 {
    if pf_tc2_eligible(ty, nt, n_in, n_out, 1) {
        2
    } else if pf_tc_smalln_eligible(ty, nt, n_in, n_out, 1) {
        3
    } else if matches!(ty, crate::kquant::GgmlType::Q8_0) && pf_tc_gemm_eligible(nt, n_in, n_out, 1)
    {
        1
    } else {
        0
    }
}


/// task 230: qkv の入力 `x` をデバイスから直接読めるならそのポインタ。
///
/// `DENNO_PF_MIXKEEP` が on で、直前の `mix` の出力がデバイスに残っている
/// (`pf_mixed_dev_in_ptr` が一致) ときだけ `Some`。値は同じバイト列なので
/// H2D を 1 本消せる (D2H していない場合もここで初めてホストを経由しない)。
///
/// 渡せないのに「D2H していない」記録が残っている場合は、ホスト経路が古い
/// バイト列を読まないよう、ここで 1 回だけ降ろして埋め直す。
fn pf2_dev_x(x: &[f32], nt: usize, e: usize) -> Option<CUdeviceptr> {
    if !pf_mix_keep_on() {
        return None;
    }
    match pf_mixed_dev_in_ptr(x.as_ptr(), nt, e) {
        Some(p) => Some(p),
        None => {
            if pf_mixed_dev_only(x.as_ptr(), nt, e) {
                pf_mixed_restore_host(x.as_ptr() as *mut f32, nt, e);
            }
            None
        }
    }
}


/// task 172: full 層 1 チャンクの前半をデバイスで回す。
///
/// 入力行 `x` を 1 回だけ H2D し、`attn_q` / `attn_k` / `attn_v` の 3 射影を
/// デバイスで続けて撃ち、q と k に per-head RMSNorm + partial NeoX rope を掛ける。
/// その後 k / v だけをホストへ戻して KV ミラーへ積む (ホストの `AttnCache` が
/// 正で、decode はそれを読む)。q (と gate) はデバイスに残す。
///
/// 戻り値 `false` は「この形/環境では連鎖を使わない」。呼び出し側は従来経路を
/// そのまま走らせる (副作用なし)。
#[allow(clippy::too_many_arguments)]
pub fn pf_full_qkv(
    x: &[f32],
    nt: usize,
    e: usize,
    q: PfLin<'_>,
    k: PfLin<'_>,
    v: PfLin<'_>,
    q_norm: &[f32],
    k_norm: &[f32],
    cs: &[f32],
    heads: usize,
    kv_heads: usize,
    dh: usize,
    n_rot: usize,
    eps: f32,
    cache_id: u64,
    n_layer: usize,
    layer: usize,
    gen: u64,
    t_total: usize,
    pos0: usize,
    k_out: &mut [f32],
    v_out: &mut [f32],
) -> bool {
    pf2_set_nt(nt);
    let qd = heads * dh;
    let kv = kv_heads * dh;
    let ok_shape = nt >= PF2_MIN_ROWS
        && e > 0
        && dh >= ATTN_PF_BQ
        && dh <= 1024
        && n_rot % 2 == 0
        && n_rot <= dh
        && x.len() == nt * e
        && k_out.len() == nt * kv
        && v_out.len() == nt * kv
        && q_norm.len() == dh
        && k_norm.len() == dh
        && cs.len() == nt * n_rot
        && q.k == e
        && q.n == 2 * qd
        && k.k == e
        && k.n == kv
        && v.k == e
        && v.n == kv
        && pos0.checked_add(nt) == Some(t_total);
    if !ok_shape {
        pf2_note(1);
        return false;
    }
    if !attn_prefill_shape_ok(nt, dh, t_total, ATTN_Q4_PF_MIN_ROWS) {
        pf2_note(1);
        return false;
    }
    let Some(kr) = pf2_kernels() else {
        pf2_note(1);
        return false;
    };
    let Some(ctx) = shared_ctx() else {
        pf2_note(1);
        return false;
    };
    if ctx.make_current().is_err() {
        pf2_note(1);
        return false;
    }
    let (Some((pq, pq_k)), Some((pk, pk_k)), Some((pv, pv_k))) =
        (pf_weight(q, false), pf_weight(k, false), pf_weight(v, false))
    else {
        pf2_note(1);
        return false;
    };
    let r = PF2_DEV.with(|cell| -> Result<()> {
        let mut d = cell.borrow_mut();
        // task 230: 直前の mix を D2H していなければ (`DENNO_PF_MIXKEEP`)、
        // デバイスの `mixed` をそのまま GEMM の入力にする (H2D 1 本を消す)。
        // 値は同じバイト列。使えないときだけ従来どおりホストから上げる。
        let xptr = match pf2_dev_x(x, nt, e) {
            Some(p) => p,
            None => {
                pf2_ensure(&mut d.x, nt * e * 4)?;
                let xb = d.x.as_ref().expect("ensure 済み");
                xb.copy_from(x)?;
                xb.ptr()
            }
        };
        pf2_ensure(&mut d.qg, nt * 2 * qd * 4)?;
        pf2_ensure(&mut d.q, nt * qd * 4)?;
        pf2_ensure(&mut d.kd, nt * kv * 4)?;
        pf2_ensure(&mut d.vd, nt * kv * 4)?;
        pf2_ensure(&mut d.cs, nt * n_rot * 4)?;
        pf2_ensure(&mut d.qn, dh * 4)?;
        pf2_ensure(&mut d.kn, dh * 4)?;
        let qgb = d.qg.as_ref().expect("ensure 済み");
        let qb = d.q.as_ref().expect("ensure 済み");
        let kdb = d.kd.as_ref().expect("ensure 済み");
        let vdb = d.vd.as_ref().expect("ensure 済み");
        let csb = d.cs.as_ref().expect("ensure 済み");
        let qnb = d.qn.as_ref().expect("ensure 済み");
        let knb = d.kn.as_ref().expect("ensure 済み");
        csb.copy_from(cs)?;
        qnb.copy_from(q_norm)?;
        knb.copy_from(k_norm)?;
        // 1. q / k / v の射影 (デバイス。同期しない)
        pf2_gemm(ctx, pq, pq_k, xptr, qgb.ptr(), nt, e, 2 * qd)?;
        pf2_gemm(ctx, pk, pk_k, xptr, kdb.ptr(), nt, e, kv)?;
        pf2_gemm(ctx, pv, pv_k, xptr, vdb.ptr(), nt, e, kv)?;
        // 2. q: [q|gate] 融合の生出力 → (norm+rope 済み q) へ詰め替え
        {
            let shmem = (dh * 4) as u32;
            let (mut src, mut dst, mut w, mut tab) =
                (qgb.ptr(), qb.ptr(), qnb.ptr(), csb.ptr());
            let (mut x_stride, mut hstride, mut out_stride) =
                ((2 * qd) as c_int, (2 * dh) as c_int, qd as c_int);
            let (mut cols, mut nrot) = (dh as c_int, n_rot as c_int);
            let mut eps_c = eps;
            let mut args = [
                &mut src as *mut _ as *mut c_void,
                &mut x_stride as *mut _ as *mut c_void,
                &mut hstride as *mut _ as *mut c_void,
                &mut w as *mut _ as *mut c_void,
                &mut tab as *mut _ as *mut c_void,
                &mut dst as *mut _ as *mut c_void,
                &mut out_stride as *mut _ as *mut c_void,
                &mut cols as *mut _ as *mut c_void,
                &mut nrot as *mut _ as *mut c_void,
                &mut eps_c as *mut _ as *mut c_void,
            ];
            unsafe {
                ctx.launch_async(
                    kr.qknorm_rope,
                    (nt as u32, heads as u32, 1),
                    (dh as u32, 1, 1),
                    shmem,
                    &mut args,
                )?;
            }
        }
        // 3. k: 同じカーネルを in-place で (kv_heads 本)
        {
            let shmem = (dh * 4) as u32;
            let (mut src, mut dst, mut w, mut tab) =
                (kdb.ptr(), kdb.ptr(), knb.ptr(), csb.ptr());
            let (mut x_stride, mut hstride, mut out_stride) =
                (kv as c_int, dh as c_int, kv as c_int);
            let (mut cols, mut nrot) = (dh as c_int, n_rot as c_int);
            let mut eps_c = eps;
            let mut args = [
                &mut src as *mut _ as *mut c_void,
                &mut x_stride as *mut _ as *mut c_void,
                &mut hstride as *mut _ as *mut c_void,
                &mut w as *mut _ as *mut c_void,
                &mut tab as *mut _ as *mut c_void,
                &mut dst as *mut _ as *mut c_void,
                &mut out_stride as *mut _ as *mut c_void,
                &mut cols as *mut _ as *mut c_void,
                &mut nrot as *mut _ as *mut c_void,
                &mut eps_c as *mut _ as *mut c_void,
            ];
            unsafe {
                ctx.launch_async(
                    kr.qknorm_rope,
                    (nt as u32, kv_heads as u32, 1),
                    (dh as u32, 1, 1),
                    shmem,
                    &mut args,
                )?;
            }
        }
        // 4. task 181: ミラーへは**チャンクの行だけ**を直接書く (D2D)。
        //    ミラーが行 [0, pos0) を持っていない (最初のチャンク・世代が変わった)
        //    ときは `Err` が返るので、`pf_full_out` が従来の H2D 経路へ戻る。
        //    ここで失敗しても値は壊れない (ホストの KV は既に積んである)。
        let mut staged = None;
        if pf_mirror_delta_on() {
            match q4_attn_mirror_stage_dev(
                cache_id, n_layer, kv, layer, gen, pos0, t_total, kdb.ptr(), vdb.ptr(),
            ) {
                Ok(_) => staged = Some((cache_id, layer, gen, t_total)),
                Err(e) => pf2_warn(&format!("full qkv: ミラーへの直接書き: {e:#}")),
            }
        }
        // 5. k / v だけ戻す (ホストの KV が正)
        //    task 369: ミラーへ直書きできたときは non-blocking の `DtoHAsync` で
        //    pinned に積む (`DENNO_PF_KV_ASYNC`)。待つのは `AttnCache::push` の
        //    直前 (`pf_kv_wb_copy_into`) まで。同期 `cuMemcpyDtoH` は呼び出しで
        //    コンテキスト全体を待つので、168 回の待ちが full 段の露出になっていた。
        let mut async_started = false;
        if staged.is_some() && pf_kv_async_on() {
            match pf_kv_wb_begin(kdb.ptr(), vdb.ptr(), nt * kv * 4, nt * kv * 4, layer, nt) {
                Ok(true) => async_started = true,
                Ok(false) => {}
                Err(e) => pf2_warn(&format!("full qkv: KV の非同期 D2H: {e:#}")),
            }
        }
        if !async_started {
            // task 334: `DENNO_PF_ATTN_NOSYNC` なら待たない (直後の `to_slice` が
            // 同じストリームの完了を待つ同期 D2H)。
            pf_chain_sync(&ctx)?;
            //    task 181: `DENNO_PF_D2H_DIRECT` なら中間 Vec を作らず caller の
            //    バッファへ直接 D2H する (値は同じ。毎チャンク 2 x 1 MB の確保を消す)。
            if pf_d2h_direct_on() {
                kdb.to_slice(&mut k_out[..nt * kv])?;
                vdb.to_slice(&mut v_out[..nt * kv])?;
            } else {
                let kgot: Vec<f32> = kdb.to_vec(nt * kv)?;
                k_out.copy_from_slice(&kgot);
                let vgot: Vec<f32> = vdb.to_vec(nt * kv)?;
                v_out.copy_from_slice(&vgot);
            }
        }
        d.mir = staged;
        Ok(())
    });
    match r {
        Ok(()) => {
            pf2_note(0);
            true
        }
        Err(e) => {
            pf2_warn(&format!("full qkv: {e:#}"));
            pf2_note(1);
            false
        }
    }
}


// ---------------------------------------------------------------------------
// task 183: 融合済み qkv (ファイル上で q|gate|k|v が 1 本) の full 層 1 チャンク
// ---------------------------------------------------------------------------
//
// lab t181c (14,018 トークン cold prefill 53.18 s、既定) の台帳では full が
// 6,964 ms (gpu 3,059 / cpu 2,922 / h2d 7.7 GB / d2h 10.7 GB) で、d2h の
// 10.68 GB は **qkv の射影出力 8.96 GB + 出力投影 1.72 GB** とぴったり一致する
// (12 層 x 14,018 トークン x (13,312 + 2,560) x 4 B)。内訳は REPORT_P183.md の表。
//
// 12 本の full 層のうち 8 本は attn_q が k=2560 n=13312 の 1 本に融合していて
// `pf_full_qkv` の「行列ごとに重みポインタが取れる」前提を満たさない
// (`FusedLinear::is_fused()` が真 → br172 の連鎖は断念していた)。ここでは
// **融合 1 本を 1 回の GEMM で撃ち**、q / k / gate / v を**行オフセット**で読む。
// q は norm+rope 済みでデバイスに残し、k / v は詰めてミラーとホストへ、
// gate は `pf_full_out` が `dno_pf3_gate_mul` で読む (ld = 融合の行数)。
//
// 数値: GEMM は他の pf 連鎖と同じ BF16 テンソルコア (prefill の射影はビット一致
// 契約の外)。qknorm/rope は pf2 のカーネル (CPU と同じ式)。ゲートと gather は
// kernels_pf3.cu (-fmad=false、明示 __*_rn)。丸めだけの差で md5 は変わりうる。

/// task 183: 融合 qkv の後処理カーネル (`kernels_pf3.cu`。PTX は実行時読み込み)。
pub(crate) struct Pf3Kernels {
    /// モジュールは生かしておく必要がある (関数ハンドルの持ち主)。
    #[allow(dead_code)]
    module: CUmodule,
    gate_mul: CUfunction,
    gather: CUfunction,
}


unsafe impl Send for Pf3Kernels {}

unsafe impl Sync for Pf3Kernels {}


/// PTX の置き場所。`DENNO_PF3_PTX` があればそれ、無ければクレートの
/// `kernels/kernels_pf3.ptx`。
fn pf3_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_PF3_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_pf3.ptx"),
    }
}


/// カーネルを取得する (初回だけ PTX を読む)。失敗は None を焼き付ける。
/// **current context を張った後に呼ぶこと**。
fn pf3_kernels() -> Option<&'static Pf3Kernels> {
    static K: OnceLock<Option<Pf3Kernels>> = OnceLock::new();
    K.get_or_init(|| {
        let path = pf3_ptx_path();
        let mut image = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                pf2_warn(&format!("pf3 PTX が読めない ({}) : {e}", path.display()));
                return None;
            }
        };
        image.push(0);
        let lib = match lib() {
            Ok(l) => l,
            Err(_) => return None,
        };
        let mut module: CUmodule = std::ptr::null_mut();
        unsafe {
            if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void) != CUDA_SUCCESS {
                pf2_warn("pf3 cuModuleLoadData 失敗");
                return None;
            }
            let mut gate_mul: CUfunction = std::ptr::null_mut();
            let mut gather: CUfunction = std::ptr::null_mut();
            for (dst, name, label) in [
                (
                    &mut gate_mul,
                    b"dno_pf3_gate_mul\0".as_ptr(),
                    "dno_pf3_gate_mul",
                ),
                (&mut gather, b"dno_pf3_gather\0".as_ptr(), "dno_pf3_gather"),
            ] {
                if (lib.module_get_function)(dst, module, name as *const c_char) != CUDA_SUCCESS {
                    pf2_warn(&format!("pf3 PTX に {label} が無い"));
                    return None;
                }
            }
            Some(Pf3Kernels {
                module,
                gate_mul,
                gather,
            })
        }
    })
    .as_ref()
}


/// task 183: 融合済み qkv 1 本の射影 (1 回の GEMM) と q / k の norm+rope。
///
/// `w` は q|gate|k|v が連続した 1 本 (`w.n` 行)。行範囲は
/// `q_off..q_off+2*qd` (q|gate)、`k_off..k_off+kv`、`v_off..v_off+kv` で、
/// `v_off + kv == w.n` を要求する。戻り値 `false` は「この形/環境では
/// 連鎖を使わない」(副作用なし。呼び出し側は従来経路をそのまま走らせる)。
///
/// 成功したら q (norm+rope 済み) と gate (生の射影) をデバイスに残す。
/// k / v は詰めた形でデバイスに置き、ホストへ戻してミラーへも D2D で書く。
#[allow(clippy::too_many_arguments)]
pub fn pf_full_qkv_fused(
    x: &[f32],
    nt: usize,
    e: usize,
    w: PfLin<'_>,
    q_off: usize,
    k_off: usize,
    v_off: usize,
    q_norm: &[f32],
    k_norm: &[f32],
    cs: &[f32],
    heads: usize,
    kv_heads: usize,
    dh: usize,
    n_rot: usize,
    eps: f32,
    cache_id: u64,
    n_layer: usize,
    layer: usize,
    gen: u64,
    t_total: usize,
    pos0: usize,
    k_out: &mut [f32],
    v_out: &mut [f32],
) -> bool {
    pf2_set_nt(nt);
    let qd = heads * dh;
    let kv = kv_heads * dh;
    let n_all = w.n;
    let ok_shape = nt >= PF2_MIN_ROWS
        && e > 0
        && dh >= ATTN_PF_BQ
        && dh <= 1024
        && n_rot % 2 == 0
        && n_rot <= dh
        && heads > 0
        && kv_heads > 0
        && kv <= 1024
        && x.len() == nt * e
        && k_out.len() == nt * kv
        && v_out.len() == nt * kv
        && q_norm.len() == dh
        && k_norm.len() == dh
        && cs.len() == nt * n_rot
        && w.k == e
        && q_off == 0
        && q_off + 2 * qd <= k_off
        && k_off + kv <= v_off
        && v_off.checked_add(kv) == Some(n_all)
        && pos0.checked_add(nt) == Some(t_total);
    if !ok_shape {
        pf2_note(1);
        return false;
    }
    if !attn_prefill_shape_ok(nt, dh, t_total, ATTN_Q4_PF_MIN_ROWS) {
        pf2_note(1);
        return false;
    }
    let Some(kr) = pf2_kernels() else {
        pf2_note(1);
        return false;
    };
    let Some(kr3) = pf3_kernels() else {
        // pf3 の PTX が無い環境 (未再生成) では従来経路へ落とす。
        pf2_note(1);
        return false;
    };
    let Some(ctx) = shared_ctx() else {
        pf2_note(1);
        return false;
    };
    if ctx.make_current().is_err() {
        pf2_note(1);
        return false;
    }
    let Some((pw, pw_k)) = pf_weight(w, false) else {
        pf2_note(1);
        return false;
    };
    let r = PF2_DEV.with(|cell| -> Result<()> {
        let mut d = cell.borrow_mut();
        // task 230: 直前の mix を D2H していなければ、デバイスの `mixed` を
        // そのまま融合 GEMM の入力にする (`pf_full_qkv` と同じ規則)。
        let xptr = match pf2_dev_x(x, nt, e) {
            Some(p) => p,
            None => {
                pf2_ensure(&mut d.x, nt * e * 4)?;
                let xb = d.x.as_ref().expect("ensure 済み");
                xb.copy_from(x)?;
                xb.ptr()
            }
        };
        pf2_ensure(&mut d.qkv, nt * n_all * 4)?;
        pf2_ensure(&mut d.q, nt * qd * 4)?;
        pf2_ensure(&mut d.kd, nt * kv * 4)?;
        pf2_ensure(&mut d.vd, nt * kv * 4)?;
        pf2_ensure(&mut d.cs, nt * n_rot * 4)?;
        pf2_ensure(&mut d.qn, dh * 4)?;
        pf2_ensure(&mut d.kn, dh * 4)?;
        let qkvb = d.qkv.as_ref().expect("ensure 済み");
        let qb = d.q.as_ref().expect("ensure 済み");
        let kdb = d.kd.as_ref().expect("ensure 済み");
        let vdb = d.vd.as_ref().expect("ensure 済み");
        let csb = d.cs.as_ref().expect("ensure 済み");
        let qnb = d.qn.as_ref().expect("ensure 済み");
        let knb = d.kn.as_ref().expect("ensure 済み");
        csb.copy_from(cs)?;
        qnb.copy_from(q_norm)?;
        knb.copy_from(k_norm)?;
        // 1. 融合 1 本の射影 (**1 回の GEMM**。同期しない)
        pf2_gemm(ctx, pw, pw_k, xptr, qkvb.ptr(), nt, e, n_all)?;
        // 2. q: 融合バッファの q_off 行から [q|gate] を読み、norm+rope して q へ
        {
            let shmem = (dh * 4) as u32;
            let (mut src, mut dst, mut wgt, mut tab) =
                (qkvb.ptr() + (q_off * 4) as CUdeviceptr, qb.ptr(), qnb.ptr(), csb.ptr());
            let (mut x_stride, mut hstride, mut out_stride) =
                (n_all as c_int, (2 * dh) as c_int, qd as c_int);
            let (mut cols, mut nrot) = (dh as c_int, n_rot as c_int);
            let mut eps_c = eps;
            let mut args = [
                &mut src as *mut _ as *mut c_void,
                &mut x_stride as *mut _ as *mut c_void,
                &mut hstride as *mut _ as *mut c_void,
                &mut wgt as *mut _ as *mut c_void,
                &mut tab as *mut _ as *mut c_void,
                &mut dst as *mut _ as *mut c_void,
                &mut out_stride as *mut _ as *mut c_void,
                &mut cols as *mut _ as *mut c_void,
                &mut nrot as *mut _ as *mut c_void,
                &mut eps_c as *mut _ as *mut c_void,
            ];
            unsafe {
                ctx.launch_async(
                    kr.qknorm_rope,
                    (nt as u32, heads as u32, 1),
                    (dh as u32, 1, 1),
                    shmem,
                    &mut args,
                )?;
            }
        }
        // 3. k: 同じカーネルで融合バッファ → 詰めた kd (out_stride = kv)
        {
            let shmem = (dh * 4) as u32;
            let (mut src, mut dst, mut wgt, mut tab) =
                (qkvb.ptr() + (k_off * 4) as CUdeviceptr, kdb.ptr(), knb.ptr(), csb.ptr());
            let (mut x_stride, mut hstride, mut out_stride) =
                (n_all as c_int, dh as c_int, kv as c_int);
            let (mut cols, mut nrot) = (dh as c_int, n_rot as c_int);
            let mut eps_c = eps;
            let mut args = [
                &mut src as *mut _ as *mut c_void,
                &mut x_stride as *mut _ as *mut c_void,
                &mut hstride as *mut _ as *mut c_void,
                &mut wgt as *mut _ as *mut c_void,
                &mut tab as *mut _ as *mut c_void,
                &mut dst as *mut _ as *mut c_void,
                &mut out_stride as *mut _ as *mut c_void,
                &mut cols as *mut _ as *mut c_void,
                &mut nrot as *mut _ as *mut c_void,
                &mut eps_c as *mut _ as *mut c_void,
            ];
            unsafe {
                ctx.launch_async(
                    kr.qknorm_rope,
                    (nt as u32, kv_heads as u32, 1),
                    (dh as u32, 1, 1),
                    shmem,
                    &mut args,
                )?;
            }
        }
        // 4. v: 融合バッファの v_off 列を詰めて vd へ (D2D の gather)
        {
            let (mut src, mut dst) =
                (qkvb.ptr() + (v_off * 4) as CUdeviceptr, vdb.ptr());
            let (mut rows_c, mut ld_c, mut off_c, mut n_c) =
                (nt as c_int, n_all as c_int, 0 as c_int, kv as c_int);
            let mut args = [
                &mut src as *mut _ as *mut c_void,
                &mut dst as *mut _ as *mut c_void,
                &mut rows_c as *mut _ as *mut c_void,
                &mut ld_c as *mut _ as *mut c_void,
                &mut off_c as *mut _ as *mut c_void,
                &mut n_c as *mut _ as *mut c_void,
            ];
            unsafe {
                ctx.launch_async(kr3.gather, (nt as u32, 1, 1), (kv as u32, 1, 1), 0, &mut args)?;
            }
        }
        // 5. ミラーへはチャンクの行だけを直接書く (D2D。`DENNO_PF_MIRROR_DELTA`)
        let mut staged = None;
        if pf_mirror_delta_on() {
            match q4_attn_mirror_stage_dev(
                cache_id, n_layer, kv, layer, gen, pos0, t_total, kdb.ptr(), vdb.ptr(),
            ) {
                Ok(_) => staged = Some((cache_id, layer, gen, t_total)),
                Err(e) => pf2_warn(&format!("full qkv (融合): ミラーへの直接書き: {e:#}")),
            }
        }
        // 6. k / v だけ戻す (ホストの KV が正。値は従来経路と同じ)
        //    task 369: ミラーへ直書きできたときは non-blocking で積む
        //    (`DENNO_PF_KV_ASYNC`。待つのは `pf_kv_wb_copy_into` まで)。
        let mut async_started = false;
        if staged.is_some() && pf_kv_async_on() {
            match pf_kv_wb_begin(kdb.ptr(), vdb.ptr(), nt * kv * 4, nt * kv * 4, layer, nt) {
                Ok(true) => async_started = true,
                Ok(false) => {}
                Err(e) => pf2_warn(&format!("full qkv (融合): KV の非同期 D2H: {e:#}")),
            }
        }
        if !async_started {
            // task 334: `DENNO_PF_ATTN_NOSYNC` なら待たない (直後の `to_slice` が
            // 同じストリームの完了を待つ同期 D2H)。
            pf_chain_sync(&ctx)?;
            if pf_d2h_direct_on() {
                kdb.to_slice(&mut k_out[..nt * kv])?;
                vdb.to_slice(&mut v_out[..nt * kv])?;
            } else {
                let kgot: Vec<f32> = kdb.to_vec(nt * kv)?;
                k_out.copy_from_slice(&kgot);
                let vgot: Vec<f32> = vdb.to_vec(nt * kv)?;
                v_out.copy_from_slice(&vgot);
            }
        }
        d.mir = staged;
        Ok(())
    });
    match r {
        Ok(()) => {
            pf2_note(0);
            true
        }
        Err(e) => {
            pf2_warn(&format!("full qkv (融合): {e:#}"));
            pf2_note(1);
            false
        }
    }
}



// ---------------------------------------------------------------------------
// task 369: full 層の K/V 書き戻しを非同期にする (`DENNO_PF_KV_ASYNC`)
// ---------------------------------------------------------------------------
//
// `pf_full_qkv` / `pf_full_qkv_fused` は k/v を `cuMemcpyDtoH` (同期) で
// ホストへ戻していた。この API は**呼び出しの時点でコンテキスト全体の完了を
// 待つ**ので、14k の prefill では「層 x チャンク」= 168 回の待ちが `full` 段の
// 露出になる (t365a b2_1 の台帳: full wall 1,157.3 / busy 793.1 /
// exposed 364.2 ms、d2h 689.01 MB / 168 呼び出し)。
//
// ホストの KV は `pf_full_out` の後、`AttnCache::push` まで読まれない
// (ミラーへ直書きできたときは `pf_full_out` も読まない)。そこで k/v を pinned へ
// `cuMemcpyDtoHAsync` (同じ null ストリーム = カーネルの後ろ) で積み、実際に
// 読む直前にイベントで待つ。**デバイスから出るバイト列は同じ**で、ホストへ
// 載る順序だけが変わる (値は不変)。
//
// ゲート: `DENNO_PF_KV_ASYNC`。**既定 on** (lab t369a で採った)。
//
// lab t369a (depth8k、同じ run の base と対で読む): 14k cold は base 7.04 /
// 6.45 s に対し 6.30 s (台帳の exposed_host は 2,867 -> 2,125 ms、gpu_busy は
// 4,907 -> 4,909 ms で不変)、3,826 cold は 2,512 / 2,524 に対し
// 2,475 / 2,469 ms、decode は 15.7-16.2 ms/tok で不変、md5 は全 COND・
// 全リクエスト SAME、errors 0、text sane。切り戻しは `DENNO_PF_KV_ASYNC=0`。
pub(crate) fn pf_kv_async_from_value(v: Option<&str>) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        None => true,
        Some(s) => !matches!(s.as_str(), "0" | "off" | "false" | "no" | ""),
    }
}


/// k/v の書き戻しを非同期にするか。
pub(crate) fn pf_kv_async_on() -> bool {
    #[cfg(test)]
    {
        match PF_KV_ASYNC_FORCE.load(std::sync::atomic::Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| pf_kv_async_from_value(std::env::var("DENNO_PF_KV_ASYNC").ok().as_deref()))
}


/// [`PF_KV_DEFER_ENV`] の解釈 (未設定・空は既定、明示的な偽値だけ off)。
pub(crate) fn pf_kv_defer_from_value(v: Option<&str>) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        None | Some("") => DEFAULT_PF_KV_DEFER,
        Some("0") | Some("off") | Some("false") | Some("no") => false,
        Some(_) => true,
    }
}


/// 待ちを後ろへずらすか。
pub(crate) fn pf_kv_defer_on() -> bool {
    #[cfg(test)]
    {
        match PF_KV_DEFER_FORCE.load(std::sync::atomic::Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| pf_kv_defer_from_value(std::env::var(PF_KV_DEFER_ENV).ok().as_deref()))
}


/// 待ちを後ろへずらしたことを数える (呼び出し側は積むのを省く)。
pub fn pf_kv_defer_note() {
    let n = PF_KV_DEFER_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    if n == 1 {
        eprintln!("[pf553] full 層 KV の書き戻しの待ちを 1 呼び出し後ろへずらす (DENNO_PF_KV_DEFER)");
    }
}


/// 待ちを後ろへずらした回数。
pub fn pf_kv_defer_count() -> u64 {
    PF_KV_DEFER_N.load(std::sync::atomic::Ordering::Relaxed)
}


/// 保留中の k/v 書き戻しがあるか。
pub fn pf_kv_wb_pending() -> bool {
    KV_WB_SC.with(|c| c.borrow().slots.iter().any(KvWbSlot::pending))
}


/// 一番古い保留の持ち主 `(層, 行数)`。無ければ `None`。
///
/// task 553: `DENNO_PF_KV_DEFER` では呼び出し側が「この層チャンクの分が
/// 保留に載ったか」をこれで確かめてから積むのを後ろへ回す。
pub fn pf_kv_wb_peek() -> Option<(usize, usize)> {
    KV_WB_SC.with(|c| {
        let s = c.borrow();
        kv_wb_oldest(&s).map(|i| s.slots[i].owner)
    })
}


/// 一番古い保留の長さ (k, v の要素数)。無ければ `None`。
pub fn pf_kv_wb_len() -> Option<(usize, usize)> {
    KV_WB_SC.with(|c| {
        let s = c.borrow();
        kv_wb_oldest(&s).map(|i| (s.slots[i].bytes.0 / 4, s.slots[i].bytes.1 / 4))
    })
}


/// `begin` の通し番号 (積むたびに増える)。
///
/// task 553: 呼び出し側は「この層チャンクの D2H を実際に積めたか」を前後の値で
/// 判定する (積めていないのに積むのを省くと、その層チャンクの KV が永久に
/// ホストへ降りず、長さと値がずれる)。持ち主だけの判定では、同じ層の同じ行数の
/// 保留と区別できない。
pub fn pf_kv_wb_seq() -> u64 {
    KV_WB_SC.with(|c| c.borrow().seq)
}


/// k/v を pinned へ非同期で積む (ストリーム順序。待たない)。
///
/// `Ok(false)` = 積まなかった (ゲート off / イベントが取れない / 空きが無い)。
pub(crate) fn pf_kv_wb_begin(
    kptr: CUdeviceptr,
    vptr: CUdeviceptr,
    k_bytes: usize,
    v_bytes: usize,
    layer: usize,
    rows: usize,
) -> Result<bool> {
    if !pf_kv_async_on() || k_bytes == 0 || v_bytes == 0 {
        return Ok(false);
    }
    let lib = lib()?;
    let (Some(create), Some(record)) = (lib.event_create, lib.event_record) else {
        return Ok(false);
    };
    let defer = pf_kv_defer_on();
    KV_WB_SC.with(|cell| -> Result<bool> {
        let mut s = cell.borrow_mut();
        let pending = [s.slots[0].pending(), s.slots[1].pending()];
        let Some(i) = kv_wb_pick_slot(pending, s.next, defer) else {
            return Ok(false);
        };
        let total = k_bytes + v_bytes;
        let need = !matches!(&s.slots[i].pin, Some(p) if p.bytes >= total);
        if need {
            s.slots[i].pin = Some(PinMem::alloc(total)?);
        }
        let base = s.slots[i].pin.as_ref().expect("確保済み").ptr as usize;
        unsafe {
            check(
                (lib.memcpy_dtoh_async)(base as *mut c_void, kptr, k_bytes, std::ptr::null_mut()),
                "cuMemcpyDtoHAsync (full k)",
            )?;
            check(
                (lib.memcpy_dtoh_async)(
                    (base + k_bytes) as *mut c_void,
                    vptr,
                    v_bytes,
                    std::ptr::null_mut(),
                ),
                "cuMemcpyDtoHAsync (full v)",
            )?;
        }
        if s.slots[i].ev.is_none() {
            let mut ev: CUevent = std::ptr::null_mut();
            if unsafe { create(&mut ev, CU_EVENT_DISABLE_TIMING) } != CUDA_SUCCESS {
                return Ok(false);
            }
            s.slots[i].ev = Some(ev);
        }
        let ev = s.slots[i].ev.expect("作成済み");
        if unsafe { record(ev, std::ptr::null_mut()) } != CUDA_SUCCESS {
            return Ok(false);
        }
        s.seq += 1;
        let seq = s.seq;
        {
            let sl = &mut s.slots[i];
            sl.bytes = (k_bytes, v_bytes);
            sl.owner = (layer, rows);
            sl.seq = seq;
        }
        if defer {
            s.next = (i + 1) % 2;
        }
        Ok(true)
    })
}


/// 保留中の転送を待って `k_out` / `v_out` へ写す。`false` = 保留なし。
pub fn pf_kv_wb_copy_into(k_out: &mut [f32], v_out: &mut [f32]) -> bool {
    KV_WB_SC.with(|cell| {
        let mut s = cell.borrow_mut();
        let Some(i) = kv_wb_oldest(&s) else {
            return false;
        };
        let (kb, vb) = s.slots[i].bytes;
        let (kf, vf) = (kb / 4, vb / 4);
        let Some(pin) = s.slots[i].pin.as_ref() else {
            s.slots[i].bytes = (0, 0);
            return false;
        };
        let base = pin.ptr as usize;
        {
            // task 505: 待ちを台帳に載せる (`full` 段の cpu 列の中身を割る)。
            let _s = crate::model::q4ledger::sub(crate::model::q4ledger::SUB_PF_KVWAIT);
            if let (Some(ev), Some(esync)) = (s.slots[i].ev, lib().ok().and_then(|l| l.event_sync)) {
                unsafe {
                    let _ = (esync)(ev);
                }
            }
        }
        s.slots[i].bytes = (0, 0);
        if k_out.len() < kf || v_out.len() < vf {
            return false;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(base as *const u8, k_out.as_mut_ptr() as *mut u8, kb);
            std::ptr::copy_nonoverlapping(
                (base + kb) as *const u8,
                v_out.as_mut_ptr() as *mut u8,
                vb,
            );
        }
        true
    })
}


/// [`PF_KV_PUSH_ENV`] の解釈 (明示的な真値だけ on)。
pub(crate) fn pf_kv_push_from_value(v: Option<&str>) -> bool {
    matches!(
        v.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("1") | Some("on") | Some("true") | Some("yes")
    )
}


/// pinned から直接 push するか (既定 off)。
pub fn pf_kv_push_on() -> bool {
    #[cfg(test)]
    {
        match PF_KV_PUSH_FORCE.load(std::sync::atomic::Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| pf_kv_push_from_value(std::env::var(PF_KV_PUSH_ENV).ok().as_deref()))
}


/// 保留中の k/v を **pinned のまま** `f` に渡す (コピーしない)。
///
/// `false` = 保留なし (呼び出し側は従来の `ks`/`vs` 経路へ)。
pub fn pf_kv_wb_with<F: FnOnce(&[f32], &[f32])>(f: F) -> bool {
    KV_WB_SC.with(|cell| {
        let mut s = cell.borrow_mut();
        let Some(i) = kv_wb_oldest(&s) else {
            return false;
        };
        let (kb, vb) = s.slots[i].bytes;
        let Some(pin) = s.slots[i].pin.as_ref() else {
            s.slots[i].bytes = (0, 0);
            return false;
        };
        let base = pin.ptr as usize;
        // task 505: 待ちを台帳に載せる (`full` 段の cpu 列の中身を割る)。
        {
            let _s = crate::model::q4ledger::sub(crate::model::q4ledger::SUB_PF_KVWAIT);
            if let (Some(ev), Some(esync)) = (s.slots[i].ev, lib().ok().and_then(|l| l.event_sync)) {
                unsafe {
                    let _ = (esync)(ev);
                }
            }
        }
        s.slots[i].bytes = (0, 0);
        // SAFETY: pinned は `kb + vb` バイト以上あり、転送はイベントで完了済み。
        let k = unsafe { std::slice::from_raw_parts(base as *const f32, kb / 4) };
        let v = unsafe { std::slice::from_raw_parts((base + kb) as *const f32, vb / 4) };
        f(k, v);
        true
    })
}


/// 保留を捨てる (デバイスのコピー完了は待つ)。`false` = 保留なし。
pub fn pf_kv_wb_discard() -> bool {
    KV_WB_SC.with(|cell| {
        let mut s = cell.borrow_mut();
        let mut any = false;
        for i in 0..s.slots.len() {
            if !s.slots[i].pending() {
                continue;
            }
            any = true;
            if let (Some(ev), Some(esync)) = (s.slots[i].ev, lib().ok().and_then(|l| l.event_sync)) {
                unsafe {
                    let _ = (esync)(ev);
                }
            }
            s.slots[i].bytes = (0, 0);
        }
        any
    })
}


/// `pf_full_out` がミラーから読める状態か (= ホスト KV を読まない)。
///
/// `pf_full_qkv` が記録した `mir` が一致し、かつミラーが生きているときだけ
/// `true`。呼び出し側はこのときだけ `AttnCache::push` を後ろへ回してよい。
/// 偽を返したときは従来どおり「ホスト KV を先に積む」経路へ戻る (安全側)。
pub fn pf_full_mirror_direct(
    cache_id: u64,
    layer: usize,
    gen: u64,
    t_total: usize,
    kv_dim: usize,
) -> bool {
    if !pf_kv_async_on() {
        return false;
    }
    let staged = PF2_DEV.with(|c| c.borrow().mir);
    let direct = matches!(
        staged,
        Some((cid, li, g, tt)) if cid == cache_id && li == layer && g == gen && tt == t_total
    );
    direct && q4_attn_mirror_ready(cache_id, kv_dim, layer, gen, t_total).is_some()
}


/// task 172: full 層 1 チャンクの後半 (attention → 出力ゲート → 出力投影)。
///
/// `pf_full_qkv` がデバイスに残した q / qg を使う。K/V はミラーから読む
/// (`k_cpu` / `v_cpu` は台帳を進めるためだけに渡す。既に積んであるので転送は起きない)。
/// 前提の形。`keep_dev` のときは `out` を書かない (`out` は空でよい)。
#[allow(clippy::too_many_arguments)]
pub fn pf_full_out(
    cache_id: u64,
    n_layer: usize,
    layer: usize,
    gen: u64,
    t_total: usize,
    pos0: usize,
    nt: usize,
    heads: usize,
    group: usize,
    dh: usize,
    kv_dim: usize,
    scale: f32,
    k_cpu: &[f32],
    v_cpu: &[f32],
    wo: PfLin<'_>,
    e: usize,
    // task 183: 融合 qkv 経路のとき true。gate を `d.qkv` の
    // `gate_off` 行から `gate_ld` ストライドで読む (pf3 のカーネル)。
    fused_gate: bool,
    gate_off: usize,
    gate_ld: usize,
    // task 308: 層出力をデバイスに残す (D2H しない)。直後の combine が
    // `pf_attn_dev_store` の記録を読む (`DENNO_PF_FULL_DEVRES`)。
    keep_dev: bool,
    out: &mut [f32],
) -> bool {
    pf2_set_nt(nt);
    let qd = heads * dh;
    // task 369: ミラーへ直書きできていればここでホスト KV を読まない。
    // `DENNO_PF_KV_ASYNC` は k/v の D2H を `AttnCache::push` の直前まで遅らせる
    // ので、そのときはホスト KV の長さを要求しない (`k_cpu` はまだ短い)。
    let staged_mir = PF2_DEV.with(|c| c.borrow().mir);
    let mirror_direct = matches!(
        staged_mir,
        Some((cid, li, g, tt)) if cid == cache_id && li == layer && g == gen && tt == t_total
    );
    let ok_shape = nt >= PF2_MIN_ROWS
        && dh >= ATTN_PF_BQ
        && dh <= 1024
        && heads > 0
        && group > 0
        && heads % group == 0
        && kv_dim == (heads / group) * dh
        && pos0 + nt == t_total
        && (mirror_direct || k_cpu.len() >= t_total * kv_dim)
        && (mirror_direct || v_cpu.len() >= t_total * kv_dim)
        && wo.k == qd
        && wo.n == e
        && (keep_dev || out.len() == nt * e);
    if !ok_shape {
        pf2_note(1);
        return false;
    }
    if !attn_prefill_shape_ok(nt, dh, t_total, ATTN_Q4_PF_MIN_ROWS) {
        pf2_note(1);
        return false;
    }
    let Some(kr) = pf2_kernels() else {
        pf2_note(1);
        return false;
    };
    let Some(ctx) = shared_ctx() else {
        pf2_note(1);
        return false;
    };
    if ctx.make_current().is_err() {
        pf2_note(1);
        return false;
    }
    let Some((pw, pw_k)) = pf_weight(wo, false) else {
        pf2_note(1);
        return false;
    };
    let r = (|| -> Result<()> {
        // task 181: `pf_full_qkv` がミラーへ直接書いていれば H2D をしない
        // (`DENNO_PF_MIRROR_DELTA`)。台帳が一致しないときは従来の差分 H2D。
        // task 369: 判定は `ok_shape` の前へ出した (上の `mirror_direct`)。
        let (kptr, vptr) = match mirror_direct {
            true => match q4_attn_mirror_ready(cache_id, kv_dim, layer, gen, t_total) {
                Some(p) => p,
                // 台帳は合うがバッファが無い (LRU で落ちた等) → 従来経路で作り直す。
                None => {
                    // task 369: `DENNO_PF_KV_ASYNC` で push を後ろへ回している
                    // ときはホスト KV がまだ短い。短いままで作り直すと足りない
                    // 行を積むので、ここで諦めて呼び出し側を従来経路へ戻す
                    // (`ok_shape` の長さ検査はミラー直書きのとき外している)。
                    if k_cpu.len() < t_total * kv_dim || v_cpu.len() < t_total * kv_dim {
                        bail!("full out: ミラーが落ちた (ホスト KV も未確定)");
                    }
                    q4_attn_mirror_prepare(
                        cache_id, n_layer, kv_dim, layer, gen, t_total, k_cpu, v_cpu, true,
                    )?
                }
            },
            false => q4_attn_mirror_prepare(
                cache_id, n_layer, kv_dim, layer, gen, t_total, k_cpu, v_cpu, true,
            )?,
        };
        PF2_DEV.with(|cell| -> Result<()> {
            let mut d = cell.borrow_mut();
            pf2_ensure(&mut d.att, nt * qd * 4)?;
            pf2_ensure(&mut d.o, nt * e * 4)?;
            // task 183: 融合 qkv 経路では gate が `d.qkv` の中にあるので、qg は
            // 要らない (q は norm+rope 済みの詰めたバッファ)。
            let qb = d.q.as_ref().ok_or_else(|| anyhow!("q が無い"))?;
            if fused_gate {
                let qkvb = d.qkv.as_ref().ok_or_else(|| anyhow!("qkv が無い"))?;
                if qkvb.bytes() < nt * gate_ld * 4 || qb.bytes() < nt * qd * 4 {
                    bail!("q / qkv の作業域が小さい (融合 qkv を先に呼ぶこと)");
                }
            } else {
                let qgb = d.qg.as_ref().ok_or_else(|| anyhow!("qg が無い"))?;
                if qgb.bytes() < nt * 2 * qd * 4 || qb.bytes() < nt * qd * 4 {
                    bail!("q / qg の作業域が小さい (qkv を先に呼ぶこと)");
                }
            }
            let attb = d.att.as_ref().expect("ensure 済み");
            let ob = d.o.as_ref().expect("ensure 済み");
            // 1. attention (K/V はミラー、q はデバイス)
            {
                // task 222: v2 カーネル (スコアを bf16 mma) が使える形ならそちら。
                // 形は dh == 256 固定 (スレッド配置)・PTX があること・ゲート on。
                // task 230: v3 (PV も bf16 mma・head 優先グリッド) が使えるなら
                // 最優先。どれも使えなければ従来 (v1) へ fail closed。
                let use_v3 = pf_attn_v3_on()
                    && dh == ATTN_PF3_DH
                    && qd == heads * dh
                    && !ctx.f_attn_pf3.is_null();
                let use_v2 = !use_v3
                    && pf_attn_tc_on()
                    && dh == ATTN_PF2_DH
                    && qd == heads * dh
                    && !ctx.f_attn_pf2.is_null();
                let (mut qp, mut kp, mut vp, mut ap) =
                    (qb.ptr(), kptr, vptr, attb.ptr());
                // task 465: K/V を bf16 にした v4 が撃てるなら最優先。撃てなければ
                // v3 -> v2 -> 従来へ fail closed (値はどれも同じ丸め順)。
                let v4kv = if dh == ATTN_PF4_DH && qd == heads * dh && nt >= ATTN_PF4_BQ {
                    pf_attn_v4_kv_prepare(kptr, vptr, pos0, nt, t_total, kv_dim)
                } else {
                    None
                };
                if let Some((kb, vb)) = v4kv {
                    kp = kb;
                    vp = vb;
                }
                let (mut heads_c, mut n_c, mut pos0_c) =
                    (heads as c_int, nt as c_int, pos0 as c_int);
                let (mut kvd_c, mut t_c, mut group_c, mut dh_c) = (
                    kv_dim as c_int,
                    t_total as c_int,
                    group as c_int,
                    dh as c_int,
                );
                let mut scale_c = scale;
                let mut args = [
                    &mut qp as *mut _ as *mut c_void,
                    &mut kp as *mut _ as *mut c_void,
                    &mut vp as *mut _ as *mut c_void,
                    &mut ap as *mut _ as *mut c_void,
                    &mut heads_c as *mut _ as *mut c_void,
                    &mut n_c as *mut _ as *mut c_void,
                    &mut pos0_c as *mut _ as *mut c_void,
                    &mut kvd_c as *mut _ as *mut c_void,
                    &mut t_c as *mut _ as *mut c_void,
                    &mut group_c as *mut _ as *mut c_void,
                    &mut dh_c as *mut _ as *mut c_void,
                    &mut scale_c as *mut _ as *mut c_void,
                ];
                let v4f = if v4kv.is_some() { pf_attn_pf4_kernels() } else { None };
                if let Some(fns) = v4f {
                    unsafe {
                        ctx.launch_async(
                            fns.attn,
                            (heads as u32, nt.div_ceil(ATTN_PF4_BQ) as u32, 1),
                            (ATTN_PF4_THREADS as u32, 1, 1),
                            ATTN_PF4_SMEM_BYTES as u32,
                            &mut args,
                        )?;
                    }
                } else if use_v3 {
                    unsafe {
                        ctx.launch_async(
                            ctx.f_attn_pf3,
                            (heads as u32, nt.div_ceil(ATTN_PF3_BQ) as u32, 1),
                            (dh as u32, 1, 1),
                            ATTN_PF3_SMEM_BYTES as u32,
                            &mut args,
                        )?;
                    }
                } else if use_v2 {
                    unsafe {
                        ctx.launch_async(
                            ctx.f_attn_pf2,
                            (nt.div_ceil(ATTN_PF2_BQ) as u32, heads as u32, 1),
                            (dh as u32, 1, 1),
                            ATTN_PF2_SMEM_BYTES as u32,
                            &mut args,
                        )?;
                    }
                } else {
                    unsafe {
                        ctx.launch_async(
                            ctx.f_attn_pf,
                            (nt.div_ceil(ATTN_PF_BQ) as u32, heads as u32, 1),
                            (dh as u32, 1, 1),
                            (attn_pf_sh_f(dh) * 4) as u32,
                            &mut args,
                        )?;
                    }
                }
            }
            // 2. 出力ゲート (att *= sigmoid(gate))
            if fused_gate {
                // task 183: 融合バッファ (ld = 融合 1 本の行数) から読む。
                let kr3 = pf3_kernels().ok_or_else(|| anyhow!("pf3 のカーネルが無い"))?;
                let qkvb = d.qkv.as_ref().ok_or_else(|| anyhow!("qkv が無い"))?;
                let (mut ap, mut gptr) =
                    (attb.ptr(), qkvb.ptr() + (gate_off * 4) as CUdeviceptr);
                let (mut h_c, mut dh_c, mut ld_c) =
                    (heads as c_int, dh as c_int, gate_ld as c_int);
                let mut args = [
                    &mut ap as *mut _ as *mut c_void,
                    &mut gptr as *mut _ as *mut c_void,
                    &mut h_c as *mut _ as *mut c_void,
                    &mut dh_c as *mut _ as *mut c_void,
                    &mut ld_c as *mut _ as *mut c_void,
                ];
                unsafe {
                    ctx.launch_async(
                        kr3.gate_mul,
                        (nt as u32, heads as u32, 1),
                        (dh as u32, 1, 1),
                        0,
                        &mut args,
                    )?;
                }
            } else {
                let qgb = d.qg.as_ref().ok_or_else(|| anyhow!("qg が無い"))?;
                let (mut ap, mut gptr) = (attb.ptr(), qgb.ptr());
                let (mut h_c, mut dh_c) = (heads as c_int, dh as c_int);
                let mut args = [
                    &mut ap as *mut _ as *mut c_void,
                    &mut gptr as *mut _ as *mut c_void,
                    &mut h_c as *mut _ as *mut c_void,
                    &mut dh_c as *mut _ as *mut c_void,
                ];
                unsafe {
                    ctx.launch_async(
                        kr.gate_mul,
                        (nt as u32, heads as u32, 1),
                        (dh as u32, 1, 1),
                        0,
                        &mut args,
                    )?;
                }
            }
            // 3. 出力投影 (同期しない) → 最後に 1 回だけ待って 1 回で戻す
            pf2_gemm(ctx, pw, pw_k, attb.ptr(), ob.ptr(), nt, qd, e)?;
            // task 334: `DENNO_PF_ATTN_NOSYNC` なら待たない (直後の D2H / combine は
            // 同じストリーム)。
            pf_chain_sync(ctx)?;
            // task 308: 層出力をデバイスに残す。直後の combine が同じポインタを
            // 読むので、D2H (層出力) と combine の H2D が 1 往復ぶん消える
            // (`DENNO_PF_FULL_DEVRES`。呼び出し側が keep_dev を立てたときだけ)。
            if keep_dev {
                pf_attn_dev_store(ob.ptr(), nt, e);
                return Ok(());
            }
            // task 181: `DENNO_PF_D2H_DIRECT` は中間 Vec を作らない (値は同じ)。
            if pf_d2h_direct_on() {
                ob.to_slice(&mut out[..nt * e])?;
            } else {
                let got: Vec<f32> = ob.to_vec(nt * e)?;
                out.copy_from_slice(&got);
            }
            Ok(())
        })
    })();
    // task 181: 直接書きの印は 1 チャンク限り (次のチャンクへ持ち越さない)。
    PF2_DEV.with(|c| c.borrow_mut().mir = None);
    match r {
        Ok(()) => {
            pf2_note(0);
            true
        }
        Err(e) => {
            pf2_warn(&format!("full out: {e:#}"));
            pf2_note(1);
            false
        }
    }
}


// ---------------------------------------------------------------------------
// shared expert (gate → silu*up → down) をデバイスで閉じる
// ---------------------------------------------------------------------------

/// task 172: shared expert の 3 射影を 1 チャンクぶんまとめてデバイスで回す。
///
/// `out` には `down_shexp(silu(gate_shexp(x)) * up_shexp(x))` を書く。ゲート
/// (`gate_inp_shexp`) と sigmoid の足し込みは呼び出し側が従来どおり CPU でやる
/// (重みが 1 行なので GPU に載せる意味が無い)。
#[allow(clippy::too_many_arguments)]
pub fn pf_shared(
    x: &[f32],
    nt: usize,
    e: usize,
    sff: usize,
    gate: PfLin<'_>,
    up: PfLin<'_>,
    down: PfLin<'_>,
    out: &mut [f32],
) -> bool {
    if x.len() != nt.saturating_mul(e) {
        pf2_note(3);
        return false;
    }
    pf_shared_impl(SharedIn::Host(x), nt, e, sff, gate, up, down, SharedOut::Host(out))
}


/// task 209: 入力 `x` が既にデバイスにある版 (`[nt][e]` f32)。
///
/// router と同じく、`pf_mix` の出力 (`mixed`) をそのまま読む。ホスト版と
/// **同じカーネル・同じ入力バイト列**なので値は変わらない。
pub fn pf_shared_dev(
    dev_x: CUdeviceptr,
    nt: usize,
    e: usize,
    sff: usize,
    gate: PfLin<'_>,
    up: PfLin<'_>,
    down: PfLin<'_>,
    out: &mut [f32],
) -> bool {
    if dev_x == 0 {
        pf2_note(3);
        return false;
    }
    pf_shared_impl(SharedIn::Dev(dev_x), nt, e, sff, gate, up, down, SharedOut::Host(out))
}


/// task 233: 入力も出力もデバイスにある版 (層優先 prefill の shared expert)。
///
/// 出力は `[nt][e]` f32。MoE の作業域の「共有 expert の行」へ直接書く
/// (呼び出し側は [`pf_moe_shared_tail`] のポインタを渡す)。
#[allow(clippy::too_many_arguments)]
pub fn pf_shared_dev_to(
    dev_x: CUdeviceptr,
    nt: usize,
    e: usize,
    sff: usize,
    gate: PfLin<'_>,
    up: PfLin<'_>,
    down: PfLin<'_>,
    out_dev: CUdeviceptr,
) -> bool {
    if dev_x == 0 || out_dev == 0 {
        pf2_note(3);
        return false;
    }
    pf_shared_impl(
        SharedIn::Dev(dev_x),
        nt,
        e,
        sff,
        gate,
        up,
        down,
        SharedOut::Dev(out_dev),
    )
}


#[allow(clippy::too_many_arguments)]
fn pf_shared_impl(
    src: SharedIn<'_>,
    nt: usize,
    e: usize,
    sff: usize,
    gate: PfLin<'_>,
    up: PfLin<'_>,
    down: PfLin<'_>,
    out: SharedOut<'_>,
) -> bool {
    pf2_set_nt(nt);
    let ok_shape = nt >= PF2_MIN_ROWS
        && e > 0
        && sff > 0
        && out.host_len().map_or(true, |n| n == nt * e)
        && gate.k == e
        && gate.n == sff
        && up.k == e
        && up.n == sff
        && down.k == sff
        && down.n == e;
    if !ok_shape {
        pf2_note(3);
        return false;
    }
    let Some(ctx) = shared_ctx() else {
        pf2_note(3);
        return false;
    };
    if ctx.make_current().is_err() {
        pf2_note(3);
        return false;
    }
    let (Some((pg, pg_k)), Some((pu, pu_k)), Some((pd, pd_k))) =
        (pf_weight(gate, false), pf_weight(up, false), pf_weight(down, false))
    else {
        pf2_note(3);
        return false;
    };
    let r = PF2_DEV.with(|cell| -> Result<()> {
        let mut d = cell.borrow_mut();
        // task 257: shared expert の作業域もプールから (タイルの行数で足りる)。
        ensure_region(&mut d.x, crate::model::pf_pool::R::ShX, 0, nt * e * 4)?;
        ensure_region(&mut d.ga, crate::model::pf_pool::R::ShGa, 0, nt * sff * 4)?;
        ensure_region(&mut d.gu, crate::model::pf_pool::R::ShGu, 0, nt * sff * 4)?;
        // task 233: デバイス出力のときは `sh` (ホストへ戻すための中間) を確保しない。
        if matches!(out, SharedOut::Host(_)) {
            ensure_region(&mut d.sh, crate::model::pf_pool::R::ShSh, 0, nt * e * 4)?;
        }
        // swiglu の出力は x の領域へ (gate/up を撃ったあとの x はもう不要。
        // ga を in-place で上書きすると別名になるので避ける)。
        let xb = d.x.as_ref().expect("ensure 済み");
        let gab = d.ga.as_ref().expect("ensure 済み");
        let gub = d.gu.as_ref().expect("ensure 済み");
        let swb = d.x.as_ref().expect("ensure 済み");
        // task 209: デバイス入力のときは H2D をせず、その領域を直接読む
        // (swiglu の出力は別の `d.x` に書くので入力は壊れない)。
        let xp = match src {
            SharedIn::Host(x) => {
                xb.copy_from(x)?;
                xb.ptr()
            }
            SharedIn::Dev(p) => p,
        };
        // task 183: `DENNO_PF_TC_SHARED2` が on なら BM=64 のカーネルで撃つ。
        pf2_gemm_shared(ctx, pg, pg_k, xp, gab.ptr(), nt, e, sff)?;
        pf2_gemm_shared(ctx, pu, pu_k, xp, gub.ptr(), nt, e, sff)?;
        // swiglu: x = silu(ga) * gu (elem の既存カーネル。CPU と同じ式)
        {
            let (mut gp, mut up_p, mut op) = (gab.ptr(), gub.ptr(), swb.ptr());
            let mut total = (nt * sff) as c_int;
            let grid = ((nt * sff).div_ceil(256)) as u32;
            let mut args = [
                &mut gp as *mut _ as *mut c_void,
                &mut up_p as *mut _ as *mut c_void,
                &mut op as *mut _ as *mut c_void,
                &mut total as *mut _ as *mut c_void,
            ];
            unsafe {
                ctx.launch_async(ctx.f_swiglu, (grid, 1, 1), (256, 1, 1), 0, &mut args)?;
            }
        }
        // task 233: デバイス出力のときは down の結果を**そのまま**宛先へ書く
        // (置き場へ D2D で写し直さない)。ホスト出力は従来どおり `d.sh` 経由。
        let dst = match &out {
            SharedOut::Host(_) => d.sh.as_ref().expect("ensure 済み").ptr(),
            SharedOut::Dev(p) => *p,
        };
        pf2_gemm_shared(ctx, pd, pd_k, swb.ptr(), dst, nt, sff, e)?;
        // task 334: `DENNO_PF_ATTN_NOSYNC` なら待たない (ホスト出力は同期 D2H、
        // デバイス出力は同じストリームの後続が読む)。
        pf_chain_sync(ctx)?;
        if let SharedOut::Host(o) = out {
            let shb = d.sh.as_ref().expect("ensure 済み");
            // task 181: ホストへは**呼び出し側のバッファへ直接**戻す
            // (`DENNO_PF_D2H_DIRECT`)。`to_vec` + `copy_from_slice` は毎層 5 MB の
            // Vec 確保 (= ページフォルト) と memcpy をもう 1 回する。
            if pf_d2h_direct_on() {
                shb.to_slice(&mut o[..nt * e])?;
            } else {
                let got: Vec<f32> = shb.to_vec(nt * e)?;
                o.copy_from_slice(&got);
            }
        }
        Ok(())
    });
    match r {
        Ok(()) => {
            pf2_note(2);
            true
        }
        Err(e) => {
            pf2_warn(&format!("shared: {e:#}"));
            pf2_note(3);
            false
        }
    }
}


// ---------------------------------------------------------------------------
// PLE の key / value 射影をデバイスで回す
// ---------------------------------------------------------------------------

/// task 172: PLE の key / value 射影 (層 1 の 1 チャンク) を GPU で回す。
///
/// 表の pread は SSD なので CPU のまま (gather は呼び出し側)。ここは
/// `gemv_batch_cpu` を置き換えるだけで、値は gemv_batch_cpu と同じ行独立の和。
pub fn pf_ple_proj(
    x: &[f32],
    nt: usize,
    e: usize,
    key: PfLin<'_>,
    value: PfLin<'_>,
    key_out: &mut [f32],
    value_out: &mut [f32],
) -> bool {
    pf2_set_nt(nt);
    let ok_shape = nt > 0
        && e > 0
        && x.len() == nt * e
        && key.k == e
        && key_out.len() == nt * key.n
        && value.k == e
        && value_out.len() == nt * value.n;
    if !ok_shape {
        pf2_note(5);
        return false;
    }
    let Some(ctx) = shared_ctx() else {
        pf2_note(5);
        return false;
    };
    if ctx.make_current().is_err() {
        pf2_note(5);
        return false;
    }
    // 量子型 (Q8_0 など) だけを受ける。F32 / BF16 は pf_weight が VRAM に
    // 載せてしまうので、形の段階で断る (このモデルの ple_key/ple_value は F32)。
    if !matches!(key.ty, crate::kquant::GgmlType::Q8_0 | crate::kquant::GgmlType::Q5_1 | crate::kquant::GgmlType::Q4K | crate::kquant::GgmlType::Q5K)
        || !matches!(value.ty, crate::kquant::GgmlType::Q8_0 | crate::kquant::GgmlType::Q5_1 | crate::kquant::GgmlType::Q4K | crate::kquant::GgmlType::Q5K)
    {
        pf2_note(5);
        return false;
    }
    let (Some((pk, pk_k)), Some((pv, pv_k))) =
        (pf_weight(key, false), pf_weight(value, false))
    else {
        pf2_note(5);
        return false;
    };
    let r = PF2_DEV.with(|cell| -> Result<()> {
        let mut d = cell.borrow_mut();
        pf2_ensure(&mut d.x, nt * e * 4)?;
        pf2_ensure(&mut d.o, nt * (key.n + value.n) * 4)?;
        let xb = d.x.as_ref().expect("ensure 済み");
        let ob = d.o.as_ref().expect("ensure 済み");
        xb.copy_from(x)?;
        pf2_gemm(ctx, pk, pk_k, xb.ptr(), ob.ptr(), nt, e, key.n)?;
        let kn = key.n;
        pf2_gemm(ctx, pv, pv_k, xb.ptr(), ob.ptr() + (nt * kn * 4) as CUdeviceptr, nt, e, value.n)?;
        ctx.sync()?;
        let kk: Vec<f32> = ob.to_vec(nt * kn)?;
        key_out.copy_from_slice(&kk);
        let vv: Vec<f32> = ob.view_at(nt * kn * 4, nt * value.n * 4).to_vec(nt * value.n)?;
        value_out.copy_from_slice(&vv);
        Ok(())
    });
    match r {
        Ok(()) => {
            pf2_note(4);
            true
        }
        Err(e) => {
            pf2_warn(&format!("ple proj: {e:#}"));
            pf2_note(5);
            false
        }
    }
}


/// s173 (P14): FFN の gate/up の与え方。
///
/// `Split` は従来どおり別々の重みで 2 回の GEMM (`CH_GA` の前半/後半へ)、
/// `Fused` は [gate 全行 | up 全行] の連結重み 1 本で 1 回の GEMM
/// (`CH_GA` へ行内 [gate|up] で書く)。GEMM の k 方向累積順は両者で同一。
pub(crate) enum PfGu<'a> {
    Split(&'a crate::quant::Linear, &'a crate::quant::Linear),
    Fused(&'a crate::quant::Linear),
}


/// s165 (P2): prefill チェーンの入力の出所。
/// `Host` は従来どおりホスト slice をタイルごとに H2D、`Slot` は
/// 「既にデバイスに居る」ことを表す — 消えるのは転送だけで、カーネル・grid・
/// 引数・演算列は 1 つも変わらない (行ごとに独立なので bit 一致)。
pub(crate) enum PfSrc<'a> {
    Host(&'a [f32]),
    /// `CH_PFH` の先頭から m 行 (rmsnorm 出力) をそのまま読む
    Slot,
}


/// s165 (P2): prefill チェーンの出力先。`Slot` はデバイス常駐バッファ
/// (`CH_DOWN` / `CH_O`) に書き、D2H と同期を行わない。
pub(crate) enum PfSink<'a> {
    Host(&'a mut [f32]),
    Slot,
}


// ---------------------------------------------------------------------------
// task 167: prefill の ssm 層 (GDN) を 1 チャンクぶんデバイスで完結させる連鎖
// ---------------------------------------------------------------------------
//
// 従来の prefill 経路 (lab t162c の台帳) は ssm 1 層 1 チャンクあたり
//   ① qkv 投影 GEMM (ホスト入力 → ホスト出力)   H2D: nt*e*4   D2H: nt*qkv_dim*4
//   ② z 投影 GEMM (同上)                        H2D: nt*e*4   D2H: nt*value_dim*4
//   ③ スキャン (ホスト入力)                     H2D: nt*(qkv+z)*4  D2H: nt*vd*4 + 状態
//   ④ out_proj GEMM (ホスト入力)                H2D: nt*vd*4  D2H: nt*e*4
// を往復する。中間 (qkv / z / y) はホストでは何にも使わないのに、毎回
// ホストへ落として上げ直していた (14k の prefill で約 105 GB)。
//
// この連鎖は ①②③④ をデバイス上でつなぎ、ホストと往復するのは
//   - 入力の活性 (nt*e*4) 1 本
//   - beta / alpha (nt*nv*4 × 2) 2 本
//   - 出力 (nt*e*4) 1 本
//   - GDN 状態のホスト写し (次の decode / CPU 経路が seed に使う)
// だけにする。カーネル・引数・丸め順は従来と同一 (同じ `dno_gemm_*` と
// 同じ `dno_gdn_scan_sig`) なので、値は 1 ビットも変わらない。
//
// 失敗の扱い: スキャンを起動する前の失敗は `Ok(false)` を返す (= 呼び出し側は
// 従来経路へ。状態は 1 つも進んでいない)。スキャン起動後は状態が進むので
// `Err` にして要求を落とす (黙って従来経路に戻すと状態が二重に進む)。

/// 連鎖が使う GGUF 行列の記述 (`GgufLinear` から作る)。
#[derive(Clone, Copy)]
pub struct PfChainW<'a> {
    pub uid: u64,
    pub ty: crate::kquant::GgmlType,
    pub blocks: &'a [u8],
    pub k: usize,
    pub n: usize,
}


/// qkv と gate の投影 (連結されていれば 1 本の重みを 2 回に分けて撃つ)。
#[derive(Clone, Copy)]
pub struct PfChainQkv<'a> {
    /// qkv の重み (連結時は qkv+gate の連結行列)。
    pub qkv: PfChainW<'a>,
    /// gate の重み (連結時は qkv と同じ行列)。
    pub gate: PfChainW<'a>,
    /// `qkv` の行列のうち qkv 行が始まる行 (未連結は 0)。
    pub qkv_row: usize,
    /// 同上 gate。
    pub gate_row: usize,
}


impl<'a> PfChainQkv<'a> {
    /// 2 本をそのまま (連結なし)。
    pub fn separate(qkv: PfChainW<'a>, gate: PfChainW<'a>) -> Self {
        Self {
            qkv,
            gate,
            qkv_row: 0,
            gate_row: 0,
        }
    }
    /// 連結済みの 1 本を行範囲つきで。
    pub fn fused(w: PfChainW<'a>, qkv_row: usize, gate_row: usize) -> Self {
        Self {
            qkv: w,
            gate: w,
            qkv_row,
            gate_row,
        }
    }
    /// 同じ行列を指すか (連結済みか)。
    pub(crate) fn is_fused(&self) -> bool {
        self.qkv.uid == self.gate.uid
    }
}


// ---------------------------------------------------------------------------
// task 186: ssm の beta/alpha 射影 (F32 重み) をデバイスで回す
// (`DENNO_PF_SSM_GEMM_GPU`、既定 off)
// ---------------------------------------------------------------------------
//
// lab t183a の 14k prefill で `beta_pf` は 712 ms (1008 呼び出し = 0.71 ms/回)。
// CPU の `gemv_batch_split` は 0.25 GFLOP/回 を 16 スレッドで回しており、
// 実効 0.36 TFLOP/s。同じ層の隣の qkv GEMM は 74.8 TFLOP/s なので 200 倍の差。
// 入力 (x) は連鎖が既にデバイスへ上げており、出力 (beta/alpha) も連鎖がデバイスで
// 使うので、**ホストを経由せずに** `dno_gemm_f32_mt` で作れる (転送 0)。
// 丸めは FMA + タイル順なので CPU の mul+add 逐次とは一致しない (丸めだけの差)。

/// task 186: beta/alpha 射影をデバイスで回すための指定 (連結済みの F32 重み)。
#[derive(Clone)]
pub struct PfBetaAlpha<'a> {
    /// beta の重み (`k = n_embd`)。連結済みなら beta 行 + alpha 行を持つ。
    pub w: PfChainW<'a>,
    /// beta の行範囲。
    pub beta_rows: std::ops::Range<usize>,
    /// alpha の行範囲 (`w` の中。`alpha_w` が Some なら未使用)。
    pub alpha_rows: std::ops::Range<usize>,
    /// task 225: alpha が別の行列のとき (`Some` なら `w` は beta 専用で、
    /// alpha はこの行列の `alpha_rows` 行)。連結されていない実モデル用。
    pub alpha_w: Option<PfChainW<'a>>,
}


/// s167 (P6): h2 の D2H をどの経路で発行するか。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PfH2Path {
    /// 専用 (非ブロッキング) コピーストリームで発行し、qkv/z GEMM と重ねる
    CopyStream,
    /// 従来の同期 D2H (フラグ OFF・デバイス準備失敗・形の不整合)
    SyncD2h,
}


/// 位相計測を回すか (`DENNO_PROFILE` か `DENNO_PF_DBG`)。未設定は分岐 1 回。
#[inline]
pub(crate) fn pf_timing_on() -> bool {
    crate::prof::enabled() || pf_dbg_on()
}


/// 1 prefill 分の位相計測の状態 (スレッドローカル・リクエストごとに使い回す)。
pub(crate) struct PfTiming {
    /// 計測中 (begin 済みで read/abort 前)。
    pub(crate) active: bool,
    /// 層ループ末尾の境界を打った = 次の `pf_timing_read` で読む。
    pending: bool,
    /// イベントのプール (grow-only。初回の prefill で層数ぶん作る)。
    pub(crate) evs: Vec<CUevent>,
    /// このリクエストで打ったイベント数。
    pub(crate) used: usize,
    /// 区間 k (= イベント k → k+1) の位相。
    pub(crate) phase_of: Vec<u8>,
    /// 位相ごとの積算 (秒)。
    acc: [f64; PF_PH_N],
    /// 層数 (表 C の calls = サンプル数)。
    pub(crate) layers: usize,
    /// トークン数 (ms/token の分母)。
    pub(crate) tokens: usize,
    /// prefill 入口の時刻 (壁時計の基準)。
    pub(crate) t0: Option<Instant>,
    /// prefill の通し番号 (DBG 表示)。
    pub(crate) req: u64,
}


impl Default for PfTiming {
    fn default() -> Self {
        Self {
            active: false,
            pending: false,
            evs: Vec::new(),
            used: 0,
            phase_of: Vec::new(),
            acc: [0.0; PF_PH_N],
            layers: 0,
            tokens: 0,
            t0: None,
            req: 0,
        }
    }
}


thread_local! {
    pub(crate) static PF_TIMING: std::cell::RefCell<PfTiming> =
        std::cell::RefCell::new(PfTiming::default());
}


/// prefill 入口: 位相計測を開く (層ループの直前で 1 回)。
pub fn pf_timing_begin(tokens: usize, layers: usize) {
    // s180 (P41): full_core の内訳計測 (別プール・別 env・既定 OFF)。
    fc_kprof_begin(tokens, layers);
    if !pf_timing_on() {
        return;
    }
    PF_TIMING.with(|c| {
        let mut t = c.borrow_mut();
        t.active = false;
        t.pending = false;
        t.used = 0;
        t.phase_of.clear();
        t.acc = [0.0; PF_PH_N];
        t.layers = layers;
        t.tokens = tokens;
        t.t0 = Some(Instant::now());
        t.req += 1;
        // 1 層あたりの境界は高々 PF_PH_N 個 + 入口 + 末尾。
        let need = layers.saturating_mul(PF_PH_N).saturating_add(2);
        if !pf_ensure_events(&mut t, need) {
            return;
        }
        if !pf_record(&mut t, PF_PH_NONE) {
            return;
        }
        t.active = true;
    });
}


/// 層ループ末尾: 最後の位相の終端を打つ (ここまでに打ったイベントは、この後の
/// **既存の**同期 D2H で必ず完了する)。`pf_timing_read` と対で 1 回ずつ。
pub fn pf_timing_end() {
    if !pf_timing_on() {
        return;
    }
    PF_TIMING.with(|c| {
        let mut t = c.borrow_mut();
        if !t.active {
            return;
        }
        if !pf_record(&mut t, PF_PH_NONE) {
            t.active = false;
            return;
        }
        t.pending = true;
    });
}


/// 残差常駐経路を途中で諦めた (Ok(None) → ホスト経路でやり直す) ときに捨てる。
/// 打ったイベントはストリーム上で勝手に完了するので、そのまま放置してよい。
pub fn pf_timing_abort() {
    if !pf_timing_on() {
        return;
    }
    PF_TIMING.with(|c| {
        let mut t = c.borrow_mut();
        t.active = false;
        t.pending = false;
    });
}


/// prefill 末尾 (**既存の**同期 D2H の後): 区間を読み、表 C の `sl_lay_*` へ
/// 1 位相 1 行で積み、`DENNO_PF_DBG=1` なら内訳を 1 回だけ stderr へ出す。
pub fn pf_timing_read() {
    // s180 (P41): full_core の内訳は DENNO_PF_DBG / DENNO_PROFILE と独立に出す
    // (こちらは DENNO_FC_KPROF だけで動く)。既存の出力には触れない。
    fc_kprof_read();
    // s240 (P43): GDN chunked 3 カーネルの積算行。ここは prefill 末尾 (= 最終行の
    // 既存 D2H の後) なので、読むのは static だけで同期は足さない。
    // `DENNO_GDN_KPROF` 未設定なら分岐 1 回で戻る。
    // s245 (P45): デバイス出力経路で保留したままの 3 区間をここで読む (直前の既存
    // D2H で 3 pass は完了している)。同期は足さない・読むだけ。
    kprof_read_pending();
    gdn_kprof_report_at_prefill_end();
    if !pf_timing_on() {
        return;
    }
    PF_TIMING.with(|c| {
        let mut t = c.borrow_mut();
        if !t.pending {
            return;
        }
        t.pending = false;
        t.active = false;
        let n_iv = t.phase_of.len().saturating_sub(1);
        let Some(elapsed) = lib().ok().and_then(|l| l.event_elapsed_time) else {
            return;
        };
        let mut acc = [0.0f64; PF_PH_N];
        let mut bad = false;
        for k in 0..n_iv {
            let mut ms: f32 = 0.0;
            let rc = unsafe { elapsed(&mut ms, t.evs[k], t.evs[k + 1]) };
            // 未完了 (NOT_READY) や負値は「読めなかった」として全体を捨てる
            // (半端な数字を表へ積むと、以後の判断が全部その数字に引きずられる)。
            if rc != CUDA_SUCCESS || !ms.is_finite() || ms < 0.0 {
                bad = true;
                break;
            }
            let ph = t.phase_of[k] as usize;
            if ph < PF_PH_N {
                acc[ph] += ms as f64 / 1e3;
            }
        }
        if bad {
            if pf_dbg_on() {
                eprintln!(
                    "[PF_DBG] unavailable (event elapsed failed; n={} layers={})",
                    t.tokens, t.layers
                );
            }
            return;
        }
        // 表 C: 1 位相 1 行・calls = 層数 (= 層あたり 1 サンプル)。
        let layers = t.layers.max(1) as u64;
        for ph in 0..PF_PH_N {
            let Some(op) = PF_PH_BUCKETS[ph] else {
                continue;
            };
            let ns = (acc[ph] * 1e6).round() as u64;
            if ns > 0 {
                crate::prof::span_end_ns_calls(ns, op, layers);
            }
        }
        if pf_dbg_on() {
            pf_dbg_print(&t, &acc);
        }
    });
}


/// 残差常駐経路の判定に使うモデル寸法。
pub struct PfDims {
    pub n: usize,
    pub e: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub dh: usize,
    pub nk: usize,
    pub nv: usize,
    pub dk: usize,
    pub dv: usize,
    pub d_conv: usize,
}


/// 実行時の env 実効値で [`pf_layer_geom`] を評価する。
pub fn pf_layer_geom_now(
    attn: &PfAttnW<'_>,
    gate: &crate::quant::Linear,
    up: &crate::quant::Linear,
    down: &crate::quant::Linear,
    d: &PfDims,
) -> Option<()> {
    pf_layer_geom(attn, gate, up, down, d, gemm_min_m(), max_m())
}


/// s165 (P2): 層 1 枚を残差常駐で通せるかの事前検査 (純粋関数・GPU 非依存)。
/// `None` なら呼び出し側はこの層を従来のホスト経路で通す (残差は触らない)。
pub fn pf_layer_geom(
    attn: &PfAttnW<'_>,
    gate: &crate::quant::Linear,
    up: &crate::quant::Linear,
    down: &crate::quant::Linear,
    d: &PfDims,
    gemm_min: usize,
    max_m: usize,
) -> Option<()> {
    let (n, e) = (d.n, d.e);
    if !pf_res_gate(n, e, true, false, gemm_min, max_m) {
        return None;
    }
    // FFN: q4・形・k/n の整合
    let (gk, gn) = q4_dims(gate)?;
    let (uk, un) = q4_dims(up)?;
    let (dkn, dn) = q4_dims(down)?;
    if gk != e || uk != e || gn != un || dkn != gn || dn != e {
        return None;
    }
    if gn == 0 || gn % 32 != 0 {
        return None;
    }
    match attn {
        PfAttnW::Gdn { qkv, z, o } => {
            // 形状・dk==dv・d_conv==4・n の帯域は gdn_prefill_q4_geom と同一契約
            gdn_prefill_q4_geom(qkv, z, o, n, d.nk, d.nv, d.dk, d.dv, d.d_conv)?;
        }
        PfAttnW::Full { q, k, v, o } => {
            let (qk, qn) = q4_dims(q)?;
            let (kk, kn) = q4_dims(k)?;
            let (vk, vn) = q4_dims(v)?;
            let (okn, on) = q4_dims(o)?;
            if d.heads == 0 || d.dh == 0 || d.kv_heads == 0 {
                return None;
            }
            let att = d.heads.checked_mul(d.dh)?;
            let kv_dim = d.kv_heads.checked_mul(d.dh)?;
            if qk != e || qn != att.checked_mul(2)? || kk != e || kn != kv_dim {
                return None;
            }
            if vk != e || vn != kv_dim || okn != att || on != e {
                return None;
            }
            if att % 32 != 0 || kv_dim % 32 != 0 {
                return None;
            }
        }
    }
    Some(())
}


#[cfg(test)]
mod pf_ple_dev_tests {
    use super::*;

    /// task 461: `DENNO_PF_PLE_DEV` の真理値表 (**既定 on** = lab t461d で採択。
    /// 切り戻しは明示的な偽値)。
    #[test]
    fn pf_ple_dev_gate_truth_table() {
        assert!(pf_ple_dev_from_value(None), "未設定は既定 on");
        for v in ["0", "off", "false", "no", " OFF ", "FALSE", ""] {
            assert!(!pf_ple_dev_from_value(Some(v)), "{v:?} が on になった");
        }
        for v in ["1", "on", "true", "yes", " ON ", "TRUE"] {
            assert!(pf_ple_dev_from_value(Some(v)), "{v:?} が off のまま");
        }
    }

    /// task 461: 形の門番。実形 (nt=1,913/2,002、hc=4、e=2,560、kern=2..4、hist=2..8)
    /// が通り、履歴の式と合わない組み合わせ・上限外が落ちること。
    #[test]
    fn pf_ple_dev_shape_gate() {
        // 実形 (kern=2 / dil=1 -> hist=1、kern=4 / dil=2 -> hist=6)
        assert!(pf_ple_dev_ok(1913, 4, 2560, 2, 1, 1));
        assert!(pf_ple_dev_ok(2002, 4, 2560, 4, 2, 6));
        assert!(pf_ple_dev_ok(2002, 4, 2560, 3, 2, 4));
        assert!(pf_ple_dev_ok(PF_PLE_DEV_MIN_TOK, 1, 1, 1, 1, 0), "kern=1 は履歴 0");
        // 落ちるもの
        assert!(!pf_ple_dev_ok(63, 4, 2560, 4, 2, 6), "短い prefill");
        assert!(!pf_ple_dev_ok(2002, 9, 2560, 4, 2, 6), "hc が上限外");
        assert!(!pf_ple_dev_ok(2002, 0, 2560, 4, 2, 6));
        assert!(!pf_ple_dev_ok(2002, 4, 0, 4, 2, 6));
        assert!(!pf_ple_dev_ok(2002, 4, PF_PLE_MAX_E + 1, 4, 2, 6));
        assert!(!pf_ple_dev_ok(2002, 4, 2560, 9, 1, 8), "kern が上限外");
        assert!(!pf_ple_dev_ok(2002, 4, 2560, 0, 1, 0));
        // hist は (kern-1)*dil でなければ落ちる
        assert!(!pf_ple_dev_ok(2002, 4, 2560, 4, 2, 5));
        assert!(!pf_ple_dev_ok(2002, 4, 2560, 4, 0, 0));
        assert!(!pf_ple_dev_ok(2002, 4, 2560, 8, 8, 56), "hist が上限外");
    }

    /// task 461: 起動形 (gate = nt ブロック / conv = (ceil(e/256), hc))。
    #[test]
    fn pf_ple_dev_grid() {
        let (nt, hc, e) = (1913usize, 4usize, 2560usize);
        assert_eq!(e.div_ceil(PF_PLE_THREADS), 10, "conv の grid.x");
        assert_eq!((nt * hc) as f64 / 1e6, 0.007652);
        assert_eq!(PF_PLE_THREADS, 256, "カーネルの PLE_THREADS");
        assert_eq!(PF_PLE_MAX_HC, 8, "カーネルの PLE_MAX_HC");
        assert_eq!(PF_PLE_MAX_E, 8192, "カーネルの PLE_MAX_E");
        assert_eq!(PF_PLE_MAX_KERN, 8, "カーネルの PLE_MAX_KERN");
        assert_eq!(PF_PLE_MAX_HIST, 32, "カーネルの PLE_MAX_HIST");
    }

    /// カーネル `dno_pf_ple_gate` の式をホストで書き下したもの。
    ///
    /// カーネル: ss1 = Σ key^2、ss2 = Σ res^2 → invk / invq =
    /// 1/sqrt(ss/e + eps)、dot = Σ (key*invk*gk)*(res*invq*gq)、
    /// sv = dot/sqrt(e)、gate = sigmoid(sign(sv)*sqrt(max(|sv|,1e-6)))、
    /// invc = 1/sqrt(Σ(value*gate)^2/e + eps)。
    #[allow(clippy::too_many_arguments)]
    fn gate_mirror(
        hc: usize,
        e: usize,
        kern_gate: &[f32],
        res_row: &[f32],
        value_row: &[f32],
        gk: &[f32],
        gq: &[f32],
        eps: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let inv_sqrt_e = 1.0f32 / (e as f32).sqrt();
        let mut gate = vec![0.0f32; hc];
        let mut invc = vec![0.0f32; hc];
        for s in 0..hc {
            let mut s1 = 0.0f32;
            let mut s2 = 0.0f32;
            for i in 0..e {
                let kv = kern_gate[s * e + i];
                let rv = res_row[s * e + i];
                s1 += kv * kv;
                s2 += rv * rv;
            }
            let invk = 1.0 / (s1 / e as f32 + eps).sqrt();
            let invq = 1.0 / (s2 / e as f32 + eps).sqrt();
            let mut dot = 0.0f32;
            for i in 0..e {
                let kn = kern_gate[s * e + i] * invk * gk[s * e + i];
                let qn = res_row[s * e + i] * invq * gq[s * e + i];
                dot += kn * qn;
            }
            let sv = dot * inv_sqrt_e;
            let av = if sv < 0.0 { -sv } else { sv };
            let mag = (if av < 1e-6 { 1e-6 } else { av }).sqrt();
            let sgn = if sv > 0.0 {
                1.0f32
            } else if sv < 0.0 {
                -1.0f32
            } else {
                0.0f32
            };
            gate[s] = 1.0 / (1.0 + mirror_exp(-(sgn * mag)));
            let mut s3 = 0.0f32;
            for i in 0..e {
                let gv = value_row[i] * gate[s];
                s3 += gv * gv;
            }
            invc[s] = 1.0 / (s3 / e as f32 + eps).sqrt();
        }
        (gate, invc)
    }

    fn mirror_exp(x: f32) -> f32 {
        if x > 88.0 {
            return f32::INFINITY;
        }
        if x < -103.0 {
            return 0.0;
        }
        (x as f64).exp() as f32
    }

    /// task 461: gate / invc の式が CPU 参照 (`grouped_rms_norm_into` + 内積) と
    /// 同じ値になること (縮約の木と `__expf` のぶんだけ違う)。
    #[test]
    fn pf_ple_gate_mirror_matches_cpu_reference() {
        let (hc, e) = (4usize, 256usize);
        let eps = 1e-6f32;
        let mut st = 424242u64;
        let mut rnd = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    st = st
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (((st >> 33) as f32) / (u32::MAX >> 1) as f32 - 0.5) * 2.0
                })
                .collect()
        };
        let key = rnd(hc * e);
        let res = rnd(hc * e);
        let value = rnd(e);
        let gk = rnd(hc * e);
        let gq = rnd(hc * e);
        let (g_mine, i_mine) = gate_mirror(hc, e, &key, &res, &value, &gk, &gq, eps);
        // CPU 参照: grouped_rms_norm_into → 内積 → sigmoid → gated の RMS。
        let hp = crate::model::hyper::HyperParams {
            hc,
            n_embd: e,
            low_rank: 4,
            eps,
        };
        let mut kn = vec![0.0f32; hc * e];
        let mut qn = vec![0.0f32; hc * e];
        crate::model::hyper::grouped_rms_norm_into(&key, &gk, &hp, &mut kn);
        crate::model::hyper::grouped_rms_norm_into(&res, &gq, &hp, &mut qn);
        let inv_sqrt_e = 1.0f32 / (e as f32).sqrt();
        for s in 0..hc {
            let mut acc = 0.0f32;
            for i in 0..e {
                acc += kn[s * e + i] * qn[s * e + i];
            }
            let sv = acc * inv_sqrt_e;
            let mag = (if sv.abs() < 1e-6 { 1e-6 } else { sv.abs() }).sqrt();
            let sgn = if sv > 0.0 {
                1.0f32
            } else if sv < 0.0 {
                -1.0f32
            } else {
                0.0f32
            };
            let want = crate::model::hyper::sigmoid(sgn * mag);
            let d = (g_mine[s] - want).abs() / want.max(1e-6);
            assert!(d < 1e-5, "gate[{s}] = {} vs CPU {want} (相対 {d})", g_mine[s]);
            // invc も CPU の rmsnorm (gated) と一致すること。
            let mut gated = vec![0.0f32; e];
            for i in 0..e {
                gated[i] = value[i] * want;
            }
            let mut nx = vec![0.0f32; e];
            crate::tensor::rmsnorm_into(&gated, &vec![1.0f32; e], e, eps, &mut nx);
            // nx = gated * invc なので、invc は nx/gated から取れる (gated != 0)。
            let mut checked = 0;
            for i in 0..e {
                if gated[i].abs() > 1e-3 {
                    let inv_cpu = nx[i] / gated[i];
                    let d = (i_mine[s] - inv_cpu).abs() / inv_cpu.abs().max(1e-6);
                    assert!(d < 1e-4, "invc[{s}] = {} vs CPU {inv_cpu}", i_mine[s]);
                    checked += 1;
                }
            }
            assert!(checked > 0, "invc の比較点が無い");
        }
    }

    /// カーネルの `dno_pf_ple_conv` の索引算術を 1 対 1 でホストへ写したもの
    /// (conv の値と res / 履歴だけを比べる。寄与の和は順序が違うので見ない)。
    #[allow(clippy::too_many_arguments)]
    fn conv_mirror(
        nt: usize,
        hc: usize,
        e: usize,
        kern: usize,
        dil: usize,
        hist: usize,
        head: usize,
        value: &[f32],
        gate: &[f32],
        invc: &[f32],
        gc: &[f32],
        wt: &[f32],
        hist_in: &[f32],
        res: &mut [f32],
    ) -> Vec<f32> {
        let hc_dim = hc * e;
        let mut hist_out = hist_in.to_vec();
        for s in 0..hc {
            for d in 0..e {
                let off = s * e + d;
                let mut xr = vec![0.0f32; hist.max(1)];
                for t in 0..nt {
                    let g = gate[t * hc + s];
                    let iv = invc[t * hc + s];
                    let gated = value[t * e + d] * g;
                    let x = gated * iv * gc[off];
                    let mut acc = 0.0f32;
                    for k in 0..kern {
                        let back = (kern - 1 - k) * dil;
                        let src = if back == 0 {
                            x
                        } else if t >= back {
                            xr[(t - back) % hist]
                        } else {
                            // カーネルは int の負の剰余 (C は被除数の符号)。
                            // usize だと 0 - back が wrap するので i64 で写す。
                            let tb = t as i64 - back as i64;
                            let slot =
                                ((head as i64 + hist as i64 + tb % hist as i64) % hist as i64)
                                    as usize;
                            hist_in[slot * hc_dim + off]
                        };
                        acc += wt[k * hc_dim + off] * src;
                    }
                    let conv = mirror_silu(acc);
                    if hist > 0 {
                        xr[t % hist] = x;
                    }
                    res[t * hc_dim + off] += gated + conv;
                }
                if hist > 0 {
                    // 書き戻しは線形 (先頭が最古)。カーネルと同じ式。
                    for j in 0..hist {
                        let back_j = hist - 1 - j;
                        let tj = nt as i64 - 1 - back_j as i64;
                        hist_out[j * hc_dim + off] = if tj >= 0 {
                            xr[(tj as usize) % hist]
                        } else {
                            let slot = ((head as i64
                                + hist as i64
                                + (tj % hist as i64) % hist as i64)
                                % hist as i64) as usize;
                            hist_in[slot * hc_dim + off]
                        };
                    }
                }
            }
        }
        hist_out
    }

    fn mirror_silu(x: f32) -> f32 {
        x * (1.0 / (1.0 + exp_approx(-x)))
    }

    /// `__expf` の粗い鏡 (f64 の exp を f32 に丸める)。2 ULP 以内。
    fn exp_approx(x: f32) -> f32 {
        if x > 88.0 {
            return f32::INFINITY;
        }
        if x < -103.0 {
            return 0.0;
        }
        (x as f64).exp() as f32
    }

    /// task 461: conv の索引算術 (履歴の ring / back の向き / 書き出しの slot) が
    /// CPU の `ple_conv_silu_t` + `ple_hist_push_ring` と同じ結果になること。
    ///
    /// 値は `__expf` vs `exp` のぶんだけ違いうるので相対 1e-5 で見る
    /// (索引を間違えると order-1 の差になる)。
    #[test]
    fn pf_ple_conv_mirror_matches_cpu_ring() {
        for (kern, dil) in [(2usize, 1usize), (3, 2), (4, 2), (4, 1)] {
            let hist = (kern - 1) * dil;
            let (nt, hc, e) = (37usize, 2usize, 24usize);
            let hc_dim = hc * e;
            let mut st = 12345u64;
            let mut rnd = |n: usize| -> Vec<f32> {
                (0..n)
                    .map(|_| {
                        st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        let u = ((st >> 33) as f32) / (u32::MAX >> 1) as f32;
                        (u - 0.5) * 2.0
                    })
                    .collect()
            };
            let value = rnd(nt * e);
            let gate = rnd(nt * hc);
            let gc = rnd(hc_dim);
            let wt = rnd(kern * hc_dim);
            let hist_in = rnd(hist.max(1) * hc_dim);
            let res0 = rnd(nt * hc_dim);
            // invc は CPU の逐次和で作り、両方に同じものを渡す (conv の索引だけを見る)。
            let mut invc = vec![0.0f32; nt * hc];
            for t in 0..nt {
                for s in 0..hc {
                    let mut ss = 0.0f32;
                    for d in 0..e {
                        let g = value[t * e + d] * gate[t * hc + s];
                        ss += g * g;
                    }
                    invc[t * hc + s] = 1.0 / (ss / e as f32 + 1e-6).sqrt();
                }
            }
            // CPU 参照
            let mut res_cpu = res0.clone();
            let mut hist_cpu = hist_in[..hist * hc_dim].to_vec();
            let mut head = 0usize;
            for t in 0..nt {
                let mut gated = vec![0.0f32; hc_dim];
                let mut x = vec![0.0f32; hc_dim];
                for s in 0..hc {
                    let g = gate[t * hc + s];
                    let iv = invc[t * hc + s];
                    for d in 0..e {
                        let gv = value[t * e + d] * g;
                        gated[s * e + d] = gv;
                        x[s * e + d] = gv * iv * gc[s * e + d];
                    }
                }
                let mut conv = vec![0.0f32; hc_dim];
                crate::model::qwen4exp::ple_conv_silu_t(
                    &mut conv, &x, &hist_cpu, head, &wt, kern, dil, hc_dim,
                );
                for i in 0..hc_dim {
                    res_cpu[t * hc_dim + i] += gated[i] + conv[i];
                }
                crate::model::qwen4exp::ple_hist_push_ring(&mut hist_cpu, &mut head, &x, hc_dim);
            }
            // CPU 経路はチャンク終端で履歴を線形へ戻す。
            {
                let mut tmp = Vec::new();
                crate::model::qwen4exp::ple_hist_to_linear(&mut hist_cpu, head, hc_dim, &mut tmp);
            }
            // カーネルの鏡
            let mut res_gpu = res0.clone();
            let hist_out = conv_mirror(
                nt, hc, e, kern, dil, hist, 0, &value, &gate, &invc, &gc, &wt, &hist_in, &mut res_gpu,
            );
            let scale = res_cpu.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
            let mut maxd = 0.0f32;
            for i in 0..res_cpu.len() {
                maxd = maxd.max((res_cpu[i] - res_gpu[i]).abs());
            }
            assert!(
                maxd / scale < 1e-5,
                "kern={kern} dil={dil}: conv の res が違う (max|diff|/scale = {})",
                maxd / scale
            );
            // 履歴も同じ並びで出ること。
            let hscale = hist_cpu.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
            let mut hmax = 0.0f32;
            for i in 0..hist_cpu.len() {
                hmax = hmax.max((hist_cpu[i] - hist_out[i]).abs());
            }
            assert!(
                hmax / hscale < 1e-5,
                "kern={kern} dil={dil}: 履歴の並びが違う (max|diff|/scale = {})",
                hmax / hscale
            );
        }
    }

    /// task 461: チャンクを跨いだ履歴 (head != 0) でも ring の対応が合うこと。
    #[test]
    fn pf_ple_conv_mirror_ring_head() {
        let (kern, dil) = (4usize, 2usize);
        let hist = (kern - 1) * dil; // 6
        let (nt, hc, e) = (23usize, 2usize, 8usize);
        let hc_dim = hc * e;
        let mut st = 777u64;
        let mut rnd = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    (((st >> 33) as f32) / (u32::MAX >> 1) as f32 - 0.5) * 2.0
                })
                .collect()
        };
        let value = rnd(nt * e);
        let gate = rnd(nt * hc);
        let gc = rnd(hc_dim);
        let wt = rnd(kern * hc_dim);
        let hist_in = rnd(hist * hc_dim);
        let res0 = rnd(nt * hc_dim);
        let mut invc = vec![0.0f32; nt * hc];
        for t in 0..nt {
            for s in 0..hc {
                let mut ss = 0.0f32;
                for d in 0..e {
                    let g = value[t * e + d] * gate[t * hc + s];
                    ss += g * g;
                }
                invc[t * hc + s] = 1.0 / (ss / e as f32 + 1e-6).sqrt();
            }
        }
        for head in 0..hist {
            let mut res_cpu = res0.clone();
            let mut hist_cpu = hist_in.clone();
            let mut h = head;
            for t in 0..nt {
                let mut x = vec![0.0f32; hc_dim];
                for s in 0..hc {
                    let g = gate[t * hc + s];
                    let iv = invc[t * hc + s];
                    for d in 0..e {
                        x[s * e + d] = value[t * e + d] * g * iv * gc[s * e + d];
                    }
                }
                let mut conv = vec![0.0f32; hc_dim];
                crate::model::qwen4exp::ple_conv_silu_t(
                    &mut conv, &x, &hist_cpu, h, &wt, kern, dil, hc_dim,
                );
                for s in 0..hc {
                    let g = gate[t * hc + s];
                    for d in 0..e {
                        let i = s * e + d;
                        res_cpu[t * hc_dim + i] += value[t * e + d] * g + conv[i];
                    }
                }
                crate::model::qwen4exp::ple_hist_push_ring(&mut hist_cpu, &mut h, &x, hc_dim);
            }
            {
                let mut tmp = Vec::new();
                crate::model::qwen4exp::ple_hist_to_linear(&mut hist_cpu, h, hc_dim, &mut tmp);
            }
            let mut res_gpu = res0.clone();
            let hist_out = conv_mirror(
                nt, hc, e, kern, dil, hist, head, &value, &gate, &invc, &gc, &wt, &hist_in,
                &mut res_gpu,
            );
            let scale = res_cpu.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
            let mut maxd = 0.0f32;
            for i in 0..res_cpu.len() {
                maxd = maxd.max((res_cpu[i] - res_gpu[i]).abs());
            }
            assert!(
                maxd / scale < 1e-5,
                "head={head}: res が違う (max|diff|/scale = {})",
                maxd / scale
            );
            let hscale = hist_cpu.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
            let mut hmax = 0.0f32;
            for i in 0..hist_cpu.len() {
                hmax = hmax.max((hist_cpu[i] - hist_out[i]).abs());
            }
            assert!(
                hmax / hscale < 1e-5,
                "head={head}: 履歴が違う (max|diff|/scale = {})",
                hmax / hscale
            );
        }
    }
}
