//! 基盤型 (CudaCtx / DevBuffer / PinMem など) の impl 群
//!
//! task 664: `src/cuda.rs` (70k 行) からの純粋な移動。ロジック・数値・カーネル・
//! PTX・env ゲートは一切変えていない。

#![allow(non_snake_case, non_camel_case_types)]

use super::*;


impl DevBuffer {
    /// task 149: タグ付きの確保。確保に失敗したら expert の LRU を 1 回追い出して
    /// 再試行する (「確保失敗で expert を CPU に固定する」経路を通常経路にしない)。
    ///
    /// ここで `DevBuffer` を一度作ってから詰め替えないこと — `Drop` が
    /// `cuMemFree` を呼ぶので、確保した領域がその場で解放されてしまう
    /// (lab t149a で実際に踏んだ)。`raw_alloc` は生のポインタだけを返す。
    #[track_caller]
    pub(crate) fn alloc_tagged(bytes: usize, tag: &'static str) -> Result<Self> {
        match Self::raw_alloc(bytes) {
            Ok(ptr) => {
                crate::mem_audit::note_alloc(tag, bytes.max(1));
                let _ = launch_dbg::note(|| format!("A tag={tag} bytes={bytes} ptr={ptr:#x}"));
                Ok(Self {
                    ptr,
                    bytes,
                    owned: true,
                    tag,
                })
            }
            Err(e) => {
                // task 149: 他の常駐 (attention のミラー・重み・prefill の
                // 一時バッファ) が原因で落ちたときは、expert を追い出して
                // やり直す。
                //
                // task 210: 追い出す量は**階段**にする。以前は「要求 + 64 MiB」の
                // 次が「全部」の 2 段で、1 段目で戻らなかった (slab はチャンクが
                // 空にならないと実メモリを返さない) だけで 22 GB の expert
                // キャッシュを捨てていた。lab t209a の 14k では prefill の頭で
                // 32 MiB が 1 本通らないだけで全部追い出され、decode が
                // 21 → 35 ms/tok になった。
                //
                // task 256: 追い出しは**チャンク単位**にする。slab はチャンクが
                // 空にならないと実メモリを返さないので、LRU を 1 本ずつ追い出しても
                // 連続した空きはできない (t253 の 32 MiB が 41,550 MiB の追い出しで
                // も通らなかった)。不足分を満たす「アドレスが連続するチャンクの窓」を
                // 冷たい順に選び、丸ごと空けてから 1 回だけ再試行する。
                // 追い出す量は要求 + チャンク 1 本未満 (evict_cap_bytes)。
                if evict_cap_bytes(
                    bytes,
                    tag,
                    crate::model::expert_slab::chunk_bytes(),
                    alloc_keep_experts(),
                )
                .is_some()
                {
                    let (rep, evicted_keys) = if alloc_force_chunk() {
                        crate::model::expert_vram::emergency_free_chunks_forced(bytes)
                    } else {
                        crate::model::expert_vram::emergency_free_chunks_keys(bytes)
                    };
                    // task 279: 追い出した鍵を控える (プールが借りた作業域を返すときに
                    // expert を戻すため)。控えるだけで挙動は変えない。
                    note_alloc_evicted(evicted_keys);
                    if rep.is_empty() {
                        // 空けられない (LRU のロックが取れない / 空チャンクが作れない)
                        // → 呼び出し側が小さいタイルか CPU 経路へ落ちる。
                        return Err(e);
                    }
                    tracing::warn!(
                        "task256: 確保失敗 ({e:#}) → チャンク単位で追い出して再試行: {}",
                        rep.line()
                    );
                    return match Self::raw_alloc(bytes) {
                        Ok(ptr) => {
                            crate::mem_audit::note_alloc(tag, bytes.max(1));
                            Ok(Self {
                                ptr,
                                bytes,
                                owned: true,
                                tag,
                            })
                        }
                        // 2 回目も駄目なら諦める (全捨てはしない)。
                        Err(e2) => Err(e2),
                    };
                }
                let mut last = e;
                for want in alloc_retry_wants(bytes) {
                    let freed = crate::model::expert_vram::emergency_free(want);
                    if freed == 0 {
                        break;
                    }
                    tracing::warn!(
                        "task149: 確保失敗 → expert を {} MiB 追い出して再試行 ({last:#})",
                        freed >> 20
                    );
                    match Self::raw_alloc(bytes) {
                        Ok(ptr) => {
                            crate::mem_audit::note_alloc(tag, bytes.max(1));
                            return Ok(Self {
                                ptr,
                                bytes,
                                owned: true,
                                tag,
                            });
                        }
                        Err(e2) => last = e2,
                    }
                }
                Err(last)
            }
        }
    }

    /// `cuMemAlloc` を 1 回だけ呼ぶ (返すのは生のポインタ。`Drop` を通さない)。
    #[track_caller]
    pub(crate) fn raw_alloc(bytes: usize) -> Result<CUdeviceptr> {
        let mut ptr: CUdeviceptr = 0;
        let res = unsafe { (lib()?.mem_alloc)(&mut ptr, bytes) };
        if res != CUDA_SUCCESS {
            // どの要求が載らなかったかを後から追えるように空き容量を添える。
            match dev_mem_mb() {
                Some((f, t)) => bail!(
                    "CUDA cuMemAlloc が失敗: error {res} (要求 {} MiB / 空き {} MiB / 総 {} MiB)",
                    bytes >> 20,
                    f,
                    t
                ),
                None => bail!("CUDA cuMemAlloc が失敗: error {res} (要求 {} MiB)", bytes >> 20),
            }
        }
        if mem_dbg() {
            if let Some((f, t)) = dev_mem_mb() {
                if bytes >= (1 << 20) && mem_bt() {
                    tracing::info!(target: "denno_engine::mem", "alloc {} KiB free={} MiB total={} MiB at={} bt={}", bytes >> 10, f, t, std::panic::Location::caller(), bt_hint());
                } else {
                    tracing::info!(target: "denno_engine::mem", "alloc {} KiB free={} MiB total={} MiB at={}", bytes >> 10, f, t, std::panic::Location::caller());
                }
            }
        }
        Ok(ptr)
    }

    #[track_caller]
    pub(crate) fn alloc(bytes: usize) -> Result<Self> {
        Self::alloc_tagged(bytes, crate::mem_audit::TAG_OTHER)
    }

    /// task 152: slab のチャンクを 1 本確保する (台帳のタグは expert 側)。
    #[track_caller]
    pub(crate) fn slab_chunk(bytes: usize) -> Result<Self> {
        Self::alloc_tagged(bytes.max(1), crate::mem_audit::TAG_EXPERTS)
    }

    /// task 152: このバッファの一部を指す**非所有**ビュー (drop で解放しない)。
    pub(crate) fn view_at(&self, offset: usize, bytes: usize) -> Self {
        assert!(
            offset.saturating_add(bytes) <= self.bytes,
            "DevBuffer::view_at: 範囲外 (offset={offset}, bytes={bytes}, len={})",
            self.bytes
        );
        Self::view(self.ptr + offset as CUdeviceptr, bytes)
    }

    /// task 152: 同じ領域を指す非所有ビューを作り直す (所有側は drop しない)。
    pub(crate) fn alias(&self) -> Self {
        Self::view(self.ptr, self.bytes)
    }

    /// task 788: 非所有ビューか (プールの区画・重み表の行範囲など)。
    ///
    /// [`crate::cuda::pf_pool`] の区画は prefill の終端で `cuMemFree` されるので、
    /// 借用ビューをスレッドローカルに残すと次回のリクエストで解放済みの領域を
    /// 触る (CUDA error 700)。作業域の再利用判定はこのフラグを見る。
    pub(crate) fn is_view(&self) -> bool {
        !self.owned
    }
    /// task 152 / task 214: `offset` の位置へ pinned staging 経由で H2D する。
    ///
    /// [`from_slice_pinned`] と同じ経路 (ホスト 1 回コピー + `cuMemcpyHtoD`) なので、
    /// slab のスロットへ上げる転送速度は per-expert 確保のときと変わらない。
    ///
    /// task 214: このスレッドで [`enable_thread_upload_stream`] が成功していれば
    /// `cuMemcpyHtoDAsync` を専用ストリームへ出してイベントで待つ。null ストリームの
    /// カーネルと直列にならないので、別スレッドから上げ直すときに GPU の帯を
    /// 塞がない (デバイスへ渡るバイト列は同じ)。
    pub(crate) fn upload_pinned_at(&self, offset: usize, data: &[u8]) -> Result<()> {
        let bytes = data.len();
        if bytes == 0 {
            return Ok(());
        }
        assert!(
            offset.saturating_add(bytes) <= self.bytes,
            "DevBuffer::upload_pinned_at: 範囲外 (offset={offset}, bytes={bytes}, len={})",
            self.bytes
        );
        // task 565: 戻しの最中 (`DENNO_PF_RESTORE_NOSYNC`) は 1 本ごとに待たない。
        if restore_defer_on() && thread_up_stream().is_some() {
            return self.upload_pinned_deferred(offset, data);
        }
        EXPERT_PIN.with(|cell| -> Result<()> {
            let mut slot = cell.borrow_mut();
            if slot.as_ref().map(|p| p.bytes).unwrap_or(0) < bytes {
                *slot = Some(PinMem::alloc(bytes)?);
            }
            let pin = slot.as_ref().expect("確保済み");
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    pin.ptr as *mut u8,
                    bytes,
                );
                self.upload_ptr_at(offset, pin.ptr as *const u8, bytes)
            }
        })
    }

    /// task 565: リング経由の待たない H2D (戻し専用)。
    ///
    /// 4 枠のリングを使い回す。使い回す枠の DMA だけをイベントで待ち、
    /// 他の枠の DMA は走らせたまま次の写しに進む。待ちは戻しの終端で
    /// 1 回 ([`restore_defer_flush_thread`])。
    fn upload_pinned_deferred(&self, offset: usize, data: &[u8]) -> Result<()> {
        let bytes = data.len();
        assert!(
            offset.saturating_add(bytes) <= self.bytes,
            "DevBuffer::upload_pinned_deferred: 範囲外 (offset={offset}, bytes={bytes}, len={})",
            self.bytes
        );
        let lib = lib()?;
        let (Some(ec), Some(rec), Some(es)) =
            (lib.event_create, lib.event_record, lib.event_sync)
        else {
            // イベントが無いドライバでは従来 (1 本ごとに待つ) 経路へ。
            return EXPERT_PIN.with(|cell| -> Result<()> {
                let mut slot = cell.borrow_mut();
                if slot.as_ref().map(|p| p.bytes).unwrap_or(0) < bytes {
                    *slot = Some(PinMem::alloc(bytes)?);
                }
                let pin = slot.as_ref().expect("確保済み");
                unsafe {
                    std::ptr::copy_nonoverlapping(data.as_ptr(), pin.ptr as *mut u8, bytes);
                    self.upload_ptr_at(offset, pin.ptr as *const u8, bytes)
                }
            });
        };
        let Some((stream, _)) = thread_up_stream() else {
            return self.upload_ptr_at(offset, data.as_ptr(), bytes);
        };
        UP_RING.with(|cell| -> Result<()> {
            let mut ring = cell.borrow_mut();
            let idx = up_ring_index(ring.calls);
            ring.calls = ring.calls.wrapping_add(1);
            if ring.slots.len() <= idx {
                ring.slots.resize_with(idx + 1, || None);
            }
            // 使い回す枠の DMA だけ待つ (通常は既に完了している)。
            if let Some(s) = ring.slots[idx].as_ref() {
                unsafe {
                    check(es(s.ev), "cuEventSynchronize (up ring)")?;
                }
            }
            let need_new = match ring.slots[idx].as_ref() {
                Some(s) => s.mem.bytes < bytes,
                None => true,
            };
            if need_new {
                let mem = PinMem::alloc(bytes)?;
                let mut ev: CUevent = std::ptr::null_mut();
                const CU_EVENT_DISABLE_TIMING: u32 = 0x02;
                unsafe {
                    check(ec(&mut ev, CU_EVENT_DISABLE_TIMING), "cuEventCreate (up ring)")?;
                }
                ring.slots[idx] = Some(UpPinSlot { mem, ev });
            }
            let slot = ring.slots[idx].as_ref().expect("確保済み");
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), slot.mem.ptr as *mut u8, bytes);
                check(
                    (lib.memcpy_htod_async)(
                        self.ptr + offset as CUdeviceptr,
                        slot.mem.ptr as *const c_void,
                        bytes,
                        stream,
                    ),
                    "cuMemcpyHtoDAsync (up ring)",
                )?;
                check(rec(slot.ev, stream), "cuEventRecord (up ring)")?;
            }
            dbg_htod(bytes);
            Ok(())
        })
    }

    /// task 260: 転送元が **すでに pinned** (RAM tier の区間) のときの H2D。
    ///
    /// `upload_pinned_at` は毎回 thread-local の bounce へ写してから DMA する。
    /// RAM tier の区間はそのまま DMA できるので、2 MB の写し (実測 0.2 ms/本) を
    /// 省ける。`src_pinned == false` は従来経路 (挙動を変えない)。
    pub(crate) fn upload_at_from(&self, offset: usize, data: &[u8], src_pinned: bool) -> Result<()> {
        if !src_pinned {
            return self.upload_pinned_at(offset, data);
        }
        let bytes = data.len();
        if bytes == 0 {
            return Ok(());
        }
        assert!(
            offset.saturating_add(bytes) <= self.bytes,
            "DevBuffer::upload_at_from: 範囲外 (offset={offset}, bytes={bytes}, len={})",
            self.bytes
        );
        self.upload_ptr_at(offset, data.as_ptr(), bytes)
    }

    /// task 834: 非同期 H2D (`DENNO_EXPERT_H2D_ASYNC`)。
    ///
    /// 戻り値 `Ok(Some(ev))` は「専用コピーストリームへ積んだ。完了は `ev`」。
    /// `Ok(None)` は同期で上げた (ゲート off / リングが使えない / 転送元が pinned)。
    /// 転送元が RAM tier の pinned 区間のときは、写さずにそのまま DMA する既存の
    /// 利点 (`task 260`) を残すため同期経路のままにする。
    pub(crate) fn upload_at_async(
        &self,
        offset: usize,
        data: &[u8],
        src_pinned: bool,
    ) -> Result<Option<u64>> {
        // 判定は純関数 ([`crate::model::h2d_ring::take_async`]) に寄せる
        // (テストで同じ関数を通せるようにするため)。
        if !crate::model::h2d_ring::take_async(
            crate::model::h2d_ring::enabled(),
            src_pinned,
            data.len(),
            super::h2d::MAX_SLOT_BYTES,
        ) {
            self.upload_at_from(offset, data, src_pinned)?;
            return Ok(None);
        }
        let bytes = data.len();
        assert!(
            offset.saturating_add(bytes) <= self.bytes,
            "DevBuffer::upload_at_async: 範囲外 (offset={offset}, bytes={bytes}, len={})",
            self.bytes
        );
        match super::h2d::upload(self.ptr + offset as CUdeviceptr, data) {
            Some(ev) => Ok(Some(ev)),
            None => {
                // リング・ストリーム・イベントのどれかが用意できない。数値は同じ。
                self.upload_pinned_at(offset, data)?;
                Ok(None)
            }
        }
    }

    /// ホストのポインタから H2D する (転送元は pinned であること)。
    ///
    /// `upload_pinned_at` と `upload_at_from` の共通部分 (ストリームの有無・
    /// イベント同期・台帳の積算)。転送するバイト列はどちらの入口でも同じ。
    fn upload_ptr_at(&self, offset: usize, src: *const u8, bytes: usize) -> Result<()> {
        let src = src as *const c_void;
        unsafe {
            match thread_up_stream() {
                Some((s, ev)) => {
                    let l = lib()?;
                    check(
                        (l.memcpy_htod_async)(self.ptr + offset as CUdeviceptr, src, bytes, s),
                        "cuMemcpyHtoDAsync (pinned slab)",
                    )?;
                    let (Some(rec), Some(es)) = (l.event_record, l.event_sync) else {
                        bail!("event_record / event_sync が無い (up stream)");
                    };
                    check(rec(ev, s), "cuEventRecord (up stream)")?;
                    check(es(ev), "cuEventSynchronize (up stream)")?;
                }
                None => {
                    check(
                        (lib()?.memcpy_htod)(self.ptr + offset as CUdeviceptr, src, bytes),
                        "cuMemcpyHtoD (pinned slab)",
                    )?;
                }
            }
        }
        dbg_htod(bytes);
        Ok(())
    }

    /// s165: 既存のデバイス領域の一部を指す**非所有**ビュー。連結重み
    /// (`gate_up_w` など) の行範囲を、元の独立した重みとして見せるために使う
    /// (同じ重みを VRAM へ 2 度上げない)。drop では解放しない。
    /// `bytes` はビュー自身の長さ (所有側の範囲内であることは呼び出し側が保証)。
    pub(crate) fn view(ptr: CUdeviceptr, bytes: usize) -> Self {
        Self { ptr, bytes, owned: false, tag: crate::mem_audit::TAG_OTHER }
    }

    /// ホストのスライスをデバイスへコピーして確保
    #[track_caller]
    pub fn from_slice<T: Copy>(data: &[T]) -> Result<Self> {
        Self::from_slice_tagged(data, crate::mem_audit::TAG_OTHER)
    }

    /// [`DevBuffer::from_slice`] のタグ付き版 (task 149 の台帳用)。
    #[track_caller]
    pub fn from_slice_tagged<T: Copy>(data: &[T], tag: &'static str) -> Result<Self> {
        let bytes = std::mem::size_of_val(data);
        let buf = Self::alloc_tagged(bytes.max(1), tag)?;
        if bytes > 0 {
            unsafe {
                check(
                    (lib()?.memcpy_htod)(buf.ptr, data.as_ptr() as *const c_void, bytes),
                    "cuMemcpyHtoD",
                )?;
            }
            // s166 (P5): 観測のみ (初回アップロード = 定常ステップでは 0 回)
            dbg_htod(bytes);
        }
        Ok(buf)
    }

    /// デバイス → ホスト (len 個の T として読む)
    /// デバイス側だけで 0 にする (f32 の 0.0 はビット 0 なのでゼロ埋めと同じ)。
    /// ホスト Vec を作らないため、小さなバッファを毎回 H2D する経路の負担が消える。
    pub fn fill_zero(&self) -> Result<()> {
        let words = self.bytes / 4;
        if words == 0 {
            return Ok(());
        }
        let rc = unsafe { (lib()?.memset_d32_async)(self.ptr, 0, words, std::ptr::null_mut()) };
        if rc != 0 {
            bail!("cuMemsetD32Async failed: CUresult {}", rc);
        }
        Ok(())
    }

    pub fn to_vec<T: Copy>(&self, len: usize) -> Result<Vec<T>> {
        let bytes = len * std::mem::size_of::<T>();
        assert!(bytes <= self.bytes, "DevBuffer::to_vec: size overflow");
        if bytes == 0 {
            return Ok(Vec::new());
        }
        // s72: ゼロ埋めした Vec を作ってから上書きすると同じ領域を 2 度書く
        // (47MB で数 ms)。未初期化領域へ直接 D2H し、コピー完了後に set_len する
        // (cuMemcpyDtoH は同期コピーなので戻った時点で全要素が埋まっている)。
        let mut out: Vec<T> = Vec::with_capacity(len);
        // task 518: 小さな往復 (`DENNO_DEC_PIN_SMALL`) と大きな転送
        // (`DENNO_PF_PIN_STAGE`) は pinned バウンス経由。ゲート off / 確保失敗 /
        // 転送エラーは下の pageable 経路で丸ごとやり直す (値は同じ)。
        if let Ok(Some(())) = with_pf_pin(bytes, |pin| unsafe {
            check(
                (lib()?.memcpy_dtoh)(pin, self.ptr, bytes),
                "cuMemcpyDtoH (pin stage)",
            )?;
            std::ptr::copy_nonoverlapping(pin as *const u8, out.as_mut_ptr() as *mut u8, bytes);
            Ok(())
        }) {
            unsafe { out.set_len(len) };
            return Ok(out);
        }
        let kp = dec_kprof::op_begin(|| "D2H".to_string(), (0, 0, 0), (0, 0, 0), 0, bytes);
        let res = unsafe {
            check(
                (lib()?.memcpy_dtoh)(out.as_mut_ptr() as *mut c_void, self.ptr, bytes),
                "cuMemcpyDtoH",
            )
        };
        dec_kprof::op_end(kp);
        res?;
        unsafe { out.set_len(len) };
        Ok(out)
    }

    /// デバイス → ホスト (呼び出し側のバッファへ直接。毎層 5 MB を
    /// `Vec` に取り直すとページフォルトだけで数 ms かかるため)。
    pub fn to_slice<T: Copy>(&self, out: &mut [T]) -> Result<()> {
        let bytes = std::mem::size_of_val(out);
        assert!(bytes <= self.bytes, "DevBuffer::to_slice: size overflow");
        // task 318: 1 MiB 以上の同期 D2H は pinned バウンス経由 (`DENNO_PF_PIN_STAGE`)。
        // pageable の D2H はこのホストで 1-3 GB/s しか出ない。
        // 失敗 (pinned が取れない / 転送エラー) は従来経路で丸ごとやり直す。
        if bytes > 0 && matches!(self.to_slice_pinned(out), Ok(true)) {
            return Ok(());
        }
        if bytes > 0 {
            // task 177: 層の内訳 (`DENNO_GDN_DEC_KPROF`)。off は分岐 1 回。
            let kp = dec_kprof::op_begin(|| "D2H".to_string(), (0, 0, 0), (0, 0, 0), 0, bytes);
            let res = unsafe {
                check(
                    (lib()?.memcpy_dtoh)(out.as_mut_ptr() as *mut c_void, self.ptr, bytes),
                    "cuMemcpyDtoH",
                )
            };
            dec_kprof::op_end(kp);
            res?;
        }
        Ok(())
    }

    /// 既存バッファへホストから上書きコピー
    pub fn copy_from<T: Copy>(&self, data: &[T]) -> Result<()> {
        let bytes = std::mem::size_of_val(data);
        assert!(bytes <= self.bytes, "DevBuffer::copy_from: size overflow");
        // task 318: 1 MiB 以上の同期 H2D は pinned バウンス経由 (`DENNO_PF_PIN_STAGE`)。
        // 失敗 (pinned が取れない / 転送エラー) は従来経路で丸ごとやり直す。
        if bytes > 0 && matches!(self.copy_from_pinned(data), Ok(true)) {
            return Ok(());
        }
        let dbg = launch_dbg::note(|| format!("H2D ptr={:#x} bytes={bytes}", self.ptr));
        // task 177: 層の内訳 (`DENNO_GDN_DEC_KPROF`)。off は分岐 1 回。
        let kp = dec_kprof::op_begin(|| "H2D".to_string(), (0, 0, 0), (0, 0, 0), 0, bytes);
        let res = if bytes > 0 {
            unsafe {
                check(
                    (lib()?.memcpy_htod)(self.ptr, data.as_ptr() as *const c_void, bytes),
                    "cuMemcpyHtoD",
                )
            }
        } else {
            Ok(())
        };
        dec_kprof::op_end(kp);
        res?;
        launch_dbg::sync(dbg)?;
        Ok(())
    }

    /// 既存バッファのバイトオフセット位置へホストから上書きコピー (P17g: KV ミラー追記用)
    pub fn copy_from_at<T: Copy>(&self, offset_bytes: usize, data: &[T]) -> Result<()> {
        let bytes = std::mem::size_of_val(data);
        assert!(
            offset_bytes + bytes <= self.bytes,
            "DevBuffer::copy_from_at: size overflow"
        );
        let dbg = launch_dbg::note(|| {
            format!("H2D@ ptr={:#x}+{offset_bytes} bytes={bytes}", self.ptr)
        });
        // task 177: 層の内訳 (`DENNO_GDN_DEC_KPROF`)。off は分岐 1 回。
        let kp = dec_kprof::op_begin(|| "H2D@".to_string(), (0, 0, 0), (0, 0, 0), 0, bytes);
        let res = if bytes > 0 {
            unsafe {
                check(
                    (lib()?.memcpy_htod)(
                        self.ptr + offset_bytes as CUdeviceptr,
                        data.as_ptr() as *const c_void,
                        bytes,
                    ),
                    "cuMemcpyHtoD",
                )
            }
        } else {
            Ok(())
        };
        dec_kprof::op_end(kp);
        res?;
        launch_dbg::sync(dbg)?;
        Ok(())
    }

    /// task 318: pinned バウンス経由の同期 H2D (`DENNO_PF_PIN_STAGE=1`)。
    ///
    /// pageable の `cuMemcpyHtoD` はこのホストで 1-2 GB/s しか出ない (P309 §7.4)。
    /// ホストの配列を pinned に写してから DMA すると、44 MB の残差 1 本が
    /// 20-40 ms → 5-6 ms になる。`Ok(false)` は「ゲートが off / 用意できない」で、
    /// 呼び出し側は従来の pageable 経路へ落ちる。
    pub fn copy_from_pinned<T: Copy>(&self, data: &[T]) -> Result<bool> {
        let bytes = std::mem::size_of_val(data);
        assert!(bytes <= self.bytes, "DevBuffer::copy_from_pinned: size overflow");
        if bytes == 0 {
            return Ok(true);
        }
        let done = with_pf_pin(bytes, |pin| unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, pin as *mut u8, bytes);
            check(
                (lib()?.memcpy_htod)(self.ptr, pin as *const c_void, bytes),
                "cuMemcpyHtoD (pin stage)",
            )
        })?;
        Ok(done.is_some())
    }

    /// task 565: [`Self::copy_from_pinned`] の「全体ゲートを無視する」版。
    ///
    /// 残差の往復だけを pinned に通すとき (`DENNO_PF_RES_PIN`) に使う。
    /// 転送するバイト列は pageable の `copy_from` と同一。
    pub fn copy_from_pinned_forced<T: Copy>(&self, data: &[T]) -> Result<bool> {
        let bytes = std::mem::size_of_val(data);
        assert!(
            bytes <= self.bytes,
            "DevBuffer::copy_from_pinned_forced: size overflow"
        );
        if bytes == 0 {
            return Ok(true);
        }
        let done = with_pf_pin_forced(bytes, |pin| unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, pin as *mut u8, bytes);
            check(
                (lib()?.memcpy_htod)(self.ptr, pin as *const c_void, bytes),
                "cuMemcpyHtoD (pin stage, forced)",
            )
        })?;
        Ok(done.is_some())
    }

    /// task 565: [`Self::to_slice_pinned`] の「全体ゲートを無視する」版。
    pub fn to_slice_pinned_forced<T: Copy>(&self, out: &mut [T]) -> Result<bool> {
        let bytes = std::mem::size_of_val(out);
        assert!(
            bytes <= self.bytes,
            "DevBuffer::to_slice_pinned_forced: size overflow"
        );
        if bytes == 0 {
            return Ok(true);
        }
        let dst = out.as_mut_ptr() as *mut u8;
        let done = with_pf_pin_forced(bytes, |pin| unsafe {
            check(
                (lib()?.memcpy_dtoh)(pin, self.ptr, bytes),
                "cuMemcpyDtoH (pin stage, forced)",
            )?;
            std::ptr::copy_nonoverlapping(pin as *const u8, dst, bytes);
            Ok(())
        })?;
        Ok(done.is_some())
    }

    /// task 318: pinned バウンス経由の同期 D2H (`DENNO_PF_PIN_STAGE=1`)。
    pub fn to_slice_pinned<T: Copy>(&self, out: &mut [T]) -> Result<bool> {
        let bytes = std::mem::size_of_val(out);
        assert!(bytes <= self.bytes, "DevBuffer::to_slice_pinned: size overflow");
        if bytes == 0 {
            return Ok(true);
        }
        let dst = out.as_mut_ptr() as *mut u8;
        let done = with_pf_pin(bytes, |pin| unsafe {
            check(
                (lib()?.memcpy_dtoh)(pin, self.ptr, bytes),
                "cuMemcpyDtoH (pin stage)",
            )?;
            std::ptr::copy_nonoverlapping(pin as *const u8, dst, bytes);
            Ok(())
        })?;
        Ok(done.is_some())
    }

    /// s141b (Task B7c): 既存バッファへ**デバイスから**上書きコピー (D2D・同期)。
    /// `src` の先頭 `len` 要素 (f32) をこのバッファの先頭へ移す。同一 GPU 内の
    /// コピーをホスト経由 (D2H + H2D) で往復させないための経路 — 転送は値そのまま
    /// なのでビットは D2H+H2D と同一。
    pub fn copy_from_dev(&self, src: CUdeviceptr, len: usize) -> Result<()> {
        let bytes = len * std::mem::size_of::<f32>();
        assert!(bytes <= self.bytes, "DevBuffer::copy_from_dev: size overflow");
        let dbg = launch_dbg::note(|| {
            format!("D2D dst={:#x} src={src:#x} bytes={bytes}", self.ptr)
        });
        if bytes > 0 {
            unsafe {
                check(
                    (lib()?.memcpy_dtod)(self.ptr, src, bytes),
                    "cuMemcpyDtoD",
                )?;
            }
        }
        launch_dbg::sync(dbg)?;
        Ok(())
    }

    /// task 181: 既存バッファの**バイトオフセット位置**へデバイスから移す (D2D)。
    ///
    /// `src` の先頭 `len` 要素 (f32) をこのバッファの `dst_offset_bytes` へ
    /// 書く。KV ミラーへチャンクの k / v を直接積むのに使う
    /// (D2H → ホスト積み → H2D の往復をしない)。
    pub fn copy_from_dev_at(&self, dst_offset_bytes: usize, src: CUdeviceptr, len: usize) -> Result<()> {
        let bytes = len * std::mem::size_of::<f32>();
        assert!(
            dst_offset_bytes + bytes <= self.bytes,
            "DevBuffer::copy_from_dev_at: size overflow"
        );
        let dbg = launch_dbg::note(|| {
            format!(
                "D2D@ dst={:#x}+{dst_offset_bytes} src={src:#x} bytes={bytes}",
                self.ptr
            )
        });
        if bytes > 0 {
            unsafe {
                check(
                    (lib()?.memcpy_dtod)(self.ptr + dst_offset_bytes as CUdeviceptr, src, bytes),
                    "cuMemcpyDtoD",
                )?;
            }
        }
        launch_dbg::sync(dbg)?;
        Ok(())
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// 生デバイスポインタ (カーネル引数の組み立て用)
    pub fn ptr(&self) -> CUdeviceptr {
        self.ptr
    }
}


impl Drop for DevBuffer {
    fn drop(&mut self) {
        // s165: ビュー (owned=false) は他バッファの一部なので解放しない
        if self.owned && self.ptr != 0 {
            let _ = launch_dbg::note(|| format!("F ptr={:#x} bytes={}", self.ptr, self.bytes));
            unsafe {
                let _ = (lib().unwrap().mem_free)(self.ptr);
            }
            // task 149: 台帳から戻す (確保と同じタグ・同じバイト数)。
            crate::mem_audit::note_free(self.tag, self.bytes.max(1));
            if mem_dbg() {
                if let Some((f, t)) = dev_mem_mb() {
                    tracing::info!(target: "denno_engine::mem", "free {} KiB free={} MiB total={} MiB", self.bytes >> 10, f, t);
                }
            }
        }
    }
}


// 中身は生ハンドル (context / module / function)。CUcontext はスレッドごとに
// current 設定して使うハンドルであり、所有権の移動は伴わないため Send/Sync を
// 与える (primary context 自体は driver がスレッド安全に管理する)。
unsafe impl Send for CudaCtx {}

unsafe impl Sync for CudaCtx {}


impl CudaCtx {
    /// デバイス 0 (または `DENNO_CUDA_DEVICE`) の primary context を初期化し、
    /// int8 カーネルの PTX をロードする。
    pub fn new() -> Result<Self> {
        let lib = lib()?;
        unsafe {
            check((lib.cu_init)(0), "cuInit")?;
            let device: CUdevice = std::env::var("DENNO_CUDA_DEVICE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            let mut dev: CUdevice = 0;
            check((lib.device_get)(&mut dev, device), "cuDeviceGet")?;

            let mut namebuf = [0u8 as c_char; 128];
            check((lib.device_name)(namebuf.as_mut_ptr(), 127, dev), "cuDeviceGetName")?;
            let name = std::ffi::CStr::from_ptr(namebuf.as_ptr())
                .to_string_lossy()
                .into_owned();

            let (mut maj, mut min) = (0 as c_int, 0 as c_int);
            check((lib.device_cc)(&mut maj, &mut min, dev), "cuDeviceComputeCapability")?;

            let mut drv = 0 as c_int;
            let _ = (lib.driver_version)(&mut drv);

            let mut ctx: CUcontext = std::ptr::null_mut();
            check((lib.primary_ctx_retain)(&mut ctx, dev), "cuDevicePrimaryCtxRetain")?;
            check((lib.ctx_set_current)(ctx), "cuCtxSetCurrent")?;

            // PTX は NUL 終端が必要
            let mut image = PTX.to_vec();
            image.push(0);
            let mut module: CUmodule = std::ptr::null_mut();
            check(
                (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void),
                "cuModuleLoadData (PTX JIT)",
            )?;

            let mut f_quant: CUfunction = std::ptr::null_mut();
            let mut f_mv: CUfunction = std::ptr::null_mut();
            let mut f_gemm: CUfunction = std::ptr::null_mut();
            check(
                (lib.module_get_function)(&mut f_quant, module, b"dno_quantize_rows\0".as_ptr() as *const c_char),
                "cuModuleGetFunction(dno_quantize_rows)",
            )?;
            check(
                (lib.module_get_function)(&mut f_mv, module, b"dno_matvec_rows_bt_i8\0".as_ptr() as *const c_char),
                "cuModuleGetFunction(dno_matvec_rows_bt_i8)",
            )?;
            // P17i: prefill 用 tiled GEMM (m が大きいときの GEMV 型カーネル置き換え)
            check(
                (lib.module_get_function)(&mut f_gemm, module, b"dno_gemm_i8\0".as_ptr() as *const c_char),
                "cuModuleGetFunction(dno_gemm_i8)",
            )?;
            // s123: Q4_0 ネイティブ経路 (同じモジュールに同居)
            let mut f_mv_q4: CUfunction = std::ptr::null_mut();
            let mut f_gemm_q4: CUfunction = std::ptr::null_mut();
            check(
                (lib.module_get_function)(&mut f_mv_q4, module, b"dno_gemv_q4_0\0".as_ptr() as *const c_char),
                "cuModuleGetFunction(dno_gemv_q4_0)",
            )?;
            check(
                (lib.module_get_function)(&mut f_gemm_q4, module, b"dno_gemm_q4_0\0".as_ptr() as *const c_char),
                "cuModuleGetFunction(dno_gemm_q4_0)",
            )?;
            // s132 (Task B1): 複数行 1 block の GEMV は「あれば使う」。古い PTX
            // (未再生成) でも起動を壊さず 1 行 1 block 経路へ落ちるよう、
            // 取得失敗は None として扱う。
            let mut f_mv_q4_mr: CUfunction = std::ptr::null_mut();
            let mv_mr_r = (lib.module_get_function)(
                &mut f_mv_q4_mr,
                module,
                b"dno_gemv_q4_0mr\0".as_ptr() as *const c_char,
            );
            let f_mv_q4_mr = if mv_mr_r == CUDA_SUCCESS {
                Some(f_mv_q4_mr)
            } else {
                None
            };
            // s133 (Task B1, attempt 2): 列ブロック版 GEMV (C = 2 / 4 / 8)。
            // 3 本そろって初めて Some。1 本でも取れなければ None で従来経路。
            let mut fc: [CUfunction; 3] = [std::ptr::null_mut(); 3];
            let mut c_ok = true;
            for (k, name) in [b"dno_gemv_q4_0c2\0", b"dno_gemv_q4_0c4\0", b"dno_gemv_q4_0c8\0"]
                .iter()
                .enumerate()
            {
                let r = (lib.module_get_function)(
                    &mut fc[k],
                    module,
                    name.as_ptr() as *const c_char,
                );
                if r != CUDA_SUCCESS {
                    c_ok = false;
                    break;
                }
            }
            let f_mv_q4_c = if c_ok { Some(fc) } else { None };

            // P17d: 要素演算モジュール (-fmad=false でコンパイル済み。FMA 縮約は
            // CPU とビット一致しないため禁止)
            let mut module_elem: CUmodule = std::ptr::null_mut();
            let mut image_elem = PTX_ELEM.to_vec();
            image_elem.push(0);
            check(
                (lib.module_load_data)(&mut module_elem, image_elem.as_ptr() as *const c_void),
                "cuModuleLoadData (elem PTX JIT)",
            )?;
            let mut f_swiglu: CUfunction = std::ptr::null_mut();
            let mut f_addnorm: CUfunction = std::ptr::null_mut();
            let mut f_expf: CUfunction = std::ptr::null_mut();
            check(
                (lib.module_get_function)(&mut f_swiglu, module_elem, b"dno_swiglu\0".as_ptr() as *const c_char),
                "cuModuleGetFunction(dno_swiglu)",
            )?;
            let mut f_swiglu_int: CUfunction = std::ptr::null_mut();
            check(
                (lib.module_get_function)(&mut f_swiglu_int, module_elem, b"dno_swiglu_int\0".as_ptr() as *const c_char),
                "cuModuleGetFunction(dno_swiglu_int)",
            )?;
            // s170: DENNO_FAST_REDUCE=1 なら二乗和を固定木で並列縮約する版を使う
            // (引数・動的 smem は同一。丸め順だけが変わる)。PTX に無ければ逐次版へ戻す。
            if fast_reduce_enabled()
                && (lib.module_get_function)(
                    &mut f_addnorm,
                    module_elem,
                    b"dno_add_rmsnorm_r\0".as_ptr() as *const c_char,
                ) == 0
            {
                // 並列版を取得できた
            } else {
                check(
                    (lib.module_get_function)(&mut f_addnorm, module_elem, b"dno_add_rmsnorm\0".as_ptr() as *const c_char),
                    "cuModuleGetFunction(dno_add_rmsnorm)",
                )?;
            }
            check(
                (lib.module_get_function)(&mut f_expf, module_elem, b"dno_expf_probe\0".as_ptr() as *const c_char),
                "cuModuleGetFunction(dno_expf_probe)",
            )?;
            // task 640: f16 -> f32 の全 65536 パターン照合 probe (elem PTX に無ければ
            // None。テストだけが skip する)。
            let mut f_f16probe: CUfunction = std::ptr::null_mut();
            let f_f16probe = if (lib.module_get_function)(
                &mut f_f16probe,
                module_elem,
                b"dno_f16cvt_probe\0".as_ptr() as *const c_char,
            ) == 0
            {
                Some(f_f16probe)
            } else {
                None
            };
            let mut f_attn: CUfunction = std::ptr::null_mut();
            check(
                (lib.module_get_function)(&mut f_attn, module_elem, b"dno_attention\0".as_ptr() as *const c_char),
                "cuModuleGetFunction(dno_attention)",
            )?;
            // P17j: prefill attention
            let mut f_attn_pf: CUfunction = std::ptr::null_mut();
            check(
                (lib.module_get_function)(&mut f_attn_pf, module_elem, b"dno_attention_prefill\0".as_ptr() as *const c_char),
                "cuModuleGetFunction(dno_attention_prefill)",
            )?;
            // task 135: t 分割の decode attention。古い PTX (未再生成) でも
            // CudaCtx 生成を壊さないよう Option (無ければ従来の decode 経路)。
            let mut f_attn_dec: CUfunction = std::ptr::null_mut();
            let mut f_attn_dec_cb: CUfunction = std::ptr::null_mut();
            let f_attn_dec = if (lib.module_get_function)(
                &mut f_attn_dec,
                module_elem,
                b"dno_attention_decode\0".as_ptr() as *const c_char,
            ) == 0
                && (lib.module_get_function)(
                    &mut f_attn_dec_cb,
                    module_elem,
                    b"dno_attention_decode_combine\0".as_ptr() as *const c_char,
                ) == 0
            {
                Some((f_attn_dec, f_attn_dec_cb))
            } else {
                None
            };
            // s48 (C1-a): 動的 smem の opt-in 試行。デバイスの opt-in 上限を取得し、
            // 48KB を超えるなら dno_attention_prefill へ設定する (qwen35 dh=256 =
            // 84.8KB を通すため)。失敗しても ctx 生成は続行 (48KB 既定のまま =
            // qwen35 prefill attention はゲートで CPU へ静かにフォールバック)。
            let mut attn_pf_dyn_smem: usize = ATTN_PF_SMEM_DEFAULT;
            if let (Some(dev_attr), Some(fn_attr)) = (lib.device_get_attribute, lib.func_set_attribute) {
                let mut optin: c_int = 0;
                if (dev_attr)(&mut optin, CU_DEV_ATTR_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev) == 0
                    && optin as usize > attn_pf_dyn_smem
                    && (fn_attr)(f_attn_pf, CU_FUNC_ATTR_MAX_DYNAMIC_SHARED_SIZE_BYTES, optin) == 0
                {
                    attn_pf_dyn_smem = optin as usize;
                }
            }
            ATTN_PF_DYN_SMEM.store(attn_pf_dyn_smem, std::sync::atomic::Ordering::Relaxed);
            // task 222: prefill attention v2 (任意。読めない・入口が無い・
            // opt-in が通らないときは null のまま = 従来の
            // dno_attention_prefill で回る。fail closed)。
            let mut f_attn_pf2: CUfunction = std::ptr::null_mut();
            {
                let pf2_path = match std::env::var("DENNO_PF_ATTN2_PTX") {
                    Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
                    _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("kernels")
                        .join("kernels_attn_pf2.ptx"),
                };
                match std::fs::read(&pf2_path) {
                    Ok(mut image) => {
                        image.push(0);
                        unsafe {
                            let mut module2: CUmodule = std::ptr::null_mut();
                            let loaded = (lib.module_load_data)(
                                &mut module2,
                                image.as_ptr() as *const c_void,
                            ) == CUDA_SUCCESS;
                            if loaded
                                && (lib.module_get_function)(
                                    &mut f_attn_pf2,
                                    module2,
                                    b"dno_attention_prefill2\0".as_ptr() as *const c_char,
                                ) == CUDA_SUCCESS
                            {
                                let ok = match lib.func_set_attribute {
                                    Some(fn_attr) => (fn_attr)(
                                        f_attn_pf2,
                                        CU_FUNC_ATTR_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                        ATTN_PF2_SMEM_BYTES as c_int,
                                    ) == 0,
                                    None => false,
                                };
                                if !ok {
                                    eprintln!(
                                        "[pf-attn2] 動的 smem の opt-in に失敗 → 従来の attention へ"
                                    );
                                    f_attn_pf2 = std::ptr::null_mut();
                                }
                            } else {
                                eprintln!("[pf-attn2] PTX を読めない/入口が無い → 従来の attention へ");
                                f_attn_pf2 = std::ptr::null_mut();
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!(
                            "[pf-attn2] PTX が無い ({}) : {e} → 従来の attention へ",
                            pf2_path.display()
                        );
                    }
                }
            }
            // task 230: prefill attention v3 (任意。読めない・入口が無い・opt-in が
            // 通らないときは null のまま = v2 / 従来の経路で回る。fail closed)。
            // ゲートが off のときは読まない (PTX は commit 済みだが、切り戻した
            // ビルドでは読み込みもしない)。
            let mut f_attn_pf3: CUfunction = std::ptr::null_mut();
            if pf_attn_v3_on() {
                let pf3_path = match std::env::var("DENNO_PF_ATTN3_PTX") {
                    Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
                    _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("kernels")
                        .join("kernels_attn_pf3.ptx"),
                };
                match std::fs::read(&pf3_path) {
                    Ok(mut image) => {
                        image.push(0);
                        unsafe {
                            let mut module3: CUmodule = std::ptr::null_mut();
                            let loaded = (lib.module_load_data)(
                                &mut module3,
                                image.as_ptr() as *const c_void,
                            ) == CUDA_SUCCESS;
                            if loaded
                                && (lib.module_get_function)(
                                    &mut f_attn_pf3,
                                    module3,
                                    b"dno_attention_prefill3\0".as_ptr() as *const c_char,
                                ) == CUDA_SUCCESS
                            {
                                let ok = match lib.func_set_attribute {
                                    Some(fn_attr) => (fn_attr)(
                                        f_attn_pf3,
                                        CU_FUNC_ATTR_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                        ATTN_PF3_SMEM_BYTES as c_int,
                                    ) == 0,
                                    None => false,
                                };
                                if !ok {
                                    eprintln!(
                                        "[pf-attn3] 動的 smem の opt-in に失敗 → v2 / 従来の attention へ"
                                    );
                                    f_attn_pf3 = std::ptr::null_mut();
                                }
                            } else {
                                eprintln!("[pf-attn3] PTX を読めない/入口が無い → v2 / 従来へ");
                                f_attn_pf3 = std::ptr::null_mut();
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!(
                            "[pf-attn3] PTX が無い ({}) : {e} → v2 / 従来へ",
                            pf3_path.display()
                        );
                    }
                }
            }
            // s169 (P10): SM 数を 1 回だけ取る (GEMV の launch 形状を n から選ぶ
            // ためだけに使う。取れなければ 0 のまま = 形状選択は従来の固定 C)。
            if let Some(dev_attr) = lib.device_get_attribute {
                let mut sms: c_int = 0;
                if (dev_attr)(&mut sms, CU_DEV_ATTR_MULTIPROCESSOR_COUNT, dev) == 0 && sms > 0 {
                    GEMV_SM_COUNT.store(sms as usize, std::sync::atomic::Ordering::Relaxed);
                }
            }
            let mut f_rope: CUfunction = std::ptr::null_mut();
            check(
                (lib.module_get_function)(&mut f_rope, module_elem, b"dno_rope_qkv\0".as_ptr() as *const c_char),
                "cuModuleGetFunction(dno_rope_qkv)",
            )?;
            // P17k ステップ 5: prefill 用行並列 rope
            let mut f_rope_pf: CUfunction = std::ptr::null_mut();
            check(
                (lib.module_get_function)(&mut f_rope_pf, module_elem, b"dno_rope_qkv_pf\0".as_ptr() as *const c_char),
                "cuModuleGetFunction(dno_rope_qkv_pf)",
            )?;
            // B3: b>1 バッチ decode 用 rope
            let mut f_rope_b: CUfunction = std::ptr::null_mut();
            check(
                (lib.module_get_function)(&mut f_rope_b, module_elem, b"dno_rope_qk_batch\0".as_ptr() as *const c_char),
                "cuModuleGetFunction(dno_rope_qk_batch)",
            )?;

            // P19-A1: GDN 逐次スキャンモジュール
            let mut module_gdn: CUmodule = std::ptr::null_mut();
            let mut image_gdn = PTX_GDN.to_vec();
            image_gdn.push(0);
            check(
                (lib.module_load_data)(&mut module_gdn, image_gdn.as_ptr() as *const c_void),
                "cuModuleLoadData (gdn PTX JIT)",
            )?;
            let mut f_gdn: CUfunction = std::ptr::null_mut();
            check(
                (lib.module_get_function)(&mut f_gdn, module_gdn, b"dno_gdn_scan\0".as_ptr() as *const c_char),
                "cuModuleGetFunction(dno_gdn_scan)",
            )?;
            // A1-r2: chunked scan 3 カーネル
            let mut f_gdn_chunk_pre: CUfunction = std::ptr::null_mut();
            check(
                (lib.module_get_function)(&mut f_gdn_chunk_pre, module_gdn, b"dno_gdn_chunk_pre\0".as_ptr() as *const c_char),
                "cuModuleGetFunction(dno_gdn_chunk_pre)",
            )?;
            let mut f_gdn_chunk_state: CUfunction = std::ptr::null_mut();
            check(
                (lib.module_get_function)(&mut f_gdn_chunk_state, module_gdn, b"dno_gdn_chunk_state\0".as_ptr() as *const c_char),
                "cuModuleGetFunction(dno_gdn_chunk_state)",
            )?;
            let mut f_gdn_chunk_scan: CUfunction = std::ptr::null_mut();
            check(
                (lib.module_get_function)(&mut f_gdn_chunk_scan, module_gdn, b"dno_gdn_chunk_scan\0".as_ptr() as *const c_char),
                "cuModuleGetFunction(dno_gdn_chunk_scan)",
            )?;
            // s89 (P1-1): 状態常駐版は「あれば使う」。古い PTX (未再生成) でも
            // 起動を壊さず 3 launch 経路へ落ちるよう、取得失敗は None として扱う。
            let mut f_gdn_chunk_scan_hoist: CUfunction = std::ptr::null_mut();
            let hoist_r = (lib.module_get_function)(
                &mut f_gdn_chunk_scan_hoist,
                module_gdn,
                b"dno_gdn_chunk_scan_hoist\0".as_ptr() as *const c_char,
            );
            // (隣接フィールドと同じく CUresult == 0 が成功。非 0 は None)
            let f_gdn_chunk_scan_hoist = if hoist_r == CUDA_SUCCESS {
                Some(f_gdn_chunk_scan_hoist)
            } else {
                None
            };

            // task 128: qwen4exp の decode 用 scan (sigmoid 出力ゲート) も
            // 「あれば使う」。task 137 で PTX を再生成したので通常は取れる。
            // 取れないとき (旧 PTX) は GPU decode scan 経路を発動させない
            // (fail-closed = CPU 経路)。
            let mut f_gdn_scan_sig: CUfunction = std::ptr::null_mut();
            let sig_r = (lib.module_get_function)(
                &mut f_gdn_scan_sig,
                module_gdn,
                b"dno_gdn_scan_sig\0".as_ptr() as *const c_char,
            );
            // task 137: 取れただけでは足りない。ホストは 23 引数を渡すので、
            // PTX 側の引数個数が dno_gdn_scan と違えば起動しない (ABI 検査)。
            let sig_abi_ok = std::str::from_utf8(PTX_GDN)
                .map(gdn_scan_sig_abi_ok)
                .unwrap_or(false);
            let f_gdn_scan_sig = if sig_r == CUDA_SUCCESS && sig_abi_ok {
                Some(f_gdn_scan_sig)
            } else {
                if sig_r == CUDA_SUCCESS && !sig_abi_ok {
                    tracing::warn!(
                        "[gdn-gpu-scan] dno_gdn_scan_sig の引数個数が dno_gdn_scan と違う \
                         (PTX とホストの不整合) → CPU 経路"
                    );
                }
                None
            };

            // s239 (P36): 4 レーン版 pass 1 も「あれば使う」(同じ理由)。
            let mut f_gdn_chunk_pre_l4: CUfunction = std::ptr::null_mut();
            let l4_r = (lib.module_get_function)(
                &mut f_gdn_chunk_pre_l4,
                module_gdn,
                b"dno_gdn_chunk_pre_l4\0".as_ptr() as *const c_char,
            );
            let f_gdn_chunk_pre_l4 = if l4_r == CUDA_SUCCESS {
                Some(f_gdn_chunk_pre_l4)
            } else {
                None
            };

            // P55: pass 1 の状態配列 (W/M) を共有メモリへ置く版 (dk=128 で 129 KiB)。
            // 「あれば使う」+ 動的 smem の opt-in の 2 段。どちらかが失敗すれば
            // None / 48 KiB 既定のままになり、ゲート (`gdn_pre_sm_gate`) が
            // 旧カーネルを選ぶ (fail-closed)。.cu を先に足して PTX を後から
            // 再生成する運用でも、起動は壊れない。
            let mut f_gdn_chunk_pre_sm: CUfunction = std::ptr::null_mut();
            let sm_r = (lib.module_get_function)(
                &mut f_gdn_chunk_pre_sm,
                module_gdn,
                b"dno_gdn_chunk_pre_sm\0".as_ptr() as *const c_char,
            );
            let (f_gdn_chunk_pre_sm, gdn_pre_sm_dyn_smem) = if sm_r == CUDA_SUCCESS {
                let mut limit = GDN_PRE_SM_SMEM_DEFAULT;
                if let (Some(dev_attr), Some(fn_attr)) = (lib.device_get_attribute, lib.func_set_attribute) {
                    let mut optin: c_int = 0;
                    if (dev_attr)(&mut optin, CU_DEV_ATTR_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev) == 0
                        && optin as usize > limit
                        && (fn_attr)(f_gdn_chunk_pre_sm, CU_FUNC_ATTR_MAX_DYNAMIC_SHARED_SIZE_BYTES, optin) == 0
                    {
                        limit = optin as usize;
                    }
                }
                (Some(f_gdn_chunk_pre_sm), limit)
            } else {
                (None, GDN_PRE_SM_SMEM_DEFAULT)
            };

            // P-MOE-E1: expert GEMV のモジュール (Q8_0 / Q5_1 / Q4_K / Q5_K)。
            // 4 本そろって初めて Some。1 本でも欠ければ None (CPU 参照へ落ちる)。
            let mut module_moe: CUmodule = std::ptr::null_mut();
            let mut f_moe: Option<[CUfunction; 4]> = None;
            {
                let mut image = PTX_MOE.to_vec();
                image.push(0);
                if (lib.module_load_data)(&mut module_moe, image.as_ptr() as *const c_void) == CUDA_SUCCESS {
                    let mut fs: [CUfunction; 4] = [std::ptr::null_mut(); 4];
                    let names: [&[u8]; 4] = [
                        b"dno_gemv_q8_0_w\0",
                        b"dno_gemv_q5_1_w\0",
                        b"dno_gemv_q4k_w\0",
                        b"dno_gemv_q5k_w\0",
                    ];
                    let mut ok = true;
                    for (k, n) in names.iter().enumerate() {
                        if (lib.module_get_function)(&mut fs[k], module_moe, n.as_ptr() as *const c_char)
                            != CUDA_SUCCESS
                        {
                            ok = false;
                            break;
                        }
                    }
                    if ok {
                        f_moe = Some(fs);
                    }
                }
            }

            // P-gemv2: 数値契約は v1 と同一で、x をレジスタへ先読みする版。
            let mut module_moe2: CUmodule = std::ptr::null_mut();
            let mut f_moe2: Option<[CUfunction; 4]> = None;
            {
                let mut image = PTX_MOE_V2.to_vec();
                image.push(0);
                if (lib.module_load_data)(&mut module_moe2, image.as_ptr() as *const c_void) == CUDA_SUCCESS {
                    let mut fs: [CUfunction; 4] = [std::ptr::null_mut(); 4];
                    let names: [&[u8]; 4] = [
                        b"dno_gemv_q8_0_v2\0",
                        b"dno_gemv_q5_1_v2\0",
                        b"dno_gemv_q4k_v2\0",
                        b"dno_gemv_q5k_v2\0",
                    ];
                    let mut ok = true;
                    for (k, n) in names.iter().enumerate() {
                        if (lib.module_get_function)(&mut fs[k], module_moe2, n.as_ptr() as *const c_char)
                            != CUDA_SUCCESS
                        {
                            ok = false;
                            break;
                        }
                    }
                    if ok {
                        f_moe2 = Some(fs);
                    }
                }
            }

            // P-PF-GB: prefill 用の量子 GEMM のモジュール。同じ規約 (4 本そろって初めて Some)。
            let mut module_gemm_q: CUmodule = std::ptr::null_mut();
            let mut f_gemm_q: Option<[CUfunction; 4]> = None;
            {
                let mut image = PTX_GEMM_Q.to_vec();
                image.push(0);
                if (lib.module_load_data)(&mut module_gemm_q, image.as_ptr() as *const c_void) == CUDA_SUCCESS {
                    let mut fs: [CUfunction; 4] = [std::ptr::null_mut(); 4];
                    let names: [&[u8]; 4] = [
                        b"dno_gemm_q8_0_w\0",
                        b"dno_gemm_q5_1_w\0",
                        b"dno_gemm_q4k_w\0",
                        b"dno_gemm_q5k_w\0",
                    ];
                    let mut ok = true;
                    for (k, n) in names.iter().enumerate() {
                        if (lib.module_get_function)(&mut fs[k], module_gemm_q, n.as_ptr() as *const c_char)
                            != CUDA_SUCCESS
                        {
                            ok = false;
                            break;
                        }
                    }
                    if ok {
                        f_gemm_q = Some(fs);
                    }
                }
            }

            // task 149: どのモジュールをどれだけの PTX から JIT したかを台帳に残す
            // (JIT 後のコード量はドライバが公開しないので、ここでは PTX の大きさ)。
            crate::mem_audit::note_module("int8", PTX.len());
            crate::mem_audit::note_module("elem", PTX_ELEM.len());
            crate::mem_audit::note_module("gdn", PTX_GDN.len());
            crate::mem_audit::note_module("moe", PTX_MOE.len());
            crate::mem_audit::note_module("moe_v2", PTX_MOE_V2.len());
            crate::mem_audit::note_module("gemm_q", PTX_GEMM_Q.len());

            // task 1377: PDL の可否は SM 番号で決まる。ここで 1 回だけ置く
            // (5090 は (12, 0)。major だけ見ると 12 < 90 で一度も付かない)。
            pdl::set_device_sm(pdl::sm_number(maj, min));

            Ok(Self {
                device: dev,
                name,
                cc: (maj, min),
                driver_ver: drv,
                ctx,
                module,
                f_quant,
                f_mv,
                f_gemm,
                f_mv_q4,
                f_gemm_q4,
                f_mv_q4_mr,
                f_mv_q4_c,
                module_elem,
                f_swiglu,
                f_swiglu_int,
                f_addnorm,
                f_expf,
                f_f16probe,
                f_attn,
                f_attn_pf,
                f_attn_pf2,
                f_attn_pf3,
                f_attn_dec: f_attn_dec.map(|(a, _)| a),
                f_attn_dec_cb: f_attn_dec.map(|(_, b)| b),
                attn_pf_dyn_smem,
                f_rope,
                f_rope_pf,
                f_rope_b,
                module_gdn,
                f_gdn,
                f_gdn_chunk_pre,
                f_gdn_chunk_pre_l4,
                f_gdn_chunk_pre_sm,
                gdn_pre_sm_dyn_smem,
                f_gdn_chunk_state,
                f_gdn_chunk_scan,
                f_gdn_chunk_scan_hoist,
                f_gdn_scan_sig,
                module_moe,
                f_moe,
                f_moe2,
                module_gemm_q,
                f_gemm_q,
            })
        }
    }

    /// "NVIDIA GeForce RTX 5090 (sm_120, driver 13010)" 形式の情報
    pub fn info(&self) -> String {
        format!(
            "{} (sm_{}{}, driver {}, attn_pf dyn smem {}KB)",
            self.name,
            self.cc.0,
            self.cc.1,
            self.driver_ver,
            self.attn_pf_dyn_smem / 1024
        )
    }

    pub fn device(&self) -> c_int {
        self.device
    }

    /// カーネル起動のみ (同期しない — デフォルトストリームにエンキュー)。
    /// P17b: 固定オーバーヘッド削減のため launch/sync を分離。
    pub(crate) unsafe fn launch_async(
        &self,
        f: CUfunction,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shmem: u32,
        args: &mut [*mut c_void],
    ) -> Result<()> {
        self.launch_async_on(
            std::ptr::null_mut(),
            f,
            grid,
            block,
            shmem,
            args,
        )
    }

    /// [`Self::launch_async`] のストリーム指定版 (task 124 の重ね打ち経路)。
    ///
    /// `stream` が null のときの挙動は従来と完全に同じ (同じ `dbg_launch` の
    /// 計数・同じ `cuLaunchKernel` 引数)。
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn launch_async_on(
        &self,
        stream: CUstream,
        f: CUfunction,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shmem: u32,
        args: &mut [*mut c_void],
    ) -> Result<()> {
        let lib = lib()?;
        // s166 (P5): decode 1 ステップの launch 件数 (DENNO_DECODE_DBG=1・既定 OFF)
        dbg_launch(f);
        // task 174: 診断 (DENNO_LAUNCH_DBG) の輪へ積む (off は None)。
        let dbg_seq = launch_dbg::note(|| {
            format!(
                "K {} grid=({},{},{}) block=({},{},{}) shmem={}",
                launch_dbg::fn_name(f),
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                shmem
            )
        });
        // task 138: 台帳 (DENNO_Q4_PROFILE) への発行 1 件の計上。off は stamp が None。
        let tl = crate::model::q4ledger::stamp();
        // task 153: イベント時系列 (DENNO_Q4_TLINE + _K)。off は None で分岐 1 回。
        let ev0 = tline::launch_begin();
        // task 177: decode 層の操作別内訳 (DENNO_GDN_DEC_KPROF)。off は分岐 1 回。
        let kp = dec_kprof::op_begin(
            || format!("K {}", launch_dbg::fn_name(f)),
            grid,
            block,
            shmem,
            0,
        );
        // task 1377: decode の鎖 (SM 90+ / ゲート on) では PDL 属性付きで発行する。
        // 属性付きが失敗したら、その 1 回は従来の launch でやり直す (fail closed)。
        let mut pdl_res: Option<CUresult> = None;
        if pdl::eligible(stream) {
            if let Some(fex) = lib.launch_ex {
                let mut attr = pdl::programmatic_attr();
                let cfg = pdl::launch_config(grid, block, shmem, stream, &mut attr);
                let res = (fex)(
                    &cfg as *const pdl::CUlaunchConfig,
                    f,
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                );
                if res == CUDA_SUCCESS {
                    pdl::note_launch();
                    pdl_res = Some(res);
                } else {
                    pdl::note_fallback(res);
                }
            }
        }
        let r = check(
            match pdl_res {
                Some(res) => res,
                None => (lib.launch)(
                    f,
                    grid.0,
                    grid.1,
                    grid.2,
                    block.0,
                    block.1,
                    block.2,
                    shmem,
                    stream,
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
            },
            if pdl_res.is_some() {
                "cuLaunchKernelEx(PDL)"
            } else {
                "cuLaunchKernel"
            },
        );
        dec_kprof::op_end(kp);
        if let Some(a) = ev0 {
            tline::launch_end(crate::model::q4ledger::current(), a);
        }
        crate::model::q4ledger::note_launch(tl);
        // task 174: 診断時だけ起動ごとに同期して、最初に落ちたカーネルを名指しする。
        r?;
        launch_dbg::sync(dbg_seq)?;
        Ok(())
    }

    /// デフォルトストリームの同期 (エンキュー済み全操作の完了待ち)
    pub fn sync(&self) -> Result<()> {
        let lib = lib()?;
        // task 138: 台帳へ待ち時間を積む (off は stamp が None で計測しない)。
        let t = crate::model::q4ledger::stamp();
        // task 177: 層の内訳 (`DENNO_GDN_DEC_KPROF`)。off は分岐 1 回。
        let kp = dec_kprof::op_begin(|| "sync".to_string(), (0, 0, 0), (0, 0, 0), 0, 0);
        let r = unsafe { check((lib.ctx_sync)(), "cuCtxSynchronize") };
        dec_kprof::op_end(kp);
        crate::model::q4ledger::note_sync(t);
        r
    }

    /// task 368: **null ストリームだけ**を待つ (このスレッドが積んだ仕事の完了)。
    ///
    /// [`Self::sync`] との違いは待つ**範囲**だけで、待つ相手の仕事は同じ:
    /// `cuEventSynchronize` は null ストリームに積まれた仕事 (このスレッドの
    /// カーネルと非同期コピー) の完了を待つ。文脈同期と違って、背景の上げ直し
    /// (expert の非ブロッキングストリーム) と shared expert の side stream は
    /// 待たない。イベントが張れない環境だけ [`Self::sync`] へ落ちる。
    pub fn sync_stream(&self) -> Result<()> {
        let t = crate::model::q4ledger::stamp();
        if seg_ev::record_raw(DEC_SYNC_SLOT) {
            let r = seg_ev::wait(DEC_SYNC_SLOT);
            crate::model::q4ledger::note_sync(t);
            DEC_SYNC_EVENTS.fetch_add(1, Ordering::Relaxed);
            r
        } else {
            DEC_SYNC_FALLBACKS.fetch_add(1, Ordering::Relaxed);
            self.sync()
        }
    }

    /// task 368: `DENNO_DEC_SYNC_STREAM` で待ちの種類を選ぶ入口。
    ///
    /// 既定 off では [`Self::sync`] そのもの (int-auto の挙動と 1 命令も変わらない)。
    pub fn sync_decode(&self) -> Result<()> {
        if dec_sync_stream_enabled() {
            self.sync_stream()
        } else {
            self.sync()
        }
    }
    /// 呼び出しスレッドに primary context を current として設定する。
    /// CUDA の current context は**スレッドローカル** (P17c の教訓: libtest は
    /// テスト毎に新スレッドを立てるため、初期化スレッドで 1 回だけ設定したのでは
    /// 他スレッドの cuMemAlloc が error 201 INVALID_CONTEXT で落ちる)。共有
    /// CudaCtx を複数スレッドから触るため、GPU 操作の入口で毎回張り直す
    /// (cuCtxSetCurrent は冪等かつ軽量)。
    pub fn make_current(&self) -> Result<()> {
        let lib = lib()?;
        unsafe { check((lib.ctx_set_current)(self.ctx), "cuCtxSetCurrent") }
    }

    /// `tensor::quantize_row_i8` の GPU 版 (m 行一括)。
    /// 戻り値: (量子化済みアクティベーション [m*k] i8, ascale [m] f32) の
    /// デバイスバッファ (後続の `matvec_rows_bt_i8_dev` にそのまま渡せる)。
    pub fn quantize_rows(&self, a: &[f32], m: usize, k: usize) -> Result<(DevBuffer, DevBuffer)> {
        assert_eq!(a.len(), m * k, "quantize_rows: a size mismatch");
        let dev_a = DevBuffer::from_slice(a)?;
        let dev_aq = DevBuffer::alloc(m * k)?;
        let dev_asc = DevBuffer::alloc(m * 4)?;
        unsafe {
            let (mut pa, mut paq, mut pasc) = (dev_a.ptr, dev_aq.ptr, dev_asc.ptr);
            let (mut mc, mut kc) = (m as c_int, k as c_int);
            let mut args = [
                &mut pa as *mut _ as *mut c_void,
                &mut paq as *mut _ as *mut c_void,
                &mut pasc as *mut _ as *mut c_void,
                &mut mc as *mut _ as *mut c_void,
                &mut kc as *mut _ as *mut c_void,
            ];
            // 1 block per row, 256 threads, shmem 256*4 bytes
            self.launch_async(
                self.f_quant,
                (m as u32, 1, 1),
                (256, 1, 1),
                256 * 4,
                &mut args,
            )?;
            self.sync()?;
        }
        Ok((dev_aq, dev_asc))
    }

    /// int8 GEMV/GEMM (nn.Linear 形式 q: [n,k]) をデバイスバッファで実行。
    /// `tensor::matvec_rows_bt_i8` とビット一致する [m,n] f32 を返す。
    pub fn matvec_rows_bt_i8_dev(
        &self,
        aq: &DevBuffer,
        asc: &DevBuffer,
        w: &DevBuffer,
        scale: &DevBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        let dev_out = DevBuffer::alloc(m * n * 4)?;
        unsafe {
            let (mut paq, mut pasc, mut pw, mut psc, mut pout) =
                (aq.ptr, asc.ptr, w.ptr, scale.ptr, dev_out.ptr);
            let (mut mc, mut kc, mut nc) = (m as c_int, k as c_int, n as c_int);
            let mut args = [
                &mut paq as *mut _ as *mut c_void,
                &mut pasc as *mut _ as *mut c_void,
                &mut pw as *mut _ as *mut c_void,
                &mut psc as *mut _ as *mut c_void,
                &mut pout as *mut _ as *mut c_void,
                &mut mc as *mut _ as *mut c_void,
                &mut kc as *mut _ as *mut c_void,
                &mut nc as *mut _ as *mut c_void,
            ];
            // 1 warp per (i,j): block 256 = 8 warps
            let total = (m * n) as u64;
            let blocks = total.div_ceil(8) as u32;
            self.launch_async(self.f_mv, (blocks, 1, 1), (256, 1, 1), 0, &mut args)?;
            self.sync()?;
        }
        dev_out.to_vec::<f32>(m * n)
    }

    /// ホストのアクティベーションから一括で実行する高レベル API
    /// (量子化 + GEMV)。重み/scale は事前に `upload_i8` / `upload_f32` で
    /// VRAM 常駐させておく。戻り値は `tensor::matvec_rows_bt_i8` とビット一致。
    pub fn linear_bt_i8(
        &self,
        a: &[f32],
        w: &DevBuffer,
        scale: &DevBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        debug_assert_eq!(w.bytes(), k * gemm_pad_n(n));
        debug_assert_eq!(scale.bytes(), n * 4);
        let (aq, asc) = self.quantize_rows(a, m, k)?;
        self.matvec_rows_bt_i8_dev(&aq, &asc, w, scale, m, k, n)
    }

    /// P17i: tiled GEMM (dno_gemm_i8) で実行。`linear_bt_i8` とビット一致。
    /// テストとベンチ用 (本番経路は `linear_gemm` を使う)。
    pub fn linear_gemm_i8_dev(
        &self,
        aq: &DevBuffer,
        asc: &DevBuffer,
        w: &DevBuffer,
        scale: &DevBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        let dev_out = DevBuffer::alloc(m * n * 4)?;
        unsafe {
            let (mut paq, mut pasc, mut pw, mut psc, mut pout) =
                (aq.ptr, asc.ptr, w.ptr, scale.ptr, dev_out.ptr);
            let (mut mc, mut kc, mut nc) = (m as c_int, k as c_int, n as c_int);
            let mut args = [
                &mut paq as *mut _ as *mut c_void,
                &mut pasc as *mut _ as *mut c_void,
                &mut pw as *mut _ as *mut c_void,
                &mut psc as *mut _ as *mut c_void,
                &mut pout as *mut _ as *mut c_void,
                &mut mc as *mut _ as *mut c_void,
                &mut kc as *mut _ as *mut c_void,
                &mut nc as *mut _ as *mut c_void,
            ];
            let gmx = m.div_ceil(GEMM_TILE) as u32;
            let gmy = n.div_ceil(GEMM_TILE_N) as u32;
            self.launch_async(self.f_gemm, (gmx, gmy, 1), (GEMM_BX, GEMM_BY, 1), 0, &mut args)?;
            self.sync()?;
        }
        dev_out.to_vec::<f32>(m * n)
    }

    /// P17i: ホストのアクティベーションから tiled GEMM を実行 (量子化込み)。
    pub fn linear_gemm_i8(
        &self,
        a: &[f32],
        w: &DevBuffer,
        scale: &DevBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        debug_assert_eq!(w.bytes(), k * gemm_pad_n(n));
        debug_assert_eq!(scale.bytes(), n * 4);
        let (aq, asc) = self.quantize_rows(a, m, k)?;
        self.linear_gemm_i8_dev(&aq, &asc, w, scale, m, k, n)
    }

    /// 重みを VRAM へアップロード (起動時に1度)
    pub fn upload_i8(&self, data: &[i8]) -> Result<DevBuffer> {
        DevBuffer::from_slice(data)
    }

    /// scale などを VRAM へアップロード (起動時に1度)
    pub fn upload_f32(&self, data: &[f32]) -> Result<DevBuffer> {
        DevBuffer::from_slice(data)
    }

    /// P-MOE-E1: Flash-Next の expert GEMV。CPU 参照
    /// (`crate::model::moe_expert::expert_gemv`) とビット一致する。
    ///
    /// 対象は実ファイルで使う 4 型のみ (Q8_0 / Q5_1 / Q4_K / Q5_K)。未対応の型、
    /// PTX 不在、カーネル取得失敗では `Ok(false)` を返し、呼び出し側が CPU 経路へ
    /// 落ちる (fail-closed。別経路の数値を黙って返さない)。
    ///
    /// カーネルは 1 block = 4 出力行 (64 スレッド/行)。行の生バイトを共有メモリへ
    /// コアレスで読み、その後 1 スレッドが i の昇順に f32 で累積する。
    pub fn moe_gemv(
        &self,
        ty: crate::kquant::GgmlType,
        w: &[u8],
        x: &[f32],
        n_in: usize,
        n_out: usize,
        out: &mut [f32],
    ) -> Result<bool> {
        use crate::kquant::GgmlType as T;
        let fs = match self.f_moe {
            Some(f) => f,
            None => return Ok(false),
        };
        let (f, rb) = match ty {
            T::Q8_0 => (fs[0], (n_in / 32) * 34),
            T::Q5_1 => (fs[1], (n_in / 32) * 24),
            T::Q4K => (fs[2], (n_in / 256) * 144),
            T::Q5K => (fs[3], (n_in / 256) * 176),
            _ => return Ok(false),
        };
        let bs = ty.blck_size();
        if n_in == 0 || n_out == 0 || n_in % bs != 0 {
            bail!("moe_gemv: 形状が不正 (n_in={n_in}, n_out={n_out}, block={bs})");
        }
        if w.len() != n_out * rb {
            bail!(
                "moe_gemv: w の長さ {} が n_out * row_bytes = {} と違う",
                w.len(),
                n_out * rb
            );
        }
        if x.len() != n_in {
            bail!("moe_gemv: x の長さ {} が n_in={n_in} と違う", x.len());
        }
        if out.len() != n_out {
            bail!("moe_gemv: out の長さ {} が n_out={n_out} と違う", out.len());
        }

        self.make_current()?;
        let dev_w = DevBuffer::from_slice(w)?;
        let used = self.moe_gemv_dev(ty, &dev_w, x, n_in, n_out, out)?;
        Ok(used)
    }

    /// P-gemv2: 使う GEMV カーネル一式を選ぶ。
    ///
    /// `DENNO_GEMV_V2=0` で v1 (従来) に固定できる。v2 が読めていなければ v1。
    /// v1 と v2 は**同じ数値契約** (i 昇順・f32 逐次加算・-fmad=false) なので、
    /// どちらを選んでも出力ビットは一致する。
    fn pick_moe_kernels(&self) -> Option<[CUfunction; 4]> {
        if moe_gemv_v2_on() {
            if let Some(f) = self.f_moe2 {
                return Some(f);
            }
        }
        self.f_moe
    }

    /// P-MOE-E5: 既に VRAM に載っている重み (`dev_w`) で GEMV する。
    /// 重みの H2D を含まないので、常駐キャッシュ越しに毎トークン呼べる。
    /// 引数検査は [`Self::moe_gemv`] と同じ (重みのバイト長は検査できない)。
    pub fn moe_gemv_dev(
        &self,
        ty: crate::kquant::GgmlType,
        dev_w: &DevBuffer,
        x: &[f32],
        n_in: usize,
        n_out: usize,
        out: &mut [f32],
    ) -> Result<bool> {
        // P-fp16 (task 108): fast mode (`DENNO_GEMV_FP16=1`) が最優先。off なら
        // 従来どおり v3 (`DENNO_GEMV_V3=1`) → v2 の順。env 未指定では V2 のまま
        // で 1 バイトも挙動が変わらない。
        let mode = gemv_kernel_mode();
        self.moe_gemv_dev_with(ty, dev_w, x, n_in, n_out, out, mode)
    }

    /// [`Self::moe_gemv_dev`] の実体。`mode` で使うカーネル一式を選ぶ。
    ///
    /// - [`GemvKernelMode::V2`]: ビット一致契約の既定経路。
    /// - [`GemvKernelMode::V3`]: レーン分割版 (`kernels_moe_v3.cu`)。k が 64 の
    ///   倍数でない / v3 の PTX が読めない / PTX が旧 ABI (引数 6 本) のときは
    ///   v2 へ落ちる。
    /// - [`GemvKernelMode::Fp16`]: fast mode (`kernels_moe_fp16.cu`)。対応型
    ///   (Q8_0 / Q5_1 / Q4_K) 以外・k が 64 の倍数でない・PTX が読めない /
    ///   旧 ABI のときは v3 (on なら) → v2 と同じ降下経路へ落ちる。
    ///
    /// 戻り値は「GPU で走ったか」(降下しても true)。fp16 / v3 が実際に走ったかは
    /// 呼び出し側のテストが「v2 とビット一致しないこと」で確かめる。
    /// decode GEMV 1 本ぶんの発行計画 (カーネル・起動形状・行バイト)。
    ///
    /// [`CudaCtx::moe_gemv_dev_with`] から切り出した**選択だけ**の部分で、
    /// デバイスメモリには触らない。task 124 の重ね打ち経路
    /// ([`shared_gemv_begin`]) と同じカーネル・同じ引数を組むために共有する。

    /// `ty` / `n_in` / `n_out` からカーネルと起動形状を決める (選択のみ)。
    ///
    /// 対象外の型・PTX が読めないときは `Ok(None)` (呼び出し側は CPU へ落ちる)。
    pub(crate) fn plan_moe_gemv(
        &self,
        ty: crate::kquant::GgmlType,
        n_in: usize,
        n_out: usize,
        mode: GemvKernelMode,
    ) -> Result<Option<MoeGemvPlan>> {
        use crate::kquant::GgmlType as T;
        // task 187: v5 を明示されたときは v5 の計画を先に試す (形が合わなければ
        // 従来どおり v2 へ落ちる)。
        if matches!(mode, GemvKernelMode::V5) {
            if let Some(p) = self.plan_moe_gemv_v5(ty, n_in, n_out) {
                return Ok(Some(p));
            }
        }
        let fs_v12 = match self.pick_moe_kernels() {
            Some(f) => f,
            None => return Ok(None),
        };
        let (ki, rb) = match ty {
            T::Q8_0 => (0usize, (n_in / 32) * 34),
            T::Q5_1 => (1usize, (n_in / 32) * 24),
            T::Q4K => (2usize, (n_in / 256) * 144),
            T::Q5K => (3usize, (n_in / 256) * 176),
            _ => return Ok(None),
        };
        // P-gemv3: 1 行を lanes レーン (4 or 2) で分担する版 (kernels_moe_v3.cu)。
        // P-fp16 (task 108): fast mode は v3 と同じ 8 引数 ABI・同じレーン分割で、
        // 中の演算だけ fp16 dequant + half2 FMA に変える。
        let fp16 = match mode {
            GemvKernelMode::Fp16 if gemv_fp16_ty_ok(ty) => {
                let lanes = moe_gemv_v3_lanes(n_in);
                if lanes == 0 {
                    None
                } else {
                    gemv_fp16_kernels().map(|k| (k.f[ki], lanes))
                }
            }
            _ => None,
        };
        let v3 = if fp16.is_none() && matches!(mode, GemvKernelMode::V3 | GemvKernelMode::Fp16) {
            let lanes = moe_gemv_v3_lanes(n_in);
            if lanes == 0 {
                None
            } else {
                gemv_v3_kernels().map(|k| (k.f[ki], lanes))
            }
        } else {
            None
        };
        const MOE_ROWS_PER_BLOCK: usize = 4; // kernels_moe.cu の MOE_ROWS と一致
        // fp16 / v3 の blockDim は rows*lanes (既定 rows=8/lanes=4 で 32)。v2 は 256 固定。
        // rows は DENNO_GEMV_V3_ROWS を共有する (起動形状の意味は同じ)。
        let (f, rows_per_block, block_dim, shmem, lanes) = match (fp16, v3) {
            (Some((f, lanes)), _) => {
                let rows = moe_gemv_v3_rows();
                (f, rows, (rows * lanes) as u32, 0u32, lanes)
            }
            (None, Some((f3, lanes))) => {
                let rows = moe_gemv_v3_rows();
                (f3, rows, (rows * lanes) as u32, 0u32, lanes)
            }
            (None, None) => (
                fs_v12[ki],
                MOE_ROWS_PER_BLOCK,
                256u32,
                (MOE_ROWS_PER_BLOCK * rb) as u32,
                1usize,
            ),
        };
        let grid = ((n_out + rows_per_block - 1) / rows_per_block) as u32;
        Ok(Some(MoeGemvPlan {
            f,
            rows_per_block,
            block_dim,
            shmem,
            lanes,
            rb,
            grid,
        }))
    }

    /// task 177: v4 (レーン数可変) の計画。形が合わない / PTX が無いときは `None`
    /// (呼び出し側は v3 へ落ちる)。
    ///
    /// 1 ブロック 8 行は v3 の既定 (`MOE_V3_ROWS_DEFAULT = 8`) と同じ。blockDim は
    /// `rows * lanes` になるので、lanes=32 で 256、lanes=16 で 128。
    pub(crate) fn plan_moe_gemv_v4(
        &self,
        ty: crate::kquant::GgmlType,
        n_in: usize,
        n_out: usize,
        lanes_cap: usize,
    ) -> Option<MoeGemvPlan> {
        use crate::kquant::GgmlType as T;
        let ki = match ty {
            T::Q8_0 => 0usize,
            T::Q5_1 => 1usize,
            T::Q4K => 2usize,
            T::Q5K => 3usize,
            _ => return None,
        };
        let rb = gguf_row_bytes(ty, n_in)?;
        if n_out == 0 {
            return None;
        }
        let lanes = v4_lanes_for(n_in, lanes_cap);
        if lanes == 0 {
            return None;
        }
        let k = gemv_v4_kernels()?;
        const ROWS: usize = 8;
        Some(MoeGemvPlan {
            f: k.f[ki],
            rows_per_block: ROWS,
            block_dim: (ROWS * lanes) as u32,
            shmem: 0,
            lanes,
            rb,
            grid: n_out.div_ceil(ROWS) as u32,
        })
    }

    /// task 184: v5 (ベクタロード + 複数アキュムレータ + 先読み) の計画。
    /// 形が合わない / PTX が無いときは `None` (呼び出し側は v4 へ落ちる)。
    ///
    /// レーン数は「レーンの担当単位が割り切れる」最大の 2 の冪:
    /// - Q8_0: ブロック対 (nb = n_in/32 が偶数であること) を `np = nb/2` 個
    /// - Q4_K: スーパーブロック (nb = n_in/256) を `nb` 個
    /// - Q5_K (task 192): Q4_K と同じくスーパーブロックを `nb` 個
    /// 1 ブロック 8 行は v4 と同じ。Q5_K は `DENNO_GDN_DEC_GEMV_Q5K=0` で v4 へ。
    pub(crate) fn plan_moe_gemv_v5(
        &self,
        ty: crate::kquant::GgmlType,
        n_in: usize,
        n_out: usize,
    ) -> Option<MoeGemvPlan> {
        use crate::kquant::GgmlType as T;
        let rb = gguf_row_bytes(ty, n_in)?;
        if n_out == 0 || n_in == 0 {
            return None;
        }
        let (f, per_unit) = match ty {
            T::Q8_0 => {
                let nb = n_in / 32;
                if nb < 2 || nb % 2 != 0 {
                    return None;
                }
                (gemv_v5_kernels()?.f[0], nb / 2)
            }
            T::Q4K => {
                let nb = n_in / 256;
                if nb == 0 {
                    return None;
                }
                (gemv_v5_kernels()?.f[1], nb)
            }
            T::Q5K => {
                if !gdn_dec_gemv_q5k_on() {
                    return None;
                }
                let nb = n_in / 256;
                if nb == 0 {
                    return None;
                }
                (gemv_v5_q5k_kernels()?.f, nb)
            }
            _ => return None,
        };
        // task 281: lane 数の上限は env (`DENNO_GEMV_V5_LANES`。既定 8 = 従来と同じ)。
        let lanes = v5_lanes_div(per_unit, v5_lanes_cap());
        if lanes == 0 {
            return None;
        }
        const ROWS: usize = 8;
        Some(MoeGemvPlan {
            f,
            rows_per_block: ROWS,
            block_dim: (ROWS * lanes) as u32,
            shmem: 0,
            lanes,
            rb,
            grid: n_out.div_ceil(ROWS) as u32,
        })
    }

    /// task 310: shared expert 用の GEMV 計画 (`DENNO_DEC_SH_V5`)。
    ///
    /// `allow_v5` が真なら [`Self::plan_moe_gemv_v5`] を先に引き、形が合わなければ
    /// 従来どおり [`Self::plan_moe_gemv`] (v3 / v2) へ落ちる。v5 の計画は
    /// 引数の並びが v3 と同一 (w, x, out, n_in, n_out, rb, rows_per_block, lanes)
    /// なので、[`Self::launch_moe_gemv_on`] は計画をそのまま積める。
    pub(crate) fn plan_shared_gemv(
        &self,
        ty: crate::kquant::GgmlType,
        n_in: usize,
        n_out: usize,
        mode: GemvKernelMode,
        allow_v5: bool,
    ) -> Option<MoeGemvPlan> {
        if allow_v5 {
            if let Some(p) = self.plan_moe_gemv_v5(ty, n_in, n_out) {
                return Some(p);
            }
        }
        self.plan_moe_gemv(ty, n_in, n_out, mode).ok().flatten()
    }

    /// decode GEMV を 1 本積む (task 124 の重ね打ち経路)。
    ///
    /// 引数の並びは [`Self::moe_gemv_dev_with`] が組むものと同一
    /// (w, x, out, n_in, n_out, row_bytes, [rows_per_block, lanes])。
    /// `stream` が null なら従来と同じ既定ストリーム。
    pub(crate) fn launch_moe_gemv_on(
        &self,
        plan: &MoeGemvPlan,
        w: CUdeviceptr,
        x: CUdeviceptr,
        o: CUdeviceptr,
        n_in: usize,
        n_out: usize,
        stream: CUstream,
    ) -> Result<()> {
        let (mut pw, mut px, mut po) = (w, x, o);
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
            self.launch_async_on(
                stream,
                plan.f,
                (plan.grid, 1, 1),
                (plan.block_dim, 1, 1),
                plan.shmem,
                &mut args[..nargs],
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn moe_gemv_dev_with(
        &self,
        ty: crate::kquant::GgmlType,
        dev_w: &DevBuffer,
        x: &[f32],
        n_in: usize,
        n_out: usize,
        out: &mut [f32],
        mode: GemvKernelMode,
    ) -> Result<bool> {
        self.moe_gemv_dev_with_pref(ty, dev_w, x, n_in, n_out, out, mode, false)
    }

    /// task 307 (`DENNO_GGUF_GEMV_V5`): dense の gguf GEMV も v5 を先に試す版。
    ///
    /// `prefer_v5` が真なら `plan_moe_gemv_v5` を先に引き、形が合わなければ
    /// 従来どおり `plan_moe_gemv` (v3 / v2) へ落ちる。カーネル・引数・丸め順は
    /// v5 のものがそのまま使われる (v3 / v2 とは加算順が違う = 丸めが変わる)。
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn moe_gemv_dev_with_pref(
        &self,
        ty: crate::kquant::GgmlType,
        dev_w: &DevBuffer,
        x: &[f32],
        n_in: usize,
        n_out: usize,
        out: &mut [f32],
        mode: GemvKernelMode,
        prefer_v5: bool,
    ) -> Result<bool> {
        let bs = ty.blck_size();
        if n_in == 0 || n_out == 0 || n_in % bs != 0 {
            bail!("moe_gemv_dev: 形状が不正 (n_in={n_in}, n_out={n_out}, block={bs})");
        }
        if x.len() != n_in {
            bail!("moe_gemv_dev: x の長さ {} が n_in={n_in} と違う", x.len());
        }
        if out.len() != n_out {
            bail!("moe_gemv_dev: out の長さ {} が n_out={n_out} と違う", out.len());
        }
        // カーネルの選択 (従来は `pick_moe_kernels` が先頭、fp16/v3 の選択が
        // 引数検査の後だった。PTX を読む順序だけが変わり、選ばれるカーネル・
        // 引数・丸め順は 1 つも変わらない)。
        // task 307: `prefer_v5` のときだけ v5 の計画を先に引く (形が合わなければ
        // 従来の選択へ落ちる)。
        let plan_v5 = if prefer_v5 {
            self.plan_moe_gemv_v5(ty, n_in, n_out)
        } else {
            None
        };
        let Some(plan) = (match plan_v5 {
            Some(p) => Some(p),
            None => self.plan_moe_gemv(ty, n_in, n_out, mode)?,
        }) else {
            return Ok(false);
        };
        let MoeGemvPlan {
            f,
            rows_per_block,
            block_dim,
            shmem,
            lanes,
            rb,
            grid,
        } = plan;
        if dev_w.bytes() != n_out * rb {
            bail!(
                "moe_gemv_dev: 重みのデバイスバイト数 {} が n_out * row_bytes = {} と違う",
                dev_w.bytes(),
                n_out * rb
            );
        }

        self.make_current()?;
        // P-pf-moe-scratch: 入出力の一時バッファは thread_local の grow-only
        // スクラッチから取る (従来は 1 呼び出しごとに cuMemAlloc 2 本 +
        // cuMemFree 2 本を回していた。decode 中に数万回)。
        // 数値は不変: カーネル・引数・発行順はそのままで、x は n_in 要素・
        // out は n_out 要素の範囲しか読み書きしない (前回の残りは読まない)。
        MOE_SCRATCH.with(|cell| -> Result<()> {
            let mut s = cell.borrow_mut();
            let s: &mut MoeScratch = &mut s;
            MoeScratch::ensure_tagged(&mut s.x, n_in * 4, crate::mem_audit::TAG_GEMV)?;
            MoeScratch::ensure_tagged(&mut s.o, n_out * 4, crate::mem_audit::TAG_GEMV)?;
            let async_on = gemv_async_on();
            if async_on {
                MoeScratch::ensure_pin(&mut s.pin_x, n_in * 4)?;
                MoeScratch::ensure_pin(&mut s.pin_o, n_out * 4)?;
            }
            let xb = s.x.as_ref().expect("x は ensure 済み");
            let ob = s.o.as_ref().expect("o は ensure 済み");
            let tr = gemv_trace_allow();
            let t0 = tr.then(std::time::Instant::now);
            if async_on {
                // P-async: pinned → dev の H2D を非同期で積み、カーネルを積み、
                // dev → pinned の D2H を非同期で積んで、**最後に 1 回だけ**同期する。
                // 従来は「同期 H2D / カーネル / 明示 sync / 同期 D2H」で暗黙の同期が
                // 3 回入っていた。カーネル・引数・発行順・丸め順は従来と同一。
                {
                    let px = s.pin_x.as_mut().expect("pin_x は ensure 済み");
                    dec_kprof::step("gemv_h2d_pack", || {
                        px.as_mut_slice::<f32>()[..n_in].copy_from_slice(x)
                    });
                    let lib = lib()?;
                    // task 179: 非同期コピーも表に出す (`H2D@a`)。
                    let kp_cp = dec_kprof::op_begin(
                        || "H2D@a".to_string(),
                        (0, 0, 0),
                        (0, 0, 0),
                        0,
                        n_in * 4,
                    );
                    unsafe {
                        check(
                            (lib.memcpy_htod_async)(
                                xb.ptr,
                                px.ptr as *const c_void,
                                n_in * 4,
                                std::ptr::null_mut(),
                            ),
                            "cuMemcpyHtoDAsync (moe gemv)",
                        )?;
                    }
                    dec_kprof::op_end(kp_cp);
                }
                dbg_htod(n_in * 4);
                let t1 = t0.map(|_| std::time::Instant::now());
                let t_sync = crate::model::devprof::sync_start();
                unsafe {
                    let (mut pw, mut px, mut po) = (dev_w.ptr, xb.ptr, ob.ptr);
                    let (mut n_in_c, mut n_out_c, mut rb_c) =
                        (n_in as c_int, n_out as c_int, rb as c_int);
                    // v3 は rows_per_block と lanes を余分に渡す (v2 は 6 本のまま)。
                    let (mut rows_c, mut lanes_c) = (rows_per_block as c_int, lanes as c_int);
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
                    let nargs = if lanes > 1 { 8 } else { 6 };
                    self.launch_async(
                        f,
                        (grid, 1, 1),
                        (block_dim, 1, 1),
                        shmem,
                        &mut args[..nargs],
                    )?;
                }
                let t2 = t0.map(|_| std::time::Instant::now());
                {
                    let po = s.pin_o.as_mut().expect("pin_o は ensure 済み");
                    let lib = lib()?;
                    // task 179: 非同期コピーも表に出す (`D2H@a`)。
                    let kp_cp = dec_kprof::op_begin(
                        || "D2H@a".to_string(),
                        (0, 0, 0),
                        (0, 0, 0),
                        0,
                        n_out * 4,
                    );
                    unsafe {
                        check(
                            (lib.memcpy_dtoh_async)(
                                po.ptr,
                                ob.ptr,
                                n_out * 4,
                                std::ptr::null_mut(),
                            ),
                            "cuMemcpyDtoHAsync (moe gemv)",
                        )?;
                    }
                    dec_kprof::op_end(kp_cp);
                }
                // task 368: 読むのは null ストリームの非同期 D2H が書いた pinned
                // だけなので、待つ範囲を null ストリームに狭めてよい
                // (`DENNO_DEC_SYNC_STREAM`、既定 off)。
                self.sync_decode()?;
                let t3 = t0.map(|_| std::time::Instant::now());
                if let Some(t) = t_sync {
                    crate::model::devprof::note_sync(t.elapsed().as_nanos() as u64);
                }
                let po = s.pin_o.as_ref().expect("pin_o は ensure 済み");
                dec_kprof::step("gemv_pack", || {
                    out.copy_from_slice(&po.as_slice::<f32>()[..n_out])
                });
                if let (Some(a), Some(b), Some(c), Some(d)) = (t0, t1, t2, t3) {
                    eprintln!(
                        "[gemvt] {}x{} async=1 h2d={} launch={} sync={} copy={}us",
                        n_out,
                        n_in,
                        (b - a).as_micros(),
                        (c - b).as_micros(),
                        (d - c).as_micros(),
                        std::time::Instant::now().duration_since(d).as_micros()
                    );
                }
                return Ok(());
            }
            xb.copy_from(x)?;
            // H2D の計上は従来の `from_slice` と同じ (1 呼び出し = n_in * 4 B)。
            dbg_htod(n_in * 4);
            let t1 = t0.map(|_| std::time::Instant::now());
            // devprof: 起動後の待ち (sync) + 結果読み出し (同期コピー) の実測。
            // off なら `None` で Instant を取らない。
            let t_sync = crate::model::devprof::sync_start();
            unsafe {
                let (mut pw, mut px, mut po) = (dev_w.ptr, xb.ptr, ob.ptr);
                let (mut n_in_c, mut n_out_c, mut rb_c) =
                    (n_in as c_int, n_out as c_int, rb as c_int);
                // v3 は rows_per_block と lanes を余分に渡す (v2 は 6 本のまま)。
                let (mut rows_c, mut lanes_c) = (rows_per_block as c_int, lanes as c_int);
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
                let nargs = if lanes > 1 { 8 } else { 6 };
                self.launch_async(
                    f,
                    (grid, 1, 1),
                    (block_dim, 1, 1),
                    shmem,
                    &mut args[..nargs],
                )?;
                // task 368: 直後の `to_vec` は null ストリームの同期コピーなので、
                // 待つ範囲を null ストリームに狭めてよい (`DENNO_DEC_SYNC_STREAM`)。
                self.sync_decode()?;
            }
            let got: Vec<f32> = ob.to_vec(n_out)?;
            let t3 = t0.map(|_| std::time::Instant::now());
            if let Some(t) = t_sync {
                crate::model::devprof::note_sync(t.elapsed().as_nanos() as u64);
            }
            out.copy_from_slice(&got);
            if let (Some(a), Some(b), Some(c)) = (t0, t1, t3) {
                eprintln!(
                    "[gemvt] {}x{} async=0 h2d={} launch+sync+d2h={}us",
                    n_out,
                    n_in,
                    (b - a).as_micros(),
                    (c - b).as_micros()
                );
            }
            Ok(())
        })?;
        Ok(true)
    }

    /// P39: expert の VRAM 常駐 GEMV 用の口。`shared_ctx` を隠したまま
    /// 既に VRAM にある `DevBuffer` 1 本で GEMV する。
    ///
    /// `weights_for_gguf` (dense の uid 登録簿) は**通さない**ので、常駐の
    /// 置き場所が 2 か所に分かれることはない。
    ///
    /// `Ok(false)` は「この型・この環境では GPU で回せない」。呼び出し側は
    /// CPU へ落ちること (fail closed)。
    #[allow(clippy::too_many_arguments)]
    pub fn moe_gemv_dev_buf(
        ty: crate::kquant::GgmlType,
        dev_w: &DevBuffer,
        x: &[f32],
        n_in: usize,
        n_out: usize,
        out: &mut [f32],
    ) -> Result<bool> {
        let Some(ctx) = shared_ctx() else {
            return Ok(false);
        };
        if let Err(e) = ctx.make_current() {
            tracing::error!("CUDA cuCtxSetCurrent failed (expert gemv): {e:#}");
            GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
            return Ok(false);
        }
        ctx.moe_gemv_dev(ty, dev_w, x, n_in, n_out, out)
    }


    /// task 261: 旧 4 型 (Q8_0 / Q5_1 / Q4_K / Q5_K) を回すカーネルの有無
    /// `(m=1 の GEMV, v2 GEMV, prefill GEMM)`。診断用 (数値経路に影響しない)。
    pub fn expert_kernel_status() -> (bool, bool, bool) {
        match shared_ctx() {
            Some(c) => (c.f_moe.is_some(), c.f_moe2.is_some(), c.f_gemm_q.is_some()),
            None => (false, false, false),
        }
    }

    /// P39: expert を VRAM に載せてよいか (カーネルと型が対応しているか)。
    ///
    /// 載せてから `false` で落ちると VRAM が無駄に埋まるので、**載せる前**に
    /// これを確かめる。`DENNO_CUDA` が無効なら false。
    pub fn moe_gemv_dev_ready(ty: crate::kquant::GgmlType, k: usize) -> bool {
        if !cuda_enabled() || gguf_row_bytes(ty, k).is_none() {
            return false;
        }
        shared_ctx()
            .map(|c| c.f_moe.is_some())
            .unwrap_or(false)
    }

    /// task 266: デバイスにある重み・入力・出力で GEMV を **1 回積むだけ**。
    ///
    /// [`Self::moe_gemv_dev`] は H2D / 起動 / 同期 / D2H を 1 呼び出しでやるが、
    /// mix の鎖 (down → silu → up → hc_mixed) は中間をデバイスに残したまま
    /// 続けたいので、起動だけを切り出す。完了待ちは呼び出し側が鎖の末尾で 1 回。
    ///
    /// `false` = この型・形では GPU で回せない (呼び出し側は CPU へ)。
    pub fn moe_gemv_dev_ptrs(
        &self,
        ty: crate::kquant::GgmlType,
        dev_w: CUdeviceptr,
        dev_x: CUdeviceptr,
        dev_out: CUdeviceptr,
        n_in: usize,
        n_out: usize,
    ) -> Result<bool> {
        let mode = gemv_kernel_mode();
        let Some(plan) = self.plan_moe_gemv(ty, n_in, n_out, mode)? else {
            return Ok(false);
        };
        unsafe {
            let (mut pw, mut px, mut po) = (dev_w, dev_x, dev_out);
            let (mut n_in_c, mut n_out_c, mut rb_c) =
                (n_in as c_int, n_out as c_int, plan.rb as c_int);
            let (mut rows_c, mut lanes_c) =
                (plan.rows_per_block as c_int, plan.lanes as c_int);
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
            self.launch_async(
                plan.f,
                (plan.grid, 1, 1),
                (plan.block_dim, 1, 1),
                plan.shmem,
                &mut args[..nargs],
            )?;
        }
        Ok(true)
    }

    /// P-PF-GB: prefill 用の量子 GEMM (nt トークン一括, `kernels_gemm.ptx`)。
    ///
    /// 数値契約は **解除済み** (2026-10-28 のオペレーター指示)。`-fmad=true` で
    /// JIT するので FMA 融合が入る。prefill の比較は argmax / 出力 md5 で行う。
    ///
    /// `false` を返したら呼び出し側は CPU バッチへ落ちること (fail closed)。
    /// 落ちる条件: 未対応の型 / nt が 1 か 128 超 / 共有メモリが 46 KiB 超 /
    /// GPU 無効 / `DENNO_GPU_GEMM=0`。
    pub fn moe_gemm_dev(
        &self,
        ty: crate::kquant::GgmlType,
        dev_w: &DevBuffer,
        x: &[f32],
        nt: usize,
        n_in: usize,
        n_out: usize,
        out: &mut [f32],
    ) -> Result<bool> {
        use crate::kquant::GgmlType as T;
        const ROWS: usize = 16; // kernels_gemm.cu の GEMM_ROWS
        const NTG_MAX: usize = 8; // kernels_gemm.cu の GEMM_NTG_MAX
        // task 169: Q8_0 は BF16 テンソルコア GEMM (ゲート on・形が合うとき)。
        // 従来カーネルは 1 起動 128 トークンまでだが、こちらは 2048 トークンまでを
        // 1 起動で畳む (起動・同期・H2D/D2H の回数が 1/16 になる)。
        // task 176: v2 カーネルが使える形 (Q8_0 / Q4_K / Q5_K・出力行 128 の倍数・
        // 十分なブロック数) は v2 を優先し、だめなら br169 のカーネルへ。
        let mut tc_sel: Option<TcSel> = None;
        if pf_tc2_gemm_on() && pf_tc2_eligible(ty, nt, n_in, n_out, pf_tc_min_m()) {
            if let Some(f) = tc2_gemm_kernels() {
                tc_sel = Some(TcSel::V2(f));
            }
        }
        // task 181: N が小さい形 (shared expert 640 / out_proj 2560) は BN=64 版。
        if tc_sel.is_none() && pf_tc_smalln_on() && pf_tc_smalln_eligible(ty, nt, n_in, n_out, pf_tc_min_m())
        {
            if let Some(f) = tc_smalln_kernels() {
                tc_sel = Some(TcSel::Sn(f));
            }
        }
        if tc_sel.is_none()
            && matches!(ty, T::Q8_0)
            && pf_tc_gemm_on()
            && pf_tc_gemm_eligible(nt, n_in, n_out, pf_tc_min_m())
        {
            if let Some(f) = tc_gemm_kernels() {
                tc_sel = Some(TcSel::V1(f));
            }
        }
        if let Some(sel) = tc_sel {
            let Some(rb) = gguf_row_bytes(ty, n_in) else {
                return Ok(false);
            };
            if x.len() != nt * n_in || out.len() != nt * n_out {
                bail!(
                    "moe_gemm_dev(tc): 形状が不正 (nt={nt}, n_in={n_in}, n_out={n_out}, \
                     x={}, out={})",
                    x.len(),
                    out.len()
                );
            }
            if dev_w.bytes() != n_out * rb {
                bail!(
                    "moe_gemm_dev(tc): 重みのデバイスバイト数 {} が n_out * row_bytes = {} と違う",
                    dev_w.bytes(),
                    n_out * rb
                );
            }
            self.make_current()?;
            MOE_SCRATCH.with(|cell| -> Result<()> {
                let mut s = cell.borrow_mut();
                MoeScratch::ensure_tagged(&mut s.x, nt * n_in * 4, crate::mem_audit::TAG_GEMV)?;
                MoeScratch::ensure_tagged(&mut s.o, nt * n_out * 4, crate::mem_audit::TAG_GEMV)?;
                let xb = s.x.as_ref().expect("x は ensure 済み");
                let ob = s.o.as_ref().expect("o は ensure 済み");
                xb.copy_from(x)?;
                dbg_htod(nt * n_in * 4);
                let t_sync = crate::model::devprof::sync_start();
                match sel {
                    TcSel::V2(f) => tc2_gemm_launch(
                        self, f, ty, dev_w.ptr, xb.ptr, ob.ptr, nt, n_in, n_out, rb,
                    )?,
                    TcSel::Sn(f) => sn_gemm_launch(
                        self, f, ty, dev_w.ptr, xb.ptr, ob.ptr, nt, n_in, n_out, rb,
                    )?,
                    TcSel::V1(f) => {
                        tc_gemm_launch(self, f, dev_w.ptr, xb.ptr, ob.ptr, nt, n_in, n_out, rb)?
                    }
                }
                self.sync()?;
                let got: Vec<f32> = ob.to_vec(nt * n_out)?;
                if let Some(t) = t_sync {
                    crate::model::devprof::note_sync(t.elapsed().as_nanos() as u64);
                }
                out[..nt * n_out].copy_from_slice(&got);
                Ok(())
            })?;
            return Ok(true);
        }
        let Some(fs) = self.f_gemm_q else {
            return Ok(false);
        };
        // 1 起動は 128 トークンまでだが、大きい nt は下で 128 ずつに割るので
        // 上限は設けない (task 169 でプリフィルのまとめ幅を広げたため)。
        if nt <= 1 {
            return Ok(false);
        }
        let (f, rb, blk) = match ty {
            T::Q8_0 => (fs[0], (n_in / 32) * 34, 32usize),
            T::Q5_1 => (fs[1], (n_in / 32) * 24, 32usize),
            T::Q4K => (fs[2], (n_in / 256) * 144, 256usize),
            T::Q5K => (fs[3], (n_in / 256) * 176, 256usize),
            _ => return Ok(false),
        };
        if n_in == 0 || n_out == 0 || n_in % blk != 0 {
            return Ok(false);
        }
        if x.len() != nt * n_in || out.len() != nt * n_out {
            bail!(
                "moe_gemm_dev: 形状が不正 (nt={nt}, n_in={n_in}, n_out={n_out}, \
                 x={}, out={})",
                x.len(),
                out.len()
            );
        }
        if dev_w.bytes() != n_out * rb {
            bail!(
                "moe_gemm_dev: 重みのデバイスバイト数 {} が n_out * row_bytes = {} と違う",
                dev_w.bytes(),
                n_out * rb
            );
        }
        // 動的共有メモリ: 1 ブロックぶんのデコード済み f32 (ROWS × blk) だけ。
        // 行の生バイトはカーネルが global から直接読む (丸ごと載せると k=5120 で
        // 62 KiB になり 48 KiB の既定を超える = 実モデルで 1 度も発火しなかった原因)。
        let shmem = ROWS * blk * 4;
        if shmem > 46_000 {
            return Ok(false);
        }
        self.make_current()?;
        let grid = ((n_out + ROWS - 1) / ROWS) as u32;
        MOE_SCRATCH.with(|cell| -> Result<()> {
            let mut s = cell.borrow_mut();
            MoeScratch::ensure_tagged(&mut s.x, nt * n_in * 4, crate::mem_audit::TAG_GEMV)?;
            MoeScratch::ensure_tagged(&mut s.o, nt * n_out * 4, crate::mem_audit::TAG_GEMV)?;
            let xb = s.x.as_ref().expect("x は ensure 済み");
            let ob = s.o.as_ref().expect("o は ensure 済み");
            xb.copy_from(x)?;
            dbg_htod(nt * n_in * 4);
            // devprof: 起動後の待ち (sync) + 結果読み出し (同期コピー) の実測。
            let t_sync = crate::model::devprof::sync_start();
            // カーネルは 1 起動 128 トークンまで (`ntg` の上限)。TC 経路が使えず
            // 呼び出し側が 2,048 行までまとめてきた場合も、ここで 128 ずつに割れば
            // 同じ結果になる (x / out はデバイス上で連続)。
            let step = ROWS * NTG_MAX;
            let mut t0 = 0usize;
            while t0 < nt {
                let m = (nt - t0).min(step);
                unsafe {
                    let (mut pw, mut px, mut po) = (
                        dev_w.ptr,
                        xb.ptr + (t0 * n_in * 4) as CUdeviceptr,
                        ob.ptr + (t0 * n_out * 4) as CUdeviceptr,
                    );
                    let (mut nt_c, mut ni_c, mut no_c, mut rb_c) =
                        (m as c_int, n_in as c_int, n_out as c_int, rb as c_int);
                    let mut args = [
                        &mut pw as *mut _ as *mut c_void,
                        &mut px as *mut _ as *mut c_void,
                        &mut po as *mut _ as *mut c_void,
                        &mut nt_c as *mut _ as *mut c_void,
                        &mut ni_c as *mut _ as *mut c_void,
                        &mut no_c as *mut _ as *mut c_void,
                        &mut rb_c as *mut _ as *mut c_void,
                    ];
                    self.launch_async(
                        f,
                        (grid, 1, 1),
                        (ROWS as u32, 16, 1),
                        shmem as u32,
                        &mut args,
                    )?;
                }
                t0 += m;
            }
            unsafe {
                self.sync()?;
            }
            let got: Vec<f32> = ob.to_vec(nt * n_out)?;
            if let Some(t) = t_sync {
                crate::model::devprof::note_sync(t.elapsed().as_nanos() as u64);
            }
            out[..nt * n_out].copy_from_slice(&got);
            Ok(())
        })?;
        Ok(true)
    }
}


// 生ハンドル (CudaCtx / GroupedKernels と同じ規約)。
unsafe impl Send for V3Kernels {}

unsafe impl Sync for V3Kernels {}


unsafe impl Send for V4Kernels {}

unsafe impl Sync for V4Kernels {}


unsafe impl Send for V5Kernels {}

unsafe impl Sync for V5Kernels {}


unsafe impl Send for V5Q5KKernels {}

unsafe impl Sync for V5Q5KKernels {}


// 生ハンドル (V3Kernels と同じ規約)。
unsafe impl Send for Fp16Kernels {}

unsafe impl Sync for Fp16Kernels {}


// 生ハンドル (context / module / function)。CudaCtx と同じ規約。
unsafe impl Send for MoeBatchKernels {}

unsafe impl Sync for MoeBatchKernels {}


impl SegLayerScratch {
    pub(crate) fn new() -> Self {
        SegLayerScratch {
            stage: None,
            host: Vec::new(),
            g: None,
            u: None,
            act: None,
            h: None,
            out: None,
        }
    }
}


// 生ハンドル (context / module / function)。CudaCtx と同じ規約。
unsafe impl Send for SegKKernels {}

unsafe impl Sync for SegKKernels {}


// 生ハンドル (context / module / function)。CudaCtx と同じ規約。
unsafe impl Send for ExpIqKernels {}

unsafe impl Sync for ExpIqKernels {}


unsafe impl Send for ExpPfKernels {}

unsafe impl Sync for ExpPfKernels {}


unsafe impl Send for ExpTcKernels {}

unsafe impl Sync for ExpTcKernels {}


unsafe impl Send for ExpKtcKernels {}

unsafe impl Sync for ExpKtcKernels {}


impl TcSet {
    /// (gate/up の関数, down の関数)。slot は `exp_pf::slot` の値。
    pub(crate) fn pair(&self, slot_gu: usize, slot_d: usize) -> (CUfunction, CUfunction) {
        match self {
            TcSet::Iq(t) => (t.gu[slot_gu], t.dn[slot_d]),
            TcSet::K(t) => {
                let o = crate::model::exp_pf::K_SLOT0;
                (t.gu[slot_gu - o], t.dn[slot_d - o])
            }
        }
    }
}


// 生ポインタを保持するだけ (スレッド局所で共有しない)
unsafe impl Send for PinMem {}


impl PinMem {
    pub(crate) fn alloc(bytes: usize) -> Result<Self> {
        Self::alloc_flags(bytes, false)
    }

    /// task 165: `write_combined` は書き込み結合の pinned にする。
    ///
    /// ストリーミングのバウンスは **CPU が書いて DMA が読むだけ**なので、
    /// write-combined にすると書き込みの read-for-ownership が消える
    /// (キャッシュには載らない。CPU から読み直さないので問題ない)。
    /// 確保に失敗したら通常の pinned へ落ちる (数値はどちらも同じ)。
    pub(crate) fn alloc_flags(bytes: usize, write_combined: bool) -> Result<Self> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // PORTABLE: どのコンテキストからでも有効 (保険)
        const CU_MEMHOSTALLOC_PORTABLE: u32 = 0x01;
        const CU_MEMHOSTALLOC_WRITECOMBINED: u32 = 0x04;
        let bytes = bytes.max(1);
        let lib = lib()?;
        if write_combined {
            let flags = CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_WRITECOMBINED;
            let ok = unsafe {
                check((lib.mem_host_alloc)(&mut ptr, bytes, flags), "cuMemHostAlloc (wc)").is_ok()
            };
            if ok {
                return Ok(Self { ptr, bytes });
            }
        }
        unsafe {
            check(
                (lib.mem_host_alloc)(&mut ptr, bytes, CU_MEMHOSTALLOC_PORTABLE),
                "cuMemHostAlloc",
            )?;
        }
        Ok(Self { ptr, bytes })
    }

    pub(crate) fn as_slice<T>(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const T, self.bytes / std::mem::size_of::<T>()) }
    }

    pub(crate) fn as_mut_slice<T>(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut T, self.bytes / std::mem::size_of::<T>()) }
    }

    /// task 382: **device から直接見える** pinned (`CU_MEMHOSTALLOC_DEVICEMAP`)。
    ///
    /// 返り値の `CUdeviceptr` は同じホストメモリを指すので、カーネルは
    /// H2D 無しでこれを読み、D2H 無しでこれへ書ける (PCIe 越しの読み書きは
    /// ユニファイドアドレスで透過)。mix の鎖は 1 回 5.7 us の H2D と 2.6 us の
    /// D2H を払っていたので、そこを消すための器。
    ///
    /// DEVICEMAP が付かない環境 (古い driver / WSL など) では `Err` を返し、
    /// 呼び出し側は従来の H2D / D2H 経路へ落ちる (fail closed)。
    fn alloc_mapped(bytes: usize) -> Result<(Self, CUdeviceptr)> {
        const CU_MEMHOSTALLOC_PORTABLE: u32 = 0x01;
        const CU_MEMHOSTALLOC_DEVICEMAP: u32 = 0x02;
        let bytes = bytes.max(1);
        let lib = lib()?;
        let Some(get_dev) = lib.mem_host_get_device_ptr else {
            bail!("cuMemHostGetDevicePointer が無い");
        };
        let mut ptr: *mut c_void = std::ptr::null_mut();
        unsafe {
            check(
                (lib.mem_host_alloc)(
                    &mut ptr,
                    bytes,
                    CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP,
                ),
                "cuMemHostAlloc (devicemap)",
            )?;
        }
        let mut dev: CUdeviceptr = 0;
        unsafe {
            let rc = get_dev(&mut dev, ptr, 0);
            if rc != CUDA_SUCCESS {
                let _ = (lib.mem_host_free)(ptr);
                bail!("cuMemHostGetDevicePointer が失敗: {rc:?}");
            }
        }
        Ok((Self { ptr, bytes }, dev))
    }
}


impl MapMem {
    /// 必要バイト数を満たすまで確保し直す (足りていれば何もしない)。
    pub(crate) fn ensure(slot: &mut Option<MapMem>, bytes: usize) -> Result<()> {
        if let Some(m) = slot.as_ref() {
            if m.pin.bytes >= bytes.max(1) {
                return Ok(());
            }
        }
        *slot = Some(MapMem::alloc(bytes)?);
        Ok(())
    }

    pub(crate) fn alloc(bytes: usize) -> Result<Self> {
        let (pin, dev) = PinMem::alloc_mapped(bytes)?;
        Ok(Self { pin, dev })
    }
}


impl Drop for PinMem {
    fn drop(&mut self) {
        unsafe {
            let _ = (lib().unwrap().mem_host_free)(self.ptr);
        }
    }
}


// 生ポインタを保持するだけ (中身はページロックされたホストメモリで、
// CPU からも GPU からも見える。排他は呼び出し側の LRU が持つ)。
unsafe impl Send for PinnedHost {}

unsafe impl Sync for PinnedHost {}


impl PinnedHost {
    /// `bytes` バイトの pinned ホストメモリを確保する。0 バイトは失敗扱い。
    #[cfg(feature = "cuda")]
    pub fn alloc(bytes: usize) -> Option<Self> {
        if bytes == 0 {
            return None;
        }
        const CU_MEMHOSTALLOC_PORTABLE: u32 = 0x01;
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let lib = lib().ok()?;
        let ok = unsafe {
            check(
                (lib.mem_host_alloc)(&mut ptr, bytes, CU_MEMHOSTALLOC_PORTABLE),
                "cuMemHostAlloc (RAM tier)",
            )
            .is_ok()
        };
        if !ok || ptr.is_null() {
            return None;
        }
        Some(Self { ptr, bytes })
    }

    /// GPU の無いビルドでは常に失敗する (ヒープへ落ちる)。
    #[cfg(not(feature = "cuda"))]
    pub fn alloc(_bytes: usize) -> Option<Self> {
        None
    }

    pub fn ptr(&self) -> *mut u8 {
        self.ptr as *mut u8
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.bytes) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut u8, self.bytes) }
    }
}


impl Drop for PinnedHost {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        {
            if let Ok(l) = lib() {
                unsafe {
                    let _ = (l.mem_host_free)(self.ptr);
                }
            }
        }
    }
}


impl UpPinRing {
    pub(crate) const fn new() -> Self {
        Self {
            slots: Vec::new(),
            calls: 0,
        }
    }
}


impl Drop for CopyStream {
    fn drop(&mut self) {
        if let Ok(l) = lib() {
            if let Some(f) = l.stream_destroy {
                unsafe {
                    let _ = f(self.0);
                }
            }
        }
    }
}


impl Drop for CopyEvent {
    fn drop(&mut self) {
        if let Ok(l) = lib() {
            if let Some(f) = l.event_destroy {
                unsafe {
                    let _ = f(self.0);
                }
            }
        }
    }
}


impl Scratch {
    pub(crate) fn ensure(cap: &mut Option<DevBuffer>, bytes: usize) -> Result<()> {
        Self::ensure_tagged(cap, bytes, crate::mem_audit::TAG_OTHER)
    }
    /// task 149: 台帳のタグ付き [`Self::ensure`]。
    ///
    /// task 788: 入っているのが借用ビュー (プール区画) なら容量が足りていても
    /// 確保し直す。プールは prefill の終端で解放されるため、残ったビューは
    /// 解放済みの領域を指す (CUDA error 700 の原因)。
    pub(crate) fn ensure_tagged(cap: &mut Option<DevBuffer>, bytes: usize, tag: &'static str) -> Result<()> {
        let (cur, view) = cap
            .as_ref()
            .map(|b| (b.bytes(), b.is_view()))
            .unwrap_or((0, false));
        if let Some(want) = moe_scratch_alloc_bytes_borrowed(cur, bytes, view) {
            *cap = Some(DevBuffer::alloc_tagged(want, tag)?);
        }
        Ok(())
    }
    pub(crate) fn ensure_pin(cap: &mut Option<PinMem>, bytes: usize) -> Result<()> {
        if cap.as_ref().map(|b| b.bytes).unwrap_or(0) < bytes {
            *cap = Some(PinMem::alloc(bytes)?);
        }
        Ok(())
    }

    /// P58-3: arena から切り出す版の [`Self::ensure`]。既に十分な容量を持つ
    /// バッファが入っていれば何もしない (arena のビューでも個別確保でも同じ)。
    /// arena が無効・確保に失敗したときは従来どおり個別に確保する (fail-open)。
    pub(crate) fn ensure_arena(cap: &mut Option<DevBuffer>, key: u32, bytes: usize, tag: &'static str) -> Result<()> {
        if cap.as_ref().map(|b| b.bytes()).unwrap_or(0) >= bytes {
            return Ok(());
        }
        *cap = Some(match arena_alloc(key, bytes, tag) {
            Some(b) => b,
            None => DevBuffer::alloc_tagged(bytes, tag)?,
        });
        Ok(())
    }
}


impl ArenaPlan {
    /// チャンク内オフセットのアライン。
    pub(crate) fn align_up(n: usize) -> usize {
        (n + ARENA_ALIGN - 1) & !(ARENA_ALIGN - 1)
    }

    /// 予約を取る。既存の予約で足りれば**同じ位置**を返す (再利用)。
    /// 足りない場合は `Err(必要なチャンクのバイト数)` を返し、**状態は変えない**
    /// (呼び出し側がチャンクを確保して `add_chunk` してから再試行する)。
    pub(crate) fn reserve(&mut self, tid: u8, key: u32, bytes: usize) -> std::result::Result<(usize, usize), usize> {
        if let Some(&(c, off, have)) = self.slots.get(&(tid, key)) {
            if have >= bytes {
                return Ok((c, off));
            }
        }
        let need = Self::align_up(bytes.max(1));
        for li in 0..self.chunks.len() {
            let (cap, used) = self.chunks[li];
            if cap - used >= need {
                let off = used;
                self.chunks[li].1 = used + need;
                self.slots.insert((tid, key), (li, off, bytes));
                self.cur_bytes += need;
                self.high_water = self.high_water.max(self.cur_bytes);
                return Ok((li, off));
            }
        }
        // チャンクは 1,024 MiB 単位 (要求が 1 GiB を超えるときはその倍数)。
        let chunk = need.max(ARENA_CHUNK);
        Err(chunk.div_ceil(ARENA_CHUNK) * ARENA_CHUNK)
    }

    /// チャンクを 1 本足す (中身は未初期化 = 呼び出し側が全要素を書く前提)。
    pub(crate) fn add_chunk(&mut self, bytes: usize) {
        self.chunks.push((bytes, 0));
    }

    /// 全チャンクと予約を捨てる (リクエスト境界・中身は全て死んでいる)。
    pub(crate) fn reset(&mut self) {
        self.chunks.clear();
        self.slots.clear();
        self.cur_bytes = 0;
    }

    /// チャンクは残したまま使用位置と予約だけ捨てる (同じ容量で足りるとき)。
    pub(crate) fn reset_keep_chunks(&mut self) {
        for c in self.chunks.iter_mut() {
            c.1 = 0;
        }
        self.slots.clear();
        self.cur_bytes = 0;
    }

    /// チャンクの総容量。
    pub(crate) fn capacity(&self) -> usize {
        self.chunks.iter().map(|(c, _)| *c).sum()
    }

    /// 次のリクエストで事前確保するバイト数の推定 (純関数)。
    ///
    /// 前回の「1 行あたり使用量 x 今回の行数 x 1.1」を 1 GiB 単位に切り上げる。
    /// 前回の実績が無ければ 0 (= 推定なし・最初の切り出しでチャンクを足す)。
    pub(crate) fn target_for(prev_rows: usize, prev_high: usize, rows: usize) -> usize {
        if prev_rows == 0 || prev_high == 0 || rows == 0 {
            return 0;
        }
        let per_row = prev_high / prev_rows + 1;
        let want = per_row.saturating_mul(rows).saturating_mul(11) / 10;
        want.saturating_add(ARENA_CHUNK - 1) & !(ARENA_CHUNK - 1)
    }
}


#[cfg(feature = "cuda")]
impl Probe {
    #[inline]
    pub fn begin() -> Self {
        LINEAR_SCOPE.with(|(depth, any)| {
            if depth.get() == 0 {
                any.set(false);
            }
            depth.set(depth.get() + 1);
        });
        Probe { _not_send: std::marker::PhantomData }
    }

    /// この区間中に GPU 線形が 1 回でも走ったか。
    #[inline]
    pub fn any_on_gpu(&self) -> bool {
        LINEAR_SCOPE.with(|(_, any)| any.get())
    }
}


#[cfg(feature = "cuda")]
impl Drop for Probe {
    #[inline]
    fn drop(&mut self) {
        LINEAR_SCOPE.with(|(depth, any)| {
            let d = depth.get().saturating_sub(1);
            depth.set(d);
            // 最深 scope が閉じた時点でフラグは外側 scope の集積に残る
            // (外側 scope がまだ open ならその値に OR 済み)。
            if d == 0 {
                // 最外を閉じたら次回 begin() が false から始める
                any.set(false);
            }
        });
    }
}


impl std::ops::Deref for GgufWRef {
    type Target = GpuGgufW;
    fn deref(&self) -> &GpuGgufW {
        self.w
    }
}


impl Drop for GgufWRef {
    fn drop(&mut self) {
        if let Ok(mut g) = gguf_registry().lock() {
            g.lru.unpin(self.uid);
        }
    }
}


#[cfg(test)]
impl GemvInterceptGuard {
    /// フラグを立てる (Drop まで有効)。
    pub fn new() -> Self {
        GEMV_INTERCEPT_FOR_TEST.store(true, std::sync::atomic::Ordering::Relaxed);
        Self
    }
}


#[cfg(test)]
impl Drop for GemvInterceptGuard {
    fn drop(&mut self) {
        GEMV_INTERCEPT_FOR_TEST.store(false, std::sync::atomic::Ordering::Relaxed);
    }
}


impl SharedSlotBuf {
    pub(crate) fn new() -> Self {
        Self {
            dout: None,
            pin_o: None,
            n: 0,
            #[cfg(test)]
            fake: Vec::new(),
        }
    }
}


impl SharedPending {
    /// 何も積んでいないハンドル (ゲート off / GPU 不発 / 失敗)。
    pub fn none() -> Self {
        Self {
            n: 0,
            _not_send: std::marker::PhantomData,
        }
    }
    /// 積んだ本数 (0 = 重ね打ちしていない)。
    pub fn len(&self) -> usize {
        self.n
    }
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
}


impl Drop for SharedPending {
    fn drop(&mut self) {
        if self.n == 0 {
            return;
        }
        self.n = 0;
        OVERLAP_ABORTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let _ = SHARED_OVERLAP.try_with(|cell| {
            let mut s = cell.borrow_mut();
            if !shared_overlap_drain(&mut s) {
                s.broken = true;
            }
        });
    }
}


unsafe impl Send for ShGuKernels {}

unsafe impl Sync for ShGuKernels {}


unsafe impl Send for TcGemm {}

unsafe impl Sync for TcGemm {}


unsafe impl Send for Tc2Gemm {}

unsafe impl Sync for Tc2Gemm {}


unsafe impl Send for SnGemm {}

unsafe impl Sync for SnGemm {}


unsafe impl Send for Sn2Gemm {}

unsafe impl Sync for Sn2Gemm {}


unsafe impl Send for Sn3Gemm {}

unsafe impl Sync for Sn3Gemm {}

unsafe impl Send for Lmv32Fn {}

unsafe impl Sync for Lmv32Fn {}

unsafe impl Send for Lmv32RFn {}

unsafe impl Sync for Lmv32RFn {}

unsafe impl Send for Lmv32RSiluFn {}

unsafe impl Sync for Lmv32RSiluFn {}

unsafe impl Send for LgemmMtFn {}

unsafe impl Sync for LgemmMtFn {}


impl SharedOut<'_> {
    pub(crate) fn host_len(&self) -> Option<usize> {
        match self {
            SharedOut::Host(o) => Some(o.len()),
            SharedOut::Dev(_) => None,
        }
    }
}


unsafe impl Send for FdtKernels {}

unsafe impl Sync for FdtKernels {}


unsafe impl Send for Fdt2Kernels {}

unsafe impl Sync for Fdt2Kernels {}


unsafe impl Send for FcbgKernels {}

unsafe impl Sync for FcbgKernels {}


unsafe impl Send for BaGemmFns {}

unsafe impl Sync for BaGemmFns {}


impl Default for FcKprof {
    fn default() -> Self {
        Self {
            active: false,
            open: false,
            have: 0,
            call: 0,
            evs: Vec::new(),
            used: 0,
            meta: Vec::new(),
            calls: 0,
            tokens: 0,
        }
    }
}


impl KprofAcc {
    /// 表示行 (`calls == 0 && miss == 0` なら None = 行を出さない)。
    /// 捨てた call があるときは calls=0 でも行を出す — 黙って消えると
    /// 「測れていない」ことが見えなくなるため (P43)。
    pub(crate) fn line(&self) -> Option<String> {
        if self.calls == 0 && self.miss == 0 {
            return None;
        }
        Some(self.render())
    }

    /// 行そのもの。pre/state/scan = 3 カーネルの積算 ms、calls = 3 区間すべて
    /// 読めた call 数、miss = 捨てた call 数、total = 3 区間の和。
    pub(crate) fn render(&self) -> String {
        let ms = |ns: u64| ns as f64 / 1e6;
        format!(
            "[GDN_KPROF] pre={:.3} state={:.3} scan={:.3} calls={} miss={} total={:.3}",
            ms(self.pre_ns),
            ms(self.state_ns),
            ms(self.scan_ns),
            self.calls,
            self.miss,
            ms(self.pre_ns + self.state_ns + self.scan_ns)
        )
    }
}


impl Drop for KprofExitGuard {
    fn drop(&mut self) {
        gdn_kprof_report();
    }
}


// SAFETY: 保持するのは CUDA の不透明ハンドルだけで、アクセスは下の Mutex で
// 直列化する (実際には multi-slot 同時実行は無いが、念のため排他する)。
unsafe impl Send for KprofEvents {}


impl KprofCall {
    /// 既定ストリームに境界イベントを 1 本打つ (`idx` 0..=3)。
    pub(crate) fn mark(&mut self, idx: usize) {
        let Some(rec) = lib().ok().and_then(|l| l.event_record) else {
            return;
        };
        if unsafe { rec((self.lock.0)[idx], std::ptr::null_mut()) } == CUDA_SUCCESS {
            self.marks |= 1 << idx;
        }
    }

    /// 既存の同期が済んだ後に 3 区間を読んで積算する。
    pub(crate) fn finish(self) {
        let Some(elapsed) = lib().ok().and_then(|l| l.event_elapsed_time) else {
            KPROF_MISS.fetch_add(1, Ordering::Relaxed);
            return;
        };
        if self.marks != 0b1111 {
            // 4 本そろっていない = 区間の意味が無い。半端な数字は積まない。
            KPROF_MISS.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let e = &self.lock.0;
        let mut ms = [0f32; 3];
        for k in 0..3 {
            let mut v: f32 = 0.0;
            let rc = unsafe { elapsed(&mut v, e[k], e[k + 1]) };
            // 未完了 (NOT_READY)・非有限・負値は「読めなかった」としてこの
            // call を丸ごと捨てる (半端な数字を積むと以後の判断が全部その
            // 数字に引きずられる)。
            if rc != CUDA_SUCCESS || !v.is_finite() || v < 0.0 {
                KPROF_MISS.fetch_add(1, Ordering::Relaxed);
                return;
            }
            ms[k] = v;
        }
        kprof_add_ms(ms[0], ms[1], ms[2]);
    }
}


unsafe impl Send for Scan2Kernel {}

unsafe impl Sync for Scan2Kernel {}


unsafe impl Send for Scan3Kernel {}

unsafe impl Sync for Scan3Kernel {}
