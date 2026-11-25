//! denno-telemetry: 推論テレメトリのワイヤ形式 + 送出ハブ (P13, denno-trace 基盤)。
//!
//! 設計原則 (アーキテクチャ解説 §11):
//!
//! 1. **推論処理と完全分離** — emit は bounded channel への `try_send` のみ。
//!    満杯時はドロップして `Gap` フレームで欠落を明示する。**絶対に待たない**。
//! 2. **OFF なら完全ゼロコスト** — 既定 OFF (`DENNO_TELEMETRY` 無し)。
//!    `enabled()` の分岐のみで数値経路には一切触れない (bit-exact 規律と両立)。
//! 3. **自己完結** — 依存クレート無し (std のみ)。エンジン・サーバー・ビューア
//!    (将来の wasm 含む) が同一の形式を共有する。
//!
//! フレーム形式 (v1):
//!
//! ```text
//! magic "DNTL" (4B) | version (1B) | kind (1B) | payload...
//! payload 内の整数は LEB128 varint、確率は f16 ビット (u16)。
//! ```

use std::cell::RefCell;
use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicI8, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

/// フレーム種別 (kind バイト)
pub mod kind {
    pub const SERVER_INFO: u8 = 1;
    pub const SESSION_ADMIT: u8 = 2;
    pub const SESSION_DONE: u8 = 3;
    pub const PREFILL: u8 = 4;
    pub const TOKEN: u8 = 5;
    pub const STEP: u8 = 6;
    pub const OP_SUMMARY: u8 = 7;
    pub const KV_USAGE: u8 = 8;
    pub const GAP: u8 = 9;
    pub const TOP_K: u8 = 10;
    /// 1 層の forward 内で実行された op 列 (実行順)。
    pub const LAYER_RUN: u8 = 11;
    /// 1 層の活性 (P53: 3D ビューア用)。層の出力の |x| 上位 K 個。
    pub const LAYER_ACT: u8 = 12;
    /// 1 層の役割別活性 (P54: 3D ビューア用)。attn / gdn / router / shex / ple の
    /// 正規化 RMS。decode は 1 トークンにつき層数ぶん。
    pub const ROLE_ACT: u8 = 13;
    /// 1 層の VRAM 常駐 expert 一覧 (P54)。変化したときと 64 トークンごと。
    pub const EXPERT_RESIDENCY: u8 = 14;
    /// 置き場所の統計 (VRAM / RAM / SSD)。1 秒に 1 回 (P54)。
    pub const TIER_STATS: u8 = 15;
    /// 1 層の MoE で実際に選ばれた expert index と gate 重み (P56: 3D ビューア用)。
    ///
    /// 13..15 は arch_telemetry 側の RoleAct (13) / ExpertResidency (14) /
    /// TierStats (15) が使う。番号を詰めない (kind はブランチを跨いで同じ意味にする)。
    pub const ROUTER_TOPK: u8 = 16;

    /// task 220: ルータが選びたかったが介入で無効化されていた expert (ビューワ用)。
    pub const EXPERT_SKIP: u8 = 17;
}

pub const MAGIC: [u8; 4] = *b"DNTL";
pub const VERSION: u8 = 1;

/// [`Frame::ExpertResidency`] の `in_vram` の上限 (本)。
///
/// 1 層の expert は最大 256 本だが、ビューア側の配列を固定長で持てるように
/// 512 で切る (超過分は落とす。層の在庫が 512 を超えるモデルは今のところ無い)。
pub const RESIDENCY_MAX: usize = 512;

/// セッション終了理由 (`engine::StopReason` に対応)
pub mod done_reason {
    pub const END_TURN: u8 = 0;
    pub const MAX_TOKENS: u8 = 1;
    pub const STOP_SEQUENCE: u8 = 2;
    pub const ERROR: u8 = 3;
}

/// op id は `denno_engine::prof` の定数と同じ値 (0..N_OPS)。
/// テレメトリ側は名前を持たない (ビューアが対応表を持つ)。
pub const N_OPS: usize = 32; // prof::N_OPS (13) より十分大きい先頭分を使う

// ---------- f16 変換 (自前実装。丸めは round-to-nearest-even) ----------

/// f32 → f16 ビット (u16)。
pub fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127;
    let mant = bits & 0x007f_ffff;
    // f32 の NaN/Inf
    if exp == 128 {
        // mant != 0 → NaN (quiet bit 立て)、そうでなければ Inf
        return if mant != 0 { sign | 0x7e00 } else { sign | 0x7c00 };
    }
    // オーバーフロー → Inf
    if exp > 15 {
        return sign | 0x7c00;
    }
    // 2^-25 未満は非正規でも表現できない (2^-25 ちょうどは RNE で 0)
    if exp < -25 {
        return sign;
    }
    // アンダーフロー (正規で表現できない) → 非正規 (丸め後 0x400 に
    // 桁上がりしたら最小正規 2^-14 になる。これも正しい f16 値)
    if exp < -14 {
        // m (bit23 = 2^exp) を 2^-24 刻みに丸める: 右シフト量 = -exp - 1
        let sh = (-exp - 1) as u32;
        let m = mant | 0x0080_0000; // 暗黙の 1
        let mut m16 = m >> sh;
        let rem = m & ((1u32 << sh) - 1);
        let half: u32 = 1u32 << (sh - 1);
        if rem > half || (rem == half && (m16 & 1) == 1) {
            m16 += 1;
        }
        return sign | m16 as u16; // m16 ≤ 0x400 (0x400 = 最小正規)
    }
    // 通常: mantissa を 13bit 丸め (RNE)
    let half = 1u32 << 12;
    let mut m16 = mant >> 13;
    let rem = mant & 0x1fff;
    if rem > half || (rem == half && (m16 & 1) == 1) {
        m16 += 1;
    }
    if m16 == 0x800 {
        // 丸めで桁上がり → exponent +1, mantissa 0
        return sign | (((exp + 16) as u16) << 10);
    }
    sign | (((exp + 15) as u16) << 10) | m16 as u16
}

/// f16 ビット (u16) → f32。
pub fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x03ff) as u32;
    let bits = match exp {
        0 => {
            if mant == 0 {
                sign // ±0
            } else {
                // 非正規 → 正規化 (value = mant * 2^-24)
                let mut e: i32 = -1;
                let mut m = mant;
                while m & 0x0400 == 0 {
                    m <<= 1;
                    e -= 1;
                }
                m &= 0x03ff;
                // ループ k 回後 (e = -1-k): value = (m/2^10) * 2^(e+10)
                // = 2^(-14-k) * (1 + frac) → f32 指数 = 113 - k = 114 + e
                sign | (((e + 114) as u32) << 23) | (m << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (mant << 13), // Inf / NaN
        _ => sign | ((exp + 127 - 15) << 23) | (mant << 13),
    };
    f32::from_bits(bits)
}

// ---------- varint (LEB128) ----------

pub fn put_varint_u32(out: &mut Vec<u8>, mut v: u32) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            break;
        }
        out.push(b | 0x80);
    }
}

pub fn put_varint_u64(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            break;
        }
        out.push(b | 0x80);
    }
}

/// 戻り値: (値, 消費バイト数)
pub fn get_varint_u64(buf: &[u8]) -> Result<(u64, usize), String> {
    let mut v: u64 = 0;
    let mut shift = 0u32;
    for (i, &b) in buf.iter().enumerate() {
        if shift >= 64 {
            return Err("varint too long".into());
        }
        v |= ((b & 0x7f) as u64) << shift;
        if b & 0x80 == 0 {
            return Ok((v, i + 1));
        }
        shift += 7;
    }
    Err("varint truncated".into())
}

pub fn get_varint_u32(buf: &[u8]) -> Result<(u32, usize), String> {
    let (v, n) = get_varint_u64(buf)?;
    if v > u32::MAX as u64 {
        return Err("varint overflows u32".into());
    }
    Ok((v as u32, n))
}

// ---------- フレーム ----------

#[derive(Debug, Clone, PartialEq)]
pub enum Frame {
    /// WS 接続時のサーバー情報
    ServerInfo { model: String, model_id: String, quant: String, slots: u32, context: u32, layers: u32 },
    /// セッションが KV スロットを取得 (install)
    SessionAdmit { session: u32, slot: u32, prompt_tokens: u32 },
    /// セッション終了 (release)
    SessionDone { session: u32, reason: u8, output_tokens: u32 },
    /// prefill 1 forward 完了 (tokens = 処理したプロンプトトークン id 列)
    Prefill { session: u32, prompt_tokens: u32, ns: u64, tokens: Vec<u32> },
    /// decode で 1 トークン確定
    Token { session: u32, token: u32, pos: u32 },
    /// バッチ decode の forward 1 回分の所要時間 (session = バッチ全体)
    Step { width: u32, ns: u64 },
    /// 直近のステップ群での op 別累積 ns (prof::N_OPS 分、0 は省略)
    OpSummary { ops: Vec<(u8, u64)> },
    /// KV スロットの使用状況
    KvUsage { busy: Vec<bool>, t_lens: Vec<u32> },
    /// フレーム欠落 (チャネル満杯 / 購読遅延)
    Gap { lost: u32 },
    /// sampler の top-k (確率は f16 ビット)。
    ///
    /// task 853: **既定 off**。`DENNO_TELEMETRY_TOPK=1` のときだけ送出する
    /// ([`topk_telemetry_enabled`])。off のときは top-k の走査自体をしないので
    /// 推論経路のコストは分岐 1 個 (ゼロオーバーヘッド)。
    TopK { session: u32, chosen: u32, entropy_f16: u16, entries: Vec<(u32, u16)> },
    /// 1 層の forward で実行された op 列 (op id, ns)。層の実行順に送出される
    /// (P14: ビューアのコールグラフ/層パネルの点灯アニメーション用)。
    LayerRun { layer: u16, ops: Vec<(u8, u64)> },
    /// 1 層の活性 (P53: 3D ビューア用)。層の出力 (残差を足した後) の |x| 上位 K。
    ///
    /// - `layer`: 層番号
    /// - `idx`: |x| の上位 K 個のニューロン index。**昇順**に正規化して送る。
    /// - `val`: 対応する |x| を**層内最大**で割った値の f16 ビット (0.0〜1.0)
    ///
    /// フレームにトークン位置は無い (P53 の仕様)。受け手は直前の `Token` /
    /// `Prefill` と並び順で対応付ける。
    LayerAct { layer: u16, idx: Vec<u32>, val: Vec<u16> },
    /// 1 層の**役割別**活性 (P54: 3D ビューア用)。
    ///
    /// 各役割の出力ベクトルの RMS を、**その層の最大役割**で割った値 (0..1) を
    /// f16 ビットで送る。5 値とも同じ正規化なので、層の中での相対的な強さが読める。
    ///
    /// - `attn`: full attention 層の attention 出力の RMS (ssm 層では 0)
    /// - `gdn`: ssm (GDN) 層の scan 出力の RMS (full attention 層では 0)
    /// - `router`: router の top-k (k = n_expert_used = 10) 確率和。スカラーなので
    ///   RMS は絶対値そのもの
    /// - `shex`: shared expert の出力 (`ffn_down_shexp` の出力) の RMS
    /// - `ple`: PLE 注入の寄与 (`out - res` = gated + conv) の RMS
    ///
    /// 層内最大が 0 なら 5 値とも 0。decode は 1 トークンにつき層数 (48) 枚、
    /// prefill はチャンク末尾のトークンだけ。
    RoleAct { layer: u16, attn: u16, gdn: u16, router: u16, shex: u16, ple: u16 },
    /// その層で**いま VRAM に載っている** expert の index (P54)。
    ///
    /// `in_vram` は昇順・重複なし・最大 [`RESIDENCY_MAX`] 本 (超過分は切る)。
    /// 変化したときだけ送り、加えて 64 トークンごとに全層のスナップショットを送る。
    ExpertResidency { layer: u16, in_vram: Vec<u16> },
    /// 置き場所の統計 (P54)。1 秒に 1 回。
    ///
    /// - `vram_used_mb` / `vram_total_mb`: expert 用 VRAM の使用量 / 予算
    ///   (`DENNO_EXPERT_VRAM_MB`)。VRAM 経路が無効なら 0 / 0。
    /// - `ram_experts`: RAM キャッシュに載っている expert の数
    ///   (`DENNO_EXPERT_CACHE_MB`、無効なら 0)
    /// - `ssd_reads`: SSD からの読み (pread) の累計回数
    TierStats { vram_used_mb: u32, vram_total_mb: u32, ram_experts: u32, ssd_reads: u32 },
    /// 1 層の MoE で実際に選ばれた expert index と gate 重み (P56: 3D ビューア用)。
    ///
    /// - `layer`: 層番号
    /// - `experts`: 選ばれた expert index (`n_expert_used` 個以下)。**重みの降順**。
    ///   同値 (f16 に丸める前の f32 が同じ) は index 昇順。
    /// - `weights`: `weights[j]` は `experts[j]` の gate 重みの f16 ビット。
    ///   softmax → top-k 再正規化後の値 (合計 1、丸め前は f32)。
    ///
    /// フレームにトークン位置は無い ([`Frame::LayerAct`] と同じ規則)。受け手は
    /// 直前の [`Frame::Token`] / [`Frame::Prefill`] と並び順で対応付ける。
    ///
    /// 送出は `DENNO_TELEMETRY_ARCH` が立っているときだけ (既定 off)。
    /// decode は 1 トークンにつき層数ぶん、prefill はチャンク末尾の 1 トークンだけ。
    RouterTopK { layer: u16, experts: Vec<u16>, weights: Vec<u16> },
    /// 1 層の MoE で**ルータが選びたかったが、介入で無効化されていた** expert。
    ///
    /// `experts` は「無効化が無ければ top-k に入っていた expert のうち、実際には
    /// 使わなかったもの」を index 昇順で並べたもの。介入で expert を切っている層で
    /// だけ、[`Frame::RouterTopK`] と同じタイミング (decode は毎トークン、prefill は
    /// チャンク末尾) で送る。
    ///
    /// ビューワは「この expert はルータに選ばれていたが切った」をこれで描ける。
    /// **既定 off**: `DENNO_TELEM_EXPERT_SKIP=1` のときだけ作る (off はフラグ 1 個の
    /// 分岐だけで、確保も余分な top-k もしない)。`DENNO_TELEMETRY_ARCH` も要る。
    ///
    /// lab t220a (depth8k, sha c537e44): 介入なしの 5 COND (b0 / base / skip / iv /
    /// base2) がすべて md5 SAME、decode ±0.2%、prefill ±0.9%。gate off と gate on
    /// (skip) と介入あり (iv, `DENNO_INTERVENE=e3-17`) の差はノイズ帯で、既定 off の
    /// 経路にコストが無いことを実測で確認した。同じ scenario の int-auto (t220base,
    /// 56877d3) とも decode 0.0% / prefill 0.0% / md5 全項目 SAME。
    ExpertSkip { layer: u16, experts: Vec<u16> },
}

/// ペイロードの境界検査付き読み取りカーソル。
///
/// [`Frame::decode`] は信頼できない入力を扱う。index の直参照 (`p[i]`) や
/// スライス範囲 (`p[i..i+n]`) は切り詰め入力で panic するため、デコード経路の
/// 読み取りは全てここを通す (必ず `Result` を返し、panic しない)。
struct Cursor<'a> {
    p: &'a [u8],
    i: usize,
}

impl<'a> Cursor<'a> {
    fn new(p: &'a [u8]) -> Self {
        Self { p, i: 0 }
    }

    fn pos(&self) -> usize {
        self.i
    }

    fn len(&self) -> usize {
        self.p.len()
    }

    fn remaining(&self) -> usize {
        self.p.len().saturating_sub(self.i)
    }

    /// `n` バイト読み進める。足りなければ `Err` (確保はしない)。
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self
            .i
            .checked_add(n)
            .ok_or_else(|| "length overflow".to_string())?;
        if end > self.p.len() {
            return Err(format!(
                "truncated: need {n} bytes at {} / {}",
                self.i,
                self.p.len()
            ));
        }
        let s = &self.p[self.i..end];
        self.i = end;
        Ok(s)
    }

    fn byte(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn u16le(&mut self) -> Result<u16, String> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32, String> {
        let tail = self
            .p
            .get(self.i..)
            .ok_or_else(|| "cursor out of range".to_string())?;
        let (v, n) = get_varint_u32(tail)?;
        self.i += n;
        Ok(v)
    }

    fn u64(&mut self) -> Result<u64, String> {
        let tail = self
            .p
            .get(self.i..)
            .ok_or_else(|| "cursor out of range".to_string())?;
        let (v, n) = get_varint_u64(tail)?;
        self.i += n;
        Ok(v)
    }

    /// 要素数を読む。1 要素が最低 `min_elem` バイトなので、残りバイト数から
    /// 上限を決めて**過大な count を先に弾く** (`Vec::with_capacity(huge)` の
    /// 確保で落ちないため)。`min_elem == 0` は上限無し (呼び出し側の責務)。
    fn count(&mut self, min_elem: usize) -> Result<usize, String> {
        let n = self.u32()? as usize;
        if min_elem > 0 {
            if n > self.remaining() / min_elem {
                return Err(format!(
                    "count {n} が残り {} バイトに対して大きすぎる",
                    self.remaining()
                ));
            }
        }
        Ok(n)
    }
}

impl Frame {
    pub fn kind(&self) -> u8 {
        match self {
            Frame::ServerInfo { .. } => kind::SERVER_INFO,
            Frame::SessionAdmit { .. } => kind::SESSION_ADMIT,
            Frame::SessionDone { .. } => kind::SESSION_DONE,
            Frame::Prefill { .. } => kind::PREFILL,
            Frame::Token { .. } => kind::TOKEN,
            Frame::Step { .. } => kind::STEP,
            Frame::OpSummary { .. } => kind::OP_SUMMARY,
            Frame::KvUsage { .. } => kind::KV_USAGE,
            Frame::Gap { .. } => kind::GAP,
            Frame::TopK { .. } => kind::TOP_K,
            Frame::LayerRun { .. } => kind::LAYER_RUN,
            Frame::LayerAct { .. } => kind::LAYER_ACT,
            Frame::RoleAct { .. } => kind::ROLE_ACT,
            Frame::ExpertResidency { .. } => kind::EXPERT_RESIDENCY,
            Frame::TierStats { .. } => kind::TIER_STATS,
            Frame::RouterTopK { .. } => kind::ROUTER_TOPK,
            Frame::ExpertSkip { .. } => kind::EXPERT_SKIP,
        }
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(16);
        b.extend_from_slice(&MAGIC);
        b.push(VERSION);
        b.push(self.kind());
        match self {
            Frame::ServerInfo { model, model_id, quant, slots, context, layers } => {
                put_varint_u32(&mut b, model.len() as u32);
                b.extend_from_slice(model.as_bytes());
                put_varint_u32(&mut b, model_id.len() as u32);
                b.extend_from_slice(model_id.as_bytes());
                put_varint_u32(&mut b, quant.len() as u32);
                b.extend_from_slice(quant.as_bytes());
                put_varint_u32(&mut b, *slots);
                put_varint_u32(&mut b, *context);
                put_varint_u32(&mut b, *layers);
            }
            Frame::SessionAdmit { session, slot, prompt_tokens } => {
                put_varint_u32(&mut b, *session);
                put_varint_u32(&mut b, *slot);
                put_varint_u32(&mut b, *prompt_tokens);
            }
            Frame::SessionDone { session, reason, output_tokens } => {
                put_varint_u32(&mut b, *session);
                b.push(*reason);
                put_varint_u32(&mut b, *output_tokens);
            }
            Frame::Prefill { session, prompt_tokens, ns, tokens } => {
                put_varint_u32(&mut b, *session);
                put_varint_u32(&mut b, *prompt_tokens);
                put_varint_u64(&mut b, *ns);
                put_varint_u32(&mut b, tokens.len() as u32);
                for &t in tokens {
                    put_varint_u32(&mut b, t);
                }
            }
            Frame::Token { session, token, pos } => {
                put_varint_u32(&mut b, *session);
                put_varint_u32(&mut b, *token);
                put_varint_u32(&mut b, *pos);
            }
            Frame::Step { width, ns } => {
                put_varint_u32(&mut b, *width);
                put_varint_u64(&mut b, *ns);
            }
            Frame::OpSummary { ops } => {
                put_varint_u32(&mut b, ops.len() as u32);
                for (op, ns) in ops {
                    b.push(*op);
                    put_varint_u64(&mut b, *ns);
                }
            }
            Frame::KvUsage { busy, t_lens } => {
                // ワイヤ形式は busy の本数だけを書く (t_lens は同じ本数を前提)。
                // 長さが食い違うフレームでも encode → decode が壊れないよう、
                // 短い方に合わせて両方を同じ本数だけ書く。
                let n = busy.len().min(t_lens.len());
                put_varint_u32(&mut b, n as u32);
                for &x in &busy[..n] {
                    b.push(x as u8);
                }
                for &t in &t_lens[..n] {
                    put_varint_u32(&mut b, t);
                }
            }
            Frame::Gap { lost } => {
                put_varint_u32(&mut b, *lost);
            }
            Frame::TopK { session, chosen, entropy_f16, entries } => {
                put_varint_u32(&mut b, *session);
                put_varint_u32(&mut b, *chosen);
                b.extend_from_slice(&entropy_f16.to_le_bytes());
                put_varint_u32(&mut b, entries.len() as u32);
                for (id, p) in entries {
                    put_varint_u32(&mut b, *id);
                    b.extend_from_slice(&p.to_le_bytes());
                }
            }
            Frame::LayerRun { layer, ops } => {
                put_varint_u32(&mut b, *layer as u32);
                put_varint_u32(&mut b, ops.len() as u32);
                for (op, ns) in ops {
                    b.push(*op);
                    put_varint_u64(&mut b, *ns);
                }
            }
            Frame::LayerAct { layer, idx, val } => {
                // idx と val は同じ本数 (decode は idx の本数で両方を読む)。
                // 食い違いは短い方に合わせる (以前は val を全部書いていて、
                // 長さが違うフレームが decode できなかった)。
                let n = idx.len().min(val.len());
                put_varint_u32(&mut b, *layer as u32);
                put_varint_u32(&mut b, n as u32);
                for &i in &idx[..n] {
                    put_varint_u32(&mut b, i);
                }
                for &v in &val[..n] {
                    b.extend_from_slice(&v.to_le_bytes());
                }
            }
            Frame::RoleAct { layer, attn, gdn, router, shex, ple } => {
                put_varint_u32(&mut b, *layer as u32);
                for v in [attn, gdn, router, shex, ple] {
                    b.extend_from_slice(&v.to_le_bytes());
                }
            }
            Frame::ExpertResidency { layer, in_vram } => {
                put_varint_u32(&mut b, *layer as u32);
                put_varint_u32(&mut b, in_vram.len() as u32);
                for &i in in_vram {
                    put_varint_u32(&mut b, i as u32);
                }
            }
            Frame::TierStats { vram_used_mb, vram_total_mb, ram_experts, ssd_reads } => {
                put_varint_u32(&mut b, *vram_used_mb);
                put_varint_u32(&mut b, *vram_total_mb);
                put_varint_u32(&mut b, *ram_experts);
                put_varint_u32(&mut b, *ssd_reads);
            }
            Frame::RouterTopK { layer, experts, weights } => {
                // expert と重みは同じ本数 (decode は experts の本数で両方を読む)。
                // 食い違いは短い方に合わせる。
                let n = experts.len().min(weights.len());
                put_varint_u32(&mut b, *layer as u32);
                put_varint_u32(&mut b, n as u32);
                for &e in &experts[..n] {
                    put_varint_u32(&mut b, e as u32);
                }
                // 重みは experts と同じ本数を f16 ビット (2B, LE) で並べる。
                for &w in &weights[..n] {
                    b.extend_from_slice(&w.to_le_bytes());
                }
            }
            Frame::ExpertSkip { layer, experts } => {
                put_varint_u32(&mut b, *layer as u32);
                put_varint_u32(&mut b, experts.len() as u32);
                for &e in experts {
                    put_varint_u32(&mut b, e as u32);
                }
            }
        }
        b
    }

    /// WS メッセージ 1 個 (= フレーム 1 個) をデコードする。
    ///
    /// 入力は信頼できない (ネットワーク / 壊れたファイル) 前提。切り詰め・
    /// 過大な count・範囲外の index は全て `Err` を返し、**panic しない**
    /// (読み取りは [`Cursor`] 経由で境界検査する)。
    pub fn decode(buf: &[u8]) -> Result<Frame, String> {
        if buf.len() < 6 || buf[..4] != MAGIC {
            return Err("bad magic".into());
        }
        if buf[4] != VERSION {
            return Err(format!("unsupported version: {}", buf[4]));
        }
        let k = buf[5];
        let mut c = Cursor::new(&buf[6..]);
        let f = match k {
            kind::SERVER_INFO => {
                let n = c.u32()? as usize;
                let model = String::from_utf8(c.take(n)?.to_vec()).map_err(|e| e.to_string())?;
                let nid = c.u32()? as usize;
                let model_id = String::from_utf8(c.take(nid)?.to_vec()).map_err(|e| e.to_string())?;
                let n2 = c.u32()? as usize;
                let quant = String::from_utf8(c.take(n2)?.to_vec()).map_err(|e| e.to_string())?;
                let slots = c.u32()?;
                let context = c.u32()?;
                let layers = c.u32()?;
                Frame::ServerInfo { model, model_id, quant, slots, context, layers }
            }
            kind::SESSION_ADMIT => {
                let session = c.u32()?;
                let slot = c.u32()?;
                let prompt_tokens = c.u32()?;
                Frame::SessionAdmit { session, slot, prompt_tokens }
            }
            kind::SESSION_DONE => {
                let session = c.u32()?;
                let reason = c.byte()?;
                let output_tokens = c.u32()?;
                Frame::SessionDone { session, reason, output_tokens }
            }
            kind::PREFILL => {
                let session = c.u32()?;
                let prompt_tokens = c.u32()?;
                let ns = c.u64()?;
                // token は varint (最低 1B)
                let n = c.count(1)?;
                let mut tokens = Vec::with_capacity(n);
                for _ in 0..n {
                    tokens.push(c.u32()?);
                }
                Frame::Prefill { session, prompt_tokens, ns, tokens }
            }
            kind::TOKEN => {
                let session = c.u32()?;
                let token = c.u32()?;
                let pos = c.u32()?;
                Frame::Token { session, token, pos }
            }
            kind::STEP => {
                let width = c.u32()?;
                let ns = c.u64()?;
                Frame::Step { width, ns }
            }
            kind::OP_SUMMARY => {
                // 1 要素 = op (1B) + ns varint (最低 1B)
                let n = c.count(2)?;
                let mut ops = Vec::with_capacity(n);
                for _ in 0..n {
                    let op = c.byte()?;
                    let ns = c.u64()?;
                    ops.push((op, ns));
                }
                Frame::OpSummary { ops }
            }
            kind::KV_USAGE => {
                // 1 要素 = busy (1B) + t_len varint (最低 1B)
                let n = c.count(2)?;
                let mut busy = Vec::with_capacity(n);
                for _ in 0..n {
                    busy.push(c.byte()? != 0);
                }
                let mut t_lens = Vec::with_capacity(n);
                for _ in 0..n {
                    t_lens.push(c.u32()?);
                }
                Frame::KvUsage { busy, t_lens }
            }
            kind::GAP => {
                let lost = c.u32()?;
                Frame::Gap { lost }
            }
            kind::TOP_K => {
                let session = c.u32()?;
                let chosen = c.u32()?;
                let entropy_f16 = c.u16le()?;
                // 1 要素 = id varint (最低 1B) + 確率 (2B)
                let n = c.count(3)?;
                let mut entries = Vec::with_capacity(n);
                for _ in 0..n {
                    let id = c.u32()?;
                    let p = c.u16le()?;
                    entries.push((id, p));
                }
                Frame::TopK { session, chosen, entropy_f16, entries }
            }
            kind::LAYER_RUN => {
                let layer = c.u32()? as u16;
                // 1 要素 = op (1B) + ns varint (最低 1B)
                let n = c.count(2)?;
                let mut ops = Vec::with_capacity(n);
                for _ in 0..n {
                    let op = c.byte()?;
                    let ns = c.u64()?;
                    ops.push((op, ns));
                }
                Frame::LayerRun { layer, ops }
            }
            kind::LAYER_ACT => {
                let layer = c.u32()? as u16;
                // 1 要素 = idx varint (最低 1B) + val f16 (2B)
                let n = c.count(3)?;
                let mut idx = Vec::with_capacity(n);
                for _ in 0..n {
                    idx.push(c.u32()?);
                }
                let mut val = Vec::with_capacity(n);
                for _ in 0..n {
                    val.push(c.u16le()?);
                }
                Frame::LayerAct { layer, idx, val }
            }
            kind::ROLE_ACT => {
                let layer = c.u32()? as u16;
                let mut v = [0u16; 5];
                for slot in v.iter_mut() {
                    *slot = c.u16le()?;
                }
                Frame::RoleAct {
                    layer,
                    attn: v[0],
                    gdn: v[1],
                    router: v[2],
                    shex: v[3],
                    ple: v[4],
                }
            }
            kind::EXPERT_RESIDENCY => {
                let layer = c.u32()? as u16;
                let n = c.count(1)?;
                let mut in_vram = Vec::with_capacity(n);
                for _ in 0..n {
                    let e = c.u32()?;
                    if e > u16::MAX as u32 {
                        return Err(format!("expert_residency: expert index {e} が u16 を超える"));
                    }
                    in_vram.push(e as u16);
                }
                Frame::ExpertResidency { layer, in_vram }
            }
            kind::TIER_STATS => {
                let vram_used_mb = c.u32()?;
                let vram_total_mb = c.u32()?;
                let ram_experts = c.u32()?;
                let ssd_reads = c.u32()?;
                Frame::TierStats { vram_used_mb, vram_total_mb, ram_experts, ssd_reads }
            }
            kind::ROUTER_TOPK => {
                let layer = c.u32()? as u16;
                // 1 要素 = expert varint (最低 1B) + 重み (2B)
                let n = c.count(3)?;
                let mut experts = Vec::with_capacity(n);
                for _ in 0..n {
                    let e = c.u32()?;
                    if e > u16::MAX as u32 {
                        return Err(format!("router_topk: expert index {e} が u16 を超える"));
                    }
                    experts.push(e as u16);
                }
                let mut weights = Vec::with_capacity(n);
                for _ in 0..n {
                    // 途中で切れたペイロードで panic しない (境界は Cursor が見る)。
                    let w = c
                        .u16le()
                        .map_err(|e| format!("router_topk: weights が途中で切れている ({e})"))?;
                    weights.push(w);
                }
                Frame::RouterTopK { layer, experts, weights }
            }
            kind::EXPERT_SKIP => {
                let layer = c.u32()? as u16;
                let n = c.count(1)?;
                let mut experts = Vec::with_capacity(n);
                for _ in 0..n {
                    let e = c.u32()?;
                    if e > u16::MAX as u32 {
                        return Err(format!("expert_skip: expert index {e} が u16 を超える"));
                    }
                    experts.push(e as u16);
                }
                Frame::ExpertSkip { layer, experts }
            }
            other => return Err(format!("unknown kind: {other}")),
        };
        if c.pos() != c.len() {
            return Err(format!("trailing bytes: {} / {}", c.pos(), c.len()));
        }
        Ok(f)
    }
}

// ---------- NDJSON トレース (task 954) ----------
//
// トレースを 1 行 1 フレームの JSON (NDJSON) で書き出し / 読み戻す。依存は
// std のみ。書き出しは `Frame` の全フィールドを過不足なく含み、読み戻しは
// 壊れた行・切り詰め行・不正な JSON を panic せずに行番号付きで報告する。
//
// 形式 (1 行 = 1 オブジェクト、キー順は固定で決定的):
//
// ```text
// {"kind":"token","session":1,"token":42,"pos":7}
// {"kind":"layer_run","layer":3,"ops":[[4,1200],[5,900]]}
// ```
//
// 数値は 10 進の整数で書く (u64 の ns も f64 を経由しないため精度が落ちない)。

/// NDJSON の壊れた行。行番号は 1 始まり。読み飛ばしても読者は止まらない。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NdjsonError {
    /// 1 始まりの行番号
    pub line: u64,
    /// 何が壊れていたか
    pub message: String,
}

impl std::fmt::Display for NdjsonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for NdjsonError {}

/// kind バイト → NDJSON の `kind` 文字列。未知の kind は `None`。
pub fn ndjson_kind_name(k: u8) -> Option<&'static str> {
    Some(match k {
        kind::SERVER_INFO => "server_info",
        kind::SESSION_ADMIT => "session_admit",
        kind::SESSION_DONE => "session_done",
        kind::PREFILL => "prefill",
        kind::TOKEN => "token",
        kind::STEP => "step",
        kind::OP_SUMMARY => "op_summary",
        kind::KV_USAGE => "kv_usage",
        kind::GAP => "gap",
        kind::TOP_K => "top_k",
        kind::LAYER_RUN => "layer_run",
        kind::LAYER_ACT => "layer_act",
        kind::ROLE_ACT => "role_act",
        kind::EXPERT_RESIDENCY => "expert_residency",
        kind::TIER_STATS => "tier_stats",
        kind::ROUTER_TOPK => "router_topk",
        kind::EXPERT_SKIP => "expert_skip",
        _ => return None,
    })
}

/// NDJSON の `kind` 文字列 → kind バイト。
pub fn ndjson_kind_from_name(name: &str) -> Option<u8> {
    Some(match name {
        "server_info" => kind::SERVER_INFO,
        "session_admit" => kind::SESSION_ADMIT,
        "session_done" => kind::SESSION_DONE,
        "prefill" => kind::PREFILL,
        "token" => kind::TOKEN,
        "step" => kind::STEP,
        "op_summary" => kind::OP_SUMMARY,
        "kv_usage" => kind::KV_USAGE,
        "gap" => kind::GAP,
        "top_k" => kind::TOP_K,
        "layer_run" => kind::LAYER_RUN,
        "layer_act" => kind::LAYER_ACT,
        "role_act" => kind::ROLE_ACT,
        "expert_residency" => kind::EXPERT_RESIDENCY,
        "tier_stats" => kind::TIER_STATS,
        "router_topk" => kind::ROUTER_TOPK,
        "expert_skip" => kind::EXPERT_SKIP,
        _ => return None,
    })
}

/// 文字列を JSON 文字列リテラルとして `out` に足す (引用符込み)。
fn push_json_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// フレーム 1 個を 1 行の JSON (改行なし) にする。
pub fn frame_to_ndjson(frame: &Frame) -> String {
    let mut s = String::with_capacity(64);
    s.push('{');
    s.push_str("\"kind\":");
    push_json_str(&mut s, ndjson_kind_name(frame.kind()).unwrap_or("unknown"));
    match frame {
        Frame::ServerInfo { model, model_id, quant, slots, context, layers } => {
            s.push_str(",\"model\":"); push_json_str(&mut s, model);
            s.push_str(",\"model_id\":"); push_json_str(&mut s, model_id);
            s.push_str(",\"quant\":"); push_json_str(&mut s, quant);
            s.push_str(",\"slots\":"); s.push_str(&slots.to_string());
            s.push_str(",\"context\":"); s.push_str(&context.to_string());
            s.push_str(",\"layers\":"); s.push_str(&layers.to_string());
        }
        Frame::SessionAdmit { session, slot, prompt_tokens } => {
            s.push_str(",\"session\":"); s.push_str(&session.to_string());
            s.push_str(",\"slot\":"); s.push_str(&slot.to_string());
            s.push_str(",\"prompt_tokens\":"); s.push_str(&prompt_tokens.to_string());
        }
        Frame::SessionDone { session, reason, output_tokens } => {
            s.push_str(",\"session\":"); s.push_str(&session.to_string());
            s.push_str(",\"reason\":"); s.push_str(&reason.to_string());
            s.push_str(",\"output_tokens\":"); s.push_str(&output_tokens.to_string());
        }
        Frame::Prefill { session, prompt_tokens, ns, tokens } => {
            s.push_str(",\"session\":"); s.push_str(&session.to_string());
            s.push_str(",\"prompt_tokens\":"); s.push_str(&prompt_tokens.to_string());
            s.push_str(",\"ns\":"); s.push_str(&ns.to_string());
            s.push_str(",\"tokens\":[");
            for (i, t) in tokens.iter().enumerate() {
                if i > 0 { s.push(','); }
                s.push_str(&t.to_string());
            }
            s.push(']');
        }
        Frame::Token { session, token, pos } => {
            s.push_str(",\"session\":"); s.push_str(&session.to_string());
            s.push_str(",\"token\":"); s.push_str(&token.to_string());
            s.push_str(",\"pos\":"); s.push_str(&pos.to_string());
        }
        Frame::Step { width, ns } => {
            s.push_str(",\"width\":"); s.push_str(&width.to_string());
            s.push_str(",\"ns\":"); s.push_str(&ns.to_string());
        }
        Frame::OpSummary { ops } => {
            s.push_str(",\"ops\":[");
            for (i, (op, ns)) in ops.iter().enumerate() {
                if i > 0 { s.push(','); }
                s.push('[');
                s.push_str(&op.to_string());
                s.push(',');
                s.push_str(&ns.to_string());
                s.push(']');
            }
            s.push(']');
        }
        Frame::KvUsage { busy, t_lens } => {
            s.push_str(",\"busy\":[");
            for (i, b) in busy.iter().enumerate() {
                if i > 0 { s.push(','); }
                s.push_str(if *b { "true" } else { "false" });
            }
            s.push_str("],\"t_lens\":[");
            for (i, t) in t_lens.iter().enumerate() {
                if i > 0 { s.push(','); }
                s.push_str(&t.to_string());
            }
            s.push(']');
        }
        Frame::Gap { lost } => {
            s.push_str(",\"lost\":"); s.push_str(&lost.to_string());
        }
        Frame::TopK { session, chosen, entropy_f16, entries } => {
            s.push_str(",\"session\":"); s.push_str(&session.to_string());
            s.push_str(",\"chosen\":"); s.push_str(&chosen.to_string());
            s.push_str(",\"entropy_f16\":"); s.push_str(&entropy_f16.to_string());
            s.push_str(",\"entries\":[");
            for (i, (id, p)) in entries.iter().enumerate() {
                if i > 0 { s.push(','); }
                s.push('[');
                s.push_str(&id.to_string());
                s.push(',');
                s.push_str(&p.to_string());
                s.push(']');
            }
            s.push(']');
        }
        Frame::LayerRun { layer, ops } => {
            s.push_str(",\"layer\":"); s.push_str(&layer.to_string());
            s.push_str(",\"ops\":[");
            for (i, (op, ns)) in ops.iter().enumerate() {
                if i > 0 { s.push(','); }
                s.push('[');
                s.push_str(&op.to_string());
                s.push(',');
                s.push_str(&ns.to_string());
                s.push(']');
            }
            s.push(']');
        }
        Frame::LayerAct { layer, idx, val } => {
            s.push_str(",\"layer\":"); s.push_str(&layer.to_string());
            s.push_str(",\"idx\":[");
            for (i, x) in idx.iter().enumerate() {
                if i > 0 { s.push(','); }
                s.push_str(&x.to_string());
            }
            s.push_str("],\"val\":[");
            for (i, x) in val.iter().enumerate() {
                if i > 0 { s.push(','); }
                s.push_str(&x.to_string());
            }
            s.push(']');
        }
        Frame::RoleAct { layer, attn, gdn, router, shex, ple } => {
            s.push_str(",\"layer\":"); s.push_str(&layer.to_string());
            s.push_str(",\"attn\":"); s.push_str(&attn.to_string());
            s.push_str(",\"gdn\":"); s.push_str(&gdn.to_string());
            s.push_str(",\"router\":"); s.push_str(&router.to_string());
            s.push_str(",\"shex\":"); s.push_str(&shex.to_string());
            s.push_str(",\"ple\":"); s.push_str(&ple.to_string());
        }
        Frame::ExpertResidency { layer, in_vram } => {
            s.push_str(",\"layer\":"); s.push_str(&layer.to_string());
            s.push_str(",\"in_vram\":[");
            for (i, e) in in_vram.iter().enumerate() {
                if i > 0 { s.push(','); }
                s.push_str(&e.to_string());
            }
            s.push(']');
        }
        Frame::TierStats { vram_used_mb, vram_total_mb, ram_experts, ssd_reads } => {
            s.push_str(",\"vram_used_mb\":"); s.push_str(&vram_used_mb.to_string());
            s.push_str(",\"vram_total_mb\":"); s.push_str(&vram_total_mb.to_string());
            s.push_str(",\"ram_experts\":"); s.push_str(&ram_experts.to_string());
            s.push_str(",\"ssd_reads\":"); s.push_str(&ssd_reads.to_string());
        }
        Frame::RouterTopK { layer, experts, weights } => {
            s.push_str(",\"layer\":"); s.push_str(&layer.to_string());
            s.push_str(",\"experts\":[");
            for (i, e) in experts.iter().enumerate() {
                if i > 0 { s.push(','); }
                s.push_str(&e.to_string());
            }
            s.push_str("],\"weights\":[");
            for (i, w) in weights.iter().enumerate() {
                if i > 0 { s.push(','); }
                s.push_str(&w.to_string());
            }
            s.push(']');
        }
        Frame::ExpertSkip { layer, experts } => {
            s.push_str(",\"layer\":"); s.push_str(&layer.to_string());
            s.push_str(",\"experts\":[");
            for (i, e) in experts.iter().enumerate() {
                if i > 0 { s.push(','); }
                s.push_str(&e.to_string());
            }
            s.push(']');
        }
    }
    s.push('}');
    s
}

/// フレーム 1 個を NDJSON の 1 行 (末尾 `\n` 込み) として書き出す。
pub fn write_frame_ndjson<W: Write>(w: &mut W, frame: &Frame) -> std::io::Result<()> {
    let mut line = frame_to_ndjson(frame);
    line.push('\n');
    w.write_all(line.as_bytes())
}

/// フレーム列を NDJSON として書き出す。書いた行数を返す。
pub fn write_frames_ndjson<W: Write>(w: &mut W, frames: &[Frame]) -> std::io::Result<usize> {
    for f in frames {
        write_frame_ndjson(w, f)?;
    }
    Ok(frames.len())
}

// ----- 最小 JSON パーサ (NDJSON 読み戻し専用。外部依存なし) -----

#[derive(Debug, Clone, PartialEq)]
enum Json {
    Null,
    Bool(bool),
    /// 数値は原文のまま保持する (u64 を f64 経由で丸めないため)。
    Num(String),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

struct JsonParser<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> JsonParser<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, i: 0 }
    }

    fn skip_ws(&mut self) {
        while let Some(c) = self.b.get(self.i).copied() {
            if c == b' ' || c == b'\t' || c == b'\r' || c == b'\n' {
                self.i += 1;
            } else {
                break;
            }
        }
    }

    fn expect(&mut self, want: u8) -> Result<(), String> {
        match self.b.get(self.i).copied() {
            Some(c) if c == want => {
                self.i += 1;
                Ok(())
            }
            Some(c) => Err(format!("expected {:?} at {}, got {:?}", want as char, self.i, c as char)),
            None => Err(format!("unexpected end of input (expected {:?})", want as char)),
        }
    }

    fn literal(&mut self, word: &str) -> Result<(), String> {
        let bytes = word.as_bytes();
        let end = self.i + bytes.len();
        if end > self.b.len() || &self.b[self.i..end] != bytes {
            return Err(format!("invalid literal at {}: expected {word}", self.i));
        }
        self.i = end;
        Ok(())
    }

    fn parse_value(&mut self, depth: usize) -> Result<Json, String> {
        if depth > 64 {
            return Err("nesting too deep".into());
        }
        self.skip_ws();
        let c = match self.b.get(self.i).copied() {
            Some(c) => c,
            None => return Err("unexpected end of input".into()),
        };
        match c {
            b'{' => {
                self.i += 1;
                self.parse_obj(depth)
            }
            b'[' => {
                self.i += 1;
                self.parse_arr(depth)
            }
            b'"' => Ok(Json::Str(self.parse_str()?)),
            b't' => {
                self.literal("true")?;
                Ok(Json::Bool(true))
            }
            b'f' => {
                self.literal("false")?;
                Ok(Json::Bool(false))
            }
            b'n' => {
                self.literal("null")?;
                Ok(Json::Null)
            }
            b'-' | b'0'..=b'9' => self.parse_num(),
            other => Err(format!("unexpected byte {other:#04x} at {}", self.i)),
        }
    }

    fn parse_obj(&mut self, depth: usize) -> Result<Json, String> {
        let mut fields = Vec::new();
        self.skip_ws();
        if self.b.get(self.i).copied() == Some(b'}') {
            self.i += 1;
            return Ok(Json::Obj(fields));
        }
        loop {
            self.skip_ws();
            let key = self.parse_str()?;
            self.skip_ws();
            self.expect(b':')?;
            let v = self.parse_value(depth + 1)?;
            fields.push((key, v));
            self.skip_ws();
            match self.b.get(self.i).copied() {
                Some(b',') => {
                    self.i += 1;
                }
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Json::Obj(fields));
                }
                Some(c) => return Err(format!("expected ',' or '}}' at {}, got {:?}", self.i, c as char)),
                None => return Err("unexpected end of input in object".into()),
            }
        }
    }

    fn parse_arr(&mut self, depth: usize) -> Result<Json, String> {
        let mut items = Vec::new();
        self.skip_ws();
        if self.b.get(self.i).copied() == Some(b']') {
            self.i += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            let v = self.parse_value(depth + 1)?;
            items.push(v);
            self.skip_ws();
            match self.b.get(self.i).copied() {
                Some(b',') => {
                    self.i += 1;
                }
                Some(b']') => {
                    self.i += 1;
                    return Ok(Json::Arr(items));
                }
                Some(c) => return Err(format!("expected ',' or ']' at {}, got {:?}", self.i, c as char)),
                None => return Err("unexpected end of input in array".into()),
            }
        }
    }

    fn parse_str(&mut self) -> Result<String, String> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            let c = match self.b.get(self.i).copied() {
                Some(c) => c,
                None => return Err("unterminated string".into()),
            };
            self.i += 1;
            match c {
                b'"' => return Ok(out),
                b'\\' => {
                    let e = match self.b.get(self.i).copied() {
                        Some(e) => e,
                        None => return Err("unterminated escape".into()),
                    };
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{08}'),
                        b'f' => out.push('\u{0c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.parse_hex4()?;
                            let ch = if (0xd800..0xdc00).contains(&hi) {
                                // サロゲートペア
                                if self.b.get(self.i).copied() != Some(b'\\')
                                    || self.b.get(self.i + 1).copied() != Some(b'u')
                                {
                                    return Err("lone high surrogate".into());
                                }
                                self.i += 2;
                                let lo = self.parse_hex4()?;
                                if !(0xdc00..0xe000).contains(&lo) {
                                    return Err("invalid low surrogate".into());
                                }
                                let cp = 0x10000 + (((hi - 0xd800) as u32) << 10) + (lo - 0xdc00) as u32;
                                char::from_u32(cp).ok_or("invalid code point")?
                            } else {
                                char::from_u32(hi as u32).ok_or("invalid code point")?
                            };
                            out.push(ch);
                        }
                        other => return Err(format!("invalid escape \\{}", other as char)),
                    }
                }
                c if c < 0x20 => return Err(format!("raw control byte {c:#04x} in string")),
                c if c < 0x80 => out.push(c as char),
                c => {
                    // UTF-8 のマルチバイト列を 1 文字ぶんまとめて読む。
                    let len = if c >= 0xf0 {
                        4
                    } else if c >= 0xe0 {
                        3
                    } else if c >= 0xc0 {
                        2
                    } else {
                        return Err(format!("invalid utf-8 lead byte {c:#04x} in string"));
                    };
                    let start = self.i - 1;
                    let end = start + len;
                    if end > self.b.len() {
                        return Err("truncated utf-8 sequence in string".into());
                    }
                    let s = std::str::from_utf8(&self.b[start..end])
                        .map_err(|_| "invalid utf-8 sequence in string".to_string())?;
                    out.push_str(s);
                    self.i = end;
                }
            }
        }
    }

    fn parse_hex4(&mut self) -> Result<u16, String> {
        let mut v: u16 = 0;
        for _ in 0..4 {
            let c = match self.b.get(self.i).copied() {
                Some(c) => c,
                None => return Err("truncated \\u escape".into()),
            };
            let d = match c {
                b'0'..=b'9' => c - b'0',
                b'a'..=b'f' => c - b'a' + 10,
                b'A'..=b'F' => c - b'A' + 10,
                _ => return Err("invalid hex digit in \\u escape".into()),
            };
            v = (v << 4) | d as u16;
            self.i += 1;
        }
        Ok(v)
    }

    fn parse_num(&mut self) -> Result<Json, String> {
        let start = self.i;
        while let Some(c) = self.b.get(self.i).copied() {
            match c {
                b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E' => self.i += 1,
                _ => break,
            }
        }
        let raw = std::str::from_utf8(&self.b[start..self.i])
            .map_err(|_| "invalid number bytes".to_string())?;
        raw.parse::<f64>().map_err(|_| format!("invalid number: {raw}"))?;
        Ok(Json::Num(raw.to_string()))
    }
}

fn field<'a>(obj: &'a [(String, Json)], key: &str) -> Result<&'a Json, String> {
    obj.iter()
        .rev()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v)
        .ok_or_else(|| format!("missing field: {key}"))
}

fn as_str(j: &Json) -> Result<&str, String> {
    match j {
        Json::Str(s) => Ok(s),
        _ => Err("expected string".into()),
    }
}

fn as_u64(j: &Json) -> Result<u64, String> {
    match j {
        Json::Num(s) => s.parse::<u64>().map_err(|_| format!("not a non-negative integer: {s}")),
        _ => Err("expected number".into()),
    }
}

fn as_u32(j: &Json) -> Result<u32, String> {
    let v = as_u64(j)?;
    if v > u32::MAX as u64 {
        return Err(format!("{v} overflows u32"));
    }
    Ok(v as u32)
}

fn as_u16(j: &Json) -> Result<u16, String> {
    let v = as_u64(j)?;
    if v > u16::MAX as u64 {
        return Err(format!("{v} overflows u16"));
    }
    Ok(v as u16)
}

fn as_u8(j: &Json) -> Result<u8, String> {
    let v = as_u64(j)?;
    if v > u8::MAX as u64 {
        return Err(format!("{v} overflows u8"));
    }
    Ok(v as u8)
}

fn as_bool(j: &Json) -> Result<bool, String> {
    match j {
        Json::Bool(b) => Ok(*b),
        _ => Err("expected bool".into()),
    }
}

fn as_arr(j: &Json) -> Result<&[Json], String> {
    match j {
        Json::Arr(a) => Ok(a),
        _ => Err("expected array".into()),
    }
}

fn u32_list(j: &Json) -> Result<Vec<u32>, String> {
    as_arr(j)?.iter().map(as_u32).collect()
}

fn u16_list(j: &Json) -> Result<Vec<u16>, String> {
    as_arr(j)?.iter().map(as_u16).collect()
}

fn op_ns_list(j: &Json) -> Result<Vec<(u8, u64)>, String> {
    let mut out = Vec::new();
    for item in as_arr(j)? {
        let pair = as_arr(item)?;
        if pair.len() != 2 {
            return Err(format!("op pair must have 2 elements, got {}", pair.len()));
        }
        out.push((as_u8(&pair[0])?, as_u64(&pair[1])?));
    }
    Ok(out)
}

/// NDJSON の 1 行を [`Frame`] にする。JSON として不正・フィールド欠落・
/// 型違い・未知の kind・末尾の余分なバイトはすべて `Err` (panic しない)。
pub fn parse_frame_line(line: &str) -> Result<Frame, String> {
    let mut p = JsonParser::new(line.as_bytes());
    let v = p.parse_value(0)?;
    p.skip_ws();
    if p.i != p.b.len() {
        return Err(format!("trailing bytes after JSON: {} / {}", p.i, p.b.len()));
    }
    let obj = match &v {
        Json::Obj(o) => o,
        _ => return Err("top-level value is not a JSON object".into()),
    };
    let kind_name = as_str(field(obj, "kind")?)?;
    let k = ndjson_kind_from_name(kind_name)
        .ok_or_else(|| format!("unknown kind: {kind_name}"))?;
    let f = match k {
        kind::SERVER_INFO => Frame::ServerInfo {
            model: as_str(field(obj, "model")?)?.to_string(),
            model_id: as_str(field(obj, "model_id")?)?.to_string(),
            quant: as_str(field(obj, "quant")?)?.to_string(),
            slots: as_u32(field(obj, "slots")?)?,
            context: as_u32(field(obj, "context")?)?,
            layers: as_u32(field(obj, "layers")?)?,
        },
        kind::SESSION_ADMIT => Frame::SessionAdmit {
            session: as_u32(field(obj, "session")?)?,
            slot: as_u32(field(obj, "slot")?)?,
            prompt_tokens: as_u32(field(obj, "prompt_tokens")?)?,
        },
        kind::SESSION_DONE => Frame::SessionDone {
            session: as_u32(field(obj, "session")?)?,
            reason: as_u8(field(obj, "reason")?)?,
            output_tokens: as_u32(field(obj, "output_tokens")?)?,
        },
        kind::PREFILL => Frame::Prefill {
            session: as_u32(field(obj, "session")?)?,
            prompt_tokens: as_u32(field(obj, "prompt_tokens")?)?,
            ns: as_u64(field(obj, "ns")?)?,
            tokens: u32_list(field(obj, "tokens")?)?,
        },
        kind::TOKEN => Frame::Token {
            session: as_u32(field(obj, "session")?)?,
            token: as_u32(field(obj, "token")?)?,
            pos: as_u32(field(obj, "pos")?)?,
        },
        kind::STEP => Frame::Step {
            width: as_u32(field(obj, "width")?)?,
            ns: as_u64(field(obj, "ns")?)?,
        },
        kind::OP_SUMMARY => Frame::OpSummary {
            ops: op_ns_list(field(obj, "ops")?)?,
        },
        kind::KV_USAGE => {
            let busy = as_arr(field(obj, "busy")?)?
                .iter()
                .map(as_bool)
                .collect::<Result<Vec<_>, _>>()?;
            let t_lens = u32_list(field(obj, "t_lens")?)?;
            Frame::KvUsage { busy, t_lens }
        }
        kind::GAP => Frame::Gap {
            lost: as_u32(field(obj, "lost")?)?,
        },
        kind::TOP_K => {
            let mut entries = Vec::new();
            for item in as_arr(field(obj, "entries")?)? {
                let pair = as_arr(item)?;
                if pair.len() != 2 {
                    return Err(format!("top_k entry must have 2 elements, got {}", pair.len()));
                }
                entries.push((as_u32(&pair[0])?, as_u16(&pair[1])?));
            }
            Frame::TopK {
                session: as_u32(field(obj, "session")?)?,
                chosen: as_u32(field(obj, "chosen")?)?,
                entropy_f16: as_u16(field(obj, "entropy_f16")?)?,
                entries,
            }
        }
        kind::LAYER_RUN => Frame::LayerRun {
            layer: as_u16(field(obj, "layer")?)?,
            ops: op_ns_list(field(obj, "ops")?)?,
        },
        kind::LAYER_ACT => Frame::LayerAct {
            layer: as_u16(field(obj, "layer")?)?,
            idx: u32_list(field(obj, "idx")?)?,
            val: u16_list(field(obj, "val")?)?,
        },
        kind::ROLE_ACT => Frame::RoleAct {
            layer: as_u16(field(obj, "layer")?)?,
            attn: as_u16(field(obj, "attn")?)?,
            gdn: as_u16(field(obj, "gdn")?)?,
            router: as_u16(field(obj, "router")?)?,
            shex: as_u16(field(obj, "shex")?)?,
            ple: as_u16(field(obj, "ple")?)?,
        },
        kind::EXPERT_RESIDENCY => Frame::ExpertResidency {
            layer: as_u16(field(obj, "layer")?)?,
            in_vram: u16_list(field(obj, "in_vram")?)?,
        },
        kind::TIER_STATS => Frame::TierStats {
            vram_used_mb: as_u32(field(obj, "vram_used_mb")?)?,
            vram_total_mb: as_u32(field(obj, "vram_total_mb")?)?,
            ram_experts: as_u32(field(obj, "ram_experts")?)?,
            ssd_reads: as_u32(field(obj, "ssd_reads")?)?,
        },
        kind::ROUTER_TOPK => Frame::RouterTopK {
            layer: as_u16(field(obj, "layer")?)?,
            experts: u16_list(field(obj, "experts")?)?,
            weights: u16_list(field(obj, "weights")?)?,
        },
        kind::EXPERT_SKIP => Frame::ExpertSkip {
            layer: as_u16(field(obj, "layer")?)?,
            experts: u16_list(field(obj, "experts")?)?,
        },
        other => return Err(format!("unknown kind byte: {other}")),
    };
    Ok(f)
}

/// NDJSON を 1 行ずつ読むイテレータ。壊れた行は `Err` として 1 個返し、
/// そのまま次の行へ進む (停止しない・panic しない)。空行は読み飛ばす。
pub struct NdjsonFrames<R: BufRead> {
    reader: R,
    buf: Vec<u8>,
    line: u64,
    done: bool,
}

impl<R: BufRead> NdjsonFrames<R> {
    pub fn new(reader: R) -> Self {
        Self { reader, buf: Vec::new(), line: 0, done: false }
    }

    /// ここまでに読んだ行数 (壊れた行・空行を含む)。
    pub fn line_no(&self) -> u64 {
        self.line
    }

    /// 残りを読み切る。`(有効なフレーム, 壊れた行)` を返す。
    pub fn collect_frames(mut self) -> (Vec<Frame>, Vec<NdjsonError>) {
        let mut frames = Vec::new();
        let mut errors = Vec::new();
        for item in &mut self {
            match item {
                Ok(f) => frames.push(f),
                Err(e) => errors.push(e),
            }
        }
        (frames, errors)
    }
}

impl<R: BufRead> Iterator for NdjsonFrames<R> {
    type Item = Result<Frame, NdjsonError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        loop {
            self.buf.clear();
            let n = match self.reader.read_until(b'\n', &mut self.buf) {
                Ok(n) => n,
                Err(e) => {
                    self.done = true;
                    self.line += 1;
                    return Some(Err(NdjsonError {
                        line: self.line,
                        message: format!("io error: {e}"),
                    }));
                }
            };
            if n == 0 {
                self.done = true;
                return None;
            }
            self.line += 1;
            let line_no = self.line;
            while matches!(self.buf.last(), Some(b'\n') | Some(b'\r')) {
                self.buf.pop();
            }
            if self.buf.iter().all(|b| b.is_ascii_whitespace()) {
                continue;
            }
            let text = match std::str::from_utf8(&self.buf) {
                Ok(s) => s,
                Err(e) => {
                    return Some(Err(NdjsonError {
                        line: line_no,
                        message: format!("invalid utf-8: {e}"),
                    }))
                }
            };
            return Some(
                parse_frame_line(text).map_err(|message| NdjsonError { line: line_no, message }),
            );
        }
    }
}

/// `impl BufRead` から NDJSON を読む。壊れた行はスキップし、行番号付きで返す。
pub fn read_frames_ndjson<R: BufRead>(reader: R) -> (Vec<Frame>, Vec<NdjsonError>) {
    NdjsonFrames::new(reader).collect_frames()
}

// ---------- 送出ハブ (エンジン側) ----------

static ON: AtomicI8 = AtomicI8::new(-1); // -1 = 未初期化, 0 = off, 1 = on
static TX: OnceLock<SyncSender<Frame>> = OnceLock::new();
static RX: OnceLock<Mutex<Receiver<Frame>>> = OnceLock::new();
static DROPPED: AtomicU64 = AtomicU64::new(0);
static SESSION_SEQ: AtomicU32 = AtomicU32::new(1);
static OP_NS: [AtomicU64; N_OPS] = {
    #[allow(clippy::declare_interior_mutable_const)]
    const Z: AtomicU64 = AtomicU64::new(0);
    [Z; N_OPS]
};

/// チャネル容量 (フレーム数)。1 フレームは数十〜数百バイトなので
/// 4096 で最大 ~1MB 程度。溢れた分はドロップして Gap で報告する。
const CHANNEL_CAP: usize = 4096;

pub fn enabled() -> bool {
    match ON.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let v = std::env::var("DENNO_TELEMETRY").map(|v| v != "0").unwrap_or(false);
            ON.store(if v { 1 } else { 0 }, Ordering::Relaxed);
            v
        }
    }
}

/// エンジン側の送出ハブ (`emit`) から 1 枚取り出す。`timeout` だけ待つ。
///
/// serve は自前の [`crate::Frame`] 送信路 (WS) を持つので、エンジンが
/// `emit()` で出すフレーム (RouterTopK / RoleAct など) はここで拾って
/// WS 側へ中継する。無効時・ハブ未初期化なら `None` を返す。
pub fn recv_global_timeout(timeout: std::time::Duration) -> Option<Frame> {
    if !enabled() {
        // 呼び出し側はループで回すので、無効時は待ってから返す (空回りさせない)。
        std::thread::sleep(timeout);
        return None;
    }
    let _tx = hub();
    let rx = match RX.get() {
        Some(r) => r,
        None => return None,
    };
    match rx.lock() {
        Ok(g) => g.recv_timeout(timeout).ok(),
        Err(_) => None,
    }
}

/// 送出可否を明示的に決める。
///
/// serve は `DENNO_TELEMETRY` を「未設定なら有効」と解釈するが、[`enabled`] は
/// 「未設定なら無効」なので、解釈結果をここで 1 回だけ流し込む
/// (エンジン側の `emit()` も同じ判定になる)。
pub fn set_enabled(v: bool) {
    ON.store(if v { 1 } else { 0 }, Ordering::Relaxed);
}

/// テスト専用: env 由来の ON/OFF を上書きする (本番コードからは使わない)。
#[doc(hidden)]
pub fn set_enabled_for_tests(v: bool) {
    ON.store(if v { 1 } else { 0 }, Ordering::Relaxed);
}

/// セッション ID の採番 (begin_chat から呼ばれる。テレメトリ OFF でも
/// 採番自体は行うが、実質コストは原子加算 1 回)。
pub fn next_session_id() -> u32 {
    SESSION_SEQ.fetch_add(1, Ordering::Relaxed)
}

// task 155: 送出先 (WS) の購読者数。ビューワが 1 つも繋がっていないときは、
// フレームを作る側 (層の役割 RMS / top-8) が組み立てごと飛ばせる。
// 値は serve 側が接続/切断のたびに更新する。
static CLIENTS: AtomicUsize = AtomicUsize::new(0);

/// task 155: テレメトリの購読者 (WS クライアント) 数を記録する。
pub fn set_clients(n: usize) {
    CLIENTS.store(n, Ordering::Relaxed);
}

/// task 155: いまの購読者数。
pub fn clients() -> usize {
    CLIENTS.load(Ordering::Relaxed)
}

/// task 155: `DENNO_TELEM_LAZY` の値解釈 (純関数・テスト用)。
///
/// **既定 on** = 購読者 (WS) が居ないときは top-8 / 層活性 / arch フレームを
/// 組み立てない。`0` / `off` / `false` で従来どおり常に組み立てる (A/B 用)。
pub fn lazy_from_value(v: Option<&str>) -> bool {
    !matches!(v, Some("0") | Some("off") | Some("false"))
}

/// task 155: 購読者が居ないときに重い観測を省くか (`DENNO_TELEM_LAZY`)。
pub fn lazy_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| lazy_from_value(std::env::var("DENNO_TELEM_LAZY").ok().as_deref()))
}

/// task 155: 読む相手が 1 つ以上いるか。
///
/// `DENNO_TELEM_LAZY=0` のときは常に `true` (= 従来どおり作る)。
///
/// 実測 (GPU ラボ t155a、14,139、plain): 購読者 0 人で topk 1.00 → 0、
/// arch 0.46 → 0、tap 0.44 → 0、roles 0.16 → 0 ms/tok (**-2.06 ms/tok**)。
/// ビューワが繋がっている間の出力は従来と同一。
pub fn has_clients() -> bool {
    CLIENTS.load(Ordering::Relaxed) > 0 || !lazy_enabled()
}

/// テスト専用: 購読者数を一時的に変え、Drop で元に戻す (panic でも戻る)。
#[doc(hidden)]
pub struct ClientsGuard(usize);

#[doc(hidden)]
impl ClientsGuard {
    pub fn new(n: usize) -> Self {
        let old = clients();
        set_clients(n);
        Self(old)
    }
}

impl Drop for ClientsGuard {
    fn drop(&mut self) {
        set_clients(self.0);
    }
}

/// 送出ハブを 1 回だけ組む。`TX` と `RX` の初期化が競合すると、
/// 片方だけ別チャネルを掴んで永久に空になるため、初期化を直列化する。
/// `RX` を先に埋めてから `TX` を公開する。
fn hub() -> &'static std::sync::mpsc::SyncSender<Frame> {
    if let Some(tx) = TX.get() {
        return tx;
    }
    static INIT: Mutex<()> = Mutex::new(());
    let _g = INIT.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(tx) = TX.get() {
        return tx;
    }
    let (tx, rx) = std::sync::mpsc::sync_channel(CHANNEL_CAP);
    let _ = RX.set(Mutex::new(rx));
    let _ = TX.set(tx);
    TX.get().expect("hub: TX を直前に set した")
}

/// フレームを送出する。非ブロッキング。満杯/切断時はドロップして計上。
/// enabled() でない場合は即座に戻る (フレーム生成コストすら掛けないよう
/// 呼び出し側も enabled() ガードを推奨)。
pub fn emit(frame: Frame) {
    if !enabled() {
        return;
    }
    let tx = hub();
    match tx.try_send(frame) {
        Ok(()) => {}
        Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
            DROPPED.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// チャネル容量 (フレーム数)。テスト・観測用。
pub fn channel_cap() -> usize {
    CHANNEL_CAP
}

pub fn dropped() -> u64 {
    DROPPED.load(Ordering::Relaxed)
}

/// prof::time フックから op の所要 ns を累積する (OpSummary 用)。
pub fn add_op(op: usize, ns: u64) {
    if op < N_OPS {
        OP_NS[op].fetch_add(ns, Ordering::Relaxed);
    }
}

/// 累積 op 時間を取り出して 0 に戻す (ステップ末に 1 回呼ぶ)。
pub fn take_ops() -> Vec<(u8, u64)> {
    (0..N_OPS)
        .map(|i| (i as u8, OP_NS[i].swap(0, Ordering::Relaxed)))
        .filter(|&(_, ns)| ns > 0)
        .collect()
}

// ---------- 層単位の op 列記録 (P14: LayerRun フレーム用) ----------

thread_local! {
    /// 現在記録中の層 (layer id, ops)。`layer_begin` 〜 `layer_flush` 間のみ有効。
    static LAYER_BUF: RefCell<Option<(u16, Vec<(u8, u64)>)>> = const { RefCell::new(None) };
}

/// 層の forward 開始を記録する (この後 `record_layer_op` で op 列を蓄積し、
/// `layer_flush` で LayerRun フレーム 1 個として送出する)。
/// テレメトリ無効時は何もしない。
pub fn layer_begin(layer: u16) {
    if !enabled() {
        return;
    }
    LAYER_BUF.with(|b| *b.borrow_mut() = Some((layer, Vec::new())));
}

/// 層内 op の所要 ns を現在の層バッファに記録する (`add_op` と併用)。
/// バッファが無い (layer_begin 未呼び出し/無効) 場合は何もしない。
pub fn record_layer_op(op: usize, ns: u64) {
    if op >= u8::MAX as usize {
        return;
    }
    LAYER_BUF.with(|b| {
        if let Some((_, ops)) = b.borrow_mut().as_mut() {
            ops.push((op as u8, ns));
        }
    });
}

/// 層の forward 終了。蓄積した op 列を LayerRun フレームとして送出する。
pub fn layer_flush() {
    let buf = LAYER_BUF.with(|b| b.borrow_mut().take());
    if let Some((layer, ops)) = buf {
        if !ops.is_empty() {
            emit(Frame::LayerRun { layer, ops });
        }
    }
}

// ---------- 役割別活性 / expert 在庫 / tier 統計 (P54) ----------

/// `DENNO_TELEMETRY_ARCH` の解釈 (既定 off)。
///
/// 未設定・空・`0` 以外で有効。**off のときは呼び出し側が役割 RMS を取らない**
/// (層ループにフックを付けない) ので、数値経路にも計測コストにも触れない。
pub fn arch_from_value(v: Option<&str>) -> bool {
    matches!(v, Some(s) if !s.is_empty() && s != "0")
}

static ARCH_ON: AtomicI8 = AtomicI8::new(-1);

/// P54: 役割別活性などの送出が有効か (プロセスで 1 回だけ env を読む)。
pub fn arch_enabled() -> bool {
    match ARCH_ON.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let v = arch_from_value(std::env::var("DENNO_TELEMETRY_ARCH").ok().as_deref());
            ARCH_ON.store(if v { 1 } else { 0 }, Ordering::Relaxed);
            v
        }
    }
}

/// テスト専用: arch の ON/OFF を上書きする (本番コードからは使わない)。
#[doc(hidden)]
pub fn set_arch_for_tests(v: bool) {
    ARCH_ON.store(if v { 1 } else { 0 }, Ordering::Relaxed);
}

/// `DENNO_TELEMETRY_ARCH_DBG` の解釈 (既定 off、`1` / `true` / `yes` で有効)。
pub fn arch_dbg_from_value(v: Option<&str>) -> bool {
    matches!(v, Some("1") | Some("true") | Some("yes"))
}

/// P54: 1 トークンあたりの所要 ns と送出バイト数を stderr に出すか。
pub fn arch_dbg() -> bool {
    static D: OnceLock<bool> = OnceLock::new();
    *D.get_or_init(|| {
        arch_dbg_from_value(std::env::var("DENNO_TELEMETRY_ARCH_DBG").ok().as_deref())
    })
}

/// ベクトルの RMS (二乗平均平方根)。空・全部 0 は 0。
pub fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    let mut s = 0.0f32;
    for &v in x {
        s += v * v;
    }
    (s / x.len() as f32).sqrt()
}

/// 5 役割の生値 `[attn, gdn, router, shex, ple]` を**層内最大**で割る (P54)。
///
/// - 最大が 0 (全部 0 / 負) なら 5 値とも 0
/// - 戻り値は 0..1 に収まる (最大の役割がちょうど 1.0)
/// - NaN / Inf は「値なし」として 0 に落とす (0/0 や NaN をビューアに流さない)
pub fn normalize_roles(roles: [f32; 5]) -> [f32; 5] {
    let mut max = 0.0f32;
    for v in roles {
        if v.is_finite() && v > max {
            max = v;
        }
    }
    if !(max > 0.0) {
        return [0.0; 5];
    }
    let mut out = [0.0f32; 5];
    for (o, v) in out.iter_mut().zip(roles) {
        *o = if v.is_finite() {
            (v / max).clamp(0.0, 1.0)
        } else {
            0.0
        };
    }
    out
}

/// P54: 1 層ぶんの役割別活性フレームを作る (正規化は層内最大)。
///
/// `roles` は `[attn, gdn, router, shex, ple]` の生値 (RMS と top-k 確率和)。
pub fn role_act_frame(layer: u16, roles: [f32; 5]) -> Frame {
    let n = normalize_roles(roles);
    Frame::RoleAct {
        layer,
        attn: f32_to_f16_bits(n[0]),
        gdn: f32_to_f16_bits(n[1]),
        router: f32_to_f16_bits(n[2]),
        shex: f32_to_f16_bits(n[3]),
        ple: f32_to_f16_bits(n[4]),
    }
}

/// P54: expert 在庫フレームを作る (昇順・重複なし・最大 [`RESIDENCY_MAX`] 本)。
///
/// 入力の順序は問わない (並べ直してから切る)。**ソートしてから切る**ので、
/// 512 本を超える層でも「小さい index の側」が落ちない。
pub fn expert_residency_frame(layer: u16, in_vram: &[u16]) -> Frame {
    let mut v = in_vram.to_vec();
    v.sort_unstable();
    v.dedup();
    v.truncate(RESIDENCY_MAX);
    Frame::ExpertResidency { layer, in_vram: v }
}

// ---------- 層活性 (P53: 3D ビューア用) ----------

/// `DENNO_TELEMETRY_ACT` の解釈 (既定 off)。
///
/// 未設定・空・`0` 以外で有効。層の出力は**読むだけ**なので数値経路は変わらない。
/// 無効時は呼び出し側がフックを付けない (`serve-flash` は `*_tapped` を呼ばない)。
pub fn act_from_value(v: Option<&str>) -> bool {
    matches!(v, Some(s) if !s.is_empty() && s != "0")
}

static ACT_ON: AtomicI8 = AtomicI8::new(-1);

/// P53: 層活性の送出が有効か (プロセスで 1 回だけ env を読む)。
pub fn act_enabled() -> bool {
    match ACT_ON.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let v = act_from_value(std::env::var("DENNO_TELEMETRY_ACT").ok().as_deref());
            ACT_ON.store(if v { 1 } else { 0 }, Ordering::Relaxed);
            v
        }
    }
}

/// テスト専用: act の ON/OFF を上書きする (本番コードからは使わない)。
#[doc(hidden)]
pub fn set_act_for_tests(v: bool) {
    ACT_ON.store(if v { 1 } else { 0 }, Ordering::Relaxed);
}

/// 上位 K の既定値 (`DENNO_TELEMETRY_ACT_TOPK` 未設定時)。
pub const ACT_TOPK_DEFAULT: usize = 64;

/// `DENNO_TELEMETRY_ACT_TOPK` の解釈。読めない値は既定 64。
pub fn act_topk_from_value(v: Option<&str>) -> usize {
    v.and_then(|s| s.trim().parse::<usize>().ok()).unwrap_or(ACT_TOPK_DEFAULT)
}

/// P53: 上位 K (プロセスで 1 回だけ env を読む)。
pub fn act_topk() -> usize {
    static K: OnceLock<usize> = OnceLock::new();
    *K.get_or_init(|| {
        act_topk_from_value(std::env::var("DENNO_TELEMETRY_ACT_TOPK").ok().as_deref())
    })
}

/// `DENNO_TELEMETRY_ACT_DBG` の解釈 (既定 off、`1` / `true` / `yes` で有効)。
pub fn act_dbg_from_value(v: Option<&str>) -> bool {
    matches!(v, Some("1") | Some("true") | Some("yes"))
}

/// P53: 1 フレームあたりの所要 ns と送出バイト数を stderr に 1 行出すか。
pub fn act_dbg() -> bool {
    static D: OnceLock<bool> = OnceLock::new();
    *D.get_or_init(|| {
        act_dbg_from_value(std::env::var("DENNO_TELEMETRY_ACT_DBG").ok().as_deref())
    })
}

/// P53: 1 層の出力 `x` (stream 1 本ぶん、n_embd 個) から [`Frame::LayerAct`] を作る。
///
/// - `idx`: |x| の上位 `k` 個の index。**昇順**に並べて返す。
///   `k` が `x.len()` より大きければ `x.len()` 個。|x| がちょうど 0 の要素は
///   送らない (上位 K に 0 が混じっても落とす)。
/// - `val`: 対応する |x| を**層内最大**で割った値の f16 ビット。最大が 0
///   (全部 0 / 空 / k == 0) なら `idx` と `val` は空。
/// - 選択は `select_nth_unstable_by` (nth_element 相当) で、全ソートしない。
///   同値 (|x| が同じ) は index が小さい方を先に採る (入力順に依らず決定的)。
///
/// `x` は読むだけで書き換えない。`scratch` は呼び出し側が使い回す作業域
/// (層をまたいで再利用すると確保が 1 回で済む)。
pub fn layer_act_frame(layer: u16, x: &[f32], k: usize, scratch: &mut Vec<(f32, u32)>) -> Frame {
    let mut max = 0.0f32;
    scratch.clear();
    if k > 0 {
        scratch.reserve(x.len());
        for (i, &v) in x.iter().enumerate() {
            // NaN / Inf は活性として送らない (f16 にすると NaN ビットになり、
            // 正規化も Inf/Inf = NaN になる)。
            if !v.is_finite() {
                continue;
            }
            let a = v.abs();
            if a > max {
                max = a;
            }
            scratch.push((a, i as u32));
        }
    }
    if k == 0 || x.is_empty() || !(max > 0.0) {
        return Frame::LayerAct { layer, idx: Vec::new(), val: Vec::new() };
    }
    let want = k.min(scratch.len());
    scratch.select_nth_unstable_by(want - 1, |a, b| {
        b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1))
    });
    scratch.truncate(want);
    // |x| がちょうど 0 の要素は送らない (値が 0 の帯は情報を持たない。
    // 「1 個だけ非 0」の層で 63 個の 0 を積まないための規則)。
    scratch.retain(|e| e.0 > 0.0);
    scratch.sort_unstable_by_key(|e| e.1);
    let idx: Vec<u32> = scratch.iter().map(|e| e.1).collect();
    let val: Vec<u16> = scratch.iter().map(|e| f32_to_f16_bits(e.0 / max)).collect();
    Frame::LayerAct { layer, idx, val }
}

// ---------- ルーティング (P56: RouterTopK) ----------

/// P56: 1 層ぶんのルータ結果から [`Frame::RouterTopK`] を作る。
///
/// - `sel`: 選ばれた expert index。`moe::select_topk` の出力 = **index 昇順**。
/// - `weights`: `sel` と同じ並びの gate 重み (softmax → top-k 再正規化後、合計 1)。
///
/// フレームの並びは **重みの降順** (同値は index 昇順) にする。ビューアは先頭から
/// 「強く選ばれた expert」として光らせるので、index 順のままでは強さが読めない。
/// 重みは f16 ビットへ丸める (合計 1 の情報は f16 で十分)。
///
/// `sel` と `weights` の長さが違うときは短い方に合わせる (長さの食い違いで
/// panic しない)。expert index は `u16` に飽和させる (実モデルは 512 本)。
pub fn router_topk_frame(layer: u16, sel: &[usize], weights: &[f32]) -> Frame {
    let n = sel.len().min(weights.len());
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| {
        weights[b].total_cmp(&weights[a]).then_with(|| sel[a].cmp(&sel[b]))
    });
    let experts: Vec<u16> = order
        .iter()
        .map(|&j| sel[j].min(u16::MAX as usize) as u16)
        .collect();
    let w: Vec<u16> = order.iter().map(|&j| f32_to_f16_bits(weights[j])).collect();
    Frame::RouterTopK { layer, experts, weights: w }
}

/// task 220: 「ルータが選びたかったが介入で無効化されていた expert」から
/// [`Frame::ExpertSkip`] を作る。`skipped` は index 昇順 (差分の作り方が昇順) を
/// 前提にし、`u16` へ飽和させる。空ならビューワに描くものが無いので `None`。
///
/// 呼び出し側 (engine) は `DENNO_TELEM_EXPERT_SKIP=1` のときだけこれを呼ぶ。
pub fn expert_skip_frame(layer: u16, skipped: &[usize]) -> Option<Frame> {
    if skipped.is_empty() {
        return None;
    }
    let experts: Vec<u16> = skipped
        .iter()
        .map(|&e| e.min(u16::MAX as usize) as u16)
        .collect();
    Some(Frame::ExpertSkip { layer, experts })
}

// ---------- 層単位のローリング統計 (task 914) ----------

/// 1 層ぶんの窓内統計。窓が空 (`count == 0`) のとき min/max/mean は 0。
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LayerWindowStats {
    /// 窓に入っているサンプル数 (≤ 窓長)
    pub count: usize,
    pub min: f32,
    pub max: f32,
    pub mean: f32,
}

#[derive(Debug, Clone)]
struct LayerSlot {
    /// リングバッファ。長さは窓長で固定。
    buf: Vec<f32>,
    /// 次に書く位置。
    next: usize,
    /// 有効なサンプル数 (≤ buf.len())。
    filled: usize,
    /// 非有限値として捨てた回数。
    rejected: u64,
}

impl LayerSlot {
    fn new(window: usize) -> Self {
        Self { buf: vec![0.0; window], next: 0, filled: 0, rejected: 0 }
    }

    fn push(&mut self, v: f32) {
        self.buf[self.next] = v;
        self.next = (self.next + 1) % self.buf.len();
        if self.filled < self.buf.len() {
            self.filled += 1;
        }
    }

    fn clear(&mut self) {
        self.next = 0;
        self.filled = 0;
    }
}

/// 層ごとの直近 `window` サンプルの count / min / max / mean を保つ。
///
/// - メモリは構築時に `n_layers * window` 個の f32 で**固定**。push しても
///   増えない (層数も窓長も後から変わらないので unbounded にならない)。
/// - NaN / ±Inf は記録せず、その層の [`RollingLayerStats::rejected`] を増やす。
/// - 範囲外の層 index は `push` が `false`、`stats` が `None`。
/// - `mean` は f64 で合計して f32 に落とす (長時間の f32 累積誤差を避ける)。
#[derive(Debug, Clone)]
pub struct RollingLayerStats {
    n_layers: usize,
    window: usize,
    slots: Vec<LayerSlot>,
}

impl RollingLayerStats {
    /// 層数 `n_layers`、窓長 `window` で作る。`window == 0` は 1 に切り上げる。
    pub fn new(n_layers: usize, window: usize) -> Self {
        let window = window.max(1);
        Self {
            n_layers,
            window,
            slots: (0..n_layers).map(|_| LayerSlot::new(window)).collect(),
        }
    }

    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    pub fn window(&self) -> usize {
        self.window
    }

    /// サンプルを 1 個足す。層が範囲外、または値が非有限なら記録せず `false`。
    pub fn push(&mut self, layer: u16, v: f32) -> bool {
        let Some(slot) = self.slots.get_mut(layer as usize) else {
            return false;
        };
        if !v.is_finite() {
            slot.rejected = slot.rejected.saturating_add(1);
            return false;
        }
        slot.push(v);
        true
    }

    /// 複数サンプルを足し、受け付けた個数を返す (非有限は捨てる)。
    pub fn push_slice(&mut self, layer: u16, values: &[f32]) -> usize {
        values.iter().filter(|&&v| self.push(layer, v)).count()
    }

    /// 窓内統計。層が範囲外なら `None`。空の窓は count 0 / min=max=mean=0。
    pub fn stats(&self, layer: u16) -> Option<LayerWindowStats> {
        let slot = self.slots.get(layer as usize)?;
        if slot.filled == 0 {
            return Some(LayerWindowStats::default());
        }
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        for &v in &slot.buf[..slot.filled] {
            min = min.min(v);
            max = max.max(v);
            sum += v as f64;
        }
        Some(LayerWindowStats {
            count: slot.filled,
            min,
            max,
            mean: (sum / slot.filled as f64) as f32,
        })
    }

    /// その層で非有限値として捨てた回数。層が範囲外なら `None`。
    pub fn rejected(&self, layer: u16) -> Option<u64> {
        self.slots.get(layer as usize).map(|s| s.rejected)
    }

    /// 全層の非有限棄却回数の合計。
    pub fn rejected_total(&self) -> u64 {
        self.slots.iter().map(|s| s.rejected).sum()
    }

    /// 1 層だけ空にする (拒否カウンタも戻す)。層が範囲外なら `false`。
    pub fn reset_layer(&mut self, layer: u16) -> bool {
        match self.slots.get_mut(layer as usize) {
            Some(slot) => {
                slot.clear();
                slot.rejected = 0;
                true
            }
            None => false,
        }
    }

    /// 全層を空にする。
    pub fn reset(&mut self) {
        for slot in &mut self.slots {
            slot.clear();
            slot.rejected = 0;
        }
    }

    /// 保持しているサンプル数の合計 (メモリ上限の確認用)。
    pub fn total_samples(&self) -> usize {
        self.slots.iter().map(|s| s.filled).sum()
    }
}

/// 消費側: 1 フレームを受信 (タイムアウト付き)。サーバーの WS 転送タスクが使う。
pub fn try_recv_timeout(d: Duration) -> Option<Frame> {
    let rx = RX.get()?;
    rx.lock().ok()?.recv_timeout(d).ok()
}

/// 消費側: 待たずに 1 フレームだけ受信 (テスト用)。
pub fn try_recv_nowait() -> Option<Frame> {
    let rx = RX.get()?;
    rx.lock().ok()?.try_recv().ok()
}

// ---------- sampler top-k (L1 統計。テレメトリ専用で数値経路に触れない) ----------

/// task 853: `DENNO_TELEMETRY_TOPK` の解釈 (**既定 off**)。
///
/// `1` / `true` / `yes` / `on` (大文字小文字を問わない) だけが有効。
/// 未設定・空・その他の値は無効 (zero overhead)。
pub fn topk_telemetry_from_value(v: Option<&str>) -> bool {
    match v {
        Some(s) => {
            let s = s.trim().to_ascii_lowercase();
            matches!(s.as_str(), "1" | "true" | "yes" | "on")
        }
        None => false,
    }
}

static TOPK_ON: AtomicI8 = AtomicI8::new(-1);

/// task 853: sampler の top-k フレーム ([`Frame::TopK`]) を送出するか。
///
/// **既定 off**。off のときは `topk_softmax` の走査 (語彙全体の 1 パス) 自体を
/// 呼び出し側が飛ばすので、推論経路のコストはフラグ 1 個の分岐だけ。
pub fn topk_telemetry_enabled() -> bool {
    match TOPK_ON.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let v = topk_telemetry_from_value(
                std::env::var("DENNO_TELEMETRY_TOPK").ok().as_deref(),
            );
            TOPK_ON.store(if v { 1 } else { 0 }, Ordering::Relaxed);
            v
        }
    }
}

/// テスト専用: top-k 送出の ON/OFF を上書きする (本番コードからは使わない)。
#[doc(hidden)]
pub fn set_topk_telemetry_for_tests(v: bool) {
    TOPK_ON.store(if v { 1 } else { 0 }, Ordering::Relaxed);
}

#[derive(Debug, Clone, PartialEq)]
pub struct TopK {
    pub chosen: u32,
    /// 全分布のエントロピー (nat)
    pub entropy: f32,
    /// 上位 k の (token id, 確率)。確率は softmax 後。
    pub entries: Vec<(u32, f32)>,
}

/// logits から top-k 確率とエントロピーを計算する。
/// テレメトリ専用 (数値経路には使わないため厳密な決定論は不要だが、
/// 実装は決定論的: 部分ソートは logit 降順・同点は id 昇順)。
pub fn topk_softmax(logits: &[f32], k: usize) -> TopK {
    if logits.is_empty() {
        return TopK { chosen: 0, entropy: 0.0, entries: Vec::new() };
    }
    let mut max = f32::NEG_INFINITY;
    for &l in logits {
        if l > max {
            max = l;
        }
    }
    // 1 パスで S = Σ exp(l - max) と T = Σ exp(l - max) * (l - max) を取る
    // (エントロピー H = ln S - T/S を closed form で得る)
    let mut s = 0f32;
    let mut t = 0f32;
    for &l in logits {
        let e = (l - max).exp();
        s += e;
        t += e * (l - max);
    }
    let entropy = (s.ln() - t / s).max(0.0);
    // 非有限 logits (NaN / ±Inf) が混じると s / t が NaN になり得る。
    // テレメトリは「読める値」だけを出すので、非有限は 0 に落とす。
    let entropy = if entropy.is_finite() { entropy } else { 0.0 };
    // top-k 選択 (logit 降順、同点は id 昇順)。
    // 全ソートは語彙全体 (約 25 万) の間接比較になり、毎トークンで数 ms かかる。
    // 上位 k 件を保つだけの 1 パス走査にすると比較関数が同じなので結果は全ソートと同一。
    let take = k.min(logits.len());
    if take == 0 {
        return TopK { chosen: 0, entropy, entries: Vec::new() };
    }
    // 「a (val, id) が b より前 = 順位が上」を返す全順序 (logit 降順・同点は id 昇順)。
    // 全ソート版の比較関数と同じ順序になる。
    let rank = |av: f32, ai: u32, bv: f32, bi: u32| -> std::cmp::Ordering {
        av.total_cmp(&bv).then(bi.cmp(&ai))
    };
    let mut best: Vec<(u32, f32)> = Vec::with_capacity(take);
    for (i, &l) in logits.iter().enumerate() {
        let id = i as u32;
        if best.len() == take {
            // 最下位より順位が下なら捨てる。
            let (wid, wl) = best[take - 1];
            if rank(l, id, wl, wid) != std::cmp::Ordering::Greater {
                continue;
            }
        }
        let pos = best
            .iter()
            .position(|&(bid, bl)| rank(l, id, bl, bid) == std::cmp::Ordering::Greater)
            .unwrap_or(best.len());
        best.insert(pos, (id, l));
        best.truncate(take);
    }
    let entries = best
        .into_iter()
        .map(|(i, l)| {
            let p = (l - max).exp() / s;
            // NaN / Inf の確率は送らない (ビューアの棒グラフが壊れる)。
            (i, if p.is_finite() { p.max(0.0) } else { 0.0 })
        })
        .collect();
    TopK { chosen: 0, entropy, entries }
}

// ---------- テスト ----------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_roundtrip() {
        for v in [0u64, 1, 127, 128, 300, u32::MAX as u64, (1u64 << 32), u64::MAX] {
            let mut b = Vec::new();
            put_varint_u64(&mut b, v);
            let (got, n) = get_varint_u64(&b).unwrap();
            assert_eq!(got, v);
            assert_eq!(n, b.len());
        }
    }

    #[test]
    fn f16_roundtrip_normal_values() {
        // 既知の期待値 (half クレート相当の RNE)
        for (x, h) in [
            (0.0f32, 0x0000u16),
            (-0.0, 0x8000),
            (1.0, 0x3c00),
            (-1.0, 0xbc00),
            (0.5, 0x3800),
            (2.0, 0x4000),
            (2.75, 0x4180),
            (3.14159, 0x4248),
            (65504.0, 0x7bff),      // f16 最大正規
            (65520.0, 0x7c00),      // RNE で Inf に丸まる
            (6.1035156e-5, 0x0400), // 最小正規 2^-14
            (5.9604645e-8, 0x0001), // 最小非正規 2^-24
            (2f32.powi(-25), 0x0000), // RNE (偶数) で 0
            (1e-10, 0x0000),        // アンダーフロー
            (1e-7, 0x0002),         // 1e-7 / 2^-24 ≈ 1.677 → 2
        ] {
            let got = f32_to_f16_bits(x);
            assert_eq!(got, h, "f32_to_f16_bits({x}) = {got:#06x}, want {h:#06x}");
        }
        // 往復: 正規域では相対誤差 2^-11 以内
        for x in [0.25f32, -2.75, 3.14159, 1234.5, 6.1e-5, 100.0] {
            let back = f16_bits_to_f32(f32_to_f16_bits(x));
            let rel = ((back - x) / x).abs();
            assert!(rel < 5e-4, "{x} -> {back}");
        }
        // 非正規のデコード (既知値)
        assert_eq!(f16_bits_to_f32(0x0001), 5.9604645e-8); // 最小非正規 2^-24
        assert_eq!(f16_bits_to_f32(0x0400), 6.1035156e-5); // 最小正規 2^-14
        let back = f16_bits_to_f32(0x03ff);
        assert!((back - 1023.0 * 2f32.powi(-24)).abs() < 1e-12, "{back}");
        // Inf / NaN
        assert_eq!(f16_bits_to_f32(f32_to_f16_bits(f32::INFINITY)), f32::INFINITY);
        assert_eq!(f16_bits_to_f32(f32_to_f16_bits(f32::NEG_INFINITY)), f32::NEG_INFINITY);
        assert!(f16_bits_to_f32(f32_to_f16_bits(f32::NAN)).is_nan());
        // オーバーフロー → Inf
        assert_eq!(f16_bits_to_f32(f32_to_f16_bits(1e10)), f32::INFINITY);
        // 非正規の往復 (最小非正規 → f32 → 最小非正規)
        assert_eq!(f32_to_f16_bits(f16_bits_to_f32(0x0001)), 0x0001);
    }

    #[test]
    fn frame_roundtrip_all_kinds() {
        let frames = vec![
            Frame::ServerInfo {
                model: "Qwen/Qwen2.5-0.5B-Instruct".into(),
                model_id: "Qwen/Qwen2.5-0.5B-Instruct".into(),
                quant: "fp32".into(),
                slots: 4,
                context: 32768,
                layers: 24,
            },
            Frame::SessionAdmit { session: 7, slot: 2, prompt_tokens: 34 },
            Frame::SessionDone { session: 7, reason: done_reason::END_TURN, output_tokens: 26 },
            Frame::Prefill {
                session: 7,
                prompt_tokens: 34,
                ns: 123_456_789,
                tokens: vec![972, 198, 3585, 370, 151644],
            },
            Frame::Token { session: 7, token: 99_999, pos: 12 },
            Frame::Step { width: 4, ns: 8_765_432 },
            Frame::OpSummary { ops: vec![(4, 1_000), (11, 2_500)] },
            Frame::KvUsage { busy: vec![true, false, true, true], t_lens: vec![10, 0, 300, 777] },
            Frame::Gap { lost: 42 },
            Frame::TopK {
                session: 3,
                chosen: 831,
                entropy_f16: f32_to_f16_bits(1.25),
                entries: vec![(831, f32_to_f16_bits(0.5)), (12, f32_to_f16_bits(0.25))],
            },
            Frame::LayerRun {
                layer: 5,
                ops: vec![(1, 2100), (3, 154_000), (2, 8800), (4, 1_204_500), (5, 96_000)],
            },
            // P53: K=1 / K=64 / 空 (0 件) の往復
            Frame::LayerAct {
                layer: 47,
                idx: vec![1531],
                val: vec![0x3c00],
            },
            Frame::LayerAct {
                layer: 0,
                idx: (0..64u32).map(|i| i * 37).collect(),
                val: (0..64u16).map(|i| 0x3c00 - i).collect(),
            },
            Frame::LayerAct { layer: 12, idx: Vec::new(), val: Vec::new() },
            // P54: 役割別活性 (全部 0 / 1 本だけ 1.0)
            Frame::RoleAct { layer: 5, attn: 0, gdn: 0, router: 0, shex: 0, ple: 0 },
            Frame::RoleAct {
                layer: 47,
                attn: 0,
                gdn: 0x3c00,
                router: 0x3800,
                shex: 0x3c00,
                ple: 0x0001,
            },
            // P54: expert 在庫 (空 / 1 本 / 512 本)
            Frame::ExpertResidency { layer: 3, in_vram: Vec::new() },
            Frame::ExpertResidency { layer: 4, in_vram: vec![7] },
            Frame::ExpertResidency { layer: 5, in_vram: (0..512u16).collect() },
            // P54: tier 統計 (0 / 実形の値)
            Frame::TierStats { vram_used_mb: 0, vram_total_mb: 0, ram_experts: 0, ssd_reads: 0 },
            Frame::TierStats {
                vram_used_mb: 4_096,
                vram_total_mb: 8_192,
                ram_experts: 64,
                ssd_reads: 1_234_567,
            },
            // P56: RouterTopK (空 / 1 要素 / 10 要素)
            Frame::RouterTopK { layer: 47, experts: Vec::new(), weights: Vec::new() },
            Frame::RouterTopK {
                layer: 3,
                experts: vec![511],
                weights: vec![f32_to_f16_bits(1.0)],
            },
            Frame::RouterTopK {
                layer: 47,
                experts: (0..10u16).map(|i| 502 + i).collect(),
                weights: (0..10u16).map(|i| 0x3c00 - i * 13).collect(),
            },
            // task 220: ExpertSkip (1 本 / 実形の top-k ぶん)
            Frame::ExpertSkip { layer: 5, experts: vec![0] },
            Frame::ExpertSkip { layer: 47, experts: (0..10u16).map(|i| 300 + i).collect() },
        ];
        for f in frames {
            let enc = f.encode();
            assert_eq!(&enc[..4], &MAGIC);
            assert_eq!(enc[5], f.kind());
            let dec = Frame::decode(&enc).unwrap();
            assert_eq!(dec, f, "roundtrip failed for kind {}", f.kind());
        }
        assert_eq!(kind::LAYER_ACT, 12, "kind 番号は既存の次の値 (11 の次)");
        // P54: kind 番号は既存の続き (12 の次から 3 個)。P56 の RouterTopK は 16。
        assert_eq!(kind::ROLE_ACT, 13);
        assert_eq!(kind::EXPERT_RESIDENCY, 14);
        assert_eq!(kind::TIER_STATS, 15);
        assert_eq!(kind::ROUTER_TOPK, 16, "kind 番号は 13..15 (RoleAct 等) の次");
        assert_eq!(kind::EXPERT_SKIP, 17, "task 220 の ExpertSkip は 16 の次");
    }

    /// P54: 役割別活性の正規化 (層内最大で割る) とフレーム化。
    #[test]
    fn role_act_frame_normalizes_by_layer_max() {
        // 最大の役割が 1.0、他は比
        let f = role_act_frame(9, [2.0, 1.0, 0.5, 4.0, 0.0]);
        assert_eq!(
            f,
            Frame::RoleAct {
                layer: 9,
                attn: f32_to_f16_bits(0.5),
                gdn: f32_to_f16_bits(0.25),
                router: f32_to_f16_bits(0.125),
                shex: f32_to_f16_bits(1.0),
                ple: 0x0000,
            }
        );
        // 最大が 0 → 5 値とも 0 (0/0 を作らない)
        assert_eq!(
            role_act_frame(1, [0.0; 5]),
            Frame::RoleAct { layer: 1, attn: 0, gdn: 0, router: 0, shex: 0, ple: 0 }
        );
        // normalize_roles は 0..1 に収まる
        let n = normalize_roles([0.0, 3.0, 0.0, 0.0, 0.0]);
        assert_eq!(n, [0.0, 1.0, 0.0, 0.0, 0.0]);
        // 負の値は 0 に落とす (RMS は負にならないが、防御的に)
        assert_eq!(normalize_roles([-1.0, -2.0, 0.0, 0.0, 0.0]), [0.0; 5]);
        // rms: 空 = 0、[3,4] = sqrt(12.5)
        assert_eq!(rms(&[]), 0.0);
        assert!((rms(&[3.0, 4.0]) - 12.5f32.sqrt()).abs() < 1e-6);
        assert_eq!(rms(&[0.0; 8]), 0.0);
    }

    /// P54: expert 在庫は昇順・重複なし・最大 512 本。
    #[test]
    fn expert_residency_frame_sorts_and_caps() {
        assert_eq!(
            expert_residency_frame(2, &[9, 3, 3, 7]),
            Frame::ExpertResidency { layer: 2, in_vram: vec![3, 7, 9] }
        );
        assert_eq!(
            expert_residency_frame(0, &[]),
            Frame::ExpertResidency { layer: 0, in_vram: Vec::new() }
        );
        // 512 本を超える入力は小さい index 側を残して切る
        let many: Vec<u16> = (0..600u16).rev().collect();
        match expert_residency_frame(1, &many) {
            Frame::ExpertResidency { layer: 1, in_vram } => {
                assert_eq!(in_vram.len(), RESIDENCY_MAX);
                assert_eq!(in_vram.first(), Some(&0));
                assert_eq!(in_vram.last(), Some(&(RESIDENCY_MAX as u16 - 1)));
            }
            other => panic!("ExpertResidency でない: {other:?}"),
        }
    }

    /// P56: RouterTopK のペイロード長 (実形: 層 47 / expert 512 本 / top-10)。
    ///
    /// 6 (magic+version+kind) + 1 (layer) + 1 (n) + 10×2 (index は 128 以上 = 2B)
    /// + 10×2 (重み f16) = 48 B。decode は途中で切れた weights を Err にする
    /// (panic しない)。
    #[test]
    fn router_topk_payload_len_and_truncation() {
        let f = Frame::RouterTopK {
            layer: 47,
            experts: (0..10u16).map(|i| 502 + i).collect(),
            weights: (0..10u16).map(|i| 0x3c00 - i * 13).collect(),
        };
        let enc = f.encode();
        assert_eq!(enc.len(), 48, "実形の RouterTopK は 48 B");
        assert_eq!(Frame::decode(&enc).unwrap(), f);
        // 空 (層 47 / 0 本) は 6 + 1 + 1 = 8 B
        let e = Frame::RouterTopK { layer: 47, experts: Vec::new(), weights: Vec::new() };
        assert_eq!(e.encode().len(), 8);
        assert_eq!(Frame::decode(&e.encode()).unwrap(), e);
        // weights を 1 バイト削ると「途中で切れている」で Err (panic しない)
        let broken = &enc[..enc.len() - 1];
        match Frame::decode(broken) {
            Err(msg) => assert!(msg.contains("weights"), "{msg}"),
            Ok(f) => panic!("切り詰めたペイロードが通った: {f:?}"),
        }
        // expert index が u16 を超えるペイロードは Err
        let mut over = Vec::new();
        over.extend_from_slice(&MAGIC);
        over.push(VERSION);
        over.push(kind::ROUTER_TOPK);
        put_varint_u32(&mut over, 0); // layer
        put_varint_u32(&mut over, 1); // n
        put_varint_u32(&mut over, 70_000); // expert index
        over.extend_from_slice(&0x3c00u16.to_le_bytes());
        assert!(Frame::decode(&over).is_err());
    }

    /// P56: フレームの並びは重みの降順 (同値は index 昇順)、重みは f16 に丸める。
    #[test]
    fn router_topk_frame_sorts_by_weight_desc() {
        // sel は index 昇順で渡ってくる (select_topk の出力)。
        let f = router_topk_frame(5, &[2, 7, 9], &[0.1, 0.7, 0.2]);
        match f {
            Frame::RouterTopK { layer: 5, experts, weights } => {
                assert_eq!(experts, vec![7, 9, 2], "重みの降順でない");
                assert_eq!(
                    weights,
                    vec![
                        f32_to_f16_bits(0.7),
                        f32_to_f16_bits(0.2),
                        f32_to_f16_bits(0.1)
                    ]
                );
            }
            other => panic!("RouterTopK でない: {other:?}"),
        }
        // f16 の丸め (0.5 はちょうど 0x3800、1.0 は 0x3c00)
        match router_topk_frame(0, &[0, 1], &[0.5, 0.5]) {
            Frame::RouterTopK { experts, weights, .. } => {
                assert_eq!(experts, vec![0, 1], "同値は index 昇順");
                assert_eq!(weights, vec![0x3800, 0x3800]);
            }
            other => panic!("RouterTopK でない: {other:?}"),
        }
        // 空 / 長さの食い違い (短い方に合わせる・panic しない)
        assert_eq!(
            router_topk_frame(9, &[], &[]),
            Frame::RouterTopK { layer: 9, experts: Vec::new(), weights: Vec::new() }
        );
        match router_topk_frame(9, &[4, 5, 6], &[0.9, 0.1]) {
            Frame::RouterTopK { experts, weights, .. } => {
                assert_eq!(experts, vec![4, 5]);
                assert_eq!(weights.len(), 2);
            }
            other => panic!("RouterTopK でない: {other:?}"),
        }
    }

    /// task 220: ExpertSkip のフレーム化 (空は送らない / index は u16 に飽和)。
    #[test]
    fn expert_skip_frame_is_absent_when_empty() {
        assert_eq!(expert_skip_frame(3, &[]), None, "空の skip を送っている");
        assert_eq!(
            expert_skip_frame(3, &[0, 511]),
            Some(Frame::ExpertSkip { layer: 3, experts: vec![0, 511] })
        );
        // 範囲外の index で panic しない (u16 に飽和)。
        match expert_skip_frame(1, &[70_000]) {
            Some(Frame::ExpertSkip { experts, .. }) => assert_eq!(experts, vec![u16::MAX]),
            other => panic!("ExpertSkip でない: {other:?}"),
        }
    }

    /// P54/P56: `DENNO_TELEMETRY_ARCH` の解釈 (既定 off)。
    #[test]
    fn arch_env_parsing() {
        assert!(!arch_from_value(None), "未設定は既定 off");
        assert!(!arch_from_value(Some("")), "空は off");
        assert!(!arch_from_value(Some("0")), "0 は off");
        assert!(arch_from_value(Some("1")));
        assert!(arch_from_value(Some("true")), "0 以外は有効");
        assert!(!arch_dbg_from_value(None));
        assert!(arch_dbg_from_value(Some("1")));
        assert!(arch_dbg_from_value(Some("yes")));
        assert!(!arch_dbg_from_value(Some("0")));
        // テスト用の上書きは元に戻せる (他のテストに漏らさない)
        let before = arch_enabled();
        set_arch_for_tests(true);
        assert!(arch_enabled());
        set_arch_for_tests(false);
        assert!(!arch_enabled());
        set_arch_for_tests(before);
    }

    /// task 853: `DENNO_TELEMETRY_TOPK` の解釈 (既定 off)。
    #[test]
    fn topk_telemetry_env_parsing() {
        assert!(!topk_telemetry_from_value(None), "未設定は既定 off");
        assert!(!topk_telemetry_from_value(Some("")), "空は off");
        assert!(!topk_telemetry_from_value(Some("0")), "0 は off");
        assert!(!topk_telemetry_from_value(Some("off")), "off は off");
        assert!(!topk_telemetry_from_value(Some("garbage")), "読めない値は off");
        assert!(topk_telemetry_from_value(Some("1")));
        assert!(topk_telemetry_from_value(Some("true")));
        assert!(topk_telemetry_from_value(Some("YES")), "大文字も受ける");
        assert!(topk_telemetry_from_value(Some(" on ")), "前後の空白は無視");
        // テスト用の上書きは元に戻せる (他のテストに漏らさない)
        let before = topk_telemetry_enabled();
        set_topk_telemetry_for_tests(true);
        assert!(topk_telemetry_enabled());
        set_topk_telemetry_for_tests(false);
        assert!(!topk_telemetry_enabled());
        set_topk_telemetry_for_tests(before);
    }

    /// P53: 層活性の選択 (上位 K・昇順・層内最大で正規化) を全ソートの参照実装と
    /// 突き合わせる。値は決定的な擬似乱数 (実形 n_embd=2560)。
    #[test]
    fn layer_act_frame_matches_full_sort_reference() {
        let n = 2560usize;
        let x: Vec<f32> = (0..n)
            .map(|i| {
                let h = (i as u64).wrapping_mul(2654435761) % 2048;
                (h as f32 / 1024.0 - 1.0) * (1.0 + (i % 7) as f32)
            })
            .collect();
        let x_before = x.clone();
        let mut scratch = Vec::new();
        for k in [1usize, 64, 200, n, n + 10] {
            let f = layer_act_frame(9, &x, k, &mut scratch);
            let (idx, val) = match f {
                Frame::LayerAct { layer: 9, idx, val } => (idx, val),
                other => panic!("LayerAct でない: {other:?}"),
            };
            let want_n = k.min(n);
            // 参照: 全ソート (|x| 降順・同値は index 昇順) → k 個 → 0 を落とす → index 昇順
            let mut all: Vec<(f32, u32)> =
                x.iter().enumerate().map(|(i, &v)| (v.abs(), i as u32)).collect();
            all.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            all.truncate(want_n);
            all.retain(|e| e.0 > 0.0);
            all.sort_unstable_by_key(|e| e.1);
            assert_eq!(idx.len(), all.len(), "k={k}");
            assert_eq!(val.len(), all.len());
            assert_eq!(
                idx,
                all.iter().map(|e| e.1).collect::<Vec<_>>(),
                "k={k}: 上位 K の集合が全ソートと違う"
            );
            let max = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            for (i, &e) in all.iter().enumerate() {
                let want = f32_to_f16_bits(e.0 / max);
                assert_eq!(val[i], want, "k={k} idx={}: 正規化が違う", e.1);
            }
            // idx は昇順 (正規化して送る約束)
            assert!(idx.windows(2).all(|w| w[0] < w[1]), "k={k}: idx が昇順でない");
            // 最大の |x| はちょうど 1.0 に正規化される (idx は昇順なので位置で見る)
            let max_pos = all
                .iter()
                .position(|e| e.0 == all.iter().fold(0.0f32, |m, e| m.max(e.0)))
                .expect("最大値が上位 K に無い");
            assert_eq!(val[max_pos], 0x3c00, "k={k}: 最大の |x| は 1.0 になるはず");
        }
        assert_eq!(x, x_before, "x を書き換えている");
    }

    /// P53: 正規化の境界 (全部 0 / 1 個だけ非 0 / 空 / k=0)。
    #[test]
    fn layer_act_frame_normalization_edges() {
        let mut scratch = Vec::new();

        // 全部 0 → 空フレーム (0/0 を作らない)
        let z = vec![0.0f32; 2560];
        assert_eq!(
            layer_act_frame(3, &z, 64, &mut scratch),
            Frame::LayerAct { layer: 3, idx: Vec::new(), val: Vec::new() }
        );
        // -0.0 も 0 として扱う
        let nz = vec![-0.0f32; 8];
        assert_eq!(
            layer_act_frame(1, &nz, 4, &mut scratch),
            Frame::LayerAct { layer: 1, idx: Vec::new(), val: Vec::new() }
        );

        // 1 個だけ非 0 → その index 1 件だけ、値は 1.0
        let mut one = vec![0.0f32; 2560];
        one[123] = -2.5;
        assert_eq!(
            layer_act_frame(7, &one, 64, &mut scratch),
            Frame::LayerAct { layer: 7, idx: vec![123], val: vec![0x3c00] }
        );

        // 空スライス / k=0 → 空フレーム
        assert_eq!(
            layer_act_frame(0, &[], 64, &mut scratch),
            Frame::LayerAct { layer: 0, idx: Vec::new(), val: Vec::new() }
        );
        assert_eq!(
            layer_act_frame(0, &[1.0, 2.0], 0, &mut scratch),
            Frame::LayerAct { layer: 0, idx: Vec::new(), val: Vec::new() }
        );

        // 同値は index が小さい方が先 (入力順に依らない)
        let t = vec![1.0f32, -1.0, 1.0, -1.0];
        assert_eq!(
            layer_act_frame(2, &t, 2, &mut scratch),
            Frame::LayerAct { layer: 2, idx: vec![0, 1], val: vec![0x3c00, 0x3c00] }
        );
        // 2 番目に大きい値の正規化 (0.5 → 0x3800)
        assert_eq!(
            layer_act_frame(2, &[2.0, 1.0], 1, &mut scratch),
            Frame::LayerAct { layer: 2, idx: vec![0], val: vec![0x3c00] }
        );
        let f = layer_act_frame(2, &[2.0, 1.0], 2, &mut scratch);
        assert_eq!(
            f,
            Frame::LayerAct { layer: 2, idx: vec![0, 1], val: vec![0x3c00, 0x3800] }
        );
    }

    /// P53: env の解釈 (既定 off・K の既定 64)。
    #[test]
    fn act_env_parsing() {
        assert!(!act_from_value(None), "未設定は既定 off");
        assert!(!act_from_value(Some("0")));
        assert!(!act_from_value(Some("")));
        assert!(act_from_value(Some("1")));
        assert!(act_from_value(Some("true")), "0 以外は有効");
        assert_eq!(act_topk_from_value(None), 64);
        assert_eq!(act_topk_from_value(Some("")), 64);
        assert_eq!(act_topk_from_value(Some("x")), 64, "読めない値は既定");
        assert_eq!(act_topk_from_value(Some("1")), 1);
        assert_eq!(act_topk_from_value(Some(" 128 ")), 128, "前後の空白は許す");
        assert!(!act_dbg_from_value(None));
        assert!(act_dbg_from_value(Some("1")));
        assert!(act_dbg_from_value(Some("true")));
        assert!(!act_dbg_from_value(Some("0")));
        // テスト用の上書きは元に戻せる (他のテストに漏らさない)
        let before = act_enabled();
        set_act_for_tests(true);
        assert!(act_enabled());
        set_act_for_tests(false);
        assert!(!act_enabled());
        set_act_for_tests(before);
    }

    #[test]
    fn decode_rejects_bad_input() {
        assert!(Frame::decode(&[]).is_err());
        assert!(Frame::decode(b"XXXX\x01\x01").is_err());
        assert!(Frame::decode(&[b'D', b'N', b'T', b'L', 9, 1]).is_err()); // 未知 version
        assert!(Frame::decode(&[b'D', b'N', b'T', b'L', 1, 99]).is_err()); // 未知 kind
    }

    #[test]
    fn topk_softmax_matches_reference() {
        // 手計算できる小さな例: logits [2, 1, 0]
        let tk = topk_softmax(&[2.0, 1.0, 0.0], 2);
        // softmax = [e^2, e^1, 1] / (e^2 + e^1 + 1)
        let s = 2f32.exp() + 1f32.exp() + 1.0;
        let p0 = 2f32.exp() / s;
        let p1 = 1f32.exp() / s;
        let e0 = tk.entries[0];
        assert_eq!(e0.0, 0);
        assert!((e0.1 - p0).abs() < 1e-6, "{} vs {}", e0.1, p0);
        assert_eq!(tk.entries[1].0, 1);
        assert!((tk.entries[1].1 - p1).abs() < 1e-6);
        let h_ref = -(p0 * p0.ln() + p1 * p1.ln() + (1.0 / s) * (1.0 / s).ln());
        assert!((tk.entropy - h_ref).abs() < 1e-5, "{} vs {}", tk.entropy, h_ref);
        // 同点 logit は id 昇順
        let tk2 = topk_softmax(&[1.0, 1.0, 1.0], 3);
        assert_eq!(tk2.entries.iter().map(|e| e.0).collect::<Vec<_>>(), vec![0, 1, 2]);
        // 確率の合計は top-k ≤ 1
        let sum: f32 = tk.entries.iter().map(|e| e.1).sum();
        assert!(sum <= 1.0 + 1e-6);
    }

    /// 1 パス選択が旧実装 (語彙全体の sort_unstable_by) と 1 件も違わないこと。
    ///
    /// 同点・NaN・k が語彙数以上・k=0 を混ぜる。乱数は固定の LCG (再現可能)。
    #[test]
    fn topk_select_matches_full_sort() {
        fn full_sort(logits: &[f32], k: usize) -> Vec<u32> {
            let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
            idx.sort_unstable_by(|&a, &b| {
                logits[b as usize].total_cmp(&logits[a as usize]).then(a.cmp(&b))
            });
            idx.truncate(k);
            idx
        }
        let mut seed = 0x1234_5678u32;
        let mut next = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((seed >> 8) as f32 / (1u32 << 24) as f32) * 20.0 - 10.0
        };
        for trial in 0..40 {
            let n = 1 + (trial * 7) % 300;
            let mut logits: Vec<f32> = (0..n).map(|_| next()).collect();
            // 同点を作る (丸めで衝突させる)
            if trial % 3 == 0 && n > 4 {
                logits[1] = logits[0];
                logits[2] = logits[0];
            }
            if trial == 7 {
                logits[n / 2] = f32::NAN;
            }
            for k in [0usize, 1, 5, 8, n, n + 3] {
                let got: Vec<u32> = topk_softmax(&logits, k).entries.iter().map(|e| e.0).collect();
                let want = full_sort(&logits, k.min(n));
                assert_eq!(got, want, "trial={trial} n={n} k={k}");
            }
        }
    }
}
