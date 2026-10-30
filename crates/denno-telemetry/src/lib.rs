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
use std::sync::atomic::{AtomicI8, AtomicU32, AtomicU64, Ordering};
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
    /// sampler の top-k (確率は f16 ビット)
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
                put_varint_u32(&mut b, busy.len() as u32);
                for &x in busy {
                    b.push(x as u8);
                }
                for &t in t_lens {
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
                put_varint_u32(&mut b, *layer as u32);
                put_varint_u32(&mut b, idx.len() as u32);
                for &i in idx {
                    put_varint_u32(&mut b, i);
                }
                for &v in val {
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
                put_varint_u32(&mut b, *layer as u32);
                put_varint_u32(&mut b, experts.len() as u32);
                for &e in experts {
                    put_varint_u32(&mut b, e as u32);
                }
                // 重みは experts と同じ本数を f16 ビット (2B, LE) で並べる。
                for &w in weights {
                    b.extend_from_slice(&w.to_le_bytes());
                }
            }
        }
        b
    }

    /// WS メッセージ 1 個 (= フレーム 1 個) をデコードする。
    pub fn decode(buf: &[u8]) -> Result<Frame, String> {
        if buf.len() < 6 || buf[..4] != MAGIC {
            return Err("bad magic".into());
        }
        if buf[4] != VERSION {
            return Err(format!("unsupported version: {}", buf[4]));
        }
        let k = buf[5];
        let p = &buf[6..];
        let mut i = 0usize;
        let u32v = |p: &[u8], i: &mut usize| -> Result<u32, String> {
            let (v, n) = get_varint_u32(&p[*i..])?;
            *i += n;
            Ok(v)
        };
        let u64v = |p: &[u8], i: &mut usize| -> Result<u64, String> {
            let (v, n) = get_varint_u64(&p[*i..])?;
            *i += n;
            Ok(v)
        };
        let f = match k {
            kind::SERVER_INFO => {
                let n = u32v(p, &mut i)? as usize;
                let model = String::from_utf8(p[i..i + n].to_vec()).map_err(|e| e.to_string())?;
                i += n;
                let nid = u32v(p, &mut i)? as usize;
                let model_id = String::from_utf8(p[i..i + nid].to_vec()).map_err(|e| e.to_string())?;
                i += nid;
                let n2 = u32v(p, &mut i)? as usize;
                let quant = String::from_utf8(p[i..i + n2].to_vec()).map_err(|e| e.to_string())?;
                i += n2;
                let slots = u32v(p, &mut i)?;
                let context = u32v(p, &mut i)?;
                let layers = u32v(p, &mut i)?;
                Frame::ServerInfo { model, model_id, quant, slots, context, layers }
            }
            kind::SESSION_ADMIT => {
                let session = u32v(p, &mut i)?;
                let slot = u32v(p, &mut i)?;
                let prompt_tokens = u32v(p, &mut i)?;
                Frame::SessionAdmit { session, slot, prompt_tokens }
            }
            kind::SESSION_DONE => {
                let session = u32v(p, &mut i)?;
                let reason = p[i];
                i += 1;
                let output_tokens = u32v(p, &mut i)?;
                Frame::SessionDone { session, reason, output_tokens }
            }
            kind::PREFILL => {
                let session = u32v(p, &mut i)?;
                let prompt_tokens = u32v(p, &mut i)?;
                let ns = u64v(p, &mut i)?;
                let n = u32v(p, &mut i)? as usize;
                let mut tokens = Vec::with_capacity(n);
                for _ in 0..n {
                    tokens.push(u32v(p, &mut i)?);
                }
                Frame::Prefill { session, prompt_tokens, ns, tokens }
            }
            kind::TOKEN => {
                let session = u32v(p, &mut i)?;
                let token = u32v(p, &mut i)?;
                let pos = u32v(p, &mut i)?;
                Frame::Token { session, token, pos }
            }
            kind::STEP => {
                let width = u32v(p, &mut i)?;
                let ns = u64v(p, &mut i)?;
                Frame::Step { width, ns }
            }
            kind::OP_SUMMARY => {
                let n = u32v(p, &mut i)? as usize;
                let mut ops = Vec::with_capacity(n);
                for _ in 0..n {
                    let op = p[i];
                    i += 1;
                    let ns = u64v(p, &mut i)?;
                    ops.push((op, ns));
                }
                Frame::OpSummary { ops }
            }
            kind::KV_USAGE => {
                let n = u32v(p, &mut i)? as usize;
                let mut busy = Vec::with_capacity(n);
                for _ in 0..n {
                    busy.push(p[i] != 0);
                    i += 1;
                }
                let mut t_lens = Vec::with_capacity(n);
                for _ in 0..n {
                    t_lens.push(u32v(p, &mut i)?);
                }
                Frame::KvUsage { busy, t_lens }
            }
            kind::GAP => {
                let lost = u32v(p, &mut i)?;
                Frame::Gap { lost }
            }
            kind::TOP_K => {
                let session = u32v(p, &mut i)?;
                let chosen = u32v(p, &mut i)?;
                let entropy_f16 = u16::from_le_bytes([p[i], p[i + 1]]);
                i += 2;
                let n = u32v(p, &mut i)? as usize;
                let mut entries = Vec::with_capacity(n);
                for _ in 0..n {
                    let id = u32v(p, &mut i)?;
                    let pb = [p[i], p[i + 1]];
                    i += 2;
                    entries.push((id, u16::from_le_bytes(pb)));
                }
                Frame::TopK { session, chosen, entropy_f16, entries }
            }
            kind::LAYER_RUN => {
                let layer = u32v(p, &mut i)? as u16;
                let n = u32v(p, &mut i)? as usize;
                let mut ops = Vec::with_capacity(n);
                for _ in 0..n {
                    let op = p[i];
                    i += 1;
                    let ns = u64v(p, &mut i)?;
                    ops.push((op, ns));
                }
                Frame::LayerRun { layer, ops }
            }
            kind::LAYER_ACT => {
                let layer = u32v(p, &mut i)? as u16;
                let n = u32v(p, &mut i)? as usize;
                let mut idx = Vec::with_capacity(n);
                for _ in 0..n {
                    idx.push(u32v(p, &mut i)?);
                }
                let mut val = Vec::with_capacity(n);
                for _ in 0..n {
                    let vb = [p[i], p[i + 1]];
                    i += 2;
                    val.push(u16::from_le_bytes(vb));
                }
                Frame::LayerAct { layer, idx, val }
            }
            kind::ROLE_ACT => {
                let layer = u32v(p, &mut i)? as u16;
                let mut v = [0u16; 5];
                for slot in v.iter_mut() {
                    *slot = u16::from_le_bytes([p[i], p[i + 1]]);
                    i += 2;
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
                let layer = u32v(p, &mut i)? as u16;
                let n = u32v(p, &mut i)? as usize;
                let mut in_vram = Vec::with_capacity(n);
                for _ in 0..n {
                    in_vram.push(u32v(p, &mut i)? as u16);
                }
                Frame::ExpertResidency { layer, in_vram }
            }
            kind::TIER_STATS => {
                let vram_used_mb = u32v(p, &mut i)?;
                let vram_total_mb = u32v(p, &mut i)?;
                let ram_experts = u32v(p, &mut i)?;
                let ssd_reads = u32v(p, &mut i)?;
                Frame::TierStats { vram_used_mb, vram_total_mb, ram_experts, ssd_reads }
            }
            kind::ROUTER_TOPK => {
                let layer = u32v(p, &mut i)? as u16;
                let n = u32v(p, &mut i)? as usize;
                let mut experts = Vec::with_capacity(n);
                for _ in 0..n {
                    let e = u32v(p, &mut i)?;
                    if e > u16::MAX as u32 {
                        return Err(format!("router_topk: expert index {e} が u16 を超える"));
                    }
                    experts.push(e as u16);
                }
                let mut weights = Vec::with_capacity(n);
                for _ in 0..n {
                    // 途中で切れたペイロードで panic しない (index の範囲を先に見る)。
                    if i + 2 > p.len() {
                        return Err(format!(
                            "router_topk: weights が途中で切れている ({i}/{})",
                            p.len()
                        ));
                    }
                    let wb = [p[i], p[i + 1]];
                    i += 2;
                    weights.push(u16::from_le_bytes(wb));
                }
                Frame::RouterTopK { layer, experts, weights }
            }
            other => return Err(format!("unknown kind: {other}")),
        };
        if i != p.len() {
            return Err(format!("trailing bytes: {} / {}", i, p.len()));
        }
        Ok(f)
    }
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

/// フレームを送出する。非ブロッキング。満杯/切断時はドロップして計上。
/// enabled() でない場合は即座に戻る (フレーム生成コストすら掛けないよう
/// 呼び出し側も enabled() ガードを推奨)。
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
pub fn normalize_roles(roles: [f32; 5]) -> [f32; 5] {
    let mut max = 0.0f32;
    for v in roles {
        if v > max {
            max = v;
        }
    }
    if !(max > 0.0) {
        return [0.0; 5];
    }
    let mut out = [0.0f32; 5];
    for (o, v) in out.iter_mut().zip(roles) {
        *o = (v / max).clamp(0.0, 1.0);
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
    // top-k 選択 (logit 降順、同点は id 昇順 — 部分ソートではなく全ソートで単純に)
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    idx.sort_unstable_by(|&a, &b| {
        logits[b as usize].total_cmp(&logits[a as usize]).then(a.cmp(&b))
    });
    idx.truncate(k);
    let entries =
        idx.into_iter().map(|i| (i, ((logits[i as usize] - max).exp() / s).max(0.0))).collect();
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
}
