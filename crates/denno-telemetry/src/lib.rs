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
}

pub const MAGIC: [u8; 4] = *b"DNTL";
pub const VERSION: u8 = 1;

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
    ServerInfo { model: String, quant: String, slots: u32, context: u32 },
    /// セッションが KV スロットを取得 (install)
    SessionAdmit { session: u32, slot: u32, prompt_tokens: u32 },
    /// セッション終了 (release)
    SessionDone { session: u32, reason: u8, output_tokens: u32 },
    /// prefill 1 forward 完了
    Prefill { session: u32, prompt_tokens: u32, ns: u64 },
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
        }
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(16);
        b.extend_from_slice(&MAGIC);
        b.push(VERSION);
        b.push(self.kind());
        match self {
            Frame::ServerInfo { model, quant, slots, context } => {
                put_varint_u32(&mut b, model.len() as u32);
                b.extend_from_slice(model.as_bytes());
                put_varint_u32(&mut b, quant.len() as u32);
                b.extend_from_slice(quant.as_bytes());
                put_varint_u32(&mut b, *slots);
                put_varint_u32(&mut b, *context);
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
            Frame::Prefill { session, prompt_tokens, ns } => {
                put_varint_u32(&mut b, *session);
                put_varint_u32(&mut b, *prompt_tokens);
                put_varint_u64(&mut b, *ns);
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
                let n2 = u32v(p, &mut i)? as usize;
                let quant = String::from_utf8(p[i..i + n2].to_vec()).map_err(|e| e.to_string())?;
                i += n2;
                let slots = u32v(p, &mut i)?;
                let context = u32v(p, &mut i)?;
                Frame::ServerInfo { model, quant, slots, context }
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
                Frame::Prefill { session, prompt_tokens, ns }
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
pub fn emit(frame: Frame) {
    if !enabled() {
        return;
    }
    let tx = TX.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::sync_channel(CHANNEL_CAP);
        let _ = RX.set(Mutex::new(rx));
        tx
    });
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
                quant: "fp32".into(),
                slots: 4,
                context: 32768,
            },
            Frame::SessionAdmit { session: 7, slot: 2, prompt_tokens: 34 },
            Frame::SessionDone { session: 7, reason: done_reason::END_TURN, output_tokens: 26 },
            Frame::Prefill { session: 7, prompt_tokens: 34, ns: 123_456_789 },
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
        ];
        for f in frames {
            let enc = f.encode();
            assert_eq!(&enc[..4], &MAGIC);
            let dec = Frame::decode(&enc).unwrap();
            assert_eq!(dec, f, "roundtrip failed for kind {}", f.kind());
        }
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
