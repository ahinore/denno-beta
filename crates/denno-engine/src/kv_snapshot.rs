//! task 870: KV cache のディスクスナップショット (save / restore)。
//!
//! # なぜ要るか
//!
//! `denno serve-flash` を再起動すると、8k-16k のシステムプロンプトを毎回
//! フル prefill し直す。プロンプト接頭辞の KV (+ GDN 再帰状態) を SSD に
//! 置いておけば、同じプロンプトの冷たい再起動で prefill を丸ごと省ける。
//!
//! # 形式 (version 1)
//!
//! ヘッダ 72 バイト + ペイロード。すべて little endian。
//!
//! | offset | 型 | 内容 |
//! | --- | --- | --- |
//! | 0 | `[u8; 8]` | magic `DNKVSNAP` |
//! | 8 | `u32` | version (現在 1) |
//! | 12 | `u32` | header_len (72) |
//! | 16 | `u64` | model fingerprint (P798 `inventory_fingerprint`) |
//! | 24 | `u64` | prompt token hash ([`prompt_hash`]) |
//! | 32 | `u64` | n_tokens (スナップショットのトークン数) |
//! | 40 | `u32` | n_layers |
//! | 44 | `u32` | kv_dim |
//! | 48 | `u32` | gdn_layers (GDN スロットを持つ層の数) |
//! | 52 | `u32` | reserved (0) |
//! | 56 | `u64` | payload_len |
//! | 64 | `u64` | payload checksum (FNV-1a 64) |
//!
//! ペイロードは層 0 から順に:
//!
//! ```text
//! u32 tag (bit0 = KV あり / bit1 = GDN あり)
//! [tag bit0] f32 * n_tokens * kv_dim   (K)
//! [tag bit0] f32 * n_tokens * kv_dim   (V)
//! [tag bit1] u64 conv_len, f32 * conv_len
//! [tag bit1] u64 ssm_len,  f32 * ssm_len
//! ```
//!
//! f32 は `to_le_bytes` のビット列なので、NaN を含めて**ビット一致**で
//! 往復する (数値の丸めは一切しない)。
//!
//! # 同一性の 3 点
//!
//! 復元を受け入れるのは、次の 3 つがすべて一致したときだけ:
//! 1. magic + version
//! 2. model fingerprint (テンソル目録の指紋。量子型の取り違えを捕まえる)
//! 3. prompt token hash + n_tokens (同じプロンプト接頭辞か)
//!
//! さらに checksum・payload_len・層数・kv_dim・GDN スロットの有無と長さを
//! 検証する。どの検証も**復元先のキャッシュを書き換える前**に行う。
//!
//! # 無効時
//!
//! `DENNO_KV_SNAPSHOT_DIR` が未設定 / 空なら [`restore_if_enabled`] /
//! [`save_if_enabled`] は何もしない (既定 off)。ディスク使用量は
//! `DENNO_KV_SNAPSHOT_MB` (既定 512 MiB、0 = 無制限) で抑え、超えるときは
//! mtime の古い `*.kvsnap` から消す。書き込みは同ディレクトリの一時ファイル
//! に書いて `fsync` してから `rename` する (途中で落ちても半端なファイルが
//! 正規の名前で見えない)。
//!
//! # 呼び出し側の契約
//!
//! GDN 状態は位置スライスできない再帰状態なので、スナップショットは
//! **prompt 境界でのみ**取る (`n_tokens == prompt.len()`・prefill 直後)。
//! 同じプロンプト接頭辞の同じ位置でしか復元してはいけない。

use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::kv_cache::KvCache;
use crate::model::exp_stats::{FNV_OFFSET, FNV_PRIME};

/// スナップショットを置くディレクトリ。未設定 = 機能 off。
pub const DIR_ENV: &str = "DENNO_KV_SNAPSHOT_DIR";

/// ディスク使用量の上限 (MiB)。既定 [`DEFAULT_QUOTA_MB`]、0 = 無制限。
pub const QUOTA_MB_ENV: &str = "DENNO_KV_SNAPSHOT_MB";

/// [`QUOTA_MB_ENV`] の既定値。
pub const DEFAULT_QUOTA_MB: u64 = 512;

/// ファイル magic。
pub const MAGIC: [u8; 8] = *b"DNKVSNAP";

/// 形式の version。
pub const VERSION: u32 = 1;

/// ヘッダ長 (バイト)。
pub const HEADER_LEN: usize = 72;

/// スナップショットの拡張子。
pub const EXTENSION: &str = "kvsnap";

static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// スナップショットの読み書きで起きうる失敗。
#[derive(Debug)]
pub enum SnapshotError {
    /// ファイル操作の失敗。
    Io(io::Error),
    /// magic が違う (KV スナップショットではない)。
    BadMagic([u8; 8]),
    /// 未知の version。
    UnsupportedVersion(u32),
    /// ヘッダまたはペイロードが短い。
    Truncated { expected: usize, actual: usize },
    /// payload_len と実際の長さが違う。
    LengthMismatch { declared: u64, actual: u64 },
    /// checksum 不一致 (壊れたファイル)。
    BadChecksum { expected: u64, actual: u64 },
    /// モデルの指紋が違う。
    ModelMismatch { expected: u64, found: u64 },
    /// プロンプトが違う。
    PromptMismatch { expected: u64, found: u64 },
    /// 層数・kv_dim・GDN 形状などが復元先と合わない。
    ShapeMismatch(String),
    /// ディスク上限を 1 本で超える。
    QuotaExceeded { needed: u64, quota: u64 },
}

impl std::fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SnapshotError::Io(e) => write!(f, "kv snapshot io: {e}"),
            SnapshotError::BadMagic(m) => write!(f, "kv snapshot magic 不一致: {m:?}"),
            SnapshotError::UnsupportedVersion(v) => {
                write!(f, "kv snapshot version {v} は未対応 (期待 {VERSION})")
            }
            SnapshotError::Truncated { expected, actual } => {
                write!(f, "kv snapshot が短い: {actual} < {expected}")
            }
            SnapshotError::LengthMismatch { declared, actual } => {
                write!(f, "kv snapshot 長不一致: ヘッダ {declared} / 実体 {actual}")
            }
            SnapshotError::BadChecksum { expected, actual } => {
                write!(f, "kv snapshot checksum 不一致: {actual:#018x} != {expected:#018x}")
            }
            SnapshotError::ModelMismatch { expected, found } => {
                write!(f, "kv snapshot モデル指紋不一致: {found:#018x} != {expected:#018x}")
            }
            SnapshotError::PromptMismatch { expected, found } => {
                write!(f, "kv snapshot プロンプト不一致: {found:#018x} != {expected:#018x}")
            }
            SnapshotError::ShapeMismatch(s) => write!(f, "kv snapshot 形状不一致: {s}"),
            SnapshotError::QuotaExceeded { needed, quota } => {
                write!(f, "kv snapshot が上限超過: {needed} > {quota} バイト")
            }
        }
    }
}

impl std::error::Error for SnapshotError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SnapshotError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for SnapshotError {
    fn from(e: io::Error) -> Self {
        SnapshotError::Io(e)
    }
}

fn fnv_bytes(mut h: u64, b: &[u8]) -> u64 {
    for &x in b {
        h ^= x as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

fn fnv_u64(h: u64, v: u64) -> u64 {
    fnv_bytes(h, &v.to_le_bytes())
}

fn checksum(b: &[u8]) -> u64 {
    fnv_bytes(FNV_OFFSET, b)
}

/// プロンプトのトークン列の指紋 (FNV-1a 64・長さ込み)。
pub fn prompt_hash(tokens: &[u32]) -> u64 {
    let mut h = fnv_u64(FNV_OFFSET, tokens.len() as u64);
    for &t in tokens {
        h = fnv_u64(h, t as u64);
    }
    h
}

/// `DENNO_KV_SNAPSHOT_DIR` の値の解釈 (純関数)。空 / 空白のみ = off。
pub fn dir_from_value(v: Option<&str>) -> Option<PathBuf> {
    let s = v?.trim();
    if s.is_empty() {
        None
    } else {
        Some(PathBuf::from(s))
    }
}

/// `DENNO_KV_SNAPSHOT_MB` の値の解釈 (純関数)。未設定 / 不正は既定値。
pub fn quota_mb_from_value(v: Option<&str>) -> u64 {
    v.and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_QUOTA_MB)
}

/// ゲートされたディレクトリ (env を毎回読む。未設定 = `None` = off)。
pub fn dir_from_env() -> Option<PathBuf> {
    dir_from_value(std::env::var(DIR_ENV).ok().as_deref())
}

/// ゲートされた上限バイト数 (0 = 無制限)。
pub fn quota_bytes_from_env() -> u64 {
    quota_mb_from_value(std::env::var(QUOTA_MB_ENV).ok().as_deref()).saturating_mul(1024 * 1024)
}

/// 機能が有効か (`DENNO_KV_SNAPSHOT_DIR` が非空)。既定 off。
pub fn enabled() -> bool {
    dir_from_env().is_some()
}

/// スナップショットのファイル名 (キーは モデル指紋 + プロンプト指紋 + 長さ)。
pub fn snapshot_path(dir: &Path, model_fingerprint: u64, prompt_hash: u64, n_tokens: usize) -> PathBuf {
    dir.join(format!(
        "{model_fingerprint:016x}-{prompt_hash:016x}-{n_tokens}.{EXTENSION}"
    ))
}

/// `snapshot_path` をプロンプトから直接作る。
pub fn snapshot_path_for(dir: &Path, model_fingerprint: u64, prompt: &[u32]) -> PathBuf {
    snapshot_path(
        dir,
        model_fingerprint,
        prompt_hash(prompt),
        prompt.len(),
    )
}

fn put_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn put_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn put_f32s(buf: &mut Vec<u8>, v: &[f32]) {
    buf.reserve(v.len() * 4);
    for x in v {
        buf.extend_from_slice(&x.to_le_bytes());
    }
}

/// ヘッダ + ペイロードを組み立てる (純関数。ディスクに触らない)。
pub fn encode(
    cache: &KvCache,
    n_tokens: usize,
    model_fingerprint: u64,
    prompt: &[u32],
) -> Result<Vec<u8>, SnapshotError> {
    if n_tokens == 0 {
        return Err(SnapshotError::ShapeMismatch("n_tokens が 0".into()));
    }
    if prompt.len() != n_tokens {
        return Err(SnapshotError::ShapeMismatch(format!(
            "prompt 長 {} != n_tokens {n_tokens} (スナップショットは prompt 境界でのみ有効)",
            prompt.len()
        )));
    }
    if n_tokens > cache.len() {
        return Err(SnapshotError::ShapeMismatch(format!(
            "n_tokens {n_tokens} > キャッシュ長 {}",
            cache.len()
        )));
    }
    let (k, v) = cache.snapshot(n_tokens);
    let gdn = cache.gdn_snapshot();
    let mut payload = Vec::with_capacity(n_tokens * cache.kv_dim() * 8);
    let mut gdn_layers = 0u32;
    for layer in 0..cache.n_layers() {
        let has_kv = !k[layer].is_empty();
        let has_gdn = gdn[layer].is_some();
        if has_gdn {
            gdn_layers += 1;
        }
        put_u32(&mut payload, (has_kv as u32) | ((has_gdn as u32) << 1));
        if has_kv {
            put_f32s(&mut payload, &k[layer]);
            put_f32s(&mut payload, &v[layer]);
        }
        if let Some((conv, ssm)) = &gdn[layer] {
            put_u64(&mut payload, conv.len() as u64);
            put_f32s(&mut payload, conv);
            put_u64(&mut payload, ssm.len() as u64);
            put_f32s(&mut payload, ssm);
        }
    }

    let mut out = Vec::with_capacity(HEADER_LEN + payload.len());
    out.extend_from_slice(&MAGIC);
    put_u32(&mut out, VERSION);
    put_u32(&mut out, HEADER_LEN as u32);
    put_u64(&mut out, model_fingerprint);
    put_u64(&mut out, prompt_hash(prompt));
    put_u64(&mut out, n_tokens as u64);
    put_u32(&mut out, cache.n_layers() as u32);
    put_u32(&mut out, cache.kv_dim() as u32);
    put_u32(&mut out, gdn_layers);
    put_u32(&mut out, 0);
    put_u64(&mut out, payload.len() as u64);
    put_u64(&mut out, checksum(&payload));
    out.extend_from_slice(&payload);
    Ok(out)
}

/// 検証済みのスナップショット (復元先へ書き込む前の中間表現)。
#[derive(Debug)]
struct Decoded {
    n_tokens: usize,
    model_fingerprint: u64,
    prompt_digest: u64,
    n_layers: usize,
    kv_dim: usize,
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    gdn: Vec<Option<(Vec<f32>, Vec<f32>)>>,
}

struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(b: &'a [u8]) -> Self {
        Reader { b, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], SnapshotError> {
        let end = self.pos.checked_add(n).ok_or(SnapshotError::Truncated {
            expected: usize::MAX,
            actual: self.b.len(),
        })?;
        if end > self.b.len() {
            return Err(SnapshotError::Truncated {
                expected: end,
                actual: self.b.len(),
            });
        }
        let s = &self.b[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u32(&mut self) -> Result<u32, SnapshotError> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    fn u64(&mut self) -> Result<u64, SnapshotError> {
        let s = self.take(8)?;
        Ok(u64::from_le_bytes([
            s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
        ]))
    }

    fn f32s(&mut self, n: usize) -> Result<Vec<f32>, SnapshotError> {
        let s = self.take(n * 4)?;
        let mut out = Vec::with_capacity(n);
        for c in s.chunks_exact(4) {
            out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        }
        Ok(out)
    }
}

/// バイト列をヘッダ・checksum・形状まで検証して中間表現にする。
fn decode(bytes: &[u8]) -> Result<Decoded, SnapshotError> {
    if bytes.len() < HEADER_LEN {
        return Err(SnapshotError::Truncated {
            expected: HEADER_LEN,
            actual: bytes.len(),
        });
    }
    let mut magic = [0u8; 8];
    magic.copy_from_slice(&bytes[0..8]);
    if magic != MAGIC {
        return Err(SnapshotError::BadMagic(magic));
    }
    let mut r = Reader::new(&bytes[HEADER_LEN..]);
    let v = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    if v != VERSION {
        return Err(SnapshotError::UnsupportedVersion(v));
    }
    let header_len = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
    if header_len != HEADER_LEN {
        return Err(SnapshotError::ShapeMismatch(format!(
            "header_len {header_len} != {HEADER_LEN}"
        )));
    }
    let model_fingerprint = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    let prompt_digest = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
    let n_tokens = u64::from_le_bytes(bytes[32..40].try_into().unwrap()) as usize;
    let n_layers = u32::from_le_bytes(bytes[40..44].try_into().unwrap()) as usize;
    let kv_dim = u32::from_le_bytes(bytes[44..48].try_into().unwrap()) as usize;
    let gdn_layers = u32::from_le_bytes(bytes[48..52].try_into().unwrap()) as usize;
    let payload_len = u64::from_le_bytes(bytes[56..64].try_into().unwrap());
    let payload_checksum = u64::from_le_bytes(bytes[64..72].try_into().unwrap());
    let payload = &bytes[HEADER_LEN..];
    if payload_len != payload.len() as u64 {
        return Err(SnapshotError::LengthMismatch {
            declared: payload_len,
            actual: payload.len() as u64,
        });
    }
    let actual = checksum(payload);
    if actual != payload_checksum {
        return Err(SnapshotError::BadChecksum {
            expected: payload_checksum,
            actual,
        });
    }
    if kv_dim == 0 {
        return Err(SnapshotError::ShapeMismatch("kv_dim が 0".into()));
    }

    let mut k = Vec::with_capacity(n_layers);
    let mut v = Vec::with_capacity(n_layers);
    let mut gdn = Vec::with_capacity(n_layers);
    let kv_elems = n_tokens
        .checked_mul(kv_dim)
        .ok_or_else(|| SnapshotError::ShapeMismatch("n_tokens * kv_dim が溢れる".into()))?;
    let mut seen_gdn = 0usize;
    for layer in 0..n_layers {
        let tag = r.u32()?;
        let has_kv = tag & 1 != 0;
        let has_gdn = tag & 2 != 0;
        if tag & !3 != 0 {
            return Err(SnapshotError::ShapeMismatch(format!(
                "layer {layer} の tag {tag} に未知ビット"
            )));
        }
        if has_kv {
            k.push(r.f32s(kv_elems)?);
            v.push(r.f32s(kv_elems)?);
        } else {
            k.push(Vec::new());
            v.push(Vec::new());
        }
        if has_gdn {
            seen_gdn += 1;
            let conv_len = r.u64()? as usize;
            let conv = r.f32s(conv_len)?;
            let ssm_len = r.u64()? as usize;
            let ssm = r.f32s(ssm_len)?;
            gdn.push(Some((conv, ssm)));
        } else {
            gdn.push(None);
        }
    }
    if seen_gdn != gdn_layers {
        return Err(SnapshotError::ShapeMismatch(format!(
            "gdn_layers {gdn_layers} != 実体 {seen_gdn}"
        )));
    }
    if r.pos != payload.len() {
        return Err(SnapshotError::LengthMismatch {
            declared: payload_len,
            actual: HEADER_LEN as u64 + r.pos as u64,
        });
    }
    Ok(Decoded {
        n_tokens,
        model_fingerprint,
        prompt_digest,
        n_layers,
        kv_dim,
        k,
        v,
        gdn,
    })
}

/// ディスク上限のための、既存スナップショットの一覧 (古い順)。
fn list_snapshots(dir: &Path) -> io::Result<Vec<(PathBuf, u64, std::time::SystemTime)>> {
    let mut out = Vec::new();
    let rd = match fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(out),
        Err(e) => return Err(e),
    };
    for ent in rd {
        let ent = ent?;
        let path = ent.path();
        if path.extension().and_then(|s| s.to_str()) != Some(EXTENSION) {
            continue;
        }
        let md = match ent.metadata() {
            Ok(md) => md,
            Err(_) => continue,
        };
        if !md.is_file() {
            continue;
        }
        let mtime = md
            .modified()
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        out.push((path, md.len(), mtime));
    }
    out.sort_by(|a, b| a.2.cmp(&b.2).then_with(|| a.0.cmp(&b.0)));
    Ok(out)
}

/// ディレクトリ内のスナップショットの合計バイト数。
pub fn total_bytes(dir: &Path) -> io::Result<u64> {
    Ok(list_snapshots(dir)?.iter().map(|(_, n, _)| *n).sum())
}

/// 合計が `budget` バイト以下になるまで、mtime の古い `*.kvsnap` を消す。
/// 戻り値は消したバイト数。`budget` を 1 本でも下回れない場合は消せるだけ消す。
pub fn prune_to_budget(dir: &Path, budget: u64) -> io::Result<u64> {
    let files = list_snapshots(dir)?;
    let mut total: u64 = files.iter().map(|(_, n, _)| *n).sum();
    let mut freed = 0u64;
    for (path, len, _) in files {
        if total <= budget {
            break;
        }
        match fs::remove_file(&path) {
            Ok(()) => {
                total = total.saturating_sub(len);
                freed += len;
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                total = total.saturating_sub(len);
                freed += len;
            }
            Err(e) => return Err(e),
        }
    }
    Ok(freed)
}

/// キャッシュの先頭 `n_tokens` 行 (+ 現在の GDN 状態) をディスクへ保存する。
///
/// - `prompt` は `n_tokens` と同長でなければならない (prompt 境界の契約)。
/// - `quota_bytes` > 0 のとき、合計が上限に収まるよう古いスナップショットを
///   消してから書く。1 本で上限を超えるなら [`SnapshotError::QuotaExceeded`]。
/// - 書き込みは同ディレクトリの一時ファイル + `fsync` + `rename`。
pub fn save(
    cache: &KvCache,
    n_tokens: usize,
    dir: &Path,
    model_fingerprint: u64,
    prompt: &[u32],
    quota_bytes: u64,
) -> Result<PathBuf, SnapshotError> {
    let bytes = encode(cache, n_tokens, model_fingerprint, prompt)?;
    fs::create_dir_all(dir)?;
    let needed = bytes.len() as u64;
    if quota_bytes > 0 {
        if needed > quota_bytes {
            return Err(SnapshotError::QuotaExceeded {
                needed,
                quota: quota_bytes,
            });
        }
        prune_to_budget(dir, quota_bytes - needed)?;
    }
    let path = snapshot_path(dir, model_fingerprint, prompt_hash(prompt), n_tokens);
    let tmp = dir.join(format!(
        ".{EXTENSION}-tmp-{}-{}",
        std::process::id(),
        TMP_SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    {
        let mut f = File::create(&tmp)?;
        if let Err(e) = f.write_all(&bytes).and_then(|()| f.sync_all()) {
            drop(f);
            let _ = fs::remove_file(&tmp);
            return Err(SnapshotError::Io(e));
        }
    }
    if let Err(e) = fs::rename(&tmp, &path) {
        let _ = fs::remove_file(&tmp);
        return Err(SnapshotError::Io(e));
    }
    // ディレクトリの fsync はベストエフォート (失敗しても内容は書けている)。
    if let Ok(d) = File::open(dir) {
        let _ = d.sync_all();
    }
    Ok(path)
}

/// ファイルを読んで検証し、キャッシュへ復元する。
///
/// 検証 (magic / version / checksum / 指紋 / プロンプト / 形状) は
/// **キャッシュを書き換える前に**すべて済ませる。失敗した場合、キャッシュは
/// 元のまま。
pub fn load(
    cache: &mut KvCache,
    path: &Path,
    model_fingerprint: u64,
    prompt: &[u32],
) -> Result<(), SnapshotError> {
    let bytes = read_file(path)?;
    let d = decode(&bytes)?;
    if d.model_fingerprint != model_fingerprint {
        return Err(SnapshotError::ModelMismatch {
            expected: model_fingerprint,
            found: d.model_fingerprint,
        });
    }
    let want_digest = prompt_hash(prompt);
    if d.prompt_digest != want_digest || d.n_tokens != prompt.len() {
        return Err(SnapshotError::PromptMismatch {
            expected: want_digest,
            found: d.prompt_digest,
        });
    }
    if d.n_layers != cache.n_layers() {
        return Err(SnapshotError::ShapeMismatch(format!(
            "層数 {} != キャッシュ {}",
            d.n_layers,
            cache.n_layers()
        )));
    }
    if d.kv_dim != cache.kv_dim() {
        return Err(SnapshotError::ShapeMismatch(format!(
            "kv_dim {} != キャッシュ {}",
            d.kv_dim,
            cache.kv_dim()
        )));
    }
    if d.n_tokens > cache.max_len() {
        return Err(SnapshotError::ShapeMismatch(format!(
            "n_tokens {} > max_len {}",
            d.n_tokens,
            cache.max_len()
        )));
    }
    for layer in 0..d.n_layers {
        match (&d.gdn[layer], cache.gdn_slot_len(layer)) {
            (Some((conv, ssm)), Some((cl, sl))) => {
                if conv.len() != cl || ssm.len() != sl {
                    return Err(SnapshotError::ShapeMismatch(format!(
                        "layer {layer} の GDN 形状 ({}, {}) != キャッシュ ({cl}, {sl})",
                        conv.len(),
                        ssm.len()
                    )));
                }
            }
            (None, None) => {
                if d.k[layer].len() != d.n_tokens * d.kv_dim {
                    return Err(SnapshotError::ShapeMismatch(format!(
                        "layer {layer} の KV 長 {} != {}",
                        d.k[layer].len(),
                        d.n_tokens * d.kv_dim
                    )));
                }
            }
            (Some(_), None) => {
                return Err(SnapshotError::ShapeMismatch(format!(
                    "layer {layer} はスナップショットで GDN、キャッシュにスロットが無い"
                )))
            }
            (None, Some(_)) => {
                return Err(SnapshotError::ShapeMismatch(format!(
                    "layer {layer} はキャッシュが GDN、スナップショットに状態が無い"
                )))
            }
        }
    }

    cache.restore(&d.k, &d.v, d.n_tokens);
    cache.gdn_restore(&d.gdn);
    Ok(())
}

fn read_file(path: &Path) -> Result<Vec<u8>, SnapshotError> {
    let mut f = File::open(path)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    Ok(buf)
}

/// ゲートが有効なら保存する。off (`DENNO_KV_SNAPSHOT_DIR` 未設定) は `None`。
pub fn save_if_enabled(
    cache: &KvCache,
    n_tokens: usize,
    model_fingerprint: u64,
    prompt: &[u32],
) -> Option<Result<PathBuf, SnapshotError>> {
    let dir = dir_from_env()?;
    Some(save(
        cache,
        n_tokens,
        &dir,
        model_fingerprint,
        prompt,
        quota_bytes_from_env(),
    ))
}

/// ゲートが有効で、そのプロンプトのスナップショットが存在するときだけ復元する。
///
/// 戻り値: `None` = ゲート off またはファイル無し。`Some(Err(_))` = 検証失敗
/// (キャッシュは元のまま)。
pub fn restore_if_enabled(
    cache: &mut KvCache,
    model_fingerprint: u64,
    prompt: &[u32],
) -> Option<Result<PathBuf, SnapshotError>> {
    let dir = dir_from_env()?;
    let path = snapshot_path_for(&dir, model_fingerprint, prompt);
    if !path.exists() {
        return None;
    }
    Some(load(cache, &path, model_fingerprint, prompt).map(|()| path))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "denno_kv_snap_{name}_{}_{}",
            std::process::id(),
            TMP_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    /// 2 層 KV + 2 層 GDN の小さなキャッシュ。f32 は NaN / -0.0 を含む
    /// ビット列で埋める (ビット一致の検証用)。
    fn filled_cache(n_tokens: usize) -> KvCache {
        let kv_dim = 3;
        let mut c = KvCache::new(4, kv_dim, 32);
        c.attach_gdn_slot(0, 2, 3);
        c.attach_gdn_slot(2, 2, 3);
        for layer in 0..4 {
            if c.has_gdn_slot(layer) {
                continue;
            }
            let (k, v) = c.kv_bufs(layer, n_tokens);
            for i in 0..n_tokens * kv_dim {
                k[i] = bits_to_f32(
                    (0x3f00_0000u32)
                        .wrapping_add((layer as u32) * 0x1000)
                        .wrapping_add(i as u32 * 0x11),
                );
                v[i] = bits_to_f32(
                    0xbf80_0000u32
                        .wrapping_add((layer as u32) * 0x2000)
                        .wrapping_add(i as u32 * 0x07),
                );
            }
        }
        c.advance(n_tokens);
        for layer in [0usize, 2] {
            let (conv, ssm) = c.gdn_state_mut(layer).unwrap();
            conv.copy_from_slice(&[f32::NAN, -0.0]);
            ssm[0] = -1.5;
            ssm[1] = 3.25;
            ssm[2] = f32::from_bits(0x7fc0_1234); // NaN (payload のビットを保つ)
        }
        c
    }

    fn bits_to_f32(b: u32) -> f32 {
        f32::from_bits(b)
    }

    fn kv_bits(c: &KvCache, n: usize) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
        let (k, v) = c.snapshot(n);
        (
            k.iter()
                .map(|l| l.iter().map(|x| x.to_bits()).collect())
                .collect(),
            v.iter()
                .map(|l| l.iter().map(|x| x.to_bits()).collect())
                .collect(),
        )
    }

    fn gdn_bits(c: &KvCache) -> Vec<Option<(Vec<u32>, Vec<u32>)>> {
        c.gdn_snapshot()
            .iter()
            .map(|o| {
                o.as_ref().map(|(a, b)| {
                    (
                        a.iter().map(|x| x.to_bits()).collect(),
                        b.iter().map(|x| x.to_bits()).collect(),
                    )
                })
            })
            .collect()
    }

    #[test]
    fn roundtrip_is_bit_identical_including_gdn() {
        let dir = tmpdir("roundtrip");
        let prompt: Vec<u32> = vec![11, 22, 33, 44, 55];
        let src = filled_cache(prompt.len());
        let fp = 0xdead_beef_1234_5678u64;
        let path = save(&src, prompt.len(), &dir, fp, &prompt, 0).unwrap();
        assert!(path.exists(), "保存ファイルが無い");
        assert_eq!(
            path.file_name().unwrap().to_str().unwrap(),
            format!("{fp:016x}-{:016x}-5.kvsnap", prompt_hash(&prompt))
        );

        let mut dst = KvCache::new(4, 3, 32);
        dst.attach_gdn_slot(0, 2, 3);
        dst.attach_gdn_slot(2, 2, 3);
        load(&mut dst, &path, fp, &prompt).unwrap();
        assert_eq!(dst.len(), src.len());
        assert_eq!(kv_bits(&src, prompt.len()), kv_bits(&dst, prompt.len()));
        assert_eq!(gdn_bits(&src), gdn_bits(&dst), "GDN 状態がビット一致しない");
        // 復元後も続きから書ける
        dst.advance(1);
        assert_eq!(dst.len(), prompt.len() + 1);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_model_fingerprint_mismatch_and_leaves_cache_untouched() {
        let dir = tmpdir("fp_mismatch");
        let prompt = vec![1u32, 2, 3];
        let src = filled_cache(prompt.len());
        let path = save(&src, prompt.len(), &dir, 0x1111, &prompt, 0).unwrap();
        let mut dst = KvCache::new(4, 3, 32);
        dst.attach_gdn_slot(0, 2, 3);
        dst.attach_gdn_slot(2, 2, 3);
        let err = load(&mut dst, &path, 0x2222, &prompt).unwrap_err();
        match err {
            SnapshotError::ModelMismatch { expected, found } => {
                assert_eq!(expected, 0x2222);
                assert_eq!(found, 0x1111);
            }
            other => panic!("ModelMismatch を期待: {other}"),
        }
        assert_eq!(dst.len(), 0, "失敗時にキャッシュが書き換わった");
        assert_eq!(dst.capacity(), 0);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_prompt_mismatch() {
        let dir = tmpdir("prompt_mismatch");
        let src = filled_cache(3);
        let path = save(&src, 3, &dir, 7, &[1, 2, 3], 0).unwrap();
        let mut dst = KvCache::new(4, 3, 32);
        dst.attach_gdn_slot(0, 2, 3);
        dst.attach_gdn_slot(2, 2, 3);
        let err = load(&mut dst, &path, 7, &[1, 2, 4]).unwrap_err();
        assert!(matches!(err, SnapshotError::PromptMismatch { .. }), "{err}");
        assert_eq!(dst.len(), 0);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_corrupted_payload() {
        let dir = tmpdir("corrupt");
        let prompt = vec![5u32, 6, 7, 8];
        let path = save(&filled_cache(prompt.len()), prompt.len(), &dir, 7, &prompt, 0).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01; // ペイロードの 1 バイトだけ壊す
        fs::write(&path, &bytes).unwrap();
        let mut dst = KvCache::new(4, 3, 32);
        dst.attach_gdn_slot(0, 2, 3);
        dst.attach_gdn_slot(2, 2, 3);
        let err = load(&mut dst, &path, 7, &prompt).unwrap_err();
        assert!(matches!(err, SnapshotError::BadChecksum { .. }), "{err}");
        assert_eq!(dst.len(), 0);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_truncated_file_and_bad_magic_and_version() {
        let dir = tmpdir("truncated");
        let prompt = vec![9u32, 10];
        let path = save(&filled_cache(prompt.len()), prompt.len(), &dir, 7, &prompt, 0).unwrap();
        let bytes = fs::read(&path).unwrap();
        let mut dst = KvCache::new(4, 3, 32);
        dst.attach_gdn_slot(0, 2, 3);
        dst.attach_gdn_slot(2, 2, 3);

        // ヘッダ未満
        fs::write(&path, &bytes[..40]).unwrap();
        let err = load(&mut dst, &path, 7, &prompt).unwrap_err();
        assert!(matches!(err, SnapshotError::Truncated { .. }), "{err}");

        // ヘッダはあるがペイロードが短い
        fs::write(&path, &bytes[..bytes.len() - 4]).unwrap();
        let err = load(&mut dst, &path, 7, &prompt).unwrap_err();
        assert!(matches!(err, SnapshotError::LengthMismatch { .. }), "{err}");

        // magic
        let mut b = bytes.clone();
        b[0] = b'X';
        fs::write(&path, &b).unwrap();
        let err = load(&mut dst, &path, 7, &prompt).unwrap_err();
        assert!(matches!(err, SnapshotError::BadMagic(_)), "{err}");

        // version
        let mut b = bytes.clone();
        b[8..12].copy_from_slice(&(VERSION + 1).to_le_bytes());
        fs::write(&path, &b).unwrap();
        let err = load(&mut dst, &path, 7, &prompt).unwrap_err();
        assert!(matches!(err, SnapshotError::UnsupportedVersion(_)), "{err}");
        assert_eq!(dst.len(), 0);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_shape_mismatch_between_snapshot_and_cache() {
        let dir = tmpdir("shape");
        let prompt = vec![1u32, 2, 3];
        let path = save(&filled_cache(prompt.len()), prompt.len(), &dir, 7, &prompt, 0).unwrap();
        // GDN スロットの長さが違うキャッシュ
        let mut dst = KvCache::new(4, 3, 32);
        dst.attach_gdn_slot(0, 5, 3);
        dst.attach_gdn_slot(2, 2, 3);
        let err = load(&mut dst, &path, 7, &prompt).unwrap_err();
        assert!(matches!(err, SnapshotError::ShapeMismatch(_)), "{err}");
        assert_eq!(dst.len(), 0);

        // スロットの無いキャッシュ
        let mut plain = KvCache::new(4, 3, 32);
        let err = load(&mut plain, &path, 7, &prompt).unwrap_err();
        assert!(matches!(err, SnapshotError::ShapeMismatch(_)), "{err}");
        assert_eq!(plain.len(), 0);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn save_is_atomic_and_leaves_no_temp_files() {
        let dir = tmpdir("atomic");
        let prompt = vec![1u32, 2, 3, 4];
        let p1 = save(&filled_cache(4), 4, &dir, 7, &prompt, 0).unwrap();
        let p2 = save(&filled_cache(4), 4, &dir, 7, &prompt, 0).unwrap();
        assert_eq!(p1, p2, "同じキーは同じパス");
        let names: Vec<String> = fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names.len(), 1, "一時ファイルが残っている: {names:?}");
        assert!(p1.exists());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn quota_prunes_oldest_snapshots() {
        let dir = tmpdir("quota");
        let src = filled_cache(3);
        // 1 本のサイズを測って、3 本入らない上限を作る
        let one = save(&src, 3, &dir, 7, &[1, 2, 3], 0).unwrap();
        let size = fs::metadata(&one).unwrap().len();
        fs::remove_file(&one).unwrap();
        let quota = size + size / 2; // 1 本 + 半分 = 2 本目で古い方が消える
        for i in 0..3u32 {
            let prompt = vec![i, 100, 200];
            save(&src, 3, &dir, 7, &prompt, quota).unwrap();
            assert!(
                total_bytes(&dir).unwrap() <= quota,
                "上限を超えた: {} > {quota}",
                total_bytes(&dir).unwrap()
            );
        }
        let files = list_snapshots(&dir).unwrap();
        assert_eq!(files.len(), 1, "古いスナップショットが消えていない");
        assert!(files[0].0.to_string_lossy().contains(&format!(
            "{:016x}",
            prompt_hash(&[2, 100, 200])
        )));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn quota_exceeded_is_rejected_without_writing() {
        let dir = tmpdir("quota_exceeded");
        let prompt = vec![1u32, 2, 3];
        let err = save(&filled_cache(3), 3, &dir, 7, &prompt, 16).unwrap_err();
        assert!(matches!(err, SnapshotError::QuotaExceeded { .. }), "{err}");
        assert_eq!(total_bytes(&dir).unwrap(), 0, "上限超過なのに書いた");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn keys_separate_prompts_and_models() {
        let dir = Path::new("/tmp/kvsnap-keys");
        let a = snapshot_path_for(dir, 1, &[1, 2, 3]);
        let b = snapshot_path_for(dir, 2, &[1, 2, 3]);
        let c = snapshot_path_for(dir, 1, &[1, 2, 4]);
        let d = snapshot_path_for(dir, 1, &[1, 2, 3, 4]);
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
        assert_eq!(a, snapshot_path_for(dir, 1, &[1, 2, 3]));
    }

    #[test]
    fn env_values_are_parsed_and_default_off() {
        assert_eq!(dir_from_value(None), None);
        assert_eq!(dir_from_value(Some("")), None);
        assert_eq!(dir_from_value(Some("   ")), None);
        assert_eq!(
            dir_from_value(Some(" /tmp/kv ")),
            Some(PathBuf::from("/tmp/kv"))
        );
        assert_eq!(quota_mb_from_value(None), DEFAULT_QUOTA_MB);
        assert_eq!(quota_mb_from_value(Some("64")), 64);
        assert_eq!(quota_mb_from_value(Some("0")), 0);
        assert_eq!(quota_mb_from_value(Some("nope")), DEFAULT_QUOTA_MB);
    }

    #[test]
    fn encode_rejects_non_prompt_boundary() {
        let c = filled_cache(4);
        let e = encode(&c, 3, 7, &[1, 2, 3, 4]).unwrap_err();
        assert!(matches!(e, SnapshotError::ShapeMismatch(_)));
        let e = encode(&c, 5, 7, &[1, 2, 3, 4, 5]).unwrap_err();
        assert!(matches!(e, SnapshotError::ShapeMismatch(_)));
        let e = encode(&c, 0, 7, &[]).unwrap_err();
        assert!(matches!(e, SnapshotError::ShapeMismatch(_)));
    }

    #[test]
    fn header_layout_is_stable() {
        // 形式の互換性の番人: ヘッダ長とフィールド位置を固定する。
        let c = filled_cache(2);
        let b = encode(&c, 2, 0x0102_0304_0506_0708, &[9, 10]).unwrap();
        assert_eq!(HEADER_LEN, 72);
        assert_eq!(&b[0..8], b"DNKVSNAP");
        assert_eq!(u32::from_le_bytes(b[8..12].try_into().unwrap()), VERSION);
        assert_eq!(u32::from_le_bytes(b[12..16].try_into().unwrap()), 72);
        assert_eq!(
            u64::from_le_bytes(b[16..24].try_into().unwrap()),
            0x0102_0304_0506_0708
        );
        assert_eq!(
            u64::from_le_bytes(b[24..32].try_into().unwrap()),
            prompt_hash(&[9, 10])
        );
        assert_eq!(u64::from_le_bytes(b[32..40].try_into().unwrap()), 2);
        assert_eq!(u32::from_le_bytes(b[40..44].try_into().unwrap()), 4);
        assert_eq!(u32::from_le_bytes(b[44..48].try_into().unwrap()), 3);
        assert_eq!(u32::from_le_bytes(b[48..52].try_into().unwrap()), 2);
        assert_eq!(
            u64::from_le_bytes(b[56..64].try_into().unwrap()) as usize,
            b.len() - HEADER_LEN
        );
    }
}
