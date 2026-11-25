//! task 881: `denno doctor` — 環境とハードウェアの自己診断。
//!
//! 見るもの: CPU のアーキ/コア数/実行時 SIMD、システム RAM、CUDA driver
//! (libcuda.so.1 と /proc/driver/nvidia)、Intel Level Zero loader の**存在**、
//! `--dir` を渡されたときの分割 GGUF の欠け検出。最後に `denno serve-flash` の
//! 推奨フラグを出す。
//!
//! **GPU を実行しない**。ライブラリのファイル存在と `/proc` の読み取りだけで、
//! CUDA API の初期化も `nvidia-smi` の起動もしない (dev コンテナでも安全)。
//! そのため VRAM の実容量は出さない (取れないものを捏造しない)。
//!
//! 数値経路には触れない。推論の既定挙動は不変。

use anyhow::Result;
use std::path::{Path, PathBuf};

/// 共有ライブラリの探索先。`LD_LIBRARY_PATH` を先に、次に固定候補。
const LIB_DIRS: [&str; 6] = [
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib/aarch64-linux-gnu",
    "/usr/lib64",
    "/usr/lib",
    "/usr/local/lib",
    "/usr/local/cuda/lib64",
];

/// 分割 GGUF のシャード 1 本。
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ShardEntry {
    pub name: String,
    pub bytes: u64,
}

/// `--dir` の検査結果。dir 未指定なら `dir = None` で他は既定値。
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ModelReport {
    pub dir: Option<String>,
    pub shards: Vec<ShardEntry>,
    /// 分割命名 (`-NNNNN-of-MMMMM.gguf`) から読んだ期待シャード数。
    /// 分割命名でない単一 GGUF なら `None` (欠けは判定しない)。
    pub expected_shards: Option<u32>,
    /// 期待されるが存在しないシャードの連番 (`"00002"` 形式)。
    pub missing_shards: Vec<String>,
    /// サイズ 0 のファイル (切り詰め/中断 DL の疑い)。
    pub zero_byte_shards: Vec<String>,
    pub total_bytes: u64,
    /// 欠けも 0 バイトも無く、少なくとも 1 本ある。
    pub ok: bool,
}

/// ハードウェア/ライブラリの検出結果。すべて「読み取れた事実」だけで、
/// 取れなかった項目は `None` のまま置く。
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct HwReport {
    pub arch: String,
    pub logical_cores: usize,
    pub physical_cores: Option<usize>,
    /// 実行時に有効な SIMD。aarch64 はビルド時 target_feature のみ見る。
    pub simd: Vec<String>,
    pub mem_total_bytes: Option<u64>,
    pub mem_available_bytes: Option<u64>,
    /// 見つかった `libcuda.so.1` のフルパス。
    pub cuda_driver: Option<String>,
    /// `/proc/driver/nvidia/version` の NVRM 行 (あれば)。
    pub nvidia_kernel_driver: Option<String>,
    /// `/proc/driver/nvidia/gpus/*/information` の Model 行。
    pub gpu_devices: Vec<String>,
    /// 見つかった `libze_loader.so.1` のフルパス。
    pub level_zero_loader: Option<String>,
    /// 見つかった Intel GPU ドライバ `libze_intel_gpu.so.1` のフルパス。
    pub level_zero_intel_driver: Option<String>,
}

/// 全体。JSON 出力はこれをそのまま `serde_json` に渡す。
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Report {
    pub cpu: CpuReport,
    pub memory: MemReport,
    pub gpu: GpuReport,
    pub model: ModelReport,
    pub recommendations: Vec<String>,
}

/// `cpu` セクション (HwReport の CPU 部分を素直に分けたもの)。
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CpuReport {
    pub arch: String,
    pub logical_cores: usize,
    pub physical_cores: Option<usize>,
    pub simd: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MemReport {
    pub total_bytes: Option<u64>,
    pub available_bytes: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GpuReport {
    pub cuda_driver: Option<String>,
    pub nvidia_kernel_driver: Option<String>,
    pub gpu_devices: Vec<String>,
    pub level_zero_loader: Option<String>,
    pub level_zero_intel_driver: Option<String>,
}

// ---------------------------------------------------------------------------
// 検出
// ---------------------------------------------------------------------------

/// x86_64: 実行時に CPUID で判定する (`is_x86_feature_detected!`)。
#[cfg(target_arch = "x86_64")]
fn simd_features() -> Vec<String> {
    let mut v: Vec<String> = Vec::new();
    // 注意: `is_x86_feature_detected!` は文字列リテラルを直接要求する
    // (別マクロの `$name` 経由だと "unknown x86 target feature" で落ちる)。
    // ので、ここは展開せずに 1 行ずつ書く。
    if is_x86_feature_detected!("sse4.2") {
        v.push("sse4.2".to_string());
    }
    if is_x86_feature_detected!("avx") {
        v.push("avx".to_string());
    }
    if is_x86_feature_detected!("avx2") {
        v.push("avx2".to_string());
    }
    if is_x86_feature_detected!("fma") {
        v.push("fma".to_string());
    }
    if is_x86_feature_detected!("avx512f") {
        v.push("avx512f".to_string());
    }
    if is_x86_feature_detected!("avx512bw") {
        v.push("avx512bw".to_string());
    }
    if is_x86_feature_detected!("avx512vnni") {
        v.push("avx512vnni".to_string());
    }
    v
}

/// x86_64 以外: `cfg!(target_feature)` はビルド時判定しかできない (実行時
/// プローブの標準手段が無い) ので、その旨を値の側で区別できるようにする。
#[cfg(not(target_arch = "x86_64"))]
fn simd_features() -> Vec<String> {
    let mut v: Vec<String> = Vec::new();
    if cfg!(target_feature = "neon") {
        v.push("neon (build-time)".to_string());
    }
    v
}

/// `/proc/cpuinfo` の (physical id, core id) の組の数 = 物理コア数。
/// 仮想化で physical id が無い環境では `None`。
fn physical_cores() -> Option<usize> {
    let s = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    let mut phys: Option<String> = None;
    let mut set = std::collections::BTreeSet::new();
    for line in s.lines() {
        if let Some((k, v)) = line.split_once(':') {
            let k = k.trim();
            let v = v.trim();
            match k {
                "physical id" => phys = Some(v.to_string()),
                "core id" => {
                    let p = phys.clone().unwrap_or_else(|| "0".to_string());
                    set.insert((p, v.to_string()));
                }
                _ => {}
            }
        }
    }
    if set.is_empty() {
        None
    } else {
        Some(set.len())
    }
}

/// `/proc/meminfo` の指定キー (kB 単位) をバイトで返す。
fn meminfo_bytes(key: &str) -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    let prefix = format!("{key}:");
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix(&prefix) {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb.saturating_mul(1024));
        }
    }
    None
}

/// `LD_LIBRARY_PATH` と固定候補から `<name>` を探す (最初に見つかったパス)。
fn find_lib(name: &str) -> Option<String> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Ok(ld) = std::env::var("LD_LIBRARY_PATH") {
        for d in ld.split(':').filter(|d| !d.is_empty()) {
            dirs.push(PathBuf::from(d));
        }
    }
    dirs.extend(LIB_DIRS.iter().map(PathBuf::from));
    for d in dirs {
        let p = d.join(name);
        if p.is_file() {
            return Some(p.display().to_string());
        }
    }
    None
}

/// `/proc/driver/nvidia/version` の先頭行 (NVRM version)。NVIDIA カーネル
/// ドライバがロードされている証拠で、ユーザー空間の libcuda とは独立。
fn nvidia_kernel_driver() -> Option<String> {
    let s = std::fs::read_to_string("/proc/driver/nvidia/version").ok()?;
    let line = s.lines().next()?.trim();
    if line.is_empty() {
        None
    } else {
        Some(line.to_string())
    }
}

/// `/proc/driver/nvidia/gpus/<busid>/information` の "Model:" を集める。
/// デバイス名だけで、VRAM は載っていない (そこは実行時クエリが要る)。
fn nvidia_gpu_models() -> Vec<String> {
    let root = Path::new("/proc/driver/nvidia/gpus");
    let rd = match std::fs::read_dir(root) {
        Ok(rd) => rd,
        Err(_) => return Vec::new(),
    };
    let mut out: Vec<String> = Vec::new();
    for e in rd.flatten() {
        let info = e.path().join("information");
        let s = match std::fs::read_to_string(&info) {
            Ok(s) => s,
            Err(_) => continue,
        };
        for line in s.lines() {
            if let Some(rest) = line.trim().strip_prefix("Model:") {
                let m = rest.trim();
                if !m.is_empty() && !out.iter().any(|x| x == m) {
                    out.push(m.to_string());
                }
            }
        }
    }
    out.sort();
    out
}

fn detect_hw() -> HwReport {
    HwReport {
        arch: std::env::consts::ARCH.to_string(),
        logical_cores: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        physical_cores: physical_cores(),
        simd: simd_features(),
        mem_total_bytes: meminfo_bytes("MemTotal"),
        mem_available_bytes: meminfo_bytes("MemAvailable"),
        cuda_driver: find_lib("libcuda.so.1"),
        nvidia_kernel_driver: nvidia_kernel_driver(),
        gpu_devices: nvidia_gpu_models(),
        level_zero_loader: find_lib("libze_loader.so.1"),
        level_zero_intel_driver: find_lib("libze_intel_gpu.so.1"),
    }
}

/// `name` が `<prefix>-NNNNN-of-MMMMM.gguf` なら `(index, total)`。
/// llama.cpp / HF の split GGUF 命名。数字は 0 始まりでも 1 始まりでもない
/// (1 始まり) ので、`idx == 0` や `idx > total` は命名として不正 = `None`。
fn parse_split_name(name: &str) -> Option<(u32, u32)> {
    let stem = name.strip_suffix(".gguf")?;
    let (left, total_s) = stem.rsplit_once("-of-")?;
    let (_, idx_s) = left.rsplit_once('-')?;
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    if !digits(idx_s) || !digits(total_s) {
        return None;
    }
    let idx: u32 = idx_s.parse().ok()?;
    let total: u32 = total_s.parse().ok()?;
    if total == 0 || idx == 0 || idx > total {
        return None;
    }
    Some((idx, total))
}

/// `--dir` の中の `.gguf` を見て、分割命名なら欠けと 0 バイトを検出する。
fn check_shards(dir: &Path) -> Result<ModelReport> {
    let rd = std::fs::read_dir(dir)
        .map_err(|e| anyhow::anyhow!("{} を読めない: {e}", dir.display()))?;
    let mut shards: Vec<ShardEntry> = Vec::new();
    for e in rd.flatten() {
        let p = e.path();
        if p.extension().map(|x| x == "gguf").unwrap_or(false) {
            let name = p
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            let bytes = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
            shards.push(ShardEntry { name, bytes });
        }
    }
    shards.sort_by(|a, b| a.name.cmp(&b.name));

    let mut expected: Option<u32> = None;
    let mut present: Vec<u32> = Vec::new();
    for s in &shards {
        if let Some((idx, total)) = parse_split_name(&s.name) {
            // 複数の分割系列が混ざるディレクトリは想定外。最大の total を採り、
            // 食い違ったら期待数を増やす側に倒さない (誤検出より過小報告)。
            expected = Some(expected.map_or(total, |e| e.min(total)));
            present.push(idx);
        }
    }
    present.sort_unstable();

    let missing_shards = match expected {
        Some(total) => (1..=total)
            .filter(|i| !present.contains(i))
            .map(|i| format!("{i:05}"))
            .collect(),
        None => Vec::new(),
    };
    let zero_byte_shards: Vec<String> = shards
        .iter()
        .filter(|s| s.bytes == 0)
        .map(|s| s.name.clone())
        .collect();
    let total_bytes: u64 = shards.iter().map(|s| s.bytes).sum();
    let ok = !shards.is_empty() && missing_shards.is_empty() && zero_byte_shards.is_empty();

    Ok(ModelReport {
        dir: Some(dir.display().to_string()),
        shards,
        expected_shards: expected,
        missing_shards,
        zero_byte_shards,
        total_bytes,
        ok,
    })
}

// ---------------------------------------------------------------------------
// 推奨 (実在するフラグだけ書く)
// ---------------------------------------------------------------------------

fn recommend(report: &Report) -> Vec<String> {
    let mut v: Vec<String> = Vec::new();
    let dir = report
        .model
        .dir
        .clone()
        .unwrap_or_else(|| "<モデルのディレクトリ>".to_string());

    if report.gpu.cuda_driver.is_some() {
        v.push(format!(
            "GPU (CUDA): denno serve-flash --dir {dir} --max-ctx 8192 --max-tokens 512"
        ));
        v.push(
            "expert を VRAM へ寄せる: DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=<MiB> を付ける"
                .to_string(),
        );
    } else if report.gpu.nvidia_kernel_driver.is_some() {
        v.push(
            "NVIDIA カーネルドライバは居るが libcuda.so.1 が見つからない: コンテナなら \
             --gpus all と NVIDIA runtime、ホストなら libcuda の入ったドライバを入れる"
                .to_string(),
        );
    } else {
        v.push(format!(
            "CPU のみ: denno serve-flash --dir {dir} --max-ctx 4096 --max-tokens 256"
        ));
        v.push(
            "CPU の expert をページキャッシュから RAM に常駐: DENNO_EXPERT_CACHE_MB=<MiB>"
                .to_string(),
        );
    }

    if let Some(lo) = &report.gpu.level_zero_loader {
        // Level Zero はここで検出するだけ。このビルドに Level Zero 経路は無いので
        // 対応フラグを案内しない (存在しないフラグを勧めない)。
        v.push(format!(
            "Intel Level Zero loader を検出 ({lo}) が、このビルドに Level Zero 経路は無い"
        ));
    }

    let has_avx2 = report.cpu.simd.iter().any(|s| s == "avx2");
    let has_fma = report.cpu.simd.iter().any(|s| s == "fma");
    if report.cpu.arch == "x86_64" && !(has_avx2 && has_fma) {
        v.push(
            "AVX2/FMA が無い: Q4_K expert の GEMV はスカラーへ落ちる (CPU 推論は遅い)"
                .to_string(),
        );
    }

    let n = report.cpu.logical_cores;
    if n >= 1 && n <= 4 {
        v.push(format!(
            "論理コア {n} 本: RAYON_NUM_THREADS を明示しないと既定 (論理-2) で 2 本になる"
        ));
    }

    if let Some(avail) = report.memory.available_bytes {
        let gibs = avail as f64 / (1024.0 * 1024.0 * 1024.0);
        if gibs < 16.0 {
            v.push(format!(
                "空き RAM {gibs:.1} GiB: 27B 級の Q4 は載らない。小さいモデルか --max-ctx を下げる"
            ));
        }
    }

    match &report.model.dir {
        Some(_) => {
            if !report.model.ok {
                if !report.model.missing_shards.is_empty() {
                    v.push(format!(
                        "モデルのシャードが欠けている: {} (再ダウンロードが要る)",
                        report.model.missing_shards.join(", ")
                    ));
                }
                if !report.model.zero_byte_shards.is_empty() {
                    v.push(format!(
                        "0 バイトのシャード: {}",
                        report.model.zero_byte_shards.join(", ")
                    ));
                }
                if report.model.shards.is_empty() {
                    v.push("--dir に .gguf が 1 本も無い".to_string());
                }
            }
        }
        None => v.push(
            "モデル検査は未実行: `denno doctor --dir <モデルディレクトリ>` で GGUF の欠けを見る"
                .to_string(),
        ),
    }

    v
}

// ---------------------------------------------------------------------------
// 出力
// ---------------------------------------------------------------------------

fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
}

fn opt_or_dash<T: ToString>(v: &Option<T>) -> String {
    match v {
        Some(x) => x.to_string(),
        None => "(無し)".to_string(),
    }
}

fn render_text(r: &Report) -> String {
    let mut o = String::new();
    o.push_str("denno doctor — 環境診断\n");

    o.push_str("[CPU]\n");
    o.push_str(&format!("  arch            {}\n", r.cpu.arch));
    o.push_str(&format!("  論理コア        {}\n", r.cpu.logical_cores));
    o.push_str(&format!(
        "  物理コア        {}\n",
        opt_or_dash(&r.cpu.physical_cores)
    ));
    let simd = if r.cpu.simd.is_empty() {
        "(検出なし)".to_string()
    } else {
        r.cpu.simd.join(", ")
    };
    o.push_str(&format!("  SIMD (実行時)   {simd}\n"));

    o.push_str("[メモリ]\n");
    o.push_str(&format!(
        "  搭載            {}\n",
        r.memory.total_bytes.map(gib).unwrap_or_else(|| "(不明)".into())
    ));
    o.push_str(&format!(
        "  空き (available) {}\n",
        r.memory
            .available_bytes
            .map(gib)
            .unwrap_or_else(|| "(不明)".into())
    ));

    o.push_str("[GPU]\n");
    o.push_str(&format!(
        "  CUDA driver     {}\n",
        opt_or_dash(&r.gpu.cuda_driver)
    ));
    o.push_str(&format!(
        "  NVIDIA kernel   {}\n",
        opt_or_dash(&r.gpu.nvidia_kernel_driver)
    ));
    o.push_str(&format!(
        "  デバイス        {}\n",
        if r.gpu.gpu_devices.is_empty() {
            "(無し)".to_string()
        } else {
            r.gpu.gpu_devices.join(", ")
        }
    ));
    o.push_str(&format!(
        "  Level Zero      {}\n",
        opt_or_dash(&r.gpu.level_zero_loader)
    ));
    if r.gpu.level_zero_intel_driver.is_some() {
        o.push_str(&format!(
            "  Intel L0 driver {}\n",
            opt_or_dash(&r.gpu.level_zero_intel_driver)
        ));
    }

    o.push_str("[モデル]\n");
    match &r.model.dir {
        None => o.push_str("  dir             (未指定: --dir で検査する)\n"),
        Some(d) => {
            o.push_str(&format!("  dir             {d}\n"));
            o.push_str(&format!(
                "  シャード        {}/{} 本\n",
                r.model.shards.len(),
                r.model
                    .expected_shards
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "?".into())
            ));
            o.push_str(&format!("  合計サイズ      {}\n", gib(r.model.total_bytes)));
            if !r.model.missing_shards.is_empty() {
                o.push_str(&format!(
                    "  欠け            {}\n",
                    r.model.missing_shards.join(", ")
                ));
            }
            if !r.model.zero_byte_shards.is_empty() {
                o.push_str(&format!(
                    "  0 バイト        {}\n",
                    r.model.zero_byte_shards.join(", ")
                ));
            }
            o.push_str(&format!(
                "  状態            {}\n",
                if r.model.ok { "OK" } else { "NG" }
            ));
        }
    }

    o.push_str("[推奨]\n");
    if r.recommendations.is_empty() {
        o.push_str("  (無し)\n");
    }
    for line in &r.recommendations {
        o.push_str(&format!("  - {line}\n"));
    }
    o
}

/// `denno doctor` の本体。`Err` を返すのは「モデル検査が NG」のときだけで、
/// その場合もレポートは先に出力する (JSON は自動化が読める状態で残す)。
pub fn run(dir: Option<&Path>, json: bool) -> Result<()> {
    let hw = detect_hw();
    let model = match dir {
        Some(d) => check_shards(d)?,
        None => ModelReport::default(),
    };
    let mut report = Report {
        cpu: CpuReport {
            arch: hw.arch.clone(),
            logical_cores: hw.logical_cores,
            physical_cores: hw.physical_cores,
            simd: hw.simd.clone(),
        },
        memory: MemReport {
            total_bytes: hw.mem_total_bytes,
            available_bytes: hw.mem_available_bytes,
        },
        gpu: GpuReport {
            cuda_driver: hw.cuda_driver.clone(),
            nvidia_kernel_driver: hw.nvidia_kernel_driver.clone(),
            gpu_devices: hw.gpu_devices.clone(),
            level_zero_loader: hw.level_zero_loader.clone(),
            level_zero_intel_driver: hw.level_zero_intel_driver.clone(),
        },
        model,
        recommendations: Vec::new(),
    };
    report.recommendations = recommend(&report);

    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print!("{}", render_text(&report));
    }

    if report.model.dir.is_some() && !report.model.ok {
        anyhow::bail!(
            "モデルディレクトリの検査が NG (欠け {} / 0 バイト {})",
            report.model.missing_shards.len(),
            report.model.zero_byte_shards.len()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(tag: &str) -> Self {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let path = std::env::temp_dir().join(format!(
                "denno_doctor_{}_{}_{}",
                tag,
                std::process::id(),
                nanos
            ));
            std::fs::create_dir_all(&path).expect("create temp dir");
            TempDir { path }
        }

        fn write(&self, name: &str, bytes: &[u8]) -> PathBuf {
            let p = self.path.join(name);
            let mut f = std::fs::File::create(&p).expect("create file");
            f.write_all(bytes).expect("write file");
            p
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn synth(cuda: bool, avx2: bool, level_zero: bool) -> Report {
        let cpu = CpuReport {
            arch: "x86_64".to_string(),
            logical_cores: 16,
            physical_cores: Some(8),
            simd: if avx2 {
                vec!["sse4.2".into(), "avx".into(), "avx2".into(), "fma".into()]
            } else {
                vec!["sse4.2".into()]
            },
        };
        let gpu = GpuReport {
            cuda_driver: cuda.then(|| "/usr/lib/x86_64-linux-gnu/libcuda.so.1".to_string()),
            nvidia_kernel_driver: cuda.then(|| "NVRM version: fake".to_string()),
            gpu_devices: if cuda {
                vec!["NVIDIA GeForce RTX 5090".into()]
            } else {
                Vec::new()
            },
            level_zero_loader: level_zero.then(|| "/usr/lib/x86_64-linux-gnu/libze_loader.so.1".into()),
            level_zero_intel_driver: None,
        };
        let mut r = Report {
            cpu,
            memory: MemReport {
                total_bytes: Some(59 * 1024 * 1024 * 1024),
                available_bytes: Some(50 * 1024 * 1024 * 1024),
            },
            gpu,
            model: ModelReport::default(),
            recommendations: Vec::new(),
        };
        r.recommendations = recommend(&r);
        r
    }

    #[test]
    fn parse_split_name_variants() {
        assert_eq!(parse_split_name("m-00001-of-00003.gguf"), Some((1, 3)));
        assert_eq!(parse_split_name("flash-next-UD-IQ3_XXS-00012-of-00014.gguf"), Some((12, 14)));
        // 分割命名でない
        assert_eq!(parse_split_name("model.gguf"), None);
        // 不正 (0 始まり / 範囲外 / 非数字)
        assert_eq!(parse_split_name("m-00000-of-00003.gguf"), None);
        assert_eq!(parse_split_name("m-00004-of-00003.gguf"), None);
        assert_eq!(parse_split_name("m-x-of-00003.gguf"), None);
        assert_eq!(parse_split_name("m-00001-of-00000.gguf"), None);
    }

    // 欠け検出: 00002 が無い 3 分割
    #[test]
    fn missing_shard_detected() {
        let td = TempDir::new("missing");
        td.write("m-00001-of-00003.gguf", b"aaa");
        td.write("m-00003-of-00003.gguf", b"ccc");
        let r = check_shards(&td.path).expect("dir を読める");
        assert_eq!(r.expected_shards, Some(3));
        assert_eq!(r.missing_shards, vec!["00002".to_string()]);
        assert_eq!(r.shards.len(), 2);
        assert_eq!(r.total_bytes, 6);
        assert!(!r.ok);
        assert!(r.zero_byte_shards.is_empty());
    }

    // 全シャード揃い
    #[test]
    fn complete_shards_ok() {
        let td = TempDir::new("complete");
        td.write("m-00001-of-00002.gguf", b"abc");
        td.write("m-00002-of-00002.gguf", b"de");
        let r = check_shards(&td.path).expect("dir を読める");
        assert_eq!(r.expected_shards, Some(2));
        assert!(r.missing_shards.is_empty());
        assert_eq!(r.total_bytes, 5);
        assert!(r.ok);
    }

    // 分割命名でない単一 GGUF は欠け判定をしない (期待数は None)
    #[test]
    fn unsplit_gguf_has_no_expectation() {
        let td = TempDir::new("unsplit");
        td.write("solo.gguf", b"xyz");
        let r = check_shards(&td.path).expect("dir を読める");
        assert_eq!(r.expected_shards, None);
        assert!(r.missing_shards.is_empty());
        assert!(r.ok);
    }

    // 0 バイトは中断 DL として NG
    #[test]
    fn zero_byte_shard_is_ng() {
        let td = TempDir::new("zero");
        td.write("m-00001-of-00002.gguf", b"");
        td.write("m-00002-of-00002.gguf", b"zz");
        let r = check_shards(&td.path).expect("dir を読める");
        assert_eq!(r.zero_byte_shards, vec!["m-00001-of-00002.gguf".to_string()]);
        assert!(!r.ok);
    }

    // .gguf が 1 本も無い
    #[test]
    fn empty_dir_is_ng() {
        let td = TempDir::new("empty");
        let r = check_shards(&td.path).expect("dir を読める");
        assert!(r.shards.is_empty());
        assert!(!r.ok);
    }

    // テキスト整形: セクションが出る / 欠けが本文に載る
    #[test]
    fn text_report_sections_and_status() {
        let td = TempDir::new("text");
        td.write("m-00001-of-00002.gguf", b"a");
        let model = check_shards(&td.path).expect("dir を読める");
        let mut r = synth(true, true, false);
        r.model = model;
        r.recommendations = recommend(&r);
        let t = render_text(&r);
        for sec in ["[CPU]", "[メモリ]", "[GPU]", "[モデル]", "[推奨]"] {
            assert!(t.contains(sec), "missing {sec} in:\n{t}");
        }
        assert!(t.contains("状態            NG"), "unexpected:\n{t}");
        assert!(t.contains("00002"), "missing shard not shown:\n{t}");
        assert!(t.contains("denno serve-flash --dir"));
    }

    // JSON: 主要キーが揃い、round-trip で値が保たれる
    #[test]
    fn json_report_has_fields_and_roundtrips() {
        let mut r = synth(false, true, false);
        r.model = ModelReport {
            dir: Some("/nonexistent/model".into()),
            expected_shards: Some(2),
            missing_shards: vec!["00002".into()],
            shards: vec![ShardEntry { name: "m-00001-of-00002.gguf".into(), bytes: 10 }],
            zero_byte_shards: vec![],
            total_bytes: 10,
            ok: false,
        };
        let s = serde_json::to_string(&r).expect("serialize");
        let v: serde_json::Value = serde_json::from_str(&s).expect("parse");
        assert!(v["cpu"]["simd"].is_array());
        assert_eq!(v["cpu"]["logical_cores"], 16);
        assert_eq!(v["gpu"]["cuda_driver"], serde_json::Value::Null);
        assert_eq!(v["model"]["missing_shards"][0], "00002");
        assert_eq!(v["model"]["ok"], false);
        assert!(v["recommendations"].as_array().map(|a| !a.is_empty()).unwrap_or(false));
        let back: Report = serde_json::from_str(&s).expect("roundtrip");
        assert_eq!(back, r);
    }

    // 推奨: CUDA 有り → GPU 行が出て expert の env を案内する
    #[test]
    fn recommend_cuda_path() {
        let r = synth(true, true, false);
        let joined = r.recommendations.join("\n");
        assert!(joined.contains("GPU (CUDA)"), "{joined}");
        assert!(joined.contains("DENNO_EXPERT_IQ_GPU=1"), "{joined}");
        assert!(!joined.contains("CPU のみ"), "{joined}");
    }

    // 推奨: CUDA 無し → CPU 行。AVX2 無しなら注意書きが増える
    #[test]
    fn recommend_cpu_path_and_avx2_warning() {
        let r = synth(false, true, false);
        let joined = r.recommendations.join("\n");
        assert!(joined.contains("CPU のみ"), "{joined}");
        assert!(!joined.contains("AVX2/FMA が無い"), "{joined}");

        let r2 = synth(false, false, false);
        let joined2 = r2.recommendations.join("\n");
        assert!(joined2.contains("AVX2/FMA が無い"), "{joined2}");
    }

    // Level Zero は「検出した」とだけ言い、存在しないフラグを勧めない
    #[test]
    fn level_zero_is_informational_only() {
        let r = synth(false, true, true);
        let l0: Vec<&String> = r
            .recommendations
            .iter()
            .filter(|s| s.contains("Level Zero"))
            .collect();
        assert_eq!(l0.len(), 1, "{:?}", r.recommendations);
        assert!(l0[0].contains("このビルドに Level Zero 経路は無い"));
        // 推奨行に --backend のような実在しないフラグを出していない
        for s in &r.recommendations {
            assert!(!s.contains("--backend"), "invented flag: {s}");
            assert!(!s.contains("--level-zero"), "invented flag: {s}");
        }
    }

    // dir 未指定ならモデル行は「未指定」で、NG にならない
    #[test]
    fn no_dir_means_no_model_failure() {
        let r = synth(true, true, false);
        let t = render_text(&r);
        assert!(t.contains("(未指定: --dir で検査する)"), "{t}");
        assert!(r.model.ok == false); // default は ok=false だが dir が None なので run() は Err にしない
        assert!(r
            .recommendations
            .iter()
            .any(|s| s.contains("`denno doctor --dir")));
    }

    // 実環境での検出が panic しないこと (値の妥当性だけ見る)
    #[test]
    fn detect_hw_is_sane() {
        let hw = detect_hw();
        assert!(!hw.arch.is_empty());
        assert!(hw.logical_cores >= 1);
        if let (Some(t), Some(a)) = (hw.mem_total_bytes, hw.mem_available_bytes) {
            assert!(a <= t, "available {a} > total {t}");
        }
        // x86_64 なら SIMD は最低 1 つ (SSE2 は必ずある) を期待するが、
        // 実行環境依存なので「空でも panic しない」だけを確認する。
    }
}
