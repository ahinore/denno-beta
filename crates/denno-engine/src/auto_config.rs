//! task 818: カードの VRAM とホスト RAM から `DENNO_*` の既定を選ぶ
//! (`--vram-budget auto` / `DENNO_AUTO_CONFIG=1`)。
//!
//! # なぜ要るか
//!
//! Flash-Next を速く動かすには 4 本の env が要る。
//! `DENNO_EXPERT_GPU=1` / `DENNO_EXPERT_IQ_GPU=1` /
//! `DENNO_EXPERT_GPU_MIN_HIT=0` / `DENNO_EXPERT_VRAM_MB` /
//! `DENNO_GPU_WEIGHT_MB`。しかし適切な MiB はカードで全く違う
//! (10 GB の RTX 3080 と 32 GB の RTX 5090 で同じ値を書くと、前者は
//! 確保に失敗して CPU へ落ち、後者は VRAM を余らせる)。
//!
//! ここは「カードの総 VRAM・ホスト RAM・モデルのバイト数・量子型」から
//! その MiB を**式で**決める純関数 [`plan_gpu_config`] と、その適用
//! ([`apply_env_pairs`]) を置く。数値経路・カーネル・丸め順には触らない
//! (変えるのは「どの重みを VRAM に載せるか」の予算だけ)。
//!
//! # 不変条件
//!
//! - **明示の env は常に勝つ**。[`apply_env_pairs`] は「未設定の変数だけ」を
//!   書き、既に値がある変数は 1 つも触らない (戻り値で「採用しなかった」を返す)。
//!   したがって `DENNO_AUTO_CONFIG=1` を付けても、手で書いた
//!   `DENNO_EXPERT_VRAM_MB` はそのまま効く (100% 後方互換)。
//! - 予算は **上限** であって予約ではない。`DENNO_EXPERT_VRAM_MB` と
//!   `DENNO_GPU_WEIGHT_MB` の和が VRAM を超えるのは既定の流儀で、実効は
//!   `mem_audit::card_expert_cap_from_free` が「空き − 余裕」で丸める
//!   (手動の定番 `22000 + 12000` も 32 GB で和が 34,000 MiB になる)。
//! - 32 GB (RTX 5090) では手動の定番と同じ値になること
//!   (`expert_vram=22000` / `gpu_weight=12000` / `min_hit=0`) をテストで縛る。
//!   これが「auto が手動ピークと 1% 以内」の前提になる。
//! - 8 / 10 / 12 / 16 / 24 / 32 GB × Q3 / Q4 の全組み合わせで
//!   「VRAM を使い切らず、かつ 0 に潰さない」ことをテストで縛る。
//! - 段の判定は**公称 VRAM の実測ずれ** ([`VRAM_TIER_SLACK_MB`]) を含める
//!   (task 886)。12 GB の RTX 4070 は 12,282 MiB、16 GB の RTX 4080 は
//!   16,376 MiB を報告するので、公称値の `>=` で切ると 12 GB が 10 GB 扱いに
//!   なって P45 ゲートが 0.5 のまま残る (thrash floor)。

/// `DENNO_AUTO_CONFIG`: auto 計画を有効にする env。`1` / `true` / `on` /
/// `auto` が真。`--vram-budget auto` と同じ意味 (CLI が優先)。
pub const AUTO_ENV: &str = "DENNO_AUTO_CONFIG";

/// CUDA context + JIT + 断片の余裕 (MiB)。`mem_audit::VRAM_RESERVE_MB_DEFAULT`
/// と同じ値。ここでは「計画の分母」として使い、実装側にも
/// `DENNO_VRAM_RESERVE_MB` として同じ値を渡す (食い違うと計画が嘘になる)。
pub const PLAN_RESERVE_MB: u64 = 1024;

/// 計画に使える VRAM に対する dense 重み上限の割合 (%)。
///
/// 32 GB 級で 12,000 MiB ([`WEIGHT_CAP_MB`]) に当たる係数。RTX 5090 の
/// 総 VRAM は 32,110 MiB (実測 lab t818b) なので、38% では 11,812 MiB に
/// なって手動ピークより 1.6% 低い。40% なら絶対上限に当たり、手動と一致する。
/// 実測 (lab の `[flash] vram budget`) では UD-Q4_K_XL の dense 側は
/// 56,794 MiB (499 本) あり、32 GB には収まらない。つまりこの上限は
/// 「全部載せる」ためではなく、**decode で毎トークン触る層を残し、
/// 残りを CPU/mmapp に落とす**ための配分。
pub const WEIGHT_PCT: u64 = 40;

/// dense 重み上限の絶対上限 (MiB)。32 GB の手動定番 `DENNO_GPU_WEIGHT_MB=12000`。
pub const WEIGHT_CAP_MB: u64 = 12_000;

/// 計画に使える VRAM に対する expert 常駐上限の割合 (%)。
///
/// 32 GB 級で [`EXPERT_CAP_MB`] に当たる係数 (5090 の 32,110 MiB - 1,024 の
/// 72% = 22,381 MiB)。
pub const EXPERT_PCT: u64 = 72;

/// expert 常駐上限の絶対上限 (MiB)。32 GB の手動定番
/// `DENNO_EXPERT_VRAM_MB=22000`。
pub const EXPERT_CAP_MB: u64 = 22_000;

/// expert に回す下限 (MiB)。これ未満なら expert 経路を切る。
pub const MIN_EXPERT_MB: u64 = 1024;

/// `DENNO_EXPERT_GPU_MIN_HIT=0` を入れる下限 (MiB)。これ未満は P45 ゲートの
/// 既定 (0.5) を残す = ヒット率が低いカードで VRAM を往復させない。
pub const MIN_HIT_ZERO_FROM_MB: u64 = 12 * 1024;

/// 公称 VRAM と実測のずれの許容 (MiB)。task 886。
///
/// 12 GB 級は公称 12,288 MiB でも実測は少し小さい (RTX 4070 = 12,282、
/// 16 GB の RTX 4080 = 16,376、5070 Ti = 16,303)。閾値を公称値の `>=` で
/// 切ると 4070 は 12 GB 未満と判定され、P45 ゲートが既定 0.5 のまま残って
/// **CPU/GPU を往復する thrash floor** に落ちる (task 794a の 2 割の揺れと
/// 同じ機構)。段の判定は「公称 - 256 MiB 以上」= 実測が公称より 256 MiB
/// 小さくても同じ段として扱う。
pub const VRAM_TIER_SLACK_MB: u64 = 256;

/// `vram_mb` が公称 `nominal_mb` の段以上か ([`VRAM_TIER_SLACK_MB`] 込み)。
pub fn tier_at_least(vram_mb: u64, nominal_mb: u64) -> bool {
    vram_mb.saturating_add(VRAM_TIER_SLACK_MB) >= nominal_mb
}

/// ホスト RAM tier (`DENNO_EXPERT_RAM_MB=auto`) を入れる VRAM の上限 (MiB)。
///
/// 24 GB 以上では入れない: lab で「手動ピーク」として使っている COND は
/// RAM tier を設定していないので、auto が勝手に on にすると同じ土俵で
/// 比べられなくなる (`REPORT_P818.md`)。
pub const RAM_TIER_MAX_VRAM_MB: u64 = 16 * 1024;

/// ホスト RAM tier を入れる最低 RAM (MiB)。
pub const RAM_TIER_MIN_RAM_MB: u64 = 32 * 1024;

/// 小さいカードで prefill の同時 pread を絞る閾値 (MiB)。
pub const SMALL_VRAM_MB: u64 = 10 * 1024;
/// 中間のカードの閾値 (MiB)。
pub const MID_VRAM_MB: u64 = 16 * 1024;

/// モデルの量子型の粗い分類。計画では「dense 重みがモデル全体の何割か」の
/// 推定にだけ使う (演算には使わない)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelQuant {
    /// IQ3_XXS / IQ2_S / IQ4_NL などの IQ 系 (既定モデル UD-IQ3_XXS)。
    Q3,
    /// Q4_K / Q5_1 / Q6_K などの K 量子 (UD-Q4_K_XL)。
    Q4,
}

impl ModelQuant {
    pub fn as_str(self) -> &'static str {
        match self {
            ModelQuant::Q3 => "Q3",
            ModelQuant::Q4 => "Q4",
        }
    }

    /// 文字列からの解釈 (純関数)。`iq3` / `q3` / `iq4` / `q4` を受ける。
    /// 判らない値は `None` (呼び出し側がモデルのテンソル型から決める)。
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "q3" | "iq3" | "iq3_xxs" | "iq3_s" | "q3_k" => Some(ModelQuant::Q3),
            "q4" | "q4_k" | "q4_k_xl" | "q5" | "q5_k" => Some(ModelQuant::Q4),
            _ => None,
        }
    }

    /// モデル全体に占める dense (expert 以外) の重みの割合 (%)。
    ///
    /// 実測: UD-Q4_K_XL は 111 GB 中 dense 側 56,794 MiB (51%)、
    /// UD-IQ3_XXS は 83.6 GB 中ほぼ同じ dense を持つので約 68%。
    /// どちらも expert の量子型だけが違う (hc_* と ple は Q8_0、ssm は F32)。
    pub fn dense_pct(self) -> u64 {
        match self {
            ModelQuant::Q3 => 68,
            ModelQuant::Q4 => 51,
        }
    }
}

/// 1 枚のカード向けの計画。すべて MiB。
#[derive(Clone, Debug, PartialEq)]
pub struct GpuConfig {
    /// 計画の入力 (ログ用)
    pub vram_mb: u64,
    pub ram_mb: u64,
    pub model_mb: u64,
    pub quant: ModelQuant,
    /// dense 重みの推定バイト数 (MiB)。`weight_mb` の頭打ちに使う。
    pub dense_est_mb: u64,
    /// `DENNO_VRAM_RESERVE_MB`
    pub reserve_mb: u64,
    /// `DENNO_GPU_WEIGHT_MB`
    pub gpu_weight_mb: u64,
    /// `DENNO_EXPERT_VRAM_MB` (0 = expert を VRAM に載せない)
    pub expert_vram_mb: u64,
    /// `DENNO_EXPERT_GPU` / `DENNO_EXPERT_IQ_GPU` (expert_vram_mb > 0 のとき真)
    pub expert_gpu: bool,
    /// `DENNO_EXPERT_GPU_MIN_HIT` (`None` = 既定 0.5 のまま)
    pub expert_gpu_min_hit: Option<f32>,
    /// `DENNO_EXPERT_RAM_MB` (`None` = 未設定 = RAM tier 無効)
    pub expert_ram: Option<&'static str>,
    /// `DENNO_PF_PREAD_THREADS` (`None` = 既定 / P814 の memadapt に任せる)
    pub pread_threads: Option<usize>,
    /// `DENNO_PF_CHUNK` (`None` = 既定 1536)
    pub pf_chunk: Option<usize>,
}

impl GpuConfig {
    /// expert を GPU で回す計画か。
    pub fn experts_on_gpu(&self) -> bool {
        self.expert_vram_mb > 0 && self.expert_gpu
    }
}

/// カード 1 枚ぶんの計画を式から組む純関数。
///
/// - `vram_mb`: カードの総 VRAM (MiB)
/// - `ram_mb`: ホスト RAM の総量 (MiB)
/// - `model_bytes`: モデルの重み合計バイト数 (GGUF のファイル長の和)
/// - `model_quant`: expert の量子型の粗い分類
///
/// 予算は [`PLAN_RESERVE_MB`] を引いた「使える VRAM」に対する割合で決める
/// ([`WEIGHT_PCT`] / [`EXPERT_PCT`])。和が使える VRAM を超えてよい理由は
/// モジュール doc の「予算は上限であって予約ではない」を見よ。
pub fn plan_gpu_config(
    vram_mb: u64,
    ram_mb: u64,
    model_bytes: u64,
    model_quant: ModelQuant,
) -> GpuConfig {
    let model_mb = model_bytes >> 20;
    let usable = vram_mb.saturating_sub(PLAN_RESERVE_MB);
    let dense_est_mb = model_mb.saturating_mul(model_quant.dense_pct()) / 100;

    // dense: 割合の上限と、実際に必要な量の小さい方 (小さいモデルで
    // 「要らないのに 12 GB を上限として書く」のを避ける)。
    let mut weight = (usable * WEIGHT_PCT / 100).min(WEIGHT_CAP_MB);
    if dense_est_mb > 0 {
        weight = weight.min(dense_est_mb);
    }
    // 小さいカードでも dense が 0 にならないように 256 MiB の床を置く。
    weight = weight.max(256.min(usable));

    // expert: 割合の上限と絶対上限の小さい方。expert 総量がそれ未満なら
    // 全部載るので、上限を expert 総量まで下げる (ログの見栄えのため)。
    let expert_total_mb = model_mb.saturating_sub(dense_est_mb);
    let mut expert = (usable * EXPERT_PCT / 100).min(EXPERT_CAP_MB);
    if expert_total_mb > 0 {
        expert = expert.min(expert_total_mb);
    }
    // 使える VRAM の 12% か 1 GiB を expert に回せないカードでは expert を切る
    // (2 GB 級。載せても 1 本も入らずに往復だけが増える)。
    if expert < MIN_EXPERT_MB || expert < usable * 12 / 100 {
        expert = 0;
    }

    let expert_gpu = expert > 0;
    let expert_gpu_min_hit = if expert > 0 && tier_at_least(vram_mb, MIN_HIT_ZERO_FROM_MB) {
        // P45 のヒット率ゲートは 0.5 だと CPU/GPU を往復する (lab t794a の
        // 2 割の揺れ)。手動ピークも 0 を明示している。
        // task 886: 12 GB 級の実測 (12,282 MiB) も 12 GB として扱う。
        Some(0.0)
    } else {
        None
    };

    let expert_ram = if vram_mb <= RAM_TIER_MAX_VRAM_MB + VRAM_TIER_SLACK_MB
        && ram_mb >= RAM_TIER_MIN_RAM_MB
    {
        Some("auto")
    } else {
        None
    };

    // prefill の資源: VRAM が小さいカードは expert が CPU/ディスクへ落ちるので、
    // 同時 pread を絞って 1 グループの待ちを短くする (大きいカードは既定 =
    // P814 の memadapt に任せる)。段は**上限**側の境界なので、公称より少し
    // 小さい実測 (12,282 / 16,376) を同じ段に残す向きに slack を足す。
    let (pread_threads, pf_chunk) = if vram_mb <= SMALL_VRAM_MB + VRAM_TIER_SLACK_MB {
        (Some(8), Some(1024))
    } else if vram_mb <= MID_VRAM_MB + VRAM_TIER_SLACK_MB {
        (Some(12), Some(1536))
    } else {
        (None, None)
    };

    GpuConfig {
        vram_mb,
        ram_mb,
        model_mb,
        quant: model_quant,
        dense_est_mb,
        reserve_mb: PLAN_RESERVE_MB,
        gpu_weight_mb: weight,
        expert_vram_mb: expert,
        expert_gpu,
        expert_gpu_min_hit,
        expert_ram,
        pread_threads,
        pf_chunk,
    }
}

/// [`GpuConfig`] が要求する env (変数名, 値)。適用順はこの順。
///
/// `DENNO_EXPERT_GPU` と `DENNO_EXPERT_IQ_GPU` の両方を立てるのは、
/// 旧 4 型 (Q4_K など) と IQ 系で入口が別だから
/// (`expert_vram::resolve_route` の doc)。
pub fn plan_env_pairs(c: &GpuConfig) -> Vec<(&'static str, String)> {
    let mut v: Vec<(&'static str, String)> = Vec::new();
    v.push(("DENNO_VRAM_RESERVE_MB", c.reserve_mb.to_string()));
    v.push(("DENNO_GPU_WEIGHT_MB", c.gpu_weight_mb.to_string()));
    v.push(("DENNO_EXPERT_VRAM_MB", c.expert_vram_mb.to_string()));
    if c.experts_on_gpu() {
        v.push(("DENNO_EXPERT_GPU", "1".to_string()));
        v.push(("DENNO_EXPERT_IQ_GPU", "1".to_string()));
    }
    if let Some(m) = c.expert_gpu_min_hit {
        v.push(("DENNO_EXPERT_GPU_MIN_HIT", format!("{m}")));
    }
    if let Some(r) = c.expert_ram {
        v.push(("DENNO_EXPERT_RAM_MB", r.to_string()));
    }
    if let Some(n) = c.pread_threads {
        v.push(("DENNO_PF_PREAD_THREADS", n.to_string()));
    }
    if let Some(n) = c.pf_chunk {
        v.push(("DENNO_PF_CHUNK", n.to_string()));
    }
    v
}

/// 起動時の 1 行。`applied` は実際に書いた変数名、`kept` は明示 env が
/// あったので書かなかった変数名。
pub fn plan_line(c: &GpuConfig, applied: &[&str], kept: &[&str]) -> String {
    let mut s = format!(
        "[auto-config] plan vram={} MiB ram={} MiB model={} MiB {} (dense~{}) -> \
         weight={} MiB expert_vram={} MiB expert_gpu={} min_hit={} ram_tier={} \
         pread={} chunk={}",
        c.vram_mb,
        c.ram_mb,
        c.model_mb,
        c.quant.as_str(),
        c.dense_est_mb,
        c.gpu_weight_mb,
        c.expert_vram_mb,
        if c.experts_on_gpu() { "on" } else { "off" },
        c.expert_gpu_min_hit
            .map(|m| format!("{m}"))
            .unwrap_or_else(|| "default(0.5)".to_string()),
        c.expert_ram.unwrap_or("off"),
        c.pread_threads
            .map(|n| n.to_string())
            .unwrap_or_else(|| "default".to_string()),
        c.pf_chunk
            .map(|n| n.to_string())
            .unwrap_or_else(|| "default".to_string()),
    );
    if !applied.is_empty() {
        s.push_str(&format!(" applied={}", applied.join(",")));
    }
    if !kept.is_empty() {
        s.push_str(&format!(" kept(explicit env)={}", kept.join(",")));
    }
    s
}

/// env の生値が「未設定」か。空文字と空白だけも未設定として扱う
/// (`DENNO_EXPERT_VRAM_MB=` は設定していないのと同じ)。
pub fn is_unset(v: Option<&str>) -> bool {
    match v {
        None => true,
        Some(s) => s.trim().is_empty(),
    }
}

/// env の生値が「auto に任せる」の意味か。
///
/// 既に `DENNO_*` を書いている環境 (本番の起動スクリプトなど) でも、
/// その値を `auto` にすれば**その変数だけ**計画値で置き換えられる。
/// 空文字と同じ扱い (`is_unset` と OR で使う)。
pub fn is_auto_sentinel(v: Option<&str>) -> bool {
    matches!(v.map(|s| s.trim().to_ascii_lowercase()).as_deref(), Some("auto"))
}

/// [`plan_env_pairs`] を **未設定 (または `auto`) の変数だけ** に書き、
/// 書いた名前を返す。
///
/// 明示 env が既にある変数は触らない (`kept` 側)。これは
/// 「auto は既定を選ぶだけで、手で書いた値を上書きしない」の実装。
/// 値が `auto` のときだけは「この変数は auto に任せる」の意味で書く
/// (本番の起動スクリプトに既存の値がある環境向け)。
pub fn apply_env_pairs(pairs: &[(&'static str, String)]) -> (Vec<&'static str>, Vec<&'static str>) {
    let mut applied = Vec::new();
    let mut kept = Vec::new();
    for (k, v) in pairs {
        let cur = std::env::var(k).ok();
        if is_unset(cur.as_deref()) || is_auto_sentinel(cur.as_deref()) {
            std::env::set_var(k, v);
            applied.push(*k);
        } else {
            kept.push(*k);
        }
    }
    (applied, kept)
}

/// `--vram-budget` の解釈。`Some("auto")` は auto、数値は「VRAM を MiB で
/// 上書きした auto」として扱う (検出できない環境で手で渡すため)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VramBudget {
    Auto,
    /// 総 VRAM を MiB で明示 (計画は auto と同じ式で組む)。
    Fixed(u64),
}

impl VramBudget {
    pub fn parse(s: &str) -> Option<Self> {
        let t = s.trim();
        if t.eq_ignore_ascii_case("auto") {
            return Some(VramBudget::Auto);
        }
        t.parse::<u64>().ok().filter(|&n| n > 0).map(VramBudget::Fixed)
    }
    /// 計画に使う総 VRAM (MiB)。
    pub fn vram_mb(self) -> Option<u64> {
        match self {
            VramBudget::Auto => None,
            VramBudget::Fixed(n) => Some(n),
        }
    }
}

/// auto 計画を立てるか。CLI (`--vram-budget`) が env より優先。
///
/// `cli` は `--vram-budget` の生値、`env` は `DENNO_AUTO_CONFIG` の生値。
pub fn auto_requested(cli: Option<&str>, env: Option<&str>) -> bool {
    if let Some(v) = cli.map(str::trim).filter(|s| !s.is_empty()) {
        return VramBudget::parse(v).is_some();
    }
    match env.map(|s| s.trim().to_ascii_lowercase()) {
        Some(s) => matches!(s.as_str(), "1" | "true" | "on" | "yes" | "auto"),
        None => false,
    }
}

/// `DENNO_VRAM_TOTAL_MB`: 計画に使う総 VRAM を MiB で上書きする env。task 886。
///
/// 存在理由は 2 つ。
/// - `--vram-budget <MiB>` と同じ用途を env でも渡せるようにする (lab の
///   COND は `DENNO_*` しか渡せないので、10 / 12 / 16 GB の段を実カード無しで
///   検証するには env 側の入口が要る)。
/// - 段の検証 (CI・ユニット) で実 GPU を要求しない。
///
/// 優先順位は `--vram-budget` の数値 > この env > 実カードの検出。
/// `auto` / 空 / 0 は「上書きしない」。
pub const VRAM_TOTAL_ENV: &str = "DENNO_VRAM_TOTAL_MB";

/// [`VRAM_TOTAL_ENV`] の解釈 (純関数)。`auto` / 空 / 0 / 非数は `None`。
pub fn vram_total_override(raw: Option<&str>) -> Option<u64> {
    let t = raw?.trim();
    if t.is_empty() || t.eq_ignore_ascii_case("auto") {
        return None;
    }
    t.parse::<u64>().ok().filter(|&n| n > 0)
}

/// `/proc/meminfo` の `MemTotal` を MiB で返す純関数 (テスト用に本文を受ける)。
pub fn parse_meminfo_total_mb(text: &str) -> Option<u64> {
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb / 1024);
        }
    }
    None
}

/// ホスト RAM の総量 (MiB)。`/proc/meminfo` が読めなければ 0。
pub fn host_ram_mb() -> u64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|t| parse_meminfo_total_mb(&t))
        .unwrap_or(0)
}

/// カードの総 VRAM (MiB)。
///
/// `cuda` feature が無い / libcuda が無い / `cuDeviceTotalMem_v2` が無い、
/// のどれでも `None` (auto 計画は「検出できない」で何も適用しない)。
///
/// **current context を作らない**引き方を使う (`cuda::device_total_mem_mb`)。
/// `cuMemGetInfo` 版 (`dev_mem_mb`) はモデル構築前だと必ず失敗する。
#[cfg(feature = "cuda")]
pub fn probe_vram_mb() -> Option<u64> {
    crate::cuda::device_total_mem_mb().map(|m| m as u64)
}

#[cfg(not(feature = "cuda"))]
pub fn probe_vram_mb() -> Option<u64> {
    None
}

/// カードの compute capability (major, minor)。ログ用。
#[cfg(feature = "cuda")]
pub fn probe_sm() -> Option<(i32, i32)> {
    crate::cuda::device_sm()
}

#[cfg(not(feature = "cuda"))]
pub fn probe_sm() -> Option<(i32, i32)> {
    None
}

/// モデルのファイル長の和から `model_bytes` を作る (純関数: 長さの列を受ける)。
pub fn sum_model_bytes<I: IntoIterator<Item = u64>>(lens: I) -> u64 {
    lens.into_iter().fold(0u64, |a, b| a.saturating_add(b))
}

/// expert のテンソル型から量子の分類を決める (純関数)。
///
/// 4 bit 系 (Q4_K / Q5_K / Q5_1 / Q6_K / Q4_0 / Q4_1 / Q5_0) が 1 本でも
/// あれば Q4、それ以外 (IQ 系 / Q2 / Q3) は Q3。expert が無い (dense のみ)
/// モデルは `None`。
pub fn quant_from_expert_types<I: IntoIterator<Item = crate::kquant::GgmlType>>(
    types: I,
) -> Option<ModelQuant> {
    use crate::kquant::GgmlType as T;
    let mut any = false;
    let mut q4 = false;
    for t in types {
        any = true;
        if matches!(
            t,
            T::Q4K | T::Q5K | T::Q5_1 | T::Q6K | T::Q4_0 | T::Q4_1 | T::Q5_0
        ) {
            q4 = true;
        }
    }
    if !any {
        return None;
    }
    Some(if q4 { ModelQuant::Q4 } else { ModelQuant::Q3 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// env を触るテストは直列化する (`--test-threads=1` 前提だが、
    /// 他モジュールの env テストと混ざらないよう自前の錠を持つ)。
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn lock() -> std::sync::MutexGuard<'static, ()> {
        ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    const GB: u64 = 1024;

    /// 8 / 10 / 12 / 16 / 24 / 32 GB x Q3 / Q4 の全 12 通り。
    fn cases() -> Vec<(u64, u64, u64, ModelQuant)> {
        // 110 GB 級 (UD-Q4_K_XL) と 84 GB 級 (UD-IQ3_XXS) のファイル長。
        let q4_bytes = 111u64 * 1000 * 1000 * 1000;
        let q3_bytes = 84u64 * 1000 * 1000 * 1000;
        let mut v = Vec::new();
        for gb in [8u64, 10, 12, 16, 24, 32] {
            v.push((gb * 1024, 64 * GB, q3_bytes, ModelQuant::Q3));
            v.push((gb * 1024, 64 * GB, q4_bytes, ModelQuant::Q4));
        }
        v
    }

    #[test]
    fn plan_covers_eight_to_thirtytwo_gb_for_q3_and_q4() {
        for (vram, ram, model_bytes, q) in cases() {
            let c = plan_gpu_config(vram, ram, model_bytes, q);
            let usable = vram - PLAN_RESERVE_MB;
            assert_eq!(c.vram_mb, vram);
            assert_eq!(c.quant, q);
            // 0 に潰さない / 上限を超えない
            assert!(c.gpu_weight_mb >= 256, "{vram} {q:?} weight=0");
            assert!(
                c.gpu_weight_mb <= WEIGHT_CAP_MB,
                "{vram} {q:?} weight={}",
                c.gpu_weight_mb
            );
            assert!(
                c.expert_vram_mb <= EXPERT_CAP_MB,
                "{vram} {q:?} expert={}",
                c.expert_vram_mb
            );
            // 8 GB 以上なら expert は必ず載せる (8 GB でも 4 GB 級が入る)
            assert!(
                c.expert_vram_mb > 0,
                "{vram} {q:?} で expert を切ってしまった"
            );
            assert!(c.experts_on_gpu());
            // 両方の和が使える VRAM の 120% を超えない (丸めの暴走を検出)
            assert!(
                c.gpu_weight_mb + c.expert_vram_mb <= usable * 120 / 100,
                "{vram} {q:?}: {} + {} > {}*1.2",
                c.gpu_weight_mb,
                c.expert_vram_mb,
                usable
            );
            // 12 GB 以上では P45 ゲートを 0 にする
            if vram >= MIN_HIT_ZERO_FROM_MB {
                assert_eq!(c.expert_gpu_min_hit, Some(0.0), "{vram}");
            } else {
                assert_eq!(c.expert_gpu_min_hit, None, "{vram}");
            }
            // prefill 資源は 16 GB 以下でだけ明示する (大きいカードは P814)
            if vram <= SMALL_VRAM_MB {
                assert_eq!(c.pread_threads, Some(8));
                assert_eq!(c.pf_chunk, Some(1024));
            } else if vram <= MID_VRAM_MB {
                assert_eq!(c.pread_threads, Some(12));
                assert_eq!(c.pf_chunk, Some(1536));
            } else {
                assert_eq!(c.pread_threads, None);
                assert_eq!(c.pf_chunk, None);
            }
        }
    }

    #[test]
    fn plan_5090_matches_manual_peak_config() {
        // RTX 5090 の実測値は 32,110 MiB (lab t818b の `[auto-config] plan`)。
        // lab の手動ピーク COND と同じ値になること。
        for vram in [32_110u64, 32_607] {
            let c = plan_gpu_config(vram, 64 * GB, 111_000_000_000, ModelQuant::Q4);
            assert_eq!(c.gpu_weight_mb, 12_000, "{vram}");
            assert_eq!(c.expert_vram_mb, 22_000, "{vram}");
            assert_eq!(c.expert_gpu_min_hit, Some(0.0));
            assert!(c.expert_gpu);
            assert_eq!(c.expert_ram, None);
            assert_eq!(c.pread_threads, None);
            assert_eq!(c.pf_chunk, None);
        }
        // 8 / 10 / 16 / 24 GB は手動ピークより小さい (カードに合わせて縮む)
        for gb in [8u64, 10, 16, 24] {
            let c = plan_gpu_config(gb * 1024, 64 * GB, 111_000_000_000, ModelQuant::Q4);
            let usable = gb * 1024 - PLAN_RESERVE_MB;
            assert_eq!(c.expert_vram_mb, usable * EXPERT_PCT / 100, "{gb}");
            assert_eq!(c.gpu_weight_mb, usable * WEIGHT_PCT / 100, "{gb}");
            // 32 GB より小さい (単調)
            assert!(c.expert_vram_mb < 22_000, "{gb}");
            assert!(c.gpu_weight_mb < 12_000, "{gb}");
        }
    }

    #[test]
    fn plan_small_model_caps_by_actual_dense_size() {
        // 20 GB の Q4 モデルなら dense 推定 10.2 GB < 12 GB の上限。
        let c = plan_gpu_config(32_607, 64 * GB, 20_000_000_000, ModelQuant::Q4);
        assert_eq!(c.dense_est_mb, 20_000_000_000 / (1 << 20) * 51 / 100);
        assert_eq!(c.gpu_weight_mb, c.dense_est_mb);
        // expert 総量 (model - dense) が上限より小さいので expert もそこまで。
        let total = 20_000_000_000 / (1 << 20);
        assert_eq!(c.expert_vram_mb, total - c.dense_est_mb);
        // Q3 は dense の割合が高い
        let q3 = plan_gpu_config(32_607, 64 * GB, 20_000_000_000, ModelQuant::Q3);
        assert!(q3.dense_est_mb > c.dense_est_mb);
    }

    #[test]
    fn plan_tiny_card_disables_experts() {
        // 2 GB のカード: 使える 1 GB の 12% も expert に回せない
        let c = plan_gpu_config(2 * 1024, 16 * GB, 111_000_000_000, ModelQuant::Q4);
        assert_eq!(c.expert_vram_mb, 0);
        assert!(!c.experts_on_gpu());
        assert_eq!(c.expert_gpu_min_hit, None);
        // VRAM 未知 (0) でも panic しない
        let z = plan_gpu_config(0, 0, 0, ModelQuant::Q3);
        assert_eq!(z.expert_vram_mb, 0);
        assert_eq!(z.gpu_weight_mb, 0);
    }

    /// task 886: 8 / 12 / 16 / 24 / 32 GB の段ごとの計画値を**表で**縛る
    /// (式を書き換えたらここで落ちる)。10 GB は 12 GB の下限側の境界として残す。
    #[test]
    fn plan_tier_table_matches_expected_values() {
        // (公称 GB, weight MiB, expert MiB, min_hit, pread, chunk, ram_tier)
        let table: [(u64, u64, u64, Option<f32>, Option<usize>, Option<usize>, bool); 6] = [
            (8, 2_867, 5_160, None, Some(8), Some(1024), true),
            (10, 3_686, 6_635, None, Some(8), Some(1024), true),
            (12, 4_505, 8_110, Some(0.0), Some(12), Some(1536), true),
            (16, 6_144, 11_059, Some(0.0), Some(12), Some(1536), true),
            (24, 9_420, 16_957, Some(0.0), None, None, false),
            (32, 12_000, 22_000, Some(0.0), None, None, false),
        ];
        for (gb, weight, expert, min_hit, pread, chunk, ram_tier) in table {
            for q in [ModelQuant::Q3, ModelQuant::Q4] {
                let c = plan_gpu_config(gb * 1024, 64 * GB, 111_000_000_000, q);
                assert_eq!(c.gpu_weight_mb, weight, "{gb} GB {q:?} weight");
                assert_eq!(c.expert_vram_mb, expert, "{gb} GB {q:?} expert");
                assert_eq!(c.expert_gpu_min_hit, min_hit, "{gb} GB {q:?} min_hit");
                assert_eq!(c.pread_threads, pread, "{gb} GB {q:?} pread");
                assert_eq!(c.pf_chunk, chunk, "{gb} GB {q:?} chunk");
                assert_eq!(
                    c.expert_ram.is_some(),
                    ram_tier,
                    "{gb} GB {q:?} ram_tier"
                );
                // 予算の和は「使える VRAM の 120%」以内 (丸めの暴走を検出。
                // 40% + 72% = 112% なので 12/16 GB でもこの範囲)
                let usable = gb * 1024 - PLAN_RESERVE_MB;
                assert!(
                    c.gpu_weight_mb + c.expert_vram_mb <= usable * 120 / 100,
                    "{gb} GB {q:?}: {} + {} > {usable}*1.20",
                    c.gpu_weight_mb,
                    c.expert_vram_mb
                );
            }
        }
    }

    /// task 886: 実カードの実測 VRAM (公称より少し小さい) が公称の段に入る。
    /// ここが 12 GB の thrash floor (P45 ゲート 0.5 のまま) の回帰点。
    #[test]
    fn plan_real_card_sizes_land_in_nominal_tier() {
        // 実測値の例: RTX 4070 12,282 / 4080 16,376 / 5070 Ti 16,303 / 4090 24,564。
        let c12 = plan_gpu_config(12_282, 64 * GB, 111_000_000_000, ModelQuant::Q4);
        assert_eq!(c12.expert_gpu_min_hit, Some(0.0), "12,282 MiB が 12 GB 扱いでない");
        assert_eq!(c12.pread_threads, Some(12));
        assert_eq!(c12.expert_ram, Some("auto"));
        // 10 GB の 3080 (10,240) は 12 GB 扱いしない (ゲートは既定のまま)
        let c10 = plan_gpu_config(10_240, 64 * GB, 111_000_000_000, ModelQuant::Q4);
        assert_eq!(c10.expert_gpu_min_hit, None);
        // 16 GB 級は 12 GB と同じ pread / RAM tier、24 GB 級は触らない
        for v in [16_303u64, 16_376] {
            let c = plan_gpu_config(v, 64 * GB, 111_000_000_000, ModelQuant::Q4);
            assert_eq!(c.expert_gpu_min_hit, Some(0.0), "{v}");
            assert_eq!(c.pread_threads, Some(12), "{v}");
            assert_eq!(c.expert_ram, Some("auto"), "{v}");
        }
        let c24 = plan_gpu_config(24_564, 64 * GB, 111_000_000_000, ModelQuant::Q4);
        assert_eq!(c24.expert_gpu_min_hit, Some(0.0));
        assert_eq!(c24.pread_threads, None, "24 GB は P814 に任せる");
        assert_eq!(c24.expert_ram, None);
        // slack は 256 MiB。境界の内と外を 1 MiB で確かめる
        assert!(tier_at_least(12 * 1024 - VRAM_TIER_SLACK_MB, 12 * 1024));
        assert!(!tier_at_least(12 * 1024 - VRAM_TIER_SLACK_MB - 1, 12 * 1024));
    }

    /// task 886: `DENNO_VRAM_TOTAL_MB` の解釈 (lab の COND は env しか渡せない)。
    #[test]
    fn vram_total_override_parses() {
        assert_eq!(vram_total_override(Some("12288")), Some(12_288));
        assert_eq!(vram_total_override(Some(" 16000 ")), Some(16_000));
        assert_eq!(vram_total_override(Some("auto")), None);
        assert_eq!(vram_total_override(Some("AUTO")), None);
        assert_eq!(vram_total_override(Some("")), None);
        assert_eq!(vram_total_override(Some("0")), None);
        assert_eq!(vram_total_override(Some("x")), None);
        assert_eq!(vram_total_override(None), None);
        // 上書きした総量で組んだ計画が、同じ総量の実カードと同じ値になる
        let sim = plan_gpu_config(
            vram_total_override(Some("12288")).unwrap(),
            64 * GB,
            111_000_000_000,
            ModelQuant::Q4,
        );
        let real = plan_gpu_config(12_288, 64 * GB, 111_000_000_000, ModelQuant::Q4);
        assert_eq!(sim, real);
        assert_eq!(sim.expert_vram_mb, 8_110);
    }

    #[test]
    fn plan_ram_tier_only_on_small_cards_with_enough_ram() {
        let big = plan_gpu_config(24 * 1024, 128 * GB, 111_000_000_000, ModelQuant::Q4);
        assert_eq!(big.expert_ram, None);
        let small = plan_gpu_config(10 * 1024, 64 * GB, 111_000_000_000, ModelQuant::Q4);
        assert_eq!(small.expert_ram, Some("auto"));
        let small_low_ram =
            plan_gpu_config(10 * 1024, 16 * GB, 111_000_000_000, ModelQuant::Q4);
        assert_eq!(small_low_ram.expert_ram, None);
    }

    #[test]
    fn env_pairs_explicit_user_values_win() {
        let _g = lock();
        for k in [
            "DENNO_VRAM_RESERVE_MB",
            "DENNO_GPU_WEIGHT_MB",
            "DENNO_EXPERT_VRAM_MB",
            "DENNO_EXPERT_GPU",
            "DENNO_EXPERT_IQ_GPU",
            "DENNO_EXPERT_GPU_MIN_HIT",
            "DENNO_PF_PREAD_THREADS",
            "DENNO_PF_CHUNK",
            "DENNO_EXPERT_RAM_MB",
        ] {
            std::env::remove_var(k);
        }
        let c = plan_gpu_config(10 * 1024, 64 * GB, 111_000_000_000, ModelQuant::Q4);
        let pairs = plan_env_pairs(&c);

        // 1 回目: 全部未設定なので全部書く
        let (applied, kept) = apply_env_pairs(&pairs);
        assert_eq!(applied.len(), pairs.len());
        assert!(kept.is_empty());
        assert_eq!(
            std::env::var("DENNO_EXPERT_VRAM_MB").unwrap(),
            c.expert_vram_mb.to_string()
        );

        // 2 回目: 既に値があるので書かない。例外は計画値が `auto` の変数
        // (10 GB の計画は `DENNO_EXPERT_RAM_MB=auto` を出す) で、これは
        // 「auto に任せる」の意味なので毎回書く。
        let (applied2, kept2) = apply_env_pairs(&pairs);
        assert_eq!(applied2, vec!["DENNO_EXPERT_RAM_MB"]);
        assert_eq!(kept2.len(), pairs.len() - 1);
        assert_eq!(std::env::var("DENNO_EXPERT_RAM_MB").unwrap(), "auto");

        // 明示値を入れてから: その変数だけ kept、他は applied
        std::env::set_var("DENNO_EXPERT_VRAM_MB", "18000");
        let (applied3, kept3) = apply_env_pairs(&pairs);
        assert!(kept3.contains(&"DENNO_EXPERT_VRAM_MB"));
        assert!(!applied3.contains(&"DENNO_EXPERT_VRAM_MB"));
        assert_eq!(std::env::var("DENNO_EXPERT_VRAM_MB").unwrap(), "18000");
        // 空文字は「未設定」と同じ
        std::env::set_var("DENNO_GPU_WEIGHT_MB", "  ");
        let (applied4, _kept4) = apply_env_pairs(&pairs);
        assert!(applied4.contains(&"DENNO_GPU_WEIGHT_MB"));
        assert_eq!(
            std::env::var("DENNO_GPU_WEIGHT_MB").unwrap(),
            c.gpu_weight_mb.to_string()
        );

        // `auto` は「この変数は auto に任せる」= 書く
        std::env::set_var("DENNO_EXPERT_VRAM_MB", "auto");
        let (applied5, kept5) = apply_env_pairs(&pairs);
        assert!(applied5.contains(&"DENNO_EXPERT_VRAM_MB"));
        assert!(!kept5.contains(&"DENNO_EXPERT_VRAM_MB"));
        assert_eq!(
            std::env::var("DENNO_EXPERT_VRAM_MB").unwrap(),
            c.expert_vram_mb.to_string()
        );
        // 大文字の AUTO も同じ
        std::env::set_var("DENNO_EXPERT_VRAM_MB", "AUTO");
        let (applied6, _k6) = apply_env_pairs(&pairs);
        assert!(applied6.contains(&"DENNO_EXPERT_VRAM_MB"));

        for k in [
            "DENNO_VRAM_RESERVE_MB",
            "DENNO_GPU_WEIGHT_MB",
            "DENNO_EXPERT_VRAM_MB",
            "DENNO_EXPERT_GPU",
            "DENNO_EXPERT_IQ_GPU",
            "DENNO_EXPERT_GPU_MIN_HIT",
            "DENNO_PF_PREAD_THREADS",
            "DENNO_PF_CHUNK",
            "DENNO_EXPERT_RAM_MB",
        ] {
            std::env::remove_var(k);
        }
    }

    #[test]
    fn env_pairs_contents() {
        let c = plan_gpu_config(32_607, 64 * GB, 111_000_000_000, ModelQuant::Q4);
        let p = plan_env_pairs(&c);
        let get = |k: &str| {
            p.iter()
                .find(|(n, _)| *n == k)
                .map(|(_, v)| v.as_str())
        };
        assert_eq!(get("DENNO_EXPERT_VRAM_MB"), Some("22000"));
        assert_eq!(get("DENNO_GPU_WEIGHT_MB"), Some("12000"));
        assert_eq!(get("DENNO_EXPERT_GPU"), Some("1"));
        assert_eq!(get("DENNO_EXPERT_IQ_GPU"), Some("1"));
        assert_eq!(get("DENNO_EXPERT_GPU_MIN_HIT"), Some("0"));
        assert_eq!(get("DENNO_VRAM_RESERVE_MB"), Some("1024"));
        // 32 GB では prefill 資源と RAM tier は触らない
        assert_eq!(get("DENNO_PF_PREAD_THREADS"), None);
        assert_eq!(get("DENNO_PF_CHUNK"), None);
        assert_eq!(get("DENNO_EXPERT_RAM_MB"), None);

        // expert を切る計画では expert の env を出さない (0 を書いても同じだが、
        // ログの kept/applied を読みやすくする)
        let tiny = plan_gpu_config(2 * 1024, 16 * GB, 111_000_000_000, ModelQuant::Q4);
        let pt = plan_env_pairs(&tiny);
        assert!(pt.iter().all(|(n, _)| *n != "DENNO_EXPERT_GPU"));
        assert!(pt.iter().any(|(n, v)| *n == "DENNO_EXPERT_VRAM_MB" && v == "0"));
    }

    #[test]
    fn auto_requested_precedence() {
        // CLI が最優先
        assert!(auto_requested(Some("auto"), None));
        assert!(auto_requested(Some("AUTO"), Some("0")));
        assert!(!auto_requested(Some("off"), Some("1")));
        // env だけ
        assert!(auto_requested(None, Some("1")));
        assert!(auto_requested(None, Some("true")));
        assert!(auto_requested(None, Some("auto")));
        assert!(!auto_requested(None, Some("0")));
        assert!(!auto_requested(None, None));
        assert!(!auto_requested(None, Some("")));
        // 空白だけの CLI は未指定と同じ
        assert!(auto_requested(Some("  "), Some("on")));
    }

    #[test]
    fn vram_budget_parse() {
        assert_eq!(VramBudget::parse("auto"), Some(VramBudget::Auto));
        assert_eq!(VramBudget::parse(" 24000 "), Some(VramBudget::Fixed(24000)));
        assert_eq!(VramBudget::parse("0"), None);
        assert_eq!(VramBudget::parse("x"), None);
        assert_eq!(VramBudget::Auto.vram_mb(), None);
        assert_eq!(VramBudget::Fixed(24000).vram_mb(), Some(24000));
    }

    #[test]
    fn meminfo_and_model_bytes() {
        let text = "MemTotal:       65834512 kB\nMemFree:  123 kB\n";
        assert_eq!(parse_meminfo_total_mb(text), Some(65834512 / 1024));
        assert_eq!(parse_meminfo_total_mb("MemFree: 1 kB\n"), None);
        assert!(host_ram_mb() > 0, "このコンテナで /proc/meminfo が読めない");
        assert_eq!(sum_model_bytes([1u64, 2, 3]), 6);
        assert_eq!(sum_model_bytes(Vec::<u64>::new()), 0);
        assert_eq!(sum_model_bytes([u64::MAX, 1]), u64::MAX);
    }

    #[test]
    fn quant_from_expert_types_classifies() {
        use crate::kquant::GgmlType as T;
        assert_eq!(
            quant_from_expert_types([T::IQ3Xxs, T::IQ2S, T::IQ4Nl]),
            Some(ModelQuant::Q3)
        );
        assert_eq!(
            quant_from_expert_types([T::Q4K, T::Q5_1, T::Q6K]),
            Some(ModelQuant::Q4)
        );
        // 混在は Q4 (4 bit が 1 本でもあれば Q4)
        assert_eq!(
            quant_from_expert_types([T::IQ3Xxs, T::Q4K]),
            Some(ModelQuant::Q4)
        );
        assert_eq!(quant_from_expert_types(Vec::<T>::new()), None);
        assert_eq!(ModelQuant::parse("iq3_xxs"), Some(ModelQuant::Q3));
        assert_eq!(ModelQuant::parse("Q4_K_XL"), Some(ModelQuant::Q4));
        assert_eq!(ModelQuant::parse("fp16"), None);
    }

    #[test]
    fn plan_line_is_one_readable_line() {
        let c = plan_gpu_config(32_607, 64 * GB, 111_000_000_000, ModelQuant::Q4);
        let line = plan_line(&c, &["DENNO_GPU_WEIGHT_MB"], &["DENNO_EXPERT_VRAM_MB"]);
        assert!(line.starts_with("[auto-config] plan vram=32607"));
        assert!(line.contains("expert_vram=22000"));
        assert!(line.contains("applied=DENNO_GPU_WEIGHT_MB"));
        assert!(line.contains("kept(explicit env)=DENNO_EXPERT_VRAM_MB"));
        assert!(!line.contains('\n'), "1 行であること");
    }
}
