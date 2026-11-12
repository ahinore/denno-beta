//! task 227: prefill の expert 重み転送のマイクロベンチ (サーバ無し)。
//!
//! 実機 (GPU) で、`cuda.rs` の `pf_dma_probe` と同じ測定を **単体で** 回す。
//! 測るのは転送経路の素の速度だけで、モデルは読み込まない (引数のファイルを
//! mmap して転送元にする)。数値は `[pf-probe]` の行にそのまま出る。
//!
//! 測るもの (詳細は `crates/denno-engine/src/cuda.rs` の `pf_dma_probe`):
//! 1. WC pinned → device の DMA (単独 / GPU が memset で埋まっているとき /
//!    ホスト 8 スレッドが別の pinned へ写しているとき)
//! 2. `cuMemcpyHtoDAsync` の呼び出しがホストをブロックするか (呼び出しの µs)
//! 3. mmap を pageable のまま積んだ速度 (写しを消したときの上限)
//! 4. mmap を `cuMemHostRegister` で pin した速度と、登録の時間
//! 5. 2 本のストリームに分けた速度
//!
//! 使い方:
//!   cargo run --release -p denno-engine --features cuda --example pf_dma_probe -- \
//!       /work/models/<model>.gguf
//!   DENNO_PF_PROBE_MB=512 ... (測るバイト数、既定 256)
//!
//! 注意: GPU の無いコンテナでは `CudaCtx::new` が失敗して 1 行出して終わる
//! (数値をでっち上げない)。実測は GPU ホスト (lab) で行う。

use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).unwrap_or_default();
    if path.is_empty() {
        eprintln!(
            "usage: pf_dma_probe <file> [bytes]\n\
             転送元にするファイル (モデルの GGUF など) を mmap して測る。"
        );
        std::process::exit(2);
    }
    let want: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            std::env::var("DENNO_PF_PROBE_MB")
                .ok()
                .and_then(|s| s.trim().parse::<usize>().ok())
                .map(|mb| mb << 20)
                .unwrap_or(256 << 20)
        });
    let f = std::fs::File::open(&path)?;
    let map = unsafe { memmap2::Mmap::map(&f)? };
    let n = want.min(map.len());
    if n < (8 << 20) {
        eprintln!("{path}: 小さすぎる ({} B)", map.len());
        std::process::exit(2);
    }
    // 先に一度触ってページを載せる (初回フォルトを測定に混ぜない)。
    let t = Instant::now();
    let mut sum = 0u64;
    let mut i = 0usize;
    while i < n {
        sum = sum.wrapping_add(map[i] as u64);
        i += 4096;
    }
    eprintln!(
        "[pf-probe] src={path} map_MB={} touch_ms={:.1} (sum={})",
        map.len() >> 20,
        t.elapsed().as_secs_f64() * 1e3,
        sum & 1
    );
    let ok = denno_engine::cuda::pf_dma_probe(&map[..n]);
    if !ok {
        eprintln!("[pf-probe] 測れなかった (GPU / driver / 確保のいずれか)");
        std::process::exit(1);
    }
    Ok(())
}
