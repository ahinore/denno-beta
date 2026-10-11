//! Hugging Face からのモデルダウンロード(pull)。
//! 単一 model.safetensors と分割 checkpoint(model.safetensors.index.json +
//! model-xxxxx-of-xxxxx.safetensors 群)の両対応。大ファイルはストリーミング保存。

use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::io::{Read, Write};
use std::path::Path;

const BASE_FILES: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
];

/// 1ファイルをダウンロードして dir に保存。
/// 既に存在すれば skip、404 なら Ok(false)(呼び出し側で任意ファイルの扱いを決める)。
fn download(agent: &ureq::Agent, repo: &str, file: &str, dir: &Path) -> Result<bool> {
    let dest = dir.join(file);
    if dest.exists() {
        println!("skip (exists): {}", file);
        return Ok(true);
    }
    let url = format!("https://huggingface.co/{}/resolve/main/{}", repo, file);
    println!("downloading {} ...", url);
    let resp = agent.get(&url).call();
    let resp = match resp {
        Ok(r) => r,
        Err(ureq::Error::Status(404, _)) => return Ok(false),
        Err(e) => return Err(e).with_context(|| format!("download {url}")),
    };

    let total = resp
        .header("content-length")
        .and_then(|v| v.parse::<u64>().ok());
    let mut reader = resp.into_reader();
    let tmp = dest.with_extension("part");
    let f = std::fs::File::create(&tmp)?;
    let mut w = std::io::BufWriter::new(f);
    let mut buf = [0u8; 64 * 1024];
    let mut written = 0u64;
    let mut next_report = 100u64; // MB
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        w.write_all(&buf[..n])?;
        written += n as u64;
        if let Some(t) = total {
            let mb = written / 1_000_000;
            if mb >= next_report {
                println!("  {}/{} MB", written / 1_000_000, t / 1_000_000);
                next_report = mb + 100;
            }
        }
    }
    w.flush()?;
    drop(w);
    std::fs::rename(&tmp, &dest)?;
    println!("  saved: {} ({:.1} MB)", dest.display(), written as f64 / 1e6);
    Ok(true)
}

pub fn pull(repo: &str, dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let agent = ureq::AgentBuilder::new().build();

    for file in BASE_FILES {
        if !download(&agent, repo, file, dir)? {
            println!("skip (404): {}", file);
        }
    }

    // 重み: model.safetensors.index.json があれば分割 checkpoint(weight_map が
    // 指す全 shard を DL)、無ければ単一 model.safetensors にフォールバック。
    let index_path = dir.join("model.safetensors.index.json");
    let sharded = if index_path.exists() {
        true
    } else if download(&agent, repo, "model.safetensors.index.json", dir)? {
        true
    } else if download(&agent, repo, "model.safetensors", dir)? {
        false
    } else {
        bail!(
            "重みファイルが見つかりません (model.safetensors / model.safetensors.index.json とも 404): {}",
            repo
        )
    };

    if sharded {
        let index: Value = serde_json::from_reader(std::fs::File::open(&index_path)?)
            .with_context(|| format!("{} の解析に失敗", index_path.display()))?;
        let weight_map = index
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| anyhow::anyhow!("index.json に weight_map がありません: {}", repo))?;
        let mut shards: Vec<&str> = weight_map.values().filter_map(|v| v.as_str()).collect();
        shards.sort_unstable();
        shards.dedup();
        println!("sharded checkpoint: {} shard(s)", shards.len());
        for shard in &shards {
            if !download(&agent, repo, shard, dir)? {
                bail!("index.json が参照する shard が 404: {} ({})", repo, shard)
            }
        }
    }
    Ok(())
}
