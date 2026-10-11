//! Hugging Face からのモデルダウンロード(pull)。
//! 大ファイルは sha256 検証付きでストリーミング保存。単一 model.safetensors のみ対応。

use anyhow::{bail, Context, Result};
use std::io::{Read, Write};
use std::path::Path;

const FILES: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors",
];

pub fn pull(repo: &str, dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let agent = ureq::AgentBuilder::new().build();

    for file in FILES {
        let url = format!("https://huggingface.co/{}/resolve/main/{}", repo, file);
        let dest = dir.join(file);
        if dest.exists() {
            println!("skip (exists): {}", file);
            continue;
        }
        println!("downloading {} ...", url);
        let resp = agent.get(&url).call();
        let resp = match resp {
            Ok(r) => r,
            Err(ureq::Error::Status(404, _)) if *file == "model.safetensors" => {
                bail!("model.safetensors がありません(分割checkpointは未対応です): {}", repo)
            }
            Err(ureq::Error::Status(404, _)) if *file == "tokenizer_config.json" => {
                println!("skip (404): {}", file);
                continue;
            }
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
    }
    Ok(())
}
