// task 347: `/metrics` の `denno_build_info` に載せるビルド SHA。
//
// `git rev-parse --short=12 HEAD` をビルド時に 1 回走らせて環境変数で渡す。
// git が無い / リポジトリでない (crates.io からのビルドなど) ときは "unknown"。
// `rerun-if-changed` は出さない: 既定の「パッケージ内のファイルが変わったら
// 走る」に任せる (バイナリが変わるときは必ず走るので SHA が古くならない)。
use std::process::Command;

fn main() {
    let sha = Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=DENNO_BUILD_SHA={sha}");
}
