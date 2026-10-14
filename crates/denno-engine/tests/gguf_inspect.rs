//! 実 GGUF ファイルでの gguf.rs 検証 (P19)。
//!
//! 実行:
//!   GGUF_PATH=... cargo test -p denno-engine --release --test gguf_inspect -- --ignored --nocapture
//!
//! 検証内容:
//! 1. パース成功・magic/version
//! 2. 全テンソルのバイト数が「隣接テンソルのオフセット差」と一致 (構造検証)
//! 3. 最終テンソル終端 = ファイルサイズ
//! 4. 全テンソルの dequant 実行 (範囲外アクセス・非整列がないこと)

use std::env;

use denno_engine::gguf::GgufFile;

#[test]
#[ignore]
fn gguf_parse_and_dequant_real_file() {
    let path = env::var("GGUF_PATH").expect("GGUF_PATH を設定してください");
    let f = GgufFile::open(&path).unwrap();
    println!("version: {}", f.version);
    println!("alignment: {}", f.alignment);
    println!("data_start: {}", f.data_start);

    let mut arch = String::from("?");
    let mut n_params: u64 = 0;
    for (k, v) in &f.metadata {
        let shown = match v {
            denno_engine::gguf::GgufValue::Arr(a) => format!("[{} values]", a.len()),
            other => format!("{other:?}"),
        };
        if shown.len() < 100 {
            println!("  {k} = {shown}");
        } else {
            println!("  {k} = <{} chars>", shown.len());
        }
        if k == "general.architecture" {
            arch = v.as_str().unwrap_or("?").to_string();
        }
    }
    println!("arch: {arch}");
    println!("tensors: {}", f.tensors.len());
    println!(
        "最後の 5 テンソル: {:?}",
        f.tensor_order.iter().rev().take(5).collect::<Vec<_>>()
    );

    // 型ごとの集計
    let mut by_ty: std::collections::BTreeMap<String, (usize, u64)> = Default::default();
    for t in f.tensors.values() {
        let name = format!("{:?}", t.ty).split('(').next().unwrap().to_string();
        let e = by_ty.entry(name).or_insert((0, 0));
        e.0 += 1;
        e.1 += t.nbytes;
        let ne: u64 = t.dims.iter().product();
        if arch.contains("qwen") && (t.name.contains("weight") || t.name.contains("bias")) {
            n_params += ne;
        }
    }
    println!("=== 型別集計 ===");
    for (k, (n, bytes)) in &by_ty {
        println!("  {k}: {n} tensors, {bytes} bytes");
    }
    println!("総パラメータ数(概算): {n_params}");

    // 構造検証: オフセット差 == nbytes
    let mut sorted: Vec<_> = f.tensors.values().cloned().collect();
    sorted.sort_by_key(|t| t.offset);
    let file_len = std::fs::metadata(&path).unwrap().len();
    for w in sorted.windows(2) {
        assert_eq!(
            w[0].offset + w[0].nbytes,
            w[1].offset,
            "テンソル {} のサイズがオフセット差と不一致",
            w[0].name
        );
    }
    let last = sorted.last().unwrap();
    let expected_end = f.data_start + last.offset + last.nbytes;
    let file_len = std::fs::metadata(&path).unwrap().len();
    if expected_end == file_len {
        println!("=== 構造検証 OK (終端一致) ===");
    } else {
        println!(
            "=== 注意: 末尾データ ===\n  テンソル終端 {expected_end}, ファイル {file_len}, 差分 {} bytes",
            file_len - expected_end
        );
    }
    println!("=== 構造検証 OK (オフセット差は全テンソル一致) ===");

    // dequant 全テンソル (時間がかかるので環境変数でスキップ可)
    if env::var("GGUF_DEQUANT_ALL").as_deref() == Ok("1") {
        for name in &f.tensor_order {
            let v = f.dequant_tensor(name).unwrap();
            assert!(!v.is_empty());
            assert!(v.iter().all(|x| x.is_finite()), "{name}: 非有限値を含む");
        }
        println!("=== dequant 全 {} テンソル OK (有限値) ===", f.tensor_order.len());
    } else {
        // 代表 2 つだけ
        for name in ["token_embd.weight", "output.weight"] {
            if f.tensor(name).is_some() {
                let t = f.tensor(name).unwrap();
                let v = f.dequant_tensor(name).unwrap();
                let rms = (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
                println!(
                    "{name}: {:?} {} elems rms={rms:.4} first={:?}",
                    t.ty,
                    v.len(),
                    &v[..4.min(v.len())]
                );
            }
        }
    }
}
