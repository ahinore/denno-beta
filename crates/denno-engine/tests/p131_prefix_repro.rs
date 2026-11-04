//! task 131: 継続会話で prefix cache が落ちる再現と修正の固定 (本番 conv の JSON)。
//!
//! 本番で 5 ターン会話の 3 ターン目だけ reuse=0 になった (462 トークンで
//! prefill 18.19 秒)。原因は chat_template が assistant の content を `|trim`
//! すること。2 ターン目の生成テキストが `... np.random.choice([-1, ` と空白で
//! 終わっており、3 ターン目のプロンプトではその空白が落ちるため、キャッシュの
//! 444 トークンと 443 トークン目で食い違って reuse が 0 になっていた。
//!
//! 修正 (`GgufTokenizer::splice_after_content`) は、生成済みトークンをそのまま
//! 前置きに残し、content より後ろだけを食う。ここでは実トークナイザと実
//! テンプレート (GGUF から読む) で、その前置きが全長再利用できることを固定する。
//!
//! 実モデルのトークナイザが要るので `DENNO_P131_GGUF` (既定
//! `/work/models/Qwen3.8-27B-Q4_0.gguf`) が無ければ skip する。

use denno_engine::gguf::GgufFile;
use denno_engine::tok_gguf::GgufTokenizer;
use std::path::PathBuf;

fn gguf_path() -> Option<PathBuf> {
    let p = std::env::var("DENNO_P131_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/work/models/Qwen3.8-27B-Q4_0.gguf"));
    if p.exists() {
        Some(p)
    } else {
        eprintln!("GGUF が無いので skip: {}", p.display());
        None
    }
}

fn conv_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/p131")
}

fn read_msgs(name: &str) -> Vec<(String, String)> {
    let p = conv_dir().join(name);
    let s = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let v: serde_json::Value = serde_json::from_str(&s).expect("JSON");
    v.as_array()
        .expect("messages が配列でない")
        .iter()
        .map(|m| {
            (
                m["role"].as_str().expect("role").to_string(),
                m["content"].as_str().expect("content").to_string(),
            )
        })
        .collect()
}

fn first_mismatch(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// 再現データ一式。返り値は
/// (トークナイザ, turn2 プロンプト, turn2 生成の代役, キャッシュ列, turn3 プロンプト)。
fn repro() -> Option<(GgufTokenizer, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>)> {
    let path = gguf_path()?;
    let f = GgufFile::open(&path).expect("GGUF を開けない");
    let tok = GgufTokenizer::from_gguf(&f).expect("トークナイザを組めない");
    let m2 = read_msgs("repro_py_turn2_request.json");
    let m3 = read_msgs("repro_py_turn3_request.json");
    let a2 = m3[3].1.clone();
    assert!(
        a2.ends_with("np.random.choice([-1, "),
        "再現ケースの assistant 末尾: {:?}",
        &a2[a2.len().saturating_sub(40)..]
    );
    let p2 = tok.encode_messages(None, &m2, None, false).expect("encode turn2");
    let p3 = tok.encode_messages(None, &m3, None, false).expect("encode turn3");
    // 実生成トークンは残っていないので、返送本文の encode を代役にする
    // (末尾の空白が独立した 1 トークンになることだけが要点)。
    let gen = tok.encode(&a2).expect("encode a2");
    let cache: Vec<u32> = p2.iter().chain(gen.iter()).copied().collect();
    Some((tok, p2, gen, cache, p3))
}

/// 原因の固定: 食った列と次ターンのプロンプトは「末尾の空白 1 トークン」だけが
/// 違い、そこで前置き一致が切れる (reuse=0 になる位置)。
#[test]
fn trailing_space_is_trimmed_by_template() {
    let Some((tok, p2, gen, cache, p3)) = repro() else {
        return;
    };
    let lcp = first_mismatch(&cache, &p3);
    eprintln!(
        "turn2 prompt={} gen={} cache={} turn3 prompt={} lcp={}",
        p2.len(),
        gen.len(),
        cache.len(),
        p3.len(),
        lcp
    );
    assert!(lcp < cache.len(), "前置きが完全一致してしまった (再現しない)");
    assert_eq!(lcp, cache.len() - 1, "ずれは末尾の 1 トークンだけのはず");
    let diff = &cache[lcp..];
    assert_eq!(tok.decode(diff).unwrap(), " ", "落ちたのは末尾の空白でない");
    // 落ちた先は次の assistant ブロックの終わり (`decode` は特殊トークンを
    // 落とすので、id は encode した 1 個と比べる)。
    assert_eq!(p3[lcp], tok.encode("<|im_end|>").expect("im_end")[0]);

    // 仮説の確認: content を trim して組み直すと turn3 のプロンプトと完全一致。
    let mut m3 = read_msgs("repro_py_turn3_request.json");
    let a2 = m3[3].1.clone();
    let trimmed = a2.trim_end();
    assert_ne!(trimmed, a2, "trim で変わらない本文");
    m3[3].1 = trimmed.to_string();
    let p3t = tok.encode_messages(None, &m3, None, false).expect("encode trimmed");
    assert_eq!(p3t, p3, "trim 仮説が成り立たない (原因は trim ではない)");
    assert_eq!(&p3[..lcp], &cache[..lcp], "trim 後も LCP までが一致しない");
}

/// 修正の固定: 実テンプレートで splice すると、キャッシュ全長が前置きとして
/// 残り、後続のトークンは trim 版と同じになる (= 差分 prefill だけで済む)。
#[test]
fn splice_keeps_generated_trailing_space() {
    let Some((tok, _p2, _gen, cache, p3)) = repro() else {
        return;
    };
    let m3 = read_msgs("repro_py_turn3_request.json");
    let lcp = first_mismatch(&cache, &p3);
    let spliced = tok
        .splice_after_content(None, &m3, None, false, 3, &cache)
        .expect("splice")
        .expect("repro で splice が成立しない");
    eprintln!(
        "cache={} lcp={} spliced={} (turn3 prompt={})",
        cache.len(),
        lcp,
        spliced.len(),
        p3.len()
    );
    assert_eq!(&spliced[..cache.len()], &cache[..], "キャッシュ全長が前置きでない");
    assert_eq!(&spliced[cache.len()..], &p3[lcp..], "後続が trim 版と違う");
    // 前置き 444 を再利用して食うのは 19 トークン (462 - 443) だけ。
    let fed = spliced.len() - cache.len();
    assert_eq!(fed, p3.len() - lcp, "食うトークン数");
    assert_eq!(fed, 19, "repro の後続は 19 トークン");
}

/// splice できないときは None (呼び出し側は full prefill に落ちる)。
#[test]
fn splice_rejects_other_shapes() {
    let Some((tok, _p2, _gen, cache, _p3)) = repro() else {
        return;
    };
    let m3 = read_msgs("repro_py_turn3_request.json");
    // index 3 は assistant だが、index 2 (user) を指すと描画は変わるが
    // 「content の直後」を切る位置が違うので後続が変わる。ここでは範囲外だけ固定する。
    assert!(tok
        .splice_after_content(None, &m3, None, false, 99, &cache)
        .expect("splice")
        .is_none());
}
