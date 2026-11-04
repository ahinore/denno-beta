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

// ---------------------------------------------------------------------------
// task 141: 何ターン前の assistant の末尾空白でも前置きを再利用する
// ---------------------------------------------------------------------------

/// 会話 1 ターン分の決定 (serve の `Flash::splice_prompt` と同じ順序)。
#[derive(Debug)]
struct Step {
    /// このターンのプロンプト長 (整列したらキャッシュ + 食い残し)
    prompt_len: usize,
    /// 前置きとして再利用したキャッシュ長
    reuse: usize,
    /// 実際に食ったトークン数
    fed: usize,
    /// このターン後のキャッシュ長 (プロンプト + 生成)
    next_cache: usize,
    /// "full" / "exact" / "align" (整列) / "splice" (sentinel 経由)
    kind: &'static str,
}

/// serve の `Cached` の代役 (プロンプト部の長さと、そのときのメッセージ列)。
struct Cache {
    toks: Vec<u32>,
    prompt_len: usize,
    msgs: Vec<(String, String)>,
}

/// 実トークナイザと実テンプレートで 1 会話を進める。
///
/// 判定は serve の `Flash::splice_prompt` と同じ順:
/// 1. キャッシュが描画の前置きと完全一致 → そのまま reuse
/// 2. 同じ会話で、返送本文が「食った生成トークンの復号」と一致
///    (クライアントが端の空白を落とした場合は許す) なら
///    a. [`GgufTokenizer::align_trimmed_prefix`] で描画と整列
///    b. 整列できなければ [`GgufTokenizer::splice_after_content`] (sentinel)
/// 3. どれも不可 → full prefill
struct Convo<'a> {
    tok: &'a GgufTokenizer,
    msgs: Vec<(String, String)>,
    cache: Option<Cache>,
}

impl<'a> Convo<'a> {
    fn new(tok: &'a GgufTokenizer) -> Self {
        Self {
            tok,
            msgs: Vec::new(),
            cache: None,
        }
    }

    /// 1 ターン進める。`reply` は「生成された本文」(実生成トークンの代役は
    /// [`GgufTokenizer::encode`]。既存テストと同じ)。
    fn turn(&mut self, user: &str, reply: &str) -> Step {
        self.msgs.push(("user".to_string(), user.to_string()));
        let (prompt, from_render) = self
            .tok
            .encode_messages_ex(None, &self.msgs, None, false)
            .expect("encode_messages_ex");
        let mut kind = "full";
        let mut reuse = 0usize;
        let mut final_prompt = prompt.clone();
        if let Some(c) = self.cache.as_ref() {
            if c.toks.len() <= prompt.len() && prompt[..c.toks.len()] == c.toks[..] {
                kind = "exact";
                reuse = c.toks.len();
            } else {
                let n = c.msgs.len();
                let echo = if self.msgs.len() > n {
                    Some(&self.msgs[n])
                } else {
                    None
                };
                let same_conversation = from_render
                    && self.msgs.len() > n
                    && self.msgs[..n] == c.msgs[..]
                    && echo.map(|m| m.0 == "assistant").unwrap_or(false);
                let echo_ok = echo
                    .map(|m| {
                        let gen = self.tok.decode(&c.toks[c.prompt_len..]).unwrap_or_default();
                        !m.1.is_empty() && !gen.is_empty() && m.1.trim() == gen.trim()
                    })
                    .unwrap_or(false);
                if same_conversation && echo_ok {
                    if let Some(i) = self.tok.align_trimmed_prefix(&c.toks, &prompt) {
                        if i < prompt.len() {
                            kind = "align";
                            reuse = c.toks.len();
                            let mut p = c.toks.clone();
                            p.extend_from_slice(&prompt[i..]);
                            final_prompt = p;
                        }
                    }
                    if kind == "full" {
                        if let Some(p) = self
                            .tok
                            .splice_after_content(None, &self.msgs, None, false, n, &c.toks)
                            .expect("splice")
                        {
                            kind = "splice";
                            reuse = c.toks.len();
                            final_prompt = p;
                        }
                    }
                }
            }
        }
        let gen = self.tok.encode(reply).expect("encode reply");
        let step = Step {
            prompt_len: final_prompt.len(),
            reuse,
            fed: final_prompt.len() - reuse,
            next_cache: final_prompt.len() + gen.len(),
            kind,
        };
        let mut toks = final_prompt;
        toks.extend(gen);
        self.cache = Some(Cache {
            toks,
            prompt_len: step.prompt_len,
            msgs: self.msgs.clone(),
        });
        self.msgs.push(("assistant".to_string(), reply.to_string()));
        step
    }
}

const FIRST_Q: &str =
    "Write Python pandas code for time series anomaly detection, with explanations.";
const FOLLOW_Q: &str = "Continue with more code.";

/// 2 ターン目と 4 ターン目の返信が末尾空白で終わる 6 ターン会話。
const REPLIES: [&str; 6] = [
    "```python\nimport pandas as pd\ndf = pd.read_csv('x.csv')\n```",
    // 2 ターン目: 末尾が空白 (本番の再現と同じ形)
    "```python\n    timestamps = pd.date_range(start, periods=n, freq='h')\n    magnitudes = np.random.choice([-1, ",
    "```python\n    df['anomaly'] = False\n    return df",
    // 4 ターン目: 末尾が空白
    "```python\n    window: If provided, use a rolling window for the z-score ",
    "```python\n    df_result['z_score'] = z\n    return df_result",
    "```python\n    return df_result.dropna()",
];

/// task 141: assistant の返信が 2 ターン目と 4 ターン目に末尾空白で終わる 6 ターン
/// 会話で、2 ターン目以降は毎ターン前置きを全長再利用する。
///
/// task 131 の splice はキャッシュに生の空白を残すので、3 ターン目以降の描画は
/// その位置 (この会話では 2 ターン目の assistant の末尾) でずれる。そのずれを
/// 吸収して食うのが「<|im_end|> 以降」だけになることを実トークナイザで固定する。
#[test]
fn six_turn_conversation_reuses_prefix_every_turn() {
    let Some((tok, _p2, _gen, _cache, _p3)) = repro() else {
        return;
    };
    let mut convo = Convo::new(&tok);
    let mut steps = Vec::new();
    for (i, reply) in REPLIES.iter().enumerate() {
        let user = if i == 0 { FIRST_Q } else { FOLLOW_Q };
        steps.push(convo.turn(user, reply));
    }
    for (i, s) in steps.iter().enumerate() {
        eprintln!("turn{}: {:?}", i + 1, s);
    }
    // 1 ターン目だけ full prefill。
    assert_eq!(steps[0].kind, "full");
    assert_eq!(steps[0].reuse, 0);
    // 2 ターン目以降は毎ターン、直前のキャッシュ全長を前置きに使う。
    let mut cache_len = steps[0].next_cache;
    for (i, s) in steps.iter().enumerate().skip(1) {
        assert_eq!(
            s.reuse,
            cache_len,
            "turn{} で前置きを全長再利用していない ({:?})",
            i + 1,
            s
        );
        // 食うのは「<|im_end|> + 次の user + assistant ヘッダ」= 19 トークン前後。
        assert!(s.fed > 0 && s.fed <= 24, "食う量がおかしい: {s:?}");
        cache_len = s.next_cache;
    }
    // 2 ターン目は末尾空白が無いので前置き一致。3 ターン目以降は古い末尾空白を
    // 吸収する整列経路 (5 ターン目は 4 ターン目の末尾空白も同時に吸収する)。
    assert_eq!(steps[1].kind, "exact");
    for s in steps.iter().skip(2) {
        assert_eq!(s.kind, "align", "3 ターン目以降が整列経路でない: {s:?}");
    }
    // 最後の描画に対して、キャッシュは 2 トークン (2・4 ターン目の末尾空白) だけ長い。
    let cache = convo.cache.as_ref().expect("cache");
    let p = tok
        .encode_messages_ex(None, &convo.msgs, None, false)
        .unwrap()
        .0;
    let i = tok
        .align_trimmed_prefix(&cache.toks, &p)
        .expect("会話の終端で整列できない");
    assert_eq!(
        cache.toks.len() - i,
        2,
        "吸収したのが 2・4 ターン目の末尾空白 2 個でない (cache={} i={})",
        cache.toks.len(),
        i
    );
}

/// task 141: 2 ターン目の末尾空白 (本番と同じ 1 トークン) を、3 ターン目以降の
/// どの描画でも吸収する (キャッシュの方が 1 トークン長いまま)。
#[test]
fn older_trim_is_absorbed_on_every_later_turn() {
    let Some((tok, _p2, _gen, _cache, _p3)) = repro() else {
        return;
    };
    let replies = [
        "```python\nimport pandas as pd\n```",
        "```python\n    magnitudes = np.random.choice([-1, ",
        "```python\n    return df",
        "```python\n    return df_result",
    ];
    let mut convo = Convo::new(&tok);
    let mut steps = Vec::new();
    for (i, reply) in replies.iter().enumerate() {
        let user = if i == 0 { FIRST_Q } else { FOLLOW_Q };
        steps.push(convo.turn(user, reply));
    }
    for s in steps.iter().skip(2) {
        assert_eq!(s.kind, "align", "{s:?}");
        assert!(s.fed > 0, "{s:?}");
    }
    // 3 ターン目と 4 ターン目の食う量は同じ (古い空白 1 個を毎回吸収するだけ)。
    assert_eq!(steps[2].fed, steps[3].fed, "食う量がターンで変わった");
    // キャッシュは 3 ターン目以降も「描画より 1 トークン長い」まま。
    let cache = convo.cache.as_ref().expect("cache");
    let p = tok
        .encode_messages_ex(None, &convo.msgs, None, false)
        .unwrap()
        .0;
    let i = tok.align_trimmed_prefix(&cache.toks, &p).expect("整列");
    assert_eq!(cache.toks.len() - i, 1, "吸収したのが末尾空白 1 個でない");
}

/// task 141: assistant の content が空白で *始まる* と `|trim` は先頭も落とす。
/// トークン整列では吸収できないので sentinel 経路に落ちるが、前置きの再利用は
/// 続く (task 131 の経路がそのまま効く)。
#[test]
fn leading_trim_falls_back_to_sentinel_splice() {
    let Some((tok, _p2, _gen, _cache, _p3)) = repro() else {
        return;
    };
    let replies = [
        "```python\nimport pandas as pd\n```",
        "\n```python\n    magnitudes = np.random.choice([-1, ",
        "```python\n    return df",
    ];
    let mut convo = Convo::new(&tok);
    let mut steps = Vec::new();
    for (i, reply) in replies.iter().enumerate() {
        let user = if i == 0 { FIRST_Q } else { FOLLOW_Q };
        steps.push(convo.turn(user, reply));
    }
    for (i, s) in steps.iter().enumerate() {
        eprintln!("turn{}: {:?}", i + 1, s);
    }
    // 先頭に改行のある 2 ターン目の返信 → 3 ターン目は整列できず sentinel。
    assert_eq!(steps[2].kind, "splice", "{:?}", steps[2]);
    // 4 ターン目も前置きは全長再利用できる (キャッシュは sentinel の結果)。
    let mut cache_len = steps[0].next_cache;
    for (i, s) in steps.iter().enumerate().skip(1) {
        assert_eq!(s.reuse, cache_len, "turn{}: {:?}", i + 1, s);
        cache_len = s.next_cache;
    }
}

/// task 141: 別の会話はキャッシュに整列しない (誤って前置きを使わない)。
///
/// - 最初の user が違う
/// - 同じ履歴だが assistant の本文が違う (単語の置換)
/// - user 本文の空白だけが違う (content の途中の空白は `|trim` の対象外)
#[test]
fn other_conversations_are_not_aligned() {
    let Some((tok, _p2, _gen, _cache, _p3)) = repro() else {
        return;
    };
    let replies = [
        "```python\nimport pandas as pd\n```",
        "```python\n    magnitudes = np.random.choice([-1, ",
        "```python\n    return df",
    ];
    let mut convo = Convo::new(&tok);
    for (i, reply) in replies.iter().enumerate() {
        let user = if i == 0 { FIRST_Q } else { FOLLOW_Q };
        convo.turn(user, reply);
    }
    let cache = convo.cache.as_ref().expect("cache");
    // この会話自身は整列できる (前提の確認)。
    let p = tok
        .encode_messages_ex(None, &convo.msgs, None, false)
        .unwrap()
        .0;
    assert!(
        tok.align_trimmed_prefix(&cache.toks, &p).is_some(),
        "この会話自身が整列できない"
    );

    // (a) 最初の user が違う会話。
    let mut other = convo.msgs.clone();
    other[0].1 = "Explain the economic history of the Hanseatic League in great detail.".into();
    let p = tok.encode_messages_ex(None, &other, None, false).unwrap().0;
    assert!(
        tok.align_trimmed_prefix(&cache.toks, &p).is_none(),
        "別の会話がキャッシュに整列した"
    );

    // (b) 同じ履歴だが assistant の本文が違う (単語の置換)。
    let mut changed = convo.msgs.clone();
    let ai = 3; // 2 ターン目の assistant (末尾空白を持つ古い返信)
    assert_eq!(changed[ai].0, "assistant");
    assert!(changed[ai].1.contains("magnitudes"), "置換対象が無い");
    changed[ai].1 = changed[ai].1.replace("magnitudes", "amplitudes");
    assert_ne!(changed[ai].1, convo.msgs[ai].1, "置換が効いていない");
    let p = tok.encode_messages_ex(None, &changed, None, false).unwrap().0;
    assert!(
        tok.align_trimmed_prefix(&cache.toks, &p).is_none(),
        "本文が違う会話がキャッシュに整列した"
    );

    // (c) user 本文の空白だけが違う (content の途中の空白は吸収しない)。
    let mut spaced = convo.msgs.clone();
    spaced[1].1 = "Continue  with more code.".into();
    let p = tok.encode_messages_ex(None, &spaced, None, false).unwrap().0;
    assert!(
        tok.align_trimmed_prefix(&cache.toks, &p).is_none(),
        "本文の空白違いが吸収された (content の途中まで吸収している)"
    );
}

/// task 141: 空白以外の不一致は full prefill に落ちる (整列しない)。
#[test]
fn non_whitespace_mismatch_needs_full_prefill() {
    let Some((tok, _p2, gen, cache, p3)) = repro() else {
        return;
    };
    // 本番の再現: キャッシュ末尾の 1 トークン (空白) だけが違う → 吸収できる。
    let at = first_mismatch(&cache, &p3);
    assert_eq!(at, cache.len() - 1, "再現ケースのずれが 1 トークンでない");
    let i = tok
        .align_trimmed_prefix(&cache, &p3)
        .expect("末尾空白の吸収");
    assert_eq!(i, at, "食い残しの開始位置が LCP でない");
    assert_eq!(&p3[i..], &p3[at..], "食う列が描画の食い残しでない");

    // 生成の途中のトークンを空白でないトークンに変える → 整列しない。
    let z = tok.encode("Z").expect("encode Z")[0];
    assert_ne!(z, cache[at], "Z がキャッシュ末尾の空白と同じ id");
    assert!(
        !tok.decode(&[z]).unwrap().trim().is_empty(),
        "Z が空白トークン"
    );
    let mut broken = cache.clone();
    let mid = cache.len() - gen.len() / 2;
    broken[mid] = z;
    assert!(
        tok.align_trimmed_prefix(&broken, &p3).is_none(),
        "非空白の不一致を吸収した"
    );

    // 空白を 2 トークンに増やしても、content の終端が続くなら吸収する
    // (連続空白はテンプレートの |trim がまとめて落とす)。
    let mut two = cache.clone();
    let ws = cache[cache.len() - 1];
    two.push(ws);
    let i = tok.align_trimmed_prefix(&two, &p3).expect("空白 2 個の吸収");
    assert_eq!(i, at, "空白 2 個の食い残し開始位置");
}
