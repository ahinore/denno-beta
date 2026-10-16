//! tool use: `tool_choice: any` / `{"type":"tool","name":...}` のトークンレベル拘束デコード (第21セッション)。
//!
//! サンプル直前に logits へマスク (許容集合以外を -inf) を適用することで、
//! モデル出力を「公式テンプレートが tool use をモデルに出力させる XML 形式」へ拘束する:
//!
//! ```text
//! 自由テキスト ... <tool_call>
//! ```
//!
//! Watch は OPEN の出現を監視する状態。byte-level BPE (実モデル) では OPEN が
//! 複数トークンに跨っても KMP 風の前進で追跡でき、WordLevel 系 (ダミーテスト用)
//! では OPEN が 1 トークンに入るケースも伸長で扱える。
//! 拘束の対象は **tool 名のみ** (arguments は自由。schema 拘束は将来課題)。
use crate::tokenizer::DennoTokenizer;

/// 拘束対象のツール名群。`tool_choice` の Named なら 1 件、`any` なら全定義ツール。
#[derive(Debug, Clone)]
pub struct ToolConstraint {
    pub names: Vec<String>,
}

/// 公式テンプレートがモデルに出力させる tool_call block の開始リテラル
pub(crate) const MASK_OPEN: &[u8] = b"<tool_call>";
/// block 内の強制スキャフォールド (open の直後からツール名まで)
const MASK_HEAD: &[u8] = b"\n<function=";
/// ツール名の直後に強制される 1 バイト
const MASK_GT: u8 = b'>';
/// block 本文の終端 (これで block 完了 = 拘束充足)
const MASK_CLOSE: &[u8] = b"</tool_call>";

/// 拘束オートマトンの状態。バイト列の消費に対する遷移のみを持つ。
#[derive(Debug, Clone, PartialEq, Eq)]
enum MaskState {
    /// 自由テキスト中。`` の k バイト目まで一致 (KMP・部分一致を跨ぐ)。
    Watch { k: usize },
    /// ` MASK_HEAD の k バイト目まで確定 (強制リテラル)。
    Head { k: usize },
    /// ツール名構築中。`chosen` は確定済みの名前接頭辞 (全候補名の共通接頭辞
    /// とは限らない — 候補の 1 つへの接頭辞)。`chosen` が候補名と完全一致して
    /// いれば `>` で本文へ抜けられる。
    Name { chosen: Vec<u8> },
    /// block 本文 (arguments 領域・自由)。` MASK_CLOSE の k バイト目まで一致 (KMP)。
    Body { k: usize },
    /// block 完了 (拘束充足)。以降は無マスク (自由。追加の tool_call も可)。
    Done,
}

/// 1 ツールの拘束状態 (ChatSession が持つ)。
#[derive(Debug)]
pub(crate) struct ToolMask {
    /// 許容するツール名 (バイト列・名前に `>` を含むものは除外 — 抽出器の
    /// `<function=NAME>` 規約と両立しないため)
    names: Vec<Vec<u8>>,
    state: MaskState,
    /// 許容集合キャッシュ (状態が変わったときだけ再計算)
    cache: Option<(MaskState, Vec<u32>)>,
}

impl ToolMask {
    /// 拘束を構築する。`names` が空 (実質拘束対象なし) なら None。
    pub(crate) fn new(names: Vec<String>) -> Option<Self> {
        let names: Vec<Vec<u8>> = names
            .into_iter()
            .filter(|n| !n.is_empty() && !n.contains('>'))
            .map(|n| n.into_bytes())
            .collect();
        if names.is_empty() {
            return None;
        }
        Some(Self { names, state: MaskState::Watch { k: 0 }, cache: None })
    }

    /// 拘束が充足済み (block 完了) か。
    pub(crate) fn satisfied(&self) -> bool {
        self.state == MaskState::Done
    }

    /// サンプル直前のマスク適用。許容集合以外の logits を -inf にする。
    /// 拘束完了後 (Done) は何もしない = 無マスク経路とビット同一。
    /// EOS は拘束未充足の間だけ許容集合から除外する (強制デコード)。
    pub(crate) fn apply(&mut self, logits: &mut [f32], tok: &DennoTokenizer, eos_ids: &[u32]) {
        if self.satisfied() {
            return;
        }
        let need = match &self.cache {
            Some((st, _)) => st != &self.state,
            None => true,
        };
        if need {
            let allowed = self.compute_allowed(tok, eos_ids);
            self.cache = Some((self.state.clone(), allowed));
        }
        let (_, allowed) = self.cache.as_ref().unwrap();
        if allowed.is_empty() {
            // 許容トークンが 1 つも無い (語彙が足りない等)。マスクすると
            // 全確率 0 で壊れるため緩めて続行する (拘束は実質解除・警告のみ)。
            tracing::warn!("tool mask: no allowed token for state {:?}, relaxing", self.state);
            return;
        }
        // 許容集合 (昇順) 以外を -inf へ。O(V + A)・アロケーション無し。
        let mut ai = 0usize;
        for (i, l) in logits.iter_mut().enumerate() {
            let id = i as u32;
            while ai < allowed.len() && allowed[ai] < id {
                ai += 1;
            }
            if ai < allowed.len() && allowed[ai] == id {
                ai += 1;
                continue;
            }
            *l = f32::NEG_INFINITY;
        }
    }

    /// 現在の状態で許容されるトークン id の昇順リスト。
    fn compute_allowed(&self, tok: &DennoTokenizer, eos_ids: &[u32]) -> Vec<u32> {
        let mut out = Vec::new();
        for id in 0..tok.vocab_size() as u32 {
            if !self.satisfied() && eos_ids.contains(&id) {
                continue;
            }
            let Some(piece) = tok.piece_bytes(id) else { continue };
            let mut probe = self.state.clone();
            if piece.iter().all(|&b| feed_byte(&mut probe, &self.names, b).is_ok()) {
                out.push(id);
            }
        }
        out
    }

    /// サンプル済みトークンを状態機械へ供給する (生成テキストの前進)。
    /// apply と同じ許容規則でしかサンプルされないため原則として失敗しない
    /// (失敗時は状態を Done に落として拘束を解除する — 無限ループ防止)。
    pub(crate) fn feed(&mut self, tok: &DennoTokenizer, id: u32) {
        if let Some(piece) = tok.piece_bytes(id) {
            for &b in &piece {
                if feed_byte(&mut self.state, &self.names, b).is_err() {
                    tracing::warn!("tool mask: dead end at state {:?}, relaxing", self.state);
                    self.state = MaskState::Done;
                    break;
                }
            }
            self.cache = None;
        }
    }
}

/// 1 バイトの状態遷移。`Ok` なら遷移済み (状態は in-place 更新)、
/// `Err` はそのバイトで死ぬ (許容集合から外れるべきトークン)。
fn feed_byte(state: &mut MaskState, names: &[Vec<u8>], b: u8) -> Result<(), ()> {
    match state {
        MaskState::Watch { k } => {
            // KMP 風の前進: まず伸長を試み、失敗したら「MASK_OPEN[..k] + b」の
            // 最長 suffix が MASK_OPEN の prefix になる長さへ落ちる。
            if MASK_OPEN.get(*k) == Some(&b) {
                let k2 = *k + 1;
                if k2 == MASK_OPEN.len() {
                    *state = MaskState::Head { k: 0 };
                } else {
                    *state = MaskState::Watch { k: k2 };
                }
                return Ok(());
            }
            // token 境界を跨ぐ完全一致 (単一トークンに `` 全体が入る場合)
            // は feed() 側で ends_with を見るので、ここでは fallback のみ。
            let nk = kmp_fallback(MASK_OPEN, *k, b);
            *state = MaskState::Watch { k: nk };
            Ok(())
        }
        MaskState::Head { k } => {
            if MASK_HEAD.get(*k) != Some(&b) {
                return Err(());
            }
            let k2 = *k + 1;
            if k2 == MASK_HEAD.len() {
                *state = MaskState::Name { chosen: Vec::new() };
            } else {
                *state = MaskState::Head { k: k2 };
            }
            Ok(())
        }
        MaskState::Name { chosen } => {
            if b == MASK_GT {
                // 名前が候補と完全一致していれば本文へ (名前確定)
                if names.iter().any(|n| n.as_slice() == chosen.as_slice()) {
                    *state = MaskState::Body { k: 0 };
                    return Ok(());
                }
                return Err(());
            }
            // 候補名の絞り込み: chosen + b が何らかの候補の接頭辞であること
            let mut next = chosen.clone();
            next.push(b);
            if names.iter().any(|n| n.starts_with(&next)) {
                *state = MaskState::Name { chosen: next };
                Ok(())
            } else {
                Err(())
            }
        }
        MaskState::Body { k } => {
            if MASK_CLOSE.get(*k) == Some(&b) {
                let k2 = *k + 1;
                if k2 == MASK_CLOSE.len() {
                    *state = MaskState::Done;
                } else {
                    *state = MaskState::Body { k: k2 };
                }
                return Ok(());
            }
            let nk = kmp_fallback(MASK_CLOSE, *k, b);
            *state = MaskState::Body { k: nk };
            Ok(())
        }
        MaskState::Done => Ok(()),
    }
}

/// パターン照合の fallback: 「pattern[..k] + b」の最長 suffix が pattern の
/// prefix である長さを返す (見つからなければ 0)。
/// MASK_OPEN / MASK_CLOSE は自己重複を持たないが、一般形で書いておく。
fn kmp_fallback(pattern: &[u8], k: usize, b: u8) -> usize {
    // 伸長失敗時: k を縮めながら pattern[..k'] + b が pattern[..k'+1] と
    // 一致する (= b で伸長できる) k' を探す。pattern が短いため単純走査で十分。
    let mut kk = k;
    loop {
        if pattern.get(kk) == Some(&b) {
            return kk + 1;
        }
        if kk == 0 {
            return 0;
        }
        kk -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// テスト用の最小トークナイザ (byte-level decoder + testutil 共有語彙)。
    fn tok_with_voc() -> DennoTokenizer {
        crate::testutil::tool_tokenizer().unwrap()
    }

    fn tok_with(vocab: &[(&str, u32)]) -> DennoTokenizer {
        DennoTokenizer::testutil_vocab_tokenizer(
            vocab,
            Some(serde_json::json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true})),
        )
        .unwrap()
    }

    fn vocab() -> Vec<(&'static str, u32)> {
        crate::testutil::TOOL_VOCAB.iter().copied().collect()
    }

    fn mask(names: &[&str]) -> ToolMask {
        ToolMask::new(names.iter().map(|s| s.to_string()).collect()).unwrap()
    }

    /// トークンを 1 個サンプルした相当の前進 (feed)
    fn feed_id(m: &mut ToolMask, tok: &DennoTokenizer, s: &str) {
        let id = tok.encode(s).unwrap()[0];
        m.feed(tok, id);
    }

    fn allowed(m: &mut ToolMask, tok: &DennoTokenizer) -> Vec<String> {
        let list = m.compute_allowed(tok, &[12]);
        list.iter()
            .filter_map(|&id| tok.id_to_token_str(id))
            .collect()
    }

    #[test]
    fn watch_state_allows_everything_except_eos() {
        let tok = tok_with(&vocab());
        let mut m = mask(&["get_weather"]);
        let a = allowed(&mut m, &tok);
        assert!(a.contains(&"<tool_call>".to_string()));
        assert!(a.contains(&"t0".to_string()));
        assert!(a.contains(&"get_weather".to_string()));
        // EOS は拘束未充足の間は禁止 (強制デコード)
        assert!(!a.contains(&"eos".to_string()));
        // 状態は変わらない (apply/feed でのみ前進)
        assert_eq!(m.state, MaskState::Watch { k: 0 });
    }

    #[test]
    fn scaffold_is_forced_byte_wise() {
        let tok = tok_with(&vocab());
        let mut m = mask(&["get_weather", "run_tool"]);
        // <tool_call> を観測 → Head (<tool_call> 完了)
        feed_id(&mut m, &tok, "<tool_call>");
        assert_eq!(m.state, MaskState::Head { k: 0 });
        // Head の許容集合は \n<function= の接頭辞トークンのみ
        let a = allowed(&mut m, &tok);
        assert_eq!(a, vec!["Ċ".to_string()]);
        feed_id(&mut m, &tok, "Ċ");
        assert_eq!(m.state, MaskState::Head { k: 1 });
        let a = allowed(&mut m, &tok);
        // "<" 単体 (Head 接頭辞の継続) も許容される — byte-level BPE では
        // "<function=" が "<" + "function=" 等の複数トークンに分かれ得るため
        assert_eq!(a, vec!["<".to_string(), "<function=".to_string()]);
        feed_id(&mut m, &tok, "<function=");
        // 名前構築へ
        assert_eq!(m.state, MaskState::Name { chosen: Vec::new() });
        // 名前候補トークンのみ許容
        let a = allowed(&mut m, &tok);
        assert!(a.contains(&"get_weather".to_string()));
        assert!(a.contains(&"run_tool".to_string()));
        assert!(!a.contains(&">".to_string()), "空名前での > は不可");
        feed_id(&mut m, &tok, "get_weather");
        assert_eq!(m.state, MaskState::Name { chosen: b"get_weather".to_vec() });
        // 名前確定後は > のみ
        let a = allowed(&mut m, &tok);
        assert_eq!(a, vec![">".to_string()]);
        feed_id(&mut m, &tok, ">");
        assert_eq!(m.state, MaskState::Body { k: 0 });
    }

    #[test]
    fn named_tool_blocks_other_names() {
        let tok = tok_with(&vocab());
        let mut m = mask(&["get_weather"]);
        for s in ["<tool_call>", "Ċ", "<function="] {
            feed_id(&mut m, &tok, s);
        }
        let a = allowed(&mut m, &tok);
        assert!(a.contains(&"get_weather".to_string()));
        assert!(!a.contains(&"run_tool".to_string()), "Named は指定外ツール名を禁止");
    }

    #[test]
    fn body_is_free_until_close_then_done() {
        let tok = tok_with(&vocab());
        let mut m = mask(&["get_weather"]);
        for s in ["<tool_call>", "Ċ", "<function=", "get_weather", ">"] {
            feed_id(&mut m, &tok, s);
        }
        // 本文は自由 + EOS はまだ禁止
        let a = allowed(&mut m, &tok);
        assert!(a.contains(&"t0".to_string()));
        assert!(a.contains(&"``".to_string()));
        assert!(!a.contains(&"eos".to_string()));
        // close の先頭 1 バイト (<) を消費 (close 途中)
        feed_id(&mut m, &tok, "<");
        assert_eq!(m.state, MaskState::Body { k: 1 });
        // 途中で別トークンに逸れても KMP fallback で復帰する
        feed_id(&mut m, &tok, "t0");
        assert_eq!(m.state, MaskState::Body { k: 0 });
        // close を 1 トークンで完成 (close は open 先頭に "/" が付いた別リテラル)
        feed_id(&mut m, &tok, "</tool_call>");
        assert_eq!(m.state, MaskState::Done);
        // 完了後は EOS も許可 (無マスク)
        let a = allowed(&mut m, &tok);
        assert!(a.contains(&"eos".to_string()));
        assert!(m.satisfied());
    }

    #[test]
    fn close_suffix_split_across_tokens() {
        // CLOSE ("</tool_call>") をバイト単位 (複数トークン) に分けても Done に至る
        let tok = tok_with(&vocab());
        let mut m = mask(&["get_weather"]);
        for s in ["<tool_call>", "Ċ", "<function=", "get_weather", ">"] {
            feed_id(&mut m, &tok, s);
        }
        assert_eq!(m.state, MaskState::Body { k: 0 });
        // close を 1 バイトずつ (< / tool_call は語彙に無いため "<" → 残りは piece 直送で代用)
        feed_id(&mut m, &tok, "<");
        assert_eq!(m.state, MaskState::Body { k: 1 });
        for &b in b"/tool_call>" {
            feed_byte(&mut m.state, &m.names, b).unwrap();
        }
        assert_eq!(m.state, MaskState::Done);
        // 逸脱からの復帰: close 途中の "<" のあとに本文に戻る
        let mut m = mask(&["get_weather"]);
        for s in ["<tool_call>", "Ċ", "<function=", "get_weather", ">", "<"] {
            feed_id(&mut m, &tok, s);
        }
        assert_eq!(m.state, MaskState::Body { k: 1 });
        feed_id(&mut m, &tok, "t0");
        assert_eq!(m.state, MaskState::Body { k: 0 });
    }

    #[test]
    fn open_literal_can_arrive_in_pieces() {
        // OPEN を 2 トークン (< + tool_call>) に分けても Watch が追跡する
        // (byte-level BPE で <tool_call> がトークン境界を跨ぐケース)。
        // TOOL_VOCAB には中間断片が無いため専用語彙を使う。
        let tok = tok_with(&[("t0", 0), ("<", 1), ("tool_call>", 2)]);
        let mut m = mask(&["get_weather"]);
        feed_id(&mut m, &tok, "<");
        assert_eq!(m.state, MaskState::Watch { k: 1 });
        // 逸脱: 't' で伸長 (k=2) した後 '0' で失速 → KMP fallback で 0 へ
        feed_id(&mut m, &tok, "t0");
        assert_eq!(m.state, MaskState::Watch { k: 0 });
        // 改めて 2 トークン (< + tool_call>) で完成 → Head へ
        feed_id(&mut m, &tok, "<");
        feed_id(&mut m, &tok, "tool_call>");
        assert_eq!(m.state, MaskState::Head { k: 0 });
    }

    #[test]
    fn whole_open_in_single_token() {
        // WordLevel 系では OPEN が 1 トークンで来る (byte-level なら複数トークン)。
        // TOOL_VOCAB の <tool_call> トークン (id 22) 1 個で Head へ直接遷移する。
        let tok = tok_with(&vocab());
        let mut m = mask(&["get_weather"]);
        feed_id(&mut m, &tok, "<tool_call>");
        assert_eq!(m.state, MaskState::Head { k: 0 });
    }

    #[test]
    fn name_split_mid_multibyte_is_handled() {
        // マルチバイト名を 1 文字 (= 複数バイト) ずつのトークンで構築する。
        // byte-level BPE 相当: テストでは raw バイトの piece を直接供給する。
        let names = vec!["天気".as_bytes().to_vec()];
        let mut state = MaskState::Name { chosen: Vec::new() };
        for &b in "天".as_bytes() {
            feed_byte(&mut state, &names, b).unwrap();
        }
        assert_eq!(state, MaskState::Name { chosen: "天".as_bytes().to_vec() });
        for &b in "気".as_bytes() {
            feed_byte(&mut state, &names, b).unwrap();
        }
        assert_eq!(state, MaskState::Name { chosen: "天気".as_bytes().to_vec() });
        feed_byte(&mut state, &names, b'>').unwrap();
        assert_eq!(state, MaskState::Body { k: 0 });
        // 異なる文字で始まるバイトは弾かれる
        let mut state = MaskState::Name { chosen: Vec::new() };
        assert!(feed_byte(&mut state, &names, b'a').is_err());
    }

    #[test]
    fn names_with_gt_or_empty_are_dropped() {
        // `>` を含む名 / 空名は拘束対象から除外される
        assert!(ToolMask::new(vec!["a>b".into()]).is_none());
        assert!(ToolMask::new(vec![String::new()]).is_none());
        assert!(ToolMask::new(vec![]).is_none());
        let m = ToolMask::new(vec!["ok".into(), "bad>x".into()]).unwrap();
        assert_eq!(m.names, vec![b"ok".to_vec()]);
    }
}
