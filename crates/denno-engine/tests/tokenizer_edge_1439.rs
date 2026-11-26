//! トークナイザの境界・異常系テスト (task 1439)。
//!
//! 公開 API (`encode` / `decode` / `piece_bytes` / `vocab_size`) のみで観測できる
//! 不変条件を固定する。対象は:
//! - 空文字列・空白のみの入力
//! - マルチバイト UTF-8 (日本語・絵文字) が語彙外のときの [UNK] 縮退
//! - 語彙外 / 範囲外トークン ID の decode (panic / Err にしない)
//! - added token (special-token 相当) の encode/decode 経路
//!
//! CPU-only・外部モデルファイル不要・決定的。ダミー語彙は
//! `testutil` の WordLevel トークナイザ (未知語 -> [UNK]) を使う。
//! 語彙に無いバイト列は復元できないため、マルチバイトの roundtrip は
//! 「[UNK] へ縮退する (lossy)」ことを固定する (byte-level BPE の完全 roundtrip
//! は実モデル側の責務)。

use denno_engine::testutil::{dummy_tokenizer, qwen_style_tokenizer};

// ------------------------------------------------------------ 空入力

#[test]
fn empty_input_encodes_to_empty_token_list() {
    let tok = dummy_tokenizer().unwrap();
    assert_eq!(tok.encode("").unwrap(), Vec::<u32>::new());
    // 空白のみ (Whitespace 前処理でトークンが残らない)
    assert_eq!(tok.encode("   \t\n\r\n  ").unwrap(), Vec::<u32>::new());
}

#[test]
fn empty_token_list_decodes_to_empty_string() {
    let tok = dummy_tokenizer().unwrap();
    assert_eq!(tok.decode(&[]).unwrap(), "");
}

// ------------------------------------------------------------ マルチバイト UTF-8

#[test]
fn japanese_text_outside_vocab_maps_to_unk_without_panic() {
    let tok = dummy_tokenizer().unwrap();
    // 空白区切りの 2 語 -> それぞれ [UNK] (id 20)
    assert_eq!(tok.encode("こんにちは 世界").unwrap(), vec![20, 20]);
    // 語彙内トークンと混在しても既知部分は保たれる
    assert_eq!(tok.encode("t0 こんにちは t1").unwrap(), vec![0, 20, 1]);
}

#[test]
fn emoji_outside_vocab_maps_to_unk_without_panic() {
    let tok = dummy_tokenizer().unwrap();
    // 空白を含まない絵文字列は 1 語として扱われる
    assert_eq!(tok.encode("🙂🙂").unwrap(), vec![20]);
    // 4 バイト UTF-8 (U+1F642) でも encode/decode が panic しない
    let ids = tok.encode("🙂").unwrap();
    assert_eq!(ids, vec![20]);
    assert!(tok.decode(&ids).is_ok());
}

#[test]
fn multibyte_roundtrip_is_lossy_via_unk_but_consistent() {
    let tok = dummy_tokenizer().unwrap();
    let ids = tok.encode("こんにちは").unwrap();
    assert_eq!(ids, vec![20]);
    // [UNK] の表示文字列に縮退する (元の文字列には戻らない)
    let back = tok.decode(&ids).unwrap();
    assert_eq!(back, "[UNK]");
    // 再度エンコードしても語彙外 -> [UNK] に落ちる (panic しない・決定的)
    let again = tok.encode(&back).unwrap();
    assert!(!again.is_empty());
    assert!(again.iter().all(|&id| id == 20), "再エンコードが [UNK] 以外: {again:?}");
}

// ------------------------------------------------------------ 語彙外 / 範囲外 ID

#[test]
fn decode_of_out_of_range_id_is_empty_not_error() {
    let tok = dummy_tokenizer().unwrap();
    // 語彙長を大きく超える ID は空文字列へ縮退する (Err にしない契約)
    assert_eq!(tok.decode(&[9_999]).unwrap(), "");
    assert_eq!(tok.decode(&[u32::MAX]).unwrap(), "");
    // 有効 ID と混在しても有効部分だけが残る
    assert_eq!(tok.decode(&[0, 9_999, 1]).unwrap().split_whitespace().count(), 2);
}

#[test]
fn piece_bytes_of_out_of_range_id_is_none() {
    let tok = dummy_tokenizer().unwrap();
    assert_eq!(tok.piece_bytes(9_999), None);
    assert_eq!(tok.piece_bytes(u32::MAX), None);
}

#[test]
fn decode_never_panics_across_id_sweep() {
    // 語彙の疎な ID (16..19 は欠番) も含めて全 ID を走査し、
    // decode が Err / panic を返さないことを固定する
    for tok in [dummy_tokenizer().unwrap(), qwen_style_tokenizer().unwrap()] {
        for id in (0u32..=40).chain([1_000, u32::MAX]) {
            let _decoded = tok.decode(&[id]).expect("decode returned Err");
            let _ = tok.piece_bytes(id);
        }
    }
}

#[test]
fn plain_vocab_ids_decode_nonempty_and_reencode_identically() {
    // t0..t11 + eos は語彙の通常語。decode -> encode が恒等になる。
    let tok = dummy_tokenizer().unwrap();
    for id in 0u32..=12 {
        let text = tok.decode(&[id]).unwrap();
        assert!(!text.is_empty(), "id {id} の decode が空");
        assert_eq!(tok.encode(&text).unwrap(), vec![id], "id {id} の再エンコードが不一致");
    }
}

#[test]
fn piece_bytes_matches_single_id_decode_utf8() {
    let tok = qwen_style_tokenizer().unwrap();
    for id in (0u32..=22).filter(|&id| tok.piece_bytes(id).is_some()) {
        let bytes = tok.piece_bytes(id).unwrap();
        let decoded = tok.decode(&[id]).unwrap();
        assert_eq!(bytes, decoded.as_bytes(), "id {id} の piece_bytes と decode が不一致");
    }
}

// ------------------------------------------------------------ added token

#[test]
fn added_token_ids_decode_to_their_content() {
    // decoder 無しのため added token の前後空白は decode 時に落ちる。
    // 内容そのものが復元され、panic しないことを固定する。
    let tok = qwen_style_tokenizer().unwrap();
    assert_eq!(tok.decode(&[16]).unwrap(), "<|im_start|>");
    assert_eq!(tok.decode(&[17]).unwrap(), "<|im_end|>");
    assert_eq!(tok.decode(&[20]).unwrap(), "<think>");
    assert_eq!(tok.decode(&[22]).unwrap(), "</think>");
    // 連続 ID も panic せず両方の内容を含む (decoder 無しのため空白区切り)
    let joined = tok.decode(&[16, 17]).unwrap();
    assert!(joined.contains("<|im_start|>") && joined.contains("<|im_end|>"));
}

#[test]
fn added_token_embedded_in_text_is_matched() {
    let tok = qwen_style_tokenizer().unwrap();
    // added_tokens は前処理より先に照合されるため、語中でも 1 トークンになる
    assert_eq!(tok.encode("a</think>b").unwrap(), vec![21, 22, 21]);
}

#[test]
fn special_like_text_absent_from_added_tokens_falls_back_to_unk() {
    let tok = qwen_style_tokenizer().unwrap();
    // added_tokens 未登録の `<|im_start|>` 表記は語彙外 -> [UNK] (id 21) へ縮退。
    // decode 側では id 16 から同じ表記が復元できる (非対称だが panic しない)。
    let ids = tok.encode("<|im_start|>").unwrap();
    assert!(!ids.is_empty());
    assert!(ids.iter().all(|&id| id == 21), "未登録の special 表記が [UNK] 以外になった: {ids:?}");
    assert_eq!(tok.decode(&[16]).unwrap(), "<|im_start|>");
}

#[test]
fn empty_and_whitespace_are_safe_on_added_token_vocab() {
    let tok = qwen_style_tokenizer().unwrap();
    assert_eq!(tok.encode("").unwrap(), Vec::<u32>::new());
    assert_eq!(tok.encode(" \n\t ").unwrap(), Vec::<u32>::new());
    assert_eq!(tok.decode(&[]).unwrap(), "");
}

// ------------------------------------------------------------ 語彙サイズ

#[test]
fn vocab_size_is_positive_and_covers_known_ids() {
    for (tok, known_max) in [(dummy_tokenizer().unwrap(), 20u32), (qwen_style_tokenizer().unwrap(), 22u32)] {
        assert!(tok.vocab_size() > 0, "vocab_size が 0");
        // 既知の最大 ID が語彙として引ける (piece_bytes が None でない)
        assert!(tok.piece_bytes(known_max).is_some(), "id {known_max} が語彙に無い");
    }
}
