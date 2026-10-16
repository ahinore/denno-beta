//! 第27: streaming tool_use の SSE アセンブラ。
//!
//! engine の delta コールバック (detok 済み token piece 単位) を受け、
//! `<tool_call>` block の境界検出を行いながら Anthropic 形式の SSE イベント列
//! (text block / tool_use block の start / delta / stop) を組み立てる。
//!
//! 設計 (`extract_tool_calls` と同一の意味論をストリームへ移植):
//!
//! - 完全な block 単位でのみ tool_use 化する (OPEN/CLOSE が delta 境界で分割
//!   されても `scan` バッファで追跡。`scan` の末尾が OPEN 接頭辞の可能性が
//!   なくなるまで確定テキストとしては出力しない)
//! - OPEN〜CLOSE 間が確定した時点で `parse_tool_call_inner` でパースし、
//!   成功なら tool_use block (start → input_json_delta → stop を即時完結)、
//!   失敗なら block 全体をテキストとして keep へ (再スキャンしない = extract の
//!   「パース失敗 block はテキストとして残す」と同一)
//! - 未完結 (CLOSE 無し) は finish 時にテキスト化 (= extract の「未完結は
//!   block とみなさない」と同一)
//! - テキストの trim 規約も非ストリーミングと一致させる: 先頭の空白は
//!   非空白を 1 つも出力するまで除去、末尾の空白は「block の後に室内空白に
//!   なるか最後で削られるか未確定」のため hold し、finish で削る。
//!   任意の delta 分割で `extract_tool_calls(全文)` の残テキストと一致する
//!
//! tool_use block を 1 つでも出した場合、`finish_run` の message_delta の
//! stop_reason は `"tool_use"` になる (非ストリーミング `message_json` と同一規則)。

use serde_json::json;

use crate::{new_tool_id, parse_tool_call_inner, ExtractedToolCall, TOOL_CALL_CLOSE, TOOL_CALL_OPEN};

/// delta ストリーム → SSE イベント列のアセンブラ。
///
/// Stream ジョブごとに 1 個作る。`admit` が index 0 の text block を
/// 先に開けるため、構築時は「index 0 が open 中」の状態から始める。
pub struct StreamAssembler {
    /// push_delta / finish が積んだ未送信イベント
    events: Vec<serde_json::Value>,
    /// 次に割り当てる block index
    next_index: usize,
    /// 現在 open 中の text block (tool_use block は start→delta→stop を
    /// 即時完結させるため保持しない)
    open_text: Option<usize>,
    /// 非空白テキストを 1 つでも出力済みか (extract の trim_start 相当)
    emitted_nonws: bool,
    /// 確定テキストのうち未出力分 (末尾の空白は hold。block 後に室内空白に
    /// なる可能性があるため即時出力しない)
    keep: String,
    /// まだ OPEN 検索をしていない入力 (未完結 block / OPEN 接頭辞を含みうる)
    scan: String,
    /// 出力済み tool_use block 数 (0 = テキストのみストリーム)
    blocks_emitted: usize,
}

impl StreamAssembler {
    /// Stream ジョブ用に構築する (`admit` が既に text block `open_index` を
    /// 開けている前提)。
    pub fn new(open_index: usize) -> Self {
        Self {
            events: Vec::new(),
            next_index: open_index + 1,
            open_text: Some(open_index),
            emitted_nonws: false,
            keep: String::new(),
            scan: String::new(),
            blocks_emitted: 0,
        }
    }

    /// detok 済み delta piece を供給し、確定した分のイベントを積む。
    pub fn push_delta(&mut self, delta: &str) {
        self.scan.push_str(delta);
        loop {
            let Some(s) = self.scan.find(TOOL_CALL_OPEN) else {
                // OPEN 無し: 末尾が OPEN 接頭辞の可能性の分だけ hold し、
                // 残りは確定テキストとして keep へ
                let hold = partial_open_suffix_len(&self.scan);
                let boundary = self.scan.len() - hold;
                self.keep.push_str(&self.scan[..boundary]);
                self.scan.drain(..boundary);
                self.flush_text();
                return;
            };
            // OPEN 手前は確定テキスト (後に block が来ることが確定済み)
            self.keep.push_str(&self.scan[..s]);
            let inner_start = s + TOOL_CALL_OPEN.len();
            let Some(e_rel) = self.scan[inner_start..].find(TOOL_CALL_CLOSE) else {
                // block 未完結: OPEN 以降は出力せず hold (CLOSE を待つ)
                self.scan.drain(..s);
                self.flush_text();
                return;
            };
            let close_end = inner_start + e_rel + TOOL_CALL_CLOSE.len();
            let inner = self.scan[inner_start..inner_start + e_rel].to_string();
            let block_text = self.scan[s..close_end].to_string();
            self.scan.drain(..close_end);
            match parse_tool_call_inner(&inner) {
                Some(c) => {
                    // block 直前までのテキスト領域を出力 (末尾空白は hold 維持)
                    self.flush_text();
                    self.emit_tool_use(c);
                }
                None => {
                    // パース失敗 block はテキストとして keep へ。
                    // 再スキャンしない (extract が scan 位置を CLOSE 後へ進めるのと同一)
                    self.keep.push_str(&block_text);
                    self.flush_text();
                }
            }
            // 残り scan のスキャンを続行
        }
    }

    /// 生成完了。残り scan をテキスト化し、open 中の block を閉じる。
    pub fn finish(&mut self) {
        let scan = std::mem::take(&mut self.scan);
        self.keep.push_str(&scan);
        self.flush_text_final();
        if let Some(i) = self.open_text.take() {
            self.events.push(json!({"type": "content_block_stop", "index": i}));
        }
    }

    /// 積んだイベントを取り出す。
    pub fn take_events(&mut self) -> std::vec::Drain<'_, serde_json::Value> {
        self.events.drain(..)
    }

    /// 出力済み tool_use block 数。
    pub fn blocks_emitted(&self) -> usize {
        self.blocks_emitted
    }

    /// keep から出力可能な範囲を出す: 末尾の空白は hold (extract の最終 trim が
    /// block 後のテキストとどう結合するか未確定のため)、先頭の空白は
    /// 非空白未出力の間のみ除去。
    fn flush_text(&mut self) {
        let trimmed_end = self.keep.trim_end().len();
        if trimmed_end == 0 {
            return; // 全部空白 (or 空) → 出力無し
        }
        let start = if self.emitted_nonws {
            0
        } else {
            let lead = self.keep.len() - self.keep.trim_start().len();
            lead.min(trimmed_end)
        };
        if start >= trimmed_end {
            return;
        }
        let text = self.keep[start..trimmed_end].to_string();
        self.emit_text_delta(&text);
        self.emitted_nonws = true;
        self.keep.drain(..trimmed_end);
    }

    /// finish 用の最終 flush: 末尾 hold 分の空白を削る (= extract の最終 trim)。
    fn flush_text_final(&mut self) {
        let trimmed_end = self.keep.trim_end().len();
        if trimmed_end > 0 {
            let start = if self.emitted_nonws {
                0
            } else {
                let lead = self.keep.len() - self.keep.trim_start().len();
                lead.min(trimmed_end)
            };
            if start < trimmed_end {
                let text = self.keep[start..trimmed_end].to_string();
                self.emit_text_delta(&text);
                self.emitted_nonws = true;
            }
        }
        self.keep.clear();
    }

    /// text_delta を出す (text block が未オープンなら lazy に開く)。
    fn emit_text_delta(&mut self, text: &str) {
        let idx = match self.open_text {
            Some(i) => i,
            None => {
                let i = self.next_index;
                self.next_index += 1;
                self.open_text = Some(i);
                self.events.push(json!({
                    "type": "content_block_start",
                    "index": i,
                    "content_block": {"type": "text", "text": ""},
                }));
                i
            }
        };
        self.events.push(json!({
            "type": "content_block_delta",
            "index": idx,
            "delta": {"type": "text_delta", "text": text},
        }));
    }

    /// tool_use block を start → input_json_delta → stop で即時完結させる。
    /// open 中の text block があれば先に閉じる。
    fn emit_tool_use(&mut self, c: ExtractedToolCall) {
        if let Some(i) = self.open_text.take() {
            self.events.push(json!({"type": "content_block_stop", "index": i}));
        }
        let idx = self.next_index;
        self.next_index += 1;
        self.events.push(json!({
            "type": "content_block_start",
            "index": idx,
            "content_block": {
                "type": "tool_use",
                "id": new_tool_id(),
                "name": c.name,
                "input": {},
            },
        }));
        self.events.push(json!({
            "type": "content_block_delta",
            "index": idx,
            "delta": {
                "type": "input_json_delta",
                "partial_json": serde_json::to_string(&c.input).unwrap_or_default(),
            },
        }));
        self.events.push(json!({"type": "content_block_stop", "index": idx}));
        self.blocks_emitted += 1;
    }
}

/// `s` の末尾が TOOL_CALL_OPEN の真の接頭辞と一致する最大長
/// (分割到着した OPEN の先頭部分を keep へ流さず hold する長さ)。
fn partial_open_suffix_len(s: &str) -> usize {
    for len in (1..TOOL_CALL_OPEN.len()).rev() {
        if s.ends_with(&TOOL_CALL_OPEN[..len]) {
            return len;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    /// block サンプル組み立て用 (リテラル直書きを避けて定数から組む)
    fn block(name: &str, inner: &str) -> String {
        format!("{TOOL_CALL_OPEN}\n<function={name}>{inner}</function>\n{TOOL_CALL_CLOSE}")
    }

    /// アセンブラに全文を chunk 分割で流し、(出力テキスト結合, tool calls, 全イベント) を返す
    fn run_asm(full: &str, chunk: usize) -> (String, Vec<(String, serde_json::Value)>, Vec<Value>) {
        let mut asm = StreamAssembler::new(0);
        let mut events = Vec::new();
        let bytes = full.as_bytes();
        let mut pos = 0;
        // push_delta は &str を受けるため文字境界で切る
        while pos < bytes.len() {
            let mut end = (pos + chunk).min(bytes.len());
            while end < bytes.len() && !full.is_char_boundary(end) {
                end += 1;
            }
            asm.push_delta(&full[pos..end]);
            events.extend(asm.take_events());
            pos = end;
        }
        asm.finish();
        events.extend(asm.take_events());
        let mut text = String::new();
        let mut calls = Vec::new();
        let mut open_name: Option<String> = None;
        for ev in &events {
            match ev["type"].as_str().unwrap() {
                "content_block_start" => {
                    open_name = ev["content_block"]["name"].as_str().map(String::from);
                }
                "content_block_delta" => {
                    if ev["delta"]["type"] == "text_delta" {
                        text.push_str(ev["delta"]["text"].as_str().unwrap());
                    } else {
                        let parsed: serde_json::Value =
                            serde_json::from_str(ev["delta"]["partial_json"].as_str().unwrap())
                                .unwrap();
                        calls.push((open_name.clone().unwrap_or_default(), parsed));
                    }
                }
                _ => {}
            }
        }
        (text, calls, events)
    }

    /// テキストのみ: 任意分割で extract_tool_calls と同一テキスト
    #[test]
    fn text_only_matches_extract_semantics() {
        let full = "  hello  world  \n foo ";
        let (clean, _) = crate::extract_tool_calls(full);
        for chunk in [1, 2, 3, 5, 100] {
            let (text, calls, _) = run_asm(full, chunk);
            assert_eq!(text, clean, "chunk = {chunk}");
            assert!(calls.is_empty(), "chunk = {chunk}");
        }
    }

    /// テキスト + tool block: block は除去・前後テキストは trim 規約どおり
    #[test]
    fn text_plus_tool_matches_extract_semantics() {
        let full = format!(
            "Checking.\n{}\nDone.",
            block("get_weather", "<parameter=city>\nTokyo\n</parameter>")
        );
        let (clean, expect) = crate::extract_tool_calls(&full);
        assert_eq!(expect.len(), 1);
        for chunk in [1, 2, 3, 7, 100] {
            let (text, calls, _) = run_asm(&full, chunk);
            assert_eq!(text, clean, "chunk = {chunk}");
            assert_eq!(calls.len(), 1, "chunk = {chunk}");
            assert_eq!(calls[0].0, "get_weather", "chunk = {chunk}");
            assert_eq!(calls[0].1, expect[0].input, "chunk = {chunk}");
        }
    }

    /// block のみ (テキスト無し): text delta は出ない・tool_use block のみ。
    /// 空の text block 0 が閉じられ tool block は index 1
    #[test]
    fn tool_only_stream_emits_single_block() {
        let full = block("run_tool", "");
        let mut asm = StreamAssembler::new(0);
        asm.push_delta(&full);
        asm.finish();
        let events: Vec<Value> = asm.take_events().collect();
        let types: Vec<&str> = events.iter().map(|e| e["type"].as_str().unwrap()).collect();
        assert_eq!(
            types,
            vec![
                "content_block_stop",  // 空の text block 0 を閉じる
                "content_block_start", // tool_use (index 1)
                "content_block_delta", // input_json_delta
                "content_block_stop",
            ]
        );
        assert_eq!(events[1]["index"], 1);
        assert_eq!(events[1]["content_block"]["type"], "tool_use");
        assert_eq!(events[1]["content_block"]["name"], "run_tool");
        assert!(events[1]["content_block"]["id"].as_str().unwrap().starts_with("toolu_"));
        assert_eq!(events[2]["delta"]["type"], "input_json_delta");
        assert_eq!(events[2]["index"], 1);
        assert_eq!(asm.blocks_emitted(), 1);
    }

    /// パース失敗 block はテキスト化 (extract と同一)
    #[test]
    fn unparsable_block_becomes_text() {
        let full = format!("a{0}\n<function=>\n</function>\n{1}b", TOOL_CALL_OPEN, TOOL_CALL_CLOSE);
        let (clean, _) = crate::extract_tool_calls(&full);
        for chunk in [1, 4, 100] {
            let (text, calls, _) = run_asm(&full, chunk);
            assert_eq!(text, clean, "chunk = {chunk}");
            assert!(calls.is_empty(), "chunk = {chunk}");
        }
    }

    /// OPEN/CLOSE が delta 境界で分割しても検出される (全バイト境界で分割)
    #[test]
    fn open_and_close_split_across_every_boundary() {
        let full = format!("x{}y", block("get_weather", ""));
        let (clean, expect) = crate::extract_tool_calls(&full);
        assert_eq!(expect.len(), 1);
        for split in 1..full.len() {
            if !full.is_char_boundary(split) {
                continue;
            }
            let mut asm = StreamAssembler::new(0);
            asm.push_delta(&full[..split]);
            asm.push_delta(&full[split..]);
            asm.finish();
            let text: String = asm
                .take_events()
                .filter(|e| e["type"] == "content_block_delta" && e["delta"]["type"] == "text_delta")
                .map(|e| e["delta"]["text"].as_str().unwrap().to_string())
                .collect();
            assert_eq!(text, clean, "split = {split}");
            assert_eq!(asm.blocks_emitted(), 1, "split = {split}");
        }
    }

    /// 部分的な OPEN 接頭辞 (テキスト末尾の "<" 等) は誤出力されず hold される
    #[test]
    fn partial_open_prefix_is_held_not_emitted() {
        // "abc" は確定 → 即出力
        let mut asm = StreamAssembler::new(0);
        asm.push_delta("abc");
        let emitted: String = asm
            .take_events()
            .filter(|e| e["type"] == "content_block_delta")
            .map(|e| e["delta"]["text"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(emitted, "abc");

        // "abc" + OPEN 接頭辞 1 文字: 接頭辞分は hold
        let mut asm = StreamAssembler::new(0);
        asm.push_delta("abc<");
        let emitted: String = asm
            .take_events()
            .filter(|e| e["type"] == "content_block_delta")
            .map(|e| e["delta"]["text"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(emitted, "abc");
        // 続きが OPEN でなければ hold 分含め全て出る
        asm.push_delta("foo");
        let emitted: String = asm
            .take_events()
            .filter(|e| e["type"] == "content_block_delta")
            .map(|e| e["delta"]["text"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(emitted, "<foo");
    }

    /// 未完結 block (CLOSE 無し) は finish でテキスト化 (extract と同一)
    #[test]
    fn incomplete_block_becomes_text_at_finish() {
        let full = format!("abc{0}never closed", TOOL_CALL_OPEN);
        let (clean, _) = crate::extract_tool_calls(&full);
        let mut asm = StreamAssembler::new(0);
        asm.push_delta(&full);
        asm.finish();
        let text: String = asm
            .take_events()
            .filter(|e| e["type"] == "content_block_delta" && e["delta"]["type"] == "text_delta")
            .map(|e| e["delta"]["text"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(text, clean);
        assert_eq!(asm.blocks_emitted(), 0);
    }

    /// 複数 block + 間テキスト: 2 tool_use block・block index は昇順
    #[test]
    fn multiple_blocks_in_order() {
        let full = format!(
            "{}mid{}end",
            block("a_tool", ""),
            block("b_tool", "")
        );
        let (clean, expect) = crate::extract_tool_calls(&full);
        assert_eq!(expect.len(), 2);
        let (text, _, events) = run_asm(&full, 3);
        assert_eq!(text, clean);
        let names: Vec<&str> = events
            .iter()
            .filter(|e| e["type"] == "content_block_start" && e["content_block"]["type"] == "tool_use")
            .map(|e| e["content_block"]["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, vec!["a_tool", "b_tool"]);
        // block index: 0 (text — start は admit が送るのでアセンブラは出さない)
        // → 1 (tool) → 2 (text) → 3 (tool) → 4 (text) と昇順
        // ("mid" と "end" は tool block を挟むため別 text block になる)
        let idx: Vec<u64> = events
            .iter()
            .filter(|e| e["type"] == "content_block_start")
            .map(|e| e["index"].as_u64().unwrap())
            .collect();
        assert_eq!(idx, vec![1, 2, 3, 4]);
        // 2 個目の tool block のために text block が再オープンされている
        let reopen = events
            .iter()
            .find(|e| e["type"] == "content_block_start" && e["content_block"]["type"] == "text")
            .unwrap();
        assert_eq!(reopen["index"], 2);
    }

    /// block 前のテキストが空白のみなら text delta は出ない (extract と同一 = 空文字)
    #[test]
    fn whitespace_only_region_before_block_emits_nothing() {
        let full = format!("   {}", block("f_tool", ""));
        let mut asm = StreamAssembler::new(0);
        asm.push_delta(&full);
        asm.finish();
        let events: Vec<Value> = asm.take_events().collect();
        let text_deltas: Vec<&Value> = events
            .iter()
            .filter(|e| e["type"] == "content_block_delta" && e["delta"]["type"] == "text_delta")
            .collect();
        assert!(text_deltas.is_empty(), "events = {events:?}");
        let tool_start = events
            .iter()
            .find(|e| e["type"] == "content_block_start" && e["content_block"]["type"] == "tool_use")
            .unwrap();
        assert_eq!(tool_start["index"], 1);
        assert_eq!(asm.blocks_emitted(), 1);
    }

    /// block 前後の空白が室内空白として保存される (extract の trim と同一結合)
    #[test]
    fn whitespace_around_block_becomes_interior() {
        let full = format!("hello  {}  world", block("f_tool", ""));
        let (clean, _) = crate::extract_tool_calls(&full);
        let (text, calls, _) = run_asm(&full, 2);
        assert_eq!(text, clean);
        assert_eq!(calls.len(), 1);
    }
}
