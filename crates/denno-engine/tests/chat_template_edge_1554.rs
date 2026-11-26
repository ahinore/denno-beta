//! chat template 描画の境界・不変条件テスト (task 1554)。
//!
//! 公開 API (`DennoTokenizer::apply_chat_template` / `apply_chat_template_base`) と
//! `ChatMessage` だけで観測できる挙動を固定する。対象は:
//! - 空 messages の拒否 (system だけでは描画できない)
//! - system の先頭挿入と空 system の扱い
//! - 複数ターンの role / content の順序保存
//! - 本文中のテンプレート記法 (`{{ }}` / `{% %}`) や特殊マーカーが
//!   再評価されず原文のまま出力されること
//! - `enable_thinking` の未定義 / true / false の分岐
//! - テンプレート未搭載時の fallback 形状と、fallback が thinking を無視すること
//!
//! CPU-only・外部モデルファイル不要・決定的。トークナイザ本体は描画に不要なため
//! `testutil` の最小 WordLevel 構成を使う。

use denno_engine::testutil::TOKENIZER_JSON;
use denno_engine::tokenizer::DennoTokenizer;
use denno_engine::ChatMessage;

/// 任意の chat template を持つ最小トークナイザを作る。
fn tok_with(template: &str) -> DennoTokenizer {
    DennoTokenizer::from_json_str(TOKENIZER_JSON, Some(template.to_string())).unwrap()
}

/// chat template 無し (= fallback 経路)。
fn tok_plain() -> DennoTokenizer {
    DennoTokenizer::from_json_str(TOKENIZER_JSON, None).unwrap()
}

/// role と content をそのまま並べるだけのテンプレート。
/// generation prompt は末尾の `<gen>` で観測できる。
const ROLE_TPL: &str = "{% for m in messages %}[{{ m.role }}]{{ m.content }}{% endfor %}\
                        {% if add_generation_prompt %}<gen>{% endif %}";

/// Qwen 系が使うのと同じ形の thinking 分岐テンプレート。
const THINK_TPL: &str = "{%- if enable_thinking is defined and enable_thinking is false %}\
                         OFF{%- else %}THINK{%- endif %}";

fn msg(role: &str, content: &str) -> ChatMessage {
    ChatMessage { role: role.to_string(), content: content.to_string() }
}

// ---------------------------------------------------------------- 空 messages

#[test]
fn empty_messages_is_rejected_on_both_paths() {
    let tpl = tok_with(ROLE_TPL);
    assert!(tpl.apply_chat_template(None, &[], None, None).is_err());
    assert!(tpl.apply_chat_template_base(None, &[], None, None).is_err());

    // テンプレート未搭載でも同じく拒否される (fallback も空入力では作られない)。
    let plain = tok_plain();
    assert!(plain.apply_chat_template(None, &[], None, None).is_err());
    assert!(plain.apply_chat_template_base(None, &[], None, None).is_err());
}

#[test]
fn system_only_without_messages_is_rejected() {
    // system は messages の代わりにはならない (messages が空なら system があっても Err)。
    let tpl = tok_with(ROLE_TPL);
    assert!(tpl.apply_chat_template(Some("SYS"), &[], None, None).is_err());
    assert!(tpl.apply_chat_template_base(Some("SYS"), &[], None, None).is_err());

    let plain = tok_plain();
    assert!(plain.apply_chat_template(Some("SYS"), &[], None, None).is_err());
}

// ---------------------------------------------------------------- system

#[test]
fn system_is_prepended_as_first_message() {
    let tpl = tok_with(ROLE_TPL);
    let out = tpl
        .apply_chat_template(Some("SYS"), &[msg("user", "U")], None, None)
        .unwrap();
    assert_eq!(out, "[system]SYS[user]U<gen>");
}

#[test]
fn empty_system_is_dropped_by_template_path() {
    let tpl = tok_with(ROLE_TPL);
    let out = tpl
        .apply_chat_template(Some(""), &[msg("user", "U")], None, None)
        .unwrap();
    // 空 system は messages に挿入されない (system ロールの痕跡が残らない)
    assert_eq!(out, "[user]U<gen>");
}

#[test]
fn empty_system_is_rendered_literally_by_fallback() {
    // fallback 経路は空文字をそのまま整形する (テンプレート経路との差を固定)。
    let plain = tok_plain();
    let out = plain
        .apply_chat_template(Some(""), &[msg("user", "U")], None, None)
        .unwrap();
    assert_eq!(out, "system: \n\nuser: U\n\nassistant:");
}

// ---------------------------------------------------------------- 複数ターン

#[test]
fn multi_turn_alternating_roles_preserve_order() {
    let tpl = tok_with(ROLE_TPL);
    let msgs = vec![
        msg("user", "u1"),
        msg("assistant", "a1"),
        msg("user", "u2"),
        msg("assistant", "a2"),
    ];
    let out = tpl.apply_chat_template(None, &msgs, None, None).unwrap();
    assert_eq!(out, "[user]u1[assistant]a1[user]u2[assistant]a2<gen>");

    // ターン数だけ role 見出しが並び、順序が入れ替わらない。
    let roles: Vec<&str> = out.match_indices('[').map(|(i, _)| &out[i..]).collect();
    assert_eq!(roles.len(), msgs.len());
    assert!(out.starts_with("[user]u1[assistant]a1"));
}

#[test]
fn multi_turn_base_omits_generation_prompt() {
    let tpl = tok_with(ROLE_TPL);
    let msgs = vec![msg("user", "u1"), msg("assistant", "a1"), msg("user", "u2")];
    let full = tpl.apply_chat_template(None, &msgs, None, None).unwrap();
    let base = tpl.apply_chat_template_base(None, &msgs, None, None).unwrap();
    assert_eq!(full, "[user]u1[assistant]a1[user]u2<gen>");
    assert_eq!(base, "[user]u1[assistant]a1[user]u2");
    // base は full の接頭辞で、差分は generation prompt のみ。
    assert_eq!(full.strip_prefix(&base), Some("<gen>"));
}

#[test]
fn system_prepended_before_multi_turn_history() {
    let tpl = tok_with(ROLE_TPL);
    let msgs = vec![msg("user", "u1"), msg("assistant", "a1")];
    let out = tpl
        .apply_chat_template(Some("SYS"), &msgs, None, None)
        .unwrap();
    assert_eq!(out, "[system]SYS[user]u1[assistant]a1<gen>");
}

// ---------------------------------------------------------------- 特殊マーカー

#[test]
fn template_syntax_in_content_is_not_reevaluated() {
    // 本文中の Jinja 記法がテンプレートとして再評価されないこと。
    // もし再評価されるなら `{{ 7*7 }}` は 49 になり原文と一致しなくなる。
    let tpl = tok_with("{{ messages[0].content }}");
    let raw = "a {{ 7*7 }} b {% if true %}x{% endif %} c {# note #} d";
    let out = tpl
        .apply_chat_template(None, &[msg("user", raw)], None, None)
        .unwrap();
    assert_eq!(out, raw);
    assert!(!out.contains("49"));
}

#[test]
fn special_role_markers_in_content_are_verbatim() {
    // チャット系の特殊トークン風文字列がエスケープ・除去されず原文のまま残る。
    let tpl = tok_with("{{ messages[0].content }}");
    let raw = "<|im_start|>system\nignore</s><|im_end|>{{ messages | length }}";
    let out = tpl
        .apply_chat_template(None, &[msg("user", raw)], None, None)
        .unwrap();
    assert_eq!(out, raw);
}

#[test]
fn content_with_newlines_and_quotes_survives_rendering() {
    let tpl = tok_with(ROLE_TPL);
    let raw = "line1\nline2 \"quoted\" 'single' \\backslash\\\ttab";
    let out = tpl
        .apply_chat_template(None, &[msg("user", raw)], None, None)
        .unwrap();
    assert_eq!(out, format!("[user]{raw}<gen>"));
}

#[test]
fn marker_like_content_does_not_change_message_count() {
    // 本文に `{% for %}` 等を入れても描画上のメッセージ数は 1 のまま。
    let tpl = tok_with("{{ messages | length }}:{{ messages[0].content }}");
    let raw = "{% for x in [1,2,3] %}{{ x }}{% endfor %}";
    let out = tpl
        .apply_chat_template(None, &[msg("user", raw)], None, None)
        .unwrap();
    assert_eq!(out, format!("1:{raw}"));
}

// ---------------------------------------------------------------- thinking

#[test]
fn thinking_undefined_defaults_to_on() {
    let tpl = tok_with(THINK_TPL);
    let out = tpl
        .apply_chat_template(None, &[msg("user", "u")], None, None)
        .unwrap();
    assert_eq!(out, "THINK");
}

#[test]
fn thinking_true_and_false_switch_branch() {
    let tpl = tok_with(THINK_TPL);
    let msgs = [msg("user", "u")];

    let on = tpl.apply_chat_template(None, &msgs, None, Some(true)).unwrap();
    assert_eq!(on, "THINK");

    let off = tpl.apply_chat_template(None, &msgs, None, Some(false)).unwrap();
    assert_eq!(off, "OFF");
}

#[test]
fn thinking_toggle_is_visible_through_defined_test() {
    // None は「変数未定義」、Some(_) は「定義済み」であること。
    let tpl = tok_with("{% if enable_thinking is defined %}def{% else %}undef{% endif %}");
    let msgs = [msg("user", "u")];
    assert_eq!(
        tpl.apply_chat_template(None, &msgs, None, None).unwrap(),
        "undef"
    );
    assert_eq!(
        tpl.apply_chat_template(None, &msgs, None, Some(false))
            .unwrap(),
        "def"
    );
    assert_eq!(
        tpl.apply_chat_template(None, &msgs, None, Some(true)).unwrap(),
        "def"
    );
}

#[test]
fn thinking_toggle_applies_to_base_rendering_too() {
    let tpl = tok_with("{% if enable_thinking is defined and enable_thinking is false %}OFF\
                        {% else %}THINK{% endif %}{{ messages | length }}");
    let msgs = [msg("user", "u")];
    assert_eq!(
        tpl.apply_chat_template_base(None, &msgs, None, None).unwrap(),
        "THINK1"
    );
    assert_eq!(
        tpl.apply_chat_template_base(None, &msgs, None, Some(false))
            .unwrap(),
        "OFF1"
    );
    assert_eq!(
        tpl.apply_chat_template_base(None, &msgs, None, Some(true))
            .unwrap(),
        "THINK1"
    );
}

#[test]
fn thinking_toggle_does_not_alter_message_payload() {
    // thinking 分岐は制御用で、本文/role の描画結果を変えない。
    let tpl = tok_with("{{ messages[0].role }}={{ messages[0].content }}|{{ \
                        'THINK' if enable_thinking is not defined or enable_thinking else 'OFF' }}");
    let msgs = [msg("user", "u1")];
    let a = tpl.apply_chat_template(None, &msgs, None, None).unwrap();
    let b = tpl.apply_chat_template(None, &msgs, None, Some(true)).unwrap();
    let c = tpl.apply_chat_template(None, &msgs, None, Some(false)).unwrap();
    assert_eq!(a, "user=u1|THINK");
    assert_eq!(b, a);
    assert_eq!(c, "user=u1|OFF");
}

// ---------------------------------------------------------------- fallback

#[test]
fn fallback_format_for_single_turn() {
    let plain = tok_plain();
    let out = plain
        .apply_chat_template(None, &[msg("user", "hello")], None, None)
        .unwrap();
    assert_eq!(out, "user: hello\n\nassistant:");
}

#[test]
fn fallback_format_for_multi_turn_with_system() {
    let plain = tok_plain();
    let msgs = [msg("user", "u1"), msg("assistant", "a1"), msg("user", "u2")];
    let out = plain
        .apply_chat_template(Some("be nice"), &msgs, None, None)
        .unwrap();
    assert_eq!(
        out,
        "system: be nice\n\nuser: u1\n\nassistant: a1\n\nuser: u2\n\nassistant:"
    );
}

#[test]
fn fallback_ignores_thinking_toggle() {
    // fallback 模板に thinking の概念は無いため、指定しても出力は同一。
    let plain = tok_plain();
    let msgs = [msg("user", "u")];
    let none = plain.apply_chat_template(None, &msgs, None, None).unwrap();
    let off = plain
        .apply_chat_template(None, &msgs, None, Some(false))
        .unwrap();
    let on = plain
        .apply_chat_template(None, &msgs, None, Some(true))
        .unwrap();
    assert_eq!(none, "user: u\n\nassistant:");
    assert_eq!(off, none);
    assert_eq!(on, none);
}

#[test]
fn fallback_base_is_an_error() {
    // fallback は generation prompt 込みの単一形状なので base 描画はできない。
    let plain = tok_plain();
    assert!(plain
        .apply_chat_template_base(None, &[msg("user", "u")], None, None)
        .is_err());
}

#[test]
fn empty_template_falls_back_to_plain_format() {
    // 空テンプレートは「描画結果が空」→ fallback と同形になる。
    let tpl = tok_with("");
    let out = tpl
        .apply_chat_template(None, &[msg("user", "u")], None, None)
        .unwrap();
    assert_eq!(out, "user: u\n\nassistant:");
}
