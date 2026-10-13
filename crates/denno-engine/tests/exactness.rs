//! 基準出力テスト (HANDOFF §5)。実モデルのDLが必要なため #[ignore] 付き。
//! 明示実行: cargo test -p denno-engine --test exactness -- --ignored --nocapture
//!
//! モデルディレクトリは環境変数で上書きできる:
//!   DENNO_QWEN_DIR  (default: /home/dev/.cache/denno/hub/models--Qwen--Qwen2.5-0.5B-Instruct)
//!   DENNO_GPT2_DIR  (default: /home/dev/.cache/denno/hub/models--openai-community--gpt2)

use denno_engine::{ChatMessage, Engine, GenParams, QuantMode};
use std::path::PathBuf;

fn model_dir(env_key: &str, default: &str) -> Option<PathBuf> {
    let dir = std::env::var(env_key)
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(default));
    if dir.join("config.json").exists()
        && (dir.join("model.safetensors").exists()
            || dir.join("model.safetensors.index.json").exists())
    {
        Some(dir)
    } else {
        eprintln!(
            "skip: model not cached ({} = {:?}); set {} to enable",
            env_key, dir, env_key
        );
        None
    }
}

fn user_msg(text: &str) -> Vec<ChatMessage> {
    vec![ChatMessage { role: "user".into(), content: text.into() }]
}

fn greedy(max_tokens: usize) -> GenParams {
    GenParams {
        max_tokens,
        temperature: Some(0.0),
        seed: Some(42),
        ..Default::default()
    }
}

/// GPT-2 基準出力 (日本の首都は? / max_tokens 40 / fallback テンプレート)。
/// HF transformers 参照実装と一致を検証済み。40 トークン目でマルチバイト
/// 文字「日」のバイト列の途中で切れるため末尾に U+FFFD が現れる (lossy)。
const GPT2_JP_BASELINE: &str =
    " 日本の首都は?\n\nassistant: 日本の首都は?\n\nassistant: 日本の\u{FFFD}";

/// §5 基準出力: Qwen2.5-0.5B / temp 0 / seed 42 / 「1+1は?」
/// §4 の決定論検証 curl と同じ条件 (server は system 無しで engine.chat に渡す)
#[test]
#[ignore]
fn qwen_1plus1_matches_baseline() {
    let Some(dir) = model_dir(
        "DENNO_QWEN_DIR",
        "/home/dev/.cache/denno/hub/models--Qwen--Qwen2.5-0.5B-Instruct",
    ) else {
        return;
    };
    let mut e = Engine::load(&dir, "denno-test", Some(42), QuantMode::Fp32).unwrap();
    let out = e.chat(None, &user_msg("1+1は?"), &greedy(50), |_| {}).unwrap();
    assert_eq!(
        out.text,
        "1+1は1です。これは数学的定理で、1と1を加えることによって得られます。"
    );
    assert_eq!(out.stop_reason, denno_engine::StopReason::EndTurn);
}

/// §5 基準出力 + prefix cache の exactness:
/// cold エンジンと warm (prefix cache 装填済み) エンジンの出力が完全一致する。
#[test]
#[ignore]
fn qwen_prefix_cache_keeps_exact_output() {
    let Some(dir) = model_dir(
        "DENNO_QWEN_DIR",
        "/home/dev/.cache/denno/hub/models--Qwen--Qwen2.5-0.5B-Instruct",
    ) else {
        return;
    };

    // cold: 何もキャッシュが無い状態で「日本の首都は?」
    let mut cold = Engine::load(&dir, "denno-test", Some(42), QuantMode::Fp32).unwrap();
    let cold_out = cold.chat(None, &user_msg("日本の首都は?"), &greedy(30), |_| {}).unwrap();

    // warm: 別の会話を挟んでから同一プロンプト (LCP 再利用が走る)
    let mut warm = Engine::load(&dir, "denno-test", Some(42), QuantMode::Fp32).unwrap();
    warm.chat(None, &user_msg("1+1は?"), &greedy(10), |_| {}).unwrap();
    let warm_out = warm.chat(None, &user_msg("日本の首都は?"), &greedy(30), |_| {}).unwrap();

    assert_eq!(cold_out.text, warm_out.text);
    // §5 基準出力
    assert_eq!(cold_out.text, "日本の首都は東京都です。");
}

/// seed 固定 + 温度 0 の決定論: 同一リクエストの再実行で完全一致
#[test]
#[ignore]
fn qwen_exact_decoding_is_reproducible() {
    let Some(dir) = model_dir(
        "DENNO_QWEN_DIR",
        "/home/dev/.cache/denno/hub/models--Qwen--Qwen2.5-0.5B-Instruct",
    ) else {
        return;
    };
    let mut e = Engine::load(&dir, "denno-test", Some(42), QuantMode::Fp32).unwrap();
    let a = e.chat(None, &user_msg("1+1は?"), &greedy(20), |_| {}).unwrap();
    let b = e.chat(None, &user_msg("1+1は?"), &greedy(20), |_| {}).unwrap();
    assert_eq!(a.text, b.text);
    assert_eq!(a.output_tokens, b.output_tokens);
}

/// GPT-2 基準出力 (seed 42)。gpt2 は chat template を持たないので
/// fallback テンプレートが使われる。基準文字列は参照実装
/// (HF transformers fp32, greedy) の出力と一致することを検証済み
/// (2026-10-12, 開発コンテナ内の参照実装スクリプト)。
/// GPT-2 base は fallback テンプレート付きプロンプトに対して入力を
/// 繰り返す (エコー) のが素の貪欲生成の挙動。
#[test]
#[ignore]
fn gpt2_japan_capital_matches_baseline() {
    let Some(dir) = model_dir(
        "DENNO_GPT2_DIR",
        "/home/dev/.cache/denno/hub/models--openai-community--gpt2",
    ) else {
        return;
    };
    let mut e = Engine::load(&dir, "gpt2-test", Some(42), QuantMode::Fp32).unwrap();
    let out = e.chat(None, &user_msg("日本の首都は?"), &greedy(40), |_| {}).unwrap();
    println!("gpt2 output: {}", out.text);
    assert_eq!(out.text, GPT2_JP_BASELINE, "unexpected output: {}", out.text);
}

/// P3: int8 モードの決定論。同一 seed + 温度 0 → 同一出力が維持される。
/// ただし fp32 と同一出力ではないことは仕様 (丸め誤差)。
#[test]
#[ignore]
fn qwen_int8_mode_is_deterministic() {
    let Some(dir) = model_dir(
        "DENNO_QWEN_DIR",
        "/home/dev/.cache/denno/hub/models--Qwen--Qwen2.5-0.5B-Instruct",
    ) else {
        return;
    };
    let mut e = Engine::load(&dir, "denno-test-i8", Some(42), QuantMode::Int8).unwrap();
    let a = e.chat(None, &user_msg("1+1は?"), &greedy(50), |_| {}).unwrap();
    let b = e.chat(None, &user_msg("1+1は?"), &greedy(50), |_| {}).unwrap();
    assert_eq!(a.text, b.text, "int8 モードで同 seed → 同一出力が崩れた");
    assert_eq!(a.output_tokens, b.output_tokens);
    println!("int8 output: {}", a.text);
}

/// P15: Qwen3 (QK-Norm) の基準出力 + 決定論。
/// Qwen3-0.6B は chat template が thinking モード用のため greedy でも思考
/// セクション (`...`) が先頭に出る。基準文字列は参照実装
/// (HF transformers fp32, greedy, 同一 chat template) の出力と完全一致を
/// 検証済み (ai_work_sync/qwen3_ref.py、開発コンテナ内)。
/// (QK-Norm 経路のバッチ/単独ビット一致は llama.rs の単体テストで担保)
#[test]
#[ignore]
fn qwen3_qknorm_matches_baseline() {
    let Some(dir) = model_dir(
        "DENNO_QWEN3_DIR",
        "/home/dev/.cache/denno/hub/models--Qwen--Qwen3-0.6B",
    ) else {
        return;
    };
    let mut e = Engine::load(&dir, "denno-test-q3", Some(42), QuantMode::Fp32).unwrap();
    let a = e.chat(None, &user_msg("1+1は?"), &greedy(40), |_| {}).unwrap();
    let b = e.chat(None, &user_msg("1+1は?"), &greedy(40), |_| {}).unwrap();
    assert_eq!(a.text, b.text, "Qwen3 で同 seed → 同一出力が崩れた");
    assert_eq!(a.output_tokens, b.output_tokens);
    assert_eq!(
        a.text,
        "<think>\nOkay, the user is asking \"1+1は?\" which translates to \"What is 1+1?\" in English. They want to know the result of adding 1 and ",
        "unexpected output: {}",
        a.text
    );
}
