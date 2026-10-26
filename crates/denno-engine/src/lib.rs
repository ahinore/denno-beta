//! denno-engine: 最小限のLLM推論エンジン
//!
//! TensorFold構成の engine 相当。checkpoint(safetensors)読込、トークナイザ、
//! KVキャッシュ、サンプリング、デコードループを持つ。CPUのみ・自前カーネル。

pub mod engine;
pub mod gguf;
pub mod gdn;
pub mod kquant;
pub mod kquant_tables;
pub mod kv_cache;
pub mod load;
pub mod model;
pub mod ple;
pub mod prefix_cache;
pub mod prof;
pub mod quant;
pub mod sampler;
pub mod tensor;
pub mod tokenizer;
pub mod toolmask;

/// P17: CUDA int8 オフロード (feature = "cuda" のみ。GPU 無し環境では完全に除外)。
#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(any(test, feature = "test-utils"))]
pub mod testutil;

pub use engine::{ChatMessage, ChatOutput, ChatSession, Engine, GenParams, PrefixHit, StopReason};
pub use quant::{Linear, Quant8Linear, QuantMode};
pub use toolmask::ToolConstraint;

/// prof フックからテレメトリハブへの薄い委譲 (P13)。
/// prof.rs を denno-telemetry の直接依存から切り離すための関数。
pub fn telemetry_on() -> bool {
    denno_telemetry::enabled()
}

/// op の所要時間をテレメトリの OpSummary 累積に加算する。
pub fn telemetry_add_op(op: usize, ns: u64) {
    denno_telemetry::add_op(op, ns);
}

/// 層 forward の op 列記録 (P14: LayerRun フレーム用)。
pub fn telemetry_layer_begin(layer: u16) {
    denno_telemetry::layer_begin(layer);
}

pub fn telemetry_layer_flush() {
    denno_telemetry::layer_flush();
}
