//! denno-engine: 最小限のLLM推論エンジン
//!
//! TensorFold構成の engine 相当。checkpoint(safetensors)読込、トークナイザ、
//! KVキャッシュ、サンプリング、デコードループを持つ。CPUのみ・自前カーネル。

pub mod engine;
pub mod kv_cache;
pub mod load;
pub mod model;
pub mod prefix_cache;
pub mod sampler;
pub mod tensor;
pub mod tokenizer;

#[cfg(any(test, feature = "test-utils"))]
pub mod testutil;

pub use engine::{ChatMessage, ChatOutput, Engine, GenParams, PrefixHit, StopReason};
