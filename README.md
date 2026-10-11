# denno-beta

Rust 製の最小限 Anthropic Messages API 互換 LLM サーバー。
LLM の動作(tokenize → forward → sample → decode)を理解するために、
[ashhart/TensorFold](https://github.com/ashhart/TensorFold) の構成を参考に実装している。

- 構成: `CLI(pull/models/info/generate/serve)` + `server(Anthropic互換API)` + `engine(safetensors読込・トークナイザ・KVキャッシュ・サンプラ)`
- 推論カーネルは自前実装(matrixmultiply + rayon、CPUのみ)。candle 等は使わない
- 対応ファミリー: GPT-2、Llama系(RMSNorm+RoPE+SwiGLU+GQA、Qwen2.5-0.5B-Instruct など)

## 使い方

```bash
cargo run --release -- pull openai-community/gpt2
cargo run --release -- info openai-community/gpt2
cargo run --release -- generate openai-community/gpt2 --prompt "Once upon a time" --raw --max-tokens 32
cargo run --release -- serve openai-community/gpt2 --name gpt2 --port 8080
```

Qwen2.5-0.5B-Instruct(チャット用、bf16 約1GB):

```bash
cargo run --release -- pull Qwen/Qwen2.5-0.5B-Instruct
cargo run --release -- serve Qwen/Qwen2.5-0.5B-Instruct --name claude-local --api-key sk-test
```

API:

```bash
curl http://127.0.0.1:8080/v1/messages \
  -H "content-type: application/json" \
  -H "x-api-key: sk-test" \
  -d '{"model":"claude-local","max_tokens":256,
       "messages":[{"role":"user","content":"こんにちは"}]}'
```

Python SDK でも可:

```python
import anthropic
c = anthropic.Anthropic(base_url="http://127.0.0.1:8080", api_key="sk-test")
print(c.messages.create(model="claude-local", max_tokens=256,
      messages=[{"role":"user","content":"こんにちは"}]).content[0].text)
```

詳細は `docs/PLAN.md`(実装計画)と `docs/api.md`(API仕様)を参照。
