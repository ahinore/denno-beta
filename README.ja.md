# denno-beta

[![CI](https://github.com/ahinore/denno-beta/actions/workflows/ci.yml/badge.svg)](https://github.com/ahinore/denno-beta/actions/workflows/ci.yml)

[English](README.md) | 日本語

Rust で書いた Anthropic Messages API 互換の LLM サーバー。推論カーネルは CPU 版も CUDA 版もすべて自前で書いていて、candle などの既存ライブラリは使っていない。いまの主な対象は **Qwen3.8-Flash-Next**(512 expert のスパース MoE)で、重み 84〜111 GB のモデルを VRAM 32 GB・RAM 64 GB の PC で動かす。

- expert を VRAM → RAM → SSD の順にキャッシュし、router が選んだものだけを引き上げる
- temperature 0 なら同じ入力に対して毎回同じ出力になる(CI で確認している)
- 推論の様子をブラウザで見られるビューア [denno-view](docs/VIEWER.ja.md) がある
- Intel Arc Pro B60 向けの版をブランチ `arc-port` で移植中

## 現時点の性能

Qwen3.8-Flash-Next を 1 リクエストずつ `/v1/messages` で計測した値。どれも温まった状態の中央値で、出力の md5 はどの回も同じ。

| 機材 | 重み | 生成(短い会話) | prefill 3,826 トークン | prefill 14,018 トークン |
|---|---|---|---|---|
| RTX 5090 32 GB | Q3(UD-IQ3_XXS) | **15.3 ms/tok(65 tok/s)** | **2.44 s** | **6.0〜6.3 s** |
| RTX 5090 32 GB | Q4(UD-Q4_K_XL) | 18.2〜19.3 ms/tok(53 tok/s) | 9.2〜9.8 s | 25.2〜25.9 s |
| RTX 3080 10 GB | Q3(UD-IQ3_XXS) | 116 ms/tok(8.6 tok/s) | 9.4 s | 27.3 s |

- **RTX 5090:** Ryzen 7 9800X3D、RAM 64 GB、2026-11-17 に `8ae66f334` で計測。
  - 14k を読んだあとの生成は、Q3 で 16.1〜16.5 ms/tok、Q4 で 27〜29 ms/tok。
  - Q4 は 111 GB あって RAM に収まらないので、prefill のたびに expert の一部を SSD から読む。
- **RTX 3080:** Core i9-13900K、RAM 64 GB、Windows 11 の WSL2 上の Docker で、2026-11-18 に計測。
  - RAM に全部は載らず、ページキャッシュの残り具合で時間が大きく変わる。表の値は温まったときの最速値。
  - 3 つの値は別々の設定で出した最速値で、1 つの設定で全部が出るわけではない(下の「VRAM が少ない GPU」を参照)。
  - 計測に使ったのは `main` より新しい開発版。
- **Intel Arc Pro B60 24 GB:** ブランチ `arc-port` の 2026-11-18 時点で、Q3 の生成が 41 ms/tok。
- 段ごとの内訳と以前の計測は [docs/DETAILS.ja.md](docs/DETAILS.ja.md) にある。

## 使い方

### 1. 重みを取る

[`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) の 2026-11-13 版(リビジョン `f5d175de`)を使う。ログインは要らない。

| 重み | 大きさ | 必要な RAM の目安 |
|---|---|---|
| Q3: `UD-IQ3_XXS`(既定) | 83.6 GB | 64 GB |
| Q4: `UD-Q4_K_XL` | 111.3 GB | 64 GB でも動くが、RAM が多いほど速い |

```bash
pip install -U huggingface_hub
hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --revision f5d175de2cf7f720d45d6064019a81fcf056e986 \
  --include "UD-IQ3_XXS/*" --local-dir ~/models/qwen38-flash-next-1113
```

Q4 を取るときは `--include "UD-Q4_K_XL/*"` に替える。`--dir` はフォルダ内の `*.gguf` を全部読むので、重みは 1 フォルダに 1 種類だけ置く。

### 2. ビルドして起動する

```bash
cargo build --release --features cuda -p denno-cli -p denno-engine

DENNO_CUDA=1 DENNO_PREFILL_BATCH=1 DENNO_PREFILL_BATCH_CHUNK=512 DENNO_STRIP_TOOLS=1 \
DENNO_START_PREFETCH=1 DENNO_GPU_WEIGHT_MB=12000 \
DENNO_EXPERT_GPU=0 DENNO_EXPERT_BATCH=1 DENNO_EXPERT_IQ_GPU=1 \
DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0 \
DENNO_GEMV_V3=1 DENNO_MIX_GPU=1 DENNO_ROUTER_GPU=1 \
DENNO_MOE_BATCH_PTX=crates/denno-engine/kernels/kernels_moe_batch.ptx \
./target/release/denno serve-flash \
  --dir ~/models/qwen38-flash-next-1113/UD-IQ3_XXS \
  --port 8098 --max-ctx 65536 --max-tokens 4096
```

- 上の性能表の RTX 5090(VRAM 32 GB)の設定。
- Q4 では `--dir` を `UD-Q4_K_XL` に替え、`DENNO_START_PREFETCH=0` にする。

### 3. 呼び出す

```bash
curl http://127.0.0.1:8098/v1/messages \
  -H "content-type: application/json" \
  -d '{"model":"qwen3.8-flash-next","max_tokens":256,
       "messages":[{"role":"user","content":"1+1は?"}]}'
```

```python
import anthropic
c = anthropic.Anthropic(base_url="http://127.0.0.1:8098", api_key="sk-test")
print(c.messages.create(model="qwen3.8-flash-next", max_tokens=256,
      messages=[{"role":"user","content":"こんにちは"}]).content[0].text)
```

- `"stream": true` にすると SSE で返る。ほかに `GET /v1/models`、`GET /health` がある。
- `/v1/chat/completions` はまだ無い。
- リクエストは 1 本ずつ処理する。続きの会話は前回の読み込み結果を使い回すので、応答が始まるまで 0.5 s ほどしかかからない。

### VRAM が少ない GPU(RTX 3080 10 GB の例)

- `crates/denno-engine/kernels/build_ptx.sh` の `compute_90` を自分の GPU の値(3080 なら `compute_86`)に替えて PTX を作り直す。
- `DENNO_GPU_WEIGHT_MB=5000`、`DENNO_EXPERT_GPU_MIN_HIT=0`、`RAYON_NUM_THREADS=8` にする。13900K では 24 スレッドにすると遅くなった。
- 生成を優先するなら expert は CPU に置いたまま(`DENNO_EXPERT_IQ_GPU` を付けない)にする。
- prefill を優先するなら `DENNO_EXPERT_IQ_GPU=1`、`DENNO_EXPERT_VRAM_MB=800`〜`1500` にする。

環境変数の一覧と、ほかのモデル(GPT-2、Qwen2.5、Qwen3.5-27B)を動かす `serve` の使い方は [docs/DETAILS.ja.md](docs/DETAILS.ja.md) にある。

## ドキュメント

- [詳細](docs/DETAILS.ja.md) — モデル構成、段ごとの内訳、環境変数、ほかのモデル
- [denno-view](docs/VIEWER.ja.md) — 可視化の起動、3D と 2D のビュー、テレメトリのフレーム
- [アーキテクチャ解説](docs/ARCHITECTURE.ja.md)、[数値方針](docs/NUMERIC_POLICY.ja.md)
- [開発史](docs/history/README.md)、[タスクと計測の手順](docs/TASKS.md)、[API 仕様](docs/api.md)

## ライセンス

コードは MIT または Apache License 2.0([LICENSE-MIT](LICENSE-MIT) / [LICENSE-APACHE](LICENSE-APACHE))。

重みはこのリポジトリに含めず、それぞれのライセンスに従う。Qwen3.8-Flash-Next は [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE) で、複製を渡すときは重みに同梱の `LICENSE` も一緒に渡す。参照した実装の出典は [docs/history/80_licenses.md](docs/history/80_licenses.md) にある(k-quant の形式と GDN の数式は llama.cpp、構成は Qwen のモデル実装と [TensorFold](https://github.com/ashhart/TensorFold))。
