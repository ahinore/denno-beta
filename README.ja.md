# denno-beta

[![CI](https://github.com/ahinore/denno-beta/actions/workflows/ci.yml/badge.svg)](https://github.com/ahinore/denno-beta/actions/workflows/ci.yml)

[English](README.md) | 日本語

Rust で書いた Anthropic Messages API 互換の LLM サーバー。
LLM の動作原理(トークン化 → forward pass → サンプリング → デコードループ)をコードで理解することを目的に、**推論カーネルを完全自前実装**している(candle 等の既存 LLM ライブラリは不使用)。構成の参考: [ashhart/TensorFold](https://github.com/ashhart/TensorFold)

- **単一バイナリ** `denno` — CLI(`pull` / `models` / `info` / `generate` / `serve` / `serve-flash`)+ axum HTTP サーバー + 推論エンジン。既定は CPU のみ、`--features cuda` ビルドで自前 CUDA カーネルを足す
- **対応モデル**
  - GPT-2 124M
  - Llama 系(RMSNorm + RoPE + SwiGLU + GQA): Qwen2.5-0.5B〜7B-Instruct、Qwen3-0.6B(QK-Norm、thinking モード)
  - Qwen3.5-27B(Gated DeltaNet ハイブリッド・GGUF から読み込み・Q4_0)
  - **Qwen3.8-Flash-Next**(arch `qwen4exp`)— スパース MoE。専用の配信経路 `serve-flash` を持つ
- **exact decoding**: seed 固定 + 温度 0 で同一入力 → ビット完全同一出力。連続バッチング・SIMD・CUDA 化を経ても壊さない規律で、CI の exactness テストが担保する
- **CUDA オフロード**: decode / prefill の両チェーンを自前 CUDA カーネルで GPU 化し、重み / KV / 隠れ状態を VRAM に常駐させる。数値契約は「線形層・rope・rmsnorm・swiglu は CPU とビット一致、緩和枠は attention のみ」([アーキテクチャ解説 §8](docs/ARCHITECTURE.ja.md)、[数値方針](docs/NUMERIC_POLICY.ja.md))
- 連続バッチング(複数セッションの decode を 1 forward に統合・KV スロットプール)、prefix cache(LCP 最大選択・GDN 状態スナップショット対応)、27B 経路の int8 量子化、repetition / frequency / presence penalty、no-repeat n-gram
- **expert の階層配置**(Flash-Next): 512 個の expert を VRAM → RAM → SSD(GGUF 直読み)の順にキャッシュし、層ごとに router の top-10 が必要なものだけを引き上げる。階層ごとに予算を持つので、重み 104 GB のモデルを遥かに小さい VRAM 枠で回せる
- **推論テレメトリ + ビューワ**: 推論を一切待たせないテレメトリ(`DENNO_TELEMETRY=1`、既定 OFF・read-only)と、別プロセスの `denno-view`。2D ダッシュボード(`/`)と、実構造をそのまま組んだ 3D ビュー(`/3d`)

## Qwen3.8-Flash-Next の構成

モデルカードではなく GGUF ヘッダから読んだ実値:

| 項目 | 値 |
|---|---|
| 層 | 48 — **full attention (QSA) 12 + GDN 36**、`(il+1) % 4 == 0` が QSA、`full_attention_interval = 4` |
| expert | 1 層 512 個、**トークンごとに top-10** を選択、加えて shared expert |
| 幅 | hidden 2560、expert FFN 640、context 262144 |
| attention | query 24 head / KV 2 head、head dim 256、indexer 4 head x 128 |
| PLE | n-gram 埋め込み層は **layer index 1 のみ**(1 トークン 1 回注入): 16 head = (ngram 3 − 1) × 8、表 16 本で約 28.8 GB を SSD に常駐 |
| 重み | UD-Q4_K_XL 分割 GGUF 約 104 GB — gate/up expert は Q4_K、down expert は Q5_1、attention / ssm / shared は Q8_0 |

`serve-flash` は dense な重みを常駐させ、expert をキャッシュとして扱う。配信経路は意図的に最小で、`/v1/messages` のみ(`/v1/chat/completions` は未実装)、1 リクエストずつ処理し、最長共通接頭辞を再利用する prefix cache を持つ。805 トークンぶん再利用できた続きの会話は 204.7 s → 5.7 s になる。

## 計測値

単体のコンシューマ GPU、1 リクエストずつの実測。どのプロセスも初回は重みのアップロードを払うので捨て、交互 A/B の中央値を出力ハッシュのアンカーとセットで載せる。

| モデル | prefill | decode |
|---|---|---|
| Qwen3.5-27B, Q4_0 | 16k ctx 9.24 s / 65k ctx 38.03 s | 17.3 ms/token |
| Qwen3.8-Flash-Next, UD-Q4_K_XL | 706 tok で 29 ms/token | 706 tok で 191 ms/token(38 tok の実行では 150 ms/token) |
| Qwen3.5-9B, int8 | 5,686 tok で 13.18 s / 431 tok/s | 16.4 ms/token |

Flash-Next の decode 内訳(入力 49 + 出力 128 トークン、193.9 ms/token): 選択 expert 71.9 ms(37 %)/ attention 62.2 ms(32 %)/ mix 31.8 / router 17.0 / shared expert 10.9。層あたり prefill 0.61 ms に対し decode 4.04 ms で 6.6 倍 — GEMV 形のバッチに固定費が乗る。

古い数値・棄却した試み・「正式値」の定義は [docs/TASKS.md](docs/TASKS.md) と[開発史](docs/history/README.md) にある。

## 使い方(Qwen の例)

```bash
# 1. モデルのダウンロード(HF hub キャッシュ ~/.cache/denno/hub に入る)
cargo run --release -- pull Qwen/Qwen2.5-0.5B-Instruct

# 2. サーバー起動(seed 固定で exact decoding)
cargo run --release -- serve Qwen/Qwen2.5-0.5B-Instruct --name denno --port 8080 --seed 42

# 3. API を叩く(Anthropic Messages 形式)
curl http://127.0.0.1:8080/v1/messages \
  -H "content-type: application/json" \
  -d '{"model":"denno","max_tokens":128,
       "messages":[{"role":"user","content":"1+1は?"}]}'
```

CLI で直接生成することもできる:

```bash
cargo run --release -- generate Qwen/Qwen2.5-0.5B-Instruct --prompt "1+1は?" --temperature 0 --seed 42
cargo run --release -- generate openai-community/gpt2 --prompt "The capital of France is" --raw --seed 42
```

Anthropic 公式の Python SDK からもそのまま使える:

```python
import anthropic
c = anthropic.Anthropic(base_url="http://127.0.0.1:8080", api_key="sk-test")
print(c.messages.create(model="denno", max_tokens=128,
      messages=[{"role":"user","content":"こんにちは"}]).content[0].text)
```

### Flash-Next

```bash
# --dir は分割 GGUF が置かれたディレクトリ
DENNO_CUDA=1 cargo run --release --features cuda -- \
  serve-flash --dir /models/Qwen3.8-Flash-Next/UD-Q4_K_XL \
              --port 8098 --max-ctx 65536 --max-tokens 1024
```

リクエスト形式は同じ `/v1/messages` で、変わるのはモデル ID だけ。実運用で効く環境変数:

| 変数 | 効果 |
|---|---|
| `DENNO_CUDA=1` | GPU 経路を有効化 |
| `DENNO_PREFILL_BATCH=1`、`DENNO_PREFILL_BATCH_CHUNK=512` | prefill を GPU バッチ GEMM で回す(chunk の既定は 8、実測で 512 が最良。1024 より上げると退行) |
| `DENNO_GPU_WEIGHT_MB`、`DENNO_EXPERT_VRAM_MB` | dense 重みと expert キャッシュの VRAM 予算 |
| `DENNO_EXPERT_GPU=1`、`DENNO_EXPERT_BATCH=1` | 選択 expert を GPU に載せる / まとめて処理する |
| `DENNO_STRIP_TOOLS=1` | プロンプトから `tools` 定義を落とす(40 ツールで prompt 6,979 トークンが 842 になる) |
| `DENNO_TELEMETRY`、`DENNO_TELEMETRY_ACT=1`、`DENNO_TELEMETRY_ARCH=1` | テレメトリ(この経路では既定 ON、`0` で無効)/ 層活性 / 構造系フレーム |

## 可視化(denno-view)を起動する

```bash
# 1. サーバーをテレメトリ ON で起てる(DENNO_TELEMETRY=1 以外は同一コマンド)
DENNO_TELEMETRY=1 cargo run --release -- serve Qwen/Qwen2.5-0.5B-Instruct --name denno --port 8080 --seed 42

# 2. 別ターミナルで denno-view を起動
#    (既定で ws://127.0.0.1:8080 のテレメトリ WS に接続し、:8093 で待受)
VIEW_TOKENIZER=~/.cache/denno/hub/models--Qwen--Qwen2.5-0.5B-Instruct/tokenizer.json \
  cargo run --release -p denno-view

# 3. ブラウザで http://localhost:8093/ を開く(ダッシュボード)。3D は /3d
```

**動作デモ** — 内蔵チャットで生成すると、モデル駆動プログラムが実装コードをステップ実行し、層 / KV / 変数パネルが連動してアニメーションする:

![denno-view 動作デモ](docs/assets/denno_view.gif)

3D ビュー(`/3d`)は絵ではなく実構造から組んである。48 層を Z 軸に並べ、各層に mixer(12 QSA / 36 GDN)、router、512 expert の 32x16 配置、shared expert、GR の read/write を置き、4 層ごとに hybrid block の枠を描く。expert は「いまどの階層にいるか」の高さ(VRAM / RAM / SSD)に置かれ、router が選ぶたびに階層を移動する。KV 帯は prefix cache の再利用分・今回書いた分・decode の読み出し位置を色で分け、予測ヘッドにはサンプラの実 top-8 候補が出て、選ばれたトークンがチャット欄へ飛ぶ。`f` キーでクォータニオンによるフライトモード(WASD + マウス、Shift 加速、Q/E 上下、Esc で俯瞰)。

`denno-view` の環境変数: `VIEW_UPSTREAM`(既定 `ws://127.0.0.1:8080/v1/telemetry/ws`)、`VIEW_PORT`(既定 8093)、`VIEW_RECORD`(フレームをファイルに追記)、`VIEW_REPLAY`(記録を `/replay` で配信)、`VIEW_CAPTURE`(連結フレーム列を `/capture.bin` で配信)、`VIEW_TOKENIZER`(トークン文字表示用の tokenizer.json)。`/chat` は上流のサーバーへ中継するので、内蔵チャットに CORS 設定は要らない。

## テレメトリのフレーム

`DNTL` マジック + version 1 バイト + kind 1 バイト + varint の payload。kind 番号はブランチを跨いで固定で、退役した番号は再利用しない。

| kind | フレーム | 内容 |
|---|---|---|
| 1 | `ServerInfo` | モデル ID・量子化・スロット数・コンテキスト長・層数 |
| 2 / 3 | `SessionAdmit` / `SessionDone` | KV スロットの取得 / 解放 |
| 4 | `Prefill` | prefill 1 forward のトークン数・所要 ns・**プロンプトのトークン id 列** |
| 5 | `Token` | decode で確定した 1 トークン(id と位置) |
| 6 | `Step` | バッチ decode の 1 forward(幅と ns) |
| 7 | `OpSummary` | op 別の累積 ns |
| 8 | `KvUsage` | スロットごとの KV 使用量 |
| 9 | `Gap` | 購読が追いつかず捨てたフレーム数 |
| 10 | `TopK` | サンプラの top-8(確率は f16 ビット)・選ばれたトークン・entropy |
| 11 | `LayerRun` | 1 層で実行された op 列(実行順) |
| 12 | `LayerAct` | 残差ストリームの活性(層ごとに絶対値の上位 K 個のニューロン index) |
| 13 | `RoleAct` | 役割別 RMS(attn / gdn / router / shared expert / PLE) |
| 14 | `ExpertResidency` | その層で VRAM に載っている expert |
| 15 | `TierStats` | VRAM / RAM / SSD の使用状況(1 Hz) |
| 16 | `RouterTopK` | router が実際に選んだ expert と gate 重み |

送出は「詰まったら落とす」チャネルなので、購読が遅いビューワが推論を待たせることはない。

## ドキュメント

- [アーキテクチャ解説(日本語)](docs/ARCHITECTURE.ja.md) / [Architecture (English)](docs/ARCHITECTURE.en.md) — 設計方針・推論フロー・KV キャッシュ・演算カーネル・モデルの違い・可視化の構成
- [開発史(索引)](docs/history/README.md) / [English](docs/history/en/README.md) — 巻に分けた読み物の開発史。ブートストラップ / API 機能 / モデル対応 / decode の速度 / prefill の速度 / GDN / 品質の切り分け / 計測・検証・運用。実測値と棄却した試みを含む
- [タスクと計測プロトコル](docs/TASKS.md)、[KPI の定義](docs/KPI.md)
- [API 仕様](docs/api.md)

## ライセンス

コード: MIT または Apache License 2.0 のデュアルライセンス — [LICENSE-MIT](LICENSE-MIT) / [LICENSE-APACHE](LICENSE-APACHE)。

重みはリポジトリに含めず、それぞれのライセンスに従う。参照した実装の出典は [docs/history/80_licenses.md](docs/history/80_licenses.md) にある: k-quant のブロック形式と GDN の数式は llama.cpp(MIT)、アーキテクチャは Qwen のモデル実装を参照している。
