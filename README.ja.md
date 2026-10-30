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
- **推論テレメトリ + ビューワ**: 推論を一切待たせないテレメトリ(`DENNO_TELEMETRY`、`serve-flash` 経路では既定 ON、`serve` では既定 OFF・read-only)と、別プロセスの `denno-view`。2D ダッシュボード(`/`)と、実構造をそのまま組んだ 3D ビュー(`/3d`)

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

## 使い方(Qwen3.8-Flash-Next)

Flash-Next はスパース MoE で、専用の `serve-flash` 経路で配信する。API は他と同じ `/v1/messages` で、`serve` と別経路なのはモデルが `Engine` 実装を持たないため(512 expert の配置と PLE の表に専用ローダが要る)。

### 重みの取得先

GGUF は Hugging Face Hub にあり、公開リポジトリなのでログインも同意ゲートも不要:

| | |
|---|---|
| リポジトリ | [`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)(量子化元は [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)) |
| 量子化 | `UD-Q4_K_XL`、4 分割、111.3 GB / 103.7 GiB |
| ライセンス | Qwen Community License 1.0。`LICENSE` が重みと同梱されているので、複製を渡すときも一緒に渡す |

```bash
pip install -U huggingface_hub
hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --include "UD-Q4_K_XL/*" --local-dir ~/models/qwen38-flash-next
```

これで `~/models/qwen38-flash-next/UD-Q4_K_XL/` に次が入る:

```
Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf   10.9 MB
Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf   49.9 GB
Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf   49.4 GB
Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf   12.1 GB
```

`--dir` はディレクトリ内の `*.gguf` をソート順に全部読むので、1 フォルダ 1 量子化にする。上の数字は 10 進 GB(`du` は 104 GiB と出る)。

```bash
# 1. CUDA 経路をビルド(フィーチャは denno-engine 側にある)
cargo build --release --features cuda -p denno-cli -p denno-engine

# 2. 重みを取得(上のコマンド)し、shard のディレクトリを --dir に渡す

# 3. サーバー起動
DENNO_CUDA=1 ./target/release/denno serve-flash \
  --dir ~/models/qwen38-flash-next/UD-Q4_K_XL \
  --port 8098 --max-ctx 65536 --max-tokens 1024
```

Anthropic Messages 形式で叩く。`stream: true` で SSE になる:

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

| エンドポイント | 内容 |
|---|---|
| `POST /v1/messages` | Messages API。JSON、`stream: true` なら SSE(`message_start` / `content_block_delta` / `message_delta` / `message_stop`) |
| `GET /v1/models` | 広告しているモデル ID(`--name`、既定は `qwen3.8-flash-next`) |
| `GET /health` | 死活確認 |
| `GET /v1/telemetry/ws` | `denno-view` 用のテレメトリ |

`/v1/chat/completions` は未実装(404)。リクエストは逐次処理で、KV スロットは 1 本しかないため 2 本目の呼び出しは 1 本目の完了を待つ。リクエスト間で prefix cache を保持し、805 トークンぶん再利用できた続きの会話は 204.7 s → 5.7 s になる。

実運用で効く環境変数:

| 変数 | 効果 |
|---|---|
| `DENNO_CUDA=1` | GPU 経路を有効化(付けなければ同じバイナリが CPU で動く。概ね一桁遅い) |
| `DENNO_PREFILL_BATCH=1`、`DENNO_PREFILL_BATCH_CHUNK=512` | prefill を GPU バッチ GEMM で回す(chunk の既定は 8、実測で 512 が最良。1024 より上げると退行) |
| `DENNO_GPU_WEIGHT_MB`、`DENNO_EXPERT_VRAM_MB` | dense 重みと expert キャッシュの VRAM 予算。dense 重みが先の枠を食い切ると expert が CPU に落ち、decode が概ね半分になる |
| `DENNO_EXPERT_GPU=1`、`DENNO_EXPERT_BATCH=1` | 選択 expert を GPU に載せる / まとめて処理する |
| `DENNO_STRIP_TOOLS=1` | プロンプトから `tools` 定義を落とす(40 ツールで prompt 6,979 トークンが 842 になる) |
| `DENNO_MAX_SYSTEM=N` | system プロンプトの先頭 N 文字だけ残す |
| `DENNO_TELEMETRY`、`DENNO_TELEMETRY_ACT=1`、`DENNO_TELEMETRY_ARCH=1` | テレメトリ(この経路では既定 ON、`0` で無効)/ 層活性 / 構造系フレーム |

### その他のモデル(dense 経路の `serve`)

`serve` は GPT-2、Llama 系(Qwen2.5-0.5B〜7B-Instruct、Qwen3-0.6B)、GGUF の Qwen3.5-27B を扱い、連続バッチング・prefix cache・int8 量子化を備える:

```bash
# 1. HF hub キャッシュ(~/.cache/denno/hub)へダウンロード
cargo run --release -p denno-cli -- pull Qwen/Qwen2.5-0.5B-Instruct

# 2. サーバー起動(seed 固定で exact decoding)
cargo run --release -p denno-cli -- serve Qwen/Qwen2.5-0.5B-Instruct --name denno --port 8080 --seed 42

# 3. サーバーを立てずに単発生成
cargo run --release -p denno-cli -- generate Qwen/Qwen2.5-0.5B-Instruct --prompt "1+1は?" --temperature 0 --seed 42
cargo run --release -p denno-cli -- generate openai-community/gpt2 --prompt "The capital of France is" --raw --seed 42
```

## 可視化(denno-view)を起動する

```bash
# 1. サーバー起動(serve-flash はテレメトリ既定 ON。DENNO_TELEMETRY=0 で無効)
DENNO_CUDA=1 ./target/release/denno serve-flash \
  --dir ~/models/flash-next --port 8098 --max-ctx 65536 --max-tokens 1024

# 2. 別ターミナルで、そのサーバーに向けて denno-view を起動
#    (VIEW_PORT の既定は 8093)
VIEW_UPSTREAM=ws://127.0.0.1:8098/v1/telemetry/ws \
  cargo run --release -p denno-view

# 3. ブラウザで http://localhost:8093/ を開く(ダッシュボード)。3D は /3d
```

dense の `serve` 経路を使う場合は、起動時に `DENNO_TELEMETRY=1` を渡し、`VIEW_TOKENIZER` にモデルの `tokenizer.json` を指定するとトークン id が文字になる(GGUF 経路はトークナイザを内包しているので、未指定なら id のまま表示される)。

**動作デモ** — 3D ビュー(`/3d`)で Flash-Next と実際に会話している様子。プロンプトが入力レールでトークン化され、router が SSD から VRAM へ expert を引き上げ、層の柱がニューロン単位で光り、予測ヘッドにサンプラの top-8 候補が出て、選ばれたトークンがチャット欄へ飛ぶ:

https://github.com/user-attachments/assets/3eb62bce-bfc8-44e7-adc8-bd95d60360f1

*27 秒 / 1256x618 / 5 fps / 2.2 MB。上のプレイヤーは GitHub の添付として読み込んでいる(GitHub の markdown はリポジトリ内の動画をインライン再生しないため)。同じ mp4 は [`docs/assets/denno_3d.mp4`](docs/assets/denno_3d.mp4) にも入れてある。*

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

重みはリポジトリに含めず、それぞれのライセンスに従う。Qwen3.8-Flash-Next は [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)(利用・改変・再配布は可。複製には著作権表示と本ライセンスを同梱すること。Model as a Service として提供する場合は Qwen との別契約が要る)。参照した実装の出典は [docs/history/80_licenses.md](docs/history/80_licenses.md) にある: k-quant のブロック形式と GDN の数式は llama.cpp(MIT)、アーキテクチャは Qwen のモデル実装を参照している。
