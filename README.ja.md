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
- **expert の階層配置**(Flash-Next): 512 個の expert を VRAM → RAM → SSD(GGUF 直読み)の順にキャッシュし、層ごとに router の top-10 が必要なものだけを引き上げる。階層ごとに予算を持つので、重み 84〜111 GB のモデルを 32 GB の VRAM と 64 GB の RAM で回せる
- **Intel GPU 版(移植中)**: Level Zero + OpenCL C の自前カーネルで Arc Pro B60 に載せる移植を、ブランチ `arc-port` で進めている(`main` にはまだ入っていない)
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
| 重み(既定) | unsloth の `UD-IQ3_XXS`(2026-11-13 版)分割 GGUF 83.6 GB — expert は層ごとに IQ3_XXS / IQ2_S / IQ4_NL / IQ4_XS などを混在、attention / ssm 出力 / shared expert は主に Q6_K、hyper connection と PLE は Q8_0、PLE の表(Engram)28.8 GB は IQ4_NL |
| 重み(大きい版) | 同じリポジトリの `UD-Q4_K_XL`(2026-11-13 版)111.3 GB。RAM に収まらないので expert の一部を毎回 SSD から読む |

`serve-flash` は dense な重みを常駐させ、expert をキャッシュとして扱う。配信経路は意図的に最小で、`/v1/messages` のみ(`/v1/chat/completions` は未実装)、1 リクエストずつ処理し、最長共通接頭辞を再利用する prefix cache を持つ。805 トークンぶん再利用できた続きの会話は 204.7 s → 5.7 s になる。

## 計測値

### Qwen3.8-Flash-Next(2026-11-17 実測)

1 リクエストずつ、HTTP の `/v1/messages` に投げた実測。サーバーは計測ごとに起動し直し、最初の条件はページキャッシュが冷えているので基準から外す(温まった状態の 2 条件 × 2 回)。prefill の「cold」は会話の prefix cache が無い状態から読んだ値。出力は temperature 0 で、どの回も同じ md5 になる。

計測機:

- **rtx5090**: GeForce RTX 5090 32 GB(PCIe Gen5 x16)、Ryzen 7 9800X3D、RAM 64 GB、NVMe SSD。コード `8ae66f334`(この README の commit の親)
- **arc**: Intel Arc Pro B60 24 GB(PCIe Gen4 x8)、Ryzen 7 8700G、RAM 64 GB、NVMe SSD。ブランチ `arc-port` の `a4f891d28`

| 機材 / 重み | 生成(短い会話) | 生成(14k 読んだあと) | prefill 3,826 トークン | prefill 14,018 トークン | 2 ターン目の応答開始 |
|---|---|---|---|---|---|
| rtx5090 / UD-IQ3_XXS(既定) | **15.3 ms/tok(65 tok/s)** | 16.1〜16.5 ms/tok(61 tok/s) | **2.44 s(1,570 tok/s)** | **6.00〜6.25 s(2,300 tok/s)** | 0.45〜0.51 s |
| rtx5090 / UD-Q4_K_XL | 18.2〜19.3 ms/tok(53 tok/s) | 27〜29 ms/tok(35 tok/s) | 9.2〜9.8 s(400 tok/s) | 25.2〜25.9 s(550 tok/s) | 0.50〜0.60 s |
| arc / UD-IQ3_XXS(参考値、文章が崩れる) | 56 ms/tok(18 tok/s) | 100 ms/tok | 29.4〜29.8 s(130 tok/s) | 107.8 s(130 tok/s) | — |
| arc / swift15 IQ3_S(旧既定) | 88〜90 ms/tok(11 tok/s) | 115 ms/tok | 15.1〜15.4 s(250 tok/s) | 62.6 s(220 tok/s) | — |

- 生成の値は rtx5090 では 128 トークン出力したときの中央値。同じ会話の 2 回目(expert が VRAM に揃った状態)は 13 ms/tok(77 tok/s)まで下がる。
- UD-Q4_K_XL は 111 GB あって RAM に収まらず、3,826 トークンの prefill では毎回約 44 GB を SSD から読む(5.95 GB/s で頭打ち)。
- サーバーを起動した直後の最初のリクエストはページキャッシュが冷えていて、UD-IQ3_XXS の 3,826 トークンで 3.8 s、14k で 7.8 s かかった。
- arc の Intel GPU 版はまだ移植中。プロセスごとに 3,826 トークン → 短い質問 → 同じ質問の順で投げ、短い質問の 2 本目(expert が VRAM に揃った状態)を生成の値にした(3 回の値が一致)。
  - UD-IQ3_XXS は速度は出るが、28 トークンの質問への答えが 44 トークン目で役割名 `user` を出して止まるなど、文章が崩れる。新モデルで変わった型の扱いがまだ合っていない。
  - swift15 IQ3_S は文章は正しいが、同じプロセスの 3 本目(短い質問の 2 本目)で毎回 GPU がページフォールトを起こして止まる(`ZE_RESULT_ERROR_DEVICE_LOST`)。そのため生成は短い質問の 1 本目(expert を VRAM へ上げている途中)の値。

rtx5090 / UD-IQ3_XXS の段ごとの内訳(温まった状態、`8ae66f334` の 1 つ前の開発ブランチで計測):

| | 合計 | 内訳 |
|---|---|---|
| 生成 | 16.3 ms/tok | expert 4.73(GPU に無い 8 % を CPU で計算する分 2.33 を含む)/ mix(hyper connection の混合)4.73 / ssm(GDN)3.52 / full attention 1.47 / head 0.73 / shared + router 0.80 |
| prefill 3,826 | 2.45 s | expert 1.32 s(転送待ち 1.09 s)/ ssm 0.62 / その他(PLE など)0.24 / mix 0.14 / full attention 0.14 |
| prefill 14k | 6.44 s | ssm 2.61 s / expert 1.88 / その他 1.05 / full attention 0.88 |

### その他のモデル(旧計測)

| モデル | prefill | decode |
|---|---|---|
| Qwen3.5-27B, Q4_0 | 16k ctx 9.24 s / 65k ctx 38.03 s | 17.3 ms/token |
| Qwen3.5-9B, int8 | 5,686 tok で 13.18 s / 431 tok/s | 16.4 ms/token |

古い数値・棄却した試み・「正式値」の定義は [docs/TASKS.md](docs/TASKS.md) と[開発史](docs/history/README.md) にある。

## 使い方(Qwen3.8-Flash-Next)

Flash-Next はスパース MoE で、専用の `serve-flash` 経路で配信する。API は他と同じ `/v1/messages` で、`serve` と別経路なのはモデルが `Engine` 実装を持たないため(512 expert の配置と PLE の表に専用ローダが要る)。

### 重みの取得先

GGUF は Hugging Face Hub にあり、公開リポジトリなのでログインも同意ゲートも不要:

| | |
|---|---|
| リポジトリ | [`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)(量子化元は [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)) |
| 量子化(既定) | `UD-IQ3_XXS`、4 分割、83.6 GB。検証している版はリビジョン `f5d175de`(2026-11-13) |
| ライセンス | Qwen Community License 1.0。`LICENSE` が重みと同梱されているので、複製を渡すときも一緒に渡す |

```bash
pip install -U huggingface_hub
hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --revision f5d175de2cf7f720d45d6064019a81fcf056e986 \
  --include "UD-IQ3_XXS/*" --local-dir ~/models/qwen38-flash-next-1113
```

これで `~/models/qwen38-flash-next-1113/UD-IQ3_XXS/` に次が入る:

```
Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf   10.9 MB  (メタデータのみ)
Qwen3.8-Flash-Next-UD-IQ3_XXS-00002-of-00004.gguf   49.9 GB
Qwen3.8-Flash-Next-UD-IQ3_XXS-00003-of-00004.gguf    4.9 GB
Qwen3.8-Flash-Next-UD-IQ3_XXS-00004-of-00004.gguf   28.8 GB  (PLE の表)
```

- `--dir` はディレクトリ内の `*.gguf` をソート順に全部読むので、1 フォルダ 1 量子化にする。上の数字は 10 進 GB。
- RAM に余裕があるなら、大きい版の `UD-Q4_K_XL`(111.3 GB)も同じ手順で取れる。`--include "UD-Q4_K_XL/*"` に替え、起動時に `DENNO_START_PREFETCH=0` を足す。
- 11-13 版は先頭ファイル(メタデータのみ)の末尾に 32 バイト境界までの詰め物が無い。これを読めるのは 2026-11-16 以降の denno(commit `42500e5`)。

```bash
# 1. CUDA 経路をビルド(フィーチャは denno-engine 側にある)
cargo build --release --features cuda -p denno-cli -p denno-engine

# 2. 重みを取得(上のコマンド)し、shard のディレクトリを --dir に渡す

# 3. サーバー起動(計測値と同じ設定。VRAM 32 GB 向け)
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
| `DENNO_EXPERT_IQ_GPU=1`、`DENNO_EXPERT_GPU_MIN_HIT=0` | IQ 系の expert を VRAM に置いて GPU で計算する / 当たり率の下限を付けずに全部 GPU に回す |
| `DENNO_EXPERT_GPU=1`、`DENNO_EXPERT_BATCH=1` | 選択 expert を GPU に載せる(旧経路)/ まとめて処理する |
| `DENNO_START_PREFETCH=1` | 起動時に expert と PLE の表をページキャッシュへ読んでから待ち受ける(起動が十数秒延び、最初のリクエストが速くなる) |
| `DENNO_GEMV_V3=1`、`DENNO_MIX_GPU=1`、`DENNO_ROUTER_GPU=1` | decode の GEMV・mix・router を GPU で回す |
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
  --dir ~/models/qwen38-flash-next-1113/UD-IQ3_XXS --port 8098 --max-ctx 65536 --max-tokens 1024

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

2D ビュー(`/2d`)は、Qwen の公式構成図と同じ並びの図の上で介入(`/v1/intervene`)とステアリングを操作する。左に 48 層の列、右に Hybrid block ×12 の詳細(各層の Expanded residual の 4 stream、GR Read / Write、Hidden state、Gated DeltaNet / Qwen Sparse Attn、MoE の router・shared expert・512 expert、層 2 の N-gram 入力)、上に Prediction head と MTP、下に embedding を置く。ホイールで拡大縮小、ドラッグで移動、ダブルクリックでその部品へ寄り、`0` で全体に戻る。拡大の度合いで細部を出し分ける(全体では層ごとの縮小画像、寄ると expert 1 つずつと番号)。クリックで選択、Shift+ドラッグで範囲選択。従来の層タイル表示は `wall` ボタン(または `?view=wall`)で残している。右のツリーは同じ構成要素を `Hybrid block → 層 → residual / mixer / MoE → expert` の木にしたもので、チェックが選択になる。「switch off / on checked」で off / on を切り替える。グループのチェックは配下の全部。ツリーの上の選択式でノードを選べる。

```
expert: count == 0 and layer >= 40              # 一度も選ばれていない後段の expert
expert: count >= 0.5 * lmax(count)              # 層内最多の半分以上選ばれた expert
expert: rate > 0.05 | top 5 by count per layer  # 層ごとに上位 5 個
mixer: kind == "qsa" and layer % 8 == 3; moe: layer in [5, 9, 17]
```

選択式は `[種類:] 条件 [| top N by 式 [per layer|hybrid]] [| limit N]` を `;` でつないだもの。種類は `expert mixer moe shared ple stream block layer hybrid residual ffn experts *` から選び、省略すると介入できる全種類になる。条件には値 `layer hybrid pos index kind type count rate rank act w acc topk off sel vram skip tokens n_expert n_used n_hc` を使い、`+ - * / % ^` の四則演算、連鎖比較(`0 <= layer < 8`)、`and or not`、`in a..b`、`in [..]` が書ける。関数は `abs min max sqrt log exp floor ceil round clamp`、集約は同じ層の `lmax lmin lmean lsum` とモデル全体の `gmax gmin gmean gsum`。`count` はビューアのリセット(clear residual)からの活性化回数で、expert ならルータの top-k に入った回数、ブロックなら役割の値が 0 より大きかったトークン数。文法の詳細は `crates/denno-view/src/nodes2d.mjs` の先頭にある。

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
