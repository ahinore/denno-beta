# 詳細(モデル構成・環境変数・その他のモデル)

[README に戻る](../README.ja.md) | [English](DETAILS.en.md)

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

prefix cache は文脈スロットの列で、`--kv-slots N`(または `DENNO_KV_SLOTS`)で本数を選ぶ。既定 2 は従来の main + aux(長い会話を主に、主よりずっと短い脇リクエストを aux に載せる)で、メモリ量も挙動も変えない。3 以上では全スロットを最長共通接頭辞で照合し、空きが無ければ最終使用が一番古いスロットを追い出す(LRU)。1 スロットあたり GDN 状態 117 MiB + KV 48 KiB/token、トークン上限は `DENNO_KV_SLOT_TOK`(既定はコンテキスト長)。

## 段ごとの内訳

rtx5090 / UD-IQ3_XXS の段ごとの内訳(温まった状態、`8ae66f334` の 1 つ前の開発ブランチで計測):

| | 合計 | 内訳 |
|---|---|---|
| 生成 | 16.3 ms/tok | expert 4.73(GPU に無い 8 % を CPU で計算する分 2.33 を含む)/ mix(hyper connection の混合)4.73 / ssm(GDN)3.52 / full attention 1.47 / head 0.73 / shared + router 0.80 |
| prefill 3,826 | 2.45 s | expert 1.32 s(転送待ち 1.09 s)/ ssm 0.62 / その他(PLE など)0.24 / mix 0.14 / full attention 0.14 |
| prefill 14k | 6.44 s | ssm 2.61 s / expert 1.88 / その他 1.05 / full attention 0.88 |

## serve-flash の環境変数


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
| `DENNO_TELEMETRY_TOPK=1` | sampler 直前の top-8 (id・確率・エントロピー) を TopK フレームで送る(既定 OFF。/2d の token probs 欄と 3D の候補表示に必要) |
| `DENNO_KV_SLOTS=N`、`DENNO_KV_SLOT_TOK` | 文脈スロット数(既定 2 = 従来の main + aux、3 以上で LCP + LRU) / N スロット経路で 1 スロットが持つトークン上限(既定はコンテキスト長)。CLI は `--kv-slots` |

## その他のモデル(dense 経路の `serve`)

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

## その他のモデル(旧計測)

| モデル | prefill | decode |
|---|---|---|
| Qwen3.5-27B, Q4_0 | 16k ctx 9.24 s / 65k ctx 38.03 s | 17.3 ms/token |
| Qwen3.5-9B, int8 | 5,686 tok で 13.18 s / 431 tok/s | 16.4 ms/token |

古い数値・棄却した試み・「正式値」の定義は [docs/TASKS.md](TASKS.md) と[開発史](history/README.md) にある。
