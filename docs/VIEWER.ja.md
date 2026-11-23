# denno-view(推論の可視化)

[README に戻る](../README.ja.md) | [English](VIEWER.en.md)

## 起動

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

*27 秒 / 1256x618 / 5 fps / 2.2 MB。上のプレイヤーは GitHub の添付として読み込んでいる(GitHub の markdown はリポジトリ内の動画をインライン再生しないため)。同じ mp4 は [`docs/assets/denno_3d.mp4`](assets/denno_3d.mp4) にも入れてある。*

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
