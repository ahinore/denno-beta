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

# 3. ブラウザで http://localhost:8093/ を開く(ダッシュボード)。/2d が層マップ、/3d が 3D
```

dense の `serve` 経路を使う場合は、起動時に `DENNO_TELEMETRY=1` を渡し、`VIEW_TOKENIZER` にモデルの `tokenizer.json` を指定するとトークン id が文字になる(GGUF 経路はトークナイザを内包しているので、未指定なら id のまま表示される)。

## ページ

| パス | 内容 |
|---|---|
| `/` | ダッシュボード: ライブレート、KV / VRAM / レイテンシ、op の所要時間、入力トークン、op グラフとコード欄、API の例 |
| `/2d` | 構成図の上での介入・ステアリング、プリセット、Abliterate の作業台 |
| `/3d` | 実構造から組んだ 3D ビュー: 層・expert・KV 帯・予測ヘッド・チャット |

3 ページとも、ビューア自身の origin の `GET /ws` につなぎ、上流のテレメトリを中継する。`/2d` は REST も `/api/*` で上流へ中継し、各ページのチャットは `/chat`(上流の `/v1/messages` へ中継)を叩くので CORS 設定は要らない。

**動作デモ** — 3D ビュー(`/3d`)で Flash-Next と実際に会話している様子。プロンプトが入力レールでトークン化され、router が SSD から VRAM へ expert を引き上げ、層の柱がニューロン単位で光り、予測ヘッドにサンプラの top-8 候補が出て、選ばれたトークンがチャット欄へ飛ぶ:

https://github.com/user-attachments/assets/3eb62bce-bfc8-44e7-adc8-bd95d60360f1

*27 秒 / 1256x618 / 5 fps / 2.2 MB。上のプレイヤーは GitHub の添付として読み込んでいる(GitHub の markdown はリポジトリ内の動画をインライン再生しないため)。同じ mp4 は [`docs/assets/denno_3d.mp4`](assets/denno_3d.mp4) にも入れてある。*

## ダッシュボード (`/`)

- ヘッダ: サーバー情報、使用中のセッション、落としたフレーム(`gap`)、`speed` スライダ(0.5–8、リプレイの再生速度)、リプレイの操作系(`replay` で live / replay を切替、`▶` 再生 / 一時停止、`⏮` 先頭へ、`◂` / `▸` ステップ、位置スライダ。`VIEW_REPLAY` を設定したときだけ有効)。
- 左: KV スロット、`hidden x (residual)`、`logits top-k`、`entropy (nat)`、`chosen token`、直近ステップの op 別所要時間、decode / prefill / VRAM のライブレート(直近 90 秒)、KV / VRAM / latency の使用率、OpenAI 互換 `/v1/chat/completions` と Anthropic 互換 `/v1/messages` の例(コピーボタンと `curl` 付き)。
- 中央: 直近プロンプトの入力トークン、op グラフ、疑似コードと実装を切り替えるコード欄(`real code ▸` / `◂ pseudo code`)。

キーボード: `2` で `/2d`、`3` で `/3d`、`R` で再読み込み。入力欄や contenteditable に打っている間は効かない。

## 2D ビュー (`/2d`)

2D ビューは、Qwen の公式構成図と同じ並びの図の上で介入(`/v1/intervene`)とステアリングを操作する。左に 48 層の列、右に Hybrid block ×12 の詳細(各層の Expanded residual の 4 stream、GR Read / Write、Hidden state、Gated DeltaNet / Qwen Sparse Attn、MoE の router・shared expert・512 expert、層 2 の N-gram 入力)、上に Prediction head と MTP、下に embedding を置く。

操作: ホイールで拡大縮小、ドラッグで移動、ダブルクリックでその部品へ寄り、`0` で全体に戻る。`+` / `=` と `-` でも拡大縮小、`Esc` で選択解除。拡大の度合いで細部を出し分ける(全体では層ごとの縮小画像、寄ると expert 1 つずつと番号)。クリックで選択、Shift+ドラッグで範囲選択。従来の層タイル表示は `wall` ボタン(または `?view=wall`)で残している。

ヘッダ:

- `panes`: `chat` / `tree` / `ctrl` で 3 つのサイドペインを開閉する。
- `colour`: `per token`(最後のトークン)、`accumulated`(clear residual からの累積)、`ablit`(Abliterate タブの層ごとの分離スコア)。ヘッダの `ablit` チップとパネルの `colour` ボタンは同じ切替。
- `model`: モデルを選ぶとサーバーがそのモデルで起動し直す(`POST /v1/model/select`)。読み込みの間は使えない。
- `epoch` と `cache` はテレメトリの epoch と現在の文脈キャッシュ。バナーに介入・ステアリングの有無が出る。

右のツリーは同じ構成要素を `Hybrid block → 層 → residual / mixer / MoE → expert` の木にしたもので、チェックが選択になる。「switch off / on checked」で off / on を切り替える。グループのチェックは配下の全部。ツリーの上の選択式でノードを選べる。

```
expert: count == 0 and layer >= 40              # 一度も選ばれていない後段の expert
expert: count >= 0.5 * lmax(count)              # 層内最多の半分以上選ばれた expert
expert: rate > 0.05 | top 5 by count per layer  # 層ごとに上位 5 個
mixer: kind == "qsa" and layer % 8 == 3; moe: layer in [5, 9, 17]
```

選択式は `[種類:] 条件 [| top N by 式 [per layer|hybrid]] [| limit N]` を `;` でつないだもの。種類は `expert mixer moe shared ple stream block layer hybrid residual ffn experts *` から選び、省略すると介入できる全種類になる。条件には値 `layer hybrid pos index kind type count rate rank act w acc topk off sel vram skip tokens n_expert n_used n_hc` を使い、`+ - * / % ^` の四則演算、連鎖比較(`0 <= layer < 8`)、`and or not`、`in a..b`、`in [..]` が書ける。関数は `abs min max sqrt log exp floor ceil round clamp`、集約は同じ層の `lmax lmin lmean lsum` とモデル全体の `gmax gmin gmean gsum`。`count` はビューアのリセット(clear residual)からの活性化回数で、expert ならルータの top-k に入った回数、ブロックなら役割の値が 0 より大きかったトークン数。文法の詳細は `crates/denno-view/src/nodes2d.mjs` の先頭にある。

選択式のボタン: `select` は一致で選択を置き換え(入力欄で `Ctrl+Enter`)、`+ add` は選択に足し(`Shift+Enter`)、`− remove` は選択から外し、`reveal` は最初の一致をツリーで開く。

操作パネル:

- `token probs`: サンプラ直前の top-8 とエントロピーをトークンごとに表示。サーバーで `DENNO_TELEMETRY_TOPK=1` が要る(既定 off)。履歴の帯をホバー / クリックするとそのトークンの内訳に固定できる。
- `intervention`: off になっている expert / block / stream の数、`switch off selected`、`switch on selected`、`clear selection`、`clear all off`。
- `steer`: スナップショットの記録(`record` / `stop`)、一覧、`layers [from,to)`、`alpha`、`mode`(`subtract` / `project`)、`terms`、`apply`、`clear apply`。
- `preset`: チェック中のノード・選択式・off の設定・ステアリングを JSON に `export`。プリセットを読み込み、`apply loaded` で介入を送って選択を戻す。`clear loaded` は表示だけ消す。
- `abliterate`: A/B の abliteration 作業台と OFF/ON の一括評価。[`ABLITERATION_GUIDE.md`](ABLITERATION_GUIDE.md) を参照。
- `state`: KV main / aux、expert の常駐(VRAM / RAM / mmap / coverage)、`clear KV cache`(`DELETE /v1/cache`。次のチャットは最初から prefill)、`clear residual`(per token / accumulated の表示を 0 に戻す)。
- `requests`: 直近 14 件の REST 呼び出しと状態。

テレメトリの帯: `ttft`、`tok/s`、`vram hit`、`kv reuse`、`mtp accept` と、grid / cell / ws / frames / tokens / last / vram / skip の各値。凡例の `vram` チェックで VRAM に常駐している expert に点が付く。

チャット: `Enter` 送信、`Shift+Enter` 改行、`stop` で生成中断(`POST /v1/abort`)、`clear chat` は画面の履歴だけ消す(サーバーの文脈は `clear KV cache`)、`続きを生成` で途切れた応答を継続する。

`/2d` のキーボード: `0` 全体、`+` / `=` 拡大、`-` 縮小、`Esc` 選択解除、選択式の入力欄で `Ctrl+Enter` / `Shift+Enter`。

## 3D ビュー (`/3d`)

3D ビュー(`/3d`)は絵ではなく実構造から組んである。48 層を Z 軸に並べ、各層に mixer(12 QSA / 36 GDN)、router、512 expert の 32x16 配置、shared expert、GR の read/write を置き、4 層ごとに hybrid block の枠を描く。expert は「いまどの階層にいるか」の高さ(VRAM / RAM / SSD)に置かれ、router が選ぶたびに階層を移動する。KV 帯は prefix cache の再利用分・今回書いた分・decode の読み出し位置を色で分け、予測ヘッドにはサンプラの実 top-8 候補が出て(`DENNO_TELEMETRY_TOPK=1` が要る)、選ばれたトークンがチャット欄へ飛ぶ。

パネル: モデルの状態(現在層と mixer の種類、点灯ニューロン数、fps、VRAM / RAM / DISK の expert 数、テレメトリの状態、prefill / decode の回数、階層の統計、PLE、トークンと KV の位置、プロンプト文、top-1 確率 / 選ばれたトークン / エントロピー)、凡例、48 層の帯(バーをクリックするか `[` / `]`)、層インスペクタ(層・mixer の種類・expert の要約・活性バー)、キーボードの凡例。

テレメトリが無いときはシミュレーションのデモ(`mode: demo (routing simulated)`)で動き、フレームが届くとライブに切り替わる。`?replay=1` は websocket を開かず `VIEW_CAPTURE` の `/capture.bin` から記録を流す。チャットは `/chat` を使い、`?api=http://host:8098` を渡すとそのサーバーへ直接つなぐ。

`/3d` のキーボード:

| キー | 動作 |
|---|---|
| `1` | カメラ: Overview |
| `2` | カメラ: Front(残差レーンを正面から) |
| `3` | カメラ: Top(層の並びを真上から) |
| `4` | カメラ: Side(expert グリッドを真横から) |
| `5` | カメラ: Reset |
| `[` / `]` | 層インスペクタ: 前 / 次の層 |
| `p` | fps オーバーレイの表示切替 |
| `f` | フライトモード(WASD + マウス、Shift 加速、Q/E 上下、`Esc` で俯瞰へ) |
| `space` | デモの一時停止 / 再開 |
| `c` | チャット欄の表示切替 |
| `?` | キーボード凡例の表示切替(`Esc` で閉じる) |

マウス: ドラッグで回転、ホイールで拡大縮小。

## 環境変数

`denno-view`:

- `VIEW_UPSTREAM` — 上流のテレメトリ websocket(既定 `ws://127.0.0.1:8080/v1/telemetry/ws`)。
- `VIEW_PORT` — 待受ポート(既定 8093)。
- `VIEW_RECORD` — 受けたフレームをファイルに追記。
- `VIEW_REPLAY` — 記録を `/replay` で配信(ダッシュボードのリプレイ操作が有効になる)。
- `VIEW_CAPTURE` — 連結フレーム列を `/capture.bin` で配信(`/3d?replay=1`)。
- `VIEW_TOKENIZER` — トークン文字表示用の `tokenizer.json`。

`serve-flash` のテレメトリ制御:

| 変数 | 既定 | 効果 |
|---|---|---|
| `DENNO_TELEMETRY` | on | `0` でテレメトリのチャネルもスレッドも作らない |
| `DENNO_TELEMETRY_ACT` | off | `1` で `LayerAct`(層ごとの絶対値上位 K ニューロン)を送る。`DENNO_TELEMETRY_ACT_TOPK` で K(既定 64) |
| `DENNO_TELEMETRY_TOPK` | off | `1` で `TopK`(サンプラの top-8 とエントロピー)を送る。`/2d` の token probs と 3D の予測ヘッドが使う |
| `DENNO_TELEMETRY_ARCH` | off | `1` で `RoleAct` と `RouterTopK` を送る |
| `DENNO_TELEM_EXPERT_SKIP` | off | `1` で `ExpertSkip`(router が選びたかったが介入で切った expert)も送る。`DENNO_TELEMETRY_ARCH` も要る |

### 層のドリルダウン (task 857)

層の箱(左の列)か右の層パネルをクリックすると、その層のドリルダウンが地図の左下に開く。中身は、その層の router が選んだ top-k の expert id とゲート重み(生値。棒は最大値で正規化)、VRAM hit / CPU miss のバッジ(`ExpertResidency` が来ていないときは `?`、介入で切った expert は `skip`)、直近 16 トークンの出現頻度のスパークライン(当たったトークンが縦棒)。`×` か Esc で閉じ、別の層を押すとその層に切り替わる。層の壁(`wall`)でも層をクリックすれば同じものが開く。hit / miss は `DENNO_EXPERT_GPU=1` で expert が VRAM に載っているときに意味を持つ(`DENNO_EXPERT_VRAM_MB>0`)。数字がまだ無いときは「RouterTopK 待ち」「ExpertResidency 待ち」とだけ出し、推測の値は出さない。

## Abliterate の作業台 (`/2d` の右パネル)

右の操作列にある `abliterate` の箱が、方向 ablation の A/B 対比の作業台。残差ストリームのステアリング (crates/denno-engine/src/model/steer.rs) と `/v1/steer` の上に載っている。

1. 2 つのプロンプト集合を貼る (1 行 1 プロンプト)。A が消したい向き、B が残したい向き。`load sample` で小さな例が入る。
2. `capture mode` を選ぶ。`last_prompt_token`(既定。prefill の最終トークンだけを層ごとに記録し、指示境界の difference-in-means になる)か `all`(prefill と decode の全トークンの平均)。
3. `capture A & B` が `ablit_A` → `ablit_B` の順に記録する。各プロンプトの前に KV を捨ててプロンプト全体を prefill させ、`/chat` を `max_tokens: 1, temperature: 0` で流し、集合ごとに記録を止めて、最後に `/v1/steer/analyze` を呼ぶ。進み具合は `A 3/8…` のように出て、`cancel` はプロンプトの切れ目で止める (そこまでの記録は保存される)。
4. 分離チャートは層ごとの `sep_score = ||mu_A - mu_B|| / (||mu_A|| + ||mu_B|| + 1e-12)`。チャートをクリック / ドラッグすると層の範囲 `[from, to)` が決まる。`use recommended` はピークの半値以上が続く連続窓 (層 0 と最終層は除く) を選ぶ。
5. `apply abliteration` が `POST /v1/steer/apply` を `terms: [{ablit_A, +1}, {ablit_B, -1}]`、`mode: "project"`、選んだ層と `alpha` (0.0–2.0、刻み 0.05、既定 1.0) で送る。`reset / clear` は `DELETE /v1/steer/apply`。
6. `eval off vs on` が一括比較。`eval prompts` に 1 行 1 プロンプトで 1〜10 本貼ると、ステアリング無し (OFF) で全部流し、いまの `[from, to)` / `alpha` を `project` で掛けてから (ON) もう一度全部流し、`prompt | OFF | ON | len delta` の表にする。進み具合は `eval 2/5 (off)…` / `eval 2/5 (on)…` で、`cancel` はプロンプトの切れ目で止める。出力長は `usage.output_tokens` (返さないサーバーでは文字数)、拒否っぽい応答には `[refuse]` が付き、変化した行は薄く着色する。元の単一規則の適用状態は完了時もキャンセル時も戻す。
7. `colour: ablit` (ヘッダの `ablit` チップも同じ) で、48 層の柱と 12 個の Hybrid block を層ごとの分離スコアで塗る。
8. `download .dns` / `download .json` は `GET /v1/steer/export?name=ablit_A` を取る (DNSTEER1 のバイナリ、または本体を base64 で載せた JSON)。`import` はそのファイルを `POST /v1/steer/import` へ送る (`.dns` は `application/octet-stream`、JSON は向きの JSON として)。
9. `save recipe (.json)` はいまの `[from, to)` / `alpha` (A +1 / B −1 / `project`) を `POST /v1/steer/recipe/export` へ送り、`<name>.recipe.json` をダウンロードする。`embed snapshots` を入れると 1 ファイルで完結する (2 つのスナップショットが base64 の DNSTEER1 として `snapshots` に載るので、移した先に `$DENNO_STEER_DIR` が要らない)。
10. `load recipe (.json)` は手元で検証してから (規則はサーバーと同じ) `POST /v1/steer/recipe/import?apply=1` で適用し、層の範囲・alpha・名前をレシピに合わせる。埋め込みスナップショットはメモリにだけ載り、ディスクは触らない。
11. `stack` は概念ベクトルの重ね掛け。`add vector` で行を足し (最大 4 行)、各行に pos / neg のスナップショット名、`mode` (project / subtract)、層範囲 `[from, to)`、alpha を持つ。左端のチェックを外した行は送らない。`apply stack` が有効な行だけを `POST /v1/steer/apply` の `rules: [{terms:[{pos,+1},{neg,-1}], layers, alpha, mode}]` として 1 回で送る (行を足したときの層範囲は、そのときのチャートの選択で埋まる)。サーバーは規則を書かれた順に 1 本ずつ掛けるので、同じ層に 2 本重なれば project が 2 回かかる。単一ベクトルの `apply abliteration` とは別のボタンで、従来の本文 (`terms` / `layers` / `alpha` / `mode`) は 1 文字も変わらない。

手順の全体、レシピのスキーマ、検証のチェックリストは [`ABLITERATION_GUIDE.md`](ABLITERATION_GUIDE.md) にある。

このために足した API (crates/denno-cli/src/serve_flash.rs):

- `POST /v1/steer/apply` — 従来の `{terms:[{snapshot,coef}], layers:[from,to], alpha, mode}` か、重ね掛けの `{rules:[{terms,layers,alpha,mode}]}`。`rules` は `terms` / `layers` / `alpha` / `mode` と同時には送れない (400)。規則は最大 8 本で、それぞれの層範囲・alpha・mode で順に掛かる。`GET /v1/steer` の `apply` は `rules` (空配列なら単一規則) も返す。
- `POST /v1/steer/analyze` — 本文は `{pos, neg}` か `{terms:[{snapshot,coef}...]}` (正の係数が A 側、負が B 側)。返りは `{pos, neg, n_layer, peak_layer, recommended_range:[from,to], layers:[{layer, diff_norm, norm_pos, norm_neg, cos_sim, sep_score}]}`。
- `GET /v1/steer/export?name=<snap>[&format=json]` — 保存済みの DNSTEER1 スナップショット、または同じバイト列を base64 にした JSON。
- `POST /v1/steer/import[?name=<new>]` — DNSTEER1 の本文か、`data_base64` / `mean` 配列を持つ JSON。形とモデル id の検証はローダと同じで、不正は 400。
- `POST /v1/steer/recipe/export` — 本文 `{name?, description?, embed_snapshots?}` で適用中の設定を、`terms` / `layers` / `alpha` / `mode` を渡せば未適用の設定をレシピ JSON (`version: 1`) にする。
- `POST /v1/steer/recipe/import[?apply=1]` — レシピ JSON (または `{recipe, apply}`)。検証し、`apply` があれば適用する。埋め込みはメモリにだけ載る。

レシピの形は `{version: 1, name, description?, model_id?, mode, alpha, layer_from, layer_to, terms: [{snapshot, coef}], snapshots?: {name: base64-DNSTEER1}}`。検証は厳密で、未知のフィールド・非有限の alpha/coef・空の terms・`layer_from > layer_to`・壊れた埋め込み・適用時の model_id の食い違いはすべてエラー。`denno serve-flash --steer-recipe <path>` (または `DENNO_STEER_RECIPE`) で起動時に読み込んで適用し、`[flash] steer recipe loaded: <name> mode=<mode> alpha=<alpha> layers=<from>..<to>` を 1 行だけ出す。フラグも env も無ければ何も変わらない。

レシピは 1 規則ぶんの形しか持たないので、重ね掛け (`rules`) の適用中に `save recipe` を押すと `POST /v1/steer/recipe/export` は 400 を返す (黙って 1 規則に畳まない)。単一規則に戻してから保存する。

`POST /v1/steer/record` は `capture_mode` も受ける。`"all"` (既定、従来どおり) は全トークンの平均、`"last_prompt_token"` は prefill の最終トークン (`nt - 1`) の残差だけを層ごとに記録し、生成中の decode トークンは載せない。それ以外の値は 400。`GET /v1/steer` は現在の `capture_mode` を返す。Abliterate パネルの `capture mode` セレクタの既定は `last_prompt_token` (指示境界での difference-in-means)。

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
| 17 | `ExpertSkip` | router が選びたかったが介入で切った expert |

送出は「詰まったら落とす」チャネルなので、購読が遅いビューワが推論を待たせることはない。
