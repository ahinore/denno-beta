# 第3巻 モデル対応 — GPT-2 から 27B ハイブリッドまで

対象期間: 2026-10-11 〜 2026-10-23。対応モデルを増やし、量子化形式を増やし、27B の出力破綻を追い詰めた記録。
速度・カーネル・計測の話は別巻。ここでは「どの重みを、どの形で読み、どう解釈するか」だけを扱う。

## この巻で分かること

- 対応モデルを増やす順序(GPT-2 → Qwen2.5 → 大型 Qwen → 量子化 → GGUF → Qwen3.5 ハイブリッド → 27B)
- 量子化形式を増やすときの検証法(参照実装とビット一致、fixture、型 id の裏取り)
- 27B の出力破綻を「構造 → 数値経路 → 参照実装比較」の順に切り分けた過程と、GDN の GQA head mapping が根因だったこと
- 重みの持ち方(常駐 f32 表・行チャンク量子化・Q4_0 生ブロック)を、数値不変を保ったまま変える方法

## 前提知識

GGUF コンテナと ggml の量子化ブロック(k-quant / IQ 系)、Gated DeltaNet(delta rule の再帰状態)、
GQA の head 対応、chat template。数値の契約は「ビット一致(R1)」と「決定論 + 出力の妥当性(緩和)」の 2 種類があり、
GDN は後者、GEMM と量子化は前者に属する。

## 読む順序

P3 → P7 → P19(基盤 → GDN 参照 → モデル統合)→ 9B(A2・A2.5)→ 27B(A3・A4)→ GPU リグレッション調査 →
27B の構造検証・品質調査・GQA 修正 → prefix cache の recurrent 対応 → Q4_0 ネイティブ →
モデル情報表示(s156)→ 量子化デコーダ(K1)→ 表示の厳密化(s158)→ ビット一致の固定(K2・K3)→
GGUF フォールバック(s164)→ 支配型(s168)→ 層数(s175)→ 重みの持ち方(P35・P37・P38・P42)。
27B の 3 エントリ(構造検証・品質調査・GQA 修正)は連続した 1 つの話なので、まとめて読む。

---

### P3: int8 weight-only 量子化 (2026-10-11, `ffabb50`)

- **意図**: メモリを半分にしたかった。当時の fp32 重みは 0.5B でも約 1 GB で、より大きいモデルを載せる余地が無かった。

- **変えたもの**
  - `ffabb50` — per-channel scale 量子化と、デ量子化込みの matvec カーネル(新規 `quant.rs`、430 行)。
  - fp32 を既定のまま維持し、`--quant int8` のときだけ有効。

- **つまずき**
  - 症状: 期待した速度向上が出ない。0.5B で 14.4 → 13.9 tok/s とほぼ横ばい。
  - 原因: 重みをデ量子化してから fp32 で積む実装なので、帯域削減の利益をデ量子化のオーバーヘッドが相殺していた。
  - 抜け方: この時点では速度を諦め、価値はメモリ半減(約 1 GB → 約 0.5 GB)に置いた。真の整数ドット積は後の SIMD カーネルで解決する。

- **分かったこと / 残った課題**
  - テスト 77 本が緑。fp32 の exactness は不変、int8 は 3 実行が完全同一(決定論)。
  - 残った課題は「デ量子化を毎回払わない形」= アクティベーションの動的量子化 + i8×i8 の整数ドット積。

- **次に効いたこと**: 「量子化はメモリの話、速度は別の話」という切り分けが定着した。後の Q4_0 ネイティブ経路でも、まず VRAM が目的で、速度は結果として付いてくる。

---

### P7: 大型 Qwen モデル対応 — 分割 safetensors と untied lm_head (2026-10-12, `b6c6c5d`)

- **意図**: 0.5B で通った経路を 1.5B / 3B / 7B に広げたかった。7B 級は単一ファイルで持てず、lm_head が embedding と共有されないモデルもある。

- **変えたもの**
  - `b6c6c5d` — 分割 safetensors のロード(`model.safetensors.index.json` の解決、全 shard の取得、テンソル名解決。新規 `WeightStore`)。
  - 非結合 lm_head(`tie_word_embeddings: false`)の経路。

- **つまずき**
  - 症状: 7B がロードできない / 3B の出力が壊れる。
  - 原因: 重みが index.json + 複数 shard に分かれており、単一ファイル前提のローダでは解決できない。3B は lm_head が独立テンソルなのに embedding を流用していた。
  - 抜け方: index.json からテンソル名 → shard の対応を引き、`WeightStore` 経由で解決する。untied のときは `lm_head` テンソルを優先する。

- **分かったこと / 残った課題**
  - テスト 93 本が緑(+2)。0.5B の exactness は不変。0.5B / 1.5B / 3B / 7B すべてで正常応答を確認。
  - サイズ別実測(20 コア CPU): 0.5B は fp32 14.4 tok/s・int8 約 51 tok/s、3B は約 2 → 約 4.7 tok/s(2.4 倍)、7B は約 1.2 → 約 2.1 tok/s(1.8 倍)。7B の重みは 28 GB で、121 GB の RAM なら余裕がある。

- **次に効いたこと**: 「untied かどうかを config から正しく解決する」という論点が、後の 9B の logits 全滅(この巻の中盤)でそのまま再発する。しかも 9B では config がラッパ形式で、キーの場所が違った。

---

### P19 基盤: GGUF コンテナリーダと k-quant / IQ 18 型のデコード (2026-10-14, `49ac073`)

- **意図**: 27B の配布形式が GGUF(混合量子)だった。safetensors 経路では届かないため、GGUF の読み込みと量子化ブロックのデコードを先に作る必要があった。

- **変えたもの**
  - `49ac073` — `gguf.rs`(mmap ベースのコンテナリーダ。ヘッダ / KV / テンソル情報)。
  - `kquant.rs` と自動生成の `kquant_tables.rs`(生成器 `ref/gen_kquant_tables.py`)。k-quant / IQ 系 18 型のデコード。
  - LCG fixture による llama.cpp とのビット一致検証(16 ブロック × 18 型すべて一致)と、実 GGUF 320 テンソルでの検証。

- **つまずき**
  - 症状: fixture を commit できない。`kquant_tables.rs` を手で直したくなる。
  - 原因: `.gitignore` の `data/` が fixture のディレクトリまで除外していた。生成物と手書きコードの区別も曖昧だった。
  - 抜け方: `.gitignore` に negation を足して fixture だけ通す。`kquant_tables.rs` は生成物に限定し、手編集を禁止する方針を明記した。

- **分かったこと / 残った課題**
  - 18 型すべてで llama.cpp とビット一致。実 GGUF の 320 テンソルも検証できた。
  - 残った課題: 混合量子に含まれる型を作り足すたびに、この検証をもう一度回す必要がある(後の K1/K2/K3 がまさにそれ)。

- **次に効いたこと**: 「参照実装をリンクして期待値を作る」検証法が、量子化まわりの標準手段になった。fixture の md5 を固定しておくことで、生成器を触ったときに既存部分が変わっていないことを毎回確認できる。

---

### P19-B: Gated DeltaNet の CPU リファレンス (2026-10-14, `0bbfa79`)

- **意図**: Qwen3.5 は Gated DeltaNet と full attention のハイブリッドで、GDN の再帰状態を扱うコードが無かった。まず参照実装を 1 本作り、それを正として統合する。

- **変えたもの**
  - `0bbfa79` — 新規 `gdn.rs`。llama.cpp の `build_delta_net_autoregressive`(逐次 AR 形式)を 1:1 で移植。
  - `cb9d30b` — `lib.rs` へのモジュール登録。

- **つまずき**
  - 症状: 数式は追えるが、どこまで llama.cpp と一致させるべきか決まらない。
  - 原因: GDN の更新は chunked 形式と逐次 AR 形式で数学的に等価だが、丸め順は一致しない。ビット一致を要求すると実装が縛られる。
  - 抜け方: GDN はビット一致の契約対象外と明示した(denno の契約は「同一入力 → 同一出力」+ 決定論)。逐次 AR 形式で書き、chunked との等価性は数学的な主張として扱う。

- **分かったこと / 残った課題**
  - 更新は g = ssm_a(負値)× softplus(alpha + dt_bias)、S ← S·exp(g)、delta ルール(sk = Σk·S、d = β(v − sk)、S += k⊗d、o = Σq·S)、q/k は L2 正規化、出力は gated RMS norm(rmsnorm(o)·w·silu(z))、causal depthwise conv + SiLU。softplus は ggml 規約に合わせた。
  - テストは gdn lib 4 本(独立オラクルとの一致、prefill 一括とトークン逐次のビット一致、決定論)。実 GGUF の layer-0 でも確認し、減衰 exp(g) が 0.0086〜0.9995 の多時間スケールを持つことを見た。

- **次に効いたこと**: 「この層はビット一致を要求しない」という例外を先に宣言したことで、後の GPU カーネルや chunked scan の設計が自由になった。同時に、GDN の数値差は「緩和契約内」として切り分ける判断基準が生まれた。

---

### P19-B2: qwen35 モデル統合と E2E (2026-10-14, `feab1c8`)

- **意図**: GGUF と GDN 参照実装が揃ったので、Qwen3.5 を 1 つのモデルとして組み立て、実モデルで端から端まで通す。

- **変えたもの**
  - `feab1c8` — 新規 `model/qwen35.rs`(`model_type: qwen3_5` で dispatch)。
  - 24 層 = 18 GDN + 6 full attention(3:1)、full attention は Q+gate 融合、partial MRoPE(interleaved [11,11,10])、QK-Norm、sigmoid 出力ゲート、tied embeddings(vocab 248,320)。
  - GGUF(混合量子含む)→ dequant → 既存の Fp32/Int8 パイプラインへ再量子化する形にし、新しい GEMM を作らない。極小の f32 テンソル(conv1d / ssm_a / ssm_dt / ssm_norm)は参照ビュー `GdnCore` + `forward_token_core`。GDN 状態は KV キャッシュの GDN スロット(conv バッファ + SSM 行列)。

- **つまずき**
  - 症状 1: 2 トークン目で範囲外 panic。
  - 症状 2: 同一 seed でも出力が変わる(非決定論)。
  - 症状 3: int8 にすると品質が落ちる。
  - 原因: (1) `cache.len()` は forward 終了まで進まないため、トークンごとの確保では足りない。(2) KV の live 再利用と prefix entry 復元は GDN の conv/SSM 状態を無効化する(再帰状態は位置スライスできない)。(3) per-tensor 量子化誤差は全テンソル 0.8〜1.5 % で正常範囲だが、最悪が `ssm_beta` / `ssm_alpha` — sigmoid / softplus を通る制御信号だった。
  - 抜け方: (1) forward 冒頭で全トークン分を一括確保。(2) `Model::is_recurrent()` を新設し、recurrent モデルは純拡張(lcp == len)のみ再利用、それ以外はフル prefill。(3) 制御信号は Int8 モードでも Fp32 維持(24 層で 1 MB 未満、コストは無視できる)。

- **分かったこと / 残った課題**
  - テストは gdn 4 本 + `qwen35_model`(ロード / forward / 決定論 2 回ビット一致 / キャッシュ連続性 forward(3)+forward(1) == forward(4) ビット一致)で lib 114 passed。
  - E2E: 日本語 QA・英語 haiku・307 トークン要約が coherent。greedy(temp 0)で int8 も fp32 も正答、seed 42 × 3 連続リクエスト完全一致。prefill 約 90 ms/token、decode 約 20 ms/token(CPU トークン逐次)で、次の律速は GPU 化だった。

- **次に効いたこと**: 「再帰状態を持つモデルは prefix cache の前提が崩れる」という発見が、後の GDN 状態スナップショット(A4)と、template 境界の split prefill(s46)の出発点になった。

---

### A2: Qwen3.5-9B の実 E2E — GDN k-head 規約と untied lm_head (2026-10-15, `9183e7e`)

- **意図**: 0.8B で通った経路を 9B(nk=16 / nv=32、untied、32 層 = 24 GDN + 8 full)へ広げる。0.8B と構造が違う点で出力が壊れた。

- **変えたもの**
  - `9183e7e` — `gdn.rs` と `gdn.cu` の v-head → k-head マッピング、`qwen35.rs` の `tie_word_embeddings` 解決。

- **つまずき**
  - 症状: 9B の出力が garble(文字化け的な無意味列)。隠れ状態は層 0〜31 まで numpy 参照と一致しているのに、出力だけ壊れる。
  - 原因 1: v-head `h` が参照する k-head を `h % nk` で計算していた。0.8B は nk == nv なので無害だが、9B では誤マッピングになる。
  - 原因 2: 9B の config.json は VLM ラッパ形式で、`tie_word_embeddings: false` が**トップレベル**にあり `text_config` の中に無い。ローダは `text_config` だけを見て `unwrap_or(true)` したため tied 扱いになり、embedding を lm_head の代わりに使って logits が全滅していた。
  - 抜け方: 原因 1 は `h * nk / nv`(HF / llama.cpp の blocked タイリング相当)へ変更。層 0 の numpy 検証で blocked 0.27 % / interleaved 73 % の誤差差を見て確定した。原因 2 は `text_config` から解決できない場合トップレベルへフォールバックする。

- **分かったこと / 残った課題**
  - 検証: numpy 参照実装との比較で fp32 単トークン全 32 層一致(最大相対誤差 1e-4)、3 トークン逐次 logits 一致、逐次とバッチ prefill のビット一致、同一 GGUF の llama.cpp は正常出力(エンジン固有バグの切り分け)。
  - E2E(int8): 正常な思考出力。md5 `90fe7e58…` が連続 2 回 + サーバー再起動後も一致。
  - **注意**: ここで入れた blocked 規約は、後の GQA head mapping 修正(この巻の終盤)で strided が正と判明し、反転している。9B は blocked でも「耐えられていただけ」だった。

- **次に効いたこと**: 「隠れ状態は合っているのに出力だけ壊れる」= 最終段(lm_head / 出力ゲート)を疑う、という切り分けが定着した。27B の調査でも最初にこの形を疑っている(そして否定された)。

---

### A2.5: repetition penalty — 9B の greedy 反復退化への対処 (2026-10-15, `4fada3c`)

- **意図**: 9B は temperature 0 の greedy で同一フレーズを繰り返し、回答に到達しないことがあった。まずチャットテンプレートを疑い、次にサンプリング側で対処する。

- **変えたもの**
  - `4fada3c` — `sampler::apply_repetition_penalty`(HF 互換・決定論)。`GenParams.repetition_penalty`、`/v1/messages` の任意フィールド、CLI `--repetition-penalty` として露出。

- **つまずき**
  - 症状: 思考ループから抜けない。テンプレートのせいか数値のせいか分からない。
  - 原因: テンプレートは公式準拠で不一致が無かった(chat_template を全量展開し、minijinja + pycompat で実 CPU テスト)。つまり原因は生成側。
  - 抜け方: 文脈(prompt + 生成済み)中のユニークな既出トークンに対し、`logit > 0` なら `logit / penalty`、`logit <= 0` なら `logit * penalty` を一度だけ適用する(HF の `RepetitionPenaltyLogitsProcessor` と同一規則)。temperature / top-k / top-p より前(生 logits)に置く。RNG を使わないので greedy は完全決定論。`penalty == 1.0` はビット完全 no-op。

- **分かったこと / 残った課題**
  - 9B int8 E2E(penalty 1.1)でループ脱出。md5 `4551f00f…` が連続 2 回 + 再起動後も一致。penalty 1.15 / 1.3 は過剰抑制で、1.1 が適正(llama.cpp の既定値と一致)。
  - 恒久回帰テスト `qwen35_9b_int8_penalty_determinism` を追加(決定論 + 最頻 4-gram カナリアで反復ループ検出)。
  - 残った課題: 反復退化の根本の数値原因は未特定。thinking 出力の扱いも未整理。

- **次に効いたこと**: 「適用位置を固定し、既定 no-op をビットで証明する」型がここで確立し、後の反復抑止 3 種(別巻)がその上に積まれた。

---

### A3: Qwen3.8-27B のスケール対応 — 合成シェイプテストと KV 予算 (2026-10-15, `646c22a`)

- **意図**: 27B(64 層 = 48 GDN + 16 full、untied)を既存経路で動かせるか、実 GGUF を置く前に構造と予算で確かめたかった。

- **変えたもの**
  - `646c22a` — `qwen35.rs` に合成 27B シェイプテスト(F32 専用の最小 GGUF v3 ライタで、構造比を保った小型モデルをテスト内で生成)。
  - GDN ヘッド比ガード、GDN 層の KV バッファ確保の廃止。

- **つまずき**
  - 症状: 27B の config に 9B には無いキー `output_gate_type: "swish"` があり、解釈が決まらない。KV 予算の見積りも 4 倍過大だった。
  - 原因: 参照実装 3 系統(transformers main / vLLM / llama.cpp master)をクロスチェックした結果、**どこもこのキーを消費していない**。KV 側は、GDN 層(KV を書かない層)まで KV バッファを確保・計上していた。
  - 抜け方: full attention の出力ゲートは sigmoid、GDN の z ゲートは silu(= swish)が正で、既存動作と一致 → 数値経路は変更なし。キーが存在するときは info ログ、未知の値は warn にする。KV は GDN スロット層を確保・`bytes_per_token`・snapshot 対象から除外。

- **分かったこと / 残った課題**
  - KV は 27B で 512 KB/トークン → 128 KB/トークン、9B も 256 KB → 64 KB に削減(実量の 1/4)。
  - 予算見積り(int8・KV 予算 12 GB 既定): 線形層 int8 約 24.6 GB + embed fp32 約 5.1 GB + lm_head int8 約 1.3 GB ≈ 約 31 GB(ホスト RAM)。KV 128 KB/トークンで 32K prefill = 4 GB、既定予算で 24 スロット、単一リクエスト上限 約 98K トークン。ctx 262,144 の全幅は KV 約 34 GB で既定予算を超えるため、予算の引き上げが必要。
  - テストは lib 128 passed(新規 7)。UD-IQ2_XXS GGUF は約 7.3 GB(単一ファイル)で、使用量子形式はすべて dequant 対応済み — ただし IQ1_M の型 id を当時は誤って認識しており、後の K1 で訂正される。

- **次に効いたこと**: 「実物が無くても構造比を保った合成モデルで契約を固定する」手法が、27B の全作業(構造検証・split prefill・GQA 修正)の土台になった。

---

### A4: prefix cache への GDN 状態スナップショット (2026-10-15, `bf2c90b`)

- **意図**: A3 まで recurrent モデルは prefix cache の entry 復元を使わず、常にフル prefill だった。KV だけコピーしても GDN 状態が戻らず、静かな非決定論になるため。マルチターンで prefill を省けるようにする。

- **変えたもの**
  - `bf2c90b` — `kv_cache.rs`(`gdn_snapshot` / `gdn_restore`、bit コピー)、`prefix_cache.rs`(状態バイトを LRU 重みへトークン換算)、`engine.rs`(退避タイミングの変更)。

- **つまずき**
  - 症状: 状態をどこに保存すればよいか決まらない。
  - 原因: GDN 状態(conv + SSM)はシーケンス全体の関数で、「トークン位置ごと」には保存できない。
  - 抜け方: prompt 境界(prefill 直後・decode 前)の状態 1 回分だけを、KV スナップショットとペアで保存する。退避は完了時(`release_session`)ではなく prefill 直後に変え、KV 行は prompt ちょうどの長さで保存する。復元条件は「新しい prompt がエントリ prompt 全長を接頭辞として含む」場合のみ。部分一致では KV のみの復元も行わず、フル prefill に落とす — KV だけ復元して再帰状態を不整合にする経路を構造的に作らない。

- **分かったこと / 残った課題**
  - テスト +13(lib 147)。**対照実験**として、KV のみ復元(GDN 状態を復元しない)では出力が変わることを確認した — テストが状態の復元を実際に検証している(teeth-check)。
  - 復元後の継続 forward はフル prefill とビット一致(キャッシュ連続性と状態の bit コピーから従う)。snapshot / restore は純粋なメモリコピーで、数値経路の変更は無い。
  - 残った課題: 実機でのマルチターン E2E 計測。27B では状態 約 150 MB/エントリなので prefix cache 予算への影響を見る必要がある。

- **次に効いたこと**: 「状態を持つモデルでは、キャッシュの前提(位置で切れる)が崩れる」という理解が、s46 の split prefill(別の原因でヒットしない問題)へ続く。

---

### GPU E2E リグレッション調査 — Qwen3.5-9B の GPU prefill 破綻 (2026-10-15, `d409d4f` / `ed09722` / `9dc6fc9`)

- **意図**: 9B の GPU サーバーだけ出力が破綻し、CPU サーバーは正常という報告。コードのリグレッションか、環境か、数値の不安定性かを決着させる。コード変更は無く、調査記録と判定基準の文書化だけが残った。

- **変えたもの**
  - `d409d4f` — 調査記録を HISTORY に追加。
  - `ed09722` — drift 注入実験の記録。
  - `9dc6fc9` — GPU 側アンカーの追記と、回帰判定基準の明文化。

- **つまずき**
  - 症状: GPU で最初のトークンから破綻。`max_tokens=1` でも CPU と不一致。5 連続で完全同一(決定論は維持)。
  - 原因: 切り分けの結果、prefill 経路が壊れていることが分かった。ただし A2 以降の変更を GPU 経路ファイルごと関数単位で全 diff すると、**どの変更も Qwen3.5 の GPU prefill 経路に到達していない**。CPU ビルドで同一 payload を流すと md5 `3a0ddf1c…` で CPU サーバーと完全一致。残った説明は「GPU 上の数値差が greedy で増幅する」だった。
  - 抜け方: 層別チェックサム(`[LCHK]`、env ゲート)で初発散層を特定 → L00(GDN 第 1 層)の attn で相対 ssq 差 約 4e-7。embedding と full attention 層はビット一致。次に CPU ビルドへ決定論的な摂動を注入する実験を行い、eps=1e-7(観測 drift より小さい)でも L00 注入で final argmax が分裂することを確認した。分裂先は最終位置の runner-up `8160 ("Here")` が最頻(9 config 中 6)。top-2 は `90700 ("Thinking")` と `8160` で logit 差 2.03 の near-tie。

- **分かったこと / 残った課題**
  - 隠れ状態の相対 ssq 差は 32 層で 約 1e-5 → O(0.1) へ 3〜4 桁カオス的に増幅する。発源が早い層ほど増幅が大きい。
  - 判定: GPU の観測は「temperature 0 greedy の不安定性(仕様上の脆弱性)」で説明でき、GDN の GPU カーネルの drift が異常に大きい証拠は無い。H1(別ソースからのビルド)と H3(モデル破損)は否定。
  - 含意: GDN モデルの GPU E2E 回帰判定は「CPU とのビット一致」ではなく、**(a) GPU 上の決定論 (b) 出力内容の妥当性**を使う。新正規 payload のアンカーは CPU `3a0ddf1c…` / `95199a33…`、GPU `1a4ea71a…` / `b4e58cb3…`(各連続 2 回一致)。

- **次に効いたこと**: 「ビット一致できない層を含むモデルの合否基準」が明文化されたことで、以後の GQA 修正のように数値を変える修正を、アンカー更新とセットで進められるようになった。

---

### 27B 構造検証とロード時ガード — out-gate 仮説の否定 + MTP 除外 (2026-10-16, `203deed`)

- **意図**: 27B の出力破綻の原因が「構造の読み違い」ではないかという仮説を潰す。同時に、想定外の構造を黙って読まないためのロード時ガードを入れる。

- **変えたもの**
  - `203deed` — `qwen35.rs` の `expect_ne`(dequant 前に GGUF ヘッダ形状を config 由来の期待値と照合)、MTP(nextn)層の除外。
  - `tests/qwen35_27b_header.rs`(実 27B GGUF のヘッダのみ検証、`#[ignore]` + 環境変数未設定で graceful skip)。

- **つまずき**
  - 症状: 27B だけ出力が壊れる。構造の解釈が違う可能性を否定できない。
  - 原因: 仮説の検証には 3 点照合が必要だった(transformers の `modeling_qwen3_5.py`、llama.cpp の `qwen35.cpp`、実 GGUF ヘッダ 0.8B / 9B / 27B の Q4_0)。結果、27B の full attention 構造は現行エンジンと完全一致。9B が `heads×dh == e`(4096)で正方形なのは偶然で、0.8B(1024→2048)と 27B(5120→6144)の非正方形が既定経路の正解だった。out-gate の位置も 3 サイズ共通(attention 後・o_proj 前の sigmoid)。
  - 抜け方: 仮説を棄却し、以後の調査対象を数値経路の drift / 品質要因へ絞った。ガードは「層テンソルを dequant する前に、ヘッダ上の形状を config 由来の期待値と照合し、末尾の 1 は無視、期待より高い次元は拒否」とした。

- **分かったこと / 残った課題**
  - MTP の扱い: GGUF の `block_count` が `num_hidden_layers` を超える分は、`nextn_predict_layers` と一致し、かつ超過ブロックが `nextn.eh_proj` を持ち `ssm_a` を持たない場合のみ読み飛ばす(27B: `block_count=65` = trunk 64 + 1)。不整合はロード拒否。nextn を通常層として読むと意味論が壊れるため。
  - テストは合成 4 件 + 実ヘッダ検証(実ファイルで PASS)。workspace 188 passed。
  - 実機: UD-IQ2_XXS は IQ1_M のデコード未対応で起動不可 → 均一 Q4_0 版(16.06 GB)でロード成功、VRAM 30.2 GB / 32.6 GB、warm decode 約 0.9 s/step、3 回 md5 一致 `1aeac03a…`。ただし**出力品質は破綻**(greedy でループ)し、未解明のまま次へ。

- **次に効いたこと**: `expect_ne` と MTP 除外は、後で「GGUF だけのディレクトリで層数を出す」処理(s164 / s175)の前提になった。block_count が 65 である理由を、この時点で理解していたことが効いている。

---

### 27B 出力品質調査 — CPU 切り分けと llama.cpp リファレンス比較 (2026-10-16, `7b07870`)

- **意図**: 構造は正しいと確定したので、次は数値経路か GGUF 自体の品質かを切り分ける。参照実装と同じ GGUF を同じ条件で走らせて比べる。

- **変えたもの**
  - `7b07870` — 診断テスト `tests/qwen35_27b_logit_probe.rs` を恒久化(`#[ignore]` + 環境変数未設定で graceful skip、所見をヘッダに記載)。

- **つまずき**
  - 症状: greedy でプロンプト断片のエコー + `assistant` / テンプレート断片のループ。
  - 原因: まず CPU で再現し、GPU の drift 増幅説を否定。repetition penalty 1.1 でも解消せず(ループが早期 end_turn に変わるだけ = logits 分布自体が誤り)。GGUF ヘッダは config と完全一致、使用型もすべて対応済み。chat template も 9B とバイト完全一致だった。
  - 抜け方: mainline llama.cpp(`59d5fad`、`qwen35.cpp` 対応)を CPU ビルドし、同一 GGUF + 同一質問で比較。llama.cpp は正答(`Paris is the capital of France.`)。さらに同一 raw プロンプト(17 トークン、トークン化も一致)で生成 step 0 の分布を比べると、llama.cpp の top-1 は `The`(logprob ≈ 0、runner-up と 10 nat 超の差)、本エンジンの top-1 は `What`(logit 15.79)で `The` は top-2(14.15、差 1.63)。本エンジンの top-1 は llama.cpp の top-5 に入らない。

- **分かったこと / 残った課題**
  - プローブで分かった実態: `"What is the capital of France?"` の greedy 24 step は、step 0〜6 で**ユーザーの質問文をトークン単位で完全エコー**し、step 7 で `<|im_end|>`(logit 19.25、runner-up 16.5)を出して終了、以後テンプレート断片のループ。prefill 自体はビット一致(決定論は維持)。
  - 結論: GGUF の品質候補は否定。**27B の破綻は本エンジンの 27B 経路固有**で、生成 step 0(= prefill 最終位置 logits)で既に乖離している。残る候補は GQA q24/kv4(group 6)、GDN v-head 48(repeat 3)、実重みのテンソル対応。
  - 9B のアンカーは再現(`3a0ddf1c…`)し、9B 経路は健全であることを裏取りした。

- **次に効いたこと**: 「参照実装と同じ入力で step 0 の分布を比べる」という手法が、次の GQA 修正で初発散層の特定に直結した。プローブを捨てずにテストとして残したことも効いている。

---

### Qwen3.5 GDN GQA head mapping 修正 — blocked からタイル repeat へ (2026-10-16, `e266090`)

- **意図**: 残った候補のうち、GDN の v-head が参照する k/q head の対応を確定させる。27B の出力破綻の根因を潰す。

- **変えたもの**
  - `e266090` — `gdn.rs` の `forward_token_core` と `scan_chunked_core`、`kernels/gdn.cu` の 3 カーネル(`dno_gdn_scan` / `dno_gdn_chunk_pre` / `dno_gdn_chunk_scan`)。
  - `qwen35.rs` の規約コメントとロード時ガードのメッセージ。

- **つまずき**
  - 症状: 27B の prefill 最終位置 logits が参照実装と違う。どの段で発散するか分からない。
  - 原因: llama.cpp 側に層別チェックサムのツールを用意して比較したところ、初発散は GDN 層の attn サブステージ(full attention・embedding・FFN・final は一致)。さらに GDN サブステージを per-token でダンプすると、scan の出力 `o` で乖離し、それ以前(qkv 〜 k/v 投影)は一致した。つまりスキャン内部の head 対応だけが違う。
  - 抜け方: head mapping を変えて再計算し、**strided(`kh = h % nk`、llama.cpp の `ggml_repeat` 相当)だけが一致**することを確認(9B: cos ≥ 0.99984、中央値 0.99998、27B: cos 0.99999 全トークン)。A2 で入れた blocked(`h * nk / nv`、HF の `repeat_interleave` 相当)は乖離した。CPU 2 箇所 + GPU 3 箇所を strided に統一した。

- **分かったこと / 残った課題**
  - 新テスト `gdn_gqa_head_mapping_is_tiled`(合成 GDN・dk=dv=2・nk=2 / nv=4)で strided 規約を固定。blocked と区別できない入力になっていないことを assertion で担保している。workspace 189 passed。
  - E2E: 27B(均一 Q4_0・int8)が正答(`The capital of France is **Paris**.`、md5 `a021580f…` 連続 2 回一致)、日本語も正答(`1483c4fe…`)。9B は新ベースライン `feea9444…` / `da0b1f57…`。
  - **PTX は未再生成**(この環境に nvcc が無い)ため、GPU 経路は旧 blocked の PTX のまま動く。再生成と GPU E2E 再検証は人間対応。
  - 残った検証: HF は `repeat_interleave`(blocked)で、GGUF + llama.cpp は cyclic。両者が正であるためには GGUF 変換が k/q 行を再配置していることになるが、変換器の直接確認はできていない。本エンジンは GGUF しか読まないため、当該 GGUF に対する規約の正しさは実測で確定している。

- **次に効いたこと**: 「head mapping のような 1 行の解釈違いが、モデル全体の出力を壊す」という事例が、以後の「形状だけでなく意味論も参照実装と突き合わせる」規律になった。`[LCHK]` と `[GDUMP]` は env ゲート付きで恒久保持され、後の調査で再利用されている。

---

### s46: recurrent モデルで prefix cache が 1 度もヒットしない問題 (2026-10-17, `262bd85`)

- **意図**: A4 で GDN 状態のペア復元を入れたのに、実モデル(Qwen3.5)では prefix cache が一度もヒットしないという観察があった。原因を確定して直す。

- **変えたもの**
  - `262bd85` — `tokenizer.rs`(`apply_chat_template_base`)、`engine.rs`(`ChatSession.gen_prompt_start`、split prefill)、`prefix_cache.rs`(`best_full_match` の最長選択)。

- **つまずき**
  - 症状: マルチターンでも毎回フル prefill。`best_full_match` が成立しない。
  - 原因: chat template は毎回 prompt を generation prompt で終える。次ターンの描画ではその部分が assistant メッセージ本文に置き換わるため、turn_{n+1} の prompt は turn_n の prompt をトークン接頭辞として含まない(実 9B で確認: turn2 の prompt は turn1 の末尾 2 トークン `<think>\n` で分岐する)。prompt 全体をキーにした A4 のエントリは永遠に一致しない。
  - 抜け方: `add_generation_prompt: false` で描画した「base」(generation prompt 直前まで = 次ターンにもそのまま現れる履歴部分)を返す関数を新設し、base が実際に prompt のトークン接頭辞になった場合だけ prefill を `[base][generation prompt]` の 2 forward に分割する。base 境界で KV + GDN スナップショットを退避し、prompt 全体のエントリは作らない(ヒットしないエントリを溜めない)。複数の完全一致候補があるときは最長を選ぶ。

- **分かったこと / 残った課題**
  - 分割契約: `forward(a) + forward(b)` は `forward(a++b)` とビット一致(GEMM は行独立の i32 累積、GDN / attention はトークン逐次なので、分割で蓄積順が変わらない)。
  - テスト +5(合成モデル)+ 実 9B の `#[ignore]` テスト(3 ターンが Entry ヒット、各ターンがコールド full prefill と完全一致。参照は逆順実行で取得 — 順方向だと参照側もキャッシュにヒットするため。実行 85.8 秒で pass)。workspace 242 passed。
  - つまずきの追記: 初版のテストは異なる位置のスナップショットを全体比較しており構造的に成立しなかった。正しい契約(base 境界の prefix 比較 + GDN は分割完了後の同一位置比較)に修正した。修正前の失敗出力で重複領域が厳密に prefix 一致していたことを確認しており、数値経路ではなくテストの不備だった。
  - 残った課題: 実機 GPU でのマルチターン効果の計測。

- **次に効いたこと**: 「キャッシュのキーは prompt 全体ではなく、次ターンにも現れる境界で取る」という設計が、thinking=off のときに base 検出が壊れないかという検証(s47 のテスト)にそのまま引き継がれた。

---

### Q4_0 ネイティブ経路 — 27B が 18.4 GB / 178〜186 ms/token で動く (2026-10-20, `c51d6a3`)

- **意図**: 27B は int8 経路だと VRAM 31,977 MiB・decode 669 ms/token で実用にならなかった。GGUF の 4 bit ブロックを fp32 に展開して int8 に詰め直す二度手間をやめ、Q4_0 ブロックのまま重みを持って GEMV / GEMM を専用カーネルで回す。

- **変えたもの**
  - `c51d6a3` — `Linear::Q4` と `Q4Linear`。ブロックは 32 要素 = f16 scale 2 B + ニブル 16 B = 18 B。要素 c(0..15)が下位ニブル、c+16 が上位、`y = d*(q-8)`、`d` は負になり得る(`ggml` の `quantize_row_q4_0_ref` と同じ規則)。
  - CPU 参照 `matvec_rows_bt_q4_into`、GPU は `dno_gemv_q4_0` / `dno_gemm_q4_0`。レイアウトは SOA(`nib` は uint4 16 B アライン、`sc` は f32)。f32 逐次累積なので決定論。
  - Q4 専用の重みレジストリ、`concat_rows` の Q4 分岐(gate+up 融合を維持)。

- **つまずき**
  - 症状: CPU フォールバックの経路で速度の前提が崩れる。
  - 原因: Q4 の CPU フォールバックは素のループで、int8 用の `gemm_min_m` の前提(タイル GEMM に載る)が当てはまらない。
  - 抜け方: m > 16 は常にタイル GEMM を通し、Q4 の CPU フォールバックには `gemm_min_m` を適用しない。

- **分かったこと / 残った課題**
  - 数値契約: CPU 参照との差は GEMV maxabs 9.5e-6(相対 1e-4 以内)、GEMM は 0(ビット一致)。決定論は 3 連続一致。
  - 実測(27B、seed 42): int8 は VRAM 31,977 MiB / 669 ms/token、Q4_0 ネイティブは **18,369 MiB / 178〜186 ms/token / prefill 34 tok/s**。
  - テストは lib 246 本 + `cuda_q4_linear_matches_cpu`(m=1/7/16/300、`--ignored`)、アンカー 3/3 MATCH。
  - 残った課題: TQ1_0 と IQ1_M のデコードが未対応で UD-IQ2_XXS はロードできない。GGUF ブロックを展開せずに渡す経路も未実装。なお、この時点で記録した型番号(TQ1_0 = 29 / IQ1_M = 24)は誤りで、後の K1 で訂正される。

- **次に効いたこと**: 「量子化形式をそのまま持つ」方針が、後の P35(生ブロック passthrough の判定)と P42(token_embd を生ブロックで持つ)の前提になった。二度手間をやめるという発想自体は正しく、問題は「再量子化の値が元と一致しない」ことにあった。

---

### s156: `denno info` が `text_config` から次元を読む (2026-10-21, `ca5e44a`)

- **意図**: Qwen3.5 の config.json はマルチモーダルラッパで、言語モデルの次元が入れ子の `text_config` にしか無い。トップレベルだけ見ていたため、層数・hidden・vocab・context がすべて 0 と表示されていた。

- **変えたもの**
  - `ca5e44a` — `crates/denno-cli/src/main.rs` の探索順(`text_config` / `llm_config` / `language_config` / トップレベル)。
  - ディレクトリ引数の対応(repo 引数が実在のディレクトリなら hub cache を経由せず `config.json` を直接読む)。

- **つまずき**
  - 症状: 27B / 9B で `layers: 0` などが出る。`family` もラッパの `qwen3_5` になる。
  - 原因: 参照先がトップレベル固定だった。ラッパ形式の config では言語モデルの値が 1 段下にある。
  - 抜け方: 探索順を決めて同じ順で `model_type` も引き、`qwen3_5_text` を出すようにした。平の別名(`n_layer` / `n_embd` / `n_ctx`)は従来どおり。

- **分かったこと / 残った課題**
  - 検証: `local/qwen35-27b` = 64 / 5120 / 248320 / 262144、`local/qwen35-9b` = 32 / 4096 / 248320 / 262144(いずれも family `qwen3_5_text`)。
  - denno-cli 5 passed、denno-engine lib 262 passed / 1 ignored で回帰なし。
  - 残った課題: 0 と「未知」の区別がまだ無い(次の s158 で `?` にする)。

- **次に効いたこと**: 「ラッパ形式の config では値が入れ子にある」という知識が、後の 9B の `tie_word_embeddings` 問題(A2)と同じ根を持つことの整理に繋がった。探索順を関数 1 つに閉じたことで、以後の表示追加(s158)がその上に積めた。

---

### K1: kquant に TQ1_0 / IQ1_M のデコーダを追加 (2026-10-21, `2344f67`)

- **意図**: IQ2 系と ternary の GGUF を、少なくとも CPU 経路ではデコードできるようにする。UD-IQ2_XXS をロードできるようにするのが目標。

- **変えたもの**
  - `2344f67` — `kquant.rs` に `block_tq1_0`(54 B / 256 要素)と `block_iq1_m`(56 B / 256 要素)のデコーダ。
  - `tests/dequant_ref.rs` に自己完結テスト 5 件。

- **つまずき**
  - 症状: 計画書の型番号(TQ1_0 = 29 / IQ1_M = 24)で実装すると、実在の IQ1_M を TQ1_0 として読んでしまう(静かに壊れる)。
  - 原因: 計画書の番号が upstream と違っていた。
  - 抜け方: llama.cpp の `ggml.h` と `gguf-py/gguf/constants.py` の双方で裏取りし、`GGML_TYPE_I8 = 24` / `GGML_TYPE_IQ1_M = 29` / `GGML_TYPE_TQ1_0 = 34` を採用。3 つともテストで固定した(`from_id(24)` は未対応のまま `None`)。

- **分かったこと / 残った課題**
  - TQ1_0 は 1 バイトに三値 5 個を base-3 で詰める(`((u8)(b * 3^n) * 3) >> 8` と同じ結果になる桁抽出)。IQ1_M は `scales[8]` を `u16[4]` として読み、f16 スケールを 4 つの上位ニブルに分割して持つ(独立した `d` は無い)。
  - テストは手計算の参照ブロック(54 / 56 生バイトと 256 個の期待値)とのビット一致、型番号と block size / type size、退化ブロック(全ゼロ・スケール 0・f16 の NaN / ±Inf・最小非正規)で panic しないこと、未対応 ID が `None` のままであること。
  - リポジトリ外のスクラッチで llama.cpp の `dequantize_row_tq1_0` / `dequantize_row_iq1_m` と 16 ブロック(4096 要素)のビット一致を確認。既存ハーネスの fixture md5 `36acb59a…` も再現したので、ツールチェーンも一致している。
  - 残り: 実 GGUF のロードは未実施。`ref/dequant_ref.c` と fixture の更新は範囲外(次の K2)。

- **次に効いたこと**: 「型 id は計画書ではなく upstream で裏取りする」という規律が、以後の量子化関連タスクの前提になった。番号を間違えると静かに壊れるため、テストで 3 つとも固定したのが効いている。

---

### s158: `denno info` の未知を `?` にし、rope / attention / dtype / 量子化を表示 (2026-10-22, `c95b44c`)

- **意図**: 未知の次元を `0` と表示していたため、未実装値と本物の 0 を区別できなかった。config.json 由来の構造情報を出し、GGUF があればテンソル型の最頻値も 1 行で見せる。

- **変えたもの**
  - `c95b44c` — `denno-cli` の表示(`layers` / `hidden` / `vocab` / `context` を `Option<u64>` 化)、rope・attention・dtype・量子化の行。
  - GGUF 直下に `*.gguf` があるときだけ `GgufFile` を開き、テンソル型の最頻値を人間可読名 + 数値 id で 1 行。

- **つまずき**
  - 症状: `0` が「未知」なのか「本当に 0」なのか読めない。Qwen3.5 は `rope_theta` の置き場所も普通と違う。
  - 原因: 表示の型が数値固定で、キーの別名(同じ事実の別の置き場所)を吸収する層が無かった。
  - 抜け方: 値が無いものは `?`、`quantization_config` が無いときだけ `none`。別名として `rope_scaling` の `type` が無ければ `rope_type`、`rope_theta` は `rope_parameters.rope_theta` も読む(値の捏造ではなく、無ければ `?` のまま)。GGUF の最頻値は同数のとき ggml type id の小さい方を選ぶ(実行ごとに変わらない)。

- **分かったこと / 残った課題**
  - 実測(CPU のみ): `denno info data/qwen35-9b` = 32 / 4096 / 248320 / 262144、`rope_theta: 10000000`(`rope_parameters` 由来)、`attn_heads: 16` / `kv_heads: 4` / `head_dim: 256`、`quant: Q8_0 (8)`。`data/qwen35-test` = 24 / 1024 / 248320 / 262144、`attn_heads: 8` / `kv_heads: 2` / `head_dim: 256`。両方とも config に `torch_dtype` と `rope_scaling` が無いので `?`(`?` は「無い」であって 0 ではない)。
  - denno-cli 13 passed(旧 5)。出力全体のスナップショットは取らず、解決と文字列化を関数に切って検証した。
  - 測っていないこと: engine / server のテスト数、GPU アンカー、実機の挙動(数値経路を触っていないため構造的に不変だが、測ってはいない)。

- **次に効いたこと**: 「推測した既定値を書かない」という方針が、GGUF フォールバック(s164)や支配型の判定(s168)でも維持された。表示の厳密化は、後に 27B の層数不一致(s175)を見つける土台になった。

---

### K2: TQ1_0 / IQ1_M を参照実装とビット一致で固定し、bitexact テストを既定ゲートへ (2026-10-22, `f6f03fd`)

- **意図**: K1 で入れた 2 型は fixture に載っていないため、一度も参照実装と突き合わされていなかった。fixture を 20 型に作り直し、`#[ignore]` を外して既定ゲートに入れる。

- **変えたもの**
  - `f6f03fd` — 生成器 `ref/dequant_ref.c` に TQ1_0(type 34)と IQ1_M(type 29)のセクションを既存 18 型の後ろに追加。
  - `tests/data/dequant_ref.bin` を 20 セクション / 267,381 バイト / md5 `fe45c5bf…` へ更新。`ref/README.md`(再生成手順と帰属)。

- **つまずき**
  - 症状: 期待値を手計算で作ると、ブロック構造の解釈違いをテスト自身が追認してしまう。
  - 原因: IQ1_M は `iq1s_grid` という大きなテーブルを参照するため、自前で持つと写し間違いのリスクがある。
  - 抜け方: llama.cpp の `dequantize_row_tq1_0` / `dequantize_row_iq1_m` を**直接リンクして呼んだ出力**を期待値にした。ブロック構造体は `ggml-common.h` の `block_tq1_0` / `block_iq1_m` を `sizeof` で使い、`iq1s_grid` は `ggml-quants.c` が `GGML_COMMON_IMPL_C` で取り込むテーブルがそのまま効いた(追加の初期化もスタブも不要)。

- **分かったこと / 残った課題**
  - 生成パイプラインの検証: 2 型を足す前の生成器で旧 fixture を再生成すると md5 `36acb59a…` で完全一致(セクション数を除く 232,815 バイトが同一)。型を足した後も既存 18 セクションのバイト列は不変。
  - テストは 6 passed(20 型すべて BIT-EXACT。TQ1_0 / IQ1_M は各 4096 要素)。失敗時のメッセージを 4 種用意した(fixture 無し / セクション数不一致 / 必須型の欠落 / 途中で切れた fixture)。
  - **最適化レベル依存(実測)**: `-O0` で生成すると IQ3_XXS の f32 出力が `-O2` と一致しない(byte 168337 以降が 0x7f / 0xff で食い違う)。commit 済み fixture は `-O2` 生成なので、再生成は必ず `-O2`。TQ1_0 / IQ1_M は `-O0` と `-O2` で同一だった。
  - ライセンス: 参照元は llama.cpp の `ggml-quants.c` / `ggml-common.h`(MIT)。本リポジトリは MIT OR Apache-2.0。生成器は関数をリンクして呼ぶだけで取り込みは無く、fixture は出力データ。

- **次に効いたこと**: 「期待値は参照実装の出力そのものにする」が、量子化まわりの唯一の正攻法として定着した。最適化レベル依存の発見は、fixture を再生成するときの前提条件として README に残った。

---

### K3: Q4_0 のエッジブロックを参照実装とビット一致で固定し、CPU gemv の累積順を固定 (2026-10-22, `45a9696`)

- **意図**: 27B 本番の重みは Q4_0(ggml type 2)。K2 の fixture には擬似乱数の Q4_0 セクションが既にあり突き合わせ済みだったが、乱数 16 ブロックが「負の d」「全ゼロのニブル」「最大ニブル」を必ず含む保証は無い。境界値を決定論的に足して固定する。

- **変えたもの**
  - `45a9696` — 生成器に `emit_q4_0_edge()` を追加し、`Q4_0_EDGE` を 1 セクション追加(合計 21 セクション / 268,278 バイト / md5 `eeff5cb0…`)。
  - テスト 2 本(`q4_0_dequant_bitexact_vs_llama_cpp_edge_blocks`、`q4_gemv_accumulation_order_is_sequential_in_block_order`)。

- **つまずき**
  - 症状: エッジケースを作ったつもりが、実は K2 の乱数 fixture がすでに Q4_0 を含んでいた(前提が誤り)。
  - 原因: 「Q4_0 のセクションが無い」という前提で計画していた。
  - 抜け方: 前提を訂正した上で、乱数では保証できない境界値だけを足す方針に変えた。6 ブロック(d = -1.5 で下位 0..15、d = +0.5 で全ニブル 0 → q = -8、d = +0.5 で全ニブル 15 → q = +7、d = -0.75 で全 0xFF、d = +2.0 で上下別パターン、d = +0.0 で積が -0.0)。d はすべて f16 で厳密な値。

- **分かったこと / 残った課題**
  - パイプラインの検証: K3 前の生成器で再生成すると md5 `fe45c5bf…` で完全一致し、追加後も既存 20 セクションの生バイトと f32 出力は 1 バイトも変わらない。
  - 累積順の固定: k = 96(3 ブロック)の 1 出力を `0xca7b15f7` に固定した。同じ項列の pairwise 和 `0xca7b15f9`、2 アキュムレータ和 `0xca7b15fa`、逆順 `0xca7b15f9`、単一丸め(fma 相当)`0xca7b15f5` が**すべて別の値**になることも同時に確認するので、順序・融合が変われば必ず落ちる。実装は読んだ結果すでに文書化された順序で、変更は 0 行だった。
  - ゲート: engine lib 270 passed / 1 ignored。最初の 1 回だけ `prof::tests::attn_core_is_split_into_gdn_and_full_rows` が落ちたが、単独実行と再実行で pass(既知の大域カウンタ共有による flaky)。dequant_ref 7 passed、server lib 37 passed。
  - 測っていないこと: GPU 側の Q4_0 経路の累積順とビット一致、Q4_0 以外の型(Q4_1 / Q5_0 など)の再検証。

- **次に効いたこと**: 「実装が既に正しくても、順序をテストで固定しておく」という方針が、後の P37(行チャンク量子化)でバイト一致を主張する根拠になった。

---

### s164: `denno info` の GGUF フォールバック (2026-10-22, `8a0525b`)

- **意図**: config.json を持たない GGUF のみのディレクトリでは `denno info` が失敗し、次元を 1 つも出せなかった。GGUF のメタデータから埋める。

- **変えたもの**
  - `8a0525b` — `denno-cli` に `gguf_dims(dir)` と `info_resolve(dir)`。`gguf.rs` に `GgufValue::as_array_len()` を additive に追加。

- **つまずき**
  - 症状: `config.json を開けません` で終了し、行自体が出ない。
  - 原因: 次元の解決が config.json 固定で、GGUF を読む経路が無かった。
  - 抜け方: `general.architecture` を読み、`<arch>.block_count` → layers、`<arch>.embedding_length` → hidden、`<arch>.context_length` → context。arch 名はメタデータの値から組み立てる(決め打ちしない)。vocab は `tokenizer.ggml.tokens` の配列長、無ければ `<arch>.vocab_size`。config.json があればそちらが勝ち、GGUF は欠けた分だけ埋める(次元ごとに `.or()`)。

- **分かったこと / 残った課題**
  - 実測(GGUF のみのディレクトリ): `layers: 65` / `hidden: 5120` / `vocab: 248320` / `context: 262144`、`family: ?`。
  - **注意(実測)**: このファイルの `qwen35.block_count` は 65 で、同じモデルの config.json の `num_hidden_layers` は 64。差分 1 は `nextn_predict_layers = 1`(MTP 層)で、GGUF の block_count はそれを含む。仕様どおり block_count をそのまま出したため、層数だけがエンジンの実際の層数とずれた(次の s175 で訂正)。
  - denno-cli 13 → 18 passed、engine lib 274 passed / 1 ignored、server lib 37 passed、dequant_ref 7 passed。

- **次に効いたこと**: 「GGUF のメタデータは config.json の代わりになるが、意味論は同じではない」という理解が、層数 65 対 64 の扱い(s175)でそのまま活きた。

---

### s168: `denno info` の `quant:` が量子化型を出す (2026-10-22, `73d2e10`)

- **意図**: 27B q4 の GGUF で `quant: F32 (0)` と表示されていた。F32 は rmsnorm / bias の置き場で、実ファイルでは本数が量子化された重みを上回る(27B の Q4_0 ファイルは F32 456 本 / Q4_0 352 本 / Q5K 48 本 / Q4_1 8 本 / Q6K 1 本 / Q8_0 1 本)。全テンソルの最頻値では F32 が勝ってしまう。

- **変えたもの**
  - `73d2e10` — `dominant_tensor_type()` を追加し、`gguf_common_type()` の投票を `most_common_type()` から差し替え。
  - `most_common_type()` の意味(最頻値・同数は ggml type id の小さい方 = 決定論)と、表示書式 `<name> (<id>)` は不変。

- **つまずき**
  - 症状: 量子化の種類を聞いている行に F32 が出る。
  - 原因: 「最頻値」が「量子化型」の意味になっていなかった。F32 は構造上必ず一定数ある。
  - 抜け方: F32 以外が 1 本でもあれば F32 を投票から外す。F32 しか無い GGUF は従来どおり F32(空投票にしない)。テンソル 0 本は `None` → `?`。

- **分かったこと / 残った課題**
  - denno-cli 18 → 23 passed。変更前に戻すと `gguf_quant_value_skips_f32_storage_tensors` が `left: "F32 (0)"` / `right: "Q4_0 (2)"` で失敗することを確認した(テストに teeth があることの確認)。
  - 実機(CPU・実ファイル): `quant: F32 (0)` → `quant: Q4_0 (2)`(他の行は無変更)。
  - `layers`(65 対 64)と `family: ?` は意図的に無変更(運用判断待ち)。

- **次に効いたこと**: 「表示は事実だけを出すが、どの統計を取るかは意味に合わせる」という判断が、後の層数(推論トランク)の決定と同じ種類の論点になった。

---

### s175: `denno info` の GGUF のみリポジトリで層数を推論トランクに合わせた (2026-10-23, `d787f66`)

- **意図**: 前提を再確認したところ、HEAD は既に GGUF から次元を埋めていた(「0 が出る」は s164 より前のビルドの観測)。残っていた実バグは層数だけだった。

- **変えたもの**
  - `d787f66` — `gguf_dims` のみ。config.json が無い(または layers キーが欠けた)ときの GGUF 由来の層数を `<arch>.block_count - <arch>.nextn_predict_layers` にした。`nextn_predict_layers` が無ければ block_count そのまま。差し引きが 0 以下になる異常値は `None` = `?`。

- **つまずき**
  - 症状: GGUF のみのディレクトリで `layers: 65`。エンジンが実際に走らせるのは 64 層。
  - 原因: 27B の GGUF は `block_count = 65`(推論トランク 64 + MTP 1)。エンジンは config の `num_hidden_layers` を正として超過分 `blk.64` を nextn 層として読み飛ばすので、走る層数は 64。表示だけが 65 だった。
  - 抜け方: 表示を「エンジンが走らせる推論トランク」に合わせて差し引く。生の block_count を見たい場合は別行(例 `mtp: 1`)を足すのが筋で、これは範囲外として触っていない。

- **分かったこと / 残った課題**
  - テスト: engine の cuda ビルド lib 321 passed / 37 ignored(不変)、denno-cli 23 → 25 passed(+2: MTP 差し引きと、nextn == block_count の異常値・nextn 無し・config 優先)。
  - 実機(CPU・実ファイル): GGUF のみのディレクトリで `layers: 65` → `layers: 64`(hidden / vocab / context / quant 行は不変)。config.json のあるディレクトリは 32 のまま、config.json + MTP GGUF も config の 64 のまま。
  - 測っていないこと: GPU アンカー・ms/token・nsys(数値経路を 1 行も変えていないため構造的に不変だが、測ってはいない)。

- **次に効いたこと**: 「同じ事実に 2 つの表現があるとき、どちらを表示するかは運用の意味で決める」という判断が、索引側にも引き継ぐべき論点として残った。

---

### P35: GGUF Q4_0 の生ブロック passthrough を実データで判定 (2026-10-23, `98dd8e9`)

- **意図**: Q4_0 ネイティブ経路の残件「GGUF → fp32 → Q4_0 の再量子化をやめて生ブロックをそのまま積む」について、まずビット一致するかを実ファイルで確かめる。

- **変えたもの**
  - `98dd8e9` — `quant.rs` に `Q4Linear::from_gguf_blocks`(検証付きコピー、形が合わなければ `None` の fail-closed)、env ゲート `gguf_passthrough_*`(許可型 = 現在 0 個)。
  - `gguf.rs` に `GgufHeader`(mmap 無しのヘッダビュー。`GgufFile` と共通の解析関数を通すので解析規則は 1 本)、`qwen35.rs` の 2-D 線形重み 10 本 + untied `output.weight` をゲート経由に。

- **つまずき**
  - 症状: 生ブロックを積むと値が変わる。
  - 原因: 実ファイルの Q4_0 ブロックは `ggml` の `quantize_row_q4_0_ref` の規則で作られていない。ブロックを 1 個ダンプすると `d = 4.543304e-3`・ニブル範囲 1..12 で `max|q-8| = 7` だったが、ref 規則は最大絶対値の要素を必ずニブル 0(`|q-8| = 8`)に写す。そのため再量子化の `d'` は `max|y|/-8 = d*7/8` になり、比は 14 本すべてで 1.1426〜1.1433(= 8/7 = 1.142857、f16 丸めの幅)。
  - 抜け方: 一致しないことを受け入れ、許可型リストを空のままにした。ゲートを ON にしても従来の fp32 経路を通るので、数値は 1 ビットも変わらない。

- **分かったこと / 残った課題**
  - 実測(`#[ignore]` の proof test、CPU): mmap せずヘッダ(10,996,700 B)だけを解析し、テンソルは行境界 4 MB ずつ読んで `dequant_row` → `Q4Linear::quantize` とバイト比較(ピーク数十 MB、全体 5.1 s)。Q4_0 は **NOT identical**(352 本 / 12.441 GiB、サンプル 16 本、一致 2 本 = `token_embd.weight` と `blk.64.ffn_up.weight`)。F32 456 本・Q4_1 8 本・Q5K 48 本・Q6K 1 本・Q8_0 1 本は Q4_0 レイアウトでないため対象外。
  - 予測効果(**推定・未計測**): Q4_0 を許可した場合に省ける fp32 展開は 88.47 GiB(12.441 GiB × 4 B/elem ÷ 0.5625 B/elem = 7.111 倍。テンソルごとの一時確保で同時常駐ではない。最大は `token_embd.weight` の 5.086 GB)。proof test の実測レート 0.77 GiB / 5.1 s からロードの CPU 側は約 82 s 減る見込み。ただし許可型 0 個なので実際の節約は 0。
  - テスト: engine の cuda ビルド lib 323 passed / 38 ignored。

- **次に効いたこと**: 「再量子化では値が変わる」という事実を、後で P38 が「デ量子化なら変わらない」と読み替える。この 2 つを混同しないことが、token_embd の 5 GB 削減に繋がった。

---

### P37: 大きな 2-D 重みのロード時 fp32 一時領域を上限つきにした (2026-10-23, `1bc1702`)

- **意図**: `token_embd.weight`(27B で 1.271 G 要素)のロードは、テンソル全体の f32(5.086 GB)を作り、tied lm_head 用に `embed.clone()` でもう 1 部(さらに 5.086 GB)を作ってから量子化していた。このテンソルだけでピーク約 10.9 GB。60 GB のホストがサーバー常駐時に OOM-kill された主因で、数値ではなくロード時メモリの問題だった。

- **変えたもの**
  - `1bc1702` — `quant.rs` に `Q4Linear::zeros` / `quantize_rows_into`、`Quant8Linear::zeros` / `quantize_rows_into`。既存の `quantize` は「`zeros` して `quantize_rows_into(0, ..)` を 1 回呼ぶ」形へ本体を切り出し、新旧で同じコードを通す(バイト一致が構造的に保証される)。
  - 行チャンク駆動の `gguf_rows_quantized_chunked`、`ChunkedQuantStats`、env ゲート `DENNO_GGUF_CHUNKED_QUANT`(既定 ON、`0` / `false` / `no` / `off` で従来経路)、`chunk_rows_for`。

- **つまずき**
  - 症状: 量子化をチャンク化すると、行の途中でブロックが切れて値が変わる危険がある。
  - 原因: 行とブロックの境界が一致するかは量子化型に依存する。
  - 抜け方: 行単位(行 = k 要素)に切る。行は GGUF 上で連続(dim0 = k が最内)なので、行境界 = ブロック境界になる(`k % blck_size == 0` を検証し、満たさなければ従来経路)。Q4_0 の 32 要素ブロックも K 系の 256 要素スーパーブロックも行内で閉じる。チャンク幅は 1 チャンクの f32 が 16 MiB(4 M 要素)になる行数(k=5120 → 819 行、6144 → 682 行、17408 → 240 行)。

- **分かったこと / 残った課題**
  - 実測(proof test、CPU): 10 本を比較して **10/10 バイト一致**(`token_embd` Q4_0、`output` Q6K、`ffn_up` Q4_0、`ffn_gate` Q4_0、`ffn_down` Q4_1、`attn_q` Q4_0、`attn_qkv` Q4_0、`attn_v` Q4_0、`attn_output` Q4_0、`ssm_out` Q5K)。int8 の副次比較も 5 本すべて一致。ピーク f32 一時領域(計算値)は chunked 20.21 MB / 全体 5,085.59 MB(252 倍)。23.5 s。
  - 予測効果(**推定・未計測**): `token_embd` のロード時ピークは 10.887 GB → 5.820 GB で約 5.07 GB 減。ffn / attn は 1 本あたり最大 356.5 MB → 19.1 MB。実機のロード時間・RSS は計測対象外。
  - テスト: engine の cuda ビルド lib 327 passed / 39 ignored。

- **次に効いたこと**: 「一時領域のピークを計算値で示し、実測は別に取る」という書き分けが、後の P42 の REPORT でもそのまま使われた。常駐 5.086 GB の表が残ることは、この時点で課題として記録されている。

---

### P38: `token_embd` の 5.086 GB f32 表を行単位デ量子化で置き換えられるかの判定 (2026-10-23, `cf917e5`)

- **意図**: P37 の後も `token_embd.weight` の f32 gather 表(5.086 GB)が常駐で残っていた。ローダ最大のホスト RAM 項目で、これが消せるかどうかを数値で決める。製品コードの変更は無く、解析と証明テストだけ。

- **変えたもの**
  - `cf917e5` — proof test `gguf::tests::q4_row_dequant_matches_embed_table_real_gguf`(`#[ignore]`、CPU、mmap せず seek + read_exact)。

- **つまずき**
  - 症状: P35 の「このファイルの Q4_0 ブロックは再量子化で再現できない(`d' = d*7/8`)」を「表は置き換えられない」と読んでしまっていた。
  - 原因: **再量子化**(f32 → Q4_0 エンコード)と**デ量子化**(Q4_0 → f32)の混同。表はファイルのブロックをデ量子化するだけで作られる。
  - 抜け方: `dequant_row` がブロック内で閉じていること(ブロック間の状態を持たない)と、`row_bytes = 160 × 18 = 2880` が 32 要素ブロックの整数倍であること(行境界 = ブロック境界)を確認。よって 1 行だけのデ量子化は、同じバイトに対する同じ 160 ブロックの計算になり、f32 のビットが一致する。

- **分かったこと / 残った課題**
  - 実測: `token_embd.weight` k=5120 / n=248320 / Q4_0(生 715.2 MB・表 5,085.6 MB)。40 行(0, 1, 2、チャンク境界 818 / 819 / 820 / 1637〜1639、等間隔、248318 / 248319)を 32 チャンクに跨って比較して **40/40 一致**。さらに全 248,320 行のスイープで**差分 0**(3.30 s)。比較はすべて f32 のビットパターン。
  - コスト(この CPU・ページキャッシュ warm の実測): 1 行 = 2,880 B の pread + `dequant_row` で B=1 が 3.09 µs/token、B=8 が 24.93 µs、B=32 が 103.91 µs。純デ量子化は 1.9 µs/行で、残りは pread とアロケート。ピーク一時領域は 23.4 KB。B=1 の 3.09 µs/token は 17.30 ms/token の KPI の 0.018 % という**算術**であり、GPU 実測ではない。
  - 判定: ビット一致で、実装する価値がある。常駐削減は、mmap 済みファイル領域から行をデ量子化するなら 5,085.6 MB(ページキャッシュ = 回収可能)、715.2 MB の生ブロックコピーを常駐させるなら 4,370.4 MB。行キャッシュは不要(64 行でも 1.31 MB)。
  - **`lm_head` の Q4 コピーは供給元にできない** — あれは再量子化されたもので(P35 の許可型は空のまま)、ファイルのブロックではないため、デ量子化してもビット一致しない。
  - テスト: engine の cuda ビルド lib 327 passed / 40 ignored。なお `tests/kernel_invariants.rs::gdn_kernel_shared_layout_matches_host_smem` は pristine な HEAD で赤(既存・無関係)だった。

- **次に効いたこと**: 「証明テストを先に書き、実装はその結果で決める」という順序が、次の P42 の設計をそのまま決めた。スケッチ(`EmbedRows::{F32, Blocks}`、3 つの gather サイト、既定 ON のゲート)が実装の形になった。

---

### P42: `token_embd` の常駐 5,086 MB f32 表を行単位デ量子化へ置き換え (2026-10-23, `955690e`)

- **意図**: 27B q4 serve のホスト RSS のうち `token_embd.weight` の gather 用 f32 表が 5,085,593,600 B を占めていた。P38 が「行単位のデ量子化でビット一致する」ことを実ファイルで証明したので、実装する。

- **変えたもの**
  - `955690e` — `gguf.rs` に `embed_blocks_and_linear_chunked`(f32 表の代わりにファイル自身の生ブロック `n*row_bytes` を確保し、tied lm_head 用の量子化コピーは従来と同じ行チャンクで作る)と前提判定 `embed_blocks_eligible`(型が Q4_0・`k % 32 == 0`・`nbytes == n*row_bytes`・`vocab*e == elems`)。
  - `qwen35.rs` の `embed: Vec<f32>` を `EmbedTable { F32, Q4Rows { ... } }` に置き換え、読者(decode 1 行 / prefill n 行 / chain upload 1 行 / テスト補助)を `row()` / `row_into()` へ移した。復元は `kquant::dequant_row_into`(新設、`dequant_row` と同一本体)で、再量子化は一切しない。

- **つまずき**
  - 症状: 表を置き換えると、Fp32 経路(tied fp32 lm_head)が 5 GB を黙って確保してしまう危険がある。
  - 原因: 表の読み手が 3 箇所(decode / prefill / GPU chain upload)に散っており、経路ごとに前提が違う。
  - 抜け方: `F32` variant は現行の `&w[i*e..][..e]` とバイト同一にし、前提が崩れたときの fail-closed 先にした。R4 / R5(tied fp32 lm_head・全行 matvec)は `quant != Fp32` では到達しないため、`Q4Rows` で呼ばれたら理由付きで panic する(黙って 5 GB を確保しない)。既定 ON、`DENNO_EMBED_ONDEMAND=0|false|no|off` で従来の f32 表に戻る。

- **分かったこと / 残った課題**
  - ビット一致の根拠: 27B Q4_0(k=5,120 / n=248,320 / row_bytes=2,880)で `EmbedTable::Q4Rows` の全 248,320 行をローダのチャンク dequant と比較し `to_bits()` 差分 0(実測 3.19 s、常駐 715.2 MB、pread 0.66 s)。合成ブロックの単体テストは行 0・最終行・チャンク境界を跨ぐ行を含み、`F32` variant との一致も固定。
  - テスト: engine の cuda ビルド lib 335 passed / 41 ignored(基準 330 + 新規 5)、`--test kernel_invariants` 11 passed、非 cuda lib 291 passed。
  - **未検証(推定)**: ホスト常駐の削減量 **-4,370.4 MB** は確保バイト数の差(5,085,593,600 → 715,161,600 B)であって RSS 実測ではない。KPI への影響(decode +0.003 ms/token = 0.018 %、prefill +9.8 ms @3,161 トークン / +39.7 ms @12,850 トークン = 0.10 %)は P38 の 3.09 µs/token から予測した値で、実機では未計測。製品経路の 1 行復元は CPU 実測 1.95 µs/row で、予測に使った 3.09 µs より小さい(予測は保守側)。アンカー md5 の再取得は未実施。

- **次に効いたこと**: 「常駐を減らす変更は、削減量を確保バイト数で示し、RSS と KPI は別に測る」という書き分けが定着した。数値経路に触れない変更を、ビット一致の証明と fail-closed の設計で安全に入れる型がここで完成した。
