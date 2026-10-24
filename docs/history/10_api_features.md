# 第2巻 API 機能 — tool use・thinking・停止条件・切断検知

対象期間: 2026-10-12 〜 2026-10-19。サーバー層(`denno-server`)とエンジンのテキスト後処理に入った機能の記録。
モデル構造の話は巻3、速度とカーネルの話は別巻に分かれている。
日付は commit の author date を JST で表す。

## この巻で分かること

- Anthropic Messages API 互換を保ったまま tool use / tool_choice を足した設計(プロンプト変換・抽出・拘束デコード)
- SSE の組み立て(streaming tool_use)と、切断検知・打ち切りがどこまで可能かという限界
- thinking 制御が「文字列 → マップ受理 → 既定の意味変更」と 3 回変わった経緯と、その理由
- 停止条件(ロールラベル停止・holdback)と反復抑止 3 種を、既定ビット不変のまま入れる方法

## 前提知識

Anthropic Messages API のリクエスト/レスポンス形(content block、`stop_reason`、SSE の event 列)、byte-level BPE のトークン境界、
chat template(Jinja)によるプロンプト描画。この巻の変更はすべて「テキスト列の後処理」か「logits のマスク」で、forward の数値には触れていない。

## 読む順序

P11(ストリーミング detok)→ tool use → tool_choice 拘束 → streaming tool_use → 切断検知 → max_tokens 既定 →
非ストリーム切断とログ → thinking(s47 → s52 → s53c)→ 停止条件 → 反復抑止 → temperature 上限、の順。
thinking の 3 エントリは必ず順に読む(後段が前段の意味を変えている)。

## 対象 phase 一覧

| phase | date | commit | one-line summary |
|---|---|---|---|
| P11 | 2026-10-12 | `1a7fee6` | ストリーミング detok の U+FFFD 保留と gpt2 exactness 基準の更新 |
| tool use | 2026-10-16 | `751dc10` | tools / tool_choice の受理、chat template 変換、`<tool_call>` 抽出 |
| tool_choice 拘束 | 2026-10-16 | `a16b203` | `any` / `tool` を logits マスクで token-level に強制 |
| streaming tool_use | 2026-10-16 | `05e5f8f` | SSE アセンブラで delta 境界を跨ぐ `<tool_call>` を組み立てる |
| 切断検知 | 2026-10-16 | `e7d3e7a` | SSE の `is_closed()` をステップ境界で見て生成を abort |
| max_tokens 既定 | 2026-10-16 | `9636337` | 未指定時のみ 4096 を適用(`max_completion_tokens` も受理) |
| s44 非ストリーム切断 | 2026-10-17 | `08e7c9b` | respond の閉鎖判定と、非ブロッキング stdout ログ |
| s47 thinking 制御 | 2026-10-17 | `4e89c11` | `thinking: off / brief / full` を受理し template 変数へ渡す |
| s52 thinking マップ | 2026-10-18 | `b9d29f8` | `{"type":"enabled"}` 形式を受理(`budget_tokens` は読まない) |
| s53c brief の意味変更 | 2026-10-18 | `e7246b8` | brief を「思考無効 + 簡潔指示」へ、マップ `enabled` を off 扱い |
| s53 停止条件 | 2026-10-19 | `f21a985` | ロールラベル停止の holdback と既定 max_tokens 1024 |
| s56/s57 反復抑止 | 2026-10-19 | `b3eaf71` | no_repeat_ngram / frequency / presence を opt-in で追加 |
| temperature 上限 | 2026-10-19 | `47e1928` | `--temperature-cap` で実効 temperature をクランプ |

---

### P11: ストリーミング detok のバイト分割文字修正と gpt2 exactness 基準の更新 (2026-10-12, `1a7fee6`)

- **意図**: gpt2 の exactness テストが文字化けで落ち、モデルを取り直しても直らなかった。原因を切り分けて、テスト基準そのものも現行化する必要があった。

- **変えたもの**
  - `1a7fee6` — ストリーミング detok の保留処理(`emit_delta_and_check_stop`)と `flush_stream` の新設。
  - CI ワークフローに gpt2 のモデル取得を追加(exactness ジョブ)。

- **つまずき**
  - 症状: 出力が「??の??は?」のように文字化けする。モデルファイルは同一(548,105,171 B)で、英語の raw 生成は正常。
  - 原因: 2 つ。(1) CI の exactness ジョブが Qwen しか取得しておらず、gpt2 テストは「モデル未キャッシュ → スキップ」で一度も実行されていなかった(基準文字列が当時のまま残存)。(2) バイトレベル BPE ではマルチバイト文字がトークン境界で割れ、未完成バイト列が U+FFFD になる。その U+FFFD を確定済みテキストへ混ぜると以後のバイト差分がずれて文字が欠ける。
  - 抜け方: 末尾が U+FFFD になるトークンを保留し、文字が完成したステップでまとめて差分として流す。生成終了時(EOS / max_tokens)に `flush_stream` で保留分を吐く。参照実装(HF transformers・fp32・greedy)と突き合わせて、denno の出力が完全一致することを確認した。

- **分かったこと / 残った課題**
  - 単体 99 + 7、exactness 5 件が緑。CI で gpt2 テストが実際に走って緑になった(基準文字列が参照実装と一致することの実証)。
  - 数値経路は不変(テキスト経路のみ)。基準出力と 4 セッション同時 == solo の実測で確認している。

- **次に効いたこと**: 「保留して確定してから流す」holdback の考え方が、後のロールラベル停止(この巻の終盤)と streaming tool_use の境界追跡にそのまま再利用された。

---

### tool use 対応 — Anthropic Messages API 互換の tools / tool_choice (2026-10-16, `751dc10`)

- **意図**: クライアント側が tool use を前提に動いており、`tools` を送っても 400 になるのが実運用の blocker だった。API 層だけで完結させ、モデルの数値経路には触らない方針を取った。

- **変えたもの**
  - `751dc10` — `crates/denno-server/src/lib.rs`(パース・検証・抽出・応答組み立て)、`tokenizer.rs`(`apply_chat_template` に tools 引数)、`engine.rs`(`begin_chat_with_tools`)。
  - `docs/api.md` に tool use 節を追加。

- **つまずき**
  - 症状: `tools` 付きリクエストが 400。モデル出力から tool 呼び出しを取り出す手段も無い。
  - 原因: リクエスト側の受理も、chat template への `tools` 変数の受け渡しも、出力の構造化抽出も未実装だった。
  - 抜け方: 検証済み tools を Qwen/HF 形式(`{"type":"function","function":{...}}`)へ変換して template の `tools` 変数へ渡す。履歴側は assistant の `tool_use` block を公式テンプレート準拠の `<tool_call>` XML に、user の `tool_result` を `<tool_response>` に変換する。応答側は完全な `<tool_call>` block だけを抽出し、**抽出に成功した block だけ**テキストから除く(パース失敗・未完結はテキストとして残す = 形式例を出力した場合の防御)。

- **分かったこと / 残った課題**
  - 実機 curl で tools 付き生成(in=277)→ tool_result を返す 2 ターン目(in=332)→ `tool_choice: none` が tools 無しと同一出力、まで確認。tools 経路の連続 2 回決定論も一致。
  - tools 無し経路は template に null を渡すため描画がビット同一。残った課題は streaming(tools + stream は 400 で拒否)と `any`/`tool` の強制。

- **次に効いたこと**: 「抽出に成功した block だけ除く」という規則が、後の streaming tool_use の意味論(未完結 block はテキスト)の基準になった。

---

### tool_choice 拘束デコード — `any` / `tool` の token-level 制約 (2026-10-16, `a16b203`)

- **意図**: `tool_choice` を受理するだけで効かせていなかった(`any`/`tool` が auto と同じ挙動)。ツール名を確実に選ばせたい用途に応えるため、サンプリング直前の logits マスクで強制する。

- **変えたもの**
  - `a16b203` — 新規 `crates/denno-engine/src/toolmask.rs`(`ToolConstraint` / `ToolMask` 状態機械)。
  - `engine.rs`(`ChatSession.tool_mask`、sample 直前 `apply` + sample 後 `feed`)、`tokenizer.rs`(`piece_bytes`)、`server/lib.rs` と `scheduler.rs`(constraint の配線)。

- **つまずき**
  - 症状: マスクを入れると、block の途中で EOS が出て壊れる / 許容集合が空になって全確率が 0 になる。
  - 原因: EOS の扱いと、語彙側に必要な綴りが存在しない場合の縮退が未定義だった。また状態機械が sample 済みトークンで前進できないと無限ループになる。
  - 抜け方: 拘束未充足の間は EOS を許容集合から外し、CLOSE 完了で許可に戻す。許容集合が空なら警告の上マスク解除。`feed` で遷移不能になったら状態を Done に落として拘束解除(死に筋の防止)。

- **分かったこと / 残った課題**
  - 状態は Watch → Head(`\n<function=` を 1 バイトずつ強制)→ Name(許容名の接頭辞のみ)→ Body(自由・CLOSE を監視)→ Done(無マスク)。許容集合は状態変化時のみ再計算してキャッシュする。
  - 準拠するスクリプトなら拘束有りと無しでビット同一。拘束無し経路はマスクが存在しないため従来とビット同一。実モデルでの `any`/`tool` 検証は未実施(当時の残課題)。

- **次に効いたこと**: 「logits の -inf 置換だけ」で強制する設計は、反復抑止(この巻の終盤)の no_repeat_ngram と同じ適用位置・同じ順序規律に揃えられた。

---

### streaming tool_use 対応 — SSE アセンブラ (2026-10-16, `05e5f8f`)

- **意図**: tools + stream を 400 で拒否していたため、クライアントはツール会話をストリーミングで使えなかった。非ストリーミングの `extract_tool_calls` と同一の意味論を delta ストリームへ移植する。

- **変えたもの**
  - `05e5f8f` — 新規 `crates/denno-server/src/stream_asm.rs`(`StreamAssembler`)。
  - `scheduler.rs`(`Run.asm`)、`lib.rs`(tools + stream の 400 を撤去)、`engine.rs`(`on_deltas` を `Box<dyn FnMut(&str) + '_>` へ)。

- **つまずき**
  - 症状: `<tool_call>` の OPEN/CLOSE が delta 境界で分割され、先頭部分がテキストとして漏れる。借用エラー(`E0505`)でコンパイルが通らない。
  - 原因: アセンブラを Run から取り出してコールバックへ渡すと、`'static` 要求と move が衝突する。分割到着する OPEN の先頭を確定テキストとして出してしまう。
  - 抜け方: `scan` バッファで追跡し、末尾が OPEN 接頭辞の可能性がなくなるまで確定テキストを出さない。コールバックの借用区間を block に閉じ、`asms` を Run へ戻す move と分離した。trim 規約も非ストリーミングと一致させ、任意の分割幅で `extract_tool_calls(全文)` の残テキストと一致することを単体テストで固定。

- **分かったこと / 残った課題**
  - `stream_asm.rs` 単体 10 件(chunk 1〜100 の全分割、OPEN/CLOSE の全バイト境界、部分 OPEN 接頭辞の hold、複数 block の index 昇順)が緑。`tools_with_stream_is_rejected` を `tools_with_stream_is_accepted` に置換し、SSE の完整性を検証した。workspace 215 passed。
  - 残った課題: 実モデルでの tool 呼び出しストリーミングの確認(当時は合成語彙での検証まで)。

- **次に効いたこと**: 「アセンブラを Run に持たせ、ステップ境界で出し入れする」形が、切断検知のステップ境界スイープと同じ場所に乗った。scheduler の Run が会話状態の置き場になる、という構造がここで固まった。

---

### クライアント切断検知と生成 abort (SSE ストリーム) (2026-10-16, `e7d3e7a`)

- **意図**: クライアントが接続を放棄してもサーバーが生成を続け、実測 13 分超の生成が単一ワーカーを占有して後続リクエストが全て列待ちになっていた。SSE に限り切断を検知して打ち切る。

- **変えたもの**
  - `e7d3e7a` — `scheduler.rs`(`Run::client_gone`、切断スイープ、`admit_job` への共通化)。
  - 新規 `crates/denno-server/tests/client_disconnect.rs`(実 TCP ソケットで切断を再現)。

- **つまずき**
  - 症状: 切断後も `in_flight` が 1 のまま減らない。delta を 1 つも送らない局面(prefill 中・待ち行列中)では永久に検知できない。
  - 原因: delta コールバック内の送信失敗を握り潰しており(`let _ = blocking_send`)、接続状態を見る機会がそこしか無かった。
  - 抜け方: SSE の応答ボディは `ReceiverStream` なので、クライアントが切れると hyper が SSE イベント / ping の書き込み時に broken pipe を検知してボディを drop し、チャネルの `Sender::is_closed()` が true になる。ワーカーはこの閉鎖を decode / prefill の**各ステップ境界**で確認し、該当 Run を除去して `release_session` する。

- **分かったこと / 残った課題**
  - 4 件のテストで「生成中の切断」「最初のトークン前の切断」「受信側先行 drop」「正常系」を固定。修正前コードでは前者 2 件が FAIL することを確認している。workspace 220 passed。
  - 限界: abort はステップ境界単位なので、1 forward の途中(数千トークンの prefill)では効かない。非ストリームは当時「接続状態を観測する機会が無い」として対象外にした。

- **次に効いたこと**: 「非ストリームも同じ sweep に乗せられるのではないか」という疑問が次々エントリ(s44)に繋がり、そこで前提が訂正されることになる。

---

### chat 用 max_tokens 既定上限(未指定時のみ適用)(2026-10-16, `9636337`)

- **意図**: thinking モデルが超長生成を続け、単一ワーカーのサーバーで後続を長時間待たせていた。生成開始前にリクエスト側で上限を保証する。

- **変えたもの**
  - `9636337` — `crates/denno-server/src/lib.rs`(`DEFAULT_MAX_TOKENS = 4096`、`resolve_max_tokens`、`MessagesRequest.max_tokens` を `Option<usize>` へ)。
  - 新規 `crates/denno-server/tests/max_tokens_default.rs`。

- **つまずき**
  - 症状: `max_tokens` を省くと 400(必須フィールド欠落)。付けると上限が効かない。
  - 原因: Anthropic 仕様上 `max_tokens` は必須で、実装も必須として読んでいた。既定値の概念が無かった。
  - 抜け方: `max_tokens` を任意化し、未指定時のみ 4096 を適用。OpenAI 互換の別名 `max_completion_tokens` も受け、両方あれば `max_tokens` を優先。明示指定(0 含む)は尊重する。適用箇所はハンドラの `GenParams` 構築 1 箇所なので、非ストリームと SSE の両経路に同じく効く。

- **分かったこと / 残った課題**
  - 既定 4096 の根拠は engine 側の 1 セッションあたり KV 予約量と同値で、KV 予算設計の前提と揃う。
  - 7 件のテスト(EOS を出さないモデルで `usage.output_tokens` を直接観測)が緑。workspace 227 passed。

- **次に効いたこと**: 「未指定なら既定、明示なら尊重」という規則は、後の既定 1024 への変更(s53)でもそのまま維持された。既定値だけを差し替えられる形にしておいたのが効いた。

---

### 非ストリーム /v1/messages のクライアント切断後処理の堅牢化(s44)(2026-10-17, `08e7c9b`)

- **意図**: 非ストリームの切断後に完了ログが欠け、さらにログ消費者が死ぬとサーバーが新規接続に応答しなくなる(wedge)事例があった。2 つの独立した問題として潰す。

- **変えたもの**
  - `08e7c9b` — `scheduler.rs`(`Run::client_gone` に respond の閉鎖判定を追加)。
  - 新規 `crates/denno-server/src/logging.rs`(`NonBlockingStdout`、有界キュー 4096 行)、`tests/nonstream_disconnect.rs`、`tests/frozen_log_robustness.rs`。

- **つまずき**
  - 症状 1: 非ストリームの切断が検知されず、単一ワーカーを塞ぎ続ける。
  - 症状 2: ログの消費者(パイプの先)が凍結すると、サーバー全体が CPU 0%・`/health` 無応答になる。
  - 原因: (1) s38 の「hyper はレスポンス送信まで接続を観測しない」という前提が誤りで、実際は応答 future が pending の間も read 側を見ており、切断時に handler future ごと drop される。(2) 同期 stdout ロギングがパイプバッファ(64 KB)満杯で `write()` に永久ブロックし、`std::io::stdout()` のグローバルロックを保持したまま全スレッドを止める。
  - 抜け方: (1) handler が持つ oneshot の `respond.is_closed()` を切断の間接観測に使い、SSE と同一の sweep に乗せる。(2) ログ行を有界キューに入れ、stdout へ書くのは専用スレッド 1 本だけにする。満杯は破棄してカウントし、復旧後に 1 行で報告する。

- **分かったこと / 残った課題**
  - テスト 6 件が緑(凍結シンク下でも `/health` と後続リクエストが完遂し、ログ行が「ブロックではなく破棄」で処理されたことまで検証)。workspace 237 passed。
  - 教訓として残ったのは「ログはベストエフォートでなければならない」こと。推論サーバーのログ経路がボトルネックになり得る。

- **次に効いたこと**: この後、計測出力(prof)でも「ログ洪水で本番を壊す」型の事故が別途起きた。出力点を増やすときは必ず上限と破棄規則をセットで考える、という規律に繋がった。

---

### thinking 制御(`thinking`: `off` | `brief` | `full`)の実装(s47)(2026-10-17, `4e89c11`)

- **意図**: 思考モデルの思考量をリクエスト側から制御したかった。Anthropic API には無い拡張だが、クライアントが既に独自フィールドを送れる前提だったので、トップレベルの任意フィールドとして受ける。

- **変えたもの**
  - `4e89c11` — `tokenizer.rs`(`apply_chat_template` 系に `enable_thinking: Option<bool>`)、`engine.rs`(`begin_chat_with_tools` に同引数)、`server/lib.rs`(`Thinking` 列挙・`parse_thinking`・`brief_system`)、`scheduler.rs`(`Job` へ伝搬)。
  - `testutil.rs` に `ThinkingSimModel` と、実モデルと同じ `enable_thinking` 分岐を持つテンプレートを追加。

- **つまずき**
  - 症状: off にしても思考が始まる / テンプレート変数が未定義のときの挙動が読めない。
  - 原因: 実テンプレートは `{% if enable_thinking is defined and enable_thinking is false %}` の形で参照しており、変数を渡さない場合は「思考 ON」になる。変数の有無と真偽を区別する必要があった。
  - 抜け方: 指定されたときだけ ctx 変数として渡す(None = 変数無し = 従来経路と同一)。off のときの generation prompt は空の think block になり、実モデルの chat_template と同じく終了タグの直前が改行 3 つになる(実ファイルを python3 で展開して確認)。

- **分かったこと / 残った課題**
  - off = full + `<think>` `</think>` の 2 トークン増、という形でテストに固定した。brief は当時「思考 ON のまま system 冒頭に短答誘導文を注入する」ソフト制御で、ハード制御ではないと明記していた。
  - テスト 11 件が緑、workspace 253 passed。実機での brief の効果確認は残課題。

- **次に効いたこと**: 「変数を渡さない = 従来経路とビット同一」という設計が、既存アンカーを壊さずに機能を足す型として定着した。後の s52 / s53c もこの型の上で意味だけを差し替えている。

---

### thinking マップ形式の受理 — Anthropic thinking マップをクライアント互換で受ける(s52)(2026-10-18, `b9d29f8`)

- **意図**: クライアントが thinking をマップ形式 `{"type":"enabled","budget_tokens":8000}` で送るため、文字列のみ受理の s47 パーサでは 400 になり E2E が通らなかった。thinking だけが blocker だった。

- **変えたもの**
  - `b9d29f8` — `server/lib.rs`(`MessagesRequest.thinking` を `Option<serde_json::Value>` へ、`parse_thinking` を値全体を見る方式に)。
  - `docs/api.md` に thinking 節(両形式・無視される `budget_tokens`・400 の条件)。

- **つまずき**
  - 症状: マップ形式で 400。文字列形式は通る。
  - 原因: パーサの入力型が `Option<String>` で、オブジェクトが serde の段階で落ちていた。
  - 抜け方: `Option<Value>` に変え、String は従来どおり、Object は `type` で分岐(`enabled` → brief、`disabled` → off、未知は 400)、それ以外の型(数値・真偽値・配列)も 400。`budget_tokens` は読まない(決定的にするため)。

- **分かったこと / 残った課題**
  - 呼び出し側は署名互換のまま無変更で、以降の brief 注入と `enable_thinking` の経路は s47 と完全に同一。テスト 3 件 + 既存 1 件更新、workspace 256 passed。
  - 残った課題: 実機でのクライアント E2E 再確認(当時は CPU 経路の wire テストまで)。

- **次に効いたこと**: 「受理は広く、解釈は決定的に」という方針(未知フィールドは拒否しない、解釈に使わないフィールドは読まない)が、後の反復抑止パラメータの検証でも踏襲された。

---

### thinking=brief の意味変更 — 思考を無効化し、マップ enabled を off 扱いへ(s53c)(2026-10-18, `e7246b8`)

- **意図**: 実測で、9B が「Thinking Process:」のメタ分析を延々と書き、回答に到達しなかった。brief が「思考 ON + 誘導文」だったため、誘導が効いていなかった。

- **変えたもの**
  - `e7246b8` — `server/lib.rs`(brief も `enable_thinking=Some(false)` に、`THINKING_BRIEF_INSTRUCTION` を「回答は簡潔に、数文でまとめてください。」へ、マップ `enabled` → `Off`)。
  - `tests/thinking_control.rs` と `docs/api.md` を追随。

- **つまずき**
  - 症状: brief(既定)でも長い思考が走り、回答が返らない。
  - 実測: 思考 ON + 誘導文の比較で、**4 候補すべて**が「Thinking Process:」の英語メタ分析を続けて回答に到達せず(max_tokens 切れ)。一方 off + 簡潔指示は 2.0 s で完答した。**誘導文だけでは直らない**と確定。
  - 原因: brief がソフト制御(誘導文のみ)だったため、モデルの生成を実際には制約していなかった。
  - 抜け方: brief を「思考無効 + 簡潔指示」に変え、マップ `enabled` も off 扱いにした。つまり既定は実質「思考しない」。

- **分かったこと / 残った課題**
  - 既定の意味が変わるため、既存の wire テストの基準側を更新した(エンジン直実行系は system 無しのままなので影響なし)。
  - 副作用として「思考は明示的に `full` を送らないと走らない」運用になった。これは後で temperature の問題(この巻の終盤)と組み合わさって別の症状を生む。

- **次に効いたこと**: 「既定を変えるときは、その既定に依存している全テストの基準側を同時に更新する」という手順が定着した。ここでの変更が、temperature 上限の判断根拠の一部になった。

---

### 停止条件の強化 — ロールラベル停止と既定 max_tokens 1024(s53)(2026-10-19, `f21a985`)

- **意図**: 生成テキストに次の発話者ヘッダが現れると、そのまま続けて別ロールの文章を書き始める(偽ターン)。llama.cpp の reverse prompt に相当する停止を入れる。同時に既定の生成長を 4096 から 1024 へ下げる。

- **変えたもの**
  - `f21a985` — `crates/denno-engine/src/engine.rs`(`sent_len`、`StopStatus`、`find_stop_marker`、`line_label_at`、`tail_is_line_end`、`partial_holdback`、`flush_stream` の holdback 解放)。
  - `server/lib.rs`(`DEFAULT_MAX_TOKENS` を 1024、`MAX_TOKENS_CLAMP = 4096` で明示指定を上限クランプ)。

- **つまずき**
  - 症状: SSE の delta に `u` / `us` / `use` のような途中文字列が漏れる。`userland` のような語で誤停止する。
  - 原因: マーカーがトークン境界をまたぐため、確定前の文字列を流してしまう。行ラベル判定と prefix 判定で語境界の見方が揃っていなかった。
  - 抜け方: 末尾がマーカーの prefix になり得る間は送出しない holdback を入れ、`sent_len`(送出済みバイト数)を不変条件にした。生成終了時は必ず `sent_len == text.len()` になるので、通常応答の最終テキストは従来と同一。`userland` は holdback した上で非停止にする。

- **分かったこと / 残った課題**
  - マーカーの最長バイト長は `MAX_STOP_MARKER_LEN = 12`(`<|im_start|>`)。holdback の上限計算にも使う。クライアント指定の `stop_sequences` とは「テキスト上の最も早い停止位置」で併用する。
  - 既知の限界: prefix 経路は語境界を見ないため、改行後の `userland` でも(holdback の後)停止し得る。安全側の挙動として受け入れた。テスト 5 件追加、workspace 261 passed。

- **次に効いたこと**: 「送出済み境界を 1 つの整数で持つ」設計が、後の切断検知・SSE アセンブラの状態管理と同じ発想で、テキスト経路の不変条件をテストで固定する手本になった。

---

### 反復抑止の opt-in 3 種 — no_repeat_ngram / frequency / presence(s56/s57)(2026-10-19, `b3eaf71`)

- **意図**: 9B の品質改善で入れた repetition penalty だけでは、長文の反復崩れを抑えきれなかった。OpenAI 互換の frequency / presence と、llama.cpp / HF 互換の no_repeat_ngram を**追加の opt-in** として足す。

- **変えたもの**
  - `b3eaf71` — `sampler.rs`(`FreqPresencePenalty`、`context_counts`、`apply_freq_presence_penalty`、`apply_no_repeat_ngram`、`AntiRepetition`)。
  - `engine.rs`(`GenParams` に 3 フィールド、`apply_session_penalty` を逐次・バッチの共通入口へ)、`server/lib.rs` の検証、CLI の 3 フラグ。
  - `a9dd857` — serve の既定 repetition penalty フラグ(9B int8 + tools の崩壊回避)。

- **つまずき**
  - 症状: 既定で有効にすると既存の出力が変わる。順序を入れ替えるとビット契約が壊れる。
  - 原因: 4 種類の補正が同じ logits に順番に効くため、適用順が数値の同一性を決める。
  - 抜け方: 適用順を「生 logits → repetition → frequency → presence → no_repeat_ngram(-inf)→ temperature → top-k → top-p」に固定。既定はすべて無効で、`AntiRepetition::is_noop()` が真なら文脈構築も counts 確保も省く(既定経路のコストまで不変)。

- **分かったこと / 残った課題**
  - frequency と presence の**内部**順序は f32 で観測可能なのでテストで殺した。一方 no_repeat_ngram を frequency/presence の前後どちらに置いても f32 では観測不能(-inf マスクが前段の結果を上書きする)ため、変異テストでも生存を確認した。よって「最後に置く」ことはテストではなくコード構造とドキュメントで担保している(正直な注記)。
  - workspace 290 passed(ベース 264 から +26)。反復崩れが実際に抑えられるかの品質評価は GPU 実機なしでは不可として残した。

- **次に効いたこと**: 「既定 no-op をビットで証明する」やり方が、後の量子化・ロード経路の変更(巻3)でも標準の検査になった。opt-in を既定 off で入れてから計測する、という順序もここで固定された。

---

### serve に temperature 上限を追加(`--temperature-cap`)(2026-10-19, `47e1928`)

- **意図**: 9B との会話が 2〜3 ターンで崩壊する件の切り分け。実セッションの履歴を見ると、turn 2 で中国語が混入し、別セッションではポーランド語・タイ語の断片が出ていた。原因を特定してサーバー側で止める。

- **変えたもの**
  - `47e1928` — `denno serve --temperature-cap <f32>`。実効 temperature を `min(要求値, c)` にクランプし、リクエストが temperature を省略した場合も sampler 既定の 1.0 を頭打ちにする。
  - 負値・非有限は起動時に拒否。ユニットテスト 4 本。

- **つまずき**
  - 症状: 多言語の混線。キャッシュを疑ったが、有効/無効で 5 ターン・10 ターンの出力がバイト一致、共有プレフィックスの 2 会話を交互実行しても混線なし(`reuse 1007/1049`)、streaming も非 streaming と同一だった。
  - 原因: サンプリング。クライアントは思考を有効にすると `temperature: 1` を必ず付ける仕様で、一方 denno は thinking のマップ `enabled` を off 扱いにする(s53c)。**思考は切られているのに T=1 だけが残る**。実セッション相当(system 11,949 文字・tools 26 個で計 7.6k トークン)で再現したところ、T=1.0 では turn 2〜3 で崩壊、T=0.7 と T=0.0 では 4 ターンとも正常だった。
  - 抜け方: サーバー側に上限フラグを置き、クライアントが T=1 を送っても実効値を頭打ちにする。フラグ未指定なら要求値素通しで従来とビット同一(アンカー不変)。

- **分かったこと / 残った課題**
  - 検証: `--repetition-penalty 1.05 --temperature-cap 0.7` で起動し、T=1.0 を送る 7.6k のプロンプトを再送 → 崩壊せず、T=0.7 を直接指定した場合と同じ応答。フラグ無しの起動では GPU アンカー 3 本が MATCH。
  - 残った課題: クライアント側で非 Anthropic エンドポイントに `temperature: 1` を付けないパッチが本来の修正だが未着手。クライアント更新で消え得るため、サーバー側の上限を先に入れた。

- **次に効いたこと**: 「キャッシュを疑う前に、同じペイロードで条件だけを振って再現する」という切り分け手順が定着した。後の 27B の品質調査(巻3)でも、まず CPU で再現し、次に参照実装と比べる同じ順序を使っている。

---

## 棄却カタログ

当時は妥当に見えたが後で捨てた判断。棄却の理由は、そのときに観測できた事実で書く。

### brief を「思考 ON + 短答誘導文」として実装する(s47 `4e89c11` → s53c `e7246b8` で棄却)
- 採用理由: thinking の意味を変えずに、system 冒頭の誘導文だけで回答に導けると考えた。ハード制御ではないと明記していた。
- 観測: 実測で 9B が「Thinking Process:」のメタ分析を延々と書き、回答に到達しなかった。誘導文は生成を制約していないので、思考が走れば効かない。
- 棄却: brief を「思考無効 + 簡潔指示」に変え、マップ `enabled` も off 扱いへ。既定は実質「思考しない」になった。副作用として、思考は明示的に `full` を送らないと走らなくなった。

### 停止マーカーをチャンク先頭オフセット基準で判定する(s53 `f21a985` で棄却)
- 採用していた形: 供給されたチャンクの先頭からの相対位置でマーカーを探す。
- 観測: 本文中の `system` / `user` で誤停止する(`userland` のような語)。マーカーがトークン境界をまたぐため `u` / `us` / `use` が delta に漏れる。行ラベル判定と prefix 判定で語境界の見方が揃っていなかった。
- 棄却: 送出済みバイト数 `sent_len` を不変条件にした holdback へ。prefix 経路が語境界を見ないという限界は残した(安全側として受け入れ)。

### 非ストリームの切断を対象外とする(切断検知 `e7d3e7a` → s44 `08e7c9b` で棄却)
- 採用理由: 「hyper はレスポンス送信まで接続を観測しない」ので、非ストリームには観測する機会が無いとした。
- 観測: 前提が誤り。応答 future が pending の間も read 側を見ており、切断時に handler future ごと drop される。実際に切断後も `in_flight` が 1 のまま減らない事例が出た。
- 棄却: handler が持つ oneshot の `respond.is_closed()` を切断の間接観測に使い、SSE と同一の sweep に乗せた。

### tools + stream を 400 で拒否する(tool use `751dc10` → streaming tool_use `05e5f8f` で棄却)
- 採用理由: 非ストリーミングの抽出器を delta ストリームへ移植する手段が無く、範囲を絞った。拒否はテスト(`tools_with_stream_is_rejected`)で固定していた。
- 観測: クライアントはツール会話をストリーミングで使えない。SSE アセンブラを入れれば、任意の分割幅で `extract_tool_calls(全文)` の残テキストと一致させられることが単体テストで示せた。
- 棄却: 400 を撤去し、`StreamAssembler` で受理。テストを `tools_with_stream_is_accepted` に置換した。

