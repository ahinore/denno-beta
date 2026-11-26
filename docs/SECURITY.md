# Security & Hardening Guide

denno の HTTP 面 (認証 / CORS / ボディ上限 / レート制限 / 秘密情報の扱い /
待受既定) を 1 か所にまとめる。対象は `crates/` の実装で、記載した
フラグ・環境変数・既定値はすべて実装と一致する (末尾 §11 の検証手順を参照)。

denno は「単一テナントのローカル推論サーバ」を想定した設計で、**TLS 終端・
ユーザ管理・監査ログを持たない**。外向きに公開する場合は必ずリバースプロキシ
(TLS + 認証) の背後に置く。本ドキュメントはその前提で「既定で何が開くか」と
「何を設定すれば閉じるか」を正確に書く。

対象プロセスは 3 つで、セキュリティ面の実装はそれぞれ別系統である。

| プロセス | 実装 | 既定の待受 |
|---|---|---|
| `denno serve` | `crates/denno-server/` (axum) | `127.0.0.1:8080` |
| `denno serve-flash` | `crates/denno-cli/src/serve_flash.rs` (自前 HTTP/1.1) | `127.0.0.1:8098` |
| `denno-view` | `crates/denno-view/` (axum) | `0.0.0.0:8093` |

`serve` と `serve-flash` はループバック既定、`denno-view` だけは全インタフェース
既定である (§9)。

---

## 1. 待受アドレス (localhost 既定)

### 1.1 `denno serve`

- `--host` 既定 `127.0.0.1` (`crates/denno-cli/src/main.rs:232`)
- `--port` 既定 `8080` (`crates/denno-cli/src/main.rs:233`)
- 実装は `format!("{}:{}", cfg.host, cfg.port)` を `TcpListener::bind` に渡すだけ
  (`crates/denno-server/src/lib.rs:1884-1886`)。ホスト名の検証や
  「非ループバックなら警告」は無い。

`--host 0.0.0.0` を指定すると全インタフェースで待ち受ける。その場合、
§2 の認証を必ず有効にする (`--api-key` / `--api-keys` / `DENNO_API_KEYS`)。

### 1.2 `denno serve-flash`

- `--host` / `--port` の既定は設定解決側で `127.0.0.1` / `8098`
  (`crates/denno-cli/src/config.rs:43`, `:49`)。環境変数は
  `DENNO_FLASH_HOST` / `DENNO_FLASH_PORT`。
- 解決順は CLI > `DENNO_*` > 設定ファイル > 既定値 (`--config`)。

### 1.3 `denno-view`

- 待受は `0.0.0.0:{port}` 固定で、ホストを変えるフラグも環境変数も無い
  (`crates/denno-view/src/main.rs:227-229`)。
- ポートは `VIEW_PORT` (既定 `8093`、`crates/denno-view/src/main.rs:145-148`)。
- ループバックに閉じたい場合は、ファイアウォール / コンテナのポート公開で
  制限するか、リバースプロキシ経由でのみ公開する。

---

## 2. 認証 (`denno serve`)

### 2.1 設定

| 指定 | 意味 | 実装 |
|---|---|---|
| `--api-key <KEY>` | 単一キー (カンマ区切りも可) | `crates/denno-cli/src/main.rs:237` |
| `--api-keys <KEYS>` | 複数キー (カンマ区切り) | `crates/denno-cli/src/main.rs:241` |
| `DENNO_API_KEYS` | 複数キー (カンマ区切り) | `crates/denno-server/src/lib.rs:2060` |

実効キー一覧は 3 つの**和**で、空白除去・空要素除去・重複除去の後に
カンマ区切り 1 本へ正規化される (`resolve_api_keys` /
`parse_api_keys`, `crates/denno-server/src/lib.rs:2064-2091`)。
**1 本も無ければ認証は無効**で、全 API が開放される (従来挙動)。

### 2.2 受理するヘッダ

- `x-api-key: <KEY>`
- `Authorization: Bearer  <KEY>` (スキーム名は大小文字無視。
  `bearer_credential`, `crates/denno-server/src/lib.rs:2111-2118`)

両方あるときは、どれか 1 つが一致すれば通る (`check_auth`,
`crates/denno-server/src/lib.rs:2120-2148`)。比較は全キー × 全候補を走査する定数時間比較
(`constant_time_eq`, `crates/denno-server/src/lib.rs:2098-2107`。長さの差も `max(len)` 回の走査で
吸収し、`black_box` で早期打ち切り最適化を防ぐ)。一致しなければ
`401` + `{"type":"error","error":{"type":"authentication_error",
"message":"invalid x-api-key"}}` (`crates/denno-server/src/lib.rs:187-189`)。メッセージは固定で、
送られたキーは応答に含めない。

### 2.3 認証が掛かるエンドポイント

`check_auth` を呼ぶハンドラは次の通り (grep で全件確認できる。

```
grep -rn "check_auth" crates/denno-server/src/*.rs
```

- Anthropic: `POST /v1/messages`, `POST /v1/messages/count_tokens`
- OpenAI: `GET /v1/models`, `GET /v1/models/:id`, `POST /v1/chat/completions`,
  `POST /v1/completions`
- Ollama: `GET /api/tags`, `GET /api/version`, `POST /api/show`, `GET /api/ps`,
  `POST /api/chat`, `POST /api/generate`
- キャッシュ管理: `GET /v1/cache/stats`, `DELETE /v1/cache`
  (`crates/denno-server/src/lib.rs:1730`, `crates/denno-server/src/lib.rs:1763` 付近)

### 2.4 認証が掛からないエンドポイント (意図的)

| パス | 理由 | 実装 |
|---|---|---|
| `GET /health` | liveness。監視系はキーを持たない | `crates/denno-server/src/lib.rs:1989` |
| `GET /ready` | readiness。同上 | `crates/denno-server/src/lib.rs:2000` |
| `GET /metrics` | Prometheus スクレイパ用 | `crates/denno-server/src/lib.rs:1637-1641` |
| `GET /v1/telemetry/ws` | 推論テレメトリの WS。`DENNO_TELEMETRY` が有効なときだけ開く | `crates/denno-server/src/lib.rs:2571-2583` |

`/metrics` はモデル名・リクエスト数・トークン数・レイテンシ分布を返す。
公開範囲に注意すること。`/v1/telemetry/ws` は `DENNO_TELEMETRY` 未設定なら
`503 telemetry disabled` を返し、有効時は認証なしでフレームを流す
(`crates/denno-server/src/lib.rs:2575-2583`)。`DENNO_TELEMETRY` は `0` 以外の任意の値で有効
(`crates/denno-telemetry/src/lib.rs:1616-1626`)。

---

## 3. CORS (`denno serve`)

### 3.1 設定と既定

| 指定 | 意味 |
|---|---|
| `--cors-origin <ORIGINS>` | `*` またはカンマ区切りの許可オリジン (`crates/denno-cli/src/main.rs:282-283`) |
| `DENNO_CORS_ORIGIN` | 同上 (env) (`crates/denno-server/src/lib.rs:1202`) |

優先順は CLI > env。未指定 / 空なら**既定 = ループバックのみ**
(`CorsConfig::default`, `crates/denno-server/src/lib.rs:1233-1237`)。

既定で許可されるオリジンは `http://localhost[:port]` /
`https://localhost[:port]` / `http://127.0.0.1[:port]` /
`https://127.0.0.1[:port]` / `http://[::1][:port]` /
`https://[::1][:port]`。ポートは任意 (`is_loopback_origin`,
`crates/denno-server/src/lib.rs:1261-1283`)。`http`/`https` 以外のスキームは許可しない。

- `*` を指定すると `Access-Control-Allow-Origin: *` を返す
  (`allow_any`, `crates/denno-server/src/lib.rs:1300-1323`, `:1352-1356`)。
- カンマ区切りの一覧を指定すると、**ループバック既定は無効化**され、
  一覧のオリジンだけを許可する (`allow_loopback: false`, `crates/denno-server/src/lib.rs:1322`)。
  ローカルブラウザも使うなら一覧に `http://localhost:5173` 等を明示する。
- オリジンは小文字化 + 末尾スラッシュ除去で比較する (`normalize_origin`,
  `crates/denno-server/src/lib.rs:1255-1257`)。

### 3.2 応答ヘッダ

- プリフライト (`OPTIONS` + `Access-Control-Request-Method`) は CORS 層が
  204 で完結させ、内側へ渡さない (`crates/denno-server/src/lib.rs:1346-1371`)。
- `Access-Control-Allow-Methods: GET, POST, OPTIONS` (`crates/denno-server/src/lib.rs:1205`)。
- `Access-Control-Allow-Headers` は要求された一覧をそのまま返す。
  要求が無ければ `content-type, x-api-key, authorization, anthropic-version,
  x-request-id` (`crates/denno-server/src/lib.rs:1209-1210`, `:1359-1364`)。
- `Access-Control-Max-Age: 86400` (`crates/denno-server/src/lib.rs:1213`)。
- 通常応答は `Access-Control-Allow-Origin` と
  `Access-Control-Expose-Headers: x-request-id, x-denno-ttft-ms,
  x-denno-decode-tps` を返す (`crates/denno-server/src/lib.rs:1216`, `:1381-1382`)。
- 許可一覧モードでは `Vary: Origin` を付ける (`crates/denno-server/src/lib.rs:1366-1368`, `:1383-1385`)。
- `Access-Control-Allow-Credentials` は**返さない** (Cookie 認証を前提にしない)。

プリフライトはレート制限のトークンを消費しない (§5.1)。

### 3.3 `denno serve-flash` の CORS

serve-flash は CORS の許可一覧を持たず、全応答に
`Access-Control-Allow-Origin: *` を固定で付ける
(`crates/denno-cli/src/serve_flash.rs:2427`)。ブラウザから使う分には便利だが、
オリジン制限はできない。ループバック既定で待ち受けることが唯一の防御である。

---

## 4. リクエストボディ上限 (`denno serve`)

| 指定 | 既定 | 意味 |
|---|---|---|
| `--max-body-mb <MB>` | 16 | ボディ上限 (MiB)。CLI は 1 以上 (`crates/denno-cli/src/main.rs:287-288`) |
| `DENNO_MAX_BODY_MB` | 16 | 同上 (env) (`crates/denno-server/src/lib.rs:355`) |

- 実効値は CLI > env > 16 MiB (`resolve_max_body_mb`,
  `crates/denno-server/src/lib.rs:366-370`)。0 は 1 MiB に、4096 MiB 超は
  4096 MiB にクランプする (`MAX_BODY_MB_LIMIT`, `crates/denno-server/src/lib.rs:362`)。
- 2 段で効く:
  1. `Content-Length` が上限を超える要求は、ボディを読む前に 413 にする
     (`body_limit_middleware`, `crates/denno-server/src/lib.rs:396-425`)。
  2. `Content-Length` を信じない / 持たない chunked ボディは
     `DefaultBodyLimit::max` が読み込み中に打ち切る (`crates/denno-server/src/lib.rs:1181`)。
- 413 の本文は他のエラーと同じ JSON 封筒
  `{"type":"error","error":{"type":"invalid_request_error", ...}}`
  (`payload_too_large_response`, `crates/denno-server/src/lib.rs:378-389`)。上限バイト数だけを含み、
  リクエスト内容は含めない。
- `--max-body-mb 0` は clap の range で拒否され、設定経路でも
  `anyhow::bail!` する (`crates/denno-server/src/lib.rs:1877-1879`)。

`denno serve-flash` にはボディ上限が無い。`Content-Length` の値だけを見て
`vec![0u8; content_length]` を確保する
(`crates/denno-cli/src/serve_flash.rs:2406-2414`)。信頼できないネットワークに
公開しないこと。

---

## 5. レート制限・キュー・期限 (`denno serve`)

### 5.1 レート制限 (トークンバケット)

| 変数 | 既定 | 意味 |
|---|---|---|
| `DENNO_RATE_LIMIT_RPM` | 未設定 = 無効 | 1 分あたりの許可数。0 / 不正 = 無効 |
| `DENNO_RATE_LIMIT_BURST` | `max(1, rpm/60)` | バケット容量 (瞬間許可数) |

実装: `crates/denno-server/src/ratelimit.rs`。`RATE_LIMIT_RPM_ENV` /
`RATE_LIMIT_BURST_ENV` は `crates/denno-server/src/ratelimit.rs:27`, `:30`。`RateLimitConfig::from_env`
は起動時に 1 回だけ読む (`crates/denno-server/src/lib.rs:1881`)。既定は無効なので、フラグを付けない
起動の挙動は従来と同一。

- 対象は推論エンドポイントのみ: `/v1/messages`,
  `/v1/messages/count_tokens`, `/v1/chat/completions`, `/v1/completions`,
  `/api/chat`, `/api/generate` (`RATE_LIMIT_INFERENCE_PATHS`,
  `crates/denno-server/src/ratelimit.rs:39-46`)。
- 素通し: `/health`, `/ready`, `/metrics` (`RATE_LIMIT_BYPASS_PATHS`,
  `crates/denno-server/src/ratelimit.rs:49`)。末尾スラッシュ 1 個の差は同一視する (`:54-60`)。
- 拒否は `429` + `Retry-After` (秒、最小 1) + OpenAI 形式の封筒
  `{"error":{"message","type":"rate_limit_error","code":"rate_limit_exceeded"}}`
  (`too_many_requests_response`, `crates/denno-server/src/ratelimit.rs:293-310`)。
- 拒否数は `/metrics` のカウンタに載る (`crates/denno-server/src/lib.rs:1447` → `metrics`)。

### 5.2 キュー深度

| 変数 | 既定 | 上限 |
|---|---|---|
| `--max-queue-depth` (別名 `--max-queue`) | 32 | 65536 |
| `DENNO_MAX_QUEUE_DEPTH` | 32 | 65536 |

解決は CLI > env > 既存フィールド > 32、0 は 1 に切り上げ
(`resolve_max_queue_depth`, `crates/denno-server/src/lib.rs:340-343`)。
キューが満杯のときの応答はプロトコルごとに異なるが、いずれも
`Retry-After` (`QUEUE_RETRY_AFTER_SECS = 1`, `crates/denno-server/src/scheduler.rs:223`) 付き:

| プロトコル | ステータス | 実装 |
|---|---|---|
| Anthropic | 503 `overloaded_error` | `crates/denno-server/src/lib.rs:2435-2445` |
| OpenAI | 429 `rate_limit_exceeded` | `crates/denno-server/src/openai.rs:1174-1184` |
| Ollama | 503 | `crates/denno-server/src/ollama.rs:112-121` |

KV スロット枯渇 (`SlotsBusy`) は **429 `rate_limit_error`**、同時実行上限は
503 `overloaded_error` (`from_admit_failure`, `crates/denno-server/src/lib.rs:219-230`)。

### 5.3 リクエスト期限

| 変数 | 既定 | 上限 |
|---|---|---|
| `--request-timeout-secs` | 0 (無効) | 86400 |
| `DENNO_REQUEST_TIMEOUT_SECS` | 0 (無効) | 86400 |

`resolve_request_timeout_secs` (`crates/denno-server/src/lib.rs:346-350`)。0 は「期限なし」で従来挙動。
期限超過は生成を打ち切り、KV スロットを解放してエラーを返す。

### 5.4 停止 (drain)

| 変数 | 既定 | 上限 |
|---|---|---|
| `DENNO_DRAIN_TIMEOUT_SECS` | 10 | 600 |
| `DENNO_DRAIN_TIMEOUT_MS` | - | 600000 |

優先順は `DENNO_DRAIN_TIMEOUT_SECS` > `DENNO_DRAIN_TIMEOUT_MS` > 10 秒
(`crates/denno-server/src/lifecycle.rs:21-27`, `:98-120`)。SIGTERM / SIGINT で
新規要求を 503 にし、進行中を上限まで待ってから停止する (`crates/denno-server/src/lib.rs:1910-1916`)。

---

## 6. 秘密情報の扱い (redaction)

### 6.1 起動サマリ

`format_startup_summary` は **API キーの値を受け取らない**。認証は
`AuthState` (無効 / 有効 + 本数) だけを持ち、出力は
`auth: enabled (N key(s), values hidden)` になる
(`crates/denno-server/src/startup.rs:15-38`, `:158-165`)。型として値を持たない
ので、整形結果に生キーが混入する経路が存在しない。
回帰テスト: `crates/denno-server/tests/startup_summary_971.rs`
(`api_keys_never_appear_in_any_summary_field`)。

### 6.2 アクセスログ

`log_requests` が出すのは `method` / `path` / `status` / `duration_ms` /
`request_id` だけ (`crates/denno-server/src/logging.rs:193-213`)。

- `path` は `req.uri().path()` で、**クエリ文字列を含めない**
  (`crates/denno-server/src/lib.rs:1460`)。`?api_key=...` のようなクエリに秘密を置いてもログには出ない
  (ただしプロキシのログには残り得るので、クエリにキーを置かないのが正しい)。
- ヘッダ (`x-api-key` / `authorization`) とボディ (プロンプト) はログに書かない。
- `request_id` はクライアントの `x-request-id` を無害化して使う。印字可能 ASCII
  (0x21..=0x7E) だけを残し、`"` と `\` を落とし、128 byte で切る。有効文字が
  残らなければ `req-<base><seq>` を生成する
  (`sanitize_request_id_bytes`, `crates/denno-server/src/lib.rs:1525-1546`; `MAX_REQUEST_ID_LEN`,
  `crates/denno-server/src/lib.rs:1504`; `generate_request_id`, `crates/denno-server/src/lib.rs:1553-1563`)。
- JSON 形式でも同じ 5 フィールドのみ (`--log-format json`, `crates/denno-cli/src/main.rs:273-277`)。

### 6.3 エラー応答

- 認証失敗は固定文言 `invalid x-api-key` で、送信値もキー本数も出さない
  (`crates/denno-server/src/lib.rs:187-189`)。
- `resolve_api_keys` / `parse_api_keys` は値をログに出さない。起動ログは
  本数のみ (`crates/denno-server/src/lib.rs:1846-1856` のコメントと `summarize_auth`)。
- 回帰テスト: `crates/denno-server/tests/auth_951.rs`
  (`responses_never_leak_a_configured_key`)。

### 6.4 秘密を扱うときの注意

- 環境変数でキーを渡す場合、`ps` / `/proc/<pid>/environ` / シェル履歴に残り得る。
  `--api-key` も同様にプロセス引数へ残る。共有ホストでは設定ファイル +
  環境変数の注入 (systemd `EnvironmentFile` 等) を推奨する。
- `DENNO_API_KEYS` はカンマ区切りなので、キー自体にカンマを含められない。
- ログは有界キューで、消費者が詰まると行が**破棄される** (秘密の保護ではなく
  可用性のための設計, `crates/denno-server/src/logging.rs:19-27`)。監査目的の完全なログを期待しない。

---

## 7. 情報露出面 (認証とは独立)

- `GET /metrics` と `GET /health` / `GET /ready` は認証なし (§2.4)。
- `404` / `405` はパスとメソッドをそのまま本文に含める
  (`crates/denno-server/src/lib.rs:1400-1417`)。クライアント由来の文字列なので秘密ではないが、
  パスに秘密を埋めないこと。
- `--log-format json` は機械可読だが、内容は §6.2 の 5 フィールドのみ。
- `denno-view` の `GET /code/src` はサーバ / エンジンのソースを JSON で返す
  (`crates/denno-view/src/main.rs:621-632`)。ビューワを公開しないこと (§9)。

---

## 8. `denno serve-flash` との差分 (要約)

serve-flash は `denno-server` のミドルウェアを使わない別実装である。
grep で確認できる通り、`serve_flash.rs` には `check_auth` / `cors` /
`rate_limit` / `authorization` の実装が無い。

| 機能 | `denno serve` | `denno serve-flash` |
|---|---|---|
| API キー認証 | あり | **なし** |
| CORS 許可一覧 | あり (既定ループバック) | なし (`*` 固定, `crates/denno-cli/src/serve_flash.rs:2427`) |
| ボディ上限 | あり (既定 16 MiB) | **なし** |
| レート制限 | あり (任意) | **なし** |
| 既定待受 | `127.0.0.1:8080` | `127.0.0.1:8098` |

serve-flash を公開ネットワークへ出す場合は、リバースプロキシ側で認証・
ボディ上限・レート制限を必ず掛ける。

---

## 9. `denno-view` の差分 (要約)

- 待受は `0.0.0.0:8093` 固定 (§1.3)。**既定で全インタフェースに開く**。
- 認証・CORS 許可一覧・レート制限は無い (`crates/denno-view/src/main.rs:236-259`
  のルータにレイヤが無い)。ボディ上限は axum の既定 (2 MiB) だけで、
  `denno-server` のような設定可能な上限は無い。
- `POST /chat` は上流 `http://<upstream>/v1/messages` へ中継するが、
  API キーは転送しない (`x-api-key` / `authorization` の参照が無い)。
  上流が認証を要求する構成ではビューワ経由のチャットは 401 になる。
- `/api/*` は上流 REST API をそのまま中継する。
- `/code/src` はサーバ / エンジンのソースを返す (§7)。
- 上流は `VIEW_UPSTREAM` (既定 `ws://127.0.0.1:8080/v1/telemetry/ws`)、
  記録・キャプチャは `VIEW_RECORD` / `VIEW_REPLAY` / `VIEW_CAPTURE`、
  トークナイザは `VIEW_TOKENIZER` で指定する
  (`crates/denno-view/src/main.rs:143-177`)。これらはローカルのファイルパスを
  読むため、ビューワを多人数で共有しないこと。

---

## 10. ハードニング・チェックリスト

1. 外向き公開するなら TLS + 認証付きリバースプロキシの背後に置く。
2. `denno serve`:
   - `--host 0.0.0.0` にするなら `--api-key` / `--api-keys` /
     `DENNO_API_KEYS` を必ず設定する。
   - ブラウザから別オリジンで使うときだけ `--cors-origin` を明示する
     (`*` は最終手段)。
   - 公開するなら `DENNO_RATE_LIMIT_RPM` と `--max-body-mb` を実運用に合わせる。
   - `--request-timeout-secs` を設定して 1 本の長時間生成がキューを占有しない
     ようにする。
3. `denno serve-flash`: ループバック既定を崩さない。公開時はプロキシで
   認証・上限を掛ける (§8)。
4. `denno-view`: `0.0.0.0` 固定なので、ファイアウォール / ポート公開設定で
   到達範囲を絞る (§9)。
5. `DENNO_TELEMETRY` は必要なときだけ有効にする (`/v1/telemetry/ws` は認証なし)。
6. API キーは環境変数 / 設定ファイル経由で渡し、シェル履歴やドキュメントに
   残さない (§6.4)。
7. `/metrics` の公開範囲を確認する (モデル名・トラフィック量が見える)。

---

## 11. 検証手順

### 11.1 フラグ・環境変数の実在確認

```sh
grep -n "DENNO_API_KEYS\|API_KEYS_ENV"            crates/denno-server/src/lib.rs
grep -n "DENNO_CORS_ORIGIN\|CORS_ORIGIN_ENV"      crates/denno-server/src/lib.rs
grep -n "DENNO_MAX_BODY_MB\|MAX_BODY_MB_ENV"      crates/denno-server/src/lib.rs
grep -n "DENNO_RATE_LIMIT_RPM\|DENNO_RATE_LIMIT_BURST" crates/denno-server/src/ratelimit.rs
grep -n "DENNO_MAX_QUEUE_DEPTH\|DENNO_REQUEST_TIMEOUT_SECS" crates/denno-server/src/lib.rs
grep -n "DENNO_DRAIN_TIMEOUT_SECS\|DENNO_DRAIN_TIMEOUT_MS" crates/denno-server/src/lifecycle.rs
grep -n "DENNO_SERVER_METRICS"                    crates/denno-server/src/metrics.rs
grep -n "DENNO_TELEMETRY"                         crates/denno-telemetry/src/lib.rs
grep -n "DENNO_FLASH_HOST\|DENNO_FLASH_PORT"      crates/denno-cli/src/config.rs
grep -n "VIEW_PORT\|VIEW_UPSTREAM"                crates/denno-view/src/main.rs
```

### 11.2 既定値・上限の確認

```sh
grep -n "DEFAULT_MAX_BODY_MB\|MAX_BODY_MB_LIMIT"      crates/denno-server/src/lib.rs
grep -n "DEFAULT_MAX_QUEUE_DEPTH\|MAX_QUEUE_DEPTH_LIMIT" crates/denno-server/src/lib.rs
grep -n "DEFAULT_REQUEST_TIMEOUT_SECS\|MAX_REQUEST_TIMEOUT_SECS" crates/denno-server/src/lib.rs
grep -n "DEFAULT_DRAIN_TIMEOUT_SECS\|MAX_DRAIN_TIMEOUT_SECS" crates/denno-server/src/lifecycle.rs
grep -n "default_burst"                               crates/denno-server/src/ratelimit.rs
```

### 11.3 テストゲート

```sh
cargo test --release -p denno-server
node --test crates/denno-view/tests/
```

`denno-server` の統合テストには認証 (`auth_951`, `auth_edge_1159`)、
CORS (`cors_916`, `cors_edge_1094`)、ボディ上限 (`body_limit_946`,
`body_limit_edge_1154`)、レート制限 (`ratelimit_931`, `ratelimit_edge_1139`)、
起動サマリの秘密非漏洩 (`startup_summary_971`) が含まれる。
