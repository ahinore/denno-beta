# Observability Guide

denno の観測面を 1 か所にまとめる。対象は次の 4 つ。

1. `denno serve` の Prometheus `/metrics`
2. 構造化 JSON アクセスログ (`--log-format json`)
3. `denno doctor` によるプリフライト検査
4. `denno-telemetry` の NDJSON トレース記録とストリーミング再生

記載した系列名・フラグ・環境変数はすべて `crates/` の実装と一致する
(末尾の検証手順を参照)。実装が変わったらこの文書も同時に直す。

---

## 1. Prometheus `/metrics` (`denno serve`)

実装: `crates/denno-server/src/metrics.rs`、ハンドラは
`crates/denno-server/src/lib.rs` の `metrics_endpoint`。

- `GET /metrics`。認証は要求しない (`/health` / `/ready` と同じ。スクレイパは
  API キーを持たない前提)。
- `Content-Type: text/plain; version=0.0.4; charset=utf-8`。
- 記録が無効でも応答は常に 200。記録系の値は凍結されるが、
  `in_flight` / `queued` と打ち切り計数 (スケジューラの累積カウンタ) は
  スクレイプ時点の値を返す。
- 依存クレートは使っていない (`prometheus` crate 非依存)。

`denno serve-flash` にも同名の `/metrics` があるが、実装もゲートも別系統
(`crates/denno-cli/src/serve_metrics.rs`、ゲートは `DENNO_METRICS`、既定 on)。
本節が対象とするのは `denno serve` (axum) の `/metrics` だけである。

### 1.1 有効化 / 無効化

| 変数 | 既定 | 意味 |
|---|---|---|
| `DENNO_SERVER_METRICS` | on | `0` / `false` / `no` のとき記録を止める |

- プロセス起動後に 1 回だけ読む (`OnceLock`)。変更には再起動が要る。
- 無効にすると HTTP カウンタ・トークン・ヒストグラムの更新が止まる。
  `/metrics` 自体は 200 を返し続け、以降の値は凍結される。
- `in_flight` / `queued` / 打ち切り計数はスクレイプ時点でスケジューラから
  読むゲージなので、記録を止めても現在値が出る。

### 1.2 系列一覧

| 系列 | 型 | ラベル | 意味 |
|---|---|---|---|
| `denno_model_info` | gauge | `model` | このプロセスが提供するモデル名。常に 1 |
| `denno_requests_in_flight` | gauge | (なし) | 実行中の生成リクエスト数 |
| `denno_requests_queued` | gauge | (なし) | 受付済みでスケジューラの空きを待つ数 |
| `denno_requests_completed_total` | counter | (なし) | 完走した生成リクエスト数 |
| `denno_requests_aborted_total` | counter | `reason` | 打ち切り数。`reason` は `timeout` / `disconnected` / `error` の 3 値 |
| `denno_prompt_tokens_total` | counter | (なし) | 完走リクエストが処理したプロンプトトークン数 |
| `denno_completion_tokens_total` | counter | (なし) | 完走リクエストが生成したトークン数 |
| `denno_kv_cache_read_tokens_total` | counter | (なし) | KV prefix cache から再利用したプロンプトトークン数 |
| `denno_kv_reuse_ratio` | gauge | (なし) | 累積の KV 再利用率 = 再利用 / プロンプト (プロンプト 0 なら 0) |
| `denno_ttft_seconds` | histogram | `le` | 受付から最初のトークンまでの秒 |
| `denno_tpot_seconds` | histogram | `le` | decode 中の平均トークン間隔 (秒) |
| `denno_http_requests_total` | counter | `endpoint`, `status` | 応答した HTTP リクエスト数 |

補足:

- `denno_requests_aborted_total` は 1 つの `# HELP` / `# TYPE` の下に 3 つの
  ラベル値が並ぶ。3 値とも常に出力される。
- `denno_ttft_seconds` は `usage.input_tokens`、`denno_tpot_seconds` は
  1 トークン以下の生成では観測されない (件数に載らない)。
- `denno_http_requests_total` の `endpoint` は既知ルート名に正規化される。
  未知のパスは `other` に丸める (ラベル基数を増やさない)。
- `/metrics` のスクレイプ自身も `endpoint="/metrics"` として数えられるが、
  記録は応答を返した後なので次のスクレイプから見える。

### 1.3 `endpoint` ラベルの取り得る値

`/health`, `/ready`, `/metrics`, `/v1/models`, `/v1/messages`,
`/v1/messages/count_tokens`, `/v1/chat/completions`, `/v1/completions`,
`/api/tags`, `/api/version`, `/api/chat`, `/api/generate`,
`/v1/telemetry/ws`, `other`。

`status` は 3 桁の HTTP ステータスコード。

### 1.4 ヒストグラムのバケット

| 系列 | `le` の上限 (秒) |
|---|---|
| `denno_ttft_seconds` | 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, `+Inf` |
| `denno_tpot_seconds` | 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, `+Inf` |

バケットは累積で出力される (`_bucket{le="+Inf"}` は `_count` と一致)。
`_sum` は秒。`_bucket` / `_sum` / `_count` は Prometheus のヒストグラム慣習で
`denno_ttft_seconds` / `denno_tpot_seconds` から導出される系列名で、実装が
個別に定義しているわけではない。

### 1.5 スクレイプ設定の例

```yaml
scrape_configs:
  - job_name: denno
    metrics_path: /metrics
    static_configs:
      - targets: ["127.0.0.1:8080"]
```

`denno serve` の既定は `--host 127.0.0.1 --port 8080`。手で確認する場合:

```console
$ curl -s http://127.0.0.1:8080/metrics | head
```

### 1.6 PromQL の例

```promql
# 全体のリクエストレート (req/s)
sum(rate(denno_http_requests_total[5m]))

# エンドポイント別の 4xx/5xx
sum by (endpoint) (rate(denno_http_requests_total{status=~"[45].."}[5m]))

# TTFT の p95 / TPOT の p50
histogram_quantile(0.95, sum by (le) (rate(denno_ttft_seconds_bucket[5m])))
histogram_quantile(0.50, sum by (le) (rate(denno_tpot_seconds_bucket[10m])))

# 生成スループット (出力 tok/s) と KV 再利用率
rate(denno_completion_tokens_total[1m])
denno_kv_reuse_ratio

# 詰まり / 打ち切りの検知
denno_requests_queued > 0
increase(denno_requests_aborted_total{reason="timeout"}[15m]) > 0
```

アラート例:

```yaml
groups:
  - name: denno
    rules:
      - alert: DennoQueueBacklog
        expr: max_over_time(denno_requests_queued[10m]) > 1
        for: 10m
      - alert: DennoTimeoutAborts
        expr: increase(denno_requests_aborted_total{reason="timeout"}[15m]) > 0
```

---

## 2. 構造化 JSON アクセスログ

実装: `crates/denno-server/src/logging.rs` (`access_log_line`,
`set_access_log_json`) と `crates/denno-server/src/lib.rs` の `log_requests`。

### 2.1 有効化

`denno serve --log-format json` でアクセスログだけを 1 リクエスト 1 行の
JSON にする。既定は `text`。値は `text` / `json` の 2 つだけ
(それ以外は引数エラー)。

- 起動・エラーなど他のログ行は `text` のまま。JSON になるのは
  アクセスログ (tracing の target `denno_access`) だけ。
- 環境変数での切り替えは無い。`--log-format` が唯一の入口。
- ログのフィルタは `RUST_LOG` (未設定時は `info`)。`denno serve` は
  非同期ロガー (`NonBlockingStdout`) を使う。

`text` (既定) は 1 リクエストにつき 2 行:

```text
POST /v1/messages rid=req-000000000001
-> 200 rid=req-000000000001
```

`json` は応答後に 1 行:

```json
{"ts_ms":1730000000000,"method":"POST","path":"/v1/messages","status":200,"duration_ms":12.5,"request_id":"req-000000000001"}
```

### 2.2 フィールド

| フィールド | 型 | 意味 |
|---|---|---|
| `ts_ms` | number | 応答時点の UNIX 時刻 (ミリ秒) |
| `method` | string | HTTP メソッド |
| `path` | string | リクエストパス (クエリは含まない) |
| `status` | number | HTTP ステータスコード |
| `duration_ms` | number | 受付から応答までのミリ秒 |
| `request_id` | string | 応答ヘッダ `x-request-id` と同じ値 |

`request_id` はクライアントが送った `x-request-id` を無害化した値、または
サーバが生成した値。ログ行とクライアント側の記録を突き合わせる鍵になる。
すべての応答 (SSE / NDJSON を含む) に `x-request-id` が付く。

### 2.3 非同期ロギングの性質

- ログ行は容量 4096 の有界キューに入り、専用スレッド 1 本が stdout へ書く。
  サーバスレッドは stdout のロックやパイプ書き込みでブロックしない。
- キュー満杯時は行を破棄し、復旧後に次の 1 行で報告する:

  ```text
  denno-logger: N log line(s) dropped (stdout consumer stalled or queue full)
  ```

- 消費者の EPIPE は無視する (行喪失のみ、クラッシュしない)。
- したがってアクセスログはベストエフォート。件数の厳密な会計には
  `denno_http_requests_total` を使う。

### 2.4 集計の例

```console
$ denno serve --log-format json 2>&1 | jq -r 'select(.status >= 500) | [.ts_ms, .path, .status, .request_id] | @tsv'
```

```console
$ denno serve --log-format json 2>&1 \
  | jq -s 'group_by(.path) | map({path: .[0].path, n: length, p95_ms: (map(.duration_ms) | sort | .[(length * 0.95 | floor)])})'
```

---

## 3. `denno doctor` (プリフライト検査)

実装: `crates/denno-cli/src/doctor.rs`、サブコマンド定義は
`crates/denno-cli/src/main.rs`。

GPU を実行しない。CUDA API の初期化も `nvidia-smi` の起動もしない。
ライブラリのファイル存在と `/proc` の読み取りだけを見る (VRAM 実容量は出さない)。

### 3.1 使い方

```console
$ denno doctor
$ denno doctor --dir /path/to/model-dir
$ denno doctor --json --dir /path/to/model-dir
```

| フラグ | 意味 |
|---|---|
| `--dir <DIR>` | 分割 GGUF のあるディレクトリ。省略するとモデル検査をしない |
| `--json` | 機械可読な JSON で stdout に出す (自動化向け) |

### 3.2 終了コード

| コード | 条件 |
|---|---|
| 0 | 正常。`--dir` 省略時、およびシャード検査が OK のとき |
| 1 | `--dir` の検査が NG (シャード欠け / 0 バイト / `.gguf` が 1 本も無い)。レポートは先に stdout へ出る |
| 2 | 引数エラー (clap の既定) |

`--json` でもレポートは NG 時にも出力される。終了コードだけで判定せず、
`.model.ok` も見ると原因を特定しやすい。

### 3.3 検査内容

- CPU: `arch`、論理コア数、物理コア数 (`/proc/cpuinfo` の physical id × core id)、
  実行時 SIMD (x86_64 は CPUID、それ以外はビルド時 `target_feature`)。
- メモリ: `/proc/meminfo` の `MemTotal` / `MemAvailable`。
- GPU: `libcuda.so.1` と `libze_loader.so.1` / `libze_intel_gpu.so.1` の
  **存在**、`/proc/driver/nvidia/version`、`/proc/driver/nvidia/gpus/*/information`
  の Model 行。
- モデル (`--dir` 指定時): `.gguf` を列挙し、シャード命名
  `<prefix>-NNNNN-of-MMMMM.gguf` から期待数を読む。
  - `missing_shards`: 期待されるが無い連番 (`"00002"` 形式)
  - `zero_byte_shards`: サイズ 0 のファイル (中断ダウンロードの疑い)
  - `expected_shards`: 分割命名でない単一 GGUF なら `null` (欠け判定をしない)
  - `ok`: 1 本以上あり、欠けも 0 バイトも無い
- 推奨: 検出結果に応じた `denno serve-flash` のフラグと環境変数。

### 3.4 JSON の形

トップレベルは `cpu` / `memory` / `gpu` / `model` / `recommendations`。

```json
{
  "cpu": {"arch": "x86_64", "logical_cores": 16, "physical_cores": 8,
          "simd": ["sse4.2", "avx", "avx2", "fma"]},
  "memory": {"total_bytes": 0, "available_bytes": 0},
  "gpu": {"cuda_driver": null, "nvidia_kernel_driver": null, "gpu_devices": [],
          "level_zero_loader": null, "level_zero_intel_driver": null},
  "model": {"dir": "/path/to/model-dir", "shards": [], "expected_shards": 3,
            "missing_shards": ["00002"], "zero_byte_shards": [],
            "total_bytes": 0, "ok": false},
  "recommendations": ["..."]
}
```

`null` は「読めなかった / 見つからなかった」を意味する。0 を事実として
出さない項目は `null` のままになる。

### 3.5 CI / 運用の例

```console
# モデルの欠けだけで判定する (レポートは捨てる)
$ denno doctor --json --dir /path/to/model-dir >/dev/null

# 原因をログに残しつつ判定する
$ denno doctor --json --dir /path/to/model-dir > doctor.json || {
    jq -r '.model.missing_shards[], .model.zero_byte_shards[]' doctor.json
    exit 1
  }

# ノードの能力を記録する (GPU は実行しない)
$ denno doctor --json | jq '{cpu: .cpu, memory: .memory, gpu: .gpu}'
```

---

## 4. `denno-telemetry` の NDJSON トレース

実装: `crates/denno-telemetry/src/lib.rs`。ワイヤ形式の仕様は
`docs/TELEMETRY_PROTOCOL.md`。

### 4.1 有効化

| 変数 | 既定 | 意味 |
|---|---|---|
| `DENNO_TELEMETRY` | 経路で異なる | `denno serve` は未設定なら無効 (`0` 以外で有効)。`denno serve-flash` は既定で有効 (`0` で無効) |
| `DENNO_TELEMETRY_ARCH` | off | `1` / `true` / `yes` で RouterTopK / ExpertSkip 系を送る |
| `DENNO_TELEMETRY_ACT` | off | `1` / `true` / `yes` で LayerAct / RoleAct を送る |
| `DENNO_TELEMETRY_ACT_TOPK` | 64 | LayerAct の上位 K |
| `DENNO_TELEMETRY_ACT_DBG` | off | 1 フレームあたりの所要 ns と送出バイトを stderr に出す |
| `DENNO_TELEMETRY_TOPK` | off | sampler の TopK / エントロピーを送る |
| `DENNO_TELEM_EXPERT_SKIP` | off | ExpertSkip を送る (`DENNO_TELEMETRY_ARCH` も要る) |

- 送出は bounded channel への `try_send` のみ。満杯時は捨てて `Gap` フレームで
  欠落数を明示する。推論は待たない。
- 無効時はチャネルもスレッドも作らない。

### 4.2 ストリームの受け取り

`GET /v1/telemetry/ws` (WebSocket) がバイナリフレームを流す。1 メッセージ =
1 フレーム (フレーム自体に長さフィールドは無い)。接続直後に `ServerInfo` が
1 枚届く。テレメトリ無効時は 503。

`denno serve` は未設定なら無効なので、`DENNO_TELEMETRY=1` (または `0` 以外) を
付けて起動する。

### 4.3 NDJSON への記録

NDJSON は 1 行 = 1 フレーム。キー順は固定で決定的、数値は 10 進整数
(u64 の ns も f64 を経由しない)。書き出しはライブラリ API のみで、
CLI サブコマンドは無い。

```rust
use denno_telemetry::{write_frame_ndjson, Frame};
use std::io::{BufWriter, Write};

// ws_msg: WebSocket の 1 バイナリメッセージ (Vec<u8>)
let mut out = BufWriter::new(std::fs::File::create("trace.ndjson")?);
// let frame = Frame::decode(&ws_msg).map_err(...)?;
// write_frame_ndjson(&mut out, &frame)?;
out.flush()?;
```

- `Frame::decode(&[u8]) -> Result<Frame, String>` は magic / version / 長さを
  検査し、切り詰め入力でも panic しない。
- まとめて書くなら `write_frames_ndjson(&mut w, &frames)` (書いた行数を返す)。
- `frame_to_ndjson(&frame)` は改行なしの 1 行文字列を返す。

### 4.4 ストリーミング再生

`NdjsonFrames` は `BufRead` から 1 行ずつ読むイテレータ。壊れた行は
`Err(NdjsonError)` として 1 個返して読み進む (行番号は 1 始まり)。
空行・空白だけの行は読み飛ばす。

```rust
use denno_telemetry::NdjsonFrames;
use std::io::BufReader;

let f = std::fs::File::open("trace.ndjson")?;
for item in NdjsonFrames::new(BufReader::new(f)) {
    match item {
        Ok(frame) => {
            // ビューアへ流すなら WS バイナリメッセージとして frame.encode() を送る
        }
        Err(e) => eprintln!("skip {e}"), // "line 12: ..." 形式
    }
}
```

- 一括読みなら `read_frames_ndjson(reader) -> (Vec<Frame>, Vec<NdjsonError>)`。
- 記録時と同じ `Frame` 型なので、`frame.encode()` でバイナリに戻して
  `/v1/telemetry/ws` と同じ形式で再生できる。
- `kind` 文字列と kind バイトの対応は `ndjson_kind_name` /
  `ndjson_kind_from_name`。未知の kind は `None`。

### 4.5 NDJSON の `kind` 一覧

| kind | 値 | NDJSON の `kind` |
|---|---|---|
| `SERVER_INFO` | 1 | `server_info` |
| `SESSION_ADMIT` | 2 | `session_admit` |
| `SESSION_DONE` | 3 | `session_done` |
| `PREFILL` | 4 | `prefill` |
| `TOKEN` | 5 | `token` |
| `STEP` | 6 | `step` |
| `OP_SUMMARY` | 7 | `op_summary` |
| `KV_USAGE` | 8 | `kv_usage` |
| `GAP` | 9 | `gap` |
| `TOP_K` | 10 | `top_k` |
| `LAYER_RUN` | 11 | `layer_run` |
| `LAYER_ACT` | 12 | `layer_act` |
| `ROLE_ACT` | 13 | `role_act` |
| `EXPERT_RESIDENCY` | 14 | `expert_residency` |
| `TIER_STATS` | 15 | `tier_stats` |
| `ROUTER_TOPK` | 16 | `router_topk` |
| `EXPERT_SKIP` | 17 | `expert_skip` |

`SESSION_DONE` の `reason` は `0` END_TURN / `1` MAX_TOKENS /
`2` STOP_SEQUENCE / `3` ERROR。

```json
{"kind":"token","session":1,"token":42,"pos":7}
{"kind":"layer_run","layer":3,"ops":[[4,1200],[5,900]]}
```

### 4.6 欠落の扱い

`GAP` はチャネル満杯・購読遅延で落ちたフレーム数 `lost` を持つ。NDJSON に
記録していても、ストリーム自体で落ちた分は復元できない。解析時は `gap` 行の
`lost` を合計し、欠測区間として扱う。

---

## 5. 検証手順

文書内の名前が実装に存在することを機械的に確認する。

```console
# メトリクス系列 (ヒストグラムの導出サフィックスは基底名に畳む)
$ grep -oE 'denno_[a-z0-9]+(_[a-z0-9]+)*' docs/OBSERVABILITY.md \
  | sed -E 's/_(bucket|sum|count)$//' | sort -u \
  | while read m; do
      case "$m" in denno_access|denno_telemetry) continue;; esac
      grep -rq "$m" crates/denno-server/src/metrics.rs || echo "MISSING $m"
    done

# 環境変数
$ grep -oE 'DENNO_[A-Z0-9_]+' docs/OBSERVABILITY.md | sort -u \
  | while read e; do grep -rq "$e" crates/ || echo "MISSING $e"; done

# CLI フラグ (serve の --log-format / doctor の --dir / --json)
$ grep -n 'log_format\|dir: Option<PathBuf>\|json: bool' crates/denno-cli/src/main.rs

# doctor の終了コード
$ denno doctor; echo "exit=$?"
$ denno doctor --dir /path/to/model-dir; echo "exit=$?"
```

`MISSING` が 1 件も出ないこと。系列名は `metrics.rs` の `metric(` /
`hist(` の引数、環境変数は文字列リテラルとして実在する。
`denno_access` は tracing の target 名、`denno_telemetry` はクレート名で、
メトリクス系列ではないため除外している。
