ADOPT: yes t347a

# REPORT P347 — sparkDash 用の Prometheus `/metrics` (と `/v1/models` の `owned_by`)

branch: **br347** = int-auto `6c19792cf` (merge: br342) から。br342 は既に int-auto に
入っていた (`6c19792cf merge: br342`) ので、merge は不要だった。ラウンド中も int-auto は
`6c19792cf` のまま (最終確認でも同じ = 追加の merge なし)。

## 0. 結論 (3 行)

1. **`GET /metrics` を serve_flash (本番 8098) に足した**。Prometheus text format 0.0.4、
   `Content-Type: text/plain; version=0.0.4; charset=utf-8`。系列名は sparkDash の
   vLLM reader がそのまま読めるように `vllm:` 名 (prompt/generation_tokens_total、
   num_requests_running/waiting、TTFT / e2e / inter-token / prefill の 4 ヒストグラム、
   prefix_cache_queries/hits、prompt_tokens_cached、kv_cache_usage_perc、
   num_preemptions、cache_config_info{kv_cache_size_tokens})、vLLM に対応名が無いものは
   `denno_` 名 (直近 decode ms/token、直近 prefill tok/s、expert の VRAM ヒット率、
   ディスク読みバイト、VRAM 使用/予算、RAM tier、build SHA)。
2. **速度は変わっていない** (lab t347a、depth8k、PROFILE=1)。decode の中央値は
   base0 16.8 / moff 16.7 / mon 16.7 / base1 16.6 ms/tok、a1/a2 の生値も 1 桁まで
   一致 (mon 2.14/2.13 s = moff 2.14/2.13 s)。prefill は round 2 で 4 COND とも
   c2000 cold **2.52 s**。14k cold だけ mon 8.19 s / moff 6.49 s と開いたが、
   台帳では launch 数 (10690) と H2D/D2H (4445.4/3053.6 MB) が**完全に同一**で、
   差は `other` 段の**帰属先の無い待ち** (825.3 vs 281.2 ms、sub 段の和は 191.5 /
   192.6 で同じ) = **PLE のページキャッシュ待ち**。同じ env の base1 (mon と同構成) は
   **6.44 s** で moff と同水準。md5 は 4 COND 全 SAME、errors 0。
3. **ゲートは両方向で発火した**。8 本のサーバログの起動行が
   `[flash] metrics: on (GET /metrics, DENNO_METRICS=既定)` ×6 と
   `[flash] metrics: off (GET /metrics, DENNO_METRICS=0)` ×2 (moff の 2 ラウンド)。
   `DENNO_METRICS=0` で経路ごと消える (404)。`/v1/models` の `owned_by` は
   `DENNO_MODELS_OWNED_BY` (既定 on)。

## 1. 前提 (task と REPORT_P342 の読み直し)

- REPORT_P342 の base は decode **16.6-16.8 ms/tok** (depth8k、同じ baseline COND)。
  P342 の結論は「mix のホスト往復が壁で、残る削り代はほぼホスト仕事」。今回は
  **速度のタスクではない**ので、同じ COND で数字が動いていないことだけを見る。
  今回の base0/base1 は 16.8 / 16.6 ms/tok で P342 と同じ帯 (差は lab の run 間 drift)。
- 前回のタスクで「採らなかった/効かなかった」ものは 1 つも触っていない。新しい
  カーネルも .cu/.ptx も足していない (PTX= は使っていない)。
- 本番 8098 は `denno serve-flash` (`Commands::ServeFlash`、`--port` の既定が 8098)。
  `denno serve` (denno-server / axum) の既定は 8080 で、**8098 は serve_flash が握る**。
  よって今回の変更は serve_flash だけで、denno-server には入れていない (必要になったら
  別タスク。axum 側は計器を持っていないので同じ atomics を渡す形になる)。

## 2. 変えたこと

### 2.1 `crates/denno-cli/src/serve_metrics.rs` (新規)

- 計器は `AtomicU64` だけ。**decode / prefill のホットループはロックも確保もしない**。
  更新点は 2 つだけ: 最初の 1 トークン (`note_first_token` = TTFT を 1 サンプル) と
  リクエスト終端 (`record_request` = カウンタ / ヒストグラム / ゲージ)。
- ヒストグラムは非累積のバケット配列 + sum(ns) + count。描画時に累積へ直し、
  `le="+Inf"` は `_count` と必ず同じ値になる。
- ゲージ (VRAM 使用/予算、RAM tier、ディスク読み) は**リクエスト終端で 1 回サンプル**
  して atomic に置く。だから `render()` はロックを取らない = 読み取りスレッドから
  生成中に呼んでも生成を止めない。
- `denno_expert_disk_read_bytes_total` は `/proc/self/io` の `read_bytes`
  (プロセスのストレージ読みの累積。expert の重みが主。ページキャッシュに載っている
  ぶんは増えない)。1 リクエスト終端と起動後に `fetch_max` で拾う。
  expert_ram tier の `bytes_read` を使う案は、RAM tier が既定 off だと常に 0 に
  なるので採らなかった。
- `denno_build_info{build_sha,version}` — `build.rs` が `git rev-parse --short=12 HEAD`
  を `DENNO_BUILD_SHA` として渡す (git が無ければ "unknown")。
- ゲート: `DENNO_METRICS` (既定 on。`0`/`false`/`no` で off)、
  `DENNO_MODELS_OWNED_BY` (既定 on)。

### 2.2 `crates/denno-cli/src/serve_flash.rs`

- `handle_req` に `GET /metrics`。加えて **`spawn_acceptor` の読み取りスレッド**でも
  `/metrics` を即答する (`POST /v1/abort` と同じ場所)。本番は逐次処理なので、
  主スレッドで受けると decode 20 s の間スクレイプが待たされる。計器が atomic だけ
  なので、読み取りスレッドから答えてよい。
- `generate()`:
  - 入口で `RunningGuard` (`num_requests_running`) と `t_req`。
  - 最初の 1 トークンで `t_first` と `note_first_token` (分岐 1 回)。
  - decode ループの外 (リクエスト終端) で usage 由来のカウンタ、e2e / inter-token /
    prefill のヒストグラム、expert のヒット率・tier ゲージ・ディスク読みを 1 回。
    ゲート off なら `metrics::enabled()` の atomic load 1 回で全部飛ぶ。
- `get_static`: `/v1/models` の要素に `"owned_by": "denno"`。他のフィールドと
  キー順 (serde_json の BTreeMap = 辞書順) はそのまま。
- `run()`: 起動時に `init_from_env` / `set_model` / 1 回目の tier サンプルと
  `[flash] metrics: on|off` の 1 行。

### 2.3 テスト (`serve_metrics.rs` / `serve_flash_synth.rs`)

- `metrics_text_is_valid_prometheus`: 本文の全行が「コメント (`# HELP` / `# TYPE`)」か
  「名前{ラベル} 値」のどちらかであることをパーサで確かめ、カウンタが 2 回目の記録で
  減らないこと、4 本のヒストグラムのバケットが累積で `le="+Inf"` == `_count` を確かめる。
- `histogram_buckets_are_cumulative`: バケット振り分け (0.05 → le 0.1、0.5 → le 1.0、
  5.0 → +Inf) と sum。
- `gate_default_on_and_off_values`: 既定 on / `0`・`false`・`no` で off。
- `metrics_endpoint_counts_match_reply_usage` (e2e): 合成モデルで `POST /v1/messages` を
  **2 回**流し、`usage.input_tokens` / `output_tokens` の合計が
  `vllm:prompt_tokens_total` / `vllm:generation_tokens_total` の増分と一致すること、
  TTFT/e2e のサンプル数が 2、`denno_last_*` が正、`kv_cache_usage_perc` が 0..1、
  `/v1/models` に `owned_by` があることを確かめる。`GET /metrics` は本物の TCP で
  流して `Content-Type` と `model_name` ラベルも見る。
- 既存の `static_get_responses_are_byte_identical` は `/v1/models` の本文を
  `owned_by` 込みの 143 バイトに更新した (`/health` は不変)。

### 2.4 docs

`docs/api.md` に「GET /metrics (Prometheus)」の節を足し、**1 つの表** (名前 / 型 / 意味)
に全系列を載せた。`/v1/models` の行にも `owned_by` を書いた。

## 3. 数値の扱い

- 数値経路は 1 ビットも変えていない (丸め順の変更なし)。md5 が 4 COND 全 SAME。
- 計測の追加コストは、decode ループでは `Option` の分岐 1 回/トークン、リクエスト
  あたり atomic 加算数個 + 終端のロック 2 個と `/proc/self/io` 1 回 (推定 50 us 未満)。
  lab の decode は 16.6-16.7 ms/tok で moff と区別できない。

## 4. Lab (t347a)

`SCENARIO=depth8k PROFILE=1`、SHA `59068438c`、model `swift15-abl-IQ3_S`、
load_before=2.91、rc=0、errors 0。4 COND × 2 ラウンド交互。
baseline COND = 既定 + `DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000
DENNO_EXPERT_GPU_MIN_HIT=0 DENNO_GPU_WEIGHT_MB=12000`。
`base0` (先頭の捨て) / `moff` (+`DENNO_METRICS=0`) / `mon` (ゲート既定 on) /
`base1` (末尾の繰り返し)。

RESULT.md の表 (decode 中央値 / prefill 中央値):

| cond | decode ms/tok | prefill s | errors | text | md5 |
| --- | --- | --- | --- | --- | --- |
| base0 | 16.8 | 0.59 | 0 | looks ok | a:65ededc3 b:f18c75fc c2000_cold:a91e17da c2000_turn2:5a8f5488 c8000_cold:e2292eb6 c8000_turn2:561c6da8 |
| moff | 16.7 | 0.58 | 0 | looks ok | 同上 (SAME) |
| mon | 16.7 | 0.59 | 0 | looks ok | 同上 (SAME) |
| base1 | 16.6 | 0.58 | 0 | looks ok | 同上 (SAME) |

長文脈 (各 COND の実値。prefill 秒 / decode ms/tok):

| cond | c2000_cold r1 | c2000_cold r2 | c2000_turn2 | c8000_cold (r1) | c8000_turn2 |
| --- | --- | --- | --- | --- | --- |
| base0 (r1 は捨て) | 9.00 / 23 | 2.52 / 16 | 0.44 / 19 | 10.14 / 18.3 | 0.46 / 16.6 |
| moff | 2.53 / 16 | 2.52 / 16 | 0.39 / 15 | 6.49 / 18.6 | 0.46 / 17.5 |
| mon | 3.09 / 18 | 2.52 / 16 | 0.39 / 16 | 8.19 / 16.3 | 0.46 / 15.2 |
| base1 | 2.52 / 16 | 2.52 / 16 | 0.38 / 15 | 6.44 / 16.2 | 0.45 / 15.2 |

(c8000 は 14k の cold。depth8k の仕様で round 1 にしか出ない。)


読み方:

- 短い decode は `moff` と `mon` が**同じ 0.1 ms 刻みの値**。a1 の生値も
  moff 2.14/2.13 s、mon 2.14/2.13 s。→ 計測のコストは lab の分解能以下。
- c2000 cold は round 2 で 4 COND とも **2.52 s**。round 1 のばらつきは既知の
  ページキャッシュ (先頭 2-4 COND)。
- 14k cold の `mon` 8.19 s は**私の変更ではない**。証拠は同じログの台帳:
  `[q4ledger-pf] tok=14018` の launches=10690・h2d=4445.4 MB・d2h=3053.6 MB が
  moff / mon / base1 で**完全一致**、差は `other` 段の wall だけ
  (moff 281.2 / mon 825.3 / base1 279.7 ms) で、その sub 段の和は 3 本とも
  191.5 / 192.6 / 193.2 ms と同一 (ple 141-143 ms も同一)。帰属先の無い待ち =
  PLE の pread のページキャッシュ待ち。同じ env の `base1` が 6.44 s で moff と
  同水準なのがその証拠。
- md5 は 4 COND 全 SAME、ERRORS 0。本文も同一
  (`TXT ... A **B-tree** (specifically, the **B+ tree) is the fundamental data structure
  used by most modern relational databases ...`、`TXTB ... Rust の「所有権（Ownership）と
  ライフタイム（Lifetimes）は、Rust の最も重要な概念であり ...`)。文字化け・空応答なし。

### ゲートの発火 (ログ)

`done/t347a/*.log` の起動行:

```
[flash] metrics: on (GET /metrics, DENNO_METRICS=既定)     ← base0_1 base0_2 mon_1 mon_2 base1_1 base1_2
[flash] metrics: off (GET /metrics, DENNO_METRICS=0)       ← moff_1 moff_2
```

= COND の env はサーバに届いていて、`DENNO_METRICS=0` で本当に経路が消えている。

## 5. `/metrics` の実物

ラボは `/metrics` を curl しない (RESULT.md にも生ログにも無い) ので、**同じコードの
実 TCP 応答**を e2e テスト (`metrics_endpoint_counts_match_reply_usage`、合成モデル) から
採った。本番での撮り方は:

```
curl -s http://127.0.0.1:8098/metrics
```

応答ヘッダ:

```
HTTP/1.1 200 OK
Content-Type: text/plain; version=0.0.4; charset=utf-8
Access-Control-Allow-Origin: *
Connection: close
Content-Length: 8526
```

本文 (2 リクエスト後の値の部分。`model_name` は合成モデルの "synth"):

```
vllm:prompt_tokens_total{model_name="synth"} 48
vllm:generation_tokens_total{model_name="synth"} 8
vllm:request_success_total{model_name="synth"} 2
vllm:num_requests_running{model_name="synth"} 0
vllm:num_requests_waiting{model_name="synth"} 0
vllm:time_to_first_token_seconds_bucket{model_name="synth",le="0.05"} 2
...
vllm:time_to_first_token_seconds_bucket{model_name="synth",le="+Inf"} 2
vllm:time_to_first_token_seconds_sum{model_name="synth"} 0.007831
vllm:time_to_first_token_seconds_count{model_name="synth"} 2
vllm:e2e_request_latency_seconds_count{model_name="synth"} 2
vllm:inter_token_latency_seconds_count{model_name="synth"} 2
vllm:request_prefill_time_seconds_count{model_name="synth"} 2
vllm:prefix_cache_queries_total{model_name="synth"} 48
vllm:prefix_cache_hits_total{model_name="synth"} 0
vllm:prompt_tokens_cached_total{model_name="synth"} 0
vllm:num_preemptions_total{model_name="synth"} 0
vllm:kv_cache_usage_perc{model_name="synth"} 0.242188
vllm:cache_config_info{model_name="synth",kv_cache_size_tokens="128"} 1
denno_last_decode_ms_per_token{model_name="synth"} 0.286687
denno_last_prefill_tokens_per_second{model_name="synth"} 5437.361196
denno_expert_vram_hit_rate{model_name="synth"} 0
denno_expert_disk_read_bytes_total{model_name="synth"} 0
denno_expert_vram_used_bytes{model_name="synth"} 0
denno_expert_vram_budget_bytes{model_name="synth"} 0
denno_expert_ram_tier_bytes{model_name="synth"} 0
denno_expert_ram_tier_experts{model_name="synth"} 0
denno_build_info{build_sha="6c19792cf1d8",version="0.1.0"} 1
```

(合成モデルなので expert の 4 系列は 0。実機では `DENNO_EXPERT_IQ_GPU=1` の run で
ヒット率・VRAM 使用/予算が入る。`/v1/models` は
`{"data":[{"created_at":"2026-10-27T00:00:00Z","display_name":"m","id":"m","object":"model","owned_by":"denno","type":"model"}],"object":"list"}`
を返す = 既存のバイト一致テストで固定。)

## 6. テスト

```
cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  → 1230 passed; 0 failed; 102 ignored
cargo test --release -p denno-engine --features cuda --test kernel_invariants
  → 102 passed; 0 failed
cargo test --release -p denno-cli --features denno-engine/cuda -- --test-threads=1
  → 79 passed; 0 failed; 1 ignored   (metrics の 4 本を含む)
```

`CARGO_TARGET_DIR=/tmp/t347`。

## 7. GPU で確認してほしいこと

1. 実機 8098 で `curl -s http://127.0.0.1:8098/metrics` を 1 回撮り、`model_name` が
   swift15 の名前になっていること、`denno_expert_vram_hit_rate` /
   `denno_expert_vram_used_bytes` / `denno_expert_disk_read_bytes_total` が
   0 でないこと (合成モデルでは 0 だった)。
2. sparkDash から見て、`/v1/models` の `owned_by: "denno"` でエンジンが選ばれ、
   `vllm:` 系列がグラフに出ること。
3. decode 中のスクレイプ (生成中に curl) が詰まらないこと (読み取りスレッドが答える)。
4. `DENNO_METRICS=0` で 404、`DENNO_MODELS_OWNED_BY=0` で `owned_by` が消えること。

## 8. 次の手 (やっていないこと)

- denno-server (axum、8080 の 27B 経路) には入れていない。8098 は serve_flash。
- `num_requests_waiting` は常に 0 (逐次処理で、待ちはソケットの backlog)。
  キュー長が要るなら scheduler 側に計器が要る。
- `inter_token_latency_seconds` は 1 リクエスト 1 サンプル (平均間隔)。vLLM は
  トークンごとに 1 サンプル入れるので、ヒストグラムの形は vLLM と一致しない
  (平均値の分布として読む)。
