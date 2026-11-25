# Telemetry Wire Protocol (v1)

`denno-telemetry` のバイナリ・ワイヤ形式の仕様。対象は
`crates/denno-telemetry/src/lib.rs` の `Frame` / `Frame::encode` / `Frame::decode`。
実装は std のみに依存し、エンジン・サーバー・ビューアが同じ形式を共有する。

バイト列は `crates/denno-telemetry/tests/wire_golden_994.rs` の golden テストで
固定している。`encode` の出力を 1 バイトでも変えるとテストが落ちる。

## 1. 封筒 (フレーム共通ヘッダ)

```text
0        4        5        6
+--------+--------+--------+------------------------
| "DNTL" | ver=1  | kind   | payload ...
+--------+--------+--------+
  4 bytes  1 byte   1 byte
```

| オフセット | 長さ | 内容 |
|---|---|---|
| 0 | 4 | magic `DNTL` (0x44 0x4E 0x54 0x4C) |
| 4 | 1 | version (`VERSION = 1`) |
| 5 | 1 | kind バイト (§2) |
| 6 | 可変 | payload (kind ごと、§3) |

1 メッセージ = 1 フレーム。フレームに長さフィールドは無い。ストリームで読む
場合は、トランスポート側のメッセージ境界 (WebSocket の 1 メッセージ等) で
区切る。NDJSON (§6) は 1 行 = 1 フレーム。

## 2. 型とエンディアン

| 記号 | 表現 | 備考 |
|---|---|---|
| `varint` | LEB128 (unsigned, little-endian base-128) | 7 ビットずつ下位から。MSB が継続フラグ。`u32` / `u64` で同じ表現 |
| `u8` | 生 1 バイト | kind / op id / reason / busy |
| `u16` | 2 バイト little-endian | f16 ビット、expert index |
| `bool` | 生 1 バイト | encode は 0 / 1。decode は 0 以外を `true` |
| `str` | `varint` 長 + UTF-8 生バイト | 長さは バイト数 (文字数ではない) |
| `f16` | `u16` の little-endian | IEEE 754 binary16 のビット。丸めは round-to-nearest-even |

- **バイトオーダー**: 固定幅 (`u16`) は little-endian。varint は定義上
  バイト順が一意。
- **f16 変換**: `f32_to_f16_bits` / `f16_bits_to_f32` は自前実装 (RNE)。
  NaN は quiet bit を立てた `0x7E00` 系、overflow は Inf に丸める。
- **確率・正規化値**はすべて f16 ビットで送る (0.0〜1.0 の範囲が前提)。
  受信側は `f16_bits_to_f32` で復元する。

### varint の例

| 値 | バイト列 |
|---|---|
| 0 | `00` |
| 127 | `7F` |
| 128 | `80 01` |
| 300 | `AC 02` |
| 16 383 | `FF 7F` |
| 16 384 | `80 80 01` |
| `u32::MAX` | `FF FF FF FF 0F` |

## 3. kind と payload

kind 番号はブランチを跨いで固定。**詰め直し・再利用を禁止**する (§5)。

| kind | 値 | 名前 (NDJSON) | payload |
|---|---|---|---|
| `SERVER_INFO` | 1 | `server_info` | §3.1 |
| `SESSION_ADMIT` | 2 | `session_admit` | §3.2 |
| `SESSION_DONE` | 3 | `session_done` | §3.3 |
| `PREFILL` | 4 | `prefill` | §3.4 |
| `TOKEN` | 5 | `token` | §3.5 |
| `STEP` | 6 | `step` | §3.6 |
| `OP_SUMMARY` | 7 | `op_summary` | §3.7 |
| `KV_USAGE` | 8 | `kv_usage` | §3.8 |
| `GAP` | 9 | `gap` | §3.9 |
| `TOP_K` | 10 | `top_k` | §3.10 |
| `LAYER_RUN` | 11 | `layer_run` | §3.11 |
| `LAYER_ACT` | 12 | `layer_act` | §3.12 |
| `ROLE_ACT` | 13 | `role_act` | §3.13 |
| `EXPERT_RESIDENCY` | 14 | `expert_residency` | §3.14 |
| `TIER_STATS` | 15 | `tier_stats` | §3.15 |
| `ROUTER_TOPK` | 16 | `router_topk` | §3.16 |
| `EXPERT_SKIP` | 17 | `expert_skip` | §3.17 |

13〜15 は `arch_telemetry` 側の RoleAct / ExpertResidency / TierStats が使う。
16 は RouterTopK、17 は ExpertSkip。

`done_reason` (SESSION_DONE の `reason`) は次の u8:

| 値 | 名前 |
|---|---|
| 0 | `END_TURN` |
| 1 | `MAX_TOKENS` |
| 2 | `STOP_SEQUENCE` |
| 3 | `ERROR` |

### 3.1 SERVER_INFO (1)

WS 接続時のサーバー情報。

| # | フィールド | 型 |
|---|---|---|
| 1 | `model` | str |
| 2 | `model_id` | str |
| 3 | `quant` | str |
| 4 | `slots` | varint u32 |
| 5 | `context` | varint u32 |
| 6 | `layers` | varint u32 |

`model` は表示名、`model_id` はリポジトリ ID、`quant` は量子化名 (例 `fp32`)。

### 3.2 SESSION_ADMIT (2)

| # | フィールド | 型 |
|---|---|---|
| 1 | `session` | varint u32 |
| 2 | `slot` | varint u32 |
| 3 | `prompt_tokens` | varint u32 |

### 3.3 SESSION_DONE (3)

| # | フィールド | 型 |
|---|---|---|
| 1 | `session` | varint u32 |
| 2 | `reason` | u8 (`done_reason`) |
| 3 | `output_tokens` | varint u32 |

### 3.4 PREFILL (4)

| # | フィールド | 型 |
|---|---|---|
| 1 | `session` | varint u32 |
| 2 | `prompt_tokens` | varint u32 |
| 3 | `ns` | varint u64 |
| 4 | `count` | varint u32 |
| 5 | `tokens[count]` | varint u32 各 |

`ns` は prefill 1 forward の所要ナノ秒。

### 3.5 TOKEN (5)

| # | フィールド | 型 |
|---|---|---|
| 1 | `session` | varint u32 |
| 2 | `token` | varint u32 |
| 3 | `pos` | varint u32 |

### 3.6 STEP (6)

| # | フィールド | 型 |
|---|---|---|
| 1 | `width` | varint u32 |
| 2 | `ns` | varint u64 |

バッチ decode の forward 1 回分。`width` はバッチ幅。

### 3.7 OP_SUMMARY (7)

| # | フィールド | 型 |
|---|---|---|
| 1 | `count` | varint u32 |
| 2 | `count` × (`op`: u8, `ns`: varint u64) | |

`op` は `denno_engine::prof` の op id (0..N_OPS, `N_OPS = 32`)。0 ns の op は
encode 前に省略される (`take_ops`)。

### 3.8 KV_USAGE (8)

| # | フィールド | 型 |
|---|---|---|
| 1 | `count` | varint u32 |
| 2 | `busy[count]` | u8 (0/1) |
| 3 | `t_len[count]` | varint u32 各 |

`busy` と `t_len` は同じ本数。encode は `min(len)` 本に切り揃える (§4)。

### 3.9 GAP (9)

| # | フィールド | 型 |
|---|---|---|
| 1 | `lost` | varint u32 |

チャネル満杯・購読遅延で落ちたフレーム数。落ちたフレームの代わりに 1 枚送る。

### 3.10 TOP_K (10)

| # | フィールド | 型 |
|---|---|---|
| 1 | `session` | varint u32 |
| 2 | `chosen` | varint u32 |
| 3 | `entropy_f16` | u16 LE (f16 ビット、エントロピー nat) |
| 4 | `count` | varint u32 |
| 5 | `count` × (`id`: varint u32, `p`: u16 LE f16) | |

`chosen` はサンプルされたトークン id。`entries` は logit 降順 (同点は id 昇順)。

### 3.11 LAYER_RUN (11)

| # | フィールド | 型 |
|---|---|---|
| 1 | `layer` | varint u32 (u16 に切り詰め) |
| 2 | `count` | varint u32 |
| 3 | `count` × (`op`: u8, `ns`: varint u64) | |

1 層の forward 内で実行された op 列 (実行順)。`op` は u8、`ns` は所要ナノ秒。

### 3.12 LAYER_ACT (12)

| # | フィールド | 型 |
|---|---|---|
| 1 | `layer` | varint u32 (u16 に切り詰め) |
| 2 | `count` | varint u32 |
| 3 | `idx[count]` | varint u32 各 (昇順) |
| 4 | `val[count]` | u16 LE f16 各 (0..1) |

`idx` は層出力 `|x|` の上位 K 個 (K = `DENNO_TELEMETRY_ACT_TOPK`、既定 64) を
**index 昇順**に並べたもの。`val` は対応する `|x|` を層内最大で割った値。
`|x|` がちょうど 0 の要素は送らない。`idx` と `val` は同じ本数 (§4)。
トークン位置はフレームに無く、直前の TOKEN / PREFILL と並び順で対応する。

### 3.13 ROLE_ACT (13)

| # | フィールド | 型 |
|---|---|---|
| 1 | `layer` | varint u32 (u16 に切り詰め) |
| 2 | `attn` | u16 LE f16 |
| 3 | `gdn` | u16 LE f16 |
| 4 | `router` | u16 LE f16 |
| 5 | `shex` | u16 LE f16 |
| 6 | `ple` | u16 LE f16 |

5 役割の値を **層内最大** で割った 0..1。順序は固定
(`attn, gdn, router, shex, ple`)。全部 0 / 非有限は 5 値とも 0。

### 3.14 EXPERT_RESIDENCY (14)

| # | フィールド | 型 |
|---|---|---|
| 1 | `layer` | varint u32 (u16 に切り詰め) |
| 2 | `count` | varint u32 |
| 3 | `count` × `expert` | varint u32 (u16 上限) |

`in_vram` は昇順・重複なしで最大 `RESIDENCY_MAX = 512` 本。decode は
u16 を超える index を `Err` にする。

### 3.15 TIER_STATS (15)

| # | フィールド | 型 |
|---|---|---|
| 1 | `vram_used_mb` | varint u32 |
| 2 | `vram_total_mb` | varint u32 |
| 3 | `ram_experts` | varint u32 |
| 4 | `ssd_reads` | varint u32 |

### 3.16 ROUTER_TOPK (16)

| # | フィールド | 型 |
|---|---|---|
| 1 | `layer` | varint u32 (u16 に切り詰め) |
| 2 | `count` | varint u32 |
| 3 | `experts[count]` | varint u32 各 (u16 上限) |
| 4 | `weights[count]` | u16 LE f16 各 |

`experts` は **gate 重みの降順** (同値は index 昇順)。`weights` は
softmax → top-k 再正規化後の値の f16。実形 (層 47 / top-10 /
index ≥ 128) は 6 + 1 + 1 + 20 + 20 = **48 バイト**。
`experts` と `weights` は同じ本数 (§4)。

### 3.17 EXPERT_SKIP (17)

| # | フィールド | 型 |
|---|---|---|
| 1 | `layer` | varint u32 (u16 に切り詰め) |
| 2 | `count` | varint u32 |
| 3 | `count` × `expert` | varint u32 (u16 上限) |

ルータが選びたかったが介入で無効化されていた expert (index 昇順)。空の
`skip` はフレーム自体を作らない (`expert_skip_frame` が `None`)。

## 4. 個数と長さの扱い

### count の上限

decode は `count` を読んだ直後に、残りバイト数と 1 要素の最小サイズから
上限を検査する (`Cursor::count(min_elem)`)。過大な count で巨大な `Vec` を
確保しない。

| kind | 1 要素の最小バイト |
|---|---|
| PREFILL | 1 (varint) |
| OP_SUMMARY / LAYER_RUN | 2 (u8 + varint) |
| KV_USAGE | 2 (u8 + varint) |
| TOP_K | 3 (varint + 2B) |
| LAYER_ACT | 3 (varint + 2B) |
| ROUTER_TOPK | 3 (varint + 2B) |
| EXPERT_RESIDENCY / EXPERT_SKIP | 1 (varint) |

固定の上限値:

| 定数 | 値 | 意味 |
|---|---|---|
| `RESIDENCY_MAX` | 512 | EXPERT_RESIDENCY の最大本数 |
| `N_OPS` | 32 | OP_SUMMARY の op id 空間 |
| `ACT_TOPK_DEFAULT` | 64 | LAYER_ACT の既定 K |
| `CHANNEL_CAP` | 4096 | 送出ハブのリング容量 (フレーム数) |

### 並列配列の長さ食い違い

KV_USAGE (`busy` / `t_lens`)、LAYER_ACT (`idx` / `val`)、ROUTER_TOPK
(`experts` / `weights`) は **短い方の本数に切り揃えて encode** し、decode は
その本数だけ読む。長さの食い違うフレームを作っても encode → decode が
壊れないための規則で、golden テストで固定している。

### 厳密長

decode は payload を読み切った後に `pos == len` を要求する。末尾に余分な
バイトがあるフレームは `Err`。切り詰めも `Err` (panic しない)。

## 5. バージョニングと互換性

- `VERSION = 1`。decode は version が 1 以外なら `Err("unsupported version")`。
- kind 番号は**追記のみ**。既存 kind の番号・意味・payload の並びは変えない。
  削除した kind の番号は再利用しない (穴を許す)。
- payload のフィールド追加・削除・並び替え・型変更は v1 では行わない。
  必要になったら `VERSION` を上げ、v1 の decoder は新 version を拒否する。
- v1 decoder は未知の kind を `Err` にする (黙って読み飛ばさない)。新しい
  フレームを古い読み手に送らないよう、送出側が feature gate で制御する。
- 文字列は UTF-8。不正な UTF-8 は `Err`。
- `layer` は decode で u16 に切り詰める (ワイヤ上は varint u32)。expert index
  は u16 を超えると `Err` (encode 側の `router_topk_frame` /
  `expert_skip_frame` は u16 に飽和させる)。

## 6. 代替表現: NDJSON (trace 用)

trace の保存・再生は 1 行 1 フレームの NDJSON を使う (`frame_to_ndjson` /
`parse_frame_line`)。バイナリ形式とは独立で、キー順は固定・決定的。

```text
{"kind":"token","session":1,"token":42,"pos":7}
{"kind":"layer_run","layer":3,"ops":[[4,1200],[5,900]]}
```

- 数値は 10 進整数で書く (`u64` の `ns` も f64 を経由しない)。
- f16 フィールドは**ビット値 (u16)** をそのまま 10 進で書く (`entropy_f16`,
  `val`, `weights`, RoleAct の 5 値)。
- `kind` 文字列は §3 の表と同じ。空行は読み飛ばし、壊れた行は行番号付きで
  報告して継続する (`NdjsonFrames`)。

## 7. feature gate (環境変数)

すべて既定 off。off のときは呼び出し側がフレームを組み立てないので、
推論経路のコストはフラグ分岐のみ。

| 変数 | 既定 | 有効値 | 効果 |
|---|---|---|---|
| `DENNO_TELEMETRY` | off | `0` 以外 | `emit()` / 層記録全体の有無 (`enabled`) |
| `DENNO_TELEM_LAZY` | **on** | `0` / `off` / `false` で無効 | 購読者 0 のとき重い観測を省く (`has_clients`) |
| `DENNO_TELEMETRY_ARCH` | off | 空・`0` 以外 | RoleAct / ExpertResidency / TierStats / RouterTopK |
| `DENNO_TELEMETRY_ACT` | off | 空・`0` 以外 | LAYER_ACT |
| `DENNO_TELEMETRY_ACT_TOPK` | 64 | 整数 | LAYER_ACT の K |
| `DENNO_TELEMETRY_TOPK` | off | `1` / `true` / `yes` / `on` (大小無視) | TOP_K |
| `DENNO_TELEM_EXPERT_SKIP` | off | `1` | EXPERT_SKIP (`DENNO_TELEMETRY_ARCH` も必要) |
| `DENNO_TELEMETRY_ARCH_DBG` | off | `1` / `true` / `yes` | arch の stderr 統計 |
| `DENNO_TELEMETRY_ACT_DBG` | off | `1` / `true` / `yes` | LAYER_ACT の stderr 統計 |

`DENNO_TELEMETRY` の既定解釈はクレート (`enabled()`) と serve で異なる。
serve は未設定を有効とみなし、判定結果を `set_enabled()` で流し込む。

## 8. 送出ハブ

- `emit(frame)` は bounded channel (`CHANNEL_CAP = 4096`) への `try_send`
  のみ。**待たない**。満杯・切断時はドロップし `dropped()` を増やす。
  欠落は `GAP` フレームで受信側に明示する。
- 初期化は `Mutex` で直列化し、`RX` を先に埋めてから `TX` を公開する。
- 受信は `try_recv_timeout` / `try_recv_nowait` / `recv_global_timeout`。
- `layer_begin` / `record_layer_op` / `layer_flush` は thread-local に
  LAYER_RUN を溜め、flush で 1 枚送る (空なら送らない)。

## 9. golden テスト

`crates/denno-telemetry/tests/wire_golden_994.rs` が次を固定する。

- 全 17 variant の期待バイト列 (リテラル) と decode 往復
- kind 番号 1..17 と NDJSON 名の対応
- varint / f16 のエンコード
- 並列配列の切り揃え、空配列の最小形
- 全 golden の全切り詰めが `Err`、末尾 1 バイトが `Err`
- 未知 version / kind / magic の拒否

`cargo test --release -p denno-telemetry` で実行する。
