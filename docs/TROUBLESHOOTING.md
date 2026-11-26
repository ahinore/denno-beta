# トラブルシューティング & 性能診断ガイド

起動しない・遅い・落ちる・HTTP がエラーになる・ビューアが空、のときに最初に読む文書。
ここに書いたフラグ・環境変数・ルート・ログ行・HTTP ステータスは、すべて `crates/` の
現在のコードで名前と値を確認したものだけを載せている (存在しないノブは書かない)。
数値のラベル: **[実装]** = コードの定数、**[推定]** = 測っていない見積り。

対象は 2 つのサーバーバイナリ:

| コマンド | 何のサーバー | 既定 | ルート |
|---|---|---|---|
| `denno serve-flash --dir <model-dir>` | Qwen3.8-Flash-Next (`qwen4exp`) 専用の最小サーバー | `127.0.0.1:8098` / `--max-ctx 4096` / `--max-tokens 512` | `POST /v1/messages`, `GET /v1/models`, `GET /health`, `GET /ready`, `GET /metrics`, `GET /v1/telemetry/ws` (ほかに介入 / ステアリングの REST) |
| `denno serve <repo>` | 27B 級 GGUF の多プロトコルサーバー | `127.0.0.1:8080` / 実効 `max_tokens` 1024 | Anthropic (`/v1/messages`)・OpenAI (`/v1/chat/completions`, `/v1/completions`)・Ollama (`/api/chat`, `/api/generate`) と `/health` `/ready` `/metrics` `/v1/telemetry/ws` |

`serve-flash` は OpenAI / Ollama のルートを持たない。叩くと
`404` + `{"type":"error","error":{"type":"not_found_error","message":"<path>"}}` を返す。
`serve-flash` は生の TCP 実装で、CORS 層・認証・ボディ上限 (413)・レート制限を持たない
(§4.3 / §4.4 / §4.5 は `serve` の話)。

---

## 0. 最初に見る 3 つ

### 0.1 `denno doctor`

```bash
denno doctor --dir <model-dir>          # 人が読む形
denno doctor --dir <model-dir> --json   # 自動化向け (JSON を stdout へ)
```

`[CPU]` / `[メモリ]` / `[ディスク]` / `[ファイル記述子]` / `[GPU]` / `[モデル]` /
`[環境変数 (DENNO_*)]` / `[推奨]` の 8 節を出す **[実装]**。
`--json` は真偽フラグで、ファイルへは書かない (stdout に JSON を出す)。
GPU は実行しない (`libcuda.so.1` と `/proc/driver/nvidia` の**存在**、Level Zero
loader の**存在**だけ見る)。`--dir` を省くとモデル検査はしない (`[モデル]` は未指定表示)。

- モデル検査が NG (シャード欠け・0 バイト・`.gguf` が 0 本) のときは非 0 で終わる。
  レポート自体は先に出力されるので、JSON は失敗時も読める。
- `[環境変数 (DENNO_*)]` は値が秘匿対象 (`*_API_KEY` / `*_TOKEN` / `*_SECRET` など) の
  ときだけ `(値は非表示)` を添える。
- `[GPU] CUDA driver` が `-` で `NVIDIA kernel` だけ出る場合は「ドライバは居るが
  `libcuda.so.1` が見えない」= コンテナなら `--gpus all` と NVIDIA runtime、
  ホストなら libcuda 入りのドライバが要る、と `[推奨]` に出る。
- SIMD に `avx2`/`fma` が無いと Q4_K expert の GEMV はスカラーへ落ちる (遅い)。

### 0.2 起動ログの先頭

`serve-flash` は listen 前に次の 1 行群を出す。ここだけで大半は切り分けできる。

```
[flash] N 分割ファイル: ["...-00001-of-0000N.gguf", ...]
[flash] vram budget: max_ctx=... weights=... MiB (M 本) staging=... MiB arena=... MiB kv=... MiB (host, max_ctx 比例) host_weights=... MiB experts=... MiB (mmap) parts=N
[pf-budget] Healthy|Tight|Critical total=... MiB avail=... MiB pread=... ra=... rc=... MiB tile=... pool=... MiB pread_kb_cap=...
[auto-config] plan vram=... MiB ram=... MiB model=... MiB Q4 (dense~...) -> weight=... MiB expert_vram=... MiB expert_gpu=on|off min_hit=... ram_tier=... pread=... chunk=... applied=... kept(explicit env)=...
[exp-stats] on file=... (iq_gpu=... vram=... ram=... ...)
[flash] telemetry: on|off (DENNO_TELEMETRY=0) (GET /v1/telemetry/ws)
[flash] telemetry act: ...
[flash] telemetry topk: on (top-8 を毎トークン)|off (DENNO_TELEMETRY_TOPK=1 で有効)
[flash] metrics: on|off (GET /metrics, DENNO_METRICS=...)
[flash] listening on http://<addr> model=<name>
```

`[pf-budget]` は配分が変わったときだけ出る (平常の Healthy では 1 行目のみ **[実装]**)。

### 0.3 `/health` / `/ready` / `/metrics` / telemetry は別物

| エンドポイント | `serve` | `serve-flash` |
|---|---|---|
| `GET /health` (`/healthz` も同じ) | 常に 200 `{"status":"ok"}` | 常に 200 `{"ok":true,"model":"<name>"}` |
| `GET /ready` | モデルロード済み + ワーカー生存 + drain 中でないとき 200。それ以外は 503 (`loading` / `draining`) | drain 中でなければ 200。モデルは listen 前にロード済みなので `loading` は出ない。drain 中は 503 (`draining`) |
| `GET /metrics` | **常に 200**。`DENNO_SERVER_METRICS` は記録の on/off だけ (§5.1) | `DENNO_METRICS` が on のときだけ 200。off は 404 `not_found_error` |
| `GET /v1/telemetry/ws` | `DENNO_TELEMETRY` が `0` 以外のとき有効。未設定は無効で 503 (§5.3) | `DENNO_TELEMETRY` が `0` のときだけ無効 (§5.3) |

`/health` / `/ready` / `/metrics` は API キーを要求しない (`serve` でも認証対象外)。
`/v1/telemetry/ws` も `serve` の認証チェックを通らない。

---

## 1. 起動とモデル読み込み

### 1.1 `--dir` に `.gguf` が無い

`serve-flash` は `--dir` の**直下**の `*.gguf` を全部集めて名前順に使う。
サブディレクトリは辿らない。

- 症状: `... に .gguf が無い` で即終了。
- 原因: 量子化ディレクトリ (例: `UD-Q4_K_XL`) を指すべきところを、その親を指している。
- 対処: `.gguf` が実際に入っているディレクトリを `--dir` にする。

`denno serve <repo>` は repo 引数が実在のディレクトリならそこを直接見る。
そうでなければ hub cache を見て、無ければ
`not cached: <path> (run `denno pull <repo>` first)` で終了する。

### 1.2 分割シャードの欠け・不整合

`denno doctor --dir` は `...-NNNNN-of-MMMMM.gguf` の命名から欠番と 0 バイトを検出する **[実装]**。
ロード時 (`GgufFile::open_split`) はさらに次を弾く。すべて起動失敗になる。

| メッセージ (先頭) | 意味 |
|---|---|
| `split GGUF に split.no / split.count が無い` | 分割ファイルなのにメタデータが欠けている (単体 GGUF を混ぜた等) |
| `split.no が連番でない: 位置 i に split.no=...` | シャードが欠けている / 別モデルのシャードが混ざっている |
| `split.count=... が渡したパート数 ... と一致しない` | 上と同じ。`--dir` の中身を混ぜた典型 |
| `テンソル名が split 間で重複: <name>` | 同じシャードを 2 回置いた、または別リビジョンの混在 |
| `split.tensors.count がパート間で食い違う: ...` | シャードの取り違え |
| `split.tensors.count=... (全体の宣言値) が実際のテンソル数 ... と一致しない` | シャードの取り違え |

対処は再ダウンロード。`--dir` にモデルの系列を 1 つだけ置く。

### 1.3 メタデータ / トークナイザ

- `メタデータを読めない`: `qwen4exp` 以外の GGUF を `serve-flash` に渡している。
  `serve-flash` は Flash-Next 専用で、27B 級は `denno serve` を使う。
- `トークナイザを組めない`: GGUF 内のトークナイザ定義が欠けている (変換時の取りこぼし)。
- どちらも `Error: <context>: <元のエラー>` の形で、どの段で落ちたかが前置される。

### 1.4 最初のリクエストだけ極端に遅い (page cache が冷えている)

症状の見分け:

- 1 本目だけ prefill が 10 秒級、2 本目以降は数秒。`[flash] prefix miss: cache=... prompt=... lcp=... cached_tok=... prompt_tok=...` が出る。
- SSD から expert 行列 (数十 GB) を読み直している。

対処 (どれも数値経路は変えない):

| ノブ | 既定 | 意味 |
|---|---|---|
| `DENNO_START_PREFETCH=1` | **off** **[実装]** | listen 前に expert のバイト範囲を `pread` で page cache へ載せる |
| `DENNO_START_PREFETCH_SECS` | 27 | 温めの打ち切り (秒)。起動の延びを抑える |
| `DENNO_START_PREFETCH_THREADS` | 12 | 読みスレッド数 |
| `DENNO_START_PREFETCH_EXTRA_MB` | 4096 | expert 以外に温める上限 (MiB)。0 で expert だけ |
| `DENNO_EXPERT_DROP_CACHE_ON_START=1` | off | 起動時に page cache を落とす (**診断用**。冷えた機械の再現) |
| `DENNO_START_DROP_CACHE=1` | off | 同上 (モデル全体版、診断用) |

page cache は誰の物でもないので、別モデルを回すと追い出される。追い出されたくない
expert は**所有する RAM tier** に置く:

| ノブ | 既定 | 意味 |
|---|---|---|
| `DENNO_EXPERT_RAM_MB` | 未設定 = off | expert 生バイトを所有バッファへ常駐 (MiB / `auto`) |
| `DENNO_EXPERT_RAM_PIN` | on | `0` で pinned を諦めてヒープ |
| `DENNO_EXPERT_RAM_FILL` | on | `0` で起動時充填と再調整をしない |
| `DENNO_EXPERT_RAM_MARGIN_MB` | 8192 | `auto` で `MemAvailable` に残す余裕 |
| `DENNO_EXPERT_RAM_REBALANCE_SECS` | 30 | 走行中の追従間隔 (秒、0 で停止) |
| `DENNO_EXPERT_RAM_DBG=1` | off | 1 リクエストごとに 1 行 |

`auto` は「`MemAvailable` − 非 expert バイト − 余裕」で、**非 expert を引く**のが要点。
引かないと tier が page cache を追い出して decode が悪化する (コードのコメントにある
実測例: 19.4 → 46.4 ms/tok)。

### 1.5 `... を listen できない`

ポート使用中か、`--host` がそのインターフェースに無い。`serve-flash` は
**逐次処理** (モデルは 1 つ、同時リクエストは受けない) なので、2 本目以降の接続は
accept の待ち行列で待つ。多重接続で速くしたいならプロセスを分けるしかない。

### 1.6 コンテキスト超過

- `serve-flash`: `プロンプト <N> トークンが上限 <M> に収まらない` → 400
  `invalid_request_error`。`--max-ctx` を上げるか入力を短くする。
- `serve`: `invalid_request_error: prompt (<N> tokens) が上限 <cap> を超えます。
  DENNO_MAX_PROMPT_TOKENS を上げるか入力を短くしてください`。
  cap の既定は 32768 **[実装]** (`DENNO_MAX_PROMPT_TOKENS`)。
  上限は「Q4 GPU 経路の作業領域がプロンプト長 × 層数で残る」ことへの防御で、
  上げると OOM に近づく。
- `serve` にはこのほか「prompt + max_tokens が KV スロット上限を超える」と
  「KV 予算超過」の 400 もある (同時リクエスト数か `DENNO_KV_BUDGET_MB` を見直す)。

`--max-ctx` を上げると増えるのは**ホスト側の KV** だけ。起動ログの `kv=... MiB (host, max_ctx 比例)`
がそれで、`weights` / `staging` / `arena` は `max_ctx` に比例しない **[実装]**。

`serve-flash` の文脈スロット:

| ノブ | 既定 | 意味 |
|---|---|---|
| `--kv-slots` / `DENNO_KV_SLOTS` | 2 | 2 = main + aux、3 以上で LCP 一致 + LRU の汎用 prefix cache |
| `DENNO_KV_SLOT_TOK` | max_ctx | 1 スロットのトークン上限 (8 未満・未設定は max_ctx) |

追加スロットの目安は、3 本以上のときだけ起動時に
`[flash] kv-slots=N (LCP + LRU)。追加 X 本ぶんのメモリ目安: GDN 状態 ... MiB + KV ... MiB (max_ctx=...)`
として出る。足りないと `[flash] slot i: LRU 追い出し (...)` が増える。

---

## 2. VRAM 予算と OOM

### 2.1 `--vram-budget auto` と手動

`--vram-budget` は `serve-flash` のフラグで、`auto` か MiB の数値を受ける。
未指定なら `DENNO_AUTO_CONFIG` を見る。CLI が env に勝つ **[実装]**。
`DENNO_AUTO_CONFIG` は `1` / `true` / `on` / `yes` / `auto` を真として受ける。

- `auto`: 起動時に VRAM / RAM / モデルサイズを引き、**未設定の** `DENNO_*` だけを埋める。
- 数値 (MiB): auto と同じ式に「総 VRAM = その値」を渡す。検出できない環境用。
- 適用結果は `[auto-config] plan ... applied=[...] kept(explicit env)=[...]` の 1 行。
  `kept` は「明示 env があったので触らなかった」変数。

予算の実体は主にこの 2 つ:

| ノブ | 意味 |
|---|---|
| `DENNO_GPU_WEIGHT_MB` | dense 重みを VRAM に載せる上限 (MiB) |
| `DENNO_EXPERT_VRAM_MB` | expert を VRAM に載せる上限 (MiB、0 = 載せない) |

補助: `DENNO_VRAM_RESERVE_MB` (計画で残す余裕)、`DENNO_EXPERT_GPU_MIN_HIT`
(expert を VRAM に置く最小ヒット数)、`DENNO_GPU_KV_MB` / `DENNO_KV_BUDGET_MB` (KV 予算)、
`DENNO_VRAM_BUDGET_CARD` (カード側の予算上書き)。

### 2.2 auto が効かないとき

`[auto-config] ...` の行が**出ない**なら auto は走っていない。順に確認する。

1. `--vram-budget` の値が解釈できるか。`auto` か正の整数 (MiB) だけ。
   `12G` のような単位付きは**黙って無視**される (`VramBudget::parse` が `None`)。
2. `cuda` feature 無しでビルドしたバイナリか。VRAM 検出は `cuDeviceTotalMem_v2` を
   dlopen 経由で呼ぶので、feature 無し / libcuda 無しでは「検出できない」= 何も適用しない。
3. `[auto-config] skip: ...` が出ていないか (計画は立ったが適用で失敗)。

検出できない環境では `--vram-budget <MiB>` で総量を渡す。それでもだめなら
`DENNO_GPU_WEIGHT_MB` / `DENNO_EXPERT_VRAM_MB` を手で決める (明示 env は常に勝つ)。

### 2.3 予算は「上限」であって「予約」ではない

`DENNO_EXPERT_VRAM_MB` と `DENNO_GPU_WEIGHT_MB` の和がカードの VRAM を超えていても
既定の流儀。実際に載るのは「使われた分」だけで、expert は miss するたびに
載せ替える。和を小さくすると載せ替えが増えて遅くなるが、起動は通る。

### 2.4 OOM の読み方と回復

典型のログ (実装コメントに残っている実例):

```
CUDA cuMemAlloc が失敗: error 2 (要求 779 MiB / 空き 315 MiB / 総 32110 MiB)
chat failed: prefill residual chain: qkv chain failed
```

`要求` は失敗した 1 本の確保、`空き` はその瞬間の空き。**合計が足りていても落ちる**
のが要点で、原因は (a) 断片化 (b) 失敗経路の解放漏れ。prefill の一時領域は
1 本の arena から切り出す経路が既定で入っている **[実装]**。

切り分けの順 (上から試す):

| 手順 | ノブ | 効果 |
|---|---|---|
| 1 | `DENNO_SCRATCH_ARENA=0` | arena を切って個別確保に戻す。arena が原因かの判定用 (数値不変) |
| 2 | `DENNO_PF_CHUNK` を小さくする | prefill のチャンクを縮め、活性のピークを下げる (**推定**: ピークはチャンク行数に比例) |
| 3 | `--max-ctx` を下げる | ホスト KV が減る。デバイス側の OOM には効かないことがある |
| 4 | `DENNO_EXPERT_VRAM_MB` を下げる / `0` | expert を VRAM に載せない |
| 5 | `DENNO_GPU_WEIGHT_MB` を下げる | dense 重みの常駐を減らす |
| 6 | `DENNO_CUDA` を外す | CPU だけで回して「モデル/シャードは正しいか」を切り分ける |

メモリ圧迫そのものは `DENNO_PF_MEMADAPT` (既定 on) が `/proc/meminfo` の
`MemAvailable` / `SwapFree` から prefill の資源配分を決める。段は
`Healthy` (恒等) / `Tight` (約半分) / `Critical` (最小構成) の 3 つで、
切り替わると `[pf-budget] <段> total=... avail=...` が 1 行出る。
`DENNO_PF_MEMADAPT=0` で従来の静的既定に戻せる (A/B 用)。
明示の `DENNO_PF_*` (例: `DENNO_PF_PREAD_THREADS`, `DENNO_PF_POOL_MB`) はこの計算に勝つ。

### 2.5 expert を CPU / RAM に置く

GPU が小さく expert が載らない場合の定番:

```
DENNO_EXPERT_IQ_GPU=0            # expert の IQ カーネルを GPU に上げない (CPU のまま)
DENNO_EXPERT_GPU_MIN_HIT=0       # VRAM に置く閾値を下げる (載せたい場合)
DENNO_EXPERT_CACHE_MB=<MiB>      # CPU expert の直近分をホスト RAM に常駐 (行列ごと、既定 0 = 無効)
DENNO_EXPERT_CACHE_CAP_MB=<MiB>  # 常駐の総量上限 (3 行列合計。要求が超えたら比例縮小)
DENNO_EXPERT_RAM_MB=auto         # 所有 RAM tier (前述)
```

`DENNO_EXPERT_CACHE_MB` は matrix 単位の pread キャッシュで、VRAM は使わない
(`3 × MB` MiB が 3 行列ぶんの目安)。

expert の当たり外れを見たいとき:

- `DENNO_EXPERT_STATS_DIR=<dir>`: モデルごとの使用回数ファイル
- `DENNO_EXPERT_IQ_STATS_FILE=<file>`: 明示パス
- `DENNO_START_WARM_EXPERTS` / `DENNO_START_WARM_MB`: 起動時に page cache へ温める expert 本数と上限
- `DENNO_EXPERT_CACHE_STATS=1`: LRU/TinyLFU の当たり統計
- `DENNO_EXPERT_CACHE_POLICY=tinylfu|lru` / `DENNO_EXPERT_GATE_TINYLFU`

VRAM tier の実績は起動後に
`[flash] gpu weights: N 本 ok / M 本 失敗 (全 K 本のうち X%)` の 1 行で出る。
失敗があると最初の理由 (`first_error`) も付くので、OOM の一次証拠はここで取る。

---

## 3. GPU バックエンドの readiness

### 3.1 CUDA はビルド時 feature + 実行時 dlopen

- `cuda` は `denno-engine` の feature (`crates/denno-engine/Cargo.toml`)。
  ビルドは README の `cargo build --release --features cuda -p denno-cli -p denno-engine`。
- 依存クレートは無く、`libcuda.so.1` を**実行時に** `dlopen` する。PTX は
  リポジトリにコミット済み。したがって「ビルドは通るが実行時に GPU が無い」が正常に起こる。
- そのときのエラー: `libcuda.so.1 を dlopen できません (NVIDIA driver が無い/見えない)`。
- ドライバが新しすぎる/古すぎる場合も `cuInit` が失敗し、同じ「GPU が使えない」に落ちる。

### 3.2 `DENNO_CUDA`

- 値は `1` / `true` / `on` のときだけ有効 **[実装]**。`yes` や空文字は無効。
- 未設定なら GPU 経路は一切通らない (CPU のみ)。**既定は CPU**。
- 一度でも初期化に失敗するとプロセス内で GPU 経路が恒久ラッチされ、
  以降は CPU に落ちる (`CUDA cuCtxSetCurrent failed, using CPU gguf: ...` など)。

### 3.3 Intel Arc / Level Zero

**この木に Level Zero の実行経路は無い。** ワークスペースと `denno-engine` の
feature は `default = ["par"]` / `par` / `test-utils` / `cuda` だけで、
`ze` / `xe` のような feature は存在しない。

- `denno doctor` は `libze_loader.so.1` と `libze_intel_gpu.so.1` の**存在**を
  表示するだけ。検出しても `[推奨]` に
  `Intel Level Zero loader を検出 (...) が、このビルドに Level Zero 経路は無い`
  と出る (存在しないフラグは勧めない)。
- したがってこの木で Arc を挿しても推論は CPU 経路になる。バックエンド非依存の
  ノブ (prefill 資源・決定論) はそのまま意味を持つ。VRAM 自動検出は CUDA の
  検出器を使うので効かない → `--vram-budget <MiB>` で総量を明示する。
- Arc 移植の詳細は `docs/PERFORMANCE_TUNING.md` §4.3 を参照。

### 3.4 GPU が使えているかの確認順

1. `denno doctor` の `[GPU]`: `CUDA driver` に `libcuda.so.1` のパスが出るか。
   `NVIDIA kernel` だけならユーザー空間の libcuda が無い (§0.1)。
2. 起動ログの `[flash] vram budget: ...` と
   `[flash] gpu weights: N 本 ok / M 本 失敗`。`M > 0` なら `first_error` を読む。
3. `GET /metrics` の `denno_expert_vram_used_bytes` / `denno_expert_vram_hit_rate` と
   `denno_last_decode_ms_per_token`。hit rate が 0 のままなら expert は VRAM に載っていない。
4. `DENNO_CUDA` の値 (§3.2)。`yes` は無効。

---

## 4. HTTP サーバー (`serve`)

### 4.1 ルートの取り違え

| 症状 | 原因 |
|---|---|
| OpenAI クライアントが 404 `not_found_error` | `serve-flash` を叩いている。`/v1/chat/completions` は `denno serve` のみ |
| Ollama クライアントが 404 | 同上 (`/api/chat` / `/api/generate` は `denno serve` のみ) |
| Anthropic クライアントが 404 | パスの綴り違い (`/v1/messages` であって `/v1/message` ではない) |
| `/v1/models` は 200 なのに生成だけ失敗 | モデル名の指定違い。`GET /v1/models` の `id` をそのまま使う |
| OpenAI の `/v1/embeddings` などが 501 | 経路は正しいが未実装 (`not_implemented`)。生成系のルートへ切り替える |

`serve-flash` の `GET /v1/models` は `context_length` と `capabilities`
(`anthropic` / `openai` / `tools` / `streaming` / `telemetry`) を返す。
`telemetry` が `false` ならビューアは繋がらない (§5.3)。

### 4.2 認証 (401)

`serve` だけが認証を持つ。`--api-key` / `--api-keys` / `DENNO_API_KEYS` の
いずれかが非空なら有効で、複数指定は**和集合** (カンマ区切り、どれか 1 つに一致すれば通る)。

- 送るヘッダは `x-api-key: <key>` か `Authorization: Bearer <key>` のどちらでもよい。
  両方あるときは、どれか 1 つが一致すれば受理される。
- 不一致は `401` + `{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}`。
- `--api-key` を付けなければ認証なし (従来どおり開放)。`serve-flash` に認証は無い。
- 認証対象は生成系と `/v1/models`。`/health` / `/ready` / `/metrics` /
  `/v1/telemetry/ws` は対象外。

### 4.3 CORS

既定は「ループバックの任意ポートだけ許可」**[実装]**。サーバは 127.0.0.1 で
待ち受けるのが既定なので、ローカルブラウザ (denno-view など) からはそのまま使え、
外部オリジンには `Access-Control-Allow-Origin` を出さない。

| 指定 | 意味 |
|---|---|
| 未指定 / 空 | `http(s)://localhost` / `127.0.0.1` / `[::1]` を任意ポートで許可 |
| `*` | 任意オリジン (`Access-Control-Allow-Origin: *`) |
| カンマ区切り | その一覧だけを許可。**ループバック既定は無効になる**ので、ローカルも使うなら一覧に入れる |

- 優先順は CLI `--cors-origin` > env `DENNO_CORS_ORIGIN`。
- プリフライト (`OPTIONS` + `Access-Control-Request-Method`) は 204 で完結する。
  許可メソッドは `GET, POST, OPTIONS`、既定の許可ヘッダは
  `content-type, x-api-key, authorization, anthropic-version, x-request-id`、
  キャッシュは 86400 秒。
- 非プリフライトの応答には `Access-Control-Allow-Origin` と
  `Access-Control-Expose-Headers: x-request-id, x-denno-ttft-ms, x-denno-decode-tps`
  を足す (`*` 以外では `Vary: Origin` も)。
- CORS 層はレート制限・ボディ上限より外側なので、404 / 405 / 413 / 429 の応答にも
  CORS ヘッダが付く。
- `serve-flash` は CORS 層を持たない。ブラウザから直接叩かず、denno-view の
  `/ws` / `/api/*` 中継を使う (§7)。

### 4.4 413 Payload Too Large

`serve` のボディ上限。`--max-body-mb` > env `DENNO_MAX_BODY_MB` > 既定 16 MiB。
値は 1〜4096 MiB にクランプされる **[実装]**。

- `Content-Length` が上限を超える要求は、本文を読む前に 413 になる。
- `Content-Length` が無い chunked ボディは、読み込み中に上限を超えた時点で 413。
- 本文は他のエラーと同じ JSON 封筒:
  `{"type":"error","error":{"type":"invalid_request_error","message":"request body too large: the configured limit is <N> bytes"}}`
- 対処: 入力を縮めるか `--max-body-mb` を上げる。上げすぎると prefill の
  メモリピークも上がるので、まずプロンプト長と添付サイズを確認する。
- `serve-flash` にボディ上限は無い。

### 4.5 429 Too Many Requests

429 は 2 系統ある。どちらも `error.type` は `rate_limit_error`。

| 発生源 | 条件 | ノブ | `Retry-After` |
|---|---|---|---|
| トークンバケット | 推論エンドポイントの RPM 超過 | `DENNO_RATE_LIMIT_RPM` / `DENNO_RATE_LIMIT_BURST` | あり (秒) |
| KV スロット枯渇 (`AdmitKind::SlotsBusy`) | 全スロットが実行中 | `DENNO_BATCH_SLOTS` (serve) | なし |

- トークンバケットは `DENNO_RATE_LIMIT_RPM` が正のときだけ有効。burst の既定は
  `rpm / 60` (最低 1)。`/health` `/ready` `/metrics` は消費しない。
- OpenAI 形式では `type` が `rate_limit_error`、`code` が `rate_limit_exceeded`。
- 対処: `Retry-After` を尊重して待つ。即時リトライを繰り返すとバケットが回復しない。
  KV 枯渇なら同時リクエストを減らすか `DENNO_BATCH_SLOTS` を見直す。

### 4.6 503 Service Unavailable

| 状況 | 本文の `error.type` | ヘッダ |
|---|---|---|
| `draining` (SIGTERM / SIGINT 後) | `overloaded_error` | なし |
| 待ち行列が上限 (`queue_full_error`) | `overloaded_error` | `Retry-After: <秒>` |
| 同時実行の防御上限 (`AdmitKind::SessionLimit`) | `overloaded_error` | なし |
| 内部要因 (`AdmitKind::Internal`) | `api_error` (500) | なし |

`/ready` は受付不可の間ずっと 503 を返す (§0.3)。本文の `status` は
`loading` (モデルロード未完 / ワーカー死亡) か `draining` (停止中)。

drain 中の新規生成要求は
`server is draining (shutdown signal received); the request was not accepted`
で 503 になる。`serve-flash` も同じ文言で、drain 中は **GET 以外**が 503。
drain の待ち時間は `DENNO_DRAIN_TIMEOUT_SECS` (秒) > `DENNO_DRAIN_TIMEOUT_MS` (ms)
> 既定 10 秒。0〜600000 にクランプされ、0 = 待たない。超過すると
`[flash] drain 上限超過: 進行中 N 本のまま終了する` (serve は
`drain 上限 (...s) 超過: pending=... in_flight=... のまま停止する`) を出して終了する。

プロトコル差 (同じ「キュー満杯」でも):

| ルート | ステータス | 本文 |
|---|---|---|
| `/v1/messages` | 503 | `{"type":"error","error":{"type":"overloaded_error",...}}` |
| `/v1/chat/completions`, `/v1/completions` | 429 | `{"error":{"type":"rate_limit_exceeded",...}}` |
| `/api/chat`, `/api/generate` | 503 | `{"error":"..."}` に `Retry-After` を添える |

Anthropic のキュー満杯の本文は
`server busy: request queue is full (max_queue = <N>, in_flight = 1); retry later`。
OpenAI が 429 を返すのは「一時的な容量不足 = 後で再試行」の慣例に合わせたもの。

調整:

| ノブ | 既定 | 意味 |
|---|---|---|
| `--max-queue-depth` (別名 `--max-queue`) / `DENNO_MAX_QUEUE_DEPTH` | 32 | 待ち行列の上限 (上限 65536) |
| `--request-timeout-secs` / `DENNO_REQUEST_TIMEOUT_SECS` | 0 (無効) | リクエストの壁時計上限。超えると生成を打ち切りスロットを解放 |

`serve-flash` は**逐次処理**なので過負荷応答は無い。2 本目は accept で待つ。
1 本が長いと全部が待つので、タイムアウトはクライアント側で持つ。

### 4.7 SSE の切断

- `serve`: hyper が書き込み失敗 (broken pipe / connection reset) を検知すると
  生成を中断し、KV スロットを解放して次のリクエストに回す。ログは
  `client disconnected; generation aborted`、指標は
  `denno_requests_aborted_total{reason="disconnected"}` (`/ready` の本文にも
  `disconnected` として出る)。
- `serve-flash`: 送信失敗は「クライアント切断」として扱い、次のトークンで止めて
  文脈をキャッシュに残す (`[flash] abort: N トークンで停止`)。復帰後のリクエストは
  そのキャッシュを再利用できる。
- SSE の先頭が `text/event-stream` で返らない場合、多くのクライアントは
  「ストリームでない」と判断して JSON として読む。`stream: true` を送っているか、
  途中のプロキシがバッファリングしていないかを確認する。

### 4.8 クライアントごとのつまずき

| 症状 | 原因 / 対処 |
|---|---|
| 401 `authentication_error` `invalid x-api-key` | `serve` を API キー付きで起動している。`x-api-key` か `Authorization: Bearer` を送る (§4.2) |
| OpenAI クライアントで `max_tokens` が無視される | `max_tokens` と `max_completion_tokens` の両方があると `max_tokens` が優先。片方だけ送る |
| 生成が途中で切れる | `max_tokens` 未指定時の既定に当たっている。`serve` は 1024 (クランプ上限 4096)、`serve-flash` は 512 |
| 同じ seed なのに出力が違う | 量子化モード (`--quant int8` は fp32 とビット一致しない)、`temperature` が 0 でない、チャンク幅が違う |
| thinking 有効のクライアントで出力が崩れる | `serve` の `--temperature-cap` で実効 temperature を頭打ちにする (小型モデル向け) |

注意: `serve` の `--max-tokens` は clap で受理されるが、現在の `ServeConfig` には
渡されておらず、実効値はサーバー定数の 1024 (リクエストで明示した値は 4096 で
クランプ) **[実装]**。既定を変えたいときはリクエスト側で `max_tokens` を送る。

### 4.9 アクセスログ

`serve` の `--log-format json` で 1 リクエスト 1 行
(`ts_ms` / `method` / `path` / `status` / `duration_ms`) になる。
パラメータ値 (`api_key` / `token` / `secret` など) とヘッダ値はログに出さない。

---

## 5. 観測: `/metrics` と telemetry

### 5.1 `serve` の `/metrics`

`GET /metrics` は **常に 200** で Prometheus text exposition format 0.0.4 を返す。
`DENNO_SERVER_METRICS=0|false|no` は**記録**を止めるだけで、エンドポイントは消えない
(記録が止まっていれば値は 0 のまま)。既定は記録 on。
prefix cache の計器は `DENNO_SERVER_METRICS=0` でも数え続ける。

主な系列:

| 系列 | 意味 |
|---|---|
| `denno_server_ready` / `denno_server_draining` | `/ready` と同じ判定 (1/0) |
| `denno_requests_in_flight` / `denno_requests_queued` | 実行中 / 待ち行列 |
| `denno_requests_completed_total` | 完走数 |
| `denno_requests_aborted_total{reason="timeout"\|"disconnected"\|"error"}` | 打ち切り内訳 |
| `denno_http_requests_total{endpoint,status}` | ルート別ステータス (未知パスは `other`) |
| `denno_prompt_tokens_total` / `denno_completion_tokens_total` | トークン計数 |
| `denno_kv_reuse_ratio` | prefix 再利用の比率 |
| `denno_prefix_cache_hit_rate` | prefix cache のヒット率 |
| `denno_ratelimit_rejections_total{reason="token_bucket"}` | バケットによる 429 数 |
| `denno_ttft_seconds` / `denno_tpot_seconds` | 初トークン / トークン間の要約 |

### 5.2 `serve-flash` の `/metrics`

`DENNO_METRICS` が既定 on。`0` / `false` / `no` で経路ごと消え、`GET /metrics` は
404 `not_found_error` になる (§0.3)。系列名は vLLM 互換が主体:

| 系列 | 意味 |
|---|---|
| `vllm:num_requests_running` / `vllm:num_requests_waiting` | 実行中 / 待ち |
| `vllm:time_to_first_token_seconds` / `vllm:inter_token_latency_seconds` | TTFT / トークン間 |
| `vllm:kv_cache_usage_perc` | KV 使用率 |
| `vllm:prompt_tokens_total` / `vllm:generation_tokens_total` | トークン計数 |
| `vllm:prefix_cache_hits_total` / `vllm:prefix_cache_queries_total` | prefix cache |
| `denno_last_prefill_tokens_per_second` / `denno_last_decode_ms_per_token` | 直近の prefill/decode 速度 |
| `denno_expert_vram_used_bytes` / `denno_expert_vram_hit_rate` | expert の VRAM 常駐とヒット率 |
| `denno_expert_disk_read_bytes_total` | expert のディスク読み |
| `denno_build_info` | ビルド SHA |

### 5.3 テレメトリのゲート

| サーバー | 既定 | 有効化 / 無効化 |
|---|---|---|
| `serve-flash` | **有効** | `DENNO_TELEMETRY=0` のときだけ無効 (それ以外の値は有効) |
| `serve` | **無効** | `DENNO_TELEMETRY` を `0` 以外にする (`1` 推奨) と有効 |

`serve` で有効化していないと `GET /v1/telemetry/ws` は **503** で
`telemetry disabled (set DENNO_TELEMETRY=1)` を返す。`GET /v1/models` の
`capabilities.telemetry` でも判別できる。

### 5.4 パネルごとの追加ゲート

| ノブ | 既定 | 何が出るか |
|---|---|---|
| `DENNO_TELEMETRY_TOPK=1` | off | sampler の top-8 とエントロピー (`/2d` の token probs 欄)。無いと「TopK フレーム待ち」のまま |
| `DENNO_TELEMETRY_ACT=1` | off | 層ごとの活性フレーム |
| `DENNO_TELEMETRY_ARCH=1` | off | ルータの top-k / 役割別 RMS など構造系 |
| `DENNO_TELEM_EXPERT_SKIP=1` | off | 「ルータは選んだが介入で切った expert」(`DENNO_TELEMETRY_ARCH` も要る) |
| `DENNO_TELEM_LAZY=0` | on | 購読者 0 人のとき重い観測を省くのを止める (A/B 用) |

`DENNO_TELEM_LAZY` が on のとき、ビューアを開くまで top-8 / 活性 / arch は
組み立てられない。開いた瞬間から出るのは正常。

---

## 6. 性能診断

### 6.1 まず数字を取る

1. `GET /metrics` の `denno_ttft_seconds` / `denno_tpot_seconds` (serve) または
   `vllm:time_to_first_token_seconds` / `denno_last_decode_ms_per_token` (serve-flash)。
2. 起動ログの `[pf-budget]` の段 (`Healthy` / `Tight` / `Critical`) と
   `[auto-config] plan` の `applied` / `kept`。
3. `[flash] gpu weights: N 本 ok / M 本 失敗` と `first_error`。
4. 1 リクエストの内訳が要るときは `DENNO_Q4_PROFILE=1` (または
   `DENNO_DEV_PROFILE=1`)。decode / prefill の段ごとに 1 行群を出す。
   計測のみで数値経路は変えない。

### 6.2 prefill が遅い

- `[pf-budget]` が `Critical` に落ちていないか。落ちていれば同時 prefill が
  最小構成になっている。メモリを空けるか `DENNO_PF_MEMADAPT=0` で静的既定に戻して
  A/B する。
- `DENNO_PF_CHUNK` が大きすぎないか。チャンクを縮めるとピークは下がるが
  オーバーヘッドは増える (**推定**)。
- `DENNO_PF_PREAD_THREADS` / `DENNO_PF_POOL_MB` の明示値が memadapt の計算に
  勝っている。意図しない値が残っていないか確認する。

### 6.3 decode が遅い

- expert が VRAM に載っているか (§3.4)。`denno_expert_vram_hit_rate` が低いなら
  `DENNO_EXPERT_VRAM_MB` を増やすか、`DENNO_EXPERT_GPU_MIN_HIT` を下げる。
- expert のディスク読みが多いなら §1.4 の prefetch / RAM tier。
- `DENNO_EXPERT_IQ_GPU=1` のまま CPU に落ちていないか (ログの `[exp-stats]`)。
- `denno_last_prefill_tokens_per_second` と `denno_last_decode_ms_per_token` を
  変更前後で比較する。数値経路を変えるノブ (量子化・チャンク) は A/B の条件を揃える。

### 6.4 CPU しか使えていない

`DENNO_CUDA` の値 (§3.2)、`denno doctor` の `[GPU]`、起動ログの
`libcuda.so.1 を dlopen できません` を確認する。CPU 経路でも SIMD に
`avx2`/`fma` が無いと Q4_K expert の GEMV はスカラーへ落ちる。

---

## 7. ビューア (`/`, `/2d`, `/3d`)

### 7.1 起動と接続

```bash
VIEW_UPSTREAM=ws://127.0.0.1:8098/v1/telemetry/ws denno-view
# 既定: VIEW_UPSTREAM=ws://127.0.0.1:8080/v1/telemetry/ws, VIEW_PORT=8093
```

| 症状 | 原因 |
|---|---|
| ページは出るが全部空 | 上流に繋がっていない。ログに `upstream disconnected, retrying in 2s ...` が 2 秒ごとに出る |
| `/2d` のチャットだけ 502 `upstream unreachable: ...` | 上流の HTTP が落ちている。`/api/*` は `VIEW_UPSTREAM` から `ws://`→`http://` に読み替えたホストへ中継する |
| `serve-flash` (8098) を動かしているのに繋がらない | 既定の 8080 を見ている。`VIEW_UPSTREAM` を明示する |
| `VIEW_PORT` を変えたのに繋がらない | 値が `u16` として読めないと黙って 8093 に戻る |

上流が落ちてもビューアは死なない。2 秒間隔で再接続し、戻ると最初に
`ServerInfo` と `KVUsage` のスナップショットを配る。

`/`, `/2d`, `/3d` はどれもビューア自身の `GET /ws` に繋ぎ、`/ws` が上流の
フレームを中継する (ブラウザから上流へ直接は繋がないので CORS 設定は要らない)。
`/2d` の REST は `/api/*` 経由、チャット欄は `/chat` 経由で上流へ流れる。

### 7.2 トークンが id のまま表示される

`VIEW_TOKENIZER=<path>/tokenizer.json` を渡す。読み込みに失敗すると
`tokenizer.json read failed (...) — token text display disabled` が出て、id 表示に落ちる。
未設定なら `tokenizer not configured (set VIEW_TOKENIZER)` を返す。
GGUF 経路 (`serve-flash`) はモデルがトークナイザを持つが、ビューア側は別プロセスなので
`VIEW_TOKENIZER` が要る。

### 7.3 フレームが落ちる / 映像がカクつく

- テレメトリは bounded チャネル (4096 フレーム) で、溢れたら**捨てて**
  `Gap { lost }` フレームで欠落を明示する。推論を待たせないための設計で、
  遅いクライアントは切り捨てられる。
- ブラウザのタブを絞ると受け側が遅くなり、`Gap` が増えやすい。
- 記録して後で見る: `VIEW_RECORD=<file>` (受信フレームを追記)、
  `VIEW_REPLAY=<file>` (`GET /replay`)、`VIEW_CAPTURE=<file>` (`GET /capture.bin`)。
  未設定のときはそれぞれ `replay record not configured (set VIEW_REPLAY)` 等を返す。

---

## 8. 早見表

| 症状 | まず見る | 対処 |
|---|---|---|
| `... に .gguf が無い` | `--dir` の中身 | 量子化ディレクトリを直接指す |
| `split.no が連番でない` | シャードの揃い | 再ダウンロード |
| 起動はするが最初の 1 本が異常に遅い | `[flash] prefix miss` | `DENNO_START_PREFETCH=1` / `DENNO_EXPERT_RAM_MB=auto` |
| `[auto-config]` が 1 行も出ない | `--vram-budget` の値 | `auto` か MiB の整数。`12G` は無視される |
| `cuMemAlloc が失敗` | `要求` と `空き` | `DENNO_SCRATCH_ARENA=0` → `DENNO_PF_CHUNK` 縮小 → expert VRAM を下げる |
| `libcuda.so.1 を dlopen できません` | `denno doctor` の `[GPU]` | ドライバ / `--gpus all`。無いなら CPU で運用 |
| `/ready` が 503 `draining` | SIGTERM を受けた | 新規を止める。待ちは `DENNO_DRAIN_TIMEOUT_MS` |
| 401 `invalid x-api-key` | API キーの指定 | `x-api-key` か `Authorization: Bearer` (§4.2) |
| 413 `request body too large` | `--max-body-mb` / `DENNO_MAX_BODY_MB` | 入力を縮める。既定 16 MiB (§4.4) |
| 429 `rate_limit_error` | RPM バケットか KV 枯渇 | `Retry-After` を尊重。`DENNO_RATE_LIMIT_RPM` / `DENNO_BATCH_SLOTS` を見直す |
| 503 `overloaded_error` | `max_queue` と `in_flight` | `--max-queue-depth` を上げる / 並列度を下げる |
| SSE が途中で切れる | サーバーログの `client disconnected` | クライアントのタイムアウト / プロキシの buffering |
| ブラウザから直叩きで CORS エラー | `--cors-origin` / `DENNO_CORS_ORIGIN` | オリジンを明示するか denno-view の中継を使う (§4.3) |
| ビューアが空 | `upstream disconnected, retrying` | `VIEW_UPSTREAM` のポート (8080 / 8098) |
| telemetry ws が 503 | `DENNO_TELEMETRY` | `serve` は `0` 以外にする (§5.3) |
| `/metrics` が 404 | `DENNO_METRICS` (serve-flash) | 既定 on。`0` になっていないか (§5.2) |
| `/2d` の token probs が空 | `DENNO_TELEMETRY_TOPK` | `=1` でサーバーを再起動 |
| トークンが数字のまま | `VIEW_TOKENIZER` | `tokenizer.json` を指す |

---

## 9. この文書の検証方法

文中のフラグ・env・ルート・ステータスは、次の形でコードに存在することを確認している。

```bash
# 環境変数
for v in DENNO_TELEMETRY DENNO_TELEMETRY_TOPK DENNO_METRICS \
         DENNO_SERVER_METRICS DENNO_CORS_ORIGIN DENNO_MAX_BODY_MB \
         DENNO_RATE_LIMIT_RPM DENNO_MAX_QUEUE_DEPTH; do
  grep -rn "\"$v\"" crates/ || echo "MISSING: $v"
done
# CLI フラグ
grep -rn -- '--vram-budget\|--max-queue-depth\|--max-ctx\|--max-body-mb\|--cors-origin' crates/denno-cli/src/
# ルート (`serve` は axum の route、`serve-flash` は path の直接比較)
grep -rn 'route("' crates/denno-server/src/lib.rs
grep -rn 'path == "/' crates/denno-cli/src/serve_flash.rs
# ステータスコード
grep -rn 'PAYLOAD_TOO_LARGE\|SERVICE_UNAVAILABLE\|TOO_MANY_REQUESTS\|UNAUTHORIZED' crates/denno-server/src/
```

存在しないノブを見つけたら、この文書ではなくコードが正しい。文書を直すこと。
docs 全体の衛生 (LAN 私有 IP / `/home` パスの混入) は
`cargo test --release -p denno-server --test docs_consistency_1249` で検査される。
