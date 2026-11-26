# ハードウェア別チューニング (VRAM 予算・キャッシュ・prefill)

この文書は **この木に実装されている** フラグと `DENNO_*` 環境変数だけを扱う。
すべて `crates/` のコードで確認した名前で、存在しないノブは書かない。
数値のラベル: **[実装]** = コードの定数、**[実測]** = このリポジトリの lab 記録、
**[推定]** = 測っていない見積り。

対象は Flash-Next (`denno serve-flash`) と、27B 系の `denno serve`。
チューニングで変わるのは **「どの重みをどこに置くか」「何本の I/O を同時に飛ばすか」
「どのキャッシュを残すか」** だけで、演算の式は変えない。丸め順が動くノブ
(チャンク幅など) はその旨を明記する。

---

## 1. VRAM 予算: `--vram-budget auto` と手動の `DENNO_*`

### 1.1 どちらを使うか

| 状況 | 使い方 |
|---|---|
| カードが 1 種類で、その総 VRAM を自動検出できる | `--vram-budget auto` (または `DENNO_AUTO_CONFIG=1`) |
| 検出できない (cuda feature 無し / libcuda 無し)、または総量を偽りたい | `--vram-budget <MiB>` |
| 予算を完全に手で決めたい | `DENNO_EXPERT_VRAM_MB` / `DENNO_GPU_WEIGHT_MB` を明示 |

`--vram-budget` は `serve-flash` のフラグで、`auto` か MiB の数値を受ける
(数値は「auto の式に、検出値の代わりにその総 VRAM を渡す」の意味)。
`DENNO_AUTO_CONFIG` は `1` / `true` / `on` / `yes` / `auto` が真で、CLI が env に勝つ
(`auto_config::auto_requested`)。

auto は **モデルを組む前** に走る (expert の常駐予算は構築時に一度だけ読まれるため)。
起動ログに 1 行出る:

```
[auto-config] plan vram=... ram=... model=... Q4 (dense~...) -> weight=... expert_vram=... expert_gpu=... min_hit=... ram_tier=... pread=... chunk=... applied=... kept(explicit env)=...
```

`applied` が実際に書いた変数、`kept` が「明示 env があったので触らなかった」変数。
**明示 env は常に勝つ**。`DENNO_*` の値が空文字・空白だけなら未設定と同じ扱いで、
値が `auto` のときだけ「この変数は auto に任せる」として上書きされる
(`auto_config::is_unset` / `is_auto_sentinel`)。

### 1.2 auto の式

`auto_config::plan_gpu_config` の定数 [実装]:

| 項目 | 値 | 意味 |
|---|---|---|
| `PLAN_RESERVE_MB` | 1024 | CUDA context + JIT + 断片の余裕。`DENNO_VRAM_RESERVE_MB` にも同じ値を渡す |
| `WEIGHT_PCT` / `WEIGHT_CAP_MB` | 40% / 12,000 MiB | dense 重みの上限 |
| `EXPERT_PCT` / `EXPERT_CAP_MB` | 72% / 22,000 MiB | expert 常駐の上限 |
| `MIN_EXPERT_MB` | 1024 | これ未満なら expert 経路を切る |
| `MIN_HIT_ZERO_FROM_MB` | 12,288 | これ以上のカードでは `DENNO_EXPERT_GPU_MIN_HIT=0` を入れる |
| `RAM_TIER_MAX_VRAM_MB` / `RAM_TIER_MIN_RAM_MB` | 16,384 / 32,768 | 小さいカード + 大きい RAM のときだけ `DENNO_EXPERT_RAM_MB=auto` |

`usable = vram - 1024` として:

- dense: `min(usable*40%, 12000, dense_est)` (下限 256 MiB)
- expert: `min(usable*72%, 22000, expert_total)`。`1024 未満` または `usable*12% 未満` なら 0
- 小さいカードの prefill 資源: `vram <= 10 GiB` → pread 8 / chunk 1024、
  `vram <= 16 GiB` → pread 12 / chunk 1536、それ以上は既定 (memadapt) に任せる

**予算は上限であって予約ではない**。`DENNO_EXPERT_VRAM_MB` と `DENNO_GPU_WEIGHT_MB` の
和がカード容量を超えるのは既定の流儀で、実効上限はカードの空きで丸められる (§1.4)。
32 GB 級では手動の定番 (`expert 22000` / `weight 12000` / `min_hit 0`) と同じ値になる。

auto を使うには総 VRAM の検出が要る (`cuda` feature + libcuda + `cuDeviceTotalMem_v2`)。
検出できなければ `[auto-config] skip: ...` を出して **何も適用しない**。
その場合は `--vram-budget <MiB>` を渡す。

### 1.3 手動のノブ

| 変数 | 既定 | 意味 |
|---|---|---|
| `DENNO_EXPERT_VRAM_MB` | 0 (off) | expert を VRAM に常駐させる上限 (MiB) |
| `DENNO_EXPERT_GPU` | off | Q8_0 / Q5_1 / Q4_K / Q5_K の expert GEMV を GPU で回す |
| `DENNO_EXPERT_IQ_GPU` | off | IQ 系 expert の VRAM 経路 (上のゲートとは独立の入口) |
| `DENNO_GPU_WEIGHT_MB` | 0 (無制限) | dense 重みレジストリのバイト上限。超えたら最終使用が古い uid から解放 |
| `DENNO_VRAM_RESERVE_MB` | 1024 | カード予算から引く余裕 |
| `DENNO_VRAM_BUDGET_CARD` | on | 実効上限を「カードの空き − 余裕」でも丸める。`0` で env 予算だけに戻す |
| `DENNO_EXPERT_FILL_FREE` | off | env 予算を下限としてカードの空きまで使う (既定 off の理由は §1.5) |
| `DENNO_EXPERT_GPU_MIN_HIT` | 0.5 | 直近 64 トークンのヒット率がこれ未満なら CPU 経路。`0` = 常に GPU |

VRAM 経路は **`DENNO_EXPERT_VRAM_MB > 0`** と
**(`DENNO_EXPERT_GPU=1` または `DENNO_EXPERT_IQ_GPU=1`)** の両方が要る。
片方でも欠ければ従来の CPU 経路で、数値も呼び出し順も変わらない (fail closed)。
1 本が上限より大きい / 型やカーネルが非対応なら CPU へ落ちる。

`DENNO_EXPERT_VRAM_MB` を立てると、ホスト RAM の pread キャッシュ
(`DENNO_EXPERT_CACHE_MB`) は使われない (起動時に 1 行警告が出る)。
VRAM tier と RAM tier は併用できる (§1.6)。

### 1.4 カード予算による丸め

`DENNO_VRAM_BUDGET_CARD` (既定 on) は、expert の実効上限を

```
実効上限 = min(env の上限, カード総量 − 余裕 − expert 以外の常駐)
```

にする (`mem_audit::card_expert_cap`)。`DENNO_EXPERT_VRAM_MB=26000` のような
「容量を超える要求」を書いても、実際には空きで頭打ちになり確保失敗にならない。
`DENNO_VRAM_BUDGET_CARD=0` は従来の env 予算だけの挙動 (切り戻し用)。

### 1.5 `DENNO_EXPERT_FILL_FREE` が既定 off の理由

on にすると `DENNO_EXPERT_VRAM_MB` を下限として、カードの空きまで expert を載せる。
ただし実効上限の計算に「slab の確保済みチャンク」を使うため、チャンクを足すと
上限も同じだけ上がる帰還ループになる。実測では 14k cold prefill で空きを 17 MiB まで
使い切り、1〜3 MiB の確保が error 2 で失敗し続けた (prefill のステージングが取れず
CPU 経路へ落ちる)。既定 off のまま、必要なときだけ明示する。

### 1.6 RAM tier

| 変数 | 既定 | 意味 |
|---|---|---|
| `DENNO_EXPERT_RAM_MB` | off | expert を所有するホストバッファに常駐させる予算 (MiB)。`auto` で自動 |
| `DENNO_EXPERT_RAM_MARGIN_MB` | 8192 | `auto` で `MemAvailable` に残す余裕 |
| `DENNO_EXPERT_RAM_PIN` | on | pinned (`cuMemHostAlloc`) を試す。`0` でヒープ |
| `DENNO_EXPERT_RAM_FILL` | on | 起動時に予算まで充填する。`0` で使用時のみ |
| `DENNO_EXPERT_RAM_REBALANCE_SECS` | 30 | 走行中に利用回数の上位へ寄せ直す間隔。`0` で停止 |
| `DENNO_EXPERT_RAM_THREADS` | 12 | 充填のスレッド数 |

`auto` は `MemAvailable − (モデル総量 − expert 総量) − margin` を expert 総量で
頭打ちにする。非 expert バイトを引くのは、dense 重み / PLE の表 / KV が使う
page cache を tier が奪うと decode が壊れるため。page cache と違い追い出されず、
H2D の転送元にもそのまま使える (pinned)。引く順番は **VRAM tier → RAM tier → mmap**。

### 1.7 expert のホスト RAM pread キャッシュ

`DENNO_EXPERT_CACHE_MB` (既定 0 = 無効) は、`pread` した expert の生バイトを
**行列ごとに** ホスト RAM へ常駐させる。`DENNO_EXPERT_CACHE_STATS=1` で統計行が出る。
VRAM tier を立てているときは使われない (§1.3)。
上限は機材の `MemAvailable` を超えない範囲に丸められる。

---

## 2. prefix cache と expert キャッシュ

### 2.1 `serve-flash` の文脈スロット (`--kv-slots`)

| フラグ / 変数 | 既定 | 意味 |
|---|---|---|
| `--kv-slots` / `DENNO_KV_SLOTS` | 2 | 文脈 (KV / GDN 状態) のスロット数。1〜8、範囲外は 2 に落ちる |
| `DENNO_KV_SLOT_TOK` | `--max-ctx` | 1 スロットが持てるトークン数の上限 (8 未満は無視) |

`2` は従来の main + aux と同じメモリ量と挙動。`3` 以上で
「LCP (最長共通前置き) が最大のスロットを延長 / 空きへ載せる / 一番古いスロットを
捨てる (LRU)」の汎用経路になる。途中までの一致でも KV splice で継げる文脈は
再利用する。メモリの目安は **1 スロットあたり GDN 状態 117 MiB + KV 48 KiB/token**
[実装コメント]。スロットを増やすとその分だけ常駐が増える。

### 2.2 `serve` (27B 系) のマルチエントリ prefix cache

`--prefix-cache-tokens` / `DENNO_PREFIX_CACHE_MAX_TOKENS` が全エントリの重み合計の
上限。未指定ならコンテキスト長。重み = KV トークン数 + GDN 状態スナップショットの
トークン換算で、超過時は LRU で捨てる。選択はプロンプトとの LCP が最大のエントリ。
再帰モデル (GDN) の状態スナップショットは **prompt 境界の 1 点** なので、
新しいプロンプトがその prompt 全長を前置きとして含むときだけ復元し、
それ以外は full prefill に戻る。

### 2.3 expert の VRAM キャッシュ: TinyLFU と LRU

`DENNO_EXPERT_CACHE_POLICY` (既定 `tinylfu`):

- `tinylfu`: 4 bit カウンタの count-min sketch (4 行 × 16,384 = 32 KiB) で
  鍵 (層, expert) の推定頻度を持ち、**候補の頻度が犠牲者より低いときは載せない**
  (admission filter)。加算が `10 × width` に達するたびに全カウンタを半減する (aging)。
  1 回しか使わない expert を載せないことで、decode の作業集合を守る。
- `lru` / `0` / `off` / `false`: 従来の LRU (最終使用が最も古い 1 本を無条件に追い出す)。
  切り戻し用で、数値は従来と 1 バイトも変わらない。

`DENNO_EXPERT_GATE_TINYLFU` (既定 on) は、TinyLFU のときに
`DENNO_EXPERT_GPU_MIN_HIT` の降格を迂回する。TinyLFU は「載せない」判断をするので、
thrash していなくてもヒット率は 0.5 を割る。そこで P45 のゲートをそのまま掛けると
当たっている expert まで CPU に落ちる。実測では Q4_K_XL / 14k / VRAM 14,000 MiB で
`MIN_HIT=0.5` が 27.3 ms/tok、`MIN_HIT=0` が 16.5 ms/tok だった。
`lru` では迂回しない (従来どおりゲートを掛ける)。`0` で切り戻せる。

> `--vram-budget auto` が 12 GiB 以上のカードで `DENNO_EXPERT_GPU_MIN_HIT=0` を
> 入れるのはこのため (§1.2)。

---

## 3. prefill のチャンクとメモリ適応

### 3.1 チャンク幅 (`DENNO_PF_CHUNK`)

| 変数 | 既定 | 意味 |
|---|---|---|
| `DENNO_PF_CHUNK` | 1536 (ストリーミング経路のとき) | 層ループのチャンク (トークン数) |
| `DENNO_PF_CHUNK_BALANCE` | off | プロンプト長を均等割りして枚数を最小化 |
| `DENNO_PF_CHUNK_BALANCE_LONG` | off | 均等割りを 3 枚以上 (14k など) にも広げる |

`DENNO_PF_CHUNK` に `0` / `off` / 空 / 解析不能を書くと旧経路
(`DENNO_PREFILL_BATCH_CHUNK` と既存の既定) に戻る。

チャンクを広げると **同じ expert 重みを VRAM へ上げる回数が減る**。実測では
3,826 トークンの cold prefill が 512 → 1536 で 13.21 s → 9.89 s (-25%)、
14,018 トークンで 50.01 s → 35.44 s (-29%)。ただし 2048 まで上げると
GDN のデバイス連鎖の境界 (2,048 トークン) を踏み、ssm が遅くなって逆転する
(3,826 で +0.1%、14,018 で +39.6%)。1536 は境界の 3/4 で構造的に余裕がある。

均等割りは上限 2,047 トークンの下で枚数を最小にする。14k では 10 枚 → 7 枚になり
転送量が約 20% 減るが、壁時計は -2.9% にとどまるため既定 off
(プレフィルの支配項はホスト側の計算で、7 枚が GDN 境界の手前という上限)。

**丸め順の注意**: チャンクはトークンのまとめ方だけを変えるが、チャンク末尾が
`DENNO_PF_STREAM_MIN_TOK` (512) を跨ぐかどうかで経路 (GPU の m 行カーネル /
CPU の flat) が変わり、縮約順が動いて md5 が変わることがある。演算と expert 選択は
同じで、差は丸め順のみ。決定論が要る測定では §4.1 と併用する。

### 3.2 `DENNO_PF_MEMADAPT` (既定 on)

`/proc/meminfo` の `MemTotal` / `MemAvailable` / `SwapFree` から prefill の資源配分を
決める。`0` / `off` / `false` / `no` を明示したときだけ静的既定へ戻る。
計画は 1 秒に 1 回まで引き直し、段か配分が変わったときだけ
`[pf-budget] ...` を 1 行出す。

| 段 | 条件 (MiB) | 動き |
|---|---|---|
| Healthy | `avail >= max(12 GiB, total*18%)` | 恒等 (既定のまま)。total が 96 GiB 以上なら置き場 2 倍・タイル 2 倍・プール 1.5 倍 (上限 2048 / 16384 / 4096) |
| Tight | `avail >= max(6 GiB, total*8%)` | pread / ra / 置き場 / プールを約半分、タイル 4/5、pread 1 回を 512 KiB 以下 |
| Critical | それ未満、または swap 空き 25% 未満かつ `avail < total*25%` | 最小構成 (pread 4 / ra 8 / 置き場 64 MiB / プール 512 MiB / pread 256 KiB 以下) |

さらに匿名メモリの上限 `avail*60%` を掛け、プールはその 1/4、置き場は 1/8 を超えない
(残り 40% は page cache / スタック / 断片化の余裕)。

注: 既定では床 (プール 512 MiB / 置き場 64 MiB) がこの上限に勝つので、
`avail < 3,414 MiB` (プール) / `avail < 854 MiB` (置き場) では上限が破れる。
一番危ない領域でだけ cap を効かせたいときは `DENNO_PF_MEMADAPT_ANON_STRICT=1`
(床は 128 MiB / 32 MiB まで下がる)。

段の閾値は 1 本の線なので、`avail` が線を跨ぐと 1 Hz で段が往復しうる。
`DENNO_PF_MEMADAPT_HYST=1` で降格は即時・復帰は閾値の 10% 外側まで待つ
非対称ヒステリシスになる (swap 由来の Critical は即時のまま)。両方とも既定 off。

静的既定 (恒等の基準) [実装]: pread 16 本 / 先読み 32 本 / 置き場 512 MiB /
層優先タイル 5,120 トークン / 作業域プール 2,688 MiB / pread 1 回の長さ上限なし。

**明示の `DENNO_PF_*` はこの計算に勝つ**。たとえば `DENNO_PF_PREAD_THREADS=16` を
書けば memadapt は pread の本数を触らない。`DENNO_PF_MEMADAPT=0` は全部を静的既定へ
戻すので、A/B の基準に使える。

主要な `DENNO_PF_*` (すべて実装済み):

| 変数 | 既定 | 意味 |
|---|---|---|
| `DENNO_PF_PREAD_THREADS` | 16 | pread の同時本数。0..=64、`0` は共有プール |
| `DENNO_PF_RA_THREADS` | 32 | 先読みの並列度 |
| `DENNO_PF_RC_CACHE_MB` | 512 | 先読みの置き場 (匿名メモリ) |
| `DENNO_PF_LM_TILE` | 5120 | 層優先 prefill のトークンタイル |
| `DENNO_PF_POOL_MB` | 2688 | prefill の作業域プール |
| `DENNO_PF_PREAD_KB` | 1024 (広い塊 on のとき。off は 384) | pread 1 回の長さ |
| `DENNO_PF_STREAM_MIN_TOK` | 512 | これ未満の末尾チャンクは従来経路 |
| `DENNO_PF_MEMADAPT` | on | 空きメモリに応じた配分 (§3.2) |
| `DENNO_PF_MEMADAPT_HYST` | off | 段の復帰に閾値の 10% の余裕を要求 (task 937) |
| `DENNO_PF_MEMADAPT_ANON_STRICT` | off | 匿名上限を床より優先 (床 128 / 32 MiB、task 937) |

### 3.3 起動時の温めと診断

- `DENNO_START_PREFETCH` 系は起動時に expert の範囲を page cache へ読む。
  最初のリクエストが SSD から読み直す時間を `listening` の前に前払いする。
- `DENNO_START_DROP_CACHE=1` は **診断用**。冷えた機械を再現する (毎回 SSD から読む)。
- `DENNO_PF_DBG=1` は prefill の位相をバケット別に出す。バケットは **wall の分割**で
  あってカーネル時間への分割ではない (重複も漏れも無いが、同時実行は片方に寄る)。
- `DENNO_SEG_PROF=1` は層セグメントの内訳を出す。

---

## 4. 決定論・ベンチ・Intel Arc

### 4.1 Q4 の決定論モード (`DENNO_Q4_DETERMINISTIC`)

既定 off。`1` / `true` / `on` / `yes` だけが真。

decode では、VRAM に載っている expert は CUDA の `i` 昇順カーネル、載っていない
expert は CPU の AVX2 融合カーネル (8 レーン累積) で計算される。後者だけが `i` 昇順
でないため、ラウンド間で expert の常駐状態が変わると最下位ビットがずれ、同じ
プロンプトでも md5 が変わる。on にすると:

1. CPU: Q4_K / Q5_K の融合カーネル (1 行版と prefill バッチ版) を通さず、
   スカラー (`i` 昇順) にする。Q5_1 / Q6_K は元からスカラーなので何もしない。
2. GPU: GEMV を v2 に固定し (`DENNO_GEMV_V3` / `DENNO_GEMV_FP16` を無視)、
   CPU とビット一致しない seg 系の経路 (`DENNO_SEG_K` / `DENNO_SEG_LAYER`) を止める。

コストは「CPU ミスの GEMV がベクトル化を失う」ぶん [推定]。
**数値 (下位ビット) が変わる**ので、通常運用では off のまま、再現性が要る測定
(ベンチの決定論チェック、md5 比較) だけ on にする。

`--quant int8` (27B 系) は同 seed 同出力の決定論を保つが、fp32 と同一出力にはならない。

### 4.2 `denno bench`

起動中のサーバーに対して llama-bench 風の pp / tg を測る。プロンプトは
サーバーの `usage.input_tokens` で目標トークン数へ較正する。

```
denno bench --url http://127.0.0.1:8098 \
  --pp 512,2048,8192 --tg 128 --depths 0,2048,8192 \
  --rounds 3 --seed 801 --json bench.json
```

| フラグ | 既定 | 意味 |
|---|---|---|
| `--url` / `--api` | `http://127.0.0.1:8098` / `anthropic` | 対象と API (`openai` も可) |
| `--pp` / `--tg` / `--depths` | `512,2048,8192` / `128` / `0,2048,8192` | 測定するケース |
| `--rounds` | 3 | 各ケースの反復。**中央値**を採る |
| `--no-flush` / `--flush-tokens` | flush する / 0 = 自動 | ラウンド間の KV/prefix cache 押し流しを止める / 長さ指定 |
| `--seed` | 801 | プロンプト生成と較正の基準 |
| `--det-tokens` | 16 | 決定論チェックの生成トークン数 |
| `--reuse-depth` | 2048 | turn-2 KV reuse を測る文脈長 |
| `--json` / `--markdown` | なし / on | JSON レポート / Markdown 表示 |

出力は pp の `ttft_ms` / `pp_tok_s`、tg の `tpot_ms` (中央値) / `tg_tok_s`、
turn-2 KV reuse の TTFT と倍率、決定論の md5 と一致/不一致。

**測定の作法**:

- 1 回目は重みの初回アップロードを含む。`--rounds` を 3 以上にして中央値を採る。
- 比較する 2 条件は `--seed` と `--depths` を揃える。片方だけ `--no-flush` にしない。
- 決定論を見るときは `DENNO_Q4_DETERMINISTIC=1` を **両方の run** に入れる。
  §3.1 のとおりチャンク幅でも md5 は動くので、`DENNO_PF_CHUNK` /
  `DENNO_PF_CHUNK_BALANCE*` も固定する。
- `--request-timeout-secs` (serve) や `--max-queue-depth` が小さいと、長い pp が
  タイムアウト・過負荷応答になる。ベンチ中は十分に取る。

### 4.3 Intel Arc (`xe`) について

**この木に `xe` feature は無い。** ワークスペースと `denno-engine` の feature は
`default = ["par"]` / `par` / `test-utils` / `cuda` だけで、`--features xe` は
存在しないフラグである。README にあるとおり、Intel Arc Pro B60 向けの移植は
別ブランチ (`arc-port`) で進行中で、このツリーには含まれていない。

したがってこの木で Arc を対象にする場合の指針は次の 3 つに限られる:

1. Arc 版を使うときは、そのブランチのビルド手順に従う (この文書の範囲外)。
2. バックエンド非依存のノブ (§3 の prefill 資源、§4.1 の決定論) はそのまま意味を持つ。
   VRAM 予算 (§1) は CUDA の検出器 (`cuDeviceTotalMem_v2`) を使うので、
   auto が効かない場合は `--vram-budget <MiB>` で総量を明示する。
3. CPU 側の数値経路は同一なので、`DENNO_Q4_DETERMINISTIC` の意味 (スカラー `i` 昇順
   への固定) は移植先でも同じ契約になる。

Arc 固有のフラグや env をこの文書に書くことはしない (この木に実装が無いため)。

---

## 5. 早見表

| 症状 | まず見る |
|---|---|
| 起動時に VRAM を検出できない | `--vram-budget <MiB>` (§1.1) |
| expert が 1 本も VRAM に載らない | `DENNO_EXPERT_VRAM_MB > 0` かつ `DENNO_EXPERT_GPU=1` か `DENNO_EXPERT_IQ_GPU=1` (§1.3) |
| 確保失敗が続いて CPU に落ちる | `DENNO_VRAM_BUDGET_CARD` が on か、`DENNO_EXPERT_FILL_FREE` を on にしていないか (§1.4 / §1.5) |
| decode が往復して遅い | `DENNO_EXPERT_GPU_MIN_HIT=0`、`DENNO_EXPERT_CACHE_POLICY=tinylfu`、`DENNO_EXPERT_GATE_TINYLFU` が on か (§2.3) |
| 長い prefill が遅い | `DENNO_PF_CHUNK=1536`、`DENNO_PF_CHUNK_BALANCE=1` (§3.1) |
| swap を押し出して prefill が落ちる | `DENNO_PF_MEMADAPT` が on か、明示 `DENNO_PF_*` が強すぎないか (§3.2) |
| 同じプロンプトで md5 が変わる | `DENNO_Q4_DETERMINISTIC=1` + チャンク幅の固定 (§4.1 / §3.1) |
| 数字を比較したい | `denno bench` で `--rounds` を増やし中央値 (§4.2) |
