ADOPT: no t265c

# REPORT_P265 — prefill の作業域を「長い prefill のあいだだけ expert から借りて返す」

task 265 (dev2)。branch `br265` (`br257` の tip `d42c66927` から。br257 は int-auto
`62a9ad16b` を merge 済みなので、「int-auto から切って br257 を merge した」のと同じ。
int-auto は最後まで `62a9ad16b` のままで、追加の merge は無かった)。

## 1. 結論 (3 行)

- **prefill の Target は満たした**: 14k cold 25.07 → 16.54 s (-34.0%、目標 -20%)、
  3,826 cold 7.27 → 3.34 s (-54.0%、目標 -30%)。短い decode +0.0%、errors 0、
  text sane。md5 DIFF は TC 経路のぶんだけ (task 257 と同じ)。
- **decode の Target は満たさない**: 14k 後の decode 18.52 → 19.40 ms/tok (+4.8%、
  許容 +2%)。t257b の +11.3% からは半減したが届かない。c2000 turn2 も +4.0%。
- **理由**: 層優先 prefill は expert をステージ経由で流すだけで decode の LRU を
  温めない (実測 `[exp-pf] experts=0`)。base は最後のチャンクを回した副作用で
  LRU がそのチャンクの expert で埋まる (ヒット率 0.974) が、こちらは prefill 前の
  内容が残る (0.956)。差は 8.5 ミス/token = 0.7 ms/tok で、これが +4% の正体。
  LRU の温めで直そうとした実験 (t265b) は VRAM を壊して失敗した。

## 2. 前提 (task / 過去の REPORT の実測値はそのまま使った)

- REPORT_P257 (lab t257b): 起動時に 2,688 MiB のプールを確保し expert 予算を
  19,312 MiB に削る形は、prefill を 25.13 → 17.51 s / 7.24 → 3.39 s にしたが、
  14k 後の decode が 18.15 → 20.2 ms/tok (+11.3%)。同じ予算でプールなしの cap も
  +8.0% なので、コストの本体は「expert 予算を削る」こと。
- 機序の手がかり (t257b): pool の `[q4prof]` は decode の ple が 3.3 → 182.7 /
  354.8 ms に跳ねた。予算を削ると PLE の置き場所が変わる。
- 14k のピークでカードはほぼ満杯 (base: dev used 31,967 / free 143 MiB)。
- 本番モデル相当の形: e=2560 / k=10 / hc=4 / ff=1024 / ne=512 / 48 層。
  1 expert = 3.072 MB、LRU 22,000 MiB = 7,161 本。

## 3. 変えたこと

### 3.1 プールを「借りて返す」形にした (`cuda.rs` / `qwen4exp.rs`)

- 起動時の確保と `DENNO_EXPERT_VRAM_MB` からの減算を**やめた** (`expert_route`)。
  起動直後の decode・短いプロンプト・PLE の置き場所は base と 1 バイトも変わらない。
- `pf_pool_init` / `pf_pool_begin` を `pf_pool_acquire(&Need) -> PfPoolGuard` /
  `pf_pool_release` に置き換えた。プールは `Mutex<Option<DevBuffer>>` で
  **1 prefill のあいだだけ** `Some`。区画は thread-local。
- 長い prefill の入口 (`forward_prefill_batch_resident{,_lm}` の先頭) で
  `DevBuffer::alloc_tagged(レイアウトの合計)` を 1 本だけ出す。足りない分は
  task 256 の経路が**チャンク単位で expert を追い出して**空ける
  (実測 14k: 2 チャンク / 23 本 / 510 MiB)。
- 終端でガードを落として `cuMemFree`。先に 3 つの置き場 (活性 / スクラッチ /
  PfDev) を落としてから解放する (非所有ビューの dangling を防ぐ)。早期撤退
  (pf_bail を含む) でもガードの Drop が同じ後始末をする。
- 予算は**触らない**。expert の実効上限は実空きから引き直される
  (`expert_vram::effective_expert_cap`) ので、返せば自動で 22,000 MiB に戻る。
- 別スレッドが借りている間は借りない。確保に失敗したときは従来の
  `alloc_tagged` に落ちる (リクエストは失敗させない)。

### 3.2 ゲート

- `DENNO_PF_POOL=1` (既定 off) — 既存のゲートをそのまま使う。
- `DENNO_PF_POOL_MB=<n>` (既定 2688) — 1 回に借りる上限。レイアウトが超えるときは
  借りない (fail open)。
- `DENNO_PF_POOL_MIN_TOK=<n>` (既定 2048) — **新ゲート**。これ未満のプロンプトは
  プールに触らない (短い prefill / decode の経路を変えない)。
- レイアウトの形は task 257 のまま (14k / TILE=5120 / TC で 2,498 MiB)。

### 3.3 数値

- **変えない**。作業域のアドレスが「都度確保」から「借りた 1 本の中」に移るだけで、
  カーネルへ渡すバイト列・起動順・丸め順は同じ。md5 の DIFF は task 250 の bf16
  tensor core 経路のぶん (`DENNO_PF_TC_EXP=1`)。base と base2/base3 は SAME。

## 4. lab (実測)

### 4.1 t265a (sha 60d491704、depth8k、model=swift15-abl-IQ3_S、load_before=1.64)

COND: toss (捨て) / base / pool / base2 / pool2 / base3。pool の env は
`DENNO_PF_LM_LAYER=1 DENNO_PF_TC_EXP=1 DENNO_PF_LM_TILE=5120 DENNO_PF_POOL=1`。

| cond | 短い decode | c2000 cold s | c2000 turn2 decode | c8000 cold s | c8000 cold 後 decode | c8000 turn2 decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| toss | 19.4 | 16.07 | 23.0 | 28.76 | 20.7 | 21.4 |
| base | 19.5 | 7.33 | 17.5 | 25.11 | 18.2 | 18.2 |
| base2 | 19.4 | 7.27 | 17.6 | 25.08 | 18.2 | 18.1 |
| base3 | 19.5 | 7.28 | 17.5 | 25.07 | 18.1 | 18.2 |
| pool | 19.4 | 3.36 | 18.6 | 16.62 | 19.6 | 19.8 |
| pool2 | 19.5 | 3.35 | 17.7 | 16.45 | 18.7 | 18.2 |

### 4.2 t265c (sha 36894283f = 最終ツリー、load_before=1.63)

| cond | 短い decode | c2000 cold s | c2000 turn2 decode | c8000 cold s | c8000 cold 後 decode | c8000 turn2 decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| toss | 19.6 | 7.22 | 17.6 | 25.11 | 20.0 | 19.8 |
| base | 19.4 | 7.23 | 17.7 | 25.07 | 18.1 | 18.1 |
| base2 | 19.4 | 7.24 | 18.1 | 25.03 | 19.4 | 19.4 |
| base3 | 19.5 | 7.24 | 17.6 | 25.04 | 18.1 | 18.2 |
| pool | 19.5 | 3.33 | 18.8 | 16.63 | 20.4 | 20.4 |
| pool2 | 19.6 | 3.33 | 18.4 | 16.44 | 18.9 | 18.3 |

- 最終ツリーは t265a と同じ数字を再現した (prefill 16.5 s、14k 後のヒット率
  0.956、ピーク free 9 MiB)。ゲート整理 (borrow_plan) は挙動を変えていない。
- VERDICT (t265c): `pool c2000_cold vs toss decode +2.9% prefill -54.0%`、
  `pool c8000_cold vs toss decode +2.1% prefill -33.8%`、
  `VERDICT baseline toss errors 0 text ok`、pool の text は `looks ok`。
  TXTD (14k 後の答え): 「ご提示いただいた文書「第5巻 prefill の速度改善」の要点を、
  技術的な背景と具体的な改善プロセスを含めて詳細に解説します。…### 1. 核心的な
  発見：ボトルネックは「計算」ではなく…」— base と同じ趣旨で筋が通っている。

### 4.3 まとめ (base 6 サンプル vs pool 4 サンプル、2 ラウンド)

| 指標 | base | pool | 差 |
| --- | ---: | ---: | ---: |
| c2000 cold prefill | 7.265 s | 3.343 s | **-54.0%** |
| c8000 cold prefill | 25.067 s | 16.535 s | **-34.0%** |
| 短い decode | 19.45 | 19.48 | +0.2% (雑音) |
| c2000 cold 直後 decode | 20.23 | 20.38 | +0.7% |
| c2000 turn2 decode | 17.67 | 18.38 | +4.0% |
| c8000 cold 直後 decode | 18.52 | 19.40 | **+4.8%** (未達) |
| c8000 turn2 decode | 18.37 | 19.18 | +4.4% |
| errors | 0 | 0 | — |

### 4.4 VRAM とヒット率 (t265c pool_1 / base_1 の生ログ)

| 時点 | dev used | free | expert ヒット率 |
| --- | ---: | ---: | ---: |
| 14k prefill の直前 (request #5 の後) | 30,207 MiB | 1,903 MiB | 0.865 (c2000 の decode) |
| 14k prefill のピーク (pool) | 32,101 MiB | **9 MiB** | — |
| 14k prefill のピーク (base) | 31,967 MiB | 143 MiB | — |
| 14k prefill の後 (request #7 の後, pool) | 30,847 MiB | 1,263 MiB | 0.956 |
| 14k prefill の後 (request #7 の後, base) | 31,129 MiB | 981 MiB | 0.974 |
| 14k turn2 の後 (pool / base) | — | — | 0.970 / 0.973 |

- `[pf-pool] expert から作業域 2498 MiB を借りた (タイル 5120 / 14018 トークン /
  空き 2137 → 149 MiB)` → 必要だった追い出しは 2 チャンク / 23 本 / 510 MiB。
- `[pf-pool] 作業域 2498 MiB を expert に返した (空き 2657 MiB)`。
- 3,826 の prefill は借りるだけで追い出し 0 (`空き 5607 → 3955 MiB`)。
- 14k 後のミスは 2,502 本 / 5,708 層 = 1.634 ms/tok (base 1,362 本 / 5,232 層 =
  0.933 ms/tok)。差 0.7 ms/tok が decode の +4% の正体。

### 4.5 t265b (sha 5221bcfd7、warm 実験) — 失敗、丸ごと削除した

decode のヒット率を戻すために「末尾 1,536 トークンの expert を LRU へ入れ直す」
(`DENNO_PF_WARM_TOK`) を実装して測った (3 ラウンド目を使う前の試行)。

- `[pf-warm] 末尾 1536 トークンの expert を 16254 本 (31020 MiB) LRU へ戻した`。
  想定 (約 5,000 本) の 3 倍で、prefill の終端は free 35 MiB。以降の decode は
  255-292 ms/tok、答えは `!!!!!!!!...` (text BROKEN、errors 0)。
- 原因 (推定): (1) 選ぶフィルタが「タイル内の行」なので 1 タイルのプロンプトでは
  40% のトークンぶん = 338 本/層になり、LRU 容量 (7,161 本) の 2 倍以上を
  出し入れした。(2) 挿入が MoE カーネルの直後 (非同期実行中) に走り、追い出した
  チャンクを新しい slab が再利用するため、実行中の読みと衝突しうる。
- warm は丸ごと削除した (コミット 36894283f)。**この機構は未解決**。次にやるなら
  「MoE の完了を待ってから」「LRU 容量以下の本数で」。

## 5. 採否

- 判定の規則 (task): 14k cold -20% 以上、3,826 cold -30% 以上、短い decode と
  14k 後の decode が +2% 以内、errors 0、text sane。
- 結果: prefill は -34.0% / -54.0% で達成。短い decode +0.2%、errors 0、text sane。
  **14k 後の decode +4.8% で未達** (2 ラウンド 4 サンプルで +4.8 / +4.4 / +4.0%。
  最良のサンプルでも +2.9%)。
- よって **ADOPT: no** (`t265c` は最終確認のラウンド)。ゲートは既定 off のまま
  (`DENNO_PF_POOL` 未設定 = 従来経路と 1 バイトも変わらない)。コードは残す:
  借りて返す形自体は prefill の Target を満たし、次の手 (LRU の温め方) の土台に
  なるため。

## 6. テスト

- 関門: `cargo test --release -p denno-engine --features cuda --lib --
  --test-threads=1` → **1,119 通過 / 0 失敗**。
  `--test kernel_invariants` → **90 通過 / 0 失敗**。
  `CARGO_TARGET_DIR=/tmp/target_p265b`。
- 追加: `pf_pool::borrow_plan` (ゲート off / しきい値 / 上限 / 形 0 で借りない)、
  3,826 のレイアウトが上限内、`DENNO_PF_POOL_MIN_TOK` の解釈、
  `cuda::tests` の 3 本 (context 無しで fail open、短いプロンプトは借りない、
  既定 off では長くても借りない)。

## 7. 見積り (lab で測っていないもの)

- 借りる量はレイアウトの合計そのもの (14k で 2,498 MiB、3,826 で 1,651 MiB)。
  `DENNO_PF_POOL_MB` は上限としてだけ効く。
- decode のコスト内訳 (推定): 予算を削らないので cap (t257b +8.0%) のぶんは無く、
  残るのは「prefill が LRU を温めない」ぶんだけ。実測の差はヒット率 0.956 vs
  0.974 = 8.5 ミス/tok。
- 14k prefill のピークで free 9 MiB まで行く (base 143 MiB)。prefill 中のプール外の
  確保 (attention のミラー) が 4 回失敗し、そのたびに 255 MiB のチャンクを
  追い出している。プールに 256 MiB の余裕を持たせれば減らせる見込み (未測定)。

## 8. 次にやるなら (GPU で確認してほしいこと)

1. **LRU の温め方**。層優先 prefill のあと decode の LRU が「prefill 前の内容」の
   ままなのが +4% の原因。候補: (a) 最後のタイルだけ `acquire` 経路で回す、
   (b) ステージから LRU へ D2D で入れる (H2D しない)、(c) 最後のチャンクの expert を
   `touch_many` で保護する。t265b の失敗から、**MoE の完了を待つ**ことと
   **LRU 容量以下の本数**が条件。
2. `DENNO_PF_POOL_MB` に余裕を持たせて prefill 中の確保失敗を減らす
   (14k のピーク free 9 MiB → 256 MiB)。
3. `DENNO_PF_LM_TILE` を 4096 にするとプールは約 2.3 GB になり、追い出しは
   さらに減る見込み (そのぶん H2D が 3 → 4 タイルで増える。未測定)。
