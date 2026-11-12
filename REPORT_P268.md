ADOPT: no t268a t268b t268c

# REPORT P268 — UD-Q4_K_XL の事前ロード / RAM tier / 本番の cold start

branch: br268 (br263 から。int-auto 62a9ad1 は br263 の祖先なので最初の merge は
不要)。lab のあと int-auto が b8e5de2 (task 265 = dev2 の prefill pool) と
89b7f8a (その赤テストの修正) まで進んだので順に取り込んだ。
commit: 9828537 + 44368eb + 9c377c5 + 2b940bc + 最終 merge + 最終。

## 0. 結論

- **採らない**。3 つの目標のうち、UD の warm decode -20% と cold start -30% は
  届かず、IQ3_S は RAM tier を入れると大きく悪化した。ゲートは既定 off のまま
  (`DENNO_EXPERT_STATS_DIR` / `DENNO_EXPERT_RAM_MB` を入れなければ今日と
  1 バイトも変わらない)。
- 実測 (lab t268b、UD-Q4_K_XL、decode、同一 run の baseline 比較):
  base 65.9 → pre 67.1 ms/tok (**+2.1%**、目標は -20%)。prefill 0.81 → 0.85 s。
- 実測 (lab t268c、IQ3_S、depth): base 19.4 → pre **32.7** ms/tok (**+68%**)。
  長文 (3,826 tok) の decode は 20.0 → 51.4 ms/tok (**+158%**)。RAM tier が
  page cache を追い出して t260c と同じ壊れ方をする。
- **唯一きいたのは「冷えた 2 本目」**: lab t268b の `b` (33 tok の別プロンプト) は
  cold 175 ms/tok → pre 67 ms/tok (**-62%**)。RAM tier が disk 読みを pinned DMA
  に変える効果は本物。ただし **1 本目 (a1) には出ない**。
- 理由は lab で測って分かった: **この機械では decode のミス読みが律速ではない**。
  `cold` COND (expert の page cache を 46 GiB 落とした) でも a1 は 66.1 ms/tok で
  base と同じ。UD の decode の `[q4prof] experts` は 39 ms/tok で、48 層あたり
  0.8 ms。これは層一括経路の 3 起動 + 2 同期の固定費と整合する (推定)。
  expert の読みは page cache でも disk でも同じ速さで隠れる。t261b の
  「46 ミス × 0.86 ms = 40/66 ms」は、少なくとも page cache の有無では説明
  できない (冷やしても 1 本目は変わらない)。

## 1. 前提 (task 文と lab の値。測り直していない)

- lab: SCENARIO=depth/depth8k、PROFILE=1、baseline COND = int-auto 既定 +
  `DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0
  DENNO_GPU_WEIGHT_MB=12000`。UD は 104 GB 中 expert 73,450 MiB、IQ3_S は
  78 GB 中 expert 47,962 MiB (drop のログの実値)。
- UD の 1 expert = 3,072,000 B (gate/up Q4_K 2560×640、down Q5_1 640×2560)。
  この値は新規テスト `plan_bytes_cover_k_and_legacy_types` で固定した。
- 実カードの有効な expert 予算は `target=18110 MiB` (22,000 - 予備 2,000 -
  dense の 12,000 との兼ね合い)。

## 2. 変えたこと (すべて既定 off のゲートの内側)

### 2.1 UD の事前ロードが載るようにする (expert_vram.rs)
- `acquire` の「載せてよい型か」の判定に **層一括カーネル
  (`kernels_moe_batch.ptx`) の対応型**を足した。UD の decode は層一括経路しか
  通らず、判定は per-expert カーネル (`kernels_moe.ptx`) しか見ていなかった。
- K 系 / legacy 型のバイト数は `ExpertShape::bytes()` が既に全部返す
  (Q4_K/Q5_K/Q5_1/Q8_0)。task 文の「s.bytes() が求まらない」は **この
  int-auto では再現しなかった** (t268a の VRAM 段は 1 本だけ載って予算で停止。
  理由はバイト数ではなく、下の 2.2)。

### 2.2 UD 級は VRAM 事前ロードをしない (qwen4exp.rs)
- t268a の実測: `[exp-plan] vram phase: loaded=1 experts 21999MiB (cover 0.002)
  in 50.61s stop=VRAM 予算に到達`。最初のリクエストの prefill が VRAM を自分の
  expert で埋めるので、事前ロードは 1 本しか入らず、18 GB の読みが最初の
  リクエストと disk を取り合うだけだった。
- 規則: `expert 総量 > MemAvailable` (UD 級) のときは VRAM へ載せず、RAM tier に
  寄せる。tier の先頭から `target / 1 本` ぶん (= decode の LRU が埋める本数) を
  除き、**VRAM の次に熱い層**を受ける (除かないと一番熱い層を抱えて 1 度も
  引かれない。t268a: miss_ram=14)。

### 2.3 RAM tier の auto 規則 (expert_ram.rs)
- `auto = MemAvailable − モデルの非 expert バイト − 余裕 (8,192 MiB)`、
  expert 総量で頭打ち。非 expert を引くのは、dense / PLE の page cache を
  tier が奪わないため (t260c の再発防止)。UD の実測値: model 106,177 MiB /
  expert 73,450 MiB → 非 expert 32,727 MiB、予算 16,224–16,935 MiB (5,428–5,666
  スロット)。IQ3_S では 14,299 MiB (7,327 スロット)。
- 明示値も同じ上限で頭打ちにする (auto と同じ規則)。テストで固定した。

### 2.4 RAM tier の充填を止めない (qwen4exp.rs)
- t268a の `pre` は `busy` 待ちで 1 リクエスト (10 秒) の間まるごと止まり、
  5,428 スロットのうち 384 本しか埋まらなかった (miss_ram=14 / miss_mmap=5,925)。
- 「1 チャンク (32 本) 読んで 2 ms 譲る」に変え、リクエストと並行に充填する。
  t268b では 16,302 MiB / 5,441 本を 13.95 秒で充填 (cover 0.090 = ちょうど
  VRAM の次の層)。

### 2.5 ミスの転送元を数える (expert_ram.rs / qwen4exp.rs)
- `[exp-ram]` 行に `miss_ram=` / `miss_mmap=` を足した。t268b の decode 1
  リクエストで miss_ram 3,455 / miss_mmap 6,437。tier が半分のミスを拾っている
  ことが数字で分かる (t268a は 14 / 5,925)。

### 2.6 cold start の再現 (expert_ram.rs / qwen4exp.rs)
- `DENNO_EXPERT_DROP_CACHE_ON_START=1` で、**起動プランの直前**に expert の
  page cache を `posix_fadvise(DONTNEED)` で落とす。dense / PLE は落とさない。
  記録の有無に依らず効くよう stats を読む前に置いた (lab の base 側も同じ条件で
  冷やせる)。mincore で前後の常駐を測って 1 行出す。
- IQ3_S: `resident=43,620MiB -> 4MiB in 1.43s` (144 範囲 / 47,962 MiB)。
  UD: `46,649MiB -> 1,185MiB in 1.85s`。

### 2.7 ログ
- `[exp-plan] vram phase:` を VRAM 段の終わりに出す (RAM 充填が長引いても
  結果が残る。t263c は最後まで出なかった)。`[exp-ram] tier:` に model /
  non_expert を足した。

### 2.8 dev1 (task 266) との接点
- 触った関数は `qwen4exp::acquire_expert` (ミス 1 本の転送元の選択) と
  `plan_experts_from_stats` / `expert_ram_fill` / `expert_vram::acquire`。
  dev1 の「ミス expert を page cache から早く読む」も同じ `acquire_expert` と
  decode の MoE ループに来る見込み。
- **残す順序**: VRAM ヒット → RAM tier (`HostExpert::new` の peek) →
  dev1 の early read / mmap。tier が先に受けるので、dev1 の early read は
  tier ミスのときだけ走る。両方が同じ関数に入る場合は、この順序を崩さずに
  両方残す (merge のときは `acquire_expert` の中の `note_miss_source` と
  `HostExpert` の 2 箇所を見る)。

## 3. 数値の扱い

- t268a / t268b / t268c の数字はすべて lab の実測 (RESULT.md の VERDICT 行と
  REQ 行をそのまま)。比較は同じ run の baseline COND だけ。最初の COND (b0) は
  捨て COND、最後に base2 を置いた。
- 「tier が 0.2 ms/ミス節約する」等の内訳は**推定**。実測できたのは
  miss_ram / miss_mmap の本数と、`b` リクエストの 175 → 67 ms/tok だけ。
- `rec` COND が base より 6% 速い (t268a/t268c とも) のは順序の癖。数値は
  採らない。

### t268a: UD-Q4_K_XL、decode、commit 9828537 (2.2/2.3/2.4 の前)

| COND | decode ms/tok | prefill s | 備考 |
| --- | --- | --- | --- |
| b0 | 66.0 | 0.80 | 捨て |
| base | 65.7 | 0.81 | |
| rec | 65.9 | 0.81 | 記録のみ |
| pre | 66.2 | **1.08** | VRAM 段 50.6 s で 1 本だけ、tier 384 本 |
| base2 | 65.8 | 0.81 | |

VERDICT pre vs b0 OK decode +0.2% prefill +35.0% md5 SAME errors 0

### t268b: UD-Q4_K_XL、decode、commit 44368eb (最終形)

| COND | decode ms/tok | prefill s | 備考 |
| --- | --- | --- | --- |
| b0 | 65.7 | 0.81 | 捨て |
| base | 65.9 | 0.81 | |
| rec | 66.2 | 0.81 | 記録のみ |
| cold | 66.1 | 0.81 | drop のみ (46,649→1,185 MiB) |
| pre | 67.1 | 0.85 | drop + tier 16,302 MiB / 5,441 本 (13.95 s) |
| base2 | 65.6 | 0.81 | |

VERDICT base vs b0 OK decode +0.3% prefill +0.6% md5 SAME errors 0
VERDICT rec vs b0 OK decode +0.8% prefill +0.0% md5 SAME errors 0
VERDICT cold vs b0 OK decode +0.7% prefill +0.0% md5 SAME errors 0
VERDICT pre vs b0 OK decode +2.1% prefill +6.2% md5 SAME errors 0
VERDICT base2 vs b0 OK decode -0.1% prefill +0.0% md5 SAME errors 0
VERDICT baseline b0 errors 0 text ok

リクエスト別 (ms/tok): a1 base 64 / cold 64 / pre 64。a2 base 66 / cold 66 /
pre 67。**b (33 tok、別プロンプト) base 71 / cold 175 / pre 67**。

### t268c: swift15-abl IQ3_S、depth、commit 44368eb

| COND | decode ms/tok | prefill s | 長文 decode | 長文 prefill s |
| --- | --- | --- | --- | --- |
| b0 | 19.5 | 0.62 | 22.6 | 10.05 |
| base | 19.4 | 0.61 | 20.0 | 7.25 |
| rec | 18.3 | 0.57 | 19.8 | 7.14 |
| cold | 19.4 | 0.60 | 22.8 | 12.27 |
| pre | **32.7** | 1.36 | **51.4** | 14.38 |
| base2 | 19.4 | 0.62 | 21.0 | 8.14 |

VERDICT pre vs b0 OK decode +68.3% prefill +119.4% md5 SAME errors 0
VERDICT_DEPTH pre c2000_cold vs b0 decode +127.7% prefill +43.1%
VERDICT_DEPTH pre c2000_turn2 vs b0 decode +85.2% prefill +54.1%
VERDICT_DEPTH cold c2000_cold vs b0 decode +1.1% prefill +22.1%
VERDICT baseline b0 errors 0 text ok

- pre のプラン: `vram=289 experts 18835MiB (cover 0.151)` (prefill が LRU を
  埋めるのでプランは 289 本で停止) + `ram=7328/7327 experts 14213MiB
  (cover 0.676) in 4.80s`、`paused=30752ms`。
- tier は 1 ミスも取りこぼしていない (`hit_rate=1.000`) のに decode が 2.6 倍。
  **pinned 16,384 MiB が dense の page cache を追い出した**のが原因 (t260c と
  同じ壊れ方)。cold (drop だけ、tier なし) は -0.4% なので、drop 自体は無害。

### テキスト (sane の確認)
- IQ3_S: `A **B-tree** (specifically, the **B+ tree** variant, which is the
  standard in modern databases) is a self-balancing tree data structure that
  maintains sorted data and allows searches, sequential access, insertions,
  and deletions in logarithmic time.` (md5 82ab0c06 / 624e0d4d / 1effbda4 /
  cd853fa2、全 COND 同一)
- UD: `# B-Trees: A Comprehensive Explanation ...` (md5 3a25e36e / 9be3fa06、
  全 COND 同一)。errors 0。

## 4. 本番の launch に入れる env

採らない判断なので**必須の追加は無い**。将来入れるなら:

```
# IQ3_S (swift15-abl): 事前ロードだけ。RAM tier は入れない
DENNO_EXPERT_STATS_DIR=/var/lib/denno/expstats
# 既存: DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000
#       DENNO_EXPERT_GPU_MIN_HIT=0 DENNO_GPU_WEIGHT_MB=12000

# UD-Q4_K_XL: 同上 + RAM tier を auto で (16 GB 前後)。ただし warm decode の
# 改善は 0-3% で、目標に届かない。入れるなら cold の 2 本目以降のため。
DENNO_EXPERT_RAM_MB=auto
```
- IQ3_S で `DENNO_EXPERT_RAM_MB` を**入れてはいけない** (t268c: +68%)。
- UD で `DENNO_EXPERT_RAM_MB` を明示するなら 16 GB 以下。22 GB 以上は
  IQ3_S の t260c (47 GB) と同じ事故になる (推定。今回は 16 GB で UD は無害、
  IQ3_S は 14.3 GB で有害だった。差はモデルの非 expert 量)。
- cold の 2 本目を速くしたいだけなら、UD は `DENNO_EXPERT_RAM_MB=auto` が
  効く (-62%、t268b の `b`)。1 本目は助かない。

## 5. テスト

- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`
  → **1142 passed / 0 failed** (100 ignored)。
- `--test kernel_invariants` → **90 passed / 0 failed**。
- 途中で int-auto の b8e5de2 が持ち込んだ
  `cuda::pf_exp_tc_tests::lm_reserve_gate_is_strictly_opt_in` の赤
  (task 265 で `tc_exp_from_value` を既定 on にしたのに、テストが既定 off を
  要求していた) が出たが、int-auto 89b7f8a がテスト側を直したので merge 後は
  全緑。私は触っていない。
- 追加: K 系 / legacy 型の expert バイト数 (UD 形で 3,072,000 B、端数は Err)、
  auto 予算の境界 (非 expert を引く / 明示値も同じ上限 / 空き不足で 0)、
  ミス転送元の数え上げと reset、drop ゲートの不活性。

## 6. GPU で確認してほしいこと

1. 本番の「モデルを切り替えた直後」で、`[exp-ram] miss_ram` が立つこと
   (= tier が効いている) と、そのとき `b` 相当 (別プロンプト) の ms/tok が
   disk 読みのときより速いこと。lab では 175 → 67 ms/tok。
2. 本番で `DENNO_EXPERT_DROP_CACHE_ON_START=1` を 1 回だけ使えば、
   「冷えた 1 本目」を lab と同じ手順で再現できる (resident の前後がログに出る)。
3. UD の warm decode を本当に 20% 縮めるには、層一括経路の 3 起動 / 2 同期
   (48 層 × 0.8 ms = 39 ms/tok) を減らすのが筋。expert の読みではない
   (t268b: cold と base が同じ)。
