ADOPT: yes t307c

# REPORT P307 — IQ3_S decode: dense GEMV の v5 化、共有 expert の層キャッシュ、
# ルータ GEMV を mix の鎖へ

branch: br307 = int-auto `02171944a` (br300 = PLE prefill の merge 後) から。
br299 は int-auto に未 merge だったので**取り込んだ** (`DENNO_MIX_NORM_HOST` と
台帳の副区間がこのタスクの材料)。lab: t307a / t307b / t307c (3 回、上限内)。
最終 commit の直前に int-auto `0179e968a` (tasks 308/309) を merge した (下記 5)。

## 0. 前提 (タスクと過去 REPORT の値。測り直していない)

- baseline COND = int-auto の既定 + `DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000
  DENNO_EXPERT_GPU_MIN_HIT=0 DENNO_GPU_WEIGHT_MB=12000`。
- モデル (swift15-abl IQ3_S): n_embd 2560、48 層 (ssm 36 / full 12)、
  full の q|gate|k|v = 13312 (= 8 層は Q8_0 の融合 1 本、4 層は Q5K/Q4K 12288 +
  Q8_0/Q5K 512 x 2)、out = 6144→2560 は Q8_0 (16.7 MB)、shared expert は 640/640/2560。
- P299 の台帳 (短い decode、ms/token): mix 3.51 / ssm 3.38 / experts 3.51 /
  full 1.6 / router 1.08 / head 0.73 / shared 0.63 (うち `sh_begin` 0.45) /
  loop 1.87。GPU busy 11.0/21.7。

## 1. 変えたこと (4 本のゲート)

### 1.1 `DENNO_GGUF_GEMV_V5` — dense gguf GEMV も v5 を先に試す (既定 on)

lab t299b の `[dec-kprof-avg]` full 表で、**同じ q8_0 GEMV が v5 (26 us) と
v3 (49 us) に割れていた**。v3 側は full 層がホスト経路へ落ちた回の
`GgufLinear::gemv` → `maybe_gpu_gguf_gemv_into` → `moe_gemv_dev` で、
`gemv_kernel_mode()` が v5 を知らないため。入口が違うだけで 1.9 倍の差。

- 実装: `moe_gemv_dev_with_pref(.., prefer_v5)` を足し、`maybe_gpu_gguf_gemv_into`
  だけ v5 → (落ちたら) 従来の v3/v2 の順に引く。形が合わない型・k は従来どおり。
- 数値: 丸め順が変わり得る (v5 のアキュムレータ 4 本)。決定性は保つ。
  **実測は md5 全 COND で SAME** (下記 2.3)。見込みは外れた側で、値は変わらなかった。

### 1.2 `DENNO_SH_BEGIN_CACHE` — 共有 expert の plan と常駐重みを層ごとに控える
### (既定 off のまま。lab で勝たなかった)

`shared_expert_begin` は毎層 3 本の `weights_for_gguf` (登録簿の mutex + LRU の
touch/pin、消費後の unpin 3 本) と plan 3 本を作り直していた。uid の組は層ごとに
固定なので、最初の 1 回で作った計画と `GgufWRef` を控える。

- 落とし穴 (P299 の指摘): pin を握り続けること。共有 expert は 1 層 3 本
  (Q8_0 1.74 MB x 3) x 48 層 = **251 MB** で、どのみち毎トークン読む。
- 世代番号 `GGUF_REG_GEN` を `gguf_unregister` (モデル組み直し) で進め、
  キャッシュは世代が変わったら丸ごと捨てる (解放済みの `&'static` を踏まない)。
- 数値: 同じ計画・同じ重みなので**ビット一致**。
- **lab t307a: `sh_begin` 0.42 → 0.41 ms/token = 効いていない**。中身は plan と
  `weights_for_gguf` ではなく 4 発の起動と H2D だった (0.45 ms の見積りは外れ)。
  よって既定 off のまま (真理値表だけテストで固定)。

### 1.3 `DENNO_ROUTER_IN_MIX` — ルータ GEMV を mix の鎖の末尾へ (既定 on)

router 行は 1.08 ms/token (gpu 0.64 / sync 0.63 / cpu 0.39)。ルータの入力は
mix の出力 (`mixed`) そのものなので、**mix の鎖 (chain3) の hc_mixed の直後**に
ルータ GEMV を 1 発積み、D2H を mixed の D2H と同じ同期でまとめる。

- 消えるもの: ルータ用の H2D (10 KB) と、鎖の後に空のストリームへ撃つ起動 +
  同期 1 回 (層あたり 5〜6 us)。
- 入力は鎖の中間バッファ `dm` (= `mixed` のデバイスコピー) をそのまま読む。
  カーネル (`dno_matvec_f32_r` 既定) も重みも同じ = **ビット一致**。
- 安全側: 層 0 の入口 (トークンの境目) で印を落とし、`begin` が消費したら
  印を消す。uid / k / n が揃わなければ従来経路へ (fail closed)。
- decode の mix 本線は chain3、次の候補が chain2。**両方**に刺した。
- 3/n: 控え (`ROUTER_FILLED`) が無い回は `ROUTER_SYNCED` を必ず落とす
  (キャッシュに残った古い logits を写さない)。

### 1.4 診断 `DENNO_FULL_DEC_DBG` (既定 off)

full 層が従来経路へ落ちた理由を 255 回ごとに 1 行で出す (数値・経路は不変)。

## 2. lab

### 2.1 t307a (SHA 79c5bf463、6 COND、load_before 1.12)

`SCENARIO=depth8k PROFILE=1`。base と base2 (同一 env) が **17.9 / 17.9** で揃った回。

| cond | 中身 | 短い decode ms/tok | c2000_turn2 | c8000_turn2 | prefill c2000 / c8000 |
| --- | --- | --- | --- | --- | --- |
| toss | baseline の写し (先頭) | 17.9 | 17.7 | 18.0 | 3.19 / 10.24 |
| base | baseline | **17.9** | 16.7 | 16.2 | 3.13 / 9.90 |
| safe | sh cache + router + norm host | 17.3 | 15.3 | 15.7 | 3.12 / 8.01 |
| v5 | gguf v5 だけ | 17.8 | 16.1 | 17.5 | 3.13 / 8.03 |
| all | 4 本とも | **17.1** | 15.8 | 17.0 | 3.13 / 9.18 |
| base2 | baseline の繰り返し (末尾) | **17.9** | 16.3 | 17.5 | 3.13 / 9.62 |

- 短い decode: base 17.9 → all 17.1 = **-4.5%**。v5 単独 -0.6%、safe 単独 -3.4%。
- 14k 後 (c8000_turn2) は base 16.2 / base2 17.5 と 2 本の baseline が 6.9% 離れ、
  この回では判定不能だった (t307b/c で取り直した)。
- errors 0、テキスト健全、md5 は全 COND で SAME。
- 台帳: router 1.15 → **0.18 ms/token** (発行 0 / 同期 0 / H2D 0、mix の d2h が
  97 → 145 = +48)。full の gpu 1.19 → 1.03 (v5 の効果 -0.16 ms/token)。

### 2.2 t307b (SHA 79c5bf463、6 COND、load_before 1.63)

COND = toss / base / nh / rmx / final / base2。短い decode の median:
toss 17.9 / base 17.9 / nh 17.6 / rmx 17.5 / **final 17.2** / base2 17.9。

```
VERDICT base vs toss OK decode +0.0% prefill +0.0% md5 SAME errors 0
VERDICT nh vs toss OK decode -1.3% prefill -1.7% md5 SAME errors 0
VERDICT rmx vs toss OK decode -2.0% prefill +0.0% md5 SAME errors 0
VERDICT final vs toss OK decode -3.7% prefill -1.7% md5 SAME errors 0
VERDICT base2 vs toss OK decode +0.2% prefill +0.0% md5 SAME errors 0
VERDICT_DEPTH final c2000_cold vs toss decode -4.5% prefill +0.3%
VERDICT_DEPTH final c2000_turn2 vs toss decode -4.4% prefill -1.3%
VERDICT_DEPTH final c8000_cold vs toss decode -11.1% prefill -17.8%
VERDICT_DEPTH final c8000_turn2 vs toss decode -10.7% prefill -2.1%
VERDICT baseline toss errors 0 text ok
```

3 本のゲート (router / v5 / norm host) の全部入りで -3.7%。t307a の -4.5% より
小さいが、base2 が +0.2% の回なので差は床のぶれ。target の 4% には届いていない。

### 2.3 t307c (SHA 0bae44f80 = 既定 on の commit、4 COND、load_before 1.81)

既定を on にした状態そのものを測る確認回。`off` は 3 本を明示的に 0 にした
baseline、`ship` は既定のまま (= on)、`off2` は off の繰り返し (末尾)。

| cond | 中身 | 短い decode ms/tok | c2000_turn2 | c8000_turn2 | prefill c2000 / c8000 | md5 |
| --- | --- | --- | --- | --- | --- | --- |
| toss | baseline の写し (先頭) | 17.9 | 16.2 | 16.2 | 4.18 / 11.04 | SAME |
| off | baseline (3 本を 0) | 17.9 | 15.9 | 16.2 | 3.13 / 9.63 | SAME |
| ship | **既定 (3 本とも on)** | **17.0** | **15.2** | **15.6** | 3.12 / 7.98 | SAME |
| off2 | off の繰り返し (末尾) | 18.0 | 15.9 | 16.2 | 3.12 / 7.99 | SAME |

```
VERDICT off vs toss OK decode +0.0% prefill -1.7% md5 SAME errors 0
VERDICT ship vs toss OK decode -4.6% prefill -0.8% md5 SAME errors 0
VERDICT off2 vs toss OK decode +0.9% prefill -0.8% md5 SAME errors 0
VERDICT_DEPTH ship c2000_cold vs toss decode -5.6% prefill -25.2%
VERDICT_DEPTH ship c2000_turn2 vs toss decode -6.0% prefill -6.0%
VERDICT_DEPTH ship c8000_cold vs toss decode -5.0% prefill -27.7%
VERDICT_DEPTH ship c8000_turn2 vs toss decode -3.8% prefill -2.1%
VERDICT baseline toss errors 0 text ok
```

- **短い decode: off 17.9 / off2 18.0 (床は 0.6% 以内) に対して ship 17.0 =
  -4.6〜-5.6%。target の 4% を満たした。**
- 14k 後 (c8000_turn2) も off / off2 16.2 に対して ship 15.6 (-3.8%)。悪化なし。
  2k 後 (c2000_turn2) は 15.9 → 15.2 (-4.4%)。
- prefill: c2000_cold は 3.12〜3.13 で不変。c8000_cold は off 9.63 / off2 7.99 /
  ship 7.98 で、off の 9.63 と先頭 toss の 11.04 は既知の深い prefill のぶれ
  (同一コードの off2 が 7.99)。**off2 と ship は 0.1% 差 = 悪化していない。**
- errors 0、md5 は ship を含めて全 COND で SAME
  (`a:65ededc3 b:f18c75fc c2000_cold:d4b6eefb c2000_turn2:8ff9345f
  c8000_cold:e2292eb6 c8000_turn2:cc2c0e8f`)。
- テキスト (ship): `TXT A **B-tree** (specifically, the **B+ tree) is the
  fundamental data structure used by most modern relational databases ...` /
  `TXTB Rust の「所有権（Ownership）とライフタイム（Lifetimes）は、Rust の最も
  重要な概念であり、同時に初心者が最もつまずきやすい部分です。` /
  `TXTD ご提示いただいた文書は、LLM推論エンジン（おそらく \`denno-engine）の
  **Prefill（事前計算）性能最適化の技術的記録です。` — off / off2 と同一。

### 2.4 採否 (task の規則: target 達成・他が 2% 超悪化なし・テキスト健全)

- 短い decode -4.6% (t307c、同一 run 内の off/off2 が床) = **target 4% を達成**。
- 14k 後 decode は -3.8% (速い側)、prefill は off2 と同値、errors 0、md5 SAME。
- したがって **ADOPT: yes t307c**。既定 on は 4/n (`0bae44f80`) で入れてあり、
  lab はその commit を測っている。1.2 の `DENNO_SH_BEGIN_CACHE` だけは
  既定 off のまま (勝たなかったため)。

## 3. テスト

最終 commit の内容 (int-auto `0179e968a` の merge 後) で:

- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`
  → 1206 passed / 0 failed / 102 ignored (ゲートの真理値表 4 本を含む)。
- `cargo test --release -p denno-engine --features cuda --test kernel_invariants`
  → 95 passed / 0 failed。
- 既知の赤 (`qwen4exp_decode spec_batch_stops_at_eos`、`qwen4exp_params` の
  validate 2 件) は踏んでいない。

## 4. 数値の扱い

- 1.1 だけ丸め順が変わり得る (v5 の加算順)。実測は全 COND で md5 SAME。
- 1.2 / 1.3 は経路と発行順だけで値は不変。
- ゲートはすべて明示的な真偽値でだけ切り替わる (真理値表をテストで固定)。
  既定 on の 3 本は「明示的な偽値だけ off」、`DENNO_SH_BEGIN_CACHE` は
  「明示的な真値だけ on」。

## 5. int-auto の取り込みと未追跡ファイル

- lab の後で `int-auto` が `02171944a` → `0179e968a` に進んだ (tasks 308/309)。
  merge した (`merge: int-auto (tasks 308/309) into br307`)。衝突なし (ort の
  自動 merge)。308 は decode のルータと full 出力をデバイス側に寄せる変更で、
  307 の `DENNO_ROUTER_IN_MIX` (mix の鎖の末尾でルータ GEMV) とは別のゲート
  (`DENNO_DEC_ROUTER_EARLY` など) なので、両方 on でも経路は競合しない
  (ルータの控えが無い回は従来経路へ落ちる作り)。merge 後にテストを再実行して緑。
- 未追跡 4 本は**このタスクの成果物ではない**ので commit していない:
  `examples/attn_q8_bench.rs` (task 87、mtime 11-01)、
  `examples/mix_host_math_bench.rs` (11-06、ヘッダに「commit しない」)、
  `examples/t5c_alloc_probe.rs` / `t5c_pf.sh` (T5c、10-19)。mtime もヘッダも
  307 の作業期間 (11-14) と一致しない。

## 6. GPU で確認してほしいこと

- t307c の ship COND が本番の既定 (env 未設定) と一致すること。
- 実運用の長い文脈で `DENNO_ROUTER_IN_MIX` の控えが別層へ漏れないこと
  (`DENNO_FULL_DEC_DBG=1` と台帳の router 行 0.18 ms/token が目安)。
- `DENNO_DEC_ROUTER_EARLY` (task 308) と同時 on での短い decode の歩留まり
  (このタスクでは測っていない)。
