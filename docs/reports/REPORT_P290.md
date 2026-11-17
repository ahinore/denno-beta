ADOPT: yes t290c

# task 290 — 長いプロンプトの prefill が VRAM 不足で落ちるのを直す (UD-Q4_K_XL)

task 290 (dev2)。branch `br290` は開始時の `int-auto` (`7890d121e`) から。
測定はすべて GPU ラボ。`SCENARIO=depth8k`、`PROFILE=1`、3 ラウンド。
ラウンド 1 `t290a` = UD-Q4_K_XL、ラウンド 2 `t290b` = IQ3_S、ラウンド 3 `t290c` = UD-Q4_K_XL の繰り返し。
測定した木は 3 ラウンドとも `ff26091ee` (`7890d121e` + このタスクの 1 コミット)。

## 1. 結論 (3 行)

- **UD-Q4_K_XL の 14k cold は 8 回走って 8 回とも答え、テキストも sane。**
  旧い木 (`t286c` の 4 COND) は全部 API エラー (`層 41: 3 MiB / 空き 3 MiB`) だった。
- **IQ3_S の 14k cold は 8.65 s (同じ run の base は 10.17 / 11.93 / 12.79 s) で悪化なし。**
  md5 は 6 出力すべて SAME、短い decode は 18.0 対 17.9 ms/tok (+0.6%)。
- 原因は 2 つ。(a) プールが固定上限 (4,864 MiB) だけで借りる量を決め、実空きを見ていなかった。
  (b) 確保失敗後の追い出しが「不足分 = 0」で 1 本も走らず、断片化した数 MiB の確保で諦めていた。

## 2. 前提 (タスク文と lab のログから)

`t286c` (dev3 `9ff968b` = `int-auto c5cb2939c` + task 286、UD-Q4_K_XL、depth8k) の 14k:

```
[pf-pool-big] 形を上げた (チャンク 1536 → 2047 / タイル 5120 → 14018 = 3 → 1 パス / 作業域 4779 MiB)
task256: 確保失敗 (要求 4779 MiB / 空き 2919 MiB) → 追い出して再試行 (released 2038 MiB)
[pf-pool] 作業域 4779 MiB を借りた (空き 2919 → 185 MiB / 追い出し 501 本)
[pf-lm] 層 0: dev MoE を諦めた (理由: gate) → CPU (タイル 14018 単位)
task256: 確保失敗 (要求 99 / 32 / 32 MiB) → 追い出して再試行
→ API エラー「層 41: CUDA cuMemAlloc が失敗: error 2 (要求 3 MiB / 空き 3 MiB / 総 32110 MiB)」
```

- 失敗したのは**層 41 の ssm の 3 MiB の確保**。そのとき空きは 3 MiB で、要求と同値。
  `plan_eviction` の不足分は `3 - 3 = 0` なので 1 本も空けず、`emergency_free` も 0 を
  返して `Err` になっていた (`task256` のログも出ない)。
- カードを埋めていたのは expert の slab (79 チャンク = 20,121 MiB) + プール 4,779 MiB +
  w:gguf 4,270 + q4attn mirror 689 + その他。**プールが空きを 185 MiB まで使った**のが発端。
- UD-Q4_K_XL の専門家の型は Q4_K 系で `exp_iq::slot` が無い → `iq_expert_mode` が false →
  層優先の dev MoE が使えず (`理由: gate`)、ホスト経路がステージングを自前で確保する。
  この経路は IQ3_S (IQ3_S/IQ4 系 = dev MoE が使える) より 2 GB 以上多く要る。

## 3. 変えたこと (1 コミット)

1. **`pf_pool::pool_budget`** — 借りる上限を
   `min(固定上限, 実空き + 追い出せる expert - 余白)` で決める。
   余白は `DENNO_PF_POOL_RESERVE_MB` (既定 1,024)、dev MoE が使えないモデルは
   `DENNO_PF_POOL_RESERVE_CPU_MB` (既定 3,072)。入らなければ **パス数を増やす**
   (`choose_big` の候補順がそのまま効く)。`=1` で従来の挙動に戻る (A/B 用)。
2. **`cuda::pf_pool_acquire_capped`** — 借りる前に余白のぶんをチャンク単位で先に空ける。
   借りられなかったときも、先に空けた expert を `PF_POOL_EVICTED` に残して
   prefill の終端で LRU へ戻す (戻す側は `big` のときだけ、をやめた)。
   ログは `[pf-pool] 余白 N MiB を残すため先に空けた (...)` と
   `[pf-pool-big] 上限 (実空き a + 追い出せる expert b - 余白 c = d / 固定上限 e)`。
3. **`expert_slab::plan_eviction_forced`** — 確保が**既に失敗した**あとの plan は、
   不足分が 0 でも冷たいチャンクを 1 本選ぶ。`DevBuffer::alloc_tagged` の失敗経路だけ
   これを使う (`DENNO_ALLOC_FORCE_CHUNK`、既定 on、`=0` で従来動作)。
4. `expert_vram::resident_bytes` (LRU の実バイト数、`try_lock`) を追加。
5. テスト: `pool_budget_leaves_the_reserve` / `plan_falls_back_to_more_passes_when_the_budget_is_small` /
   `reserve_defaults_and_parsing` / `plan_eviction_forced_always_frees_one_chunk` /
   `alloc_force_chunk_env_truth_table`。

変えたのは確保の計画と失敗経路だけで、カーネル・数値の順序は触っていない。
UD の 14k で md5 が変わるのは、タイルが 1 パス → 2 パスになり、CPU の MoE / shared の
GEMM の行数が変わるため (丸めの差。テキストはどちらも sane)。

## 4. 数値 (lab。すべて実測)

### t290a — UD-Q4_K_XL、4 COND × 2 ラウンド (sha ff26091ee)

| cond | 14k cold prefill | 14k md5 | 短 decode | 3,826 cold | errors |
| --- | ---: | --- | ---: | ---: | ---: |
| x0 (捨て、余白 1 MiB) | 216.15 s | 31722961 | 19.9 ms/tok | 50.23 s | 0 |
| b0 (base、余白 1 MiB) | 212.41 s | 31722961 | 19.8 | 50.83 s | 0 |
| **fix (既定)** | **194.12 s** | 20779eab (DIFF) | 20.1 | 52.05 s | 0 |
| base2 (余白 1 MiB) | 213.35 s | 31722961 | 20.2 | 50.97 s | 0 |

- base の 3 COND は 14k の途中で 136/32/32/32 MiB の確保が失敗し、そのたびに
  チャンクを空けて復帰していた (空き 149 → 137 MiB)。fix は失敗 0 回。
- fix の 14k は 2 パス (タイル 7,009、作業域 3,203 MiB、余白 3,072 MiB、947 本を先に追い出し、
  終端で 947 本 = 2,996 MiB を LRU へ戻した)。

### t290b — IQ3_S、4 COND × 2 ラウンド

| cond | 14k cold prefill | 14k md5 | 短 decode | 14k turn2 decode | 3,826 cold |
| --- | ---: | --- | ---: | ---: | ---: |
| x0 (余白 1 MiB) | 11.93 s | e2292eb6 | 18.0 ms/tok | 19.9 | 14.99 s |
| b0 (base) | 10.17 s | e2292eb6 | 17.9 | 16.3 | 3.79 s |
| **fix (既定)** | **8.65 s** | e2292eb6 | 18.0 | 16.4 | 3.53 s |
| base2 (余白 1 MiB) | 12.79 s | e2292eb6 | 18.2 | 20.2 | 3.98 s |

- base の 3 COND はどれも 14k で 4 回 (95/32/32/32 MiB) 失敗して復帰。fix は 0 回。
  fix は余白 1,024 MiB のため 1,371 本 (2,659 MiB) を先に追い出し、終端で全部戻した。
- **md5 は 6 出力すべて SAME**。タイルは 1 パスのまま (4,094 MiB)、パス数も 1 のまま。
- 同じ設定の 3 回 (x0/b0/base2) が 10.17-12.79 s なので、8.65 s はその外側 (失敗 4 回ぶんの
  待ちが消えた)。ノイズを考えても「悪化」ではない。

### t290c — UD-Q4_K_XL の繰り返し (fix を 2 COND)

| cond | 14k cold prefill | 14k md5 | 3,826 cold | 短 decode |
| --- | ---: | --- | ---: | ---: |
| x0 (余白 1 MiB) | 218.32 s | 31722961 | 57.77 s | 26.5 ms/tok |
| fix | 200.10 s | 20779eab | 56.06 s | 21.2 |
| fix2 | 195.61 s | 20779eab | 52.00 s | 21.5 |
| base2 (余白 1 MiB) | 211.62 s | 31722961 | 54.86 s | 21.5 |

- fix と fix2 の 14k は md5 まで一致 (20779eab) = 決定性は保たれている。
- UD の 14k は 3 回とも 195-200 s で完走。base 側 (211-218 s) より速い。

### 目標の判定

- UD-Q4_K_XL の 14k cold: **8/8 回で sane なテキスト** (t290a 4 COND + t290c 4 COND。
  fix 3 回、base 5 回)。旧い木 (t286c) は 4/4 回で API エラー。
- IQ3_S の 14k cold prefill: 8.65 s 対 base 10.17-12.79 s (**-15% 以上**)。目標は -3% 以内。
- IQ3_S の decode: 短 18.0 対 17.9 (**+0.6%**、目標 +2% 以内)、14k turn2 16.4 対 16.3 (+0.6%)。
- その他 2% 超の悪化: UD の 3,826 cold は t290a で +2.4% (52.05 対 50.83)、t290c で
  -5% (52.00 対 54.86) とばらつき、同じ設定の base 3 回も 50.23-57.77 s と振れる。
  一定の悪化とは判断しない。UD の decode も 20.1 対 19.8 (+1.5%) / 21.2-21.5 対 21.5 (0%)。

## 5. 数値の扱い

- lab の値はすべて実測。このコンテナでは GPU を触れないので、ここでの推定は明記する。
- UD の 14k が 195-200 s かかるのは CPU の MoE (Q4_K の専門家が IQ カーネル非対応で
  全部ホスト) の計算律速で、このタスクの前から同じ (t286c は途中で落ちたので比較不能)。
  パス数を 3 → 2 にした効果は expert 重みの上げ直しが減るぶん (推定 -1〜2 s) で、
  実測の差 12-18 s の主因は「確保失敗 → チャンク追い出し → 再試行」の列が消えたこと。
- IQ3_S の 8.65 s は「同じ run の base 3 回 (10.17/11.93/12.79) より速い」までしか
  言わない。ラボの COND 間ノイズは同じ設定で ±13% ある。

## 6. テスト

```
cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  → 1182 passed / 0 failed / 100 ignored (ff26091ee で実行)
cargo test --release -p denno-engine --features cuda --test kernel_invariants -- --test-threads=1
  → 95 passed / 0 failed
```

新規の 5 本を含めて緑。cuda feature 無しのビルドは base (`7890d121e`) でも 36 error で
落ちる (既存。`crates/denno-engine/src/model/qwen4exp.rs` の `crate::cuda` 参照) ので、
関門はタスク指定どおり `--features cuda` で取った。

## 7. lab のエラー計上について (タスクの手順 3)

lab のランナーはこのツリーに無い (`/work/gpu_lab` のジョブはホスト側) ので、直していない。
t286c の RESULT.md は、14k の応答が
`{"error": {"message": "層 41: CUDA cuMemAlloc が失敗: ...", "type": "api_error"}, "type": "error"}`
で text が空 (md5 d41d8cd9) だったのに **errors 0** と数えていた。修正案:

- 各リクエストの応答 JSON を読むとき、`error` キーがある (または `choices` が無い /
  `choices[0].message.content` が空) なら **エラーとして数える**。いまは HTTP/転送の
  失敗だけを数えているので、API エラーが 200 で返ると素通りする。
- 併せて `text` の判定を「md5 が d41d8cd9 (空) なら BROKEN」にすると、
  `ERRORS <cond> 0` と `text ok` が同時に出る矛盾を防げる。
- RESULT.md の `errors` 列は `ERRORS` 行の合計なので、上を直せば 14k の失敗が
  `errors 1` として表に出る (VERDICT も自動で FAIL 側に落ちる)。

## 8. GPU で確認してほしいこと

- 実機の本番 (8098) で UD-Q4_K_XL の 14k を 2-3 回流し、`[pf-pool]` の
  「余白 3072 MiB を残すため先に空けた」が出ること・`task256` の失敗が 0 回であること。
- IQ3_S の 14k で `[pf-pool-big] 上限 (... - 余白 1024 MiB ...)` が 4,864 MiB を返し、
  1 パスのままであること。
- `DENNO_ALLOC_FORCE_CHUNK=0` にすると t286c と同じ「3 MiB / 空き 3 MiB」で落ちるはず
  (切り戻しの確認。このコンテナでは再現できない)。

## 9. lab の VERDICT 行

t290a (UD-Q4_K_XL):
```
VERDICT b0 vs x0 OK decode -0.4% prefill -10.9% md5 SAME errors 0
VERDICT fix vs x0 OK decode +1.0% prefill -10.3% md5 DIFF errors 0
VERDICT base2 vs x0 OK decode +1.2% prefill -11.4% md5 SAME errors 0
VERDICT_DEPTH fix c8000_cold vs x0 decode -30.6% prefill -10.2%
VERDICT_DEPTH b0 c8000_cold vs x0 decode -20.7% prefill -1.7%
VERDICT baseline x0 errors 0 text ok
```
t290b (IQ3_S):
```
VERDICT b0 vs x0 OK decode -0.4% prefill -3.3% md5 SAME errors 0
VERDICT fix vs x0 OK decode -0.2% prefill -2.4% md5 SAME errors 0
VERDICT base2 vs x0 OK decode +0.9% prefill -2.4% md5 SAME errors 0
VERDICT_DEPTH fix c8000_cold vs x0 decode -22.1% prefill -27.5%
VERDICT_DEPTH b0 c8000_cold vs x0 decode -22.1% prefill -14.8%
VERDICT baseline x0 errors 0 text ok
```
t290c (UD-Q4_K_XL):
```
VERDICT fix vs x0 OK decode -19.9% prefill -23.3% md5 DIFF errors 0
VERDICT fix2 vs x0 OK decode -18.9% prefill -10.6% md5 DIFF errors 0
VERDICT base2 vs x0 OK decode -19.0% prefill -27.3% md5 SAME errors 0
VERDICT_DEPTH fix c8000_cold vs x0 decode -29.6% prefill -8.3%
VERDICT_DEPTH fix2 c8000_cold vs x0 decode -36.8% prefill -10.4%
VERDICT_DEPTH base2 c8000_cold vs x0 decode -22.2% prefill -3.1%
VERDICT baseline x0 errors 0 text ok
```
(t290c の decode は x0 が 26.5 ms/tok と外れ値。同じ設定の base2 は 21.5。)
