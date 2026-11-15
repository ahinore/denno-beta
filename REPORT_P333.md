ADOPT: yes t333a t333b

# task 333 — 14k cold の再検、`理由: stream` に名前を付ける、冷えたホストでの dev MoE

branch `br333` は開始時の `int-auto` `716e9a8ae` (merge: br325) から。int-auto は
このタスクの間ずっと `716e9a8ae` のまま (最後にもう一度 fetch して確認した)。
`br328` は int-auto に merge されていないが、**merge する必要は無かった** (task 328 の
`DENNO_PF_STREAM_SLOTS` は採らず、コードは br328 にしか無い。`REPORT_P328.md` だけを
`git show 0cbd27c5a:REPORT_P328.md` で読んだ)。数値はすべて GPU ラボの実測。

## 0. 結論 (3 行)

- **14k cold の退行は無い。** `tchk716e` の 8.78 s は**先頭 COND の偏り**で、
  同じ SHA の `base` / `base2` は 6.52 / 6.54 s。`3ae8b0046` の 6.61-6.66 s と
  比べて -1.2%。319 / 324 / 325 のどれも原因ではない (同じバイナリが 3 時間後に
  6.5 s を出すので、bisect する対象が無い)。理由はホストの page cache で、
  14k の写し (`[exp-pf-stream] copy_s`) が 1.37 s → 4.40 s に伸びた分が
  そのまま prefill に出る (§2、§4)。
- **`理由: stream` で「CPU 経路に落ちた」わけではない。** 冷えた 3 ラウンド
  (t328a/b/c) でも `[exp-pf] ... gpu_layers=48` で、MoE は**デバイスの
  hit/miss 経路**で回っていた (rows=1,106,480、gpu_ms=13,665)。落ちていたのは
  ストリーミング (層優先 dev MoE) だけで、層全体の CPU 計算ではない。
  task 328 の「CPU 経路」という表現は誤りだったので、ここで訂正する。
- どの門で落ちたのかを**名前にした** (`pf_stream_why`、19 + 4 箇所) ので、次に
  冷えたホストに当たったときは `理由: stream / stream: <門>` で出る。
  併せて「上げスレッドが居ない」門だけ、写しを計算スレッドへ落として 1 回
  やり直す `DENNO_PF_STREAM_RETRY` (既定 on) を足した。**2 ラウンドとも
  発火していない** (`retry=0`)。採否は「壊していないこと」で見た (§4、§6)。

## 1. Do 1: REPORT_P328 と前提の読み直し

- 読んだもの: `REPORT_P328.md` (br328 の commit から)、その §4 が指す
  `REPORT_P324.md` / `REPORT_P325.md`、`done/tchk716e/RESULT.md`。
- tchk716e (`716e9a8ae`、PROFILE=1、base vs `DENNO_PF_PREAD=0`) の生ログを
  読み直した。指摘の 8.1-9.1 s は**先頭の `warm` COND** の 8.78 s で、
  `base` / `base2` は 6.52 / 6.54 s、`pread0` / `pread0b` も 6.53 / 6.53 s。
  **同じ実行の中で先頭だけ 2.2 s 遅い** — README の「最初の COND は偏る」の
  とおりで、これが 8.1-9.1 s の正体。
- 現在の int-auto の基準は取り直していない (tchk716e が 12:28 UTC の実測で、
  それが現在の int-auto そのもの)。今回はその数値を基準に使う。

## 2. Do 2 の前半: 14k cold の退行は実在するか / どの merge か

**実在しない。** 14k cold (3,826 も併記、すべて GPU ラボの実測):

| 実行 (UTC) | SHA | cond ごとの 14k cold (s) | 3,826 cold (s) |
| --- | --- | --- | --- |
| 08:04 | `3ae8b0046` | 6.66 / 6.62 / 6.61 | 2.64 / 2.63 / 2.63 |
| 08:44 | `716e9a8ae` | **9.09 / 8.09 / 8.21** | 2.56 / 2.55 / 2.53 |
| 12:28 | `716e9a8ae` | 8.78 / **6.52 / 6.54** | 2.54 / 2.53 / 2.54 |
| 13:31 (t333a) | `80d3b7728` (br333) | 9.47 / 6.56 / 6.61 / 7.89 | 3.83 / 4.66 / 3.36 / 3.43 → r2 は全部 2.55-2.59 |
| 14:07 (t333b) | `afd9b1b77` (br333) | 6.54 / 6.52 / 6.51 | 2.55 / 2.55 / 2.55 (r2 も 2.56) |

- 08:44 の 8.1-9.1 s は `716e9a8ae` の**性質ではない**。同じバイナリが 12:28 に
  6.52 / 6.54 s を出している。したがって 319 / 324 / 325 のどれも原因ではない
  (bisect する差がバイナリ側に無い)。
- 3,826 は 08:04 / 08:44 / 12:28 / 14:07 で 2.53-2.64 とほぼ動かない。14k だけが
  動くのは、14k の作業集合が page cache に収まるかどうかの境目にあるため。
- 遅いときの中身 (同じ実行内の比較が効く。t333a):
  `[exp-pf-stream] ... up_MB=28377.7 copy_s=4.40` (x0) → prefill **9.47 s**、
  `copy_s=1.37` (base) → prefill **6.56 s**。写しの差 3.03 s が prefill の差
  2.91 s にほぼ 1:1 で出る。3,826 も同じ (`copy_s=3.22` → 4.66 s、
  `copy_s=1.10` → 2.55 s)。**ホストの読み出し速度だけが違う。**
- 1 リクエストで上げる量は 3,826 で 22.65 GB、14k で 28.38 GB。ホストは 59 GiB
  で、サーバ自身の RSS が 15-20 GB あるから、14k の 28 GB は page cache の
  端に載る。載らなかった分が disk (実測 ~1.3 GB/s) から来る。
- 08:44 の 14k だけ `copy_s=1.34` (写しは速い) で 8.21 s だったので、
  その回は写しの外 (ssm / PLE) が遅れている。どちらにせよ**同じ SHA が
  3 時間後に 6.5 s** なので、コードの退行ではない。
- **直すものは無い。** 14k cold の振れはホストの状態で、コードの性質ではない。

## 3. Do 2 の後半: `理由: stream` の正体

### 3.1 冷えた 3 ラウンドで本当に何が起きていたか (t328a/b/c の生ログ)

- `[pf-lm] 層 0: dev MoE を諦めた (理由: stream)` は出ている。しかし同じ
  リクエストの集計は
  `[exp-pf] lookups=41746 hit_rate=0.470 gpu_layers=48 skip_layers=0 experts=9808 rows=1106480 gpu_ms=13665.0`
  で、**48 層すべてがデバイスの hit/miss 経路**。CPU が計算したのは 53% の
  ミス行だけで、層ごと CPU ではない。
- つまり「dev MoE を諦めた」の意味は「**ストリーミング (層優先で活性を
  デバイスに置いたまま) を諦めて、従来の hit/miss 経路に落ちた**」であって、
  MoE が CPU になったわけではない。
- 冷えた状態でストリーミングが速いかというと、**速くない**。同じ日の
  `tint_fcbdf641e` の最後の COND は、ストリーミングのまま冷えて
  `copy_s=17.32` / prefill **18.79 s** だった。落ちた側 (t328a base) は
  **16.52 s**。冷えたホストでは写しが disk 律速になり、ストリーミングの
  方が 12% 遅い。**「落ちる = 崖」ではない。**
- では 16.5 s は何か。22.65 GB を disk から読む時間 (実測 1.3 GB/s で 17 s)
  である。page cache が冷えている間の 3,826 の下限はここで、コードでは
  動かせない (モデルの作業集合が RAM より大きい)。
- pread (`DENNO_PF_PREAD`、task 325 の既定 on) はこの状態で効くか:
  tchk716e の `pread0` と base は 6.53 対 6.52 (温えた状態) で差が無く、
  **冷えた状態の A/B はまだ取れていない**。今回は `DENNO_PF_PREAD` を触らない
  (既定のまま)。

### 3.2 変えたこと (2 つ、どちらも数値経路は変えない)

1. **`pf_stream_why` 診断** (`cuda.rs`)。ストリーミング経路の `Ok(None)` 19
   箇所に門の名前を付けた: `dead` / `shape` / `type` / `slot` / `ctx` /
   `kernels` / `accum` / `current` / `row_bytes` / `shared_table` /
   `host_missing` / `no_bytes` / `no_miss` / `stage` / `bounds` /
   `group_rows` / `seg_local` / `no_upload` / `ptr_table`。
   `iq_pf_stream_layer` 側にも `args` / `entries` / `plan` / `row_table`。
   `[pf-lm] 層 N: dev MoE を諦めた` の 1 行が
   `(理由: stream / stream: no_upload)` の形になる。数値は 1 バイトも変えない。
2. **`DENNO_PF_STREAM_RETRY`** (既定 on)。落ちた門が `no_upload` (上げ
   スレッドが居ない) / `ptr_table` (依頼したグループのデバイスポインタ表が
   合わない) のときだけ、**同じカーネル・同じ行表・同じ和の順序**のまま、
   写しの担い手を計算スレッドへ落として 1 回やり直す (`serial` 引数。
   変わるのは `ut` の判定だけ)。1 グループでも計算した後 (`wrote`) の門は
   `*_late` に分けて**再試行しない** — `Ok(None)` の契約は「`out` を触って
   いない」で、書いた後に従来経路へ落ちると二重加算になるため。
   発火回数は `[exp-pf-stream]` の `retry=N` に出る。

### 3.3 まだ分かっていないこと

- 冷えたときに**どの門**で落ちるのかは、今回の 2 ラウンドでは再現しなかった
  (§4: 全 COND で `理由:` 無し)。t328a の時間帯 (load 3-4、majflt 25,691、
  28 token の prefill まで 7 倍) の再現が要る。診断を入れたので、次に
  その状態に当たれば 1 行で分かる。
- `stage` (ステージングの確保) / `no_miss` (全 expert 常駐) / `dead`
  (PF_DEAD) で落ちた場合は、転送の担い手を替えても直らないので再試行しない
  (`stream_retry_pays` が false)。その 3 つは別の手当てが要る。

## 4. Do 3: lab (2 ラウンド、`SCENARIO=depth8k PROFILE=1`、IQ3_S)

COND は交互、x0 は捨て (先頭の偏り)、base2 は末尾の繰り返し。

### t333a (`80d3b7728`、4 COND × 2 ラウンド、load 4.6-6.0)

**測定した SHA は 2 つ目の commit の前** (`80d3b7728`)。差分 `afd9b1b77` は
ループ内の門の名前を `*_late` に分けただけで、一度も通っていない経路の
文字列しか変えない。最終コードは t333b で測った。

| cond | decode ms/tok | 3,826 cold r1 / r2 | 14k cold |
| --- | ---: | ---: | ---: |
| x0 (捨て) | 17.0 | 3.83 / 2.59 | 9.47 |
| base | 16.9 | 4.66 / 2.55 | 6.56 |
| retry0 (`DENNO_PF_STREAM_RETRY=0`) | 17.0 | 3.36 / 2.55 | 6.61 |
| base2 | 17.1 | 3.43 / 2.57 | 7.89 |

- ラウンド 1 が全 COND で遅く (3.36-4.66)、ラウンド 2 が揃って 2.55-2.59。
  同じ COND の 2 ラウンドで 30-80% 動く。**この時間帯のホストでは 3,826 の
  2% は切れない** (他の dev の実行と重なった)。14k も同じ実行内で 6.56 と
  7.89 (同じ設定) なので ±20%。
- `[exp-pf-stream] ... retry=0` は**全 COND で 0**、`理由:` の行は**全 COND で
  1 回も出ていない** = この時間帯は冷えが浅く、ストリーミングが落ちなかった。
  したがって `DENNO_PF_STREAM_RETRY` の分岐は発火していない。
- この 1 ラウンドだけでは 2% を切れないので、静かな時間帯にもう 1 回取った
  (t333b)。

### t333b (`afd9b1b77` = 最終コード、3 COND × 2 ラウンド、load 1.88)

| cond | decode ms/tok | 3,826 cold r1 / r2 | 14k cold |
| --- | ---: | ---: | ---: |
| x0 (捨て) | 17.0 | 2.55 / 2.56 | 6.54 |
| base | 16.9 | 2.55 / 2.56 | 6.52 |
| base2 | 16.9 | 2.55 / 2.56 | 6.51 |

- **全 COND・全ラウンドで 10 ms 以内**。3,826 2.55-2.56、14k 6.51-6.54、
  decode 16.9。int-auto (tchk716e) の 2.53-2.54 / 6.52-6.54 / 17.0 と比べて
  3,826 +0.4-0.8%、14k ±0%、decode -0.6%。**退行なし。**
- `[pf-lm-dev]` は 6 本の生ログすべてに 1 行、`理由:` は 0 行、
  `retry=0`。`copy_s=1.10-1.32` (温えている)。
- デバイス MoE が毎ラウンド使われた証拠 (要求どおり 1 行引用):

  ```
  [pf-lm-dev] 層優先: 活性はデバイスに置いたまま (router / shared / expert / combine)
  ```

- md5 は全 COND / 全ラウンドで SAME
  (`c2000_cold:a91e17da c2000_turn2:5a8f5488 c8000_cold:e2292eb6
  c8000_turn2:561c6da8`)。`c2000_turn2` が `5a8f5488` = **デバイス経路**の
  本文であることの証拠。errors 0。
- 本文 (c2000_cold、全 COND 同一 md5、引用):
  「prefillの遅延は計算量ではなく、ホスト往復とデータコピーが律速であった。
  計測の誤認や撤回された最適化を経て、GPU常駐化とレイアウト改善により速度を
  大幅に向上させた。」— sane。TXTD も「ご提示いただいた文書は、LLM推論
  エンジン（おそらく `denno-engine）の **Prefill（事前計算）における性能
  最適化の記録です」で sane。

### VERDICT 行

t333a:
```
VERDICT base vs x0 OK decode -0.2% prefill +0.0% md5 SAME errors 0
VERDICT retry0 vs x0 OK decode +0.5% prefill +0.0% md5 SAME errors 0
VERDICT base2 vs x0 OK decode +0.7% prefill +0.8% md5 SAME errors 0
VERDICT_DEPTH base c2000_cold vs x0 decode -1.4% prefill +12.3%
VERDICT_DEPTH base c8000_cold vs x0 decode +0.0% prefill -30.7%
VERDICT_DEPTH retry0 c2000_cold vs x0 decode -1.4% prefill -7.9%
VERDICT_DEPTH retry0 c8000_cold vs x0 decode +0.7% prefill -30.2%
VERDICT_DEPTH base2 c2000_cold vs x0 decode +1.1% prefill -6.5%
VERDICT_DEPTH base2 c8000_cold vs x0 decode +0.7% prefill -16.7%
VERDICT baseline x0 errors 0 text ok
```
t333b:
```
VERDICT base vs x0 OK decode -0.7% prefill +0.9% md5 SAME errors 0
VERDICT base2 vs x0 OK decode -0.7% prefill +0.9% md5 SAME errors 0
VERDICT_DEPTH base c2000_cold vs x0 decode +0.4% prefill +0.0%
VERDICT_DEPTH base c8000_cold vs x0 decode -0.7% prefill -0.3%
VERDICT_DEPTH base2 c2000_cold vs x0 decode +0.4% prefill +0.0%
VERDICT_DEPTH base2 c8000_cold vs x0 decode +0.0% prefill -0.5%
VERDICT baseline x0 errors 0 text ok
```
(t333a の `prefill +12.3%` は base のラウンド 1 が 4.66 s だったため。
ラウンド 2 は 2.55 s。t333b では 0.0% で、ホストの状態であることが分かる。)

## 5. テスト

```
CARGO_TARGET_DIR=/tmp/t333 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1,225 passed / 0 failed / 102 ignored
CARGO_TARGET_DIR=/tmp/t333 cargo test --release -p denno-engine --features cuda --test kernel_invariants -- --test-threads=1
  -> 102 passed / 0 failed  (新規 2 本)
```

- `model::exp_pf::tests::stream_retry_gate_truth_table` — `DENNO_PF_STREAM_RETRY`
  の既定 on / `0` 系だけ偽。
- `model::exp_pf::tests::retry_pays_only_for_the_transfer_guards` — 再試行する
  門が `no_upload` / `ptr_table` だけ、`*_late` とその他 20 個は false。
- `kernel_invariants::t333_stream_decline_names_the_gate` — 19 + 4 個の門の
  名前、入口での理由のリセット、`[pf-lm]` 行への内訳の配線。
- `kernel_invariants::t333_stream_retry_swaps_only_the_transfer_owner` —
  `serial` が `ut` の判定だけを変えること、`wrote` の追跡、`*_late` の別名、
  `retry=` の台帳。
- **緑は発火の証拠ではない。** 再試行の分岐は 2 ラウンドで 1 回も発火して
  いない (`retry=0`)。ユニットテストが見ているのは純関数と配線だけ。
- 注意: `--features cuda` を外すとこのツリーはビルドできない (`qwen4exp` が
  `crate::cuda` を無条件に参照する。私の追加分のエラーではない)。task 328 も
  同じく `--features cuda` で回している。

## 6. 採否

- 目標は満たした: 14k cold の退行は無い (§2)、デバイス MoE は毎ラウンド
  (§4 の `[pf-lm-dev]` 引用)、3,826 / 14k は int-auto と同等 (t333b で
  +0.8% / ±0%)、decode -0.6%、errors 0、text SAME かつ sane。
- ゲートは既定 on のまま (`DENNO_PF_STREAM_RETRY`)。**発火していないので
  速度の改善は測っていない** (測ったのは「壊していないこと」と「門に名前が
  付いたこと」)。
- 次にやるなら: 冷えた状態 (t328a の時間帯) で `理由: stream / stream: <門>`
  を 1 本取り、`stage` / `dead` だった場合の手当てを別タスクにする。

## 7. GPU で確認してほしいこと

- 冷えたホスト (別の大きなモデルを積んだ直後) で 3,826 を 1 回流し、
  `[pf-lm] 層 0: dev MoE を諦めた (理由: stream / stream: ???)` の `???` を
  取る。ここが `no_upload` なら `DENNO_PF_STREAM_RETRY=0` との A/B が意味を持つ。
- `DENNO_PF_STREAM_RETRY=0` と既定の A/B は、冷えた状態でしか差が出ない。
  温えた状態では両方 `retry=0`、`[exp-pf-stream]` の他の値も同一になるはず
  (t333a の base と retry0 がそれ)。
