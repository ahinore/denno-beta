ADOPT: yes t368b t368c

# REPORT P368 — flash-next-UD-IQ3_XXS decode: mix norm の二乗和と greedy argmax を 1 本の鎖から外す

branch `br368` = 開始時の `int-auto` **5df54268c** (merge: br364)。作業中に int-auto は
動かず (最終確認でも `git rev-list --left-right --count int-auto...HEAD` = `0 3`)、
merge は不要だった。数値はすべて GPU ラボの実測。lab 名と測った SHA は
**t368a = 62047f51a** / **t368b = 6c4150a9e** / **t368c = 16002d687**。
lab は 3 本で上限 (at most 3 rounds) に達した。

## 0. 結論

1. **採る。** 短い decode が t368b の 2 ラウンド中央値で **16.6 → 16.2 ms/tok
   (-2.1%、表の値は -2.4%)**、既定 on にした後の t368c (2 ラウンド) でも
   **既定 off 16.6 / 既定 on 16.3 (+1.4%)**。段は **mix 4.67-4.75 → 4.40-4.45
   (-5.6〜-6.3%、うち norm 0.49-0.51 → 0.21-0.23)** と
   **sample 0.11 → 0.04 (-64%)**。prefill は 0.68-0.69 s のまま (悪化なし)、
   errors 0、**md5 は 3 本の lab の全 COND・全コンテキストで SAME**。
2. 14k 後も悪くない: t368b の c8000_turn2 は qdqa 15.2 / base2 15.5 / base 16.5
   (x0 18.9 は最初の COND の順序バイアス)、c8000_cold は qdqa 16.4 / base2 16.8。
3. 待ちを狭める `DENNO_DEC_SYNC_STREAM` (t368a の `sst`) は **2 ラウンドとも
   16.6 で差 0.0%**。採否規則 (段で 2% 以上) に届かないので **既定 off のまま**
   残した (A/B 用のゲートとしては生きている)。
4. 最後の commit で `hc_mix_quant_ref` (A/B 経路) の norm を高速経路と**同じ関数**に
   揃えた。これが無いと `DENNO_HC_NORM_QUAD` 既定 on のとき
   「fast と ref がビット一致」の契約が崩れ、`hc_mix_*_bitexact` 系が落ちる。

## 1. 前提: どこを削ることにしたか (task の Do 1/2)

モデルは既定の `flash-next-UD-IQ3_XXS` (83.6 GB)。前タスク t364 の後、
base の台帳はこうなっていた (t368b `base_1` の a1 窓 = 128 tok、`[q4ledger]`、
step 16.39 ms/tok。ゲート off = int-auto と同じ):

| 段 | ms/tok | 中身 |
| --- | ---: | --- |
| mix | **4.67** | qchain (Q8_0)、97 呼び出し。**cpu 3.77** / launch 0.60/436 |
| ssm | 3.45 | GDN 36 層 (GPU の完了待ちが本体、P245) |
| experts | 3.36 | GPU 鎖 + ミス CPU (P352/P356/P361) |
| full | 2.12 | 12 層、gpu 1.86 |
| head | 0.73 | 963 GB/s |
| shared | 0.61 | side stream |
| router | 0.18 | t364 で mix の鎖へ移した後 |
| **sample** | **0.11** | 語彙 248320 の素直な比較鎖 |

`[q4ledger] sub mix` = `sum=0.49 ... norm=0.49` = mix 4.67 のうち **norm が 0.49**。

**選んだ理由**: mix / experts / ssm の本体は t266/282 (qchain の GEMV)、
t352/356/361 (ミス経路)、t245 (GDN の完了待ち) が既に触っていて、残りは
**.cu を書かないと動かない** (このタスクは .cu/.ptx 禁止)。ホスト側だけで完結する
残りは (a) **mix の norm の二乗和 0.49 ms/tok** = 1 トークンあたり 97 回の
`n_embd` 依存鎖、(b) **greedy の argmax 0.11 ms/tok** = 248320 要素の比較鎖、の
2 つで、どちらも「値を変えずに (argmax) / 丸め順だけで (norm)」並列化できる。

## 2. 変えたこと

すべて env ゲート付き (`=0` が int-auto の挙動そのもの)。

- **`DENNO_HC_NORM_QUAD` (既定 on)** — `hyper::grouped_rms_norm_batch_into` の
  二乗和を、**`xs.len() == 1` (decode) のときだけ** 4 本のアキュムレータで足す
  (`sq_sum_quad`: 4 要素ずつ読み、最後に `(a0+a1)+(a2+a3)`、端数は a0)。
  prefill (`nt > 1`) は従来の逐次和のまま = ビット不変。
  **丸め順が変わる**ので、採るまでは既定 off だった。
- **`DENNO_ARGMAX_FAST` (既定 on)** — greedy の argmax を
  「8 本の部分最大 → 全体の最大値 → その値に一致する最初の添字」の 2 パスに替える。
  同点は先頭・NaN は無視・-0.0/+0.0 は同値として扱うので、**返す添字は素直な鎖と
  同一 = 出力は変わらない** (md5 の根拠はここ)。
- **`DENNO_DEC_SYNC_STREAM` (既定 off のまま)** — decode の待ち
  (`moe_gemv_dev_with` / `dec_full_dev` / `q4_attn_decode`) を `cuCtxSynchronize`
  から null ストリームのイベント 1 本へ狭める。expert の上げ直し (1 本 ~178 us) と
  shared expert の side stream を待たなくなるが、**lab では差が出なかった** (§4.1)。
- **契約の維持 (最終 commit)** — `hc_mix_quant_ref` の norm を
  `hyper::grouped_rms_norm_batch_into` (高速経路と同じ関数) に揃えた。
  値の意味は同じで、変わるのは丸め順だけ。ゲート `=0` で両方とも逐次和に戻る。
- 発火の証拠: `[dec-sync-stream] events=N fallback=N` (PROFILE の 1 行、
  `DEC_SYNC_STREAM=1` で 1 回でも通ったときだけ出る)、
  `[q4ledger] sub mix` の norm、`[q4ledger] sample`。

## 3. lab の結果

すべて `SCENARIO=depth8k PROFILE=1`、`MODEL` 指定なし (既定 = flash-next-UD-IQ3_XXS)、
ベースライン COND は task の指定どおり
`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0 DENNO_GPU_WEIGHT_MB=12000`。
最初の COND (`x0`) は捨て、最後に同じもの (`base2` / `b2` / `off2`) を置いた。

### 3.1 t368a (`62047f51a`、6 COND x 2 ラウンド、load_before 2.12、ゲートは既定 off)

| cond | env | decode ms/tok | prefill s |
| --- | --- | ---: | ---: |
| x0 | (捨て) | 16.6 | 0.69 |
| base | — | 16.6 | 0.69 |
| kp | `DENNO_GDN_DEC_KPROF=1 DENNO_MIX_KPROF=1` | 16.7 | 0.68 |
| **qd** | `DENNO_HC_NORM_QUAD=1` | **16.4** | 0.67 |
| sst | `DENNO_DEC_SYNC_STREAM=1` | 16.6 | 0.68 |
| base2 | — | 16.6 | 0.68 |

```text
VERDICT base  vs x0 OK decode -0.2% prefill +0.7% md5 SAME errors 0
VERDICT kp    vs x0 OK decode +0.2% prefill -1.5% md5 SAME errors 0
VERDICT qd    vs x0 OK decode -1.4% prefill -2.9% md5 SAME errors 0
VERDICT sst   vs x0 OK decode +0.0% prefill -1.5% md5 SAME errors 0
VERDICT base2 vs x0 OK decode -0.5% prefill -1.5% md5 SAME errors 0
VERDICT baseline x0 errors 0 text ok
```

配線は生きている (ゲートの実効性を数えた): `sst` のログは
`[dec-sync-stream] events=2364 fallback=0` を出し、`base` / `qd` は 0 回 (= off)。
それでも decode の中央値は 16.6 のままで、2 ラウンドとも同じだった。

### 3.2 t368b (`6c4150a9e`、6 COND x 2 ラウンド、load_before 1.66) — 採否の根拠

| cond | env | decode ms/tok | prefill s | md5 |
| --- | --- | ---: | ---: | --- |
| x0 | (捨て) | 16.6 | 0.69 | 全 SAME |
| base | — | 16.6 | 0.69 | 全 SAME |
| qd | `DENNO_HC_NORM_QUAD=1` | 16.4 | 0.68 | 全 SAME |
| qa | `DENNO_ARGMAX_FAST=1` | 16.5 | 0.71 | 全 SAME |
| **qdqa** | 両方 `=1` | **16.2** | 0.68 | 全 SAME |
| base2 | — | 16.6 | 0.69 | 全 SAME |

```text
VERDICT base  vs x0 OK decode +0.0% prefill -0.7% md5 SAME errors 0
VERDICT qd    vs x0 OK decode -1.4% prefill -2.9% md5 SAME errors 0
VERDICT qa    vs x0 OK decode -0.5% prefill +2.9% md5 SAME errors 0
VERDICT qdqa  vs x0 OK decode -2.1% prefill -2.9% md5 SAME errors 0
VERDICT base2 vs x0 OK decode -0.2% prefill +0.0% md5 SAME errors 0
VERDICT_DEPTH qdqa c8000_cold  vs x0 decode -23.0% prefill -21.8%
VERDICT_DEPTH qdqa c8000_turn2 vs x0 decode -19.8% prefill -5.5%
VERDICT baseline x0 errors 0 text ok
```

段の台帳 (同じ lab の `[q4ledger] sub mix` と `sample`、ms/tok):

| cond | mix | うち norm | sample |
| --- | ---: | ---: | ---: |
| base_1 / base_2 | 4.67 / 4.75 | 0.49 / 0.51 | 0.11 / 0.11 |
| qd_1 / qd_2 | 4.44 / 4.46 | 0.22 / 0.23 | 0.11 / 0.11 |
| qa_1 / qa_2 | 4.69 / 4.72 | 0.49 / 0.51 | 0.04 / 0.04 |
| qdqa_1 / qdqa_2 | **4.40 / 4.45** | **0.21 / 0.23** | **0.04 / 0.04** |

`mix` のほかの内訳 (h2d 3.98MB/97、launch 436、d2h 1.09MB/145、full/ssm/experts/head)
は COND 間で同じ = **GPU の仕事は増えていない**。減ったのは cpu 3.77 → 3.50。

### 3.3 t368c (`16002d687`、既定 on の確認、5 COND x 2 ラウンド、load_before 1.47)

`x0` も既定 on のバイナリなので、旧挙動は `off` (= 両ゲート `=0`) で測っている。

| cond | env | decode ms/tok | prefill s |
| --- | --- | ---: | ---: |
| x0 | (捨て、既定 on) | 16.3 | 0.68 |
| base | 既定 on | 16.3 | 0.67 |
| b2 | 既定 on | 16.3 | 0.68 |
| **off** | `DENNO_HC_NORM_QUAD=0 DENNO_ARGMAX_FAST=0` | **16.6** | 0.69 |
| off2 | 同上 | 16.6 | 0.69 |

```text
VERDICT base  vs x0 OK decode +0.0% prefill -1.5% md5 SAME errors 0
VERDICT b2    vs x0 OK decode -0.2% prefill +0.0% md5 SAME errors 0
VERDICT off   vs x0 OK decode +1.4% prefill +1.5% md5 SAME errors 0
VERDICT off2  vs x0 OK decode +1.4% prefill +2.2% md5 SAME errors 0
VERDICT baseline x0 errors 0 text ok
```

= 既定 on が 2 ラウンドとも 1.4% 速い。t368b の -2.1〜-2.4% と向きが一致する
(絶対値が 16.3 と低いのは load_before が低い回だったため。採否は同一 run 内の比で読む)。

### 3.4 出力テキスト (sane の証拠、t368c から引用)

- `TXT` (英語プロンプト): `# B-Trees: A Comprehensive Explanation\n\n## 1. What is a B-Tree?\n\nA **B-tree (Balanced Tree) is a self-balancing tree data structure that maintains sorted data and allows searches, sequential access, insertions, and deletions in logarithmic time.`
- `TXTB` (日本語): `Rust の「所有権（Ownership）とライフタイム（Lifetime）は、Rust を学ぶ上で最も重要であり、かつ最も難しい概念です。\n\nしかし、これらは「メモリを安全に管理するためのルール**です。`
- `TXTD` (14k 後): `ご提示いただいた文書は、LLM推論エンジン（denno-engine）における **Prefill（事前計算）フェーズの性能最適化の歴史を詳細に記録した技術レポートです。`

md5 は 3 本の lab の全 COND で `a:edaa84ad b:1938059f c2000_cold:b42d79d2
c2000_turn2:2081cdeb c8000_cold:672fec07 c8000_turn2:bc4aaf54` (SAME)。

## 4. 数値の扱い

- **argmax は値を変えない** (同点 = 先頭、NaN 無視、±0 同値、返す添字が同一)。
- **norm は丸め順だけ変わる** (二乗和が数 ulp)。決定性・行チャンク不変・
  par/seq 一致・アンカーは維持 (ゲート off が旧経路そのもの)。
  ゲート on の丸めは lab の 6 コンテキストでは出力に現れなかった (md5 SAME) が、
  原理的には現れうる箇所なので、そこは意図した変更として扱う。
- `hyper_batch_matches_seq_bitwise` (逐次版とバッチ版のビット一致契約) は
  **ゲート off の経路**で検査するようにした。on のときの差は数 ulp。
- decode の中央値は lab が 0.1 ms/tok 刻みで出す。16.6 → 16.2 は 4 段。

## 5. 見積りと実測

| 項目 | 見積り (コンテナ内の相対実測から) | lab 実測 |
| --- | --- | --- |
| mix norm の二乗和 | 1 回 4.23 → 1.33 us × 97 回 = -0.28 ms/tok | norm 0.49 → 0.21 = **-0.28 ms/tok** |
| argmax | 110.5 → 11.9 us = -0.10 ms/tok | sample 0.11 → 0.04 = **-0.07 ms/tok** |
| 合計 | -0.38 ms/tok | mix -0.27〜-0.30 + sample -0.07 = **-0.35**、16.6 → 16.25 (-2.1%) |

手元のマイクロベンチ (Ryzen 8700G、hc=4 / n_embd=2560) は相対値としてだけ扱った。

## 6. テスト

- 関門 (ブランチ最終 HEAD で実行):
  - `CARGO_TARGET_DIR=/tmp/t368 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`
    = **1259 passed / 0 failed / 103 ignored**
  - 同 `--test kernel_invariants` = **108 passed / 0 failed**
- 新しい分岐が**発火した回数を数える**テストを足した:
  - `hyper::tests::hc_norm_quad_gate_reads_only_explicit_values` (ゲート解釈)、
    `sq_sum_quad_is_close_to_serial_sum` (4 本和の誤差)、
    `hc_norm_quad_branch_fires_only_for_single_token` (nt=1 でだけ発火)、
  - `sampler::tests::argmax_first_matches_naive_scan` (n=0..248320 と同点/NaN/-0.0 の端)、
    `argmax_fast_gate_switches_the_greedy_path` (on で 8 本経路 / off で素直な鎖に入る回数)、
  - `cuda::tests::dec_sync_stream_gate_reads_only_explicit_values`、
    `dec_sync_stream_line_only_when_on_and_used`。
- cuda 無しビルドは**この変更の前から赤** (`expert_vram` / `crate::cuda` を直接使う
  箇所が 38 エラー。今回触っていない行。`--features cuda` が前提のツリー)。

## 7. GPU で確認してほしいこと

1. 実運用に近い長さ (`tools` / c8000 の turn2) で `[q4ledger] sub mix` の norm が
   0.2 前後まで落ちていること (短い decode では 0.21-0.23 を確認済み)。
2. 既定 on のまま、長い生成で md5 がゲート off と一致すること。norm は丸め順を
   変えるので、lab の 6 コンテキストの外では一致しない可能性がゼロではない。
3. `DENNO_DEC_SYNC_STREAM=1` は差が出なかったので既定 off のまま。ホスト負荷の
   低い回で試す価値は残っている (0.0% は「無効」ではなく「この 2 ラウンドでは
   差が分解できなかった」)。

## 8. 次に削れるもの (推定を含む)

- mix 4.40 のうち **cpu 3.50** がまだ大きい。qchain の GEMV は GPU 側に移っており、
  残りは詰め替えと inject。0.49 の norm を削った後の最大のホスト仕事。
- ssm 3.45 は GDN 3 カーネルの完了待ち (GPU 時間そのもの、P245)。.cu 無しでは動かない。
- experts 3.32 も同じ (P352/P356/P361 の続きはカーネル側)。
- sample は 0.04 になり、削り代はもう無い。router も 0.18 で t364 の後は小さい。
