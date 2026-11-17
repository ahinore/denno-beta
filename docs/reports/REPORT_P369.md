ADOPT: yes t369a t369b t369c

# task 369 — IQ3_S prefill、`full` 段の K/V 書き戻し (まだ誰も触っていない段)

branch `br369` は開始時の `int-auto` `c1532cf5b` から。途中で `int-auto` が
`e213685c4` (br356 / br358 / brm1116) まで進んだので merge し、両側を残した
(`7e23820ab`。衝突は `cuda.rs` の tests 末尾だけで、両方のテストを残した)。
**br365 は merge していない** (採用 0 件の既定 off ゲートだけで、`int-auto` に
入っていない。本タスクはその 3 本に触れていない)。

数値はすべて GPU ラボの実測 (`SCENARIO=depth8k`, `PROFILE=1`,
`swift15-abl-IQ3_S`)、lab 名は t369a / t369b / t369c (3 ラウンド、上限 3)。
3 ラウンドとも全 COND・全リクエストで md5 SAME、errors 0、text sane。

## 0. 結論 (5 行)

- **採る。`DENNO_PF_KV_ASYNC` (full 層の k/v 書き戻しを `DtoHAsync` + pinned に
  して `AttnCache::push` の直前まで待たない) を既定 on にした。**
- 効いた理由は**待ちの正体**だった: `cuMemcpyDtoH` は呼び出し時点で
  **コンテキスト全体の完了を待つ**。14k では「12 full 層 x 7 チャンク = 168 回」の
  この待ちが `full` 段の露出になっていた (t369a b1: full wall 1,178.9 /
  busy 792.9 / **exposed 386.0** ms)。非同期化で **exposed 214 ms**、
  `full` の壁 1,006 ms (-14.8%)。
- 14k cold は **base 中央値 6.74 s (6 本) に対し 6.30 s (-6.5%)**、
  **同じ run の末尾 base (位置を揃えた比較、6.45 / 6.46 / 6.47) に対し -2.5%**。
  3,826 cold は **2.52 -> 2.49 s (-1.2%)** で悪化なし。decode は 16.6-16.7
  ms/tok で不変 (base 16.7-17.0)。台帳の `gpu_busy` は 4,907 -> 4,911 ms で不変、
  `exposed_host` は 2,867 / 2,839 -> 2,125 / 2,128 ms。
- **採らない: `DENNO_PF_KV_PUSH`** (pinned から直接 push して `ks`/`vs` への写しを
  省く)。t369c の p1 は d2 と 14k 6.29 対 6.30、3,826 2.49 対 2.48 で**差なし**。
  ゲートは発火した (`full` の cpu 538.6 -> 500.5 ms = 689 MB の写しが消えた) が、
  その写しは元から重なっていた。既定 off のまま置いた (切り戻し不要)。
- `DENNO_PF_PIN_STAGE` (既存、既定 off) も測った (t369a c2): 3,826 2.62 /
  14k 6.71 で**両方悪化**。触らない。

## 1. Do 1: 基準の取り直しと、どの段を取ったか

3 ラウンドとも同じ形で、先頭に捨ての baseline (x0)、末尾に baseline の繰り返し
(b2) を置いた。depth8k は 14k を **round 1 にしか流さない**ので、14k は
1 COND 1 サンプル。だから**同じ run の b2 との対**を主に見る (タスクの指示)。

### 1.1 段ごと (t369a b1_1、14k、`[q4ledger-pf]`、ms)

| 段 | wall | busy | exposed | 備考 |
| --- | ---: | ---: | ---: | --- |
| ssm | 2,535.1 | 1,578.9 | 956.3 | d2h 823.69 MB/504 (GDN 状態) |
| experts | 2,137.4 | 1,501.0 | 636.4 | |
| other | 1,162.8 | 274.5 | 888.4 | h2d 4,128.72 MB/**4,110 呼び出し** |
| full | 1,178.9 | 792.9 | **386.0** | d2h 689.01 MB/168 = **同期 D2H** |
| mix | 20.3 | 759.8 | 0.0 | GPU 仕事は ssm の裏 |

`close wall=7,009.4 gpu_busy=4,909.9 exposed_host=2,838.6`

### 1.2 どの段を取ったか

- **`full` が唯一まだ手を入れられていない段**だった。t365a の REPORT_P365 §4 が
  「残る余地は `mix` の f32 GEMM / `other` の PLE 注入 / `full` の K/V D2H /
  expert の転送 bytes」と挙げ、そのうち 14k の壁の 18% を占める `full` の
  K/V D2H (689 MB / 364 ms) だけが「ホスト KV が正本なので触らなかった」と
  書かれていた。本タスクは `.cu` を触れないので、**ホスト側だけで閉じる
  `full` の待ち**が最も大きい未着手点だった。
- 読んで分かったのは、コストの正体がバイト数ではなく **API の同期性**だったこと:
  `cuMemcpyDtoH` (同期) は呼び出しでコンテキスト全体の完了を待つ。14k では
  168 回、3,826 では 48 回。しかもホストの KV は `pf_full_out` の後、
  `AttnCache::push` まで**誰も読まない** (ミラーへ直書きできているときは
  `pf_full_out` も読まない)。
- `other` の 4,110 呼び出し (4.1 GB の pageable H2D) も同じ同期 API なので
  同じ待ちを踏んでいた。これは今回の変更で**副産物として一緒に軽くなった**
  (other の露出 888 -> 546 ms)。中身の特定は §7 の次にやることにした
  (`DENNO_PF_XFER_DBG=1` を x0 に付けてサイズを取った: 524,800 / 627,200 /
  921,600 / 460,800 / 704,000 B が 1,402 / 1,288 / 1,089 / 271 / 30 回。
  sub は付いていない = `other` の残差)。

## 2. Do 2: 変えたこと

### 2.1 `DENNO_PF_KV_ASYNC` (既定 on、`da3ce3318` + `30e5fa1b7`)

- `cuda::pf_kv_wb_begin` / `pf_kv_wb_copy_into` / `pf_kv_wb_discard` /
  `pf_kv_wb_pending` / `pf_full_mirror_direct`。k/v を pinned へ
  `cuMemcpyDtoHAsync` (null ストリーム = カーネルの後ろ) で積み、イベントを
  1 本だけ持つ。**ミラーへ直書きできたときだけ**非同期にする。
- `pf_full_qkv` / `pf_full_qkv_fused`: ミラーへの直書きを D2H の**前**へ移し、
  直書きできたら async、できなければ従来どおり `pf_chain_sync` + 同期 D2H。
- `pf_full_out`: `mirror_direct` の判定を `ok_shape` の前へ出し、直書きの
  ときはホスト KV の長さを要求しない (push が後ろへ回るので `k_cpu` はまだ短い)。
  ミラーが LRU で落ちていてホスト KV も短いときは **bail して従来経路へ戻す**
  (fail-closed、`ba539f5ea`)。
- `qwen4exp_attn.rs`: `pf_full_mirror_direct` が真のときだけ push を
  `pf_full_out` の後ろへ回す。偽なら従来どおり先に積む。落ちた枝では保留を
  捨ててから `truncate_layer` で巻き戻す。
- **デバイスから出るバイト列は同じ**。変わるのはホストへ載る順序だけ。

### 2.2 `DENNO_PF_KV_PUSH` (既定 off、`546d74fd5`)

- `pf_kv_wb_with` で保留中の k/v を pinned のまま closure へ渡し、キャッシュへ
  直接積む (ks/vs への写しを省く)。**測ったら差が無かった**ので既定 off のまま。

## 3. 数値 (lab、同じ run の中の対)

### 3.1 14k cold (`c8000_cold`、s)

| run | x0 (捨て) | b1 (base) | 候補 | b2 (base) | 候補 / b2 | 候補 / base 中央値 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| t369a | 7.97 | 7.04 | c1 **6.30** | 6.45 | **-2.3%** | -6.5% |
| t369b | 8.56 | 7.01 | d1 **6.30** / d2 **6.31** | 6.46 | **-2.4%** | -6.5% |
| t369c | 20.46* | 7.04 | d1 6.89* / p1 6.29 / d2 6.30 | 6.47 | -2.8% | -6.6% |

`*` t369c は load 1.83 で先頭の 2 COND が汚れた (x0 20.46 s、b1 の 3,826 も 3.38 s)。
採否は位置の揃った b2 と、全 base 6 本の中央値 (6.74) の両方で読んだ。

- **候補の中央値 6.30 s (5 サンプル: 6.29 / 6.30 / 6.30 / 6.30 / 6.31)**。
  base の中央値 6.74 s。**-6.5%**。位置を揃えた b2 (6.45 / 6.46 / 6.47、
  中央値 6.46) に対し **-2.5%**。
- 台帳 (`[q4ledger-pf] close`、14k): `exposed_host` は
  t369a b1 2,867.0 -> c1 2,124.8、t369b b1 2,838.6 / d1 2,128.2 / d2 2,126.6 /
  b2 2,279.3、t369c d2 2,122.3 / p1 2,117.3。**gpu_busy は 4,907-4,922 で全 COND 同じ**。
- `full` 段 (14k): wall 1,178.9 / 1,183.2 / 1,161.3 (base) ->
  **1,006.3 / 1,006.9 / 1,008.3 / 1,009.8** (候補)。exposed 386.0 / 389.1 /
  368.1 -> **214.0 / 214.3 / 213.8 / 214.7**。**段の露出 -43%、壁 -14.8%**。

### 3.2 3,826 cold (`c2000_cold`、2 ラウンドの中央値、s)

| run | b1 (base) | 候補 | b2 (base) |
| --- | ---: | ---: | ---: |
| t369a | 2.51 | c1 **2.48** | 2.52 |
| t369b | 2.70* | d1 **2.50** / d2 **2.49** | 2.52 |
| t369c | 3.38* | p1 2.49 / d2 2.48 / d1 2.92* | 2.52 |

`*` 先頭側の COND は load で汚れた (b1 は毎回 x0 の直後)。汚れていない
base 3 本 (2.51 / 2.52 / 2.52) の中央値 2.52 に対し候補の中央値 **2.49 = -1.2%**。
`full` 段は 39.0 / 39.2 -> 42.0 / 41.3 / 40.9 ms (露出はほぼ同じ。3,826 は
チャンクが 2 つしかなく、もともと待ちが小さい)。

### 3.3 decode (`ms/tok` の中央値)

| run | base | 候補 |
| --- | --- | --- |
| t369a | 16.8 / 16.6 | 16.6 |
| t369b | 17.0 / 16.7 | 16.7 / 16.7 |
| t369c | 16.8 / 16.7 | 16.6 (p1) / 16.6 (d2) |

0.1 ms 刻みの範囲で **悪化なし** (むしろ 0.1-0.3 良いが、これは経路の外)。

### 3.4 `DENNO_PF_KV_PUSH` の A/B (t369c、同じ run)

| 14k | d2 (async) | p1 (async+push) | 差 |
| --- | ---: | ---: | ---: |
| c8000_cold | 6.30 | 6.29 | -0.2% |
| c2000_cold | 2.48 | 2.49 | +0.4% |
| `full` の cpu (14k) | 538.6 ms | 500.5 ms | -38 ms |

写しは確かに消えた (38 ms) が、壁は動かない。**既定 off**。

## 4. 採否 (タスクの規則で)

タスクの「Why」の規則: **「2 ラウンド以上の中央値で、その段が 2% 以上勝ち、
他が 1% を超えて悪化せず、テキストがまともなら、タスク目標に届かなくても
既定 on で採る」**。

- 2% 以上勝った段: `full` (露出 -43%、壁 -14.8%)。3 ラウンド・9 サンプルで
  符号が揃っている (386/389/368 -> 214/214/214/210)。
- 全体: 14k **-2.5%** (位置を揃えた対) / **-6.5%** (全 base の中央値)。
  目標 (2.5%) は位置を揃えた読みで**ぎりぎり満たす**。
- 他は悪化していない: 3,826 -1.2%、decode 0.0%、`gpu_busy` 不変。
- よって **`DENNO_PF_KV_ASYNC` を既定 on**。切り戻しは `DENNO_PF_KV_ASYNC=0`。

## 5. 数値の扱い

- md5 は 3 ラウンド・全 COND・全リクエストで SAME:
  `a:65ededc3` / `b:f18c75fc` / `c2000_cold:a91e17da` / `c2000_turn2:5a8f5488` /
  `c8000_cold:e2292eb6` / `c8000_turn2:561c6da8`。errors 0。
- 本変更は**数値を 1 ビットも変えない** (同じバイト列を同じ場所へ書く。順序と
  待つ位置だけが変わる)。md5 SAME はその確認。
- 14k は depth8k が round 1 にしか流さないので 1 COND 1 サンプル。**14k の
  1 本の比較は ±9% 動く** (t369c の x0 20.46 s、t369b の x0 8.56 s) ので、
  同じ run の b2 との対と、3 ラウンド分の中央値で読んだ。3,826 は 2 ラウンドの
  中央値だが、先頭側の COND は load で 2.70 / 3.38 まで振れた (base の
  汚れていない 3 本は 2.51 / 2.52 / 2.52 と揃っている)。
- テキスト (t369b d1、`c2000_cold`):

```
prefillの遅延は計算量ではなく、ホスト往復とデータコピーが律速であった。
計測の誤認や撤回された最適化を経て、GPU常駐化とレイアウト改善により速度を大幅に向上させた。
最終的に、PCIe帯域やメモリコピーの最適化がボトルネックから外れ、計算リソースの効率的な活用へと移行した。
```

## 6. テスト

```
CARGO_TARGET_DIR=/tmp/t369 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1249 passed / 0 failed / 103 ignored   (int-auto e213685c4 merge 後)
CARGO_TARGET_DIR=/tmp/t369 cargo test --release -p denno-engine --features cuda --test kernel_invariants -- --test-threads=1
  -> 108 passed / 0 failed
```

- 追加 (lib): `t369_kv_async_gate_truth_table` (既定 on を含む 11 通り)、
  `t369_kv_push_gate_truth_table` (既定 off)、
  `t369_kv_wb_copy_without_pending_touches_nothing`、
  `t369_kv_wb_with_without_pending_does_not_call`、
  `t369_kv_wb_begin_is_gated`。
- 追加 (kernel_invariants): `t369_kv_async_is_wired` — 同期 `to_slice` が
  `!async_started` の内側にだけ 2 箇所、非同期の開始が
  `staged.is_some() && pf_kv_async_on()` の 2 箇所、`pf_full_out` が
  ミラー直書きのときホスト KV の長さを要求しない、呼び出し側が push を
  `pf_full_out` の後ろへ回す、`copy_into` 2 箇所 + `discard`、pinned 直送の
  env 名と入口、を字面で確認する。
- 緑は発火の証拠ではないので、**発火は lab の台帳で数えた**: `full` の
  exposed 386/389/368 -> 214/214/214/210 ms、`full` の cpu 241/217 -> 543/539 ms
  (pinned -> ks/vs の写しが cpu 列へ移った)、p1 では cpu 538.6 -> 500.5 ms。
  d2h の呼び出し数は 168 のまま (コピーは同じ回数、待ち方だけが違う)。

## 7. 次にやること (この 3 ラウンドで分かったこと)

- **`ssm` の GDN 状態書き戻し (14k で 823.69 MB / 504 回、3,826 で 235.34 MB / 144 回)
  も同じ同期 D2H。** 今回の変更で `full` の待ちが消えたので、次はここが同じ形の
  直列点として残っている。t343 / t365 が試したのは「書き戻す**回数**を減らす」
  方向 (層の最後のチャンクだけ) で壁は動かなかったが、**回数はそのままで
  待たない**方向 (今回のやり方) は未測定。
- `other` の 4,110 呼び出し / 4.1 GB の pageable H2D は、サイズが
  524,800 / 627,200 / 921,600 / 460,800 / 704,000 B (512 B の端数を含む) で
  呼び出し元が特定できていない。`DENNO_PF_XFER_DBG=1` の sub が `-` なので
  副区間の外。ここも同期 H2D なので、正体が分かれば同じ手が使える。
  露出は今回の変更で 888 -> 546 ms まで下がっている。
- `DENNO_PF_PIN_STAGE` (br358 が窓と育ち方を変えた) は今回も両サイズで悪化。
  既定 off のまま。

## 8. VERDICT 行

t369a (SHA `da3ce3318`、5 COND):

```
VERDICT b1 vs x0 OK decode -1.6% prefill -3.3% md5 SAME errors 0
VERDICT c1 vs x0 OK decode -2.5% prefill -2.4% md5 SAME errors 0
VERDICT c2 vs x0 OK decode -2.3% prefill -2.4% md5 SAME errors 0
VERDICT b2 vs x0 OK decode -2.5% prefill -1.6% md5 SAME errors 0
VERDICT_DEPTH c1 c8000_cold vs x0 decode -16.3% prefill -21.0%
VERDICT baseline x0 errors 0 text ok
```

t369b (SHA `30e5fa1b7` = 既定 on、5 COND):

```
VERDICT b1 vs x0 OK decode +0.2% prefill -4.0% md5 SAME errors 0
VERDICT d1 vs x0 OK decode -1.2% prefill -5.6% md5 SAME errors 0
VERDICT d2 vs x0 OK decode -1.2% prefill -6.5% md5 SAME errors 0
VERDICT b2 vs x0 OK decode -1.4% prefill -5.6% md5 SAME errors 0
VERDICT_DEPTH d1 c8000_cold vs x0 decode -12.3% prefill -26.4%
VERDICT baseline x0 errors 0 text ok
```

t369c (SHA `546d74fd5` = on + KV_PUSH、6 COND):

```
VERDICT b1 vs x0 OK decode -9.2% prefill -4.1% md5 SAME errors 0
VERDICT d1 vs x0 OK decode -3.5% prefill -4.1% md5 SAME errors 0
VERDICT p1 vs x0 OK decode -9.9% prefill -4.1% md5 SAME errors 0
VERDICT d2 vs x0 OK decode -10.1% prefill -4.9% md5 SAME errors 0
VERDICT b2 vs x0 OK decode -9.7% prefill -5.7% md5 SAME errors 0
VERDICT_DEPTH p1 c8000_cold vs x0 decode -51.6% prefill -69.3%
VERDICT baseline x0 errors 0 text ok
```

(VERDICT は先頭 COND = x0 との比。x0 は毎回いちばん冷えた 1 本目なので、
採否は §3 の表 (同じ run の b2 と、3 ラウンド分の中央値) で読んだ。)

## 9. GPU で確認してほしいこと

- 既定 on のまま、本番 tint で 3,826 / 14k の cold prefill が §3 と同じ向きに
  動くこと (`full` の露出 214 ms 前後、`gpu_busy` 不変)。
- 既定の挙動が 1 バイトも変わっていないこと: `DENNO_PF_KV_ASYNC=0` の run が
  従来と同一の md5・同一の decode になる (t369a/b/c の base COND がそれ)。
- 長い文脈 (14k 以上、ミラーが LRU で落ちる領域) で、`[pf-full]` の
  「ミラーが落ちた」bail が頻発しないこと。頻発すると prefill が従来経路へ
  戻る (値は正しいが速くない)。
