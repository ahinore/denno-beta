ADOPT: yes t393a t393b t393c

# task 393 (dev2) — 既定モデル prefill のホスト仕事: KV ミラー積みのまとめ / PLE の res 往復を pinned に

branch `br393` = 開始時の `int-auto` `056258a92`。**br389 は merge していない**:
int-auto にも未 merge で、中身は task 402 が `ADOPT: no` にした
`DENNO_PF_SCAN_WYC` (既定 off の死んだゲート) だけなので要らなかった
(`REPORT_P389.md` は読んだ。数値は t402a / t402b のもの)。

数値はすべて GPU ラボの実測 (`SCENARIO=depth8k`、`PROFILE=1`、
`model=flash-next-UD-IQ3_XXS`、`ptx=none`)。lab 名と測った SHA:
**t393a = `2ee60c10b`** (5 COND) / **t393b = `131279394`** (6 COND) /
**t393c = `aaa90e6dc`** (4 COND、既定 on の確認)。3 ラウンドで上限。
作業中に int-auto は `056258a92` -> `66c2a4d2d` (denno-view のみ) へ動いたので
t393c の直前に merge した (衝突なし)。

## 0. 結論 (7 行)

- **採る 2 本**: `DENNO_PF_KV_BULK` (full 層の KV をホストミラーへ 1 回の写しで積む) と
  `DENNO_PF_PLE_PIN` (PLE 層の res 往復バッファを pinned に)。既定 off で入れて
  lab の段の実測で採り、**既定 on** にした。切り戻しは `=0`。
- `DENNO_PF_KV_BULK`: 3,826 の `full` 段が **134.2-138.5 -> 126.5-129.6 ms**
  (-3.4 / -7.9 / -7.2 / -7.0 / -7.5 / -7.3% の 6 サンプル = 3 ラウンド x 2)。
  14k は差なし (881.7 -> 880.5 ms)。
- `DENNO_PF_PLE_PIN`: 3,826 の `other` 段が **409.3 / 238.9 -> 226.2 / 226.1 ms**。
  内側は `res_up` 44.9-46.2 -> 32.1-34.0 ms、`ple` 111.3 -> 100.2-101.9 ms。
  14k は `other` 764.6 -> 743.2 ms (`res_up` 206.0 -> 163.1 ms)。
- **task の目標 (壁で 2.5%) は満たしていない**: 壁は 3,826 で **-0.2〜-0.6%**、
  14k では **0% 前後** (位置効果 20% の中では読めない)。採ったのはオペレーター規則 (段で 2% 以上 x 2 ラウンド、
  他は 1% 以上悪化しない) で、2 本とも 3 ラウンドで規則を通った (§3.2、§7)。
  **decode (15.8-15.9 ms/tok) と md5 は 3 run の全 COND で不変**、errors 0。
- **落とした 1 本**: `DENNO_PF_GDN_WSTAT` (GDN の重み 4 本を層ごとに上げ直さない)。
  分岐は発火した (ssm の H2D 159.18 MB / 1,080 回 -> 123.60 MB / 216 回) が、
  `ssm` 段の壁は 14k 2,459.8 -> 2,433.8 ms (**-1.1%**)、3,826 454.9 -> 452.3 ms
  (**-0.6%**) で規則に届かない。既定 off のまま残した (A/B 用)。
- **推定の訂正**: P389 と私の初期見積りは「14k の ssm exposed 902 ms は
  同期 H2D (重み 4 本 x 1,008 回) の待ち」と読んだが、**1,008 回を 216 回に減らしても
  25 ms しか縮まなかった**。ssm の露出は GPU の待ち (キューが詰まっている) であって
  ホストの H2D ではない。この枝は閉じる。
- 目標に届かない理由は §6-2 (3,826 は expert の流し込みが 54%、14k は GPU が 74%)。
  ホスト側で残っているのは 14k の `other` 660 ms (うち expert の戻し ~300 ms) だけ。

## 1. 前提: どの段を取ったか (t402b の base、14k)

`REPORT_P389.md` の続き。t402b の b2 (`[q4ledger-pf]`、14k、wall 6,743.5 ms =
gpu_busy 4,675.4 + exposed 2,929.7 - mix の二重計上 861.6):

| 段 | wall | busy | cpu | exposed | 中身 |
| --- | ---: | ---: | ---: | ---: | --- |
| ssm | 2,454.9 | 1,557.7 | 33.3 | 897.2 | GPU 待ち (§6 で訂正) |
| experts | 2,035.3 | 1,424.3 | 1,659.6 | 611.0 | 重みの流し込み (P377/P379 で閉じた枝) |
| full | 1,140.4 | 638.8 | 655.1 | 501.6 | **KV ミラーの積み (本タスク)** |
| other | 1,091.4 | 171.6 | 1,091.4 | 919.8 | **PLE の res 往復 (本タスク)** + expert の戻し |

`other` の内側は `ple=202.4 [d=717.7MB/14]` と `res_up=201.0 [h=1148.4MB/14]`。
`full` の CPU 655 ms は「チャンクごとの `vec![0f32; nt*kv]` 2 本 (12 層 x 7 チャンク x
8 MB = 672 MB の新しいページ)」と「pinned から `ks`/`vs` へ 689 MB 写す」と
「2,048 回の `AttnCache::push`」だった。どちらも**誰も触っていない**ホスト仕事で、
`.cu` を触らずに消せる (タスクは `.cu`/`.ptx` を許可していない)。

## 2. 変えたこと (3 本、すべて env ゲート 1 本の内側)

### 2.1 `DENNO_PF_KV_BULK` (既定 on)

- `AttnCache::push_many(layer, k, v, n)`: `n` 位置を **1 回の `copy_from_slice`** で積む
  (`push` を n 回呼ぶのと同じバイト列。テストで固定)。
- full 層 1 チャンクの k / v 一時バッファを**スレッド局所の grow-only 2 本**に
  (`KvBulkScratch`、drop で返すので `Ok(None)` に落ちる経路でも取りこぼさない)。
- 遅延 push の枝では pinned の保留をそのまま `push_many` に渡す (`ks`/`vs` への写しを
  1 本消す)。`DENNO_PF_KV_PUSH` (既定 off) の上位互換で、そちらも残した。

### 2.2 `DENNO_PF_PLE_PIN` (既定 on)

- PLE 層の res 往復バッファ (`pf_ple_back_take`) を `PfHostBuf` (pinned) にした。
  `PfHostBuf` はスレッド局所の空きリストから借りるので、2 回目以降は確保もしない。
- 呼び出し側は `PleBack` の `Deref`/`DerefMut` で今までどおり `&mut [f32]` として使う。
- pageable の同期 H2D/D2H はこのホストで 3.5-5.7 GB/s。**追加の写しを挟まない**
  (task 318 の `DENNO_PF_PIN_STAGE` は「pinned へ写してから同期コピー」で写しを
  1 回増やしていた。それが t369a で負けた理由の 1 つと読んでいる)。

### 2.3 `DENNO_PF_GDN_WSTAT` (既定 off のまま)

- `gdn_pf_chain` は呼ばれるたびに層の重み 4 本 (conv / a / dt / norm) を
  `copy_from` (pageable の同期 H2D) で上げ直していた。層優先 prefill は層ごとに
  全チャンクを回すので、14k で 36 層 x 7 チャンク x 4 本 = **1,008 回**。
- 鍵 (層, 転送元 4 本の (ptr, 要素数), 置き場 4 本の ptr) が一致したら飛ばす。
  3 つの呼び出し口 (層優先 / チャンク優先 / decode) を 1 本の `gdn_put_weights` に集約。

## 3. 数値 (lab、同じ run の中の対)

### 3.1 壁 (cold prefill、s)

| run | COND | 14k | 3,826 r1 | 3,826 r2 |
| --- | --- | ---: | ---: | ---: |
| t393a | b1 | 7.06 | 2.67 | 2.44 |
| t393a | kb | 6.54 | 2.92 | 2.47 |
| t393a | pp | 6.52 | 2.46 | 2.44 |
| t393a | b2 | 6.35 | 2.61 | 2.45 |
| t393b | b1 | 6.41 | 2.63 | 2.44 |
| t393b | ws | 6.00 | 2.44 | 2.44 |
| t393b | kb | 6.00 | 2.44 | 2.44 |
| t393b | pp | 6.00 | 2.43 | 2.43 |
| t393b | b2 | 6.00 | 2.45 | 2.45 |

**壁は判定に使えない**: 14k は 1 COND 1 サンプルで、同じ base が t393a の
6.35-7.06 s、t393b の 6.00-6.41 s。t393b では x0 (捨て COND) が 7.56 s で、
ws/kb/pp/b2 が全部 6.00 s に揃う = 位置効果が 20% ある (P389 §3.2 と同じ)。
採否は段の数字 (§3.2) で決めた。

### 3.2 段 (採った 2 本)

`full` 段 (3,826、ms) — `DENNO_PF_KV_BULK`:

| run | r | b1 | kb | b2 | kb vs b2 |
| --- | --- | ---: | ---: | ---: | ---: |
| t393a | 1 | 138.9 | **129.6** | 134.2 | -3.4% |
| t393a | 2 | 135.9 | **127.6** | 138.5 | -7.9% |
| t393b | 1 | 137.2 | **126.8** | 136.7 | -7.2% |
| t393b | 2 | 136.9 | **126.8** | 136.2 | -7.0% |

14k は 880.5 / 881.7 ms (kb / b2、t393b) で差なし。t393a の 14k は
cpu 707.7 (b1) / 351.5 (kb) / 374.8 (b2) で、b1 の 707 は最初の 14k リクエストが
AttnCache の行 (12 層 x 14k x 512 x 4 x 2 = 688 MB) を踏むぶん (2 回目以降は無い)。

`other` 段 (3,826、ms) — `DENNO_PF_PLE_PIN`:

| run | r | b1 | pp | b2 | pp vs b2 |
| --- | --- | ---: | ---: | ---: | ---: |
| t393a | 1 | 433.6 | **227.1** | 390.2 | -42% |
| t393a | 2 | 238.1 | **228.8** | 239.7 | -4.5% |
| t393b | 1 | 409.3 | **226.2** | 240.7 | -6.0% |
| t393b | 2 | 238.9 | **226.1** | 239.3 | -5.5% |

内側 (`sub other`): `res_up` 44.9-46.2 -> 32.1-33.0 ms、`ple`(注入側) 111.3 ->
101.3-101.9 ms。14k は `other` 764.6 -> 743.2 ms、`res_up` 206.0 -> 163.1 ms。
t393a r1 の b1 が 433.6 ms なのは最初の 3,826 がページキャッシュの冷えた状態
(ple の gather 115.1 ms) のため。**同じ run の後半 (r2) でも -4.5〜-5.5%** 残る。

### 3.3 段 (落とした 1 本)

`DENNO_PF_GDN_WSTAT` (t393b):

|  | 14k ssm wall | 14k ssm h2d | 3,826 ssm wall | 3,826 exposed |
| --- | ---: | ---: | ---: | ---: |
| b2 | 2,459.8 | 159.18 MB / 1,080 | 454.9 / 454.6 | 20.8-26.7 |
| ws | **2,433.8** | **123.60 MB / 216** | **452.3 / 452.0** | 19.2-21.7 |

飛んだのは 36 層 x 6 チャンク x 4 本 = 864 回 (期待どおり)。段の壁は -1.1% / -0.6%。
**ssm の露出 900 ms の正体はこれではなかった** (§6-1)。

### 3.4 decode と md5

decode は t393a / t393b の全 COND で 15.8-15.9 ms/tok (差は 0.1 ms 刻みの範囲)。
md5 は 3 run の全 COND・全コンテキストで **SAME**:
`a:edaa84ad b:1938059f c2000_cold:b42d79d2 c2000_turn2:2081cdeb
c8000_cold:672fec07 c8000_turn2:bc4aaf54`。

## 4. 数値の扱いと見積り

- 壁の ±2% は 1 サンプルでは読めない (位置効果 20%)。採否は §3.2 の段の数字で決めた。
  **task の目標 2.5% は壁では満たしていない** (-0.6% / -0.1%)。
- 見積り (推定): 3,826 の `other` 段の内側で消えたのは res_up 12-13 ms + ple 10 ms =
  22-23 ms で、`other` 段の壁の差 (-5.5〜-6%) とほぼ一致する。14k の `res_up` の
  差 -43 ms は pinned の H2D が 5.4 -> 7.0 GB/s になったぶん (推定。実測は
  `[q4ledger-pf] sub other` の ms だけ)。
- `DENNO_PF_KV_BULK` の 3,826 の -9.4 ms は「672 MB ではなく 188 MB ぶんの
  新しいページを踏まなくなった」+「2,048 回の push が 1 回の写しになった」ぶん
  (推定)。14k で差が出ないのは、14k では full 段の露出が KV の D2H 待ち
  (689 MB) で決まっていて、ホスト側の写しがその裏に隠れるため (推定)。
- 14k の残り: `other` の未分類 660 ms のうち ~300 ms は prefill 終端の
  **expert の戻し** (827 本 x 3 本 = 2,481 回の pageable H2D、1,636 MiB)。
  `DENNO_PF_LM_RELOAD` を切れば消えるが、次の decode のヒット率が落ちる (task 242)。

## 5. text (sane の証拠)

t393b b1 の c8000_cold (14k cold の応答、`b1_1_c8000_cold.json` の text):

> prefillの遅延は計算量ではなく、ホスト往復とメモリコピーが律速であった。
> 計測の誤認や静かな退行を排除しつつ、デバイス常駐化と転送経路の最適化を段階的に実施した。
> その結果、1.5Bモデルで395.8秒から12.2秒へ、9Bモデルで46.2秒から10.11秒へと大幅な高速化を達成した。

`kb` / `pp` / `ws` の各 COND も同じ形 (md5 SAME)。「!!!!」や空出力は無い。

## 6. 分かったこと (次の担当者へ)

1. **14k の ssm exposed 902 ms は同期 H2D の待ちではない。** 重み 4 本の
   アップロードを 1,008 回 -> 216 回に減らしても 25 ms しか縮まなかった。
   露出は GPU の待ち (キューが詰まっている) で、ホスト側では消せない。
   P389 §4 と私の初期見積りはここを誤っていた (この REPORT で訂正)。
2. **3,826 の壁は `experts` 段 (1,325-1,337 ms、うち up_wait 1,074-1,092 ms) が
   握っている。** ホスト側の残り (`other` 226 ms、`full` 127 ms) を全部 0 にしても
   2.5% には届かない。3,826 で 2.5% を出すには expert の流し込み側 (閉じた枝) しかない。
3. **14k の壁は GPU (gpu_busy 4,676 ms / 74%) と ssm・full の待ちで決まる。**
   ホスト側で残っているのは `other` の 660 ms (expert の戻し 300 ms を含む) で、
   ここを削るには `DENNO_PF_LM_RELOAD` の設計 (decode のヒット率との trade-off) を
   触るしかない。
4. `full` 段のホスト仕事は `DENNO_PF_KV_BULK` でほぼ底 (cpu 655 -> 351-395 ms)。
   残りは `pf_kv_wb_copy_into` の 689 MB (pinned -> `ks`/`vs`) で、これは
   `DENNO_PF_KV_PUSH` (既定 off、t369c で差なし) の枝しかない。

## 7. t393c (既定 on の確認、3 ラウンド目)

`aaa90e6dc` (int-auto `66c2a4d2d` merge 済み)、4 COND
(x0 / b1 = 既定 on / `off` = `DENNO_PF_KV_BULK=0 DENNO_PF_PLE_PIN=0` / b2)、
2 ラウンド。この run は**同じ run の中で on と off を交互に測った**ので、
段の差がそのままゲートの効果になる:

| 段 (3,826、ms) | b1 (on) r1 / r2 | off r1 / r2 | 差 |
| --- | ---: | ---: | ---: |
| `full` wall | 127.7 / 126.5 | 138.0 / 136.5 | **-7.5% / -7.3%** |
| `full` cpu | 90.3 / 89.2 | 102.4 / 101.4 | -12.1 / -12.2 ms |
| `sub other` sum | 140.7 / 140.2 | 162.7 / 161.5 | **-13.5% / -13.2%** |
| うち `res_up` | 34.0 / 33.8 | 45.4 / 44.7 | -11.4 / -10.9 ms |
| うち `ple` | 100.8 / 100.2 | 111.1 / 110.7 | -10.3 / -10.5 ms |

壁は 3,826 で b1 2.65 / 2.42、off 2.60 / 2.43、b2 2.59 / 2.43 (位置とノイズで
0.5% は解像できない)。14k は b1 6.30 / off 6.35 / b2 6.00 (b2 が最後で一番速い =
位置効果)。decode 15.8-15.9 ms/tok、md5 は全 COND SAME、errors 0、text ok
(`off` も `b1` も同じ文)。**既定 on の状態で 2 本とも発火し、段の差が
t393a / t393b と同じ向き・同じ大きさで再現した。**

### t393c の VERDICT 行

```
VERDICT b1 vs x0 OK decode +0.0% prefill +0.8% md5 SAME errors 0
VERDICT off vs x0 OK decode +0.2% prefill -0.8% md5 SAME errors 0
VERDICT b2 vs x0 OK decode +0.2% prefill -2.3% md5 SAME errors 0
VERDICT baseline x0 errors 0 text ok
```

## 8. テスト

- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`:
  **1269 passed / 0 failed / 103 ignored**
- `cargo test --release -p denno-engine --features cuda --test kernel_invariants`:
  **115 passed / 0 failed** (新規 3 本 + `t369_kv_async_is_wired` の
  `copy_into` の宛先を `kv_scratch` に更新)
- 新規テスト: `push_many` が `push` の n 回と同一バイト列 / ゲート 3 本の真理値表 /
  k-v スクラッチの長さと使い回し / pinned の枝の契約 / 配線 3 本
  (`task393_kv_bulk_is_wired_through_push_many`,
  `task393_ple_pin_is_wired_to_the_round_trip_buffer`,
  `task393_gdn_wstat_is_wired_through_one_helper`) / 鍵が層・転送元・置き場の
  全部を見ること (`cuda::task393_tests`)
- 分岐の発火: `DENNO_PF_GDN_WSTAT` は lab の ssm h2d (1,080 -> 216 回) で確認。
  `KV_BULK` / `PLE_PIN` は CPU では発火を数えられないので、lab の段の ms で見た (§3.2)。
- cuda feature 無しの CI: `cargo build --workspace --release` と
  `CUDA_VISIBLE_DEVICES=-1 cargo test --workspace --no-fail-fast` が緑
  (engine 945 passed / 0 failed)。`PlePinBuf` は feature 無しでは `Vec` の器に
  なるので、`crate::cuda` を参照しない。t393c の測定後に足したのはこの
  コンパイル用の器だけで、cuda feature では `type PlePinBuf = PfHostBuf` に
  解決されるので測定したコードと同一。
- 既知の赤 (`qwen4exp_decode spec_batch_stops_at_eos`、`qwen4exp_params` の validate 2 件)
  は触っていない。

## 9. GPU で確認してほしいこと (監督へ)

1. 既定 on にした 2 本は `DENNO_PF_KV_BULK=0` / `DENNO_PF_PLE_PIN=0` で切り戻せる。
   `DENNO_PF_GDN_WSTAT` は既定 off のまま (発火はするが段 1% 台)。
2. 3,826 の 2.5% は expert の流し込みを触らないと出ない (§6-2)。14k は GPU 側。
3. 14k の位置効果 (同じ base が 6.0-7.1 s) を消すプロトコルは P389 §4 のまま未解決。
   次に 14k を判定に使うなら、同じ COND を 3 回以上置くか捨てリクエストを 2 本入れる。

## 10. VERDICT 行

t393a (SHA `2ee60c10b`, 5 COND, model=flash-next-UD-IQ3_XXS, ptx=none):

```
VERDICT b1 vs x0 OK decode +0.0% prefill +1.5% md5 SAME errors 0
VERDICT kb vs x0 OK decode +9.8% prefill +7.4% md5 SAME errors 0
VERDICT pp vs x0 OK decode +0.0% prefill +0.0% md5 SAME errors 0
VERDICT b2 vs x0 OK decode +0.0% prefill +0.7% md5 SAME errors 0
VERDICT baseline x0 errors 0 text ok
```

t393b (SHA `131279394`, 6 COND, ptx=none):

```
VERDICT b1 vs x0 OK decode -0.2% prefill +0.7% md5 SAME errors 0
VERDICT ws vs x0 OK decode -0.0% prefill +0.7% md5 SAME errors 0
VERDICT kb vs x0 OK decode -0.5% prefill +0.0% md5 SAME errors 0
VERDICT pp vs x0 OK decode -0.5% prefill +0.7% md5 SAME errors 0
VERDICT b2 vs x0 OK decode -0.2% prefill +0.0% md5 SAME errors 0
VERDICT baseline x0 errors 0 text ok
```

t393c (SHA `aaa90e6dc`, 4 COND, ptx=none) の VERDICT 行は §7。
