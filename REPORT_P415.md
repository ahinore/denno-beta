ADOPT: yes t415b t415c

# task 415 (dev2) — 既定モデル prefill: 層ループ前の埋め込み (token_embd の行引き) を N 本で引く

branch `br415` = 開始時の `int-auto` `ff1a1eccb` (**br411 は merge していない**: 中身は
`DENNO_PF_LM_PIPE` の既定 off のゲートと切り出しだけで `ADOPT: no`。この task の変更は
それを必要としない)。作業中に int-auto は 2 回動いた: `ad1cfe1b5` (merge: br420 =
UD-Q4_K_XL の 14k cold -23%) と `c5bfab6f9` (merge: br418 = mix down の dn5 カーネル)。
どちらも終端で merge した (br420 は衝突なし。br418 は `kernel_invariants.rs` の末尾だけ
衝突 = 両側がテストを追記 -> 両方残した)。

数値はすべて GPU ラボの実測 (`SCENARIO=depth8k`、`PROFILE=1`、`model=flash-next-UD-IQ3_XXS`、
`ptx=none`)。lab 名と測った SHA:
**t415a = `ff1a1eccb`** (3 COND、基準の取り直し) /
**t415b = `1d5c8095f`** (6 COND、`p16` / `ov` / `p32` の掃引) /
**t415c = `a02f851d4`** (5 COND、**全部 `DENNO_START_DROP_CACHE=1`** で page cache を揃えた A/B)。
3 ラウンド (上限いっぱい)。

## 0. 結論 (6 行)

1. **採る。ただし task の目標 (壁で 2.5%) は満たしていない。** 採ったのは
   オペレーター規則 (段で 2% x 2 ラウンド、他は 1% 以上悪化しない) で、段は
   **14k cold の `[pre-kprof] embed` が 101.5 / 103.8 ms -> 29.6 / 27.1 ms (-71% / -74%)**
   (t415c、2 ラウンド、page cache を全 COND で揃えた A/B)。
2. **壁への寄与は 14k で -1.0〜-1.4%** (t415c: off2 7,082.4 -> d2 6,981.2 ms、off2 -> d1 は
   +0.3%、t415b: b2 5.70 -> ov 5.63 s)。**3,826 は変わらない** (t415b の round 2 で
   b1 2,338.2 / b2 2,306.2 -> p16 2,282.5 / ov 2,290.3 = -0.9%〜+0.7%、t415c の冷えた
   3,826 は全 COND 8.25-8.27 s で同一)。decode は 13.2-13.6 ms/tok で不変。
3. **変えたこと**: 層優先 prefill の層ループ前の埋め込み (全トークン x 1 行 2,100 B の
   ランダム読み) は **1 トークン 1 回の直列ループ**だった。行は独立なので
   `DENNO_PF_EMBED_PAR` (既定 **16** 本) で引く。さらに `DENNO_PF_EMBED_OVL` (既定 **on**)
   で PLE の gather (96 本 + パイプ、task 349/357) を背景スレッドへ出し、その裏で引く。
4. **値は 1 bit も変えていない**: 全 3 run・全 COND・全文脈で **md5 は SAME**
   (`edaa84ad / 1938059f / b42d79d2 / 12fc5c36 / 672fec07 / 98e879d2`)、errors 0、text sane (§8)。
5. **なぜこの塊か**: 14k cold の台帳 (t415a) で最大の未着手は `other`
   (**1,402.7 ms / 22%**、ssm 2,222.6 と experts 1,952.6 は 397/402/407 と 250/416/411 が
   詰めており、`other` は 393 (PLE pin / KV bulk) と 394 (restore MT) が一部だけ触った)。
   その中身は層ループ前の **ple 240.0 + embed 15.1 + res_up 152.7** と、層ループ内の
   未計測 ~450 ms。このうち **埋め込みだけが直列のランダム読み**として残っていた
   (§1.2)。
6. **分かったこと**: 埋め込みのコストは page cache 次第で 27-166 ms 動く
   (14k、直列)。並列化で床は ~27 ms。**層ループ前の 2 つの読み (PLE 表 90 B x 167k 回、
   token_embd 2,100 B x 14k 回) は同じディスクを取り合う**ので、重ねるのが正しい
   (t415b の `ov` が 14k で 5.63 s = 6 COND で最速)。

## 1. Do 1 / Do 2: 何を狙ったか (t415a のプロファイル)

基準 COND はタスク指定の 4 本 (`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000
DENNO_EXPERT_GPU_MIN_HIT=0 DENNO_GPU_WEIGHT_MB=12000`)、`x0` (捨て) のあと `b1` / `b2`。
t415a は 3,826 が b1 2.44 / b2 2.33 s、14k が b1 6.48 / b2 5.95 s の run
(最後の `b2` が最も暖かい。**14k はラウンド 1 にしか無いので、1 COND = 1 サンプル**)。

### 1.1 段の台帳 (t415a b1、`[q4ledger-pf]`、ms)

| 段 | 3,826 wall | / busy | 14k wall | / busy |
| --- | ---: | ---: | ---: | ---: |
| ssm | 412.3 | 360.5 | **2,222.6** | 1,316.2 |
| experts | 1,284.8 | 531.4 | **1,952.6** | 1,423.6 |
| other | 230.7 | 47.4 | **1,402.7** | 171.5 |
| full | 128.1 | 88.9 | 880.1 | 637.1 |
| mix | 260.9 | 242.4 | 21.0 | 882.1 (別 stream) |
| close | **2,316.8** | **1,270.5** | **6,479.0** | **4,430.4** |

- 3,826 の露出 1,046 ms のうち **1,035 は `up_wait`** (expert の上げ待ち。22,425.6 MB /
  1.09 s = 20.6 GB/s = DRAM 床、P385 §2.2 を追認)。**ここは閉じている** (P411 §6-4)。
  `up_MB` も 10,938 本 = distinct な (層, expert) の数で最小。
- 14k の露出 2,910 ms のうち **`other` が 1,231** (1 位)、ssm 906、experts 529、full 243。
  14k は GPU が 68% しか埋まらず、**層ループ前のホスト仕事が丸ごと露出する**。

### 1.2 `other` の中身 (t415a b1 の `[pre-kprof]`、14k)

`embed 111.9 | ple 504.4 | layer 5,688.3 | other 174.4 ms`。
PLE の gather は `rows=166,942 / bytes=15.0 MB / avg_read=90 B / read=37,225 ms (累積)` =
**90 B のランダム pread が 167k 回**で、既に 96 本 + パイプ (task 349/357)。
一方 **埋め込みは直列** (14k x 2,100 B のランダム読み、表は 317 MB) で、
t415a 111.9 / t411b 325.7 / t415b 165.9 ms と page cache 次第で 3 倍動く。
`sub other` の内訳は ple 240.0 / res_up 152.7 / embed 15.1 / combine 3.9 / res_park 2.5 で、
**残り ~990 ms は未計測** (§6-3 に次の担当者向けの地図)。

### 1.3 選んだ理由と、外した理由

- **埋め込み**: 層ループの前に全トークンぶん走る = **GPU が遊んでいる時間**。
  行独立 = ビット一致で並列化できる。PLE の gather (同じ場所、同じ性質) は
  task 349/357 で 96 本 + パイプにして **14k で -5%/-13%** を取っており、
  **埋め込みだけが直列のまま残っていた**。
- 外したもの: ssm の残り (397 wy5 / 402 C=16 / 407 tc1+wy7 が詰めた)、experts の
  IQ TC (250 で既定 on)、3,826 の上げ待ち (P411 で「隠す」は +80% バイトで負け、
  バイトは最小、写しは DRAM 床)、`full` (flashdec の分割は P196/P201 の未着手玉)、
  `other` の未計測 ~990 ms (正体を特定できず、当てずっぽうで触らない)。
  `.cu` / `.ptx` は触っていない (新しいカーネルは作っていない)。

## 2. 変えたこと (ゲート 2 本 + 1 か所の切り出し)

### 2.1 `DENNO_PF_EMBED_PAR=N` (新規、**既定 16**)

`linear_gguf::embed_rows_into(l, tokens, hc, out, threads)` を追加。`tokens` の行を
`dequant_row_into` で f32 にし、`hc` 本に複製して `out` (`tokens.len() * hc * e`) へ書く。
**行ごとに直列経路と同じ関数を同じ順序で呼ぶ**ので値はビット一致。行を
`par_chunks_mut(hc * e)` で分け、専用プール (OnceLock で 1 回だけ作る) で回す。
`threads <= 1` はプールを作らず従来と同じ逐次。形の門は fail closed
(長さ違い / 行の範囲外 / 空は Err)。

`forward_prefill_batch_resident_lm` 側は `res_all` を `vec![0.0; nt_all * hc_dim]` で
先に確保し、並列経路と直列経路が同じ場所へ書く (直列は `copy_from_slice`)。
**直列経路はそのまま残してある** (`DENNO_PF_EMBED_PAR=0` で切り戻し)。

### 2.2 `DENNO_PF_EMBED_OVL` (新規、**既定 on**)

`ple_wide` (task 349) のとき、PLE の gather を **背景スレッド**へ出し (96 本のプールは
同じ)、その裏で埋め込みを引く。join は gather の直後で、失敗は従来どおり Err。
読み手が 2 本になるだけで、読む行・綴じ方は変わらない。

### 2.3 既存テスト 1 本の参照先を更新

`t349_ple_pipe_widens_the_lm_gather_pool` の「専用プールで gather を回している」検査を
`Ok(p) => ples_all = p.install(run)?,` -> `Ok(p) => p.install(run)?,` +
`ples_all = match ple_bg {` に更新した (検査内容 = 専用プールで回す、は同じ)。

## 3. 数値 (lab、同じ run の中の対)

### 3.1 t415b (`1d5c8095f`、6 COND、掃引)

| COND | env | 14k cold s | 14k embed ms | 14k ple ms | 3,826 (r2) ms |
| --- | --- | ---: | ---: | ---: | ---: |
| x0 (捨て) | — | 6.91 | (—) | (—) | 2,377.0 |
| b1 | — | 6.53 | 165.9 | 508.3 | 2,338.2 |
| p16 | `EMBED_PAR=16` | 5.93 | **24.4** | 379.2 | **2,282.5** |
| ov | `=16` + `OVL=1` | **5.63** | 32.0 | 50.5 | 2,290.3 |
| p32 | `EMBED_PAR=32` | 5.62 | 27.3 | 57.9 | 2,287.0 |
| b2 | — | 5.70 | 108.3 | 63.4 | 2,306.2 |

- **埋め込み**: 直列 165.9 (b1) / 108.3 (b2) -> 24.4-32.0 ms。**-71%〜-85%**。
- **14k の壁**は COND の位置で 6.91 -> 5.62 まで動く (page cache が回を追うごとに暖まる) ので、
  **`vs x0` の行は採否に使わない**。同じ run の後半 3 本 (ov 5.63 / p32 5.62 / b2 5.70) で見ると
  ov / p32 は b2 より -1.2% / -1.4%。
- 3,826 は round 2 (全部暖かい) で p16 -2.4% / p32 -2.2% / ov -2.0% 対 b1、対 b2 で
  -1.0% / -0.8% / -0.7%。

### 3.2 t415c (`a02f851d4`、5 COND、**全 COND `DENNO_START_DROP_CACHE=1`**)

page cache を COND ごとに落として**位置効果を消した** A/B (task 353/410 と同じ手)。
`off` = `DENNO_PF_EMBED_PAR=0` (従来)、`d1` / `d2` = 既定 (16 + ovl)。

| COND | 14k cold s | 14k embed ms | 14k ple ms | 14k layer ms | 3,826 cold s |
| --- | ---: | ---: | ---: | ---: | ---: |
| x0 (捨て) | 7.53 | (—) | (—) | (—) | 8.25 |
| off | 7.64 | 101.5 | 420.2 | 6,770.4 | 8.27 |
| d1 | 7.10 | **29.6** | 408.0 | 6,414.4 | 8.25 |
| off2 | 7.08 | 103.8 | 420.0 | 6,409.2 | 8.27 |
| d2 | **6.98** | **27.1** | 419.0 | 6,384.6 | 8.25 |

- **埋め込み (段)**: off 101.5 / off2 103.8 -> d1 29.6 / d2 27.1 = **-71% / -74%**。
  2 ラウンド (t415b と t415c) とも同じ向き・同じ大きさ。
- **14k の壁**: off2 7,082.4 -> d2 6,981.2 = **-1.4%**、off2 -> d1 は +0.3%。
  **-1.0〜-1.4% が正直な線** (埋め込みの節約 ~75 ms と一致)。
- **3,826 は冷えた状態では全 COND 8.25-8.27 s で同一** (壁は冷えたモデル読みが決める)。
- `ple` 420 ms は全 COND 同じ (重ねても gather の総量は変わらない) = **他は悪化していない**。
- **OVL が効いている証拠** (t415c、同じ run): 既定の d1 / d2 は `embed 29.6 / 27.1` が
  `ple 408.0 / 419.0` の**内側に隠れて**いる (層ループ前の合計 437 / 446 ms)。従来は
  `embed 101.5 / 103.8` + `ple 420.2 / 420.0` = **521 / 524 ms**。差 -84 / -78 ms が
  そのまま壁の差 (-101 ms) になる。

### 3.3 decode / 値

- decode: t415b 13.3-13.6、t415c 13.2 ms/tok で全 COND 同じ (0.1 ms/tok 刻み)。
- **md5 は 3 run x 全 COND x 全文脈で SAME**、errors 0。
- 14k 後の decode も t415c で 15.3-16.6 (off 16.6 / d2 16.0 = 悪化していない)。

## 4. 数値の扱いと見積り

- **14k は 1 COND = 1 サンプル** (depth8k の 14k プロンプトはラウンド 1 だけ) なので、
  同一 run の中でも **COND の位置で最大 13% 動く** (t415b: 6.91 -> 5.62)。
  採否は (a) 段 (`[pre-kprof] embed`)、(b) t415c の page cache を揃えた対、で読んだ。
- **見積り (推定)**: 埋め込みの節約は「直列 27-166 ms -> 並列 ~27 ms」で、
  実測できた範囲は -75 ms (t415c) 〜 -135 ms (t415b)。壁への寄与は 14k で
  **-1.0〜-1.4%**、3,826 で ~0 (埋め込みが 28 ms しかない)。
  **task の目標 (2.5%) には届かない。**
- 3,826 は「他が悪化していない」ことを 2 run (暖/冷) で確認した。

## 5. テスト

int-auto `c5bfab6f9` (merge: br418) を merge した後の最終 HEAD で:

```
CARGO_TARGET_DIR=/tmp/t415 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1302 passed / 0 failed / 103 ignored
CARGO_TARGET_DIR=/tmp/t415 cargo test --release -p denno-engine --features cuda --test kernel_invariants
  -> 122 passed / 0 failed (task 415 の 1 本 + br418 の 1 本を含む)
CARGO_TARGET_DIR=/tmp/t415-nocuda cargo build --workspace --release            (cuda feature 無し)
  -> ok
CARGO_TARGET_DIR=/tmp/t415-nocuda CUDA_VISIBLE_DEVICES=-1 cargo test --workspace --release -- --test-threads=1
  -> §5.1
```

merge の衝突は `kernel_invariants.rs` の末尾だけ (両側がテストを追記)。両側を残した。

追加した 4 本:
- `task415_embed_rows_into_matches_serial` (lib) — Q8_0 / F32 / F16 で、本数 0/1/2/4 の
  `embed_rows_into` が `dequant_row` + hc 回複製と **ビット一致**。形の門 (長さ違い /
  範囲外 / 空) も固定。
- `task415_embed_par_rounds_and_defaults` (lib) — `DENNO_PF_EMBED_PAR` の真理値表
  (未設定 16、0 と 1 は直列、上限 64)。
- `task415_embed_ovl_gate_truth_table` (lib) — `DENNO_PF_EMBED_OVL` は既定 on、
  0 系だけ off。
- `task415_embed_par_and_overlap_are_wired` (kernel_invariants) — env 名 / 既定 /
  層優先 prefill がゲートを読み `embed_rows_into` を呼び、背景 gather を join し、
  **直列の従来経路も残している**こと / `res_all` を先に確保していること /
  `embed_rows_into` が `dequant_row_into` と `par_chunks_mut` を通ること。

- **発火の証拠**は lab: t415c の `[pre-kprof] embed` が 101.5 / 103.8 -> 29.6 / 27.1 ms
  (2 ラウンド)、t415b の 165.9 / 108.3 -> 24.4-32.0 ms。GPU 無しでは
  `embed_rows_into` の並列経路はテストで直接叩いてある (プール 4 本)。
- 既知の赤 (`qwen4exp_decode spec_batch_stops_at_eos`、`qwen4exp_params` の validate 2 件)
  はこの木では出ていない (0 failed)。

### 5.1 cuda 無しの workspace テスト

```
CARGO_TARGET_DIR=/tmp/t415-nocuda cargo build --workspace --release
  -> ok
CARGO_TARGET_DIR=/tmp/t415-nocuda CUDA_VISIBLE_DEVICES=-1 \
  cargo test --workspace --release -- --test-threads=1
  -> 1 回目: 1 本だけ落ちた
     denno-server --test admit_error_message :: all_slots_busy_returns_429_with_capacity_message
     ("応答が返らない: Elapsed" / このスイートが 274.5 s かかった)
  -> 単体で再実行: 7 passed / 0 failed (45.0 s)
  -> 2 回目 (workspace 全体): 63 スイート / 1,475 passed / 0 failed (exit 0)
```
このタスクの変更は engine の prefill 経路 (層ループ前の埋め込み) だけで、
denno-server の受理制御は触っていない。全体実行の負荷でスロット待ちの
タイムアウトが 1 本出た **flaky** と判断した (P411 の `slot_leak` と同じ扱い)。

## 6. 分かったこと (次の担当者へ)

1. **層ループ前のホスト仕事は 14k で 0.6-1.0 s あり、全部露出する** (GPU は遊ぶ)。
   内訳は `ple` (90 B x 167k の pread、96 本 + パイプで ~420 ms が床) と
   `embed` (2,100 B x 14k、直列 101.5-165.9 -> 並列 27)、`other` 150-345。
   **両方を重ねるのが正しく、t415b の `ov` が 14k で最速 (5.63 s)。**
2. **14k の 1 COND = 1 サンプル**で、位置効果が最大 13%。**`DENNO_START_DROP_CACHE=1`
   を全 COND に付けると対が読める** (t415c)。次に 14k を A/B する人はこれを使う。
   t415c では 3,826 も 8.25 s に揃い、3,826 の「他が悪化していない」確認にも使える。
3. **`other` の未計測 ~990 ms (14k) はまだ正体不明。** 分かっていること:
   `[pre-kprof]` の layer 5,688.3 に対し named span の和が 5,077 なので **層ループ内の
   非 span が ~450 ms**、層ループ外 (embed/ple/other) が ~790 ms。層ループ内は
   2 チャンク x 48 層 = 96 回で ~0.9 ms/回 (3,826) だが 14k は 336 回で ~1.3 ms/回 =
   **nt に比例しない**ので、**同期を伴う呼び出し**が疑わしい (候補: `pf_lm_park_mixed` /
   `pf_lm_inj_save|load` / `pf_res_park_save|load` の null stream D2D。台帳の sub は
   どれも 0.7-2.5 ms で、待ちは sub の外へ漏れている可能性)。**まず sub を足して測る**こと。
4. **PLE 表の RAM 常駐は不可** (P182: 28.8 GB。expert の page cache を追い出す)。
   90 B のランダム pread は IOPS 律速で、96 本 + パイプで ~420 ms が床
   (task 357 の 128 本は 386 ms だが外乱で跳ねた)。
5. `DENNO_PF_EMBED_PAR` は 16 でも 32 でも同じ (暖 24.4 / 27.3、冷 29.6 / 27.1)。
   **本数を増やす方向はもう効かない** (埋め込みは 27 ms で床)。

## 7. GPU で確認してほしいこと (監督へ)

1. 既定は **on** (`EMBED_PAR=16` / `OVL=1`)。切り戻しは `DENNO_PF_EMBED_PAR=0`
   (OVL は PAR>1 のときだけ効くので、これ 1 本で従来に戻る)。
   `DENNO_PF_EMBED_OVL=0` で重ねる方だけも切れる。
2. **本番で見てほしいのは「モデルを切り替えた直後 / 起動直後の 14k prefill」**。
   埋め込みの節約は表が page cache に無いときほど大きい (直列 166 ms -> 27 ms)。
   暖まった状態では -75 ms (-1.1%) しかない。
3. 3,826 は 2 run で「悪化なし」(暖 -0.7〜-2.4%、冷 同一)。実機で 2.3 s 台のままか
   だけ見てほしい。

## 8. text (sane の証拠)

t415c の d2、c8000_cold (14k cold の応答、`TXTD`):

> ご提示いただいた文書は、LLM推論エンジン（`denno-engine）における **Prefill（事前計算）処理の性能最適化の記録です。

`off` / `off2` / `d1` も同じ文で、md5 も全 COND 一致 (`672fec07` 系)。「!!!!」や空出力は無い。

## 9. VERDICT 行

t415b (`1d5c8095f`、6 COND、ptx=none、`p16` = EMBED_PAR=16 / `ov` = +OVL / `p32` = EMBED_PAR=32):
```
VERDICT b1 vs x0 OK decode -1.4% prefill -5.6% md5 SAME errors 0
VERDICT p16 vs x0 OK decode -2.0% prefill -3.2% md5 SAME errors 0
VERDICT ov vs x0 OK decode -1.1% prefill -4.8% md5 SAME errors 0
VERDICT p32 vs x0 OK decode -0.9% prefill -4.0% md5 SAME errors 0
VERDICT b2 vs x0 OK decode -1.7% prefill -2.4% md5 SAME errors 0
VERDICT_DEPTH b1 c2000_cold vs x0 decode -2.0% prefill -0.4%
VERDICT_DEPTH b1 c8000_cold vs x0 decode -7.7% prefill -5.5%
VERDICT_DEPTH p16 c8000_cold vs x0 decode -9.3% prefill -14.2%
VERDICT_DEPTH ov c8000_cold vs x0 decode -9.8% prefill -18.5%
VERDICT_DEPTH p32 c8000_cold vs x0 decode -9.8% prefill -18.7%
VERDICT_DEPTH b2 c8000_cold vs x0 decode -9.8% prefill -17.5%
VERDICT baseline x0 errors 0 text ok
```
(最後の 2 行が位置効果の証拠: **b2 は素の baseline なのに `vs x0` で -17.5%** と出る。
14k はラウンド 1 の 1 本目が極端に冷えるので、`vs x0` の 14k 行は採否に使えない。)

t415c (`a02f851d4`、5 COND、全 COND `DENNO_START_DROP_CACHE=1`、`off` = EMBED_PAR=0):
```
VERDICT off vs x0 OK decode +0.3% prefill -0.9% md5 SAME errors 0
VERDICT d1 vs x0 OK decode +0.3% prefill -1.8% md5 SAME errors 0
VERDICT off2 vs x0 OK decode +0.0% prefill -0.0% md5 SAME errors 0
VERDICT d2 vs x0 OK decode +0.0% prefill -0.0% md5 SAME errors 0
VERDICT_DEPTH off c2000_cold vs x0 decode +0.3% prefill +0.1%
VERDICT_DEPTH off c8000_cold vs x0 decode +2.4% prefill +1.5%
VERDICT_DEPTH d1 c8000_cold vs x0 decode -1.2% prefill -5.7%
VERDICT_DEPTH off2 c8000_cold vs x0 decode -1.8% prefill -6.0%
VERDICT_DEPTH d2 c8000_cold vs x0 decode -1.8% prefill -7.3%
VERDICT baseline x0 errors 0 text ok
```
(x0 は毎回いちばん冷えた 1 本目なので `vs x0` の 14k 行は採否に使わない。§3.1 の
同じ run の対と §3.2 の page cache を揃えた対で読む。)
