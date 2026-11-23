ADOPT: yes t561a t561b

# task 561 (dev2) — 既定モデル prefill: ホスト KV ミラーの伸ばし方を直した
# (`DENNO_PF_KV_RESERVE` を既定 on)。`full` 段が 3,826 で **-14.0%**、
# 14k で **-5.5%** (2 ラウンドずつ)。`kv_push` は -35% / -61%、realloc は 48/96 -> 0。
# **タスクの目標 (壁 -2.5%) は満たしていない** — 14k の壁は COND の位置で
# 8% 動いて読めず、3,826 の壁は 1.88-1.90 s の床。採ったのは段の勝ちの側。

branch **`br561`** = 開始時の `int-auto` **`7510bc00a`** (= merge: br534)。
`br557` (task 557、ADOPT: no) は int-auto に入っていないが、残したのは既定 off の
ゲートだけで本タスクの対象 (`AttnCache::push_many`) と 1 行も重ならないので merge していない。
終端で int-auto が **`dd9183fb4`** (merge: br538) へ動いたので merge した (§7)。
br538 が触るのは `exp_pf.rs` / `cuda.rs` の decode と expert 経路で、本タスクの prefill
full 経路とは重ならない。

数値はすべて GPU ラボの実測 (`SCENARIO=depth8k`、`PROFILE=1`、
`MODEL=flash-next-UD-IQ3_XXS`、baseline COND = int-auto の既定 +
`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0
DENNO_GPU_WEIGHT_MB=12000`、先頭に捨て COND、末尾に base の繰り返し)。

lab 名と SHA:
- **t561a** = `086d136ed`、COND: x0 / b / **r** (RESERVE) / **rh** (RESERVE+HPAGE) / b2
- **t561b** = `97a935682` (RESERVE 既定 on)、COND: x0 / b / **off** (`DENNO_PF_KV_RESERVE=0`) / b2

`.cu` / `.ptx` は触っていない (PTX 再生成不要)。

## 0. 結論 (7 行)

1. **採る (`ADOPT: yes`)。段の勝ちで、壁は動かない。** 3,826 cold の `full` 段は
   **51.3 / 53.0 -> 44.5 / 45.1 ms (-14.0%、2 ラウンド)**、14k cold は
   **750.0 -> 709.7 / 708.4 ms (-5.5%)**。14k の内訳は `kv_push`
   **254.8 -> 101.7 / 98.1 ms (-61%)**、`exposed` **196.5 -> 156.4 / 154.8 ms**。
2. **機序**: `AttnCache::push_many` は層に積むたびに `Vec::reserve` で倍々に伸ばし、
   **14 チャンク x 12 層で realloc を 96 回 (14k) / 48 回 (3,826)** 起こしていた
   (古い領域を捨てて新しいページを踏み直す)。要求中の minor fault は 14k で 47 万回。
   層に初めて積むときに `min(max_ctx, 32,768) * row` まで容量を確保すると
   **realloc が 0 回**になり、`kv_push` が半分以下になる。**値・順序・D2H の
   バイト列は同一** (md5 は全 COND x 全 6 文脈 SAME)。
3. **14k で壁が動かない理由がわかった**: `full` 段はもう **GPU 律速**で、
   `gpu_busy 553.3` に対して CPU は 288.6 ms しかない。CPU を 148 ms 減らしても
   壁は 709.7 / 708.4 で変わらない (`kv_wait` も 87.5 -> 86.2 で不変 = GPU 待ち)。
   3,826 は段が短く CPU が露出するので、そのまま -14% になる。
4. **`DENNO_PF_KV_HPAGE` (MADV_HUGEPAGE) は reserve の上に足すと差が無い**:
   3,826 で 46.1 / 44.6 (r) 対 45.0 / 44.8 (rh)、14k の `kv_push` で 108.9 対 103.9。
   noise の内側なので**既定 off** のまま残した。
5. **14k の残りはもうホスト側ではない** (§1)。`full` の 553.3 ms のうち
   約 450 ms が attention カーネル本体 (qkv/out の GEMM が約 100 ms、P465 の実測)、
   残りは `kv_wait` 86 と起動類。**次に触るならカーネル** (§8)。
6. **タスクの目標は満たしていない。** 14k の壁は**同じコードの b と b2 が
   5.37 / 4.97 s** (8%) と動き、`other` 段が 942 / 573 ms と 370 ms 振れるので
   41 ms の勝ちは壁では読めない。3,826 の壁は全 COND 1.88-1.90 s の床
   (experts 段が 1.1 s で支配)。**段で 2 ラウンド 2% 以上の規則**で採った。
7. 切り戻しは `DENNO_PF_KV_RESERVE=0` (1 行)。`DENNO_PF_KV_HPAGE` は既定 off。

## 1. 前提: 現在の int-auto の段プロファイルと、選んだ段 (Do 1 / Do 2)

### 1.1 14k cold (t561b b_1、14,018 トークン、wall 5,359 ms / gpu_busy 4,150)

| 段 | wall ms | gpu busy ms | exposed ms | 中身 |
| --- | ---: | ---: | ---: | --- |
| ssm | 1,944.7 | 1,318.9 | 625.7 | GEMM + scan (t392/397/402/407/441/447) |
| experts | 1,727.0 | 1,421.9 | 305.1 | `up_wait` / 28 GB の上げ (t557 で閉じた) |
| other | 942.1 | 201.3 | 740.8 | `iter_top` 326.7 / `pf_er` 241.8 / `iter_ple` 229.1 / `pf_teardown` 88.8 |
| full | 709.7 | 553.3 | 156.4 | **attention 約 450 + qkv/out 約 100** |
| mix | 36.2 | 657.8 | 0.0 | 完全に隠れている |

### 1.2 なぜ `full` を取ったか

タスクの指示どおり (`full` の残りを消す)。台帳では `full` の CPU が
`kv_wait 87.5 + kv_push 254.8` で、689 MB を **2.7 GB/s** でしか書けていなかった
(同じホストの単発 memcpy は 15 GB/s、ローカルの再現ベンチで 46 ms)。
`push_many` が 14 チャンク x 12 層で毎回伸ばし直しているのが疑わしかった。

## 2. やったことと結果

### 2.1 `DENNO_PF_KV_RESERVE` (t561a) — 層の初回に容量まで確保する

`push_many_opts` の `cur == 0` (層に初めて積む) で、`k` / `v` の `Vec` を
`min(max_ctx, 32,768) * row` 要素まで `reserve` する。**アドレス空間だけ**で
ページは書いた分しか触らない (「max_ctx まで確保しない」元の方針は変えない)。

| COND | 3,826 `full` 壁 (2 周) | 3,826 `kv_push` | 14k `full` 壁 | 14k `kv_push` |
| --- | ---: | ---: | ---: | ---: |
| b (base) | 53.2 / 50.6 | 15.0 / 13.5 (realloc 48) | 708.7 | 224.4 (realloc 96) |
| **r** (RESERVE) | **46.1 / 44.6** | **9.3 / 9.6** (0) | 715.6 | **108.9** (0) |
| **rh** (RESERVE+HPAGE) | **45.0 / 44.8** | **10.0 / 9.3** (0) | 710.5 | **103.9** (0) |
| b2 (base) | 52.8 / 50.6 | 15.7 / 13.5 (realloc 48) | 709.0 | 214.9 (realloc 96) |

### 2.2 既定 on にした後の確認 (t561b、同じ run で b と off を並べる)

| COND | 3,826 `full` 壁 (2 周) | `kv_push` | 14k `full` 壁 | `kv_push` | `exposed` |
| --- | ---: | ---: | ---: | ---: | ---: |
| **b** (既定 = RESERVE on) | **44.5 / 45.1** | 9.2 / 9.2 | 709.7 | 101.7 | 156.4 |
| **off** (`RESERVE=0`) | 51.3 / 53.0 | 14.4 / 13.2 (realloc 48) | 750.0 | 254.8 | 196.5 |
| **b2** (既定) | **45.0 / 44.7** | 9.3 / 9.5 | 708.4 | 98.1 | 154.8 |

3,826 の `full` 段: 51.3 / 53.0 -> 44.5 / 45.1 と 45.0 / 44.7 = **-14.0%**。
14k の `full` 段: 750.0 -> 709.7 / 708.4 = **-5.5%**。
同じ run の `ssm` / `experts` / `mix` は不変 (例: ssm 1,945.0 -> 1,944.7 / 1,948.8)。
14k の `other` は 591.8 / 942.1 / 572.7 (off / b / b2) で、これは COND の位置の
ノイズ (変更は `full` の中で完結している)。decode は 10.4 ms/tok で全 COND 同じ、
14k の後の decode は 12.2 / 12.1 / 12.0。

### 2.3 計器: realloc の回数を台帳に出す

`[q4ledger-pf] sub full` の行に `kv_realloc=<回> (<MB>)` を足した
(容量が足りずに伸ばした回数だけ。`reserve` が効いていれば 0 になる)。
t561a/b のログで 48 回 (188.1 MB) / 96 回 (402.5 MB) -> 0 を確認している。

### 2.4 `DENNO_PF_KV_HPAGE` は採らない

`madvise(MADV_HUGEPAGE)` を確保した領域に 1 回かける案。3,826 で 45.0 / 44.8、
14k の `kv_push` で 103.9 と、reserve 単体 (46.1 / 44.6、108.9) との差が noise の
内側なので**既定 off のままコードだけ残す** (このホストの THP は `madvise`)。

## 3. 数値の扱い

- 各 run の**同じ run の中の対**で読んだ (t561a は b/b2、t561b は b/off/b2)。
- **14k の壁は採否に使っていない**: 同じコードの b と b2 が 5.37 / 4.97 s (8%)、
  `other` 段が 942 / 573 ms と 370 ms 動く。段の内訳 (`full` / `kv_push`) は
  同じ run で 709.7 / 708.4 と 101.7 / 98.1 で安定している。
- 3,826 の壁も 1.88-1.90 s で全 COND 同じ (experts 段 1.1 s が支配)。
- md5 は両 run の全 COND x 全 6 文脈で **SAME** (t553 と同じ
  `a:edaa84ad ... c8000_cold:672fec07`)。数値経路を触っていない。
- 見積りは「推定」と書いた。それ以外は実測。

## 4. 出力テキスト (健全性)

t561b b_1 の `c8000_cold` (14,018 トークン、md5 `672fec07`):

> prefillの遅延は計算量ではなく、ホスト往復とメモリコピーが律速であった。
> 計測の誤認や静かな退行を排除しつつ、デバイス常駐化と転送経路の最適化を段階的に実施した。
> その結果、1.5Bモデルで395.8秒から12.2秒へ、9Bモデルで46.2秒から10.11秒へと大幅な高速化を達成した。

全 COND で `!!!!` / 空出力 / プロンプトの反復は無し、errors 0。

## 5. VERDICT 行

```text
t561a: VERDICT b  vs x0 OK decode -1.8% prefill -0.9% md5 SAME errors 0
       VERDICT r  vs x0 OK decode -1.8% prefill -1.8% md5 SAME errors 0
       VERDICT rh vs x0 OK decode -2.2% prefill +0.0% md5 SAME errors 0
       VERDICT_DEPTH r  c8000_cold vs x0 decode -4.5% prefill -32.9%
       VERDICT_DEPTH rh c8000_cold vs x0 decode -6.0% prefill -35.3%
t561b: VERDICT b   vs x0 OK decode -0.4% prefill +0.9% md5 SAME errors 0
       VERDICT off vs x0 OK decode +0.0% prefill +0.0% md5 SAME errors 0
       VERDICT_DEPTH b   c8000_cold vs x0 decode -2.3% prefill -12.1%
       VERDICT_DEPTH off c8000_cold vs x0 decode -3.1% prefill -17.5%
```

(`vs x0` は捨て COND との比較なので使っていない。採否は §2 の同じ run の対で読んだ。)

## 6. テスト

```text
CARGO_TARGET_DIR=/tmp/t561 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1392 passed; 0 failed; 103 ignored      (int-auto dd9183fb4 を merge した後)
同 --test kernel_invariants
  -> 150 passed; 0 failed
CARGO_TARGET_DIR=/tmp/t561nc cargo build --workspace --release -> OK
同 cargo test --release -p denno-engine --lib (cuda feature 無し) -> 999 passed; 0 failed
同 cargo test --workspace --release -- --test-threads=1 (cuda feature 無し) -> 全緑 (rc=0)
```

足したテスト (3 本):
- `task561_kv_reserve_and_hpage_gates_truth_table`: reserve は未設定/空で on、
  明示的な偽値だけ off。hpage は既定 off。
- `task561_reserve_takes_capacity_in_one_go`: reserve 後は容量が `max_ctx*row` に
  なり、続くチャンクで増えない (realloc が起きない)。
- `task561_realloc_counter_sees_the_growth_branch`: 計器が reserve 無しで立ち、
  reserve ありで 0 になる (分岐が発火した証拠をテストでも押さえる)。
- 既存の `task505_push_many_noresize_and_par_match_resize` に reserve 経路を足し、
  clear 後も含めて**バイト列が従来と同一**であることを固定した。

## 7. merge

終端で int-auto を fetch し直し、**`7510bc00a` -> `dd9183fb4`** (merge: br538) を
merge した (衝突なし。`build_ptx.sh` の targets_all も衝突なし)。merge 後に
上のテストを回し直して緑。

## 8. 次に触る人へ (実測に基づく残り)

1. **`full` の残りは attention カーネル (約 450 ms) だけ**。14k の `full` は
   `gpu_busy 553.3` 対 CPU 288.6 でもう GPU 律速で、ホスト側をいくら削っても
   壁は動かない (今回それを実測で示した)。
2. そのカーネルは **tensor peak の 2 割弱**しか出ていない
   (attention の mma は 14k 全体で約 16 TFLOP、450 ms で 35 TFLOPS。bf16 dense
   209 TFLOPS の 17% = **推定**)。1 ブロック 89.6 KB の smem で
   占有が上がらず、key タイル 1 枚ごとに `__syncthreads` が 5 回入る。
   BQ を 96/128 に広げる案は smem の上限 (P465 が実測した 99 KB) で入らない
   (BQ=96/BK=32 で 113 KB 必要)。
   次は (a) mask+scale+exp を 1 パスに融合して sync を 1 回減らす、
   (b) key 方向の split (flash-decoding 型) で重いタイルを割る、の順が安い。
   どちらも `.cu` の変更所以外 (host 側は `pf_full_out` の引数だけ)。
3. `DENNO_PF_KV_HPAGE` は既定 off で残した。**メモリが逼迫する条件**
   (production で複数セッション) では効く可能性があるが、この lab では差が無い。
4. 14k の壁を A/B で読むには `other` 段 (`iter_ple` / `iter_top`) の位置依存を
   先に潰す必要がある (同じコードで 370 ms 動く)。段の内訳で読むのが現状は正しい。
