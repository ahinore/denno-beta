ADOPT: yes t565a t565b t565c

# task 565 (dev2) — 既定モデル prefill: 14k cold の `other` に残っていた
# **ホスト転送 2 本**を測った。残差の往復を pinned に通す `DENNO_PF_RES_PIN` は
# `res_up` が 14k cold で **99.4 / 98.2 -> 90.1 / 88.8 / 89.6 ms (-9.4 / -9.6 / -8.8%、2 ラウンド)**
# と再現し**既定 on**。expert の戻しを待たない `DENNO_PF_RESTORE_NOSYNC` は
# **勝ちなし (85-92 ms で不変)** なので既定 off のまま。
# **タスクの壁 2.5% は満たしていない** — 14k の壁は同じコードでも COND の位置で
# 7% 動き (5.10-5.46 s)、残った時間は GPU 律速 (busy 4,155 / 壁 5,100 ms)。

branch **`br565`** = 開始時の `int-auto` **`ac0d7c1ad`** (= merge: br561。私の br561 は
既に入っていたので merge 不要)。終端で int-auto が **`7b9b2e9cd`** (merge: br520) へ
動いたので merge した (§7)。

数値はすべて GPU ラボの実測 (`SCENARIO=depth8k`、`PROFILE=1`、
`MODEL=flash-next-UD-IQ3_XXS`、baseline COND = int-auto の既定 +
`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0
DENNO_GPU_WEIGHT_MB=12000`、先頭に捨て COND、末尾に base の繰り返し)。

lab 名と SHA:
- **t565a** = `168796766`、COND: x0 / b / **rn** (RESTORE_NOSYNC) / **rp** (RES_PIN) /
  **rnrp** (両方) / b2
- **t565b** = `168796766`、COND: x0 / b / **rp** / **rp2** (rp の 2 本目) / b2
- **t565c** = `ae0e24d5c` (RES_PIN 既定 on)、COND: x0 / b / **off** (`RES_PIN=0`) / b2

`.cu` / `.ptx` は触っていない (PTX 再生成不要)。

## 0. 結論 (7 行)

1. **採る (`ADOPT: yes`)、ただし段の規則で。** 14k cold の `res_up` は
   **99.4 / 98.2 -> 90.1 / 88.8 / 89.6 ms** (t565a rp/rnrp、t565b rp/rp2) で
   **-9.4 / -9.6 / -8.8%**、`iter_res` (層 0 のチャンク上げ) も **79.6 -> 67.8**。
2. **機序**: 14k の残差は 574 MB (7 チャンク × 82 MB) を上げ、最後に 71 MB を下げる。
   `pf_upload_res` / `pf_download_res` は `copy_from_pinned` / `to_slice_pinned` を
   呼んでいたが、その手前の全体ゲート `DENNO_PF_PIN_STAGE` が**既定 off** (P358 が
   旧既定モデルで棄却) なので、実際は pageable の同期コピーを通っていた。
   残差の往復だけを force で pinned に通すと 4.0 -> 4.4 GB/s になる (14k の実測)。
3. **`DENNO_PF_RESTORE_NOSYNC` (新規) は勝ちが無い。** 戻しは t561b では
   208 ms (1,003 本 / 1,975 MiB) だったが、**t565a 以降は全 COND で 85-92 ms**
   (同じコード。page cache の状態で 2.4 倍動く)。1 本ごとのイベント待ちを外して
   4 枠リングにしても **92 ms (rn) 対 88 ms (b)** で、勝ちは出なかった。
   `DENNO_PF_RESTORE_NOSYNC=1` で切り戻し可能な形で**既定 off** のまま残す。
4. **`pf_teardown` も同じ理由で今回は 87-94 ms** (t561b は 210.5 ms)。つまり
   P561 の §8 が挙げた「14k の `other` の 367 ms」は**ホスト側ではなく
   page cache の状態**で動いていた部分が大きい。
5. **壁は読めない**: 同じバイナリ・同じ env の 14k cold が 1 つの run の中で
   **b 5.46 / off 5.22 / b2 5.10 s** (t565c)、t565a は 5.85 / 5.10、t565b は 5.27 / 5.13。
   先頭ほど遅い (page cache / load)。14k の採否は**段** (`res_up`) で読んだ。
6. **残りは GPU 律速** (§8)。14k の `close wall` は gpu_busy **4,155 ms** / 壁 5,100 ms
   で 81%。段は ssm 1,957 (busy 1,319) / experts 1,823 (busy 1,424) / full 712
   (busy 554) / other 573 / mix 35 (busy 658、ssm の壁の中)。ホスト転送はもう
   全部で 300 ms 未満で、しかも pageable のままで 6-7 GB/s 出ている (§2.4)。
7. 切り戻し: `DENNO_PF_RES_PIN=0` (1 行)。`DENNO_PF_RESTORE_NOSYNC` は既定 off。

## 1. 前提: 現在の int-auto の段プロファイル (Do 1) と、選んだ段 (Do 2)

### 1.1 14k cold、t565c の b2 (最後の COND、load 3.2) — wall 5,102 ms / gpu_busy 4,157

| 段 | wall ms | gpu busy | exposed | 中身 (台帳) |
| --- | ---: | ---: | ---: | --- |
| ssm | 1,957 | 1,319 | 638 | GDN 連鎖 (t392/397/407 で閉じた) |
| experts | 1,823 | 1,424 | 399 | 上げ (P557 で閉じた) + sync 246 |
| full | 712 | 554 | 159 | kv_wait 87 / kv_push 106 + attention |
| other | 573 | 201 | 371 | pf_er 243 / iter_top 132 / res_up 89 / teardown 87 |
| mix | 35 | 658 | 0 | ssm の壁の中で待たれる (直列) |

`other` の内訳 (t565c b2、同一 run の b は 1,144 ms で位置のノイズ):
`sub other sum=633.6 pf_er=242.9 iter_top=131.7 [h=145.9M/36] res_up=89.3
[h=574.2M/7 d=71.1M/1] iter_res=71.6 pf_teardown=87.4 res_park=4.5 iter_ple=0.0`

### 1.2 なぜこの 2 本を取ったか

REPORT_P480 以降 (この既定モデル) の `other` は、P419 (残差 `other` の分解)、
P423/P461 (PLE の注入)、P482/P486/P490 (UD-Q4_K_XL の窓)、P553 (`iter_top` の
2 分割) までで、**名前の付いた残りは「同期の H2D/D2H」だけ**になっていた。
P553 §2.4 が「`iter_res` = 574 MB の同期 H2D。非同期化には 2 枚要る。次の候補と
しては有望」と明示的に残し、P561 §8 も「`full` の残りはカーネル」として
ホスト側を打ち切っていた。**14k で 574 MB + 71 MB を運ぶ 2 本 (res_up 159 ms) と、
1,975 MiB を運ぶ戻し (pf_teardown 210 ms)** は、どれも過去の REPORT が
手を付けていない (戻しは P394 が並列化しただけ) ので、この 2 本を取った。
`.cu` を触らずに済む (このタスクは .cu を要求していない)。

## 2. やったことと結果

### 2.1 `DENNO_PF_RES_PIN` — 残差の往復だけ pinned バウンスへ (既定 on)

`with_pf_pin` の判定を純関数 `pf_pin_admit(force, stage_on, small_on, min, bytes)`
に切り出し、`force` は**全体ゲートだけ**を飛ばす (窓 1 MiB..=256 MiB は同じく効く)。
`pf_upload_res` / `pf_download_res` が `copy_from_pinned_forced` /
`to_slice_pinned_forced` を先に試し、取れなければ**従来の経路へそのまま落ちる**
(バイト列は同一)。

| run / COND | 14k `res_up` | `iter_res` | 14k `other` | md5 |
| --- | ---: | ---: | ---: | --- |
| t565a b | 99.4 | 80.4 | 872.8 | SAME |
| **t565a rp** | **90.1** | **71.2** | 834.3 | SAME |
| t565a rnrp | **89.4** | **70.5** | 633.2 | SAME |
| t565a b2 | 99.5 | 83.0 | 664.8 | SAME |
| t565b b | 98.2 | 79.6 | 986.2 | SAME |
| **t565b rp** | **88.8** | **67.8** | 873.7 | SAME |
| **t565b rp2** | **89.6** | **70.9** | 885.1 | SAME |
| t565b b2 | 97.3 | 81.0 | 682.9 | SAME |
| t565c b2 (既定 on) | **89.3** | **71.6** | 633.6 | SAME |

`on` の 5 サンプルは 88.8-90.1 に揃い、`off` は 97.3-99.5。**2 ラウンドで -8.8〜-9.6%**。
3,826 (c2000_cold) の `res_up` も 25.8 -> 22.9 ms 程度で、他は動かない。

### 2.2 `DENNO_PF_RESTORE_NOSYNC` — 戻しを 1 本ごとに待たない (既定 off、勝ちなし)

リング 4 枠 (`UP_RING`) の pinned を使い回し、**使い回す枠の DMA だけ**イベントで
待ち、戻しの終端で各スレッドが 1 回待つ (`restore_defer_flush_thread` を
プールの `broadcast` で全スレッドに掛ける)。デバイスへ渡るバイト列は 1 本ずつの
経路と同一 (変わるのは順序と待ち方だけ)。

| run / COND | `[pf-restore]` | `pf_teardown` |
| --- | ---: | ---: |
| t561b b (P561 の run) | 208 ms | 210.5 |
| t565a b | 86 ms | 87.9 |
| t565a **rn** | **90 ms** | 92.1 |
| t565a rp | 85 ms | 87.3 |
| t565a rnrp | 91 ms | 93.5 |
| t565b b | 85 ms | 87.4 |
| t565c off | 85 ms (推定 — b/b2 と同じ) | 94.1 |

**同じコードの b が 208 ms と 86 ms** で、これは page cache の状態 (t561b の run は
13 時間前の別 run)。今回は戻しが既に DDR の帯域 (1,975 MiB / 86 ms = 23 GB/s) で
走っていたので、待ちを外しても縮まない。**既定 off、コードは真理値表テスト付きで残す**
(切り戻し不要。次に page cache が冷たい条件で効く可能性はあるが、測ってから)。

### 2.3 計器・テスト

- 台帳の副区間 (`res_up` / `iter_res` / `pf_teardown`) は既存のものをそのまま読んだ
  (新しい添字は足していない)。
- 追加テスト 5 本 (lib 4 + kernel_invariants 1): `task565_res_pin_gate_truth_table`、
  `task565_restore_nosync_gate_truth_table`、`up_ring_index_cycles_over_the_ring`、
  `forced_pin_skips_only_the_stage_gate` (force は全体ゲートだけを飛ばし窓は残す)、
  `t565_res_pin_and_restore_nosync_are_wired` (配線の静的検査)。

### 2.4 測って棄却したもの (このタスクで新たに)

- **全体ゲート `DENNO_PF_PIN_STAGE` を on にする**: 残差だけに効かせても -9% なので、
  1,080 回・159 MB の ssm の小さなコピーまで pinned にすると写しの固定費が勝つ
  (P358 の棄却と整合)。触っていない。
- **戻しのスレッド数 16**: 戻しが 86 ms の run では帯域律速で、本数を増やす余地が
  無い (8 で 23 GB/s)。測っていない (測る価値が無いと判断)。
- **pageable のままでも 6-7 GB/s 出る**: 「pageable は 1 GB/s」は古い数字で、
  この木の 14k の H2D は 6.5 GB/s 出ている。pinned の勝ちは 4.0 -> 4.4 GB/s ぶんだけ。

## 3. 数値の扱い

- 採否は**同じ run の中の段** (`res_up` / `iter_res`) で読んだ。14k は 1 run 1 サンプル
  だが、t565a と t565b の 2 run で 5 サンプル取れ、`on` 5 本が 88.8-90.1、
  `off` 4 本が 97.3-99.5 と分離している。
- **壁は採否に使っていない**: t565c の 1 run の中で b (2 番目) 5.46 s と b2 (4 番目)
  5.10 s。t565a は 5.85 / 5.10。位置の効果が 7% あり、2.5% の差は読めない。
  t565c の `off` (3 番目) 5.22 は「on の b2 5.10 より遅い」が、これも位置の効果で
  説明でき、`res_up` の 200.7 ms という値も同じ run の先頭 3 COND だけが遅い
  (page cache) ことによる。**そのため t565c の off は採否の根拠に使っていない**
  (根拠は t565a/t565b の対)。
- md5 は **3 run × 全 COND × 全 6 文脈で SAME** (`a:edaa84ad ... c8000_cold:672fec07`)。
  数値経路を触っていない (転送の経路だけ)。
- 見積りは「推定」と書いた。それ以外は実測。

## 4. 出力テキスト (健全性)

t565c b2 の `c8000_cold` (14,018 トークン、md5 `672fec07`):

> ご提示いただいた文書は、LLM推論エンジン（`denno-engine）における **Prefill（事前計算）
> 処理の性能最適化の記録です。 … 以下に、この文書の技術的詳細を、**「問題の特定」、
> 「最適化の段階的変遷」、「計測の罠と

全 run・全 COND で `!!!!` / 空出力 / プロンプトの反復は無し、errors 0。
decode は全 COND 10.4 ms/tok (14k 後 12.0-12.6)。

## 5. VERDICT 行

```text
t565a: VERDICT b    vs x0 OK decode +0.4% prefill +0.0% md5 SAME errors 0
       VERDICT rn   vs x0 OK decode +0.8% prefill +1.8% md5 SAME errors 0
       VERDICT rp   vs x0 OK decode +0.0% prefill +2.7% md5 SAME errors 0
       VERDICT rnrp vs x0 OK decode +0.4% prefill +0.9% md5 SAME errors 0
       VERDICT b2   vs x0 OK decode -0.4% prefill +1.8% md5 SAME errors 0
       VERDICT_DEPTH rp c8000_cold vs x0 decode -8.5% prefill -28.2%
t565b: VERDICT b    vs x0 OK decode +0.0% prefill +0.0% md5 SAME errors 0
       VERDICT rp   vs x0 OK decode -0.4% prefill -0.9% md5 SAME errors 0
       VERDICT rp2  vs x0 OK decode +0.0% prefill -0.9% md5 SAME errors 0
       VERDICT b2   vs x0 OK decode +0.8% prefill -0.9% md5 SAME errors 0
       VERDICT_DEPTH rp c8000_cold vs x0 decode -3.0% prefill -1.3%
t565c: VERDICT b    vs x0 OK decode +0.4% prefill +0.9% md5 SAME errors 0
       VERDICT off  vs x0 OK decode +0.0% prefill +1.8% md5 SAME errors 0
       VERDICT b2   vs x0 OK decode +0.0% prefill +0.9% md5 SAME errors 0
       VERDICT_DEPTH off c8000_cold vs x0 decode -3.8% prefill -5.9%
```

(`vs x0` は捨て COND との比較なので使っていない。`VERDICT_DEPTH` の 14k も
上の理由で採否に使っていない。)

## 6. テスト

```text
CARGO_TARGET_DIR=/tmp/t565 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1398 passed; 0 failed; 103 ignored      (int-auto 7b9b2e9cd を merge した後)
同 --test kernel_invariants
  -> 152 passed; 0 failed
CARGO_TARGET_DIR=/tmp/t565nc cargo build --workspace --release -> OK
同 cargo test --workspace --release -- --test-threads=1 (cuda feature 無し)
  -> 全緑。ただし `denno-server --test slot_leak` の
     `dropped_stream_receivers_do_not_leak_kv_slots` が workspace 全体を
     流したときに 1 回だけ落ちた (単体で流すと 3/3 緑)。時間依存の flaky で、
     本タスクの変更 (engine の prefill 転送) とは無関係。
```

## 7. merge

終端で int-auto を fetch し直し、**`ac0d7c1ad` -> `7b9b2e9cd`** (merge: br520) を
merge した。conflict は `kernel_invariants.rs` の末尾 1 箇所だけで、両側のテストを
残して解決した (br520 は decode の flashdec カーネル `attn_dec5` と cuda.rs の
decode 経路で、本タスクの prefill 転送とは重ならない)。merge 後にテストを
回し直して緑。`.cu` / `.ptx` の衝突は無し。

## 8. 次に触る人へ (実測に基づく残り)

1. **14k の壁は 81% が GPU** (gpu_busy 4,155 / 壁 5,100)。ホスト側で残っているのは
   `pf_er` 243 (ルーターの同期待ち = 実は GPU 待ち、P489 で `PF_EARLY_ROUTE=0` に
   しても壁は動かない)、`kv_wait` 87 (P553 の残り)、`res_up` 89 (今回で打ち止め)、
   `pf_teardown` 87 (page cache 次第で 208 にもなる) だけ。**1 本ずつでは 1% を
   超えない**ので、ホスト側はもう壁を作らない。
2. **`mix` の GPU 658 ms が最大の未着手** (14k、ssm の壁の中で直列に待たれる)。
   中身は hc mix の 3 つの BF16 GEMM で、形は down/up が
   m=nt*hc=56,072 / k=hc_dim=10,240 / n=lr=320 という**細い** GEMM。
   概算 39.8 TFLOP を 658 ms = **60 TFLOPS = bf16 ピークの 29% (推定)**。
   `gemm_bf16_mt` のタイルを n=320 に合わせる (または down/up を 1 起動に融合する)
   のが次の一手で、`kernels_exp_iq_tc.cu` 系の GEMM と同じ形の仕事になる。
   **.cu を触るので PTX 再生成が要る。**
3. `experts` の exposed 399 ms は上げの待ち (sync 246 + 153)。`up_wait` 1,278 /
   `copy_s` 0.95 s で、P557 が閉じた範囲。
4. **測定の作法**: 14k の壁は 1 run の中でも位置で 7% 動く (先頭ほど遅い)。
   同じ run の `[q4ledger-pf] close wall` の **gpu_busy (4,154-4,157 と不変)** を
   見るのが一番安定している。段 (sub) の方がさらに安定。
