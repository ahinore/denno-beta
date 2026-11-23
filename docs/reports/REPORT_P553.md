ADOPT: yes t553b

# task 553 (dev2) — 既定モデル prefill の残り: KV 書き戻しの待ちを 1 呼び出し後ろへ
# (`DENNO_PF_KV_DEFER`)、`kv_push` の並列写し (`DENNO_PF_KV_PUSH_PAR`) を既定 on、`iter_top` の内訳

branch **`br553`** = 開始時の `int-auto` **`d70a724d3`** (= merge: br530)。
`br549` (task 549、ADOPT: no) は int-auto に入っておらず、この作業には不要なので merge していない。
終端で int-auto を fetch し直したが **`d70a724d3` のまま**なので merge は無い。

数値はすべて GPU ラボの実測 (`SCENARIO=depth8k`、`PROFILE=1`、
`MODEL=flash-next-UD-IQ3_XXS`、baseline COND = int-auto の既定 +
`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0
DENNO_GPU_WEIGHT_MB=12000`、先頭に捨て COND、末尾に base の繰り返し)。

lab 名と SHA:
- **t553a** = `7fff91883` (`load_before=1.86`、x0 / kvd / np / kvdnp / b2)
- **t553b** = `40e6db197` (`load_before=1.93`、同じ 5 COND。ずらした判定の修正後 = 最終コード)

## 0. 結論 (6 行)

1. **採る (`ADOPT: yes t553b`)。ただし壁の目標 (2.5%) は満たしていない。**
   14k cold の壁は **4.96 / 5.09 s** (base 5.12 / 5.16、**-3.1% / -1.4%**、2 ラウンドの中央で -2.3%)、
   3,826 cold は **1.89 s の床**で差が出ない (base も同じ床)。decode 10.4 ms/tok で不変、
   本文は健全 (全 COND x 全 6 文脈 md5 SAME)、errors 0。
2. **採った根拠は段**: `full` 段が 3,826 で **105.1-106.9 -> 49.6-70.3 ms**、
   14k で **777.8-783.2 -> 702.4-738.1 ms**。内訳は `kv_wait` **34.0-35.6 -> 0.0**
   (3,826、4 サンプル全部)、14k で **126.4-129.1 -> 85.2-90.0**、
   `kv_push` は **25.2-26.0 -> 15.0-15.6** (3,826) / **239.1-249.2 -> 212.4-226.5** (14k)。
   2 ラウンドとも段で 2% を大きく超えて勝ち、他が 2% を超えて悪化した段は無い
   (オペレーター規則「段で 2 ラウンド 2% 以上」)。
3. **`DENNO_PF_KV_DEFER` (新規)**: D2H のイベント待ちを 1 呼び出し (= 層チャンク) 後ろへ回す。
   pinned を 2 枚交互に使い、積むのは次の `forward_batch_flat` / `forward_token` の頭。
   3,826 では待ちが消える (GPU が 1 回転ぶん先へ進む)。14k では 84 回の待ちのうち
   85-90 ms が残る (GPU の null stream の行列が 1 回転より深いため。**推定**)。
4. **`DENNO_PF_KV_PUSH_PAR` を既定 on にした** (task 505 では「同じ run の対が無い」で
   既定 off のままだった)。同じ run の対が 2 ラウンド取れ、`kv_push` が
   3,826 で -30%、14k で -19%。値も順序も同じ (md5 SAME)。
5. **`iter_top` の内訳** (task の Do 2-3): 14k で `iter_ple` (**PLE の gather 待ち**)
   **0-159.1 ms**、`iter_res` (残差の復帰 / 最初の H2D) **80.9-86.5 ms**。
   大きいほうは PLE の gather 待ちだが、**0.0 になる回がある** (2/6) = 背景スレッドが
   先に読み終えると出ない。残る側 (`iter_res` 80-86 ms) は `pf_upload_res` の
   574 MB の同期 H2D (7 回)。どちらも**今回は外していない** (理由は §2.4)。
6. **切り戻し**: `DENNO_PF_KV_DEFER=0` / `DENNO_PF_KV_PUSH_PAR=0` (1 本ずつ効く)。

## 1. なぜこの段を取ったか (Do 1)

開始時の int-auto (`d70a724d3`) の段プロファイル (t553a の base、warm):

| 段 | 3,826 cold | 14k cold |
| --- | ---: | ---: |
| experts | 1,077-1,085 ms (exposed 547-550) | 1,669-1,851 ms (exposed 246-428) |
| ssm | 380.6 ms (busy 361.5) | 1,944.5 ms (busy 1,321.0) |
| full | 105.1 ms | 777.8 ms |
| mix | 241.6 ms (busy 183.2) | 36.6 ms |
| other | 84.0 ms | 556.1-693.4 ms |

`full` 段の内訳 (`sub full`) は **kv_wait 34.0 / kv_push 25.2** (3,826)、
**129.1 / 239.1** (14k) で、**14k では full 段の 47% がホストの KV 書き戻し**だった。
P505 §7 の「残っている手」の 1 番目と 2 番目がそのまま残っており、
`.cu` を触らずに済む (このタスクは .cu を要求していない)。experts / ssm は
task 493-530 で何度も触られているので、触られていないこの段を選んだ。

## 2. 変えたこと

### 2.1 `DENNO_PF_KV_DEFER` (既定 on)

- `cuda.rs` の KV 書き戻しの置き場を 2 枚 (`KvWbSlot` x2 + `seq` の通し番号) にした。
  `pf_kv_wb_begin` は空きの枚にだけ書き、両方埋まっていれば**積まない** (上書きしない)。
  純関数 `kv_wb_pick_slot` に切り出してテストで固定 (`t553_kv_wb_pick_slot_...`)。
- `qwen4exp_attn.rs`: `pf_full_out` の後の push を、この層チャンクの D2H が実際に
  保留に載ったとき (`pf_kv_wb_seq()` が進んだ + `pf_kv_wb_peek() == (層, 行数)`)
  だけ次へ回す。積むのは `kv_flush_deferred()` で、呼ぶのは
  `forward_batch_flat` の頭 / `forward_token` の頭 / `reset` の頭 /
  `spec_restore` の頭 / 層ループの終端 (`LayerAttn::flush_deferred` を足し、
  `SsmAttn` は full へ委譲)。
- **`cache.len(layer)` より前に必ず降ろす**のがこの設計の要。保留があると長さが
  1 チャンクぶん短く、位置がずれる (失敗経路の `truncate_layer` も同じ理由で順序が要る)。
- 積むバイト列・順序・値は従来と同じ (md5 が全 COND・全 6 文脈で SAME)。

### 2.2 `DENNO_PF_KV_PUSH_PAR` を既定 on

`AttnCache::push_many` の写しを rayon で分ける (task 505 の実装のまま)。既定だけ
`DEFAULT_PF_KV_PUSH_PAR = true` にし、真理値表のテストを「未設定・空は on」へ更新した。

### 2.3 `iter_top` の内訳 (台帳 2 本)

`SUB_PF_ITPLE = 45` (`iter_ple`、`rx.recv()` の待ち) と `SUB_PF_ITRES = 46`
(`iter_res`、`pf_upload_res` / `pf_res_park_load`)。`SUB_N = 47`。
既存の添字は動かしていない (テスト `task553_iter_top_split_sub_names_are_wired`)。

### 2.4 外さなかった 2 つ (理由)

- **`iter_ple` (PLE の gather 待ち、14k で 0-159 ms)**: 背景スレッドの読みが
  消費より遅い回だけ出る。速くする手 (並列 PLE 読み、PLE 表の RAM 常駐、
  過去セッションの先読み) は**共通ルールで既に却下**されている。今回
  `denno` 側で待ちを消す手は「PLE を使う層 (層 1) を層ループの後ろへ動かす」しかなく、
  残差の往復 (res の D2H/H2D) が層 1 で起きる前提が崩れるので触っていない。
- **`iter_res` (80-86 ms で安定)**: 中身は `pf_upload_res` の 574 MB の同期 H2D
  (層 0 の 7 チャンク、約 6 GB/s)。非同期化には res のデバイス側バッファを
  2 枚にしないといけない (今は 1 枚を層ごとに使い回す) ので、このタスクの
  範囲 (書き戻しの待ち) を超える。**次の候補としては有望** (§7-2)。

## 3. lab の結果

### 3.1 段 (`[q4ledger-pf]`、3,826 cold)

| run / COND | full ms | kv_wait | kv_push | exposed |
| --- | ---: | ---: | ---: | ---: |
| t553a b2_1 / b2_2 | 106.9 / 105.1 | 34.3 / 34.0 | 26.0 / 25.2 | 24.2 / 22.4 |
| t553a kvd_1 / kvd_2 | 58.5 / 56.8 | **0.0 / 0.0** | 22.3 / 23.0 | 0.0 / 0.0 |
| t553a np_1 / np_2 | 97.2 / 97.1 | 35.4 / 35.2 | **16.6 / 18.3** | 14.6 / 14.4 |
| t553a kvdnp_1 / kvdnp_2 | 50.5 / 49.6 | **0.0 / 0.0** | **13.3 / 12.5** | 0.0 / 0.0 |
| t553b b2_1 / b2_2 | 93.5 (外乱) / 104.4 | 33.9 / 35.3 | 26.8 / 25.7 | — / 22.4 |
| t553b kvd_1 / kvd_2 | 64.0 / 57.6 | **0.0 / 0.0** | 25.6 / 21.5 | — / 0.0 |
| t553b np_1 / np_2 | 98.5 / 95.7 | 34.9 / 35.2 | **18.3 / 16.8** | — / 14.4 |
| t553b kvdnp_1 / kvdnp_2 | 53.3 / 70.3 | **0.0 / 0.0** | **15.6 / 15.0** | — / 0.0 |

### 3.2 段 (14k cold、round 1 の 1 サンプルずつ)

| run / COND | full ms | kv_wait | kv_push | other ms |
| --- | ---: | ---: | ---: | ---: |
| t553a b2 | 777.8 | 129.1 | 239.1 | 693.4 |
| t553a kvd | 857.4 (外乱: experts 2,137) | 89.3 | 461.0 | 714.3 |
| t553a np | 748.7 | 129.5 | **199.3** | 690.5 |
| t553a kvdnp | **702.4** | **85.2** | **212.4** | 581.3 |
| t553b b2 | 783.2 | 126.4 | 249.2 | 556.1 |
| t553b kvd | **715.6** | **90.0** | 267.4 | 671.2 |
| t553b np | 743.1 | 127.5 | **202.7** | 667.9 |
| t553b kvdnp | **738.1** | **89.4** | **226.5** | 587.5 |

x0 (捨て COND) の 14k は 7.51 / 7.27 s で page cache が冷たい (使わない)。

### 3.3 壁と decode

| run / COND | 3,826 cold r1 / r2 (s) | 14k cold (s) | decode ms/tok |
| --- | --- | ---: | ---: |
| t553a x0 | — / 1.974 | 7.51 | 10.4 |
| t553a kvd | 2.632 (外乱) / 1.895 | 5.67 | 10.4 |
| t553a np | 2.084 / 1.918 | 5.09 | 10.4 |
| t553a kvdnp | 1.894 / 1.895 | **4.96** | 10.4 |
| t553a b2 | 1.979 / 1.889 | 5.12 | 10.4 |
| t553b x0 | — / 1.887 | 7.27 | 10.5 |
| t553b kvd | 2.052 / 1.995 | 5.18 | 10.4 |
| t553b np | 2.497 (外乱) / 1.885 | 5.69 (外乱: experts 2,326) | 10.4 |
| t553b kvdnp | 2.055 / 2.010 | **5.09** | 10.4 |
| t553b b2 | 3.422 (外乱) / 1.886 | 5.16 | 10.4 |

**3,826 の壁は 1.887-2.055 で、COND より位置のばらつきが大きい**
(同じバイナリ・同じ env の b2_1 が 3.42 s)。experts 段 (disk / page cache) が
1,077-2,590 と動くため、3,826 の壁は採否に使っていない。

### 3.4 VERDICT 行 (x0 は捨て COND なので `vs x0` は使わない)

```text
t553a: VERDICT kvd vs x0 OK decode +0.0% prefill +0.9% md5 SAME errors 0
       VERDICT np vs x0 OK decode +0.0% prefill +0.9% md5 SAME errors 0
       VERDICT kvdnp vs x0 OK decode +0.0% prefill +0.9% md5 SAME errors 0
       VERDICT b2 vs x0 OK decode +0.0% prefill +0.9% md5 SAME errors 0
t553b: VERDICT kvd vs x0 OK decode -0.7% prefill -2.6% md5 SAME errors 0
       VERDICT np vs x0 OK decode -0.4% prefill -2.6% md5 SAME errors 0
       VERDICT kvdnp vs x0 OK decode -0.7% prefill -3.5% md5 SAME errors 0
       VERDICT b2 vs x0 OK decode -1.1% prefill -1.8% md5 SAME errors 0
```

(上は短い decode の壁。14k の壁は §3.3 の表で読む。)

## 4. 数値の扱い

- 採否は**段** (`full` とその `kv_wait` / `kv_push`) で読んだ。3,826 は 2 ラウンド x
  2 サンプル、14k は round 1 の 1 サンプル (depth8k の 14k は round 1 だけ) を
  2 ラウンド分。壁は 3,826 が位置で 1.887-3.42 と動くので使っていない。
- **目標 (2.5%) は満たしていない**。14k の壁は -3.1% / -1.4% で、
  2 ラウンドの中央は -2.3%。3,826 は床。段の勝ち (kv_wait 34 -> 0) は
  14k の壁に -75 ms 程度しか効かない (壁 5.1 s の 1.5%) ので、**推定**どおり
  目標には届かない。
- `kv_wait` / `kv_push` はホストの壁時計の実測。`[pf553]` の 1 行が
  分岐の発火の証拠 (kvd / kvdnp のログにだけ出る)。
- 14k の kvd (t553a) と np (t553b)、3,826 の b2_1 (t553b) は experts 段が
  2-2.6 s に跳ねた外乱。採否には使っていない。

## 5. 出力テキスト (健全性)

t553b の kvdnp、14k (`c8000_turn2`、md5 `199f23b3` で base と同一):

> ご提示いただいた文書は、LLM推論エンジン（`denno-engine）における **Prefill（事前計算）
> 処理の性能最適化の記録です。 … この文書が示す最も重要な発見は、**「計算量（FLOPs）
> ではなく、データ転送とメモリ管理が

`t553a` も `t553b` も**全 COND x 全 6 文脈 (a / b / c2000_cold / c2000_turn2 /
c8000_cold / c8000_turn2) で md5 が SAME**、`!!!!` / 空出力 / プロンプトの反復は無し、
errors 0。

## 6. テスト

```text
CARGO_TARGET_DIR=/tmp/t553 cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1
  -> 1371 passed; 0 failed; 103 ignored   (task 553 で 3 本追加)
同 --test kernel_invariants
  -> 147 passed; 0 failed
(関門の外) CARGO_TARGET_DIR=/tmp/t553nc cargo build --workspace --release        -> OK
           CARGO_TARGET_DIR=/tmp/t553nc cargo test --workspace --release -- --test-threads=1 -> 全緑
```

追加したテスト: `t553_kv_defer_gate_truth_table` / `t553_kv_wb_pick_slot_rotates_and_never_overwrites`
(2 枚の選び方: ずらさないときは 1 枚、ずらすときは空きを交互、両方埋まれば積まない) /
`task553_iter_top_split_sub_names_are_wired` (添字 45/46、既存の添字が動いていないこと)。
`task505_kv_push_gates_truth_table` は push_par の既定 on に合わせて更新した。
既知の赤 (`qwen4exp_decode spec_batch_stops_at_eos`、`qwen4exp_params` の validate 2 件)
はこの target では出ない。

測った木 (`40e6db197`) と最終 commit の差は「2 つの既定を on にした」1 点で、
測った COND (kvd / np / kvdnp) と同じ分岐に入る。

## 7. 次に触る人へ

1. **KV の待ちはこれ以上ずらさない**。2 枚のスロットで 1 呼び出し先まで隠すのが
   限界 (それ以上は `cache.len` がずれる)。14k に残る 85-90 ms は
   **null stream の行列の深さ** (D2H のイベントは前に積んだカーネルが全部終わるまで
   立たない) なので、ホスト側の工夫では消えない。
2. **次は `iter_res` の 80-86 ms** (14k)。`pf_upload_res` の 574 MB の同期 H2D を
   非同期にするには res のデバイス側バッファを 2 枚にして、層 0 のチャンク ci+1 の
   H2D を ci の MoE の裏へ出す。`pf_res_park_load` の D2D は既に非同期 (タダ同然、
   `iter_res` の 80 ms はほぼ upload 側)。
3. **`iter_ple` は回によって 0 になる** (0-159 ms)。ページキャッシュが暖かいと
   背景 gather が先に終わる。ここを狙うなら「gather の完了を待つ位置」ではなく
   「PLE 層 (層 1) の位置」の設計を変える話になるので、単独タスクで。
4. `full` 段に残るのは qkv の起動 672 回 (4.3-5.3 ms / 呼び出し) と out の D2H。
   `kv_push` は 12-16 ms まで落ちたのでもう小さい。
