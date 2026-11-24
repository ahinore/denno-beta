ADOPT: yes t788c (A/B の before は t788b)

# REPORT P788 — prefill の CUDA context 未初期化 (バグ 1) と、プール区画ビューの持ち越し (バグ 2)

- ブランチ: `br788` (int-auto 20132ae00 から分岐)
- 対象: コードは `crates/` のみ。加えて lab の再現用に `scripts/lab_gen/788_*.json`
  (プロンプト 8 本、コードではない) を足した。`.cu` / `.ptx` / `docker` / `push` は
  触っていない
- ゲート: 新規 env なし (純粋なバグ修正。切り戻しは commit の revert)
- 数値: カーネル・引数・丸め順・発行順・決定性は不変 (変えたのは「どの作業域を
  再利用してよいか」と「context を用意する順序」だけ)

## 変えたこと (1 行ずつ)

| ファイル | 変更 |
|---|---|
| `cuda/pf_chain.rs` `pf_begin` | `shared_ctx()` + `make_current` を `pf_kernels()` より**前**へ |
| `cuda/pf_chain.rs` `pf_kernels` / `pf_elem_kernels` / `pf2_kernels` / `pf3_kernels` | 初回ロード時に current context が無ければ **None を焼き付けずに** `None` を返す |
| `cuda/gemm.rs` `tc_gemm_kernels` / `cuda/moe_pf.rs` `exp_tc_kernels` | 同上 ([pf-tc] / [pf-ktc] の経路) |
| `cuda/mod.rs` `ensure_current_for_ptx_load` | 上記の共通ヘルパ (新規) |
| `model/qwen4exp/prefill.rs` `forward_prefill_batch_core` | 最初のチャンクが `PF_RESIDENT_MIN_TOK` 以上なら、層を進める前に `make_current_on_this_thread()` |
| `model/qwen4exp/prefill.rs` `Qwen4Exp::pf_init_ctx_first` | 上記の判定 (純関数・テスト可能) |
| `cuda/pf_chain.rs` `pf_pool_release` | 解放前に `pf2_forget_pool_views()` と `pf_stream_forget_pool_views()` |
| `cuda/pf_chain.rs` / `cuda/moe_dec.rs` / `cuda/ctx.rs` | 作業域の再利用判定に `moe_scratch_alloc_bytes_borrowed` (借用ビューは容量が足りても確保し直す) を入れる |
| `cuda/ctx.rs` `DevBuffer::is_view` | 非所有ビュー (プール区画・重みの行範囲) の判定 (新規) |

## バグ 1 の原因 (file:line)

`crates/denno-engine/src/cuda/pf_chain.rs:1877` 付近の `pf_begin` は

1. `pf_kernels()` で PTX を `cuModuleLoadData` する
2. そのあとで `shared_ctx()` / `make_current()` を呼ぶ

順序だった。`cuModuleLoadData` は **current context を要求する** ので、サーバ起動後
まだ `CudaCtx` が作られていないスレッドで最初の prefill が来ると
`CUDA_ERROR_INVALID_CONTEXT` (201) で失敗し、`pf_kernels` の
`OnceLock<Option<PfKernels>>` に **None が焼き付く** (以後そのプロセスでは二度と
ロードしない)。同じ穴が `pf_elem_kernels` / `pf2_kernels` / `pf3_kernels` /
`tc_gemm_kernels` / `exp_tc_kernels` にもあった (「初回だけ。失敗は None を焼き付ける」)。

ログの並び (`[pf-tc] cuModuleLoadData 失敗 → 従来の GEMV` →
`[pf] resident を見送り` → そのあとで `CUDA int8 offload enabled`) はこれで説明できる。
ctx の生成は decode 側の遅延初期化 (`shared_ctx`) で起きるため、失敗した prefill の
**あと**に出すぎる。

修正: 常駐経路に載る大きさの最初のチャンクで、PTX を読む前に ctx を作って current に
する。加えて、ctx が用意できない場合 (VRAM 不足など) に失敗を焼き付けないようにした
(次のリクエストで再試行できる = 恒久フォールバックをやめる)。

## バグ 2 の原因 (file:line)

`crates/denno-engine/src/cuda/pf_chain.rs:2219` の `pf_pool_release` は
`PF_DEV` (`pf_release`)、`PF_SCRATCH` (`pf_scratch_drop`)、`PF_LM` (`pf_lm_release`)
を落としてから `cuMemFree` するが、**`PF2_DEV` と `PF_STREAM` を落としていなかった**。

- `PF2_DEV.x / .ga / .gu / .sh` は `pf_shared_impl` (`pf_chain.rs:5466` 付近) の
  `ensure_region` が `R::ShX / ShGa / ShGu / ShSh` の**ビュー**を入れる。
- `PF_STREAM.slot[s].buf` は `pf_stage_alloc` (`cuda/mod.rs:3891`) が `R::Stage` の
  ビューを入れることがある。

プールを借りるのは `nt_all >= DENNO_PF_POOL_MIN_TOK` (既定 2048) のときだけ。
観測された並びは 512 → 2048 → 8192 (ここでプールを借り、終端で `cuMemFree`) →
flush → **plen=65** (65 < 2048 なのでプールを借りない)。このとき
`ensure_region` は `pool_view` が None を返すので `MoeScratch::ensure` に落ちるが、
そこは「容量が足りていれば何もしない」なので、**解放済みの VRAM を指す
`DevBuffer` がそのまま再利用される**。最初の gate GEMM がそこへ書き、非同期の
不正アクセスが次の同期 (`pf_chain_sync` → `cuMemcpyDtoH`) で error 700 として出る。
これが `[pf2] shared: CUDA cuMemcpyDtoH failed: error 700` の正体で、
カーネル (normp / mixed_fs / pf2_gemm_shared) の grid/境界そのものは正常
(nt=65 でも grid は行数・要素数から正しく出ている。テストで寸法を固定した)。

修正 (二段):

1. `pf_pool_release` が解放前に、プール区画を指しているビューだけを落とす
   (`pf2_forget_pool_views` / `pf_stream_forget_pool_views`)。所有バッファ
   (qg / q / o / cs など) は残すので VRAM の使い回しは維持。
2. 再利用判定そのものを安全側にした。`moe_scratch_alloc_bytes_borrowed(cur, want,
   borrowed)` は借用ビューなら必ず `Some(want)` を返す (= 確保し直す)。これを
   `MoeScratch::ensure_tagged` / `Scratch::ensure_tagged` / `pf2_ensure` /
   `pf_slot_prepare` に入れた。rayon ワーカーのように「解放を実行したスレッドとは
   別のスレッド」に残ったビューも、次回の ensure で捨てられる。

## 再現 (before / after) — lab で実測

GPU の無いこのコンテナでは `cuModuleLoadData` / error 700 を踏めないので、
lab の `SCENARIO=gen` で同一サーバに 8 本を順に送って before / after を取った
(`scripts/lab_gen/788_*.json`。max_tokens=16 / temperature=0 で決定論的)。

並び: 491 tok (初回 = バグ 1 の条件) → 1871 → 7161 (ここでプールを借りる) →
28 / 28 / 39 / 56 / 80 tok (プールを借りない短い prefill = バグ 2 の条件)。

| cond | SHA | errors | 08 番目 (80 tok) の応答 |
|---|---|---|---|
| base | 28c0f8763 (20132ae + lab_gen のみ) | **6** | `{"error":{"message":"prefill 常駐 (DENNO_PF_RESIDENT): ffn の combine に失敗"}}` |
| fix | e399ff3d0 | **0** | 正常な本文 16 tok (md5 e3fe5479) |

base のエラー行 (t788b の RESULT.md より):

```
[pf2] shared: CUDA cuMemcpyDtoH が失敗: error 700 → 従来経路
gguf gemm failed (uid=1580): CUDA cuMemcpyHtoD が失敗: error 700
gguf gemm failed (uid=1581): CUDA cuMemcpyHtoD が失敗: error 700
[pf] bo H2D: CUDA cuMemcpyHtoD が失敗: error 700 → 以降の prefill は CPU 経路
```

= タスク本文のログと同じ形 (共有 expert の D2H で 700 → 以後の H2D も全部 700 →
恒久 CPU 化)。fix は同じ並びで 0 行。

md5 (base → fix、greedy 16 tok):

| 788_01 | 788_02 | 788_03 | 788_04 | 788_05 | 788_06 | 788_07 |
|---|---|---|---|---|---|---|
| bdbd5ada 同一 | 788d3a53 同一 | 788d3a53 同一 | 63b73f15 同一 | 603a0d79 同一 | 8322f297 同一 | 33d669cb 同一 |

(base は 8 本目で要求エラーになったので 8 本目の比較は無い。fix は 8/8 が本文を返し、
base が返した 7 本はすべて md5 一致 = この修正は数値を 1 ビットも変えない。)

バグ 1 について: base の 8 本目までは `cuModuleLoadData 失敗` が 1 行も出なかった。
この環境では最初のリクエスト (491 tok) の前に context が既に current になっており
(モデル読み込み側の経路)、タスク本文の観測条件を再現できなかった。したがって
バグ 1 の修正は「起きないときは 1 バイトも挙動を変えない防御」(ctx が既にあれば
`make_current` が 1 回増えるだけ) として入っている。実機で再現した人がいれば、
この修正で `cuModuleLoadData 失敗` が消えることを見てほしい。

## テスト (CPU 関門)

- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`
  → 1435 passed / 0 failed / 103 ignored
- `cargo test --release -p denno-engine --features cuda --test kernel_invariants`
  → 156 passed / 0 failed
- `cargo test --release -p denno-server` → 8 passed / 0 failed (+ doc 0)
- `cargo test --release --workspace -- --test-threads=1` (cuda feature 無し) → 全緑
- `cargo build --release --workspace` (cuda feature 無し) → OK
- `cargo build --release -p denno-engine --features cuda` → OK

追加した回帰テスト 3 本 (すべて純関数。GPU 不要):

1. `model::qwen4exp::prefill::p788_tests::pf788_ctx_init_boundary`
   — 冷えた起動直後に「PTX を読む前に ctx を作る」側へ入る plen の境界。
   {64, 65, 67, 127, 128, 129, 506, 512, 1536, 8192} が入り、
   {0, 1, 28, 63} が入らない。境界が常駐経路の判定と同じであること。
2. `model::pf_pool::tests::small_prefill_shared_regions_cover_nt`
   — nt ∈ {64, 65, 67, 127, 128, 129, 512} で shared expert の 4 区画
   (ShX / ShGa / ShGu / ShSh) が `nt * e * 4` / `nt * sff * 4` ちょうどで
   取れ、互いに重ならず `total` に収まること。16/32/64 の倍数でない nt の
   端数行が切り捨てられていないことの固定。
3. `cuda::tests::p59_dedup_tests::moe_scratch_never_reuses_pool_views`
   — 借用ビュー (プール区画) は容量が足りていても確保し直す。
   所有バッファは従来どおり grow-only。nt の各値の寸法で 8k prefill の
   借用ビューが残っていても取り直すこと。

GPU が要る挙動 (実際の cuModuleLoadData / error 700) はテストでは踏めないので、
そこは lab で確認する。

## GPU ラボ (実測、model=flash-next-UD-IQ3_XXS)

| 依頼 | SHA | シナリオ | 結果 |
|---|---|---|---|
| t788a | 3d3e9061a (修正後) | depth8k, PROFILE=1 | errors 0 / text ok / md5 a:edaa84ad b:1938059f c2000_cold:9b28e018 c2000_turn2:2081cdeb c8000_cold:fea937f5 c8000_turn2:3765c8dd。decode 32.1 ms/tok、c8000_cold prefill 188.1 s。`cuModuleLoadData 失敗` 0 行 / `error 700` 0 行。`[pf-pool] 4094 MiB を借りた → 返した` が各ラウンドで出る |
| t788b | 28c0f8763 (20132ae + lab_gen) | gen (8 本) | **errors 6 / 8 本目が要求エラー** (バグ 2 の再現)。1-7 本目の md5 は下記 |
| t788c | e399ff3d0 (修正後 + lab_gen) | gen (8 本) | errors 0 / 8 本すべて正常 / 1-7 本目の md5 が base と一致 |

depth8k の注意: このシナリオの最初のプロンプトは 28 tok、8k の次のターンは plen=19
(< 64) なので、**depth8k はバグ 1 もバグ 2 も踏まない** (t788a が errors 0 なのは
修正の証拠ではなく、シナリオが条件に届いていないため)。再現と修正の証拠は
t788b / t788c の gen 並びの方。t788a は「標準シナリオで退行が無い」ことの確認。

速度: t788a の decode 32.1 ms/tok / c8000_cold 188.1 s は、同モデル・同 env の
直近値と同水準 (この修正は経路もカーネルも増やさないので、差はノイズの見込み =
推定)。base の depth8k は 3 ラウンド上限のため取っていない。

## GPU で確認してほしいこと

1. 修正後 (t788a) の depth8k で、`[pf] resident を見送り: 作業域/PTX/GPU の前提外`
   と `[pf2] shared: ... error 700` が **1 回も出ない**こと。
2. 8k のあとの plen=65 / 64 / 67 / 127 / 129 が GPU のまま (CPU 6.7 tok/s に
   落ちない) こと。
3. t788a と t788b の md5 が全プロンプトで一致すること (この修正は数値を変えない)。
4. 修正後もプールの借用が起きていること (`[pf-pool] expert から作業域 ... 借りた`)
   と、解放ログ (`expert に返した`) が毎回出ること (t788a で確認済み)。
5. バグ 1 の実機ログ (`[pf-tc] cuModuleLoadData 失敗`) が取れている環境があれば、
   修正後にその行が消えることの確認 (この lab では再現しなかった)。
