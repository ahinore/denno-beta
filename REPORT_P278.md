ADOPT: pending (t278a の結果待ち)

# REPORT P278 — UD-Q4_K_XL decode の層一括 expert 経路から固定費を外す

branch: br278 (int-auto 2a7dbf7 = br271 の merge から)。lab: t278a (1 ラウンド目)。

## 0. 前提 (測り直していない。lab の実測)

task 文と、lab t268b の `done/t268b/base_1.log` の実測 (UD-Q4_K_XL、`decode`、
`PROFILE=1`、base COND = `DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000
DENNO_EXPERT_GPU_MIN_HIT=0 DENNO_GPU_WEIGHT_MB=12000`):

| 項目 | 実測 (128 tok の decode) |
| --- | --- |
| decode 全体 | 64.8 ms/tok (base 65.9) |
| `[q4prof] moe experts` | 40.5 ms/tok (5186.9 ms / 128) |
| `[q4ledger] loop` | launches=143/tok syncs=96/tok (2/層) sync=23.2 ms/tok |
| 同 h2d | 235.73 MB/tok / 554 回 (= ミス 76.7 本 x 3.072 MB) |
| 同 d2h | 7.27 MB/tok / 143 回 (= 153 KB/層) |
| `sync_floor` | 2.4 us |

読み: 1 層 = 0.84 ms。うち **0.48 ms が 2 回の同期の中**、**0.36 ms がホスト仕事**
(ミス 1.6 本/層 x 3 MB の写し + H2D + 表の構築 + silu + 足し込み)。
1 層の小口 H2D は約 7 回 (554 - 77 = 477 回 / 48 層)。旧経路の形
(`moe_gemv_seg_pair` + `moe_gemv_seg_one` = 3 起動 / 2 同期、gate/up の D2H →
ホスト silu → act の H2D → down → D2H → ホスト足し込み) と一致する。

UD の 1 expert = 3,072,000 B (gate/up Q4_K 2560x640、down Q5_1 640x2560)。

## 1. 変えたこと (すべて `DENNO_SEG_LAYER=1` の内側。既定 off)

### 1.1 層を 1 待ちで閉じる (`cuda::moe_seg_layer_sum`)

1 層 = **prologue 1 回の H2D + 5 起動 + イベント 1 本の待ち + D2H e 要素**:

```
[ptr_g|ptr_u|ptr_d|seg_gu|seg_d|w|x] を 1 本のホストバッファに詰めて H2D 1 回
→ gate → up → silu_mul → down → 重み付き和 → イベント待ち → D2H(e)
```

- silu と和は既存の `kernels_dec_epi.ptx` (`dno_dec_silu_mul` / `dno_dec_exp_sum`)。
  **新しい .cu / .ptx は作っていない** (PTX= 不要)。
- 和は `acc += w[t] * h[t*e+j]` を t の昇順に 1 本のアキュムレータで回す
  (`-fmad=false`) ので、**ミスの無い層では従来のホスト足し込みと同じ演算列**。
- 待ちは `cuCtxSynchronize` ではなく **null ストリームに打ったイベント**。
  文脈同期は非ブロッキングの上げ直しストリームまで待つため、他人の転送の
  完了待ちを毎層 2 回払っていた (推定。t278a の `prof` COND で切り分ける)。
- prologue のレイアウトは純関数 `expert_batch::seg_stage_layout` (テストで固定)。

### 1.2 ミスは VRAM へ上げず CPU で回す (`qwen4exp::moe_experts_seg_gpu_batched`)

- `lookup_many` でヒットだけ集め、**ミスの `acquire` (1 本 3 MB の写し + H2D) をしない**。
  ミスは `expert_miss_contribs_into` (base と同じ 8 スレッド経路) で GPU の裏で計算。
- 上げ直しは層の後で `iq_warm_misses` に渡す = **非同期** (背景スレッド +
  非ブロッキングストリーム、既存の task 214 経路)。VRAM のヒット率は保つ。
- ヒットが 0 本の層は `Contribs(全て None)` を返し、呼び出し側の CPU 経路へ。
- カーネル / PTX / 形が合わなければ `Ok(None)` で旧経路へ (fail closed)。

### 1.3 計測 (`model/seg_prof.rs`、`DENNO_SEG_PROF=1`、既定 off)

層の区間を VRAM 参照 / 表 / H2D / 起動 / 待ち / D2H / silu / 足し込み / ミス /
確保 / acquire に分け、カーネルの GPU 時間は **CUDA イベント**で測る。
トークン先頭 (層 0 の入口) で 1 行出す。旧経路にも同じ計器を入れた (t278a の
`prof` COND が旧経路の内訳、`segp` が新経路の内訳)。

## 2. 数値の扱い

- 上の表は lab の実測 (t268b のログ)。t278a の数字も lab の実測として書く。
- 「0.48 ms のうちいくらがカーネルか」は t278a のイベント計測で確定する。
  本 REPORT の 1.1 の「他人の転送待ち」は**推定**。
- 足し込みの順序が変わる (ヒットを先に全部、ミスを後) ので、ミスのある層では
  丸めが変わる。`DENNO_EXPERT_GPU_CHECK=1` で層ごとの max_abs / max_rel を出す。
  silu は `expf` の実装差で 1 ulp 級の差が出る (速度優先、タスク 278 の規則)。

## 3. 見積り

- 固定費 (2 同期 -> 1、小口 H2D 7 -> 1、D2H 153 KB -> 10 KB、ホスト silu/和の
  廃止) で **1 層 0.2〜0.4 ms (推定)**。
- ミス 1.6 本/層の写し + H2D (1 本 0.4 ms 級) を CPU 計算 (8 スレッド) に置換して
  **1 層 0.3〜0.5 ms (推定)**。
- 合計 0.5〜0.9 ms/層 = 24〜43 ms/tok。目標 (base 65.9 -> 49.4 ms/tok) は
  前者寄りなら届く。t278a で確定する。

## 4. テスト (CPU)

- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`
  -> 1146 passed / 0 failed / 100 ignored
- `cargo test --release -p denno-engine --features cuda --test kernel_invariants`
  -> 90 passed / 0 failed
- 追加: `expert_batch::tests::seg_layer_gate_defaults_off` /
  `seg_stage_layout_is_packed_and_aligned`、`seg_prof::tests::*` (off で積まない /
  on で積んで 1 行に出す)。

## 5. GPU で確認してほしいこと (lab t278a)

- `base` vs `seg` (`DENNO_SEG_LAYER=1`) の decode / prefill / md5 / errors。
- `prof` (旧経路 + `DENNO_SEG_PROF=1`) と `segp` (新経路 + prof) の `[seg-prof]` 行:
  1 層あたりの gate/up/silu/down/sum の GPU 時間と GB/s、待ち、ミス CPU 時間。
- `DENNO_EXPERT_GPU_CHECK=1` を付けた COND で `expert_gpu_check` の
  max_abs / max_rel (新経路の数値が CPU と合っているか)。

## 6. t278a の結果

(未記入。lab の RESULT.md から転記する)
