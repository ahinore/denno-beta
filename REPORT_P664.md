ADOPT: yes t664a

# task 664 — `src/cuda.rs` (70,795 行) を `src/cuda/` の 10 ファイルへ純粋分割

## 前提
- ブランチ `br664` = `int-auto` (d0483acab)。作業後の再 fetch でも int-auto は動いておらず、
  マージは不要だった (d0483acab のまま)。
- 分割前の `crates/denno-engine/src/cuda.rs` は 70,795 行 (非空行 67,418)。
- 数値・ロジック・カーネル・PTX・env ゲートは一切変えていない。`.cu` / `.ptx` は触っていない。

## 変えたこと
`crates/denno-engine/src/cuda.rs` を削除し、`crates/denno-engine/src/cuda/` へ移した。
アイテムの順序は元ファイル順のまま (モジュールごとに切り分けただけ)。

| ファイル | 行数 | 中身 |
| --- | --- | --- |
| `src/cuda/mod.rs` | 11,119 | driver FFI (`CudaLib` 他)、`CudaCtx`/`DevBuffer`/`PinMem`/`MapMem`/`Scratch` の定義、`weights_for_gguf`、`shared_ctx`、`tline`、`launch_dbg`、`dec_kprof`、共通ヘルパ、`pub use self::<submod>::*;` |
| `src/cuda/attn.rs` | 10,143 | decode/prefill attention、GPU KV ミラー、flashdec (`dec_full_dev`, `q4attn_kprof`)、`rope`、legacy `chain_*` / qwen35 solo 鎖 |
| `src/cuda/tests.rs` | 9,949 | 複数ドメインにまたがる `mod tests` と p59/taskNNN 系の共有テスト |
| `src/cuda/gdn.rs` | 8,498 | GDN decode / prefill (`gdn_decode_submit`, `gdn_pf_*`, `gdn_ba_gpu`, `dec_exp_kprof`) |
| `src/cuda/mix.rs` | 7,713 | Hyper-Connections mix (decode / prefill / inj / steer / `pf_mix_kprof`) |
| `src/cuda/gemm.rs` | 7,219 | dense / tensor-core prefill GEMM、`router_gpu`、`lm_head_f32`、`f32_gemm` |
| `src/cuda/pf_chain.rs` | 6,516 | prefill 常駐チェーン (`pf_begin`, `pf_pool`, `pf_res_park`, `pf_elem`, `pf_ple_dev`, `pf_full_*`, `res_xfer`) |
| `src/cuda/moe_pf.rs` | 5,993 | prefill MoE GEMM / streaming (`exp_iq_pf`, `pf_stream`, `pf_tab`, `pf_up_worker`, `exp_*_tc`) |
| `src/cuda/ctx.rs` | 3,220 | 基盤型の impl 群 (`impl CudaCtx` 1,813 行ほか。impl は名前を持たないので re-export しない) |
| `src/cuda/moe_dec.rs` | 2,995 | decode MoE GEMV (`v1..v5`, `seg_k`, `seg_ev`, `exp_iq`, `dec_epi`, `shared_overlap`) |
| 合計 | 73,365 | 元 70,795 行 + 前置き/可視性トークン |

外部 API は不変 (`crate::cuda::*` のパスは全て同じ)。`pub use self::<submod>::*;` で
再輸出しているため、`denno_engine::cuda::gdn_scan_batch` 等の外部参照は無変更で通る。

移動に伴って必要だった機械的な差分 (ロジックの変更ではない):
- 可視性: サブモジュール化で狭くなる分だけ `pub(crate)` を足した。欄 (フィールド) は
  コンパイラが「private field です」と言ったものだけを広げた (過剰に広げない)。
  メソッドも同様にコンパイル結果に従って広げた。
- `pub(super)` → `pub(crate)` (階層が 1 段深くなり、`pub(super)` の意味が変わるため)。
- `include_bytes!` / `include_str!("../kernels/…")` → `"../../kernels/…"`。
- 自ファイルのテキストを読むテスト (`include_str!("cuda.rs")` 2 箇所) は
  `crate::cuda::cuda_src()` (テスト時のみ。`src/cuda/*.rs` をソート順に `"\n"` 連結) へ。
- `anyhow::anyhow!` → `anyhow!` (33 箇所)。glob 経由で入る `anyhow` 名と extern
  prelude の crate 名が衝突するため (マクロの解決先は同じ)。
- `tests/kernel_invariants.rs`: `const CUDA_RS: &str = include_str!("../src/cuda.rs")` を
  `fn cuda_rs() -> &'static str` (OnceLock + `src/cuda/*.rs` をソート順に読む) に置換。
  実行時読み込み 2 箇所も同じヘルパへ。テストの主張は落としていない。唯一
  `assert!(cuda_rs().contains("    pin_ev: usize,"))` を `"pin_ev: usize,"` に変えた
  (この欄は `cuda/mod.rs` の構造体リテラルからも触るため `pub(crate)` になり、字下げを
  固定できなくなった。欄名と型の検査はそのまま)。

## 数値の扱い
数値の変更なし。行の内容が「可視性の付与 / `anyhow!` / `../../kernels` /
`pub(crate)`」以外で元と一致することを行単位で照合した (原 67,418 非空行に対して
説明のつかない差分は 0)。決定的な順序: ソース連結は `mod.rs` を含む `src/cuda/*.rs`
のファイル名ソート順。

## テスト (分割前と同数・全緑)
- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`
  → **1430 passed / 0 failed / 103 ignored** (合計 1533 = 分割前と同数)
- `cargo test --release -p denno-engine --features cuda --test kernel_invariants -- --test-threads=1`
  → **155 passed / 0 failed** (分割前と同数)
- `cargo build --workspace --release` → ok
- `cargo test --workspace -- --test-threads=1` (cuda feature 無し) → **63 テストバイナリすべて ok、0 failed**
- `cargo check --workspace --features denno-engine/cuda` → **0 error、警告 75 件で分割前と完全一致**
  (`cargo check -p denno-engine --features cuda` の警告も分割前と 1 件も変わらない)

## ラボ (実測)
### t664a — この commit (a20441c32)
```
tree=denno-beta sha=a20441c32e745730d5b316f9c100df7faf322d97 scenario=decode
model=flash-next-UD-IQ3_XXS ptx=none load_before=1.67

| cond | decode ms/tok (median) | prefill s (median) | errors | text | md5 |
| x0   | 9.8 | 0.56 | 0 | looks ok | a:edaa84ad b:1938059f |
| base | 9.8 | 0.56 | 0 | looks ok | a:edaa84ad b:1938059f |

VERDICT base vs x0 OK decode +0.0% prefill +1.8% md5 SAME errors 0
VERDICT baseline x0 errors 0 text ok
```
COND は両方とも baseline 環境 (`DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000
DENNO_EXPERT_GPU_MIN_HIT=0 DENNO_GPU_WEIGHT_MB=12000`)。x0 は捨て COND。

### t664b — 比較用の int-auto (d0483acab)
```
tree=denno-beta sha=d0483acab scenario=decode model=flash-next-UD-IQ3_XXS ptx=none load_before=1.84

| cond | decode ms/tok (median) | prefill s (median) | errors | text | md5 |
| x0   | 9.8 | 0.56 | 0 | looks ok | a:edaa84ad b:1938059f |
| base | 9.8 | 0.56 | 0 | looks ok | a:edaa84ad b:1938059f |

VERDICT base vs x0 OK decode +0.0% prefill +1.8% md5 SAME errors 0
VERDICT baseline x0 errors 0 text ok
```
**md5 は int-auto と完全一致** (`a:edaa84ad` / `b:1938059f`)、decode 9.8 ms/tok で
同じ (差 0.0%)、errors 0、TXT/TXTB の文面も sane。タスクの「md5 一致・速度 ±1% 以内」を
実測で満たした。2 ラウンド使ったのは md5 の同一性を推測ではなく実測にするため
(タスクの想定は 1 ラウンド)。

## GPU で確認してほしいこと
- `SCENARIO=depth` と `SCENARIO=tools` でも md5 が int-auto と一致すること
  (この 2 本は decode シナリオに含まれない prefill 経路のテキストを通すため)。
- 追加の確認は無し。構造だけの変更なので、以後の perf タスクの基準線は従来どおりでよい。
