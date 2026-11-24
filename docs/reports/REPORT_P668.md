ADOPT: yes t668a

# REPORT P668 — qwen4exp.rs (31,171 行) の純粋分割

## 前提
- int-auto `599f871fb` (merge: br664) から `br668`。int-auto は作業中に動かなかった
  (最終確認でも `599f871fb`) ので merge は不要。
- 原ファイル `crates/denno-engine/src/model/qwen4exp.rs` は 31,171 行 / 544 個の
  トップレベル項目。`impl Qwen4Exp` は 2 つあり、1 つは 3,028-6,476 行の 3,455 行
  (decode 本線 + prefill 本体)、もう 1 つは 11,192-11,272 行の arch 出力 4 本だけ。
  前者の中は associated item が 1 段インデントで並ぶ (カラム 0 の項目も混在する) ので、
  行頭の字下げではなくブレース深さで項目境界を求めて切り分けた。

## 変えたこと
`src/model/qwen4exp.rs` → `src/model/qwen4exp/` の 8 ファイルへの**純粋な移動**。

| ファイル | 行数 | 中身 |
| --- | ---: | --- |
| mod.rs | 5,740 | ヘッダ / 型・パラメータ・テンソル読み込み / `Qwen4Exp`・`LayerW` 等の型 / decode 本線 (`forward_token*` / `forward_token_at` / `run_layer` / spec / arch 出力) |
| prefill.rs | 6,849 | `forward_prefill*` / 常駐連鎖 (`pf_*`) / `moe_ffn_quant_batch*` / expert 先読み / GEMV バッチ |
| mix.rs | 2,599 | `hc_mix_quant*` / mix ゲート / PLE 注入 (`ple_inject*` / `ple_project_kv` / hist) |
| moe.rs | 4,478 | `moe_ffn_quant*` / expert 選択・ゲート・VRAM 常駐・miss 貢献 / expert batch GEMV |
| prof.rs | 1,392 | `q4prof` / `mix_kprof` / `attn_kprof` / `gdn_gpu_win` / `headprof` / `pf_ple_kprof` |
| tests.rs | 4,483 | `mod tests` の前半 (合成 GGUF ヘルパー + 81 本の `#[test]`) |
| tests_h2.rs | 5,505 | `mod tests` の後半 `mod h2` (91 本の `#[test]`)。tests.rs から `include!` |
| tests2.rs | 198 | `mod task225_router_dev_mirror` と `tensor_role_groups_by_suffix` |

最大 6,849 行 (目標の 8,000 行以下)。`mod.rs` に `pub use <submod>::*;` を置き、
`crate::model::qwen4exp::*` の外部経路は 1 本も変えていない
(denno-cli が使う `LayerOut` / `q4prof` / `mix_kprof` / `attn_kprof` /
`moe_sub_take` / `mix_qchain_line` / `exp_miss_rows_report` /
`expert_identity_bytes` / `spawn_expert_plan` はすべて従来どおり解決)。

移動に伴う機械的な差分だけ:
- 可視性: サブモジュールで狭くなる分に `pub(crate)` を足した (トップレベル項目・
  `impl` 内の associated item・struct フィールド・tuple struct のフィールド・
  `thread_local!` 内の `static`)。**公開 API の可視性は広げていない**
  (元が `pub` のものは `pub` のまま、元が無指定のものだけ `pub(crate)`)。
- `impl Qwen4Exp` は移動先ごとに impl ブロックを分けて再開した (associated item を
  別の impl ブロックに置くのは意味的に等価)。
- ソース文字列を見るテストの配線 (assert は弱めていない):
  - `tests/kernel_invariants.rs`: `include_str!("../src/model/qwen4exp.rs")` 2 箇所
    (QWEN4EXP_RS / QWEN_SRC) と実行時読み込み 2 箇所を `qwen4exp_src()` に置換。
    `src/model/qwen4exp/*.rs` を**ファイル名の昇順**で連結する (OnceLock で 1 回だけ。
    決定的)。分割前の 1 ファイルと同じ「全体」を返すので、行数・出現回数を見る
    assert (`host.matches("hc_mix_quant_defer(").count()` 等) も従来どおり効く。
  - `qwen4exp/tests.rs` (元 L21004) の `include_str!("qwen4exp.rs")` は
    `crate::model::qwen4exp::src_text()` (実装 5 ファイルの `concat!(include_str!..)`) へ。
    この 1 本は `fn pf_early_route_prefetch(` の本体を見るだけなので実装側で足りる。
- 変更禁止のものは触っていない: `.cu` / `.ptx` の変更なし、ゲートの既定の変更なし、
  数式・丸め順の変更なし、テストの削除なし (`#[test]` は原 188 本 → 新 188 本)。

### 行単位の照合 (純粋移動の証拠)
原ファイルの非空行 29,675 行を新しい 8 ファイルの非空行 29,728 行と多重集合で突き合わせ、
「元の行」または「元の行に `pub(crate) ` を足した行」で説明できるかを 1 行ずつ見た。
説明できなかった元の行は 4 行だけ:
1. `mod tests {` → `mod tests;` + ファイル分割 (意図した変更)
2. `let src = include_str!("qwen4exp.rs");` → `src_text()` 呼び出し (意図した変更)
3. `struct PrefillSeqGuard(bool);` → `pub(crate) struct PrefillSeqGuard(pub(crate) bool);`
4. `    pub struct GdnGpuWinGuard(bool);` → フィールドに `pub(crate)`
増えた側の説明できない行は `src_text()` の `    )` 1 行だけ。3/4 は tuple struct の
フィールド可視性 (crate 内の可視性を広げただけ。値・API は不変)。

## 数値の扱い
数値の変更なし。丸め順・カーネル選択・env ゲートの既定は 1 つも変えていないので、
lab の md5 が分割前 (task 664 の `a20441c32`) と一致することを確認した。

## テスト (分割前と同数・全緑)
- `cargo test --release -p denno-engine --features cuda --lib -- --test-threads=1`
  → **1430 passed / 0 failed / 103 ignored** (task 664 の分割前と同じ本数)
- `cargo test --release -p denno-engine --features cuda --test kernel_invariants -- --test-threads=1`
  → **155 passed / 0 failed** (同上)
- `cargo build --workspace --release` → ok
- `cargo test --workspace -- --test-threads=1` (cuda feature 無し) → **63 テストバイナリ
  すべて ok、0 failed** (rc=0)
  - 1 回目は並行ビルドで機械が飽和していたため
    `denno-server --test admit_error_message` の
    `all_slots_busy_returns_429_with_capacity_message` が 20 s タイムアウトで落ちた。
    その 1 本だけを単独再実行 → 7 passed、その後の full run → rc=0。負荷由来の
    フレークで、分割とは無関係 (`denno-server` の admission の待ち時間だけを見るテスト)。
- ソースの文字列を見るテストは全部そのまま通った (assert を緩めた箇所はない)。

## ラボ (実測)
### t668a — この commit (af1eaad40)
```
tree=denno-beta sha=af1eaad40 scenario=decode model=flash-next-UD-IQ3_XXS ptx=none
load_before=6.18

| cond | decode ms/tok (median) | prefill s (median) | errors | text | md5 |
| x0   | 9.8 | 0.56 | 0 | looks ok | a:edaa84ad b:1938059f |
| base | 9.8 | 0.56 | 0 | looks ok | a:edaa84ad b:1938059f |

VERDICT base vs x0 OK decode +0.4% prefill +0.9% md5 SAME errors 0
VERDICT baseline x0 errors 0 text ok
```
- md5 は task 664 のベースライン (`a20441c32` の x0/base) と同じ
  `a:edaa84ad b:1938059f`。**分割前後で出力が 1 バイトも変わっていない**。
- decode 9.8 ms/tok・prefill 0.56 s も task 664 の値と同一。差は +0.4% / +0.9% で
  測定ノイズの範囲 (この回は load_before=6.18 と高負荷だったので、その割に安定)。
- errors 0、TXT/TXTB の本文も sane (`# B-Trees: ...` / Rust の所有権の説明)。

## 見積り (推定)
- ビルド時間への影響はほぼ無い (同一クレート内のファイル分割なので、rustc の
  クレート単位の並列度は変わらない)。cuda 無しの `cargo test --workspace` が
  分割前 5-10 分台と同じ所要だった。
- 今後の decode/prefill 変更 (644 / 636 / 648 / 661) では、触る関数が
  どのファイルかが名前で分かるぶんレビューが速くなる見込み (推定)。

## GPU で確認してほしいこと
- 無し (t668a で decode の md5 一致・errors 0・性能同等を確認済み)。
- 次に prefill 系のタスク (644 など) を回すとき、`prefill.rs` の
  `forward_prefill_batch_resident*` と `pf_*` が同一ファイルにあることを前提にできる。
