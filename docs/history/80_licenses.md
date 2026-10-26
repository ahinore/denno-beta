> [索引](README.md) / 前: [第8巻 計測・検証・運用](70_infra.md) / 次: [付録A 全 phase 一覧](90_phase_index.md) / [English](en/80_licenses.md)

# 付録B 参照実装とライセンス帰属

このリポジトリは外部実装を参照して書かれている。**どれを、どこまで、どのライセンスで取り込んだか**を
1 枚にまとめる。各ソースファイルのヘッダにも同じ内容を明記してあり、こちらが台帳、ヘッダが履行になる。

| 参照元 | ライセンス | 使い方 | 判定 |
|---|---|---|---|
| llama.cpp `ggml-quants.c` (`dequantize_row_*`) | MIT (Copyright (c) 2023-2026 The ggml authors) | **忠実移植**（ビット一致契約） | OK — MIT の表示義務を `kquant.rs` ヘッダで履行 |
| `ggml-common.h` の kquant テーブル（`kquant_tables.rs` を自動生成） | 同上 | **テーブル値の逐語抽出**（`gen_kquant_tables.py`） | OK — 生成物ヘッダに著作権表示を埋め込み（再生成済み） |
| llama.cpp `delta-net-base.cpp` / `qwen35.cpp`（GDN の数学） | 同上 | 数式・演算順序の移植（Rust 再実装・逐語コピー無し） | OK — `gdn.rs` ヘッダに出典表示 |
| llama.cpp `delta-net-base.cpp`（GDN の数学 → CUDA カーネル） | 同上 | 同一の数式を CUDA へ移植（`kernels/gdn.cu`）。GDN は bit 一致不要の契約なので `expf` 等を使い、決定論のみ保証 | OK — `gdn.cu` ヘッダに MIT 表示 |
| GGUF 形式仕様（`gguf.cpp`） | 同上 | 形式仕様の参照のみ（パーサは独自実装） | OK — `gguf.rs` ヘッダに出典表示 |
| llama.cpp `src/models/qwen4exp.cpp`（commit `d854d98`、2026-10-26 取得） | MIT (Copyright (c) 2023-2026 The ggml authors) | Qwen4 系アーキテクチャの数式参照（PLE の n-gram 索引、hyper-connection、QSA indexer、MoE のルーティング順序）。Rust / CUDA へ再実装し、逐語コピーは無し | OK — 移植先（`moe.rs` / `ple.rs` / `hyper.rs` / `indexer.rs`）のヘッダに出典表示 |
| Qwen3.5 モデル（設定値・重み） | Apache-2.0 | 設定値の参照と GGUF 重みのロード | OK — `qwen35.rs` ヘッダに出典表示 |
| Qwen3.8-Flash-Next（設定値・重み、`qwen4exp` アーキテクチャ） | Qwen Community License 1.0 | 設定値の参照と GGUF 重みのロード、ローカル推論のみ。重みは再配布しない | OK — 著作権表示を本付録に保持。Model as a Service 用途には使わない |

**結論** — 本リポジトリのライセンス（MIT OR Apache-2.0）とすべて両立する。MIT 由来の部分（kquant の移植・
テーブル抽出）は MIT の表示義務に従い、upstream の著作権表示を該当ファイルに保持している。
GPL などの汚染は無い。モデル重みのライセンス（Apache-2.0 / Qwen Community License 1.0）は重みそのものに
適用され、本リポジトリのコードのライセンスには及ばない。重みは再配布しない。

**運用** — 新しく参照実装を取り込むときは、この表と該当ファイルのヘッダの**両方**に追記する。
片方だけでは台帳として不十分になる（表だけ直してヘッダを忘れると、ファイル単体を持ち出したときに
出典が消える）。

> この表は旧 `docs/HISTORY.ja.md` の「参照実装とライセンス確認（2026-10-14）」を、開発史の巻構成へ
> 移したもの。記載内容は当時のまま（判定・ライセンス・使い方の区別）で、行だけ読みやすく整えた。
