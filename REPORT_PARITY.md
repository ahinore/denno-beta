# REPORT_PARITY — 開発史の日英同期チェッカ

タスク: `/tmp/task_parity.txt`。branch `parity`、親 `d4ee7d9`。
CPU のみ・ネットワークなし・docker なし・GPU なし。`docs/history/` の中身は触っていない。

## 前提 (与えられた条件)

- `docs/history/` に ja 11 本、`docs/history/en/` に en 11 本 (索引・8 巻・付録 A/B)。
- 英語版は翻訳なので、ja を直すと en が取り残される。機械的に検出したい。
- CI には入れない。手で回せれば十分。
- 既存 `scripts/*.py` の流儀 (argparse / 0=OK,1=判定失敗,2=前提壊れ / 標準ライブラリのみ) に合わせる。

## 調べたこと

- `scripts/prefill_bench.py` は位置引数 + `sys.argv`、`scripts/needle_check.py` は argparse で
  `EXIT_OK=0 / EXIT_RECALL_FAIL=1 / EXIT_TRANSPORT=2`。終了コードの意味を 0/1/2 で揃えた。
- `scripts/README.md` は存在する (115 行)。表 + 節の構成なので、表に 1 行足し、末尾に節を足した。
- 実データの形 (実測、`python3` で数えた):
  - ja 11 本すべてに `> [索引]` ナビ行があり、`[English](en/<同名>.md)` を含む。
  - `README.md` (索引) だけはナビ行を持たない → 例外扱いが必要 (下記)。
  - en 11 本すべての 1〜2 行目に `> Japanese original: [../<同名>.md](../<同名>.md)`。
  - コードフェンスは ja/en とも 0 個。表の行は `90_phase_index.md` が 147 行で最大。
  - backtick 囲みの 16 進トークンは `40_prefill_perf.md` に md5 相当の 32 桁が 1 個
    (`e36644365c6746b5ee1a06720d5ad4a1`)。両側に同じものがあるので SHA 集合の差は 0。
  - 数値トークンは ja/en で最大の差が `70_infra.md` の 148 / 122 (en が 17.6% 少ない)。
    中身を追うと「2 回」→ "twice"、「2 通り」→ "two ways" の翻訳で数字が語に変わる箇所が主因。
    **データの欠落ではない** (17.6% は WARN 帯。FAIL 帯の 30% には届かない)。

## 変えたこと

- 追加: `scripts/check_history_parity.py` (標準ライブラリのみ、読み取り専用)。
- 追記: `scripts/README.md` — 表に 1 行、末尾に「The history parity check (parity)」節。
- 追加: この `REPORT_PARITY.md`。
- `docs/history/` は 1 バイトも変えていない (`git status --porcelain` に docs 配下は出ない)。

## 検査項目と判定基準

| 項目 | 比較 | FAIL | WARN |
|---|---|---|---|
| ファイル対応 | ja の `*.md` 集合 vs en の `*.md` 集合 | 片方にしか無いファイルがある | — |
| 見出し | `^#` で始まる行の数 | 不一致 | — |
| 表の行 | `^\|` で始まる行の数 | 不一致 | — |
| コードブロック | コードフェンスで閉じた区間の数 | 不一致 / 閉じ忘れ | — |
| commit SHA | backtick 囲み `[0-9A-Fa-f]{7,40}` の集合 | ja にあって en に無い SHA | en にしか無い SHA |
| 数値トークン | `[0-9]+(\.[0-9]+)?` の個数 | en が 30% 超少ない | en が 10% 超少ない |
| en の冒頭 | 先頭 10 行の `Japanese original:` 行とリンク先 | 行が無い / リンクが無い / リンク先が無い | リンク先が同名の ja でない |
| ja のナビ行 | `> [索引]` 行の `[English](en/...)` | ナビ行が無い / リンクが無い / リンク先が無い | (なし) |

索引 `README.md` はナビ行を持たないので、代わりに本文の `en/README.md` リンクを見る
(これが無ければ FAIL)。この例外はスクリプトの docstring と `scripts/README.md` に書いた。

数値の閾値は「en が ja より何%少ないか」= `(ja - en) / ja`。en が多い分は PASS (訳が増えた)。

## 実行結果 (実物)

```
$ python3 scripts/check_history_parity.py; echo EXIT=$?
history parity: /work/denno-beta/docs/history
PASS files                        ja 11 本 / en 11 本 / 11 対
PASS 01_bootstrap.md headings     見出し 17/17 (ja/en)
PASS 01_bootstrap.md table        表の行 9/9 (ja/en)
PASS 01_bootstrap.md fences       コードブロック 0/0 (ja/en)
PASS 01_bootstrap.md sha          SHA 7/7 (ja/en)
PASS 01_bootstrap.md numbers      数値トークン 236/237 (ja/en): en が 0.4% 多い
PASS 01_bootstrap.md backlink     リンク先 ../01_bootstrap.md が実在
PASS 01_bootstrap.md nav          ナビ行 -> en/01_bootstrap.md
PASS 10_api_features.md headings  見出し 23/23 (ja/en)
PASS 10_api_features.md table     表の行 15/15 (ja/en)
PASS 10_api_features.md fences    コードブロック 0/0 (ja/en)
PASS 10_api_features.md sha       SHA 14/14 (ja/en)
PASS 10_api_features.md numbers   数値トークン 385/379 (ja/en): en が 1.6% 少ない
PASS 10_api_features.md backlink  リンク先 ../10_api_features.md が実在
PASS 10_api_features.md nav       ナビ行 -> en/10_api_features.md
PASS 20_models.md headings        見出し 38/38 (ja/en)
PASS 20_models.md table           表の行 29/29 (ja/en)
PASS 20_models.md fences          コードブロック 0/0 (ja/en)
PASS 20_models.md sha             SHA 31/31 (ja/en)
PASS 20_models.md numbers         数値トークン 1752/1733 (ja/en): en が 1.1% 少ない
PASS 20_models.md backlink        リンク先 ../20_models.md が実在
PASS 20_models.md nav             ナビ行 -> en/20_models.md
PASS 30_decode_perf.md headings   見出し 39/39 (ja/en)
PASS 30_decode_perf.md table      表の行 35/35 (ja/en)
PASS 30_decode_perf.md fences     コードブロック 0/0 (ja/en)
PASS 30_decode_perf.md sha        SHA 27/27 (ja/en)
PASS 30_decode_perf.md numbers    数値トークン 1831/1812 (ja/en): en が 1.0% 少ない
PASS 30_decode_perf.md backlink   リンク先 ../30_decode_perf.md が実在
PASS 30_decode_perf.md nav        ナビ行 -> en/30_decode_perf.md
PASS 40_prefill_perf.md headings  見出し 58/58 (ja/en)
PASS 40_prefill_perf.md table     表の行 71/71 (ja/en)
PASS 40_prefill_perf.md fences    コードブロック 0/0 (ja/en)
PASS 40_prefill_perf.md sha       SHA 56/56 (ja/en)
PASS 40_prefill_perf.md numbers   数値トークン 3135/3055 (ja/en): en が 2.6% 少ない
PASS 40_prefill_perf.md backlink  リンク先 ../40_prefill_perf.md が実在
PASS 40_prefill_perf.md nav       ナビ行 -> en/40_prefill_perf.md
PASS 50_gdn.md headings           見出し 31/31 (ja/en)
PASS 50_gdn.md table              表の行 26/26 (ja/en)
PASS 50_gdn.md fences             コードブロック 0/0 (ja/en)
PASS 50_gdn.md sha                SHA 23/23 (ja/en)
PASS 50_gdn.md numbers            数値トークン 1157/1129 (ja/en): en が 2.4% 少ない
PASS 50_gdn.md backlink           リンク先 ../50_gdn.md が実在
PASS 50_gdn.md nav                ナビ行 -> en/50_gdn.md
PASS 60_quality.md headings       見出し 23/23 (ja/en)
PASS 60_quality.md table          表の行 10/10 (ja/en)
PASS 60_quality.md fences         コードブロック 0/0 (ja/en)
PASS 60_quality.md sha            SHA 12/12 (ja/en)
PASS 60_quality.md numbers        数値トークン 443/430 (ja/en): en が 2.9% 少ない
PASS 60_quality.md backlink       リンク先 ../60_quality.md が実在
PASS 60_quality.md nav            ナビ行 -> en/60_quality.md
PASS 70_infra.md headings         見出し 26/26 (ja/en)
PASS 70_infra.md table            表の行 16/16 (ja/en)
PASS 70_infra.md fences           コードブロック 0/0 (ja/en)
PASS 70_infra.md sha              SHA 0/0 (ja/en)
WARN 70_infra.md numbers          数値トークン 148/122 (ja/en): en が 17.6% 少ない (>10.0%)
PASS 70_infra.md backlink         リンク先 ../70_infra.md が実在
PASS 70_infra.md nav              ナビ行 -> en/70_infra.md
PASS 80_licenses.md headings      見出し 1/1 (ja/en)
PASS 80_licenses.md table         表の行 8/8 (ja/en)
PASS 80_licenses.md fences        コードブロック 0/0 (ja/en)
PASS 80_licenses.md sha           SHA 0/0 (ja/en)
PASS 80_licenses.md numbers       数値トークン 15/15 (ja/en): en が 0.0% 少ない
PASS 80_licenses.md backlink      リンク先 ../80_licenses.md が実在
PASS 80_licenses.md nav           ナビ行 -> en/80_licenses.md
PASS 90_phase_index.md headings   見出し 6/6 (ja/en)
PASS 90_phase_index.md table      表の行 147/147 (ja/en)
PASS 90_phase_index.md fences     コードブロック 0/0 (ja/en)
PASS 90_phase_index.md sha        SHA 132/132 (ja/en)
PASS 90_phase_index.md numbers    数値トークン 1128/1114 (ja/en): en が 1.2% 少ない
PASS 90_phase_index.md backlink   リンク先 ../90_phase_index.md が実在
PASS 90_phase_index.md nav        ナビ行 -> en/90_phase_index.md
PASS README.md headings           見出し 11/11 (ja/en)
PASS README.md table              表の行 81/81 (ja/en)
PASS README.md fences             コードブロック 0/0 (ja/en)
PASS README.md sha                SHA 0/0 (ja/en)
PASS README.md numbers            数値トークン 640/620 (ja/en): en が 3.1% 少ない
PASS README.md backlink           リンク先 ../README.md が実在
PASS README.md nav                索引: en/README.md へのリンクあり
------------------------------------------------------------
summary: 77 PASS / 1 WARN / 0 FAIL (78 checks, 0 file missing) -> OK
EXIT=0
```

FAIL 0 件、WARN 1 件 (`70_infra.md numbers`)、PASS 77 件。終了コード 0。
同じ入力を 2 回走らせて出力はバイト一致 (`cmp` で確認)。

## わざとずらした場合に FAIL するか

リポジトリのファイルは触らず、`cp -a docs/history /tmp/parity_*` のコピーに対して
`--history-dir` で回した。13 ケースすべて期待どおりの判定になった。

| ずらし方 | 結果 | 出た行 |
|---|---|---|
| `en/60_quality.md` を削除 | exit 1 | `FAIL files  ja にしか無い: 60_quality.md` |
| `en/99_new.md` を追加 | exit 1 | `FAIL files  en にしか無い: 99_new.md` |
| en の見出しを 1 本削除 | exit 1 | `FAIL 01_bootstrap.md headings  見出し 17/16 (ja/en)` |
| en の表の行を全部削除 | exit 1 | `FAIL 01_bootstrap.md table  表の行 9/0 (ja/en)` |
| en の SHA 1 種を全置換 | exit 1 | `FAIL 20_models.md sha  en に無い SHA 1 件: 0bbfa79` |
| en にだけ `deadbeef` を追加 | exit 0 | `WARN 70_infra.md sha  ja に無い SHA 1 件: deadbeef` |
| ja に数値を +400 (en が 51.7% 少) | exit 1 | `FAIL 10_api_features.md numbers  数値トークン 785/379 ... (>30.0%)` |
| ja に数値を +60 (en が 14.8% 少) | exit 0 | `WARN 10_api_features.md numbers  数値トークン 445/379 ... (>10.0%)` |
| en の `Japanese original:` 行を削除 | exit 1 | `FAIL 01_bootstrap.md backlink  先頭 10 行に "Japanese original:" の行が無い` |
| ja のナビ行から `[English](...)` を削除 | exit 1 | `FAIL 01_bootstrap.md nav  ナビ行に [English](...) が無い` |
| en に閉じないフェンスを追加 | exit 1 | `FAIL 80_licenses.md fences  閉じていないフェンスがある (ja=なし en=あり)` |
| en に閉じたフェンスを 1 個追加 | exit 1 | `FAIL 80_licenses.md fences  コードブロック 0/1 (ja/en)` |
| 存在しないディレクトリ | exit 2 | `ERROR: history ディレクトリが無い: /tmp/does_not_exist_history` |

`/tmp` のコピーだけを壊しているので、リポジトリ側の復元作業は発生していない
(実行後の `git status --porcelain` に `docs/` の変更が無いことで確認)。

## 数値不変の根拠

- 実行するコード (エンジン・サーバ) を 1 行も変えていない。`scripts/` と `REPORT_PARITY.md` だけ。
- したがって出力・速度・アンカーの数値は定義上不変。ここに実測値を書く必要はない。
- このタスクで私が測ったのは「ファイル内の件数」だけ (上記の表の数字)。
  GPU 実測 (ms/token・アンカー md5) は取っていないし、書いていない。

## テスト

- `python3 scripts/check_history_parity.py` → exit 0 (77 PASS / 1 WARN / 0 FAIL)。
- 2 回実行して出力バイト一致。`--help` → exit 0。
- 上表の 13 ケース (FAIL 10 / WARN 2 = exit 0 / exit 2 が 1) を `/tmp` のコピーで確認。
- `cargo test` は走らせていない: Rust 側を触っていないため (CPU ゲートに影響なし)。
  `scripts/run_cpu_checks.sh` にもこのスクリプトは入れていない (タスクが CI 不要と言っている)。

## 次にやるなら

- `70_infra.md` の WARN を消すには、英語側で「twice」等にした箇所へ数字を戻すか、
  判定を「数値トークン + 日本語の助数詞」で数えるように変える。後者は実装が増えるので保留。
- 数値の「値そのもの」の一致 (ja の 5,686 が en にも 5,686 であるか) は見ていない。
  集合比較にすると訳文の語順で誤検知が出るため、件数だけにした。必要なら別スクリプト。
- `scripts/run_cpu_checks.sh` の 1 ステージに足すのは簡単 (0.2 秒) だが、タスクの指示どおり入れていない。
