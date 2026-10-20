#!/usr/bin/env python3
"""s120: scripts/ngram_oracle.py の CPU テスト (GPU 不要・標準ライブラリのみ)。

確かめること:
  - 計測の定義: 直前の出現を採る / 機会は「前回の続きが k トークンそろっている」ときだけ /
    一致は「前回の続き == 今回の続き」のときだけ (手計算できる小さな入力で固定する)
  - 母数 0 の扱い: p は None になり 0 除算にならない (空入力・短すぎる入力・
    文脈が一度も再出現しない入力)
  - k が文脈長を超える場合 (k > n) も機会・一致の定義どおりに動く
  - 非 UTF-8 入力は skip され、その旨が JSON の skipped に出る (例外で落ちない)
  - 集計: 文書をまたいで hit / opportunity を足し上げる (文書間の文脈は見ない)
  - 決定論: 同じ入力からは常に同じ結果 (2 回走査しても byte 一致の JSON)
  - コーパス: docs_proxy の内訳 (docs/*.ja.md, *.en.md, crates/**/*.rs) と、
    合成コーパスが固定シードで決定的であること
  - 実行経路: --self-test / --json / --path / 不正な引数の終了コード

実行:
  python3 scripts/test_ngram_oracle.py -v
  python3 -m unittest discover -s scripts -p 'test_ngram_oracle.py'
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "ngram_oracle.py")
sys.path.insert(0, HERE)

import ngram_oracle as orc  # noqa: E402  (同じディレクトリの実装を直接読み込む)


def run(args, cwd=None):
    return subprocess.run([sys.executable, SCRIPT] + args,
                          capture_output=True, text=True, cwd=cwd)


def rows(report):
    return {row["k"]: row for row in report["k"]}


class TestOracleDefinition(unittest.TestCase):
    """手計算できる入力で、定義 (直前の出現・機会・一致) を固定する。"""

    def test_previous_occurrence_is_used_not_the_first(self):
        # "a b x c a b y d a b e" の 2 回目以降の (a,b) の続きは前回と食い違う。
        # 最初の出現 (x) ではなく直前の出現 (y) を採るので両方不一致。
        toks = list("abxcabydabe")
        row = rows(orc.oracle(toks, n=2, ks=(1,)))[1]
        self.assertEqual(row["opportunity"], 2)
        self.assertEqual(row["hit"], 0)
        self.assertEqual(row["p"], 0.0)

    def test_periodic_text_never_matches_at_the_same_phase(self):
        # [1,2,3,4] x 2 + [1,2]: 直前の出現は常に 4 つ前なので続きがずれる。
        toks = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2]
        got = rows(orc.oracle(toks, n=2, ks=(1, 2)))
        self.assertEqual(got[1]["opportunity"], 4)      # i = 6..9
        self.assertEqual(got[1]["hit"], 0)
        self.assertEqual(got[2]["opportunity"], 4)      # 足切りは前回側だけで判定する
        self.assertEqual(got[2]["hit"], 0)

    def test_repeated_block_at_period_length_never_matches(self):
        # 周期 5 の列 [1,2,3,4,5] x 3・n = 3: 文脈 (1,2,3) の直前の出現は 5 つ前で、
        # 続きは [2,3,4] (今回の続きは [4,5,...]) -> 一致しない。
        # 「続きは周期の最後に来るので、周期 = 文脈長 + 1 だと必ずずれる」の確認。
        toks = ([1, 2, 3, 4, 5] * 3)
        got = rows(orc.oracle(toks, n=3, ks=(1, 2, 3)))
        self.assertEqual(got[1]["opportunity"], 7)       # i = 8..14
        self.assertEqual(got[1]["hit"], 0)
        self.assertEqual(got[1]["p"], 0.0)
        self.assertEqual(got[3]["opportunity"], 7)
        self.assertEqual(got[3]["hit"], 0)

    def test_repeated_block_without_phase_shift_matches(self):
        # 周期 3 の列 [a,b,c] x 4・n = 3: 直前の出現はちょうど 1 ブロック前なので一致。
        # 足切り (前回の続きが k 個そろうか) で k = 2 は 5 機会、k = 3 は 4 機会。
        got = rows(orc.oracle(list("abcabcabcabc"), n=3, ks=(1, 2, 3)))
        self.assertEqual(got[1]["opportunity"], 6)       # i = 6..11
        self.assertEqual(got[1]["hit"], 6)
        self.assertEqual(got[2]["opportunity"], 5)
        self.assertEqual(got[2]["hit"], 5)
        self.assertEqual(got[3]["opportunity"], 4)
        self.assertEqual(got[3]["hit"], 4)

    def test_opportunity_zero_gives_none_not_zero_division(self):
        for toks, n in (([], 12), ([1], 12), ([1, 2, 3], 3), ([1, 2, 3], 0)):
            rep = orc.oracle(toks, n=n, ks=(1, 2))
            for row in rep["k"]:
                self.assertEqual(row["opportunity"], 0, (toks, n, row))
                self.assertIsNone(row["p"], (toks, n, row))
                self.assertEqual(row["hit"], 0)

    def test_k_zero_and_k_above_context_length(self):
        # k = 0 は何も数えない (機会 0)
        rep = orc.oracle([1, 2, 1, 2], n=1, ks=(0,))
        self.assertEqual(rep["k"][0]["opportunity"], 0)
        self.assertIsNone(rep["k"][0]["p"])
        # k > n でも定義どおり: 前回の続きが k 個そろっていれば機会になる
        toks = [1, 2, 3, 4, 1, 2, 3, 4, 1, 5]
        got = rows(orc.oracle(toks, n=2, ks=(3,)))
        self.assertEqual(got[3]["opportunity"], 4)
        self.assertEqual(got[3]["hit"], 0)

    def test_overlapping_context_counts(self):
        # "abababab": 文脈 (a,b) の直前の出現は 2 文字前 (自身の先頭と重なる) で、
        # 続きは一致する。過去に出現した位置は i = 4..7 の 4 つ。
        toks = list("abababab")
        got = rows(orc.oracle(toks, n=2, ks=(1, 2)))
        self.assertEqual(got[1]["opportunity"], 4)
        self.assertEqual(got[1]["hit"], 4)
        self.assertEqual(got[2]["opportunity"], 3)      # 続きが 2 個そろう i = 4..6
        self.assertEqual(got[2]["hit"], 3)

    def test_k_list_is_reported_in_the_order_given(self):
        rep = orc.oracle([1, 2, 1, 2, 1, 2], n=1, ks=(3, 1))
        self.assertEqual([row["k"] for row in rep["k"]], [3, 1])


class TestTokenizers(unittest.TestCase):
    def test_words_ascii_runs(self):
        self.assertEqual(orc.tokenize_words("foo(1) ok_x"),
                         ["foo", "(", "1", ")", " ", "ok_x"])

    def test_words_japanese_class_boundaries(self):
        self.assertEqual(orc.tokenize_words("漢字とかな"), ["漢字", "とかな"])
        self.assertEqual(orc.tokenize_words("関数を呼ぶ"), ["関数", "を", "呼", "ぶ"])
        self.assertEqual(orc.tokenize_words("テスト関数"),
                         ["テスト", "関数"])

    def test_bytes_are_utf8_units(self):
        self.assertEqual(orc.tokenize_bytes("ab"), [97, 98])
        self.assertEqual(len(orc.tokenize_bytes("あ")), 3)
        self.assertEqual(len(orc.tokenize_bytes("漢")), 3)

    def test_tokenizer_registry_names(self):
        self.assertEqual(sorted(orc.TOKENIZERS), ["bytes", "words"])


class TestAggregation(unittest.TestCase):
    def test_merge_sums_hits_and_opportunities(self):
        # 1 ブロック前の続きと一致する形 (n = 3・周期 3 の列を 3 周)
        a = orc.oracle(list("abcabcabc"), n=3, ks=(1,))
        b = orc.oracle(list("defdefdef"), n=3, ks=(1,))
        self.assertEqual(a["k"][0]["hit"], a["k"][0]["opportunity"])
        self.assertEqual(a["k"][0]["opportunity"], 3)
        merged = orc.merge_reports([a, b], 2, (1,))
        self.assertEqual(merged["k"][0]["hit"], a["k"][0]["hit"] + b["k"][0]["hit"])
        self.assertEqual(merged["k"][0]["opportunity"],
                         a["k"][0]["opportunity"] + b["k"][0]["opportunity"])
        self.assertEqual(merged["tokens"], 18)
        self.assertEqual(merged["k"][0]["p"], 1.0)

    def test_documents_are_not_crossed(self):
        # 同じ列を 2 文書に割ると、文書内の再出現だけが機会になる (またいだ文脈は見ない)
        one = orc.run_corpus([("x", "a b a b a b")], "words", n=2, ks=(1,))
        two = orc.run_corpus([("x", "a b a b a b"), ("y", "a b a b a b")],
                             "words", n=2, ks=(1,))
        self.assertEqual(two["k"][0]["opportunity"], 2 * one["k"][0]["opportunity"])
        self.assertEqual(two["k"][0]["hit"], 2 * one["k"][0]["hit"])

    def test_empty_document_list(self):
        rep = orc.run_corpus([], "words", n=12, ks=(1, 2))
        self.assertEqual(rep["tokens"], 0)
        self.assertEqual(rep["documents"], 0)
        self.assertIsNone(rep["k"][0]["p"])


class TestSyntheticCorpus(unittest.TestCase):
    def test_deterministic_for_a_fixed_seed(self):
        a = orc.synth_corpus()
        b = orc.synth_corpus()
        self.assertEqual(a, b)
        self.assertTrue(a.startswith("### step 0"))
        # 同じシード・違うシードは違う (rng が本当に効いている)
        self.assertNotEqual(orc.synth_corpus(seed=1), orc.synth_corpus(seed=2))

    def test_contains_the_repetition_it_is_meant_to_contain(self):
        text = orc.synth_corpus(n_calls=20)
        self.assertIn("<tool_call>", text)
        for name in ("read_file", "write_file", "search_files"):
            self.assertIn('"name": "%s"' % name, text)
        self.assertGreaterEqual(text.count("use std::collections::HashMap;"), 10)
        self.assertGreaterEqual(text.count('"path": "crates/denno-engine/src/cuda.rs"'), 1)

    def test_measured_report_is_reproducible(self):
        a = orc.run_corpus([("synth", orc.synth_corpus())], "words", n=12, ks=(1,))
        b = orc.run_corpus([("synth", orc.synth_corpus())], "words", n=12, ks=(1,))
        self.assertEqual(a, b)
        self.assertGreater(a["tokens"], 0)
        self.assertGreater(a["k"][0]["opportunity"], 0)


class TestProxyCorpus(unittest.TestCase):
    def test_loads_the_three_groups(self):
        docs, skipped = orc.load_docs_proxy()
        names = [name for name, _ in docs]
        self.assertEqual(skipped, [])
        self.assertTrue(any(n.startswith("docs/") and n.endswith(".ja.md") for n in names))
        self.assertTrue(any(n.startswith("docs/") and n.endswith(".en.md") for n in names))
        self.assertTrue(any(n.startswith("crates/") and n.endswith(".rs") for n in names))
        # 決定的な順序: docs 群の後に crates 群、それぞれソート済み
        docs_part = [n for n in names if n.startswith("docs/")]
        crates_part = [n for n in names if n.startswith("crates/")]
        self.assertEqual(docs_part, sorted(docs_part))
        self.assertEqual(crates_part, sorted(crates_part))
        self.assertEqual(names, docs_part + crates_part)
        self.assertNotIn("docs/SPEC_DECODE_DESIGN.md", names)  # .ja.md/.en.md のみ

    def test_walk_is_sorted_and_filters_suffixes(self):
        files = orc._walk([os.path.join(orc.REPO, "crates")], suffixes=(".rs",))
        self.assertTrue(files)
        self.assertEqual(files, sorted(files))
        self.assertTrue(all(f.endswith(".rs") for f in files))


class TestCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def write(self, name, text, encoding="utf-8"):
        path = os.path.join(self.tmp.name, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding=encoding) as f:
            f.write(text)
        return path

    def test_self_test_exits_zero(self):
        r = run(["--self-test"])
        self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-2000:])
        self.assertIn("self-test: ok", r.stdout)

    def test_help_lists_the_contract(self):
        r = run(["--help"])
        self.assertEqual(r.returncode, 0)
        for word in ("--corpus", "--path", "--tokenizer", "--n", "--k",
                     "--json", "--self-test"):
            self.assertIn(word, r.stdout)

    def test_json_reports_both_tokenizers_with_notes(self):
        path = self.write("a.txt", "alpha beta alpha beta alpha beta\n")
        r = run(["--path", path, "--n", "2", "--json"])
        self.assertEqual(r.returncode, 0, r.stderr)
        doc = json.loads(r.stdout)
        self.assertEqual(doc["tokenizers"], ["words", "bytes"])
        self.assertEqual(doc["n"], 2)
        self.assertEqual(doc["k"], [1, 2, 3, 4])
        self.assertEqual(len(doc["reports"]), 2)
        for rep in doc["reports"]:
            self.assertIn("NOT the model tokenizer",
                          doc["tokenizer_notes"][rep["tokenizer"]])
            self.assertGreater(rep["k"][0]["opportunity"], 0)
        self.assertTrue(any("not the real workload" in n for n in doc["notes"]))

    def test_text_output_has_one_row_per_k(self):
        path = self.write("a.txt", "a b a b a b a b\n")
        r = run(["--path", path, "--tokenizer", "words", "--n", "2", "--k", "1,3"])
        self.assertEqual(r.returncode, 0, r.stderr)
        body = [ln for ln in r.stdout.splitlines() if ln[:1].isdigit()]
        self.assertEqual(len(body), 2)
        self.assertTrue(body[0].startswith("1 "))
        self.assertTrue(body[1].startswith("3 "))

    def test_non_utf8_file_is_skipped_not_fatal(self):
        good = self.write("good.txt", "x y x y x y\n")
        bad = os.path.join(self.tmp.name, "bad.bin")
        with open(bad, "wb") as f:
            f.write(b"ok\xff\xfe\x00bad\n")
        r = run(["--path", self.tmp.name, "--tokenizer", "words", "--json"])
        self.assertEqual(r.returncode, 0, r.stderr)
        doc = json.loads(r.stdout)
        self.assertEqual(len(doc["skipped"]), 1)
        self.assertIn("bad.bin", doc["skipped"][0]["name"])
        self.assertIn("UTF-8", doc["skipped"][0]["reason"])
        self.assertEqual(doc["reports"][0]["documents"], 1)  # good.txt だけ

        # テキスト出力でも落ちず、skipped の行が出る
        r = run(["--path", bad, "--tokenizer", "words"])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("skipped:", r.stdout)
        self.assertIn("UTF-8", r.stdout)

    def test_missing_path_and_bad_args_exit_two(self):
        for args in (["--path", os.path.join(self.tmp.name, "nope.txt")],
                     ["--corpus", "nonsense"],
                     ["--path", self.tmp.name, "--k", "0"],
                     ["--path", self.tmp.name, "--k", "x"],
                     ["--path", self.tmp.name, "--n", "-1"]):
            r = run(args)
            self.assertEqual(r.returncode, 2, (args, r.stdout, r.stderr))

    def test_no_source_exits_two(self):
        r = run([])
        self.assertEqual(r.returncode, 2)

    def test_synth_corpus_is_flagged_as_synthetic(self):
        r = run(["--corpus", "synth", "--tokenizer", "words", "--json"])
        self.assertEqual(r.returncode, 0, r.stderr)
        doc = json.loads(r.stdout)
        self.assertTrue(any("synthetic" in n for n in doc["notes"]))
        self.assertTrue(any("NOT a log of real prompts" in n for n in doc["notes"]))

    def test_output_is_byte_identical_across_runs(self):
        r1 = run(["--corpus", "synth", "--json"])
        r2 = run(["--corpus", "synth", "--json"])
        self.assertEqual(r1.returncode, 0, r1.stderr)
        self.assertEqual(r1.stdout, r2.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)