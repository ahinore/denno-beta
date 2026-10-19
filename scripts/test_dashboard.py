#!/usr/bin/env python3
"""P5-1 (s109): scripts/dashboard.py の CPU テスト (GPU 不要・標準ライブラリのみ)。

確かめること:
  1. 同じ入力で 2 回走らせた HTML が byte 一致 (決定性)
  2. prefill と decode の 2 パネルが別々の単位で出る / 縦線は --protocol-change が
     一致したときだけ入る
  3. 点が 1 つのパネルは "not enough points" を出し、SVG に NaN/Inf を書かない
  4. 壊れた JSON は exit 1 で、メッセージにそのファイル名（フルパス）が入る
  5. PF_BENCH ログの行数だけ prefill / decode の点が増える
  6. --help が exit 0

実行:
  python3 scripts/test_dashboard.py -v
  python3 -m unittest discover -s scripts -p 'test_dashboard.py'
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "dashboard.py")
sys.path.insert(0, HERE)

import dashboard  # noqa: E402  (同じディレクトリの実装を直接読み込む)


# ---------------------------------------------------------------------------
# 材料作り
# ---------------------------------------------------------------------------

def cold_ab_payload(era, pairs, delta=None):
    """cold_ab.py --json と同じ形の payload を組む (pairs = [(binary, wall), ...])。

    実装の形は scripts/cold_ab.py の run_ab() が返す dict に合わせる。era は
    cold_ab.py --era が書くフィールド (s110 以降)。era=None なら キー自体を
    付けない (s109 以前の JSON / fallback の経路を再現する)。
    """
    measurements = []
    for binary, wall in pairs:
        measurements.append({"round": 1, "binary": binary, "binary_path": "/nowhere",
                             "wall": wall, "input_tokens": 5686, "output_tokens": 32,
                             "boot_s": 1.0, "ok": True, "error": "", "log": "/tmp/l"})
    runs = {}
    for name in ("old", "new"):
        runs[name] = [w for b, w in pairs if b == name and w is not None]
    stats = {}
    for name in ("old", "new"):
        stats[name] = {"n": len(runs[name]),
                       "median": sorted(runs[name])[len(runs[name]) // 2]
                       if runs[name] else None,
                       "runs": runs[name]}
    if delta is None:
        if stats["old"]["median"] and stats["new"]["median"]:
            delta = ((stats["new"]["median"] - stats["old"]["median"])
                     / stats["old"]["median"] * 100.0)
    stats["delta_pct"] = delta
    payload = {"payload_sha256": "0" * 64, "payload_bytes": 1,
               "measurements": measurements, "stats": stats,
               "guard_tripped": False, "leftover_pids": [], "rounds": 1,
               "binaries": {"old": "/nowhere/old", "new": "/nowhere/new"},
               "protocol": {"model": "local/qwen35-9b", "quant": "int8", "port": 8095,
                            "seed": 42, "max_tokens": 32, "expect_input_tokens": 5686,
                            "source_doc": "docs/ARCHITECTURE.en.md"}}
    if era is not None:
        payload["era"] = era
    return payload


def write_json(directory, name, payload):
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    return path


def run(args, cwd=None):
    return subprocess.run([sys.executable, SCRIPT] + args,
                          capture_output=True, text=True, cwd=cwd)


def svg_polylines(html):
    return re.findall(r'<polyline points="([^"]*)"', html)


def markers(html):
    return re.findall(r'<line[^>]*class="marker"', html)


class DashboardCase(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory(prefix="dash_test_")
        self.td = self._td.name
        self.runs = os.path.join(self.td, "runs")
        os.makedirs(self.runs)
        self.out = os.path.join(self.td, "out.html")

    def tearDown(self):
        self._td.cleanup()

    def make_runs(self):
        """era の境目が分かる 3 つの cold A/B + 3 行の PF ログ。"""
        write_json(self.runs, "a_s104.json",
                   cold_ab_payload("s104", [("old", 13.40), ("new", 13.18)]))
        write_json(self.runs, "b_s107.json",
                   cold_ab_payload("s107", [("old", 12.95), ("new", 12.88)]))
        write_json(self.runs, "c_s109.json",
                   cold_ab_payload("s109", [("old", 12.70), ("new", 12.62)]))
        self.log = os.path.join(self.td, "pf.log")
        with open(self.log, "w", encoding="utf-8") as f:
            f.write("PF wall=13.180s input_tokens=5686 prefill_tok_per_s=431.4\n")
            f.write("PF wall=12.880s input_tokens=5686 prefill_tok_per_s=441.5\n")
            f.write("PF wall=12.610s input_tokens=5686 prefill_tok_per_s=450.9\n")
        return self.runs, self.log


class TestDeterminism(DashboardCase):
    def test_two_runs_are_byte_identical(self):
        runs, log = self.make_runs()
        out2 = os.path.join(self.td, "out2.html")
        args = ["--cold-ab-dir", runs, "--pf-bench-log", log, "--max-tokens", "32",
                "--protocol-change", "s107"]
        r1 = run(args + ["--out", self.out])
        r2 = run(args + ["--out", out2])
        self.assertEqual(r1.returncode, 0, r1.stdout + r1.stderr)
        self.assertEqual(r2.returncode, 0, r2.stdout + r2.stderr)
        with open(self.out, "rb") as f:
            b1 = f.read()
        with open(out2, "rb") as f:
            b2 = f.read()
        self.assertEqual(b1, b2, "the dashboard must be deterministic")
        self.assertGreater(len(b1), 1000)

    def test_file_order_does_not_depend_on_readdir(self):
        # ファイル名順に読むので、作る順を変えても同じ HTML になる
        runs, log = self.make_runs()
        r1 = run(["--cold-ab-dir", runs, "--out", self.out])
        self.assertEqual(r1.returncode, 0, r1.stderr)
        with open(self.out, "rb") as f:
            first = f.read()
        os.remove(os.path.join(runs, "a_s104.json"))
        write_json(runs, "a_s104.json",
                   cold_ab_payload("s104", [("old", 13.40), ("new", 13.18)]))
        r2 = run(["--cold-ab-dir", runs, "--out", self.out])
        self.assertEqual(r2.returncode, 0, r2.stderr)
        with open(self.out, "rb") as f:
            self.assertEqual(f.read(), first)


class TestPanelsAndMarker(DashboardCase):
    def test_two_panels_with_their_own_units(self):
        runs, log = self.make_runs()
        r = run(["--cold-ab-dir", runs, "--pf-bench-log", log, "--max-tokens", "32",
                 "--out", self.out])
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(self.out, "r", encoding="utf-8") as f:
            html = f.read()
        self.assertIn("Prefill (s)", html)
        self.assertIn("Decode (ms/token)", html)
        # 2 パネル = 2 つの svg。1 つの共有軸ではない。
        self.assertEqual(html.count("<svg"), 2)
        self.assertIn("ms/token", html)
        # 凡例 (点の一覧) が両パネルの点を持つ
        self.assertIn("<li>prefill[0]", html)
        self.assertIn("<li>decode[0]", html)

    def test_marker_only_when_protocol_change_matches(self):
        runs, log = self.make_runs()
        with_mark = run(["--cold-ab-dir", runs, "--pf-bench-log", log,
                         "--max-tokens", "32", "--protocol-change", "s107",
                         "--out", self.out])
        self.assertEqual(with_mark.returncode, 0, with_mark.stderr)
        with open(self.out, "r", encoding="utf-8") as f:
            html = f.read()
        # cold A/B の点は prefill パネルにしか無いので、線もそこに 1 本だけ
        # (decode パネルの点は PF 行由来で era を持たない)
        self.assertEqual(len(markers(html)), 1,
                         "one marker per era boundary, in the panel that has the era")
        self.assertIn('class="marklabel">s107<', html)

        no_match = run(["--cold-ab-dir", runs, "--pf-bench-log", log,
                        "--max-tokens", "32", "--protocol-change", "s999",
                        "--out", self.out])
        self.assertEqual(no_match.returncode, 0, no_match.stderr)
        with open(self.out, "r", encoding="utf-8") as f:
            html = f.read()
        self.assertEqual(markers(html), [], "no marker when no era matches")
        self.assertNotIn('class="marklabel"', html)

        none_at_all = run(["--cold-ab-dir", runs, "--out", self.out])
        self.assertEqual(none_at_all.returncode, 0, none_at_all.stderr)
        with open(self.out, "r", encoding="utf-8") as f:
            self.assertEqual(markers(f.read()), [])

    def test_repeatable_protocol_change(self):
        runs = self.runs
        write_json(runs, "a_s104.json",
                   cold_ab_payload("s104", [("old", 13.40), ("new", 13.18)]))
        write_json(runs, "b_s107.json",
                   cold_ab_payload("s107", [("old", 12.95), ("new", 12.88)]))
        write_json(runs, "c_s109.json",
                   cold_ab_payload("s109", [("old", 12.70), ("new", 12.62)]))
        r = run(["--cold-ab-dir", runs, "--protocol-change", "s107",
                 "--protocol-change", "s109", "--out", self.out])
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(self.out, "r", encoding="utf-8") as f:
            html = f.read()
        # prefill パネルに 2 本 + 点が無い decode パネルは線を引かない
        self.assertEqual(len(markers(html)), 2, html[html.find("marker") - 200:][:400])
        self.assertIn("s107", html)
        self.assertIn("s109", html)


class TestSinglePoint(DashboardCase):
    def test_single_point_says_not_enough_and_has_no_nan(self):
        # 1 ラウンドだけの cold A/B (成功 1 件) -> 両パネルとも 1 点
        write_json(self.runs, "only.json",
                   cold_ab_payload("s109", [("old", 12.88)]))
        r = run(["--cold-ab-dir", self.runs, "--out", self.out])
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(self.out, "r", encoding="utf-8") as f:
            html = f.read()
        self.assertIn("not enough points", html)
        self.assertEqual(svg_polylines(html), [])
        low = html.lower()
        self.assertNotIn("nan", low)
        self.assertNotRegex(low, r"[^a-z]inf[^a-z]")

    def test_equal_values_do_not_break_the_scale(self):
        write_json(self.runs, "a.json",
                   cold_ab_payload("s104", [("old", 12.0), ("new", 12.0), ("old", 12.0)]))
        r = run(["--cold-ab-dir", self.runs, "--out", self.out])
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(self.out, "r", encoding="utf-8") as f:
            html = f.read()
        self.assertEqual(len(svg_polylines(html)), 1)
        self.assertNotIn("nan", html.lower())
        for nums in re.findall(r'cx="([^"]*)"', html):
            self.assertNotIn("nan", nums.lower())


class TestBadInput(DashboardCase):
    def test_malformed_json_exits_1_and_names_the_file(self):
        bad = os.path.join(self.runs, "broken_run.json")
        with open(bad, "w", encoding="utf-8") as f:
            f.write('{"stats": {"old": {"median": 1.0}, oops\n')
        r = run(["--cold-ab-dir", self.runs, "--out", self.out])
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("broken_run.json", r.stderr)
        self.assertIn(bad, r.stderr, "the message must name the offending file")
        self.assertFalse(os.path.exists(self.out),
                         "no partial --out on failure")

    def test_payload_without_median_structure_exits_1(self):
        write_json(self.runs, "no_stats.json", {"payload_sha256": "abc"})
        self.out_existing = self.out
        with open(self.out, "w", encoding="utf-8") as f:
            f.write("stale")
        r = run(["--cold-ab-dir", self.runs, "--out", self.out])
        self.assertEqual(r.returncode, 1)
        self.assertIn("no_stats.json", r.stderr)
        self.assertFalse(os.path.exists(self.out),
                         "a stale --out must be removed when the input is rejected")

    def test_missing_dir_exits_1(self):
        r = run(["--cold-ab-dir", os.path.join(self.td, "nope"), "--out", self.out])
        self.assertEqual(r.returncode, 1)
        self.assertIn("nope", r.stderr)

    def test_empty_dir_exits_1(self):
        r = run(["--cold-ab-dir", self.runs, "--out", self.out])
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("no measurements", r.stderr)
        self.assertFalse(os.path.exists(self.out))

    def test_bad_usage_exits_2(self):
        r = run(["--cold-ab-dir", self.runs])   # --out 無し
        self.assertEqual(r.returncode, 2, r.stderr)
        r2 = run(["--out", self.out])           # --cold-ab-dir 無し
        self.assertEqual(r2.returncode, 2, r2.stderr)


class TestPfBenchPoints(DashboardCase):
    def test_every_pf_line_is_one_point(self):
        write_json(self.runs, "a.json",
                   cold_ab_payload("s104", [("old", 13.40), ("new", 13.18)]))
        runs = self.runs
        log = os.path.join(self.td, "pf.log")
        with open(log, "w", encoding="utf-8") as f:
            for i in range(5):
                f.write("PF wall=%d.500s input_tokens=5686 prefill_tok_per_s=100.0\n"
                        % (10 + i))
        r = run(["--cold-ab-dir", runs, "--pf-bench-log", log, "--max-tokens", "32",
                 "--out", self.out])
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(self.out, "r", encoding="utf-8") as f:
            html = f.read()
        pre = re.findall(r"<li>prefill\[\d+\]", html)
        dec = re.findall(r"<li>decode\[\d+\]", html)
        # cold A/B 2 点 + PF 5 点 / decode は PF 5 点
        self.assertEqual(len(pre), 7, pre)
        self.assertEqual(len(dec), 5, dec)
        # 値そのもの (wall = 10.5 .. 14.5 s)
        self.assertIn("prefill[2] = 10.500 s (pf_bench: PF line 1)", html)
        self.assertIn("prefill[6] = 14.500 s (pf_bench: PF line 5)", html)
        # decode = wall * 1000 / 32
        # 10500 ms / 32 tokens = 328.125 -> "%.2f" は 328.12
        self.assertIn("decode[0] = 328.12 ms/token", html)

    def test_noise_lines_are_ignored(self):
        pts = dashboard.parse_pf_bench_log(
            "starting bench\n"
            "PF wall=12.345s input_tokens=5686 prefill_tok_per_s=460.6\n"
            "PF ERROR: no input_tokens (see /tmp/pf_bench_out.json)\n")
        self.assertEqual(len(pts), 1)
        self.assertEqual(pts[0]["wall"], 12.345)

    def test_lines_without_a_token_count_are_not_guessed(self):
        runs = self.runs
        write_json(runs, "a.json", cold_ab_payload("s104", [("old", 13.40)]))
        log = os.path.join(self.td, "pf.log")
        with open(log, "w", encoding="utf-8") as f:
            f.write("PF wall=12.880s input_tokens=5686\n")
        r = run(["--cold-ab-dir", runs, "--pf-bench-log", log, "--out", self.out])
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(self.out, "r", encoding="utf-8") as f:
            html = f.read()
        self.assertEqual(re.findall(r"<li>decode\[\d+\]", html), [])
        self.assertIn("no output_tokens", html)


class TestEraRoundTrip(unittest.TestCase):
    """s110: cold_ab.py --era -> JSON -> dashboard.py の往復を CPU だけで通す。

    cold_ab.py は本物を子プロセスで走らせる (偽サーバは test_cold_ab.py と同じ
    仕組みを借りる)。ネットワークは 127.0.0.1 の使い捨てポートのみ、GPU も
    本物のサーバも使わない。
    """

    def setUp(self):
        import test_cold_ab  # 偽サーバ / 偽バイナリの材料を共有する
        self.helper = test_cold_ab
        self._td = tempfile.TemporaryDirectory(prefix="dash_era_")
        self.td = self._td.name
        self.runs = os.path.join(self.td, "runs")
        os.makedirs(self.runs)

    def tearDown(self):
        self._td.cleanup()

    def cold_ab_run(self, era=None, name="run_s110.json"):
        """cold_ab.py --json を本物の子プロセスで 1 ラウンド回し、JSON を返す。"""
        a = self.helper.write_fake_binary(os.path.join(self.td, "old_bin"), "OLD", 5686,
                                          delay=0.02)
        b = self.helper.write_fake_binary(os.path.join(self.td, "new_bin"), "NEW", 5686,
                                          delay=0.01)
        out_json = os.path.join(self.runs, name)
        argv = [sys.executable, self.helper.SCRIPT, "--old", a, "--new", b,
                "--port", str(self.helper.free_port()), "--rounds", "1",
                "--expect-input-tokens", "5686", "--health-timeout", "20",
                "--log-dir", self.td, "--json", out_json]
        if era is not None:
            argv += ["--era", era]
        r = subprocess.run(argv, capture_output=True, text=True, timeout=180)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-2000:])
        with open(out_json, "r", encoding="utf-8") as f:
            return json.load(f), r.stdout

    def test_era_survives_into_the_dashboard(self):
        payload, _ = self.cold_ab_run(era="s110", name="run_s110.json")
        # cold_ab 側: era が本当に書かれている
        self.assertEqual(payload.get("era"), "s110")
        r = run(["--cold-ab-dir", self.runs, "--out", os.path.join(self.td, "o.html")])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        got = dashboard.load_cold_ab_dir(self.runs)
        self.assertEqual([g["label"] for g in got], ["s110"])
        self.assertEqual([g["era_source"] for g in got], ["era"],
                         "an era written by cold_ab.py must not fall back to the file name")
        # 中央値は JSON の stats と一致する (往復で数字が変わらない)
        self.assertEqual(got[0]["old_median"], payload["stats"]["old"]["median"])
        self.assertEqual(got[0]["new_median"], payload["stats"]["new"]["median"])
        with open(os.path.join(self.td, "o.html"), "r", encoding="utf-8") as f:
            html = f.read()
        self.assertIn("<td>run_s110.json</td><td>s110</td>", html)

    def test_era_works_with_protocol_change(self):
        # 2 つの run はどちらも era=s110 なので、境目は先頭 (index 0) に 1 本だけ。
        # ファイル名は a_s104 / b_s109 だが、era が実在するので名前は使われない。
        self.cold_ab_run(era="s110", name="a_s104.json")
        self.cold_ab_run(era="s110", name="b_s109.json")
        out = os.path.join(self.td, "o.html")
        r = run(["--cold-ab-dir", self.runs, "--protocol-change", "s110", "--out", out])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        with open(out, "r", encoding="utf-8") as f:
            html = f.read()
        # PF ログを渡していないので decode パネルには点が無く、線は prefill にだけ入る
        pre_svg, dec_svg = html.split("</svg>", 1)
        self.assertEqual(len(markers(pre_svg)), 1, "one marker in the prefill panel")
        self.assertEqual(len(markers(dec_svg)), 0, "no points, no marker in decode")
        self.assertIn("protocol change marker at era: s110", html)
        # ファイル名に一致した線ではないこと (名前を era として渡しても引かれない)
        r2 = run(["--cold-ab-dir", self.runs, "--protocol-change", "a_s104",
                  "--out", os.path.join(self.td, "o2.html")])
        self.assertEqual(r2.returncode, 0, r2.stdout + r2.stderr)
        with open(os.path.join(self.td, "o2.html"), "r", encoding="utf-8") as f:
            self.assertEqual(markers(f.read()), [])

    def test_without_era_the_file_name_is_still_the_fallback(self):
        payload, stdout = self.cold_ab_run(era=None, name="run_named_s110.json")
        self.assertNotIn("era", payload)
        self.assertNotIn("(era=", stdout)
        got = dashboard.load_cold_ab_dir(self.runs)
        self.assertEqual([g["label"] for g in got], ["run_named_s110.json"])
        self.assertEqual([g["era_source"] for g in got], ["filename"])

    def test_label_key_still_wins_over_the_file_name(self):
        # s110 以前の JSON のために "label" も読む (precedence は era -> label -> 名前)
        payload = cold_ab_payload(None, [("old", 13.0), ("new", 12.9)])
        payload["label"] = "hand-added"
        write_json(self.runs, "x.json", payload)
        got = dashboard.load_cold_ab_dir(self.runs)
        self.assertEqual(got[0]["label"], "hand-added")
        self.assertEqual(got[0]["era_source"], "label")


class TestCli(unittest.TestCase):
    def test_help_exits_zero(self):
        r = run(["--help"])
        self.assertEqual(r.returncode, 0)
        for word in ("--cold-ab-dir", "--pf-bench-log", "--protocol-change",
                     "--out", "--self-test"):
            self.assertIn(word, r.stdout)

    def test_self_test_exits_zero(self):
        r = run(["--self-test"])
        self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-2000:])
        self.assertIn("SELF-TEST PASSED", r.stdout)


if __name__ == "__main__":
    unittest.main()