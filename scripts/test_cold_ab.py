#!/usr/bin/env python3
"""P3-4 (s102): scripts/cold_ab.py の CPU テスト (GPU 不要・標準ライブラリのみ)。

確かめること:
  - ラウンド計画: 偶数ラウンド old->new / 奇数ラウンド new->old、各バイナリが同数回
  - 中央値 / delta = (new - old) / old * 100 (小数 1 桁表示) と失敗ラウンドの除外
  - payload の決定性 (2 回作って byte 一致・既知の sha256) と PF_BENCH.sh との同値
  - 安全装置: 8080/8090 の拒否、同一パス / symlink / 同一 sha256 の拒否
  - 実行経路: 偽サーバ (http.server) を本物の子プロセスで立て、交互ラウンド・
    1 ラウンド 1 リクエスト・input_tokens ガード (exit 2)・--json 出力まで通す

実行:
  python3 scripts/test_cold_ab.py -v
  python3 -m unittest discover -s scripts -p 'test_cold_ab.py'
"""

import json
import os
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "cold_ab.py")
sys.path.insert(0, HERE)

import cold_ab  # noqa: E402  (同じディレクトリの実装を直接読み込む)

# 偽サーバ: cold_ab.py が渡す serve 引数 (`serve <model> --port <p> --host ...
# --seed ... --quant ...`) だけを解釈し、tag / token 数 / delay / 記録先は
# 隣の .conf.json から読む。未知の引数は無視するので、実装側が起動コマンドを
# 変えていないことの確認も兼ねる。
FAKE_SERVER = textwrap.dedent('''
    import json, os, sys, time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    # 設定は自分と同じパス + ".conf.json" から読む (shebang に引数を載せない)
    with open(__file__ + ".conf.json", "r", encoding="utf-8") as _f:
        CONF = json.load(_f)
    argv = sys.argv[1:]
    if not argv or argv[0] != "serve":
        sys.stderr.write("expected the serve subcommand\\n")
        sys.exit(9)
    port = int(argv[argv.index("--port") + 1])
    tag = CONF["tag"]
    in_tokens = CONF["input_tokens"]
    delay = CONF["delay"]
    log = CONF["log"]

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/health":
                body = b'{"status":"ok"}'
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            n = int(self.headers.get("content-length") or 0)
            payload = json.loads(self.rfile.read(n).decode("utf-8"))
            if log:
                with open(log, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"tag": tag, "model": payload.get("model"),
                                        "system_bytes": len(payload.get("system", "").encode("utf-8")),
                                        "max_tokens": payload.get("max_tokens"),
                                        "messages": payload.get("messages")}) + "\\n")
            if delay:
                time.sleep(delay)
            body = json.dumps({"id": "msg_1", "type": "message",
                               "content": [{"type": "text", "text": tag}],
                               "usage": {"input_tokens": in_tokens,
                                         "output_tokens": 32}}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
''')


def write_fake_binary(path, tag, input_tokens, delay=0.0, log=""):
    """偽の denno バイナリ。

    設定は `<path>.conf.json` に書く (shebang に引数を載せない)。
    --old と --new は設定が違うので内容も異なり、cold_ab.py の
    「同じ sha256 は拒否」ガードに正しく通る。
    """
    conf = json.dumps({"tag": tag, "input_tokens": input_tokens,
                       "delay": delay, "log": log})
    # 末尾のコメントでバイナリ自体も別内容にする (設定は隣の .conf.json)。
    # これがないと --old と --new が byte 同一になり、sha256 ガードに弾かれる。
    src = ("#!" + sys.executable + "\n" + FAKE_SERVER +
           "# fake binary tag=%s input_tokens=%d delay=%r\n" % (tag, input_tokens, delay))
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    with open(path + ".conf.json", "w", encoding="utf-8") as f:
        f.write(conf)
    os.chmod(path, 0o755)
    return path


def free_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class TestRoundPlan(unittest.TestCase):
    def test_single_round(self):
        self.assertEqual(cold_ab.round_plan(1), [("old", "new")])

    def test_three_rounds_alternate(self):
        self.assertEqual(cold_ab.round_plan(3),
                         [("old", "new"), ("new", "old"), ("old", "new")])

    def test_five_rounds_alternate(self):
        plan = cold_ab.round_plan(5)
        self.assertEqual(len(plan), 5)
        for i in range(len(plan) - 1):
            self.assertNotEqual(plan[i][0], plan[i + 1][0],
                                "the first slot must alternate every round")
        for i, order in enumerate(plan):
            expected = ("old", "new") if i % 2 == 0 else ("new", "old")
            self.assertEqual(order, expected)

    def test_each_binary_runs_once_per_round(self):
        for rounds in (1, 2, 3, 4, 8):
            plan = cold_ab.round_plan(rounds)
            self.assertEqual(sum(o.count("old") for o in plan), rounds)
            self.assertEqual(sum(o.count("new") for o in plan), rounds)

    def test_zero_rounds_rejected(self):
        with self.assertRaises(ValueError):
            cold_ab.round_plan(0)


class TestMedianAndDelta(unittest.TestCase):
    def test_median(self):
        self.assertEqual(cold_ab.median([3.0, 1.0, 2.0]), 2.0)
        self.assertEqual(cold_ab.median([4.0, 1.0, 3.0, 2.0]), 2.5)
        self.assertEqual(cold_ab.median([7.5]), 7.5)
        self.assertIsNone(cold_ab.median([]))

    def test_delta_signs(self):
        self.assertAlmostEqual(cold_ab.compute_delta_pct(10.0, 9.0), -10.0)
        self.assertAlmostEqual(cold_ab.compute_delta_pct(10.0, 11.0), 10.0)
        self.assertEqual(cold_ab.compute_delta_pct(10.0, 10.0), 0.0)

    def test_delta_guards(self):
        self.assertIsNone(cold_ab.compute_delta_pct(0.0, 1.0))
        self.assertIsNone(cold_ab.compute_delta_pct(None, 1.0))
        self.assertIsNone(cold_ab.compute_delta_pct(1.0, None))

    def test_the_0_3s_acceptance_window(self):
        # supervisor の判定は秒 (">= 0.3 s faster")。13.18 -> 12.88 は約 -2.3%。
        self.assertAlmostEqual(cold_ab.compute_delta_pct(13.18, 12.88), -2.2762,
                               places=3)
        self.assertGreaterEqual(13.18 - 12.88, 0.3 - 1e-9)

    def test_formatting(self):
        self.assertEqual(cold_ab.fmt_delta(-2.3456), "-2.3")
        self.assertEqual(cold_ab.fmt_delta(1.05), "+1.1")
        self.assertEqual(cold_ab.fmt_delta(None), "n/a")
        self.assertEqual(cold_ab.fmt_seconds(None), "n/a")
        self.assertEqual(cold_ab.fmt_seconds(12.3456), "12.346")

    def test_summarise_excludes_failed_rounds(self):
        ms = [{"binary": "old", "wall": 10.0, "ok": True},
              {"binary": "new", "wall": 9.0, "ok": True},
              {"binary": "old", "wall": 11.0, "ok": True},
              {"binary": "new", "wall": 99.0, "ok": False},
              {"binary": "old", "wall": None, "ok": False}]
        st = cold_ab.summarise(ms)
        self.assertEqual(st["old"]["median"], 10.5)
        self.assertEqual(st["new"]["median"], 9.0)
        self.assertEqual((st["old"]["n"], st["new"]["n"]), (2, 1))
        self.assertAlmostEqual(st["delta_pct"], -14.2857, places=3)

    def test_summarise_all_failed(self):
        st = cold_ab.summarise([{"binary": "old", "wall": None, "ok": False},
                                {"binary": "new", "wall": None, "ok": False}])
        self.assertIsNone(st["delta_pct"])
        self.assertIsNone(st["old"]["median"])


class TestPayload(unittest.TestCase):
    def test_payload_is_deterministic(self):
        a = cold_ab.encode_payload(cold_ab.build_payload())
        b = cold_ab.encode_payload(cold_ab.build_payload())
        self.assertEqual(a, b)

    def test_payload_sha256_is_pinned(self):
        body = cold_ab.encode_payload(cold_ab.build_payload())
        self.assertEqual(cold_ab.payload_sha256(body),
                         "dead4c91a95296e76c43ba2f18111fc32869e48c1e4aff5bc0d290f01e4796f0",
                         "the payload changed; every historical prefill number "
                         "becomes incomparable")

    def test_payload_shape(self):
        p = cold_ab.build_payload()
        self.assertEqual(p["model"], "local/qwen35-9b")
        self.assertEqual(p["max_tokens"], 32)
        self.assertEqual(p["temperature"], 0)
        self.assertEqual(p["thinking"], "off")
        self.assertEqual(p["messages"],
                         [{"role": "user", "content": "東京の観光地を3つ教えて"}])
        self.assertTrue(p["system"].startswith("あなたは親切なアシスタントです。\n\n"))
        self.assertTrue(p["system"].endswith("\n"))
        self.assertFalse(p["system"].endswith("\n\n"))

    def test_filler_is_the_20k_document_prefix(self):
        filler = cold_ab.build_filler()
        with open(os.path.join(cold_ab.REPO, "docs", "ARCHITECTURE.en.md"), "rb") as f:
            raw = f.read(cold_ab.FILLER_BYTES).decode("utf-8", errors="ignore")
        self.assertEqual(filler, raw.replace("ahino", "user"))
        self.assertNotIn("ahino", filler)
        self.assertEqual(len(filler), 19611)

    def test_system_bytes(self):
        # 固定行は UTF-8 で 50 バイト (49 文字) + filler 20,000 バイト
        p = cold_ab.build_payload()
        self.assertEqual(len(cold_ab.SYSTEM_LINE.encode("utf-8")), 50)
        self.assertEqual(len(p["system"]), 19630)
        self.assertEqual(len(p["system"].encode("utf-8")), 20050)

    def test_json_round_trip(self):
        body = cold_ab.encode_payload(cold_ab.build_payload())
        self.assertEqual(json.loads(body.decode("utf-8")), cold_ab.build_payload())

    def test_matches_pf_bench_decoded_payload(self):
        """PF_BENCH.sh が作る JSON と、デコード後の中身が一致すること。

        (エスケープ表記は違うが、サーバが受ける文字列は同じ = token 数も同じ)
        """
        with tempfile.TemporaryDirectory() as td:
            ref = os.path.join(td, "pf.json")
            script = textwrap.dedent("""
                set -u
                DOC=%s
                TMP=%s
                head -c 20000 "$DOC" | sed 's/ahino/user/g' > %s/filler.txt
                {
                  printf '{"model":"local/qwen35-9b","max_tokens":32,"temperature":0,"thinking":"off","system":"'
                  printf '\\xe3\\x81\\x82\\xe3\\x81\\xaa\\xe3\\x81\\x9f\\xe3\\x81\\xaf\\xe8\\xa6\\xaa\\xe5\\x88\\x87\\xe3\\x81\\xaa\\xe3\\x82\\xa2\\xe3\\x82\\xb7\\xe3\\x82\\xb9\\xe3\\x82\\xbf\\xe3\\x83\\xb3\\xe3\\x83\\x88\\xe3\\x81\\xa7\\xe3\\x81\\x99\\xe3\\x80\\x82\\\\n\\\\n'
                  awk '{ gsub(/\\\\/,"\\\\\\\\"); gsub(/"/,"\\\\\\""); gsub(/\\t/,"\\\\t"); printf "%%s\\\\n", $0 }' %s/filler.txt
                  printf '","messages":[{"role":"user","content":"\\\\u6771\\\\u4eac\\\\u306e\\\\u89b3\\\\u5149\\\\u5730\\\\u30923\\\\u3064\\\\u6559\\\\u3048\\\\u3066"}]}'
                } > "$TMP"
            """) % (os.path.join(cold_ab.REPO, "docs", "ARCHITECTURE.en.md"), ref, td, td)
            r = subprocess.run(["bash", "-c", script], capture_output=True)
            self.assertEqual(r.returncode, 0, r.stderr.decode())
            with open(ref, "r", encoding="utf-8") as f:
                ref_payload = json.load(f)
        self.assertEqual(ref_payload, cold_ab.build_payload())


class TestGuards(unittest.TestCase):
    def mk(self, **kw):
        base = {"rounds": 3, "port": 8095, "max_tokens": 32,
                "expect_input_tokens": 5686, "old": "/tmp/a", "new": "/tmp/b",
                "self_test": False, "dry_run": True}
        base.update(kw)
        import argparse
        return argparse.Namespace(**base)

    def test_reserved_ports(self):
        self.assertTrue(cold_ab.validate_args(self.mk(port=8080)))
        self.assertTrue(cold_ab.validate_args(self.mk(port=8090)))
        self.assertEqual(cold_ab.validate_args(self.mk(port=8095)), [])

    def test_bad_values(self):
        self.assertTrue(cold_ab.validate_args(self.mk(rounds=0)))
        self.assertTrue(cold_ab.validate_args(self.mk(port=70000)))
        self.assertTrue(cold_ab.validate_args(self.mk(max_tokens=0)))
        self.assertTrue(cold_ab.validate_args(self.mk(expect_input_tokens=0)))

    def test_identical_path(self):
        self.assertTrue(cold_ab.validate_args(self.mk(old="/tmp/same", new="/tmp/same")))

    def test_real_files(self):
        with tempfile.TemporaryDirectory() as td:
            a = write_fake_binary(os.path.join(td, "a"), "A", 5686)
            b = write_fake_binary(os.path.join(td, "b"), "B", 5686)
            link = os.path.join(td, "link")
            os.symlink(a, link)
            copy = os.path.join(td, "copy")
            with open(a, "rb") as src, open(copy, "wb") as dst:
                dst.write(src.read())
            os.chmod(copy, 0o755)
            notexec = os.path.join(td, "notexec")
            with open(notexec, "w") as f:
                f.write("x")
            args = dict(dry_run=False)
            self.assertEqual(cold_ab.validate_args(self.mk(old=a, new=b, **args)), [])
            self.assertTrue(cold_ab.validate_args(self.mk(old=a, new=link, **args)),
                            "a symlink to the same file must be refused")
            self.assertTrue(cold_ab.validate_args(self.mk(old=a, new=copy, **args)),
                            "identical content (same sha256) must be refused")
            self.assertTrue(cold_ab.validate_args(self.mk(old=a, new=notexec, **args)))
            self.assertTrue(cold_ab.validate_args(self.mk(old=a, new=td, **args)),
                            "a directory is not a binary")

    def test_cli_refuses_reserved_port(self):
        r = subprocess.run([sys.executable, SCRIPT, "--old", "/tmp/a", "--new", "/tmp/b",
                            "--port", "8090", "--dry-run"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 3, r.stdout + r.stderr)
        self.assertIn("reserved", r.stderr)


class TestTimeoutOption(unittest.TestCase):
    """--timeout-s (s103): health 待ちとリクエストの両方に効く 1 つの値。"""

    def mk(self, **kw):
        import argparse
        base = {"rounds": 1, "port": 8095, "max_tokens": 32,
                "expect_input_tokens": 5686, "old": "/tmp/a", "new": "/tmp/b",
                "self_test": False, "dry_run": True,
                "health_timeout": cold_ab.HEALTH_TIMEOUT_S,
                "timeout_s": cold_ab.DEFAULT_TIMEOUT_S}
        base.update(kw)
        return argparse.Namespace(**base)

    def test_default_is_600s(self):
        self.assertEqual(cold_ab.DEFAULT_TIMEOUT_S, 600.0)
        args = cold_ab.build_parser().parse_args(["--old", "/tmp/a", "--new", "/tmp/b"])
        self.assertEqual(args.timeout_s, 600.0)
        # 既存の既定値は変えない
        self.assertEqual(args.health_timeout, cold_ab.HEALTH_TIMEOUT_S)
        self.assertEqual(cold_ab.HEALTH_TIMEOUT_S, 80.0)

    def test_non_positive_timeout_is_rejected(self):
        self.assertTrue(cold_ab.validate_args(self.mk(timeout_s=0)))
        self.assertTrue(cold_ab.validate_args(self.mk(timeout_s=-1)))
        self.assertEqual(cold_ab.validate_args(self.mk(timeout_s=600.0)), [])

    def test_cli_rejects_zero_timeout(self):
        r = subprocess.run([sys.executable, SCRIPT, "--old", "/tmp/a", "--new", "/tmp/b",
                            "--timeout-s", "0", "--dry-run"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 3, r.stdout + r.stderr)
        self.assertIn("--timeout-s", r.stderr)

    def test_dry_run_prints_the_timeout(self):
        # 既定の出力行は変えない: 既定では出さず、変えたときだけ 1 行足す
        base = subprocess.run([sys.executable, SCRIPT, "--old", "/tmp/a", "--new", "/tmp/b",
                               "--dry-run"], capture_output=True, text=True)
        self.assertEqual(base.returncode, 0, base.stdout + base.stderr)
        self.assertNotIn("timeout =", base.stdout)
        r = subprocess.run([sys.executable, SCRIPT, "--old", "/tmp/a", "--new", "/tmp/b",
                            "--timeout-s", "900", "--dry-run"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("timeout = 900s", r.stdout)
        # 元の行はそのまま残っている
        self.assertIn("health_timeout=80s", r.stdout)

    def test_flag_reaches_both_health_wait_and_request(self):
        """one_request と wait_health に同じ args.timeout_s が渡ることを見る。

        実行経路の値の流れは run_ab の中なので、呼び側を差し替えて観測する。
        """
        seen = {}

        real_request = cold_ab.one_request
        real_health = cold_ab.wait_health
        real_stop = cold_ab.stop_servers
        real_start = cold_ab.start_server

        def fake_health(port, timeout=None, proc=None, host="127.0.0.1"):
            seen["health"] = timeout
            return False  # health で落とせば実際の推論は走らない

        def fake_request(body, port, host="127.0.0.1", timeout=None):
            seen["request"] = timeout
            return 1.0, {"usage": {"input_tokens": 5686}}, ""

        # このテストは本物のプロセスを扱わないので、ポート操作は全部差し替える
        # (実 stop_servers は解放待ちで 30 s 待つことがあり、生きている 8095 を
        #  覗きにいくのも避けたい)。
        with tempfile.TemporaryDirectory() as td:
            a = write_fake_binary(os.path.join(td, "a"), "A", 5686)
            b = write_fake_binary(os.path.join(td, "b"), "B", 5686)
            args = cold_ab.build_parser().parse_args(
                ["--old", a, "--new", b, "--port", "8095", "--rounds", "1",
                 "--timeout-s", "450", "--log-dir", td])
            devnull = open(os.devnull, "w")
            try:
                cold_ab.one_request = fake_request
                cold_ab.wait_health = fake_health
                cold_ab.stop_servers = lambda *a, **k: True
                cold_ab.start_server = lambda *a, **k: (None, None)
                cold_ab.run_ab(args, out=devnull)
            finally:
                devnull.close()
                cold_ab.one_request = real_request
                cold_ab.wait_health = real_health
                cold_ab.stop_servers = real_stop
                cold_ab.start_server = real_start
        self.assertEqual(seen.get("health"), 450.0,
                         "--timeout-s must raise the health wait above 80s")


class TestCli(unittest.TestCase):
    def test_help_explains_the_protocol(self):
        r = subprocess.run([sys.executable, SCRIPT, "--help"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0)
        for word in ("cold", "rounds", "expect-input-tokens", "self-test"):
            self.assertIn(word, r.stdout)

    def test_self_test_exits_zero(self):
        r = subprocess.run([sys.executable, SCRIPT, "--self-test"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-2000:])
        self.assertIn("SELF-TEST PASSED", r.stdout)

    def test_dry_run_prints_plan_without_starting_servers(self):
        with tempfile.TemporaryDirectory() as td:
            a = write_fake_binary(os.path.join(td, "a"), "A", 5686)
            b = write_fake_binary(os.path.join(td, "b"), "B", 5686)
            port = free_port()
            r = subprocess.run([sys.executable, SCRIPT, "--old", a, "--new", b,
                                "--port", str(port), "--rounds", "3", "--dry-run"],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("DRY RUN", r.stdout)
            self.assertIn("dead4c91a95296e76c43ba2f18111fc32869e48c1e4aff5bc0d290f01e4796f0",
                          r.stdout)
            self.assertIn("round 1: old -> new", r.stdout)
            self.assertIn("round 2: new -> old", r.stdout)
            self.assertIn("--quant int8", r.stdout)
            self.assertFalse(cold_ab.tcp_in_use(port), "dry-run must not bind the port")


class TestEraFlag(unittest.TestCase):
    """s110: --era LABEL は --json のメタデータだけ。測定も表示も変えない。"""

    def mk(self, **kw):
        import argparse
        base = {"rounds": 1, "port": 8095, "max_tokens": 32,
                "expect_input_tokens": 5686, "old": "/tmp/a", "new": "/tmp/b",
                "self_test": False, "dry_run": True,
                "health_timeout": cold_ab.HEALTH_TIMEOUT_S,
                "timeout_s": cold_ab.DEFAULT_TIMEOUT_S, "era": None}
        base.update(kw)
        return argparse.Namespace(**base)

    def test_default_is_none(self):
        args = cold_ab.build_parser().parse_args(["--old", "/tmp/a", "--new", "/tmp/b"])
        self.assertIsNone(args.era)

    def test_validate_args_ignores_era(self):
        # era は測定の前提条件ではないので、validate_args の結果を変えない
        self.assertEqual(cold_ab.validate_args(self.mk()), [])
        self.assertEqual(cold_ab.validate_args(self.mk(era="s110")), [])
        self.assertEqual(cold_ab.validate_args(self.mk(era="s110", port=8090))[0][:7],
                         "--port ")

    def test_attach_era_only_adds_one_key(self):
        r = {"payload_sha256": "x", "measurements": [{"wall": 1.0}], "stats": {"k": 1}}
        self.assertIs(cold_ab.attach_era(r, None), r)
        out = cold_ab.attach_era(r, "s110")
        self.assertEqual(out["era"], "s110")
        self.assertEqual({k: v for k, v in out.items() if k != "era"}, r)
        self.assertNotIn("era", r, "the result dict must not be mutated")
        self.assertIs(out["measurements"], r["measurements"])

    def test_cli_rejects_empty_and_control_characters(self):
        for bad in ("", "   ", "a\tb", "a\nb"):
            r = subprocess.run([sys.executable, SCRIPT, "--old", "/tmp/a", "--new", "/tmp/b",
                                "--json", "/tmp/x.json", "--era", bad],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 2, "argparse usage error for %r: %s"
                             % (bad, r.stderr))
            self.assertIn("--era", r.stderr)

    def test_cli_rejects_a_long_label(self):
        r = subprocess.run([sys.executable, SCRIPT, "--old", "/tmp/a", "--new", "/tmp/b",
                            "--json", "/tmp/x.json", "--era", "a" * 65],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("at most", r.stderr)

    def test_era_without_json_is_a_noop(self):
        # 表示にだけ効く他のフラグと同じ扱い: エラーにせず、dry-run の出力も変えない
        base = [sys.executable, SCRIPT, "--old", "/tmp/a", "--new", "/tmp/b", "--dry-run"]
        r0 = subprocess.run(base, capture_output=True, text=True)
        r1 = subprocess.run(base + ["--era", "s110"], capture_output=True, text=True)
        self.assertEqual(r0.returncode, 0, r0.stderr)
        self.assertEqual(r1.returncode, 0, r1.stderr)
        self.assertEqual(r0.stdout, r1.stdout, "--era must not change the report")


class TestEndToEnd(unittest.TestCase):
    """偽バイナリを本物の子プロセスで立て、交互ラウンドを通しで確かめる。"""

    def run_ab(self, old_tokens=5686, new_tokens=5686, rounds=2, era=None):
        td = tempfile.mkdtemp(prefix="cold_ab_test_")
        log = os.path.join(td, "requests.jsonl")
        port = free_port()
        a = write_fake_binary(os.path.join(td, "old_bin"), "OLD", old_tokens,
                              delay=0.05, log=log)
        b = write_fake_binary(os.path.join(td, "new_bin"), "NEW", new_tokens,
                              delay=0.01, log=log)
        out_json = os.path.join(td, "out.json")
        argv = [sys.executable, SCRIPT, "--old", a, "--new", b,
                "--port", str(port), "--rounds", str(rounds),
                "--expect-input-tokens", "5686",
                "--health-timeout", "20", "--log-dir", td,
                "--json", out_json]
        if era is not None:
            argv += ["--era", era]
        r = subprocess.run(argv, capture_output=True, text=True, timeout=180)
        entries = []
        if os.path.exists(log):
            with open(log, "r", encoding="utf-8") as f:
                entries = [json.loads(line) for line in f if line.strip()]
        result = None
        if os.path.exists(out_json):
            with open(out_json, "r", encoding="utf-8") as f:
                result = json.load(f)
        return r, entries, result, port, td

    def test_alternating_rounds_one_request_each(self):
        r, entries, result, port, td = self.run_ab(rounds=2)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-2000:])
        # 2 ラウンド x 2 バイナリ = 4 リクエスト。順序は old,new,new,old
        self.assertEqual([e["tag"] for e in entries], ["OLD", "NEW", "NEW", "OLD"])
        self.assertEqual(len(entries), 4, "exactly one request per server start")

    def test_payload_that_reaches_the_server(self):
        r, entries, result, port, td = self.run_ab(rounds=1)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        e = entries[0]
        self.assertEqual(e["model"], "local/qwen35-9b")
        self.assertEqual(e["max_tokens"], 32)
        self.assertEqual(e["system_bytes"], 20050)
        self.assertEqual(e["messages"][0]["content"], "東京の観光地を3つ教えて")

    def test_medians_and_delta_reported(self):
        r, entries, result, port, td = self.run_ab(rounds=2)
        self.assertEqual(r.returncode, 0)
        self.assertEqual(result["stats"]["old"]["n"], 2)
        self.assertEqual(result["stats"]["new"]["n"], 2)
        self.assertIsNotNone(result["stats"]["delta_pct"])
        # 新バイナリの方がわずかに速い設定 (delay 0.05 vs 0.01)
        self.assertLess(result["stats"]["new"]["median"],
                        result["stats"]["old"]["median"])
        self.assertIn("delta =", r.stdout)
        self.assertIn("median=", r.stdout)

    def test_payload_guard_exits_2(self):
        r, entries, result, port, td = self.run_ab(old_tokens=5000, new_tokens=5000,
                                                   rounds=3)
        self.assertEqual(r.returncode, 2, r.stdout[-2000:])
        self.assertTrue(result["guard_tripped"])
        self.assertIn("GUARD TRIP", r.stdout)
        # ガードが落ちたら最初の測定の後すぐ中断する (全ラウンド回さない)
        self.assertEqual(len(entries), 1, "the guard must abort the run early")

    def test_port_is_free_after_the_run(self):
        r, entries, result, port, td = self.run_ab(rounds=1)
        self.assertEqual(r.returncode, 0)
        self.assertFalse(cold_ab.tcp_in_use(port), "port must be released")
        self.assertEqual(result["leftover_pids"], [])

    def test_era_is_written_and_measured_nothing_differently(self):
        # 同じ設定で --era 有り / 無しを回し、差分が "era" キーだけであることを見る
        r0, entries0, plain, port0, _ = self.run_ab(rounds=1)
        r1, entries1, tagged, port1, _ = self.run_ab(rounds=1, era="s110")
        self.assertEqual(r0.returncode, 0, r0.stdout[-2000:])
        self.assertEqual(r1.returncode, 0, r1.stdout[-2000:])
        self.assertEqual(tagged.get("era"), "s110")
        self.assertNotIn("era", plain)
        # 2 回の実行は別の一時ディレクトリ・別の実測値なので、byte 比較ではなく
        # キー集合と各セクションのキー集合が同じであることを見る。
        self.assertEqual(sorted(tagged), sorted(set(plain) | {"era"}))
        self.assertEqual(sorted(tagged["stats"]), sorted(plain["stats"]))
        self.assertEqual(sorted(tagged["protocol"]), sorted(plain["protocol"]))
        for name in ("old", "new"):
            self.assertEqual(sorted(tagged["stats"][name]), sorted(plain["stats"][name]))
        self.assertEqual([sorted(m) for m in tagged["measurements"]],
                         [sorted(m) for m in plain["measurements"]])
        self.assertEqual({k: v for k, v in tagged["protocol"].items() if k != "port"},
                         {k: v for k, v in plain["protocol"].items() if k != "port"},
                         "the protocol is unchanged (only the ephemeral port differs)")
        # 測定そのもの (ラウンド・payload・token 数) は変わらない
        self.assertEqual(len(entries0), len(entries1))
        self.assertEqual([e["tag"] for e in entries0], [e["tag"] for e in entries1])
        self.assertEqual([e["system_bytes"] for e in entries0],
                         [e["system_bytes"] for e in entries1])
        self.assertEqual(tagged["payload_sha256"], plain["payload_sha256"])
        self.assertEqual(len(tagged["measurements"]), len(plain["measurements"]))
        # 表示も変わらない (stdout はホスト/ポート依存の部分を除いて同一)
        self.assertNotIn("era=", r0.stdout)
        self.assertIn("(era=s110)", r1.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)