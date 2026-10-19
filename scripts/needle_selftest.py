#!/usr/bin/env python3
"""P4-4 (s98): needle_check.py の自己テスト (サーバ不要・標準ライブラリのみ)。

dev コンテナからは 8095 (GPU 機) に到達できないので、判定の正しさはここで示す。
標準ライブラリの http.server で偽サーバを立て、決め打ちの応答を返して
needle_check.py を子プロセスで走らせ、exit code が期待どおりかを確かめる。

確かめること (handoff の (a)-(f)):
  (a) needle をそのまま返す            -> exit 0
  (b) 間違った値 (QX-9999) を返す      -> exit 1
  (c) needle に触れない応答            -> exit 1
  (d) HTTP 500 / 接続不能              -> exit 2 (recall 失敗と混同しない)
  (e) 404 の fallback 経路が使われ、needle がリクエスト本文を素通りする
  (f) fixture から作る payload が決定的 (2 回作って byte 一致)
  (おまけ) --api openai / anthropic の固定、深度、fixture 形、純関数の境界

使い方: python3 scripts/needle_selftest.py [-v]
"""

import importlib.util
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
CHECK = os.path.join(HERE, "needle_check.py")
FIXTURE = os.path.join(HERE, "needle_fixture.json")

with open(FIXTURE, "r", encoding="utf-8") as _f:
    FX = json.load(_f)

NEEDLE = FX["needle"]
ANSWER = FX["expected_answer"]


def free_port():
    """未使用の TCP ポートを取る (bind して即 close)。"""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class FakeServer:
    """決め打ちの応答を返す偽サーバ。

    paths: 応答する経路 (既定は denno serve と同じ /v1/messages のみ。
           これにより auto の chat/completions -> 404 -> messages が必ず踏まれる)
    reply: 200 で返す本文 (needle を含む文字列など)
    status: 強制的に返す HTTP status (500 の模擬)。None なら 200。
    usage_input_tokens: usage の入力 token 数 (実サーバの代役)。
    """

    def __init__(self, reply, paths=("/v1/messages",), status=None,
                 usage_input_tokens=4671, fmt_hint="anthropic"):
        self.reply = reply
        self.paths = tuple(paths)
        self.status = status
        self.usage_input_tokens = usage_input_tokens
        self.fmt_hint = fmt_hint
        self.requests = []          # (path, payload)
        self.raw_bodies = []        # 生の request body (byte 比較用)
        self.lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                # bytes は「わざと JSON でない本文」を返すための経路
                if isinstance(obj, (bytes, bytearray)):
                    body = bytes(obj)
                else:
                    body = json.dumps(obj).encode("utf-8")
                self.send_response(code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                n = int(self.headers.get("content-length") or 0)
                raw = self.rfile.read(n) if n else b"{}"
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except Exception:
                    payload = {}
                with outer.lock:
                    outer.requests.append((self.path, payload))
                    outer.raw_bodies.append(raw)

                if self.path not in outer.paths:
                    # denno serve と同じ: 未実装の経路は 404
                    self._send(404, {"error": {"type": "not_found",
                                               "message": "not found"}})
                    return
                if outer.status is not None:
                    self._send(outer.status, {"error": {"type": "server_error",
                                                        "message": "forced"}})
                    return
                if isinstance(outer.reply, (bytes, bytearray)):
                    # 200 だが本文が JSON でない (解釈失敗の模擬)
                    self._send(200, outer.reply)
                    return

                if self.path == "/v1/chat/completions":
                    obj = {
                        "id": "chatcmpl-fake",
                        "object": "chat.completion",
                        "model": payload.get("model", "fake"),
                        "choices": [{"index": 0,
                                     "message": {"role": "assistant",
                                                 "content": outer.reply},
                                     "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": outer.usage_input_tokens,
                                  "completion_tokens": 5,
                                  "total_tokens": outer.usage_input_tokens + 5},
                    }
                else:
                    obj = {
                        "id": "msg_fake",
                        "type": "message",
                        "role": "assistant",
                        "model": payload.get("model", "fake"),
                        "content": [{"type": "text", "text": outer.reply}],
                        "stop_reason": "end_turn",
                        "stop_sequence": None,
                        "usage": {"input_tokens": outer.usage_input_tokens,
                                  "output_tokens": 5},
                    }
                self._send(200, obj)

            def do_GET(self):
                self._send(404, {"error": {"type": "not_found"}})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.url = "http://127.0.0.1:%d" % self.port
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)
        return False

    def paths_seen(self):
        with self.lock:
            return [p for p, _ in self.requests]


def run_check(base_url, extra=None, timeout=120):
    """needle_check.py を子プロセスで走らせて (exit, stdout, stderr) を返す。"""
    cmd = [sys.executable, CHECK, "--base-url", base_url, "--fixture", FIXTURE,
           "--model", "fake-model", "--timeout", "20"]
    if extra:
        cmd.extend(extra)
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_module():
    spec = importlib.util.spec_from_file_location("needle_check", CHECK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class NeedleSelftest(unittest.TestCase):
    maxDiff = None

    # ------------------------------------------------ (a) 正解 -> exit 0

    def test_a_correct_answer_exits_zero(self):
        """needle をそのまま返す応答 = recall 成功 -> exit 0。"""
        with tempfile.TemporaryDirectory() as td:
            jout = os.path.join(td, "out.json")
            with FakeServer("The access code is " + ANSWER + ".") as srv:
                code, out, err = run_check(srv.url, ["--json-out", jout])
            self.assertEqual(code, 0, "exit=%d\n--stdout--\n%s\n--stderr--\n%s"
                             % (code, out, err))
            rep = load_json(jout)
            self.assertEqual(rep["exit"], 0)
            self.assertTrue(rep["verdict"]["matched"])
            self.assertEqual(rep["verdict"]["reason"], "found")
            self.assertEqual(rep["usage_input_tokens"], 4671)
            self.assertIn("RESULT: RECALL OK", out)
            self.assertIn("usage.input_tokens = 4671", out)

    def test_a_correct_answer_with_whitespace_noise(self):
        """大小文字・空白のゆれ (改行/連続空白) を許容して exit 0。

        fixture の別表記 QX4471 (ハイフン無し) も recall 成功とみなす。
        """
        with FakeServer("  the  ACCESS\ncode for bay 7 is  qx4471  ") as srv:
            code, out, err = run_check(srv.url)
        self.assertEqual(code, 0, "exit=%d\n%s\n%s" % (code, out, err))

    # ------------------------------------------------ (b) 誤答 -> exit 1

    def test_b_wrong_answer_exits_one(self):
        """間違った値 -> exit 1 (recall 失敗)。"""
        with tempfile.TemporaryDirectory() as td:
            jout = os.path.join(td, "out.json")
            with FakeServer("The access code is QX-9999.") as srv:
                code, out, err = run_check(srv.url, ["--json-out", jout])
            self.assertEqual(code, 1, "exit=%d\n%s\n%s" % (code, out, err))
            rep = load_json(jout)
            self.assertEqual(rep["exit"], 1)
            self.assertFalse(rep["verdict"]["matched"])
            self.assertEqual(rep["verdict"]["reason"], "missing")
            self.assertIn("RECALL FAIL", out)

    def test_b_near_miss_code_is_not_accepted(self):
        """QX-44715 のような紛らわしい値で recall 成功にしない (語境界の確認)。"""
        with FakeServer("The access code is QX-44715.") as srv:
            code, out, err = run_check(srv.url)
        self.assertEqual(code, 1, "exit=%d\n%s\n%s" % (code, out, err))

    # ------------------------------------------------ (c) needle を無視 -> exit 1

    def test_c_missing_needle_exits_one(self):
        """needle に触れない応答 (知らないと答える) -> exit 1。"""
        with tempfile.TemporaryDirectory() as td:
            jout = os.path.join(td, "out.json")
            with FakeServer("I do not know. The text does not mention it.") as srv:
                code, out, err = run_check(srv.url, ["--json-out", jout])
            self.assertEqual(code, 1, "exit=%d\n%s\n%s" % (code, out, err))
            rep = load_json(jout)
            self.assertFalse(rep["verdict"]["matched"])
            self.assertEqual(rep["verdict"]["reason"], "forbidden_only")

    def test_c_empty_answer_exits_one(self):
        """空応答も recall 失敗 -> exit 1 (通信失敗の 2 ではない)。"""
        with FakeServer("") as srv:
            code, out, err = run_check(srv.url)
        self.assertEqual(code, 1, "exit=%d\n%s\n%s" % (code, out, err))

    # ------------------------------------------------ (d) 通信失敗 -> exit 2

    def test_d_http_500_exits_two(self):
        """HTTP 500 は判定不能 -> exit 2 (誤答の 1 と混同しない)。"""
        with tempfile.TemporaryDirectory() as td:
            jout = os.path.join(td, "out.json")
            with FakeServer(ANSWER, status=500) as srv:
                code, out, err = run_check(srv.url, ["--json-out", jout])
            self.assertEqual(code, 2, "exit=%d\n%s\n%s" % (code, out, err))
            rep = load_json(jout)
            self.assertEqual(rep["exit"], 2)
            self.assertEqual(rep["status"], 500)
            self.assertIn("ERROR (exit 2)", out)
            self.assertIn("HTTP 500", err)

    def test_d_connection_refused_exits_two(self):
        """接続できない -> exit 2。"""
        port = free_port()
        with tempfile.TemporaryDirectory() as td:
            jout = os.path.join(td, "out.json")
            code, out, err = run_check("http://127.0.0.1:%d" % port,
                                       ["--json-out", jout, "--timeout", "2"])
            self.assertEqual(code, 2, "exit=%d\n%s\n%s" % (code, out, err))
            if os.path.exists(jout):
                self.assertEqual(load_json(jout)["exit"], 2)

    def test_d_malformed_json_exits_two(self):
        """200 だが本文が JSON でない -> exit 2 (解釈不能)。"""
        payload = b"this is not json"
        with FakeServer(payload, paths=("/v1/messages",)) as srv:
            # FakeServer は bytes をそのまま返す (json 化しない)
            code, out, err = run_check(srv.url)
            paths = srv.paths_seen()
        # auto なので chat/completions (404) -> messages (200, 非 JSON) の 2 回。
        # messages まで到達していないと、このテストは解釈失敗ではなく
        # 接続失敗を測ってしまう (空振り) ので、経路そのものを確認する。
        self.assertEqual(paths, ["/v1/chat/completions", "/v1/messages"])
        self.assertEqual(code, 2, "exit=%d\n%s\n%s" % (code, out, err))
        self.assertIn("応答の解釈に失敗", err)
        self.assertIn("ERROR (exit 2)", out)

    # ------------------------------------------------ (e) 404 fallback

    def test_e_auto_falls_back_to_messages_and_needle_survives(self):
        """auto は chat/completions -> 404 -> messages に落ち、needle が本文を素通りする。

        (e) の本体: 実際に post されたリクエスト本文を取り出し、
        組み立てた payload の needle がそのまま含まれていることを確認する。
        ここが崩れると「recall 失敗」が本物の回帰か通信の取り違えか分からなくなる。
        """
        with FakeServer(ANSWER) as srv:  # /v1/chat/completions は 404
            code, out, err = run_check(srv.url)
            paths = srv.paths_seen()
            bodies = list(srv.raw_bodies)
        self.assertEqual(code, 0, "exit=%d\n%s\n%s" % (code, out, err))
        # 1 回目が chat/completions (404)、2 回目が messages (200)
        self.assertEqual(paths[0], "/v1/chat/completions")
        self.assertIn("/v1/messages", paths)
        self.assertTrue(any(p == "/v1/messages" for p in paths))

        ok_body = json.loads(bodies[paths.index("/v1/messages")].decode("utf-8"))
        # anthropic 形式: messages[0].content に本文、question は別発話
        self.assertIn("system", ok_body)
        text0 = ok_body["messages"][0]["content"]
        self.assertIn(NEEDLE, text0,
                      "needle がリクエスト本文に素通りしていない")
        self.assertEqual(ok_body["messages"][1]["content"], FX["question"])
        self.assertEqual(ok_body["temperature"], 0.0)
        self.assertAlmostEqual(ok_body["max_tokens"], FX["gen"]["max_tokens"])

    def test_e_404_on_both_paths_exits_two(self):
        """どの経路も 404 なら exit 2 (判定不能)。"""
        with FakeServer(ANSWER, paths=()) as srv:
            code, out, err = run_check(srv.url)
        self.assertEqual(code, 2, "exit=%d\n%s\n%s" % (code, out, err))

    def test_e_forced_openai_does_not_fall_back(self):
        """--api openai なら messages には落ちない (404 のまま exit 2)。"""
        with FakeServer(ANSWER) as srv:
            code, out, err = run_check(srv.url, ["--api", "openai"])
            paths = srv.paths_seen()
        self.assertEqual(code, 2, "exit=%d\n%s\n%s" % (code, out, err))
        self.assertEqual(paths, ["/v1/chat/completions"])

    def test_e_forced_anthropic_skips_chat_completions(self):
        """--api anthropic なら chat/completions は叩かない。"""
        with FakeServer(ANSWER) as srv:
            code, out, err = run_check(srv.url, ["--api", "anthropic"])
            paths = srv.paths_seen()
        self.assertEqual(code, 0, "exit=%d\n%s\n%s" % (code, out, err))
        self.assertEqual(paths, ["/v1/messages"])

    def test_e_openai_shape_works_when_available(self):
        """server が chat/completions を持っていればそちらで完結する (usage も拾う)。"""
        with tempfile.TemporaryDirectory() as td:
            jout = os.path.join(td, "out.json")
            with FakeServer(ANSWER, paths=("/v1/chat/completions",)) as srv:
                code, out, err = run_check(srv.url, ["--json-out", jout])
                paths = srv.paths_seen()
            self.assertEqual(code, 0, "exit=%d\n%s\n%s" % (code, out, err))
            self.assertEqual(paths, ["/v1/chat/completions"])
            rep = load_json(jout)
            self.assertEqual(rep["api_format"], "openai")
            self.assertFalse(rep["fallback_used"])
            self.assertEqual(rep["usage_input_tokens"], 4671)

    # ------------------------------------------------ (f) 決定性

    def test_f_payload_is_deterministic(self):
        """同じ fixture から 2 回 payload を作ると byte 一致する。"""
        m = load_module()
        a = m.build_prompt(FX, 0.5)
        b = m.build_prompt(FX, 0.5)
        self.assertEqual(a["text"].encode("utf-8"), b["text"].encode("utf-8"))
        self.assertEqual(a, b)
        # 実際に送られた本文どうしも一致する (子プロセス経路で確認)
        with FakeServer(ANSWER) as s1:
            run_check(s1.url)
            body1 = [b for p, b in zip(s1.paths_seen(), s1.raw_bodies)
                     if p == "/v1/messages"][0]
        with FakeServer(ANSWER) as s2:
            run_check(s2.url)
            body2 = [b for p, b in zip(s2.paths_seen(), s2.raw_bodies)
                     if p == "/v1/messages"][0]
        self.assertEqual(body1, body2, "2 回の実リクエスト本文が byte 不一致")

    def test_f_filler_is_stable_and_needle_unique(self):
        """filler は決定的で、needle は payload にちょうど 1 回だけ入る。"""
        m = load_module()
        f1 = m.build_filler(FX)
        f2 = m.build_filler(FX)
        self.assertEqual(f1, f2)
        self.assertGreater(len(f1), 1000)
        p = m.build_prompt(FX, 0.5)
        self.assertEqual(p["text"].count(NEEDLE), 1)
        # filler 側に needle の話題 (bay / access code) が漏れていないこと
        bare = p["text"].replace(NEEDLE, "")
        self.assertNotIn("access code", bare.lower())
        self.assertNotIn("bay 7", bare.lower())

    def test_f_different_depth_moves_needle(self):
        """--depth が効く (needle の実位置が depth に追従する)。"""
        m = load_module()
        head = m.build_prompt(FX, 0.05)
        tail = m.build_prompt(FX, 0.95)
        self.assertLess(head["achieved_depth"], 0.2)
        self.assertGreater(tail["achieved_depth"], 0.8)
        self.assertIn(NEEDLE, head["text"])
        self.assertIn(NEEDLE, tail["text"])
        self.assertLess(head["text"].find(NEEDLE), tail["text"].find(NEEDLE))

    def test_f_depth_flag_changes_payload(self):
        """--depth 0.9 で送ると needle の位置が後ろになる (実リクエストで確認)。"""
        with FakeServer(ANSWER) as srv:
            code, out, err = run_check(srv.url, ["--depth", "0.9"])
            body = [b for p, b in zip(srv.paths_seen(), srv.raw_bodies)
                    if p == "/v1/messages"][0]
        self.assertEqual(code, 0, "exit=%d\n%s\n%s" % (code, out, err))
        text = json.loads(body.decode("utf-8"))["messages"][0]["content"]
        frac = text.find(NEEDLE) / float(len(text))
        self.assertGreater(frac, 0.8)

    # ------------------------------------------------ fixture / 引数

    def test_fixture_shape(self):
        """fixture が必要なキーを持ち、期待値としきい値が整合していること。"""
        for k in ("id", "needle", "question", "expected_answer", "depth",
                  "system_prompt", "gen", "filler", "thresholds"):
            self.assertIn(k, FX)
        self.assertIn(NEEDLE, FX["question"] + NEEDLE)
        self.assertTrue(FX["question"].strip())
        self.assertIn(ANSWER, FX["expected_alternatives"])
        th = FX["thresholds"]
        self.assertLess(th["target_input_tokens_min"], th["target_input_tokens_max"])
        self.assertGreaterEqual(th["orig_input_tokens"], th["target_input_tokens_min"])
        self.assertLessEqual(th["orig_input_tokens"], th["target_input_tokens_max"])
        self.assertTrue(0.0 <= FX["depth"] <= 1.0)

    def test_token_band_warning(self):
        """目標 token 帯から外れたら警告を出すが、recall 成功なら exit 0 のまま。"""
        with FakeServer(ANSWER, usage_input_tokens=12345) as srv:
            code, out, err = run_check(srv.url)
        self.assertEqual(code, 0, "exit=%d\n%s\n%s" % (code, out, err))
        self.assertIn("warning:", out)
        self.assertIn("12345", out)

    def test_bad_depth_rejected(self):
        """--depth が範囲外なら exit 2 で止まる (黙って丸めない)。"""
        with FakeServer(ANSWER) as srv:
            code, out, err = run_check(srv.url, ["--depth", "1.5"])
        self.assertEqual(code, 2)

    def test_missing_fixture_exits_two(self):
        """fixture が無い -> exit 2。"""
        cmd = [sys.executable, CHECK, "--fixture", "/nonexistent/fx.json",
               "--base-url", "http://127.0.0.1:1", "--timeout", "1"]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 2, p.stderr)

    # ------------------------------------------------ 純関数

    def test_judge_boundaries(self):
        """照合の境界: 前置き許容 / 別表記 / 部分一致の棄却。"""
        m = load_module()
        expected, alts, forb = FX["expected_answer"], FX["expected_alternatives"], \
            FX["forbidden_answers"]
        cases = [
            (ANSWER, True),
            ("the code is " + ANSWER.lower(), True),
            ("QX4471", True),
            ("QX 4471", False),        # 空白を挟んだ形は合言葉ではない
            ("QX-9999", False),
            ("QX-44715", False),       # 語境界で棄却
            ("QX44710", False),
            ("I do not know", False),
            ("", False),
            ("   \n\t ", False),
        ]
        for text, want in cases:
            got = m.judge(text, expected, alts, forb)
            self.assertEqual(got["matched"], want,
                             "judge(%r) -> %r" % (text, got))
        self.assertEqual(m.judge("", expected, alts, forb)["reason"], "empty")
        self.assertEqual(m.judge("I do not know", expected, alts,
                                 forb)["reason"], "forbidden_only")
        self.assertEqual(m.judge("nope", expected, alts, forb)["reason"], "missing")

    def test_normalize(self):
        m = load_module()
        self.assertEqual(m.normalize("  A\tB\n\nC "), "a b c")
        self.assertEqual(m.normalize(None), "")


def _stdlib_imports_only(path):
    """script が標準ライブラリ以外を import していないことを確かめる。"""
    import ast
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue  # 相対 import (パッケージ内) は対象外
            if node.module:
                names.add(node.module.split(".")[0])
    # 自前のモジュールは同じディレクトリなので許容
    local = {"needle_check"}
    ext = sorted(n for n in names if n not in local and n not in sys.stdlib_module_names)
    return ext


class DependencySelftest(unittest.TestCase):
    def test_no_external_dependencies(self):
        """needle_check.py / needle_selftest.py が標準ライブラリだけで動くこと。"""
        for path in (CHECK, os.path.abspath(__file__)):
            ext = _stdlib_imports_only(path)
            self.assertEqual(ext, [], "%s が外部依存を import している: %s"
                             % (os.path.basename(path), ext))


if __name__ == "__main__":
    unittest.main(verbosity=2)