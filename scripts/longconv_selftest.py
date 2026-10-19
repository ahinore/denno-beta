#!/usr/bin/env python3
"""P4-1 (s97): longconv_check.py の自己テスト (サーバ不要・標準ライブラリのみ)。

dev コンテナは denno-gpu / 8095 に到達できないので、検出器の正しさはここで示す。
標準ライブラリの http.server で偽の OpenAI 互換サーバ (と Anthropic 互換サーバ) を
立て、決め打ちの応答を返して longconv_check.py を子プロセスで走らせ、
exit code が期待どおりかを確かめる。

確かめること:
  clean      10 ターンの正常応答                     -> exit 0 (崩壊検出なし)
  repetition 反復崩壊 (短い言い回しの繰り返し)        -> exit 1 (repetition)
  contamination 特殊トークン/ロール接頭辞の漏出       -> exit 1 (contamination)
  degenerate 空応答                                  -> exit 1 (degenerate)
  duplicate  直前ターンと完全一致する応答            -> exit 1 (duplicate)
  (おまけ) anthropic 形式の経路でも clean -> exit 0 / 空 -> exit 1
  (おまけ) --temperature が --max-temp を超えたら丸める / --expect-cap で回帰検出

使い方: python3 scripts/longconv_selftest.py [-v]
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
CHECK = os.path.join(HERE, "longconv_check.py")
FIXTURE = os.path.join(HERE, "longconv_fixture.json")

# 正常応答 (反復なし・特殊トークンなし・十分な長さ・毎回違う内容)
CLEAN_REPLIES = [
    "得意なことは、文章の要約と、考えを整理して箇条書きにすることです。",
    "これは簡単に言うと、長い話から大切なところだけを取り出す技術です。"
    "まず全体を読み、次に大事な文を選び、最後に短くまとめます。",
    "「要約」は「短くまとめる」、「整理」は「順番を決める」と言い換えられます。",
    "はい、覚えています。得意なことを 2 つ挙げてほしい、というお願いでした。",
    "朝の散歩では、次の 3 つに気をつけてください。\n1. 足元の段差\n"
    "2. 車の通り道\n3. 天気と気温",
    "3 つ目の天気と気温については、出かける前に予報を見て、"
    "雨なら傘を、暑ければ水筒を持って行くと安心です。",
    "4 ター目です。そこで最初のお願いを確認されました。",
    "あなたは得意なことを尋ね、その説明を深掘りしました。"
    "その後で朝の散歩の注意点を 3 つ挙げ、3 つ目を詳しく説明しました。",
    "入っています。",
    "得意なことは、文章の要約と、考えを整理して箇条書きにすることです。",
]

# 崩壊応答 (各ターンに埋め込む)
REPEAT_REPLY = "はい。とても良い質問ですね。" * 12
CONTAM_REPLY = "承知しました。<|im_start|>assistant\nでは続けます。今日は良い天気です。"
EMPTY_REPLY = ""
# 同一文: ターン 3 と 4 で同じ文字列を返す

# しきい値の検証: CLEAN_REPLIES 同士が「同一文」と誤判定されないこと。
# (最後のターンは 1 ターン目と同じ文を返す台本なので、clean ケースでは
#  ターン 9 の直前と比べる形になり、類似判定に引っかからない)


class FakeServer:
    """決め打ちの応答を順に返す偽サーバ。openai / anthropic の 2 形式に対応。"""

    def __init__(self, replies, fmt="openai"):
        self.replies = list(replies)
        self.fmt = fmt
        # 実サーバの模擬: fmt に対応する経路だけを公開し、他は 404。
        # (denno serve は /v1/messages のみ。/v1/chat/completions は 404)
        self.allow_paths = (("/v1/chat/completions",) if fmt == "openai"
                            else ("/v1/messages",))
        self.requests = []          # (path, payload)
        self.i = 0
        self.lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            # ログを出さない (self-test の出力を汚さない)
            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("content-length") or 0)
                raw = self.rfile.read(n) if n else b"{}"
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except Exception:
                    payload = {}
                if self.path not in outer.allow_paths:
                    self.send_response(404)
                    body = b'{"error":"not found"}'
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                with outer.lock:
                    idx = outer.i
                    outer.i += 1
                    outer.requests.append((self.path, payload))
                text = outer.replies[idx] if idx < len(outer.replies) else "（応答なし）"
                if outer.fmt == "openai":
                    obj = {
                        "id": "chatcmpl-fake%d" % idx,
                        "object": "chat.completion",
                        "model": payload.get("model", "fake"),
                        "choices": [{"index": 0, "message": {"role": "assistant",
                                                             "content": text},
                                     "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 20,
                                  "total_tokens": 30},
                    }
                else:
                    obj = {
                        "id": "msg_fake%d" % idx,
                        "type": "message",
                        "role": "assistant",
                        "model": payload.get("model", "fake"),
                        "content": [{"type": "text", "text": text}],
                        "stop_reason": "end_turn",
                        "stop_sequence": None,
                        "usage": {"input_tokens": 10, "output_tokens": 20},
                    }
                body = json.dumps(obj).encode("utf-8")
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                # /v1/models などは未対応 (404 で十分)
                self.send_response(404)
                self.end_headers()

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


def run_check(base_url, extra=None, timeout=60):
    """longconv_check.py を子プロセスで走らせて (exit, stdout, stderr) を返す。"""
    cmd = [sys.executable, CHECK, "--base-url", base_url, "--fixture", FIXTURE,
           "--model", "fake-model", "--turns", "10", "--timeout", "20"]
    if extra:
        cmd.extend(extra)
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def load_report(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class DetectorSelftest(unittest.TestCase):
    maxDiff = None

    # ------------------------------------------------------------ 正常

    def test_clean_openai_exits_zero(self):
        """正常 10 ターン (OpenAI 形式) は崩壊なし = exit 0。"""
        with tempfile.TemporaryDirectory() as td:
            jout = os.path.join(td, "out.json")
            with FakeServer(CLEAN_REPLIES, "openai") as srv:
                code, out, err = run_check(srv.url, ["--json-out", jout])
                self.assertEqual(srv.requests[0][0], "/v1/chat/completions")
                self.assertEqual(len(srv.requests), 10, "10 ターン送ること")
            self.assertEqual(code, 0, "exit=%d\n--stdout--\n%s\n--stderr--\n%s"
                             % (code, out, err))
            rep = load_report(jout)
            self.assertEqual(rep["exit"], 0)
            self.assertEqual(rep["summary"]["collapsed_turns"], [])
            self.assertEqual(rep["api_format"], "openai")
            self.assertEqual(len(rep["turns"]), 10)
            self.assertIn("RESULT: OK", out)

    def test_clean_anthropic_exits_zero(self):
        """denno serve の実経路 (/v1/messages) でも clean は exit 0。

        auto 指定で chat/completions が 404 → messages に落ちることを確かめる。
        """
        with tempfile.TemporaryDirectory() as td:
            jout = os.path.join(td, "out.json")
            with FakeServer(CLEAN_REPLIES, "anthropic") as srv:
                code, out, err = run_check(srv.url, ["--json-out", jout])
                paths = [p for p, _ in srv.requests]
            self.assertEqual(code, 0, "exit=%d\n%s\n%s" % (code, out, err))
            self.assertIn("/v1/messages", paths)
            rep = load_report(jout)
            self.assertEqual(rep["api_format"], "anthropic")
            self.assertEqual(rep["summary"]["collapsed_turns"], [])

    # ------------------------------------------------------------ 崩壊 4 種

    def test_repetition_collapse_exits_one(self):
        """反復崩壊: 同じ短い言い回しの連続 -> exit 1 (repetition)。"""
        replies = list(CLEAN_REPLIES)
        replies[5] = REPEAT_REPLY
        with tempfile.TemporaryDirectory() as td:
            jout = os.path.join(td, "out.json")
            with FakeServer(replies, "openai") as srv:
                code, out, err = run_check(srv.url, ["--json-out", jout])
            self.assertEqual(code, 1, "exit=%d\n%s\n%s" % (code, out, err))
            rep = load_report(jout)
            self.assertEqual(rep["exit"], 1)
            self.assertEqual(rep["summary"]["collapsed_turns"], [6])
            self.assertIn("repetition", rep["summary"]["categories"])
            self.assertIn("repetition", rep["turns"][5]["issues"])
            self.assertIn("RESULT: COLLAPSE", out)

    def test_contamination_collapse_exits_one(self):
        """テンプレ片の漏出: <|im_start|> / assistant: が本文に出る -> exit 1。"""
        replies = list(CLEAN_REPLIES)
        replies[2] = CONTAM_REPLY
        with tempfile.TemporaryDirectory() as td:
            jout = os.path.join(td, "out.json")
            with FakeServer(replies, "openai") as srv:
                code, out, err = run_check(srv.url, ["--json-out", jout])
            self.assertEqual(code, 1, "exit=%d\n%s\n%s" % (code, out, err))
            rep = load_report(jout)
            self.assertEqual(rep["summary"]["collapsed_turns"], [3])
            self.assertIn("contamination", rep["turns"][2]["issues"])
            markers = rep["turns"][2]["detail"]["contamination"]["markers"]
            self.assertIn("<|im_start|>", markers)

    def test_empty_reply_exits_one(self):
        """空応答: 空文字 -> exit 1 (degenerate)。"""
        replies = list(CLEAN_REPLIES)
        replies[3] = EMPTY_REPLY
        with tempfile.TemporaryDirectory() as td:
            jout = os.path.join(td, "out.json")
            with FakeServer(replies, "openai") as srv:
                code, out, err = run_check(srv.url, ["--json-out", jout])
            self.assertEqual(code, 1, "exit=%d\n%s\n%s" % (code, out, err))
            rep = load_report(jout)
            self.assertEqual(rep["summary"]["collapsed_turns"], [4])
            self.assertIn("degenerate", rep["turns"][3]["issues"])
            self.assertIn("empty", rep["turns"][3]["detail"]["degenerate"]["reasons"])

    def test_whitespace_only_reply_exits_one(self):
        """空応答 (空白のみ) も degenerate -> exit 1。"""
        replies = list(CLEAN_REPLIES)
        replies[7] = "   \n\t  \n"
        with tempfile.TemporaryDirectory() as td:
            with FakeServer(replies, "openai") as srv:
                code, out, err = run_check(srv.url)
            self.assertEqual(code, 1, "exit=%d\n%s\n%s" % (code, out, err))
            self.assertIn("turn  8", out)

    def test_duplicate_reply_exits_one(self):
        """同一文の連続: 直前ターンと完全一致 -> exit 1 (duplicate)。"""
        replies = list(CLEAN_REPLIES)
        replies[4] = CLEAN_REPLIES[3]  # ターン 5 がターン 4 と同じ文
        with tempfile.TemporaryDirectory() as td:
            jout = os.path.join(td, "out.json")
            with FakeServer(replies, "openai") as srv:
                code, out, err = run_check(srv.url, ["--json-out", jout])
            self.assertEqual(code, 1, "exit=%d\n%s\n%s" % (code, out, err))
            rep = load_report(jout)
            self.assertEqual(rep["summary"]["collapsed_turns"], [5])
            d = rep["turns"][4]["detail"]["duplicate"]
            self.assertTrue(d["identical"])
            self.assertGreaterEqual(d["similarity"], 0.999)

    def test_near_duplicate_reply_exits_one(self):
        """類似度が高すぎる応答 (完全一致ではない) も duplicate -> exit 1。"""
        replies = list(CLEAN_REPLIES)
        base = CLEAN_REPLIES[3]
        replies[4] = base + "。"  # 1 文字だけ足す
        with tempfile.TemporaryDirectory() as td:
            with FakeServer(replies, "openai") as srv:
                code, out, err = run_check(srv.url)
            self.assertEqual(code, 1, "exit=%d\n%s\n%s" % (code, out, err))

    def test_anthropic_route_detects_collapse(self):
        """Anthropic 形式の応答でも崩壊を検出する (経路で判定が変わらない)。"""
        replies = list(CLEAN_REPLIES)
        replies[1] = REPEAT_REPLY
        with FakeServer(replies, "anthropic") as srv:
            code, out, err = run_check(srv.url)
        self.assertEqual(code, 1, "exit=%d\n%s\n%s" % (code, out, err))

    def test_explicit_anthropic_api_flag(self):
        """--api anthropic を明示すれば chat/completions は叩かない。"""
        with FakeServer(CLEAN_REPLIES, "anthropic") as srv:
            code, out, err = run_check(srv.url, ["--api", "anthropic"])
            paths = {p for p, _ in srv.requests}
        self.assertEqual(code, 0, "exit=%d\n%s\n%s" % (code, out, err))
        self.assertEqual(paths, {"/v1/messages"})

    def test_explicit_openai_api_flag(self):
        """--api openai を明示すれば messages には落ちない。"""
        with FakeServer(CLEAN_REPLIES, "openai") as srv:
            code, out, err = run_check(srv.url, ["--api", "openai"])
            paths = {p for p, _ in srv.requests}
        self.assertEqual(code, 0, "exit=%d\n%s\n%s" % (code, out, err))
        self.assertEqual(paths, {"/v1/chat/completions"})

    # ------------------------------------------------------------ 通信失敗

    def test_unreachable_server_exits_two(self):
        """どこにも繋がらないときは exit 2 (崩壊と区別する)。"""
        with tempfile.TemporaryDirectory() as td:
            jout = os.path.join(td, "out.json")
            # 使われていないポートを取る (bind して即 close)
            import socket
            s = socket.socket()
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
            s.close()
            code, out, err = run_check("http://127.0.0.1:%d" % port,
                                       ["--json-out", jout, "--timeout", "2"])
            self.assertEqual(code, 2, "exit=%d\n%s\n%s" % (code, out, err))
            if os.path.exists(jout):
                self.assertEqual(load_report(jout)["exit"], 2)

    def test_no_endpoint_found_exits_two(self):
        """どの経路も 404 なら exit 2 (崩壊ではない・判定不能)。"""
        with FakeServer(CLEAN_REPLIES, "openai") as srv:
            # 公開経路を潰して auto の両形式を 404 にする
            srv.allow_paths = ()
            code, out, err = run_check(srv.url, ["--timeout", "5"])
        self.assertEqual(code, 2, "exit=%d\n%s\n%s" % (code, out, err))
        self.assertIn("RESULT: ERROR", out)

    # ------------------------------------------------------------ 温度 / cap

    def test_temperature_is_sent(self):
        """--temperature が実際にリクエストに載る (既定は --max-temp と同じ)。"""
        with FakeServer(CLEAN_REPLIES, "openai") as srv:
            code, _, _ = run_check(srv.url, ["--temperature", "0.5"])
            self.assertEqual(code, 0)
            temps = [p.get("temperature") for _, p in srv.requests]
            self.assertTrue(temps)
            for t in temps:
                self.assertAlmostEqual(t, 0.5, places=6)

    def test_temperature_over_max_temp_is_clamped(self):
        """--temperature が --max-temp を超えたら警告して丸める。"""
        with FakeServer(CLEAN_REPLIES, "openai") as srv:
            code, out, err = run_check(srv.url, ["--temperature", "1.0",
                                                 "--max-temp", "0.7"])
            temps = [p.get("temperature") for _, p in srv.requests]
        self.assertEqual(code, 0)
        for t in temps:
            self.assertAlmostEqual(t, 0.7, places=6)
        self.assertIn("WARNING", err)

    def test_expect_cap_flags_too_hot_request(self):
        """--expect-cap 0.7 で 0.9 を送ろうとしたら cap 回帰として exit 1。"""
        with FakeServer(CLEAN_REPLIES, "openai") as srv:
            code, out, err = run_check(srv.url, ["--temperature", "0.9",
                                                 "--max-temp", "0.9",
                                                 "--expect-cap", "0.7"])
        self.assertEqual(code, 1, "exit=%d\n%s\n%s" % (code, out, err))
        self.assertIn("cap check: FAIL", out)

    def test_expect_cap_ok_when_within(self):
        """--expect-cap 0.7 で 0.7 を送るのは回帰ではない (exit 0)。"""
        with FakeServer(CLEAN_REPLIES, "openai") as srv:
            code, out, err = run_check(srv.url, ["--temperature", "0.7",
                                                 "--expect-cap", "0.7"])
        self.assertEqual(code, 0, "exit=%d\n%s\n%s" % (code, out, err))
        self.assertIn("cap check", out)
        self.assertNotIn("cap check: FAIL", out)

    # ------------------------------------------------------------ fixture

    def test_fixture_shape(self):
        """fixture は 10 ターン + 判定しきい値を持つこと。"""
        with open(FIXTURE, "r", encoding="utf-8") as f:
            fx = json.load(f)
        self.assertEqual(len(fx["turns"]), 10)
        for i, t in enumerate(fx["turns"], 1):
            self.assertEqual(t["n"], i)
            self.assertTrue(t["user"].strip())
        th = fx["thresholds"]
        for k in ("repeat_period_min", "repeat_min_repeats", "repeat_coverage_max",
                  "similarity_max", "min_chars", "max_chars"):
            self.assertIn(k, th)
        self.assertGreaterEqual(th["repeat_period_min"], 8)
        self.assertTrue(th["contamination_markers"])

    def test_turns_is_honoured(self):
        """--turns 3 なら 3 リクエストだけ送る。"""
        with FakeServer(CLEAN_REPLIES, "openai") as srv:
            code, out, err = run_check(srv.url, ["--turns", "3"])
            n = len(srv.requests)
        self.assertEqual(code, 0, "exit=%d\n%s\n%s" % (code, out, err))
        self.assertEqual(n, 3)


class DetectorUnit(unittest.TestCase):
    """検出器の純関数を直接叩く (HTTP を介さない・しきい値の境界を見る)。"""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec = importlib.util.spec_from_file_location("longconv_check", CHECK)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls.m = mod

    def test_longest_repeat_finds_period_8(self):
        text = "あ" * 40 + "末尾"
        cov, period, reps, unit = self.m.longest_repeat(text, 8, 3)
        self.assertGreaterEqual(reps, 3)
        self.assertGreater(cov, 0.8)

    def test_longest_repeat_ignores_short_norm(self):
        """周期 4 の短い反復だけの本文は、周期 4 を対象外にすれば拾わない。

        handoff の指定「n=8 以上の反復」を守るため、period_min 未満の周期は
        探索しないこと (長さ 12 の "abcd" x3 は周期 8 の n-gram を 3 回
        反復できないので、period_min=8 では coverage 0 になる)。
        """
        text = "abcd" * 3  # 12 文字。周期 4 x3 だが周期 8 x3 には足りない
        cov, period, reps, _ = self.m.longest_repeat(text, 8, 3)
        self.assertEqual((cov, period, reps), (0.0, 0, 0))
        # period_min=4 なら拾える (指定を変えれば検出できる = 探索が効いている)
        cov4, period4, _, _ = self.m.longest_repeat(text, 4, 3)
        self.assertEqual(period4, 4)
        self.assertGreater(cov4, 0.9)

    def test_longest_repeat_reports_coverage_and_period(self):
        """周期 8 x 5 の反復で coverage/period/repeats が期待どおり。"""
        unit = "12345678"
        text = unit * 5
        cov, period, reps, u = self.m.longest_repeat(text, 8, 3)
        self.assertEqual(reps, 5)
        self.assertEqual(period, len(u))
        self.assertAlmostEqual(cov, 1.0)

    def test_longest_repeat_clean_text_is_low(self):
        cov, _, _, _ = self.m.longest_repeat(CLEAN_REPLIES[1], 8, 3)
        self.assertLess(cov, 0.35)

    def test_contamination_finds_special_tokens(self):
        hits = self.m.find_contamination(CONTAM_REPLY, ["<|im_start|>", "assistant:"])
        self.assertIn("<|im_start|>", hits)

    def test_contamination_requires_line_start_for_roles(self):
        """"assistant:" は行頭のみ。本文中の言及は拾いすぎない。"""
        hits = self.m.find_contamination("the assistant: a helper", ["assistant:"])
        self.assertEqual(hits, [])
        hits2 = self.m.find_contamination("前置き\nassistant: 続き", ["assistant:"])
        self.assertIn("assistant:", hits2)

    def test_similarity_identical(self):
        self.assertAlmostEqual(self.m.similarity("同じ文です", "同じ文です"), 1.0)

    def test_detect_turn_clean(self):
        th = dict(self.m.DEFAULTS)
        issues, _ = self.m.detect_turn(CLEAN_REPLIES[1], "別の文", th, False)
        self.assertEqual(issues, [])

    def test_detect_turn_all_categories_individually(self):
        th = dict(self.m.DEFAULTS)
        cases = {
            "repetition": REPEAT_REPLY,
            "contamination": CONTAM_REPLY,
            "degenerate": "",
        }
        for cat, text in cases.items():
            issues, _ = self.m.detect_turn(text, "別の文", th, False)
            self.assertIn(cat, issues, "%s を検出できていない" % cat)
        issues, _ = self.m.detect_turn("同じ文", "同じ文", th, False)
        self.assertIn("duplicate", issues)


if __name__ == "__main__":
    unittest.main(verbosity=2)