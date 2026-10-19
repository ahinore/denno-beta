#!/usr/bin/env python3
"""P4-1 (s103): scripts/long_conversation_check.py の CPU テスト (GPU 不要・標準ライブラリのみ)。

確かめること:
  - 検出器: きれいな応答 / 繰り返し行 / 繰り返し末尾文 / 長い行の支配 /
    template 印 6 種 / 偽の user ターン / max_tokens / 空応答
  - 判定の決定性: 同じトランスクリプト JSON からは常に同じ判定 (2 回パースしても同じ)
  - 実行経路: 偽サーバ (http.server をスレッドで) を ephemeral port に立て、
    きれいな 10 ターン / turn 4 の崩れ (exit 2) / template 混入 / max_tokens は
    報告のみ / HTTP 500 -> exit 3 / 接続拒否 -> exit 3 / 8080 と 8090 の拒否 /
    --dry-run は何も送らない / --out が書いたトランスクリプトを --replay すると同じ判定
  - 会話が実際に伸びること (turn N の messages 長 = 2N-1) と、偽サーバがそれを観測すること

実行:
  python3 scripts/test_long_conversation_check.py -v
  python3 -m unittest discover -s scripts -p 'test_long_conversation_check.py'
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "long_conversation_check.py")
FIXTURE = os.path.join(HERE, "fixtures", "long_conversation_turns.json")
sys.path.insert(0, HERE)

import long_conversation_check as lcc  # noqa: E402  (同じディレクトリの実装を読む)

# 検出器の材料 (self_test と同じ趣旨だが、テスト側で独立に固定する)
CLEAN_REPLY = ("初日は北鎌倉から入ると人が少ないです。"
               "建長寺を開門と同時に見てから鶴岡八幡宮へ歩きます。"
               "昼は小町通りを外した路地の店にすると待ちません。")
REP_LINE = "鎌倉の回り方をご案内しますので、ご確認ください。"  # 24 chars


def free_port():
    """ephemeral port を 1 つ確保して番号だけ返す (テスト用)。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class FakeMessages:
    """Anthropic 形式の /v1/messages を返す偽サーバ (スレッド)。

    mode:
      "clean"     すべて CLEAN_REPLY
      "collapse"  4 ターン目だけ繰り返し行の崩れ
      "template"  3 ターン目だけ template 印が行頭に出る
      "max_tokens" すべて CLEAN_REPLY だが stop_reason=max_tokens
      "http500"   すべて HTTP 500
    受けたリクエストの (messages 長, model, max_tokens) は self.seen に残す。
    """

    def __init__(self, mode="clean"):
        self.mode = mode
        self.seen = []
        self.requests = 0
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                outer.requests += 1
                n = int(self.headers.get("content-length") or 0)
                raw = self.rfile.read(n)
                try:
                    body = json.loads(raw.decode("utf-8"))
                except ValueError:
                    body = {}
                msgs = body.get("messages") or []
                outer.seen.append({
                    "path": self.path,
                    "messages": len(msgs),
                    "model": body.get("model"),
                    "max_tokens": body.get("max_tokens"),
                    "temperature": body.get("temperature"),
                    "roles": [m.get("role") for m in msgs],
                    "contents": [m.get("content") for m in msgs],
                })
                if outer.mode == "http500":
                    self.send_response(500)
                    self.send_header("content-length", "0")
                    self.end_headers()
                    return
                turn = len(outer.seen)
                reply = CLEAN_REPLY
                stop = "end_turn"
                if outer.mode == "collapse" and turn == 4:
                    reply = "候補です。\n" + (REP_LINE + "\n") * 4 + "以上です。"
                if outer.mode == "template" and turn == 3:
                    reply = "回答です。\nassistant\n次の回答です。"
                if outer.mode == "max_tokens":
                    stop = "max_tokens"
                out = json.dumps({
                    "id": "msg_fake",
                    "type": "message",
                    "role": "assistant",
                    "model": body.get("model"),
                    "content": [{"type": "text", "text": reply}],
                    "stop_reason": stop,
                    "usage": {"input_tokens": 10 * turn, "output_tokens": 20},
                }, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        self.base_url = "http://127.0.0.1:%d" % self.port
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)
        return False


def run_script(args, timeout=120):
    """スクリプトを子プロセスとして走らせ、CompletedProcess を返す。"""
    return subprocess.run([sys.executable, SCRIPT] + list(args),
                          capture_output=True, text=True, timeout=timeout)


def flags_of(reply, stop_reason="end_turn"):
    return [f for f, _ in lcc.detect_turn({"reply": reply, "stop_reason": stop_reason})]


# ---------------------------------------------------------------------------
# 検出器
# ---------------------------------------------------------------------------

class TestDetectors(unittest.TestCase):
    def test_clean_reply_has_no_flags(self):
        self.assertEqual(flags_of(CLEAN_REPLY), [])

    def test_repeated_long_line_detected(self):
        rep = "候補です。\n" + (REP_LINE + "\n") * 4 + "以上です。"
        self.assertIn("repeat_line", flags_of(rep))

    def test_repeat_line_threshold_is_24_chars(self):
        # 24 文字 -> 検出、23 文字 -> 検出しない (境界)
        self.assertEqual(len(REP_LINE), lcc.MIN_REPEAT_LINE_CHARS)
        just_under = REP_LINE[:-1]
        self.assertEqual(len(just_under), 23)
        under = "候補です。\n" + (just_under + "\n") * 4 + "以上です。"
        self.assertNotIn("repeat_line", flags_of(under))

    def test_repeat_line_needs_three_in_a_row(self):
        two = "候補です。\n" + (REP_LINE + "\n") * 2 + "以上です。"
        self.assertNotIn("repeat_line", flags_of(two))

    def test_repeated_trailing_sentence_detected(self):
        s = "初日は歩きます。時間に余裕があります。時間に余裕があります。時間に余裕があります。"
        self.assertIn("repeat_sentence", flags_of(s))

    def test_distinct_sentences_are_clean(self):
        s = "初日は歩きます。昼は駅前です。二日目は早く出ます。"
        self.assertNotIn("repeat_sentence", flags_of(s))

    def test_dominant_long_line_detected(self):
        text = "前置きです。" + "".join(
            "この行の %02d 番目の説明は応答の大半を占めています。" % i for i in range(1, 21))
        self.assertGreaterEqual(len(text), lcc.LONG_REPLY_CHARS)
        self.assertIn("long_line", flags_of(text))

    def test_short_reply_is_never_a_long_line(self):
        # 1 行でも 400 文字未満なら long_line にしない
        text = "あ" * 300
        self.assertNotIn("long_line", flags_of(text))

    def test_each_template_marker_detected(self):
        for m in lcc.TEMPLATE_MARKERS:
            text = "回答です。\n%s\n続きです。" % m
            self.assertIn("template_marker", flags_of(text), m)

    def test_inline_marker_is_not_a_leak(self):
        for m in lcc.TEMPLATE_MARKERS:
            text = "回答です。%s という語を説明します。" % m
            self.assertNotIn("template_marker", flags_of(text), m)

    def test_fake_user_turn_detected(self):
        text = "初日は歩きます。\n\nuser\n二日目も教えてください。"
        self.assertIn("fake_turn", flags_of(text))

    def test_mentioning_user_is_not_a_fake_turn(self):
        text = "初日は歩きます。\n\nuser の意図は混雑回避です。"
        self.assertNotIn("fake_turn", flags_of(text))

    def test_max_tokens_is_soft(self):
        flags = flags_of(CLEAN_REPLY, stop_reason="max_tokens")
        self.assertEqual(flags, ["max_tokens"])
        v = lcc.transcript_verdict({"turns": [{"index": 2, "reply": CLEAN_REPLY,
                                               "stop_reason": "max_tokens"}]})
        self.assertTrue(v["clean"])
        self.assertEqual(v["exit_code"], 0)
        self.assertEqual([r["flag"] for r in v["soft"]], ["max_tokens"])

    def test_empty_replies_detected(self):
        for text in ("", "   ", "\n\n\t "):
            self.assertIn("empty", flags_of(text), repr(text))

    def test_non_string_reply_is_empty(self):
        self.assertEqual([f for f, _ in lcc.detect(None)], ["empty"])


class TestVerdict(unittest.TestCase):
    def transcript(self, *replies):
        return {"turns": [{"index": i, "reply": r, "stop_reason": "end_turn"}
                          for i, r in enumerate(replies, 1)]}

    def test_clean_transcript_exits_zero(self):
        v = lcc.transcript_verdict(self.transcript(CLEAN_REPLY, CLEAN_REPLY))
        self.assertEqual(v["exit_code"], 0)
        self.assertEqual(v["turn_count"], 2)

    def test_hard_flag_exits_two_with_turn_index(self):
        bad = "候補です。\n" + (REP_LINE + "\n") * 4 + "以上です。"
        v = lcc.transcript_verdict(self.transcript(CLEAN_REPLY, CLEAN_REPLY,
                                                   CLEAN_REPLY, bad))
        self.assertEqual(v["exit_code"], 2)
        self.assertEqual(v["failures"][0]["turn"], 4)
        self.assertEqual(v["failures"][0]["flag"], "repeat_line")
        self.assertTrue(v["failures"][0]["evidence"])

    def test_verdict_is_deterministic(self):
        bad = "候補です。\n" + (REP_LINE + "\n") * 4 + "以上です。"
        t = self.transcript(CLEAN_REPLY, bad, "", CLEAN_REPLY)
        a = lcc.transcript_verdict(t)
        b = lcc.transcript_verdict(json.loads(json.dumps(t)))
        self.assertEqual(a, b)

    def test_malformed_transcript_does_not_crash(self):
        self.assertEqual(lcc.transcript_verdict({})["exit_code"], 0)
        self.assertEqual(lcc.transcript_verdict({"turns": []})["turn_count"], 0)

    def test_failure_table_lists_turn_flag_evidence(self):
        bad = "候補です。\n" + (REP_LINE + "\n") * 4 + "以上です。"
        v = lcc.transcript_verdict(self.transcript(bad))
        lines = lcc.format_failure_table(v)
        self.assertIn("turn", lines[0])
        self.assertTrue(any("repeat_line" in ln and " 1 " in ln for ln in lines),
                        "\n".join(lines))


# ---------------------------------------------------------------------------
# fixture / plan / 会話の組み立て
# ---------------------------------------------------------------------------

class TestFixtureAndPlan(unittest.TestCase):
    def test_fixture_has_ten_turns_in_order(self):
        prompts = lcc.load_fixture(FIXTURE)
        self.assertEqual(len(prompts), 10)
        self.assertTrue(all(p.strip() for p in prompts))
        self.assertTrue(prompts[0].startswith("来月の三連休"))

    def test_fixture_has_a_notes_field(self):
        with open(FIXTURE, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("notes", data)
        self.assertIn("NOT measurements", data["notes"])

    def test_plan_takes_the_first_n(self):
        prompts = lcc.load_fixture(FIXTURE)
        self.assertEqual(lcc.plan_turns(prompts, 3), prompts[:3])

    def test_plan_refuses_more_than_the_fixture_has(self):
        prompts = lcc.load_fixture(FIXTURE)
        with self.assertRaises(lcc.FixtureError):
            lcc.plan_turns(prompts, 11)

    def test_missing_fixture_is_a_fixture_error(self):
        with self.assertRaises(lcc.FixtureError):
            lcc.load_fixture("/nonexistent/turns.json")

    def test_bad_fixture_shapes_are_rejected(self):
        td = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, td, True)
        cases = {"empty": {"turns": []},
                 "nocl": {},
                 "badtext": {"turns": [{"index": 1, "text": "  "}]},
                 "badindex": {"turns": [{"index": 2, "text": "a"}]}}
        for name, data in cases.items():
            p = os.path.join(td, name + ".json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump(data, f)
            with self.assertRaises(lcc.FixtureError, msg=name):
                lcc.load_fixture(p)

    def test_messages_grow_by_two_per_turn(self):
        history = []
        lengths = []
        for i in range(1, 11):
            msgs = lcc.build_messages(history, "turn %d" % i)
            lengths.append(len(msgs))
            history.append(("user", "turn %d" % i))
            history.append(("assistant", "reply %d" % i))
        self.assertEqual(lengths, list(range(1, 20, 2)))

    def test_build_messages_does_not_mutate_history(self):
        history = [("user", "a"), ("assistant", "b")]
        lcc.build_messages(history, "c")
        self.assertEqual(history, [("user", "a"), ("assistant", "b")])

    def test_request_body_has_no_temperature(self):
        # cap はサーバ側の設定。ここで温度を上書きすると cap の回帰を測れない。
        body = lcc.build_request_body("m", [{"role": "user", "content": "x"}], 32)
        self.assertNotIn("temperature", body)
        self.assertEqual(body["max_tokens"], 32)
        self.assertEqual(body["model"], "m")

    def test_response_text_joins_text_blocks_only(self):
        body = {"content": [{"type": "text", "text": "ab"},
                            {"type": "thinking", "thinking": "zzz"},
                            {"type": "text", "text": "cd"}]}
        self.assertEqual(lcc.response_text(body), "abcd")

    def test_response_text_rejects_a_malformed_body(self):
        for body in ({}, {"content": "text"}, []):
            with self.assertRaises(ValueError):
                lcc.response_text(body)


class TestGuards(unittest.TestCase):
    class Args:
        def __init__(self, base_url, turns=10, max_tokens=512, timeout_s=300.0):
            self.base_url = base_url
            self.turns = turns
            self.max_tokens = max_tokens
            self.timeout_s = timeout_s

    def test_reserved_ports_refused(self):
        for port in (8080, 8090):
            errs = lcc.validate_args(self.Args("http://127.0.0.1:%d" % port))
            self.assertTrue(any("reserved" in e for e in errs), str(errs))

    def test_allowed_port_passes(self):
        self.assertEqual(lcc.validate_args(self.Args("http://127.0.0.1:8095")), [])

    def test_bad_values_rejected(self):
        self.assertTrue(lcc.validate_args(self.Args("http://h:8095", turns=0)))
        self.assertTrue(lcc.validate_args(self.Args("http://h:8095", max_tokens=0)))
        self.assertTrue(lcc.validate_args(self.Args("http://h:8095", timeout_s=0)))

    def test_bad_scheme_rejected(self):
        self.assertTrue(lcc.validate_args(self.Args("ftp://h:8095")))
        self.assertTrue(lcc.validate_args(self.Args("not-a-url")))

    def test_cli_refuses_reserved_port(self):
        for port in (8080, 8090):
            r = run_script(["--base-url", "http://127.0.0.1:%d" % port])
            self.assertEqual(r.returncode, 3, r.stderr)
            self.assertIn("reserved", r.stderr)


# ---------------------------------------------------------------------------
# CLI (ネットワークに出ないもの)
# ---------------------------------------------------------------------------

class TestCliOffline(unittest.TestCase):
    def test_help_documents_the_flags(self):
        r = run_script(["--help"])
        self.assertEqual(r.returncode, 0)
        for flag in ("--base-url", "--model", "--turns", "--max-tokens",
                     "--seed-note", "--fixture", "--out", "--dry-run",
                     "--self-test", "--json"):
            self.assertIn(flag, r.stdout, flag)

    def test_self_test_passes(self):
        r = run_script(["--self-test"])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("SELF-TEST PASSED", r.stdout)

    def test_dry_run_prints_ten_bodies_and_sends_nothing(self):
        # ポートは開いたままだが、dry-run は接続しない (受けたら seen が増える)
        with FakeMessages() as fake:
            r = run_script(["--dry-run", "--fixture", FIXTURE,
                            "--base-url", fake.base_url])
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(r.stdout.count("REQUEST:"), 10)
            self.assertIn("DRY RUN OK", r.stdout)
            self.assertEqual(fake.requests, 0, "dry-run must not touch the network")

    def test_dry_run_on_9095_default_needs_no_server(self):
        # 既定のポートに何も居なくても dry-run は成功する (接続しない)
        r = run_script(["--dry-run"])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.count("REQUEST:"), 10)

    def test_json_flag_in_dry_run_is_ignored_politely(self):
        r = run_script(["--dry-run", "--json"])
        self.assertEqual(r.returncode, 0)
        self.assertIn("DRY RUN OK", r.stdout)

    def test_missing_fixture_exits_three(self):
        r = run_script(["--dry-run", "--fixture", "/nonexistent/x.json"])
        self.assertEqual(r.returncode, 3)
        self.assertIn("ERROR", r.stderr)


# ---------------------------------------------------------------------------
# 実行経路 (偽サーバをローカルに立てる)
# ---------------------------------------------------------------------------

class TestEndToEnd(unittest.TestCase):
    def test_clean_ten_turn_run_exits_zero(self):
        with FakeMessages("clean") as fake:
            r = run_script(["--base-url", fake.base_url, "--seed-note", "s103 test"])
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("RESULT: CLEAN", r.stdout)
            self.assertEqual(fake.requests, 10)
            self.assertEqual(len(fake.seen), 10)

    def test_conversation_actually_grows(self):
        with FakeMessages("clean") as fake:
            r = run_script(["--base-url", fake.base_url, "--turns", "5"])
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual([e["messages"] for e in fake.seen], [1, 3, 5, 7, 9])

    def test_the_previous_reply_is_sent_back(self):
        with FakeMessages("clean") as fake:
            run_script(["--base-url", fake.base_url, "--turns", "3"])
            second = fake.seen[1]
            self.assertEqual(second["roles"], ["user", "assistant", "user"])
            self.assertEqual(second["contents"][1], CLEAN_REPLY)

    def test_request_goes_to_v1_messages_without_temperature(self):
        with FakeMessages("clean") as fake:
            run_script(["--base-url", fake.base_url, "--turns", "1",
                        "--model", "local/test", "--max-tokens", "128"])
            e = fake.seen[0]
            self.assertEqual(e["path"], "/v1/messages")
            self.assertEqual(e["model"], "local/test")
            self.assertEqual(e["max_tokens"], 128)
            self.assertIsNone(e["temperature"])

    def test_collapse_in_turn_four_exits_two(self):
        with FakeMessages("collapse") as fake:
            r = run_script(["--base-url", fake.base_url])
            self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
            self.assertIn("RESULT: FAILED", r.stdout)
            self.assertIn("repeat_line", r.stdout)
            # 表の行に turn 4 と出ている
            rows = [ln for ln in r.stdout.splitlines() if "repeat_line" in ln]
            self.assertTrue(any(ln.strip().startswith("4") for ln in rows),
                            "\n".join(rows))
            self.assertEqual(fake.requests, 10, "all turns run; the verdict comes after")

    def test_template_marker_exits_two(self):
        with FakeMessages("template") as fake:
            r = run_script(["--base-url", fake.base_url, "--turns", "4"])
            self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
            self.assertIn("template_marker", r.stdout)

    def test_max_tokens_is_reported_but_not_failed(self):
        with FakeMessages("max_tokens") as fake:
            r = run_script(["--base-url", fake.base_url, "--turns", "3"])
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("RESULT: CLEAN", r.stdout)
            self.assertIn("max_tokens", r.stdout)

    def test_http_500_exits_three(self):
        with FakeMessages("http500") as fake:
            r = run_script(["--base-url", fake.base_url])
            self.assertEqual(r.returncode, 3, r.stdout + r.stderr)
            self.assertIn("INFRA ERROR", r.stderr)
            self.assertIn("500", r.stderr)

    def test_connection_refused_exits_three(self):
        port = free_port()  # 誰も listen していない port
        r = run_script(["--base-url", "http://127.0.0.1:%d" % port,
                        "--timeout-s", "2"])
        self.assertEqual(r.returncode, 3, r.stdout + r.stderr)
        self.assertIn("INFRA ERROR", r.stderr)

    def test_out_writes_a_transcript_that_replays_to_the_same_verdict(self):
        td = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, td, True)
        out = os.path.join(td, "transcript.json")
        with FakeMessages("collapse") as fake:
            r1 = run_script(["--base-url", fake.base_url, "--out", out,
                             "--seed-note", "s103 replay"])
        self.assertEqual(r1.returncode, 2, r1.stderr)
        self.assertTrue(os.path.exists(out))
        with open(out, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["turn_count"], 10)
        self.assertEqual(data["seed_note"], "s103 replay")
        self.assertIn("turns", data)

        # 2 回目のパース (別プロセス・ネットワーク無し) で同じ判定になる。
        # --out 側だけは末尾に「transcript written to ...」の 1 行が増えるので、
        # 判定部分 (RESULT 行とフラグ表) を比べる。
        r2 = run_script(["--replay", out])
        self.assertEqual(r2.returncode, r1.returncode)
        self.assertEqual(r2.stdout, r1.stdout.replace(
            "transcript written to %s\n" % out, ""))

        # 実装内で 2 回判定しても同じ
        with open(out, "r", encoding="utf-8") as f:
            again = json.load(f)
        self.assertEqual(lcc.transcript_verdict(data), lcc.transcript_verdict(again))

    def test_json_output_is_parseable_and_matches_the_verdict(self):
        with FakeMessages("clean") as fake:
            r = run_script(["--base-url", fake.base_url, "--turns", "2", "--json"])
        self.assertEqual(r.returncode, 0, r.stderr)
        doc = json.loads(r.stdout)
        self.assertEqual(doc["verdict"]["exit_code"], 0)
        self.assertEqual(doc["verdict"]["turn_count"], 2)
        self.assertEqual(doc["transcript"]["turn_count"], 2)

    def test_server_is_not_started_or_stopped(self):
        # 最初のターンで 500 を返しても、スクリプトはサーバを落とさない
        with FakeMessages("http500") as fake:
            run_script(["--base-url", fake.base_url, "--turns", "1"])
            # 同じサーバがまだ生きている (2 回目の接続が失敗しない)
            self.assertEqual(fake.requests, 1)
            r = run_script(["--dry-run", "--base-url", fake.base_url])
            self.assertEqual(r.returncode, 0)


if __name__ == "__main__":
    unittest.main()